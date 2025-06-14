use anyhow::Result;
use std::path::PathBuf;
use tauri::{AppHandle, Emitter};
use serde_json::{json, Value};
use hf_hub::api::tokio::Api;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::io::AsyncWriteExt;

use crate::state::Message;

// Candle ML framework 
use candle_core::Device;

// Tokenizer
use tokenizers::Tokenizer;

pub struct ModelManager {
    app_handle: AppHandle,
    cache_dir: PathBuf,
    current_model: Option<String>,
    tokenizer: Option<Arc<Mutex<Tokenizer>>>,
    device: Device,
}

impl ModelManager {
    async fn download_file_with_progress(
        &self,
        url: &str,
        local_path: &PathBuf,
        filename: &str,
        model_name: &str,
        base_progress: f64,
        progress_weight: f64,
    ) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = local_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to download {}: HTTP {}", filename, response.status()));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut file = tokio::fs::File::create(local_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Emit progress update
            if total_size > 0 {
                let file_progress = (downloaded as f64 / total_size as f64) * progress_weight;
                let total_progress = base_progress + file_progress;
                
                let _ = self.app_handle.emit("model-loading-progress", json!({
                    "progress": total_progress.min(100.0) as u64,
                    "status": format!("Downloading {} ({:.1} MB / {:.1} MB)", 
                        filename,
                        downloaded as f64 / 1_048_576.0,
                        total_size as f64 / 1_048_576.0
                    ),
                    "model_name": model_name
                }));
            } else {
                // If we don't know the total size, show indeterminate progress
                let _ = self.app_handle.emit("model-loading-progress", json!({
                    "progress": (base_progress + progress_weight * 0.5) as u64,
                    "status": format!("Downloading {} ({:.1} MB downloaded)", 
                        filename,
                        downloaded as f64 / 1_048_576.0
                    ),
                    "model_name": model_name
                }));
            }
        }

        file.flush().await?;
        Ok(())
    }
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        // Use HuggingFace Hub cache directory for model storage
        let cache_dir = Self::get_hf_cache_dir()?;
        std::fs::create_dir_all(&cache_dir)?;
        
        // Initialize Candle device (prefer Metal on Apple Silicon, fallback to CPU)
        let device = Device::new_metal(0).unwrap_or(Device::Cpu);
        
        Ok(Self {
            app_handle,
            cache_dir,
            current_model: None,
            tokenizer: None,
            device,
        })
    }
    
    fn get_hf_cache_dir() -> Result<PathBuf> {
        // Use the same cache directory that HuggingFace Hub uses
        if let Ok(cache_dir) = std::env::var("HF_HOME") {
            return Ok(PathBuf::from(cache_dir));
        }
        
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(xdg_cache).join("huggingface"));
        }
        
        // Default HF cache location
        if let Some(home_dir) = dirs::home_dir() {
            return Ok(home_dir.join(".cache").join("huggingface"));
        }
        
        Err(anyhow::anyhow!("Could not determine HuggingFace cache directory"))
    }
    
    pub async fn download_model(&self, model_name: &str, _force: bool) -> Result<Value> {
        tracing::info!("Downloading model with native Rust: {}", model_name);
        
        // Emit initial progress event
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 0,
            "status": "Initializing download...",
            "model_name": model_name
        }));
        
        let api = Api::new()?;
        let repo = api.model(model_name.to_string());
        
        // Emit progress for checking model
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 10,
            "status": "Checking model availability...",
            "model_name": model_name
        }));
        
        // Check if model exists
        let _info = repo.info().await.map_err(|e| {
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": format!("Model not found: {}", e)
            }));
            anyhow::anyhow!("Model not found: {}", e)
        })?;
        
        // List files to download
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 20,
            "status": "Getting file list...",
            "model_name": model_name
        }));
        
        // Get model info to check which files exist
        let model_info = repo.info().await?;
        let available_files: Vec<&str> = model_info
            .siblings
            .iter()
            .filter_map(|sibling| {
                let filename = &sibling.rfilename;
                // Only download essential files we need
                if filename.ends_with(".json") || 
                   filename.ends_with(".safetensors") ||
                   filename == "tokenizer.json" ||
                   filename == "config.json" {
                    Some(filename.as_str())
                } else {
                    None
                }
            })
            .collect();

        if available_files.is_empty() {
            return Err(anyhow::anyhow!("No compatible model files found"));
        }

        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 25,
            "status": format!("Found {} files to download", available_files.len()),
            "model_name": model_name
        }));

        let mut downloaded_files = Vec::new();
        let total_files = available_files.len();
        let progress_per_file = 70.0 / total_files as f64; // 25% to 95%

        for (i, filename) in available_files.iter().enumerate() {
            let base_progress = 25.0 + (i as f64 * progress_per_file);
            
            // Get download URL for the file
            let download_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                model_name, filename
            );

            // Create local path in HF cache format
            let model_cache_name = model_name.replace("/", "--");
            let local_path = self.cache_dir
                .join("hub")
                .join(format!("models--{}", model_cache_name))
                .join("snapshots")
                .join("main") // Using main branch
                .join(filename);

            // Check if file already exists
            if local_path.exists() && !_force {
                let _ = self.app_handle.emit("model-loading-progress", json!({
                    "progress": (base_progress + progress_per_file) as u64,
                    "status": format!("{} already cached", filename),
                    "model_name": model_name
                }));
                downloaded_files.push((filename.to_string(), local_path));
                continue;
            }

            // Download with progress tracking
            match self.download_file_with_progress(
                &download_url,
                &local_path,
                filename,
                model_name,
                base_progress,
                progress_per_file,
            ).await {
                Ok(()) => {
                    tracing::info!("Downloaded: {} -> {:?}", filename, local_path);
                    downloaded_files.push((filename.to_string(), local_path));
                }
                Err(e) => {
                    tracing::warn!("Could not download {}: {}", filename, e);
                    // Some files might be optional, continue with others
                }
            }
        }
        
        if downloaded_files.is_empty() {
            let error_msg = "No model files could be downloaded";
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error_msg
            }));
            return Err(anyhow::anyhow!(error_msg));
        }
        
        // Emit completion event
        let _ = self.app_handle.emit("model-loading-complete", json!({
            "success": true
        }));
        
        Ok(json!({
            "success": true,
            "message": "Model downloaded successfully",
            "files": downloaded_files.len(),
            "downloaded_files": downloaded_files.iter().map(|(name, _)| name).collect::<Vec<_>>()
        }))
    }
    
    pub async fn list_cached_models(&self) -> Result<Value> {
        tracing::info!("Listing cached models from HuggingFace cache");
        
        let mut cached_models = Vec::new();
        let models_dir = self.cache_dir.join("models--");
        
        if models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry in entries.flatten() {
                    if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        let dir_name = entry.file_name().to_string_lossy().to_string();
                        
                        // Convert HF cache directory naming back to model names
                        if let Some(model_name) = dir_name.strip_prefix("models--") {
                            let model_name = model_name.replace("--", "/");
                            
                            // Calculate directory size
                            let size = calculate_dir_size(&entry.path()).unwrap_or(0);
                            
                            cached_models.push(json!({
                                "name": model_name,
                                "path": entry.path().to_string_lossy(),
                                "size_bytes": size,
                                "size_gb": (size as f64) / (1024_f64.powi(3))
                            }));
                        }
                    }
                }
            }
        }
        
        Ok(json!({
            "success": true,
            "models": cached_models,
            "cache_dir": self.cache_dir.to_string_lossy()
        }))
    }
    
    pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
        tracing::info!("Loading model with native MLX: {}", model_name);
        
        // First, download the model if it's not already cached
        let _download_result = self.download_model(model_name, false).await?;
        
        // Emit loading progress
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 85,
            "status": "Loading tokenizer...",
            "model_name": model_name
        }));
        
        // Try to load tokenizer from downloaded files
        let model_cache_name = model_name.replace("/", "--");
        let tokenizer_path = self.cache_dir
            .join("hub")
            .join(format!("models--{}", model_cache_name))
            .join("snapshots")
            .join("main")
            .join("tokenizer.json");
        
        let tokenizer = if tokenizer_path.exists() {
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tok) => {
                    tracing::info!("Loaded tokenizer from: {:?}", tokenizer_path);
                    Some(Arc::new(Mutex::new(tok)))
                }
                Err(e) => {
                    tracing::warn!("Failed to load tokenizer: {}", e);
                    None
                }
            }
        } else {
            tracing::warn!("Tokenizer file not found at: {:?}", tokenizer_path);
            None
        };
        
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 95,
            "status": "Loading model weights...",
            "model_name": model_name
        }));
        
        // For now, we'll create a placeholder for model weights
        // In a full implementation, you would load safetensors files here
        let _model_weights: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        
        // Set the loaded components
        self.current_model = Some(model_name.to_string());
        self.tokenizer = tokenizer;
        
        tracing::info!("Model loaded successfully: {}", model_name);
        
        // Emit completion event
        let _ = self.app_handle.emit("model-loading-complete", json!({
            "success": true
        }));
        
        Ok(())
    }
    
    pub async fn generate_response(&self, prompt: &str, history: &[Message]) -> Result<String> {
        // Check if model is loaded
        if self.current_model.is_none() {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        // Check if tokenizer is available
        let tokenizer = match &self.tokenizer {
            Some(tok) => tok,
            None => {
                return Err(anyhow::anyhow!("Tokenizer not loaded"));
            }
        };
        
        // Format conversation with history
        let formatted_prompt = self.format_conversation(prompt, history);
        
        tracing::info!("Generating response with native MLX");
        
        // Tokenize the input
        let tokens = {
            let tok = tokenizer.lock().await;
            let encoding = tok.encode(formatted_prompt.clone(), false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            encoding.get_ids().to_vec()
        };
        
        // Check if the prompt contains action-related keywords
        let is_action_request = self.is_action_request(&formatted_prompt);
        
        let response = if is_action_request {
            // Generate FAST action tokens (placeholder implementation)
            let action_tokens = self.generate_action_tokens(&formatted_prompt).await?;
            format!("<ACTION>{}</ACTION>", action_tokens)
        } else {
            // Generate text response
            format!(
                "I understand your message: '{}'. I'm running with native Rust and Candle ML framework! \
                I can process your request and generate appropriate responses. (Tokenized {} tokens for processing.)",
                prompt,
                tokens.len()
            )
        };
        
        Ok(response)
    }
    
    pub async fn get_status(&self) -> serde_json::Value {
        // Check Candle device type
        let device_info = match &self.device {
            Device::Metal(_) => "Apple Silicon GPU (Candle/Metal)",
            Device::Cpu => "CPU (Candle)",
            Device::Cuda(_) => "CUDA GPU (Candle)",
        };
        
        json!({
            "loaded": self.current_model.is_some(),
            "model": self.current_model.clone().unwrap_or_else(|| "None".to_string()),
            "device": device_info,
            "tokenizer_loaded": self.tokenizer.is_some(),
            "framework": "Native Rust (Candle)",
            "cache_dir": self.cache_dir.to_string_lossy()
        })
    }
    
    fn is_action_request(&self, prompt: &str) -> bool {
        let action_keywords = [
            "pick", "grab", "move", "turn", "rotate", "push", "pull", 
            "grasp", "release", "drop", "place", "put", "go", "stop",
            "forward", "backward", "left", "right", "up", "down"
        ];
        
        let prompt_lower = prompt.to_lowercase();
        action_keywords.iter().any(|keyword| prompt_lower.contains(keyword))
    }
    
    async fn generate_action_tokens(&self, prompt: &str) -> Result<String> {
        // Placeholder FAST token generation based on prompt analysis
        // In a full implementation, this would use the trained model
        
        let prompt_lower = prompt.to_lowercase();
        let mut tokens = Vec::new();
        
        // Simple keyword-to-token mapping (based on specification examples)
        if prompt_lower.contains("pick") || prompt_lower.contains("grab") {
            tokens.extend_from_slice(&[67, 18]); // grasp + release sequence
        }
        if prompt_lower.contains("move") || prompt_lower.contains("forward") {
            tokens.push(32); // move_forward
        }
        if prompt_lower.contains("turn") || prompt_lower.contains("left") {
            tokens.push(44); // turn_left
        }
        if prompt_lower.contains("stop") || prompt_lower.contains("release") {
            tokens.push(18); // release/stop
        }
        
        // Default action if no specific keywords detected
        if tokens.is_empty() {
            tokens.push(32); // Default to move_forward
        }
        
        // Format as space-separated token string
        Ok(tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))
    }
    
    fn format_conversation(&self, current_prompt: &str, history: &[Message]) -> String {
        let mut formatted = String::new();
        
        // Add recent conversation history (last 10 messages)
        for msg in history.iter().rev().take(10).rev() {
            formatted.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        
        // Add current prompt
        formatted.push_str(&format!("user: {}\nassistant: ", current_prompt));
        
        formatted
    }
}

fn calculate_dir_size(dir: &std::path::Path) -> Result<u64> {
    let mut size = 0;
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                size += calculate_dir_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    }
    Ok(size)
}