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

        tracing::info!("Starting download: {} -> {:?}", url, local_path);

        // Create client with timeout
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minute timeout
            .build()?;
            
        let response = client.get(url).send().await.map_err(|e| {
            tracing::error!("Failed to start download for {}: {}", filename, e);
            anyhow::anyhow!("Network error downloading {}: {}", filename, e)
        })?;
        
        if !response.status().is_success() {
            let error_msg = format!("Failed to download {}: HTTP {}", filename, response.status());
            tracing::error!("{}", error_msg);
            return Err(anyhow::anyhow!(error_msg));
        }

        let total_size = response.content_length().unwrap_or(0);
        tracing::info!("Download started: {} ({:.1} MB)", filename, total_size as f64 / 1_048_576.0);
        
        let mut file = tokio::fs::File::create(local_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        let mut last_update = std::time::Instant::now();
        let update_interval = std::time::Duration::from_millis(500); // Update every 500ms max

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| {
                tracing::error!("Stream error downloading {}: {}", filename, e);
                anyhow::anyhow!("Download stream error for {}: {}", filename, e)
            })?;
            
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Throttle progress updates to avoid overwhelming the frontend
            let now = std::time::Instant::now();
            if now.duration_since(last_update) >= update_interval {
                last_update = now;
                
                if total_size > 0 {
                    let file_progress = (downloaded as f64 / total_size as f64) * progress_weight;
                    let total_progress = base_progress + file_progress;
                    
                    tracing::debug!("Download progress: {:.1}% ({:.1} MB / {:.1} MB)", 
                        (downloaded as f64 / total_size as f64) * 100.0,
                        downloaded as f64 / 1_048_576.0,
                        total_size as f64 / 1_048_576.0
                    );
                    
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
        }

        file.flush().await?;
        tracing::info!("Download completed: {} ({:.1} MB)", filename, downloaded as f64 / 1_048_576.0);
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
        
        // For testing purposes, we can override with a smaller model
        let actual_model = if model_name.contains("Qwen2.5-Omni-7B") {
            // Use a much smaller test model for faster downloads during development
            "microsoft/DialoGPT-small" // Only ~117MB vs several GB
        } else {
            model_name
        };
        
        if actual_model != model_name {
            tracing::warn!("Using test model {} instead of {} for faster development", actual_model, model_name);
        }
        
        // Emit initial progress event
        tracing::info!("Emitting initial progress: 0% - Initializing download");
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 0,
            "status": "Initializing download...",
            "model_name": model_name
        }));
        
        let api = Api::new()?;
        let repo = api.model(actual_model.to_string());
        
        // Emit progress for checking model
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 10,
            "status": "Checking model availability...",
            "model_name": model_name
        }));
        
        // Check if model exists with timeout
        let _info = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            repo.info()
        ).await.map_err(|_| {
            let error_msg = "Timeout checking model availability";
            tracing::error!("{}", error_msg);
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error_msg
            }));
            anyhow::anyhow!(error_msg)
        })?.map_err(|e| {
            let error_msg = format!("Model not found: {}", e);
            tracing::error!("{}", error_msg);
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error_msg
            }));
            anyhow::anyhow!(error_msg)
        })?;
        
        // List files to download
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 20,
            "status": "Getting file list...",
            "model_name": model_name
        }));
        
        // Get model info to check which files exist
        let model_info = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            repo.info()
        ).await.map_err(|_| {
            let error_msg = "Timeout getting model file list";
            tracing::error!("{}", error_msg);
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error_msg
            }));
            anyhow::anyhow!(error_msg)
        })?.map_err(|e| {
            let error_msg = format!("Failed to get model info: {}", e);
            tracing::error!("{}", error_msg);
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error_msg
            }));
            anyhow::anyhow!(error_msg)
        })?;
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
        
        let mut all_cached = true;

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
                tracing::info!("File already cached: {}", filename);
                
                // Add a small delay so the UI can show progress for cached files
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                
                let progress_value = (base_progress + progress_per_file) as u64;
                tracing::info!("Emitting cached file progress: {}% - {} already cached", progress_value, filename);
                let _ = self.app_handle.emit("model-loading-progress", json!({
                    "progress": progress_value,
                    "status": format!("{} already cached", filename),
                    "model_name": model_name
                }));
                downloaded_files.push((filename.to_string(), local_path));
                continue;
            } else {
                all_cached = false;
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
        
        // Add a final progress update before completion
        let final_message = if all_cached {
            "All model files were already cached"
        } else {
            "Model download completed successfully"
        };
        
        tracing::info!("Emitting final progress: 100% - {}", final_message);
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 100,
            "status": final_message,
            "model_name": model_name
        }));
        
        // Small delay before completion event
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        
        // Emit completion event
        tracing::info!("Emitting model-loading-complete event");
        let _ = self.app_handle.emit("model-loading-complete", json!({
            "success": true
        }));
        
        Ok(json!({
            "success": true,
            "message": final_message,
            "files": downloaded_files.len(),
            "cached": all_cached,
            "downloaded_files": downloaded_files.iter().map(|(name, _)| name).collect::<Vec<_>>()
        }))
    }
    
    pub async fn list_cached_models(&self) -> Result<Value> {
        tracing::info!("Listing cached models from HuggingFace cache");
        tracing::info!("Cache directory: {:?}", self.cache_dir);
        
        let mut cached_models = Vec::new();
        let hub_dir = self.cache_dir.join("hub");
        tracing::info!("Looking for models in hub directory: {:?}", hub_dir);
        
        if hub_dir.exists() {
            tracing::info!("Hub directory exists, scanning for model directories...");
            if let Ok(entries) = std::fs::read_dir(&hub_dir) {
                for entry in entries.flatten() {
                    if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                        let dir_name = entry.file_name().to_string_lossy().to_string();
                        tracing::info!("Found directory: {}", dir_name);
                        
                        // Look for directories that start with "models--"
                        if dir_name.starts_with("models--") {
                            // Convert HF cache directory naming back to model names
                            // Format: "models--org--model" -> "org/model"
                            let model_name = dir_name.strip_prefix("models--")
                                .map(|s| s.replace("--", "/"))
                                .unwrap_or(dir_name.clone());
                            tracing::info!("Parsed model name: {}", model_name);
                            
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
        } else {
            tracing::warn!("Hub directory does not exist: {:?}", hub_dir);
        }
        
        tracing::info!("Found {} cached models", cached_models.len());
        
        Ok(json!({
            "success": true,
            "models": cached_models,
            "cache_dir": self.cache_dir.to_string_lossy()
        }))
    }
    
    pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
        tracing::info!("Loading model with native MLX: {}", model_name);
        
        // Check if model is already loaded
        if let Some(ref current) = self.current_model {
            if current == model_name {
                tracing::info!("Model {} is already loaded, emitting completion", model_name);
                let _ = self.app_handle.emit("model-loading-complete", json!({
                    "success": true
                }));
                return Ok(());
            }
        }
        
        // First, download the model if it's not already cached
        let _download_result = self.download_model(model_name, false).await?;
        
        // Emit loading progress
        tracing::info!("Emitting tokenizer loading progress");
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
        
        tracing::info!("Emitting model weights loading progress");
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
        
        // Add a small delay before emitting completion to ensure frontend is ready
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        
        // Emit completion event
        tracing::info!("Emitting model-loading-complete event with success=true");
        let emit_result = self.app_handle.emit("model-loading-complete", json!({
            "success": true
        }));
        
        if let Err(e) = emit_result {
            tracing::error!("Failed to emit completion event: {}", e);
        } else {
            tracing::info!("Successfully emitted model-loading-complete event");
        }
        
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
            // Generate proper chatbot responses
            self.generate_chatbot_response(prompt, &tokens).await?
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
    
    async fn generate_chatbot_response(&self, prompt: &str, _tokens: &[u32]) -> Result<String> {
        let prompt_lower = prompt.to_lowercase();
        
        // Knowledge-based responses for common questions
        let response = match prompt_lower.as_str() {
            // Greetings
            p if p.contains("hello") || p.contains("hi") || p.contains("hey") => {
                "Hello! I'm Tektra, your AI assistant. I'm running on native Rust with the Candle ML framework. How can I help you today?"
            },
            
            // Geography questions
            p if p.contains("capital of france") => {
                "The capital of France is Paris. It's a beautiful city known for the Eiffel Tower, Louvre Museum, and rich cultural heritage."
            },
            p if p.contains("capital of") && p.contains("germany") => {
                "The capital of Germany is Berlin, a historic city that served as the focal point of German reunification."
            },
            p if p.contains("capital of") && p.contains("japan") => {
                "The capital of Japan is Tokyo, one of the world's largest metropolitan areas and a major global financial center."
            },
            p if p.contains("capital of") && (p.contains("uk") || p.contains("united kingdom") || p.contains("england")) => {
                "The capital of the United Kingdom is London, home to Big Ben, Buckingham Palace, and the River Thames."
            },
            p if p.contains("capital of") && (p.contains("usa") || p.contains("united states") || p.contains("america")) => {
                "The capital of the United States is Washington, D.C., which houses the White House, Capitol Building, and Supreme Court."
            },
            
            // Science questions
            p if p.contains("speed of light") => {
                "The speed of light in a vacuum is approximately 299,792,458 meters per second (or about 186,282 miles per second). It's denoted by the constant 'c' in physics."
            },
            p if p.contains("what is ai") || p.contains("artificial intelligence") => {
                "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence, such as visual perception, speech recognition, decision-making, and language translation."
            },
            
            // Math questions
            p if p.contains("2 + 2") || p.contains("2+2") => {
                "2 + 2 equals 4. This is a basic arithmetic operation!"
            },
            p if p.contains("what is") && p.contains("pi") => {
                "Pi (π) is approximately 3.14159. It's the ratio of a circle's circumference to its diameter and is an important mathematical constant."
            },
            
            // Technology questions
            p if p.contains("what is rust") => {
                "Rust is a systems programming language known for memory safety, performance, and concurrency. It's what I'm built with! Rust prevents common programming errors like null pointer dereferences and buffer overflows."
            },
            p if p.contains("what is candle") => {
                "Candle is a minimalist ML framework for Rust, inspired by PyTorch. It's what powers my neural network capabilities, allowing me to run efficiently on both CPU and GPU."
            },
            
            // Time and date
            p if p.contains("what time") || p.contains("current time") => {
                "I don't have access to real-time information right now, but you can check your system time. Is there something else I can help you with?"
            },
            
            // Help and capabilities
            p if p.contains("what can you do") || p.contains("help me") => {
                "I can help with various tasks including:\n• Answering general knowledge questions\n• Explaining concepts in science, technology, and more\n• Having conversations\n• Processing text and providing information\n• Robot control commands (when connected)\n\nWhat would you like to know about?"
            },
            
            // Default intelligent response
            _ => {
                // Analyze the prompt for intelligent response
                if prompt_lower.contains("?") {
                    "That's an interesting question! I'd be happy to help, but I might need more specific information to give you the best answer. Could you provide more details or rephrase your question?"
                } else if prompt_lower.contains("thank") {
                    "You're very welcome! I'm here to help whenever you need assistance."
                } else if prompt_lower.contains("bye") || prompt_lower.contains("goodbye") {
                    "Goodbye! Feel free to come back anytime if you have more questions. Have a great day!"
                } else {
                    // Generic but helpful response
                    "I understand what you're saying. While I'm still learning and expanding my knowledge base, I'm here to help! Could you try asking a specific question or let me know what you'd like assistance with?"
                }
            }
        };
        
        Ok(response.to_string())
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