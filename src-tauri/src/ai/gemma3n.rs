use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Manager};
use tracing::{error, info, warn};
use super::inference_backend::{InferenceConfig, BackendType};
use super::inference_manager::{InferenceManager};

// Gemma-3n model information
// E2B = 2 billion parameters (faster, good for desktop)
// E4B = 4 billion parameters (more capable, needs more resources)
const GEMMA_E2B_ID: &str = "google/gemma-3n-E2B-it";
const GEMMA_E2B_GGUF_ID: &str = "unsloth/gemma-3n-E2B-it-GGUF";
const GEMMA_E2B_FILE_Q4: &str = "gemma-3n-E2B-it-Q4_K_M.gguf"; // 2.79GB
const GEMMA_E2B_FILE_Q8: &str = "gemma-3n-E2B-it-Q8_0.gguf"; // 4.79GB

// For more capable version (if user has resources)
const GEMMA_E4B_GGUF_ID: &str = "unsloth/gemma-3n-E4B-it-GGUF";
const GEMMA_E4B_FILE_Q4: &str = "gemma-3n-E4B-it-Q4_K_M.gguf"; // ~3GB estimated

// MLX format models from mlx-community (for Apple Silicon)
const GEMMA2_2B_MLX_ID: &str = "mlx-community/gemma-2-2b-it-4bit";
const GEMMA2_2B_MLX_FP16_ID: &str = "mlx-community/gemma-2-2b-it-fp16";
const GEMMA2_9B_MLX_ID: &str = "mlx-community/gemma-2-9b-it-4bit";
const GEMMA2_9B_MLX_FP16_ID: &str = "mlx-community/gemma-2-9b-it-fp16";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadProgress {
    pub progress: u32,
    pub status: String,
    pub model_name: String,
}

#[derive(Debug, Clone)]
struct GemmaChatTemplate {
    // Gemma uses a specific chat template
    bos: String,
    start_turn: String,
    end_turn: String,
    user_role: String,
    model_role: String,
}

impl Default for GemmaChatTemplate {
    fn default() -> Self {
        Self {
            // Gemma-3n uses specific tokens as per Google's format
            bos: "<bos>".to_string(),
            start_turn: "<start_of_turn>".to_string(),
            end_turn: "<end_of_turn>".to_string(),
            user_role: "user".to_string(),
            model_role: "model".to_string(), // Gemma uses "model" not "assistant"
        }
    }
}

pub struct AIManager {
    app_handle: AppHandle,
    model_loaded: bool,
    model_path: Option<PathBuf>,
    chat_template: GemmaChatTemplate,
    selected_model: String,
    inference_manager: InferenceManager,
    backend_type: BackendType,
}

impl AIManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        // Default to Auto backend selection
        let backend_type = BackendType::Auto;
        let inference_manager = InferenceManager::new(backend_type)?;
        
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: GemmaChatTemplate::default(),
            selected_model: GEMMA_E2B_GGUF_ID.to_string(), // Default to 2B model
            inference_manager,
            backend_type,
        })
    }
    
    pub fn with_backend(app_handle: AppHandle, backend_type: BackendType) -> Result<Self> {
        let inference_manager = InferenceManager::new(backend_type)?;
        
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: GemmaChatTemplate::default(),
            selected_model: GEMMA_E2B_GGUF_ID.to_string(),
            inference_manager,
            backend_type,
        })
    }

    pub async fn load_model(&mut self) -> Result<()> {
        self.emit_progress(0, "Starting Gemma-3n initialization...", &self.selected_model).await;
        
        // Download model if needed
        match self.download_model().await {
            Ok(path) => {
                self.model_path = Some(path.clone());
                info!("Gemma-3n model downloaded to: {:?}", path);
                
                self.emit_progress(90, "Loading Gemma-3n model into memory...", &self.selected_model).await;
                
                // Load the model into the inference engine
                match self.inference_manager.load_model(&path).await {
                    Ok(_) => {
                        self.model_loaded = true;
                        let backend_name = self.inference_manager.backend_name().await;
                        self.emit_progress(100, &format!("Gemma-3n ready with {} backend! Google's latest AI model at your service.", backend_name), &self.selected_model).await;
                        info!("Model loaded successfully using {} backend", backend_name);
                        Ok(())
                    }
                    Err(e) => {
                        error!("Failed to load model: {}", e);
                        self.emit_progress(0, &format!("Failed to load model: {}", e), &self.selected_model).await;
                        Err(e)
                    }
                }
            }
            Err(e) => {
                error!("Failed to download Gemma model: {}", e);
                Err(e)
            }
        }
    }

    async fn download_model(&self) -> Result<PathBuf> {
        // Check if we should use MLX (Apple Silicon only)
        let use_mlx = match self.backend_type {
            BackendType::Auto => {
                cfg!(target_os = "macos") && std::env::consts::ARCH == "aarch64"
            }
            BackendType::MLX => true,
            BackendType::GGUF => false,
        };
        
        if use_mlx {
            self.download_mlx_model().await
        } else {
            self.download_gguf_model().await
        }
    }
    
    async fn download_mlx_model(&self) -> Result<PathBuf> {
        self.emit_progress(10, "Preparing to download MLX model...", &self.selected_model).await;
        
        // Use Gemma-2 2B 4-bit model for efficiency
        let model_id = GEMMA2_2B_MLX_ID;
        
        // Get cache directory for MLX models
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?
            .join("huggingface")
            .join("hub")
            .join(model_id.replace('/', "--"));
        
        // Check if model already exists
        if cache_dir.exists() {
            let config_path = cache_dir.join("config.json");
            let weights_path = cache_dir.join("model.safetensors");
            
            if config_path.exists() && weights_path.exists() {
                self.emit_progress(100, "Found cached MLX model", model_id).await;
                return Ok(cache_dir);
            }
        }
        
        // Download the MLX model
        self.emit_progress(20, "Downloading MLX model from mlx-community...", model_id).await;
        
        // Download files directly using URLs
        let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);
        
        // Download all necessary files
        let files = vec![
            ("config.json", true),
            ("model.safetensors", true),
            ("tokenizer.json", true),
            ("tokenizer_config.json", false),
            ("special_tokens_map.json", false),
        ];
        
        std::fs::create_dir_all(&cache_dir)?;
        
        let client = reqwest::Client::new();
        
        for (idx, (file, required)) in files.iter().enumerate() {
            let progress = 20 + (idx as u32 * 15);
            self.emit_progress(progress, &format!("Downloading {}", file), model_id).await;
            
            let file_path = cache_dir.join(file);
            if !file_path.exists() {
                let url = format!("{}/{}", base_url, file);
                match client.get(&url).send().await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let bytes = response.bytes().await?;
                            std::fs::write(&file_path, bytes)?;
                            info!("Downloaded {} successfully", file);
                        } else if *required {
                            return Err(anyhow::anyhow!("Failed to download {}: HTTP {}", file, response.status()));
                        } else {
                            warn!("Optional file {} not found", file);
                        }
                    }
                    Err(e) => {
                        if *required {
                            return Err(anyhow::anyhow!("Failed to download {}: {}", file, e));
                        } else {
                            warn!("Optional file {} not found: {}", file, e);
                        }
                    }
                }
            }
        }
        
        self.emit_progress(100, "MLX model downloaded successfully", model_id).await;
        Ok(cache_dir)
    }
    
    async fn download_gguf_model(&self) -> Result<PathBuf> {
        self.emit_progress(10, "Checking for GGUF model...", &self.selected_model).await;
        
        // Get cache directory
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?
            .join("huggingface")
            .join("hub")
            .join(self.selected_model.replace('/', "--"));
        
        std::fs::create_dir_all(&cache_dir)?;
        
        // Choose file based on model selection
        let model_file = if self.selected_model.contains("E4B") {
            GEMMA_E4B_FILE_Q4
        } else {
            GEMMA_E2B_FILE_Q4 // Use Q4 quantization for balance of size/quality
        };
        
        let model_path = cache_dir.join(model_file);
        
        if model_path.exists() {
            let metadata = std::fs::metadata(&model_path)?;
            if metadata.len() > 100_000_000 { // At least 100MB
                self.emit_progress(70, "Found cached GGUF model", &self.selected_model).await;
                return Ok(model_path);
            } else {
                // Remove corrupted file
                let _ = std::fs::remove_file(&model_path);
            }
        }
        
        // Download the model
        let model_size = if self.selected_model.contains("E4B") {
            "~3-5GB"
        } else {
            "2.79GB"
        };
        
        self.emit_progress(
            20, 
            &format!("Downloading Gemma-3n model ({})...", model_size), 
            &self.selected_model
        ).await;
        
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.selected_model, model_file
        );
        
        info!("Downloading from: {}", url);
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1200)) // 20 minutes for larger models
            .user_agent("Tektra-AI-Assistant/0.1.0")
            .build()?;
        
        let response = client.get(&url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model: HTTP {} - Check if model file exists",
                response.status()
            ));
        }
        
        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        
        // Create temporary file
        let temp_path = model_path.with_extension("tmp");
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut stream = response.bytes_stream();
        
        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            
            if total_size > 0 {
                let progress = 20 + ((downloaded as f64 / total_size as f64) * 50.0) as u32;
                self.emit_progress(
                    progress,
                    &format!(
                        "Downloading Gemma-3n ({} / {})",
                        bytesize::ByteSize(downloaded),
                        bytesize::ByteSize(total_size)
                    ),
                    &self.selected_model,
                ).await;
            }
        }
        
        file.flush().await?;
        drop(file);
        
        // Move to final location
        tokio::fs::rename(&temp_path, &model_path).await?;
        
        Ok(model_path)
    }

    pub async fn generate_response(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_response_with_system_prompt(prompt, max_tokens, None).await
    }
    
    pub async fn generate_response_with_image(&self, prompt: &str, image_data: &[u8], max_tokens: usize) -> Result<String> {
        self.generate_response_with_image_and_system_prompt(prompt, image_data, max_tokens, None).await
    }
    
    pub async fn generate_response_with_system_prompt(&self, prompt: &str, _max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant powered by the Gemma-3n model. You provide accurate, thoughtful, and detailed responses.".to_string()
        );
        
        // Format prompt using correct Gemma chat template
        let _formatted_prompt = if !system.is_empty() {
            format!(
                "{}{}{}\n{}\n{}{}\n{}{}",
                self.chat_template.bos,
                self.chat_template.start_turn,
                self.chat_template.user_role,
                format!("{}\n\n{}", system, prompt),
                self.chat_template.end_turn,
                self.chat_template.start_turn,
                self.chat_template.model_role,
                "\n" // Model's response will go here
            )
        } else {
            format!(
                "{}{}{}\n{}{}\n{}{}",
                self.chat_template.bos,
                self.chat_template.start_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_turn,
                self.chat_template.start_turn,
                self.chat_template.model_role
            )
        };
        
        info!("Processing prompt: {}", prompt);
        info!("Formatted for Gemma: {}", _formatted_prompt);
        
        // Create inference config
        let config = InferenceConfig {
            max_tokens: _max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        };
        
        // Use actual inference backend
        match self.inference_manager.generate(&_formatted_prompt, &config).await {
            Ok(response) => {
                info!("Generated response: {}", response);
                Ok(response)
            }
            Err(e) => {
                error!("Inference error: {}", e);
                // Return the error instead of falling back to demo responses
                Err(anyhow::anyhow!("Model inference failed: {}", e))
            }
        }
    }
    
    pub async fn generate_response_with_image_and_system_prompt(&self, prompt: &str, _image_data: &[u8], _max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant powered by the Gemma-3n model. You can see and analyze images to answer questions about them.".to_string()
        );
        
        info!("Processing prompt with image: {}", prompt);
        
        // Multimodal inference is not yet implemented
        error!("Multimodal inference with images is not yet implemented");
        Err(anyhow::anyhow!("Multimodal inference with images is not yet implemented. The model can only process text prompts at this time."))
    }
    
    

    async fn emit_progress(&self, progress: u32, status: &str, model_name: &str) {
        let progress_data = ModelLoadProgress {
            progress,
            status: status.to_string(),
            model_name: model_name.to_string(),
        };
        
        if let Err(e) = self.app_handle.emit_all("model-loading-progress", &progress_data) {
            error!("Failed to emit progress: {}", e);
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }
    
    pub async fn get_backend_info(&self) -> String {
        let backend_name = self.inference_manager.backend_name().await;
        let system_info = InferenceManager::get_system_info();
        
        format!(
            "Current Backend: {}\nBackend Type: {:?}\n\n{}",
            backend_name, self.backend_type, system_info
        )
    }
    
    pub async fn benchmark_backends(&self, prompt: &str, max_tokens: usize) -> Result<Vec<(String, super::inference_backend::InferenceMetrics)>> {
        let config = InferenceConfig {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: Some(42), // Fixed seed for consistent benchmarks
            n_threads: None,
        };
        
        self.inference_manager.benchmark_backends(prompt, &config).await
    }
}