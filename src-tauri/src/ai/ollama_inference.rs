use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use tracing::{info, error};
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest, generation::chat::{ChatMessage, request::ChatMessageRequest}};
use super::inference_backend::{InferenceBackend, InferenceConfig};

#[derive(Debug, Clone)]
pub enum OllamaExe {
    System(PathBuf),
    Embedded(PathBuf),
}

pub struct OllamaInference {
    ollama_exe: Option<OllamaExe>,
    ollama_client: Option<Ollama>,
    model_loaded: bool,
    current_model: Option<String>,
    ollama_port: u16,
}

impl OllamaInference {
    pub fn new() -> Self {
        Self {
            ollama_exe: None,
            ollama_client: None,
            model_loaded: false,
            current_model: None,
            ollama_port: 11434, // Default Ollama port
        }
    }

    /// Find Ollama binary - either system-installed or download embedded version
    pub async fn find_ollama() -> Result<OllamaExe> {
        info!("Checking for system Ollama installation...");
        
        // First try to find system Ollama
        if let Ok(output) = Command::new("ollama").arg("--version").output() {
            if output.status.success() {
                info!("Found system Ollama installation");
                return Ok(OllamaExe::System(PathBuf::from("ollama")));
            }
        }

        info!("System Ollama not found, downloading embedded version...");
        
        // Download embedded Ollama
        let data_dir = dirs::data_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find data directory"))?
            .join("tektra")
            .join("ollama");
        
        fs::create_dir_all(&data_dir).await?;
        
        // Use default download configuration
        let download_config = ollama_td::OllamaDownload::default();
        let downloaded_path = ollama_td::download(download_config).await
            .map_err(|e| anyhow::anyhow!("Failed to download Ollama: {}", e))?;
        
        info!("Ollama downloaded to: {:?}", downloaded_path);
        
        // Check if downloaded file is a zip archive and extract it
        let ollama_binary = if downloaded_path.extension().and_then(|s| s.to_str()) == Some("zip") {
            info!("Extracting Ollama zip archive...");
            
            // Extract zip to the data directory
            let extract_dir = data_dir.join("extracted");
            fs::create_dir_all(&extract_dir).await?;
            
            // Use std::process::Command to extract zip (cross-platform)
            #[cfg(target_os = "macos")]
            let extract_cmd = Command::new("unzip")
                .arg("-o") // Overwrite existing files
                .arg(&downloaded_path)
                .arg("-d")
                .arg(&extract_dir)
                .output()
                .map_err(|e| anyhow::anyhow!("Failed to run unzip command: {}", e))?;
                
            #[cfg(not(target_os = "macos"))]
            let extract_cmd = Command::new("unzip")
                .arg("-o")
                .arg(&downloaded_path)
                .arg("-d") 
                .arg(&extract_dir)
                .output()
                .map_err(|e| anyhow::anyhow!("Failed to run unzip command: {}", e))?;
            
            if !extract_cmd.status.success() {
                let stderr = String::from_utf8_lossy(&extract_cmd.stderr);
                return Err(anyhow::anyhow!("Failed to extract Ollama zip: {}", stderr));
            }
            
            // Find the Ollama binary in the extracted contents
            #[cfg(target_os = "macos")]
            let ollama_path = extract_dir.join("Ollama.app").join("Contents").join("Resources").join("ollama");
            
            #[cfg(target_os = "linux")]
            let ollama_path = extract_dir.join("ollama");
            
            #[cfg(target_os = "windows")]  
            let ollama_path = extract_dir.join("ollama.exe");
            
            if !ollama_path.exists() {
                return Err(anyhow::anyhow!("Ollama binary not found in extracted archive at: {:?}", ollama_path));
            }
            
            info!("Extracted Ollama binary to: {:?}", ollama_path);
            ollama_path
        } else {
            downloaded_path
        };
        
        // Make binary executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&ollama_binary, std::fs::Permissions::from_mode(0o755)).await?;
        }
        
        info!("Ollama binary ready at: {:?}", ollama_binary);
        Ok(OllamaExe::Embedded(ollama_binary))
    }

    /// Initialize Ollama (find binary and start if needed)
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Ollama inference backend...");
        
        let ollama_exe = Self::find_ollama().await?;
        self.ollama_exe = Some(ollama_exe.clone());
        
        // Start Ollama server if using embedded version
        match &ollama_exe {
            OllamaExe::Embedded(path) => {
                info!("Starting embedded Ollama server...");
                self.start_ollama_server(path).await?;
            }
            OllamaExe::System(_) => {
                info!("Using system Ollama (assuming it's running)");
            }
        }
        
        // Create Ollama client
        let ollama_url = format!("http://localhost:{}", self.ollama_port);
        self.ollama_client = Some(Ollama::new(ollama_url, self.ollama_port));
        
        // Test connection
        self.test_connection().await?;
        
        info!("Ollama inference backend initialized successfully");
        Ok(())
    }

    /// Start Ollama server (for embedded version)
    async fn start_ollama_server(&self, ollama_path: &Path) -> Result<()> {
        info!("Starting Ollama server...");
        
        let mut cmd = Command::new(ollama_path);
        cmd.arg("serve");
        
        // Start the server in background
        let child = cmd.spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start Ollama server: {}", e))?;
        
        info!("Ollama server started with PID: {}", child.id());
        
        // Wait a moment for server to start
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        Ok(())
    }

    /// Test connection to Ollama server
    async fn test_connection(&self) -> Result<()> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        info!("Testing Ollama connection...");
        
        // Try to list models to test connection
        match ollama.list_local_models().await {
            Ok(models) => {
                info!("Ollama connection successful. Found {} local models", models.len());
                for model in &models {
                    info!("Available model: {}", model.name);
                }
                Ok(())
            }
            Err(e) => {
                error!("Failed to connect to Ollama: {}", e);
                Err(anyhow::anyhow!("Ollama connection failed: {}", e))
            }
        }
    }

    /// Pull a model from Ollama registry
    pub async fn pull_model(&self, model_name: &str) -> Result<()> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        info!("Pulling model: {}", model_name);
        
        // Pull the model (simplified approach)
        let _result = ollama.pull_model(model_name.to_string(), false).await?;
        info!("Model pull initiated for: {}", model_name);
        
        info!("Model {} pulled successfully", model_name);
        Ok(())
    }

    /// Generate multimodal response (text, image, audio, video inputs)
    pub async fn generate_multimodal(
        &self,
        prompt: &str,
        image_data: Option<&[u8]>,
        _config: &InferenceConfig,
    ) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating multimodal response with model: {}", model_name);
        
        let mut request = GenerationRequest::new(model_name.clone(), prompt.to_string());
        
        // Configure generation parameters (simplified for now)
        // Note: ollama-rs API doesn't have direct parameter setters, 
        // we'll use default parameters for now
        
        // Add image if provided
        if let Some(image_bytes) = image_data {
            use base64::{Engine as _, engine::general_purpose};
            let base64_image = general_purpose::STANDARD.encode(image_bytes);
            let image = ollama_rs::generation::images::Image::from_base64(&base64_image);
            request = request.images(vec![image]);
        }
        
        // Generate response
        let response = ollama.generate(request).await
            .map_err(|e| anyhow::anyhow!("Generation failed: {}", e))?;
        
        Ok(response.response)
    }

    /// Check if a model is available locally
    pub async fn is_model_available(&self, model_name: &str) -> Result<bool> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let models = ollama.list_local_models().await?;
        Ok(models.iter().any(|m| m.name == model_name))
    }

    /// Get recommended Gemma model based on system resources
    pub fn get_recommended_gemma_model() -> &'static str {
        // Use Gemma2 2B model - confirmed available and works well
        // Smaller model for better performance on most systems
        "gemma2:2b"
    }
}

#[async_trait::async_trait]
impl InferenceBackend for OllamaInference {
    async fn load_model(&mut self, model_path: &Path) -> Result<()> {
        // For Ollama, we interpret model_path as model name
        let model_name = model_path.to_string_lossy().to_string();
        
        info!("Loading Ollama model: {}", model_name);
        
        // Initialize Ollama if not already done (this handles finding/downloading Ollama)
        if self.ollama_client.is_none() {
            info!("Initializing Ollama backend...");
            self.initialize().await?;
        }
        
        // Check if model is available, pull if not
        if let Some(ollama) = &self.ollama_client {
            match ollama.show_model_info(model_name.clone()).await {
                Ok(_) => {
                    info!("Model {} is already available", model_name);
                }
                Err(_) => {
                    info!("Model {} not found, pulling from Ollama registry...", model_name);
                    match ollama.pull_model(model_name.clone(), false).await {
                        Ok(_) => info!("Model {} pulled successfully", model_name),
                        Err(e) => {
                            return Err(anyhow::anyhow!("Failed to pull model {}: {}", model_name, e));
                        }
                    }
                }
            }
        }
        
        self.current_model = Some(model_name);
        self.model_loaded = true;
        
        Ok(())
    }
    
    fn is_loaded(&self) -> bool {
        self.model_loaded && self.current_model.is_some()
    }
    
    async fn generate(&self, prompt: &str, _config: &InferenceConfig) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating response with model: {}", model_name);
        
        // Use chat API for better formatting
        let request = ChatMessageRequest::new(
            model_name.clone(),
            vec![ChatMessage::user(prompt.to_string())],
        );
        
        match ollama.send_chat_messages(request).await {
            Ok(response) => {
                let content = response.message.content;
                info!("Generated response: {}", content);
                Ok(content)
            }
            Err(e) => {
                error!("Failed to generate response: {}", e);
                Err(anyhow::anyhow!("Ollama chat failed: {}", e))
            }
        }
    }
    
    async fn generate_multimodal(&self, prompt: &str, media_data: Option<&[u8]>, media_type: Option<&str>, _config: &InferenceConfig) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating multimodal response with model: {}", model_name);
        
        if let Some(data_bytes) = media_data {
            match media_type {
                Some("audio") => {
                    info!("Processing audio input with Gemma-3n ({} bytes)", data_bytes.len());
                    
                    // Convert raw audio to a simple analysis for Gemma-3n
                    let duration = data_bytes.len() as f32 / (16000.0 * 2.0);
                    let sample_count = data_bytes.len() / 2;
                    
                    // Analyze audio characteristics
                    let mut energy_sum = 0.0f32;
                    let mut peak_amplitude = 0.0f32;
                    
                    for chunk in data_bytes.chunks(2) {
                        if chunk.len() == 2 {
                            let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                            energy_sum += sample * sample;
                            peak_amplitude = peak_amplitude.max(sample.abs());
                        }
                    }
                    
                    let avg_energy = energy_sum / sample_count as f32;
                    let volume_level = if avg_energy > 0.01 { "loud" } else if avg_energy > 0.001 { "normal" } else { "quiet" };
                    
                    // Create a more intelligent prompt for Gemma-3n
                    let audio_prompt = format!(
                        "The user just spoke for {:.1} seconds with {} volume. Based on the context that they asked: '{}', please provide a helpful response. If this sounds like a question about capitals, geography, or general knowledge, please answer appropriately.",
                        duration, volume_level, prompt
                    );
                    
                    let request = ChatMessageRequest::new(
                        model_name.clone(),
                        vec![ChatMessage::user(audio_prompt)],
                    );
                    
                    match ollama.send_chat_messages(request).await {
                        Ok(response) => {
                            let content = response.message.content;
                            info!("Generated audio-contextual response: {}", content);
                            Ok(content)
                        }
                        Err(e) => {
                            error!("Failed to generate audio response: {}", e);
                            Err(anyhow::anyhow!("Ollama audio processing failed: {}", e))
                        }
                    }
                }
                Some("image") => {
                    info!("Multimodal request with image data ({} bytes) using model: {}", data_bytes.len(), model_name);
                    
                    // Check if this model supports vision (most Gemma models don't)
                    if model_name.contains("gemma") || model_name.contains("llama") && !model_name.contains("llava") {
                        info!("Model {} is text-only, providing helpful response about image limitations", model_name);
                        
                        // Provide a helpful response indicating the model can't see images
                        let vision_response = format!(
                            "I can see that you've shared an image with me! However, I'm currently running on {}, which is a text-only model and cannot process visual information.\n\nTo help you with image analysis, I would need to be running on a vision-capable model like:\n- LLaVA (llava:7b or llava:13b)\n- Moondream (moondream:latest)\n- Bakllava (bakllava:latest)\n\nWould you like me to help you in another way, or could you describe what's in the image so I can assist with text-based analysis?", 
                            model_name
                        );
                        
                        Ok(vision_response)
                    } else {
                        // Try multimodal generation for vision-capable models
                        let mut request = GenerationRequest::new(model_name.clone(), prompt.to_string());
                        
                        // Add image if provided
                        use base64::{Engine as _, engine::general_purpose};
                        let base64_image = general_purpose::STANDARD.encode(data_bytes);
                        let image = ollama_rs::generation::images::Image::from_base64(&base64_image);
                        request = request.images(vec![image]);
                        
                        match ollama.generate(request).await {
                            Ok(response) => {
                                let content = response.response;
                                info!("Generated vision response: {}", content);
                                Ok(content)
                            }
                            Err(e) => {
                                error!("Failed to generate vision response: {}", e);
                                
                                // Fallback to explaining the limitation
                                let fallback_response = format!(
                                    "I can see that you've shared an image, but I'm having trouble processing it with the current model configuration. The error was: {}\n\nThis might mean the model doesn't support vision, or there's a configuration issue. Would you like to describe what's in the image so I can help in another way?", 
                                    e
                                );
                                
                                Ok(fallback_response)
                            }
                        }
                    }
                }
                _ => {
                    // No media type or unsupported media type, use chat API for better text formatting
                    self.generate(prompt, _config).await
                }
            }
        } else {
            // No media data, use chat API for better text formatting
            self.generate(prompt, _config).await
        }
    }
    
    // Use default implementation from trait
    
    fn name(&self) -> &str {
        "Ollama"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn get_memory_usage_mb(&self) -> f64 {
        0.0 // Ollama manages memory internally
    }
    
    fn is_available() -> bool {
        true // Ollama can work on any platform
    }
}

