use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter};
use tracing::{error, info};
use super::inference_backend::{InferenceConfig, BackendType};
use super::inference_manager::{InferenceManager};

// Ollama Gemma models - no need for explicit model IDs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadProgress {
    pub progress: u32,
    pub status: String,
    pub model_name: String,
}

#[derive(Debug, Clone)]
struct MultimodalChatTemplate {
    // Gemma-3n uses specific turn-based format
    start_of_turn: String,
    end_of_turn: String,
    user_role: String,
    model_role: String,
}

impl Default for MultimodalChatTemplate {
    fn default() -> Self {
        Self {
            // Gemma-3n official chat template format
            start_of_turn: "<start_of_turn>".to_string(),
            end_of_turn: "<end_of_turn>".to_string(),
            user_role: "user".to_string(),
            model_role: "model".to_string(),
        }
    }
}

pub struct AIManager {
    app_handle: AppHandle,
    model_loaded: bool,
    model_path: Option<PathBuf>,
    chat_template: MultimodalChatTemplate,
    selected_model: String,
    inference_manager: InferenceManager,
    backend_type: BackendType,
}

impl AIManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        // Always use Ollama backend (only option)
        let backend_type = BackendType::Ollama;
        let inference_manager = InferenceManager::new(backend_type)?;
        
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: MultimodalChatTemplate::default(),
            selected_model: "gemma3n:e2b".to_string(), // Gemma 3n E2B with multimodal capabilities
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
            chat_template: MultimodalChatTemplate::default(),
            selected_model: "gemma3n:e2b".to_string(),
            inference_manager,
            backend_type,
        })
    }

    pub async fn load_model(&mut self) -> Result<()> {
        self.emit_progress(0, "Starting Gemma-3n E2B multimodal model initialization...", &self.selected_model).await;
        self.load_ollama_model().await
    }

    /// Load model using Ollama backend
    async fn load_ollama_model(&mut self) -> Result<()> {
        
        self.emit_progress(10, "Initializing Ollama...", &self.selected_model).await;
        
        // Unfortunately, we need a workaround for the mutable reference issue
        // For now, we'll create a simplified approach
        
        self.emit_progress(40, "Checking model availability...", &self.selected_model).await;
        
        // For Ollama, we use the model name as the "path"
        let model_path = std::path::Path::new(&self.selected_model);
        
        self.emit_progress(90, "Loading model...", &self.selected_model).await;
        
        // Load the model through the inference manager
        match self.inference_manager.load_model(model_path).await {
            Ok(_) => {
                self.model_loaded = true;
                self.emit_progress(100, &format!("Gemma-3n {} ready with Ollama! Multimodal AI with vision capabilities at your service.", self.selected_model), &self.selected_model).await;
                self.emit_completion(true, None).await;
                
                info!("Gemma-3n model {} loaded successfully with Ollama", self.selected_model);
                Ok(())
            }
            Err(e) => {
                error!("Failed to load Ollama model: {}", e);
                self.emit_progress(0, &format!("Failed to load Ollama model: {}", e), &self.selected_model).await;
                self.emit_completion(false, Some(e.to_string())).await;
                Err(e)
            }
        }
    }

    pub async fn generate_response(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_response_with_system_prompt(prompt, max_tokens, None).await
    }
    
    pub async fn generate_response_with_image(&self, prompt: &str, image_data: &[u8], max_tokens: usize) -> Result<String> {
        self.generate_response_with_image_and_system_prompt(prompt, image_data, max_tokens, None).await
    }

    pub async fn generate_response_with_audio(&self, prompt: &str, audio_data: &[u8], max_tokens: usize) -> Result<String> {
        self.generate_response_with_audio_and_system_prompt(prompt, audio_data, max_tokens, None).await
    }
    
    pub async fn generate_response_with_audio_and_system_prompt(&self, prompt: &str, audio_data: &[u8], max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant. You are listening to a user and should respond to their voice input.".to_string()
        );
        
        info!("Processing multimodal prompt with audio ({} bytes): {}", audio_data.len(), prompt);
        
        // Format prompt using Gemma-3n official chat template for multimodal
        let formatted_prompt = if !system.is_empty() {
            format!(
                "{}{}
{}
{}
{}{}
{}
{}
{}{}
",
                self.chat_template.start_of_turn,
                "system",
                system,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        } else {
            format!(
                "{}{}
{}
{}
{}{}
",
                self.chat_template.start_of_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        };
        
        info!("Formatted multimodal prompt for Gemma-3n: {}", formatted_prompt);
        
        // Create inference config for multimodal generation
        let config = InferenceConfig {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        };
        
        // Use Ollama multimodal generation
        match self.inference_manager.generate_multimodal(&formatted_prompt, Some(audio_data), Some("audio"), &config).await {
            Ok(response) => {
                info!("Generated multimodal response: {}", response);
                Ok(response)
            }
            Err(e) => {
                error!("Multimodal inference error: {}", e);
                Err(anyhow::anyhow!("Multimodal inference failed: {}", e))
            }
        }
    }
    
    pub async fn generate_response_with_system_prompt(&self, prompt: &str, max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant. Provide clear, conversational responses. Use natural formatting with line breaks and structure your responses naturally. Be helpful and friendly in your interactions.".to_string()
        );
        
        // Format prompt using Gemma-3n official chat template
        let formatted_prompt = if !system.is_empty() {
            format!(
                "{}{}
{}
{}
{}{}
{}
{}
{}{}
",
                self.chat_template.start_of_turn,
                "system",
                system,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        } else {
            format!(
                "{}{}
{}
{}
{}{}
",
                self.chat_template.start_of_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        };
        
        info!("Processing prompt: {}", prompt);
        info!("Formatted for Gemma-3n: {}", formatted_prompt);
        
        // Create inference config
        let config = InferenceConfig {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        };
        
        // Use Ollama inference backend
        match self.inference_manager.generate(&formatted_prompt, &config).await {
            Ok(response) => {
                info!("Generated response: {}", response);
                Ok(response)
            }
            Err(e) => {
                error!("Inference error: {}", e);
                Err(anyhow::anyhow!("Model inference failed: {}", e))
            }
        }
    }
    
    pub async fn generate_response_with_image_and_system_prompt(&self, prompt: &str, image_data: &[u8], max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant powered by Gemma-3n. You can see and analyze images, understand visual content, and answer questions about what you see in images.".to_string()
        );
        
        info!("Processing multimodal prompt with image ({} bytes): {}", image_data.len(), prompt);
        
        // Format prompt using Gemma-3n official chat template for multimodal
        let formatted_prompt = if !system.is_empty() {
            format!(
                "{}{}
{}
{}
{}{}
{}
{}
{}{}
",
                self.chat_template.start_of_turn,
                "system",
                system,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        } else {
            format!(
                "{}{}
{}
{}
{}{}
",
                self.chat_template.start_of_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        };
        
        info!("Formatted multimodal prompt for Gemma-3n: {}", formatted_prompt);
        
        // Create inference config for multimodal generation
        let config = InferenceConfig {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        };
        
        // Use Ollama multimodal generation
        match self.inference_manager.generate_multimodal(&formatted_prompt, Some(image_data), Some("image"), &config).await {
            Ok(response) => {
                info!("Generated multimodal response: {}", response);
                Ok(response)
            }
            Err(e) => {
                error!("Multimodal inference error: {}", e);
                Err(anyhow::anyhow!("Multimodal inference failed: {}", e))
            }
        }
    }

    async fn emit_progress(&self, progress: u32, status: &str, model_name: &str) {
        let progress_data = ModelLoadProgress {
            progress,
            status: status.to_string(),
            model_name: model_name.to_string(),
        };
        
        if let Err(e) = self.app_handle.emit("model-loading-progress", &progress_data) {
            error!("Failed to emit progress: {}", e);
        }
    }

    async fn emit_completion(&self, success: bool, error: Option<String>) {
        let _ = self.app_handle.emit(
            "model-loading-complete",
            serde_json::json!({
                "success": success,
                "error": error,
                "model_name": self.selected_model,
            }),
        );
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }
    
    pub async fn get_backend_info(&self) -> String {
        let backend_name = self.inference_manager.backend_name().await;
        let system_info = InferenceManager::get_system_info();
        
        format!(
            "Current Backend: {}\\nBackend Type: {:?}\\n\\n{}",
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
        
        Ok(vec![(
            self.inference_manager.backend_name().await,
            self.inference_manager.benchmark_backend(prompt, &config).await?
        )])
    }
}
