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
        let inference_manager = InferenceManager::with_app_handle(backend_type, app_handle.clone())?;
        
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: MultimodalChatTemplate::default(),
            selected_model: "gemma3n:e4b".to_string(), // Gemma 3n E4B with enhanced multimodal capabilities
            inference_manager,
            backend_type,
        })
    }
    
    pub fn with_backend(app_handle: AppHandle, backend_type: BackendType) -> Result<Self> {
        let inference_manager = InferenceManager::with_app_handle(backend_type, app_handle.clone())?;
        
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: MultimodalChatTemplate::default(),
            selected_model: "gemma3n:e4b".to_string(), // Gemma 3n E4B with enhanced multimodal capabilities
            inference_manager,
            backend_type,
        })
    }

    pub async fn load_model(&mut self) -> Result<()> {
        self.emit_progress(0, "Starting Gemma3n E4B multimodal model initialization...", &self.selected_model).await;
        self.load_ollama_model().await
    }

    /// Load model using Ollama backend
    async fn load_ollama_model(&mut self) -> Result<()> {
        
        self.emit_progress(5, "🚀 Starting Ollama model initialization...", &self.selected_model).await;
        
        // For Ollama, we use the model name as the "path"
        let model_path = std::path::Path::new(&self.selected_model);
        
        // The inference manager will handle detailed progress through the ollama_inference.rs progress system
        // The ollama_inference.rs pull_model_with_progress() method provides granular file-by-file progress
        // which will emit progress events directly to the frontend
        
        // Load the model through the inference manager (this handles all the detailed progress internally)
        match self.inference_manager.load_model(model_path).await {
            Ok(_) => {
                self.model_loaded = true;
                self.emit_progress(100, &format!("🎉 {} ready with Ollama! Multimodal AI with vision capabilities at your service.", self.selected_model), &self.selected_model).await;
                self.emit_completion(true, None).await;
                
                info!("Gemma-3n model {} loaded successfully with Ollama", self.selected_model);
                Ok(())
            }
            Err(e) => {
                error!("Failed to load Ollama model: {}", e);
                self.emit_progress(0, &format!("❌ Failed to load Ollama model: {}", e), &self.selected_model).await;
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

    /// Comprehensive multimodal conversation method supporting text, image, audio, and context
    pub async fn generate_multimodal_response(
        &self, 
        text_prompt: &str,
        image_data: Option<&[u8]>,
        audio_data: Option<&[u8]>,
        system_prompt: Option<String>,
        conversation_context: Option<&str>,
        max_tokens: usize
    ) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }

        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant powered by Gemma 3N with advanced multimodal capabilities. You can see images, understand audio input, and engage in natural conversation. Respond naturally and helpfully.".to_string()
        );

        info!("Processing comprehensive multimodal request - Text: {} chars, Image: {}, Audio: {}", 
              text_prompt.len(),
              image_data.map(|d| format!("{} bytes", d.len())).unwrap_or("None".to_string()),
              audio_data.map(|d| format!("{} bytes", d.len())).unwrap_or("None".to_string()));

        // Build comprehensive prompt with context
        let mut full_prompt = String::new();
        
        // Add conversation context if provided
        if let Some(context) = conversation_context {
            if !context.is_empty() {
                full_prompt.push_str("Previous conversation context:\n");
                full_prompt.push_str(context);
                full_prompt.push_str("\n\nCurrent request:\n");
            }
        }
        
        // Add current text prompt
        full_prompt.push_str(text_prompt);

        // Format prompt using Gemma 3N official chat template for multimodal
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
                full_prompt,
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
                full_prompt,
                self.chat_template.end_of_turn,
                self.chat_template.start_of_turn,
                self.chat_template.model_role
            )
        };

        // Create inference config optimized for multimodal
        let config = InferenceConfig {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        };

        // Determine media type and data based on what's provided
        let (media_data, media_type) = if let Some(img_data) = image_data {
            // Prioritize image if both are provided (most common multimodal case)
            (Some(img_data), Some("image"))
        } else if let Some(aud_data) = audio_data {
            (Some(aud_data), Some("audio"))
        } else {
            (None, None)
        };

        // Use enhanced Ollama multimodal generation with Gemma3NProcessor
        match self.inference_manager.generate_multimodal(&formatted_prompt, media_data, media_type, &config).await {
            Ok(response) => {
                info!("Generated comprehensive multimodal response: {} chars", response.len());
                Ok(response)
            }
            Err(e) => {
                error!("Comprehensive multimodal inference error: {}", e);
                Err(anyhow::anyhow!("Multimodal conversation failed: {}", e))
            }
        }
    }

    async fn emit_progress(&self, progress: u32, status: &str, model_name: &str) {
        let progress_data = ModelLoadProgress {
            progress,
            status: status.to_string(),
            model_name: model_name.to_string(),
        };
        
        if let Err(e) = self.app_handle.emit_to(tauri::EventTarget::Any, "model-loading-progress", &progress_data) {
            error!("Failed to emit progress: {}", e);
        }
    }

    async fn emit_completion(&self, success: bool, error: Option<String>) {
        let _ = self.app_handle.emit_to(
            tauri::EventTarget::Any,
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
