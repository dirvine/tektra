use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::info;

use crate::ai::{
    UnifiedModelManager, GenerationParams,
    TemplateManager, ChatMessage, MessageRole,
    ModelConfigLoader, ModelInfo,
    unified_model_manager::MultimodalInput as UnifiedMultimodalInput,
};
use tauri::Emitter;

/// Integration layer between UnifiedModelManager and existing Tektra infrastructure
pub struct TektraModelIntegration {
    unified_manager: Arc<UnifiedModelManager>,
    template_manager: Arc<TemplateManager>,
    config_loader: Arc<Mutex<ModelConfigLoader>>,
    app_handle: tauri::AppHandle,
    current_conversation: Arc<RwLock<Vec<ChatMessage>>>,
}

impl TektraModelIntegration {
    pub async fn new(app_handle: tauri::AppHandle) -> Result<Self> {
        info!("Initializing Tektra model integration");
        
        // Initialize managers
        let unified_manager = Arc::new(UnifiedModelManager::new());
        let template_manager = Arc::new(TemplateManager::new());
        
        // Load model configuration
        let mut config_loader = ModelConfigLoader::new("models.toml");
        config_loader.load().await?;
        
        // Set backend preferences
        let preferences = config_loader.get_backend_preferences();
        unified_manager.load_preferences(preferences).await?;
        
        Ok(Self {
            unified_manager,
            template_manager,
            config_loader: Arc::new(Mutex::new(config_loader)),
            app_handle,
            current_conversation: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Load a model by ID (compatible with existing Tektra commands)
    pub async fn load_model(&self, model_id: &str) -> Result<()> {
        info!("Loading model: {}", model_id);
        
        // Emit loading event
        let _ = self.app_handle.emit("model-loading-progress", serde_json::json!({
            "progress": 0,
            "status": "Initializing model loading...",
            "model_name": model_id
        }));
        
        // Get model configuration
        let config_loader = self.config_loader.lock().await;
        let model_def = config_loader.get_model_definition(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in configuration", model_id))?;
        
        let model_config = config_loader.to_model_config(model_def);
        drop(config_loader);
        
        // Load the model
        self.unified_manager.load_model(model_config).await?;
        
        // Emit completion event
        let _ = self.app_handle.emit("model-loading-complete", serde_json::json!({
            "success": true,
            "model_name": model_id
        }));
        
        info!("Model '{}' loaded successfully", model_id);
        Ok(())
    }
    
    /// Process text input (compatible with existing process_input command)
    pub async fn process_input(&self, input: &str) -> Result<String> {
        // Add user message to conversation
        let mut conversation = self.current_conversation.write().await;
        conversation.push(ChatMessage {
            role: MessageRole::User,
            content: input.to_string(),
        });
        
        // Get current model and template
        let model_config = self.unified_manager.current_model_config().await
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let template_name = model_config.template_name.as_deref().unwrap_or("chatml");
        let template = self.template_manager.get_template(template_name).await
            .ok_or_else(|| anyhow::anyhow!("Template '{}' not found", template_name))?;
        
        // Format prompt
        let prompt = self.template_manager.format_prompt(
            &template,
            &conversation,
            true, // Add generation prompt
        );
        
        drop(conversation);
        
        // Generate response
        let params = GenerationParams::default();
        let response = self.unified_manager.generate_text(&prompt, &params).await?;
        
        // Add assistant response to conversation
        let mut conversation = self.current_conversation.write().await;
        conversation.push(ChatMessage {
            role: MessageRole::Assistant,
            content: response.clone(),
        });
        
        // Keep conversation history limited
        if conversation.len() > 20 {
            conversation.drain(0..2);
        }
        
        Ok(response)
    }
    
    /// Process multimodal input (new capability)
    pub async fn process_multimodal(
        &self,
        text: Option<String>,
        image_data: Option<Vec<u8>>,
        audio_data: Option<Vec<u8>>,
    ) -> Result<String> {
        // Check if current model supports multimodal
        let capabilities = self.unified_manager.get_capabilities().await
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        if !capabilities.image_understanding && image_data.is_some() {
            return Err(anyhow::anyhow!("Current model does not support images"));
        }
        
        if !capabilities.audio_processing && audio_data.is_some() {
            return Err(anyhow::anyhow!("Current model does not support audio"));
        }
        
        // Create multimodal input
        let mut input = UnifiedMultimodalInput {
            text: text.clone(),
            images: vec![],
            audio: audio_data,
            video: None,
        };
        
        if let Some(img_data) = image_data {
            input.images.push(img_data);
        }
        
        // Get template for multimodal formatting
        let model_config = self.unified_manager.current_model_config().await
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let template_name = model_config.template_name.as_deref().unwrap_or("chatml");
        let template = self.template_manager.get_template(template_name).await
            .ok_or_else(|| anyhow::anyhow!("Template '{}' not found", template_name))?;
        
        // Format multimodal content if text is provided
        if let Some(text_content) = &text {
            let formatted_text = self.template_manager.format_multimodal_content(
                &template,
                text_content,
                !input.images.is_empty(),
                input.audio.is_some(),
                false, // No video for now
            );
            input.text = Some(formatted_text);
        }
        
        // Save input state before moving
        let has_images = !input.images.is_empty();
        let has_audio = input.audio.is_some();
        
        // Generate response
        let params = GenerationParams::default();
        let response = self.unified_manager.generate_multimodal(input, &params).await?;
        
        // Add to conversation history
        let mut conversation = self.current_conversation.write().await;
        
        let user_content = if let Some(t) = text {
            if has_images {
                format!("[Image] {}", t)
            } else if has_audio {
                format!("[Audio] {}", t)
            } else {
                t
            }
        } else {
            "[Multimodal input]".to_string()
        };
        
        conversation.push(ChatMessage {
            role: MessageRole::User,
            content: user_content,
        });
        
        conversation.push(ChatMessage {
            role: MessageRole::Assistant,
            content: response.clone(),
        });
        
        Ok(response)
    }
    
    /// Stream generation (new capability)
    pub async fn stream_response(&self, input: &str) -> Result<tokio::sync::mpsc::Receiver<String>> {
        // Add user message
        let mut conversation = self.current_conversation.write().await;
        conversation.push(ChatMessage {
            role: MessageRole::User,
            content: input.to_string(),
        });
        
        // Get template and format prompt
        let model_config = self.unified_manager.current_model_config().await
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let template_name = model_config.template_name.as_deref().unwrap_or("chatml");
        let template = self.template_manager.get_template(template_name).await
            .ok_or_else(|| anyhow::anyhow!("Template '{}' not found", template_name))?;
        
        let prompt = self.template_manager.format_prompt(
            &template,
            &conversation,
            true,
        );
        
        drop(conversation);
        
        // Stream generation
        let mut params = GenerationParams::default();
        params.stream = true;
        
        let receiver = self.unified_manager.generate_stream(&prompt, &params).await?;
        
        // Create a channel for the response we'll return
        let (tx, rx) = tokio::sync::mpsc::channel::<String>(100);
        
        // Spawn task to collect full response for conversation history
        let conversation = self.current_conversation.clone();
        tokio::spawn(async move {
            let mut full_response = String::new();
            let mut receiver = receiver;
            while let Some(chunk) = receiver.recv().await {
                full_response.push_str(&chunk);
                // Forward the chunk
                if tx.send(chunk).await.is_err() {
                    break;
                }
            }
            
            let mut conv = conversation.write().await;
            conv.push(ChatMessage {
                role: MessageRole::Assistant,
                content: full_response,
            });
        });
        
        Ok(rx)
    }
    
    /// List available models
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let config_loader = self.config_loader.lock().await;
        Ok(config_loader.list_models())
    }
    
    /// Get current model info
    pub async fn current_model(&self) -> Result<Option<String>> {
        Ok(self.unified_manager.current_backend().await)
    }
    
    /// Switch backend for current model
    pub async fn switch_backend(&self, backend: &str) -> Result<()> {
        info!("Switching to backend: {}", backend);
        self.unified_manager.switch_backend(backend).await
    }
    
    /// Clear conversation history
    pub async fn clear_conversation(&self) -> Result<()> {
        let mut conversation = self.current_conversation.write().await;
        conversation.clear();
        info!("Conversation history cleared");
        Ok(())
    }
    
    /// Set system prompt
    pub async fn set_system_prompt(&self, prompt: &str) -> Result<()> {
        let mut conversation = self.current_conversation.write().await;
        
        // Remove existing system message if any
        conversation.retain(|msg| msg.role != MessageRole::System);
        
        // Add new system message at the beginning
        conversation.insert(0, ChatMessage {
            role: MessageRole::System,
            content: prompt.to_string(),
        });
        
        info!("System prompt updated");
        Ok(())
    }
    
    /// Get memory usage
    pub async fn get_memory_usage(&self) -> usize {
        self.unified_manager.total_memory_usage().await
    }
    
    /// Validate configuration
    pub async fn validate_config(&self) -> Vec<String> {
        let config_loader = self.config_loader.lock().await;
        config_loader.validate()
    }
    
    /// Reload configuration from file
    pub async fn reload_config(&self) -> Result<()> {
        let mut config_loader = self.config_loader.lock().await;
        config_loader.reload().await?;
        
        // Update backend preferences
        let preferences = config_loader.get_backend_preferences();
        drop(config_loader);
        
        self.unified_manager.load_preferences(preferences).await?;
        
        info!("Configuration reloaded");
        Ok(())
    }
}

/// Compatibility layer for existing Tektra commands
impl TektraModelIntegration {
    /// Load Gemma 3N model (for backward compatibility)
    pub async fn load_gemma3n(&self) -> Result<()> {
        self.load_model("gemma3n:e4b").await
    }
    
    /// Process audio input with Whisper transcription
    pub async fn process_audio_with_transcription(
        &self,
        transcription: &str,
        audio_data: Vec<u8>,
    ) -> Result<String> {
        // Use multimodal processing if model supports it
        let capabilities = self.unified_manager.get_capabilities().await;
        
        if let Some(caps) = capabilities {
            if caps.audio_processing {
                // Send both transcription and audio
                return self.process_multimodal(
                    Some(transcription.to_string()),
                    None,
                    Some(audio_data),
                ).await;
            }
        }
        
        // Fall back to text-only processing
        self.process_input(transcription).await
    }
    
    /// Process image with description
    pub async fn process_image_with_description(
        &self,
        description: &str,
        image_data: Vec<u8>,
    ) -> Result<String> {
        self.process_multimodal(
            Some(description.to_string()),
            Some(image_data),
            None,
        ).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_conversation_management() {
        // This would require a mock app_handle for testing
        // Placeholder for actual tests
    }
}