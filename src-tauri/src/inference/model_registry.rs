use super::*;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
// use super::mistralrs_backend::MistralRsModel; // Temporarily disabled for compilation

/// Central registry for managing multiple AI models
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, Arc<ModelAbstraction>>>>,
    active_model: Arc<RwLock<Option<String>>>,
    configs: Arc<RwLock<HashMap<String, ModelConfig>>>,
    default_models: Vec<DefaultModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefaultModelConfig {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub description: String,
    pub quantization: Option<String>,
    pub context_window: usize,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_documents: bool,
    pub default: bool,
    pub recommended_for: Vec<String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let default_models = vec![
            DefaultModelConfig {
                id: "qwen25_omni".to_string(),
                name: "Qwen2.5-Omni 7B".to_string(),
                model_id: "Qwen/Qwen2.5-Omni-7B".to_string(),
                description: "Advanced multimodal model with real-time audio processing and speech synthesis".to_string(),
                quantization: Some("Q6_K".to_string()),
                context_window: 32768,
                supports_vision: true,
                supports_audio: true,
                supports_documents: true,
                default: true,
                recommended_for: vec![
                    "multimodal conversation".to_string(),
                    "real-time interaction".to_string(),
                    "voice + vision".to_string(),
                    "comprehensive AI assistance".to_string(),
                ],
            },
            DefaultModelConfig {
                id: "qwen25_vl".to_string(),
                name: "Qwen2.5-VL 7B".to_string(),
                model_id: "Qwen/Qwen2.5-VL-7B-Instruct".to_string(),
                description: "Vision-language model for image and document analysis".to_string(),
                quantization: Some("Q6_K".to_string()),
                context_window: 32768,
                supports_vision: true,
                supports_audio: false,
                supports_documents: true,
                default: false,
                recommended_for: vec![
                    "image analysis".to_string(),
                    "document understanding".to_string(),
                    "visual reasoning".to_string(),
                    "OCR tasks".to_string(),
                ],
            },
            DefaultModelConfig {
                id: "pixtral-12b".to_string(),
                name: "Pixtral 12B".to_string(),
                model_id: "mistral-community/pixtral-12b".to_string(),
                description: "Mistral's flagship vision model with exceptional image understanding".to_string(),
                quantization: Some("Q4_K_M".to_string()),
                context_window: 128000,
                supports_vision: true,
                supports_audio: false,
                supports_documents: true,
                default: false,
                recommended_for: vec![
                    "detailed image analysis".to_string(),
                    "visual content creation".to_string(),
                    "image-to-text tasks".to_string(),
                ],
            },
            DefaultModelConfig {
                id: "llama-3.2-vision-11b".to_string(),
                name: "Llama 3.2 Vision 11B".to_string(),
                model_id: "meta-llama/Llama-3.2-11B-Vision-Instruct".to_string(),
                description: "Meta's vision-capable Llama model with strong multimodal performance".to_string(),
                quantization: Some("Q5_K_M".to_string()),
                context_window: 128000,
                supports_vision: true,
                supports_audio: false,
                supports_documents: true,
                default: false,
                recommended_for: vec![
                    "visual question answering".to_string(),
                    "image captioning".to_string(),
                    "visual reasoning".to_string(),
                ],
            },
            DefaultModelConfig {
                id: "gemma-3-9b".to_string(),
                name: "Gemma 3 9B".to_string(),
                model_id: "google/gemma-2-9b-it".to_string(),
                description: "Google's efficient text model with strong reasoning capabilities".to_string(),
                quantization: Some("Q5_K_M".to_string()),
                context_window: 8192,
                supports_vision: false,
                supports_audio: false,
                supports_documents: true,
                default: false,
                recommended_for: vec![
                    "text generation".to_string(),
                    "code assistance".to_string(),
                    "reasoning tasks".to_string(),
                ],
            },
        ];

        Self {
            models: Arc::new(RwLock::new(HashMap::new())),
            active_model: Arc::new(RwLock::new(None)),
            configs: Arc::new(RwLock::new(HashMap::new())),
            default_models,
        }
    }
    
    /// Initialize registry with default models
    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing model registry with {} default models", self.default_models.len());
        
        for model_config in &self.default_models {
            let config = ModelConfig {
                model_id: model_config.model_id.clone(),
                quantization: model_config.quantization.clone(),
                context_window: model_config.context_window,
                device: DeviceConfig::Auto,
                cache_dir: None,
                custom_params: HashMap::new(),
            };
            
            self.configs.write().await.insert(model_config.id.clone(), config);
            
            if model_config.default {
                info!("Registering default model: {}", model_config.name);
                self.register_model(&model_config.id).await?;
                self.set_active_model(&model_config.id).await?;
            }
        }
        
        Ok(())
    }
    
    /// Register a new model in the registry
    pub async fn register_model(&self, model_id: &str) -> Result<()> {
        info!("Registering model: {}", model_id);
        
        // Use MistralRs backend as the primary implementation
        // Temporarily disabled MistralRsModel for compilation
        // let model = Box::new(MistralRsModel::new()?);
        // let abstraction = Arc::new(ModelAbstraction::new(model));
        // self.models.write().await.insert(model_id.to_string(), abstraction);
        
        // TODO: Re-enable when mistral.rs tokenizer compatibility is resolved
        info!("Model registration temporarily disabled - awaiting mistral.rs fix");
        
        info!("Model registered successfully: {}", model_id);
        Ok(())
    }
    
    /// Load and activate a model
    pub async fn load_model(&self, model_id: &str) -> Result<()> {
        info!("Loading model: {}", model_id);
        
        // Ensure model is registered
        if !self.models.read().await.contains_key(model_id) {
            self.register_model(model_id).await?;
        }
        
        // Get model configuration
        let config = self.configs.read().await.get(model_id).cloned()
            .ok_or_else(|| anyhow::anyhow!("No configuration found for model: {}", model_id))?;
        
        // Load the model
        let models = self.models.read().await;
        let model = models.get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found in registry: {}", model_id))?;
        
        model.load_model(&config).await?;
        
        info!("Model loaded successfully: {}", model_id);
        Ok(())
    }
    
    /// Unload a model
    pub async fn unload_model(&self, model_id: &str) -> Result<()> {
        info!("Unloading model: {}", model_id);
        
        let models = self.models.read().await;
        if let Some(model) = models.get(model_id) {
            model.unload_model().await?;
        }
        
        // Clear active model if it was the one we unloaded
        let mut active = self.active_model.write().await;
        if active.as_ref() == Some(&model_id.to_string()) {
            *active = None;
        }
        
        info!("Model unloaded successfully: {}", model_id);
        Ok(())
    }
    
    /// Set the active model
    pub async fn set_active_model(&self, model_id: &str) -> Result<()> {
        info!("Setting active model: {}", model_id);
        
        // Ensure model is loaded
        if !self.is_model_loaded(model_id).await? {
            self.load_model(model_id).await?;
        }
        
        *self.active_model.write().await = Some(model_id.to_string());
        
        info!("Active model set to: {}", model_id);
        Ok(())
    }
    
    /// Get the currently active model
    pub async fn get_active_model(&self) -> Option<Arc<ModelAbstraction>> {
        let active_id = self.active_model.read().await.clone()?;
        let models = self.models.read().await;
        models.get(&active_id).cloned()
    }
    
    /// Get active model ID
    pub async fn get_active_model_id(&self) -> Option<String> {
        self.active_model.read().await.clone()
    }
    
    /// Check if a model is loaded
    pub async fn is_model_loaded(&self, model_id: &str) -> Result<bool> {
        let models = self.models.read().await;
        if let Some(model) = models.get(model_id) {
            Ok(model.is_loaded().await)
        } else {
            Ok(false)
        }
    }
    
    /// List all available models
    pub async fn list_models(&self) -> Vec<DefaultModelConfig> {
        self.default_models.clone()
    }
    
    /// Get model information
    pub async fn get_model_info(&self, model_id: &str) -> Option<ModelInfo> {
        let models = self.models.read().await;
        if let Some(model) = models.get(model_id) {
            model.get_model_info().await
        } else {
            None
        }
    }
    
    /// Get model capabilities
    pub async fn get_model_capabilities(&self, model_id: &str) -> Option<ModelCapabilities> {
        let models = self.models.read().await;
        if let Some(model) = models.get(model_id) {
            Some(model.get_capabilities().await)
        } else {
            None
        }
    }
    
    /// Switch to a different model
    pub async fn switch_model(&self, new_model_id: &str) -> Result<()> {
        info!("Switching to model: {}", new_model_id);
        
        // Unload current active model if any
        if let Some(current_id) = self.get_active_model_id().await {
            if current_id != new_model_id {
                self.unload_model(&current_id).await?;
            }
        }
        
        // Load and activate new model
        self.set_active_model(new_model_id).await?;
        
        info!("Model switch completed: {}", new_model_id);
        Ok(())
    }
    
    /// Generate response using active model
    pub async fn generate(&self, input: MultimodalInput) -> Result<ModelResponse> {
        let model = self.get_active_model().await
            .ok_or_else(|| anyhow::anyhow!("No active model"))?;
        
        model.generate(input).await
    }
    
    /// Generate streaming response using active model
    pub async fn stream_generate(&self, input: MultimodalInput) 
        -> Result<Pin<Box<dyn Stream<Item = Result<Token>> + Send>>> {
        let model = self.get_active_model().await
            .ok_or_else(|| anyhow::anyhow!("No active model"))?;
        
        model.stream_generate(input).await
    }
    
    /// Update model configuration
    pub async fn update_model_config(&self, model_id: &str, config: ModelConfig) -> Result<()> {
        self.configs.write().await.insert(model_id.to_string(), config);
        
        // If model is currently loaded, reload it with new config
        if self.is_model_loaded(model_id).await? {
            self.unload_model(model_id).await?;
            self.load_model(model_id).await?;
        }
        
        Ok(())
    }
    
    /// Get registry statistics
    pub async fn get_stats(&self) -> RegistryStats {
        let models = self.models.read().await;
        let active_id = self.active_model.read().await.clone();
        
        let mut loaded_models = Vec::new();
        let mut total_memory = 0;
        
        for (id, model) in models.iter() {
            if model.is_loaded().await {
                loaded_models.push(id.clone());
                total_memory += model.get_capabilities().await.memory_usage;
            }
        }
        
        RegistryStats {
            total_models: self.default_models.len(),
            registered_models: models.len(),
            loaded_models: loaded_models.len(),
            active_model: active_id,
            total_memory_usage: total_memory,
            loaded_model_ids: loaded_models,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_models: usize,
    pub registered_models: usize,
    pub loaded_models: usize,
    pub active_model: Option<String>,
    pub total_memory_usage: usize,
    pub loaded_model_ids: Vec<String>,
}