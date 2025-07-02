use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::info;

// Use backend implementations
use crate::ai::backends::EnhancedOllamaBackend;

// Core types for multimodal inputs
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub text: Option<String>,
    pub images: Vec<Vec<u8>>,
    pub audio: Option<Vec<u8>>,
    pub video: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<String>,
    pub stream: bool,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: None,
            stop_sequences: vec![],
            stream: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub model_path: Option<String>,
    pub context_length: usize,
    pub quantization: Option<String>,
    pub device: DeviceConfig,
    pub rope_scale: Option<f32>,
    pub template_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),    // GPU index
    Metal,          // Apple Silicon
    Auto,           // Auto-detect
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub text_generation: bool,
    pub image_understanding: bool,
    pub audio_processing: bool,
    pub video_processing: bool,
    pub streaming: bool,
    pub function_calling: bool,
    pub quantization_formats: Vec<String>,
    pub max_context_length: usize,
}

// Core trait for model backends
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Get the name of this backend
    fn name(&self) -> &str;
    
    /// Check if this backend supports a specific model
    async fn supports_model(&self, model_id: &str) -> bool;
    
    /// Load a model with the given configuration
    async fn load_model(&mut self, config: &ModelConfig) -> Result<()>;
    
    /// Unload the currently loaded model
    async fn unload_model(&mut self) -> Result<()>;
    
    /// Check if a model is currently loaded
    fn is_loaded(&self) -> bool;
    
    /// Generate text from a prompt
    async fn generate_text(&self, prompt: &str, params: &GenerationParams) -> Result<String>;
    
    /// Generate from multimodal inputs
    async fn generate_multimodal(&self, inputs: MultimodalInput, params: &GenerationParams) -> Result<String>;
    
    /// Stream generation (returns a channel receiver)
    async fn generate_stream(&self, prompt: &str, params: &GenerationParams) -> Result<tokio::sync::mpsc::Receiver<String>>;
    
    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;
    
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

// Backend registry and factory
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn Fn() -> Result<Box<dyn ModelBackend>> + Send + Sync>>,
}

impl BackendRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            backends: HashMap::new(),
        };
        
        // Register default backends
        registry.register("enhanced_ollama", Box::new(|| {
            Ok(Box::new(EnhancedOllamaBackend::new()?))
        }));
        
        registry
    }
    
    pub fn register(&mut self, name: &str, factory: Box<dyn Fn() -> Result<Box<dyn ModelBackend>> + Send + Sync>) {
        self.backends.insert(name.to_string(), factory);
    }
    
    pub fn create(&self, name: &str) -> Result<Box<dyn ModelBackend>> {
        let factory = self.backends.get(name)
            .ok_or_else(|| anyhow::anyhow!("Backend '{}' not registered", name))?;
        factory()
    }
    
    pub fn list_backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }
}

// Main unified model manager
pub struct UnifiedModelManager {
    registry: Arc<BackendRegistry>,
    backends: Arc<RwLock<HashMap<String, Box<dyn ModelBackend>>>>,
    active_backend: Arc<RwLock<Option<String>>>,
    model_config: Arc<RwLock<Option<ModelConfig>>>,
    backend_preferences: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl UnifiedModelManager {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(BackendRegistry::new()),
            backends: Arc::new(RwLock::new(HashMap::new())),
            active_backend: Arc::new(RwLock::new(None)),
            model_config: Arc::new(RwLock::new(None)),
            backend_preferences: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Load backend preferences from configuration
    pub async fn load_preferences(&self, preferences: HashMap<String, Vec<String>>) -> Result<()> {
        *self.backend_preferences.write().await = preferences;
        Ok(())
    }
    
    /// Initialize a backend if not already initialized
    async fn ensure_backend(&self, backend_name: &str) -> Result<()> {
        let backends = self.backends.read().await;
        if !backends.contains_key(backend_name) {
            drop(backends);
            
            info!("Initializing backend: {}", backend_name);
            let backend = self.registry.create(backend_name)?;
            
            let mut backends = self.backends.write().await;
            backends.insert(backend_name.to_string(), backend);
        }
        Ok(())
    }
    
    /// Select the best backend for a model
    async fn select_backend_for_model(&self, model_id: &str) -> Result<String> {
        let preferences = self.backend_preferences.read().await;
        
        // Check if we have specific preferences for this model
        if let Some(backend_list) = preferences.get(model_id) {
            for backend_name in backend_list {
                self.ensure_backend(backend_name).await?;
                
                let backends = self.backends.read().await;
                if let Some(backend) = backends.get(backend_name) {
                    if backend.supports_model(model_id).await {
                        info!("Selected backend '{}' for model '{}'", backend_name, model_id);
                        return Ok(backend_name.clone());
                    }
                }
            }
        }
        
        // Otherwise, try all backends in order
        let backend_names = self.registry.list_backends();
        for backend_name in &backend_names {
            self.ensure_backend(backend_name).await?;
            
            let backends = self.backends.read().await;
            if let Some(backend) = backends.get(backend_name) {
                if backend.supports_model(model_id).await {
                    info!("Auto-selected backend '{}' for model '{}'", backend_name, model_id);
                    return Ok(backend_name.clone());
                }
            }
        }
        
        Err(anyhow::anyhow!("No backend supports model '{}'", model_id))
    }
    
    /// Load a model with automatic backend selection
    pub async fn load_model(&self, config: ModelConfig) -> Result<()> {
        let model_id = config.model_id.clone();
        
        // Select appropriate backend
        let backend_name = self.select_backend_for_model(&model_id).await?;
        
        // Load the model
        let mut backends = self.backends.write().await;
        if let Some(backend) = backends.get_mut(&backend_name) {
            info!("Loading model '{}' with backend '{}'", model_id, backend_name);
            backend.load_model(&config).await?;
            
            drop(backends);
            
            // Update state
            *self.active_backend.write().await = Some(backend_name);
            *self.model_config.write().await = Some(config);
            
            info!("Model '{}' loaded successfully", model_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Backend '{}' not found", backend_name))
        }
    }
    
    /// Unload the current model
    pub async fn unload_model(&self) -> Result<()> {
        if let Some(backend_name) = self.active_backend.read().await.clone() {
            let mut backends = self.backends.write().await;
            if let Some(backend) = backends.get_mut(&backend_name) {
                backend.unload_model().await?;
            }
        }
        
        *self.active_backend.write().await = None;
        *self.model_config.write().await = None;
        
        Ok(())
    }
    
    /// Check if a model is loaded
    pub async fn is_loaded(&self) -> bool {
        if let Some(backend_name) = self.active_backend.read().await.as_ref() {
            let backends = self.backends.read().await;
            if let Some(backend) = backends.get(backend_name) {
                return backend.is_loaded();
            }
        }
        false
    }
    
    /// Generate text using the active backend
    pub async fn generate_text(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let backend_name = self.active_backend.read().await.clone()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let backends = self.backends.read().await;
        let backend = backends.get(&backend_name)
            .ok_or_else(|| anyhow::anyhow!("Backend not found"))?;
        
        backend.generate_text(prompt, params).await
    }
    
    /// Generate from multimodal inputs
    pub async fn generate_multimodal(&self, inputs: MultimodalInput, params: &GenerationParams) -> Result<String> {
        let backend_name = self.active_backend.read().await.clone()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let backends = self.backends.read().await;
        let backend = backends.get(&backend_name)
            .ok_or_else(|| anyhow::anyhow!("Backend not found"))?;
        
        backend.generate_multimodal(inputs, params).await
    }
    
    /// Stream generation
    pub async fn generate_stream(&self, prompt: &str, params: &GenerationParams) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let backend_name = self.active_backend.read().await.clone()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let backends = self.backends.read().await;
        let backend = backends.get(&backend_name)
            .ok_or_else(|| anyhow::anyhow!("Backend not found"))?;
        
        backend.generate_stream(prompt, params).await
    }
    
    /// Get capabilities of the active backend
    pub async fn get_capabilities(&self) -> Option<BackendCapabilities> {
        if let Some(backend_name) = self.active_backend.read().await.as_ref() {
            let backends = self.backends.read().await;
            if let Some(backend) = backends.get(backend_name) {
                return Some(backend.capabilities());
            }
        }
        None
    }
    
    /// Get current backend name
    pub async fn current_backend(&self) -> Option<String> {
        self.active_backend.read().await.clone()
    }
    
    /// Get current model configuration
    pub async fn current_model_config(&self) -> Option<ModelConfig> {
        self.model_config.read().await.clone()
    }
    
    /// Get memory usage across all backends
    pub async fn total_memory_usage(&self) -> usize {
        let backends = self.backends.read().await;
        backends.values().map(|b| b.memory_usage()).sum()
    }
    
    /// Switch to a different backend for the same model
    pub async fn switch_backend(&self, new_backend: &str) -> Result<()> {
        let config = self.model_config.read().await.clone()
            .ok_or_else(|| anyhow::anyhow!("No model configuration available"))?;
        
        // Unload from current backend
        self.unload_model().await?;
        
        // Ensure new backend exists
        self.ensure_backend(new_backend).await?;
        
        // Load in new backend
        let mut backends = self.backends.write().await;
        if let Some(backend) = backends.get_mut(new_backend) {
            if !backend.supports_model(&config.model_id).await {
                return Err(anyhow::anyhow!("Backend '{}' does not support model '{}'", new_backend, config.model_id));
            }
            
            backend.load_model(&config).await?;
            drop(backends);
            
            *self.active_backend.write().await = Some(new_backend.to_string());
            
            info!("Switched to backend '{}' for model '{}'", new_backend, config.model_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Backend '{}' not found", new_backend))
        }
    }
}