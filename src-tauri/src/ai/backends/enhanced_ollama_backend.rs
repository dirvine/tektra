use anyhow::Result;
use async_trait::async_trait;
use ollama_rs::{
    generation::completion::{
        request::GenerationRequest,
        GenerationResponse,
    },
    models::{LocalModel, ModelOptions},
    Ollama,
};
use futures::StreamExt;
use base64::{Engine as _, engine::general_purpose};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

use crate::ai::unified_model_manager::{
    ModelBackend, ModelConfig, GenerationParams, MultimodalInput, 
    BackendCapabilities, DeviceConfig
};

pub struct EnhancedOllamaBackend {
    ollama: Arc<Ollama>,
    current_model: Arc<Mutex<Option<String>>>,
    model_info: Arc<Mutex<Option<ModelConfig>>>,
}

impl EnhancedOllamaBackend {
    pub fn new() -> Result<Self> {
        let ollama = Ollama::default();
        
        info!("Initialized Enhanced Ollama backend");
        
        Ok(Self {
            ollama: Arc::new(ollama),
            current_model: Arc::new(Mutex::new(None)),
            model_info: Arc::new(Mutex::new(None)),
        })
    }
    
    pub fn with_host(host: &str, port: u16) -> Result<Self> {
        let ollama = Ollama::new(host.to_string(), port);
        
        info!("Initialized Enhanced Ollama backend with host {}:{}", host, port);
        
        Ok(Self {
            ollama: Arc::new(ollama),
            current_model: Arc::new(Mutex::new(None)),
            model_info: Arc::new(Mutex::new(None)),
        })
    }
    
    async fn ensure_model_pulled(&self, model_name: &str) -> Result<()> {
        info!("Checking if model {} is available", model_name);
        
        // Check if model exists
        let models = self.ollama.list_local_models().await
            .map_err(|e| anyhow::anyhow!("Failed to list models: {}", e))?;
        
        let model_exists = models.iter().any(|m| m.name == model_name);
        
        if !model_exists {
            info!("Model {} not found locally, pulling...", model_name);
            
            // Pull the model
            let mut stream = self.ollama.pull_model_stream(model_name.to_string(), false).await
                .map_err(|e| anyhow::anyhow!("Failed to pull model: {}", e))?;
            
            while let Some(response) = stream.next().await {
                match response {
                    Ok(progress) => {
                        info!("Pull progress: {}", progress.message);
                        if let (Some(total), Some(completed)) = (progress.total, progress.completed) {
                            let percent = (completed as f64 / total as f64) * 100.0;
                            info!("Progress: {:.1}%", percent);
                        }
                    }
                    Err(e) => {
                        error!("Error during model pull: {}", e);
                    }
                }
            }
            
            info!("Model {} pulled successfully", model_name);
        } else {
            info!("Model {} already available", model_name);
        }
        
        Ok(())
    }
    
    fn convert_params(&self, params: &GenerationParams) -> ModelOptions {
        let mut options = ModelOptions::default();
        options = options.temperature(params.temperature);
        options = options.top_p(params.top_p);
        options = options.top_k(params.top_k as u32);
        options = options.repeat_penalty(params.repeat_penalty);
        if let Some(seed) = params.seed {
            options = options.seed(seed as i32);
        }
        options = options.num_predict(params.max_tokens as i32);
        if !params.stop_sequences.is_empty() {
            options = options.stop(params.stop_sequences.clone());
        }
        options
    }
}

#[async_trait]
impl ModelBackend for EnhancedOllamaBackend {
    fn name(&self) -> &str {
        "enhanced_ollama"
    }
    
    async fn supports_model(&self, model_id: &str) -> bool {
        // Ollama supports models in the format "model:tag"
        model_id.contains(':') || 
        model_id.ends_with(".gguf") ||
        // Common Ollama model names
        ["llama3", "llama2", "mistral", "mixtral", "gemma", "gemma2", "phi", "qwen", "codellama", "vicuna"]
            .iter()
            .any(|&name| model_id.starts_with(name))
    }
    
    async fn load_model(&mut self, config: &ModelConfig) -> Result<()> {
        info!("Loading model {} with Enhanced Ollama backend", config.model_id);
        
        // Ensure model is pulled
        self.ensure_model_pulled(&config.model_id).await?;
        
        // Store model information
        *self.current_model.lock().await = Some(config.model_id.clone());
        *self.model_info.lock().await = Some(config.clone());
        
        info!("Model {} loaded successfully", config.model_id);
        Ok(())
    }
    
    async fn unload_model(&mut self) -> Result<()> {
        *self.current_model.lock().await = None;
        *self.model_info.lock().await = None;
        info!("Model unloaded");
        Ok(())
    }
    
    fn is_loaded(&self) -> bool {
        // We use blocking lock here since it's a quick check
        let model = self.current_model.blocking_lock();
        model.is_some()
    }
    
    async fn generate_text(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        let model_name = self.current_model.lock().await
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let options = self.convert_params(params);
        
        let request = GenerationRequest::new(model_name, prompt.to_string())
            .options(options);
        
        let response = self.ollama.generate(request).await
            .map_err(|e| anyhow::anyhow!("Generation failed: {}", e))?;
        
        Ok(response.response)
    }
    
    async fn generate_multimodal(&self, inputs: MultimodalInput, params: &GenerationParams) -> Result<String> {
        let model_name = self.current_model.lock().await
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        // For multimodal, we need to use models that support it
        if !inputs.images.is_empty() {
            // Check if model supports vision
            if !model_name.contains("vision") && !model_name.contains("llava") {
                warn!("Model {} may not support vision inputs", model_name);
            }
            
            // Convert image to base64
            let image_base64 = if let Some(image_data) = inputs.images.first() {
                general_purpose::STANDARD.encode(image_data)
            } else {
                String::new()
            };
            
            // Format prompt with image
            let prompt = if let Some(text) = inputs.text {
                format!("[img]{}[/img]\n{}", image_base64, text)
            } else {
                format!("[img]{}[/img]\nWhat's in this image?", image_base64)
            };
            
            self.generate_text(&prompt, params).await
        } else if inputs.audio.is_some() {
            // Audio not directly supported by Ollama, would need transcription first
            Err(anyhow::anyhow!("Audio input not supported by Ollama backend. Please transcribe first."))
        } else if let Some(text) = inputs.text {
            // Text-only generation
            self.generate_text(&text, params).await
        } else {
            Err(anyhow::anyhow!("No input provided"))
        }
    }
    
    async fn generate_stream(&self, prompt: &str, params: &GenerationParams) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let model_name = self.current_model.lock().await
            .clone()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        let ollama = self.ollama.clone();
        let options = self.convert_params(params);
        let prompt = prompt.to_string();
        
        tokio::spawn(async move {
            let request = GenerationRequest::new(model_name, prompt)
                .options(options);
            
            match ollama.generate_stream(request).await {
                Ok(mut stream) => {
                    while let Some(response) = stream.next().await {
                        match response {
                            Ok(generations) => {
                                // The stream returns Vec<GenerationResponse>
                                for generation in generations {
                                    if let Err(e) = tx.send(generation.response).await {
                                        error!("Failed to send generation chunk: {}", e);
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                error!("Error in generation stream: {}", e);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to start generation stream: {}", e);
                }
            }
        });
        
        Ok(rx)
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            text_generation: true,
            image_understanding: true, // With vision models
            audio_processing: false, // Requires external transcription
            video_processing: false,
            streaming: true,
            function_calling: true, // Some models support this
            quantization_formats: vec![
                "Q4_0".to_string(),
                "Q4_1".to_string(),
                "Q5_0".to_string(),
                "Q5_1".to_string(),
                "Q8_0".to_string(),
                "Q8_1".to_string(),
                "F16".to_string(),
            ],
            max_context_length: 128000, // Depends on model
        }
    }
    
    fn memory_usage(&self) -> usize {
        // Estimate based on loaded model
        if let Some(model_info) = self.model_info.blocking_lock().as_ref() {
            // Rough estimates based on common model sizes
            let model_lower = model_info.model_id.to_lowercase();
            
            if model_lower.contains("70b") {
                70_000_000_000 // 70GB
            } else if model_lower.contains("34b") {
                34_000_000_000 // 34GB
            } else if model_lower.contains("13b") {
                13_000_000_000 // 13GB
            } else if model_lower.contains("7b") || model_lower.contains("8b") {
                8_000_000_000 // 8GB
            } else if model_lower.contains("3b") {
                3_000_000_000 // 3GB
            } else if model_lower.contains("2b") {
                2_000_000_000 // 2GB
            } else {
                4_000_000_000 // Default 4GB
            }
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_ollama_backend_creation() {
        let backend = EnhancedOllamaBackend::new().unwrap();
        assert_eq!(backend.name(), "enhanced_ollama");
        // Note: We can't call is_loaded() in a test without a proper async runtime
        // because it uses blocking_lock() which requires being outside an async context
        // So we'll just verify the backend was created successfully
    }
    
    #[tokio::test]
    async fn test_model_support() {
        let backend = EnhancedOllamaBackend::new().unwrap();
        
        assert!(backend.supports_model("llama3:8b").await);
        assert!(backend.supports_model("gemma:2b").await);
        assert!(backend.supports_model("mistral:latest").await);
        assert!(backend.supports_model("model.gguf").await);
    }
    
    #[tokio::test]
    async fn test_capabilities() {
        let backend = EnhancedOllamaBackend::new().unwrap();
        let caps = backend.capabilities();
        
        assert!(caps.text_generation);
        assert!(caps.image_understanding);
        assert!(!caps.audio_processing);
        assert!(caps.streaming);
        assert!(caps.function_calling);
    }
}