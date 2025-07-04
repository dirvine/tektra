use super::*;
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Unified interface for all AI model backends
pub struct ModelAbstraction {
    inner: Arc<RwLock<Box<dyn MultimodalModel>>>,
    model_info: Arc<RwLock<Option<ModelInfo>>>,
}

impl ModelAbstraction {
    pub fn new(model: Box<dyn MultimodalModel>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(model)),
            model_info: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Load model with configuration
    pub async fn load_model(&self, config: &ModelConfig) -> Result<()> {
        info!("Loading model: {}", config.model_id);
        
        let mut model = self.inner.write().await;
        model.load(config).await?;
        
        // Cache model info after successful load
        let info = model.model_info();
        *self.model_info.write().await = Some(info);
        
        info!("Model loaded successfully: {}", config.model_id);
        Ok(())
    }
    
    /// Unload current model
    pub async fn unload_model(&self) -> Result<()> {
        info!("Unloading model");
        
        let mut model = self.inner.write().await;
        model.unload().await?;
        
        *self.model_info.write().await = None;
        
        info!("Model unloaded successfully");
        Ok(())
    }
    
    /// Check if model is ready
    pub async fn is_loaded(&self) -> bool {
        let model = self.inner.read().await;
        model.is_loaded()
    }
    
    /// Generate response for text input
    pub async fn generate_text(&self, text: &str) -> Result<String> {
        let input = MultimodalInput::Text(text.to_string());
        let response = self.generate(input).await?;
        Ok(response.text)
    }
    
    /// Generate response for text + image input
    pub async fn generate_text_with_image(&self, text: &str, image_data: &[u8]) -> Result<String> {
        let image = ImageData {
            data: image_data.to_vec(),
            format: ImageFormat::Png, // Default to PNG, could be auto-detected
            width: None,
            height: None,
        };
        
        let input = MultimodalInput::TextWithImage {
            text: text.to_string(),
            image,
        };
        
        let response = self.generate(input).await?;
        Ok(response.text)
    }
    
    /// Generate response for any multimodal input
    pub async fn generate(&self, input: MultimodalInput) -> Result<ModelResponse> {
        let model = self.inner.read().await;
        
        if !model.is_loaded() {
            return Err(InferenceError::ModelNotLoaded.into());
        }
        
        // Validate input compatibility with model capabilities
        match &input {
            MultimodalInput::TextWithImage { .. } => {
                if !model.supports_vision() {
                    return Err(InferenceError::UnsupportedInput(
                        "Model does not support vision".to_string()
                    ).into());
                }
            }
            MultimodalInput::Combined { images, .. } if !images.is_empty() => {
                if !model.supports_vision() {
                    return Err(InferenceError::UnsupportedInput(
                        "Model does not support vision".to_string()
                    ).into());
                }
            }
            MultimodalInput::TextWithAudio { .. } | MultimodalInput::Combined { audio: Some(_), .. } => {
                if !model.supports_audio() {
                    return Err(InferenceError::UnsupportedInput(
                        "Model does not support audio".to_string()
                    ).into());
                }
            }
            MultimodalInput::TextWithDocument { .. } => {
                if !model.supports_documents() {
                    return Err(InferenceError::UnsupportedInput(
                        "Model does not support document processing".to_string()
                    ).into());
                }
            }
            MultimodalInput::Combined { documents, .. } if !documents.is_empty() => {
                if !model.supports_documents() {
                    return Err(InferenceError::UnsupportedInput(
                        "Model does not support document processing".to_string()
                    ).into());
                }
            }
            _ => {} // Text-only inputs are always supported
        }
        
        model.generate(input).await
    }
    
    /// Generate streaming response
    pub async fn stream_generate(&self, input: MultimodalInput) 
        -> Result<Pin<Box<dyn Stream<Item = Result<Token>> + Send>>> {
        let model = self.inner.read().await;
        
        if !model.is_loaded() {
            return Err(InferenceError::ModelNotLoaded.into());
        }
        
        model.stream_generate(input).await
    }
    
    /// Get model information
    pub async fn get_model_info(&self) -> Option<ModelInfo> {
        self.model_info.read().await.clone()
    }
    
    /// Get model capabilities summary
    pub async fn get_capabilities(&self) -> ModelCapabilities {
        let model = self.inner.read().await;
        
        ModelCapabilities {
            supports_vision: model.supports_vision(),
            supports_audio: model.supports_audio(),
            supports_documents: model.supports_documents(),
            context_window: model.context_window(),
            memory_usage: model.memory_usage(),
        }
    }
    
    /// Switch to a different model
    pub async fn switch_model(&self, new_model: Box<dyn MultimodalModel>, config: &ModelConfig) -> Result<()> {
        info!("Switching to model: {}", config.model_id);
        
        // Unload current model first
        {
            let mut current_model = self.inner.write().await;
            if current_model.is_loaded() {
                if let Err(e) = current_model.unload().await {
                    warn!("Error unloading current model: {}", e);
                }
            }
        }
        
        // Replace with new model
        *self.inner.write().await = new_model;
        
        // Load the new model
        self.load_model(config).await?;
        
        info!("Model switch completed successfully");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_documents: bool,
    pub context_window: usize,
    pub memory_usage: usize,
}

/// Helper function to detect image format from bytes
pub fn detect_image_format(data: &[u8]) -> Option<ImageFormat> {
    if data.len() < 4 {
        return None;
    }
    
    match &data[0..4] {
        [0xFF, 0xD8, 0xFF, _] => Some(ImageFormat::Jpeg),
        [0x89, 0x50, 0x4E, 0x47] => Some(ImageFormat::Png),
        [0x47, 0x49, 0x46, 0x38] => Some(ImageFormat::Gif),
        [0x52, 0x49, 0x46, 0x46] if &data[8..12] == b"WEBP" => Some(ImageFormat::WebP),
        [0x42, 0x4D, _, _] => Some(ImageFormat::Bmp),
        _ => None,
    }
}

/// Helper function to detect audio format from bytes
pub fn detect_audio_format(data: &[u8]) -> Option<AudioFormat> {
    if data.len() < 4 {
        return None;
    }
    
    match &data[0..4] {
        [0x52, 0x49, 0x46, 0x46] if &data[8..12] == b"WAVE" => Some(AudioFormat::Wav),
        [0xFF, 0xFB, _, _] | [0xFF, 0xFA, _, _] | [0xFF, 0xF3, _, _] | [0xFF, 0xF2, _, _] => Some(AudioFormat::Mp3),
        [0x66, 0x4C, 0x61, 0x43] => Some(AudioFormat::Flac),
        [0x4F, 0x67, 0x67, 0x53] => Some(AudioFormat::Ogg),
        _ => None,
    }
}

/// Helper function to detect document format from filename or bytes
pub fn detect_document_format(filename: Option<&str>, data: &[u8]) -> Option<DocumentFormat> {
    // Try filename extension first
    if let Some(name) = filename {
        if name.ends_with(".pdf") {
            return Some(DocumentFormat::Pdf);
        } else if name.ends_with(".docx") {
            return Some(DocumentFormat::Docx);
        } else if name.ends_with(".txt") {
            return Some(DocumentFormat::Txt);
        } else if name.ends_with(".md") || name.ends_with(".markdown") {
            return Some(DocumentFormat::Markdown);
        } else if name.ends_with(".json") {
            return Some(DocumentFormat::Json);
        }
    }
    
    // Try magic bytes detection
    if data.len() >= 4 {
        match &data[0..4] {
            [0x25, 0x50, 0x44, 0x46] => return Some(DocumentFormat::Pdf), // %PDF
            [0x50, 0x4B, 0x03, 0x04] => return Some(DocumentFormat::Docx), // ZIP-based formats
            _ => {}
        }
    }
    
    // Try UTF-8 text detection for plain text formats
    if let Ok(text) = std::str::from_utf8(data) {
        if text.trim_start().starts_with('{') || text.trim_start().starts_with('[') {
            return Some(DocumentFormat::Json);
        } else if text.contains("# ") || text.contains("## ") || text.contains("```") {
            return Some(DocumentFormat::Markdown);
        } else {
            return Some(DocumentFormat::Txt);
        }
    }
    
    None
}