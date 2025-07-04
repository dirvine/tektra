use super::*;
use crate::inference::{
    EnhancedModelRegistry, MultimodalInput, ModelResponse, ImageData, AudioData, DocumentData,
    ImageFormat, AudioFormat, DocumentFormat, TokenEstimator, ContextUtilization
};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enhanced multimodal processor with direct inference integration
pub struct EnhancedMultimodalProcessor {
    /// Core processors for different modalities
    vision_processor: Arc<VisionProcessor>,
    audio_processor: Arc<AudioProcessor>,
    document_processor: Arc<DocumentProcessor>,
    
    /// Inference system integration
    model_registry: Arc<EnhancedModelRegistry>,
    token_estimator: TokenEstimator,
    
    /// Processing configuration
    config: ProcessingConfig,
    
    /// Statistics and metrics
    stats: Arc<RwLock<EnhancedProcessingStats>>,
    
    /// Processing cache for optimization
    processing_cache: Arc<RwLock<HashMap<String, CachedProcessing>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    /// Maximum image resolution for processing
    pub max_image_resolution: (u32, u32),
    
    /// Image quality settings
    pub image_quality: ImageQuality,
    
    /// Audio processing settings
    pub audio_sample_rate: u32,
    pub audio_chunk_size: usize,
    
    /// Document processing settings
    pub max_document_size_mb: usize,
    pub extract_text_from_images: bool,
    
    /// Context management
    pub max_context_utilization: f32,
    pub auto_chunk_large_inputs: bool,
    
    /// Vision model preferences
    pub preferred_vision_models: Vec<String>,
    
    /// Processing timeouts
    pub vision_timeout_ms: u64,
    pub audio_timeout_ms: u64,
    pub document_timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageQuality {
    Low,    // Fast processing, lower quality
    Medium, // Balanced quality and speed
    High,   // Best quality, slower processing
    Auto,   // Automatic based on content
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedProcessingStats {
    pub images_processed: usize,
    pub audio_files_processed: usize,
    pub documents_processed: usize,
    pub total_processing_time_ms: u64,
    pub average_image_processing_ms: f64,
    pub average_audio_processing_ms: f64,
    pub average_document_processing_ms: f64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub failed_processings: usize,
    pub context_overflows: usize,
}

#[derive(Debug, Clone)]
struct CachedProcessing {
    input_hash: String,
    processed_input: MultimodalInput,
    processing_time_ms: u64,
    timestamp: std::time::SystemTime,
}

impl EnhancedMultimodalProcessor {
    pub async fn new(model_registry: Arc<EnhancedModelRegistry>) -> Result<Self> {
        info!("Initializing enhanced multimodal processor");
        
        let config = ProcessingConfig::default();
        
        Ok(Self {
            vision_processor: Arc::new(VisionProcessor::new()?),
            audio_processor: Arc::new(AudioProcessor::new()?),
            document_processor: Arc::new(DocumentProcessor::new()?),
            model_registry,
            token_estimator: TokenEstimator::new(),
            config,
            stats: Arc::new(RwLock::new(EnhancedProcessingStats::default())),
            processing_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Process single image with advanced features
    pub async fn process_image_advanced(
        &self,
        image_data: &[u8],
        text_prompt: Option<String>,
        processing_options: ImageProcessingOptions,
    ) -> Result<ProcessedImageResult> {
        let start_time = std::time::Instant::now();
        
        info!("Processing image with advanced options: {:?}", processing_options);
        
        // Detect image format
        let format = self.detect_image_format(image_data)?;
        
        // Pre-process image based on quality settings
        let processed_data = self.preprocess_image(image_data, &format, &processing_options).await?;
        
        // Create image data structure
        let image = ImageData {
            data: processed_data.clone(),
            format: format.clone(),
            width: processing_options.target_width,
            height: processing_options.target_height,
        };
        
        // Create multimodal input
        let input = match text_prompt {
            Some(text) => MultimodalInput::TextWithImage { text, image },
            None => MultimodalInput::TextWithImage {
                text: "Analyze this image in detail".to_string(),
                image,
            },
        };
        
        // Check context utilization
        let active_model_id = self.model_registry.get_active_model_id().await;
        let utilization = self.check_context_utilization(&input, active_model_id.as_deref()).await?;
        
        // Generate response using inference system
        let response = self.model_registry.generate(input.clone()).await?;
        
        let processing_time = start_time.elapsed();
        
        // Update statistics
        self.update_image_stats(processing_time.as_millis() as u64).await;
        
        Ok(ProcessedImageResult {
            response,
            processing_time_ms: processing_time.as_millis() as u64,
            context_utilization: utilization,
            image_metadata: ImageMetadata {
                format,
                original_size: image_data.len(),
                processed_size: processed_data.len(),
                dimensions: (
                    processing_options.target_width.unwrap_or(0),
                    processing_options.target_height.unwrap_or(0),
                ),
            },
        })
    }
    
    /// Process image with streaming response
    pub async fn process_image_streaming(
        &self,
        image_data: &[u8],
        text_prompt: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::inference::Token>> + Send>>> {
        info!("Processing image with streaming response");
        
        let format = self.detect_image_format(image_data)?;
        let image = ImageData {
            data: image_data.to_vec(),
            format,
            width: None,
            height: None,
        };
        
        let input = MultimodalInput::TextWithImage {
            text: text_prompt,
            image,
        };
        
        self.model_registry.stream_generate(input).await
    }
    
    /// Process multiple images with comparison analysis
    pub async fn process_image_comparison(
        &self,
        images: Vec<&[u8]>,
        comparison_prompt: String,
    ) -> Result<ProcessedComparisonResult> {
        info!("Processing {} images for comparison", images.len());
        
        if images.is_empty() {
            return Err(anyhow::anyhow!("No images provided for comparison"));
        }
        
        let start_time = std::time::Instant::now();
        
        // Process all images
        let mut processed_images = Vec::new();
        for (i, image_data) in images.iter().enumerate() {
            let format = self.detect_image_format(image_data)?;
            processed_images.push(ImageData {
                data: image_data.to_vec(),
                format,
                width: None,
                height: None,
            });
            
            debug!("Processed image {} for comparison", i + 1);
        }
        
        // Create combined input
        let input = MultimodalInput::Combined {
            text: Some(comparison_prompt),
            images: processed_images,
            audio: None,
            documents: Vec::new(),
        };
        
        // Check context utilization
        let active_model_id = self.model_registry.get_active_model_id().await;
        let utilization = self.check_context_utilization(&input, active_model_id.as_deref()).await?;
        
        // Generate comparison response
        let response = self.model_registry.generate(input).await?;
        
        let processing_time = start_time.elapsed();
        
        Ok(ProcessedComparisonResult {
            response,
            num_images: images.len(),
            processing_time_ms: processing_time.as_millis() as u64,
            context_utilization: utilization,
        })
    }
    
    /// Process document with image extraction
    pub async fn process_document_with_images(
        &self,
        document_data: &[u8],
        document_format: DocumentFormat,
        extract_images: bool,
    ) -> Result<ProcessedDocumentResult> {
        info!("Processing document with image extraction: {}", extract_images);
        
        let start_time = std::time::Instant::now();
        
        // Process document text
        let document = self.document_processor.process_document_data(document_data, document_format).await?;
        
        let mut images = Vec::new();
        if extract_images {
            // Extract embedded images (if any)
            images = self.extract_images_from_document(&document).await?;
        }
        
        // Store image count before moving images
        let images_count = images.len();
        
        // Create appropriate input based on whether images were found
        let input = if images.is_empty() {
            MultimodalInput::TextWithDocument {
                text: "Analyze this document".to_string(),
                document,
            }
        } else {
            MultimodalInput::Combined {
                text: Some("Analyze this document and its embedded images".to_string()),
                images,
                audio: None,
                documents: vec![document],
            }
        };
        
        // Generate response
        let response = self.model_registry.generate(input).await?;
        
        let processing_time = start_time.elapsed();
        
        Ok(ProcessedDocumentResult {
            response,
            extracted_images_count: images_count,
            processing_time_ms: processing_time.as_millis() as u64,
        })
    }
    
    /// Process complex multimodal input with intelligent chunking
    pub async fn process_complex_multimodal(
        &self,
        complex_input: ComplexMultimodalInput,
    ) -> Result<ProcessedComplexResult> {
        info!("Processing complex multimodal input");
        
        let start_time = std::time::Instant::now();
        
        // Convert complex input to standard format
        let mut base_input = self.convert_complex_input(complex_input).await?;
        
        // Check if chunking is needed
        let active_model_id = self.model_registry.get_active_model_id().await;
        let utilization = self.check_context_utilization(&base_input, active_model_id.as_deref()).await?;
        
        let responses = if utilization.utilization_percentage > self.config.max_context_utilization {
            warn!("Context utilization too high ({}%), chunking input", utilization.utilization_percentage);
            self.process_with_chunking(base_input, active_model_id.as_deref()).await?
        } else {
            vec![self.model_registry.generate(base_input).await?]
        };
        
        let processing_time = start_time.elapsed();
        let total_chunks = responses.len();
        
        Ok(ProcessedComplexResult {
            responses,
            total_chunks,
            processing_time_ms: processing_time.as_millis() as u64,
            context_utilization: utilization,
        })
    }
    
    /// Get optimal model for multimodal input
    pub async fn get_optimal_model(&self, input: &MultimodalInput) -> Result<String> {
        let active_models = self.model_registry.list_models().await;
        
        // Filter models that support required capabilities
        let suitable_models: Vec<_> = active_models
            .into_iter()
            .filter(|model| {
                match input {
                    MultimodalInput::TextWithImage { .. } => {
                        model.supports_vision
                    }
                    MultimodalInput::Combined { images, .. } if !images.is_empty() => {
                        model.supports_vision
                    }
                    MultimodalInput::TextWithAudio { .. } |
                    MultimodalInput::Combined { audio: Some(_), .. } => {
                        model.supports_audio
                    }
                    _ => true, // Text-only inputs work with any model
                }
            })
            .collect();
        
        if suitable_models.is_empty() {
            return Err(anyhow::anyhow!("No suitable models found for input type"));
        }
        
        // Prefer vision models for image inputs
        if matches!(input, MultimodalInput::TextWithImage { .. }) {
            for preferred in &self.config.preferred_vision_models {
                if suitable_models.iter().any(|m| m.id == *preferred) {
                    return Ok(preferred.clone());
                }
            }
        }
        
        // Return first suitable model
        Ok(suitable_models[0].id.clone())
    }
    
    /// Check context utilization for input
    async fn check_context_utilization(
        &self,
        input: &MultimodalInput,
        model_id: Option<&str>,
    ) -> Result<ContextUtilization> {
        let context_window = if let Some(model_id) = model_id {
            // Get context window from model info
            if let Some(info) = self.model_registry.get_enhanced_model_info(model_id).await? {
                info.context_length.unwrap_or(8192)
            } else {
                8192
            }
        } else {
            8192
        };
        
        Ok(self.token_estimator.get_utilization(input, context_window, model_id))
    }
    
    /// Process input with automatic chunking
    async fn process_with_chunking(
        &self,
        input: MultimodalInput,
        model_id: Option<&str>,
    ) -> Result<Vec<ModelResponse>> {
        match input {
            MultimodalInput::Text(text) => {
                // Chunk text input
                let chunks = self.token_estimator.chunk_text(&text, 4000, 200, model_id);
                let mut responses = Vec::new();
                
                for chunk in chunks {
                    let chunk_input = MultimodalInput::Text(chunk);
                    responses.push(self.model_registry.generate(chunk_input).await?);
                }
                
                Ok(responses)
            }
            MultimodalInput::Combined { text, images, audio, documents } => {
                // For combined inputs, process in batches
                let mut responses = Vec::new();
                
                // Process images in smaller batches
                if !images.is_empty() {
                    for image_batch in images.chunks(2) {
                        let batch_input = MultimodalInput::Combined {
                            text: text.clone(),
                            images: image_batch.to_vec(),
                            audio: None,
                            documents: Vec::new(),
                        };
                        responses.push(self.model_registry.generate(batch_input).await?);
                    }
                }
                
                // Process other modalities separately if needed
                if audio.is_some() {
                    let audio_input = MultimodalInput::Combined {
                        text: Some("Process this audio content".to_string()),
                        images: Vec::new(),
                        audio,
                        documents: Vec::new(),
                    };
                    responses.push(self.model_registry.generate(audio_input).await?);
                }
                
                for document in documents {
                    let doc_input = MultimodalInput::TextWithDocument {
                        text: "Analyze this document".to_string(),
                        document,
                    };
                    responses.push(self.model_registry.generate(doc_input).await?);
                }
                
                Ok(responses)
            }
            _ => {
                // For other input types, process as single chunk
                Ok(vec![self.model_registry.generate(input).await?])
            }
        }
    }
    
    /// Update image processing statistics
    async fn update_image_stats(&self, processing_time_ms: u64) {
        let mut stats = self.stats.write().await;
        stats.images_processed += 1;
        stats.total_processing_time_ms += processing_time_ms;
        stats.average_image_processing_ms = 
            stats.total_processing_time_ms as f64 / stats.images_processed as f64;
    }
    
    /// Get comprehensive processing statistics
    pub async fn get_enhanced_stats(&self) -> EnhancedProcessingStats {
        self.stats.read().await.clone()
    }
    
    /// Clear processing cache
    pub async fn clear_cache(&self) {
        self.processing_cache.write().await.clear();
        info!("Processing cache cleared");
    }
    
    // Helper methods
    
    fn detect_image_format(&self, data: &[u8]) -> Result<ImageFormat> {
        if data.len() < 4 {
            return Err(anyhow::anyhow!("Insufficient data to detect image format"));
        }
        
        match &data[0..4] {
            [0xFF, 0xD8, 0xFF, _] => Ok(ImageFormat::Jpeg),
            [0x89, 0x50, 0x4E, 0x47] => Ok(ImageFormat::Png),
            [0x47, 0x49, 0x46, 0x38] => Ok(ImageFormat::Gif),
            [0x52, 0x49, 0x46, 0x46] if &data[8..12] == b"WEBP" => Ok(ImageFormat::WebP),
            [0x42, 0x4D, _, _] => Ok(ImageFormat::Bmp),
            _ => Err(anyhow::anyhow!("Unsupported image format")),
        }
    }
    
    async fn preprocess_image(
        &self,
        data: &[u8],
        format: &ImageFormat,
        options: &ImageProcessingOptions,
    ) -> Result<Vec<u8>> {
        // For now, return original data
        // In a full implementation, this would resize, compress, or enhance the image
        Ok(data.to_vec())
    }
    
    async fn extract_images_from_document(&self, document: &DocumentData) -> Result<Vec<ImageData>> {
        // Placeholder for document image extraction
        // In a full implementation, this would extract embedded images from PDFs, DOCX, etc.
        Ok(Vec::new())
    }
    
    async fn convert_complex_input(&self, complex: ComplexMultimodalInput) -> Result<MultimodalInput> {
        // Convert complex input structure to standard MultimodalInput
        Ok(MultimodalInput::Combined {
            text: complex.primary_text,
            images: complex.images,
            audio: complex.audio,
            documents: complex.documents,
        })
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            max_image_resolution: (2048, 2048),
            image_quality: ImageQuality::Medium,
            audio_sample_rate: 16000,
            audio_chunk_size: 1024,
            max_document_size_mb: 50,
            extract_text_from_images: true,
            max_context_utilization: 80.0,
            auto_chunk_large_inputs: true,
            preferred_vision_models: vec![
                "qwen2.5-vl-7b".to_string(),
                "pixtral-12b".to_string(),
                "llama-3.2-vision-11b".to_string(),
            ],
            vision_timeout_ms: 30000,
            audio_timeout_ms: 15000,
            document_timeout_ms: 10000,
        }
    }
}

impl Default for EnhancedProcessingStats {
    fn default() -> Self {
        Self {
            images_processed: 0,
            audio_files_processed: 0,
            documents_processed: 0,
            total_processing_time_ms: 0,
            average_image_processing_ms: 0.0,
            average_audio_processing_ms: 0.0,
            average_document_processing_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            failed_processings: 0,
            context_overflows: 0,
        }
    }
}

// Supporting types

use futures::Stream;
use std::pin::Pin;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingOptions {
    pub target_width: Option<u32>,
    pub target_height: Option<u32>,
    pub quality: ImageQuality,
    pub preserve_aspect_ratio: bool,
    pub enhance_contrast: bool,
    pub normalize_colors: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessedImageResult {
    pub response: ModelResponse,
    pub processing_time_ms: u64,
    pub context_utilization: ContextUtilization,
    pub image_metadata: ImageMetadata,
}

#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub format: ImageFormat,
    pub original_size: usize,
    pub processed_size: usize,
    pub dimensions: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct ProcessedComparisonResult {
    pub response: ModelResponse,
    pub num_images: usize,
    pub processing_time_ms: u64,
    pub context_utilization: ContextUtilization,
}

#[derive(Debug, Clone)]
pub struct ProcessedDocumentResult {
    pub response: ModelResponse,
    pub extracted_images_count: usize,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct ProcessedComplexResult {
    pub responses: Vec<ModelResponse>,
    pub total_chunks: usize,
    pub processing_time_ms: u64,
    pub context_utilization: ContextUtilization,
}

#[derive(Debug, Clone)]
pub struct ComplexMultimodalInput {
    pub primary_text: Option<String>,
    pub images: Vec<ImageData>,
    pub audio: Option<AudioData>,
    pub documents: Vec<DocumentData>,
    pub metadata: HashMap<String, String>,
}

impl Default for ImageProcessingOptions {
    fn default() -> Self {
        Self {
            target_width: None,
            target_height: None,
            quality: ImageQuality::Medium,
            preserve_aspect_ratio: true,
            enhance_contrast: false,
            normalize_colors: false,
        }
    }
}