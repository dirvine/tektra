use super::*;
use crate::inference::{
    EnhancedModelRegistry, MultimodalInput, ModelResponse, ImageData, ImageFormat,
    TokenEstimator, ContextUtilization
};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use image::{DynamicImage, ImageFormat as ImgFormat, GenericImageView, ImageBuffer, Pixel};
use futures::Stream;
use std::pin::Pin;

/// Enhanced vision processor with advanced image analysis capabilities
pub struct EnhancedVisionProcessor {
    model_registry: Arc<EnhancedModelRegistry>,
    token_estimator: TokenEstimator,
    config: VisionConfig,
    stats: Arc<RwLock<VisionStats>>,
    image_cache: Arc<RwLock<HashMap<String, CachedImage>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionConfig {
    /// Maximum image dimensions for processing
    pub max_width: u32,
    pub max_height: u32,
    
    /// Image preprocessing options
    pub auto_resize: bool,
    pub auto_enhance: bool,
    pub auto_normalize: bool,
    
    /// Quality settings
    pub jpeg_quality: u8,
    pub png_compression: u8,
    
    /// Analysis preferences
    pub default_analysis_prompt: String,
    pub detailed_analysis_prompt: String,
    pub comparison_prompt: String,
    
    /// Performance settings
    pub enable_caching: bool,
    pub cache_max_size: usize,
    pub processing_timeout_ms: u64,
    
    /// Model preferences for different tasks
    pub preferred_models: VisionModelPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionModelPreferences {
    pub general_analysis: Vec<String>,
    pub detailed_analysis: Vec<String>,
    pub comparison: Vec<String>,
    pub ocr: Vec<String>,
    pub scene_understanding: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionStats {
    pub images_processed: u64,
    pub total_processing_time_ms: u64,
    pub average_processing_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub preprocessing_time_ms: u64,
    pub inference_time_ms: u64,
    pub failed_processings: u64,
    pub by_format: HashMap<String, u64>,
    pub by_resolution: HashMap<String, u64>,
}

#[derive(Debug, Clone)]
struct CachedImage {
    processed_data: Vec<u8>,
    metadata: EnhancedImageMetadata,
    timestamp: std::time::SystemTime,
    access_count: u32,
}

impl EnhancedVisionProcessor {
    pub async fn new(model_registry: Arc<EnhancedModelRegistry>) -> Result<Self> {
        info!("Initializing enhanced vision processor");
        
        Ok(Self {
            model_registry,
            token_estimator: TokenEstimator::new(),
            config: VisionConfig::default(),
            stats: Arc::new(RwLock::new(VisionStats::default())),
            image_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Analyze single image with customizable analysis type
    pub async fn analyze_image(
        &self,
        image_data: &[u8],
        analysis_type: ImageAnalysisType,
        custom_prompt: Option<String>,
    ) -> Result<ImageAnalysisResult> {
        let start_time = std::time::Instant::now();
        info!("Analyzing image with type: {:?}", analysis_type);
        
        // Detect and validate image format
        let format = self.detect_image_format(image_data)?;
        let image_hash = self.calculate_image_hash(image_data);
        
        // Check cache first
        if self.config.enable_caching {
            if let Some(cached) = self.get_cached_result(&image_hash, &analysis_type).await {
                self.update_cache_stats(true).await;
                return Ok(cached);
            }
            self.update_cache_stats(false).await;
        }
        
        // Preprocess image
        let preprocessed = self.preprocess_image(image_data, &format, &analysis_type).await?;
        let preprocessing_time = start_time.elapsed();
        
        // Create image data structure (clone to avoid ownership issues)
        let processed_image = ImageData {
            data: preprocessed.data.clone(),
            format: format.clone(),
            width: Some(preprocessed.width),
            height: Some(preprocessed.height),
        };
        
        // Generate analysis prompt
        let prompt = custom_prompt.unwrap_or_else(|| self.get_analysis_prompt(&analysis_type));
        
        // Create multimodal input
        let input = MultimodalInput::TextWithImage {
            text: prompt.clone(),
            image: processed_image,
        };
        
        // Check context utilization
        let active_model_id = self.model_registry.get_active_model_id().await;
        let utilization = self.check_context_utilization(&input, active_model_id.as_deref()).await?;
        
        // Select optimal model for analysis type
        let optimal_model = self.select_optimal_model(&analysis_type).await?;
        if let Some(current_model) = active_model_id {
            if current_model != optimal_model {
                info!("Switching to optimal model for analysis: {}", optimal_model);
                self.model_registry.switch_model(&optimal_model).await?;
            }
        }
        
        // Generate analysis
        let inference_start = std::time::Instant::now();
        let response = self.model_registry.generate(input).await?;
        let inference_time = inference_start.elapsed();
        
        let total_time = start_time.elapsed();
        
        // Extract analysis components
        let analysis = self.parse_analysis_response(&response, &analysis_type)?;
        
        let result = ImageAnalysisResult {
            analysis,
            confidence_score: self.calculate_confidence(&response),
            processing_time_ms: total_time.as_millis() as u64,
            preprocessing_time_ms: preprocessing_time.as_millis() as u64,
            inference_time_ms: inference_time.as_millis() as u64,
            context_utilization: utilization,
            model_used: optimal_model.clone(),
            image_metadata: EnhancedImageMetadata {
                original_format: format,
                original_size: image_data.len(),
                processed_size: preprocessed.data.len(),
                original_dimensions: (preprocessed.original_width, preprocessed.original_height),
                processed_dimensions: (preprocessed.width, preprocessed.height),
                color_space: preprocessed.color_space,
                compression_ratio: image_data.len() as f32 / preprocessed.data.len() as f32,
            },
        };
        
        // Cache result if enabled
        if self.config.enable_caching {
            self.cache_result(&image_hash, &analysis_type, &result).await;
        }
        
        // Update statistics
        self.update_stats(&result, &format).await;
        
        Ok(result)
    }
    
    /// Analyze multiple images for comparison
    pub async fn compare_images(
        &self,
        images: Vec<&[u8]>,
        comparison_type: ComparisonType,
        custom_prompt: Option<String>,
    ) -> Result<ImageComparisonResult> {
        if images.len() < 2 {
            return Err(anyhow::anyhow!("At least 2 images required for comparison"));
        }
        
        let start_time = std::time::Instant::now();
        info!("Comparing {} images with type: {:?}", images.len(), comparison_type);
        
        // Process all images
        let mut processed_images = Vec::new();
        for (i, image_data) in images.iter().enumerate() {
            let format = self.detect_image_format(image_data)?;
            let preprocessed = self.preprocess_image(
                image_data, 
                &format, 
                &ImageAnalysisType::General
            ).await?;
            
            processed_images.push(ImageData {
                data: preprocessed.data,
                format,
                width: Some(preprocessed.width),
                height: Some(preprocessed.height),
            });
            
            debug!("Processed image {} for comparison", i + 1);
        }
        
        // Generate comparison prompt
        let prompt = custom_prompt.unwrap_or_else(|| self.get_comparison_prompt(&comparison_type, images.len()));
        
        // Create combined input
        let input = MultimodalInput::Combined {
            text: Some(prompt),
            images: processed_images,
            audio: None,
            documents: Vec::new(),
        };
        
        // Generate comparison analysis
        let response = self.model_registry.generate(input).await?;
        
        let total_time = start_time.elapsed();
        
        // Parse comparison results
        let comparison = self.parse_comparison_response(&response, &comparison_type)?;
        
        Ok(ImageComparisonResult {
            comparison,
            num_images: images.len(),
            processing_time_ms: total_time.as_millis() as u64,
            confidence_score: self.calculate_confidence(&response),
            model_used: self.model_registry.get_active_model_id().await.unwrap_or_default(),
        })
    }
    
    /// Extract text from image using OCR analysis
    pub async fn extract_text(
        &self,
        image_data: &[u8],
        ocr_options: OCROptions,
    ) -> Result<TextExtractionResult> {
        info!("Extracting text from image with OCR");
        
        let start_time = std::time::Instant::now();
        
        // Preprocess for OCR (different preprocessing than general analysis)
        let format = self.detect_image_format(image_data)?;
        let preprocessed = self.preprocess_for_ocr(image_data, &format, &ocr_options).await?;
        
        let image = ImageData {
            data: preprocessed.data,
            format,
            width: Some(preprocessed.width),
            height: Some(preprocessed.height),
        };
        
        // Create OCR-specific prompt
        let prompt = self.create_ocr_prompt(&ocr_options);
        
        let input = MultimodalInput::TextWithImage {
            text: prompt,
            image,
        };
        
        // Use OCR-optimized model if available
        let ocr_model = self.select_ocr_model().await?;
        let current_model = self.model_registry.get_active_model_id().await;
        if current_model.as_deref() != Some(&ocr_model) {
            self.model_registry.switch_model(&ocr_model).await?;
        }
        
        let response = self.model_registry.generate(input).await?;
        
        let processing_time = start_time.elapsed();
        
        // Parse extracted text
        let extraction = self.parse_ocr_response(&response, &ocr_options)?;
        
        Ok(TextExtractionResult {
            extracted_text: extraction.text,
            text_regions: extraction.regions,
            confidence_score: self.calculate_confidence(&response),
            processing_time_ms: processing_time.as_millis() as u64,
            language_detected: extraction.language,
            model_used: ocr_model,
        })
    }
    
    /// Analyze image composition and visual elements
    pub async fn analyze_composition(
        &self,
        image_data: &[u8],
        composition_aspects: Vec<CompositionAspect>,
    ) -> Result<CompositionAnalysisResult> {
        info!("Analyzing image composition for {:?}", composition_aspects);
        
        let analysis_result = self.analyze_image(
            image_data,
            ImageAnalysisType::Detailed,
            Some(self.create_composition_prompt(&composition_aspects)),
        ).await?;
        
        // Parse composition-specific information
        let composition = self.parse_composition_analysis(&analysis_result.analysis, &composition_aspects)?;
        
        Ok(CompositionAnalysisResult {
            composition,
            overall_score: composition.overall_quality_score,
            suggestions: composition.improvement_suggestions.clone(),
            processing_time_ms: analysis_result.processing_time_ms,
            model_used: analysis_result.model_used,
        })
    }
    
    /// Stream analysis for real-time processing
    pub async fn stream_analysis(
        &self,
        image_data: &[u8],
        prompt: String,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<crate::inference::Token>> + Send>>> {
        info!("Starting streaming analysis for image");
        
        let format = self.detect_image_format(image_data)?;
        let preprocessed = self.preprocess_image(
            image_data, 
            &format, 
            &ImageAnalysisType::General
        ).await?;
        
        let image = ImageData {
            data: preprocessed.data,
            format,
            width: Some(preprocessed.width),
            height: Some(preprocessed.height),
        };
        
        let input = MultimodalInput::TextWithImage {
            text: prompt,
            image,
        };
        
        self.model_registry.stream_generate(input).await
    }
    
    /// Get processing statistics
    pub async fn get_stats(&self) -> VisionStats {
        self.stats.read().await.clone()
    }
    
    /// Clear image cache
    pub async fn clear_cache(&self) {
        self.image_cache.write().await.clear();
        info!("Vision processor cache cleared");
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
            [0x52, 0x49, 0x46, 0x46] if data.len() >= 12 && &data[8..12] == b"WEBP" => Ok(ImageFormat::WebP),
            [0x42, 0x4D, _, _] => Ok(ImageFormat::Bmp),
            _ => Err(anyhow::anyhow!("Unsupported image format")),
        }
    }
    
    fn calculate_image_hash(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    async fn preprocess_image(
        &self,
        data: &[u8],
        format: &ImageFormat,
        analysis_type: &ImageAnalysisType,
    ) -> Result<PreprocessedImage> {
        // Load image using the image crate
        let img = image::load_from_memory(data)
            .map_err(|e| anyhow::anyhow!("Failed to load image: {}", e))?;
        
        let (original_width, original_height) = img.dimensions();
        
        // Apply preprocessing based on analysis type and config
        let processed_img = match analysis_type {
            ImageAnalysisType::OCR => self.enhance_for_ocr(img),
            ImageAnalysisType::Detailed => self.enhance_for_detail(img),
            _ => self.standard_preprocessing(img),
        };
        
        // Resize if needed
        let final_img = if original_width > self.config.max_width || original_height > self.config.max_height {
            processed_img.resize(
                self.config.max_width,
                self.config.max_height,
                image::imageops::FilterType::Lanczos3,
            )
        } else {
            processed_img
        };
        
        let (width, height) = final_img.dimensions();
        
        // Convert back to bytes
        let mut output = Vec::new();
        let output_format = match format {
            ImageFormat::Png => ImgFormat::Png,
            ImageFormat::Jpeg => ImgFormat::Jpeg,
            ImageFormat::Gif => ImgFormat::Png, // Convert GIF to PNG for processing
            ImageFormat::WebP => ImgFormat::Png, // Convert WebP to PNG for processing
            ImageFormat::Bmp => ImgFormat::Png, // Convert BMP to PNG for processing
        };
        
        final_img.write_to(&mut std::io::Cursor::new(&mut output), output_format)
            .map_err(|e| anyhow::anyhow!("Failed to encode processed image: {}", e))?;
        
        Ok(PreprocessedImage {
            data: output,
            width,
            height,
            original_width,
            original_height,
            color_space: "RGB".to_string(), // Simplified for now
        })
    }
    
    fn standard_preprocessing(&self, img: DynamicImage) -> DynamicImage {
        if self.config.auto_enhance {
            // Apply basic enhancements
            let enhanced = imageproc::contrast::stretch_contrast(&img.to_luma8(), 5, 250, 5, 250);
            DynamicImage::ImageLuma8(enhanced)
        } else {
            img
        }
    }
    
    fn enhance_for_ocr(&self, img: DynamicImage) -> DynamicImage {
        // Convert to grayscale and enhance contrast for better OCR
        let gray = img.to_luma8();
        let enhanced = imageproc::contrast::stretch_contrast(&gray, 0, 255, 0, 255);
        DynamicImage::ImageLuma8(enhanced)
    }
    
    fn enhance_for_detail(&self, img: DynamicImage) -> DynamicImage {
        // Apply sharpening for detailed analysis
        if self.config.auto_enhance {
            // Apply unsharp mask for detail enhancement
            img // Simplified - would apply actual sharpening
        } else {
            img
        }
    }
    
    async fn preprocess_for_ocr(
        &self,
        data: &[u8],
        format: &ImageFormat,
        options: &OCROptions,
    ) -> Result<PreprocessedImage> {
        // Specialized preprocessing for OCR
        self.preprocess_image(data, format, &ImageAnalysisType::OCR).await
    }
    
    fn get_analysis_prompt(&self, analysis_type: &ImageAnalysisType) -> String {
        match analysis_type {
            ImageAnalysisType::General => self.config.default_analysis_prompt.clone(),
            ImageAnalysisType::Detailed => self.config.detailed_analysis_prompt.clone(),
            ImageAnalysisType::OCR => "Extract all visible text from this image, maintaining the original structure and formatting as much as possible.".to_string(),
            ImageAnalysisType::Scene => "Describe the scene in this image, including the setting, objects, people, activities, and overall atmosphere.".to_string(),
            ImageAnalysisType::Technical => "Provide a technical analysis of this image, including composition, lighting, color theory, and photographic techniques used.".to_string(),
        }
    }
    
    fn get_comparison_prompt(&self, comparison_type: &ComparisonType, num_images: usize) -> String {
        match comparison_type {
            ComparisonType::Similarity => format!("Compare these {} images and describe their similarities and differences in detail.", num_images),
            ComparisonType::Quality => format!("Analyze the quality of these {} images and rank them from best to worst, explaining your reasoning.", num_images),
            ComparisonType::Content => format!("Compare the content of these {} images, focusing on objects, people, scenes, and activities.", num_images),
            ComparisonType::Style => format!("Compare the artistic style and visual characteristics of these {} images.", num_images),
        }
    }
    
    fn create_ocr_prompt(&self, options: &OCROptions) -> String {
        let mut prompt = "Extract all visible text from this image".to_string();
        
        if options.preserve_formatting {
            prompt.push_str(", maintaining the original formatting and structure");
        }
        
        if let Some(ref language) = options.target_language {
            prompt.push_str(&format!(", focusing on {} text", language));
        }
        
        if options.include_confidence {
            prompt.push_str(", and indicate confidence levels for each text segment");
        }
        
        prompt.push('.');
        prompt
    }
    
    fn create_composition_prompt(&self, aspects: &[CompositionAspect]) -> String {
        let aspect_descriptions: Vec<String> = aspects.iter().map(|aspect| {
            match aspect {
                CompositionAspect::RuleOfThirds => "rule of thirds application",
                CompositionAspect::Balance => "visual balance and weight distribution",
                CompositionAspect::Leading => "leading lines and directional elements",
                CompositionAspect::Symmetry => "symmetry and patterns",
                CompositionAspect::Color => "color harmony and contrast",
                CompositionAspect::Lighting => "lighting quality and direction",
                CompositionAspect::Depth => "depth of field and layering",
                CompositionAspect::Focus => "focal points and emphasis",
            }.to_string()
        }).collect();
        
        format!(
            "Analyze the composition of this image, specifically focusing on: {}. Provide detailed feedback and suggestions for improvement.",
            aspect_descriptions.join(", ")
        )
    }
    
    async fn select_optimal_model(&self, analysis_type: &ImageAnalysisType) -> Result<String> {
        let preferences = match analysis_type {
            ImageAnalysisType::General => &self.config.preferred_models.general_analysis,
            ImageAnalysisType::Detailed => &self.config.preferred_models.detailed_analysis,
            ImageAnalysisType::OCR => &self.config.preferred_models.ocr,
            ImageAnalysisType::Scene => &self.config.preferred_models.scene_understanding,
            ImageAnalysisType::Technical => &self.config.preferred_models.detailed_analysis,
        };
        
        let available_models = self.model_registry.list_models().await;
        
        for preferred in preferences {
            if available_models.iter().any(|m| m.id == *preferred && m.supports_vision) {
                return Ok(preferred.clone());
            }
        }
        
        // Fallback to any vision-capable model
        available_models
            .iter()
            .find(|m| m.supports_vision)
            .map(|m| m.id.clone())
            .ok_or_else(|| anyhow::anyhow!("No vision-capable models available"))
    }
    
    async fn select_ocr_model(&self) -> Result<String> {
        self.select_optimal_model(&ImageAnalysisType::OCR).await
    }
    
    async fn check_context_utilization(
        &self,
        input: &MultimodalInput,
        model_id: Option<&str>,
    ) -> Result<ContextUtilization> {
        let context_window = if let Some(model_id) = model_id {
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
    
    fn calculate_confidence(&self, response: &ModelResponse) -> f32 {
        // Simplified confidence calculation based on response characteristics
        let text_length = response.text.len() as f32;
        let token_count = response.tokens.len() as f32;
        
        // Basic heuristic: longer, more detailed responses indicate higher confidence
        let length_score = (text_length / 1000.0).min(1.0);
        let detail_score = (token_count / 500.0).min(1.0);
        
        ((length_score + detail_score) / 2.0 * 100.0).round() / 100.0
    }
    
    fn parse_analysis_response(&self, response: &ModelResponse, analysis_type: &ImageAnalysisType) -> Result<ImageAnalysis> {
        // Parse the response text into structured analysis
        // This would be more sophisticated in a real implementation
        Ok(ImageAnalysis {
            description: response.text.clone(),
            objects_detected: Vec::new(), // Would extract from response
            scene_type: None,
            dominant_colors: Vec::new(),
            composition_notes: Vec::new(),
            technical_details: None,
        })
    }
    
    fn parse_comparison_response(&self, response: &ModelResponse, comparison_type: &ComparisonType) -> Result<ImageComparison> {
        Ok(ImageComparison {
            summary: response.text.clone(),
            similarities: Vec::new(), // Would extract from response
            differences: Vec::new(),
            ranking: None,
            recommendations: Vec::new(),
        })
    }
    
    fn parse_ocr_response(&self, response: &ModelResponse, options: &OCROptions) -> Result<OCRExtraction> {
        Ok(OCRExtraction {
            text: response.text.clone(),
            regions: Vec::new(), // Would extract text regions
            language: None,
        })
    }
    
    fn parse_composition_analysis(&self, analysis: &ImageAnalysis, aspects: &[CompositionAspect]) -> Result<CompositionAnalysis> {
        Ok(CompositionAnalysis {
            overall_quality_score: 0.8, // Would calculate from analysis
            aspect_scores: HashMap::new(),
            improvement_suggestions: Vec::new(),
            strengths: Vec::new(),
            weaknesses: Vec::new(),
        })
    }
    
    async fn get_cached_result(&self, image_hash: &str, analysis_type: &ImageAnalysisType) -> Option<ImageAnalysisResult> {
        // Simplified cache lookup
        None
    }
    
    async fn cache_result(&self, image_hash: &str, analysis_type: &ImageAnalysisType, result: &ImageAnalysisResult) {
        // Simplified cache storage
    }
    
    async fn update_cache_stats(&self, hit: bool) {
        let mut stats = self.stats.write().await;
        if hit {
            stats.cache_hits += 1;
        } else {
            stats.cache_misses += 1;
        }
    }
    
    async fn update_stats(&self, result: &ImageAnalysisResult, format: &ImageFormat) {
        let mut stats = self.stats.write().await;
        stats.images_processed += 1;
        stats.total_processing_time_ms += result.processing_time_ms;
        stats.preprocessing_time_ms += result.preprocessing_time_ms;
        stats.inference_time_ms += result.inference_time_ms;
        stats.average_processing_time_ms = stats.total_processing_time_ms as f64 / stats.images_processed as f64;
        
        let format_key = format!("{:?}", format);
        *stats.by_format.entry(format_key).or_insert(0) += 1;
        
        let resolution_key = format!("{}x{}", 
            result.image_metadata.processed_dimensions.0,
            result.image_metadata.processed_dimensions.1
        );
        *stats.by_resolution.entry(resolution_key).or_insert(0) += 1;
    }
}

// Supporting types and implementations

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            max_width: 2048,
            max_height: 2048,
            auto_resize: true,
            auto_enhance: true,
            auto_normalize: false,
            jpeg_quality: 85,
            png_compression: 6,
            default_analysis_prompt: "Describe this image in detail, including objects, people, scenery, colors, and any notable features.".to_string(),
            detailed_analysis_prompt: "Provide a comprehensive analysis of this image, including composition, lighting, mood, context, and artistic elements.".to_string(),
            comparison_prompt: "Compare these images and describe their similarities and differences.".to_string(),
            enable_caching: true,
            cache_max_size: 100,
            processing_timeout_ms: 30000,
            preferred_models: VisionModelPreferences::default(),
        }
    }
}

impl Default for VisionModelPreferences {
    fn default() -> Self {
        Self {
            general_analysis: vec![
                "qwen2.5-vl-7b".to_string(),
                "pixtral-12b".to_string(),
                "llama-3.2-vision-11b".to_string(),
            ],
            detailed_analysis: vec![
                "pixtral-12b".to_string(),
                "qwen2.5-vl-7b".to_string(),
            ],
            comparison: vec![
                "qwen2.5-vl-7b".to_string(),
                "pixtral-12b".to_string(),
            ],
            ocr: vec![
                "qwen2.5-vl-7b".to_string(),
                "pixtral-12b".to_string(),
            ],
            scene_understanding: vec![
                "llama-3.2-vision-11b".to_string(),
                "qwen2.5-vl-7b".to_string(),
            ],
        }
    }
}

impl Default for VisionStats {
    fn default() -> Self {
        Self {
            images_processed: 0,
            total_processing_time_ms: 0,
            average_processing_time_ms: 0.0,
            cache_hits: 0,
            cache_misses: 0,
            preprocessing_time_ms: 0,
            inference_time_ms: 0,
            failed_processings: 0,
            by_format: HashMap::new(),
            by_resolution: HashMap::new(),
        }
    }
}

// Type definitions for vision processing

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageAnalysisType {
    General,
    Detailed,
    OCR,
    Scene,
    Technical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonType {
    Similarity,
    Quality,
    Content,
    Style,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionAspect {
    RuleOfThirds,
    Balance,
    Leading,
    Symmetry,
    Color,
    Lighting,
    Depth,
    Focus,
}

#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub original_width: u32,
    pub original_height: u32,
    pub color_space: String,
}

#[derive(Debug, Clone)]
pub struct EnhancedImageMetadata {
    pub original_format: ImageFormat,
    pub original_size: usize,
    pub processed_size: usize,
    pub original_dimensions: (u32, u32),
    pub processed_dimensions: (u32, u32),
    pub color_space: String,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct ImageAnalysisResult {
    pub analysis: ImageAnalysis,
    pub confidence_score: f32,
    pub processing_time_ms: u64,
    pub preprocessing_time_ms: u64,
    pub inference_time_ms: u64,
    pub context_utilization: ContextUtilization,
    pub model_used: String,
    pub image_metadata: EnhancedImageMetadata,
}

#[derive(Debug, Clone)]
pub struct ImageAnalysis {
    pub description: String,
    pub objects_detected: Vec<DetectedObject>,
    pub scene_type: Option<String>,
    pub dominant_colors: Vec<String>,
    pub composition_notes: Vec<String>,
    pub technical_details: Option<TechnicalDetails>,
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
    pub name: String,
    pub confidence: f32,
    pub bounding_box: Option<(f32, f32, f32, f32)>, // x, y, width, height (normalized)
}

#[derive(Debug, Clone)]
pub struct TechnicalDetails {
    pub lighting_analysis: String,
    pub composition_score: f32,
    pub color_harmony: String,
    pub sharpness_assessment: String,
}

#[derive(Debug, Clone)]
pub struct ImageComparisonResult {
    pub comparison: ImageComparison,
    pub num_images: usize,
    pub processing_time_ms: u64,
    pub confidence_score: f32,
    pub model_used: String,
}

#[derive(Debug, Clone)]
pub struct ImageComparison {
    pub summary: String,
    pub similarities: Vec<String>,
    pub differences: Vec<String>,
    pub ranking: Option<Vec<i32>>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCROptions {
    pub preserve_formatting: bool,
    pub target_language: Option<String>,
    pub include_confidence: bool,
    pub extract_tables: bool,
    pub recognize_handwriting: bool,
}

#[derive(Debug, Clone)]
pub struct TextExtractionResult {
    pub extracted_text: String,
    pub text_regions: Vec<TextRegion>,
    pub confidence_score: f32,
    pub processing_time_ms: u64,
    pub language_detected: Option<String>,
    pub model_used: String,
}

#[derive(Debug, Clone)]
pub struct TextRegion {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: (f32, f32, f32, f32), // x, y, width, height (normalized)
}

#[derive(Debug, Clone)]
struct OCRExtraction {
    pub text: String,
    pub regions: Vec<TextRegion>,
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CompositionAnalysisResult {
    pub composition: CompositionAnalysis,
    pub overall_score: f32,
    pub suggestions: Vec<String>,
    pub processing_time_ms: u64,
    pub model_used: String,
}

#[derive(Debug, Clone)]
pub struct CompositionAnalysis {
    pub overall_quality_score: f32,
    pub aspect_scores: HashMap<CompositionAspect, f32>,
    pub improvement_suggestions: Vec<String>,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
}

impl Default for OCROptions {
    fn default() -> Self {
        Self {
            preserve_formatting: true,
            target_language: None,
            include_confidence: false,
            extract_tables: false,
            recognize_handwriting: false,
        }
    }
}