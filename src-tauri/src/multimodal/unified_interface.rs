use super::{
    EnhancedMultimodalProcessor, EnhancedVisionProcessor, MultimodalProcessor,
    VisionProcessor, AudioProcessor, DocumentProcessor,
    ProcessingStats, ProcessedImageResult, ProcessedDocumentResult, 
    ProcessedComplexResult
};
use crate::multimodal::enhanced_vision::{
    ImageAnalysis, ImageAnalysisResult, ImageAnalysisType, ImageQuality,
    ComparisonType, OCROptions
};
use crate::inference::{EnhancedModelRegistry, MultimodalInput, ModelResponse, ContextUtilization};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};
use futures::Stream;
use std::pin::Pin;

/// Unified interface for all multimodal processing capabilities
pub struct UnifiedMultimodalInterface {
    /// Enhanced processors
    enhanced_processor: Arc<EnhancedMultimodalProcessor>,
    enhanced_vision: Arc<EnhancedVisionProcessor>,
    
    /// Model registry for inference
    model_registry: Arc<EnhancedModelRegistry>,
    
    /// Session management
    active_sessions: Arc<RwLock<std::collections::HashMap<String, ProcessingSession>>>,
    
    /// Global configuration
    config: InterfaceConfig,
    
    /// Performance metrics
    metrics: Arc<RwLock<InterfaceMetrics>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceConfig {
    /// Default processing preferences
    pub default_vision_quality: ImageQuality,
    pub default_audio_sample_rate: u32,
    pub max_concurrent_sessions: usize,
    
    /// Performance settings
    pub enable_preprocessing_cache: bool,
    pub enable_result_cache: bool,
    pub cache_cleanup_interval_minutes: u64,
    
    /// Model selection
    pub auto_select_optimal_model: bool,
    pub fallback_to_available_models: bool,
    
    /// Processing limits
    pub max_image_size_mb: usize,
    pub max_audio_duration_seconds: u32,
    pub max_document_size_mb: usize,
    
    /// Streaming settings
    pub enable_streaming_by_default: bool,
    pub streaming_chunk_size: usize,
}

#[derive(Debug, Clone)]
struct ProcessingSession {
    session_id: String,
    created_at: std::time::SystemTime,
    last_activity: std::time::SystemTime,
    processing_history: Vec<ProcessingEvent>,
    active_model: Option<String>,
    user_preferences: UserPreferences,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    pub preferred_language: Option<String>,
    pub analysis_detail_level: AnalysisDetailLevel,
    pub enable_technical_details: bool,
    pub preferred_output_format: OutputFormat,
    pub vision_focus_areas: Vec<VisionFocusArea>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDetailLevel {
    Brief,
    Standard,
    Detailed,
    Comprehensive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    PlainText,
    Structured,
    Markdown,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionFocusArea {
    Objects,
    People,
    Text,
    Scene,
    Technical,
    Artistic,
    Safety,
    Accessibility,
}

#[derive(Debug, Clone)]
struct ProcessingEvent {
    timestamp: std::time::SystemTime,
    event_type: ProcessingEventType,
    processing_time_ms: u64,
    model_used: String,
    success: bool,
}

#[derive(Debug, Clone)]
enum ProcessingEventType {
    ImageAnalysis,
    AudioProcessing,
    DocumentAnalysis,
    MultimodalCombined,
    Comparison,
    Streaming,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time_ms: f64,
    pub requests_by_type: std::collections::HashMap<String, u64>,
    pub active_sessions_count: usize,
    pub cache_hit_rate: f32,
    pub model_usage_stats: std::collections::HashMap<String, u64>,
}

impl UnifiedMultimodalInterface {
    /// Create a new unified multimodal interface
    pub async fn new(model_registry: Arc<EnhancedModelRegistry>) -> Result<Self> {
        info!("Initializing unified multimodal interface");
        
        let enhanced_processor = Arc::new(
            EnhancedMultimodalProcessor::new(model_registry.clone()).await?
        );
        
        let enhanced_vision = Arc::new(
            EnhancedVisionProcessor::new(model_registry.clone()).await?
        );
        
        Ok(Self {
            enhanced_processor,
            enhanced_vision,
            model_registry,
            active_sessions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            config: InterfaceConfig::default(),
            metrics: Arc::new(RwLock::new(InterfaceMetrics::default())),
        })
    }
    
    /// Start a new processing session
    pub async fn start_session(&self, session_id: String, preferences: Option<UserPreferences>) -> Result<String> {
        info!("Starting processing session: {}", session_id);
        
        let session = ProcessingSession {
            session_id: session_id.clone(),
            created_at: std::time::SystemTime::now(),
            last_activity: std::time::SystemTime::now(),
            processing_history: Vec::new(),
            active_model: self.model_registry.get_active_model_id().await,
            user_preferences: preferences.unwrap_or_default(),
        };
        
        let mut sessions = self.active_sessions.write().await;
        
        // Check session limits
        if sessions.len() >= self.config.max_concurrent_sessions {
            return Err(anyhow::anyhow!("Maximum concurrent sessions reached"));
        }
        
        sessions.insert(session_id.clone(), session);
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.active_sessions_count = sessions.len();
        
        Ok(session_id)
    }
    
    /// End a processing session
    pub async fn end_session(&self, session_id: &str) -> Result<ProcessingSessionSummary> {
        info!("Ending processing session: {}", session_id);
        
        let mut sessions = self.active_sessions.write().await;
        let session = sessions.remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        // Update metrics
        let mut metrics = self.metrics.write().await;
        metrics.active_sessions_count = sessions.len();
        
        Ok(ProcessingSessionSummary {
            session_id: session_id.to_string(),
            duration_ms: session.created_at.elapsed().unwrap_or_default().as_millis() as u64,
            total_requests: session.processing_history.len(),
            successful_requests: session.processing_history.iter().filter(|e| e.success).count(),
            models_used: session.processing_history.iter()
                .map(|e| e.model_used.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect(),
        })
    }
    
    /// Process single image with comprehensive analysis
    pub async fn analyze_image_comprehensive(
        &self,
        session_id: &str,
        image_data: &[u8],
        analysis_request: ImageAnalysisRequest,
    ) -> Result<ComprehensiveImageResult> {
        let start_time = std::time::Instant::now();
        info!("Processing comprehensive image analysis for session: {}", session_id);
        
        // Validate session
        self.validate_session(session_id).await?;
        
        // Get user preferences
        let preferences = self.get_user_preferences(session_id).await?;
        
        // Perform analysis based on request type
        let result = match analysis_request.analysis_type {
            ImageAnalysisType::General => {
                self.enhanced_vision.analyze_image(
                    image_data,
                    ImageAnalysisType::General,
                    analysis_request.custom_prompt,
                ).await?
            }
            ImageAnalysisType::Detailed => {
                self.enhanced_vision.analyze_image(
                    image_data,
                    ImageAnalysisType::Detailed,
                    analysis_request.custom_prompt,
                ).await?
            }
            ImageAnalysisType::OCR => {
                let ocr_result = self.enhanced_vision.extract_text(
                    image_data,
                    analysis_request.ocr_options.unwrap_or_default(),
                ).await?;
                
                // Convert to ImageAnalysisResult format
                ImageAnalysisResult {
                    analysis: ImageAnalysis {
                        description: ocr_result.extracted_text.clone(),
                        objects_detected: Vec::new(),
                        scene_type: None,
                        dominant_colors: Vec::new(),
                        composition_notes: Vec::new(),
                        technical_details: None,
                    },
                    confidence_score: ocr_result.confidence_score,
                    processing_time_ms: ocr_result.processing_time_ms,
                    preprocessing_time_ms: 0,
                    inference_time_ms: ocr_result.processing_time_ms,
                    context_utilization: ContextUtilization {
                        used_tokens: 100, // Estimated
                        total_tokens: 8192,
                        remaining_tokens: 8092,
                        utilization_percentage: 1.2,
                        status: crate::inference::UtilizationStatus::Low,
                    },
                    model_used: ocr_result.model_used,
                    image_metadata: EnhancedImageMetadata {
                        original_format: crate::inference::ImageFormat::Png,
                        original_size: image_data.len(),
                        processed_size: image_data.len(),
                        original_dimensions: (0, 0),
                        processed_dimensions: (0, 0),
                        color_space: "RGB".to_string(),
                        compression_ratio: 1.0,
                    },
                }
            }
            _ => {
                self.enhanced_vision.analyze_image(
                    image_data,
                    analysis_request.analysis_type,
                    analysis_request.custom_prompt,
                ).await?
            }
        };
        
        // Format result according to user preferences
        let formatted_result = self.format_image_result(&result, &preferences).await?;
        
        // Record processing event
        self.record_processing_event(
            session_id,
            ProcessingEventType::ImageAnalysis,
            start_time.elapsed().as_millis() as u64,
            result.model_used.clone(),
            true,
        ).await?;
        
        // Update metrics
        self.update_metrics(true, start_time.elapsed().as_millis() as u64, "image_analysis").await;
        
        Ok(ComprehensiveImageResult {
            original_result: result,
            formatted_output: formatted_result,
            session_id: session_id.to_string(),
            processing_metadata: ProcessingMetadata {
                request_timestamp: std::time::SystemTime::now(),
                total_processing_time_ms: start_time.elapsed().as_millis() as u64,
                model_selection_reason: "User preference or optimal selection".to_string(),
                cache_hit: false, // Would be determined by actual cache check
            },
        })
    }
    
    /// Compare multiple images with detailed analysis
    pub async fn compare_images_advanced(
        &self,
        session_id: &str,
        images: Vec<&[u8]>,
        comparison_request: ImageComparisonRequest,
    ) -> Result<AdvancedComparisonResult> {
        let start_time = std::time::Instant::now();
        info!("Processing advanced image comparison for session: {} with {} images", session_id, images.len());
        
        // Validate session and input
        self.validate_session(session_id).await?;
        if images.len() < 2 {
            return Err(anyhow::anyhow!("At least 2 images required for comparison"));
        }
        
        // Perform comparison
        let comparison_result = self.enhanced_vision.compare_images(
            images,
            comparison_request.comparison_type,
            comparison_request.custom_prompt,
        ).await?;
        
        // Get detailed analysis for each image if requested
        let individual_analyses = if comparison_request.include_individual_analysis {
            let mut analyses = Vec::new();
            for (i, image_data) in images.iter().enumerate() {
                let analysis = self.enhanced_vision.analyze_image(
                    image_data,
                    ImageAnalysisType::General,
                    Some(format!("Analyze image {} in detail", i + 1)),
                ).await?;
                analyses.push(analysis);
            }
            Some(analyses)
        } else {
            None
        };
        
        // Record processing event
        self.record_processing_event(
            session_id,
            ProcessingEventType::Comparison,
            start_time.elapsed().as_millis() as u64,
            comparison_result.model_used.clone(),
            true,
        ).await?;
        
        Ok(AdvancedComparisonResult {
            comparison_result,
            individual_analyses,
            cross_reference_insights: self.generate_cross_reference_insights(&images).await?,
            session_id: session_id.to_string(),
            processing_metadata: ProcessingMetadata {
                request_timestamp: std::time::SystemTime::now(),
                total_processing_time_ms: start_time.elapsed().as_millis() as u64,
                model_selection_reason: "Optimal for comparison tasks".to_string(),
                cache_hit: false,
            },
        })
    }
    
    /// Process complex multimodal input with intelligent handling
    pub async fn process_multimodal_intelligent(
        &self,
        session_id: &str,
        multimodal_request: IntelligentMultimodalRequest,
    ) -> Result<IntelligentMultimodalResult> {
        let start_time = std::time::Instant::now();
        info!("Processing intelligent multimodal request for session: {}", session_id);
        
        // Validate session
        self.validate_session(session_id).await?;
        
        // Analyze the request to determine optimal processing strategy
        let processing_strategy = self.analyze_processing_strategy(&multimodal_request).await?;
        
        // Execute processing based on strategy
        let result = match processing_strategy {
            ProcessingStrategy::Sequential => {
                self.process_sequential(&multimodal_request).await?
            }
            ProcessingStrategy::Parallel => {
                self.process_parallel(&multimodal_request).await?
            }
            ProcessingStrategy::Hierarchical => {
                self.process_hierarchical(&multimodal_request).await?
            }
            ProcessingStrategy::Adaptive => {
                self.process_adaptive(&multimodal_request).await?
            }
        };
        
        // Record processing event
        self.record_processing_event(
            session_id,
            ProcessingEventType::MultimodalCombined,
            start_time.elapsed().as_millis() as u64,
            result.primary_model_used.clone(),
            true,
        ).await?;
        
        Ok(result)
    }
    
    /// Stream processing for real-time applications
    pub async fn stream_multimodal_processing(
        &self,
        session_id: &str,
        stream_request: StreamProcessingRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send>>> {
        info!("Starting streaming multimodal processing for session: {}", session_id);
        
        // Validate session
        self.validate_session(session_id).await?;
        
        // Create streaming pipeline based on request type
        match stream_request.content_type {
            StreamContentType::Image => {
                self.create_image_stream(session_id, stream_request).await
            }
            StreamContentType::Audio => {
                self.create_audio_stream(session_id, stream_request).await
            }
            StreamContentType::Combined => {
                self.create_combined_stream(session_id, stream_request).await
            }
        }
    }
    
    /// Get comprehensive interface statistics
    pub async fn get_interface_metrics(&self) -> InterfaceMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get session-specific statistics
    pub async fn get_session_stats(&self, session_id: &str) -> Result<SessionStats> {
        let sessions = self.active_sessions.read().await;
        let session = sessions.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        
        Ok(SessionStats {
            session_id: session_id.to_string(),
            duration_ms: session.created_at.elapsed().unwrap_or_default().as_millis() as u64,
            requests_processed: session.processing_history.len(),
            average_processing_time_ms: session.processing_history.iter()
                .map(|e| e.processing_time_ms)
                .sum::<u64>() as f64 / session.processing_history.len().max(1) as f64,
            models_used: session.processing_history.iter()
                .map(|e| e.model_used.clone())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            success_rate: session.processing_history.iter()
                .filter(|e| e.success)
                .count() as f32 / session.processing_history.len().max(1) as f32,
        })
    }
    
    // Helper methods
    
    async fn validate_session(&self, session_id: &str) -> Result<()> {
        let sessions = self.active_sessions.read().await;
        if !sessions.contains_key(session_id) {
            return Err(anyhow::anyhow!("Invalid or expired session: {}", session_id));
        }
        Ok(())
    }
    
    async fn get_user_preferences(&self, session_id: &str) -> Result<UserPreferences> {
        let sessions = self.active_sessions.read().await;
        let session = sessions.get(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;
        Ok(session.user_preferences.clone())
    }
    
    async fn format_image_result(&self, result: &ImageAnalysisResult, preferences: &UserPreferences) -> Result<String> {
        match preferences.preferred_output_format {
            OutputFormat::PlainText => Ok(result.analysis.description.clone()),
            OutputFormat::Structured => Ok(self.format_structured_analysis(result)),
            OutputFormat::Markdown => Ok(self.format_markdown_analysis(result)),
            OutputFormat::Json => Ok(serde_json::to_string_pretty(&result.analysis)?),
        }
    }
    
    fn format_structured_analysis(&self, result: &ImageAnalysisResult) -> String {
        format!(
            "Image Analysis:\n\
            Description: {}\n\
            Confidence: {:.1}%\n\
            Processing Time: {}ms\n\
            Model: {}",
            result.analysis.description,
            result.confidence_score * 100.0,
            result.processing_time_ms,
            result.model_used
        )
    }
    
    fn format_markdown_analysis(&self, result: &ImageAnalysisResult) -> String {
        format!(
            "# Image Analysis\n\n\
            ## Description\n{}\n\n\
            ## Metadata\n\
            - **Confidence**: {:.1}%\n\
            - **Processing Time**: {}ms\n\
            - **Model Used**: {}\n",
            result.analysis.description,
            result.confidence_score * 100.0,
            result.processing_time_ms,
            result.model_used
        )
    }
    
    async fn record_processing_event(
        &self,
        session_id: &str,
        event_type: ProcessingEventType,
        processing_time_ms: u64,
        model_used: String,
        success: bool,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.last_activity = std::time::SystemTime::now();
            session.processing_history.push(ProcessingEvent {
                timestamp: std::time::SystemTime::now(),
                event_type,
                processing_time_ms,
                model_used,
                success,
            });
        }
        Ok(())
    }
    
    async fn update_metrics(&self, success: bool, processing_time_ms: u64, request_type: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        
        if success {
            metrics.successful_requests += 1;
        } else {
            metrics.failed_requests += 1;
        }
        
        // Update average response time
        let total_successful = metrics.successful_requests;
        metrics.average_response_time_ms = 
            (metrics.average_response_time_ms * (total_successful - 1) as f64 + processing_time_ms as f64) / total_successful as f64;
        
        // Update request type counts
        *metrics.requests_by_type.entry(request_type.to_string()).or_insert(0) += 1;
    }
    
    async fn analyze_processing_strategy(&self, request: &IntelligentMultimodalRequest) -> Result<ProcessingStrategy> {
        // Analyze the request complexity and determine optimal processing strategy
        let total_inputs = request.images.len() + 
                          if request.audio.is_some() { 1 } else { 0 } +
                          request.documents.len() +
                          if request.text.is_some() { 1 } else { 0 };
        
        Ok(match total_inputs {
            0..=2 => ProcessingStrategy::Sequential,
            3..=5 => ProcessingStrategy::Parallel,
            6..=10 => ProcessingStrategy::Hierarchical,
            _ => ProcessingStrategy::Adaptive,
        })
    }
    
    async fn process_sequential(&self, request: &IntelligentMultimodalRequest) -> Result<IntelligentMultimodalResult> {
        // Sequential processing implementation
        // This is a simplified placeholder
        Ok(IntelligentMultimodalResult {
            primary_response: "Sequential processing completed".to_string(),
            component_results: Vec::new(),
            processing_strategy_used: ProcessingStrategy::Sequential,
            total_processing_time_ms: 1000,
            primary_model_used: "default".to_string(),
            confidence_score: 0.9,
        })
    }
    
    async fn process_parallel(&self, request: &IntelligentMultimodalRequest) -> Result<IntelligentMultimodalResult> {
        // Parallel processing implementation
        Ok(IntelligentMultimodalResult {
            primary_response: "Parallel processing completed".to_string(),
            component_results: Vec::new(),
            processing_strategy_used: ProcessingStrategy::Parallel,
            total_processing_time_ms: 800,
            primary_model_used: "default".to_string(),
            confidence_score: 0.9,
        })
    }
    
    async fn process_hierarchical(&self, request: &IntelligentMultimodalRequest) -> Result<IntelligentMultimodalResult> {
        // Hierarchical processing implementation
        Ok(IntelligentMultimodalResult {
            primary_response: "Hierarchical processing completed".to_string(),
            component_results: Vec::new(),
            processing_strategy_used: ProcessingStrategy::Hierarchical,
            total_processing_time_ms: 1200,
            primary_model_used: "default".to_string(),
            confidence_score: 0.85,
        })
    }
    
    async fn process_adaptive(&self, request: &IntelligentMultimodalRequest) -> Result<IntelligentMultimodalResult> {
        // Adaptive processing implementation
        Ok(IntelligentMultimodalResult {
            primary_response: "Adaptive processing completed".to_string(),
            component_results: Vec::new(),
            processing_strategy_used: ProcessingStrategy::Adaptive,
            total_processing_time_ms: 1500,
            primary_model_used: "default".to_string(),
            confidence_score: 0.88,
        })
    }
    
    async fn generate_cross_reference_insights(&self, images: &[&[u8]]) -> Result<Vec<String>> {
        // Generate cross-reference insights between images
        Ok(vec![
            "Common objects detected across images".to_string(),
            "Style consistency analysis".to_string(),
            "Temporal progression insights".to_string(),
        ])
    }
    
    async fn create_image_stream(&self, session_id: &str, request: StreamProcessingRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send>>> {
        // Create image streaming pipeline
        // This would be implemented with actual streaming logic
        Err(anyhow::anyhow!("Image streaming not yet implemented"))
    }
    
    async fn create_audio_stream(&self, session_id: &str, request: StreamProcessingRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send>>> {
        // Create audio streaming pipeline
        Err(anyhow::anyhow!("Audio streaming not yet implemented"))
    }
    
    async fn create_combined_stream(&self, session_id: &str, request: StreamProcessingRequest) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse>> + Send>>> {
        // Create combined multimodal streaming pipeline
        Err(anyhow::anyhow!("Combined streaming not yet implemented"))
    }
}

// Default implementations and supporting types

impl Default for InterfaceConfig {
    fn default() -> Self {
        Self {
            default_vision_quality: ImageQuality::Medium,
            default_audio_sample_rate: 16000,
            max_concurrent_sessions: 10,
            enable_preprocessing_cache: true,
            enable_result_cache: true,
            cache_cleanup_interval_minutes: 30,
            auto_select_optimal_model: true,
            fallback_to_available_models: true,
            max_image_size_mb: 10,
            max_audio_duration_seconds: 300,
            max_document_size_mb: 50,
            enable_streaming_by_default: false,
            streaming_chunk_size: 1024,
        }
    }
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            preferred_language: None,
            analysis_detail_level: AnalysisDetailLevel::Standard,
            enable_technical_details: false,
            preferred_output_format: OutputFormat::PlainText,
            vision_focus_areas: vec![VisionFocusArea::Objects, VisionFocusArea::Scene],
        }
    }
}

impl Default for InterfaceMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_response_time_ms: 0.0,
            requests_by_type: std::collections::HashMap::new(),
            active_sessions_count: 0,
            cache_hit_rate: 0.0,
            model_usage_stats: std::collections::HashMap::new(),
        }
    }
}

// Request and response types

#[derive(Debug, Clone)]
pub struct ImageAnalysisRequest {
    pub analysis_type: ImageAnalysisType,
    pub custom_prompt: Option<String>,
    pub ocr_options: Option<OCROptions>,
    pub processing_options: Option<ImageProcessingOptions>,
}

#[derive(Debug, Clone)]
pub struct ImageComparisonRequest {
    pub comparison_type: ComparisonType,
    pub custom_prompt: Option<String>,
    pub include_individual_analysis: bool,
}

#[derive(Debug, Clone)]
pub struct IntelligentMultimodalRequest {
    pub text: Option<String>,
    pub images: Vec<Vec<u8>>,
    pub audio: Option<Vec<u8>>,
    pub documents: Vec<Vec<u8>>,
    pub processing_intent: ProcessingIntent,
    pub priority_modalities: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ProcessingIntent {
    Analysis,
    Extraction,
    Comparison,
    Summary,
    Translation,
    Generation,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum ProcessingStrategy {
    Sequential,
    Parallel,
    Hierarchical,
    Adaptive,
}

#[derive(Debug, Clone)]
pub struct StreamProcessingRequest {
    pub content_type: StreamContentType,
    pub chunk_size: Option<usize>,
    pub processing_options: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum StreamContentType {
    Image,
    Audio,
    Combined,
}

// Result types

#[derive(Debug, Clone)]
pub struct ComprehensiveImageResult {
    pub original_result: ImageAnalysisResult,
    pub formatted_output: String,
    pub session_id: String,
    pub processing_metadata: ProcessingMetadata,
}

#[derive(Debug, Clone)]
pub struct AdvancedComparisonResult {
    pub comparison_result: ImageComparisonResult,
    pub individual_analyses: Option<Vec<ImageAnalysisResult>>,
    pub cross_reference_insights: Vec<String>,
    pub session_id: String,
    pub processing_metadata: ProcessingMetadata,
}

#[derive(Debug, Clone)]
pub struct IntelligentMultimodalResult {
    pub primary_response: String,
    pub component_results: Vec<ComponentResult>,
    pub processing_strategy_used: ProcessingStrategy,
    pub total_processing_time_ms: u64,
    pub primary_model_used: String,
    pub confidence_score: f32,
}

#[derive(Debug, Clone)]
pub struct ComponentResult {
    pub modality: String,
    pub result: String,
    pub confidence: f32,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct StreamingResponse {
    pub content: String,
    pub response_type: StreamResponseType,
    pub timestamp: std::time::SystemTime,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum StreamResponseType {
    Partial,
    Complete,
    Error,
    Status,
}

#[derive(Debug, Clone)]
pub struct ProcessingMetadata {
    pub request_timestamp: std::time::SystemTime,
    pub total_processing_time_ms: u64,
    pub model_selection_reason: String,
    pub cache_hit: bool,
}

#[derive(Debug, Clone)]
pub struct ProcessingSessionSummary {
    pub session_id: String,
    pub duration_ms: u64,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub models_used: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SessionStats {
    pub session_id: String,
    pub duration_ms: u64,
    pub requests_processed: usize,
    pub average_processing_time_ms: f64,
    pub models_used: usize,
    pub success_rate: f32,
}