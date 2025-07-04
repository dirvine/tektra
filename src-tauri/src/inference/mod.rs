use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use futures::Stream;

// pub mod mistralrs_backend; // Temporarily disabled for compilation
pub mod model_abstraction;
pub mod model_registry;
pub mod streaming;
pub mod quantization;
pub mod huggingface;
pub mod token_estimator;
// pub mod qwen_omni; // Temporarily disabled for compilation

pub use model_abstraction::*;
pub use model_registry::*;
pub use streaming::*;
pub use huggingface::*;
pub use token_estimator::*;
// pub use qwen_omni::*; // Temporarily disabled for compilation

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub context_window: usize,
    pub parameters: Option<u64>,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_documents: bool,
    pub quantization: Option<String>,
    pub architecture: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultimodalInput {
    Text(String),
    TextWithImage { text: String, image: ImageData },
    TextWithAudio { text: String, audio: AudioData },
    TextWithVideo { text: String, video: VideoData },
    TextWithDocument { text: String, document: DocumentData },
    // Enhanced Omni-specific variants
    RealTimeAudio { audio_stream: AudioStream },
    MultimodalConversation {
        text: Option<String>,
        images: Vec<ImageData>,
        audio: Option<AudioData>,
        video: Option<VideoData>,
        documents: Vec<DocumentData>,
        real_time: bool,
        conversation_context: Option<ConversationContext>,
    },
    Combined {
        text: Option<String>,
        images: Vec<ImageData>,
        audio: Option<AudioData>,
        documents: Vec<DocumentData>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub format: ImageFormat,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    Gif,
    WebP,
    Bmp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    pub data: Vec<u8>,
    pub format: AudioFormat,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub duration: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    Raw, // For real-time streams
    Opus, // For low-latency streaming
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    pub data: Vec<u8>,
    pub format: VideoFormat,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fps: Option<f32>,
    pub duration: Option<f32>,
    pub frames: Option<Vec<ImageData>>, // For frame-by-frame analysis
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VideoFormat {
    Mp4,
    Webm,
    Avi,
    Mov,
    Frames, // Sequence of image frames
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStream {
    pub chunk_data: Vec<u8>,
    pub format: AudioFormat,
    pub sample_rate: u32,
    pub channels: u16,
    pub timestamp: u64, // Timestamp in milliseconds
    pub is_final: bool, // Whether this is the final chunk
    pub vad_confidence: Option<f32>, // Voice activity detection confidence
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationContext {
    pub session_id: String,
    pub turn_count: usize,
    pub history: Vec<ConversationTurn>,
    pub speaker_profile: Option<SpeakerProfile>,
    pub emotion_state: Option<EmotionState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String, // "user" or "assistant"
    pub content: String,
    pub timestamp: u64,
    pub modalities: Vec<String>, // ["text", "audio", "vision"]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub voice_id: String,
    pub language: String,
    pub accent: Option<String>,
    pub speaking_rate: f32,
    pub pitch_range: (f32, f32),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionState {
    pub primary_emotion: String,
    pub confidence: f32,
    pub valence: f32, // -1.0 to 1.0
    pub arousal: f32, // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentData {
    pub data: Vec<u8>,
    pub format: DocumentFormat,
    pub title: Option<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DocumentFormat {
    Pdf,
    Docx,
    Txt,
    Markdown,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub logprob: Option<f32>,
    pub special: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    pub text: String,
    pub tokens: Vec<Token>,
    pub finish_reason: FinishReason,
    pub usage: UsageStats,
    // Enhanced for Omni capabilities
    pub audio: Option<AudioData>, // For speech output
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    pub model_used: String,
    pub processing_time_ms: u64,
    pub modalities_processed: Vec<String>,
    pub thinker_talker_timing: Option<ThinkerTalkerTiming>,
    pub confidence_scores: HashMap<String, f32>,
    pub quality_metrics: Option<ResponseQuality>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkerTalkerTiming {
    pub thinker_time_ms: u64, // Time spent on understanding/reasoning
    pub talker_time_ms: u64,  // Time spent on speech synthesis
    pub total_time_ms: u64,
    pub parallel_processing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseQuality {
    pub text_coherence: f32,
    pub speech_naturalness: Option<f32>,
    pub multimodal_alignment: Option<f32>,
    pub overall_quality: f32,
}

impl Default for ResponseMetadata {
    fn default() -> Self {
        Self {
            model_used: "unknown".to_string(),
            processing_time_ms: 0,
            modalities_processed: vec!["text".to_string()],
            thinker_talker_timing: None,
            confidence_scores: HashMap::new(),
            quality_metrics: None,
        }
    }
}

impl Default for ThinkerTalkerTiming {
    fn default() -> Self {
        Self {
            thinker_time_ms: 0,
            talker_time_ms: 0,
            total_time_ms: 0,
            parallel_processing: false,
        }
    }
}

impl Default for ResponseQuality {
    fn default() -> Self {
        Self {
            text_coherence: 0.8,
            speech_naturalness: None,
            multimodal_alignment: None,
            overall_quality: 0.8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FinishReason {
    Length,
    Stop,
    ToolCall,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_id: String,
    pub quantization: Option<String>,
    pub context_window: usize,
    pub device: DeviceConfig,
    pub cache_dir: Option<String>,
    pub custom_params: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceConfig {
    Cpu,
    Cuda(usize),
    Metal,
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    TextOnly,
    Vision,
    Audio,
    Multimodal,
}

/// Core trait for multimodal AI models
#[async_trait]
pub trait MultimodalModel: Send + Sync {
    /// Load the model with the given configuration
    async fn load(&mut self, config: &ModelConfig) -> Result<()>;
    
    /// Unload the model and free resources
    async fn unload(&mut self) -> Result<()>;
    
    /// Check if model is loaded and ready
    fn is_loaded(&self) -> bool;
    
    /// Generate response for multimodal input
    async fn generate(&self, input: MultimodalInput) -> Result<ModelResponse>;
    
    /// Generate streaming response
    async fn stream_generate(&self, input: MultimodalInput) 
        -> Result<Pin<Box<dyn Stream<Item = Result<Token>> + Send>>>;
    
    /// Get model capabilities
    fn supports_vision(&self) -> bool;
    fn supports_audio(&self) -> bool;
    fn supports_documents(&self) -> bool;
    fn context_window(&self) -> usize;
    
    /// Get model information
    fn model_info(&self) -> ModelInfo;
    
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// Error types for inference operations
#[derive(thiserror::Error, Debug)]
pub enum InferenceError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    
    #[error("Inference failed: {0}")]
    InferenceError(String),
    
    #[error("Unsupported input type: {0}")]
    UnsupportedInput(String),
    
    #[error("Resource error: {0}")]
    ResourceError(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}