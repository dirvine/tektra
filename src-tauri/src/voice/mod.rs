pub mod unmute_client;
pub mod voice_pipeline;
pub mod realtime_audio;
pub mod unmute_service_manager;
pub mod rust_backend_server;

pub use unmute_client::{
    UnmuteWebSocketClient, UnmuteMessage, UnmuteMessageData, UnmuteResponse, UnmuteError
};
pub use voice_pipeline::{
    VoicePipeline, ConversationState, VoicePipelineConfig, VoicePipelineEvent,
    AudioSettings, ModelSettings, VadSettings, AudioStatus
};
pub use realtime_audio::{
    RealtimeAudioManager, AudioFormat, AudioFormatConfig, AudioCodec, AudioEvent
};
pub use unmute_service_manager::{
    UnmuteServiceManager, UnmuteConfig, ServiceConfig, GpuConfig, ServiceEvent
};
pub use rust_backend_server::{
    RustBackendServer, BackendServerConfig
};

use serde::{Deserialize, Serialize};

/// Voice character configuration for personality and behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceCharacterConfig {
    /// Voice ID to use for TTS (e.g., "alloy", "echo", "fable")
    pub voice: String,
    /// System instructions for the AI character
    pub instructions: Option<String>,
    /// Whether to allow recording/transcription
    pub allow_recording: bool,
}

impl Default for VoiceCharacterConfig {
    fn default() -> Self {
        Self {
            voice: "default".to_string(),
            instructions: Some("You are Tektra, a helpful multimodal AI assistant. Respond naturally and conversationally in a friendly, concise manner.".to_string()),
            allow_recording: true,
        }
    }
}

/// Session configuration for Unmute WebSocket communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// System instructions for the conversation
    pub instructions: Option<String>,
    /// Voice to use for TTS responses
    pub voice: Option<String>,
    /// Whether recording is allowed for this session
    pub allow_recording: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            instructions: Some("You are Tektra, a helpful multimodal AI assistant. Respond naturally and conversationally.".to_string()),
            voice: Some("default".to_string()),
            allow_recording: true,
        }
    }
}