// Google Gemma-3n implementation - latest and more capable
mod gemma3n;
pub use gemma3n::AIManager;

// Inference backend abstraction
mod inference_backend;
pub use inference_backend::{InferenceBackend, InferenceConfig, InferenceMetrics, BackendType};

// GGUF inference engine
mod gguf_inference;
pub use gguf_inference::GGUFInference;

// MLX inference engine (Apple Silicon only)
mod mlx_inference;
pub use mlx_inference::MLXInference;

// Unified inference manager
mod inference_manager;
pub use inference_manager::InferenceManager;

// Speech processing module for conversational AI
mod speech_processor;
pub use speech_processor::{SpeechProcessor, ConversationState};

// Whisper STT and VAD module
mod whisper;
pub use whisper::{WhisperSTT, SileroVAD};

// Keep other implementations for reference
mod tinyllama_v2;