// Google Gemma-3n implementation - latest and more capable
mod gemma3n;
pub use gemma3n::AIManager;

// Inference backend abstraction
mod inference_backend;
pub use inference_backend::{InferenceMetrics, BackendType};

// Ollama inference backend for reliable model management
mod ollama_inference;
// pub use ollama_inference::OllamaInference;

// Unified inference manager
mod inference_manager;
// pub use inference_manager::InferenceManager;

// Speech processing module for conversational AI
mod speech_processor;
pub use speech_processor::SpeechProcessor;

// Whisper STT and VAD module
mod whisper;
pub use whisper::{WhisperSTT, SileroVAD};

// Gemma 3N multimodal processor
mod multimodal_processor;
pub use multimodal_processor::{Gemma3NProcessor, MultimodalInput, ProcessedMultimodal};

// Keep other implementations for reference
// Removed legacy tinyllama_v2 implementation