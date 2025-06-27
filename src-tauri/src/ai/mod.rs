// Google Gemma-3n implementation - latest and more capable
mod gemma3n;
pub use gemma3n::AIManager;

// Speech processing module for conversational AI
mod speech_processor;
pub use speech_processor::{SpeechProcessor, ConversationState};

// Whisper STT and VAD module
mod whisper;
pub use whisper::{WhisperSTT, SileroVAD};

// Keep other implementations for reference
mod tinyllama_v2;