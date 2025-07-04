// Ollama backends removed - now using mistral.rs via inference module
// pub mod candle_backend; // Has compilation issues with candle API
// pub mod mistral_rs_backend; // Disabled due to dependency issues
// pub mod llama_cpp_backend; // Disabled due to dependency issues

// Legacy Ollama backends removed
// pub use enhanced_ollama_backend::EnhancedOllamaBackend;
// pub use ollama_gguf_backend::OllamaGgufBackend;
// pub use candle_backend::CandleBackend;
// pub use mistral_rs_backend::MistralRsBackend;
// pub use llama_cpp_backend::LlamaCppBackend;

// Optional backends
#[cfg(feature = "triton")]
pub mod triton_backend;
#[cfg(feature = "triton")]
pub use triton_backend::TritonBackend;