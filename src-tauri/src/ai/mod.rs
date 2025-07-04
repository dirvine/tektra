// Google Gemma-3n implementation - latest and more capable
mod gemma3n;
pub use gemma3n::AIManager;

// Inference backend abstraction
mod inference_backend;
pub use inference_backend::{InferenceMetrics, BackendType, InferenceBackend, InferenceConfig};

// Legacy Ollama inference backend removed - now using mistral.rs via inference module

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

// Backend implementations
pub mod backends;

// Unified model manager for multi-backend support
pub mod unified_model_manager;
pub use unified_model_manager::{
    UnifiedModelManager, ModelConfig, GenerationParams, 
    MultimodalInput as UnifiedMultimodalInput, DeviceConfig
};

// Template manager for model-specific prompt formatting
pub mod template_manager;
pub use template_manager::{TemplateManager, PromptTemplate, ChatMessage, MessageRole};

// Model configuration loader
pub mod model_config_loader;
pub use model_config_loader::{ModelConfigLoader, ModelsConfiguration, ModelInfo};

// Tektra integration layer
pub mod tektra_integration;
pub use tektra_integration::TektraModelIntegration;

// Test modules
#[cfg(test)]
mod tests;

// Document processing for RAG
pub mod document_processor;
pub use document_processor::{
    DocumentFormat, ProcessedDocument, DocumentChunk, ChunkingStrategy,
    UnifiedDocumentProcessor, DocumentProcessor, DocumentMetadata,
};

// Embeddings generation
pub mod embeddings;
pub use embeddings::{
    EmbeddingGenerator, EmbeddingManager, SimpleEmbeddingGenerator, 
    EmbeddingConfig,
};

// Input pipeline for combining documents with queries
pub mod input_pipeline;
pub use input_pipeline::{
    InputPipeline, CombinedInput, DocumentContext, DocumentSource,
    InputMetadata, QueryType, PipelineConfig,
};

// MCP (Model Context Protocol) server implementation
pub mod mcp_server;
pub use mcp_server::{
    MCPServer, TektraMCPServer, MCPServerInfo, ServerCapabilities,
    Resource, Tool, MCPPrompt, ToolRequest, ToolResult, MCPClient,
};

// Keep other implementations for reference
// Removed legacy tinyllama_v2 implementation