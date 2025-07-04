pub mod ai;
pub mod vector_db;
pub mod database;
pub mod types;

// New modular architecture
pub mod inference;
pub mod multimodal;
pub mod conversation;
pub mod voice;

// Temporarily disable MCP until unified interface is fixed
// #[cfg(feature = "mcp-server")]
// pub mod mcp;

// Re-export commonly used types for tests
pub use crate::types::{ChatMessage, AppSettings};

// Re-export new architecture types
pub use crate::inference::{ModelRegistry, MultimodalInput, ModelResponse, EnhancedModelRegistry};
pub use crate::multimodal::{MultimodalProcessor, ProcessingStats}; // UnifiedMultimodalInterface temporarily disabled
pub use crate::conversation::{ConversationManager}; // EnhancedConversationManager temporarily disabled
pub use crate::voice::{VoicePipeline, VoiceCharacterConfig, SessionConfig, VoicePipelineEvent};

// #[cfg(feature = "mcp-server")]
// pub use crate::mcp::{TektraMCPServer, MCPServerConfig};

// Test modules
#[cfg(test)]
mod tests;

#[cfg(test)]
pub use tests::new_architecture_test;