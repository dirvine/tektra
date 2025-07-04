use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::info;

use super::MCPServerConfig;

/// MCP server capabilities manager
pub struct CapabilityManager {
    capabilities: ServerCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tool calling capabilities
    pub tools: Option<ToolCapabilities>,
    
    /// Resource access capabilities
    pub resources: Option<ResourceCapabilities>,
    
    /// Prompt template capabilities
    pub prompts: Option<PromptCapabilities>,
    
    /// Sampling capabilities
    pub sampling: Option<SamplingCapabilities>,
    
    /// Experimental capabilities
    pub experimental: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapabilities {
    /// Whether the server supports listing available tools
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    /// Whether the server supports subscribing to resource changes
    pub subscribe: Option<bool>,
    
    /// Whether the server supports listing available resources
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCapabilities {
    /// Whether the server supports listing available prompts
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCapabilities {
    /// Whether the server supports sampling operations
    #[serde(rename = "supportsSampling")]
    pub supports_sampling: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientCapabilities {
    /// Client tool capabilities
    pub tools: Option<ClientToolCapabilities>,
    
    /// Client resource capabilities
    pub resources: Option<ClientResourceCapabilities>,
    
    /// Client prompt capabilities
    pub prompts: Option<ClientPromptCapabilities>,
    
    /// Client sampling capabilities
    pub sampling: Option<ClientSamplingCapabilities>,
    
    /// Root listing capability
    pub roots: Option<RootCapabilities>,
    
    /// Experimental capabilities
    pub experimental: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientToolCapabilities {
    /// Whether the client supports tool calling
    #[serde(rename = "supportsToolCalling")]
    pub supports_tool_calling: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientResourceCapabilities {
    /// Whether the client supports resource subscription
    pub subscribe: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientPromptCapabilities {
    /// Whether the client supports prompt templates
    #[serde(rename = "supportsPrompts")]
    pub supports_prompts: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientSamplingCapabilities {
    /// Whether the client supports sampling
    #[serde(rename = "supportsSampling")]
    pub supports_sampling: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootCapabilities {
    /// Whether the client supports listing roots
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

impl CapabilityManager {
    pub fn new() -> Self {
        Self {
            capabilities: ServerCapabilities::default(),
        }
    }
    
    pub async fn initialize(&self, config: &MCPServerConfig) -> Result<()> {
        info!("Initializing MCP server capabilities");
        
        // Capabilities are determined by config
        // This would normally validate and set up the actual capabilities
        
        Ok(())
    }
    
    pub fn get_capabilities(&self) -> &ServerCapabilities {
        &self.capabilities
    }
    
    pub fn supports_tools(&self) -> bool {
        self.capabilities.tools.is_some()
    }
    
    pub fn supports_resources(&self) -> bool {
        self.capabilities.resources.is_some()
    }
    
    pub fn supports_prompts(&self) -> bool {
        self.capabilities.prompts.is_some()
    }
    
    pub fn supports_sampling(&self) -> bool {
        self.capabilities.sampling.is_some()
    }
    
    /// Check if a client capability is compatible
    pub fn is_compatible_with_client(&self, client_caps: &ClientCapabilities) -> bool {
        // Check basic compatibility
        if self.supports_tools() && client_caps.tools.is_some() {
            return true;
        }
        
        if self.supports_resources() && client_caps.resources.is_some() {
            return true;
        }
        
        if self.supports_prompts() && client_caps.prompts.is_some() {
            return true;
        }
        
        if self.supports_sampling() && client_caps.sampling.is_some() {
            return true;
        }
        
        // At least one capability should overlap
        false
    }
    
    /// Get capability negotiation result
    pub fn negotiate_capabilities(&self, client_caps: &ClientCapabilities) -> NegotiationResult {
        let mut agreed_capabilities = AgreedCapabilities::default();
        
        // Negotiate tools
        if self.supports_tools() && client_caps.tools.is_some() {
            if let Some(client_tools) = &client_caps.tools {
                agreed_capabilities.tools_enabled = client_tools.supports_tool_calling.unwrap_or(false);
            }
        }
        
        // Negotiate resources
        if self.supports_resources() && client_caps.resources.is_some() {
            agreed_capabilities.resources_enabled = true;
        }
        
        // Negotiate prompts
        if self.supports_prompts() && client_caps.prompts.is_some() {
            agreed_capabilities.prompts_enabled = true;
        }
        
        // Negotiate sampling
        if self.supports_sampling() && client_caps.sampling.is_some() {
            if let Some(client_sampling) = &client_caps.sampling {
                agreed_capabilities.sampling_enabled = client_sampling.supports_sampling.unwrap_or(false);
            }
        }
        
        NegotiationResult {
            success: agreed_capabilities.has_any_capability(),
            agreed_capabilities,
            incompatible_features: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NegotiationResult {
    pub success: bool,
    pub agreed_capabilities: AgreedCapabilities,
    pub incompatible_features: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct AgreedCapabilities {
    pub tools_enabled: bool,
    pub resources_enabled: bool,
    pub prompts_enabled: bool,
    pub sampling_enabled: bool,
}

impl AgreedCapabilities {
    pub fn has_any_capability(&self) -> bool {
        self.tools_enabled || self.resources_enabled || self.prompts_enabled || self.sampling_enabled
    }
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            tools: Some(ToolCapabilities {
                list_changed: Some(true),
            }),
            resources: None, // Disabled for now
            prompts: None,   // Disabled for now
            sampling: Some(SamplingCapabilities {
                supports_sampling: Some(true),
            }),
            experimental: Some({
                let mut exp = HashMap::new();
                exp.insert(
                    "tektra_multimodal".to_string(),
                    serde_json::json!({
                        "version": "1.0",
                        "supports_vision": true,
                        "supports_audio": false,
                        "supports_documents": false
                    })
                );
                exp.insert(
                    "tektra_conversation".to_string(),
                    serde_json::json!({
                        "version": "1.0",
                        "supports_personas": true,
                        "supports_memory": true,
                        "supports_context_management": true
                    })
                );
                exp
            }),
        }
    }
}

impl Default for ClientCapabilities {
    fn default() -> Self {
        Self {
            tools: Some(ClientToolCapabilities {
                supports_tool_calling: Some(true),
            }),
            resources: None,
            prompts: None,
            sampling: Some(ClientSamplingCapabilities {
                supports_sampling: Some(true),
            }),
            roots: None,
            experimental: None,
        }
    }
}

/// Capability validation utilities
pub mod validation {
    use super::*;
    
    /// Validate that server capabilities are properly configured
    pub fn validate_server_capabilities(caps: &ServerCapabilities) -> Result<()> {
        // Ensure at least one capability is enabled
        if caps.tools.is_none() && caps.resources.is_none() && 
           caps.prompts.is_none() && caps.sampling.is_none() {
            return Err(anyhow::anyhow!("Server must have at least one capability enabled"));
        }
        
        Ok(())
    }
    
    /// Validate client capabilities format
    pub fn validate_client_capabilities(caps: &ClientCapabilities) -> Result<()> {
        // Basic format validation
        // In a real implementation, this would be more comprehensive
        Ok(())
    }
    
    /// Check if specific feature is supported
    pub fn is_feature_supported(
        server_caps: &ServerCapabilities,
        client_caps: &ClientCapabilities,
        feature: &str,
    ) -> bool {
        match feature {
            "tools" => {
                server_caps.tools.is_some() && 
                client_caps.tools.as_ref()
                    .and_then(|t| t.supports_tool_calling)
                    .unwrap_or(false)
            }
            "resources" => {
                server_caps.resources.is_some() && client_caps.resources.is_some()
            }
            "prompts" => {
                server_caps.prompts.is_some() && client_caps.prompts.is_some()
            }
            "sampling" => {
                server_caps.sampling.is_some() && 
                client_caps.sampling.as_ref()
                    .and_then(|s| s.supports_sampling)
                    .unwrap_or(false)
            }
            _ => false,
        }
    }
}

/// Extended capabilities for Tektra-specific features
pub mod tektra_capabilities {
    use super::*;
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TektraCapabilities {
        pub multimodal: Option<MultimodalCapabilities>,
        pub conversation: Option<ConversationCapabilities>,
        pub model_management: Option<ModelManagementCapabilities>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct MultimodalCapabilities {
        pub supports_vision: bool,
        pub supports_audio: bool,
        pub supports_documents: bool,
        pub supports_streaming: bool,
        pub max_image_size_mb: usize,
        pub supported_image_formats: Vec<String>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConversationCapabilities {
        pub supports_personas: bool,
        pub supports_memory: bool,
        pub supports_context_management: bool,
        pub supports_branching: bool,
        pub max_context_length: usize,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelManagementCapabilities {
        pub supports_model_switching: bool,
        pub supports_dynamic_loading: bool,
        pub supports_quantization: bool,
        pub available_model_types: Vec<String>,
    }
    
    impl Default for TektraCapabilities {
        fn default() -> Self {
            Self {
                multimodal: Some(MultimodalCapabilities {
                    supports_vision: true,
                    supports_audio: false,
                    supports_documents: false,
                    supports_streaming: true,
                    max_image_size_mb: 10,
                    supported_image_formats: vec![
                        "jpeg".to_string(),
                        "png".to_string(),
                        "gif".to_string(),
                        "webp".to_string(),
                    ],
                }),
                conversation: Some(ConversationCapabilities {
                    supports_personas: true,
                    supports_memory: true,
                    supports_context_management: true,
                    supports_branching: true,
                    max_context_length: 32768,
                }),
                model_management: Some(ModelManagementCapabilities {
                    supports_model_switching: true,
                    supports_dynamic_loading: true,
                    supports_quantization: true,
                    available_model_types: vec![
                        "mistral".to_string(),
                        "qwen".to_string(),
                        "llama".to_string(),
                        "phi".to_string(),
                    ],
                }),
            }
        }
    }
}