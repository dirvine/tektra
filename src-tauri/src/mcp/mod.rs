use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::{info, warn, error, debug};

pub mod server;
pub mod tools;
pub mod capabilities;
pub mod transport;

pub use server::*;
pub use tools::*;
pub use capabilities::*;
pub use transport::*;

use crate::inference::EnhancedModelRegistry;
use crate::multimodal::UnifiedMultimodalInterface;
use crate::conversation::EnhancedConversationManager;

/// Model Context Protocol server implementation for Tektra AI Assistant
pub struct TektraMCPServer {
    /// Core AI components
    model_registry: Arc<EnhancedModelRegistry>,
    multimodal_interface: Arc<UnifiedMultimodalInterface>,
    conversation_manager: Arc<EnhancedConversationManager>,
    
    /// MCP server components
    tool_registry: Arc<RwLock<ToolRegistry>>,
    capability_manager: Arc<CapabilityManager>,
    transport: Arc<Mutex<MCPTransport>>,
    
    /// Server state
    active_sessions: Arc<RwLock<HashMap<String, MCPSession>>>,
    server_info: MCPServerInfo,
    
    /// Configuration
    config: MCPServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerConfig {
    /// Server identification
    pub server_name: String,
    pub server_version: String,
    pub protocol_version: String,
    
    /// Capabilities
    pub enable_tools: bool,
    pub enable_resources: bool,
    pub enable_prompts: bool,
    pub enable_sampling: bool,
    
    /// Performance settings
    pub max_concurrent_sessions: usize,
    pub request_timeout_ms: u64,
    pub enable_request_logging: bool,
    
    /// Security settings
    pub require_authentication: bool,
    pub allowed_clients: Vec<String>,
    pub rate_limit_requests_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerInfo {
    pub name: String,
    pub version: String,
    pub protocol_version: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub homepage: Option<String>,
    pub capabilities: ServerCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPSession {
    pub session_id: String,
    pub client_info: ClientInfo,
    pub created_at: std::time::SystemTime,
    pub last_activity: std::time::SystemTime,
    pub request_count: u64,
    pub active_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientInfo {
    pub name: String,
    pub version: String,
    pub user_agent: Option<String>,
    pub capabilities: ClientCapabilities,
}

impl TektraMCPServer {
    /// Create a new Tektra MCP server
    pub async fn new(
        model_registry: Arc<EnhancedModelRegistry>,
        multimodal_interface: Arc<UnifiedMultimodalInterface>,
        conversation_manager: Arc<EnhancedConversationManager>,
    ) -> Result<Self> {
        info!("Initializing Tektra MCP server");
        
        let config = MCPServerConfig::default();
        let server_info = MCPServerInfo::default();
        
        // Initialize MCP components
        let tool_registry = Arc::new(RwLock::new(ToolRegistry::new()));
        let capability_manager = Arc::new(CapabilityManager::new());
        let transport = Arc::new(Mutex::new(MCPTransport::stdio()));
        
        // Register default tools
        let mut registry = tool_registry.write().await;
        Self::register_default_tools(&mut registry, &model_registry, &multimodal_interface).await?;
        drop(registry);
        
        Ok(Self {
            model_registry,
            multimodal_interface,
            conversation_manager,
            tool_registry,
            capability_manager,
            transport,
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            server_info,
            config,
        })
    }
    
    /// Start the MCP server
    pub async fn start(&self) -> Result<()> {
        info!("Starting Tektra MCP server on stdio transport");
        
        // Initialize server capabilities
        self.capability_manager.initialize(&self.config).await?;
        
        // Start transport layer
        let transport = self.transport.lock().await;
        transport.start().await?;
        drop(transport);
        
        // Start message processing loop
        self.run_message_loop().await
    }
    
    /// Main message processing loop
    async fn run_message_loop(&self) -> Result<()> {
        info!("Starting MCP server message processing loop");
        
        loop {
            // Read incoming message
            let message = {
                let transport = self.transport.lock().await;
                transport.read_message().await?
            };
            
            // Process message asynchronously
            let server = self.clone();
            tokio::spawn(async move {
                if let Err(e) = server.process_message(message).await {
                    error!("Error processing MCP message: {}", e);
                }
            });
        }
    }
    
    /// Process incoming MCP message
    async fn process_message(&self, message: MCPMessage) -> Result<()> {
        debug!("Processing MCP message: {:?}", message.method);
        
        let response = match message.method.as_str() {
            "initialize" => self.handle_initialize(message).await?,
            "tools/list" => self.handle_tools_list(message).await?,
            "tools/call" => self.handle_tools_call(message).await?,
            "resources/list" => self.handle_resources_list(message).await?,
            "resources/read" => self.handle_resources_read(message).await?,
            "prompts/list" => self.handle_prompts_list(message).await?,
            "prompts/get" => self.handle_prompts_get(message).await?,
            "sampling/createMessage" => self.handle_sampling_create_message(message).await?,
            _ => self.handle_unknown_method(message).await?,
        };
        
        // Send response
        let transport = self.transport.lock().await;
        transport.send_message(response).await?;
        
        Ok(())
    }
    
    /// Handle server initialization
    async fn handle_initialize(&self, message: MCPMessage) -> Result<MCPMessage> {
        info!("Handling MCP initialize request");
        
        let init_params: InitializeParams = serde_json::from_value(message.params.unwrap_or_default())?;
        
        // Create new session
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = MCPSession {
            session_id: session_id.clone(),
            client_info: init_params.client_info,
            created_at: std::time::SystemTime::now(),
            last_activity: std::time::SystemTime::now(),
            request_count: 0,
            active_tools: Vec::new(),
        };
        
        self.active_sessions.write().await.insert(session_id, session);
        
        let response = InitializeResponse {
            protocol_version: self.server_info.protocol_version.clone(),
            capabilities: self.server_info.capabilities.clone(),
            server_info: self.server_info.clone(),
            instructions: Some("Tektra AI Assistant MCP Server - Multimodal AI capabilities available".to_string()),
        };
        
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "initialize".to_string(),
            params: Some(serde_json::to_value(response)?),
            error: None,
        })
    }
    
    /// Handle tools list request
    async fn handle_tools_list(&self, message: MCPMessage) -> Result<MCPMessage> {
        debug!("Handling tools/list request");
        
        let registry = self.tool_registry.read().await;
        let tools = registry.list_tools().await;
        
        let response = ToolsListResponse { tools };
        
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "tools/list".to_string(),
            params: Some(serde_json::to_value(response)?),
            error: None,
        })
    }
    
    /// Handle tool call request
    async fn handle_tools_call(&self, message: MCPMessage) -> Result<MCPMessage> {
        debug!("Handling tools/call request");
        
        let call_params: ToolCallParams = serde_json::from_value(message.params.unwrap_or_default())?;
        
        let registry = self.tool_registry.read().await;
        let result = registry.execute_tool(
            &call_params.name,
            call_params.arguments.unwrap_or_default(),
            &self.model_registry,
            &self.multimodal_interface,
        ).await?;
        
        let response = ToolCallResponse {
            content: vec![result],
            is_error: false,
        };
        
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "tools/call".to_string(),
            params: Some(serde_json::to_value(response)?),
            error: None,
        })
    }
    
    /// Register default tools for Tektra
    async fn register_default_tools(
        registry: &mut ToolRegistry,
        model_registry: &Arc<EnhancedModelRegistry>,
        multimodal_interface: &Arc<UnifiedMultimodalInterface>,
    ) -> Result<()> {
        info!("Registering default Tektra MCP tools");
        
        // Text generation tool
        registry.register_tool(create_text_generation_tool()).await;
        
        // Image analysis tool
        registry.register_tool(create_image_analysis_tool()).await;
        
        // Model management tools
        registry.register_tool(create_model_list_tool()).await;
        registry.register_tool(create_model_switch_tool()).await;
        
        // Conversation tools
        registry.register_tool(create_conversation_start_tool()).await;
        registry.register_tool(create_conversation_continue_tool()).await;
        
        // Multimodal tools
        registry.register_tool(create_multimodal_analysis_tool()).await;
        registry.register_tool(create_image_comparison_tool()).await;
        
        // System tools
        registry.register_tool(create_system_status_tool()).await;
        registry.register_tool(create_performance_metrics_tool()).await;
        
        info!("Registered {} default tools", registry.tool_count().await);
        Ok(())
    }
    
    /// Handle unknown method
    async fn handle_unknown_method(&self, message: MCPMessage) -> Result<MCPMessage> {
        warn!("Unknown MCP method: {}", message.method);
        
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: message.method,
            params: None,
            error: Some(MCPError {
                code: -32601,
                message: "Method not found".to_string(),
                data: None,
            }),
        })
    }
    
    // Placeholder implementations for additional handlers
    async fn handle_resources_list(&self, message: MCPMessage) -> Result<MCPMessage> {
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "resources/list".to_string(),
            params: Some(serde_json::json!({"resources": []})),
            error: None,
        })
    }
    
    async fn handle_resources_read(&self, message: MCPMessage) -> Result<MCPMessage> {
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "resources/read".to_string(),
            params: Some(serde_json::json!({"contents": []})),
            error: None,
        })
    }
    
    async fn handle_prompts_list(&self, message: MCPMessage) -> Result<MCPMessage> {
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "prompts/list".to_string(),
            params: Some(serde_json::json!({"prompts": []})),
            error: None,
        })
    }
    
    async fn handle_prompts_get(&self, message: MCPMessage) -> Result<MCPMessage> {
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "prompts/get".to_string(),
            params: Some(serde_json::json!({"messages": []})),
            error: None,
        })
    }
    
    async fn handle_sampling_create_message(&self, message: MCPMessage) -> Result<MCPMessage> {
        Ok(MCPMessage {
            jsonrpc: "2.0".to_string(),
            id: message.id,
            method: "sampling/createMessage".to_string(),
            params: Some(serde_json::json!({"content": {"type": "text", "text": "Sampling not implemented"}})),
            error: None,
        })
    }
}

// Clone implementation for async spawning
impl Clone for TektraMCPServer {
    fn clone(&self) -> Self {
        Self {
            model_registry: self.model_registry.clone(),
            multimodal_interface: self.multimodal_interface.clone(),
            conversation_manager: self.conversation_manager.clone(),
            tool_registry: self.tool_registry.clone(),
            capability_manager: self.capability_manager.clone(),
            transport: self.transport.clone(),
            active_sessions: self.active_sessions.clone(),
            server_info: self.server_info.clone(),
            config: self.config.clone(),
        }
    }
}

// Default implementations
impl Default for MCPServerConfig {
    fn default() -> Self {
        Self {
            server_name: "tektra-mcp-server".to_string(),
            server_version: "0.1.0".to_string(),
            protocol_version: "2024-11-05".to_string(),
            enable_tools: true,
            enable_resources: false,
            enable_prompts: false,
            enable_sampling: true,
            max_concurrent_sessions: 10,
            request_timeout_ms: 30000,
            enable_request_logging: true,
            require_authentication: false,
            allowed_clients: Vec::new(),
            rate_limit_requests_per_minute: 60,
        }
    }
}

impl Default for MCPServerInfo {
    fn default() -> Self {
        Self {
            name: "Tektra AI Assistant MCP Server".to_string(),
            version: "0.1.0".to_string(),
            protocol_version: "2024-11-05".to_string(),
            description: "Model Context Protocol server for Tektra AI Assistant providing multimodal AI capabilities".to_string(),
            author: "Tektra Development Team".to_string(),
            license: "MIT".to_string(),
            homepage: Some("https://github.com/tektra/tektra".to_string()),
            capabilities: ServerCapabilities::default(),
        }
    }
}

// Basic MCP message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPMessage {
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    pub params: Option<serde_json::Value>,
    pub error: Option<MCPError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    pub client_info: ClientInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitializeResponse {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: MCPServerInfo,
    pub instructions: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsListResponse {
    pub tools: Vec<ToolDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    pub arguments: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub content: Vec<ToolResult>,
    #[serde(rename = "isError")]
    pub is_error: bool,
}