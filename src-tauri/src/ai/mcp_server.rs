use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// MCP (Model Context Protocol) Server Implementation
/// Provides a standardized interface for AI systems to interact with data sources

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPServerInfo {
    pub name: String,
    pub version: String,
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerCapabilities {
    pub resources: ResourceCapabilities,
    pub tools: ToolCapabilities,
    pub prompts: PromptCapabilities,
    pub sampling: Option<SamplingCapabilities>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    pub subscribe: bool,
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCapabilities {
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptCapabilities {
    pub list_changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingCapabilities {
    pub max_tokens: usize,
    pub temperature_range: (f32, f32),
}

/// Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub uri: String,
    pub name: String,
    pub description: String,
    pub mime_type: String,
    pub metadata: HashMap<String, Value>,
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Value,
}

/// Prompt template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPrompt {
    pub name: String,
    pub description: String,
    pub arguments: Vec<PromptArgument>,
    pub template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptArgument {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub default: Option<String>,
}

/// Tool execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequest {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ContentPart>,
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: Option<String>,
    pub data: Option<String>, // Base64 encoded
    pub mime_type: Option<String>,
}

/// MCP Server trait
#[async_trait]
pub trait MCPServer: Send + Sync {
    /// Get server information
    async fn get_info(&self) -> Result<MCPServerInfo>;
    
    /// List available resources
    async fn list_resources(&self) -> Result<Vec<Resource>>;
    
    /// Read a resource
    async fn read_resource(&self, uri: &str) -> Result<String>;
    
    /// Subscribe to resource changes
    async fn subscribe_resource(&self, uri: &str) -> Result<()>;
    
    /// List available tools
    async fn list_tools(&self) -> Result<Vec<Tool>>;
    
    /// Call a tool
    async fn call_tool(&self, request: ToolRequest) -> Result<ToolResult>;
    
    /// List available prompts
    async fn list_prompts(&self) -> Result<Vec<MCPPrompt>>;
    
    /// Get a prompt
    async fn get_prompt(&self, name: &str, arguments: HashMap<String, String>) -> Result<String>;
}

/// Tektra MCP Server Implementation
#[derive(Clone)]
pub struct TektraMCPServer {
    resources: Arc<RwLock<HashMap<String, Resource>>>,
    tools: Arc<RwLock<HashMap<String, Tool>>>,
    prompts: Arc<RwLock<HashMap<String, MCPPrompt>>>,
    subscriptions: Arc<RwLock<HashMap<String, Vec<String>>>>, // resource_uri -> subscriber_ids
}

impl TektraMCPServer {
    pub fn new() -> Self {
        let server = Self {
            resources: Arc::new(RwLock::new(HashMap::new())),
            tools: Arc::new(RwLock::new(HashMap::new())),
            prompts: Arc::new(RwLock::new(HashMap::new())),
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        };
        
        // Initialize with default resources, tools, and prompts
        let server_clone = server.clone();
        tokio::spawn(async move {
            server_clone.initialize_defaults().await;
        });
        
        server
    }
    
    async fn initialize_defaults(&self) {
        // Add default resources
        self.add_resource(Resource {
            uri: "file:///workspace/documents".to_string(),
            name: "Workspace Documents".to_string(),
            description: "Access to workspace document files".to_string(),
            mime_type: "application/x-directory".to_string(),
            metadata: HashMap::new(),
        }).await;
        
        self.add_resource(Resource {
            uri: "memory://conversation".to_string(),
            name: "Conversation Memory".to_string(),
            description: "Current conversation context and history".to_string(),
            mime_type: "application/json".to_string(),
            metadata: HashMap::new(),
        }).await;
        
        // Add default tools
        self.add_tool(Tool {
            name: "search_documents".to_string(),
            description: "Search through indexed documents".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 5
                    }
                },
                "required": ["query"]
            }),
            output_schema: json!({
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "document": {"type": "string"},
                        "snippet": {"type": "string"},
                        "relevance": {"type": "number"}
                    }
                }
            }),
        }).await;
        
        self.add_tool(Tool {
            name: "analyze_image".to_string(),
            description: "Analyze an image using vision models".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Analysis prompt"
                    }
                },
                "required": ["image_data"]
            }),
            output_schema: json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "objects": {"type": "array", "items": {"type": "string"}},
                    "text": {"type": "string"}
                }
            }),
        }).await;
        
        // Add default prompts
        self.add_prompt(MCPPrompt {
            name: "code_review".to_string(),
            description: "Review code for quality and suggest improvements".to_string(),
            arguments: vec![
                PromptArgument {
                    name: "code".to_string(),
                    description: "The code to review".to_string(),
                    required: true,
                    default: None,
                },
                PromptArgument {
                    name: "language".to_string(),
                    description: "Programming language".to_string(),
                    required: false,
                    default: Some("auto-detect".to_string()),
                },
            ],
            template: r#"Please review the following {{language}} code and provide feedback on:
1. Code quality and best practices
2. Potential bugs or issues
3. Performance considerations
4. Suggested improvements

Code:
```{{language}}
{{code}}
```

Provide a detailed analysis with specific examples and recommendations."#.to_string(),
        }).await;
        
        self.add_prompt(MCPPrompt {
            name: "document_qa".to_string(),
            description: "Answer questions based on document context".to_string(),
            arguments: vec![
                PromptArgument {
                    name: "question".to_string(),
                    description: "The question to answer".to_string(),
                    required: true,
                    default: None,
                },
                PromptArgument {
                    name: "context".to_string(),
                    description: "Document context".to_string(),
                    required: true,
                    default: None,
                },
            ],
            template: r#"Based on the following document context, please answer the question accurately and concisely.

Context:
{{context}}

Question: {{question}}

Answer:"#.to_string(),
        }).await;
    }
    
    pub async fn add_resource(&self, resource: Resource) {
        let mut resources = self.resources.write().await;
        resources.insert(resource.uri.clone(), resource);
    }
    
    pub async fn add_tool(&self, tool: Tool) {
        let mut tools = self.tools.write().await;
        tools.insert(tool.name.clone(), tool);
    }
    
    pub async fn add_prompt(&self, prompt: MCPPrompt) {
        let mut prompts = self.prompts.write().await;
        prompts.insert(prompt.name.clone(), prompt);
    }
}

#[async_trait]
impl MCPServer for TektraMCPServer {
    async fn get_info(&self) -> Result<MCPServerInfo> {
        Ok(MCPServerInfo {
            name: "Tektra MCP Server".to_string(),
            version: "0.1.0".to_string(),
            protocol_version: "2024-11-05".to_string(),
            capabilities: ServerCapabilities {
                resources: ResourceCapabilities {
                    subscribe: true,
                    list_changed: true,
                },
                tools: ToolCapabilities {
                    list_changed: true,
                },
                prompts: PromptCapabilities {
                    list_changed: true,
                },
                sampling: Some(SamplingCapabilities {
                    max_tokens: 32000,
                    temperature_range: (0.0, 2.0),
                }),
            },
        })
    }
    
    async fn list_resources(&self) -> Result<Vec<Resource>> {
        let resources = self.resources.read().await;
        Ok(resources.values().cloned().collect())
    }
    
    async fn read_resource(&self, uri: &str) -> Result<String> {
        let resources = self.resources.read().await;
        
        if let Some(_resource) = resources.get(uri) {
            // Handle different resource types
            if uri.starts_with("file://") {
                // Read file resource
                let path = uri.strip_prefix("file://").unwrap();
                tokio::fs::read_to_string(path).await
                    .map_err(|e| anyhow::anyhow!("Failed to read file: {}", e))
            } else if uri.starts_with("memory://") {
                // Read memory resource
                match uri {
                    "memory://conversation" => {
                        Ok(json!({
                            "messages": [],
                            "context": {}
                        }).to_string())
                    }
                    _ => Err(anyhow::anyhow!("Unknown memory resource")),
                }
            } else {
                Err(anyhow::anyhow!("Unsupported resource type"))
            }
        } else {
            Err(anyhow::anyhow!("Resource not found: {}", uri))
        }
    }
    
    async fn subscribe_resource(&self, uri: &str) -> Result<()> {
        let resources = self.resources.read().await;
        
        if resources.contains_key(uri) {
            let mut subscriptions = self.subscriptions.write().await;
            let subscriber_id = uuid::Uuid::new_v4().to_string();
            
            subscriptions
                .entry(uri.to_string())
                .or_insert_with(Vec::new)
                .push(subscriber_id);
            
            info!("Subscribed to resource: {}", uri);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Resource not found: {}", uri))
        }
    }
    
    async fn list_tools(&self) -> Result<Vec<Tool>> {
        let tools = self.tools.read().await;
        Ok(tools.values().cloned().collect())
    }
    
    async fn call_tool(&self, request: ToolRequest) -> Result<ToolResult> {
        let tools = self.tools.read().await;
        
        if let Some(_tool) = tools.get(&request.name) {
            match request.name.as_str() {
                "search_documents" => {
                    let _query = request.arguments.get("query")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing query parameter"))?;
                    
                    let _limit = request.arguments.get("limit")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(5) as usize;
                    
                    // Mock search results
                    let results = vec![
                        json!({
                            "document": "technical_spec.md",
                            "snippet": "The multimodal AI system processes text, images, and audio...",
                            "relevance": 0.95
                        }),
                        json!({
                            "document": "sample.txt",
                            "snippet": "Machine learning is a subset of artificial intelligence...",
                            "relevance": 0.87
                        }),
                    ];
                    
                    Ok(ToolResult {
                        content: vec![ContentPart {
                            content_type: "text".to_string(),
                            text: Some(serde_json::to_string_pretty(&results)?),
                            data: None,
                            mime_type: Some("application/json".to_string()),
                        }],
                        is_error: false,
                    })
                }
                "analyze_image" => {
                    let _image_data = request.arguments.get("image_data")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| anyhow::anyhow!("Missing image_data parameter"))?;
                    
                    let prompt = request.arguments.get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("Describe this image");
                    
                    // Mock image analysis
                    let analysis = json!({
                        "description": format!("Analysis based on prompt: {}", prompt),
                        "objects": ["rectangle", "circle", "triangle"],
                        "text": "Test Image for Multimodal AI"
                    });
                    
                    Ok(ToolResult {
                        content: vec![ContentPart {
                            content_type: "text".to_string(),
                            text: Some(serde_json::to_string_pretty(&analysis)?),
                            data: None,
                            mime_type: Some("application/json".to_string()),
                        }],
                        is_error: false,
                    })
                }
                _ => Err(anyhow::anyhow!("Tool not implemented: {}", request.name)),
            }
        } else {
            Err(anyhow::anyhow!("Tool not found: {}", request.name))
        }
    }
    
    async fn list_prompts(&self) -> Result<Vec<MCPPrompt>> {
        let prompts = self.prompts.read().await;
        Ok(prompts.values().cloned().collect())
    }
    
    async fn get_prompt(&self, name: &str, arguments: HashMap<String, String>) -> Result<String> {
        let prompts = self.prompts.read().await;
        
        if let Some(prompt) = prompts.get(name) {
            let mut result = prompt.template.clone();
            
            // Validate required arguments
            for arg in &prompt.arguments {
                if arg.required && !arguments.contains_key(&arg.name) {
                    return Err(anyhow::anyhow!("Missing required argument: {}", arg.name));
                }
            }
            
            // Replace template variables
            for (key, value) in &arguments {
                result = result.replace(&format!("{{{{{}}}}}", key), value);
            }
            
            // Replace missing optional arguments with defaults
            for arg in &prompt.arguments {
                if !arg.required && !arguments.contains_key(&arg.name) {
                    if let Some(default) = &arg.default {
                        result = result.replace(&format!("{{{{{}}}}}", arg.name), default);
                    }
                }
            }
            
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Prompt not found: {}", name))
        }
    }
}

/// MCP Client for connecting to external MCP servers
pub struct MCPClient {
    server_url: String,
    client: reqwest::Client,
}

impl MCPClient {
    pub fn new(server_url: String) -> Self {
        Self {
            server_url,
            client: reqwest::Client::new(),
        }
    }
    
    pub async fn initialize(&self) -> Result<MCPServerInfo> {
        let response = self.client
            .post(&format!("{}/initialize", self.server_url))
            .json(&json!({}))
            .send()
            .await?;
        
        let info: MCPServerInfo = response.json().await?;
        Ok(info)
    }
    
    pub async fn list_resources(&self) -> Result<Vec<Resource>> {
        let response = self.client
            .post(&format!("{}/resources/list", self.server_url))
            .json(&json!({}))
            .send()
            .await?;
        
        let resources: Vec<Resource> = response.json().await?;
        Ok(resources)
    }
    
    pub async fn call_tool(&self, request: ToolRequest) -> Result<ToolResult> {
        let response = self.client
            .post(&format!("{}/tools/call", self.server_url))
            .json(&request)
            .send()
            .await?;
        
        let result: ToolResult = response.json().await?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mcp_server_creation() {
        let server = TektraMCPServer::new();
        
        // Wait for initialization
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let info = server.get_info().await.unwrap();
        assert_eq!(info.name, "Tektra MCP Server");
        assert_eq!(info.protocol_version, "2024-11-05");
        assert!(info.capabilities.resources.subscribe);
    }
    
    #[tokio::test]
    async fn test_resource_management() {
        let server = TektraMCPServer::new();
        
        server.add_resource(Resource {
            uri: "test://resource".to_string(),
            name: "Test Resource".to_string(),
            description: "A test resource".to_string(),
            mime_type: "text/plain".to_string(),
            metadata: HashMap::new(),
        }).await;
        
        let resources = server.list_resources().await.unwrap();
        assert!(resources.iter().any(|r| r.uri == "test://resource"));
    }
    
    #[tokio::test]
    async fn test_prompt_rendering() {
        let server = TektraMCPServer::new();
        
        // Wait for defaults to load
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let mut args = HashMap::new();
        args.insert("code".to_string(), "fn main() { println!(\"Hello\"); }".to_string());
        args.insert("language".to_string(), "rust".to_string());
        
        let rendered = server.get_prompt("code_review", args).await.unwrap();
        assert!(rendered.contains("fn main()"));
        assert!(rendered.contains("rust"));
        assert!(rendered.contains("Please review"));
    }
}