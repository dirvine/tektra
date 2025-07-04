use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, debug, error};

use crate::inference::EnhancedModelRegistry;
use crate::multimodal::{UnifiedMultimodalInterface, ImageAnalysisRequest, ImageAnalysisType};

/// Tool registry for managing MCP tools
pub struct ToolRegistry {
    tools: HashMap<String, ToolDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: ToolInputSchema,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputSchema {
    #[serde(rename = "type")]
    pub schema_type: String,
    pub properties: HashMap<String, PropertyDefinition>,
    pub required: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyDefinition {
    #[serde(rename = "type")]
    pub property_type: String,
    pub description: String,
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    #[serde(rename = "type")]
    pub result_type: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }
    
    pub async fn register_tool(&mut self, tool: ToolDefinition) {
        info!("Registering MCP tool: {}", tool.name);
        self.tools.insert(tool.name.clone(), tool);
    }
    
    pub async fn list_tools(&self) -> Vec<ToolDefinition> {
        self.tools.values().cloned().collect()
    }
    
    pub async fn tool_count(&self) -> usize {
        self.tools.len()
    }
    
    pub async fn execute_tool(
        &self,
        tool_name: &str,
        arguments: Value,
        model_registry: &Arc<EnhancedModelRegistry>,
        multimodal_interface: &Arc<UnifiedMultimodalInterface>,
    ) -> Result<ToolResult> {
        debug!("Executing tool: {} with args: {:?}", tool_name, arguments);
        
        match tool_name {
            "text_generation" => execute_text_generation(arguments, model_registry).await,
            "image_analysis" => execute_image_analysis(arguments, multimodal_interface).await,
            "model_list" => execute_model_list(arguments, model_registry).await,
            "model_switch" => execute_model_switch(arguments, model_registry).await,
            "conversation_start" => execute_conversation_start(arguments).await,
            "conversation_continue" => execute_conversation_continue(arguments).await,
            "multimodal_analysis" => execute_multimodal_analysis(arguments, multimodal_interface).await,
            "image_comparison" => execute_image_comparison(arguments, multimodal_interface).await,
            "system_status" => execute_system_status(arguments, model_registry).await,
            "performance_metrics" => execute_performance_metrics(arguments, multimodal_interface).await,
            _ => {
                error!("Unknown tool: {}", tool_name);
                Ok(ToolResult {
                    result_type: "error".to_string(),
                    text: format!("Unknown tool: {}", tool_name),
                    data: None,
                })
            }
        }
    }
}

// Tool creation functions

pub fn create_text_generation_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "prompt".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "The text prompt to generate completion for".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "max_tokens".to_string(),
        PropertyDefinition {
            property_type: "integer".to_string(),
            description: "Maximum number of tokens to generate".to_string(),
            enum_values: None,
            default: Some(Value::Number(2048.into())),
        },
    );
    
    properties.insert(
        "temperature".to_string(),
        PropertyDefinition {
            property_type: "number".to_string(),
            description: "Sampling temperature between 0.0 and 2.0".to_string(),
            enum_values: None,
            default: Some(Value::Number(serde_json::Number::from_f64(0.7).unwrap())),
        },
    );
    
    ToolDefinition {
        name: "text_generation".to_string(),
        description: "Generate text completion using the active AI model".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["prompt".to_string()],
        },
    }
}

pub fn create_image_analysis_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "image_data".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Base64 encoded image data".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "analysis_type".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Type of image analysis to perform".to_string(),
            enum_values: Some(vec![
                "general".to_string(),
                "detailed".to_string(),
                "ocr".to_string(),
                "scene".to_string(),
                "technical".to_string(),
            ]),
            default: Some(Value::String("general".to_string())),
        },
    );
    
    properties.insert(
        "custom_prompt".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Custom prompt for image analysis".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    ToolDefinition {
        name: "image_analysis".to_string(),
        description: "Analyze an image using multimodal AI capabilities".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["image_data".to_string()],
        },
    }
}

pub fn create_model_list_tool() -> ToolDefinition {
    ToolDefinition {
        name: "model_list".to_string(),
        description: "List all available AI models and their capabilities".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
    }
}

pub fn create_model_switch_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "model_id".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "ID of the model to switch to".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    ToolDefinition {
        name: "model_switch".to_string(),
        description: "Switch to a different AI model".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["model_id".to_string()],
        },
    }
}

pub fn create_conversation_start_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "session_id".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Unique identifier for the conversation session".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "persona".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "AI persona to use for the conversation".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    ToolDefinition {
        name: "conversation_start".to_string(),
        description: "Start a new conversation session with the AI".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["session_id".to_string()],
        },
    }
}

pub fn create_conversation_continue_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "session_id".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Conversation session identifier".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "message".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "User message to send to the AI".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    ToolDefinition {
        name: "conversation_continue".to_string(),
        description: "Continue an existing conversation session".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["session_id".to_string(), "message".to_string()],
        },
    }
}

pub fn create_multimodal_analysis_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "session_id".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Processing session identifier".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "content".to_string(),
        PropertyDefinition {
            property_type: "object".to_string(),
            description: "Multimodal content including text, images, audio, documents".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    ToolDefinition {
        name: "multimodal_analysis".to_string(),
        description: "Analyze complex multimodal input with intelligent processing".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["session_id".to_string(), "content".to_string()],
        },
    }
}

pub fn create_image_comparison_tool() -> ToolDefinition {
    let mut properties = HashMap::new();
    
    properties.insert(
        "session_id".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Processing session identifier".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "images".to_string(),
        PropertyDefinition {
            property_type: "array".to_string(),
            description: "Array of base64 encoded images to compare".to_string(),
            enum_values: None,
            default: None,
        },
    );
    
    properties.insert(
        "comparison_type".to_string(),
        PropertyDefinition {
            property_type: "string".to_string(),
            description: "Type of comparison to perform".to_string(),
            enum_values: Some(vec![
                "similarity".to_string(),
                "quality".to_string(),
                "content".to_string(),
                "style".to_string(),
            ]),
            default: Some(Value::String("similarity".to_string())),
        },
    );
    
    ToolDefinition {
        name: "image_comparison".to_string(),
        description: "Compare multiple images with detailed analysis".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties,
            required: vec!["session_id".to_string(), "images".to_string()],
        },
    }
}

pub fn create_system_status_tool() -> ToolDefinition {
    ToolDefinition {
        name: "system_status".to_string(),
        description: "Get current system status and health information".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
    }
}

pub fn create_performance_metrics_tool() -> ToolDefinition {
    ToolDefinition {
        name: "performance_metrics".to_string(),
        description: "Get performance metrics and usage statistics".to_string(),
        input_schema: ToolInputSchema {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: vec![],
        },
    }
}

// Tool execution functions

async fn execute_text_generation(
    arguments: Value,
    model_registry: &Arc<EnhancedModelRegistry>,
) -> Result<ToolResult> {
    let prompt = arguments["prompt"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing prompt parameter"))?;
    
    let max_tokens = arguments["max_tokens"].as_u64().unwrap_or(2048) as usize;
    let temperature = arguments["temperature"].as_f64().unwrap_or(0.7) as f32;
    
    info!("Generating text with prompt length: {}", prompt.len());
    
    // Create text input
    let input = crate::inference::MultimodalInput::Text(prompt.to_string());
    
    // Generate response
    let response = model_registry.generate(input).await?;
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: response.text,
        data: Some(serde_json::json!({
            "tokens_generated": response.tokens.len(),
            "finish_reason": format!("{:?}", response.finish_reason),
            "usage": response.usage
        })),
    })
}

async fn execute_image_analysis(
    arguments: Value,
    multimodal_interface: &Arc<UnifiedMultimodalInterface>,
) -> Result<ToolResult> {
    let image_data_b64 = arguments["image_data"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing image_data parameter"))?;
    
    let analysis_type_str = arguments["analysis_type"].as_str().unwrap_or("general");
    let analysis_type = match analysis_type_str {
        "general" => ImageAnalysisType::General,
        "detailed" => ImageAnalysisType::Detailed,
        "ocr" => ImageAnalysisType::OCR,
        "scene" => ImageAnalysisType::Scene,
        "technical" => ImageAnalysisType::Technical,
        _ => ImageAnalysisType::General,
    };
    
    let custom_prompt = arguments["custom_prompt"].as_str().map(|s| s.to_string());
    
    // Decode base64 image
    let image_data = base64::decode(image_data_b64)
        .map_err(|e| anyhow::anyhow!("Invalid base64 image data: {}", e))?;
    
    info!("Analyzing image of {} bytes", image_data.len());
    
    // Create session for analysis
    let session_id = format!("mcp_session_{}", uuid::Uuid::new_v4());
    multimodal_interface.start_session(session_id.clone(), None).await?;
    
    // Perform analysis
    let analysis_request = ImageAnalysisRequest {
        analysis_type,
        custom_prompt,
        ocr_options: None,
        processing_options: None,
    };
    
    let result = multimodal_interface.analyze_image_comprehensive(
        &session_id,
        &image_data,
        analysis_request,
    ).await?;
    
    // End session
    multimodal_interface.end_session(&session_id).await?;
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: result.formatted_output,
        data: Some(serde_json::json!({
            "confidence": result.original_result.confidence_score,
            "processing_time_ms": result.original_result.processing_time_ms,
            "model_used": result.original_result.model_used,
            "image_metadata": result.original_result.image_metadata
        })),
    })
}

async fn execute_model_list(
    _arguments: Value,
    model_registry: &Arc<EnhancedModelRegistry>,
) -> Result<ToolResult> {
    info!("Listing available models");
    
    let models = model_registry.list_models().await;
    
    let model_info: Vec<Value> = models.iter().map(|model| {
        serde_json::json!({
            "id": model.id,
            "name": model.name,
            "description": model.description,
            "supports_vision": model.supports_vision,
            "supports_audio": model.supports_audio,
            "context_length": model.context_length,
            "quantization": model.quantization,
            "model_type": format!("{:?}", model.model_type)
        })
    }).collect();
    
    let active_model = model_registry.get_active_model_id().await;
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Found {} available models. Active model: {:?}", models.len(), active_model),
        data: Some(serde_json::json!({
            "models": model_info,
            "active_model": active_model,
            "total_count": models.len()
        })),
    })
}

async fn execute_model_switch(
    arguments: Value,
    model_registry: &Arc<EnhancedModelRegistry>,
) -> Result<ToolResult> {
    let model_id = arguments["model_id"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing model_id parameter"))?;
    
    info!("Switching to model: {}", model_id);
    
    model_registry.switch_model(model_id).await?;
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Successfully switched to model: {}", model_id),
        data: Some(serde_json::json!({
            "previous_model": model_registry.get_active_model_id().await,
            "new_model": model_id
        })),
    })
}

async fn execute_conversation_start(arguments: Value) -> Result<ToolResult> {
    let session_id = arguments["session_id"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing session_id parameter"))?;
    
    let persona = arguments["persona"].as_str();
    
    info!("Starting conversation session: {}", session_id);
    
    // This would integrate with the conversation manager
    // For now, return a simple success response
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Started conversation session: {}", session_id),
        data: Some(serde_json::json!({
            "session_id": session_id,
            "persona": persona,
            "created_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        })),
    })
}

async fn execute_conversation_continue(arguments: Value) -> Result<ToolResult> {
    let session_id = arguments["session_id"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing session_id parameter"))?;
    
    let message = arguments["message"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing message parameter"))?;
    
    info!("Continuing conversation in session: {}", session_id);
    
    // This would process the message through the conversation manager
    // For now, return a simple echo response
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Received message in session {}: {}", session_id, message),
        data: Some(serde_json::json!({
            "session_id": session_id,
            "message_length": message.len(),
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        })),
    })
}

async fn execute_multimodal_analysis(
    arguments: Value,
    multimodal_interface: &Arc<UnifiedMultimodalInterface>,
) -> Result<ToolResult> {
    let session_id = arguments["session_id"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing session_id parameter"))?;
    
    info!("Executing multimodal analysis for session: {}", session_id);
    
    // This would perform complex multimodal analysis
    // For now, return a placeholder response
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Multimodal analysis completed for session: {}", session_id),
        data: Some(serde_json::json!({
            "session_id": session_id,
            "analysis_type": "multimodal",
            "components_analyzed": 0
        })),
    })
}

async fn execute_image_comparison(
    arguments: Value,
    multimodal_interface: &Arc<UnifiedMultimodalInterface>,
) -> Result<ToolResult> {
    let session_id = arguments["session_id"].as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing session_id parameter"))?;
    
    let images_array = arguments["images"].as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing images parameter"))?;
    
    info!("Comparing {} images for session: {}", images_array.len(), session_id);
    
    if images_array.len() < 2 {
        return Ok(ToolResult {
            result_type: "error".to_string(),
            text: "At least 2 images required for comparison".to_string(),
            data: None,
        });
    }
    
    // This would perform actual image comparison
    // For now, return a placeholder response
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Compared {} images successfully", images_array.len()),
        data: Some(serde_json::json!({
            "session_id": session_id,
            "images_compared": images_array.len(),
            "comparison_type": arguments["comparison_type"].as_str().unwrap_or("similarity")
        })),
    })
}

async fn execute_system_status(
    _arguments: Value,
    model_registry: &Arc<EnhancedModelRegistry>,
) -> Result<ToolResult> {
    info!("Getting system status");
    
    let active_model = model_registry.get_active_model_id().await;
    let available_models = model_registry.list_models().await.len();
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: "System is operational".to_string(),
        data: Some(serde_json::json!({
            "status": "healthy",
            "active_model": active_model,
            "available_models": available_models,
            "uptime_seconds": 0, // Would calculate actual uptime
            "memory_usage": "N/A", // Would get actual memory usage
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        })),
    })
}

async fn execute_performance_metrics(
    _arguments: Value,
    multimodal_interface: &Arc<UnifiedMultimodalInterface>,
) -> Result<ToolResult> {
    info!("Getting performance metrics");
    
    let metrics = multimodal_interface.get_interface_metrics().await;
    
    Ok(ToolResult {
        result_type: "text".to_string(),
        text: format!("Processed {} total requests with {:.1}% success rate", 
                     metrics.total_requests, 
                     (metrics.successful_requests as f64 / metrics.total_requests.max(1) as f64) * 100.0),
        data: Some(serde_json::json!({
            "total_requests": metrics.total_requests,
            "successful_requests": metrics.successful_requests,
            "failed_requests": metrics.failed_requests,
            "average_response_time_ms": metrics.average_response_time_ms,
            "active_sessions_count": metrics.active_sessions_count,
            "cache_hit_rate": metrics.cache_hit_rate,
            "requests_by_type": metrics.requests_by_type
        })),
    })
}