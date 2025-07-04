use anyhow::Result;
use std::sync::Arc;
use tracing::{info, error};
use tracing_subscriber;

use tektra::{
    EnhancedModelRegistry, ConversationManager,
    // TektraMCPServer, MCPServerConfig,  // Temporarily disabled
    // EnhancedConversationManager,  // Temporarily disabled
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    info!("Starting Tektra MCP Server");

    // Initialize core components
    let model_registry = Arc::new(
        EnhancedModelRegistry::new()
            .await
            .map_err(|e| {
                error!("Failed to initialize model registry: {}", e);
                e
            })?
    );

    // Temporarily disable multimodal interface
    // let multimodal_interface = Arc::new(
    //     UnifiedMultimodalInterface::new(model_registry.clone())
    //         .await
    //         .map_err(|e| {
    //             error!("Failed to initialize multimodal interface: {}", e);
    //             e
    //         })?
    // );

    let conversation_manager = Arc::new(
        ConversationManager::new(None)
        .map_err(|e| {
            error!("Failed to initialize conversation manager: {}", e);
            e
        })?
    );

    // Create MCP server with default configuration (temporarily disabled)
    // let server = TektraMCPServer::new();

    info!("Tektra MCP Server initialized successfully");

    // Start the server (temporarily disabled)
    // server.start().await.map_err(|e| {
    //     error!("Failed to start MCP server: {}", e);
    //     e
    // })?;

    info!("MCP server would start here - currently disabled for compilation fixes");

    Ok(())
}