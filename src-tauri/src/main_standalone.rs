// Standalone desktop application without Tauri
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, error};
use tracing_subscriber;

mod ai;
mod audio;
mod vision;
mod robot;
mod state;
mod standalone;

use state::AppState;
use standalone::{WebServer, create_window};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("Starting Tektra AI Assistant (Standalone)");

    // Create a mock app handle for compatibility
    let mock_handle = MockAppHandle::new();
    
    // Initialize application state
    let app_state = Arc::new(RwLock::new(AppState::new(mock_handle)?));
    
    // Start web server in background
    let web_server = WebServer::new();
    let server_url = web_server.get_url();
    
    tokio::spawn(async move {
        if let Err(e) = web_server.start().await {
            error!("Web server error: {}", e);
        }
    });
    
    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    info!("Opening window at {}", server_url);
    
    // Create and show window
    create_window(&server_url)?;
    
    Ok(())
}

// Mock AppHandle for compatibility with existing code
#[derive(Clone)]
pub struct MockAppHandle {
    // We'll implement basic event emission to stdout for now
}

impl MockAppHandle {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn emit<S: serde::Serialize>(&self, event: &str, payload: S) -> Result<(), Box<dyn std::error::Error>> {
        let payload_json = serde_json::to_string(&payload)?;
        println!("EVENT: {} -> {}", event, payload_json);
        Ok(())
    }
    
    pub fn clone(&self) -> Self {
        Self {}
    }
}