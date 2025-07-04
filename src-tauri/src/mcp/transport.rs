use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use super::MCPMessage;

/// MCP transport layer for communication
pub struct MCPTransport {
    transport_type: TransportType,
    message_queue: Arc<Mutex<VecDeque<MCPMessage>>>,
    is_running: Arc<Mutex<bool>>,
}

#[derive(Debug, Clone)]
pub enum TransportType {
    Stdio,
    Tcp { port: u16 },
    WebSocket { port: u16 },
    Ipc { path: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportConfig {
    pub transport_type: String,
    pub buffer_size: usize,
    pub timeout_ms: u64,
    pub enable_compression: bool,
    pub max_message_size: usize,
}

impl MCPTransport {
    /// Create stdio transport (most common for MCP)
    pub fn stdio() -> Self {
        Self {
            transport_type: TransportType::Stdio,
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Create TCP transport
    pub fn tcp(port: u16) -> Self {
        Self {
            transport_type: TransportType::Tcp { port },
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Create WebSocket transport
    pub fn websocket(port: u16) -> Self {
        Self {
            transport_type: TransportType::WebSocket { port },
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Create IPC transport
    pub fn ipc(path: String) -> Self {
        Self {
            transport_type: TransportType::Ipc { path },
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// Start the transport layer
    pub async fn start(&self) -> Result<()> {
        let mut running = self.is_running.lock().await;
        if *running {
            return Ok(());
        }
        
        match &self.transport_type {
            TransportType::Stdio => {
                info!("Starting MCP stdio transport");
                *running = true;
                Ok(())
            }
            TransportType::Tcp { port } => {
                info!("Starting MCP TCP transport on port {}", port);
                *running = true;
                self.start_tcp_server(*port).await
            }
            TransportType::WebSocket { port } => {
                info!("Starting MCP WebSocket transport on port {}", port);
                *running = true;
                self.start_websocket_server(*port).await
            }
            TransportType::Ipc { path } => {
                info!("Starting MCP IPC transport at {}", path);
                *running = true;
                self.start_ipc_server(path.clone()).await
            }
        }
    }
    
    /// Stop the transport layer
    pub async fn stop(&self) -> Result<()> {
        let mut running = self.is_running.lock().await;
        *running = false;
        info!("Stopped MCP transport");
        Ok(())
    }
    
    /// Read a message from the transport
    pub async fn read_message(&self) -> Result<MCPMessage> {
        match &self.transport_type {
            TransportType::Stdio => self.read_stdio_message().await,
            _ => {
                // For other transports, check the message queue
                let mut queue = self.message_queue.lock().await;
                if let Some(message) = queue.pop_front() {
                    Ok(message)
                } else {
                    // Simulate waiting for a message
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    Err(anyhow::anyhow!("No message available"))
                }
            }
        }
    }
    
    /// Send a message through the transport
    pub async fn send_message(&self, message: MCPMessage) -> Result<()> {
        match &self.transport_type {
            TransportType::Stdio => self.send_stdio_message(message).await,
            _ => {
                // For other transports, add to queue (simplified)
                let mut queue = self.message_queue.lock().await;
                queue.push_back(message);
                Ok(())
            }
        }
    }
    
    /// Read message from stdin (JSON-RPC over stdio)
    async fn read_stdio_message(&self) -> Result<MCPMessage> {
        let stdin = tokio::io::stdin();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();
        
        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line).await?;
            
            if bytes_read == 0 {
                return Err(anyhow::anyhow!("EOF reached"));
            }
            
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            debug!("Received MCP message: {}", line);
            
            match serde_json::from_str::<MCPMessage>(line) {
                Ok(message) => return Ok(message),
                Err(e) => {
                    warn!("Failed to parse MCP message: {} - Error: {}", line, e);
                    continue;
                }
            }
        }
    }
    
    /// Send message to stdout (JSON-RPC over stdio)
    async fn send_stdio_message(&self, message: MCPMessage) -> Result<()> {
        let json = serde_json::to_string(&message)?;
        
        debug!("Sending MCP message: {}", json);
        
        let mut stdout = tokio::io::stdout();
        stdout.write_all(json.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
        
        Ok(())
    }
    
    /// Start TCP server (placeholder implementation)
    async fn start_tcp_server(&self, _port: u16) -> Result<()> {
        // This would implement a TCP server for MCP
        warn!("TCP transport not fully implemented");
        Ok(())
    }
    
    /// Start WebSocket server (placeholder implementation)
    async fn start_websocket_server(&self, _port: u16) -> Result<()> {
        // This would implement a WebSocket server for MCP
        warn!("WebSocket transport not fully implemented");
        Ok(())
    }
    
    /// Start IPC server (placeholder implementation)
    async fn start_ipc_server(&self, _path: String) -> Result<()> {
        // This would implement an IPC server for MCP
        warn!("IPC transport not fully implemented");
        Ok(())
    }
    
    /// Check if transport is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.lock().await
    }
    
    /// Get transport statistics
    pub async fn get_stats(&self) -> TransportStats {
        let queue_size = self.message_queue.lock().await.len();
        
        TransportStats {
            transport_type: format!("{:?}", self.transport_type),
            is_running: self.is_running().await,
            queue_size,
            messages_sent: 0,    // Would track in real implementation
            messages_received: 0, // Would track in real implementation
            bytes_sent: 0,       // Would track in real implementation
            bytes_received: 0,   // Would track in real implementation
            connection_count: 1, // Would track in real implementation
            last_activity: std::time::SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportStats {
    pub transport_type: String,
    pub is_running: bool,
    pub queue_size: usize,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connection_count: u32,
    pub last_activity: std::time::SystemTime,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            transport_type: "stdio".to_string(),
            buffer_size: 8192,
            timeout_ms: 30000,
            enable_compression: false,
            max_message_size: 1024 * 1024, // 1MB
        }
    }
}

/// Message validation utilities
pub mod validation {
    use super::*;
    
    /// Validate MCP message format
    pub fn validate_message(message: &MCPMessage) -> Result<()> {
        // Check JSON-RPC version
        if message.jsonrpc != "2.0" {
            return Err(anyhow::anyhow!("Invalid JSON-RPC version: {}", message.jsonrpc));
        }
        
        // Check method name format
        if message.method.is_empty() {
            return Err(anyhow::anyhow!("Empty method name"));
        }
        
        // Additional validation would go here
        Ok(())
    }
    
    /// Validate message size
    pub fn validate_message_size(message: &MCPMessage, max_size: usize) -> Result<()> {
        let serialized = serde_json::to_string(message)?;
        if serialized.len() > max_size {
            return Err(anyhow::anyhow!(
                "Message size {} exceeds maximum {}",
                serialized.len(),
                max_size
            ));
        }
        Ok(())
    }
    
    /// Sanitize message content
    pub fn sanitize_message(message: &mut MCPMessage) {
        // Remove any potentially dangerous content
        // This is a simplified implementation
        if message.method.len() > 100 {
            message.method.truncate(100);
        }
    }
}

/// Transport security utilities
pub mod security {
    use super::*;
    
    /// Rate limiting for message processing
    #[derive(Debug)]
    pub struct RateLimiter {
        requests_per_minute: u32,
        request_timestamps: VecDeque<std::time::Instant>,
    }
    
    impl RateLimiter {
        pub fn new(requests_per_minute: u32) -> Self {
            Self {
                requests_per_minute,
                request_timestamps: VecDeque::new(),
            }
        }
        
        /// Check if request is allowed under rate limit
        pub fn is_allowed(&mut self) -> bool {
            let now = std::time::Instant::now();
            let minute_ago = now - std::time::Duration::from_secs(60);
            
            // Remove old timestamps
            while let Some(&front) = self.request_timestamps.front() {
                if front < minute_ago {
                    self.request_timestamps.pop_front();
                } else {
                    break;
                }
            }
            
            // Check if under limit
            if self.request_timestamps.len() < self.requests_per_minute as usize {
                self.request_timestamps.push_back(now);
                true
            } else {
                false
            }
        }
        
        /// Get current request count
        pub fn current_count(&self) -> usize {
            self.request_timestamps.len()
        }
        
        /// Get time until next request is allowed
        pub fn time_until_allowed(&self) -> Option<std::time::Duration> {
            if self.request_timestamps.len() < self.requests_per_minute as usize {
                return None;
            }
            
            if let Some(&oldest) = self.request_timestamps.front() {
                let elapsed = oldest.elapsed();
                if elapsed < std::time::Duration::from_secs(60) {
                    Some(std::time::Duration::from_secs(60) - elapsed)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
    
    /// Message authentication utilities
    pub mod auth {
        use super::*;
        
        /// Simple authentication token validation
        pub fn validate_auth_token(token: Option<&str>, allowed_tokens: &[String]) -> bool {
            if allowed_tokens.is_empty() {
                return true; // No authentication required
            }
            
            if let Some(token) = token {
                allowed_tokens.contains(&token.to_string())
            } else {
                false
            }
        }
        
        /// Extract authentication token from message
        pub fn extract_auth_token(message: &MCPMessage) -> Option<String> {
            // This would extract auth token from message headers or params
            // Simplified implementation
            message.params.as_ref()
                .and_then(|p| p.get("auth_token"))
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
        }
    }
}

/// Connection management utilities
pub mod connection {
    use super::*;
    use std::collections::HashMap;
    
    /// Connection manager for handling multiple clients
    #[derive(Debug)]
    pub struct ConnectionManager {
        connections: HashMap<String, ConnectionInfo>,
        max_connections: usize,
    }
    
    #[derive(Debug, Clone)]
    pub struct ConnectionInfo {
        pub id: String,
        pub client_info: Option<String>,
        pub connected_at: std::time::SystemTime,
        pub last_activity: std::time::SystemTime,
        pub message_count: u64,
        pub is_authenticated: bool,
    }
    
    impl ConnectionManager {
        pub fn new(max_connections: usize) -> Self {
            Self {
                connections: HashMap::new(),
                max_connections,
            }
        }
        
        /// Add a new connection
        pub fn add_connection(&mut self, connection_id: String, client_info: Option<String>) -> Result<()> {
            if self.connections.len() >= self.max_connections {
                return Err(anyhow::anyhow!("Maximum connections reached"));
            }
            
            let connection = ConnectionInfo {
                id: connection_id.clone(),
                client_info,
                connected_at: std::time::SystemTime::now(),
                last_activity: std::time::SystemTime::now(),
                message_count: 0,
                is_authenticated: false,
            };
            
            self.connections.insert(connection_id, connection);
            Ok(())
        }
        
        /// Remove a connection
        pub fn remove_connection(&mut self, connection_id: &str) -> Option<ConnectionInfo> {
            self.connections.remove(connection_id)
        }
        
        /// Update connection activity
        pub fn update_activity(&mut self, connection_id: &str) {
            if let Some(connection) = self.connections.get_mut(connection_id) {
                connection.last_activity = std::time::SystemTime::now();
                connection.message_count += 1;
            }
        }
        
        /// Authenticate a connection
        pub fn authenticate_connection(&mut self, connection_id: &str) {
            if let Some(connection) = self.connections.get_mut(connection_id) {
                connection.is_authenticated = true;
            }
        }
        
        /// Get connection info
        pub fn get_connection(&self, connection_id: &str) -> Option<&ConnectionInfo> {
            self.connections.get(connection_id)
        }
        
        /// List all connections
        pub fn list_connections(&self) -> Vec<&ConnectionInfo> {
            self.connections.values().collect()
        }
        
        /// Clean up inactive connections
        pub fn cleanup_inactive(&mut self, timeout_duration: std::time::Duration) {
            let now = std::time::SystemTime::now();
            let mut to_remove = Vec::new();
            
            for (id, connection) in &self.connections {
                if let Ok(elapsed) = now.duration_since(connection.last_activity) {
                    if elapsed > timeout_duration {
                        to_remove.push(id.clone());
                    }
                }
            }
            
            for id in to_remove {
                self.remove_connection(&id);
            }
        }
        
        /// Get connection statistics
        pub fn get_stats(&self) -> ConnectionStats {
            let total_connections = self.connections.len();
            let authenticated_connections = self.connections.values()
                .filter(|c| c.is_authenticated)
                .count();
            
            let total_messages: u64 = self.connections.values()
                .map(|c| c.message_count)
                .sum();
            
            ConnectionStats {
                total_connections,
                authenticated_connections,
                total_messages,
                average_messages_per_connection: if total_connections > 0 {
                    total_messages as f64 / total_connections as f64
                } else {
                    0.0
                },
            }
        }
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ConnectionStats {
        pub total_connections: usize,
        pub authenticated_connections: usize,
        pub total_messages: u64,
        pub average_messages_per_connection: f64,
    }
}