use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, Mutex, RwLock, Semaphore};
use tokio::time::{timeout, Duration};
use tokio_tungstenite::{
    accept_async, tungstenite::Message, WebSocketStream,
};
use tracing::{debug, error, info, warn};

use super::{
    SessionConfig, UnmuteMessage, UnmuteMessageData, UnmuteResponse, VoiceCharacterConfig,
};
use crate::inference::{ModelRegistry, MultimodalInput};

/// Rust-based backend server replacing Python uvicorn
/// Implements the same WebSocket API as Unmute's main_websocket.py
pub struct RustBackendServer {
    /// Server address
    bind_addr: SocketAddr,
    /// STT service URL
    stt_service_url: String,
    /// TTS service URL
    tts_service_url: String,
    /// Model registry for LLM inference
    model_registry: Arc<Mutex<ModelRegistry>>,
    /// Active client connections
    active_connections: Arc<Mutex<HashMap<String, ClientConnection>>>,
    /// Connection semaphore (max 4 concurrent)
    connection_semaphore: Arc<Semaphore>,
    /// Server shutdown signal
    shutdown_tx: Option<mpsc::UnboundedSender<()>>,
    /// Server running state
    is_running: Arc<RwLock<bool>>,
}

/// Individual client connection handler
struct ClientConnection {
    /// Client ID
    id: String,
    /// WebSocket stream
    ws_stream: WebSocketStream<TcpStream>,
    /// Current session configuration
    session_config: SessionConfig,
    /// STT client for speech-to-text
    stt_client: Option<STTClient>,
    /// TTS client for text-to-speech
    tts_client: Option<TTSClient>,
    /// Connection start time
    connected_at: std::time::Instant,
}

/// STT service client
struct STTClient {
    /// WebSocket connection to STT service
    ws_stream: WebSocketStream<TcpStream>,
    /// Current transcription buffer
    transcription_buffer: String,
    /// Speech detection state
    is_speech_active: bool,
}

/// TTS service client
struct TTSClient {
    /// WebSocket connection to TTS service
    ws_stream: WebSocketStream<TcpStream>,
    /// Audio output buffer
    audio_buffer: Vec<u8>,
    /// Synthesis state
    is_synthesizing: bool,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct BackendServerConfig {
    /// Server bind address
    pub bind_addr: SocketAddr,
    /// STT service URL
    pub stt_service_url: String,
    /// TTS service URL
    pub tts_service_url: String,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Connection timeout in seconds
    pub connection_timeout: u64,
    /// Health check interval in seconds
    pub health_check_interval: u64,
}

impl Default for BackendServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:8000".parse().unwrap(),
            stt_service_url: "ws://127.0.0.1:8090".to_string(),
            tts_service_url: "ws://127.0.0.1:8089".to_string(),
            max_connections: 4,
            connection_timeout: 300, // 5 minutes
            health_check_interval: 30, // 30 seconds
        }
    }
}

/// Health check response
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    stt_status: String,
    tts_status: String,
    llm_status: String,
    active_connections: usize,
    uptime_seconds: u64,
}

impl RustBackendServer {
    /// Create a new Rust backend server
    pub async fn new(
        config: BackendServerConfig,
        model_registry: Arc<Mutex<ModelRegistry>>,
    ) -> Result<Self> {
        info!("Initializing Rust backend server on {}", config.bind_addr);

        Ok(Self {
            bind_addr: config.bind_addr,
            stt_service_url: config.stt_service_url,
            tts_service_url: config.tts_service_url,
            model_registry,
            active_connections: Arc::new(Mutex::new(HashMap::new())),
            connection_semaphore: Arc::new(Semaphore::new(config.max_connections)),
            shutdown_tx: None,
            is_running: Arc::new(RwLock::new(false)),
        })
    }

    /// Start the backend server
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting Rust backend server...");

        // Health check services before starting
        self.check_service_health().await?;

        let listener = TcpListener::bind(self.bind_addr).await?;
        info!("Backend server listening on {}", self.bind_addr);

        *self.is_running.write().await = true;

        let (shutdown_tx, mut shutdown_rx) = mpsc::unbounded_channel();
        self.shutdown_tx = Some(shutdown_tx);

        let active_connections = Arc::clone(&self.active_connections);
        let connection_semaphore = Arc::clone(&self.connection_semaphore);
        let stt_service_url = self.stt_service_url.clone();
        let tts_service_url = self.tts_service_url.clone();
        let model_registry = Arc::clone(&self.model_registry);
        let is_running = Arc::clone(&self.is_running);

        // Spawn main server loop
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    // Handle incoming connections
                    result = listener.accept() => {
                        match result {
                            Ok((mut stream, addr)) => {
                                info!("New connection from {}", addr);

                                // Check connection limit
                                if let Ok(_permit) = connection_semaphore.try_acquire() {
                                    let connections = Arc::clone(&active_connections);
                                    let stt_url = stt_service_url.clone();
                                    let tts_url = tts_service_url.clone();
                                    let registry = Arc::clone(&model_registry);

                                    tokio::spawn(async move {
                                        if let Err(e) = Self::handle_client_connection(
                                            stream,
                                            addr,
                                            connections,
                                            stt_url,
                                            tts_url,
                                            registry,
                                        ).await {
                                            error!("Client connection error: {}", e);
                                        }
                                        // Permit is automatically released when dropped
                                    });
                                } else {
                                    warn!("Connection limit reached, rejecting {}", addr);
                                    let _ = stream.shutdown().await;
                                }
                            }
                            Err(e) => {
                                error!("Failed to accept connection: {}", e);
                            }
                        }
                    }

                    // Handle shutdown signal
                    _ = shutdown_rx.recv() => {
                        info!("Received shutdown signal");
                        break;
                    }
                }
            }

            *is_running.write().await = false;
            info!("Backend server stopped");
        });

        Ok(())
    }

    /// Stop the backend server
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping Rust backend server...");

        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }

        // Close all active connections
        let mut connections = self.active_connections.lock().await;
        for (id, mut connection) in connections.drain() {
            info!("Closing connection: {}", id);
            let _ = connection.ws_stream.close(None).await;
        }

        *self.is_running.write().await = false;
        info!("Backend server stopped successfully");

        Ok(())
    }

    /// Check health of dependent services
    async fn check_service_health(&self) -> Result<()> {
        info!("Checking service health...");

        // Check STT service
        if let Err(e) = self.check_stt_health().await {
            warn!("STT service not ready: {}", e);
        }

        // Check TTS service
        if let Err(e) = self.check_tts_health().await {
            warn!("TTS service not ready: {}", e);
        }

        // Check LLM service (model registry)
        let registry = self.model_registry.lock().await;
        if registry.get_active_model_id().await.is_none() {
            warn!("No active LLM model loaded");
        }

        Ok(())
    }

    /// Check STT service health
    async fn check_stt_health(&self) -> Result<()> {
        let url = self.stt_service_url.replace("ws://", "http://").replace("wss://", "https://");
        let health_url = format!("{}/health", url);
        
        match timeout(Duration::from_secs(5), reqwest::get(&health_url)).await {
            Ok(Ok(response)) if response.status().is_success() => {
                debug!("STT service is healthy");
                Ok(())
            }
            _ => Err(anyhow!("STT service health check failed"))
        }
    }

    /// Check TTS service health
    async fn check_tts_health(&self) -> Result<()> {
        let url = self.tts_service_url.replace("ws://", "http://").replace("wss://", "https://");
        let health_url = format!("{}/health", url);
        
        match timeout(Duration::from_secs(5), reqwest::get(&health_url)).await {
            Ok(Ok(response)) if response.status().is_success() => {
                debug!("TTS service is healthy");
                Ok(())
            }
            _ => Err(anyhow!("TTS service health check failed"))
        }
    }

    /// Handle individual client connection
    async fn handle_client_connection(
        stream: TcpStream,
        addr: SocketAddr,
        active_connections: Arc<Mutex<HashMap<String, ClientConnection>>>,
        stt_service_url: String,
        tts_service_url: String,
        model_registry: Arc<Mutex<ModelRegistry>>,
    ) -> Result<()> {
        // Accept WebSocket connection
        let ws_stream = accept_async(stream).await?;
        info!("WebSocket connection established with {}", addr);

        let connection_id = format!("{}_{}", addr, std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis());

        let mut client_connection = ClientConnection {
            id: connection_id.clone(),
            ws_stream,
            session_config: SessionConfig::default(),
            stt_client: None,
            tts_client: None,
            connected_at: std::time::Instant::now(),
        };

        // Add to active connections
        {
            let mut connections = active_connections.lock().await;
            connections.insert(connection_id.clone(), client_connection);
        }

        // Handle the connection
        let result = Self::process_client_messages(
            &connection_id,
            &active_connections,
            &stt_service_url,
            &tts_service_url,
            &model_registry,
        ).await;

        // Remove from active connections
        {
            let mut connections = active_connections.lock().await;
            connections.remove(&connection_id);
        }

        info!("Connection {} closed: {:?}", connection_id, result);
        result
    }

    /// Process messages from a client
    async fn process_client_messages(
        connection_id: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        stt_service_url: &str,
        tts_service_url: &str,
        model_registry: &Arc<Mutex<ModelRegistry>>,
    ) -> Result<()> {
        loop {
            // Get message from client
            let message = {
                let mut connections = active_connections.lock().await;
                if let Some(connection) = connections.get_mut(connection_id) {
                    match connection.ws_stream.next().await {
                        Some(Ok(msg)) => msg,
                        Some(Err(e)) => return Err(anyhow!("WebSocket error: {}", e)),
                        None => return Ok(()), // Connection closed
                    }
                } else {
                    return Err(anyhow!("Connection not found"));
                }
            };

            match message {
                Message::Text(text) => {
                    if let Err(e) = Self::handle_text_message(
                        connection_id,
                        &text,
                        active_connections,
                        stt_service_url,
                        tts_service_url,
                        model_registry,
                    ).await {
                        error!("Error handling text message: {}", e);
                        Self::send_error(connection_id, active_connections, &e.to_string()).await?;
                    }
                }
                Message::Binary(data) => {
                    debug!("Received binary data: {} bytes", data.len());
                    // Handle binary audio data if needed
                }
                Message::Close(_) => {
                    info!("Client {} requested close", connection_id);
                    return Ok(());
                }
                Message::Ping(data) => {
                    Self::send_pong(connection_id, active_connections, data).await?;
                }
                _ => {
                    debug!("Received unhandled message type");
                }
            }
        }
    }

    /// Handle text message from client
    async fn handle_text_message(
        connection_id: &str,
        text: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        stt_service_url: &str,
        tts_service_url: &str,
        model_registry: &Arc<Mutex<ModelRegistry>>,
    ) -> Result<()> {
        debug!("Processing message from {}: {}", connection_id, text);

        let unmute_message: UnmuteMessage = serde_json::from_str(text)?;

        match unmute_message {
            UnmuteMessage::SessionUpdate { session, .. } => {
                Self::handle_session_update(connection_id, active_connections, session).await?;
            }
            UnmuteMessage::InputAudioBufferAppend { audio, .. } => {
                Self::handle_audio_input(
                    connection_id,
                    &audio,
                    active_connections,
                    stt_service_url,
                ).await?;
            }
            _ => {
                debug!("Received message type: {:?}", unmute_message);
                // Handle other message types as needed
            }
        }

        Ok(())
    }

    /// Handle session update
    async fn handle_session_update(
        connection_id: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        session: SessionConfig,
    ) -> Result<()> {
        info!("Updating session for connection {}", connection_id);

        {
            let mut connections = active_connections.lock().await;
            if let Some(connection) = connections.get_mut(connection_id) {
                connection.session_config = session.clone();
            }
        }

        // Send session updated confirmation
        let response = UnmuteMessage::SessionUpdated {
            event_id: None,
            session,
        };

        Self::send_message(connection_id, active_connections, response).await
    }

    /// Handle audio input
    async fn handle_audio_input(
        connection_id: &str,
        audio_base64: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        stt_service_url: &str,
    ) -> Result<()> {
        // Decode base64 audio
        let audio_data = general_purpose::STANDARD.decode(audio_base64)?;
        debug!("Received audio data: {} bytes", audio_data.len());

        // TODO: Forward to STT service and handle response
        // This is a placeholder - implement STT forwarding based on your STT service API

        Ok(())
    }

    /// Send message to client
    async fn send_message(
        connection_id: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        message: UnmuteMessage,
    ) -> Result<()> {
        let json = serde_json::to_string(&message)?;
        let ws_message = Message::Text(json);

        let mut connections = active_connections.lock().await;
        if let Some(connection) = connections.get_mut(connection_id) {
            connection.ws_stream.send(ws_message).await?;
        }

        Ok(())
    }

    /// Send error message to client
    async fn send_error(
        connection_id: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        error_message: &str,
    ) -> Result<()> {
        let error_response = serde_json::json!({
            "type": "error",
            "error": {
                "type": "server_error",
                "message": error_message
            }
        });

        let ws_message = Message::Text(error_response.to_string());

        let mut connections = active_connections.lock().await;
        if let Some(connection) = connections.get_mut(connection_id) {
            connection.ws_stream.send(ws_message).await?;
        }

        Ok(())
    }

    /// Send pong response
    async fn send_pong(
        connection_id: &str,
        active_connections: &Arc<Mutex<HashMap<String, ClientConnection>>>,
        data: Vec<u8>,
    ) -> Result<()> {
        let ws_message = Message::Pong(data);

        let mut connections = active_connections.lock().await;
        if let Some(connection) = connections.get_mut(connection_id) {
            connection.ws_stream.send(ws_message).await?;
        }

        Ok(())
    }

    /// Get server health status
    pub async fn get_health(&self) -> HealthResponse {
        let active_count = self.active_connections.lock().await.len();
        
        HealthResponse {
            status: if *self.is_running.read().await { "healthy".to_string() } else { "stopped".to_string() },
            stt_status: if self.check_stt_health().await.is_ok() { "healthy".to_string() } else { "unhealthy".to_string() },
            tts_status: if self.check_tts_health().await.is_ok() { "healthy".to_string() } else { "unhealthy".to_string() },
            llm_status: {
                let registry = self.model_registry.lock().await;
                if registry.get_active_model_id().await.is_some() {
                    "healthy".to_string()
                } else {
                    "no_model".to_string()
                }
            },
            active_connections: active_count,
            uptime_seconds: 0, // TODO: Track uptime
        }
    }

    /// Check if server is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get active connection count
    pub async fn get_connection_count(&self) -> usize {
        self.active_connections.lock().await.len()
    }
}

// Drop implementation to ensure clean shutdown
impl Drop for RustBackendServer {
    fn drop(&mut self) {
        if let Some(tx) = &self.shutdown_tx {
            let _ = tx.send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ModelRegistry;

    #[tokio::test]
    async fn test_server_creation() {
        let config = BackendServerConfig::default();
        let model_registry = Arc::new(Mutex::new(ModelRegistry::new().await.unwrap()));
        
        let server = RustBackendServer::new(config, model_registry).await;
        assert!(server.is_ok());
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = BackendServerConfig::default();
        let model_registry = Arc::new(Mutex::new(ModelRegistry::new().await.unwrap()));
        
        let server = RustBackendServer::new(config, model_registry).await.unwrap();
        let health = server.get_health().await;
        
        assert_eq!(health.active_connections, 0);
        assert!(!server.is_running().await);
    }
}