use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::net::TcpStream;
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream};
use tracing::{debug, error, info, warn};

use super::{SessionConfig, VoiceCharacterConfig};

/// WebSocket client for communicating with Unmute services
/// Based on OpenAI Realtime API protocol as implemented by Kyutai Unmute
pub struct UnmuteWebSocketClient {
    /// WebSocket connection to Unmute backend
    ws_stream: Option<WebSocketStream<MaybeTlsStream<TcpStream>>>,
    /// Sender for outgoing messages
    outgoing_tx: Option<mpsc::UnboundedSender<UnmuteMessage>>,
    /// Receiver for incoming messages
    incoming_rx: Option<mpsc::UnboundedReceiver<UnmuteMessage>>,
    /// Current session configuration
    session_config: Arc<RwLock<SessionConfig>>,
    /// Connection status
    is_connected: Arc<RwLock<bool>>,
    /// Backend URL for Unmute
    backend_url: String,
}

/// Message types for Unmute WebSocket communication
/// Based on OpenAI Realtime API events from unmute/openai_realtime_api_events.py
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UnmuteMessage {
    /// Session configuration update
    #[serde(rename = "session.update")]
    SessionUpdate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        session: SessionConfig,
    },
    
    /// Session updated confirmation
    #[serde(rename = "session.updated")]
    SessionUpdated {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        session: SessionConfig,
    },
    
    /// Append audio data to input buffer
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Base64-encoded Opus audio data
        audio: String,
    },
    
    /// Speech started detection
    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    
    /// Speech stopped detection
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    
    /// Response created
    #[serde(rename = "response.created")]
    ResponseCreated {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        response: UnmuteResponse,
    },
    
    /// Text delta for streaming response
    #[serde(rename = "response.text.delta")]
    ResponseTextDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        delta: String,
    },
    
    /// Text response complete
    #[serde(rename = "response.text.done")]
    ResponseTextDone {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        text: String,
    },
    
    /// Audio delta for streaming TTS
    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        /// Base64-encoded Opus audio data
        delta: String,
    },
    
    /// Audio response complete
    #[serde(rename = "response.audio.done")]
    ResponseAudioDone {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    
    /// Transcription delta for real-time STT
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    ConversationItemInputAudioTranscriptionDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        delta: String,
        /// Unmute extension: start time of the transcription
        start_time: f64,
    },
    
    /// VAD interruption event
    #[serde(rename = "unmute.interrupted_by_vad")]
    UnmuteInterruptedByVAD {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
    },
    
    /// Text delta ready for processing
    #[serde(rename = "unmute.response.text.delta.ready")]
    UnmuteResponseTextDeltaReady {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        delta: String,
    },
    
    /// Audio delta ready for playback
    #[serde(rename = "unmute.response.audio.delta.ready")]
    UnmuteResponseAudioDeltaReady {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        number_of_samples: i32,
    },
    
    /// Error event
    #[serde(rename = "error")]
    Error {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        error: UnmuteError,
    },
}

/// Response structure for Unmute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmuteResponse {
    pub object: String,
    pub status: String,
    pub voice: String,
    #[serde(default)]
    pub chat_history: Vec<serde_json::Value>,
}

/// Error structure for Unmute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmuteError {
    pub r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

/// Message data for easier handling
#[derive(Debug, Clone)]
pub enum UnmuteMessageData {
    SessionUpdate(SessionConfig),
    SessionUpdated(SessionConfig),
    AudioInput(Vec<u8>), // Decoded Opus audio
    SpeechStarted,
    SpeechStopped,
    ResponseCreated(UnmuteResponse),
    TextDelta(String),
    TextDone(String),
    AudioDelta(Vec<u8>), // Decoded Opus audio  
    AudioDone,
    TranscriptionDelta { text: String, start_time: f64 },
    InterruptedByVAD,
    TextDeltaReady(String),
    AudioDeltaReady(i32),
    Error(UnmuteError),
}

impl UnmuteWebSocketClient {
    /// Create a new Unmute WebSocket client
    pub fn new(backend_url: String) -> Self {
        Self {
            ws_stream: None,
            outgoing_tx: None,
            incoming_rx: None,
            session_config: Arc::new(RwLock::new(SessionConfig::default())),
            is_connected: Arc::new(RwLock::new(false)),
            backend_url,
        }
    }

    /// Connect to Unmute backend WebSocket
    pub async fn connect(&mut self) -> Result<()> {
        info!("Connecting to Unmute backend at: {}", self.backend_url);
        
        let (ws_stream, _) = connect_async(&self.backend_url).await
            .map_err(|e| anyhow!("Failed to connect to WebSocket: {}", e))?;
        
        info!("Connected to Unmute backend successfully");
        
        // Create channels for message handling
        let (outgoing_tx, mut outgoing_rx) = mpsc::unbounded_channel::<UnmuteMessage>();
        let (incoming_tx, incoming_rx) = mpsc::unbounded_channel::<UnmuteMessage>();
        
        self.outgoing_tx = Some(outgoing_tx);
        self.incoming_rx = Some(incoming_rx);
        
        let (mut ws_sink, mut ws_stream_reader) = ws_stream.split();
        
        // Spawn task to handle outgoing messages
        tokio::spawn(async move {
            while let Some(message) = outgoing_rx.recv().await {
                match serde_json::to_string(&message) {
                    Ok(json) => {
                        debug!("Sending message: {}", json);
                        if let Err(e) = ws_sink.send(Message::Text(json)).await {
                            error!("Failed to send message: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        error!("Failed to serialize message: {}", e);
                    }
                }
            }
        });
        
        // Spawn task to handle incoming messages
        let incoming_tx_clone = incoming_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = ws_stream_reader.next().await {
                match msg {
                    Ok(Message::Text(text)) => {
                        debug!("Received message: {}", text);
                        match serde_json::from_str::<UnmuteMessage>(&text) {
                            Ok(parsed_msg) => {
                                if let Err(e) = incoming_tx_clone.send(parsed_msg) {
                                    error!("Failed to forward incoming message: {}", e);
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse incoming message: {}", e);
                            }
                        }
                    }
                    Ok(Message::Binary(data)) => {
                        debug!("Received binary message: {} bytes", data.len());
                        // Handle binary data if needed (e.g., raw audio)
                    }
                    Ok(Message::Close(_)) => {
                        info!("WebSocket connection closed by server");
                        break;
                    }
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {
                        // Handle other message types (ping, pong, etc.)
                    }
                }
            }
        });
        
        *self.is_connected.write().await = true;
        
        Ok(())
    }

    /// Disconnect from Unmute backend
    pub async fn disconnect(&mut self) -> Result<()> {
        info!("Disconnecting from Unmute backend");
        
        *self.is_connected.write().await = false;
        self.outgoing_tx = None;
        self.incoming_rx = None;
        self.ws_stream = None;
        
        Ok(())
    }

    /// Check if client is connected
    pub async fn is_connected(&self) -> bool {
        *self.is_connected.read().await
    }

    /// Update session configuration
    pub async fn update_session(&mut self, config: SessionConfig) -> Result<()> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to Unmute backend"));
        }
        
        let message = UnmuteMessage::SessionUpdate {
            event_id: Some(generate_event_id()),
            session: config.clone(),
        };
        
        self.send_message(message).await?;
        *self.session_config.write().await = config;
        
        Ok(())
    }

    /// Send audio data to Unmute
    pub async fn send_audio(&mut self, audio_data: &[u8]) -> Result<()> {
        if !self.is_connected().await {
            return Err(anyhow!("Not connected to Unmute backend"));
        }
        
        // Encode audio data as base64
        use base64::{Engine as _, engine::general_purpose};
        let base64_audio = general_purpose::STANDARD.encode(audio_data);
        
        let message = UnmuteMessage::InputAudioBufferAppend {
            event_id: Some(generate_event_id()),
            audio: base64_audio,
        };
        
        self.send_message(message).await
    }

    /// Send a message to Unmute backend
    async fn send_message(&self, message: UnmuteMessage) -> Result<()> {
        if let Some(tx) = &self.outgoing_tx {
            tx.send(message)
                .map_err(|e| anyhow!("Failed to send message: {}", e))?;
            Ok(())
        } else {
            Err(anyhow!("WebSocket connection not established"))
        }
    }

    /// Receive next message from Unmute backend
    pub async fn receive_message(&mut self) -> Result<Option<UnmuteMessage>> {
        if let Some(rx) = &mut self.incoming_rx {
            match rx.try_recv() {
                Ok(message) => Ok(Some(message)),
                Err(mpsc::error::TryRecvError::Empty) => Ok(None),
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    Err(anyhow!("WebSocket connection disconnected"))
                }
            }
        } else {
            Err(anyhow!("No incoming message receiver available"))
        }
    }

    /// Get current session configuration
    pub async fn get_session_config(&self) -> SessionConfig {
        self.session_config.read().await.clone()
    }

    /// Initialize session with voice character config
    pub async fn initialize_session(&mut self, voice_config: VoiceCharacterConfig) -> Result<()> {
        let session_config = SessionConfig {
            instructions: voice_config.instructions,
            voice: Some(voice_config.voice),
            allow_recording: voice_config.allow_recording,
        };
        
        self.update_session(session_config).await
    }

    /// Handle incoming message and convert to structured data
    pub fn process_message(message: &UnmuteMessage) -> Result<UnmuteMessageData> {
        use base64::{Engine as _, engine::general_purpose};
        
        match message {
            UnmuteMessage::SessionUpdate { session, .. } => {
                Ok(UnmuteMessageData::SessionUpdate(session.clone()))
            }
            UnmuteMessage::SessionUpdated { session, .. } => {
                Ok(UnmuteMessageData::SessionUpdated(session.clone()))
            }
            UnmuteMessage::InputAudioBufferAppend { audio, .. } => {
                let audio_bytes = general_purpose::STANDARD.decode(audio)
                    .map_err(|e| anyhow!("Failed to decode base64 audio: {}", e))?;
                Ok(UnmuteMessageData::AudioInput(audio_bytes))
            }
            UnmuteMessage::InputAudioBufferSpeechStarted { .. } => {
                Ok(UnmuteMessageData::SpeechStarted)
            }
            UnmuteMessage::InputAudioBufferSpeechStopped { .. } => {
                Ok(UnmuteMessageData::SpeechStopped)
            }
            UnmuteMessage::ResponseCreated { response, .. } => {
                Ok(UnmuteMessageData::ResponseCreated(response.clone()))
            }
            UnmuteMessage::ResponseTextDelta { delta, .. } => {
                Ok(UnmuteMessageData::TextDelta(delta.clone()))
            }
            UnmuteMessage::ResponseTextDone { text, .. } => {
                Ok(UnmuteMessageData::TextDone(text.clone()))
            }
            UnmuteMessage::ResponseAudioDelta { delta, .. } => {
                let audio_bytes = general_purpose::STANDARD.decode(delta)
                    .map_err(|e| anyhow!("Failed to decode base64 audio: {}", e))?;
                Ok(UnmuteMessageData::AudioDelta(audio_bytes))
            }
            UnmuteMessage::ResponseAudioDone { .. } => {
                Ok(UnmuteMessageData::AudioDone)
            }
            UnmuteMessage::ConversationItemInputAudioTranscriptionDelta { delta, start_time, .. } => {
                Ok(UnmuteMessageData::TranscriptionDelta {
                    text: delta.clone(),
                    start_time: *start_time,
                })
            }
            UnmuteMessage::UnmuteInterruptedByVAD { .. } => {
                Ok(UnmuteMessageData::InterruptedByVAD)
            }
            UnmuteMessage::UnmuteResponseTextDeltaReady { delta, .. } => {
                Ok(UnmuteMessageData::TextDeltaReady(delta.clone()))
            }
            UnmuteMessage::UnmuteResponseAudioDeltaReady { number_of_samples, .. } => {
                Ok(UnmuteMessageData::AudioDeltaReady(*number_of_samples))
            }
            UnmuteMessage::Error { error, .. } => {
                Ok(UnmuteMessageData::Error(error.clone()))
            }
        }
    }
}

/// Generate a unique event ID for messages
fn generate_event_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("event_{}", timestamp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_id_generation() {
        let id1 = generate_event_id();
        let id2 = generate_event_id();
        
        assert!(id1.starts_with("event_"));
        assert!(id2.starts_with("event_"));
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_session_config_serialization() {
        let config = SessionConfig {
            instructions: Some("Test instructions".to_string()),
            voice: Some("default".to_string()),
            allow_recording: true,
        };
        
        let message = UnmuteMessage::SessionUpdate {
            event_id: Some("test_id".to_string()),
            session: config,
        };
        
        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("session.update"));
        assert!(json.contains("Test instructions"));
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = UnmuteWebSocketClient::new("ws://localhost:8080".to_string());
        assert!(!client.is_connected().await);
    }
}