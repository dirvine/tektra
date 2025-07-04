use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use tracing::{debug, error, info, warn};

use super::{
    UnmuteWebSocketClient, UnmuteMessage, UnmuteMessageData, VoiceCharacterConfig,
    RealtimeAudioManager, SessionConfig,
};
use crate::inference::{ModelRegistry, MultimodalInput};

/// Main voice pipeline for real-time conversational AI
/// Coordinates STT, LLM inference, and TTS through Unmute services
pub struct VoicePipeline {
    /// WebSocket client for Unmute backend
    unmute_client: Arc<Mutex<UnmuteWebSocketClient>>,
    /// Audio manager for real-time audio processing
    audio_manager: Arc<Mutex<RealtimeAudioManager>>,
    /// Model registry for AI inference
    model_registry: Arc<Mutex<ModelRegistry>>,
    /// Current conversation state
    conversation_state: Arc<RwLock<ConversationState>>,
    /// Pipeline configuration
    config: Arc<RwLock<VoicePipelineConfig>>,
    /// Event channels for UI communication
    event_tx: Option<mpsc::UnboundedSender<VoicePipelineEvent>>,
    /// Processing control
    is_processing: Arc<RwLock<bool>>,
    /// Current session active
    is_session_active: Arc<RwLock<bool>>,
}

/// Configuration for the voice pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoicePipelineConfig {
    /// Unmute backend URL (typically ws://localhost:80/api)  
    pub unmute_backend_url: String,
    /// Voice character configuration
    pub voice_character: VoiceCharacterConfig,
    /// Audio processing settings
    pub audio_settings: AudioSettings,
    /// Model settings for inference
    pub model_settings: ModelSettings,
    /// VAD (Voice Activity Detection) settings
    pub vad_settings: VadSettings,
}

/// Audio processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSettings {
    /// Sample rate for audio processing (typically 24000 for Opus)
    pub sample_rate: u32,
    /// Buffer size for audio chunks
    pub buffer_size: usize,
    /// Audio format (Opus recommended for real-time)
    pub audio_format: String,
    /// Enable noise reduction
    pub noise_reduction: bool,
    /// Automatic gain control
    pub auto_gain: bool,
}

/// Model inference settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSettings {
    /// Model ID to use for inference
    pub model_id: String,
    /// Maximum tokens for response generation
    pub max_tokens: usize,
    /// Temperature for response generation
    pub temperature: f32,
    /// Enable streaming responses
    pub streaming: bool,
}

/// Voice Activity Detection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadSettings {
    /// VAD sensitivity (0.0 to 1.0)
    pub sensitivity: f32,
    /// Minimum speech duration (ms)
    pub min_speech_duration: u32,
    /// Silence timeout before stopping (ms)
    pub silence_timeout: u32,
    /// Enable interruption by VAD
    pub enable_interruption: bool,
}

/// Current state of the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationState {
    /// Current conversation ID
    pub conversation_id: String,
    /// Whether user is currently speaking
    pub user_speaking: bool,
    /// Whether AI is currently responding  
    pub ai_responding: bool,
    /// Current transcription text
    pub current_transcription: String,
    /// Current AI response text
    pub current_response: String,
    /// Audio processing status
    pub audio_status: AudioStatus,
    /// Last error if any
    pub last_error: Option<String>,
    /// Conversation turn count
    pub turn_count: u32,
}

/// Audio processing status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioStatus {
    Idle,
    Listening,
    Processing,
    Speaking,
    Error(String),
}

/// Events emitted by the voice pipeline for UI updates
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum VoicePipelineEvent {
    /// Session started successfully
    SessionStarted {
        conversation_id: String,
        config: VoicePipelineConfig,
    },
    /// Session ended
    SessionEnded {
        conversation_id: String,
        reason: String,
    },
    /// User started speaking
    SpeechStarted {
        timestamp: f64,
    },
    /// User stopped speaking
    SpeechStopped {
        timestamp: f64,
    },
    /// Real-time transcription update
    TranscriptionDelta {
        text: String,
        is_partial: bool,
        start_time: f64,
    },
    /// Final transcription completed
    TranscriptionComplete {
        text: String,
        duration: f64,
    },
    /// AI response generation started
    ResponseStarted {
        model_id: String,
    },
    /// Streaming AI response text
    ResponseTextDelta {
        text: String,
        is_partial: bool,
    },
    /// AI response text completed
    ResponseTextComplete {
        text: String,
    },
    /// Audio synthesis started
    AudioSynthesisStarted,
    /// Streaming TTS audio data
    ResponseAudioDelta {
        audio_samples: i32,
    },
    /// Audio synthesis completed
    AudioSynthesisComplete,
    /// VAD interruption occurred
    InterruptedByVAD,
    /// Error occurred in pipeline
    Error {
        message: String,
        error_type: String,
    },
    /// Pipeline status update
    StatusUpdate {
        state: ConversationState,
    },
}

impl Default for VoicePipelineConfig {
    fn default() -> Self {
        Self {
            unmute_backend_url: "ws://localhost:8000".to_string(),
            voice_character: VoiceCharacterConfig::default(),
            audio_settings: AudioSettings {
                sample_rate: 24000,
                buffer_size: 1024,
                audio_format: "opus".to_string(),
                noise_reduction: true,
                auto_gain: true,
            },
            model_settings: ModelSettings {
                model_id: "Qwen2.5-Omni".to_string(),
                max_tokens: 512,
                temperature: 0.7,
                streaming: true,
            },
            vad_settings: VadSettings {
                sensitivity: 0.6,
                min_speech_duration: 300,
                silence_timeout: 2000,
                enable_interruption: true,
            },
        }
    }
}

impl Default for ConversationState {
    fn default() -> Self {
        Self {
            conversation_id: format!("conv_{}", chrono::Utc::now().timestamp()),
            user_speaking: false,
            ai_responding: false,
            current_transcription: String::new(),
            current_response: String::new(),
            audio_status: AudioStatus::Idle,
            last_error: None,
            turn_count: 0,
        }
    }
}

impl VoicePipeline {
    /// Create a new voice pipeline
    pub async fn new(
        model_registry: Arc<Mutex<ModelRegistry>>,
        config: VoicePipelineConfig,
    ) -> Result<Self> {
        info!("Initializing voice pipeline");

        // Initialize Unmute WebSocket client
        let unmute_client = UnmuteWebSocketClient::new(config.unmute_backend_url.clone());
        
        // Initialize audio manager
        let audio_manager = RealtimeAudioManager::new(config.audio_settings.clone()).await?;

        let pipeline = Self {
            unmute_client: Arc::new(Mutex::new(unmute_client)),
            audio_manager: Arc::new(Mutex::new(audio_manager)),
            model_registry,
            conversation_state: Arc::new(RwLock::new(ConversationState::default())),
            config: Arc::new(RwLock::new(config)),
            event_tx: None,
            is_processing: Arc::new(RwLock::new(false)),
            is_session_active: Arc::new(RwLock::new(false)),
        };

        info!("Voice pipeline initialized successfully");
        Ok(pipeline)
    }

    /// Set event channel for UI communication
    pub fn set_event_channel(&mut self, tx: mpsc::UnboundedSender<VoicePipelineEvent>) {
        self.event_tx = Some(tx);
    }

    /// Start a voice conversation session
    pub async fn start_session(&mut self) -> Result<()> {
        info!("Starting voice conversation session");

        if *self.is_session_active.read().await {
            return Err(anyhow!("Session already active"));
        }

        // Connect to Unmute backend
        let mut client = self.unmute_client.lock().await;
        client.connect().await?;

        // Initialize session with voice character
        let config = self.config.read().await;
        client.initialize_session(config.voice_character.clone()).await?;
        drop(client);

        // Start audio recording
        let mut audio_manager = self.audio_manager.lock().await;
        audio_manager.start_recording().await?;
        drop(audio_manager);

        // Update state
        let mut state = self.conversation_state.write().await;
        state.conversation_id = format!("conv_{}", chrono::Utc::now().timestamp());
        state.audio_status = AudioStatus::Listening;
        state.turn_count = 0;
        drop(state);

        *self.is_session_active.write().await = true;
        *self.is_processing.write().await = true;

        // Emit session started event
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(VoicePipelineEvent::SessionStarted {
                conversation_id: self.conversation_state.read().await.conversation_id.clone(),
                config: config.clone(),
            });
        }

        // Start processing loop
        self.start_processing_loop().await;

        info!("Voice session started successfully");
        Ok(())
    }

    /// Stop the voice conversation session
    pub async fn stop_session(&mut self) -> Result<()> {
        info!("Stopping voice conversation session");

        if !*self.is_session_active.read().await {
            return Ok(());
        }

        // Stop processing
        *self.is_processing.write().await = false;

        // Stop audio recording
        let mut audio_manager = self.audio_manager.lock().await;
        audio_manager.stop_recording().await?;
        drop(audio_manager);

        // Disconnect from Unmute
        let mut client = self.unmute_client.lock().await;
        client.disconnect().await?;
        drop(client);

        // Update state
        let mut state = self.conversation_state.write().await;
        let conversation_id = state.conversation_id.clone();
        state.audio_status = AudioStatus::Idle;
        state.user_speaking = false;
        state.ai_responding = false;
        drop(state);

        *self.is_session_active.write().await = false;

        // Emit session ended event
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(VoicePipelineEvent::SessionEnded {
                conversation_id,
                reason: "User requested".to_string(),
            });
        }

        info!("Voice session stopped successfully");
        Ok(())
    }

    /// Main processing loop for handling real-time voice interaction
    async fn start_processing_loop(&self) {
        let unmute_client = Arc::clone(&self.unmute_client);
        let audio_manager = Arc::clone(&self.audio_manager);
        let model_registry = Arc::clone(&self.model_registry);
        let conversation_state = Arc::clone(&self.conversation_state);
        let is_processing = Arc::clone(&self.is_processing);
        let config = Arc::clone(&self.config);
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            info!("Starting voice pipeline processing loop");

            while *is_processing.read().await {
                // Check for audio input
                if let Ok(mut audio_mgr) = audio_manager.try_lock() {
                    if let Ok(Some(audio_data)) = audio_mgr.get_audio_buffer().await {
                        // Send audio to Unmute for STT
                        if let Ok(mut client) = unmute_client.try_lock() {
                            if let Err(e) = client.send_audio(&audio_data).await {
                                error!("Failed to send audio data: {}", e);
                                Self::emit_error_event(&event_tx, "Audio transmission failed", &e.to_string()).await;
                            }
                        }
                    }
                }

                // Process incoming messages from Unmute
                if let Ok(mut client) = unmute_client.try_lock() {
                    match client.receive_message().await {
                        Ok(Some(message)) => {
                            Self::handle_unmute_message(
                                &message,
                                &model_registry,
                                &conversation_state,
                                &config,
                                &event_tx,
                            ).await;
                        }
                        Ok(None) => {
                            // No message available, continue
                        }
                        Err(e) => {
                            error!("Failed to receive message: {}", e);
                            Self::emit_error_event(&event_tx, "Message reception failed", &e.to_string()).await;
                        }
                    }
                }

                // Small delay to prevent busy waiting
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }

            info!("Voice pipeline processing loop ended");
        });
    }

    /// Handle incoming messages from Unmute backend
    async fn handle_unmute_message(
        message: &UnmuteMessage,
        model_registry: &Arc<Mutex<ModelRegistry>>,
        conversation_state: &Arc<RwLock<ConversationState>>,
        config: &Arc<RwLock<VoicePipelineConfig>>,
        event_tx: &Option<mpsc::UnboundedSender<VoicePipelineEvent>>,
    ) {
        match UnmuteWebSocketClient::process_message(message) {
            Ok(data) => {
                match data {
                    UnmuteMessageData::SpeechStarted => {
                        debug!("Speech started detected");
                        let mut state = conversation_state.write().await;
                        state.user_speaking = true;
                        state.audio_status = AudioStatus::Listening;
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::SpeechStarted {
                            timestamp: chrono::Utc::now().timestamp_millis() as f64,
                        }).await;
                    }
                    UnmuteMessageData::SpeechStopped => {
                        debug!("Speech stopped detected");
                        let mut state = conversation_state.write().await;
                        state.user_speaking = false;
                        state.audio_status = AudioStatus::Processing;
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::SpeechStopped {
                            timestamp: chrono::Utc::now().timestamp_millis() as f64,
                        }).await;
                    }
                    UnmuteMessageData::TranscriptionDelta { text, start_time } => {
                        debug!("Transcription delta: {}", text);
                        let mut state = conversation_state.write().await;
                        state.current_transcription.push_str(&text);
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::TranscriptionDelta {
                            text,
                            is_partial: true,
                            start_time,
                        }).await;
                    }
                    UnmuteMessageData::TextDone(final_text) => {
                        info!("Final transcription: {}", final_text);
                        let mut state = conversation_state.write().await;
                        state.current_transcription = final_text.clone();
                        state.turn_count += 1;
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::TranscriptionComplete {
                            text: final_text.clone(),
                            duration: 0.0, // Will be calculated by audio manager
                        }).await;

                        // Trigger AI response generation
                        Self::generate_ai_response(&final_text, model_registry, config, event_tx).await;
                    }
                    UnmuteMessageData::ResponseCreated(_) => {
                        debug!("AI response creation started");
                        let mut state = conversation_state.write().await;
                        state.ai_responding = true;
                        state.audio_status = AudioStatus::Speaking;
                        drop(state);

                        let config_guard = config.read().await;
                        Self::emit_event(event_tx, VoicePipelineEvent::ResponseStarted {
                            model_id: config_guard.model_settings.model_id.clone(),
                        }).await;
                    }
                    UnmuteMessageData::TextDelta(text_delta) => {
                        debug!("Response text delta: {}", text_delta);
                        let mut state = conversation_state.write().await;
                        state.current_response.push_str(&text_delta);
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::ResponseTextDelta {
                            text: text_delta,
                            is_partial: true,
                        }).await;
                    }
                    UnmuteMessageData::AudioDelta(_audio_data) => {
                        debug!("Received TTS audio delta");
                        // Audio data would be processed for playback here
                        Self::emit_event(event_tx, VoicePipelineEvent::ResponseAudioDelta {
                            audio_samples: 1024, // Placeholder
                        }).await;
                    }
                    UnmuteMessageData::AudioDone => {
                        info!("TTS audio generation completed");
                        let mut state = conversation_state.write().await;
                        state.ai_responding = false;
                        state.audio_status = AudioStatus::Listening;
                        state.current_response.clear();
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::AudioSynthesisComplete).await;
                    }
                    UnmuteMessageData::InterruptedByVAD => {
                        warn!("Response interrupted by VAD");
                        let mut state = conversation_state.write().await;
                        state.ai_responding = false;
                        state.audio_status = AudioStatus::Listening;
                        drop(state);

                        Self::emit_event(event_tx, VoicePipelineEvent::InterruptedByVAD).await;
                    }
                    UnmuteMessageData::Error(error) => {
                        error!("Unmute error: {}", error.message);
                        let mut state = conversation_state.write().await;
                        state.last_error = Some(error.message.clone());
                        state.audio_status = AudioStatus::Error(error.message.clone());
                        drop(state);

                        Self::emit_error_event(event_tx, &error.r#type, &error.message).await;
                    }
                    _ => {
                        debug!("Unhandled message type: {:?}", data);
                    }
                }
            }
            Err(e) => {
                error!("Failed to process Unmute message: {}", e);
                Self::emit_error_event(event_tx, "Message processing failed", &e.to_string()).await;
            }
        }
    }

    /// Generate AI response using the model registry
    async fn generate_ai_response(
        user_input: &str,
        _model_registry: &Arc<Mutex<ModelRegistry>>,
        config: &Arc<RwLock<VoicePipelineConfig>>,
        event_tx: &Option<mpsc::UnboundedSender<VoicePipelineEvent>>,
    ) {
        info!("Generating AI response for: {}", user_input);

        let config_guard = config.read().await;
        let _model_id = config_guard.model_settings.model_id.clone();
        drop(config_guard);

        // Create multimodal input for the AI model
        let _input = MultimodalInput::Text(user_input.to_string());

        // This will integrate with mistral.rs when dependency issue #1523 is resolved
        // For now, we'll emit an event indicating response generation is pending
        Self::emit_event(event_tx, VoicePipelineEvent::ResponseTextComplete {
            text: format!("AI response processing pending for: {}", user_input),
        }).await;

        // Future implementation:
        // match model_registry.generate_response(&model_id, input).await {
        //     Ok(response) => {
        //         // Send response back to Unmute for TTS
        //     }
        //     Err(e) => {
        //         Self::emit_error_event(event_tx, "AI generation failed", &e.to_string()).await;
        //     }
        // }
    }

    /// Emit an event to the UI
    async fn emit_event(
        event_tx: &Option<mpsc::UnboundedSender<VoicePipelineEvent>>,
        event: VoicePipelineEvent,
    ) {
        if let Some(tx) = event_tx {
            let _ = tx.send(event);
        }
    }

    /// Emit an error event
    async fn emit_error_event(
        event_tx: &Option<mpsc::UnboundedSender<VoicePipelineEvent>>,
        error_type: &str,
        message: &str,
    ) {
        Self::emit_event(event_tx, VoicePipelineEvent::Error {
            message: message.to_string(),
            error_type: error_type.to_string(),
        }).await;
    }

    /// Get current conversation state
    pub async fn get_conversation_state(&self) -> ConversationState {
        self.conversation_state.read().await.clone()
    }

    /// Update pipeline configuration
    pub async fn update_config(&mut self, new_config: VoicePipelineConfig) -> Result<()> {
        info!("Updating voice pipeline configuration");
        *self.config.write().await = new_config;
        Ok(())
    }

    /// Check if session is active
    pub async fn is_session_active(&self) -> bool {
        *self.is_session_active.read().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ModelRegistry;

    #[tokio::test]
    async fn test_voice_pipeline_creation() {
        let model_registry = Arc::new(ModelRegistry::new().await.unwrap());
        let config = VoicePipelineConfig::default();
        
        let pipeline = VoicePipeline::new(model_registry, config).await;
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_conversation_state_default() {
        let state = ConversationState::default();
        assert!(!state.user_speaking);
        assert!(!state.ai_responding);
        assert_eq!(state.turn_count, 0);
        assert!(matches!(state.audio_status, AudioStatus::Idle));
    }

    #[test]
    fn test_pipeline_config_serialization() {
        let config = VoicePipelineConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("unmute_backend_url"));
        assert!(json.contains("voice_character"));
    }
}