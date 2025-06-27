use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::info;
use super::WhisperSTT;

// Speech processing state machine
#[derive(Debug, Clone, PartialEq)]
pub enum ConversationState {
    Idle,
    UserSpeaking,
    UserPausedBriefly,  // Short pause, might continue
    UserFinished,       // Long pause, likely done
    AssistantSpeaking,
    AssistantInterrupted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechSegment {
    pub start_time: f32,
    pub end_time: f32,
    pub text: String,
    pub is_final: bool,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub speaker: String,  // "user" or "assistant"
    pub segments: Vec<SpeechSegment>,
    pub started_at: Instant,
    pub ended_at: Option<Instant>,
    pub was_interrupted: bool,
}

pub struct SpeechProcessor {
    state: Arc<Mutex<ConversationState>>,
    current_turn: Arc<Mutex<Option<ConversationTurn>>>,
    conversation_history: Arc<Mutex<Vec<ConversationTurn>>>,
    
    // Timing parameters (in milliseconds)
    brief_pause_threshold: u64,    // 500ms - might continue speaking
    turn_end_threshold: u64,       // 1500ms - probably done speaking
    interruption_threshold: u64,   // 200ms - minimum overlap to count as interruption
    
    // Audio buffer for partial recognition
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    last_speech_time: Arc<Mutex<Instant>>,
    
    // Whisper for speech-to-text
    whisper_stt: Option<Arc<WhisperSTT>>,
    sample_rate: u32,
}

impl SpeechProcessor {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(ConversationState::Idle)),
            current_turn: Arc::new(Mutex::new(None)),
            conversation_history: Arc::new(Mutex::new(Vec::new())),
            
            brief_pause_threshold: 500,
            turn_end_threshold: 1500,
            interruption_threshold: 200,
            
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            last_speech_time: Arc::new(Mutex::new(Instant::now())),
            whisper_stt: None,
            sample_rate: 16000,
        }
    }
    
    pub fn set_whisper(&mut self, whisper: Arc<WhisperSTT>) {
        self.whisper_stt = Some(whisper);
    }
    
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.sample_rate = sample_rate;
    }
    
    // Process incoming audio with VAD results
    pub async fn process_audio_chunk(
        &self,
        audio_data: &[f32],
        has_speech: bool,
        _sample_rate: u32,
    ) -> Result<Option<String>> {
        let now = Instant::now();
        
        // Add to buffer
        self.audio_buffer.lock().unwrap().extend_from_slice(audio_data);
        
        // Get current state and decide what to do
        let should_process_turn = {
            let mut state = self.state.lock().unwrap();
            
            match *state {
                ConversationState::Idle => {
                    if has_speech {
                        info!("User started speaking");
                        *state = ConversationState::UserSpeaking;
                        *self.last_speech_time.lock().unwrap() = now;
                        
                        // Start new turn
                        let mut current_turn = self.current_turn.lock().unwrap();
                        *current_turn = Some(ConversationTurn {
                            speaker: "user".to_string(),
                            segments: Vec::new(),
                            started_at: now,
                            ended_at: None,
                            was_interrupted: false,
                        });
                    }
                    false
                }
                
                ConversationState::UserSpeaking => {
                    if has_speech {
                        *self.last_speech_time.lock().unwrap() = now;
                    } else {
                        let silence_duration = now.duration_since(*self.last_speech_time.lock().unwrap());
                        
                        if silence_duration > Duration::from_millis(self.brief_pause_threshold) {
                            info!("User paused briefly");
                            *state = ConversationState::UserPausedBriefly;
                        }
                    }
                    false
                }
                
                ConversationState::UserPausedBriefly => {
                    if has_speech {
                        // User resumed speaking
                        info!("User resumed speaking");
                        *state = ConversationState::UserSpeaking;
                        *self.last_speech_time.lock().unwrap() = now;
                        false
                    } else {
                        let silence_duration = now.duration_since(*self.last_speech_time.lock().unwrap());
                        
                        if silence_duration > Duration::from_millis(self.turn_end_threshold) {
                            info!("User finished speaking");
                            *state = ConversationState::UserFinished;
                            true // Process the turn
                        } else {
                            false
                        }
                    }
                }
                
                ConversationState::UserFinished => {
                    if has_speech {
                        // User started speaking again before assistant responded
                        info!("User started new turn");
                        *state = ConversationState::UserSpeaking;
                        *self.last_speech_time.lock().unwrap() = now;
                        
                        // Start new turn
                        let mut current_turn = self.current_turn.lock().unwrap();
                        *current_turn = Some(ConversationTurn {
                            speaker: "user".to_string(),
                            segments: Vec::new(),
                            started_at: now,
                            ended_at: None,
                            was_interrupted: false,
                        });
                    }
                    false
                }
                
                ConversationState::AssistantSpeaking => {
                    if has_speech {
                        // User interrupted the assistant
                        info!("User interrupted assistant");
                        *state = ConversationState::AssistantInterrupted;
                        
                        // Mark current assistant turn as interrupted
                        if let Some(ref mut turn) = *self.current_turn.lock().unwrap() {
                            turn.was_interrupted = true;
                            turn.ended_at = Some(now);
                        }
                        
                        // Start new user turn
                        let mut current_turn = self.current_turn.lock().unwrap();
                        *current_turn = Some(ConversationTurn {
                            speaker: "user".to_string(),
                            segments: Vec::new(),
                            started_at: now,
                            ended_at: None,
                            was_interrupted: false,
                        });
                        
                        *state = ConversationState::UserSpeaking;
                    }
                    false
                }
                
                ConversationState::AssistantInterrupted => {
                    // Handled by assistant stopping its speech
                    *state = ConversationState::UserSpeaking;
                    false
                }
            }
        }; // Lock is dropped here
        
        // Process turn if needed (outside of lock)
        if should_process_turn {
            return self.process_complete_turn().await;
        }
        
        Ok(None)
    }
    
    // Process complete audio buffer when turn ends
    async fn process_complete_turn(&self) -> Result<Option<String>> {
        let audio_buffer = self.audio_buffer.lock().unwrap().clone();
        
        if audio_buffer.is_empty() {
            return Ok(None);
        }
        
        info!("Processing {} audio samples for speech-to-text", audio_buffer.len());
        
        // Clear buffer for next turn
        self.audio_buffer.lock().unwrap().clear();
        
        // Use Whisper for speech-to-text if available
        if let Some(ref whisper) = self.whisper_stt {
            match whisper.transcribe(&audio_buffer, self.sample_rate).await {
                Ok(text) => {
                    if !text.is_empty() {
                        info!("Transcribed: {}", text);
                        Ok(Some(text))
                    } else {
                        Ok(None)
                    }
                }
                Err(e) => {
                    info!("Transcription error: {}", e);
                    Ok(None)
                }
            }
        } else {
            // Fallback if Whisper not loaded
            Ok(Some("[Whisper not loaded - please wait for model download]".to_string()))
        }
    }
    
    // Called when assistant starts speaking
    pub fn assistant_started_speaking(&self) {
        let mut state = self.state.lock().unwrap();
        *state = ConversationState::AssistantSpeaking;
        
        let mut current_turn = self.current_turn.lock().unwrap();
        *current_turn = Some(ConversationTurn {
            speaker: "assistant".to_string(),
            segments: Vec::new(),
            started_at: Instant::now(),
            ended_at: None,
            was_interrupted: false,
        });
    }
    
    // Called when assistant stops speaking
    pub fn assistant_stopped_speaking(&self) {
        let mut state = self.state.lock().unwrap();
        
        // Only go to idle if not interrupted
        if *state == ConversationState::AssistantSpeaking {
            *state = ConversationState::Idle;
        }
        
        // End current turn
        if let Some(ref mut turn) = *self.current_turn.lock().unwrap() {
            if turn.speaker == "assistant" {
                turn.ended_at = Some(Instant::now());
                
                // Save to history
                let turn_clone = turn.clone();
                self.conversation_history.lock().unwrap().push(turn_clone);
            }
        }
    }
    
    // Get conversation state
    pub fn get_state(&self) -> ConversationState {
        self.state.lock().unwrap().clone()
    }
    
    // Check if we should interrupt the assistant
    pub fn should_interrupt_assistant(&self) -> bool {
        matches!(
            *self.state.lock().unwrap(),
            ConversationState::AssistantInterrupted
        )
    }
    
    // Get conversation history
    pub fn get_conversation_history(&self) -> Vec<ConversationTurn> {
        self.conversation_history.lock().unwrap().clone()
    }
    
    // Synchronous version that doesn't do transcription
    pub fn process_audio_chunk_sync(
        &mut self,
        audio_data: &[f32],
        has_speech: bool,
        _sample_rate: u32,
    ) -> bool {
        let now = Instant::now();
        
        // Add to buffer
        self.audio_buffer.lock().unwrap().extend_from_slice(audio_data);
        
        // Get current state and decide what to do
        let mut state = self.state.lock().unwrap();
        
        match *state {
            ConversationState::Idle => {
                if has_speech {
                    info!("User started speaking");
                    *state = ConversationState::UserSpeaking;
                    *self.last_speech_time.lock().unwrap() = now;
                    
                    // Start new turn
                    let mut current_turn = self.current_turn.lock().unwrap();
                    *current_turn = Some(ConversationTurn {
                        speaker: "user".to_string(),
                        segments: Vec::new(),
                        started_at: now,
                        ended_at: None,
                        was_interrupted: false,
                    });
                }
                false
            }
            
            ConversationState::UserSpeaking => {
                if has_speech {
                    *self.last_speech_time.lock().unwrap() = now;
                } else {
                    let silence_duration = now.duration_since(*self.last_speech_time.lock().unwrap());
                    
                    if silence_duration > Duration::from_millis(self.brief_pause_threshold) {
                        info!("User paused briefly");
                        *state = ConversationState::UserPausedBriefly;
                    }
                }
                false
            }
            
            ConversationState::UserPausedBriefly => {
                if has_speech {
                    // User resumed speaking
                    info!("User resumed speaking");
                    *state = ConversationState::UserSpeaking;
                    *self.last_speech_time.lock().unwrap() = now;
                    false
                } else {
                    let silence_duration = now.duration_since(*self.last_speech_time.lock().unwrap());
                    
                    if silence_duration > Duration::from_millis(self.turn_end_threshold) {
                        info!("User finished speaking");
                        *state = ConversationState::UserFinished;
                        true // Should transcribe
                    } else {
                        false
                    }
                }
            }
            
            ConversationState::UserFinished => {
                if has_speech {
                    // User started speaking again before assistant responded
                    info!("User started new turn");
                    *state = ConversationState::UserSpeaking;
                    *self.last_speech_time.lock().unwrap() = now;
                    
                    // Start new turn
                    let mut current_turn = self.current_turn.lock().unwrap();
                    *current_turn = Some(ConversationTurn {
                        speaker: "user".to_string(),
                        segments: Vec::new(),
                        started_at: now,
                        ended_at: None,
                        was_interrupted: false,
                    });
                }
                false
            }
            
            ConversationState::AssistantSpeaking => {
                if has_speech {
                    // User interrupted the assistant
                    info!("User interrupted assistant");
                    *state = ConversationState::AssistantInterrupted;
                    
                    // Mark current assistant turn as interrupted
                    if let Some(ref mut turn) = *self.current_turn.lock().unwrap() {
                        turn.was_interrupted = true;
                        turn.ended_at = Some(now);
                    }
                    
                    // Start new user turn
                    let mut current_turn = self.current_turn.lock().unwrap();
                    *current_turn = Some(ConversationTurn {
                        speaker: "user".to_string(),
                        segments: Vec::new(),
                        started_at: now,
                        ended_at: None,
                        was_interrupted: false,
                    });
                    
                    *state = ConversationState::UserSpeaking;
                }
                false
            }
            
            ConversationState::AssistantInterrupted => {
                // Handled by assistant stopping its speech
                *state = ConversationState::UserSpeaking;
                false
            }
        }
    }
    
    // Get and clear the audio buffer
    pub fn get_and_clear_audio_buffer(&mut self) -> Vec<f32> {
        let mut buffer = self.audio_buffer.lock().unwrap();
        let audio_data = buffer.clone();
        buffer.clear();
        
        // Reset state to idle
        *self.state.lock().unwrap() = ConversationState::Idle;
        
        audio_data
    }
}

// Whisper and Silero VAD are now in whisper.rs