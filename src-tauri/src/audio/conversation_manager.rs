use anyhow::Result;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tauri::{AppHandle, Emitter};
use tracing::{info, warn, debug};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConversationMode {
    Idle,              // Not in conversation
    WakeWordDetected,  // "Tektra" detected, waiting for command
    ActiveListening,   // Actively listening to user
    Processing,        // Processing user input
    Responding,        // AI is responding
    WaitingForUser,    // Waiting for user's turn
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationConfig {
    pub wake_word: String,
    pub wake_word_timeout_ms: u64,     // How long to wait after wake word
    pub turn_timeout_ms: u64,          // How long silence before ending turn
    pub interrupt_threshold_ms: u64,   // How quickly user can interrupt
    pub continuous_conversation: bool,  // Stay active after first interaction
    pub auto_end_timeout_ms: u64,      // End conversation after silence
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            wake_word: "tektra".to_string(),
            wake_word_timeout_ms: 3000,      // 3 seconds to start speaking
            turn_timeout_ms: 1500,           // 1.5 seconds silence = end turn
            interrupt_threshold_ms: 500,     // 0.5 seconds to interrupt
            continuous_conversation: true,    // Keep listening after response
            auto_end_timeout_ms: 30000,      // 30 seconds of no activity
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub speaker: String,
    pub text: String,
    pub audio_data: Option<Vec<u8>>,
    pub started_at: std::time::Instant,
    pub ended_at: Option<std::time::Instant>,
    pub was_interrupted: bool,
}

pub struct ConversationManager {
    app_handle: AppHandle,
    mode: Arc<RwLock<ConversationMode>>,
    config: Arc<RwLock<ConversationConfig>>,
    current_conversation: Arc<Mutex<Vec<ConversationTurn>>>,
    last_activity: Arc<RwLock<std::time::Instant>>,
    wake_word_buffer: Arc<Mutex<String>>, // Rolling buffer for wake word detection
}

impl ConversationManager {
    pub fn new(app_handle: AppHandle) -> Self {
        Self {
            app_handle,
            mode: Arc::new(RwLock::new(ConversationMode::Idle)),
            config: Arc::new(RwLock::new(ConversationConfig::default())),
            current_conversation: Arc::new(Mutex::new(Vec::new())),
            last_activity: Arc::new(RwLock::new(std::time::Instant::now())),
            wake_word_buffer: Arc::new(Mutex::new(String::new())),
        }
    }
    
    /// Process transcribed text from Whisper
    pub async fn process_transcription(&self, text: &str, audio_data: Option<Vec<u8>>) -> Result<()> {
        let text_lower = text.to_lowercase();
        let mode = self.mode.read().await.clone();
        
        debug!("Processing transcription: '{}' in mode {:?}", text, mode);
        
        // Update activity timestamp
        *self.last_activity.write().await = std::time::Instant::now();
        
        match mode {
            ConversationMode::Idle => {
                // Check for wake word
                if self.contains_wake_word(&text_lower).await {
                    self.handle_wake_word_detected(text, audio_data).await?;
                } else {
                    // Update rolling buffer for partial wake word detection
                    self.update_wake_word_buffer(&text_lower).await;
                }
            }
            
            ConversationMode::WakeWordDetected | ConversationMode::ActiveListening => {
                // Process user input
                self.handle_user_input(text, audio_data).await?;
            }
            
            ConversationMode::Responding => {
                // User interrupted
                if !text.trim().is_empty() {
                    self.handle_interruption(text, audio_data).await?;
                }
            }
            
            ConversationMode::WaitingForUser => {
                // Continue conversation
                self.handle_user_input(text, audio_data).await?;
            }
            
            ConversationMode::Processing => {
                // Queue for later or ignore
                debug!("Ignoring input while processing previous request");
            }
        }
        
        Ok(())
    }
    
    /// Check if text contains wake word
    async fn contains_wake_word(&self, text: &str) -> bool {
        let config = self.config.read().await;
        let wake_word = &config.wake_word.to_lowercase();
        
        // Check current text
        if text.contains(wake_word) {
            return true;
        }
        
        // Check with buffer for split wake words
        let buffer = self.wake_word_buffer.lock().await;
        let combined = format!("{} {}", buffer, text);
        combined.contains(wake_word)
    }
    
    /// Update rolling buffer for wake word detection
    async fn update_wake_word_buffer(&self, text: &str) {
        let mut buffer = self.wake_word_buffer.lock().await;
        buffer.push_str(text);
        buffer.push(' ');
        
        // Keep only last 100 characters
        if buffer.len() > 100 {
            *buffer = buffer.chars().skip(buffer.len() - 100).collect();
        }
    }
    
    /// Handle wake word detection
    async fn handle_wake_word_detected(&self, text: &str, audio_data: Option<Vec<u8>>) -> Result<()> {
        info!("Wake word 'Tektra' detected!");
        
        *self.mode.write().await = ConversationMode::WakeWordDetected;
        self.wake_word_buffer.lock().await.clear();
        
        // Emit wake word event
        self.emit_conversation_event("wake-word-detected", serde_json::json!({
            "text": text,
            "has_command": self.extract_command_after_wake_word(text).is_some()
        })).await;
        
        // If there's a command after wake word, process it immediately
        if let Some(command) = self.extract_command_after_wake_word(text) {
            self.handle_user_input(&command, audio_data).await?;
        } else {
            // Start timeout for user to speak
            let config = self.config.read().await.clone();
            self.start_wake_word_timeout(config.wake_word_timeout_ms).await;
        }
        
        Ok(())
    }
    
    /// Extract command after wake word if present
    fn extract_command_after_wake_word(&self, text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        if let Some(idx) = lower.find("tektra") {
            let after_wake = &text[idx + 6..].trim();
            if !after_wake.is_empty() {
                return Some(after_wake.to_string());
            }
        }
        None
    }
    
    /// Handle user input during conversation
    async fn handle_user_input(&self, text: &str, audio_data: Option<Vec<u8>>) -> Result<()> {
        if text.trim().is_empty() {
            return Ok(());
        }
        
        info!("User input: {}", text);
        
        *self.mode.write().await = ConversationMode::Processing;
        
        // Add to conversation history
        let turn = ConversationTurn {
            speaker: "user".to_string(),
            text: text.to_string(),
            audio_data: audio_data.clone(),
            started_at: std::time::Instant::now(),
            ended_at: None,
            was_interrupted: false,
        };
        
        self.current_conversation.lock().await.push(turn);
        
        // Emit user input event for processing
        self.emit_conversation_event("user-input", serde_json::json!({
            "text": text,
            "has_audio": audio_data.is_some(),
            "conversation_length": self.current_conversation.lock().await.len()
        })).await;
        
        // The main app will process this and call start_ai_response when ready
        
        Ok(())
    }
    
    /// Handle interruption during AI response
    async fn handle_interruption(&self, text: &str, audio_data: Option<Vec<u8>>) -> Result<()> {
        info!("User interrupted with: {}", text);
        
        // Mark current AI turn as interrupted
        let mut conversation = self.current_conversation.lock().await;
        if let Some(last_turn) = conversation.last_mut() {
            if last_turn.speaker == "assistant" {
                last_turn.was_interrupted = true;
                last_turn.ended_at = Some(std::time::Instant::now());
            }
        }
        
        // Emit interruption event
        self.emit_conversation_event("user-interrupted", serde_json::json!({
            "text": text
        })).await;
        
        // Process the interruption as new input
        self.handle_user_input(text, audio_data).await?;
        
        Ok(())
    }
    
    /// Called when AI starts responding
    pub async fn start_ai_response(&self, text: &str) -> Result<()> {
        *self.mode.write().await = ConversationMode::Responding;
        
        let turn = ConversationTurn {
            speaker: "assistant".to_string(),
            text: text.to_string(),
            audio_data: None,
            started_at: std::time::Instant::now(),
            ended_at: None,
            was_interrupted: false,
        };
        
        self.current_conversation.lock().await.push(turn);
        
        self.emit_conversation_event("ai-responding", serde_json::json!({
            "text": text
        })).await;
        
        Ok(())
    }
    
    /// Called when AI finishes responding
    pub async fn end_ai_response(&self) -> Result<()> {
        let config = self.config.read().await.clone();
        
        // Mark AI turn as ended
        let mut conversation = self.current_conversation.lock().await;
        if let Some(last_turn) = conversation.last_mut() {
            if last_turn.speaker == "assistant" && last_turn.ended_at.is_none() {
                last_turn.ended_at = Some(std::time::Instant::now());
            }
        }
        
        if config.continuous_conversation {
            *self.mode.write().await = ConversationMode::WaitingForUser;
            
            // Start auto-end timeout
            self.start_auto_end_timeout(config.auto_end_timeout_ms).await;
        } else {
            self.end_conversation().await?;
        }
        
        Ok(())
    }
    
    /// End the current conversation
    pub async fn end_conversation(&self) -> Result<()> {
        *self.mode.write().await = ConversationMode::Idle;
        
        let conversation = self.current_conversation.lock().await.clone();
        
        self.emit_conversation_event("conversation-ended", serde_json::json!({
            "turns": conversation.len(),
            "duration_seconds": conversation.first()
                .and_then(|first| conversation.last().map(|last| {
                    let end = last.ended_at.unwrap_or(std::time::Instant::now());
                    end.duration_since(first.started_at).as_secs()
                }))
                .unwrap_or(0)
        })).await;
        
        // Clear conversation history
        self.current_conversation.lock().await.clear();
        
        Ok(())
    }
    
    /// Start wake word timeout
    async fn start_wake_word_timeout(&self, timeout_ms: u64) {
        let mode = self.mode.clone();
        
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(timeout_ms)).await;
            
            let current_mode = mode.read().await.clone();
            if current_mode == ConversationMode::WakeWordDetected {
                info!("Wake word timeout - returning to idle");
                *mode.write().await = ConversationMode::Idle;
            }
        });
    }
    
    /// Start auto-end timeout for continuous conversation
    async fn start_auto_end_timeout(&self, timeout_ms: u64) {
        let mode = self.mode.clone();
        let last_activity = self.last_activity.clone();
        let manager = self.clone();
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                
                let elapsed = last_activity.read().await.elapsed().as_millis() as u64;
                if elapsed > timeout_ms {
                    let current_mode = mode.read().await.clone();
                    if current_mode == ConversationMode::WaitingForUser {
                        info!("Auto-ending conversation due to inactivity");
                        let _ = manager.end_conversation().await;
                        break;
                    }
                }
            }
        });
    }
    
    /// Get current conversation mode
    pub async fn get_mode(&self) -> ConversationMode {
        self.mode.read().await.clone()
    }
    
    /// Get conversation history
    pub async fn get_conversation_history(&self) -> Vec<ConversationTurn> {
        self.current_conversation.lock().await.clone()
    }
    
    /// Update configuration
    pub async fn update_config(&self, config: ConversationConfig) -> Result<()> {
        *self.config.write().await = config;
        Ok(())
    }
    
    /// Check if in active conversation
    pub async fn is_in_conversation(&self) -> bool {
        !matches!(
            *self.mode.read().await,
            ConversationMode::Idle
        )
    }
    
    /// Emit conversation event
    async fn emit_conversation_event(&self, event: &str, payload: serde_json::Value) {
        if let Err(e) = self.app_handle.emit_to(
            tauri::EventTarget::Any,
            &format!("conversation-{}", event),
            payload
        ) {
            warn!("Failed to emit conversation event {}: {}", event, e);
        }
    }
}

// Clone implementation for spawning
impl Clone for ConversationManager {
    fn clone(&self) -> Self {
        Self {
            app_handle: self.app_handle.clone(),
            mode: self.mode.clone(),
            config: self.config.clone(),
            current_conversation: self.current_conversation.clone(),
            last_activity: self.last_activity.clone(),
            wake_word_buffer: self.wake_word_buffer.clone(),
        }
    }
}