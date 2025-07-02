use anyhow::{Result, anyhow};
use std::sync::Arc;
use tokio::sync::Mutex;
use tauri::{AppHandle, Emitter};
use tracing::{info, error};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTSConfig {
    pub voice: String,
    pub speed: f32,      // 0.5 - 2.0
    pub pitch: f32,      // 0.5 - 2.0
    pub volume: f32,     // 0.0 - 1.0
    pub language: String,
}

impl Default for TTSConfig {
    fn default() -> Self {
        Self {
            voice: "default".to_string(),
            speed: 1.0,
            pitch: 1.0,
            volume: 1.0,
            language: "en-US".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TTSBackend {
    SystemNative,    // macOS: NSSpeechSynthesizer, Windows: SAPI, Linux: espeak
    WebSpeech,       // Browser-based TTS via Tauri
    Piper,           // Local neural TTS
    ElevenLabs,      // Cloud-based high quality (API key required)
    Coqui,           // Open source neural TTS
}

pub struct TTSManager {
    app_handle: AppHandle,
    config: Arc<Mutex<TTSConfig>>,
    backend: TTSBackend,
    is_speaking: Arc<Mutex<bool>>,
    speech_queue: Arc<Mutex<Vec<String>>>,
}

impl TTSManager {
    pub fn new(app_handle: AppHandle) -> Self {
        // Detect best available backend
        let backend = Self::detect_best_backend();
        info!("Initialized TTS with backend: {:?}", backend);
        
        Self {
            app_handle,
            config: Arc::new(Mutex::new(TTSConfig::default())),
            backend,
            is_speaking: Arc::new(Mutex::new(false)),
            speech_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    fn detect_best_backend() -> TTSBackend {
        // For now, use system native which works on all platforms
        // In future, we can check for Piper models or API keys
        TTSBackend::SystemNative
    }
    
    pub async fn speak(&self, text: &str) -> Result<()> {
        // Add to queue
        {
            let mut queue = self.speech_queue.lock().await;
            queue.push(text.to_string());
        }
        
        // Process queue if not already speaking
        if !*self.is_speaking.lock().await {
            self.process_speech_queue().await?;
        }
        
        Ok(())
    }
    
    pub async fn speak_with_emotion(&self, text: &str, emotion: &str) -> Result<()> {
        // Modify text or config based on emotion
        let modified_text = self.apply_emotion_to_text(text, emotion);
        self.speak(&modified_text).await
    }
    
    async fn process_speech_queue(&self) -> Result<()> {
        loop {
            let next_text = {
                let mut queue = self.speech_queue.lock().await;
                queue.pop()
            };
            
            match next_text {
                Some(text) => {
                    *self.is_speaking.lock().await = true;
                    self.speak_text(&text).await?;
                }
                None => {
                    *self.is_speaking.lock().await = false;
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    async fn speak_text(&self, text: &str) -> Result<()> {
        let config = self.config.lock().await.clone();
        
        // Emit speaking started event
        self.emit_tts_event("speaking-started", serde_json::json!({
            "text": text,
            "voice": config.voice,
        })).await;
        
        match self.backend {
            TTSBackend::SystemNative => {
                self.speak_system_native(text, &config).await?;
            }
            TTSBackend::WebSpeech => {
                self.speak_web_speech(text, &config).await?;
            }
            TTSBackend::Piper => {
                self.speak_piper(text, &config).await?;
            }
            TTSBackend::ElevenLabs => {
                self.speak_elevenlabs(text, &config).await?;
            }
            TTSBackend::Coqui => {
                self.speak_coqui(text, &config).await?;
            }
        }
        
        // Emit speaking finished event
        self.emit_tts_event("speaking-finished", serde_json::json!({})).await;
        
        Ok(())
    }
    
    async fn speak_system_native(&self, text: &str, config: &TTSConfig) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            
            // Use macOS 'say' command
            let mut cmd = Command::new("say");
            cmd.arg("-r").arg((config.speed * 200.0).to_string()); // Rate in words per minute
            
            if config.voice != "default" {
                cmd.arg("-v").arg(&config.voice);
            }
            
            cmd.arg(text);
            
            let output = cmd.output()?;
            if !output.status.success() {
                error!("Failed to speak: {:?}", String::from_utf8_lossy(&output.stderr));
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Use Windows SAPI via PowerShell
            use std::process::Command;
            
            let ps_script = format!(
                r#"Add-Type -AssemblyName System.Speech; 
                $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; 
                $speak.Rate = {}; 
                $speak.Volume = {}; 
                $speak.Speak("{}")"#,
                (config.speed * 5.0 - 5.0) as i32, // SAPI rate is -10 to 10
                (config.volume * 100.0) as i32,
                text.replace("\"", "\\\"")
            );
            
            Command::new("powershell")
                .arg("-Command")
                .arg(&ps_script)
                .output()?;
        }
        
        #[cfg(target_os = "linux")]
        {
            // Use espeak on Linux
            use std::process::Command;
            
            let mut cmd = Command::new("espeak");
            cmd.arg("-s").arg((config.speed * 175.0).to_string()); // Speed in words per minute
            cmd.arg("-a").arg((config.volume * 200.0).to_string()); // Amplitude 0-200
            cmd.arg(text);
            
            let output = cmd.output()?;
            if !output.status.success() {
                error!("Failed to speak: {:?}", String::from_utf8_lossy(&output.stderr));
            }
        }
        
        Ok(())
    }
    
    async fn speak_web_speech(&self, text: &str, config: &TTSConfig) -> Result<()> {
        // Send to frontend to use Web Speech API
        self.app_handle.emit_to(
            tauri::EventTarget::Any,
            "tts-speak-request",
            serde_json::json!({
                "text": text,
                "config": config,
            })
        )?;
        
        // Wait for completion event from frontend
        // In a real implementation, we'd use a proper async channel
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        Ok(())
    }
    
    async fn speak_piper(&self, text: &str, config: &TTSConfig) -> Result<()> {
        // Piper TTS integration - requires piper binary to be installed
        // For now, return an informative error
        error!("Piper TTS requested but not available. Please install Piper from https://github.com/rhasspy/piper");
        
        // Emit error event to frontend
        self.app_handle.emit_to(
            tauri::EventTarget::Any,
            "tts-error",
            serde_json::json!({
                "error": "Piper TTS not available. Please use System or WebSpeech provider instead.",
                "provider": "piper",
                "text": text,
            })
        )?;
        
        Err(anyhow!("Piper TTS not available. To use Piper, install it from https://github.com/rhasspy/piper and configure the binary path."))
    }
    
    async fn speak_elevenlabs(&self, text: &str, config: &TTSConfig) -> Result<()> {
        // ElevenLabs API integration - requires API key
        error!("ElevenLabs TTS requested but not configured. API key required.");
        
        // Check for API key in environment or config
        let api_key = std::env::var("ELEVENLABS_API_KEY").ok();
        if api_key.is_none() {
            self.app_handle.emit_to(
                tauri::EventTarget::Any,
                "tts-error",
                serde_json::json!({
                    "error": "ElevenLabs API key not found. Set ELEVENLABS_API_KEY environment variable.",
                    "provider": "elevenlabs",
                    "text": text,
                })
            )?;
            
            return Err(anyhow!("ElevenLabs TTS requires an API key. Set ELEVENLABS_API_KEY environment variable."));
        }
        
        // Note: Full implementation would make API requests to ElevenLabs
        // For now, return error indicating feature is not fully implemented
        Err(anyhow!("ElevenLabs TTS integration is not fully implemented yet."))
    }
    
    async fn speak_coqui(&self, text: &str, config: &TTSConfig) -> Result<()> {
        // Coqui TTS integration - requires Coqui TTS Python package
        error!("Coqui TTS requested but not available. Please install Coqui TTS.");
        
        // Emit error event to frontend
        self.app_handle.emit_to(
            tauri::EventTarget::Any,
            "tts-error",
            serde_json::json!({
                "error": "Coqui TTS not available. Please use System or WebSpeech provider instead.",
                "provider": "coqui",
                "text": text,
            })
        )?;
        
        // Note: Full implementation would require Python interop or Coqui Rust bindings
        Err(anyhow!("Coqui TTS not available. To use Coqui, install the Python package and configure the integration."))
    }
    
    pub async fn stop_speaking(&self) -> Result<()> {
        // Clear queue
        self.speech_queue.lock().await.clear();
        
        // Stop current speech
        match self.backend {
            TTSBackend::SystemNative => {
                #[cfg(target_os = "macos")]
                {
                    // Kill any 'say' processes
                    let _ = std::process::Command::new("pkill")
                        .arg("-f")
                        .arg("say")
                        .output();
                }
                
                #[cfg(target_os = "windows")]
                {
                    // Stop SAPI (more complex, might need dedicated process)
                }
                
                #[cfg(target_os = "linux")]
                {
                    // Kill espeak
                    let _ = std::process::Command::new("pkill")
                        .arg("espeak")
                        .output();
                }
            }
            _ => {
                // Send stop event to frontend or other backends
                self.emit_tts_event("stop-speaking", serde_json::json!({})).await;
            }
        }
        
        *self.is_speaking.lock().await = false;
        
        Ok(())
    }
    
    pub async fn is_speaking(&self) -> bool {
        *self.is_speaking.lock().await
    }
    
    pub async fn set_config(&self, config: TTSConfig) -> Result<()> {
        *self.config.lock().await = config;
        Ok(())
    }
    
    pub async fn get_available_voices(&self) -> Result<Vec<String>> {
        match self.backend {
            TTSBackend::SystemNative => {
                #[cfg(target_os = "macos")]
                {
                    // Get macOS voices
                    let output = std::process::Command::new("say")
                        .arg("-v")
                        .arg("?")
                        .output()?;
                    
                    let voices_str = String::from_utf8_lossy(&output.stdout);
                    let voices: Vec<String> = voices_str
                        .lines()
                        .filter_map(|line| {
                            line.split_whitespace().next().map(|s| s.to_string())
                        })
                        .collect();
                    
                    return Ok(voices);
                }
            }
            _ => Ok(vec!["default".to_string()]),
        }
    }
    
    fn apply_emotion_to_text(&self, text: &str, emotion: &str) -> String {
        // Simple emotion markup for SSML-capable engines
        match emotion {
            "happy" => format!("<prosody pitch=\"+10%\" rate=\"110%\">{}</prosody>", text),
            "sad" => format!("<prosody pitch=\"-10%\" rate=\"90%\">{}</prosody>", text),
            "excited" => format!("<prosody pitch=\"+20%\" rate=\"120%\" volume=\"+6dB\">{}</prosody>", text),
            "calm" => format!("<prosody rate=\"85%\" volume=\"-3dB\">{}</prosody>", text),
            _ => text.to_string(),
        }
    }
    
    async fn emit_tts_event(&self, event: &str, payload: serde_json::Value) {
        if let Err(e) = self.app_handle.emit_to(
            tauri::EventTarget::Any,
            &format!("tts-{}", event),
            payload
        ) {
            error!("Failed to emit TTS event {}: {}", event, e);
        }
    }
}

// Speech synthesis progress callback
#[derive(Debug, Clone, Serialize)]
pub struct TTSProgress {
    pub char_index: usize,
    pub char_total: usize,
    pub word: String,
    pub elapsed_ms: u64,
}