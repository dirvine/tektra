mod real_audio;
mod conversation_manager;
mod tts;

pub use conversation_manager::{ConversationManager, ConversationMode, ConversationConfig};
pub use tts::{TTSManager, TTSConfig};

use anyhow::Result;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;
use tauri::{AppHandle, Emitter};
use tracing::{info, error};
use real_audio::RealAudioRecorder;
use crate::ai::{SpeechProcessor, SileroVAD, WhisperSTT};

pub struct AudioRecorder {
    app_handle: AppHandle,
    is_recording: Arc<StdMutex<bool>>,
    real_recorder: Option<RealAudioRecorder>,
    sample_rate: u32,
    speech_processor: Arc<StdMutex<SpeechProcessor>>,
    vad: Option<SileroVAD>,
    whisper: Option<Arc<WhisperSTT>>,
    conversation_manager: Arc<ConversationManager>,
    tts_manager: Arc<TTSManager>,
}

impl AudioRecorder {
    pub fn new(app_handle: AppHandle) -> Self {
        // Try to create real audio recorder
        let real_recorder = match RealAudioRecorder::new() {
            Ok(recorder) => {
                info!("Real audio recorder initialized successfully");
                Some(recorder)
            }
            Err(e) => {
                error!("Failed to initialize real audio recorder: {}. Falling back to simulation.", e);
                None
            }
        };
        
        // Initialize VAD
        let vad = match SileroVAD::new() {
            Ok(vad) => {
                info!("Silero VAD initialized successfully");
                Some(vad)
            }
            Err(e) => {
                error!("Failed to initialize VAD: {}", e);
                None
            }
        };
        
        let mut speech_processor = SpeechProcessor::new();
        speech_processor.set_sample_rate(16000);
        
        // Initialize conversation and TTS managers
        let conversation_manager = Arc::new(ConversationManager::new(app_handle.clone()));
        let tts_manager = Arc::new(TTSManager::new(app_handle.clone()));
        
        Self {
            app_handle,
            is_recording: Arc::new(StdMutex::new(false)),
            real_recorder,
            sample_rate: 16000,
            speech_processor: Arc::new(StdMutex::new(speech_processor)),
            vad,
            whisper: None,
            conversation_manager,
            tts_manager,
        }
    }

    pub async fn start_recording(&self) -> Result<()> {
        // Check if already recording
        {
            let mut is_recording = self.is_recording.lock().unwrap();
            if *is_recording {
                return Err(anyhow::anyhow!("Already recording"));
            }
            *is_recording = true;
        }

        info!("Starting audio recording");

        // Emit recording started event
        let _ = self.app_handle.emit_to(tauri::EventTarget::Any, "recording-started", ());

        // Use real audio if available
        if let Some(ref recorder) = self.real_recorder {
            info!("Using real microphone input");
            recorder.start_recording()?;
        } else {
            // Fallback to simulation
            info!("Using simulated audio (real audio not available)");
            let is_recording = Arc::clone(&self.is_recording);
            
            tokio::spawn(async move {
                while *is_recording.lock().unwrap() {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            });
        }

        Ok(())
    }

    pub async fn stop_recording(&self) -> Result<Vec<f32>> {
        // Stop recording
        {
            let mut is_recording = self.is_recording.lock().unwrap();
            if !*is_recording {
                return Err(anyhow::anyhow!("Not recording"));
            }
            *is_recording = false;
        }

        // Get audio data
        let audio_data = if let Some(ref recorder) = self.real_recorder {
            recorder.stop_recording()?
        } else {
            // Return empty buffer for simulation
            vec![]
        };
        
        // Emit recording stopped event
        let _ = self.app_handle.emit_to(tauri::EventTarget::Any, "recording-stopped", audio_data.len());

        info!("Recorded {} samples ({:.2} seconds)", audio_data.len(), audio_data.len() as f32 / self.sample_rate as f32);

        Ok(audio_data)
    }

    pub fn is_recording(&self) -> bool {
        *self.is_recording.lock().unwrap()
    }

    pub async fn save_to_wav(&self, audio_data: &[f32], path: &str) -> Result<()> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut writer = hound::WavWriter::create(path, spec)?;
        
        // Convert f32 samples to i16
        for &sample in audio_data {
            let amplitude = (sample * i16::MAX as f32) as i16;
            writer.write_sample(amplitude)?;
        }

        writer.finalize()?;
        
        info!("Saved audio to {}", path);
        Ok(())
    }
    
    // Process continuous audio stream with VAD and speech recognition
    pub async fn process_audio_stream(&mut self) -> Result<()> {
        if !self.is_recording() {
            return Err(anyhow::anyhow!("Not recording"));
        }
        
        // Get current audio buffer from recorder
        let audio_chunk = if let Some(ref recorder) = self.real_recorder {
            recorder.get_current_buffer()?
        } else {
            vec![]
        };
        
        if audio_chunk.is_empty() {
            return Ok(());
        }
        
        tracing::debug!("Processing audio chunk with {} samples", audio_chunk.len());
        
        // Run VAD on the chunk
        let has_speech = if let Some(ref vad) = self.vad {
            vad.detect_speech(&audio_chunk)?
        } else {
            // Fallback: simple energy-based detection
            audio_chunk.iter().any(|&x| x.abs() > 0.1)
        };
        
        // Process audio chunk and check if we should transcribe
        let should_transcribe = {
            let mut processor = self.speech_processor.lock().unwrap();
            processor.process_audio_chunk_sync(&audio_chunk, has_speech, self.sample_rate)
        };
        
        // If we should transcribe, do it now (outside of lock)
        if should_transcribe {
            // Get audio buffer to transcribe
            let audio_buffer = {
                let mut processor = self.speech_processor.lock().unwrap();
                processor.get_and_clear_audio_buffer()
            };
            
            if !audio_buffer.is_empty() {
                info!("Processing audio buffer with {} samples for speech recognition", audio_buffer.len());
                
                let duration = audio_buffer.len() as f32 / self.sample_rate as f32;
                
                // Use Whisper to transcribe the audio
                let (transcribed_text, should_process) = if let Some(ref whisper) = self.whisper {
                    info!("Transcribing {:.1}s of audio with Whisper...", duration);
                    match whisper.transcribe(&audio_buffer, self.sample_rate).await {
                        Ok(text) => {
                            let cleaned_text = text.trim();
                            if cleaned_text.is_empty() {
                                info!("Whisper returned empty transcription, skipping");
                                return Ok(());
                            }
                            info!("Whisper transcription: '{}'", cleaned_text);
                            (cleaned_text.to_string(), true)
                        }
                        Err(e) => {
                            error!("Whisper transcription failed: {}", e);
                            return Ok(());
                        }
                    }
                } else {
                    error!("Whisper not initialized, cannot transcribe audio. Initializing now...");
                    
                    // Try to initialize Whisper on-demand
                    let mut whisper = WhisperSTT::new(self.app_handle.clone())?;
                    match whisper.initialize().await {
                        Ok(_) => {
                            info!("Whisper initialized successfully on-demand");
                            let whisper_arc = Arc::new(whisper);
                            self.whisper = Some(whisper_arc.clone());
                            
                            // Set Whisper in speech processor
                            {
                                let mut processor = self.speech_processor.lock().unwrap();
                                processor.set_whisper(whisper_arc.clone());
                            } // Release the lock here
                            
                            // Now try transcription again
                            let _duration = audio_buffer.len() as f32 / 16000.0;
                            match whisper_arc.transcribe(&audio_buffer, 16000).await {
                                Ok(text) => {
                                    let cleaned_text = text.trim();
                                    if cleaned_text.is_empty() {
                                        info!("Empty transcription result, skipping");
                                        return Ok(());
                                    }
                                    info!("Whisper transcription (on-demand): '{}'", cleaned_text);
                                    (cleaned_text.to_string(), true)
                                }
                                Err(e) => {
                                    error!("Whisper transcription failed even after initialization: {}", e);
                                    return Ok(());
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to initialize Whisper on-demand: {}", e);
                            return Ok(());
                        }
                    }
                };
                
                if should_process {
                    // Convert audio to bytes for multimodal processing
                    let mut audio_bytes = Vec::with_capacity(audio_buffer.len() * 2);
                    for sample in &audio_buffer {
                        let sample_i16 = (*sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                        audio_bytes.extend_from_slice(&sample_i16.to_le_bytes());
                    }
                    
                    // Process through conversation manager first
                    if let Err(e) = self.conversation_manager.process_transcription(&transcribed_text, Some(audio_bytes.clone())).await {
                        error!("Failed to process transcription in conversation manager: {}", e);
                    }
                    
                    // Only emit for processing if in active conversation or wake word detected
                    let conversation_mode = self.conversation_manager.get_mode().await;
                    if !matches!(conversation_mode, ConversationMode::Idle) {
                        // Call the process_audio_input command directly
                        let app_handle = self.app_handle.clone();
                        let transcribed_text_clone = transcribed_text.clone();
                        tokio::spawn(async move {
                            if let Err(e) = app_handle.emit_to(tauri::EventTarget::Any, "process_audio_input", serde_json::json!({
                                "message": transcribed_text_clone,
                                "audio_data": audio_bytes,
                            })) {
                                error!("Failed to emit process_audio_input event: {}", e);
                            }
                        });
                    }
                }
            }
        }
        
        // Check if we should interrupt the assistant
        {
            let processor = self.speech_processor.lock().unwrap();
            if processor.should_interrupt_assistant() {
                let _ = self.app_handle.emit_to(tauri::EventTarget::Any, "interrupt-assistant", ());
            }
        }
        
        Ok(())
    }
    
    // Get speech processor for external use
    pub fn get_speech_processor(&self) -> Arc<StdMutex<SpeechProcessor>> {
        Arc::clone(&self.speech_processor)
    }
    
    // Initialize Whisper STT
    pub async fn initialize_whisper(&mut self) -> Result<()> {
        info!("Initializing Whisper speech-to-text...");
        
        let mut whisper = WhisperSTT::new(self.app_handle.clone())?;
        whisper.initialize().await?;
        
        let whisper_arc = Arc::new(whisper);
        self.whisper = Some(whisper_arc.clone());
        
        // Set Whisper in speech processor
        let mut processor = self.speech_processor.lock().unwrap();
        processor.set_whisper(whisper_arc);
        
        info!("Whisper initialized successfully");
        Ok(())
    }
    
    // Get conversation manager
    pub fn get_conversation_manager(&self) -> Arc<ConversationManager> {
        Arc::clone(&self.conversation_manager)
    }
    
    // Get TTS manager
    pub fn get_tts_manager(&self) -> Arc<TTSManager> {
        Arc::clone(&self.tts_manager)
    }
    
    // Start always-listening mode
    pub async fn start_always_listening(&self) -> Result<()> {
        info!("Starting always-listening mode for wake word detection");
        
        // Start recording if not already
        if !self.is_recording() {
            self.start_recording().await?;
        }
        
        // Start continuous processing loop
        let audio_recorder = Arc::new(Mutex::new(self.clone()));
        let audio_recorder_clone = audio_recorder.clone();
        tokio::spawn(async move {
            loop {
                let mut recorder = audio_recorder_clone.lock().await;
                if !recorder.is_recording() {
                    break;
                }
                
                // Process audio stream continuously
                if let Err(e) = recorder.process_audio_stream().await {
                    error!("Error in always-listening loop: {}", e);
                }
                drop(recorder);
                
                // Small delay to prevent CPU overuse
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        });
        
        Ok(())
    }
}

// Clone implementation for AudioRecorder to use in spawned tasks
impl Clone for AudioRecorder {
    fn clone(&self) -> Self {
        Self {
            app_handle: self.app_handle.clone(),
            is_recording: Arc::clone(&self.is_recording),
            real_recorder: None, // Don't clone the actual recorder
            sample_rate: self.sample_rate,
            speech_processor: Arc::clone(&self.speech_processor),
            vad: None, // VAD is not Clone
            whisper: self.whisper.clone(),
            conversation_manager: Arc::clone(&self.conversation_manager),
            tts_manager: Arc::clone(&self.tts_manager),
        }
    }
}