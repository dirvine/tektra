mod real_audio;

use anyhow::Result;
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Manager};
use tracing::{info, error};
use real_audio::RealAudioRecorder;
use crate::ai::{SpeechProcessor, SileroVAD, WhisperSTT};

pub struct AudioRecorder {
    app_handle: AppHandle,
    is_recording: Arc<Mutex<bool>>,
    real_recorder: Option<RealAudioRecorder>,
    sample_rate: u32,
    speech_processor: Arc<Mutex<SpeechProcessor>>,
    vad: Option<SileroVAD>,
    whisper: Option<Arc<WhisperSTT>>,
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
        
        Self {
            app_handle,
            is_recording: Arc::new(Mutex::new(false)),
            real_recorder,
            sample_rate: 16000,
            speech_processor: Arc::new(Mutex::new(speech_processor)),
            vad,
            whisper: None,
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
        let _ = self.app_handle.emit_all("recording-started", ());

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
        let _ = self.app_handle.emit_all("recording-stopped", audio_data.len());

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
    pub async fn process_audio_stream(&self) -> Result<()> {
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
        
        info!("Processing audio chunk with {} samples", audio_chunk.len());
        
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
                // Transcribe with Whisper
                let text = if let Some(ref whisper) = self.whisper {
                    match whisper.transcribe(&audio_buffer, self.sample_rate).await {
                        Ok(text) if !text.is_empty() => {
                            info!("Transcribed: {}", text);
                            Some(text)
                        }
                        Ok(_) => {
                            info!("Whisper returned empty transcription");
                            None
                        }
                        Err(e) => {
                            error!("Whisper transcription error: {}", e);
                            None
                        }
                    }
                } else {
                    error!("Whisper not loaded - cannot transcribe audio");
                    None
                };
                
                if let Some(text) = text {
                    // Only emit transcribed text if we actually got something
                    let _ = self.app_handle.emit_all("speech-transcribed", text);
                }
            }
        }
        
        // Check if we should interrupt the assistant
        {
            let processor = self.speech_processor.lock().unwrap();
            if processor.should_interrupt_assistant() {
                let _ = self.app_handle.emit_all("interrupt-assistant", ());
            }
        }
        
        Ok(())
    }
    
    // Get speech processor for external use
    pub fn get_speech_processor(&self) -> Arc<Mutex<SpeechProcessor>> {
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
}