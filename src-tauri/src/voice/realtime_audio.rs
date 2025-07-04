use anyhow::{anyhow, Result};
use cpal::{
    traits::{DeviceTrait, HostTrait},
    Device, Host,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use tracing::info;

use super::AudioSettings;

/// Real-time audio manager for voice pipeline
/// Handles audio input/output, format conversion, and buffering
pub struct RealtimeAudioManager {
    /// Audio host (CPAL)
    host: Host,
    /// Input device
    input_device: Option<Device>,
    /// Output device
    output_device: Option<Device>,
    /// Audio settings
    settings: Arc<RwLock<AudioSettings>>,
    /// Audio buffer for recording
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    /// Buffer for audio chunks ready for transmission
    chunk_buffer: Arc<Mutex<Vec<Vec<u8>>>>,
    /// Sample rate for recording
    sample_rate: Arc<RwLock<u32>>,
    /// Recording state
    is_recording: Arc<RwLock<bool>>,
    /// Playback state
    is_playing: Arc<RwLock<bool>>,
    /// Audio format configuration
    format_config: Arc<RwLock<AudioFormatConfig>>,
}

/// Audio format configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFormatConfig {
    /// Sample rate (typically 24000 for real-time, 48000 for high quality)
    pub sample_rate: u32,
    /// Number of channels (1 for mono, 2 for stereo)
    pub channels: u16,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Audio codec (opus, pcm, etc.)
    pub codec: AudioCodec,
    /// Buffer size in samples
    pub buffer_size: u32,
}

/// Supported audio codecs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioCodec {
    /// Raw PCM (uncompressed)
    Pcm,
    /// Opus codec (compressed, ideal for real-time)
    Opus,
    /// AAC codec (compressed)
    Aac,
}

/// Audio format enum for processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioFormat {
    /// 16-bit PCM
    Pcm16,
    /// 32-bit float PCM
    PcmF32,
    /// Opus compressed
    Opus,
    /// AAC compressed
    Aac,
}

/// Audio processing event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioEvent {
    /// Recording started
    RecordingStarted {
        sample_rate: u32,
        channels: u16,
    },
    /// Recording stopped
    RecordingStopped,
    /// Audio chunk available
    AudioChunkReady {
        chunk_size: usize,
        duration_ms: f64,
    },
    /// Playback started
    PlaybackStarted,
    /// Playback stopped
    PlaybackStopped,
    /// Audio level update (for UI visualization)
    AudioLevel {
        level: f32,
        peak: f32,
    },
    /// Error occurred
    Error {
        message: String,
    },
}

impl Default for AudioFormatConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000, // Optimal for real-time speech
            channels: 1,        // Mono for voice
            bits_per_sample: 16,
            codec: AudioCodec::Opus,
            buffer_size: 1024,
        }
    }
}

impl RealtimeAudioManager {
    /// Create a new real-time audio manager
    pub async fn new(settings: AudioSettings) -> Result<Self> {
        info!("Initializing real-time audio manager");

        let host = cpal::default_host();
        
        let manager = Self {
            host,
            input_device: None,
            output_device: None,
            settings: Arc::new(RwLock::new(settings)),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            chunk_buffer: Arc::new(Mutex::new(Vec::new())),
            sample_rate: Arc::new(RwLock::new(24000)),
            is_recording: Arc::new(RwLock::new(false)),
            is_playing: Arc::new(RwLock::new(false)),
            format_config: Arc::new(RwLock::new(AudioFormatConfig::default())),
        };

        info!("Real-time audio manager initialized");
        Ok(manager)
    }

    /// Initialize audio devices
    pub async fn initialize_devices(&mut self) -> Result<()> {
        info!("Initializing audio devices");

        // Get default input device
        let input_device = self.host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device available"))?;
        
        // Get default output device
        let output_device = self.host.default_output_device()
            .ok_or_else(|| anyhow!("No default output device available"))?;

        info!("Input device: {}", input_device.name().unwrap_or_else(|_| "Unknown".to_string()));
        info!("Output device: {}", output_device.name().unwrap_or_else(|_| "Unknown".to_string()));

        self.input_device = Some(input_device);
        self.output_device = Some(output_device);

        Ok(())
    }

    /// Start recording audio
    pub async fn start_recording(&mut self) -> Result<()> {
        if *self.is_recording.read().await {
            return Ok(());
        }

        info!("Starting audio recording");

        // Initialize devices if not already done
        if self.input_device.is_none() {
            self.initialize_devices().await?;
        }

        // For now, just mark as recording - actual implementation will be added later
        *self.is_recording.write().await = true;

        info!("Audio recording started successfully (placeholder implementation)");
        Ok(())
    }

    /// Stop recording audio
    pub async fn stop_recording(&mut self) -> Result<()> {
        if !*self.is_recording.read().await {
            return Ok(());
        }

        info!("Stopping audio recording");

        *self.is_recording.write().await = false;
        
        // Clear buffers
        self.audio_buffer.lock().await.clear();
        self.chunk_buffer.lock().await.clear();

        info!("Audio recording stopped");
        Ok(())
    }

    /// Start audio playback
    pub async fn start_playback(&mut self) -> Result<()> {
        if *self.is_playing.read().await {
            return Ok(());
        }

        info!("Starting audio playback");

        // Initialize devices if not already done
        if self.output_device.is_none() {
            self.initialize_devices().await?;
        }

        // For now, just mark as playing - actual implementation will be added later
        *self.is_playing.write().await = true;

        info!("Audio playback started successfully (placeholder implementation)");
        Ok(())
    }

    /// Stop audio playback
    pub async fn stop_playback(&mut self) -> Result<()> {
        if !*self.is_playing.read().await {
            return Ok(());
        }

        info!("Stopping audio playback");

        *self.is_playing.write().await = false;

        info!("Audio playback stopped");
        Ok(())
    }

    /// Get processed audio buffer for transmission
    pub async fn get_audio_buffer(&mut self) -> Result<Option<Vec<u8>>> {
        let mut chunks = self.chunk_buffer.lock().await;
        
        if chunks.is_empty() {
            return Ok(None);
        }

        // Get the first available chunk
        let chunk = chunks.remove(0);
        Ok(Some(chunk))
    }

    /// Process F32 audio input
    fn process_audio_input_f32(
        data: &[f32],
        audio_buffer: &Arc<Mutex<Vec<f32>>>,
        chunk_buffer: &Arc<Mutex<Vec<Vec<u8>>>>,
        settings: &Arc<RwLock<AudioSettings>>,
        sample_rate: u32,
    ) {
        if let Ok(mut buffer) = audio_buffer.try_lock() {
            buffer.extend_from_slice(data);

            // Process buffer when it reaches target size
            let target_size = 1024; // Configurable chunk size
            
            while buffer.len() >= target_size {
                let chunk: Vec<f32> = buffer.drain(0..target_size).collect();
                
                // Convert to appropriate format for transmission
                if let Ok(encoded_chunk) = Self::encode_audio_chunk(&chunk, sample_rate) {
                    if let Ok(mut chunks) = chunk_buffer.try_lock() {
                        chunks.push(encoded_chunk);
                        
                        // Limit buffer size to prevent memory issues
                        if chunks.len() > 100 {
                            chunks.remove(0);
                        }
                    }
                }
            }
        }
    }

    /// Process I16 audio input
    fn process_audio_input_i16(
        data: &[i16],
        audio_buffer: &Arc<Mutex<Vec<f32>>>,
        chunk_buffer: &Arc<Mutex<Vec<Vec<u8>>>>,
        settings: &Arc<RwLock<AudioSettings>>,
        sample_rate: u32,
    ) {
        // Convert i16 to f32
        let f32_data: Vec<f32> = data.iter()
            .map(|&sample| sample as f32 / i16::MAX as f32)
            .collect();
        
        Self::process_audio_input_f32(&f32_data, audio_buffer, chunk_buffer, settings, sample_rate);
    }

    /// Encode audio chunk for transmission (placeholder for future Opus encoding)
    fn encode_audio_chunk(chunk: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
        // For now, convert to 16-bit PCM bytes
        // Future: Use Opus encoding for better compression and real-time performance
        let mut encoded = Vec::with_capacity(chunk.len() * 2);
        
        for &sample in chunk {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            encoded.extend_from_slice(&sample_i16.to_le_bytes());
        }
        
        Ok(encoded)
    }

    /// Calculate audio level for visualization
    pub fn calculate_audio_level(samples: &[f32]) -> (f32, f32) {
        if samples.is_empty() {
            return (0.0, 0.0);
        }

        let rms = (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
        let peak = samples.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        
        (rms, peak)
    }

    /// Check if currently recording
    pub async fn is_recording(&self) -> bool {
        *self.is_recording.read().await
    }

    /// Check if currently playing
    pub async fn is_playing(&self) -> bool {
        *self.is_playing.read().await
    }

    /// Get current sample rate
    pub async fn get_sample_rate(&self) -> u32 {
        *self.sample_rate.read().await
    }

    /// Update audio settings
    pub async fn update_settings(&mut self, new_settings: AudioSettings) -> Result<()> {
        info!("Updating audio settings");
        *self.settings.write().await = new_settings;
        
        // Restart recording if active to apply new settings
        let was_recording = self.is_recording().await;
        if was_recording {
            self.stop_recording().await?;
            self.start_recording().await?;
        }
        
        Ok(())
    }

    /// Get audio format configuration
    pub async fn get_format_config(&self) -> AudioFormatConfig {
        self.format_config.read().await.clone()
    }

    /// List available audio devices
    pub fn list_input_devices(&self) -> Result<Vec<String>> {
        let devices = self.host.input_devices()
            .map_err(|e| anyhow!("Failed to enumerate input devices: {}", e))?;
        
        let device_names: Result<Vec<String>> = devices
            .map(|device| {
                device.name().map_err(|e| anyhow!("Failed to get device name: {}", e))
            })
            .collect();
        
        device_names
    }

    /// List available output devices
    pub fn list_output_devices(&self) -> Result<Vec<String>> {
        let devices = self.host.output_devices()
            .map_err(|e| anyhow!("Failed to enumerate output devices: {}", e))?;
        
        let device_names: Result<Vec<String>> = devices
            .map(|device| {
                device.name().map_err(|e| anyhow!("Failed to get device name: {}", e))
            })
            .collect();
        
        device_names
    }
}

// Drop implementation removed since we no longer store streams

// Safety: RealtimeAudioManager is safe to Send and Sync since all fields are thread-safe
unsafe impl Send for RealtimeAudioManager {}
unsafe impl Sync for RealtimeAudioManager {}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audio_manager_creation() {
        let settings = AudioSettings {
            sample_rate: 24000,
            buffer_size: 1024,
            audio_format: "opus".to_string(),
            noise_reduction: true,
            auto_gain: true,
        };
        
        let manager = RealtimeAudioManager::new(settings).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_audio_level_calculation() {
        let samples = vec![0.5, -0.3, 0.8, -0.1, 0.0];
        let (rms, peak) = RealtimeAudioManager::calculate_audio_level(&samples);
        
        assert!(rms > 0.0);
        assert!(peak > 0.0);
        assert!(peak >= rms);
    }

    #[test]
    fn test_audio_format_config_default() {
        let config = AudioFormatConfig::default();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.channels, 1);
        assert!(matches!(config.codec, AudioCodec::Opus));
    }

    #[test]
    fn test_encode_audio_chunk() {
        let chunk = vec![0.5, -0.3, 0.8, -0.1];
        let encoded = RealtimeAudioManager::encode_audio_chunk(&chunk, 24000);
        assert!(encoded.is_ok());
        assert_eq!(encoded.unwrap().len(), chunk.len() * 2); // 16-bit samples
    }
}