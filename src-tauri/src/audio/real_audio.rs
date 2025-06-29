use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate, Stream, StreamConfig};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use tracing::{error, info};

pub enum AudioCommand {
    StartRecording,
    StopRecording,
    GetBuffer,
}

pub enum AudioResponse {
    RecordingStarted,
    RecordingStopped(Vec<f32>),
    Buffer(Vec<f32>),
    Error(String),
}

pub struct RealAudioRecorder {
    command_tx: Sender<AudioCommand>,
    response_rx: Arc<Mutex<Receiver<AudioResponse>>>,
    _audio_thread: thread::JoinHandle<()>,
}

impl RealAudioRecorder {
    pub fn new() -> Result<Self> {
        let (command_tx, command_rx) = channel();
        let (response_tx, response_rx) = channel();
        let response_rx = Arc::new(Mutex::new(response_rx));

        // Spawn audio thread
        let audio_thread = thread::spawn(move || {
            if let Err(e) = audio_thread_main(command_rx, response_tx) {
                error!("Audio thread error: {}", e);
            }
        });

        Ok(Self {
            command_tx,
            response_rx,
            _audio_thread: audio_thread,
        })
    }

    pub fn start_recording(&self) -> Result<()> {
        self.command_tx.send(AudioCommand::StartRecording)?;
        
        // Wait for response
        if let Ok(response) = self.response_rx.lock().unwrap().recv() {
            match response {
                AudioResponse::RecordingStarted => Ok(()),
                AudioResponse::Error(e) => Err(anyhow::anyhow!(e)),
                _ => Err(anyhow::anyhow!("Unexpected response")),
            }
        } else {
            Err(anyhow::anyhow!("No response from audio thread"))
        }
    }

    pub fn stop_recording(&self) -> Result<Vec<f32>> {
        self.command_tx.send(AudioCommand::StopRecording)?;
        
        // Wait for response
        if let Ok(response) = self.response_rx.lock().unwrap().recv() {
            match response {
                AudioResponse::RecordingStopped(buffer) => Ok(buffer),
                AudioResponse::Error(e) => Err(anyhow::anyhow!(e)),
                _ => Err(anyhow::anyhow!("Unexpected response")),
            }
        } else {
            Err(anyhow::anyhow!("No response from audio thread"))
        }
    }

    pub fn get_current_buffer(&self) -> Result<Vec<f32>> {
        self.command_tx.send(AudioCommand::GetBuffer)?;
        
        // Wait for response
        if let Ok(response) = self.response_rx.lock().unwrap().recv() {
            match response {
                AudioResponse::Buffer(buffer) => {
                    if !buffer.is_empty() {
                        tracing::debug!("Got audio buffer with {} samples", buffer.len());
                    }
                    Ok(buffer)
                },
                AudioResponse::Error(e) => Err(anyhow::anyhow!(e)),
                _ => Err(anyhow::anyhow!("Unexpected response")),
            }
        } else {
            Err(anyhow::anyhow!("No response from audio thread"))
        }
    }
}

fn audio_thread_main(
    command_rx: Receiver<AudioCommand>,
    response_tx: Sender<AudioResponse>,
) -> Result<()> {
    // Initialize audio
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

    info!("Using audio device: {}", device.name().unwrap_or_default());

    // Get supported configs
    let supported_configs_range = device.supported_input_configs()?;
    
    // Find a suitable config
    let supported_config = supported_configs_range
        .filter(|c| c.channels() == 1 || c.channels() == 2)
        .filter_map(|c| {
            // Check if 16kHz is supported, otherwise use max sample rate
            let sample_rate = if c.min_sample_rate().0 <= 16000 && c.max_sample_rate().0 >= 16000 {
                SampleRate(16000)
            } else {
                c.max_sample_rate()
            };
            Some(c.with_sample_rate(sample_rate))
        })
        .next()
        .ok_or_else(|| anyhow::anyhow!("No suitable audio config found"))?;

    let config = StreamConfig {
        channels: supported_config.channels(),
        sample_rate: supported_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    info!(
        "Audio config: {} channels @ {} Hz",
        config.channels, config.sample_rate.0
    );

    // Audio buffer
    let audio_buffer = Arc::new(Mutex::new(Vec::new()));
    let is_recording = Arc::new(Mutex::new(false));

    // Create stream (but don't start it yet)
    let stream_buffer = Arc::clone(&audio_buffer);
    let stream_recording = Arc::clone(&is_recording);
    let channels = config.channels;

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !*stream_recording.lock().unwrap() {
                return;
            }

            let mut buffer = stream_buffer.lock().unwrap();

            // Convert to mono if needed
            if channels == 2 {
                for chunk in data.chunks(2) {
                    let mono = (chunk[0] + chunk[1]) / 2.0;
                    buffer.push(mono);
                }
            } else {
                buffer.extend_from_slice(data);
            }
        },
        |err| error!("Audio stream error: {}", err),
        None,
    )?;

    // Main command loop
    let current_stream: Option<Stream> = Some(stream);

    loop {
        match command_rx.recv() {
            Ok(AudioCommand::StartRecording) => {
                info!("Starting audio recording");
                *is_recording.lock().unwrap() = true;
                audio_buffer.lock().unwrap().clear();

                if let Some(ref stream) = current_stream {
                    if let Err(e) = stream.play() {
                        let _ = response_tx.send(AudioResponse::Error(format!(
                            "Failed to start stream: {}",
                            e
                        )));
                        continue;
                    }
                }

                let _ = response_tx.send(AudioResponse::RecordingStarted);
            }
            Ok(AudioCommand::StopRecording) => {
                info!("Stopping audio recording");
                *is_recording.lock().unwrap() = false;

                if let Some(ref stream) = current_stream {
                    if let Err(e) = stream.pause() {
                        error!("Failed to pause stream: {}", e);
                    }
                }

                let buffer = audio_buffer.lock().unwrap().clone();
                info!("Recorded {} samples", buffer.len());
                let _ = response_tx.send(AudioResponse::RecordingStopped(buffer));
            }
            Ok(AudioCommand::GetBuffer) => {
                let mut buffer_guard = audio_buffer.lock().unwrap();
                let buffer = buffer_guard.clone();
                // Clear the buffer after getting it
                buffer_guard.clear();
                drop(buffer_guard);
                tracing::debug!("Sending buffer with {} samples", buffer.len());
                let _ = response_tx.send(AudioResponse::Buffer(buffer));
            }
            Err(_) => {
                info!("Audio thread shutting down");
                break;
            }
        }
    }

    Ok(())
}