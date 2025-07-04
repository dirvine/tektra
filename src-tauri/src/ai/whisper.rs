use anyhow::Result;
use hf_hub::api::tokio::Api;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Emitter};
use tokio::fs;
use tracing::{info, error};
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

// Whisper model options (GGML format for CPU inference)
const WHISPER_REPO: &str = "ggerganov/whisper.cpp";
const WHISPER_TINY_MODEL: &str = "ggml-tiny.bin";  // 39 MB - fastest
const WHISPER_BASE_MODEL: &str = "ggml-base.bin";  // 74 MB - good balance
const WHISPER_SMALL_MODEL: &str = "ggml-small.bin"; // 244 MB - better quality

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    pub model_size: String,  // "tiny", "base", "small"
    pub language: Option<String>,  // None for auto-detect
    pub translate: bool,  // Translate to English
    pub temperature: f32,
    pub beam_size: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            model_size: "base".to_string(),
            language: None,  // Auto-detect
            translate: false,
            temperature: 0.0,
            beam_size: 5,
        }
    }
}

pub struct WhisperSTT {
    app_handle: AppHandle,
    model_path: Option<PathBuf>,
    config: WhisperConfig,
    context: Option<WhisperContext>,
}

impl WhisperSTT {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        Ok(Self {
            app_handle,
            model_path: None,
            config: WhisperConfig::default(),
            context: None,
        })
    }

    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Whisper speech-to-text model");
        
        // Emit initial progress
        self.emit_progress(0, "Starting Whisper model download...", "Whisper").await;
        
        // Download the model with better error handling
        let model_path = match self.download_model().await {
            Ok(path) => path,
            Err(e) => {
                let error_msg = format!("Failed to download Whisper model: {}", e);
                error!("{}", error_msg);
                self.emit_completion(false, Some(error_msg.clone())).await;
                return Err(anyhow::anyhow!(error_msg));
            }
        };
        
        self.model_path = Some(model_path.clone());
        
        // Verify model file exists and is valid
        if !model_path.exists() {
            let error_msg = "Downloaded Whisper model file does not exist".to_string();
            error!("{}", error_msg);
            self.emit_completion(false, Some(error_msg.clone())).await;
            return Err(anyhow::anyhow!(error_msg));
        }
        
        // Check file size
        let file_size = std::fs::metadata(&model_path)?.len();
        if file_size < 1_000_000 { // Less than 1MB indicates a corrupt download
            let error_msg = format!("Whisper model file appears corrupted (size: {} bytes)", file_size);
            error!("{}", error_msg);
            // Remove the corrupt file
            let _ = std::fs::remove_file(&model_path);
            self.emit_completion(false, Some(error_msg.clone())).await;
            return Err(anyhow::anyhow!(error_msg));
        }
        
        // Load the actual Whisper model
        self.emit_progress(95, "Loading Whisper model...", "Whisper").await;
        
        let context = match WhisperContext::new_with_params(
            &model_path.to_string_lossy(),
            WhisperContextParameters::default()
        ) {
            Ok(ctx) => ctx,
            Err(e) => {
                let error_msg = format!("Failed to create Whisper context: {}. The model file may be corrupted.", e);
                error!("{}", error_msg);
                // Remove the potentially corrupt file
                let _ = std::fs::remove_file(&model_path);
                self.emit_completion(false, Some(error_msg.clone())).await;
                return Err(anyhow::anyhow!(error_msg));
            }
        };
        
        self.context = Some(context);
        info!("Whisper model loaded successfully");
        
        // Complete
        self.emit_progress(100, "Whisper ready!", "Whisper").await;
        self.emit_completion(true, None).await;
        
        Ok(())
    }

    async fn download_model(&self) -> Result<PathBuf> {
        let model_file = match self.config.model_size.as_str() {
            "tiny" => WHISPER_TINY_MODEL,
            "small" => WHISPER_SMALL_MODEL,
            _ => WHISPER_BASE_MODEL,
        };

        let api = Api::new()?;
        let _repo = api.model(WHISPER_REPO.to_string());
        
        // Check cache first
        let cache_dir = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join(WHISPER_REPO.replace('/', "--"));

        let model_path = cache_dir.join(model_file);

        if model_path.exists() {
            info!("Whisper model already cached at: {:?}", model_path);
            self.emit_progress(100, "Whisper model found in cache", "Whisper").await;
            return Ok(model_path);
        }

        // Create cache directory
        fs::create_dir_all(&cache_dir).await?;

        // Download the model
        info!("Downloading Whisper {} model...", self.config.model_size);
        self.emit_progress(10, &format!("Downloading Whisper {} model...", self.config.model_size), "Whisper").await;

        // Get file size for progress tracking
        let file_size = match self.config.model_size.as_str() {
            "tiny" => 39_000_000,   // 39 MB
            "small" => 244_000_000, // 244 MB
            _ => 74_000_000,        // 74 MB (base)
        };

        // Download with progress
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            WHISPER_REPO, model_file
        );

        self.download_file_with_progress(
            &url,
            &model_path,
            model_file,
            "Whisper",
            10.0,
            80.0,
            file_size,
        ).await?;

        Ok(model_path)
    }

    async fn download_file_with_progress(
        &self,
        url: &str,
        local_path: &PathBuf,
        filename: &str,
        model_name: &str,
        base_progress: f64,
        progress_weight: f64,
        expected_size: u64,
    ) -> Result<()> {
        let client = reqwest::Client::new();
        let response = client.get(url).send().await?;
        
        let total_size = response.content_length().unwrap_or(expected_size);
        
        let mut file = tokio::fs::File::create(local_path).await?;
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;

            // Calculate progress
            let file_progress = (downloaded as f64 / total_size as f64) * 100.0;
            let overall_progress = base_progress + (file_progress * progress_weight / 100.0);
            
            // Format file sizes
            let downloaded_mb = downloaded as f64 / 1_048_576.0;
            let total_mb = total_size as f64 / 1_048_576.0;
            
            let status = format!(
                "Downloading {} ({:.1} MB / {:.1} MB)",
                filename, downloaded_mb, total_mb
            );
            
            self.emit_progress(overall_progress as u32, &status, model_name).await;
        }

        file.flush().await?;
        Ok(())
    }

    pub async fn transcribe(&self, audio_data: &[f32], sample_rate: u32) -> Result<String> {
        // Check if model is loaded
        let context = self.context.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Whisper model not loaded"))?;

        info!("Transcribing {} audio samples at {} Hz", audio_data.len(), sample_rate);
        
        let duration_secs = audio_data.len() as f32 / sample_rate as f32;
        info!("Audio duration: {:.2} seconds", duration_secs);
        
        // Skip transcription for very short audio
        if duration_secs < 0.1 {
            return Ok("".to_string());
        }

        // Convert to 16kHz if needed (Whisper expects 16kHz)
        let audio_16k = if sample_rate != 16000 {
            self.resample_audio(audio_data, sample_rate, 16000)?
        } else {
            audio_data.to_vec()
        };
        
        // Create a state for running the model
        let state = context.create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {}", e))?;
        
        // Create parameters for Whisper
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        
        // Set language if specified
        if let Some(ref lang) = self.config.language {
            params.set_language(Some(lang.as_str()));
        }
        
        params.set_translate(self.config.translate);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_temperature(self.config.temperature);
        
        // Create a state for running the model
        let mut state = context.create_state()
            .map_err(|e| anyhow::anyhow!("Failed to create Whisper state: {}", e))?;
        
        // Run Whisper inference
        state.full(params, &audio_16k)
            .map_err(|e| anyhow::anyhow!("Whisper inference failed: {}", e))?;
        
        // Extract transcribed text
        let num_segments = state.full_n_segments()
            .map_err(|e| anyhow::anyhow!("Failed to get segments: {}", e))?;
        
        let mut result = String::new();
        for i in 0..num_segments {
            let segment = state.full_get_segment_text(i)
                .map_err(|e| anyhow::anyhow!("Failed to get segment text: {}", e))?;
            result.push_str(&segment);
        }
        
        let transcribed_text = result.trim().to_string();
        info!("Whisper transcription: '{}'" , transcribed_text);
        
        Ok(transcribed_text)
    }

    pub fn is_loaded(&self) -> bool {
        self.context.is_some()
    }

    pub fn get_model_info(&self) -> String {
        format!("Whisper {} model", self.config.model_size)
    }

    /// Simple linear interpolation resampling
    /// For production use, consider using a proper resampling library like rubato
    fn resample_audio(&self, input: &[f32], input_rate: u32, output_rate: u32) -> Result<Vec<f32>> {
        if input_rate == output_rate {
            return Ok(input.to_vec());
        }
        
        let ratio = output_rate as f64 / input_rate as f64;
        let output_len = (input.len() as f64 * ratio) as usize;
        let mut output = Vec::with_capacity(output_len);
        
        for i in 0..output_len {
            let input_index = i as f64 / ratio;
            let index = input_index as usize;
            
            if index >= input.len() - 1 {
                output.push(input[input.len() - 1]);
            } else {
                let frac = input_index - index as f64;
                let sample = input[index] * (1.0 - frac) as f32 + input[index + 1] * frac as f32;
                output.push(sample);
            }
        }
        
        Ok(output)
    }

    async fn emit_progress(&self, progress: u32, status: &str, model_name: &str) {
        let _ = self.app_handle.emit_to(
            tauri::EventTarget::Any,
            "model-loading-progress",
            serde_json::json!({
                "progress": progress,
                "status": status,
                "model_name": model_name,
            }),
        );
    }

    async fn emit_completion(&self, success: bool, error: Option<String>) {
        let _ = self.app_handle.emit_to(
            tauri::EventTarget::Any,
            "model-loading-complete",
            serde_json::json!({
                "success": success,
                "error": error,
            }),
        );
    }
}

// Silero VAD implementation placeholder
pub struct SileroVAD {
    threshold: f32,
    min_speech_duration_ms: u32,
    min_silence_duration_ms: u32,
}

impl SileroVAD {
    pub fn new() -> Result<Self> {
        Ok(Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            min_silence_duration_ms: 100,
        })
    }

    pub fn detect_speech(&self, audio_chunk: &[f32]) -> Result<bool> {
        // Simple energy-based VAD for now
        // In a real implementation, we'd use the actual Silero VAD model
        
        if audio_chunk.is_empty() {
            return Ok(false);
        }

        // Calculate RMS energy
        let energy: f32 = audio_chunk.iter()
            .map(|&x| x * x)
            .sum::<f32>() / audio_chunk.len() as f32;
        
        let rms = energy.sqrt();
        
        // Simple threshold-based detection
        Ok(rms > 0.01) // Adjust threshold as needed
    }
}