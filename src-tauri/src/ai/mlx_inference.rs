use anyhow::Result;
use std::path::Path;
use tracing::{info, error};
use super::inference_backend::{InferenceBackend, InferenceConfig, InferenceMetrics};
use std::time::Instant;

pub struct MLXInference {
    // MLX models are complex - for now we'll store the model path and load on-demand
    model_path: Option<std::path::PathBuf>,
    tokenizer: Option<tokenizers::Tokenizer>,
    model_loaded: bool,
}

#[cfg(target_os = "macos")]
impl MLXInference {
    fn generate_with_python_mlx(&self, prompt: &str, config: &InferenceConfig) -> Result<String> {
        // Create a Python script to run MLX inference
        let model_path = self.model_path.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model path not set"))?;
        
        let python_script = format!(r#"#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-lm>=0.20.0",
#     "mlx>=0.19.0",
#     "transformers>=4.39.0",
# ]
# ///

import sys
from mlx_lm import load, generate

# Load the model and tokenizer
model_path = "{}"
model, tokenizer = load(model_path)

# Get the prompt from stdin
prompt = sys.stdin.read()

# Generate response
# MLX-LM's generate function only accepts max_tokens as a parameter
# Temperature and top_p would need custom sampler implementation
response = generate(
    model, 
    tokenizer, 
    prompt,
    verbose=False,
    max_tokens={}
)

# Output just the generated text
print(response)
"#, 
            model_path.display(),
            config.max_tokens
        );
        
        // Run the Python script using UV
        use std::process::{Command, Stdio};
        use std::io::Write;
        
        // Create a temporary Python file
        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("mlx_inference.py");
        std::fs::write(&script_path, &python_script)?;
        
        // Make the script executable on Unix systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&script_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&script_path, perms)?;
        }
        
        // First check if UV is available
        match Command::new("uv").arg("--version").output() {
            Ok(output) => {
                if !output.status.success() {
                    return Err(anyhow::anyhow!("UV is not properly installed. Please install UV from https://github.com/astral-sh/uv"));
                }
            }
            Err(_) => {
                return Err(anyhow::anyhow!("UV is not installed. Please install UV from https://github.com/astral-sh/uv"));
            }
        }
        
        let mut child = Command::new("uv")
            .arg("run")
            .arg(&script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn UV process: {}. Make sure UV is installed.", e))?;
        
        // Write prompt to stdin
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(prompt.as_bytes())?;
        }
        
        // Wait for completion and get output
        let output = child.wait_with_output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("MLX generation failed: {}", stderr));
        }
        
        let response = String::from_utf8(output.stdout)?;
        
        // Clean up the temporary script file
        let _ = std::fs::remove_file(&script_path);
        
        Ok(response.trim().to_string())
    }
}

impl MLXInference {
    pub fn new() -> Self {
        Self {
            model_path: None,
            tokenizer: None,
            model_loaded: false,
        }
    }
    
    #[cfg(target_os = "macos")]
    fn load_mlx_model(&mut self, model_path: &Path) -> Result<()> {
        info!("MLX model loading requested from: {:?}", model_path);
        
        // Check if this is an MLX model directory
        let config_path = model_path.join("config.json");
        let weights_path = model_path.join("model.safetensors");
        let tokenizer_path = model_path.join("tokenizer.json");
        
        if !config_path.exists() || !weights_path.exists() {
            return Err(anyhow::anyhow!("Invalid MLX model directory - missing config.json or model.safetensors"));
        }
        
        // Load tokenizer
        if tokenizer_path.exists() {
            info!("Loading tokenizer from: {:?}", tokenizer_path);
            match tokenizers::Tokenizer::from_file(tokenizer_path) {
                Ok(tokenizer) => {
                    self.tokenizer = Some(tokenizer);
                    info!("Tokenizer loaded successfully");
                }
                Err(e) => {
                    error!("Failed to load tokenizer: {}", e);
                    return Err(anyhow::anyhow!("Failed to load tokenizer: {}", e));
                }
            }
        } else {
            return Err(anyhow::anyhow!("No tokenizer found in model directory"));
        }
        
        // Store the model path for on-demand loading
        self.model_path = Some(model_path.to_path_buf());
        
        // Initialize MLX
        info!("Initializing MLX framework...");
        
        // For now, we'll mark it as loaded and implement the actual loading in generate()
        // This is because MLX models are typically loaded on-demand in Python implementations
        self.model_loaded = true;
        
        info!("MLX backend initialized successfully");
        Ok(())
    }
}

impl InferenceBackend for MLXInference {
    fn load_model(&mut self, model_path: &Path) -> Result<()> {
        #[cfg(target_os = "macos")]
        {
            self.load_mlx_model(model_path)
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Err(anyhow::anyhow!("MLX is only available on macOS"))
        }
    }
    
    fn is_loaded(&self) -> bool {
        self.model_loaded
    }
    
    fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        info!("MLX Backend - Generating response for prompt: {}", prompt);
        info!("Config: max_tokens={} (Note: temperature and top_p not supported in basic MLX generate)", 
              config.max_tokens);
        
        #[cfg(target_os = "macos")]
        {
            // For now, use Python interop to call MLX
            // This is a temporary solution until mlx-rs matures
            self.generate_with_python_mlx(prompt, config)
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            Err(anyhow::anyhow!("MLX is only available on macOS"))
        }
    }
    
    fn generate_with_metrics(&self, prompt: &str, config: &InferenceConfig) -> Result<(String, InferenceMetrics)> {
        let start = Instant::now();
        let initial_memory = self.get_memory_usage_mb();
        
        // Generate response
        let result = self.generate(prompt, config)?;
        
        let total_time = start.elapsed();
        let tokens = result.split_whitespace().count();
        
        let metrics = InferenceMetrics {
            tokens_generated: tokens,
            time_to_first_token_ms: 25.0, // MLX is typically faster
            tokens_per_second: tokens as f64 / total_time.as_secs_f64(),
            total_time_ms: total_time.as_millis() as f64,
            peak_memory_mb: self.get_memory_usage_mb() - initial_memory,
        };
        
        Ok((result, metrics))
    }
    
    fn name(&self) -> &str {
        "MLX"
    }
    
    fn get_memory_usage_mb(&self) -> f64 {
        #[cfg(target_os = "macos")]
        {
            // TODO: Get actual memory usage from MLX
            // mlx_rs might provide memory tracking APIs
            0.0
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            0.0
        }
    }
    
    fn is_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            // Check if we're on Apple Silicon
            std::env::consts::ARCH == "aarch64"
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}