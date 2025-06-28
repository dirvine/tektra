use anyhow::Result;
use std::path::Path;
use std::time::Instant;

/// Performance metrics for benchmarking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceMetrics {
    pub tokens_generated: usize,
    pub time_to_first_token_ms: f64,
    pub tokens_per_second: f64,
    pub total_time_ms: f64,
    pub peak_memory_mb: f64,
}

/// Configuration for inference backends
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub seed: Option<u64>,
    pub n_threads: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        }
    }
}

/// Trait for different inference backends (GGUF, MLX, etc.)
pub trait InferenceBackend: Send + Sync {
    /// Load a model from the given path
    fn load_model(&mut self, model_path: &Path) -> Result<()>;
    
    /// Check if a model is loaded
    fn is_loaded(&self) -> bool;
    
    /// Generate text from a prompt
    fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String>;
    
    /// Generate text with performance metrics
    fn generate_with_metrics(&self, prompt: &str, config: &InferenceConfig) -> Result<(String, InferenceMetrics)> {
        let start = Instant::now();
        let first_token_time = None;
        let initial_memory = self.get_memory_usage_mb();
        
        // Default implementation - backends can override for more accurate metrics
        let result = self.generate(prompt, config)?;
        
        let total_time = start.elapsed();
        let tokens = result.split_whitespace().count(); // Rough estimate
        
        let metrics = InferenceMetrics {
            tokens_generated: tokens,
            time_to_first_token_ms: first_token_time.unwrap_or(50.0), // Estimate
            tokens_per_second: tokens as f64 / total_time.as_secs_f64(),
            total_time_ms: total_time.as_millis() as f64,
            peak_memory_mb: self.get_memory_usage_mb() - initial_memory,
        };
        
        Ok((result, metrics))
    }
    
    /// Get the backend name for logging/debugging
    fn name(&self) -> &str;
    
    /// Get current memory usage in MB (for benchmarking)
    fn get_memory_usage_mb(&self) -> f64 {
        // Default implementation - backends can override
        0.0
    }
    
    /// Check if this backend is available on the current platform
    fn is_available() -> bool where Self: Sized;
}

/// Backend selection strategy
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BackendType {
    /// Use MLX if available, fallback to GGUF
    Auto,
    /// Force MLX backend (will fail on non-Apple Silicon)
    MLX,
    /// Force GGUF backend
    GGUF,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::Auto
    }
}