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

/// Trait for different inference backends (Ollama only)
#[async_trait::async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Load a model from the given path
    async fn load_model(&mut self, model_path: &Path) -> Result<()>;
    
    /// Check if a model is loaded
    fn is_loaded(&self) -> bool;
    
    /// Generate text from a prompt
    async fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String>;
    
    /// Generate multimodal response with text and optional image or audio
    async fn generate_multimodal(&self, prompt: &str, media_data: Option<&[u8]>, media_type: Option<&str>, config: &InferenceConfig) -> Result<String>;
    
    /// Allow downcasting to concrete types for backend-specific methods
    fn as_any(&self) -> &dyn std::any::Any;
    
    /// Generate text with performance metrics
    async fn generate_with_metrics(&self, prompt: &str, config: &InferenceConfig) -> Result<(String, InferenceMetrics)> {
        let start = Instant::now();
        let first_token_time = None;
        let initial_memory = self.get_memory_usage_mb();
        
        // Default implementation - backends can override for more accurate metrics
        let result = self.generate(prompt, config).await?;
        
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

/// Backend selection strategy - Ollama only
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BackendType {
    /// Use Ollama backend (cross-platform, reliable)
    Ollama,
}

impl Default for BackendType {
    fn default() -> Self {
        BackendType::Ollama
    }
}