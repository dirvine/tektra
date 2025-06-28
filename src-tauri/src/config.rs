use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::Result;
use crate::ai::BackendType;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// AI inference backend configuration
    pub inference: InferenceConfig,
    
    /// Model settings
    pub model: ModelConfig,
    
    /// Performance settings
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Backend type to use (Auto, MLX, GGUF)
    pub backend: BackendType,
    
    /// Enable backend benchmarking on startup
    pub benchmark_on_startup: bool,
    
    /// Maximum tokens for generation
    pub max_tokens: usize,
    
    /// Temperature for sampling
    pub temperature: f32,
    
    /// Top-p for nucleus sampling
    pub top_p: f32,
    
    /// Repetition penalty
    pub repeat_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Which Gemma model to use (E2B or E4B)
    pub gemma_variant: String,
    
    /// Use 8-bit quantization instead of 4-bit
    pub use_8bit: bool,
    
    /// Custom model path (if not using HuggingFace)
    pub custom_model_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of threads to use (None = auto)
    pub n_threads: Option<usize>,
    
    /// Enable memory profiling
    pub profile_memory: bool,
    
    /// Log inference metrics
    pub log_metrics: bool,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            inference: InferenceConfig {
                backend: BackendType::Auto,  // Auto will use MLX on Apple Silicon, GGUF elsewhere
                benchmark_on_startup: false,
                max_tokens: 512,
                temperature: 0.7,
                top_p: 0.9,
                repeat_penalty: 1.1,
            },
            model: ModelConfig {
                gemma_variant: "E2B".to_string(),
                use_8bit: false,
                custom_model_path: None,
            },
            performance: PerformanceConfig {
                n_threads: None,
                profile_memory: false,
                log_metrics: true,
            },
        }
    }
}

impl AppConfig {
    /// Load configuration from file
    pub fn load(path: &PathBuf) -> Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            let config: AppConfig = serde_json::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }
    
    /// Save configuration to file
    pub fn save(&self, path: &PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Get the configuration file path
    pub fn get_config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get config directory"))?
            .join("tektra");
        
        Ok(config_dir.join("config.json"))
    }
}

/// Environment variable overrides
impl AppConfig {
    pub fn apply_env_overrides(&mut self) {
        // Backend override
        if let Ok(backend_str) = std::env::var("TEKTRA_BACKEND") {
            match backend_str.to_lowercase().as_str() {
                "mlx" => self.inference.backend = BackendType::MLX,
                "gguf" => self.inference.backend = BackendType::GGUF,
                "auto" => self.inference.backend = BackendType::Auto,
                _ => tracing::warn!("Unknown backend type in TEKTRA_BACKEND: {}", backend_str),
            }
        }
        
        // Benchmark override
        if let Ok(benchmark_str) = std::env::var("TEKTRA_BENCHMARK") {
            self.inference.benchmark_on_startup = benchmark_str == "1" || benchmark_str.to_lowercase() == "true";
        }
        
        // Thread count override
        if let Ok(threads_str) = std::env::var("TEKTRA_THREADS") {
            if let Ok(threads) = threads_str.parse::<usize>() {
                self.performance.n_threads = Some(threads);
            }
        }
    }
}