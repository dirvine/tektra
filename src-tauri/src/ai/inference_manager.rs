use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};
use super::inference_backend::{InferenceBackend, InferenceConfig, InferenceMetrics, BackendType};
use super::gguf_inference::GGUFInference;

use super::mlx_inference::MLXInference;

/// Unified inference manager that can switch between different backends
pub struct InferenceManager {
    backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    backend_type: BackendType,
}

impl InferenceManager {
    /// Create a new inference manager with the specified backend type
    pub fn new(backend_type: BackendType) -> Result<Self> {
        let backend = Self::create_backend(backend_type)?;
        
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
            backend_type,
        })
    }
    
    /// Create the appropriate backend based on the type and platform
    fn create_backend(backend_type: BackendType) -> Result<Box<dyn InferenceBackend>> {
        match backend_type {
            BackendType::Auto => {
                // Try MLX first on Apple Silicon, fall back to GGUF
                if MLXInference::is_available() {
                    info!("Auto-selecting MLX backend for Apple Silicon");
                    return Ok(Box::new(MLXInference::new()));
                }
                
                info!("Auto-selecting GGUF backend (MLX not available)");
                Ok(Box::new(GGUFInference::new()))
            }
            
            BackendType::MLX => {
                if MLXInference::is_available() {
                    info!("Using MLX backend");
                    Ok(Box::new(MLXInference::new()))
                } else {
                    Err(anyhow::anyhow!(
                        "MLX backend requested but not available. To enable MLX:\n\
                        1. Install XCode Command Line Tools: xcode-select --install\n\
                        2. Ensure Metal compiler is available: xcrun -find metal\n\
                        3. Uncomment mlx-rs dependency in Cargo.toml\n\
                        4. Rebuild the application"
                    ))
                }
            }
            
            BackendType::GGUF => {
                info!("Using GGUF backend");
                Ok(Box::new(GGUFInference::new()))
            }
        }
    }
    
    /// Load a model file
    pub async fn load_model(&self, model_path: &Path) -> Result<()> {
        let mut backend = self.backend.lock().await;
        backend.load_model(model_path)
    }
    
    /// Check if a model is loaded
    pub async fn is_loaded(&self) -> bool {
        let backend = self.backend.lock().await;
        backend.is_loaded()
    }
    
    /// Generate text from a prompt
    pub async fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String> {
        let backend = self.backend.lock().await;
        backend.generate(prompt, config)
    }
    
    /// Generate text with performance metrics
    pub async fn generate_with_metrics(&self, prompt: &str, config: &InferenceConfig) -> Result<(String, InferenceMetrics)> {
        let backend = self.backend.lock().await;
        backend.generate_with_metrics(prompt, config)
    }
    
    /// Get the current backend name
    pub async fn backend_name(&self) -> String {
        let backend = self.backend.lock().await;
        backend.name().to_string()
    }
    
    /// Get the current backend type
    pub fn backend_type(&self) -> BackendType {
        self.backend_type
    }
    
    /// Switch to a different backend (requires reloading the model)
    pub async fn switch_backend(&mut self, backend_type: BackendType) -> Result<()> {
        warn!("Switching inference backend from {:?} to {:?}", self.backend_type, backend_type);
        
        let new_backend = Self::create_backend(backend_type)?;
        self.backend = Arc::new(Mutex::new(new_backend));
        self.backend_type = backend_type;
        
        info!("Backend switched successfully. Model needs to be reloaded.");
        Ok(())
    }
    
    /// Run a benchmark comparison between available backends
    pub async fn benchmark_backends(&self, prompt: &str, config: &InferenceConfig) -> Result<Vec<(String, InferenceMetrics)>> {
        let mut results = Vec::new();
        
        // Benchmark GGUF (always available)
        if self.backend_type != BackendType::GGUF {
            match self.benchmark_backend(BackendType::GGUF, prompt, config).await {
                Ok(metrics) => results.push(("GGUF".to_string(), metrics)),
                Err(e) => warn!("GGUF benchmark failed: {}", e),
            }
        }
        
        // Benchmark MLX if available
        if self.backend_type != BackendType::MLX && MLXInference::is_available() {
            match self.benchmark_backend(BackendType::MLX, prompt, config).await {
                Ok(metrics) => results.push(("MLX".to_string(), metrics)),
                Err(e) => warn!("MLX benchmark failed: {}", e),
            }
        }
        
        // Add current backend if not already benchmarked
        let backend_name = self.backend_name().await;
        if !results.iter().any(|(name, _)| name == &backend_name) {
            match self.generate_with_metrics(prompt, config).await {
                Ok((_, metrics)) => results.push((backend_name, metrics)),
                Err(e) => warn!("Current backend benchmark failed: {}", e),
            }
        }
        
        Ok(results)
    }
    
    /// Benchmark a specific backend
    async fn benchmark_backend(&self, backend_type: BackendType, prompt: &str, config: &InferenceConfig) -> Result<InferenceMetrics> {
        // Create temporary backend
        let backend = Self::create_backend(backend_type)?;
        
        // Note: This assumes the model is already loaded in the backend
        // In practice, you'd need to load the same model for fair comparison
        let (_, metrics) = backend.generate_with_metrics(prompt, config)?;
        
        Ok(metrics)
    }
    
    /// Get system information about available backends
    pub fn get_system_info() -> String {
        let mut info = String::new();
        
        info.push_str("Available Inference Backends:\n");
        info.push_str("- GGUF: Available (cross-platform)\n");
        
        if std::env::consts::OS == "macos" && std::env::consts::ARCH == "aarch64" {
            if MLXInference::is_available() {
                info.push_str("- MLX: Available (Apple Silicon detected)\n");
            } else {
                info.push_str("- MLX: Not available (requires XCode Command Line Tools with Metal compiler)\n");
            }
        } else {
            info.push_str("- MLX: Not available (requires Apple Silicon Mac)\n");
        }
        
        info.push_str(&format!("\nCurrent platform: {} {}\n", 
            std::env::consts::OS, 
            std::env::consts::ARCH
        ));
        
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_auto_backend_selection() {
        let manager = InferenceManager::new(BackendType::Auto).unwrap();
        
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            // Should select MLX on Apple Silicon if available
            let backend_name = manager.backend_name().await;
            assert!(backend_name == "MLX" || backend_name == "GGUF");
        }
        
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            // Should select GGUF on other platforms
            let backend_name = manager.backend_name().await;
            assert_eq!(backend_name, "GGUF");
        }
    }
    
    #[test]
    fn test_system_info() {
        let info = InferenceManager::get_system_info();
        assert!(info.contains("Available Inference Backends"));
        assert!(info.contains("GGUF: Available"));
    }
}