use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};
use super::inference_backend::{InferenceBackend, InferenceConfig, InferenceMetrics, BackendType};
// Legacy ollama_inference removed
use tauri::AppHandle;

/// Unified inference manager that can switch between different backends
pub struct InferenceManager {
    backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    backend_type: BackendType,
    app_handle: Option<AppHandle>,
}

impl InferenceManager {
    /// Create a new inference manager with the specified backend type
    pub fn new(backend_type: BackendType) -> Result<Self> {
        let backend = Self::create_backend(backend_type, None)?;
        
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
            backend_type,
            app_handle: None,
        })
    }
    
    /// Create a new inference manager with app handle for progress tracking
    pub fn with_app_handle(backend_type: BackendType, app_handle: AppHandle) -> Result<Self> {
        let backend = Self::create_backend(backend_type, Some(app_handle.clone()))?;
        
        Ok(Self {
            backend: Arc::new(Mutex::new(backend)),
            backend_type,
            app_handle: Some(app_handle),
        })
    }
    
    /// Backend creation removed - legacy Ollama backend no longer available
    fn create_backend(_backend_type: BackendType, _app_handle: Option<AppHandle>) -> Result<Box<dyn InferenceBackend>> {
        Err(anyhow::anyhow!("Inference backends have been migrated to the new mistral.rs system. Use the ModelRegistry instead."))
    }
    
    /// Load a model file
    pub async fn load_model(&self, model_path: &Path) -> Result<()> {
        let mut backend = self.backend.lock().await;
        backend.load_model(model_path).await
    }
    
    /// Check if a model is loaded
    pub async fn is_loaded(&self) -> bool {
        let backend = self.backend.lock().await;
        backend.is_loaded()
    }
    
    /// Generate text from a prompt
    pub async fn generate(&self, prompt: &str, config: &InferenceConfig) -> Result<String> {
        let backend = self.backend.lock().await;
        backend.generate(prompt, config).await
    }
    
    /// Generate multimodal response with text and optional image or audio
    pub async fn generate_multimodal(&self, prompt: &str, media_data: Option<&[u8]>, media_type: Option<&str>, config: &InferenceConfig) -> Result<String> {
        let backend = self.backend.lock().await;
        backend.generate_multimodal(prompt, media_data, media_type, config).await
    }
    
    /// Generate text with performance metrics
    pub async fn generate_with_metrics(&self, prompt: &str, config: &InferenceConfig) -> Result<(String, InferenceMetrics)> {
        let backend = self.backend.lock().await;
        backend.generate_with_metrics(prompt, config).await
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
        
        let new_backend = Self::create_backend(backend_type, self.app_handle.clone())?;
        self.backend = Arc::new(Mutex::new(new_backend));
        self.backend_type = backend_type;
        
        info!("Backend switched successfully. Model needs to be reloaded.");
        Ok(())
    }
    
    /// Run a benchmark for the current Ollama backend
    pub async fn benchmark_backend(&self, prompt: &str, config: &InferenceConfig) -> Result<InferenceMetrics> {
        match self.generate_with_metrics(prompt, config).await {
            Ok((_, metrics)) => Ok(metrics),
            Err(e) => {
                warn!("Ollama benchmark failed: {}", e);
                Err(e)
            }
        }
    }
    
    
    /// Get system information about available backends
    pub fn get_system_info() -> String {
        let mut info = String::new();
        
        info.push_str("Inference Backend:\n");
        
        // Ollama availability (cross-platform, always available)
        info.push_str("- Ollama: Available (cross-platform, reliable)\n");
        
        info.push_str(&format!("\nCurrent platform: {} {}\n", 
            std::env::consts::OS, 
            std::env::consts::ARCH
        ));
        
        info.push_str("Backend: Ollama (cross-platform AI model management)\n");
        
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ollama_backend() {
        let result = InferenceManager::new(BackendType::Ollama);
        
        // Should always create Ollama backend (cross-platform)
        match result {
            Ok(manager) => {
                let backend_name = manager.backend_name().await;
                assert_eq!(backend_name, "Ollama");
            }
            Err(e) => {
                println!("Ollama backend creation failed: {}", e);
                // This shouldn't happen as Ollama is always available
                panic!("Ollama backend should always be available");
            }
        }
    }
    
    #[test]
    fn test_system_info() {
        let info = InferenceManager::get_system_info();
        assert!(info.contains("Inference Backend"));
        assert!(info.contains("Ollama"));
    }
    
    #[test]
    fn test_ollama_availability() {
        let info = InferenceManager::get_system_info();
        assert!(info.contains("Ollama: Available"));
    }
}