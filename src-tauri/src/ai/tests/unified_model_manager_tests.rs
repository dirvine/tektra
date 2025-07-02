use crate::ai::unified_model_manager::*;
use std::collections::HashMap;

#[tokio::test]
async fn test_unified_model_manager_creation() {
    let manager = UnifiedModelManager::new();
    assert!(!manager.is_loaded().await);
    assert_eq!(manager.current_backend().await, None);
    assert_eq!(manager.total_memory_usage().await, 0);
}

#[tokio::test]
async fn test_backend_registry() {
    let registry = BackendRegistry::new();
    let backends = registry.list_backends();
    assert!(backends.contains(&"enhanced_ollama".to_string()));
    assert_eq!(backends.len(), 1); // Only enhanced_ollama backend is currently enabled
}

#[tokio::test]
async fn test_backend_preferences() {
    let manager = UnifiedModelManager::new();
    
    let mut preferences = HashMap::new();
    preferences.insert("test_model".to_string(), vec!["mistral_rs".to_string(), "llama_cpp".to_string()]);
    
    manager.load_preferences(preferences).await.unwrap();
    
    // The actual backend selection would happen during model loading
    // This test just verifies preference loading works
}

#[tokio::test]
async fn test_model_config_creation() {
    let config = ModelConfig {
        model_id: "test_model".to_string(),
        model_path: Some("/path/to/model".to_string()),
        context_length: 4096,
        quantization: Some("Q4_K_M".to_string()),
        device: DeviceConfig::Auto,
        rope_scale: None,
        template_name: Some("chatml".to_string()),
    };
    
    assert_eq!(config.model_id, "test_model");
    assert_eq!(config.context_length, 4096);
}

#[tokio::test]
async fn test_generation_params_default() {
    let params = GenerationParams::default();
    
    assert_eq!(params.max_tokens, 512);
    assert_eq!(params.temperature, 0.7);
    assert_eq!(params.top_p, 0.9);
    assert_eq!(params.top_k, 40);
    assert_eq!(params.repeat_penalty, 1.1);
    assert_eq!(params.seed, None);
    assert!(params.stop_sequences.is_empty());
    assert!(!params.stream);
}

#[tokio::test]
async fn test_multimodal_input_creation() {
    let input = MultimodalInput {
        text: Some("Hello world".to_string()),
        images: vec![vec![1, 2, 3]],
        audio: Some(vec![4, 5, 6]),
        video: None,
    };
    
    assert_eq!(input.text, Some("Hello world".to_string()));
    assert_eq!(input.images.len(), 1);
    assert!(input.audio.is_some());
    assert!(input.video.is_none());
}

#[tokio::test]
async fn test_backend_capabilities() {
    let caps = BackendCapabilities {
        text_generation: true,
        image_understanding: true,
        audio_processing: false,
        video_processing: false,
        streaming: true,
        function_calling: false,
        quantization_formats: vec!["Q4_K_M".to_string()],
        max_context_length: 8192,
    };
    
    assert!(caps.text_generation);
    assert!(caps.image_understanding);
    assert!(!caps.audio_processing);
    assert_eq!(caps.max_context_length, 8192);
}

#[tokio::test]
async fn test_device_config() {
    let cpu = DeviceConfig::Cpu;
    let cuda = DeviceConfig::Cuda(0);
    let metal = DeviceConfig::Metal;
    let auto = DeviceConfig::Auto;
    
    match cpu {
        DeviceConfig::Cpu => assert!(true),
        _ => assert!(false),
    }
    
    match cuda {
        DeviceConfig::Cuda(idx) => assert_eq!(idx, 0),
        _ => assert!(false),
    }
}

// Mock backend for testing
pub struct MockBackend {
    loaded: bool,
    model_id: Option<String>,
}

impl MockBackend {
    pub fn new() -> Self {
        Self {
            loaded: false,
            model_id: None,
        }
    }
}

#[async_trait::async_trait]
impl ModelBackend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }
    
    async fn supports_model(&self, model_id: &str) -> bool {
        model_id.starts_with("mock_")
    }
    
    async fn load_model(&mut self, config: &ModelConfig) -> anyhow::Result<()> {
        self.loaded = true;
        self.model_id = Some(config.model_id.clone());
        Ok(())
    }
    
    async fn unload_model(&mut self) -> anyhow::Result<()> {
        self.loaded = false;
        self.model_id = None;
        Ok(())
    }
    
    fn is_loaded(&self) -> bool {
        self.loaded
    }
    
    async fn generate_text(&self, prompt: &str, _params: &GenerationParams) -> anyhow::Result<String> {
        if !self.loaded {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        Ok(format!("Mock response to: {}", prompt))
    }
    
    async fn generate_multimodal(&self, inputs: MultimodalInput, _params: &GenerationParams) -> anyhow::Result<String> {
        if !self.loaded {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        Ok(format!("Mock multimodal response: {} images", inputs.images.len()))
    }
    
    async fn generate_stream(&self, prompt: &str, _params: &GenerationParams) -> anyhow::Result<tokio::sync::mpsc::Receiver<String>> {
        if !self.loaded {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let prompt = prompt.to_string();
        
        tokio::spawn(async move {
            for word in prompt.split_whitespace() {
                let _ = tx.send(word.to_string()).await;
            }
        });
        
        Ok(rx)
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            text_generation: true,
            image_understanding: false,
            audio_processing: false,
            video_processing: false,
            streaming: true,
            function_calling: false,
            quantization_formats: vec!["mock".to_string()],
            max_context_length: 4096,
        }
    }
    
    fn memory_usage(&self) -> usize {
        if self.loaded { 1024 * 1024 } else { 0 }
    }
}

#[tokio::test]
async fn test_mock_backend() {
    let mut backend = MockBackend::new();
    
    assert_eq!(backend.name(), "mock");
    assert!(!backend.is_loaded());
    assert!(backend.supports_model("mock_model").await);
    assert!(!backend.supports_model("other_model").await);
    
    let config = ModelConfig {
        model_id: "mock_model".to_string(),
        model_path: None,
        context_length: 4096,
        quantization: None,
        device: DeviceConfig::Cpu,
        rope_scale: None,
        template_name: None,
    };
    
    backend.load_model(&config).await.unwrap();
    assert!(backend.is_loaded());
    
    let response = backend.generate_text("Hello", &GenerationParams::default()).await.unwrap();
    assert_eq!(response, "Mock response to: Hello");
    
    backend.unload_model().await.unwrap();
    assert!(!backend.is_loaded());
}