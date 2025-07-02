use crate::ai::unified_model_manager::{ModelBackend, ModelConfig, DeviceConfig, GenerationParams, MultimodalInput, BackendCapabilities};

#[cfg(feature = "multimodal-backends")]
mod backend_specific_tests {
    use super::*;
    use crate::ai::backends::{MistralRsBackend, LlamaCppBackend};
    
    #[tokio::test]
    async fn test_mistral_rs_backend_creation() {
        let backend = MistralRsBackend::new().unwrap();
        
        assert_eq!(backend.name(), "mistral_rs");
        assert!(!backend.is_loaded());
        
        let caps = backend.capabilities();
        assert!(caps.text_generation);
        assert!(caps.image_understanding);
        assert!(caps.audio_processing);
        assert!(caps.streaming);
        assert!(caps.function_calling);
    }

    #[tokio::test]
    async fn test_mistral_rs_model_support() {
        let backend = MistralRsBackend::new().unwrap();
        
        assert!(backend.supports_model("model.gguf").await);
        assert!(backend.supports_model("model.safetensors").await);
        assert!(backend.supports_model("mistral-7b").await);
        assert!(backend.supports_model("gemma-2b").await);
        assert!(backend.supports_model("llama3-8b").await);
        assert!(backend.supports_model("phi-3").await);
        assert!(backend.supports_model("qwen2").await);
    }

    #[tokio::test]
    async fn test_llama_cpp_backend_creation() {
        let backend = LlamaCppBackend::new().unwrap();
        
        assert_eq!(backend.name(), "llama_cpp");
        assert!(!backend.is_loaded());
        
        let caps = backend.capabilities();
        assert!(caps.text_generation);
        assert!(caps.image_understanding); // Via multimodal models
        assert!(!caps.audio_processing);
        assert!(caps.streaming);
        assert!(!caps.function_calling);
    }

    #[tokio::test]
    async fn test_llama_cpp_model_support() {
        let backend = LlamaCppBackend::new().unwrap();
        
        assert!(backend.supports_model("model.gguf").await);
        assert!(backend.supports_model("model.ggml").await); // Legacy
        assert!(backend.supports_model("anything-gguf").await);
        assert!(backend.supports_model("llama-3").await);
        assert!(backend.supports_model("mistral-v0.1").await);
        assert!(backend.supports_model("gemma-2b").await);
    }
    
    #[tokio::test]
    async fn test_memory_usage_unloaded() {
        let mistral_backend = MistralRsBackend::new().unwrap();
        assert_eq!(mistral_backend.memory_usage(), 0);
        
        let llama_backend = LlamaCppBackend::new().unwrap();
        assert_eq!(llama_backend.memory_usage(), 0);
    }
    
    #[tokio::test]
    async fn test_mock_generation_without_model() {
        let mut mistral_backend = MistralRsBackend::new().unwrap();
        
        let result = mistral_backend.generate_text("Hello", &GenerationParams::default()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No model loaded"));
        
        let mut llama_backend = LlamaCppBackend::new().unwrap();
        
        let result = llama_backend.generate_text("Hello", &GenerationParams::default()).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No model loaded"));
    }
    
    #[cfg(test)]
    mod llama_cpp_specific_tests {
        use super::*;
        use crate::ai::backends::LlamaCppBackend;
        
        #[test]
        fn test_supported_quantizations() {
            let quants = LlamaCppBackend::supported_quantizations();
            assert!(quants.contains(&"Q4_0"));
            assert!(quants.contains(&"Q8_0"));
            assert!(quants.contains(&"F16"));
            assert!(quants.contains(&"F32"));
        }
        
        #[test]
        fn test_estimate_model_size() {
            // 7B model with different quantizations
            let param_count = 7_000_000_000;
            
            let q4_size = LlamaCppBackend::estimate_model_size(param_count, "Q4_0");
            assert_eq!(q4_size, 3_500_000_000); // 3.5GB
            
            let q8_size = LlamaCppBackend::estimate_model_size(param_count, "Q8_0");
            assert_eq!(q8_size, 7_000_000_000); // 7GB
            
            let f16_size = LlamaCppBackend::estimate_model_size(param_count, "F16");
            assert_eq!(f16_size, 14_000_000_000); // 14GB
            
            let f32_size = LlamaCppBackend::estimate_model_size(param_count, "F32");
            assert_eq!(f32_size, 28_000_000_000); // 28GB
        }
    }
}

// Tests that work without backend implementations
#[tokio::test]
async fn test_quantization_formats() {
    let caps = BackendCapabilities {
        text_generation: true,
        image_understanding: true,
        audio_processing: false,
        video_processing: false,
        streaming: true,
        function_calling: false,
        quantization_formats: vec!["Q4_K_M".to_string(), "F16".to_string()],
        max_context_length: 32768,
    };
    
    assert!(caps.quantization_formats.contains(&"Q4_K_M".to_string()));
    assert!(caps.quantization_formats.contains(&"F16".to_string()));
}

#[tokio::test]
async fn test_generation_params_clone() {
    let params = GenerationParams {
        max_tokens: 1024,
        temperature: 0.8,
        top_p: 0.95,
        top_k: 50,
        repeat_penalty: 1.2,
        seed: Some(42),
        stop_sequences: vec!["END".to_string()],
        stream: true,
    };
    
    let cloned = params.clone();
    assert_eq!(cloned.max_tokens, 1024);
    assert_eq!(cloned.temperature, 0.8);
    assert_eq!(cloned.seed, Some(42));
    assert_eq!(cloned.stop_sequences, vec!["END".to_string()]);
}

#[tokio::test]
async fn test_multimodal_input_construction() {
    let input = MultimodalInput {
        text: Some("Describe this image".to_string()),
        images: vec![vec![1, 2, 3, 4, 5]],
        audio: None,
        video: None,
    };
    
    assert!(input.text.is_some());
    assert_eq!(input.images.len(), 1);
    assert!(input.audio.is_none());
    assert!(input.video.is_none());
}

#[tokio::test]
async fn test_device_config_variants() {
    let configs = vec![
        DeviceConfig::Cpu,
        DeviceConfig::Cuda(0),
        DeviceConfig::Cuda(1),
        DeviceConfig::Metal,
        DeviceConfig::Auto,
    ];
    
    for config in configs {
        match config {
            DeviceConfig::Cpu => assert!(true),
            DeviceConfig::Cuda(idx) => assert!(idx == 0 || idx == 1),
            DeviceConfig::Metal => assert!(true),
            DeviceConfig::Auto => assert!(true),
        }
    }
}

#[tokio::test]
async fn test_model_config_with_options() {
    let config = ModelConfig {
        model_id: "test/model".to_string(),
        model_path: Some("/path/to/model.gguf".to_string()),
        context_length: 16384,
        quantization: Some("Q4_K_M".to_string()),
        device: DeviceConfig::Metal,
        rope_scale: Some(2.0),
        template_name: Some("llama".to_string()),
    };
    
    assert_eq!(config.model_id, "test/model");
    assert!(config.model_path.is_some());
    assert_eq!(config.context_length, 16384);
    assert!(config.quantization.is_some());
    assert_eq!(config.rope_scale, Some(2.0));
}

#[tokio::test]
async fn test_backend_capabilities_serialization() {
    let caps = BackendCapabilities {
        text_generation: true,
        image_understanding: true,
        audio_processing: false,
        video_processing: false,
        streaming: true,
        function_calling: false,
        quantization_formats: vec!["Q4_K_M".to_string(), "F16".to_string()],
        max_context_length: 32768,
    };
    
    // Test serialization
    let json = serde_json::to_string(&caps).unwrap();
    assert!(json.contains("\"text_generation\":true"));
    assert!(json.contains("\"max_context_length\":32768"));
    
    // Test deserialization
    let deserialized: BackendCapabilities = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.max_context_length, 32768);
    assert_eq!(deserialized.quantization_formats.len(), 2);
}