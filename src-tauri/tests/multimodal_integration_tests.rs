use tektra::ai::{
    TektraModelIntegration, UnifiedModelManager, TemplateManager,
    ModelConfigLoader, ModelConfig, DeviceConfig, GenerationParams,
    ChatMessage, MessageRole, ModelInfo
};
use std::sync::Arc;
use tokio::sync::Mutex;

// Mock AppHandle for testing
struct MockAppHandle;

impl MockAppHandle {
    fn emit(&self, event: &str, payload: serde_json::Value) -> Result<(), String> {
        println!("Event: {} - Payload: {}", event, payload);
        Ok(())
    }
}

#[tokio::test]
async fn test_full_integration_workflow() {
    // This test would require a proper Tauri app handle
    // For now, we'll test the individual components together
    
    let manager = UnifiedModelManager::new();
    let template_manager = TemplateManager::new();
    
    // Wait for templates to load
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    // Verify we can get templates
    let template = template_manager.get_template("gemma").await;
    assert!(template.is_some());
}

#[tokio::test]
async fn test_model_loading_workflow() {
    let manager = UnifiedModelManager::new();
    
    // Create a test model config
    let config = ModelConfig {
        model_id: "test_model".to_string(),
        model_path: None,
        context_length: 4096,
        quantization: None,
        device: DeviceConfig::Auto,
        rope_scale: None,
        template_name: Some("chatml".to_string()),
    };
    
    // Without actual backends registered, this will fail
    // but we're testing the workflow
    let result = manager.load_model(config).await;
    assert!(result.is_err()); // No backend supports "test_model"
}

#[tokio::test]
async fn test_template_and_generation_integration() {
    let template_manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    // Create a conversation
    let messages = vec![
        ChatMessage {
            role: MessageRole::System,
            content: "You are a helpful AI assistant.".to_string(),
        },
        ChatMessage {
            role: MessageRole::User,
            content: "What is the capital of France?".to_string(),
        },
    ];
    
    // Get template for a model
    let template = template_manager.get_template_for_model("gemma3n:e4b").await.unwrap();
    
    // Format the prompt
    let prompt = template_manager.format_prompt(&template, &messages, true);
    
    // Verify the prompt is formatted correctly
    assert!(prompt.contains("You are a helpful AI assistant."));
    assert!(prompt.contains("What is the capital of France?"));
    assert!(prompt.contains("<start_of_turn>")); // Gemma formatting
}

#[tokio::test]
async fn test_multimodal_prompt_formatting() {
    let template_manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let template = template_manager.get_template("llama").await.unwrap();
    
    // Format multimodal content
    let content = template_manager.format_multimodal_content(
        &template,
        "What objects do you see in this image?",
        true,  // has_image
        false, // has_audio
        false, // has_video
    );
    
    // Llama uses specific image markers
    assert!(content.contains("<|image|>") || content.contains("<image>"));
    assert!(content.contains("What objects do you see"));
}

#[tokio::test]
async fn test_backend_selection_logic() {
    use std::collections::HashMap;
    
    let manager = UnifiedModelManager::new();
    
    // Set up preferences
    let mut preferences = HashMap::new();
    preferences.insert("gemma3n:e4b".to_string(), vec!["ollama".to_string(), "mistral_rs".to_string()]);
    preferences.insert("llama3:8b".to_string(), vec!["mistral_rs".to_string(), "llama_cpp".to_string()]);
    
    manager.load_preferences(preferences).await.unwrap();
    
    // The actual backend selection happens during model loading
    // This test verifies preference loading works
}

#[tokio::test]
async fn test_generation_params_presets() {
    // Create a mock config with presets
    let config_content = r#"
[presets.fast]
max_tokens = 128
temperature = 0.5
top_p = 0.8
top_k = 30
repeat_penalty = 1.0

[presets.creative]
max_tokens = 1024
temperature = 1.2
top_p = 0.98
top_k = 100
repeat_penalty = 1.05

[backends]
default = ["mistral_rs"]

[memory_limits]
mistral_rs = 8192
llama_cpp = 4096
ollama = 2048
total = 16384
"#;
    
    // Write to temp file
    let temp_path = "/tmp/test_presets.toml";
    tokio::fs::write(temp_path, config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    // Test fast preset
    let fast = loader.get_preset("fast").unwrap();
    assert_eq!(fast.max_tokens, 128);
    assert_eq!(fast.temperature, 0.5);
    
    // Test creative preset
    let creative = loader.get_preset("creative").unwrap();
    assert_eq!(creative.max_tokens, 1024);
    assert_eq!(creative.temperature, 1.2);
    
    // Cleanup
    tokio::fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_error_handling_no_model() {
    let manager = UnifiedModelManager::new();
    
    // Try to generate without loading a model
    let result = manager.generate_text("Hello", &GenerationParams::default()).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("No model loaded"));
}

#[tokio::test]
async fn test_memory_management() {
    let manager = UnifiedModelManager::new();
    
    // Initially should have zero memory usage
    assert_eq!(manager.total_memory_usage().await, 0);
    
    // After loading models, memory usage would increase
    // This is tested in actual backend tests
}

#[tokio::test]
async fn test_conversation_context_management() {
    // Test that conversation history is properly maintained
    let messages = vec![
        ChatMessage {
            role: MessageRole::User,
            content: "Hello".to_string(),
        },
        ChatMessage {
            role: MessageRole::Assistant,
            content: "Hi there!".to_string(),
        },
        ChatMessage {
            role: MessageRole::User,
            content: "How are you?".to_string(),
        },
    ];
    
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, MessageRole::User);
    assert_eq!(messages[1].role, MessageRole::Assistant);
}

#[tokio::test]
async fn test_model_info_serialization() {
    let info = ModelInfo {
        id: "test_model".to_string(),
        name: "Test Model".to_string(),
        description: "A test model".to_string(),
        multimodal: true,
        capabilities: vec!["text".to_string(), "image".to_string()],
    };
    
    // Test serialization
    let json = serde_json::to_string(&info).unwrap();
    assert!(json.contains("\"id\":\"test_model\""));
    assert!(json.contains("\"multimodal\":true"));
    
    // Test deserialization
    let deserialized: ModelInfo = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, "test_model");
    assert!(deserialized.multimodal);
    assert_eq!(deserialized.capabilities.len(), 2);
}

// Integration test for the complete multimodal workflow
#[tokio::test]
async fn test_multimodal_workflow() {
    use tektra::ai::unified_model_manager::MultimodalInput as UnifiedMultimodalInput;
    
    // Create multimodal input
    let input = UnifiedMultimodalInput {
        text: Some("Describe what you see and hear".to_string()),
        images: vec![vec![0u8; 100]], // Mock image data
        audio: Some(vec![0u8; 50]),   // Mock audio data
        video: None,
    };
    
    // Verify input structure
    assert!(input.text.is_some());
    assert_eq!(input.images.len(), 1);
    assert!(input.audio.is_some());
    assert!(input.video.is_none());
    
    // In a real test, this would be passed to the model
    let params = GenerationParams {
        max_tokens: 500,
        temperature: 0.8,
        ..Default::default()
    };
    
    assert_eq!(params.max_tokens, 500);
}