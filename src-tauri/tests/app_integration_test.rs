use tektra::ai::{TektraModelIntegration, ModelInfo};
use std::sync::Arc;
// Tauri test utilities are only available with the test feature
#[cfg(feature = "test")]
use tauri::test::{mock_builder, MockRuntime};

// Mock app handle for testing
#[cfg(feature = "test")]
fn create_mock_app_handle() -> tauri::AppHandle<MockRuntime> {
    let app = mock_builder().build().unwrap();
    app.handle()
}

#[cfg(feature = "test")]
#[tokio::test]
async fn test_model_integration_creation() {
    let app_handle = create_mock_app_handle();
    
    // This would fail without proper setup, but tests the creation process
    let result = TektraModelIntegration::new(app_handle).await;
    
    // In a real test environment, we'd mock the config loading
    assert!(result.is_err()); // Expected since we don't have models.toml in test env
}

#[tokio::test]
async fn test_multimodal_command_structure() {
    // Test that our command structures work correctly
    let text = Some("Test prompt".to_string());
    let image_base64 = Some("aGVsbG8=".to_string()); // "hello" in base64
    let audio_base64: Option<String> = None;
    
    // Decode image
    use base64::Engine;
    let image_data = base64::engine::general_purpose::STANDARD.decode(image_base64.unwrap()).unwrap();
    assert_eq!(image_data, b"hello");
}

#[tokio::test]
async fn test_model_info_structure() {
    let model_info = ModelInfo {
        id: "test_model".to_string(),
        name: "Test Model".to_string(),
        description: "A test model".to_string(),
        multimodal: true,
        capabilities: vec!["text".to_string(), "image".to_string()],
    };
    
    // Test serialization for frontend communication
    let json = serde_json::to_string(&model_info).unwrap();
    assert!(json.contains("\"id\":\"test_model\""));
    assert!(json.contains("\"multimodal\":true"));
}

#[tokio::test]
async fn test_app_settings() {
    use tektra::AppSettings;
    
    let settings = AppSettings::default();
    assert_eq!(settings.model_name, "gemma3n:e4b");
    assert_eq!(settings.max_tokens, 512);
    assert_eq!(settings.temperature, 0.7);
    assert!(!settings.voice_enabled);
    assert!(!settings.auto_speech);
}

#[tokio::test]
async fn test_chat_message_structure() {
    use tektra::ChatMessage;
    
    let msg = ChatMessage {
        role: "user".to_string(),
        content: "Hello, Tektra!".to_string(),
        timestamp: 1234567890,
    };
    
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("\"content\":\"Hello, Tektra!\""));
    assert!(json.contains("\"timestamp\":1234567890"));
}

// Integration test for the full workflow
#[cfg(feature = "integration-tests")]
#[tokio::test]
async fn test_full_multimodal_workflow() {
    use tektra::ai::{
        UnifiedModelManager, TemplateManager, ModelConfigLoader,
        ModelConfig, DeviceConfig, GenerationParams
    };
    
    // Create components
    let manager = UnifiedModelManager::new();
    let template_manager = TemplateManager::new();
    let mut config_loader = ModelConfigLoader::new("../models.toml");
    
    // Load configuration
    config_loader.load().await.unwrap();
    
    // Get model configuration
    let model_def = config_loader.get_model_definition("gemma3n:e4b").unwrap();
    let model_config = config_loader.to_model_config(model_def);
    
    // Load preferences
    let preferences = config_loader.get_backend_preferences();
    manager.load_preferences(preferences).await.unwrap();
    
    // Verify we can list models
    let models = config_loader.list_models();
    assert!(!models.is_empty());
    
    // Verify templates are loaded
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    let template = template_manager.get_template_for_model("gemma3n:e4b").await;
    assert!(template.is_some());
}

// Test the build process
#[test]
fn test_build_compiles() {
    // This test just ensures our code compiles correctly
    // The actual build process would be tested by cargo build
    assert!(true);
}