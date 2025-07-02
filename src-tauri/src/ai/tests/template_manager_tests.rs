use crate::ai::template_manager::*;

#[tokio::test]
async fn test_template_manager_creation() {
    let manager = TemplateManager::new();
    
    // Wait for default templates to load
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let templates = manager.list_templates().await;
    assert!(!templates.is_empty());
}

#[tokio::test]
async fn test_default_templates() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    // Check that default templates are loaded
    let gemma = manager.get_template("gemma").await;
    assert!(gemma.is_some());
    
    let llama = manager.get_template("llama").await;
    assert!(llama.is_some());
    
    let mistral = manager.get_template("mistral").await;
    assert!(mistral.is_some());
    
    let chatml = manager.get_template("chatml").await;
    assert!(chatml.is_some());
}

#[tokio::test]
async fn test_template_formatting() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let template = manager.get_template("gemma").await.unwrap();
    
    let messages = vec![
        ChatMessage {
            role: MessageRole::System,
            content: "You are a helpful assistant.".to_string(),
        },
        ChatMessage {
            role: MessageRole::User,
            content: "Hello!".to_string(),
        },
        ChatMessage {
            role: MessageRole::Assistant,
            content: "Hi there!".to_string(),
        },
    ];
    
    let prompt = manager.format_prompt(&template, &messages, false);
    
    // Check that the prompt contains expected elements
    assert!(prompt.contains("<start_of_turn>system"));
    assert!(prompt.contains("You are a helpful assistant."));
    assert!(prompt.contains("<start_of_turn>user"));
    assert!(prompt.contains("Hello!"));
    assert!(prompt.contains("<start_of_turn>model"));
    assert!(prompt.contains("Hi there!"));
}

#[tokio::test]
async fn test_template_with_generation_prompt() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let template = manager.get_template("gemma").await.unwrap();
    
    let messages = vec![
        ChatMessage {
            role: MessageRole::User,
            content: "What is 2+2?".to_string(),
        },
    ];
    
    let prompt = manager.format_prompt(&template, &messages, true);
    
    // Should end with assistant prefix for generation
    assert!(prompt.ends_with("<start_of_turn>model\n"));
}

#[tokio::test]
async fn test_model_template_mapping() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    // Test model name matching
    let template = manager.get_template_for_model("gemma3n:e4b").await;
    assert!(template.is_some());
    assert_eq!(template.unwrap().name, "gemma");
    
    let template = manager.get_template_for_model("llama3-8b-instruct").await;
    assert!(template.is_some());
    assert_eq!(template.unwrap().name, "llama");
    
    let template = manager.get_template_for_model("mistral-7b-v0.1").await;
    assert!(template.is_some());
    assert_eq!(template.unwrap().name, "mistral");
    
    // Unknown model should default to chatml
    let template = manager.get_template_for_model("unknown-model").await;
    assert!(template.is_some());
    assert_eq!(template.unwrap().name, "chatml");
}

#[tokio::test]
async fn test_multimodal_markers() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let template = manager.get_template("gemma").await.unwrap();
    
    let content = manager.format_multimodal_content(
        &template,
        "What's in this image?",
        true,  // has_image
        false, // has_audio
        false, // has_video
    );
    
    assert!(content.contains("<image>"));
    assert!(content.contains("</image>"));
    assert!(content.contains("What's in this image?"));
}

#[tokio::test]
async fn test_custom_template() {
    let manager = TemplateManager::new();
    
    let custom_template = PromptTemplate {
        name: "custom".to_string(),
        system_prefix: Some("System: ".to_string()),
        system_suffix: Some("\n".to_string()),
        user_prefix: "User: ".to_string(),
        user_suffix: "\n".to_string(),
        assistant_prefix: "Assistant: ".to_string(),
        assistant_suffix: "\n".to_string(),
        add_bos: false,
        add_eos: false,
        multimodal_markers: MultimodalMarkers::default(),
        supports_system: true,
        stop_sequences: vec!["\n\n".to_string()],
    };
    
    manager.add_template(custom_template).await.unwrap();
    
    let retrieved = manager.get_template("custom").await;
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().name, "custom");
}

#[tokio::test]
async fn test_template_without_system_support() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let template = manager.get_template("mistral").await.unwrap();
    assert!(!template.supports_system);
    
    let messages = vec![
        ChatMessage {
            role: MessageRole::System,
            content: "System prompt".to_string(),
        },
        ChatMessage {
            role: MessageRole::User,
            content: "User message".to_string(),
        },
    ];
    
    let prompt = manager.format_prompt(&template, &messages, false);
    
    // System message should be merged into user message
    assert!(prompt.contains("System prompt"));
    assert!(prompt.contains("User message"));
}

#[tokio::test]
async fn test_stop_sequences() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let gemma_template = manager.get_template("gemma").await.unwrap();
    assert!(gemma_template.stop_sequences.contains(&"<end_of_turn>".to_string()));
    
    let llama_template = manager.get_template("llama").await.unwrap();
    assert!(llama_template.stop_sequences.contains(&"<|eot_id|>".to_string()));
}

#[tokio::test]
async fn test_multimodal_markers_variations() {
    let markers = MultimodalMarkers {
        image_start: "[IMG]".to_string(),
        image_end: "[/IMG]".to_string(),
        audio_start: "[AUDIO]".to_string(),
        audio_end: "[/AUDIO]".to_string(),
        video_start: "[VIDEO]".to_string(),
        video_end: "[/VIDEO]".to_string(),
        replace_content: true,
    };
    
    assert_eq!(markers.image_start, "[IMG]");
    assert!(markers.replace_content);
}

#[tokio::test]
async fn test_message_role_serialization() {
    let system = MessageRole::System;
    let user = MessageRole::User;
    let assistant = MessageRole::Assistant;
    
    // Test serialization
    let system_json = serde_json::to_string(&system).unwrap();
    assert_eq!(system_json, "\"system\"");
    
    let user_json = serde_json::to_string(&user).unwrap();
    assert_eq!(user_json, "\"user\"");
    
    let assistant_json = serde_json::to_string(&assistant).unwrap();
    assert_eq!(assistant_json, "\"assistant\"");
}

#[tokio::test]
async fn test_template_list() {
    let manager = TemplateManager::new();
    tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    
    let templates = manager.list_templates().await;
    
    // Should have at least the default templates
    assert!(templates.len() >= 5);
    assert!(templates.contains(&"gemma".to_string()));
    assert!(templates.contains(&"llama".to_string()));
    assert!(templates.contains(&"mistral".to_string()));
    assert!(templates.contains(&"chatml".to_string()));
    assert!(templates.contains(&"phi".to_string()));
}