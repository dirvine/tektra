use crate::ai::model_config_loader::*;
use crate::ai::DeviceConfig;
use tokio::fs;

async fn create_test_config() -> String {
    r#"
[backends]
default = ["mistral_rs", "llama_cpp"]

[backends.mistral_rs]
enabled = true
flash_attention = true
mcp_enabled = true
device = "auto"

[backends.llama_cpp]
enabled = true
n_threads = 8
use_mmap = true
n_gpu_layers = -1
device = "auto"

[backends.ollama]
enabled = false
host = "localhost"
port = 11434
timeout_seconds = 300

[[models]]
id = "test_model"
name = "Test Model"
description = "A test model"
backend_preferences = ["mistral_rs", "llama_cpp"]
template = "chatml"
context_length = 4096
multimodal = false
capabilities = ["text"]

[[models]]
id = "multimodal_test"
name = "Multimodal Test"
description = "A multimodal test model"
backend_preferences = ["mistral_rs"]
template = "gemma"
context_length = 8192
multimodal = true
capabilities = ["text", "image", "audio"]
quantization = "Q4_K_M"
device = "metal"

[presets.test]
max_tokens = 256
temperature = 0.7
top_p = 0.9
top_k = 40
repeat_penalty = 1.1

[memory_limits]
mistral_rs = 8192
llama_cpp = 4096
ollama = 2048
total = 16384
"#
    .to_string()
}

#[tokio::test]
async fn test_model_config_loader_creation() {
    let loader = ModelConfigLoader::new("test_models.toml");
    assert!(loader.config().is_none());
}

#[tokio::test]
async fn test_load_config_from_string() {
    // Create temporary file
    let temp_path = "/tmp/test_models.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    assert!(loader.config().is_some());
    
    // Clean up
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_backend_preferences() {
    let temp_path = "/tmp/test_models_prefs.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let preferences = loader.get_backend_preferences();
    
    assert!(preferences.contains_key("_default"));
    assert_eq!(preferences["_default"], vec!["mistral_rs", "llama_cpp"]);
    
    assert!(preferences.contains_key("test_model"));
    assert_eq!(preferences["test_model"], vec!["mistral_rs", "llama_cpp"]);
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_get_model_definition() {
    let temp_path = "/tmp/test_models_def.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let model_def = loader.get_model_definition("test_model");
    assert!(model_def.is_some());
    
    let model = model_def.unwrap();
    assert_eq!(model.id, "test_model");
    assert_eq!(model.name, "Test Model");
    assert_eq!(model.context_length, 4096);
    assert!(!model.multimodal);
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_to_model_config() {
    let temp_path = "/tmp/test_models_config.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let model_def = loader.get_model_definition("multimodal_test").unwrap();
    let config = loader.to_model_config(model_def);
    
    assert_eq!(config.model_id, "multimodal_test");
    assert_eq!(config.context_length, 8192);
    assert_eq!(config.quantization, Some("Q4_K_M".to_string()));
    
    match config.device {
        DeviceConfig::Metal => assert!(true),
        _ => assert!(false, "Expected Metal device"),
    }
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_get_preset() {
    let temp_path = "/tmp/test_models_preset.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let preset = loader.get_preset("test");
    assert!(preset.is_some());
    
    let params = preset.unwrap();
    assert_eq!(params.max_tokens, 256);
    assert_eq!(params.temperature, 0.7);
    assert_eq!(params.top_p, 0.9);
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_list_models() {
    let temp_path = "/tmp/test_models_list.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let models = loader.list_models();
    assert_eq!(models.len(), 2);
    
    let test_model = models.iter().find(|m| m.id == "test_model");
    assert!(test_model.is_some());
    
    let multimodal = models.iter().find(|m| m.id == "multimodal_test");
    assert!(multimodal.is_some());
    assert!(multimodal.unwrap().multimodal);
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_backend_enabled() {
    let temp_path = "/tmp/test_models_backend.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    assert!(loader.is_backend_enabled("mistral_rs"));
    assert!(loader.is_backend_enabled("llama_cpp"));
    assert!(!loader.is_backend_enabled("ollama"));
    assert!(!loader.is_backend_enabled("unknown"));
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_memory_limits() {
    let temp_path = "/tmp/test_models_memory.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let mistral_limit = loader.get_memory_limit("mistral_rs");
    assert_eq!(mistral_limit, Some(8192 * 1024 * 1024));
    
    let total_limit = loader.get_total_memory_limit();
    assert_eq!(total_limit, 16384 * 1024 * 1024);
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_validation() {
    let temp_path = "/tmp/test_models_valid.toml";
    let config_content = create_test_config().await;
    fs::write(temp_path, &config_content).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    loader.load().await.unwrap();
    
    let issues = loader.validate();
    // Should have no issues with the test config
    assert!(issues.is_empty());
    
    fs::remove_file(temp_path).await.unwrap();
}

#[tokio::test]
async fn test_model_config_builder() {
    let model = ModelConfigBuilder::new("builder_test")
        .name("Builder Test Model")
        .description("A model built with the builder")
        .backends(vec!["mistral_rs".to_string()])
        .template("llama")
        .context_length(16384)
        .multimodal(true)
        .capabilities(vec!["text".to_string(), "image".to_string()])
        .quantization("Q5_K_M")
        .device("cuda:0")
        .model_file("/path/to/model.gguf")
        .build();
    
    assert_eq!(model.id, "builder_test");
    assert_eq!(model.name, "Builder Test Model");
    assert_eq!(model.template, "llama");
    assert_eq!(model.context_length, 16384);
    assert!(model.multimodal);
    assert_eq!(model.capabilities.len(), 2);
    assert_eq!(model.quantization, Some("Q5_K_M".to_string()));
    assert_eq!(model.device, Some("cuda:0".to_string()));
    assert_eq!(model.model_file, Some("/path/to/model.gguf".to_string()));
}

#[tokio::test]
async fn test_invalid_config() {
    let invalid_config = r#"
[backends]
default = []

[memory_limits]
mistral_rs = 10000
llama_cpp = 10000
ollama = 10000
total = 5000

[[models]]
# At least one model is required for valid config
"#;
    
    let temp_path = "/tmp/test_models_invalid.toml";
    fs::write(temp_path, invalid_config).await.unwrap();
    
    let mut loader = ModelConfigLoader::new(temp_path);
    // The config is missing required fields in models, so load should fail
    match loader.load().await {
        Ok(_) => {
            // If it somehow loads (partial config), validate should still find issues
            let issues = loader.validate();
            assert!(!issues.is_empty());
        },
        Err(e) => {
            // Expected: TOML parse error due to incomplete models section
            assert!(e.to_string().contains("models") || e.to_string().contains("TOML"));
        }
    }
    
    fs::remove_file(temp_path).await.unwrap();
}