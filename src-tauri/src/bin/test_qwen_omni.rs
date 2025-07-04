use tektra::inference::qwen_omni::QwenOmniModel;
use tektra::inference::{MultimodalInput, ModelConfig};
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Testing Qwen2.5-Omni model initialization...");
    
    // Test model creation
    let model = QwenOmniModel::new().await?;
    println!("✓ QwenOmniModel created successfully");
    
    // Test basic configuration
    let config = ModelConfig {
        model_id: "Qwen/Qwen2.5-Omni-7B".to_string(),
        quantization: Some("Q6_K".to_string()),
        context_window: 32768,
        device: tektra::inference::DeviceConfig::Auto,
        cache_dir: None,
        custom_params: std::collections::HashMap::new(),
    };
    
    println!("✓ Model configuration created");
    
    // Test basic input creation
    let input = MultimodalInput::Text("Hello, test the Qwen2.5-Omni model!".to_string());
    println!("✓ Multimodal input created");
    
    println!("🎉 Basic Qwen2.5-Omni components are working!");
    
    Ok(())
}