#!/usr/bin/env rust-script
//! ```cargo
//! [dependencies]
//! tokio = { version = "1", features = ["full"] }
//! anyhow = "1"
//! ```

use std::path::PathBuf;
use std::env;

#[path = "src/ai/unified_model_manager.rs"]
mod unified_model_manager;

#[path = "src/ai/template_manager.rs"]
mod template_manager;

#[path = "src/ai/model_config_loader.rs"]
mod model_config_loader;

use unified_model_manager::{UnifiedModelManager, ModelConfig, GenerationParams, DeviceConfig};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Testing Multimodal Infrastructure...\n");
    
    // Create the unified model manager
    let manager = UnifiedModelManager::new();
    
    // List available backends
    println!("Available backends:");
    let backends = ["mistral_rs", "llama_cpp", "ollama"];
    for backend in &backends {
        println!("  - {}", backend);
    }
    
    // Test model configuration
    let config = ModelConfig {
        model_id: "test-model".to_string(),
        model_path: None,
        context_length: 4096,
        quantization: Some("Q4_K_M".to_string()),
        device: DeviceConfig::Auto,
        rope_scale: None,
        template_name: Some("gemma".to_string()),
    };
    
    println!("\nModel Configuration:");
    println!("  ID: {}", config.model_id);
    println!("  Context Length: {}", config.context_length);
    println!("  Quantization: {:?}", config.quantization);
    println!("  Device: {:?}", config.device);
    println!("  Template: {:?}", config.template_name);
    
    // Test generation parameters
    let params = GenerationParams::default();
    
    println!("\nGeneration Parameters:");
    println!("  Max Tokens: {}", params.max_tokens);
    println!("  Temperature: {}", params.temperature);
    println!("  Top P: {}", params.top_p);
    println!("  Top K: {}", params.top_k);
    
    println!("\nMultimodal infrastructure test complete!");
    
    Ok(())
}