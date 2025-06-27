#!/usr/bin/env rust-script

//! Simple test script to verify MLX integration
//! Run with: cargo run --bin test_mlx_integration

use std::path::PathBuf;
use tokio;

#[path = "ai/mod.rs"]
mod ai;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("=== MLX Integration Test ===\n");
    
    // Check system info
    let system_info = ai::InferenceManager::get_system_info();
    println!("System Information:");
    println!("{}", system_info);
    
    // Test backend creation
    println!("\nTesting Backend Creation:");
    
    // Test GGUF (should always work)
    match ai::InferenceManager::new(ai::BackendType::GGUF) {
        Ok(_) => println!("✓ GGUF backend created successfully"),
        Err(e) => println!("✗ GGUF backend failed: {}", e),
    }
    
    // Test MLX (expected to fail without XCode tools)
    match ai::InferenceManager::new(ai::BackendType::MLX) {
        Ok(_) => println!("✓ MLX backend created successfully"),
        Err(e) => println!("✗ MLX backend failed (expected): {}", e),
    }
    
    // Test Auto (should fall back to GGUF)
    match ai::InferenceManager::new(ai::BackendType::Auto) {
        Ok(manager) => {
            println!("✓ Auto backend created successfully");
            println!("  Selected backend: {}", manager.backend_name().await);
        }
        Err(e) => println!("✗ Auto backend failed: {}", e),
    }
    
    // Test configuration
    println!("\nTesting Configuration:");
    
    // Set environment variable for backend
    std::env::set_var("TEKTRA_BACKEND", "mlx");
    let mut config = crate::config::AppConfig::default();
    config.apply_env_overrides();
    println!("Backend from env: {:?}", config.inference.backend);
    
    // Benchmark test (with dummy model)
    println!("\nTesting Benchmark (without actual model):");
    if let Ok(manager) = ai::InferenceManager::new(ai::BackendType::Auto) {
        match manager.benchmark_backends("What is 2+2?", &ai::InferenceConfig::default()).await {
            Ok(results) => {
                for (backend, metrics) in results {
                    println!("  {} benchmark completed (placeholder)", backend);
                }
            }
            Err(e) => println!("  Benchmark failed (expected without model): {}", e),
        }
    }
    
    println!("\n=== Test Complete ===");
}