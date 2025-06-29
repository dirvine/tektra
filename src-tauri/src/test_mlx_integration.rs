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
    
    // Test MLX (may fail without XCode tools on non-Apple Silicon)
    match ai::InferenceManager::new(ai::BackendType::MLX) {
        Ok(manager) => {
            println!("✓ MLX backend created successfully");
            println!("  Selected backend: {}", manager.backend_name().await);
        }
        Err(e) => println!("✗ MLX backend failed: {}", e),
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
    if let Ok(manager) = ai::InferenceManager::new(ai::BackendType::MLX) {
        match manager.benchmark_backend("What is 2+2?", &ai::InferenceConfig::default()).await {
            Ok(metrics) => {
                println!("  MLX benchmark completed (placeholder)");
                println!("  Metrics: {:?}", metrics);
            }
            Err(e) => println!("  Benchmark failed (expected without model): {}", e),
        }
    }
    
    println!("\n=== Test Complete ===");
}