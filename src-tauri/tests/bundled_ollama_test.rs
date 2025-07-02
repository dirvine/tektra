use tektra::ai::{OllamaInference, InferenceBackend, InferenceConfig};
use std::path::Path;
use std::time::Instant;

/// Test that uses Tektra's bundled Ollama infrastructure
/// This test will:
/// 1. Automatically download Ollama if not present
/// 2. Start the embedded Ollama server
/// 3. Pull the model if needed
/// 4. Run actual inference
///
/// Run with: cargo test --test bundled_ollama_test -- --nocapture

#[tokio::test]
async fn test_bundled_ollama_basic_inference() {
    println!("🚀 Testing Tektra's Bundled Ollama Infrastructure");
    println!("================================================");
    
    let start = Instant::now();
    
    // Create OllamaInference - this will automatically download Ollama if needed
    println!("\n📦 Initializing Ollama (will download if needed)...");
    let mut ollama = OllamaInference::new();
    
    match ollama.initialize().await {
        Ok(_) => {
            println!("✅ Ollama initialized successfully in {:?}", start.elapsed());
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize Ollama: {}", e);
            panic!("Cannot proceed without Ollama");
        }
    }
    
    // Test model availability
    let model_name = "gemma3n:e4b";
    println!("\n🔍 Checking if model {} is available...", model_name);
    
    let has_model = ollama.is_model_available(model_name).await.unwrap_or(false);
    println!("   Model available: {}", if has_model { "Yes" } else { "No" });
    
    // Load the model (this will pull it if not available)
    println!("\n📥 Loading model (will download if needed)...");
    let load_start = Instant::now();
    
    match ollama.load_model(Path::new(model_name)).await {
        Ok(_) => {
            println!("✅ Model loaded successfully in {:?}", load_start.elapsed());
        }
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            eprintln!("   This might be due to:");
            eprintln!("   - Network issues preventing model download");
            eprintln!("   - Insufficient disk space");
            eprintln!("   - Model not available in Ollama registry");
            return;
        }
    }
    
    // Test basic inference
    println!("\n🧠 Testing inference...");
    let inference_config = InferenceConfig::default();
    
    let test_prompts = vec![
        ("Simple", "Hello! Please respond with a greeting.", 50),
        ("Math", "What is 2 + 2?", 30),
        ("Creative", "Write a haiku about coding.", 100),
    ];
    
    for (name, prompt, max_tokens) in test_prompts {
        println!("\n📝 {} Test:", name);
        println!("   Prompt: {}", prompt);
        
        let mut config = inference_config.clone();
        config.max_tokens = max_tokens;
        
        let inference_start = Instant::now();
        match ollama.generate(prompt, &config).await {
            Ok(response) => {
                let duration = inference_start.elapsed();
                println!("   ✅ Response: {}", response.trim());
                println!("   ⏱️  Duration: {:?}", duration);
                
                // Basic validation
                assert!(!response.is_empty(), "Response should not be empty");
                assert!(response.len() > 5, "Response should be meaningful");
            }
            Err(e) => {
                eprintln!("   ❌ Inference failed: {}", e);
                panic!("Inference should succeed");
            }
        }
    }
    
    // Test model info
    println!("\n📊 Model Status:");
    println!("   Loaded: {}", ollama.is_loaded());
    
    println!("\n🎉 All tests passed! Total time: {:?}", start.elapsed());
}

#[tokio::test]
async fn test_bundled_ollama_multimodal() {
    use tektra::ai::{Gemma3NProcessor, MultimodalInput};
    use tokio::fs;
    
    println!("🎨 Testing Multimodal with Bundled Ollama");
    println!("=========================================");
    
    // Initialize Ollama
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    // Load model
    let model_name = "gemma3n:e4b";
    ollama.load_model(Path::new(model_name)).await
        .expect("Failed to load model");
    
    // Test with image if available
    let image_path = Path::new("test_data/images/simple_shapes.png");
    if image_path.exists() {
        println!("\n🖼️  Testing with image...");
        
        let image_data = fs::read(image_path).await.expect("Failed to read image");
        let processor = Gemma3NProcessor::new();
        
        let input = MultimodalInput {
            text: Some("Describe what you see in this image.".to_string()),
            image_data: Some(image_data),
            audio_data: None,
            video_data: None,
        };
        
        let processed = processor.process_multimodal(input).await
            .expect("Failed to process multimodal input");
        
        let formatted = processor.format_for_gemma3n(&processed, Some("You are a helpful AI assistant."));
        
        let config = InferenceConfig::default();
        let response = ollama.generate(&formatted, &config).await
            .expect("Failed to generate response");
        
        println!("   Response: {}", &response[..200.min(response.len())].trim());
        assert!(!response.is_empty());
    } else {
        println!("   ⚠️  Test image not found, skipping image test");
    }
    
    println!("\n✅ Multimodal test completed!");
}

#[tokio::test]
async fn test_bundled_ollama_performance() {
    use std::time::Duration;
    
    println!("⚡ Testing Bundled Ollama Performance");
    println!("====================================");
    
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    ollama.load_model(Path::new("gemma3n:e4b")).await
        .expect("Failed to load model");
    
    let config = InferenceConfig::default();
    let prompts = vec![
        ("Short", "Hi!", 20),
        ("Medium", "Explain AI in one sentence.", 50),
        ("Long", "Write a brief story about a robot.", 200),
    ];
    
    let mut total_duration = Duration::ZERO;
    let mut total_tokens = 0;
    
    for (name, prompt, max_tokens) in prompts {
        let mut test_config = config.clone();
        test_config.max_tokens = max_tokens;
        
        let start = Instant::now();
        let response = ollama.generate(prompt, &test_config).await
            .expect("Failed to generate");
        let duration = start.elapsed();
        
        let token_count = response.split_whitespace().count();
        total_duration += duration;
        total_tokens += token_count;
        
        println!("\n{} prompt:", name);
        println!("  Duration: {:?}", duration);
        println!("  Tokens: ~{}", token_count);
        println!("  Speed: ~{:.1} tokens/sec", token_count as f64 / duration.as_secs_f64());
    }
    
    println!("\n📊 Overall Performance:");
    println!("  Total duration: {:?}", total_duration);
    println!("  Total tokens: ~{}", total_tokens);
    println!("  Average speed: ~{:.1} tokens/sec", 
             total_tokens as f64 / total_duration.as_secs_f64());
    
    // Performance assertions
    assert!(total_duration.as_secs() < 60, "Should complete in under 60 seconds");
    let avg_speed = total_tokens as f64 / total_duration.as_secs_f64();
    assert!(avg_speed > 1.0, "Should generate at least 1 token per second");
}

/// Test that verifies Ollama server management
#[tokio::test]
async fn test_bundled_ollama_server_lifecycle() {
    println!("🔄 Testing Ollama Server Lifecycle");
    println!("==================================");
    
    // Create first instance
    let mut ollama1 = OllamaInference::new();
    ollama1.initialize().await
        .expect("Failed to create first instance");
    
    // The server should be running now
    println!("✅ First instance created");
    
    // Create second instance - should reuse existing server
    let mut ollama2 = OllamaInference::new();
    ollama2.initialize().await
        .expect("Failed to create second instance");
    
    println!("✅ Second instance created (should reuse server)");
    
    // Both should work
    assert!(ollama1.is_model_available("gemma3n:e4b").await.is_ok());
    assert!(ollama2.is_model_available("gemma3n:e4b").await.is_ok());
    
    println!("✅ Both instances can query models");
    
    // Cleanup happens automatically when instances are dropped
    drop(ollama1);
    drop(ollama2);
    
    println!("✅ Cleanup completed");
}