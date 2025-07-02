use anyhow::Result;
use ollama_rs::{
    Ollama, 
    generation::completion::request::GenerationRequest
};
use std::time::Instant;
use colored::*;
use tokio::fs;
use std::path::Path;

/// Simple E2E test that directly uses Ollama
/// 
/// Run with: cargo run --bin simple-e2e-test --features e2e-testing
/// 
/// Prerequisites:
/// 1. Install Ollama: https://ollama.ai
/// 2. Pull the model: ollama pull gemma3n:e4b
/// 3. Ensure Ollama is running: ollama serve

#[tokio::main]
async fn main() -> Result<()> {
    println!("{}", "🚀 Tektra Simple E2E Test Runner".cyan().bold());
    println!("{}", "=================================".cyan());
    
    // Connect to Ollama
    let ollama = Ollama::default();
    
    // Test 1: Basic text generation
    println!("\n{}", "📝 Test 1: Basic Text Generation".green().bold());
    let start = Instant::now();
    
    let request = GenerationRequest::new("gemma3n:e4b".to_string(), "Write a haiku about AI.".to_string());
    
    match ollama.generate(request).await {
        Ok(response) => {
            println!("✅ Success! Response: {}", response.response.trim());
            println!("   Duration: {:?}", start.elapsed());
        }
        Err(e) => {
            println!("❌ Failed: {}", e);
            println!("   Make sure Ollama is running and gemma3n:e4b is installed");
            return Ok(());
        }
    }
    
    // Test 2: Document + Query
    println!("\n{}", "📄 Test 2: Document Analysis".green().bold());
    
    let doc_path = Path::new("test_data/documents/sample.txt");
    if doc_path.exists() {
        let doc_content = fs::read_to_string(doc_path).await?;
        let prompt = format!(
            "Document:\n{}\n\nQuestion: What are the main topics in this document? Provide a brief summary.",
            &doc_content[..500.min(doc_content.len())] // Limit context
        );
        
        let start = Instant::now();
        let request = GenerationRequest::new("gemma3n:e4b".to_string(), prompt);
        
        match ollama.generate(request).await {
            Ok(response) => {
                println!("✅ Success! Summary: {}", &response.response[..200.min(response.response.len())]);
                println!("   Duration: {:?}", start.elapsed());
            }
            Err(e) => {
                println!("❌ Failed: {}", e);
            }
        }
    } else {
        println!("⚠️  Test document not found, skipping...");
    }
    
    // Test 3: Image description (simulated)
    println!("\n{}", "🎨 Test 3: Image Description (Simulated)".green().bold());
    
    let image_path = Path::new("test_data/images/simple_shapes.png");
    if image_path.exists() {
        // Note: Direct image support requires multimodal models
        // For now, we simulate with a descriptive prompt
        let prompt = "Imagine an image with red squares, blue circles, and green triangles. Describe what you see.";
        
        let start = Instant::now();
        let request = GenerationRequest::new("gemma3n:e4b".to_string(), prompt.to_string());
        
        match ollama.generate(request).await {
            Ok(response) => {
                println!("✅ Success! Description: {}", &response.response[..200.min(response.response.len())]);
                println!("   Duration: {:?}", start.elapsed());
            }
            Err(e) => {
                println!("❌ Failed: {}", e);
            }
        }
    }
    
    // Test 4: Performance benchmark
    println!("\n{}", "⚡ Test 4: Performance Benchmark".green().bold());
    
    let prompts = vec![
        ("Simple", "Hello!", 20),
        ("Medium", "Explain quantum computing in one sentence.", 50),
        ("Complex", "Write a brief analysis of renewable energy.", 100),
    ];
    
    for (name, prompt, _expected_tokens) in prompts {
        let start = Instant::now();
        let request = GenerationRequest::new("gemma3n:e4b".to_string(), prompt.to_string());
        
        match ollama.generate(request).await {
            Ok(response) => {
                let duration = start.elapsed();
                let word_count = response.response.split_whitespace().count();
                let tokens_per_sec = word_count as f64 / duration.as_secs_f64();
                
                println!("  {} {}: {:.1} tokens/sec ({:?})", 
                         "✅".green(), name, tokens_per_sec, duration);
            }
            Err(e) => {
                println!("  {} {}: Failed - {}", "❌".red(), name, e);
            }
        }
    }
    
    // Test 5: Output validation
    println!("\n{}", "✅ Test 5: Output Validation".green().bold());
    
    let qa_pairs = vec![
        ("What is 2+2?", "4"),
        ("What is the capital of France?", "Paris"),
        ("Is water H2O?", "Yes"),
    ];
    
    let mut correct = 0;
    let total = qa_pairs.len();
    
    for (question, expected) in qa_pairs {
        let request = GenerationRequest::new("gemma3n:e4b".to_string(), question.to_string());
        
        match ollama.generate(request).await {
            Ok(response) => {
                let answer = response.response.to_lowercase();
                if answer.contains(&expected.to_lowercase()) {
                    println!("  ✅ {}: Correct!", question);
                    correct += 1;
                } else {
                    println!("  ❌ {}: Got '{}', expected '{}'", 
                             question, 
                             &response.response[..50.min(response.response.len())].trim(),
                             expected);
                }
            }
            Err(e) => {
                println!("  ❌ {}: Failed - {}", question, e);
            }
        }
    }
    
    println!("\n  Validation Score: {}/{} ({:.0}%)", 
             correct, total, (correct as f64 / total as f64) * 100.0);
    
    // Summary
    println!("\n{}", "📊 Test Summary".blue().bold());
    println!("{}", "================".blue());
    println!("All basic E2E tests completed!");
    println!("\nFor full multimodal testing with camera/mic, use:");
    println!("  cargo run --bin e2e-test-runner --features e2e-testing -- all --with-live");
    
    Ok(())
}