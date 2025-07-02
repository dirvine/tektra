use tektra::ai::{OllamaInference, InferenceBackend, InferenceConfig};
use std::path::Path;
use colored::*;

/// Demo of actual AI inference using bundled Ollama
/// Run with: cargo run --bin demo-inference --features e2e-testing

#[tokio::main]
async fn main() {
    println!("{}", "🤖 Tektra AI Demo with Bundled Ollama".cyan().bold());
    println!("{}", "=====================================".cyan());
    
    // Initialize bundled Ollama
    let mut ollama = OllamaInference::new();
    
    println!("\n📦 Initializing bundled Ollama...");
    match ollama.initialize().await {
        Ok(_) => println!("✅ Ollama ready!"),
        Err(e) => {
            eprintln!("❌ Failed to initialize: {}", e);
            return;
        }
    }
    
    // Load model
    let model = "gemma3n:e4b";
    println!("\n📥 Loading model {}...", model);
    match ollama.load_model(Path::new(model)).await {
        Ok(_) => println!("✅ Model loaded!"),
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            eprintln!("   The model will be downloaded automatically if not present.");
            eprintln!("   This may take several minutes on first run.");
            return;
        }
    }
    
    // Demo inference
    println!("\n🧠 Running AI inference demos...\n");
    
    let demos = vec![
        ("Math", "What is 25 + 17?", 50),
        ("Greeting", "Hello! How are you today?", 100),
        ("Code", "Write a Python function to reverse a string.", 200),
        ("Creative", "Write a haiku about artificial intelligence.", 150),
    ];
    
    let config = InferenceConfig::default();
    
    for (name, prompt, max_tokens) in demos {
        println!("{}:", name.green().bold());
        println!("  Q: {}", prompt.yellow());
        
        let mut demo_config = config.clone();
        demo_config.max_tokens = max_tokens;
        
        match ollama.generate(prompt, &demo_config).await {
            Ok(response) => {
                println!("  A: {}", response.trim().bright_white());
            }
            Err(e) => {
                eprintln!("  ❌ Error: {}", e);
            }
        }
        println!();
    }
    
    println!("{}", "✨ Demo complete!".green().bold());
    println!("\nThe bundled Ollama infrastructure is fully functional!");
    println!("No external Ollama installation was required.");
}