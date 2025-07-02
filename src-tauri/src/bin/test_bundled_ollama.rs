use tektra::ai::OllamaInference;
use colored::*;

/// Simple test binary to verify bundled Ollama works
/// Run with: cargo run --bin test-bundled-ollama --features e2e-testing

#[tokio::main]
async fn main() {
    println!("{}", "🚀 Testing Bundled Ollama".cyan().bold());
    println!("{}", "========================".cyan());
    
    println!("\n📦 Creating OllamaInference instance...");
    println!("   This will automatically:");
    println!("   1. Check for system Ollama");
    println!("   2. Download bundled Ollama if needed (~10-50MB)");
    println!("   3. Start the Ollama server");
    
    let mut ollama = OllamaInference::new();
    
    println!("\n🔧 Initializing Ollama backend...");
    match ollama.initialize().await {
        Ok(_) => {
            println!("\n{} OllamaInference created successfully!", "✅".green());
            
            // Check if we can query models
            println!("\n🔍 Checking model availability...");
            match ollama.is_model_available("gemma3n:e4b").await {
                Ok(has_model) => {
                    if has_model {
                        println!("{} Model gemma3n:e4b is available!", "✅".green());
                    } else {
                        println!("{} Model gemma3n:e4b not found locally", "⚠️".yellow());
                        println!("   Run: ollama pull gemma3n:e4b");
                    }
                }
                Err(e) => {
                    println!("{} Failed to check model: {}", "❌".red(), e);
                }
            }
            
            println!("\n{} Bundled Ollama is working!", "🎉".green());
            println!("\nNext steps:");
            println!("1. Pull a model: ollama pull gemma3n:e4b");
            println!("2. Run full tests: cargo test --test bundled_ollama_test");
        }
        Err(e) => {
            println!("\n{} Failed to create OllamaInference: {}", "❌".red(), e);
            println!("\nPossible issues:");
            println!("- Network issues preventing download");
            println!("- Insufficient disk space");
            println!("- Permission issues in temp directory");
            
            // Print more detailed error info
            println!("\nDetailed error: {:?}", e);
        }
    }
}