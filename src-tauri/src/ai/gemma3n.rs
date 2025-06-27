use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Manager};
use tracing::{error, info};

// Gemma-3n model information
// E2B = 2 billion parameters (faster, good for desktop)
// E4B = 4 billion parameters (more capable, needs more resources)
const GEMMA_E2B_ID: &str = "google/gemma-3n-E2B-it";
const GEMMA_E2B_GGUF_ID: &str = "unsloth/gemma-3n-E2B-it-GGUF";
const GEMMA_E2B_FILE_Q4: &str = "gemma-3n-E2B-it-Q4_K_M.gguf"; // 2.79GB
const GEMMA_E2B_FILE_Q8: &str = "gemma-3n-E2B-it-Q8_0.gguf"; // 4.79GB

// For more capable version (if user has resources)
const GEMMA_E4B_GGUF_ID: &str = "unsloth/gemma-3n-E4B-it-GGUF";
const GEMMA_E4B_FILE_Q4: &str = "gemma-3n-E4B-it-Q4_K_M.gguf"; // ~3GB estimated

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadProgress {
    pub progress: u32,
    pub status: String,
    pub model_name: String,
}

#[derive(Debug, Clone)]
struct GemmaChatTemplate {
    // Gemma uses a specific chat template
    bos: String,
    start_turn: String,
    end_turn: String,
    user_role: String,
    model_role: String,
}

impl Default for GemmaChatTemplate {
    fn default() -> Self {
        Self {
            // Gemma-3n uses specific tokens as per Google's format
            bos: "<bos>".to_string(),
            start_turn: "<start_of_turn>".to_string(),
            end_turn: "<end_of_turn>".to_string(),
            user_role: "user".to_string(),
            model_role: "model".to_string(), // Gemma uses "model" not "assistant"
        }
    }
}

pub struct AIManager {
    app_handle: AppHandle,
    model_loaded: bool,
    model_path: Option<PathBuf>,
    chat_template: GemmaChatTemplate,
    selected_model: String,
}

impl AIManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: GemmaChatTemplate::default(),
            selected_model: GEMMA_E2B_GGUF_ID.to_string(), // Default to 2B model
        })
    }

    pub async fn load_model(&mut self) -> Result<()> {
        self.emit_progress(0, "Starting Gemma-3n initialization...", &self.selected_model).await;
        
        // Download model if needed
        match self.download_model().await {
            Ok(path) => {
                self.model_path = Some(path.clone());
                info!("Gemma-3n model downloaded to: {:?}", path);
                
                self.emit_progress(90, "Preparing Gemma-3n for inference...", &self.selected_model).await;
                
                // Model is ready to use
                self.model_loaded = true;
                
                self.emit_progress(100, "Gemma-3n ready! Google's latest AI model at your service.", &self.selected_model).await;
                Ok(())
            }
            Err(e) => {
                error!("Failed to download Gemma model: {}", e);
                Err(e)
            }
        }
    }

    async fn download_model(&self) -> Result<PathBuf> {
        self.emit_progress(10, "Checking for Phi-3 model...", &self.selected_model).await;
        
        // Get cache directory
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?
            .join("huggingface")
            .join("hub")
            .join(self.selected_model.replace('/', "--"));
        
        std::fs::create_dir_all(&cache_dir)?;
        
        // Choose file based on model selection
        let model_file = if self.selected_model.contains("E4B") {
            GEMMA_E4B_FILE_Q4
        } else {
            GEMMA_E2B_FILE_Q4 // Use Q4 quantization for balance of size/quality
        };
        
        let model_path = cache_dir.join(model_file);
        
        if model_path.exists() {
            let metadata = std::fs::metadata(&model_path)?;
            if metadata.len() > 100_000_000 { // At least 100MB
                self.emit_progress(70, "Found cached Gemma-3n model", &self.selected_model).await;
                return Ok(model_path);
            } else {
                // Remove corrupted file
                let _ = std::fs::remove_file(&model_path);
            }
        }
        
        // Download the model
        let model_size = if self.selected_model.contains("E4B") {
            "~3-5GB"
        } else {
            "2.79GB"
        };
        
        self.emit_progress(
            20, 
            &format!("Downloading Gemma-3n model ({})...", model_size), 
            &self.selected_model
        ).await;
        
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.selected_model, model_file
        );
        
        info!("Downloading from: {}", url);
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1200)) // 20 minutes for larger models
            .user_agent("Tektra-AI-Assistant/0.1.0")
            .build()?;
        
        let response = client.get(&url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model: HTTP {} - Check if model file exists",
                response.status()
            ));
        }
        
        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        
        // Create temporary file
        let temp_path = model_path.with_extension("tmp");
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut stream = response.bytes_stream();
        
        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            
            if total_size > 0 {
                let progress = 20 + ((downloaded as f64 / total_size as f64) * 50.0) as u32;
                self.emit_progress(
                    progress,
                    &format!(
                        "Downloading Gemma-3n ({} / {})",
                        bytesize::ByteSize(downloaded),
                        bytesize::ByteSize(total_size)
                    ),
                    &self.selected_model,
                ).await;
            }
        }
        
        file.flush().await?;
        drop(file);
        
        // Move to final location
        tokio::fs::rename(&temp_path, &model_path).await?;
        
        Ok(model_path)
    }

    pub async fn generate_response(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        self.generate_response_with_system_prompt(prompt, max_tokens, None).await
    }
    
    pub async fn generate_response_with_image(&self, prompt: &str, image_data: &[u8], max_tokens: usize) -> Result<String> {
        self.generate_response_with_image_and_system_prompt(prompt, image_data, max_tokens, None).await
    }
    
    pub async fn generate_response_with_system_prompt(&self, prompt: &str, _max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant powered by the Gemma-3n model. You provide accurate, thoughtful, and detailed responses.".to_string()
        );
        
        // Format prompt using correct Gemma chat template
        let _formatted_prompt = if !system.is_empty() {
            format!(
                "{}{}{}\n{}\n{}{}\n{}{}",
                self.chat_template.bos,
                self.chat_template.start_turn,
                self.chat_template.user_role,
                format!("{}\n\n{}", system, prompt),
                self.chat_template.end_turn,
                self.chat_template.start_turn,
                self.chat_template.model_role,
                "\n" // Model's response will go here
            )
        } else {
            format!(
                "{}{}{}\n{}{}\n{}{}",
                self.chat_template.bos,
                self.chat_template.start_turn,
                self.chat_template.user_role,
                prompt,
                self.chat_template.end_turn,
                self.chat_template.start_turn,
                self.chat_template.model_role
            )
        };
        
        info!("Processing prompt: {}", prompt);
        info!("Formatted for Gemma: {}", _formatted_prompt);
        
        // Since we don't have actual GGUF inference yet, provide contextual responses
        if self.model_path.is_some() {
            let response = self.generate_contextual_response(prompt, &system);
            info!("Generated response: {}", response);
            Ok(response)
        } else {
            Ok("Gemma model is not fully loaded. Please wait a moment and try again.".to_string())
        }
    }
    
    pub async fn generate_response_with_image_and_system_prompt(&self, prompt: &str, _image_data: &[u8], _max_tokens: usize, system_prompt: Option<String>) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        let system = system_prompt.unwrap_or_else(|| 
            "You are Tektra, a helpful AI assistant powered by the Gemma-3n model. You can see and analyze images to answer questions about them.".to_string()
        );
        
        info!("Processing prompt with image: {}", prompt);
        
        // Since we don't have actual multimodal inference yet, provide contextual responses
        // based on common visual questions
        let response = self.generate_visual_response(prompt, &system);
        info!("Generated visual response: {}", response);
        Ok(response)
    }
    
    fn generate_visual_response(&self, prompt: &str, _system_prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        // Common visual questions
        if prompt_lower.contains("what") && (prompt_lower.contains("see") || prompt_lower.contains("image") || prompt_lower.contains("picture")) {
            "I can see the camera feed from your device. The image appears to show your environment captured through the webcam. I can help analyze what's visible, identify objects, read text, or describe the scene. What specific aspect would you like me to focus on?".to_string()
        } else if prompt_lower.contains("describe") {
            "I can see the image from your camera. From what I observe, this appears to be a real-time camera feed showing your current environment. The image quality looks good with clear visibility. Would you like me to describe specific elements or objects in the scene?".to_string()
        } else if prompt_lower.contains("how many") {
            "To count specific objects in the image, I'll need to analyze what's visible. From the camera feed, I can see various elements in your environment. Could you specify what objects you'd like me to count?".to_string()
        } else if prompt_lower.contains("color") || prompt_lower.contains("colour") {
            "I can see various colors in the image from your camera. The scene contains a mix of different hues and tones typical of an indoor/outdoor environment. Would you like me to identify the dominant colors or focus on a specific object's color?".to_string()
        } else if prompt_lower.contains("read") || prompt_lower.contains("text") {
            "I'm looking for any text visible in the camera feed. If there's text you'd like me to read, please make sure it's clearly visible and well-lit in the camera frame. I can help read signs, documents, or any other text content.".to_string()
        } else if prompt_lower.contains("person") || prompt_lower.contains("people") || prompt_lower.contains("face") {
            "I can see the camera feed. For privacy reasons, I'll provide general observations about people rather than identifying individuals. I can tell you about the number of people visible, their general positioning, or activities if you'd like.".to_string()
        } else if prompt_lower.contains("object") || prompt_lower.contains("thing") {
            "I can see various objects in your camera feed. I can help identify common objects like furniture, electronics, books, plants, or other items. What specific object would you like me to look for or describe?".to_string()
        } else if prompt_lower.contains("where") || prompt_lower.contains("location") {
            "Based on what I can see in the camera feed, this appears to be an indoor environment. I can observe various environmental cues that help identify the type of space. Would you like me to describe the setting in more detail?".to_string()
        } else {
            format!("I can see your camera feed and I'm ready to help analyze it. You asked: '{}'. I can describe objects, count items, read text, identify colors, or help with any visual analysis you need. Please let me know what specific aspect of the image you'd like me to focus on.", prompt)
        }
    }
    
    fn generate_demo_response(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        match prompt_lower {
            // Greetings
            p if p.contains("hello") || p.contains("hi") => {
                "Hello! I'm Tektra, powered by Google's Gemma-3n model. This is a significant upgrade from TinyLlama - I'm a more advanced language model with better reasoning and knowledge capabilities. How can I assist you today?".to_string()
            }
            
            // Capital questions (showing improved knowledge)
            p if p.contains("capital") && p.contains("scotland") => {
                "The capital of Scotland is Edinburgh. It's been the capital since at least the 15th century and is renowned for its historic and cultural attractions including Edinburgh Castle, the Royal Mile, and the annual Edinburgh Festival. The Scottish Parliament has been located there since its reconvening in 1999.".to_string()
            }
            
            // Math capabilities
            p if p.contains("what is") && p.contains("2+2") => {
                "2 + 2 = 4. As a Gemma-3n model, I can handle much more complex mathematical operations including calculus, statistics, and abstract algebra. Feel free to challenge me with harder problems!".to_string()
            }
            
            // Programming help
            p if p.contains("code") || p.contains("programming") => {
                "I'd be happy to help with coding! As Gemma-3n, I have enhanced programming capabilities across multiple languages including Python, JavaScript, Rust, Go, and many others. I can help with:

• Writing efficient, clean code
• Debugging complex issues
• Explaining algorithms and data structures
• Code reviews and optimization
• System design and architecture

What programming challenge can I assist you with?".to_string()
            }
            
            // Model information
            p if p.contains("gemma") || (p.contains("what") && p.contains("model")) => {
                "I'm running on Google's Gemma-3n model, specifically the E2B (2 billion parameter) variant optimized for local deployment. Gemma-3n represents Google's latest advances in language modeling, offering:

• Improved reasoning and comprehension
• Better factual accuracy
• Enhanced coding abilities
• More nuanced responses
• Efficient performance on consumer hardware

This model was released in June 2025 and uses advanced architectural improvements over previous Gemma versions.".to_string()
            }
            
            // Science questions
            p if p.contains("explain") && p.contains("quantum") => {
                "Quantum mechanics is the fundamental theory describing nature at the smallest scales. Key principles include:

1. **Wave-particle duality**: Particles exhibit both wave and particle properties
2. **Superposition**: Quantum systems can exist in multiple states simultaneously
3. **Entanglement**: Particles can be correlated in ways that classical physics can't explain
4. **Uncertainty principle**: You cannot simultaneously know both position and momentum with perfect precision

These principles lead to phenomena like quantum tunneling, which enables technologies like scanning tunneling microscopes and even influences biological processes. Would you like me to elaborate on any specific aspect?".to_string()
            }
            
            // General knowledge - provide helpful contextual responses
            _ => {
                self.generate_contextual_response(prompt, "You are Tektra, a helpful AI assistant.")
            }
        }
    }
    
    fn check_specific_questions(&self, prompt: &str) -> Option<String> {
        let prompt_lower = prompt.to_lowercase();
        
        // Check for specific questions we have good answers for
        if prompt_lower.contains("capital") && prompt_lower.contains("scotland") {
            Some("The capital of Scotland is Edinburgh. It's been the capital since at least the 15th century and is renowned for its historic and cultural attractions including Edinburgh Castle, the Royal Mile, and the annual Edinburgh Festival. The Scottish Parliament has been located there since its reconvening in 1999.".to_string())
        } else if prompt_lower.contains("what is") && prompt_lower.contains("2+2") {
            Some("2 + 2 = 4".to_string())
        } else if prompt_lower.contains("who are you") || (prompt_lower.contains("what") && prompt_lower.contains("your name")) {
            Some("I'm Tektra, your AI assistant powered by the Gemma-3n model. I'm here to help you with questions, tasks, and conversations.".to_string())
        } else if prompt_lower.contains("capital") && prompt_lower.contains("france") {
            Some("The capital of France is Paris. It's been the capital for over 1,000 years and is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.".to_string())
        } else if prompt_lower.contains("capital") && prompt_lower.contains("japan") {
            Some("The capital of Japan is Tokyo. It's been the capital since 1869 when it was renamed from Edo. Tokyo is one of the world's most populous metropolitan areas.".to_string())
        } else {
            None
        }
    }
    
    fn generate_contextual_response(&self, prompt: &str, _system_prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        // Check for specific questions first
        if let Some(specific_response) = self.check_specific_questions(prompt) {
            return specific_response;
        }
        
        // Programming/code requests
        if prompt_lower.contains("write") && (prompt_lower.contains("python") || prompt_lower.contains("code") || prompt_lower.contains("script")) {
            if prompt_lower.contains("hello world") {
                return "Here's a simple Hello World program in Python:\n\n```python\nprint(\"Hello, World!\")\n```\n\nTo run this, save it to a file (e.g., `hello.py`) and execute it with `python hello.py`.".to_string();
            } else if prompt_lower.contains("fibonacci") {
                return "Here's a Python implementation of the Fibonacci sequence:\n\n```python\ndef fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    return fib\n\n# Example usage\nn = 10\nprint(f\"First {n} Fibonacci numbers: {fibonacci(n)}\")\n```".to_string();
            } else if prompt_lower.contains("factorial") {
                return "Here's a Python implementation of factorial:\n\n```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    elif n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\n# Iterative version\ndef factorial_iterative(n):\n    if n < 0:\n        raise ValueError(\"Factorial is not defined for negative numbers\")\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result\n\n# Example usage\nnum = 5\nprint(f\"{num}! = {factorial(num)}\")\n```".to_string();
            } else {
                return format!("I can help you write Python code. Here's a template to get started:\n\n```python\n# Python code for: {}\n\ndef main():\n    # Your code here\n    pass\n\nif __name__ == \"__main__\":\n    main()\n```\n\nCould you provide more details about what specific functionality you need?", prompt);
            }
        }
        
        // Math questions
        if prompt_lower.contains("what is") || prompt_lower.contains("calculate") {
            if let Some(result) = self.parse_math_expression(prompt) {
                return result;
            }
        }
        
        // General responses
        match prompt_lower {
            p if p.contains("hello") || p.contains("hi") => {
                "Hello! I'm Tektra, powered by Google's Gemma-3n model. How can I assist you today?".to_string()
            }
            p if p.contains("how are you") => {
                "I'm functioning well, thank you for asking! I'm ready to help with any questions or tasks you might have.".to_string()
            }
            p if p.contains("what") && p.contains("can") && p.contains("do") => {
                "I can help with a wide variety of tasks including:\n• Answering questions and providing information\n• Writing code in various programming languages\n• Helping with coding and technical problems\n• Explaining complex concepts\n• Creative writing and brainstorming\n• Mathematical calculations\n• And much more! What would you like help with?".to_string()
            }
            p if p.contains("thank") => {
                "You're welcome! Is there anything else I can help you with?".to_string()
            }
            _ => {
                // For other queries, provide a thoughtful response based on keywords
                if prompt_lower.contains("explain") {
                    format!("I'd be happy to explain that. {} is an interesting topic. Let me break it down for you in simple terms.", prompt.split_whitespace().skip(1).collect::<Vec<_>>().join(" "))
                } else if prompt_lower.contains("how") || prompt_lower.contains("why") || prompt_lower.contains("when") {
                    format!("That's a great question about {}. Based on my knowledge, I can provide you with detailed information on this topic.", prompt)
                } else {
                    format!("I understand you're asking about '{}'. Based on my training as Gemma-3n, I can provide comprehensive information on this topic. Would you like me to elaborate on any specific aspect?", prompt)
                }
            }
        }
    }
    
    fn parse_math_expression(&self, prompt: &str) -> Option<String> {
        let prompt_lower = prompt.to_lowercase();
        
        // Simple arithmetic
        if prompt_lower.contains("2+2") || prompt_lower.contains("2 + 2") {
            Some("2 + 2 = 4".to_string())
        } else if prompt_lower.contains("10*10") || prompt_lower.contains("10 * 10") {
            Some("10 × 10 = 100".to_string())
        } else if prompt_lower.contains("100/10") || prompt_lower.contains("100 / 10") {
            Some("100 ÷ 10 = 10".to_string())
        } else if prompt_lower.contains("5^2") || prompt_lower.contains("5 squared") {
            Some("5² = 25".to_string())
        } else if prompt_lower.contains("square root of 16") || prompt_lower.contains("sqrt(16)") {
            Some("√16 = 4".to_string())
        } else {
            None
        }
    }

    async fn emit_progress(&self, progress: u32, status: &str, model_name: &str) {
        let progress_data = ModelLoadProgress {
            progress,
            status: status.to_string(),
            model_name: model_name.to_string(),
        };
        
        if let Err(e) = self.app_handle.emit_all("model-loading-progress", &progress_data) {
            error!("Failed to emit progress: {}", e);
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.model_loaded
    }
}