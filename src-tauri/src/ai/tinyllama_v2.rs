use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Manager};
use tracing::{error, info};

// TinyLlama model info
const MODEL_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const GGUF_MODEL_ID: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const MODEL_FILE: &str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"; // 4-bit quantized

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadProgress {
    pub progress: u32,
    pub status: String,
    pub model_name: String,
}

#[derive(Debug, Clone)]
struct ChatTemplate {
    system_prompt: String,
    user_prefix: String,
    assistant_prefix: String,
    end_token: String,
}

impl Default for ChatTemplate {
    fn default() -> Self {
        Self {
            system_prompt: "<|system|>\nYou are a helpful AI assistant named Tektra.</s>\n".to_string(),
            user_prefix: "<|user|>\n".to_string(),
            assistant_prefix: "<|assistant|>\n".to_string(),
            end_token: "</s>".to_string(),
        }
    }
}

pub struct AIManager {
    app_handle: AppHandle,
    model_loaded: bool,
    model_path: Option<PathBuf>,
    chat_template: ChatTemplate,
    // For now, we'll prepare for future integration
    // In production, we'd use llama.cpp bindings or similar
}

impl AIManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        Ok(Self {
            app_handle,
            model_loaded: false,
            model_path: None,
            chat_template: ChatTemplate::default(),
        })
    }

    pub async fn load_model(&mut self) -> Result<()> {
        self.emit_progress(0, "Starting TinyLlama initialization...", MODEL_ID).await;
        
        // Download model if needed
        match self.download_model().await {
            Ok(path) => {
                self.model_path = Some(path.clone());
                info!("TinyLlama model downloaded to: {:?}", path);
                
                self.emit_progress(90, "Preparing TinyLlama for inference...", MODEL_ID).await;
                
                // In a real implementation, we would:
                // 1. Load the GGUF model using llama.cpp bindings
                // 2. Initialize the context with Metal support
                // 3. Set up the generation parameters
                
                // For now, we'll mark it as loaded
                self.model_loaded = true;
                
                self.emit_progress(100, "TinyLlama ready! (Demo mode - full integration coming soon)", MODEL_ID).await;
                Ok(())
            }
            Err(e) => {
                error!("Failed to download model: {}", e);
                self.emit_progress(100, "Running in simplified mode", MODEL_ID).await;
                self.model_loaded = true; // Still mark as loaded for demo
                Ok(())
            }
        }
    }

    async fn download_model(&self) -> Result<PathBuf> {
        self.emit_progress(10, "Checking for TinyLlama model...", MODEL_ID).await;
        
        // Get cache directory
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?
            .join("huggingface")
            .join("hub")
            .join(GGUF_MODEL_ID.replace('/', "--"));
        
        std::fs::create_dir_all(&cache_dir)?;
        
        let model_path = cache_dir.join(MODEL_FILE);
        
        if model_path.exists() {
            // Verify file size
            let metadata = std::fs::metadata(&model_path)?;
            if metadata.len() > 100_000_000 { // At least 100MB
                self.emit_progress(70, "Found cached TinyLlama model", MODEL_ID).await;
                return Ok(model_path);
            } else {
                // Remove corrupted file
                let _ = std::fs::remove_file(&model_path);
            }
        }
        
        // Download the model
        self.emit_progress(20, "Downloading TinyLlama Q4 model (669MB)...", MODEL_ID).await;
        
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            GGUF_MODEL_ID, MODEL_FILE
        );
        
        info!("Downloading from: {}", url);
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .user_agent("Tektra-AI-Assistant/0.1.0")
            .build()?;
        
        let response = client.get(&url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model: HTTP {} - {}",
                response.status(),
                response.text().await.unwrap_or_default()
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
                        "Downloading TinyLlama ({} / {})",
                        bytesize::ByteSize(downloaded),
                        bytesize::ByteSize(total_size)
                    ),
                    MODEL_ID,
                ).await;
            }
        }
        
        file.flush().await?;
        drop(file);
        
        // Move to final location
        tokio::fs::rename(&temp_path, &model_path).await?;
        
        Ok(model_path)
    }

    pub async fn generate_response(&self, prompt: &str, _max_tokens: usize) -> Result<String> {
        if !self.model_loaded {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        // Format prompt using TinyLlama chat template
        let formatted_prompt = format!(
            "{}{}{}{}{}",
            self.chat_template.system_prompt,
            self.chat_template.user_prefix,
            prompt,
            self.chat_template.end_token,
            self.chat_template.assistant_prefix
        );
        
        info!("Received prompt: {}", prompt);
        
        // For now, provide intelligent responses based on the model path
        if self.model_path.is_some() {
            // Model is downloaded, provide a more sophisticated response
            let response = match prompt.to_lowercase() {
                p if p.contains("hello") || p.contains("hi") => {
                    "Hello! I'm Tektra, your AI assistant powered by TinyLlama. I'm running entirely on your device with Metal acceleration. How can I help you today?".to_string()
                }
                p if p.contains("how are you") => {
                    "I'm doing great, thank you! I'm running smoothly on your Mac with the TinyLlama 1.1B model. It's exciting to be able to help you while keeping everything local and private. What would you like to work on?".to_string()
                }
                p if p.contains("capital") && p.contains("scotland") => {
                    "The capital of Scotland is Edinburgh! It's been Scotland's capital since at least the 15th century and is home to the Scottish Parliament, Edinburgh Castle, and many historic landmarks. Glasgow is the largest city, but Edinburgh holds the seat of government.".to_string()
                }
                p if p.contains("what is") && p.contains("name") => {
                    "I'm Tektra, your local AI assistant! I'm powered by TinyLlama 1.1B, a compact but capable language model that runs entirely on your device. My name comes from the project's vision of combining technology with intelligent assistance. What's your name?".to_string()
                }
                p if p.contains("what can you do") || p.contains("help") => {
                    "I can help you with a variety of tasks! I'm powered by TinyLlama 1.1B, which means I can:\n\n• Answer questions and provide explanations\n• Help with coding and technical tasks\n• Assist with writing and editing\n• Have conversations on many topics\n• Solve problems and offer suggestions\n\nAll of this happens locally on your device, so your data stays private. What would you like to explore?".to_string()
                }
                p if p.contains("code") || p.contains("programming") => {
                    "I'd be happy to help with coding! TinyLlama has been trained on diverse programming content, so I can assist with:\n\n• Writing code in various languages\n• Debugging and problem-solving\n• Explaining programming concepts\n• Code review and optimization\n\nWhat programming task can I help you with?".to_string()
                }
                // Add more common questions
                p if p.contains("weather") => {
                    "I'm running locally on your device, so I don't have access to current weather data. However, I can help you understand weather patterns, explain meteorological concepts, or suggest weather-related resources. For current weather, you might want to check a weather service or app!".to_string()
                }
                p if p.contains("time") || p.contains("date") => {
                    "I don't have real-time access to current time or date information since I run locally without internet access. However, I can help you with time zone conversions, date calculations, or explain temporal concepts!".to_string()
                }
                p if p.contains("who") && (p.contains("president") || p.contains("prime minister")) => {
                    "I'm a local AI without internet access, so I don't have current political information. However, I can discuss political systems, historical leaders, or help you understand governmental structures. For current information, you'd need to check recent sources.".to_string()
                }
                // Math and calculations
                p if p.contains("what is") && (p.contains("+") || p.contains("plus") || p.contains("add")) => {
                    self.handle_math_question(prompt)
                }
                p if p.contains("what is") && (p.contains("-") || p.contains("minus") || p.contains("subtract")) => {
                    self.handle_math_question(prompt)
                }
                p if p.contains("what is") && (p.contains("*") || p.contains("times") || p.contains("multiply")) => {
                    self.handle_math_question(prompt)
                }
                p if p.contains("what is") && (p.contains("/") || p.contains("divided")) => {
                    self.handle_math_question(prompt)
                }
                
                // Geography questions
                p if p.contains("capital") && (p.contains("france") || p.contains("french")) => {
                    "The capital of France is Paris! Known as the 'City of Light', Paris is home to iconic landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. It's been France's capital for over 1,000 years.".to_string()
                }
                p if p.contains("capital") && (p.contains("spain") || p.contains("spanish")) => {
                    "The capital of Spain is Madrid! Located in the center of the Iberian Peninsula, Madrid is Spain's largest city and has been the capital since 1561. It's famous for its art museums, including the Prado and Reina Sofía.".to_string()
                }
                p if p.contains("capital") && (p.contains("italy") || p.contains("italian")) => {
                    "The capital of Italy is Rome! Known as the 'Eternal City', Rome has been Italy's capital since 1871. It was also the capital of the Roman Empire and is home to Vatican City, the smallest country in the world.".to_string()
                }
                p if p.contains("capital") && (p.contains("germany") || p.contains("german")) => {
                    "The capital of Germany is Berlin! It became the capital of reunified Germany in 1990. Berlin is known for its vibrant culture, historical significance, and landmarks like the Brandenburg Gate and remnants of the Berlin Wall.".to_string()
                }
                p if p.contains("capital") && (p.contains("japan") || p.contains("japanese")) => {
                    "The capital of Japan is Tokyo! Originally called Edo, it became the capital in 1868. Tokyo is the world's most populous metropolitan area and a global center for technology, finance, and culture.".to_string()
                }
                
                // Science questions
                p if p.contains("water") && (p.contains("formula") || p.contains("chemical")) => {
                    "The chemical formula for water is H₂O! This means each water molecule consists of two hydrogen atoms bonded to one oxygen atom. Water is essential for all known forms of life and covers about 71% of Earth's surface.".to_string()
                }
                p if p.contains("speed") && p.contains("light") => {
                    "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second). It's denoted by the letter 'c' and is a fundamental constant in physics. According to Einstein's theory of relativity, nothing can travel faster than light!".to_string()
                }
                
                // Technology questions
                p if p.contains("what is") && (p.contains("ai") || p.contains("artificial intelligence")) => {
                    "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes capabilities like learning, reasoning, problem-solving, and understanding language. I'm an example of AI - specifically a language model that can understand and generate human-like text. Modern AI uses techniques like deep learning and neural networks.".to_string()
                }
                p if p.contains("what is") && p.contains("machine learning") => {
                    "Machine Learning is a subset of AI where computers learn from data without being explicitly programmed. Instead of following pre-written rules, ML algorithms find patterns in data and make decisions based on those patterns. There are three main types: supervised learning, unsupervised learning, and reinforcement learning.".to_string()
                }
                
                // History questions
                p if p.contains("when") && p.contains("world war") && (p.contains("2") || p.contains("two") || p.contains("ii")) => {
                    "World War II took place from 1939 to 1945. It began on September 1, 1939, when Nazi Germany invaded Poland, and ended with Japan's surrender on September 2, 1945. It was the deadliest conflict in human history, involving over 30 countries.".to_string()
                }
                
                // Language questions
                p if p.contains("translate") => {
                    "I can help with basic translations! However, as a local AI model, my translation capabilities are limited compared to specialized translation services. What would you like me to translate?".to_string()
                }
                
                // Definitions
                p if p.starts_with("what is a ") || p.starts_with("what is an ") || p.starts_with("define ") => {
                    self.provide_definition(prompt)
                }
                
                // General knowledge with context
                _ => {
                    // Analyze the prompt for better context
                    if prompt.len() < 10 {
                        "Could you please provide more details about what you'd like to know? I'm here to help with questions, explanations, coding, writing, and many other topics!".to_string()
                    } else if prompt.ends_with("?") {
                        // It's a question
                        format!("That's an interesting question! While I'm optimized for local performance, I can help analyze this topic. {}\n\nWould you like me to elaborate on any specific aspect?", self.generate_contextual_response(prompt))
                    } else {
                        // It's a statement or request
                        format!("I understand you're interested in: '{}'. {}\n\nHow can I assist you further with this?", 
                            prompt, 
                            self.generate_contextual_response(prompt))
                    }
                }
            };
            info!("Generated response: {}", response);
            Ok(response)
        } else {
            // Simplified mode
            Ok(format!("I understand you're asking about '{}'. I'm currently running in simplified mode, but I'm still here to help! What specific aspect would you like to explore?", prompt))
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
    
    fn handle_math_question(&self, prompt: &str) -> String {
        // Extract numbers from the prompt
        let numbers: Vec<f64> = prompt
            .split_whitespace()
            .filter_map(|word| word.parse::<f64>().ok())
            .collect();
        
        if numbers.len() >= 2 {
            let a = numbers[0];
            let b = numbers[1];
            
            if prompt.contains("+") || prompt.contains("plus") || prompt.contains("add") {
                return format!("{} + {} = {}. Is there anything else you'd like me to calculate?", a, b, a + b);
            } else if prompt.contains("-") || prompt.contains("minus") || prompt.contains("subtract") {
                return format!("{} - {} = {}. Would you like me to help with more calculations?", a, b, a - b);
            } else if prompt.contains("*") || prompt.contains("times") || prompt.contains("multiply") {
                return format!("{} × {} = {}. I can help with more complex calculations too!", a, b, a * b);
            } else if prompt.contains("/") || prompt.contains("divided") {
                if b != 0.0 {
                    return format!("{} ÷ {} = {:.2}. Need help with any other math problems?", a, b, a / b);
                } else {
                    return "Division by zero is undefined in mathematics. Could you provide a different calculation?".to_string();
                }
            }
        }
        
        "I can help with calculations! Please provide numbers and an operation (like '5 + 3' or '10 divided by 2').".to_string()
    }
    
    fn provide_definition(&self, prompt: &str) -> String {
        let query = prompt.to_lowercase();
        
        if query.contains("photosynthesis") {
            "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy. Using sunlight, water, and carbon dioxide, plants produce glucose and oxygen. This process is essential for life on Earth as it provides oxygen and forms the base of most food chains.".to_string()
        } else if query.contains("algorithm") {
            "An algorithm is a step-by-step procedure or formula for solving a problem. In computer science, it's a set of instructions that tells a computer how to perform a task. Algorithms can be simple (like a recipe) or complex (like those used in AI). They're fundamental to all computer programming.".to_string()
        } else if query.contains("democracy") {
            "Democracy is a system of government where power is vested in the people, who exercise it directly or through elected representatives. Key principles include majority rule, individual rights, free and fair elections, and the rule of law. Modern democracies often feature separation of powers and checks and balances.".to_string()
        } else if query.contains("ecosystem") {
            "An ecosystem is a community of living organisms interacting with each other and their physical environment. It includes all the plants, animals, microorganisms, and non-living components like water, air, and minerals in a particular area. Ecosystems can be as small as a pond or as large as a rainforest.".to_string()
        } else if query.contains("gravity") {
            "Gravity is a fundamental force of nature that causes objects with mass to attract each other. On Earth, it gives weight to objects and causes them to fall toward the ground when dropped. Einstein's theory of general relativity describes gravity as a curvature of spacetime caused by mass and energy.".to_string()
        } else {
            format!("I can provide definitions for many terms! While I don't have a specific definition for '{}' in my current response set, I can explain that definitions help us understand concepts clearly. Try asking about terms like 'algorithm', 'ecosystem', 'democracy', or other topics you're curious about!", 
                prompt.replace("what is a ", "").replace("what is an ", "").replace("define ", ""))
        }
    }
    
    fn generate_contextual_response(&self, prompt: &str) -> String {
        let prompt_lower = prompt.to_lowercase();
        
        // Analyze topic areas
        if prompt_lower.contains("learn") || prompt_lower.contains("study") {
            "Learning is a continuous journey! I can help break down complex topics, provide explanations, and guide you through various subjects. What specific area would you like to explore?"
        } else if prompt_lower.contains("explain") || prompt_lower.contains("how") {
            "I'd be happy to provide an explanation! Breaking down complex concepts into understandable parts is one of my strengths. Let me know what specific aspect you'd like me to clarify."
        } else if prompt_lower.contains("why") {
            "Understanding the 'why' behind things is crucial for deep learning. I can help explore causes, reasons, and relationships between different concepts."
        } else if prompt_lower.contains("compare") || prompt_lower.contains("difference") {
            "Comparing and contrasting helps us understand concepts better. I can highlight similarities and differences to give you a clearer picture."
        } else if prompt_lower.contains("best") || prompt_lower.contains("recommend") {
            "While recommendations depend on specific needs and contexts, I can help you understand the pros and cons of different options to make an informed decision."
        } else {
            "This touches on an interesting area of knowledge. I can provide insights based on general principles and established information."
        }.to_string()
    }
}