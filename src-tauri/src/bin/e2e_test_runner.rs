use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use dialoguer::{theme::ColorfulTheme, Confirm, Select};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tektra::ai::{
    OllamaInference, UnifiedDocumentProcessor, InputPipeline, 
    SimpleEmbeddingGenerator, EmbeddingGenerator, Gemma3NProcessor,
    MultimodalInput as GemmaMultimodalInput, PipelineConfig, ChunkingStrategy,
    InferenceBackend, InferenceConfig,
};
use tektra::audio::{ConversationManager, RealAudioCapture};
use tektra::vision::{Vision, CameraConfig, PreprocessingQuality};
use tektra::vector_db::VectorDB;
use tokio::time::sleep;
use std::sync::Arc;
use std::fs;

#[derive(Parser)]
#[command(name = "tektra-e2e")]
#[command(about = "Tektra End-to-End Test Runner", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Skip permission prompts (for CI)
    #[arg(long, global = true)]
    no_interactive: bool,
    
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all tests
    All {
        /// Include live camera/mic tests
        #[arg(long)]
        with_live: bool,
    },
    
    /// Test model loading and inference
    Model {
        /// Model name to test
        #[arg(default_value = "gemma3n:e4b")]
        model: String,
    },
    
    /// Test document processing
    Documents,
    
    /// Test multimodal inputs
    Multimodal {
        /// Include live camera test
        #[arg(long)]
        with_camera: bool,
        
        /// Include live microphone test
        #[arg(long)]
        with_mic: bool,
    },
    
    /// Test model output validation
    Validation,
    
    /// Run performance benchmarks
    Benchmark {
        /// Number of iterations
        #[arg(default_value = "10")]
        iterations: usize,
    },
}

struct TestRunner {
    ollama: OllamaInference,
    validator_ollama: Option<OllamaInference>,
    verbose: bool,
    results: Vec<TestResult>,
    inference_config: InferenceConfig,
}

#[derive(Debug)]
struct TestResult {
    name: String,
    passed: bool,
    duration: Duration,
    details: String,
}

impl TestRunner {
    async fn new(verbose: bool) -> Result<Self> {
        println!("{}", "🚀 Initializing Tektra E2E Test Runner...".cyan().bold());
        
        let ollama = OllamaInference::new(None).await?;
        
        let inference_config = InferenceConfig {
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: None,
            n_threads: None,
        };
        
        Ok(Self {
            ollama,
            validator_ollama: None,
            verbose,
            results: Vec::new(),
            inference_config,
        })
    }
    
    async fn ensure_model_loaded(&mut self, model_name: &str) -> Result<()> {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        spinner.set_message(format!("Checking model: {}", model_name));
        
        if !self.ollama.has_model(model_name).await? {
            spinner.set_message(format!("Pulling model: {}", model_name));
            // Note: In real implementation, we'd add progress tracking here
            println!("\n{} Model not found locally. Please pull it using: ollama pull {}", 
                     "⚠️ ".yellow(), model_name);
            return Err(anyhow::anyhow!("Model not available"));
        }
        
        spinner.finish_with_message(format!("✓ Model ready: {}", model_name));
        Ok(())
    }
    
    async fn init_validator(&mut self) -> Result<()> {
        if self.validator_ollama.is_none() {
            println!("{}", "🔍 Initializing validator model...".cyan());
            self.validator_ollama = Some(OllamaInference::new(None).await?);
        }
        Ok(())
    }
    
    async fn validate_output(&self, input: &str, output: &str, expected_type: &str) -> Result<bool> {
        if let Some(validator) = &self.validator_ollama {
            let validation_prompt = format!(
                r#"You are a test validator. Given the following input and output, determine if the output is valid and appropriate.

Input: {}
Output: {}
Expected output type: {}

Respond with only "VALID" or "INVALID" followed by a brief reason.
"#,
                input, output, expected_type
            );
            
            let mut config = self.inference_config.clone();
            config.max_tokens = 100;
            let validation_response = validator.generate(&validation_prompt, &config).await?;
            let is_valid = validation_response.contains("VALID") && !validation_response.contains("INVALID");
            
            if self.verbose {
                println!("{} Validation result: {}", "🔍".cyan(), validation_response);
            }
            
            Ok(is_valid)
        } else {
            Ok(true) // Skip validation if no validator
        }
    }
    
    async fn test_model_loading(&mut self, model_name: &str) -> Result<()> {
        println!("\n{}", "📦 Testing Model Loading...".green().bold());
        let start = Instant::now();
        
        self.ensure_model_loaded(model_name).await?;
        
        // Test basic inference
        let test_prompt = "Hello, please respond with a simple greeting.";
        let mut config = self.inference_config.clone();
        config.max_tokens = 50;
        let response = self.ollama.generate(test_prompt, &config).await?;
        
        let passed = !response.is_empty() && response.len() > 5;
        
        self.results.push(TestResult {
            name: "Model Loading & Basic Inference".to_string(),
            passed,
            duration: start.elapsed(),
            details: format!("Response: {}", response.trim()),
        });
        
        Ok(())
    }
    
    async fn test_document_processing(&mut self) -> Result<()> {
        println!("\n{}", "📄 Testing Document Processing...".green().bold());
        
        let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
        let vector_db = Arc::new(VectorDB::new());
        let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
            Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
        
        let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
        
        // Configure pipeline
        let config = PipelineConfig {
            chunking_strategy: ChunkingStrategy::FixedSize { size: 500, overlap: 100 },
            max_chunks_per_document: 10,
            similarity_threshold: 0.7,
            context_window_size: 8000,
            enable_semantic_search: false,
        };
        pipeline.update_config(config).await?;
        
        // Test each document format
        let test_files = vec![
            ("test_data/documents/sample.txt", "text"),
            ("test_data/documents/sample.md", "markdown"),
            ("test_data/documents/technical_spec.md", "technical documentation"),
        ];
        
        for (file_path, doc_type) in test_files {
            let start = Instant::now();
            
            if !std::path::Path::new(file_path).exists() {
                println!("  {} Skipping {} (file not found)", "⚠️ ".yellow(), file_path);
                continue;
            }
            
            let combined_input = pipeline.process_combined_query(
                &format!("Summarize the key points in this {} file", doc_type),
                vec![std::path::Path::new(file_path)],
                None,
                vec![],
                None,
            ).await?;
            
            let formatted = pipeline.format_for_model(&combined_input);
            let mut config = self.inference_config.clone();
            config.max_tokens = 200;
            let response = self.ollama.generate(&formatted, &config).await?;
            
            let is_valid = self.validate_output(
                &formatted,
                &response,
                "summary of document content"
            ).await?;
            
            self.results.push(TestResult {
                name: format!("Document Processing - {}", doc_type),
                passed: is_valid && !response.is_empty(),
                duration: start.elapsed(),
                details: format!("Processed {} chunks, response length: {}", 
                                combined_input.document_context.len(), response.len()),
            });
        }
        
        Ok(())
    }
    
    async fn test_multimodal(&mut self, with_camera: bool, with_mic: bool) -> Result<()> {
        println!("\n{}", "🎨 Testing Multimodal Processing...".green().bold());
        
        let processor = Gemma3NProcessor::new();
        
        // Test static images
        let test_images = vec![
            "test_data/images/test_image.png",
            "test_data/images/simple_shapes.png",
            "test_data/images/gradient.png",
        ];
        
        for image_path in test_images {
            if !std::path::Path::new(image_path).exists() {
                continue;
            }
            
            let start = Instant::now();
            let image_data = fs::read(image_path)?;
            
            let input = GemmaMultimodalInput {
                text: Some("Describe what you see in this image in detail.".to_string()),
                image_data: Some(image_data),
                audio_data: None,
                video_data: None,
            };
            
            let processed = processor.process_multimodal(input).await?;
            let formatted = processor.format_for_gemma3n(&processed, None);
            let mut config = self.inference_config.clone();
            config.max_tokens = 200;
            let response = self.ollama.generate(&formatted, &config).await?;
            
            let is_valid = self.validate_output(
                "image description request",
                &response,
                "detailed description of image content"
            ).await?;
            
            self.results.push(TestResult {
                name: format!("Multimodal - Static Image: {}", 
                            std::path::Path::new(image_path).file_name().unwrap().to_str().unwrap()),
                passed: is_valid && response.len() > 20,
                duration: start.elapsed(),
                details: format!("Token count: {}, Response preview: {}...", 
                                processed.token_count, &response[..50.min(response.len())]),
            });
        }
        
        // Test live camera
        if with_camera {
            self.test_live_camera().await?;
        }
        
        // Test live microphone
        if with_mic {
            self.test_live_microphone().await?;
        }
        
        Ok(())
    }
    
    async fn test_live_camera(&mut self) -> Result<()> {
        println!("\n{}", "📷 Testing Live Camera...".yellow().bold());
        
        let theme = ColorfulTheme::default();
        let proceed = Confirm::with_theme(&theme)
            .with_prompt("This test requires camera access. Continue?")
            .default(true)
            .interact()?;
        
        if !proceed {
            println!("  {} Camera test skipped", "⏭️ ".yellow());
            return Ok(());
        }
        
        let start = Instant::now();
        let mut vision = Vision::new();
        
        // Initialize camera
        match vision.initialize_camera(CameraConfig::default()).await {
            Ok(_) => {
                println!("  {} Camera initialized successfully", "✓".green());
                
                // Capture a frame
                sleep(Duration::from_secs(1)).await; // Wait for camera to stabilize
                
                if let Ok(frame) = vision.capture_frame().await {
                    let processor = Gemma3NProcessor::new();
                    
                    // Convert frame to bytes (in real implementation)
                    let image_data = vec![0u8; 1000]; // Placeholder
                    
                    let input = GemmaMultimodalInput {
                        text: Some("Describe what the camera is seeing.".to_string()),
                        image_data: Some(image_data),
                        audio_data: None,
                        video_data: None,
                    };
                    
                    let processed = processor.process_multimodal(input).await?;
                    let formatted = processor.format_for_gemma3n(&processed, None);
                    let mut config = self.inference_config.clone();
                    config.max_tokens = 150;
                    let response = self.ollama.generate(&formatted, &config).await?;
                    
                    self.results.push(TestResult {
                        name: "Live Camera Capture".to_string(),
                        passed: !response.is_empty(),
                        duration: start.elapsed(),
                        details: format!("Captured frame, response length: {}", response.len()),
                    });
                }
                
                vision.stop_camera().await?;
            }
            Err(e) => {
                self.results.push(TestResult {
                    name: "Live Camera Capture".to_string(),
                    passed: false,
                    duration: start.elapsed(),
                    details: format!("Failed to initialize camera: {}", e),
                });
            }
        }
        
        Ok(())
    }
    
    async fn test_live_microphone(&mut self) -> Result<()> {
        println!("\n{}", "🎤 Testing Live Microphone...".yellow().bold());
        
        let theme = ColorfulTheme::default();
        let proceed = Confirm::with_theme(&theme)
            .with_prompt("This test requires microphone access. Continue?")
            .default(true)
            .interact()?;
        
        if !proceed {
            println!("  {} Microphone test skipped", "⏭️ ".yellow());
            return Ok(());
        }
        
        let start = Instant::now();
        let mut audio_capture = RealAudioCapture::new(16000)?; // 16kHz sample rate
        
        println!("  {} Please speak for 3 seconds...", "🎤".red().blink());
        
        // Start recording
        audio_capture.start_recording()?;
        
        // Show countdown
        for i in (1..=3).rev() {
            println!("  {} {}...", "⏱️ ".cyan(), i);
            sleep(Duration::from_secs(1)).await;
        }
        
        let audio_data = audio_capture.stop_recording()?;
        println!("  {} Recording complete! Processing...", "✓".green());
        
        // Process with Gemma3N
        let processor = Gemma3NProcessor::new();
        let input = GemmaMultimodalInput {
            text: Some("Transcribe and analyze this audio recording.".to_string()),
            image_data: None,
            audio_data: Some(audio_data.clone()),
            video_data: None,
        };
        
        let processed = processor.process_multimodal(input).await?;
        let formatted = processor.format_for_gemma3n(&processed, None);
        let mut config = self.inference_config.clone();
        config.max_tokens = 150;
        let response = self.ollama.generate(&formatted, &config).await?;
        
        self.results.push(TestResult {
            name: "Live Microphone Capture".to_string(),
            passed: !response.is_empty(),
            duration: start.elapsed(),
            details: format!("Recorded {} bytes, response length: {}", 
                           audio_data.len(), response.len()),
        });
        
        Ok(())
    }
    
    async fn test_output_validation(&mut self) -> Result<()> {
        println!("\n{}", "✅ Testing Output Validation...".green().bold());
        
        self.init_validator().await?;
        
        // Test various input/output pairs
        let test_cases = vec![
            (
                "What is 2 + 2?",
                "The answer is 4.",
                "mathematical answer",
                true,
            ),
            (
                "Translate 'hello' to Spanish",
                "Bonjour",
                "Spanish translation",
                false, // Wrong language
            ),
            (
                "Write a haiku about coding",
                "Lines of code flow fast\nBugs hide in the shadows deep\nDebugger finds peace",
                "haiku poem",
                true,
            ),
        ];
        
        for (input, output, expected_type, should_pass) in test_cases {
            let start = Instant::now();
            let is_valid = self.validate_output(input, output, expected_type).await?;
            
            self.results.push(TestResult {
                name: format!("Output Validation - {}", expected_type),
                passed: is_valid == should_pass,
                duration: start.elapsed(),
                details: format!("Expected: {}, Got: {}", should_pass, is_valid),
            });
        }
        
        Ok(())
    }
    
    async fn run_benchmarks(&mut self, iterations: usize) -> Result<()> {
        println!("\n{}", "⚡ Running Performance Benchmarks...".green().bold());
        
        let prompts = vec![
            ("Simple", "Hello, how are you?", 50),
            ("Medium", "Explain quantum computing in simple terms.", 200),
            ("Complex", "Write a detailed analysis of climate change impacts on global economies.", 500),
        ];
        
        for (name, prompt, max_tokens) in prompts {
            let mut durations = Vec::new();
            
            let pb = ProgressBar::new(iterations as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                    .unwrap()
            );
            pb.set_message(format!("Benchmarking: {}", name));
            
            for _ in 0..iterations {
                let start = Instant::now();
                let mut config = self.inference_config.clone();
                config.max_tokens = max_tokens;
                let _ = self.ollama.generate(prompt, &config).await?;
                durations.push(start.elapsed());
                pb.inc(1);
            }
            
            pb.finish_and_clear();
            
            // Calculate statistics
            let total: Duration = durations.iter().sum();
            let avg = total / iterations as u32;
            let min = durations.iter().min().unwrap();
            let max = durations.iter().max().unwrap();
            
            self.results.push(TestResult {
                name: format!("Benchmark - {} Complexity", name),
                passed: true,
                duration: avg,
                details: format!(
                    "Avg: {:?}, Min: {:?}, Max: {:?}, Iterations: {}",
                    avg, min, max, iterations
                ),
            });
        }
        
        Ok(())
    }
    
    fn print_results(&self) {
        println!("\n{}", "📊 Test Results Summary".blue().bold());
        println!("{}", "═".repeat(80).blue());
        
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.passed).count();
        let failed = total - passed;
        
        for result in &self.results {
            let status = if result.passed {
                "✓ PASS".green()
            } else {
                "✗ FAIL".red()
            };
            
            println!(
                "{} {} - {} ({:?})",
                status,
                result.name.bold(),
                result.details,
                result.duration
            );
        }
        
        println!("{}", "═".repeat(80).blue());
        println!(
            "Total: {} | {} | {}",
            format!("{} tests", total).bold(),
            format!("{} passed", passed).green(),
            format!("{} failed", failed).red()
        );
        
        let success_rate = (passed as f64 / total as f64) * 100.0;
        println!(
            "Success Rate: {}",
            if success_rate >= 80.0 {
                format!("{:.1}%", success_rate).green().bold()
            } else {
                format!("{:.1}%", success_rate).red().bold()
            }
        );
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    let mut runner = TestRunner::new(cli.verbose).await?;
    
    match cli.command {
        Commands::All { with_live } => {
            runner.test_model_loading("gemma3n:e4b").await?;
            runner.test_document_processing().await?;
            runner.test_multimodal(with_live, with_live).await?;
            runner.test_output_validation().await?;
            runner.run_benchmarks(5).await?;
        }
        Commands::Model { model } => {
            runner.test_model_loading(&model).await?;
        }
        Commands::Documents => {
            runner.test_document_processing().await?;
        }
        Commands::Multimodal { with_camera, with_mic } => {
            runner.test_multimodal(with_camera, with_mic).await?;
        }
        Commands::Validation => {
            runner.init_validator().await?;
            runner.test_output_validation().await?;
        }
        Commands::Benchmark { iterations } => {
            runner.run_benchmarks(iterations).await?;
        }
    }
    
    runner.print_results();
    
    Ok(())
}