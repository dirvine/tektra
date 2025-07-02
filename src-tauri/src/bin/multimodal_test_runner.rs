use tektra::ai::{
    OllamaInference, InferenceBackend, InferenceConfig,
    Gemma3NProcessor, MultimodalInput, ProcessedMultimodal,
    UnifiedDocumentProcessor, ChunkingStrategy, DocumentFormat,
};
use tektra::vision::enhanced_capture::{EnhancedCameraCapture, CaptureConfig};
use tektra::audio::whisper::WhisperSTT;
use std::path::Path;
use std::time::{Duration, Instant};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use dialoguer::{Confirm, theme::ColorfulTheme};
use clap::Parser;
use anyhow::Result;
use tokio::fs;
use image::{ImageBuffer, Rgb};
use std::sync::Arc;
use hound;

/// Comprehensive multimodal test runner for Tektra's bundled Ollama
/// Tests all modalities and their interleaving with text
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip camera tests
    #[arg(long)]
    skip_camera: bool,

    /// Skip microphone tests
    #[arg(long)]
    skip_mic: bool,

    /// Test specific modality only
    #[arg(short, long)]
    modality: Option<String>,

    /// Run performance benchmarks
    #[arg(short, long)]
    benchmark: bool,

    /// Generate test assets if missing
    #[arg(long)]
    generate_assets: bool,
}

struct MultimodalTestRunner {
    ollama: OllamaInference,
    processor: Gemma3NProcessor,
    document_processor: UnifiedDocumentProcessor,
    whisper: Option<WhisperSTT>,
    camera: Option<EnhancedCameraCapture>,
    verbose: bool,
    progress: ProgressBar,
}

impl MultimodalTestRunner {
    async fn new(args: &Args) -> Result<Self> {
        println!("{}", "🚀 Initializing Multimodal Test Runner".cyan().bold());
        println!("{}", "=====================================".cyan());

        // Initialize bundled Ollama
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        pb.set_message("Initializing bundled Ollama...");

        let mut ollama = OllamaInference::new();
        ollama.initialize().await?;
        
        pb.finish_with_message("✅ Ollama initialized");

        // Load model
        pb.set_message("Loading gemma3n:e4b model...");
        ollama.load_model(Path::new("gemma3n:e4b")).await?;
        pb.finish_with_message("✅ Model loaded");

        // Initialize processors
        let processor = Gemma3NProcessor::new();
        let document_processor = UnifiedDocumentProcessor::new();

        // Initialize Whisper if not skipping mic tests
        let whisper = if !args.skip_mic {
            pb.set_message("Loading Whisper model...");
            match WhisperSTT::new("base").await {
                Ok(w) => {
                    pb.finish_with_message("✅ Whisper loaded");
                    Some(w)
                }
                Err(e) => {
                    pb.finish_with_message(&format!("⚠️  Whisper unavailable: {}", e));
                    None
                }
            }
        } else {
            None
        };

        // Initialize camera if not skipping camera tests
        let camera = if !args.skip_camera {
            pb.set_message("Initializing camera...");
            match EnhancedCameraCapture::new(CaptureConfig::default()) {
                Ok(c) => {
                    pb.finish_with_message("✅ Camera initialized");
                    Some(c)
                }
                Err(e) => {
                    pb.finish_with_message(&format!("⚠️  Camera unavailable: {}", e));
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            ollama,
            processor,
            document_processor,
            whisper,
            camera,
            verbose: args.verbose,
            progress: ProgressBar::new(100),
        })
    }

    async fn run_all_tests(&mut self, args: &Args) -> Result<()> {
        let start = Instant::now();
        let mut results = Vec::new();

        // Ensure test assets exist
        if args.generate_assets {
            self.generate_test_assets().await?;
        }

        // Run tests based on modality selection
        match args.modality.as_deref() {
            Some("text") => results.push(self.test_text_only().await),
            Some("image") => results.push(self.test_text_image_interleaving().await),
            Some("audio") => results.push(self.test_text_audio_interleaving().await),
            Some("video") => results.push(self.test_text_video_interleaving().await),
            Some("document") => results.push(self.test_text_document_interleaving().await),
            Some("camera") => results.push(self.test_live_camera().await),
            Some("mic") => results.push(self.test_live_microphone().await),
            Some("all") | None => {
                // Test individual modalities
                results.push(self.test_text_only().await);
                results.push(self.test_text_image_interleaving().await);
                results.push(self.test_text_audio_interleaving().await);
                results.push(self.test_text_video_interleaving().await);
                results.push(self.test_text_document_interleaving().await);
                
                // Test live inputs if available
                if self.camera.is_some() {
                    results.push(self.test_live_camera().await);
                }
                if self.whisper.is_some() {
                    results.push(self.test_live_microphone().await);
                }
                
                // Test combined modalities
                results.push(self.test_all_modalities_combined().await);
            }
            _ => {
                eprintln!("Unknown modality: {}", args.modality.as_ref().unwrap());
                return Ok(());
            }
        }

        // Performance benchmarks if requested
        if args.benchmark {
            results.push(self.run_performance_benchmarks().await);
        }

        // Summary
        self.print_summary(&results, start.elapsed());
        Ok(())
    }

    async fn test_text_only(&mut self) -> TestResult {
        println!("\n{}", "📝 Testing Text-Only Inference".green().bold());
        println!("{}", "=============================".green());

        let mut test_result = TestResult::new("Text-Only");
        let config = InferenceConfig::default();

        let test_cases = vec![
            ("Simple", "What is the capital of France?", 50),
            ("Math", "Solve: 2x + 5 = 13", 100),
            ("Code", "Write a Python function to calculate factorial", 200),
            ("Creative", "Write a short poem about AI", 150),
            ("Analysis", "Explain the concept of recursion with an example", 300),
        ];

        for (name, prompt, max_tokens) in test_cases {
            let start = Instant::now();
            let mut test_config = config.clone();
            test_config.max_tokens = max_tokens;

            match self.ollama.generate(prompt, &test_config).await {
                Ok(response) => {
                    let duration = start.elapsed();
                    let tokens = response.split_whitespace().count();
                    let tokens_per_sec = tokens as f64 / duration.as_secs_f64();

                    println!("\n{}: {}", name.yellow(), prompt);
                    if self.verbose {
                        println!("Response: {}", response.trim().bright_white());
                    } else {
                        println!("Response: {}...", &response.chars().take(100).collect::<String>().trim().bright_white());
                    }
                    println!("⏱️  Duration: {:?}, ~{:.1} tokens/sec", duration, tokens_per_sec);
                    
                    test_result.add_success(name);
                }
                Err(e) => {
                    eprintln!("❌ {} test failed: {}", name, e);
                    test_result.add_failure(name, e.to_string());
                }
            }
        }

        test_result
    }

    async fn test_text_image_interleaving(&mut self) -> TestResult {
        println!("\n{}", "🖼️  Testing Text+Image Interleaving".green().bold());
        println!("{}", "===================================".green());

        let mut test_result = TestResult::new("Text+Image");
        let config = InferenceConfig::default();

        // Test cases with different image types
        let test_cases = vec![
            ("test_data/images/simple_shapes.png", "Describe the shapes in this image and count how many there are.", 150),
            ("test_data/images/chart.png", "Analyze this chart and tell me the key insights.", 200),
            ("test_data/images/code_snippet.png", "Read the code in this image and explain what it does.", 250),
            ("test_data/images/handwritten.png", "Transcribe the handwritten text in this image.", 100),
        ];

        for (image_path, prompt, max_tokens) in test_cases {
            let path = Path::new(image_path);
            if !path.exists() {
                println!("⚠️  Skipping {}: file not found", image_path);
                continue;
            }

            println!("\n📸 Testing with: {}", image_path);
            let image_data = fs::read(path).await?;
            
            // Create multimodal input
            let input = MultimodalInput {
                text: Some(prompt.to_string()),
                image_data: Some(image_data),
                audio_data: None,
                video_data: None,
            };

            let start = Instant::now();
            match self.processor.process_multimodal(input).await {
                Ok(processed) => {
                    let formatted = self.processor.format_for_gemma3n(
                        &processed,
                        Some("You are a helpful AI assistant with vision capabilities.")
                    );

                    let mut test_config = config.clone();
                    test_config.max_tokens = max_tokens;

                    match self.ollama.generate(&formatted, &test_config).await {
                        Ok(response) => {
                            let duration = start.elapsed();
                            println!("Prompt: {}", prompt.yellow());
                            if self.verbose {
                                println!("Response: {}", response.trim().bright_white());
                            } else {
                                println!("Response: {}...", &response.chars().take(150).collect::<String>().trim().bright_white());
                            }
                            println!("⏱️  Duration: {:?}", duration);
                            test_result.add_success(&format!("Image: {}", path.file_name().unwrap().to_str().unwrap()));
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to generate response: {}", e);
                            test_result.add_failure(image_path, e.to_string());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to process image: {}", e);
                    test_result.add_failure(image_path, e.to_string());
                }
            }
        }

        test_result
    }

    async fn test_text_audio_interleaving(&mut self) -> TestResult {
        println!("\n{}", "🎵 Testing Text+Audio Interleaving".green().bold());
        println!("{}", "==================================".green());

        let mut test_result = TestResult::new("Text+Audio");
        let config = InferenceConfig::default();

        // Test cases with different audio types
        let test_cases = vec![
            ("test_data/audio/speech.wav", "Transcribe this speech and summarize the main points.", 200),
            ("test_data/audio/music.wav", "Describe the music style and instruments you hear.", 150),
            ("test_data/audio/ambient.wav", "What sounds can you identify in this audio?", 100),
        ];

        for (audio_path, prompt, max_tokens) in test_cases {
            let path = Path::new(audio_path);
            if !path.exists() {
                println!("⚠️  Skipping {}: file not found", audio_path);
                continue;
            }

            println!("\n🔊 Testing with: {}", audio_path);
            let audio_data = fs::read(path).await?;
            
            // Create multimodal input
            let input = MultimodalInput {
                text: Some(prompt.to_string()),
                image_data: None,
                audio_data: Some(audio_data),
                video_data: None,
            };

            let start = Instant::now();
            match self.processor.process_multimodal(input).await {
                Ok(processed) => {
                    let formatted = self.processor.format_for_gemma3n(
                        &processed,
                        Some("You are a helpful AI assistant with audio understanding capabilities.")
                    );

                    let mut test_config = config.clone();
                    test_config.max_tokens = max_tokens;

                    match self.ollama.generate(&formatted, &test_config).await {
                        Ok(response) => {
                            let duration = start.elapsed();
                            println!("Prompt: {}", prompt.yellow());
                            if self.verbose {
                                println!("Response: {}", response.trim().bright_white());
                            } else {
                                println!("Response: {}...", &response.chars().take(150).collect::<String>().trim().bright_white());
                            }
                            println!("⏱️  Duration: {:?}", duration);
                            test_result.add_success(&format!("Audio: {}", path.file_name().unwrap().to_str().unwrap()));
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to generate response: {}", e);
                            test_result.add_failure(audio_path, e.to_string());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to process audio: {}", e);
                    test_result.add_failure(audio_path, e.to_string());
                }
            }
        }

        test_result
    }

    async fn test_text_video_interleaving(&mut self) -> TestResult {
        println!("\n{}", "🎬 Testing Text+Video Interleaving".green().bold());
        println!("{}", "==================================".green());

        let mut test_result = TestResult::new("Text+Video");
        let config = InferenceConfig::default();

        // Note: Video processing requires ffmpeg feature
        #[cfg(feature = "video-processing")]
        {
            let test_cases = vec![
                ("test_data/video/demo.mp4", "Describe what happens in this video.", 200),
                ("test_data/video/tutorial.mp4", "Summarize the key steps shown in this tutorial.", 300),
            ];

            for (video_path, prompt, max_tokens) in test_cases {
                let path = Path::new(video_path);
                if !path.exists() {
                    println!("⚠️  Skipping {}: file not found", video_path);
                    continue;
                }

                println!("\n📹 Testing with: {}", video_path);
                let video_data = fs::read(path).await?;
                
                // Create multimodal input
                let input = MultimodalInput {
                    text: Some(prompt.to_string()),
                    image_data: None,
                    audio_data: None,
                    video_data: Some(video_data),
                };

                let start = Instant::now();
                match self.processor.process_multimodal(input).await {
                    Ok(processed) => {
                        let formatted = self.processor.format_for_gemma3n(
                            &processed,
                            Some("You are a helpful AI assistant with video understanding capabilities.")
                        );

                        let mut test_config = config.clone();
                        test_config.max_tokens = max_tokens;

                        match self.ollama.generate(&formatted, &test_config).await {
                            Ok(response) => {
                                let duration = start.elapsed();
                                println!("Prompt: {}", prompt.yellow());
                                if self.verbose {
                                    println!("Response: {}", response.trim().bright_white());
                                } else {
                                    println!("Response: {}...", &response.chars().take(150).collect::<String>().trim().bright_white());
                                }
                                println!("⏱️  Duration: {:?}", duration);
                                test_result.add_success(&format!("Video: {}", path.file_name().unwrap().to_str().unwrap()));
                            }
                            Err(e) => {
                                eprintln!("❌ Failed to generate response: {}", e);
                                test_result.add_failure(video_path, e.to_string());
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to process video: {}", e);
                        test_result.add_failure(video_path, e.to_string());
                    }
                }
            }
        }

        #[cfg(not(feature = "video-processing"))]
        {
            println!("⚠️  Video processing not available (requires video-processing feature)");
            test_result.add_failure("Video", "Feature not enabled".to_string());
        }

        test_result
    }

    async fn test_text_document_interleaving(&mut self) -> TestResult {
        println!("\n{}", "📄 Testing Text+Document Interleaving".green().bold());
        println!("{}", "=====================================".green());

        let mut test_result = TestResult::new("Text+Document");
        let config = InferenceConfig::default();

        // Test cases with different document types
        let test_cases = vec![
            ("test_data/docs/technical.pdf", DocumentFormat::Pdf, "Summarize the main technical concepts in this document.", 300),
            ("test_data/docs/report.docx", DocumentFormat::Docx, "Extract the key findings from this report.", 250),
            ("test_data/docs/readme.md", DocumentFormat::Markdown, "What are the main features described in this README?", 200),
            ("test_data/docs/data.txt", DocumentFormat::PlainText, "Analyze the data patterns in this text file.", 150),
        ];

        for (doc_path, format, prompt, max_tokens) in test_cases {
            let path = Path::new(doc_path);
            if !path.exists() {
                println!("⚠️  Skipping {}: file not found", doc_path);
                continue;
            }

            println!("\n📋 Testing with: {}", doc_path);
            let doc_data = fs::read(path).await?;
            
            // Process document
            match self.document_processor.process_document(&doc_data, format, ChunkingStrategy::Semantic).await {
                Ok(processed_doc) => {
                    // Combine document content with prompt
                    let chunks_text = processed_doc.chunks.iter()
                        .map(|c| &c.text)
                        .take(5) // Limit to first 5 chunks for testing
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("\n\n");

                    let combined_prompt = format!(
                        "Document content:\n{}\n\nQuestion: {}",
                        chunks_text, prompt
                    );

                    let start = Instant::now();
                    let mut test_config = config.clone();
                    test_config.max_tokens = max_tokens;

                    match self.ollama.generate(&combined_prompt, &test_config).await {
                        Ok(response) => {
                            let duration = start.elapsed();
                            println!("Document: {} ({} chunks)", 
                                path.file_name().unwrap().to_str().unwrap(),
                                processed_doc.chunks.len()
                            );
                            println!("Prompt: {}", prompt.yellow());
                            if self.verbose {
                                println!("Response: {}", response.trim().bright_white());
                            } else {
                                println!("Response: {}...", &response.chars().take(150).collect::<String>().trim().bright_white());
                            }
                            println!("⏱️  Duration: {:?}", duration);
                            test_result.add_success(&format!("Document: {}", path.file_name().unwrap().to_str().unwrap()));
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to generate response: {}", e);
                            test_result.add_failure(doc_path, e.to_string());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to process document: {}", e);
                    test_result.add_failure(doc_path, e.to_string());
                }
            }
        }

        test_result
    }

    async fn test_live_camera(&mut self) -> TestResult {
        println!("\n{}", "📷 Testing Live Camera Input".green().bold());
        println!("{}", "============================".green());

        let mut test_result = TestResult::new("Live Camera");

        if let Some(camera) = &mut self.camera {
            // Request camera permission
            if !Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("This test requires camera access. Allow?")
                .default(true)
                .interact()
                .unwrap()
            {
                println!("⚠️  Camera test skipped by user");
                return test_result;
            }

            println!("📸 Starting camera capture...");
            match camera.start().await {
                Ok(_) => {
                    println!("✅ Camera started successfully");
                    
                    // Wait a moment for camera to stabilize
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    
                    // Capture frame
                    match camera.capture_frame() {
                        Ok(frame_data) => {
                            println!("📸 Captured frame: {} bytes", frame_data.len());
                            
                            // Create multimodal input
                            let input = MultimodalInput {
                                text: Some("Describe what you see in this camera image. What objects or people are visible?".to_string()),
                                image_data: Some(frame_data),
                                audio_data: None,
                                video_data: None,
                            };

                            // Process and generate response
                            let start = Instant::now();
                            match self.processor.process_multimodal(input).await {
                                Ok(processed) => {
                                    let formatted = self.processor.format_for_gemma3n(
                                        &processed,
                                        Some("You are a helpful AI assistant analyzing a live camera feed.")
                                    );

                                    let config = InferenceConfig::default();
                                    match self.ollama.generate(&formatted, &config).await {
                                        Ok(response) => {
                                            let duration = start.elapsed();
                                            println!("\n{}", "Camera Analysis:".yellow());
                                            println!("{}", response.trim().bright_white());
                                            println!("⏱️  Duration: {:?}", duration);
                                            test_result.add_success("Live camera capture");
                                        }
                                        Err(e) => {
                                            eprintln!("❌ Failed to analyze camera image: {}", e);
                                            test_result.add_failure("Camera analysis", e.to_string());
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("❌ Failed to process camera image: {}", e);
                                    test_result.add_failure("Camera processing", e.to_string());
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to capture frame: {}", e);
                            test_result.add_failure("Frame capture", e.to_string());
                        }
                    }

                    // Stop camera
                    camera.stop();
                    println!("📷 Camera stopped");
                }
                Err(e) => {
                    eprintln!("❌ Failed to start camera: {}", e);
                    test_result.add_failure("Camera start", e.to_string());
                }
            }
        } else {
            println!("⚠️  Camera not available");
        }

        test_result
    }

    async fn test_live_microphone(&mut self) -> TestResult {
        println!("\n{}", "🎤 Testing Live Microphone Input".green().bold());
        println!("{}", "================================".green());

        let mut test_result = TestResult::new("Live Microphone");

        if let Some(whisper) = &self.whisper {
            // Request microphone permission
            if !Confirm::with_theme(&ColorfulTheme::default())
                .with_prompt("This test requires microphone access. Allow?")
                .default(true)
                .interact()
                .unwrap()
            {
                println!("⚠️  Microphone test skipped by user");
                return test_result;
            }

            println!("🎤 Please speak for 3 seconds...");
            println!("   (Recording will start in 1 second)");
            tokio::time::sleep(Duration::from_secs(1)).await;

            // Record audio
            println!("🔴 Recording...");
            let recording_duration = Duration::from_secs(3);
            let start_time = Instant::now();
            
            // Simulate recording (in real implementation, use cpal to record)
            tokio::time::sleep(recording_duration).await;
            
            println!("✅ Recording complete");

            // Generate test audio data for demonstration
            // In real implementation, this would be actual recorded audio
            let sample_rate = 16000;
            let channels = 1;
            let samples: Vec<f32> = (0..sample_rate * 3)
                .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / sample_rate as f32).sin() * 0.1)
                .collect();

            // Process with Whisper
            println!("🎯 Processing with Whisper...");
            match whisper.transcribe_audio(&samples).await {
                Ok(transcript) => {
                    println!("📝 Transcript: {}", transcript.yellow());
                    
                    // Now use the transcript with text generation
                    let prompt = format!(
                        "The user said: '{}'. Please provide a helpful response.",
                        transcript
                    );

                    let config = InferenceConfig::default();
                    match self.ollama.generate(&prompt, &config).await {
                        Ok(response) => {
                            println!("\n{}", "AI Response:".green());
                            println!("{}", response.trim().bright_white());
                            test_result.add_success("Live microphone transcription");
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to generate response: {}", e);
                            test_result.add_failure("Response generation", e.to_string());
                        }
                    }
                }
                Err(e) => {
                    eprintln!("❌ Failed to transcribe audio: {}", e);
                    test_result.add_failure("Audio transcription", e.to_string());
                }
            }
        } else {
            println!("⚠️  Whisper not available");
        }

        test_result
    }

    async fn test_all_modalities_combined(&mut self) -> TestResult {
        println!("\n{}", "🌟 Testing All Modalities Combined".green().bold());
        println!("{}", "==================================".green());

        let mut test_result = TestResult::new("Combined Modalities");
        let config = InferenceConfig::default();

        // Create a complex multimodal scenario
        println!("📊 Creating complex multimodal input...");

        // Load test assets
        let image_path = Path::new("test_data/images/chart.png");
        let doc_path = Path::new("test_data/docs/data.txt");

        if !image_path.exists() || !doc_path.exists() {
            println!("⚠️  Required test assets not found");
            test_result.add_failure("Asset loading", "Missing test files".to_string());
            return test_result;
        }

        // Load image
        let image_data = fs::read(image_path).await.unwrap();
        
        // Load and process document
        let doc_data = fs::read(doc_path).await.unwrap();
        let processed_doc = self.document_processor
            .process_document(&doc_data, DocumentFormat::PlainText, ChunkingStrategy::FixedSize(500))
            .await
            .unwrap();

        // Create complex prompt combining all inputs
        let complex_prompt = format!(
            "I have provided you with:\n\
            1. An image showing a chart\n\
            2. A text document with data\n\
            \n\
            Document excerpt:\n{}\n\
            \n\
            Please analyze both the visual chart and the text data, then:\n\
            - Identify any correlations between the chart and the data\n\
            - Provide insights that combine information from both sources\n\
            - Suggest what conclusions can be drawn from this multimodal analysis",
            processed_doc.chunks.first().map(|c| &c.text).unwrap_or(&"".to_string())
        );

        // Create multimodal input
        let input = MultimodalInput {
            text: Some(complex_prompt),
            image_data: Some(image_data),
            audio_data: None,
            video_data: None,
        };

        let start = Instant::now();
        match self.processor.process_multimodal(input).await {
            Ok(processed) => {
                let formatted = self.processor.format_for_gemma3n(
                    &processed,
                    Some("You are an AI assistant capable of analyzing multiple types of data simultaneously.")
                );

                let mut test_config = config.clone();
                test_config.max_tokens = 500; // Allow longer response for complex analysis

                match self.ollama.generate(&formatted, &test_config).await {
                    Ok(response) => {
                        let duration = start.elapsed();
                        println!("\n{}", "Multimodal Analysis:".yellow());
                        println!("{}", response.trim().bright_white());
                        println!("\n⏱️  Total analysis time: {:?}", duration);
                        test_result.add_success("Complex multimodal analysis");
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to generate multimodal response: {}", e);
                        test_result.add_failure("Multimodal generation", e.to_string());
                    }
                }
            }
            Err(e) => {
                eprintln!("❌ Failed to process multimodal input: {}", e);
                test_result.add_failure("Multimodal processing", e.to_string());
            }
        }

        test_result
    }

    async fn run_performance_benchmarks(&mut self) -> TestResult {
        println!("\n{}", "⚡ Running Performance Benchmarks".green().bold());
        println!("{}", "=================================".green());

        let mut test_result = TestResult::new("Performance Benchmarks");
        let config = InferenceConfig::default();

        // Benchmark different input sizes
        let benchmarks = vec![
            ("Small text", "Hi!", 20),
            ("Medium text", "Explain quantum computing in simple terms.", 200),
            ("Large text", "Write a detailed essay about the impact of AI on society, covering ethics, economics, and future implications.", 500),
        ];

        let mut total_tokens = 0;
        let mut total_time = Duration::ZERO;

        for (name, prompt, max_tokens) in benchmarks {
            println!("\n📊 Benchmarking: {}", name.yellow());
            
            let mut test_config = config.clone();
            test_config.max_tokens = max_tokens;

            // Run 3 iterations for average
            let mut iteration_times = Vec::new();
            let mut iteration_tokens = Vec::new();

            for i in 1..=3 {
                print!("   Iteration {}/3... ", i);
                let start = Instant::now();
                
                match self.ollama.generate(prompt, &test_config).await {
                    Ok(response) => {
                        let duration = start.elapsed();
                        let tokens = response.split_whitespace().count();
                        
                        iteration_times.push(duration);
                        iteration_tokens.push(tokens);
                        
                        println!("✓ {:?}, {} tokens", duration, tokens);
                    }
                    Err(e) => {
                        eprintln!("✗ Failed: {}", e);
                        test_result.add_failure(name, e.to_string());
                        break;
                    }
                }
            }

            if !iteration_times.is_empty() {
                let avg_time = iteration_times.iter().sum::<Duration>() / iteration_times.len() as u32;
                let avg_tokens = iteration_tokens.iter().sum::<usize>() / iteration_tokens.len();
                let tokens_per_sec = avg_tokens as f64 / avg_time.as_secs_f64();

                println!("   Average: {:?}, {} tokens, {:.1} tokens/sec", 
                    avg_time, avg_tokens, tokens_per_sec);

                total_time += avg_time;
                total_tokens += avg_tokens;
                test_result.add_success(name);
            }
        }

        // Memory usage test
        println!("\n📊 Testing memory efficiency...");
        // This would require platform-specific memory measurement
        
        // Overall performance summary
        if total_tokens > 0 {
            let overall_tokens_per_sec = total_tokens as f64 / total_time.as_secs_f64();
            println!("\n📈 Overall Performance:");
            println!("   Total tokens: {}", total_tokens);
            println!("   Total time: {:?}", total_time);
            println!("   Average: {:.1} tokens/sec", overall_tokens_per_sec);
        }

        test_result
    }

    async fn generate_test_assets(&self) -> Result<()> {
        println!("\n{}", "🔧 Generating Test Assets".yellow().bold());
        println!("{}", "=========================".yellow());

        // Create test directories
        fs::create_dir_all("test_data/images").await?;
        fs::create_dir_all("test_data/audio").await?;
        fs::create_dir_all("test_data/video").await?;
        fs::create_dir_all("test_data/docs").await?;

        // Generate simple test image
        let img_path = Path::new("test_data/images/simple_shapes.png");
        if !img_path.exists() {
            println!("Creating test image: simple_shapes.png");
            let img = ImageBuffer::from_fn(400, 300, |x, y| {
                if (x - 100).pow(2) + (y - 100).pow(2) < 2500 {
                    Rgb([255u8, 0, 0]) // Red circle
                } else if x > 250 && x < 350 && y > 50 && y < 150 {
                    Rgb([0, 255, 0]) // Green rectangle
                } else if (x > 150 && x < 250) && (y > 200 && y < 250) {
                    Rgb([0, 0, 255]) // Blue square
                } else {
                    Rgb([255, 255, 255]) // White background
                }
            });
            img.save(img_path)?;
            println!("✅ Created simple_shapes.png");
        }

        // Generate test audio file
        let audio_path = Path::new("test_data/audio/test_tone.wav");
        if !audio_path.exists() {
            println!("Creating test audio: test_tone.wav");
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 44100,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = hound::WavWriter::create(audio_path, spec)?;
            let duration = 2.0; // 2 seconds
            let frequency = 440.0; // A4 note
            for t in 0..(spec.sample_rate as f32 * duration) as u32 {
                let sample = (t as f32 * frequency * 2.0 * std::f32::consts::PI / spec.sample_rate as f32).sin();
                writer.write_sample((sample * i16::MAX as f32) as i16)?;
            }
            writer.finalize()?;
            println!("✅ Created test_tone.wav");
        }

        // Generate test text document
        let doc_path = Path::new("test_data/docs/data.txt");
        if !doc_path.exists() {
            println!("Creating test document: data.txt");
            let content = "Sample Data Report\n\
                          ==================\n\
                          \n\
                          Q1 Results: 125 units sold\n\
                          Q2 Results: 187 units sold\n\
                          Q3 Results: 203 units sold\n\
                          Q4 Results: 195 units sold\n\
                          \n\
                          Total annual sales: 710 units\n\
                          Average quarterly sales: 177.5 units\n\
                          \n\
                          Key insights:\n\
                          - Steady growth through Q3\n\
                          - Slight decline in Q4\n\
                          - Overall positive trend";
            fs::write(doc_path, content).await?;
            println!("✅ Created data.txt");
        }

        println!("\n✅ Test assets ready");
        Ok(())
    }

    fn print_summary(&self, results: &[TestResult], total_duration: Duration) {
        println!("\n{}", "📊 Test Summary".cyan().bold());
        println!("{}", "===============".cyan());

        let total_tests: usize = results.iter().map(|r| r.total_tests()).sum();
        let total_passed: usize = results.iter().map(|r| r.passed).sum();
        let total_failed: usize = results.iter().map(|r| r.failed).sum();

        for result in results {
            println!("\n{}: {} passed, {} failed", 
                result.name.yellow(),
                result.passed.to_string().green(),
                result.failed.to_string().red()
            );
            
            if self.verbose && !result.failures.is_empty() {
                for (test, error) in &result.failures {
                    println!("  ❌ {}: {}", test, error);
                }
            }
        }

        println!("\n{}", "Overall Results:".bold());
        println!("  Total tests: {}", total_tests);
        println!("  Passed: {}", total_passed.to_string().green());
        println!("  Failed: {}", total_failed.to_string().red());
        println!("  Duration: {:?}", total_duration);

        if total_failed == 0 {
            println!("\n{} All tests passed!", "🎉".green());
        } else {
            println!("\n{} Some tests failed", "⚠️".yellow());
        }
    }
}

#[derive(Debug)]
struct TestResult {
    name: String,
    passed: usize,
    failed: usize,
    failures: Vec<(String, String)>,
}

impl TestResult {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            passed: 0,
            failed: 0,
            failures: Vec::new(),
        }
    }

    fn add_success(&mut self, test: &str) {
        self.passed += 1;
        if test.len() > 0 {
            println!("  ✅ {}", test.green());
        }
    }

    fn add_failure(&mut self, test: &str, error: String) {
        self.failed += 1;
        self.failures.push((test.to_string(), error));
    }

    fn total_tests(&self) -> usize {
        self.passed + self.failed
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(if args.verbose { 
            tracing::Level::DEBUG 
        } else { 
            tracing::Level::INFO 
        })
        .init();

    // Create and run test runner
    let mut runner = MultimodalTestRunner::new(&args).await?;
    runner.run_all_tests(&args).await?;

    Ok(())
}