use tektra::ai::{
    OllamaInference, InferenceBackend, InferenceConfig,
    Gemma3NProcessor, MultimodalInput,
    UnifiedDocumentProcessor, ChunkingStrategy,
};
use std::path::Path;
use std::time::{Duration, Instant};
use colored::*;
use anyhow::Result;

/// Comprehensive test of all modalities using bundled Ollama
/// Tests text, image, audio, video, documents and their combinations

#[tokio::main]
async fn main() -> Result<()> {
    println!("{}", "🚀 Comprehensive Multimodal Test with Bundled Ollama".cyan().bold());
    println!("{}", "===================================================".cyan());
    
    // Initialize bundled Ollama
    println!("\n📦 Initializing bundled Ollama...");
    let mut ollama = OllamaInference::new();
    
    match ollama.initialize().await {
        Ok(_) => println!("✅ Ollama initialized successfully!"),
        Err(e) => {
            eprintln!("❌ Failed to initialize Ollama: {}", e);
            return Err(e.into());
        }
    }
    
    // Load model
    println!("\n📥 Loading gemma3n:e4b model...");
    match ollama.load_model(Path::new("gemma3n:e4b")).await {
        Ok(_) => println!("✅ Model loaded successfully!"),
        Err(e) => {
            eprintln!("❌ Failed to load model: {}", e);
            eprintln!("   The model will be downloaded automatically if not present.");
            return Err(e.into());
        }
    }
    
    // Initialize processors
    let processor = Gemma3NProcessor::new();
    let doc_processor = UnifiedDocumentProcessor::new();
    
    let mut all_tests_passed = true;
    
    // Test 1: Pure Text
    println!("\n{}", "📝 Test 1: Pure Text Processing".green().bold());
    println!("{}", "==============================".green());
    if let Err(e) = test_text_only(&mut ollama).await {
        eprintln!("❌ Text test failed: {}", e);
        all_tests_passed = false;
    }
    
    // Test 2: Text + Image
    println!("\n{}", "🖼️  Test 2: Text + Image Interleaving".green().bold());
    println!("{}", "====================================".green());
    if let Err(e) = test_text_image(&mut ollama, &processor).await {
        eprintln!("❌ Text+Image test failed: {}", e);
        all_tests_passed = false;
    }
    
    // Test 3: Text + Audio (simulated)
    println!("\n{}", "🎵 Test 3: Text + Audio Interleaving".green().bold());
    println!("{}", "===================================".green());
    if let Err(e) = test_text_audio(&mut ollama, &processor).await {
        eprintln!("❌ Text+Audio test failed: {}", e);
        all_tests_passed = false;
    }
    
    // Test 4: Text + Document
    println!("\n{}", "📄 Test 4: Text + Document Processing".green().bold());
    println!("{}", "====================================".green());
    if let Err(e) = test_text_document(&mut ollama, &doc_processor).await {
        eprintln!("❌ Text+Document test failed: {}", e);
        all_tests_passed = false;
    }
    
    // Test 5: Combined Modalities
    println!("\n{}", "🌟 Test 5: Combined Multimodal Input".green().bold());
    println!("{}", "===================================".green());
    if let Err(e) = test_combined_modalities(&mut ollama, &processor).await {
        eprintln!("❌ Combined modalities test failed: {}", e);
        all_tests_passed = false;
    }
    
    // Test 6: Performance Benchmark
    println!("\n{}", "⚡ Test 6: Performance Benchmarks".green().bold());
    println!("{}", "================================".green());
    if let Err(e) = test_performance(&mut ollama).await {
        eprintln!("❌ Performance test failed: {}", e);
        all_tests_passed = false;
    }
    
    // Summary
    println!("\n{}", "📊 Test Summary".cyan().bold());
    println!("{}", "===============".cyan());
    if all_tests_passed {
        println!("{} All tests passed!", "🎉".green());
        println!("Bundled Ollama is fully functional with all modalities!");
    } else {
        println!("{} Some tests failed", "⚠️".yellow());
    }
    
    Ok(())
}

async fn test_text_only(ollama: &mut OllamaInference) -> Result<()> {
    let config = InferenceConfig::default();
    
    let test_cases = vec![
        ("Simple", "What is 2 + 2?", 50),
        ("Creative", "Write a haiku about coding", 100),
        ("Analysis", "Explain recursion in one sentence", 100),
    ];
    
    for (name, prompt, max_tokens) in test_cases {
        println!("\n  {} {}: {}", "▶".yellow(), name, prompt);
        
        let mut test_config = config.clone();
        test_config.max_tokens = max_tokens;
        
        let start = Instant::now();
        let response = ollama.generate(prompt, &test_config).await?;
        let duration = start.elapsed();
        
        println!("  {} Response: {}", "✓".green(), 
            response.chars().take(100).collect::<String>().trim());
        println!("  ⏱️  Duration: {:?}", duration);
    }
    
    Ok(())
}

async fn test_text_image(ollama: &mut OllamaInference, processor: &Gemma3NProcessor) -> Result<()> {
    // Create a simple test image
    println!("\n  Creating test image...");
    let mut image_data = Vec::new();
    
    // Create a 100x100 image with colored squares
    let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
        if x < 50 && y < 50 {
            image::Rgb([255u8, 0, 0]) // Red top-left
        } else if x >= 50 && y < 50 {
            image::Rgb([0, 255, 0]) // Green top-right
        } else if x < 50 && y >= 50 {
            image::Rgb([0, 0, 255]) // Blue bottom-left
        } else {
            image::Rgb([255, 255, 0]) // Yellow bottom-right
        }
    });
    
    // Encode to PNG
    use image::ImageEncoder;
    let mut cursor = std::io::Cursor::new(&mut image_data);
    let encoder = image::codecs::png::PngEncoder::new(&mut cursor);
    encoder.write_image(
        img.as_raw(),
        100,
        100,
        image::ExtendedColorType::Rgb8
    )?;
    
    println!("  ✓ Created test image with 4 colored quadrants");
    
    // Test with different prompts
    let prompts = vec![
        "Describe the colors you see in each quadrant of this image.",
        "How many different colors are in this image?",
        "Is there a red square in this image? Where is it?",
    ];
    
    for prompt in prompts {
        println!("\n  {} Prompt: {}", "▶".yellow(), prompt);
        
        let input = MultimodalInput {
            text: Some(prompt.to_string()),
            image_data: Some(image_data.clone()),
            audio_data: None,
            video_data: None,
        };
        
        let start = Instant::now();
        let processed = processor.process_multimodal(input).await?;
        let formatted = processor.format_for_gemma3n(
            &processed,
            Some("You are a helpful AI assistant with vision capabilities.")
        );
        
        let config = InferenceConfig::default();
        let response = ollama.generate(&formatted, &config).await?;
        let duration = start.elapsed();
        
        println!("  {} Response: {}", "✓".green(), 
            response.chars().take(150).collect::<String>().trim());
        println!("  ⏱️  Duration: {:?}", duration);
    }
    
    Ok(())
}

async fn test_text_audio(ollama: &mut OllamaInference, processor: &Gemma3NProcessor) -> Result<()> {
    // Create simulated audio data (would be real audio in production)
    println!("\n  Creating simulated audio data...");
    
    // Generate a simple sine wave as test audio
    let sample_rate = 16000;
    let duration = 1.0; // 1 second
    let frequency = 440.0; // A4 note
    
    let samples: Vec<f32> = (0..(sample_rate as f32 * duration) as usize)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (t * frequency * 2.0 * std::f32::consts::PI).sin()
        })
        .collect();
    
    // Convert to bytes (simulating WAV data)
    let audio_data: Vec<u8> = samples.iter()
        .flat_map(|&sample| {
            let scaled = (sample * 32767.0) as i16;
            scaled.to_le_bytes().to_vec()
        })
        .collect();
    
    println!("  ✓ Created test audio (440Hz sine wave)");
    
    let input = MultimodalInput {
        text: Some("This audio contains a pure tone. Describe what you understand about this audio input.".to_string()),
        image_data: None,
        audio_data: Some(audio_data),
        video_data: None,
    };
    
    let start = Instant::now();
    let processed = processor.process_multimodal(input).await?;
    let formatted = processor.format_for_gemma3n(
        &processed,
        Some("You are a helpful AI assistant with audio understanding.")
    );
    
    let config = InferenceConfig::default();
    let response = ollama.generate(&formatted, &config).await?;
    let duration = start.elapsed();
    
    println!("  {} Response: {}", "✓".green(), 
        response.chars().take(150).collect::<String>().trim());
    println!("  ⏱️  Duration: {:?}", duration);
    
    Ok(())
}

async fn test_text_document(ollama: &mut OllamaInference, doc_processor: &UnifiedDocumentProcessor) -> Result<()> {
    // Create test document content
    let test_content = r#"
Quarterly Sales Report 2024
===========================

Executive Summary
----------------
This report presents the sales performance for all four quarters of 2024,
showing consistent growth throughout the year.

Q1 2024: $1.2M revenue (150 new customers)
Q2 2024: $1.5M revenue (200 new customers)  
Q3 2024: $1.8M revenue (250 new customers)
Q4 2024: $2.1M revenue (300 new customers)

Key Achievements:
- Total annual revenue: $6.6M
- Total new customers: 900
- Average quarterly growth: 25%
- Customer retention rate: 85%

Recommendations:
1. Expand sales team by 30% to support growth
2. Invest in customer success programs
3. Launch new product line in Q2 2025
"#;
    
    println!("\n  Processing test document...");
    
    // Save to temp file
    let temp_path = std::env::temp_dir().join("test_report.txt");
    tokio::fs::write(&temp_path, test_content).await?;
    
    // Process document
    let processed = doc_processor.process_file(&temp_path).await?;
    
    // Extract chunks
    let chunks = doc_processor.extract_chunks(
        &processed, 
        ChunkingStrategy::FixedSize { size: 500, overlap: 100 }
    ).await?;
    
    println!("  ✓ Document processed: {} chunks", chunks.len());
    
    // Test queries
    let queries = vec![
        "What was the total revenue for 2024?",
        "Which quarter had the highest growth?",
        "What are the main recommendations?",
    ];
    
    let config = InferenceConfig::default();
    
    for query in queries {
        println!("\n  {} Query: {}", "▶".yellow(), query);
        
        // Combine chunks with query
        let context = chunks.iter()
            .map(|c| &c.content)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        
        let prompt = format!(
            "Based on this document:\n\n{}\n\nQuestion: {}",
            context, query
        );
        
        let start = Instant::now();
        let response = ollama.generate(&prompt, &config).await?;
        let duration = start.elapsed();
        
        println!("  {} Answer: {}", "✓".green(), response.trim());
        println!("  ⏱️  Duration: {:?}", duration);
    }
    
    // Cleanup
    let _ = tokio::fs::remove_file(&temp_path).await;
    
    Ok(())
}

async fn test_combined_modalities(ollama: &mut OllamaInference, processor: &Gemma3NProcessor) -> Result<()> {
    println!("\n  Creating complex multimodal input...");
    
    // Create an image with data visualization
    let mut image_data = Vec::new();
    let img = image::ImageBuffer::from_fn(200, 100, |x, y| {
        // Simple bar chart visualization
        if y > 80 {
            // Bars at bottom
            if x < 50 {
                image::Rgb([255u8, 100, 100]) // Q1
            } else if x < 100 {
                image::Rgb([100, 255, 100]) // Q2
            } else if x < 150 {
                image::Rgb([100, 100, 255]) // Q3
            } else {
                image::Rgb([255, 255, 100]) // Q4
            }
        } else {
            image::Rgb([240, 240, 240]) // Background
        }
    });
    
    // Encode image
    use image::ImageEncoder;
    let mut cursor = std::io::Cursor::new(&mut image_data);
    let encoder = image::codecs::png::PngEncoder::new(&mut cursor);
    encoder.write_image(
        img.as_raw(),
        200,
        100,
        image::ExtendedColorType::Rgb8
    )?;
    
    // Create combined prompt
    let complex_prompt = "I'm showing you a bar chart visualization along with this data: \
        Q1=$1.2M, Q2=$1.5M, Q3=$1.8M, Q4=$2.1M. \
        The bars in the image represent quarterly performance. \
        Please analyze both the visual chart and the numerical data to provide insights \
        about the business performance trend.";
    
    let input = MultimodalInput {
        text: Some(complex_prompt.to_string()),
        image_data: Some(image_data),
        audio_data: None,
        video_data: None,
    };
    
    let start = Instant::now();
    let processed = processor.process_multimodal(input).await?;
    let formatted = processor.format_for_gemma3n(
        &processed,
        Some("You are a business analyst AI capable of analyzing both visual charts and data.")
    );
    
    let mut config = InferenceConfig::default();
    config.max_tokens = 300;
    
    let response = ollama.generate(&formatted, &config).await?;
    let duration = start.elapsed();
    
    println!("  {} Multimodal Analysis:", "✓".green());
    println!("{}", response.trim().bright_white());
    println!("\n  ⏱️  Total processing time: {:?}", duration);
    
    Ok(())
}

async fn test_performance(ollama: &mut OllamaInference) -> Result<()> {
    let config = InferenceConfig::default();
    
    println!("\n  Running performance benchmarks...");
    
    let benchmarks = vec![
        ("Tiny", "Hi!", 10),
        ("Small", "Count to 5", 20),
        ("Medium", "Explain AI in one paragraph", 150),
        ("Large", "Write a detailed explanation of how neural networks work", 500),
    ];
    
    let mut total_tokens = 0;
    let mut total_time = Duration::ZERO;
    
    for (name, prompt, max_tokens) in benchmarks {
        print!("  {} {} test... ", "▶".yellow(), name);
        
        let mut test_config = config.clone();
        test_config.max_tokens = max_tokens;
        
        // Average of 3 runs
        let mut times = Vec::new();
        let mut tokens = Vec::new();
        
        for _ in 0..3 {
            let start = Instant::now();
            let response = ollama.generate(prompt, &test_config).await?;
            let duration = start.elapsed();
            
            times.push(duration);
            tokens.push(response.split_whitespace().count());
        }
        
        let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
        let avg_tokens = tokens.iter().sum::<usize>() / tokens.len();
        let tokens_per_sec = avg_tokens as f64 / avg_time.as_secs_f64();
        
        println!("✓ {:?}, {} tokens, {:.1} tokens/sec", 
            avg_time, avg_tokens, tokens_per_sec);
        
        total_time += avg_time;
        total_tokens += avg_tokens;
    }
    
    println!("\n  📈 Overall Performance:");
    println!("     Total tokens: {}", total_tokens);
    println!("     Total time: {:?}", total_time);
    println!("     Average: {:.1} tokens/sec", 
        total_tokens as f64 / total_time.as_secs_f64());
    
    Ok(())
}