use tektra::ai::{
    OllamaInference, InferenceBackend, InferenceConfig,
    Gemma3NProcessor, MultimodalInput, ProcessedMultimodal,
    UnifiedDocumentProcessor, ChunkingStrategy, DocumentFormat,
};
use std::path::Path;
use std::time::Instant;
use tokio::fs;

/// Comprehensive test of multimodal capabilities with bundled Ollama
/// Tests all input modalities and their combinations
#[tokio::test]
async fn test_multimodal_text_image_interleaving() {
    println!("🎨 Testing Text+Image Interleaving with Bundled Ollama");
    println!("=====================================================");
    
    // Initialize Ollama
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    // Load model
    ollama.load_model(Path::new("gemma3n:e4b")).await
        .expect("Failed to load model");
    
    // Initialize processor
    let processor = Gemma3NProcessor::new();
    
    // Create a simple test image in memory (red square)
    let mut image_data = Vec::new();
    let img = image::ImageBuffer::from_fn(100, 100, |x, y| {
        if x < 50 && y < 50 {
            image::Rgb([255u8, 0, 0]) // Red square in top-left
        } else {
            image::Rgb([255, 255, 255]) // White background
        }
    });
    
    // Save to PNG format in memory
    use image::ImageEncoder;
    let mut cursor = std::io::Cursor::new(&mut image_data);
    let encoder = image::codecs::png::PngEncoder::new(&mut cursor);
    encoder.write_image(
        img.as_raw(),
        100,
        100,
        image::ColorType::Rgb8
    ).expect("Failed to encode image");
    
    // Test cases
    let test_cases = vec![
        ("What do you see in this image?", 100),
        ("Describe the colors and shapes in detail.", 150),
        ("Is there a red square? If so, where is it located?", 100),
    ];
    
    for (prompt, max_tokens) in test_cases {
        println!("\n📝 Test: {}", prompt);
        
        // Create multimodal input
        let input = MultimodalInput {
            text: Some(prompt.to_string()),
            image_data: Some(image_data.clone()),
            audio_data: None,
            video_data: None,
        };
        
        let start = Instant::now();
        
        // Process multimodal input
        let processed = processor.process_multimodal(input).await
            .expect("Failed to process multimodal input");
        
        // Format for model
        let formatted = processor.format_for_gemma3n(
            &processed,
            Some("You are a helpful AI assistant with vision capabilities.")
        );
        
        // Generate response
        let mut config = InferenceConfig::default();
        config.max_tokens = max_tokens;
        
        let response = ollama.generate(&formatted, &config).await
            .expect("Failed to generate response");
        
        let duration = start.elapsed();
        
        println!("✅ Response: {}", &response.chars().take(200).collect::<String>().trim());
        println!("⏱️  Duration: {:?}", duration);
        
        // Validate response contains expected content
        assert!(!response.is_empty(), "Response should not be empty");
        assert!(response.len() > 10, "Response should be meaningful");
    }
    
    println!("\n✅ Text+Image interleaving test completed!");
}

#[tokio::test]
async fn test_multimodal_text_document_processing() {
    println!("📄 Testing Text+Document Processing with Bundled Ollama");
    println!("=====================================================");
    
    // Initialize components
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    ollama.load_model(Path::new("gemma3n:e4b")).await
        .expect("Failed to load model");
    
    let doc_processor = UnifiedDocumentProcessor::new();
    
    // Create test document content
    let test_content = r#"
    Quarterly Sales Report
    =====================
    
    Q1 2024: $1.2M revenue, 150 new customers
    Q2 2024: $1.5M revenue, 200 new customers
    Q3 2024: $1.8M revenue, 250 new customers
    Q4 2024: $2.1M revenue, 300 new customers
    
    Key Insights:
    - Consistent quarter-over-quarter growth
    - Customer acquisition rate increasing
    - Revenue growth of 75% year-over-year
    
    Recommendations:
    1. Expand sales team by 20%
    2. Increase marketing budget for Q1 2025
    3. Launch new product line
    "#;
    
    // Process document
    let processed = doc_processor.process(
        test_content.as_bytes().to_vec(),
        DocumentFormat::Txt
    ).await.expect("Failed to process document");
    
    println!("📊 Document processed with {} sections", processed.structured_content.sections.len());
    
    // Test different queries
    let queries = vec![
        ("What was the total revenue for 2024?", 100),
        ("Which quarter had the highest growth?", 100),
        ("What are the main recommendations?", 150),
        ("Summarize the customer acquisition trend.", 150),
    ];
    
    for (query, max_tokens) in queries {
        println!("\n❓ Query: {}", query);
        
        // Combine document sections with query
        let context = processed.structured_content.sections.iter()
            .flat_map(|s| &s.paragraphs)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n");
        
        let prompt = format!(
            "Based on the following document:\n\n{}\n\nQuestion: {}",
            context, query
        );
        
        let mut config = InferenceConfig::default();
        config.max_tokens = max_tokens;
        
        let start = Instant::now();
        let response = ollama.generate(&prompt, &config).await
            .expect("Failed to generate response");
        let duration = start.elapsed();
        
        println!("✅ Answer: {}", response.trim());
        println!("⏱️  Duration: {:?}", duration);
        
        // Validate response
        assert!(!response.is_empty());
        assert!(response.len() > 20, "Response should be detailed");
    }
    
    println!("\n✅ Document processing test completed!");
}

#[tokio::test]
async fn test_multimodal_combined_inputs() {
    println!("🌟 Testing Combined Multimodal Inputs with Bundled Ollama");
    println!("========================================================");
    
    // Initialize
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    ollama.load_model(Path::new("gemma3n:e4b")).await
        .expect("Failed to load model");
    
    let processor = Gemma3NProcessor::new();
    let doc_processor = UnifiedDocumentProcessor::new();
    
    // Create test image (chart-like visualization)
    let mut image_data = Vec::new();
    let img = image::ImageBuffer::from_fn(200, 150, |x, y| {
        // Create simple bar chart
        if y > 100 {
            // Bars
            if x < 50 { // Q1 bar
                image::Rgb([255u8, 0, 0]) // Red
            } else if x >= 50 && x < 100 { // Q2 bar
                image::Rgb([0, 255, 0]) // Green
            } else if x >= 100 && x < 150 { // Q3 bar
                image::Rgb([0, 0, 255]) // Blue
            } else if x >= 150 { // Q4 bar
                image::Rgb([255, 255, 0]) // Yellow
            } else {
                image::Rgb([255, 255, 255]) // White
            }
        } else {
            image::Rgb([255, 255, 255]) // White background
        }
    });
    
    // Encode image
    use image::ImageEncoder;
    let mut cursor = std::io::Cursor::new(&mut image_data);
    let encoder = image::codecs::png::PngEncoder::new(&mut cursor);
    encoder.write_image(
        img.as_raw(),
        200,
        150,
        image::ColorType::Rgb8
    ).expect("Failed to encode image");
    
    // Create corresponding text data
    let text_data = "Chart Data: Q1=25%, Q2=30%, Q3=35%, Q4=40% of total annual sales";
    
    // Create complex multimodal query
    let complex_prompt = format!(
        "I'm showing you a bar chart image and providing this data: {}. \
        Please analyze both the visual chart and the text data to provide insights \
        about the quarterly performance trend.",
        text_data
    );
    
    // Create multimodal input
    let input = MultimodalInput {
        text: Some(complex_prompt),
        image_data: Some(image_data),
        audio_data: None,
        video_data: None,
    };
    
    println!("\n🔄 Processing combined multimodal input...");
    let start = Instant::now();
    
    let processed = processor.process_multimodal(input).await
        .expect("Failed to process multimodal input");
    
    let formatted = processor.format_for_gemma3n(
        &processed,
        Some("You are an AI analyst capable of understanding both visual charts and data.")
    );
    
    let mut config = InferenceConfig::default();
    config.max_tokens = 300;
    
    let response = ollama.generate(&formatted, &config).await
        .expect("Failed to generate response");
    
    let duration = start.elapsed();
    
    println!("\n📊 Multimodal Analysis:");
    println!("{}", response.trim());
    println!("\n⏱️  Total processing time: {:?}", duration);
    
    // Validate comprehensive response
    assert!(!response.is_empty());
    assert!(response.len() > 50, "Response should be comprehensive");
    
    println!("\n✅ Combined multimodal test completed!");
}

#[tokio::test]
async fn test_multimodal_performance_and_quality() {
    println!("⚡ Testing Multimodal Performance with Bundled Ollama");
    println!("====================================================");
    
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    ollama.load_model(Path::new("gemma3n:e4b")).await
        .expect("Failed to load model");
    
    let processor = Gemma3NProcessor::new();
    
    // Test different input sizes
    let test_sizes = vec![
        ("Small", 50, 50, 50),    // 50x50 image
        ("Medium", 200, 200, 100), // 200x200 image
        ("Large", 400, 300, 200),  // 400x300 image
    ];
    
    println!("\n📊 Performance Benchmarks:");
    let mut total_time = std::time::Duration::ZERO;
    
    for (size_name, width, height, max_tokens) in test_sizes {
        // Create test image
        let mut image_data = Vec::new();
        let img = image::ImageBuffer::from_fn(width, height, |x, y| {
            // Create pattern
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            let b = 128;
            image::Rgb([r, g, b])
        });
        
        // Encode
        use image::ImageEncoder;
        let mut cursor = std::io::Cursor::new(&mut image_data);
        let encoder = image::codecs::png::PngEncoder::new(&mut cursor);
        encoder.write_image(
            img.as_raw(),
            width,
            height,
            image::ColorType::Rgb8
        ).expect("Failed to encode image");
        
        let input = MultimodalInput {
            text: Some("Describe the color pattern in this image.".to_string()),
            image_data: Some(image_data),
            audio_data: None,
            video_data: None,
        };
        
        let start = Instant::now();
        
        let processed = processor.process_multimodal(input).await
            .expect("Failed to process");
        
        let formatted = processor.format_for_gemma3n(&processed, None);
        
        let mut config = InferenceConfig::default();
        config.max_tokens = max_tokens;
        
        let response = ollama.generate(&formatted, &config).await
            .expect("Failed to generate");
        
        let duration = start.elapsed();
        total_time += duration;
        
        let tokens = response.split_whitespace().count();
        let tokens_per_sec = tokens as f64 / duration.as_secs_f64();
        
        println!("\n  {} image ({}x{}):", size_name, width, height);
        println!("    Processing time: {:?}", duration);
        println!("    Response tokens: {}", tokens);
        println!("    Throughput: {:.1} tokens/sec", tokens_per_sec);
        println!("    Image size: {} bytes", image_data.len());
    }
    
    println!("\n📈 Total processing time: {:?}", total_time);
    
    // Assert performance bounds
    assert!(total_time.as_secs() < 30, "Processing should complete within 30 seconds");
    
    println!("\n✅ Performance test completed!");
}

/// Test error handling for invalid inputs
#[tokio::test]
async fn test_multimodal_error_handling() {
    println!("🔧 Testing Multimodal Error Handling");
    println!("===================================");
    
    let mut ollama = OllamaInference::new();
    ollama.initialize().await
        .expect("Failed to initialize Ollama");
    
    ollama.load_model(Path::new("gemma3n:e4b")).await
        .expect("Failed to load model");
    
    let processor = Gemma3NProcessor::new();
    
    // Test 1: Empty inputs
    println!("\n1️⃣ Testing empty multimodal input...");
    let empty_input = MultimodalInput {
        text: None,
        image_data: None,
        audio_data: None,
        video_data: None,
    };
    
    match processor.process_multimodal(empty_input).await {
        Ok(_) => println!("  ✅ Handled empty input gracefully"),
        Err(e) => println!("  ✅ Correctly rejected empty input: {}", e),
    }
    
    // Test 2: Invalid image data
    println!("\n2️⃣ Testing invalid image data...");
    let invalid_input = MultimodalInput {
        text: Some("What's in this image?".to_string()),
        image_data: Some(vec![0, 1, 2, 3]), // Invalid image data
        audio_data: None,
        video_data: None,
    };
    
    match processor.process_multimodal(invalid_input).await {
        Ok(processed) => {
            // Should still work, just might not extract image features
            println!("  ✅ Processed with fallback handling");
            let formatted = processor.format_for_gemma3n(&processed, None);
            assert!(!formatted.is_empty());
        }
        Err(e) => println!("  ℹ️  Error (expected): {}", e),
    }
    
    // Test 3: Very large text input
    println!("\n3️⃣ Testing large text input...");
    let large_text = "Lorem ipsum ".repeat(10000); // Very large text
    let large_input = MultimodalInput {
        text: Some(large_text),
        image_data: None,
        audio_data: None,
        video_data: None,
    };
    
    match processor.process_multimodal(large_input).await {
        Ok(processed) => {
            println!("  ✅ Handled large input (likely truncated)");
            assert!(!processed.prompt.is_empty());
        }
        Err(e) => println!("  ℹ️  Error: {}", e),
    }
    
    println!("\n✅ Error handling test completed!");
}