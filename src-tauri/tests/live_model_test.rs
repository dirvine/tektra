use tektra::ai::{
    OllamaInference, UnifiedDocumentProcessor, InputPipeline,
    SimpleEmbeddingGenerator, EmbeddingGenerator, Gemma3NProcessor,
    MultimodalInput, PipelineConfig, ChunkingStrategy, GenerationParams,
};
use tektra::vector_db::VectorDB;
use std::sync::Arc;
use std::path::Path;
use tokio::fs;

/// This test requires Ollama to be running with the gemma3n:e4b model installed
/// Run: ollama pull gemma3n:e4b
/// Then: ollama serve
#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_text_generation() {
    // Initialize Ollama
    let mut ollama = OllamaInference::new();
    let _result = match ollama.initialize().await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}. Make sure Ollama is running.", e);
            return;
        }
    };
    
    // Check if model is available
    if !ollama.is_model_available("gemma3n:e4b").await.unwrap_or(false) {
        eprintln!("Model gemma3n:e4b not found. Run: ollama pull gemma3n:e4b");
        return;
    }
    
    // Test simple text generation
    let input = MultimodalInput {
        text: Some("Write a haiku about artificial intelligence.".to_string()),
        image_data: None,
        audio_data: None,
        video_data: None,
    };
    
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: None,
        stop_sequences: vec![],
        stream: false,
    };
    
    let response = ollama.generate_multimodal(&input, &params, "gemma3n:e4b").await.unwrap();
    
    println!("Prompt: {:?}", input.text);
    println!("Response: {}", response);
    
    // Validate response
    assert!(!response.is_empty());
    assert!(response.len() > 20); // Should be more than a few words
}

#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_with_image() {
    let mut ollama = OllamaInference::new();
    let _result = match ollama.initialize().await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}", e);
            return;
        }
    };
    
    // Load test image
    let image_path = Path::new("test_data/images/simple_shapes.png");
    if !image_path.exists() {
        eprintln!("Test image not found: {:?}", image_path);
        return;
    }
    
    let image_data = fs::read(image_path).await.unwrap();
    
    // Process with Gemma3N processor
    let processor = Gemma3NProcessor::new();
    let input = MultimodalInput {
        text: Some("Describe the shapes and colors in this image.".to_string()),
        image_data: Some(image_data),
        audio_data: None,
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    let formatted = processor.format_for_gemma3n(&processed, Some("You are a helpful AI assistant."));
    
    // Generate response
    let gen_input = MultimodalInput {
        text: Some(formatted),
        image_data: input.image_data,
        audio_data: None,
        video_data: None,
    };
    
    let params = GenerationParams {
        max_tokens: 200,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: None,
        stop_sequences: vec![],
        stream: false,
    };
    
    let response = ollama.generate_multimodal(&gen_input, &params, "gemma3n:e4b").await.unwrap();
    
    println!("Image Analysis Response: {}", response);
    
    // Validate response mentions shapes or colors
    assert!(!response.is_empty());
    let lower = response.to_lowercase();
    assert!(
        lower.contains("shape") || lower.contains("color") || 
        lower.contains("circle") || lower.contains("square") ||
        lower.contains("red") || lower.contains("blue"),
        "Response should mention shapes or colors"
    );
}

#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_with_documents() {
    let mut ollama = OllamaInference::new();
    let _result = match ollama.initialize().await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}", e);
            return;
        }
    };
    
    // Set up document processing pipeline
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Configure pipeline
    let config = PipelineConfig {
        chunking_strategy: ChunkingStrategy::FixedSize { size: 500, overlap: 100 },
        max_chunks_per_document: 5,
        similarity_threshold: 0.1,
        context_window_size: 8000,
        enable_semantic_search: false,
    };
    pipeline.update_config(config).await.unwrap();
    
    // Process document
    let doc_path = Path::new("test_data/documents/sample.txt");
    if !doc_path.exists() {
        eprintln!("Test document not found: {:?}", doc_path);
        return;
    }
    
    let combined_input = pipeline.process_combined_query(
        "What are the main topics discussed in this document?",
        vec![doc_path],
        None,
        vec![],
        None,
    ).await.unwrap();
    
    let formatted = pipeline.format_for_model(&combined_input);
    
    // Generate response
    let gen_input = MultimodalInput {
        text: Some(formatted),
        image_data: None,
        audio_data: None,
        video_data: None,
    };
    
    let params = GenerationParams {
        max_tokens: 300,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: None,
        stop_sequences: vec![],
        stream: false,
    };
    
    let response = ollama.generate_multimodal(&gen_input, &params, "gemma3n:e4b").await.unwrap();
    
    println!("Document Analysis Response: {}", response);
    
    // Validate response
    assert!(!response.is_empty());
    assert!(response.len() > 50); // Should be a substantial response
}

#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_multimodal_combined() {
    let mut ollama = OllamaInference::new();
    let _result = match ollama.initialize().await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}", e);
            return;
        }
    };
    
    // Load both image and audio
    let image_path = Path::new("test_data/images/gradient.png");
    let audio_path = Path::new("test_data/audio/test_speech.wav");
    
    if !image_path.exists() || !audio_path.exists() {
        eprintln!("Test files not found");
        return;
    }
    
    let image_data = fs::read(image_path).await.unwrap();
    let audio_data = fs::read(audio_path).await.unwrap();
    
    // Process multimodal input
    let processor = Gemma3NProcessor::new();
    let input = MultimodalInput {
        text: Some("Analyze both the visual gradient pattern and the audio content.".to_string()),
        image_data: Some(image_data),
        audio_data: Some(audio_data),
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    let formatted = processor.format_for_gemma3n(&processed, None);
    
    // Generate response
    let gen_input = MultimodalInput {
        text: Some(formatted),
        image_data: input.image_data.clone(),
        audio_data: input.audio_data.clone(),
        video_data: None,
    };
    
    let params = GenerationParams {
        max_tokens: 250,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        repeat_penalty: 1.1,
        seed: None,
        stop_sequences: vec![],
        stream: false,
    };
    
    let response = ollama.generate_multimodal(&gen_input, &params, "gemma3n:e4b").await.unwrap();
    
    println!("Multimodal Analysis Response: {}", response);
    
    // Validate response
    assert!(!response.is_empty());
    assert!(processed.images.len() == 1);
    assert!(processed.prompt.contains("[Audio:"));
}

#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_validation() {
    // Initialize two Ollama instances for validation
    let generator = match OllamaInference::new(None).await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}", e);
            return;
        }
    };
    
    let validator = match OllamaInference::new(None).await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to create validator instance: {}", e);
            return;
        }
    };
    
    // Generate a response
    let prompt = "What is the capital of France?";
    let response = generator.generate(prompt, 50).await.unwrap();
    
    // Validate the response
    let validation_prompt = format!(
        "Is this a correct answer? Question: '{}' Answer: '{}'. Reply with only YES or NO.",
        prompt, response
    );
    
    let validation = validator.generate(&validation_prompt, 10).await.unwrap();
    
    println!("Question: {}", prompt);
    println!("Answer: {}", response);
    println!("Validation: {}", validation);
    
    // Check validation
    assert!(validation.to_uppercase().contains("YES"));
}

/// Run all live tests with: cargo test --test live_model_test -- --ignored --nocapture
#[tokio::test]
#[ignore = "requires live Ollama instance"]
async fn test_live_model_performance() {
    use std::time::Instant;
    
    let mut ollama = OllamaInference::new();
    let _result = match ollama.initialize().await {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to connect to Ollama: {}", e);
            return;
        }
    };
    
    let prompts = vec![
        ("Simple", "Hello!", 20),
        ("Medium", "Explain quantum computing.", 100),
        ("Complex", "Write a detailed analysis of climate change.", 300),
    ];
    
    println!("\n=== Performance Test Results ===");
    
    for (name, prompt, max_tokens) in prompts {
        let start = Instant::now();
        let gen_input = MultimodalInput {
            text: Some(prompt.to_string()),
            image_data: None,
            audio_data: None,
            video_data: None,
        };
        
        let params = GenerationParams {
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: None,
            stop_sequences: vec![],
            stream: false,
        };
        
        let response = ollama.generate_multimodal(&gen_input, &params, "gemma3n:e4b").await.unwrap();
        let duration = start.elapsed();
        
        let tokens_generated = response.split_whitespace().count();
        let tokens_per_second = tokens_generated as f64 / duration.as_secs_f64();
        
        println!("\n{} Prompt:", name);
        println!("  Duration: {:?}", duration);
        println!("  Tokens generated: ~{}", tokens_generated);
        println!("  Tokens/second: ~{:.1}", tokens_per_second);
        println!("  Response preview: {}...", &response[..50.min(response.len())]);
    }
}