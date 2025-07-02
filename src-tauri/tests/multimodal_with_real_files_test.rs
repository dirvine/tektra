use tektra::ai::{
    UnifiedDocumentProcessor, InputPipeline, SimpleEmbeddingGenerator,
    EmbeddingGenerator, Gemma3NProcessor, MultimodalInput,
};
use tektra::vector_db::VectorDB;
use std::sync::Arc;
use std::path::Path;
use tokio::fs;

#[tokio::test]
async fn test_real_image_processing() {
    let processor = Gemma3NProcessor::new();
    
    // Load real test image
    let image_path = Path::new("test_data/images/test_image.png");
    let image_data = fs::read(image_path).await.expect("Failed to read test image");
    
    println!("Loaded test image: {} bytes", image_data.len());
    
    let input = MultimodalInput {
        text: Some("Describe what you see in this image. What shapes and colors are present?".to_string()),
        image_data: Some(image_data),
        audio_data: None,
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Verify processing
    assert!(processed.prompt.contains("Describe what you see"));
    assert_eq!(processed.images.len(), 1);
    assert!(!processed.images[0].is_empty()); // Base64 encoded
    assert!(processed.token_count > 250); // Image adds ~256 tokens
    
    println!("Processed image tokens: {}", processed.token_count);
    println!("Base64 image length: {}", processed.images[0].len());
}

#[tokio::test]
async fn test_real_audio_processing() {
    let processor = Gemma3NProcessor::new();
    
    // Load real test audio
    let audio_path = Path::new("test_data/audio/test_speech.wav");
    let audio_data = fs::read(audio_path).await.expect("Failed to read test audio");
    
    println!("Loaded test audio: {} bytes", audio_data.len());
    
    let input = MultimodalInput {
        text: Some("Transcribe this audio recording.".to_string()),
        image_data: None,
        audio_data: Some(audio_data),
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Verify processing
    assert!(processed.prompt.contains("Transcribe this audio"));
    assert!(processed.prompt.contains("[Audio:")); // Should show audio token count
    assert!(processed.prompt.contains("USM tokens]"));
    
    // Extract token count from prompt
    if let Some(audio_marker) = processed.prompt.find("USM tokens]") {
        let audio_section = &processed.prompt[audio_marker.saturating_sub(20)..audio_marker];
        println!("Audio section: {}", audio_section);
    }
}

#[tokio::test]
async fn test_document_with_real_image_query() {
    // Initialize pipeline
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Load real image
    let image_path = Path::new("test_data/images/simple_shapes.png");
    let image_data = fs::read(image_path).await.expect("Failed to read shapes image");
    
    // Update pipeline config to use keyword search for tests
    let config = tektra::ai::PipelineConfig {
        chunking_strategy: tektra::ai::ChunkingStrategy::FixedSize { size: 500, overlap: 100 },
        max_chunks_per_document: 5,
        similarity_threshold: 0.1, // Lower threshold for keyword matching
        context_window_size: 8000,
        enable_semantic_search: false, // Use keyword search
    };
    pipeline.update_config(config).await.unwrap();
    
    // Combine document with image query
    let combined_input = pipeline.process_combined_query(
        "Based on the technical specification and the image, what visual elements match the system architecture?",
        vec![Path::new("test_data/documents/technical_spec.md")],
        None,
        vec![image_data],
        None,
    ).await.unwrap();
    
    // Verify combined input
    assert!(!combined_input.document_context.is_empty());
    assert_eq!(combined_input.images.len(), 1);
    
    // Format for model
    let formatted = pipeline.format_for_model(&combined_input);
    assert!(formatted.contains("### Context Documents:"));
    assert!(formatted.contains("visual elements match"));
    assert!(formatted.contains("[1 image(s) attached]"));
    
    println!("Combined query formatted length: {}", formatted.len());
}

#[tokio::test]
async fn test_multiple_images_processing() {
    let processor = Gemma3NProcessor::new();
    
    // Load multiple test images
    let images = vec![
        "test_data/images/simple_shapes.png",
        "test_data/images/gradient.png",
        "test_data/images/pattern.png",
    ];
    
    let mut image_data_vec = Vec::new();
    for path in &images {
        let data = fs::read(path).await.expect(&format!("Failed to read {}", path));
        image_data_vec.push(data);
        println!("Loaded {}: {} bytes", path, image_data_vec.last().unwrap().len());
    }
    
    // Process each image separately to test MobileNet-V5 optimization
    for (idx, image_data) in image_data_vec.into_iter().enumerate() {
        let input = MultimodalInput {
            text: Some(format!("Analyze image {} and describe its content.", idx + 1)),
            image_data: Some(image_data),
            audio_data: None,
            video_data: None,
        };
        
        let processed = processor.process_multimodal(input).await.unwrap();
        
        assert_eq!(processed.images.len(), 1);
        assert!(processed.token_count > 250);
        
        // Get optimization suggestions
        let suggestions = processor.get_optimization_suggestions(&processed);
        println!("Image {} optimization suggestions: {:?}", idx + 1, suggestions);
    }
}

#[tokio::test]
async fn test_audio_variants_processing() {
    let processor = Gemma3NProcessor::new();
    
    // Test different audio files
    let audio_files = vec![
        ("test_data/audio/command.wav", "command"),
        ("test_data/audio/question.wav", "question"),
    ];
    
    for (path, description) in audio_files {
        let audio_data = fs::read(path).await.expect(&format!("Failed to read {}", path));
        
        let input = MultimodalInput {
            text: Some(format!("Process this {} audio.", description)),
            image_data: None,
            audio_data: Some(audio_data.clone()),
            video_data: None,
        };
        
        let processed = processor.process_multimodal(input).await.unwrap();
        
        // Calculate expected tokens based on audio duration
        let sample_count = audio_data.len() / 2; // 16-bit samples
        let duration_ms = (sample_count as f32 / 16000.0) * 1000.0; // 16kHz
        let expected_tokens = ((duration_ms / 160.0).ceil() as usize).max(1);
        
        println!("{} audio: {} bytes, ~{:.1}ms, expected {} tokens", 
                 description, audio_data.len(), duration_ms, expected_tokens);
        
        assert!(processed.prompt.contains(&format!("{} audio", description)));
        assert!(processed.prompt.contains("[Audio:"));
    }
}

#[tokio::test]
async fn test_multimodal_combination() {
    let processor = Gemma3NProcessor::new();
    
    // Load both image and audio
    let image_data = fs::read("test_data/images/test_image.png")
        .await.expect("Failed to read image");
    let audio_data = fs::read("test_data/audio/command.wav")
        .await.expect("Failed to read audio");
    
    let input = MultimodalInput {
        text: Some("Describe what you see and what you hear.".to_string()),
        image_data: Some(image_data),
        audio_data: Some(audio_data),
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Should have both image and audio processing
    assert_eq!(processed.images.len(), 1);
    assert!(processed.prompt.contains("[Audio:"));
    assert!(processed.prompt.contains("USM tokens]"));
    assert!(processed.prompt.contains("Describe what you see and what you hear"));
    
    // Token count should include both modalities
    assert!(processed.token_count > 256); // At least image tokens
    
    // Format for Gemma3N
    let formatted = processor.format_for_gemma3n(&processed, Some("You are a multimodal AI assistant"));
    assert!(formatted.contains("<start_of_turn>system"));
    assert!(formatted.contains("<start_of_image>"));
    assert!(formatted.contains("<start_of_turn>model"));
    
    println!("Multimodal prompt length: {}", formatted.len());
}

#[tokio::test]
async fn test_document_rag_with_multimodal() {
    // Full pipeline test with real files
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    let gemma_processor = Gemma3NProcessor::new();
    
    // Update pipeline config to use keyword search for tests
    let config = tektra::ai::PipelineConfig {
        chunking_strategy: tektra::ai::ChunkingStrategy::FixedSize { size: 500, overlap: 100 },
        max_chunks_per_document: 5,
        similarity_threshold: 0.1, // Lower threshold for keyword matching
        context_window_size: 8000,
        enable_semantic_search: false, // Use keyword search
    };
    pipeline.update_config(config).await.unwrap();
    
    // Load real files
    let image_data = fs::read("test_data/images/gradient.png")
        .await.expect("Failed to read gradient image");
    let audio_data = fs::read("test_data/audio/question.wav")
        .await.expect("Failed to read question audio");
    
    // Process query with documents
    let combined_input = pipeline.process_combined_query(
        "Analyze the gradient patterns in the image and the question in the audio, relating them to the machine learning concepts in the document.",
        vec![Path::new("test_data/documents/sample.txt")],
        None,
        vec![image_data.clone()],
        Some(audio_data.clone()),
    ).await.unwrap();
    
    // Convert to model input
    let model_input = pipeline.to_model_input(&combined_input);
    
    // Process through Gemma3N processor
    let multimodal_input = MultimodalInput {
        text: model_input.text.clone(),
        image_data: Some(image_data),
        audio_data: Some(audio_data),
        video_data: None,
    };
    
    let processed = gemma_processor.process_multimodal(multimodal_input).await.unwrap();
    
    // Verify full integration
    assert!(processed.prompt.contains("### Context Documents:")); // From pipeline
    assert!(processed.prompt.contains("machine learning")); // From document
    assert_eq!(processed.images.len(), 1); // Image processed
    assert!(processed.prompt.contains("[Audio:")); // Audio processed
    
    println!("Full RAG + Multimodal token count: {}", processed.token_count);
}