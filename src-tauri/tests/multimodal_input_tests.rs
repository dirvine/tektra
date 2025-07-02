use tektra::ai::{
    UnifiedModelManager, ModelConfig, GenerationParams, DeviceConfig,
    UnifiedDocumentProcessor, InputPipeline, SimpleEmbeddingGenerator,
    EmbeddingGenerator, Gemma3NProcessor, MultimodalInput, ProcessedMultimodal,
};
use tektra::vector_db::VectorDB;
use std::sync::Arc;
use std::path::Path;
use tokio::fs;

#[tokio::test]
async fn test_multimodal_input_creation() {
    let text_input = "Describe what you see and hear";
    let image_data = vec![0u8; 1000]; // Mock image data
    let audio_data = vec![0u8; 500];  // Mock audio data
    
    let input = MultimodalInput {
        text: Some(text_input.to_string()),
        image_data: Some(image_data.clone()),
        audio_data: Some(audio_data.clone()),
        video_data: None,
    };
    
    assert!(input.text.is_some());
    assert!(input.image_data.is_some());
    assert_eq!(input.image_data.as_ref().unwrap().len(), 1000);
    assert!(input.audio_data.is_some());
    assert_eq!(input.audio_data.as_ref().unwrap().len(), 500);
    assert!(input.video_data.is_none());
}

#[tokio::test]
async fn test_document_plus_query_multimodal() {
    // Initialize pipeline
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Create multimodal input with document context
    let image_data = vec![0u8; 100];
    let audio_data = vec![0u8; 50];
    
    let combined_input = pipeline.process_combined_query(
        "Based on the document and the image, explain the concept",
        vec![Path::new("test_data/documents/sample.txt")],
        None,
        vec![image_data],
        Some(audio_data),
    ).await.unwrap();
    
    // Convert to model input
    let model_input = pipeline.to_model_input(&combined_input);
    
    // Verify all modalities are present
    assert!(model_input.text.is_some());
    assert_eq!(model_input.images.len(), 1);
    assert!(model_input.audio.is_some());
    
    // Check that text includes document context
    let text = model_input.text.unwrap();
    assert!(text.contains("### Context Documents:"));
    assert!(text.contains("Based on the document and the image"));
    assert!(text.contains("[1 image(s) attached]"));
    assert!(text.contains("[Audio attached]"));
}

#[tokio::test]
async fn test_gemma3n_multimodal_processing() {
    let processor = Gemma3NProcessor::new();
    
    // Create test multimodal input
    let input = MultimodalInput {
        text: Some("What objects do you see in this image?".to_string()),
        image_data: Some(vec![0u8; 10000]), // Mock image
        audio_data: Some(vec![0u8; 32000]), // Mock 1 second of 16kHz audio
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Verify processing
    assert!(processed.prompt.contains("What objects do you see"));
    assert_eq!(processed.images.len(), 1);
    assert!(processed.token_count > 0);
    
    // Check audio processing message
    assert!(processed.prompt.contains("[Audio:"));
    assert!(processed.prompt.contains("USM tokens]"));
}

#[tokio::test]
async fn test_image_format_optimization() {
    let processor = Gemma3NProcessor::new();
    
    // Test with different "image sizes" (mocked)
    let test_cases = vec![
        (800, 600),   // Landscape
        (600, 800),   // Portrait
        (1024, 1024), // Square
    ];
    
    for (width, height) in test_cases {
        // In real test, would create actual image
        let mock_image_data = vec![0u8; (width * height * 3) as usize];
        
        let input = MultimodalInput {
            text: Some("Analyze this image".to_string()),
            image_data: Some(mock_image_data),
            audio_data: None,
            video_data: None,
        };
        
        let processed = processor.process_multimodal(input).await.unwrap();
        
        assert_eq!(processed.images.len(), 1);
        assert!(!processed.images[0].is_empty());
    }
}

#[tokio::test]
async fn test_context_window_management() {
    let processor = Gemma3NProcessor::new();
    
    // Create input that would exceed context window
    let large_text = "x".repeat(100000); // Way over 32K tokens
    
    let input = MultimodalInput {
        text: Some(large_text),
        image_data: None,
        audio_data: None,
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Should truncate to fit context window
    assert!(processed.prompt.contains("[content truncated]"));
    assert!(processed.token_count <= 32000);
}

#[tokio::test]
async fn test_formatted_multimodal_prompt() {
    let processor = Gemma3NProcessor::new();
    
    let input = MultimodalInput {
        text: Some("Describe this scene".to_string()),
        image_data: Some(vec![0u8; 1000]),
        audio_data: None,
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    let formatted = processor.format_for_gemma3n(&processed, Some("You are a helpful assistant"));
    
    // Check Gemma3N formatting
    assert!(formatted.contains("<start_of_turn>system"));
    assert!(formatted.contains("<start_of_turn>user"));
    assert!(formatted.contains("<start_of_image>"));
    assert!(formatted.contains("<end_of_turn>"));
    assert!(formatted.contains("<start_of_turn>model"));
}

#[tokio::test]
async fn test_multimodal_with_rag_pipeline() {
    // This tests the full pipeline: documents + multimodal inputs
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Load test image data (mock)
    let image_data = vec![0u8; 5000];
    
    // Process with both document and image
    let combined_input = pipeline.process_combined_query(
        "Compare the concepts in the document with what you see in the image",
        vec![Path::new("test_data/documents/technical_spec.md")],
        None,
        vec![image_data],
        None,
    ).await.unwrap();
    
    // Verify combined processing
    assert!(!combined_input.document_context.is_empty());
    assert_eq!(combined_input.images.len(), 1);
    
    // Format for model
    let formatted = pipeline.format_for_model(&combined_input);
    assert!(formatted.contains("### Context Documents:"));
    assert!(formatted.contains("Compare the concepts"));
    assert!(formatted.contains("[1 image(s) attached]"));
}

#[tokio::test]
async fn test_audio_token_calculation() {
    let processor = Gemma3NProcessor::new();
    
    // Test various audio durations
    // 16kHz sampling rate, 16-bit (2 bytes per sample)
    let test_cases = vec![
        (16000 * 2, 6),    // 1 second = 6 tokens
        (8000 * 2, 3),     // 0.5 seconds = 3 tokens
        (32000 * 2, 12),   // 2 seconds = 12 tokens
        (100 * 2, 1),      // Very short = 1 token minimum
    ];
    
    for (audio_size, expected_tokens) in test_cases {
        let input = MultimodalInput {
            text: Some("Transcribe this audio".to_string()),
            image_data: None,
            audio_data: Some(vec![0u8; audio_size]),
            video_data: None,
        };
        
        let processed = processor.process_multimodal(input).await.unwrap();
        
        // Extract token count from prompt
        if let Some(token_match) = processed.prompt.find("USM tokens") {
            let token_str = &processed.prompt[token_match - 10..token_match];
            if let Some(num_str) = token_str.split_whitespace().last() {
                if let Ok(tokens) = num_str.parse::<usize>() {
                    assert_eq!(tokens, expected_tokens);
                }
            }
        }
    }
}

#[tokio::test]
async fn test_memory_estimation() {
    let processor = Gemma3NProcessor::new();
    
    let input = MultimodalInput {
        text: Some("Test".to_string()),
        image_data: Some(vec![0u8; 1_000_000]), // 1MB image
        audio_data: Some(vec![0u8; 500_000]),   // 500KB audio
        video_data: None,
    };
    
    let memory_mb = processor.estimate_memory_usage(&input);
    
    // Should estimate reasonable memory usage
    assert!(memory_mb > 0);
    assert!(memory_mb < 100); // Reasonable upper bound
}

#[tokio::test]
async fn test_optimization_suggestions() {
    let processor = Gemma3NProcessor::new();
    
    // Test with large context
    let large_processed = ProcessedMultimodal {
        prompt: "x".repeat(80000), // Large prompt
        images: vec!["img1".to_string(), "img2".to_string(), "img3".to_string()],
        audio_embeddings: None,
        token_count: 25000,
    };
    
    let suggestions = processor.get_optimization_suggestions(&large_processed);
    
    assert!(!suggestions.is_empty());
    assert!(suggestions.iter().any(|s| s.contains("context size")));
    assert!(suggestions.iter().any(|s| s.contains("Multiple images")));
    
    // Test with small context
    let small_processed = ProcessedMultimodal {
        prompt: "Short prompt".to_string(),
        images: vec![],
        audio_embeddings: None,
        token_count: 500,
    };
    
    let suggestions = processor.get_optimization_suggestions(&small_processed);
    assert!(suggestions.iter().any(|s| s.contains("More context")));
}

#[tokio::test]
async fn test_error_handling_invalid_audio() {
    let processor = Gemma3NProcessor::new();
    
    // Odd-length audio data (not 16-bit aligned)
    let input = MultimodalInput {
        text: Some("Process audio".to_string()),
        image_data: None,
        audio_data: Some(vec![0u8; 101]), // Odd number
        video_data: None,
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Should handle gracefully
    assert!(processed.prompt.contains("[Audio processing unavailable]"));
}

#[tokio::test]
async fn test_video_placeholder_handling() {
    let processor = Gemma3NProcessor::new();
    
    let input = MultimodalInput {
        text: Some("Analyze this video".to_string()),
        image_data: None,
        audio_data: None,
        video_data: Some(vec![0u8; 1000]),
    };
    
    let processed = processor.process_multimodal(input).await.unwrap();
    
    // Should indicate video not supported
    assert!(processed.prompt.contains("[Video data provided but not processed"));
    assert!(processed.prompt.contains("Ollama limitation"));
}