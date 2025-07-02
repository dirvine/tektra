use tektra::ai::{
    UnifiedDocumentProcessor, DocumentFormat, ChunkingStrategy,
    InputPipeline, PipelineConfig, SimpleEmbeddingGenerator,
    EmbeddingGenerator, CombinedInput, QueryType,
};
use tektra::vector_db::VectorDB;
use std::sync::Arc;
use std::path::Path;

#[tokio::test]
async fn test_document_processing_pipeline() {
    // Initialize components
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Test document processing
    let test_doc_path = Path::new("test_data/documents/sample.txt");
    
    let combined_input = pipeline.process_combined_query(
        "What are the types of machine learning?",
        vec![test_doc_path],
        None,
        vec![],
        None,
    ).await.unwrap();
    
    // Verify results
    assert_eq!(combined_input.user_query, "What are the types of machine learning?");
    assert!(!combined_input.document_context.is_empty());
    assert!(matches!(combined_input.metadata.query_type, QueryType::Question));
    
    // Check that relevant content was extracted
    let context_text: String = combined_input.document_context
        .iter()
        .map(|ctx| ctx.content.clone())
        .collect::<Vec<_>>()
        .join(" ");
    
    assert!(context_text.contains("Supervised Learning"));
    assert!(context_text.contains("Unsupervised Learning"));
    assert!(context_text.contains("Reinforcement Learning"));
}

#[tokio::test]
async fn test_multiple_document_query() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Process multiple documents
    let doc_paths = vec![
        Path::new("test_data/documents/sample.txt"),
        Path::new("test_data/documents/technical_spec.md"),
    ];
    
    let combined_input = pipeline.process_combined_query(
        "What are the core components of the system?",
        doc_paths,
        None,
        vec![],
        None,
    ).await.unwrap();
    
    // Should have contexts from both documents
    assert!(combined_input.document_context.len() > 1);
    
    // Should prioritize content from technical_spec.md for this query
    let most_relevant = &combined_input.document_context[0];
    assert!(most_relevant.content.contains("Input Pipeline") || 
            most_relevant.content.contains("Model Backend") ||
            most_relevant.content.contains("Vector Database"));
}

#[tokio::test]
async fn test_chunking_strategies() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor.clone(), vector_db, embedding_gen);
    
    // Test different chunking strategies
    let strategies = vec![
        ChunkingStrategy::FixedSize { size: 500, overlap: 100 },
        ChunkingStrategy::Semantic,
        ChunkingStrategy::SlidingWindow { min_size: 200, max_size: 800 },
    ];
    
    for strategy in strategies {
        // Update pipeline config
        let config = PipelineConfig {
            chunking_strategy: strategy,
            max_chunks_per_document: 5,
            similarity_threshold: 0.7,
            context_window_size: 8000,
            enable_semantic_search: false, // Use keyword search for consistency
        };
        
        pipeline.update_config(config).await.unwrap();
        
        let combined_input = pipeline.process_combined_query(
            "machine learning",
            vec![Path::new("test_data/documents/sample.txt")],
            None,
            vec![],
            None,
        ).await.unwrap();
        
        assert!(!combined_input.document_context.is_empty());
    }
}

#[tokio::test]
async fn test_context_window_limiting() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Set a small context window
    let config = PipelineConfig {
        chunking_strategy: ChunkingStrategy::FixedSize { size: 100, overlap: 20 },
        max_chunks_per_document: 10,
        similarity_threshold: 0.1, // Low threshold to get many results
        context_window_size: 500, // Small window
        enable_semantic_search: false,
    };
    
    pipeline.update_config(config).await.unwrap();
    
    let combined_input = pipeline.process_combined_query(
        "learning",
        vec![Path::new("test_data/documents/sample.txt")],
        None,
        vec![],
        None,
    ).await.unwrap();
    
    // Verify context is limited by window size
    let total_chars: usize = combined_input.document_context
        .iter()
        .map(|ctx| ctx.content.len())
        .sum();
    
    // Rough check - should be under 2000 chars (500 tokens * ~4 chars/token)
    assert!(total_chars < 2000);
}

#[tokio::test]
async fn test_json_document_processing() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    let combined_input = pipeline.process_combined_query(
        "What is the primary model?",
        vec![Path::new("test_data/documents/data.json")],
        None,
        vec![],
        None,
    ).await.unwrap();
    
    assert!(!combined_input.document_context.is_empty());
    
    let context_text: String = combined_input.document_context
        .iter()
        .map(|ctx| ctx.content.clone())
        .collect::<Vec<_>>()
        .join(" ");
    
    assert!(context_text.contains("gemma3n:e4b"));
}

#[tokio::test]
async fn test_additional_context_injection() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    let additional_context = vec![
        "The user prefers detailed technical explanations.".to_string(),
        "Previous conversation mentioned neural networks.".to_string(),
    ];
    
    let combined_input = pipeline.process_combined_query(
        "Explain deep learning",
        vec![Path::new("test_data/documents/sample.txt")],
        Some(additional_context),
        vec![],
        None,
    ).await.unwrap();
    
    // Should include both document context and additional context
    assert!(combined_input.document_context.len() >= 2);
    
    // Check that additional context is included
    let has_additional = combined_input.document_context
        .iter()
        .any(|ctx| ctx.content.contains("prefers detailed technical"));
    
    assert!(has_additional);
}

#[tokio::test]
async fn test_formatted_output_for_model() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    let combined_input = pipeline.process_combined_query(
        "Summarize the document",
        vec![Path::new("test_data/documents/sample.txt")],
        None,
        vec![],
        None,
    ).await.unwrap();
    
    let formatted = pipeline.format_for_model(&combined_input);
    
    // Check formatting
    assert!(formatted.contains("### Context Documents:"));
    assert!(formatted.contains("### Summarization Query:"));
    assert!(formatted.contains("Summarize the document"));
    assert!(formatted.contains("Source:"));
    assert!(formatted.contains("relevance:"));
}

#[tokio::test]
async fn test_query_type_detection() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    let test_cases = vec![
        ("What is machine learning?", QueryType::Question),
        ("Summarize this content", QueryType::Summary),
        ("Analyze the performance data", QueryType::Analysis),
        ("Translate to Spanish", QueryType::Translation),
        ("Write code to implement a binary search", QueryType::CodeGeneration),
    ];
    
    for (query, expected_type) in test_cases {
        let combined_input = pipeline.process_combined_query(
            query,
            vec![],
            None,
            vec![],
            None,
        ).await.unwrap();
        
        match (expected_type, &combined_input.metadata.query_type) {
            (QueryType::Question, QueryType::Question) => assert!(true),
            (QueryType::Summary, QueryType::Summary) => assert!(true),
            (QueryType::Analysis, QueryType::Analysis) => assert!(true),
            (QueryType::Translation, QueryType::Translation) => assert!(true),
            (QueryType::CodeGeneration, QueryType::CodeGeneration) => assert!(true),
            _ => panic!("Query type mismatch for: {}", query),
        }
    }
}

#[tokio::test]
async fn test_cache_functionality() {
    let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
    let vector_db = Arc::new(VectorDB::new());
    let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
        Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
    
    let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
    
    // Get initial stats
    let stats_before = pipeline.get_stats().await;
    assert_eq!(stats_before.cached_documents, 0);
    assert_eq!(stats_before.cached_embeddings, 0);
    
    // Process a document
    pipeline.process_combined_query(
        "test query",
        vec![Path::new("test_data/documents/sample.txt")],
        None,
        vec![],
        None,
    ).await.unwrap();
    
    // Check cache has items
    let stats_after = pipeline.get_stats().await;
    assert!(stats_after.cached_documents > 0);
    assert!(stats_after.cached_embeddings > 0);
    
    // Clear cache
    pipeline.clear_cache().await.unwrap();
    
    let stats_cleared = pipeline.get_stats().await;
    assert_eq!(stats_cleared.cached_documents, 0);
    assert_eq!(stats_cleared.cached_embeddings, 0);
}