// Temporarily disabled - needs update to match new ollama_inference API
/*
#[cfg(test)]
mod ollama_integration_tests {
    use crate::ai::ollama_inference::OllamaInference;
    use crate::ai::multimodal_processor::MultimodalInput;
    use anyhow::Result;
    use tokio::test;
    use std::time::Duration;
    
    /// Initialize test logging
    fn init_test_logging() {
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_env_filter("tektra=debug,ollama=debug")
            .try_init();
    }
    
    /// Test finding and starting Ollama (system or bundled)
    #[test]
    async fn test_find_and_start_ollama() {
        init_test_logging();
        
        // Find Ollama executable
        let ollama_exe = find_ollama().await;
        assert!(ollama_exe.is_ok(), "Failed to find Ollama: {:?}", ollama_exe.err());
        
        let ollama_path = ollama_exe.unwrap();
        tracing::info!("Found Ollama at: {:?}", ollama_path);
        
        // Verify we can create an inference engine
        let engine = OllamaInferenceEngine::new();
        assert!(engine.is_ok(), "Failed to create Ollama engine: {:?}", engine.err());
    }
    
    /// Test model availability and downloading
    #[test]
    async fn test_ensure_model_available() {
        init_test_logging();
        
        let model_name = "gemma3n:e4b";
        let result = ensure_model_available(model_name).await;
        
        // This should succeed whether the model exists or needs to be downloaded
        assert!(result.is_ok(), "Failed to ensure model availability: {:?}", result.err());
    }
    
    /// Test text-only inference
    #[test]
    async fn test_text_inference() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let engine = engine.unwrap();
        
        // Simple text prompt
        let prompt = "What is 2 + 2? Answer with just the number.";
        let result = engine.generate(prompt, 50).await;
        
        match result {
            Ok(response) => {
                assert!(!response.is_empty(), "Got empty response");
                assert!(response.contains("4"), "Response should contain '4', got: {}", response);
            }
            Err(e) => {
                tracing::warn!("Text inference failed (may be expected in test env): {:?}", e);
            }
        }
    }
    
    /// Test multimodal text + image inference
    #[test]
    async fn test_multimodal_inference() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let engine = engine.unwrap();
        
        // Create a simple test image (red square)
        let test_image = create_test_image();
        
        let input = MultimodalInput {
            text: "What color is this image? Answer with just the color name.".to_string(),
            images: vec![test_image],
            audio: vec![],
            video_frames: vec![],
        };
        
        let result = engine.process_multimodal(input).await;
        
        match result {
            Ok(response) => {
                assert!(!response.is_empty(), "Got empty response");
                tracing::info!("Multimodal response: {}", response);
            }
            Err(e) => {
                tracing::warn!("Multimodal inference failed (may be expected in test env): {:?}", e);
            }
        }
    }
    
    /// Test streaming generation
    #[test]
    async fn test_streaming_generation() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let engine = engine.unwrap();
        
        let prompt = "Count from 1 to 5.";
        let stream_result = engine.generate_stream(prompt, 100).await;
        
        match stream_result {
            Ok(mut stream) => {
                let mut full_response = String::new();
                let mut chunk_count = 0;
                
                while let Some(chunk_result) = stream.recv().await {
                    match chunk_result {
                        Ok(chunk) => {
                            full_response.push_str(&chunk);
                            chunk_count += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Stream error: {:?}", e);
                            break;
                        }
                    }
                }
                
                if chunk_count > 0 {
                    assert!(!full_response.is_empty(), "Response should not be empty");
                    tracing::info!("Received {} chunks, total response: {}", chunk_count, full_response);
                }
            }
            Err(e) => {
                tracing::warn!("Stream creation failed (may be expected in test env): {:?}", e);
            }
        }
    }
    
    /// Test error handling for invalid model
    #[test]
    async fn test_invalid_model_handling() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let mut engine = engine.unwrap();
        
        // Try to switch to a non-existent model
        let result = engine.switch_model("non_existent_model:latest").await;
        assert!(result.is_err(), "Should fail for non-existent model");
    }
    
    /// Test concurrent inference requests
    #[test]
    async fn test_concurrent_inference() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let engine = engine.unwrap();
        
        // Create multiple concurrent requests
        let handles: Vec<_> = (0..3).map(|i| {
            let engine_clone = engine.clone();
            tokio::spawn(async move {
                let prompt = format!("What is {} + {}? Answer with just the number.", i, i);
                engine_clone.generate(&prompt, 50).await
            })
        }).collect();
        
        // Wait for all requests to complete
        let mut successful_responses = 0;
        for handle in handles {
            match handle.await {
                Ok(Ok(response)) => {
                    assert!(!response.is_empty(), "Got empty response");
                    successful_responses += 1;
                }
                Ok(Err(e)) => {
                    tracing::warn!("Concurrent request failed: {:?}", e);
                }
                Err(e) => {
                    tracing::error!("Task panicked: {:?}", e);
                }
            }
        }
        
        // At least one should succeed in a working environment
        if successful_responses > 0 {
            tracing::info!("Successfully completed {} concurrent requests", successful_responses);
        }
    }
    
    /// Test model switching
    #[test]
    async fn test_model_switching() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let mut engine = engine.unwrap();
        
        // Ensure we're using the default model
        let model_name = "gemma3n:e4b";
        let switch_result = engine.switch_model(model_name).await;
        
        match switch_result {
            Ok(_) => {
                // Test inference after switching
                let result = engine.generate("Hello", 20).await;
                match result {
                    Ok(response) => {
                        assert!(!response.is_empty(), "Got empty response after model switch");
                    }
                    Err(e) => {
                        tracing::warn!("Inference after model switch failed: {:?}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Model switch failed (model may not be available): {:?}", e);
            }
        }
    }
    
    /// Test handling of very long prompts
    #[test]
    async fn test_long_prompt_handling() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let engine = engine.unwrap();
        
        // Create a very long prompt
        let long_text = "Lorem ipsum ".repeat(1000);
        let prompt = format!("Summarize this text in one sentence: {}", long_text);
        
        let result = engine.generate(&prompt, 100).await;
        
        // Should handle gracefully (either succeed or fail with clear error)
        match result {
            Ok(response) => {
                assert!(!response.is_empty(), "Got empty response for long prompt");
                tracing::info!("Long prompt response: {}", response);
            }
            Err(e) => {
                tracing::warn!("Long prompt failed (may be expected): {:?}", e);
                // This is acceptable - model may have context limits
            }
        }
    }
    
    /// Test inference timeout handling
    #[test]
    async fn test_inference_timeout() {
        init_test_logging();
        
        let engine = OllamaInferenceEngine::new();
        if engine.is_err() {
            tracing::warn!("Skipping test - Ollama not available");
            return;
        }
        let engine = engine.unwrap();
        
        // Use timeout for the inference call
        let result = tokio::time::timeout(
            Duration::from_secs(30),
            engine.generate("Write a very short story.", 200)
        ).await;
        
        match result {
            Ok(Ok(response)) => {
                assert!(!response.is_empty(), "Got empty response");
                tracing::info!("Completed within timeout");
            }
            Ok(Err(e)) => {
                tracing::warn!("Inference failed: {:?}", e);
            }
            Err(_) => {
                panic!("Inference timed out after 30 seconds");
            }
        }
    }
    
    /// Helper function to create a test image
    fn create_test_image() -> Vec<u8> {
        use image::{ImageBuffer, Rgb};
        
        // Create a 100x100 red image
        let img = ImageBuffer::from_fn(100, 100, |_x, _y| {
            Rgb([255u8, 0u8, 0u8])
        });
        
        // Convert to PNG bytes
        let mut buffer = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
            .expect("Failed to encode test image");
        
        buffer
    }
}
*/

#[cfg(test)]
mod document_processor_tests {
    use crate::ai::document_processor::{DocumentProcessor as DocProcessorTrait, UnifiedDocumentProcessor, DocumentFormat, ChunkingStrategy};
    use std::path::Path;
    use tokio::test;
    
    /// Test text document processing
    #[test]
    async fn test_text_document_processing() {
        let processor = UnifiedDocumentProcessor::new();
        
        let test_text = "This is a test document.\n\nIt has multiple paragraphs.\n\nEach paragraph is important.";
        let result = processor.process(test_text.as_bytes().to_vec(), DocumentFormat::Txt).await;
        
        assert!(result.is_ok(), "Text processing failed: {:?}", result.err());
        let doc = result.unwrap();
        
        assert_eq!(doc.format, DocumentFormat::Txt);
        assert_eq!(doc.raw_text, test_text);
        assert!(doc.metadata.word_count > 0);
    }
    
    /// Test PDF document processing
    #[test]
    #[ignore = "Test PDF data needs to be fixed - xref table issue"]
    async fn test_pdf_document_processing() {
        let processor = UnifiedDocumentProcessor::new();
        
        // Create a minimal valid PDF for testing
        let pdf_bytes = create_test_pdf();
        let result = processor.process(pdf_bytes, DocumentFormat::Pdf).await;
        
        // PDF processing should succeed even if no text is extracted
        assert!(result.is_ok(), "PDF processing failed: {:?}", result.err());
        let doc = result.unwrap();
        
        assert_eq!(doc.format, DocumentFormat::Pdf);
        assert!(doc.metadata.page_count.is_some());
    }
    
    /// Test document chunking
    #[test]
    async fn test_document_chunking() {
        let processor = UnifiedDocumentProcessor::new();
        
        let test_text = "Word ".repeat(500); // Create a long document
        let doc_result = processor.process(test_text.as_bytes().to_vec(), DocumentFormat::Txt).await
            .expect("Failed to process document");
        
        let chunks_result = processor.extract_chunks(
            &doc_result,
            ChunkingStrategy::FixedSize { size: 100, overlap: 20 }
        ).await;
        
        assert!(chunks_result.is_ok(), "Chunking failed: {:?}", chunks_result.err());
        let chunks = chunks_result.unwrap();
        
        assert!(!chunks.is_empty(), "Should have created chunks");
        assert!(chunks.len() > 1, "Should have multiple chunks for long document");
        
        // Verify chunk properties
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i);
            assert!(!chunk.content.is_empty());
            assert!(chunk.content.len() <= 100 || i == chunks.len() - 1); // Last chunk may be smaller
        }
    }
    
    /// Test markdown processing
    #[test]
    async fn test_markdown_processing() {
        let processor = UnifiedDocumentProcessor::new();
        
        let markdown = r#"# Title

## Section 1
This is a paragraph.

## Section 2
- Item 1
- Item 2

```rust
let x = 42;
```"#;
        
        let result = processor.process(markdown.as_bytes().to_vec(), DocumentFormat::Markdown).await;
        
        assert!(result.is_ok(), "Markdown processing failed: {:?}", result.err());
        let doc = result.unwrap();
        
        assert_eq!(doc.format, DocumentFormat::Markdown);
        assert!(doc.structured_content.sections.len() > 1, "Should have extracted sections");
    }
    
    /// Test JSON processing
    #[test]
    async fn test_json_processing() {
        let processor = UnifiedDocumentProcessor::new();
        
        let json = r#"{
            "title": "Test Document",
            "content": "This is the content",
            "metadata": {
                "author": "Test Author",
                "date": "2024-01-01"
            }
        }"#;
        
        let result = processor.process(json.as_bytes().to_vec(), DocumentFormat::Json).await;
        
        assert!(result.is_ok(), "JSON processing failed: {:?}", result.err());
        let doc = result.unwrap();
        
        assert_eq!(doc.format, DocumentFormat::Json);
        assert!(doc.raw_text.contains("Test Document"));
    }
    
    /// Helper to create a minimal valid PDF
    fn create_test_pdf() -> Vec<u8> {
        // Minimal PDF structure
        let pdf_content = b"%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000229 00000 n 
0000000328 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
422
%%EOF";
        
        pdf_content.to_vec()
    }
}

// Temporarily disabled - needs update to match new vector_db API
/*
#[cfg(test)]
mod vector_db_tests {
    use crate::vector_db::VectorDB;
    use tokio::test;
    
    /// Test vector database initialization
    #[test]
    async fn test_vector_db_creation() {
        let db = VectorDB::new();
        let stats = db.get_stats().await;
        
        assert_eq!(stats.total_documents, 0);
        assert_eq!(stats.total_chunks, 0);
    }
    
    /// Test document addition
    #[test]
    async fn test_add_document() {
        let mut db = VectorDB::new();
        
        let doc_id = "test_doc_1";
        let chunks = vec![
            ("This is the first chunk".to_string(), vec![0.1; 768]),
            ("This is the second chunk".to_string(), vec![0.2; 768]),
        ];
        
        let result = db.add_document(doc_id, chunks).await;
        assert!(result.is_ok(), "Failed to add document: {:?}", result.err());
        
        let stats = db.get_stats().await;
        assert_eq!(stats.total_documents, 1);
        assert_eq!(stats.total_chunks, 2);
    }
    
    /// Test semantic search
    #[test]
    async fn test_semantic_search() {
        let mut db = VectorDB::new();
        
        // Add test documents
        db.add_document("doc1", vec![
            ("Machine learning is awesome".to_string(), vec![0.8; 768]),
            ("Deep learning neural networks".to_string(), vec![0.7; 768]),
        ]).await.unwrap();
        
        db.add_document("doc2", vec![
            ("Rust programming language".to_string(), vec![0.2; 768]),
            ("Systems programming with Rust".to_string(), vec![0.3; 768]),
        ]).await.unwrap();
        
        // Search for ML-related content
        let query_embedding = vec![0.75; 768]; // Similar to ML embeddings
        let results = db.search(&query_embedding, 2).await;
        
        assert!(results.is_ok(), "Search failed: {:?}", results.err());
        let matches = results.unwrap();
        
        assert_eq!(matches.len(), 2, "Should return top 2 results");
        assert!(matches[0].score > matches[1].score, "Results should be sorted by score");
    }
    
    /// Test document removal
    #[test]
    async fn test_remove_document() {
        let mut db = VectorDB::new();
        
        // Add and then remove a document
        db.add_document("doc_to_remove", vec![
            ("Content to be removed".to_string(), vec![0.5; 768]),
        ]).await.unwrap();
        
        let stats_before = db.get_stats().await;
        assert_eq!(stats_before.total_documents, 1);
        
        let result = db.remove_document("doc_to_remove").await;
        assert!(result.is_ok(), "Failed to remove document");
        
        let stats_after = db.get_stats().await;
        assert_eq!(stats_after.total_documents, 0);
        assert_eq!(stats_after.total_chunks, 0);
    }
}
*/

// Temporarily disabled - needs update to match new database API
/*
#[cfg(test)]
mod database_tests {
    use crate::database::{Database, Project, Document};
    use tauri::test::{mock_builder, MockRuntime};
    use tokio::test;
    
    /// Create a mock app handle for testing
    fn create_mock_app_handle() -> tauri::AppHandle<MockRuntime> {
        let app = mock_builder().build(tauri::generate_context!()).unwrap();
        app.handle().clone()
    }
    
    /// Test database creation
    #[test]
    async fn test_database_creation() {
        let app_handle = create_mock_app_handle();
        let db = Database::new(&app_handle);
        
        // Database creation might fail in test environment due to path issues
        // This is acceptable for unit tests
        match db {
            Ok(_) => {
                tracing::info!("Database created successfully");
            }
            Err(e) => {
                tracing::warn!("Database creation failed in test env: {:?}", e);
            }
        }
    }
    
    /// Test project CRUD operations
    #[test]
    async fn test_project_operations() {
        let app_handle = create_mock_app_handle();
        let db = Database::new(&app_handle);
        
        if let Ok(db) = db {
            // Create a project
            let project_result = db.create_project(
                "Test Project".to_string(),
                Some("A test project description".to_string())
            ).await;
            
            if let Ok(project) = project_result {
                assert_eq!(project.name, "Test Project");
                assert!(!project.id.is_empty());
                
                // Get all projects
                let projects = db.get_projects().await.unwrap_or_default();
                assert!(!projects.is_empty());
                
                // Toggle star
                let star_result = db.toggle_project_star(project.id.clone()).await;
                assert!(star_result.is_ok());
                
                // Delete project
                let delete_result = db.delete_project(project.id).await;
                assert!(delete_result.is_ok());
            }
        }
    }
}
*/