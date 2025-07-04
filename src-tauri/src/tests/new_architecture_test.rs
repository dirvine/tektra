#[cfg(test)]
mod new_architecture_tests {
    use crate::inference::*;
    use crate::inference::quantization::QuantizationManager;
    use crate::multimodal::*;
    use crate::conversation::*;
    use tokio;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_model_registry_initialization() {
        let registry = ModelRegistry::new();
        
        // Test that we can initialize the registry
        let result = registry.initialize().await;
        assert!(result.is_ok(), "Model registry should initialize successfully");
        
        // Test that default models are available
        let models = registry.list_models().await;
        assert!(!models.is_empty(), "Should have default models available");
        
        // Check that at least one model supports vision
        let vision_models: Vec<_> = models.iter()
            .filter(|m| m.supports_vision)
            .collect();
        assert!(!vision_models.is_empty(), "Should have at least one vision model");
    }

    #[tokio::test]
    async fn test_multimodal_processor_initialization() {
        let processor = MultimodalProcessor::new();
        assert!(processor.is_ok(), "Multimodal processor should initialize successfully");
        
        let processor = processor.unwrap();
        let stats = processor.get_stats();
        
        // Initial stats should be zero
        assert_eq!(stats.images_processed, 0);
        assert_eq!(stats.audio_processed, 0);
        assert_eq!(stats.documents_processed, 0);
    }

    #[tokio::test]
    async fn test_conversation_manager_initialization() {
        let config = ConversationConfig::default();
        let manager = ConversationManager::new(Some(config));
        assert!(manager.is_ok(), "Conversation manager should initialize successfully");
        
        let manager = manager.unwrap();
        let session_count = manager.get_active_session_count().await;
        assert_eq!(session_count, 0, "Should start with no active sessions");
    }

    #[tokio::test]
    async fn test_conversation_session_creation() {
        let config = ConversationConfig::default();
        let mut manager = ConversationManager::new(Some(config)).unwrap();
        
        // Start a new session
        let session_id = manager.start_session("test_session".to_string(), None).await;
        assert!(session_id.is_ok(), "Should be able to start a session");
        
        let session_id = session_id.unwrap();
        assert_eq!(session_id, "test_session", "Session ID should match");
        
        // Check that session is active
        let session_count = manager.get_active_session_count().await;
        assert_eq!(session_count, 1, "Should have one active session");
        
        // End the session
        let result = manager.end_session(&session_id).await;
        assert!(result.is_ok(), "Should be able to end session");
        
        // Check that session is no longer active
        let session_count = manager.get_active_session_count().await;
        assert_eq!(session_count, 0, "Should have no active sessions");
    }

    #[tokio::test]
    async fn test_multimodal_input_creation() {
        // Test text input
        let text_input = MultimodalInput::Text("Hello, world!".to_string());
        match text_input {
            MultimodalInput::Text(text) => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected text input"),
        }

        // Test text with image input
        let image_data = ImageData {
            data: vec![1, 2, 3, 4], // Dummy image data
            format: ImageFormat::Png,
            width: Some(100),
            height: Some(100),
        };
        
        let image_input = MultimodalInput::TextWithImage {
            text: "Describe this image".to_string(),
            image: image_data,
        };
        
        match image_input {
            MultimodalInput::TextWithImage { text, image } => {
                assert_eq!(text, "Describe this image");
                assert_eq!(image.format, ImageFormat::Png);
                assert_eq!(image.width, Some(100));
                assert_eq!(image.height, Some(100));
            }
            _ => panic!("Expected text with image input"),
        }
    }

    #[tokio::test]
    async fn test_content_type_detection() {
        let processor = MultimodalProcessor::new().unwrap();
        
        // Test PNG detection
        let png_header = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let result = processor.process_file_data(&png_header, Some("test.png")).await;
        assert!(result.is_ok(), "Should process PNG data");

        // Test JPEG detection
        let jpeg_header = vec![0xFF, 0xD8, 0xFF, 0xE0];
        let result = processor.process_file_data(&jpeg_header, Some("test.jpg")).await;
        assert!(result.is_ok(), "Should process JPEG data");

        // Test text detection
        let text_data = "Hello, this is a test document.".as_bytes();
        let result = processor.process_file_data(text_data, Some("test.txt")).await;
        assert!(result.is_ok(), "Should process text data");
    }

    #[tokio::test]
    async fn test_vision_processor() {
        let processor = VisionProcessor::new().unwrap();
        
        // Test with dummy PNG data
        let png_data = create_dummy_png();
        let result = processor.process_image_data(&png_data, ImageFormat::Png).await;
        assert!(result.is_ok(), "Should process image data");
        
        let processed = result.unwrap();
        assert_eq!(processed.format, ImageFormat::Png);
        assert!(processed.width.is_some());
        assert!(processed.height.is_some());
        
        // Check processing count
        assert_eq!(processor.get_processed_count(), 1);
    }

    #[tokio::test]
    async fn test_audio_processor() {
        let processor = AudioProcessor::new().unwrap();
        
        // Test with dummy WAV data
        let wav_data = create_dummy_wav();
        let result = processor.process_audio_data(&wav_data, AudioFormat::Wav).await;
        assert!(result.is_ok(), "Should process audio data");
        
        let processed = result.unwrap();
        assert_eq!(processed.format, AudioFormat::Wav);
        assert!(processed.sample_rate.is_some());
        assert!(processed.channels.is_some());
        
        // Check processing count
        assert_eq!(processor.get_processed_count(), 1);
    }

    #[tokio::test]
    async fn test_document_processor() {
        let processor = DocumentProcessor::new().unwrap();
        
        // Test with text document
        let text_data = "This is a test document with some content.".as_bytes();
        let result = processor.process_document_data(text_data, DocumentFormat::Txt).await;
        assert!(result.is_ok(), "Should process document data");
        
        let processed = result.unwrap();
        assert_eq!(processed.format, DocumentFormat::Txt);
        
        // Test document analysis
        let analysis = processor.analyze_document(&processed).await;
        assert!(analysis.is_ok(), "Should analyze document");
        
        let analysis = analysis.unwrap();
        assert!(analysis.word_count > 0);
        assert!(analysis.text_length > 0);
        
        // Check processing count
        assert_eq!(processor.get_processed_count(), 1);
    }

    #[tokio::test]
    async fn test_memory_store() {
        let mut memory_store = MemoryStore::new().unwrap();
        
        // Test adding a fact
        let result = memory_store.add_fact("testing", "This is a test fact").await;
        assert!(result.is_ok(), "Should be able to add facts");
        
        // Test setting preferences
        let result = memory_store.set_preference("test_session", "language", "en").await;
        assert!(result.is_ok(), "Should be able to set preferences");
        
        // Test searching (should return None for empty conversation history)
        let input = MultimodalInput::Text("Tell me about testing".to_string());
        let result = memory_store.search_relevant_context("test_session", &input).await;
        assert!(result.is_ok(), "Should be able to search context");
    }

    #[tokio::test]
    async fn test_prompt_template_manager() {
        let template_manager = PromptTemplateManager::new().unwrap();
        
        // Test getting a template for a known model
        let template = template_manager.get_template("qwen2.5-vl");
        assert!(template.is_some(), "Should have template for Qwen2.5-VL");
        
        let template = template.unwrap();
        assert!(template.supports_images, "Qwen2.5-VL should support images");
        assert!(template.supports_system, "Qwen2.5-VL should support system prompts");
    }

    #[tokio::test]
    async fn test_persona_manager() {
        let persona_manager = PersonaManager::new().unwrap();
        
        // Test getting default persona
        let persona = persona_manager.get_default_persona().await;
        assert!(persona.is_ok(), "Should have default persona");
        
        let persona = persona.unwrap();
        assert_eq!(persona.name, "Helpful Assistant");
        
        // Test listing personas
        let personas = persona_manager.list_personas().await;
        assert!(!personas.is_empty(), "Should have multiple personas");
        assert!(personas.contains(&"default".to_string()));
        assert!(personas.contains(&"technical".to_string()));
    }

    #[tokio::test]
    async fn test_quantization_manager() {
        let qm = QuantizationManager::new();
        
        // Test getting format info
        let format_info = qm.get_format_info("Q4_K_M");
        assert!(format_info.is_some(), "Should have Q4_K_M format info");
        
        let format_info = format_info.unwrap();
        assert_eq!(format_info.name, "Q4_K_M");
        assert!(format_info.memory_reduction > 0.0);
        
        // Test listing supported formats
        let formats = qm.list_supported_formats();
        assert!(!formats.is_empty(), "Should have supported formats");
        
        // Test format validation
        let result = qm.validate_format("Qwen/Qwen2.5-VL-7B", "Q4_K_M");
        assert!(result.is_ok(), "Q4_K_M should be valid for Qwen models");
    }

    #[tokio::test]
    async fn test_token_estimator() {
        let estimator = TokenEstimator::new();
        
        // Test token estimation
        let text = "This is a test sentence with several words.";
        let token_count = estimator.estimate_tokens(text);
        assert!(token_count > 0, "Should estimate some tokens");
        assert!(token_count < text.len(), "Token count should be less than character count");
        
        // Test text truncation
        let long_text = "This is a very long text that should be truncated when it exceeds the maximum token limit.";
        let truncated = estimator.truncate_to_tokens(long_text, 5);
        assert!(truncated.len() < long_text.len(), "Text should be truncated");
        assert!(truncated.ends_with("..."), "Truncated text should end with ellipsis");
    }

    // Helper functions for creating dummy data

    fn create_dummy_png() -> Vec<u8> {
        // Create a minimal valid PNG header
        vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, // IHDR chunk length
            0x49, 0x48, 0x44, 0x52, // IHDR
            0x00, 0x00, 0x00, 0x01, // Width: 1
            0x00, 0x00, 0x00, 0x01, // Height: 1
            0x08, 0x02, 0x00, 0x00, 0x00, // Bit depth, color type, compression, filter, interlace
            0x90, 0x77, 0x53, 0xDE, // CRC
            0x00, 0x00, 0x00, 0x00, // IEND chunk length
            0x49, 0x45, 0x4E, 0x44, // IEND
            0xAE, 0x42, 0x60, 0x82, // CRC
        ]
    }

    fn create_dummy_wav() -> Vec<u8> {
        // Create a minimal valid WAV header
        let mut data = Vec::new();
        
        // RIFF header
        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&36u32.to_le_bytes()); // File size - 8
        data.extend_from_slice(b"WAVE");
        
        // fmt chunk
        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // Chunk size
        data.extend_from_slice(&1u16.to_le_bytes());  // Audio format (PCM)
        data.extend_from_slice(&1u16.to_le_bytes());  // Channels
        data.extend_from_slice(&16000u32.to_le_bytes()); // Sample rate
        data.extend_from_slice(&32000u32.to_le_bytes()); // Byte rate
        data.extend_from_slice(&2u16.to_le_bytes());  // Block align
        data.extend_from_slice(&16u16.to_le_bytes()); // Bits per sample
        
        // data chunk
        data.extend_from_slice(b"data");
        data.extend_from_slice(&0u32.to_le_bytes()); // Data size
        
        data
    }
}