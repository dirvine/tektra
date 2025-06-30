use anyhow::Result;
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};
use tracing::{info, error, warn};

/// Gemma 3N-specific multimodal processing optimizations
/// Handles preprocessing for USM audio encoder and MobileNet-V5 vision encoder

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalInput {
    pub text: Option<String>,
    pub image_data: Option<Vec<u8>>,
    pub audio_data: Option<Vec<u8>>,
    pub video_data: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedMultimodal {
    pub prompt: String,
    pub images: Vec<String>, // Base64 encoded
    pub audio_embeddings: Option<Vec<f32>>, // For future direct audio support
    pub token_count: usize,
}

#[derive(Debug, Clone)]
pub struct Gemma3NProcessor {
    /// Context window management for 32K tokens
    max_context_tokens: usize,
    /// Audio preprocessing for USM encoder (16kHz, 160ms tokens)
    audio_sample_rate: u32,
    audio_chunk_duration_ms: u32,
    /// Vision preprocessing for MobileNet-V5 (multiple resolutions)
    supported_image_sizes: Vec<(u32, u32)>,
}

impl Default for Gemma3NProcessor {
    fn default() -> Self {
        Self {
            max_context_tokens: 32000, // Gemma 3N context window
            audio_sample_rate: 16000,  // USM encoder requirement
            audio_chunk_duration_ms: 160, // 6 tokens/second for USM
            supported_image_sizes: vec![
                (256, 256),  // MobileNet-V5 primary
                (512, 512),  // MobileNet-V5 enhanced
                (768, 768),  // MobileNet-V5 high-res
                (896, 896),  // Gemma 3N standard
            ],
        }
    }
}

impl Gemma3NProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Process multimodal input for optimal Gemma 3N performance
    pub async fn process_multimodal(&self, input: MultimodalInput) -> Result<ProcessedMultimodal> {
        let mut prompt_parts = Vec::new();
        let mut images = Vec::new();
        let mut token_count = 0;

        // Process text content
        if let Some(text) = input.text {
            token_count += self.estimate_tokens(&text);
            prompt_parts.push(text);
        }

        // Process image data with MobileNet-V5 optimization
        if let Some(image_data) = input.image_data {
            match self.process_image_for_mobilenet_v5(&image_data).await {
                Ok(processed_image) => {
                    images.push(processed_image);
                    token_count += 256; // Standard image token count for Gemma 3N
                    info!("Processed image for MobileNet-V5 encoder");
                }
                Err(e) => {
                    error!("Failed to process image for MobileNet-V5: {}", e);
                    return Err(e);
                }
            }
        }

        // Process audio data for USM encoder
        if let Some(audio_data) = input.audio_data {
            match self.process_audio_for_usm(&audio_data).await {
                Ok(audio_tokens) => {
                    token_count += audio_tokens;
                    prompt_parts.push(format!("[Audio: {} USM tokens]", audio_tokens));
                    info!("Processed audio for USM encoder: {} tokens", audio_tokens);
                }
                Err(e) => {
                    warn!("Audio processing failed, using fallback: {}", e);
                    prompt_parts.push("[Audio processing unavailable]".to_string());
                }
            }
        }

        // Handle video data (if supported by Ollama in future)
        if let Some(_video_data) = input.video_data {
            warn!("Video processing not yet supported in Ollama for Gemma 3N");
            prompt_parts.push("[Video data provided but not processed - Ollama limitation]".to_string());
        }

        // Check context window limits
        if token_count > self.max_context_tokens {
            warn!("Content exceeds 32K context window ({} tokens), truncating", token_count);
            // Implement smart truncation strategy
            prompt_parts = self.truncate_content(prompt_parts, token_count)?;
            token_count = self.max_context_tokens - 1000; // Leave buffer for response
        }

        let final_prompt = prompt_parts.join("\n\n");

        Ok(ProcessedMultimodal {
            prompt: final_prompt,
            images,
            audio_embeddings: None, // Future enhancement
            token_count,
        })
    }

    /// Optimize image for MobileNet-V5 encoder requirements
    async fn process_image_for_mobilenet_v5(&self, image_data: &[u8]) -> Result<String> {
        use image::{ImageReader, GenericImageView, imageops::FilterType};
        use std::io::Cursor;

        // Decode image
        let img = ImageReader::new(Cursor::new(image_data))
            .with_guessed_format()?
            .decode()?;

        // Choose optimal resolution for MobileNet-V5
        let (width, height) = img.dimensions();
        let target_size = self.choose_optimal_resolution(width, height);
        
        info!("Resizing image from {}x{} to {}x{} for MobileNet-V5", 
              width, height, target_size.0, target_size.1);

        // Resize using high-quality filter for better MobileNet-V5 performance
        let resized_img = img.resize_exact(target_size.0, target_size.1, FilterType::Lanczos3);

        // Convert to RGB for consistent processing
        let rgb_img = resized_img.to_rgb8();

        // Encode as PNG for better quality preservation
        let mut png_data = Vec::new();
        {
            use image::codecs::png::PngEncoder;
            use image::ImageEncoder;
            
            let encoder = PngEncoder::new(&mut png_data);
            encoder.write_image(
                rgb_img.as_raw(),
                target_size.0,
                target_size.1,
                image::ColorType::Rgb8.into(),
            )?;
        }

        // Base64 encode for Ollama
        let base64_image = general_purpose::STANDARD.encode(&png_data);
        
        info!("Generated base64 image: {} bytes → {} chars", 
              image_data.len(), base64_image.len());

        Ok(base64_image)
    }

    /// Choose optimal resolution for MobileNet-V5 encoder
    fn choose_optimal_resolution(&self, width: u32, height: u32) -> (u32, u32) {
        let aspect_ratio = width as f32 / height as f32;
        
        // Find best size that preserves aspect ratio while optimizing for MobileNet-V5
        let mut best_size = self.supported_image_sizes[0];
        let mut best_score = f32::MAX;
        
        for &(target_w, target_h) in &self.supported_image_sizes {
            let target_aspect = target_w as f32 / target_h as f32;
            let aspect_diff = (aspect_ratio - target_aspect).abs();
            let size_score = aspect_diff + (target_w as f32 / 1000.0); // Prefer smaller sizes if equal
            
            if size_score < best_score {
                best_score = size_score;
                best_size = (target_w, target_h);
            }
        }
        
        info!("Selected {}x{} (score: {:.3}) for aspect ratio {:.3}", 
              best_size.0, best_size.1, best_score, aspect_ratio);
        
        best_size
    }

    /// Process audio for USM encoder (16kHz, 160ms chunks)
    async fn process_audio_for_usm(&self, audio_data: &[u8]) -> Result<usize> {
        // Calculate duration and expected token count
        let sample_count = audio_data.len() / 2; // 16-bit samples
        let duration_ms = (sample_count as f32 / self.audio_sample_rate as f32) * 1000.0;
        
        // USM encoder produces 6 tokens per second (160ms per token)
        let expected_tokens = ((duration_ms / 160.0).ceil() as usize).max(1);
        
        info!("Audio duration: {:.1}ms, expected USM tokens: {}", duration_ms, expected_tokens);
        
        // Validate audio format for USM encoder
        if audio_data.len() % 2 != 0 {
            return Err(anyhow::anyhow!("Audio data must be 16-bit aligned"));
        }
        
        // For now, return expected token count
        // Future: implement actual USM preprocessing when Ollama supports direct audio
        Ok(expected_tokens)
    }

    /// Estimate token count for text (rough approximation)
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough approximation: 4 characters per token on average
        (text.len() / 4).max(1)
    }

    /// Smart truncation to fit context window
    fn truncate_content(&self, mut parts: Vec<String>, current_tokens: usize) -> Result<Vec<String>> {
        let target_tokens = self.max_context_tokens - 1000; // Leave buffer
        let reduction_ratio = target_tokens as f32 / current_tokens as f32;
        
        info!("Truncating content by {:.1}% to fit 32K context window", 
              (1.0 - reduction_ratio) * 100.0);
        
        // Smart truncation: preserve beginning and end, compress middle
        for part in &mut parts {
            if part.len() > 1000 {
                let new_len = (part.len() as f32 * reduction_ratio) as usize;
                let keep_start = new_len / 3;
                let keep_end = new_len / 3;
                
                if keep_start + keep_end < part.len() {
                    let start_text = &part[..keep_start];
                    let end_text = &part[part.len() - keep_end..];
                    *part = format!("{}...[content truncated]...{}", start_text, end_text);
                }
            }
        }
        
        Ok(parts)
    }

    /// Format prompt with proper Gemma 3N multimodal markers
    pub fn format_for_gemma3n(&self, processed: &ProcessedMultimodal, system_prompt: Option<&str>) -> String {
        let mut formatted = String::new();
        
        // Add system prompt if provided
        if let Some(system) = system_prompt {
            formatted.push_str(&format!(
                "<start_of_turn>system\n{}\n<end_of_turn>\n",
                system
            ));
        }
        
        // Add user content with multimodal markers
        formatted.push_str("<start_of_turn>user\n");
        
        // Add image markers if present
        for (i, _) in processed.images.iter().enumerate() {
            formatted.push_str(&format!("<start_of_image>Image {}<end_of_image>\n", i + 1));
        }
        
        // Add the main prompt
        formatted.push_str(&processed.prompt);
        formatted.push_str("\n<end_of_turn>\n<start_of_turn>model\n");
        
        info!("Formatted prompt for Gemma 3N: {} tokens, {} images", 
              processed.token_count, processed.images.len());
        
        formatted
    }

    /// Get memory usage estimate for processing
    pub fn estimate_memory_usage(&self, input: &MultimodalInput) -> usize {
        let mut memory_mb = 0;
        
        if let Some(image_data) = &input.image_data {
            memory_mb += (image_data.len() * 3) / (1024 * 1024); // RGB expansion
        }
        
        if let Some(audio_data) = &input.audio_data {
            memory_mb += (audio_data.len() * 2) / (1024 * 1024); // Processing overhead
        }
        
        memory_mb.max(1)
    }

    /// Performance optimization recommendations
    pub fn get_optimization_suggestions(&self, processed: &ProcessedMultimodal) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        if processed.token_count > 20000 {
            suggestions.push("Consider reducing context size for faster processing".to_string());
        }
        
        if processed.images.len() > 2 {
            suggestions.push("Multiple images may impact response time".to_string());
        }
        
        if processed.token_count < 1000 {
            suggestions.push("More context could improve response quality".to_string());
        }
        
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_processor_creation() {
        let processor = Gemma3NProcessor::new();
        assert_eq!(processor.max_context_tokens, 32000);
        assert_eq!(processor.audio_sample_rate, 16000);
    }

    #[tokio::test]
    async fn test_token_estimation() {
        let processor = Gemma3NProcessor::new();
        let text = "Hello world, this is a test message for token estimation.";
        let tokens = processor.estimate_tokens(text);
        assert!(tokens > 0);
        assert!(tokens < 100); // Reasonable range
    }

    #[test]
    fn test_resolution_selection() {
        let processor = Gemma3NProcessor::new();
        
        // Test square image
        let (w, h) = processor.choose_optimal_resolution(800, 800);
        assert!(w == h); // Should remain square
        
        // Test landscape image
        let (w, h) = processor.choose_optimal_resolution(1920, 1080);
        assert!(w > h); // Should maintain aspect ratio preference
        
        // Test portrait image
        let (w, h) = processor.choose_optimal_resolution(600, 900);
        assert!(h > w); // Should maintain aspect ratio preference
    }
}