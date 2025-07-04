use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, debug};
use crate::inference::ModelResponse;

pub mod vision_processor;
pub mod audio_processor;
pub mod document_processor;
pub mod input_pipeline;
pub mod output_formatter;
// Temporarily disable complex modules to fix core compilation
// pub mod enhanced_processor;
// pub mod enhanced_vision;
// pub mod unified_interface;
// Omni-specific components
pub mod enhanced_audio_processor;
pub mod speech_synthesizer;
pub mod video_processor;
pub mod thinker_talker_processor;

pub use vision_processor::*;
pub use audio_processor::*;
pub use document_processor::*;
pub use input_pipeline::*;
pub use output_formatter::*;
// pub use enhanced_processor::*;
// pub use enhanced_vision::*;
// pub use unified_interface::*;
// Omni-specific exports
pub use enhanced_audio_processor::*;
pub use speech_synthesizer::*;
pub use video_processor::*;
pub use thinker_talker_processor::*;

use crate::inference::{
    MultimodalInput, ImageData, AudioData, DocumentData,
    ImageFormat, AudioFormat, DocumentFormat
};

/// Central coordinator for multimodal input processing
pub struct MultimodalProcessor {
    vision_processor: VisionProcessor,
    audio_processor: AudioProcessor,
    document_processor: DocumentProcessor,
    input_pipeline: InputPipeline,
    output_formatter: OutputFormatter,
}

impl MultimodalProcessor {
    pub fn new() -> Result<Self> {
        info!("Initializing multimodal processor");
        
        Ok(Self {
            vision_processor: VisionProcessor::new()?,
            audio_processor: AudioProcessor::new()?,
            document_processor: DocumentProcessor::new()?,
            input_pipeline: InputPipeline::new(),
            output_formatter: OutputFormatter::new(),
        })
    }
    
    /// Process a single file into appropriate multimodal input
    pub async fn process_file(&self, file_path: &str, file_name: &str) -> Result<MultimodalInput> {
        info!("Processing file: {} ({})", file_name, file_path);
        
        let file_data = std::fs::read(file_path)?;
        self.process_file_data(&file_data, Some(file_name)).await
    }
    
    /// Process file data with optional filename hint
    pub async fn process_file_data(&self, data: &[u8], filename: Option<&str>) -> Result<MultimodalInput> {
        debug!("Processing {} bytes of file data", data.len());
        
        // Auto-detect content type
        let content_type = self.detect_content_type(data, filename)?;
        
        match content_type {
            ContentType::Image(format) => {
                let image_data = self.vision_processor.process_image_data(data, format).await?;
                Ok(MultimodalInput::TextWithImage {
                    text: "Analyze this image".to_string(),
                    image: image_data,
                })
            }
            
            ContentType::Audio(format) => {
                let audio_data = self.audio_processor.process_audio_data(data, format).await?;
                Ok(MultimodalInput::TextWithAudio {
                    text: "Process this audio".to_string(),
                    audio: audio_data,
                })
            }
            
            ContentType::Document(format) => {
                let document_data = self.document_processor.process_document_data(data, format).await?;
                Ok(MultimodalInput::TextWithDocument {
                    text: "Analyze this document".to_string(),
                    document: document_data,
                })
            }
            
            ContentType::Text => {
                let text = String::from_utf8_lossy(data).to_string();
                Ok(MultimodalInput::Text(text))
            }
        }
    }
    
    /// Process multiple inputs into a combined multimodal input
    pub async fn process_combined_input(
        &self,
        text: Option<String>,
        images: Vec<&[u8]>,
        audio: Option<&[u8]>,
        documents: Vec<&[u8]>,
    ) -> Result<MultimodalInput> {
        info!("Processing combined multimodal input with {} images, {} audio, {} documents", 
              images.len(), if audio.is_some() { 1 } else { 0 }, documents.len());
        
        let mut processed_images = Vec::new();
        for image_data in images {
            if let Ok(ContentType::Image(format)) = self.detect_content_type(image_data, None) {
                let image = self.vision_processor.process_image_data(image_data, format).await?;
                processed_images.push(image);
            }
        }
        
        let processed_audio = if let Some(audio_data) = audio {
            if let Ok(ContentType::Audio(format)) = self.detect_content_type(audio_data, None) {
                Some(self.audio_processor.process_audio_data(audio_data, format).await?)
            } else {
                None
            }
        } else {
            None
        };
        
        let mut processed_documents = Vec::new();
        for doc_data in documents {
            if let Ok(ContentType::Document(format)) = self.detect_content_type(doc_data, None) {
                let document = self.document_processor.process_document_data(doc_data, format).await?;
                processed_documents.push(document);
            }
        }
        
        Ok(MultimodalInput::Combined {
            text,
            images: processed_images,
            audio: processed_audio,
            documents: processed_documents,
        })
    }
    
    /// Format multimodal output for display
    pub async fn format_output(&self, response: &ModelResponse) -> Result<FormattedResponse> {
        self.output_formatter.format_response(response)
    }
    
    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        ProcessingStats {
            images_processed: self.vision_processor.get_processed_count(),
            audio_processed: self.audio_processor.get_processed_count(),
            documents_processed: self.document_processor.get_processed_count(),
            total_processing_time_ms: 0, // Would track across all processors
        }
    }
    
    fn detect_content_type(&self, data: &[u8], filename: Option<&str>) -> Result<ContentType> {
        // Check by filename extension first
        if let Some(name) = filename {
            let name_lower = name.to_lowercase();
            
            // Image extensions
            if name_lower.ends_with(".jpg") || name_lower.ends_with(".jpeg") {
                return Ok(ContentType::Image(ImageFormat::Jpeg));
            } else if name_lower.ends_with(".png") {
                return Ok(ContentType::Image(ImageFormat::Png));
            } else if name_lower.ends_with(".gif") {
                return Ok(ContentType::Image(ImageFormat::Gif));
            } else if name_lower.ends_with(".webp") {
                return Ok(ContentType::Image(ImageFormat::WebP));
            } else if name_lower.ends_with(".bmp") {
                return Ok(ContentType::Image(ImageFormat::Bmp));
            }
            
            // Audio extensions
            else if name_lower.ends_with(".wav") {
                return Ok(ContentType::Audio(AudioFormat::Wav));
            } else if name_lower.ends_with(".mp3") {
                return Ok(ContentType::Audio(AudioFormat::Mp3));
            } else if name_lower.ends_with(".flac") {
                return Ok(ContentType::Audio(AudioFormat::Flac));
            } else if name_lower.ends_with(".ogg") {
                return Ok(ContentType::Audio(AudioFormat::Ogg));
            }
            
            // Document extensions
            else if name_lower.ends_with(".pdf") {
                return Ok(ContentType::Document(DocumentFormat::Pdf));
            } else if name_lower.ends_with(".docx") {
                return Ok(ContentType::Document(DocumentFormat::Docx));
            } else if name_lower.ends_with(".txt") {
                return Ok(ContentType::Document(DocumentFormat::Txt));
            } else if name_lower.ends_with(".md") || name_lower.ends_with(".markdown") {
                return Ok(ContentType::Document(DocumentFormat::Markdown));
            } else if name_lower.ends_with(".json") {
                return Ok(ContentType::Document(DocumentFormat::Json));
            }
        }
        
        // Fall back to magic byte detection
        self.detect_content_type_by_magic(data)
    }
    
    fn detect_content_type_by_magic(&self, data: &[u8]) -> Result<ContentType> {
        if data.len() < 4 {
            return Ok(ContentType::Text);
        }
        
        // Image magic bytes
        match &data[0..4] {
            [0xFF, 0xD8, 0xFF, _] => return Ok(ContentType::Image(ImageFormat::Jpeg)),
            [0x89, 0x50, 0x4E, 0x47] => return Ok(ContentType::Image(ImageFormat::Png)),
            [0x47, 0x49, 0x46, 0x38] => return Ok(ContentType::Image(ImageFormat::Gif)),
            [0x42, 0x4D, _, _] => return Ok(ContentType::Image(ImageFormat::Bmp)),
            _ => {}
        }
        
        // Check for WEBP (needs more bytes)
        if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
            return Ok(ContentType::Image(ImageFormat::WebP));
        }
        
        // Audio magic bytes
        match &data[0..4] {
            [0x52, 0x49, 0x46, 0x46] if data.len() >= 12 && &data[8..12] == b"WAVE" => {
                return Ok(ContentType::Audio(AudioFormat::Wav));
            }
            [0xFF, 0xFB, _, _] | [0xFF, 0xFA, _, _] | [0xFF, 0xF3, _, _] | [0xFF, 0xF2, _, _] => {
                return Ok(ContentType::Audio(AudioFormat::Mp3));
            }
            [0x66, 0x4C, 0x61, 0x43] => return Ok(ContentType::Audio(AudioFormat::Flac)),
            [0x4F, 0x67, 0x67, 0x53] => return Ok(ContentType::Audio(AudioFormat::Ogg)),
            _ => {}
        }
        
        // Document magic bytes
        match &data[0..4] {
            [0x25, 0x50, 0x44, 0x46] => return Ok(ContentType::Document(DocumentFormat::Pdf)), // %PDF
            [0x50, 0x4B, 0x03, 0x04] => return Ok(ContentType::Document(DocumentFormat::Docx)), // ZIP-based
            _ => {}
        }
        
        // Try to detect if it's valid UTF-8 text
        if let Ok(text) = std::str::from_utf8(data) {
            let trimmed = text.trim_start();
            if trimmed.starts_with('{') || trimmed.starts_with('[') {
                return Ok(ContentType::Document(DocumentFormat::Json));
            } else if trimmed.contains("# ") || trimmed.contains("## ") || trimmed.contains("```") {
                return Ok(ContentType::Document(DocumentFormat::Markdown));
            } else {
                return Ok(ContentType::Text);
            }
        }
        
        // Default to text if nothing else matches
        Ok(ContentType::Text)
    }
}

#[derive(Debug, Clone)]
enum ContentType {
    Image(ImageFormat),
    Audio(AudioFormat),
    Document(DocumentFormat),
    Text,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    pub images_processed: usize,
    pub audio_processed: usize,
    pub documents_processed: usize,
    pub total_processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedOutput {
    pub text: String,
    pub structured_data: Option<HashMap<String, serde_json::Value>>,
    pub media_references: Vec<MediaReference>,
    pub formatting_metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaReference {
    pub media_type: String,
    pub description: String,
    pub confidence: f32,
}

/// Utility functions for common multimodal operations
pub mod utils {
    use super::*;
    
    /// Resize image data to maximum dimensions
    pub fn resize_image_if_needed(data: &[u8], max_width: u32, max_height: u32) -> Result<Vec<u8>> {
        use image::{ImageFormat as ImgFormat, GenericImageView};
        
        let img = image::load_from_memory(data)?;
        let (width, height) = img.dimensions();
        
        if width <= max_width && height <= max_height {
            return Ok(data.to_vec());
        }
        
        // Calculate new dimensions maintaining aspect ratio
        let ratio = (max_width as f32 / width as f32).min(max_height as f32 / height as f32);
        let new_width = (width as f32 * ratio) as u32;
        let new_height = (height as f32 * ratio) as u32;
        
        let resized = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);
        
        let mut buffer = Vec::new();
        resized.write_to(&mut std::io::Cursor::new(&mut buffer), ImgFormat::Png)?;
        
        Ok(buffer)
    }
    
    /// Convert audio to standard format if needed
    pub fn normalize_audio_format(data: &[u8], target_sample_rate: u32) -> Result<Vec<u8>> {
        // This would implement audio format conversion
        // For now, just return the original data
        Ok(data.to_vec())
    }
    
    /// Extract text content from document data
    pub async fn extract_document_text(data: &[u8], format: DocumentFormat) -> Result<String> {
        match format {
            DocumentFormat::Txt | DocumentFormat::Markdown => {
                Ok(String::from_utf8_lossy(data).to_string())
            }
            DocumentFormat::Json => {
                let json_value: serde_json::Value = serde_json::from_slice(data)?;
                Ok(json_value.to_string())
            }
            DocumentFormat::Pdf => {
                // Would implement PDF text extraction
                Ok("PDF text extraction not yet implemented".to_string())
            }
            DocumentFormat::Docx => {
                // Would implement DOCX text extraction
                Ok("DOCX text extraction not yet implemented".to_string())
            }
        }
    }
}