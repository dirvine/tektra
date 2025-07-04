use super::*;
use anyhow::Result;
use std::path::Path;

/// Input pipeline for processing and validating multimodal inputs
pub struct InputPipeline {
    max_file_size: usize,
    supported_image_formats: Vec<ImageFormat>,
    supported_audio_formats: Vec<AudioFormat>,
    supported_document_formats: Vec<DocumentFormat>,
}

impl InputPipeline {
    pub fn new() -> Self {
        Self {
            max_file_size: 100 * 1024 * 1024, // 100MB
            supported_image_formats: vec![
                ImageFormat::Jpeg,
                ImageFormat::Png,
                ImageFormat::Gif,
                ImageFormat::WebP,
                ImageFormat::Bmp,
            ],
            supported_audio_formats: vec![
                AudioFormat::Wav,
                AudioFormat::Mp3,
                AudioFormat::Flac,
                AudioFormat::Ogg,
            ],
            supported_document_formats: vec![
                DocumentFormat::Pdf,
                DocumentFormat::Docx,
                DocumentFormat::Txt,
                DocumentFormat::Markdown,
                DocumentFormat::Json,
            ],
        }
    }

    /// Validate a multimodal input
    pub async fn validate_input(&self, input: &MultimodalInput) -> Result<()> {
        match input {
            MultimodalInput::Text(text) => {
                if text.is_empty() {
                    return Err(anyhow::anyhow!("Text input cannot be empty"));
                }
                if text.len() > 100_000 {
                    return Err(anyhow::anyhow!("Text input too long"));
                }
            }
            MultimodalInput::TextWithImage { text, image } => {
                if text.is_empty() {
                    return Err(anyhow::anyhow!("Text input cannot be empty"));
                }
                self.validate_image(image)?;
            }
            MultimodalInput::TextWithAudio { text, audio } => {
                if text.is_empty() {
                    return Err(anyhow::anyhow!("Text input cannot be empty"));
                }
                self.validate_audio(audio)?;
            }
            MultimodalInput::TextWithDocument { text, document } => {
                if text.is_empty() {
                    return Err(anyhow::anyhow!("Text input cannot be empty"));
                }
                self.validate_document(document)?;
            }
            MultimodalInput::Combined { text, images, audio, documents } => {
                if let Some(text) = text {
                    if text.is_empty() {
                        return Err(anyhow::anyhow!("Text input cannot be empty"));
                    }
                }
                
                for image in images {
                    self.validate_image(image)?;
                }
                
                if let Some(audio) = audio {
                    self.validate_audio(audio)?;
                }
                
                for document in documents {
                    self.validate_document(document)?;
                }
            }
            // Handle new Omni input types
            MultimodalInput::TextWithVideo { text, video } => {
                if text.is_empty() {
                    return Err(anyhow::anyhow!("Text input cannot be empty"));
                }
                self.validate_video(video)?;
            }
            MultimodalInput::RealTimeAudio { audio_stream } => {
                self.validate_audio_stream(audio_stream)?;
            }
            MultimodalInput::MultimodalConversation { text, images, audio, video, documents, .. } => {
                if let Some(text) = text {
                    if text.is_empty() {
                        return Err(anyhow::anyhow!("Text input cannot be empty"));
                    }
                }
                
                for image in images {
                    self.validate_image(image)?;
                }
                
                if let Some(audio) = audio {
                    self.validate_audio(audio)?;
                }
                
                if let Some(video) = video {
                    self.validate_video(video)?;
                }
                
                for document in documents {
                    self.validate_document(document)?;
                }
            }
        }
        
        Ok(())
    }

    /// Validate an image
    fn validate_image(&self, image: &ImageData) -> Result<()> {
        if !self.supported_image_formats.contains(&image.format) {
            return Err(anyhow::anyhow!("Unsupported image format: {:?}", image.format));
        }
        
        if image.data.len() > self.max_file_size {
            return Err(anyhow::anyhow!("Image too large"));
        }
        
        Ok(())
    }

    /// Validate an audio file
    fn validate_audio(&self, audio: &AudioData) -> Result<()> {
        if !self.supported_audio_formats.contains(&audio.format) {
            return Err(anyhow::anyhow!("Unsupported audio format: {:?}", audio.format));
        }
        
        if audio.data.len() > self.max_file_size {
            return Err(anyhow::anyhow!("Audio file too large"));
        }
        
        Ok(())
    }

    /// Validate a document
    fn validate_document(&self, document: &DocumentData) -> Result<()> {
        if !self.supported_document_formats.contains(&document.format) {
            return Err(anyhow::anyhow!("Unsupported document format: {:?}", document.format));
        }
        
        if document.data.len() > self.max_file_size {
            return Err(anyhow::anyhow!("Document too large"));
        }
        
        Ok(())
    }

    /// Validate a video
    fn validate_video(&self, video: &crate::inference::VideoData) -> Result<()> {
        if video.data.len() > self.max_file_size {
            return Err(anyhow::anyhow!("Video too large"));
        }
        
        // Check duration if available
        if let Some(duration) = video.duration {
            if duration > 300.0 { // 5 minute limit
                return Err(anyhow::anyhow!("Video too long (max 5 minutes)"));
            }
        }
        
        Ok(())
    }

    /// Validate an audio stream
    fn validate_audio_stream(&self, stream: &crate::inference::AudioStream) -> Result<()> {
        if stream.chunk_data.len() > 1_000_000 { // 1MB chunk limit
            return Err(anyhow::anyhow!("Audio chunk too large"));
        }
        
        if stream.sample_rate < 8000 || stream.sample_rate > 48000 {
            return Err(anyhow::anyhow!("Unsupported sample rate: {}", stream.sample_rate));
        }
        
        Ok(())
    }

    /// Process a file path into multimodal input
    pub async fn process_file_path(&self, path: &Path) -> Result<MultimodalInput> {
        let data = tokio::fs::read(path).await?;
        let filename = path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown");
        
        // Determine file type by extension
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        match extension.as_str() {
            "jpg" | "jpeg" => {
                let image = ImageData {
                    data,
                    format: ImageFormat::Jpeg,
                    width: None,
                    height: None,
                };
                Ok(MultimodalInput::TextWithImage {
                    text: format!("Analyze this image: {}", filename),
                    image,
                })
            }
            "png" => {
                let image = ImageData {
                    data,
                    format: ImageFormat::Png,
                    width: None,
                    height: None,
                };
                Ok(MultimodalInput::TextWithImage {
                    text: format!("Analyze this image: {}", filename),
                    image,
                })
            }
            "gif" => {
                let image = ImageData {
                    data,
                    format: ImageFormat::Gif,
                    width: None,
                    height: None,
                };
                Ok(MultimodalInput::TextWithImage {
                    text: format!("Analyze this image: {}", filename),
                    image,
                })
            }
            "webp" => {
                let image = ImageData {
                    data,
                    format: ImageFormat::WebP,
                    width: None,
                    height: None,
                };
                Ok(MultimodalInput::TextWithImage {
                    text: format!("Analyze this image: {}", filename),
                    image,
                })
            }
            "bmp" => {
                let image = ImageData {
                    data,
                    format: ImageFormat::Bmp,
                    width: None,
                    height: None,
                };
                Ok(MultimodalInput::TextWithImage {
                    text: format!("Analyze this image: {}", filename),
                    image,
                })
            }
            "wav" => {
                let audio = AudioData {
                    data,
                    format: AudioFormat::Wav,
                    sample_rate: None,
                    channels: None,
                    duration: None,
                };
                Ok(MultimodalInput::TextWithAudio {
                    text: format!("Transcribe this audio: {}", filename),
                    audio,
                })
            }
            "mp3" => {
                let audio = AudioData {
                    data,
                    format: AudioFormat::Mp3,
                    sample_rate: None,
                    channels: None,
                    duration: None,
                };
                Ok(MultimodalInput::TextWithAudio {
                    text: format!("Transcribe this audio: {}", filename),
                    audio,
                })
            }
            "txt" => {
                let document = DocumentData {
                    data,
                    format: DocumentFormat::Txt,
                    title: Some(filename.to_string()),
                    metadata: std::collections::HashMap::new(),
                };
                Ok(MultimodalInput::TextWithDocument {
                    text: format!("Analyze this document: {}", filename),
                    document,
                })
            }
            "pdf" => {
                let document = DocumentData {
                    data,
                    format: DocumentFormat::Pdf,
                    title: Some(filename.to_string()),
                    metadata: std::collections::HashMap::new(),
                };
                Ok(MultimodalInput::TextWithDocument {
                    text: format!("Analyze this document: {}", filename),
                    document,
                })
            }
            _ => {
                // Default to text
                let text = String::from_utf8_lossy(&data).to_string();
                Ok(MultimodalInput::Text(text))
            }
        }
    }
}