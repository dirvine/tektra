use super::*;
use anyhow::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::collections::HashMap;
use tracing::{info, warn, debug};

/// Document processing for PDFs, DOCX, text files, and other document formats
pub struct DocumentProcessor {
    processed_count: AtomicUsize,
    max_document_size: usize,
    supported_formats: Vec<DocumentFormat>,
}

impl DocumentProcessor {
    pub fn new() -> Result<Self> {
        info!("Initializing document processor");
        
        Ok(Self {
            processed_count: AtomicUsize::new(0),
            max_document_size: 10 * 1024 * 1024, // 10MB max
            supported_formats: vec![
                DocumentFormat::Pdf,
                DocumentFormat::Docx,
                DocumentFormat::Txt,
                DocumentFormat::Markdown,
                DocumentFormat::Json,
            ],
        })
    }
    
    /// Process document data and return structured DocumentData
    pub async fn process_document_data(&self, data: &[u8], format: DocumentFormat) -> Result<DocumentData> {
        debug!("Processing document data: {} bytes, format: {:?}", data.len(), format);
        
        if data.len() > self.max_document_size {
            return Err(anyhow::anyhow!(
                "Document too large: {} bytes (max: {} bytes)", 
                data.len(), 
                self.max_document_size
            ));
        }
        
        // Extract text content
        let text_content = self.extract_text_content(data, &format).await?;
        let metadata = self.extract_metadata(data, &format).await?;
        
        info!("Processed document: {} characters extracted", text_content.len());
        
        // Update processing count
        self.processed_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(DocumentData {
            data: data.to_vec(),
            format,
            title: metadata.get("title").cloned(),
            metadata,
        })
    }
    
    /// Analyze document content and structure
    pub async fn analyze_document(&self, document_data: &DocumentData) -> Result<DocumentAnalysis> {
        debug!("Analyzing document content");
        
        let text_content = self.extract_text_content(&document_data.data, &document_data.format).await?;
        
        let analysis = DocumentAnalysis {
            text_length: text_content.len(),
            word_count: self.count_words(&text_content),
            paragraph_count: self.count_paragraphs(&text_content),
            estimated_reading_time: self.estimate_reading_time(&text_content),
            language: self.detect_language(&text_content),
            document_type: self.classify_document_type(&text_content),
            key_topics: self.extract_key_topics(&text_content),
            complexity_score: self.calculate_complexity(&text_content),
        };
        
        debug!("Document analysis complete: {:?}", analysis);
        Ok(analysis)
    }
    
    /// Extract text content for search indexing
    pub async fn extract_searchable_text(&self, document_data: &DocumentData) -> Result<String> {
        let text_content = self.extract_text_content(&document_data.data, &document_data.format).await?;
        
        // Clean and normalize text for search
        let cleaned_text = self.clean_text_for_search(&text_content);
        
        Ok(cleaned_text)
    }
    
    /// Generate document summary
    pub async fn generate_summary(&self, document_data: &DocumentData, max_length: usize) -> Result<String> {
        let text_content = self.extract_text_content(&document_data.data, &document_data.format).await?;
        
        if text_content.len() <= max_length {
            return Ok(text_content);
        }
        
        // Simple extractive summarization
        let sentences = self.split_into_sentences(&text_content);
        let scored_sentences = self.score_sentences(&sentences);
        
        // Select top sentences up to max_length
        let mut summary = String::new();
        for (sentence, _score) in scored_sentences {
            if summary.len() + sentence.len() > max_length {
                break;
            }
            if !summary.is_empty() {
                summary.push(' ');
            }
            summary.push_str(&sentence);
        }
        
        Ok(summary)
    }
    
    /// Process multiple documents for batch operations
    pub async fn process_document_batch(&self, documents: Vec<(&[u8], DocumentFormat)>) -> Result<Vec<DocumentData>> {
        info!("Processing batch of {} documents", documents.len());
        
        let mut results = Vec::with_capacity(documents.len());
        
        let document_count = documents.len();
        for (data, format) in documents {
            match self.process_document_data(data, format).await {
                Ok(document_data) => results.push(document_data),
                Err(e) => {
                    warn!("Failed to process document in batch: {}", e);
                    // Continue processing other documents
                }
            }
        }
        
        info!("Successfully processed {}/{} documents in batch", results.len(), document_count);
        Ok(results)
    }
    
    /// Get processing statistics
    pub fn get_processed_count(&self) -> usize {
        self.processed_count.load(Ordering::Relaxed)
    }
    
    /// Check if format is supported
    pub fn is_format_supported(&self, format: &DocumentFormat) -> bool {
        self.supported_formats.contains(format)
    }
    
    // Private helper methods
    
    async fn extract_text_content(&self, data: &[u8], format: &DocumentFormat) -> Result<String> {
        match format {
            DocumentFormat::Txt => {
                String::from_utf8(data.to_vec())
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in text file: {}", e))
            }
            
            DocumentFormat::Markdown => {
                let markdown_text = String::from_utf8(data.to_vec())
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in markdown file: {}", e))?;
                
                // For now, return markdown as-is
                // Could implement markdown parsing to extract just text
                Ok(markdown_text)
            }
            
            DocumentFormat::Json => {
                let json_text = String::from_utf8(data.to_vec())
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in JSON file: {}", e))?;
                
                // Parse JSON and extract meaningful text
                self.extract_text_from_json(&json_text)
            }
            
            DocumentFormat::Pdf => {
                self.extract_pdf_text(data).await
            }
            
            DocumentFormat::Docx => {
                self.extract_docx_text(data).await
            }
        }
    }
    
    async fn extract_metadata(&self, data: &[u8], format: &DocumentFormat) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        
        match format {
            DocumentFormat::Pdf => {
                // Extract PDF metadata
                metadata.insert("format".to_string(), "PDF".to_string());
                metadata.insert("size".to_string(), data.len().to_string());
            }
            
            DocumentFormat::Docx => {
                // Extract DOCX metadata
                metadata.insert("format".to_string(), "DOCX".to_string());
                metadata.insert("size".to_string(), data.len().to_string());
            }
            
            DocumentFormat::Json => {
                // Try to extract metadata from JSON structure
                if let Ok(text) = String::from_utf8(data.to_vec()) {
                    if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&text) {
                        if let Some(obj) = json_value.as_object() {
                            // Look for common metadata fields
                            for (key, value) in obj {
                                if ["title", "author", "description", "version"].contains(&key.as_str()) {
                                    if let Some(string_value) = value.as_str() {
                                        metadata.insert(key.clone(), string_value.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
                metadata.insert("format".to_string(), "JSON".to_string());
            }
            
            _ => {
                metadata.insert("format".to_string(), format!("{:?}", format));
                metadata.insert("size".to_string(), data.len().to_string());
            }
        }
        
        Ok(metadata)
    }
    
    fn extract_text_from_json(&self, json_text: &str) -> Result<String> {
        let json_value: serde_json::Value = serde_json::from_str(json_text)?;
        
        let mut extracted_text = Vec::new();
        self.extract_strings_from_json_value(&json_value, &mut extracted_text);
        
        Ok(extracted_text.join(" "))
    }
    
    fn extract_strings_from_json_value(&self, value: &serde_json::Value, strings: &mut Vec<String>) {
        match value {
            serde_json::Value::String(s) => {
                if !s.trim().is_empty() {
                    strings.push(s.clone());
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr {
                    self.extract_strings_from_json_value(item, strings);
                }
            }
            serde_json::Value::Object(obj) => {
                for (_key, val) in obj {
                    self.extract_strings_from_json_value(val, strings);
                }
            }
            _ => {}
        }
    }
    
    async fn extract_pdf_text(&self, _data: &[u8]) -> Result<String> {
        // PDF text extraction would be implemented here using a library like pdf-extract
        warn!("PDF text extraction not yet implemented");
        Ok("PDF text extraction not available".to_string())
    }
    
    async fn extract_docx_text(&self, _data: &[u8]) -> Result<String> {
        // DOCX text extraction would be implemented here
        warn!("DOCX text extraction not yet implemented");
        Ok("DOCX text extraction not available".to_string())
    }
    
    fn count_words(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
    
    fn count_paragraphs(&self, text: &str) -> usize {
        text.split("\n\n").filter(|p| !p.trim().is_empty()).count()
    }
    
    fn estimate_reading_time(&self, text: &str) -> f32 {
        let word_count = self.count_words(text);
        let words_per_minute = 200.0; // Average reading speed
        word_count as f32 / words_per_minute
    }
    
    fn detect_language(&self, _text: &str) -> Option<String> {
        // Language detection would be implemented here
        // Could use statistical analysis or ML models
        Some("en".to_string()) // Default to English for now
    }
    
    fn classify_document_type(&self, text: &str) -> DocumentType {
        let text_lower = text.to_lowercase();
        
        // Simple heuristic-based classification
        if text_lower.contains("abstract") && text_lower.contains("conclusion") {
            DocumentType::Academic
        } else if text_lower.contains("summary") && text_lower.contains("quarterly") {
            DocumentType::Report
        } else if text_lower.contains("dear") && text_lower.contains("sincerely") {
            DocumentType::Letter
        } else if text.lines().any(|line| line.trim().starts_with("# ")) {
            DocumentType::Documentation
        } else if text_lower.contains("recipe") || text_lower.contains("ingredients") {
            DocumentType::Recipe
        } else {
            DocumentType::General
        }
    }
    
    fn extract_key_topics(&self, text: &str) -> Vec<String> {
        // Simple keyword extraction
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_freq = HashMap::new();
        
        for word in words {
            let cleaned_word = word.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase();
            if cleaned_word.len() > 4 { // Only consider longer words
                *word_freq.entry(cleaned_word).or_insert(0) += 1;
            }
        }
        
        // Get top 5 most frequent words
        let mut freq_pairs: Vec<_> = word_freq.into_iter().collect();
        freq_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        
        freq_pairs.into_iter()
            .take(5)
            .map(|(word, _)| word)
            .collect()
    }
    
    fn calculate_complexity(&self, text: &str) -> f32 {
        let word_count = self.count_words(text);
        let sentence_count = text.split(&['.', '!', '?']).count();
        let unique_words = text.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        if sentence_count == 0 || word_count == 0 {
            return 0.0;
        }
        
        let avg_sentence_length = word_count as f32 / sentence_count as f32;
        let vocabulary_richness = unique_words as f32 / word_count as f32;
        
        // Normalize complexity score between 0.0 and 1.0
        ((avg_sentence_length / 20.0) + vocabulary_richness).min(1.0)
    }
    
    fn clean_text_for_search(&self, text: &str) -> String {
        // Remove extra whitespace and normalize text
        text.split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    }
    
    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(&['.', '!', '?'])
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }
    
    fn score_sentences(&self, sentences: &[String]) -> Vec<(String, f32)> {
        // Simple sentence scoring based on length and position
        sentences.iter()
            .enumerate()
            .map(|(i, sentence)| {
                let length_score = (sentence.len() as f32 / 100.0).min(1.0);
                let position_score = if i < sentences.len() / 3 { 1.0 } else { 0.5 };
                (sentence.clone(), length_score * position_score)
            })
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentAnalysis {
    pub text_length: usize,
    pub word_count: usize,
    pub paragraph_count: usize,
    pub estimated_reading_time: f32, // minutes
    pub language: Option<String>,
    pub document_type: DocumentType,
    pub key_topics: Vec<String>,
    pub complexity_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentType {
    Academic,
    Report,
    Letter,
    Documentation,
    Recipe,
    Legal,
    Technical,
    Creative,
    General,
}

/// OCR processing for image-based documents
pub struct OCRProcessor {
    confidence_threshold: f32,
}

impl OCRProcessor {
    pub fn new() -> Result<Self> {
        Ok(Self {
            confidence_threshold: 0.8,
        })
    }
    
    /// Extract text from image using OCR
    pub async fn extract_text_from_image(&self, _image_data: &[u8]) -> Result<OCRResult> {
        // OCR implementation would go here
        // Could use libraries like tesseract-rs
        warn!("OCR text extraction not yet implemented");
        
        Ok(OCRResult {
            text: "OCR not available".to_string(),
            confidence: 0.0,
            bounding_boxes: Vec::new(),
        })
    }
    
    /// Process scanned document with multiple pages
    pub async fn process_scanned_document(&self, _pages: Vec<&[u8]>) -> Result<Vec<OCRResult>> {
        warn!("Scanned document processing not yet implemented");
        Ok(Vec::new())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OCRResult {
    pub text: String,
    pub confidence: f32,
    pub bounding_boxes: Vec<BoundingBox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub text: String,
    pub confidence: f32,
}

/// Document utility functions
pub mod document_utils {
    use super::*;
    
    /// Convert document format enum to string
    pub fn format_to_string(format: &DocumentFormat) -> &'static str {
        match format {
            DocumentFormat::Pdf => "pdf",
            DocumentFormat::Docx => "docx",
            DocumentFormat::Txt => "txt",
            DocumentFormat::Markdown => "markdown",
            DocumentFormat::Json => "json",
        }
    }
    
    /// Convert string to document format enum
    pub fn string_to_format(format_str: &str) -> Option<DocumentFormat> {
        match format_str.to_lowercase().as_str() {
            "pdf" => Some(DocumentFormat::Pdf),
            "docx" | "doc" => Some(DocumentFormat::Docx),
            "txt" | "text" => Some(DocumentFormat::Txt),
            "md" | "markdown" => Some(DocumentFormat::Markdown),
            "json" => Some(DocumentFormat::Json),
            _ => None,
        }
    }
    
    /// Estimate processing time based on document size
    pub fn estimate_processing_time(file_size_bytes: usize) -> std::time::Duration {
        // Rough estimate: 1 second per MB
        let size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);
        std::time::Duration::from_secs_f64(size_mb.max(0.1))
    }
    
    /// Check if document size is within limits
    pub fn validate_document_size(size_bytes: usize, max_size_bytes: usize) -> Result<()> {
        if size_bytes > max_size_bytes {
            return Err(anyhow::anyhow!(
                "Document too large: {} bytes (max: {} bytes)",
                size_bytes,
                max_size_bytes
            ));
        }
        Ok(())
    }
    
    /// Extract filename extension
    pub fn get_file_extension(filename: &str) -> Option<String> {
        std::path::Path::new(filename)
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
    }
}