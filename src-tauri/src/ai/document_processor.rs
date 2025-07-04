use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

/// Supported document formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentFormat {
    Pdf,
    Docx,
    Txt,
    Markdown,
    Html,
    Rtf,
    Json,
    Xml,
    Csv,
}

impl DocumentFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "pdf" => Some(DocumentFormat::Pdf),
            "docx" | "doc" => Some(DocumentFormat::Docx),
            "txt" | "text" => Some(DocumentFormat::Txt),
            "md" | "markdown" => Some(DocumentFormat::Markdown),
            "html" | "htm" => Some(DocumentFormat::Html),
            "rtf" => Some(DocumentFormat::Rtf),
            "json" => Some(DocumentFormat::Json),
            "xml" => Some(DocumentFormat::Xml),
            "csv" => Some(DocumentFormat::Csv),
            _ => None,
        }
    }
}

/// Chunking strategies for document processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkingStrategy {
    /// Fixed-size chunks with overlap
    FixedSize { size: usize, overlap: usize },
    /// Semantic chunking based on paragraphs/sections
    Semantic,
    /// Sliding window with dynamic size
    SlidingWindow { min_size: usize, max_size: usize },
    /// Recursive chunking that respects document structure
    Recursive,
    /// Sentence-based chunking
    Sentence { max_sentences: usize },
}

impl Default for ChunkingStrategy {
    fn default() -> Self {
        ChunkingStrategy::FixedSize {
            size: 1000,
            overlap: 200,
        }
    }
}

/// Processed document metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub creation_date: Option<String>,
    pub modification_date: Option<String>,
    pub page_count: Option<usize>,
    pub word_count: usize,
    pub language: Option<String>,
    pub keywords: Vec<String>,
    pub custom_metadata: std::collections::HashMap<String, String>,
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            title: None,
            author: None,
            creation_date: None,
            modification_date: None,
            page_count: None,
            word_count: 0,
            language: None,
            keywords: Vec::new(),
            custom_metadata: std::collections::HashMap::new(),
        }
    }
}

/// Processed document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedDocument {
    pub id: String,
    pub source_path: Option<PathBuf>,
    pub format: DocumentFormat,
    pub raw_text: String,
    pub structured_content: StructuredContent,
    pub metadata: DocumentMetadata,
    pub processing_timestamp: chrono::DateTime<chrono::Utc>,
}

/// Structured content representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredContent {
    pub sections: Vec<DocumentSection>,
    pub tables: Vec<Table>,
    pub images: Vec<ImageReference>,
    pub links: Vec<Link>,
    pub footnotes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    pub level: u8, // Heading level (1-6)
    pub title: Option<String>,
    pub content: String,
    pub start_offset: usize,
    pub end_offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Table {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub caption: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageReference {
    pub path: Option<String>,
    pub caption: Option<String>,
    pub alt_text: Option<String>,
    pub data: Option<Vec<u8>>, // Embedded image data
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Link {
    pub url: String,
    pub text: String,
    pub link_type: LinkType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkType {
    External,
    Internal,
    Email,
    Phone,
}

/// Document chunk for vector storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub content: String,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub section_path: Vec<String>, // Hierarchical section path
    pub chunk_type: ChunkType,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChunkType {
    Text,
    Code { language: Option<String> },
    Heading { level: u8 },
    List,
    Table,
    Quote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub keywords: Vec<String>,
    pub entities: Vec<String>, // Named entities
    pub importance_score: f32,
    pub has_math: bool,
    pub has_code: bool,
    pub language: Option<String>,
}

/// Core trait for document processing
#[async_trait]
pub trait DocumentProcessor: Send + Sync {
    /// Process raw document data into structured format
    async fn process(&self, data: Vec<u8>, format: DocumentFormat) -> Result<ProcessedDocument>;
    
    /// Extract chunks from processed document
    async fn extract_chunks(
        &self,
        doc: &ProcessedDocument,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>>;
    
    /// Extract text from specific page range (if applicable)
    async fn extract_page_range(
        &self,
        data: Vec<u8>,
        format: DocumentFormat,
        start_page: usize,
        end_page: usize,
    ) -> Result<String>;
    
    /// Get format-specific capabilities
    fn supported_formats(&self) -> Vec<DocumentFormat>;
}

/// Unified document processor that delegates to format-specific processors
pub struct UnifiedDocumentProcessor {
    processors: Arc<RwLock<std::collections::HashMap<DocumentFormat, Box<dyn DocumentProcessor>>>>,
}

impl UnifiedDocumentProcessor {
    pub fn new() -> Self {
        let mut processors: std::collections::HashMap<DocumentFormat, Box<dyn DocumentProcessor>> =
            std::collections::HashMap::new();
        
        // Register format-specific processors
        processors.insert(DocumentFormat::Txt, Box::new(TextProcessor::new()));
        processors.insert(DocumentFormat::Markdown, Box::new(MarkdownProcessor::new()));
        processors.insert(DocumentFormat::Json, Box::new(JsonProcessor::new()));
        processors.insert(DocumentFormat::Pdf, Box::new(PdfProcessor::new()));
        
        Self {
            processors: Arc::new(RwLock::new(processors)),
        }
    }
    
    pub async fn process_file(&self, path: &Path) -> Result<ProcessedDocument> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| anyhow::anyhow!("No file extension found"))?;
        
        let format = DocumentFormat::from_extension(ext)
            .ok_or_else(|| anyhow::anyhow!("Unsupported file format: {}", ext))?;
        
        let data = tokio::fs::read(path).await?;
        self.process(data, format).await
    }
    
    pub async fn process(&self, data: Vec<u8>, format: DocumentFormat) -> Result<ProcessedDocument> {
        let processors = self.processors.read().await;
        let processor = processors
            .get(&format)
            .ok_or_else(|| anyhow::anyhow!("No processor for format: {:?}", format))?;
        
        processor.process(data, format).await
    }
    
    pub async fn extract_chunks(
        &self,
        doc: &ProcessedDocument,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>> {
        let processors = self.processors.read().await;
        let processor = processors
            .get(&doc.format)
            .ok_or_else(|| anyhow::anyhow!("No processor for format: {:?}", doc.format))?;
        
        processor.extract_chunks(doc, strategy).await
    }
}

/// Text file processor
struct TextProcessor;

impl TextProcessor {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentProcessor for TextProcessor {
    async fn process(&self, data: Vec<u8>, _format: DocumentFormat) -> Result<ProcessedDocument> {
        let text = String::from_utf8(data)?;
        let word_count = text.split_whitespace().count();
        
        Ok(ProcessedDocument {
            id: uuid::Uuid::new_v4().to_string(),
            source_path: None,
            format: DocumentFormat::Txt,
            raw_text: text.clone(),
            structured_content: StructuredContent {
                sections: vec![DocumentSection {
                    level: 1,
                    title: None,
                    content: text,
                    start_offset: 0,
                    end_offset: word_count,
                }],
                tables: Vec::new(),
                images: Vec::new(),
                links: Vec::new(),
                footnotes: Vec::new(),
            },
            metadata: DocumentMetadata {
                word_count,
                ..Default::default()
            },
            processing_timestamp: chrono::Utc::now(),
        })
    }
    
    async fn extract_chunks(
        &self,
        doc: &ProcessedDocument,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        
        match strategy {
            ChunkingStrategy::FixedSize { size, overlap } => {
                let text = &doc.raw_text;
                let chars: Vec<char> = text.chars().collect();
                let mut start = 0;
                let mut chunk_index = 0;
                
                while start < chars.len() {
                    let end = (start + size).min(chars.len());
                    let chunk_text: String = chars[start..end].iter().collect();
                    
                    chunks.push(DocumentChunk {
                        id: format!("{}-chunk-{}", doc.id, chunk_index),
                        document_id: doc.id.clone(),
                        content: chunk_text,
                        chunk_index,
                        start_char: start,
                        end_char: end,
                        section_path: vec!["root".to_string()],
                        chunk_type: ChunkType::Text,
                        metadata: ChunkMetadata {
                            keywords: Vec::new(),
                            entities: Vec::new(),
                            importance_score: 1.0,
                            has_math: false,
                            has_code: false,
                            language: None,
                        },
                    });
                    
                    chunk_index += 1;
                    start = if start + size >= chars.len() {
                        chars.len()
                    } else {
                        start + size - overlap
                    };
                }
            }
            _ => {
                // For other strategies, fall back to single chunk
                chunks.push(DocumentChunk {
                    id: format!("{}-chunk-0", doc.id),
                    document_id: doc.id.clone(),
                    content: doc.raw_text.clone(),
                    chunk_index: 0,
                    start_char: 0,
                    end_char: doc.raw_text.len(),
                    section_path: vec!["root".to_string()],
                    chunk_type: ChunkType::Text,
                    metadata: ChunkMetadata {
                        keywords: Vec::new(),
                        entities: Vec::new(),
                        importance_score: 1.0,
                        has_math: false,
                        has_code: false,
                        language: None,
                    },
                });
            }
        }
        
        Ok(chunks)
    }
    
    async fn extract_page_range(
        &self,
        data: Vec<u8>,
        _format: DocumentFormat,
        _start_page: usize,
        _end_page: usize,
    ) -> Result<String> {
        // Text files don't have pages, return full content
        Ok(String::from_utf8(data)?)
    }
    
    fn supported_formats(&self) -> Vec<DocumentFormat> {
        vec![DocumentFormat::Txt]
    }
}

/// Markdown processor
struct MarkdownProcessor;

impl MarkdownProcessor {
    fn new() -> Self {
        Self
    }
    
    fn parse_markdown(&self, text: &str) -> StructuredContent {
        let mut sections = Vec::new();
        let tables = Vec::new();
        let mut links = Vec::new();
        let mut current_section = String::new();
        let mut current_level = 0u8;
        let mut current_title = None;
        let mut start_offset = 0;
        
        for (line_num, line) in text.lines().enumerate() {
            // Parse headings
            if line.starts_with('#') {
                // Save previous section if any
                if !current_section.is_empty() {
                    sections.push(DocumentSection {
                        level: current_level,
                        title: current_title.clone(),
                        content: current_section.trim().to_string(),
                        start_offset,
                        end_offset: line_num,
                    });
                    current_section.clear();
                    start_offset = line_num;
                }
                
                let level = line.chars().take_while(|&c| c == '#').count() as u8;
                current_level = level;
                current_title = Some(line.trim_start_matches('#').trim().to_string());
            } else {
                current_section.push_str(line);
                current_section.push('\n');
            }
            
            // Parse links
            let link_regex = regex::Regex::new(r"\[([^\]]+)\]\(([^\)]+)\)").unwrap();
            for cap in link_regex.captures_iter(line) {
                links.push(Link {
                    text: cap[1].to_string(),
                    url: cap[2].to_string(),
                    link_type: if cap[2].starts_with("http") {
                        LinkType::External
                    } else if cap[2].starts_with("mailto:") {
                        LinkType::Email
                    } else {
                        LinkType::Internal
                    },
                });
            }
        }
        
        // Don't forget the last section
        if !current_section.is_empty() {
            sections.push(DocumentSection {
                level: current_level,
                title: current_title,
                content: current_section.trim().to_string(),
                start_offset,
                end_offset: text.lines().count(),
            });
        }
        
        StructuredContent {
            sections,
            tables,
            images: Vec::new(),
            links,
            footnotes: Vec::new(),
        }
    }
}

#[async_trait]
impl DocumentProcessor for MarkdownProcessor {
    async fn process(&self, data: Vec<u8>, _format: DocumentFormat) -> Result<ProcessedDocument> {
        let text = String::from_utf8(data)?;
        let word_count = text.split_whitespace().count();
        let structured_content = self.parse_markdown(&text);
        
        Ok(ProcessedDocument {
            id: uuid::Uuid::new_v4().to_string(),
            source_path: None,
            format: DocumentFormat::Markdown,
            raw_text: text,
            structured_content,
            metadata: DocumentMetadata {
                word_count,
                ..Default::default()
            },
            processing_timestamp: chrono::Utc::now(),
        })
    }
    
    async fn extract_chunks(
        &self,
        doc: &ProcessedDocument,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>> {
        let mut chunks = Vec::new();
        
        match strategy {
            ChunkingStrategy::Semantic => {
                // Chunk by sections
                for (idx, section) in doc.structured_content.sections.iter().enumerate() {
                    let chunk_type = if section.title.is_some() {
                        ChunkType::Heading { level: section.level }
                    } else {
                        ChunkType::Text
                    };
                    
                    chunks.push(DocumentChunk {
                        id: format!("{}-chunk-{}", doc.id, idx),
                        document_id: doc.id.clone(),
                        content: format!(
                            "{}{}",
                            section.title.as_deref().unwrap_or(""),
                            section.content
                        ),
                        chunk_index: idx,
                        start_char: section.start_offset,
                        end_char: section.end_offset,
                        section_path: vec![
                            section.title.clone().unwrap_or_else(|| format!("section-{}", idx))
                        ],
                        chunk_type,
                        metadata: ChunkMetadata {
                            keywords: Vec::new(),
                            entities: Vec::new(),
                            importance_score: 1.0 / (section.level as f32), // Higher level = lower score
                            has_math: section.content.contains("$"),
                            has_code: section.content.contains("```"),
                            language: None,
                        },
                    });
                }
            }
            _ => {
                // Fall back to text processor for other strategies
                let text_processor = TextProcessor::new();
                return text_processor.extract_chunks(doc, strategy).await;
            }
        }
        
        Ok(chunks)
    }
    
    async fn extract_page_range(
        &self,
        data: Vec<u8>,
        _format: DocumentFormat,
        _start_page: usize,
        _end_page: usize,
    ) -> Result<String> {
        // Markdown doesn't have pages, return full content
        Ok(String::from_utf8(data)?)
    }
    
    fn supported_formats(&self) -> Vec<DocumentFormat> {
        vec![DocumentFormat::Markdown]
    }
}

/// JSON processor
struct JsonProcessor;

impl JsonProcessor {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentProcessor for JsonProcessor {
    async fn process(&self, data: Vec<u8>, _format: DocumentFormat) -> Result<ProcessedDocument> {
        let text = String::from_utf8(data)?;
        let json_value: serde_json::Value = serde_json::from_str(&text)?;
        let pretty_json = serde_json::to_string_pretty(&json_value)?;
        let word_count = pretty_json.split_whitespace().count();
        
        Ok(ProcessedDocument {
            id: uuid::Uuid::new_v4().to_string(),
            source_path: None,
            format: DocumentFormat::Json,
            raw_text: pretty_json.clone(),
            structured_content: StructuredContent {
                sections: vec![DocumentSection {
                    level: 1,
                    title: Some("JSON Document".to_string()),
                    content: pretty_json,
                    start_offset: 0,
                    end_offset: word_count,
                }],
                tables: Vec::new(),
                images: Vec::new(),
                links: Vec::new(),
                footnotes: Vec::new(),
            },
            metadata: DocumentMetadata {
                word_count,
                ..Default::default()
            },
            processing_timestamp: chrono::Utc::now(),
        })
    }
    
    async fn extract_chunks(
        &self,
        doc: &ProcessedDocument,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>> {
        // Delegate to text processor for now
        let text_processor = TextProcessor::new();
        text_processor.extract_chunks(doc, strategy).await
    }
    
    async fn extract_page_range(
        &self,
        data: Vec<u8>,
        _format: DocumentFormat,
        _start_page: usize,
        _end_page: usize,
    ) -> Result<String> {
        Ok(String::from_utf8(data)?)
    }
    
    fn supported_formats(&self) -> Vec<DocumentFormat> {
        vec![DocumentFormat::Json]
    }
}

/// PDF processor using lopdf crate
struct PdfProcessor;

impl PdfProcessor {
    fn new() -> Self {
        Self
    }
}

#[async_trait]
impl DocumentProcessor for PdfProcessor {
    async fn process(&self, data: Vec<u8>, _format: DocumentFormat) -> Result<ProcessedDocument> {
        use lopdf::Document as PdfDocument;
        
        // Parse PDF document
        let pdf = PdfDocument::load_mem(&data)
            .map_err(|e| anyhow!("Failed to parse PDF: {}", e))?;
        
        let mut all_text = String::new();
        let mut sections = Vec::new();
        let page_count = pdf.get_pages().len();
        
        // Extract text from each page
        for (page_num, page_id) in pdf.get_pages().iter().enumerate() {
            match pdf.extract_text(&[*page_id.0]) {
                Ok(text) => {
                    if !text.trim().is_empty() {
                        let start_offset = all_text.len();
                        all_text.push_str(&text);
                        all_text.push_str("\n\n");
                        
                        sections.push(DocumentSection {
                            level: 2,
                            title: Some(format!("Page {}", page_num + 1)),
                            content: text.clone(),
                            start_offset,
                            end_offset: all_text.len(),
                        });
                    }
                }
                Err(e) => {
                    warn!("Failed to extract text from page {}: {}", page_num + 1, e);
                    // Continue processing other pages
                }
            }
        }
        
        // If no text was extracted, provide informative message
        if all_text.trim().is_empty() {
            all_text = format!("[PDF with {} pages - no extractable text found. The PDF may contain only images or be encrypted.]", page_count);
            sections.push(DocumentSection {
                level: 1,
                title: Some("PDF Document".to_string()),
                content: all_text.clone(),
                start_offset: 0,
                end_offset: all_text.len(),
            });
        }
        
        // Extract metadata
        let metadata = DocumentMetadata {
            word_count: all_text.split_whitespace().count(),
            page_count: Some(page_count),
            title: None, // PDF metadata extraction temporarily disabled due to API changes
            author: None, // PDF metadata extraction temporarily disabled due to API changes
            creation_date: None, // Would need to parse PDF date format
            modification_date: None,
            language: None,
            ..Default::default()
        };
        
        Ok(ProcessedDocument {
            id: uuid::Uuid::new_v4().to_string(),
            source_path: None,
            format: DocumentFormat::Pdf,
            raw_text: all_text,
            structured_content: StructuredContent {
                sections,
                tables: Vec::new(), // PDF table extraction would require more complex parsing
                images: Vec::new(), // Image extraction not implemented yet
                links: Vec::new(),
                footnotes: Vec::new(),
            },
            metadata,
            processing_timestamp: chrono::Utc::now(),
        })
    }
    
    async fn extract_chunks(
        &self,
        doc: &ProcessedDocument,
        strategy: ChunkingStrategy,
    ) -> Result<Vec<DocumentChunk>> {
        // Delegate to text processor for now
        let text_processor = TextProcessor::new();
        text_processor.extract_chunks(doc, strategy).await
    }
    
    async fn extract_page_range(
        &self,
        data: Vec<u8>,
        _format: DocumentFormat,
        start_page: usize,
        end_page: usize,
    ) -> Result<String> {
        use lopdf::Document as PdfDocument;
        
        // Parse PDF document
        let pdf = PdfDocument::load_mem(&data)
            .map_err(|e| anyhow!("Failed to parse PDF: {}", e))?;
        
        let pages = pdf.get_pages();
        let total_pages = pages.len();
        
        // Validate page range
        if start_page == 0 || start_page > total_pages {
            return Err(anyhow!("Invalid start page: {}. PDF has {} pages", start_page, total_pages));
        }
        if end_page > total_pages {
            return Err(anyhow!("Invalid end page: {}. PDF has {} pages", end_page, total_pages));
        }
        if start_page > end_page {
            return Err(anyhow!("Start page {} is greater than end page {}", start_page, end_page));
        }
        
        let mut extracted_text = String::new();
        
        // Extract text from specified page range (1-indexed)
        for page_num in start_page..=end_page {
            if let Some(page_id) = pages.iter().nth(page_num - 1) {
                match pdf.extract_text(&[*page_id.0]) {
                    Ok(text) => {
                        if !extracted_text.is_empty() {
                            extracted_text.push_str("\n\n");
                        }
                        extracted_text.push_str(&format!("--- Page {} ---\n", page_num));
                        extracted_text.push_str(&text);
                    }
                    Err(e) => {
                        warn!("Failed to extract text from page {}: {}", page_num, e);
                    }
                }
            }
        }
        
        if extracted_text.is_empty() {
            Ok(format!("[No extractable text found in pages {}-{}]", start_page, end_page))
        } else {
            Ok(extracted_text)
        }
    }
    
    fn supported_formats(&self) -> Vec<DocumentFormat> {
        vec![DocumentFormat::Pdf]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_text_processor() {
        let processor = TextProcessor::new();
        let text = "This is a test document with some content.".as_bytes().to_vec();
        
        let doc = processor.process(text, DocumentFormat::Txt).await.unwrap();
        assert_eq!(doc.format, DocumentFormat::Txt);
        assert_eq!(doc.metadata.word_count, 8);
        
        // Test chunking
        let chunks = processor.extract_chunks(
            &doc,
            ChunkingStrategy::FixedSize { size: 20, overlap: 5 }
        ).await.unwrap();
        
        assert!(chunks.len() > 1);
        assert_eq!(chunks[0].chunk_index, 0);
    }
    
    #[tokio::test]
    async fn test_markdown_processor() {
        let processor = MarkdownProcessor::new();
        let markdown = r#"# Title

This is a paragraph.

## Subtitle

Another paragraph with a [link](https://example.com).

### Sub-subtitle

Final content."#.as_bytes().to_vec();
        
        let doc = processor.process(markdown, DocumentFormat::Markdown).await.unwrap();
        assert_eq!(doc.format, DocumentFormat::Markdown);
        assert_eq!(doc.structured_content.sections.len(), 3);
        assert_eq!(doc.structured_content.links.len(), 1);
        
        // Test semantic chunking
        let chunks = processor.extract_chunks(&doc, ChunkingStrategy::Semantic).await.unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].chunk_type, ChunkType::Heading { level: 1 });
    }
    
    #[tokio::test]
    async fn test_unified_processor() {
        let processor = UnifiedDocumentProcessor::new();
        
        // Test text processing
        let text = "Simple text content".as_bytes().to_vec();
        let doc = processor.process(text, DocumentFormat::Txt).await.unwrap();
        assert_eq!(doc.format, DocumentFormat::Txt);
        
        // Test markdown processing
        let markdown = "# Markdown\n\nContent".as_bytes().to_vec();
        let doc = processor.process(markdown, DocumentFormat::Markdown).await.unwrap();
        assert_eq!(doc.format, DocumentFormat::Markdown);
    }
    
    #[test]
    fn test_format_detection() {
        assert_eq!(DocumentFormat::from_extension("pdf"), Some(DocumentFormat::Pdf));
        assert_eq!(DocumentFormat::from_extension("TXT"), Some(DocumentFormat::Txt));
        assert_eq!(DocumentFormat::from_extension("md"), Some(DocumentFormat::Markdown));
        assert_eq!(DocumentFormat::from_extension("unknown"), None);
    }
}