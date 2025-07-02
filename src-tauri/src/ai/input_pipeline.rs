use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug, warn};

use crate::ai::document_processor::{
    DocumentFormat, ProcessedDocument, DocumentChunk, ChunkingStrategy,
    UnifiedDocumentProcessor,
};
use crate::ai::unified_model_manager::{MultimodalInput as ModelMultimodalInput};
use crate::ai::embeddings::EmbeddingGenerator;
use crate::vector_db::VectorDB;

/// Combined input for RAG (Retrieval-Augmented Generation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedInput {
    pub user_query: String,
    pub document_context: Vec<DocumentContext>,
    pub images: Vec<Vec<u8>>,
    pub audio: Option<Vec<u8>>,
    pub metadata: InputMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentContext {
    pub source: DocumentSource,
    pub content: String,
    pub relevance_score: f32,
    pub chunk_indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentSource {
    File { path: String, format: DocumentFormat },
    Url { url: String },
    Direct { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMetadata {
    pub query_type: QueryType,
    pub max_context_tokens: usize,
    pub include_source_citations: bool,
    pub temperature_override: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryType {
    Question,
    Summary,
    Analysis,
    Translation,
    CodeGeneration,
    Custom(String),
}

/// Configuration for the input pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub chunking_strategy: ChunkingStrategy,
    pub max_chunks_per_document: usize,
    pub similarity_threshold: f32,
    pub context_window_size: usize,
    pub enable_semantic_search: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            chunking_strategy: ChunkingStrategy::default(),
            max_chunks_per_document: 5,
            similarity_threshold: 0.7,
            context_window_size: 8000,
            enable_semantic_search: true,
        }
    }
}

/// Main input pipeline for combining documents with queries
pub struct InputPipeline {
    document_processor: Arc<UnifiedDocumentProcessor>,
    vector_db: Arc<VectorDB>,
    embedding_generator: Arc<Box<dyn EmbeddingGenerator>>,
    config: Arc<RwLock<PipelineConfig>>,
    cache: Arc<RwLock<ProcessingCache>>,
}

/// Cache for processed documents
struct ProcessingCache {
    documents: std::collections::HashMap<String, ProcessedDocument>,
    embeddings: std::collections::HashMap<String, Vec<f32>>,
}

impl InputPipeline {
    pub fn new(
        document_processor: Arc<UnifiedDocumentProcessor>,
        vector_db: Arc<VectorDB>,
        embedding_generator: Arc<Box<dyn EmbeddingGenerator>>,
    ) -> Self {
        Self {
            document_processor,
            vector_db,
            embedding_generator,
            config: Arc::new(RwLock::new(PipelineConfig::default())),
            cache: Arc::new(RwLock::new(ProcessingCache {
                documents: std::collections::HashMap::new(),
                embeddings: std::collections::HashMap::new(),
            })),
        }
    }

    /// Process a combined query with documents
    pub async fn process_combined_query(
        &self,
        user_query: &str,
        document_paths: Vec<&Path>,
        additional_context: Option<Vec<String>>,
        images: Vec<Vec<u8>>,
        audio: Option<Vec<u8>>,
    ) -> Result<CombinedInput> {
        info!("Processing combined query with {} documents", document_paths.len());
        
        let config = self.config.read().await;
        let mut all_contexts = Vec::new();
        
        // Process each document
        for path in document_paths {
            let doc_contexts = self.process_document_for_query(
                path,
                user_query,
                &config,
            ).await?;
            all_contexts.extend(doc_contexts);
        }
        
        // Add additional context if provided
        if let Some(contexts) = additional_context {
            for (idx, context) in contexts.into_iter().enumerate() {
                all_contexts.push(DocumentContext {
                    source: DocumentSource::Direct { id: format!("direct-{}", idx) },
                    content: context,
                    relevance_score: 1.0,
                    chunk_indices: vec![],
                });
            }
        }
        
        // Sort by relevance and limit to context window
        all_contexts.sort_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap());
        let limited_contexts = self.limit_to_context_window(&all_contexts, &config).await?;
        
        let metadata = InputMetadata {
            query_type: self.detect_query_type(user_query),
            max_context_tokens: config.context_window_size,
            include_source_citations: true,
            temperature_override: None,
        };
        
        Ok(CombinedInput {
            user_query: user_query.to_string(),
            document_context: limited_contexts,
            images,
            audio,
            metadata,
        })
    }

    /// Process a single document for a query
    async fn process_document_for_query(
        &self,
        path: &Path,
        query: &str,
        config: &PipelineConfig,
    ) -> Result<Vec<DocumentContext>> {
        debug!("Processing document: {:?}", path);
        
        // Check cache first
        let cache_key = path.to_string_lossy().to_string();
        let processed_doc = {
            let cache = self.cache.read().await;
            cache.documents.get(&cache_key).cloned()
        };
        
        let processed_doc = match processed_doc {
            Some(doc) => doc,
            None => {
                // Process the document
                let doc = self.document_processor.process_file(path).await?;
                
                // Cache it
                let mut cache = self.cache.write().await;
                cache.documents.insert(cache_key.clone(), doc.clone());
                doc
            }
        };
        
        // Extract chunks
        let chunks = self.document_processor
            .extract_chunks(&processed_doc, config.chunking_strategy)
            .await?;
        
        // Find relevant chunks
        let relevant_chunks = if config.enable_semantic_search {
            self.find_relevant_chunks_semantic(query, &chunks, config).await?
        } else {
            self.find_relevant_chunks_keyword(query, &chunks, config)
        };
        
        // Convert to document contexts
        let contexts = relevant_chunks
            .into_iter()
            .map(|(chunk, score)| DocumentContext {
                source: DocumentSource::File {
                    path: path.to_string_lossy().to_string(),
                    format: processed_doc.format,
                },
                content: chunk.content,
                relevance_score: score,
                chunk_indices: vec![chunk.chunk_index],
            })
            .collect();
        
        Ok(contexts)
    }

    /// Find relevant chunks using semantic search
    async fn find_relevant_chunks_semantic(
        &self,
        query: &str,
        chunks: &[DocumentChunk],
        config: &PipelineConfig,
    ) -> Result<Vec<(DocumentChunk, f32)>> {
        // Generate query embedding
        let query_embedding = self.embedding_generator.generate_embedding(query).await?;
        
        // Generate embeddings for chunks (with caching)
        let mut chunk_embeddings = Vec::new();
        for chunk in chunks {
            let embedding = self.get_or_generate_embedding(&chunk.content).await?;
            chunk_embeddings.push(embedding);
        }
        
        // Calculate similarities
        let mut scored_chunks = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            let similarity = cosine_similarity(&query_embedding, &chunk_embeddings[idx]);
            if similarity >= config.similarity_threshold {
                scored_chunks.push((chunk.clone(), similarity));
            }
        }
        
        // Sort by similarity and limit
        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_chunks.truncate(config.max_chunks_per_document);
        
        Ok(scored_chunks)
    }

    /// Find relevant chunks using keyword matching
    fn find_relevant_chunks_keyword(
        &self,
        query: &str,
        chunks: &[DocumentChunk],
        config: &PipelineConfig,
    ) -> Vec<(DocumentChunk, f32)> {
        let query_words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        let mut scored_chunks = Vec::new();
        
        for chunk in chunks {
            let chunk_lower = chunk.content.to_lowercase();
            let mut score = 0.0;
            let mut _matched_words = 0;
            
            for word in &query_words {
                if chunk_lower.contains(word) {
                    _matched_words += 1;
                    score += 1.0 / query_words.len() as f32;
                }
            }
            
            if score >= config.similarity_threshold {
                scored_chunks.push((chunk.clone(), score));
            }
        }
        
        scored_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_chunks.truncate(config.max_chunks_per_document);
        
        scored_chunks
    }

    /// Get or generate embedding with caching
    async fn get_or_generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let hash = md5::compute(text);
        let cache_key = format!("{:x}", hash);
        
        // Check cache
        {
            let cache = self.cache.read().await;
            if let Some(embedding) = cache.embeddings.get(&cache_key) {
                return Ok(embedding.clone());
            }
        }
        
        // Generate new embedding
        let embedding = self.embedding_generator.generate_embedding(text).await?;
        
        // Cache it
        {
            let mut cache = self.cache.write().await;
            cache.embeddings.insert(cache_key, embedding.clone());
        }
        
        Ok(embedding)
    }

    /// Limit contexts to fit within token window
    async fn limit_to_context_window(
        &self,
        contexts: &[DocumentContext],
        config: &PipelineConfig,
    ) -> Result<Vec<DocumentContext>> {
        let mut selected = Vec::new();
        let mut total_tokens = 0;
        
        for context in contexts {
            let estimated_tokens = self.estimate_tokens(&context.content);
            if total_tokens + estimated_tokens <= config.context_window_size {
                selected.push(context.clone());
                total_tokens += estimated_tokens;
            } else {
                info!("Reached context window limit at {} tokens", total_tokens);
                break;
            }
        }
        
        Ok(selected)
    }

    /// Estimate token count for text
    fn estimate_tokens(&self, text: &str) -> usize {
        // Rough approximation: 4 characters per token
        (text.len() / 4).max(1)
    }

    /// Detect query type from user input
    fn detect_query_type(&self, query: &str) -> QueryType {
        let query_lower = query.to_lowercase();
        
        if query_lower.starts_with("summarize") || query_lower.contains("summary") {
            QueryType::Summary
        } else if query_lower.starts_with("analyze") || query_lower.contains("analysis") {
            QueryType::Analysis
        } else if query_lower.starts_with("translate") {
            QueryType::Translation
        } else if query_lower.contains("write code") || query_lower.contains("generate code") {
            QueryType::CodeGeneration
        } else if query_lower.contains('?') {
            QueryType::Question
        } else {
            QueryType::Custom("General".to_string())
        }
    }

    /// Format combined input for model consumption
    pub fn format_for_model(&self, input: &CombinedInput) -> String {
        let mut formatted = String::new();
        
        // Add document context
        if !input.document_context.is_empty() {
            formatted.push_str("### Context Documents:\n\n");
            
            for (idx, context) in input.document_context.iter().enumerate() {
                formatted.push_str(&format!("**Document {} (relevance: {:.2})**\n", idx + 1, context.relevance_score));
                
                // Add source info if citations enabled
                if input.metadata.include_source_citations {
                    match &context.source {
                        DocumentSource::File { path, format } => {
                            formatted.push_str(&format!("Source: {} ({:?})\n", path, format));
                        }
                        DocumentSource::Url { url } => {
                            formatted.push_str(&format!("Source: {}\n", url));
                        }
                        DocumentSource::Direct { id } => {
                            formatted.push_str(&format!("Source: Direct input {}\n", id));
                        }
                    }
                }
                
                formatted.push_str(&format!("\n{}\n\n", context.content));
            }
            
            formatted.push_str("---\n\n");
        }
        
        // Add query with type indicator
        formatted.push_str(&format!("### {} Query:\n\n{}\n", 
            match input.metadata.query_type {
                QueryType::Question => "Question",
                QueryType::Summary => "Summarization",
                QueryType::Analysis => "Analysis",
                QueryType::Translation => "Translation",
                QueryType::CodeGeneration => "Code Generation",
                QueryType::Custom(ref s) => s,
            },
            input.user_query
        ));
        
        // Add multimodal indicators
        if !input.images.is_empty() {
            formatted.push_str(&format!("\n[{} image(s) attached]\n", input.images.len()));
        }
        
        if input.audio.is_some() {
            formatted.push_str("\n[Audio attached]\n");
        }
        
        formatted
    }

    /// Convert to model multimodal input
    pub fn to_model_input(&self, input: &CombinedInput) -> ModelMultimodalInput {
        let formatted_text = self.format_for_model(input);
        
        ModelMultimodalInput {
            text: Some(formatted_text),
            images: input.images.clone(),
            audio: input.audio.clone(),
            video: None,
        }
    }

    /// Update pipeline configuration
    pub async fn update_config(&self, config: PipelineConfig) -> Result<()> {
        *self.config.write().await = config;
        Ok(())
    }

    /// Clear processing cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.documents.clear();
        cache.embeddings.clear();
        info!("Cleared processing cache");
        Ok(())
    }

    /// Get pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        let cache = self.cache.read().await;
        PipelineStats {
            cached_documents: cache.documents.len(),
            cached_embeddings: cache.embeddings.len(),
            estimated_memory_mb: (cache.documents.len() * 1024 + cache.embeddings.len() * 512) / 1_000_000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineStats {
    pub cached_documents: usize,
    pub cached_embeddings: usize,
    pub estimated_memory_mb: usize,
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::embeddings::SimpleEmbeddingGenerator;

    #[tokio::test]
    async fn test_query_type_detection() {
        let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
        let vector_db = Arc::new(VectorDB::new());
        let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
            Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
        
        let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
        
        assert!(matches!(pipeline.detect_query_type("What is the capital of France?"), QueryType::Question));
        assert!(matches!(pipeline.detect_query_type("Summarize this document"), QueryType::Summary));
        assert!(matches!(pipeline.detect_query_type("Analyze the trends in this data"), QueryType::Analysis));
        assert!(matches!(pipeline.detect_query_type("Translate this to Spanish"), QueryType::Translation));
        assert!(matches!(pipeline.detect_query_type("Write code to sort an array"), QueryType::CodeGeneration));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);
        
        let c = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &c), 0.0);
        
        let d = vec![0.707, 0.707, 0.0];
        assert!((cosine_similarity(&a, &d) - 0.707).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_format_for_model() {
        let doc_processor = Arc::new(UnifiedDocumentProcessor::new());
        let vector_db = Arc::new(VectorDB::new());
        let embedding_gen: Arc<Box<dyn EmbeddingGenerator>> = 
            Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
        
        let pipeline = InputPipeline::new(doc_processor, vector_db, embedding_gen);
        
        let input = CombinedInput {
            user_query: "What is machine learning?".to_string(),
            document_context: vec![
                DocumentContext {
                    source: DocumentSource::File {
                        path: "ml_intro.txt".to_string(),
                        format: DocumentFormat::Txt,
                    },
                    content: "Machine learning is a subset of AI.".to_string(),
                    relevance_score: 0.95,
                    chunk_indices: vec![0],
                }
            ],
            images: vec![],
            audio: None,
            metadata: InputMetadata {
                query_type: QueryType::Question,
                max_context_tokens: 1000,
                include_source_citations: true,
                temperature_override: None,
            },
        };
        
        let formatted = pipeline.format_for_model(&input);
        assert!(formatted.contains("### Context Documents:"));
        assert!(formatted.contains("relevance: 0.95"));
        assert!(formatted.contains("Source: ml_intro.txt"));
        assert!(formatted.contains("### Question Query:"));
        assert!(formatted.contains("What is machine learning?"));
    }
}