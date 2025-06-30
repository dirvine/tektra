use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_id: String,
    pub project_id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: ChunkMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkMetadata {
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub chunk_type: ChunkType,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkType {
    Text,
    Code,
    Heading,
    List,
    Table,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: DocumentChunk,
    pub similarity_score: f32,
    pub context_chunks: Vec<DocumentChunk>,
}

/// Simple in-memory vector database for document chunks
/// In production, this would be replaced with a proper vector database like Qdrant, Weaviate, or Pinecone
pub struct VectorDB {
    chunks: Arc<Mutex<HashMap<String, DocumentChunk>>>,
    project_index: Arc<Mutex<HashMap<String, Vec<String>>>>, // project_id -> chunk_ids
    document_index: Arc<Mutex<HashMap<String, Vec<String>>>>, // document_id -> chunk_ids
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            chunks: Arc::new(Mutex::new(HashMap::new())),
            project_index: Arc::new(Mutex::new(HashMap::new())),
            document_index: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Add a document chunk to the vector database
    pub async fn add_chunk(&self, chunk: DocumentChunk) -> Result<()> {
        let chunk_id = chunk.id.clone();
        let project_id = chunk.project_id.clone();
        let document_id = chunk.document_id.clone();

        // Store the chunk
        {
            let mut chunks = self.chunks.lock().unwrap();
            chunks.insert(chunk_id.clone(), chunk);
        }

        // Update project index
        {
            let mut project_index = self.project_index.lock().unwrap();
            project_index
                .entry(project_id)
                .or_insert_with(Vec::new)
                .push(chunk_id.clone());
        }

        // Update document index
        {
            let mut document_index = self.document_index.lock().unwrap();
            document_index
                .entry(document_id)
                .or_insert_with(Vec::new)
                .push(chunk_id);
        }

        info!("Added chunk to vector database");
        Ok(())
    }

    /// Search for similar chunks using cosine similarity
    pub async fn search(
        &self,
        query_embedding: Vec<f32>,
        project_id: Option<String>,
        limit: usize,
        min_similarity: f32,
    ) -> Result<Vec<SearchResult>> {
        let chunks = self.chunks.lock().unwrap();
        let mut results = Vec::new();

        // Get chunks to search
        let chunk_ids_to_search = if let Some(project_id) = project_id {
            let project_index = self.project_index.lock().unwrap();
            project_index
                .get(&project_id)
                .cloned()
                .unwrap_or_default()
        } else {
            chunks.keys().cloned().collect()
        };

        // Calculate similarities
        for chunk_id in chunk_ids_to_search {
            if let Some(chunk) = chunks.get(&chunk_id) {
                let similarity = cosine_similarity(&query_embedding, &chunk.embedding);
                
                if similarity >= min_similarity {
                    let context_chunks = self.get_context_chunks(&chunk.document_id, chunk.metadata.chunk_index);
                    
                    results.push(SearchResult {
                        chunk: chunk.clone(),
                        similarity_score: similarity,
                        context_chunks,
                    });
                }
            }
        }

        // Sort by similarity score (descending)
        results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
        
        // Limit results
        results.truncate(limit);

        info!("Found {} similar chunks", results.len());
        Ok(results)
    }

    /// Get chunks for a specific document
    pub async fn get_document_chunks(&self, document_id: &str) -> Result<Vec<DocumentChunk>> {
        let chunks = self.chunks.lock().unwrap();
        let document_index = self.document_index.lock().unwrap();
        
        let chunk_ids = document_index
            .get(document_id)
            .cloned()
            .unwrap_or_default();
        
        let mut document_chunks: Vec<DocumentChunk> = chunk_ids
            .iter()
            .filter_map(|id| chunks.get(id).cloned())
            .collect();
        
        // Sort by chunk index
        document_chunks.sort_by_key(|chunk| chunk.metadata.chunk_index);
        
        Ok(document_chunks)
    }

    /// Get context chunks around a specific chunk
    fn get_context_chunks(&self, document_id: &str, chunk_index: usize) -> Vec<DocumentChunk> {
        let chunks = self.chunks.lock().unwrap();
        let document_index = self.document_index.lock().unwrap();
        
        let chunk_ids = document_index
            .get(document_id)
            .cloned()
            .unwrap_or_default();
        
        let mut context_chunks = Vec::new();
        
        // Get chunks within +/- 2 positions
        for id in chunk_ids {
            if let Some(chunk) = chunks.get(&id) {
                let index_diff = if chunk.metadata.chunk_index > chunk_index {
                    chunk.metadata.chunk_index - chunk_index
                } else {
                    chunk_index - chunk.metadata.chunk_index
                };
                
                if index_diff <= 2 && chunk.metadata.chunk_index != chunk_index {
                    context_chunks.push(chunk.clone());
                }
            }
        }
        
        // Sort by chunk index
        context_chunks.sort_by_key(|chunk| chunk.metadata.chunk_index);
        context_chunks
    }

    /// Remove all chunks for a document
    pub async fn remove_document(&self, document_id: &str) -> Result<()> {
        let chunk_ids = {
            let document_index = self.document_index.lock().unwrap();
            document_index.get(document_id).cloned().unwrap_or_default()
        };

        // Remove chunks
        {
            let mut chunks = self.chunks.lock().unwrap();
            for chunk_id in &chunk_ids {
                chunks.remove(chunk_id);
            }
        }

        // Remove from document index
        {
            let mut document_index = self.document_index.lock().unwrap();
            document_index.remove(document_id);
        }

        // Remove from project index
        {
            let mut project_index = self.project_index.lock().unwrap();
            for project_chunks in project_index.values_mut() {
                project_chunks.retain(|id| !chunk_ids.contains(id));
            }
        }

        info!("Removed {} chunks for document {}", chunk_ids.len(), document_id);
        Ok(())
    }

    /// Get database statistics
    pub async fn get_stats(&self) -> HashMap<String, serde_json::Value> {
        let chunks = self.chunks.lock().unwrap();
        let project_index = self.project_index.lock().unwrap();
        let document_index = self.document_index.lock().unwrap();
        
        let mut stats = HashMap::new();
        stats.insert("total_chunks".to_string(), serde_json::Value::Number(chunks.len().into()));
        stats.insert("total_projects".to_string(), serde_json::Value::Number(project_index.len().into()));
        stats.insert("total_documents".to_string(), serde_json::Value::Number(document_index.len().into()));
        
        // Memory usage estimate (rough)
        let memory_mb = chunks.len() * 1024 / 1_000_000; // Very rough estimate
        stats.insert("estimated_memory_mb".to_string(), serde_json::Value::Number(memory_mb.into()));
        
        stats
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        warn!("Vector dimension mismatch: {} vs {}", a.len(), b.len());
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

/// Simple text chunking for documents
pub fn chunk_text(text: &str, document_id: &str, project_id: &str) -> Vec<DocumentChunk> {
    const CHUNK_SIZE: usize = 512;
    const OVERLAP: usize = 50;
    
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut chunk_index = 0;
    
    while start < text.len() {
        let end = std::cmp::min(start + CHUNK_SIZE, text.len());
        let chunk_text = &text[start..end];
        
        // Skip very short chunks
        if chunk_text.trim().len() < 50 {
            break;
        }
        
        // Generate simple embedding (in production, use a proper embedding model)
        let embedding = generate_simple_embedding(chunk_text);
        
        // Extract keywords (simple approach)
        let keywords = extract_keywords(chunk_text);
        
        // Determine chunk type
        let chunk_type = determine_chunk_type(chunk_text);
        
        let chunk = DocumentChunk {
            id: uuid::Uuid::new_v4().to_string(),
            document_id: document_id.to_string(),
            project_id: project_id.to_string(),
            content: chunk_text.to_string(),
            embedding,
            metadata: ChunkMetadata {
                chunk_index,
                start_char: start,
                end_char: end,
                chunk_type,
                keywords,
            },
        };
        
        chunks.push(chunk);
        
        // Move start position with overlap
        if end == text.len() {
            break;
        }
        start = end - OVERLAP;
        chunk_index += 1;
    }
    
    info!("Created {} chunks from text", chunks.len());
    chunks
}

/// Generate a simple embedding (in production, use a proper embedding model like sentence-transformers)
pub fn generate_simple_embedding(text: &str) -> Vec<f32> {
    const EMBEDDING_DIM: usize = 128;
    
    // Very simple hash-based embedding for demonstration
    // In production, use proper embedding models
    let mut embedding = vec![0.0; EMBEDDING_DIM];
    
    for (i, word) in text.split_whitespace().enumerate() {
        let hash = word.chars().map(|c| c as u32).sum::<u32>();
        let index = (hash as usize) % EMBEDDING_DIM;
        embedding[index] += 1.0 / (i + 1) as f32;
    }
    
    // Normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut embedding {
            *val /= norm;
        }
    }
    
    embedding
}

/// Extract simple keywords from text
fn extract_keywords(text: &str) -> Vec<String> {
    let stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those"];
    
    text.split_whitespace()
        .map(|word| word.to_lowercase().trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|word| word.len() > 3 && !stop_words.contains(&word.as_str()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .take(10)
        .collect()
}

/// Determine chunk type based on content
fn determine_chunk_type(text: &str) -> ChunkType {
    let text = text.trim();
    
    if text.starts_with('#') {
        ChunkType::Heading
    } else if text.starts_with('-') || text.starts_with('*') || text.starts_with("1.") {
        ChunkType::List
    } else if text.starts_with('|') || text.contains('|') {
        ChunkType::Table
    } else if text.contains("```") || text.contains("fn ") || text.contains("class ") || text.contains("def ") {
        ChunkType::Code
    } else {
        ChunkType::Text
    }
}