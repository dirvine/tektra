use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Trait for generating embeddings from text
#[async_trait]
pub trait EmbeddingGenerator: Send + Sync {
    /// Generate embedding vector for text
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Generate embeddings for multiple texts (batch processing)
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    
    /// Get embedding dimension
    fn embedding_dimension(&self) -> usize;
    
    /// Get model name
    fn model_name(&self) -> &str;
}

/// Configuration for embedding models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    pub model_name: String,
    pub dimension: usize,
    pub max_batch_size: usize,
    pub normalize: bool,
}

/// Simple embedding generator for testing
pub struct SimpleEmbeddingGenerator {
    config: EmbeddingConfig,
}

impl SimpleEmbeddingGenerator {
    pub fn new() -> Self {
        Self {
            config: EmbeddingConfig {
                model_name: "simple-hash".to_string(),
                dimension: 128,
                max_batch_size: 32,
                normalize: true,
            },
        }
    }
}

#[async_trait]
impl EmbeddingGenerator for SimpleEmbeddingGenerator {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Simple hash-based embedding for testing
        let mut embedding = vec![0.0; self.config.dimension];
        
        for (i, word) in text.split_whitespace().enumerate() {
            let hash = word.chars().map(|c| c as u32).sum::<u32>();
            let index = (hash as usize) % self.config.dimension;
            embedding[index] += 1.0 / (i + 1) as f32;
        }
        
        // Normalize if configured
        if self.config.normalize {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut embedding {
                    *val /= norm;
                }
            }
        }
        
        Ok(embedding)
    }
    
    async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.generate_embedding(text).await?);
        }
        Ok(embeddings)
    }
    
    fn embedding_dimension(&self) -> usize {
        self.config.dimension
    }
    
    fn model_name(&self) -> &str {
        &self.config.model_name
    }
}

// Legacy OllamaEmbeddingGenerator removed - will be replaced with mistral.rs implementation

/// Embedding manager that can switch between different generators
pub struct EmbeddingManager {
    generators: Arc<RwLock<std::collections::HashMap<String, Arc<Box<dyn EmbeddingGenerator>>>>>,
    active_generator: Arc<RwLock<Option<String>>>,
}

impl EmbeddingManager {
    pub fn new() -> Self {
        let mut generators = std::collections::HashMap::new();
        
        // Add default simple generator
        let simple_gen: Arc<Box<dyn EmbeddingGenerator>> = 
            Arc::new(Box::new(SimpleEmbeddingGenerator::new()));
        generators.insert("simple".to_string(), simple_gen);
        
        Self {
            generators: Arc::new(RwLock::new(generators)),
            active_generator: Arc::new(RwLock::new(Some("simple".to_string()))),
        }
    }
    
    /// Register a new embedding generator
    pub async fn register_generator(&self, name: String, generator: Arc<Box<dyn EmbeddingGenerator>>) {
        let mut generators = self.generators.write().await;
        generators.insert(name.clone(), generator);
        info!("Registered embedding generator: {}", name);
    }
    
    /// Set the active generator
    pub async fn set_active(&self, name: &str) -> Result<()> {
        let generators = self.generators.read().await;
        if !generators.contains_key(name) {
            return Err(anyhow::anyhow!("Embedding generator '{}' not found", name));
        }
        
        *self.active_generator.write().await = Some(name.to_string());
        info!("Set active embedding generator to: {}", name);
        Ok(())
    }
    
    /// Get the active generator
    pub async fn get_active(&self) -> Result<Arc<Box<dyn EmbeddingGenerator>>> {
        let active_name = self.active_generator.read().await.clone()
            .ok_or_else(|| anyhow::anyhow!("No active embedding generator"))?;
        
        let generators = self.generators.read().await;
        generators.get(&active_name)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Active generator '{}' not found", active_name))
    }
    
    /// Generate embedding using active generator
    pub async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        let generator = self.get_active().await?;
        generator.generate_embedding(text).await
    }
    
    /// Generate embeddings using active generator
    pub async fn generate_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let generator = self.get_active().await?;
        generator.generate_embeddings(texts).await
    }
    
    /// List available generators
    pub async fn list_generators(&self) -> Vec<(String, String, usize)> {
        let generators = self.generators.read().await;
        generators.iter().map(|(name, gen)| {
            (name.clone(), gen.model_name().to_string(), gen.embedding_dimension())
        }).collect()
    }
}

/// Utility functions for embeddings
pub mod utils {
    
    
    /// Calculate cosine similarity between two embeddings
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    
    /// Calculate euclidean distance between two embeddings
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::MAX;
        }
        
        a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    /// Find k nearest neighbors
    pub fn find_k_nearest(
        query: &[f32],
        embeddings: &[Vec<f32>],
        k: usize,
        use_cosine: bool,
    ) -> Vec<(usize, f32)> {
        let mut distances: Vec<(usize, f32)> = embeddings
            .iter()
            .enumerate()
            .map(|(idx, emb)| {
                let distance = if use_cosine {
                    1.0 - cosine_similarity(query, emb) // Convert similarity to distance
                } else {
                    euclidean_distance(query, emb)
                };
                (idx, distance)
            })
            .collect();
        
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        
        distances
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_simple_embedding_generator() {
        let generator = SimpleEmbeddingGenerator::new();
        
        let embedding = generator.generate_embedding("test text").await.unwrap();
        assert_eq!(embedding.len(), 128);
        
        // Test normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
    
    #[tokio::test]
    async fn test_batch_embeddings() {
        let generator = SimpleEmbeddingGenerator::new();
        
        let texts = vec![
            "first text".to_string(),
            "second text".to_string(),
            "third text".to_string(),
        ];
        
        let embeddings = generator.generate_embeddings(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), 128);
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(utils::cosine_similarity(&a, &b), 1.0);
        
        let c = vec![0.0, 1.0, 0.0];
        assert_eq!(utils::cosine_similarity(&a, &c), 0.0);
    }
    
    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert_eq!(utils::euclidean_distance(&a, &b), 5.0);
    }
    
    #[test]
    fn test_k_nearest() {
        let query = vec![1.0, 0.0];
        let embeddings = vec![
            vec![1.0, 0.0],     // Distance 0
            vec![0.0, 1.0],     // Distance sqrt(2)
            vec![0.5, 0.5],     // Distance ~0.707
            vec![-1.0, 0.0],    // Distance 2
        ];
        
        let nearest = utils::find_k_nearest(&query, &embeddings, 2, false);
        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 0); // Closest is index 0
        assert_eq!(nearest[1].0, 2); // Second closest is index 2
    }
}