use anyhow::Result;
use tracing::{info, warn};
use std::collections::HashMap;

/// Multimodal fusion strategies for combining different input modalities
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// Simple concatenation of feature vectors
    Concatenation,
    /// Weighted average of feature vectors (requires same dimensions)
    WeightedAverage(Vec<f32>),
    /// Cross-attention between modalities
    CrossAttention,
    /// Hierarchical fusion with multiple stages
    Hierarchical,
}

/// Multimodal input container for different types of data
#[derive(Debug)]
pub struct MultimodalData {
    pub text_features: Option<Vec<f32>>,
    pub image_features: Option<Vec<f32>>,
    pub audio_features: Option<Vec<f32>>,
    pub video_features: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
}

/// Configuration for multimodal fusion
#[derive(Debug, Clone)]
pub struct FusionConfig {
    pub strategy: FusionStrategy,
    pub target_dimension: Option<usize>,
    pub normalize_inputs: bool,
    pub enable_cross_modal_attention: bool,
    pub attention_heads: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::Concatenation,
            target_dimension: None,
            normalize_inputs: true,
            enable_cross_modal_attention: false,
            attention_heads: 8,
        }
    }
}

/// Multimodal fusion processor
pub struct MultimodalFusion {
    config: FusionConfig,
}

impl MultimodalFusion {
    pub fn new(config: FusionConfig) -> Self {
        info!("Initializing multimodal fusion with strategy: {:?}", config.strategy);
        Self { config }
    }

    pub fn new_default() -> Self {
        Self::new(FusionConfig::default())
    }

    /// Fuse multiple modalities into a single representation
    pub fn fuse(&self, data: &MultimodalData) -> Result<Vec<f32>> {
        info!("Starting multimodal fusion");
        
        // Collect available modalities
        let mut modalities = Vec::new();
        let mut modality_names = Vec::new();
        
        if let Some(ref text_features) = data.text_features {
            modalities.push(text_features.clone());
            modality_names.push("text");
        }
        
        if let Some(ref image_features) = data.image_features {
            modalities.push(image_features.clone());
            modality_names.push("image");
        }
        
        if let Some(ref audio_features) = data.audio_features {
            modalities.push(audio_features.clone());
            modality_names.push("audio");
        }
        
        if let Some(ref video_features) = data.video_features {
            modalities.push(video_features.clone());
            modality_names.push("video");
        }
        
        if modalities.is_empty() {
            return Err(anyhow::anyhow!("No input modalities provided"));
        }
        
        info!("Fusing {} modalities: {:?}", modalities.len(), modality_names);
        
        // Normalize inputs if requested
        let normalized_modalities = if self.config.normalize_inputs {
            modalities.into_iter()
                .map(|features| self.normalize_features(&features))
                .collect::<Result<Vec<_>>>()?
        } else {
            modalities
        };
        
        // Apply fusion strategy
        let fused_features = match &self.config.strategy {
            FusionStrategy::Concatenation => {
                self.concatenate_features(&normalized_modalities)
            }
            FusionStrategy::WeightedAverage(weights) => {
                self.weighted_average_features(&normalized_modalities, weights)
            }
            FusionStrategy::CrossAttention => {
                self.cross_attention_fusion(&normalized_modalities)
            }
            FusionStrategy::Hierarchical => {
                self.hierarchical_fusion(&normalized_modalities)
            }
        }?;
        
        // Apply target dimension projection if specified
        let final_features = if let Some(target_dim) = self.config.target_dimension {
            self.project_to_dimension(&fused_features, target_dim)?
        } else {
            fused_features
        };
        
        info!("Fusion complete. Output dimension: {}", final_features.len());
        Ok(final_features)
    }

    /// Simple concatenation of all feature vectors
    fn concatenate_features(&self, modalities: &[Vec<f32>]) -> Result<Vec<f32>> {
        info!("Applying concatenation fusion");
        
        let total_size: usize = modalities.iter().map(|m| m.len()).sum();
        let mut concatenated = Vec::with_capacity(total_size);
        
        for modality in modalities {
            concatenated.extend_from_slice(modality);
        }
        
        Ok(concatenated)
    }

    /// Weighted average of feature vectors (requires same dimensions)
    fn weighted_average_features(&self, modalities: &[Vec<f32>], weights: &[f32]) -> Result<Vec<f32>> {
        info!("Applying weighted average fusion");
        
        if modalities.is_empty() {
            return Err(anyhow::anyhow!("No modalities to average"));
        }
        
        if weights.len() != modalities.len() {
            return Err(anyhow::anyhow!(
                "Weight count ({}) doesn't match modality count ({})", 
                weights.len(), modalities.len()
            ));
        }
        
        // Check that all modalities have the same dimension
        let first_dim = modalities[0].len();
        for (i, modality) in modalities.iter().enumerate() {
            if modality.len() != first_dim {
                return Err(anyhow::anyhow!(
                    "Modality {} has dimension {} but expected {}", 
                    i, modality.len(), first_dim
                ));
            }
        }
        
        // Normalize weights
        let weight_sum: f32 = weights.iter().sum();
        if weight_sum == 0.0 {
            return Err(anyhow::anyhow!("Sum of weights is zero"));
        }
        
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / weight_sum).collect();
        
        // Compute weighted average
        let mut averaged = vec![0.0; first_dim];
        for (weight, modality) in normalized_weights.iter().zip(modalities.iter()) {
            for (i, &value) in modality.iter().enumerate() {
                averaged[i] += weight * value;
            }
        }
        
        Ok(averaged)
    }

    /// Cross-attention fusion between modalities
    fn cross_attention_fusion(&self, modalities: &[Vec<f32>]) -> Result<Vec<f32>> {
        info!("Applying cross-attention fusion");
        
        if modalities.len() < 2 {
            warn!("Cross-attention requires at least 2 modalities, falling back to concatenation");
            return self.concatenate_features(modalities);
        }
        
        // For a simplified cross-attention implementation:
        // 1. Use the first modality as query
        // 2. Use other modalities as keys/values
        // 3. Compute attention weights and weighted sum
        
        let query = &modalities[0];
        let mut attended_features = query.clone();
        
        for other_modality in &modalities[1..] {
            // Simplified attention: dot product similarity
            let attention_weight = self.compute_attention_weight(query, other_modality)?;
            
            // Add weighted contribution
            for (i, &value) in other_modality.iter().enumerate() {
                if i < attended_features.len() {
                    attended_features[i] += attention_weight * value;
                }
            }
        }
        
        Ok(attended_features)
    }

    /// Hierarchical fusion with multiple stages
    fn hierarchical_fusion(&self, modalities: &[Vec<f32>]) -> Result<Vec<f32>> {
        info!("Applying hierarchical fusion");
        
        if modalities.len() <= 2 {
            // For small numbers of modalities, fall back to concatenation
            return self.concatenate_features(modalities);
        }
        
        // Stage 1: Group similar modalities
        let mut current_representations = modalities.to_vec();
        
        // Stage 2: Iteratively fuse pairs until we have a single representation
        while current_representations.len() > 1 {
            let mut next_stage = Vec::new();
            
            // Fuse pairs of representations
            for chunk in current_representations.chunks(2) {
                if chunk.len() == 2 {
                    // Fuse two representations using weighted average
                    let weights = vec![0.5, 0.5];
                    let fused = self.weighted_average_features(&chunk.to_vec(), &weights)?;
                    next_stage.push(fused);
                } else {
                    // Odd number of representations - keep the last one as is
                    next_stage.push(chunk[0].clone());
                }
            }
            
            current_representations = next_stage;
        }
        
        Ok(current_representations.into_iter().next().unwrap())
    }

    /// Normalize feature vector to unit length
    fn normalize_features(&self, features: &[f32]) -> Result<Vec<f32>> {
        let magnitude: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude == 0.0 {
            warn!("Feature vector has zero magnitude, returning as-is");
            return Ok(features.to_vec());
        }
        
        Ok(features.iter().map(|x| x / magnitude).collect())
    }

    /// Project features to target dimension using simple truncation or padding
    fn project_to_dimension(&self, features: &[f32], target_dim: usize) -> Result<Vec<f32>> {
        if features.len() == target_dim {
            return Ok(features.to_vec());
        }
        
        if features.len() > target_dim {
            // Truncate to target dimension
            info!("Truncating features from {} to {}", features.len(), target_dim);
            Ok(features[..target_dim].to_vec())
        } else {
            // Pad with zeros to reach target dimension
            info!("Padding features from {} to {}", features.len(), target_dim);
            let mut padded = features.to_vec();
            padded.resize(target_dim, 0.0);
            Ok(padded)
        }
    }

    /// Compute simplified attention weight between two feature vectors
    fn compute_attention_weight(&self, query: &[f32], key: &[f32]) -> Result<f32> {
        let min_len = query.len().min(key.len());
        if min_len == 0 {
            return Ok(0.0);
        }
        
        // Compute dot product for the overlapping dimensions
        let dot_product: f32 = query.iter()
            .zip(key.iter())
            .take(min_len)
            .map(|(q, k)| q * k)
            .sum();
        
        // Normalize by the length
        let attention_weight = dot_product / min_len as f32;
        
        // Apply softmax-like transformation
        Ok(attention_weight.exp() / (1.0 + attention_weight.exp()))
    }

    /// Create multimodal data from individual components
    pub fn create_multimodal_data(
        text_features: Option<Vec<f32>>,
        image_features: Option<Vec<f32>>,
        audio_features: Option<Vec<f32>>,
        video_features: Option<Vec<f32>>,
    ) -> MultimodalData {
        MultimodalData {
            text_features,
            image_features,
            audio_features,
            video_features,
            metadata: HashMap::new(),
        }
    }

    /// Get information about the fusion configuration
    pub fn get_info(&self) -> String {
        format!(
            "Fusion Strategy: {:?}\nTarget Dimension: {:?}\nNormalize Inputs: {}\nCross-modal Attention: {}",
            self.config.strategy,
            self.config.target_dimension,
            self.config.normalize_inputs,
            self.config.enable_cross_modal_attention
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concatenation_fusion() {
        let fusion = MultimodalFusion::new_default();
        
        let data = MultimodalData {
            text_features: Some(vec![1.0, 2.0, 3.0]),
            image_features: Some(vec![4.0, 5.0]),
            audio_features: None,
            video_features: None,
            metadata: HashMap::new(),
        };
        
        let result = fusion.fuse(&data);
        assert!(result.is_ok());
        
        let fused = result.unwrap();
        assert_eq!(fused.len(), 5);
        // Note: actual values will be normalized
    }

    #[test]
    fn test_weighted_average_fusion() {
        let config = FusionConfig {
            strategy: FusionStrategy::WeightedAverage(vec![0.3, 0.7]),
            normalize_inputs: false,
            ..Default::default()
        };
        
        let fusion = MultimodalFusion::new(config);
        
        let data = MultimodalData {
            text_features: Some(vec![1.0, 2.0, 3.0]),
            image_features: Some(vec![4.0, 5.0, 6.0]),
            audio_features: None,
            video_features: None,
            metadata: HashMap::new(),
        };
        
        let result = fusion.fuse(&data);
        assert!(result.is_ok());
        
        let fused = result.unwrap();
        assert_eq!(fused.len(), 3);
        // Should be weighted average: [0.3*1 + 0.7*4, 0.3*2 + 0.7*5, 0.3*3 + 0.7*6]
        assert!((fused[0] - 3.1).abs() < 0.001);
        assert!((fused[1] - 4.1).abs() < 0.001);
        assert!((fused[2] - 5.1).abs() < 0.001);
    }

    #[test]
    fn test_empty_modalities() {
        let fusion = MultimodalFusion::new_default();
        
        let data = MultimodalData {
            text_features: None,
            image_features: None,
            audio_features: None,
            video_features: None,
            metadata: HashMap::new(),
        };
        
        let result = fusion.fuse(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_dimension_projection() {
        let config = FusionConfig {
            target_dimension: Some(3),
            ..Default::default()
        };
        
        let fusion = MultimodalFusion::new(config);
        
        // Test truncation
        let long_features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let projected = fusion.project_to_dimension(&long_features, 3).unwrap();
        assert_eq!(projected.len(), 3);
        assert_eq!(projected, vec![1.0, 2.0, 3.0]);
        
        // Test padding
        let short_features = vec![1.0, 2.0];
        let projected = fusion.project_to_dimension(&short_features, 5).unwrap();
        assert_eq!(projected.len(), 5);
        assert_eq!(projected, vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_feature_normalization() {
        let fusion = MultimodalFusion::new_default();
        
        let features = vec![3.0, 4.0]; // Magnitude = 5.0
        let normalized = fusion.normalize_features(&features).unwrap();
        
        assert_eq!(normalized.len(), 2);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
        
        // Check that the normalized vector has unit magnitude
        let magnitude: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }
}