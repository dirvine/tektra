use super::*;
use anyhow::Result;
use std::collections::HashMap;

/// Quantization utilities for optimizing model performance and memory usage
pub struct QuantizationManager {
    supported_formats: HashMap<String, QuantizationConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub name: String,
    pub description: String,
    pub bits_per_weight: f32,
    pub memory_reduction: f32,
    pub quality_impact: QualityImpact,
    pub speed_improvement: f32,
    pub supported_architectures: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityImpact {
    Minimal,
    Low,
    Medium,
    High,
}

impl QuantizationManager {
    pub fn new() -> Self {
        let mut supported_formats = HashMap::new();
        
        // Q4 formats
        supported_formats.insert("Q4_0".to_string(), QuantizationConfig {
            name: "Q4_0".to_string(),
            description: "4-bit quantization with good quality/speed balance".to_string(),
            bits_per_weight: 4.5,
            memory_reduction: 0.75,
            quality_impact: QualityImpact::Low,
            speed_improvement: 2.0,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        supported_formats.insert("Q4_K_S".to_string(), QuantizationConfig {
            name: "Q4_K_S".to_string(),
            description: "4-bit K-quant, small version with better quality".to_string(),
            bits_per_weight: 4.25,
            memory_reduction: 0.73,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.8,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        supported_formats.insert("Q4_K_M".to_string(), QuantizationConfig {
            name: "Q4_K_M".to_string(),
            description: "4-bit K-quant, medium version with excellent quality".to_string(),
            bits_per_weight: 4.85,
            memory_reduction: 0.70,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.7,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        // Q5 formats
        supported_formats.insert("Q5_0".to_string(), QuantizationConfig {
            name: "Q5_0".to_string(),
            description: "5-bit quantization with high quality".to_string(),
            bits_per_weight: 5.5,
            memory_reduction: 0.65,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.5,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        supported_formats.insert("Q5_K_S".to_string(), QuantizationConfig {
            name: "Q5_K_S".to_string(),
            description: "5-bit K-quant, small version".to_string(),
            bits_per_weight: 5.25,
            memory_reduction: 0.63,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.4,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        supported_formats.insert("Q5_K_M".to_string(), QuantizationConfig {
            name: "Q5_K_M".to_string(),
            description: "5-bit K-quant, medium version with excellent quality".to_string(),
            bits_per_weight: 5.85,
            memory_reduction: 0.60,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.3,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        // Q6 and Q8 formats
        supported_formats.insert("Q6_K".to_string(), QuantizationConfig {
            name: "Q6_K".to_string(),
            description: "6-bit K-quant with near-original quality".to_string(),
            bits_per_weight: 6.85,
            memory_reduction: 0.50,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.2,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        supported_formats.insert("Q8_0".to_string(), QuantizationConfig {
            name: "Q8_0".to_string(),
            description: "8-bit quantization with virtually no quality loss".to_string(),
            bits_per_weight: 8.5,
            memory_reduction: 0.35,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.1,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string()],
        });
        
        // Half precision formats
        supported_formats.insert("F16".to_string(), QuantizationConfig {
            name: "F16".to_string(),
            description: "16-bit half precision floating point".to_string(),
            bits_per_weight: 16.0,
            memory_reduction: 0.50,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.2,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string(), "gemma".to_string()],
        });
        
        supported_formats.insert("BF16".to_string(), QuantizationConfig {
            name: "BF16".to_string(),
            description: "16-bit brain floating point format".to_string(),
            bits_per_weight: 16.0,
            memory_reduction: 0.50,
            quality_impact: QualityImpact::Minimal,
            speed_improvement: 1.2,
            supported_architectures: vec!["llama".to_string(), "mistral".to_string(), "qwen".to_string(), "gemma".to_string()],
        });
        
        Self {
            supported_formats,
        }
    }
    
    /// Get information about a specific quantization format
    pub fn get_format_info(&self, format: &str) -> Option<&QuantizationConfig> {
        self.supported_formats.get(format)
    }
    
    /// List all supported quantization formats
    pub fn list_supported_formats(&self) -> Vec<&QuantizationConfig> {
        self.supported_formats.values().collect()
    }
    
    /// Get formats compatible with a specific architecture
    pub fn get_compatible_formats(&self, architecture: &str) -> Vec<&QuantizationConfig> {
        self.supported_formats
            .values()
            .filter(|config| config.supported_architectures.contains(&architecture.to_lowercase()))
            .collect()
    }
    
    /// Recommend optimal quantization format based on requirements
    pub fn recommend_format(&self, requirements: &QuantizationRequirements) -> Option<&QuantizationConfig> {
        let mut candidates: Vec<&QuantizationConfig> = self.supported_formats.values().collect();
        
        // Filter by architecture compatibility
        if let Some(arch) = &requirements.architecture {
            candidates.retain(|config| {
                config.supported_architectures.contains(&arch.to_lowercase())
            });
        }
        
        // Filter by memory constraints
        if let Some(max_memory) = requirements.max_memory_gb {
            let target_reduction = 1.0 - (requirements.model_size_gb.unwrap_or(7.0) / max_memory as f64);
            candidates.retain(|config| config.memory_reduction >= target_reduction as f32);
        }
        
        // Filter by quality requirements
        match requirements.min_quality {
            QualityLevel::Highest => {
                candidates.retain(|config| matches!(config.quality_impact, QualityImpact::Minimal));
            }
            QualityLevel::High => {
                candidates.retain(|config| matches!(
                    config.quality_impact, 
                    QualityImpact::Minimal | QualityImpact::Low
                ));
            }
            QualityLevel::Medium => {
                candidates.retain(|config| !matches!(config.quality_impact, QualityImpact::High));
            }
            QualityLevel::Any => {}
        }
        
        // Sort by preference: quality first, then speed
        candidates.sort_by(|a, b| {
            // First prioritize quality
            let quality_order = match (&a.quality_impact, &b.quality_impact) {
                (QualityImpact::Minimal, QualityImpact::Minimal) => std::cmp::Ordering::Equal,
                (QualityImpact::Minimal, _) => std::cmp::Ordering::Less,
                (_, QualityImpact::Minimal) => std::cmp::Ordering::Greater,
                (QualityImpact::Low, QualityImpact::Low) => std::cmp::Ordering::Equal,
                (QualityImpact::Low, _) => std::cmp::Ordering::Less,
                (_, QualityImpact::Low) => std::cmp::Ordering::Greater,
                _ => std::cmp::Ordering::Equal,
            };
            
            if quality_order != std::cmp::Ordering::Equal {
                return quality_order;
            }
            
            // Then prioritize speed
            b.speed_improvement.partial_cmp(&a.speed_improvement).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        candidates.first().copied()
    }
    
    /// Calculate estimated memory usage for a model with specific quantization
    pub fn estimate_memory_usage(&self, model_params: u64, quantization: &str) -> Option<usize> {
        let config = self.get_format_info(quantization)?;
        let bytes_per_param = config.bits_per_weight / 8.0;
        Some((model_params as f64 * bytes_per_param as f64) as usize)
    }
    
    /// Get quantization recommendation text for UI display
    pub fn get_recommendation_text(&self, format: &str) -> String {
        if let Some(config) = self.get_format_info(format) {
            format!(
                "{}: {} (Memory reduction: {:.0}%, Speed: {:.1}x, Quality: {:?})",
                config.name,
                config.description,
                config.memory_reduction * 100.0,
                config.speed_improvement,
                config.quality_impact
            )
        } else {
            format!("Unknown quantization format: {}", format)
        }
    }
    
    /// Validate quantization format for a specific model
    pub fn validate_format(&self, model_id: &str, quantization: &str) -> Result<()> {
        let config = self.get_format_info(quantization)
            .ok_or_else(|| anyhow::anyhow!("Unsupported quantization format: {}", quantization))?;
        
        // Try to detect model architecture from model_id
        let architecture = self.detect_architecture(model_id);
        
        if let Some(arch) = architecture {
            if !config.supported_architectures.contains(&arch) {
                return Err(anyhow::anyhow!(
                    "Quantization format '{}' is not compatible with architecture '{}'",
                    quantization, arch
                ));
            }
        }
        
        Ok(())
    }
    
    fn detect_architecture(&self, model_id: &str) -> Option<String> {
        let model_lower = model_id.to_lowercase();
        
        if model_lower.contains("llama") {
            Some("llama".to_string())
        } else if model_lower.contains("mistral") || model_lower.contains("pixtral") {
            Some("mistral".to_string())
        } else if model_lower.contains("qwen") {
            Some("qwen".to_string())
        } else if model_lower.contains("gemma") {
            Some("gemma".to_string())
        } else if model_lower.contains("phi") {
            Some("phi".to_string())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationRequirements {
    pub architecture: Option<String>,
    pub model_size_gb: Option<f64>,
    pub max_memory_gb: Option<u32>,
    pub min_quality: QualityLevel,
    pub prefer_speed: bool,
}

#[derive(Debug, Clone)]
pub enum QualityLevel {
    Highest,
    High,
    Medium,
    Any,
}

impl Default for QuantizationRequirements {
    fn default() -> Self {
        Self {
            architecture: None,
            model_size_gb: None,
            max_memory_gb: None,
            min_quality: QualityLevel::High,
            prefer_speed: false,
        }
    }
}

/// Auto-detect optimal quantization based on system capabilities
pub struct AutoQuantizationSelector {
    quantization_manager: QuantizationManager,
}

impl AutoQuantizationSelector {
    pub fn new() -> Self {
        Self {
            quantization_manager: QuantizationManager::new(),
        }
    }
    
    /// Select optimal quantization based on system resources
    pub fn select_optimal(&self, model_id: &str, model_params: Option<u64>) -> Result<String> {
        let available_memory = self.get_available_memory()?;
        let architecture = self.quantization_manager.detect_architecture(model_id);
        
        let requirements = QuantizationRequirements {
            architecture,
            model_size_gb: model_params.map(|p| p as f64 * 4.0 / 1e9), // Assume F32 baseline
            max_memory_gb: Some((available_memory / (1024 * 1024 * 1024)) as u32),
            min_quality: if available_memory > 16 * 1024 * 1024 * 1024 {
                QualityLevel::Highest
            } else if available_memory > 8 * 1024 * 1024 * 1024 {
                QualityLevel::High
            } else {
                QualityLevel::Medium
            },
            prefer_speed: available_memory < 8 * 1024 * 1024 * 1024,
        };
        
        let recommended = self.quantization_manager.recommend_format(&requirements)
            .ok_or_else(|| anyhow::anyhow!("No suitable quantization format found"))?;
        
        Ok(recommended.name.clone())
    }
    
    fn get_available_memory(&self) -> Result<usize> {
        // This is a simplified version - in practice, you'd want to use
        // platform-specific APIs to get accurate memory information
        #[cfg(target_os = "macos")]
        {
            // On macOS, we can assume unified memory architecture
            // This would need actual system memory detection
            Ok(16 * 1024 * 1024 * 1024) // Default to 16GB
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // For other platforms, be more conservative
            Ok(8 * 1024 * 1024 * 1024) // Default to 8GB
        }
    }
}