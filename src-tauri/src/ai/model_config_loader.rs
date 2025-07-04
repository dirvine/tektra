use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{info, warn};

use crate::ai::unified_model_manager::{ModelConfig, DeviceConfig, GenerationParams};

/// Root configuration structure for models.toml
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfiguration {
    pub backends: BackendsConfig,
    pub models: Vec<ModelDefinition>,
    pub custom_models: Option<HashMap<String, String>>,
    pub presets: HashMap<String, GenerationPreset>,
    pub memory_limits: MemoryLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendsConfig {
    pub default: Vec<String>,
    pub mistral_rs: Option<MistralRsConfig>,
    pub llama_cpp: Option<LlamaCppConfig>,
    pub ollama: Option<OllamaConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralRsConfig {
    pub enabled: bool,
    pub flash_attention: bool,
    pub mcp_enabled: bool,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    pub enabled: bool,
    pub n_threads: i32,
    pub use_mmap: bool,
    pub n_gpu_layers: i32,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub enabled: bool,
    pub host: String,
    pub port: u16,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub backend_preferences: Vec<String>,
    pub template: String,
    pub context_length: usize,
    pub multimodal: bool,
    pub capabilities: Vec<String>,
    pub quantization: Option<String>,
    pub device: Option<String>,
    pub model_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationPreset {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    pub mistral_rs: usize,
    pub llama_cpp: usize,
    pub ollama: usize,
    pub total: usize,
}

/// Loader for model configuration files
pub struct ModelConfigLoader {
    config: Option<ModelsConfiguration>,
    config_path: String,
}

impl ModelConfigLoader {
    pub fn new(config_path: impl Into<String>) -> Self {
        Self {
            config: None,
            config_path: config_path.into(),
        }
    }
    
    /// Load configuration from TOML file
    pub async fn load(&mut self) -> Result<()> {
        let content = tokio::fs::read_to_string(&self.config_path).await?;
        self.config = Some(toml::from_str(&content)?);
        
        info!("Loaded model configuration from {}", self.config_path);
        Ok(())
    }
    
    /// Reload configuration from file
    pub async fn reload(&mut self) -> Result<()> {
        self.load().await
    }
    
    /// Get the loaded configuration
    pub fn config(&self) -> Option<&ModelsConfiguration> {
        self.config.as_ref()
    }
    
    /// Get backend preferences for the system
    pub fn get_backend_preferences(&self) -> HashMap<String, Vec<String>> {
        let mut preferences = HashMap::new();
        
        if let Some(config) = &self.config {
            // Add default preferences
            preferences.insert("_default".to_string(), config.backends.default.clone());
            
            // Add model-specific preferences
            for model in &config.models {
                preferences.insert(model.id.clone(), model.backend_preferences.clone());
            }
        }
        
        preferences
    }
    
    /// Get model configuration by ID
    pub fn get_model_definition(&self, model_id: &str) -> Option<&ModelDefinition> {
        self.config.as_ref()?.models.iter().find(|m| m.id == model_id)
    }
    
    /// Convert model definition to ModelConfig for UnifiedModelManager
    pub fn to_model_config(&self, model_def: &ModelDefinition) -> ModelConfig {
        let device = match model_def.device.as_deref() {
            Some("cpu") => DeviceConfig::Cpu,
            Some("metal") => DeviceConfig::Metal,
            Some(d) if d.starts_with("cuda:") => {
                let idx = d.strip_prefix("cuda:")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                DeviceConfig::Cuda(idx)
            }
            _ => DeviceConfig::Auto,
        };
        
        ModelConfig {
            model_id: model_def.id.clone(),
            model_path: model_def.model_file.clone(),
            context_length: model_def.context_length,
            quantization: model_def.quantization.clone(),
            device,
            rope_scale: None,
            template_name: Some(model_def.template.clone()),
        }
    }
    
    /// Get generation parameters from preset
    pub fn get_preset(&self, name: &str) -> Option<GenerationParams> {
        let preset = self.config.as_ref()?.presets.get(name)?;
        
        Some(GenerationParams {
            max_tokens: preset.max_tokens,
            temperature: preset.temperature,
            top_p: preset.top_p,
            top_k: preset.top_k,
            repeat_penalty: preset.repeat_penalty,
            seed: None,
            stop_sequences: vec![],
            stream: false,
        })
    }
    
    /// List all available models
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.config.as_ref()
            .map(|c| c.models.iter().map(|m| ModelInfo {
                id: m.id.clone(),
                name: m.name.clone(),
                description: m.description.clone(),
                multimodal: m.multimodal,
                capabilities: m.capabilities.clone(),
            }).collect())
            .unwrap_or_default()
    }
    
    /// Check if a backend is enabled
    pub fn is_backend_enabled(&self, backend: &str) -> bool {
        if let Some(config) = &self.config {
            match backend {
                "mistral_rs" => config.backends.mistral_rs.as_ref().map(|c| c.enabled).unwrap_or(false),
                "llama_cpp" => config.backends.llama_cpp.as_ref().map(|c| c.enabled).unwrap_or(false),
                "ollama" => config.backends.ollama.as_ref().map(|c| c.enabled).unwrap_or(false),
                _ => false,
            }
        } else {
            false
        }
    }
    
    /// Get memory limit for a backend
    pub fn get_memory_limit(&self, backend: &str) -> Option<usize> {
        if let Some(config) = &self.config {
            match backend {
                "mistral_rs" => Some(config.memory_limits.mistral_rs * 1024 * 1024), // Convert MB to bytes
                "llama_cpp" => Some(config.memory_limits.llama_cpp * 1024 * 1024),
                "ollama" => Some(config.memory_limits.ollama * 1024 * 1024),
                _ => None,
            }
        } else {
            None
        }
    }
    
    /// Get total memory limit
    pub fn get_total_memory_limit(&self) -> usize {
        self.config.as_ref()
            .map(|c| c.memory_limits.total * 1024 * 1024)
            .unwrap_or(32 * 1024 * 1024 * 1024) // Default 32GB
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        if let Some(config) = &self.config {
            // Check if at least one backend is enabled
            let any_enabled = [
                config.backends.mistral_rs.as_ref().map(|c| c.enabled).unwrap_or(false),
                config.backends.llama_cpp.as_ref().map(|c| c.enabled).unwrap_or(false),
                config.backends.ollama.as_ref().map(|c| c.enabled).unwrap_or(false),
            ].iter().any(|&e| e);
            
            if !any_enabled {
                issues.push("No backends are enabled".to_string());
            }
            
            // Check model definitions
            for model in &config.models {
                if model.backend_preferences.is_empty() {
                    issues.push(format!("Model '{}' has no backend preferences", model.id));
                }
                
                // Check if preferred backends are in the enabled list
                for backend in &model.backend_preferences {
                    if !self.is_backend_enabled(backend) {
                        warn!("Model '{}' prefers backend '{}' which is not enabled", model.id, backend);
                    }
                }
            }
            
            // Check memory limits
            let backend_total = config.memory_limits.mistral_rs 
                + config.memory_limits.llama_cpp 
                + config.memory_limits.ollama;
            
            if backend_total > config.memory_limits.total {
                issues.push(format!(
                    "Backend memory limits ({} MB) exceed total limit ({} MB)",
                    backend_total, config.memory_limits.total
                ));
            }
        } else {
            issues.push("No configuration loaded".to_string());
        }
        
        issues
    }
}

/// Simplified model information for listing
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub multimodal: bool,
    pub capabilities: Vec<String>,
}

/// Builder for creating model configurations programmatically
pub struct ModelConfigBuilder {
    id: String,
    name: String,
    description: String,
    backend_preferences: Vec<String>,
    template: String,
    context_length: usize,
    multimodal: bool,
    capabilities: Vec<String>,
    quantization: Option<String>,
    device: Option<String>,
    model_file: Option<String>,
}

impl ModelConfigBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: String::new(),
            description: String::new(),
            backend_preferences: vec![],
            template: "chatml".to_string(),
            context_length: 4096,
            multimodal: false,
            capabilities: vec!["text".to_string()],
            quantization: None,
            device: None,
            model_file: None,
        }
    }
    
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
    
    pub fn backends(mut self, backends: Vec<String>) -> Self {
        self.backend_preferences = backends;
        self
    }
    
    pub fn template(mut self, template: impl Into<String>) -> Self {
        self.template = template.into();
        self
    }
    
    pub fn context_length(mut self, length: usize) -> Self {
        self.context_length = length;
        self
    }
    
    pub fn multimodal(mut self, multimodal: bool) -> Self {
        self.multimodal = multimodal;
        self
    }
    
    pub fn capabilities(mut self, caps: Vec<String>) -> Self {
        self.capabilities = caps;
        self
    }
    
    pub fn quantization(mut self, quant: impl Into<String>) -> Self {
        self.quantization = Some(quant.into());
        self
    }
    
    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.device = Some(device.into());
        self
    }
    
    pub fn model_file(mut self, file: impl Into<String>) -> Self {
        self.model_file = Some(file.into());
        self
    }
    
    pub fn build(self) -> ModelDefinition {
        ModelDefinition {
            id: self.id,
            name: self.name,
            description: self.description,
            backend_preferences: self.backend_preferences,
            template: self.template,
            context_length: self.context_length,
            multimodal: self.multimodal,
            capabilities: self.capabilities,
            quantization: self.quantization,
            device: self.device,
            model_file: self.model_file,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_model_builder() {
        let model = ModelConfigBuilder::new("test-model")
            .name("Test Model")
            .description("A test model")
            .backends(vec!["mistral_rs".to_string()])
            .context_length(8192)
            .multimodal(true)
            .capabilities(vec!["text".to_string(), "image".to_string()])
            .build();
        
        assert_eq!(model.id, "test-model");
        assert_eq!(model.name, "Test Model");
        assert!(model.multimodal);
        assert_eq!(model.capabilities.len(), 2);
    }
}