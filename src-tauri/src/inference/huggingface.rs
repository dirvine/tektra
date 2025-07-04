use super::*;
use anyhow::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// HuggingFace API integration for model discovery and metadata
pub struct HuggingFaceClient {
    client: Client,
    api_base: String,
    cache: std::sync::Arc<std::sync::RwLock<HashMap<String, ModelMetadata>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Vec<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub model_size: Option<String>,
    pub model_type: Option<String>,
    pub architecture: Option<String>,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub context_length: Option<usize>,
    pub quantizations: Vec<String>,
    pub created_at: Option<String>,
    pub last_modified: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HuggingFaceModelResponse {
    #[serde(rename = "modelId")]
    model_id: Option<String>,
    id: Option<String>,
    #[serde(rename = "modelName")]
    model_name: Option<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    tags: Option<Vec<String>>,
    #[serde(rename = "pipelineTag")]
    pipeline_tag: Option<String>,
    library: Option<String>,
    #[serde(rename = "modelSize")]
    model_size: Option<String>,
    description: Option<String>,
    #[serde(rename = "createdAt")]
    created_at: Option<String>,
    #[serde(rename = "lastModified")]
    last_modified: Option<String>,
}

impl HuggingFaceClient {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_base: "https://huggingface.co/api".to_string(),
            cache: std::sync::Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Search for models on HuggingFace Hub
    pub async fn search_models(&self, query: &str, filters: &SearchFilters) -> Result<Vec<ModelMetadata>> {
        info!("Searching HuggingFace models: query='{}', filters={:?}", query, filters);
        
        let url = format!("{}/models", self.api_base);
        let limit_str = filters.limit.to_string();
        let mut params = vec![
            ("search", query),
            ("limit", &limit_str),
        ];

        if let Some(sort) = &filters.sort {
            params.push(("sort", sort));
        }

        let mut filter_strings = Vec::new();
        
        if !filters.tags.is_empty() {
            for tag in &filters.tags {
                filter_strings.push(format!("tag:{}", tag));
            }
        }

        if let Some(library) = &filters.library {
            filter_strings.push(format!("library:{}", library));
        }
        
        for filter_str in &filter_strings {
            params.push(("filter", filter_str.as_str()));
        }

        let response = self.client
            .get(&url)
            .query(&params)
            .header("User-Agent", "Tektra/0.2.3")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HuggingFace API error: {} - {}", 
                response.status(), 
                response.text().await.unwrap_or_default()
            ));
        }

        let models: Vec<HuggingFaceModelResponse> = response.json().await?;
        let mut results = Vec::new();

        for model_response in models {
            if let Some(metadata) = self.parse_model_metadata(model_response).await {
                results.push(metadata);
            }
        }

        info!("Found {} models matching search criteria", results.len());
        Ok(results)
    }

    /// Get detailed information about a specific model
    pub async fn get_model_info(&self, model_id: &str) -> Result<Option<ModelMetadata>> {
        // Check cache first
        if let Ok(cache) = self.cache.read() {
            if let Some(cached) = cache.get(model_id) {
                debug!("Retrieved model info from cache: {}", model_id);
                return Ok(Some(cached.clone()));
            }
        }

        info!("Fetching model info from HuggingFace: {}", model_id);
        
        let url = format!("{}/models/{}", self.api_base, model_id);
        let response = self.client
            .get(&url)
            .header("User-Agent", "Tektra/0.2.3")
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            warn!("Model not found on HuggingFace: {}", model_id);
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HuggingFace API error for {}: {} - {}", 
                model_id,
                response.status(), 
                response.text().await.unwrap_or_default()
            ));
        }

        let model_response: HuggingFaceModelResponse = response.json().await?;
        let metadata = self.parse_model_metadata(model_response).await;

        // Cache the result
        if let Some(ref meta) = metadata {
            if let Ok(mut cache) = self.cache.write() {
                cache.insert(model_id.to_string(), meta.clone());
            }
        }

        Ok(metadata)
    }

    /// Get available quantized versions of a model
    pub async fn get_model_quantizations(&self, model_id: &str) -> Result<Vec<String>> {
        info!("Fetching quantizations for model: {}", model_id);
        
        // Search for quantized versions of this model
        let search_query = format!("{} quantized", model_id);
        let filters = SearchFilters {
            tags: vec!["gguf".to_string(), "quantized".to_string()],
            library: Some("gguf".to_string()),
            sort: Some("downloads".to_string()),
            limit: 50,
        };

        let models = self.search_models(&search_query, &filters).await?;
        let mut quantizations = Vec::new();

        for model in models {
            if model.id.contains(model_id) || model.id.contains(&model_id.replace("/", "-")) {
                // Extract quantization format from model ID or tags
                for tag in &model.tags {
                    if tag.contains("Q4") || tag.contains("Q5") || tag.contains("Q6") || tag.contains("Q8") {
                        if !quantizations.contains(tag) {
                            quantizations.push(tag.clone());
                        }
                    }
                }

                // Extract from model ID patterns
                if let Some(quant) = extract_quantization_from_id(&model.id) {
                    if !quantizations.contains(&quant) {
                        quantizations.push(quant);
                    }
                }
            }
        }

        info!("Found {} quantizations for {}: {:?}", quantizations.len(), model_id, quantizations);
        Ok(quantizations)
    }

    /// Check if a model supports vision
    pub async fn check_vision_support(&self, model_id: &str) -> Result<bool> {
        let metadata = self.get_model_info(model_id).await?;
        
        if let Some(meta) = metadata {
            return Ok(meta.supports_vision);
        }

        // Fallback: check model ID patterns
        let model_lower = model_id.to_lowercase();
        let supports_vision = model_lower.contains("vision") ||
                             model_lower.contains("vl") ||
                             model_lower.contains("llava") ||
                             model_lower.contains("pixtral") ||
                             model_lower.contains("qwen2.5-vl") ||
                             model_lower.contains("cogvlm") ||
                             model_lower.contains("moondream");

        Ok(supports_vision)
    }

    /// Get recommended models for specific use cases
    pub async fn get_recommended_models(&self, use_case: &str) -> Result<Vec<ModelMetadata>> {
        info!("Getting recommended models for use case: {}", use_case);
        
        let (query, filters) = match use_case.to_lowercase().as_str() {
            "vision" | "image" | "multimodal" => {
                ("vision language model", SearchFilters {
                    tags: vec!["vision".to_string(), "multimodal".to_string()],
                    library: Some("transformers".to_string()),
                    sort: Some("downloads".to_string()),
                    limit: 20,
                })
            }
            "text" | "conversation" | "chat" => {
                ("conversational", SearchFilters {
                    tags: vec!["conversational".to_string(), "text-generation".to_string()],
                    library: Some("transformers".to_string()),
                    sort: Some("downloads".to_string()),
                    limit: 20,
                })
            }
            "code" | "coding" | "programming" => {
                ("code generation", SearchFilters {
                    tags: vec!["code".to_string(), "coding".to_string()],
                    library: Some("transformers".to_string()),
                    sort: Some("downloads".to_string()),
                    limit: 20,
                })
            }
            _ => {
                ("language model", SearchFilters::default())
            }
        };

        self.search_models(query, &filters).await
    }

    /// Parse HuggingFace API response into our metadata format
    async fn parse_model_metadata(&self, response: HuggingFaceModelResponse) -> Option<ModelMetadata> {
        let model_id = response.model_id.or(response.id)?;
        let tags = response.tags.unwrap_or_default();
        
        // Detect capabilities from tags and model ID
        let supports_vision = tags.iter().any(|tag| {
            tag.contains("vision") || tag.contains("multimodal") || tag.contains("vl")
        }) || model_id.to_lowercase().contains("vision") || 
            model_id.to_lowercase().contains("vl") ||
            model_id.to_lowercase().contains("llava") ||
            model_id.to_lowercase().contains("pixtral");

        let supports_audio = tags.iter().any(|tag| {
            tag.contains("audio") || tag.contains("speech") || tag.contains("whisper")
        });

        // Extract architecture information
        let architecture = tags.iter()
            .find(|tag| tag.contains("llama") || tag.contains("mistral") || 
                        tag.contains("qwen") || tag.contains("gemma"))
            .map(|tag| tag.clone())
            .or_else(|| {
                let model_lower = model_id.to_lowercase();
                if model_lower.contains("llama") {
                    Some("llama".to_string())
                } else if model_lower.contains("mistral") {
                    Some("mistral".to_string())
                } else if model_lower.contains("qwen") {
                    Some("qwen".to_string())
                } else if model_lower.contains("gemma") {
                    Some("gemma".to_string())
                } else {
                    None
                }
            });

        // Extract context length from tags or model name
        let context_length = tags.iter()
            .find_map(|tag| {
                if tag.contains("32k") || tag.contains("32768") {
                    Some(32768)
                } else if tag.contains("128k") || tag.contains("131072") {
                    Some(131072)
                } else if tag.contains("8k") || tag.contains("8192") {
                    Some(8192)
                } else if tag.contains("4k") || tag.contains("4096") {
                    Some(4096)
                } else {
                    None
                }
            });

        // Extract available quantizations
        let quantizations = tags.iter()
            .filter(|tag| tag.contains("Q4") || tag.contains("Q5") || 
                          tag.contains("Q6") || tag.contains("Q8") ||
                          tag.contains("gguf") || tag.contains("quantized"))
            .map(|tag| tag.clone())
            .collect();

        Some(ModelMetadata {
            id: model_id.clone(),
            name: response.model_name.unwrap_or_else(|| model_id.clone()),
            description: response.description,
            downloads: response.downloads,
            likes: response.likes,
            tags,
            pipeline_tag: response.pipeline_tag,
            library_name: response.library,
            model_size: response.model_size,
            model_type: None, // Not provided in basic API response
            architecture,
            supports_vision,
            supports_audio,
            context_length,
            quantizations,
            created_at: response.created_at,
            last_modified: response.last_modified,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SearchFilters {
    pub tags: Vec<String>,
    pub library: Option<String>,
    pub sort: Option<String>,
    pub limit: usize,
}

impl Default for SearchFilters {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            library: None,
            sort: Some("downloads".to_string()),
            limit: 20,
        }
    }
}

/// Extract quantization format from model ID
fn extract_quantization_from_id(model_id: &str) -> Option<String> {
    let id_upper = model_id.to_uppercase();
    
    // Common quantization patterns
    if id_upper.contains("Q4_K_M") {
        Some("Q4_K_M".to_string())
    } else if id_upper.contains("Q4_K_S") {
        Some("Q4_K_S".to_string())
    } else if id_upper.contains("Q5_K_M") {
        Some("Q5_K_M".to_string())
    } else if id_upper.contains("Q5_K_S") {
        Some("Q5_K_S".to_string())
    } else if id_upper.contains("Q6_K") {
        Some("Q6_K".to_string())
    } else if id_upper.contains("Q8_0") {
        Some("Q8_0".to_string())
    } else if id_upper.contains("F16") {
        Some("F16".to_string())
    } else if id_upper.contains("F32") {
        Some("F32".to_string())
    } else {
        None
    }
}

/// HuggingFace model downloader with progress tracking
pub struct ModelDownloader {
    client: Client,
    progress_callback: Option<Box<dyn Fn(u64, u64) + Send + Sync>>,
}

impl Clone for ModelDownloader {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            progress_callback: None, // Callbacks can't be cloned, create new instance without callback
        }
    }
}

impl ModelDownloader {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            progress_callback: None,
        }
    }

    pub fn with_progress_callback<F>(mut self, callback: F) -> Self 
    where
        F: Fn(u64, u64) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Download a model from HuggingFace Hub
    pub async fn download_model(&self, model_id: &str, local_path: &std::path::Path) -> Result<()> {
        info!("Starting download of model: {} to {:?}", model_id, local_path);
        
        // Create directory if it doesn't exist
        if let Some(parent) = local_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Download model files (simplified - in practice would need to handle multiple files)
        let download_url = format!("https://huggingface.co/{}/resolve/main/config.json", model_id);
        
        let response = self.client
            .get(&download_url)
            .header("User-Agent", "Tektra/0.2.3")
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model {}: HTTP {}", 
                model_id, 
                response.status()
            ));
        }

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();

        let mut file = std::fs::File::create(local_path)?;
        
        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            use std::io::Write;
            file.write_all(&chunk)?;
            
            downloaded += chunk.len() as u64;
            
            if let Some(ref callback) = self.progress_callback {
                callback(downloaded, total_size);
            }
        }

        info!("Model download completed: {}", model_id);
        Ok(())
    }

    /// Get download URL for a specific model file
    pub fn get_download_url(&self, model_id: &str, filename: &str) -> String {
        format!("https://huggingface.co/{}/resolve/main/{}", model_id, filename)
    }

    /// Check if a model file exists on HuggingFace
    pub async fn check_file_exists(&self, model_id: &str, filename: &str) -> Result<bool> {
        let url = self.get_download_url(model_id, filename);
        
        let response = self.client
            .head(&url)
            .header("User-Agent", "Tektra/0.2.3")
            .send()
            .await?;

        Ok(response.status().is_success())
    }
}

/// Enhanced model registry with HuggingFace integration
pub struct EnhancedModelRegistry {
    base_registry: ModelRegistry,
    hf_client: HuggingFaceClient,
    downloader: ModelDownloader,
}

impl EnhancedModelRegistry {
    pub fn new() -> Self {
        Self {
            base_registry: ModelRegistry::new(),
            hf_client: HuggingFaceClient::new(),
            downloader: ModelDownloader::new(),
        }
    }

    /// Initialize with both default and discovered models
    pub async fn initialize(&self) -> Result<()> {
        // Initialize base registry first
        self.base_registry.initialize().await?;

        // Discover additional models from HuggingFace
        self.discover_popular_models().await?;

        Ok(())
    }

    /// Discover and add popular models from HuggingFace
    async fn discover_popular_models(&self) -> Result<()> {
        info!("Discovering popular models from HuggingFace");

        // Get vision models
        let vision_models = self.hf_client.get_recommended_models("vision").await?;
        for model in vision_models.into_iter().take(5) {
            if let Err(e) = self.add_discovered_model(&model).await {
                warn!("Failed to add discovered model {}: {}", model.id, e);
            }
        }

        // Get text models
        let text_models = self.hf_client.get_recommended_models("text").await?;
        for model in text_models.into_iter().take(5) {
            if let Err(e) = self.add_discovered_model(&model).await {
                warn!("Failed to add discovered model {}: {}", model.id, e);
            }
        }

        Ok(())
    }

    /// Add a discovered model to the registry
    async fn add_discovered_model(&self, metadata: &ModelMetadata) -> Result<()> {
        debug!("Adding discovered model: {}", metadata.id);

        let default_config = DefaultModelConfig {
            id: metadata.id.replace("/", "_"),
            name: metadata.name.clone(),
            model_id: metadata.id.clone(),
            description: metadata.description.clone()
                .unwrap_or_else(|| "Discovered from HuggingFace".to_string()),
            quantization: metadata.quantizations.first().cloned(),
            context_window: metadata.context_length.unwrap_or(8192),
            supports_vision: metadata.supports_vision,
            supports_audio: metadata.supports_audio,
            supports_documents: true,
            default: false,
            recommended_for: vec!["discovered".to_string()],
        };

        // This would be added to the base registry's model list
        // For now, we just log the discovery
        info!("Discovered model: {} ({})", default_config.name, default_config.model_id);

        Ok(())
    }

    /// Search for models with specific criteria
    pub async fn search_models(&self, query: &str, filters: &SearchFilters) -> Result<Vec<ModelMetadata>> {
        self.hf_client.search_models(query, filters).await
    }

    /// Get enhanced model information
    pub async fn get_enhanced_model_info(&self, model_id: &str) -> Result<Option<ModelMetadata>> {
        self.hf_client.get_model_info(model_id).await
    }

    /// Download a model with progress tracking
    pub async fn download_model_with_progress<F>(&self, model_id: &str, local_path: &std::path::Path, progress_callback: F) -> Result<()>
    where
        F: Fn(u64, u64) + Send + Sync + 'static,
    {
        let downloader = self.downloader.clone().with_progress_callback(progress_callback);
        downloader.download_model(model_id, local_path).await
    }

    /// Get all functionality from base registry
    pub async fn generate(&self, input: MultimodalInput) -> Result<ModelResponse> {
        self.base_registry.generate(input).await
    }

    pub async fn stream_generate(&self, input: MultimodalInput) -> Result<Pin<Box<dyn Stream<Item = Result<Token>> + Send>>> {
        self.base_registry.stream_generate(input).await
    }

    pub async fn list_models(&self) -> Vec<DefaultModelConfig> {
        self.base_registry.list_models().await
    }

    pub async fn get_active_model_id(&self) -> Option<String> {
        self.base_registry.get_active_model_id().await
    }

    pub async fn switch_model(&self, model_id: &str) -> Result<()> {
        self.base_registry.switch_model(model_id).await
    }

    pub async fn get_stats(&self) -> RegistryStats {
        self.base_registry.get_stats().await
    }
}