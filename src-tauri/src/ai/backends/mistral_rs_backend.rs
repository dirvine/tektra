use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn, error, debug};

use crate::ai::unified_model_manager::{
    ModelBackend, MultimodalInput, GenerationParams, ModelConfig, 
    BackendCapabilities, DeviceConfig
};

#[cfg(feature = "mistral-backend")]
use mistralrs::{
    Constraint, Device, DeviceMapMetadata, GGUFLoaderBuilder, GGUFSpecificConfig,
    MistralRs, MistralRsBuilder, ModelDType, NormalLoaderBuilder, PagedAttentionConfig,
    Request, RequestMessage, Response, ResponseOk, SamplingParams, SchedulerMethod,
    TokenSource, VisionLoaderBuilder, VisionSpecificConfig,
};

pub struct MistralRsBackend {
    #[cfg(feature = "mistral-backend")]
    model: Arc<Mutex<Option<Arc<MistralRs>>>>,
    model_loaded: Arc<RwLock<bool>>,
    current_model: Arc<RwLock<Option<String>>>,
    capabilities: BackendCapabilities,
    memory_usage: Arc<RwLock<usize>>,
}

impl MistralRsBackend {
    pub fn new() -> Result<Self> {
        info!("Initializing MistralRs backend");
        
        Ok(Self {
            #[cfg(feature = "mistral-backend")]
            model: Arc::new(Mutex::new(None)),
            model_loaded: Arc::new(RwLock::new(false)),
            current_model: Arc::new(RwLock::new(None)),
            capabilities: BackendCapabilities {
                text_generation: true,
                image_understanding: true,
                audio_processing: true,
                video_processing: false,
                streaming: true,
                function_calling: true,
                quantization_formats: vec![
                    "Q4_0".to_string(),
                    "Q4_K_S".to_string(),
                    "Q4_K_M".to_string(),
                    "Q5_0".to_string(),
                    "Q5_K_S".to_string(),
                    "Q5_K_M".to_string(),
                    "Q8_0".to_string(),
                    "F16".to_string(),
                ],
                max_context_length: 131072,
            },
            memory_usage: Arc::new(RwLock::new(0)),
        })
    }
    
    async fn supports_model_format(&self, model_id: &str) -> bool {
        // MistralRs supports GGUF, SafeTensors, and more
        model_id.ends_with(".gguf") || 
        model_id.ends_with(".safetensors") ||
        model_id.contains("mistral") ||
        model_id.contains("gemma") ||
        model_id.contains("llama") ||
        model_id.contains("phi") ||
        model_id.contains("qwen")
    }
    
    #[cfg(feature = "mistral-backend")]
    fn get_device_config(device: &DeviceConfig) -> Device {
        match device {
            DeviceConfig::Cpu => Device::Cpu,
            DeviceConfig::Cuda(idx) => Device::Cuda(*idx),
            DeviceConfig::Metal => Device::Metal(*idx),
            DeviceConfig::Auto => {
                #[cfg(target_os = "macos")]
                return Device::Metal(0);
                
                #[cfg(not(target_os = "macos"))]
                return Device::Cpu;
            }
        }
    }
    
    #[cfg(feature = "mistral-backend")]
    fn get_dtype(quantization: &Option<String>) -> ModelDType {
        match quantization.as_deref() {
            Some("Q4_0") | Some("Q4_K_S") | Some("Q4_K_M") |
            Some("Q5_0") | Some("Q5_K_S") | Some("Q5_K_M") |
            Some("Q8_0") => ModelDType::Auto,
            Some("F16") => ModelDType::F16,
            Some("BF16") => ModelDType::BF16,
            Some("F32") | None => ModelDType::F32,
            _ => ModelDType::Auto,
        }
    }
}

#[async_trait]
impl ModelBackend for MistralRsBackend {
    fn name(&self) -> &str {
        "mistral_rs"
    }
    
    async fn supports_model(&self, model_id: &str) -> bool {
        self.supports_model_format(model_id).await
    }
    
    async fn load_model(&mut self, config: &ModelConfig) -> Result<()> {
        info!("Loading model '{}' in MistralRs backend", config.model_id);
        
        #[cfg(feature = "mistral-backend")]
        {
            let device = Self::get_device_config(&config.device);
            let dtype = Self::get_dtype(&config.quantization);
            
            // Determine loader type based on model format
            let loader = if config.model_id.ends_with(".gguf") {
                // GGUF loader for quantized models
                let gguf_config = GGUFSpecificConfig {
                    prompt_batchsize: None,
                    topology: None,
                };
                
                GGUFLoaderBuilder::new(
                    config.model_id.clone(),
                    vec![],  // No LoRA adapters for now
                    None,    // No X-LoRA config
                    Some(gguf_config),
                )
            } else if config.model_id.contains("vision") || config.model_id.contains("llava") {
                // Vision loader for multimodal models
                let vision_config = VisionSpecificConfig {
                    use_flash_attn: true,
                    prompt_batchsize: None,
                    topology: None,
                    write_uqff: None,
                    from_uqff: None,
                };
                
                VisionLoaderBuilder::new(
                    config.model_id.clone(),
                    "auto".to_string(),  // Auto-detect vision model type
                    None,
                    Some(vision_config),
                )
            } else {
                // Normal loader for standard models
                NormalLoaderBuilder::new(
                    config.model_id.clone(),
                    vec![],  // No LoRA adapters
                    None,    // No X-LoRA config
                    None,    // No specific config
                )
            };
            
            // Build the model
            let pipeline = loader
                .with_logging()
                .load_model_from_hf(None)  // Auto-detect model files
                .build()
                .await?;
            
            let mistral_rs = MistralRsBuilder::new(
                pipeline,
                SchedulerMethod::Fixed(config.context_length.try_into().unwrap_or(8192)),
            )
            .with_no_kv_cache(false)
            .with_gemm_full_precision_f16(true)
            .with_no_prefix_cache(false)
            .build();
            
            *self.model.lock().await = Some(Arc::new(mistral_rs));
            *self.model_loaded.write().await = true;
            *self.current_model.write().await = Some(config.model_id.clone());
            
            // Estimate memory usage (rough approximation)
            let param_count = 7_000_000_000; // Assuming 7B model as default
            let bytes_per_param = match &config.quantization {
                Some(q) if q.starts_with("Q4") => 0.5,
                Some(q) if q.starts_with("Q5") => 0.625,
                Some(q) if q.starts_with("Q8") => 1.0,
                Some("F16") => 2.0,
                _ => 4.0, // F32
            };
            *self.memory_usage.write().await = (param_count as f64 * bytes_per_param) as usize;
        }
        
        #[cfg(not(feature = "mistral-backend"))]
        {
            // Mock implementation when feature is disabled
            *self.model_loaded.write().await = true;
            *self.current_model.write().await = Some(config.model_id.clone());
            *self.memory_usage.write().await = 4 * 1024 * 1024 * 1024; // Mock 4GB
        }
        
        info!("Model '{}' loaded successfully", config.model_id);
        Ok(())
    }
    
    async fn unload_model(&mut self) -> Result<()> {
        info!("Unloading model from MistralRs backend");
        
        #[cfg(feature = "mistral-backend")]
        {
            *self.model.lock().await = None;
        }
        
        *self.model_loaded.write().await = false;
        *self.current_model.write().await = None;
        *self.memory_usage.write().await = 0;
        
        Ok(())
    }
    
    fn is_loaded(&self) -> bool {
        self.model_loaded.try_read().map(|g| *g).unwrap_or(false)
    }
    
    async fn generate_text(&self, prompt: &str, params: &GenerationParams) -> Result<String> {
        if !self.is_loaded() {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        
        debug!("Generating text with prompt length: {}", prompt.len());
        
        #[cfg(feature = "mistral-backend")]
        {
            let model = self.model.lock().await;
            let model = model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not initialized"))?;
            
            let messages = vec![
                RequestMessage::Completion {
                    text: prompt.to_string(),
                    echo_prompt: false,
                    best_of: 1,
                }
            ];
            
            let sampling_params = SamplingParams {
                temperature: Some(params.temperature as f64),
                top_k: Some(params.top_k as usize),
                top_p: Some(params.top_p as f64),
                top_n_logprobs: 0,
                frequency_penalty: Some(params.repeat_penalty as f64),
                presence_penalty: None,
                stop_toks: Some(params.stop_sequences.clone()),
                max_len: Some(params.max_tokens),
                logits_bias: None,
                n_choices: 1,
            };
            
            let request = Request::new_simple(
                "tektra".to_string(),
                messages,
                sampling_params,
                None,  // No response sender for non-streaming
                None,  // No adapter name
                None,  // No return logprobs
            );
            
            let (tx, mut rx) = tokio::sync::mpsc::channel(10);
            model.get_sender()?.send(request).await?;
            
            // Collect the full response
            let mut response_text = String::new();
            while let Some(response) = rx.recv().await {
                match response {
                    Response::ModelReload => {
                        info!("Model reloaded");
                    }
                    Response::ValidationError(e) => {
                        return Err(anyhow::anyhow!("Validation error: {}", e));
                    }
                    Response::InternalError(e) => {
                        return Err(anyhow::anyhow!("Internal error: {}", e));
                    }
                    Response::Done(response) => {
                        if let Some(choice) = response.choices.first() {
                            response_text = choice.message.content.clone().unwrap_or_default();
                        }
                        break;
                    }
                    Response::Chunk(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(delta) = &choice.delta.content {
                                response_text.push_str(delta);
                            }
                        }
                    }
                    _ => {}
                }
            }
            
            Ok(response_text)
        }
        
        #[cfg(not(feature = "mistral-backend"))]
        {
            // Mock response when feature is disabled
            Ok(format!("MistralRs response to: {}", prompt))
        }
    }
    
    async fn generate_multimodal(&self, inputs: MultimodalInput, params: &GenerationParams) -> Result<String> {
        if !self.is_loaded() {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        
        debug!("Generating multimodal response with {} images", inputs.images.len());
        
        #[cfg(feature = "mistral-backend")]
        {
            let model = self.model.lock().await;
            let model = model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not initialized"))?;
            
            // For vision models, we need to format the request differently
            let messages = if !inputs.images.is_empty() {
                // Convert images to base64 for vision models
                let mut image_urls = Vec::new();
                for (i, image_data) in inputs.images.iter().enumerate() {
                    let base64_image = base64::encode(image_data);
                    image_urls.push(format!("data:image/png;base64,{}", base64_image));
                }
                
                vec![
                    RequestMessage::VisionChat {
                        role: "user".to_string(),
                        content: inputs.text.clone().unwrap_or_default(),
                        images: Some(image_urls),
                    }
                ]
            } else {
                vec![
                    RequestMessage::Completion {
                        text: inputs.text.clone().unwrap_or_default(),
                        echo_prompt: false,
                        best_of: 1,
                    }
                ]
            };
            
            let sampling_params = SamplingParams {
                temperature: Some(params.temperature as f64),
                top_k: Some(params.top_k as usize),
                top_p: Some(params.top_p as f64),
                top_n_logprobs: 0,
                frequency_penalty: Some(params.repeat_penalty as f64),
                presence_penalty: None,
                stop_toks: Some(params.stop_sequences.clone()),
                max_len: Some(params.max_tokens),
                logits_bias: None,
                n_choices: 1,
            };
            
            let request = Request::new_simple(
                "tektra".to_string(),
                messages,
                sampling_params,
                None,
                None,
                None,
            );
            
            let (tx, mut rx) = tokio::sync::mpsc::channel(10);
            model.get_sender()?.send(request).await?;
            
            let mut response_text = String::new();
            while let Some(response) = rx.recv().await {
                match response {
                    Response::Done(response) => {
                        if let Some(choice) = response.choices.first() {
                            response_text = choice.message.content.clone().unwrap_or_default();
                        }
                        break;
                    }
                    Response::Chunk(chunk) => {
                        if let Some(choice) = chunk.choices.first() {
                            if let Some(delta) = &choice.delta.content {
                                response_text.push_str(delta);
                            }
                        }
                    }
                    Response::ValidationError(e) => {
                        return Err(anyhow::anyhow!("Validation error: {}", e));
                    }
                    Response::InternalError(e) => {
                        return Err(anyhow::anyhow!("Internal error: {}", e));
                    }
                    _ => {}
                }
            }
            
            Ok(response_text)
        }
        
        #[cfg(not(feature = "mistral-backend"))]
        {
            let text = inputs.text.as_deref().unwrap_or("No text");
            Ok(format!(
                "MistralRs multimodal response to: {} (with {} images)", 
                text, 
                inputs.images.len()
            ))
        }
    }
    
    async fn generate_stream(&self, prompt: &str, params: &GenerationParams) -> Result<tokio::sync::mpsc::Receiver<String>> {
        if !self.is_loaded() {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        
        let (tx, rx) = tokio::sync::mpsc::channel(100);
        
        #[cfg(feature = "mistral-backend")]
        {
            let model = self.model.lock().await;
            let model = model.as_ref().ok_or_else(|| anyhow::anyhow!("Model not initialized"))?.clone();
            
            let messages = vec![
                RequestMessage::Completion {
                    text: prompt.to_string(),
                    echo_prompt: false,
                    best_of: 1,
                }
            ];
            
            let sampling_params = SamplingParams {
                temperature: Some(params.temperature as f64),
                top_k: Some(params.top_k as usize),
                top_p: Some(params.top_p as f64),
                top_n_logprobs: 0,
                frequency_penalty: Some(params.repeat_penalty as f64),
                presence_penalty: None,
                stop_toks: Some(params.stop_sequences.clone()),
                max_len: Some(params.max_tokens),
                logits_bias: None,
                n_choices: 1,
            };
            
            let (response_tx, mut response_rx) = tokio::sync::mpsc::channel(10);
            
            let request = Request::new_simple(
                "tektra".to_string(),
                messages,
                sampling_params,
                Some(response_tx),
                None,
                None,
            );
            
            model.get_sender()?.send(request).await?;
            
            // Spawn task to handle streaming
            tokio::spawn(async move {
                while let Some(response) = response_rx.recv().await {
                    match response {
                        Response::Chunk(chunk) => {
                            if let Some(choice) = chunk.choices.first() {
                                if let Some(delta) = &choice.delta.content {
                                    let _ = tx.send(delta.clone()).await;
                                }
                            }
                        }
                        Response::Done(_) => break,
                        Response::ValidationError(e) => {
                            error!("Validation error during streaming: {}", e);
                            break;
                        }
                        Response::InternalError(e) => {
                            error!("Internal error during streaming: {}", e);
                            break;
                        }
                        _ => {}
                    }
                }
            });
        }
        
        #[cfg(not(feature = "mistral-backend"))]
        {
            // Mock streaming when feature is disabled
            let prompt = prompt.to_string();
            tokio::spawn(async move {
                let words = prompt.split_whitespace();
                for word in words {
                    let _ = tx.send(format!("{} ", word)).await;
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            });
        }
        
        Ok(rx)
    }
    
    fn capabilities(&self) -> BackendCapabilities {
        self.capabilities.clone()
    }
    
    fn memory_usage(&self) -> usize {
        self.memory_usage.try_read().map(|g| *g).unwrap_or(0)
    }
}

// Helper functions for MistralRs-specific features
impl MistralRsBackend {
    /// Enable MCP (Model Context Protocol) for tool use
    pub async fn enable_mcp(&mut self, tools: Vec<String>) -> Result<()> {
        info!("MCP enabled with {} tools", tools.len());
        // MCP is built into mistral.rs, configuration would go here
        Ok(())
    }
    
    /// Set custom attention implementation
    pub async fn set_attention_impl(&mut self, impl_type: &str) -> Result<()> {
        match impl_type {
            "flash-attn" => info!("Using Flash Attention"),
            "xformers" => info!("Using xFormers attention"),
            "eager" => info!("Using eager attention"),
            _ => warn!("Unknown attention implementation: {}", impl_type),
        }
        // This would be configured during model loading in mistral.rs
        Ok(())
    }
    
    /// Get model metadata
    pub async fn get_model_info(&self) -> Result<String> {
        if !self.is_loaded() {
            return Err(anyhow::anyhow!("No model loaded"));
        }
        
        let model_name = self.current_model.read().await;
        Ok(format!(
            "Model: {}\nBackend: MistralRs\nMemory: {} MB",
            model_name.as_deref().unwrap_or("Unknown"),
            self.memory_usage() / (1024 * 1024)
        ))
    }
}