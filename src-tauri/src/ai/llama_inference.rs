use anyhow::Result;
use llama_cpp_rs::{
    LlamaContext, LlamaContextParams, LlamaModel, LlamaModelParams,
    SessionParams, TokenData, TokenDataArray,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tauri::{AppHandle, Manager};
use tokio::sync::Mutex;
use tracing::{error, info};

// TinyLlama model info
const MODEL_ID: &str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
const GGUF_MODEL_ID: &str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF";
const MODEL_FILE: &str = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoadProgress {
    pub progress: u32,
    pub status: String,
    pub model_name: String,
}

pub struct AIManager {
    app_handle: AppHandle,
    model: Option<Arc<Mutex<LlamaModel>>>,
    context: Option<Arc<Mutex<LlamaContext>>>,
    model_path: Option<PathBuf>,
}

impl AIManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        Ok(Self {
            app_handle,
            model: None,
            context: None,
            model_path: None,
        })
    }

    pub async fn load_model(&mut self) -> Result<()> {
        self.emit_progress(0, "Starting TinyLlama initialization...", MODEL_ID).await;
        
        // Download model if needed
        let model_path = self.download_model().await?;
        self.model_path = Some(model_path.clone());
        
        self.emit_progress(80, "Loading TinyLlama into memory...", MODEL_ID).await;
        
        // Initialize model parameters
        let model_params = LlamaModelParams {
            n_gpu_layers: 32,  // Use Metal/GPU for all layers
            main_gpu: 0,
            tensor_split: None,
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
        };
        
        // Load the model
        let model = LlamaModel::load_from_file(
            model_path.to_str().unwrap(),
            model_params
        ).map_err(|e| anyhow::anyhow!("Failed to load model: {:?}", e))?;
        
        // Create context parameters
        let ctx_params = LlamaContextParams {
            n_ctx: 2048,      // Context window
            n_batch: 512,     // Batch size for prompt processing
            n_threads: 4,     // CPU threads
            n_threads_batch: 4,
            rope_scaling_type: 0,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: 0.0,
            yarn_attn_factor: 0.0,
            yarn_beta_fast: 0.0,
            yarn_beta_slow: 0.0,
            yarn_orig_ctx: 0,
            cb_eval: None,
            cb_eval_user_data: std::ptr::null_mut(),
            type_k: 0,
            type_v: 0,
            logits_all: false,
            embeddings: false,
            offload_kqv: true,
        };
        
        // Create context
        let context = model.new_context(ctx_params)
            .map_err(|e| anyhow::anyhow!("Failed to create context: {:?}", e))?;
        
        self.model = Some(Arc::new(Mutex::new(model)));
        self.context = Some(Arc::new(Mutex::new(context)));
        
        self.emit_progress(100, "TinyLlama fully loaded and ready!", MODEL_ID).await;
        info!("TinyLlama model loaded successfully with Metal acceleration");
        
        Ok(())
    }

    pub async fn generate_response(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let model = self.model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        let context = self.context.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Context not loaded"))?;
        
        // Format prompt for TinyLlama chat template
        let formatted_prompt = format!(
            "<|system|>\nYou are a helpful AI assistant named Tektra.</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            prompt
        );
        
        info!("Generating response for prompt: {}", prompt);
        
        let mut model_guard = model.lock().await;
        let mut context_guard = context.lock().await;
        
        // Tokenize the prompt
        let tokens = model_guard.tokenize(&formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {:?}", e))?;
        
        // Clear context and evaluate prompt
        context_guard.clear();
        context_guard.eval(&model_guard, &tokens, 1)
            .map_err(|e| anyhow::anyhow!("Prompt evaluation failed: {:?}", e))?;
        
        // Generate response tokens
        let mut response_tokens = Vec::new();
        let mut response_text = String::new();
        
        for _ in 0..max_tokens {
            // Sample next token
            let candidates = context_guard.candidates();
            
            // Apply temperature and top-p sampling
            let mut candidates_array = TokenDataArray::from_iter(
                candidates.iter().enumerate().map(|(id, &logit)| {
                    TokenData {
                        id: id as i32,
                        logit,
                        p: 0.0,
                    }
                })
            );
            
            // Sample with temperature
            candidates_array.sample_temperature(0.7);
            candidates_array.sample_top_p(0.9, 1);
            
            let token_id = candidates_array.sample_token(&mut rand::thread_rng());
            response_tokens.push(token_id);
            
            // Check for end token
            if token_id == model_guard.token_eos() {
                break;
            }
            
            // Decode token to text
            let token_str = model_guard.token_to_str(token_id)
                .unwrap_or_else(|_| String::new());
            response_text.push_str(&token_str);
            
            // Evaluate the new token
            context_guard.eval(&model_guard, &[token_id], 1)
                .map_err(|e| anyhow::anyhow!("Token evaluation failed: {:?}", e))?;
            
            // Check for natural stopping points
            if response_text.ends_with("</s>") || 
               response_text.ends_with("\n\nUser:") || 
               response_text.ends_with("\n\nHuman:") {
                break;
            }
        }
        
        // Clean up response
        let response = response_text
            .trim()
            .replace("</s>", "")
            .replace("<|assistant|>", "")
            .trim()
            .to_string();
        
        info!("Generated response: {}", response);
        
        Ok(response)
    }

    async fn download_model(&self) -> Result<PathBuf> {
        self.emit_progress(10, "Checking for TinyLlama model...", MODEL_ID).await;
        
        // Get cache directory
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?
            .join("huggingface")
            .join("hub")
            .join(GGUF_MODEL_ID.replace('/', "--"));
        
        std::fs::create_dir_all(&cache_dir)?;
        
        let model_path = cache_dir.join(MODEL_FILE);
        
        if model_path.exists() {
            let metadata = std::fs::metadata(&model_path)?;
            if metadata.len() > 100_000_000 {
                self.emit_progress(70, "Found cached TinyLlama model", MODEL_ID).await;
                return Ok(model_path);
            } else {
                let _ = std::fs::remove_file(&model_path);
            }
        }
        
        // Download the model
        self.emit_progress(20, "Downloading TinyLlama Q4 model (669MB)...", MODEL_ID).await;
        
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            GGUF_MODEL_ID, MODEL_FILE
        );
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .user_agent("Tektra-AI-Assistant/0.1.0")
            .build()?;
        
        let response = client.get(&url).send().await?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download model: HTTP {}",
                response.status()
            ));
        }
        
        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        
        let temp_path = model_path.with_extension("tmp");
        let mut file = tokio::fs::File::create(&temp_path).await?;
        let mut stream = response.bytes_stream();
        
        use futures::StreamExt;
        use tokio::io::AsyncWriteExt;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            
            if total_size > 0 {
                let progress = 20 + ((downloaded as f64 / total_size as f64) * 50.0) as u32;
                self.emit_progress(
                    progress,
                    &format!(
                        "Downloading TinyLlama ({} / {})",
                        bytesize::ByteSize(downloaded),
                        bytesize::ByteSize(total_size)
                    ),
                    MODEL_ID,
                ).await;
            }
        }
        
        file.flush().await?;
        drop(file);
        
        tokio::fs::rename(&temp_path, &model_path).await?;
        
        Ok(model_path)
    }

    async fn emit_progress(&self, progress: u32, status: &str, model_name: &str) {
        let progress_data = ModelLoadProgress {
            progress,
            status: status.to_string(),
            model_name: model_name.to_string(),
        };
        
        if let Err(e) = self.app_handle.emit_all("model-loading-progress", &progress_data) {
            error!("Failed to emit progress: {}", e);
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.model.is_some() && self.context.is_some()
    }
}