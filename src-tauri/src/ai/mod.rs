use anyhow::Result;
use std::path::PathBuf;
use tauri::{AppHandle, Manager, Emitter};
use serde_json::{json, Value};
use tokio::process::Command as AsyncCommand;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::state::Message;

pub struct ModelManager {
    app_handle: AppHandle,
    model_path: PathBuf,
    current_model: Option<String>,
    bridge_script_path: PathBuf,
}

impl ModelManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        // Use HuggingFace Hub cache directory for model storage
        let model_path = Self::get_hf_cache_dir()?;
        std::fs::create_dir_all(&model_path)?;
        
        // Get the bridge script path (in the src-tauri directory)
        let bridge_script_path = PathBuf::from("mlx_bridge.py");
        
        Ok(Self {
            app_handle,
            model_path,
            current_model: None,
            bridge_script_path,
        })
    }
    
    fn get_hf_cache_dir() -> Result<PathBuf> {
        // Use the same cache directory that HuggingFace Hub uses
        if let Ok(cache_dir) = std::env::var("HF_HOME") {
            return Ok(PathBuf::from(cache_dir));
        }
        
        if let Ok(xdg_cache) = std::env::var("XDG_CACHE_HOME") {
            return Ok(PathBuf::from(xdg_cache).join("huggingface"));
        }
        
        // Default HF cache location
        if let Some(home_dir) = dirs::home_dir() {
            return Ok(home_dir.join(".cache").join("huggingface"));
        }
        
        Err(anyhow::anyhow!("Could not determine HuggingFace cache directory"))
    }
    
    pub async fn download_model(&self, model_name: &str, force: bool) -> Result<Value> {
        tracing::info!("Downloading model via MLX bridge: {}", model_name);
        
        // Emit initial progress event
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 0,
            "status": "Initializing download...",
            "model_name": model_name
        }));
        
        let mut cmd = AsyncCommand::new("python3");
        cmd.arg(&self.bridge_script_path)
           .arg("--command")
           .arg("download")
           .arg("--model")
           .arg(model_name)
           .arg("--progress")  // Request progress output
           .stdout(Stdio::piped())
           .stderr(Stdio::piped());
        
        if force {
            cmd.arg("--force");
        }
        
        let mut child = cmd.spawn()?;
        
        // Handle progress updates from stdout
        if let Some(stdout) = child.stdout.take() {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            
            let app_handle = self.app_handle.clone();
            let model_name_clone = model_name.to_string();
            
            tokio::spawn(async move {
                while let Ok(Some(line)) = lines.next_line().await {
                    if line.starts_with("PROGRESS:") {
                        if let Ok(progress_data) = serde_json::from_str::<Value>(&line[9..]) {
                            let _ = app_handle.emit("model-loading-progress", json!({
                                "progress": progress_data["progress"],
                                "status": progress_data["status"],
                                "model_name": model_name_clone
                            }));
                        }
                    }
                }
            });
        }
        
        let output = child.wait_with_output().await?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error.to_string()
            }));
            return Err(anyhow::anyhow!("Failed to download model: {}", error));
        }
        
        // Parse the final response (excluding progress lines)
        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let response_lines: Vec<&str> = stdout_str.lines()
            .filter(|line| !line.starts_with("PROGRESS:"))
            .collect();
        
        let response_str = response_lines.join("\n");
        let response: Value = if response_str.trim().is_empty() {
            json!({"success": true})
        } else {
            serde_json::from_str(&response_str)?
        };
        
        // Emit completion event
        let _ = self.app_handle.emit("model-loading-complete", json!({
            "success": true
        }));
        
        Ok(response)
    }
    
    pub async fn list_cached_models(&self) -> Result<Value> {
        tracing::info!("Listing cached models via MLX bridge");
        
        let output = AsyncCommand::new("python3")
            .arg(&self.bridge_script_path)
            .arg("--command")
            .arg("list")
            .output()
            .await?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to list models: {}", error));
        }
        
        let response: Value = serde_json::from_slice(&output.stdout)?;
        Ok(response)
    }
    
    pub async fn load_model(&mut self, model_name: &str) -> Result<()> {
        tracing::info!("Loading model via MLX bridge: {}", model_name);
        
        // First, download the model if it's not already cached
        let _ = self.download_model(model_name, false).await?;
        
        // Emit loading progress
        let _ = self.app_handle.emit("model-loading-progress", json!({
            "progress": 90,
            "status": "Loading model into memory...",
            "model_name": model_name
        }));
        
        let output = AsyncCommand::new("python3")
            .arg(&self.bridge_script_path)
            .arg("--command")
            .arg("load")
            .arg("--model")
            .arg(model_name)
            .output()
            .await?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error.to_string()
            }));
            return Err(anyhow::anyhow!("Failed to load model: {}", error));
        }
        
        let response: Value = serde_json::from_slice(&output.stdout)?;
        
        if response["success"].as_bool().unwrap_or(false) {
            self.current_model = Some(model_name.to_string());
            tracing::info!("Model loaded successfully: {}", model_name);
            
            // Emit completion event
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": true
            }));
            
            Ok(())
        } else {
            let error = response["error"].as_str().unwrap_or("Unknown error");
            let _ = self.app_handle.emit("model-loading-complete", json!({
                "success": false,
                "error": error.to_string()
            }));
            Err(anyhow::anyhow!("MLX bridge error: {}", error))
        }
    }
    
    pub async fn generate_response(&self, prompt: &str, history: &[Message]) -> Result<String> {
        // Check if model is loaded
        if self.current_model.is_none() {
            return Err(anyhow::anyhow!("Model not loaded"));
        }
        
        // Format conversation with history
        let formatted_prompt = self.format_conversation(prompt, history);
        
        tracing::info!("Generating response via MLX bridge");
        
        let output = AsyncCommand::new("python3")
            .arg(&self.bridge_script_path)
            .arg("--command")
            .arg("generate")
            .arg("--prompt")
            .arg(&formatted_prompt)
            .arg("--max-tokens")
            .arg("512")
            .arg("--temperature")
            .arg("0.7")
            .output()
            .await?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to generate response: {}", error));
        }
        
        let response: Value = serde_json::from_slice(&output.stdout)?;
        
        if response["success"].as_bool().unwrap_or(false) {
            let generated_text = response["response"].as_str().unwrap_or("No response generated");
            Ok(generated_text.to_string())
        } else {
            let error = response["error"].as_str().unwrap_or("Unknown error");
            Err(anyhow::anyhow!("MLX bridge error: {}", error))
        }
    }
    
    pub async fn get_status(&self) -> serde_json::Value {
        // Get status from MLX bridge
        let bridge_status = match AsyncCommand::new("python3")
            .arg(&self.bridge_script_path)
            .arg("--command")
            .arg("status")
            .output()
            .await
        {
            Ok(output) if output.status.success() => {
                serde_json::from_slice(&output.stdout).unwrap_or_else(|_| json!({}))
            }
            _ => json!({})
        };
        
        // Combine with local status
        json!({
            "loaded": self.current_model.is_some(),
            "model": self.current_model.clone().unwrap_or_else(|| "None".to_string()),
            "device": "Apple Silicon (MLX Bridge)",
            "bridge_status": bridge_status
        })
    }
    
    fn format_conversation(&self, current_prompt: &str, history: &[Message]) -> String {
        let mut formatted = String::new();
        
        // Add recent conversation history (last 10 messages)
        for msg in history.iter().rev().take(10).rev() {
            formatted.push_str(&format!("{}: {}\n", msg.role, msg.content));
        }
        
        // Add current prompt
        formatted.push_str(&format!("user: {}\nassistant: ", current_prompt));
        
        formatted
    }
}