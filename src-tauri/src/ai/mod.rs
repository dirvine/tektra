use anyhow::Result;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};
use serde_json::{json, Value};
use tokio::process::Command as AsyncCommand;

use crate::state::Message;

pub struct ModelManager {
    app_handle: AppHandle,
    model_path: PathBuf,
    current_model: Option<String>,
    bridge_script_path: PathBuf,
}

impl ModelManager {
    pub fn new(app_handle: AppHandle) -> Result<Self> {
        // Get model storage path
        let app_data_dir = app_handle.path().app_data_dir()?;
        let model_path = app_data_dir.join("models");
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
    
    pub async fn download_model(&self, model_name: &str, force: bool) -> Result<Value> {
        tracing::info!("Downloading model via MLX bridge: {}", model_name);
        
        let mut cmd = AsyncCommand::new("python3");
        cmd.arg(&self.bridge_script_path)
           .arg("--command")
           .arg("download")
           .arg("--model")
           .arg(model_name);
        
        if force {
            cmd.arg("--force");
        }
        
        let output = cmd.output().await?;
        
        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Failed to download model: {}", error));
        }
        
        let response: Value = serde_json::from_slice(&output.stdout)?;
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
            return Err(anyhow::anyhow!("Failed to load model: {}", error));
        }
        
        let response: Value = serde_json::from_slice(&output.stdout)?;
        
        if response["success"].as_bool().unwrap_or(false) {
            self.current_model = Some(model_name.to_string());
            tracing::info!("Model loaded successfully: {}", model_name);
            Ok(())
        } else {
            let error = response["error"].as_str().unwrap_or("Unknown error");
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