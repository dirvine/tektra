use anyhow::Result;
use futures::StreamExt;
use hf_hub::api::tokio::Api;
use hf_hub::{Repo, RepoType};
use std::io::Write;
use std::path::PathBuf;
use tauri::AppHandle;
use tracing::info;

pub struct ModelFiles {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config_path: PathBuf,
}

pub async fn download_model_files(app_handle: &AppHandle, model_id: &str, revision: &str) -> Result<ModelFiles> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));
    
    // Get cache directory
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Failed to get cache directory"))?
        .join("huggingface")
        .join("hub")
        .join(model_id.replace('/', "--"));
    
    std::fs::create_dir_all(&cache_dir)?;
    
    // Files to download
    let files = vec![
        ("model.safetensors", "model.safetensors"),
        ("tokenizer.json", "tokenizer.json"),
        ("config.json", "config.json"),
    ];
    
    let mut model_files = ModelFiles {
        model_path: PathBuf::new(),
        tokenizer_path: PathBuf::new(),
        config_path: PathBuf::new(),
    };
    
    let total_files = files.len();
    
    for (idx, (remote_file, local_file)) in files.iter().enumerate() {
        let local_path = cache_dir.join(local_file);
        
        // Check if file already exists
        if local_path.exists() {
            info!("File {} already exists, skipping download", local_file);
            match *local_file {
                "model.safetensors" => model_files.model_path = local_path,
                "tokenizer.json" => model_files.tokenizer_path = local_path,
                "config.json" => model_files.config_path = local_path,
                _ => {}
            }
            
            let progress = ((idx + 1) as f32 / total_files as f32 * 80.0) as u32;
            emit_download_progress(app_handle, progress, &format!("Found cached {}", local_file), model_id).await;
            continue;
        }
        
        // Download file
        info!("Downloading {} from HuggingFace", remote_file);
        let base_progress = (idx as f32 / total_files as f32 * 80.0) as u32;
        
        download_file_with_progress(
            app_handle,
            &repo,
            remote_file,
            &local_path,
            model_id,
            base_progress,
            80.0 / total_files as f32,
        ).await?;
        
        match *local_file {
            "model.safetensors" => model_files.model_path = local_path,
            "tokenizer.json" => model_files.tokenizer_path = local_path,
            "config.json" => model_files.config_path = local_path,
            _ => {}
        }
    }
    
    Ok(model_files)
}

async fn download_file_with_progress(
    app_handle: &AppHandle,
    repo: &hf_hub::api::tokio::RepoInfo,
    filename: &str,
    local_path: &PathBuf,
    model_name: &str,
    base_progress: u32,
    progress_weight: f32,
) -> Result<()> {
    // Get download URL
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo.repo_id(),
        filename
    );
    
    // Start download
    let client = reqwest::Client::new();
    let response = client.get(&url).send().await?;
    let total_size = response.content_length().unwrap_or(0);
    
    // Create file
    let mut file = std::fs::File::create(local_path)?;
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            let file_progress = (downloaded as f32 / total_size as f32 * progress_weight) as u32;
            let total_progress = base_progress + file_progress;
            
            emit_download_progress(
                app_handle,
                total_progress,
                &format!(
                    "Downloading {} ({:.1} MB / {:.1} MB)",
                    filename,
                    downloaded as f32 / 1_048_576.0,
                    total_size as f32 / 1_048_576.0
                ),
                model_name,
            ).await;
        }
    }
    
    Ok(())
}

async fn emit_download_progress(app_handle: &AppHandle, progress: u32, status: &str, model_name: &str) {
    let _ = app_handle.emit("model-loading-progress", serde_json::json!({
        "progress": progress,
        "status": status,
        "model_name": model_name
    }));
}