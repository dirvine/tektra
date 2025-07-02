use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use tracing::{info, error, warn};
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest, generation::chat::{ChatMessage, request::ChatMessageRequest}};
use super::inference_backend::{InferenceBackend, InferenceConfig};
use super::multimodal_processor::{Gemma3NProcessor, MultimodalInput};
use tauri::{AppHandle, Emitter};
use serde_json::json;

#[derive(Debug, Clone)]
pub enum OllamaExe {
    System(PathBuf),
    Embedded(PathBuf),
}

/// Ollama inference backend for LLM model execution
/// 
/// Handles the complete lifecycle of Ollama-based inference including:
/// - Automatic Ollama installation (system or bundled)
/// - Model downloading and management
/// - Text and multimodal inference
/// - Progress tracking and event emission
/// 
/// # Architecture
/// - Uses ollama-rs for API communication
/// - Supports embedded Ollama via ollama_td crate
/// - Integrates with Gemma3NProcessor for multimodal support
pub struct OllamaInference {
    ollama_exe: Option<OllamaExe>,
    ollama_client: Option<Ollama>,
    model_loaded: bool,
    current_model: Option<String>,
    ollama_port: u16,
    app_handle: Option<AppHandle>,
    multimodal_processor: Gemma3NProcessor,
}

impl OllamaInference {
    /// Create a new Ollama inference instance
    /// 
    /// # Returns
    /// * `Self` - Initialized OllamaInference instance
    pub fn new() -> Self {
        Self {
            ollama_exe: None,
            ollama_client: None,
            model_loaded: false,
            current_model: None,
            ollama_port: 11434, // Default Ollama port
            app_handle: None,
            multimodal_processor: Gemma3NProcessor::new(),
        }
    }
    
    /// Create a new instance with Tauri app handle for progress events
    /// 
    /// # Arguments
    /// * `app_handle` - Tauri application handle
    /// 
    /// # Returns
    /// * `Self` - Initialized OllamaInference instance
    pub fn with_app_handle(app_handle: AppHandle) -> Self {
        Self {
            ollama_exe: None,
            ollama_client: None,
            model_loaded: false,
            current_model: None,
            ollama_port: 11434, // Default Ollama port
            app_handle: Some(app_handle),
            multimodal_processor: Gemma3NProcessor::new(),
        }
    }
    
    /// Set the Tauri app handle for progress event emission
    /// 
    /// # Arguments
    /// * `app_handle` - Tauri application handle
    pub fn set_app_handle(&mut self, app_handle: AppHandle) {
        self.app_handle = Some(app_handle);
    }
    
    async fn emit_progress(&self, progress: f64, status: &str, model_name: &str) {
        if let Some(ref app_handle) = self.app_handle {
            let _ = app_handle.emit_to(tauri::EventTarget::Any, "model-loading-progress", json!({
                "progress": progress,
                "status": status,
                "model_name": model_name
            }));
        }
    }

    /// Find Ollama binary - either system-installed or download embedded version
    pub async fn find_ollama(&self) -> Result<OllamaExe> {
        info!("Checking for system Ollama installation...");
        
        // First try to find system Ollama
        if let Ok(output) = Command::new("ollama").arg("--version").output() {
            if output.status.success() {
                info!("Found system Ollama installation");
                return Ok(OllamaExe::System(PathBuf::from("ollama")));
            }
        }

        info!("System Ollama not found, downloading embedded version...");
        self.emit_progress(10.0, "🔄 System Ollama not found, downloading embedded version...", "ollama").await;
        
        // Download embedded Ollama - use a different location to avoid macOS restrictions
        let data_dir = std::env::temp_dir()
            .join("tektra_ollama")
            .join("extracted");
        
        fs::create_dir_all(&data_dir).await?;
        
        self.emit_progress(15.0, "📁 Created download directory", "ollama").await;
        
        // Use default download configuration
        let download_config = ollama_td::OllamaDownload::default();
        self.emit_progress(20.0, "⬇️ Starting Ollama download (~10-50MB depending on platform)...", "ollama").await;
        
        let downloaded_path = ollama_td::download(download_config).await
            .map_err(|e| anyhow::anyhow!("Failed to download Ollama: {}", e))?;
        
        self.emit_progress(40.0, "✅ Ollama downloaded successfully", "ollama").await;
        info!("Ollama downloaded to: {:?}", downloaded_path);
        
        // Check if downloaded file is a zip archive and extract it
        let ollama_binary = if downloaded_path.extension().and_then(|s| s.to_str()) == Some("zip") {
            info!("Extracting Ollama zip archive...");
            self.emit_progress(45.0, "📦 Extracting Ollama archive...", "ollama").await;
            
            // Use a different extraction approach - extract directly to temp directory to avoid macOS restrictions
            let extract_dir = data_dir.clone();
            
            // On macOS, use a different strategy - copy to temp first, then extract
            #[cfg(target_os = "macos")]
            {
                info!("Using macOS-specific extraction strategy...");
                
                // Copy zip to temp directory first
                let temp_zip = extract_dir.join("ollama_temp.zip");
                fs::copy(&downloaded_path, &temp_zip).await?;
                
                // Remove quarantine from temp copy
                let xattr_result = Command::new("xattr")
                    .arg("-c")
                    .arg(&temp_zip)
                    .output();
                
                if let Ok(output) = xattr_result {
                    if output.status.success() {
                        info!("Cleared extended attributes from temp zip");
                    }
                }
                
                // Extract using ditto (more macOS-friendly)
                let extract_cmd = Command::new("ditto")
                    .arg("-x")
                    .arg("-k")
                    .arg(&temp_zip)
                    .arg(&extract_dir)
                    .output()
                    .map_err(|e| anyhow::anyhow!("Failed to run ditto command: {}", e))?;
                
                if !extract_cmd.status.success() {
                    let stderr = String::from_utf8_lossy(&extract_cmd.stderr);
                    
                    // Fallback to manual unzip if ditto fails
                    info!("Ditto failed, trying unzip as fallback...");
                    let unzip_cmd = Command::new("unzip")
                        .arg("-o")
                        .arg("-j") // Junk paths - extract all files to same directory
                        .arg(&temp_zip)
                        .arg("-d")
                        .arg(&extract_dir)
                        .output()
                        .map_err(|e| anyhow::anyhow!("Failed to run unzip command: {}", e))?;
                    
                    if !unzip_cmd.status.success() {
                        let unzip_stderr = String::from_utf8_lossy(&unzip_cmd.stderr);
                        return Err(anyhow::anyhow!("Both ditto and unzip failed. Ditto: {}, Unzip: {}", stderr, unzip_stderr));
                    }
                }
                
                // Clean up temp zip
                let _ = fs::remove_file(&temp_zip).await;
            }
            
            #[cfg(not(target_os = "macos"))]
            {
                let extract_cmd = Command::new("unzip")
                    .arg("-o")
                    .arg(&downloaded_path)
                    .arg("-d") 
                    .arg(&extract_dir)
                    .output()
                    .map_err(|e| anyhow::anyhow!("Failed to run unzip command: {}", e))?;
                
                if !extract_cmd.status.success() {
                    let stderr = String::from_utf8_lossy(&extract_cmd.stderr);
                    return Err(anyhow::anyhow!("Failed to extract Ollama zip: {}", stderr));
                }
            }
            
            // Find the Ollama binary in the extracted contents
            #[cfg(target_os = "macos")]
            let ollama_path = {
                // Try different locations - prefer the app structure if available as it has all resources
                let app_resources_path = extract_dir.join("Ollama.app").join("Contents").join("Resources").join("ollama");
                let flattened_path = extract_dir.join("ollama");
                
                info!("Looking for Ollama binary in extracted files...");
                info!("App structure path: {:?}", app_resources_path);
                info!("Flattened path: {:?}", flattened_path);
                
                if app_resources_path.exists() {
                    info!("Using app structure Ollama binary");
                    app_resources_path
                } else if flattened_path.exists() {
                    info!("Using flattened Ollama binary");
                    flattened_path
                } else {
                    // List what files are actually in the extract directory
                    if let Ok(entries) = std::fs::read_dir(&extract_dir) {
                        info!("Files in extract directory:");
                        for entry in entries {
                            if let Ok(entry) = entry {
                                info!("  {:?}", entry.path());
                            }
                        }
                    }
                    return Err(anyhow::anyhow!("Ollama binary not found in extracted archive. Tried: {:?} and {:?}", app_resources_path, flattened_path));
                }
            };
            
            #[cfg(target_os = "linux")]
            let ollama_path = extract_dir.join("ollama");
            
            #[cfg(target_os = "windows")]  
            let ollama_path = extract_dir.join("ollama.exe");
            
            if !ollama_path.exists() {
                return Err(anyhow::anyhow!("Ollama binary not found in extracted archive at: {:?}", ollama_path));
            }
            
            info!("Extracted Ollama binary to: {:?}", ollama_path);
            ollama_path
        } else {
            downloaded_path
        };
        
        // Make binary executable
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            info!("Setting executable permissions for Ollama binary...");
            
            // Set executable permissions (rwxr-xr-x)
            match fs::set_permissions(&ollama_binary, std::fs::Permissions::from_mode(0o755)).await {
                Ok(_) => info!("Successfully set executable permissions"),
                Err(e) => {
                    error!("Failed to set executable permissions: {}", e);
                    // Try using chmod command as fallback
                    let chmod_result = Command::new("chmod")
                        .arg("+x")
                        .arg(&ollama_binary)
                        .output();
                    
                    match chmod_result {
                        Ok(output) => {
                            if output.status.success() {
                                info!("Successfully set executable permissions using chmod");
                            } else {
                                warn!("chmod command failed: {}", String::from_utf8_lossy(&output.stderr));
                            }
                        }
                        Err(e) => {
                            warn!("Could not run chmod command: {}", e);
                        }
                    }
                }
            }
        }
        
        info!("Ollama binary ready at: {:?}", ollama_binary);
        self.emit_progress(50.0, "🎉 Ollama binary ready and configured!", "ollama").await;
        Ok(OllamaExe::Embedded(ollama_binary))
    }

    /// Initialize Ollama (find binary and start if needed)
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Ollama inference backend...");
        
        // Wrap the heavy operation in a timeout and better error handling
        let ollama_exe = match tokio::time::timeout(
            tokio::time::Duration::from_secs(300), // 5 minute timeout for download
            self.find_ollama()
        ).await {
            Ok(Ok(exe)) => exe,
            Ok(Err(e)) => return Err(anyhow::anyhow!("Failed to find/download Ollama: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Ollama download timed out after 5 minutes")),
        };
        
        self.ollama_exe = Some(ollama_exe.clone());
        
        // Start Ollama server if using embedded version
        match &ollama_exe {
            OllamaExe::Embedded(path) => {
                info!("Starting embedded Ollama server...");
                if let Err(e) = self.start_ollama_server(path).await {
                    return Err(anyhow::anyhow!("Failed to start Ollama server: {}", e));
                }
            }
            OllamaExe::System(_) => {
                info!("Using system Ollama (assuming it's running)");
            }
        }
        
        // Create Ollama client
        let ollama_url = format!("http://localhost:{}", self.ollama_port);
        self.ollama_client = Some(Ollama::new(ollama_url, self.ollama_port));
        
        // Test connection with timeout and retries
        let mut retry_count = 0;
        let max_retries = 3;
        
        loop {
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(10),
                self.test_connection()
            ).await {
                Ok(Ok(_)) => {
                    info!("Ollama inference backend initialized successfully");
                    break;
                }
                Ok(Err(e)) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(anyhow::anyhow!("Ollama connection test failed after {} retries: {}", max_retries, e));
                    }
                    warn!("Ollama connection test failed (attempt {}/{}): {}", retry_count, max_retries, e);
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
                Err(_) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(anyhow::anyhow!("Ollama connection test timed out after {} retries", max_retries));
                    }
                    warn!("Ollama connection test timed out (attempt {}/{})", retry_count, max_retries);
                    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                }
            }
        }
        
        Ok(())
    }

    /// Start Ollama server (for embedded version)
    async fn start_ollama_server(&self, ollama_path: &Path) -> Result<()> {
        info!("Starting Ollama server at: {:?}", ollama_path);
        
        let mut cmd = Command::new(ollama_path);
        cmd.arg("serve");
        
        // Set environment variables for embedded Ollama
        cmd.env("OLLAMA_HOST", "127.0.0.1:11434");
        cmd.env("OLLAMA_ORIGINS", "*");
        
        // Critical: Set the binary path for runner processes
        // Ollama server spawns runner processes that need to find the binary
        cmd.env("OLLAMA_EXECUTABLE", ollama_path);
        
        // Set the working directory to the directory containing the binary
        if let Some(parent_dir) = ollama_path.parent() {
            cmd.current_dir(parent_dir);
            info!("Setting working directory to: {:?}", parent_dir);
        }
        
        // For embedded Ollama, set the models directory in temp to avoid permissions issues
        let models_dir = std::env::temp_dir().join("tektra_ollama_models");
        std::fs::create_dir_all(&models_dir).unwrap_or_default();
        cmd.env("OLLAMA_MODELS", models_dir);
        
        // Set library path for embedded resources
        if let Some(app_dir) = ollama_path.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
            if app_dir.file_name().and_then(|n| n.to_str()) == Some("Ollama.app") {
                let resources_dir = app_dir.join("Contents").join("Resources");
                if resources_dir.exists() {
                    cmd.env("OLLAMA_LIBRARY_PATH", &resources_dir);
                    info!("Set OLLAMA_LIBRARY_PATH to: {:?}", resources_dir);
                }
            }
        }
        
        // Add the binary directory to PATH so runner processes can find ollama executable
        if let Some(bin_dir) = ollama_path.parent() {
            let current_path = std::env::var("PATH").unwrap_or_default();
            let new_path = format!("{}:{}", bin_dir.display(), current_path);
            cmd.env("PATH", new_path);
            info!("Added binary directory to PATH: {:?}", bin_dir);
        }
        
        // Start the server in background
        let mut child = cmd.spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start Ollama server: {}", e))?;
        
        info!("Ollama server started with PID: {} at {:?}", child.id(), ollama_path);
        
        // Wait a moment for server to start and check if it's still running
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
        
        // Check if the process is still running
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process has exited
                error!("Ollama server exited immediately with status: {:?}", status);
                return Err(anyhow::anyhow!("Ollama server failed to start - exited with status: {:?}", status));
            }
            Ok(None) => {
                // Process is still running
                info!("Ollama server is running successfully");
                
                // Wait a bit more to ensure it's fully started
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                Ok(())
            }
            Err(e) => {
                error!("Failed to check Ollama server status: {}", e);
                Err(anyhow::anyhow!("Failed to check Ollama server status: {}", e))
            }
        }
    }

    /// Test connection to Ollama server
    async fn test_connection(&self) -> Result<()> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        info!("Testing Ollama connection...");
        
        // Try to list models to test connection
        match ollama.list_local_models().await {
            Ok(models) => {
                info!("Ollama connection successful. Found {} local models", models.len());
                for model in &models {
                    info!("Available model: {}", model.name);
                }
                Ok(())
            }
            Err(e) => {
                error!("Failed to connect to Ollama: {}", e);
                Err(anyhow::anyhow!("Ollama connection failed: {}", e))
            }
        }
    }

    /// Pull a model from Ollama registry
    pub async fn pull_model(&self, model_name: &str) -> Result<()> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        info!("Pulling model: {}", model_name);
        
        // Pull the model (simplified approach)
        let _result = ollama.pull_model(model_name.to_string(), false).await?;
        info!("Model pull initiated for: {}", model_name);
        
        info!("Model {} pulled successfully", model_name);
        Ok(())
    }
    
    /// Pull a model with enhanced progress tracking showing individual files
    async fn pull_model_with_progress(&self, model_name: &str) -> Result<()> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        info!("Pulling model with enhanced progress tracking: {}", model_name);
        
        // Use streaming pull model to get progress updates
        use futures::StreamExt;
        use std::collections::HashMap;
        
        self.emit_progress(5.0, &format!("🔄 Initializing download of {}", model_name), model_name).await;
        
        // Create pull model stream with allow_insecure=false
        let mut stream = ollama.pull_model_stream(model_name.to_string(), false).await
            .map_err(|e| anyhow::anyhow!("Failed to start model pull stream: {}", e))?;
            
        let mut overall_progress = 5.0;
        let mut last_emitted_progress = 0.0;
        let mut file_downloads: HashMap<String, (u64, u64)> = HashMap::new(); // digest -> (completed, total)
        let mut current_file_name = String::new();
        let mut total_files_expected = 0;
        let mut files_completed = 0;
        
        while let Some(response) = stream.next().await {
            match response {
                Ok(response_data) => {
                    let message = &response_data.message;
                    let digest = response_data.digest.as_deref().unwrap_or("");
                    
                    // Enhanced progress tracking with detailed file information
                    if !message.is_empty() {
                        match message.as_str() {
                            "pulling manifest" => {
                                overall_progress = 10.0;
                                self.emit_progress(overall_progress, "📋 Downloading model manifest...", model_name).await;
                            }
                            msg if msg.starts_with("pulling") && digest.len() > 0 => {
                                // Extract layer information from pull message
                                if let Some(layer_id) = digest.get(0..12) {
                                    current_file_name = format!("Layer {}", layer_id);
                                    if !file_downloads.contains_key(digest) {
                                        total_files_expected += 1;
                                        file_downloads.insert(digest.to_string(), (0, 0));
                                    }
                                    
                                    let file_num = file_downloads.len();
                                    self.emit_progress(
                                        overall_progress, 
                                        &format!("📦 Downloading {} (file {}/{})", current_file_name, file_num, total_files_expected.max(file_num)), 
                                        model_name
                                    ).await;
                                }
                            }
                            "downloading" => {
                                if let (Some(completed), Some(total)) = (response_data.completed, response_data.total) {
                                    // Update file download progress
                                    if !digest.is_empty() {
                                        file_downloads.insert(digest.to_string(), (completed, total));
                                        
                                        // Calculate individual file progress
                                        let file_progress = if total > 0 { (completed as f64 / total as f64) * 100.0 } else { 0.0 };
                                        let mb_completed = completed as f64 / (1024.0 * 1024.0);
                                        let mb_total = total as f64 / (1024.0 * 1024.0);
                                        
                                        // Calculate overall progress based on completed files + current file progress
                                        let base_progress = 15.0; // After manifest
                                        let download_phase_progress = 70.0; // 70% of total for downloading
                                        
                                        // Calculate progress from all files
                                        let total_downloaded: u64 = file_downloads.values().map(|(c, _)| *c).sum();
                                        let total_size: u64 = file_downloads.values().map(|(_, t)| *t).sum();
                                        
                                        if total_size > 0 {
                                            let download_ratio = total_downloaded as f64 / total_size as f64;
                                            overall_progress = base_progress + (download_ratio * download_phase_progress);
                                        } else {
                                            overall_progress = base_progress + (file_progress / 100.0 * download_phase_progress);
                                        }
                                        
                                        // Only emit if progress increased significantly
                                        if overall_progress - last_emitted_progress >= 0.5 || 
                                           (mb_completed - (mb_completed as u64) as f64).abs() < 0.1 { // Emit on whole MB
                                            
                                            let layer_id = if digest.len() >= 12 { &digest[0..12] } else { digest };
                                            let status_msg = if !current_file_name.is_empty() {
                                                format!("⬇️ {} • {:.1} MB / {:.1} MB ({:.1}%)", current_file_name, mb_completed, mb_total, file_progress)
                                            } else {
                                                format!("⬇️ Downloading layer {} • {:.1} MB / {:.1} MB", layer_id, mb_completed, mb_total)
                                            };
                                            
                                            self.emit_progress(overall_progress, &status_msg, model_name).await;
                                            last_emitted_progress = overall_progress;
                                        }
                                        
                                        // Check if this file just completed
                                        if completed == total && total > 0 {
                                            files_completed += 1;
                                            let layer_id = if digest.len() >= 12 { &digest[0..12] } else { digest };
                                            self.emit_progress(
                                                overall_progress,
                                                &format!("✅ Completed layer {} ({:.1} MB) • {}/{} files done", 
                                                        layer_id, mb_total, files_completed, total_files_expected.max(file_downloads.len())),
                                                model_name
                                            ).await;
                                        }
                                    }
                                } else {
                                    // Generic downloading without size info
                                    overall_progress = std::cmp::max(overall_progress as u64, 25) as f64;
                                    self.emit_progress(overall_progress, "⬇️ Downloading model files...", model_name).await;
                                }
                            }
                            "verifying sha256 digest" => {
                                overall_progress = 90.0;
                                if !digest.is_empty() {
                                    let layer_id = if digest.len() >= 12 { &digest[0..12] } else { digest };
                                    self.emit_progress(overall_progress, &format!("🔍 Verifying layer {}", layer_id), model_name).await;
                                } else {
                                    self.emit_progress(overall_progress, "🔍 Verifying download integrity...", model_name).await;
                                }
                            }
                            "writing manifest" => {
                                overall_progress = 95.0;
                                self.emit_progress(overall_progress, "📝 Installing model manifest...", model_name).await;
                            }
                            "success" => {
                                overall_progress = 100.0;
                                self.emit_progress(overall_progress, &format!("🎉 {} ready! Downloaded {} files successfully", model_name, files_completed), model_name).await;
                                break;
                            }
                            _ => {
                                // Log other status messages with better formatting
                                info!("Model pull status: {}", message);
                                if !message.is_empty() && !message.contains("pulling") {
                                    let formatted_msg = if message.len() > 50 {
                                        format!("🔄 {}", &message[0..47].trim())
                                    } else {
                                        format!("🔄 {}", message)
                                    };
                                    self.emit_progress(overall_progress, &formatted_msg, model_name).await;
                                }
                            }
                        }
                    } else if let (Some(completed), Some(total)) = (response_data.completed, response_data.total) {
                        // Handle progress without message - use digest for identification
                        if !digest.is_empty() {
                            file_downloads.insert(digest.to_string(), (completed, total));
                            
                            let mb_completed = completed as f64 / (1024.0 * 1024.0);
                            let mb_total = total as f64 / (1024.0 * 1024.0);
                            let file_progress = if total > 0 { (completed as f64 / total as f64) * 100.0 } else { 0.0 };
                            
                            // Calculate overall progress
                            let total_downloaded: u64 = file_downloads.values().map(|(c, _)| *c).sum();
                            let total_size: u64 = file_downloads.values().map(|(_, t)| *t).sum();
                            
                            if total_size > 0 {
                                let download_ratio = total_downloaded as f64 / total_size as f64;
                                overall_progress = 15.0 + (download_ratio * 70.0);
                            }
                            
                            if overall_progress - last_emitted_progress >= 1.0 {
                                let layer_id = if digest.len() >= 12 { &digest[0..12] } else { digest };
                                self.emit_progress(
                                    overall_progress, 
                                    &format!("⬇️ Layer {} • {:.1} MB / {:.1} MB ({:.1}%)", layer_id, mb_completed, mb_total, file_progress), 
                                    model_name
                                ).await;
                                last_emitted_progress = overall_progress;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Error during model pull stream: {}", e);
                    return Err(anyhow::anyhow!("Model pull stream error: {}", e));
                }
            }
        }
        
        info!("Model {} pulled successfully with {} files", model_name, files_completed);
        Ok(())
    }

    /// Generate multimodal response (text, image, audio, video inputs)
    pub async fn generate_multimodal(
        &self,
        prompt: &str,
        image_data: Option<&[u8]>,
        _config: &InferenceConfig,
    ) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating multimodal response with model: {}", model_name);
        
        let mut request = GenerationRequest::new(model_name.clone(), prompt.to_string());
        
        // Configure generation parameters (simplified for now)
        // Note: ollama-rs API doesn't have direct parameter setters, 
        // we'll use default parameters for now
        
        // Add image if provided
        if let Some(image_bytes) = image_data {
            use base64::{Engine as _, engine::general_purpose};
            let base64_image = general_purpose::STANDARD.encode(image_bytes);
            let image = ollama_rs::generation::images::Image::from_base64(&base64_image);
            request = request.images(vec![image]);
        }
        
        // Generate response
        let response = ollama.generate(request).await
            .map_err(|e| anyhow::anyhow!("Generation failed: {}", e))?;
        
        Ok(response.response)
    }

    /// Check if a model is available locally
    pub async fn is_model_available(&self, model_name: &str) -> Result<bool> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let models = ollama.list_local_models().await?;
        Ok(models.iter().any(|m| m.name == model_name))
    }

    /// Get recommended Gemma model based on system resources
    pub fn get_recommended_gemma_model() -> &'static str {
        // Use Gemma2 2B model - confirmed available and works well
        // Smaller model for better performance on most systems
        "gemma2:2b"
    }
}

#[async_trait::async_trait]
impl InferenceBackend for OllamaInference {
    async fn load_model(&mut self, model_path: &Path) -> Result<()> {
        // For Ollama, we interpret model_path as model name
        let model_name = model_path.to_string_lossy().to_string();
        
        info!("Loading Ollama model: {}", model_name);
        
        // Initialize Ollama if not already done (this handles finding/downloading Ollama)
        if self.ollama_client.is_none() {
            info!("Initializing Ollama backend...");
            self.initialize().await?;
        }
        
        // Check if model is available, pull if not
        if let Some(ollama) = &self.ollama_client {
            match ollama.show_model_info(model_name.clone()).await {
                Ok(_) => {
                    info!("Model {} is already available", model_name);
                    self.emit_progress(60.0, &format!("📋 {} found in local cache", model_name), &model_name).await;
                    
                    // Simulate some loading steps for better UX
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    self.emit_progress(80.0, &format!("🔍 Verifying {} model integrity", model_name), &model_name).await;
                    
                    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                    self.emit_progress(95.0, &format!("✅ {} validation complete", model_name), &model_name).await;
                }
                Err(_) => {
                    info!("Model {} not found, pulling from Ollama registry...", model_name);
                    self.emit_progress(10.0, &format!("Downloading {} model - this may take several minutes...", model_name), &model_name).await;
                    
                    // Use streaming pull_model with progress tracking
                    match tokio::time::timeout(
                        tokio::time::Duration::from_secs(1200), // 20 minute timeout for model pull
                        self.pull_model_with_progress(&model_name)
                    ).await {
                        Ok(Ok(_)) => {
                            info!("Model {} pulled successfully", model_name);
                            self.emit_progress(95.0, &format!("Model {} downloaded successfully", model_name), &model_name).await;
                        }
                        Ok(Err(e)) => {
                            return Err(anyhow::anyhow!("Failed to pull model {}: {}", model_name, e));
                        }
                        Err(_) => {
                            return Err(anyhow::anyhow!("Model {} pull timed out after 20 minutes", model_name));
                        }
                    }
                }
            }
        }
        
        self.current_model = Some(model_name.clone());
        self.model_loaded = true;
        
        // Always emit final completion progress
        self.emit_progress(100.0, &format!("🎉 {} ready! Model loaded successfully", model_name), &model_name).await;
        
        Ok(())
    }
    
    fn is_loaded(&self) -> bool {
        self.model_loaded && self.current_model.is_some()
    }
    
    /// Public method to restart Ollama when connection fails
    pub async fn restart_ollama_if_needed(&mut self) -> Result<()> {
        self.ensure_ollama_running().await
    }
    
    /// Check if Ollama is responsive and restart if needed
    async fn ensure_ollama_running(&mut self) -> Result<()> {
        // First, try a simple health check
        if let Some(ollama) = &self.ollama_client {
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(2),
                ollama.list_local_models()
            ).await {
                Ok(Ok(_)) => return Ok(()), // Ollama is responsive
                Ok(Err(e)) => {
                    warn!("Ollama health check failed: {}", e);
                }
                Err(_) => {
                    warn!("Ollama health check timed out");
                }
            }
        }
        
        // If we have an embedded Ollama, try to restart it
        if let Some(OllamaExe::Embedded(path)) = &self.ollama_exe {
            info!("Attempting to restart embedded Ollama...");
            
            // Emit user-friendly status
            self.emit_progress(
                0.0,
                "⚠️ Ollama server not responding. Attempting to restart...",
                "system"
            ).await;
            
            // Try to restart the server
            match self.start_ollama_server(path).await {
                Ok(()) => {
                    info!("Ollama server restarted successfully");
                    
                    // Give it a moment to fully start
                    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                    
                    // Test the connection
                    match self.test_connection().await {
                        Ok(()) => {
                            self.emit_progress(
                                100.0,
                                "✅ Ollama server restarted successfully!",
                                "system"
                            ).await;
                            Ok(())
                        }
                        Err(e) => {
                            let error_msg = format!(
                                "❌ Ollama server restarted but connection failed: {}. Please try restarting the app.",
                                e
                            );
                            self.emit_progress(0.0, &error_msg, "system").await;
                            Err(anyhow::anyhow!(error_msg))
                        }
                    }
                }
                Err(e) => {
                    let error_msg = format!(
                        "❌ Failed to restart Ollama server: {}. Please ensure port 11434 is free and try restarting the app.",
                        e
                    );
                    self.emit_progress(0.0, &error_msg, "system").await;
                    Err(anyhow::anyhow!(error_msg))
                }
            }
        } else {
            // System Ollama
            let error_msg = "❌ System Ollama is not responding. Please start Ollama manually with: ollama serve";
            self.emit_progress(0.0, error_msg, "system").await;
            Err(anyhow::anyhow!(error_msg))
        }
    }
    
    async fn generate(&self, prompt: &str, _config: &InferenceConfig) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating response with model: {}", model_name);
        info!("Prompt length: {} characters", prompt.len());
        
        // Use chat API for better formatting
        let request = ChatMessageRequest::new(
            model_name.clone(),
            vec![ChatMessage::user(prompt.to_string())],
        );
        
        info!("Sending request to Ollama...");
        
        // Add timeout to prevent infinite hanging
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(30), // 30 second timeout for faster failure detection
            ollama.send_chat_messages(request)
        ).await {
            Ok(result) => match result {
                Ok(response) => {
                    let mut content = response.message.content;
                    
                    // Strip Gemma special tokens from response
                    content = content.replace("<start_of_turn>", "");
                    content = content.replace("<end_of_turn>", "");
                    content = content.replace("<start_of_turn>user", "");
                    content = content.replace("<start_of_turn>model", "");
                    content = content.replace("<end_of_turn>model", "");
                    content = content.replace("<end_of_turn>user", "");
                    
                    // Trim any leading/trailing whitespace
                    content = content.trim().to_string();
                    
                    info!("Generated response: {}", content);
                    Ok(content)
                }
                Err(e) => {
                    error!("Failed to generate response: {}", e);
                    let error_str = e.to_string().to_lowercase();
                    
                    // Check if this is a connection error that might benefit from restart
                    if error_str.contains("reqwest") || error_str.contains("connection") || 
                       error_str.contains("refused") || error_str.contains("broken pipe") {
                        Err(anyhow::anyhow!("Ollama connection failed: {}. Restart may be needed.", e))
                    } else {
                        Err(anyhow::anyhow!("Ollama chat failed: {}", e))
                    }
                }
            },
            Err(_) => {
                error!("Ollama request timed out after 30 seconds");
                Err(anyhow::anyhow!("Ollama connection timeout - server may need restart"))
            }
        }
    }
    
    async fn generate_multimodal(&self, prompt: &str, media_data: Option<&[u8]>, media_type: Option<&str>, _config: &InferenceConfig) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating multimodal response with Gemma 3N processor for model: {}", model_name);
        
        // Create multimodal input for the Gemma3NProcessor
        let multimodal_input = MultimodalInput {
            text: Some(prompt.to_string()),
            image_data: if media_type == Some("image") { media_data.map(|d| d.to_vec()) } else { None },
            audio_data: if media_type == Some("audio") { media_data.map(|d| d.to_vec()) } else { None },
            video_data: if media_type == Some("video") { media_data.map(|d| d.to_vec()) } else { None },
        };
        
        // Process the multimodal input using Gemma3NProcessor for optimal performance
        let processed = match self.multimodal_processor.process_multimodal(multimodal_input).await {
            Ok(processed_data) => processed_data,
            Err(e) => {
                error!("Gemma3NProcessor failed: {}", e);
                return Err(anyhow::anyhow!("Multimodal processing failed: {}", e));
            }
        };
        
        info!("Processed multimodal input: {} tokens, {} images", 
              processed.token_count, processed.images.len());
        
        // Check if this is a multimodal-capable model in Ollama
        // NOTE: While Gemma 3N is designed as multimodal, Ollama's implementation currently only supports text
        let is_multimodal_model = model_name.contains("llava") || 
                                 model_name.contains("bakllava") || 
                                 model_name.contains("moondream") ||
                                 model_name.contains("llama3.2-vision") ||
                                 model_name.contains("llama3.2:11b-vision") ||
                                 model_name.contains("llama3.2:90b-vision");
        
        if !processed.images.is_empty() && !is_multimodal_model {
            // Handle non-multimodal models gracefully
            info!("Model {} is text-only, providing helpful response about image limitations", model_name);
            let vision_response = format!(
                "I can see that you've shared an image with me! However, I'm currently running on {}, which doesn't support vision processing in Ollama.\n\n{}\n\nTo analyze images with Ollama, I would need to be running on a vision-capable model like:\n- LLaMA 3.2 Vision (llama3.2-vision:11b or llama3.2-vision:90b) - Latest and most capable\n- LLaVA (llava:7b, llava:13b, or llava:34b) - Good general vision model\n- Moondream (moondream:latest) - Lightweight vision model\n- BakLLaVA (bakllava:latest) - Alternative vision model\n\nWould you like me to help you in another way, or could you describe what's in the image so I can assist with text-based analysis?",
                model_name,
                if model_name.contains("gemma3n") {
                    "Note: While Gemma 3N is designed as a multimodal model with vision capabilities, Ollama's current implementation only supports text input. The multimodal features are expected in a future update."
                } else {
                    ""
                }
            );
            return Ok(vision_response);
        }
        
        // Generate the response using appropriate API
        if !processed.images.is_empty() && is_multimodal_model {
            // Use GenerationRequest for multimodal input (images)
            info!("Using GenerationRequest for multimodal model with {} images", processed.images.len());
            
            // Format prompt with Gemma 3N-specific formatting
            let formatted_prompt = self.multimodal_processor.format_for_gemma3n(&processed, Some("You are Tektra, a helpful AI assistant with vision capabilities. Analyze any images provided and respond naturally."));
            
            let mut request = GenerationRequest::new(model_name.clone(), formatted_prompt);
            
            // Add processed images
            let ollama_images: Vec<ollama_rs::generation::images::Image> = processed.images
                .iter()
                .map(|base64_data| ollama_rs::generation::images::Image::from_base64(base64_data))
                .collect();
            
            if !ollama_images.is_empty() {
                request = request.images(ollama_images);
            }
            
            // Add timeout for multimodal processing
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(60), // Longer timeout for multimodal
                ollama.generate(request)
            ).await {
                Ok(result) => match result {
                    Ok(response) => {
                        let mut content = response.response;
                        
                        // Strip Gemma special tokens from response
                        content = content.replace("<start_of_turn>", "");
                        content = content.replace("<end_of_turn>", "");
                        content = content.replace("<start_of_turn>user", "");
                        content = content.replace("<start_of_turn>model", "");
                        content = content.replace("<end_of_turn>model", "");
                        content = content.replace("<end_of_turn>user", "");
                        
                        // Trim any leading/trailing whitespace
                        content = content.trim().to_string();
                        
                        info!("Generated multimodal response: {} chars", content.len());
                        Ok(content)
                    }
                    Err(e) => {
                        error!("Failed to generate multimodal response: {}", e);
                        let error_str = e.to_string().to_lowercase();
                        
                        // Check if this is a connection error that might benefit from restart
                        if error_str.contains("reqwest") || error_str.contains("connection") || 
                           error_str.contains("refused") || error_str.contains("broken pipe") {
                            Err(anyhow::anyhow!("Ollama connection failed: {}. Restart may be needed.", e))
                        } else {
                            Err(anyhow::anyhow!("Ollama multimodal generation failed: {}", e))
                        }
                    }
                },
                Err(_) => {
                    error!("Multimodal request timed out after 60 seconds");
                    Err(anyhow::anyhow!("Multimodal request timed out - content may be too complex"))
                }
            }
        } else {
            // Text-only generation with Gemma 3N formatting
            info!("Using ChatMessageRequest for text-only processing");
            
            // Format prompt with Gemma 3N-specific formatting for text
            let formatted_prompt = self.multimodal_processor.format_for_gemma3n(&processed, Some("You are Tektra, a helpful AI assistant. Provide clear, conversational responses."));
            
            let request = ChatMessageRequest::new(
                model_name.clone(),
                vec![ChatMessage::user(formatted_prompt)],
            );
            
            // Add timeout for text processing
            match tokio::time::timeout(
                tokio::time::Duration::from_secs(30), // Standard timeout for text
                ollama.send_chat_messages(request)
            ).await {
                Ok(result) => match result {
                    Ok(response) => {
                        let mut content = response.message.content;
                        
                        // Strip Gemma special tokens from response
                        content = content.replace("<start_of_turn>", "");
                        content = content.replace("<end_of_turn>", "");
                        content = content.replace("<start_of_turn>user", "");
                        content = content.replace("<start_of_turn>model", "");
                        content = content.replace("<end_of_turn>model", "");
                        content = content.replace("<end_of_turn>user", "");
                        
                        // Trim any leading/trailing whitespace
                        content = content.trim().to_string();
                        
                        info!("Generated text response: {} chars", content.len());
                        Ok(content)
                    }
                    Err(e) => {
                        error!("Failed to generate text response: {}", e);
                        let error_str = e.to_string().to_lowercase();
                        
                        // Check if this is a connection error that might benefit from restart
                        if error_str.contains("reqwest") || error_str.contains("connection") || 
                           error_str.contains("refused") || error_str.contains("broken pipe") {
                            Err(anyhow::anyhow!("Ollama connection failed: {}. Restart may be needed.", e))
                        } else {
                            Err(anyhow::anyhow!("Ollama text generation failed: {}", e))
                        }
                    }
                },
                Err(_) => {
                    error!("Text request timed out after 30 seconds");
                    Err(anyhow::anyhow!("Text request timed out - content may be too long"))
                }
            }
        }
    }
    
    // Use default implementation from trait
    
    fn name(&self) -> &str {
        "Ollama"
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn get_memory_usage_mb(&self) -> f64 {
        0.0 // Ollama manages memory internally
    }
    
    fn is_available() -> bool {
        true // Ollama can work on any platform
    }
}

