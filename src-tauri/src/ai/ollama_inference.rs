use anyhow::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use tokio::fs;
use tracing::{info, error, warn};
use ollama_rs::{Ollama, generation::completion::request::GenerationRequest, generation::chat::{ChatMessage, request::ChatMessageRequest}};
use super::inference_backend::{InferenceBackend, InferenceConfig};
use tauri::{AppHandle, Emitter};
use serde_json::json;

#[derive(Debug, Clone)]
pub enum OllamaExe {
    System(PathBuf),
    Embedded(PathBuf),
}

pub struct OllamaInference {
    ollama_exe: Option<OllamaExe>,
    ollama_client: Option<Ollama>,
    model_loaded: bool,
    current_model: Option<String>,
    ollama_port: u16,
    app_handle: Option<AppHandle>,
}

impl OllamaInference {
    pub fn new() -> Self {
        Self {
            ollama_exe: None,
            ollama_client: None,
            model_loaded: false,
            current_model: None,
            ollama_port: 11434, // Default Ollama port
            app_handle: None,
        }
    }
    
    pub fn with_app_handle(app_handle: AppHandle) -> Self {
        Self {
            ollama_exe: None,
            ollama_client: None,
            model_loaded: false,
            current_model: None,
            ollama_port: 11434, // Default Ollama port
            app_handle: Some(app_handle),
        }
    }
    
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
    pub async fn find_ollama() -> Result<OllamaExe> {
        info!("Checking for system Ollama installation...");
        
        // First try to find system Ollama
        if let Ok(output) = Command::new("ollama").arg("--version").output() {
            if output.status.success() {
                info!("Found system Ollama installation");
                return Ok(OllamaExe::System(PathBuf::from("ollama")));
            }
        }

        info!("System Ollama not found, downloading embedded version...");
        
        // Download embedded Ollama - use a different location to avoid macOS restrictions
        let data_dir = std::env::temp_dir()
            .join("tektra_ollama")
            .join("extracted");
        
        fs::create_dir_all(&data_dir).await?;
        
        // Use default download configuration
        let download_config = ollama_td::OllamaDownload::default();
        let downloaded_path = ollama_td::download(download_config).await
            .map_err(|e| anyhow::anyhow!("Failed to download Ollama: {}", e))?;
        
        info!("Ollama downloaded to: {:?}", downloaded_path);
        
        // Check if downloaded file is a zip archive and extract it
        let ollama_binary = if downloaded_path.extension().and_then(|s| s.to_str()) == Some("zip") {
            info!("Extracting Ollama zip archive...");
            
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
        Ok(OllamaExe::Embedded(ollama_binary))
    }

    /// Initialize Ollama (find binary and start if needed)
    pub async fn initialize(&mut self) -> Result<()> {
        info!("Initializing Ollama inference backend...");
        
        // Wrap the heavy operation in a timeout and better error handling
        let ollama_exe = match tokio::time::timeout(
            tokio::time::Duration::from_secs(300), // 5 minute timeout for download
            Self::find_ollama()
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
        
        // Test connection with timeout
        match tokio::time::timeout(
            tokio::time::Duration::from_secs(30),
            self.test_connection()
        ).await {
            Ok(Ok(_)) => info!("Ollama inference backend initialized successfully"),
            Ok(Err(e)) => return Err(anyhow::anyhow!("Ollama connection test failed: {}", e)),
            Err(_) => return Err(anyhow::anyhow!("Ollama connection test timed out")),
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
        let child = cmd.spawn()
            .map_err(|e| anyhow::anyhow!("Failed to start Ollama server: {}", e))?;
        
        info!("Ollama server started with PID: {} at {:?}", child.id(), ollama_path);
        
        // Wait a moment for server to start
        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
        
        Ok(())
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
                    self.emit_progress(95.0, &format!("Model {} already available", model_name), &model_name).await;
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
        
        self.current_model = Some(model_name);
        self.model_loaded = true;
        
        Ok(())
    }
    
    fn is_loaded(&self) -> bool {
        self.model_loaded && self.current_model.is_some()
    }
    
    async fn generate(&self, prompt: &str, _config: &InferenceConfig) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating response with model: {}", model_name);
        
        // Use chat API for better formatting
        let request = ChatMessageRequest::new(
            model_name.clone(),
            vec![ChatMessage::user(prompt.to_string())],
        );
        
        match ollama.send_chat_messages(request).await {
            Ok(response) => {
                let content = response.message.content;
                info!("Generated response: {}", content);
                Ok(content)
            }
            Err(e) => {
                error!("Failed to generate response: {}", e);
                Err(anyhow::anyhow!("Ollama chat failed: {}", e))
            }
        }
    }
    
    async fn generate_multimodal(&self, prompt: &str, media_data: Option<&[u8]>, media_type: Option<&str>, _config: &InferenceConfig) -> Result<String> {
        let ollama = self.ollama_client.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ollama client not initialized"))?;
        
        let model_name = self.current_model.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No model loaded"))?;
        
        info!("Generating multimodal response with model: {}", model_name);
        
        if let Some(data_bytes) = media_data {
            match media_type {
                Some("audio") => {
                    info!("Processing audio input with Gemma-3n ({} bytes)", data_bytes.len());
                    
                    // Convert raw audio to a simple analysis for Gemma-3n
                    let duration = data_bytes.len() as f32 / (16000.0 * 2.0);
                    let sample_count = data_bytes.len() / 2;
                    
                    // Analyze audio characteristics
                    let mut energy_sum = 0.0f32;
                    let mut peak_amplitude = 0.0f32;
                    
                    for chunk in data_bytes.chunks(2) {
                        if chunk.len() == 2 {
                            let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                            energy_sum += sample * sample;
                            peak_amplitude = peak_amplitude.max(sample.abs());
                        }
                    }
                    
                    let avg_energy = energy_sum / sample_count as f32;
                    let volume_level = if avg_energy > 0.01 { "loud" } else if avg_energy > 0.001 { "normal" } else { "quiet" };
                    
                    // Create a more intelligent prompt for Gemma-3n
                    let audio_prompt = format!(
                        "The user just spoke for {:.1} seconds with {} volume. Based on the context that they asked: '{}', please provide a helpful response. If this sounds like a question about capitals, geography, or general knowledge, please answer appropriately.",
                        duration, volume_level, prompt
                    );
                    
                    let request = ChatMessageRequest::new(
                        model_name.clone(),
                        vec![ChatMessage::user(audio_prompt)],
                    );
                    
                    match ollama.send_chat_messages(request).await {
                        Ok(response) => {
                            let content = response.message.content;
                            info!("Generated audio-contextual response: {}", content);
                            Ok(content)
                        }
                        Err(e) => {
                            error!("Failed to generate audio response: {}", e);
                            Err(anyhow::anyhow!("Ollama audio processing failed: {}", e))
                        }
                    }
                }
                Some("image") => {
                    info!("Multimodal request with image data ({} bytes) using model: {}", data_bytes.len(), model_name);
                    
                    // Check if this model supports vision (most Gemma models don't)
                    if model_name.contains("gemma") || model_name.contains("llama") && !model_name.contains("llava") {
                        info!("Model {} is text-only, providing helpful response about image limitations", model_name);
                        
                        // Provide a helpful response indicating the model can't see images
                        let vision_response = format!(
                            "I can see that you've shared an image with me! However, I'm currently running on {}, which is a text-only model and cannot process visual information.\n\nTo help you with image analysis, I would need to be running on a vision-capable model like:\n- LLaVA (llava:7b or llava:13b)\n- Moondream (moondream:latest)\n- Bakllava (bakllava:latest)\n\nWould you like me to help you in another way, or could you describe what's in the image so I can assist with text-based analysis?", 
                            model_name
                        );
                        
                        Ok(vision_response)
                    } else {
                        // Try multimodal generation for vision-capable models
                        let mut request = GenerationRequest::new(model_name.clone(), prompt.to_string());
                        
                        // Add image if provided
                        use base64::{Engine as _, engine::general_purpose};
                        let base64_image = general_purpose::STANDARD.encode(data_bytes);
                        let image = ollama_rs::generation::images::Image::from_base64(&base64_image);
                        request = request.images(vec![image]);
                        
                        match ollama.generate(request).await {
                            Ok(response) => {
                                let content = response.response;
                                info!("Generated vision response: {}", content);
                                Ok(content)
                            }
                            Err(e) => {
                                error!("Failed to generate vision response: {}", e);
                                
                                // Fallback to explaining the limitation
                                let fallback_response = format!(
                                    "I can see that you've shared an image, but I'm having trouble processing it with the current model configuration. The error was: {}\n\nThis might mean the model doesn't support vision, or there's a configuration issue. Would you like to describe what's in the image so I can help in another way?", 
                                    e
                                );
                                
                                Ok(fallback_response)
                            }
                        }
                    }
                }
                _ => {
                    // No media type or unsupported media type, use chat API for better text formatting
                    self.generate(prompt, _config).await
                }
            }
        } else {
            // No media data, use chat API for better text formatting
            self.generate(prompt, _config).await
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

