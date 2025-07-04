use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

use super::{RustBackendServer, BackendServerConfig};
use crate::inference::ModelRegistry;

/// Manages Unmute services without Docker
/// Handles automatic installation, startup, and lifecycle management
pub struct UnmuteServiceManager {
    /// Map of service name to process handle
    services: Arc<RwLock<HashMap<String, ServiceProcess>>>,
    /// Configuration for Unmute services
    config: Arc<RwLock<UnmuteConfig>>,
    /// Base directory where Unmute is cloned/installed
    unmute_dir: PathBuf,
    /// Event channel for service status updates
    event_tx: Option<mpsc::UnboundedSender<ServiceEvent>>,
    /// Whether services are currently running
    is_running: Arc<RwLock<bool>>,
    /// Dependency checker and installer
    dependency_manager: Arc<Mutex<DependencyManager>>,
    /// Rust backend server (replaces Python uvicorn)
    rust_backend: Option<RustBackendServer>,
    /// Model registry for LLM integration
    model_registry: Option<Arc<Mutex<ModelRegistry>>>,
}

/// Configuration for Unmute services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnmuteConfig {
    /// Backend service configuration
    pub backend: ServiceConfig,
    /// STT service configuration
    pub stt: ServiceConfig,
    /// TTS service configuration
    pub tts: ServiceConfig,
    /// LLM service configuration (we'll use our mistral.rs instead)
    pub llm: Option<ServiceConfig>,
    /// Whether to use our own LLM instead of vLLM
    pub use_internal_llm: bool,
    /// GPU settings
    pub gpu_config: GpuConfig,
}

/// Configuration for individual services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// Service name
    pub name: String,
    /// Port to run on
    pub port: u16,
    /// Whether service is enabled
    pub enabled: bool,
    /// Custom arguments
    pub args: Vec<String>,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Startup timeout in seconds
    pub startup_timeout: u64,
}

/// GPU configuration for Unmute services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Whether to use GPU acceleration
    pub enabled: bool,
    /// GPU memory utilization (0.0 - 1.0)
    pub memory_utilization: f32,
    /// CUDA version requirement
    pub cuda_version: String,
    /// Specific GPU devices to use
    pub device_ids: Vec<u32>,
}

/// Service process wrapper
#[derive(Debug)]
struct ServiceProcess {
    /// Process handle
    child: Child,
    /// Service configuration
    config: ServiceConfig,
    /// Start time
    started_at: std::time::Instant,
    /// Whether service is healthy
    is_healthy: bool,
}

/// Events emitted by service manager
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServiceEvent {
    /// Service starting
    ServiceStarting {
        service: String,
        port: u16,
    },
    /// Service started successfully
    ServiceStarted {
        service: String,
        port: u16,
        startup_time_ms: u64,
    },
    /// Service failed to start
    ServiceStartFailed {
        service: String,
        error: String,
    },
    /// Service stopped
    ServiceStopped {
        service: String,
        reason: String,
    },
    /// Service health check
    ServiceHealthCheck {
        service: String,
        is_healthy: bool,
    },
    /// Dependency installation progress
    DependencyInstalling {
        dependency: String,
        progress: String,
    },
    /// Dependency installed
    DependencyInstalled {
        dependency: String,
    },
    /// Dependency installation failed
    DependencyFailed {
        dependency: String,
        error: String,
    },
    /// All services ready
    AllServicesReady,
    /// Error occurred
    Error {
        message: String,
    },
}

/// Manages installation and checking of dependencies
#[derive(Debug)]
struct DependencyManager {
    /// Whether dependencies are checked
    dependencies_checked: bool,
    /// Installation status for each dependency
    dependency_status: HashMap<String, DependencyStatus>,
}

/// Status of a dependency
#[derive(Debug, Clone, PartialEq)]
enum DependencyStatus {
    NotInstalled,
    Installing,
    Installed,
    Failed(String),
}

impl Default for UnmuteConfig {
    fn default() -> Self {
        Self {
            backend: ServiceConfig {
                name: "backend".to_string(),
                port: 8000,
                enabled: true,
                args: vec![],
                env: HashMap::new(),
                startup_timeout: 30,
            },
            stt: ServiceConfig {
                name: "stt".to_string(),
                port: 8090,
                enabled: true,
                args: vec![],
                env: HashMap::new(),
                startup_timeout: 60, // STT takes longer to load models
            },
            tts: ServiceConfig {
                name: "tts".to_string(),
                port: 8089,
                enabled: true,
                args: vec![],
                env: HashMap::new(),
                startup_timeout: 60, // TTS takes longer to load models
            },
            llm: None, // We'll use our mistral.rs backend instead
            use_internal_llm: true,
            gpu_config: GpuConfig {
                enabled: true,
                memory_utilization: 0.3,
                cuda_version: "12.1".to_string(),
                device_ids: vec![0],
            },
        }
    }
}

impl UnmuteServiceManager {
    /// Create a new Unmute service manager
    pub async fn new(unmute_dir: PathBuf, config: Option<UnmuteConfig>) -> Result<Self> {
        info!("Initializing Unmute service manager at: {:?}", unmute_dir);
        
        let config = config.unwrap_or_default();
        
        Ok(Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            unmute_dir,
            event_tx: None,
            is_running: Arc::new(RwLock::new(false)),
            dependency_manager: Arc::new(Mutex::new(DependencyManager::new())),
            rust_backend: None,
            model_registry: None,
        })
    }

    /// Create a new Unmute service manager with model registry
    pub async fn new_with_model_registry(
        unmute_dir: PathBuf, 
        config: Option<UnmuteConfig>,
        model_registry: Arc<Mutex<ModelRegistry>>,
    ) -> Result<Self> {
        info!("Initializing Unmute service manager with model registry at: {:?}", unmute_dir);
        
        let config = config.unwrap_or_default();
        
        Ok(Self {
            services: Arc::new(RwLock::new(HashMap::new())),
            config: Arc::new(RwLock::new(config)),
            unmute_dir,
            event_tx: None,
            is_running: Arc::new(RwLock::new(false)),
            dependency_manager: Arc::new(Mutex::new(DependencyManager::new())),
            rust_backend: None,
            model_registry: Some(model_registry),
        })
    }

    /// Set event channel for status updates
    pub fn set_event_channel(&mut self, tx: mpsc::UnboundedSender<ServiceEvent>) {
        self.event_tx = Some(tx);
    }

    /// Check if Unmute directory exists and clone if needed
    pub async fn ensure_unmute_repository(&self) -> Result<()> {
        if !self.unmute_dir.exists() {
            info!("Cloning Unmute repository to: {:?}", self.unmute_dir);
            
            self.emit_event(ServiceEvent::DependencyInstalling {
                dependency: "unmute-repository".to_string(),
                progress: "Cloning from GitHub...".to_string(),
            }).await;

            let output = Command::new("git")
                .args(&[
                    "clone",
                    "https://github.com/kyutai-labs/unmute.git",
                    self.unmute_dir.to_str().unwrap(),
                ])
                .output()
                .map_err(|e| anyhow!("Failed to clone Unmute repository: {}", e))?;

            if !output.status.success() {
                let error = String::from_utf8_lossy(&output.stderr);
                return Err(anyhow!("Git clone failed: {}", error));
            }

            self.emit_event(ServiceEvent::DependencyInstalled {
                dependency: "unmute-repository".to_string(),
            }).await;
        } else {
            info!("Unmute repository already exists at: {:?}", self.unmute_dir);
        }

        Ok(())
    }

    /// Check and install required dependencies
    pub async fn check_dependencies(&self) -> Result<()> {
        info!("Checking Unmute dependencies");
        
        let mut dep_manager = self.dependency_manager.lock().await;
        
        if dep_manager.dependencies_checked {
            return Ok(());
        }

        // Check for required tools
        let dependencies = vec![
            ("git", "git --version"),
            ("uv", "uv --version"),
            ("cargo", "cargo --version"),
            ("pnpm", "pnpm --version"),
        ];

        for (name, check_cmd) in dependencies {
            self.check_and_install_dependency(name, check_cmd, &mut dep_manager).await?;
        }

        // Check CUDA if GPU is enabled
        let config = self.config.read().await;
        if config.gpu_config.enabled {
            self.check_cuda(&mut dep_manager).await?;
        }

        dep_manager.dependencies_checked = true;
        Ok(())
    }

    /// Check and install a specific dependency
    async fn check_and_install_dependency(
        &self,
        name: &str,
        check_cmd: &str,
        dep_manager: &mut DependencyManager,
    ) -> Result<()> {
        info!("Checking dependency: {}", name);
        
        // Check if already installed
        let output = Command::new("sh")
            .args(&["-c", check_cmd])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                info!("{} is already installed", name);
                dep_manager.dependency_status.insert(name.to_string(), DependencyStatus::Installed);
                return Ok(());
            }
            _ => {
                warn!("{} is not installed, attempting to install", name);
            }
        }

        self.emit_event(ServiceEvent::DependencyInstalling {
            dependency: name.to_string(),
            progress: "Installing...".to_string(),
        }).await;

        dep_manager.dependency_status.insert(name.to_string(), DependencyStatus::Installing);

        // Install based on dependency type
        let install_result = match name {
            "uv" => self.install_uv().await,
            "pnpm" => self.install_pnpm().await,
            "cargo" => self.install_rust().await,
            _ => {
                warn!("Don't know how to install {}, please install manually", name);
                Err(anyhow!("Manual installation required for {}", name))
            }
        };

        match install_result {
            Ok(_) => {
                dep_manager.dependency_status.insert(name.to_string(), DependencyStatus::Installed);
                self.emit_event(ServiceEvent::DependencyInstalled {
                    dependency: name.to_string(),
                }).await;
            }
            Err(e) => {
                let error_msg = e.to_string();
                dep_manager.dependency_status.insert(name.to_string(), DependencyStatus::Failed(error_msg.clone()));
                self.emit_event(ServiceEvent::DependencyFailed {
                    dependency: name.to_string(),
                    error: error_msg,
                }).await;
                return Err(e);
            }
        }

        Ok(())
    }

    /// Install uv package manager
    async fn install_uv(&self) -> Result<()> {
        info!("Installing uv package manager");
        
        let output = Command::new("sh")
            .args(&["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
            .output()
            .map_err(|e| anyhow!("Failed to install uv: {}", e))?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("uv installation failed: {}", error));
        }

        Ok(())
    }

    /// Install pnpm package manager
    async fn install_pnpm(&self) -> Result<()> {
        info!("Installing pnpm package manager");
        
        let output = Command::new("sh")
            .args(&["-c", "curl -fsSL https://get.pnpm.io/install.sh | sh -"])
            .output()
            .map_err(|e| anyhow!("Failed to install pnpm: {}", e))?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("pnpm installation failed: {}", error));
        }

        Ok(())
    }

    /// Install Rust and Cargo
    async fn install_rust(&self) -> Result<()> {
        info!("Installing Rust and Cargo");
        
        let output = Command::new("sh")
            .args(&["-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"])
            .output()
            .map_err(|e| anyhow!("Failed to install Rust: {}", e))?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("Rust installation failed: {}", error));
        }

        Ok(())
    }

    /// Check CUDA installation
    async fn check_cuda(&self, dep_manager: &mut DependencyManager) -> Result<()> {
        info!("Checking CUDA installation");
        
        let output = Command::new("nvcc")
            .args(&["--version"])
            .output();

        match output {
            Ok(output) if output.status.success() => {
                let version_output = String::from_utf8_lossy(&output.stdout);
                info!("CUDA is available: {}", version_output.lines().next().unwrap_or(""));
                dep_manager.dependency_status.insert("cuda".to_string(), DependencyStatus::Installed);
            }
            _ => {
                warn!("CUDA not found - GPU acceleration will be disabled");
                // Update config to disable GPU
                let mut config = self.config.write().await;
                config.gpu_config.enabled = false;
                dep_manager.dependency_status.insert("cuda".to_string(), DependencyStatus::Failed("Not installed".to_string()));
            }
        }

        Ok(())
    }

    /// Start all enabled Unmute services
    pub async fn start_services(&mut self) -> Result<()> {
        info!("Starting Unmute services");

        if *self.is_running.read().await {
            return Ok(());
        }

        // Ensure repository and dependencies
        self.ensure_unmute_repository().await?;
        self.check_dependencies().await?;

        let config = self.config.read().await.clone();
        let mut services = self.services.write().await;

        // Start backend service
        if config.backend.enabled {
            drop(services); // Drop services lock before calling start_backend_service
            self.start_backend_service(&config.backend).await?;
            services = self.services.write().await; // Re-acquire lock
        }

        // Start STT service
        if config.stt.enabled {
            self.start_stt_service(&config.stt, &mut services).await?;
        }

        // Start TTS service
        if config.tts.enabled {
            self.start_tts_service(&config.tts, &mut services).await?;
        }

        // Skip LLM service if using internal mistral.rs
        if !config.use_internal_llm {
            if let Some(ref llm_config) = config.llm {
                if llm_config.enabled {
                    self.start_llm_service(llm_config, &mut services).await?;
                }
            }
        } else {
            info!("Using internal mistral.rs backend instead of vLLM");
        }

        *self.is_running.write().await = true;

        // Wait for services to be healthy
        self.wait_for_services_healthy().await?;

        self.emit_event(ServiceEvent::AllServicesReady).await;

        info!("All Unmute services started successfully");
        Ok(())
    }

    /// Start the backend service using Rust server
    async fn start_backend_service(
        &mut self,
        config: &ServiceConfig,
    ) -> Result<()> {
        info!("Starting Rust backend server on port {}", config.port);

        self.emit_event(ServiceEvent::ServiceStarting {
            service: config.name.clone(),
            port: config.port,
        }).await;

        let start_time = std::time::Instant::now();

        // Create Rust backend server configuration
        let server_config = BackendServerConfig {
            bind_addr: format!("127.0.0.1:{}", config.port).parse()
                .map_err(|e| anyhow!("Invalid bind address: {}", e))?,
            stt_service_url: "ws://127.0.0.1:8090".to_string(),
            tts_service_url: "ws://127.0.0.1:8089".to_string(),
            max_connections: 4,
            connection_timeout: 300,
            health_check_interval: 30,
        };

        // Get model registry
        let model_registry = self.model_registry.clone()
            .ok_or_else(|| anyhow!("Model registry not available"))?;

        // Create and start Rust backend server
        let mut rust_server = RustBackendServer::new(server_config, model_registry).await?;
        
        rust_server.start().await.map_err(|e| anyhow!("Failed to start Rust backend: {}", e))?;

        info!("Rust backend server started successfully on port {}", config.port);

        // Store the server instance (we don't use Child processes for Rust server)
        self.rust_backend = Some(rust_server);

        self.emit_event(ServiceEvent::ServiceStarted {
            service: config.name.clone(),
            port: config.port,
            startup_time_ms: start_time.elapsed().as_millis() as u64,
        }).await;

        Ok(())
    }

    /// Start the STT service
    async fn start_stt_service(
        &self,
        config: &ServiceConfig,
        services: &mut HashMap<String, ServiceProcess>,
    ) -> Result<()> {
        info!("Starting Unmute STT service on port {}", config.port);

        self.emit_event(ServiceEvent::ServiceStarting {
            service: config.name.clone(),
            port: config.port,
        }).await;

        let start_time = std::time::Instant::now();

        // Install moshi-server if not already installed
        let install_output = Command::new("cargo")
            .args(&["install", "--features", "cuda", "moshi-server@0.6.3"])
            .env("CARGO_NET_GIT_FETCH_WITH_CLI", "true")
            .current_dir(&self.unmute_dir)
            .output()
            .map_err(|e| anyhow!("Failed to install moshi-server: {}", e))?;

        if !install_output.status.success() {
            let error = String::from_utf8_lossy(&install_output.stderr);
            warn!("Moshi-server install output: {}", error);
        }

        // Start STT service
        let mut cmd = Command::new("moshi-server")
            .args(&["worker"])
            .args(&["--config", "services/moshi-server/configs/stt.toml"])
            .arg(&format!("--port={}", config.port))
            .current_dir(&self.unmute_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to start STT service: {}", e))?;

        let process = ServiceProcess {
            child: cmd,
            config: config.clone(),
            started_at: start_time,
            is_healthy: false,
        };

        services.insert(config.name.clone(), process);

        self.emit_event(ServiceEvent::ServiceStarted {
            service: config.name.clone(),
            port: config.port,
            startup_time_ms: start_time.elapsed().as_millis() as u64,
        }).await;

        Ok(())
    }

    /// Start the TTS service
    async fn start_tts_service(
        &self,
        config: &ServiceConfig,
        services: &mut HashMap<String, ServiceProcess>,
    ) -> Result<()> {
        info!("Starting Unmute TTS service on port {}", config.port);

        self.emit_event(ServiceEvent::ServiceStarting {
            service: config.name.clone(),
            port: config.port,
        }).await;

        let start_time = std::time::Instant::now();

        // Start TTS service (moshi-server should already be installed from STT)
        let mut cmd = Command::new("moshi-server")
            .args(&["worker"])
            .args(&["--config", "services/moshi-server/configs/tts.toml"])
            .arg(&format!("--port={}", config.port))
            .current_dir(&self.unmute_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to start TTS service: {}", e))?;

        let process = ServiceProcess {
            child: cmd,
            config: config.clone(),
            started_at: start_time,
            is_healthy: false,
        };

        services.insert(config.name.clone(), process);

        self.emit_event(ServiceEvent::ServiceStarted {
            service: config.name.clone(),
            port: config.port,
            startup_time_ms: start_time.elapsed().as_millis() as u64,
        }).await;

        Ok(())
    }

    /// Start the LLM service (only if not using internal mistral.rs)
    async fn start_llm_service(
        &self,
        config: &ServiceConfig,
        services: &mut HashMap<String, ServiceProcess>,
    ) -> Result<()> {
        info!("Starting Unmute LLM service on port {}", config.port);

        self.emit_event(ServiceEvent::ServiceStarting {
            service: config.name.clone(),
            port: config.port,
        }).await;

        let start_time = std::time::Instant::now();

        let gpu_config = &self.config.read().await.gpu_config;

        let mut cmd = Command::new("uv")
            .args(&["tool", "run", "vllm@v0.9.1", "serve"])
            .args(&["--model=google/gemma-3-1b-it"])
            .args(&["--max-model-len=8192"])
            .args(&["--dtype=bfloat16"])
            .arg(&format!("--gpu-memory-utilization={}", gpu_config.memory_utilization))
            .arg(&format!("--port={}", config.port))
            .current_dir(&self.unmute_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("Failed to start LLM service: {}", e))?;

        let process = ServiceProcess {
            child: cmd,
            config: config.clone(),
            started_at: start_time,
            is_healthy: false,
        };

        services.insert(config.name.clone(), process);

        self.emit_event(ServiceEvent::ServiceStarted {
            service: config.name.clone(),
            port: config.port,
            startup_time_ms: start_time.elapsed().as_millis() as u64,
        }).await;

        Ok(())
    }

    /// Wait for all services to be healthy
    async fn wait_for_services_healthy(&self) -> Result<()> {
        info!("Waiting for services to be healthy");
        
        let timeout = Duration::from_secs(120); // 2 minutes timeout
        let start_time = std::time::Instant::now();
        
        while start_time.elapsed() < timeout {
            let all_healthy = self.check_services_health().await?;
            
            if all_healthy {
                info!("All services are healthy");
                return Ok(());
            }
            
            // Wait before next check
            sleep(Duration::from_secs(5)).await;
        }
        
        Err(anyhow!("Timeout waiting for services to be healthy"))
    }

    /// Check health of all running services
    async fn check_services_health(&self) -> Result<bool> {
        let services = self.services.read().await;
        let mut all_healthy = true;

        for (name, _process) in services.iter() {
            let config = match name.as_str() {
                "backend" => &self.config.read().await.backend,
                "stt" => &self.config.read().await.stt,
                "tts" => &self.config.read().await.tts,
                _ => continue,
            };

            let is_healthy = self.check_service_health(name, config.port).await;
            
            self.emit_event(ServiceEvent::ServiceHealthCheck {
                service: name.clone(),
                is_healthy,
            }).await;

            if !is_healthy {
                all_healthy = false;
            }
        }

        Ok(all_healthy)
    }

    /// Check health of a specific service
    async fn check_service_health(&self, service_name: &str, port: u16) -> bool {
        // Simple TCP connection check
        match tokio::net::TcpStream::connect(format!("127.0.0.1:{}", port)).await {
            Ok(_) => {
                debug!("Service {} is responding on port {}", service_name, port);
                true
            }
            Err(_) => {
                debug!("Service {} is not responding on port {}", service_name, port);
                false
            }
        }
    }

    /// Stop all services
    pub async fn stop_services(&mut self) -> Result<()> {
        info!("Stopping Unmute services");

        if !*self.is_running.read().await {
            return Ok(());
        }

        // Stop Rust backend server first
        if let Some(mut rust_backend) = self.rust_backend.take() {
            info!("Stopping Rust backend server");
            if let Err(e) = rust_backend.stop().await {
                warn!("Failed to stop Rust backend gracefully: {}", e);
            }
            
            self.emit_event(ServiceEvent::ServiceStopped {
                service: "backend".to_string(),
                reason: "User requested".to_string(),
            }).await;
        }

        // Stop other services (STT, TTS)
        let mut services = self.services.write().await;
        
        for (name, mut process) in services.drain() {
            info!("Stopping service: {}", name);
            
            // Try graceful shutdown first
            if let Err(e) = process.child.kill() {
                warn!("Failed to kill service {}: {}", name, e);
            }
            
            // Wait for process to exit
            if let Err(e) = process.child.wait() {
                warn!("Failed to wait for service {} to exit: {}", name, e);
            }
            
            self.emit_event(ServiceEvent::ServiceStopped {
                service: name.clone(),
                reason: "User requested".to_string(),
            }).await;
        }

        *self.is_running.write().await = false;
        info!("All Unmute services stopped");
        
        Ok(())
    }

    /// Check if services are running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Get service status
    pub async fn get_service_status(&self) -> HashMap<String, bool> {
        let services = self.services.read().await;
        let mut status = HashMap::new();
        
        for name in &["backend", "stt", "tts"] {
            status.insert(name.to_string(), services.contains_key(*name));
        }
        
        status
    }

    /// Update configuration
    pub async fn update_config(&mut self, new_config: UnmuteConfig) -> Result<()> {
        info!("Updating Unmute service configuration");
        *self.config.write().await = new_config;
        Ok(())
    }

    /// Emit an event
    async fn emit_event(&self, event: ServiceEvent) {
        if let Some(tx) = &self.event_tx {
            let _ = tx.send(event);
        }
    }
}

impl DependencyManager {
    fn new() -> Self {
        Self {
            dependencies_checked: false,
            dependency_status: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_service_manager_creation() {
        let temp_dir = TempDir::new().unwrap();
        let manager = UnmuteServiceManager::new(temp_dir.path().to_path_buf(), None).await;
        assert!(manager.is_ok());
    }

    #[test]
    fn test_default_config() {
        let config = UnmuteConfig::default();
        assert_eq!(config.backend.port, 8000);
        assert_eq!(config.stt.port, 8090);
        assert_eq!(config.tts.port, 8089);
        assert!(config.use_internal_llm);
    }
}