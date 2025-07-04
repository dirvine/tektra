use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{info, warn, error};

use super::{
    TektraMCPServer, MCPServerConfig, MCPServerInfo,
    transport::security::RateLimiter,
    transport::connection::ConnectionManager,
};
use crate::inference::EnhancedModelRegistry;
use crate::multimodal::UnifiedMultimodalInterface;
use crate::conversation::EnhancedConversationManager;

/// MCP server builder for easy configuration
pub struct MCPServerBuilder {
    config: MCPServerConfig,
    model_registry: Option<Arc<EnhancedModelRegistry>>,
    multimodal_interface: Option<Arc<UnifiedMultimodalInterface>>,
    conversation_manager: Option<Arc<EnhancedConversationManager>>,
}

impl MCPServerBuilder {
    pub fn new() -> Self {
        Self {
            config: MCPServerConfig::default(),
            model_registry: None,
            multimodal_interface: None,
            conversation_manager: None,
        }
    }
    
    pub fn with_config(mut self, config: MCPServerConfig) -> Self {
        self.config = config;
        self
    }
    
    pub fn with_model_registry(mut self, registry: Arc<EnhancedModelRegistry>) -> Self {
        self.model_registry = Some(registry);
        self
    }
    
    pub fn with_multimodal_interface(mut self, interface: Arc<UnifiedMultimodalInterface>) -> Self {
        self.multimodal_interface = Some(interface);
        self
    }
    
    pub fn with_conversation_manager(mut self, manager: Arc<EnhancedConversationManager>) -> Self {
        self.conversation_manager = Some(manager);
        self
    }
    
    pub fn enable_tools(mut self, enabled: bool) -> Self {
        self.config.enable_tools = enabled;
        self
    }
    
    pub fn enable_resources(mut self, enabled: bool) -> Self {
        self.config.enable_resources = enabled;
        self
    }
    
    pub fn enable_prompts(mut self, enabled: bool) -> Self {
        self.config.enable_prompts = enabled;
        self
    }
    
    pub fn enable_sampling(mut self, enabled: bool) -> Self {
        self.config.enable_sampling = enabled;
        self
    }
    
    pub fn with_rate_limit(mut self, requests_per_minute: u32) -> Self {
        self.config.rate_limit_requests_per_minute = requests_per_minute;
        self
    }
    
    pub fn with_max_sessions(mut self, max_sessions: usize) -> Self {
        self.config.max_concurrent_sessions = max_sessions;
        self
    }
    
    pub async fn build(self) -> Result<TektraMCPServer> {
        let model_registry = self.model_registry
            .ok_or_else(|| anyhow::anyhow!("Model registry is required"))?;
        
        let multimodal_interface = self.multimodal_interface
            .ok_or_else(|| anyhow::anyhow!("Multimodal interface is required"))?;
        
        let conversation_manager = self.conversation_manager
            .ok_or_else(|| anyhow::anyhow!("Conversation manager is required"))?;
        
        TektraMCPServer::new(model_registry, multimodal_interface, conversation_manager).await
    }
}

/// MCP server manager for handling server lifecycle
pub struct MCPServerManager {
    server: Option<TektraMCPServer>,
    rate_limiter: RateLimiter,
    connection_manager: ConnectionManager,
    server_stats: ServerStats,
    config: MCPServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    pub start_time: std::time::SystemTime,
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_sessions: usize,
    pub peak_sessions: usize,
    pub average_response_time_ms: f64,
    pub last_request_time: Option<std::time::SystemTime>,
}

impl MCPServerManager {
    pub fn new(config: MCPServerConfig) -> Self {
        let rate_limiter = RateLimiter::new(config.rate_limit_requests_per_minute);
        let connection_manager = ConnectionManager::new(config.max_concurrent_sessions);
        
        Self {
            server: None,
            rate_limiter,
            connection_manager,
            server_stats: ServerStats::new(),
            config,
        }
    }
    
    pub async fn start(&mut self, server: TektraMCPServer) -> Result<()> {
        info!("Starting MCP server manager");
        
        self.server = Some(server);
        self.server_stats.start_time = std::time::SystemTime::now();
        
        if let Some(ref server) = self.server {
            server.start().await?;
        }
        
        info!("MCP server manager started successfully");
        Ok(())
    }
    
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping MCP server manager");
        
        if let Some(server) = self.server.take() {
            // Server doesn't have a stop method in our current implementation
            // In a real implementation, this would gracefully shut down the server
            drop(server);
        }
        
        info!("MCP server manager stopped");
        Ok(())
    }
    
    pub fn is_running(&self) -> bool {
        self.server.is_some()
    }
    
    pub fn get_stats(&self) -> &ServerStats {
        &self.server_stats
    }
    
    pub fn get_connection_stats(&self) -> crate::mcp::transport::connection::ConnectionStats {
        self.connection_manager.get_stats()
    }
    
    pub fn update_request_stats(&mut self, success: bool, response_time_ms: u64) {
        self.server_stats.total_requests += 1;
        self.server_stats.last_request_time = Some(std::time::SystemTime::now());
        
        if success {
            self.server_stats.successful_requests += 1;
        } else {
            self.server_stats.failed_requests += 1;
        }
        
        // Update average response time
        let total_successful = self.server_stats.successful_requests;
        if total_successful > 0 {
            self.server_stats.average_response_time_ms = 
                (self.server_stats.average_response_time_ms * (total_successful - 1) as f64 + response_time_ms as f64) / total_successful as f64;
        }
        
        // Update uptime
        if let Ok(uptime) = self.server_stats.start_time.elapsed() {
            self.server_stats.uptime_seconds = uptime.as_secs();
        }
    }
    
    pub fn check_rate_limit(&mut self) -> bool {
        self.rate_limiter.is_allowed()
    }
    
    pub async fn health_check(&self) -> HealthCheckResult {
        let mut checks = Vec::new();
        
        // Check if server is running
        checks.push(HealthCheck {
            name: "server_running".to_string(),
            status: if self.is_running() { "healthy" } else { "unhealthy" }.to_string(),
            message: if self.is_running() { 
                "Server is running".to_string() 
            } else { 
                "Server is not running".to_string() 
            },
            details: None,
        });
        
        // Check connection count
        let connection_stats = self.connection_manager.get_stats();
        checks.push(HealthCheck {
            name: "connections".to_string(),
            status: if connection_stats.total_connections <= self.config.max_concurrent_sessions {
                "healthy"
            } else {
                "warning"
            }.to_string(),
            message: format!("{} active connections", connection_stats.total_connections),
            details: Some(serde_json::json!({
                "total_connections": connection_stats.total_connections,
                "max_connections": self.config.max_concurrent_sessions,
                "authenticated_connections": connection_stats.authenticated_connections
            })),
        });
        
        // Check rate limiting
        let rate_limit_status = if self.rate_limiter.current_count() < self.config.rate_limit_requests_per_minute as usize {
            "healthy"
        } else {
            "warning"
        };
        
        checks.push(HealthCheck {
            name: "rate_limiting".to_string(),
            status: rate_limit_status.to_string(),
            message: format!("{} requests in current minute", self.rate_limiter.current_count()),
            details: Some(serde_json::json!({
                "current_requests": self.rate_limiter.current_count(),
                "limit": self.config.rate_limit_requests_per_minute
            })),
        });
        
        // Overall health status
        let overall_status = if checks.iter().all(|c| c.status == "healthy") {
            "healthy"
        } else if checks.iter().any(|c| c.status == "unhealthy") {
            "unhealthy"
        } else {
            "warning"
        };
        
        HealthCheckResult {
            overall_status: overall_status.to_string(),
            timestamp: std::time::SystemTime::now(),
            checks,
            server_info: Some(MCPServerInfo::default()),
            uptime_seconds: self.server_stats.uptime_seconds,
        }
    }
    
    /// Perform maintenance tasks
    pub async fn perform_maintenance(&mut self) {
        // Clean up inactive connections
        let timeout = std::time::Duration::from_secs(300); // 5 minutes
        self.connection_manager.cleanup_inactive(timeout);
        
        // Update peak sessions
        let current_sessions = self.connection_manager.get_stats().total_connections;
        if current_sessions > self.server_stats.peak_sessions {
            self.server_stats.peak_sessions = current_sessions;
        }
        
        // Additional maintenance tasks would go here
    }
}

impl Default for MCPServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ServerStats {
    fn new() -> Self {
        Self {
            start_time: std::time::SystemTime::now(),
            uptime_seconds: 0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            active_sessions: 0,
            peak_sessions: 0,
            average_response_time_ms: 0.0,
            last_request_time: None,
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        (self.successful_requests as f64 / self.total_requests as f64) * 100.0
    }
    
    pub fn requests_per_second(&self) -> f64 {
        if self.uptime_seconds == 0 {
            return 0.0;
        }
        self.total_requests as f64 / self.uptime_seconds as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub overall_status: String,
    pub timestamp: std::time::SystemTime,
    pub checks: Vec<HealthCheck>,
    pub server_info: Option<MCPServerInfo>,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

/// Convenience functions for common server operations
pub mod operations {
    use super::*;
    
    /// Create a fully configured Tektra MCP server
    pub async fn create_tektra_server(
        model_registry: Arc<EnhancedModelRegistry>,
        multimodal_interface: Arc<UnifiedMultimodalInterface>,
        conversation_manager: Arc<EnhancedConversationManager>,
    ) -> Result<TektraMCPServer> {
        MCPServerBuilder::new()
            .with_model_registry(model_registry)
            .with_multimodal_interface(multimodal_interface)
            .with_conversation_manager(conversation_manager)
            .enable_tools(true)
            .enable_sampling(true)
            .with_rate_limit(60)
            .with_max_sessions(10)
            .build()
            .await
    }
    
    /// Create a development server with relaxed limits
    pub async fn create_development_server(
        model_registry: Arc<EnhancedModelRegistry>,
        multimodal_interface: Arc<UnifiedMultimodalInterface>,
        conversation_manager: Arc<EnhancedConversationManager>,
    ) -> Result<TektraMCPServer> {
        let mut config = MCPServerConfig::default();
        config.enable_request_logging = true;
        config.require_authentication = false;
        config.rate_limit_requests_per_minute = 120;
        config.max_concurrent_sessions = 20;
        
        MCPServerBuilder::new()
            .with_config(config)
            .with_model_registry(model_registry)
            .with_multimodal_interface(multimodal_interface)
            .with_conversation_manager(conversation_manager)
            .enable_tools(true)
            .enable_resources(true)
            .enable_prompts(true)
            .enable_sampling(true)
            .build()
            .await
    }
    
    /// Create a production server with strict limits
    pub async fn create_production_server(
        model_registry: Arc<EnhancedModelRegistry>,
        multimodal_interface: Arc<UnifiedMultimodalInterface>,
        conversation_manager: Arc<EnhancedConversationManager>,
    ) -> Result<TektraMCPServer> {
        let mut config = MCPServerConfig::default();
        config.enable_request_logging = false;
        config.require_authentication = true;
        config.rate_limit_requests_per_minute = 30;
        config.max_concurrent_sessions = 5;
        config.request_timeout_ms = 15000;
        
        MCPServerBuilder::new()
            .with_config(config)
            .with_model_registry(model_registry)
            .with_multimodal_interface(multimodal_interface)
            .with_conversation_manager(conversation_manager)
            .enable_tools(true)
            .enable_sampling(true)
            .build()
            .await
    }
    
    /// Start server with automatic management
    pub async fn start_managed_server(
        server: TektraMCPServer,
        config: MCPServerConfig,
    ) -> Result<MCPServerManager> {
        let mut manager = MCPServerManager::new(config);
        manager.start(server).await?;
        Ok(manager)
    }
}

/// Server monitoring utilities
pub mod monitoring {
    use super::*;
    use std::collections::VecDeque;
    
    /// Performance monitor for tracking server metrics
    #[derive(Debug)]
    pub struct PerformanceMonitor {
        response_times: VecDeque<u64>,
        request_counts: VecDeque<(std::time::SystemTime, u64)>,
        error_counts: VecDeque<(std::time::SystemTime, u64)>,
        max_samples: usize,
    }
    
    impl PerformanceMonitor {
        pub fn new(max_samples: usize) -> Self {
            Self {
                response_times: VecDeque::new(),
                request_counts: VecDeque::new(),
                error_counts: VecDeque::new(),
                max_samples,
            }
        }
        
        pub fn record_response_time(&mut self, time_ms: u64) {
            self.response_times.push_back(time_ms);
            if self.response_times.len() > self.max_samples {
                self.response_times.pop_front();
            }
        }
        
        pub fn record_request_count(&mut self, count: u64) {
            let now = std::time::SystemTime::now();
            self.request_counts.push_back((now, count));
            
            // Keep only last hour of data
            let hour_ago = now - std::time::Duration::from_secs(3600);
            while let Some((timestamp, _)) = self.request_counts.front() {
                if *timestamp < hour_ago {
                    self.request_counts.pop_front();
                } else {
                    break;
                }
            }
        }
        
        pub fn record_error_count(&mut self, count: u64) {
            let now = std::time::SystemTime::now();
            self.error_counts.push_back((now, count));
            
            // Keep only last hour of data
            let hour_ago = now - std::time::Duration::from_secs(3600);
            while let Some((timestamp, _)) = self.error_counts.front() {
                if *timestamp < hour_ago {
                    self.error_counts.pop_front();
                } else {
                    break;
                }
            }
        }
        
        pub fn get_metrics(&self) -> PerformanceMetrics {
            let avg_response_time = if !self.response_times.is_empty() {
                self.response_times.iter().sum::<u64>() as f64 / self.response_times.len() as f64
            } else {
                0.0
            };
            
            let max_response_time = self.response_times.iter().max().copied().unwrap_or(0);
            let min_response_time = self.response_times.iter().min().copied().unwrap_or(0);
            
            let total_requests: u64 = self.request_counts.iter().map(|(_, count)| count).sum();
            let total_errors: u64 = self.error_counts.iter().map(|(_, count)| count).sum();
            
            PerformanceMetrics {
                avg_response_time_ms: avg_response_time,
                max_response_time_ms: max_response_time,
                min_response_time_ms: min_response_time,
                total_requests_last_hour: total_requests,
                total_errors_last_hour: total_errors,
                error_rate: if total_requests > 0 {
                    (total_errors as f64 / total_requests as f64) * 100.0
                } else {
                    0.0
                },
                samples_count: self.response_times.len(),
            }
        }
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PerformanceMetrics {
        pub avg_response_time_ms: f64,
        pub max_response_time_ms: u64,
        pub min_response_time_ms: u64,
        pub total_requests_last_hour: u64,
        pub total_errors_last_hour: u64,
        pub error_rate: f64,
        pub samples_count: usize,
    }
}