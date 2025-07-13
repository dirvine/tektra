#!/usr/bin/env python3
"""
Production Configuration Management

Comprehensive configuration system for production deployment with environment
variables, validation, secrets management, and deployment-specific settings.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with pydantic,loguru,cryptography python production_config.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pydantic>=2.0.0",
#     "pydantic-settings>=2.0.0",
#     "loguru>=0.7.0",
#     "cryptography>=41.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///

import os
import json
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from enum import Enum

from pydantic import BaseModel, Field, validator, root_validator
from pydantic_settings import BaseSettings
from loguru import logger
from cryptography.fernet import Fernet

from ..security.context import SecurityLevel
from ..core.tektra_system import TektraSystemConfig


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    url: str = Field(default="sqlite:///tektra.db", description="Database URL")
    pool_size: int = Field(default=5, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=100, description="Max pool overflow")
    pool_timeout: int = Field(default=30, ge=1, le=300, description="Pool timeout seconds")
    echo: bool = Field(default=False, description="Echo SQL statements")
    
    # Connection encryption
    ssl_required: bool = Field(default=False, description="Require SSL connection")
    ssl_cert_path: Optional[str] = Field(default=None, description="SSL certificate path")
    ssl_key_path: Optional[str] = Field(default=None, description="SSL key path")
    ssl_ca_path: Optional[str] = Field(default=None, description="SSL CA path")


class RedisConfig(BaseModel):
    """Redis configuration for caching and sessions."""
    
    url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL/TLS")
    
    # Connection pool settings
    max_connections: int = Field(default=50, ge=1, le=500, description="Max connections")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")
    socket_timeout: float = Field(default=5.0, ge=0.1, le=60.0, description="Socket timeout")
    
    # Cache settings
    default_ttl: int = Field(default=3600, ge=1, description="Default TTL in seconds")
    key_prefix: str = Field(default="tektra:", description="Key prefix")


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    # Security levels
    default_security_level: SecurityLevel = Field(default=SecurityLevel.MEDIUM)
    max_security_level: SecurityLevel = Field(default=SecurityLevel.HIGH)
    
    # Encryption
    encryption_key: Optional[str] = Field(default=None, description="Base64 encryption key")
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    
    # Authentication
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, ge=1, le=168, description="JWT expiry in hours")
    
    # Session management
    session_timeout_minutes: int = Field(default=60, ge=5, le=480, description="Session timeout")
    max_concurrent_sessions: int = Field(default=10, ge=1, le=100, description="Max sessions per user")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window seconds")
    
    # CORS
    cors_origins: List[str] = Field(default=["http://localhost:3000"], description="CORS origins")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS methods")
    cors_headers: List[str] = Field(default=["Content-Type", "Authorization"], description="CORS headers")
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if v is not None:
            try:
                # Validate it's a valid base64 Fernet key
                Fernet(v.encode() if isinstance(v, str) else v)
            except Exception:
                raise ValueError("Invalid encryption key format")
        return v


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""
    
    # Prometheus metrics
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8090, ge=1024, le=65535, description="Prometheus port")
    prometheus_path: str = Field(default="/metrics", description="Prometheus metrics path")
    
    # Health checks
    health_check_enabled: bool = Field(default=True, description="Enable health checks")
    health_check_path: str = Field(default="/health", description="Health check path")
    health_check_interval: float = Field(default=30.0, ge=1.0, le=300.0, description="Health check interval")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(default="json", description="Log format (json/text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_rotation: str = Field(default="1 day", description="Log rotation")
    log_retention: str = Field(default="30 days", description="Log retention")
    
    # Tracing
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    tracing_endpoint: Optional[str] = Field(default=None, description="Tracing endpoint")
    tracing_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Tracing sample rate")
    
    # Alerting
    alerting_enabled: bool = Field(default=True, description="Enable alerting")
    alert_webhook_url: Optional[str] = Field(default=None, description="Alert webhook URL")
    alert_email: Optional[str] = Field(default=None, description="Alert email")


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    
    # Memory management
    max_memory_mb: int = Field(default=4096, ge=512, le=32768, description="Max memory MB")
    memory_warning_threshold: float = Field(default=0.8, ge=0.1, le=0.95, description="Memory warning threshold")
    memory_critical_threshold: float = Field(default=0.9, ge=0.1, le=0.99, description="Memory critical threshold")
    
    # Caching
    cache_size_mb: int = Field(default=1024, ge=64, le=8192, description="Cache size MB")
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400, description="Cache TTL seconds")
    cache_compression: bool = Field(default=True, description="Enable cache compression")
    
    # Task scheduling
    max_workers: int = Field(default=10, ge=1, le=100, description="Max worker threads")
    task_queue_size: int = Field(default=1000, ge=10, le=10000, description="Task queue size")
    task_timeout_seconds: float = Field(default=300.0, ge=1.0, le=3600.0, description="Task timeout")
    
    # Connection pooling
    max_connections: int = Field(default=50, ge=5, le=500, description="Max connections")
    connection_timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="Connection timeout")
    keepalive_timeout: float = Field(default=300.0, ge=10.0, le=3600.0, description="Keepalive timeout")


class AgentConfig(BaseModel):
    """Agent configuration."""
    
    # Agent limits
    max_concurrent_agents: int = Field(default=20, ge=1, le=1000, description="Max concurrent agents")
    agent_timeout_seconds: float = Field(default=600.0, ge=10.0, le=3600.0, description="Agent timeout")
    max_agent_memory_mb: int = Field(default=512, ge=64, le=2048, description="Max agent memory MB")
    
    # Model configuration
    default_model: str = Field(default="gpt-3.5-turbo", description="Default model")
    model_timeout_seconds: float = Field(default=120.0, ge=10.0, le=600.0, description="Model timeout")
    model_cache_size: int = Field(default=5, ge=1, le=20, description="Model cache size")
    
    # Tool execution
    tool_execution_timeout: float = Field(default=60.0, ge=1.0, le=300.0, description="Tool execution timeout")
    max_tool_output_size: int = Field(default=1048576, ge=1024, description="Max tool output size bytes")
    sandbox_enabled: bool = Field(default=True, description="Enable sandboxing")
    
    # SmolAgents specific
    smolagents_model_path: Optional[str] = Field(default=None, description="SmolAgents model path")
    smolagents_cache_dir: Optional[str] = Field(default=None, description="SmolAgents cache directory")


class ProductionConfig(BaseSettings):
    """
    Production configuration with environment variable support.
    
    Configuration is loaded from:
    1. Environment variables (highest priority)
    2. .env files
    3. Default values (lowest priority)
    """
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="TEKTRA_ENV")
    debug: bool = Field(default=False, env="TEKTRA_DEBUG")
    testing: bool = Field(default=False, env="TEKTRA_TESTING")
    
    # Application
    app_name: str = Field(default="Tektra AI Assistant", env="TEKTRA_APP_NAME")
    app_version: str = Field(default="1.0.0", env="TEKTRA_VERSION")
    bind_host: str = Field(default="0.0.0.0", env="TEKTRA_HOST")
    bind_port: int = Field(default=8000, ge=1024, le=65535, env="TEKTRA_PORT")
    
    # Paths
    data_dir: str = Field(default="./data", env="TEKTRA_DATA_DIR")
    logs_dir: str = Field(default="./logs", env="TEKTRA_LOGS_DIR")
    temp_dir: str = Field(default="/tmp", env="TEKTRA_TEMP_DIR")
    config_dir: str = Field(default="./config", env="TEKTRA_CONFIG_DIR")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    
    # Feature flags
    enable_agents: bool = Field(default=True, env="TEKTRA_ENABLE_AGENTS")
    enable_security: bool = Field(default=True, env="TEKTRA_ENABLE_SECURITY")
    enable_performance: bool = Field(default=True, env="TEKTRA_ENABLE_PERFORMANCE")
    enable_monitoring: bool = Field(default=True, env="TEKTRA_ENABLE_MONITORING")
    enable_ui: bool = Field(default=True, env="TEKTRA_ENABLE_UI")
    enable_api: bool = Field(default=True, env="TEKTRA_ENABLE_API")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Environment variable nesting
        env_nested_delimiter = "__"
        
        # Example: TEKTRA_DATABASE__URL=postgresql://...
        # Maps to: database.url
    
    @root_validator
    def validate_environment_config(cls, values):
        """Validate configuration based on environment."""
        env = values.get("environment", Environment.DEVELOPMENT)
        
        # Production-specific validations
        if env == Environment.PRODUCTION:
            # Require strong security in production
            security = values.get("security", SecurityConfig())
            if security.default_security_level == SecurityLevel.LOW:
                raise ValueError("Cannot use LOW security level in production")
            
            # Require monitoring in production
            monitoring = values.get("monitoring", MonitoringConfig())
            if not monitoring.prometheus_enabled:
                logger.warning("Prometheus disabled in production - monitoring recommended")
            
            # Disable debug in production
            if values.get("debug", False):
                logger.warning("Debug mode should be disabled in production")
                values["debug"] = False
        
        # Development-specific settings
        elif env == Environment.DEVELOPMENT:
            # Allow debug mode
            values["debug"] = values.get("debug", True)
        
        return values
    
    @validator('data_dir', 'logs_dir', 'config_dir')
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        if v:
            Path(v).mkdir(parents=True, exist_ok=True)
        return v
    
    def to_tektra_config(self) -> TektraSystemConfig:
        """Convert to TektraSystemConfig for the system."""
        return TektraSystemConfig(
            system_name=self.app_name,
            environment=self.environment.value,
            debug_mode=self.debug,
            
            # Component configuration
            enable_agents=self.enable_agents,
            enable_performance=self.enable_performance,
            enable_security=self.enable_security,
            enable_monitoring=self.enable_monitoring,
            
            # Agent configuration
            max_concurrent_agents=self.agents.max_concurrent_agents,
            agent_timeout_seconds=self.agents.agent_timeout_seconds,
            
            # Performance configuration
            cache_size_mb=self.performance.cache_size_mb,
            memory_limit_mb=self.performance.max_memory_mb,
            task_queue_size=self.performance.task_queue_size,
            
            # Security configuration
            security_level=self.security.default_security_level,
            sandbox_enabled=self.agents.sandbox_enabled,
            tool_validation_enabled=True,
            consent_mode="interactive" if self.environment == Environment.PRODUCTION else "automatic",
            
            # Monitoring configuration
            metrics_enabled=self.monitoring.prometheus_enabled,
            prometheus_port=self.monitoring.prometheus_port,
            health_check_interval=self.monitoring.health_check_interval,
            
            # Integration configuration
            ui_enabled=self.enable_ui,
            api_enabled=self.enable_api,
            websocket_enabled=True
        )
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> "ProductionConfig":
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_secrets(self) -> Dict[str, str]:
        """Get sensitive configuration values that should be handled securely."""
        return {
            "database_url": self.database.url if "password" in self.database.url else "",
            "redis_password": self.redis.password or "",
            "encryption_key": self.security.encryption_key or "",
            "jwt_secret": self.security.jwt_secret,
            "secret_key": self.security.secret_key
        }
    
    def mask_secrets(self) -> Dict[str, Any]:
        """Get configuration with secrets masked for logging."""
        config_dict = self.dict()
        
        # Mask sensitive fields
        sensitive_paths = [
            ["database", "url"],
            ["redis", "password"],
            ["security", "encryption_key"],
            ["security", "jwt_secret"],
            ["security", "secret_key"]
        ]
        
        for path in sensitive_paths:
            current = config_dict
            for key in path[:-1]:
                if key in current and isinstance(current[key], dict):
                    current = current[key]
                else:
                    break
            else:
                if path[-1] in current and current[path[-1]]:
                    current[path[-1]] = "***MASKED***"
        
        return config_dict


def load_production_config(
    config_file: Optional[Union[str, Path]] = None,
    env_file: Optional[Union[str, Path]] = None
) -> ProductionConfig:
    """
    Load production configuration from various sources.
    
    Args:
        config_file: Optional JSON configuration file
        env_file: Optional .env file path
        
    Returns:
        Loaded configuration
    """
    # Set environment file if provided
    if env_file:
        os.environ["TEKTRA_ENV_FILE"] = str(env_file)
    
    # Load from file if provided
    if config_file and Path(config_file).exists():
        config = ProductionConfig.load_from_file(config_file)
    else:
        # Load from environment and defaults
        config = ProductionConfig()
    
    # Validate configuration
    logger.info(f"Loaded configuration for environment: {config.environment.value}")
    logger.debug(f"Configuration: {config.mask_secrets()}")
    
    return config


def create_development_config() -> ProductionConfig:
    """Create a development configuration."""
    return ProductionConfig(
        environment=Environment.DEVELOPMENT,
        debug=True,
        
        # Development database
        database=DatabaseConfig(
            url="sqlite:///tektra_dev.db",
            echo=True
        ),
        
        # Development Redis
        redis=RedisConfig(
            url="redis://localhost:6379/1"
        ),
        
        # Development security
        security=SecurityConfig(
            default_security_level=SecurityLevel.MEDIUM,
            cors_origins=["http://localhost:3000", "http://localhost:8080"]
        ),
        
        # Development monitoring
        monitoring=MonitoringConfig(
            log_level=LogLevel.DEBUG,
            prometheus_port=8091
        ),
        
        # Development performance
        performance=PerformanceConfig(
            max_memory_mb=2048,
            cache_size_mb=256,
            max_workers=5
        ),
        
        # Development agents
        agents=AgentConfig(
            max_concurrent_agents=5,
            agent_timeout_seconds=120.0
        )
    )


def create_production_config() -> ProductionConfig:
    """Create a production configuration template."""
    return ProductionConfig(
        environment=Environment.PRODUCTION,
        debug=False,
        
        # Production database (should be overridden with env vars)
        database=DatabaseConfig(
            url="postgresql://tektra:password@localhost:5432/tektra",
            pool_size=20,
            ssl_required=True
        ),
        
        # Production Redis
        redis=RedisConfig(
            url="redis://localhost:6379/0",
            ssl=True,
            max_connections=100
        ),
        
        # Production security
        security=SecurityConfig(
            default_security_level=SecurityLevel.HIGH,
            cors_origins=["https://tektra.example.com"]
        ),
        
        # Production monitoring
        monitoring=MonitoringConfig(
            log_level=LogLevel.INFO,
            prometheus_enabled=True,
            tracing_enabled=True,
            alerting_enabled=True
        ),
        
        # Production performance
        performance=PerformanceConfig(
            max_memory_mb=8192,
            cache_size_mb=2048,
            max_workers=20
        ),
        
        # Production agents
        agents=AgentConfig(
            max_concurrent_agents=50,
            agent_timeout_seconds=600.0
        )
    )


if __name__ == "__main__":
    def demo_configuration():
        """Demonstrate configuration management."""
        print("⚙️ Tektra Configuration Management Demo")
        print("=" * 50)
        
        # Development configuration
        print("\n📋 Development Configuration:")
        dev_config = create_development_config()
        print(f"  Environment: {dev_config.environment.value}")
        print(f"  Debug Mode: {dev_config.debug}")
        print(f"  Security Level: {dev_config.security.default_security_level.value}")
        print(f"  Max Agents: {dev_config.agents.max_concurrent_agents}")
        print(f"  Cache Size: {dev_config.performance.cache_size_mb}MB")
        
        # Production configuration
        print("\n🏭 Production Configuration:")
        prod_config = create_production_config()
        print(f"  Environment: {prod_config.environment.value}")
        print(f"  Debug Mode: {prod_config.debug}")
        print(f"  Security Level: {prod_config.security.default_security_level.value}")
        print(f"  Max Agents: {prod_config.agents.max_concurrent_agents}")
        print(f"  Cache Size: {prod_config.performance.cache_size_mb}MB")
        
        # Convert to TektraSystemConfig
        print("\n🔄 Tektra System Config Conversion:")
        tektra_config = dev_config.to_tektra_config()
        print(f"  System Name: {tektra_config.system_name}")
        print(f"  Environment: {tektra_config.environment}")
        print(f"  Security Level: {tektra_config.security_level.value}")
        print(f"  Components Enabled: Agents={tektra_config.enable_agents}, "
              f"Security={tektra_config.enable_security}, "
              f"Performance={tektra_config.enable_performance}")
        
        # Configuration masking
        print("\n🔐 Configuration Security:")
        masked = dev_config.mask_secrets()
        secrets = dev_config.get_secrets()
        print(f"  Secrets detected: {len([k for k, v in secrets.items() if v])}")
        print(f"  Masking applied: {any('***MASKED***' in str(v) for v in masked.values())}")
        
        # Save and load
        print("\n💾 Configuration Persistence:")
        config_file = "test_config.json"
        dev_config.save_to_file(config_file)
        loaded_config = ProductionConfig.load_from_file(config_file)
        print(f"  Saved and loaded: {loaded_config.environment.value}")
        
        # Clean up
        Path(config_file).unlink(missing_ok=True)
        
        print("\n⚙️ Configuration Demo Complete")
    
    demo_configuration()