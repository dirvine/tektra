"""
Enhanced Model Management Interface

This module provides a standardized interface for AI models, making it easy
to swap between different models and maintain consistent behavior.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


class ModelType(Enum):
    """Types of AI models supported."""
    TEXT_GENERATION = "text_generation"
    VISION_LANGUAGE = "vision_language"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"


class ModelStatus(Enum):
    """Model loading and execution status."""
    NOT_LOADED = "not_loaded"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    UNLOADING = "unloading"


class QuantizationType(Enum):
    """Model quantization options."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    FLOAT16 = "float16"
    AUTO = "auto"


@dataclass
class ModelCapabilities:
    """Model capabilities and features."""
    supports_streaming: bool = False
    supports_vision: bool = False
    supports_audio: bool = False
    supports_code: bool = False
    supports_function_calling: bool = False
    max_context_length: int = 2048
    max_output_length: int = 512
    languages: List[str] = None
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en"]


@dataclass
class ModelMetadata:
    """Model metadata and configuration."""
    name: str
    display_name: str
    version: str
    model_type: ModelType
    provider: str
    description: str
    capabilities: ModelCapabilities
    size_mb: int
    memory_requirement_mb: int
    download_url: Optional[str] = None
    checksum: Optional[str] = None
    license: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetadata':
        """Create ModelMetadata from dictionary."""
        if 'capabilities' in data and isinstance(data['capabilities'], dict):
            data['capabilities'] = ModelCapabilities(**data['capabilities'])
        if 'model_type' in data and isinstance(data['model_type'], str):
            data['model_type'] = ModelType(data['model_type'])
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert ModelMetadata to dictionary."""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        return result


@dataclass
class ModelConfig:
    """Model configuration for initialization."""
    model_name: str
    device: str = "auto"
    quantization: QuantizationType = QuantizationType.AUTO
    max_memory: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 512
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class ModelPerformance:
    """Model performance metrics."""
    avg_latency_ms: float = 0.0
    avg_throughput_tokens_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    total_requests: int = 0
    total_tokens_generated: int = 0
    error_count: int = 0
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()


class ModelInterface(ABC):
    """
    Abstract base class for all AI models.
    
    This interface ensures consistent behavior across different model types
    and providers, enabling easy model swapping and management.
    """
    
    def __init__(self, metadata: ModelMetadata, config: ModelConfig):
        """
        Initialize model interface.
        
        Args:
            metadata: Model metadata and capabilities
            config: Model configuration
        """
        self.metadata = metadata
        self.config = config
        self.status = ModelStatus.NOT_LOADED
        self.performance = ModelPerformance()
        self._last_request_time = 0.0
        
    @abstractmethod
    async def load(self) -> bool:
        """
        Load the model and prepare for inference.
        
        Returns:
            bool: True if model loaded successfully
        """
        pass
    
    @abstractmethod
    async def unload(self) -> bool:
        """
        Unload the model and free resources.
        
        Returns:
            bool: True if model unloaded successfully
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: Any, **kwargs) -> Any:
        """
        Process input and return output.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional processing parameters
            
        Returns:
            Any: Model output
        """
        pass
    
    async def process_stream(self, input_data: Any, **kwargs):
        """
        Process input and yield streaming output.
        
        Args:
            input_data: Input data for processing
            **kwargs: Additional processing parameters
            
        Yields:
            Any: Streaming model output
        """
        # Default implementation for non-streaming models
        result = await self.process(input_data, **kwargs)
        yield result
    
    def get_capabilities(self) -> ModelCapabilities:
        """Get model capabilities."""
        return self.metadata.capabilities
    
    def get_status(self) -> ModelStatus:
        """Get current model status."""
        return self.status
    
    def get_metadata(self) -> ModelMetadata:
        """Get model metadata."""
        return self.metadata
    
    def get_performance(self) -> ModelPerformance:
        """Get model performance metrics."""
        return self.performance
    
    def is_ready(self) -> bool:
        """Check if model is ready for inference."""
        return self.status == ModelStatus.READY
    
    def supports_feature(self, feature: str) -> bool:
        """
        Check if model supports a specific feature.
        
        Args:
            feature: Feature name to check
            
        Returns:
            bool: True if feature is supported
        """
        capabilities = self.get_capabilities()
        feature_map = {
            "streaming": capabilities.supports_streaming,
            "vision": capabilities.supports_vision,
            "audio": capabilities.supports_audio,
            "code": capabilities.supports_code,
            "function_calling": capabilities.supports_function_calling
        }
        return feature_map.get(feature, False)
    
    def _update_performance(self, request_time: float, tokens_generated: int = 0, error: bool = False):
        """Update performance metrics."""
        self.performance.total_requests += 1
        
        if error:
            self.performance.error_count += 1
        else:
            # Update latency (exponential moving average)
            if self.performance.avg_latency_ms == 0:
                self.performance.avg_latency_ms = request_time * 1000
            else:
                alpha = 0.1  # Smoothing factor
                self.performance.avg_latency_ms = (
                    alpha * request_time * 1000 + 
                    (1 - alpha) * self.performance.avg_latency_ms
                )
            
            # Update throughput
            if tokens_generated > 0:
                self.performance.total_tokens_generated += tokens_generated
                if request_time > 0:
                    current_throughput = tokens_generated / request_time
                    if self.performance.avg_throughput_tokens_per_sec == 0:
                        self.performance.avg_throughput_tokens_per_sec = current_throughput
                    else:
                        self.performance.avg_throughput_tokens_per_sec = (
                            alpha * current_throughput + 
                            (1 - alpha) * self.performance.avg_throughput_tokens_per_sec
                        )
        
        self.performance.last_updated = datetime.now()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform model health check.
        
        Returns:
            Dict containing health status and metrics
        """
        try:
            if not self.is_ready():
                return {
                    "healthy": False,
                    "status": self.status.value,
                    "message": "Model not ready"
                }
            
            # Simple test inference
            start_time = time.time()
            test_result = await self.process("test")
            end_time = time.time()
            
            return {
                "healthy": True,
                "status": self.status.value,
                "test_latency_ms": (end_time - start_time) * 1000,
                "memory_usage_mb": self.performance.memory_usage_mb,
                "total_requests": self.performance.total_requests,
                "error_rate": (
                    self.performance.error_count / max(1, self.performance.total_requests)
                ) * 100
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "status": self.status.value,
                "error": str(e)
            }
    
    def __str__(self) -> str:
        """String representation of the model."""
        return f"{self.metadata.display_name} ({self.metadata.version}) - {self.status.value}"
    
    def __repr__(self) -> str:
        """Detailed representation of the model."""
        return (
            f"ModelInterface({self.metadata.name}, "
            f"status={self.status.value}, "
            f"type={self.metadata.model_type.value})"
        )


class ModelRegistry:
    """
    Central registry for managing AI models.
    
    Features:
    - Model discovery and registration
    - Version management
    - Update notifications
    - Model comparison and selection
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to model registry file
        """
        self.registry_path = registry_path or Path.home() / ".tektra" / "model_registry.json"
        self.models: Dict[str, ModelMetadata] = {}
        self.update_channels: Dict[str, str] = {
            "stable": "https://registry.tektra.ai/stable/",
            "beta": "https://registry.tektra.ai/beta/",
            "experimental": "https://registry.tektra.ai/experimental/"
        }
        
        self._load_registry()
    
    def _load_registry(self):
        """Load model registry from file."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    data = json.load(f)
                    self.models = {
                        name: ModelMetadata.from_dict(meta)
                        for name, meta in data.get('models', {}).items()
                    }
                    self.update_channels.update(data.get('update_channels', {}))
        except Exception as e:
            logger.warning(f"Failed to load model registry: {e}")
    
    def _save_registry(self):
        """Save model registry to file."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'models': {name: meta.to_dict() for name, meta in self.models.items()},
                'update_channels': self.update_channels
            }
            with open(self.registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, metadata: ModelMetadata):
        """
        Register a new model in the registry.
        
        Args:
            metadata: Model metadata to register
        """
        self.models[metadata.name] = metadata
        self._save_registry()
        logger.info(f"Registered model: {metadata.name}")
    
    def get_model(self, name: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by name.
        
        Args:
            name: Model name
            
        Returns:
            ModelMetadata if found, None otherwise
        """
        return self.models.get(name)
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """
        List available models.
        
        Args:
            model_type: Filter by model type
            
        Returns:
            List of model metadata
        """
        models = list(self.models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return sorted(models, key=lambda m: m.display_name)
    
    def find_models_by_capability(self, capability: str) -> List[ModelMetadata]:
        """
        Find models that support a specific capability.
        
        Args:
            capability: Capability to search for
            
        Returns:
            List of compatible models
        """
        compatible_models = []
        for model in self.models.values():
            if hasattr(model.capabilities, f"supports_{capability}"):
                if getattr(model.capabilities, f"supports_{capability}"):
                    compatible_models.append(model)
        return compatible_models
    
    async def check_for_updates(self, channel: str = "stable") -> List[Dict[str, Any]]:
        """
        Check for model updates.
        
        Args:
            channel: Update channel to check
            
        Returns:
            List of available updates
        """
        # TODO: Implement remote registry checking
        logger.info(f"Checking for updates on {channel} channel")
        return []
    
    def get_recommended_models(self, 
                             use_case: str, 
                             memory_limit_mb: Optional[int] = None) -> List[ModelMetadata]:
        """
        Get recommended models for a specific use case.
        
        Args:
            use_case: Use case description
            memory_limit_mb: Memory limit constraint
            
        Returns:
            List of recommended models
        """
        # Simple recommendation logic
        # TODO: Implement more sophisticated recommendation system
        suitable_models = []
        
        for model in self.models.values():
            # Check memory constraint
            if memory_limit_mb and model.memory_requirement_mb > memory_limit_mb:
                continue
            
            # Simple keyword matching for use case
            use_case_lower = use_case.lower()
            if "vision" in use_case_lower and model.capabilities.supports_vision:
                suitable_models.append(model)
            elif "text" in use_case_lower and model.model_type == ModelType.TEXT_GENERATION:
                suitable_models.append(model)
            elif "code" in use_case_lower and model.capabilities.supports_code:
                suitable_models.append(model)
        
        # Sort by size (smaller models first for efficiency)
        return sorted(suitable_models, key=lambda m: m.size_mb)


class ModelFactory:
    """
    Factory for creating model instances.
    
    This factory abstracts model creation and provides a consistent
    interface for instantiating different model types.
    """
    
    def __init__(self, registry: ModelRegistry):
        """
        Initialize model factory.
        
        Args:
            registry: Model registry for metadata lookup
        """
        self.registry = registry
        self._model_classes: Dict[str, type] = {}
    
    def register_model_class(self, model_name: str, model_class: type):
        """
        Register a model implementation class.
        
        Args:
            model_name: Model name identifier
            model_class: Model implementation class
        """
        self._model_classes[model_name] = model_class
        logger.info(f"Registered model class for: {model_name}")
    
    async def create_model(self, 
                          model_name: str, 
                          config: Optional[ModelConfig] = None) -> Optional[ModelInterface]:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration
            
        Returns:
            ModelInterface instance if successful, None otherwise
        """
        try:
            # Get metadata from registry
            metadata = self.registry.get_model(model_name)
            if not metadata:
                logger.error(f"Model not found in registry: {model_name}")
                return None
            
            # Get model class
            model_class = self._model_classes.get(model_name)
            if not model_class:
                logger.error(f"No implementation class registered for: {model_name}")
                return None
            
            # Use provided config or create default
            if config is None:
                config = ModelConfig(model_name=model_name)
            
            # Create model instance
            model = model_class(metadata, config)
            logger.info(f"Created model instance: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """
        Get list of models with registered implementation classes.
        
        Returns:
            List of available model names
        """
        return list(self._model_classes.keys())


# Initialize default registry
default_registry = ModelRegistry()

# Initialize default factory
default_factory = ModelFactory(default_registry)