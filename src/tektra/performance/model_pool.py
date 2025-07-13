#!/usr/bin/env python3
"""
Model Pool Management

Specialized resource pool for managing ML model instances with:
- Lazy loading and unloading
- Memory-aware allocation
- Model versioning support
- Multi-GPU distribution
- Shared memory optimization

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run with dependencies: uv run --with torch,psutil,loguru python model_pool.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.0.0",
#     "psutil>=5.9.0",
#     "loguru>=0.7.0",
# ]
# ///

import gc
import mmap
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import threading
import weakref

import psutil
import torch
from loguru import logger

from .resource_pool import ResourcePool, ResourceType, create_resource_pool


@dataclass
class ModelInfo:
    """Information about a pooled model."""
    
    model_name: str
    model_type: str
    version: str
    
    # Resource requirements
    memory_mb: float
    requires_gpu: bool
    gpu_memory_mb: float = 0.0
    
    # Loading information
    load_path: Optional[Path] = None
    load_time_seconds: float = 0.0
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    
    # Usage statistics
    total_inferences: int = 0
    total_inference_time: float = 0.0
    last_inference: Optional[float] = None
    
    @property
    def average_inference_time(self) -> float:
        """Get average inference time in seconds."""
        if self.total_inferences == 0:
            return 0.0
        return self.total_inference_time / self.total_inferences


@dataclass 
class PooledModel:
    """Wrapper for a pooled model instance."""
    
    model_id: str
    model: Any  # The actual model instance
    info: ModelInfo
    
    # Memory mapping for shared weights
    mmap_file: Optional[mmap.mmap] = None
    shared_tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    # Reference counting for cleanup
    ref_count: int = 0
    weak_refs: Set[weakref.ReferenceType] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize model on the correct device."""
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.info.device)
            if hasattr(self.model, 'eval'):
                self.model.eval()  # Set to evaluation mode


class ModelPool:
    """
    Specialized pool for managing ML model instances.
    
    Features:
    - Memory-aware model loading/unloading
    - GPU memory management
    - Model versioning and variants
    - Shared memory optimization
    - Automatic quantization support
    """
    
    def __init__(
        self,
        max_models: int = 5,
        max_memory_gb: float = 8.0,
        max_gpu_memory_gb: float = 0.0,
        enable_sharing: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize model pool.
        
        Args:
            max_models: Maximum number of models to keep loaded
            max_memory_gb: Maximum CPU memory to use (GB)
            max_gpu_memory_gb: Maximum GPU memory to use (GB)
            enable_sharing: Enable memory sharing between models
            cache_dir: Directory for caching models
        """
        self.max_models = max_models
        self.max_memory_gb = max_memory_gb
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.enable_sharing = enable_sharing
        self.cache_dir = cache_dir or Path.home() / ".cache" / "tektra" / "models"
        
        # Model registry
        self.model_registry: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, PooledModel] = {}
        
        # Memory tracking
        self.current_memory_mb = 0.0
        self.current_gpu_memory_mb = 0.0
        
        # Device management
        self.available_devices = self._detect_devices()
        self.device_usage: Dict[str, float] = {device: 0.0 for device in self.available_devices}
        
        # Shared memory management
        self.shared_memory_maps: Dict[str, mmap.mmap] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize resource pools for different model types
        self._pools: Dict[str, ResourcePool] = {}
        
        logger.info(f"Model pool initialized with {len(self.available_devices)} devices")
    
    def _detect_devices(self) -> List[str]:
        """Detect available compute devices."""
        devices = ["cpu"]
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
        
        return devices
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        memory_mb: float,
        requires_gpu: bool = False,
        gpu_memory_mb: float = 0.0,
        load_path: Optional[Path] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a model with the pool.
        
        Args:
            model_name: Unique name for the model
            model_type: Type of model (e.g., "transformers", "torch")
            version: Model version
            memory_mb: Estimated memory usage in MB
            requires_gpu: Whether model requires GPU
            gpu_memory_mb: GPU memory usage in MB
            load_path: Path to model files
            config: Model configuration
        """
        with self._lock:
            info = ModelInfo(
                model_name=model_name,
                model_type=model_type,
                version=version,
                memory_mb=memory_mb,
                requires_gpu=requires_gpu,
                gpu_memory_mb=gpu_memory_mb,
                load_path=load_path,
                config=config or {}
            )
            
            self.model_registry[model_name] = info
            logger.info(f"Registered model: {model_name} v{version}")
    
    def _get_optimal_device(self, info: ModelInfo) -> str:
        """Get optimal device for loading a model."""
        if not info.requires_gpu:
            return "cpu"
        
        # Find GPU with most available memory
        best_device = "cpu"
        best_available = 0.0
        
        for device in self.available_devices:
            if device.startswith("cuda"):
                try:
                    gpu_id = int(device.split(":")[1])
                    torch.cuda.set_device(gpu_id)
                    
                    # Get available memory
                    total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**2)
                    used = self.device_usage.get(device, 0.0)
                    available = total - used
                    
                    if available > best_available and available >= info.gpu_memory_mb:
                        best_device = device
                        best_available = available
                        
                except Exception as e:
                    logger.debug(f"Error checking device {device}: {e}")
        
        return best_device
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        total_size = 0
        
        try:
            # For PyTorch models
            if hasattr(model, 'parameters'):
                for param in model.parameters():
                    total_size += param.data.nelement() * param.data.element_size()
            
            # For other attributes
            elif hasattr(model, '__dict__'):
                for attr_value in model.__dict__.values():
                    if isinstance(attr_value, torch.Tensor):
                        total_size += attr_value.nelement() * attr_value.element_size()
            
            return total_size / (1024**2)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not estimate model memory: {e}")
            return 100.0  # Default estimate
    
    def _create_shared_memory(self, model_name: str, tensors: Dict[str, torch.Tensor]) -> Tuple[mmap.mmap, Dict[str, torch.Tensor]]:
        """Create shared memory mapping for model tensors."""
        if not self.enable_sharing:
            return None, {}
        
        try:
            # Calculate total size needed
            total_size = 0
            tensor_info = []
            
            for name, tensor in tensors.items():
                size = tensor.nelement() * tensor.element_size()
                tensor_info.append((name, tensor.shape, tensor.dtype, size))
                total_size += size
            
            # Create memory-mapped file
            mmap_path = self.cache_dir / f"{model_name}_shared.mmap"
            
            with open(mmap_path, 'wb') as f:
                f.write(b'\0' * total_size)
            
            # Create memory map
            mmap_file = mmap.mmap(open(mmap_path, 'r+b').fileno(), total_size)
            
            # Create shared tensors
            shared_tensors = {}
            offset = 0
            
            for name, shape, dtype, size in tensor_info:
                # Create tensor from memory map
                buffer = memoryview(mmap_file)[offset:offset + size]
                shared_tensor = torch.frombuffer(buffer, dtype=dtype).reshape(shape)
                shared_tensors[name] = shared_tensor
                
                # Copy original data
                shared_tensor.copy_(tensors[name])
                offset += size
            
            logger.debug(f"Created shared memory for {model_name}: {total_size / (1024**2):.1f}MB")
            return mmap_file, shared_tensors
            
        except Exception as e:
            logger.warning(f"Failed to create shared memory: {e}")
            return None, {}
    
    async def load_model(self, model_name: str, loader_func=None) -> str:
        """
        Load a model into the pool.
        
        Args:
            model_name: Name of the model to load
            loader_func: Optional custom loader function
            
        Returns:
            Model ID
        """
        with self._lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                model = self.loaded_models[model_name]
                model.ref_count += 1
                logger.debug(f"Model {model_name} already loaded (refs={model.ref_count})")
                return model.model_id
            
            # Get model info
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not registered")
            
            info = self.model_registry[model_name]
            
            # Check memory constraints
            if self.current_memory_mb + info.memory_mb > self.max_memory_gb * 1024:
                # Try to evict unused models
                self._evict_unused_models()
                
                # Check again
                if self.current_memory_mb + info.memory_mb > self.max_memory_gb * 1024:
                    raise MemoryError(f"Insufficient memory to load model {model_name}")
            
            # Select device
            device = self._get_optimal_device(info)
            info.device = device
            
            # Load the model
            start_time = time.time()
            
            try:
                if loader_func:
                    model = loader_func(info)
                else:
                    model = self._default_loader(info)
                
                # Move to device
                if hasattr(model, 'to'):
                    model = model.to(device)
                
                info.load_time_seconds = time.time() - start_time
                
                # Create pooled model
                model_id = f"{model_name}_{int(time.time() * 1000)}"
                pooled = PooledModel(
                    model_id=model_id,
                    model=model,
                    info=info,
                    ref_count=1
                )
                
                # Try to create shared memory
                if self.enable_sharing and hasattr(model, 'state_dict'):
                    state_dict = model.state_dict()
                    mmap_file, shared_tensors = self._create_shared_memory(model_name, state_dict)
                    pooled.mmap_file = mmap_file
                    pooled.shared_tensors = shared_tensors
                
                # Update tracking
                self.loaded_models[model_name] = pooled
                self.current_memory_mb += info.memory_mb
                
                if device.startswith("cuda"):
                    self.current_gpu_memory_mb += info.gpu_memory_mb
                    self.device_usage[device] += info.gpu_memory_mb
                
                logger.info(f"Loaded model {model_name} on {device} in {info.load_time_seconds:.2f}s")
                return model_id
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    def _default_loader(self, info: ModelInfo) -> Any:
        """Default model loader."""
        if not info.load_path:
            raise ValueError(f"No load path specified for model {info.model_name}")
        
        # Try to load as PyTorch model
        try:
            model = torch.load(info.load_path, map_location='cpu')
            logger.debug(f"Loaded PyTorch model from {info.load_path}")
            return model
        except Exception as e:
            logger.debug(f"Not a PyTorch model: {e}")
        
        # Try to load as pickle
        try:
            with open(info.load_path, 'rb') as f:
                model = pickle.load(f)
            logger.debug(f"Loaded pickled model from {info.load_path}")
            return model
        except Exception as e:
            logger.debug(f"Not a pickled model: {e}")
        
        raise ValueError(f"Could not load model from {info.load_path}")
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a model from the pool.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model instance
        """
        with self._lock:
            if model_name not in self.loaded_models:
                raise ValueError(f"Model {model_name} not loaded")
            
            pooled = self.loaded_models[model_name]
            pooled.info.last_inference = time.time()
            
            return pooled.model
    
    def release_model(self, model_name: str) -> None:
        """
        Release a reference to a model.
        
        Args:
            model_name: Name of the model
        """
        with self._lock:
            if model_name not in self.loaded_models:
                return
            
            pooled = self.loaded_models[model_name]
            pooled.ref_count = max(0, pooled.ref_count - 1)
            
            logger.debug(f"Released model {model_name} (refs={pooled.ref_count})")
    
    def unload_model(self, model_name: str, force: bool = False) -> None:
        """
        Unload a model from the pool.
        
        Args:
            model_name: Name of the model
            force: Force unload even if references exist
        """
        with self._lock:
            if model_name not in self.loaded_models:
                return
            
            pooled = self.loaded_models[model_name]
            
            if pooled.ref_count > 0 and not force:
                logger.warning(f"Cannot unload model {model_name} with {pooled.ref_count} references")
                return
            
            # Clean up shared memory
            if pooled.mmap_file:
                pooled.mmap_file.close()
            
            # Update tracking
            self.current_memory_mb -= pooled.info.memory_mb
            
            if pooled.info.device.startswith("cuda"):
                self.current_gpu_memory_mb -= pooled.info.gpu_memory_mb
                self.device_usage[pooled.info.device] -= pooled.info.gpu_memory_mb
            
            # Delete model
            del pooled.model
            del self.loaded_models[model_name]
            
            # Force garbage collection
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model {model_name}")
    
    def _evict_unused_models(self) -> None:
        """Evict models with no references."""
        to_evict = []
        
        for model_name, pooled in self.loaded_models.items():
            if pooled.ref_count == 0:
                to_evict.append(model_name)
        
        # Sort by last inference time (oldest first)
        to_evict.sort(key=lambda name: self.loaded_models[name].info.last_inference or 0)
        
        for model_name in to_evict:
            self.unload_model(model_name, force=True)
            
            # Check if we have enough memory now
            if self.current_memory_mb < self.max_memory_gb * 1024 * 0.9:  # 90% threshold
                break
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage across all models."""
        with self._lock:
            initial_memory = self.current_memory_mb
            initial_gpu_memory = self.current_gpu_memory_mb
            
            # Garbage collect
            gc.collect()
            
            # Clear PyTorch caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Evict unused models
            self._evict_unused_models()
            
            # Compact memory if possible
            if hasattr(gc, 'collect'):
                gc.collect(2)  # Full collection
            
            return {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": self.current_memory_mb,
                "memory_freed_mb": initial_memory - self.current_memory_mb,
                "initial_gpu_memory_mb": initial_gpu_memory,
                "final_gpu_memory_mb": self.current_gpu_memory_mb,
                "gpu_memory_freed_mb": initial_gpu_memory - self.current_gpu_memory_mb,
                "models_evicted": int((initial_memory - self.current_memory_mb) > 0)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get pool status and metrics."""
        with self._lock:
            status = {
                "registered_models": len(self.model_registry),
                "loaded_models": len(self.loaded_models),
                "memory_usage_mb": self.current_memory_mb,
                "memory_limit_mb": self.max_memory_gb * 1024,
                "memory_usage_percent": (self.current_memory_mb / (self.max_memory_gb * 1024)) * 100,
                "gpu_memory_usage_mb": self.current_gpu_memory_mb,
                "gpu_memory_limit_mb": self.max_gpu_memory_gb * 1024,
                "available_devices": self.available_devices,
                "device_usage": self.device_usage.copy(),
                "models": {}
            }
            
            # Add per-model information
            for name, pooled in self.loaded_models.items():
                info = pooled.info
                status["models"][name] = {
                    "model_id": pooled.model_id,
                    "version": info.version,
                    "device": info.device,
                    "memory_mb": info.memory_mb,
                    "gpu_memory_mb": info.gpu_memory_mb,
                    "ref_count": pooled.ref_count,
                    "total_inferences": info.total_inferences,
                    "average_inference_time": info.average_inference_time,
                    "load_time_seconds": info.load_time_seconds,
                    "has_shared_memory": pooled.mmap_file is not None
                }
            
            return status


def create_model_pool(**kwargs) -> ModelPool:
    """
    Create a model pool with the given configuration.
    
    Args:
        **kwargs: Model pool configuration
        
    Returns:
        Configured model pool
    """
    return ModelPool(**kwargs)


if __name__ == "__main__":
    import asyncio
    
    async def demo_model_pool():
        """Demonstrate model pool functionality."""
        print("🤖 Model Pool Demo")
        print("=" * 40)
        
        # Create model pool
        pool = create_model_pool(
            max_models=3,
            max_memory_gb=2.0,
            enable_sharing=True
        )
        
        # Mock model class
        class MockModel:
            def __init__(self, size_mb: int = 100):
                self.weights = torch.randn(size_mb * 256 * 1024 // 4)  # Approximate size
                self.name = f"mock_model_{size_mb}mb"
            
            def forward(self, x):
                return x @ self.weights[:x.shape[-1]]
            
            def to(self, device):
                # Mock device movement
                self.device = device
                return self
            
            def state_dict(self):
                return {"weights": self.weights}
        
        # Register models
        pool.register_model(
            model_name="small_model",
            model_type="mock",
            version="1.0",
            memory_mb=100.0
        )
        
        pool.register_model(
            model_name="medium_model",
            model_type="mock",
            version="1.0",
            memory_mb=300.0
        )
        
        pool.register_model(
            model_name="large_model",
            model_type="mock",
            version="1.0",
            memory_mb=500.0,
            requires_gpu=torch.cuda.is_available(),
            gpu_memory_mb=500.0
        )
        
        print(f"Registered {len(pool.model_registry)} models")
        
        # Custom loader
        def mock_loader(info: ModelInfo) -> MockModel:
            size_mb = int(info.memory_mb)
            return MockModel(size_mb)
        
        # Load models
        print("\nLoading models...")
        
        try:
            # Load small model
            model_id = await pool.load_model("small_model", mock_loader)
            print(f"✅ Loaded small_model: {model_id}")
            
            # Load medium model
            model_id = await pool.load_model("medium_model", mock_loader)
            print(f"✅ Loaded medium_model: {model_id}")
            
            # Try to load large model (might fail due to memory limit)
            try:
                model_id = await pool.load_model("large_model", mock_loader)
                print(f"✅ Loaded large_model: {model_id}")
            except MemoryError as e:
                print(f"❌ Could not load large_model: {e}")
        
        except Exception as e:
            print(f"Error loading models: {e}")
        
        # Check status
        print("\nPool status:")
        status = pool.get_status()
        print(f"   Loaded models: {status['loaded_models']}")
        print(f"   Memory usage: {status['memory_usage_mb']:.1f}MB / {status['memory_limit_mb']:.1f}MB")
        print(f"   Memory usage: {status['memory_usage_percent']:.1f}%")
        
        # Use a model
        if "small_model" in pool.loaded_models:
            print("\nUsing small_model...")
            model = pool.get_model("small_model")
            
            # Simulate inference
            input_data = torch.randn(10, 100)
            start_time = time.time()
            output = model.forward(input_data)
            inference_time = time.time() - start_time
            
            # Update statistics
            pool.loaded_models["small_model"].info.total_inferences += 1
            pool.loaded_models["small_model"].info.total_inference_time += inference_time
            
            print(f"   Inference completed in {inference_time*1000:.2f}ms")
            print(f"   Output shape: {output.shape}")
        
        # Optimize memory
        print("\nOptimizing memory...")
        optimization_result = pool.optimize_memory()
        print(f"   Memory freed: {optimization_result['memory_freed_mb']:.1f}MB")
        
        # Final status
        print("\nFinal pool status:")
        final_status = pool.get_status()
        for name, info in final_status["models"].items():
            print(f"   {name}:")
            print(f"      Device: {info['device']}")
            print(f"      Memory: {info['memory_mb']:.1f}MB")
            print(f"      References: {info['ref_count']}")
            print(f"      Shared memory: {info['has_shared_memory']}")
        
        print("\n🤖 Model Pool Demo Complete")
    
    # Run demo
    asyncio.run(demo_model_pool())