"""
Advanced Memory Monitoring for Qwen Model Loading

This module provides comprehensive memory monitoring and OOM protection
for large language model loading and inference operations.
"""

import gc
import platform
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import psutil
import torch
from loguru import logger

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
    nvml.nvmlInit()
except (ImportError, Exception):
    NVML_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Snapshot of current memory usage across all devices."""
    
    # System memory
    system_total_gb: float
    system_available_gb: float
    system_used_gb: float
    system_percent: float
    
    # Process memory
    process_rss_gb: float
    process_vms_gb: float
    
    # GPU memory (if available)
    gpu_total_gb: float = 0.0
    gpu_available_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_percent: float = 0.0
    gpu_device_name: str = "N/A"
    
    # Torch memory (if CUDA available)
    torch_allocated_gb: float = 0.0
    torch_cached_gb: float = 0.0
    torch_reserved_gb: float = 0.0
    
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    @property
    def total_available_gb(self) -> float:
        """Get total available memory across system and GPU."""
        if self.gpu_available_gb > 0:
            return min(self.system_available_gb, self.gpu_available_gb)
        return self.system_available_gb
    
    @property
    def memory_pressure(self) -> str:
        """Get memory pressure level."""
        if self.system_percent > 90 or self.gpu_percent > 90:
            return "critical"
        elif self.system_percent > 80 or self.gpu_percent > 80:
            return "high"
        elif self.system_percent > 60 or self.gpu_percent > 60:
            return "moderate"
        else:
            return "low"


class AdvancedMemoryMonitor:
    """
    Advanced memory monitoring with OOM protection and automatic optimization.
    
    Features:
    - Real-time memory tracking across CPU and GPU
    - Automatic quantization recommendation
    - OOM prediction and prevention
    - Memory usage optimization
    - Detailed logging and alerting
    """
    
    def __init__(self, 
                 safety_margin_gb: float = 2.0,
                 monitoring_interval: float = 1.0,
                 enable_gpu_monitoring: bool = True):
        """
        Initialize memory monitor.
        
        Args:
            safety_margin_gb: Memory to keep free as safety buffer
            monitoring_interval: How often to check memory (seconds)
            enable_gpu_monitoring: Whether to monitor GPU memory
        """
        self.safety_margin_gb = safety_margin_gb
        self.monitoring_interval = monitoring_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.snapshots: list[MemorySnapshot] = []
        self.max_snapshots = 100
        
        # Callbacks
        self.on_memory_warning: Optional[Callable[[MemorySnapshot], None]] = None
        self.on_oom_prediction: Optional[Callable[[MemorySnapshot], None]] = None
        
        # Initialize GPU monitoring
        self.gpu_handle = None
        if self.enable_gpu_monitoring and NVML_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_name = nvml.nvmlDeviceGetName(self.gpu_handle).decode()
                    logger.info(f"GPU monitoring enabled for: {gpu_name}")
            except Exception as e:
                logger.warning(f"Could not initialize GPU monitoring: {e}")
                self.gpu_handle = None
        
        # Get baseline memory
        self.baseline_snapshot = self.get_memory_snapshot()
        logger.info(f"Memory monitor initialized - Available: {self.baseline_snapshot.total_available_gb:.1f}GB")
    
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get comprehensive memory snapshot."""
        # System memory
        memory = psutil.virtual_memory()
        system_total_gb = memory.total / (1024**3)
        system_available_gb = memory.available / (1024**3)
        system_used_gb = memory.used / (1024**3)
        system_percent = memory.percent
        
        # Process memory
        process = psutil.Process()
        process_info = process.memory_info()
        process_rss_gb = process_info.rss / (1024**3)
        process_vms_gb = process_info.vms / (1024**3)
        
        # GPU memory
        gpu_total_gb = 0.0
        gpu_available_gb = 0.0
        gpu_used_gb = 0.0
        gpu_percent = 0.0
        gpu_device_name = "N/A"
        
        if self.enable_gpu_monitoring and self.gpu_handle:
            try:
                gpu_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_total_gb = gpu_info.total / (1024**3)
                gpu_used_gb = gpu_info.used / (1024**3)
                gpu_available_gb = (gpu_info.total - gpu_info.used) / (1024**3)
                gpu_percent = (gpu_used_gb / gpu_total_gb) * 100
                gpu_device_name = nvml.nvmlDeviceGetName(self.gpu_handle).decode()
            except Exception as e:
                logger.debug(f"Could not get GPU memory info: {e}")
        
        # Torch memory (if CUDA)
        torch_allocated_gb = 0.0
        torch_cached_gb = 0.0
        torch_reserved_gb = 0.0
        
        if torch.cuda.is_available():
            try:
                torch_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                torch_cached_gb = torch.cuda.memory_cached() / (1024**3)  
                torch_reserved_gb = torch.cuda.memory_reserved() / (1024**3)
            except Exception as e:
                logger.debug(f"Could not get Torch memory info: {e}")
        
        return MemorySnapshot(
            system_total_gb=system_total_gb,
            system_available_gb=system_available_gb,
            system_used_gb=system_used_gb,
            system_percent=system_percent,
            process_rss_gb=process_rss_gb,
            process_vms_gb=process_vms_gb,
            gpu_total_gb=gpu_total_gb,
            gpu_available_gb=gpu_available_gb,
            gpu_used_gb=gpu_used_gb,
            gpu_percent=gpu_percent,
            gpu_device_name=gpu_device_name,
            torch_allocated_gb=torch_allocated_gb,
            torch_cached_gb=torch_cached_gb,
            torch_reserved_gb=torch_reserved_gb,
        )
    
    def predict_oom_risk(self, required_memory_gb: float) -> tuple[bool, str]:
        """
        Predict if loading a model requiring specified memory would cause OOM.
        
        Args:
            required_memory_gb: Estimated memory requirement
            
        Returns:
            Tuple of (high_risk, reason)
        """
        snapshot = self.get_memory_snapshot()
        
        # Check system memory
        available_with_safety = snapshot.system_available_gb - self.safety_margin_gb
        if required_memory_gb > available_with_safety:
            return True, f"Insufficient system memory: {required_memory_gb:.1f}GB needed, {available_with_safety:.1f}GB available"
        
        # Check GPU memory if relevant
        if self.enable_gpu_monitoring and snapshot.gpu_total_gb > 0:
            gpu_available_with_safety = snapshot.gpu_available_gb - (self.safety_margin_gb / 2)
            if required_memory_gb > gpu_available_with_safety:
                return True, f"Insufficient GPU memory: {required_memory_gb:.1f}GB needed, {gpu_available_with_safety:.1f}GB available"
        
        # Check memory pressure
        if snapshot.memory_pressure in ["critical", "high"]:
            return True, f"High memory pressure: {snapshot.system_percent:.1f}% system, {snapshot.gpu_percent:.1f}% GPU"
        
        return False, "Memory availability looks good"
    
    def recommend_quantization(self, model_size_gb: float) -> tuple[int, str]:
        """
        Recommend optimal quantization level based on available memory.
        
        Args:
            model_size_gb: Estimated full-precision model size
            
        Returns:
            Tuple of (quantization_bits, reason)
        """
        snapshot = self.get_memory_snapshot()
        available = snapshot.total_available_gb - self.safety_margin_gb
        
        # Memory requirements by quantization level
        fp16_size = model_size_gb
        int8_size = model_size_gb * 0.5
        int4_size = model_size_gb * 0.25
        
        if available >= fp16_size * 1.2:  # 20% overhead
            return 16, f"Sufficient memory for FP16: {available:.1f}GB available"
        elif available >= int8_size * 1.3:  # 30% overhead
            return 8, f"Recommend INT8 quantization: {available:.1f}GB available"
        elif available >= int4_size * 1.5:  # 50% overhead
            return 4, f"Recommend INT4 quantization: {available:.1f}GB available"
        else:
            return 4, f"Critical memory constraint: {available:.1f}GB available, INT4 required"
    
    def optimize_memory_for_loading(self) -> dict[str, float]:
        """
        Optimize memory usage before model loading.
        
        Returns:
            Dict with memory freed amounts
        """
        logger.info("Optimizing memory for model loading...")
        
        before_snapshot = self.get_memory_snapshot()
        
        # Python garbage collection
        collected = gc.collect()
        
        # Torch cache cleanup
        torch_freed = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch_freed = before_snapshot.torch_cached_gb
        
        # Force another GC pass
        gc.collect()
        
        after_snapshot = self.get_memory_snapshot()
        
        system_freed = before_snapshot.system_used_gb - after_snapshot.system_used_gb
        gpu_freed = before_snapshot.gpu_used_gb - after_snapshot.gpu_used_gb
        
        optimization_result = {
            "python_objects_collected": collected,
            "system_memory_freed_gb": max(0, system_freed),
            "gpu_memory_freed_gb": max(0, gpu_freed),
            "torch_cache_freed_gb": torch_freed,
        }
        
        logger.info(f"Memory optimization complete: {optimization_result}")
        return optimization_result
    
    def start_monitoring(self):
        """Start continuous memory monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self.is_monitoring:
            try:
                snapshot = self.get_memory_snapshot()
                self.snapshots.append(snapshot)
                
                # Keep only recent snapshots
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots = self.snapshots[-self.max_snapshots:]
                
                # Check for warnings
                if snapshot.memory_pressure == "critical":
                    if self.on_oom_prediction:
                        self.on_oom_prediction(snapshot)
                elif snapshot.memory_pressure in ["high", "critical"]:
                    if self.on_memory_warning:
                        self.on_memory_warning(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_memory_history(self, minutes: int = 5) -> list[MemorySnapshot]:
        """Get memory usage history for specified time period."""
        cutoff_time = time.time() - (minutes * 60)
        return [s for s in self.snapshots if s.timestamp >= cutoff_time]
    
    def log_memory_status(self, level: str = "info"):
        """Log detailed memory status."""
        snapshot = self.get_memory_snapshot()
        
        status_msg = f"""
Memory Status Report:
┌─ System Memory ─────────────────────────────────────┐
│ Total: {snapshot.system_total_gb:.1f}GB                             │
│ Available: {snapshot.system_available_gb:.1f}GB                     │ 
│ Used: {snapshot.system_used_gb:.1f}GB ({snapshot.system_percent:.1f}%)    │
│ Process: {snapshot.process_rss_gb:.1f}GB RSS, {snapshot.process_vms_gb:.1f}GB VMS    │
├─ GPU Memory ({snapshot.gpu_device_name[:20]:20s}) ─────┤
│ Total: {snapshot.gpu_total_gb:.1f}GB                               │
│ Available: {snapshot.gpu_available_gb:.1f}GB                       │
│ Used: {snapshot.gpu_used_gb:.1f}GB ({snapshot.gpu_percent:.1f}%)          │
│ Torch: {snapshot.torch_allocated_gb:.1f}GB alloc, {snapshot.torch_cached_gb:.1f}GB cache │
├─ Assessment ────────────────────────────────────────┤
│ Pressure: {snapshot.memory_pressure:8s}                     │
│ Total Available: {snapshot.total_available_gb:.1f}GB                │
│ Safety Margin: {self.safety_margin_gb:.1f}GB                     │
└─────────────────────────────────────────────────────┘
"""
        
        if level == "info":
            logger.info(status_msg)
        elif level == "warning":
            logger.warning(status_msg)
        elif level == "error":
            logger.error(status_msg)
        else:
            logger.debug(status_msg)
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


def create_memory_monitor(model_size_estimate_gb: float = 7.0) -> AdvancedMemoryMonitor:
    """
    Create memory monitor optimized for model loading.
    
    Args:
        model_size_estimate_gb: Estimated model size for safety calculations
        
    Returns:
        Configured AdvancedMemoryMonitor instance
    """
    # Adjust safety margin based on model size
    safety_margin = max(2.0, model_size_estimate_gb * 0.2)
    
    monitor = AdvancedMemoryMonitor(
        safety_margin_gb=safety_margin,
        monitoring_interval=0.5,  # More frequent during loading
        enable_gpu_monitoring=True
    )
    
    # Set up warning callbacks
    def memory_warning_handler(snapshot: MemorySnapshot):
        logger.warning(f"Memory pressure detected: {snapshot.memory_pressure} - "
                      f"System: {snapshot.system_percent:.1f}%, GPU: {snapshot.gpu_percent:.1f}%")
    
    def oom_prediction_handler(snapshot: MemorySnapshot):
        logger.error(f"OOM risk detected! "
                    f"System: {snapshot.system_available_gb:.1f}GB available, "
                    f"GPU: {snapshot.gpu_available_gb:.1f}GB available")
    
    monitor.on_memory_warning = memory_warning_handler
    monitor.on_oom_prediction = oom_prediction_handler
    
    return monitor