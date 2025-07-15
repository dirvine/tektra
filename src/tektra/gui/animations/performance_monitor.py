"""
UI Performance Monitor

This module monitors animation performance and provides optimization recommendations.
Implements comprehensive performance tracking, automatic quality adjustment, and
system resource monitoring for optimal UI experience.
"""

import asyncio
import time
import psutil
import os
import threading
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from loguru import logger

from .animation_config import PerformanceMetrics, AnimationConfig


@dataclass
class SystemPerformanceMetrics:
    """Extended system performance metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_available: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_io_sent: float = 0.0
    network_io_recv: float = 0.0
    gpu_usage: float = 0.0  # If available
    temperature: float = 0.0  # If available
    battery_level: float = 100.0  # If on battery
    power_plugged: bool = True


@dataclass
class AnimationPerformanceMetrics:
    """Animation-specific performance metrics."""
    total_animations_started: int = 0
    total_animations_completed: int = 0
    total_animations_cancelled: int = 0
    average_animation_duration: float = 0.0
    longest_animation_duration: float = 0.0
    shortest_animation_duration: float = float('inf')
    animations_per_second: float = 0.0
    memory_per_animation: float = 0.0
    animation_types_count: Dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for optimization decisions."""
    target_fps: float = 60.0
    minimum_fps: float = 30.0
    critical_fps: float = 15.0
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 85.0
    max_dropped_frame_percentage: float = 20.0
    optimization_check_interval: float = 2.0
    performance_history_length: int = 300


class UIPerformanceMonitor:
    """
    Comprehensive UI performance monitor that tracks animation frame rates,
    system resources, and automatically adjusts animation quality.
    """
    
    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        """Initialize the performance monitor."""
        self.thresholds = thresholds or PerformanceThresholds()
        self.metrics = PerformanceMetrics()
        self.system_metrics = SystemPerformanceMetrics()
        self.animation_metrics = AnimationPerformanceMetrics()
        
        # Performance monitoring state
        self.monitoring_enabled = True
        self.auto_optimization_enabled = True
        self.optimization_callbacks: List[Callable] = []
        self.last_optimization_check = time.time()
        
        # Performance history for trend analysis
        self.performance_history: List[Dict[str, float]] = []
        self.frame_time_history: List[float] = []
        self.cpu_history: List[float] = []
        self.memory_history: List[float] = []
        
        # Animation tracking
        self.active_animations: Dict[str, Dict[str, Any]] = {}
        self.animation_start_times: Dict[str, float] = {}
        
        # Performance state
        self.current_performance_level = "high"  # high, medium, low, critical
        self.performance_degradation_count = 0
        self.last_performance_level = "high"
        
        # Threading for background monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        logger.info("Enhanced UI Performance Monitor initialized")
    
    def record_frame_time(self, frame_time: float) -> None:
        """Record a frame time measurement."""
        if not self.monitoring_enabled:
            return
            
        self.metrics.add_frame_time(frame_time)
        self.frame_time_history.append(frame_time)
        
        # Maintain history length
        if len(self.frame_time_history) > self.thresholds.performance_history_length:
            self.frame_time_history.pop(0)
        
        # Check if we need to optimize
        current_time = time.time()
        if current_time - self.last_optimization_check > self.thresholds.optimization_check_interval:
            self._check_performance_optimization()
            self.last_optimization_check = current_time
    
    def record_animation_start(self, animation_id: str, animation_type: str = "unknown") -> None:
        """Record that an animation has started."""
        current_time = time.time()
        self.metrics.animation_count += 1
        self.animation_metrics.total_animations_started += 1
        
        # Track animation details
        self.active_animations[animation_id] = {
            "type": animation_type,
            "start_time": current_time,
            "memory_at_start": self._get_current_memory_usage()
        }
        self.animation_start_times[animation_id] = current_time
        
        # Update animation type counts
        if animation_type in self.animation_metrics.animation_types_count:
            self.animation_metrics.animation_types_count[animation_type] += 1
        else:
            self.animation_metrics.animation_types_count[animation_type] = 1
    
    def record_animation_end(self, animation_id: str, completed: bool = True) -> None:
        """Record that an animation has ended."""
        if self.metrics.animation_count > 0:
            self.metrics.animation_count -= 1
        
        if completed:
            self.animation_metrics.total_animations_completed += 1
        else:
            self.animation_metrics.total_animations_cancelled += 1
        
        # Calculate animation duration if we have the start time
        if animation_id in self.animation_start_times:
            duration = time.time() - self.animation_start_times[animation_id]
            self._update_animation_duration_stats(duration)
            del self.animation_start_times[animation_id]
        
        # Clean up tracking
        if animation_id in self.active_animations:
            del self.active_animations[animation_id]
    
    def should_reduce_animations(self) -> bool:
        """Check if animations should be reduced based on performance."""
        if not self.monitoring_enabled:
            return False
        
        return self._determine_performance_level() in ["low", "critical"]
    
    def _determine_performance_level(self) -> str:
        """Determine current performance level based on multiple metrics."""
        score = 100  # Start with perfect score
        
        # FPS scoring
        current_fps = self.metrics.fps
        if current_fps > 0:
            if current_fps < self.thresholds.critical_fps:
                score -= 40
            elif current_fps < self.thresholds.minimum_fps:
                score -= 25
            elif current_fps < self.thresholds.target_fps:
                score -= 10
        
        # Dropped frames scoring
        total_frames = len(self.metrics.frame_times)
        if total_frames > 10:
            dropped_percentage = (self.metrics.dropped_frames / total_frames) * 100
            if dropped_percentage > self.thresholds.max_dropped_frame_percentage:
                score -= 20
            elif dropped_percentage > 10:
                score -= 10
        
        # System resource scoring
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.thresholds.max_cpu_usage:
                score -= 15
            elif cpu_percent > 60:
                score -= 8
            
            if memory_percent > self.thresholds.max_memory_usage:
                score -= 15
            elif memory_percent > 70:
                score -= 8
                
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
            score -= 5  # Penalty for monitoring failure
        
        # Animation load scoring
        if self.metrics.animation_count > 10:
            score -= 10
        elif self.metrics.animation_count > 5:
            score -= 5
        
        # Determine level based on score
        if score >= 80:
            return "high"
        elif score >= 60:
            return "medium"
        elif score >= 40:
            return "low"
        else:
            return "critical"
    
    def get_performance_recommendations(self) -> List[str]:
        """Get comprehensive performance optimization recommendations."""
        recommendations = []
        performance_level = self._determine_performance_level()
        
        # FPS-based recommendations
        current_fps = self.metrics.fps
        if current_fps > 0:
            if current_fps < self.thresholds.critical_fps:
                recommendations.append("Critical performance: Disable all non-essential animations")
                recommendations.append("Consider switching to static UI mode")
            elif current_fps < self.thresholds.minimum_fps:
                recommendations.append("Enable reduced motion mode")
                recommendations.append("Reduce animation complexity and duration")
            elif current_fps < self.thresholds.target_fps:
                recommendations.append("Consider optimizing animation performance")
        
        # Dropped frames analysis
        total_frames = len(self.metrics.frame_times)
        if total_frames > 10:
            dropped_percentage = (self.metrics.dropped_frames / total_frames) * 100
            if dropped_percentage > self.thresholds.max_dropped_frame_percentage:
                recommendations.append(f"High dropped frame rate: {dropped_percentage:.1f}%")
                recommendations.append("Reduce concurrent animations")
            elif dropped_percentage > 10:
                recommendations.append(f"Moderate frame drops: {dropped_percentage:.1f}%")
        
        # System resource recommendations
        try:
            self._update_system_metrics()
            
            if self.system_metrics.cpu_usage > self.thresholds.max_cpu_usage:
                recommendations.append(f"High CPU usage: {self.system_metrics.cpu_usage:.1f}%")
                recommendations.append("Reduce animation complexity or frequency")
            
            if self.system_metrics.memory_usage > self.thresholds.max_memory_usage:
                recommendations.append(f"High memory usage: {self.system_metrics.memory_usage:.1f}%")
                recommendations.append("Enable memory optimization mode")
            
            # Battery-specific recommendations
            if not self.system_metrics.power_plugged and self.system_metrics.battery_level < 20:
                recommendations.append("Low battery: Enable power saving mode")
                recommendations.append("Reduce animation frequency to preserve battery")
                
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
        
        # Animation load recommendations
        if self.metrics.animation_count > 10:
            recommendations.append(f"High animation load: {self.metrics.animation_count} active")
            recommendations.append("Limit concurrent animations")
        
        # Performance trend recommendations
        if len(self.performance_history) > 30:
            recent_avg = sum(self.performance_history[-30:]) / 30
            older_avg = sum(self.performance_history[-60:-30]) / 30 if len(self.performance_history) > 60 else recent_avg
            
            if recent_avg < older_avg * 0.8:
                recommendations.append("Performance degrading over time")
                recommendations.append("Consider restarting animations or clearing cache")
        
        return recommendations
    
    def get_optimized_config(self, base_config: AnimationConfig) -> AnimationConfig:
        """Get an optimized animation configuration based on current performance."""
        performance_level = self._determine_performance_level()
        
        if performance_level == "high":
            return base_config
        elif performance_level == "medium":
            # Slightly reduce animation complexity
            return AnimationConfig(
                duration=base_config.duration * 0.8,
                easing=base_config.easing,
                delay=base_config.delay * 0.5,
                reduced_motion=base_config.reduced_motion,
                performance_mode="balanced"
            )
        elif performance_level == "low":
            # Significantly reduce animation complexity
            return AnimationConfig(
                duration=min(base_config.duration * 0.5, 0.2),
                easing=base_config.easing,
                delay=0.0,
                reduced_motion=True,
                performance_mode="performance"
            )
        else:  # critical
            # Minimal animations only
            return AnimationConfig(
                duration=min(base_config.duration * 0.25, 0.1),
                easing=base_config.easing,
                delay=0.0,
                reduced_motion=True,
                performance_mode="performance"
            )
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """Add a callback to be called when performance optimization is needed."""
        self.optimization_callbacks.append(callback)
    
    def remove_optimization_callback(self, callback: Callable) -> None:
        """Remove an optimization callback."""
        if callback in self.optimization_callbacks:
            self.optimization_callbacks.remove(callback)
    
    def _check_performance_optimization(self) -> None:
        """Check if performance optimization is needed and notify callbacks."""
        if self.should_reduce_animations():
            logger.info("Performance degradation detected, triggering optimization callbacks")
            for callback in self.optimization_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in optimization callback: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get a summary of current performance metrics."""
        return {
            "fps": self.metrics.fps,
            "average_frame_time": self.metrics.average_frame_time,
            "dropped_frames": self.metrics.dropped_frames,
            "active_animations": self.metrics.animation_count,
            "should_reduce_animations": self.should_reduce_animations(),
            "recommendations": self.get_performance_recommendations()
        }
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics = PerformanceMetrics()
        logger.info("Performance metrics reset")
    
    def enable_monitoring(self, enabled: bool = True) -> None:
        """Enable or disable performance monitoring."""
        self.monitoring_enabled = enabled
        logger.info(f"Performance monitoring {'enabled' if enabled else 'disabled'}")
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return 0.0
    
    def _update_animation_duration_stats(self, duration: float) -> None:
        """Update animation duration statistics."""
        # Update average duration
        total_completed = self.animation_metrics.total_animations_completed
        if total_completed > 0:
            current_avg = self.animation_metrics.average_animation_duration
            self.animation_metrics.average_animation_duration = (
                (current_avg * (total_completed - 1) + duration) / total_completed
            )
        else:
            self.animation_metrics.average_animation_duration = duration
        
        # Update longest and shortest durations
        if duration > self.animation_metrics.longest_animation_duration:
            self.animation_metrics.longest_animation_duration = duration
        
        if duration < self.animation_metrics.shortest_animation_duration:
            self.animation_metrics.shortest_animation_duration = duration
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics."""
        try:
            # CPU and memory
            self.system_metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.system_metrics.memory_usage = memory.percent
            self.system_metrics.memory_available = memory.available / 1024 / 1024  # MB
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                self.system_metrics.disk_io_read = disk_io.read_bytes / 1024 / 1024  # MB
                self.system_metrics.disk_io_write = disk_io.write_bytes / 1024 / 1024  # MB
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                self.system_metrics.network_io_sent = net_io.bytes_sent / 1024 / 1024  # MB
                self.system_metrics.network_io_recv = net_io.bytes_recv / 1024 / 1024  # MB
            
            # Battery status
            battery = psutil.sensors_battery()
            if battery:
                self.system_metrics.battery_level = battery.percent
                self.system_metrics.power_plugged = battery.power_plugged
            
            # Update history
            self.cpu_history.append(self.system_metrics.cpu_usage)
            self.memory_history.append(self.system_metrics.memory_usage)
            
            # Maintain history length
            if len(self.cpu_history) > self.thresholds.performance_history_length:
                self.cpu_history.pop(0)
            if len(self.memory_history) > self.thresholds.performance_history_length:
                self.memory_history.pop(0)
                
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")
    
    def get_detailed_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary with all metrics."""
        self._update_system_metrics()
        
        return {
            "performance_level": self._determine_performance_level(),
            "fps_metrics": {
                "current_fps": self.metrics.fps,
                "average_frame_time": self.metrics.average_frame_time,
                "dropped_frames": self.metrics.dropped_frames,
                "total_frames": len(self.metrics.frame_times)
            },
            "animation_metrics": {
                "active_animations": self.metrics.animation_count,
                "total_started": self.animation_metrics.total_animations_started,
                "total_completed": self.animation_metrics.total_animations_completed,
                "total_cancelled": self.animation_metrics.total_animations_cancelled,
                "average_duration": self.animation_metrics.average_animation_duration,
                "longest_duration": self.animation_metrics.longest_animation_duration,
                "shortest_duration": self.animation_metrics.shortest_animation_duration,
                "animation_types": dict(self.animation_metrics.animation_types_count)
            },
            "system_metrics": {
                "cpu_usage": self.system_metrics.cpu_usage,
                "memory_usage": self.system_metrics.memory_usage,
                "memory_available": self.system_metrics.memory_available,
                "battery_level": self.system_metrics.battery_level,
                "power_plugged": self.system_metrics.power_plugged
            },
            "optimization": {
                "should_reduce_animations": self.should_reduce_animations(),
                "auto_optimization_enabled": self.auto_optimization_enabled,
                "recommendations": self.get_performance_recommendations()
            }
        }
    
    def enable_auto_optimization(self, enabled: bool = True) -> None:
        """Enable or disable automatic performance optimization."""
        self.auto_optimization_enabled = enabled
        logger.info(f"Auto optimization {'enabled' if enabled else 'disabled'}")
    
    def start_background_monitoring(self) -> None:
        """Start background monitoring in a separate thread."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Background monitoring already running")
            return
        
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Background monitoring started")
    
    def stop_background_monitoring(self) -> None:
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Background monitoring stopped")
    
    def _background_monitoring_loop(self) -> None:
        """Background monitoring loop that runs in a separate thread."""
        while not self._stop_monitoring.is_set():
            try:
                self._update_system_metrics()
                
                # Update performance level
                new_level = self._determine_performance_level()
                if new_level != self.current_performance_level:
                    logger.info(f"Performance level changed: {self.current_performance_level} -> {new_level}")
                    self.last_performance_level = self.current_performance_level
                    self.current_performance_level = new_level
                    
                    # Trigger optimization if auto-optimization is enabled
                    if self.auto_optimization_enabled and new_level in ["low", "critical"]:
                        self._check_performance_optimization()
                
                # Add to performance history
                performance_snapshot = {
                    "fps": self.metrics.fps,
                    "cpu": self.system_metrics.cpu_usage,
                    "memory": self.system_metrics.memory_usage,
                    "animations": self.metrics.animation_count,
                    "timestamp": time.time()
                }
                self.performance_history.append(performance_snapshot)
                
                # Maintain history length
                if len(self.performance_history) > self.thresholds.performance_history_length:
                    self.performance_history.pop(0)
                
                # Sleep for the monitoring interval
                self._stop_monitoring.wait(self.thresholds.optimization_check_interval)
                
            except Exception as e:
                logger.error(f"Error in background monitoring loop: {e}")
                self._stop_monitoring.wait(5.0)  # Wait longer on error
    
    async def start_monitoring_loop(self) -> None:
        """Start the performance monitoring loop (async version)."""
        logger.info("Starting async performance monitoring loop")
        
        while self.monitoring_enabled:
            try:
                self._update_system_metrics()
                
                # Update performance level
                new_level = self._determine_performance_level()
                if new_level != self.current_performance_level:
                    logger.info(f"Performance level changed: {self.current_performance_level} -> {new_level}")
                    self.last_performance_level = self.current_performance_level
                    self.current_performance_level = new_level
                    
                    # Trigger optimization if auto-optimization is enabled
                    if self.auto_optimization_enabled and new_level in ["low", "critical"]:
                        self._check_performance_optimization()
                
                # Add to performance history
                performance_snapshot = {
                    "fps": self.metrics.fps,
                    "cpu": self.system_metrics.cpu_usage,
                    "memory": self.system_metrics.memory_usage,
                    "animations": self.metrics.animation_count,
                    "timestamp": time.time()
                }
                self.performance_history.append(performance_snapshot)
                
                # Maintain history length
                if len(self.performance_history) > self.thresholds.performance_history_length:
                    self.performance_history.pop(0)
                
                await asyncio.sleep(self.thresholds.optimization_check_interval)
                
            except Exception as e:
                logger.error(f"Error in async performance monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error