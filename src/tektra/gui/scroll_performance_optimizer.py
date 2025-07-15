"""
Scroll Performance Optimizer - Maintains 60fps During Rapid Updates

This module provides performance optimization for scrolling operations,
ensuring smooth 60fps performance even during rapid message updates.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for scroll operations."""
    frame_time: float
    fps: float
    memory_usage: float
    cpu_usage: float
    scroll_velocity: float
    rendered_items: int
    timestamp: float


class ScrollPerformanceOptimizer:
    """
    Optimizes scroll performance to maintain 60fps during rapid updates.
    
    Monitors performance metrics and automatically adjusts rendering quality,
    animation complexity, and virtual scrolling parameters to maintain smooth performance.
    """
    
    def __init__(
        self,
        target_fps: float = 60.0,
        performance_window: int = 30,
        optimization_threshold: float = 45.0
    ):
        """
        Initialize the scroll performance optimizer.
        
        Args:
            target_fps: Target frames per second
            performance_window: Number of frames to average for performance calculation
            optimization_threshold: FPS threshold below which optimizations are triggered
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.performance_window = performance_window
        self.optimization_threshold = optimization_threshold
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_fps = target_fps
        self.average_frame_time = self.target_frame_time
        
        # Optimization state
        self.optimization_level = 0  # 0 = none, 1 = light, 2 = moderate, 3 = aggressive
        self.optimizations_active: Dict[str, bool] = {
            "reduced_animations": False,
            "virtual_scrolling": False,
            "frame_skipping": False,
            "quality_reduction": False,
            "batch_rendering": False
        }
        
        # Callbacks for optimization actions
        self.optimization_callbacks: Dict[str, List[Callable]] = {
            "reduce_animations": [],
            "enable_virtual_scrolling": [],
            "skip_frames": [],
            "reduce_quality": [],
            "batch_render": []
        }
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.last_optimization_time = 0
        self.optimization_cooldown = 2.0  # Seconds between optimizations
        
        logger.info(f"ScrollPerformanceOptimizer initialized with target {target_fps} fps")
    
    def record_frame_metrics(
        self,
        frame_time: float,
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None,
        scroll_velocity: Optional[float] = None,
        rendered_items: Optional[int] = None
    ) -> None:
        """
        Record performance metrics for a frame.
        
        Args:
            frame_time: Time taken to render the frame in seconds
            memory_usage: Current memory usage in MB
            cpu_usage: Current CPU usage percentage
            scroll_velocity: Current scroll velocity
            rendered_items: Number of items currently rendered
        """
        if not self.monitoring_enabled:
            return
        
        fps = 1.0 / frame_time if frame_time > 0 else self.target_fps
        
        metrics = PerformanceMetrics(
            frame_time=frame_time,
            fps=fps,
            memory_usage=memory_usage or 0.0,
            cpu_usage=cpu_usage or 0.0,
            scroll_velocity=scroll_velocity or 0.0,
            rendered_items=rendered_items or 0,
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > self.performance_window * 2:
            self.metrics_history = self.metrics_history[-self.performance_window:]
        
        # Update current performance calculations
        self._update_performance_calculations()
        
        # Check if optimization is needed
        self._check_optimization_needed()
    
    def add_optimization_callback(self, optimization_type: str, callback: Callable) -> None:
        """
        Add a callback for a specific optimization type.
        
        Args:
            optimization_type: Type of optimization
            callback: Callback function to execute
        """
        if optimization_type in self.optimization_callbacks:
            self.optimization_callbacks[optimization_type].append(callback)
            logger.debug(f"Added optimization callback for {optimization_type}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get current performance summary.
        
        Returns:
            Dictionary with performance information
        """
        recent_metrics = self.metrics_history[-self.performance_window:] if self.metrics_history else []
        
        if not recent_metrics:
            return {
                "current_fps": self.target_fps,
                "average_frame_time": self.target_frame_time,
                "optimization_level": self.optimization_level,
                "optimizations_active": self.optimizations_active.copy(),
                "performance_status": "unknown"
            }
        
        avg_fps = sum(m.fps for m in recent_metrics) / len(recent_metrics)
        avg_frame_time = sum(m.frame_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        # Determine performance status
        if avg_fps >= self.target_fps * 0.95:
            status = "excellent"
        elif avg_fps >= self.optimization_threshold:
            status = "good"
        elif avg_fps >= self.optimization_threshold * 0.8:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "current_fps": avg_fps,
            "target_fps": self.target_fps,
            "average_frame_time": avg_frame_time * 1000,  # Convert to ms
            "target_frame_time": self.target_frame_time * 1000,
            "average_memory_usage": avg_memory,
            "optimization_level": self.optimization_level,
            "optimizations_active": self.optimizations_active.copy(),
            "performance_status": status,
            "metrics_count": len(recent_metrics)
        }
    
    def force_optimization_level(self, level: int) -> None:
        """
        Force a specific optimization level.
        
        Args:
            level: Optimization level (0-3)
        """
        level = max(0, min(3, level))
        old_level = self.optimization_level
        self.optimization_level = level
        
        if level != old_level:
            self._apply_optimization_level()
            logger.info(f"Forced optimization level from {old_level} to {level}")
    
    def reset_optimizations(self) -> None:
        """Reset all optimizations to default state."""
        self.optimization_level = 0
        self.optimizations_active = {key: False for key in self.optimizations_active}
        self._apply_optimization_level()
        logger.info("Reset all performance optimizations")
    
    def enable_monitoring(self, enabled: bool) -> None:
        """
        Enable or disable performance monitoring.
        
        Args:
            enabled: Whether to enable monitoring
        """
        self.monitoring_enabled = enabled
        logger.debug(f"Performance monitoring {'enabled' if enabled else 'disabled'}")
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Get recommendations for improving performance.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if self.current_fps < self.optimization_threshold:
            recommendations.append("Consider enabling virtual scrolling for large conversations")
            recommendations.append("Reduce animation complexity during scrolling")
            
        if self.current_fps < self.optimization_threshold * 0.8:
            recommendations.append("Enable frame skipping for very poor performance")
            recommendations.append("Reduce rendering quality temporarily")
            
        if self.current_fps < self.optimization_threshold * 0.6:
            recommendations.append("Consider aggressive optimizations")
            recommendations.append("Limit concurrent animations")
            
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else []
        if recent_metrics:
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            if avg_memory > 500:  # MB
                recommendations.append("High memory usage detected - consider virtual scrolling")
        
        return recommendations
    
    def _update_performance_calculations(self) -> None:
        """Update current performance calculations."""
        if not self.metrics_history:
            return
        
        # Calculate current FPS from recent frames
        recent_metrics = self.metrics_history[-min(10, len(self.metrics_history)):]
        self.current_fps = sum(m.fps for m in recent_metrics) / len(recent_metrics)
        self.average_frame_time = sum(m.frame_time for m in recent_metrics) / len(recent_metrics)
    
    def _check_optimization_needed(self) -> None:
        """Check if performance optimization is needed."""
        current_time = time.time()
        
        # Don't optimize too frequently
        if current_time - self.last_optimization_time < self.optimization_cooldown:
            return
        
        # Need at least some metrics to make decisions
        if len(self.metrics_history) < 5:
            return
        
        # Check if performance is below threshold
        if self.current_fps < self.optimization_threshold:
            self._increase_optimization_level()
            self.last_optimization_time = current_time
        elif self.current_fps > self.target_fps * 0.95 and self.optimization_level > 0:
            # Performance is good, we can reduce optimizations
            self._decrease_optimization_level()
            self.last_optimization_time = current_time
    
    def _increase_optimization_level(self) -> None:
        """Increase the optimization level."""
        if self.optimization_level < 3:
            self.optimization_level += 1
            self._apply_optimization_level()
            logger.info(f"Increased optimization level to {self.optimization_level} "
                       f"(FPS: {self.current_fps:.1f})")
    
    def _decrease_optimization_level(self) -> None:
        """Decrease the optimization level."""
        if self.optimization_level > 0:
            self.optimization_level -= 1
            self._apply_optimization_level()
            logger.info(f"Decreased optimization level to {self.optimization_level} "
                       f"(FPS: {self.current_fps:.1f})")
    
    def _apply_optimization_level(self) -> None:
        """Apply the current optimization level."""
        # Reset all optimizations
        for key in self.optimizations_active:
            self.optimizations_active[key] = False
        
        # Apply optimizations based on level
        if self.optimization_level >= 1:
            # Light optimizations
            self.optimizations_active["reduced_animations"] = True
            self._trigger_callbacks("reduce_animations")
        
        if self.optimization_level >= 2:
            # Moderate optimizations
            self.optimizations_active["virtual_scrolling"] = True
            self.optimizations_active["batch_rendering"] = True
            self._trigger_callbacks("enable_virtual_scrolling")
            self._trigger_callbacks("batch_render")
        
        if self.optimization_level >= 3:
            # Aggressive optimizations
            self.optimizations_active["frame_skipping"] = True
            self.optimizations_active["quality_reduction"] = True
            self._trigger_callbacks("skip_frames")
            self._trigger_callbacks("reduce_quality")
    
    def _trigger_callbacks(self, optimization_type: str) -> None:
        """
        Trigger callbacks for a specific optimization type.
        
        Args:
            optimization_type: Type of optimization
        """
        if optimization_type in self.optimization_callbacks:
            for callback in self.optimization_callbacks[optimization_type]:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in optimization callback {optimization_type}: {e}")


class ScrollFrameRateMonitor:
    """
    Monitors frame rate specifically for scroll operations.
    
    Provides detailed frame rate analysis and automatic quality adjustment
    to maintain smooth scrolling performance.
    """
    
    def __init__(self, target_fps: float = 60.0):
        """
        Initialize the frame rate monitor.
        
        Args:
            target_fps: Target frames per second
        """
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        
        # Frame timing
        self.frame_times: List[float] = []
        self.last_frame_time = time.time()
        self.dropped_frames = 0
        self.total_frames = 0
        
        # Quality adjustment
        self.current_quality = 1.0  # 1.0 = full quality, 0.5 = half quality
        self.quality_adjustment_enabled = True
        
        # Performance thresholds
        self.good_fps_threshold = target_fps * 0.95
        self.poor_fps_threshold = target_fps * 0.75
        self.critical_fps_threshold = target_fps * 0.5
        
        logger.info(f"ScrollFrameRateMonitor initialized with target {target_fps} fps")
    
    def record_frame(self) -> Dict[str, float]:
        """
        Record a frame and return timing information.
        
        Returns:
            Dictionary with frame timing information
        """
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(frame_time)
        self.total_frames += 1
        
        # Keep only recent frame times
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Check for dropped frames
        if frame_time > self.target_frame_time * 1.5:
            self.dropped_frames += 1
        
        # Calculate current FPS
        current_fps = 1.0 / frame_time if frame_time > 0 else self.target_fps
        
        # Adjust quality if needed
        if self.quality_adjustment_enabled:
            self._adjust_quality_based_on_fps(current_fps)
        
        return {
            "frame_time": frame_time * 1000,  # Convert to ms
            "fps": current_fps,
            "quality": self.current_quality,
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames
        }
    
    def get_average_fps(self, window_size: int = 30) -> float:
        """
        Get average FPS over a window of recent frames.
        
        Args:
            window_size: Number of recent frames to average
            
        Returns:
            Average FPS
        """
        if not self.frame_times:
            return self.target_fps
        
        recent_times = self.frame_times[-window_size:]
        if not recent_times:
            return self.target_fps
        
        avg_frame_time = sum(recent_times) / len(recent_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else self.target_fps
    
    def get_performance_grade(self) -> str:
        """
        Get a performance grade based on current FPS.
        
        Returns:
            Performance grade ("A", "B", "C", "D", "F")
        """
        avg_fps = self.get_average_fps()
        
        if avg_fps >= self.good_fps_threshold:
            return "A"
        elif avg_fps >= self.poor_fps_threshold:
            return "B"
        elif avg_fps >= self.critical_fps_threshold:
            return "C"
        elif avg_fps >= self.critical_fps_threshold * 0.5:
            return "D"
        else:
            return "F"
    
    def reset_statistics(self) -> None:
        """Reset all performance statistics."""
        self.frame_times.clear()
        self.dropped_frames = 0
        self.total_frames = 0
        self.current_quality = 1.0
        logger.debug("Frame rate statistics reset")
    
    def set_quality_adjustment_enabled(self, enabled: bool) -> None:
        """
        Enable or disable automatic quality adjustment.
        
        Args:
            enabled: Whether to enable quality adjustment
        """
        self.quality_adjustment_enabled = enabled
        if not enabled:
            self.current_quality = 1.0
        logger.debug(f"Quality adjustment {'enabled' if enabled else 'disabled'}")
    
    def _adjust_quality_based_on_fps(self, current_fps: float) -> None:
        """
        Adjust rendering quality based on current FPS.
        
        Args:
            current_fps: Current frames per second
        """
        if current_fps >= self.good_fps_threshold:
            # Performance is good, increase quality
            self.current_quality = min(1.0, self.current_quality + 0.05)
        elif current_fps < self.poor_fps_threshold:
            # Performance is poor, decrease quality
            self.current_quality = max(0.3, self.current_quality - 0.1)
        elif current_fps < self.critical_fps_threshold:
            # Performance is critical, aggressively decrease quality
            self.current_quality = max(0.1, self.current_quality - 0.2)
        
        # Ensure quality stays within bounds
        self.current_quality = max(0.1, min(1.0, self.current_quality))