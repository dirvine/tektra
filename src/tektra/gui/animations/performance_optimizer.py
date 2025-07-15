"""
Performance Optimizer for UI Animations

This module provides automatic performance optimization and fallback systems
for lower-end systems or when performance degrades.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from loguru import logger

import toga

from .animation_config import AnimationConfig, AnimationType, EasingFunction
from .performance_monitor import UIPerformanceMonitor, PerformanceThresholds


@dataclass
class OptimizationProfile:
    """Performance optimization profile for different system capabilities."""
    name: str
    description: str
    max_concurrent_animations: int
    animation_quality_multiplier: float  # 0.0 to 1.0
    enable_complex_easing: bool
    enable_staggered_animations: bool
    enable_micro_interactions: bool
    frame_rate_target: float
    memory_limit_mb: float


class PerformanceOptimizer:
    """
    Automatic performance optimizer that adjusts animation quality and
    provides fallbacks for lower-end systems.
    """
    
    def __init__(self, performance_monitor: UIPerformanceMonitor):
        """Initialize the performance optimizer."""
        self.performance_monitor = performance_monitor
        self.current_profile = self._get_default_profile()
        self.optimization_profiles = self._create_optimization_profiles()
        
        # Optimization state
        self.optimization_active = False
        self.fallback_mode_active = False
        self.animation_queue: List[Dict[str, Any]] = []
        self.max_queue_size = 50
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.last_optimization_time = 0.0
        self.optimization_cooldown = 5.0  # Seconds between optimizations
        
        # Callbacks for optimization events
        self.optimization_callbacks: List[Callable[[str, OptimizationProfile], None]] = []
        
        logger.info("Performance Optimizer initialized")
    
    def _create_optimization_profiles(self) -> Dict[str, OptimizationProfile]:
        """Create predefined optimization profiles."""
        return {
            "ultra": OptimizationProfile(
                name="ultra",
                description="Ultra-high performance with all features enabled",
                max_concurrent_animations=20,
                animation_quality_multiplier=1.0,
                enable_complex_easing=True,
                enable_staggered_animations=True,
                enable_micro_interactions=True,
                frame_rate_target=60.0,
                memory_limit_mb=200.0
            ),
            "high": OptimizationProfile(
                name="high",
                description="High performance with most features enabled",
                max_concurrent_animations=15,
                animation_quality_multiplier=0.9,
                enable_complex_easing=True,
                enable_staggered_animations=True,
                enable_micro_interactions=True,
                frame_rate_target=60.0,
                memory_limit_mb=150.0
            ),
            "balanced": OptimizationProfile(
                name="balanced",
                description="Balanced performance and quality",
                max_concurrent_animations=10,
                animation_quality_multiplier=0.8,
                enable_complex_easing=True,
                enable_staggered_animations=False,
                enable_micro_interactions=True,
                frame_rate_target=45.0,
                memory_limit_mb=100.0
            ),
            "performance": OptimizationProfile(
                name="performance",
                description="Performance-focused with reduced quality",
                max_concurrent_animations=6,
                animation_quality_multiplier=0.6,
                enable_complex_easing=False,
                enable_staggered_animations=False,
                enable_micro_interactions=False,
                frame_rate_target=30.0,
                memory_limit_mb=75.0
            ),
            "minimal": OptimizationProfile(
                name="minimal",
                description="Minimal animations for low-end systems",
                max_concurrent_animations=3,
                animation_quality_multiplier=0.4,
                enable_complex_easing=False,
                enable_staggered_animations=False,
                enable_micro_interactions=False,
                frame_rate_target=30.0,
                memory_limit_mb=50.0
            ),
            "fallback": OptimizationProfile(
                name="fallback",
                description="Emergency fallback mode with minimal animations",
                max_concurrent_animations=1,
                animation_quality_multiplier=0.2,
                enable_complex_easing=False,
                enable_staggered_animations=False,
                enable_micro_interactions=False,
                frame_rate_target=15.0,
                memory_limit_mb=25.0
            )
        }
    
    def _get_default_profile(self) -> OptimizationProfile:
        """Get the default optimization profile based on system capabilities."""
        # This could be enhanced to detect system capabilities
        return OptimizationProfile(
            name="balanced",
            description="Default balanced profile",
            max_concurrent_animations=10,
            animation_quality_multiplier=0.8,
            enable_complex_easing=True,
            enable_staggered_animations=False,
            enable_micro_interactions=True,
            frame_rate_target=45.0,
            memory_limit_mb=100.0
        )
    
    def optimize_animation_config(self, base_config: AnimationConfig) -> AnimationConfig:
        """Optimize an animation configuration based on current performance."""
        # Get performance-optimized config from monitor
        optimized_config = self.performance_monitor.get_optimized_config(base_config)
        
        # Apply profile-specific optimizations
        profile = self.current_profile
        
        # Adjust duration based on quality multiplier
        optimized_duration = optimized_config.duration * profile.animation_quality_multiplier
        
        # Simplify easing if complex easing is disabled
        easing = optimized_config.easing
        if not profile.enable_complex_easing:
            if easing in [EasingFunction.BOUNCE, EasingFunction.ELASTIC]:
                easing = EasingFunction.EASE_OUT
        
        return AnimationConfig(
            duration=max(optimized_duration, 0.05),  # Minimum 50ms
            easing=easing,
            delay=optimized_config.delay if profile.enable_staggered_animations else 0.0,
            reduced_motion=optimized_config.reduced_motion,
            performance_mode=optimized_config.performance_mode
        )
    
    def should_skip_animation(self, animation_type: AnimationType) -> bool:
        """Determine if an animation should be skipped based on current performance."""
        # Check if we're at animation limit
        active_animations = self.performance_monitor.metrics.animation_count
        if active_animations >= self.current_profile.max_concurrent_animations:
            return True
        
        # Skip micro-interactions if disabled
        if not self.current_profile.enable_micro_interactions:
            if animation_type in [AnimationType.BUTTON_PRESS]:
                return True
        
        # Skip animations in fallback mode except critical ones
        if self.fallback_mode_active:
            critical_animations = [AnimationType.FADE_IN, AnimationType.FADE_OUT]
            if animation_type not in critical_animations:
                return True
        
        return False
    
    def queue_animation(self, animation_data: Dict[str, Any]) -> bool:
        """Queue an animation for later execution if system is overloaded."""
        if len(self.animation_queue) >= self.max_queue_size:
            logger.warning("Animation queue full, dropping animation")
            return False
        
        animation_data["queued_at"] = time.time()
        self.animation_queue.append(animation_data)
        logger.debug(f"Animation queued: {animation_data.get('type', 'unknown')}")
        return True
    
    async def process_animation_queue(self) -> None:
        """Process queued animations when performance improves."""
        if not self.animation_queue:
            return
        
        # Check if we can process animations
        active_animations = self.performance_monitor.metrics.animation_count
        available_slots = max(0, self.current_profile.max_concurrent_animations - active_animations)
        
        if available_slots <= 0:
            return
        
        # Process animations from queue
        processed = 0
        while self.animation_queue and processed < available_slots:
            animation_data = self.animation_queue.pop(0)
            
            # Check if animation is still relevant (not too old)
            age = time.time() - animation_data.get("queued_at", 0)
            if age > 5.0:  # Skip animations older than 5 seconds
                logger.debug("Skipping stale queued animation")
                continue
            
            # Execute the animation
            try:
                callback = animation_data.get("callback")
                if callback:
                    await callback()
                processed += 1
                logger.debug("Processed queued animation")
            except Exception as e:
                logger.error(f"Error processing queued animation: {e}")
    
    def auto_optimize_performance(self) -> bool:
        """Automatically optimize performance based on current metrics."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_optimization_time < self.optimization_cooldown:
            return False
        
        performance_level = self.performance_monitor._determine_performance_level()
        current_profile_name = self.current_profile.name
        
        # Determine target profile based on performance level
        target_profile_name = self._get_target_profile_for_performance(performance_level)
        
        if target_profile_name != current_profile_name:
            self.switch_to_profile(target_profile_name)
            self.last_optimization_time = current_time
            
            # Log optimization
            optimization_event = {
                "timestamp": current_time,
                "from_profile": current_profile_name,
                "to_profile": target_profile_name,
                "performance_level": performance_level,
                "trigger": "auto_optimization"
            }
            self.optimization_history.append(optimization_event)
            
            logger.info(f"Auto-optimized: {current_profile_name} -> {target_profile_name}")
            return True
        
        return False
    
    def _get_target_profile_for_performance(self, performance_level: str) -> str:
        """Get the target optimization profile for a given performance level."""
        profile_mapping = {
            "high": "high",
            "medium": "balanced",
            "low": "performance",
            "critical": "fallback"
        }
        return profile_mapping.get(performance_level, "balanced")
    
    def switch_to_profile(self, profile_name: str) -> bool:
        """Switch to a specific optimization profile."""
        if profile_name not in self.optimization_profiles:
            logger.error(f"Unknown optimization profile: {profile_name}")
            return False
        
        old_profile = self.current_profile
        self.current_profile = self.optimization_profiles[profile_name]
        
        # Update fallback mode status
        self.fallback_mode_active = (profile_name == "fallback")
        
        # Notify callbacks
        for callback in self.optimization_callbacks:
            try:
                callback(profile_name, self.current_profile)
            except Exception as e:
                logger.error(f"Error in optimization callback: {e}")
        
        logger.info(f"Switched optimization profile: {old_profile.name} -> {profile_name}")
        return True
    
    def add_optimization_callback(self, callback: Callable[[str, OptimizationProfile], None]) -> None:
        """Add a callback to be notified when optimization profile changes."""
        self.optimization_callbacks.append(callback)
    
    def remove_optimization_callback(self, callback: Callable[[str, OptimizationProfile], None]) -> None:
        """Remove an optimization callback."""
        if callback in self.optimization_callbacks:
            self.optimization_callbacks.remove(callback)
    
    def get_memory_usage_for_animations(self) -> float:
        """Get estimated memory usage for current animations in MB."""
        # This is a simplified estimation
        base_memory_per_animation = 0.5  # MB per animation
        active_animations = self.performance_monitor.metrics.animation_count
        return active_animations * base_memory_per_animation
    
    def is_memory_limit_exceeded(self) -> bool:
        """Check if animation memory usage exceeds the current profile limit."""
        current_usage = self.get_memory_usage_for_animations()
        return current_usage > self.current_profile.memory_limit_mb
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of current optimization state."""
        return {
            "current_profile": {
                "name": self.current_profile.name,
                "description": self.current_profile.description,
                "max_concurrent_animations": self.current_profile.max_concurrent_animations,
                "quality_multiplier": self.current_profile.animation_quality_multiplier,
                "frame_rate_target": self.current_profile.frame_rate_target
            },
            "optimization_state": {
                "optimization_active": self.optimization_active,
                "fallback_mode_active": self.fallback_mode_active,
                "queued_animations": len(self.animation_queue),
                "memory_usage_mb": self.get_memory_usage_for_animations(),
                "memory_limit_exceeded": self.is_memory_limit_exceeded()
            },
            "recent_optimizations": self.optimization_history[-5:] if self.optimization_history else []
        }
    
    def enable_optimization(self, enabled: bool = True) -> None:
        """Enable or disable automatic optimization."""
        self.optimization_active = enabled
        logger.info(f"Performance optimization {'enabled' if enabled else 'disabled'}")
    
    def clear_animation_queue(self) -> int:
        """Clear all queued animations and return the count cleared."""
        count = len(self.animation_queue)
        self.animation_queue.clear()
        logger.info(f"Cleared {count} queued animations")
        return count
    
    def get_recommended_profile(self) -> str:
        """Get the recommended optimization profile based on current system state."""
        performance_level = self.performance_monitor._determine_performance_level()
        return self._get_target_profile_for_performance(performance_level)
    
    async def start_optimization_loop(self) -> None:
        """Start the automatic optimization loop."""
        logger.info("Starting performance optimization loop")
        
        while self.optimization_active:
            try:
                # Auto-optimize performance
                self.auto_optimize_performance()
                
                # Process animation queue
                await self.process_animation_queue()
                
                # Sleep for optimization interval
                await asyncio.sleep(self.optimization_cooldown)
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error