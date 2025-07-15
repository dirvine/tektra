"""
Tests for Performance Optimizer

Tests the automatic performance optimization system including profile switching,
animation queuing, and fallback modes.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from src.tektra.gui.animations.performance_optimizer import (
    PerformanceOptimizer,
    OptimizationProfile
)
from src.tektra.gui.animations.performance_monitor import UIPerformanceMonitor, PerformanceThresholds
from src.tektra.gui.animations.animation_config import AnimationConfig, AnimationType, EasingFunction


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a performance monitor for testing."""
        return UIPerformanceMonitor(PerformanceThresholds())
    
    @pytest.fixture
    def optimizer(self, performance_monitor):
        """Create a performance optimizer for testing."""
        return PerformanceOptimizer(performance_monitor)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.current_profile.name == "balanced"
        assert not optimizer.optimization_active
        assert not optimizer.fallback_mode_active
        assert len(optimizer.animation_queue) == 0
        assert len(optimizer.optimization_profiles) == 6  # All predefined profiles
    
    def test_optimization_profiles(self, optimizer):
        """Test that all optimization profiles are properly defined."""
        expected_profiles = ["ultra", "high", "balanced", "performance", "minimal", "fallback"]
        
        for profile_name in expected_profiles:
            assert profile_name in optimizer.optimization_profiles
            profile = optimizer.optimization_profiles[profile_name]
            assert isinstance(profile, OptimizationProfile)
            assert profile.name == profile_name
            assert profile.max_concurrent_animations > 0
            assert 0.0 <= profile.animation_quality_multiplier <= 1.0
    
    def test_animation_config_optimization(self, optimizer):
        """Test animation configuration optimization."""
        base_config = AnimationConfig(
            duration=0.5,
            easing=EasingFunction.BOUNCE,
            delay=0.2
        )
        
        # Test with high performance profile
        optimizer.switch_to_profile("high")
        optimized = optimizer.optimize_animation_config(base_config)
        assert optimized.duration <= base_config.duration
        assert optimized.easing == EasingFunction.BOUNCE  # Complex easing allowed
        
        # Test with performance profile (no complex easing)
        optimizer.switch_to_profile("performance")
        optimized = optimizer.optimize_animation_config(base_config)
        assert optimized.duration < base_config.duration
        assert optimized.easing == EasingFunction.EASE_OUT  # Simplified easing
        assert optimized.delay == 0.0  # No staggered animations
    
    def test_animation_skipping_logic(self, optimizer):
        """Test animation skipping based on performance constraints."""
        # Test with balanced profile
        optimizer.switch_to_profile("balanced")
        
        # Should not skip with low animation count
        assert not optimizer.should_skip_animation(AnimationType.FADE_IN)
        
        # Simulate high animation count
        optimizer.performance_monitor.metrics.animation_count = 15
        assert optimizer.should_skip_animation(AnimationType.FADE_IN)
        
        # Test micro-interaction skipping
        optimizer.switch_to_profile("performance")  # Disables micro-interactions
        assert optimizer.should_skip_animation(AnimationType.BUTTON_PRESS)
        
        # Test fallback mode
        optimizer.switch_to_profile("fallback")  # This sets fallback_mode_active = True
        optimizer.performance_monitor.metrics.animation_count = 0  # Reset animation count
        assert optimizer.should_skip_animation(AnimationType.SLIDE_IN)
        assert not optimizer.should_skip_animation(AnimationType.FADE_IN)  # Critical animation
    
    def test_animation_queuing(self, optimizer):
        """Test animation queuing system."""
        # Queue some animations
        animation1 = {"type": "fade_in", "callback": Mock()}
        animation2 = {"type": "slide_in", "callback": Mock()}
        
        assert optimizer.queue_animation(animation1)
        assert optimizer.queue_animation(animation2)
        assert len(optimizer.animation_queue) == 2
        
        # Test queue size limit
        for i in range(optimizer.max_queue_size):
            optimizer.queue_animation({"type": f"test_{i}"})
        
        # Should reject when queue is full
        assert not optimizer.queue_animation({"type": "overflow"})
    
    @pytest.mark.asyncio
    async def test_animation_queue_processing(self, optimizer):
        """Test processing of queued animations."""
        # Set up profile with limited concurrent animations
        optimizer.switch_to_profile("minimal")  # Max 3 concurrent
        
        # Queue some animations
        callback1 = Mock()
        callback2 = Mock()
        
        optimizer.queue_animation({"callback": callback1})
        optimizer.queue_animation({"callback": callback2})
        
        # Process queue
        await optimizer.process_animation_queue()
        
        # Callbacks should have been called
        callback1.assert_called_once()
        callback2.assert_called_once()
        assert len(optimizer.animation_queue) == 0
    
    @pytest.mark.asyncio
    async def test_stale_animation_cleanup(self, optimizer):
        """Test cleanup of stale queued animations."""
        # Queue an old animation
        old_animation = {
            "type": "fade_in",
            "callback": Mock(),
            "queued_at": time.time() - 10.0  # 10 seconds ago
        }
        optimizer.animation_queue.append(old_animation)
        
        # Process queue
        await optimizer.process_animation_queue()
        
        # Old animation should not have been executed
        old_animation["callback"].assert_not_called()
    
    def test_profile_switching(self, optimizer):
        """Test switching between optimization profiles."""
        # Test valid profile switch
        assert optimizer.switch_to_profile("high")
        assert optimizer.current_profile.name == "high"
        assert not optimizer.fallback_mode_active
        
        # Test fallback mode activation
        assert optimizer.switch_to_profile("fallback")
        assert optimizer.current_profile.name == "fallback"
        assert optimizer.fallback_mode_active
        
        # Test invalid profile
        assert not optimizer.switch_to_profile("nonexistent")
        assert optimizer.current_profile.name == "fallback"  # Should remain unchanged
    
    def test_auto_optimization(self, optimizer):
        """Test automatic performance optimization."""
        # Mock performance monitor to return different performance levels
        with patch.object(optimizer.performance_monitor, '_determine_performance_level') as mock_perf:
            # Test optimization from high to low performance
            mock_perf.return_value = "low"
            optimizer.current_profile = optimizer.optimization_profiles["high"]
            
            result = optimizer.auto_optimize_performance()
            assert result is True
            assert optimizer.current_profile.name == "performance"
            
            # Test no optimization needed
            mock_perf.return_value = "high"
            optimizer.current_profile = optimizer.optimization_profiles["high"]
            
            result = optimizer.auto_optimize_performance()
            assert result is False
    
    def test_optimization_cooldown(self, optimizer):
        """Test optimization cooldown mechanism."""
        with patch.object(optimizer.performance_monitor, '_determine_performance_level', return_value="critical"):
            # Reset the optimizer to a different profile first
            optimizer.switch_to_profile("high")
            
            # First optimization should work
            result1 = optimizer.auto_optimize_performance()
            assert result1 is True
            
            # Second optimization should be blocked by cooldown
            result2 = optimizer.auto_optimize_performance()
            assert result2 is False
            
            # After cooldown period, should work again
            # Reset to a different profile and adjust time
            optimizer.switch_to_profile("high")
            optimizer.last_optimization_time = time.time() - optimizer.optimization_cooldown - 1
            result3 = optimizer.auto_optimize_performance()
            assert result3 is True
    
    def test_memory_usage_tracking(self, optimizer):
        """Test memory usage estimation and limits."""
        # Test memory usage calculation
        optimizer.performance_monitor.metrics.animation_count = 10
        memory_usage = optimizer.get_memory_usage_for_animations()
        assert memory_usage == 5.0  # 10 animations * 0.5 MB each
        
        # Test memory limit checking
        optimizer.switch_to_profile("minimal")  # 50MB limit
        optimizer.performance_monitor.metrics.animation_count = 200  # 100MB usage
        assert optimizer.is_memory_limit_exceeded()
        
        optimizer.performance_monitor.metrics.animation_count = 50  # 25MB usage
        assert not optimizer.is_memory_limit_exceeded()
    
    def test_optimization_callbacks(self, optimizer):
        """Test optimization callback system."""
        callback_called = False
        callback_profile = None
        
        def test_callback(profile_name, profile):
            nonlocal callback_called, callback_profile
            callback_called = True
            callback_profile = profile_name
        
        # Add callback
        optimizer.add_optimization_callback(test_callback)
        
        # Switch profile to trigger callback
        optimizer.switch_to_profile("performance")
        
        assert callback_called
        assert callback_profile == "performance"
        
        # Remove callback
        optimizer.remove_optimization_callback(test_callback)
        
        # Reset test state
        callback_called = False
        callback_profile = None
        
        # Switch profile again - callback should not be called
        optimizer.switch_to_profile("high")
        assert not callback_called
    
    def test_optimization_summary(self, optimizer):
        """Test optimization summary generation."""
        # Set up some test state
        optimizer.switch_to_profile("performance")
        optimizer.queue_animation({"type": "test"})
        optimizer.performance_monitor.metrics.animation_count = 5
        
        summary = optimizer.get_optimization_summary()
        
        assert "current_profile" in summary
        assert "optimization_state" in summary
        assert "recent_optimizations" in summary
        
        # Check specific values
        assert summary["current_profile"]["name"] == "performance"
        assert summary["optimization_state"]["queued_animations"] == 1
        assert summary["optimization_state"]["memory_usage_mb"] == 2.5  # 5 * 0.5
    
    def test_recommended_profile(self, optimizer):
        """Test recommended profile calculation."""
        with patch.object(optimizer.performance_monitor, '_determine_performance_level') as mock_perf:
            mock_perf.return_value = "high"
            assert optimizer.get_recommended_profile() == "high"
            
            mock_perf.return_value = "medium"
            assert optimizer.get_recommended_profile() == "balanced"
            
            mock_perf.return_value = "low"
            assert optimizer.get_recommended_profile() == "performance"
            
            mock_perf.return_value = "critical"
            assert optimizer.get_recommended_profile() == "fallback"
    
    def test_animation_queue_clearing(self, optimizer):
        """Test clearing the animation queue."""
        # Add some animations to queue
        for i in range(5):
            optimizer.queue_animation({"type": f"test_{i}"})
        
        assert len(optimizer.animation_queue) == 5
        
        # Clear queue
        cleared_count = optimizer.clear_animation_queue()
        
        assert cleared_count == 5
        assert len(optimizer.animation_queue) == 0
    
    def test_optimization_history_tracking(self, optimizer):
        """Test optimization history tracking."""
        with patch.object(optimizer.performance_monitor, '_determine_performance_level', return_value="critical"):
            # Perform optimization
            optimizer.auto_optimize_performance()
            
            # Check history was recorded
            assert len(optimizer.optimization_history) == 1
            
            history_entry = optimizer.optimization_history[0]
            assert "timestamp" in history_entry
            assert "from_profile" in history_entry
            assert "to_profile" in history_entry
            assert "performance_level" in history_entry
            assert history_entry["trigger"] == "auto_optimization"
    
    @pytest.mark.asyncio
    async def test_optimization_loop(self, optimizer):
        """Test the automatic optimization loop."""
        optimizer.enable_optimization(True)
        
        # Create a task for the optimization loop
        optimization_task = asyncio.create_task(optimizer.start_optimization_loop())
        
        # Let it run for a short time
        await asyncio.sleep(0.1)
        
        # Stop optimization and cancel task
        optimizer.enable_optimization(False)
        optimization_task.cancel()
        
        try:
            await optimization_task
        except asyncio.CancelledError:
            pass  # Expected when cancelling
    
    def test_enable_disable_optimization(self, optimizer):
        """Test enabling and disabling optimization."""
        assert not optimizer.optimization_active
        
        optimizer.enable_optimization(True)
        assert optimizer.optimization_active
        
        optimizer.enable_optimization(False)
        assert not optimizer.optimization_active