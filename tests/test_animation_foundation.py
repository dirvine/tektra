"""
Tests for Animation System Foundation

Simple tests for the core animation system without complex dependencies.
"""

import asyncio
import pytest
import time
import sys
import os
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tektra.gui.animations.animation_config import (
    AnimationConfig,
    AnimationType,
    EasingFunction,
    AnimationState,
    PerformanceMetrics,
    DEFAULT_ANIMATIONS,
    REDUCED_MOTION_ANIMATIONS
)


class TestAnimationConfig:
    """Test animation configuration."""
    
    def test_default_config(self):
        """Test default animation configuration."""
        config = AnimationConfig()
        assert config.duration == 0.3
        assert config.easing == EasingFunction.EASE_OUT
        assert config.delay == 0.0
        assert config.reduced_motion is False
        assert config.performance_mode == "balanced"
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = AnimationConfig(duration=-1, delay=-1, performance_mode="invalid")
        assert config.duration == 0.3  # Should be corrected
        assert config.delay == 0.0  # Should be corrected
        assert config.performance_mode == "balanced"  # Should be corrected
    
    def test_animation_types(self):
        """Test animation type enumeration."""
        assert AnimationType.FADE_IN.value == "fade_in"
        assert AnimationType.SLIDE_IN.value == "slide_in"
        assert AnimationType.SCALE_IN.value == "scale_in"
    
    def test_easing_functions(self):
        """Test easing function enumeration."""
        assert EasingFunction.LINEAR.value == "linear"
        assert EasingFunction.EASE_OUT.value == "ease_out"
        assert EasingFunction.BOUNCE.value == "bounce"
    
    def test_default_animations(self):
        """Test default animation configurations."""
        assert AnimationType.FADE_IN in DEFAULT_ANIMATIONS
        assert AnimationType.SLIDE_IN in DEFAULT_ANIMATIONS
        assert AnimationType.TYPING_INDICATOR in DEFAULT_ANIMATIONS
        
        fade_config = DEFAULT_ANIMATIONS[AnimationType.FADE_IN]
        assert fade_config.duration == 0.3
        assert fade_config.easing == EasingFunction.EASE_OUT
    
    def test_reduced_motion_animations(self):
        """Test reduced motion configurations."""
        assert AnimationType.FADE_IN in REDUCED_MOTION_ANIMATIONS
        
        reduced_fade = REDUCED_MOTION_ANIMATIONS[AnimationType.FADE_IN]
        normal_fade = DEFAULT_ANIMATIONS[AnimationType.FADE_IN]
        
        # Reduced motion should be faster
        assert reduced_fade.duration < normal_fade.duration
        assert reduced_fade.reduced_motion is True


class TestAnimationState:
    """Test animation state tracking."""
    
    def test_animation_state_creation(self):
        """Test creating animation state."""
        config = AnimationConfig(duration=1.0)
        mock_widget = Mock()
        
        state = AnimationState(
            animation_id="test-123",
            animation_type=AnimationType.FADE_IN,
            target_widget=mock_widget,
            config=config
        )
        
        assert state.animation_id == "test-123"
        assert state.animation_type == AnimationType.FADE_IN
        assert state.target_widget == mock_widget
        assert state.config == config
        assert state.progress == 0.0
        assert state.is_running is True
        assert state.is_paused is False
    
    def test_progress_calculation(self):
        """Test animation progress calculation."""
        config = AnimationConfig(duration=1.0, delay=0.0)
        state = AnimationState(
            animation_id="test",
            animation_type=AnimationType.FADE_IN,
            target_widget=Mock(),
            config=config
        )
        
        # Simulate time passing
        time.sleep(0.1)
        progress = state.update_progress()
        
        # Should have some progress
        assert 0.0 <= progress <= 1.0
        
        # If enough time passes, should complete
        state.start_time = time.time() - 2.0  # 2 seconds ago
        progress = state.update_progress()
        assert progress == 1.0
        assert state.is_complete is True
    
    def test_animation_with_delay(self):
        """Test animation with delay."""
        config = AnimationConfig(duration=1.0, delay=0.5)
        state = AnimationState(
            animation_id="test",
            animation_type=AnimationType.FADE_IN,
            target_widget=Mock(),
            config=config
        )
        
        # Should be at 0 progress during delay
        progress = state.update_progress()
        assert progress == 0.0
        
        # Simulate delay passing
        state.start_time = time.time() - 0.6  # Past delay
        progress = state.update_progress()
        assert progress > 0.0


class TestPerformanceMetrics:
    """Test performance metrics tracking."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        assert len(metrics.frame_times) == 0
        assert metrics.animation_count == 0
        assert metrics.dropped_frames == 0
        assert metrics.average_frame_time == 0.0
        assert metrics.fps == 0.0
    
    def test_frame_time_recording(self):
        """Test frame time recording."""
        metrics = PerformanceMetrics()
        
        metrics.add_frame_time(10.0)
        metrics.add_frame_time(20.0)
        
        assert len(metrics.frame_times) == 2
        assert metrics.average_frame_time == 15.0
        assert abs(metrics.fps - 66.67) < 1.0  # 1000/15 ≈ 66.67
    
    def test_dropped_frame_detection(self):
        """Test dropped frame detection."""
        metrics = PerformanceMetrics()
        
        # Good frame
        metrics.add_frame_time(10.0)
        assert metrics.dropped_frames == 0
        
        # Dropped frame (>16.67ms)
        metrics.add_frame_time(20.0)
        assert metrics.dropped_frames == 1
        
        # Another dropped frame
        metrics.add_frame_time(25.0)
        assert metrics.dropped_frames == 2
    
    def test_frame_time_limit(self):
        """Test frame time list size limit."""
        metrics = PerformanceMetrics()
        
        # Add more than 60 frame times
        for i in range(70):
            metrics.add_frame_time(10.0)
        
        # Should be limited to 60
        assert len(metrics.frame_times) == 60


def test_easing_functions():
    """Test easing function imports and basic functionality."""
    from tektra.gui.animations.easing import (
        linear, 
        ease_in_quad, 
        ease_out_quad,
        apply_easing,
        get_easing_function
    )
    
    # Test linear function
    assert linear(0.0) == 0.0
    assert linear(0.5) == 0.5
    assert linear(1.0) == 1.0
    
    # Test quadratic functions
    assert ease_in_quad(0.0) == 0.0
    assert ease_in_quad(1.0) == 1.0
    assert ease_out_quad(0.0) == 0.0
    assert ease_out_quad(1.0) == 1.0
    
    # Test easing application
    result = apply_easing(0.5, EasingFunction.LINEAR)
    assert result == 0.5
    
    # Test getting easing function
    func = get_easing_function(EasingFunction.LINEAR)
    assert func(0.5) == 0.5


def test_performance_monitor_basic():
    """Test basic performance monitor functionality."""
    try:
        from tektra.gui.animations.performance_monitor import UIPerformanceMonitor
        
        monitor = UIPerformanceMonitor()
        assert monitor.performance_threshold == 16.67
        assert monitor.monitoring_enabled is True
        
        # Test frame time recording
        monitor.record_frame_time(10.0)
        assert len(monitor.metrics.frame_times) == 1
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        assert isinstance(summary, dict)
        assert "fps" in summary
        assert "should_reduce_animations" in summary
        
    except ImportError as e:
        pytest.skip(f"Performance monitor dependencies not available: {e}")


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", __file__, "-v"
    ], cwd=os.path.dirname(os.path.dirname(__file__)))
    
    sys.exit(result.returncode)