"""
Tests for the Animation System

This module contains tests for the core animation system components.
"""

import asyncio
import pytest
import time
import sys
import os
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tektra.gui.animations.animation_config import (
    AnimationConfig,
    AnimationType,
    EasingFunction,
    AnimationState,
    PerformanceMetrics
)
from tektra.gui.animations.performance_monitor import UIPerformanceMonitor
from tektra.gui.animations.transition_engine import TransitionEngine
from tektra.gui.animations.animation_manager import AnimationManager


class MockWidget:
    """Mock Toga widget for testing."""
    
    def __init__(self):
        self.style = Mock()
        self.visible = True


@pytest.fixture
def mock_widget():
    """Create a mock widget for testing."""
    return MockWidget()


@pytest.fixture
def performance_monitor():
    """Create a performance monitor for testing."""
    return UIPerformanceMonitor()


@pytest.fixture
def transition_engine(performance_monitor):
    """Create a transition engine for testing."""
    return TransitionEngine(performance_monitor)


@pytest.fixture
def animation_manager():
    """Create an animation manager for testing."""
    return AnimationManager()


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


class TestUIPerformanceMonitor:
    """Test UI performance monitoring."""
    
    def test_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.thresholds.target_fps == 60.0
        assert performance_monitor.monitoring_enabled is True
        assert len(performance_monitor.metrics.frame_times) == 0
    
    def test_frame_time_recording(self, performance_monitor):
        """Test frame time recording."""
        performance_monitor.record_frame_time(10.0)
        performance_monitor.record_frame_time(20.0)
        
        assert len(performance_monitor.metrics.frame_times) == 2
        assert performance_monitor.metrics.average_frame_time == 15.0
    
    def test_fps_calculation(self, performance_monitor):
        """Test FPS calculation."""
        performance_monitor.record_frame_time(16.67)  # 60 FPS
        assert abs(performance_monitor.metrics.fps - 60.0) < 1.0
    
    def test_dropped_frame_detection(self, performance_monitor):
        """Test dropped frame detection."""
        performance_monitor.record_frame_time(20.0)  # Dropped frame
        assert performance_monitor.metrics.dropped_frames == 1
    
    def test_performance_recommendations(self, performance_monitor):
        """Test performance recommendations."""
        # Add some poor performance data
        for _ in range(20):
            performance_monitor.record_frame_time(40.0)  # Poor performance
        
        recommendations = performance_monitor.get_performance_recommendations()
        assert len(recommendations) > 0
        assert any("reduced motion" in rec.lower() for rec in recommendations)
    
    def test_should_reduce_animations(self, performance_monitor):
        """Test animation reduction logic."""
        # Good performance - should not reduce
        for _ in range(10):
            performance_monitor.record_frame_time(10.0)
        assert not performance_monitor.should_reduce_animations()
        
        # Poor performance - should reduce
        for _ in range(20):
            performance_monitor.record_frame_time(50.0)
        assert performance_monitor.should_reduce_animations()


class TestTransitionEngine:
    """Test transition engine."""
    
    def test_initialization(self, transition_engine):
        """Test transition engine initialization."""
        assert transition_engine.frame_rate == 60
        assert transition_engine.reduced_motion_enabled is False
        assert len(transition_engine.active_animations) == 0
    
    @pytest.mark.asyncio
    async def test_fade_in_animation(self, transition_engine, mock_widget):
        """Test fade in animation."""
        animation_id = await transition_engine.fade_in(mock_widget, duration=0.1)
        
        assert animation_id is not None
        assert len(animation_id) > 0
        assert animation_id in transition_engine.active_animations
        
        # Wait for animation to complete
        await asyncio.sleep(0.2)
        assert animation_id not in transition_engine.active_animations
    
    @pytest.mark.asyncio
    async def test_slide_in_animation(self, transition_engine, mock_widget):
        """Test slide in animation."""
        animation_id = await transition_engine.slide_in(
            mock_widget, direction="bottom", duration=0.1
        )
        
        assert animation_id is not None
        assert animation_id in transition_engine.active_animations
        
        # Check that direction parameter is stored
        animation_state = transition_engine.active_animations[animation_id]
        assert hasattr(animation_state, 'direction')
        assert animation_state.direction == "bottom"
    
    @pytest.mark.asyncio
    async def test_scale_in_animation(self, transition_engine, mock_widget):
        """Test scale in animation."""
        animation_id = await transition_engine.scale_in(
            mock_widget, from_scale=0.5, duration=0.1
        )
        
        assert animation_id is not None
        animation_state = transition_engine.active_animations[animation_id]
        assert hasattr(animation_state, 'from_scale')
        assert animation_state.from_scale == 0.5
    
    def test_cancel_animation(self, transition_engine, mock_widget):
        """Test animation cancellation."""
        # Start an animation
        animation_id = asyncio.run(
            transition_engine.fade_in(mock_widget, duration=1.0)
        )
        
        # Cancel it
        result = transition_engine.cancel_animation(animation_id)
        assert result is True
        assert animation_id not in transition_engine.active_animations
        
        # Try to cancel non-existent animation
        result = transition_engine.cancel_animation("non-existent")
        assert result is False
    
    def test_pause_resume_animation(self, transition_engine, mock_widget):
        """Test animation pause and resume."""
        animation_id = asyncio.run(
            transition_engine.fade_in(mock_widget, duration=1.0)
        )
        
        # Pause animation
        result = transition_engine.pause_animation(animation_id)
        assert result is True
        assert transition_engine.active_animations[animation_id].is_paused is True
        
        # Resume animation
        result = transition_engine.resume_animation(animation_id)
        assert result is True
        assert transition_engine.active_animations[animation_id].is_paused is False
    
    def test_reduced_motion_mode(self, transition_engine):
        """Test reduced motion mode."""
        transition_engine.set_reduced_motion(True)
        assert transition_engine.reduced_motion_enabled is True
        
        transition_engine.set_reduced_motion(False)
        assert transition_engine.reduced_motion_enabled is False


class TestAnimationManager:
    """Test animation manager."""
    
    def test_initialization(self, animation_manager):
        """Test animation manager initialization."""
        assert animation_manager.global_animation_enabled is True
        assert animation_manager.performance_monitor is not None
        assert animation_manager.transition_engine is not None
        assert len(animation_manager.animation_presets) > 0
    
    @pytest.mark.asyncio
    async def test_message_animation(self, animation_manager, mock_widget):
        """Test message appearance animation."""
        # Test user message
        animation_id = await animation_manager.animate_message_appearance(
            mock_widget, role="user"
        )
        assert animation_id is not None or animation_id == ""  # May be empty if disabled
        
        # Test assistant message
        animation_id = await animation_manager.animate_message_appearance(
            mock_widget, role="assistant"
        )
        assert animation_id is not None or animation_id == ""
    
    @pytest.mark.asyncio
    async def test_button_press_animation(self, animation_manager):
        """Test button press animation."""
        mock_button = Mock()
        animation_id = await animation_manager.animate_button_press(mock_button)
        assert animation_id is not None or animation_id == ""
    
    def test_global_animation_toggle(self, animation_manager):
        """Test global animation enable/disable."""
        animation_manager.set_global_animations_enabled(False)
        assert animation_manager.global_animation_enabled is False
        
        animation_manager.set_global_animations_enabled(True)
        assert animation_manager.global_animation_enabled is True
    
    def test_reduced_motion_toggle(self, animation_manager):
        """Test reduced motion toggle."""
        animation_manager.set_reduced_motion(True)
        assert animation_manager.reduced_motion_enabled is True
        assert animation_manager.transition_engine.reduced_motion_enabled is True
    
    def test_animation_presets(self, animation_manager):
        """Test animation presets."""
        # Test built-in presets
        assert "performance" in animation_manager.animation_presets
        assert "accessibility" in animation_manager.animation_presets
        assert "full" in animation_manager.animation_presets
        
        # Test adding custom preset
        custom_preset = {"reduced_motion": True, "global_enabled": False}
        animation_manager.add_animation_preset("custom", custom_preset)
        assert "custom" in animation_manager.animation_presets
        
        # Test applying preset
        result = animation_manager.apply_animation_preset("performance")
        assert result is True
        
        # Test applying non-existent preset
        result = animation_manager.apply_animation_preset("non-existent")
        assert result is False
    
    def test_performance_summary(self, animation_manager):
        """Test performance summary."""
        summary = animation_manager.get_performance_summary()
        assert isinstance(summary, dict)
        assert "fps" in summary
        assert "should_reduce_animations" in summary
    
    def test_cancel_all_animations(self, animation_manager):
        """Test cancelling all animations."""
        # This should work even with no active animations
        cancelled_count = animation_manager.cancel_all_animations()
        assert cancelled_count >= 0


@pytest.mark.asyncio
async def test_animation_integration():
    """Test integration between animation components."""
    # Create components
    performance_monitor = UIPerformanceMonitor()
    transition_engine = TransitionEngine(performance_monitor)
    animation_manager = AnimationManager()
    
    # Create mock widget
    mock_widget = MockWidget()
    
    # Test that performance monitoring affects animations
    # Simulate poor performance
    for _ in range(30):
        performance_monitor.record_frame_time(50.0)  # Poor performance
    
    # Performance monitor should recommend reduced animations
    assert performance_monitor.should_reduce_animations()
    
    # Animation manager should respect performance settings
    config = performance_monitor.get_optimized_config(AnimationConfig())
    assert config.reduced_motion is True
    assert config.duration <= 0.15


if __name__ == "__main__":
    pytest.main([__file__])