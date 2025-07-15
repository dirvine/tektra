"""
Animation Configuration and State Management

This module defines configuration classes and state tracking for animations.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any, Callable


class AnimationType(Enum):
    """Types of animations supported by the system."""
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    SLIDE_IN = "slide_in"
    SLIDE_OUT = "slide_out"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    TYPING_INDICATOR = "typing_indicator"
    BUTTON_PRESS = "button_press"
    THEME_TRANSITION = "theme_transition"


class EasingFunction(Enum):
    """Easing functions for animations."""
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    BOUNCE = "bounce"
    ELASTIC = "elastic"


@dataclass
class AnimationConfig:
    """Configuration for UI animations."""
    duration: float = 0.3
    easing: EasingFunction = EasingFunction.EASE_OUT
    delay: float = 0.0
    reduced_motion: bool = False
    performance_mode: str = "balanced"  # "performance", "balanced", "quality"
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.duration < 0:
            self.duration = 0.3
        if self.delay < 0:
            self.delay = 0.0
        if self.performance_mode not in ["performance", "balanced", "quality"]:
            self.performance_mode = "balanced"


@dataclass
class AnimationState:
    """Tracks the state of an ongoing animation."""
    animation_id: str
    animation_type: AnimationType
    target_widget: Any  # toga.Widget
    config: AnimationConfig
    start_time: float = field(default_factory=time.time)
    progress: float = 0.0
    is_running: bool = True
    is_paused: bool = False
    completion_callback: Optional[Callable] = None
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since animation started."""
        return time.time() - self.start_time
    
    @property
    def is_complete(self) -> bool:
        """Check if animation is complete."""
        return self.progress >= 1.0 or not self.is_running
    
    def update_progress(self) -> float:
        """Update and return current progress (0.0 to 1.0)."""
        if not self.is_running or self.is_paused:
            return self.progress
            
        elapsed = self.elapsed_time - self.config.delay
        if elapsed < 0:
            self.progress = 0.0
        elif elapsed >= self.config.duration:
            self.progress = 1.0
            self.is_running = False
        else:
            self.progress = elapsed / self.config.duration
            
        return self.progress


@dataclass
class PerformanceMetrics:
    """Performance metrics for animation system."""
    frame_times: list = field(default_factory=list)
    animation_count: int = 0
    dropped_frames: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def average_frame_time(self) -> float:
        """Get average frame time in milliseconds."""
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)
    
    @property
    def fps(self) -> float:
        """Get current FPS."""
        avg_frame_time = self.average_frame_time
        if avg_frame_time <= 0:
            return 0.0
        return 1000.0 / avg_frame_time
    
    def add_frame_time(self, frame_time: float):
        """Add a frame time measurement."""
        self.frame_times.append(frame_time)
        # Keep only last 60 measurements
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Count dropped frames (>16.67ms = below 60fps)
        if frame_time > 16.67:
            self.dropped_frames += 1


# Default animation configurations
DEFAULT_ANIMATIONS = {
    AnimationType.FADE_IN: AnimationConfig(duration=0.3, easing=EasingFunction.EASE_OUT),
    AnimationType.FADE_OUT: AnimationConfig(duration=0.2, easing=EasingFunction.EASE_IN),
    AnimationType.SLIDE_IN: AnimationConfig(duration=0.4, easing=EasingFunction.EASE_OUT),
    AnimationType.SLIDE_OUT: AnimationConfig(duration=0.3, easing=EasingFunction.EASE_IN),
    AnimationType.SCALE_IN: AnimationConfig(duration=0.25, easing=EasingFunction.EASE_OUT),
    AnimationType.SCALE_OUT: AnimationConfig(duration=0.2, easing=EasingFunction.EASE_IN),
    AnimationType.TYPING_INDICATOR: AnimationConfig(duration=1.5, easing=EasingFunction.EASE_IN_OUT),
    AnimationType.BUTTON_PRESS: AnimationConfig(duration=0.1, easing=EasingFunction.EASE_OUT),
    AnimationType.THEME_TRANSITION: AnimationConfig(duration=0.5, easing=EasingFunction.EASE_IN_OUT),
}

# Reduced motion configurations (faster, simpler animations)
REDUCED_MOTION_ANIMATIONS = {
    AnimationType.FADE_IN: AnimationConfig(duration=0.1, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.FADE_OUT: AnimationConfig(duration=0.1, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.SLIDE_IN: AnimationConfig(duration=0.15, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.SLIDE_OUT: AnimationConfig(duration=0.1, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.SCALE_IN: AnimationConfig(duration=0.1, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.SCALE_OUT: AnimationConfig(duration=0.1, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.TYPING_INDICATOR: AnimationConfig(duration=0.5, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.BUTTON_PRESS: AnimationConfig(duration=0.05, easing=EasingFunction.LINEAR, reduced_motion=True),
    AnimationType.THEME_TRANSITION: AnimationConfig(duration=0.2, easing=EasingFunction.LINEAR, reduced_motion=True),
}