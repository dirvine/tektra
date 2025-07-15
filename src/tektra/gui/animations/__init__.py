"""
Animation System for Tektra AI Assistant

This module provides a comprehensive animation system for creating smooth,
performant UI animations and transitions.
"""

from .animation_manager import AnimationManager
from .transition_engine import TransitionEngine
from .performance_monitor import UIPerformanceMonitor
from .animation_config import AnimationConfig, AnimationState

__all__ = [
    "AnimationManager",
    "TransitionEngine", 
    "UIPerformanceMonitor",
    "AnimationConfig",
    "AnimationState"
]