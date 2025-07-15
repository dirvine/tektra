"""
Easing Functions for Animations

This module provides mathematical easing functions for smooth animations.
"""

import math
from typing import Callable

from .animation_config import EasingFunction


def linear(t: float) -> float:
    """Linear easing function."""
    return t


def ease_in_quad(t: float) -> float:
    """Quadratic ease-in function."""
    return t * t


def ease_out_quad(t: float) -> float:
    """Quadratic ease-out function."""
    return 1 - (1 - t) * (1 - t)


def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out function."""
    if t < 0.5:
        return 2 * t * t
    return 1 - pow(-2 * t + 2, 2) / 2


def ease_in_cubic(t: float) -> float:
    """Cubic ease-in function."""
    return t * t * t


def ease_out_cubic(t: float) -> float:
    """Cubic ease-out function."""
    return 1 - pow(1 - t, 3)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out function."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_bounce(t: float) -> float:
    """Bounce ease-out function."""
    n1 = 7.5625
    d1 = 2.75
    
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


def ease_out_elastic(t: float) -> float:
    """Elastic ease-out function."""
    if t == 0:
        return 0
    if t == 1:
        return 1
    
    c4 = (2 * math.pi) / 3
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_in_back(t: float) -> float:
    """Back ease-in function."""
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * t * t * t - c1 * t * t


def ease_out_back(t: float) -> float:
    """Back ease-out function."""
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


# Mapping of easing functions
EASING_FUNCTIONS: dict[EasingFunction, Callable[[float], float]] = {
    EasingFunction.LINEAR: linear,
    EasingFunction.EASE_IN: ease_in_cubic,
    EasingFunction.EASE_OUT: ease_out_cubic,
    EasingFunction.EASE_IN_OUT: ease_in_out_cubic,
    EasingFunction.BOUNCE: ease_out_bounce,
    EasingFunction.ELASTIC: ease_out_elastic,
}


def get_easing_function(easing: EasingFunction) -> Callable[[float], float]:
    """Get the easing function for the given type."""
    return EASING_FUNCTIONS.get(easing, linear)


def apply_easing(progress: float, easing: EasingFunction) -> float:
    """Apply easing function to progress value."""
    # Clamp progress to [0, 1]
    progress = max(0.0, min(1.0, progress))
    
    easing_func = get_easing_function(easing)
    return easing_func(progress)