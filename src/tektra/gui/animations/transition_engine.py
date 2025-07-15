"""
Transition Engine for Smooth UI Animations

This module provides the core transition engine for creating smooth animations
between UI states using Toga widgets.
"""

import asyncio
import time
import uuid
from typing import Dict, Optional, Callable, Any, Tuple
from loguru import logger

import toga
from toga.style import Pack

from .animation_config import AnimationType, AnimationConfig, AnimationState, DEFAULT_ANIMATIONS, REDUCED_MOTION_ANIMATIONS
from .easing import apply_easing
from .performance_monitor import UIPerformanceMonitor


class TransitionEngine:
    """Handles smooth transitions between UI states."""
    
    def __init__(self, performance_monitor: Optional[UIPerformanceMonitor] = None):
        """Initialize the transition engine."""
        self.performance_monitor = performance_monitor or UIPerformanceMonitor()
        self.active_animations: Dict[str, AnimationState] = {}
        self.animation_loop_running = False
        self.reduced_motion_enabled = False
        self.frame_rate = 60  # Target FPS
        self.frame_time = 1.0 / self.frame_rate
        
        logger.info("Transition Engine initialized")
    
    async def fade_in(self, widget: toga.Widget, duration: float = 0.3, 
                     easing: str = "ease_out", delay: float = 0.0,
                     completion_callback: Optional[Callable] = None) -> str:
        """
        Fade in a widget smoothly.
        
        Args:
            widget: The widget to animate
            duration: Animation duration in seconds
            easing: Easing function name
            delay: Delay before starting animation
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        config = self._get_animation_config(AnimationType.FADE_IN, duration, easing, delay)
        return await self._start_animation(
            AnimationType.FADE_IN, widget, config, completion_callback
        )
    
    async def fade_out(self, widget: toga.Widget, duration: float = 0.2,
                      easing: str = "ease_in", delay: float = 0.0,
                      completion_callback: Optional[Callable] = None) -> str:
        """
        Fade out a widget smoothly.
        
        Args:
            widget: The widget to animate
            duration: Animation duration in seconds
            easing: Easing function name
            delay: Delay before starting animation
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        config = self._get_animation_config(AnimationType.FADE_OUT, duration, easing, delay)
        return await self._start_animation(
            AnimationType.FADE_OUT, widget, config, completion_callback
        )
    
    async def slide_in(self, widget: toga.Widget, direction: str = "bottom",
                      duration: float = 0.4, easing: str = "ease_out", delay: float = 0.0,
                      completion_callback: Optional[Callable] = None) -> str:
        """
        Slide in a widget from the specified direction.
        
        Args:
            widget: The widget to animate
            direction: Direction to slide from ("top", "bottom", "left", "right")
            duration: Animation duration in seconds
            easing: Easing function name
            delay: Delay before starting animation
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        config = self._get_animation_config(AnimationType.SLIDE_IN, duration, easing, delay)
        return await self._start_animation(
            AnimationType.SLIDE_IN, widget, config, completion_callback, 
            extra_params={"direction": direction}
        )
    
    async def slide_out(self, widget: toga.Widget, direction: str = "top",
                       duration: float = 0.3, easing: str = "ease_in", delay: float = 0.0,
                       completion_callback: Optional[Callable] = None) -> str:
        """
        Slide out a widget in the specified direction.
        
        Args:
            widget: The widget to animate
            direction: Direction to slide to ("top", "bottom", "left", "right")
            duration: Animation duration in seconds
            easing: Easing function name
            delay: Delay before starting animation
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        config = self._get_animation_config(AnimationType.SLIDE_OUT, duration, easing, delay)
        return await self._start_animation(
            AnimationType.SLIDE_OUT, widget, config, completion_callback,
            extra_params={"direction": direction}
        )
    
    async def scale_in(self, widget: toga.Widget, from_scale: float = 0.8,
                      duration: float = 0.25, easing: str = "ease_out", delay: float = 0.0,
                      completion_callback: Optional[Callable] = None) -> str:
        """
        Scale in a widget from the specified scale.
        
        Args:
            widget: The widget to animate
            from_scale: Starting scale (0.0 to 1.0)
            duration: Animation duration in seconds
            easing: Easing function name
            delay: Delay before starting animation
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        config = self._get_animation_config(AnimationType.SCALE_IN, duration, easing, delay)
        return await self._start_animation(
            AnimationType.SCALE_IN, widget, config, completion_callback,
            extra_params={"from_scale": from_scale}
        )
    
    async def scale_out(self, widget: toga.Widget, to_scale: float = 0.8,
                       duration: float = 0.2, easing: str = "ease_in", delay: float = 0.0,
                       completion_callback: Optional[Callable] = None) -> str:
        """
        Scale out a widget to the specified scale.
        
        Args:
            widget: The widget to animate
            to_scale: Ending scale (0.0 to 1.0)
            duration: Animation duration in seconds
            easing: Easing function name
            delay: Delay before starting animation
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        config = self._get_animation_config(AnimationType.SCALE_OUT, duration, easing, delay)
        return await self._start_animation(
            AnimationType.SCALE_OUT, widget, config, completion_callback,
            extra_params={"to_scale": to_scale}
        )
    
    async def morph_layout(self, container: toga.Box, new_layout: dict,
                          duration: float = 0.3, easing: str = "ease_in_out",
                          completion_callback: Optional[Callable] = None) -> str:
        """
        Smoothly transition a container to a new layout.
        
        Args:
            container: The container to animate
            new_layout: New layout properties
            duration: Animation duration in seconds
            easing: Easing function name
            completion_callback: Callback when animation completes
            
        Returns:
            Animation ID for tracking
        """
        # For now, this is a placeholder - Toga doesn't support layout morphing directly
        # We'll implement this as a fade transition
        logger.warning("Layout morphing not fully supported in Toga, using fade transition")
        return await self.fade_in(container, duration, easing, completion_callback=completion_callback)
    
    def cancel_animation(self, animation_id: str) -> bool:
        """
        Cancel an active animation.
        
        Args:
            animation_id: ID of the animation to cancel
            
        Returns:
            True if animation was cancelled, False if not found
        """
        if animation_id in self.active_animations:
            animation_state = self.active_animations[animation_id]
            animation_state.is_running = False
            del self.active_animations[animation_id]
            logger.debug(f"Cancelled animation {animation_id}")
            return True
        return False
    
    def pause_animation(self, animation_id: str) -> bool:
        """
        Pause an active animation.
        
        Args:
            animation_id: ID of the animation to pause
            
        Returns:
            True if animation was paused, False if not found
        """
        if animation_id in self.active_animations:
            self.active_animations[animation_id].is_paused = True
            logger.debug(f"Paused animation {animation_id}")
            return True
        return False
    
    def resume_animation(self, animation_id: str) -> bool:
        """
        Resume a paused animation.
        
        Args:
            animation_id: ID of the animation to resume
            
        Returns:
            True if animation was resumed, False if not found
        """
        if animation_id in self.active_animations:
            self.active_animations[animation_id].is_paused = False
            logger.debug(f"Resumed animation {animation_id}")
            return True
        return False
    
    def set_reduced_motion(self, enabled: bool) -> None:
        """
        Enable or disable reduced motion mode.
        
        Args:
            enabled: Whether to enable reduced motion
        """
        self.reduced_motion_enabled = enabled
        logger.info(f"Reduced motion {'enabled' if enabled else 'disabled'}")
    
    def get_active_animation_count(self) -> int:
        """Get the number of currently active animations."""
        return len(self.active_animations)
    
    async def _start_animation(self, animation_type: AnimationType, widget: toga.Widget,
                              config: AnimationConfig, completion_callback: Optional[Callable] = None,
                              extra_params: Optional[Dict] = None) -> str:
        """
        Start a new animation.
        
        Args:
            animation_type: Type of animation
            widget: Widget to animate
            config: Animation configuration
            completion_callback: Callback when animation completes
            extra_params: Additional parameters for the animation
            
        Returns:
            Animation ID
        """
        animation_id = str(uuid.uuid4())
        
        # Create animation state
        animation_state = AnimationState(
            animation_id=animation_id,
            animation_type=animation_type,
            target_widget=widget,
            config=config,
            completion_callback=completion_callback
        )
        
        # Store extra parameters
        if extra_params:
            for key, value in extra_params.items():
                setattr(animation_state, key, value)
        
        # Add to active animations
        self.active_animations[animation_id] = animation_state
        
        # Start animation loop if not running
        if not self.animation_loop_running:
            asyncio.create_task(self._animation_loop())
        
        # Record animation start for performance monitoring
        if self.performance_monitor:
            self.performance_monitor.record_animation_start(animation_id, animation_type.value)
        
        logger.debug(f"Started {animation_type.value} animation {animation_id}")
        return animation_id
    
    async def _animation_loop(self) -> None:
        """Main animation loop that updates all active animations."""
        if self.animation_loop_running:
            return
            
        self.animation_loop_running = True
        logger.debug("Animation loop started")
        
        try:
            while self.active_animations:
                frame_start = time.time()
                
                # Update all active animations
                completed_animations = []
                for animation_id, animation_state in self.active_animations.items():
                    if await self._update_animation(animation_state):
                        completed_animations.append(animation_id)
                
                # Remove completed animations
                for animation_id in completed_animations:
                    animation_state = self.active_animations.pop(animation_id, None)
                    if animation_state and animation_state.completion_callback:
                        try:
                            animation_state.completion_callback()
                        except Exception as e:
                            logger.error(f"Error in animation completion callback: {e}")
                    
                    # Record animation end for performance monitoring
                    if self.performance_monitor:
                        self.performance_monitor.record_animation_end(animation_id, completed=True)
                
                # Calculate frame time and sleep
                frame_time = (time.time() - frame_start) * 1000  # Convert to ms
                if self.performance_monitor:
                    self.performance_monitor.record_frame_time(frame_time)
                
                # Sleep for remaining frame time
                sleep_time = max(0, self.frame_time - (time.time() - frame_start))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Error in animation loop: {e}")
        finally:
            self.animation_loop_running = False
            logger.debug("Animation loop stopped")
    
    async def _update_animation(self, animation_state: AnimationState) -> bool:
        """
        Update a single animation.
        
        Args:
            animation_state: The animation state to update
            
        Returns:
            True if animation is complete, False otherwise
        """
        if not animation_state.is_running or animation_state.is_paused:
            return False
        
        # Update progress
        progress = animation_state.update_progress()
        
        # Apply easing
        eased_progress = apply_easing(progress, animation_state.config.easing)
        
        # Apply animation based on type
        try:
            await self._apply_animation_frame(animation_state, eased_progress)
        except Exception as e:
            logger.error(f"Error applying animation frame: {e}")
            animation_state.is_running = False
            return True
        
        return animation_state.is_complete
    
    async def _apply_animation_frame(self, animation_state: AnimationState, progress: float) -> None:
        """
        Apply a single frame of animation.
        
        Args:
            animation_state: The animation state
            progress: Current progress (0.0 to 1.0, eased)
        """
        widget = animation_state.target_widget
        animation_type = animation_state.animation_type
        
        # Note: Toga has limited animation support, so we'll simulate what we can
        # In a real implementation, this would use platform-specific animation APIs
        
        if animation_type == AnimationType.FADE_IN:
            # Simulate fade by adjusting visibility or style
            # Toga doesn't support opacity, so we'll use a workaround
            pass  # Placeholder for fade implementation
            
        elif animation_type == AnimationType.FADE_OUT:
            # Simulate fade out
            pass  # Placeholder for fade implementation
            
        elif animation_type == AnimationType.SLIDE_IN:
            # Simulate slide by adjusting margins or position
            direction = getattr(animation_state, 'direction', 'bottom')
            # Implementation would adjust widget position based on direction and progress
            pass  # Placeholder for slide implementation
            
        elif animation_type == AnimationType.SLIDE_OUT:
            # Simulate slide out
            direction = getattr(animation_state, 'direction', 'top')
            pass  # Placeholder for slide implementation
            
        elif animation_type == AnimationType.SCALE_IN:
            # Simulate scale by adjusting size
            from_scale = getattr(animation_state, 'from_scale', 0.8)
            current_scale = from_scale + (1.0 - from_scale) * progress
            # Implementation would adjust widget size based on scale
            pass  # Placeholder for scale implementation
            
        elif animation_type == AnimationType.SCALE_OUT:
            # Simulate scale out
            to_scale = getattr(animation_state, 'to_scale', 0.8)
            current_scale = 1.0 - (1.0 - to_scale) * progress
            pass  # Placeholder for scale implementation
    
    def _get_animation_config(self, animation_type: AnimationType, duration: float,
                             easing: str, delay: float) -> AnimationConfig:
        """
        Get animation configuration, applying performance optimizations if needed.
        
        Args:
            animation_type: Type of animation
            duration: Requested duration
            easing: Requested easing
            delay: Requested delay
            
        Returns:
            Optimized animation configuration
        """
        # Get base configuration
        if self.reduced_motion_enabled:
            base_config = REDUCED_MOTION_ANIMATIONS.get(animation_type, DEFAULT_ANIMATIONS[animation_type])
        else:
            base_config = DEFAULT_ANIMATIONS.get(animation_type, DEFAULT_ANIMATIONS[AnimationType.FADE_IN])
        
        # Create custom configuration
        config = AnimationConfig(
            duration=duration if duration > 0 else base_config.duration,
            easing=base_config.easing,  # Use easing from config for now
            delay=delay,
            reduced_motion=self.reduced_motion_enabled,
            performance_mode=base_config.performance_mode
        )
        
        # Apply performance optimizations
        if self.performance_monitor:
            config = self.performance_monitor.get_optimized_config(config)
        
        return config