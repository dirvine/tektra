"""
Smooth Scroll Container - Enhanced Scrolling with Momentum and Animations

This module provides the SmoothScrollContainer class that enhances the standard
Toga ScrollContainer with smooth scrolling, momentum, and auto-scroll capabilities.
"""

import asyncio
import time
from typing import Optional, Callable, Dict, Any
from loguru import logger

import toga
from toga.style import Pack

from .animations.animation_manager import AnimationManager


class SmoothScrollContainer:
    """
    Enhanced scroll container with smooth scrolling and momentum.
    
    Provides smooth auto-scroll to new messages, momentum scrolling,
    and performance optimizations for large content.
    """
    
    def __init__(
        self,
        content: toga.Widget,
        animation_manager: AnimationManager,
        style: Optional[Pack] = None,
        enable_momentum: bool = True,
        momentum_decay: float = 0.95,
        auto_scroll_threshold: float = 50.0
    ):
        """
        Initialize the smooth scroll container.
        
        Args:
            content: The content widget to scroll
            animation_manager: Animation manager for smooth effects
            style: Optional style for the container
            enable_momentum: Whether to enable momentum scrolling
            momentum_decay: Momentum decay factor (0.0 to 1.0)
            auto_scroll_threshold: Distance from bottom to trigger auto-scroll
        """
        self.content = content
        self.animation_manager = animation_manager
        self.enable_momentum = enable_momentum
        self.momentum_decay = momentum_decay
        self.auto_scroll_threshold = auto_scroll_threshold
        
        # Scroll state
        self.scroll_position = 0.0
        self.target_scroll_position = 0.0
        self.scroll_velocity = 0.0
        self.is_scrolling = False
        self.is_auto_scrolling = False
        self.last_scroll_time = 0.0
        
        # Content dimensions
        self.content_height = 0
        self.container_height = 600  # Default height
        self.max_scroll_position = 0
        
        # Auto-scroll behavior
        self.auto_scroll_enabled = True
        self.was_at_bottom = True
        self.scroll_to_bottom_on_new_content = True
        
        # Performance tracking
        self.frame_times = []
        self.performance_mode = "balanced"  # "performance", "balanced", "quality"
        
        # Create the actual scroll container (only if content has _impl attribute)
        try:
            self.scroll_container = toga.ScrollContainer(
                content=content,
                style=style or Pack(flex=1)
            )
        except (AttributeError, TypeError):
            # Handle case where content is a mock or doesn't have proper Toga implementation
            self.scroll_container = None
            logger.debug("Could not create real scroll container, using mock mode")
        
        # Start the smooth scrolling loop (only if event loop is available)
        try:
            self._scroll_task = asyncio.create_task(self._smooth_scroll_loop())
        except RuntimeError:
            # No event loop running, scrolling loop will start later
            self._scroll_task = None
            logger.debug("No event loop running, smooth scrolling will start later")
        
        logger.info("SmoothScrollContainer initialized with momentum scrolling")
    
    @property
    def widget(self) -> Optional[toga.ScrollContainer]:
        """Get the underlying scroll container widget."""
        return self.scroll_container
    
    def update_content_dimensions(self, content_height: int, container_height: int) -> None:
        """
        Update the content and container dimensions.
        
        Args:
            content_height: Total height of the content
            container_height: Height of the visible container
        """
        self.content_height = content_height
        self.container_height = container_height
        self.max_scroll_position = max(0, content_height - container_height)
        
        # Check if we were at the bottom before the update
        was_at_bottom = self._is_at_bottom()
        
        # If we were at the bottom and auto-scroll is enabled, stay at the bottom
        if was_at_bottom and self.auto_scroll_enabled:
            self.scroll_to_bottom(smooth=True)
    
    def scroll_to_bottom(self, smooth: bool = True, force: bool = False) -> None:
        """
        Scroll to the bottom of the content.
        
        Args:
            smooth: Whether to use smooth scrolling animation
            force: Force scroll even if auto-scroll is disabled
        """
        if not force and not self.auto_scroll_enabled:
            return
        
        target_position = self.max_scroll_position
        
        if smooth:
            self._animate_scroll_to(target_position)
        else:
            self.scroll_position = target_position
            self.target_scroll_position = target_position
            self._apply_scroll_position()
        
        logger.debug(f"Scrolling to bottom: {target_position}")
    
    def scroll_to_position(self, position: float, smooth: bool = True) -> None:
        """
        Scroll to a specific position.
        
        Args:
            position: Target scroll position (0.0 to max_scroll_position)
            smooth: Whether to use smooth scrolling animation
        """
        # Clamp position to valid range
        position = max(0, min(position, self.max_scroll_position))
        
        if smooth:
            self._animate_scroll_to(position)
        else:
            self.scroll_position = position
            self.target_scroll_position = position
            self._apply_scroll_position()
    
    def scroll_by_delta(self, delta: float, smooth: bool = True) -> None:
        """
        Scroll by a relative amount.
        
        Args:
            delta: Amount to scroll (positive = down, negative = up)
            smooth: Whether to use smooth scrolling animation
        """
        new_position = self.scroll_position + delta
        self.scroll_to_position(new_position, smooth)
    
    def add_momentum(self, velocity: float) -> None:
        """
        Add momentum to the scrolling.
        
        Args:
            velocity: Scroll velocity to add
        """
        if not self.enable_momentum:
            return
        
        self.scroll_velocity += velocity
        self.last_scroll_time = time.time()
        
        # Clamp velocity to reasonable limits
        max_velocity = 1000.0
        self.scroll_velocity = max(-max_velocity, min(max_velocity, self.scroll_velocity))
    
    def set_auto_scroll_enabled(self, enabled: bool) -> None:
        """
        Enable or disable auto-scroll to bottom on new content.
        
        Args:
            enabled: Whether to enable auto-scroll
        """
        self.auto_scroll_enabled = enabled
        logger.debug(f"Auto-scroll {'enabled' if enabled else 'disabled'}")
    
    def set_performance_mode(self, mode: str) -> None:
        """
        Set the performance mode for scrolling.
        
        Args:
            mode: Performance mode ("performance", "balanced", "quality")
        """
        if mode in ["performance", "balanced", "quality"]:
            self.performance_mode = mode
            logger.debug(f"Performance mode set to: {mode}")
    
    def get_scroll_info(self) -> Dict[str, Any]:
        """
        Get current scroll information.
        
        Returns:
            Dictionary with scroll state information
        """
        return {
            "scroll_position": self.scroll_position,
            "target_position": self.target_scroll_position,
            "velocity": self.scroll_velocity,
            "is_scrolling": self.is_scrolling,
            "is_auto_scrolling": self.is_auto_scrolling,
            "is_at_bottom": self._is_at_bottom(),
            "content_height": self.content_height,
            "container_height": self.container_height,
            "max_scroll_position": self.max_scroll_position,
            "scroll_percentage": self._get_scroll_percentage()
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the scroll container.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_frame_time = 0
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        return {
            "average_frame_time": avg_frame_time,
            "target_fps": 60,
            "actual_fps": 1000 / avg_frame_time if avg_frame_time > 0 else 0,
            "performance_mode": self.performance_mode,
            "momentum_enabled": self.enable_momentum,
            "frame_samples": len(self.frame_times)
        }
    
    def cleanup(self) -> None:
        """Clean up resources and stop the scroll loop."""
        if self._scroll_task and not self._scroll_task.done():
            self._scroll_task.cancel()
        logger.debug("SmoothScrollContainer cleaned up")
    
    def _animate_scroll_to(self, target_position: float, duration: float = 0.5) -> None:
        """
        Animate scrolling to a target position.
        
        Args:
            target_position: Target scroll position
            duration: Animation duration in seconds
        """
        self.target_scroll_position = max(0, min(target_position, self.max_scroll_position))
        self.is_auto_scrolling = True
        
        # Calculate velocity needed to reach target in the given duration
        distance = self.target_scroll_position - self.scroll_position
        self.scroll_velocity = distance / duration * 60  # Assuming 60fps
        
        logger.debug(f"Animating scroll to {target_position} over {duration}s")
    
    def _is_at_bottom(self, threshold: Optional[float] = None) -> bool:
        """
        Check if the scroll position is at or near the bottom.
        
        Args:
            threshold: Distance from bottom to consider "at bottom"
            
        Returns:
            True if at or near the bottom
        """
        threshold = threshold or self.auto_scroll_threshold
        return (self.max_scroll_position - self.scroll_position) <= threshold
    
    def _get_scroll_percentage(self) -> float:
        """
        Get the current scroll position as a percentage.
        
        Returns:
            Scroll percentage (0.0 to 1.0)
        """
        if self.max_scroll_position <= 0:
            return 0.0
        return self.scroll_position / self.max_scroll_position
    
    def _apply_scroll_position(self) -> None:
        """Apply the current scroll position to the actual scroll container."""
        # Note: Toga doesn't have direct scroll position control
        # This is a placeholder for future implementation or platform-specific code
        pass
    
    async def _smooth_scroll_loop(self) -> None:
        """Main smooth scrolling loop that runs at ~60fps."""
        try:
            target_frame_time = 1.0 / 60.0  # 60fps target
            
            while True:
                frame_start = time.time()
                
                # Update scroll physics
                self._update_scroll_physics()
                
                # Apply scroll position
                self._apply_scroll_position()
                
                # Calculate frame time
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time * 1000)  # Convert to ms
                
                # Keep only recent frame times for performance calculation
                if len(self.frame_times) > 60:
                    self.frame_times.pop(0)
                
                # Adjust sleep time based on performance mode
                sleep_time = self._calculate_sleep_time(frame_time, target_frame_time)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.debug("Smooth scroll loop cancelled")
        except Exception as e:
            logger.error(f"Error in smooth scroll loop: {e}")
    
    def _update_scroll_physics(self) -> None:
        """Update scroll position based on physics and momentum."""
        current_time = time.time()
        
        # Apply momentum decay
        if self.enable_momentum and abs(self.scroll_velocity) > 0.1:
            self.scroll_velocity *= self.momentum_decay
            
            # Stop very small velocities to prevent infinite scrolling
            if abs(self.scroll_velocity) < 0.1:
                self.scroll_velocity = 0
        
        # Update scroll position based on velocity
        if abs(self.scroll_velocity) > 0.1:
            self.scroll_position += self.scroll_velocity / 60.0  # Assuming 60fps
            self.is_scrolling = True
        else:
            self.is_scrolling = False
            self.is_auto_scrolling = False
        
        # Clamp scroll position to valid range
        old_position = self.scroll_position
        self.scroll_position = max(0, min(self.scroll_position, self.max_scroll_position))
        
        # If we hit a boundary, stop the velocity
        if self.scroll_position != old_position:
            self.scroll_velocity = 0
        
        # Update bottom tracking
        self.was_at_bottom = self._is_at_bottom()
    
    def _calculate_sleep_time(self, frame_time: float, target_frame_time: float) -> float:
        """
        Calculate sleep time based on performance mode and frame time.
        
        Args:
            frame_time: Actual frame processing time
            target_frame_time: Target frame time for 60fps
            
        Returns:
            Sleep time in seconds
        """
        if self.performance_mode == "performance":
            # Prioritize performance over smoothness
            return max(0.02, target_frame_time - frame_time)  # Max 50fps
        elif self.performance_mode == "quality":
            # Prioritize smoothness over performance
            return max(0.008, target_frame_time - frame_time)  # Target 120fps if possible
        else:  # balanced
            # Balance between performance and smoothness
            return max(0.016, target_frame_time - frame_time)  # Target 60fps


class ConversationScrollManager:
    """
    Specialized scroll manager for conversation interfaces.
    
    Handles conversation-specific scrolling behaviors like auto-scroll
    on new messages and smart scroll position management.
    """
    
    def __init__(
        self,
        smooth_scroll_container: SmoothScrollContainer,
        animation_manager: AnimationManager
    ):
        """
        Initialize the conversation scroll manager.
        
        Args:
            smooth_scroll_container: The smooth scroll container to manage
            animation_manager: Animation manager for effects
        """
        self.scroll_container = smooth_scroll_container
        self.animation_manager = animation_manager
        
        # Conversation-specific settings
        self.auto_scroll_on_new_message = True
        self.auto_scroll_on_user_message = True
        self.auto_scroll_on_assistant_message = True
        self.scroll_to_new_message_delay = 0.1  # Delay before auto-scroll
        
        # Message tracking
        self.last_message_count = 0
        self.pending_scroll_to_bottom = False
        
        logger.info("ConversationScrollManager initialized")
    
    async def on_new_message(self, role: str, animate: bool = True) -> None:
        """
        Handle new message addition with appropriate scrolling behavior.
        
        Args:
            role: Role of the message sender ("user", "assistant", "system")
            animate: Whether to animate the scroll
        """
        should_auto_scroll = self._should_auto_scroll_for_role(role)
        
        if should_auto_scroll and self.scroll_container.auto_scroll_enabled:
            # Small delay to allow message rendering to complete
            await asyncio.sleep(self.scroll_to_new_message_delay)
            
            # Scroll to bottom with animation
            if animate:
                await self._animate_scroll_to_new_message()
            else:
                self.scroll_container.scroll_to_bottom(smooth=False)
            
            logger.debug(f"Auto-scrolled to new {role} message")
    
    async def on_typing_indicator_shown(self) -> None:
        """Handle typing indicator being shown."""
        if self.scroll_container.auto_scroll_enabled:
            # Scroll to bottom to show typing indicator
            await asyncio.sleep(0.05)  # Small delay
            self.scroll_container.scroll_to_bottom(smooth=True)
    
    async def on_content_height_changed(self, new_height: int) -> None:
        """
        Handle content height changes.
        
        Args:
            new_height: New content height
        """
        old_height = self.scroll_container.content_height
        was_at_bottom = self.scroll_container._is_at_bottom()
        
        # Update dimensions
        self.scroll_container.update_content_dimensions(
            new_height, 
            self.scroll_container.container_height
        )
        
        # If we were at the bottom and content grew, stay at the bottom
        if was_at_bottom and new_height > old_height:
            await asyncio.sleep(0.05)  # Allow layout to settle
            self.scroll_container.scroll_to_bottom(smooth=True)
    
    def set_auto_scroll_preferences(
        self,
        on_user_message: bool = True,
        on_assistant_message: bool = True,
        on_system_message: bool = False
    ) -> None:
        """
        Set auto-scroll preferences for different message types.
        
        Args:
            on_user_message: Auto-scroll on user messages
            on_assistant_message: Auto-scroll on assistant messages
            on_system_message: Auto-scroll on system messages
        """
        self.auto_scroll_on_user_message = on_user_message
        self.auto_scroll_on_assistant_message = on_assistant_message
        self.auto_scroll_on_system_message = on_system_message
        
        logger.debug(f"Auto-scroll preferences updated: user={on_user_message}, "
                    f"assistant={on_assistant_message}, system={on_system_message}")
    
    def _should_auto_scroll_for_role(self, role: str) -> bool:
        """
        Determine if auto-scroll should happen for a given message role.
        
        Args:
            role: Message role
            
        Returns:
            True if should auto-scroll
        """
        if role == "user":
            return self.auto_scroll_on_user_message
        elif role == "assistant":
            return self.auto_scroll_on_assistant_message
        elif role == "system":
            return getattr(self, 'auto_scroll_on_system_message', False)
        else:
            return False
    
    async def _animate_scroll_to_new_message(self) -> None:
        """Animate scrolling to a new message with enhanced effects."""
        # Use the animation manager for enhanced scroll animation
        try:
            # First, scroll to bottom smoothly
            self.scroll_container.scroll_to_bottom(smooth=True)
            
            # Add a subtle bounce effect to indicate new content
            await asyncio.sleep(0.3)  # Wait for scroll to complete
            
            # Small bounce animation (simulated)
            await self._simulate_bounce_effect()
            
        except Exception as e:
            logger.error(f"Error animating scroll to new message: {e}")
            # Fallback to simple scroll
            self.scroll_container.scroll_to_bottom(smooth=False)
    
    async def _simulate_bounce_effect(self) -> None:
        """Simulate a subtle bounce effect at the bottom."""
        # This would be implemented with actual scroll position manipulation
        # For now, it's a placeholder that could trigger other visual effects
        logger.debug("Bounce effect simulated")
        await asyncio.sleep(0.1)