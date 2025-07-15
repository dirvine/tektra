"""
Typing Indicator - Animated Visual Feedback for AI Processing

This module provides an animated typing indicator that shows when the AI
is processing responses, similar to modern messaging applications.
"""

import asyncio
import time
from typing import Optional, Callable
from loguru import logger

import toga
from toga.style import Pack
from toga.style.pack import ROW, COLUMN

from .themes import theme_manager
from .animations.animation_manager import AnimationManager
from .animations.animation_config import AnimationType


class TypingIndicator:
    """
    Animated typing indicator for AI responses.
    
    Provides visual feedback when the AI is processing, using smooth
    pulsing animations that match modern messaging app patterns.
    """
    
    def __init__(self, animation_manager: Optional[AnimationManager] = None):
        """
        Initialize the typing indicator.
        
        Args:
            animation_manager: Animation manager for controlling animations
        """
        self.animation_manager = animation_manager or AnimationManager()
        self.is_visible = False
        self.is_animating = False
        self.animation_task: Optional[asyncio.Task] = None
        self.animation_id: Optional[str] = None
        
        # Create the widget
        self.widget = self._create_indicator_widget()
        
        # Initially hidden - we'll manage visibility manually
        self._widget_visible = False
        
        logger.debug("TypingIndicator initialized")
    
    def _create_indicator_widget(self) -> toga.Box:
        """Create the typing indicator widget with modern styling."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        # Main container
        container = toga.Box(
            style=Pack(
                direction=ROW,
                padding=(spacing["sm"], spacing["md"]),
                background_color=colors.surface
            )
        )
        
        # AI avatar/icon
        avatar = toga.Label(
            "🤖",
            style=Pack(
                font_size=16,
                margin_right=spacing["sm"]
            )
        )
        container.add(avatar)
        
        # Typing animation container
        typing_container = toga.Box(
            style=Pack(
                direction=ROW
            )
        )
        
        # Create three dots for the typing animation
        self.dots = []
        for i in range(3):
            dot = toga.Label(
                "●",
                style=Pack(
                    font_size=12,
                    color=colors.text_secondary,
                    margin=(0, 2)
                )
            )
            self.dots.append(dot)
            typing_container.add(dot)
        
        container.add(typing_container)
        
        # Status text
        self.status_label = toga.Label(
            "Tektra is thinking...",
            style=Pack(
                font_size=theme.typography["caption"]["size"],
                color=colors.text_secondary,
                padding_left=spacing["sm"]
            )
        )
        container.add(self.status_label)
        
        return container
    
    async def show(self, message: str = "Tektra is thinking...") -> None:
        """
        Show the typing indicator with animation.
        
        Args:
            message: Custom status message to display
        """
        # Always update the message, even if already visible
        self.status_label.text = message
        
        if self.is_visible:
            # Already visible, just update message and continue
            return
        
        self.is_visible = True
        
        # Make widget visible (Toga doesn't support visibility, so we manage it manually)
        self._widget_visible = True
        
        # Animate appearance
        try:
            self.animation_id = await self.animation_manager.animate_message_appearance(
                self.widget, role="system"
            )
        except Exception as e:
            logger.error(f"Error animating typing indicator appearance: {e}")
        
        # Start typing animation
        await self.start_typing_animation()
        
        logger.debug(f"Typing indicator shown with message: {message}")
    
    async def hide(self) -> None:
        """Hide the typing indicator with animation."""
        if not self.is_visible:
            return
        
        self.is_visible = False
        
        # Stop typing animation
        await self.stop_typing_animation()
        
        # Animate disappearance
        try:
            fade_id = await self.animation_manager.transition_engine.fade_out(
                self.widget, duration=0.2
            )
            
            # Wait for fade to complete, then hide
            await asyncio.sleep(0.2)
            self._widget_visible = False
            
        except Exception as e:
            logger.error(f"Error animating typing indicator disappearance: {e}")
            # Fallback: immediately hide
            self._widget_visible = False
        
        logger.debug("Typing indicator hidden")
    
    async def start_typing_animation(self) -> None:
        """Start the continuous typing animation."""
        if self.is_animating:
            return
        
        self.is_animating = True
        
        # Start the animation loop
        self.animation_task = asyncio.create_task(self._typing_animation_loop())
        
        logger.debug("Typing animation started")
    
    async def stop_typing_animation(self) -> None:
        """Stop the typing animation."""
        if not self.is_animating:
            return
        
        self.is_animating = False
        
        # Cancel animation task
        if self.animation_task and not self.animation_task.done():
            self.animation_task.cancel()
            try:
                await self.animation_task
            except asyncio.CancelledError:
                pass
        
        # Reset dot colors
        theme = theme_manager.get_theme()
        for dot in self.dots:
            dot.style.color = theme.colors.text_secondary
        
        logger.debug("Typing animation stopped")
    
    async def _typing_animation_loop(self) -> None:
        """
        Main typing animation loop with wave effect.
        
        Creates a wave-like pulsing effect across the three dots,
        similar to modern messaging applications.
        """
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        # Animation timing
        wave_duration = 1.2  # Total wave cycle duration
        dot_delay = 0.2      # Delay between each dot
        pulse_duration = 0.4 # How long each dot stays highlighted
        
        try:
            while self.is_animating:
                cycle_start = time.time()
                
                # Animate each dot in sequence
                for i, dot in enumerate(self.dots):
                    if not self.is_animating:
                        break
                    
                    # Calculate timing for this dot
                    dot_start_time = i * dot_delay
                    
                    # Wait for this dot's turn
                    elapsed = time.time() - cycle_start
                    if elapsed < dot_start_time:
                        await asyncio.sleep(dot_start_time - elapsed)
                    
                    if not self.is_animating:
                        break
                    
                    # Pulse this dot
                    asyncio.create_task(self._pulse_dot(dot, pulse_duration))
                
                # Wait for the full wave cycle to complete
                elapsed = time.time() - cycle_start
                remaining = wave_duration - elapsed
                if remaining > 0:
                    await asyncio.sleep(remaining)
                
        except asyncio.CancelledError:
            logger.debug("Typing animation loop cancelled")
        except Exception as e:
            logger.error(f"Error in typing animation loop: {e}")
    
    async def _pulse_dot(self, dot: toga.Label, duration: float) -> None:
        """
        Pulse a single dot with color animation.
        
        Args:
            dot: The dot label to animate
            duration: Duration of the pulse effect
        """
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        try:
            # Highlight the dot
            dot.style.color = colors.primary
            
            # Wait for pulse duration
            await asyncio.sleep(duration)
            
            # Return to normal color
            if self.is_animating:  # Only if still animating
                dot.style.color = colors.text_secondary
                
        except Exception as e:
            logger.error(f"Error pulsing dot: {e}")
    
    def update_message(self, message: str) -> None:
        """
        Update the status message without restarting animation.
        
        Args:
            message: New status message
        """
        self.status_label.text = message
        logger.debug(f"Typing indicator message updated: {message}")
    
    def set_theme(self, theme_name: str) -> None:
        """
        Update the typing indicator for a new theme.
        
        Args:
            theme_name: Name of the new theme
        """
        # This would be called when theme changes
        # For now, we'll recreate the styling
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        # Update container styling
        self.widget.style.background_color = colors.surface
        self.widget.style.border = f"{theme.borders['width']}px solid {colors.border}"
        
        # Update status label color
        self.status_label.style.color = colors.text_secondary
        
        # Update dot colors if not animating
        if not self.is_animating:
            for dot in self.dots:
                dot.style.color = colors.text_secondary
        
        logger.debug(f"Typing indicator updated for theme: {theme_name}")
    
    @property
    def visible(self) -> bool:
        """Check if the typing indicator is currently visible."""
        return self.is_visible
    
    @property
    def animating(self) -> bool:
        """Check if the typing animation is currently running."""
        return self.is_animating


class TypingIndicatorManager:
    """
    Manager for coordinating typing indicators across the chat interface.
    
    Handles timing, queueing, and integration with the chat flow.
    """
    
    def __init__(self, animation_manager: Optional[AnimationManager] = None):
        """
        Initialize the typing indicator manager.
        
        Args:
            animation_manager: Animation manager for controlling animations
        """
        self.animation_manager = animation_manager or AnimationManager()
        self.indicators: dict[str, TypingIndicator] = {}
        self.active_indicator: Optional[str] = None
        self.show_timeout: Optional[asyncio.Task] = None
        self.hide_timeout: Optional[asyncio.Task] = None
        
        logger.debug("TypingIndicatorManager initialized")
    
    def create_indicator(self, indicator_id: str = "default") -> TypingIndicator:
        """
        Create a new typing indicator.
        
        Args:
            indicator_id: Unique identifier for the indicator
            
        Returns:
            The created typing indicator
        """
        indicator = TypingIndicator(self.animation_manager)
        self.indicators[indicator_id] = indicator
        
        logger.debug(f"Created typing indicator: {indicator_id}")
        return indicator
    
    async def show_indicator(self, indicator_id: str = "default", 
                           message: str = "Tektra is thinking...",
                           delay: float = 0.0) -> None:
        """
        Show a typing indicator with optional delay.
        
        Args:
            indicator_id: ID of the indicator to show
            message: Status message to display
            delay: Delay before showing (useful for preventing flicker)
        """
        # Cancel any pending hide timeout
        if self.hide_timeout and not self.hide_timeout.done():
            self.hide_timeout.cancel()
        
        # If there's a delay, schedule the show and wait for it
        if delay > 0:
            self.show_timeout = asyncio.create_task(
                self._delayed_show(indicator_id, message, delay)
            )
            await self.show_timeout
        else:
            await self._show_indicator_now(indicator_id, message)
    
    async def hide_indicator(self, indicator_id: str = "default",
                           delay: float = 0.0) -> None:
        """
        Hide a typing indicator with optional delay.
        
        Args:
            indicator_id: ID of the indicator to hide
            delay: Delay before hiding
        """
        # Cancel any pending show timeout
        if self.show_timeout and not self.show_timeout.done():
            self.show_timeout.cancel()
        
        # If there's a delay, schedule the hide and wait for it
        if delay > 0:
            self.hide_timeout = asyncio.create_task(
                self._delayed_hide(indicator_id, delay)
            )
            await self.hide_timeout
        else:
            await self._hide_indicator_now(indicator_id)
    
    async def _delayed_show(self, indicator_id: str, message: str, delay: float) -> None:
        """Show indicator after delay."""
        try:
            await asyncio.sleep(delay)
            await self._show_indicator_now(indicator_id, message)
        except asyncio.CancelledError:
            logger.debug(f"Delayed show cancelled for indicator: {indicator_id}")
    
    async def _delayed_hide(self, indicator_id: str, delay: float) -> None:
        """Hide indicator after delay."""
        try:
            await asyncio.sleep(delay)
            await self._hide_indicator_now(indicator_id)
        except asyncio.CancelledError:
            logger.debug(f"Delayed hide cancelled for indicator: {indicator_id}")
    
    async def _show_indicator_now(self, indicator_id: str, message: str) -> None:
        """Show indicator immediately."""
        if indicator_id not in self.indicators:
            self.create_indicator(indicator_id)
        
        indicator = self.indicators[indicator_id]
        await indicator.show(message)
        self.active_indicator = indicator_id
        
        logger.debug(f"Showed typing indicator: {indicator_id}")
    
    async def _hide_indicator_now(self, indicator_id: str) -> None:
        """Hide indicator immediately."""
        if indicator_id in self.indicators:
            indicator = self.indicators[indicator_id]
            await indicator.hide()
            
            if self.active_indicator == indicator_id:
                self.active_indicator = None
        
        logger.debug(f"Hid typing indicator: {indicator_id}")
    
    def get_indicator(self, indicator_id: str = "default") -> Optional[TypingIndicator]:
        """
        Get a typing indicator by ID.
        
        Args:
            indicator_id: ID of the indicator
            
        Returns:
            The typing indicator or None if not found
        """
        return self.indicators.get(indicator_id)
    
    def is_any_visible(self) -> bool:
        """Check if any typing indicator is currently visible."""
        return any(indicator.visible for indicator in self.indicators.values())
    
    async def hide_all(self) -> None:
        """Hide all typing indicators."""
        for indicator_id in list(self.indicators.keys()):
            await self.hide_indicator(indicator_id)
        
        logger.debug("All typing indicators hidden")
    
    def cleanup(self) -> None:
        """Clean up all typing indicators and cancel pending tasks."""
        # Cancel pending tasks
        if self.show_timeout and not self.show_timeout.done():
            self.show_timeout.cancel()
        if self.hide_timeout and not self.hide_timeout.done():
            self.hide_timeout.cancel()
        
        # Stop all animations
        for indicator in self.indicators.values():
            if indicator.is_animating:
                asyncio.create_task(indicator.stop_typing_animation())
        
        self.indicators.clear()
        self.active_indicator = None
        
        logger.debug("TypingIndicatorManager cleaned up")