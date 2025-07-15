"""
Micro-Interaction Manager - Coordinates Subtle UI Feedback

This module provides the MicroInteractionManager class that handles all the subtle
micro-interactions throughout the UI, including hover effects, button press animations,
focus indicators, and other visual feedback that makes the interface feel responsive.
"""

import asyncio
from typing import Dict, Set, Optional, Callable, Any
from loguru import logger

import toga
from toga.style import Pack

from .animation_manager import AnimationManager
from .transition_engine import TransitionEngine
from ..themes import theme_manager


class MicroInteractionManager:
    """Manages subtle micro-interactions throughout the UI."""
    
    def __init__(self, animation_manager: AnimationManager):
        """
        Initialize the micro-interaction manager.
        
        Args:
            animation_manager: The animation manager to coordinate with
        """
        self.animation_manager = animation_manager
        self.transition_engine = animation_manager.transition_engine
        
        # Track interactive elements and their states
        self.interactive_elements: Dict[str, Dict] = {}
        self.hover_states: Dict[str, bool] = {}
        self.focus_states: Dict[str, bool] = {}
        self.press_states: Dict[str, bool] = {}
        
        # Animation tasks for cleanup
        self.active_animations: Dict[str, asyncio.Task] = {}
        
        logger.info("Micro-interaction manager initialized")
    
    def setup_button_interactions(self, button: toga.Button, 
                                button_id: Optional[str] = None,
                                interaction_config: Optional[Dict] = None) -> str:
        """
        Set up comprehensive interactions for a button.
        
        Args:
            button: The button widget to enhance
            button_id: Unique identifier for the button
            interaction_config: Custom interaction configuration
            
        Returns:
            The button ID for tracking
        """
        if button_id is None:
            button_id = f"button_{id(button)}"
        
        # Default interaction configuration
        default_config = {
            "hover_scale": 1.02,
            "press_scale": 0.98,
            "hover_duration": 0.15,
            "press_duration": 0.1,
            "spring_back_duration": 0.15,
            "color_transition_duration": 0.2,
            "enable_hover": True,
            "enable_press": True,
            "enable_focus": True,
            "enable_spring_back": True
        }
        
        # Merge with custom config
        config = {**default_config, **(interaction_config or {})}
        
        # Store element info
        self.interactive_elements[button_id] = {
            "widget": button,
            "type": "button",
            "config": config,
            "original_style": self._capture_style_state(button),
            "current_state": "normal"
        }
        
        # Set up event handlers (simulated since Toga has limited event support)
        original_on_press = button.on_press
        button.on_press = lambda widget: asyncio.create_task(
            self._handle_button_press(button_id, original_on_press, widget)
        )
        
        logger.debug(f"Set up button interactions for: {button_id}")
        return button_id
    
    def setup_input_interactions(self, input_widget: toga.TextInput,
                               input_id: Optional[str] = None,
                               interaction_config: Optional[Dict] = None) -> str:
        """
        Set up interactions for an input field.
        
        Args:
            input_widget: The input widget to enhance
            input_id: Unique identifier for the input
            interaction_config: Custom interaction configuration
            
        Returns:
            The input ID for tracking
        """
        if input_id is None:
            input_id = f"input_{id(input_widget)}"
        
        # Default input interaction configuration
        default_config = {
            "focus_scale": 1.01,
            "focus_duration": 0.25,
            "unfocus_duration": 0.2,
            "border_highlight_duration": 0.3,
            "enable_focus_animation": True,
            "enable_border_highlight": True,
            "enable_subtle_glow": True
        }
        
        config = {**default_config, **(interaction_config or {})}
        
        # Store element info
        self.interactive_elements[input_id] = {
            "widget": input_widget,
            "type": "input",
            "config": config,
            "original_style": self._capture_style_state(input_widget),
            "current_state": "normal"
        }
        
        logger.debug(f"Set up input interactions for: {input_id}")
        return input_id
    
    def setup_hover_effects(self, widget: toga.Widget,
                          widget_id: Optional[str] = None,
                          hover_config: Optional[Dict] = None) -> str:
        """
        Set up hover effects for any widget.
        
        Args:
            widget: The widget to add hover effects to
            widget_id: Unique identifier for the widget
            hover_config: Custom hover configuration
            
        Returns:
            The widget ID for tracking
        """
        if widget_id is None:
            widget_id = f"hover_{id(widget)}"
        
        # Default hover configuration
        default_config = {
            "hover_scale": 1.02,
            "hover_duration": 0.15,
            "color_transition": True,
            "subtle_shadow": True,
            "brightness_boost": 1.05
        }
        
        config = {**default_config, **(hover_config or {})}
        
        # Store element info
        self.interactive_elements[widget_id] = {
            "widget": widget,
            "type": "hoverable",
            "config": config,
            "original_style": self._capture_style_state(widget),
            "current_state": "normal"
        }
        
        logger.debug(f"Set up hover effects for: {widget_id}")
        return widget_id
    
    def setup_focus_indicators(self, widget: toga.Widget,
                             widget_id: Optional[str] = None,
                             focus_config: Optional[Dict] = None) -> str:
        """
        Set up clear focus indicators for accessibility.
        
        Args:
            widget: The widget to add focus indicators to
            widget_id: Unique identifier for the widget
            focus_config: Custom focus configuration
            
        Returns:
            The widget ID for tracking
        """
        if widget_id is None:
            widget_id = f"focus_{id(widget)}"
        
        # Default focus configuration
        default_config = {
            "focus_outline_width": 2,
            "focus_outline_color": "#007AFF",  # System blue
            "focus_scale": 1.01,
            "focus_duration": 0.2,
            "focus_glow": True,
            "high_contrast_mode": False
        }
        
        config = {**default_config, **(focus_config or {})}
        
        # Store element info
        self.interactive_elements[widget_id] = {
            "widget": widget,
            "type": "focusable",
            "config": config,
            "original_style": self._capture_style_state(widget),
            "current_state": "normal"
        }
        
        logger.debug(f"Set up focus indicators for: {widget_id}")
        return widget_id
    
    async def animate_button_press(self, button_id: str) -> None:
        """
        Animate a button press with immediate visual feedback and spring-back.
        
        Args:
            button_id: ID of the button to animate
        """
        if button_id not in self.interactive_elements:
            logger.warning(f"Button not found for animation: {button_id}")
            return
        
        element = self.interactive_elements[button_id]
        button = element["widget"]
        config = element["config"]
        
        if not config.get("enable_press", True):
            return
        
        try:
            # Cancel any existing animation for this button
            await self._cancel_animation(button_id)
            
            # Phase 1: Immediate press down (scale down + slight color change)
            press_task = asyncio.create_task(
                self.transition_engine.scale_out(
                    button, 
                    to_scale=config["press_scale"],
                    duration=config["press_duration"],
                    easing="ease_out"
                )
            )
            
            # Apply press styling
            await self._apply_press_styling(button, element)
            
            await press_task
            
            # Phase 2: Spring back with enhanced effect
            if config.get("enable_spring_back", True):
                spring_task = asyncio.create_task(
                    self.transition_engine.scale_in(
                        button,
                        from_scale=config["press_scale"],
                        duration=config["spring_back_duration"],
                        easing="ease_out"
                    )
                )
                
                # Restore normal styling
                await self._restore_normal_styling(button, element)
                
                await spring_task
            
            element["current_state"] = "normal"
            
        except Exception as e:
            logger.error(f"Error animating button press for {button_id}: {e}")
    
    async def animate_hover_enter(self, widget_id: str) -> None:
        """
        Animate hover enter effect.
        
        Args:
            widget_id: ID of the widget being hovered
        """
        if widget_id not in self.interactive_elements:
            return
        
        element = self.interactive_elements[widget_id]
        widget = element["widget"]
        config = element["config"]
        
        if not config.get("enable_hover", True):
            return
        
        try:
            # Cancel any existing hover animation
            await self._cancel_animation(f"{widget_id}_hover")
            
            # Scale up slightly
            hover_task = asyncio.create_task(
                self.transition_engine.scale_in(
                    widget,
                    from_scale=1.0,
                    to_scale=config["hover_scale"],
                    duration=config["hover_duration"],
                    easing="ease_out"
                )
            )
            
            # Apply hover styling
            await self._apply_hover_styling(widget, element)
            
            self.active_animations[f"{widget_id}_hover"] = hover_task
            await hover_task
            
            element["current_state"] = "hover"
            self.hover_states[widget_id] = True
            
        except Exception as e:
            logger.error(f"Error animating hover enter for {widget_id}: {e}")
    
    async def animate_hover_exit(self, widget_id: str) -> None:
        """
        Animate hover exit effect.
        
        Args:
            widget_id: ID of the widget no longer being hovered
        """
        if widget_id not in self.interactive_elements:
            return
        
        element = self.interactive_elements[widget_id]
        widget = element["widget"]
        config = element["config"]
        
        try:
            # Cancel any existing hover animation
            await self._cancel_animation(f"{widget_id}_hover")
            
            # Scale back to normal
            unhover_task = asyncio.create_task(
                self.transition_engine.scale_out(
                    widget,
                    to_scale=1.0,
                    duration=config["hover_duration"],
                    easing="ease_in"
                )
            )
            
            # Restore normal styling
            await self._restore_normal_styling(widget, element)
            
            self.active_animations[f"{widget_id}_hover"] = unhover_task
            await unhover_task
            
            element["current_state"] = "normal"
            self.hover_states[widget_id] = False
            
        except Exception as e:
            logger.error(f"Error animating hover exit for {widget_id}: {e}")
    
    async def animate_focus_enter(self, widget_id: str) -> None:
        """
        Animate focus enter with clear visual indicators.
        
        Args:
            widget_id: ID of the widget receiving focus
        """
        if widget_id not in self.interactive_elements:
            return
        
        element = self.interactive_elements[widget_id]
        widget = element["widget"]
        config = element["config"]
        
        if not config.get("enable_focus_animation", True):
            return
        
        try:
            # Cancel any existing focus animation
            await self._cancel_animation(f"{widget_id}_focus")
            
            # Apply focus styling first
            await self._apply_focus_styling(widget, element)
            
            # Animate focus with scale and glow
            focus_task = asyncio.create_task(
                self.transition_engine.scale_in(
                    widget,
                    from_scale=1.0,
                    to_scale=config.get("focus_scale", 1.01),
                    duration=config.get("focus_duration", 0.2),
                    easing="ease_out"
                )
            )
            
            self.active_animations[f"{widget_id}_focus"] = focus_task
            await focus_task
            
            element["current_state"] = "focused"
            self.focus_states[widget_id] = True
            
        except Exception as e:
            logger.error(f"Error animating focus enter for {widget_id}: {e}")
    
    async def animate_focus_exit(self, widget_id: str) -> None:
        """
        Animate focus exit.
        
        Args:
            widget_id: ID of the widget losing focus
        """
        if widget_id not in self.interactive_elements:
            return
        
        element = self.interactive_elements[widget_id]
        widget = element["widget"]
        config = element["config"]
        
        try:
            # Cancel any existing focus animation
            await self._cancel_animation(f"{widget_id}_focus")
            
            # Scale back to normal
            unfocus_task = asyncio.create_task(
                self.transition_engine.scale_out(
                    widget,
                    to_scale=1.0,
                    duration=config.get("unfocus_duration", 0.2),
                    easing="ease_in"
                )
            )
            
            # Restore normal styling
            await self._restore_normal_styling(widget, element)
            
            self.active_animations[f"{widget_id}_focus"] = unfocus_task
            await unfocus_task
            
            element["current_state"] = "normal"
            self.focus_states[widget_id] = False
            
        except Exception as e:
            logger.error(f"Error animating focus exit for {widget_id}: {e}")
    
    async def pulse_attention(self, widget_id: str, pulse_count: int = 3) -> None:
        """
        Create a subtle attention-grabbing pulse animation.
        
        Args:
            widget_id: ID of the widget to pulse
            pulse_count: Number of pulses to perform
        """
        if widget_id not in self.interactive_elements:
            return
        
        element = self.interactive_elements[widget_id]
        widget = element["widget"]
        
        try:
            for _ in range(pulse_count):
                # Pulse out
                await self.transition_engine.scale_in(
                    widget, from_scale=1.0, to_scale=1.05, duration=0.3
                )
                
                # Pulse in
                await self.transition_engine.scale_out(
                    widget, to_scale=1.0, duration=0.3
                )
                
                # Small delay between pulses
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error pulsing widget {widget_id}: {e}")
    
    def cleanup_element(self, widget_id: str) -> None:
        """
        Clean up tracking for an element.
        
        Args:
            widget_id: ID of the element to clean up
        """
        if widget_id in self.interactive_elements:
            # Cancel any active animations (handle event loop gracefully)
            try:
                asyncio.create_task(self._cancel_animation(widget_id))
            except RuntimeError:
                # No event loop running, just clean up synchronously
                if widget_id in self.active_animations:
                    task = self.active_animations[widget_id]
                    if not task.done():
                        task.cancel()
                    del self.active_animations[widget_id]
            
            # Remove from tracking
            del self.interactive_elements[widget_id]
            self.hover_states.pop(widget_id, None)
            self.focus_states.pop(widget_id, None)
            self.press_states.pop(widget_id, None)
            
            logger.debug(f"Cleaned up element: {widget_id}")
    
    def cleanup_all(self) -> None:
        """Clean up all tracked elements."""
        for widget_id in list(self.interactive_elements.keys()):
            self.cleanup_element(widget_id)
        
        logger.info("Cleaned up all micro-interactions")
    
    async def _handle_button_press(self, button_id: str, 
                                 original_callback: Callable, 
                                 widget: toga.Button) -> None:
        """
        Handle button press with animation and callback.
        
        Args:
            button_id: ID of the button
            original_callback: Original button callback
            widget: The button widget
        """
        try:
            # Animate the press
            await self.animate_button_press(button_id)
            
            # Call the original callback
            if original_callback:
                if asyncio.iscoroutinefunction(original_callback):
                    await original_callback(widget)
                else:
                    original_callback(widget)
                    
        except Exception as e:
            logger.error(f"Error handling button press for {button_id}: {e}")
    
    def _capture_style_state(self, widget: toga.Widget) -> Dict:
        """
        Capture the current style state of a widget.
        
        Args:
            widget: Widget to capture style from
            
        Returns:
            Dictionary of style properties
        """
        try:
            style = widget.style
            return {
                "background_color": getattr(style, "background_color", None),
                "color": getattr(style, "color", None),
                "padding": getattr(style, "padding", None),
                "margin": getattr(style, "margin", None),
                # Add more style properties as needed
            }
        except Exception as e:
            logger.debug(f"Could not capture style state: {e}")
            return {}
    
    async def _apply_press_styling(self, widget: toga.Widget, element: Dict) -> None:
        """Apply styling for pressed state."""
        try:
            theme = theme_manager.get_theme()
            colors = theme.colors
            
            # Slightly darken the background for press effect
            if hasattr(widget.style, "background_color"):
                # This is a simplified approach - in a real implementation,
                # you'd want to darken the existing color
                widget.style.background_color = colors.surface_variant
                
        except Exception as e:
            logger.debug(f"Could not apply press styling: {e}")
    
    async def _apply_hover_styling(self, widget: toga.Widget, element: Dict) -> None:
        """Apply styling for hover state."""
        try:
            theme = theme_manager.get_theme()
            colors = theme.colors
            
            # Slightly brighten for hover effect
            if hasattr(widget.style, "background_color"):
                widget.style.background_color = colors.surface_bright
                
        except Exception as e:
            logger.debug(f"Could not apply hover styling: {e}")
    
    async def _apply_focus_styling(self, widget: toga.Widget, element: Dict) -> None:
        """Apply styling for focused state."""
        try:
            theme = theme_manager.get_theme()
            colors = theme.colors
            config = element["config"]
            
            # Apply focus outline effect (simulated with background)
            if hasattr(widget.style, "background_color"):
                widget.style.background_color = colors.primary_container
                
        except Exception as e:
            logger.debug(f"Could not apply focus styling: {e}")
    
    async def _restore_normal_styling(self, widget: toga.Widget, element: Dict) -> None:
        """Restore normal styling from captured state."""
        try:
            original_style = element["original_style"]
            
            # Restore original styling
            for prop, value in original_style.items():
                if value is not None and hasattr(widget.style, prop):
                    setattr(widget.style, prop, value)
                    
        except Exception as e:
            logger.debug(f"Could not restore normal styling: {e}")
    
    async def _cancel_animation(self, animation_key: str) -> None:
        """Cancel an active animation."""
        if animation_key in self.active_animations:
            task = self.active_animations[animation_key]
            if not task.done():
                task.cancel()
                try:
                    # Only await if it's actually a coroutine/task
                    if hasattr(task, '__await__') or asyncio.iscoroutine(task):
                        await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # Handle any other exceptions gracefully
                    pass
            del self.active_animations[animation_key]