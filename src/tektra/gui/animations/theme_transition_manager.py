"""
Theme Transition Manager - Smooth Theme Switching with Animated Color Transitions

This module provides the ThemeTransitionManager class that coordinates smooth
theme changes across all UI components with animated color transitions.
"""

import asyncio
import platform
import subprocess
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from loguru import logger

import toga

from .animation_manager import AnimationManager
from .transition_engine import TransitionEngine
from ..themes import Theme, ThemeManager, theme_manager


@dataclass
class ColorTransition:
    """Represents a color transition between two values."""
    from_color: str
    to_color: str
    property_name: str
    widget: toga.Widget
    duration: float = 0.3


@dataclass
class ThemeTransitionState:
    """Tracks the state of a theme transition."""
    transition_id: str
    from_theme: str
    to_theme: str
    progress: float = 0.0
    is_active: bool = False
    start_time: float = 0.0
    color_transitions: List[ColorTransition] = None
    
    def __post_init__(self):
        if self.color_transitions is None:
            self.color_transitions = []


class SystemThemeDetector:
    """Detects system theme changes and preferences."""
    
    def __init__(self):
        """Initialize the system theme detector."""
        self.current_system_theme = self._detect_current_system_theme()
        self.theme_change_callbacks: List[Callable[[str], None]] = []
        self.monitoring_active = False
        
    def _detect_current_system_theme(self) -> str:
        """
        Detect the current system theme.
        
        Returns:
            "light", "dark", or "auto"
        """
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                result = subprocess.run([
                    "defaults", "read", "-g", "AppleInterfaceStyle"
                ], capture_output=True, text=True)
                
                if result.returncode == 0 and "Dark" in result.stdout:
                    return "dark"
                else:
                    return "light"
                    
            elif system == "windows":
                # Check Windows theme setting
                import winreg
                try:
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                       r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
                    value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                    winreg.CloseKey(key)
                    return "light" if value == 1 else "dark"
                except:
                    pass
                    
            elif system == "linux":
                # Check GTK theme setting
                try:
                    result = subprocess.run([
                        "gsettings", "get", "org.gnome.desktop.interface", "gtk-theme"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        theme_name = result.stdout.strip().lower()
                        if "dark" in theme_name:
                            return "dark"
                        else:
                            return "light"
                except:
                    pass
                    
        except Exception as e:
            logger.debug(f"Could not detect system theme: {e}")
        
        # Default to light theme if detection fails
        return "light"
    
    def add_theme_change_callback(self, callback: Callable[[str], None]) -> None:
        """
        Add a callback to be called when system theme changes.
        
        Args:
            callback: Function to call with new theme name
        """
        self.theme_change_callbacks.append(callback)
    
    def remove_theme_change_callback(self, callback: Callable[[str], None]) -> None:
        """
        Remove a theme change callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.theme_change_callbacks:
            self.theme_change_callbacks.remove(callback)
    
    async def start_monitoring(self) -> None:
        """Start monitoring for system theme changes."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        logger.info("Started system theme monitoring")
        
        try:
            while self.monitoring_active:
                new_theme = self._detect_current_system_theme()
                
                if new_theme != self.current_system_theme:
                    logger.info(f"System theme changed from {self.current_system_theme} to {new_theme}")
                    self.current_system_theme = new_theme
                    
                    # Notify all callbacks
                    for callback in self.theme_change_callbacks:
                        try:
                            callback(new_theme)
                        except Exception as e:
                            logger.error(f"Error in theme change callback: {e}")
                
                # Check every 2 seconds
                await asyncio.sleep(2.0)
                
        except asyncio.CancelledError:
            logger.info("System theme monitoring stopped")
        except Exception as e:
            logger.error(f"Error in system theme monitoring: {e}")
        finally:
            self.monitoring_active = False
    
    def stop_monitoring(self) -> None:
        """Stop monitoring for system theme changes."""
        self.monitoring_active = False


class ThemeTransitionManager:
    """Manages smooth theme transitions with animated color changes."""
    
    def __init__(self, animation_manager: AnimationManager, theme_manager: ThemeManager):
        """
        Initialize the theme transition manager.
        
        Args:
            animation_manager: The animation manager instance
            theme_manager: The theme manager instance
        """
        self.animation_manager = animation_manager
        self.theme_manager = theme_manager
        self.system_theme_detector = SystemThemeDetector()
        
        # Transition state
        self.active_transitions: Dict[str, ThemeTransitionState] = {}
        self.registered_widgets: List[toga.Widget] = []
        self.transition_callbacks: List[Callable[[str, str], None]] = []
        
        # Settings
        self.auto_system_theme = False
        self.transition_duration = 0.4
        self.accessibility_mode = False
        
        # Set up system theme monitoring
        self.system_theme_detector.add_theme_change_callback(self._on_system_theme_change)
        
        logger.info("Theme Transition Manager initialized")
    
    def register_widget(self, widget: toga.Widget) -> None:
        """
        Register a widget for theme transitions.
        
        Args:
            widget: Widget to include in theme transitions
        """
        if widget not in self.registered_widgets:
            self.registered_widgets.append(widget)
            logger.debug(f"Registered widget for theme transitions: {type(widget).__name__}")
    
    def unregister_widget(self, widget: toga.Widget) -> None:
        """
        Unregister a widget from theme transitions.
        
        Args:
            widget: Widget to remove from theme transitions
        """
        if widget in self.registered_widgets:
            self.registered_widgets.remove(widget)
            logger.debug(f"Unregistered widget from theme transitions: {type(widget).__name__}")
    
    def add_transition_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Add a callback to be called when theme transitions occur.
        
        Args:
            callback: Function to call with (from_theme, to_theme)
        """
        self.transition_callbacks.append(callback)
    
    def remove_transition_callback(self, callback: Callable[[str, str], None]) -> None:
        """
        Remove a theme transition callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.transition_callbacks:
            self.transition_callbacks.remove(callback)
    
    async def transition_to_theme(self, theme_name: str, 
                                 animated: bool = True,
                                 duration: Optional[float] = None) -> str:
        """
        Transition to a new theme with smooth animations.
        
        Args:
            theme_name: Name of the theme to transition to
            animated: Whether to animate the transition
            duration: Custom transition duration (uses default if None)
            
        Returns:
            Transition ID for tracking
        """
        if theme_name not in self.theme_manager.themes:
            raise ValueError(f"Unknown theme: {theme_name}")
        
        current_theme_name = self.theme_manager.current_theme_name
        if current_theme_name == theme_name:
            logger.debug(f"Already using theme: {theme_name}")
            return ""
        
        # Use custom duration or default
        transition_duration = duration if duration is not None else self.transition_duration
        
        # Create transition ID
        transition_id = f"theme_transition_{current_theme_name}_to_{theme_name}_{asyncio.get_event_loop().time()}"
        
        # Create transition state
        transition_state = ThemeTransitionState(
            transition_id=transition_id,
            from_theme=current_theme_name,
            to_theme=theme_name,
            start_time=asyncio.get_event_loop().time()
        )
        
        self.active_transitions[transition_id] = transition_state
        
        logger.info(f"Starting theme transition from {current_theme_name} to {theme_name}")
        
        try:
            if animated and not self.accessibility_mode:
                # Perform animated transition
                await self._perform_animated_transition(transition_state, transition_duration)
            else:
                # Perform instant transition
                await self._perform_instant_transition(transition_state)
            
            # Notify callbacks
            for callback in self.transition_callbacks:
                try:
                    callback(current_theme_name, theme_name)
                except Exception as e:
                    logger.error(f"Error in theme transition callback: {e}")
            
            logger.info(f"Theme transition completed: {current_theme_name} -> {theme_name}")
            
        except Exception as e:
            logger.error(f"Error during theme transition: {e}")
        finally:
            # Clean up transition state
            if transition_id in self.active_transitions:
                del self.active_transitions[transition_id]
        
        return transition_id
    
    async def _perform_animated_transition(self, transition_state: ThemeTransitionState,
                                         duration: float) -> None:
        """
        Perform an animated theme transition.
        
        Args:
            transition_state: The transition state
            duration: Transition duration in seconds
        """
        transition_state.is_active = True
        
        # Get themes
        from_theme = self.theme_manager.themes[transition_state.from_theme]
        to_theme = self.theme_manager.themes[transition_state.to_theme]
        
        # Phase 1: Fade out current theme (25% of duration)
        fade_out_duration = duration * 0.25
        fade_out_tasks = []
        
        for widget in self.registered_widgets:
            if hasattr(widget, 'style') and widget.style:
                task = asyncio.create_task(
                    self.animation_manager.transition_engine.fade_out(
                        widget, duration=fade_out_duration
                    )
                )
                fade_out_tasks.append(task)
        
        # Wait for fade out to complete
        if fade_out_tasks:
            await asyncio.gather(*fade_out_tasks, return_exceptions=True)
        
        # Phase 2: Apply new theme (instant)
        self.theme_manager.switch_theme(transition_state.to_theme)
        await self._apply_theme_to_widgets(to_theme)
        
        # Phase 3: Fade in new theme (75% of duration)
        fade_in_duration = duration * 0.75
        fade_in_tasks = []
        
        for widget in self.registered_widgets:
            if hasattr(widget, 'style') and widget.style:
                task = asyncio.create_task(
                    self.animation_manager.transition_engine.fade_in(
                        widget, duration=fade_in_duration
                    )
                )
                fade_in_tasks.append(task)
        
        # Wait for fade in to complete
        if fade_in_tasks:
            await asyncio.gather(*fade_in_tasks, return_exceptions=True)
        
        transition_state.is_active = False
        transition_state.progress = 1.0
    
    async def _perform_instant_transition(self, transition_state: ThemeTransitionState) -> None:
        """
        Perform an instant theme transition.
        
        Args:
            transition_state: The transition state
        """
        # Get new theme
        to_theme = self.theme_manager.themes[transition_state.to_theme]
        
        # Apply theme immediately
        self.theme_manager.switch_theme(transition_state.to_theme)
        await self._apply_theme_to_widgets(to_theme)
        
        transition_state.progress = 1.0
    
    async def _apply_theme_to_widgets(self, theme: Theme) -> None:
        """
        Apply theme colors to all registered widgets.
        
        Args:
            theme: Theme to apply
        """
        colors = theme.colors
        
        for widget in self.registered_widgets:
            try:
                if hasattr(widget, 'style') and widget.style:
                    # Apply background color based on widget type
                    if isinstance(widget, toga.Box):
                        if hasattr(widget.style, 'background_color'):
                            widget.style.background_color = colors.background
                    elif isinstance(widget, toga.ScrollContainer):
                        if hasattr(widget.style, 'background_color'):
                            widget.style.background_color = colors.surface
                    elif isinstance(widget, toga.Button):
                        if hasattr(widget.style, 'background_color'):
                            widget.style.background_color = colors.primary
                        if hasattr(widget.style, 'color'):
                            widget.style.color = "#ffffff"
                    elif isinstance(widget, toga.Label):
                        if hasattr(widget.style, 'color'):
                            widget.style.color = colors.text_primary
                    elif isinstance(widget, toga.TextInput):
                        if hasattr(widget.style, 'background_color'):
                            widget.style.background_color = colors.surface
                        if hasattr(widget.style, 'color'):
                            widget.style.color = colors.text_primary
                    
                    # Force widget refresh if possible
                    if hasattr(widget, 'refresh'):
                        widget.refresh()
                        
            except Exception as e:
                logger.debug(f"Could not apply theme to widget {type(widget).__name__}: {e}")
    
    def set_auto_system_theme(self, enabled: bool) -> None:
        """
        Enable or disable automatic system theme following.
        
        Args:
            enabled: Whether to automatically follow system theme
        """
        self.auto_system_theme = enabled
        
        if enabled:
            # Start monitoring and apply current system theme
            asyncio.create_task(self.system_theme_detector.start_monitoring())
            current_system_theme = self.system_theme_detector.current_system_theme
            
            if current_system_theme != self.theme_manager.current_theme_name:
                asyncio.create_task(self.transition_to_theme(current_system_theme))
        else:
            # Stop monitoring
            self.system_theme_detector.stop_monitoring()
        
        logger.info(f"Auto system theme {'enabled' if enabled else 'disabled'}")
    
    def set_accessibility_mode(self, enabled: bool) -> None:
        """
        Enable or disable accessibility mode (disables animations).
        
        Args:
            enabled: Whether to enable accessibility mode
        """
        self.accessibility_mode = enabled
        logger.info(f"Theme transition accessibility mode {'enabled' if enabled else 'disabled'}")
    
    def set_transition_duration(self, duration: float) -> None:
        """
        Set the default transition duration.
        
        Args:
            duration: Transition duration in seconds
        """
        if duration < 0.1:
            duration = 0.1
        elif duration > 2.0:
            duration = 2.0
            
        self.transition_duration = duration
        logger.debug(f"Theme transition duration set to {duration}s")
    
    def get_current_theme_name(self) -> str:
        """Get the current theme name."""
        return self.theme_manager.current_theme_name
    
    def get_available_themes(self) -> List[str]:
        """Get list of available theme names."""
        return list(self.theme_manager.themes.keys())
    
    def is_transition_active(self) -> bool:
        """Check if any theme transition is currently active."""
        return any(state.is_active for state in self.active_transitions.values())
    
    def cancel_all_transitions(self) -> int:
        """
        Cancel all active theme transitions.
        
        Returns:
            Number of transitions cancelled
        """
        cancelled_count = 0
        
        for transition_id, state in list(self.active_transitions.items()):
            if state.is_active:
                state.is_active = False
                cancelled_count += 1
        
        self.active_transitions.clear()
        
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} active theme transitions")
        
        return cancelled_count
    
    def _on_system_theme_change(self, new_theme: str) -> None:
        """
        Handle system theme change.
        
        Args:
            new_theme: New system theme name
        """
        if self.auto_system_theme and new_theme != self.theme_manager.current_theme_name:
            logger.info(f"System theme changed to {new_theme}, transitioning...")
            asyncio.create_task(self.transition_to_theme(new_theme))
    
    async def start_system_monitoring(self) -> None:
        """Start system theme monitoring."""
        await self.system_theme_detector.start_monitoring()
    
    def stop_system_monitoring(self) -> None:
        """Stop system theme monitoring."""
        self.system_theme_detector.stop_monitoring()


# Global theme transition manager instance (will be initialized when needed)
_theme_transition_manager: Optional[ThemeTransitionManager] = None


def get_theme_transition_manager(animation_manager: Optional[AnimationManager] = None) -> ThemeTransitionManager:
    """
    Get the global theme transition manager instance.
    
    Args:
        animation_manager: Animation manager instance (required for first call)
        
    Returns:
        ThemeTransitionManager instance
    """
    global _theme_transition_manager
    
    if _theme_transition_manager is None:
        if animation_manager is None:
            raise ValueError("Animation manager required for first initialization")
        
        _theme_transition_manager = ThemeTransitionManager(animation_manager, theme_manager)
        logger.info("Global theme transition manager created")
    
    return _theme_transition_manager