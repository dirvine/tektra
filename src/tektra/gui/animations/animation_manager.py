"""
Animation Manager - Central Coordinator for UI Animations

This module provides the main AnimationManager class that coordinates all
UI animations and provides a high-level API for animation operations.
"""

import asyncio
import os
import platform
import time
from typing import Dict, List, Optional, Callable, Any
from loguru import logger

import toga

from .animation_config import AnimationType, AnimationConfig, DEFAULT_ANIMATIONS, REDUCED_MOTION_ANIMATIONS
from .transition_engine import TransitionEngine
from .performance_monitor import UIPerformanceMonitor
from .performance_optimizer import PerformanceOptimizer


class AnimationManager:
    """Central coordinator for all UI animations."""
    
    def __init__(self):
        """Initialize the animation manager."""
        self.performance_monitor = UIPerformanceMonitor()
        self.performance_optimizer = PerformanceOptimizer(self.performance_monitor)
        self.transition_engine = TransitionEngine(self.performance_monitor)
        self.reduced_motion_enabled = self._detect_reduced_motion_preference()
        self.animation_presets: Dict[str, Dict] = {}
        self.global_animation_enabled = True
        
        # Initialize micro-interaction manager (lazy import to avoid circular dependency)
        self._micro_interaction_manager = None
        
        # Set up performance optimization callbacks
        self.performance_monitor.add_optimization_callback(self._on_performance_optimization)
        self.performance_optimizer.add_optimization_callback(self._on_profile_optimization)
        
        # Initialize animation presets
        self._initialize_presets()
        
        # Start background performance monitoring
        self.performance_monitor.start_background_monitoring()
        self.performance_optimizer.enable_optimization(True)
        
        logger.info("Animation Manager initialized with performance optimization")
    
    @property
    def micro_interaction_manager(self):
        """Get the micro-interaction manager (lazy initialization)."""
        if self._micro_interaction_manager is None:
            from .micro_interaction_manager import MicroInteractionManager
            self._micro_interaction_manager = MicroInteractionManager(self)
        return self._micro_interaction_manager
    
    async def animate_message_appearance(self, message_widget: toga.Widget,
                                       role: str = "assistant") -> str:
        """
        Animate the appearance of a new message with enhanced effects.
        
        Args:
            message_widget: The message widget to animate
            role: The role of the message sender ("user", "assistant", "system")
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Enhanced animations based on role with modern chat app feel
        if role == "user":
            # User messages: slide in from right with slight bounce
            slide_id = await self.transition_engine.slide_in(
                message_widget, direction="right", duration=0.35, easing="ease_out"
            )
            # Add subtle scale effect for more dynamic feel
            await asyncio.sleep(0.1)  # Small delay for staggered effect
            await self.transition_engine.scale_in(
                message_widget, from_scale=0.98, duration=0.2, easing="ease_out"
            )
            return slide_id
            
        elif role == "assistant":
            # Assistant messages: fade in with gentle scale and slight slide from left
            # Start with fade and scale simultaneously
            fade_task = asyncio.create_task(
                self.transition_engine.fade_in(message_widget, duration=0.4, easing="ease_out")
            )
            scale_task = asyncio.create_task(
                self.transition_engine.scale_in(message_widget, from_scale=0.92, duration=0.35, easing="ease_out")
            )
            slide_task = asyncio.create_task(
                self.transition_engine.slide_in(message_widget, direction="left", duration=0.3, easing="ease_out")
            )
            
            # Wait for all animations to complete
            fade_id, _, _ = await asyncio.gather(fade_task, scale_task, slide_task)
            return fade_id
            
        else:
            # System messages: subtle fade in with very gentle scale
            fade_task = asyncio.create_task(
                self.transition_engine.fade_in(message_widget, duration=0.25, easing="ease_out")
            )
            scale_task = asyncio.create_task(
                self.transition_engine.scale_in(message_widget, from_scale=0.97, duration=0.2, easing="ease_out")
            )
            
            fade_id, _ = await asyncio.gather(fade_task, scale_task)
            return fade_id
    
    async def animate_typing_indicator(self, indicator: toga.Widget) -> str:
        """
        Animate a typing indicator with pulsing effect.
        
        Args:
            indicator: The typing indicator widget
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Create a pulsing animation by combining fade in/out
        animation_id = await self.transition_engine.fade_in(indicator, duration=0.8)
        
        # Start a continuous pulsing loop
        asyncio.create_task(self._pulse_animation_loop(indicator))
        
        return animation_id
    
    async def animate_button_press(self, button: toga.Button) -> str:
        """
        Animate a button press with scale effect.
        
        Args:
            button: The button to animate
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Quick scale down and back up
        scale_down_id = await self.transition_engine.scale_out(
            button, to_scale=0.95, duration=0.1
        )
        
        # Scale back up after a short delay
        await asyncio.sleep(0.1)
        scale_up_id = await self.transition_engine.scale_in(
            button, from_scale=0.95, duration=0.1
        )
        
        return scale_up_id
    
    async def animate_theme_transition(self, container: toga.Widget,
                                     from_theme: str, to_theme: str) -> str:
        """
        Animate a smooth transition between themes.
        
        Args:
            container: The container to animate
            from_theme: Name of the current theme
            to_theme: Name of the target theme
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Use the theme transition manager for proper theme transitions
        try:
            from .theme_transition_manager import get_theme_transition_manager
            theme_transition_manager = get_theme_transition_manager(self)
            
            # Register the container for theme transitions
            theme_transition_manager.register_widget(container)
            
            # Perform the animated transition
            transition_id = await theme_transition_manager.transition_to_theme(
                to_theme, animated=True, duration=0.5
            )
            
            logger.info(f"Animated theme transition from {from_theme} to {to_theme}")
            return transition_id
            
        except Exception as e:
            logger.error(f"Error in theme transition: {e}")
            
            # Fallback to simple fade transition
            fade_out_id = await self.transition_engine.fade_out(container, duration=0.25)
            await asyncio.sleep(0.25)
            fade_in_id = await self.transition_engine.fade_in(container, duration=0.25)
            
            logger.info(f"Fallback theme transition from {from_theme} to {to_theme}")
            return fade_in_id
    
    async def animate_input_focus(self, input_widget: toga.TextInput, focused: bool) -> str:
        """
        Animate input field focus state with enhanced effects.
        
        Args:
            input_widget: The input widget
            focused: Whether the input is being focused or unfocused
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        if focused:
            # Enhanced focus animation with scale and subtle glow effect
            scale_task = asyncio.create_task(
                self.transition_engine.scale_in(input_widget, from_scale=0.98, duration=0.25, easing="ease_out")
            )
            # Simulate border glow by adjusting background slightly
            fade_task = asyncio.create_task(
                self.transition_engine.fade_in(input_widget, duration=0.2, easing="ease_out")
            )
            
            scale_id, _ = await asyncio.gather(scale_task, fade_task)
            return scale_id
        else:
            # Gentle unfocus animation
            return await self.transition_engine.scale_out(
                input_widget, to_scale=0.98, duration=0.2, easing="ease_in"
            )
    
    async def animate_button_deactivate(self, button: toga.Button) -> str:
        """
        Animate button deactivation with subtle fade effect.
        
        Args:
            button: The button to animate
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Gentle scale down to indicate deactivation
        return await self.transition_engine.scale_out(
            button, to_scale=0.97, duration=0.15, easing="ease_in"
        )
    
    async def animate_text_update(self, text_widget: toga.Label) -> str:
        """
        Animate text content updates with subtle pulse effect.
        
        Args:
            text_widget: The text widget to animate
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Quick pulse animation to draw attention to text change
        scale_down_task = asyncio.create_task(
            self.transition_engine.scale_out(text_widget, to_scale=0.95, duration=0.1)
        )
        await scale_down_task
        
        return await self.transition_engine.scale_in(
            text_widget, from_scale=0.95, duration=0.15, easing="ease_out"
        )
    
    async def animate_text_clear(self, input_widget: toga.TextInput) -> str:
        """
        Animate text clearing with smooth fade effect.
        
        Args:
            input_widget: The input widget to animate
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Quick fade out and back in to indicate clearing
        fade_out_task = asyncio.create_task(
            self.transition_engine.fade_out(input_widget, duration=0.1)
        )
        await fade_out_task
        
        return await self.transition_engine.fade_in(
            input_widget, duration=0.15, easing="ease_out"
        )
    
    async def animate_scroll_to_bottom(self, scroll_container: toga.ScrollContainer) -> str:
        """
        Animate smooth scrolling to the bottom of a container.
        
        Args:
            scroll_container: The scroll container to animate
            
        Returns:
            Animation ID for tracking
        """
        if not self.global_animation_enabled:
            return ""
        
        # Toga doesn't have direct scroll animation support
        # This is a placeholder for future implementation
        logger.debug("Scroll animation requested (not implemented in Toga)")
        return ""
    
    def set_reduced_motion(self, enabled: bool) -> None:
        """
        Enable or disable reduced motion mode.
        
        Args:
            enabled: Whether to enable reduced motion
        """
        self.reduced_motion_enabled = enabled
        self.transition_engine.set_reduced_motion(enabled)
        logger.info(f"Reduced motion {'enabled' if enabled else 'disabled'}")
    
    def set_global_animations_enabled(self, enabled: bool) -> None:
        """
        Enable or disable all animations globally.
        
        Args:
            enabled: Whether to enable animations
        """
        self.global_animation_enabled = enabled
        logger.info(f"Global animations {'enabled' if enabled else 'disabled'}")
    
    def get_performance_summary(self) -> Dict:
        """Get current animation performance summary."""
        return self.performance_monitor.get_performance_summary()
    
    def cancel_all_animations(self) -> int:
        """
        Cancel all active animations.
        
        Returns:
            Number of animations cancelled
        """
        active_count = self.transition_engine.get_active_animation_count()
        
        # Cancel all animations in the transition engine
        for animation_id in list(self.transition_engine.active_animations.keys()):
            self.transition_engine.cancel_animation(animation_id)
        
        logger.info(f"Cancelled {active_count} active animations")
        return active_count
    
    async def start_performance_monitoring(self) -> None:
        """Start the performance monitoring loop."""
        await self.performance_monitor.start_monitoring_loop()
    
    def add_animation_preset(self, name: str, config: Dict) -> None:
        """
        Add a custom animation preset.
        
        Args:
            name: Name of the preset
            config: Animation configuration dictionary
        """
        self.animation_presets[name] = config
        logger.debug(f"Added animation preset: {name}")
    
    def apply_animation_preset(self, preset_name: str) -> bool:
        """
        Apply an animation preset.
        
        Args:
            preset_name: Name of the preset to apply
            
        Returns:
            True if preset was applied, False if not found
        """
        if preset_name not in self.animation_presets:
            logger.warning(f"Animation preset not found: {preset_name}")
            return False
        
        preset = self.animation_presets[preset_name]
        
        # Apply preset settings
        if "reduced_motion" in preset:
            self.set_reduced_motion(preset["reduced_motion"])
        
        if "global_enabled" in preset:
            self.set_global_animations_enabled(preset["global_enabled"])
        
        logger.info(f"Applied animation preset: {preset_name}")
        return True
    
    def _detect_reduced_motion_preference(self) -> bool:
        """
        Detect system reduced motion preference.
        
        Returns:
            True if reduced motion is preferred
        """
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                # Check macOS accessibility setting
                import subprocess
                result = subprocess.run([
                    "defaults", "read", "com.apple.universalaccess", "reduceMotion"
                ], capture_output=True, text=True)
                return result.returncode == 0 and "1" in result.stdout
                
            elif system == "windows":
                # Check Windows animation settings
                import winreg
                try:
                    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                       r"Control Panel\Desktop\WindowMetrics")
                    value, _ = winreg.QueryValueEx(key, "MinAnimate")
                    winreg.CloseKey(key)
                    return value == "0"
                except:
                    pass
                    
            elif system == "linux":
                # Check GTK settings
                gtk_settings_file = os.path.expanduser("~/.config/gtk-3.0/settings.ini")
                if os.path.exists(gtk_settings_file):
                    with open(gtk_settings_file, 'r') as f:
                        content = f.read()
                        if "gtk-enable-animations=false" in content:
                            return True
                            
        except Exception as e:
            logger.debug(f"Could not detect reduced motion preference: {e}")
        
        return False
    
    def _initialize_presets(self) -> None:
        """Initialize default animation presets."""
        self.animation_presets = {
            "performance": {
                "reduced_motion": True,
                "global_enabled": True,
                "description": "Optimized for performance"
            },
            "accessibility": {
                "reduced_motion": True,
                "global_enabled": True,
                "description": "Accessibility-friendly animations"
            },
            "full": {
                "reduced_motion": False,
                "global_enabled": True,
                "description": "Full animation experience"
            },
            "minimal": {
                "reduced_motion": True,
                "global_enabled": True,
                "description": "Minimal animations only"
            },
            "disabled": {
                "reduced_motion": True,
                "global_enabled": False,
                "description": "All animations disabled"
            }
        }
    
    async def _pulse_animation_loop(self, widget: toga.Widget) -> None:
        """
        Create a continuous pulsing animation loop.
        
        Args:
            widget: Widget to animate
        """
        try:
            while True:
                # Fade out
                await self.transition_engine.fade_out(widget, duration=0.8)
                await asyncio.sleep(0.8)
                
                # Fade in
                await self.transition_engine.fade_in(widget, duration=0.8)
                await asyncio.sleep(0.8)
                
        except asyncio.CancelledError:
            logger.debug("Pulse animation loop cancelled")
        except Exception as e:
            logger.error(f"Error in pulse animation loop: {e}")
    
    def _on_performance_optimization(self) -> None:
        """Handle performance optimization callback."""
        logger.info("Performance optimization triggered - enabling reduced motion")
        self.set_reduced_motion(True)
    
    def _on_profile_optimization(self, profile_name: str, profile) -> None:
        """Handle optimization profile change callback."""
        logger.info(f"Optimization profile changed to: {profile_name}")
        
        # Adjust animation settings based on profile
        if profile_name in ["performance", "minimal", "fallback"]:
            self.set_reduced_motion(True)
        else:
            self.set_reduced_motion(False)
        
        # Clear animation queue if in fallback mode
        if profile_name == "fallback":
            cleared = self.performance_optimizer.clear_animation_queue()
            if cleared > 0:
                logger.info(f"Cleared {cleared} queued animations due to fallback mode")
    
    def get_comprehensive_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary including optimization state."""
        monitor_summary = self.performance_monitor.get_detailed_performance_summary()
        optimizer_summary = self.performance_optimizer.get_optimization_summary()
        
        return {
            "animation_manager": {
                "global_animations_enabled": self.global_animation_enabled,
                "reduced_motion_enabled": self.reduced_motion_enabled,
                "active_animation_presets": list(self.animation_presets.keys())
            },
            "performance_monitor": monitor_summary,
            "performance_optimizer": optimizer_summary,
            "recommendations": self._get_integrated_recommendations()
        }
    
    def _get_integrated_recommendations(self) -> List[str]:
        """Get integrated recommendations from all performance systems."""
        recommendations = []
        
        # Get recommendations from monitor
        monitor_recs = self.performance_monitor.get_performance_recommendations()
        recommendations.extend(monitor_recs)
        
        # Add animation manager specific recommendations
        if self.performance_monitor.metrics.animation_count > 15:
            recommendations.append("Consider reducing concurrent animations")
        
        if not self.reduced_motion_enabled and self.performance_monitor.should_reduce_animations():
            recommendations.append("Enable reduced motion mode for better performance")
        
        # Check if optimization profile is appropriate
        recommended_profile = self.performance_optimizer.get_recommended_profile()
        current_profile = self.performance_optimizer.current_profile.name
        
        if recommended_profile != current_profile:
            recommendations.append(f"Consider switching to '{recommended_profile}' optimization profile")
        
        return recommendations
    
    async def animate_with_performance_optimization(self, animation_type: AnimationType, 
                                                  widget: toga.Widget, **kwargs) -> Optional[str]:
        """
        Execute an animation with automatic performance optimization.
        
        Args:
            animation_type: Type of animation to perform
            widget: Widget to animate
            **kwargs: Animation parameters
            
        Returns:
            Animation ID if executed, None if skipped
        """
        # Check if animation should be skipped
        if self.performance_optimizer.should_skip_animation(animation_type):
            logger.debug(f"Skipping {animation_type.value} animation due to performance constraints")
            return None
        
        # Record animation start for performance tracking
        animation_id = f"{animation_type.value}_{id(widget)}_{time.time()}"
        self.performance_monitor.record_animation_start(animation_id, animation_type.value)
        
        try:
            # Get optimized configuration
            base_config = DEFAULT_ANIMATIONS.get(animation_type, AnimationConfig())
            optimized_config = self.performance_optimizer.optimize_animation_config(base_config)
            
            # Execute animation based on type
            result_id = None
            if animation_type == AnimationType.FADE_IN:
                result_id = await self.transition_engine.fade_in(
                    widget, duration=optimized_config.duration, 
                    easing=optimized_config.easing.value
                )
            elif animation_type == AnimationType.FADE_OUT:
                result_id = await self.transition_engine.fade_out(
                    widget, duration=optimized_config.duration,
                    easing=optimized_config.easing.value
                )
            elif animation_type == AnimationType.SCALE_IN:
                result_id = await self.transition_engine.scale_in(
                    widget, duration=optimized_config.duration,
                    easing=optimized_config.easing.value,
                    **kwargs
                )
            elif animation_type == AnimationType.SCALE_OUT:
                result_id = await self.transition_engine.scale_out(
                    widget, duration=optimized_config.duration,
                    easing=optimized_config.easing.value,
                    **kwargs
                )
            elif animation_type == AnimationType.SLIDE_IN:
                result_id = await self.transition_engine.slide_in(
                    widget, duration=optimized_config.duration,
                    easing=optimized_config.easing.value,
                    **kwargs
                )
            elif animation_type == AnimationType.SLIDE_OUT:
                result_id = await self.transition_engine.slide_out(
                    widget, duration=optimized_config.duration,
                    easing=optimized_config.easing.value,
                    **kwargs
                )
            
            # Record successful completion
            self.performance_monitor.record_animation_end(animation_id, completed=True)
            return result_id or animation_id
            
        except Exception as e:
            logger.error(f"Error in optimized animation {animation_type.value}: {e}")
            # Record failed animation
            self.performance_monitor.record_animation_end(animation_id, completed=False)
            return None
    
    def cleanup_and_shutdown(self) -> None:
        """Clean up resources and stop background processes."""
        logger.info("Shutting down Animation Manager")
        
        # Stop background monitoring
        self.performance_monitor.stop_background_monitoring()
        
        # Cancel all active animations
        cancelled_count = self.cancel_all_animations()
        
        # Clear animation queue
        cleared_count = self.performance_optimizer.clear_animation_queue()
        
        logger.info(f"Shutdown complete: {cancelled_count} animations cancelled, {cleared_count} queued animations cleared")