#!/usr/bin/env python3
"""
Theme Transition Demo

This demo showcases the theme transition functionality including:
- Smooth animated theme switching
- System theme detection and automatic switching
- Accessibility mode support
- Performance monitoring during transitions
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from tektra.gui.animations.animation_manager import AnimationManager
from tektra.gui.animations.theme_transition_manager import (
    ThemeTransitionManager,
    get_theme_transition_manager
)
from tektra.gui.themes import ThemeManager, theme_manager


class ThemeTransitionDemo(toga.App):
    """Demo application for theme transitions."""
    
    def startup(self):
        """Initialize the demo application."""
        self.main_window = toga.MainWindow(title="Theme Transition Demo")
        
        # Initialize animation and theme systems
        self.animation_manager = AnimationManager()
        self.theme_transition_manager = get_theme_transition_manager(self.animation_manager)
        
        # Build the UI
        self.main_box = self._build_interface()
        self.main_window.content = self.main_box
        
        # Register main components for theme transitions
        self._register_widgets_for_transitions()
        
        # Set up theme transition callbacks
        self.theme_transition_manager.add_transition_callback(self._on_theme_transition)
        
        # Start system theme monitoring
        asyncio.create_task(self._start_monitoring())
        
        self.main_window.show()
    
    def _build_interface(self) -> toga.Box:
        """Build the demo interface."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        # Main container
        main_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["lg"],
                background_color=colors.background,
                flex=1
            )
        )
        
        # Title
        title = toga.Label(
            "Theme Transition Demo",
            style=Pack(
                font_size=24,
                font_weight="bold",
                color=colors.text_primary,
                padding_bottom=spacing["lg"]
            )
        )
        main_container.add(title)
        
        # Theme controls section
        theme_section = self._build_theme_controls()
        main_container.add(theme_section)
        
        # Settings section
        settings_section = self._build_settings_section()
        main_container.add(settings_section)
        
        # Demo content section
        content_section = self._build_demo_content()
        main_container.add(content_section)
        
        # Status section
        status_section = self._build_status_section()
        main_container.add(status_section)
        
        return main_container
    
    def _build_theme_controls(self) -> toga.Box:
        """Build theme control buttons."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        section = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["md"],
                background_color=colors.surface,
                margin_bottom=spacing["lg"]
            )
        )
        
        section_title = toga.Label(
            "Theme Controls",
            style=Pack(
                font_size=18,
                font_weight="bold",
                color=colors.text_primary,
                padding_bottom=spacing["sm"]
            )
        )
        section.add(section_title)
        
        # Button container
        button_container = toga.Box(
            style=Pack(direction=ROW, padding=spacing["sm"])
        )
        
        # Light theme button
        self.light_button = toga.Button(
            "Light Theme",
            on_press=self._switch_to_light,
            style=Pack(
                background_color=colors.primary,
                color="#ffffff",
                padding=spacing["sm"],
                margin_right=spacing["sm"]
            )
        )
        button_container.add(self.light_button)
        
        # Dark theme button
        self.dark_button = toga.Button(
            "Dark Theme",
            on_press=self._switch_to_dark,
            style=Pack(
                background_color=colors.primary,
                color="#ffffff",
                padding=spacing["sm"],
                margin_right=spacing["sm"]
            )
        )
        button_container.add(self.dark_button)
        
        # Auto theme button
        self.auto_button = toga.Button(
            "Auto (System)",
            on_press=self._toggle_auto_theme,
            style=Pack(
                background_color=colors.accent,
                color="#ffffff",
                padding=spacing["sm"]
            )
        )
        button_container.add(self.auto_button)
        
        section.add(button_container)
        return section
    
    def _build_settings_section(self) -> toga.Box:
        """Build settings controls."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        section = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["md"],
                background_color=colors.surface,
                margin_bottom=spacing["lg"]
            )
        )
        
        section_title = toga.Label(
            "Settings",
            style=Pack(
                font_size=18,
                font_weight="bold",
                color=colors.text_primary,
                padding_bottom=spacing["sm"]
            )
        )
        section.add(section_title)
        
        # Settings container
        settings_container = toga.Box(
            style=Pack(direction=COLUMN, padding=spacing["sm"])
        )
        
        # Accessibility mode toggle
        accessibility_container = toga.Box(
            style=Pack(direction=ROW, padding_bottom=spacing["sm"])
        )
        
        accessibility_label = toga.Label(
            "Accessibility Mode (No Animations):",
            style=Pack(
                color=colors.text_primary,
                padding_right=spacing["sm"],
                flex=1
            )
        )
        accessibility_container.add(accessibility_label)
        
        self.accessibility_button = toga.Button(
            "Disabled",
            on_press=self._toggle_accessibility,
            style=Pack(
                background_color=colors.accent,
                color="#ffffff",
                padding=(spacing["xs"], spacing["sm"])
            )
        )
        accessibility_container.add(self.accessibility_button)
        
        settings_container.add(accessibility_container)
        
        # Transition duration
        duration_container = toga.Box(
            style=Pack(direction=ROW, padding_bottom=spacing["sm"])
        )
        
        duration_label = toga.Label(
            "Transition Duration:",
            style=Pack(
                color=colors.text_primary,
                padding_right=spacing["sm"],
                flex=1
            )
        )
        duration_container.add(duration_label)
        
        self.duration_input = toga.TextInput(
            value="0.4",
            style=Pack(
                background_color=colors.background,
                color=colors.text_primary,
                padding=spacing["xs"],
                width=80
            )
        )
        duration_container.add(self.duration_input)
        
        duration_apply = toga.Button(
            "Apply",
            on_press=self._apply_duration,
            style=Pack(
                background_color=colors.primary,
                color="#ffffff",
                padding=(spacing["xs"], spacing["sm"]),
                margin_left=spacing["xs"]
            )
        )
        duration_container.add(duration_apply)
        
        settings_container.add(duration_container)
        
        section.add(settings_container)
        return section
    
    def _build_demo_content(self) -> toga.Box:
        """Build demo content to show theme effects."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        section = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["md"],
                background_color=colors.surface,
                margin_bottom=spacing["lg"],
                flex=1
            )
        )
        
        section_title = toga.Label(
            "Demo Content",
            style=Pack(
                font_size=18,
                font_weight="bold",
                color=colors.text_primary,
                padding_bottom=spacing["sm"]
            )
        )
        section.add(section_title)
        
        # Sample content
        content_text = toga.Label(
            "This is sample content to demonstrate theme transitions. "
            "Notice how all colors smoothly transition when you switch themes. "
            "The background, text colors, and button colors all change together "
            "in a coordinated animation.",
            style=Pack(
                color=colors.text_primary,
                padding=spacing["sm"],
                text_align="left"
            )
        )
        section.add(content_text)
        
        # Sample input
        sample_input = toga.TextInput(
            placeholder="Sample input field",
            style=Pack(
                background_color=colors.background,
                color=colors.text_primary,
                padding=spacing["sm"],
                margin_top=spacing["sm"]
            )
        )
        section.add(sample_input)
        
        # Sample buttons
        button_container = toga.Box(
            style=Pack(direction=ROW, padding_top=spacing["sm"])
        )
        
        sample_button1 = toga.Button(
            "Primary Button",
            style=Pack(
                background_color=colors.primary,
                color="#ffffff",
                padding=spacing["sm"],
                margin_right=spacing["sm"]
            )
        )
        button_container.add(sample_button1)
        
        sample_button2 = toga.Button(
            "Accent Button",
            style=Pack(
                background_color=colors.accent,
                color="#ffffff",
                padding=spacing["sm"]
            )
        )
        button_container.add(sample_button2)
        
        section.add(button_container)
        
        return section
    
    def _build_status_section(self) -> toga.Box:
        """Build status display section."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        section = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["md"],
                background_color=colors.surface
            )
        )
        
        section_title = toga.Label(
            "Status",
            style=Pack(
                font_size=18,
                font_weight="bold",
                color=colors.text_primary,
                padding_bottom=spacing["sm"]
            )
        )
        section.add(section_title)
        
        # Status labels
        self.current_theme_label = toga.Label(
            f"Current Theme: {theme_manager.current_theme_name}",
            style=Pack(
                color=colors.text_secondary,
                padding_bottom=spacing["xs"]
            )
        )
        section.add(self.current_theme_label)
        
        self.system_theme_label = toga.Label(
            f"System Theme: {self.theme_transition_manager.system_theme_detector.current_system_theme}",
            style=Pack(
                color=colors.text_secondary,
                padding_bottom=spacing["xs"]
            )
        )
        section.add(self.system_theme_label)
        
        self.auto_theme_label = toga.Label(
            f"Auto Theme: {'Enabled' if self.theme_transition_manager.auto_system_theme else 'Disabled'}",
            style=Pack(
                color=colors.text_secondary,
                padding_bottom=spacing["xs"]
            )
        )
        section.add(self.auto_theme_label)
        
        self.transition_status_label = toga.Label(
            "Transition Status: Ready",
            style=Pack(
                color=colors.text_secondary
            )
        )
        section.add(self.transition_status_label)
        
        return section
    
    def _register_widgets_for_transitions(self):
        """Register widgets for theme transitions."""
        # Register main container and key components
        self.theme_transition_manager.register_widget(self.main_box)
        
        # Note: In a real implementation, you would recursively register
        # all widgets that need theme transitions. For this demo, we'll
        # rely on the theme manager's style updates.
    
    async def _start_monitoring(self):
        """Start system theme monitoring."""
        try:
            await self.theme_transition_manager.start_system_monitoring()
        except Exception as e:
            print(f"Could not start system monitoring: {e}")
    
    def _switch_to_light(self, widget):
        """Switch to light theme."""
        asyncio.create_task(self._perform_theme_switch("light"))
    
    def _switch_to_dark(self, widget):
        """Switch to dark theme."""
        asyncio.create_task(self._perform_theme_switch("dark"))
    
    def _toggle_auto_theme(self, widget):
        """Toggle automatic system theme following."""
        current_auto = self.theme_transition_manager.auto_system_theme
        self.theme_transition_manager.set_auto_system_theme(not current_auto)
        
        # Update button text
        self.auto_button.text = "Auto: ON" if not current_auto else "Auto: OFF"
        
        # Update status
        self._update_status_labels()
    
    def _toggle_accessibility(self, widget):
        """Toggle accessibility mode."""
        current_accessibility = self.theme_transition_manager.accessibility_mode
        self.theme_transition_manager.set_accessibility_mode(not current_accessibility)
        
        # Update button text
        self.accessibility_button.text = "Enabled" if not current_accessibility else "Disabled"
    
    def _apply_duration(self, widget):
        """Apply new transition duration."""
        try:
            duration = float(self.duration_input.value)
            self.theme_transition_manager.set_transition_duration(duration)
            print(f"Transition duration set to {duration}s")
        except ValueError:
            print("Invalid duration value")
    
    async def _perform_theme_switch(self, theme_name: str):
        """Perform theme switch with status updates."""
        self.transition_status_label.text = f"Transitioning to {theme_name}..."
        
        try:
            await self.theme_transition_manager.transition_to_theme(theme_name)
            self.transition_status_label.text = f"Transitioned to {theme_name}"
            self._update_status_labels()
        except Exception as e:
            self.transition_status_label.text = f"Transition failed: {e}"
            print(f"Theme transition error: {e}")
    
    def _on_theme_transition(self, from_theme: str, to_theme: str):
        """Handle theme transition completion."""
        print(f"Theme transition completed: {from_theme} -> {to_theme}")
        self._update_status_labels()
    
    def _update_status_labels(self):
        """Update status display labels."""
        self.current_theme_label.text = f"Current Theme: {theme_manager.current_theme_name}"
        self.system_theme_label.text = f"System Theme: {self.theme_transition_manager.system_theme_detector.current_system_theme}"
        self.auto_theme_label.text = f"Auto Theme: {'Enabled' if self.theme_transition_manager.auto_system_theme else 'Disabled'}"


def main():
    """Run the theme transition demo."""
    app = ThemeTransitionDemo(
        "Theme Transition Demo",
        "org.tektra.theme_demo"
    )
    return app


if __name__ == "__main__":
    app = main()
    app.main_loop()