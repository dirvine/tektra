"""
Startup Dialog for Tektra AI Assistant

This module provides a startup dialog that allows users to choose
between local model loading or API-only mode.
"""

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from typing import Optional
from loguru import logger

from .themes import theme_manager
from .animations.animation_manager import AnimationManager


class StartupDialog:
    """
    Modern startup dialog for Tektra AI Assistant.
    
    Allows users to choose between:
    - Full mode with local AI models
    - API-only mode for faster startup
    """
    
    def __init__(self, app: toga.App, animation_manager: Optional[AnimationManager] = None):
        """
        Initialize the startup dialog.
        
        Args:
            app: The main Toga application
            animation_manager: Animation manager for micro-interactions
        """
        self.app = app
        self.window = None
        self.choice = None
        self.completed = False
        self.animation_manager = animation_manager or AnimationManager()
        
        # Track interactive elements for micro-interactions
        self.interactive_elements = {}
        
        # Build the dialog
        self._build_dialog()
        
    def _build_dialog(self):
        """Build the startup dialog interface."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        typography = theme.typography
        borders = theme.borders
        
        # Create window
        self.window = toga.Window(
            title="Welcome to Tektra AI Assistant",
            size=(600, 400),
            resizable=False
        )
        
        # Main container
        main_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["xl"],
                background_color=colors.surface
            )
        )
        
        # Logo/Title section
        title_label = toga.Label(
            "Tektra AI Assistant",
            style=Pack(
                font_size=typography["heading1"]["size"],
                font_weight=typography["heading1"]["weight"],
                color=colors.primary,
                margin_bottom=spacing["sm"],
                text_align="center"
            )
        )
        main_box.add(title_label)
        
        subtitle_label = toga.Label(
            "Choose how you'd like to start",
            style=Pack(
                font_size=typography["body1"]["size"],
                color=colors.text_secondary,
                margin_bottom=spacing["xl"],
                text_align="center"
            )
        )
        main_box.add(subtitle_label)
        
        # Options container
        options_container = toga.Box(
            style=Pack(
                direction=ROW,
                padding=spacing["md"]
            )
        )
        
        # Full mode option
        full_mode_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1,
                padding=spacing["lg"],
                margin_right=spacing["md"],
                background_color=colors.background
            )
        )
        
        full_mode_title = toga.Label(
            "🚀 Full Mode",
            style=Pack(
                font_size=typography["heading3"]["size"],
                font_weight=typography["heading3"]["weight"],
                color=colors.primary,
                margin_bottom=spacing["sm"],
                text_align="center"
            )
        )
        full_mode_box.add(full_mode_title)
        
        full_mode_desc = toga.Label(
            "Download and run AI models locally\n\n"
            "• Best performance\n"
            "• Works offline\n"
            "• ~2.5GB download\n"
            "• Takes 1-2 minutes",
            style=Pack(
                font_size=typography["body2"]["size"],
                color=colors.text_primary,
                margin_bottom=spacing["md"],
                text_align="center",
            )
        )
        full_mode_box.add(full_mode_desc)
        
        self.full_mode_button = toga.Button(
            "Start Full Mode",
            on_press=self._on_full_mode_selected,
            style=Pack(
                width=200,
                background_color=colors.primary,
                color="#ffffff",
                padding=(spacing["sm"], spacing["md"]),
                font_size=typography["button"]["size"],
                font_weight=typography["button"]["weight"]
            )
        )
        
        # Set up micro-interactions for full mode button
        self._setup_button_micro_interactions(
            self.full_mode_button,
            "full_mode_button",
            {
                "hover_scale": 1.05,
                "press_scale": 0.95,
                "hover_duration": 0.2,
                "press_duration": 0.1,
                "spring_back_duration": 0.25,
                "enable_spring_back": True
            }
        )
        
        # Center button
        full_button_container = toga.Box(style=Pack(direction=ROW))
        full_button_container.add(toga.Box(style=Pack(flex=1)))
        full_button_container.add(self.full_mode_button)
        full_button_container.add(toga.Box(style=Pack(flex=1)))
        full_mode_box.add(full_button_container)
        
        options_container.add(full_mode_box)
        
        # API mode option
        api_mode_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1,
                padding=spacing["lg"],
                background_color=colors.background
            )
        )
        
        api_mode_title = toga.Label(
            "☁️ API Mode",
            style=Pack(
                font_size=typography["heading3"]["size"],
                font_weight=typography["heading3"]["weight"],
                color=colors.accent,
                margin_bottom=spacing["sm"],
                text_align="center"
            )
        )
        api_mode_box.add(api_mode_title)
        
        api_mode_desc = toga.Label(
            "Use cloud-based AI services\n\n"
            "• Instant startup\n"
            "• Requires internet\n"
            "• No downloads\n"
            "• API key required",
            style=Pack(
                font_size=typography["body2"]["size"],
                color=colors.text_primary,
                margin_bottom=spacing["md"],
                text_align="center",
            )
        )
        api_mode_box.add(api_mode_desc)
        
        self.api_mode_button = toga.Button(
            "Start API Mode",
            on_press=self._on_api_mode_selected,
            style=Pack(
                width=200,
                background_color=colors.surface,
                color=colors.accent,
                padding=(spacing["sm"], spacing["md"]),
                font_size=typography["button"]["size"],
                font_weight=typography["button"]["weight"]
            )
        )
        
        # Set up micro-interactions for API mode button
        self._setup_button_micro_interactions(
            self.api_mode_button,
            "api_mode_button",
            {
                "hover_scale": 1.05,
                "press_scale": 0.95,
                "hover_duration": 0.2,
                "press_duration": 0.1,
                "spring_back_duration": 0.25,
                "enable_spring_back": True
            }
        )
        
        # Center button
        api_button_container = toga.Box(style=Pack(direction=ROW))
        api_button_container.add(toga.Box(style=Pack(flex=1)))
        api_button_container.add(self.api_mode_button)
        api_button_container.add(toga.Box(style=Pack(flex=1)))
        api_mode_box.add(api_button_container)
        
        options_container.add(api_mode_box)
        
        main_box.add(options_container)
        
        # Note at bottom
        note_label = toga.Label(
            "You can change this later in Settings",
            style=Pack(
                font_size=typography["caption"]["size"],
                color=colors.text_secondary,
                margin_top=spacing["lg"],
                text_align="center",
            )
        )
        main_box.add(note_label)
        
        self.window.content = main_box
    
    def _setup_button_micro_interactions(self, button: toga.Button, button_id: str, config: dict = None):
        """Set up micro-interactions for a button."""
        try:
            micro_manager = self.animation_manager.micro_interaction_manager
            element_id = micro_manager.setup_button_interactions(
                button,
                button_id=button_id,
                interaction_config=config
            )
            self.interactive_elements[button_id] = element_id
            logger.debug(f"Set up micro-interactions for button: {button_id}")
        except Exception as e:
            logger.debug(f"Could not set up micro-interactions for {button_id}: {e}")
        
    def show(self):
        """Show the startup dialog and wait for user choice."""
        self.window.show()
        
    def _on_full_mode_selected(self, widget):
        """Handle full mode selection."""
        self.choice = "full"
        self.completed = True
        self.window.close()
        
    def _on_api_mode_selected(self, widget):
        """Handle API mode selection."""
        self.choice = "api"
        self.completed = True
        self.window.close()
        
    def get_choice(self) -> str:
        """
        Get the user's choice.
        
        Returns:
            "full" or "api"
        """
        return self.choice