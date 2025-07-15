"""
Enhanced Input Field - Advanced Input Controls with Animations

This module provides an enhanced input field component with smooth animations,
auto-expansion, character counting, and visual feedback for the conversational UI.
"""

import asyncio
from typing import Callable, Optional, Dict, Any
from loguru import logger

import toga
from toga.style import Pack
from toga.style.pack import ROW, COLUMN

from .animations.animation_manager import AnimationManager
from .themes import theme_manager


class EnhancedInputField:
    """
    Enhanced input field with animations and advanced interactions.
    
    Features:
    - Smooth focus animations with border and shadow effects
    - Auto-expanding input field with smooth height transitions
    - Send button state animations with enabled/disabled feedback
    - Character count display with smooth updates and validation
    - Voice recording visual feedback
    """
    
    def __init__(
        self,
        animation_manager: AnimationManager,
        on_message_send: Optional[Callable[[str], None]] = None,
        on_voice_toggle: Optional[Callable[[], None]] = None,
        on_file_upload: Optional[Callable[[], None]] = None,
        placeholder: str = "Type your message...",
        max_characters: int = 4000,
        min_height: int = 40,
        max_height: int = 200
    ):
        """
        Initialize the enhanced input field.
        
        Args:
            animation_manager: Animation manager for smooth transitions
            on_message_send: Callback when user sends a message
            on_voice_toggle: Callback when voice mode is toggled
            on_file_upload: Callback when user uploads a file
            placeholder: Placeholder text for the input
            max_characters: Maximum character limit
            min_height: Minimum height of the input field
            max_height: Maximum height when expanded
        """
        self.animation_manager = animation_manager
        self.on_message_send = on_message_send
        self.on_voice_toggle = on_voice_toggle
        self.on_file_upload = on_file_upload
        self.placeholder = placeholder
        self.max_characters = max_characters
        self.min_height = min_height
        self.max_height = max_height
        
        # State tracking
        self.is_focused = False
        self.is_voice_recording = False
        self.current_character_count = 0
        self.is_send_enabled = False
        self.current_height = min_height
        
        # Feature flags
        self.voice_enabled = False
        self.file_enabled = False
        
        # Build the UI
        self.widget = self._build_interface()
        
        # Set up micro-interactions for all interactive elements
        self._setup_micro_interactions()
        
        logger.info("Enhanced input field initialized with micro-interactions")
    
    def _build_interface(self) -> toga.Box:
        """Build the enhanced input field interface."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        typography = theme.typography
        
        # Main container
        main_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["md"],
                background_color=colors.surface
            )
        )
        
        # Input area container
        input_area = toga.Box(
            style=Pack(
                direction=ROW,
                padding=spacing["sm"],
                background_color=colors.background,
                # Add subtle border and shadow effect simulation
                margin=2
            )
        )
        
        # Text input with enhanced styling
        self.text_input = toga.TextInput(
            placeholder=self.placeholder,
            style=Pack(
                flex=1,
                padding=spacing["sm"],
                font_size=typography["body2"]["size"],
                background_color=colors.background,
                margin=(0, spacing["sm"], 0, 0),
                # Enhanced focus styling will be applied dynamically
            ),
            on_change=self._on_input_change
        )
        
        # Set up focus and key event handling
        self.text_input.on_confirm = self._on_enter_pressed
        # Note: Toga doesn't have direct focus events, we'll simulate them
        
        input_area.add(self.text_input)
        
        # Action buttons container
        buttons_container = toga.Box(
            style=Pack(direction=ROW, padding=0)
        )
        
        # Voice button with enhanced styling
        self.voice_button = toga.Button(
            "🎤",
            on_press=self._on_voice_button_pressed,
            style=Pack(
                width=40,
                height=40,
                margin=(0, spacing["xs"], 0, 0),
                background_color=colors.surface,
                color=colors.primary,
                font_size=16
            ),
            enabled=False
        )
        buttons_container.add(self.voice_button)
        
        # File upload button
        self.file_button = toga.Button(
            "📎",
            on_press=self._on_file_button_pressed,
            style=Pack(
                width=40,
                height=40,
                margin=(0, spacing["xs"], 0, 0),
                background_color=colors.surface,
                color=colors.primary,
                font_size=16
            ),
            enabled=False
        )
        buttons_container.add(self.file_button)
        
        # Send button with enhanced state management
        self.send_button = toga.Button(
            "Send",
            on_press=self._on_send_button_pressed,
            style=Pack(
                padding=(spacing["sm"], spacing["md"]),
                background_color=colors.surface,  # Disabled state
                color=colors.text_secondary,
                font_size=typography["button"]["size"],
                font_weight=typography["button"]["weight"]
            ),
            enabled=False
        )
        buttons_container.add(self.send_button)
        
        input_area.add(buttons_container)
        main_container.add(input_area)
        
        # Character count and status area
        status_area = toga.Box(
            style=Pack(
                direction=ROW,
                padding=(spacing["xs"], spacing["sm"]),
                background_color=colors.surface
            )
        )
        
        # Character count label
        self.char_count_label = toga.Label(
            f"0/{self.max_characters}",
            style=Pack(
                font_size=typography["caption"]["size"],
                color=colors.text_secondary,
                flex=1
            )
        )
        status_area.add(self.char_count_label)
        
        # Status indicator (for validation feedback)
        self.status_indicator = toga.Label(
            "",
            style=Pack(
                font_size=typography["caption"]["size"],
                color=colors.text_secondary
            )
        )
        status_area.add(self.status_indicator)
        
        main_container.add(status_area)
        
        return main_container
    
    def _setup_micro_interactions(self):
        """Set up micro-interactions for all interactive elements."""
        try:
            micro_manager = self.animation_manager.micro_interaction_manager
            
            # Set up button interactions with enhanced configurations
            
            # Send button - primary action with prominent feedback
            micro_manager.setup_button_interactions(
                self.send_button,
                button_id="send_button",
                interaction_config={
                    "hover_scale": 1.05,
                    "press_scale": 0.95,
                    "hover_duration": 0.2,
                    "press_duration": 0.1,
                    "spring_back_duration": 0.2,
                    "enable_spring_back": True
                }
            )
            
            # Voice button - secondary action with subtle feedback
            micro_manager.setup_button_interactions(
                self.voice_button,
                button_id="voice_button",
                interaction_config={
                    "hover_scale": 1.08,
                    "press_scale": 0.92,
                    "hover_duration": 0.15,
                    "press_duration": 0.08,
                    "spring_back_duration": 0.18,
                    "enable_spring_back": True
                }
            )
            
            # File button - secondary action with subtle feedback
            micro_manager.setup_button_interactions(
                self.file_button,
                button_id="file_button",
                interaction_config={
                    "hover_scale": 1.08,
                    "press_scale": 0.92,
                    "hover_duration": 0.15,
                    "press_duration": 0.08,
                    "spring_back_duration": 0.18,
                    "enable_spring_back": True
                }
            )
            
            # Input field - focus interactions
            micro_manager.setup_input_interactions(
                self.text_input,
                input_id="main_input",
                interaction_config={
                    "focus_scale": 1.02,
                    "focus_duration": 0.25,
                    "unfocus_duration": 0.2,
                    "enable_focus_animation": True,
                    "enable_border_highlight": True,
                    "enable_subtle_glow": True
                }
            )
            
            # Character count label - subtle hover for information
            micro_manager.setup_hover_effects(
                self.char_count_label,
                widget_id="char_count_label",
                hover_config={
                    "hover_scale": 1.02,
                    "hover_duration": 0.1,
                    "color_transition": True,
                    "subtle_shadow": False,
                    "brightness_boost": 1.1
                }
            )
            
            logger.debug("Micro-interactions set up for enhanced input field")
            
        except Exception as e:
            logger.error(f"Error setting up micro-interactions: {e}")
    
    async def _on_input_change(self, widget):
        """Handle text input changes with enhanced feedback."""
        text = widget.value
        self.current_character_count = len(text)
        
        # Update character count with smooth animation
        await self._update_character_count()
        
        # Update send button state with animation
        await self._update_send_button_state()
        
        # Handle auto-expansion
        await self._handle_auto_expansion(text)
        
        # Input validation feedback
        await self._update_validation_feedback(text)
    
    async def _update_character_count(self):
        """Update character count display with smooth animation."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        # Calculate color based on character count
        ratio = self.current_character_count / self.max_characters if self.max_characters > 0 else 0
        
        if ratio < 0.8:
            color = colors.text_secondary
            status = ""
        elif ratio < 0.95:
            color = getattr(colors, 'warning', "#ff9800")
            status = "⚠️"
        else:
            color = getattr(colors, 'error', "#f44336")
            status = "❌" if self.current_character_count > self.max_characters else "⚠️"
        
        # Update the label
        self.char_count_label.text = f"{self.current_character_count}/{self.max_characters}"
        self.char_count_label.style.color = color
        self.status_indicator.text = status
        self.status_indicator.style.color = color
        
        # Animate the character count update
        if hasattr(self.animation_manager, 'animate_text_update'):
            await self.animation_manager.animate_text_update(self.char_count_label)
    
    async def _update_send_button_state(self):
        """Update send button state with smooth animation."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        text = self.text_input.value.strip()
        should_enable = (
            len(text) > 0 and 
            self.current_character_count <= self.max_characters and
            not self.is_voice_recording
        )
        
        if should_enable != self.is_send_enabled:
            self.is_send_enabled = should_enable
            
            # Animate button state change
            if should_enable:
                # Enable button with animation
                self.send_button.enabled = True
                self.send_button.style.background_color = colors.primary
                self.send_button.style.color = "#ffffff"
                
                # Animate button activation
                await self.animation_manager.animate_button_press(self.send_button)
                
            else:
                # Disable button with animation
                self.send_button.enabled = False
                self.send_button.style.background_color = colors.surface
                self.send_button.style.color = colors.text_secondary
                
                # Animate button deactivation
                if hasattr(self.animation_manager, 'animate_button_deactivate'):
                    await self.animation_manager.animate_button_deactivate(self.send_button)
    
    async def _handle_auto_expansion(self, text: str):
        """Handle auto-expanding input field with smooth transitions."""
        # Calculate required height based on text length and line breaks
        lines = text.count('\n') + 1
        # Also consider text wrapping - more aggressive estimate for testing
        char_based_lines = max(1, len(text) // 50)  # Assume ~50 chars per line (more aggressive)
        estimated_lines = max(lines, char_based_lines)
        
        # Calculate new height with padding
        line_height = 20  # Approximate line height
        padding = 20      # Top and bottom padding
        new_height = min(
            self.max_height,
            max(self.min_height, estimated_lines * line_height + padding)
        )
        
        if new_height != self.current_height:
            old_height = self.current_height
            self.current_height = new_height
            
            # Animate height change
            # Note: Toga doesn't support smooth height transitions directly
            # This is a placeholder for the animation logic
            logger.debug(f"Auto-expanding input from {old_height} to {new_height}")
            
            # In a real implementation, we'd animate the height change
            # For now, we'll just update the style
            self.text_input.style.height = new_height
    
    async def _update_validation_feedback(self, text: str):
        """Update input validation feedback."""
        # Only update validation feedback for specific cases that override character count status
        if self.current_character_count > self.max_characters:
            # Override character count status for over-limit case
            self.status_indicator.text = "❌ Too long"
            self.status_indicator.style.color = "#f44336"
        elif len(text.strip()) == 0 and len(text) > 0:
            # Only show empty message if we're not already showing character count warning
            ratio = self.current_character_count / self.max_characters if self.max_characters > 0 else 0
            if ratio < 0.8:  # Only override if not in warning/error range
                self.status_indicator.text = "⚠️ Empty message"
                self.status_indicator.style.color = "#ff9800"
        # Don't clear the status indicator - let character count logic handle it
    
    async def _on_enter_pressed(self, widget):
        """Handle Enter key press with enhanced behavior."""
        # Check if Shift+Enter for new line (simulated)
        # In a real implementation, we'd check for modifier keys
        
        if self.is_send_enabled:
            await self._send_message()
    
    async def _on_send_button_pressed(self, widget):
        """Handle Send button press with enhanced micro-interactions."""
        # The micro-interaction manager handles the button press animation automatically
        # through the setup in _setup_micro_interactions, so we just need to send the message
        await self._send_message()
    
    async def _on_voice_button_pressed(self, widget):
        """Handle voice button press with enhanced micro-interactions."""
        if not self.voice_enabled:
            return
        
        # The micro-interaction manager handles the button press animation automatically
        # Toggle voice recording state
        await self._toggle_voice_recording()
        
        # Call the callback
        if self.on_voice_toggle:
            try:
                await self.on_voice_toggle()
            except Exception as e:
                logger.error(f"Error in voice toggle callback: {e}")
    
    async def _on_file_button_pressed(self, widget):
        """Handle file upload button press with enhanced micro-interactions."""
        if not self.file_enabled:
            return
        
        # The micro-interaction manager handles the button press animation automatically
        # Call the callback
        if self.on_file_upload:
            try:
                await self.on_file_upload()
            except Exception as e:
                logger.error(f"Error in file upload callback: {e}")
    
    async def _send_message(self):
        """Send the current message with enhanced feedback."""
        text = self.text_input.value.strip()
        if not text or not self.is_send_enabled:
            return
        
        # Animate message sending
        await self._animate_message_send()
        
        # Clear input with animation
        await self._clear_input_animated()
        
        # Call the callback
        if self.on_message_send:
            try:
                await self.on_message_send(text)
            except Exception as e:
                logger.error(f"Error in message send callback: {e}")
    
    async def _animate_message_send(self):
        """Animate the message sending process."""
        # Flash the send button to indicate message sent
        original_color = self.send_button.style.background_color
        
        # Quick flash animation
        self.send_button.style.background_color = "#4caf50"  # Success green
        await asyncio.sleep(0.1)
        self.send_button.style.background_color = original_color
    
    async def _clear_input_animated(self):
        """Clear the input field with smooth animation."""
        # Animate the clearing process
        if hasattr(self.animation_manager, 'animate_text_clear'):
            await self.animation_manager.animate_text_clear(self.text_input)
        
        # Clear the text
        self.text_input.value = ""
        self.current_character_count = 0
        
        # Reset height to minimum
        self.current_height = self.min_height
        self.text_input.style.height = self.min_height
        
        # Update all dependent states
        await self._update_character_count()
        await self._update_send_button_state()
        await self._update_validation_feedback("")
    
    async def _toggle_voice_recording(self):
        """Toggle voice recording state with visual feedback."""
        self.is_voice_recording = not self.is_voice_recording
        
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        if self.is_voice_recording:
            # Recording state
            self.voice_button.text = "🔴"
            self.voice_button.style.background_color = "#ffebee"
            self.voice_button.style.color = "#f44336"
            
            # Disable other inputs during recording
            self.text_input.enabled = False
            self.send_button.enabled = False
            self.file_button.enabled = False
            
            # Update status
            self.status_indicator.text = "🎤 Recording..."
            self.status_indicator.style.color = "#f44336"
            
        else:
            # Ready state
            self.voice_button.text = "🎤"
            self.voice_button.style.background_color = "#e3f2fd"
            self.voice_button.style.color = colors.primary
            
            # Re-enable inputs
            self.text_input.enabled = True
            self.file_button.enabled = self.file_enabled
            
            # Clear status
            self.status_indicator.text = ""
            
            # Update send button state
            await self._update_send_button_state()
    
    async def animate_focus(self, focused: bool):
        """Animate input field focus state."""
        self.is_focused = focused
        
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        theme = theme_manager.get_theme()
        spacing = theme.spacing
        
        if focused:
            # Focus animation - enhance border and add shadow effect
            self.text_input.style.background_color = colors.background
            # Simulate border enhancement by adjusting margin
            self.text_input.style.margin = (0, spacing["sm"] - 1, 0, 0)
            
            # Animate focus
            await self.animation_manager.animate_input_focus(self.text_input, True)
            
        else:
            # Unfocus animation
            self.text_input.style.margin = (0, spacing["sm"], 0, 0)
            
            # Animate unfocus
            await self.animation_manager.animate_input_focus(self.text_input, False)
    
    def enable_voice_features(self, enabled: bool):
        """Enable or disable voice features."""
        self.voice_enabled = enabled
        self.voice_button.enabled = enabled
        
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        if enabled:
            self.voice_button.style.background_color = "#e3f2fd"
            self.voice_button.style.color = colors.primary
        else:
            self.voice_button.style.background_color = colors.surface
            self.voice_button.style.color = colors.text_secondary
    
    def enable_file_features(self, enabled: bool):
        """Enable or disable file upload features."""
        self.file_enabled = enabled
        self.file_button.enabled = enabled and not self.is_voice_recording
        
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        if enabled:
            self.file_button.style.background_color = "#e8f5e8"
            self.file_button.style.color = colors.primary
        else:
            self.file_button.style.background_color = colors.surface
            self.file_button.style.color = colors.text_secondary
    
    def get_current_text(self) -> str:
        """Get the current input text."""
        return self.text_input.value
    
    def set_text(self, text: str):
        """Set the input text programmatically."""
        self.text_input.value = text
        # Update character count synchronously for immediate feedback
        self.current_character_count = len(text)
        
        # Try to trigger change handler if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._on_input_change(self.text_input))
        except RuntimeError:
            # No event loop running, just update the basic state
            logger.debug("No event loop running, skipping async input change handler")
    
    def clear_text(self):
        """Clear the input text."""
        asyncio.create_task(self._clear_input_animated())
    
    def get_character_count(self) -> int:
        """Get the current character count."""
        return self.current_character_count
    
    def is_valid_input(self) -> bool:
        """Check if the current input is valid."""
        return (
            self.current_character_count > 0 and
            self.current_character_count <= self.max_characters and
            len(self.text_input.value.strip()) > 0
        )