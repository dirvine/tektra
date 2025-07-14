"""
Chat Panel - Core Conversational Interface

This module provides the main chat interface for Tektra AI Assistant,
focusing on natural conversation flow and minimal UI.
"""

import asyncio
from datetime import datetime
from typing import Callable, Optional

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .markdown_renderer import get_markdown_renderer
from .themes import theme_manager


class ChatPanel:
    """
    Main chat interface for Tektra AI Assistant.
    
    Design Philosophy:
    - Conversation is the primary interface
    - Minimal chrome, maximum content
    - Natural message flow like messaging apps
    - Voice-first design with text fallback
    """

    def __init__(
        self,
        on_message_send: Callable[[str], None],
        on_voice_toggle: Callable[[], None],
        on_file_upload: Callable[[], None],
    ):
        """
        Initialize the chat panel.
        
        Args:
            on_message_send: Callback when user sends a message
            on_voice_toggle: Callback when voice mode is toggled
            on_file_upload: Callback when user uploads a file
        """
        self.on_message_send = on_message_send
        self.on_voice_toggle = on_voice_toggle
        self.on_file_upload = on_file_upload
        
        # State
        self.messages = []
        self.voice_enabled = False
        self.file_enabled = False
        self.is_voice_active = False
        
        # Initialize markdown renderer
        self.markdown_renderer = get_markdown_renderer()
        
        # Build the UI
        self.widget = self._build_interface()
        
        logger.info("Chat panel initialized")

    def _build_interface(self) -> toga.Box:
        """Build the main chat interface."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        
        # Main container
        main_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1,
                padding=0,
                background_color=colors.background
            )
        )
        
        # Messages area (scrollable)
        self._build_messages_area()
        main_container.add(self.messages_scroll)
        
        # Input area
        input_area = self._build_input_area()
        main_container.add(input_area)
        
        return main_container

    def _build_messages_area(self):
        """Build the scrollable messages area."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        # Container for all messages
        self.messages_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["lg"],
                background_color=colors.surface
            )
        )
        
        # Scroll container
        self.messages_scroll = toga.ScrollContainer(
            content=self.messages_container,
            style=Pack(
                flex=1,
                background_color=colors.surface
            )
        )
        
        # Add welcome message
        self._add_welcome_message()

    def _build_input_area(self) -> toga.Box:
        """Build the input area at the bottom."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        typography = theme.typography
        
        input_container = toga.Box(
            style=Pack(
                direction=ROW,
                padding=spacing["md"],
                background_color=colors.surface,
            )
        )
        
        # Text input (auto-expanding)
        self.text_input = toga.TextInput(
            placeholder="Type your message...",
            style=Pack(
                flex=1,
                padding=spacing["sm"],
                font_size=typography["body2"]["size"],
                background_color=colors.background,
                margin=(0, spacing["sm"], 0, 0)
            ),
            on_change=self._on_input_change
        )
        
        # Set up enter key handling
        self.text_input.on_confirm = self._on_enter_pressed
        
        input_container.add(self.text_input)
        
        # Action buttons
        buttons_container = toga.Box(
            style=Pack(direction=ROW)
        )
        
        # Voice button
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
        
        # Send button
        self.send_button = toga.Button(
            "Send",
            on_press=self._on_send_button_pressed,
            style=Pack(
                padding=(spacing["sm"], spacing["md"]),
                background_color=colors.primary,
                color="#ffffff",
                font_size=typography["button"]["size"],
                font_weight=typography["button"]["weight"]
            ),
            enabled=False
        )
        buttons_container.add(self.send_button)
        
        input_container.add(buttons_container)
        
        return input_container

    def _add_welcome_message(self):
        """Add initial welcome message."""
        welcome_text = (
            "👋 Welcome to Tektra AI Assistant!\n\n"
            "I'm initializing my AI models... This may take a moment.\n\n"
            "Once ready, you can:\n"
            "• Chat with me by typing messages\n"
            "• Talk to me using voice mode (🎤)\n"
            "• Upload files for analysis (📎)\n"
            "• Ask me to create AI agents for you\n\n"
            "Just start a conversation naturally!"
        )
        
        self.add_message("assistant", welcome_text, timestamp=datetime.now())

    def add_message(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """
        Add a message to the chat.
        
        Args:
            role: 'user', 'assistant', or 'system'
            content: The message content
            timestamp: When the message was sent
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        
        self.messages.append(message)
        self._render_message(message)
        
        # Scroll to bottom
        self._scroll_to_bottom()
        
        logger.debug(f"Added {role} message: {content[:50]}...")

    def _get_message_bubble_style(self, role: str) -> dict:
        """Get the style dictionary for a message bubble based on role."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        borders = theme.borders
        
        if role == "user":
            return {
                "background_color": colors.primary,
                "margin": (0, 0, 0, 50)
            }
        elif role == "assistant":
            return {
                "background_color": colors.card,
                "margin": (0, 50, 0, 0)
            }
        else:  # system
            return {
                "background_color": colors.surface,
                "margin": (0, 50, 0, 50)
            }
    
    def _render_message(self, message: dict):
        """Render a single message in the chat with markdown support."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        borders = theme.borders
        
        # Outer container for alignment
        outer_container = toga.Box(
            style=Pack(
                direction=ROW,
                flex=1,
                margin_bottom=spacing["sm"]
            )
        )
        
        # Add spacer for right alignment (user messages)
        if message["role"] == "user":
            spacer = toga.Box(style=Pack(flex=1))
            outer_container.add(spacer)
        
        # Message bubble container with modern styling
        bubble_style = self._get_message_bubble_style(message["role"])
        message_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["md"],
                width=600,
                **bubble_style
            )
        )
        
        # Role and timestamp header
        if message["role"] != "system":
            header = toga.Box(style=Pack(direction=ROW, margin_bottom=5))
            
            # Role with emoji
            role_emoji = "👤" if message["role"] == "user" else "🤖"
            role_name = "You" if message["role"] == "user" else "Tektra"
            
            role_label = toga.Label(
                f"{role_emoji} {role_name}",
                style=Pack(
                    font_weight="bold",
                    font_size=theme.typography["caption"]["size"],
                    color=(
                        "#ffffff" if message["role"] == "user"
                        else colors.primary
                    )
                )
            )
            header.add(role_label)
            
            # Add timestamp
            time_str = message["timestamp"].strftime("%H:%M")
            time_label = toga.Label(
                time_str,
                style=Pack(
                    font_size=theme.typography["caption"]["size"] - 2,
                    color=(
                        "rgba(255, 255, 255, 0.7)" if message["role"] == "user"
                        else colors.text_secondary
                    ),
                    margin=(0, 0, 0, spacing["sm"])
                )
            )
            header.add(time_label)
            
            message_container.add(header)
        
        # Message content with markdown rendering
        if message["role"] == "system":
            # System messages are simple
            content_label = toga.Label(
                f"ℹ️ {message['content']}",
                style=Pack(
                    font_size=theme.typography["caption"]["size"],
                    color=colors.text_secondary,
                )
            )
            message_container.add(content_label)
        else:
            # Use markdown renderer for user and assistant messages
            try:
                markdown_content = self.markdown_renderer.render_simple_message(
                    message["content"], 
                    message["role"]
                )
                message_container.add(markdown_content)
            except Exception as e:
                # Fallback to plain text if markdown rendering fails
                logger.warning(f"Markdown rendering failed: {e}")
                content_label = toga.Label(
                    message["content"],
                    style=Pack(
                        font_size=theme.typography["body1"]["size"],
                        color=(
                            "#ffffff" if message["role"] == "user"
                            else colors.text_primary
                        ),
                    )
                )
                message_container.add(content_label)
        
        # Add message container to outer container
        outer_container.add(message_container)
        
        # Add spacer for left alignment (assistant messages)
        if message["role"] == "assistant":
            spacer = toga.Box(style=Pack(flex=1))
            outer_container.add(spacer)
        
        # Add to messages container
        self.messages_container.add(outer_container)

    def _scroll_to_bottom(self):
        """Scroll the messages area to the bottom."""
        # Note: Toga doesn't have direct scroll control yet
        # This is a placeholder for future implementation
        pass

    def _on_input_change(self, widget):
        """Handle text input changes."""
        text = widget.value.strip()
        self.send_button.enabled = len(text) > 0

    def _on_enter_pressed(self, widget):
        """Handle Enter key press in text input."""
        self._send_message()

    def _on_send_button_pressed(self, widget):
        """Handle Send button press."""
        self._send_message()

    def _on_voice_button_pressed(self, widget):
        """Handle voice button press."""
        if self.on_voice_toggle:
            asyncio.create_task(self._handle_voice_toggle())

    def _on_file_button_pressed(self, widget):
        """Handle file upload button press."""
        if self.on_file_upload:
            asyncio.create_task(self._handle_file_upload())

    def _send_message(self):
        """Send the current message."""
        text = self.text_input.value.strip()
        if not text:
            return
        
        # Add user message
        self.add_message("user", text)
        
        # Clear input
        self.text_input.value = ""
        self.send_button.enabled = False
        
        # Send to handler
        if self.on_message_send:
            asyncio.create_task(self._handle_message_send(text))

    async def _handle_message_send(self, message: str):
        """Handle message sending asynchronously."""
        try:
            await self.on_message_send(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.add_message("system", f"Error sending message: {e}")

    async def _handle_voice_toggle(self):
        """Handle voice toggle asynchronously."""
        try:
            await self.on_voice_toggle()
        except Exception as e:
            logger.error(f"Error toggling voice: {e}")
            self.add_message("system", f"Voice error: {e}")

    async def _handle_file_upload(self):
        """Handle file upload asynchronously."""
        try:
            await self.on_file_upload()
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            self.add_message("system", f"File upload error: {e}")

    def enable_voice_features(self, enabled: bool):
        """Enable or disable voice features."""
        self.voice_enabled = enabled
        self.voice_button.enabled = enabled
        
        if enabled:
            self.voice_button.style.background_color = "#e3f2fd"
            self.add_message("system", "🎤 Voice features enabled! Click the microphone to start talking.")
        else:
            self.voice_button.style.background_color = "#eeeeee"

    def enable_file_features(self, enabled: bool):
        """Enable or disable file upload features."""
        self.file_enabled = enabled
        self.file_button.enabled = enabled
        
        if enabled:
            self.file_button.style.background_color = "#e8f5e8"
            self.add_message("system", "📎 File upload enabled! Click the paperclip to analyze files.")
        else:
            self.file_button.style.background_color = "#eeeeee"

    def update_voice_status(self, is_active: bool):
        """Update the voice status indicator."""
        self.is_voice_active = is_active
        
        if is_active:
            self.voice_button.text = "🔴"  # Recording
            self.voice_button.style.background_color = "#ffebee"
        else:
            self.voice_button.text = "🎤"  # Ready
            self.voice_button.style.background_color = "#e3f2fd"

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)

    def clear_messages(self):
        """Clear all messages."""
        self.messages.clear()
        
        # Clear the UI
        self.messages_container.clear()
        
        # Re-add welcome message
        self._add_welcome_message()

    def show_typing_indicator(self, show: bool = True):
        """Show or hide typing indicator."""
        if show:
            self.add_message("system", "🤖 Tektra is thinking...")
        # Note: In a real implementation, this would show an animated typing indicator


class ChatManager:
    """
    Manages chat conversations and integrates with AI backends.
    
    This class handles the business logic for conversations, including
    routing messages to appropriate AI models and managing conversation state.
    """

    def __init__(self, chat_panel: ChatPanel):
        """
        Initialize the chat manager.
        
        Args:
            chat_panel: The chat panel to manage
        """
        self.chat_panel = chat_panel
        self.conversation_id = "main"
        self.message_history = []
        
        logger.info("Chat manager initialized")

    async def process_user_message(self, message: str, smart_router, context: dict = None):
        """
        Process a user message and generate a response with memory enhancement.
        
        Args:
            message: The user's message
            smart_router: The smart router for AI processing
            context: Additional context for the message
        """
        try:
            # Show typing indicator
            self.chat_panel.show_typing_indicator(True)
            
            # Add to message history
            self.message_history.append({"role": "user", "content": message})
            
            # Get conversation memory from the main app if available
            conversation_memory = getattr(smart_router, 'conversation_memory', None)
            
            # Enhanced context with memory
            enhanced_context = context or {}
            
            if conversation_memory:
                # Get relevant context from MemOS
                memory_context = await conversation_memory.get_relevant_context(
                    query=message,
                    max_memories=5,
                    include_recent=True
                )
                
                if memory_context:
                    enhanced_context["memory_context"] = memory_context
                    logger.debug(f"Added memory context: {len(memory_context)} chars")
            
            # Process through smart router
            if smart_router:
                response = await smart_router.process_message(
                    message, 
                    context=enhanced_context,
                    conversation_history=self.message_history
                )
                
                # Add response to chat
                self.chat_panel.add_message("assistant", response)
                
                # Add to message history
                self.message_history.append({"role": "assistant", "content": response})
                
                # Store conversation turn in memory
                if conversation_memory:
                    await conversation_memory.add_conversation_turn(
                        user_message=message,
                        assistant_response=response,
                        context=enhanced_context
                    )
                    logger.debug("Stored conversation turn in MemOS")
                    
            else:
                # Fallback response
                response = "I'm still initializing my AI models. Please wait a moment and try again."
                self.chat_panel.add_message("assistant", response)
                
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            error_msg = f"I encountered an error processing your message: {e}"
            self.chat_panel.add_message("assistant", error_msg)

    async def handle_voice_transcription(self, text: str):
        """
        Handle voice transcription from the voice pipeline.
        
        Args:
            text: The transcribed text
        """
        # Add as user message
        self.chat_panel.add_message("user", f"🎤 {text}")
        
        # Add to message history
        self.message_history.append({"role": "user", "content": text})

    async def handle_voice_response(self, text: str):
        """
        Handle voice response from the voice pipeline.
        
        Args:
            text: The response text
        """
        # Add as assistant message
        self.chat_panel.add_message("assistant", f"🔊 {text}")
        
        # Add to message history
        self.message_history.append({"role": "assistant", "content": text})

    async def handle_file_processed(self, filename: str, status: str):
        """
        Handle file processing completion.
        
        Args:
            filename: The processed file name
            status: The processing status
        """
        if status == "success":
            msg = f"📄 I've processed '{filename}' and it's ready for analysis."
        else:
            msg = f"❌ Failed to process '{filename}': {status}"
        
        self.chat_panel.add_message("system", msg)

    def get_conversation_history(self) -> list:
        """Get the current conversation history."""
        return self.message_history.copy()

    def clear_conversation(self):
        """Clear the conversation history."""
        self.message_history.clear()
        self.chat_panel.clear_messages()