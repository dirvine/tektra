"""
Chat Panel Component

This module provides the main chat interface for Tektra AI Assistant,
handling message display, input processing, and interaction with the
smart router and voice systems.
"""

import asyncio
import time
from collections.abc import Callable
from typing import Any

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class MessageBubble:
    """Individual message bubble component."""

    def __init__(self, role: str, content: str, timestamp: float | None = None):
        """
        Initialize message bubble.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message text content
            timestamp: Message timestamp (defaults to current time)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
        self.widget = self._create_bubble()

    def _create_bubble(self) -> toga.Box:
        """Create the message bubble widget."""
        # Main container for the message
        message_container = toga.Box(
            style=Pack(
                direction=ROW,
                margin=(5, 10),
                align_items="start" if self.role == "assistant" else "end",
            )
        )

        # Message bubble styling based on role
        if self.role == "user":
            bubble_style = Pack(
                margin=(10, 15),
                background_color="#2196f3",
                color="#ffffff",
                font_size=14,
                text_align="right",
            )
            # Add spacer on left for right alignment
            spacer = toga.Box(style=Pack(flex=1))
            message_container.add(spacer)
        elif self.role == "assistant":
            bubble_style = Pack(
                margin=(10, 15),
                background_color="#e3f2fd",
                color="#1976d2",
                font_size=14,
                text_align="left",
            )
        else:  # system messages
            bubble_style = Pack(
                margin=(8, 12),
                background_color="#f5f5f5",
                color="#666666",
                font_size=12,
                font_style="italic",
                text_align="center",
            )

        # Create message label
        message_label = toga.Label(self.content, style=bubble_style)

        # Wrapper for styling (border radius simulation)
        bubble_wrapper = toga.Box(
            style=Pack(
                direction=COLUMN,
                width=600 if self.role in ["user", "assistant"] else 400,
            )
        )
        bubble_wrapper.add(message_label)

        message_container.add(bubble_wrapper)

        # Add spacer on right for left alignment (assistant messages)
        if self.role == "assistant":
            spacer = toga.Box(style=Pack(flex=1))
            message_container.add(spacer)

        return message_container


class ChatPanel:
    """
    Main chat interface panel for Tektra AI Assistant.

    Provides message display, input handling, and integration with
    voice and AI processing systems.
    """

    def __init__(
        self,
        on_message_send: Callable[[str], None] | None = None,
        on_voice_toggle: Callable[[], None] | None = None,
        on_file_upload: Callable[[], None] | None = None,
    ):
        """
        Initialize chat panel.

        Args:
            on_message_send: Callback for when user sends a message
            on_voice_toggle: Callback for voice input toggle
            on_file_upload: Callback for file upload requests
        """
        self.on_message_send = on_message_send
        self.on_voice_toggle = on_voice_toggle
        self.on_file_upload = on_file_upload

        # Message storage
        self.messages: list[MessageBubble] = []
        self.conversation_history: list[dict[str, str]] = []

        # UI state
        self.is_voice_enabled = False
        self.is_processing = False
        self.current_typing_indicator = None

        # Create the main panel widget
        self.widget = self._create_panel()

        # Add welcome message
        self._add_welcome_message()

    def _create_panel(self) -> toga.Box:
        """Create the main chat panel widget."""
        # Main chat container
        chat_container = toga.Box(style=Pack(direction=COLUMN, flex=1, margin=10))

        # Chat display area (scrollable)
        self.chat_display = toga.ScrollContainer(
            style=Pack(flex=1, background_color="#ffffff", margin=10)
        )

        # Messages container inside scroll area
        self.messages_container = toga.Box(
            style=Pack(direction=COLUMN, align_items="start")
        )
        self.chat_display.content = self.messages_container

        chat_container.add(self.chat_display)

        # Input area
        input_area = self._create_input_area()
        chat_container.add(input_area)

        return chat_container

    def _create_input_area(self) -> toga.Box:
        """Create the message input area."""
        input_container = toga.Box(
            style=Pack(direction=ROW, margin_top=10, align_items="center")
        )

        # Text input field
        self.message_input = toga.TextInput(
            placeholder="Type your message here...",
            style=Pack(flex=1, margin_right=10, font_size=14),
            on_change=self._on_input_change,
        )
        input_container.add(self.message_input)

        # Voice input button
        self.voice_btn = toga.Button(
            "🎤",
            on_press=self._on_voice_toggle,
            style=Pack(width=50, margin_right=5, font_size=16),
            enabled=False,
        )
        input_container.add(self.voice_btn)

        # File upload button
        self.file_btn = toga.Button(
            "📎",
            on_press=self._on_file_upload,
            style=Pack(width=50, margin_right=5, font_size=16),
            enabled=True,  # Enable file upload immediately
        )
        input_container.add(self.file_btn)

        # Send button
        self.send_btn = toga.Button(
            "Send",
            on_press=self._on_send_message,
            style=Pack(width=80, background_color="#2196f3", color="#ffffff"),
            enabled=False,
        )
        input_container.add(self.send_btn)

        return input_container

    def _add_welcome_message(self):
        """Add initial welcome message."""
        welcome_text = """Welcome to Tektra AI Assistant! 🤖

I'm your voice-interactive AI assistant powered by:
• Kyutai Unmute for natural voice conversations
• Qwen 2.5-VL for complex analysis and vision tasks
• Smart routing for optimal AI selection

You can:
💬 Type messages for text chat
🎤 Use voice mode for spoken conversation
📷 Enable camera for vision analysis
📎 Upload files for document analysis

How can I help you today?"""

        self.add_message("assistant", welcome_text)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat display.

        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
        # Create message bubble
        message_bubble = MessageBubble(role, content)
        self.messages.append(message_bubble)

        # Add to UI
        self.messages_container.add(message_bubble.widget)

        # Update conversation history
        self.conversation_history.append({"role": role, "content": content})

        # Scroll to bottom
        self._scroll_to_bottom()

        logger.debug(f"Added {role} message: {content[:50]}...")

    def add_typing_indicator(self) -> None:
        """Show typing indicator for assistant."""
        if self.current_typing_indicator:
            return

        typing_bubble = MessageBubble("assistant", "Thinking...")
        self.current_typing_indicator = typing_bubble
        self.messages_container.add(typing_bubble.widget)
        self._scroll_to_bottom()

    def remove_typing_indicator(self) -> None:
        """Remove typing indicator."""
        if self.current_typing_indicator:
            try:
                self.messages_container.remove(self.current_typing_indicator.widget)
            except Exception as e:
                logger.debug(f"Failed to remove typing indicator widget: {e}")
            self.current_typing_indicator = None

    def clear_messages(self) -> None:
        """Clear all messages from the chat."""
        # Clear UI
        for message in self.messages:
            try:
                self.messages_container.remove(message.widget)
            except Exception as e:
                logger.debug(f"Failed to remove message widget: {e}")

        # Clear data
        self.messages.clear()
        self.conversation_history.clear()

        # Re-add welcome message
        self._add_welcome_message()

    def set_processing_state(self, is_processing: bool) -> None:
        """Set the processing state of the chat."""
        self.is_processing = is_processing

        # Update UI state
        self.send_btn.enabled = not is_processing and bool(
            self.message_input.value.strip()
        )
        self.message_input.enabled = not is_processing

        if is_processing:
            self.add_typing_indicator()
        else:
            self.remove_typing_indicator()

    def enable_voice_features(self, enabled: bool) -> None:
        """Enable or disable voice-related features."""
        self.is_voice_enabled = enabled
        self.voice_btn.enabled = enabled

    def enable_file_features(self, enabled: bool) -> None:
        """Enable or disable file upload features."""
        self.file_btn.enabled = enabled

    def update_voice_status(self, is_listening: bool) -> None:
        """Update voice button status."""
        if is_listening:
            self.voice_btn.text = "🔴"  # Recording indicator
            self.voice_btn.style.background_color = "#ff5722"
        else:
            self.voice_btn.text = "🎤"
            self.voice_btn.style.background_color = None

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)

    def _scroll_to_bottom(self) -> None:
        """Scroll chat display to the bottom."""
        try:
            # Attempt to scroll to bottom if the scroll container supports it
            if hasattr(self.chat_scroll, 'vertical_position'):
                # Set to maximum scroll position to show latest messages
                self.chat_scroll.vertical_position = 1.0
            
            # Alternative: If scroll container has content height properties
            elif hasattr(self.chat_scroll, 'max_vertical_scroll'):
                self.chat_scroll.vertical_position = self.chat_scroll.max_vertical_scroll
            
            # Log the attempt since Toga's scrolling behavior varies by platform
            logger.debug("Attempted to scroll chat to bottom")
            
        except Exception as e:
            # Toga scrolling capabilities vary by platform, so handle gracefully
            logger.debug(f"Auto-scroll not supported on this platform: {e}")

    def _on_input_change(self, widget) -> None:
        """Handle text input changes."""
        has_text = bool(widget.value.strip())
        self.send_btn.enabled = has_text and not self.is_processing

    async def _on_send_message(self, widget) -> None:
        """Handle send button press."""
        message_text = self.message_input.value.strip()
        if not message_text or self.is_processing:
            return

        # Clear input
        self.message_input.value = ""
        self.send_btn.enabled = False

        # Add user message to display
        self.add_message("user", message_text)

        # Set processing state
        self.set_processing_state(True)

        # Call callback
        if self.on_message_send:
            try:
                if asyncio.iscoroutinefunction(self.on_message_send):
                    await self.on_message_send(message_text)
                else:
                    self.on_message_send(message_text)
            except Exception as e:
                logger.error(f"Error in message send callback: {e}")
                self.add_message("system", f"Error sending message: {e}")
            finally:
                self.set_processing_state(False)

    async def _on_voice_toggle(self, widget) -> None:
        """Handle voice toggle button press."""
        if self.on_voice_toggle:
            try:
                if asyncio.iscoroutinefunction(self.on_voice_toggle):
                    await self.on_voice_toggle()
                else:
                    self.on_voice_toggle()
            except Exception as e:
                logger.error(f"Error in voice toggle callback: {e}")

    async def _on_file_upload(self, widget) -> None:
        """Handle file upload button press."""
        try:
            # Show file upload dialog
            await self._show_file_upload_dialog()
        except Exception as e:
            logger.error(f"Error in file upload: {e}")
            self.add_message("system", f"File upload error: {e}")

    async def _show_file_upload_dialog(self) -> None:
        """Show file upload dialog."""
        try:
            # Import here to avoid circular imports
            from ..ai.multimodal import MultimodalProcessor
            from .file_upload import FileUploadPanel

            # Create multimodal processor instance
            processor = MultimodalProcessor()

            # Create upload panel
            FileUploadPanel(
                on_files_selected=self._on_files_selected_for_chat,
                on_file_processed=self._on_file_processed_for_chat,
                multimodal_processor=processor,
            )

            # For now, trigger file processing directly
            # In a full implementation, this would show a modal dialog
            self.add_message(
                "system",
                "📎 File upload feature ready! Click 'Choose Files' to select files for analysis.",
            )

            # Trigger the file upload callback if provided
            if self.on_file_upload:
                if asyncio.iscoroutinefunction(self.on_file_upload):
                    await self.on_file_upload()
                else:
                    self.on_file_upload()

        except Exception as e:
            logger.error(f"Error showing file upload dialog: {e}")
            self.add_message("system", f"Error: {e}")

    async def _on_files_selected_for_chat(self, file_paths) -> None:
        """Handle files selected for chat analysis."""
        file_names = [f.name for f in file_paths]
        self.add_message("system", f"📁 Selected files: {', '.join(file_names)}")
        logger.info(f"Files selected for chat: {file_names}")

    async def _on_file_processed_for_chat(self, file_path, result) -> None:
        """Handle file processed for chat analysis."""
        status = result.get("processing_status", "unknown")
        content_type = result.get("content_type", "unknown")

        if status == "success":
            self.add_message(
                "system", f"✅ Processed {file_path.name} ({content_type})"
            )

            # Add the processed result to conversation context for AI analysis
            await self._add_file_to_conversation_context(file_path, result)
        else:
            error_msg = result.get("error", "Unknown error")
            self.add_message(
                "system", f"❌ Failed to process {file_path.name}: {error_msg}"
            )

    async def _add_file_to_conversation_context(self, file_path, result) -> None:
        """Add processed file to conversation context for AI."""
        try:
            content_type = result.get("content_type")

            if content_type == "image":
                # For images, we have the processed image and metadata
                image_data = result.get("image")
                analysis = result.get("analysis", {})

                context_message = (
                    f"📷 Image '{file_path.name}' uploaded and ready for analysis.\n"
                )
                context_message += (
                    f"Dimensions: {analysis.get('dimensions', 'unknown')}\n"
                )
                context_message += (
                    f"File size: {self._format_file_size(file_path.stat().st_size)}"
                )

                self.add_message("assistant", context_message)

                # Store image data for vision analysis
                if not hasattr(self, "uploaded_files"):
                    self.uploaded_files = {}

                self.uploaded_files[str(file_path)] = {
                    "type": "image",
                    "data": image_data,
                    "metadata": analysis,
                    "file_path": file_path,
                }

            elif content_type == "document":
                # For documents, we have extracted text
                text_content = result.get("content", "")
                word_count = len(text_content.split()) if text_content else 0

                context_message = f"📄 Document '{file_path.name}' processed.\n"
                context_message += f"Word count: {word_count}\n"

                if word_count > 0:
                    # Show preview of content
                    preview = (
                        text_content[:200] + "..."
                        if len(text_content) > 200
                        else text_content
                    )
                    context_message += f"Preview: {preview}"

                self.add_message("assistant", context_message)

                # Store document data for analysis
                if not hasattr(self, "uploaded_files"):
                    self.uploaded_files = {}

                self.uploaded_files[str(file_path)] = {
                    "type": "document",
                    "content": text_content,
                    "metadata": result.get("metadata", {}),
                    "file_path": file_path,
                }

            # Update conversation history to include file context
            self.conversation_history.append(
                {
                    "role": "system",
                    "content": f"File uploaded: {file_path.name} ({content_type})",
                    "file_data": result,
                }
            )

        except Exception as e:
            logger.error(f"Error adding file to conversation context: {e}")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

    def get_uploaded_files(self) -> dict:
        """Get all uploaded files data."""
        return getattr(self, "uploaded_files", {})


class ChatManager:
    """
    Manager class for chat functionality integration.

    Provides higher-level chat management and integration
    with AI backends and voice systems.
    """

    def __init__(self, chat_panel: ChatPanel):
        """Initialize chat manager with chat panel."""
        self.chat_panel = chat_panel
        self.message_queue = asyncio.Queue()
        self.is_processing = False

    async def process_user_message(
        self, message: str, smart_router, context: dict[str, Any] | None = None
    ) -> None:
        """
        Process user message through smart router.

        Args:
            message: User message text
            smart_router: Smart router instance
            context: Optional context information
        """
        try:
            self.is_processing = True

            # Prepare query context
            from ..ai.smart_router import QueryContext

            # Check for uploaded files that can be analyzed
            uploaded_files = self.chat_panel.get_uploaded_files()
            has_image = False
            image_data = None
            file_attachments = []

            # Look for images in uploaded files
            for file_path, file_info in uploaded_files.items():
                if file_info["type"] == "image":
                    has_image = True
                    image_data = file_info["data"]

                file_attachments.append(
                    {
                        "content_type": file_info["type"],
                        "file_path": file_path,
                        "data": file_info.get("data") or file_info.get("content"),
                        "metadata": file_info.get("metadata", {}),
                    }
                )

            query_context = QueryContext(
                query_text=message,
                has_image=has_image,
                image_data=image_data,
                file_attachments=file_attachments,
                conversation_history=self.chat_panel.get_conversation_history(),
                is_voice_input=False,
                session_context=context or {},
            )

            # Route through smart router
            result = await smart_router.route_query(query_context)

            # Handle response based on routing decision
            route = result["routing_decision"].route

            if route.value in ["qwen_analytical", "qwen_vision"]:
                # Synchronous response from Qwen
                response = result.get("response", "No response received")
                self.chat_panel.add_message("assistant", response)

            # Async responses (Unmute) are handled via callbacks

        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            self.chat_panel.add_message(
                "assistant", f"Sorry, I encountered an error: {e}"
            )
        finally:
            self.is_processing = False

    async def handle_voice_transcription(self, text: str) -> None:
        """Handle voice transcription result."""
        self.chat_panel.add_message("user", f"🎤 {text}")

    async def handle_voice_response(self, text: str) -> None:
        """Handle voice response from AI."""
        self.chat_panel.add_message("assistant", f"🔊 {text}")

    async def handle_file_processed(self, filename: str, status: str) -> None:
        """Handle file processing result."""
        if status == "success":
            self.chat_panel.add_message("system", f"📎 Uploaded: {filename}")
        else:
            self.chat_panel.add_message("system", f"❌ Failed to process: {filename}")
