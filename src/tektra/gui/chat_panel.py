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
from .message_bubble_renderer import get_message_bubble_renderer
from .animations.animation_manager import AnimationManager
from .typing_indicator import TypingIndicatorManager
from .enhanced_input_field import EnhancedInputField
from .smooth_scroll_container import SmoothScrollContainer, ConversationScrollManager
from .virtual_scroll_manager import VirtualScrollManager
from .scroll_performance_optimizer import ScrollPerformanceOptimizer, ScrollFrameRateMonitor


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
        
        # Initialize renderers and animation system
        self.animation_manager = AnimationManager()
        self.markdown_renderer = get_markdown_renderer()
        self.message_bubble_renderer = get_message_bubble_renderer(self.animation_manager)
        self.typing_indicator_manager = TypingIndicatorManager(self.animation_manager)
        
        # Initialize scroll performance optimization
        self.scroll_performance_optimizer = ScrollPerformanceOptimizer(
            target_fps=60.0,
            performance_window=30,
            optimization_threshold=45.0
        )
        self.frame_rate_monitor = ScrollFrameRateMonitor(target_fps=60.0)
        
        # Scroll management components (will be initialized in _build_messages_area)
        self.smooth_scroll_container = None
        self.conversation_scroll_manager = None
        self.virtual_scroll_manager = None
        
        # Start animation performance monitoring (only if event loop is running)
        try:
            asyncio.create_task(self.animation_manager.start_performance_monitoring())
        except RuntimeError:
            # No event loop running, performance monitoring will start later
            logger.debug("No event loop running, performance monitoring will start later")
        
        # Build the UI
        self.widget = self._build_interface()
        
        logger.info("Chat panel initialized with enhanced message rendering")

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
        """Build the scrollable messages area with enhanced smooth scrolling."""
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
        
        # Create and add typing indicator
        self.typing_indicator = self.typing_indicator_manager.create_indicator("main")
        self.messages_container.add(self.typing_indicator.widget)
        
        # Create the base scroll container
        base_scroll_container = toga.ScrollContainer(
            content=self.messages_container,
            style=Pack(
                flex=1,
                background_color=colors.surface
            )
        )
        
        # Initialize smooth scroll container with enhanced features
        self.smooth_scroll_container = SmoothScrollContainer(
            content=self.messages_container,
            animation_manager=self.animation_manager,
            style=Pack(flex=1, background_color=colors.surface),
            enable_momentum=True,
            momentum_decay=0.95,
            auto_scroll_threshold=50.0
        )
        
        # Initialize conversation scroll manager for chat-specific behaviors
        self.conversation_scroll_manager = ConversationScrollManager(
            smooth_scroll_container=self.smooth_scroll_container,
            animation_manager=self.animation_manager
        )
        
        # Initialize virtual scroll manager for large conversations
        self.virtual_scroll_manager = VirtualScrollManager(
            container=base_scroll_container,
            item_renderer=self._render_virtual_message,
            estimated_item_height=100,
            buffer_size=5,
            performance_threshold=100  # Enable virtual scrolling after 100 messages
        )
        
        # Set up performance optimization callbacks
        self._setup_performance_optimization()
        
        # Use the base scroll container as the main widget for now
        # (The smooth scroll container will be integrated more deeply in a full implementation)
        self.messages_scroll = base_scroll_container
        
        # Add welcome message
        self._add_welcome_message()

    def _setup_performance_optimization(self):
        """Set up performance optimization callbacks and monitoring."""
        # Add optimization callbacks
        self.scroll_performance_optimizer.add_optimization_callback(
            "reduce_animations", self._reduce_animation_complexity
        )
        self.scroll_performance_optimizer.add_optimization_callback(
            "enable_virtual_scrolling", self._enable_virtual_scrolling_mode
        )
        self.scroll_performance_optimizer.add_optimization_callback(
            "batch_render", self._enable_batch_rendering
        )
        
        # Start performance monitoring loop
        asyncio.create_task(self._performance_monitoring_loop())
        
        logger.debug("Performance optimization callbacks set up")

    def _render_virtual_message(self, message_data: dict) -> toga.Widget:
        """
        Render a message for virtual scrolling.
        
        Args:
            message_data: Message data dictionary
            
        Returns:
            Rendered message widget
        """
        try:
            # Use the enhanced message bubble renderer
            return self.message_bubble_renderer.render_message_bubble(
                message_data, 
                theme=theme_manager.get_theme(),
                animate=False  # No animations in virtual mode for performance
            )
        except Exception as e:
            logger.error(f"Error rendering virtual message: {e}")
            # Fallback to simple rendering
            return self._render_simple_message_widget(message_data)

    def _render_simple_message_widget(self, message: dict) -> toga.Widget:
        """
        Render a simple message widget for fallback cases.
        
        Args:
            message: Message data dictionary
            
        Returns:
            Simple message widget
        """
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        # Simple container
        container = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["sm"],
                margin_bottom=spacing["xs"],
                background_color=colors.surface
            )
        )
        
        # Simple label
        role_prefix = {
            "user": "👤 You: ",
            "assistant": "🤖 Tektra: ",
            "system": "ℹ️ System: "
        }.get(message["role"], "")
        
        label = toga.Label(
            f"{role_prefix}{message['content']}",
            style=Pack(
                font_size=14,
                color=colors.text_primary
            )
        )
        container.add(label)
        
        return container

    async def _performance_monitoring_loop(self):
        """Monitor performance and record metrics."""
        try:
            while True:
                # Record frame metrics
                frame_info = self.frame_rate_monitor.record_frame()
                
                # Record metrics in the performance optimizer
                self.scroll_performance_optimizer.record_frame_metrics(
                    frame_time=frame_info["frame_time"] / 1000.0,  # Convert ms to seconds
                    rendered_items=len(self.messages),
                    scroll_velocity=0.0  # Would be actual scroll velocity in full implementation
                )
                
                # Sleep for next frame
                await asyncio.sleep(1.0 / 60.0)  # 60fps monitoring
                
        except asyncio.CancelledError:
            logger.debug("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in performance monitoring loop: {e}")

    def _reduce_animation_complexity(self):
        """Reduce animation complexity for better performance."""
        if self.animation_manager:
            self.animation_manager.set_reduced_motion(True)
            logger.debug("Reduced animation complexity for performance")

    def _enable_virtual_scrolling_mode(self):
        """Enable virtual scrolling mode for large conversations."""
        if self.virtual_scroll_manager and len(self.messages) > 50:
            # Transfer existing messages to virtual scroll manager
            for message in self.messages:
                self.virtual_scroll_manager.add_item(message)
            logger.debug("Enabled virtual scrolling mode")

    def _enable_batch_rendering(self):
        """Enable batch rendering for better performance."""
        # This would implement batched message rendering
        # For now, it's a placeholder
        logger.debug("Enabled batch rendering mode")

    def _build_input_area(self) -> toga.Box:
        """Build the enhanced input area at the bottom."""
        # Create the enhanced input field
        self.enhanced_input = EnhancedInputField(
            animation_manager=self.animation_manager,
            on_message_send=self._handle_enhanced_message_send,
            on_voice_toggle=self._handle_enhanced_voice_toggle,
            on_file_upload=self._handle_enhanced_file_upload,
            placeholder="Type your message...",
            max_characters=4000
        )
        
        # Store references to the internal components for compatibility
        self.text_input = self.enhanced_input.text_input
        self.send_button = self.enhanced_input.send_button
        self.voice_button = self.enhanced_input.voice_button
        self.file_button = self.enhanced_input.file_button
        
        return self.enhanced_input.widget

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
        Add a message to the chat with enhanced scrolling behavior.
        
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
        
        # Add to virtual scroll manager if enabled
        if self.virtual_scroll_manager and len(self.messages) > self.virtual_scroll_manager.performance_threshold:
            self.virtual_scroll_manager.add_item(message)
        else:
            # Render normally for smaller conversations
            self._render_message(message)
        
        # Handle scrolling with conversation scroll manager
        asyncio.create_task(self._handle_new_message_scroll(role))
        
        logger.debug(f"Added {role} message: {content[:50]}...")

    async def _handle_new_message_scroll(self, role: str):
        """
        Handle scrolling behavior when a new message is added.
        
        Args:
            role: Role of the message sender
        """
        try:
            if self.conversation_scroll_manager:
                # Use the conversation scroll manager for intelligent scrolling
                await self.conversation_scroll_manager.on_new_message(role, animate=True)
            else:
                # Fallback to basic scroll to bottom
                self._scroll_to_bottom()
        except Exception as e:
            logger.error(f"Error handling new message scroll: {e}")

    def _render_message(self, message: dict):
        """Render a single message using the enhanced message bubble renderer."""
        try:
            # Use the enhanced message bubble renderer
            message_widget = self.message_bubble_renderer.render_message_bubble(
                message, 
                theme=theme_manager.get_theme(),
                animate=True
            )
            
            # Add to messages container
            self.messages_container.add(message_widget)
            
        except Exception as e:
            logger.error(f"Error rendering message with enhanced renderer: {e}")
            # Fallback to simple rendering
            self._render_message_fallback(message)
    
    def _render_message_fallback(self, message: dict):
        """Fallback message rendering for error cases."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        
        # Simple container
        container = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=spacing["sm"],
                margin_bottom=spacing["xs"],
                background_color=colors.surface
            )
        )
        
        # Simple label
        role_prefix = {
            "user": "👤 You: ",
            "assistant": "🤖 Tektra: ",
            "system": "ℹ️ System: "
        }.get(message["role"], "")
        
        label = toga.Label(
            f"{role_prefix}{message['content']}",
            style=Pack(
                font_size=14,
                color=colors.text_primary
            )
        )
        container.add(label)
        
        # Add to messages container
        self.messages_container.add(container)

    def _scroll_to_bottom(self):
        """Scroll the messages area to the bottom with smooth animation."""
        try:
            # Use the conversation scroll manager for intelligent scrolling
            if self.conversation_scroll_manager:
                # This will be handled by the conversation scroll manager
                # when new messages are added
                pass
            elif self.smooth_scroll_container:
                # Fallback to direct smooth scroll container
                self.smooth_scroll_container.scroll_to_bottom(smooth=True)
            else:
                # Final fallback - Toga doesn't have direct scroll control
                # This would need platform-specific implementation
                logger.debug("Scroll to bottom requested (limited Toga support)")
        except Exception as e:
            logger.error(f"Error scrolling to bottom: {e}")

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

    # Enhanced input field handlers
    async def _handle_enhanced_message_send(self, message: str):
        """Handle message sending from enhanced input field."""
        # Add user message
        self.add_message("user", message)
        
        # Send to handler
        if self.on_message_send:
            await self._handle_message_send(message)

    async def _handle_enhanced_voice_toggle(self):
        """Handle voice toggle from enhanced input field."""
        await self._handle_voice_toggle()

    async def _handle_enhanced_file_upload(self):
        """Handle file upload from enhanced input field."""
        await self._handle_file_upload()

    def enable_voice_features(self, enabled: bool):
        """Enable or disable voice features."""
        self.voice_enabled = enabled
        
        # Enable in the enhanced input field
        if hasattr(self, 'enhanced_input'):
            self.enhanced_input.enable_voice_features(enabled)
        else:
            # Fallback for compatibility
            self.voice_button.enabled = enabled
            if enabled:
                self.voice_button.style.background_color = "#e3f2fd"
            else:
                self.voice_button.style.background_color = "#eeeeee"
        
        if enabled:
            self.add_message("system", "🎤 Voice features enabled! Click the microphone to start talking.")

    def enable_file_features(self, enabled: bool):
        """Enable or disable file upload features."""
        self.file_enabled = enabled
        
        # Enable in the enhanced input field
        if hasattr(self, 'enhanced_input'):
            self.enhanced_input.enable_file_features(enabled)
        else:
            # Fallback for compatibility
            self.file_button.enabled = enabled
            if enabled:
                self.file_button.style.background_color = "#e8f5e8"
            else:
                self.file_button.style.background_color = "#eeeeee"
        
        if enabled:
            self.add_message("system", "📎 File upload enabled! Click the paperclip to analyze files.")

    def update_voice_status(self, is_active: bool):
        """Update the voice status indicator."""
        self.is_voice_active = is_active
        
        # Update through enhanced input field if available
        if hasattr(self, 'enhanced_input'):
            # The enhanced input field handles its own voice status updates
            # We just need to trigger the toggle if needed
            if is_active != self.enhanced_input.is_voice_recording:
                asyncio.create_task(self.enhanced_input._toggle_voice_recording())
        else:
            # Fallback for compatibility
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

    async def show_typing_indicator(self, show: bool = True, message: str = "Tektra is thinking..."):
        """
        Show or hide the animated typing indicator.
        
        Args:
            show: Whether to show or hide the indicator
            message: Custom message to display with the indicator
        """
        try:
            if show:
                await self.typing_indicator_manager.show_indicator(
                    indicator_id="main",
                    message=message,
                    delay=0.1  # Small delay to prevent flicker on quick responses
                )
                logger.debug(f"Typing indicator shown: {message}")
            else:
                await self.typing_indicator_manager.hide_indicator(
                    indicator_id="main",
                    delay=0.0  # Hide immediately
                )
                logger.debug("Typing indicator hidden")
        except Exception as e:
            logger.error(f"Error controlling typing indicator: {e}")
            # Fallback: add a simple system message
            if show:
                self.add_message("system", f"🤖 {message}")


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
            await self.chat_panel.show_typing_indicator(True)
            
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
                
                # Hide typing indicator before showing response
                await self.chat_panel.show_typing_indicator(False)
                
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
                # Hide typing indicator before showing fallback response
                await self.chat_panel.show_typing_indicator(False)
                
                # Fallback response
                response = "I'm still initializing my AI models. Please wait a moment and try again."
                self.chat_panel.add_message("assistant", response)
                
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            # Hide typing indicator on error
            await self.chat_panel.show_typing_indicator(False)
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