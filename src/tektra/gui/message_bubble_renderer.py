"""
Enhanced Message Bubble Renderer

This module provides advanced message rendering with animations, improved styling,
and role-based visual treatments for a modern conversational UI experience.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Callable, Any, List
from loguru import logger

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .themes import theme_manager, Theme
from .markdown_renderer import get_markdown_renderer
from .animations.animation_manager import AnimationManager


class MessageBubbleRenderer:
    """
    Advanced message rendering with animations and rich formatting.
    
    Features:
    - Role-based styling (user vs assistant vs system)
    - Smooth appearance animations
    - Enhanced markdown rendering
    - Modern chat app conventions
    - Accessibility support
    """
    
    def __init__(self, animation_manager: Optional[AnimationManager] = None):
        """
        Initialize the message bubble renderer.
        
        Args:
            animation_manager: Animation manager for smooth transitions
        """
        self.animation_manager = animation_manager
        self.markdown_renderer = get_markdown_renderer()
        self.message_cache: Dict[str, toga.Widget] = {}
        
        # Track interactive elements for micro-interactions
        self.interactive_elements: Dict[str, str] = {}  # widget_id -> element_id mapping
        
        logger.info("Message Bubble Renderer initialized")
    
    def render_message_bubble(self, message: dict, theme: Optional[Theme] = None,
                            animate: bool = True) -> toga.Box:
        """
        Render a complete message bubble with role-based styling.
        
        Args:
            message: Message dictionary with role, content, timestamp, etc.
            theme: Theme to use (defaults to current theme)
            animate: Whether to animate the message appearance
            
        Returns:
            Complete message bubble widget
        """
        if theme is None:
            theme = theme_manager.get_theme()
        
        role = message.get("role", "assistant")
        content = message.get("content", "")
        timestamp = message.get("timestamp", datetime.now())
        message_id = message.get("id", f"{role}_{timestamp.timestamp()}")
        
        # Check cache first
        if message_id in self.message_cache:
            return self.message_cache[message_id]
        
        # Create the main container for alignment
        alignment_container = self._create_alignment_container(role, theme)
        
        # Create the message bubble
        message_bubble = self._create_message_bubble(message, theme)
        
        # Add bubble to alignment container
        if role == "user":
            # User messages: spacer on left, bubble on right
            spacer = toga.Box(style=Pack(flex=1))
            alignment_container.add(spacer)
            alignment_container.add(message_bubble)
        elif role == "assistant":
            # Assistant messages: bubble on left, spacer on right
            alignment_container.add(message_bubble)
            spacer = toga.Box(style=Pack(flex=1))
            alignment_container.add(spacer)
        else:
            # System messages: centered
            left_spacer = toga.Box(style=Pack(flex=1))
            right_spacer = toga.Box(style=Pack(flex=1))
            alignment_container.add(left_spacer)
            alignment_container.add(message_bubble)
            alignment_container.add(right_spacer)
        
        # Cache the result
        self.message_cache[message_id] = alignment_container
        
        # Animate appearance if requested
        if animate and self.animation_manager:
            try:
                asyncio.create_task(self._animate_message_appearance(alignment_container, role))
            except RuntimeError:
                # No event loop running, skip animation
                logger.debug("No event loop running, skipping message animation")
        
        return alignment_container
    
    def _create_alignment_container(self, role: str, theme: Theme) -> toga.Box:
        """Create the outer container for message alignment with modern chat conventions."""
        spacing = theme.spacing
        
        # Different spacing based on role for better visual hierarchy
        if role == "user":
            # User messages: more space from left, less from right
            margin_bottom = spacing["sm"]
            padding = (0, spacing["sm"], 0, spacing["xl"])
        elif role == "assistant":
            # Assistant messages: more space from right, less from left
            margin_bottom = spacing["sm"]
            padding = (0, spacing["xl"], 0, spacing["sm"])
        else:
            # System messages: centered with equal padding
            margin_bottom = spacing["xs"]  # Less space for system messages
            padding = (0, spacing["lg"])
        
        return toga.Box(
            style=Pack(
                direction=ROW,
                flex=1,
                margin_bottom=margin_bottom,
                padding=padding
            )
        )
    
    def _create_message_bubble(self, message: dict, theme: Theme) -> toga.Box:
        """Create the actual message bubble with content."""
        role = message.get("role", "assistant")
        content = message.get("content", "")
        timestamp = message.get("timestamp", datetime.now())
        
        # Get role-specific styling
        bubble_style = self._get_bubble_style(role, theme)
        
        # Create bubble container
        bubble_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                **bubble_style
            )
        )
        
        # Add header (role and timestamp) for user and assistant messages
        if role in ["user", "assistant"]:
            header = self._create_message_header(role, timestamp, theme)
            if header:
                bubble_container.add(header)
        
        # Add message content
        content_widget = self._create_message_content(content, role, theme)
        bubble_container.add(content_widget)
        
        # Add status indicators if needed
        status_widget = self._create_status_indicators(message, role, theme)
        if status_widget:
            bubble_container.add(status_widget)
        
        return bubble_container
    
    def _get_bubble_style(self, role: str, theme: Theme) -> Dict[str, Any]:
        """Get styling for message bubble based on role with modern chat app conventions."""
        colors = theme.colors
        spacing = theme.spacing
        borders = theme.borders
        
        # Modern chat bubble styling (using supported Pack properties)
        base_style = {
            "padding": spacing["md"],
            "margin": (spacing["xs"], 0),
            "width": 600,  # Fixed width (Toga doesn't support max_width)
        }
        
        if role == "user":
            # User messages: primary color, right-aligned
            return {
                **base_style,
                "background_color": colors.primary,
                "color": "#ffffff",
            }
        elif role == "assistant":
            # Assistant messages: surface color, left-aligned
            return {
                **base_style,
                "background_color": colors.surface,
                "color": colors.text_primary,
            }
        else:
            # System messages: subtle styling, centered, smaller and more compact
            return {
                **base_style,
                "background_color": colors.card,
                "color": colors.text_secondary,
                "width": 400,  # Smaller width for system messages
                "padding": spacing["sm"],  # Less padding for system messages
            }
    
    def _create_message_header(self, role: str, timestamp: datetime, theme: Theme) -> Optional[toga.Box]:
        """Create message header with role and timestamp using modern styling."""
        if role == "system":
            return None
        
        colors = theme.colors
        spacing = theme.spacing
        typography = theme.typography
        
        header_container = toga.Box(
            style=Pack(
                direction=ROW,
                margin_bottom=spacing["xs"]
            )
        )
        
        # Role indicator with emoji and improved styling
        role_info = self._get_role_info(role)
        role_label = toga.Label(
            f"{role_info['emoji']} {role_info['name']}",
            style=Pack(
                font_size=typography["caption"]["size"],
                font_weight="bold",
                color=role_info["color"] if role == "assistant" else "rgba(255, 255, 255, 0.95)",
                margin_right=spacing["sm"]
            )
        )
        header_container.add(role_label)
        
        # Add spacer to push timestamp to the right
        spacer = toga.Box(style=Pack(flex=1))
        header_container.add(spacer)
        
        # Timestamp with improved formatting
        time_str = timestamp.strftime("%H:%M")
        time_label = toga.Label(
            time_str,
            style=Pack(
                font_size=typography["caption"]["size"] - 1,
                color=(
                    "rgba(255, 255, 255, 0.8)" if role == "user"
                    else colors.text_secondary
                ),
                # Right-align timestamp
                text_align="right"
            )
        )
        header_container.add(time_label)
        
        return header_container
    
    def _get_role_info(self, role: str) -> Dict[str, str]:
        """Get display information for a role."""
        role_info = {
            "user": {
                "name": "You",
                "emoji": "👤",
                "color": "#ffffff"
            },
            "assistant": {
                "name": "Tektra",
                "emoji": "🤖",
                "color": theme_manager.get_theme().colors.primary
            },
            "system": {
                "name": "System",
                "emoji": "ℹ️",
                "color": theme_manager.get_theme().colors.text_secondary
            }
        }
        return role_info.get(role, role_info["assistant"])
    
    def _create_message_content(self, content: str, role: str, theme: Theme) -> toga.Widget:
        """Create the message content widget with enhanced markdown support and styling."""
        colors = theme.colors
        typography = theme.typography
        spacing = theme.spacing
        
        if role == "system":
            # System messages with improved styling
            return toga.Label(
                f"ℹ️ {content}",
                style=Pack(
                    font_size=typography["caption"]["size"],
                    color=colors.text_secondary,
                    text_align="center",
                    padding=(spacing["xs"], 0)
                )
            )
        
        # For user and assistant messages, use enhanced markdown rendering
        try:
            # Create a container for the markdown content with proper spacing
            content_container = toga.Box(
                style=Pack(
                    direction=COLUMN,
                    flex=1,
                    # Add subtle padding for better text readability
                    padding=(spacing["xs"], 0)
                )
            )
            
            # Check if content has markdown elements
            has_markdown = self._detect_markdown_content(content)
            
            if has_markdown:
                # Use full markdown rendering with role-specific styling
                markdown_widget = self._render_enhanced_markdown(content, role, theme)
                content_container.add(markdown_widget)
            else:
                # Simple text with enhanced formatting
                text_widget = self._render_simple_text(content, role, theme)
                content_container.add(text_widget)
            
            return content_container
            
        except Exception as e:
            logger.warning(f"Error rendering message content: {e}")
            # Enhanced fallback with better styling
            return toga.Label(
                content,
                style=Pack(
                    font_size=typography["body1"]["size"],
                    color="#ffffff" if role == "user" else colors.text_primary,
                    text_align="left",
                    padding=(spacing["xs"], 0)
                )
            )
    
    def _detect_markdown_content(self, content: str) -> bool:
        """Detect if content contains markdown elements."""
        markdown_indicators = [
            "```",  # Code blocks
            "`",    # Inline code
            "**",   # Bold
            "*",    # Italic
            "#",    # Headers
            "-",    # Lists
            "1.",   # Numbered lists
            "[",    # Links
            ">",    # Quotes
        ]
        
        return any(indicator in content for indicator in markdown_indicators)
    
    def _render_enhanced_markdown(self, content: str, role: str, theme: Theme) -> toga.Widget:
        """Render content with enhanced markdown support."""
        # Use the existing markdown renderer but with role-specific styling
        markdown_widget = self.markdown_renderer.render_simple_message(content, role)
        
        # Apply role-specific color adjustments
        if role == "user":
            # For user messages, we need to adjust colors for dark background
            self._adjust_colors_for_dark_background(markdown_widget)
        
        return markdown_widget
    
    def _render_simple_text(self, content: str, role: str, theme: Theme) -> toga.Label:
        """Render simple text content with enhanced typography."""
        colors = theme.colors
        typography = theme.typography
        spacing = theme.spacing
        
        # Enhanced text styling based on role
        if role == "user":
            text_color = "#ffffff"
            font_size = typography["body1"]["size"]
        else:
            text_color = colors.text_primary
            font_size = typography["body1"]["size"]
        
        return toga.Label(
            content,
            style=Pack(
                font_size=font_size,
                color=text_color,
                text_align="left",
                # Add subtle padding for better text flow
                padding=(spacing["xs"] // 2, 0)
            )
        )
    
    def _adjust_colors_for_dark_background(self, widget: toga.Widget) -> None:
        """Adjust colors in a widget tree for dark backgrounds."""
        # This is a placeholder for color adjustment logic
        # In a real implementation, this would recursively adjust colors
        # for all child widgets to ensure readability on dark backgrounds
        pass
    
    def _create_status_indicators(self, message: dict, role: str, theme: Theme) -> Optional[toga.Box]:
        """Create enhanced status indicators for messages with better visual feedback."""
        status = message.get("status")
        if not status or role != "user":
            return None
        
        colors = theme.colors
        spacing = theme.spacing
        typography = theme.typography
        
        status_container = toga.Box(
            style=Pack(
                direction=ROW,
                margin_top=spacing["xs"],
                # Right-align status indicators
                text_align="right"
            )
        )
        
        # Enhanced status indicators with better visual hierarchy
        status_config = {
            "sending": {
                "icon": "⏳",
                "color": "rgba(255, 255, 255, 0.6)",
                "text": "Sending..."
            },
            "sent": {
                "icon": "✓",
                "color": "rgba(255, 255, 255, 0.8)",
                "text": "Sent"
            },
            "delivered": {
                "icon": "✓✓",
                "color": "rgba(255, 255, 255, 0.9)",
                "text": "Delivered"
            },
            "error": {
                "icon": "❌",
                "color": colors.error,
                "text": "Failed"
            },
            "read": {
                "icon": "✓✓",
                "color": "#4CAF50",  # Green for read status
                "text": "Read"
            }
        }
        
        if status in status_config:
            config = status_config[status]
            
            # Status icon
            status_label = toga.Label(
                config["icon"],
                style=Pack(
                    font_size=typography["caption"]["size"] - 1,
                    color=config["color"],
                    margin_right=spacing["xs"] // 2
                )
            )
            status_container.add(status_label)
            
            # Optional status text for error states
            if status == "error":
                status_text = toga.Label(
                    config["text"],
                    style=Pack(
                        font_size=typography["caption"]["size"] - 2,
                        color=config["color"],
                    )
                )
                status_container.add(status_text)
            
            return status_container
        
        return None
    
    async def _animate_message_appearance(self, message_widget: toga.Widget, role: str) -> None:
        """Animate the appearance of a message bubble."""
        if not self.animation_manager:
            return
        
        try:
            # Different animations based on role
            if role == "user":
                # User messages slide in from the right
                await self.animation_manager.animate_message_appearance(message_widget, role)
            elif role == "assistant":
                # Assistant messages fade in with slight scale
                await self.animation_manager.animate_message_appearance(message_widget, role)
            else:
                # System messages just fade in
                await self.animation_manager.animate_message_appearance(message_widget, role)
                
        except Exception as e:
            logger.error(f"Error animating message appearance: {e}")
    
    def apply_message_animations(self, bubble: toga.Box, role: str) -> None:
        """Apply animations to a message bubble (synchronous version)."""
        if self.animation_manager:
            asyncio.create_task(self._animate_message_appearance(bubble, role))
    
    def render_code_block_with_copy(self, code: str, language: str) -> toga.Box:
        """Render a code block with enhanced styling and copy functionality."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        borders = theme.borders
        typography = theme.typography
        
        # Enhanced container for code block with modern styling
        code_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                background_color=colors.card,
                margin=(spacing["sm"], 0),
                padding=spacing["md"]
            )
        )
        
        # Enhanced header with language and copy button
        if language and language != "text":
            header = toga.Box(
                style=Pack(
                    direction=ROW,
                    margin_bottom=spacing["sm"]
                )
            )
            
            # Enhanced language label with better styling
            lang_label = toga.Label(
                f"📝 {language.upper()}",
                style=Pack(
                    font_size=typography["caption"]["size"],
                    color=colors.text_secondary,
                    font_weight="bold"
                )
            )
            header.add(lang_label)
            
            # Spacer to push copy button to the right
            spacer = toga.Box(style=Pack(flex=1))
            header.add(spacer)
            
            # Enhanced copy button with better styling and micro-interactions
            copy_button = toga.Button(
                "📋 Copy",
                style=Pack(
                    font_size=typography["caption"]["size"],
                    padding=(spacing["xs"], spacing["sm"]),
                    background_color=colors.primary,
                    color="#ffffff"
                ),
                on_press=lambda x: self._copy_code_to_clipboard(code)
            )
            
            # Set up micro-interactions for the copy button
            if self.animation_manager:
                try:
                    micro_manager = self.animation_manager.micro_interaction_manager
                    copy_button_id = f"copy_button_{id(copy_button)}"
                    
                    # Set up button interactions with enhanced feedback for copy action
                    element_id = micro_manager.setup_button_interactions(
                        copy_button,
                        button_id=copy_button_id,
                        interaction_config={
                            "hover_scale": 1.05,
                            "press_scale": 0.95,
                            "hover_duration": 0.15,
                            "press_duration": 0.1,
                            "spring_back_duration": 0.2,
                            "enable_spring_back": True
                        }
                    )
                    
                    # Track the element for cleanup
                    self.interactive_elements[copy_button_id] = element_id
                    
                except Exception as e:
                    logger.debug(f"Could not set up micro-interactions for copy button: {e}")
            
            header.add(copy_button)
            
            code_container.add(header)
        
        # Enhanced code content with better typography
        code_label = toga.Label(
            code,
            style=Pack(
                font_family="monospace",
                font_size=typography["body2"]["size"],
                color=colors.text_primary,
                background_color=colors.background,
                padding=spacing["sm"]
            )
        )
        code_container.add(code_label)
        
        return code_container
    
    def _copy_code_to_clipboard(self, code: str) -> None:
        """Copy code to clipboard (placeholder implementation)."""
        # Toga doesn't have built-in clipboard support
        # This would need platform-specific implementation
        logger.info(f"Copy to clipboard requested: {code[:50]}...")
    
    def render_markdown_with_syntax_highlighting(self, content: str) -> toga.Box:
        """Render markdown with enhanced syntax highlighting."""
        # Use the existing markdown renderer with enhancements
        return self.markdown_renderer.render_markdown(content)
    
    def clear_cache(self) -> None:
        """Clear the message cache."""
        self.message_cache.clear()
        logger.debug("Message cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self.message_cache)
    
    def set_animation_manager(self, animation_manager: AnimationManager) -> None:
        """Set the animation manager."""
        self.animation_manager = animation_manager
        logger.info("Animation manager updated")


# Global renderer instance
_message_bubble_renderer = None

def get_message_bubble_renderer(animation_manager: Optional[AnimationManager] = None) -> MessageBubbleRenderer:
    """Get the global message bubble renderer instance."""
    global _message_bubble_renderer
    if _message_bubble_renderer is None:
        _message_bubble_renderer = MessageBubbleRenderer(animation_manager)
    elif animation_manager and not _message_bubble_renderer.animation_manager:
        _message_bubble_renderer.set_animation_manager(animation_manager)
    return _message_bubble_renderer