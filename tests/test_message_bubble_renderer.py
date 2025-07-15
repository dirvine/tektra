"""
Tests for Enhanced Message Bubble Renderer

This module tests the MessageBubbleRenderer class and its integration
with the chat system for improved visual styling and animations.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

import toga
from toga.style import Pack

from src.tektra.gui.message_bubble_renderer import MessageBubbleRenderer, get_message_bubble_renderer
from src.tektra.gui.animations.animation_manager import AnimationManager
from src.tektra.gui.themes import theme_manager


class TestMessageBubbleRenderer:
    """Test cases for MessageBubbleRenderer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.animation_manager = Mock(spec=AnimationManager)
        self.animation_manager.animate_message_appearance = AsyncMock()
        self.renderer = MessageBubbleRenderer(self.animation_manager)
    
    def test_renderer_initialization(self):
        """Test that renderer initializes correctly."""
        assert self.renderer.animation_manager == self.animation_manager
        assert self.renderer.markdown_renderer is not None
        assert isinstance(self.renderer.message_cache, dict)
        assert len(self.renderer.message_cache) == 0
    
    def test_render_user_message_bubble(self):
        """Test rendering a user message bubble."""
        message = {
            "role": "user",
            "content": "Hello, how are you?",
            "timestamp": datetime.now(),
            "id": "user_test_1"
        }
        
        bubble = self.renderer.render_message_bubble(message, animate=False)
        
        # Verify the bubble is created
        assert isinstance(bubble, toga.Box)
        
        # Verify message is cached
        assert message["id"] in self.renderer.message_cache
    
    def test_render_assistant_message_bubble(self):
        """Test rendering an assistant message bubble."""
        message = {
            "role": "assistant",
            "content": "I'm doing well, thank you for asking!",
            "timestamp": datetime.now(),
            "id": "assistant_test_1"
        }
        
        bubble = self.renderer.render_message_bubble(message, animate=False)
        
        # Verify the bubble is created
        assert isinstance(bubble, toga.Box)
        
        # Verify message is cached
        assert message["id"] in self.renderer.message_cache
    
    def test_render_system_message_bubble(self):
        """Test rendering a system message bubble."""
        message = {
            "role": "system",
            "content": "Connection established",
            "timestamp": datetime.now(),
            "id": "system_test_1"
        }
        
        bubble = self.renderer.render_message_bubble(message, animate=False)
        
        # Verify the bubble is created
        assert isinstance(bubble, toga.Box)
        
        # Verify message is cached
        assert message["id"] in self.renderer.message_cache
    
    def test_role_based_styling(self):
        """Test that different roles get different styling."""
        user_message = {
            "role": "user",
            "content": "User message",
            "timestamp": datetime.now(),
            "id": "user_style_test"
        }
        
        assistant_message = {
            "role": "assistant", 
            "content": "Assistant message",
            "timestamp": datetime.now(),
            "id": "assistant_style_test"
        }
        
        system_message = {
            "role": "system",
            "content": "System message", 
            "timestamp": datetime.now(),
            "id": "system_style_test"
        }
        
        # Get bubble styles
        user_style = self.renderer._get_bubble_style("user", theme_manager.get_theme())
        assistant_style = self.renderer._get_bubble_style("assistant", theme_manager.get_theme())
        system_style = self.renderer._get_bubble_style("system", theme_manager.get_theme())
        
        # Verify styles are different
        assert user_style != assistant_style
        assert assistant_style != system_style
        assert user_style != system_style
        
        # Verify user messages have primary color background
        assert user_style["background_color"] == theme_manager.get_theme().colors.primary
        
        # Verify assistant messages have surface color background
        assert assistant_style["background_color"] == theme_manager.get_theme().colors.surface
    
    def test_message_with_markdown_content(self):
        """Test rendering messages with markdown content."""
        message = {
            "role": "assistant",
            "content": "Here's some **bold text** and `inline code`:\n\n```python\nprint('Hello, World!')\n```",
            "timestamp": datetime.now(),
            "id": "markdown_test_1"
        }
        
        bubble = self.renderer.render_message_bubble(message, animate=False)
        
        # Verify the bubble is created
        assert isinstance(bubble, toga.Box)
        
        # Verify markdown detection works
        assert self.renderer._detect_markdown_content(message["content"]) == True
    
    def test_message_without_markdown_content(self):
        """Test rendering simple text messages."""
        message = {
            "role": "user",
            "content": "This is just plain text without any special formatting.",
            "timestamp": datetime.now(),
            "id": "plain_text_test_1"
        }
        
        bubble = self.renderer.render_message_bubble(message, animate=False)
        
        # Verify the bubble is created
        assert isinstance(bubble, toga.Box)
        
        # Verify markdown detection works
        assert self.renderer._detect_markdown_content(message["content"]) == False
    
    def test_markdown_detection(self):
        """Test markdown content detection."""
        # Test cases with markdown
        markdown_cases = [
            "Here's some **bold** text",
            "And some `inline code`",
            "```python\ncode block\n```",
            "# Header text",
            "- List item",
            "1. Numbered item",
            "[Link text](url)",
            "> Quote text"
        ]
        
        for case in markdown_cases:
            assert self.renderer._detect_markdown_content(case) == True
        
        # Test cases without markdown
        plain_cases = [
            "Just plain text",
            "No special formatting here",
            "Regular conversation text"
        ]
        
        for case in plain_cases:
            assert self.renderer._detect_markdown_content(case) == False
    
    def test_role_info_retrieval(self):
        """Test role information retrieval."""
        user_info = self.renderer._get_role_info("user")
        assistant_info = self.renderer._get_role_info("assistant")
        system_info = self.renderer._get_role_info("system")
        
        # Verify user info
        assert user_info["name"] == "You"
        assert user_info["emoji"] == "👤"
        
        # Verify assistant info
        assert assistant_info["name"] == "Tektra"
        assert assistant_info["emoji"] == "🤖"
        
        # Verify system info
        assert system_info["name"] == "System"
        assert system_info["emoji"] == "ℹ️"
    
    @pytest.mark.asyncio
    async def test_message_animation(self):
        """Test message appearance animation."""
        message = {
            "role": "assistant",
            "content": "Animated message",
            "timestamp": datetime.now(),
            "id": "animation_test_1"
        }
        
        # Render with animation
        bubble = self.renderer.render_message_bubble(message, animate=True)
        
        # Wait a bit for animation to start
        await asyncio.sleep(0.1)
        
        # Verify animation was called
        self.animation_manager.animate_message_appearance.assert_called()
    
    def test_message_caching(self):
        """Test message caching functionality."""
        message = {
            "role": "user",
            "content": "Cached message",
            "timestamp": datetime.now(),
            "id": "cache_test_1"
        }
        
        # First render
        bubble1 = self.renderer.render_message_bubble(message, animate=False)
        
        # Second render (should use cache)
        bubble2 = self.renderer.render_message_bubble(message, animate=False)
        
        # Should be the same object from cache
        assert bubble1 is bubble2
        assert len(self.renderer.message_cache) == 1
    
    def test_cache_management(self):
        """Test cache management methods."""
        # Add some messages to cache
        for i in range(3):
            message = {
                "role": "user",
                "content": f"Message {i}",
                "timestamp": datetime.now(),
                "id": f"cache_mgmt_test_{i}"
            }
            self.renderer.render_message_bubble(message, animate=False)
        
        # Verify cache size
        assert self.renderer.get_cache_size() == 3
        
        # Clear cache
        self.renderer.clear_cache()
        assert self.renderer.get_cache_size() == 0
        assert len(self.renderer.message_cache) == 0
    
    def test_code_block_rendering(self):
        """Test code block rendering with copy functionality."""
        code = "print('Hello, World!')"
        language = "python"
        
        code_widget = self.renderer.render_code_block_with_copy(code, language)
        
        # Verify widget is created
        assert isinstance(code_widget, toga.Box)
    
    def test_animation_manager_setting(self):
        """Test setting animation manager."""
        new_animation_manager = Mock(spec=AnimationManager)
        
        self.renderer.set_animation_manager(new_animation_manager)
        
        assert self.renderer.animation_manager == new_animation_manager
    
    def test_status_indicators(self):
        """Test message status indicators."""
        message_with_status = {
            "role": "user",
            "content": "Message with status",
            "timestamp": datetime.now(),
            "id": "status_test_1",
            "status": "sent"
        }
        
        status_widget = self.renderer._create_status_indicators(
            message_with_status, "user", theme_manager.get_theme()
        )
        
        # Should create status widget for user messages with status
        assert status_widget is not None
        assert isinstance(status_widget, toga.Box)
        
        # Test assistant message (should not have status indicators)
        status_widget_assistant = self.renderer._create_status_indicators(
            message_with_status, "assistant", theme_manager.get_theme()
        )
        
        assert status_widget_assistant is None


class TestMessageBubbleRendererIntegration:
    """Integration tests for MessageBubbleRenderer with other components."""
    
    def test_global_renderer_instance(self):
        """Test global renderer instance management."""
        # Get first instance
        renderer1 = get_message_bubble_renderer()
        
        # Get second instance (should be same)
        renderer2 = get_message_bubble_renderer()
        
        assert renderer1 is renderer2
    
    def test_global_renderer_with_animation_manager(self):
        """Test global renderer with animation manager."""
        # Reset global renderer to ensure clean test
        import src.tektra.gui.message_bubble_renderer as mbr_module
        mbr_module._message_bubble_renderer = None
        
        animation_manager = Mock(spec=AnimationManager)
        
        # Get renderer with animation manager
        renderer = get_message_bubble_renderer(animation_manager)
        
        assert renderer.animation_manager == animation_manager
    
    @patch('src.tektra.gui.message_bubble_renderer.theme_manager')
    def test_theme_integration(self, mock_theme_manager):
        """Test integration with theme manager."""
        mock_theme = Mock()
        mock_theme.colors.primary = "#0066cc"
        mock_theme.colors.surface = "#f8f9fa"
        mock_theme.colors.text_primary = "#1a1a1a"
        mock_theme.colors.text_secondary = "#6c757d"
        mock_theme.colors.card = "#ffffff"
        mock_theme.colors.background = "#ffffff"
        mock_theme.spacing = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32}
        mock_theme.typography = {
            "caption": {"size": 12},
            "body1": {"size": 16}
        }
        mock_theme.borders = {"radius_sm": 4}
        mock_theme_manager.get_theme.return_value = mock_theme
        
        renderer = MessageBubbleRenderer()
        
        message = {
            "role": "user",
            "content": "Theme test",
            "timestamp": datetime.now(),
            "id": "theme_test_1"
        }
        
        bubble = renderer.render_message_bubble(message, animate=False)
        
        # Verify theme was used
        mock_theme_manager.get_theme.assert_called()
        assert isinstance(bubble, toga.Box)


class TestMessageBubbleRendererErrorHandling:
    """Test error handling in MessageBubbleRenderer."""
    
    def test_render_with_missing_fields(self):
        """Test rendering with missing message fields."""
        renderer = MessageBubbleRenderer()
        
        # Message with minimal fields
        message = {
            "content": "Minimal message"
        }
        
        # Should not crash and should use defaults
        bubble = renderer.render_message_bubble(message, animate=False)
        assert isinstance(bubble, toga.Box)
    
    def test_render_with_invalid_role(self):
        """Test rendering with invalid role."""
        renderer = MessageBubbleRenderer()
        
        message = {
            "role": "invalid_role",
            "content": "Message with invalid role",
            "timestamp": datetime.now(),
            "id": "invalid_role_test"
        }
        
        # Should not crash and should handle gracefully
        bubble = renderer.render_message_bubble(message, animate=False)
        assert isinstance(bubble, toga.Box)
    
    @patch('src.tektra.gui.message_bubble_renderer.logger')
    def test_animation_error_handling(self, mock_logger):
        """Test error handling during animation."""
        animation_manager = Mock(spec=AnimationManager)
        animation_manager.animate_message_appearance = AsyncMock(side_effect=Exception("Animation error"))
        
        renderer = MessageBubbleRenderer(animation_manager)
        
        message = {
            "role": "user",
            "content": "Animation error test",
            "timestamp": datetime.now(),
            "id": "animation_error_test"
        }
        
        # Should not crash even if animation fails
        bubble = renderer.render_message_bubble(message, animate=True)
        assert isinstance(bubble, toga.Box)


if __name__ == "__main__":
    pytest.main([__file__])