"""
Tests for Enhanced Input Field

This module tests the enhanced input field component to ensure all
requirements for task 4 are properly implemented.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

import toga
from toga.style import Pack

from src.tektra.gui.enhanced_input_field import EnhancedInputField
from src.tektra.gui.animations.animation_manager import AnimationManager


class TestEnhancedInputField:
    """Test suite for the enhanced input field component."""
    
    @pytest.fixture
    def animation_manager(self):
        """Create a mock animation manager for testing."""
        manager = Mock(spec=AnimationManager)
        manager.animate_button_press = AsyncMock(return_value="test_id")
        manager.animate_input_focus = AsyncMock(return_value="test_id")
        manager.animate_button_deactivate = AsyncMock(return_value="test_id")
        manager.animate_text_update = AsyncMock(return_value="test_id")
        manager.animate_text_clear = AsyncMock(return_value="test_id")
        manager.global_animation_enabled = True
        return manager
    
    @pytest.fixture
    def enhanced_input(self, animation_manager):
        """Create an enhanced input field for testing."""
        return EnhancedInputField(
            animation_manager=animation_manager,
            placeholder="Test message...",
            max_characters=100,
            min_height=40,
            max_height=200
        )
    
    def test_initialization(self, enhanced_input):
        """Test that the enhanced input field initializes correctly."""
        assert enhanced_input.placeholder == "Test message..."
        assert enhanced_input.max_characters == 100
        assert enhanced_input.min_height == 40
        assert enhanced_input.max_height == 200
        assert enhanced_input.current_character_count == 0
        assert not enhanced_input.is_send_enabled
        assert not enhanced_input.is_voice_recording
        assert not enhanced_input.voice_enabled
        assert not enhanced_input.file_enabled
    
    def test_widget_structure(self, enhanced_input):
        """Test that the widget structure is built correctly."""
        # Main widget should be a Box
        assert isinstance(enhanced_input.widget, toga.Box)
        
        # Should have text input, buttons, and status area
        assert hasattr(enhanced_input, 'text_input')
        assert hasattr(enhanced_input, 'send_button')
        assert hasattr(enhanced_input, 'voice_button')
        assert hasattr(enhanced_input, 'file_button')
        assert hasattr(enhanced_input, 'char_count_label')
        assert hasattr(enhanced_input, 'status_indicator')
        
        # Text input should have correct placeholder
        assert enhanced_input.text_input.placeholder == "Test message..."
    
    @pytest.mark.asyncio
    async def test_character_count_update(self, enhanced_input):
        """Test requirement 4.1: Real-time character count and input validation."""
        # Simulate text input change
        enhanced_input.text_input.value = "Hello world"
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Check character count is updated
        assert enhanced_input.current_character_count == 11
        assert "11/100" in enhanced_input.char_count_label.text
        
        # Test warning state (80% of max)
        enhanced_input.text_input.value = "x" * 85  # 85% of 100
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        assert enhanced_input.current_character_count == 85
        # The status indicator should show warning emoji for 85% usage
        assert enhanced_input.status_indicator.text == "⚠️"
        
        # Test error state (over limit)
        enhanced_input.text_input.value = "x" * 105  # Over limit
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        assert enhanced_input.current_character_count == 105
        # Should show error message for over limit
        assert "❌" in enhanced_input.status_indicator.text
    
    @pytest.mark.asyncio
    async def test_focus_animations(self, enhanced_input, animation_manager):
        """Test requirement 4.2: Clear visual focus indicators."""
        # Test focus animation
        await enhanced_input.animate_focus(True)
        
        # Verify animation manager was called
        animation_manager.animate_input_focus.assert_called_with(
            enhanced_input.text_input, True
        )
        
        # Test unfocus animation
        await enhanced_input.animate_focus(False)
        
        animation_manager.animate_input_focus.assert_called_with(
            enhanced_input.text_input, False
        )
    
    @pytest.mark.asyncio
    async def test_enter_key_handling(self, enhanced_input):
        """Test requirement 4.3: Enter key sends message with keyboard shortcuts."""
        # Mock the send callback
        send_callback = AsyncMock()
        enhanced_input.on_message_send = send_callback
        
        # Set up valid input
        enhanced_input.text_input.value = "Test message"
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Simulate Enter key press
        await enhanced_input._on_enter_pressed(enhanced_input.text_input)
        
        # Verify message was sent
        send_callback.assert_called_once_with("Test message")
    
    @pytest.mark.asyncio
    async def test_send_button_state_management(self, enhanced_input, animation_manager):
        """Test requirement 4.4: Send button disabled when input is empty with visual feedback."""
        # Initially, send button should be disabled
        assert not enhanced_input.send_button.enabled
        assert not enhanced_input.is_send_enabled
        
        # Add text - button should become enabled
        enhanced_input.text_input.value = "Hello"
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        assert enhanced_input.send_button.enabled
        assert enhanced_input.is_send_enabled
        
        # Clear text - button should become disabled
        enhanced_input.text_input.value = ""
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        assert not enhanced_input.send_button.enabled
        assert not enhanced_input.is_send_enabled
        
        # Verify animation was called for button deactivation
        animation_manager.animate_button_deactivate.assert_called()
    
    @pytest.mark.asyncio
    async def test_auto_expanding_input(self, enhanced_input):
        """Test requirement 4.5: Auto-expanding input field with smooth height transitions."""
        # Test with short text
        enhanced_input.text_input.value = "Short"
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Height should be at minimum
        assert enhanced_input.current_height == enhanced_input.min_height
        
        # Test with long text that should trigger expansion
        # Create text long enough to trigger multiple lines
        long_text = "x" * 200  # 200 characters should trigger expansion
        enhanced_input.text_input.value = long_text
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Store the height for assertion
        height_after_long_text = enhanced_input.current_height
        
        # Height should have increased but not exceed maximum
        assert height_after_long_text > enhanced_input.min_height, f"Height {height_after_long_text} should be > {enhanced_input.min_height}"
        assert height_after_long_text <= enhanced_input.max_height
        
        # Test that the auto-expansion mechanism is working by checking the height tracking
        # The core requirement is that the input field can expand - we've verified this above
        # Additional test: verify height resets when text is cleared
        enhanced_input.text_input.value = ""
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Height should return to minimum when text is cleared
        assert enhanced_input.current_height == enhanced_input.min_height
    
    @pytest.mark.asyncio
    async def test_voice_recording_feedback(self, enhanced_input):
        """Test requirement 4.6: Visual feedback for voice recording state."""
        # Enable voice features first
        enhanced_input.enable_voice_features(True)
        assert enhanced_input.voice_enabled
        
        # Test voice recording toggle
        await enhanced_input._toggle_voice_recording()
        
        # Should be in recording state
        assert enhanced_input.is_voice_recording
        assert enhanced_input.voice_button.text == "🔴"
        assert not enhanced_input.text_input.enabled  # Input disabled during recording
        assert not enhanced_input.send_button.enabled
        assert not enhanced_input.file_button.enabled
        
        # Toggle back to ready state
        await enhanced_input._toggle_voice_recording()
        
        # Should be back to ready state
        assert not enhanced_input.is_voice_recording
        assert enhanced_input.voice_button.text == "🎤"
        assert enhanced_input.text_input.enabled  # Input re-enabled
    
    @pytest.mark.asyncio
    async def test_button_press_animations(self, enhanced_input, animation_manager):
        """Test that button presses trigger micro-interactions."""
        # Mock the micro-interaction manager
        mock_micro_manager = Mock()
        mock_micro_manager.setup_button_interactions = Mock(return_value="mock_element_id")
        mock_micro_manager.setup_input_interactions = Mock(return_value="mock_input_id")
        mock_micro_manager.setup_hover_effects = Mock(return_value="mock_hover_id")
        animation_manager.micro_interaction_manager = mock_micro_manager
        
        # Create a new enhanced input field to test micro-interaction setup
        from src.tektra.gui.enhanced_input_field import EnhancedInputField
        test_input = EnhancedInputField(animation_manager=animation_manager)
        
        # Verify that micro-interactions were set up for buttons
        assert mock_micro_manager.setup_button_interactions.call_count >= 3  # send, voice, file buttons
        assert mock_micro_manager.setup_input_interactions.call_count >= 1   # main input
        assert mock_micro_manager.setup_hover_effects.call_count >= 1        # character count label
    
    @pytest.mark.asyncio
    async def test_message_sending_flow(self, enhanced_input, animation_manager):
        """Test the complete message sending flow with animations."""
        # Mock the send callback
        send_callback = AsyncMock()
        enhanced_input.on_message_send = send_callback
        
        # Set up valid input
        test_message = "Test message for sending"
        enhanced_input.text_input.value = test_message
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Send the message
        await enhanced_input._send_message()
        
        # Verify message was sent
        send_callback.assert_called_once_with(test_message)
        
        # Verify input was cleared
        assert enhanced_input.text_input.value == ""
        assert enhanced_input.current_character_count == 0
        assert not enhanced_input.is_send_enabled
        
        # Verify animations were called
        animation_manager.animate_text_clear.assert_called()
    
    def test_input_validation(self, enhanced_input):
        """Test input validation logic."""
        # Valid input
        enhanced_input.text_input.value = "Valid message"
        enhanced_input.current_character_count = 13
        assert enhanced_input.is_valid_input()
        
        # Empty input
        enhanced_input.text_input.value = ""
        enhanced_input.current_character_count = 0
        assert not enhanced_input.is_valid_input()
        
        # Over character limit
        enhanced_input.text_input.value = "x" * 150
        enhanced_input.current_character_count = 150
        assert not enhanced_input.is_valid_input()
        
        # Whitespace only
        enhanced_input.text_input.value = "   "
        enhanced_input.current_character_count = 3
        assert not enhanced_input.is_valid_input()
    
    def test_feature_enabling(self, enhanced_input):
        """Test enabling and disabling voice and file features."""
        # Test voice features
        enhanced_input.enable_voice_features(True)
        assert enhanced_input.voice_enabled
        assert enhanced_input.voice_button.enabled
        
        enhanced_input.enable_voice_features(False)
        assert not enhanced_input.voice_enabled
        assert not enhanced_input.voice_button.enabled
        
        # Test file features
        enhanced_input.enable_file_features(True)
        assert enhanced_input.file_enabled
        assert enhanced_input.file_button.enabled
        
        enhanced_input.enable_file_features(False)
        assert not enhanced_input.file_enabled
        assert not enhanced_input.file_button.enabled
    
    def test_utility_methods(self, enhanced_input):
        """Test utility methods for getting and setting text."""
        # Test setting text (synchronous part)
        enhanced_input.set_text("Hello world")
        assert enhanced_input.text_input.value == "Hello world"
        # Character count should be updated immediately
        assert enhanced_input.current_character_count == 11
        
        # Test getting text
        assert enhanced_input.get_current_text() == "Hello world"
        
        # Test character count getter
        assert enhanced_input.get_character_count() == 11
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, enhanced_input):
        """Test that callback errors are handled gracefully."""
        # Mock a callback that raises an exception
        error_callback = AsyncMock(side_effect=Exception("Test error"))
        enhanced_input.on_message_send = error_callback
        
        # Set up valid input
        enhanced_input.text_input.value = "Test message"
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Send message - should not raise exception
        await enhanced_input._send_message()
        
        # Callback should have been called despite the error
        error_callback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_character_limit_enforcement(self, enhanced_input):
        """Test that character limits are properly enforced."""
        # Set text at exactly the limit
        enhanced_input.text_input.value = "x" * 100
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Should be valid at the limit
        assert enhanced_input.is_valid_input()
        assert enhanced_input.is_send_enabled
        
        # Set text over the limit
        enhanced_input.text_input.value = "x" * 101
        await enhanced_input._on_input_change(enhanced_input.text_input)
        
        # Should be invalid over the limit
        assert not enhanced_input.is_valid_input()
        assert not enhanced_input.is_send_enabled
        assert "❌" in enhanced_input.status_indicator.text


@pytest.mark.asyncio
async def test_integration_with_chat_panel():
    """Test integration between enhanced input field and chat panel."""
    from src.tektra.gui.chat_panel import ChatPanel
    
    # Mock callbacks
    on_message_send = AsyncMock()
    on_voice_toggle = AsyncMock()
    on_file_upload = AsyncMock()
    
    # Create chat panel
    chat_panel = ChatPanel(
        on_message_send=on_message_send,
        on_voice_toggle=on_voice_toggle,
        on_file_upload=on_file_upload
    )
    
    # Verify enhanced input field is created
    assert hasattr(chat_panel, 'enhanced_input')
    assert isinstance(chat_panel.enhanced_input, EnhancedInputField)
    
    # Verify compatibility references are set
    assert chat_panel.text_input is chat_panel.enhanced_input.text_input
    assert chat_panel.send_button is chat_panel.enhanced_input.send_button
    assert chat_panel.voice_button is chat_panel.enhanced_input.voice_button
    assert chat_panel.file_button is chat_panel.enhanced_input.file_button
    
    # Test feature enabling through chat panel
    chat_panel.enable_voice_features(True)
    assert chat_panel.enhanced_input.voice_enabled
    
    chat_panel.enable_file_features(True)
    assert chat_panel.enhanced_input.file_enabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])