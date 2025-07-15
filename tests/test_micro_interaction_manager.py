"""
Tests for MicroInteractionManager - Micro-Interactions and Button Animations

This module tests the MicroInteractionManager class that handles all subtle
UI feedback including hover effects, button press animations, and focus indicators.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch

import toga
from toga.style import Pack

from src.tektra.gui.animations.micro_interaction_manager import MicroInteractionManager
from src.tektra.gui.animations.animation_manager import AnimationManager


class TestMicroInteractionManager:
    """Test suite for MicroInteractionManager."""
    
    @pytest.fixture
    def animation_manager(self):
        """Create a mock animation manager for testing."""
        mock_manager = Mock(spec=AnimationManager)
        mock_manager.transition_engine = Mock()
        return mock_manager
    
    @pytest.fixture
    def micro_manager(self, animation_manager):
        """Create a MicroInteractionManager instance for testing."""
        animation_manager.transition_engine = Mock()
        return MicroInteractionManager(animation_manager)
    
    @pytest.fixture
    def mock_button(self):
        """Create a mock button for testing."""
        button = Mock(spec=toga.Button)
        button.style = Mock()
        button.on_press = Mock()
        return button
    
    @pytest.fixture
    def mock_input(self):
        """Create a mock input field for testing."""
        input_field = Mock(spec=toga.TextInput)
        input_field.style = Mock()
        return input_field
    
    def test_initialization(self, animation_manager):
        """Test MicroInteractionManager initialization."""
        micro_manager = MicroInteractionManager(animation_manager)
        
        assert micro_manager.animation_manager == animation_manager
        assert micro_manager.transition_engine == animation_manager.transition_engine
        assert isinstance(micro_manager.interactive_elements, dict)
        assert isinstance(micro_manager.hover_states, dict)
        assert isinstance(micro_manager.focus_states, dict)
        assert isinstance(micro_manager.press_states, dict)
        assert isinstance(micro_manager.active_animations, dict)
    
    def test_setup_button_interactions(self, micro_manager, mock_button):
        """Test setting up button interactions."""
        button_id = micro_manager.setup_button_interactions(
            mock_button,
            button_id="test_button",
            interaction_config={"hover_scale": 1.1}
        )
        
        assert button_id == "test_button"
        assert "test_button" in micro_manager.interactive_elements
        
        element = micro_manager.interactive_elements["test_button"]
        assert element["widget"] == mock_button
        assert element["type"] == "button"
        assert element["config"]["hover_scale"] == 1.1
        assert element["current_state"] == "normal"
    
    def test_setup_button_interactions_auto_id(self, micro_manager, mock_button):
        """Test setting up button interactions with auto-generated ID."""
        button_id = micro_manager.setup_button_interactions(mock_button)
        
        assert button_id.startswith("button_")
        assert button_id in micro_manager.interactive_elements
    
    def test_setup_input_interactions(self, micro_manager, mock_input):
        """Test setting up input field interactions."""
        input_id = micro_manager.setup_input_interactions(
            mock_input,
            input_id="test_input",
            interaction_config={"focus_scale": 1.05}
        )
        
        assert input_id == "test_input"
        assert "test_input" in micro_manager.interactive_elements
        
        element = micro_manager.interactive_elements["test_input"]
        assert element["widget"] == mock_input
        assert element["type"] == "input"
        assert element["config"]["focus_scale"] == 1.05
    
    def test_setup_hover_effects(self, micro_manager, mock_button):
        """Test setting up hover effects."""
        widget_id = micro_manager.setup_hover_effects(
            mock_button,
            widget_id="test_hover",
            hover_config={"hover_scale": 1.03}
        )
        
        assert widget_id == "test_hover"
        assert "test_hover" in micro_manager.interactive_elements
        
        element = micro_manager.interactive_elements["test_hover"]
        assert element["type"] == "hoverable"
        assert element["config"]["hover_scale"] == 1.03
    
    def test_setup_focus_indicators(self, micro_manager, mock_input):
        """Test setting up focus indicators."""
        widget_id = micro_manager.setup_focus_indicators(
            mock_input,
            widget_id="test_focus",
            focus_config={"focus_outline_color": "#FF0000"}
        )
        
        assert widget_id == "test_focus"
        assert "test_focus" in micro_manager.interactive_elements
        
        element = micro_manager.interactive_elements["test_focus"]
        assert element["type"] == "focusable"
        assert element["config"]["focus_outline_color"] == "#FF0000"
    
    @pytest.mark.asyncio
    async def test_animate_button_press(self, micro_manager, mock_button):
        """Test button press animation."""
        # Set up button first
        button_id = micro_manager.setup_button_interactions(mock_button, "test_button")
        
        # Mock the transition engine methods
        micro_manager.transition_engine.scale_out = AsyncMock(return_value="scale_out_id")
        micro_manager.transition_engine.scale_in = AsyncMock(return_value="scale_in_id")
        
        # Test the animation
        await micro_manager.animate_button_press(button_id)
        
        # Verify scale_out was called for press down
        micro_manager.transition_engine.scale_out.assert_called_once()
        
        # Verify scale_in was called for spring back
        micro_manager.transition_engine.scale_in.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_animate_button_press_nonexistent(self, micro_manager):
        """Test button press animation with non-existent button."""
        # Should not raise an exception
        await micro_manager.animate_button_press("nonexistent_button")
    
    @pytest.mark.asyncio
    async def test_animate_hover_enter(self, micro_manager, mock_button):
        """Test hover enter animation."""
        # Set up hover effects first
        widget_id = micro_manager.setup_hover_effects(mock_button, "test_hover")
        
        # Mock the transition engine
        micro_manager.transition_engine.scale_in = AsyncMock(return_value="hover_id")
        
        # Test hover enter
        await micro_manager.animate_hover_enter(widget_id)
        
        # Verify animation was called
        micro_manager.transition_engine.scale_in.assert_called_once()
        
        # Verify state tracking
        assert micro_manager.hover_states[widget_id] is True
        assert micro_manager.interactive_elements[widget_id]["current_state"] == "hover"
    
    @pytest.mark.asyncio
    async def test_animate_hover_exit(self, micro_manager, mock_button):
        """Test hover exit animation."""
        # Set up hover effects first
        widget_id = micro_manager.setup_hover_effects(mock_button, "test_hover")
        
        # Mock the transition engine
        micro_manager.transition_engine.scale_out = AsyncMock(return_value="unhover_id")
        
        # Test hover exit
        await micro_manager.animate_hover_exit(widget_id)
        
        # Verify animation was called
        micro_manager.transition_engine.scale_out.assert_called_once()
        
        # Verify state tracking
        assert micro_manager.hover_states[widget_id] is False
        assert micro_manager.interactive_elements[widget_id]["current_state"] == "normal"
    
    @pytest.mark.asyncio
    async def test_animate_focus_enter(self, micro_manager, mock_input):
        """Test focus enter animation."""
        # Set up focus indicators first
        widget_id = micro_manager.setup_focus_indicators(mock_input, "test_focus")
        
        # Mock the transition engine
        micro_manager.transition_engine.scale_in = AsyncMock(return_value="focus_id")
        
        # Test focus enter
        await micro_manager.animate_focus_enter(widget_id)
        
        # Verify animation was called
        micro_manager.transition_engine.scale_in.assert_called_once()
        
        # Verify state tracking
        assert micro_manager.focus_states[widget_id] is True
        assert micro_manager.interactive_elements[widget_id]["current_state"] == "focused"
    
    @pytest.mark.asyncio
    async def test_animate_focus_exit(self, micro_manager, mock_input):
        """Test focus exit animation."""
        # Set up focus indicators first
        widget_id = micro_manager.setup_focus_indicators(mock_input, "test_focus")
        
        # Mock the transition engine
        micro_manager.transition_engine.scale_out = AsyncMock(return_value="unfocus_id")
        
        # Test focus exit
        await micro_manager.animate_focus_exit(widget_id)
        
        # Verify animation was called
        micro_manager.transition_engine.scale_out.assert_called_once()
        
        # Verify state tracking
        assert micro_manager.focus_states[widget_id] is False
        assert micro_manager.interactive_elements[widget_id]["current_state"] == "normal"
    
    @pytest.mark.asyncio
    async def test_pulse_attention(self, micro_manager, mock_button):
        """Test attention pulse animation."""
        # Set up element first
        widget_id = micro_manager.setup_hover_effects(mock_button, "test_pulse")
        
        # Mock the transition engine
        micro_manager.transition_engine.scale_in = AsyncMock(return_value="pulse_in_id")
        micro_manager.transition_engine.scale_out = AsyncMock(return_value="pulse_out_id")
        
        # Test pulse with 2 pulses
        await micro_manager.pulse_attention(widget_id, pulse_count=2)
        
        # Verify scale_in and scale_out were called for each pulse
        assert micro_manager.transition_engine.scale_in.call_count == 2
        assert micro_manager.transition_engine.scale_out.call_count == 2
    
    @pytest.mark.asyncio
    async def test_handle_button_press_with_callback(self, micro_manager, mock_button):
        """Test button press handling with callback."""
        # Set up button
        button_id = micro_manager.setup_button_interactions(mock_button, "test_button")
        
        # Mock the animation
        micro_manager.animate_button_press = AsyncMock()
        
        # Create a mock callback
        callback = AsyncMock()
        
        # Test the handler
        await micro_manager._handle_button_press(button_id, callback, mock_button)
        
        # Verify animation was called
        micro_manager.animate_button_press.assert_called_once_with(button_id)
        
        # Verify callback was called
        callback.assert_called_once_with(mock_button)
    
    @pytest.mark.asyncio
    async def test_handle_button_press_with_sync_callback(self, micro_manager, mock_button):
        """Test button press handling with synchronous callback."""
        # Set up button
        button_id = micro_manager.setup_button_interactions(mock_button, "test_button")
        
        # Mock the animation
        micro_manager.animate_button_press = AsyncMock()
        
        # Create a mock synchronous callback
        callback = Mock()
        
        # Test the handler
        await micro_manager._handle_button_press(button_id, callback, mock_button)
        
        # Verify animation was called
        micro_manager.animate_button_press.assert_called_once_with(button_id)
        
        # Verify callback was called
        callback.assert_called_once_with(mock_button)
    
    def test_cleanup_element(self, micro_manager, mock_button):
        """Test element cleanup."""
        # Set up button first
        button_id = micro_manager.setup_button_interactions(mock_button, "test_button")
        
        # Verify element exists
        assert button_id in micro_manager.interactive_elements
        
        # Clean up
        micro_manager.cleanup_element(button_id)
        
        # Verify element is removed
        assert button_id not in micro_manager.interactive_elements
        assert button_id not in micro_manager.hover_states
        assert button_id not in micro_manager.focus_states
        assert button_id not in micro_manager.press_states
    
    def test_cleanup_all(self, micro_manager, mock_button, mock_input):
        """Test cleanup of all elements."""
        # Set up multiple elements
        button_id = micro_manager.setup_button_interactions(mock_button, "test_button")
        input_id = micro_manager.setup_input_interactions(mock_input, "test_input")
        
        # Verify elements exist
        assert len(micro_manager.interactive_elements) == 2
        
        # Clean up all
        micro_manager.cleanup_all()
        
        # Verify all elements are removed
        assert len(micro_manager.interactive_elements) == 0
        assert len(micro_manager.hover_states) == 0
        assert len(micro_manager.focus_states) == 0
        assert len(micro_manager.press_states) == 0
    
    def test_capture_style_state(self, micro_manager, mock_button):
        """Test capturing widget style state."""
        # Set up mock style attributes
        mock_button.style.background_color = "#FF0000"
        mock_button.style.color = "#000000"
        mock_button.style.padding = 10
        
        # Capture style
        style_state = micro_manager._capture_style_state(mock_button)
        
        # Verify captured values
        assert style_state["background_color"] == "#FF0000"
        assert style_state["color"] == "#000000"
        assert style_state["padding"] == 10
    
    def test_capture_style_state_with_missing_attributes(self, micro_manager, mock_button):
        """Test capturing style state when some attributes are missing."""
        # Set up mock with limited attributes
        mock_button.style = Mock()
        # Don't set any attributes to test graceful handling
        
        # Should not raise an exception
        style_state = micro_manager._capture_style_state(mock_button)
        
        # Should return a dictionary (possibly empty or with None values)
        assert isinstance(style_state, dict)
    
    @pytest.mark.asyncio
    async def test_cancel_animation(self, micro_manager):
        """Test cancelling active animations."""
        # Create a mock task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        
        # Add to active animations
        micro_manager.active_animations["test_animation"] = mock_task
        
        # Cancel the animation
        await micro_manager._cancel_animation("test_animation")
        
        # Verify task was cancelled
        mock_task.cancel.assert_called_once()
        
        # Verify task was removed from active animations
        assert "test_animation" not in micro_manager.active_animations
    
    @pytest.mark.asyncio
    async def test_cancel_animation_already_done(self, micro_manager):
        """Test cancelling animation that's already done."""
        # Create a mock task that's already done
        mock_task = AsyncMock()
        mock_task.done.return_value = True
        
        # Add to active animations
        micro_manager.active_animations["test_animation"] = mock_task
        
        # Cancel the animation
        await micro_manager._cancel_animation("test_animation")
        
        # Verify cancel was not called since task was already done
        mock_task.cancel.assert_not_called()
        
        # Verify task was still removed from active animations
        assert "test_animation" not in micro_manager.active_animations


class TestMicroInteractionIntegration:
    """Integration tests for micro-interactions with real components."""
    
    @pytest.fixture
    def animation_manager(self):
        """Create a real animation manager for integration testing."""
        return AnimationManager()
    
    @pytest.fixture
    def micro_manager(self, animation_manager):
        """Create a real MicroInteractionManager for integration testing."""
        return animation_manager.micro_interaction_manager
    
    def test_integration_with_animation_manager(self, animation_manager):
        """Test that MicroInteractionManager integrates properly with AnimationManager."""
        # Access the micro_interaction_manager property
        micro_manager = animation_manager.micro_interaction_manager
        
        # Verify it's the correct type
        assert isinstance(micro_manager, MicroInteractionManager)
        
        # Verify it has the correct animation manager reference
        assert micro_manager.animation_manager == animation_manager
    
    def test_lazy_initialization(self, animation_manager):
        """Test that MicroInteractionManager is lazily initialized."""
        # Initially should be None
        assert animation_manager._micro_interaction_manager is None
        
        # Accessing the property should initialize it
        micro_manager = animation_manager.micro_interaction_manager
        assert micro_manager is not None
        
        # Second access should return the same instance
        micro_manager2 = animation_manager.micro_interaction_manager
        assert micro_manager is micro_manager2
    
    @pytest.mark.asyncio
    async def test_button_interaction_configuration_defaults(self, micro_manager):
        """Test that button interaction configurations have sensible defaults."""
        mock_button = Mock(spec=toga.Button)
        mock_button.style = Mock()
        mock_button.on_press = Mock()
        
        button_id = micro_manager.setup_button_interactions(mock_button)
        element = micro_manager.interactive_elements[button_id]
        config = element["config"]
        
        # Verify default configuration values are reasonable
        assert 0.9 <= config["press_scale"] <= 1.0  # Should scale down when pressed
        assert 1.0 <= config["hover_scale"] <= 1.1  # Should scale up when hovered
        assert 0.05 <= config["press_duration"] <= 0.2  # Quick press animation
        assert 0.1 <= config["hover_duration"] <= 0.3  # Smooth hover animation
        assert config["enable_press"] is True
        assert config["enable_hover"] is True
        assert config["enable_spring_back"] is True
    
    @pytest.mark.asyncio
    async def test_input_interaction_configuration_defaults(self, micro_manager):
        """Test that input interaction configurations have sensible defaults."""
        mock_input = Mock(spec=toga.TextInput)
        mock_input.style = Mock()
        
        input_id = micro_manager.setup_input_interactions(mock_input)
        element = micro_manager.interactive_elements[input_id]
        config = element["config"]
        
        # Verify default configuration values are reasonable
        assert 1.0 <= config["focus_scale"] <= 1.1  # Should scale up slightly when focused
        assert 0.1 <= config["focus_duration"] <= 0.5  # Smooth focus animation
        assert 0.1 <= config["unfocus_duration"] <= 0.5  # Smooth unfocus animation
        assert config["enable_focus_animation"] is True
        assert config["enable_border_highlight"] is True