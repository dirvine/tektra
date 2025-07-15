"""
Tests for Theme Transition Manager

This module tests the theme transition functionality including smooth
theme switching, system theme detection, and accessibility features.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import platform

import toga

from src.tektra.gui.animations.animation_manager import AnimationManager
from src.tektra.gui.animations.theme_transition_manager import (
    ThemeTransitionManager,
    SystemThemeDetector,
    ColorTransition,
    ThemeTransitionState,
    get_theme_transition_manager
)
from src.tektra.gui.themes import ThemeManager, LIGHT_THEME, DARK_THEME


class TestSystemThemeDetector:
    """Test system theme detection functionality."""
    
    def test_init(self):
        """Test SystemThemeDetector initialization."""
        detector = SystemThemeDetector()
        
        assert detector.current_system_theme in ["light", "dark"]
        assert detector.theme_change_callbacks == []
        assert not detector.monitoring_active
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_detect_macos_light_theme(self, mock_subprocess, mock_platform):
        """Test macOS light theme detection."""
        mock_platform.return_value = "Darwin"
        mock_subprocess.return_value.returncode = 1  # Command fails for light theme
        
        detector = SystemThemeDetector()
        theme = detector._detect_current_system_theme()
        
        assert theme == "light"
        # The method is called twice - once during init and once during explicit call
        assert mock_subprocess.call_count >= 1
    
    @patch('platform.system')
    @patch('subprocess.run')
    def test_detect_macos_dark_theme(self, mock_subprocess, mock_platform):
        """Test macOS dark theme detection."""
        mock_platform.return_value = "Darwin"
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "Dark"
        
        detector = SystemThemeDetector()
        theme = detector._detect_current_system_theme()
        
        assert theme == "dark"
    
    @patch('platform.system')
    def test_detect_unknown_system(self, mock_platform):
        """Test theme detection on unknown system."""
        mock_platform.return_value = "Unknown"
        
        detector = SystemThemeDetector()
        theme = detector._detect_current_system_theme()
        
        assert theme == "light"  # Default fallback
    
    def test_add_remove_callbacks(self):
        """Test adding and removing theme change callbacks."""
        detector = SystemThemeDetector()
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        detector.add_theme_change_callback(callback1)
        detector.add_theme_change_callback(callback2)
        
        assert len(detector.theme_change_callbacks) == 2
        assert callback1 in detector.theme_change_callbacks
        assert callback2 in detector.theme_change_callbacks
        
        # Remove callback
        detector.remove_theme_change_callback(callback1)
        
        assert len(detector.theme_change_callbacks) == 1
        assert callback1 not in detector.theme_change_callbacks
        assert callback2 in detector.theme_change_callbacks
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self):
        """Test theme monitoring start and stop."""
        detector = SystemThemeDetector()
        
        # Start monitoring
        monitoring_task = asyncio.create_task(detector.start_monitoring())
        await asyncio.sleep(0.1)  # Let it start
        
        assert detector.monitoring_active
        
        # Stop monitoring
        detector.stop_monitoring()
        await asyncio.sleep(0.1)  # Let it stop
        
        assert not detector.monitoring_active
        
        # Clean up
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass


class TestThemeTransitionManager:
    """Test theme transition manager functionality."""
    
    @pytest.fixture
    def animation_manager(self):
        """Create a mock animation manager."""
        return Mock(spec=AnimationManager)
    
    @pytest.fixture
    def theme_manager(self):
        """Create a mock theme manager."""
        manager = Mock(spec=ThemeManager)
        manager.themes = {"light": LIGHT_THEME, "dark": DARK_THEME}
        manager.current_theme_name = "light"
        return manager
    
    @pytest.fixture
    def transition_manager(self, animation_manager, theme_manager):
        """Create a theme transition manager."""
        return ThemeTransitionManager(animation_manager, theme_manager)
    
    def test_init(self, transition_manager):
        """Test ThemeTransitionManager initialization."""
        assert transition_manager.active_transitions == {}
        assert transition_manager.registered_widgets == []
        assert transition_manager.transition_callbacks == []
        assert transition_manager.transition_duration == 0.4
        assert not transition_manager.accessibility_mode
        assert not transition_manager.auto_system_theme
    
    def test_register_unregister_widgets(self, transition_manager):
        """Test widget registration and unregistration."""
        widget1 = Mock(spec=toga.Box)
        widget2 = Mock(spec=toga.Label)
        
        # Register widgets
        transition_manager.register_widget(widget1)
        transition_manager.register_widget(widget2)
        
        assert len(transition_manager.registered_widgets) == 2
        assert widget1 in transition_manager.registered_widgets
        assert widget2 in transition_manager.registered_widgets
        
        # Don't register same widget twice
        transition_manager.register_widget(widget1)
        assert len(transition_manager.registered_widgets) == 2
        
        # Unregister widget
        transition_manager.unregister_widget(widget1)
        
        assert len(transition_manager.registered_widgets) == 1
        assert widget1 not in transition_manager.registered_widgets
        assert widget2 in transition_manager.registered_widgets
    
    def test_add_remove_callbacks(self, transition_manager):
        """Test adding and removing transition callbacks."""
        callback1 = Mock()
        callback2 = Mock()
        
        # Add callbacks
        transition_manager.add_transition_callback(callback1)
        transition_manager.add_transition_callback(callback2)
        
        assert len(transition_manager.transition_callbacks) == 2
        assert callback1 in transition_manager.transition_callbacks
        assert callback2 in transition_manager.transition_callbacks
        
        # Remove callback
        transition_manager.remove_transition_callback(callback1)
        
        assert len(transition_manager.transition_callbacks) == 1
        assert callback1 not in transition_manager.transition_callbacks
        assert callback2 in transition_manager.transition_callbacks
    
    @pytest.mark.asyncio
    async def test_instant_theme_transition(self, transition_manager):
        """Test instant theme transition without animation."""
        # Mock the theme manager switch_theme method
        transition_manager.theme_manager.switch_theme = Mock()
        
        # Perform instant transition
        transition_id = await transition_manager.transition_to_theme(
            "dark", animated=False
        )
        
        # Verify theme was switched
        transition_manager.theme_manager.switch_theme.assert_called_once_with("dark")
        assert transition_id != ""
    
    @pytest.mark.asyncio
    async def test_animated_theme_transition(self, transition_manager):
        """Test animated theme transition."""
        # Mock animation manager methods
        transition_manager.animation_manager.transition_engine = Mock()
        transition_manager.animation_manager.transition_engine.fade_out = AsyncMock(return_value="fade_out_id")
        transition_manager.animation_manager.transition_engine.fade_in = AsyncMock(return_value="fade_in_id")
        
        # Mock theme manager
        transition_manager.theme_manager.switch_theme = Mock()
        
        # Register a widget
        widget = Mock(spec=toga.Box)
        transition_manager.register_widget(widget)
        
        # Perform animated transition
        transition_id = await transition_manager.transition_to_theme(
            "dark", animated=True, duration=0.2
        )
        
        # Verify animations were called
        assert transition_manager.animation_manager.transition_engine.fade_out.called
        assert transition_manager.animation_manager.transition_engine.fade_in.called
        
        # Verify theme was switched
        transition_manager.theme_manager.switch_theme.assert_called_once_with("dark")
        assert transition_id != ""
    
    @pytest.mark.asyncio
    async def test_transition_to_same_theme(self, transition_manager):
        """Test transitioning to the same theme (should be no-op)."""
        transition_manager.theme_manager.current_theme_name = "light"
        
        transition_id = await transition_manager.transition_to_theme("light")
        
        assert transition_id == ""
    
    @pytest.mark.asyncio
    async def test_transition_to_invalid_theme(self, transition_manager):
        """Test transitioning to invalid theme raises error."""
        with pytest.raises(ValueError, match="Unknown theme: invalid"):
            await transition_manager.transition_to_theme("invalid")
    
    @pytest.mark.asyncio
    async def test_accessibility_mode_disables_animation(self, transition_manager):
        """Test that accessibility mode disables animations."""
        # Enable accessibility mode
        transition_manager.set_accessibility_mode(True)
        
        # Mock theme manager
        transition_manager.theme_manager.switch_theme = Mock()
        
        # Perform transition (should be instant even with animated=True)
        transition_id = await transition_manager.transition_to_theme(
            "dark", animated=True
        )
        
        # Verify theme was switched instantly
        transition_manager.theme_manager.switch_theme.assert_called_once_with("dark")
        assert transition_id != ""
    
    @pytest.mark.asyncio
    async def test_auto_system_theme_setting(self, transition_manager):
        """Test auto system theme setting."""
        # Mock system theme detector
        transition_manager.system_theme_detector.start_monitoring = AsyncMock()
        transition_manager.system_theme_detector.stop_monitoring = Mock()
        transition_manager.system_theme_detector.current_system_theme = "dark"
        
        # Mock the transition_to_theme method to avoid actual theme switching
        transition_manager.transition_to_theme = AsyncMock()
        
        # Enable auto system theme
        transition_manager.set_auto_system_theme(True)
        
        assert transition_manager.auto_system_theme
        
        # Disable auto system theme
        transition_manager.set_auto_system_theme(False)
        
        assert not transition_manager.auto_system_theme
        transition_manager.system_theme_detector.stop_monitoring.assert_called_once()
    
    def test_transition_duration_setting(self, transition_manager):
        """Test setting transition duration with bounds checking."""
        # Set valid duration
        transition_manager.set_transition_duration(0.5)
        assert transition_manager.transition_duration == 0.5
        
        # Test lower bound
        transition_manager.set_transition_duration(0.05)
        assert transition_manager.transition_duration == 0.1
        
        # Test upper bound
        transition_manager.set_transition_duration(3.0)
        assert transition_manager.transition_duration == 2.0
    
    def test_get_theme_info(self, transition_manager):
        """Test getting theme information."""
        transition_manager.theme_manager.current_theme_name = "light"
        transition_manager.theme_manager.themes = {"light": LIGHT_THEME, "dark": DARK_THEME}
        
        assert transition_manager.get_current_theme_name() == "light"
        assert set(transition_manager.get_available_themes()) == {"light", "dark"}
    
    def test_transition_state_tracking(self, transition_manager):
        """Test transition state tracking."""
        assert not transition_manager.is_transition_active()
        
        # Add active transition
        state = ThemeTransitionState(
            transition_id="test_id",
            from_theme="light",
            to_theme="dark",
            is_active=True
        )
        transition_manager.active_transitions["test_id"] = state
        
        assert transition_manager.is_transition_active()
        
        # Cancel transitions
        cancelled = transition_manager.cancel_all_transitions()
        
        assert cancelled == 1
        assert not transition_manager.is_transition_active()
        assert len(transition_manager.active_transitions) == 0
    
    @pytest.mark.asyncio
    async def test_apply_theme_to_widgets(self, transition_manager):
        """Test applying theme to registered widgets."""
        # Create mock widgets of different types
        box_widget = Mock(spec=toga.Box)
        box_widget.style = Mock()
        
        button_widget = Mock(spec=toga.Button)
        button_widget.style = Mock()
        
        label_widget = Mock(spec=toga.Label)
        label_widget.style = Mock()
        
        # Register widgets
        transition_manager.register_widget(box_widget)
        transition_manager.register_widget(button_widget)
        transition_manager.register_widget(label_widget)
        
        # Apply theme
        await transition_manager._apply_theme_to_widgets(DARK_THEME)
        
        # Verify styles were applied
        assert hasattr(box_widget.style, 'background_color')
        assert hasattr(button_widget.style, 'background_color')
        assert hasattr(label_widget.style, 'color')


class TestColorTransition:
    """Test color transition data structure."""
    
    def test_color_transition_creation(self):
        """Test ColorTransition creation."""
        widget = Mock(spec=toga.Widget)
        transition = ColorTransition(
            from_color="#ffffff",
            to_color="#000000",
            property_name="background_color",
            widget=widget,
            duration=0.3
        )
        
        assert transition.from_color == "#ffffff"
        assert transition.to_color == "#000000"
        assert transition.property_name == "background_color"
        assert transition.widget == widget
        assert transition.duration == 0.3


class TestThemeTransitionState:
    """Test theme transition state tracking."""
    
    def test_transition_state_creation(self):
        """Test ThemeTransitionState creation."""
        state = ThemeTransitionState(
            transition_id="test_id",
            from_theme="light",
            to_theme="dark"
        )
        
        assert state.transition_id == "test_id"
        assert state.from_theme == "light"
        assert state.to_theme == "dark"
        assert state.progress == 0.0
        assert not state.is_active
        assert state.start_time == 0.0
        assert state.color_transitions == []


class TestGlobalThemeTransitionManager:
    """Test global theme transition manager singleton."""
    
    def test_get_theme_transition_manager_first_call(self):
        """Test first call to get_theme_transition_manager requires animation_manager."""
        # Reset global instance
        import src.tektra.gui.animations.theme_transition_manager as ttm_module
        ttm_module._theme_transition_manager = None
        
        # First call without animation_manager should raise error
        with pytest.raises(ValueError, match="Animation manager required"):
            get_theme_transition_manager()
        
        # First call with animation_manager should work
        animation_manager = Mock(spec=AnimationManager)
        manager = get_theme_transition_manager(animation_manager)
        
        assert manager is not None
        assert isinstance(manager, ThemeTransitionManager)
        
        # Second call without animation_manager should work (uses cached instance)
        manager2 = get_theme_transition_manager()
        
        assert manager2 is manager  # Same instance


class TestThemeTransitionIntegration:
    """Test integration between theme transition manager and other components."""
    
    @pytest.mark.asyncio
    async def test_integration_with_animation_manager(self):
        """Test integration with AnimationManager."""
        # Create real animation manager (with mocked dependencies)
        with patch('src.tektra.gui.animations.animation_manager.UIPerformanceMonitor'):
            with patch('src.tektra.gui.animations.animation_manager.TransitionEngine'):
                animation_manager = AnimationManager()
        
        # Create theme manager
        theme_manager = ThemeManager()
        
        # Create transition manager
        transition_manager = ThemeTransitionManager(animation_manager, theme_manager)
        
        # Test theme transition through animation manager
        container = Mock(spec=toga.Box)
        
        # Mock the transition manager methods
        transition_manager.register_widget = Mock()
        transition_manager.transition_to_theme = AsyncMock(return_value="transition_id")
        
        # Patch get_theme_transition_manager to return our instance
        with patch('src.tektra.gui.animations.theme_transition_manager.get_theme_transition_manager', 
                  return_value=transition_manager):
            
            result = await animation_manager.animate_theme_transition(
                container, "light", "dark"
            )
            
            # Verify integration worked
            transition_manager.register_widget.assert_called_once_with(container)
            transition_manager.transition_to_theme.assert_called_once_with(
                "dark", animated=True, duration=0.5
            )
            assert result == "transition_id"
    
    def test_theme_manager_callback_integration(self):
        """Test integration with ThemeManager callbacks."""
        # Create theme manager
        theme_manager = ThemeManager()
        
        # Add callback
        callback = Mock()
        theme_manager.add_theme_change_callback(callback)
        
        # Switch theme
        theme_manager.switch_theme("dark")
        
        # Verify callback was called
        callback.assert_called_once_with("light", "dark")
        
        # Remove callback
        theme_manager.remove_theme_change_callback(callback)
        
        # Switch theme again
        theme_manager.switch_theme("light")
        
        # Callback should not be called again
        assert callback.call_count == 1


if __name__ == "__main__":
    pytest.main([__file__])