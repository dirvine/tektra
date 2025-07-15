"""
Tests for Smooth Scrolling and Conversation Flow System

This module tests the enhanced scrolling capabilities including:
- VirtualScrollManager for large conversations
- SmoothScrollContainer with momentum
- ConversationScrollManager for chat-specific behaviors
- ScrollPerformanceOptimizer for 60fps performance
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

import toga
from toga.style import Pack

from src.tektra.gui.virtual_scroll_manager import VirtualScrollManager, VirtualScrollItem
from src.tektra.gui.smooth_scroll_container import SmoothScrollContainer, ConversationScrollManager
from src.tektra.gui.scroll_performance_optimizer import ScrollPerformanceOptimizer, ScrollFrameRateMonitor
from src.tektra.gui.animations.animation_manager import AnimationManager


class TestVirtualScrollManager:
    """Test the VirtualScrollManager for efficient large conversation handling."""
    
    @pytest.fixture
    def mock_container(self):
        """Create a mock scroll container."""
        container = Mock(spec=toga.ScrollContainer)
        return container
    
    @pytest.fixture
    def mock_item_renderer(self):
        """Create a mock item renderer function."""
        def renderer(data):
            widget = Mock(spec=toga.Widget)
            widget.data = data
            return widget
        return renderer
    
    @pytest.fixture
    def virtual_scroll_manager(self, mock_container, mock_item_renderer):
        """Create a VirtualScrollManager instance for testing."""
        return VirtualScrollManager(
            container=mock_container,
            item_renderer=mock_item_renderer,
            estimated_item_height=100,
            buffer_size=5,
            performance_threshold=10,  # Low threshold for testing
            test_mode=True  # Enable test mode to use mocks
        )
    
    def test_initialization(self, virtual_scroll_manager):
        """Test VirtualScrollManager initialization."""
        assert virtual_scroll_manager.estimated_item_height == 100
        assert virtual_scroll_manager.buffer_size == 5
        assert virtual_scroll_manager.performance_threshold == 10
        assert len(virtual_scroll_manager.items) == 0
        assert not virtual_scroll_manager.is_virtual_mode
    
    def test_add_item(self, virtual_scroll_manager):
        """Test adding items to the virtual scroll manager."""
        # Add a few items
        for i in range(5):
            data = {"role": "user", "content": f"Message {i}"}
            virtual_scroll_manager.add_item(data)
        
        assert len(virtual_scroll_manager.items) == 5
        assert virtual_scroll_manager.items[0].index == 0
        assert virtual_scroll_manager.items[4].index == 4
        assert not virtual_scroll_manager.is_virtual_mode  # Below threshold
    
    def test_virtual_mode_activation(self, virtual_scroll_manager):
        """Test that virtual mode activates when threshold is reached."""
        # Add items beyond threshold
        for i in range(15):
            data = {"role": "user", "content": f"Message {i}"}
            virtual_scroll_manager.add_item(data)
        
        assert len(virtual_scroll_manager.items) == 15
        assert virtual_scroll_manager.is_virtual_mode  # Above threshold
    
    def test_remove_item(self, virtual_scroll_manager):
        """Test removing items from the virtual scroll manager."""
        # Add some items
        for i in range(5):
            data = {"role": "user", "content": f"Message {i}"}
            virtual_scroll_manager.add_item(data)
        
        # Remove middle item
        success = virtual_scroll_manager.remove_item(2)
        
        assert success
        assert len(virtual_scroll_manager.items) == 4
        # Check that indices were updated correctly
        assert virtual_scroll_manager.items[2].data["content"] == "Message 3"
        assert virtual_scroll_manager.items[2].index == 2
    
    def test_clear_items(self, virtual_scroll_manager):
        """Test clearing all items."""
        # Add some items
        for i in range(5):
            data = {"role": "user", "content": f"Message {i}"}
            virtual_scroll_manager.add_item(data)
        
        virtual_scroll_manager.clear_items()
        
        assert len(virtual_scroll_manager.items) == 0
        assert virtual_scroll_manager.visible_start_index == 0
        assert virtual_scroll_manager.visible_end_index == 0
    
    def test_scroll_to_bottom(self, virtual_scroll_manager):
        """Test scrolling to bottom functionality."""
        # Add items to create scrollable content
        for i in range(20):
            data = {"role": "user", "content": f"Message {i}"}
            virtual_scroll_manager.add_item(data, estimated_height=50)
        
        # Scroll to bottom
        virtual_scroll_manager.scroll_to_bottom(smooth=False)
        
        # Should be at the bottom position
        expected_position = max(0, virtual_scroll_manager.total_height - virtual_scroll_manager.container_height)
        assert virtual_scroll_manager.scroll_position == expected_position
    
    def test_performance_stats(self, virtual_scroll_manager):
        """Test performance statistics reporting."""
        # Add items
        for i in range(15):
            data = {"role": "user", "content": f"Message {i}"}
            virtual_scroll_manager.add_item(data)
        
        stats = virtual_scroll_manager.get_performance_stats()
        
        assert stats["total_items"] == 15
        assert stats["virtual_mode_enabled"] == True
        assert "memory_efficiency" in stats
        assert "visible_range" in stats


class TestSmoothScrollContainer:
    """Test the SmoothScrollContainer for enhanced scrolling experience."""
    
    @pytest.fixture
    def mock_content(self):
        """Create mock content widget."""
        mock_content = Mock(spec=toga.Widget)
        # Add _impl attribute to prevent Toga initialization errors
        mock_content._impl = Mock()
        return mock_content
    
    @pytest.fixture
    def mock_animation_manager(self):
        """Create mock animation manager."""
        return Mock(spec=AnimationManager)
    
    @pytest.fixture
    def smooth_scroll_container(self, mock_content, mock_animation_manager):
        """Create a SmoothScrollContainer for testing."""
        return SmoothScrollContainer(
            content=mock_content,
            animation_manager=mock_animation_manager,
            enable_momentum=True,
            momentum_decay=0.95,
            auto_scroll_threshold=50.0
        )
    
    def test_initialization(self, smooth_scroll_container):
        """Test SmoothScrollContainer initialization."""
        assert smooth_scroll_container.enable_momentum == True
        assert smooth_scroll_container.momentum_decay == 0.95
        assert smooth_scroll_container.auto_scroll_threshold == 50.0
        assert smooth_scroll_container.scroll_position == 0.0
        assert smooth_scroll_container.scroll_velocity == 0.0
    
    def test_scroll_to_position(self, smooth_scroll_container):
        """Test scrolling to a specific position."""
        # Set up container dimensions
        smooth_scroll_container.update_content_dimensions(1000, 500)
        
        # Scroll to position
        smooth_scroll_container.scroll_to_position(200, smooth=False)
        
        assert smooth_scroll_container.scroll_position == 200
        assert smooth_scroll_container.target_scroll_position == 200
    
    def test_scroll_by_delta(self, smooth_scroll_container):
        """Test scrolling by a relative amount."""
        # Set up container dimensions
        smooth_scroll_container.update_content_dimensions(1000, 500)
        
        # Start at position 100
        smooth_scroll_container.scroll_to_position(100, smooth=False)
        
        # Scroll by 50 pixels
        smooth_scroll_container.scroll_by_delta(50, smooth=False)
        
        assert smooth_scroll_container.scroll_position == 150
    
    def test_momentum_scrolling(self, smooth_scroll_container):
        """Test momentum scrolling functionality."""
        # Add momentum
        smooth_scroll_container.add_momentum(100.0)
        
        assert smooth_scroll_container.scroll_velocity == 100.0
        
        # Add more momentum
        smooth_scroll_container.add_momentum(50.0)
        
        assert smooth_scroll_container.scroll_velocity == 150.0
    
    def test_auto_scroll_enabled(self, smooth_scroll_container):
        """Test auto-scroll enable/disable functionality."""
        assert smooth_scroll_container.auto_scroll_enabled == True
        
        smooth_scroll_container.set_auto_scroll_enabled(False)
        
        assert smooth_scroll_container.auto_scroll_enabled == False
    
    def test_performance_mode(self, smooth_scroll_container):
        """Test performance mode settings."""
        smooth_scroll_container.set_performance_mode("performance")
        assert smooth_scroll_container.performance_mode == "performance"
        
        smooth_scroll_container.set_performance_mode("quality")
        assert smooth_scroll_container.performance_mode == "quality"
    
    def test_scroll_info(self, smooth_scroll_container):
        """Test scroll information reporting."""
        # Set up some state
        smooth_scroll_container.update_content_dimensions(1000, 500)
        smooth_scroll_container.scroll_to_position(200, smooth=False)
        smooth_scroll_container.add_momentum(50.0)
        
        info = smooth_scroll_container.get_scroll_info()
        
        assert info["scroll_position"] == 200
        assert info["velocity"] == 50.0
        assert info["content_height"] == 1000
        assert info["container_height"] == 500
        assert "is_at_bottom" in info
        assert "scroll_percentage" in info


class TestConversationScrollManager:
    """Test the ConversationScrollManager for chat-specific scrolling."""
    
    @pytest.fixture
    def mock_smooth_scroll_container(self):
        """Create mock smooth scroll container."""
        container = Mock(spec=SmoothScrollContainer)
        container.auto_scroll_enabled = True
        container.scroll_to_bottom = Mock()
        return container
    
    @pytest.fixture
    def mock_animation_manager(self):
        """Create mock animation manager."""
        return Mock(spec=AnimationManager)
    
    @pytest.fixture
    def conversation_scroll_manager(self, mock_smooth_scroll_container, mock_animation_manager):
        """Create a ConversationScrollManager for testing."""
        return ConversationScrollManager(
            smooth_scroll_container=mock_smooth_scroll_container,
            animation_manager=mock_animation_manager
        )
    
    def test_initialization(self, conversation_scroll_manager):
        """Test ConversationScrollManager initialization."""
        assert conversation_scroll_manager.auto_scroll_on_new_message == True
        assert conversation_scroll_manager.auto_scroll_on_user_message == True
        assert conversation_scroll_manager.auto_scroll_on_assistant_message == True
    
    @pytest.mark.asyncio
    async def test_on_new_user_message(self, conversation_scroll_manager, mock_smooth_scroll_container):
        """Test handling new user message."""
        await conversation_scroll_manager.on_new_message("user", animate=True)
        
        # Should trigger auto-scroll for user messages
        assert mock_smooth_scroll_container.scroll_to_bottom.called
    
    @pytest.mark.asyncio
    async def test_on_new_assistant_message(self, conversation_scroll_manager, mock_smooth_scroll_container):
        """Test handling new assistant message."""
        await conversation_scroll_manager.on_new_message("assistant", animate=True)
        
        # Should trigger auto-scroll for assistant messages
        assert mock_smooth_scroll_container.scroll_to_bottom.called
    
    @pytest.mark.asyncio
    async def test_typing_indicator_scroll(self, conversation_scroll_manager, mock_smooth_scroll_container):
        """Test scrolling when typing indicator is shown."""
        await conversation_scroll_manager.on_typing_indicator_shown()
        
        # Should scroll to show typing indicator
        assert mock_smooth_scroll_container.scroll_to_bottom.called
    
    def test_auto_scroll_preferences(self, conversation_scroll_manager):
        """Test setting auto-scroll preferences."""
        conversation_scroll_manager.set_auto_scroll_preferences(
            on_user_message=False,
            on_assistant_message=True,
            on_system_message=True
        )
        
        assert conversation_scroll_manager.auto_scroll_on_user_message == False
        assert conversation_scroll_manager.auto_scroll_on_assistant_message == True
        assert conversation_scroll_manager.auto_scroll_on_system_message == True


class TestScrollPerformanceOptimizer:
    """Test the ScrollPerformanceOptimizer for maintaining 60fps."""
    
    @pytest.fixture
    def performance_optimizer(self):
        """Create a ScrollPerformanceOptimizer for testing."""
        return ScrollPerformanceOptimizer(
            target_fps=60.0,
            performance_window=10,  # Small window for testing
            optimization_threshold=45.0
        )
    
    def test_initialization(self, performance_optimizer):
        """Test ScrollPerformanceOptimizer initialization."""
        assert performance_optimizer.target_fps == 60.0
        assert performance_optimizer.optimization_threshold == 45.0
        assert performance_optimizer.optimization_level == 0
        assert performance_optimizer.monitoring_enabled == True
    
    def test_record_good_performance(self, performance_optimizer):
        """Test recording good performance metrics."""
        # Record good frame times (60fps = ~16.67ms per frame)
        for _ in range(10):
            performance_optimizer.record_frame_metrics(
                frame_time=0.016,  # 16ms = ~62fps
                memory_usage=100.0,
                rendered_items=50
            )
        
        summary = performance_optimizer.get_performance_summary()
        
        assert summary["current_fps"] > 55  # Should be around 60fps
        assert summary["optimization_level"] == 0  # No optimization needed
        assert summary["performance_status"] == "excellent"
    
    def test_record_poor_performance(self, performance_optimizer):
        """Test recording poor performance metrics."""
        # Record poor frame times (30fps = ~33ms per frame)
        for _ in range(10):
            performance_optimizer.record_frame_metrics(
                frame_time=0.033,  # 33ms = ~30fps
                memory_usage=200.0,
                rendered_items=100
            )
        
        summary = performance_optimizer.get_performance_summary()
        
        assert summary["current_fps"] < 35  # Should be around 30fps
        assert summary["performance_status"] in ["fair", "poor"]
    
    def test_optimization_callbacks(self, performance_optimizer):
        """Test optimization callback system."""
        callback_called = False
        
        def test_callback():
            nonlocal callback_called
            callback_called = True
        
        performance_optimizer.add_optimization_callback("reduce_animations", test_callback)
        
        # Force optimization level that triggers animation reduction
        performance_optimizer.force_optimization_level(1)
        
        assert callback_called == True
    
    def test_force_optimization_level(self, performance_optimizer):
        """Test forcing specific optimization levels."""
        performance_optimizer.force_optimization_level(2)
        
        assert performance_optimizer.optimization_level == 2
        assert performance_optimizer.optimizations_active["virtual_scrolling"] == True
        assert performance_optimizer.optimizations_active["batch_rendering"] == True
    
    def test_reset_optimizations(self, performance_optimizer):
        """Test resetting all optimizations."""
        # Set some optimizations
        performance_optimizer.force_optimization_level(3)
        
        # Reset
        performance_optimizer.reset_optimizations()
        
        assert performance_optimizer.optimization_level == 0
        assert all(not active for active in performance_optimizer.optimizations_active.values())
    
    def test_optimization_recommendations(self, performance_optimizer):
        """Test getting optimization recommendations."""
        # Record poor performance to trigger recommendations
        for _ in range(10):
            performance_optimizer.record_frame_metrics(
                frame_time=0.050,  # 50ms = 20fps (very poor)
                memory_usage=600.0,  # High memory usage
                rendered_items=200
            )
        
        recommendations = performance_optimizer.get_optimization_recommendations()
        
        assert len(recommendations) > 0
        assert any("virtual scrolling" in rec.lower() for rec in recommendations)
        assert any("memory" in rec.lower() for rec in recommendations)


class TestScrollFrameRateMonitor:
    """Test the ScrollFrameRateMonitor for detailed frame rate analysis."""
    
    @pytest.fixture
    def frame_rate_monitor(self):
        """Create a ScrollFrameRateMonitor for testing."""
        return ScrollFrameRateMonitor(target_fps=60.0)
    
    def test_initialization(self, frame_rate_monitor):
        """Test ScrollFrameRateMonitor initialization."""
        assert frame_rate_monitor.target_fps == 60.0
        assert frame_rate_monitor.current_quality == 1.0
        assert frame_rate_monitor.quality_adjustment_enabled == True
        assert frame_rate_monitor.dropped_frames == 0
    
    def test_record_good_frames(self, frame_rate_monitor):
        """Test recording good frame performance."""
        # Simulate good frame times
        with patch('time.time', side_effect=[0.0, 0.016, 0.032, 0.048]):  # 60fps timing
            frame_info = frame_rate_monitor.record_frame()
            frame_info = frame_rate_monitor.record_frame()
            frame_info = frame_rate_monitor.record_frame()
        
        avg_fps = frame_rate_monitor.get_average_fps()
        grade = frame_rate_monitor.get_performance_grade()
        
        assert avg_fps > 55  # Should be close to 60fps
        assert grade == "A"  # Excellent performance
        assert frame_rate_monitor.current_quality == 1.0  # Full quality maintained
    
    def test_record_poor_frames(self, frame_rate_monitor):
        """Test recording poor frame performance."""
        # Manually set frame times to simulate poor performance
        frame_rate_monitor.frame_times = [0.050, 0.050, 0.050]  # 20fps frame times
        frame_rate_monitor.last_frame_time = 0.0
        
        # Simulate recording a poor frame
        with patch('time.time', return_value=0.050):
            frame_info = frame_rate_monitor.record_frame()
        
        avg_fps = frame_rate_monitor.get_average_fps()
        grade = frame_rate_monitor.get_performance_grade()
        
        assert avg_fps < 25  # Should be around 20fps
        assert grade in ["D", "F"]  # Poor performance
        assert frame_rate_monitor.current_quality < 1.0  # Quality should be reduced
    
    def test_quality_adjustment_disable(self, frame_rate_monitor):
        """Test disabling automatic quality adjustment."""
        frame_rate_monitor.set_quality_adjustment_enabled(False)
        
        # Simulate poor performance
        with patch('time.time', side_effect=[0.0, 0.050]):  # 20fps
            frame_rate_monitor.record_frame()
        
        # Quality should remain at 1.0 despite poor performance
        assert frame_rate_monitor.current_quality == 1.0
    
    def test_reset_statistics(self, frame_rate_monitor):
        """Test resetting frame rate statistics."""
        # Record some frames
        with patch('time.time', side_effect=[0.0, 0.050, 0.100]):
            frame_rate_monitor.record_frame()
            frame_rate_monitor.record_frame()
        
        # Reset
        frame_rate_monitor.reset_statistics()
        
        assert len(frame_rate_monitor.frame_times) == 0
        assert frame_rate_monitor.dropped_frames == 0
        assert frame_rate_monitor.total_frames == 0
        assert frame_rate_monitor.current_quality == 1.0


class TestIntegratedScrollingSystem:
    """Test the integrated scrolling system working together."""
    
    @pytest.fixture
    def mock_chat_panel_components(self):
        """Create mock components for integrated testing."""
        animation_manager = Mock(spec=AnimationManager)
        
        # Mock scroll container
        scroll_container = Mock(spec=toga.ScrollContainer)
        
        # Mock content container
        content_container = Mock(spec=toga.Box)
        content_container.clear = Mock()
        content_container.add = Mock()
        
        return {
            "animation_manager": animation_manager,
            "scroll_container": scroll_container,
            "content_container": content_container
        }
    
    @pytest.mark.asyncio
    async def test_message_addition_with_scrolling(self, mock_chat_panel_components):
        """Test that adding messages triggers appropriate scrolling behavior."""
        # Create mock smooth scroll container instead of real one
        mock_smooth_scroll = Mock(spec=SmoothScrollContainer)
        mock_smooth_scroll.auto_scroll_enabled = True
        mock_smooth_scroll.scroll_to_bottom = Mock()
        
        conversation_scroll = ConversationScrollManager(
            smooth_scroll_container=mock_smooth_scroll,
            animation_manager=mock_chat_panel_components["animation_manager"]
        )
        
        # Test adding different types of messages
        await conversation_scroll.on_new_message("user", animate=True)
        await conversation_scroll.on_new_message("assistant", animate=True)
        
        # Should have triggered scroll operations
        assert mock_smooth_scroll.scroll_to_bottom.call_count == 2
    
    def test_performance_optimization_integration(self):
        """Test that performance optimization works with the scrolling system."""
        optimizer = ScrollPerformanceOptimizer(target_fps=60.0)
        frame_monitor = ScrollFrameRateMonitor(target_fps=60.0)
        
        # Simulate performance degradation
        for _ in range(20):
            optimizer.record_frame_metrics(frame_time=0.040)  # 25fps
        
        # Should trigger optimizations
        summary = optimizer.get_performance_summary()
        assert summary["optimization_level"] > 0
        
        # Frame monitor should also detect poor performance
        # Manually set poor frame times
        frame_monitor.frame_times = [0.040] * 10  # 25fps frame times
        
        grade = frame_monitor.get_performance_grade()
        assert grade in ["B", "C", "D", "F"]  # Poor performance detected (B is acceptable too)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])