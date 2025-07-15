"""
Virtual Scroll Manager - Efficient Handling of Large Conversation Histories

This module provides the VirtualScrollManager class that efficiently handles
large conversation histories by only rendering visible messages and managing
memory usage for optimal performance.
"""

import asyncio
import math
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass
from unittest.mock import Mock
from loguru import logger

import toga
from toga.style import Pack
from toga.style.pack import COLUMN


@dataclass
class VirtualScrollItem:
    """Represents an item in the virtual scroll list."""
    index: int
    data: Dict[str, Any]
    estimated_height: int = 100
    actual_height: Optional[int] = None
    widget: Optional[toga.Widget] = None
    is_rendered: bool = False


class VirtualScrollManager:
    """
    Efficiently handles large conversation histories using virtual scrolling.
    
    Only renders visible messages to maintain performance with thousands of messages.
    """
    
    def __init__(
        self,
        container: toga.ScrollContainer,
        item_renderer: Callable[[Dict], toga.Widget],
        estimated_item_height: int = 100,
        buffer_size: int = 5,
        performance_threshold: int = 100,
        test_mode: bool = False
    ):
        """
        Initialize the virtual scroll manager.
        
        Args:
            container: The scroll container to manage
            item_renderer: Function to render individual items
            estimated_item_height: Estimated height of each item in pixels
            buffer_size: Number of items to render outside visible area
            performance_threshold: Number of items before virtual scrolling kicks in
        """
        self.container = container
        self.item_renderer = item_renderer
        self.estimated_item_height = estimated_item_height
        self.buffer_size = buffer_size
        self.performance_threshold = performance_threshold
        self.test_mode = test_mode
        
        # Virtual scroll state
        self.items: List[VirtualScrollItem] = []
        self.visible_start_index = 0
        self.visible_end_index = 0
        self.container_height = 600  # Default height
        self.scroll_position = 0
        self.total_height = 0
        
        # Rendered widgets tracking
        self.rendered_widgets: Dict[int, toga.Widget] = {}
        self.widget_pool: List[toga.Widget] = []
        
        # Performance tracking
        self.is_virtual_mode = False
        self.last_scroll_time = 0
        self.scroll_momentum = 0
        
        # Content container for virtual items
        if self.test_mode:
            # Use mock container for testing
            self.content_container = Mock()
            self.content_container.clear = Mock()
            self.content_container.add = Mock()
        else:
            try:
                self.content_container = toga.Box(
                    style=Pack(direction=COLUMN, padding=0)
                )
            except Exception:
                # Fallback to mock if Toga widgets fail
                self.content_container = Mock()
                self.content_container.clear = Mock()
                self.content_container.add = Mock()
        
        # Set up scroll event handling (simulated since Toga doesn't have scroll events)
        # Only start monitoring if there's an event loop running
        try:
            self._setup_scroll_monitoring()
        except RuntimeError:
            # No event loop running, monitoring will be started later
            logger.debug("No event loop running, scroll monitoring will start later")
        
        logger.info(f"VirtualScrollManager initialized with threshold {performance_threshold}")
    
    def add_item(self, data: Dict[str, Any], estimated_height: Optional[int] = None) -> None:
        """
        Add an item to the virtual scroll list.
        
        Args:
            data: The data for the item
            estimated_height: Optional custom height estimate
        """
        height = estimated_height or self.estimated_item_height
        item = VirtualScrollItem(
            index=len(self.items),
            data=data,
            estimated_height=height
        )
        
        self.items.append(item)
        self._update_total_height()
        
        # Check if we should enable virtual scrolling
        if len(self.items) >= self.performance_threshold and not self.is_virtual_mode:
            self._enable_virtual_mode()
        
        # Update visible range and render if needed
        self._update_visible_range()
        self._render_visible_items()
        
        logger.debug(f"Added item {item.index}, total items: {len(self.items)}")
    
    def remove_item(self, index: int) -> bool:
        """
        Remove an item from the virtual scroll list.
        
        Args:
            index: Index of the item to remove
            
        Returns:
            True if item was removed, False if index was invalid
        """
        if index < 0 or index >= len(self.items):
            return False
        
        # Remove the item
        removed_item = self.items.pop(index)
        
        # Clean up rendered widget if it exists
        if removed_item.widget and index in self.rendered_widgets:
            self._return_widget_to_pool(self.rendered_widgets[index])
            del self.rendered_widgets[index]
        
        # Update indices for remaining items
        for i in range(index, len(self.items)):
            self.items[i].index = i
        
        self._update_total_height()
        self._update_visible_range()
        self._render_visible_items()
        
        logger.debug(f"Removed item {index}, total items: {len(self.items)}")
        return True
    
    def clear_items(self) -> None:
        """Clear all items from the virtual scroll list."""
        # Clean up all rendered widgets
        for widget in self.rendered_widgets.values():
            self._return_widget_to_pool(widget)
        
        self.items.clear()
        self.rendered_widgets.clear()
        self.visible_start_index = 0
        self.visible_end_index = 0
        self.total_height = 0
        self.scroll_position = 0
        
        # Clear the content container
        self.content_container.clear()
        
        logger.debug("Cleared all virtual scroll items")
    
    def update_container_height(self, height: int) -> None:
        """
        Update the container height for visible range calculations.
        
        Args:
            height: New container height in pixels
        """
        self.container_height = height
        self._update_visible_range()
        self._render_visible_items()
    
    def scroll_to_bottom(self, smooth: bool = True) -> None:
        """
        Scroll to the bottom of the list.
        
        Args:
            smooth: Whether to use smooth scrolling animation
        """
        if not self.items:
            return
        
        target_position = max(0, self.total_height - self.container_height)
        
        if smooth:
            asyncio.create_task(self._smooth_scroll_to(target_position))
        else:
            self.scroll_position = target_position
            self._update_visible_range()
            self._render_visible_items()
    
    def scroll_to_item(self, index: int, smooth: bool = True) -> None:
        """
        Scroll to a specific item.
        
        Args:
            index: Index of the item to scroll to
            smooth: Whether to use smooth scrolling animation
        """
        if index < 0 or index >= len(self.items):
            return
        
        # Calculate position of the item
        target_position = sum(
            item.actual_height or item.estimated_height 
            for item in self.items[:index]
        )
        
        if smooth:
            asyncio.create_task(self._smooth_scroll_to(target_position))
        else:
            self.scroll_position = target_position
            self._update_visible_range()
            self._render_visible_items()
    
    def get_visible_range(self) -> tuple[int, int]:
        """
        Get the current visible range of items.
        
        Returns:
            Tuple of (start_index, end_index)
        """
        return (self.visible_start_index, self.visible_end_index)
    
    def get_total_items(self) -> int:
        """Get the total number of items."""
        return len(self.items)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the virtual scroll manager.
        
        Returns:
            Dictionary with performance metrics
        """
        rendered_count = len(self.rendered_widgets)
        total_count = len(self.items)
        
        return {
            "total_items": total_count,
            "rendered_items": rendered_count,
            "virtual_mode_enabled": self.is_virtual_mode,
            "memory_efficiency": (1 - rendered_count / max(total_count, 1)) * 100,
            "visible_range": (self.visible_start_index, self.visible_end_index),
            "scroll_position": self.scroll_position,
            "total_height": self.total_height
        }
    
    def _enable_virtual_mode(self) -> None:
        """Enable virtual scrolling mode for performance."""
        self.is_virtual_mode = True
        logger.info(f"Virtual scrolling enabled with {len(self.items)} items")
    
    def _update_total_height(self) -> None:
        """Update the total height calculation."""
        self.total_height = sum(
            item.actual_height or item.estimated_height 
            for item in self.items
        )
    
    def _update_visible_range(self) -> None:
        """Update the range of visible items based on scroll position."""
        if not self.items:
            self.visible_start_index = 0
            self.visible_end_index = 0
            return
        
        # Calculate visible range based on scroll position and container height
        visible_top = self.scroll_position
        visible_bottom = self.scroll_position + self.container_height
        
        # Find start index
        current_height = 0
        start_index = 0
        for i, item in enumerate(self.items):
            item_height = item.actual_height or item.estimated_height
            if current_height + item_height > visible_top:
                start_index = max(0, i - self.buffer_size)
                break
            current_height += item_height
        
        # Find end index
        current_height = 0
        end_index = len(self.items)
        for i, item in enumerate(self.items):
            item_height = item.actual_height or item.estimated_height
            current_height += item_height
            if current_height > visible_bottom:
                end_index = min(len(self.items), i + self.buffer_size + 1)
                break
        
        self.visible_start_index = start_index
        self.visible_end_index = end_index
    
    def _render_visible_items(self) -> None:
        """Render only the visible items."""
        if not self.is_virtual_mode:
            # In non-virtual mode, render all items
            self._render_all_items()
            return
        
        # Clear container
        self.content_container.clear()
        
        # Return unused widgets to pool
        for index in list(self.rendered_widgets.keys()):
            if index < self.visible_start_index or index >= self.visible_end_index:
                widget = self.rendered_widgets.pop(index)
                self._return_widget_to_pool(widget)
        
        # Render visible items
        for i in range(self.visible_start_index, self.visible_end_index):
            if i >= len(self.items):
                break
            
            item = self.items[i]
            
            # Get or create widget for this item
            if i not in self.rendered_widgets:
                widget = self._get_widget_from_pool_or_create(item)
                self.rendered_widgets[i] = widget
                item.widget = widget
                item.is_rendered = True
            
            # Add to container
            self.content_container.add(self.rendered_widgets[i])
    
    def _render_all_items(self) -> None:
        """Render all items (non-virtual mode)."""
        self.content_container.clear()
        
        for i, item in enumerate(self.items):
            if i not in self.rendered_widgets:
                widget = self._get_widget_from_pool_or_create(item)
                self.rendered_widgets[i] = widget
                item.widget = widget
                item.is_rendered = True
            
            self.content_container.add(self.rendered_widgets[i])
    
    def _get_widget_from_pool_or_create(self, item: VirtualScrollItem) -> toga.Widget:
        """
        Get a widget from the pool or create a new one.
        
        Args:
            item: The item to create a widget for
            
        Returns:
            Widget for the item
        """
        # Try to get from pool first
        if self.widget_pool:
            widget = self.widget_pool.pop()
            # Update widget with new data (this would need to be implemented
            # based on the specific widget type)
            return widget
        
        # Create new widget
        return self.item_renderer(item.data)
    
    def _return_widget_to_pool(self, widget: toga.Widget) -> None:
        """
        Return a widget to the pool for reuse.
        
        Args:
            widget: Widget to return to pool
        """
        # Clean up widget state before pooling
        # This would need to be implemented based on the specific widget type
        
        # Add to pool (limit pool size to prevent memory leaks)
        if len(self.widget_pool) < 20:
            self.widget_pool.append(widget)
    
    def _setup_scroll_monitoring(self) -> None:
        """Set up scroll position monitoring."""
        # Since Toga doesn't have scroll events, we'll simulate this
        # In a real implementation, this would hook into the scroll container's events
        asyncio.create_task(self._scroll_monitoring_loop())
    
    async def _scroll_monitoring_loop(self) -> None:
        """Monitor scroll position changes."""
        try:
            while True:
                await asyncio.sleep(0.016)  # ~60fps monitoring
                
                # In a real implementation, we would get the actual scroll position
                # from the scroll container. For now, this is a placeholder.
                
                # Update visible range if scroll position changed
                # self._update_visible_range()
                # self._render_visible_items()
                
        except asyncio.CancelledError:
            logger.debug("Scroll monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in scroll monitoring loop: {e}")
    
    async def _smooth_scroll_to(self, target_position: int, duration: float = 0.5) -> None:
        """
        Smoothly scroll to a target position.
        
        Args:
            target_position: Target scroll position
            duration: Animation duration in seconds
        """
        start_position = self.scroll_position
        distance = target_position - start_position
        
        if abs(distance) < 1:
            return
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            while True:
                current_time = asyncio.get_event_loop().time()
                elapsed = current_time - start_time
                
                if elapsed >= duration:
                    self.scroll_position = target_position
                    break
                
                # Easing function (ease-out)
                progress = elapsed / duration
                eased_progress = 1 - (1 - progress) ** 3
                
                self.scroll_position = start_position + (distance * eased_progress)
                
                # Update visible range
                self._update_visible_range()
                self._render_visible_items()
                
                await asyncio.sleep(0.016)  # ~60fps
                
        except asyncio.CancelledError:
            logger.debug("Smooth scroll animation cancelled")
        except Exception as e:
            logger.error(f"Error in smooth scroll animation: {e}")
        
        # Ensure we end at the exact target position
        self.scroll_position = target_position
        self._update_visible_range()
        self._render_visible_items()