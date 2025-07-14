"""
Progress Overlay Component

An in-app overlay that shows progress without opening a separate window.
"""

from datetime import datetime
from typing import Optional, Callable

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from .themes import theme_manager


class ProgressOverlay:
    """
    In-app progress overlay that covers the main content during loading.
    """
    
    def __init__(self, parent_container: toga.Box):
        """
        Initialize the progress overlay.
        
        Args:
            parent_container: The parent container to overlay
        """
        self.parent_container = parent_container
        self.overlay_box = None
        self.is_visible = False
        self.cancelled = False
        self.start_time = datetime.now()
        self._last_update = None
        
        # Build overlay components
        self._build_overlay()
        
    def _build_overlay(self):
        """Build the overlay UI components."""
        theme = theme_manager.get_theme()
        colors = theme.colors
        spacing = theme.spacing
        typography = theme.typography
        
        # Main overlay container
        self.overlay_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                background_color="#00000088",  # Semi-transparent background
                flex=1,
                padding=spacing["xl"]
            )
        )
        
        # Center container for progress content
        center_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                flex=1
            )
        )
        
        # Progress card
        self.progress_card = toga.Box(
            style=Pack(
                direction=COLUMN,
                background_color=colors.card,
                padding=spacing["xl"],
                width=600
            )
        )
        
        # Title
        self.title_label = toga.Label(
            "Loading Tektra AI Assistant",
            style=Pack(
                font_size=typography["heading2"]["size"],
                font_weight=typography["heading2"]["weight"],
                color=colors.primary,
                margin_bottom=spacing["md"]
            )
        )
        self.progress_card.add(self.title_label)
        
        # Operation label
        self.operation_label = toga.Label(
            "Initializing...",
            style=Pack(
                font_size=typography["body1"]["size"],
                color=colors.text_primary,
                margin_bottom=spacing["sm"]
            )
        )
        self.progress_card.add(self.operation_label)
        
        # Detail label
        self.detail_label = toga.Label(
            "",
            style=Pack(
                font_size=typography["caption"]["size"],
                color=colors.text_secondary,
                margin_bottom=spacing["md"],
                height=60  # Fixed height for multi-line
            )
        )
        self.progress_card.add(self.detail_label)
        
        # Progress bar container
        progress_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                margin_bottom=spacing["md"]
            )
        )
        
        # Progress bar background
        self.progress_bg = toga.Box(
            style=Pack(
                height=8,
                background_color=colors.surface,
                width=600
            )
        )
        
        # Progress bar fill
        self.progress_fill = toga.Box(
            style=Pack(
                height=8,
                background_color=colors.primary,
                width=0
            )
        )
        self.progress_bg.add(self.progress_fill)
        progress_container.add(self.progress_bg)
        
        # Progress percentage
        self.progress_label = toga.Label(
            "0%",
            style=Pack(
                color=colors.text_secondary,
                font_size=typography["caption"]["size"],
                margin_top=spacing["xs"]
            )
        )
        progress_container.add(self.progress_label)
        
        self.progress_card.add(progress_container)
        
        # Stats row
        stats_row = toga.Box(
            style=Pack(
                direction=ROW,
                margin_bottom=spacing["md"]
            )
        )
        
        # Speed label
        self.speed_label = toga.Label(
            "",
            style=Pack(
                flex=1,
                color=colors.text_secondary,
                font_size=typography["caption"]["size"]
            )
        )
        stats_row.add(self.speed_label)
        
        # Time remaining label
        self.time_label = toga.Label(
            "",
            style=Pack(
                flex=1,
                color=colors.text_secondary,
                font_size=typography["caption"]["size"]
            )
        )
        stats_row.add(self.time_label)
        
        self.progress_card.add(stats_row)
        
        # Cancel button
        button_row = toga.Box(
            style=Pack(
                direction=ROW
            )
        )
        
        self.cancel_button = toga.Button(
            "Cancel",
            on_press=self._on_cancel,
            style=Pack(
                padding=(spacing["sm"], spacing["lg"]),
                background_color=colors.surface,
                color=colors.text_secondary
            )
        )
        button_row.add(self.cancel_button)
        
        self.progress_card.add(button_row)
        
        # Add card to center box
        center_box.add(self.progress_card)
        
        # Add center box to overlay
        self.overlay_box.add(center_box)
        
    def show(self, title: str = "Loading...", cancellable: bool = True):
        """
        Show the progress overlay.
        
        Args:
            title: The title to display
            cancellable: Whether to show cancel button
        """
        if not self.is_visible:
            self.title_label.text = title
            self.cancel_button.enabled = cancellable
            self.cancelled = False
            self.start_time = datetime.now()
            self._last_update = None
            
            # Add overlay to parent
            self.parent_container.add(self.overlay_box)
            self.is_visible = True
            
            logger.debug(f"Progress overlay shown: {title}")
    
    def hide(self):
        """Hide the progress overlay."""
        if self.is_visible:
            try:
                self.parent_container.remove(self.overlay_box)
                self.is_visible = False
                logger.debug("Progress overlay hidden")
            except Exception as e:
                logger.error(f"Error hiding progress overlay: {e}")
    
    def update_progress(
        self,
        progress: float,
        operation: str = "",
        bytes_downloaded: int = 0,
        total_bytes: int = 0
    ):
        """
        Update progress overlay.
        
        Args:
            progress: Progress percentage (0-100)
            operation: Current operation description
            bytes_downloaded: Bytes downloaded so far
            total_bytes: Total bytes to download
        """
        if not self.is_visible or self.cancelled:
            return
            
        # Update progress bar
        progress = max(0, min(100, progress))
        # Calculate width based on percentage (600px total width)
        progress_width = int(600 * progress / 100)
        self.progress_fill.style.width = progress_width
        self.progress_label.text = f"{int(progress)}%"
        
        # Update operation text
        if operation:
            # Split multi-line operations
            lines = operation.split('\n', 1)
            if len(lines) > 1:
                self.operation_label.text = lines[0]
                # Clean up detail text
                detail_text = lines[1].strip()
                if detail_text:
                    self.detail_label.text = detail_text
            else:
                self.operation_label.text = operation
                if not operation.endswith("..."):
                    self.detail_label.text = ""
        
        # Update download stats
        if total_bytes > 0 and bytes_downloaded > 0:
            # Track progress for speed calculation
            if not hasattr(self, '_last_update') or self._last_update is None:
                self._last_update = {
                    'time': self.start_time,
                    'bytes': 0
                }
            
            # Calculate speed based on recent progress
            current_time = datetime.now()
            time_diff = (current_time - self._last_update['time']).total_seconds()
            
            if time_diff > 0.5:  # Update every 0.5 seconds
                bytes_diff = bytes_downloaded - self._last_update['bytes']
                
                if bytes_diff > 0:
                    # Calculate speed
                    speed = bytes_diff / time_diff
                    speed_text = self._format_bytes(speed) + "/s"
                    self.speed_label.text = f"Speed: {speed_text}"
                    
                    # Calculate time remaining
                    if speed > 0:
                        remaining_bytes = total_bytes - bytes_downloaded
                        remaining_seconds = remaining_bytes / speed
                        time_text = self._format_time(remaining_seconds)
                        self.time_label.text = f"Time remaining: {time_text}"
                
                # Update last values
                self._last_update['time'] = current_time
                self._last_update['bytes'] = bytes_downloaded
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to human readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def _on_cancel(self, widget):
        """Handle cancel button press."""
        self.cancelled = True
        self.operation_label.text = "Cancelling..."
        self.cancel_button.enabled = False
        logger.info("Progress cancelled by user")