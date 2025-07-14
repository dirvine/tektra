"""
Progress Dialog Component for Tektra AI Assistant

Provides a modern, non-blocking progress dialog for long-running operations
like model downloads and initialization.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import toga
from loguru import logger
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class ProgressDialog:
    """
    Modern progress dialog for showing operation progress.
    
    Features:
    - Non-blocking modal dialog
    - Progress bar with percentage
    - Current operation description
    - Download speed and time remaining
    - Cancel button for long operations
    """
    
    def __init__(self, app: toga.App, title: str = "Loading...", cancellable: bool = True):
        """
        Initialize progress dialog.
        
        Args:
            app: The main Toga application
            title: Dialog title
            cancellable: Whether to show cancel button
        """
        self.app = app
        self.title = title
        self.cancellable = cancellable
        self.cancelled = False
        self.start_time = datetime.now()
        self.bytes_downloaded = 0
        self.total_bytes = 0
        
        # Create dialog window
        self.window = toga.Window(title=title)
        # self.window.closeable = False  # Not supported in current Toga version
        self.window.size = (600, 350)
        
        # Build dialog content
        self._build_dialog()
        
    def _build_dialog(self):
        """Build the dialog UI."""
        # Main container
        main_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                padding=20,
                background_color="#ffffff"
            )
        )
        
        # Title
        self.title_label = toga.Label(
            self.title,
            style=Pack(
                font_size=18,
                font_weight="bold",
                color="#2c3e50",
                margin_bottom=20,
                text_align="center"
            )
        )
        main_box.add(self.title_label)
        
        # Current operation label
        self.operation_label = toga.Label(
            "Initializing...",
            style=Pack(
                font_size=14,
                color="#7f8c8d",
                margin_bottom=10
            )
        )
        main_box.add(self.operation_label)
        
        # Detailed status label (for multi-line status)
        self.detail_label = toga.Label(
            "",
            style=Pack(
                font_size=12,
                color="#95a5a6",
                margin_bottom=10
            )
        )
        main_box.add(self.detail_label)
        
        # Progress container
        progress_container = toga.Box(
            style=Pack(
                direction=COLUMN,
                margin_bottom=20,
                width=550
            )
        )
        
        # Progress bar background
        self.progress_bg = toga.Box(
            style=Pack(
                height=30,
                background_color="#ecf0f1",
                margin_bottom=10
            )
        )
        
        # Progress bar fill (will be updated)
        self.progress_fill = toga.Box(
            style=Pack(
                height=30,
                background_color="#3498db",
                width=0
            )
        )
        self.progress_bg.add(self.progress_fill)
        progress_container.add(self.progress_bg)
        
        # Progress text
        self.progress_label = toga.Label(
            "0%",
            style=Pack(
                font_size=12,
                color="#7f8c8d",
                text_align="center"
            )
        )
        progress_container.add(self.progress_label)
        
        main_box.add(progress_container)
        
        # Details section
        details_box = toga.Box(
            style=Pack(
                direction=COLUMN,
                margin_bottom=20
            )
        )
        
        # Speed label
        self.speed_label = toga.Label(
            "",
            style=Pack(
                font_size=11,
                color="#95a5a6",
                margin_bottom=5,
                text_align="center"
            )
        )
        details_box.add(self.speed_label)
        
        # Time remaining label
        self.time_label = toga.Label(
            "",
            style=Pack(
                font_size=11,
                color="#95a5a6",
                text_align="center"
            )
        )
        details_box.add(self.time_label)
        
        main_box.add(details_box)
        
        # Button container
        button_box = toga.Box(
            style=Pack(
                direction=ROW,
                margin_top=20
            )
        )
        
        # Spacer
        button_box.add(toga.Box(style=Pack(flex=1)))
        
        if self.cancellable:
            # Cancel button
            self.cancel_button = toga.Button(
                "Cancel",
                on_press=self._on_cancel,
                style=Pack(
                    width=100,
                    background_color="#e74c3c",
                    color="#ffffff",
                    padding=10
                )
            )
            button_box.add(self.cancel_button)
        
        # Spacer
        button_box.add(toga.Box(style=Pack(flex=1)))
        
        main_box.add(button_box)
        
        # Set window content
        self.window.content = main_box
        
    def show(self):
        """Show the progress dialog."""
        self.window.show()
        self.app.current_window = self.window
        
    def close(self):
        """Close the progress dialog."""
        try:
            self.window.close()
            # Restore focus to main window
            if hasattr(self.app, 'main_window'):
                self.app.current_window = self.app.main_window
        except Exception as e:
            logger.error(f"Error closing progress dialog: {e}")
    
    def update_progress(
        self,
        progress: float,
        operation: str = "",
        bytes_downloaded: int = 0,
        total_bytes: int = 0
    ):
        """
        Update progress dialog.
        
        Args:
            progress: Progress percentage (0-100)
            operation: Current operation description
            bytes_downloaded: Bytes downloaded so far
            total_bytes: Total bytes to download
        """
        if self.cancelled:
            return
            
        # Update progress bar
        progress = max(0, min(100, progress))
        self.progress_fill.style.width = int(550 * progress / 100)
        self.progress_label.text = f"{int(progress)}%"
        
        # Update operation
        if operation:
            # Check if operation has multiple lines
            lines = operation.split('\n', 1)
            if len(lines) > 1:
                self.operation_label.text = lines[0]
                self.detail_label.text = lines[1]
            else:
                self.operation_label.text = operation
                self.detail_label.text = ""
        
        # Update download stats
        if total_bytes > 0 and bytes_downloaded > 0:
            # Track download progress
            if not hasattr(self, '_last_update'):
                self._last_update = {
                    'time': self.start_time,
                    'bytes': 0
                }
            
            # Calculate speed based on recent progress
            current_time = datetime.now()
            time_diff = (current_time - self._last_update['time']).total_seconds()
            
            if time_diff > 0.5:  # Update speed every 0.5 seconds
                bytes_diff = bytes_downloaded - self._last_update['bytes']
                
                if bytes_diff > 0:
                    # Calculate speed based on recent progress
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
            
            self.bytes_downloaded = bytes_downloaded
            self.total_bytes = total_bytes
        
        # Update file size info
        if total_bytes > 0 and bytes_downloaded > 0:
            downloaded_text = self._format_bytes(bytes_downloaded)
            total_text = self._format_bytes(total_bytes)
            # Don't duplicate the size info if it's already in the operation text
            if downloaded_text not in operation and total_text not in operation:
                self.operation_label.text = f"{operation} ({downloaded_text} / {total_text})"
    
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
        

class ProgressTracker:
    """
    Helper class to track progress across multiple operations.
    """
    
    def __init__(self, dialog: ProgressDialog, total_steps: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            dialog: Progress dialog to update
            total_steps: Total number of steps
        """
        self.dialog = dialog
        self.total_steps = total_steps
        self.current_step = 0
        self.step_weights = {}
        
    def add_step(self, name: str, weight: float = 1.0):
        """Add a step with optional weight."""
        self.step_weights[name] = weight
        
    def update_step(self, step_name: str, progress: float, operation: str = ""):
        """Update progress for a specific step."""
        if self.dialog.cancelled:
            return False
            
        # Calculate overall progress
        total_weight = sum(self.step_weights.values()) or self.total_steps
        step_weight = self.step_weights.get(step_name, 1.0)
        step_progress = (step_weight / total_weight) * progress
        
        # Update dialog
        self.dialog.update_progress(
            progress=self.current_step + step_progress,
            operation=operation or step_name
        )
        
        return not self.dialog.cancelled
    
    def complete_step(self, step_name: str):
        """Mark a step as complete."""
        if step_name in self.step_weights:
            weight = self.step_weights[step_name]
            total_weight = sum(self.step_weights.values()) or self.total_steps
            self.current_step += (weight / total_weight) * 100
            
            self.dialog.update_progress(
                progress=self.current_step,
                operation=f"{step_name} complete"
            )