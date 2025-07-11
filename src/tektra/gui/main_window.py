"""
Main Window Component

This module provides the main window management and layout utilities
for the Tektra AI Assistant desktop application.
"""

from collections.abc import Callable
from typing import Any

import toga
from loguru import logger


class MainWindow:
    """
    Main window manager for Tektra AI Assistant.

    Provides window management, layout utilities, and common window operations
    for the desktop application.
    """

    def __init__(self, app: toga.App, title: str = "Tektra AI Assistant"):
        """
        Initialize main window.

        Args:
            app: Toga application instance
            title: Window title
        """
        self.app = app
        self.title = title
        self.window = None
        self._dialog_callbacks = {}

    def create_window(self, content: toga.Widget, **kwargs) -> toga.MainWindow:
        """
        Create and configure the main window.

        Args:
            content: Main content widget
            **kwargs: Additional window configuration options

        Returns:
            Configured main window
        """
        # Default window configuration
        window_config = {
            "title": self.title,
            "size": (1200, 800),
            "resizable": True,
            **kwargs,
        }

        # Create main window
        self.window = toga.MainWindow(**window_config)
        self.window.content = content
        self.window.app = self.app

        logger.info(f"Created main window: {self.title}")
        return self.window

    def set_title(self, title: str) -> None:
        """Update window title."""
        self.title = title
        if self.window:
            self.window.title = title

    def show_info_dialog(
        self, title: str, message: str, callback: Callable | None = None
    ) -> None:
        """
        Show an information dialog.

        Args:
            title: Dialog title
            message: Dialog message
            callback: Optional callback when dialog is closed
        """
        if self.window:
            try:
                self.window.info_dialog(title, message)
                if callback:
                    callback()
            except Exception as e:
                logger.error(f"Error showing info dialog: {e}")

    def show_error_dialog(
        self, title: str, message: str, callback: Callable | None = None
    ) -> None:
        """
        Show an error dialog.

        Args:
            title: Dialog title
            message: Error message
            callback: Optional callback when dialog is closed
        """
        if self.window:
            try:
                self.window.error_dialog(title, message)
                if callback:
                    callback()
            except Exception as e:
                logger.error(f"Error showing error dialog: {e}")

    def show_question_dialog(
        self, title: str, message: str, on_result: Callable | None = None
    ) -> None:
        """
        Show a question dialog with Yes/No options.

        Args:
            title: Dialog title
            message: Question message
            on_result: Callback with boolean result (True=Yes, False=No)
        """
        if self.window:
            try:
                result = self.window.question_dialog(title, message)
                if on_result:
                    on_result(result)
            except Exception as e:
                logger.error(f"Error showing question dialog: {e}")
                if on_result:
                    on_result(False)

    async def open_file_dialog(
        self,
        title: str = "Select File",
        file_types: list | None = None,
        multiple_select: bool = False,
    ) -> str | None:
        """
        Open file selection dialog.

        Args:
            title: Dialog title
            file_types: List of allowed file extensions
            multiple_select: Allow multiple file selection

        Returns:
            Selected file path(s) or None
        """
        if not self.window:
            return None

        try:
            if multiple_select:
                return await self.window.open_file_dialog(
                    title=title, file_types=file_types, multiselect=True
                )
            else:
                return await self.window.open_file_dialog(
                    title=title, file_types=file_types
                )
        except Exception as e:
            logger.error(f"Error opening file dialog: {e}")
            return None

    async def save_file_dialog(
        self,
        title: str = "Save File",
        suggested_filename: str | None = None,
        file_types: list | None = None,
    ) -> str | None:
        """
        Open file save dialog.

        Args:
            title: Dialog title
            suggested_filename: Default filename
            file_types: List of allowed file extensions

        Returns:
            Selected save path or None
        """
        if not self.window:
            return None

        try:
            return await self.window.save_file_dialog(
                title=title,
                suggested_filename=suggested_filename,
                file_types=file_types,
            )
        except Exception as e:
            logger.error(f"Error opening save dialog: {e}")
            return None

    def set_window_state(self, state: str) -> None:
        """
        Set window state (minimized, maximized, normal).

        Args:
            state: Window state ('minimize', 'maximize', 'normal')
        """
        if not self.window:
            return

        try:
            if state == "minimize":
                # Note: Toga may not support all window states on all platforms
                logger.debug("Window minimize requested - may not be supported on all platforms")
                # Toga doesn't have built-in minimize support, this is a platform limitation
            elif state == "maximize":
                # Platform-specific maximization
                logger.debug("Window maximize requested - using fullscreen as fallback")
                # Attempt to maximize by setting full screen (Toga limitation workaround)
                if hasattr(self.window, 'full_screen'):
                    self.window.full_screen = True
            elif state == "normal":
                # Restore to normal size
                logger.debug("Window normal state requested")
                # Restore from full screen if available
                if hasattr(self.window, 'full_screen'):
                    self.window.full_screen = False
                # Reset to default size
                if hasattr(self.window, 'size'):
                    self.window.size = (1200, 800)  # Default size
        except Exception as e:
            logger.debug(f"Window state change not supported: {e}")

    def get_window_info(self) -> dict[str, Any]:
        """
        Get current window information.

        Returns:
            Dict containing window properties
        """
        if not self.window:
            return {"created": False}

        return {
            "created": True,
            "title": self.window.title,
            "app": self.app.formal_name if self.app else None,
            "has_content": self.window.content is not None,
        }

    def close_window(self) -> None:
        """Close the main window."""
        if self.window:
            try:
                self.window.close()
                logger.info("Main window closed")
            except Exception as e:
                logger.error(f"Error closing window: {e}")


class WindowManager:
    """
    Application-wide window management utilities.

    Provides centralized window management, dialog handling,
    and cross-window communication.
    """

    def __init__(self):
        """Initialize window manager."""
        self.windows = {}
        self.active_dialogs = {}
        self.window_count = 0

    def register_window(self, window_id: str, window: MainWindow) -> None:
        """Register a window with the manager."""
        self.windows[window_id] = window
        self.window_count += 1
        logger.debug(f"Registered window: {window_id}")

    def unregister_window(self, window_id: str) -> None:
        """Unregister a window from the manager."""
        if window_id in self.windows:
            del self.windows[window_id]
            self.window_count -= 1
            logger.debug(f"Unregistered window: {window_id}")

    def get_window(self, window_id: str) -> MainWindow | None:
        """Get a registered window by ID."""
        return self.windows.get(window_id)

    def broadcast_message(self, message: str, data: Any = None) -> None:
        """
        Broadcast a message to all registered windows.

        Args:
            message: Message type/identifier
            data: Message data
        """
        logger.debug(f"Broadcasting message: {message}")
        for window_id, _window in self.windows.items():
            try:
                # Send message to window if it has a message handler
                if hasattr(_window, 'handle_broadcast_message'):
                    _window.handle_broadcast_message(message, data)
                else:
                    # Log that window doesn't support message handling
                    logger.debug(f"Window {window_id} doesn't support broadcast messages")
            except Exception as e:
                logger.error(f"Error broadcasting to {window_id}: {e}")

    def get_manager_stats(self) -> dict[str, Any]:
        """Get window manager statistics."""
        return {
            "window_count": self.window_count,
            "active_windows": list(self.windows.keys()),
            "active_dialogs": len(self.active_dialogs),
        }


# Global window manager instance
window_manager = WindowManager()
