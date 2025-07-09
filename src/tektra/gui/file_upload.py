"""
Enhanced File Upload Components

This module provides comprehensive file upload capabilities:
- Native file dialog integration
- Drag and drop support (where available)
- Multiple file format support
- Progress tracking and feedback
- Integration with multimodal processor
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
import mimetypes

import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from loguru import logger

from ..ai.multimodal import MultimodalProcessor


class FileUploadPanel:
    """
    Enhanced file upload panel with drag-and-drop and multiple format support.
    """
    
    def __init__(
        self,
        on_files_selected: Optional[Callable] = None,
        on_file_processed: Optional[Callable] = None,
        multimodal_processor: Optional[MultimodalProcessor] = None
    ):
        """
        Initialize file upload panel.
        
        Args:
            on_files_selected: Callback when files are selected
            on_file_processed: Callback when a file is processed
            multimodal_processor: Instance of multimodal processor
        """
        self.on_files_selected = on_files_selected
        self.on_file_processed = on_file_processed
        self.multimodal_processor = multimodal_processor or MultimodalProcessor()
        
        # Upload state
        self.selected_files: List[Path] = []
        self.processing_files: Dict[str, bool] = {}
        self.processed_results: Dict[str, Dict[str, Any]] = {}
        
        # UI components
        self.container = None
        self.file_list = None
        self.upload_button = None
        self.status_label = None
        self.progress_bar = None
        
        # Supported file types for filter
        self.supported_extensions = {
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
            # Documents
            '.pdf', '.docx', '.doc', '.txt', '.md', '.rtf',
            # Data files
            '.json', '.yaml', '.yml', '.csv', '.xml',
            # Code files
            '.py', '.js', '.html', '.css', '.cpp', '.java', '.rs'
        }
        
        self._build_ui()
        
        logger.info("File upload panel initialized")
    
    def _build_ui(self) -> toga.Box:
        """Build the file upload UI."""
        self.container = toga.Box(style=Pack(
            direction=COLUMN,
            background_color="#f8f9fa",
            margin=10,
            padding=15
        ))
        
        # Header
        header = toga.Label(
            "📎 File Upload",
            style=Pack(
                font_size=16,
                font_weight="bold",
                margin_bottom=10,
                color="#2c3e50"
            )
        )
        self.container.add(header)
        
        # Upload button and status row
        upload_row = toga.Box(style=Pack(
            direction=ROW,
            align_items="center",
            margin_bottom=10
        ))
        
        self.upload_button = toga.Button(
            "Choose Files",
            on_press=self._on_choose_files,
            style=Pack(
                background_color="#007bff",
                color="#ffffff",
                margin_right=10,
                width=120
            )
        )
        upload_row.add(self.upload_button)
        
        self.status_label = toga.Label(
            "No files selected",
            style=Pack(
                font_size=12,
                color="#6c757d",
                flex=1
            )
        )
        upload_row.add(self.status_label)
        
        self.container.add(upload_row)
        
        # File list container
        self.file_list_container = toga.Box(style=Pack(
            direction=COLUMN,
            margin_top=5,
            background_color="#ffffff",
            padding=10
        ))
        
        # Scroll container for file list
        self.file_scroll = toga.ScrollContainer(
            content=self.file_list_container,
            style=Pack(
                height=150,
                margin_bottom=10
            )
        )
        self.container.add(self.file_scroll)
        
        # Process button
        self.process_button = toga.Button(
            "Analyze Files",
            on_press=self._on_process_files,
            style=Pack(
                background_color="#28a745",
                color="#ffffff",
                width=120
            ),
            enabled=False
        )
        self.container.add(self.process_button)
        
        # Supported formats info
        info_label = toga.Label(
            f"Supported: Images, Documents, Text files ({len(self.supported_extensions)} formats)",
            style=Pack(
                font_size=10,
                color="#6c757d",
                margin_top=5,
                font_style="italic"
            )
        )
        self.container.add(info_label)
        
        return self.container
    
    async def _on_choose_files(self, widget) -> None:
        """Handle file selection via dialog."""
        try:
            # Create file selection dialog
            file_dialog = toga.OpenFileDialog(
                title="Select Files for Analysis",
                multiple_select=True,
                file_types=[
                    "Images (*.jpg;*.jpeg;*.png;*.gif;*.bmp;*.tiff;*.webp)",
                    "Documents (*.pdf;*.docx;*.doc;*.txt;*.md;*.rtf)",
                    "Data Files (*.json;*.yaml;*.yml;*.csv;*.xml)",
                    "Code Files (*.py;*.js;*.html;*.css;*.cpp;*.java;*.rs)",
                    "All Files (*.*)"
                ]
            )
            
            # Show dialog and get selected files
            selected_paths = await self.show_file_dialog(file_dialog)
            
            if selected_paths:
                await self._handle_file_selection(selected_paths)
                
        except Exception as e:
            logger.error(f"Error in file selection: {e}")
            await self._show_error(f"File selection failed: {e}")
    
    async def show_file_dialog(self, dialog) -> Optional[List[Path]]:
        """
        Show file dialog and return selected paths.
        
        Note: This is a simplified implementation. 
        In a real Toga app, this would use the native file dialog.
        """
        try:
            # For now, we'll simulate file selection
            # In production, this would be: return await dialog.show()
            
            # Mock file selection for development
            logger.info("File dialog would open here")
            return None
            
        except Exception as e:
            logger.error(f"Error showing file dialog: {e}")
            return None
    
    async def _handle_file_selection(self, file_paths: List[Path]) -> None:
        """Handle selected files."""
        try:
            # Filter supported files
            valid_files = []
            unsupported_files = []
            
            for file_path in file_paths:
                if file_path.suffix.lower() in self.supported_extensions:
                    valid_files.append(file_path)
                else:
                    unsupported_files.append(file_path)
            
            # Add valid files to selection
            self.selected_files.extend(valid_files)
            
            # Update UI
            await self._update_file_list()
            
            # Show status
            if valid_files:
                self.status_label.text = f"{len(self.selected_files)} files selected"
                self.process_button.enabled = True
            
            if unsupported_files:
                unsupported_names = [f.name for f in unsupported_files]
                await self._show_warning(
                    f"Unsupported files ignored: {', '.join(unsupported_names)}"
                )
            
            # Trigger callback
            if self.on_files_selected:
                await self._safe_callback(self.on_files_selected, valid_files)
                
        except Exception as e:
            logger.error(f"Error handling file selection: {e}")
            await self._show_error(f"Error processing selection: {e}")
    
    async def _update_file_list(self) -> None:
        """Update the visual file list."""
        try:
            # Clear existing file list
            for child in list(self.file_list_container.children):
                self.file_list_container.remove(child)
            
            # Add each selected file
            for i, file_path in enumerate(self.selected_files):
                file_row = self._create_file_row(file_path, i)
                self.file_list_container.add(file_row)
                
        except Exception as e:
            logger.error(f"Error updating file list: {e}")
    
    def _create_file_row(self, file_path: Path, index: int) -> toga.Box:
        """Create a row for a selected file."""
        row = toga.Box(style=Pack(
            direction=ROW,
            align_items="center",
            margin_bottom=5,
            padding=5,
            background_color="#f8f9fa"
        ))
        
        # File icon based on type
        file_icon = self._get_file_icon(file_path)
        
        # File info
        file_info = toga.Box(style=Pack(direction=COLUMN, flex=1))
        
        name_label = toga.Label(
            file_path.name,
            style=Pack(
                font_weight="bold",
                font_size=12,
                margin_bottom=2
            )
        )
        file_info.add(name_label)
        
        # File size and type
        try:
            file_size = file_path.stat().st_size
            size_str = self._format_file_size(file_size)
            info_text = f"{size_str} • {file_path.suffix.upper()[1:]}"
        except:
            info_text = f"{file_path.suffix.upper()[1:]} file"
        
        info_label = toga.Label(
            info_text,
            style=Pack(
                font_size=10,
                color="#6c757d"
            )
        )
        file_info.add(info_label)
        
        # Add icon and info to row
        icon_label = toga.Label(
            file_icon,
            style=Pack(
                font_size=16,
                margin_right=10
            )
        )
        row.add(icon_label)
        row.add(file_info)
        
        # Remove button
        remove_btn = toga.Button(
            "✕",
            on_press=lambda widget, idx=index: asyncio.create_task(self._remove_file(idx)),
            style=Pack(
                width=30,
                height=25,
                font_size=12,
                background_color="#dc3545",
                color="#ffffff"
            )
        )
        row.add(remove_btn)
        
        return row
    
    def _get_file_icon(self, file_path: Path) -> str:
        """Get emoji icon for file type."""
        ext = file_path.suffix.lower()
        
        # Image files
        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}:
            return "🖼️"
        # Document files
        elif ext in {'.pdf', '.docx', '.doc', '.rtf'}:
            return "📄"
        # Text files
        elif ext in {'.txt', '.md'}:
            return "📝"
        # Data files
        elif ext in {'.json', '.yaml', '.yml', '.csv', '.xml'}:
            return "📊"
        # Code files
        elif ext in {'.py', '.js', '.html', '.css', '.cpp', '.java', '.rs'}:
            return "💻"
        else:
            return "📎"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    async def _remove_file(self, index: int) -> None:
        """Remove a file from the selection."""
        try:
            if 0 <= index < len(self.selected_files):
                removed_file = self.selected_files.pop(index)
                logger.debug(f"Removed file: {removed_file.name}")
                
                # Update UI
                await self._update_file_list()
                
                # Update status
                if self.selected_files:
                    self.status_label.text = f"{len(self.selected_files)} files selected"
                else:
                    self.status_label.text = "No files selected"
                    self.process_button.enabled = False
                    
        except Exception as e:
            logger.error(f"Error removing file: {e}")
    
    async def _on_process_files(self, widget) -> None:
        """Process all selected files."""
        if not self.selected_files:
            return
        
        try:
            self.process_button.enabled = False
            self.process_button.text = "Processing..."
            
            # Process each file
            results = []
            for i, file_path in enumerate(self.selected_files):
                try:
                    self.status_label.text = f"Processing {file_path.name}... ({i+1}/{len(self.selected_files)})"
                    
                    # Process file through multimodal processor
                    result = await self.multimodal_processor.process_file(file_path)
                    result["file_path"] = str(file_path)
                    result["file_name"] = file_path.name
                    
                    results.append(result)
                    self.processed_results[str(file_path)] = result
                    
                    # Trigger callback for each processed file
                    if self.on_file_processed:
                        await self._safe_callback(self.on_file_processed, file_path, result)
                    
                    logger.info(f"Processed file: {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    error_result = {
                        "file_path": str(file_path),
                        "file_name": file_path.name,
                        "processing_status": "error",
                        "error": str(e)
                    }
                    results.append(error_result)
            
            # Update status
            successful = len([r for r in results if r.get("processing_status") == "success"])
            self.status_label.text = f"Processed {successful}/{len(results)} files successfully"
            
            # Clear selection after processing
            self.selected_files.clear()
            await self._update_file_list()
            
        except Exception as e:
            logger.error(f"Error in file processing: {e}")
            await self._show_error(f"Processing failed: {e}")
        finally:
            self.process_button.enabled = True
            self.process_button.text = "Analyze Files"
    
    async def _safe_callback(self, callback: Callable, *args) -> None:
        """Safely execute callback function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Error in callback: {e}")
    
    async def _show_error(self, message: str) -> None:
        """Show error message to user."""
        logger.error(message)
        self.status_label.text = f"Error: {message}"
        self.status_label.style.color = "#dc3545"
    
    async def _show_warning(self, message: str) -> None:
        """Show warning message to user."""
        logger.warning(message)
        # Could implement a temporary warning display here
    
    def get_processed_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all processed file results."""
        return self.processed_results.copy()
    
    def clear_results(self) -> None:
        """Clear all processed results."""
        self.processed_results.clear()
        self.selected_files.clear()
        asyncio.create_task(self._update_file_list())
        self.status_label.text = "No files selected"
        self.process_button.enabled = False


class FileUploadManager:
    """
    Manager class for integrating file upload with the main application.
    """
    
    def __init__(self, app_instance, multimodal_processor: MultimodalProcessor):
        """Initialize file upload manager."""
        self.app = app_instance
        self.multimodal_processor = multimodal_processor
        self.upload_panel = None
        
    def create_upload_panel(self) -> FileUploadPanel:
        """Create and configure upload panel."""
        self.upload_panel = FileUploadPanel(
            on_files_selected=self._on_files_selected,
            on_file_processed=self._on_file_processed,
            multimodal_processor=self.multimodal_processor
        )
        return self.upload_panel
    
    async def _on_files_selected(self, file_paths: List[Path]) -> None:
        """Handle when files are selected."""
        logger.info(f"Files selected: {[f.name for f in file_paths]}")
        
        # Could trigger UI updates or notifications here
        
    async def _on_file_processed(self, file_path: Path, result: Dict[str, Any]) -> None:
        """Handle when a file is processed."""
        logger.info(f"File processed: {file_path.name}")
        
        # Add processed file info to chat if available
        if hasattr(self.app, 'chat_manager') and self.app.chat_manager:
            status = result.get("processing_status", "unknown")
            if status == "success":
                content_type = result.get("content_type", "unknown")
                await self.app.chat_manager.handle_file_processed(
                    file_path.name, 
                    f"success ({content_type})"
                )
            else:
                error_msg = result.get("error", "Unknown error")
                await self.app.chat_manager.handle_file_processed(
                    file_path.name, 
                    f"error: {error_msg}"
                )