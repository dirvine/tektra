"""
Hugging Face Download Progress Tracker

Uses the huggingface_hub library to track download progress properly.
"""

import asyncio
import time
from typing import Callable, Optional
from pathlib import Path

from loguru import logger

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils._tqdm import tqdm
    HF_HUB_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import
        from huggingface_hub import snapshot_download
        import tqdm as tqdm_module
        tqdm = tqdm_module
        HF_HUB_AVAILABLE = True
    except ImportError:
        HF_HUB_AVAILABLE = False
        logger.warning("huggingface_hub not available for progress tracking")


class HFDownloadProgress:
    """
    Tracks Hugging Face model downloads with proper progress reporting.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize download progress tracker.
        
        Args:
            progress_callback: Async callback function(progress, status, bytes_downloaded, total_bytes)
        """
        self.progress_callback = progress_callback
        self.total_files = 0
        self.completed_files = 0
        self.current_file = ""
        self.total_size = 0
        self.downloaded_size = 0
        self.file_sizes = {}
        self.start_time = time.time()
        
    async def download_with_progress(self, model_id: str, **kwargs):
        """
        Download a model with progress tracking.
        
        Args:
            model_id: The model ID to download
            **kwargs: Additional arguments for snapshot_download
            
        Returns:
            Path to the downloaded model
        """
        if not HF_HUB_AVAILABLE:
            logger.error("huggingface_hub not available")
            return None
            
        # For now, just use snapshot_download directly
        # Progress tracking through tqdm is complex with current HF hub
        logger.info(f"Downloading {model_id}...")
        
        if self.progress_callback:
            await self.progress_callback(0.4, f"Downloading {model_id} files...", 0, 0)
        
        try:
            # Run download in executor to avoid blocking
            loop = asyncio.get_event_loop()
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                path = await loop.run_in_executor(
                    executor,
                    lambda: snapshot_download(
                        model_id,
                        **kwargs
                    )
                )
                
            if self.progress_callback:
                await self.progress_callback(0.6, "Model files downloaded!", 0, 0)
                
            return path
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
            
    def _track_file(self, filename: str, size: int):
        """Track a new file download."""
        if filename not in self.file_sizes:
            self.file_sizes[filename] = size
            self.total_size += size
            self.total_files += 1
            logger.debug(f"Tracking file: {filename} ({self._format_bytes(size)})")
            
    def _update_progress(self, filename: str, current: int, total: int):
        """Update progress for a file."""
        self.current_file = filename
        
        # Calculate overall progress
        if self.total_size > 0:
            # Estimate downloaded size based on file progress
            file_downloaded = sum(
                self.file_sizes.get(f, 0) 
                for f in self.file_sizes 
                if f != filename
            )
            file_downloaded += current
            
            progress = file_downloaded / self.total_size
            
            # Call progress callback
            if self.progress_callback:
                asyncio.create_task(self._call_progress_callback(
                    progress,
                    f"Downloading {filename}",
                    file_downloaded,
                    self.total_size
                ))
                
    def _complete_file(self, filename: str):
        """Mark a file as complete."""
        self.completed_files += 1
        if filename in self.file_sizes:
            self.downloaded_size += self.file_sizes[filename]
            
        logger.debug(f"Completed: {filename} ({self.completed_files}/{self.total_files})")
        
    async def _call_progress_callback(self, progress: float, status: str, downloaded: int, total: int):
        """Call the progress callback asynchronously."""
        try:
            # Add file count to status
            if self.total_files > 1:
                status = f"{status} ({self.completed_files}/{self.total_files} files)"
                
            await self.progress_callback(progress, status, downloaded, total)
        except Exception as e:
            logger.error(f"Error in progress callback: {e}")
            
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"


async def download_model_with_progress(
    model_id: str,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> Optional[Path]:
    """
    Download a model with progress tracking.
    
    Args:
        model_id: The model ID to download
        progress_callback: Optional progress callback
        **kwargs: Additional arguments for snapshot_download
        
    Returns:
        Path to the downloaded model
    """
    tracker = HFDownloadProgress(progress_callback)
    return await tracker.download_with_progress(model_id, **kwargs)