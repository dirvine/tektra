"""
Download Progress Tracking for Model Downloads

Provides hooks into Hugging Face model downloads to track progress with
file sizes and download speeds.
"""

import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional
from urllib.parse import urlparse

from loguru import logger


class DownloadProgressTracker:
    """
    Tracks download progress for Hugging Face model downloads.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize download progress tracker.
        
        Args:
            progress_callback: Async callback function(progress, status, bytes_downloaded, total_bytes)
        """
        self.progress_callback = progress_callback
        self.downloads = {}  # Track multiple file downloads
        self.total_size = 0
        self.downloaded_size = 0
        self.start_time = time.time()
        
    def track_download(self, url: str, filename: str, size: int):
        """Track a new download."""
        self.downloads[filename] = {
            'url': url,
            'size': size,
            'downloaded': 0,
            'start_time': time.time()
        }
        self.total_size += size
        logger.debug(f"Tracking download: {filename} ({size} bytes)")
        
    def update_progress(self, filename: str, downloaded: int):
        """Update download progress for a file."""
        if filename in self.downloads:
            old_downloaded = self.downloads[filename]['downloaded']
            self.downloads[filename]['downloaded'] = downloaded
            self.downloaded_size += (downloaded - old_downloaded)
            
            # Calculate overall progress
            if self.total_size > 0:
                progress = (self.downloaded_size / self.total_size) * 100
                
                # Call progress callback if provided
                if self.progress_callback:
                    import asyncio
                    try:
                        # Get or create event loop
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        
                        # Create task for async callback
                        loop.create_task(
                            self.progress_callback(
                                progress,
                                f"Downloading {filename}",
                                self.downloaded_size,
                                self.total_size
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error calling progress callback: {e}")
    
    def complete_download(self, filename: str):
        """Mark a download as complete."""
        if filename in self.downloads:
            # Ensure we count full size as downloaded
            remaining = self.downloads[filename]['size'] - self.downloads[filename]['downloaded']
            if remaining > 0:
                self.downloaded_size += remaining
                self.downloads[filename]['downloaded'] = self.downloads[filename]['size']
            
            logger.debug(f"Download complete: {filename}")
    
    def get_download_stats(self) -> Dict:
        """Get current download statistics."""
        elapsed = time.time() - self.start_time
        speed = self.downloaded_size / elapsed if elapsed > 0 else 0
        
        return {
            'total_files': len(self.downloads),
            'total_size': self.total_size,
            'downloaded_size': self.downloaded_size,
            'progress': (self.downloaded_size / self.total_size * 100) if self.total_size > 0 else 0,
            'speed': speed,
            'elapsed': elapsed,
            'remaining': (self.total_size - self.downloaded_size) / speed if speed > 0 else 0
        }


def patch_transformers_download(progress_callback: Optional[Callable] = None):
    """
    Monkey patch transformers download functions to track progress.
    
    This is a bit hacky but necessary since transformers doesn't provide
    proper download progress callbacks.
    """
    try:
        from transformers.utils import hub
        from transformers.file_utils import cached_path, http_get
        
        tracker = DownloadProgressTracker(progress_callback)
        
        # Store original functions
        original_http_get = http_get if hasattr(hub, 'http_get') else None
        original_cached_download = hub.cached_download if hasattr(hub, 'cached_download') else None
        
        # Patch http_get if it exists
        if original_http_get:
            def patched_http_get(url, temp_file, proxies=None, resume_size=0, headers=None):
                """Patched version that tracks download progress."""
                # Extract filename from URL
                filename = os.path.basename(urlparse(url).path) or "model_file"
                
                # Try to get file size from headers
                import requests
                try:
                    response = requests.head(url, headers=headers, proxies=proxies, allow_redirects=True)
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size > 0:
                        tracker.track_download(url, filename, total_size)
                except:
                    pass
                
                # Call original function with progress tracking
                def progress_wrapper(count, block_size, total_size):
                    if total_size > 0:
                        downloaded = count * block_size
                        tracker.update_progress(filename, min(downloaded, total_size))
                
                # Call original
                result = original_http_get(url, temp_file, proxies, resume_size, headers)
                
                # Mark complete
                tracker.complete_download(filename)
                
                return result
            
            # Apply patch
            hub.http_get = patched_http_get
        
        # Return tracker for access to stats
        return tracker
        
    except Exception as e:
        logger.warning(f"Could not patch transformers download tracking: {e}")
        return None


def restore_transformers_download():
    """Restore original transformers download functions."""
    try:
        from transformers.utils import hub
        
        # Restore if we have the originals stored
        if hasattr(hub, '_original_http_get'):
            hub.http_get = hub._original_http_get
            
        if hasattr(hub, '_original_cached_download'):
            hub.cached_download = hub._original_cached_download
            
    except Exception as e:
        logger.warning(f"Could not restore transformers download functions: {e}")