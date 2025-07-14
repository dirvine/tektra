"""
Hugging Face Progress Hook

Directly hooks into the transformers/huggingface_hub download system
to provide real-time progress updates.
"""

import asyncio
import functools
import time
from typing import Callable, Optional
import threading

from loguru import logger


class HFProgressHook:
    """
    Hooks into HF downloads to provide progress updates.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.downloads = {}
        self.total_size = 0
        self.downloaded_size = 0
        self.last_update = time.time()
        self._lock = threading.Lock()
        
    def install_hooks(self):
        """Install progress hooks into transformers/HF hub."""
        try:
            # Try to hook into the tqdm progress bars used by HF
            import sys
            from unittest.mock import patch
            
            # Store original stderr write
            original_write = sys.stderr.write
            
            def progress_write_hook(text):
                """Intercept progress bar updates."""
                # Call original
                result = original_write(text)
                
                # Parse progress info if it looks like a progress bar
                if '%|' in text and '/' in text:
                    try:
                        # Extract progress info
                        self._parse_progress(text)
                    except:
                        pass
                        
                return result
            
            # Patch stderr write
            sys.stderr.write = progress_write_hook
            
            # Also try to patch file download functions
            try:
                from transformers.utils import hub
                
                # Store original
                if hasattr(hub, 'http_file_size'):
                    original_file_size = hub.http_file_size
                    
                    def file_size_hook(url, *args, **kwargs):
                        """Track file sizes."""
                        size = original_file_size(url, *args, **kwargs)
                        if size and size > 0:
                            with self._lock:
                                self.total_size += size
                        return size
                    
                    hub.http_file_size = file_size_hook
                    
            except Exception as e:
                logger.debug(f"Could not hook file size: {e}")
                
            # Try to hook into requests progress
            try:
                import requests
                from requests.models import Response
                
                original_iter_content = Response.iter_content
                
                def iter_content_hook(self_response, chunk_size=1, decode_unicode=False):
                    """Track download progress."""
                    # Get URL from response
                    url = self_response.url
                    filename = url.split('/')[-1][:20] + '...'
                    
                    # Track this download
                    total = int(self_response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    # Iterate original content
                    for chunk in original_iter_content(self_response, chunk_size, decode_unicode):
                        if chunk:
                            downloaded += len(chunk)
                            
                            # Update progress
                            with self._lock:
                                if url not in self.downloads:
                                    self.downloads[url] = {
                                        'filename': filename,
                                        'total': total,
                                        'downloaded': 0
                                    }
                                
                                self.downloads[url]['downloaded'] = downloaded
                                
                                # Trigger callback
                                self._trigger_callback()
                                
                        yield chunk
                
                Response.iter_content = iter_content_hook
                
            except Exception as e:
                logger.debug(f"Could not hook requests: {e}")
                
            logger.info("Progress hooks installed")
            
        except Exception as e:
            logger.error(f"Failed to install progress hooks: {e}")
            
    def _parse_progress(self, text):
        """Parse progress from tqdm output."""
        try:
            # Look for patterns like "1.5GB/2.4GB"
            import re
            
            # Pattern for size progress
            size_match = re.search(r'(\d+\.?\d*)(GB|MB|KB|B)/(\d+\.?\d*)(GB|MB|KB|B)', text)
            if size_match:
                downloaded_val = float(size_match.group(1))
                downloaded_unit = size_match.group(2)
                total_val = float(size_match.group(3))
                total_unit = size_match.group(4)
                
                # Convert to bytes
                units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
                downloaded_bytes = downloaded_val * units.get(downloaded_unit, 1)
                total_bytes = total_val * units.get(total_unit, 1)
                
                with self._lock:
                    self.downloaded_size = downloaded_bytes
                    self.total_size = max(self.total_size, total_bytes)
                    
                self._trigger_callback()
                
            # Also look for percentage
            pct_match = re.search(r'(\d+)%\|', text)
            if pct_match:
                progress = int(pct_match.group(1)) / 100.0
                
                with self._lock:
                    if self.total_size > 0:
                        self.downloaded_size = int(self.total_size * progress)
                        
                self._trigger_callback()
                
        except Exception as e:
            pass
            
    def _trigger_callback(self):
        """Trigger progress callback if enough time has passed."""
        current_time = time.time()
        
        if current_time - self.last_update > 0.5:  # Update every 0.5 seconds
            self.last_update = current_time
            
            if self.progress_callback and self.total_size > 0:
                progress = self.downloaded_size / self.total_size
                
                # Create status message
                status = f"Downloading model files"
                
                # Add file details if available
                if self.downloads:
                    file_list = []
                    for url, info in list(self.downloads.items())[:3]:
                        filename = info['filename']
                        downloaded = info['downloaded']
                        total = info['total']
                        if total > 0:
                            pct = (downloaded / total) * 100
                            file_list.append(f"{filename}: {pct:.0f}%")
                    
                    if file_list:
                        status += ":\n" + "\n".join(file_list)
                
                # Run callback in asyncio
                try:
                    loop = asyncio.get_event_loop()
                    loop.create_task(
                        self.progress_callback(
                            progress,
                            status,
                            self.downloaded_size,
                            self.total_size
                        )
                    )
                except:
                    # Try without asyncio
                    asyncio.create_task(
                        self.progress_callback(
                            progress,
                            status,
                            self.downloaded_size,
                            self.total_size
                        )
                    )