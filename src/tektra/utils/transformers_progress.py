"""
Transformers Download Progress Monitor

Monitors the transformers/huggingface cache for download progress.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set

from loguru import logger


class TransformersProgressMonitor:
    """
    Monitors transformers model downloads by watching cache activity.
    """
    
    def __init__(self, model_name: str, progress_callback: Optional[Callable] = None):
        """
        Initialize the progress monitor.
        
        Args:
            model_name: The model being downloaded
            progress_callback: Async callback function(progress, status, bytes_downloaded, total_bytes)
        """
        self.model_name = model_name
        self.progress_callback = progress_callback
        self.cache_dir = self._get_cache_dir()
        self.monitoring = False
        self.tracked_files = {}
        self.file_progress = {}  # Track individual file progress
        self.total_size = 0
        self.downloaded_size = 0
        self.start_time = None
        self.last_update_time = None
        self.last_downloaded_size = 0
        
    def _get_cache_dir(self) -> Path:
        """Get the HuggingFace cache directory."""
        # Check environment variables
        cache_dir = os.environ.get('HF_HOME')
        if not cache_dir:
            cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE')
        if not cache_dir:
            cache_dir = os.environ.get('TRANSFORMERS_CACHE')
        if not cache_dir:
            # Default location
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        
        return Path(cache_dir)
    
    async def monitor_download(self, duration: float = 60):
        """
        Monitor download progress for a specified duration.
        
        Args:
            duration: Maximum time to monitor (seconds)
        """
        self.monitoring = True
        self.start_time = time.time()
        self.last_update_time = self.start_time
        
        # Known file patterns and sizes for Phi-3
        expected_files = {
            "model.safetensors": 2_400_000_000,  # ~2.4GB
            "pytorch_model.bin": 2_400_000_000,   # Alternative format
            "model-00001-of-00002.safetensors": 2_400_000_000,  # First shard
            "model-00002-of-00002.safetensors": 2_400_000_000,  # Second shard
            "tokenizer.json": 17_000_000,         # ~17MB
            "tokenizer_config.json": 10_000,      # ~10KB
            "config.json": 10_000,                # ~10KB
        }
        
        last_progress = -1
        no_activity_count = 0
        
        while self.monitoring and (time.time() - self.start_time) < duration:
            try:
                # Look for downloading files
                active_downloads = self._find_active_downloads()
                
                if active_downloads:
                    no_activity_count = 0
                    
                    # Calculate progress
                    total_size = 0
                    downloaded_size = 0
                    current_files = []
                    file_details = []
                    
                    # Estimate total size based on known files
                    for file_path, size in active_downloads.items():
                        filename = file_path.name
                        file_id = str(file_path)
                        
                        # Track individual file progress
                        if file_id not in self.file_progress:
                            self.file_progress[file_id] = {
                                'name': filename,
                                'start_size': size,
                                'last_size': size,
                                'estimated_total': 0
                            }
                        
                        # Estimate total size for this file
                        estimated_total = 0
                        for pattern, expected_size in expected_files.items():
                            if pattern in filename or filename.endswith('.incomplete'):
                                # For incomplete files, use the expected size
                                estimated_total = expected_size
                                break
                        
                        if estimated_total == 0:
                            # Estimate based on growth rate
                            estimated_total = max(size * 1.5, 100_000_000)  # At least 100MB
                        
                        self.file_progress[file_id]['estimated_total'] = estimated_total
                        total_size += estimated_total
                        downloaded_size += size
                        
                        # Calculate individual file progress
                        file_progress_pct = min(100, (size / estimated_total) * 100) if estimated_total > 0 else 0
                        
                        # Create detailed file entry
                        file_display = filename.replace('.incomplete', '')[:40] + '...' if len(filename) > 40 else filename.replace('.incomplete', '')
                        file_details.append(f"{file_display}: {self._format_bytes(size)}/{self._format_bytes(estimated_total)} ({file_progress_pct:.0f}%)")
                    
                    if total_size > 0:
                        overall_progress = min(0.95, downloaded_size / total_size)  # Cap at 95%
                        
                        # Calculate actual download speed
                        current_time = time.time()
                        time_diff = current_time - self.last_update_time
                        
                        if time_diff > 0.5:  # Update speed every 0.5 seconds
                            size_diff = downloaded_size - self.last_downloaded_size
                            speed = size_diff / time_diff if time_diff > 0 else 0
                            
                            self.last_update_time = current_time
                            self.last_downloaded_size = downloaded_size
                            
                            # Only update if progress changed
                            if abs(overall_progress - last_progress) > 0.005:  # 0.5% change
                                last_progress = overall_progress
                                
                                # Create detailed status message
                                status = f"Downloading {len(file_details)} files"
                                if len(file_details) > 0:
                                    status += ":\n" + "\n".join(file_details[:5])  # Show up to 5 files
                                
                                if self.progress_callback:
                                    await self.progress_callback(
                                        overall_progress,
                                        status,
                                        downloaded_size,
                                        total_size
                                    )
                else:
                    no_activity_count += 1
                    
                    # If no activity for a while, assume download complete
                    if no_activity_count > 20 and last_progress > 0:  # 10 seconds
                        if self.progress_callback:
                            await self.progress_callback(
                                1.0,
                                "Download complete!",
                                total_size,
                                total_size
                            )
                        break
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.debug(f"Monitor error (expected during download): {e}")
                await asyncio.sleep(1)
        
        self.monitoring = False
    
    def _find_active_downloads(self) -> Dict[Path, int]:
        """Find files being actively downloaded."""
        active = {}
        
        try:
            # Look for .incomplete files (new HF hub format)
            for path in self.cache_dir.rglob("*.incomplete"):
                if path.is_file():
                    size = path.stat().st_size
                    if size > 0:
                        active[path] = size
            
            # Look for tmp files
            for path in self.cache_dir.rglob("*.tmp*"):
                if path.is_file():
                    size = path.stat().st_size
                    if size > 0:
                        active[path] = size
            
            # Look for partial downloads
            for path in self.cache_dir.rglob("*.download"):
                if path.is_file():
                    size = path.stat().st_size
                    if size > 0:
                        active[path] = size
            
            # Check blobs directory for active downloads
            blobs_dir = self.cache_dir / "blobs"
            if blobs_dir.exists():
                for path in blobs_dir.iterdir():
                    if path.is_file():
                        # Check if file is growing
                        size = path.stat().st_size
                        if path in self.tracked_files:
                            old_size = self.tracked_files[path]
                            if size > old_size:
                                active[path] = size
                        else:
                            # New file
                            active[path] = size
                        
                        self.tracked_files[path] = size
                        
        except Exception as e:
            logger.debug(f"Error scanning downloads: {e}")
            
        return active
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"