"""
Model Download Monitor

Monitors the Hugging Face cache directory for active downloads and provides
real-time progress updates.
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger


class ModelDownloadMonitor:
    """
    Monitors model downloads by watching the Hugging Face cache directory.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the download monitor.
        
        Args:
            progress_callback: Async callback function(progress, status, bytes_downloaded, total_bytes)
        """
        self.progress_callback = progress_callback
        self.cache_dir = self._get_cache_dir()
        self.active_downloads = {}
        self.total_size = 0
        self.downloaded_size = 0
        self.monitoring = False
        
    def _get_cache_dir(self) -> Path:
        """Get the Hugging Face cache directory."""
        # Check environment variable first
        cache_dir = os.environ.get('HF_HOME')
        if not cache_dir:
            cache_dir = os.environ.get('HUGGINGFACE_HUB_CACHE')
        if not cache_dir:
            # Default location
            cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
        
        return Path(cache_dir)
    
    async def start_monitoring(self, model_name: str):
        """Start monitoring downloads for a specific model."""
        self.monitoring = True
        self.model_name = model_name
        
        logger.info(f"Starting download monitor for {model_name}")
        logger.info(f"Monitoring cache directory: {self.cache_dir}")
        
        # Run monitoring in background
        asyncio.create_task(self._monitor_loop())
        
    async def stop_monitoring(self):
        """Stop monitoring downloads."""
        self.monitoring = False
        
    async def _monitor_loop(self):
        """Main monitoring loop."""
        last_update = 0
        no_activity_count = 0
        
        while self.monitoring:
            try:
                # Scan for temporary download files
                temp_files = self._find_temp_files()
                
                if temp_files:
                    no_activity_count = 0
                    # Calculate total progress
                    total_size = 0
                    downloaded_size = 0
                    current_files = []
                    
                    for temp_file, (size, expected_size) in temp_files.items():
                        total_size += expected_size
                        downloaded_size += size
                        
                        # Track individual file
                        filename = temp_file.name.replace('.tmp', '').replace('.download', '')
                        current_files.append(f"{filename} ({self._format_bytes(size)}/{self._format_bytes(expected_size)})")
                    
                    # Update progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        
                        # Create detailed status
                        if len(current_files) == 1:
                            status = f"Downloading {current_files[0]}"
                        else:
                            status = f"Downloading {len(current_files)} files..."
                            if len(current_files) <= 3:
                                status += "\n" + "\n".join(current_files)
                        
                        # Only update if progress changed
                        current_update = int(progress * 10)  # Update every 0.1%
                        if current_update != last_update:
                            last_update = current_update
                            
                            if self.progress_callback:
                                await self.progress_callback(
                                    progress / 100,  # Convert to 0-1 range
                                    status,
                                    downloaded_size,
                                    total_size
                                )
                else:
                    no_activity_count += 1
                    
                    # If no activity for a while, check if download completed
                    if no_activity_count > 10:  # 5 seconds of no activity
                        if self.progress_callback and last_update > 0:
                            await self.progress_callback(
                                1.0,
                                "Model download complete!",
                                total_size,
                                total_size
                            )
                        self.monitoring = False
                        break
                
                # Check every 500ms
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in download monitor: {e}")
                await asyncio.sleep(1)
    
    def _find_temp_files(self) -> Dict[Path, Tuple[int, int]]:
        """Find temporary download files in the cache directory."""
        temp_files = {}
        
        try:
            # Look for .tmp and .download files
            for path in self.cache_dir.rglob("*.tmp"):
                if path.is_file():
                    size = path.stat().st_size
                    
                    # Try to find expected size from metadata
                    expected_size = self._get_expected_size(path)
                    if expected_size > 0:
                        temp_files[path] = (size, expected_size)
            
            # Also look for incomplete downloads
            for path in self.cache_dir.rglob("*.download"):
                if path.is_file():
                    size = path.stat().st_size
                    expected_size = self._get_expected_size(path)
                    if expected_size > 0:
                        temp_files[path] = (size, expected_size)
                        
            # Look for blobs being downloaded
            blobs_dir = self.cache_dir / "blobs"
            if blobs_dir.exists():
                for path in blobs_dir.iterdir():
                    if path.is_file() and path.suffix == "" and len(path.name) == 64:
                        # This might be a blob being downloaded
                        size = path.stat().st_size
                        # For blobs, we estimate size based on model
                        if "phi" in self.model_name.lower():
                            expected_size = 2_500_000_000  # ~2.5GB for Phi-3
                        else:
                            expected_size = size * 2  # Rough estimate
                        
                        # Only track if file is growing
                        if path in self.active_downloads:
                            old_size = self.active_downloads[path]
                            if size > old_size:
                                temp_files[path] = (size, expected_size)
                        else:
                            temp_files[path] = (size, expected_size)
                        
                        self.active_downloads[path] = size
                        
        except Exception as e:
            logger.debug(f"Error scanning for temp files: {e}")
            
        return temp_files
    
    def _get_expected_size(self, temp_file: Path) -> int:
        """Try to get expected file size from metadata."""
        # Check for .json metadata file
        meta_file = temp_file.with_suffix('.json')
        if meta_file.exists():
            try:
                import json
                with open(meta_file) as f:
                    meta = json.load(f)
                    return meta.get('size', 0)
            except:
                pass
        
        # Check for size in filename
        if 'size' in temp_file.name:
            try:
                # Extract size from filename pattern
                parts = temp_file.name.split('_')
                for part in parts:
                    if part.isdigit():
                        return int(part)
            except:
                pass
        
        # Default estimates based on file patterns
        name_lower = temp_file.name.lower()
        if 'model' in name_lower or 'pytorch' in name_lower:
            return 2_500_000_000  # 2.5GB estimate
        elif 'tokenizer' in name_lower:
            return 1_000_000  # 1MB estimate
        elif 'config' in name_lower:
            return 10_000  # 10KB estimate
        
        return 0
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"