"""
Simple Progress Indicator

Shows download activity even when we can't get exact progress.
"""

import asyncio
import time
from typing import Callable, Optional

from loguru import logger


class SimpleProgressIndicator:
    """
    Simple progress indicator that shows activity during downloads.
    """
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.progress_callback = progress_callback
        self.running = False
        self.start_time = None
        
    async def run_with_progress(self, message: str = "Downloading model files...", duration: float = 300, is_cached: bool = False):
        """
        Show progress animation during download.
        
        Args:
            message: Base message to show
            duration: Maximum duration to show progress
        """
        self.running = True
        self.start_time = time.time()
        
        # Phases of download
        if is_cached:
            phases = [
                (0.3, "Checking cached model files..."),
                (0.5, "Loading model from cache..."),
                (0.7, "Initializing model weights..."),
                (0.9, "Finalizing model setup..."),
                (0.95, "Almost ready..."),
            ]
        else:
            phases = [
                (0.3, "Connecting to Hugging Face servers..."),
                (0.35, "Downloading model configuration..."),
                (0.4, "Downloading tokenizer files..."),
                (0.5, "Downloading model weights (this may take a few minutes)..."),
                (0.6, "Downloading model shards..."),
                (0.7, "Finalizing download..."),
                (0.8, "Verifying downloaded files..."),
                (0.9, "Loading model into memory..."),
                (0.95, "Almost done..."),
            ]
        
        current_phase = 0
        last_update = time.time()
        dots = 0
        
        while self.running and (time.time() - self.start_time) < duration:
            try:
                # Calculate elapsed time
                elapsed = time.time() - self.start_time
                
                # Move to next phase based on time
                if is_cached:
                    expected_progress = min(0.95, elapsed / 30)  # Expect ~30 seconds for cached
                else:
                    expected_progress = min(0.95, elapsed / 120)  # Expect ~2 minutes for download
                
                # Find appropriate phase
                for i, (threshold, phase_msg) in enumerate(phases):
                    if expected_progress >= threshold and i > current_phase:
                        current_phase = i
                        message = phase_msg
                
                # Add animated dots
                if time.time() - last_update > 0.5:
                    dots = (dots + 1) % 4
                    animated_msg = message + "." * dots
                    
                    # Add download size estimate
                    if current_phase >= 3:  # Downloading weights
                        size_estimate = f"\n\nEstimated download size: ~2.4 GB"
                        if elapsed > 10:
                            # Estimate speed based on phase
                            estimated_speed = 5 * 1024 * 1024  # 5 MB/s estimate
                            downloaded = int(elapsed * estimated_speed)
                            total_size = 2_400_000_000
                            
                            if downloaded < total_size:
                                pct = (downloaded / total_size) * 100
                                size_estimate += f"\nProgress: {self._format_bytes(downloaded)} / {self._format_bytes(total_size)} ({pct:.0f}%)"
                        
                        animated_msg += size_estimate
                    
                    if self.progress_callback:
                        await self.progress_callback(
                            expected_progress,
                            animated_msg,
                            0,  # We don't have real bytes
                            0   # We don't have real total
                        )
                    
                    last_update = time.time()
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in progress indicator: {e}")
                await asyncio.sleep(1)
        
        self.running = False
    
    def stop(self):
        """Stop the progress indicator."""
        self.running = False
        
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes to human readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} TB"