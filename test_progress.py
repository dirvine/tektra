#!/usr/bin/env python3
"""Test the improved progress tracking."""

import asyncio
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from tektra.utils.model_download_monitor import ModelDownloadMonitor
from tektra.ai.simple_llm import SimpleLLM


async def test_progress_callback(progress: float, status: str, bytes_downloaded: int = 0, total_bytes: int = 0):
    """Test callback to print progress."""
    if total_bytes > 0:
        mb_downloaded = bytes_downloaded / (1024 * 1024)
        mb_total = total_bytes / (1024 * 1024)
        print(f"\rProgress: {progress*100:.1f}% - {status} ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end="", flush=True)
    else:
        print(f"\rProgress: {progress*100:.1f}% - {status}", end="", flush=True)


async def main():
    """Test model download with progress tracking."""
    print("Testing Tektra progress tracking...")
    print("This will download the Phi-3 model (~2.4GB)")
    print()
    
    # Create LLM instance
    llm = SimpleLLM()
    
    # Initialize with progress tracking
    print("Starting model initialization...")
    success = await llm.initialize(test_progress_callback)
    
    print()
    if success:
        print("✅ Model loaded successfully!")
        
        # Test generation
        print("\nTesting model generation...")
        response = await llm.generate_response("Hello! How are you?", max_tokens=50)
        print(f"Response: {response}")
    else:
        print("❌ Failed to load model")
    
    # Cleanup
    await llm.cleanup()


if __name__ == "__main__":
    asyncio.run(main())