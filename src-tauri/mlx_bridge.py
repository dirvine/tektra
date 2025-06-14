#!/usr/bin/env python3
"""
MLX Bridge for Tektra AI Assistant
Provides a bridge between Rust/Tauri backend and MLX models on Apple Silicon.
"""

import sys
import json
import argparse
from pathlib import Path
import asyncio
import logging
import os
from typing import Optional

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
    from huggingface_hub import snapshot_download, hf_hub_download, HfApi
    import requests
    from tqdm import tqdm
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLXBridge:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        # Use standard HuggingFace Hub cache directory
        self.cache_dir = self.get_hf_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_hf_cache_dir(self) -> Path:
        """Get the HuggingFace Hub cache directory"""
        # Use the same logic as HuggingFace Hub
        if "HF_HOME" in os.environ:
            return Path(os.environ["HF_HOME"])
        
        if "XDG_CACHE_HOME" in os.environ:
            return Path(os.environ["XDG_CACHE_HOME"]) / "huggingface"
        
        # Default location
        return Path.home() / ".cache" / "huggingface"
    
    def emit_progress(self, progress: float, status: str):
        """Emit progress information for the Rust backend to parse"""
        progress_data = {
            "progress": min(100, max(0, progress)),
            "status": status
        }
        print(f"PROGRESS:{json.dumps(progress_data)}", flush=True)
        
    def download_model(self, model_name: str, force: bool = False, with_progress: bool = False) -> dict:
        """Download an MLX model with progress tracking"""
        try:
            if not MLX_AVAILABLE:
                return {
                    "success": False,
                    "error": "MLX not available. Install with: pip install mlx-lm huggingface-hub"
                }
            
            if with_progress:
                self.emit_progress(5, "Checking model availability...")
            
            # Check if model exists in HF Hub
            api = HfApi()
            try:
                repo_info = api.repo_info(repo_id=model_name)
                if with_progress:
                    self.emit_progress(10, f"Found model: {model_name}")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Model not found in HuggingFace Hub: {model_name}"
                }
            
            # Check if already cached using HF Hub's native caching
            try:
                # This will use the standard HF cache directory
                if with_progress:
                    self.emit_progress(20, "Checking cache...")
                
                # Try to download without forcing to check if cached
                if not force:
                    cached_path = snapshot_download(
                        repo_id=model_name,
                        cache_dir=str(self.cache_dir),
                        local_files_only=True  # Only check cache
                    )
                    if with_progress:
                        self.emit_progress(100, "Model loaded from cache")
                    return {
                        "success": True,
                        "message": "Model already cached",
                        "path": cached_path
                    }
            except Exception:
                # Model not in cache, proceed to download
                pass
            
            if with_progress:
                self.emit_progress(25, f"Starting download: {model_name}")
                
            logger.info(f"Downloading model: {model_name}")
            
            # Download with progress tracking
            class ProgressCallback:
                def __init__(self, bridge):
                    self.bridge = bridge
                    self.last_progress = 25
                
                def __call__(self, chunk_size: int, total_size: int, downloaded: int):
                    if total_size > 0:
                        progress = 25 + (downloaded / total_size) * 70  # 25-95% for download
                        if progress - self.last_progress >= 5:  # Update every 5%
                            self.bridge.emit_progress(progress, f"Downloading... {downloaded}/{total_size} bytes")
                            self.last_progress = progress
            
            progress_callback = ProgressCallback(self) if with_progress else None
            
            # Download model using huggingface_hub with standard caching
            downloaded_path = snapshot_download(
                repo_id=model_name,
                cache_dir=str(self.cache_dir),
                resume_download=True,
                force_download=force
            )
            
            if with_progress:
                self.emit_progress(95, "Download complete, verifying...")
            
            return {
                "success": True,
                "message": "Model downloaded successfully",
                "path": downloaded_path
            }
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_model(self, model_name: str, auto_download: bool = True) -> dict:
        """Load an MLX model"""
        try:
            if not MLX_AVAILABLE:
                return {
                    "success": False,
                    "error": "MLX not available. Install with: pip install mlx-lm"
                }
            
            # Check if model is cached
            model_cache_path = self.cache_dir / model_name.replace("/", "_")
            if not model_cache_path.exists() and auto_download:
                logger.info(f"Model not cached, downloading: {model_name}")
                download_result = self.download_model(model_name)
                if not download_result["success"]:
                    return download_result
            
            logger.info(f"Loading model: {model_name}")
            
            # Try to load from cache first, then from hub
            try:
                if model_cache_path.exists():
                    self.model, self.tokenizer = load(str(model_cache_path))
                else:
                    self.model, self.tokenizer = load(model_name)
            except Exception as load_error:
                # Fallback to loading from hub
                logger.warning(f"Failed to load from cache, trying from hub: {load_error}")
                self.model, self.tokenizer = load(model_name)
            
            self.model_name = model_name
            
            return {
                "success": True,
                "model": model_name,
                "device": "Apple Silicon (MLX)",
                "cached": model_cache_path.exists()
            }
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> dict:
        """Generate a response using the loaded model"""
        try:
            if not self.model or not self.tokenizer:
                return {
                    "success": False,
                    "error": "No model loaded"
                }
            
            logger.info(f"Generating response for prompt: {prompt[:50]}...")
            
            # Generate response using MLX
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False
            )
            
            return {
                "success": True,
                "response": response,
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_cached_models(self) -> dict:
        """List all cached models"""
        try:
            cached_models = []
            if self.cache_dir.exists():
                for model_dir in self.cache_dir.iterdir():
                    if model_dir.is_dir():
                        # Convert directory name back to model name
                        model_name = model_dir.name.replace("_", "/")
                        size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                        cached_models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "size_bytes": size,
                            "size_gb": round(size / (1024**3), 2)
                        })
            
            return {
                "success": True,
                "models": cached_models,
                "cache_dir": str(self.cache_dir)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_status(self) -> dict:
        """Get the current status of the bridge"""
        status = {
            "mlx_available": MLX_AVAILABLE,
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "device": "Apple Silicon (MLX)" if MLX_AVAILABLE else "CPU (MLX not available)",
            "cache_dir": str(self.cache_dir)
        }
        
        # Add cache info
        cached_models = self.list_cached_models()
        if cached_models["success"]:
            status["cached_models_count"] = len(cached_models["models"])
            status["cached_models"] = cached_models["models"]
        
        return status


def main():
    parser = argparse.ArgumentParser(description="MLX Bridge for Tektra")
    parser.add_argument("--command", required=True, 
                       choices=["load", "generate", "status", "download", "list"])
    parser.add_argument("--model", help="Model name for loading/downloading")
    parser.add_argument("--prompt", help="Prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--force", action="store_true", help="Force download even if cached")
    parser.add_argument("--auto-download", action="store_true", default=True,
                       help="Auto-download model if not cached")
    parser.add_argument("--progress", action="store_true", help="Enable progress reporting for downloads")
    
    args = parser.parse_args()
    
    bridge = MLXBridge()
    
    if args.command == "load":
        if not args.model:
            result = {"success": False, "error": "Model name required for load command"}
        else:
            result = bridge.load_model(args.model, auto_download=args.auto_download)
    
    elif args.command == "download":
        if not args.model:
            result = {"success": False, "error": "Model name required for download command"}
        else:
            result = bridge.download_model(args.model, force=args.force, with_progress=args.progress)
    
    elif args.command == "generate":
        if not args.prompt:
            result = {"success": False, "error": "Prompt required for generate command"}
        else:
            result = bridge.generate_response(
                args.prompt, 
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
    
    elif args.command == "list":
        result = bridge.list_cached_models()
    
    elif args.command == "status":
        result = bridge.get_status()
    
    # Output result as JSON
    print(json.dumps(result))


if __name__ == "__main__":
    main()