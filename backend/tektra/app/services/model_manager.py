"""
Model Management Service for MLX and Hugging Face models.

This service handles:
- Model downloading and caching
- Model validation and integrity checks
- Storage management and cleanup
- Model metadata management
"""

import asyncio
import json
import logging
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a cached model."""
    name: str
    type: str  # "mlx" or "huggingface"
    size_mb: float
    download_date: str
    last_used: str
    version: str
    checksum: str
    path: str
    status: str  # "downloading", "ready", "error"


class ModelCache:
    """Manages local model cache."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".tektra" / "models"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, ModelMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load model metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self.metadata = {
                        name: ModelMetadata(**meta) 
                        for name, meta in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load model metadata: {e}")
                self.metadata = {}
    
    def _save_metadata(self):
        """Save model metadata to disk."""
        try:
            data = {name: asdict(meta) for name, meta in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the local path for a model."""
        if model_name in self.metadata:
            path = Path(self.metadata[model_name].path)
            if path.exists():
                return path
        return None
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is cached locally."""
        return self.get_model_path(model_name) is not None
    
    def add_model(self, model_name: str, model_type: str, local_path: Path, 
                  version: str = "unknown") -> bool:
        """Add a model to the cache."""
        try:
            # Calculate file size and checksum
            if local_path.is_dir():
                size_mb = sum(f.stat().st_size for f in local_path.rglob('*') if f.is_file()) / (1024 * 1024)
                checksum = self._calculate_dir_checksum(local_path)
            else:
                size_mb = local_path.stat().st_size / (1024 * 1024)
                checksum = self._calculate_file_checksum(local_path)
            
            metadata = ModelMetadata(
                name=model_name,
                type=model_type,
                size_mb=size_mb,
                download_date=datetime.now().isoformat(),
                last_used=datetime.now().isoformat(),
                version=version,
                checksum=checksum,
                path=str(local_path),
                status="ready"
            )
            
            self.metadata[model_name] = metadata
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model {model_name} to cache: {e}")
            return False
    
    def update_last_used(self, model_name: str):
        """Update the last used timestamp for a model."""
        if model_name in self.metadata:
            self.metadata[model_name].last_used = datetime.now().isoformat()
            self._save_metadata()
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the cache."""
        if model_name not in self.metadata:
            return False
        
        try:
            model_path = Path(self.metadata[model_name].path)
            if model_path.exists():
                if model_path.is_dir():
                    shutil.rmtree(model_path)
                else:
                    model_path.unlink()
            
            del self.metadata[model_name]
            self._save_metadata()
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove model {model_name}: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the model cache."""
        total_size = sum(meta.size_mb for meta in self.metadata.values())
        return {
            "cache_dir": str(self.cache_dir),
            "total_models": len(self.metadata),
            "total_size_mb": total_size,
            "models": {name: asdict(meta) for name, meta in self.metadata.items()}
        }
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _calculate_dir_checksum(self, dir_path: Path) -> str:
        """Calculate SHA256 checksum of a directory."""
        sha256_hash = hashlib.sha256()
        for file_path in sorted(dir_path.rglob('*')):
            if file_path.is_file():
                sha256_hash.update(str(file_path.relative_to(dir_path)).encode())
                with open(file_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


class ModelDownloader:
    """Downloads and manages AI models."""
    
    def __init__(self, cache: ModelCache):
        self.cache = cache
        self.download_progress: Dict[str, float] = {}
    
    async def download_mlx_model(self, model_name: str, hf_model_id: str) -> bool:
        """Download an MLX model from Hugging Face."""
        if not self._is_apple_silicon():
            logger.warning("MLX models only supported on Apple Silicon")
            return False
        
        try:
            from mlx_lm import download
            
            logger.info(f"Downloading MLX model {model_name} ({hf_model_id})")
            self.download_progress[model_name] = 0.0
            
            # Download to temporary location first
            temp_path = self.cache.cache_dir / f"{model_name}_temp"
            final_path = self.cache.cache_dir / model_name
            
            # Remove existing temp directory
            if temp_path.exists():
                shutil.rmtree(temp_path)
            
            # Download model
            await asyncio.to_thread(
                download, 
                hf_model_id, 
                str(temp_path)
            )
            
            # Move to final location
            if final_path.exists():
                shutil.rmtree(final_path)
            temp_path.rename(final_path)
            
            # Add to cache
            success = self.cache.add_model(model_name, "mlx", final_path)
            
            if success:
                self.download_progress[model_name] = 100.0
                logger.info(f"Successfully downloaded MLX model {model_name}")
            else:
                logger.error(f"Failed to register MLX model {model_name} in cache")
            
            return success
            
        except ImportError:
            logger.error("MLX not available - cannot download MLX models")
            return False
        except Exception as e:
            logger.error(f"Failed to download MLX model {model_name}: {e}")
            # Cleanup on failure
            temp_path = self.cache.cache_dir / f"{model_name}_temp"
            if temp_path.exists():
                shutil.rmtree(temp_path)
            return False
        finally:
            self.download_progress.pop(model_name, None)
    
    async def download_huggingface_model(self, model_name: str, hf_model_id: str) -> bool:
        """Download a Hugging Face model."""
        try:
            from huggingface_hub import snapshot_download
            
            logger.info(f"Downloading Hugging Face model {model_name} ({hf_model_id})")
            self.download_progress[model_name] = 0.0
            
            # Download to cache directory
            final_path = self.cache.cache_dir / model_name
            
            # Remove existing directory
            if final_path.exists():
                shutil.rmtree(final_path)
            
            # Download model
            await asyncio.to_thread(
                snapshot_download,
                repo_id=hf_model_id,
                local_dir=str(final_path),
                local_dir_use_symlinks=False
            )
            
            # Add to cache
            success = self.cache.add_model(model_name, "huggingface", final_path)
            
            if success:
                self.download_progress[model_name] = 100.0
                logger.info(f"Successfully downloaded Hugging Face model {model_name}")
            else:
                logger.error(f"Failed to register Hugging Face model {model_name} in cache")
            
            return success
            
        except ImportError:
            logger.error("huggingface_hub not available - cannot download HF models")
            return False
        except Exception as e:
            logger.error(f"Failed to download Hugging Face model {model_name}: {e}")
            # Cleanup on failure
            final_path = self.cache.cache_dir / model_name
            if final_path.exists():
                shutil.rmtree(final_path)
            return False
        finally:
            self.download_progress.pop(model_name, None)
    
    def get_download_progress(self, model_name: str) -> float:
        """Get download progress for a model (0-100)."""
        return self.download_progress.get(model_name, 0.0)
    
    def is_downloading(self, model_name: str) -> bool:
        """Check if a model is currently downloading."""
        return model_name in self.download_progress
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return platform.system() == "Darwin" and platform.processor() == "arm"


class ModelManager:
    """High-level model management interface."""
    
    def __init__(self):
        self.cache = ModelCache()
        self.downloader = ModelDownloader(self.cache)
        
        # Define available models with their Hugging Face IDs
        self.available_models = {
            # MLX Models (Apple Silicon only)
            "phi-3-mini": {
                "type": "mlx",
                "hf_id": "microsoft/Phi-3-mini-4k-instruct",
                "description": "Microsoft's efficient small language model optimized for Apple Silicon",
                "size_gb": 2.3,
                "parameters": "3.8B"
            },
            "llama-3.2-1b": {
                "type": "mlx", 
                "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
                "description": "Meta's compact Llama model for fast local inference",
                "size_gb": 1.2,
                "parameters": "1B"
            },
            "gemma-2b": {
                "type": "mlx",
                "hf_id": "google/gemma-2b-it",
                "description": "Google's Gemma model optimized for Apple Silicon",
                "size_gb": 1.6,
                "parameters": "2B"
            },
            "qwen-2.5-1.5b": {
                "type": "mlx",
                "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
                "description": "Alibaba's efficient Qwen model for Apple Silicon",
                "size_gb": 1.4,
                "parameters": "1.5B"
            },
            
            # Hugging Face Models (cross-platform)
            "llama-2-7b-chat": {
                "type": "huggingface",
                "hf_id": "meta-llama/Llama-2-7b-chat-hf", 
                "description": "Meta's Llama 2 chat model",
                "size_gb": 13.5,
                "parameters": "7B"
            },
            "mistral-7b-instruct": {
                "type": "huggingface",
                "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
                "description": "Mistral AI's instruction-tuned model",
                "size_gb": 14.2,
                "parameters": "7B"
            }
        }
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available locally, downloading if necessary."""
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        # Check if already cached
        if self.cache.is_model_cached(model_name):
            self.cache.update_last_used(model_name)
            return True
        
        # Download if not cached
        model_info = self.available_models[model_name]
        if model_info["type"] == "mlx":
            return await self.downloader.download_mlx_model(model_name, model_info["hf_id"])
        elif model_info["type"] == "huggingface":
            return await self.downloader.download_huggingface_model(model_name, model_info["hf_id"])
        else:
            logger.error(f"Unknown model type: {model_info['type']}")
            return False
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get the local path for a model."""
        return self.cache.get_model_path(model_name)
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models with their status."""
        models = {}
        for name, info in self.available_models.items():
            models[name] = {
                **info,
                "cached": self.cache.is_model_cached(name),
                "downloading": self.downloader.is_downloading(name),
                "download_progress": self.downloader.get_download_progress(name)
            }
        return models
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        return self.cache.get_cache_info()
    
    async def remove_model(self, model_name: str) -> bool:
        """Remove a model from the cache."""
        return self.cache.remove_model(model_name)
    
    def get_download_status(self, model_name: str) -> Dict[str, Any]:
        """Get download status for a model."""
        return {
            "model_name": model_name,
            "downloading": self.downloader.is_downloading(model_name),
            "progress": self.downloader.get_download_progress(model_name),
            "cached": self.cache.is_model_cached(model_name)
        }


# Global model manager instance
model_manager = ModelManager()