#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "aiohttp>=3.8.0",
#     "loguru>=0.7.0",
#     "tqdm>=4.65.0",
#     "asyncio-extensions>=0.1.0",
# ]
# ///
"""
Model Manager

Handles downloading, caching, and loading of AI models for embedded use.
"""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Callable
import aiohttp
from loguru import logger
import tqdm


class ModelManager:
    """Manages AI model files for the embedded application."""
    
    # Model registry with download URLs and metadata
    MODEL_REGISTRY = {
        "unmute_stt": {
            "url": "https://huggingface.co/kyutai/unmute-stt/resolve/main/model.pt",
            "size_mb": 512,
            "checksum": "sha256:placeholder_checksum_for_stt_model",
            "description": "Unmute Speech Recognition Model",
            "required_files": ["model.pt", "config.json"]
        },
        "unmute_llm": {
            "url": "https://huggingface.co/kyutai/unmute-llm/resolve/main/model.pt",
            "size_mb": 2048,
            "checksum": "sha256:placeholder_checksum_for_llm_model",
            "description": "Unmute Language Model",
            "required_files": ["model.pt", "config.json", "tokenizer.json"]
        },
        "unmute_tts": {
            "url": "https://huggingface.co/kyutai/unmute-tts/resolve/main/model.pt",
            "size_mb": 256,
            "checksum": "sha256:placeholder_checksum_for_tts_model",
            "description": "Unmute Voice Synthesis Model",
            "required_files": ["model.pt", "config.json", "vocoder.pt"]
        },
        "qwen_vl": {
            "url": "https://huggingface.co/Qwen/Qwen2.5-VL-7B/resolve/main/model.safetensors",
            "size_mb": 7168,
            "checksum": "sha256:placeholder_checksum_for_qwen_model",
            "description": "Qwen Vision-Language Model",
            "required_files": ["model.safetensors", "config.json", "tokenizer.json"]
        }
    }
    
    def __init__(self, app_data_dir: Path):
        """
        Initialize model manager.
        
        Args:
            app_data_dir: Application data directory
        """
        self.app_data_dir = Path(app_data_dir)
        self.model_dir = self.app_data_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model metadata cache
        self.metadata_file = self.model_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
        # Download progress tracking
        self.download_progress = {}
        
        logger.info(f"Model manager initialized with directory: {self.model_dir}")
        
    def _load_metadata(self) -> Dict:
        """Load model metadata from cache."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load metadata, starting fresh: {e}")
                return {}
        return {}
        
    def _save_metadata(self):
        """Save model metadata to cache."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")
            
    def get_model_path(self, model_name: str) -> Path:
        """Get local path for a model."""
        model_info = self.MODEL_REGISTRY.get(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Create model-specific directory
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        
        # Return path to main model file
        main_file = model_info["required_files"][0]
        return model_path / main_file
        
    def get_model_directory(self, model_name: str) -> Path:
        """Get the directory containing all model files."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_path = self.model_dir / model_name
        model_path.mkdir(exist_ok=True)
        return model_path
        
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available locally."""
        model_info = self.MODEL_REGISTRY.get(model_name)
        if not model_info:
            return False
            
        model_dir = self.get_model_directory(model_name)
        
        # Check if all required files exist
        for filename in model_info["required_files"]:
            file_path = model_dir / filename
            if not file_path.exists():
                return False
                
        # Verify checksum if available
        if model_name in self.metadata:
            stored_checksum = self.metadata[model_name].get("checksum")
            expected_checksum = model_info.get("checksum")
            
            if stored_checksum and expected_checksum:
                if stored_checksum != expected_checksum:
                    logger.warning(f"Model {model_name} checksum mismatch")
                    return False
                    
        return True
        
    async def ensure_models_available(
        self,
        models: Optional[list] = None,
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        Ensure required models are available, downloading if needed.
        
        Args:
            models: List of model names to ensure (None = all voice models)
            progress_callback: Callback for download progress
            
        Returns:
            bool: True if all models available
        """
        if models is None:
            models = ["unmute_stt", "unmute_llm", "unmute_tts"]
            
        success = True
        
        for i, model_name in enumerate(models):
            if not self.is_model_available(model_name):
                logger.info(f"Model {model_name} not available, downloading...")
                
                try:
                    await self.download_model(
                        model_name,
                        progress_callback=lambda p, s: progress_callback(
                            f"Downloading {model_name}",
                            (i + p) / len(models)
                        ) if progress_callback else None
                    )
                except Exception as e:
                    logger.error(f"Failed to download {model_name}: {e}")
                    success = False
                    
            else:
                logger.info(f"Model {model_name} already available")
                
                if progress_callback:
                    await progress_callback(
                        f"{model_name} ready",
                        (i + 1) / len(models)
                    )
                    
        return success
        
    async def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable] = None
    ):
        """
        Download a model from the registry.
        
        Args:
            model_name: Name of model to download
            progress_callback: Callback for progress updates
        """
        model_info = self.MODEL_REGISTRY.get(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_dir = self.get_model_directory(model_name)
        
        # Download main model file
        await self._download_file(
            model_info["url"],
            model_dir / model_info["required_files"][0],
            model_name,
            progress_callback
        )
        
        # Download additional required files (config, tokenizer, etc.)
        for filename in model_info["required_files"][1:]:
            # Construct URL for additional files
            base_url = model_info["url"].rsplit("/", 1)[0]
            file_url = f"{base_url}/{filename}"
            
            try:
                await self._download_file(
                    file_url,
                    model_dir / filename,
                    f"{model_name}_{filename}",
                    None  # No progress callback for auxiliary files
                )
            except Exception as e:
                logger.warning(f"Failed to download {filename} for {model_name}: {e}")
                # Create placeholder file
                (model_dir / filename).write_text(f"# Placeholder for {filename}\n")
                
        # Update metadata
        self.metadata[model_name] = {
            "downloaded_at": asyncio.get_event_loop().time(),
            "size": model_info["size_mb"] * 1024 * 1024,  # Convert to bytes
            "checksum": model_info["checksum"],
            "version": "1.0.0"
        }
        self._save_metadata()
        
        logger.success(f"Downloaded {model_name} successfully")
        
    async def _download_file(
        self,
        url: str,
        file_path: Path,
        model_name: str,
        progress_callback: Optional[Callable] = None
    ):
        """Download a single file with progress tracking."""
        temp_path = file_path.with_suffix(file_path.suffix + ".download")
        
        logger.info(f"Downloading {model_name} from {url}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get("content-length", 0))
                    
                    if total_size == 0:
                        logger.warning(f"Unknown file size for {model_name}")
                        
                    # Initialize progress tracking
                    self.download_progress[model_name] = {
                        "downloaded": 0,
                        "total": total_size,
                        "progress": 0.0
                    }
                    
                    # Download with progress tracking
                    with open(temp_path, "wb") as f:
                        downloaded = 0
                        
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            if total_size > 0:
                                progress = downloaded / total_size
                                self.download_progress[model_name]["downloaded"] = downloaded
                                self.download_progress[model_name]["progress"] = progress
                                
                                if progress_callback:
                                    await progress_callback(
                                        progress,
                                        f"{downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB"
                                    )
                                    
            except aiohttp.ClientError as e:
                logger.error(f"Network error downloading {model_name}: {e}")
                if temp_path.exists():
                    temp_path.unlink()
                raise
                
        # Verify download
        if temp_path.stat().st_size == 0:
            temp_path.unlink()
            raise ValueError(f"Downloaded file for {model_name} is empty")
            
        # Move to final location
        shutil.move(str(temp_path), str(file_path))
        
        # Clean up progress tracking
        if model_name in self.download_progress:
            del self.download_progress[model_name]
            
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
                
        return f"sha256:{sha256.hexdigest()}"
        
    def get_total_model_size(self, models: Optional[list] = None) -> int:
        """Get total size of models in MB."""
        if models is None:
            models = list(self.MODEL_REGISTRY.keys())
            
        total_size = 0
        for model_name in models:
            if model_name in self.MODEL_REGISTRY:
                total_size += self.MODEL_REGISTRY[model_name]["size_mb"]
                
        return total_size
        
    def get_available_models(self) -> list[str]:
        """Get list of available (downloaded) models."""
        available = []
        for model_name in self.MODEL_REGISTRY:
            if self.is_model_available(model_name):
                available.append(model_name)
        return available
        
    def get_missing_models(self, required_models: Optional[list] = None) -> list[str]:
        """Get list of missing models."""
        if required_models is None:
            required_models = ["unmute_stt", "unmute_llm", "unmute_tts"]
            
        missing = []
        for model_name in required_models:
            if not self.is_model_available(model_name):
                missing.append(model_name)
        return missing
        
    def get_download_progress(self, model_name: str) -> dict:
        """Get download progress for a model."""
        return self.download_progress.get(model_name, {
            "downloaded": 0,
            "total": 0,
            "progress": 0.0
        })
        
    def get_model_info(self, model_name: str) -> dict:
        """Get comprehensive model information."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
            
        registry_info = self.MODEL_REGISTRY[model_name]
        metadata = self.metadata.get(model_name, {})
        
        return {
            "name": model_name,
            "description": registry_info["description"],
            "size_mb": registry_info["size_mb"],
            "url": registry_info["url"],
            "required_files": registry_info["required_files"],
            "available": self.is_model_available(model_name),
            "model_path": str(self.get_model_path(model_name)),
            "model_directory": str(self.get_model_directory(model_name)),
            "downloaded_at": metadata.get("downloaded_at"),
            "version": metadata.get("version", "unknown"),
            "checksum": metadata.get("checksum"),
            "download_progress": self.get_download_progress(model_name)
        }
        
    def cleanup_models(self, keep_models: Optional[list] = None):
        """
        Clean up model files.
        
        Args:
            keep_models: List of models to keep (None = delete all)
        """
        if keep_models is None:
            keep_models = []
            
        for model_name in self.MODEL_REGISTRY:
            if model_name not in keep_models:
                model_dir = self.get_model_directory(model_name)
                
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                    logger.info(f"Deleted model directory for {model_name}")
                    
                if model_name in self.metadata:
                    del self.metadata[model_name]
                    
        self._save_metadata()
        logger.info("Model cleanup completed")
        
    def get_storage_info(self) -> dict:
        """Get storage information for the model directory."""
        import shutil
        
        total, used, free = shutil.disk_usage(self.model_dir)
        
        # Calculate model directory size
        model_size = 0
        for model_name in self.MODEL_REGISTRY:
            model_dir = self.get_model_directory(model_name)
            if model_dir.exists():
                model_size += sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )
                
        return {
            "model_directory": str(self.model_dir),
            "total_disk_space": total,
            "used_disk_space": used,
            "free_disk_space": free,
            "model_directory_size": model_size,
            "available_models": self.get_available_models(),
            "total_models": len(self.MODEL_REGISTRY)
        }
        
    async def verify_model_integrity(self, model_name: str) -> bool:
        """Verify model file integrity."""
        if not self.is_model_available(model_name):
            return False
            
        model_info = self.MODEL_REGISTRY[model_name]
        model_dir = self.get_model_directory(model_name)
        
        # Check all required files exist
        for filename in model_info["required_files"]:
            file_path = model_dir / filename
            if not file_path.exists():
                logger.error(f"Missing required file: {filename}")
                return False
                
        # Verify checksum of main model file
        main_file = model_dir / model_info["required_files"][0]
        if main_file.exists():
            calculated_checksum = self._calculate_checksum(main_file)
            expected_checksum = model_info.get("checksum")
            
            if expected_checksum and calculated_checksum != expected_checksum:
                logger.error(f"Checksum mismatch for {model_name}")
                return False
                
        logger.info(f"Model {model_name} integrity verified")
        return True
        
    async def update_model(self, model_name: str, progress_callback: Optional[Callable] = None):
        """
        Update a model to the latest version.
        
        Args:
            model_name: Name of model to update
            progress_callback: Callback for progress updates
        """
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
            
        logger.info(f"Updating model {model_name}...")
        
        # Remove existing model
        model_dir = self.get_model_directory(model_name)
        if model_dir.exists():
            shutil.rmtree(model_dir)
            
        # Download fresh copy
        await self.download_model(model_name, progress_callback)
        
        logger.success(f"Model {model_name} updated successfully")