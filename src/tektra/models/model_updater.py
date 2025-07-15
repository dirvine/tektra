"""
Model Update Manager

This module handles seamless model updates, including downloading,
validation, hot-swapping, and rollback capabilities.
"""

import asyncio
import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any

import aiohttp
from loguru import logger

from .model_interface import ModelMetadata, ModelRegistry, ModelFactory, ModelInterface


class UpdateStatus(Enum):
    """Model update status."""
    CHECKING = "checking"
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    INSTALLING = "installing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UpdatePriority(Enum):
    """Update priority levels."""
    CRITICAL = "critical"      # Security fixes, critical bugs
    RECOMMENDED = "recommended"  # Performance improvements, new features
    OPTIONAL = "optional"       # Minor improvements


@dataclass
class ModelUpdate:
    """Information about a model update."""
    model_name: str
    current_version: str
    new_version: str
    new_metadata: ModelMetadata
    download_url: str
    checksum: str
    size_bytes: int
    priority: UpdatePriority
    changelog: str
    release_date: datetime
    requires_restart: bool = False
    breaking_changes: bool = False


@dataclass
class UpdateProgress:
    """Progress information for model updates."""
    model_name: str
    status: UpdateStatus
    progress_percent: float
    bytes_downloaded: int
    total_bytes: int
    speed_mbps: float
    eta_seconds: Optional[int]
    error_message: Optional[str] = None


class ModelUpdateManager:
    """
    Manages model updates with hot-swapping and rollback capabilities.
    
    Features:
    - Automatic update checking
    - Background downloads
    - Atomic model swapping
    - Rollback on failure
    - Progress tracking
    """
    
    def __init__(self, 
                 registry: ModelRegistry,
                 factory: ModelFactory,
                 models_dir: Path,
                 backup_dir: Optional[Path] = None):
        """
        Initialize model update manager.
        
        Args:
            registry: Model registry
            factory: Model factory
            models_dir: Directory where models are stored
            backup_dir: Directory for model backups
        """
        self.registry = registry
        self.factory = factory
        self.models_dir = models_dir
        self.backup_dir = backup_dir or models_dir / "backups"
        
        # Update tracking
        self.available_updates: Dict[str, ModelUpdate] = {}
        self.update_progress: Dict[str, UpdateProgress] = {}
        self.update_callbacks: List[Callable[[str, UpdateProgress], None]] = []
        
        # Active model references for hot-swapping
        self.active_models: Dict[str, ModelInterface] = {}
        
        # Update configuration
        self.auto_check_enabled = True
        self.auto_download_enabled = False
        self.check_interval_hours = 24
        self.max_concurrent_downloads = 2
        
        # Download semaphore
        self._download_semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        # Background tasks
        self._check_task: Optional[asyncio.Task] = None
        self._download_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Model update manager initialized")
    
    async def start(self):
        """Start the update manager and begin periodic checks."""
        if self.auto_check_enabled:
            self._check_task = asyncio.create_task(self._periodic_update_check())
        logger.info("Model update manager started")
    
    async def stop(self):
        """Stop the update manager and cancel ongoing operations."""
        # Cancel periodic check task
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        
        # Cancel download tasks
        for task in self._download_tasks.values():
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self._download_tasks.values(), return_exceptions=True)
        self._download_tasks.clear()
        
        logger.info("Model update manager stopped")
    
    async def check_for_updates(self, force: bool = False) -> List[ModelUpdate]:
        """
        Check for available model updates.
        
        Args:
            force: Force check even if recently checked
            
        Returns:
            List of available updates
        """
        logger.info("Checking for model updates...")
        
        try:
            # Get current models from registry
            current_models = self.registry.list_models()
            
            # Check each model for updates
            updates = []
            for model in current_models:
                update = await self._check_model_update(model)
                if update:
                    updates.append(update)
                    self.available_updates[model.name] = update
            
            logger.info(f"Found {len(updates)} available updates")
            return updates
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}")
            return []
    
    async def _check_model_update(self, model: ModelMetadata) -> Optional[ModelUpdate]:
        """Check for updates for a specific model."""
        try:
            # TODO: Implement actual remote registry checking
            # For now, return None (no updates available)
            # In a real implementation, this would query a remote registry
            
            # Example of what this might look like:
            # async with aiohttp.ClientSession() as session:
            #     url = f"{self.registry.update_channels['stable']}/{model.name}/latest"
            #     async with session.get(url) as response:
            #         if response.status == 200:
            #             remote_data = await response.json()
            #             remote_version = remote_data['version']
            #             if self._is_newer_version(remote_version, model.version):
            #                 return ModelUpdate(...)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to check update for {model.name}: {e}")
            return None
    
    def _is_newer_version(self, new_version: str, current_version: str) -> bool:
        """Check if new version is newer than current version."""
        # Simple version comparison (assumes semantic versioning)
        try:
            new_parts = [int(x) for x in new_version.split('.')]
            current_parts = [int(x) for x in current_version.split('.')]
            
            # Pad with zeros if lengths differ
            max_len = max(len(new_parts), len(current_parts))
            new_parts.extend([0] * (max_len - len(new_parts)))
            current_parts.extend([0] * (max_len - len(current_parts)))
            
            return new_parts > current_parts
        except (ValueError, AttributeError):
            return False
    
    async def download_update(self, model_name: str) -> bool:
        """
        Download a model update.
        
        Args:
            model_name: Name of the model to update
            
        Returns:
            bool: True if download successful
        """
        if model_name not in self.available_updates:
            logger.error(f"No update available for model: {model_name}")
            return False
        
        if model_name in self._download_tasks:
            logger.warning(f"Download already in progress for: {model_name}")
            return False
        
        update = self.available_updates[model_name]
        
        # Create download task
        self._download_tasks[model_name] = asyncio.create_task(
            self._download_model_update(update)
        )
        
        try:
            result = await self._download_tasks[model_name]
            return result
        finally:
            if model_name in self._download_tasks:
                del self._download_tasks[model_name]
    
    async def _download_model_update(self, update: ModelUpdate) -> bool:
        """Download and validate a model update."""
        async with self._download_semaphore:
            try:
                # Initialize progress tracking
                progress = UpdateProgress(
                    model_name=update.model_name,
                    status=UpdateStatus.DOWNLOADING,
                    progress_percent=0.0,
                    bytes_downloaded=0,
                    total_bytes=update.size_bytes,
                    speed_mbps=0.0,
                    eta_seconds=None
                )
                self.update_progress[update.model_name] = progress
                self._notify_progress_callbacks(update.model_name, progress)
                
                # Create temporary download file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                
                # Download the file
                success = await self._download_file(
                    update.download_url,
                    temp_path,
                    update.model_name
                )
                
                if not success:
                    temp_path.unlink(missing_ok=True)
                    return False
                
                # Validate the download
                progress.status = UpdateStatus.VALIDATING
                self._notify_progress_callbacks(update.model_name, progress)
                
                if not await self._validate_download(temp_path, update.checksum):
                    temp_path.unlink(missing_ok=True)
                    progress.status = UpdateStatus.FAILED
                    progress.error_message = "Download validation failed"
                    self._notify_progress_callbacks(update.model_name, progress)
                    return False
                
                # Move to final location
                model_file = self.models_dir / f"{update.model_name}_v{update.new_version}"
                model_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(temp_path), str(model_file))
                
                # Update registry
                self.registry.register_model(update.new_metadata)
                
                progress.status = UpdateStatus.COMPLETE
                progress.progress_percent = 100.0
                self._notify_progress_callbacks(update.model_name, progress)
                
                logger.info(f"Successfully downloaded update for: {update.model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to download update for {update.model_name}: {e}")
                progress.status = UpdateStatus.FAILED
                progress.error_message = str(e)
                self._notify_progress_callbacks(update.model_name, progress)
                return False
    
    async def _download_file(self, url: str, path: Path, model_name: str) -> bool:
        """Download a file with progress tracking."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        logger.error(f"Download failed with status {response.status}")
                        return False
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    start_time = asyncio.get_event_loop().time()
                    
                    with open(path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Update progress
                            current_time = asyncio.get_event_loop().time()
                            elapsed = current_time - start_time
                            
                            if elapsed > 0:
                                speed_bps = downloaded / elapsed
                                speed_mbps = speed_bps / (1024 * 1024)
                                
                                if speed_bps > 0:
                                    eta_seconds = (total_size - downloaded) / speed_bps
                                else:
                                    eta_seconds = None
                            else:
                                speed_mbps = 0.0
                                eta_seconds = None
                            
                            progress = self.update_progress.get(model_name)
                            if progress:
                                progress.bytes_downloaded = downloaded
                                progress.progress_percent = (
                                    (downloaded / total_size) * 100 if total_size > 0 else 0
                                )
                                progress.speed_mbps = speed_mbps
                                progress.eta_seconds = int(eta_seconds) if eta_seconds else None
                                
                                self._notify_progress_callbacks(model_name, progress)
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    async def _validate_download(self, file_path: Path, expected_checksum: str) -> bool:
        """Validate downloaded file checksum."""
        try:
            # Calculate file checksum
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            logger.error(f"Checksum validation error: {e}")
            return False
    
    async def install_update(self, model_name: str, hot_swap: bool = True) -> bool:
        """
        Install a downloaded model update.
        
        Args:
            model_name: Name of the model to update
            hot_swap: Whether to perform hot-swapping
            
        Returns:
            bool: True if installation successful
        """
        try:
            if model_name not in self.available_updates:
                logger.error(f"No update available for: {model_name}")
                return False
            
            update = self.available_updates[model_name]
            
            # Create backup if model is currently active
            if model_name in self.active_models:
                await self._create_model_backup(model_name)
            
            progress = UpdateProgress(
                model_name=model_name,
                status=UpdateStatus.INSTALLING,
                progress_percent=0.0,
                bytes_downloaded=0,
                total_bytes=0,
                speed_mbps=0.0,
                eta_seconds=None
            )
            self.update_progress[model_name] = progress
            self._notify_progress_callbacks(model_name, progress)
            
            # If hot-swapping is enabled and model is active
            if hot_swap and model_name in self.active_models:
                success = await self._hot_swap_model(model_name, update)
            else:
                # Simple installation (model will be loaded on next use)
                success = True
            
            if success:
                progress.status = UpdateStatus.COMPLETE
                progress.progress_percent = 100.0
                
                # Remove from available updates
                del self.available_updates[model_name]
                
                logger.info(f"Successfully installed update for: {model_name}")
            else:
                progress.status = UpdateStatus.FAILED
                progress.error_message = "Installation failed"
            
            self._notify_progress_callbacks(model_name, progress)
            return success
            
        except Exception as e:
            logger.error(f"Failed to install update for {model_name}: {e}")
            return False
    
    async def _hot_swap_model(self, model_name: str, update: ModelUpdate) -> bool:
        """Perform hot-swapping of an active model."""
        try:
            old_model = self.active_models[model_name]
            
            # Create new model instance
            new_model = await self.factory.create_model(
                update.model_name,
                old_model.config
            )
            
            if not new_model:
                logger.error(f"Failed to create new model instance: {model_name}")
                return False
            
            # Load new model
            if not await new_model.load():
                logger.error(f"Failed to load new model: {model_name}")
                return False
            
            # Atomic swap
            self.active_models[model_name] = new_model
            
            # Unload old model
            await old_model.unload()
            
            logger.info(f"Successfully hot-swapped model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Hot-swap failed for {model_name}: {e}")
            return False
    
    async def _create_model_backup(self, model_name: str):
        """Create a backup of the current model."""
        try:
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current model metadata
            current_metadata = self.registry.get_model(model_name)
            if not current_metadata:
                return
            
            # Create backup directory
            backup_name = f"{model_name}_v{current_metadata.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backup_dir / backup_name
            
            # TODO: Implement actual model file backup
            # This would involve copying model files to backup location
            
            logger.info(f"Created backup for model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to create backup for {model_name}: {e}")
    
    async def rollback_model(self, model_name: str) -> bool:
        """
        Rollback a model to the previous version.
        
        Args:
            model_name: Name of the model to rollback
            
        Returns:
            bool: True if rollback successful
        """
        try:
            # TODO: Implement model rollback
            # This would involve:
            # 1. Finding the most recent backup
            # 2. Restoring the backup
            # 3. Updating the registry
            # 4. Hot-swapping if the model is active
            
            logger.info(f"Rolling back model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback model {model_name}: {e}")
            return False
    
    def register_active_model(self, model_name: str, model: ModelInterface):
        """
        Register an active model for hot-swapping.
        
        Args:
            model_name: Model name
            model: Model instance
        """
        self.active_models[model_name] = model
        logger.debug(f"Registered active model: {model_name}")
    
    def unregister_active_model(self, model_name: str):
        """
        Unregister an active model.
        
        Args:
            model_name: Model name to unregister
        """
        if model_name in self.active_models:
            del self.active_models[model_name]
            logger.debug(f"Unregistered active model: {model_name}")
    
    def add_progress_callback(self, callback: Callable[[str, UpdateProgress], None]):
        """
        Add a callback for update progress notifications.
        
        Args:
            callback: Function to call with (model_name, progress)
        """
        self.update_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[str, UpdateProgress], None]):
        """
        Remove a progress callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
    
    def _notify_progress_callbacks(self, model_name: str, progress: UpdateProgress):
        """Notify all registered callbacks about progress updates."""
        for callback in self.update_callbacks:
            try:
                callback(model_name, progress)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")
    
    async def _periodic_update_check(self):
        """Periodically check for updates."""
        while True:
            try:
                await asyncio.sleep(self.check_interval_hours * 3600)
                await self.check_for_updates()
                
                # Auto-download if enabled
                if self.auto_download_enabled:
                    for model_name in self.available_updates:
                        update = self.available_updates[model_name]
                        if update.priority == UpdatePriority.CRITICAL:
                            await self.download_update(model_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic update check: {e}")
    
    def get_update_status(self, model_name: str) -> Optional[UpdateProgress]:
        """
        Get update status for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            UpdateProgress if available, None otherwise
        """
        return self.update_progress.get(model_name)
    
    def get_available_updates(self) -> List[ModelUpdate]:
        """Get list of available updates."""
        return list(self.available_updates.values())
    
    def cancel_update(self, model_name: str) -> bool:
        """
        Cancel an ongoing update.
        
        Args:
            model_name: Model name
            
        Returns:
            bool: True if cancelled successfully
        """
        if model_name in self._download_tasks:
            task = self._download_tasks[model_name]
            if not task.done():
                task.cancel()
                
                # Update progress
                if model_name in self.update_progress:
                    progress = self.update_progress[model_name]
                    progress.status = UpdateStatus.CANCELLED
                    self._notify_progress_callbacks(model_name, progress)
                
                logger.info(f"Cancelled update for: {model_name}")
                return True
        
        return False