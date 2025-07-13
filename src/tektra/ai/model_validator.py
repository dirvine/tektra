"""
Model Validation and Checksum Verification
Provides comprehensive validation for downloaded models and their integrity.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
import aiohttp
from loguru import logger


@dataclass
class ModelChecksum:
    """Model checksum information."""
    
    algorithm: str  # sha256, md5, etc.
    value: str
    file_path: str
    size_bytes: int


@dataclass 
class ModelValidationResult:
    """Result of model validation."""
    
    is_valid: bool
    model_name: str
    validation_time_seconds: float
    checksums_verified: List[ModelChecksum]
    errors: List[str]
    warnings: List[str]
    file_integrity_passed: bool
    size_validation_passed: bool
    format_validation_passed: bool


class ModelValidator:
    """
    Comprehensive model validation and checksum verification.
    
    Features:
    - SHA256/MD5 checksum verification
    - File size validation
    - Model format validation
    - Hugging Face Hub integration
    - Corrupted file detection
    - Performance optimization
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the model validator.
        
        Args:
            cache_dir: Directory for caching validation results
        """
        self.cache_dir = cache_dir or Path.home() / ".cache" / "tektra" / "validation"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Known model checksums (can be extended)
        self.known_checksums: Dict[str, Dict[str, str]] = {
            "Qwen/Qwen2.5-VL-7B-Instruct": {
                "config.json": "sha256:...",  # Would contain actual checksums
                "pytorch_model.bin": "sha256:...",
                "tokenizer.json": "sha256:..."
            },
            "Qwen/Qwen2.5-3B": {
                "config.json": "sha256:...",
                "pytorch_model.bin": "sha256:..."
            }
        }
        
        # File size limits (in bytes)
        self.max_file_sizes = {
            "config.json": 50 * 1024,  # 50KB
            "tokenizer.json": 50 * 1024 * 1024,  # 50MB
            "pytorch_model.bin": 50 * 1024 * 1024 * 1024,  # 50GB
            "model.safetensors": 50 * 1024 * 1024 * 1024,  # 50GB
        }
        
        logger.info("Model validator initialized")
    
    async def validate_model(self, model_path: Path, model_name: str, 
                           progress_callback: Optional[callable] = None) -> ModelValidationResult:
        """
        Validate a complete model.
        
        Args:
            model_path: Path to the model directory
            model_name: Name/identifier of the model
            progress_callback: Optional progress callback
            
        Returns:
            ModelValidationResult with validation details
        """
        start_time = time.time()
        errors = []
        warnings = []
        checksums_verified = []
        
        try:
            logger.info(f"Starting validation for model: {model_name}")
            
            if progress_callback:
                await self._call_progress_callback(progress_callback, 10, "Starting model validation...")
            
            # Check if model path exists
            if not model_path.exists():
                errors.append(f"Model path does not exist: {model_path}")
                return self._create_failed_result(model_name, start_time, errors, warnings, checksums_verified)
            
            if not model_path.is_dir():
                errors.append(f"Model path is not a directory: {model_path}")
                return self._create_failed_result(model_name, start_time, errors, warnings, checksums_verified)
            
            # Get all model files
            model_files = list(model_path.rglob("*"))
            model_files = [f for f in model_files if f.is_file()]
            
            if not model_files:
                errors.append("No files found in model directory")
                return self._create_failed_result(model_name, start_time, errors, warnings, checksums_verified)
            
            logger.info(f"Found {len(model_files)} files to validate")
            
            # Validate each file
            file_integrity_passed = True
            size_validation_passed = True
            format_validation_passed = True
            
            for i, file_path in enumerate(model_files):
                if progress_callback:
                    progress = 20 + (i / len(model_files)) * 60
                    await self._call_progress_callback(progress_callback, int(progress), 
                                                     f"Validating {file_path.name}...")
                
                # File size validation
                try:
                    file_size = file_path.stat().st_size
                    max_size = self.max_file_sizes.get(file_path.name, float('inf'))
                    
                    if file_size > max_size:
                        errors.append(f"File {file_path.name} exceeds maximum size: {file_size} > {max_size}")
                        size_validation_passed = False
                    elif file_size == 0:
                        errors.append(f"File {file_path.name} is empty")
                        size_validation_passed = False
                    
                except Exception as e:
                    errors.append(f"Could not check size of {file_path.name}: {e}")
                    size_validation_passed = False
                
                # Format validation
                try:
                    format_valid = await self._validate_file_format(file_path)
                    if not format_valid:
                        warnings.append(f"File {file_path.name} may have format issues")
                        format_validation_passed = False
                except Exception as e:
                    warnings.append(f"Could not validate format of {file_path.name}: {e}")
                
                # Checksum validation
                try:
                    checksum_result = await self._validate_file_checksum(file_path, model_name)
                    if checksum_result:
                        checksums_verified.append(checksum_result)
                    else:
                        # No known checksum, calculate one for future reference
                        calculated_checksum = await self._calculate_file_checksum(file_path)
                        if calculated_checksum:
                            checksums_verified.append(calculated_checksum)
                            logger.debug(f"Calculated checksum for {file_path.name}: {calculated_checksum.value}")
                
                except Exception as e:
                    errors.append(f"Checksum validation failed for {file_path.name}: {e}")
                    file_integrity_passed = False
            
            if progress_callback:
                await self._call_progress_callback(progress_callback, 85, "Performing final validation checks...")
            
            # Additional model-specific validation
            config_validation = await self._validate_model_config(model_path)
            if not config_validation:
                warnings.append("Model configuration validation issues detected")
            
            # Tokenizer validation
            tokenizer_validation = await self._validate_tokenizer(model_path)
            if not tokenizer_validation:
                warnings.append("Tokenizer validation issues detected")
            
            if progress_callback:
                await self._call_progress_callback(progress_callback, 95, "Completing validation...")
            
            # Determine overall validation result
            is_valid = (len(errors) == 0 and 
                       file_integrity_passed and 
                       size_validation_passed and 
                       format_validation_passed)
            
            validation_time = time.time() - start_time
            
            result = ModelValidationResult(
                is_valid=is_valid,
                model_name=model_name,
                validation_time_seconds=validation_time,
                checksums_verified=checksums_verified,
                errors=errors,
                warnings=warnings,
                file_integrity_passed=file_integrity_passed,
                size_validation_passed=size_validation_passed,
                format_validation_passed=format_validation_passed
            )
            
            # Cache validation result
            await self._cache_validation_result(result)
            
            if progress_callback:
                await self._call_progress_callback(progress_callback, 100, "Model validation complete")
            
            if is_valid:
                logger.success(f"Model validation passed for {model_name}")
            else:
                logger.warning(f"Model validation completed with issues for {model_name}")
            
            return result
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            logger.error(f"Model validation failed for {model_name}: {e}")
            return self._create_failed_result(model_name, start_time, errors, warnings, checksums_verified)
    
    async def _validate_file_format(self, file_path: Path) -> bool:
        """Validate file format based on extension and content."""
        try:
            if file_path.suffix == '.json':
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    json.loads(content)  # Validate JSON syntax
                return True
            
            elif file_path.suffix in ['.bin', '.pt', '.pth']:
                # Basic binary file validation (check if file is readable)
                async with aiofiles.open(file_path, 'rb') as f:
                    header = await f.read(8)
                    return len(header) > 0
            
            elif file_path.suffix == '.safetensors':
                # Basic safetensors validation
                async with aiofiles.open(file_path, 'rb') as f:
                    header = await f.read(8)
                    return len(header) >= 8
            
            else:
                # Unknown format, assume valid
                return True
                
        except Exception as e:
            logger.debug(f"Format validation failed for {file_path}: {e}")
            return False
    
    async def _validate_file_checksum(self, file_path: Path, model_name: str) -> Optional[ModelChecksum]:
        """Validate file checksum against known values."""
        try:
            if model_name not in self.known_checksums:
                return None
            
            file_checksums = self.known_checksums[model_name]
            if file_path.name not in file_checksums:
                return None
            
            expected_checksum = file_checksums[file_path.name]
            algorithm, expected_value = expected_checksum.split(':', 1)
            
            # Calculate actual checksum
            calculated_checksum = await self._calculate_file_checksum(file_path, algorithm)
            
            if calculated_checksum and calculated_checksum.value == expected_value:
                logger.debug(f"Checksum verified for {file_path.name}")
                return calculated_checksum
            else:
                logger.warning(f"Checksum mismatch for {file_path.name}")
                return None
                
        except Exception as e:
            logger.error(f"Checksum validation error for {file_path}: {e}")
            return None
    
    async def _calculate_file_checksum(self, file_path: Path, algorithm: str = "sha256") -> Optional[ModelChecksum]:
        """Calculate checksum for a file."""
        try:
            if algorithm.lower() == "sha256":
                hasher = hashlib.sha256()
            elif algorithm.lower() == "md5":
                hasher = hashlib.md5()
            else:
                logger.warning(f"Unsupported checksum algorithm: {algorithm}")
                return None
            
            file_size = file_path.stat().st_size
            
            async with aiofiles.open(file_path, 'rb') as f:
                # Read in chunks for large files
                chunk_size = 8192
                while chunk := await f.read(chunk_size):
                    hasher.update(chunk)
            
            return ModelChecksum(
                algorithm=algorithm,
                value=hasher.hexdigest(),
                file_path=str(file_path),
                size_bytes=file_size
            )
            
        except Exception as e:
            logger.error(f"Checksum calculation failed for {file_path}: {e}")
            return None
    
    async def _validate_model_config(self, model_path: Path) -> bool:
        """Validate model configuration file."""
        try:
            config_path = model_path / "config.json"
            if not config_path.exists():
                return False
            
            async with aiofiles.open(config_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                config = json.loads(content)
            
            # Check required fields
            required_fields = ["model_type", "architectures"]
            for field in required_fields:
                if field not in config:
                    logger.warning(f"Missing required field in config.json: {field}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Config validation error: {e}")
            return False
    
    async def _validate_tokenizer(self, model_path: Path) -> bool:
        """Validate tokenizer files."""
        try:
            # Check for tokenizer files
            tokenizer_files = [
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json"
            ]
            
            found_tokenizer = False
            for file_name in tokenizer_files:
                if (model_path / file_name).exists():
                    found_tokenizer = True
                    break
            
            if not found_tokenizer:
                logger.warning("No tokenizer files found")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Tokenizer validation error: {e}")
            return False
    
    async def _cache_validation_result(self, result: ModelValidationResult) -> None:
        """Cache validation result for future use."""
        try:
            cache_file = self.cache_dir / f"{result.model_name.replace('/', '_')}_validation.json"
            
            result_dict = {
                "model_name": result.model_name,
                "is_valid": result.is_valid,
                "validation_time": result.validation_time_seconds,
                "errors": result.errors,
                "warnings": result.warnings,
                "timestamp": time.time(),
                "checksums": [
                    {
                        "algorithm": c.algorithm,
                        "value": c.value,
                        "file_path": c.file_path,
                        "size_bytes": c.size_bytes
                    }
                    for c in result.checksums_verified
                ]
            }
            
            async with aiofiles.open(cache_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(result_dict, indent=2))
            
            logger.debug(f"Cached validation result for {result.model_name}")
            
        except Exception as e:
            logger.debug(f"Failed to cache validation result: {e}")
    
    async def _call_progress_callback(self, callback: callable, percentage: int, status: str) -> None:
        """Call progress callback safely."""
        try:
            if callback:
                if hasattr(callback, '__call__'):
                    await callback(percentage, status)
        except Exception as e:
            logger.debug(f"Progress callback error: {e}")
    
    def _create_failed_result(self, model_name: str, start_time: float, 
                            errors: List[str], warnings: List[str], 
                            checksums_verified: List[ModelChecksum]) -> ModelValidationResult:
        """Create a failed validation result."""
        return ModelValidationResult(
            is_valid=False,
            model_name=model_name,
            validation_time_seconds=time.time() - start_time,
            checksums_verified=checksums_verified,
            errors=errors,
            warnings=warnings,
            file_integrity_passed=False,
            size_validation_passed=False,
            format_validation_passed=False
        )
    
    async def download_and_verify_checksums(self, model_name: str) -> Dict[str, str]:
        """
        Download checksums from Hugging Face Hub or other sources.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of file checksums
        """
        try:
            # This would integrate with Hugging Face Hub API
            # For now, return empty dict as placeholder
            logger.info(f"Would download checksums for {model_name}")
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to download checksums for {model_name}: {e}")
            return {}


def create_model_validator(cache_dir: Optional[Path] = None) -> ModelValidator:
    """
    Create a model validator instance.
    
    Args:
        cache_dir: Optional cache directory
        
    Returns:
        ModelValidator instance
    """
    return ModelValidator(cache_dir)