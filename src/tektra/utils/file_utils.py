"""
File Utilities

This module provides file management utilities for Tektra AI Assistant,
including file processing, validation, and format conversion.
"""

import os
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from loguru import logger
import base64

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - image processing will be limited")


class FileUtils:
    """File management utilities for Tektra AI Assistant."""
    
    # Supported file types
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
    SUPPORTED_DOCUMENT_FORMATS = {'.txt', '.md', '.pdf', '.docx', '.doc', '.rtf', '.odt'}
    SUPPORTED_AUDIO_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    
    # File size limits (in bytes)
    MAX_IMAGE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_AUDIO_SIZE = 200 * 1024 * 1024  # 200MB
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
    
    def __init__(self):
        """Initialize file utilities."""
        self.temp_dir = Path.home() / ".tektra" / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize mimetypes
        mimetypes.init()
        logger.info("File utilities initialized")
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a file for processing.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dict containing validation results
        """
        file_path = Path(file_path)
        
        validation_result = {
            "valid": False,
            "file_path": str(file_path),
            "exists": file_path.exists(),
            "size": 0,
            "extension": "",
            "mime_type": "",
            "file_type": "unknown",
            "errors": []
        }
        
        if not file_path.exists():
            validation_result["errors"].append("File does not exist")
            return validation_result
        
        if not file_path.is_file():
            validation_result["errors"].append("Path is not a file")
            return validation_result
        
        # Get file info
        validation_result["size"] = file_path.stat().st_size
        validation_result["extension"] = file_path.suffix.lower()
        validation_result["mime_type"] = mimetypes.guess_type(str(file_path))[0] or "unknown"
        
        # Determine file type and validate
        if validation_result["extension"] in self.SUPPORTED_IMAGE_FORMATS:
            validation_result["file_type"] = "image"
            if validation_result["size"] > self.MAX_IMAGE_SIZE:
                validation_result["errors"].append(f"Image file too large (max {self.MAX_IMAGE_SIZE // (1024*1024)}MB)")
        elif validation_result["extension"] in self.SUPPORTED_DOCUMENT_FORMATS:
            validation_result["file_type"] = "document"
            if validation_result["size"] > self.MAX_DOCUMENT_SIZE:
                validation_result["errors"].append(f"Document file too large (max {self.MAX_DOCUMENT_SIZE // (1024*1024)}MB)")
        elif validation_result["extension"] in self.SUPPORTED_AUDIO_FORMATS:
            validation_result["file_type"] = "audio"
            if validation_result["size"] > self.MAX_AUDIO_SIZE:
                validation_result["errors"].append(f"Audio file too large (max {self.MAX_AUDIO_SIZE // (1024*1024)}MB)")
        elif validation_result["extension"] in self.SUPPORTED_VIDEO_FORMATS:
            validation_result["file_type"] = "video"
            if validation_result["size"] > self.MAX_VIDEO_SIZE:
                validation_result["errors"].append(f"Video file too large (max {self.MAX_VIDEO_SIZE // (1024*1024)}MB)")
        else:
            validation_result["errors"].append(f"Unsupported file type: {validation_result['extension']}")
        
        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except Exception as e:
            validation_result["errors"].append(f"Cannot read file: {e}")
        
        validation_result["valid"] = len(validation_result["errors"]) == 0
        return validation_result
    
    def get_file_hash(self, file_path: Union[str, Path], algorithm: str = "sha256") -> str:
        """
        Calculate file hash.
        
        Args:
            file_path: Path to the file
            algorithm: Hash algorithm to use
            
        Returns:
            Hex string of the file hash
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hash_obj = hashlib.new(algorithm)
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            raise
    
    def read_text_file(self, file_path: Union[str, Path], encoding: str = "utf-8") -> str:
        """
        Read text content from a file.
        
        Args:
            file_path: Path to the text file
            encoding: Text encoding to use
            
        Returns:
            Text content of the file
        """
        file_path = Path(file_path)
        
        validation = self.validate_file(file_path)
        if not validation["valid"]:
            raise ValueError(f"File validation failed: {validation['errors']}")
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
                
        except UnicodeDecodeError:
            # Try different encodings
            for fallback_encoding in ['latin-1', 'cp1252', 'utf-16']:
                try:
                    with open(file_path, 'r', encoding=fallback_encoding) as f:
                        content = f.read()
                        logger.info(f"Read file with {fallback_encoding} encoding")
                        return content
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"Could not decode file {file_path} with any supported encoding")
    
    def process_image(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dict containing image information and processed data
        """
        file_path = Path(file_path)
        
        validation = self.validate_file(file_path)
        if not validation["valid"] or validation["file_type"] != "image":
            raise ValueError(f"Invalid image file: {validation['errors']}")
        
        result = {
            "file_path": str(file_path),
            "format": validation["extension"],
            "size": validation["size"],
            "dimensions": None,
            "mode": None,
            "base64_data": None,
            "thumbnail": None
        }
        
        if PIL_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    result["dimensions"] = img.size
                    result["mode"] = img.mode
                    
                    # Convert to RGB if necessary
                    if img.mode not in ('RGB', 'RGBA'):
                        img = img.convert('RGB')
                    
                    # Create thumbnail
                    thumbnail_size = (200, 200)
                    img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Convert to base64
                    import io
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    result["base64_data"] = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                    
            except Exception as e:
                logger.error(f"Error processing image {file_path}: {e}")
                raise
        else:
            # Fallback: just encode as base64
            try:
                with open(file_path, 'rb') as f:
                    result["base64_data"] = base64.b64encode(f.read()).decode('utf-8')
            except Exception as e:
                logger.error(f"Error encoding image {file_path}: {e}")
                raise
        
        return result
    
    def create_temp_file(self, content: bytes, suffix: str = ".tmp") -> Path:
        """
        Create a temporary file with the given content.
        
        Args:
            content: Binary content to write
            suffix: File suffix/extension
            
        Returns:
            Path to the created temporary file
        """
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.temp_dir,
                suffix=suffix,
                delete=False
            ) as tmp_file:
                tmp_file.write(content)
                temp_path = Path(tmp_file.name)
            
            logger.debug(f"Created temporary file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating temporary file: {e}")
            raise
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files.
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
            
        Returns:
            Number of files cleaned up
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        cleaned_count = 0
        
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file():
                    file_mtime = file_path.stat().st_mtime
                    if file_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
            return 0
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get comprehensive information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing file information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {"error": "File does not exist"}
        
        stat_info = file_path.stat()
        
        return {
            "path": str(file_path),
            "name": file_path.name,
            "stem": file_path.stem,
            "suffix": file_path.suffix,
            "size": stat_info.st_size,
            "size_human": self._format_bytes(stat_info.st_size),
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "accessed": stat_info.st_atime,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "mime_type": mimetypes.guess_type(str(file_path))[0],
            "permissions": oct(stat_info.st_mode)[-3:],
            "hash_sha256": self.get_file_hash(file_path) if file_path.is_file() else None
        }
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def find_files(self, 
                  directory: Union[str, Path], 
                  pattern: str = "*",
                  recursive: bool = True,
                  file_types: Optional[List[str]] = None) -> List[Path]:
        """
        Find files matching criteria.
        
        Args:
            directory: Directory to search in
            pattern: Glob pattern to match
            recursive: Search recursively
            file_types: List of file extensions to include
            
        Returns:
            List of matching file paths
        """
        directory = Path(directory)
        
        if not directory.exists():
            return []
        
        try:
            if recursive:
                files = directory.rglob(pattern)
            else:
                files = directory.glob(pattern)
            
            result = []
            for file_path in files:
                if file_path.is_file():
                    if file_types is None or file_path.suffix.lower() in file_types:
                        result.append(file_path)
            
            return sorted(result)
            
        except Exception as e:
            logger.error(f"Error finding files in {directory}: {e}")
            return []
    
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """
        Copy a file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        import shutil
        
        source = Path(source)
        destination = Path(destination)
        
        try:
            # Create destination directory if it doesn't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source, destination)
            
            logger.info(f"File copied from {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying file from {source} to {destination}: {e}")
            return False
    
    def move_file(self, source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """
        Move a file from source to destination.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        import shutil
        
        source = Path(source)
        destination = Path(destination)
        
        try:
            # Create destination directory if it doesn't exist
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(source, destination)
            
            logger.info(f"File moved from {source} to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving file from {source} to {destination}: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            else:
                logger.warning(f"File does not exist: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False