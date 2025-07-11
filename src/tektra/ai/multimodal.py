"""
Multimodal Processor

This module provides comprehensive multimodal processing capabilities:
- Document processing and analysis
- Image analysis and understanding
- Audio processing integration
- File format handling
- Data preparation for AI models
"""

import asyncio
import io
import mimetypes
from pathlib import Path
from typing import Any

import aiofiles
import numpy as np
from loguru import logger
from PIL import Image, ImageEnhance

# Document processing
try:
    import docx

    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not available - .docx files won't be processed")

try:
    import pypdf

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pypdf not available - .pdf files won't be processed")


class DocumentProcessor:
    """Process various document formats for AI analysis."""

    SUPPORTED_TEXT_FORMATS = {".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".log"}
    SUPPORTED_DOCUMENT_FORMATS = {".pdf", ".docx", ".doc"}

    @staticmethod
    async def process_file(file_path: str | Path) -> dict[str, Any]:
        """
        Process a file and extract text content.

        Args:
            file_path: Path to the file to process

        Returns:
            Dict containing processed content and metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file metadata
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "mime_type": mime_type,
            "extension": file_path.suffix.lower(),
            "content": "",
            "metadata": {},
            "processing_status": "success",
            "error": None,
        }

        try:
            extension = file_path.suffix.lower()

            if extension in DocumentProcessor.SUPPORTED_TEXT_FORMATS:
                result["content"] = await DocumentProcessor._process_text_file(
                    file_path
                )
            elif extension == ".pdf" and PDF_AVAILABLE:
                result["content"] = await DocumentProcessor._process_pdf_file(file_path)
            elif extension in {".docx", ".doc"} and DOCX_AVAILABLE:
                result["content"] = await DocumentProcessor._process_docx_file(
                    file_path
                )
            else:
                result["processing_status"] = "unsupported"
                result["error"] = f"Unsupported file format: {extension}"

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            result["processing_status"] = "error"
            result["error"] = str(e)

        return result

    @staticmethod
    async def _process_text_file(file_path: Path) -> str:
        """Process plain text files."""
        async with aiofiles.open(
            file_path, encoding="utf-8", errors="ignore"
        ) as f:
            return await f.read()

    @staticmethod
    async def _process_pdf_file(file_path: Path) -> str:
        """Process PDF files using pypdf."""
        content = []

        with open(file_path, "rb") as file:
            pdf_reader = pypdf.PdfReader(file)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    content.append(f"--- Page {page_num + 1} ---\n{text}")

        return "\n\n".join(content)

    @staticmethod
    async def _process_docx_file(file_path: Path) -> str:
        """Process DOCX files using python-docx."""
        doc = docx.Document(file_path)
        content = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text)

        return "\n\n".join(content)


class ImageProcessor:
    """Process and enhance images for AI analysis."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
    MAX_IMAGE_SIZE = (1024, 1024)  # Max dimensions for processing

    @staticmethod
    async def process_image(
        image_input: str | Path | bytes | Image.Image
    ) -> dict[str, Any]:
        """
        Process image for AI analysis.

        Args:
            image_input: Image file path, bytes, or PIL Image

        Returns:
            Dict containing processed image and metadata
        """
        try:
            # Load image from various input types
            if isinstance(image_input, str | Path):
                image = Image.open(image_input)
                source_type = "file"
                source_info = str(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
                source_type = "bytes"
                source_info = f"{len(image_input)} bytes"
            elif isinstance(image_input, Image.Image):
                image = image_input
                source_type = "pil_image"
                source_info = "PIL Image object"
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get original metadata
            original_size = image.size
            original_format = image.format

            # Resize if too large
            processed_image = ImageProcessor._resize_image(image)

            # Enhance image quality
            enhanced_image = ImageProcessor._enhance_image(processed_image)

            # Generate analysis metadata
            analysis = ImageProcessor._analyze_image(enhanced_image)

            result = {
                "image": enhanced_image,
                "original_size": original_size,
                "processed_size": enhanced_image.size,
                "original_format": original_format,
                "source_type": source_type,
                "source_info": source_info,
                "analysis": analysis,
                "processing_status": "success",
                "error": None,
            }

            logger.debug(f"Processed image: {source_info} -> {enhanced_image.size}")
            return result

        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"image": None, "processing_status": "error", "error": str(e)}

    @staticmethod
    def _resize_image(image: Image.Image) -> Image.Image:
        """Resize image to optimal size for AI processing."""
        if (
            image.size[0] <= ImageProcessor.MAX_IMAGE_SIZE[0]
            and image.size[1] <= ImageProcessor.MAX_IMAGE_SIZE[1]
        ):
            return image

        # Calculate new size maintaining aspect ratio
        ratio = min(
            ImageProcessor.MAX_IMAGE_SIZE[0] / image.size[0],
            ImageProcessor.MAX_IMAGE_SIZE[1] / image.size[1],
        )

        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))

        return image.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _enhance_image(image: Image.Image) -> Image.Image:
        """Apply basic enhancement to improve AI analysis."""
        try:
            # Slightly enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)

            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)

            return image
        except Exception as e:
            logger.debug(f"Image enhancement failed: {e}")
            return image

    @staticmethod
    def _analyze_image(image: Image.Image) -> dict[str, Any]:
        """Analyze image properties for metadata."""
        # Convert to numpy for analysis
        img_array = np.array(image)

        # Basic statistics
        mean_brightness = np.mean(img_array)
        std_brightness = np.std(img_array)

        # Color analysis
        dominant_colors = ImageProcessor._get_dominant_colors(img_array)

        return {
            "dimensions": image.size,
            "mean_brightness": float(mean_brightness),
            "brightness_std": float(std_brightness),
            "dominant_colors": dominant_colors,
            "aspect_ratio": image.size[0] / image.size[1],
            "total_pixels": image.size[0] * image.size[1],
        }

    @staticmethod
    def _get_dominant_colors(
        img_array: np.ndarray, n_colors: int = 3
    ) -> list[list[int]]:
        """Extract dominant colors from image."""
        try:
            # Reshape image to list of pixels
            pixels = img_array.reshape(-1, 3)

            # Simple clustering by finding most common colors
            # (This is a basic implementation - could use k-means for better results)
            unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

            # Get top colors by frequency
            top_indices = np.argsort(counts)[-n_colors:]
            dominant_colors = [unique_colors[i].tolist() for i in reversed(top_indices)]

            return dominant_colors

        except Exception as e:
            logger.debug(f"Color analysis failed: {e}")
            return [[128, 128, 128]]  # Default gray


class MultimodalProcessor:
    """
    Comprehensive multimodal processor for Tektra.

    Handles processing of various file types and media formats
    for AI analysis and integration.
    """

    def __init__(self):
        """Initialize multimodal processor."""
        self.document_processor = DocumentProcessor()
        self.image_processor = ImageProcessor()

        # Processing statistics
        self.processed_files = 0
        self.processing_errors = 0
        self.supported_formats = (
            DocumentProcessor.SUPPORTED_TEXT_FORMATS
            | DocumentProcessor.SUPPORTED_DOCUMENT_FORMATS
            | ImageProcessor.SUPPORTED_FORMATS
        )

        logger.info(
            f"Multimodal processor initialized with {len(self.supported_formats)} supported formats"
        )

    async def process_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        Process any supported file type.

        Args:
            file_path: Path to file to process

        Returns:
            Dict containing processed content and metadata
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        try:
            if (
                extension
                in DocumentProcessor.SUPPORTED_TEXT_FORMATS
                | DocumentProcessor.SUPPORTED_DOCUMENT_FORMATS
            ):
                result = await self.document_processor.process_file(file_path)
                result["content_type"] = "document"
            elif extension in ImageProcessor.SUPPORTED_FORMATS:
                result = await self.image_processor.process_image(file_path)
                result["content_type"] = "image"
            else:
                result = {
                    "file_path": str(file_path),
                    "content_type": "unsupported",
                    "processing_status": "unsupported",
                    "error": f"Unsupported file format: {extension}",
                    "supported_formats": list(self.supported_formats),
                }

            # Update statistics
            if result["processing_status"] == "success":
                self.processed_files += 1
            else:
                self.processing_errors += 1

            return result

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            self.processing_errors += 1

            return {
                "file_path": str(file_path),
                "content_type": "error",
                "processing_status": "error",
                "error": str(e),
            }

    async def process_multiple_files(
        self, file_paths: list[str | Path]
    ) -> list[dict[str, Any]]:
        """
        Process multiple files concurrently.

        Args:
            file_paths: List of file paths to process

        Returns:
            List of processing results
        """
        tasks = [self.process_file(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "file_path": str(file_paths[i]),
                        "content_type": "error",
                        "processing_status": "error",
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def prepare_for_ai_analysis(
        self, content: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """
        Prepare processed content for AI analysis.

        Args:
            content: Processed content from process_file
            query: User query about the content

        Returns:
            Dict formatted for AI model input
        """
        if content["content_type"] == "document":
            return await self._prepare_document_analysis(content, query)
        elif content["content_type"] == "image":
            return await self._prepare_image_analysis(content, query)
        else:
            raise ValueError(f"Cannot prepare content type: {content['content_type']}")

    async def _prepare_document_analysis(
        self, content: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Prepare document content for AI analysis."""
        document_text = content.get("content", "")

        # Truncate if too long (models have token limits)
        max_chars = 8000  # Approximate token limit consideration
        if len(document_text) > max_chars:
            document_text = document_text[:max_chars] + "... [truncated]"

        analysis_prompt = f"""Please analyze the following document and answer the user's question.

Document: {content['file_name']}
Content:
{document_text}

User Question: {query}

Please provide a comprehensive analysis addressing the user's question."""

        return {
            "type": "text_analysis",
            "prompt": analysis_prompt,
            "metadata": {
                "file_name": content["file_name"],
                "file_size": content.get("file_size"),
                "original_length": len(content.get("content", "")),
                "truncated": len(content.get("content", "")) > max_chars,
            },
        }

    async def _prepare_image_analysis(
        self, content: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Prepare image content for AI vision analysis."""
        if content["processing_status"] != "success":
            raise ValueError(
                f"Cannot analyze failed image processing: {content.get('error')}"
            )

        analysis_prompt = f"""Please analyze the provided image and answer the user's question.

Image: {content.get('source_info', 'Unknown')}
Dimensions: {content['processed_size']}

User Question: {query}

Please provide a detailed analysis of the image addressing the user's question."""

        return {
            "type": "vision_analysis",
            "prompt": analysis_prompt,
            "image": content["image"],
            "metadata": {
                "original_size": content["original_size"],
                "processed_size": content["processed_size"],
                "source_type": content["source_type"],
                "analysis": content.get("analysis", {}),
            },
        }

    def get_processor_stats(self) -> dict[str, Any]:
        """Get processing statistics."""
        return {
            "processed_files": self.processed_files,
            "processing_errors": self.processing_errors,
            "success_rate": self.processed_files
            / max(1, self.processed_files + self.processing_errors),
            "supported_formats": list(self.supported_formats),
            "document_formats_available": {
                "pdf": PDF_AVAILABLE,
                "docx": DOCX_AVAILABLE,
            },
        }

    def is_supported_format(self, file_path: str | Path) -> bool:
        """Check if file format is supported."""
        extension = Path(file_path).suffix.lower()
        return extension in self.supported_formats
