"""
OpenAI Whisper Speech-to-Text Service.

Provides speech recognition capabilities using OpenAI's Whisper model
with support for multiple languages, real-time processing, and audio optimization.
"""

import asyncio
import io
import logging
import tempfile
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import librosa
    import soundfile as sf

    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class WhisperService:
    """OpenAI Whisper speech-to-text service."""

    def __init__(self):
        self.model = None
        self.model_name = "base"
        self.device = "cpu"
        self.is_loaded = False
        self.supported_languages = self._get_supported_languages()

        # Audio processing settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_length = 30  # Seconds per chunk for long audio

        # Initialize on startup if available
        if WHISPER_AVAILABLE:
            asyncio.create_task(self._initialize())

    async def _initialize(self):
        """Initialize Whisper model on startup."""
        try:
            await self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")

    def _get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi",
            "nl": "Dutch",
            "pl": "Polish",
            "tr": "Turkish",
            "sv": "Swedish",
            "da": "Danish",
            "no": "Norwegian",
            "fi": "Finnish",
        }

    async def load_model(self, model_name: str = "base") -> bool:
        """
        Load Whisper model.

        Args:
            model_name: Model size ('tiny', 'base', 'small', 'medium', 'large')

        Returns:
            bool: True if model loaded successfully
        """
        if not WHISPER_AVAILABLE:
            logger.error(
                "Whisper not available. Install with: pip install openai-whisper"
            )
            return False

        try:
            # Determine device
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"  # Apple Silicon
            else:
                self.device = "cpu"

            logger.info(f"Loading Whisper model '{model_name}' on {self.device}")

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, lambda: whisper.load_model(model_name, device=self.device)
            )

            self.model_name = model_name
            self.is_loaded = True

            logger.info(f"Whisper model '{model_name}' loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.is_loaded = False
            return False

    async def transcribe_audio(
        self,
        audio_data: Union[bytes, BinaryIO, str, Path],
        language: Optional[str] = None,
        task: str = "transcribe",
        temperature: float = 0.0,
        best_of: int = 5,
        beam_size: int = 5,
        patience: float = 1.0,
        length_penalty: float = 1.0,
        suppress_tokens: str = "-1",
        initial_prompt: Optional[str] = None,
        decode_options: Optional[Dict] = None,
    ) -> Dict:
        """
        Transcribe audio using Whisper.

        Args:
            audio_data: Audio data (bytes, file-like object, or file path)
            language: Language code (auto-detect if None)
            task: 'transcribe' or 'translate'
            temperature: Sampling temperature
            best_of: Number of candidates when sampling
            beam_size: Beam size for beam search
            patience: Patience for beam search
            length_penalty: Length penalty for beam search
            suppress_tokens: Comma-separated list of token ids to suppress
            initial_prompt: Optional text to guide the transcription style
            decode_options: Additional decode options

        Returns:
            Dict containing transcription results
        """
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded. Call load_model() first.")

        try:
            # Process audio data
            audio_array = await self._process_audio_input(audio_data)

            # Prepare decode options
            options = {
                "language": language,
                "task": task,
                "temperature": temperature,
                "best_of": best_of,
                "beam_size": beam_size,
                "patience": patience,
                "length_penalty": length_penalty,
                "suppress_tokens": suppress_tokens,
                "initial_prompt": initial_prompt,
            }

            if decode_options:
                options.update(decode_options)

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self.model.transcribe(audio_array, **options)
            )

            # Process and enhance result
            enhanced_result = self._enhance_transcription_result(result)

            logger.info(
                f"Transcription completed: {len(enhanced_result['text'])} characters"
            )
            return enhanced_result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    async def transcribe_streaming(
        self,
        audio_chunks: List[bytes],
        language: Optional[str] = None,
        overlap_duration: float = 1.0,
    ) -> List[Dict]:
        """
        Transcribe streaming audio chunks with overlap handling.

        Args:
            audio_chunks: List of audio chunk data
            language: Language code
            overlap_duration: Overlap between chunks in seconds

        Returns:
            List of transcription results for each chunk
        """
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded")

        results = []
        overlap_samples = int(overlap_duration * self.sample_rate)

        try:
            for i, chunk_data in enumerate(audio_chunks):
                # Process chunk
                audio_array = await self._process_audio_input(chunk_data)

                # Add overlap from previous chunk if available
                if i > 0 and len(audio_array) > overlap_samples:
                    # This is a simplified overlap handling
                    # In practice, you'd want more sophisticated logic
                    pass

                # Transcribe chunk
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.model.transcribe(
                        audio_array, language=language, task="transcribe"
                    ),
                )

                chunk_result = self._enhance_transcription_result(result)
                chunk_result["chunk_index"] = i
                results.append(chunk_result)

            return results

        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            raise

    async def detect_language(
        self, audio_data: Union[bytes, BinaryIO, str, Path]
    ) -> Dict:
        """
        Detect the language of the audio.

        Args:
            audio_data: Audio data

        Returns:
            Dict containing detected language and confidence scores
        """
        if not self.is_loaded:
            raise RuntimeError("Whisper model not loaded")

        try:
            # Process audio
            audio_array = await self._process_audio_input(audio_data)

            # Detect language
            loop = asyncio.get_event_loop()

            # Load audio and pad/trim it to fit 30 seconds
            audio_array = whisper.pad_or_trim(audio_array)

            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio_array).to(self.model.device)

            # Detect the spoken language
            _, probs = await loop.run_in_executor(
                None, lambda: self.model.detect_language(mel)
            )

            # Get top languages
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]

            result = {
                "detected_language": top_languages[0][0],
                "confidence": top_languages[0][1],
                "all_languages": [
                    {
                        "language": lang,
                        "code": lang,
                        "name": self.supported_languages.get(lang, lang),
                        "probability": prob,
                    }
                    for lang, prob in top_languages
                ],
            }

            logger.info(
                f"Detected language: {result['detected_language']} ({result['confidence']:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            raise

    async def _process_audio_input(
        self, audio_data: Union[bytes, BinaryIO, str, Path]
    ) -> np.ndarray:
        """
        Process various audio input formats into numpy array.

        Args:
            audio_data: Audio input in various formats

        Returns:
            np.ndarray: Audio data as numpy array
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning(
                "Audio processing libraries not available. Install with: pip install librosa soundfile"
            )
            # Fallback: assume raw audio data
            if isinstance(audio_data, bytes):
                return np.frombuffer(audio_data, dtype=np.float32)
            raise ValueError("Audio processing libraries required for this input type")

        try:
            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                # File path
                audio, sr = librosa.load(str(audio_data), sr=self.sample_rate)

            elif isinstance(audio_data, bytes):
                # Raw bytes
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)

            elif hasattr(audio_data, "read"):
                # File-like object
                audio_bytes = audio_data.read()
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp_file:
                    tmp_file.write(audio_bytes)
                    tmp_file.flush()
                    audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)

            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_data)}")

            # Ensure correct sample rate
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Normalize audio
            audio = librosa.util.normalize(audio)

            return audio

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise

    def _enhance_transcription_result(self, result: Dict) -> Dict:
        """
        Enhance the transcription result with additional metadata.

        Args:
            result: Raw Whisper transcription result

        Returns:
            Enhanced result dictionary
        """
        enhanced = {
            "text": result["text"].strip(),
            "language": result.get("language"),
            "segments": [],
            "word_count": len(result["text"].split()),
            "duration": 0.0,
            "confidence": 0.0,
        }

        # Process segments if available
        if "segments" in result:
            total_duration = 0.0
            total_confidence = 0.0

            for segment in result["segments"]:
                segment_data = {
                    "id": segment.get("id"),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", "").strip(),
                    "confidence": segment.get("avg_logprob", 0.0),
                }

                enhanced["segments"].append(segment_data)
                total_duration = max(total_duration, segment_data["end"])
                total_confidence += segment_data["confidence"]

            enhanced["duration"] = total_duration
            if len(enhanced["segments"]) > 0:
                enhanced["confidence"] = total_confidence / len(enhanced["segments"])

        return enhanced

    async def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "is_loaded": self.is_loaded,
            "model_name": self.model_name,
            "device": self.device,
            "supported_languages": self.supported_languages,
            "available": WHISPER_AVAILABLE,
            "audio_processing_available": AUDIO_PROCESSING_AVAILABLE,
        }

    async def unload_model(self):
        """Unload the current model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False

            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Whisper model unloaded")


# Global service instance
whisper_service = WhisperService()


# Utility functions for common operations
async def transcribe_audio_file(
    file_path: Union[str, Path], language: Optional[str] = None
) -> Dict:
    """
    Convenience function to transcribe an audio file.

    Args:
        file_path: Path to audio file
        language: Language code (auto-detect if None)

    Returns:
        Transcription result
    """
    return await whisper_service.transcribe_audio(
        audio_data=file_path, language=language
    )


async def transcribe_audio_bytes(
    audio_bytes: bytes, language: Optional[str] = None
) -> Dict:
    """
    Convenience function to transcribe audio from bytes.

    Args:
        audio_bytes: Raw audio data
        language: Language code (auto-detect if None)

    Returns:
        Transcription result
    """
    return await whisper_service.transcribe_audio(
        audio_data=audio_bytes, language=language
    )


async def detect_audio_language(audio_data: Union[bytes, str, Path]) -> str:
    """
    Convenience function to detect audio language.

    Args:
        audio_data: Audio data or file path

    Returns:
        Detected language code
    """
    result = await whisper_service.detect_language(audio_data)
    return result["detected_language"]
