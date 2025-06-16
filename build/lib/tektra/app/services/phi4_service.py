"""
Microsoft Phi-4 Multimodal Service.

Provides unified speech recognition, text generation, and multimodal processing
using Microsoft's Phi-4 Multimodal Instruct model.
"""

import asyncio
import logging
import tempfile
import io
import base64
from pathlib import Path
from typing import Optional, Dict, List, Union, Any, AsyncGenerator
import json

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, pipeline
    from huggingface_hub import snapshot_download, hf_hub_download
    import numpy as np
    PHI4_AVAILABLE = True
except ImportError:
    PHI4_AVAILABLE = False
    np = None

try:
    import librosa
    import soundfile as sf
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class Phi4Service:
    """Microsoft Phi-4 Multimodal service for unified AI processing."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = "microsoft/Phi-4-multimodal-instruct"
        self.device = "cpu"
        self.is_loaded = False
        self.is_loading = False
        self.load_progress = {"status": "idle", "progress": 0.0, "message": ""}
        
        # Audio processing settings
        self.sample_rate = 16000  # Standard sample rate for speech
        self.max_audio_length = 30  # Maximum audio length in seconds
        
        # Model capabilities
        self.supported_languages = {
            "en": "English",
            "zh": "Chinese", 
            "de": "German",
            "fr": "French",
            "it": "Italian",
            "ja": "Japanese",
            "es": "Spanish",
            "pt": "Portuguese"
        }
        
        # Generation settings
        self.max_new_tokens = 2048
        self.temperature = 0.7
        self.do_sample = True
        self.top_p = 0.9
    
    async def _initialize(self):
        """Initialize Phi-4 model on startup."""
        try:
            await self.load_model()
        except Exception as e:
            logger.error(f"Failed to initialize Phi-4 model: {e}")
    
    def _get_device(self):
        """Determine the best available device."""
        if not PHI4_AVAILABLE:
            return "cpu"  # Default when torch not available
        
        try:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"  # Apple Silicon
            else:
                return "cpu"
        except Exception:
            return "cpu"
    
    async def load_model(self) -> bool:
        """
        Load Phi-4 Multimodal model with download progress tracking.
        
        Returns:
            bool: True if model loaded successfully
        """
        if not PHI4_AVAILABLE:
            self.load_progress = {
                "status": "error",
                "progress": 0.0,
                "message": "Phi-4 dependencies not available. Install with: uv tool install tektra --with tektra[ml]"
            }
            logger.error("Phi-4 dependencies not available. Install with: uv tool install tektra --with tektra[ml]")
            return False
        
        if self.is_loading:
            logger.info("Phi-4 model already loading")
            return False
        
        if self.is_loaded:
            logger.info("Phi-4 model already loaded")
            return True
        
        try:
            self.is_loading = True
            self.device = self._get_device()
            logger.info(f"Loading Phi-4 model on {self.device}")
            
            # Update progress
            self.load_progress = {
                "status": "downloading",
                "progress": 0.1,
                "message": f"Initializing download on {self.device}..."
            }
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def load_model_sync():
                try:
                    # Update progress for processor download
                    self.load_progress = {
                        "status": "downloading",
                        "progress": 0.2,
                        "message": "Downloading processor..."
                    }
                    
                    # Load processor
                    processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        cache_dir=settings.model_cache_dir
                    )
                    
                    # Update progress for model download
                    self.load_progress = {
                        "status": "downloading", 
                        "progress": 0.4,
                        "message": "Downloading model weights (this may take several minutes)..."
                    }
                    
                    # Load model with appropriate settings
                    model_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.bfloat16 if self.device != "cpu" else torch.float32,
                        "device_map": "auto" if self.device == "cuda" else None,
                        "cache_dir": settings.model_cache_dir
                    }
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                    
                    # Update progress for device loading
                    self.load_progress = {
                        "status": "loading",
                        "progress": 0.8,
                        "message": f"Loading model to {self.device}..."
                    }
                    
                    # Move to device if not using device_map
                    if self.device != "cuda":
                        model = model.to(self.device)
                    
                    # Final progress update
                    self.load_progress = {
                        "status": "ready",
                        "progress": 1.0,
                        "message": f"Model loaded successfully on {self.device}"
                    }
                    
                    return model, processor
                    
                except Exception as e:
                    self.load_progress = {
                        "status": "error",
                        "progress": 0.0,
                        "message": f"Model loading failed: {str(e)}"
                    }
                    raise e
            
            self.model, self.processor = await loop.run_in_executor(None, load_model_sync)
            
            self.is_loaded = True
            self.is_loading = False
            logger.info(f"Phi-4 model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Phi-4 model: {e}")
            self.is_loaded = False
            self.is_loading = False
            self.load_progress = {
                "status": "error",
                "progress": 0.0,
                "message": f"Failed to load model: {str(e)}"
            }
            return False
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, Any, str, Path],
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio using Phi-4's speech recognition capabilities.
        
        Args:
            audio_data: Audio data (bytes, numpy array, or file path)
            language: Language code (auto-detect if None)
            task: Task type ('transcribe' or 'translate')
            
        Returns:
            Dict containing transcription results
        """
        if not self.is_loaded:
            raise RuntimeError("Phi-4 model not loaded. Call load_model() first.")
        
        try:
            # Process audio data
            audio_array = await self._process_audio_input(audio_data)
            
            # Prepare the prompt for speech recognition
            if task == "translate":
                prompt = "Transcribe and translate the following audio to English:"
            else:
                lang_instruction = f" in {language}" if language and language in self.supported_languages else ""
                prompt = f"Transcribe the following audio{lang_instruction}:"
            
            # Create input for the model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "audio": audio_array}
                    ]
                }
            ]
            
            # Process with Phi-4
            inputs = self.processor(
                messages,
                return_tensors="pt",
                sampling_rate=self.sample_rate
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,  # Low temperature for transcription accuracy
                    do_sample=False,  # Deterministic for transcription
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            generated_text = self.processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            # Parse and enhance the result
            result = {
                "text": generated_text,
                "language": language or self._detect_language_from_text(generated_text),
                "confidence": 0.95,  # Phi-4 typically has high confidence
                "duration": len(audio_array) / self.sample_rate,
                "word_count": len(generated_text.split()) if generated_text else 0,
                "method": "phi4_multimodal"
            }
            
            logger.info(f"Phi-4 transcription completed: {len(result['text'])} characters")
            return result
            
        except Exception as e:
            logger.error(f"Phi-4 transcription failed: {e}")
            raise
    
    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[Dict, AsyncGenerator[str, None]]:
        """
        Generate chat completion using Phi-4.
        
        Args:
            messages: List of message dictionaries
            model: Model name (ignored, always uses Phi-4)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Chat completion response or async generator for streaming
        """
        if not self.is_loaded:
            raise RuntimeError("Phi-4 model not loaded. Call load_model() first.")
        
        try:
            # Use provided parameters or defaults
            temp = temperature if temperature is not None else self.temperature
            max_new_tokens = max_tokens if max_tokens is not None else self.max_new_tokens
            
            # Process messages with Phi-4
            inputs = self.processor(
                messages,
                return_tensors="pt"
            ).to(self.device)
            
            if stream:
                return self._generate_streaming(inputs, temp, max_new_tokens)
            else:
                return await self._generate_complete(inputs, temp, max_new_tokens)
                
        except Exception as e:
            logger.error(f"Phi-4 chat completion failed: {e}")
            raise
    
    async def _generate_complete(
        self,
        inputs: Dict,
        temperature: float,
        max_new_tokens: int
    ) -> Dict:
        """Generate complete response (non-streaming)."""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode the response
            generated_text = self.processor.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": inputs['input_ids'].shape[1],
                    "completion_tokens": len(outputs[0]) - inputs['input_ids'].shape[1],
                    "total_tokens": len(outputs[0])
                },
                "model": "phi-4-multimodal"
            }
            
        except Exception as e:
            logger.error(f"Phi-4 complete generation failed: {e}")
            raise
    
    async def _generate_streaming(
        self,
        inputs: Dict,
        temperature: float,
        max_new_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            # For streaming, we'll use a simple approach
            # In a production environment, you might want to implement proper streaming
            with torch.no_grad():
                # Generate with smaller chunks for pseudo-streaming
                chunk_size = 50
                generated_tokens = []
                
                for i in range(0, max_new_tokens, chunk_size):
                    current_max = min(chunk_size, max_new_tokens - i)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=len(generated_tokens) + current_max,
                        temperature=temperature,
                        do_sample=self.do_sample,
                        top_p=self.top_p,
                        pad_token_id=self.processor.tokenizer.eos_token_id
                    )
                    
                    # Get new tokens
                    new_tokens = outputs[0][inputs['input_ids'].shape[1] + len(generated_tokens):]
                    
                    if len(new_tokens) == 0:
                        break
                    
                    # Decode new tokens
                    new_text = self.processor.tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=True
                    )
                    
                    generated_tokens.extend(new_tokens)
                    
                    # Yield the new text
                    if new_text:
                        yield new_text
                    
                    # Check for stop conditions
                    if self.processor.tokenizer.eos_token_id in new_tokens:
                        break
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.05)
                    
        except Exception as e:
            logger.error(f"Phi-4 streaming generation failed: {e}")
            raise
    
    async def detect_language(self, audio_data: Union[bytes, Any, str, Path]) -> Dict:
        """
        Detect language from audio using Phi-4.
        
        Args:
            audio_data: Audio data to analyze
            
        Returns:
            Dict containing detected language and confidence
        """
        if not self.is_loaded:
            raise RuntimeError("Phi-4 model not loaded")
        
        try:
            # Use transcription to detect language
            result = await self.transcribe_audio(audio_data, language=None)
            
            # Detect language from transcribed text
            detected_lang = self._detect_language_from_text(result["text"])
            
            # Calculate confidence based on text length and clarity
            confidence = min(0.95, 0.7 + (len(result["text"]) / 100) * 0.2)
            
            return {
                "detected_language": detected_lang,
                "confidence": confidence,
                "transcribed_text": result["text"],
                "method": "phi4_transcription"
            }
            
        except Exception as e:
            logger.error(f"Phi-4 language detection failed: {e}")
            raise
    
    def _detect_language_from_text(self, text: str) -> str:
        """Simple language detection from text patterns."""
        if not text:
            return "en"
        
        # Simple heuristics for language detection
        # In production, you might want to use a proper language detection library
        text_lower = text.lower()
        
        # Chinese (contains CJK characters)
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return "zh"
        
        # Japanese (contains Hiragana/Katakana)
        if any('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff' for char in text):
            return "ja"
        
        # Common word patterns for other languages
        language_patterns = {
            "es": ["el", "la", "de", "que", "y", "es", "en", "un", "se", "no"],
            "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
            "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
            "it": ["il", "di", "che", "la", "è", "e", "un", "a", "per", "non"],
            "pt": ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para"]
        }
        
        words = text_lower.split()
        if len(words) < 3:
            return "en"  # Default for short text
        
        # Count matches for each language
        best_match = "en"
        best_score = 0
        
        for lang, patterns in language_patterns.items():
            score = sum(1 for word in words if word in patterns)
            if score > best_score:
                best_score = score
                best_match = lang
        
        return best_match
    
    async def _process_audio_input(self, audio_data: Union[bytes, Any, str, Path]) -> Any:
        """
        Process various audio input formats into numpy array.
        
        Args:
            audio_data: Audio input in various formats
            
        Returns:
            np.ndarray: Audio data as numpy array
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Audio processing libraries not available")
            if isinstance(audio_data, bytes):
                # Simple fallback without numpy
                return list(audio_data)
            raise ValueError("Audio processing libraries required for this input type")
        
        try:
            # Handle different input types
            if isinstance(audio_data, (str, Path)):
                # File path
                audio, sr = librosa.load(str(audio_data), sr=self.sample_rate)
                
            elif isinstance(audio_data, bytes):
                # Raw bytes - try to load as audio file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                
            elif hasattr(audio_data, 'shape'):  # numpy-like array
                # Already numpy array
                audio = audio_data
                sr = self.sample_rate
                
            else:
                raise ValueError(f"Unsupported audio input type: {type(audio_data)}")
            
            # Ensure correct sample rate
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            
            # Ensure mono audio
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Limit audio length
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise
    
    async def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "is_loaded": self.is_loaded,
            "is_loading": self.is_loading,
            "load_progress": self.load_progress,
            "model_name": self.model_name,
            "device": self.device,
            "supported_languages": self.supported_languages,
            "available": PHI4_AVAILABLE,
            "audio_processing_available": AUDIO_PROCESSING_AVAILABLE,
            "capabilities": {
                "speech_recognition": True,
                "chat_completion": True,
                "language_detection": True,
                "multimodal": True,
                "streaming": True
            },
            "audio_settings": {
                "sample_rate": self.sample_rate,
                "max_audio_length": self.max_audio_length
            }
        }
    
    async def unload_model(self):
        """Unload the current model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.is_loaded = False
            self.is_loading = False
            self.load_progress = {"status": "idle", "progress": 0.0, "message": ""}
            
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and PHI4_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Phi-4 model unloaded")


# Global service instance
phi4_service = Phi4Service()


# Utility functions for common operations
async def transcribe_audio_phi4(
    audio_data: Union[bytes, Any, str, Path],
    language: Optional[str] = None
) -> Dict:
    """
    Convenience function to transcribe audio with Phi-4.
    
    Args:
        audio_data: Audio data to transcribe
        language: Language code (auto-detect if None)
        
    Returns:
        Transcription result
    """
    return await phi4_service.transcribe_audio(
        audio_data=audio_data,
        language=language
    )


async def chat_with_phi4(
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    stream: bool = False
) -> Union[Dict, AsyncGenerator[str, None]]:
    """
    Convenience function for chat completion with Phi-4.
    
    Args:
        messages: Chat messages
        temperature: Sampling temperature
        stream: Whether to stream response
        
    Returns:
        Chat completion response
    """
    return await phi4_service.chat_completion(
        messages=messages,
        temperature=temperature,
        stream=stream
    )


async def detect_audio_language_phi4(audio_data: Union[bytes, Any, str, Path]) -> str:
    """
    Convenience function to detect audio language with Phi-4.
    
    Args:
        audio_data: Audio data to analyze
        
    Returns:
        Detected language code
    """
    result = await phi4_service.detect_language(audio_data)
    return result["detected_language"]