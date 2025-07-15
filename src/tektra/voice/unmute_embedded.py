#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.0.0",
#     "numpy>=1.24.0",
#     "loguru>=0.7.0",
#     "asyncio-extensions>=0.1.0",
# ]
# ///
"""
Embedded Unmute Integration

Direct Python integration of Unmute models without Docker or external services.
This module loads and runs Unmute models in-process.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
import torch
import numpy as np
from loguru import logger

# Add unmute to Python path
UNMUTE_PATH = Path(__file__).parent.parent.parent.parent / "unmute"
if UNMUTE_PATH.exists():
    sys.path.insert(0, str(UNMUTE_PATH))
    
    # Import Unmute components directly
    try:
        from unmute.stt.speech_to_text import SpeechToText, STTWordMessage
        from unmute.tts.text_to_speech import TextToSpeech, TTSAudioMessage
        from unmute.llm.chatbot import Chatbot
        from unmute.llm.system_prompt import ConstantInstructions
        from unmute.kyutai_constants import SAMPLE_RATE
        UNMUTE_AVAILABLE = True
    except ImportError as e:
        logger.error(f"Failed to import Unmute components: {e}")
        UNMUTE_AVAILABLE = False
else:
    UNMUTE_AVAILABLE = False
    logger.warning(f"Unmute submodule not found at {UNMUTE_PATH}")


class EmbeddedUnmute:
    """Direct integration of Unmute models without containers."""
    
    def __init__(
        self, 
        model_dir: Path,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        memory_limit_gb: float = 4.0
    ):
        """
        Initialize embedded Unmute.
        
        Args:
            model_dir: Directory containing model files
            device: Torch device to use (cuda/cpu)
            memory_limit_gb: Maximum memory to use for models
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Model instances
        self.stt_model = None
        self.llm_model = None
        self.tts_model = None
        
        # Conversation management
        self.chatbot = None
        self.conversation_context = []
        
        # Model configurations
        self.stt_config = {
            "model_path": self.model_dir / "stt" / "model.pt",
            "config_path": self.model_dir / "stt" / "config.json",
            "sample_rate": SAMPLE_RATE,
            "chunk_size": 480,  # 30ms chunks at 16kHz
        }
        
        self.llm_config = {
            "model_path": self.model_dir / "llm" / "model.pt",
            "config_path": self.model_dir / "llm" / "config.json",
            "max_tokens": 512,
            "temperature": 0.7,
        }
        
        self.tts_config = {
            "model_path": self.model_dir / "tts" / "model.pt",
            "config_path": self.model_dir / "tts" / "config.json",
            "sample_rate": SAMPLE_RATE,
            "voice_preset": "default",
        }
        
        logger.info(f"Initialized EmbeddedUnmute on {device}")
        
    async def initialize_models(self, progress_callback=None) -> bool:
        """
        Load all Unmute models into memory.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            bool: True if all models loaded successfully
        """
        if not UNMUTE_AVAILABLE:
            logger.error("Unmute components not available")
            return False
            
        try:
            # Initialize chatbot first (lightweight)
            if progress_callback:
                await progress_callback("Initializing conversation system", 0.1)
            
            self.chatbot = Chatbot()
            
            # Load STT model
            if progress_callback:
                await progress_callback("Loading Speech Recognition", 0.2)
                
            self.stt_model = await self._load_stt_model()
            
            if progress_callback:
                await progress_callback("Speech Recognition loaded", 0.4)
                
            # Load LLM model
            if progress_callback:
                await progress_callback("Loading Language Model", 0.4)
                
            self.llm_model = await self._load_llm_model()
            
            if progress_callback:
                await progress_callback("Language Model loaded", 0.7)
                
            # Load TTS model
            if progress_callback:
                await progress_callback("Loading Voice Synthesis", 0.7)
                
            self.tts_model = await self._load_tts_model()
            
            if progress_callback:
                await progress_callback("Voice Synthesis loaded", 1.0)
                
            logger.success("All Unmute models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Unmute models: {e}")
            return False
            
    async def _load_stt_model(self):
        """Load the STT model with memory management."""
        # For now, we'll use a mock implementation since we need actual model files
        # This would be replaced with actual model loading code
        logger.info("Loading STT model (mock implementation)")
        
        class MockSTTModel:
            def __init__(self, device):
                self.device = device
                
            async def transcribe_stream(self, audio_stream):
                """Mock transcription - in reality would use actual STT model."""
                buffer = []
                async for audio_chunk in audio_stream:
                    buffer.append(audio_chunk)
                    # Mock: every 5 chunks, yield a transcription
                    if len(buffer) >= 5:
                        yield "This is a mock transcription"
                        buffer = []
                        
        return MockSTTModel(self.device)
        
    async def _load_llm_model(self):
        """Load the LLM model with memory management."""
        # For now, we'll use a mock implementation since we need actual model files
        # This would be replaced with actual model loading code
        logger.info("Loading LLM model (mock implementation)")
        
        class MockLLMModel:
            def __init__(self, device):
                self.device = device
                
            async def generate_stream(self, prompt, max_tokens=512, temperature=0.7):
                """Mock LLM generation - in reality would use actual LLM model."""
                response_words = [
                    "I", "understand", "your", "question", "and", "I'm", "happy", 
                    "to", "help", "you", "with", "that", "topic", "."
                ]
                
                for word in response_words:
                    yield word + " "
                    await asyncio.sleep(0.1)  # Simulate streaming delay
                    
        return MockLLMModel(self.device)
        
    async def _load_tts_model(self):
        """Load the TTS model with memory management."""
        # For now, we'll use a mock implementation since we need actual model files
        # This would be replaced with actual model loading code
        logger.info("Loading TTS model (mock implementation)")
        
        class MockTTSModel:
            def __init__(self, device, sample_rate):
                self.device = device
                self.sample_rate = sample_rate
                
            async def synthesize_stream(self, text_stream):
                """Mock TTS synthesis - in reality would use actual TTS model."""
                async for text_chunk in text_stream:
                    # Generate mock audio data (sine wave)
                    duration = len(text_chunk) * 0.1  # 100ms per character
                    samples = int(duration * self.sample_rate)
                    
                    # Generate sine wave as mock audio
                    t = np.linspace(0, duration, samples)
                    frequency = 440  # A4 note
                    audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
                    
                    # Convert to int16 and then to bytes
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
                    
        return MockTTSModel(self.device, self.stt_config["sample_rate"])
        
    async def transcribe_stream(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[str, None]:
        """
        Transcribe streaming audio to text.
        
        Args:
            audio_stream: Async generator of audio chunks
            
        Yields:
            Transcribed text segments
        """
        if not self.stt_model:
            raise RuntimeError("STT model not initialized")
            
        async def audio_generator():
            async for audio_chunk in audio_stream:
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
                
                # Normalize audio
                audio_tensor = audio_array.astype(np.float32) / 32768.0
                yield audio_tensor
                
        # Process through STT model
        async for transcript in self.stt_model.transcribe_stream(audio_generator()):
            if transcript:
                yield transcript
                
    async def generate_response_stream(
        self,
        text: str,
        context: Optional[list] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate LLM response with streaming.
        
        Args:
            text: Input text
            context: Optional conversation context
            
        Yields:
            Response text tokens
        """
        if not self.llm_model:
            raise RuntimeError("LLM model not initialized")
            
        if not self.chatbot:
            raise RuntimeError("Chatbot not initialized")
            
        # Add user message to chatbot
        await self.chatbot.add_chat_message_delta(text, "user")
        
        # Get preprocessed messages for LLM
        messages = self.chatbot.preprocessed_messages()
        
        # Generate response using LLM
        full_response = ""
        async for token in self.llm_model.generate_stream(
            messages,
            max_tokens=self.llm_config["max_tokens"],
            temperature=self.llm_config["temperature"]
        ):
            full_response += token
            yield token
            
        # Add assistant response to chatbot
        await self.chatbot.add_chat_message_delta(full_response, "assistant")
        
    async def synthesize_speech_stream(
        self,
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from streaming text.
        
        Args:
            text_stream: Async generator of text segments
            
        Yields:
            Audio chunks as bytes
        """
        if not self.tts_model:
            raise RuntimeError("TTS model not initialized")
            
        # Buffer for accumulating text
        text_buffer = ""
        
        async def buffered_text_generator():
            nonlocal text_buffer
            async for text_chunk in text_stream:
                text_buffer += text_chunk
                
                # Process when we have enough text (e.g., complete sentence)
                if self._is_sentence_boundary(text_buffer):
                    yield text_buffer.strip()
                    text_buffer = ""
                    
            # Process any remaining text
            if text_buffer.strip():
                yield text_buffer.strip()
                
        # Generate audio for buffered text
        async for audio_chunk in self.tts_model.synthesize_stream(buffered_text_generator()):
            yield audio_chunk
            
    def _is_sentence_boundary(self, text: str) -> bool:
        """Check if text ends with sentence boundary."""
        sentence_endings = [".", "!", "?", "。", "！", "？"]
        return any(text.rstrip().endswith(end) for end in sentence_endings)
        
    async def process_conversation_turn(
        self,
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Process a complete conversation turn: STT -> LLM -> TTS.
        
        Args:
            audio_stream: Input audio stream
            
        Yields:
            Response audio chunks
        """
        try:
            # Step 1: Transcribe audio to text
            full_transcript = ""
            async for transcript_chunk in self.transcribe_stream(audio_stream):
                full_transcript += transcript_chunk
                
            if not full_transcript.strip():
                logger.warning("No transcription received")
                return
                
            logger.info(f"Transcribed: {full_transcript}")
            
            # Step 2: Generate LLM response
            async def response_generator():
                async for response_chunk in self.generate_response_stream(full_transcript):
                    yield response_chunk
                    
            # Step 3: Synthesize speech from response
            async for audio_chunk in self.synthesize_speech_stream(response_generator()):
                yield audio_chunk
                
        except Exception as e:
            logger.error(f"Error in conversation turn: {e}")
            raise
            
    async def cleanup(self):
        """Clean up resources and free memory."""
        logger.info("Cleaning up Unmute models")
        
        # Clear models from memory
        self.stt_model = None
        self.llm_model = None
        self.tts_model = None
        self.chatbot = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
    def get_memory_usage(self) -> dict:
        """Get current memory usage of models."""
        memory_info = {
            "stt_model": 0,
            "llm_model": 0,
            "tts_model": 0,
            "total": 0
        }
        
        # For mock implementation, return mock memory usage
        if self.stt_model:
            memory_info["stt_model"] = 512  # MB
            
        if self.llm_model:
            memory_info["llm_model"] = 2048  # MB
            
        if self.tts_model:
            memory_info["tts_model"] = 256  # MB
            
        memory_info["total"] = sum(
            memory_info[k] for k in ["stt_model", "llm_model", "tts_model"]
        )
        
        return memory_info
        
    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "stt_available": self.stt_model is not None,
            "llm_available": self.llm_model is not None,
            "tts_available": self.tts_model is not None,
            "chatbot_available": self.chatbot is not None,
            "device": self.device,
            "sample_rate": self.stt_config["sample_rate"],
            "memory_usage": self.get_memory_usage()
        }
        
    async def reset_conversation(self):
        """Reset the conversation context."""
        if self.chatbot:
            self.chatbot = Chatbot()
            logger.info("Conversation context reset")