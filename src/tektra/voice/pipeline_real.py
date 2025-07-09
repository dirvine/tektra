"""
Voice Conversation Pipeline

This module orchestrates the complete voice conversation flow using Kyutai Unmute:
- Audio capture and streaming to STT
- LLM conversation processing  
- TTS audio playback
- Integration with the GUI and smart routing
"""

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
import pyaudio
import numpy as np
from loguru import logger

from .services import UnmuteServiceManager
from .unmute_client import UnmuteWebSocketClient


class VoiceConversationPipeline:
    """
    Complete voice conversation pipeline using Kyutai Unmute.
    
    This class orchestrates the entire voice conversation flow:
    1. Audio capture from microphone
    2. Streaming to Unmute STT service
    3. LLM processing via Unmute backend
    4. TTS audio generation and playback
    5. Integration with GUI callbacks
    """

    def __init__(
        self, 
        unmute_path: Path,
        on_transcription: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_audio_response: Optional[Callable[[bytes], None]] = None,
        on_status_change: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the voice conversation pipeline.
        
        Args:
            unmute_path: Path to Unmute directory (git submodule)
            on_transcription: Callback for STT transcription results
            on_response: Callback for LLM text responses
            on_audio_response: Callback for TTS audio data
            on_status_change: Callback for status updates
        """
        self.unmute_path = Path(unmute_path)
        
        # Initialize Unmute components
        self.service_manager = UnmuteServiceManager(unmute_path)
        self.unmute_client = UnmuteWebSocketClient()
        
        # Audio configuration
        self.audio_config = {
            "chunk_size": 1024,
            "sample_rate": 16000,
            "channels": 1,
            "format": pyaudio.paInt16,
            "input": True
        }
        
        # State management
        self.is_initialized = False
        self.is_recording = False
        self.is_speaking = False
        self.audio_stream = None
        self.audio = None
        self.recording_task = None
        
        # Callbacks
        self.on_transcription = on_transcription
        self.on_response = on_response
        self.on_audio_response = on_audio_response
        self.on_status_change = on_status_change
        
        # Performance tracking
        self.last_audio_timestamp = 0
        self.transcription_start_time = 0
        
        # Setup message handlers
        self._setup_unmute_handlers()

    def _setup_unmute_handlers(self):
        """Setup WebSocket message handlers for Unmute responses."""
        
        async def handle_transcription(data: Dict[str, Any]):
            """Handle STT transcription results."""
            text = data.get("text", "").strip()
            confidence = data.get("confidence", 0.0)
            is_final = data.get("final", False)
            
            if text:
                # Calculate transcription latency
                if self.transcription_start_time > 0:
                    latency = time.time() - self.transcription_start_time
                    logger.debug(f"STT latency: {latency:.2f}s")
                
                logger.info(f"Transcription ({confidence:.2f}): {text}")
                
                if self.on_transcription and is_final:
                    try:
                        await self._safe_callback(self.on_transcription, text)
                    except Exception as e:
                        logger.error(f"Error in transcription callback: {e}")
        
        async def handle_llm_response(data: Dict[str, Any]):
            """Handle LLM text responses."""
            text = data.get("text", "")
            is_final = data.get("final", True)
            response_id = data.get("response_id")
            
            if text and is_final:
                logger.info(f"LLM Response: {text[:100]}...")
                
                if self.on_response:
                    try:
                        await self._safe_callback(self.on_response, text)
                    except Exception as e:
                        logger.error(f"Error in response callback: {e}")
        
        async def handle_audio_response(data: Dict[str, Any]):
            """Handle TTS audio responses."""
            audio_b64 = data.get("audio_data")
            format_info = data.get("format", {})
            is_final = data.get("final", True)
            
            if audio_b64:
                try:
                    import base64
                    audio_data = base64.b64decode(audio_b64)
                    logger.debug(f"Received TTS audio: {len(audio_data)} bytes")
                    
                    # Set speaking state
                    self.is_speaking = True
                    await self._update_status("Speaking...")
                    
                    if self.on_audio_response:
                        await self._safe_callback(self.on_audio_response, audio_data)
                    
                    # Auto-play audio (platform dependent)
                    await self._play_audio(audio_data, format_info)
                    
                    if is_final:
                        self.is_speaking = False
                        if self.is_recording:
                            await self._update_status("Listening...")
                        else:
                            await self._update_status("Ready")
                    
                except Exception as e:
                    logger.error(f"Error handling audio response: {e}")
        
        async def handle_error(data: Dict[str, Any]):
            """Handle error messages from Unmute."""
            error_type = data.get("error_type", "unknown")
            error_message = data.get("message", "Unknown error")
            
            logger.error(f"Unmute error ({error_type}): {error_message}")
            await self._update_status(f"Error: {error_message}")
        
        async def handle_status(data: Dict[str, Any]):
            """Handle status updates from Unmute."""
            status = data.get("status", "")
            service = data.get("service", "")
            
            if status:
                logger.debug(f"Unmute status ({service}): {status}")
                await self._update_status(f"{service}: {status}")
        
        # Register handlers
        self.unmute_client.on_message("transcription", handle_transcription)
        self.unmute_client.on_message("transcription_partial", handle_transcription) 
        self.unmute_client.on_message("llm_response", handle_llm_response)
        self.unmute_client.on_message("llm_response_partial", handle_llm_response)
        self.unmute_client.on_message("audio_response", handle_audio_response)
        self.unmute_client.on_message("tts_audio", handle_audio_response)
        self.unmute_client.on_message("error", handle_error)
        self.unmute_client.on_message("status", handle_status)

    async def initialize(self) -> bool:
        """
        Initialize the voice pipeline.
        
        Returns:
            bool: True if initialization successful
        """
        if self.is_initialized:
            return True
        
        try:
            await self._update_status("Initializing voice services...")
            
            # Start Unmute services
            logger.info("Starting Unmute services...")
            services_started = await self.service_manager.setup_unmute_services()
            
            if not services_started:
                await self._update_status("Failed to start voice services")
                return False
            
            await self._update_status("Connecting to voice backend...")
            
            # Connect to WebSocket
            connected = await self.unmute_client.connect()
            if not connected:
                await self._update_status("Failed to connect to voice backend")
                return False
            
            # Initialize PyAudio
            await self._update_status("Initializing audio system...")
            self.audio = pyaudio.PyAudio()
            
            # Start voice conversation session
            conversation_started = await self.unmute_client.start_voice_conversation()
            if not conversation_started:
                await self._update_status("Failed to start voice conversation")
                return False
            
            self.is_initialized = True
            await self._update_status("Voice system ready")
            logger.success("Voice conversation pipeline initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize voice pipeline: {e}")
            await self._update_status(f"Initialization error: {e}")
            return False

    async def start_listening(self) -> bool:
        """
        Start listening for voice input.
        
        Returns:
            bool: True if listening started successfully
        """
        if not self.is_initialized:
            logger.error("Cannot start listening: pipeline not initialized")
            return False
        
        if self.is_recording:
            logger.warning("Already listening")
            return True
        
        try:
            await self._update_status("Starting voice input...")
            
            # Start audio stream
            self.audio_stream = self.audio.open(
                format=self.audio_config["format"],
                channels=self.audio_config["channels"],
                rate=self.audio_config["sample_rate"],
                input=True,
                frames_per_buffer=self.audio_config["chunk_size"],
                stream_callback=None  # We'll read manually for better control
            )
            
            self.is_recording = True
            self.transcription_start_time = time.time()
            
            # Start recording loop
            self.recording_task = asyncio.create_task(self._recording_loop())
            
            await self._update_status("Listening...")
            logger.info("Voice input started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice input: {e}")
            await self._update_status(f"Voice input error: {e}")
            return False

    async def stop_listening(self) -> bool:
        """
        Stop listening for voice input.
        
        Returns:
            bool: True if listening stopped successfully
        """
        if not self.is_recording:
            return True
        
        try:
            self.is_recording = False
            
            # Cancel recording task
            if self.recording_task and not self.recording_task.done():
                self.recording_task.cancel()
                try:
                    await self.recording_task
                except asyncio.CancelledError:
                    pass
            
            # Stop audio stream
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            await self._update_status("Ready")
            logger.info("Voice input stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping voice input: {e}")
            return False

    async def _recording_loop(self):
        """Continuous audio recording and streaming loop."""
        logger.debug("Audio recording loop started")
        
        try:
            while self.is_recording and self.audio_stream:
                try:
                    # Read audio chunk
                    audio_data = self.audio_stream.read(
                        self.audio_config["chunk_size"], 
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy array for processing if needed
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Simple voice activity detection (basic energy threshold)
                    energy = np.sum(audio_np.astype(np.float32) ** 2)
                    if energy > 1000000:  # Adjust threshold as needed
                        # Send to Unmute for processing
                        success = await self.unmute_client.send_audio_chunk(audio_data)
                        if not success:
                            logger.warning("Failed to send audio chunk")
                    
                    # Update timestamp
                    self.last_audio_timestamp = time.time()
                    
                    # Small delay to prevent overwhelming the system
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error in recording loop: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Recording loop error: {e}")
        finally:
            logger.debug("Audio recording loop ended")

    async def send_text_message(self, text: str) -> bool:
        """
        Send text message directly to Unmute LLM.
        
        Args:
            text: Text message to send
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.is_initialized:
            logger.error("Cannot send text: pipeline not initialized")
            return False
        
        try:
            success = await self.unmute_client.send_text_message(text)
            if success:
                logger.info(f"Sent text message: {text[:50]}...")
                await self._update_status("Processing...")
            return success
        except Exception as e:
            logger.error(f"Failed to send text message: {e}")
            return False

    async def _play_audio(self, audio_data: bytes, format_info: Dict[str, Any]):
        """
        Play audio data (basic implementation).
        
        Args:
            audio_data: Raw audio data
            format_info: Audio format information
        """
        try:
            # This is a basic implementation - could be enhanced with proper audio playback
            # For now, we assume the Unmute TTS service handles playback
            pass
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    async def _safe_callback(self, callback: Callable, *args):
        """Safely execute callback with error handling."""
        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)

    async def _update_status(self, status: str):
        """Update status via callback."""
        if self.on_status_change:
            await self._safe_callback(self.on_status_change, status)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Dict containing pipeline status information
        """
        service_status = asyncio.create_task(self.service_manager.get_service_status())
        connection_status = self.unmute_client.get_connection_status()
        
        return {
            "initialized": self.is_initialized,
            "recording": self.is_recording,
            "speaking": self.is_speaking,
            "unmute_path": str(self.unmute_path),
            "audio_config": self.audio_config,
            "last_audio_timestamp": self.last_audio_timestamp,
            "service_manager": service_status,
            "websocket_client": connection_status
        }

    async def cleanup(self):
        """Cleanup pipeline resources."""
        logger.info("Cleaning up voice conversation pipeline...")
        
        # Stop listening
        await self.stop_listening()
        
        # End conversation
        await self.unmute_client.end_conversation()
        
        # Disconnect from WebSocket
        await self.unmute_client.cleanup()
        
        # Stop services
        await self.service_manager.cleanup()
        
        # Cleanup PyAudio
        if self.audio:
            try:
                self.audio.terminate()
            except Exception as e:
                logger.debug(f"Error terminating PyAudio: {e}")
        
        self.is_initialized = False
        logger.info("Voice conversation pipeline cleanup complete")