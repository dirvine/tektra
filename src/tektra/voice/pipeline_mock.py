"""
Mock Voice Conversation Pipeline

This module provides a mock implementation of the voice conversation pipeline
that doesn't require audio dependencies, for clean builds and testing.
"""

import asyncio
import time
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from loguru import logger

from .services import UnmuteServiceManager
from .unmute_client import UnmuteWebSocketClient


class VoiceConversationPipeline:
    """
    Mock voice conversation pipeline for clean builds.
    
    This class provides the same interface as the real pipeline but
    without audio dependencies.
    """
    
    def __init__(self, 
                 service_manager: Optional[UnmuteServiceManager] = None,
                 unmute_path: Optional[Path] = None,
                 sample_rate: int = 16000,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 on_transcription: Optional[Callable] = None,
                 on_response: Optional[Callable] = None,
                 on_audio_response: Optional[Callable] = None,
                 on_status_change: Optional[Callable] = None):
        """
        Initialize mock voice conversation pipeline.
        
        Args:
            service_manager: Unmute service manager instance (optional)
            unmute_path: Path to Unmute installation (optional)
            sample_rate: Audio sample rate (Hz)
            chunk_size: Audio chunk size (samples)
            channels: Number of audio channels
            on_transcription: Callback for transcription events
            on_response: Callback for response events
            on_audio_response: Callback for audio response events
            on_status_change: Callback for status change events
        """
        self.service_manager = service_manager
        self.unmute_path = unmute_path
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        
        # Pipeline state
        self.is_recording = False
        self.is_speaking = False
        self.is_processing = False
        self.audio_enabled = False
        self.is_initialized = False
        
        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_transcription: Optional[Callable[[str], None]] = on_transcription
        self.on_response: Optional[Callable[[str], None]] = on_response
        self.on_audio_playback: Optional[Callable[[bytes], None]] = on_audio_response
        self.on_status_change: Optional[Callable] = on_status_change
        self.on_error: Optional[Callable[[Exception], None]] = None
        
        # WebSocket client
        self.ws_client: Optional[UnmuteWebSocketClient] = None
        
        logger.info("Mock voice conversation pipeline initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the mock voice pipeline.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Check if Unmute services are available
            if self.service_manager and not await self.service_manager.are_services_healthy():
                logger.warning("Unmute services not available, using mock mode")
            else:
                logger.info("Using mock voice pipeline (no service manager)")
            
            # Mock initialization - just mark as initialized
            self.is_initialized = True
            logger.info("Mock voice pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing mock voice pipeline: {e}")
            return False
    
    async def start_conversation(self) -> bool:
        """
        Start voice conversation (mock implementation).
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_recording:
                logger.warning("Voice conversation already active")
                return False
            
            self.is_recording = True
            self.audio_enabled = True
            
            if self.on_speech_start:
                self.on_speech_start()
            
            logger.info("Mock voice conversation started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting voice conversation: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def stop_conversation(self) -> bool:
        """
        Stop voice conversation (mock implementation).
        
        Returns:
            bool: True if stopped successfully
        """
        try:
            self.is_recording = False
            self.audio_enabled = False
            
            if self.on_speech_end:
                self.on_speech_end()
            
            logger.info("Mock voice conversation stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping voice conversation: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def process_text_input(self, text: str) -> Optional[str]:
        """
        Process text input through the mock pipeline.
        
        Args:
            text: Input text to process
            
        Returns:
            Optional[str]: Response text or None if error
        """
        try:
            if self.is_processing:
                logger.warning("Already processing input")
                return None
            
            self.is_processing = True
            
            # Simulate processing delay
            await asyncio.sleep(0.5)
            
            # Mock response
            response = f"Mock response to: '{text}'"
            
            if self.on_response:
                self.on_response(response)
            
            self.is_processing = False
            return response
            
        except Exception as e:
            logger.error(f"Error processing text input: {e}")
            if self.on_error:
                self.on_error(e)
            self.is_processing = False
            return None
    
    async def send_text_message(self, text: str) -> bool:
        """
        Send text message through the mock pipeline (compatible with smart router).
        
        Args:
            text: Text message to send
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Process the text and trigger callbacks
            response = await self.process_text_input(text)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error sending text message: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def _handle_transcription(self, text: str) -> None:
        """Handle transcription from STT service (mock)."""
        if self.on_transcription:
            self.on_transcription(text)
    
    async def _handle_response(self, text: str) -> None:
        """Handle response from LLM service (mock)."""
        if self.on_response:
            self.on_response(text)
    
    async def _handle_audio_chunk(self, audio_data: bytes) -> None:
        """Handle audio chunk from TTS service (mock)."""
        if self.on_audio_playback:
            self.on_audio_playback(audio_data)
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle pipeline errors (mock)."""
        logger.error(f"Mock voice pipeline error: {error}")
        if self.on_error:
            self.on_error(error)
    
    def get_audio_stats(self) -> Dict[str, Any]:
        """
        Get audio processing statistics (mock).
        
        Returns:
            Dict[str, Any]: Audio statistics
        """
        return {
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'channels': self.channels,
            'is_recording': self.is_recording,
            'is_speaking': self.is_speaking,
            'is_processing': self.is_processing,
            'audio_enabled': self.audio_enabled,
            'mode': 'mock'
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline status (mock).
        
        Returns:
            Dict[str, Any]: Pipeline status
        """
        return {
            'initialized': True,
            'connected': self.ws_client is not None,
            'recording': self.is_recording,
            'processing': self.is_processing,
            'audio_enabled': self.audio_enabled,
            'services_healthy': True,
            'mode': 'mock'
        }
    
    async def cleanup(self) -> None:
        """Clean up mock pipeline resources."""
        try:
            await self.stop_conversation()
            
            if self.ws_client:
                await self.ws_client.disconnect()
            
            logger.info("Mock voice pipeline cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up mock voice pipeline: {e}")