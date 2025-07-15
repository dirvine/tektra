#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.24.0",
#     "pyaudio>=0.2.11",
#     "loguru>=0.7.0",
#     "asyncio-extensions>=0.1.0",
# ]
# ///
"""
Embedded Voice Pipeline

Voice conversation pipeline using embedded Unmute models.
No external services, no WebSockets, everything runs in-process.
"""

import asyncio
import time
import pyaudio
import numpy as np
from typing import Optional, Callable, Any
from loguru import logger
from pathlib import Path

from .unmute_embedded import EmbeddedUnmute
from .voice_patterns import get_voice_patterns, VoiceMode


class EmbeddedVoicePipeline:
    """Voice pipeline using embedded Unmute models."""
    
    def __init__(
        self,
        unmute: EmbeddedUnmute,
        on_transcription: Optional[Callable] = None,
        on_response: Optional[Callable] = None,
        on_audio_response: Optional[Callable] = None,
        on_status_change: Optional[Callable] = None
    ):
        """
        Initialize embedded voice pipeline.
        
        Args:
            unmute: Embedded Unmute instance
            on_transcription: Callback for transcribed text
            on_response: Callback for LLM response text
            on_audio_response: Callback for TTS audio
            on_status_change: Callback for status updates
        """
        self.unmute = unmute
        
        # Callbacks
        self.on_transcription = on_transcription
        self.on_response = on_response
        self.on_audio_response = on_audio_response
        self.on_status_change = on_status_change
        
        # Audio configuration
        self.audio_config = {
            "chunk_size": 1024,
            "sample_rate": 16000,
            "channels": 1,
            "format": pyaudio.paInt16,
            "input": True,
        }
        
        # Audio interface
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        
        # State
        self.is_initialized = False
        self.is_recording = False
        self.is_speaking = False
        self.recording_task = None
        self.processing_task = None
        
        # Audio queues
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        
        # Voice interaction patterns
        self.voice_patterns = get_voice_patterns()
        self.current_voice_mode = VoiceMode.INACTIVE
        
        # Performance tracking
        self.last_audio_timestamp = 0
        self.transcription_start_time = 0
        
        logger.info("Initialized embedded voice pipeline with voice patterns")
        
    async def initialize(self) -> bool:
        """Initialize the voice pipeline."""
        if self.is_initialized:
            return True
            
        try:
            await self._update_status("Initializing embedded voice system...")
            
            # Initialize audio system
            self.audio = pyaudio.PyAudio()
            
            # Initialize input stream
            self.input_stream = self.audio.open(
                format=self.audio_config["format"],
                channels=self.audio_config["channels"],
                rate=self.audio_config["sample_rate"],
                input=True,
                frames_per_buffer=self.audio_config["chunk_size"],
                stream_callback=None  # We'll read manually for better control
            )
            
            # Initialize output stream
            self.output_stream = self.audio.open(
                format=self.audio_config["format"],
                channels=self.audio_config["channels"],
                rate=self.audio_config["sample_rate"],
                output=True,
                frames_per_buffer=self.audio_config["chunk_size"]
            )
            
            # Start processing tasks
            self.processing_task = asyncio.create_task(self._process_audio_pipeline())
            
            self.is_initialized = True
            self.current_voice_mode = VoiceMode.INACTIVE
            await self._update_status("Embedded voice system ready")
            logger.success("Embedded voice pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedded voice pipeline: {e}")
            await self._update_status(f"Initialization error: {e}")
            return False
            
    async def start_listening(self) -> bool:
        """Start listening for voice input."""
        if not self.is_initialized:
            logger.error("Cannot start listening: pipeline not initialized")
            return False
            
        if self.is_recording:
            logger.warning("Already listening")
            return True
            
        try:
            await self._update_status("Starting voice input...")
            
            self.is_recording = True
            self.current_voice_mode = VoiceMode.LISTENING
            self.transcription_start_time = time.time()
            
            # Start recording loop
            self.recording_task = asyncio.create_task(self._recording_loop())
            
            status_msg = self.voice_patterns.get_voice_status_message(VoiceMode.LISTENING)
            await self._update_status(status_msg)
            logger.info("Voice input started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start voice input: {e}")
            await self._update_status(f"Voice input error: {e}")
            return False
            
    async def stop_listening(self) -> bool:
        """Stop listening for voice input."""
        if not self.is_recording:
            return True
            
        try:
            self.is_recording = False
            self.current_voice_mode = VoiceMode.INACTIVE
            
            # Cancel recording task
            if self.recording_task and not self.recording_task.done():
                self.recording_task.cancel()
                try:
                    await self.recording_task
                except asyncio.CancelledError:
                    logger.debug("Recording task cancelled during voice input stop")
                    
            status_msg = self.voice_patterns.get_voice_status_message(VoiceMode.INACTIVE)
            await self._update_status(status_msg)
            logger.info("Voice input stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping voice input: {e}")
            return False
            
    async def _recording_loop(self):
        """Continuous audio recording loop."""
        logger.debug("Audio recording loop started")
        
        try:
            while self.is_recording and self.input_stream:
                try:
                    # Read audio chunk
                    audio_data = self.input_stream.read(
                        self.audio_config["chunk_size"], 
                        exception_on_overflow=False
                    )
                    
                    # Convert to numpy array for processing
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Simple voice activity detection (basic energy threshold)
                    energy = np.sum(audio_np.astype(np.float32) ** 2)
                    if energy > 1000000:  # Adjust threshold as needed
                        # Add audio to processing queue
                        await self.audio_input_queue.put(audio_data)
                        
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
            
    async def _process_audio_pipeline(self):
        """Process audio through the complete STT -> LLM -> TTS pipeline."""
        while True:
            try:
                if not self.is_recording:
                    await asyncio.sleep(0.1)
                    continue
                    
                # Collect audio for processing
                audio_chunks = []
                collection_start = time.time()
                
                # Collect audio for a short duration (e.g., 2 seconds)
                while time.time() - collection_start < 2.0:
                    try:
                        audio_chunk = await asyncio.wait_for(
                            self.audio_input_queue.get(), 
                            timeout=0.1
                        )
                        audio_chunks.append(audio_chunk)
                    except asyncio.TimeoutError:
                        if audio_chunks:
                            break  # Process what we have
                        continue
                        
                if not audio_chunks:
                    continue
                    
                # Process through complete pipeline
                await self._process_conversation_turn(audio_chunks)
                
            except Exception as e:
                logger.error(f"Error in audio processing pipeline: {e}")
                await asyncio.sleep(0.1)
                
    async def _process_conversation_turn(self, audio_chunks: list[bytes]):
        """Process a complete conversation turn through STT -> LLM -> TTS."""
        try:
            await self._update_status("Processing...")
            
            # Create audio stream generator
            async def audio_stream():
                for chunk in audio_chunks:
                    yield chunk
                    
            # Step 1: Transcribe audio
            full_transcript = ""
            async for transcript_chunk in self.unmute.transcribe_stream(audio_stream()):
                full_transcript += transcript_chunk
                
                # Check for wake words as we receive chunks
                if self.voice_patterns.detect_wake_word(transcript_chunk):
                    logger.info(f"Wake word detected in chunk: {transcript_chunk}")
                    self.current_voice_mode = VoiceMode.LISTENING
                
                # Notify UI of transcription
                if self.on_transcription:
                    await self._safe_callback(self.on_transcription, transcript_chunk)
                    
            if not full_transcript.strip():
                logger.debug("No transcription received")
                return
                
            # Process voice input through patterns
            voice_input_result = self.voice_patterns.process_voice_input(
                full_transcript, 
                context={"voice_mode": self.current_voice_mode}
            )
            
            logger.debug(f"Voice pattern result: {voice_input_result}")
            
            # Check if we should activate voice mode
            if not self.voice_patterns.should_activate_voice(full_transcript):
                logger.debug("Voice not activated, ignoring transcript")
                return
                
            # Calculate transcription latency
            if self.transcription_start_time > 0:
                latency = time.time() - self.transcription_start_time
                logger.debug(f"STT latency: {latency:.2f}s")
                
            logger.info(f"Full transcription: {full_transcript}")
            
            # Step 2: Generate LLM response
            self.current_voice_mode = VoiceMode.THINKING
            await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.THINKING))
            
            # Check if we have a predefined response from voice patterns
            if voice_input_result["response"]:
                full_response = voice_input_result["response"]
                
                # Use pattern response directly
                if self.on_response:
                    await self._safe_callback(self.on_response, full_response)
                    
                async def response_stream():
                    yield full_response
            else:
                # Generate response using LLM
                full_response = ""
                async def response_stream():
                    nonlocal full_response
                    async for response_chunk in self.unmute.generate_response_stream(full_transcript):
                        full_response += response_chunk
                        
                        # Stream response to UI
                        if self.on_response:
                            await self._safe_callback(self.on_response, response_chunk)
                            
                        yield response_chunk
                    
            # Step 3: Synthesize speech
            self.current_voice_mode = VoiceMode.SPEAKING
            await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.SPEAKING))
            self.is_speaking = True
            
            # Adjust response for voice-optimized output
            voice_optimized_response = self.voice_patterns.adjust_response_for_voice(full_response)
            
            # Create stream for the voice-optimized response
            async def optimized_response_stream():
                yield voice_optimized_response
            
            async for audio_chunk in self.unmute.synthesize_speech_stream(optimized_response_stream()):
                # Queue audio for playback
                await self.audio_output_queue.put(audio_chunk)
                
                # Notify UI
                if self.on_audio_response:
                    await self._safe_callback(self.on_audio_response, audio_chunk)
                    
            # Play audio output
            await self._play_queued_audio()
            
            self.is_speaking = False
            
            if self.is_recording:
                self.current_voice_mode = VoiceMode.LISTENING
                await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.LISTENING))
            else:
                self.current_voice_mode = VoiceMode.INACTIVE
                await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.INACTIVE))
                
            logger.info(f"Conversation turn complete. Response: {full_response[:100]}...")
            
        except Exception as e:
            logger.error(f"Error in conversation turn: {e}")
            self.is_speaking = False
            await self._update_status("Error in conversation")
            
    async def _play_queued_audio(self):
        """Play all queued audio chunks."""
        try:
            while not self.audio_output_queue.empty():
                audio_chunk = await self.audio_output_queue.get()
                
                if self.output_stream and self.output_stream.is_active():
                    self.output_stream.write(audio_chunk)
                    
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            
    async def send_text_message(self, text: str) -> bool:
        """
        Send text message directly to embedded LLM.
        
        Args:
            text: Text message to send
            
        Returns:
            bool: True if message sent successfully
        """
        if not self.is_initialized:
            logger.error("Cannot send text: pipeline not initialized")
            return False
            
        try:
            await self._update_status("Processing text message...")
            
            # Process text through voice patterns
            voice_input_result = self.voice_patterns.process_voice_input(
                text, 
                context={"voice_mode": self.current_voice_mode, "is_text_input": True}
            )
            
            logger.debug(f"Voice pattern result for text: {voice_input_result}")
            
            # Check if we have a predefined response
            if voice_input_result["response"]:
                full_response = voice_input_result["response"]
                
                # Use pattern response directly
                if self.on_response:
                    await self._safe_callback(self.on_response, full_response)
                    
                async def response_stream():
                    yield full_response
            else:
                # Generate response using LLM
                full_response = ""
                async def response_stream():
                    nonlocal full_response
                    async for response_chunk in self.unmute.generate_response_stream(text):
                        full_response += response_chunk
                        
                        # Stream response to UI
                        if self.on_response:
                            await self._safe_callback(self.on_response, response_chunk)
                            
                        yield response_chunk
                    
            # Synthesize speech response
            self.current_voice_mode = VoiceMode.SPEAKING
            await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.SPEAKING))
            self.is_speaking = True
            
            # Adjust response for voice-optimized output
            voice_optimized_response = self.voice_patterns.adjust_response_for_voice(full_response)
            
            # Create stream for the voice-optimized response
            async def optimized_response_stream():
                yield voice_optimized_response
            
            async for audio_chunk in self.unmute.synthesize_speech_stream(optimized_response_stream()):
                # Queue audio for playback
                await self.audio_output_queue.put(audio_chunk)
                
                # Notify UI
                if self.on_audio_response:
                    await self._safe_callback(self.on_audio_response, audio_chunk)
                    
            # Play audio output
            await self._play_queued_audio()
            
            self.is_speaking = False
            self.current_voice_mode = VoiceMode.INACTIVE
            await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.INACTIVE))
            
            logger.info(f"Text message processed: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process text message: {e}")
            self.is_speaking = False
            await self._update_status("Error processing text")
            return False
            
    async def _safe_callback(self, callback: Callable, *args):
        """Safely execute callback with error handling."""
        if callback:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
                
    async def _update_status(self, status: str):
        """Update status via callback."""
        if self.on_status_change:
            await self._safe_callback(self.on_status_change, status)
            
    def get_pipeline_status(self) -> dict[str, Any]:
        """
        Get comprehensive pipeline status.
        
        Returns:
            Dict containing pipeline status information
        """
        return {
            "initialized": self.is_initialized,
            "recording": self.is_recording,
            "speaking": self.is_speaking,
            "current_voice_mode": self.current_voice_mode.value,
            "voice_patterns_enabled": True,
            "audio_config": self.audio_config,
            "last_audio_timestamp": self.last_audio_timestamp,
            "unmute_info": self.unmute.get_model_info(),
            "audio_input_queue_size": self.audio_input_queue.qsize(),
            "audio_output_queue_size": self.audio_output_queue.qsize(),
            "memory_usage": self.unmute.get_memory_usage()
        }
        
    async def check_interruption(self, new_input: str) -> bool:
        """
        Check if new voice input should interrupt current speaking.
        
        Args:
            new_input: New voice input to check
            
        Returns:
            bool: True if should interrupt current speaking
        """
        if not self.is_speaking:
            return False
            
        should_interrupt = self.voice_patterns.should_interrupt_speaking(new_input)
        
        if should_interrupt:
            logger.info(f"Interruption detected: {new_input}")
            # Stop current speech synthesis
            self.is_speaking = False
            self.current_voice_mode = VoiceMode.LISTENING
            await self._update_status(self.voice_patterns.get_voice_status_message(VoiceMode.LISTENING))
            
        return should_interrupt
    
    def get_conversation_starters(self) -> list[str]:
        """Get natural conversation starters for voice interaction."""
        return self.voice_patterns.get_conversation_starters()
    
    def get_natural_transitions(self) -> dict[str, str]:
        """Get natural transition phrases for voice interactions."""
        return self.voice_patterns.get_natural_transitions()

    async def reset_conversation(self):
        """Reset the conversation context."""
        try:
            await self.unmute.reset_conversation()
            self.current_voice_mode = VoiceMode.INACTIVE
            logger.info("Conversation context reset")
        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")
            
    async def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up embedded voice pipeline")
        
        # Stop recording
        await self.stop_listening()
        
        # Cancel processing task
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                logger.debug("Processing task cancelled during cleanup")
                
        # Stop audio streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        # Terminate PyAudio
        if self.audio:
            self.audio.terminate()
            
        # Clear queues
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        
        # Cleanup Unmute
        await self.unmute.cleanup()
        
        # Reset voice patterns state
        self.current_voice_mode = VoiceMode.INACTIVE
        
        self.is_initialized = False
        logger.info("Embedded voice pipeline cleanup complete")
        
    async def test_audio_loopback(self) -> bool:
        """Test audio input/output functionality."""
        try:
            logger.info("Testing audio loopback...")
            
            # Record a short audio snippet
            await self.start_listening()
            await asyncio.sleep(2.0)  # Record for 2 seconds
            await self.stop_listening()
            
            # Check if we captured audio
            if self.audio_input_queue.empty():
                logger.warning("No audio captured during loopback test")
                return False
                
            # Test audio playback with a simple tone
            test_duration = 1.0  # 1 second
            sample_rate = self.audio_config["sample_rate"]
            samples = int(test_duration * sample_rate)
            
            # Generate a simple sine wave
            t = np.linspace(0, test_duration, samples)
            frequency = 440  # A4 note
            audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
            
            # Convert to int16 and play
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            if self.output_stream:
                self.output_stream.write(audio_int16.tobytes())
                
            logger.success("Audio loopback test completed")
            return True
            
        except Exception as e:
            logger.error(f"Audio loopback test failed: {e}")
            return False