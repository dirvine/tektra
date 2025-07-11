"""
Unmute WebSocket Client

This module provides a WebSocket client for communicating with Kyutai Unmute services:
- Real-time audio streaming for STT
- Text message sending for LLM processing
- Audio response handling from TTS
- Event-driven message handling
"""

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import websockets
from loguru import logger
from websockets.exceptions import ConnectionClosed, WebSocketException


class UnmuteWebSocketClient:
    """
    WebSocket client for Kyutai Unmute backend communication.

    Handles real-time communication with Unmute services including:
    - Audio streaming for speech-to-text
    - Text message sending for LLM processing
    - Event-driven response handling
    - Connection management and reconnection
    """

    def __init__(self, base_url: str | None = None, config: dict | None = None):
        """
        Initialize the Unmute WebSocket client.

        Args:
            base_url: Base WebSocket URL for Unmute backend (deprecated, use config)
            config: Service configuration dictionary
        """
        # Support legacy base_url parameter for backward compatibility
        if config:
            self.base_url = config.get("websocket_url", "ws://localhost:8000")
        elif base_url:
            self.base_url = base_url
        else:
            self.base_url = "ws://localhost:8000"
        if self.base_url.endswith("/ws"):
            self.websocket_url = self.base_url
        else:
            self.websocket_url = f"{self.base_url}/ws"
        self.websocket = None
        self.is_connected = False
        self.message_handlers = {}
        self.connection_task = None
        self.listen_task = None
        self.conversation_id = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 2.0

    async def connect(self) -> bool:
        """
        Connect to the Unmute WebSocket server.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Unmute WebSocket at {self.websocket_url}")

            # Connect with timeout and error handling
            self.websocket = await websockets.connect(
                self.websocket_url, ping_interval=20, ping_timeout=10, close_timeout=10
            )

            self.is_connected = True
            self.reconnect_attempts = 0
            logger.success("Connected to Unmute WebSocket server")

            # Start listening for messages
            self.listen_task = asyncio.create_task(self._listen_for_messages())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to Unmute WebSocket: {e}")
            self.is_connected = False
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket server."""
        self.is_connected = False

        # Cancel listening task
        if self.listen_task and not self.listen_task.done():
            self.listen_task.cancel()
            try:
                await self.listen_task
            except asyncio.CancelledError:
                logger.debug("Listen task cancelled during disconnect")

        # Close WebSocket connection
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None

        logger.info("Disconnected from Unmute WebSocket server")

    async def _listen_for_messages(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Received invalid JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except ConnectionClosed:
            logger.warning("WebSocket connection closed by server")
            self.is_connected = False
            await self._attempt_reconnect()
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            await self._attempt_reconnect()
        except Exception as e:
            logger.error(f"Unexpected error in message listener: {e}")
            self.is_connected = False

    async def _handle_message(self, data: dict[str, Any]):
        """Handle incoming WebSocket message."""
        message_type = data.get("type")

        if not message_type:
            logger.warning(f"Message without type received: {data}")
            return

        logger.debug(f"Received message type: {message_type}")

        # Call registered handler if available
        if message_type in self.message_handlers:
            try:
                await self.message_handlers[message_type](data)
            except Exception as e:
                logger.error(f"Error in message handler for {message_type}: {e}")
        else:
            logger.debug(f"No handler registered for message type: {message_type}")

    async def _attempt_reconnect(self):
        """Attempt to reconnect to the WebSocket server."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(
                f"Max reconnection attempts ({self.max_reconnect_attempts}) reached"
            )
            return

        self.reconnect_attempts += 1
        delay = self.reconnect_delay * self.reconnect_attempts

        logger.info(
            f"Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {delay}s"
        )
        await asyncio.sleep(delay)

        success = await self.connect()
        if success:
            logger.success("Reconnection successful")
            # Restart conversation if we had one
            if self.conversation_id:
                await self.start_voice_conversation()
        else:
            await self._attempt_reconnect()

    def on_message(self, message_type: str, handler: Callable[[dict[str, Any]], None]):
        """
        Register a handler for a specific message type.

        Args:
            message_type: Type of message to handle
            handler: Async function to call when message is received
        """
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")

    async def start_voice_conversation(
        self, config: dict[str, Any] | None = None
    ) -> bool:
        """
        Start a voice conversation session with Unmute.

        Args:
            config: Optional conversation configuration

        Returns:
            bool: True if conversation started successfully
        """
        if not self.is_connected:
            logger.error("Cannot start conversation: not connected to Unmute")
            return False

        default_config = {
            "voice_enabled": True,
            "streaming": True,
            "language": "en",
            "model": "default",
        }

        if config:
            default_config.update(config)

        message = {"type": "start_conversation", "config": default_config}

        try:
            await self.websocket.send(json.dumps(message))
            logger.info("Voice conversation session started")
            return True
        except Exception as e:
            logger.error(f"Failed to start voice conversation: {e}")
            return False

    async def send_audio_chunk(
        self, audio_data: bytes, format: str = "wav", sample_rate: int = 16000
    ) -> bool:
        """
        Send audio chunk for STT processing.

        Args:
            audio_data: Raw audio data
            format: Audio format (wav, raw, etc.)
            sample_rate: Audio sample rate

        Returns:
            bool: True if audio sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot send audio: not connected to Unmute")
            return False

        try:
            # Encode audio data as base64
            audio_b64 = base64.b64encode(audio_data).decode("utf-8")

            message = {
                "type": "audio_chunk",
                "data": audio_b64,
                "format": format,
                "sample_rate": sample_rate,
                "conversation_id": self.conversation_id,
            }

            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent audio chunk: {len(audio_data)} bytes")
            return True

        except Exception as e:
            logger.error(f"Failed to send audio chunk: {e}")
            return False

    async def send_text_message(
        self, text: str, context: dict[str, Any] | None = None
    ) -> bool:
        """
        Send text message for LLM processing.

        Args:
            text: Text message to send
            context: Optional context information

        Returns:
            bool: True if message sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot send text: not connected to Unmute")
            return False

        try:
            message = {
                "type": "text_message",
                "text": text,
                "conversation_id": self.conversation_id,
                "context": context or {},
            }

            await self.websocket.send(json.dumps(message))
            logger.debug(f"Sent text message: {text[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to send text message: {e}")
            return False

    async def end_conversation(self) -> bool:
        """
        End the current voice conversation.

        Returns:
            bool: True if conversation ended successfully
        """
        if not self.is_connected:
            return True  # Already disconnected

        try:
            message = {
                "type": "end_conversation",
                "conversation_id": self.conversation_id,
            }

            await self.websocket.send(json.dumps(message))
            self.conversation_id = None
            logger.info("Voice conversation ended")
            return True

        except Exception as e:
            logger.error(f"Failed to end conversation: {e}")
            return False

    async def request_audio_playback(
        self, text: str, voice_config: dict[str, Any] | None = None
    ) -> bool:
        """
        Request TTS audio playback for given text.

        Args:
            text: Text to convert to speech
            voice_config: Optional voice configuration

        Returns:
            bool: True if request sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot request audio playback: not connected to Unmute")
            return False

        try:
            message = {
                "type": "tts_request",
                "text": text,
                "voice_config": voice_config or {},
                "conversation_id": self.conversation_id,
            }

            await self.websocket.send(json.dumps(message))
            logger.debug(f"Requested TTS for text: {text[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to request audio playback: {e}")
            return False

    async def ping(self) -> bool:
        """
        Send a ping to check connection health.

        Returns:
            bool: True if ping successful
        """
        if not self.is_connected or not self.websocket:
            return False

        try:
            await self.websocket.ping()
            return True
        except Exception as e:
            logger.debug(f"Ping failed: {e}")
            return False

    def get_connection_status(self) -> dict[str, Any]:
        """
        Get current connection status information.

        Returns:
            Dict containing connection status details
        """
        return {
            "connected": self.is_connected,
            "websocket_url": self.websocket_url,
            "conversation_id": self.conversation_id,
            "reconnect_attempts": self.reconnect_attempts,
            "max_reconnect_attempts": self.max_reconnect_attempts,
            "handlers_registered": list(self.message_handlers.keys()),
            "websocket_state": str(self.websocket.state) if self.websocket else "None",
        }

    async def cleanup(self):
        """Cleanup resources and close connections."""
        await self.end_conversation()
        await self.disconnect()
