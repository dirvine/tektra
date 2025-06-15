"""
WebSocket handlers for real-time communication.

Handles real-time chat, audio streaming, and voice processing via WebSockets.
"""

import asyncio
import json
import logging
import uuid
import base64
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState
from pydantic import BaseModel, ValidationError

from ..services.ai_service import ai_service
from ..services.whisper_service import whisper_service
from ..services.phi4_service import phi4_service
from ..services.tts_service import tts_service
from ..services.vad_service import vad_service, detect_voice_in_audio
from ..services.language_service import language_service, detect_language_auto, configure_voice_for_content
from ..models.conversation import Conversation
from ..models.message import Message
from ..database import get_db_session
from ..dependencies import get_current_user_websocket
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()


# WebSocket Message Models
class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""
    type: str
    data: Dict[str, Any]
    message_id: Optional[str] = None
    timestamp: Optional[datetime] = None


class AudioChunk(BaseModel):
    """Audio chunk data model."""
    audio_data: str  # Base64 encoded audio
    format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    chunk_index: int = 0
    is_final: bool = False


class VoiceConfig(BaseModel):
    """Voice configuration model."""
    voice_id: Optional[str] = None
    language: Optional[str] = None
    rate: Optional[str] = None
    pitch: Optional[str] = None
    volume: Optional[str] = None


class ChatMessage(BaseModel):
    """Chat message model."""
    content: str
    conversation_id: Optional[int] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    stream: bool = True


# Connection Manager
class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[int, str] = {}  # user_id -> session_id
        self.audio_sessions: Dict[str, Dict] = {}  # session_id -> audio session data
    
    async def connect(self, websocket: WebSocket, user: User) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Store connection
        self.active_connections[session_id] = websocket
        self.user_sessions[user.id] = session_id
        
        # Initialize audio session
        self.audio_sessions[session_id] = {
            "user_id": user.id,
            "audio_buffer": [],
            "transcription_buffer": "",
            "voice_config": VoiceConfig(),
            "is_speaking": False,
            "conversation_id": None,
            "vad_enabled": True,
            "noise_reduction_enabled": True,
            "silence_timeout": 2.0,  # seconds
            "last_voice_activity": None,
            "auto_start_threshold": 0.7,  # VAD confidence threshold for auto-start
            "preferred_language": None,
            "auto_language_detection": True,
            "voice_auto_config": True,
            "language_confidence_threshold": 0.7
        }
        
        logger.info(f"WebSocket connected: user {user.id}, session {session_id}")
        return session_id
    
    async def disconnect(self, session_id: str):
        """Disconnect a WebSocket connection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            
            # Close connection if still open
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
            
            # Clean up session data
            if session_id in self.audio_sessions:
                user_id = self.audio_sessions[session_id]["user_id"]
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
                del self.audio_sessions[session_id]
            
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: session {session_id}")
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific session."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                await self.disconnect(session_id)
    
    async def broadcast_to_user(self, user_id: int, message: Dict[str, Any]):
        """Send a message to all sessions for a user."""
        if user_id in self.user_sessions:
            session_id = self.user_sessions[user_id]
            await self.send_message(session_id, message)
    
    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information."""
        return self.audio_sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict):
        """Update session data."""
        if session_id in self.audio_sessions:
            self.audio_sessions[session_id].update(updates)


# Global connection manager
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: Optional[str] = None
):
    """
    Main WebSocket endpoint for real-time communication.
    
    Handles chat messages, audio streaming, and voice processing.
    """
    session_id = None
    
    try:
        # Authenticate user
        user = await get_current_user_websocket(websocket, token)
        if not user:
            await websocket.close(code=4001, reason="Authentication required")
            return
        
        # Connect and get session ID
        session_id = await manager.connect(websocket, user)
        
        # Send connection confirmation
        await manager.send_message(session_id, {
            "type": "connection_established",
            "data": {
                "session_id": session_id,
                "user_id": user.id,
                "message": "Connected successfully"
            }
        })
        
        # Main message loop
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Validate message structure
                try:
                    message = WebSocketMessage(**message_data)
                except ValidationError as e:
                    await manager.send_message(session_id, {
                        "type": "error",
                        "data": {"message": f"Invalid message format: {e}"}
                    })
                    continue
                
                # Route message based on type
                await handle_websocket_message(session_id, message, user)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_message(session_id, {
                    "type": "error",
                    "data": {"message": "Invalid JSON format"}
                })
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                await manager.send_message(session_id, {
                    "type": "error",
                    "data": {"message": f"Message processing failed: {str(e)}"}
                })
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        if session_id:
            await manager.disconnect(session_id)


async def handle_websocket_message(session_id: str, message: WebSocketMessage, user: User):
    """Handle incoming WebSocket messages based on type."""
    
    message_type = message.type
    data = message.data
    
    if message_type == "chat_message":
        await handle_chat_message(session_id, data, user)
    
    elif message_type == "audio_chunk":
        await handle_audio_chunk(session_id, data, user)
    
    elif message_type == "audio_start":
        await handle_audio_start(session_id, data, user)
    
    elif message_type == "audio_stop":
        await handle_audio_stop(session_id, data, user)
    
    elif message_type == "voice_config":
        await handle_voice_config(session_id, data, user)
    
    elif message_type == "transcription_request":
        await handle_transcription_request(session_id, data, user)
    
    elif message_type == "tts_request":
        await handle_tts_request(session_id, data, user)
    
    elif message_type == "vad_config":
        await handle_vad_config(session_id, data, user)
    
    elif message_type == "language_detect":
        await handle_language_detection(session_id, data, user)
    
    elif message_type == "language_config":
        await handle_language_config(session_id, data, user)
    
    elif message_type == "ping":
        await manager.send_message(session_id, {
            "type": "pong",
            "data": {"timestamp": datetime.utcnow().isoformat()}
        })
    
    else:
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Unknown message type: {message_type}"}
        })


async def handle_chat_message(session_id: str, data: Dict, user: User):
    """Handle text chat messages with AI streaming responses."""
    try:
        chat_msg = ChatMessage(**data)
        
        # Get conversation or create new one
        async with get_db_session() as db:
            if chat_msg.conversation_id:
                conversation = await db.get(Conversation, chat_msg.conversation_id)
                if not conversation or conversation.user_id != user.id:
                    raise HTTPException(status_code=404, detail="Conversation not found")
            else:
                # Create new conversation
                conversation = Conversation(
                    user_id=user.id,
                    title=chat_msg.content[:50] + "..." if len(chat_msg.content) > 50 else chat_msg.content,
                    category="general"
                )
                db.add(conversation)
                await db.commit()
                await db.refresh(conversation)
        
        # Update session with conversation ID
        manager.update_session(session_id, {"conversation_id": conversation.id})
        
        # Send message acknowledgment
        await manager.send_message(session_id, {
            "type": "message_received",
            "data": {
                "conversation_id": conversation.id,
                "message": "Processing your message..."
            }
        })
        
        # Stream AI response
        async for chunk in ai_service.stream_chat_completion(
            messages=[{"role": "user", "content": chat_msg.content}],
            model=chat_msg.model,
            temperature=chat_msg.temperature
        ):
            await manager.send_message(session_id, {
                "type": "ai_response_chunk",
                "data": {
                    "conversation_id": conversation.id,
                    "content": chunk,
                    "finished": False
                }
            })
        
        # Send completion signal
        await manager.send_message(session_id, {
            "type": "ai_response_complete",
            "data": {
                "conversation_id": conversation.id,
                "finished": True
            }
        })
        
    except Exception as e:
        logger.error(f"Chat message handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Chat processing failed: {str(e)}"}
        })


async def handle_audio_chunk(session_id: str, data: Dict, user: User):
    """Handle incoming audio chunks for real-time transcription."""
    try:
        audio_chunk = AudioChunk(**data)
        session_info = manager.get_session_info(session_id)
        
        if not session_info:
            raise ValueError("Session not found")
        
        # Decode audio data
        try:
            audio_bytes = base64.b64decode(audio_chunk.audio_data)
        except Exception as e:
            raise ValueError(f"Invalid audio data: {e}")
        
        # Add to audio buffer
        session_info["audio_buffer"].append({
            "data": audio_bytes,
            "chunk_index": audio_chunk.chunk_index,
            "timestamp": datetime.utcnow()
        })
        
        # Voice Activity Detection for chunk
        vad_result = None
        if session_info.get("vad_enabled", True) and len(audio_bytes) > 512:
            try:
                vad_result = await vad_service.detect_voice_activity(audio_bytes)
                
                # Send VAD feedback
                await manager.send_message(session_id, {
                    "type": "vad_result",
                    "data": {
                        "has_voice": vad_result["has_voice"],
                        "confidence": vad_result["confidence"],
                        "speech_ratio": vad_result["speech_ratio"],
                        "chunk_index": audio_chunk.chunk_index
                    }
                })
                
                # Update last voice activity time
                if vad_result["has_voice"]:
                    session_info["last_voice_activity"] = datetime.utcnow()
                
            except Exception as e:
                logger.warning(f"VAD processing failed: {e}")
        
        # Process chunk for real-time transcription if voice detected or VAD disabled
        should_transcribe = (
            not session_info.get("vad_enabled", True) or  # VAD disabled
            (vad_result and vad_result["has_voice"]) or   # Voice detected
            len(audio_bytes) > 2048                       # Large chunk (likely contains speech)
        )
        
        if should_transcribe and len(audio_bytes) > 1024:
            try:
                # Preprocess audio if noise reduction enabled
                if session_info.get("noise_reduction_enabled", True):
                    processed_audio, _ = await vad_service.preprocess_audio(
                        audio_bytes, reduce_noise=True
                    )
                    # Convert back to bytes for transcription
                    processed_bytes = (processed_audio * 32767).astype(np.int16).tobytes()
                else:
                    processed_bytes = audio_bytes
                
                # Transcribe chunk - use Phi-4 first, fallback to Whisper
                try:
                    if phi4_service.is_loaded:
                        result = await phi4_service.transcribe_audio(
                            audio_data=processed_bytes,
                            language=session_info["voice_config"].language,
                            task="transcribe"
                        )
                    else:
                        raise RuntimeError("Phi-4 not available")
                except Exception:
                    result = await whisper_service.transcribe_audio(
                        audio_data=processed_bytes,
                        language=session_info["voice_config"].language,
                        task="transcribe"
                    )
                
                # Send partial transcription
                if result["text"].strip():
                    await manager.send_message(session_id, {
                        "type": "transcription_partial",
                        "data": {
                            "text": result["text"],
                            "chunk_index": audio_chunk.chunk_index,
                            "confidence": result.get("confidence", 0.0),
                            "is_final": audio_chunk.is_final,
                            "vad_confidence": vad_result["confidence"] if vad_result else 1.0
                        }
                    })
                
            except Exception as e:
                logger.warning(f"Real-time transcription failed: {e}")
        
        # If final chunk, process complete audio
        if audio_chunk.is_final:
            await process_complete_audio(session_id, user)
        
    except Exception as e:
        logger.error(f"Audio chunk handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Audio processing failed: {str(e)}"}
        })


async def handle_audio_start(session_id: str, data: Dict, user: User):
    """Handle audio recording start."""
    try:
        session_info = manager.get_session_info(session_id)
        if session_info:
            # Clear previous audio buffer
            session_info["audio_buffer"] = []
            session_info["transcription_buffer"] = ""
            session_info["is_speaking"] = True
            
            # Update voice config if provided
            if "voice_config" in data:
                voice_config = VoiceConfig(**data["voice_config"])
                session_info["voice_config"] = voice_config
        
        await manager.send_message(session_id, {
            "type": "audio_recording_started",
            "data": {"message": "Audio recording started", "session_id": session_id}
        })
        
    except Exception as e:
        logger.error(f"Audio start handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Failed to start audio recording: {str(e)}"}
        })


async def handle_audio_stop(session_id: str, data: Dict, user: User):
    """Handle audio recording stop."""
    try:
        session_info = manager.get_session_info(session_id)
        if session_info:
            session_info["is_speaking"] = False
        
        # Process complete audio
        await process_complete_audio(session_id, user)
        
        await manager.send_message(session_id, {
            "type": "audio_recording_stopped",
            "data": {"message": "Audio recording stopped", "session_id": session_id}
        })
        
    except Exception as e:
        logger.error(f"Audio stop handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Failed to stop audio recording: {str(e)}"}
        })


async def process_complete_audio(session_id: str, user: User):
    """Process complete audio buffer for final transcription and AI response."""
    try:
        session_info = manager.get_session_info(session_id)
        if not session_info or not session_info["audio_buffer"]:
            return
        
        # Combine audio chunks
        combined_audio = b""
        for chunk in session_info["audio_buffer"]:
            combined_audio += chunk["data"]
        
        if len(combined_audio) < 1024:  # Skip very short audio
            return
        
        # Check if combined audio contains voice activity
        vad_result = None
        if session_info.get("vad_enabled", True):
            try:
                vad_result = await vad_service.detect_voice_activity(combined_audio)
                
                # Send final VAD result
                await manager.send_message(session_id, {
                    "type": "vad_final",
                    "data": {
                        "has_voice": vad_result["has_voice"],
                        "confidence": vad_result["confidence"],
                        "speech_ratio": vad_result["speech_ratio"],
                        "duration": vad_result["duration"],
                        "segments": vad_result.get("segments", [])
                    }
                })
                
                # Skip transcription if no voice detected and VAD is enabled
                if not vad_result["has_voice"]:
                    await manager.send_message(session_id, {
                        "type": "no_voice_detected",
                        "data": {
                            "message": "No speech detected in audio",
                            "vad_confidence": vad_result["confidence"]
                        }
                    })
                    return
                    
            except Exception as e:
                logger.warning(f"Final VAD processing failed: {e}")
        
        # Preprocess audio for better transcription
        try:
            if session_info.get("noise_reduction_enabled", True):
                processed_audio, _ = await vad_service.preprocess_audio(
                    combined_audio, reduce_noise=True, normalize=True
                )
                # Convert back to bytes for transcription
                processed_combined = (processed_audio * 32767).astype(np.int16).tobytes()
            else:
                processed_combined = combined_audio
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            processed_combined = combined_audio
        
        # Auto-detect language if enabled
        target_language = session_info["voice_config"].language
        if session_info.get("auto_language_detection", True):
            try:
                lang_detection = await language_service.detect_language_from_audio(processed_combined)
                
                if lang_detection["is_reliable"]:
                    detected_lang = lang_detection["detected_language"]
                    
                    # Send language detection result
                    await manager.send_message(session_id, {
                        "type": "language_detected",
                        "data": {
                            "detected_language": detected_lang,
                            "confidence": lang_detection["confidence"],
                            "language_name": lang_detection.get("language_name", detected_lang),
                            "native_name": lang_detection.get("native_name", detected_lang)
                        }
                    })
                    
                    # Update target language
                    target_language = detected_lang
                    
                    # Auto-configure voice if enabled
                    if session_info.get("voice_auto_config", True):
                        voice_config = await language_service.auto_configure_voice(
                            detected_lang,
                            lang_detection["confidence"],
                            {
                                "gender": getattr(session_info["voice_config"], "gender", None),
                                "voice_type": "Neural"
                            }
                        )
                        
                        # Update session voice config
                        session_info["voice_config"].voice_id = voice_config["voice_id"]
                        session_info["voice_config"].language = voice_config["language"]
                        
                        # Notify client of voice change
                        await manager.send_message(session_id, {
                            "type": "voice_auto_configured",
                            "data": voice_config
                        })
                        
            except Exception as e:
                logger.warning(f"Language auto-detection failed: {e}")
        
        # Final transcription with appropriate language - use Phi-4 first, fallback to Whisper
        try:
            if phi4_service.is_loaded:
                result = await phi4_service.transcribe_audio(
                    audio_data=processed_combined,
                    language=target_language,
                    task="transcribe"
                )
            else:
                raise RuntimeError("Phi-4 not available")
        except Exception as phi4_error:
            logger.warning(f"Phi-4 final transcription failed, using Whisper: {phi4_error}")
            result = await whisper_service.transcribe_audio(
                audio_data=processed_combined,
                language=target_language,
                task="transcribe"
            )
        
        transcribed_text = result["text"].strip()
        
        if not transcribed_text:
            return
        
        # Send final transcription
        await manager.send_message(session_id, {
            "type": "transcription_final",
            "data": {
                "text": transcribed_text,
                "language": result.get("language"),
                "confidence": result.get("confidence", 0.0),
                "duration": result.get("duration", 0.0)
            }
        })
        
        # Get AI response
        conversation_id = session_info.get("conversation_id")
        
        # Stream AI response
        await manager.send_message(session_id, {
            "type": "ai_processing",
            "data": {"message": "Processing your voice message..."}
        })
        
        full_response = ""
        async for chunk in ai_service.stream_chat_completion(
            messages=[{"role": "user", "content": transcribed_text}]
        ):
            full_response += chunk
            await manager.send_message(session_id, {
                "type": "ai_response_chunk",
                "data": {
                    "conversation_id": conversation_id,
                    "content": chunk,
                    "finished": False
                }
            })
        
        # Generate TTS for AI response
        voice_config = session_info["voice_config"]
        try:
            audio_data = await tts_service.synthesize_speech(
                text=full_response,
                voice=voice_config.voice_id,
                rate=voice_config.rate,
                pitch=voice_config.pitch,
                volume=voice_config.volume
            )
            
            # Send TTS audio
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            await manager.send_message(session_id, {
                "type": "tts_audio",
                "data": {
                    "audio_data": audio_b64,
                    "format": "mp3",
                    "text": full_response
                }
            })
            
        except Exception as e:
            logger.warning(f"TTS generation failed: {e}")
        
        # Send completion
        await manager.send_message(session_id, {
            "type": "voice_interaction_complete",
            "data": {
                "transcription": transcribed_text,
                "response": full_response,
                "conversation_id": conversation_id
            }
        })
        
    except Exception as e:
        logger.error(f"Complete audio processing error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Audio processing failed: {str(e)}"}
        })


async def handle_voice_config(session_id: str, data: Dict, user: User):
    """Handle voice configuration updates."""
    try:
        voice_config = VoiceConfig(**data)
        session_info = manager.get_session_info(session_id)
        
        if session_info:
            session_info["voice_config"] = voice_config
        
        await manager.send_message(session_id, {
            "type": "voice_config_updated",
            "data": {"message": "Voice configuration updated", "config": voice_config.dict()}
        })
        
    except Exception as e:
        logger.error(f"Voice config handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Voice config update failed: {str(e)}"}
        })


async def handle_transcription_request(session_id: str, data: Dict, user: User):
    """Handle standalone transcription requests."""
    try:
        if "audio_data" not in data:
            raise ValueError("Audio data required")
        
        # Decode audio
        audio_bytes = base64.b64decode(data["audio_data"])
        
        # Transcribe - use Phi-4 first, fallback to Whisper
        try:
            if phi4_service.is_loaded:
                result = await phi4_service.transcribe_audio(
                    audio_data=audio_bytes,
                    language=data.get("language"),
                    task=data.get("task", "transcribe")
                )
            else:
                raise RuntimeError("Phi-4 not available")
        except Exception:
            result = await whisper_service.transcribe_audio(
                audio_data=audio_bytes,
                language=data.get("language"),
                task=data.get("task", "transcribe")
            )
        
        await manager.send_message(session_id, {
            "type": "transcription_result",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Transcription request error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Transcription failed: {str(e)}"}
        })


async def handle_tts_request(session_id: str, data: Dict, user: User):
    """Handle standalone TTS requests."""
    try:
        if "text" not in data:
            raise ValueError("Text required")
        
        # Generate speech
        audio_data = await tts_service.synthesize_speech(
            text=data["text"],
            voice=data.get("voice"),
            rate=data.get("rate"),
            pitch=data.get("pitch"),
            volume=data.get("volume"),
            output_format=data.get("format", "mp3"),
            quality=data.get("quality", "medium")
        )
        
        # Send audio
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        await manager.send_message(session_id, {
            "type": "tts_result",
            "data": {
                "audio_data": audio_b64,
                "format": data.get("format", "mp3"),
                "text": data["text"]
            }
        })
        
    except Exception as e:
        logger.error(f"TTS request error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"TTS generation failed: {str(e)}"}
        })


async def handle_vad_config(session_id: str, data: Dict, user: User):
    """Handle VAD configuration updates."""
    try:
        session_info = manager.get_session_info(session_id)
        
        if not session_info:
            raise ValueError("Session not found")
        
        # Update VAD settings
        if "vad_enabled" in data:
            session_info["vad_enabled"] = bool(data["vad_enabled"])
        
        if "noise_reduction_enabled" in data:
            session_info["noise_reduction_enabled"] = bool(data["noise_reduction_enabled"])
        
        if "silence_timeout" in data:
            timeout = float(data["silence_timeout"])
            if 0.5 <= timeout <= 10.0:  # Reasonable bounds
                session_info["silence_timeout"] = timeout
        
        if "auto_start_threshold" in data:
            threshold = float(data["auto_start_threshold"])
            if 0.0 <= threshold <= 1.0:
                session_info["auto_start_threshold"] = threshold
        
        # Update global VAD service settings if provided
        if "aggressiveness" in data:
            level = int(data["aggressiveness"])
            if 0 <= level <= 3:
                vad_service.set_aggressiveness(level)
        
        await manager.send_message(session_id, {
            "type": "vad_config_updated",
            "data": {
                "message": "VAD configuration updated",
                "settings": {
                    "vad_enabled": session_info["vad_enabled"],
                    "noise_reduction_enabled": session_info["noise_reduction_enabled"],
                    "silence_timeout": session_info["silence_timeout"],
                    "auto_start_threshold": session_info["auto_start_threshold"],
                    "aggressiveness": vad_service.aggressiveness
                }
            }
        })
        
    except Exception as e:
        logger.error(f"VAD config handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"VAD config update failed: {str(e)}"}
        })


async def handle_language_detection(session_id: str, data: Dict, user: User):
    """Handle language detection requests."""
    try:
        # Detect language from text or audio
        text = data.get("text")
        audio_data_b64 = data.get("audio_data")
        
        if audio_data_b64:
            # Decode audio and detect language
            audio_bytes = base64.b64decode(audio_data_b64)
            result = await language_service.detect_language_from_audio(audio_bytes)
        elif text:
            # Detect language from text
            result = await language_service.detect_language_from_text(text)
        else:
            raise ValueError("Either text or audio_data required")
        
        await manager.send_message(session_id, {
            "type": "language_detection_result",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Language detection failed: {str(e)}"}
        })


async def handle_language_config(session_id: str, data: Dict, user: User):
    """Handle language configuration updates."""
    try:
        session_info = manager.get_session_info(session_id)
        
        if not session_info:
            raise ValueError("Session not found")
        
        # Update language settings
        if "preferred_language" in data:
            preferred_lang = data["preferred_language"]
            if language_service.is_language_supported(preferred_lang):
                session_info["preferred_language"] = preferred_lang
                
                # Update voice config language
                session_info["voice_config"].language = preferred_lang
                
                # Auto-configure voice for new language
                if session_info.get("voice_auto_config", True):
                    voice_config = await language_service.auto_configure_voice(
                        preferred_lang,
                        1.0,  # High confidence since user-selected
                        {
                            "gender": getattr(session_info["voice_config"], "gender", None),
                            "voice_type": "Neural"
                        }
                    )
                    session_info["voice_config"].voice_id = voice_config["voice_id"]
        
        if "auto_language_detection" in data:
            session_info["auto_language_detection"] = bool(data["auto_language_detection"])
        
        if "voice_auto_config" in data:
            session_info["voice_auto_config"] = bool(data["voice_auto_config"])
        
        if "language_confidence_threshold" in data:
            threshold = float(data["language_confidence_threshold"])
            if 0.0 <= threshold <= 1.0:
                session_info["language_confidence_threshold"] = threshold
        
        # Get available languages for response
        supported_languages = await language_service.get_supported_languages()
        
        await manager.send_message(session_id, {
            "type": "language_config_updated",
            "data": {
                "message": "Language configuration updated",
                "settings": {
                    "preferred_language": session_info.get("preferred_language"),
                    "auto_language_detection": session_info["auto_language_detection"],
                    "voice_auto_config": session_info["voice_auto_config"],
                    "language_confidence_threshold": session_info["language_confidence_threshold"],
                    "current_voice": session_info["voice_config"].voice_id
                },
                "supported_languages": supported_languages[:10]  # Top 10 for brevity
            }
        })
        
    except Exception as e:
        logger.error(f"Language config handling error: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "data": {"message": f"Language config update failed: {str(e)}"}
        })


# WebSocket utility functions
async def broadcast_system_message(message: str, message_type: str = "system"):
    """Broadcast a system message to all connected users."""
    broadcast_data = {
        "type": message_type,
        "data": {
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    for session_id in manager.active_connections:
        await manager.send_message(session_id, broadcast_data)


async def send_user_notification(user_id: int, message: str, notification_type: str = "info"):
    """Send a notification to a specific user."""
    notification_data = {
        "type": "notification",
        "data": {
            "type": notification_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
    }
    
    await manager.broadcast_to_user(user_id, notification_data)


def get_connection_stats() -> Dict[str, Any]:
    """Get current connection statistics."""
    return {
        "total_connections": len(manager.active_connections),
        "active_users": len(manager.user_sessions),
        "audio_sessions": len([s for s in manager.audio_sessions.values() if s.get("is_speaking")]),
        "session_details": [
            {
                "session_id": session_id[:8] + "...",
                "user_id": session_data["user_id"],
                "is_speaking": session_data["is_speaking"],
                "conversation_id": session_data.get("conversation_id")
            }
            for session_id, session_data in manager.audio_sessions.items()
        ]
    }


# Add stats endpoint for monitoring
@router.get("/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return get_connection_stats()


# HTTP endpoints for WebSocket management
@router.get("/connections")
async def get_connections() -> Dict[str, Any]:
    """Get current WebSocket connection status."""
    return get_connection_stats()


@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Broadcast message to all connected clients."""
    await broadcast_system_message(message.get("message", ""), message.get("type", "system"))
    return {
        "status": "success",
        "message": "Message broadcasted to all connections",
        "connections_reached": len(manager.active_connections)
    }


@router.post("/send/{user_id}")
async def send_user_message(user_id: int, message: Dict[str, Any]) -> Dict[str, Any]:
    """Send message to specific user."""
    if user_id in manager.user_sessions:
        await send_user_notification(user_id, message.get("message", ""), message.get("type", "info"))
        return {
            "status": "success",
            "message": f"Message sent to user {user_id}"
        }
    else:
        return {
            "status": "error",
            "message": f"User {user_id} not connected"
        }