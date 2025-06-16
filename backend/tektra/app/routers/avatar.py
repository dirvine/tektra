"""Avatar control and animation endpoints."""

from typing import Dict, Any, List
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import uuid
import logging

from ..services.lip_sync_service import lip_sync_service
from ..services.tts_service import tts_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ExpressionRequest(BaseModel):
    """Avatar expression request schema."""
    expression: str
    intensity: float = 1.0
    duration: float = 2.0


class GestureRequest(BaseModel):
    """Avatar gesture request schema."""
    gesture: str
    speed: float = 1.0
    repeat: int = 1


class SpeakRequest(BaseModel):
    """Avatar speak request schema."""
    text: str
    lip_sync: bool = True
    expression: str = "neutral"


class AvatarStatus(BaseModel):
    """Avatar status schema."""
    active: bool
    current_expression: str
    current_gesture: str
    speaking: bool
    position: Dict[str, float]


@router.get("/status", response_model=AvatarStatus)
async def get_avatar_status() -> AvatarStatus:
    """Get current avatar status."""
    # TODO: Implement actual avatar status
    return AvatarStatus(
        active=True,
        current_expression="neutral",
        current_gesture="idle",
        speaking=False,
        position={"x": 0.0, "y": 0.0, "z": 0.0}
    )


@router.post("/expression")
async def set_expression(request: ExpressionRequest) -> Dict[str, Any]:
    """Set avatar facial expression."""
    # TODO: Implement actual expression control
    
    valid_expressions = [
        "neutral", "happy", "sad", "angry", "surprised", 
        "confused", "thinking", "excited", "calm"
    ]
    
    if request.expression not in valid_expressions:
        return {
            "error": f"Invalid expression. Valid options: {valid_expressions}",
            "status": "error"
        }
    
    return {
        "status": "success",
        "expression": request.expression,
        "intensity": request.intensity,
        "duration": request.duration,
        "message": f"Avatar expression set to {request.expression}"
    }


@router.post("/gesture")
async def trigger_gesture(request: GestureRequest) -> Dict[str, Any]:
    """Trigger avatar gesture or animation."""
    # TODO: Implement actual gesture control
    
    valid_gestures = [
        "wave", "nod", "shake_head", "point", "thumbs_up",
        "shrug", "clap", "peace_sign", "thinking_pose"
    ]
    
    if request.gesture not in valid_gestures:
        return {
            "error": f"Invalid gesture. Valid options: {valid_gestures}",
            "status": "error"
        }
    
    return {
        "status": "success", 
        "gesture": request.gesture,
        "speed": request.speed,
        "repeat": request.repeat,
        "message": f"Avatar performing {request.gesture} gesture"
    }


@router.post("/speak")
async def make_avatar_speak(request: SpeakRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Make avatar speak with lip sync and expression."""
    try:
        session_id = str(uuid.uuid4())
        
        # Start lip-sync session
        lip_sync_session = await lip_sync_service.start_speech_session(session_id, request.text)
        
        if not lip_sync_session.get('status') == 'started':
            logger.error(f"Failed to start lip-sync session: {lip_sync_session}")
            
        # Generate TTS audio
        try:
            audio_result = await tts_service.synthesize_speech(
                text=request.text,
                voice_name=None,  # Use default voice
                format="wav"
            )
            
            if audio_result.get('success'):
                audio_data = audio_result.get('audio_data', b'')
                
                # Generate lip-sync data
                if request.lip_sync and audio_data:
                    lip_sync_data = await lip_sync_service.complete_speech_session(session_id, audio_data)
                    
                    return {
                        "status": "success",
                        "text": request.text,
                        "lip_sync": request.lip_sync,
                        "expression": request.expression,
                        "estimated_duration": len(request.text) * 0.05,
                        "audio_data": audio_data.hex() if audio_data else None,
                        "lip_sync_data": lip_sync_data if request.lip_sync else None,
                        "session_id": session_id,
                        "message": "Avatar speech generated with lip-sync"
                    }
                else:
                    return {
                        "status": "success",
                        "text": request.text,
                        "lip_sync": False,
                        "expression": request.expression,
                        "estimated_duration": len(request.text) * 0.05,
                        "audio_data": audio_data.hex() if audio_data else None,
                        "session_id": session_id,
                        "message": "Avatar speech generated without lip-sync"
                    }
            else:
                raise Exception(f"TTS synthesis failed: {audio_result.get('error', 'Unknown error')}")
                
        except Exception as tts_error:
            logger.error(f"TTS synthesis error: {tts_error}")
            # Return basic response without audio
            return {
                "status": "partial_success",
                "text": request.text,
                "lip_sync": False,
                "expression": request.expression,
                "estimated_duration": len(request.text) * 0.05,
                "error": f"TTS synthesis failed: {str(tts_error)}",
                "message": "Avatar speech text ready (audio synthesis failed)"
            }
            
    except Exception as e:
        logger.error(f"Avatar speak error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to generate avatar speech"
        }


@router.get("/expressions")
async def list_expressions() -> List[str]:
    """List available avatar expressions."""
    return [
        "neutral", "happy", "sad", "angry", "surprised",
        "confused", "thinking", "excited", "calm", "focused",
        "worried", "relaxed", "determined", "friendly"
    ]


@router.get("/gestures") 
async def list_gestures() -> List[str]:
    """List available avatar gestures."""
    return [
        "wave", "nod", "shake_head", "point", "thumbs_up",
        "shrug", "clap", "peace_sign", "thinking_pose",
        "salute", "bow", "stretch", "dance", "celebrate"
    ]


@router.post("/reset")
async def reset_avatar() -> Dict[str, Any]:
    """Reset avatar to default state."""
    # TODO: Implement actual avatar reset
    
    return {
        "status": "success",
        "expression": "neutral",
        "gesture": "idle",
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "message": "Avatar reset to default state"
    }


@router.get("/lip-sync/capabilities")
async def get_lip_sync_capabilities() -> Dict[str, Any]:
    """Get lip-sync analysis capabilities."""
    return lip_sync_service.analyzer.get_analysis_capabilities()


@router.get("/lip-sync/sessions")
async def get_active_lip_sync_sessions() -> Dict[str, Any]:
    """Get active lip-sync sessions."""
    return {
        "active_sessions": lip_sync_service.get_active_sessions(),
        "total_sessions": len(lip_sync_service.get_active_sessions())
    }


class RealTimeSpeechRequest(BaseModel):
    """Real-time speech request schema."""
    session_id: str
    audio_chunk: str  # Base64 encoded audio
    is_final: bool = False


@router.post("/speak/real-time")
async def process_real_time_speech(request: RealTimeSpeechRequest) -> Dict[str, Any]:
    """Process real-time audio for immediate lip-sync feedback."""
    try:
        import base64
        
        # Decode audio chunk
        audio_bytes = base64.b64decode(request.audio_chunk)
        
        # Process with lip-sync service
        result = await lip_sync_service.process_audio_chunk(request.session_id, audio_bytes)
        
        return {
            "status": "success",
            "session_id": request.session_id,
            "is_final": request.is_final,
            "lip_sync_data": result,
            "timestamp": result.get('timestamp') if result else None
        }
        
    except Exception as e:
        logger.error(f"Real-time speech processing error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "session_id": request.session_id
        }