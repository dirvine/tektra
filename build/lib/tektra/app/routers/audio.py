"""
Audio processing API endpoints.

Handles speech-to-text (STT) and text-to-speech (TTS) operations,
including real-time audio streaming and voice management.
"""

import asyncio
import io
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import tempfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field

from ..services.whisper_service import whisper_service, transcribe_audio_bytes, detect_audio_language
from ..services.phi4_service import phi4_service, transcribe_audio_phi4, detect_audio_language_phi4
from ..services.tts_service import tts_service, synthesize_text_to_speech, get_voice_recommendations
from ..services.vad_service import vad_service, detect_voice_in_audio, preprocess_voice_audio
from ..services.language_service import language_service, detect_language_auto, get_recommended_voice
from ..dependencies import get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])


# Request/Response Models
class TranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    language: Optional[str] = Field(None, description="Language code (auto-detect if None)")
    task: str = Field("transcribe", description="Task type: 'transcribe' or 'translate'")
    temperature: float = Field(0.0, description="Sampling temperature")
    initial_prompt: Optional[str] = Field(None, description="Optional text to guide transcription")


class TranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    text: str = Field(description="Transcribed text")
    language: Optional[str] = Field(description="Detected/specified language")
    confidence: float = Field(description="Transcription confidence score")
    duration: float = Field(description="Audio duration in seconds")
    word_count: int = Field(description="Number of words transcribed")
    segments: List[Dict] = Field(description="Detailed segment information")


class LanguageDetectionResponse(BaseModel):
    """Response model for language detection."""
    detected_language: str = Field(description="Detected language code")
    confidence: float = Field(description="Detection confidence")
    all_languages: List[Dict] = Field(description="All detected languages with probabilities")


class TTSRequest(BaseModel):
    """Request model for text-to-speech synthesis."""
    text: str = Field(description="Text to synthesize", min_length=1, max_length=10000)
    voice: Optional[str] = Field(None, description="Voice ID (auto-select if None)")
    rate: Optional[str] = Field(None, description="Speaking rate (e.g., '+20%', '-10%')")
    pitch: Optional[str] = Field(None, description="Voice pitch (e.g., '+5Hz', '-10Hz')")
    volume: Optional[str] = Field(None, description="Voice volume (e.g., '+20%', '-10%')")
    output_format: str = Field("mp3", description="Output audio format")
    quality: str = Field("medium", description="Audio quality")


class SSMLRequest(BaseModel):
    """Request model for SSML-based speech synthesis."""
    ssml: str = Field(description="SSML markup text", min_length=1)
    voice: Optional[str] = Field(None, description="Voice ID")
    output_format: str = Field("mp3", description="Output audio format")
    quality: str = Field("medium", description="Audio quality")


class VoiceInfo(BaseModel):
    """Voice information model."""
    id: str = Field(description="Voice identifier")
    name: str = Field(description="Friendly voice name")
    gender: str = Field(description="Voice gender")
    locale: str = Field(description="Voice locale")
    language: str = Field(description="Language code")
    voice_type: str = Field(description="Voice type (e.g., Standard, Neural)")


class ServiceInfoResponse(BaseModel):
    """Service information response."""
    whisper_available: bool = Field(description="Whisper STT availability")
    tts_available: bool = Field(description="TTS availability")
    vad_available: bool = Field(description="VAD availability")
    supported_languages: List[str] = Field(description="Supported languages")
    supported_formats: List[str] = Field(description="Supported audio formats")
    model_info: Dict = Field(description="Loaded model information")


class VADRequest(BaseModel):
    """Request model for voice activity detection."""
    aggressiveness: int = Field(2, description="VAD aggressiveness level (0-3)")
    enable_noise_reduction: bool = Field(True, description="Enable noise reduction")
    enable_preprocessing: bool = Field(True, description="Enable audio preprocessing")


class VADResponse(BaseModel):
    """Response model for voice activity detection."""
    has_voice: bool = Field(description="Whether voice activity was detected")
    confidence: float = Field(description="Detection confidence score")
    speech_ratio: float = Field(description="Ratio of speech to total audio")
    duration: float = Field(description="Audio duration in seconds")
    segments: List[Dict] = Field(description="Voice segments with timing")
    method: str = Field(description="Detection method used")


# Audio Transcription Endpoints
@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(description="Audio file to transcribe"),
    language: Optional[str] = Form(None, description="Language code"),
    task: str = Form("transcribe", description="Task type"),
    temperature: float = Form(0.0, description="Sampling temperature"),
    initial_prompt: Optional[str] = Form(None, description="Initial prompt"),
    current_user: User = Depends(get_current_user)
):
    """
    Transcribe uploaded audio file to text.
    
    Supports various audio formats including MP3, WAV, M4A, FLAC, etc.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Validate file size (max 25MB)
    if file.size and file.size > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 25MB)")
    
    try:
        # Read audio data
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Try Phi-4 first, fallback to Whisper
        try:
            if phi4_service.is_loaded:
                result = await phi4_service.transcribe_audio(
                    audio_data=audio_data,
                    language=language,
                    task=task
                )
            else:
                raise RuntimeError("Phi-4 not available")
        except Exception as phi4_error:
            logger.warning(f"Phi-4 transcription failed, using Whisper fallback: {phi4_error}")
            result = await whisper_service.transcribe_audio(
                audio_data=audio_data,
                language=language,
                task=task,
                temperature=temperature,
                initial_prompt=initial_prompt
            )
        
        logger.info(f"Transcribed audio for user {current_user.id}: {len(result['text'])} characters")
        
        return TranscriptionResponse(
            text=result["text"],
            language=result.get("language"),
            confidence=result.get("confidence", 0.0),
            duration=result.get("duration", 0.0),
            word_count=result.get("word_count", 0),
            segments=result.get("segments", [])
        )
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@router.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(
    file: UploadFile = File(description="Audio file for language detection"),
    current_user: User = Depends(get_current_user)
):
    """
    Detect the language of uploaded audio.
    
    Useful for automatic language selection before transcription.
    """
    try:
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Try Phi-4 first, fallback to Whisper
        try:
            if phi4_service.is_loaded:
                result = await phi4_service.detect_language(audio_data)
                # Convert Phi-4 format to expected format
                result = {
                    "detected_language": result["detected_language"],
                    "confidence": result["confidence"],
                    "all_languages": [{"language": result["detected_language"], "probability": result["confidence"]}]
                }
            else:
                raise RuntimeError("Phi-4 not available")
        except Exception as phi4_error:
            logger.warning(f"Phi-4 language detection failed, using Whisper fallback: {phi4_error}")
            result = await whisper_service.detect_language(audio_data)
        
        logger.info(f"Detected language for user {current_user.id}: {result['detected_language']}")
        
        return LanguageDetectionResponse(
            detected_language=result["detected_language"],
            confidence=result["confidence"],
            all_languages=result["all_languages"]
        )
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")


# Text-to-Speech Endpoints
@router.post("/synthesize")
async def synthesize_speech(
    request: TTSRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Synthesize speech from text.
    
    Returns audio data in the specified format.
    """
    try:
        audio_data = await tts_service.synthesize_speech(
            text=request.text,
            voice=request.voice,
            rate=request.rate,
            pitch=request.pitch,
            volume=request.volume,
            output_format=request.output_format,
            quality=request.quality
        )
        
        # Determine content type
        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg"
        }
        content_type = content_type_map.get(request.output_format, "audio/mpeg")
        
        logger.info(f"Synthesized speech for user {current_user.id}: {len(request.text)} chars -> {len(audio_data)} bytes")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.output_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Speech synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


@router.post("/synthesize/ssml")
async def synthesize_ssml(
    request: SSMLRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Synthesize speech from SSML markup.
    
    Allows for advanced speech control using SSML tags.
    """
    try:
        audio_data = await tts_service.synthesize_ssml(
            ssml=request.ssml,
            voice=request.voice,
            output_format=request.output_format,
            quality=request.quality
        )
        
        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg"
        }
        content_type = content_type_map.get(request.output_format, "audio/mpeg")
        
        return Response(
            content=audio_data,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.output_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"SSML synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"SSML synthesis failed: {str(e)}")


@router.post("/synthesize/stream")
async def synthesize_speech_streaming(
    request: TTSRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Synthesize speech with streaming response.
    
    Returns audio data as it's generated for lower latency.
    """
    try:
        content_type_map = {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg"
        }
        content_type = content_type_map.get(request.output_format, "audio/mpeg")
        
        async def generate_audio():
            async for chunk in tts_service.synthesize_streaming(
                text=request.text,
                voice=request.voice,
                rate=request.rate,
                pitch=request.pitch,
                volume=request.volume
            ):
                yield chunk
        
        return StreamingResponse(
            generate_audio(),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.output_format}"
            }
        )
        
    except Exception as e:
        logger.error(f"Streaming synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Streaming synthesis failed: {str(e)}")


# Voice Management Endpoints
@router.get("/voices", response_model=List[VoiceInfo])
async def get_available_voices(
    language: Optional[str] = Query(None, description="Filter by language code"),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of available TTS voices.
    
    Optionally filter by language.
    """
    try:
        if language:
            voices = await tts_service.get_voices_by_language(language)
        else:
            voices = await tts_service.get_available_voices()
        
        return [
            VoiceInfo(
                id=voice["id"],
                name=voice["name"],
                gender=voice["gender"],
                locale=voice["locale"],
                language=voice["language"],
                voice_type=voice["voice_type"]
            )
            for voice in voices
        ]
        
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@router.get("/voices/recommendations", response_model=List[VoiceInfo])
async def get_voice_recommendations_endpoint(
    language: str = Query("en", description="Language code"),
    current_user: User = Depends(get_current_user)
):
    """
    Get recommended voices for a language.
    
    Returns a curated selection of high-quality voices.
    """
    try:
        voices = await get_voice_recommendations(language)
        
        return [
            VoiceInfo(
                id=voice["id"],
                name=voice["name"],
                gender=voice["gender"],
                locale=voice["locale"],
                language=voice["language"],
                voice_type=voice["voice_type"]
            )
            for voice in voices
        ]
        
    except Exception as e:
        logger.error(f"Failed to get voice recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


# Service Information Endpoints
@router.get("/info", response_model=ServiceInfoResponse)
async def get_audio_service_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about available audio services.
    
    Returns service capabilities and status.
    """
    try:
        # Get service information
        whisper_info = await whisper_service.get_model_info()
        phi4_info = await phi4_service.get_model_info()
        tts_info = await tts_service.get_service_info()
        vad_info = await vad_service.get_service_info()
        
        # Determine primary STT service
        primary_stt = "phi4" if phi4_info["is_loaded"] else "whisper"
        stt_available = phi4_info["is_loaded"] or whisper_info["available"]
        
        return ServiceInfoResponse(
            whisper_available=stt_available,
            tts_available=tts_info["available"],
            vad_available=vad_info["webrtc_vad_available"],
            supported_languages=list(set(
                list(phi4_info.get("supported_languages", {}).keys()) +
                list(whisper_info.get("supported_languages", {}).keys()) +
                tts_info.get("languages", [])
            )),
            supported_formats=tts_info.get("supported_formats", []),
            model_info={
                "primary_stt": primary_stt,
                "phi4": phi4_info,
                "whisper": whisper_info,
                "tts": tts_info,
                "vad": vad_info
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get service info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}")


# Model Management Endpoints
@router.post("/whisper/load")
async def load_whisper_model(
    model_name: str = Query("base", description="Whisper model size"),
    current_user: User = Depends(get_current_user)
):
    """
    Load a specific Whisper model.
    
    Available models: tiny, base, small, medium, large
    """
    try:
        success = await whisper_service.load_model(model_name)
        
        if success:
            return {"message": f"Whisper model '{model_name}' loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}'")
            
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/whisper/unload")
async def unload_whisper_model(
    current_user: User = Depends(get_current_user)
):
    """
    Unload the current Whisper model to free memory.
    """
    try:
        await whisper_service.unload_model()
        return {"message": "Whisper model unloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to unload Whisper model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


# Phi-4 Model Management Endpoints
@router.post("/phi4/load")
async def load_phi4_model(
    current_user: User = Depends(get_current_user)
):
    """
    Load the Phi-4 Multimodal model for unified processing.
    
    Phi-4 provides superior speech recognition and chat completion.
    """
    try:
        success = await phi4_service.load_model()
        
        if success:
            return {"message": "Phi-4 Multimodal model loaded successfully", "status": "loaded"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load Phi-4 model")
            
    except Exception as e:
        logger.error(f"Failed to load Phi-4 model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/phi4/unload")
async def unload_phi4_model(
    current_user: User = Depends(get_current_user)
):
    """
    Unload the Phi-4 model to free memory.
    """
    try:
        await phi4_service.unload_model()
        return {"message": "Phi-4 model unloaded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to unload Phi-4 model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@router.get("/phi4/info")
async def get_phi4_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get information about the Phi-4 model status and capabilities.
    """
    try:
        info = await phi4_service.get_model_info()
        return info
        
    except Exception as e:
        logger.error(f"Failed to get Phi-4 info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# Utility Endpoints
@router.post("/create-ssml")
async def create_ssml_markup(
    text: str = Form(description="Text content"),
    voice: Optional[str] = Form(None, description="Voice ID"),
    rate: Optional[str] = Form(None, description="Speaking rate"),
    pitch: Optional[str] = Form(None, description="Voice pitch"),
    volume: Optional[str] = Form(None, description="Voice volume"),
    emphasis: Optional[str] = Form(None, description="Text emphasis"),
    break_time: Optional[str] = Form(None, description="Pause duration"),
    current_user: User = Depends(get_current_user)
):
    """
    Create SSML markup from text and parameters.
    
    Returns SSML that can be used with the SSML synthesis endpoint.
    """
    try:
        ssml = tts_service.create_ssml(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
            emphasis=emphasis,
            break_time=break_time
        )
        
        return {"ssml": ssml}
        
    except Exception as e:
        logger.error(f"SSML creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"SSML creation failed: {str(e)}")


# Voice Activity Detection Endpoints
@router.post("/vad/detect", response_model=VADResponse)
async def detect_voice_activity(
    file: UploadFile = File(description="Audio file for voice activity detection"),
    request: VADRequest = VADRequest(),
    current_user: User = Depends(get_current_user)
):
    """
    Detect voice activity in uploaded audio.
    
    Analyzes audio for speech presence, timing, and confidence.
    """
    try:
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Configure VAD if needed
        if request.aggressiveness != vad_service.aggressiveness:
            vad_service.set_aggressiveness(request.aggressiveness)
        
        # Detect voice activity
        result = await vad_service.detect_voice_activity(audio_data)
        
        logger.info(f"VAD completed for user {current_user.id}: {result['speech_ratio']:.2f} speech ratio")
        
        return VADResponse(
            has_voice=result["has_voice"],
            confidence=result["confidence"],
            speech_ratio=result["speech_ratio"],
            duration=result["duration"],
            segments=result.get("segments", []),
            method=result["method"]
        )
        
    except Exception as e:
        logger.error(f"Voice activity detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"VAD failed: {str(e)}")


@router.post("/vad/preprocess")
async def preprocess_audio(
    file: UploadFile = File(description="Audio file to preprocess"),
    reduce_noise: bool = Query(True, description="Apply noise reduction"),
    normalize: bool = Query(True, description="Normalize audio levels"),
    current_user: User = Depends(get_current_user)
):
    """
    Preprocess audio with noise reduction and filtering.
    
    Returns the processed audio with enhanced quality.
    """
    try:
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Preprocess audio
        processed_audio, sample_rate = await vad_service.preprocess_audio(
            audio_data, 
            reduce_noise=reduce_noise,
            normalize=normalize
        )
        
        # Convert back to WAV format
        import io
        import wave
        
        output_buffer = io.BytesIO()
        with wave.open(output_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Convert float32 to int16
            audio_int16 = (processed_audio * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
        
        processed_data = output_buffer.getvalue()
        
        logger.info(f"Audio preprocessed for user {current_user.id}: {len(audio_data)} -> {len(processed_data)} bytes")
        
        return Response(
            content=processed_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=processed_audio.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")


@router.post("/vad/segments")
async def get_voice_segments(
    file: UploadFile = File(description="Audio file for voice segmentation"),
    min_segment_duration: float = Query(0.5, description="Minimum segment duration in seconds"),
    max_silence_gap: float = Query(0.3, description="Maximum silence gap to bridge segments"),
    current_user: User = Depends(get_current_user)
):
    """
    Extract voice segments from audio with precise timing.
    
    Returns segments of continuous speech with start/end timestamps.
    """
    try:
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Get voice segments
        segments = await vad_service.get_voice_segments(
            audio_data,
            min_segment_duration=min_segment_duration,
            max_silence_gap=max_silence_gap
        )
        
        logger.info(f"Voice segmentation completed for user {current_user.id}: {len(segments)} segments found")
        
        return {
            "segments": segments,
            "total_segments": len(segments),
            "total_speech_duration": sum(seg["end"] - seg["start"] for seg in segments),
            "processing_info": {
                "min_segment_duration": min_segment_duration,
                "max_silence_gap": max_silence_gap
            }
        }
        
    except Exception as e:
        logger.error(f"Voice segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.post("/vad/silence-check")
async def check_silence(
    file: UploadFile = File(description="Audio file to check for silence"),
    threshold_duration: float = Query(0.5, description="Minimum duration to consider for silence"),
    current_user: User = Depends(get_current_user)
):
    """
    Check if audio contains only silence.
    
    Useful for filtering out empty recordings.
    """
    try:
        audio_data = await file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Check for silence
        is_silent = await vad_service.is_silence(audio_data, threshold_duration=threshold_duration)
        
        logger.info(f"Silence check for user {current_user.id}: {'silent' if is_silent else 'contains audio'}")
        
        return {
            "is_silence": is_silent,
            "threshold_duration": threshold_duration,
            "recommendation": "Discard recording" if is_silent else "Process recording"
        }
        
    except Exception as e:
        logger.error(f"Silence check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Silence check failed: {str(e)}")


@router.post("/vad/configure")
async def configure_vad(
    aggressiveness: int = Query(2, description="VAD aggressiveness level (0-3)"),
    current_user: User = Depends(get_current_user)
):
    """
    Configure VAD service parameters.
    
    Updates global VAD settings for all subsequent operations.
    """
    try:
        # Validate aggressiveness level
        if not 0 <= aggressiveness <= 3:
            raise HTTPException(status_code=400, detail="Aggressiveness must be between 0 and 3")
        
        # Update VAD configuration
        vad_service.set_aggressiveness(aggressiveness)
        
        # Get updated service info
        service_info = await vad_service.get_service_info()
        
        logger.info(f"VAD configured by user {current_user.id}: aggressiveness={aggressiveness}")
        
        return {
            "message": "VAD configuration updated",
            "settings": {
                "aggressiveness": aggressiveness,
                "current_config": service_info["settings"]
            }
        }
        
    except Exception as e:
        logger.error(f"VAD configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")


# Multi-Language Support Endpoints
@router.get("/languages")
async def get_supported_languages(
    include_voices: bool = Query(False, description="Include voice information"),
    current_user: User = Depends(get_current_user)
):
    """
    Get list of supported languages with metadata.
    
    Returns comprehensive language information including voice availability.
    """
    try:
        languages = await language_service.get_supported_languages(include_voices=include_voices)
        
        logger.info(f"Languages requested by user {current_user.id}: {len(languages)} languages")
        
        return {
            "languages": languages,
            "total_languages": len(languages),
            "default_language": language_service.default_language,
            "capabilities": {
                "audio_detection": True,
                "text_detection": True,
                "voice_auto_config": True,
                "rtl_support": any(lang.get("is_rtl", False) for lang in languages)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported languages: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get languages: {str(e)}")


@router.post("/languages/detect")
async def detect_language(
    file: Optional[UploadFile] = File(None, description="Audio file for language detection"),
    text: Optional[str] = Form(None, description="Text for language detection"),
    prefer_audio: bool = Form(True, description="Prefer audio detection when both available"),
    current_user: User = Depends(get_current_user)
):
    """
    Detect language from audio or text input.
    
    Supports both audio-based and text-based language detection.
    """
    try:
        audio_data = None
        if file:
            audio_data = await file.read()
            if not audio_data:
                audio_data = None
        
        if not audio_data and not text:
            raise HTTPException(status_code=400, detail="Either audio file or text must be provided")
        
        # Detect language
        result = await detect_language_auto(
            audio_data=audio_data,
            text=text,
            prefer_audio=prefer_audio
        )
        
        logger.info(f"Language detection for user {current_user.id}: {result['detected_language']} ({result['confidence']:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Language detection failed: {str(e)}")


@router.get("/languages/{language}/voices")
async def get_voices_for_language(
    language: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get available voices for a specific language.
    
    Returns all voices that support the specified language.
    """
    try:
        if not language_service.is_language_supported(language):
            raise HTTPException(status_code=404, detail=f"Language '{language}' not supported")
        
        voices = await language_service.get_language_voices(language)
        language_info = language_service.get_language_info(language)
        
        logger.info(f"Voices for {language} requested by user {current_user.id}: {len(voices)} voices")
        
        return {
            "language": language,
            "language_info": language_info,
            "voices": voices,
            "total_voices": len(voices),
            "default_voice": language_info.get("default_voice") if language_info else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get voices for language {language}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get voices: {str(e)}")


@router.post("/languages/recommend-voice")
async def recommend_voice_for_language(
    language: str = Form(description="Language code"),
    gender: Optional[str] = Form(None, description="Preferred gender (Male/Female)"),
    voice_type: str = Form("Neural", description="Voice type preference"),
    current_user: User = Depends(get_current_user)
):
    """
    Get voice recommendation for language and preferences.
    
    Returns the best matching voice based on language and user preferences.
    """
    try:
        if not language_service.is_language_supported(language):
            raise HTTPException(status_code=404, detail=f"Language '{language}' not supported")
        
        recommended_voice = await get_recommended_voice(language, gender, voice_type)
        
        if not recommended_voice:
            raise HTTPException(status_code=404, detail=f"No voices available for language '{language}'")
        
        # Get voice details
        voices = await language_service.get_language_voices(language)
        voice_details = next((v for v in voices if v["id"] == recommended_voice), None)
        
        logger.info(f"Voice recommendation for user {current_user.id}: {language} -> {recommended_voice}")
        
        return {
            "language": language,
            "recommended_voice": recommended_voice,
            "voice_details": voice_details,
            "selection_criteria": {
                "gender_preference": gender,
                "voice_type_preference": voice_type
            }
        }
        
    except Exception as e:
        logger.error(f"Voice recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice recommendation failed: {str(e)}")


@router.post("/languages/auto-configure")
async def auto_configure_voice(
    file: Optional[UploadFile] = File(None, description="Audio file for language detection"),
    text: Optional[str] = Form(None, description="Text for language detection"),
    gender_preference: Optional[str] = Form(None, description="Gender preference"),
    voice_type_preference: str = Form("Neural", description="Voice type preference"),
    current_user: User = Depends(get_current_user)
):
    """
    Automatically configure voice based on detected language and preferences.
    
    Detects language from input and recommends optimal voice configuration.
    """
    try:
        audio_data = None
        if file:
            audio_data = await file.read()
            if not audio_data:
                audio_data = None
        
        if not audio_data and not text:
            raise HTTPException(status_code=400, detail="Either audio file or text must be provided")
        
        # Configure voice automatically
        user_preferences = {
            "gender": gender_preference,
            "voice_type": voice_type_preference
        }
        
        configuration = await configure_voice_for_content(
            audio_data=audio_data,
            text=text,
            user_preferences=user_preferences
        )
        
        logger.info(f"Auto voice configuration for user {current_user.id}: {configuration['language']} -> {configuration['voice_id']}")
        
        return configuration
        
    except Exception as e:
        logger.error(f"Auto voice configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Auto configuration failed: {str(e)}")


@router.get("/languages/service-info")
async def get_language_service_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get language service capabilities and information.
    
    Returns comprehensive information about language support features.
    """
    try:
        service_info = await language_service.get_service_info()
        
        return service_info
        
    except Exception as e:
        logger.error(f"Failed to get language service info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get service info: {str(e)}")