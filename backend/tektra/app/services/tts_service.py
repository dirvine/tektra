"""
Text-to-Speech (TTS) Service.

Provides speech synthesis capabilities using Edge-TTS with support for
multiple voices, languages, and output formats.
"""

import asyncio
import io
import logging
import tempfile
import uuid
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

try:
    import edge_tts

    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

try:
    import pydub
    from pydub import AudioSegment

    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service using Edge-TTS."""

    def __init__(self):
        self.default_voice = "en-US-AriaNeural"
        self.default_rate = "+0%"
        self.default_pitch = "+0Hz"
        self.voices_cache = None
        self.supported_formats = ["mp3", "wav", "ogg"]

        # Voice quality settings
        self.quality_settings = {
            "low": {"bitrate": "32k", "sample_rate": 16000},
            "medium": {"bitrate": "64k", "sample_rate": 22050},
            "high": {"bitrate": "128k", "sample_rate": 44100},
        }

    async def get_available_voices(self) -> List[Dict]:
        """
        Get list of available TTS voices.

        Returns:
            List of voice information dictionaries
        """
        if not EDGE_TTS_AVAILABLE:
            logger.error("Edge-TTS not available. Install with: pip install edge-tts")
            return []

        if self.voices_cache is not None:
            return self.voices_cache

        try:
            voices = await edge_tts.list_voices()

            # Process and categorize voices
            processed_voices = []
            for voice in voices:
                processed_voice = {
                    "id": voice["ShortName"],
                    "name": voice["FriendlyName"],
                    "gender": voice["Gender"],
                    "locale": voice["Locale"],
                    "language": voice["Locale"][:2],
                    "suggested_codec": voice.get(
                        "SuggestedCodec", "audio-24khz-48kbitrate-mono-mp3"
                    ),
                    "status": voice.get("Status", "GA"),
                    "voice_type": voice.get("VoiceType", "Standard"),
                }
                processed_voices.append(processed_voice)

            # Sort by language and name
            processed_voices.sort(key=lambda x: (x["language"], x["name"]))

            self.voices_cache = processed_voices
            logger.info(f"Loaded {len(processed_voices)} TTS voices")
            return processed_voices

        except Exception as e:
            logger.error(f"Failed to get available voices: {e}")
            return []

    async def get_voices_by_language(self, language: str) -> List[Dict]:
        """
        Get voices filtered by language.

        Args:
            language: Language code (e.g., 'en', 'es', 'fr')

        Returns:
            List of voices for the specified language
        """
        all_voices = await self.get_available_voices()
        return [voice for voice in all_voices if voice["language"] == language]

    async def synthesize_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
        output_format: str = "mp3",
        quality: str = "medium",
    ) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice ID (e.g., 'en-US-AriaNeural')
            rate: Speaking rate (e.g., '+20%', '-10%')
            pitch: Voice pitch (e.g., '+5Hz', '-10Hz')
            volume: Voice volume (e.g., '+20%', '-10%')
            output_format: Output audio format ('mp3', 'wav', 'ogg')
            quality: Audio quality ('low', 'medium', 'high')

        Returns:
            Audio data as bytes
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError(
                "Edge-TTS not available. Install with: pip install edge-tts"
            )

        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Use defaults if not specified
        voice = voice or self.default_voice
        rate = rate or self.default_rate
        pitch = pitch or self.default_pitch
        volume = volume or "+0%"

        try:
            # Create TTS communicator
            communicate = edge_tts.Communicate(
                text, voice, rate=rate, pitch=pitch, volume=volume
            )

            # Generate audio
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                raise ValueError("No audio data generated")

            # Convert format if needed
            if output_format != "mp3" or quality != "medium":
                audio_data = await self._convert_audio_format(
                    audio_data, output_format, quality
                )

            logger.info(
                f"Synthesized {len(text)} characters to {len(audio_data)} bytes ({output_format})"
            )
            return audio_data

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            raise

    async def synthesize_ssml(
        self,
        ssml: str,
        voice: Optional[str] = None,
        output_format: str = "mp3",
        quality: str = "medium",
    ) -> bytes:
        """
        Synthesize speech from SSML (Speech Synthesis Markup Language).

        Args:
            ssml: SSML markup text
            voice: Voice ID
            output_format: Output audio format
            quality: Audio quality

        Returns:
            Audio data as bytes
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("Edge-TTS not available")

        voice = voice or self.default_voice

        try:
            # Create TTS communicator with SSML
            communicate = edge_tts.Communicate(ssml, voice)

            # Generate audio
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            if not audio_data:
                raise ValueError("No audio data generated from SSML")

            # Convert format if needed
            if output_format != "mp3" or quality != "medium":
                audio_data = await self._convert_audio_format(
                    audio_data, output_format, quality
                )

            logger.info(
                f"Synthesized SSML to {len(audio_data)} bytes ({output_format})"
            )
            return audio_data

        except Exception as e:
            logger.error(f"SSML synthesis failed: {e}")
            raise

    async def synthesize_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
        output_format: Optional[str] = None,
        quality: str = "medium",
    ) -> Path:
        """
        Synthesize speech and save to file.

        Args:
            text: Text to synthesize
            output_path: Output file path
            voice: Voice ID
            rate: Speaking rate
            pitch: Voice pitch
            volume: Voice volume
            output_format: Output format (auto-detected from file extension if not provided)
            quality: Audio quality

        Returns:
            Path to the generated audio file
        """
        output_path = Path(output_path)

        # Auto-detect format from file extension
        if output_format is None:
            output_format = output_path.suffix.lstrip(".").lower()
            if output_format not in self.supported_formats:
                output_format = "mp3"

        # Generate audio
        audio_data = await self.synthesize_speech(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
            output_format=output_format,
            quality=quality,
        )

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_data)

        logger.info(f"Saved synthesized audio to {output_path}")
        return output_path

    async def synthesize_streaming(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
        chunk_size: int = 1024,
    ):
        """
        Synthesize speech with streaming output.

        Args:
            text: Text to synthesize
            voice: Voice ID
            rate: Speaking rate
            pitch: Voice pitch
            volume: Voice volume
            chunk_size: Size of audio chunks to yield

        Yields:
            Audio data chunks as bytes
        """
        if not EDGE_TTS_AVAILABLE:
            raise RuntimeError("Edge-TTS not available")

        voice = voice or self.default_voice
        rate = rate or self.default_rate
        pitch = pitch or self.default_pitch
        volume = volume or "+0%"

        try:
            # Create TTS communicator
            communicate = edge_tts.Communicate(
                text, voice, rate=rate, pitch=pitch, volume=volume
            )

            # Stream audio chunks
            buffer = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buffer += chunk["data"]

                    # Yield chunks of specified size
                    while len(buffer) >= chunk_size:
                        yield buffer[:chunk_size]
                        buffer = buffer[chunk_size:]

            # Yield remaining data
            if buffer:
                yield buffer

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise

    async def _convert_audio_format(
        self, audio_data: bytes, output_format: str, quality: str
    ) -> bytes:
        """
        Convert audio data to different format/quality.

        Args:
            audio_data: Input audio data (MP3)
            output_format: Target format
            quality: Target quality

        Returns:
            Converted audio data
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Audio processing not available. Returning original data")
            return audio_data

        try:
            # Load audio data
            audio = AudioSegment.from_mp3(io.BytesIO(audio_data))

            # Apply quality settings
            quality_config = self.quality_settings.get(
                quality, self.quality_settings["medium"]
            )

            # Resample if needed
            if audio.frame_rate != quality_config["sample_rate"]:
                audio = audio.set_frame_rate(quality_config["sample_rate"])

            # Export to target format
            output_buffer = io.BytesIO()

            if output_format == "wav":
                audio.export(output_buffer, format="wav")
            elif output_format == "ogg":
                audio.export(output_buffer, format="ogg", codec="libvorbis")
            elif output_format == "mp3":
                audio.export(
                    output_buffer, format="mp3", bitrate=quality_config["bitrate"]
                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            converted_data = output_buffer.getvalue()
            logger.info(
                f"Converted audio: {len(audio_data)} -> {len(converted_data)} bytes ({output_format})"
            )
            return converted_data

        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_data  # Return original on failure

    def create_ssml(
        self,
        text: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        pitch: Optional[str] = None,
        volume: Optional[str] = None,
        emphasis: Optional[str] = None,
        break_time: Optional[str] = None,
    ) -> str:
        """
        Create SSML markup for advanced speech synthesis.

        Args:
            text: Text content
            voice: Voice ID
            rate: Speaking rate
            pitch: Voice pitch
            volume: Voice volume
            emphasis: Text emphasis ('strong', 'moderate', 'reduced')
            break_time: Pause duration (e.g., '500ms', '2s')

        Returns:
            SSML markup string
        """
        ssml_parts = [
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        ]

        # Voice selection
        if voice:
            ssml_parts.append(f'<voice name="{voice}">')

        # Prosody (rate, pitch, volume)
        prosody_attrs = []
        if rate:
            prosody_attrs.append(f'rate="{rate}"')
        if pitch:
            prosody_attrs.append(f'pitch="{pitch}"')
        if volume:
            prosody_attrs.append(f'volume="{volume}"')

        if prosody_attrs:
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')

        # Emphasis
        if emphasis:
            ssml_parts.append(f'<emphasis level="{emphasis}">{text}</emphasis>')
        else:
            ssml_parts.append(text)

        # Break
        if break_time:
            ssml_parts.append(f'<break time="{break_time}"/>')

        # Close tags
        if prosody_attrs:
            ssml_parts.append("</prosody>")
        if voice:
            ssml_parts.append("</voice>")

        ssml_parts.append("</speak>")

        return "".join(ssml_parts)

    async def get_service_info(self) -> Dict:
        """Get information about the TTS service."""
        voices = await self.get_available_voices()

        return {
            "available": EDGE_TTS_AVAILABLE,
            "audio_processing_available": AUDIO_PROCESSING_AVAILABLE,
            "supported_formats": self.supported_formats,
            "quality_levels": list(self.quality_settings.keys()),
            "default_voice": self.default_voice,
            "total_voices": len(voices),
            "languages": list(set(voice["language"] for voice in voices)),
            "voice_types": list(set(voice["voice_type"] for voice in voices)),
        }


# Global service instance
tts_service = TTSService()


# Utility functions for common operations
async def synthesize_text_to_speech(
    text: str, voice: Optional[str] = None, output_format: str = "mp3"
) -> bytes:
    """
    Convenience function to synthesize text to speech.

    Args:
        text: Text to synthesize
        voice: Voice ID (auto-select if None)
        output_format: Output audio format

    Returns:
        Audio data as bytes
    """
    return await tts_service.synthesize_speech(
        text=text, voice=voice, output_format=output_format
    )


async def get_voice_recommendations(language: str = "en") -> List[Dict]:
    """
    Get recommended voices for a language.

    Args:
        language: Language code

    Returns:
        List of recommended voice information
    """
    voices = await tts_service.get_voices_by_language(language)

    # Prioritize neural voices and balanced gender representation
    neural_voices = [v for v in voices if "Neural" in v["id"]]
    if neural_voices:
        voices = neural_voices

    # Get balanced selection
    male_voices = [v for v in voices if v["gender"] == "Male"][:3]
    female_voices = [v for v in voices if v["gender"] == "Female"][:3]

    return male_voices + female_voices


async def create_speech_from_template(
    template: str, variables: Dict[str, str], voice: Optional[str] = None
) -> bytes:
    """
    Create speech from a text template with variables.

    Args:
        template: Text template with {variable} placeholders
        variables: Dictionary of variable values
        voice: Voice ID

    Returns:
        Audio data as bytes
    """
    # Replace template variables
    text = template.format(**variables)

    return await tts_service.synthesize_speech(text=text, voice=voice)
