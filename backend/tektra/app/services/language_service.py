"""
Multi-Language Support Service.

Provides comprehensive language detection, voice matching, and cross-language
processing capabilities for voice interactions.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

try:
    import langdetect
    from langdetect import LangDetectException, detect, detect_langs

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from ..config import settings
from ..services.tts_service import tts_service
from ..services.whisper_service import whisper_service

logger = logging.getLogger(__name__)


class LanguageService:
    """Multi-language support and processing service."""

    def __init__(self):
        # Language mappings and configurations
        self.supported_languages = self._load_language_config()
        self.default_language = "en"
        self.fallback_language = "en"

        # Voice-to-language mappings
        self.voice_language_map = {}
        self.language_voice_map = {}

        # Language detection settings
        self.text_detection_threshold = 0.7  # Minimum confidence for text detection
        self.audio_detection_enabled = True
        self.text_detection_enabled = True

        # Translation settings (for future expansion)
        self.translation_enabled = False
        self.auto_translate = False

        # Initialize language mappings (lazy initialization)
        self._voice_mappings_initialized = False

    async def _ensure_initialized(self):
        """Ensure voice mappings are initialized."""
        if not self._voice_mappings_initialized:
            await self._initialize_voice_mappings()
            self._voice_mappings_initialized = True

    def _load_language_config(self) -> Dict[str, Dict]:
        """Load comprehensive language configuration."""
        return {
            # Major languages with full support
            "en": {
                "name": "English",
                "native_name": "English",
                "iso_639_1": "en",
                "iso_639_3": "eng",
                "whisper_code": "en",
                "tts_locales": ["en-US", "en-GB", "en-AU", "en-CA"],
                "default_voice": "en-US-AriaNeural",
                "rtl": False,
                "regions": ["US", "GB", "AU", "CA", "IN"],
                "priority": 1,
            },
            "es": {
                "name": "Spanish",
                "native_name": "Español",
                "iso_639_1": "es",
                "iso_639_3": "spa",
                "whisper_code": "es",
                "tts_locales": ["es-ES", "es-MX", "es-AR", "es-CO"],
                "default_voice": "es-ES-ElviraNeural",
                "rtl": False,
                "regions": ["ES", "MX", "AR", "CO", "CL", "PE"],
                "priority": 2,
            },
            "fr": {
                "name": "French",
                "native_name": "Français",
                "iso_639_1": "fr",
                "iso_639_3": "fra",
                "whisper_code": "fr",
                "tts_locales": ["fr-FR", "fr-CA", "fr-BE", "fr-CH"],
                "default_voice": "fr-FR-DeniseNeural",
                "rtl": False,
                "regions": ["FR", "CA", "BE", "CH"],
                "priority": 3,
            },
            "de": {
                "name": "German",
                "native_name": "Deutsch",
                "iso_639_1": "de",
                "iso_639_3": "deu",
                "whisper_code": "de",
                "tts_locales": ["de-DE", "de-AT", "de-CH"],
                "default_voice": "de-DE-KatjaNeural",
                "rtl": False,
                "regions": ["DE", "AT", "CH"],
                "priority": 4,
            },
            "it": {
                "name": "Italian",
                "native_name": "Italiano",
                "iso_639_1": "it",
                "iso_639_3": "ita",
                "whisper_code": "it",
                "tts_locales": ["it-IT"],
                "default_voice": "it-IT-ElsaNeural",
                "rtl": False,
                "regions": ["IT"],
                "priority": 5,
            },
            "pt": {
                "name": "Portuguese",
                "native_name": "Português",
                "iso_639_1": "pt",
                "iso_639_3": "por",
                "whisper_code": "pt",
                "tts_locales": ["pt-PT", "pt-BR"],
                "default_voice": "pt-PT-RaquelNeural",
                "rtl": False,
                "regions": ["PT", "BR"],
                "priority": 6,
            },
            "ru": {
                "name": "Russian",
                "native_name": "Русский",
                "iso_639_1": "ru",
                "iso_639_3": "rus",
                "whisper_code": "ru",
                "tts_locales": ["ru-RU"],
                "default_voice": "ru-RU-SvetlanaNeural",
                "rtl": False,
                "regions": ["RU"],
                "priority": 7,
            },
            "ja": {
                "name": "Japanese",
                "native_name": "日本語",
                "iso_639_1": "ja",
                "iso_639_3": "jpn",
                "whisper_code": "ja",
                "tts_locales": ["ja-JP"],
                "default_voice": "ja-JP-NanamiNeural",
                "rtl": False,
                "regions": ["JP"],
                "priority": 8,
            },
            "ko": {
                "name": "Korean",
                "native_name": "한국어",
                "iso_639_1": "ko",
                "iso_639_3": "kor",
                "whisper_code": "ko",
                "tts_locales": ["ko-KR"],
                "default_voice": "ko-KR-SunHiNeural",
                "rtl": False,
                "regions": ["KR"],
                "priority": 9,
            },
            "zh": {
                "name": "Chinese",
                "native_name": "中文",
                "iso_639_1": "zh",
                "iso_639_3": "zho",
                "whisper_code": "zh",
                "tts_locales": ["zh-CN", "zh-TW", "zh-HK"],
                "default_voice": "zh-CN-XiaoxiaoNeural",
                "rtl": False,
                "regions": ["CN", "TW", "HK"],
                "priority": 10,
            },
            "ar": {
                "name": "Arabic",
                "native_name": "العربية",
                "iso_639_1": "ar",
                "iso_639_3": "ara",
                "whisper_code": "ar",
                "tts_locales": ["ar-SA", "ar-EG", "ar-AE"],
                "default_voice": "ar-SA-ZariyahNeural",
                "rtl": True,
                "regions": ["SA", "EG", "AE", "MA", "DZ"],
                "priority": 11,
            },
            "hi": {
                "name": "Hindi",
                "native_name": "हिन्दी",
                "iso_639_1": "hi",
                "iso_639_3": "hin",
                "whisper_code": "hi",
                "tts_locales": ["hi-IN"],
                "default_voice": "hi-IN-SwaraNeural",
                "rtl": False,
                "regions": ["IN"],
                "priority": 12,
            },
            "nl": {
                "name": "Dutch",
                "native_name": "Nederlands",
                "iso_639_1": "nl",
                "iso_639_3": "nld",
                "whisper_code": "nl",
                "tts_locales": ["nl-NL", "nl-BE"],
                "default_voice": "nl-NL-ColetteNeural",
                "rtl": False,
                "regions": ["NL", "BE"],
                "priority": 13,
            },
            "pl": {
                "name": "Polish",
                "native_name": "Polski",
                "iso_639_1": "pl",
                "iso_639_3": "pol",
                "whisper_code": "pl",
                "tts_locales": ["pl-PL"],
                "default_voice": "pl-PL-ZofiaNeural",
                "rtl": False,
                "regions": ["PL"],
                "priority": 14,
            },
            "tr": {
                "name": "Turkish",
                "native_name": "Türkçe",
                "iso_639_1": "tr",
                "iso_639_3": "tur",
                "whisper_code": "tr",
                "tts_locales": ["tr-TR"],
                "default_voice": "tr-TR-EmelNeural",
                "rtl": False,
                "regions": ["TR"],
                "priority": 15,
            },
            "sv": {
                "name": "Swedish",
                "native_name": "Svenska",
                "iso_639_1": "sv",
                "iso_639_3": "swe",
                "whisper_code": "sv",
                "tts_locales": ["sv-SE"],
                "default_voice": "sv-SE-SofieNeural",
                "rtl": False,
                "regions": ["SE"],
                "priority": 16,
            },
            "da": {
                "name": "Danish",
                "native_name": "Dansk",
                "iso_639_1": "da",
                "iso_639_3": "dan",
                "whisper_code": "da",
                "tts_locales": ["da-DK"],
                "default_voice": "da-DK-ChristelNeural",
                "rtl": False,
                "regions": ["DK"],
                "priority": 17,
            },
            "no": {
                "name": "Norwegian",
                "native_name": "Norsk",
                "iso_639_1": "no",
                "iso_639_3": "nor",
                "whisper_code": "no",
                "tts_locales": ["nb-NO"],
                "default_voice": "nb-NO-PernilleNeural",
                "rtl": False,
                "regions": ["NO"],
                "priority": 18,
            },
            "fi": {
                "name": "Finnish",
                "native_name": "Suomi",
                "iso_639_1": "fi",
                "iso_639_3": "fin",
                "whisper_code": "fi",
                "tts_locales": ["fi-FI"],
                "default_voice": "fi-FI-SelmaNeural",
                "rtl": False,
                "regions": ["FI"],
                "priority": 19,
            },
        }

    async def _initialize_voice_mappings(self):
        """Initialize voice-to-language mappings from TTS service."""
        try:
            # Get available voices from TTS service
            voices = await tts_service.get_available_voices()

            # Create mappings
            for voice in voices:
                voice_id = voice["id"]
                locale = voice["locale"]
                language = voice["language"]

                # Map voice to language
                self.voice_language_map[voice_id] = language

                # Map language to voices
                if language not in self.language_voice_map:
                    self.language_voice_map[language] = []
                self.language_voice_map[language].append(
                    {
                        "id": voice_id,
                        "name": voice["name"],
                        "locale": locale,
                        "gender": voice.get("gender", "unknown"),
                        "voice_type": voice.get("voice_type", "Standard"),
                    }
                )

            logger.info(
                f"Initialized voice mappings: {len(self.voice_language_map)} voices, {len(self.language_voice_map)} languages"
            )

        except Exception as e:
            logger.error(f"Failed to initialize voice mappings: {e}")

    async def detect_language_from_audio(
        self, audio_data: Union[bytes, str], confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Detect language from audio using Whisper.

        Args:
            audio_data: Audio data or file path
            confidence_threshold: Minimum confidence for detection

        Returns:
            Language detection result with confidence
        """
        try:
            result = await whisper_service.detect_language(audio_data)

            detected_lang = result["detected_language"]
            confidence = result["confidence"]

            # Enhance result with language info
            enhanced_result = {
                "detected_language": detected_lang,
                "confidence": confidence,
                "method": "whisper_audio",
                "is_reliable": confidence >= confidence_threshold,
                "all_languages": result.get("all_languages", []),
            }

            # Add language metadata if available
            if detected_lang in self.supported_languages:
                lang_info = self.supported_languages[detected_lang]
                enhanced_result.update(
                    {
                        "language_name": lang_info["name"],
                        "native_name": lang_info["native_name"],
                        "default_voice": lang_info["default_voice"],
                        "is_rtl": lang_info["rtl"],
                    }
                )

            logger.info(f"Audio language detection: {detected_lang} ({confidence:.2f})")
            return enhanced_result

        except Exception as e:
            logger.error(f"Audio language detection failed: {e}")
            return {
                "detected_language": self.default_language,
                "confidence": 0.0,
                "method": "fallback",
                "is_reliable": False,
                "error": str(e),
            }

    async def detect_language_from_text(
        self, text: str, confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Detect language from text using langdetect.

        Args:
            text: Text to analyze
            confidence_threshold: Minimum confidence for detection

        Returns:
            Language detection result with confidence
        """
        if not LANGDETECT_AVAILABLE:
            logger.warning("langdetect not available for text language detection")
            return {
                "detected_language": self.default_language,
                "confidence": 0.0,
                "method": "unavailable",
                "is_reliable": False,
            }

        if not text or len(text.strip()) < 10:
            return {
                "detected_language": self.default_language,
                "confidence": 0.0,
                "method": "text_too_short",
                "is_reliable": False,
            }

        try:
            # Detect language with confidence
            lang_probs = detect_langs(text)

            if not lang_probs:
                raise LangDetectException("No language detected", "")

            top_detection = lang_probs[0]
            detected_lang = top_detection.lang
            confidence = top_detection.prob

            # Map to our language codes
            mapped_lang = self._map_langdetect_code(detected_lang)

            result = {
                "detected_language": mapped_lang,
                "confidence": confidence,
                "method": "langdetect_text",
                "is_reliable": confidence >= confidence_threshold,
                "all_languages": [
                    {
                        "language": self._map_langdetect_code(lp.lang),
                        "probability": lp.prob,
                    }
                    for lp in lang_probs[:5]
                ],
            }

            # Add language metadata
            if mapped_lang in self.supported_languages:
                lang_info = self.supported_languages[mapped_lang]
                result.update(
                    {
                        "language_name": lang_info["name"],
                        "native_name": lang_info["native_name"],
                        "default_voice": lang_info["default_voice"],
                        "is_rtl": lang_info["rtl"],
                    }
                )

            logger.info(f"Text language detection: {mapped_lang} ({confidence:.2f})")
            return result

        except LangDetectException as e:
            logger.warning(f"Text language detection failed: {e}")
            return {
                "detected_language": self.default_language,
                "confidence": 0.0,
                "method": "langdetect_failed",
                "is_reliable": False,
                "error": str(e),
            }
        except Exception as e:
            logger.error(f"Text language detection error: {e}")
            return {
                "detected_language": self.default_language,
                "confidence": 0.0,
                "method": "error",
                "is_reliable": False,
                "error": str(e),
            }

    def _map_langdetect_code(self, langdetect_code: str) -> str:
        """Map langdetect language codes to our standard codes."""
        # Language code mappings
        mapping = {
            "ca": "es",  # Catalan -> Spanish
            "eu": "es",  # Basque -> Spanish
            "gl": "es",  # Galician -> Spanish
            "cy": "en",  # Welsh -> English
            "ga": "en",  # Irish -> English
            "mt": "en",  # Maltese -> English
            "af": "en",  # Afrikaans -> English
            "eo": "en",  # Esperanto -> English
            "la": "en",  # Latin -> English
            "nb": "no",  # Norwegian Bokmål -> Norwegian
            "nn": "no",  # Norwegian Nynorsk -> Norwegian
        }

        # Return mapped code or original if no mapping exists
        mapped = mapping.get(langdetect_code, langdetect_code)

        # Ensure we return a supported language
        if mapped in self.supported_languages:
            return mapped
        else:
            return self.default_language

    async def get_best_voice_for_language(
        self,
        language: str,
        gender_preference: Optional[str] = None,
        voice_type_preference: str = "Neural",
    ) -> Optional[str]:
        """
        Get the best TTS voice for a specific language.

        Args:
            language: Language code
            gender_preference: "Male", "Female", or None
            voice_type_preference: "Neural", "Standard", or "Premium"

        Returns:
            Best matching voice ID or None
        """
        try:
            # Ensure voice mappings are initialized
            await self._ensure_initialized()

            # Check if language is supported
            if language not in self.supported_languages:
                language = self.default_language

            # Get available voices for language
            available_voices = self.language_voice_map.get(language, [])

            if not available_voices:
                # Fallback to default language
                available_voices = self.language_voice_map.get(
                    self.default_language, []
                )

            if not available_voices:
                return None

            # Filter by preferences
            filtered_voices = available_voices.copy()

            # Filter by voice type preference
            type_filtered = [
                v for v in filtered_voices if v["voice_type"] == voice_type_preference
            ]
            if type_filtered:
                filtered_voices = type_filtered

            # Filter by gender preference
            if gender_preference:
                gender_filtered = [
                    v for v in filtered_voices if v["gender"] == gender_preference
                ]
                if gender_filtered:
                    filtered_voices = gender_filtered

            # Return the first (default) voice or fallback to language default
            if filtered_voices:
                return filtered_voices[0]["id"]
            else:
                # Fallback to language default voice
                return self.supported_languages[language]["default_voice"]

        except Exception as e:
            logger.error(f"Failed to get best voice for language {language}: {e}")
            return self.supported_languages[self.default_language]["default_voice"]

    async def get_language_voices(self, language: str) -> List[Dict]:
        """
        Get all available voices for a specific language.

        Args:
            language: Language code

        Returns:
            List of voice information dictionaries
        """
        try:
            # Ensure voice mappings are initialized
            await self._ensure_initialized()

            if language not in self.language_voice_map:
                return []

            voices = self.language_voice_map[language]

            # Enhance with language metadata
            enhanced_voices = []
            for voice in voices:
                enhanced_voice = voice.copy()
                if language in self.supported_languages:
                    lang_info = self.supported_languages[language]
                    enhanced_voice.update(
                        {
                            "language_name": lang_info["name"],
                            "native_name": lang_info["native_name"],
                            "is_rtl": lang_info["rtl"],
                        }
                    )
                enhanced_voices.append(enhanced_voice)

            return enhanced_voices

        except Exception as e:
            logger.error(f"Failed to get voices for language {language}: {e}")
            return []

    async def auto_configure_voice(
        self,
        detected_language: str,
        confidence: float,
        user_preferences: Optional[Dict] = None,
    ) -> Dict:
        """
        Automatically configure voice settings based on detected language.

        Args:
            detected_language: Detected language code
            confidence: Detection confidence
            user_preferences: User voice preferences

        Returns:
            Voice configuration recommendation
        """
        try:
            # Use detected language if confidence is high enough
            target_language = (
                detected_language
                if confidence >= self.text_detection_threshold
                else self.default_language
            )

            # Get user preferences
            gender_pref = user_preferences.get("gender") if user_preferences else None
            voice_type_pref = (
                user_preferences.get("voice_type", "Neural")
                if user_preferences
                else "Neural"
            )

            # Get best voice
            recommended_voice = await self.get_best_voice_for_language(
                target_language, gender_pref, voice_type_pref
            )

            # Get language info
            lang_info = self.supported_languages.get(target_language, {})

            configuration = {
                "language": target_language,
                "voice_id": recommended_voice,
                "confidence": confidence,
                "language_name": lang_info.get("name", target_language),
                "native_name": lang_info.get("native_name", target_language),
                "is_rtl": lang_info.get("rtl", False),
                "auto_configured": True,
                "fallback_used": target_language != detected_language,
            }

            logger.info(
                f"Auto-configured voice: {target_language} -> {recommended_voice}"
            )
            return configuration

        except Exception as e:
            logger.error(f"Auto voice configuration failed: {e}")
            return {
                "language": self.default_language,
                "voice_id": self.supported_languages[self.default_language][
                    "default_voice"
                ],
                "confidence": 0.0,
                "auto_configured": False,
                "error": str(e),
            }

    async def get_supported_languages(self, include_voices: bool = False) -> List[Dict]:
        """
        Get list of all supported languages with metadata.

        Args:
            include_voices: Whether to include voice information

        Returns:
            List of supported language information
        """
        languages = []

        for lang_code, lang_info in self.supported_languages.items():
            language_data = {
                "code": lang_code,
                "name": lang_info["name"],
                "native_name": lang_info["native_name"],
                "iso_639_1": lang_info["iso_639_1"],
                "default_voice": lang_info["default_voice"],
                "is_rtl": lang_info["rtl"],
                "regions": lang_info["regions"],
                "priority": lang_info["priority"],
            }

            if include_voices:
                language_data["voices"] = await self.get_language_voices(lang_code)
                language_data["voice_count"] = len(language_data["voices"])

            languages.append(language_data)

        # Sort by priority
        languages.sort(key=lambda x: x["priority"])
        return languages

    def get_language_info(self, language: str) -> Optional[Dict]:
        """
        Get detailed information about a specific language.

        Args:
            language: Language code

        Returns:
            Language information or None
        """
        return self.supported_languages.get(language)

    def is_language_supported(self, language: str) -> bool:
        """
        Check if a language is supported.

        Args:
            language: Language code

        Returns:
            True if language is supported
        """
        return language in self.supported_languages

    async def get_service_info(self) -> Dict:
        """Get language service information and capabilities."""
        # Ensure voice mappings are initialized
        await self._ensure_initialized()

        return {
            "supported_languages_count": len(self.supported_languages),
            "supported_languages": list(self.supported_languages.keys()),
            "default_language": self.default_language,
            "fallback_language": self.fallback_language,
            "capabilities": {
                "audio_language_detection": True,
                "text_language_detection": LANGDETECT_AVAILABLE,
                "voice_auto_configuration": True,
                "multi_language_voices": len(self.language_voice_map) > 0,
                "rtl_support": any(
                    lang["rtl"] for lang in self.supported_languages.values()
                ),
            },
            "detection_settings": {
                "text_detection_threshold": self.text_detection_threshold,
                "audio_detection_enabled": self.audio_detection_enabled,
                "text_detection_enabled": self.text_detection_enabled,
            },
            "voice_mappings": {
                "total_voices": len(self.voice_language_map),
                "languages_with_voices": len(self.language_voice_map),
            },
        }


# Global service instance
language_service = LanguageService()


# Utility functions for common operations
async def detect_language_auto(
    audio_data: Optional[Union[bytes, str]] = None,
    text: Optional[str] = None,
    prefer_audio: bool = True,
) -> Dict:
    """
    Automatically detect language from audio or text.

    Args:
        audio_data: Audio data for detection
        text: Text for detection
        prefer_audio: Prefer audio detection when both available

    Returns:
        Language detection result
    """
    if audio_data and (prefer_audio or not text):
        return await language_service.detect_language_from_audio(audio_data)
    elif text:
        return await language_service.detect_language_from_text(text)
    else:
        return {
            "detected_language": language_service.default_language,
            "confidence": 0.0,
            "method": "no_input",
            "is_reliable": False,
        }


async def get_recommended_voice(
    language: str, gender: Optional[str] = None, voice_type: str = "Neural"
) -> Optional[str]:
    """
    Get recommended voice for language and preferences.

    Args:
        language: Language code
        gender: Gender preference
        voice_type: Voice type preference

    Returns:
        Recommended voice ID
    """
    return await language_service.get_best_voice_for_language(
        language, gender, voice_type
    )


async def configure_voice_for_content(
    audio_data: Optional[Union[bytes, str]] = None,
    text: Optional[str] = None,
    user_preferences: Optional[Dict] = None,
) -> Dict:
    """
    Automatically configure voice based on content and preferences.

    Args:
        audio_data: Audio data for language detection
        text: Text for language detection
        user_preferences: User voice preferences

    Returns:
        Voice configuration recommendation
    """
    # Detect language
    detection_result = await detect_language_auto(audio_data, text)

    # Configure voice
    return await language_service.auto_configure_voice(
        detection_result["detected_language"],
        detection_result["confidence"],
        user_preferences,
    )
