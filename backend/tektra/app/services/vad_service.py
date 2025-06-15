"""
Voice Activity Detection (VAD) and Audio Processing Service.

Provides intelligent voice activity detection, noise cancellation, and audio
preprocessing capabilities for enhanced voice interactions.
"""

import asyncio
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
import io
import wave

try:
    import webrtcvad
    WEBRTC_VAD_AVAILABLE = True
except ImportError:
    WEBRTC_VAD_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    from scipy import signal
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False

try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False

from ..config import settings

logger = logging.getLogger(__name__)


class VADService:
    """Voice Activity Detection and audio preprocessing service."""
    
    def __init__(self):
        self.vad = None
        self.sample_rate = 16000  # WebRTC VAD requires 16kHz
        self.frame_duration = 30  # Frame duration in ms (10, 20, or 30)
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.aggressiveness = 2  # VAD aggressiveness (0-3, higher = more aggressive)
        
        # Voice activity settings
        self.min_speech_frames = 3  # Minimum consecutive frames to consider speech
        self.min_silence_frames = 10  # Minimum consecutive frames to consider silence
        self.speech_threshold = 0.7  # Percentage of frames that must be speech
        
        # Noise reduction settings
        self.noise_reduce_enabled = True
        self.noise_sample_duration = 1.0  # Duration in seconds for noise sampling
        self.noise_reduction_strength = 0.8  # Reduction strength (0.0-1.0)
        
        # Audio preprocessing settings
        self.normalize_audio = True
        self.apply_highpass_filter = True
        self.highpass_cutoff = 80  # Hz - remove low frequency noise
        self.apply_lowpass_filter = True
        self.lowpass_cutoff = 8000  # Hz - remove high frequency noise
        
        # Initialize VAD if available
        if WEBRTC_VAD_AVAILABLE:
            self._initialize_vad()
    
    def _initialize_vad(self):
        """Initialize WebRTC VAD."""
        try:
            self.vad = webrtcvad.Vad(self.aggressiveness)
            logger.info(f"WebRTC VAD initialized with aggressiveness {self.aggressiveness}")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC VAD: {e}")
            self.vad = None
    
    def set_aggressiveness(self, level: int):
        """
        Set VAD aggressiveness level.
        
        Args:
            level: Aggressiveness level (0-3)
                  0 = least aggressive (more sensitive)
                  3 = most aggressive (less sensitive)
        """
        if not WEBRTC_VAD_AVAILABLE:
            logger.warning("WebRTC VAD not available")
            return
        
        if 0 <= level <= 3:
            self.aggressiveness = level
            self._initialize_vad()
        else:
            raise ValueError("Aggressiveness level must be between 0 and 3")
    
    async def preprocess_audio(
        self,
        audio_data: Union[bytes, np.ndarray],
        input_sample_rate: Optional[int] = None,
        reduce_noise: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio data with noise reduction and filtering.
        
        Args:
            audio_data: Raw audio data
            input_sample_rate: Original sample rate (auto-detect if None)
            reduce_noise: Apply noise reduction
            normalize: Normalize audio levels
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        if not AUDIO_PROCESSING_AVAILABLE:
            logger.warning("Audio processing libraries not available")
            # Convert bytes to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            return audio_array, self.sample_rate
        
        try:
            # Convert input to numpy array
            if isinstance(audio_data, bytes):
                # Try to load as WAV first
                try:
                    with io.BytesIO(audio_data) as audio_io:
                        audio_array, sr = sf.read(audio_io)
                        if input_sample_rate is None:
                            input_sample_rate = sr
                except:
                    # Fallback: assume raw 16-bit PCM
                    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if input_sample_rate is None:
                        input_sample_rate = self.sample_rate
            else:
                audio_array = audio_data.astype(np.float32)
                if input_sample_rate is None:
                    input_sample_rate = self.sample_rate
            
            # Ensure mono audio
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to target sample rate if needed
            if input_sample_rate != self.sample_rate:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=input_sample_rate, 
                    target_sr=self.sample_rate
                )
            
            # Apply filters
            audio_array = await self._apply_filters(audio_array)
            
            # Noise reduction
            if reduce_noise and NOISE_REDUCE_AVAILABLE and self.noise_reduce_enabled:
                audio_array = await self._reduce_noise(audio_array)
            
            # Normalize audio
            if normalize and self.normalize_audio:
                audio_array = await self._normalize_audio(audio_array)
            
            logger.info(f"Preprocessed audio: {len(audio_array)} samples at {self.sample_rate}Hz")
            return audio_array, self.sample_rate
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            # Return original audio on failure
            if isinstance(audio_data, bytes):
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = audio_data
            return audio_array, input_sample_rate or self.sample_rate
    
    async def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio filters (highpass and lowpass)."""
        try:
            # Apply highpass filter to remove low frequency noise
            if self.apply_highpass_filter:
                nyquist = self.sample_rate / 2
                high_cutoff = self.highpass_cutoff / nyquist
                if high_cutoff < 1.0:
                    b, a = signal.butter(4, high_cutoff, btype='high')
                    audio = signal.filtfilt(b, a, audio)
            
            # Apply lowpass filter to remove high frequency noise
            if self.apply_lowpass_filter:
                nyquist = self.sample_rate / 2
                low_cutoff = self.lowpass_cutoff / nyquist
                if low_cutoff < 1.0:
                    b, a = signal.butter(4, low_cutoff, btype='low')
                    audio = signal.filtfilt(b, a, audio)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")
            return audio
    
    async def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio."""
        try:
            # Use the first portion of audio as noise sample if long enough
            if len(audio) > self.sample_rate * self.noise_sample_duration:
                noise_sample_size = int(self.sample_rate * self.noise_sample_duration)
                noise_sample = audio[:noise_sample_size]
                
                # Apply noise reduction
                reduced_audio = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=self.noise_reduction_strength
                )
                return reduced_audio
            else:
                # Audio too short for noise sampling, apply gentle noise reduction
                reduced_audio = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    prop_decrease=self.noise_reduction_strength * 0.5
                )
                return reduced_audio
                
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    async def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        try:
            # RMS-based normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level
                audio = audio * (target_rms / rms)
            
            # Ensure audio doesn't clip
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val
                
            return audio
            
        except Exception as e:
            logger.warning(f"Audio normalization failed: {e}")
            return audio
    
    async def detect_voice_activity(
        self,
        audio_data: Union[bytes, np.ndarray],
        input_sample_rate: Optional[int] = None
    ) -> Dict:
        """
        Detect voice activity in audio data.
        
        Args:
            audio_data: Audio data to analyze
            input_sample_rate: Original sample rate
            
        Returns:
            Dictionary with VAD results
        """
        if not WEBRTC_VAD_AVAILABLE or not self.vad:
            # Fallback to energy-based VAD
            return await self._energy_based_vad(audio_data, input_sample_rate)
        
        try:
            # Preprocess audio
            audio_array, sr = await self.preprocess_audio(
                audio_data, input_sample_rate, reduce_noise=False
            )
            
            # Convert to 16-bit PCM for WebRTC VAD
            audio_pcm = (audio_array * 32767).astype(np.int16).tobytes()
            
            # Analyze frames
            frames = self._split_into_frames(audio_pcm)
            voice_frames = []
            
            for frame in frames:
                if len(frame) == self.frame_size * 2:  # 2 bytes per sample
                    is_speech = self.vad.is_speech(frame, self.sample_rate)
                    voice_frames.append(is_speech)
                    
            # Calculate voice activity statistics
            total_frames = len(voice_frames)
            speech_frames = sum(voice_frames)
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0.0
            
            # Determine overall voice activity
            has_voice = speech_ratio >= self.speech_threshold
            
            # Find speech segments
            segments = self._find_speech_segments(voice_frames)
            
            result = {
                "has_voice": has_voice,
                "speech_ratio": speech_ratio,
                "total_frames": total_frames,
                "speech_frames": speech_frames,
                "segments": segments,
                "confidence": min(speech_ratio * 2, 1.0),  # Boost confidence for display
                "duration": len(audio_array) / self.sample_rate,
                "method": "webrtc_vad"
            }
            
            logger.debug(f"VAD result: {speech_ratio:.2f} speech ratio, {has_voice} voice detected")
            return result
            
        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            # Fallback to energy-based VAD
            return await self._energy_based_vad(audio_data, input_sample_rate)
    
    async def _energy_based_vad(
        self,
        audio_data: Union[bytes, np.ndarray],
        input_sample_rate: Optional[int] = None
    ) -> Dict:
        """Fallback energy-based voice activity detection."""
        try:
            # Preprocess audio
            audio_array, sr = await self.preprocess_audio(
                audio_data, input_sample_rate, reduce_noise=False
            )
            
            # Calculate RMS energy
            frame_length = int(sr * 0.025)  # 25ms frames
            hop_length = int(sr * 0.010)   # 10ms hop
            
            # Split into frames and calculate energy
            energies = []
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frame = audio_array[i:i + frame_length]
                energy = np.sqrt(np.mean(frame**2))
                energies.append(energy)
            
            if not energies:
                return {
                    "has_voice": False,
                    "speech_ratio": 0.0,
                    "confidence": 0.0,
                    "duration": 0.0,
                    "method": "energy_based"
                }
            
            # Calculate energy threshold (adaptive)
            energy_array = np.array(energies)
            mean_energy = np.mean(energy_array)
            std_energy = np.std(energy_array)
            threshold = mean_energy + (std_energy * 1.5)
            
            # Count frames above threshold
            voice_frames = energy_array > threshold
            speech_ratio = np.sum(voice_frames) / len(voice_frames)
            
            # Determine voice activity
            has_voice = speech_ratio >= 0.3  # Lower threshold for energy-based
            
            result = {
                "has_voice": has_voice,
                "speech_ratio": speech_ratio,
                "confidence": min(speech_ratio * 1.5, 1.0),
                "duration": len(audio_array) / sr,
                "energy_threshold": threshold,
                "mean_energy": mean_energy,
                "method": "energy_based"
            }
            
            logger.debug(f"Energy-based VAD: {speech_ratio:.2f} speech ratio, {has_voice} voice detected")
            return result
            
        except Exception as e:
            logger.error(f"Energy-based VAD failed: {e}")
            return {
                "has_voice": False,
                "speech_ratio": 0.0,
                "confidence": 0.0,
                "duration": 0.0,
                "method": "fallback",
                "error": str(e)
            }
    
    def _split_into_frames(self, audio_pcm: bytes) -> List[bytes]:
        """Split audio PCM data into frames for VAD processing."""
        frame_byte_size = self.frame_size * 2  # 2 bytes per 16-bit sample
        frames = []
        
        for i in range(0, len(audio_pcm) - frame_byte_size, frame_byte_size):
            frame = audio_pcm[i:i + frame_byte_size]
            frames.append(frame)
            
        return frames
    
    def _find_speech_segments(self, voice_frames: List[bool]) -> List[Dict]:
        """Find continuous speech segments from VAD frame results."""
        segments = []
        current_segment = None
        
        for i, is_speech in enumerate(voice_frames):
            frame_time = i * self.frame_duration / 1000  # Convert to seconds
            
            if is_speech:
                if current_segment is None:
                    # Start new speech segment
                    current_segment = {
                        "start": frame_time,
                        "end": frame_time + self.frame_duration / 1000,
                        "frames": 1
                    }
                else:
                    # Extend current segment
                    current_segment["end"] = frame_time + self.frame_duration / 1000
                    current_segment["frames"] += 1
            else:
                if current_segment is not None:
                    # End current segment if it meets minimum length
                    if current_segment["frames"] >= self.min_speech_frames:
                        segments.append(current_segment)
                    current_segment = None
        
        # Add final segment if exists
        if current_segment is not None and current_segment["frames"] >= self.min_speech_frames:
            segments.append(current_segment)
        
        return segments
    
    async def is_silence(
        self,
        audio_data: Union[bytes, np.ndarray],
        input_sample_rate: Optional[int] = None,
        threshold_duration: float = 0.5
    ) -> bool:
        """
        Check if audio contains only silence.
        
        Args:
            audio_data: Audio data to check
            input_sample_rate: Original sample rate
            threshold_duration: Minimum duration to consider for silence
            
        Returns:
            True if audio is considered silence
        """
        try:
            vad_result = await self.detect_voice_activity(audio_data, input_sample_rate)
            
            # Consider silence if:
            # 1. No voice detected, OR
            # 2. Very low speech ratio, OR  
            # 3. Duration is very short
            
            is_silent = (
                not vad_result["has_voice"] or
                vad_result["speech_ratio"] < 0.1 or
                vad_result["duration"] < threshold_duration
            )
            
            return is_silent
            
        except Exception as e:
            logger.error(f"Silence detection failed: {e}")
            return True  # Assume silence on error
    
    async def get_voice_segments(
        self,
        audio_data: Union[bytes, np.ndarray],
        input_sample_rate: Optional[int] = None,
        min_segment_duration: float = 0.5,
        max_silence_gap: float = 0.3
    ) -> List[Dict]:
        """
        Extract voice segments from audio with timing information.
        
        Args:
            audio_data: Audio data to process
            input_sample_rate: Original sample rate
            min_segment_duration: Minimum duration for voice segments
            max_silence_gap: Maximum silence gap to bridge segments
            
        Returns:
            List of voice segments with start/end times
        """
        try:
            vad_result = await self.detect_voice_activity(audio_data, input_sample_rate)
            
            if not vad_result.get("segments"):
                return []
            
            # Filter and merge segments
            filtered_segments = []
            for segment in vad_result["segments"]:
                duration = segment["end"] - segment["start"]
                if duration >= min_segment_duration:
                    filtered_segments.append(segment)
            
            # Merge segments with small gaps
            merged_segments = []
            for segment in filtered_segments:
                if not merged_segments:
                    merged_segments.append(segment)
                else:
                    last_segment = merged_segments[-1]
                    gap = segment["start"] - last_segment["end"]
                    
                    if gap <= max_silence_gap:
                        # Merge segments
                        last_segment["end"] = segment["end"]
                        last_segment["frames"] += segment["frames"]
                    else:
                        merged_segments.append(segment)
            
            return merged_segments
            
        except Exception as e:
            logger.error(f"Voice segment extraction failed: {e}")
            return []
    
    async def get_service_info(self) -> Dict:
        """Get information about the VAD service capabilities."""
        return {
            "webrtc_vad_available": WEBRTC_VAD_AVAILABLE,
            "audio_processing_available": AUDIO_PROCESSING_AVAILABLE,
            "noise_reduce_available": NOISE_REDUCE_AVAILABLE,
            "sample_rate": self.sample_rate,
            "frame_duration": self.frame_duration,
            "aggressiveness": self.aggressiveness,
            "features": {
                "voice_activity_detection": True,
                "noise_reduction": NOISE_REDUCE_AVAILABLE and self.noise_reduce_enabled,
                "audio_filtering": AUDIO_PROCESSING_AVAILABLE,
                "silence_detection": True,
                "voice_segmentation": True
            },
            "settings": {
                "speech_threshold": self.speech_threshold,
                "min_speech_frames": self.min_speech_frames,
                "min_silence_frames": self.min_silence_frames,
                "noise_reduction_strength": self.noise_reduction_strength,
                "highpass_cutoff": self.highpass_cutoff,
                "lowpass_cutoff": self.lowpass_cutoff
            }
        }


# Global service instance
vad_service = VADService()


# Utility functions for common operations
async def detect_voice_in_audio(
    audio_data: Union[bytes, np.ndarray],
    sample_rate: Optional[int] = None
) -> bool:
    """
    Convenience function to detect if audio contains voice.
    
    Args:
        audio_data: Audio data to analyze
        sample_rate: Original sample rate
        
    Returns:
        True if voice is detected
    """
    result = await vad_service.detect_voice_activity(audio_data, sample_rate)
    return result["has_voice"]


async def preprocess_voice_audio(
    audio_data: Union[bytes, np.ndarray],
    sample_rate: Optional[int] = None,
    reduce_noise: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Convenience function to preprocess audio for voice processing.
    
    Args:
        audio_data: Raw audio data
        sample_rate: Original sample rate
        reduce_noise: Apply noise reduction
        
    Returns:
        Tuple of (processed_audio, sample_rate)
    """
    return await vad_service.preprocess_audio(
        audio_data, sample_rate, reduce_noise=reduce_noise
    )


async def is_audio_silence(
    audio_data: Union[bytes, np.ndarray],
    sample_rate: Optional[int] = None
) -> bool:
    """
    Convenience function to check if audio is silence.
    
    Args:
        audio_data: Audio data to check
        sample_rate: Original sample rate
        
    Returns:
        True if audio is considered silence
    """
    return await vad_service.is_silence(audio_data, sample_rate)