"""
Lip-Sync Analysis Service.

Analyzes audio data to generate synchronized lip-sync animation data for avatars.
Integrates with TTS service to provide real-time lip-sync for avatar speech.
"""

import logging
import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import asyncio
import io
import wave

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None
    logging.info("librosa not available - using basic lip-sync analysis (install with: pip install librosa)")

logger = logging.getLogger(__name__)


class LipSyncAnalyzer:
    """Real-time lip-sync analysis for avatar animation."""
    
    def __init__(self):
        self.sample_rate = 16000
        self.frame_duration = 0.025  # 25ms frames
        self.hop_length = int(self.sample_rate * self.frame_duration)
        
        # Phoneme to viseme mapping for basic lip-sync
        self.phoneme_to_viseme = {
            # Silence
            'sil': 'sil', 'sp': 'sil', '': 'sil',
            
            # Vowels
            'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'ao', 'aw': 'aw',
            'ay': 'ay', 'eh': 'eh', 'er': 'er', 'ey': 'ey', 'ih': 'ih',
            'iy': 'iy', 'ow': 'ow', 'oy': 'oy', 'uh': 'uh', 'uw': 'uw',
            
            # Consonants
            'b': 'b', 'ch': 'ch', 'd': 'd', 'dh': 'dh', 'f': 'f',
            'g': 'g', 'hh': 'hh', 'jh': 'jh', 'k': 'k', 'l': 'l',
            'm': 'm', 'n': 'n', 'ng': 'ng', 'p': 'p', 'r': 'r',
            's': 's', 'sh': 'sh', 't': 't', 'th': 'th', 'v': 'v',
            'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'zh'
        }
        
        # Viseme intensity mapping based on mouth opening
        self.viseme_intensity = {
            'sil': 0.0,  # Closed mouth
            'aa': 0.9,   # Wide open
            'ae': 0.7,   # Medium open
            'ah': 0.8,   # Open
            'ao': 0.6,   # Rounded
            'aw': 0.5,   # Rounded small
            'ay': 0.4,   # Slight open
            'b': 0.1,    # Closed (bilabial)
            'ch': 0.3,   # Slight open
            'd': 0.2,    # Slight open
            'dh': 0.3,   # Tongue between teeth
            'eh': 0.5,   # Medium
            'er': 0.4,   # Slight open
            'ey': 0.3,   # Slight open
            'f': 0.2,    # Lower lip to teeth
            'g': 0.3,    # Slight open
            'hh': 0.2,   # Slight open
            'ih': 0.3,   # Small open
            'iy': 0.2,   # Small open
            'jh': 0.4,   # Medium open
            'k': 0.2,    # Slight open
            'l': 0.3,    # Tongue tip
            'm': 0.0,    # Closed (bilabial)
            'n': 0.2,    # Slight open
            'ng': 0.2,   # Slight open
            'ow': 0.7,   # Rounded open
            'oy': 0.6,   # Rounded medium
            'p': 0.0,    # Closed (bilabial)
            'r': 0.4,    # Slight open
            's': 0.1,    # Small gap
            'sh': 0.2,   # Rounded small
            't': 0.2,    # Tongue to teeth
            'th': 0.3,   # Tongue between teeth
            'uh': 0.4,   # Medium
            'uw': 0.7,   # Rounded open
            'v': 0.2,    # Lower lip to teeth
            'w': 0.7,    # Rounded
            'y': 0.2,    # Slight open
            'z': 0.1,    # Small gap
            'zh': 0.2,   # Rounded small
        }
    
    def _estimate_phonemes_from_audio(self, audio_data: np.ndarray) -> List[Tuple[str, float, float]]:
        """
        Estimate phonemes from audio using basic audio analysis.
        Returns list of (phoneme, start_time, end_time) tuples.
        """
        if not LIBROSA_AVAILABLE:
            return self._basic_phoneme_estimation(audio_data)
        
        try:
            # Use librosa for better analysis
            # Extract MFCCs for phoneme classification
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            
            # Simple energy-based segmentation
            energy = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
            times = librosa.frames_to_time(np.arange(len(energy)), sr=self.sample_rate, hop_length=self.hop_length)
            
            # Basic phoneme estimation based on spectral features
            phonemes = []
            for i, (time, eng) in enumerate(zip(times, energy)):
                if eng > 0.01:  # Voice activity threshold
                    # Simple classification based on spectral centroid
                    if i < len(mfccs[0]):
                        spectral_centroid = np.mean(mfccs[:, i])
                        phoneme = self._classify_phoneme_from_features(spectral_centroid, eng)
                        duration = self.frame_duration
                        phonemes.append((phoneme, time, time + duration))
                else:
                    phonemes.append(('sil', time, time + self.frame_duration))
            
            return phonemes
            
        except Exception as e:
            logger.warning(f"Advanced phoneme estimation failed: {e}, using basic method")
            return self._basic_phoneme_estimation(audio_data)
    
    def _basic_phoneme_estimation(self, audio_data: np.ndarray) -> List[Tuple[str, float, float]]:
        """Basic phoneme estimation using energy and zero-crossing rate."""
        # Simple energy-based segmentation
        frame_length = int(self.sample_rate * self.frame_duration)
        num_frames = len(audio_data) // frame_length
        
        phonemes = []
        for i in range(num_frames):
            start_idx = i * frame_length
            end_idx = start_idx + frame_length
            frame = audio_data[start_idx:end_idx]
            
            # Calculate frame energy
            energy = np.mean(frame ** 2)
            
            # Calculate zero crossing rate
            zcr = np.sum(np.diff(np.sign(frame)) != 0) / len(frame)
            
            # Simple phoneme classification
            start_time = i * self.frame_duration
            end_time = start_time + self.frame_duration
            
            if energy < 0.001:
                phoneme = 'sil'
            elif energy > 0.1:
                # High energy - likely vowel
                if zcr < 0.1:
                    phoneme = 'aa'  # Open vowel
                else:
                    phoneme = 'iy'  # Close vowel
            elif zcr > 0.2:
                # High ZCR - likely fricative
                phoneme = 's'
            else:
                # Medium energy/ZCR - likely stop or nasal
                phoneme = 'd'
            
            phonemes.append((phoneme, start_time, end_time))
        
        return phonemes
    
    def _classify_phoneme_from_features(self, spectral_centroid: float, energy: float) -> str:
        """Classify phoneme based on spectral features."""
        # Simple rule-based classification
        if energy < 0.01:
            return 'sil'
        elif spectral_centroid > 0.5:
            # High spectral centroid - fricatives
            return 's' if energy > 0.05 else 'sh'
        elif spectral_centroid > 0.0:
            # Medium spectral centroid - vowels
            return 'aa' if energy > 0.1 else 'ih'
        else:
            # Low spectral centroid - stops/nasals
            return 'm' if energy < 0.05 else 'd'
    
    def analyze_audio_for_visemes(self, audio_data: bytes) -> List[Dict]:
        """
        Extract viseme data from audio for lip-sync animation.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            List of viseme dictionaries with timing and intensity
        """
        try:
            # Convert bytes to numpy array
            if isinstance(audio_data, bytes):
                # Assume 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                audio_array = np.array(audio_data, dtype=np.float32)
            
            # Estimate phonemes from audio
            phonemes = self._estimate_phonemes_from_audio(audio_array)
            
            # Convert phonemes to visemes
            visemes = []
            for phoneme, start_time, end_time in phonemes:
                viseme = self.phoneme_to_viseme.get(phoneme, 'sil')
                intensity = self.viseme_intensity.get(viseme, 0.0)
                
                visemes.append({
                    'viseme': viseme,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'intensity': intensity,
                    'phoneme': phoneme
                })
            
            return visemes
            
        except Exception as e:
            logger.error(f"Error analyzing audio for visemes: {e}")
            return []
    
    def generate_lip_sync_data(self, text: str, audio_data: bytes) -> Dict[str, Any]:
        """
        Generate synchronized lip-sync data for text and audio.
        
        Args:
            text: The text being spoken
            audio_data: Raw audio data
            
        Returns:
            Dictionary containing lip-sync animation data
        """
        try:
            # Analyze audio for visemes
            visemes = self.analyze_audio_for_visemes(audio_data)
            
            # Calculate total duration
            total_duration = max([v['end_time'] for v in visemes]) if visemes else 0.0
            
            # Generate keyframes for animation
            keyframes = self._generate_animation_keyframes(visemes)
            
            # Create timeline data
            timeline = {
                'visemes': visemes,
                'keyframes': keyframes,
                'duration': total_duration,
                'sample_rate': self.sample_rate,
                'frame_rate': 30  # Target animation frame rate
            }
            
            return {
                'text': text,
                'timeline': timeline,
                'viseme_count': len(visemes),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error generating lip-sync data: {e}")
            return {
                'text': text,
                'error': str(e),
                'success': False,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
    
    def _generate_animation_keyframes(self, visemes: List[Dict]) -> List[Dict]:
        """Generate smooth animation keyframes from viseme data."""
        keyframes = []
        
        for i, viseme in enumerate(visemes):
            # Add keyframe at viseme start
            keyframes.append({
                'time': viseme['start_time'],
                'mouth_shape': viseme['viseme'],
                'mouth_opening': viseme['intensity'],
                'transition_type': 'ease_in'
            })
            
            # Add transition keyframe if not the last viseme
            if i < len(visemes) - 1:
                next_viseme = visemes[i + 1]
                transition_time = viseme['end_time']
                
                keyframes.append({
                    'time': transition_time,
                    'mouth_shape': viseme['viseme'],
                    'mouth_opening': viseme['intensity'] * 0.7,  # Reduce intensity for transition
                    'transition_type': 'ease_out'
                })
        
        return keyframes
    
    async def analyze_streaming_audio(self, audio_stream: bytes) -> Dict[str, Any]:
        """
        Analyze streaming audio for real-time lip-sync.
        
        Args:
            audio_stream: Streaming audio data
            
        Returns:
            Real-time viseme data
        """
        try:
            # Analyze current audio chunk
            visemes = self.analyze_audio_for_visemes(audio_stream)
            
            # Return immediate viseme data for real-time animation
            current_viseme = visemes[-1] if visemes else {
                'viseme': 'sil',
                'intensity': 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return {
                'current_viseme': current_viseme,
                'viseme_buffer': visemes[-5:],  # Last 5 visemes for smoothing
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error analyzing streaming audio: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def get_analysis_capabilities(self) -> Dict[str, Any]:
        """Get information about lip-sync analysis capabilities."""
        return {
            'sample_rate': self.sample_rate,
            'frame_duration': self.frame_duration,
            'supported_visemes': list(self.viseme_intensity.keys()),
            'phoneme_count': len(self.phoneme_to_viseme),
            'librosa_available': LIBROSA_AVAILABLE,
            'real_time_capable': True,
            'streaming_supported': True
        }


# Global service instance
lip_sync_analyzer = LipSyncAnalyzer()


class LipSyncService:
    """Service for coordinating lip-sync analysis with other services."""
    
    def __init__(self):
        self.analyzer = lip_sync_analyzer
        self.active_sessions: Dict[str, Dict] = {}
    
    async def start_speech_session(self, session_id: str, text: str) -> Dict[str, Any]:
        """Start a new speech session with lip-sync tracking."""
        try:
            self.active_sessions[session_id] = {
                'text': text,
                'start_time': datetime.utcnow(),
                'status': 'active'
            }
            
            return {
                'session_id': session_id,
                'status': 'started',
                'text': text,
                'capabilities': self.analyzer.get_analysis_capabilities()
            }
            
        except Exception as e:
            logger.error(f"Error starting speech session: {e}")
            return {'error': str(e), 'success': False}
    
    async def process_audio_chunk(self, session_id: str, audio_chunk: bytes) -> Dict[str, Any]:
        """Process audio chunk for real-time lip-sync."""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found', 'success': False}
        
        return await self.analyzer.analyze_streaming_audio(audio_chunk)
    
    async def complete_speech_session(self, session_id: str, final_audio: bytes) -> Dict[str, Any]:
        """Complete speech session and generate final lip-sync data."""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found', 'success': False}
        
        session = self.active_sessions[session_id]
        text = session.get('text', '')
        
        # Generate complete lip-sync data
        lip_sync_data = self.analyzer.generate_lip_sync_data(text, final_audio)
        
        # Clean up session
        del self.active_sessions[session_id]
        
        return lip_sync_data
    
    def get_active_sessions(self) -> Dict[str, Dict]:
        """Get information about active lip-sync sessions."""
        return self.active_sessions.copy()


# Global service instance
lip_sync_service = LipSyncService()