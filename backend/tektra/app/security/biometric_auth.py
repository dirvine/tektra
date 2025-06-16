"""
Biometric Authentication Service.

Handles face recognition and voice authentication for secure user identification.
"""

import asyncio
import logging
import tempfile
import io
import base64
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import json

# Dynamic imports - will be loaded when needed
cv2 = None
face_recognition = None
librosa = None
sf = None
SpeakerRecognition = None

from .key_derivation import key_derivation_service

logger = logging.getLogger(__name__)


class BiometricAuthService:
    """Service for biometric authentication using face and voice recognition."""
    
    def __init__(self):
        self.face_model_loaded = False
        self.voice_model_loaded = False
        self.voice_model = None
        
        # Face recognition settings
        self.face_tolerance = 0.6  # Lower = more strict
        self.face_encoding_model = "large"  # "small" or "large"
        
        # Voice recognition settings
        self.sample_rate = 16000
        self.min_voice_duration = 2.0  # Minimum seconds for voice sample
        self.voice_similarity_threshold = 0.25  # Lower = more strict
        
        # User database (in production, this would be persistent storage)
        self.registered_users: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize biometric models with automatic dependency installation."""
        try:
            # Start background installation of biometric dependencies
            from ..services.auto_installer import auto_installer
            
            logger.info("Initializing biometric capabilities...")
            
            # Try to load face recognition capabilities
            face_available = await self._ensure_face_recognition()
            if face_available:
                self.face_model_loaded = True
                logger.info("✓ Face recognition ready")
            else:
                # Start background installation
                auto_installer.start_background_installation("biometric")
                logger.info("○ Face recognition installing in background")
            
            # Try to load voice recognition capabilities
            voice_available = await self._ensure_voice_recognition()
            if voice_available:
                await self._load_voice_model()
                logger.info("✓ Voice recognition ready")
            else:
                # Start background installation
                auto_installer.start_background_installation("advanced_audio")
                logger.info("○ Voice recognition installing in background")
                
            logger.info("Biometric services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize biometric auth: {e}")
    
    async def _ensure_face_recognition(self) -> bool:
        """Ensure face recognition dependencies are available."""
        global cv2, face_recognition
        
        try:
            if cv2 is None:
                import cv2 as cv2_module
                cv2 = cv2_module
            
            if face_recognition is None:
                import face_recognition as fr_module
                face_recognition = fr_module
            
            return True
        except ImportError:
            return False
    
    async def _ensure_voice_recognition(self) -> bool:
        """Ensure voice recognition dependencies are available."""
        global librosa, sf, SpeakerRecognition
        
        try:
            if librosa is None:
                import librosa as librosa_module
                librosa = librosa_module
            
            if sf is None:
                import soundfile as sf_module
                sf = sf_module
            
            if SpeakerRecognition is None:
                from speechbrain.pretrained import SpeakerRecognition as SR
                SpeakerRecognition = SR
            
            return True
        except ImportError:
            return False
            
    async def _load_voice_model(self):
        """Load the voice recognition model."""
        try:
            if not await self._ensure_voice_recognition():
                return False
                
            loop = asyncio.get_event_loop()
            
            def load_model():
                return SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="./pretrained_models/spkrec-ecapa-voxceleb"
                )
            
            self.voice_model = await loop.run_in_executor(None, load_model)
            self.voice_model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load voice model: {e}")
            self.voice_model_loaded = False
    
    async def extract_face_encoding(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Extract face encoding from image data.
        
        Args:
            image_data: Image bytes (JPEG/PNG)
            
        Returns:
            Face encoding array or None if no face found
        """
        if not self.face_model_loaded:
            # Try to ensure face recognition is available
            if await self._ensure_face_recognition():
                self.face_model_loaded = True
            else:
                logger.error("Face recognition not available")
                return None
            
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Could not decode image")
                return None
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image, model="hog")
            
            if not face_locations:
                logger.warning("No faces found in image")
                return None
                
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces found ({len(face_locations)}), using first one")
            
            # Extract face encoding
            face_encodings = face_recognition.face_encodings(
                rgb_image, 
                face_locations, 
                model=self.face_encoding_model
            )
            
            if not face_encodings:
                logger.warning("Could not extract face encoding")
                return None
                
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Face encoding extraction failed: {e}")
            return None
    
    async def extract_voice_print(self, audio_data: bytes) -> Optional[np.ndarray]:
        """
        Extract voice print from audio data.
        
        Args:
            audio_data: Audio bytes (WAV/WebM)
            
        Returns:
            Voice print array or None if extraction failed
        """
        if not self.voice_model_loaded:
            logger.error("Voice recognition not available")
            return None
            
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                
                # Load audio with librosa
                audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                
            # Check minimum duration
            duration = len(audio) / self.sample_rate
            if duration < self.min_voice_duration:
                logger.warning(f"Audio too short: {duration:.1f}s < {self.min_voice_duration}s")
                return None
            
            # Extract voice embedding using SpeechBrain
            loop = asyncio.get_event_loop()
            
            def extract_embedding():
                # Convert to tensor format expected by SpeechBrain
                import torch
                audio_tensor = torch.tensor(audio).unsqueeze(0)
                embedding = self.voice_model.encode_batch(audio_tensor)
                return embedding.squeeze().numpy()
            
            voice_print = await loop.run_in_executor(None, extract_embedding)
            
            logger.debug(f"Extracted voice print with shape: {voice_print.shape}")
            return voice_print
            
        except Exception as e:
            logger.error(f"Voice print extraction failed: {e}")
            return None
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Compare two face encodings and return similarity score.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            Distance score (lower = more similar)
        """
        try:
            distance = face_recognition.face_distance([encoding1], encoding2)[0]
            return float(distance)
            
        except Exception as e:
            logger.error(f"Face comparison failed: {e}")
            return 1.0  # Maximum distance (no match)
    
    def compare_voices(self, print1: np.ndarray, print2: np.ndarray) -> float:
        """
        Compare two voice prints and return similarity score.
        
        Args:
            print1: First voice print
            print2: Second voice print
            
        Returns:
            Distance score (lower = more similar)
        """
        try:
            # Calculate cosine distance
            from scipy.spatial.distance import cosine
            distance = cosine(print1, print2)
            return float(distance)
            
        except Exception as e:
            logger.error(f"Voice comparison failed: {e}")
            return 1.0  # Maximum distance (no match)
    
    async def register_user(
        self,
        user_id: str,
        face_image: bytes,
        voice_audio: bytes,
        pin: str
    ) -> Dict[str, Any]:
        """
        Register a new user with biometric data.
        
        Args:
            user_id: Unique user identifier
            face_image: Face image bytes
            voice_audio: Voice sample bytes
            pin: User's chosen PIN
            
        Returns:
            Registration result with user credentials
        """
        try:
            # Extract biometric features
            face_encoding = await self.extract_face_encoding(face_image)
            if face_encoding is None:
                return {"success": False, "error": "Could not extract face encoding"}
                
            voice_print = await self.extract_voice_print(voice_audio)
            if voice_print is None:
                return {"success": False, "error": "Could not extract voice print"}
            
            # Create user credentials
            biometric_hash, salt, vault_key = key_derivation_service.create_user_credentials(
                face_encoding.tobytes(),
                voice_print.tobytes(),
                pin
            )
            
            # Store user data
            self.registered_users[user_id] = {
                "biometric_hash": biometric_hash,
                "face_encoding": face_encoding.tolist(),  # Convert to JSON-serializable
                "voice_print": voice_print.tolist(),
                "salt": base64.b64encode(salt).decode('utf-8'),
                "registration_time": asyncio.get_event_loop().time()
            }
            
            logger.info(f"Successfully registered user: {user_id}")
            return {
                "success": True,
                "user_id": user_id,
                "biometric_hash": biometric_hash,
                "vault_key": key_derivation_service.encode_key_b64(vault_key)
            }
            
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def authenticate_user(
        self,
        face_image: bytes,
        voice_audio: bytes,
        pin: str
    ) -> Dict[str, Any]:
        """
        Authenticate user with biometric data and PIN.
        
        Args:
            face_image: Current face image
            voice_audio: Current voice sample
            pin: Provided PIN
            
        Returns:
            Authentication result with vault key if successful
        """
        try:
            # Extract current biometric features
            face_encoding = await self.extract_face_encoding(face_image)
            if face_encoding is None:
                return {"success": False, "error": "Could not extract face encoding"}
                
            voice_print = await self.extract_voice_print(voice_audio)
            if voice_print is None:
                return {"success": False, "error": "Could not extract voice print"}
            
            # Find matching user
            best_match = None
            best_score = float('inf')
            
            for user_id, user_data in self.registered_users.items():
                # Compare face
                stored_face = np.array(user_data["face_encoding"])
                face_distance = self.compare_faces(stored_face, face_encoding)
                
                # Compare voice
                stored_voice = np.array(user_data["voice_print"])
                voice_distance = self.compare_voices(stored_voice, voice_print)
                
                # Combined score (weighted average)
                combined_score = (face_distance * 0.6) + (voice_distance * 0.4)
                
                logger.debug(f"User {user_id}: face={face_distance:.3f}, voice={voice_distance:.3f}, combined={combined_score:.3f}")
                
                if combined_score < best_score:
                    best_score = combined_score
                    best_match = user_id
            
            # Check if match is good enough
            if (best_match is None or 
                best_score > max(self.face_tolerance, self.voice_similarity_threshold)):
                logger.warning(f"Authentication failed - no sufficient match (best score: {best_score:.3f})")
                return {"success": False, "error": "Biometric authentication failed"}
            
            # Authenticate with PIN and get vault key
            user_data = self.registered_users[best_match]
            salt = base64.b64decode(user_data["salt"].encode('utf-8'))
            
            vault_key = key_derivation_service.authenticate_user(
                face_encoding.tobytes(),
                voice_print.tobytes(),
                pin,
                salt
            )
            
            if vault_key is None:
                return {"success": False, "error": "PIN verification failed"}
            
            logger.info(f"Successfully authenticated user: {best_match}")
            return {
                "success": True,
                "user_id": best_match,
                "biometric_hash": user_data["biometric_hash"],
                "vault_key": key_derivation_service.encode_key_b64(vault_key),
                "match_score": best_score
            }
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_registered_users(self) -> List[str]:
        """Get list of registered user IDs."""
        return list(self.registered_users.keys())
    
    def remove_user(self, user_id: str) -> bool:
        """Remove a registered user."""
        if user_id in self.registered_users:
            del self.registered_users[user_id]
            logger.info(f"Removed user: {user_id}")
            return True
        return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Get current biometric capabilities."""
        return {
            "face_recognition": self.face_model_loaded,
            "voice_recognition": self.voice_model_loaded,
            "registration": self.face_model_loaded and self.voice_model_loaded,
            "authentication": self.face_model_loaded and self.voice_model_loaded
        }


# Global service instance
biometric_auth_service = BiometricAuthService()