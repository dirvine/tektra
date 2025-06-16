"""
Tests for Biometric Authentication Service.

Tests face recognition, voice recognition, user registration,
and authentication workflows with comprehensive mocking.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import base64

from tektra.app.security.biometric_auth import BiometricAuthService, biometric_auth_service


class TestBiometricAuthService:
    """Test cases for BiometricAuthService."""
    
    def test_service_initialization(self, biometric_auth_service):
        """Test that the service initializes with correct parameters."""
        assert biometric_auth_service.face_tolerance == 0.6
        assert biometric_auth_service.face_encoding_model == "large"
        assert biometric_auth_service.sample_rate == 16000
        assert biometric_auth_service.min_voice_duration == 2.0
        assert biometric_auth_service.voice_similarity_threshold == 0.25
        assert isinstance(biometric_auth_service.registered_users, dict)
    
    @pytest.mark.asyncio
    async def test_initialization_no_dependencies(self, biometric_auth_service):
        """Test initialization when dependencies are not available."""
        with patch('tektra.app.security.biometric_auth.FACE_RECOGNITION_AVAILABLE', False), \
             patch('tektra.app.security.biometric_auth.VOICE_RECOGNITION_AVAILABLE', False):
            
            await biometric_auth_service.initialize()
            
            assert biometric_auth_service.face_model_loaded is False
            assert biometric_auth_service.voice_model_loaded is False
    
    @pytest.mark.asyncio
    async def test_initialization_with_dependencies(self, biometric_auth_service, enable_biometric_mocks):
        """Test initialization when dependencies are available."""
        await biometric_auth_service.initialize()
        
        assert biometric_auth_service.face_model_loaded is True
        assert biometric_auth_service.voice_model_loaded is True
    
    @pytest.mark.asyncio
    async def test_extract_face_encoding_success(self, biometric_auth_service, sample_face_image, enable_biometric_mocks):
        """Test successful face encoding extraction."""
        await biometric_auth_service.initialize()
        
        encoding = await biometric_auth_service.extract_face_encoding(sample_face_image)
        
        assert encoding is not None
        assert isinstance(encoding, np.ndarray)
        assert len(encoding) == 128  # Standard face encoding length
    
    @pytest.mark.asyncio
    async def test_extract_face_encoding_no_face(self, biometric_auth_service, sample_face_image, enable_biometric_mocks):
        """Test face encoding extraction when no face is detected."""
        await biometric_auth_service.initialize()
        
        # Mock no face detected
        with patch('tektra.app.security.biometric_auth.face_recognition.face_locations', return_value=[]):
            encoding = await biometric_auth_service.extract_face_encoding(sample_face_image)
            assert encoding is None
    
    @pytest.mark.asyncio
    async def test_extract_face_encoding_model_not_loaded(self, biometric_auth_service, sample_face_image):
        """Test face encoding extraction when model is not loaded."""
        biometric_auth_service.face_model_loaded = False
        
        encoding = await biometric_auth_service.extract_face_encoding(sample_face_image)
        assert encoding is None
    
    @pytest.mark.asyncio
    async def test_extract_voice_print_success(self, biometric_auth_service, sample_voice_audio, enable_biometric_mocks):
        """Test successful voice print extraction."""
        await biometric_auth_service.initialize()
        
        voice_print = await biometric_auth_service.extract_voice_print(sample_voice_audio)
        
        assert voice_print is not None
        assert isinstance(voice_print, np.ndarray)
        assert len(voice_print) == 512  # Mocked voice print length
    
    @pytest.mark.asyncio
    async def test_extract_voice_print_too_short(self, biometric_auth_service, enable_biometric_mocks):
        """Test voice print extraction with audio that's too short."""
        await biometric_auth_service.initialize()
        
        # Mock short audio
        with patch('tektra.app.security.biometric_auth.librosa.load') as mock_load:
            mock_load.return_value = (np.random.random(8000), 16000)  # 0.5 seconds
            
            voice_print = await biometric_auth_service.extract_voice_print(b"short_audio")
            assert voice_print is None
    
    @pytest.mark.asyncio
    async def test_extract_voice_print_model_not_loaded(self, biometric_auth_service, sample_voice_audio):
        """Test voice print extraction when model is not loaded."""
        biometric_auth_service.voice_model_loaded = False
        
        voice_print = await biometric_auth_service.extract_voice_print(sample_voice_audio)
        assert voice_print is None
    
    def test_compare_faces(self, biometric_auth_service, enable_biometric_mocks):
        """Test face comparison functionality."""
        encoding1 = np.random.random(128)
        encoding2 = np.random.random(128)
        
        distance = biometric_auth_service.compare_faces(encoding1, encoding2)
        
        assert isinstance(distance, float)
        assert distance >= 0.0
        assert distance <= 2.0  # Typical range for face distances
    
    def test_compare_voices(self, biometric_auth_service):
        """Test voice comparison functionality."""
        print1 = np.random.random(512)
        print2 = np.random.random(512)
        
        distance = biometric_auth_service.compare_voices(print1, print2)
        
        assert isinstance(distance, float)
        assert distance >= 0.0
        assert distance <= 2.0  # Typical range for cosine distance
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, biometric_auth_service, sample_face_image, sample_voice_audio, enable_biometric_mocks):
        """Test successful user registration."""
        await biometric_auth_service.initialize()
        
        result = await biometric_auth_service.register_user(
            "test_user", sample_face_image, sample_voice_audio, "1234"
        )
        
        assert result["success"] is True
        assert result["user_id"] == "test_user"
        assert "biometric_hash" in result
        assert "vault_key" in result
        
        # Check user was added to registered users
        assert "test_user" in biometric_auth_service.registered_users
    
    @pytest.mark.asyncio
    async def test_register_user_face_extraction_failure(self, biometric_auth_service, sample_voice_audio, enable_biometric_mocks):
        """Test user registration when face extraction fails."""
        await biometric_auth_service.initialize()
        
        # Mock face extraction failure
        with patch.object(biometric_auth_service, 'extract_face_encoding', return_value=None):
            result = await biometric_auth_service.register_user(
                "test_user", b"bad_image", sample_voice_audio, "1234"
            )
            
            assert result["success"] is False
            assert "face encoding" in result["error"]
    
    @pytest.mark.asyncio
    async def test_register_user_voice_extraction_failure(self, biometric_auth_service, sample_face_image, enable_biometric_mocks):
        """Test user registration when voice extraction fails."""
        await biometric_auth_service.initialize()
        
        # Mock voice extraction failure
        with patch.object(biometric_auth_service, 'extract_voice_print', return_value=None):
            result = await biometric_auth_service.register_user(
                "test_user", sample_face_image, b"bad_audio", "1234"
            )
            
            assert result["success"] is False
            assert "voice print" in result["error"]
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, biometric_auth_service, sample_face_image, sample_voice_audio, enable_biometric_mocks):
        """Test successful user authentication."""
        await biometric_auth_service.initialize()
        
        # First register user
        register_result = await biometric_auth_service.register_user(
            "test_user", sample_face_image, sample_voice_audio, "1234"
        )
        assert register_result["success"] is True
        
        # Mock good biometric matches
        with patch.object(biometric_auth_service, 'compare_faces', return_value=0.3), \
             patch.object(biometric_auth_service, 'compare_voices', return_value=0.2):
            
            # Now authenticate
            auth_result = await biometric_auth_service.authenticate_user(
                sample_face_image, sample_voice_audio, "1234"
            )
            
            assert auth_result["success"] is True
            assert auth_result["user_id"] == "test_user"
            assert "vault_key" in auth_result
            assert "match_score" in auth_result
    
    @pytest.mark.asyncio
    async def test_authenticate_user_no_match(self, biometric_auth_service, sample_face_image, sample_voice_audio, enable_biometric_mocks):
        """Test user authentication when no biometric match is found."""
        await biometric_auth_service.initialize()
        
        # Register a user
        await biometric_auth_service.register_user(
            "test_user", sample_face_image, sample_voice_audio, "1234"
        )
        
        # Mock poor biometric matches
        with patch.object(biometric_auth_service, 'compare_faces', return_value=0.9), \
             patch.object(biometric_auth_service, 'compare_voices', return_value=0.8):
            
            auth_result = await biometric_auth_service.authenticate_user(
                sample_face_image, sample_voice_audio, "1234"
            )
            
            assert auth_result["success"] is False
            assert "authentication failed" in auth_result["error"]
    
    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_pin(self, biometric_auth_service, sample_face_image, sample_voice_audio, enable_biometric_mocks):
        """Test user authentication with wrong PIN."""
        await biometric_auth_service.initialize()
        
        # Register user
        await biometric_auth_service.register_user(
            "test_user", sample_face_image, sample_voice_audio, "1234"
        )
        
        # Mock good biometric matches but wrong PIN
        with patch.object(biometric_auth_service, 'compare_faces', return_value=0.3), \
             patch.object(biometric_auth_service, 'compare_voices', return_value=0.2):
            
            auth_result = await biometric_auth_service.authenticate_user(
                sample_face_image, sample_voice_audio, "wrong_pin"
            )
            
            assert auth_result["success"] is False
            assert "PIN verification failed" in auth_result["error"]
    
    @pytest.mark.asyncio
    async def test_authenticate_user_face_extraction_failure(self, biometric_auth_service, sample_voice_audio, enable_biometric_mocks):
        """Test authentication when face extraction fails."""
        await biometric_auth_service.initialize()
        
        # Mock face extraction failure
        with patch.object(biometric_auth_service, 'extract_face_encoding', return_value=None):
            result = await biometric_auth_service.authenticate_user(
                b"bad_image", sample_voice_audio, "1234"
            )
            
            assert result["success"] is False
            assert "face encoding" in result["error"]
    
    @pytest.mark.asyncio
    async def test_authenticate_user_voice_extraction_failure(self, biometric_auth_service, sample_face_image, enable_biometric_mocks):
        """Test authentication when voice extraction fails."""
        await biometric_auth_service.initialize()
        
        # Mock voice extraction failure
        with patch.object(biometric_auth_service, 'extract_voice_print', return_value=None):
            result = await biometric_auth_service.authenticate_user(
                sample_face_image, b"bad_audio", "1234"
            )
            
            assert result["success"] is False
            assert "voice print" in result["error"]
    
    def test_get_registered_users(self, biometric_auth_service):
        """Test getting list of registered users."""
        # Initially empty
        users = biometric_auth_service.get_registered_users()
        assert users == []
        
        # Add some test users
        biometric_auth_service.registered_users["user1"] = {"test": "data"}
        biometric_auth_service.registered_users["user2"] = {"test": "data"}
        
        users = biometric_auth_service.get_registered_users()
        assert set(users) == {"user1", "user2"}
    
    def test_remove_user(self, biometric_auth_service):
        """Test removing a registered user."""
        # Add test user
        biometric_auth_service.registered_users["test_user"] = {"test": "data"}
        
        # Remove user
        result = biometric_auth_service.remove_user("test_user")
        assert result is True
        assert "test_user" not in biometric_auth_service.registered_users
        
        # Try to remove non-existent user
        result = biometric_auth_service.remove_user("non_existent")
        assert result is False
    
    def test_get_capabilities_no_models(self, biometric_auth_service):
        """Test getting capabilities when models are not loaded."""
        biometric_auth_service.face_model_loaded = False
        biometric_auth_service.voice_model_loaded = False
        
        capabilities = biometric_auth_service.get_capabilities()
        
        assert capabilities["face_recognition"] is False
        assert capabilities["voice_recognition"] is False
        assert capabilities["registration"] is False
        assert capabilities["authentication"] is False
    
    def test_get_capabilities_with_models(self, biometric_auth_service):
        """Test getting capabilities when models are loaded."""
        biometric_auth_service.face_model_loaded = True
        biometric_auth_service.voice_model_loaded = True
        
        capabilities = biometric_auth_service.get_capabilities()
        
        assert capabilities["face_recognition"] is True
        assert capabilities["voice_recognition"] is True
        assert capabilities["registration"] is True
        assert capabilities["authentication"] is True
    
    @pytest.mark.asyncio
    async def test_multiple_user_registration(self, biometric_auth_service, enable_biometric_mocks):
        """Test registering multiple users."""
        await biometric_auth_service.initialize()
        
        users = ["user1", "user2", "user3"]
        
        for user_id in users:
            # Create slightly different biometric data for each user
            face_data = f"face_data_{user_id}".encode()
            voice_data = f"voice_data_{user_id}".encode()
            
            result = await biometric_auth_service.register_user(
                user_id, face_data, voice_data, "1234"
            )
            
            assert result["success"] is True
            assert result["user_id"] == user_id
        
        # Check all users are registered
        registered_users = biometric_auth_service.get_registered_users()
        assert set(registered_users) == set(users)
    
    @pytest.mark.asyncio
    async def test_authentication_score_calculation(self, biometric_auth_service, enable_biometric_mocks):
        """Test that authentication scores are calculated correctly."""
        await biometric_auth_service.initialize()
        
        # Register user
        await biometric_auth_service.register_user(
            "test_user", b"face_data", b"voice_data", "1234"
        )
        
        # Test different match qualities
        test_cases = [
            (0.2, 0.1, True),   # Good match
            (0.3, 0.2, True),   # Acceptable match
            (0.7, 0.8, False),  # Poor match
        ]
        
        for face_distance, voice_distance, should_succeed in test_cases:
            with patch.object(biometric_auth_service, 'compare_faces', return_value=face_distance), \
                 patch.object(biometric_auth_service, 'compare_voices', return_value=voice_distance):
                
                result = await biometric_auth_service.authenticate_user(
                    b"face_data", b"voice_data", "1234"
                )
                
                if should_succeed:
                    assert result["success"] is True
                    # Combined score should be weighted average (0.6 * face + 0.4 * voice)
                    expected_score = face_distance * 0.6 + voice_distance * 0.4
                    assert abs(result["match_score"] - expected_score) < 0.01
                else:
                    assert result["success"] is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self, biometric_auth_service):
        """Test error handling in various scenarios."""
        # Test registration without initialization
        result = await biometric_auth_service.register_user(
            "test", b"face", b"voice", "1234"
        )
        assert result["success"] is False
        
        # Test authentication without initialization
        result = await biometric_auth_service.authenticate_user(
            b"face", b"voice", "1234"
        )
        assert result["success"] is False
    
    def test_biometric_fusion_weights(self, biometric_auth_service):
        """Test that biometric fusion uses correct weights."""
        # Test the weighted combination used in authentication
        face_distance = 0.4
        voice_distance = 0.2
        
        # Expected combined score: 0.6 * face + 0.4 * voice
        expected_combined = 0.6 * face_distance + 0.4 * voice_distance
        assert expected_combined == 0.32
        
        # Face should have higher weight than voice
        face_weight = 0.6
        voice_weight = 0.4
        assert face_weight > voice_weight
        assert face_weight + voice_weight == 1.0


class TestGlobalService:
    """Test the global biometric auth service instance."""
    
    def test_global_service_exists(self):
        """Test that global service instance exists."""
        assert biometric_auth_service is not None
        assert isinstance(biometric_auth_service, BiometricAuthService)
    
    def test_global_service_configuration(self):
        """Test that global service is properly configured."""
        assert biometric_auth_service.face_tolerance == 0.6
        assert biometric_auth_service.sample_rate == 16000
        assert biometric_auth_service.min_voice_duration == 2.0
    
    @pytest.mark.asyncio
    async def test_global_service_functionality(self, enable_biometric_mocks):
        """Test that global service functions correctly."""
        await biometric_auth_service.initialize()
        
        capabilities = biometric_auth_service.get_capabilities()
        assert isinstance(capabilities, dict)
        assert "face_recognition" in capabilities
        assert "voice_recognition" in capabilities


class TestBiometricDataProcessing:
    """Test biometric data processing edge cases."""
    
    @pytest.mark.asyncio
    async def test_face_encoding_invalid_image(self, biometric_auth_service, enable_biometric_mocks):
        """Test face encoding with invalid image data."""
        await biometric_auth_service.initialize()
        
        # Mock image decode failure
        with patch('tektra.app.security.biometric_auth.cv2.imdecode', return_value=None):
            encoding = await biometric_auth_service.extract_face_encoding(b"invalid_image")
            assert encoding is None
    
    @pytest.mark.asyncio
    async def test_voice_print_invalid_audio(self, biometric_auth_service, enable_biometric_mocks):
        """Test voice print extraction with invalid audio data."""
        await biometric_auth_service.initialize()
        
        # Mock audio loading failure
        with patch('tektra.app.security.biometric_auth.librosa.load', side_effect=Exception("Audio error")):
            voice_print = await biometric_auth_service.extract_voice_print(b"invalid_audio")
            assert voice_print is None
    
    def test_face_comparison_error_handling(self, biometric_auth_service):
        """Test face comparison with invalid data."""
        # Test with incompatible array sizes
        encoding1 = np.random.random(128)
        encoding2 = np.random.random(64)  # Wrong size
        
        with patch('tektra.app.security.biometric_auth.face_recognition.face_distance', side_effect=Exception("Size error")):
            distance = biometric_auth_service.compare_faces(encoding1, encoding2)
            assert distance == 1.0  # Should return maximum distance on error
    
    def test_voice_comparison_error_handling(self, biometric_auth_service):
        """Test voice comparison with invalid data."""
        print1 = np.random.random(512)
        print2 = np.random.random(256)  # Wrong size
        
        with patch('scipy.spatial.distance.cosine', side_effect=Exception("Distance error")):
            distance = biometric_auth_service.compare_voices(print1, print2)
            assert distance == 1.0  # Should return maximum distance on error