"""
Test configuration and fixtures for Tektra security tests.
"""

import pytest
import asyncio
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import base64
import io
from typing import Dict, Any

# Import security services for testing
from tektra.app.security.key_derivation import KeyDerivationService
from tektra.app.security.vault_manager import VaultManager
from tektra.app.security.biometric_auth import BiometricAuthService
from tektra.app.security.anonymization import AnonymizationService
from tektra.app.services.security_service import SecurityService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def key_derivation_service():
    """Create a fresh key derivation service for testing."""
    return KeyDerivationService()


@pytest.fixture
async def vault_manager(temp_dir):
    """Create a vault manager with temporary directory."""
    vault_mgr = VaultManager()
    vault_mgr.vault_directory = temp_dir / "vaults"
    await vault_mgr.initialize()
    return vault_mgr


@pytest.fixture
def biometric_auth_service():
    """Create a biometric auth service for testing."""
    return BiometricAuthService()


@pytest.fixture
def anonymization_service():
    """Create an anonymization service for testing."""
    return AnonymizationService()


@pytest.fixture
async def security_service(temp_dir):
    """Create a security service with temporary storage."""
    sec_service = SecurityService()
    
    # Mock the vault manager to use temp directory
    with patch('tektra.app.services.security_service.vault_manager') as mock_vault:
        mock_vault.vault_directory = temp_dir / "vaults"
        mock_vault.initialize = Mock(return_value=None)
        mock_vault.create_vault = Mock(return_value=True)
        mock_vault.open_vault = Mock(return_value={"user_id": "test_user"})
        mock_vault.close_vault = Mock(return_value=True)
        
        yield sec_service


@pytest.fixture
def sample_face_encoding():
    """Generate a sample face encoding for testing."""
    # Create a realistic face encoding (128-dimensional vector)
    np.random.seed(42)  # For reproducible tests
    return np.random.random(128).astype(np.float64)


@pytest.fixture
def sample_voice_print():
    """Generate a sample voice print for testing."""
    # Create a realistic voice print (512-dimensional vector)
    np.random.seed(123)  # For reproducible tests
    return np.random.random(512).astype(np.float32)


@pytest.fixture
def sample_face_image():
    """Generate a sample face image as bytes."""
    # Create a simple test image
    from PIL import Image
    
    # Create a 100x100 RGB image
    img = Image.new('RGB', (100, 100), color='red')
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


@pytest.fixture
def sample_voice_audio():
    """Generate a sample voice audio as bytes."""
    # Create a simple test audio file
    import wave
    
    # Create a 1-second audio file at 16kHz
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    
    # Generate a simple sine wave
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples, False)
    audio_data = np.sin(2 * np.pi * frequency * t)
    
    # Convert to 16-bit integers
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Convert to WAV bytes
    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    return wav_bytes.getvalue()


@pytest.fixture
def sample_user_credentials():
    """Generate sample user credentials for testing."""
    return {
        "user_id": "test_user_123",
        "pin": "1234",
        "biometric_hash": "abc123def456",
        "salt": b"test_salt_16bytes"
    }


@pytest.fixture
def sample_conversation_data():
    """Generate sample conversation data for testing."""
    return {
        "conversation_id": "conv_123",
        "title": "Test Conversation",
        "messages": [
            {
                "id": "msg_1",
                "role": "user",
                "content": "Hello, this is a test message",
                "timestamp": "2024-01-01T12:00:00Z",
                "type": "text"
            },
            {
                "id": "msg_2",
                "role": "assistant", 
                "content": "Hello! I'm here to help you with your test.",
                "timestamp": "2024-01-01T12:00:01Z",
                "type": "text"
            }
        ]
    }


@pytest.fixture
def sample_pii_text():
    """Generate sample text with PII for anonymization testing."""
    return """
    Hi, my name is John Doe and my email is john.doe@example.com.
    You can reach me at +1-555-123-4567 or visit our lab at 192.168.1.100.
    My project SECRET_PROJECT_X uses API key abc123def456ghi789.
    The file is located at /home/john/secret_research/data.csv.
    We're working with robot ARM-7543 in our internal.lab domain.
    """


@pytest.fixture
def mock_face_recognition():
    """Mock face recognition functionality for testing."""
    with patch('tektra.app.security.biometric_auth.face_recognition') as mock_fr:
        # Mock face detection
        mock_fr.face_locations.return_value = [(50, 150, 150, 50)]  # top, right, bottom, left
        
        # Mock face encoding
        test_encoding = np.random.random(128).astype(np.float64)
        mock_fr.face_encodings.return_value = [test_encoding]
        
        # Mock face comparison
        mock_fr.face_distance.return_value = [0.3]  # Good match
        
        yield mock_fr


@pytest.fixture
def mock_voice_recognition():
    """Mock voice recognition functionality for testing."""
    with patch('tektra.app.security.biometric_auth.SpeakerRecognition') as mock_sr:
        # Create mock model
        mock_model = Mock()
        mock_sr.from_hparams.return_value = mock_model
        
        # Mock voice embedding
        mock_embedding = Mock()
        mock_embedding.squeeze.return_value.numpy.return_value = np.random.random(512)
        mock_model.encode_batch.return_value = mock_embedding
        
        yield mock_sr


@pytest.fixture
def mock_cv2():
    """Mock OpenCV functionality for testing."""
    with patch('tektra.app.security.biometric_auth.cv2') as mock_cv2:
        # Mock image decode
        mock_cv2.imdecode.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.COLOR_BGR2RGB = 4  # Constant value
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        
        yield mock_cv2


@pytest.fixture
def mock_librosa():
    """Mock librosa functionality for testing."""
    with patch('tektra.app.security.biometric_auth.librosa') as mock_librosa:
        # Mock audio loading
        sample_rate = 16000
        duration = 3.0
        mock_audio = np.random.random(int(sample_rate * duration))
        mock_librosa.load.return_value = (mock_audio, sample_rate)
        mock_librosa.util.normalize.return_value = mock_audio
        
        yield mock_librosa


@pytest.fixture
def enable_biometric_mocks(mock_face_recognition, mock_voice_recognition, mock_cv2, mock_librosa):
    """Enable all biometric mocks for comprehensive testing."""
    with patch('tektra.app.security.biometric_auth.FACE_RECOGNITION_AVAILABLE', True), \
         patch('tektra.app.security.biometric_auth.VOICE_RECOGNITION_AVAILABLE', True):
        yield


# Test data generators
def generate_test_vault_data(user_id: str = "test_user") -> Dict[str, Any]:
    """Generate test vault data structure."""
    return {
        "version": "1.0",
        "user_id": user_id,
        "biometric_hash": "test_hash_123",
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:00:00Z",
        "conversations": [],
        "preferences": {},
        "knowledge_base": {},
        "metadata": {
            "total_conversations": 0,
            "total_messages": 0,
            "last_access": None
        }
    }


def encode_test_data(data: bytes) -> str:
    """Encode test data as base64 string."""
    return base64.b64encode(data).decode('utf-8')


def create_test_session_data(user_id: str = "test_user") -> Dict[str, Any]:
    """Create test session data."""
    from datetime import datetime, timezone
    
    return {
        "user_id": user_id,
        "biometric_hash": "test_hash_123",
        "vault_key": b"test_vault_key_32_bytes_long!!",
        "created_at": datetime.now(timezone.utc),
        "last_activity": datetime.now(timezone.utc),
        "match_score": 0.15
    }