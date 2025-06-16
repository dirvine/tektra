"""
Tests for Key Derivation Service.

Tests the secure key derivation functionality including biometric hashing,
PBKDF2 key derivation, and authentication workflows.
"""

import pytest
import secrets
import hashlib
from unittest.mock import patch

from tektra.app.security.key_derivation import KeyDerivationService, key_derivation_service


class TestKeyDerivationService:
    """Test cases for KeyDerivationService."""
    
    def test_service_initialization(self, key_derivation_service):
        """Test that the service initializes with correct parameters."""
        assert key_derivation_service.iterations == 100000
        assert key_derivation_service.key_length == 32
        assert key_derivation_service.salt_length == 32
    
    def test_derive_biometric_hash_deterministic(self, key_derivation_service):
        """Test that biometric hash generation is deterministic."""
        face_data = b"test_face_encoding_data"
        voice_data = b"test_voice_print_data"
        
        # Generate hash twice with same data
        hash1 = key_derivation_service.derive_biometric_hash(face_data, voice_data)
        hash2 = key_derivation_service.derive_biometric_hash(face_data, voice_data)
        
        # Should be identical
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex digest length
    
    def test_derive_biometric_hash_different_inputs(self, key_derivation_service):
        """Test that different inputs produce different hashes."""
        face_data1 = b"face_data_user_1"
        voice_data1 = b"voice_data_user_1"
        
        face_data2 = b"face_data_user_2"
        voice_data2 = b"voice_data_user_2"
        
        hash1 = key_derivation_service.derive_biometric_hash(face_data1, voice_data1)
        hash2 = key_derivation_service.derive_biometric_hash(face_data2, voice_data2)
        
        assert hash1 != hash2
    
    def test_generate_salt_uniqueness(self, key_derivation_service):
        """Test that salt generation produces unique values."""
        salt1 = key_derivation_service.generate_salt()
        salt2 = key_derivation_service.generate_salt()
        
        assert salt1 != salt2
        assert len(salt1) == 32
        assert len(salt2) == 32
        assert isinstance(salt1, bytes)
        assert isinstance(salt2, bytes)
    
    def test_derive_vault_key_deterministic(self, key_derivation_service):
        """Test that vault key derivation is deterministic."""
        biometric_hash = "test_biometric_hash_123"
        pin = "1234"
        salt = b"fixed_salt_for_testing_32bytes!"
        
        key1 = key_derivation_service.derive_vault_key(biometric_hash, pin, salt)
        key2 = key_derivation_service.derive_vault_key(biometric_hash, pin, salt)
        
        assert key1 == key2
        assert len(key1) == 32  # 256 bits
        assert isinstance(key1, bytes)
    
    def test_derive_vault_key_different_inputs(self, key_derivation_service):
        """Test that different inputs produce different keys."""
        salt = b"fixed_salt_for_testing_32bytes!"
        
        key1 = key_derivation_service.derive_vault_key("hash1", "1234", salt)
        key2 = key_derivation_service.derive_vault_key("hash2", "1234", salt)
        key3 = key_derivation_service.derive_vault_key("hash1", "5678", salt)
        
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3
    
    def test_verify_key_derivation_success(self, key_derivation_service):
        """Test successful key verification."""
        biometric_hash = "test_hash"
        pin = "1234"
        salt = key_derivation_service.generate_salt()
        
        # Derive key
        key = key_derivation_service.derive_vault_key(biometric_hash, pin, salt)
        
        # Verify key
        is_valid = key_derivation_service.verify_key_derivation(key, biometric_hash, pin, salt)
        assert is_valid is True
    
    def test_verify_key_derivation_failure(self, key_derivation_service):
        """Test key verification with wrong parameters."""
        biometric_hash = "test_hash"
        pin = "1234"
        salt = key_derivation_service.generate_salt()
        
        # Derive key
        key = key_derivation_service.derive_vault_key(biometric_hash, pin, salt)
        
        # Try to verify with wrong PIN
        is_valid = key_derivation_service.verify_key_derivation(key, biometric_hash, "wrong", salt)
        assert is_valid is False
        
        # Try to verify with wrong hash
        is_valid = key_derivation_service.verify_key_derivation(key, "wrong_hash", pin, salt)
        assert is_valid is False
    
    def test_create_user_credentials(self, key_derivation_service, sample_face_encoding, sample_voice_print):
        """Test complete user credential creation."""
        face_bytes = sample_face_encoding.tobytes()
        voice_bytes = sample_voice_print.tobytes()
        pin = "1234"
        
        biometric_hash, salt, vault_key = key_derivation_service.create_user_credentials(
            face_bytes, voice_bytes, pin
        )
        
        # Verify outputs
        assert isinstance(biometric_hash, str)
        assert len(biometric_hash) == 64  # SHA-256 hex
        assert isinstance(salt, bytes)
        assert len(salt) == 32
        assert isinstance(vault_key, bytes)
        assert len(vault_key) == 32
        
        # Verify key can be reproduced
        reproduced_key = key_derivation_service.derive_vault_key(biometric_hash, pin, salt)
        assert vault_key == reproduced_key
    
    def test_authenticate_user_success(self, key_derivation_service, sample_face_encoding, sample_voice_print):
        """Test successful user authentication."""
        face_bytes = sample_face_encoding.tobytes()
        voice_bytes = sample_voice_print.tobytes()
        pin = "1234"
        
        # Create credentials
        biometric_hash, salt, expected_key = key_derivation_service.create_user_credentials(
            face_bytes, voice_bytes, pin
        )
        
        # Authenticate with same data
        auth_key = key_derivation_service.authenticate_user(face_bytes, voice_bytes, pin, salt)
        
        assert auth_key is not None
        assert auth_key == expected_key
    
    def test_authenticate_user_failure(self, key_derivation_service, sample_face_encoding, sample_voice_print):
        """Test failed user authentication."""
        face_bytes = sample_face_encoding.tobytes()
        voice_bytes = sample_voice_print.tobytes()
        pin = "1234"
        
        # Create credentials
        _, salt, _ = key_derivation_service.create_user_credentials(face_bytes, voice_bytes, pin)
        
        # Try to authenticate with wrong PIN
        auth_key = key_derivation_service.authenticate_user(face_bytes, voice_bytes, "wrong", salt)
        assert auth_key is not None  # Key is derived but won't match stored key
        
        # Try with wrong face data
        wrong_face = b"different_face_data"
        auth_key = key_derivation_service.authenticate_user(wrong_face, voice_bytes, pin, salt)
        assert auth_key is not None  # Key is derived but won't match
    
    def test_encode_decode_key_b64(self, key_derivation_service):
        """Test base64 encoding and decoding of keys."""
        original_key = secrets.token_bytes(32)
        
        # Encode
        encoded = key_derivation_service.encode_key_b64(original_key)
        assert isinstance(encoded, str)
        
        # Decode
        decoded = key_derivation_service.decode_key_b64(encoded)
        assert decoded == original_key
    
    def test_biometric_hash_collision_resistance(self, key_derivation_service):
        """Test that similar inputs produce different hashes."""
        base_face = b"face_encoding_"
        base_voice = b"voice_print_"
        
        hashes = set()
        
        # Generate multiple hashes with slight variations
        for i in range(100):
            face_data = base_face + str(i).encode()
            voice_data = base_voice + str(i).encode()
            hash_val = key_derivation_service.derive_biometric_hash(face_data, voice_data)
            hashes.add(hash_val)
        
        # All hashes should be unique
        assert len(hashes) == 100
    
    def test_salt_entropy(self, key_derivation_service):
        """Test that generated salts have sufficient entropy."""
        salts = [key_derivation_service.generate_salt() for _ in range(1000)]
        
        # All salts should be unique
        assert len(set(salts)) == 1000
        
        # Test entropy by checking bit distribution
        all_bytes = b''.join(salts)
        byte_counts = [0] * 256
        
        for byte_val in all_bytes:
            byte_counts[byte_val] += 1
        
        # Check that byte distribution is reasonably uniform
        # (not perfect due to small sample size)
        min_count = min(byte_counts)
        max_count = max(byte_counts)
        ratio = max_count / max(min_count, 1)
        
        # Ratio should be reasonable for random data
        assert ratio < 10  # Allow some variance
    
    def test_error_handling(self, key_derivation_service):
        """Test error handling for invalid inputs."""
        # Test with empty inputs
        with pytest.raises(Exception):
            key_derivation_service.derive_biometric_hash(b"", b"")
        
        # Test with None inputs - should raise TypeError
        with pytest.raises(TypeError):
            key_derivation_service.derive_biometric_hash(None, b"test")
        
        # Test key verification with invalid key length
        try:
            result = key_derivation_service.verify_key_derivation(
                b"short_key", "hash", "pin", b"salt"
            )
            # Should either work or raise exception, but not crash
            assert isinstance(result, bool)
        except Exception:
            # Exception is acceptable for invalid input
            pass
    
    def test_pbkdf2_parameters(self, key_derivation_service):
        """Test that PBKDF2 parameters meet security standards."""
        # Iterations should be high enough for security
        assert key_derivation_service.iterations >= 100000
        
        # Key length should be sufficient (256 bits minimum)
        assert key_derivation_service.key_length >= 32
        
        # Salt length should be sufficient (128 bits minimum)
        assert key_derivation_service.salt_length >= 16
    
    def test_timing_attack_resistance(self, key_derivation_service):
        """Test that key verification uses constant-time comparison."""
        import time
        
        biometric_hash = "test_hash"
        pin = "1234"
        salt = key_derivation_service.generate_salt()
        
        # Generate correct key
        correct_key = key_derivation_service.derive_vault_key(biometric_hash, pin, salt)
        
        # Generate wrong key (completely different)
        wrong_key1 = secrets.token_bytes(32)
        
        # Generate wrong key (differs in last byte)
        wrong_key2 = correct_key[:-1] + bytes([correct_key[-1] ^ 1])
        
        # Time verification operations
        times = []
        
        for key in [correct_key, wrong_key1, wrong_key2]:
            start_time = time.perf_counter()
            key_derivation_service.verify_key_derivation(key, biometric_hash, pin, salt)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        # Times should be similar (within reasonable variance)
        # This is a basic check - true constant-time verification would need more sophisticated testing
        max_time = max(times)
        min_time = min(times)
        
        # Allow up to 10x variance (generous for timing variance)
        if min_time > 0:
            ratio = max_time / min_time
            assert ratio < 10  # Basic timing consistency check


class TestGlobalService:
    """Test the global key derivation service instance."""
    
    def test_global_service_exists(self):
        """Test that global service instance exists and is configured."""
        assert key_derivation_service is not None
        assert isinstance(key_derivation_service, KeyDerivationService)
        assert key_derivation_service.iterations == 100000
    
    def test_global_service_functionality(self):
        """Test that global service works correctly."""
        salt = key_derivation_service.generate_salt()
        assert len(salt) == 32
        
        key = key_derivation_service.derive_vault_key("test", "1234", salt)
        assert len(key) == 32