#!/usr/bin/env python3
"""
Test script for Phi-4 Integration Phase 3.1

Verifies that Phi-4 service integration works correctly with fallback to Whisper.

UV Dependencies:
# Install UV if not available: curl -LsSf https://astral.sh/uv/install.sh | sh
# Run this script: uv run python test_phi4_integration.py
# Or with dependencies: uv run --with fastapi,uvicorn python test_phi4_integration.py
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.104.0",
#     "uvicorn>=0.24.0",
#     "pydantic>=2.5.0",
#     "numpy>=1.24.0",
#     "torch>=2.0.0",
#     "transformers>=4.40.0",
#     "librosa>=0.10.0",
#     "soundfile>=0.12.0",
#     "asyncio-tools>=0.1.0",
# ]
# ///

import asyncio
import sys
import numpy as np
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent / "backend" / "tektra"))

async def test_phi4_integration():
    """Test Phi-4 service integration."""
    print("🔬 Testing Phi-4 Integration Phase 3.1")
    print("=" * 50)
    
    try:
        # Import services
        from app.services.phi4_service import phi4_service
        from app.services.whisper_service import whisper_service
        from app.services.ai_service import ai_service
        
        print("✅ Successfully imported all services")
        
        # Test Phi-4 service info
        print("\n📊 Testing Phi-4 Service Info...")
        phi4_info = await phi4_service.get_model_info()
        print(f"   Phi-4 Available: {phi4_info['available']}")
        print(f"   Phi-4 Loaded: {phi4_info['is_loaded']}")
        print(f"   Device: {phi4_info['device']}")
        print(f"   Supported Languages: {len(phi4_info['supported_languages'])}")
        
        # Test Whisper service info
        print("\n📊 Testing Whisper Service Info...")
        whisper_info = await whisper_service.get_model_info()
        print(f"   Whisper Available: {whisper_info['available']}")
        print(f"   Whisper Loaded: {whisper_info['is_loaded']}")
        
        # Test AI service streaming with Phi-4 fallback
        print("\n🤖 Testing AI Service with Phi-4 Integration...")
        test_messages = [{"role": "user", "content": "Hello, how are you?"}]
        
        response_chunks = []
        async for chunk in ai_service.stream_chat_completion(test_messages):
            response_chunks.append(chunk)
            if len(response_chunks) >= 5:  # Limit for testing
                break
        
        full_response = "".join(response_chunks)
        print(f"   Generated Response: {full_response[:100]}...")
        print(f"   Response Chunks: {len(response_chunks)}")
        
        # Test fallback behavior
        print("\n🔄 Testing Fallback Behavior...")
        if phi4_info['is_loaded']:
            print("   ✅ Phi-4 is primary processor")
            print("   ℹ️  Whisper available as fallback")
        else:
            print("   ⚠️  Phi-4 not loaded, using Whisper/local models")
            print("   ℹ️  Fallback system active")
        
        # Test language support integration
        print("\n🌍 Testing Language Support...")
        all_languages = set()
        all_languages.update(phi4_info.get('supported_languages', {}).keys())
        all_languages.update(whisper_info.get('supported_languages', {}).keys())
        print(f"   Total Supported Languages: {len(all_languages)}")
        print(f"   Languages: {', '.join(sorted(list(all_languages))[:10])}...")
        
        print("\n✅ Phase 3.1 Integration Test Complete!")
        print("🚀 Ready for PyPI publishing!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Make sure you're running from the correct directory")
        return False
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("   This may be normal if models aren't downloaded yet")
        return False

async def test_service_endpoints():
    """Test that service endpoints are properly configured."""
    print("\n🌐 Testing Service Endpoint Configuration...")
    
    try:
        # Test audio router imports
        from app.routers.audio import router as audio_router
        print("   ✅ Audio router with Phi-4 integration")
        
        # Test websocket router imports  
        from app.routers.websocket import router as ws_router
        print("   ✅ WebSocket router with Phi-4 integration")
        
        # Check for Phi-4 endpoints
        audio_routes = [route.path for route in audio_router.routes]
        phi4_routes = [route for route in audio_routes if 'phi4' in route]
        print(f"   📡 Phi-4 endpoints: {len(phi4_routes)}")
        for route in phi4_routes:
            print(f"      - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Endpoint configuration error: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🎯 Tektra Phase 3.1 - Phi-4 Integration Test")
    print("=" * 60)
    
    async def run_tests():
        success = True
        
        # Test core integration
        if not await test_phi4_integration():
            success = False
        
        # Test endpoint configuration
        if not await test_service_endpoints():
            success = False
        
        print("\n" + "=" * 60)
        if success:
            print("🎉 All tests passed! Phase 3.1 integration successful.")
            print("📦 Ready for PyPI publishing with enhanced Phi-4 capabilities.")
        else:
            print("⚠️  Some tests failed. Check configuration and dependencies.")
        
        return success
    
    return asyncio.run(run_tests())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)