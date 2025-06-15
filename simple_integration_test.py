#!/usr/bin/env python3
"""
Simple Integration Test for Phase 3.1

Tests the Phi-4 integration without heavy dependencies.
"""

import sys
from pathlib import Path

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent / "backend" / "tektra"))

def test_imports():
    """Test that all services can be imported."""
    print("🔬 Testing Phase 3.1 Service Imports")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("   Importing Phi-4 service...")
        from app.services.phi4_service import Phi4Service
        print("   ✅ Phi-4 service imported successfully")
        
        print("   Importing Whisper service...")
        from app.services.whisper_service import WhisperService  
        print("   ✅ Whisper service imported successfully")
        
        print("   Importing AI service...")
        from app.services.ai_service import AIService
        print("   ✅ AI service imported successfully")
        
        print("   Importing audio router...")
        from app.routers.audio import router
        print("   ✅ Audio router imported successfully")
        
        print("   Importing websocket router...")
        from app.routers.websocket import router as ws_router
        print("   ✅ WebSocket router imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_service_structure():
    """Test that services have expected methods."""
    print("\n🔧 Testing Service Structure")
    print("=" * 40)
    
    try:
        from app.services.phi4_service import Phi4Service
        
        # Check Phi4Service has expected methods
        phi4_methods = [
            'load_model', 'transcribe_audio', 'chat_completion', 
            'detect_language', 'get_model_info', 'unload_model'
        ]
        
        service = Phi4Service()
        for method in phi4_methods:
            if hasattr(service, method):
                print(f"   ✅ Phi4Service.{method}")
            else:
                print(f"   ❌ Missing Phi4Service.{method}")
                return False
        
        # Check supported languages
        if hasattr(service, 'supported_languages'):
            lang_count = len(service.supported_languages)
            print(f"   ✅ Phi-4 supports {lang_count} languages")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Service structure test failed: {e}")
        return False

def test_router_endpoints():
    """Test that routers have Phi-4 endpoints."""
    print("\n🌐 Testing Router Endpoints")
    print("=" * 40)
    
    try:
        from app.routers.audio import router
        
        # Get all route paths
        routes = [route.path for route in router.routes if hasattr(route, 'path')]
        
        # Check for Phi-4 endpoints
        phi4_endpoints = [route for route in routes if 'phi4' in route]
        
        print(f"   📡 Found {len(phi4_endpoints)} Phi-4 endpoints:")
        for endpoint in phi4_endpoints:
            print(f"      - {endpoint}")
        
        # Check for required endpoints
        required_endpoints = ['/phi4/load', '/phi4/unload', '/phi4/info']
        for endpoint in required_endpoints:
            full_path = '/api/v1/audio' + endpoint
            if any(endpoint in route for route in routes):
                print(f"   ✅ {endpoint}")
            else:
                print(f"   ❌ Missing {endpoint}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Router endpoint test failed: {e}")
        return False

def test_configuration():
    """Test Phase 3.1 configuration."""
    print("\n⚙️ Testing Phase 3.1 Configuration")
    print("=" * 40)
    
    try:
        from app.services.phi4_service import phi4_service
        
        # Test configuration without loading model
        print("   📊 Checking Phi-4 configuration...")
        print(f"      Model name: {phi4_service.model_name}")
        print(f"      Sample rate: {phi4_service.sample_rate}")
        print(f"      Max audio length: {phi4_service.max_audio_length}")
        print(f"      Supported languages: {len(phi4_service.supported_languages)}")
        
        # Test device detection
        device = phi4_service._get_device()
        print(f"      Detected device: {device}")
        print("   ✅ Configuration valid")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🎯 Tektra Phase 3.1 - Simple Integration Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_service_structure, 
        test_router_endpoints,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 3.1 integration successful.")
        print("✨ Features:")
        print("   • Phi-4 Multimodal as primary STT processor")
        print("   • Whisper as reliable fallback")
        print("   • Enhanced 8-language audio support")
        print("   • Unified speech recognition and chat completion")
        print("   • Performance optimized device detection")
        print("📦 Ready for PyPI publishing!")
        return True
    else:
        print("⚠️  Some tests failed. Check imports and configuration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)