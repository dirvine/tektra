#!/usr/bin/env python3
"""Test WebSocket chat functionality with conversation persistence."""

import asyncio
import json
import websockets
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket_chat():
    """Test WebSocket chat with conversation persistence."""
    try:
        uri = "ws://localhost:8000/ws/chat/1"  # Chat WebSocket with user_id
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket")
            
            # Send a chat message
            test_message = {
                "type": "chat",
                "data": {
                    "message": "Hello AI! Can you help me test the conversation system?",
                    "model": "phi-3-mini",
                    "conversation_id": 1  # Use the conversation we created earlier
                }
            }
            
            await websocket.send(json.dumps(test_message))
            logger.info(f"📤 Sent: {test_message}")
            
            # Receive responses with timeout
            response_count = 0
            full_response = ""
            timeout_seconds = 10
            
            try:
                async with asyncio.timeout(timeout_seconds):
                    async for message in websocket:
                        try:
                            data = json.loads(message)
                            response_count += 1
                            
                            if data.get("type") == "ai_response_token":
                                token = data.get("token", "")
                                full_response += token
                                print(token, end="", flush=True)
                            elif data.get("type") == "ai_response_complete":
                                print("\n")
                                logger.info(f"✅ Response complete. Total tokens: {response_count}")
                                logger.info(f"📝 Full response: {full_response}")
                                break
                            elif data.get("type") == "ai_response_error":
                                logger.error(f"❌ Error: {data.get('error')}")
                                break
                            elif data.get("type") == "conversation_created":
                                logger.info(f"📁 New conversation created: {data.get('conversation_id')}")
                            elif data.get("type") == "ai_response_start":
                                logger.info(f"🤖 AI started responding with {data.get('model')}")
                                
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ JSON decode error: {e}")
                            break
            except asyncio.TimeoutError:
                logger.warning(f"⏱️ Test timed out after {timeout_seconds} seconds")
                    
            logger.info("✅ WebSocket test completed")
            
    except Exception as e:
        logger.error(f"❌ WebSocket test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_websocket_chat())