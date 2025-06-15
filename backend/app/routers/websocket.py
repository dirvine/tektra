"""WebSocket endpoints for real-time communication."""

from typing import Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import json
import asyncio
from datetime import datetime

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id:
            self.user_connections[user_id] = websocket
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if user_id and user_id in self.user_connections:
            del self.user_connections[user_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific connection."""
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_text(message)
    
    async def send_to_user(self, message: str, user_id: str):
        """Send message to specific user."""
        if user_id in self.user_connections:
            websocket = self.user_connections[user_id]
            await self.send_personal_message(message, websocket)
    
    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            if connection.client_state == WebSocketState.CONNECTED:
                await connection.send_text(message)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for general communication."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Echo back with timestamp
            response = {
                "type": "echo",
                "original_message": message_data,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "received"
            }
            
            await manager.send_personal_message(
                json.dumps(response), 
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/chat/{user_id}")
async def chat_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for chat communication."""
    from app.services.ai_service import ai_manager, ChatMessage as AIChatMessage
    
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "chat":
                try:
                    # Extract message and model info
                    user_message = message_data.get("data", {}).get("message", "")
                    model_name = message_data.get("data", {}).get("model", "phi-3-mini")
                    
                    # Convert to AI service format
                    ai_messages = [AIChatMessage(role="user", content=user_message)]
                    
                    # Send start of response
                    start_response = {
                        "type": "ai_response_start",
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": model_name
                    }
                    await manager.send_personal_message(json.dumps(start_response), websocket)
                    
                    # Stream AI response
                    full_response = ""
                    async for token in ai_manager.chat(
                        messages=ai_messages,
                        model_name=model_name,
                        stream=True
                    ):
                        full_response += token
                        
                        # Send token
                        token_response = {
                            "type": "ai_response_token",
                            "token": token,
                            "user_id": user_id,
                            "model": model_name,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        await manager.send_personal_message(
                            json.dumps(token_response),
                            websocket
                        )
                    
                    # Send completion
                    completion_response = {
                        "type": "ai_response_complete",
                        "full_response": full_response,
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "tokens_used": len(full_response.split()),
                        "model": model_name
                    }
                    
                    await manager.send_personal_message(
                        json.dumps(completion_response),
                        websocket
                    )
                    
                except Exception as e:
                    # Send error response
                    error_response = {
                        "type": "ai_response_error",
                        "error": str(e),
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await manager.send_personal_message(
                        json.dumps(error_response),
                        websocket
                    )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


@router.websocket("/ws/audio/{user_id}")
async def audio_websocket(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time audio streaming."""
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # TODO: Process audio data for real-time transcription
            # For now, send back audio processing status
            response = {
                "type": "audio_processed",
                "user_id": user_id,
                "audio_length": len(data),
                "timestamp": datetime.utcnow().isoformat(),
                "status": "processing"
            }
            
            await manager.send_personal_message(
                json.dumps(response),
                websocket
            )
            
            # Simulate transcription delay
            await asyncio.sleep(0.1)
            
            # Send transcription result
            transcription = {
                "type": "transcription",
                "text": "Mock transcription of audio data",
                "confidence": 0.95,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.send_personal_message(
                json.dumps(transcription),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)


@router.websocket("/ws/camera")
async def camera_websocket(websocket: WebSocket):
    """WebSocket endpoint for camera video streaming."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            
            # TODO: Process video frame for computer vision
            # For now, send back frame processing status
            response = {
                "type": "frame_processed",
                "frame_size": len(data),
                "timestamp": datetime.utcnow().isoformat(),
                "objects_detected": 2,
                "faces_detected": 1
            }
            
            await manager.send_personal_message(
                json.dumps(response),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/robot/{robot_id}")
async def robot_websocket(websocket: WebSocket, robot_id: str):
    """WebSocket endpoint for real-time robot communication."""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            command_data = json.loads(data)
            
            # TODO: Send commands to actual robot and stream status
            # For now, simulate robot command execution
            response = {
                "type": "robot_status",
                "robot_id": robot_id,
                "command": command_data.get("command"),
                "status": "executing",
                "position": {"x": 0.1, "y": 0.2, "z": 0.5},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.send_personal_message(
                json.dumps(response),
                websocket
            )
            
            # Simulate command execution time
            await asyncio.sleep(1.0)
            
            # Send completion status
            completion = {
                "type": "robot_status",
                "robot_id": robot_id,
                "command": command_data.get("command"),
                "status": "completed",
                "position": {"x": 0.2, "y": 0.3, "z": 0.6},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await manager.send_personal_message(
                json.dumps(completion),
                websocket
            )
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# HTTP endpoints for WebSocket management
@router.get("/connections")
async def get_connections() -> Dict[str, Any]:
    """Get current WebSocket connection status."""
    return {
        "active_connections": len(manager.active_connections),
        "user_connections": len(manager.user_connections),
        "connected_users": list(manager.user_connections.keys())
    }


@router.post("/broadcast")
async def broadcast_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Broadcast message to all connected clients."""
    await manager.broadcast(json.dumps(message))
    return {
        "status": "success",
        "message": "Message broadcasted to all connections",
        "connections_reached": len(manager.active_connections)
    }


@router.post("/send/{user_id}")
async def send_to_user(user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
    """Send message to specific user."""
    if user_id in manager.user_connections:
        await manager.send_to_user(json.dumps(message), user_id)
        return {
            "status": "success",
            "message": f"Message sent to user {user_id}"
        }
    else:
        return {
            "status": "error",
            "message": f"User {user_id} not connected"
        }