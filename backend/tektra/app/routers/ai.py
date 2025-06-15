"""AI model management and chat endpoints."""

import json
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..database import get_db
from ..services.ai_service import ai_manager, ChatMessage as AIChatMessage, ModelInfo

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message schema."""
    message: str
    stream: bool = False
    model: str = "default"


class ChatResponse(BaseModel):
    """Chat response schema."""
    response: str
    model: str
    tokens_used: int = 0
    processing_time: float = 0.0


class ModelInfoResponse(BaseModel):
    """Model information response schema."""
    name: str
    description: str
    size: str
    status: str
    loaded: bool = False
    memory_usage: str = "0 GB"
    error_message: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    message: ChatMessage,
    db: AsyncSession = Depends(get_db)
) -> ChatResponse:
    """Send a message to the AI and get a response."""
    try:
        # Convert to AI service format
        ai_messages = [AIChatMessage(role="user", content=message.message)]
        
        # Generate response using AI service
        if message.stream:
            # For non-streaming endpoint, we still get the full response
            response = await ai_manager.chat(
                messages=ai_messages,
                model_name=message.model,
                stream=False
            )
            
            return ChatResponse(
                response=response.content,
                model=response.model,
                tokens_used=response.tokens_used,
                processing_time=response.processing_time
            )
        else:
            response = await ai_manager.chat(
                messages=ai_messages,
                model_name=message.model,
                stream=False
            )
            
            return ChatResponse(
                response=response.content,
                model=response.model,
                tokens_used=response.tokens_used,
                processing_time=response.processing_time
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {str(e)}")


@router.post("/chat/stream")
async def chat_stream(
    message: ChatMessage,
    db: AsyncSession = Depends(get_db)
):
    """Send a message to the AI and get a streaming response."""
    try:
        # Convert to AI service format
        ai_messages = [AIChatMessage(role="user", content=message.message)]
        
        async def generate_stream():
            try:
                # Generate streaming response using AI service
                async for token in ai_manager.chat(
                    messages=ai_messages,
                    model_name=message.model,
                    stream=True
                ):
                    # Format as Server-Sent Events
                    data = json.dumps({"token": token, "model": message.model})
                    yield f"data: {data}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                error_data = json.dumps({"error": str(e)})
                yield f"data: {error_data}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain"
            }
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI streaming failed: {str(e)}")


@router.get("/models", response_model=List[ModelInfoResponse])
async def list_models() -> List[ModelInfoResponse]:
    """List available AI models."""
    try:
        models = ai_manager.list_models()
        
        return [
            ModelInfoResponse(
                name=model.name,
                description=model.description,
                size=model.size,
                status=model.status.value,
                loaded=model.status.value == "loaded",
                memory_usage=model.memory_usage or "0 GB",
                error_message=model.error_message
            )
            for model in models
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.post("/models/{model_name}/load")
async def load_model(model_name: str) -> Dict[str, Any]:
    """Load a specific AI model."""
    try:
        success = await ai_manager.load_model(model_name)
        
        if success:
            model_info = ai_manager.get_model_info(model_name)
            return {
                "message": f"Model {model_name} loaded successfully",
                "model": model_name,
                "status": "loaded",
                "memory_usage": model_info.memory_usage or "Unknown"
            }
        else:
            model_info = ai_manager.get_model_info(model_name)
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to load model {model_name}: {model_info.error_message if model_info else 'Unknown error'}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


@router.delete("/models/{model_name}")
async def unload_model(model_name: str) -> Dict[str, Any]:
    """Unload a specific AI model."""
    try:
        success = await ai_manager.unload_model(model_name)
        
        if success:
            return {
                "message": f"Model {model_name} unloaded successfully",
                "model": model_name,
                "status": "unloaded"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model unloading failed: {str(e)}")


@router.get("/models/{model_name}/status")
async def get_model_status(model_name: str) -> Dict[str, Any]:
    """Get status of a specific model."""
    try:
        model_info = ai_manager.get_model_info(model_name)
        
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        return {
            "model": model_name,
            "status": model_info.status.value,
            "memory_usage": model_info.memory_usage or "0 GB",
            "type": model_info.type.value,
            "loaded": model_info.status.value == "loaded",
            "error_message": model_info.error_message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")