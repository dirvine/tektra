"""
Model Management API Routes.

Endpoints for managing AI models, downloading, caching, and monitoring.
"""

import logging
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..services.model_manager import model_manager
from ..services.ai_service import ai_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class ModelDownloadRequest(BaseModel):
    """Request to download a model."""
    model_name: str


class ModelRemoveRequest(BaseModel):
    """Request to remove a model."""
    model_name: str


@router.get("/available")
async def list_available_models() -> Dict[str, Any]:
    """List all available models with their status."""
    try:
        models = model_manager.list_available_models()
        return {
            "status": "success",
            "models": models,
            "total_models": len(models)
        }
    except Exception as e:
        logger.error(f"Error listing available models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/loaded")
async def list_loaded_models() -> Dict[str, Any]:
    """List currently loaded models in memory."""
    try:
        loaded_models = ai_manager.get_loaded_models()
        model_info = []
        
        for model_name in loaded_models:
            info = ai_manager.get_model_info(model_name)
            if info:
                model_info.append({
                    "name": info.name,
                    "type": info.type.value,
                    "status": info.status.value,
                    "memory_usage": info.memory_usage,
                    "parameters": info.parameters
                })
        
        return {
            "status": "success", 
            "loaded_models": model_info,
            "total_loaded": len(loaded_models)
        }
    except Exception as e:
        logger.error(f"Error listing loaded models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def get_cache_info() -> Dict[str, Any]:
    """Get model cache information."""
    try:
        cache_info = model_manager.get_cache_info()
        return {
            "status": "success",
            "cache": cache_info
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download")
async def download_model(
    request: ModelDownloadRequest, 
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Download a model to local cache."""
    model_name = request.model_name
    
    try:
        # Check if model is already cached
        if model_manager.cache.is_model_cached(model_name):
            return {
                "status": "success",
                "message": f"Model {model_name} is already cached",
                "model_name": model_name
            }
        
        # Check if model is currently downloading
        if model_manager.downloader.is_downloading(model_name):
            progress = model_manager.downloader.get_download_progress(model_name)
            return {
                "status": "downloading",
                "message": f"Model {model_name} is already downloading",
                "model_name": model_name,
                "progress": progress
            }
        
        # Start download in background
        async def download_task():
            try:
                success = await model_manager.ensure_model_available(model_name)
                if success:
                    logger.info(f"Successfully downloaded model {model_name}")
                else:
                    logger.error(f"Failed to download model {model_name}")
            except Exception as e:
                logger.error(f"Error downloading model {model_name}: {e}")
        
        background_tasks.add_task(download_task)
        
        return {
            "status": "started",
            "message": f"Download started for model {model_name}",
            "model_name": model_name
        }
        
    except Exception as e:
        logger.error(f"Error starting download for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{model_name}/status")
async def get_download_status(model_name: str) -> Dict[str, Any]:
    """Get download status for a specific model."""
    try:
        status = model_manager.get_download_status(model_name)
        return {
            "status": "success",
            **status
        }
    except Exception as e:
        logger.error(f"Error getting download status for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_model(request: ModelDownloadRequest) -> Dict[str, Any]:
    """Load a model into memory."""
    model_name = request.model_name
    
    try:
        # Ensure model is available locally
        if not await model_manager.ensure_model_available(model_name):
            raise HTTPException(
                status_code=400, 
                detail=f"Model {model_name} could not be made available"
            )
        
        # Load model into AI manager
        success = await ai_manager.load_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} loaded successfully",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model {model_name}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload")
async def unload_model(request: ModelRemoveRequest) -> Dict[str, Any]:
    """Unload a model from memory."""
    model_name = request.model_name
    
    try:
        success = await ai_manager.unload_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} unloaded successfully",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found or not loaded"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/{model_name}")
async def remove_model_from_cache(model_name: str) -> Dict[str, Any]:
    """Remove a model from local cache."""
    try:
        # First unload from memory if loaded
        await ai_manager.unload_model(model_name)
        
        # Remove from cache
        success = await model_manager.remove_model(model_name)
        
        if success:
            return {
                "status": "success",
                "message": f"Model {model_name} removed from cache",
                "model_name": model_name
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found in cache"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing model {model_name} from cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}/info")
async def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        # Get model info from AI manager
        ai_info = ai_manager.get_model_info(model_name)
        
        # Get cache info
        cache_info = model_manager.cache.metadata.get(model_name)
        
        # Get availability info
        available_models = model_manager.list_available_models()
        availability_info = available_models.get(model_name)
        
        if not ai_info and not cache_info and not availability_info:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_name} not found"
            )
        
        model_info = {
            "name": model_name,
            "available": availability_info is not None,
            "cached": cache_info is not None,
            "loaded": ai_info is not None,
        }
        
        if availability_info:
            model_info.update({
                "type": availability_info["type"],
                "description": availability_info["description"],
                "size_gb": availability_info["size_gb"],
                "parameters": availability_info["parameters"],
                "downloading": availability_info["downloading"],
                "download_progress": availability_info["download_progress"]
            })
        
        if cache_info:
            model_info.update({
                "cache_info": {
                    "size_mb": cache_info.size_mb,
                    "download_date": cache_info.download_date,
                    "last_used": cache_info.last_used,
                    "checksum": cache_info.checksum,
                    "cache_status": cache_info.status
                }
            })
        
        if ai_info:
            model_info.update({
                "memory_info": {
                    "status": ai_info.status.value,
                    "memory_usage": ai_info.memory_usage,
                    "parameters": ai_info.parameters,
                    "error_message": ai_info.error_message
                }
            })
        
        return {
            "status": "success",
            "model": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting info for model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))