from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Dict, Any
from utils.config_loader import ConfigManager
from utils.model_loader import ModelManager
import time

router = APIRouter(
    prefix="/api/models",
    tags=["models"],
)
config_manager = ConfigManager()


def get_config_manager():
    return config_manager


model_manager = ModelManager(config_manager)


def get_model_manager():
    return model_manager


@router.get("/", response_model=List[Dict[str, Any]])
async def list_models(model_manager: ModelManager = Depends(get_model_manager)):
    """Get all status of models"""
    config_manager.get_all_models()
    return model_manager.get_all_models_status()


@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_model_info(model_id: str, model_manager: ModelManager = Depends(get_model_manager)):
    """Get details info of model"""
    model_status = model_manager.get_model_status(model_id)
    if model_status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return model_status


@router.post("/{model_id}/load", response_model=Dict[str, Any])
async def load_model(model_id: str, model_manager: ModelManager = Depends(get_model_manager)):
    """Load and Save Model to memory"""
    success = await model_manager.load_model(model_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Cannot load model {model_id}")
    return {"id": model_id, "status": "loaded"}


@router.post("/{model_id}/unload", response_model=Dict[str, Any])
async def unload_model(model_id: str, model_manager: ModelManager = Depends(get_model_manager)):
    """Release a model from memory"""
    try:
        # Check status first
        model_status = model_manager.get_model_status(model_id)
        if model_status["status"] == "not_found":
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
            
        if model_status["status"] != "loaded":
            return {"id": model_id, "status": "unloaded", "message": "Model was not loaded"}
        
        # Try to unload
        success = model_manager.unload_model(model_id)
        if not success:
            return {
                "id": model_id, 
                "status": "error", 
                "message": f"Failed to unload model {model_id}"
            }
            
        return {"id": model_id, "status": "unloaded", "message": "Successfully unloaded model"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error when unloading model {model_id}: {str(e)}")
        return {
            "id": model_id, 
            "status": "error", 
            "message": f"Error during unload: {str(e)}"
        }


@router.get("/{model_id}/loading-status", response_model=Dict[str, Any])
async def get_model_loading_status(model_id: str, model_manager: ModelManager = Depends(get_model_manager)):
    """Get information about model loading progress"""
    loading_status = model_manager.get_loading_status(model_id)
    if loading_status:
        return loading_status
    
    # If no loading information, return current status
    model_status = model_manager.get_model_status(model_id)
    if model_status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")
    
    return {
        "status": model_status["status"],
        "progress": 100 if model_status["status"] == "loaded" else 0,
        "message": f"Model {model_id} has status {model_status['status']}"
    }


@router.post("/{model_id}/load-async", response_model=Dict[str, Any])
async def load_model_async(model_id: str, 
                           background_tasks: BackgroundTasks, 
                           model_manager: ModelManager = Depends(get_model_manager)):
    """Load model in background and return immediately"""
    model_config = model_manager.config_manager.get_model_config(model_id)
    if not model_config:
        raise HTTPException(status_code=404, detail=f"Model {model_id} does not exist")

    # Check if model is already loaded
    if model_id in model_manager.loaded_models:
        return {"id": model_id, "status": "loaded", "message": "Model was previously loaded"}

    # Check if model is already being loaded
    loading_status = model_manager.get_loading_status(model_id)
    if loading_status and loading_status["status"] == "loading":
        return {
            "id": model_id,
            "status": "loading",
            "progress": loading_status["progress"],
            "message": "Model is being loaded"
        }

    # Initialize loading status
    async with model_manager.loading_lock:
        model_manager.loading_status[model_id] = {
            "status": "loading",
            "progress": 0,
            "start_time": time.time(),
            "message": f"Started loading model {model_id}",
            "error": None
        }
    
    # Create task to load model in background with proper async handling
    async def load_model_task():
        try:
            await model_manager.load_model(model_id)
        except Exception as e:
            print(f"Error in background task when loading model {model_id}: {str(e)}")
            async with model_manager.loading_lock:
                if model_id in model_manager.loading_status:
                    model_manager.loading_status[model_id].update({
                        "status": "error",
                        "progress": 0,
                        "message": f"Error loading model: {str(e)}",
                        "error": str(e)
                    })
    
    background_tasks.add_task(load_model_task)
    
    return {
        "id": model_id, 
        "status": "loading", 
        "progress": 0,
        "message": f"Started loading model {model_id} in background"
    }