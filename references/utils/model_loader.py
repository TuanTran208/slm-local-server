import os
import time
import psutil
from typing import Dict, Any, List, Optional
from utils.config_loader import ConfigManager
import asyncio
from asyncio import Lock
from collections import deque
import uuid

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.model_sizes: Dict[str, int] = {}
        
        # Tracking loading status
        self.loading_status: Dict[str, Dict[str, Any]] = {}
        self.loading_lock = Lock()
        
        # Queue system for concurrent requests
        self.model_locks: Dict[str, Lock] = {}
        self.request_queues: Dict[str, deque] = {}
        self.request_times: Dict[str, Dict[str, float]] = {}
        
        # Memory management
        self.system_limit = 0.9

    def get_free_memory(self) -> int:
        """Get free system memory in bytes"""
        return psutil.virtual_memory().available
    
    async def load_model(self, model_id: str, timeout: int = 300) -> bool:
        """Load GGUF model from local with improved error handling and timeout"""
        loading_start_time = time.time()
        print(f"===== STARTING MODEL LOADING {model_id} =====")
        
        try:
            # Check if model is already loaded
            if model_id in self.loaded_models:
                print(f"Model {model_id} was previously loaded.")
                return True
                
            # Get model configuration info
            model_config = self.config_manager.get_model_config(model_id)
            if not model_config:
                print(f"Configuration not found for model: {model_id}")
                return False
            
            # Reset loading status before starting
            async with self.loading_lock:
                self.loading_status[model_id] = {
                    "status": "loading",
                    "progress": 0,
                    "start_time": time.time(),
                    "message": f"Started loading model {model_id}",
                    "error": None
                }
            
            # Path to model file
            model_path = model_config.model_path
            
            # Check if file exists
            print(f"Checking model file at: {model_path}")
            if not os.path.exists(model_path):
                error_msg = f"GGUF file not found at path: {model_path}"
                print(f"ERROR: {error_msg}")
                await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                return False
            
            # Check and display memory information
            model_size = os.path.getsize(model_path)
            available_memory = self.get_free_memory()
            estimated_ram_needed = model_size * 2  # Estimate double the file size
            
            print(f"Memory information:")
            print(f"- Model file size: {model_size / (1024**3):.2f} GB")
            print(f"- Available memory: {available_memory / (1024**3):.2f} GB")
            print(f"- Estimated RAM needed: {estimated_ram_needed / (1024**3):.2f} GB")
            print(f"- Memory usage limit: {self.system_limit * 100}%")
            
            # Check if enough memory is available
            if estimated_ram_needed > available_memory * self.system_limit:
                print("Warning: Not enough memory. Trying to free memory from other models...")
                if not self._free_memory_for_model(estimated_ram_needed):
                    error_msg = f"Not enough memory: needed {estimated_ram_needed/(1024**3):.2f} GB, available {available_memory/(1024**3):.2f} GB"
                    print(f"ERROR: {error_msg}")
                    await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                    return False
                else:
                    # Update available memory after freeing
                    available_memory = self.get_free_memory()
                    print(f"New available memory after freeing: {available_memory / (1024**3):.2f} GB")
            
            # Start loading process
            print(f"Starting to load model {model_id}...")
            await self._update_loading_status(model_id, 5, "Preparing to load model...")
            
            # Check file format
            if not model_path.lower().endswith('.gguf'):
                error_msg = f"File is not in GGUF format: {model_path}"
                print(f"ERROR: {error_msg}")
                await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                return False
            
            # Prepare parameters
            await self._update_loading_status(model_id, 10, "Preparing parameters for GGUF...")
            
            # Get parameters from configuration
            n_gpu_layers = model_config.get_provider_param('n_gpu_layers', 0)
            n_ctx = model_config.get_provider_param('n_ctx', model_config.context_length)
            use_mlock = model_config.get_provider_param('use_mlock', False)
            use_mmap = model_config.get_provider_param('use_mmap', True)
            
            # Check for GPU
            cuda_available = False
            if TORCH_AVAILABLE:
                try:
                    cuda_available = torch.cuda.is_available()
                    if cuda_available:
                        cuda_info = torch.cuda.get_device_properties(0)
                        print(f"GPU detected: {cuda_info.name} with {cuda_info.total_memory / (1024**3):.2f} GB")
                    else:
                        print("No CUDA-compatible GPU detected")
                except Exception as e:
                    print(f"Error when checking GPU: {str(e)}")
                    cuda_available = False
            
            # Adjust n_gpu_layers based on GPU availability
            original_n_gpu_layers = n_gpu_layers
            n_gpu_layers = -1 if cuda_available and n_gpu_layers != 0 else 0
            print(f"n_gpu_layers parameter: {original_n_gpu_layers} -> {n_gpu_layers} (after adjustment)")
            
            # Display parameters to be used
            print(f"Model loading parameters:")
            print(f"- model_path: {model_path}")
            print(f"- n_gpu_layers: {n_gpu_layers}")
            print(f"- n_ctx: {n_ctx}")
            print(f"- use_mlock: {use_mlock}")
            print(f"- use_mmap: {use_mmap}")
            
            # Update status
            await self._update_loading_status(model_id, 20, f"Loading GGUF model {model_id}...")
            
            try:
                # Import library
                try:
                    from llama_cpp import Llama
                except ImportError:
                    error_msg = "Missing llama-cpp-python library. Please install using pip install llama-cpp-python"
                    print(f"ERROR: {error_msg}")
                    await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                    return False
                
                print("Initializing Llama model...")
                await self._update_loading_status(model_id, 30, "Loading GGUF model into memory...")
                
                # Create task with timeout
                try:
                    # Create model loading task with timeout
                    load_task = asyncio.create_task(
                        self._load_model_with_timeout(
                            Llama, model_path, n_gpu_layers, n_ctx, use_mlock, use_mmap, timeout
                        )
                    )
                    
                    # Wait for task to complete with progress updates
                    start_wait = time.time()
                    while not load_task.done():
                        elapsed = time.time() - start_wait
                        if elapsed > timeout:
                            load_task.cancel()
                            error_msg = f"Model loading timed out ({timeout}s)"
                            print(f"ERROR: {error_msg}")
                            await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                            return False
                        
                        # Update progress based on elapsed time
                        progress = min(80, 30 + int(50 * elapsed / (timeout * 0.8)))
                        await self._update_loading_status(model_id, progress, f"Loading model... ({int(elapsed)}s elapsed)")
                        await asyncio.sleep(2)  # Update every 2 seconds
                    
                    # Get results from task
                    model = await load_task
                    
                    if model is None:
                        error_msg = "Model load failed with unknown error"
                        print(f"ERROR: {error_msg}")
                        await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                        return False
                    
                except asyncio.CancelledError:
                    error_msg = f"Model loading task was cancelled after {timeout}s"
                    print(f"ERROR: {error_msg}")
                    await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=error_msg)
                    return False
                except Exception as e:
                    error_msg = f"Error when loading model: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=str(e))
                    return False
                
                # Update status
                await self._update_loading_status(model_id, 90, "Successfully loaded GGUF model, initializing...")
                
                # Save model in loaded list
                self.loaded_models[model_id] = model
                self.loaded_tokenizers[model_id] = None  # GGUF handles tokenizer internally
                self.model_sizes[model_id] = estimated_ram_needed  # Save model size
                
                # Complete
                load_time = time.time() - loading_start_time
                success_msg = f"Successfully loaded model {model_id} in {load_time:.2f}s"
                print(f"SUCCESS: {success_msg}")
                await self._update_loading_status(model_id, 100, success_msg)
                
                # Try to get model attributes to verify correct loading
                try:
                    ctx_size = model.n_ctx()
                    print(f"Model context size: {ctx_size}")
                except Exception as e:
                    print(f"Warning: Unable to get context size information: {str(e)}")
                
                return True
                
            except Exception as e:
                error_msg = f"Unexpected error when loading model: {str(e)}"
                print(f"ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=str(e))
                return False
        
        except Exception as e:
            error_msg = f"General error when loading model {model_id}: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            await self._update_loading_status(model_id, 0, f"Error: {error_msg}", error=str(e))
            return False
    
    async def _load_model_with_timeout(self, Llama, model_path, n_gpu_layers, n_ctx, use_mlock, use_mmap, timeout):
        """Helper function to load model with timeout"""
        try:
            # Create synchronous function to run model loading (since llama-cpp-python is synchronous)
            def load_llama_model():
                return Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    use_mlock=use_mlock,
                    use_mmap=use_mmap
                )
            
            # Run in thread pool executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            model = await asyncio.wait_for(
                loop.run_in_executor(None, load_llama_model),
                timeout=timeout
            )
            return model
        except asyncio.TimeoutError:
            print(f"Timeout: Model loading exceeded {timeout} seconds")
            return None
        except Exception as e:
            print(f"Error in _load_model_with_timeout: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _free_memory_for_model(self, needed_memory: int) -> bool:
        """Try to free memory by unloading models until we have enough"""
        if needed_memory > self.get_free_memory():
            # Sort models by last used time (implement tracking if needed)
            for model_id in list(self.loaded_models.keys()):
                print(f"Trying to unload {model_id} to free memory...")
                self.unload_model(model_id)
                if needed_memory <= self.get_free_memory():
                    return True
            return False
        return True

    async def _update_loading_status(self, model_id: str, progress: int, message: str, error: str = None):
        """Update model loading status"""
        try:
            async with self.loading_lock:
                if model_id in self.loading_status:
                    self.loading_status[model_id]["progress"] = progress
                    self.loading_status[model_id]["message"] = message
                    if error:
                        self.loading_status[model_id]["error"] = error
                        self.loading_status[model_id]["status"] = "error"
                    elif progress >= 100:
                        self.loading_status[model_id]["status"] = "loaded"
                    else:
                        self.loading_status[model_id]["status"] = "loading"
                    
                    # Log update for debugging
                    print(f"Loading status update: {model_id} - {progress}% - {message}")
        except Exception as e:
            print(f"Error updating loading status: {str(e)}")
    
    def get_loading_status(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model loading status"""
        return self.loading_status.get(model_id)

    def unload_model(self, model_id: str) -> bool:
        """Unload model from memory with improved error handling"""
        try:
            print(f"Unloading model {model_id}...")
            
            if model_id not in self.loaded_models:
                print(f"Model {model_id} not found in loaded_models list")
                return True
                
            model = self.loaded_models[model_id]
            
            # GGUF model handling - explicitly delete fields
            if isinstance(model, object) and hasattr(model, '__class__') and 'llama_cpp' in str(model.__class__):
                print("This is a GGUF model, performing special unloading...")
                
                # Set all important attributes to None to help garbage collection
                for attr in ['ctx', '_ctx', 'session', 'model', 'params']:
                    if hasattr(model, attr):
                        try:
                            setattr(model, attr, None)
                        except:
                            pass
                
                # Call reset method if exists
                if hasattr(model, 'reset'):
                    try:
                        model.reset()
                    except:
                        pass
            
            # Remove from tracking dictionaries
            del self.loaded_models[model_id]
            
            if model_id in self.loaded_tokenizers:
                del self.loaded_tokenizers[model_id]
            
            if model_id in self.model_sizes:
                del self.model_sizes[model_id]
            
            # Update loading status
            with self.loading_lock:
                if model_id in self.loading_status:
                    del self.loading_status[model_id]
            
            # Clean up memory
            model = None  # Remove reference
            
            # Free GPU memory
            if TORCH_AVAILABLE:
                try:
                    torch.cuda.empty_cache()
                    print("GPU memory cache cleared")
                except Exception as e:
                    print(f"Note: Could not clear GPU memory: {str(e)}")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Optional memory sync on Linux/Unix
            if hasattr(os, "sync"):
                try:
                    os.sync()
                except:
                    pass
            
            print(f"Successfully unloaded model {model_id}")
            return True
            
        except Exception as e:
            print(f"Error when unloading model {model_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Update status to reflect error
            with self.loading_lock:
                if model_id in self.loading_status:
                    self.loading_status[model_id]["error"] = str(e)
            return False

    def get_model_status(self, model_id: str) -> Dict[str, Any]:
        """Return information about model status"""
        model_config = self.config_manager.get_model_config(model_id)
        if not model_config:
            return {"id": model_id, "status": "not_found"}
        
        # Check loading status
        loading_info = self.get_loading_status(model_id)
        if loading_info and loading_info["status"] == "loading":
            return {
                "id": model_id,
                "name": model_config.name,
                "status": "loading",
                "progress": loading_info["progress"],
                "message": loading_info["message"],
                "start_time": loading_info["start_time"],
                "elapsed_time": time.time() - loading_info["start_time"],
                "local": model_config.local,
                "model_family": model_config.model_family,
                "context_length": model_config.context_length
            }
        
        is_loaded = model_id in self.loaded_models
        
        # Check if GGUF file exists
        model_exists = os.path.isfile(model_config.model_path)
        
        # Determine status
        if is_loaded:
            status = "loaded"
        elif model_exists:
            status = "available"
        else:
            status = "not_downloaded"
        
        return {
            "id": model_id,
            "name": model_config.name,
            "status": status,
            "local": model_config.local,
            "model_family": model_config.model_family,
            "context_length": model_config.context_length
        }
    
    def get_all_models_status(self) -> List[Dict[str, Any]]:
        """Return information about all models"""
        return [self.get_model_status(model.id) for model in self.config_manager.get_all_models()]
    
    def generate_text(self, model_id: str, prompt: str, parameters: Dict[str, Any] = None, stream: bool = False):
        """Generate text from prompt using GGUF model"""
        try:
            if model_id not in self.loaded_models:
                return {"error": f"Model {model_id} is not loaded. Please load the model first using the /api/models/{model_id}/load endpoint."}
            
            model = self.loaded_models[model_id]
            model_config = self.config_manager.get_model_config(model_id)
            
            # Combine default parameters with provided parameters
            default_params = model_config.default_parameters.dict()
            gen_params = default_params.copy()
            if parameters:
                gen_params.update(parameters)
            
            # Process special parameters
            max_tokens = gen_params.pop('max_tokens', 2048)
            stop_sequences = gen_params.pop('stop', None)
            
            # Convert parameters from configuration to llama-cpp parameters
            temperature = gen_params.pop('temperature', 0.7)
            top_p = gen_params.pop('top_p', 0.95)
            repeat_penalty = gen_params.pop('repeat_penalty', 1.1)
            
            # Process stop sequences
            stop_sequences_list = stop_sequences if stop_sequences else []
            
            if stream:
                # Stream mode for GGUF
                def generate_gguf_stream():
                    response = model.create_completion(
                        prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        repeat_penalty=repeat_penalty,
                        stop=stop_sequences_list,
                        stream=True
                    )
                    
                    for chunk in response:
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            yield {
                                "text": chunk["choices"][0]["text"],
                                "stop": chunk["choices"][0].get("finish_reason") == "stop"
                            }
                    
                    yield {"text": "", "stop": True}
                
                return generate_gguf_stream()
            else:
                # Non-stream mode for GGUF
                response = model.create_completion(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=repeat_penalty,
                    stop=stop_sequences_list
                )
                
                if "choices" in response and len(response["choices"]) > 0:
                    response_text = response["choices"][0]["text"]
                    return {"text": response_text.strip()}
                else:
                    return {"error": "No response received from model"}
        
        except Exception as e:
            print(f"Error generating text with model {model_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_embedding(self, model_id: str, text: str) -> Dict[str, Any]:
        """Generate embedding for text using selected model (if supported)"""
        # This feature may be implemented in the future
        return {"error": "Embedding feature not yet supported"}

    async def generate_text_async(self, model_id: str, prompt: str, parameters: Dict[str, Any] = None, stream: bool = False):
        """Async version of generate_text with queue management"""
        request_id = str(uuid.uuid4())
        
        try:
            # Acquire lock for this model
            await self.acquire_model_lock(model_id, request_id)
            
            # Call standard generate_text
            result = self.generate_text(model_id, prompt, parameters, stream)
            return result
        
        finally:
            # Always release lock
            await self.release_model_lock(model_id)

    async def acquire_model_lock(self, model_id: str, request_id: str = None, timeout: int = 60) -> bool:
        """Acquire lock for model with queue management and timeout"""
        if model_id not in self.model_locks:
            self.model_locks[model_id] = Lock()
            self.request_queues[model_id] = deque()
            self.request_times[model_id] = {}
        
        request_id = request_id or str(uuid.uuid4())
        print(f"Request {request_id}: Waiting for lock on model {model_id}")
        
        # Add request to queue
        self.request_queues[model_id].append(request_id)
        self.request_times[model_id][request_id] = time.time()
        
        # Wait for our turn with timeout
        start_wait = time.time()
        while self.request_queues[model_id][0] != request_id:
            # Check timeout
            if time.time() - start_wait > timeout:
                # Remove request from queue if timeout
                if request_id in self.request_queues[model_id]:
                    self.request_queues[model_id].remove(request_id)
                if request_id in self.request_times[model_id]:
                    del self.request_times[model_id][request_id]
                print(f"Request {request_id}: Timeout waiting for model lock")
                raise TimeoutError(f"Timeout waiting for model lock after {timeout} seconds")
                
            await asyncio.sleep(0.1)
        
        # Acquire lock with timeout
        try:
            # Set timeout for lock acquisition
            print(f"Request {request_id}: Acquiring lock for model {model_id}")
            await asyncio.wait_for(self.model_locks[model_id].acquire(), timeout)
            print(f"Request {request_id}: Lock acquired for model {model_id}")
            
            # Remove from queue
            self.request_queues[model_id].popleft()
            self.request_times[model_id].pop(request_id, None)
            
            return True
        except asyncio.TimeoutError:
            # Handle timeout
            if request_id in self.request_queues[model_id]:
                self.request_queues[model_id].remove(request_id)
            if request_id in self.request_times[model_id]:
                del self.request_times[model_id][request_id]
            print(f"Request {request_id}: Timeout acquiring model lock")
            raise TimeoutError(f"Timeout acquiring model lock after {timeout} seconds")
            
    async def release_model_lock(self, model_id: str) -> None:
        """Release model lock with improved error handling"""
        if model_id in self.model_locks:
            try:
                print(f"Releasing lock for model {model_id}")
                if self.model_locks[model_id].locked():
                    self.model_locks[model_id].release()
                    print(f"Lock released for model {model_id}")
                else:
                    print(f"Warning: Attempted to release an unlocked lock for model {model_id}")
            except Exception as e:
                print(f"Error releasing model lock: {str(e)}")