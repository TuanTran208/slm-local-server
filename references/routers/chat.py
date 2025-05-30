from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import asyncio
import json
import time
import uuid
from utils.model_loader import ModelManager
from routers.models import get_model_manager
from routers.conversations import conversation_manager
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, field_validator

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
)


class Message(BaseModel):
    role: str = Field(..., description="Role of the message (system, user, assistant)")
    content: str = Field(..., description="Content of the message")
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        valid_roles = ['system', 'user', 'assistant']
        if v not in valid_roles:
            raise ValueError(f"Role must be one of: {', '.join(valid_roles)}")
        return v
    
    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Content must be a non-empty string")
        return v


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="ID of the model to use")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: Optional[float] = Field(0.7, description="Temperature for text generation", ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.95, description="Top-p (nucleus sampling) parameter", ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(2048, description="Maximum number of tokens to generate", ge=1, le=8192)
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    stop: Optional[List[str]] = Field(None, description="List of stop sequences")
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("At least one message is required")
        
        roles = [msg.role for msg in v]
        if "user" not in roles:
            raise ValueError("At least one user message is required")
            
        return v
    
    @field_validator('model')
    @classmethod
    def validate_model_id(cls, v):
        if not v or not isinstance(v, str) or len(v) < 1:
            raise ValueError("Model ID must be a non-empty string")
        return v


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


@router.post("/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Create chat completion from a list of messages"""
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request, model_manager),
            media_type="text/event-stream"
        )
    
    # Check if model is available
    model_status = model_manager.get_model_status(request.model)
    if model_status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Model {request.model} does not exist")
    
    # check model is loaded
    if model_status["status"] != "loaded":
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not loaded. Please load it first using the /api/models/{request.model}/load endpoint.")
    
    # Create prompt from messages
    prompt = format_chat_prompt(request.messages)
    
    # Set parameters for generation
    params = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop
    }
    
    # Call model to generate text - using async function
    result = await model_manager.generate_text_async(request.model, prompt, params)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Create response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=result["text"]),
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    )

    return response


@router.post("/completions/with_context", response_model=ChatCompletionResponse)
async def create_chat_completion_with_context(
    model: str = Query(..., min_length=1, description="Model ID to use"),
    message: str = Query(..., min_length=1, description="User message content"),
    conv_id: Optional[str] = Query(None, min_length=1, description="Conversation ID"),
    temperature: Optional[float] = Query(0.7, ge=0.0, le=2.0, description="Temperature for text generation"),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Send a message to the model with conversation context"""
    # Validate model ID format
    if not model or not model.strip():
        raise HTTPException(status_code=400, detail="Invalid model ID")
    
    # Validate message content
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Rest of the function remains the same
    # check model loaded
    model_status = model_manager.get_model_status(model)
    if model_status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Model {model} does not exist")
    
    if model_status["status"] != "loaded":
        raise HTTPException(status_code=400, detail=f"Model {model} is not loaded. Please load it first using the /api/models/{model}/load endpoint.")
    
    # Get or create conversation
    if conv_id:
        conversation = conversation_manager.get_conversation(conv_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation = conversation_manager.create_conversation(model)

    # Add user message
    conversation_manager.add_message(conversation.id, "user", message)

    # Prepare list of messages to send to the model
    messages = [Message(role=msg.role, content=msg.content) for msg in conversation.messages]

    # Set parameters
    request = ChatCompletionRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=False
    )

    # Call model to generate text
    response = await create_chat_completion(request, model_manager)

    # Add response to conversation
    assistant_response = response.choices[0].message.content
    conversation_manager.add_message(conversation.id, "assistant", assistant_response)

    # Return both response and conversation ID
    return {
        **response.model_dump(),
        "conversation_id": conversation.id
    }


async def stream_chat_completion(request: ChatCompletionRequest, model_manager: ModelManager):
    """Create chat completion and stream results with improved streaming"""
    # Check if model is available
    model_status = model_manager.get_model_status(request.model)
    if model_status["status"] == "not_found":
        yield f"data: {json.dumps({'error': f'Model {request.model} does not exist'})}\n\n"
        return
    
    # check model loaded
    if model_status["status"] != "loaded":
        yield f"data: {json.dumps({'error': f'Model {request.model} is not loaded. Please load it first using the /api/models/{request.model}/load endpoint.'})}\n\n"
        return

    # Create prompt from messages
    prompt = format_chat_prompt(request.messages)

    # Set parameters for generation
    params = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop
    }

    # ID for response
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # Call model to generate text with stream=True
    try:
        # Acquire model lock first
        await model_manager.acquire_model_lock(request.model)
        
        # Get stream generator - use non-blocking and flush after each token
        model = model_manager.loaded_models.get(request.model)
        if not model:
            yield f"data: {json.dumps({'error': f'Model {request.model} not found in loaded models'})}\n\n"
            return
            
        # Send first role token
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created_time,
            model=request.model,
            choices=[{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": None
            }]
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
        await asyncio.sleep(0.01)  # Ensure flush
        
        # Get parameters
        max_tokens = params.get('max_tokens', 2048)
        temperature = params.get('temperature', 0.7)
        top_p = params.get('top_p', 0.95)
        repeat_penalty = params.get('repeat_penalty', 1.1)
        stop_sequences = params.get('stop', None) or []
        
        print(f"Starting streaming generation for {request.model}")
        
        # Create stream generator directly from llama-cpp
        response_stream = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop_sequences,
            stream=True
        )
        
        # Iterate through each token and send immediately
        for chunk_data in response_stream:
            if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                token_text = chunk_data["choices"][0]["text"]
                is_stop = chunk_data["choices"][0].get("finish_reason") == "stop"
                
                # Create response chunk
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created_time,
                    model=request.model,
                    choices=[{
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": "stop" if is_stop else None
                    }]
                )
                
                # Send chunk and ensure flush
                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                await asyncio.sleep(0.01)  # Small but enough to ensure flush
                
                if is_stop:
                    break

        # Send final DONE marker
        yield "data: [DONE]\n\n"

    except Exception as e:
        print(f"Stream error: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Always release the lock
        await model_manager.release_model_lock(request.model)
        print("Stream completed and lock released")


def format_chat_prompt(messages: List[Message]) -> str:
    """Format list of messages into input prompt"""
    prompt = ""
    for msg in messages:
        if msg.role == "system":
            prompt += f"System: {msg.content}\n\n"
        elif msg.role == "user":
            prompt += f"Human: {msg.content}\n\n"
        elif msg.role == "assistant":
            prompt += f"Assistant: {msg.content}\n\n"
    prompt += "Assistant: "
    return prompt


@router.post("/completions/cancel", response_model=Dict[str, Any])
async def cancel_completion(request: Dict[str, Any]):
    """Cancel a running completion process (not implemented)"""
    return {"status": "not_implemented", "message": "Cancel function not implemented"}
