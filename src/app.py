from fastapi import FastAPI, HTTPException
import torch
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import schemas
import time
from config import settings
from enhanced_npc import config


# Path to the locally stored model

# Load model at startup
def load_model():
    try:
        print(f"Loading model from: {settings.MODEL_DIR}")

        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(settings.MODEL_DIR, torch_dtype = torch.float16, device_map = "auto")
        # Ensure model is loaded locally
        # device = torch.device("cpu")  # Use GPU if available
        # model.to(device)
        # model.eval()
        print(model.device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise
    return model, tokenizer


slm_model = {}
prompt_manager = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, tokenizer = load_model()
    slm_model["model"] = model
    slm_model["tokenizer"] = tokenizer
    character_cfg = config.CharacterConfig.load()
    prompt_manager["templates"] = character_cfg.templates
    # print(slm_model["slm_model"])
    yield
    # Clean up the ML models and release the resources
    slm_model.clear()
    prompt_manager.clear()


app = FastAPI(
    title="Offline SLM Server",
    description="Locally deployed Small Language Model server",
    version="0.1.0",
    lifespan=lifespan
)


@app.post("/generate_character", response_model=schemas.InferenceResponse)
async def generate_character(request: schemas.InferenceRequest):
    try:
        # Prepare context memory
        request.prompt += prompt_manager['templates'].get("character_generate")
        tokenizer = slm_model["tokenizer"]
        model = slm_model["model"]
        context_memory = [
            {"role": cm.role if cm.role else "NPC", "content": cm.content}
            for cm in (request.context_memory or [])
        ]

        # Add new prompt
        full_context = context_memory or []
        full_context.append({"role": "user", "content": request.prompt})

        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            full_context,
            return_tensors="pt",
            add_generation_prompt=True
        ).to("cuda")

        # Generate response
        outputs = model.generate(
            inputs,
            max_length=request.max_tokens,
            num_return_sequences=1
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(response)
        return {
            "response": response,
            "input_tokens": inputs.shape[1],
            "output_tokens": outputs.shape[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate", response_model=schemas.InferenceResponse)
async def generate_text(request: schemas.InferenceRequest):
    try:
        start_time = time.time()

        # Prepare context memory
        request.prompt += """
Current Personality State:
- Openness: {personality.openness}
- Conscientiousness: {personality.conscientiousness}
...

Current Needs:
{format_needs_status(maslow_needs)}

Emotional State:
{format_emotional_state(plutchik_emotions)}

Given the above psychological state, respond to: {user_input}
Consider:
1. How your personality traits influence your perception
2. Which needs are currently driving your behavior
3. Your emotional state and its impact on your response
"""
        context_memory = [
            {"role": cm.role, "content": cm.content}
            for cm in (request.context_memory or [])
        ]
        tokenizer = slm_model["tokenizer"]
        model = slm_model["model"]
        # Add new prompt
        full_context = context_memory or []
        full_context.append({"role": "user", "content": request.prompt})

        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            full_context,
            return_tensors="pt",
            add_generation_prompt=True
        )


        # Generate response
        outputs = model.generate(
            inputs,
            max_length=request.max_tokens,
            num_return_sequences=1
        )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        processing_time = time.time() - start_time
        print(processing_time, response, processing_time)
        return {
            "response": response,
            "input_tokens": inputs.shape[1],
            "output_tokens": outputs.shape[1]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel
from typing import Optional, Any, Dict, List


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str  # \"user\", \"assistant\", or \"system\"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.7
    conversation_id: Optional[str] = None


class AIResponse(BaseModel):
    text: str
    metadata: Dict[str, Any]


# Routes
@app.post("/completions", response_model=AIResponse)
async def generate_completion(request: CompletionRequest):
    # Process the completion request
    request.prompt += prompt_manager['templates'].get("character_generate")
    response = await model_manager.generate_completion(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system_prompt=request.system_prompt,
        conversation_id=request.conversation_id
    )

    return response


import model_provider as model_manager


@app.post("/chat", response_model=AIResponse)
async def chat(request: ChatRequest):
    # Process the chat request
    response = await model_manager.generate_chat_response(
        request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        conversation_id=request.conversation_id
    )
    return response


@app.get("/models")
async def list_models():
    return {"models": model_manager.get_available_models()}
