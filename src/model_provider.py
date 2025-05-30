from transformers import AutoModelForCausalLM, AutoTokenizer

import os


def download_model(model_name="microsoft/Phi-3-mini-4k-instruct", output_dir="./model_files"):
    """Download model and tokenizer files to a local directory"""
    print(f"Downloading model: {model_name} to {output_dir}")
    print("Downloading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Not found {model_name}")
    # Create output directory if it doesn't exist
    output_dir= os.path.join(output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Download the model and tokenizer

    tokenizer.save_pretrained(output_dir)

    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    print(f"Model and tokenizer successfully downloaded to {output_dir}")


import os
from typing import List, Dict, Any, Optional
import asyncio
import time

import memory_manager
from config import  settings
# Load models at startup
models = {}
tokenizers = {}


def load_models():
    model_path = settings.MODEL_NAME
    models["default"] = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizers["default"] = AutoTokenizer.from_pretrained(model_path)

load_models()

def get_available_models():
    return list(models.keys())


async def generate_completion(
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    # Record start time for performance metrics
    start_time = time.time()

    # Get model and tokenizer
    model = models["default"]
    tokenizer = tokenizers["default"]

    # Prepare input
    if system_prompt:
        full_prompt = f"{system_prompt}\\{prompt}"
    else:
        full_prompt = prompt

    # Add conversation history if needed
    if conversation_id:
        history = memory_manager.get_conversation_history(conversation_id)
        if history:
            full_prompt = history + "\\" + full_prompt

    # Tokenize input
    inputs = tokenizer(full_prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=inputs["input_ids"].shape[1] + max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
    )

    # Decode response (remove prompt)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response_text[len(full_prompt):].strip()

    # Update conversation memory if needed
    if conversation_id:
        memory_manager.update_conversation_history(
            conversation_id,
            prompt,
            response_text
        )

    # Calculate metrics
    end_time = time.time()
    processing_time = end_time - start_time

    return {
        "text": response_text,
        "metadata": {
            "processing_time": processing_time,
            "model": "default",
            "input_tokens": len(inputs["input_ids"][0]),
            "output_tokens": len(tokenizer(response_text)["input_ids"])
        }
    }


async def generate_chat_response(
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        conversation_id: Optional[str] = None
) -> Dict[str, Any]:
    # Format chat messages into a single prompt
    formatted_prompt = ""
    for msg in messages:
        print(msg)
        role = msg.role
        content = msg.content

        if role == "system":
            # System messages go at the beginning
            formatted_prompt = content + "\\" + formatted_prompt
        elif role == "user":
            formatted_prompt += f"User: {content}"
        elif role == "assistant":
            formatted_prompt += f"Assistant: {content}"

        formatted_prompt += "Assistant: "

        # Generate completion based on the formatted chat
        system_prompt = None  # Already included in formatted_prompt
        response = await generate_completion(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            conversation_id=conversation_id
        )

    return response


if __name__ == "__main__":
    download_model()
