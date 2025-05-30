
import torch
from config import settings

from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel:
    def __init__(self, model_name=settings.MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def generate_response(self, prompt: str, context_memory: list = None, max_tokens: int = settings.MAX_TOKENS):
        # Prepare context memory if provided
        full_context = context_memory or []
        full_context.append({"role": "user", "content": prompt})

        # Tokenize input
        inputs = self.tokenizer.apply_chat_template(
            full_context,
            return_tensors="pt",
            add_generation_prompt=True
        )

        # Generate response
        outputs = self.model.generate(
            inputs,
            max_length=max_tokens,
            num_return_sequences=1
        )

        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "response": response,
            "input_tokens": inputs.shape[1],
            "output_tokens": outputs.shape[1]
        }