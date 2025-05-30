from pydantic import BaseModel
from typing import List, Optional, Dict

from config import settings
class ContextMemory(BaseModel):
    role: str
    content: str


class InferenceRequest(BaseModel):
    prompt: str
    context_memory: Optional[List[ContextMemory]] = None
    max_tokens: Optional[int] = settings.MAX_TOKENS


class InferenceResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int