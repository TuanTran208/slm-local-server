from fastapi import APIRouter, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import time
import uuid

router = APIRouter(
    prefix="/api/conversations",
    tags=["conversations"],
)


class Message(BaseModel):
    role: str
    content: str
    created_at: int = Field(default_factory=lambda: int(time.time()))


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    messages: List[Message] = []
    model: str
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))


# In-context memory management
class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, Conversation] = {}
    
    def create_conversation(self, model: str, system_message: Optional[str] = None) -> Conversation:
        """Create a new conversation"""
        conv = Conversation(model=model)
        
        # Add system message if provided
        if system_message:
            conv.messages.append(Message(role="system", content=system_message))
            
        self.conversations[conv.id] = conv
        return conv
    
    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self.conversations.get(conv_id)
    
    def add_message(self, conv_id: str, role: str, content: str) -> Optional[Message]:
        """Add message to conversation"""
        conv = self.conversations.get(conv_id)
        if not conv:
            return None
            
        message = Message(role=role, content=content)
        conv.messages.append(message)
        conv.updated_at = int(time.time())
        return message
    
    def list_conversations(self) -> List[Conversation]:
        """List all conversations"""
        return list(self.conversations.values())
    
    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation"""
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            return True
        return False


# Create global instance
conversation_manager = ConversationManager()


# API endpoints
@router.post("/", response_model=Conversation)
async def create_conversation(
    model: str, 
    system_message: Optional[str] = None
):
    """Create a new conversation"""
    return conversation_manager.create_conversation(model, system_message)


@router.get("/", response_model=List[Conversation])
async def list_conversations():
    """List all conversations"""
    return conversation_manager.list_conversations()


@router.get("/{conv_id}", response_model=Conversation)
async def get_conversation(conv_id: str):
    """Get conversation by ID"""
    conv = conversation_manager.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.post("/{conv_id}/messages", response_model=Message)
async def add_message(conv_id: str, role: str, content: str):
    """Add message to conversation"""
    message = conversation_manager.add_message(conv_id, role, content)
    if not message:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return message


@router.delete("/{conv_id}", response_model=Dict[str, bool])
async def delete_conversation(conv_id: str):
    """Delete a conversation"""
    success = conversation_manager.delete_conversation(conv_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True}
