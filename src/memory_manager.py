import time
from typing import Dict, List, Optional, Any
import uuid

# Simple in-memory storage
# In production, consider using Redis, a database, or vector stores
conversations = {}
memory_ttl = 3600  # 1 hour by default


class Conversation:
    def __init__(self, id: str = None):
        self.id = id or str(uuid.uuid4())
        self.messages = []
        self.created_at = time.time()
        self.last_updated = time.time()

    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        self.last_updated = time.time()

    def get_history_text(self):
        """Get conversation history as a formatted string"""
        result = ""
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                # Skip system messages in the formatted output
                continue
            elif role == "user":
                result += f"User: {content}"
            elif role == "assistant":
                result += f"Assistant: {content}"

        return result


    def is_expired(self):
        """Check if conversation has expired"""
        return (time.time() - self.last_updated) > memory_ttl


def create_conversation() -> str:
    """Create a new conversation and return its ID"""
    conv = Conversation()
    conversations[conv.id] = conv
    return conv.id


def get_conversation(conversation_id: str) -> Optional[Conversation]:
    """Get a conversation by ID, return None if not found or expired"""
    conv = conversations.get(conversation_id)
    if not conv:
        return None

    if conv.is_expired():
        # Clean up expired conversation
        del conversations[conversation_id]
        return None

    return conv


def get_conversation_history(conversation_id: str) -> str:
    """Get conversation history as text, or empty string if not found"""
    conv = get_conversation(conversation_id)
    if not conv:
        return ""

    return conv.get_history_text()


def update_conversation_history(conversation_id: str, user_message: str, assistant_response: str):
    """Add messages to conversation history"""
    conv = get_conversation(conversation_id)
    if not conv:
        # Create new conversation if it doesn't exist
        conv = Conversation(id=conversation_id)
        conversations[conversation_id] = conv

    # Add the new messages
    conv.add_message("user", user_message)
    conv.add_message("assistant", assistant_response)


def cleanup_expired_conversations():
    """Remove expired conversations"""
    to_remove = []
    for conv_id, conv in conversations.items():
        if conv.is_expired():
            to_remove.append(conv_id)

    for conv_id in to_remove:
        del conversations[conv_id]


# Run cleanup periodically (in a production app, this would be in a background task)
def start_cleanup_task():
    import threading

    def cleanup_task():
        while True:
            cleanup_expired_conversations()
            time.sleep(300)  # Check every 5 minutes

    thread = threading.Thread(target=cleanup_task, daemon=True)
    thread.start()


# Start the cleanup task when the module is imported
start_cleanup_task()
