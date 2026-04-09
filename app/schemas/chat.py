# app/schemas/chat.py
from pydantic import BaseModel
from datetime import datetime

class ChatMessageResponse(BaseModel):
    """
    Safe message schema — content is decompressed str, never raw bytes.
    Only used by endpoints that explicitly decompress before returning.
    """
    id:          int
    role:        str
    content:     str        # decompressed string, NOT the raw LargeBinary bytes
    token_count: int | None = None
    created_at:  datetime

    model_config = {"from_attributes": False}  # built manually, not from ORM directly

class ChatSessionListResponse(BaseModel):
    """Slim schema for session list — no messages."""
    id:         int
    title:      str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

class ChatSessionResponse(BaseModel):
    """Full session schema — no messages (LargeBinary must never leak)."""
    id:         int
    title:      str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}