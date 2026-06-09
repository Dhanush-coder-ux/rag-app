from datetime import datetime
from pydantic import BaseModel, HttpUrl


class DocumentOut(BaseModel):
    id: int
    filename: str
    status: str
    source_type: str = "file"
    source_url: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ChunkOut(BaseModel):
    id: int
    document_id: int
    content: str
    chunk_index: int

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkOut]
    
class TaskResponse(BaseModel):
    task_id: str


class AskRequest(BaseModel):
    question: str
    session_id: str


class YouTubeIngestRequest(BaseModel):
    """Request model for ingesting YouTube video transcripts"""
    url: str  # YouTube URL (flexible format)
    model: str = "gemini"  # Embedding model ("gemini" or "llama3")
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "model": "gemini"
            }
        }