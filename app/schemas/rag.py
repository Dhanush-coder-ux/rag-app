# app/schemas/rag.py

from __future__ import annotations
from pydantic import BaseModel
from typing import List, Literal


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SourceItem(BaseModel):
    url:         str | None = None
    title:       str | None = None
    snippet:     str | None = None
    source_type: Literal["web", "document"] = "web"


class QuestionRequest(BaseModel):
    question:   str
    session_id: int | None = None 
    mode:       Literal["documents", "web", "hybrid"] = "hybrid"
    history:    List[HistoryMessage] = []


class RagResponse(BaseModel):
    answer:     str
    steps:      List[str]            = []
    session_id: int | None           = None
    tool_used:  str                  = "none"
    trace_id:   str | None           = None
    error:      str | None           = None
    sources:    List[SourceItem]     = []   # ← typed properly now
    history:    List[HistoryMessage] = []
    confidence: float | None         = None