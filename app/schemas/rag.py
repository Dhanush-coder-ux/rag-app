from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Mode ──────────────────────────────────────────────────────────────────────

RagMode = Literal["documents", "web", "hybrid"]


# ── History ───────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


# ── Source ────────────────────────────────────────────────────────────────────

class SourceItem(BaseModel):
    """Rich source object returned to the UI."""
    url: Optional[str] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    score: Optional[float] = None
    source_type: Literal["document", "web"] = "web"


# ── Request ───────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    history: List[HistoryMessage] = Field(default_factory=list)
    mode: RagMode = Field(
        default="hybrid",
        description="Controls retrieval strategy: documents | web | hybrid",
    )


# ── Response ──────────────────────────────────────────────────────────────────

class RagResponse(BaseModel):
    answer: str
    steps: List[str] = Field(default_factory=list)
    tool_used: Literal["document", "web", "hybrid", "none"] = "none"
    trace_id: Optional[str] = None
    error: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)
    history: List[HistoryMessage] = Field(default_factory=list)
    confidence: Optional[float] = Field(
        default=None,
        description="Optional 0-1 confidence score",
    )