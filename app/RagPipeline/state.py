from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

ToolName = Literal["retrieve_node", "web_search", "both", "none"]
RagMode  = Literal["documents", "web", "hybrid"]


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    question:           str
    rewritten_question: str
    mode:               RagMode          # NEW — injected at graph entry

    # ── Pipeline internals ─────────────────────────────────────────────────
    context:            List[str]
    answer:             str
    steps:              List[str]
    tool:               ToolName
    sources:            List[str]        # raw URL strings; enriched in service layer

    # ── Conversation ───────────────────────────────────────────────────────
    history:            List[HistoryMessage]

    # ── Meta ───────────────────────────────────────────────────────────────
    error:              Optional[str]
    trace_id:           str
    created_at:         str
    confidence:         Optional[float]  # NEW — optional quality signal