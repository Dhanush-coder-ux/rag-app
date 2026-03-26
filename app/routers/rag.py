from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.RagPipeline.service import LangGraphService
from app.schemas.rag import HistoryMessage, QuestionRequest, RagResponse
from . import db  # your Depends(get_db) import

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


# ── POST /rag/ask ──────────────────────────────────────────────────────────────

@router.post("/ask", response_model=RagResponse)
async def ask(body: QuestionRequest, db: db):
    """
    Non-streaming RAG endpoint.
    mode: "documents" | "web" | "hybrid"  (default: "hybrid")
    """
    svc          = LangGraphService(db=db)
    history_dicts = [m.model_dump() for m in body.history]

    state = await svc.run(
        question=body.question,
        history=history_dicts,
        mode=body.mode,          # ← pass mode
    )

    logger.info(
        "/ask trace_id=%s mode=%s tool_used=%s",
        state.get("trace_id"), body.mode, state.get("tool_used"),
    )

    return RagResponse(
        answer    = state.get("answer", ""),
        steps     = state.get("steps", []),
        tool_used = state.get("tool_used", "none"),
        trace_id  = state.get("trace_id"),
        error     = state.get("error"),
        sources   = state.get("sources", []),
        history   = [HistoryMessage(**m) for m in state.get("history", [])],
        confidence= state.get("confidence"),
    )


# ── POST /rag/stream ───────────────────────────────────────────────────────────

@router.post("/stream")
async def ask_stream(body: QuestionRequest, db: db):
    """
    SSE streaming RAG endpoint.
    Emits events:  trace | mode | step | sources | tool_used | data (tokens) | [DONE]
    mode: "documents" | "web" | "hybrid"  (default: "hybrid")
    """
    svc           = LangGraphService(db=db)
    history_dicts = [m.model_dump() for m in body.history]

    logger.info(
        "/stream mode=%s question=%r",
        body.mode, body.question,
    )

    return StreamingResponse(
        svc.stream(
            question=body.question,
            history=history_dicts,
            mode=body.mode,      # ← pass mode
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )