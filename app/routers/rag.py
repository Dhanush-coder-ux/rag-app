from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.rag_services.chat_service import ChatServices
from app.RagPipeline.service import LangGraphService
from app.schemas.rag import HistoryMessage, QuestionRequest, RagResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


# ── POST /rag/ask ──────────────────────────────────────────────────────────────

@router.post("/ask", response_model=RagResponse)
async def ask(
    body: QuestionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Non-streaming RAG endpoint.
    mode: "documents" | "web" | "hybrid"  (default: "hybrid")
    """
    svc           = LangGraphService(db=db)
    chat_service  = ChatServices(db)
    history_dicts = [m.model_dump() for m in body.history]

    # Update sidebar title on first message (only if session exists)
    if body.session_id:
        await chat_service.update_title_if_needed(
            session_id=body.session_id,
            message=body.question,
        )

    # session_id is NOT a LangGraph state param — don't pass it to run()
    state = await svc.run(
        question=body.question,
        history=history_dicts,
        mode=body.mode,
    )

    logger.info(
        "/ask session_id=%s trace_id=%s mode=%s tool_used=%s",
        body.session_id,
        state.get("trace_id"),
        body.mode,
        state.get("tool_used"),
    )

    return RagResponse(
        answer     = state.get("answer", ""),
        steps      = state.get("steps", []),
        session_id = body.session_id,          # ← from request, not state
        tool_used  = state.get("tool_used", "none"),
        trace_id   = state.get("trace_id"),
        error      = state.get("error"),
        sources    = state.get("sources", []),
        history    = [HistoryMessage(**m) for m in state.get("history", [])],
        confidence = state.get("confidence"),
    )


# ── POST /rag/stream ───────────────────────────────────────────────────────────

@router.post("/stream")
async def ask_stream(
    body: QuestionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    SSE streaming RAG endpoint.
    Emits events: trace | mode | step | sources | tool_used | data (tokens) | [DONE]
    mode: "documents" | "web" | "hybrid"  (default: "hybrid")
    """
    svc           = LangGraphService(db=db)
    chat_service  = ChatServices(db)
    history_dicts = [m.model_dump() for m in body.history]

    # Update sidebar title on first message (only if session exists)
    if body.session_id:
        await chat_service.update_title_if_needed(
            session_id=body.session_id,
            message=body.question,
        )

    logger.info(
        "/stream session_id=%s mode=%s question=%r",
        body.session_id, body.mode, body.question,
    )

    # session_id is NOT a LangGraph state param — don't pass it to stream()
    return StreamingResponse(
        svc.stream(
            question=body.question,
            history=history_dicts,
            mode=body.mode,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )