from __future__ import annotations

import json
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



@router.post("/ask", response_model=RagResponse)
async def ask(
    body: QuestionRequest,
    db: AsyncSession = Depends(get_db),
) -> RagResponse:
    svc          = LangGraphService(db=db)
    chat_service = ChatServices(db)

    session_id = body.session_id
    if not session_id:
        new_session = await chat_service.create_chat_session()
        session_id = new_session.id
    else:
        if not await chat_service._session_exists(session_id):
            logger.warning("/ask received non-existent session_id=%s", session_id)
            new_session = await chat_service.create_chat_session()
            session_id = new_session.id

    await chat_service.update_title_if_needed(
        session_id=session_id, message=body.question
    )
    await chat_service.save_user_message(session_id=session_id, content=body.question)

    history_dicts = [m.model_dump() for m in body.history]
    state = await svc.run(
        question=body.question, history=history_dicts, mode=body.mode
    )

    logger.info(
        "/ask session_id=%s trace_id=%s mode=%s tool_used=%s",
        session_id, state.get("trace_id"), body.mode, state.get("tool_used"),
    )

    answer = state.get("answer", "")
    if answer:
        await chat_service.save_assistant_message(
            session_id=session_id, content=answer
        )

    return RagResponse(
        answer     = answer,
        steps      = state.get("steps", []),
        session_id = session_id,       
        tool_used  = state.get("tool_used", "none"),
        trace_id   = state.get("trace_id"),
        error      = state.get("error"),
        sources    = state.get("sources", []),
        history    = [HistoryMessage(**m) for m in state.get("history", [])],
        confidence = state.get("confidence"),
    )


@router.post("/stream")
async def ask_stream(
    body: QuestionRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    svc          = LangGraphService(db=db)
    chat_service = ChatServices(db)

    if body.session_id:
        await chat_service.update_title_if_needed(
            session_id=body.session_id,
            message=body.question,
        )

    await chat_service.save_user_message(
        session_id=body.session_id,
        content=body.question,
    )

    history_dicts = [m.model_dump() for m in body.history]

    logger.info(
        "/stream session_id=%s mode=%s question=%r",
        body.session_id, body.mode, body.question,
    )

    return StreamingResponse(
        _stream_and_store(
            svc=svc,
            chat_service=chat_service,
            session_id=body.session_id,
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


async def _stream_and_store(
    *,
    svc: LangGraphService,
    chat_service: ChatServices,
    session_id: int | None,
    question: str,
    history: list[dict],
    mode: str,
):

    accumulated: list[str] = []

    try:
        async for chunk in svc.stream(
            question=question,
            history=history,
            mode=mode,
        ):
            yield chunk
            text = _extract_chunk_text(chunk)
            if text:
                accumulated.append(text)

    except Exception:
        logger.exception(
            "/stream generation error for session %s — partial response may be stored",
            session_id,
        )

    finally:
        full_response = "".join(accumulated).strip()
        if full_response and session_id:
            await chat_service.save_assistant_message(
                session_id=session_id,
                content=full_response,
            )

def _extract_chunk_text(chunk: str | bytes) -> str:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")

    chunk = chunk.strip()
    if not chunk or "data:" not in chunk:
        return ""

    # 🚨 ignore ALL non-answer events
    if chunk.startswith("event: step") \
        or chunk.startswith("event: trace") \
        or chunk.startswith("event: mode") \
        or chunk.startswith("event: sources") \
        or chunk.startswith("event: tool_used"):
        return ""

    payload = chunk.split("data:", 1)[1].strip()

    if payload in ("[DONE]", ""):
        return ""

    try:
        parsed = json.loads(payload)

        # ✅ only extract actual answer text
        if isinstance(parsed, dict):
            return parsed.get("answer", "") or parsed.get("content", "") or ""
        
        if isinstance(parsed, str):
            return parsed

    except json.JSONDecodeError:
        # ❌ DO NOT return raw payload anymore
        return ""

    return ""