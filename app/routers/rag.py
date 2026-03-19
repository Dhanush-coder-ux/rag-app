from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.RagPipeline.service import LangGraphService

from app.schemas.rag import QuestionRequest, RagResponse

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/ask", response_model=RagResponse)
async def ask(body: QuestionRequest, db: AsyncSession = Depends(get_db)):
    svc = LangGraphService(db=db)
    state = await svc.run(body.question)
    return RagResponse(
        answer=state["answer"],
        steps=state["steps"],
        tool_used=state["tool"],
        trace_id=state["trace_id"],
        error=state.get("error"),
    )


@router.post("/stream")
async def ask_stream(body: QuestionRequest, db: AsyncSession = Depends(get_db)):
    svc = LangGraphService(db=db)
    return StreamingResponse(
        svc.stream(body.question),
        media_type="text/event-stream",
            headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables Nginx response buffering
        },
    )