from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.rag_services.chat_service import ChatServices
from app.core.database import get_db
from app.schemas.chat import ChatSessionResponse, ChatSessionListResponse, ChatMessageResponse

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/create", response_model=ChatSessionResponse)
async def create_chat_session(db: AsyncSession = Depends(get_db)):
    try:
        return await ChatServices(db).create_chat_session()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/", response_model=list[ChatSessionListResponse])
async def get_chat_sessions(db: AsyncSession = Depends(get_db)):
    try:
        return await ChatServices(db).get_chat_sessions()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(session_id: int, db: AsyncSession = Depends(get_db)):
    chat = await ChatServices(db).get_chat_session(session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat


@router.get("/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_chat_messages(session_id: int, db: AsyncSession = Depends(get_db)):
    svc = ChatServices(db)
    if not await svc._session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return await svc.get_messages(session_id)


@router.delete("/{session_id}")
async def delete_chat_session(session_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await ChatServices(db).delete_chat_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}