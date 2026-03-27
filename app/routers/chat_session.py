# app/api/chat_router.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession          # ← fixed import
from app.rag_services.chat_service import ChatServices   # ← fixed filename
from app.core.database import get_db

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/create")
async def create_chat_session(db: AsyncSession = Depends(get_db)):
    try:
        return await ChatServices(db).create_chat_session()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/")
async def get_chat_sessions(db: AsyncSession = Depends(get_db)):
    try:
        return await ChatServices(db).get_chat_sessions()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{session_id}")
async def get_chat_session(session_id: int, db: AsyncSession = Depends(get_db)):
    chat = await ChatServices(db).get_chat_session(session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Session not found")
    return chat


@router.delete("/{session_id}")
async def delete_chat_session(session_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await ChatServices(db).delete_chat_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}
