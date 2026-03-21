from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.models.document import Document
from app.schemas.document import TaskResponse,DocumentOut
from app.rag_services.document_service import DocumentService
from app.core.worker import ingest_document_task
from sqlalchemy import select
from typing import Annotated
# query

from app.rag_services.gemini import generate_answer
from app.schemas.document import QueryRequest, QueryResponse, ChunkOut
db = Annotated[AsyncSession,Depends(get_db)]