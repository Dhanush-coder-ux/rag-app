from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.document import TaskResponse,DocumentOut
from app.services.document_service import ingest_document, list_documents, delete_document
from app.core.worker import ingest_document_task

router = APIRouter(prefix="/documents", tags=["Documents"])
# for setting the type of docs
ALLOWED_TYPES = {"application/pdf", "text/plain"}


@router.post("/upload", response_model=TaskResponse)
async def upload_document(
    file: UploadFile = File(...)
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")

    file_bytes = await file.read()

    task = ingest_document_task.delay(
        filename=file.filename,
        file_bytes=file_bytes,
        content_type=file.content_type
    )

    return {"task_id": task.id}


@router.get("/", response_model=list[DocumentOut])
async def get_documents(db: AsyncSession = Depends(get_db)):
    return await list_documents(db)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(document_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
