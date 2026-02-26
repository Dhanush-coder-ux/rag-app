from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.document import DocumentOut
from app.services.document_service import ingest_document, list_documents, delete_document

router = APIRouter(prefix="/documents", tags=["Documents"])
# for setting the type of docs
ALLOWED_TYPES = {"application/pdf", "text/plain"}


@router.post("/upload", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF or .txt file, chunk it, embed it, and store in NeonDB."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use PDF or plain text.",
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    doc = await ingest_document(
        db=db,
        filename=file.filename or "untitled",
        file_bytes=file_bytes,
        content_type=file.content_type,
    )
    return doc


@router.get("/", response_model=list[DocumentOut])
async def get_documents(db: AsyncSession = Depends(get_db)):
    return await list_documents(db)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(document_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await delete_document(db, document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
