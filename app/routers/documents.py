from . import (
db,TaskResponse,status,UploadFile,File,HTTPException,ingest_document_task,
DocumentOut,Document,DocumentService,select,APIRouter
)

router = APIRouter(prefix="/documents", tags=["Documents"])
ALLOWED_TYPES = {"application/pdf", "text/plain"}


@router.post("/upload", response_model=TaskResponse,status_code=status.HTTP_202_ACCEPTED)
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
async def get_documents(db: db):
    return await DocumentService(db).list_documents()

@router.get("/{document_id}", response_model=DocumentOut, status_code=status.HTTP_200_OK)
async def get_document(document_id: int, db:db):
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_document(document_id: int, db: db):
    deleted = await DocumentService(db).delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
