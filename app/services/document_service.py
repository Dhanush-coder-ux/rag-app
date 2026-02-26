"""
Core RAG document operations:
  ->ingest: upload → chunk → embed → store
  ->search: embed query → cosine similarity → return top-k chunks
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.document import Document, Chunk
from app.services.gemini import get_embedding, get_query_embedding
from app.services.chunker import chunk_text, extract_text_from_pdf
from app.core.config import settings


async def ingest_document(
    db: AsyncSession,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> Document:
    """
    Full ingestion pipeline:
      1. Create a Document record (status=processing)
      2. Extract text (PDF or plain text)
      3. Chunk the text
      4. Embed each chunk via Gemini
      5. Persist chunks with embeddings
      6. Mark document as 'ready'
    """
    # 1. Create document record
    doc = Document(filename=filename, status="processing")
    db.add(doc)
    await db.flush()  # get doc.id without committing

    try:
        if content_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_bytes)
        else:
            raw_text = file_bytes.decode("utf-8", errors="replace")

        # 3. Chunk
        chunks_text = chunk_text(raw_text)

        # 4 & 5. Embed and store each chunk
        for idx, chunk_content in enumerate(chunks_text):
            embedding = await get_embedding(chunk_content)
            chunk = Chunk(
                document_id=doc.id,
                content=chunk_content,
                chunk_index=idx,
                embedding=embedding,
            )
            db.add(chunk)

        # 6. Mark ready
        doc.status = "ready"
        await db.commit()
        await db.refresh(doc)

    except Exception as exc:
        await db.rollback()
        doc.status = "failed"
        await db.commit()
        raise exc

    return doc


async def similarity_search(
    db: AsyncSession,
    question: str,
    top_k: int | None = None,
) -> list[Chunk]:
    """
    Embed the query and retrieve the top-k most similar chunks using
    pgvector cosine distance (<=>).
    """
    top_k = top_k or settings.TOP_K_RESULTS
    query_embedding = await get_query_embedding(question)

    # pgvector cosine distance operator: <=>
    # Cast the Python list to a vector literal that Postgres understands
    stmt = (
        select(Chunk)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def list_documents(db: AsyncSession) -> list[Document]:
    result = await db.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())


async def delete_document(db: AsyncSession, document_id: int) -> bool:
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        return False
    await db.delete(doc)
    await db.commit()
    return True
