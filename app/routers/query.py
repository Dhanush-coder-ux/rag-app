from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.schemas.document import QueryRequest, QueryResponse, ChunkOut
from app.services.document_service import similarity_search
from app.services.gemini import generate_answer

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    RAG query endpoint:
      1. Embed the user's question
      2. Retrieve top-k similar chunks via pgvector cosine search
      3. Send chunks + question to Gemini for grounded answer
    """
    chunks = await similarity_search(db, payload.question, payload.top_k)

    if not chunks:
        return QueryResponse(
            answer="No relevant documents found. Please upload some documents first.",
            sources=[],
        )

    context_texts = [c.content for c in chunks]
    answer = await generate_answer(payload.question, context_texts)

    return QueryResponse(
        answer=answer,
        sources=[ChunkOut.model_validate(c) for c in chunks],
    )
