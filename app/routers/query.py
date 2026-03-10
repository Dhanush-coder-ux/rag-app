from . import (
APIRouter,db,QueryRequest,QueryResponse,DocumentService,generate_answer,ChunkOut,create_rag_graph
)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/", response_model=QueryResponse)
async def query_documents(
    payload: QueryRequest,
    db:db
):
    """
    RAG query endpoint:
      1. Embed the user's question
      2. Retrieve top-k similar chunks via pgvector cosine search
      3. Send chunks + question to Gemini for grounded answer
    """
    chunks = await DocumentService(db).similarity_search(payload.question, payload.top_k)

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
@router.post("/ask")
async def ask_question( question: str,db:db ):
    graph = create_rag_graph(db=db)
    
    result = await graph.ainvoke({
        "question":question,
        "answer":"",
        "context":[],
        "steps":[]
    })

    return result