from app.models import chat_session
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import init_db
from app.routers import documents, rag
from app.routers import chat_session



@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation + LangGraph using FastAPI + SQLAlchemy + pgvector (PgAdmin) + Gemini",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(rag.router)
app.include_router(chat_session.router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
