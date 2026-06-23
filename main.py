
import sys


if sys.platform !=  "win32" :
    import uvloop
    import asyncio
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
from app.models import chat_session
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import documents, rag
from app.routers import chat_session
from app.routers import voice_ws
        
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Database initialization is now handled sequentially in runner.py
    # to prevent race conditions across multiple gunicorn workers.
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
app.include_router(voice_ws.router)


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
