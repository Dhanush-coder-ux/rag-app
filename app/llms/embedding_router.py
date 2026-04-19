# app/llms/embedding_router.py

from app.llms.gemini.embeddings import GeminiEmbeddings
from app.llms.llama3.embeddings import Llama3Embeddings
from app.core.config import settings

gemini = GeminiEmbeddings()
llama = Llama3Embeddings()


class EmbeddingRouter:

    @staticmethod
    async def get_embedding(text: str):
        if settings.EMBEDDING_PROVIDER == "gemini":
            return await gemini.get_embedding(text)

        elif settings.EMBEDDING_PROVIDER == "ollama":
            return llama.get_embedding(text)

        else:
            raise ValueError("Invalid embedding provider")