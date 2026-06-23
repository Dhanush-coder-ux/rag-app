# app/llms/embedding_router.py

from app.llms.gemini.embeddings  import GeminiEmbeddings
from app.llms.llama3.embeddings  import Llama3Embeddings
from app.llms.nvidia.embeddings  import NvidiaEmbeddings
from app.core.config import settings

gemini = GeminiEmbeddings()
llama  = Llama3Embeddings()
nvidia = NvidiaEmbeddings()


class EmbeddingRouter:

    @staticmethod
    async def get_embedding(text: str):
        if settings.EMBEDDING_PROVIDER == "gemini":
            return await gemini.get_embedding(text)

        elif settings.EMBEDDING_PROVIDER == "ollama":
            return await llama.get_embedding(text)

        elif settings.EMBEDDING_PROVIDER == "nvidia":
            return await nvidia.get_embedding(text)

        else:
            raise ValueError(f"Invalid embedding provider: {settings.EMBEDDING_PROVIDER}")

    @staticmethod
    async def get_query_embedding(text: str):
        if settings.EMBEDDING_PROVIDER == "gemini":
            return await gemini.get_query_embedding(text)

        elif settings.EMBEDDING_PROVIDER == "ollama":
            return await llama.get_query_embedding(text)

        elif settings.EMBEDDING_PROVIDER == "nvidia":
            return await nvidia.get_query_embedding(text)

        else:
            raise ValueError(f"Invalid embedding provider: {settings.EMBEDDING_PROVIDER}")