import requests
from app.core.config import settings

class Llama3Embeddings:
    def get_embedding(self, text: str) -> list[float]:
        response = requests.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={
                "model": "nomic-embed-text",
            "prompt": text
        }
        )
        return response.json()["embedding"]


    def get_query_embedding(self, text: str) -> list[float]:
        return self.get_embedding(text)