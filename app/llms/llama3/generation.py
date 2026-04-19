import requests
from app.core.config import settings

class Llama3Generation:
    def __init__(self):
        self.session = requests.Session()


    def rewrite_query(self, question: str) -> str:
        prompt = f"""
    Rewrite the user query into a clear question.

    User query:
    {question}

    Rewritten query:
    """
        response = self.session.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_chat_title(self, message: str) -> str:
        prompt = f"""
    Generate a short 3-5 word title for this chat.

    Message:
    {message}

    Only return the title.
    """

        response = self.session.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        
        response.raise_for_status()
        return response.json()["response"].strip()
    
    def select_route(self, prompt: str) -> str:
        response = self.session.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    
    def generate_text(self, prompt: str) -> str:
        response = self.session.post(
           f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"]