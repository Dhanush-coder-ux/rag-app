from google import genai
from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)

class GeminiGeneration:
    async def generate_text(self, prompt: str) -> str:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
        )
        return response.text.strip()

    async def rewrite_query(self,question: str):
        prompt = f"""
        Rewrite the user query into a clear question.

        User query:
        {question}

        Rewritten query:
        """
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text.strip()

    async def generate_answer_stream( self, question: str, context_chunks: list[str]):
        context = "\n\n---\n\n".join(context_chunks)

        prompt = f"""
                You are a helpful assistant. Use ONLY the context below to answer the question.
                If the answer cannot be found in the context, say "I don't have enough information to answer that."

                Context:
                {context}
                Question: {question}
                Answer:
                """
        async for response in client.aio.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=prompt,
        ):
            if response.text:
                yield response.text


    async def generate_chat_title( self, message: str) -> str:
        prompt = f"""
        Generate a short 3-5 word title for this chat.

        Message:
        {message}

        Only return the title.
        """

        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text.strip().replace("\n", "")
    
    async def select_route(self, prompt: str) -> str:
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text.strip()