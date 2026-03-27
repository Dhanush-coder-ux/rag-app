from google import genai
from google.genai import types
from app.core.config import settings

client = genai.Client(api_key=settings.GEMINI_API_KEY)



async def get_embedding(text: str) -> list[float]:
    response = await client.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=768 
        )
    )

    return response.embeddings[0].values


async def get_query_embedding(text: str) -> list[float]:
    response = await client.aio.models.embed_content(
        model="gemini-embedding-001",
        contents=text,

        config=types.EmbedContentConfig(
            output_dimensionality=768 
        )
    )

    return response.embeddings[0].values


async def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    print(context)
    prompt = f"""
        You are a helpful assistant. Use ONLY the context below to answer the question.
        If the answer cannot be found in the context, say "I don't have enough information to answer that."

        Context:
        {context}

        Question: {question}

        Answer:
        """

    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt,
    )

    return response.text

async def rewrite_query(question: str):
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

async def generate_answer_stream(question: str, context_chunks: list[str]):
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


async def generate_chat_title(message: str) -> str:
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
