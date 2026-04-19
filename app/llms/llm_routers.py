from app.llms.gemini.generation import GeminiGeneration
from app.llms.llama3.generation import Llama3Generation

gemini = GeminiGeneration()
llama = Llama3Generation()

class LLMRouter:

    @staticmethod
    def _build_rag_prompt(question: str, context_chunks: list[str]) -> str:
        """Helper to format the RAG prompt for Llama 3"""
        context = "\n\n---\n\n".join(context_chunks)
        return f"""
        You are a helpful assistant. Use ONLY the context below to answer the question.
        If the answer cannot be found in the context, say "I don't have enough information to answer that."

        Context:
        {context}

        Question: {question}

        Answer:
        """

    @staticmethod
    async def generate_answer_stream(question: str, context_chunks: list[str], model: str = "auto"):
        if model == "gemini":
            try:
                async for chunk in gemini.generate_answer_stream(question, context_chunks):
                    yield chunk
            except Exception as e:
                if "429" in str(e):
                    # ✨ FIXED: Build prompt and use generate_text
                    prompt = LLMRouter._build_rag_prompt(question, context_chunks)
                    yield llama.generate_text(prompt)
                else:
                    raise e

        elif model == "llama3":
            # ✨ FIXED: Build prompt and use generate_text
            prompt = LLMRouter._build_rag_prompt(question, context_chunks)
            yield llama.generate_text(prompt)

        else: # auto
            try:
                async for chunk in gemini.generate_answer_stream(question, context_chunks):
                    yield chunk
            except Exception:
                # ✨ FIXED: Build prompt and use generate_text
                prompt = LLMRouter._build_rag_prompt(question, context_chunks)
                yield llama.generate_text(prompt)

    @staticmethod
    async def rewrite_query(question: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await gemini.rewrite_query(question)
            except Exception as e:
                if "429" in str(e):
                    return llama.rewrite_query(question)
                raise e

        elif model == "llama3":
            return llama.rewrite_query(question)

        else: # auto
            try:
                return await gemini.rewrite_query(question)
            except Exception:
                return llama.rewrite_query(question)

    @staticmethod
    async def generate_chat_title(message: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await gemini.generate_chat_title(message)
            except Exception as e:
                if "429" in str(e):
                    return llama.generate_chat_title(message)
                raise e

        elif model == "llama3":
            return llama.generate_chat_title(message)

        else: # auto
            try:
                return await gemini.generate_chat_title(message)
            except Exception:
                return llama.generate_chat_title(message)
            
    @staticmethod
    async def select_route(prompt: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await gemini.select_route(prompt)
            except Exception as e:
                if "429" in str(e):
                    return llama.select_route(prompt)
                raise e

        elif model == "llama3":
            return llama.select_route(prompt)

        else: # auto
            try:
                return await gemini.select_route(prompt)
            except Exception:
                return llama.select_route(prompt)
            
    @staticmethod
    async def generate_text(prompt: str, model: str = "auto") -> str:
        if model == "gemini":
            try:
                return await gemini.generate_text(prompt)
            except Exception as e:
                if "429" in str(e):
                    return llama.generate_text(prompt)
                raise e

        elif model == "llama3":
            return llama.generate_text(prompt)

        else: # auto
            try:
                return await gemini.generate_text(prompt)
            except Exception:
                return llama.generate_text(prompt)