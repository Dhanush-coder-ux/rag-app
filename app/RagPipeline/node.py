import asyncio
import logging
from typing import List,AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from app.web_search_tool.web_search_tool import WebSearchTool
from app.core.config import settings
from app.rag_services.document_service import DocumentService
from app.rag_services.gemini import generate_answer,generate_answer_stream
from app.RagPipeline.state import AgentState,ToolName
logger = logging.getLogger(__name__)

VALID_TOOLS: set[ToolName] = {"retrieve_docs", "web_search"}

_web_search = WebSearchTool(max_results=5)


class RagNodes:

    def __init__(self, db: AsyncSession) :
        self._db = db

    async def router_node(self, state: AgentState) -> dict:
     
        MAX_RETRIES = 2
        prompt = (
            "You are a routing assistant. Respond with EXACTLY one of these "
            "two strings and nothing else:\n"
            "  retrieve_docs   — for questions about internal documents\n"
            "  web_search   — for questions requiring live internet data\n\n"
            f"Question: {state['question']}"
        )

        for attempt in range(MAX_RETRIES + 1):
            try:
                raw = (await generate_answer(prompt, [])).strip().lower()
                tool: ToolName = raw if raw in VALID_TOOLS else "none"
                if tool != "none":
                    break
                logger.warning("Router returned invalid tool %r (attempt %d)", raw, attempt)
            except Exception as exc:
                logger.error("Router LLM error on attempt %d: %s", attempt, exc)
                tool = "none"

        return {
            "tool": tool,
            "steps": state.get("steps", []) + ["router"],
        }

    async def retriever_node(self, state: AgentState) -> dict:
        try:
            service = DocumentService(db=self._db)
            chunks = await service.similarity_search(state["question"])
            context = self._deduplicate([c.content for c in chunks])
        except Exception as exc:
            logger.error("Retriever failed: %s", exc)
            context = []

        return {
            "context": context,
            "steps": state.get("steps", []) + ["retriever"],
            "error": None if context else "retriever returned no results",
        }

    async def web_search_node(self, state: AgentState) -> dict:
        try:
            response = await asyncio.to_thread(
                _web_search.search,
                state["question"]
            )

            context = self._deduplicate([
                f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}"
                for r in response.get("results", [])
                if r.get("content")
            ])

        except Exception as exc:
            logger.error("Custom web search failed: %s", exc)
            context = []

        return {
            "context": context,
            "sources": [r.get("url") for r in response["results"]],
            "steps": state.get("steps", []) + ["web_search"],
            "error": None if context else "web search returned no results",
        }

    
    async def error_node(self, state: AgentState) -> dict:
  
        logger.warning("Entering error fallback for question: %s", state["question"])
        return {
            "context": [],
            "steps": state.get("steps", []) + ["error_fallback"],
            "error": state.get("error") or "routing failed: unknown tool",
        }


    async def reranker_node(self, state: AgentState) -> dict:
 
        TOP_K = settings.RERANKER_TOP_K 

        ranked = sorted(
            state.get("context", []),
            key=lambda c: len(c),
            reverse=True,
        )[:TOP_K]

        return {
            "context": ranked,
            "steps": state.get("steps", []) + ["reranker"],
        }


    async def generator_node(self, state: AgentState) -> dict:
        if not state.get("context"):
            answer = (
                "I could not find relevant information to answer your question. "
                "Please rephrase or try a more specific query."
            )
        else:
            answer = await generate_answer(state["question"], state["context"])

        return {
            "answer": answer,
            "steps": state.get("steps", []) + ["generator"],
        }

    async def generator_node_stream(
        self, state: AgentState
    ) -> AsyncGenerator[str, None]:
    
        if not state.get("context"):
            yield "I could not find relevant information to answer your question."
            return

        async for chunk in generate_answer_stream(state["question"], state["context"]):
            yield chunk

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for item in items:
            key = item[:200]
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out