import asyncio
import logging
import re
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from app.web_search_tool.web_search_tool import WebSearchTool
from app.core.config import settings
from app.rag_services.document_service import DocumentService
from app.rag_services.gemini import generate_answer
from app.RagPipeline.state import AgentState, ToolName
from app.helper.rag_pipeline import RagPipeLineHelper
logger = logging.getLogger(__name__)

VALID_TOOLS: set[ToolName] = {"retrieve_node", "web_search", "both", "none"}
MIN_CONTEXT_LENGTH = 200   
MAX_STEPS = 20            
TOP_K_RERANK = settings.RERANKER_TOP_K

_web_search = WebSearchTool(max_results=5)
_helper = RagPipeLineHelper() 



class RagNodes:
    def __init__(self, db: AsyncSession):
        self._db = db



    async def query_rewrite(self, state: AgentState) -> dict:

        original = state["question"].strip()

        prompt = (
            "You are a query-optimisation assistant for a RAG search engine.\n"
            "Task: rewrite the user question into a concise, unambiguous search query "
            "suitable for both vector similarity search and web search.\n\n"
            "Rules:\n"
            "- Remove filler words and pronouns; make it self-contained.\n"
            "- Do NOT answer the question.\n"
            "- Do NOT add information that wasn't in the original.\n"
            "- Output ONLY the rewritten query — no explanations, no punctuation changes.\n"
            "- If the question is already clear and specific, return it unchanged.\n\n"
            f"Original question: {original}\n"
            "Rewritten query:"
        )

        try:
            raw = await generate_answer(prompt, [])
            rewritten = raw.strip().strip('"').strip("'")
        except Exception as exc:
            logger.error("query_rewrite LLM error: %s", exc)
            rewritten = original

        if _helper._is_bad_rewrite(original, rewritten):
            logger.warning(
                "query_rewrite rejected (bad output). original=%r rewritten=%r",
                original, rewritten,
            )
            rewritten = original

        logger.info("query_rewrite: %r → %r", original, rewritten)
        return {
            "rewritten_question": rewritten,
            "steps": _helper._safe_steps(state, "query_rewrite"),
        }



    async def router_node(self, state: AgentState) -> dict:

        q = state.get("rewritten_question") or state["question"]

        prompt = (
            "You are a routing classifier. Output EXACTLY one token from this list:\n"
            "  retrieve_docs | web_search | both | none\n\n"
            "Routing rules:\n"
            "  retrieve_node → internal docs, course content, platform policies, past Q&A\n"
            "  web_search    → current events, live data, public knowledge not in our docs\n"
            "  both          → needs internal context AND external knowledge together\n"
            "  none          → off-topic, harmful, or completely unanswerable\n\n"
            "Examples:\n"
            "  Q: What is the refund policy?          → retrieve_node\n"
            "  Q: Who won the 2024 US election?       → web_search\n"
            "  Q: Compare our pricing with AWS costs? → both\n"
            "  Q: Write me a poem about cats          → none\n\n"
            f"Question: {q}\n"
            "Answer (one token only):"
        )

        tool: ToolName = "retrieve_docs" 
        for attempt in range(3):
            try:
                raw = await generate_answer(prompt, [])
                candidate = _helper._extract_first_word(raw)
                if candidate in VALID_TOOLS:
                    tool = candidate
                    break
                logger.warning(
                    "router_node invalid output %r on attempt %d", raw, attempt
                )
            except Exception as exc:
                logger.error("router_node LLM error attempt %d: %s", attempt, exc)

        logger.info("router_node → tool=%r for question=%r", tool, q)
        return {"tool": tool, "steps": _helper._safe_steps(state, "router")}

    # ── 3. Retriever ──────────────────────────────────────────────────────

    async def retriever_node(self, state: AgentState) -> dict:
   
        query = state.get("rewritten_question") or state["question"]
        try:
            service = DocumentService(db=self._db)
            chunks = await service.similarity_search(query)

            context = self._deduplicate([
                c.content for c in chunks
                if c.content and len(c.content.strip()) >= MIN_CONTEXT_LENGTH
            ])

            if not context:
                logger.warning("retriever_node: no usable chunks for query=%r", query)
                return {
                    "context": [],
                    "steps": _helper._safe_steps(state, "retriever"),
                    "error": "retriever_empty",
                }

        except Exception as exc:
            logger.error("retriever_node failed: %s", exc)
            return {
                "context": [],
                "steps":  _helper._safe_steps(state, "retriever"),
                "error": f"retriever_exception:{exc}",
            }

        return {
            "context": context,
            "steps": _helper._safe_steps(state, "retriever"),
            "error": None,
        }



    async def web_search(self, state: AgentState) -> dict:

        query = state.get("rewritten_question") or state["question"]
        try:
            res = await asyncio.to_thread(_web_search.search, query)
        except Exception as exc:
            logger.error("web_search failed: %s", exc)
            return {
                "context": [],
                "sources": [],
                "steps":_helper._safe_steps(state, "web_search"),
                "error": f"web_search_exception:{exc}",
            }

        raw_results = res.get("results", [])

        filtered = [
            r for r in raw_results
            if r.get("content") and len(r["content"].strip()) >= MIN_CONTEXT_LENGTH
        ]

        context = [
            f"[Source: {r['url']}]\n{r['content'].strip()}"
            for r in filtered
            if r.get("url")
        ]
        sources = [r["url"] for r in filtered if r.get("url")]

        if not context:
            logger.warning("web_search: no usable results for query=%r", query)

        return {
            "context": context,
            "sources": sources,
            "steps": _helper._safe_steps(state, "web_search"),
            "error": None if context else "web_search_empty",
        }



    async def both(self, state: AgentState) -> dict:
        doc_result, web_result = await asyncio.gather(
            self._retriever_raw(state),
            self._web_search_raw(state),
            return_exceptions=True,
        )

        doc_context: List[str] = []
        web_context: List[str] = []
        sources: List[str] = []

        if isinstance(doc_result, Exception):
            logger.error("both.retriever failed: %s", doc_result)
        else:
            doc_context = doc_result.get("context", [])

        if isinstance(web_result, Exception):
            logger.error("both.web_search failed: %s", web_result)
        else:
            web_context = web_result.get("context", [])
            sources = web_result.get("sources", [])

        # Interleave: doc first (higher trust), then web
        merged = _helper._deduplicate(doc_context + web_context)

        return {
            "context": merged,
            "sources": sources,
            "steps": _helper._safe_steps(state, "both"),
            "error": None if merged else "both_empty",
        }

  

    async def reranker_node(self, state: AgentState) -> dict:

        query = (state.get("rewritten_question") or state["question"]).lower()
        query_tokens = set(re.findall(r'\w+', query))

        def _score(chunk: str) -> float:
            chunk_tokens = set(re.findall(r'\w+', chunk.lower()))
            if not query_tokens:
                return 0.0
            overlap = len(query_tokens & chunk_tokens)
            return overlap / len(query_tokens)

        ranked = sorted(
            state.get("context", []),
            key=_score,
            reverse=True,
        )[:TOP_K_RERANK]

        return {
            "context": ranked,
            "steps": _helper._safe_steps(state, "reranker"),
        }

    # ── 7. Generator ──────────────────────────────────────────────────────

    async def generator_node(self, state: AgentState) -> dict:

        context = state.get("context", [])
        question = state.get("rewritten_question") or state["question"]

        if not context:
            return {
                "answer": (
                    "I could not find relevant information to answer your question. "
                    "Please try rephrasing or check a more specific source."
                ),
                "steps": _helper._safe_steps(state, "generator"),
            }

        context_block = "\n\n---\n\n".join(context)

        prompt = (
            "You are a precise question-answering assistant.\n\n"
            "STRICT RULES — you MUST follow all of them:\n"
            "1. Answer ONLY using the information in the CONTEXT block below.\n"
            "2. If the context does not contain enough information to answer, "
            'respond with exactly: "I don\'t have enough information to answer this question."\n'
            "3. Do NOT use your training knowledge, prior assumptions, or external facts.\n"
            "4. If context contains [Source: URL] markers, cite them inline.\n"
            "5. Keep the answer concise and factual — no padding or speculation.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            "ANSWER:"
        )

        try:
            answer = await generate_answer(prompt, [])
            answer = answer.strip()
        except Exception as exc:
            logger.error("generator_node LLM error: %s", exc)
            answer = "An error occurred while generating the answer. Please try again."

        return {
            "answer": answer,
            "steps": _helper._safe_steps(state, "generator"),
        }

    # ── 8. Reflection ─────────────────────────────────────────────────────

    async def reflection(self, state: AgentState) -> dict:
     
        answer = state.get("answer", "")
        question = state.get("rewritten_question") or state["question"]
        context_block = "\n\n---\n\n".join(state.get("context", []))

        if not answer:
            return {"answer": answer, "steps": _helper._safe_steps(state, "reflection")}

        prompt = (
            "You are a quality-control reviewer for RAG-generated answers.\n\n"
            "Your job:\n"
            "1. Check if the ANSWER correctly addresses the QUESTION.\n"
            "2. Check if every claim in the ANSWER is supported by the CONTEXT.\n"
            "3. If the answer is correct and well-grounded — return it unchanged.\n"
            "4. If there are unsupported claims — remove them.\n"
            "5. If the answer is incomplete but context supports more — complete it.\n"
            "6. Do NOT add information that is not in the CONTEXT.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER TO REVIEW:\n{answer}\n\n"
            "REVIEWED ANSWER:"
        )

        try:
            improved = (await generate_answer(prompt, [])).strip()
            if not improved or _helper._is_bad_rewrite(answer, improved):
                improved = answer
        except Exception as exc:
            logger.error("reflection LLM error: %s", exc)
            improved = answer

        return {"answer": improved, "steps": _helper._safe_steps(state, "reflection")}

  

    async def error_node(self, state: AgentState) -> dict:
        logger.warning(
            "error_node reached. question=%r error=%r steps=%r",
            state.get("question"), state.get("error"), state.get("steps"),
        )
        return {
            "context": [],
            "answer": "Sorry, I encountered an internal error. Please try again.",
            "steps": _helper._safe_steps(state, "error_fallback"),
            "error": state.get("error") or "unknown_error",
        }

  

    async def _retriever_raw(self, state: AgentState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        service = DocumentService(db=self._db)
        chunks = await service.similarity_search(query)
        context = _helper._deduplicate([
            c.content for c in chunks
            if c.content and len(c.content.strip()) >= MIN_CONTEXT_LENGTH
        ])
        return {"context": context}

    async def _web_search_raw(self, state: AgentState) -> dict:

        query = state.get("rewritten_question") or state["question"]
        res = await asyncio.to_thread(_web_search.search, query)
        filtered = [
            r for r in res.get("results", [])
            if r.get("content") and len(r["content"].strip()) >= MIN_CONTEXT_LENGTH
        ]
        context = [
            f"[Source: {r['url']}]\n{r['content'].strip()}"
            for r in filtered if r.get("url")
        ]
        sources = [r["url"] for r in filtered if r.get("url")]
        return {"context": context, "sources": sources}

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        return _helper._deduplicate(items)