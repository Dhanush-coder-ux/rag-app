from __future__ import annotations

import asyncio
import logging
import re
from typing import List

from sqlalchemy.ext.asyncio import AsyncSession

from app.web_search_tool.web_search_tool import WebSearchTool
from app.core.config import settings
from app.rag_services.document_service import DocumentService
from app.rag_services.gemini import generate_answer
from app.RagPipeline.state import AgentState, RagMode, ToolName
from app.helper.rag_pipeline import RagPipeLineHelper

logger = logging.getLogger(__name__)

VALID_TOOLS: set[ToolName]  = {"retrieve_node", "web_search", "both", "none"}
MIN_CONTEXT_LENGTH          = 200
MAX_STEPS                   = 20
TOP_K_RERANK                = settings.RERANKER_TOP_K

_web_search = WebSearchTool(max_results=5)
_helper     = RagPipeLineHelper()

INTERNAL_SOURCE_TAG = "[Source: YOUR-DOCUMENT]"

# ── Hybrid weight: doc chunks are duplicated so they outrank web results ───────
HYBRID_DOC_WEIGHT = 2   # internal doc chunks count twice in merged context


# ── Shared utility ─────────────────────────────────────────────────────────────

def _build_history_block(history: list, max_messages: int = 6) -> str:
    if not history:
        return ""
    turns = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history[-max_messages:]
    ]
    return "Recent conversation:\n" + "\n".join(turns) + "\n\n"


# ── Step labels per mode ────────────────────────────────────────────────────────

def _steps_for_mode(mode: RagMode) -> dict[str, str]:
    """Return canonical step labels keyed by pipeline phase."""
    if mode == "documents":
        return {
            "retrieve":  "📄 Searching documents...",
            "rerank":    "🔍 Ranking results...",
            "generate":  "🧠 Generating answer...",
            "reflect":   "✅ Reviewing answer...",
        }
    if mode == "web":
        return {
            "retrieve":  "🌐 Searching web...",
            "rerank":    "🔍 Ranking results...",
            "generate":  "🧠 Generating answer...",
            "reflect":   "✅ Reviewing answer...",
        }
    # hybrid / default
    return {
        "retrieve_doc": "📄 Searching documents...",
        "retrieve_web": "🌐 Searching web...",
        "rerank":       "🔍 Ranking results...",
        "merge":        "🧠 Combining results...",
        "generate":     "✍️ Generating answer...",
        "reflect":      "✅ Reviewing answer...",
    }


class RagNodes:
    def __init__(self, db: AsyncSession):
        self._db = db

    # ── 1. Query Rewrite ────────────────────────────────────────────────────

    async def query_rewrite(self, state: AgentState) -> dict:
        original     = state["question"].strip()
        history      = state.get("history", [])
        history_block = _build_history_block(history, max_messages=6)

        prompt = (
            "You are a query-optimisation assistant for a RAG search engine.\n"
            "Task: rewrite the user question into a concise, unambiguous search query "
            "suitable for both vector similarity search and web search.\n\n"
            f"{history_block}"
            "Rules:\n"
            "- If the question uses pronouns like 'it', 'that', 'they', resolve them "
            "using the conversation history above.\n"
            "- Remove filler words; make the query self-contained.\n"
            "- IMPORTANT: If the question mentions 'my document', 'my profile', "
            "'my resume', or 'my skills', keep that intent explicit.\n"
            "- Do NOT answer the question.\n"
            "- Do NOT add information that wasn't in the original or history.\n"
            "- Output ONLY the rewritten query — no explanations.\n"
            "- If the question is already clear and specific, return it unchanged.\n\n"
            f"Original question: {original}\n"
            "Rewritten query:"
        )

        try:
            raw       = await generate_answer(prompt, [])
            rewritten = raw.strip().strip('"').strip("'")
        except Exception as exc:
            logger.error("query_rewrite LLM error: %s", exc)
            rewritten = original

        if _helper._is_bad_rewrite(original, rewritten):
            logger.warning("query_rewrite rejected bad output. original=%r rewritten=%r",
                           original, rewritten)
            rewritten = original

        logger.info("query_rewrite: %r → %r", original, rewritten)
        return {
            "rewritten_question": rewritten,
            "steps": _helper._safe_steps(state, "query_rewrite"),
        }

    # ── 2. Router (LLM fallback — only used when mode is not set) ──────────

    async def router_node(self, state: AgentState) -> dict:
        q             = state.get("rewritten_question") or state["question"]
        history       = state.get("history", [])
        history_block = _build_history_block(history, max_messages=4)

        prompt = (
            "You are a routing classifier. Output EXACTLY one token from this list:\n"
            "  retrieve_node | web_search | both | none\n\n"
            f"{history_block}"
            "Routing rules:\n"
            "  retrieve_node → internal docs, resumes, profiles, course content, "
            "platform policies, past Q&A\n"
            "  web_search    → current events, live data, public knowledge not in our docs\n"
            "  both          → question mentions a person/entity from our docs AND needs "
            "external knowledge (comparisons, gap analysis, benchmarks)\n"
            "  none          → off-topic, harmful, or completely unanswerable\n\n"
            "Key rule: If the question compares an internal person/document against "
            "external standards or industry knowledge, always use 'both'.\n\n"
            f"Question: {q}\n"
            "Answer (one token only):"
        )

        tool: ToolName = "retrieve_node"
        for attempt in range(3):
            try:
                raw       = await generate_answer(prompt, [])
                candidate = _helper._extract_first_word(raw)
                if candidate in VALID_TOOLS:
                    tool = candidate
                    break
                logger.warning("router_node invalid output %r on attempt %d", raw, attempt)
            except Exception as exc:
                logger.error("router_node LLM error attempt %d: %s", attempt, exc)

        logger.info("router_node → tool=%r for question=%r", tool, q)
        return {"tool": tool, "steps": _helper._safe_steps(state, "router")}

    # ── 3. Retriever (documents mode) ───────────────────────────────────────

    async def retriever_node(self, state: AgentState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        mode  = state.get("mode", "documents")
        label = _steps_for_mode(mode).get("retrieve", "📄 Searching documents...")

        try:
            service = DocumentService(db=self._db)
            chunks  = await service.similarity_search(query)

            context = _helper._deduplicate([
                f"{INTERNAL_SOURCE_TAG}\n{c.content.strip()}"
                for c in chunks
                if c.content and len(c.content.strip()) >= MIN_CONTEXT_LENGTH
            ])

            if not context:
                logger.warning("retriever_node: no usable chunks for query=%r", query)
                return {
                    "context": [],
                    "tool_used": "document",
                    "steps": _helper._safe_steps(state, label),
                    "error": "retriever_empty",
                }

        except Exception as exc:
            logger.error("retriever_node failed: %s", exc)
            return {
                "context": [],
                "tool_used": "document",
                "steps": _helper._safe_steps(state, label),
                "error": f"retriever_exception:{exc}",
            }

        return {
            "context":   context,
            "tool_used": "document",
            "steps":     _helper._safe_steps(state, label),
            "error":     None,
        }

    # ── 4. Web Search (web mode) ────────────────────────────────────────────

    async def web_search(self, state: AgentState) -> dict:
        query = state.get("rewritten_question") or state["question"]
        mode  = state.get("mode", "web")
        label = _steps_for_mode(mode).get("retrieve", "🌐 Searching web...")

        try:
            res = await asyncio.to_thread(_web_search.search, query)
        except Exception as exc:
            logger.error("web_search failed: %s", exc)
            return {
                "context":   [],
                "sources":   [],
                "tool_used": "web",
                "steps":     _helper._safe_steps(state, label),
                "error":     f"web_search_exception:{exc}",
            }

        raw_results = res.get("results", [])
        filtered    = [
            r for r in raw_results
            if r.get("content") and len(r["content"].strip()) >= MIN_CONTEXT_LENGTH
        ]
        context = [
            f"[Source: {r['url']}]\n{r['content'].strip()}"
            for r in filtered if r.get("url")
        ]
        sources = [r["url"] for r in filtered if r.get("url")]

        if not context:
            logger.warning("web_search: no usable results for query=%r", query)

        return {
            "context":   context,
            "sources":   sources,
            "tool_used": "web",
            "steps":     _helper._safe_steps(state, label),
            "error":     None if context else "web_search_empty",
        }

    # ── 5. Both / Hybrid ────────────────────────────────────────────────────

    async def both(self, state: AgentState) -> dict:
        labels = _steps_for_mode("hybrid")

        doc_result, web_result = await asyncio.gather(
            self._retriever_raw(state),
            self._web_search_raw(state),
            return_exceptions=True,
        )

        doc_context: List[str] = []
        web_context: List[str] = []
        sources:     List[str] = []

        if isinstance(doc_result, Exception):
            logger.error("both.retriever failed: %s", doc_result)
        else:
            doc_context = doc_result.get("context", [])

        if isinstance(web_result, Exception):
            logger.error("both.web_search failed: %s", web_result)
        else:
            web_context = web_result.get("context", [])
            sources     = web_result.get("sources", [])

        # ── Intelligent merge: doc chunks weighted higher ────────────────
        # Duplicate doc chunks (HYBRID_DOC_WEIGHT × 2) so they win reranking
        weighted_doc = doc_context * HYBRID_DOC_WEIGHT
        merged       = _helper._deduplicate(weighted_doc + web_context)

        steps = list(state.get("steps", []))
        steps.append(labels["retrieve_doc"])
        steps.append(labels["retrieve_web"])
        steps.append(labels["merge"])

        return {
            "context":   merged,
            "sources":   sources,
            "tool_used": "hybrid",
            "steps":     steps,
            "error":     None if merged else "both_empty",
        }

    # ── 6. Reranker ─────────────────────────────────────────────────────────

    async def reranker_node(self, state: AgentState) -> dict:
        query        = (state.get("rewritten_question") or state["question"]).lower()
        query_tokens = set(re.findall(r'\w+', query))
        mode         = state.get("mode", "hybrid")
        label        = _steps_for_mode(mode).get("rerank", "🔍 Ranking results...")

        def _score(chunk: str) -> float:
            is_internal    = chunk.startswith(INTERNAL_SOURCE_TAG)
            internal_boost = 0.15 if is_internal else 0.0
            chunk_tokens   = set(re.findall(r'\w+', chunk.lower()))
            if not query_tokens:
                return internal_boost
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)
            return overlap + internal_boost

        ranked = sorted(
            state.get("context", []),
            key=_score,
            reverse=True,
        )[:TOP_K_RERANK]

        return {
            "context": ranked,
            "steps":   _helper._safe_steps(state, label),
        }

    # ── 7. Generator ────────────────────────────────────────────────────────

    async def generator_node(self, state: AgentState) -> dict:
        context  = state.get("context", [])
        question = state.get("rewritten_question") or state["question"]
        history  = state.get("history", [])
        mode     = state.get("mode", "hybrid")
        label    = _steps_for_mode(mode).get("generate", "🧠 Generating answer...")

        # Fallback to memory if context is empty
        if not context and history:
            context = [
                f"[Source: MEMORY]\n{m['content']}"
                for m in history if m["role"] == "assistant"
            ]

        if not context:
            return {
                "answer": "I don't have enough information to answer this question.",
                "steps":  _helper._safe_steps(state, label),
            }

        context_block = "\n\n---\n\n".join(context)
        history_block = ""
        if history:
            turns = [
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                for m in history[-6:]
            ]
            history_block = (
                "CONVERSATION HISTORY (for continuity, do NOT re-answer old questions):\n"
                + "\n".join(turns) + "\n\n"
            )

        prompt = (
            "You are a precise question-answering assistant.\n\n"
            "RULES:\n"
            "1. Use BOTH retrieved CONTEXT and CONVERSATION HISTORY.\n"
            "2. INTERNAL documents (tagged [Source: YOUR-DOCUMENT]) have highest priority.\n"
            "3. If context is weak, fall back to conversation history.\n"
            "4. Keep answers concise and factual.\n"
            "5. Never fabricate information.\n\n"
            f"{history_block}"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        try:
            answer = (await generate_answer(prompt, [])).strip()
        except Exception as exc:
            logger.error("generator_node LLM error: %s", exc)
            answer = "An error occurred while generating the answer. Please try again."

        # ── Simple confidence heuristic ─────────────────────────────────
        confidence = self._compute_confidence(context, answer)

        return {
            "answer":     answer,
            "confidence": confidence,
            "steps":      _helper._safe_steps(state, label),
        }

    # ── 8. Reflection ────────────────────────────────────────────────────────

    async def reflection(self, state: AgentState) -> dict:
        answer        = state.get("answer", "")
        question      = state.get("rewritten_question") or state["question"]
        context_block = "\n\n---\n\n".join(state.get("context", []))
        mode          = state.get("mode", "hybrid")
        label         = _steps_for_mode(mode).get("reflect", "✅ Reviewing answer...")

        if not answer:
            return {"answer": answer, "steps": _helper._safe_steps(state, label)}

        prompt = (
            "You are a quality-control reviewer for RAG-generated answers.\n\n"
            "Your job:\n"
            "1. Check if the ANSWER correctly addresses the QUESTION.\n"
            "2. Check if every claim is supported by the CONTEXT.\n"
            "3. If correct and well-grounded — return it unchanged.\n"
            "4. Remove unsupported claims.\n"
            "5. Complete the answer if context supports more.\n"
            "6. Do NOT add information not in CONTEXT.\n\n"
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

        return {
            "answer": improved,
            "steps":  _helper._safe_steps(state, label),
        }

    # ── 9. Error ──────────────────────────────────────────────────────────────

    async def error_node(self, state: AgentState) -> dict:
        logger.warning(
            "error_node reached. question=%r error=%r steps=%r",
            state.get("question"), state.get("error"), state.get("steps"),
        )
        return {
            "context": [],
            "answer":  "Sorry, I encountered an internal error. Please try again.",
            "steps":   _helper._safe_steps(state, "❌ Internal error"),
            "error":   state.get("error") or "unknown_error",
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _retriever_raw(self, state: AgentState) -> dict:
        """Used by `both` node."""
        query   = state.get("rewritten_question") or state["question"]
        service = DocumentService(db=self._db)
        chunks  = await service.similarity_search(query)
        context = _helper._deduplicate([
            f"{INTERNAL_SOURCE_TAG}\n{c.content.strip()}"
            for c in chunks
            if c.content and len(c.content.strip()) >= MIN_CONTEXT_LENGTH
        ])
        return {"context": context}

    async def _web_search_raw(self, state: AgentState) -> dict:
        """Used by `both` node."""
        query    = state.get("rewritten_question") or state["question"]
        res      = await asyncio.to_thread(_web_search.search, query)
        filtered = [
            r for r in res.get("results", [])
            if r.get("content") and len(r["content"].strip()) >= MIN_CONTEXT_LENGTH
        ]
        context  = [
            f"[Source: {r['url']}]\n{r['content'].strip()}"
            for r in filtered if r.get("url")
        ]
        sources  = [r["url"] for r in filtered if r.get("url")]
        return {"context": context, "sources": sources}

    @staticmethod
    def _deduplicate(items: List[str]) -> List[str]:
        return _helper._deduplicate(items)

    @staticmethod
    def _compute_confidence(context: List[str], answer: str) -> float:
        """
        Lightweight heuristic confidence score (0.0–1.0).
        Based on: context availability + answer length reasonableness.
        Replace with an LLM-based scorer for production.
        """
        if not context:
            return 0.1
        if not answer or len(answer) < 20:
            return 0.3
        # How many context chunks contain tokens from the answer?
        answer_tokens = set(re.findall(r'\w+', answer.lower()))
        grounded = sum(
            1 for c in context
            if len(answer_tokens & set(re.findall(r'\w+', c.lower()))) > 3
        )
        ratio = min(grounded / max(len(context), 1), 1.0)
        return round(0.4 + 0.6 * ratio, 2)