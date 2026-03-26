from __future__ import annotations

from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession

from app.RagPipeline.state import AgentState, RagMode
from app.RagPipeline.node import RagNodes


# ── Mode → retrieval node name ────────────────────────────────────────────────

_MODE_TO_NODE: dict[RagMode, str] = {
    "documents": "retriever",
    "web":       "web_search",
    "hybrid":    "both",
}


def _route_after_rewrite(state: AgentState) -> str:
    """
    Skip the LLM router when the client supplies an explicit mode.
    Fall back to the LLM router only for 'auto' / missing mode.
    """
    mode: RagMode = state.get("mode", "hybrid")
    return _MODE_TO_NODE.get(mode, "router")  # unknown mode → LLM router


def _route_after_router(state: AgentState) -> str:
    """Fallback: honour whatever the LLM router decided."""
    tool = state.get("tool", "none")
    return {
        "retrieve_node": "retriever",
        "web_search":    "web_search",
        "both":          "both",
        "none":          "error",
    }.get(tool, "error")


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph(db: AsyncSession):
    nodes    = RagNodes(db=db)
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("query_rewrite", nodes.query_rewrite)
    workflow.add_node("router",        nodes.router_node)     # LLM-based fallback
    workflow.add_node("retriever",     nodes.retriever_node)
    workflow.add_node("web_search",    nodes.web_search)
    workflow.add_node("both",          nodes.both)
    workflow.add_node("reranker",      nodes.reranker_node)
    workflow.add_node("generator",     nodes.generator_node)
    workflow.add_node("reflection",    nodes.reflection)
    workflow.add_node("error",         nodes.error_node)

    # Entry point
    workflow.set_entry_point("query_rewrite")

    # ── After rewrite: branch on mode (or fall through to LLM router) ──────
    workflow.add_conditional_edges(
        "query_rewrite",
        _route_after_rewrite,
        {
            "retriever":  "retriever",
            "web_search": "web_search",
            "both":       "both",
            "router":     "router",    # LLM router path
        },
    )

    # ── LLM router fallback branches ───────────────────────────────────────
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retriever":  "retriever",
            "web_search": "web_search",
            "both":       "both",
            "error":      "error",
        },
    )

    # ── All retrieval paths converge on reranker ───────────────────────────
    for node in ("retriever", "web_search", "both", "error"):
        workflow.add_edge(node, "reranker")

    # ── Linear tail ────────────────────────────────────────────────────────
    workflow.add_edge("reranker",   "generator")
    workflow.add_edge("generator",  "reflection")
    workflow.add_edge("reflection", END)

    return workflow.compile()