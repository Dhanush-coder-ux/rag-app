from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession
from app.RagPipeline.state import AgentState
from app.RagPipeline.node import RagNodes



def _route_after_router(state: AgentState) -> str:
    """
    Conditional edge function.
    LangGraph calls this after the router node to pick the next node.
    """
    tool = state.get("tool", "none")
    if tool == "retrieve_docs":
        return "retriever"
    if tool == "tavily_search":
        return "web_search"
    return "error"


def build_rag_graph(db: AsyncSession) -> "CompiledGraph":  # type: ignore[name-defined]
    """
    Builds and compiles the RAG graph for a single request.
    Node instances are scoped to the db session — do not cache this.
    Cache the *structure* separately if startup cost matters.
    """
    nodes = RagNodes(db=db)

    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("router",    nodes.router_node)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("web_search",nodes.web_search_node)
    workflow.add_node("error",     nodes.error_node)
    workflow.add_node("reranker",  nodes.reranker_node)
    workflow.add_node("generator", nodes.generator_node)

    # Entry point
    workflow.set_entry_point("router")

    # Conditional branch after router
    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retriever":  "retriever",
            "web_search": "web_search",
            "error":      "error",
        },
    )

    # Both tool branches converge at reranker
    workflow.add_edge("retriever",  "reranker")
    workflow.add_edge("web_search", "reranker")
    workflow.add_edge("error",      "reranker")   # reranker handles empty ctx gracefully

    # Linear tail
    workflow.add_edge("reranker",  "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()