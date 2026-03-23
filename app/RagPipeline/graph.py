from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession
from app.RagPipeline.state import AgentState
from app.RagPipeline.node import RagNodes

def _route_after_router(state: AgentState) -> str:
    tool = state.get("tool", "none")
    if tool == "retrieve_node":
        return "retriever"
    elif tool == "web_search":
        return "web_search"
    elif tool == "both":
        return "both"

    return "error"

def build_rag_graph(db: AsyncSession):
    nodes = RagNodes(db=db)
    workflow = StateGraph(AgentState)
    
    workflow.add_node("query_rewrite", nodes.query_rewrite)
    workflow.add_node("router", nodes.router_node)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("both", nodes.both)
    workflow.add_node("error", nodes.error_node)
    workflow.add_node("reflection", nodes.reflection)
    workflow.add_node("reranker", nodes.reranker_node)
    workflow.add_node("generator", nodes.generator_node)

    workflow.set_entry_point("query_rewrite")
    workflow.add_edge("query_rewrite", "router")

    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retriever": "retriever",
            "web_search": "web_search",
            "both": "both",
            "error": "error",
        },
    )
  
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("web_search", "reranker")
    workflow.add_edge("both", "reranker") # Fixed: This was entirely missing!
    workflow.add_edge("error", "reranker")  

    workflow.add_edge("reranker", "generator")
    
    # Fixed: Connected generator to reflection, and reflection to END (no more infinite loop)
    workflow.add_edge("generator", "reflection")
    workflow.add_edge("reflection", END)

    return workflow.compile()