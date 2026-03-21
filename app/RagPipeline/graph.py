from langgraph.graph import StateGraph, END
from sqlalchemy.ext.asyncio import AsyncSession
from app.RagPipeline.state import AgentState
from app.RagPipeline.node import RagNodes



def _route_after_router(state: AgentState) -> str:

    tool = state.get("tool", "none")
    if tool == "retrieve_docs":
        return "retriever"
    if tool == "web_search":
        return "web_search"
    return "error"


def build_rag_graph(db: AsyncSession) -> "CompiledGraph":  # type: ignore[name-defined]

    nodes = RagNodes(db=db)

    workflow = StateGraph(AgentState)

    workflow.add_node("router",    nodes.router_node)
    workflow.add_node("retriever", nodes.retriever_node)
    workflow.add_node("web_search",nodes.web_search_node)
    workflow.add_node("error",     nodes.error_node)
    workflow.add_node("reranker",  nodes.reranker_node)
    workflow.add_node("generator", nodes.generator_node)


    workflow.set_entry_point("router")


    workflow.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "retriever":  "retriever",
            "web_search": "web_search",
            "error":      "error",
        },
    )

  
    workflow.add_edge("retriever",  "reranker")
    workflow.add_edge("web_search", "reranker")
    workflow.add_edge("error",      "reranker")  

    workflow.add_edge("reranker",  "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()