from typing import TypedDict, List, Optional, Literal

ToolName = Literal["retrieve_docs", "tavily_search", "none"]

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    steps: List[str]
    tool: ToolName
    error: Optional[str]
    trace_id: str
    created_at: str