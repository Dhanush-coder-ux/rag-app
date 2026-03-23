from typing import TypedDict, List, Optional, Literal

ToolName = Literal["retrieve_node", "web_search", "both", "none"]

class AgentState(TypedDict, total=False):
    question: str
    rewritten_question: str
    context: List[str]
    answer: str
    steps: List[str]
    tool: ToolName
    sources: List[str]
    history: List[str]
    error: Optional[str]
    trace_id: str
    created_at: str