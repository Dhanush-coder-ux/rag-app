from . import (
APIRouter,db,LangGraphServices
)
from app.routers.web_search_tool import WebSearchTool
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
router = APIRouter(prefix="/query", tags=["Query"])
tool = WebSearchTool(max_results=5)
@router.post("/ask")
async def ask_question( req:AskRequest,db:db ):
    service = LangGraphServices(db=db)
    graph = service.create_rag_graph()
    result = await graph.ainvoke({
        "question":req.question,
        "answer":"",
        "context":[],
        "steps":[],
    })

    return result

@router.post("/search")
async def search_endpoint(query: str):
    return tool.search(query)