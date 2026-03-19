from . import (
APIRouter,db,LangGraphServices
)
from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
router = APIRouter(prefix="/query", tags=["Query"])

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
