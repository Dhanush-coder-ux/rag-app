from . import (
APIRouter,db,LangGraphServices
)


router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/ask")
async def ask_question( question: str,db:db ):
    graph = LangGraphServices(db=db).create_rag_graph()
    
    result = await graph.ainvoke({
        "question":question,
        "answer":"",
        "context":[],
        "steps":[]
    })

    return result