from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from agent.graph import dental_agent

router = APIRouter()

@router.post("/query", response_model=ChatResponse)
def query_agent(payload: ChatRequest):
    """Gửi câu hỏi tới AI Agent hỏi đáp nha khoa thường thức."""
    try:
        # Gọi tới logic Agent RAG
        result = dental_agent.run(payload.message)
        return {
            "session_id": payload.session_id or "new_session",
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
