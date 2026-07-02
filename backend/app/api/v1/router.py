from fastapi import APIRouter
from app.api.v1.endpoints import auth, chat, documents

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(chat.router, prefix="/chat", tags=["AI Chat Agent"])
api_router.include_router(documents.router, prefix="/documents", tags=["Document Management"])
