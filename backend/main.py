from fastapi import FastAPI
from core.config import settings
from app.api.router import main_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API cho hệ thống AI Agent hỏi đáp nha khoa thường thức",
    version="0.1.0"
)

app.include_router(main_router)

@app.get("/")
def read_root():
    return {
        "message": "Chào mừng đến với API Hỏi đáp Nha khoa Thường thức!",
        "status": "active"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}
