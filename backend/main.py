import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Thêm thư mục backend vào sys.path để giải quyết import
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.config import settings
from app.api.router import main_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API cho hệ thống AI Agent hỏi đáp nha khoa thường thức",
    version="1.0.0"
)

# Cấu hình CORS cho phép gọi API từ frontend bất kỳ
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount main router của ứng dụng
app.include_router(main_router)

@app.get("/")
def read_root():
    return {
        "message": "Chào mừng đến với API Hỏi đáp Nha khoa Thường thức!",
        "status": "active",
        "model": settings.OLLAMA_MODEL
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
