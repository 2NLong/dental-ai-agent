import os
from dotenv import load_dotenv

# Tìm và đọc file .env từ thư mục gốc dự án (đi lên 2 cấp từ backend/core)
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/core
backend_dir = os.path.dirname(current_dir) # backend
project_root = os.path.dirname(backend_dir) # root/

load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Dental Q&A AI Agent")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/dental_db")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-qat")
    
    RAW_DATA_DIR: str = os.path.join(project_root, "data", "raw")
    PROCESSED_DATA_DIR: str = os.path.join(project_root, "data", "processed")

settings = Settings()

# Tạo các thư mục dữ liệu nếu chưa tồn tại
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
