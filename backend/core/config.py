import os

# Tìm và đọc file .env từ thư mục gốc dự án (đi lên 2 cấp từ backend/core)
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "..", "..", ".env")

if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip().strip('"').strip("'")
                os.environ[key.strip()] = val

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Dental Q&A AI Agent")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/dental_db")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen3-dental")
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" hoặc "gemini"
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


settings = Settings()
