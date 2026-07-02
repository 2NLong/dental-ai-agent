import os

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Dental Q&A AI Agent")
    API_V1_STR: str = os.getenv("API_V1_STR", "/api/v1")
    
    # Ở đây bạn có thể cấu hình LLM sau, ví dụ:
    # GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

settings = Settings()
