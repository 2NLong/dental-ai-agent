import requests
from agent.state import AgentState
from agent.prompts import DENTAL_SYSTEM_PROMPT
from agent.tools import search_dental_pdf
from core.config import settings

class DentalAgent:
    """Quy trình điều hướng câu hỏi và suy nghĩ của AI Agent nha khoa (RAG)."""
    
    def __init__(self):
        self.system_prompt = DENTAL_SYSTEM_PROMPT
        self.ollama_url = f"{settings.OLLAMA_URL}/api/chat"
        self.model_name = settings.OLLAMA_MODEL
        
    def run(self, user_question: str) -> dict:
        # Bước 1: Gọi retriever tool để tìm tài liệu PDF nha khoa và danh sách nguồn
        context, sources = search_dental_pdf(user_question)
        
        # Bước 2: Chuẩn bị prompt với ngữ cảnh
        prompt = f"""Dưới đây là tài liệu ngữ cảnh nha khoa liên quan:
---
{context}
---

Dựa trên tài liệu ngữ cảnh trên, hãy trả lời câu hỏi sau của người dùng một cách chính xác, ngắn gọn và dễ hiểu. Nếu tài liệu ngữ cảnh không có thông tin hoặc không đủ để trả lời câu hỏi, hãy nói rõ rằng bạn không tìm thấy câu trả lời trong tài liệu và lịch sự khuyên người dùng nên tới phòng khám nha khoa để được bác sĩ khám trực tiếp. Không tự ý bịa đặt câu trả lời.

Câu hỏi của người dùng: {user_question}
"""
        
        # Bước 3: Gửi yêu cầu tới Ollama
        try:
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("message", {}).get("content", "").strip()
                return {
                    "answer": answer,
                    "sources": sources
                }
            else:
                return {
                    "answer": f"Lỗi từ Ollama API (Status {response.status_code}): {response.text}",
                    "sources": []
                }
        except requests.exceptions.RequestException as e:
            # Fallback nếu Ollama chưa được bật
            return {
                "answer": f"Không thể kết nối đến Ollama server tại {settings.OLLAMA_URL}. Vui lòng đảm bảo dịch vụ Ollama đang chạy.\nChi tiết lỗi: {e}",
                "sources": sources
            }

# Khởi tạo instance của Agent để sử dụng chung
dental_agent = DentalAgent()
