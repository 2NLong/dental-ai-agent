from agent.state import AgentState
from agent.prompts import DENTAL_SYSTEM_PROMPT
from agent.tools import search_dental_pdf, web_search

class DentalAgent:
    """Mô phỏng luồng điều hướng câu hỏi và suy nghĩ của AI Agent."""
    
    def __init__(self):
        self.system_prompt = DENTAL_SYSTEM_PROMPT
        
    def run(self, user_question: str) -> str:
        # Bước 1: Gọi retriever tool để tìm tài liệu PDF
        context = search_dental_pdf(user_question)
        
        # Bước 2: Kiểm tra nếu tài liệu không đủ thông tin, có thể gọi web search
        # (Đây là logic quyết định luồng đơn giản hóa)
        if "không tìm thấy" in context.lower():
            context = web_search(user_question)
            
        # Bước 3: Đưa context vào LLM và nhận câu trả lời cuối cùng
        # (Ở đây sẽ gọi LLM thực tế trong tương lai)
        response = f"[AI Dental Agent Answer]: Dựa trên tài liệu nha khoa: '{context}'"
        return response

# Khởi tạo instance của Agent để sử dụng chung
dental_agent = DentalAgent()
