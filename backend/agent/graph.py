from agent.state import AgentState
from agent.prompts import DENTAL_SYSTEM_PROMPT
from agent.tools import search_dental_pdf
from core.llm_providers import get_llm_provider


class DentalAgent:
    """Quy trình điều hướng câu hỏi và suy nghĩ của AI Agent nha khoa (RAG)."""
    
    def __init__(self):
        self.system_prompt = DENTAL_SYSTEM_PROMPT
        # Khởi tạo LLM Provider linh hoạt thông qua Factory
        self.llm_provider = get_llm_provider()
        
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
        
        # Bước 3: Gửi yêu cầu tới nhà cung cấp LLM đã chọn
        try:
            answer = self.llm_provider.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=prompt
            )
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Lỗi xảy ra khi xử lý phản hồi từ LLM Provider: {e}",
                "sources": sources
            }


# Khởi tạo instance của Agent để sử dụng chung
dental_agent = DentalAgent()
