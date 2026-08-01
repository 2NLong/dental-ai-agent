import os
import sys
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# Thêm thư mục backend vào sys.path để giải quyết import
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/agent
backend_dir = os.path.dirname(current_dir) # backend
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from core.config import settings
from agent.tools.retriever_tool import search_dental_pdf

# Khởi tạo mô hình Ollama LLM
llm = OllamaLLM(
    base_url=settings.OLLAMA_URL,
    model=settings.OLLAMA_MODEL,
    temperature=0.0
)

# Cấu hình Prompt chuyên sâu cho Bác sĩ Nha Khoa
PROMPT_TEMPLATE = """
Bạn là trợ lý AI của phòng khám nha khoa.

Nhiệm vụ của bạn là trả lời câu hỏi CHỈ dựa trên thông tin trong tài liệu được cung cấp.

Nguyên tắc:

- Chỉ sử dụng thông tin trong CONTEXT.
- Không tự suy diễn hoặc bổ sung kiến thức bên ngoài.
- Nếu CONTEXT không đủ để trả lời, hãy nói:
"Kiến thức này nằm ngoài phạm vi tài liệu hiện tại của phòng khám."
- Trả lời đúng trọng tâm câu hỏi.
- Trình bày rõ ràng bằng Markdown nếu nội dung dài.
- Nếu câu hỏi ngắn thì trả lời ngắn.
- Nếu câu hỏi cần nhiều thông tin thì trình bày theo các mục phù hợp.
- Không cần ép buộc phải theo một mẫu cố định.

--------------------
CONTEXT:
{context}

--------------------
CÂU HỎI:
{query}

TRẢ LỜI:
"""

prompt_template = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "query"])
chain = prompt_template | llm

class DentalAgent:
    """Quy trình điều hướng hỏi đáp RAG."""
    
    def run(self, user_question: str, limit: int = 5) -> dict:
        # 1. Gọi retriever tool để tìm kiếm văn bản tương đồng
        context, sources = search_dental_pdf(user_question, limit=limit)
        
        if not sources:
            return {
                "answer": "Kiến thức này nằm ngoài phạm vi tài liệu hiện tại của phòng khám.",
                "sources": []
            }
            
        try:
            # 2. Suy luận tạo câu trả lời qua Ollama
            answer = chain.invoke({"context": context, "query": user_question})
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            return {
                "answer": f"Khong the ket noi den Ollama server tai {settings.OLLAMA_URL}. Chi tiet loi: {e}",
                "sources": sources
            }

# Khởi tạo instance dùng chung
dental_agent = DentalAgent()
