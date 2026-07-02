from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """Trạng thái hiện tại của luồng suy nghĩ của AI Agent."""
    messages: List[Dict[str, Any]]  # Lịch sử hội thoại
    next_step: str                  # Bước tiếp theo cần thực hiện
    context: str                    # Ngữ cảnh lấy từ RAG hoặc search
