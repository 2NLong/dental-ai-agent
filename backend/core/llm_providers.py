import os
import requests
from abc import ABC, abstractmethod
from core.config import settings


class BaseLLMProvider(ABC):
    """
    Interface chuẩn (Strategy Pattern) cho các nhà cung cấp mô hình ngôn ngữ lớn (LLM).
    """

    @abstractmethod
    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gửi yêu cầu tới LLM và nhận câu trả lời dạng chuỗi (string).

        :param system_prompt: Chỉ thị hệ thống định nghĩa vai trò của AI.
        :param user_prompt: Câu hỏi kèm ngữ cảnh của người dùng.
        :return: Câu trả lời từ LLM (chuỗi văn bản sạch).
        """
        pass


class OllamaProvider(BaseLLMProvider):
    """
    Nhà cung cấp LLM chạy cục bộ (offline) qua Ollama (ví dụ: Qwen3).
    """
    def __init__(self):
        self.url = f"{settings.OLLAMA_URL}/api/chat"
        self.model = settings.OLLAMA_MODEL

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        print(f"[OllamaProvider] Đang gửi yêu cầu tới mô hình '{self.model}' qua Ollama...")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("message", {}).get("content", "").strip()


class GeminiProvider(BaseLLMProvider):
    """
    Nhà cung cấp LLM trực tuyến thông qua Google AI Studio API (Gemini).
    """
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL
        # Sử dụng API của Google AI Studio (v1beta) để tạo nội dung
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                "Không tìm thấy GEMINI_API_KEY trong file .env hoặc cấu hình hệ thống. "
                "Vui lòng thiết lập khóa API để sử dụng nhà cung cấp Gemini."
            )

        print(f"[GeminiProvider] Đang gửi yêu cầu tới mô hình '{self.model}' qua Google AI Studio...")
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt}
                    ]
                }
            ],
            "systemInstruction": {
                "parts": [
                    {"text": system_prompt}
                ]
            },
            "generationConfig": {
                "temperature": 0.3
            }
        }

        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        try:
            # Trích xuất nội dung văn bản từ phản hồi chuẩn của Gemini API
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            return answer.strip()
        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Không thể đọc kết quả trả về từ Gemini API. Phản hồi thực tế: {result}"
            ) from e


def get_llm_provider() -> BaseLLMProvider:
    """
    Hàm Factory để khởi tạo và trả về đối tượng provider tương ứng dựa trên cấu hình LLM_PROVIDER.
    """
    provider_name = settings.LLM_PROVIDER.lower().strip()
    
    if provider_name == "ollama":
        return OllamaProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    else:
        raise ValueError(
            f"Nhà cung cấp LLM '{provider_name}' không được hỗ trợ. "
            "Các giá trị hợp lệ là: 'ollama', 'gemini'."
        )
