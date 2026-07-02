from typing import List

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Chuyển đổi danh sách văn bản sang danh sách các vector số thực (embeddings)."""
    # Sau này kết nối OpenAI Embeddings, Gemini Embeddings, hoặc SentenceTransformers
    print(f"Đang tạo vector cho {len(chunks)} chunks...")
    # Trả về vector giả lập có kích thước 1536 chiều
    return [[0.0] * 1536 for _ in chunks]
