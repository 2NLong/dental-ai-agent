from typing import List

def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Cắt văn bản thô ra thành các đoạn nhỏ (chunks) để đưa vào Vector DB."""
    # Sau này bạn có thể sử dụng RecursiveCharacterTextSplitter của LangChain
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
    return chunks
