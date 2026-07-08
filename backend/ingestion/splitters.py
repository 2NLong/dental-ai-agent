from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

def get_text_splitter(
    method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embeddings = None
):
    """
    Trả về đối tượng chia nhỏ văn bản (text splitter) tương ứng với phương pháp được chọn.
    
    Hỗ trợ các phương pháp:
    - "recursive": RecursiveCharacterTextSplitter của LangChain (cắt thông minh dựa trên ranh giới tự nhiên)
    - "semantic": SemanticChunker của LangChain (cắt dựa trên sự tương đồng ngữ nghĩa giữa các câu)
    """
    method = method.lower().strip()
    if method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", "\n", " ", ""]
        )
    elif method == "semantic":
        if embeddings is None:
            raise ValueError("Phương pháp 'semantic' yêu cầu truyền đối tượng embeddings để tính toán độ tương đồng ngữ nghĩa.")
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"  # Phân tách dựa trên phân vị khoảng cách ngữ nghĩa
        )
    else:
        raise ValueError(f"Phương pháp chia văn bản (splitter method) '{method}' chưa được hỗ trợ.")

