from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter

def get_text_splitter(method: str = "recursive", chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Trả về đối tượng chia nhỏ văn bản (text splitter) tương ứng với phương pháp được chọn.
    
    Hỗ trợ các phương pháp:
    - "recursive": RecursiveCharacterTextSplitter của LangChain (cắt thông minh dựa trên ranh giới tự nhiên)
    - "character": CharacterTextSplitter đơn giản (cắt dựa trên dấu ngắt cụ thể)
    """
    method = method.lower().strip()
    if method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", "\n", " ", ""]
        )
    elif method == "character":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator="\n"
        )
    else:
        raise ValueError(f"Phương pháp chia văn bản (splitter method) '{method}' chưa được hỗ trợ.")
