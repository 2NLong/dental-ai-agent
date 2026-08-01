from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(
    method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Trả về đối tượng chia nhỏ văn bản (text splitter) tương ứng.
    Sử dụng RecursiveCharacterTextSplitter chia nhỏ văn bản theo ranh giới câu tự nhiên.
    """
    method = method.lower().strip()
    if method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", ". ", "\n", " ", ""]
        )
    else:
        raise ValueError(f"Splitter method '{method}' khong duoc ho tro.")
