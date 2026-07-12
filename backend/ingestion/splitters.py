from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document

class MarkdownHeaderSplitterWrapper:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        # Xác định các cấp độ tiêu đề Markdown cần cắt đoạn
        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Giữ lại tiêu đề trong nội dung text để LLM đọc dễ hiểu
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )

    def split_documents(self, documents):
        final_chunks = []
        for doc in documents:
            # 1. Phân tách theo cấu trúc tiêu đề Markdown trước
            header_splits = self.markdown_splitter.split_text(doc.page_content)
            
            # 2. Với mỗi đoạn con dưới tiêu đề, nếu vẫn quá dài -> chia tiếp bằng Recursive Character
            for split in header_splits:
                # Gộp metadata gốc của tài liệu với metadata tiêu đề mới nhận dạng được
                split_metadata = doc.metadata.copy()
                split_metadata.update(split.metadata)
                
                sub_splits = self.text_splitter.split_text(split.page_content)
                for sub_text in sub_splits:
                    final_chunks.append(
                        Document(page_content=sub_text, metadata=split_metadata)
                    )
        return final_chunks

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
    - "markdown": MarkdownHeaderSplitterWrapper (cắt theo cấu trúc tiêu đề Markdown rồi tinh chỉnh theo Recursive)
    """
    method = method.lower().strip()
    if method == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
    elif method == "semantic":
        if embeddings is None:
            raise ValueError("Phương pháp 'semantic' yêu cầu truyền đối tượng embeddings để tính toán độ tương đồng ngữ nghĩa.")
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"  # Phân tách dựa trên phân vị khoảng cách ngữ nghĩa
        )
    elif method == "markdown":
        return MarkdownHeaderSplitterWrapper(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError(f"Phương pháp chia văn bản (splitter method) '{method}' chưa được hỗ trợ.")

