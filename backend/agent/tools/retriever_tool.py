import os
import sys

# Thêm thư mục gốc của backend vào sys.path để import dễ dàng khi chạy file độc lập
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from core.vector_db import qdrant_vector_store

def search_dental_pdf(query: str, limit: int = 3) -> tuple[str, list[str]]:
    """Tìm kiếm các đoạn văn liên quan nhất trong Qdrant Vector Database sử dụng LangChain."""
    try:
        # Tìm kiếm các vector tương đồng trên Qdrant qua LangChain QdrantVectorStore
        search_results = qdrant_vector_store.similarity_search(query, k=limit)
        
        # Trích xuất và định dạng kết quả trả về
        contexts = []
        sources = []
        for idx, doc in enumerate(search_results):
            text = doc.page_content
            metadata = doc.metadata
            
            # Đọc metadata nâng cao mới, hỗ trợ tương thích ngược với dữ liệu cũ
            source_path = metadata.get("source", "Tài liệu nha khoa")
            file_name = metadata.get("file_name", os.path.basename(source_path))
            title = metadata.get("title", file_name)
            page = metadata.get("page", 0) + 1  # Trang bắt đầu từ 0
            
            contexts.append(f"[Đoạn trích {idx+1} - Nguồn: {title} (Trang {page})]:\n{text}")
            
            # Trả về file_name trong danh sách sources để hiển thị tên tài liệu gốc
            if file_name not in sources:
                sources.append(file_name)
            
        if not contexts:
            return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu tài liệu nha khoa.", []
            
        return "\n\n".join(contexts), sources
    except Exception as e:
        print(f"Lỗi khi truy vấn Qdrant Vector DB bằng LangChain: {e}")
        return f"Không thể lấy dữ liệu từ tài liệu nha khoa do lỗi: {e}", []

