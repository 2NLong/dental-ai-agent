import os
import sys

# Thêm thư mục backend vào sys.path để giải quyết import
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/agent/tools
backend_dir = os.path.dirname(os.path.dirname(current_dir)) # backend
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from core.vector_db import load_vector_db

def search_dental_pdf(query: str, limit: int = 5) -> tuple[str, list[str]]:
    """Tìm kiếm các đoạn văn liên quan nhất trong Qdrant Vector Database sử dụng LangChain."""
    try:
        vector_store = load_vector_db()
        search_results = vector_store.similarity_search(query, k=limit)
        
        contexts = []
        sources = []
        for idx, doc in enumerate(search_results):
            text = doc.page_content
            source = doc.metadata.get("source", "Tai lieu nha khoa")
            source_name = os.path.basename(source)
            page = doc.metadata.get("page", 0) + 1  # 0-indexed to 1-indexed
            
            contexts.append(f"[Doan trich {idx+1} - Nguon: {source_name} (Trang {page})]:\n{text}")
            source_info = f"{source_name} (Trang {page})"
            if source_info not in sources:
                sources.append(source_info)
                
        if not contexts:
            return "Khong tim thay thong tin phu hop trong co so du lieu tai lieu nha khoa.", []
            
        return "\n\n".join(contexts), sources
    except Exception as e:
        print(f"Loi khi truy van Qdrant Vector DB: {e}")
        return f"Khong the lay du lieu tu tai lieu nha khoa do loi: {e}", []
