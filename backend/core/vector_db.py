from qdrant_client import QdrantClient
from core.config import settings

# Khởi tạo kết nối tới Qdrant Vector Database
# (Mặc định kết nối tới http://localhost:6333)
qdrant_client = QdrantClient(url=settings.QDRANT_URL)

def init_vector_db():
    """Hàm khởi tạo hoặc kiểm tra kết nối với Qdrant."""
    try:
        # Lấy danh sách collections để kiểm tra kết nối
        qdrant_client.get_collections()
        print("Kết nối Qdrant Vector DB thành công!")
    except Exception as e:
        print(f"Lỗi kết nối Qdrant Vector DB: {e}")
