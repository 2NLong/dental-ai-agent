import os

# Cấu hình thư mục lưu trữ cache model HuggingFace cục bộ trong dự án (ổ D)
# Bắt buộc phải đặt trước khi import bất kỳ thư viện nào liên quan đến HuggingFace/Transformers
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)
os.environ["HF_HOME"] = os.path.join(project_root, ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(project_root, ".cache", "torch")

import torch
from qdrant_client import QdrantClient
from core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Tự động chọn thiết bị chạy mô hình nhúng (ưu tiên GPU CUDA nếu có RTX 4050)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Vector DB] Khoi tao model BGE-M3 tren thiet bi: {device.upper()}")

# Khởi tạo mô hình embeddings dùng chung bằng BGE-M3
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True}
)

COLLECTION_NAME = "dental_documents"

# Khởi tạo kết nối tới Qdrant Vector Database
qdrant_client = QdrantClient(url=settings.QDRANT_URL)

# Tự động tạo collection nếu chưa có để tránh lỗi 404 khi khởi tạo QdrantVectorStore của LangChain
try:
    collections = qdrant_client.get_collections().collections
    exists = any(c.name == COLLECTION_NAME for c in collections)
    if not exists:
        print(f"[Vector DB] Collection '{COLLECTION_NAME}' chưa tồn tại. Tiến hành tạo mới...")
        from qdrant_client.models import Distance, VectorParams
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        print(f"[Vector DB] Đã tạo thành công collection '{COLLECTION_NAME}'!")
except Exception as e:
    print(f"[Vector DB] Cảnh báo khi kiểm tra/khởi tạo collection: {e}")

# Khởi tạo Qdrant Vector Store của LangChain bọc quanh client
qdrant_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

def init_vector_db():
    """Hàm khởi tạo hoặc kiểm tra kết nối với Qdrant."""
    try:
        # Lấy danh sách collections để kiểm tra kết nối
        qdrant_client.get_collections()
        print("Kết nối Qdrant Vector DB thành công!")
    except Exception as e:
        print(f"Lỗi kết nối Qdrant Vector DB: {e}")
