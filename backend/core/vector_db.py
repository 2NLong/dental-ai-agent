import os
import sys

# Cấu hình thư mục lưu trữ cache model HuggingFace cục bộ trong dự án
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/core
backend_dir = os.path.dirname(current_dir) # backend
project_root = os.path.dirname(backend_dir) # root/

os.environ["HF_HOME"] = os.path.join(project_root, ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(project_root, ".cache", "torch")

import torch
from qdrant_client import QdrantClient
from core.config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

# Tự động chọn thiết bị chạy mô hình nhúng (ưu tiên GPU CUDA nếu có)
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

def init_vector_db():
    """Tự động tạo collection nếu chưa có để tránh lỗi 404 khi truy vấn."""
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            print(f"[Vector DB] Collection '{COLLECTION_NAME}' chua ton tai. Tien hanh tao moi...")
            from qdrant_client.models import Distance, VectorParams
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            print(f"[Vector DB] Da tao thanh cong collection '{COLLECTION_NAME}'!")
    except Exception as e:
        clean_msg = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"[Vector DB] Canh bao khi kiem tra/khoi tao collection: {clean_msg}")

def load_vector_db():
    """Khởi tạo thực thể QdrantVectorStore kết nối với Server (Lazy Loading)."""
    init_vector_db()
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
