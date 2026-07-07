# Import pyarrow đầu tiên để tránh lỗi tranh chấp DLL (Segmentation fault) với PyTorch CUDA trên Windows
import pyarrow
import os
import uuid
import sys
import json

# Thêm thư mục gốc của backend vào sys.path để import dễ dàng khi chạy file độc lập
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from core.vector_db import qdrant_client, qdrant_vector_store, COLLECTION_NAME, embeddings
from qdrant_client.models import Distance, VectorParams
from ingestion.loaders import get_pdf_loader
from ingestion.splitters import get_text_splitter

def init_qdrant_collection():
    """Khởi tạo collection trên Qdrant nếu chưa tồn tại với cấu hình vector 1024 chiều (BGE-M3)."""
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            print(f"Collection '{COLLECTION_NAME}' chưa tồn tại. Tiến hành tạo mới...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )
            print(f"Đã tạo thành công collection '{COLLECTION_NAME}'!")
        else:
            print(f"Collection '{COLLECTION_NAME}' đã tồn tại.")
    except Exception as e:
        print(f"Lỗi khi kết nối hoặc khởi tạo collection Qdrant: {e}")
        raise e

def run_ingestion_pipeline(
    pdf_folder_path: str,
    loader_method: str = "pypdf",
    splitter_method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Quy trình toàn diện từ việc tìm file PDF gốc tới khi nạp thành công vào Vector DB bằng LangChain."""
    if not os.path.exists(pdf_folder_path):
        print(f"Thư mục tài liệu {pdf_folder_path} không tồn tại.")
        return
        
    init_qdrant_collection()
    
    # Khởi tạo bộ chia văn bản từ mô-đun splitters
    text_splitter = get_text_splitter(
        method=splitter_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, file_name)
            print(f"\n=== ĐANG XỬ LÝ (MÔ-ĐUN HÓA): {file_name} ===")
            print(f"Loader: {loader_method} | Splitter: {splitter_method} (size={chunk_size}, overlap={chunk_overlap})")
            
            try:
                # 1. Trích xuất text sử dụng loader được chọn từ mô-đun loaders
                loader = get_pdf_loader(file_path, method=loader_method)
                documents = loader.load()
                if not documents:
                    print(f"Bỏ qua file {file_name} vì không trích xuất được nội dung text.")
                    continue
                    
                # 2. Cắt nhỏ tài liệu
                split_docs = text_splitter.split_documents(documents)
                # Lọc bỏ các chunk trống hoặc quá ngắn (dưới 10 ký tự)
                split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]
                print(f"Đã phân cắt thành {len(split_docs)} chunks hợp lệ.")
                if not split_docs:
                    continue
                
                # 3. Tạo ID UUID v5 theo nội dung để tránh trùng lặp dữ liệu (Idempotent upsert)
                ids = []
                for doc in split_docs:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
                    ids.append(point_id)
                    
                # 4. Lưu vào Qdrant Vector Store
                print(f"Đang tạo embeddings và lưu {len(split_docs)} chunks vào Qdrant...")
                batch_size = 100
                for j in range(0, len(split_docs), batch_size):
                    qdrant_vector_store.add_documents(
                        documents=split_docs[j:j + batch_size],
                        ids=ids[j:j + batch_size]
                    )
                print(f"Nạp thành công tài liệu: {file_name}")
            except Exception as e:
                print(f"Lỗi khi xử lý file {file_name}: {e}")

def export_ingestion_to_json(
    pdf_folder_path: str,
    output_json_path: str,
    loader_method: str = "pypdf",
    splitter_method: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """Quy trình xử lý file PDF, tạo embeddings và xuất ra file JSON (chạy trên Colab)."""
    if not os.path.exists(pdf_folder_path):
        print(f"Thư mục tài liệu {pdf_folder_path} không tồn tại.")
        return
        
    text_splitter = get_text_splitter(
        method=splitter_method,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_points = []
    
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, file_name)
            print(f"\n=== ĐANG XỬ LÝ ĐỂ XUẤT JSON: {file_name} ===")
            
            try:
                loader = get_pdf_loader(file_path, method=loader_method)
                documents = loader.load()
                if not documents:
                    continue
                    
                split_docs = text_splitter.split_documents(documents)
                split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]
                print(f"Đã phân cắt thành {len(split_docs)} chunks hợp lệ.")
                if not split_docs:
                    continue
                
                print("Đang tạo embeddings...")
                texts = [doc.page_content for doc in split_docs]
                vectors = embeddings.embed_documents(texts)
                
                for doc, vector in zip(split_docs, vectors):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
                    all_points.append({
                        "id": point_id,
                        "vector": vector,
                        "payload": {
                            "page_content": doc.page_content,
                            "metadata": doc.metadata
                        }
                    })
                print(f"Đã xử lý xong file: {file_name}")
            except Exception as e:
                print(f"Lỗi khi xử lý file {file_name}: {e}")
                
    print(f"\nĐang ghi {len(all_points)} points ra file {output_json_path}...")
    # Tạo thư mục cha nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_points, f, ensure_ascii=False, indent=4)
    print("Xuất file JSON thành công!")

def import_from_json(json_file_path: str):
    """Import dữ liệu đã được chunking và embedding từ file JSON bên ngoài vào Qdrant local."""
    if not os.path.exists(json_file_path):
        print(f"File JSON {json_file_path} không tồn tại.")
        return
        
    init_qdrant_collection()
    
    print(f"Đang đọc dữ liệu từ {json_file_path}...")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Đã đọc {len(data)} points. Tiến hành nạp vào Qdrant...")
    
    from qdrant_client.models import PointStruct
    
    points = []
    for item in data:
        points.append(
            PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item["payload"]
            )
        )
        
    batch_size = 100
    for j in range(0, len(points), batch_size):
        batch = points[j:j + batch_size]
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )
        print(f"Đã nạp thành công {j + len(batch)}/{len(points)} points...")
        
    print("Hoàn thành import dữ liệu vào Qdrant!")

if __name__ == "__main__":
    raw_pdfs_path = os.path.join(backend_dir, "data", "raw_pdfs")
    
    # Lựa chọn 1 trong 3 chế độ:
    # 1. Chạy ingestion pipeline trực tiếp thông thường:
    run_ingestion_pipeline(
        pdf_folder_path=raw_pdfs_path,
        loader_method="pypdf",
        splitter_method="recursive",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 2. Xuất dữ liệu ra file JSON để mang đi nơi khác (hoặc chạy trên Colab):
    # export_json_path = os.path.join(backend_dir, "data", "qdrant_export.json")
    # export_ingestion_to_json(
    #     pdf_folder_path=raw_pdfs_path,
    #     output_json_path=export_json_path,
    #     loader_method="pypdf",
    #     splitter_method="recursive",
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    
    # 3. Nạp dữ liệu từ file JSON bên ngoài vào Qdrant local:
    # import_json_path = os.path.join(backend_dir, "data", "qdrant_export.json")
    # import_from_json(import_json_path)

