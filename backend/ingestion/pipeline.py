import os
import sys
import uuid
import json
import argparse
import glob

# Thêm thư mục backend vào sys.path để giải quyết import
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/ingestion
backend_dir = os.path.dirname(current_dir) # backend
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from qdrant_client.models import PointStruct
from core.config import settings
from core.vector_db import qdrant_client, load_vector_db, COLLECTION_NAME, embeddings, init_vector_db
from ingestion.loaders import get_pdf_loader
from ingestion.splitters import get_text_splitter

def run_ingestion():
    """Đọc các tài liệu PDF trong data/raw, chia nhỏ và đẩy trực tiếp lên Qdrant Server."""
    print("Bat dau nap tai lieu vao Qdrant Server...")
    
    pdf_files = glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"Khong tim thay file PDF nao trong: {settings.RAW_DATA_DIR}")
        return
        
    init_vector_db()
    text_splitter = get_text_splitter(method="recursive")
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"Dang xu ly: {file_name}...")
        try:
            loader = get_pdf_loader(pdf_path, method="unstructured")
            documents = loader.load()
            if not documents:
                print(f"Khong trich xuat duoc noi dung tu {file_name}. Bo qua.")
                continue
                
            split_docs = text_splitter.split_documents(documents)
            # Lọc bỏ các chunk trống hoặc quá ngắn
            split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]
            print(f"Cat thanh {len(split_docs)} doan van ban.")
            
            if not split_docs:
                continue
                
            # Sinh ID UUID v5 theo nội dung để tránh trùng lặp dữ liệu
            ids = []
            for doc in split_docs:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
                ids.append(point_id)
                
            # Nạp vào Qdrant Vector Store
            print(f"Dang tinh embeddings va nap {len(split_docs)} chunks vao Qdrant...")
            vector_store = load_vector_db()
            batch_size = 100
            for j in range(0, len(split_docs), batch_size):
                vector_store.add_documents(
                    documents=split_docs[j:j + batch_size],
                    ids=ids[j:j + batch_size]
                )
            print(f"Nap thanh cong: {file_name}")
        except Exception as e:
            print(f"Loi khi xu ly file {file_name}: {e}")

def export_to_json(output_path: str):
    """Đọc tài liệu, tính embeddings và xuất tất cả ra file JSON cục bộ."""
    print("Bat dau trich xuat va tinh embeddings xuat ra JSON...")
    
    pdf_files = glob.glob(os.path.join(settings.RAW_DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"Khong tim thay file PDF nao trong: {settings.RAW_DATA_DIR}")
        return
        
    text_splitter = get_text_splitter(method="recursive")
    all_points = []
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"Dang tinh toan vector cho: {file_name}...")
        try:
            loader = get_pdf_loader(pdf_path, method="unstructured")
            documents = loader.load()
            if not documents:
                continue
                
            split_docs = text_splitter.split_documents(documents)
            split_docs = [doc for doc in split_docs if len(doc.page_content.strip()) > 10]
            print(f"Cat thanh {len(split_docs)} doan.")
            
            if not split_docs:
                continue
                
            print("Dang tinh embeddings...")
            texts = [doc.page_content for doc in split_docs]
            # Tính toán vector embedding cho toàn bộ chunks
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
            print(f"Da xu ly xong: {file_name}")
        except Exception as e:
            print(f"Loi khi xu ly file {file_name}: {e}")
            
    if not all_points:
        print("Khong co du lieu de xuat!")
        return
        
    print(f"Dang ghi {len(all_points)} points ra JSON tai {output_path}...")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_points, f, ensure_ascii=False, indent=4)
    print("Xuat file JSON thanh cong!")

def import_from_json(input_path: str):
    """Nhập dữ liệu embeddings và nội dung từ file JSON trực tiếp vào Qdrant Server."""
    if not os.path.exists(input_path):
        print(f"File JSON khong ton tai tai: {input_path}")
        return
        
    init_vector_db()
    
    print(f"Dang doc du lieu tu file {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"Da doc {len(data)} points. Tien hanh nap vao Qdrant Server...")
    
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
        print(f"  -> Da nap thanh cong {j + len(batch)}/{len(points)} points...")
        
    print("Hoan thanh nhap du lieu vao Qdrant Server!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion Pipeline cho Dental AI Agent")
    parser.add_argument("--mode", type=str, default="run", choices=["run", "export", "import"],
                        help="Che do chay: run (nap truc tiep), export (xuat ra JSON), import (nhap tu JSON)")
    parser.add_argument("--output", type=str, default="./data/processed/embeddings_export.json",
                        help="Duong dan luu file JSON khi chon che do export")
    parser.add_argument("--input", type=str, default="./data/processed/embeddings_export.json",
                        help="Duong dan file JSON nguon khi chon che do import")
                        
    args = parser.parse_args()
    
    if args.mode == "run":
        run_ingestion()
    elif args.mode == "export":
        export_to_json(args.output)
    elif args.mode == "import":
        import_from_json(args.input)
