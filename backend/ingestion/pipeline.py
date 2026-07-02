import os
from ingestion.parser import parse_pdf
from ingestion.splitter import split_text
from ingestion.embedder import embed_chunks

def run_ingestion_pipeline(pdf_folder_path: str):
    """Quy trình toàn diện từ việc tìm file PDF gốc tới khi nạp thành công vào Vector DB."""
    if not os.path.exists(pdf_folder_path):
        print(f"Thư mục tài liệu {pdf_folder_path} không tồn tại.")
        return
        
    for file_name in os.listdir(pdf_folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, file_name)
            
            # 1. Trích xuất text
            text = parse_pdf(file_path)
            
            # 2. Cắt nhỏ text (Chunking)
            chunks = split_text(text)
            
            # 3. Tạo vector embeddings
            vectors = embed_chunks(chunks)
            
            # 4. Lưu vào Vector Database (sử dụng core/vector_db.py)
            print(f"Đã nạp thành công tài liệu: {file_name}")

if __name__ == "__main__":
    # Điểm chạy thử độc lập cho pipeline nạp dữ liệu
    run_ingestion_pipeline("backend/data/raw_pdfs")
