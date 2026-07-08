import os
from qdrant_client import QdrantClient

QDRANT_PATH = "./data/processed/qdrant_local"
COLLECTION_NAME = "dental_knowledge"
OUTPUT_FILE = "tat_ca_chunks.txt"

def export_to_text():
    print("⏳ Đang kết nối Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        total_chunks = collection_info.points_count
    except Exception as e:
        print(f"❌ Lỗi kết nối DB: {e}")
        return

    print(f"📥 Đang lấy dữ liệu {total_chunks} chunk để xuất file...")
    
    # Lấy toàn bộ chunk (đặt limit bằng tổng số chunk luôn)
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=total_chunks,
        with_payload=True,
        with_vectors=False
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"=== TỔNG CỘNG CÓ {total_chunks} CHUNK DỮ LIỆU NHA KHOA ===\n\n")
        
        for idx, point in enumerate(points):
            payload = point.payload
            text_content = payload.get("page_content", "")
            metadata = payload.get("metadata", {})
            
            f.write(f"============================================================\n")
            f.write(f"📍 CHUNK SỐ {idx + 1} | File: {metadata.get('source', 'Không rõ')} | Trang: {metadata.get('page', 'Không rõ')}\n")
            f.write(f"============================================================\n")
            f.write(f"{text_content.strip()}\n\n")
            
    print(f"✅ ĐÃ XUẤT THÀNH CÔNG! Long mở file '{OUTPUT_FILE}' ở thư mục gốc ra xem nhé.")

if __name__ == "__main__":
    export_to_text()