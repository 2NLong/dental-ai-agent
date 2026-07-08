import json
import os
from qdrant_client import QdrantClient

QDRANT_PATH = "./data/processed/qdrant_local"
COLLECTION_NAME = "dental_knowledge"
OUTPUT_FILE = "tat_ca_chunks.json"  # Đổi đuôi thành .json

def export_all_chunks_to_json():
    print("⏳ Đang kết nối Qdrant...")
    client = QdrantClient(path=QDRANT_PATH)
    
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        total_chunks = collection_info.points_count
        print(f"📊 Tìm thấy tổng cộng: {total_chunks} chunk trong database.")
    except Exception as e:
        print(f"❌ Lỗi kết nối DB: {e}")
        return

    print(f"📥 Đang tiến hành quét sạch {total_chunks} chunk để xuất file JSON...")
    
    # Dùng vòng lặp scroll để lấy toàn bộ dữ liệu từ DB
    all_points = []
    next_page_offset = None
    
    while True:
        points, next_page_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,  # Lấy mỗi lượt 100 đoạn
            offset=next_page_offset,
            with_payload=True,
            with_vectors=False
        )
        all_points.extend(points)
        if next_page_offset is None:  # Hết dữ liệu thì dừng
            break

    # 1. Tạo bộ đếm/thống kê các file đã được lưu
    file_summary = {}
    chunks_list = []
    
    # 2. Duyệt qua từng point để cấu trúc lại dữ liệu
    for idx, point in enumerate(all_points):
        payload = point.payload or {}
        text_content = payload.get("page_content", "").strip()
        metadata = payload.get("metadata", {})
        
        # Lấy thông tin nguồn file
        src = metadata.get("source", "Không rõ")
        # Chuẩn hóa tên file (bỏ đường dẫn nếu có để nhìn cho gọn)
        file_name = os.path.basename(src) if src != "Không rõ" else src
        
        # Cập nhật thống kê số lượng chunk theo từng file
        file_summary[file_name] = file_summary.get(file_name, 0) + 1
        
        # Đóng gói dữ liệu của chunk này vào dict
        chunk_data = {
            "id": point.id,
            "chunk_index": idx + 1,
            "source_file": file_name,
            "page": metadata.get("page", "Không rõ"),
            "content": text_content,
            "full_metadata": metadata  # Giữ lại toàn bộ metadata gốc nếu cần
        }
        chunks_list.append(chunk_data)

    # 3. Gom tất cả vào một cấu trúc JSON tổng thể hoàn chỉnh
    final_json_data = {
        "report_title": f"BÁO CÁO TOÀN BỘ {len(all_points)} CHUNK DỮ LIỆU NHA KHOA",
        "total_chunks": len(all_points),
        "statistics": {
            "description": "Thống kê số lượng đoạn (chunks) trích xuất từ các file gốc",
            "files": file_summary
        },
        "chunks": chunks_list
    }

    # 4. Tiến hành ghi vào file JSON với định dạng đẹp (indent=4) và hỗ trợ tiếng Việt (ensure_ascii=False)
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_json_data, f, ensure_ascii=False, indent=4)
        print(f"✅ HOÀN THÀNH! Long mở file '{OUTPUT_FILE}' để xem cấu trúc JSON cực kỳ đẹp và chi tiết nhé!")
    except Exception as e:
        print(f"❌ Lỗi khi ghi file JSON: {e}")

if __name__ == "__main__":
    export_all_chunks_to_json()