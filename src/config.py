import os

# Đường dẫn gốc của dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cấu hình các thư mục dữ liệu
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
CHROMA_DB_DIR = os.path.join(PROCESSED_DATA_DIR, "chroma_db")

# Khởi tạo thư mục nếu chưa tồn tại
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Cấu hình mô hình Embedding Local (Tiếng Việt tốt nhất của BKAI)
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"