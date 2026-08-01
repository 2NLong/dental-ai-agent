import os
import sys
from pymongo import MongoClient

# Thêm thư mục backend vào sys.path để giải quyết import
current_dir = os.path.dirname(os.path.abspath(__file__)) # backend/core
backend_dir = os.path.dirname(current_dir) # backend
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from core.config import settings

# Khởi tạo kết nối tới MongoDB
client = MongoClient(settings.MONGODB_URI)

# Lấy Database mặc định (tự động trích xuất từ URI, mặc định là dental_db)
db = client.get_default_database(default="dental_db")
