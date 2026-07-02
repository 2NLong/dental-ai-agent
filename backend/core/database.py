from pymongo import MongoClient
from core.config import settings

# Khởi tạo kết nối tới MongoDB
# (Tham số kết nối sẽ được tự động cấu hình từ biến môi trường MONGODB_URI)
client = MongoClient(settings.MONGODB_URI)

# Lấy Database mặc định (tự động trích xuất từ URI, ví dụ: .../dental_db thì tên DB sẽ là dental_db)
db = client.get_default_database(default="dental_db")
