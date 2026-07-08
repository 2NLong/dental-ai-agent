# Dental Q&A AI Agent

Hệ thống hỏi đáp tài liệu nha khoa thông minh dựa trên phương pháp **RAG (Retrieval-Augmented Generation)** sử dụng cơ sở dữ liệu vector **Qdrant**, mô hình nhúng **BGE-M3 (chạy offline)** và mô hình ngôn ngữ **Qwen3 (chạy offline qua Ollama)**.

---

## 🛠️ Yêu cầu hệ thống (Prerequisites)

Trước khi bắt đầu, hãy đảm bảo máy tính của bạn đã cài đặt các công cụ sau:
* **Python**: Phiên bản `3.13` (hoặc từ `3.10` trở lên)
* **Docker Desktop**: Để chạy Qdrant Vector DB
* **Ollama**: Để chạy mô hình ngôn ngữ lớn (LLM) offline

---

## 🚀 Hướng dẫn cài đặt nhanh (Quick Start)

### Bước 1: Khởi tạo cấu hình môi trường
Sao chép tệp cấu hình mẫu và điền thông tin kết nối phù hợp:
```bash
cp .env.example .env
```
*Mở tệp `.env` vừa tạo và cập nhật các thông số cần thiết (ví dụ: `MONGODB_URI` để lưu lịch sử hội thoại).*

### Bước 2: Thiết lập môi trường ảo Python và cài đặt thư viện
Tại thư mục gốc dự án (`dental-qa-agent`), thực hiện các lệnh sau:

* **Tạo môi trường ảo:**
  ```bash
  python -m venv .venv
  ```
* **Kích hoạt môi trường ảo:**
  * *Trên Windows (PowerShell):* `.\.venv\Scripts\Activate.ps1`
  * *Trên Windows (Git Bash):* `source .venv/Scripts/activate`
  * *Trên Linux/MacOS:* `source .venv/bin/activate`
* **Cài đặt các gói phụ thuộc:**
  ```bash
  pip install -r requirements.txt
  ```

---

## 💾 Chuẩn bị hạ tầng phụ trợ (Qdrant & Ollama)

### 1. Khởi động Qdrant Vector DB (Docker)
Chạy container Qdrant trên local bằng Docker Compose:
```bash
docker compose up -d
```
Hoặc khởi động lại nếu container đã được tạo từ trước:
```bash
docker start dental-qdrant
```

### 2. Đăng ký và chạy LLM Qwen3 cục bộ qua Ollama
1. Tải mô hình `Qwen3-4B-Q4_K_M.gguf` đặt vào thư mục: `backend/models/` (nếu chưa có).
2. Đăng ký mô hình vào Ollama thông qua file cấu hình `Modelfile` của dự án:
   * *Trên Windows (PowerShell):*
     ```powershell
     & "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" create qwen3-dental -f Modelfile
     ```
   * *Trên Git Bash:*
     ```bash
     ~/AppData/Local/Programs/Ollama/ollama create qwen3-dental -f Modelfile
     ```
3. Chạy thử nghiệm mô hình:
   ```bash
   ollama run qwen3-dental
   ```

---

## 📂 Hướng dẫn chạy Dự án

### Bước 1: Nạp tài liệu vào Cơ sở dữ liệu Vector (Data Ingestion)
Hãy bỏ các file tài liệu định dạng PDF của bạn vào thư mục `backend/data/raw_pdfs/`. Sau đó chạy lệnh nạp dữ liệu:

* *Trên Windows (để tránh lỗi tiếng Việt khi in ra terminal):*
  ```bash
  python -X utf8 backend/ingestion/pipeline.py
  ```
* *Trên Linux/MacOS:*
  ```bash
  python backend/ingestion/pipeline.py
  ```

### Bước 2: Khởi chạy FastAPI Server (Backend)
Để tránh lỗi nạp thư viện `ModuleNotFoundError: No module named 'core'`, bạn **phải chuyển vào thư mục `backend`** trước khi chạy server:

```bash
# 1. Chuyển vào thư mục backend
cd backend

# 2. Khởi chạy server uvicorn
uvicorn main:app --host 127.0.0.1 --port 8000
```

* **Trang chủ API:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
* **Tài liệu Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Kiểm tra hoạt động (Test API)

Bạn có thể gửi yêu cầu hỏi đáp trực tiếp tới API endpoint bằng lệnh `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d '{"message": "Dự phòng răng miệng là gì?", "session_id": "test_session"}'
```

---

## 💡 Các lưu ý đặc biệt trên hệ điều hành Windows

1. **Kích hoạt GPU (CUDA) để tăng tốc nhúng (Embedding):**
   Mô hình nhúng BGE-M3 chạy rất nhanh nếu có GPU. Nếu máy bạn sử dụng card đồ họa **NVIDIA**, vui lòng xem chi tiết hướng dẫn gỡ phiên bản Torch CPU và cài đặt phiên bản Torch CUDA 12.4 trong tệp [venv_guide.md](venv_guide.md#5-huong-dan-kich-hoat-gpu-cuda-de-tang-toc-nhung-embedding).
2. **Lỗi Crash Im Lặng (Segmentation Fault):**
   Luôn luôn giữ lệnh `import pyarrow` ở dòng đầu tiên của các file chạy chính (như `pipeline.py`, `main.py`) để tránh lỗi crash do xung đột thư viện CUDA trên Windows.
3. **Cấu hình lưu trữ Ollama Models:**
   Nếu ổ C đầy, bạn có thể chuyển thư mục chứa mô hình của Ollama sang ổ khác bằng cách thiết lập biến môi trường `OLLAMA_MODELS`. Xem chi tiết trong [venv_guide.md](venv_guide.md#6-huong-dan-chay-llm-cuc-bo-qua-ollama-qwen3-4b).
