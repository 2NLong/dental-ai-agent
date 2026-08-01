# 🦷 Trợ Lý AI Nha Khoa (Dental Q&A AI Agent)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B.svg)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![Ollama](https://img.shields.io/badge/Ollama-Gemma%203-black.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/Offline-100%25-green.svg)](#)

Hệ thống hỏi đáp tài liệu y khoa & nha khoa thông minh dựa trên kiến trúc **RAG (Retrieval-Augmented Generation)** hoạt động **hoàn toàn ngoại tuyến (100% Offline)**. 

Ứng dụng giúp bác sĩ, y sĩ, phụ tá và học viên nha khoa nhanh chóng tra cứu thông tin chuyên môn chính xác từ kho tài liệu PDF nội bộ, ngăn ngừa hiện tượng bịa thông tin (hallucination) và bảo mật tuyệt đối dữ liệu y tế.

---

## ✨ Tính Năng Nổi Bật

* **100% Offline & Bảo mật:** Chạy toàn bộ mô hình Embeddings và LLM cục bộ, không gửi dữ liệu ra môi trường đám mây bên ngoài.
* **Đa giao diện truy cập (Multi-interface):**
  * **Streamlit Web UI:** Giao diện người dùng tương tự ChatGPT với lịch sử trò chuyện, đo thời gian suy luận và hiển thị nguồn tài liệu tham chiếu.
  * **FastAPI Backend REST API:** Cung cấp API chuẩn tích hợp với hệ thống bên ngoài hoặc ứng dụng di động.
  * **CLI Tool:** Giao diện dòng lệnh phục vụ thử nghiệm nhanh.
* **Số hóa tài liệu thông minh (OCR Ingestion):** Tự động đọc file PDF y khoa (kể cả sách scan/ảnh) bằng `UnstructuredPDFLoader` + Tesseract OCR (hỗ trợ Tiếng Việt & Tiếng Anh).
* **Tìm kiếm ngữ cảnh tối ưu (Vector Search):** Sử dụng mô hình nhúng **BAAI/bge-m3** đa ngôn ngữ kết hợp cơ sở dữ liệu vector **Qdrant (Docker)**.
* **Suy luận chính xác (Gemma 3 local):** Tích hợp LLM **Gemma 3 (4B QAT)** qua **Ollama** với Prompt kiểm soát nghiêm ngặt, bắt buộc câu trả lời bám sát tài liệu tham chiếu.
* **Bộ công cụ Ingest / Export / Import:** Hỗ trợ trích xuất vector embeddings ra file JSON để chuyển giao giữa các máy mà không cần tính toán lại embedding.

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

| Thành phần | Công nghệ / Thư viện | Vai trò |
| :--- | :--- | :--- |
| **Giao diện Web** | Streamlit | Chatbot UI hỗ trợ sidebar lịch sử hội thoại & metrics |
| **Backend API** | FastAPI + Uvicorn | RESTful API server với Swagger UI (`/docs`) |
| **Orchestration** | LangChain (`core`, `community`, `qdrant`, `ollama`) | Điều phối luồng RAG, Prompt & LLM Chain |
| **Vector Database** | Qdrant Server (Docker Container) | Lưu trữ & tìm kiếm vector tương đồng |
| **Embedding Model** | `BAAI/bge-m3` (HuggingFace) | Mã hóa ngữ cảnh văn bản thành vector (Offline CPU/GPU) |
| **LLM Engine** | Gemma 3 (`gemma3:4b-it-qat` qua Ollama) | Sinh câu trả lời y khoa chuẩn hóa |
| **OCR & Data Extraction** | Unstructured, Tesseract OCR, Poppler | Xử lý & trích xuất văn bản từ tài liệu PDF |

---

## 📂 Cấu Trúc Dự Án

```
AI_NhaKhoa/
├── backend/                  # Mã nguồn Backend FastAPI & RAG Core
│   ├── agent/                # Logic LangChain Agent & Tools
│   ├── app/                  # FastAPI Routers, Schemas, Services
│   │   └── api/              # Định tuyến API (v1)
│   ├── core/                 # Cấu hình dự án (config.py, vector_db.py)
│   ├── ingestion/            # Pipeline xử lý dữ liệu (pipeline.py)
│   └── main.py               # Entrypoint khởi chạy FastAPI Server
├── data/                     # Thư mục chứa dữ liệu
│   ├── raw/                  # Lưu tài liệu PDF gốc đầu vào
│   ├── processed/            # Lưu dữ liệu xuất/nhập JSON & cache
│   └── qdrant_storage/       # Dữ liệu lưu trữ persistent của Qdrant DB
├── docker-compose.yml        # Khởi tạo Qdrant Vector Server
├── streamlit_app.py          # Frontend Giao diện Web Chatbot (Streamlit)
├── test_api.py               # Giao diện dòng lệnh (CLI interactive)
├── inspect_db.py             # Công cụ kiểm tra dữ liệu vector DB
├── export_chunks.py          # Công cụ xuất chunks ra file JSON
├── project_analysis.md       # Báo cáo phân tích chi tiết dự án
├── requirements.txt          # Danh sách thư viện Python
└── .env.example              # Cấu hình biến môi trường mẫu
```

---

## 🚀 Hướng Dẫn Cài Đặt (Quick Start)

### 1. Yêu Cầu Tiền Trạm (Prerequisites)

* **Python:** `3.10` trở lên (Khuyên dùng Python `3.10` - `3.12`)
* **Docker Desktop:** Khởi chạy Qdrant Container
* **Ollama:** Cài đặt Ollama và tải model Gemma 3:
  ```bash
  ollama pull gemma3:4b-it-qat
  ```
* **Tesseract OCR & Poppler** (Dành cho tính năng OCR PDF):
  * **Windows:** Tải Tesseract OCR và thêm vào biến môi trường PATH.

---

### 2. Các Bước Cài Đặt

#### Bước 1: Khởi động Qdrant Server (Docker)
Tại thư mục gốc của dự án:
```bash
docker compose up -d
```
> Server Qdrant sẽ hoạt động tại `http://localhost:6333`.

#### Bước 2: Thiết lập biến môi trường
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```
Nội dung file `.env`:
```env
QDRANT_URL=http://localhost:6333
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b-it-qat
```

#### Bước 3: Cài đặt Dependencies
Kích hoạt môi trường ảo (virtual environment) và cài đặt thư viện:
```bash
# Khởi tạo venv (nếu chưa có)
python -m venv venv

# Kích hoạt venv (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Cài đặt thư viện
pip install -r requirements.txt
```

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Nạp & Quản Lý Dữ Liệu Tài Liệu (Pipeline Ingestion)

Đặt các tệp tài liệu PDF y khoa vào thư mục `data/raw/`.

* **Nạp & tính embedding trực tiếp vào Qdrant:**
  ```bash
  python backend/ingestion/pipeline.py --mode run
  ```
* **Trích xuất PDF và xuất vector ra tệp JSON (Export):**
  ```bash
  python backend/ingestion/pipeline.py --mode export --output data/processed/embeddings_export.json
  ```
* **Nhập nhanh vector từ tệp JSON vào Qdrant (Import không cần tính lại embedding):**
  ```bash
  python backend/ingestion/pipeline.py --mode import --input data/processed/embeddings_export.json
  ```

---

### 2. Khởi Chạy Giao Diện Web Streamlit

Dành cho người dùng tương tác trực tiếp với Trợ lý AI:
```bash
streamlit run streamlit_app.py
```
> Truy cập trình duyệt tại: [http://localhost:8501](http://localhost:8501)

*Tính năng giao diện Streamlit:*
* Quản lý nhiều phiên trò chuyện ở thanh Sidebar.
* Đo thời gian tìm kiếm ngữ cảnh & thời gian suy luận AI theo từng câu hỏi.
* Bảng hiển thị trích dẫn nguồn tài liệu tham chiếu (tên file PDF, trang).

---

### 3. Khởi Chạy FastAPI Server (Backend REST API)

Dành cho việc kết nối API với ứng dụng di động hoặc bên thứ 3:
```bash
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

* **API Base URL:** `http://127.0.0.1:8000`
* **Swagger UI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Docs:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

#### Endpoint Hỏi Đáp Example (Curl):
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" \
     -H "Content-Type: application/json" \
     -d "{\"message\": \"Các triệu chứng của viêm tủy răng là gì?\", \"limit\": 5}"
```

---

### 4. Khởi Chạy Chế Độ Dòng Lệnh (CLI Mode)

Dành cho kiểm tra trực tiếp qua Terminal:
```bash
python test_api.py
```

---

### 5. Công Cụ Trợ Giúp (Tools)

* **Kiểm tra dữ liệu trong Qdrant DB:**
  ```bash
  python inspect_db.py
  ```
  *(Xuất dữ liệu thô ra file `tat_ca_chunks.txt`)*

* **Xuất danh sách chunks sang dạng JSON:**
  ```bash
  python export_chunks.py
  ```
  *(Xuất ra file `tat_ca_chunks.json` kèm thống kê chi tiết)*

---

## ⚠️ Lưu Ý Y Tế & Miễn Trừ Trách Nhiệm (Disclaimer)

1. Hệ thống Trợ lý AI Nha khoa được thiết kế như một **công cụ hỗ trợ tra cứu thông tin chuyên môn** cho cán bộ y tế và học viên.
2. Các phản hồi của AI **không thay thế cho chẩn đoán lâm sàng, chỉ định y khoa hoặc quyết định chuyên môn của Bác sĩ có bằng cấp**.
