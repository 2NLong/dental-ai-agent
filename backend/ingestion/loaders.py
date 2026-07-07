import os
import io
import re
import urllib.request
import zipfile
import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class FallbackOCRPDFLoader:
    def __init__(self, file_path: str, cache_dir: str = None):
        self.file_path = file_path
        if cache_dir is None:
            # Thư mục cache mặc định nằm trong thư mục gốc của dự án: .cache/easyocr
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(os.path.dirname(current_dir))
            self.cache_dir = os.path.join(project_dir, ".cache", "easyocr")
        else:
            self.cache_dir = cache_dir

    def _ensure_models_downloaded(self):
        """Đảm bảo các file model đã được tải về ổ D trước khi khởi tạo EasyOCR."""
        os.makedirs(self.cache_dir, exist_ok=True)
        models = {
            "craft_mlt_25k.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
            "latin_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip"
        }
        for name, url in models.items():
            dest_zip = os.path.join(self.cache_dir, name)
            actual_pth = os.path.join(self.cache_dir, name.replace(".zip", ".pth"))
            
            if not os.path.exists(actual_pth):
                print(f"[FallbackOCR] Đang tự động tải mô hình {name}...")
                try:
                    req = urllib.request.Request(
                        url,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                    )
                    with urllib.request.urlopen(req) as response, open(dest_zip, 'wb') as out_file:
                        out_file.write(response.read())
                    
                    # Giải nén lấy file .pth
                    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
                        zip_ref.extractall(self.cache_dir)
                    os.remove(dest_zip)
                    print(f"[FallbackOCR] Tải và giải nén thành công {name}.")
                except Exception as e:
                    print(f"[FallbackOCR] Lỗi khi tải mô hình {name}: {e}")
                    # Xóa file zip lỗi nếu có
                    if os.path.exists(dest_zip):
                        os.remove(dest_zip)
                    raise e

    def _clean_ocr_results(self, results: list) -> str:
        """
        Nối các đoạn văn bản OCR lại với nhau một cách thông minh.
        Nếu dòng hiện tại không kết thúc bằng các dấu câu (như ., ?, !, :),
        ta sẽ nối với dòng tiếp theo bằng khoảng trắng thay vì dấu xuống dòng.
        """
        if not results:
            return ""
            
        cleaned_text = []
        for i, text in enumerate(results):
            text = text.strip()
            if not text:
                continue
                
            is_hyphenated = False
            if text.endswith('-') and len(text) > 1:
                text = text[:-1]
                is_hyphenated = True
                
            cleaned_text.append(text)
            
            if i < len(results) - 1:
                if is_hyphenated:
                    # Không thêm gì vì từ bị gạch ngang ngắt dòng
                    continue
                elif text[-1] in {'.', '?', '!', ':'}:
                    cleaned_text.append("\n")
                else:
                    cleaned_text.append(" ")
                    
        joined = "".join(cleaned_text)
        # Thay thế nhiều dấu xuống dòng liên tiếp bằng tối đa 2 dấu xuống dòng
        joined = re.sub(r'\n\s*\n', '\n\n', joined)
        return joined

    def load(self):
        # 1. Thử đọc bằng PyPDFLoader trước
        print(f"[FallbackOCR] Thử đọc bằng PyPDFLoader cho: {self.file_path}")
        pypdf_loader = PyPDFLoader(self.file_path)
        try:
            docs = pypdf_loader.load()
        except Exception as e:
            print(f"[FallbackOCR] PyPDFLoader gặp lỗi: {e}. Tiến hành chuyển sang luồng OCR.")
            docs = []

        total_text_len = sum(len(doc.page_content.strip()) for doc in docs)
        
        # Nếu trích xuất được lượng văn bản đáng kể (> 30 ký tự), giữ nguyên kết quả
        if docs and total_text_len > 30:
            print(f"[FallbackOCR] Đã trích xuất thành công {total_text_len} ký tự dạng kỹ thuật số. Giữ nguyên kết quả.")
            return docs
            
        # 2. Ngược lại, thực hiện OCR fallback sử dụng EasyOCR
        print(f"[FallbackOCR] Phát hiện tài liệu không có văn bản hoặc văn bản quá ngắn ({total_text_len} ký tự). Bắt đầu nhận diện OCR...")
        
        # Đảm bảo các mô hình đã tải xuống
        self._ensure_models_downloaded()
        
        # Khởi tạo EasyOCR Reader ngoại tuyến
        print("[FallbackOCR] Khởi tạo EasyOCR Reader ở chế độ offline...")
        reader = easyocr.Reader(
            ['vi', 'en'],
            model_storage_directory=self.cache_dir,
            download_enabled=False
        )
        
        ocr_docs = []
        try:
            # Mở file PDF bằng PyMuPDF (fitz)
            pdf_doc = fitz.open(self.file_path)
            for page_num, page in enumerate(pdf_doc):
                print(f"[FallbackOCR] Đang chạy OCR trang {page_num + 1}/{len(pdf_doc)}...")
                # Kết xuất trang thành ảnh PNG với DPI cao (300 DPI) để tăng độ nét chữ
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                
                # Chuyển đổi thành ảnh PIL và chuyển sang numpy array
                img = Image.open(io.BytesIO(img_bytes))
                img_np = np.array(img)
                
                # Nhận diện chữ bằng EasyOCR (bật paragraph=True để tự động gộp các cụm chữ gần nhau)
                results = reader.readtext(img_np, detail=0, paragraph=True)
                page_content = self._clean_ocr_results(results)
                
                # Tạo đối tượng Document chuẩn của LangChain
                doc = Document(
                    page_content=page_content,
                    metadata={
                        "source": self.file_path,
                        "page": page_num
                    }
                )
                ocr_docs.append(doc)
            
            pdf_doc.close()
            print(f"[FallbackOCR] Đã hoàn thành OCR cho {len(ocr_docs)} trang.")
            return ocr_docs
        except Exception as e:
            print(f"[FallbackOCR] Gặp lỗi nghiêm trọng trong quá trình xử lý OCR: {e}")
            # Trả về kết quả PyPDFLoader ban đầu hoặc danh sách rỗng thay vì làm sập pipeline
            return docs

def get_pdf_loader(file_path: str, method: str = "pypdf"):
    """
    Trả về đối tượng loader tương ứng với phương pháp được chọn.
    
    Hỗ trợ các phương pháp:
    - "pypdf": Sử dụng FallbackOCRPDFLoader tự động chuyển sang OCR nếu gặp file scan.
    """
    method = method.lower().strip()
    if method == "pypdf":
        return FallbackOCRPDFLoader(file_path)
    else:
        raise ValueError(f"Phương pháp nạp tài liệu (loader method) '{method}' chưa được hỗ trợ.")
