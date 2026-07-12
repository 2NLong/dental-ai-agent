import os

# Configure Docling and PyTorch cache directories at the absolute top of the file
# to ensure downstream packages (like langchain, huggingface_hub, transformers) read them first.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(os.path.dirname(current_dir))
cache_dir = os.path.join(project_dir, ".cache")

os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")

import pyarrow  # Import pyarrow first to avoid DLL conflict crash on Windows
import io
import re
import urllib.request
import zipfile
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Global cached instance of DocumentConverter to avoid reloading models on every file load
_converter = None

def get_docling_converter():
    """Lazily initializes and returns the shared DocumentConverter with CUDA support."""
    global _converter
    if _converter is None:
        print("[FallbackOCR] Khởi tạo IBM Docling với cấu hình GPU...")
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
            
            # Cấu hình rõ ràng ngôn ngữ tiếng Việt + tiếng Anh cho EasyOCR mặc định của Docling
            # Đồng thời đặt cache của EasyOCR trong thư mục dự án để tránh đầy ổ C
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = EasyOcrOptions(
                lang=["vi", "en"],
                force_full_page_ocr=True,
                model_storage_directory=os.path.join(cache_dir, "easyocr")
            )
            
            # Kích hoạt nhận diện cấu trúc bảng biểu
            pipeline_options.do_table_structure = True
            
            # Initialize DocumentConverter with format_options and PyPdfiumDocumentBackend
            _converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend
                    )
                }
            )
            print("[FallbackOCR] Khởi tạo IBM Docling thành công trên GPU!")
        except Exception as e:
            print(f"[FallbackOCR] Lỗi khi khởi tạo IBM Docling: {e}")
            raise e
    return _converter


class FallbackOCRPDFLoader:
    def __init__(self, file_path: str, cache_dir: str = None):
        self.file_path = file_path
        # Keeping parameter for backward compatibility
        self.cache_dir = cache_dir

    def load(self):
        print(f"[FallbackOCR] Đang xử lý tài liệu: {self.file_path}")
        
        try:
            doc = fitz.open(self.file_path)
            num_pages = len(doc)
            docs = [None] * num_pages  # Dùng placeholder để giữ đúng thứ tự trang
            scanned_pages = []
            
            # Bước 1: Quét nhanh từng trang bằng PyMuPDF để lấy chữ kỹ thuật số
            for i in range(num_pages):
                page = doc[i]
                text = page.get_text()
                
                # Kiểm tra xem trang có chứa bảng biểu hay không
                has_tables = len(page.find_tables().tables) > 0
                
                # Nếu trang có đủ chữ VÀ không có bảng biểu -> dùng PyMuPDF đọc trực tiếp
                if len(text.strip()) > 30 and not has_tables:
                    docs[i] = Document(
                        page_content=text,
                        metadata={
                            "source": self.file_path,
                            "page": i  # 0-indexed
                        }
                    )
                else:
                    # Nếu là trang scan (không có chữ) HOẶC trang có chứa bảng biểu
                    # -> Đưa vào danh sách chạy Docling để dựng cấu trúc Markdown/OCR
                    scanned_pages.append(i + 1)
            
            # Bước 2: Chạy OCR/Layout bằng IBM Docling cho các trang bị thiếu text hoặc chứa bảng (nếu có)
            if scanned_pages:
                print(f"[FallbackOCR] Phát hiện {len(scanned_pages)}/{num_pages} trang cần chạy Docling (scan/bảng): {scanned_pages}")
                converter = get_docling_converter()
                
                # Nếu số trang cần xử lý chiếm phần lớn (ví dụ >= 50% tổng số trang), chạy một lần cho cả tài liệu
                if len(scanned_pages) >= num_pages / 2:
                    print(f"[FallbackOCR] Đang chạy Docling trên GPU cho toàn bộ tài liệu...")
                    result = converter.convert(self.file_path)
                    for page_no in scanned_pages:
                        try:
                            page_markdown = result.document.export_to_markdown(page_no=page_no)
                            if page_markdown.strip():
                                docs[page_no - 1] = Document(
                                    page_content=page_markdown,
                                    metadata={
                                        "source": self.file_path,
                                        "page": page_no - 1
                                    }
                                )
                        except Exception as page_err:
                            print(f"[FallbackOCR] Lỗi trích xuất trang Docling {page_no}: {page_err}")
                else:
                    # Nếu chỉ có vài trang cần xử lý, chạy riêng lẻ từng trang để tối ưu thời gian GPU
                    for page_no in scanned_pages:
                        try:
                            print(f"[FallbackOCR] Đang chạy Docling trên GPU cho trang {page_no}...")
                            result = converter.convert(self.file_path, page_range=(page_no, page_no))
                            page_markdown = result.document.export_to_markdown()
                            if page_markdown.strip():
                                docs[page_no - 1] = Document(
                                    page_content=page_markdown,
                                    metadata={
                                        "source": self.file_path,
                                        "page": page_no - 1
                                    }
                                )
                        except Exception as page_err:
                            print(f"[FallbackOCR] Lỗi Docling trang {page_no}: {page_err}")
            
            # Lọc bỏ các trang None (các trang không có chữ và Docling cũng không quét ra gì - ví dụ trang trắng)
            final_docs = [d for d in docs if d is not None]
            print(f"[FallbackOCR] Đã hoàn thành xử lý. Tổng số trang hợp lệ: {len(final_docs)}/{num_pages}")
            return final_docs
            
        except Exception as e:
            print(f"[FallbackOCR] Gặp lỗi nghiêm trọng trong quá trình xử lý tài liệu: {e}")
            import traceback
            traceback.print_exc()
            # Trả về các trang đã đọc được (nếu có) hoặc danh sách trống thay vì làm sập ứng dụng
            return [d for d in docs if d is not None]


def get_pdf_loader(file_path: str, method: str = "pypdf"):
    """
    Returns the loader instance based on selected method.
    
    Supported methods:
    - "pypdf": Uses FallbackOCRPDFLoader which automatically switches to Docling OCR for scanned files.
    """
    method = method.lower().strip()
    if method == "pypdf":
        return FallbackOCRPDFLoader(file_path)
    else:
        raise ValueError(f"Phương pháp nạp tài liệu (loader method) '{method}' chưa được hỗ trợ.")
