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
        # 1. Đọc bằng PyMuPDF (fitz) trước để tránh lỗi tách chữ tiếng Việt của pypdf
        print(f"[FallbackOCR] Đọc bằng PyMuPDF (fitz) cho: {self.file_path}")
        docs = []
        try:
            doc = fitz.open(self.file_path)
            for i, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": self.file_path,
                            "page": i  # 0-indexed để khớp với các loader khác
                        }
                    ))
        except Exception as e:
            print(f"[FallbackOCR] PyMuPDF gặp lỗi: {e}. Tiến hành chuyển sang luồng OCR.")
            docs = []

        total_text_len = sum(len(doc.page_content.strip()) for doc in docs)
        
        # Nếu trích xuất thành công text kỹ thuật số (> 30 ký tự), trả về kết quả
        if docs and total_text_len > 30:
            print(f"[FallbackOCR] Đã trích xuất thành công {total_text_len} ký tự dạng kỹ thuật số bằng PyMuPDF. Giữ nguyên kết quả.")
            return docs
            
        # 2. Ngược lại, nếu là file scan (không có chữ), chuyển sang OCR bằng IBM Docling
        print(f"[FallbackOCR] Phát hiện tài liệu không có văn bản hoặc văn bản quá ngắn ({total_text_len} ký tự). Bắt đầu nhận diện OCR bằng IBM Docling...")
        
        try:
            converter = get_docling_converter()
            print(f"[FallbackOCR] Đang chạy nhận dạng layout và cấu trúc cho: {self.file_path}")
            
            result = converter.convert(self.file_path)
            doc = result.document
            
            ocr_docs = []
            for page_no in range(1, len(result.pages) + 1):
                page_markdown = doc.export_to_markdown(page_no=page_no)
                if page_markdown.strip():
                    page_doc = Document(
                        page_content=page_markdown,
                        metadata={
                            "source": self.file_path,
                            "page": page_no - 1  # 0-indexed
                        }
                    )
                    ocr_docs.append(page_doc)
            
            print(f"[FallbackOCR] Đã hoàn thành OCR và trích xuất cấu trúc bằng Docling cho {len(ocr_docs)} trang thành công!")
            return ocr_docs
        except Exception as e:
            print(f"[FallbackOCR] Gặp lỗi nghiêm trọng trong quá trình xử lý OCR bằng Docling: {e}")
            import traceback
            traceback.print_exc()
            # Trả về kết quả PyMuPDF trước đó (nếu có) hoặc danh sách trống thay vì làm sập ứng dụng
            return docs


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
