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
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)
            
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
        # 1. Try reading with PyPDFLoader first
        print(f"[FallbackOCR] Thử đọc bằng PyPDFLoader cho: {self.file_path}")
        pypdf_loader = PyPDFLoader(self.file_path)
        try:
            docs = pypdf_loader.load()
        except Exception as e:
            print(f"[FallbackOCR] PyPDFLoader gặp lỗi: {e}. Tiến hành chuyển sang luồng OCR.")
            docs = []

        total_text_len = sum(len(doc.page_content.strip()) for doc in docs)
        
        # If successfully extracted digital text (> 30 characters), return it
        if docs and total_text_len > 30:
            print(f"[FallbackOCR] Đã trích xuất thành công {total_text_len} ký tự dạng kỹ thuật số. Giữ nguyên kết quả.")
            return docs
            
        # 2. Otherwise, fall back to OCR using IBM Docling
        print(f"[FallbackOCR] Phát hiện tài liệu không có văn bản hoặc văn bản quá ngắn ({total_text_len} ký tự). Bắt đầu nhận diện OCR bằng IBM Docling...")
        
        try:
            converter = get_docling_converter()
            print(f"[FallbackOCR] Đang chạy nhận dạng layout và cấu trúc cho: {self.file_path}")
            
            # Convert PDF using Docling
            result = converter.convert(self.file_path)
            
            # Export the entire document to structured Markdown
            markdown_content = result.document.export_to_markdown()
            
            # Create a LangChain Document with the structured Markdown
            doc = Document(
                page_content=markdown_content,
                metadata={
                    "source": self.file_path,
                    "page": 0
                }
            )
            
            print(f"[FallbackOCR] Đã hoàn thành OCR và trích xuất cấu trúc bằng Docling thành công!")
            return [doc]
        except Exception as e:
            print(f"[FallbackOCR] Gặp lỗi nghiêm trọng trong quá trình xử lý OCR bằng Docling: {e}")
            import traceback
            traceback.print_exc()
            # Return original PyPDFLoader results or empty list instead of crashing
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
