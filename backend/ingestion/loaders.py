import os
import sys
import pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader

# Cấu hình Tesseract và Poppler cho Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["PATH"] += os.pathsep + r'C:\poppler-26.02.0\Library\bin'
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Tesseract-OCR'

def get_pdf_loader(file_path: str, method: str = "unstructured"):
    """
    Trả về đối tượng Loader tài liệu tương ứng.
    Sử dụng UnstructuredPDFLoader với cấu hình Tesseract Tiếng Việt + Anh dạng hi_res.
    """
    method = method.lower().strip()
    if method == "unstructured":
        return UnstructuredPDFLoader(
            file_path=file_path,
            strategy="hi_res",
            languages=["vie", "eng"]
        )
    else:
        raise ValueError(f"Loader method '{method}' khong duoc ho tro trong he thong nay.")
