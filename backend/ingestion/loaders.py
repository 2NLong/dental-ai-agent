import io
import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyMuPDFLoader,
    PyPDFLoader,
)


class UnstructuredOCRPDFLoader:
    """
    Loader sử dụng UnstructuredPDFLoader (như trong notebook RAG_NhaKhoa_Final_OCR.ipynb).
    Tự động nhận diện lớp văn bản có sẵn hoặc sử dụng Tesseract OCR cho PDF dạng scan/ảnh.
    """
    def __init__(self, file_path: str, strategy: str = "auto", languages: list[str] | None = None):
        self.file_path = file_path
        self.strategy = strategy
        # 'vie' cho tiếng Việt, 'eng' cho tiếng Anh (thuật ngữ y khoa)
        self.languages = languages or ["vie", "eng"]

    def load(self) -> list[Document]:
        print(f"[UnstructuredOCR] Đang nạp tài liệu bằng UnstructuredPDFLoader (strategy={self.strategy}): {self.file_path}")
        try:
            loader = UnstructuredPDFLoader(
                self.file_path,
                mode="paged",
                strategy=self.strategy,
                languages=self.languages,
            )
            docs = loader.load()
            
            # Chuẩn hóa metadata để tương thích với phần còn lại của dự án
            for doc in docs:
                # Lấy page_number từ Unstructured và phòng tránh trường hợp page_number là None
                page_number = doc.metadata.get("page_number") or 1
                doc.metadata["page"] = page_number - 1
                doc.metadata["source"] = self.file_path
                
            return docs
        except Exception as e:
            print(f"[UnstructuredOCR] Lỗi nghiêm trọng khi đọc file {self.file_path} qua Unstructured: {e}")
            return []


class EasyOCRPDFLoader:
    """
    Loader sử dụng PyMuPDF kết hợp với EasyOCR cho các file PDF dạng scan/ảnh hoặc tiếng Việt.
    """
    # Cache reader để tái sử dụng, tránh khởi tạo lại weights PyTorch nhiều lần
    _reader_cache = {}

    def __init__(self, file_path: str, languages: list[str] | None = None, gpu: bool | None = None):
        self.file_path = file_path
        # 'vi' cho tiếng Việt, 'en' cho tiếng Anh (thuật ngữ y khoa)
        self.languages = languages or ["vi", "en"]
        
        # Tự động phát hiện GPU nếu không được chỉ định rõ
        if gpu is None:
            try:
                import torch
                self.gpu = torch.cuda.is_available()
            except ImportError:
                self.gpu = False
        else:
            self.gpu = gpu

    def _get_reader(self) -> easyocr.Reader:
        cache_key = (tuple(sorted(self.languages)), self.gpu)
        if cache_key not in EasyOCRPDFLoader._reader_cache:
            print(f"[EasyOCR] Khởi tạo Reader mới cho {self.languages} (GPU={self.gpu})")
            EasyOCRPDFLoader._reader_cache[cache_key] = easyocr.Reader(self.languages, gpu=self.gpu)
        return EasyOCRPDFLoader._reader_cache[cache_key]

    def load(self) -> list[Document]:
        print(f"[EasyOCR] Đang nạp tài liệu bằng EasyOCRPDFLoader: {self.file_path}")
        documents = []
        try:
            reader = self._get_reader()
            
            # Sử dụng context manager 'with' để tự động đóng file
            with fitz.open(self.file_path) as doc:
                for page_idx, page in enumerate(doc):
                    # Tăng độ phân giải trang (zoom=2 ~144 DPI) để cải thiện độ chính xác OCR
                    zoom = 2
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    
                    png_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(png_bytes))
                    img_np = np.array(img)
                    text_list = reader.readtext(img_np, detail=0)
                    page_text = " ".join(text_list)
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={"source": self.file_path, "page": page_idx}
                        )
                    )
            return documents
        except Exception as e:
            print(f"[EasyOCR] Lỗi nghiêm trọng khi đọc file {self.file_path} qua EasyOCR: {e}")
            return []


def get_pdf_loader(file_path: str, method: str = "unstructured"):
    """
    Trả về đối tượng loader tương ứng với phương pháp được chọn.
    
    Hỗ trợ 'unstructured', 'easyocr', 'pymupdf', và 'pypdf'.
    """
    method = method.lower().strip()
    if method == "unstructured":
        return UnstructuredOCRPDFLoader(file_path)
    elif method == "easyocr":
        return EasyOCRPDFLoader(file_path)
    elif method == "pymupdf":
        return PyMuPDFLoader(file_path)
    elif method == "pypdf":
        return PyPDFLoader(file_path)
    else:
        raise ValueError(f"Phương pháp nạp tài liệu (loader method) '{method}' chưa được hỗ trợ.")
