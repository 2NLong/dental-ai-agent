import os
import glob
import pytesseract
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from src import config

# =====================================================================
# CẤU HÌNH ĐƯỜNG DẪN TESSERACT VÀ POPPLER CHO WINDOWS
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["PATH"] += os.pathsep + r'C:\poppler-26.02.0\Library\bin'
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Tesseract-OCR'
# =====================================================================

QDRANT_PATH = "./data/processed/qdrant_local"
COLLECTION_NAME = "dental_knowledge"

def get_embedding_model():
    print("⏳ Đang khởi tạo mô hình nhúng BAAI/bge-m3 cục bộ...")
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_vector_db():
    pdf_files = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào!")
        return None

    all_docs = []
    print(f"📚 Tìm thấy {len(pdf_files)} file PDF. Đang quét OCR bằng Unstructured...")
    
    for pdf_path in pdf_files:
        print(f"📖 Đang xử lý chuyên sâu: {os.path.basename(pdf_path)}...")
        
        # Dùng UnstructuredPDFLoader với Tesseract Tiếng Việt + Anh
        loader = UnstructuredPDFLoader(
            file_path=pdf_path,
            strategy="hi_res",          
            languages=["vie", "eng"],   
        )
        all_docs.extend(loader.load())

    # Băm nhỏ văn bản
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"✂️ Đã băm tài liệu thành {len(chunks)} đoạn nhỏ tối ưu.")

    print("🔄 Đang tiến hành số hóa và lưu vào Qdrant...")
    embeddings = get_embedding_model()
    
    vector_db = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print(f"✅ HOÀN THÀNH! Dữ liệu đã lưu vào Qdrant tại: {QDRANT_PATH}")
    return vector_db

def load_vector_db():
    embeddings = get_embedding_model()
    if os.path.exists(QDRANT_PATH):
        client = QdrantClient(path=QDRANT_PATH)
        return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embeddings)
    return None

if __name__ == "__main__":
    build_vector_db()