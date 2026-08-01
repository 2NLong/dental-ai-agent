from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload tài liệu nha khoa định dạng PDF để chuẩn bị nạp dữ liệu RAG (Placeholder)."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chi chap nhan file dinh dang PDF")
    
    return {
        "filename": file.filename,
        "status": "uploaded",
        "message": "File da duoc tai len thanh cong."
    }

@router.get("/list", response_model=List[str])
def list_documents():
    """Lấy danh sách các tài liệu đã nạp vào hệ thống (Placeholder)."""
    return ["tailieu_nha_khoa_1.pdf", "cam_nang_rang_mieng.pdf"]
