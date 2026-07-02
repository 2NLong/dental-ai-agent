from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload tài liệu nha khoa định dạng PDF để nạp dữ liệu RAG."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file định dạng PDF")
    
    # Placeholder save & ingest logic
    return {
        "filename": file.filename,
        "status": "uploaded",
        "message": "File đã được tải lên thành công và đang chờ xử lý."
    }

@router.get("/list", response_model=List[str])
def list_documents():
    """Lấy danh sách các tài liệu đã nạp vào hệ thống."""
    return ["tailieu_nha_khoa_1.pdf", "cam_nang_rang_mieng.pdf"]
