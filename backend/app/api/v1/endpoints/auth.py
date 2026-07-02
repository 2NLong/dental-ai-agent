from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter()

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    """Đăng nhập để lấy Access Token truy cập hệ thống."""
    # Placeholder login logic
    if payload.username == "admin" and payload.password == "admin123":
        return {
            "access_token": "mocked_jwt_token_for_dental_agent",
            "token_type": "bearer"
        }
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Tên đăng nhập hoặc mật khẩu không chính xác"
    )
