# Xử lý bảo mật, băm mật khẩu, sinh JWT token
# Thư viện đề xuất: passlib, jose

def hash_password(password: str) -> str:
    """Băm mật khẩu (Placeholder cho cấu hình sau)."""
    return password

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Đối sánh mật khẩu gốc với mật khẩu đã băm (Placeholder)."""
    return plain_password == hashed_password
