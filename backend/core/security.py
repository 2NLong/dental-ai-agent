# Xử lý bảo mật, băm mật khẩu, sinh JWT token
# Thư viện đề xuất: passlib, jose
def hash_password(password: str) -> str:
    return password  # Placeholder

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return plain_password == hashed_password  # Placeholder
