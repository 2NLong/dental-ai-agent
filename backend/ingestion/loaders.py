from langchain_community.document_loaders import PyPDFLoader

def get_pdf_loader(file_path: str, method: str = "pypdf"):
    """
    Trả về đối tượng loader tương ứng với phương pháp được chọn.
    
    Hỗ trợ các phương pháp:
    - "pypdf": Sử dụng PyPDFLoader của LangChain (giữ siêu dữ liệu trang)
    """
    method = method.lower().strip()
    if method == "pypdf":
        return PyPDFLoader(file_path)
    else:
        raise ValueError(f"Phương pháp nạp tài liệu (loader method) '{method}' chưa được hỗ trợ.")
