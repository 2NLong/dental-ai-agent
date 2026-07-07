import os
import urllib.request
import zipfile

# Đường dẫn thư mục cache lưu trữ models
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, ".cache", "easyocr")
os.makedirs(cache_dir, exist_ok=True)

models = {
    "craft_mlt_25k.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
    "latin_g2.zip": "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/latin_g2.zip"
}

print(f"=== BẮT ĐẦU TẢI MODEL OFFLINE CHO EASYOCR ===")
print(f"Thư mục lưu trữ: {cache_dir}\n")

for name, url in models.items():
    dest_zip = os.path.join(cache_dir, name)
    dest_pth = os.path.join(cache_dir, name.replace(".zip", ".pth"))
    
    # Kiểm tra nếu model đã tồn tại thì bỏ qua tải xuống
    if os.path.exists(dest_pth):
        print(f"[-] Gói model {name.replace('.zip', '.pth')} đã tồn tại. Bỏ qua...")
        continue
        
    print(f"[+] Đang tải {url} -> {dest_zip}...")
    try:
        # Sử dụng User-Agent giả lập trình duyệt để tránh bị chặn tải từ GitHub
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_zip, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"[+] Tải thành công {name}. Đang tiến hành giải nén...")
        
        # Giải nén file zip lấy file .pth
        with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print(f"[+] Giải nén thành công {name} vào {cache_dir}")
        
        # Xóa file zip sau khi giải nén xong
        os.remove(dest_zip)
    except Exception as e:
        print(f"[!] Lỗi khi tải/giải nén {name}: {e}")

print("\n=== HOÀN THÀNH TẢI MODEL EASYOCR ===")
