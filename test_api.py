import requests
import json

# URL API của dự án
url = "http://127.0.0.1:8000/api/v1/chat/query"

# Các câu hỏi mẫu để test
questions = [
    " Tỷ lệ viêm nướu của cả nước?"
]

for idx, q in enumerate(questions):
    print(f"\n================ TEST CAU HOI {idx+1} ================")
    print(f"Cau hoi: {q}")
    payload = {
        "message": q,
        "session_id": f"session_test_{idx+1}"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            res_data = response.json()
            print("\n-> CAU TRA LOI CUA AI AGENT:")
            print(res_data.get("answer"))
            print("\n-> NGUON TAI LIEU SU DUNG:")
            print(res_data.get("sources"))
        else:
            print(f"Loi API: Status {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Loi ket noi den API: {e}")
