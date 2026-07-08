import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM  # Cập nhật thư viện để kết nối trực tiếp với Ollama

# Nạp các biến môi trường (Chạy Local hoàn toàn nên không bắt buộc phải có GEMINI_API_KEY nữa)
load_dotenv()

def get_llm_response(query: str, context_documents: list) -> str:
    """
    Đưa ngữ cảnh (Context) từ PDF và câu hỏi (Query) vào mô hình Local Gemma 3 
    để suy luận và tạo câu trả lời của Bác sĩ.
    """
    # 1. Khởi tạo mô hình Local Gemma 3 (Bản 4B nén QAT chạy siêu mượt và nhẹ trên máy 16GB RAM)
    llm = OllamaLLM(
        model="gemma3:4b-it-qat",
        temperature=0.0  # Đặt nhiệt độ bằng 0.0 để AI tuyệt đối bám sát tài liệu, không bịa thông tin y khoa
    )
    
    # 2. Gộp nội dung các đoạn văn bản tìm được từ PDF làm Ngữ cảnh
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    
    # 3. Định hình cấu trúc Prompt và ép Format câu trả lời rõ ràng dành cho Bác sĩ Nha Khoa
    prompt_template = """
    Bạn là một Trợ lý AI Nha Khoa chuyên nghiệp, có kiến thức chuyên sâu và tận tâm. Nhiệm vụ của bạn là giải đáp thắc mắc dựa HOÀN TOÀN vào tài liệu nha khoa cung cấp dưới đây.

    YÊU CẦU QUY TRÌNH TƯ VẤN VÀ KIỂM TRA:
    1. Phân tích triệu chứng mà khách hàng đưa ra dựa trên tài liệu.
    2. Đưa ra các phương án kiểm tra, chẩn đoán hoặc hướng xử lý phù hợp được ghi nhận trong tài liệu.
    3. Nếu tài liệu không có thông tin này, hãy từ tốn giải thích: "Kiến thức này nằm ngoài phạm vi tài liệu hiện tại của phòng khám". Tuyệt đối không tự bịa thông tin y khoa.

    YÊU CẦU BẮT BUỘC VỀ FORMAT (ĐỊNH DẠNG CÂU TRẢ LỜI):
    - Sử dụng các tiêu đề rõ ràng bằng Markdown (dùng dấu ## hoặc ###).
    - Xuống dòng hợp lý, phân tách các ý bằng dấu gạch đầu dòng (*) hoặc số thứ tự (1, 2, 3).
    - In đậm (**chữ cần nhấn mạnh**) các thuật ngữ nha khoa quan trọng, triệu chứng nguy hiểm hoặc tên thuốc/phương pháp.
    - Cấu trúc bài viết phải đi theo form: 
       ### 1. Nhận định / Chẩn đoán sơ bộ
       ### 2. Các bước kiểm tra / Triệu chứng lâm sàng liên quan
       ### 3. Hướng xử lý & Lời khuyên bác sĩ

    NỘI DUNG TÀI LIỆU KHẢO SÁT (CONTEXT):
    {context}

    CÂU HỎI CỦA KHÁCH HÀNG (QUERY):
    {query}

    CÂU TRẢ LỜI CỦA BÁC SĨ AI:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])
    
    # 4. Định hình chuỗi lệnh gửi đi thông qua cơ chế Pipe (|) của LangChain
    chain = prompt | llm
    
    # 5. Chạy mô hình Gemma 3 Offline và lấy kết quả trả về dưới dạng chuỗi văn bản (String)
    response = chain.invoke({"context": context_text, "query": query})
    
    return response