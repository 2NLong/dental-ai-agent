import os
import time
import streamlit as st
from src.database import load_vector_db
from src.agent import get_llm_response

# Cấu hình trang
st.set_page_config(page_title="Trợ lý AI Nha Khoa", page_icon="🦷", layout="wide")

# --- 1. KHỞI TẠO BỘ NHỚ LƯU TRỮ (SESSION STATE) ---
if "chats" not in st.session_state:
    st.session_state.chats = {"Cuộc trò chuyện 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Cuộc trò chuyện 1"

# --- 2. GIAO DIỆN THANH BÊN (SIDEBAR) TƯƠNG TỰ CHATGPT ---
with st.sidebar:
    st.title("📚 Lịch sử hội thoại")
    
    # Nút tạo cuộc trò chuyện mới
    if st.button("➕ Hội thoại mới", use_container_width=True):
        new_chat_id = f"Cuộc trò chuyện {len(st.session_state.chats) + 1}"
        st.session_state.chats[new_chat_id] = []
        st.session_state.current_chat = new_chat_id
        st.rerun()

    st.divider()
    
    # Danh sách các cuộc trò chuyện (Hiển thị nút chọn và nút xóa)
    st.write("**Gần đây:**")
    for chat_name in list(st.session_state.chats.keys()):
        cols = st.columns([4, 1]) # Chia tỷ lệ 4 phần cho tên chat, 1 phần cho nút xóa
        
        # Nút để chọn/chuyển đổi cuộc trò chuyện
        with cols[0]:
            is_active = (st.session_state.current_chat == chat_name)
            if st.button(chat_name, key=f"btn_{chat_name}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.current_chat = chat_name
                st.rerun()
                
        # Nút để xóa cuộc trò chuyện
        with cols[1]:
            if st.button("🗑️", key=f"del_{chat_name}", help="Xóa hội thoại này"):
                del st.session_state.chats[chat_name]
                # Xử lý logic nếu xóa trúng cuộc trò chuyện đang mở
                if st.session_state.current_chat == chat_name:
                    st.session_state.current_chat = list(st.session_state.chats.keys())[0] if st.session_state.chats else None
                # Nếu xóa hết sạch, tự động tạo lại hội thoại mặc định
                if not st.session_state.chats:
                    st.session_state.chats = {"Cuộc trò chuyện 1": []}
                    st.session_state.current_chat = "Cuộc trò chuyện 1"
                st.rerun()

# --- 3. GIAO DIỆN MAIN (KHUNG CHAT CHÍNH) ---
st.title(f"🦷 Trợ lý AI Nha Khoa - {st.session_state.current_chat}")

@st.cache_resource
def init_db():
    return load_vector_db()

db = init_db()

if db is None:
    st.error("❌ Hệ thống chưa có dữ liệu Vector!")
else:
    # A. Load lịch sử tin nhắn của cuộc trò chuyện hiện tại lên màn hình
    current_chat_history = st.session_state.chats[st.session_state.current_chat]
    for msg in current_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # B. Ô nhập liệu cho người dùng (nằm cố định ở dưới cùng như ChatGPT)
    if user_query := st.chat_input("💬 Nhập câu hỏi hoặc triệu chứng cần tư vấn tại đây..."):
        
        # 1. Hiển thị câu hỏi của User lên giao diện ngay lập tức
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # 2. Lưu câu hỏi vào lịch sử (RAM)
        st.session_state.chats[st.session_state.current_chat].append({"role": "user", "content": user_query})

        # 3. Kích hoạt quy trình xử lý RAG & LLM
        with st.chat_message("assistant"):
            with st.spinner("🔍 Đang tra cứu tài liệu và phân tích..."):
                
                # --- ĐO THỜI GIAN TÌM KIẾM ---
                start_retrieval = time.time()
                relevant_docs = db.similarity_search(user_query, k=7)
                retrieval_time = time.time() - start_retrieval
                
                # --- ĐO THỜI GIAN SUY LUẬN AI ---
                start_generation = time.time()
                ai_answer = get_llm_response(query=user_query, context_documents=relevant_docs)
                generation_time = time.time() - start_generation
                
            # Hiển thị câu trả lời chính của AI
            st.markdown(ai_answer)
            
            # Khung hiển thị thông số và trích dẫn tài liệu cực ngầu
            with st.expander("📊 Thông số thực thi & Nguồn tài liệu tham chiếu"):
                # Cột thông số
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="🔍 Tìm kiếm ngữ cảnh", value=f"{retrieval_time:.2f} s")
                with col2:
                    st.metric(label="🧠 Suy luận AI (Local)", value=f"{generation_time:.2f} s")
                with col3:
                    st.metric(label="📚 Số đoạn trích xuất", value=f"{len(relevant_docs)} chunks")
                
                st.divider()
                st.write("**Nội dung tài liệu được dùng để suy luận:**")
                # Hiển thị nội dung nguồn
                for i, doc in enumerate(relevant_docs):
                    src = doc.metadata.get('source', 'Không rõ')
                    page = doc.metadata.get('page', 'Không rõ')
                    st.markdown(f"**📍 Trích đoạn #{i+1} | Nguồn: {src} (Trang {page})**")
                    st.caption(doc.page_content)
                    st.markdown("---")
                    
        # 4. Lưu câu trả lời của AI vào lịch sử (RAM) để lần sau mở lại vẫn thấy
        st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": ai_answer})