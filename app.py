import streamlit as st
import time
from bot.agent import gemini_chatbot
from bot.pipe_rag import get_answer
from bot.agent_intent import supervised_model

st.set_page_config(page_title="Chat với Gemini", page_icon="💬")
st.title("🤖 Chatbot Gemini")


for message in st.session_state.chat_history.messages:
    if message.type == "AIMessageChunk": #Loại của nó là AIMessageChunk nhưng web hiểu là assistant mới đúng
        message.type = "assistant"
    with st.chat_message(message.type):
        st.markdown(message.content)


# Nhập tin nhắn mới
prompt = st.chat_input("Nhập tin nhắn...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Xử lý tin nhắn của assistant
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""
        
        intent = supervised_model(prompt)
        if intent == 'Có':
            # response_text = get_answer(prompt)
            for chunk in get_answer(prompt):
                time.sleep(0.01)
                response_text += chunk
                response_container.markdown(response_text + "▌")  # Hiển thị hiệu ứng typing
                
            #Lưu câu hỏi và câu trả lời vào lịch sử hội thoại
            st.session_state.chat_history.add_user_message(prompt)
            st.session_state.chat_history.add_ai_message(response_text)
        else:
            # Gọi API và xử lý stream
            for chunk in gemini_chatbot(prompt):
                chunk_split = chunk.content.split(' ')
                for chunk_nho in chunk_split:
                    time.sleep(0.01)  # Giả lập typing
                    response_text += chunk_nho + " "
                    response_container.markdown(response_text + "▌")  # Hiển thị hiệu ứng typing
        response_container.markdown(response_text)  # Xóa hiệu ứng khi hoàn tất