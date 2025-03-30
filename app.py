import streamlit as st
import time
from agent import gemini_chatbot

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

        # Gọi API và xử lý stream
        for chunk in gemini_chatbot(prompt):
            time.sleep(0.2)  # Giả lập typing
            response_text += chunk.content
            response_container.markdown(response_text + "▌")  # Hiển thị hiệu ứng typing

        response_container.markdown(response_text)  # Xóa hiệu ứng khi hoàn tất