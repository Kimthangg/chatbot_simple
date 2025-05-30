﻿# Chatbot Simple
- ## 🎥 Video Demo  
https://github.com/user-attachments/assets/fe4df64e-480e-42f9-bc0d-3a2bcf8e3161

## Giới thiệu

Chatbot Simple là một chatbot được xây dựng bằng LangChain và Streamlit, có khả năng:

- Trả lời các câu hỏi tự nhiên của người dùng.
- Xử lý dữ liệu bảng điểm và cung cấp thông tin liên quan đến môn học, điểm số.
- Ghi nhớ ngữ cảnh hội thoại bằng kỹ thuật memory agent.
- Áp dụng phương pháp Retrieval-Augmented Generation (RAG) với kỹ thuật Naive Retriever để tìm kiếm thông tin nhanh chóng.

## Công nghệ sử dụng

- **LangChain**: Xây dựng pipeline cho chatbot.
- **FAISS**: Lưu trữ và tìm kiếm vector từ dữ liệu bảng điểm.
- **Streamlit**: Tạo giao diện web thân thiện cho người dùng.
- **GEMINI API**: Sử dụng mô hình ngôn ngữ lớn (LLM) để tạo câu trả lời.

## Cài đặt

1. Clone repository:
   ```sh
   git clone https://github.com/Kimthangg/chatbot_simple.git
   cd chatbot_simple
   ```
2. Cài đặt thư viện cần thiết:
   ```sh
   pip install -r requirements.txt
   ```
3. Chạy ứng dụng:
   ```sh
   streamlit run app.py
   ```

## Cách sử dụng

- Mở giao diện web tại `http://localhost:8501`.
- Nhập câu hỏi vào ô chat, chatbot sẽ trả lời dựa trên ngữ cảnh hội thoại và dữ liệu bảng điểm.
