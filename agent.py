from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import time
import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# Cấu hình LLM với Gemini
API_KEY = "THAY THẾ BẰNG KEY CỦA BẠN LẤY TẠI AI STUDIO"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

llm = ChatOpenAI(
    model='gemini-2.0-flash',
    temperature=1,
    max_tokens=1024,
    api_key=API_KEY,
    base_url=API_BASE,
    streaming=True
)

system_prompt = "You are a helpful assistant called Neura."

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
pipeline = prompt_template | llm

# Quản lý lịch sử hội thoại với InMemoryChatMessageHistory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

def get_chat_history(_):
    return st.session_state.chat_history


pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
)

def gemini_chatbot(prompt: str):
    response = pipeline_with_history.stream({"query": prompt}, config={"session_id": "id_123"})
    return response
