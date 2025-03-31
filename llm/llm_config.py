from langchain_openai import ChatOpenAI
# Cấu hình LLM với Gemini
API_KEY = "THAY BẰNG API KEY CỦA BẠN LẤY TỪ AI STUDIO"
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

llm = ChatOpenAI(
    model='gemini-2.0-flash',
    temperature=1,
    api_key=API_KEY,
    base_url=API_BASE,
    streaming=True
)