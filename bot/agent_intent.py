from llm.llm_config import llm
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)

system_prompt = """Bạn là trợ lý tuyệt vời có tên là Neura.
Nhiệm vụ của bạn là phân loại xem câu hỏi của người dùng có liên quan đến điểm số môn X, bảng điểm, điểm thi,... hay không.
Hãy suy luận theo từng bước:
1. Đọc câu hỏi của người dùng.
2. Xác định xem câu hỏi có liên quan đến điểm số môn X, bảng điểm, điểm thi,... hay không.
3. Nếu có, hãy trả lời 'Có', nếu không có, hãy trả lời 'Không'.
Bạn chỉ được phép trả lời 1 trong 2 đáp án là 'Có' hoặc 'Không'."""

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)
pipeline = prompt_template | llm

  
def supervised_model(prompt):
    response = pipeline.invoke({"query": prompt})
    return response.content.strip()

# response = pipeline.invoke({"query": "Tôi muốn biết về các loại thuốc điều trị bệnh tiểu đường."})
# print(response)
# if response.content.strip() == 'Có':
#     print("Câu hỏi cần sử dụng retrieval.")