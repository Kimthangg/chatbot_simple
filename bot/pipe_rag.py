#Load llm
from llm.llm_config import llm
def load_llm():
    llm.temperature = 0.1
    return llm

#Tạo template prompt
from langchain.prompts import PromptTemplate
# Tạo template prompt
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=['context', 'question'])
    return prompt
#Tạo pipeline retrieval
from langchain.chains import RetrievalQA
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents = False,
        chain_type_kwargs = {'prompt': prompt},

    )
    return llm_chain
#Load db
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
embedding_model = GPT4AllEmbeddings(model_file='./models/all-MiniLM-L6-v2-f16.gguf')
db = FAISS.load_local('./VectorDB/faiss_db',embedding_model,allow_dangerous_deserialization=True)

#Đưa ra câu trả lời tù context
def get_answer(question):
    #Tạo LLM
    llm = load_llm()
    #Tạo template prompt
    template = """    Hãy trả lời câu hỏi dựa trên thông tin sau(trả lời tự nhiên thêm 1 dòng chú ý ở đầu là: "THÔNG TIN TỪ DATABASE"). Nếu không có đủ thông tin, hãy trả lời 'Tôi không biết'.

    Câu hỏi: {question}
    Ngữ cảnh: {context}

    Bước 1: Xác định thông tin liên quan trong ngữ cảnh.
    Bước 2: Tổng hợp thông tin để trả lời câu hỏi một cách chính xác.
    Bước 3: Đưa ra câu trả lời cuối cùng.
"""
    prompt = create_prompt(template)
    #Tạo chain
    chain = create_qa_chain(prompt, llm, db)
    response = chain.invoke({'query': question})
    return response['result']
