from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
#Khai báo biến
path_pdf = 'data'
path_db = 'vectorDB/faiss_db'


def read_file():
    df = pd.read_csv("./data/Diem_new.csv", encoding='utf-8')

    df = df.astype(str)  # Đảm bảo toàn bộ dữ liệu là dạng string
    return df

def format_data_for_langchain(df):
    columns = " | ".join(df.columns)  # Lấy tiêu đề
    text_list = [columns]  # Bắt đầu với tiêu đề

    for _, row in df.iterrows():
        text = " | ".join(row.astype(str))  # Convert tất cả dữ liệu sang string
        text_list.append(text)

    return "\n".join(text_list)  # Ghép lại thành 1 chuỗi văn bản
raw_text = format_data_for_langchain(read_file())
def create_db():

    #Chia nhỏ văn bản
    text_splitter = CharacterTextSplitter(
        # separator='\n',
        chunk_size=10,
        chunk_overlap=5,
        # length_function=len
    )

    chucks = text_splitter.split_text(raw_text)
    #Tạo vector từ văn bản
    embeddings = GPT4AllEmbeddings(model_file = './models/all-MiniLM-L6-v2-f16.gguf')
    #Đưa vao FAISS Vector DB
    db = FAISS.from_texts(texts=chucks, embedding=embeddings)
    db.save_local(path_db)
    return db
create_db()
