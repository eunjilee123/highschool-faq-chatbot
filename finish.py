

## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

## 환경변수 불러오기
from dotenv import load_dotenv,dotenv_values
load_dotenv()



############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
import tiktoken

enc = tiktoken.encoding_for_model("text-embedding-3-small")

def get_token_count(text: str) -> int:
    return len(enc.encode(text))

def chunk_documents_by_tokens(documents, max_tokens=300000):
    chunks = []
    current_chunk = []
    current_token_count = 0

    for doc in documents:
        token_count = get_token_count(doc.page_content)
        if current_token_count + token_count > max_tokens:
            chunks.append(current_chunk)
            current_chunk = [doc]
            current_token_count = token_count
        else:
            current_chunk.append(doc)
            current_token_count += token_count

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def save_to_vector_store(documents: List[Document]) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chunks = chunk_documents_by_tokens(documents, max_tokens=300000)
    vector_store = None

    for chunk in chunks:
        current_index = FAISS.from_documents(chunk, embedding=embeddings)
        if vector_store is None:
            vector_store = current_index
        else:
            vector_store.merge_from(current_index)

    vector_store.save_local("faiss_index")



############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################


## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    ## 사용자 질문을 기반으로 관련문서 3개 검색 
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs



def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘

    컨텍스트 : {context}

    질문: {question}

    응답:"""

    custom_rag_prompt = PromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o-mini")

    return custom_rag_prompt | model | StrOutputParser()



############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    
    # 이미지 저장용 폴더 생성
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  #  각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

import streamlit as st
import os  # os도 필요하니까 추가
# 다른 필요한 import도 여기에

# ✅ set_page_config는 가장 위에서 단 한 번만!
st.set_page_config(page_title="고교학점제 FAQ 챗봇", layout="wide")

# 상단 안내 메시지
st.markdown("""
<style>
.notice-box {
    padding: 1em;
    background-color: #fff3cd;
    border-left: 6px solid #ffa500;
    color: #856404;
    border-radius: 5px;
    margin-bottom: 1em;
    font-weight: 500;
}
</style>

<div class="notice-box">
    ⚠️ 챗봇 답변이 부족할 수 있어요!<br>
    📄 아래 요약된 <strong>PDF 이미지 자료</strong>를 참고해 주세요.
</div>
""", unsafe_allow_html=True)











def main():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.header("고교학점제 FAQ 챗봇")

        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요",
                                        placeholder="윤리와 사상 성적 산출에 대해 알려줘")

        if user_question:
            response, context = process_question(user_question)
            st.write(response)
            i = 0 
            for document in context:
                with st.expander("관련 문서"):
                    st.write(document.page_content)
                    file_path = document.metadata.get('source', '')
                    page_number = document.metadata.get('page', 0) + 1
                    button_key =f"link_{file_path}_{page_number}_{i}"
                    reference_button = st.button(f"🔎 {os.path.basename(file_path)} pg.{page_number}", key=button_key)
                    if reference_button:
                        st.session_state.page_number = str(page_number)
                    i = i + 1
    with right_column:
        # page_number 호출
        page_number = st.session_state.get('page_number')
        if page_number:
            page_number = int(page_number)
            image_folder = "pdf_이미지"
            images = sorted(os.listdir(image_folder), key=natural_sort_key)
            print(images)
            image_paths = [os.path.join(image_folder, image) for image in images]
            print(page_number)
            print(image_paths[page_number - 1])
            display_pdf_page(image_paths[page_number - 1], page_number)


if __name__ == "__main__":
    main()

# 고정 PDF 경로 지정
# 고정 PDF 경로 지정 (Streamlit Cloud에서 사용 가능한 상대 경로로 수정)

# 파일 경로 설정
import os
from langchain_community.document_loaders import PyMuPDFLoader

def pdf_to_documents(pdf_path):
    if not os.path.isfile(pdf_path):
        raise ValueError(f"File path {pdf_path} is not a valid file.")
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(CURRENT_DIR, "public", "2022개정 고등학교 과목 선택 안내자료-경기도교육청.pdf")

pdf_document = pdf_to_documents(pdf_path)









 # 예시 파일명, 본인이 넣은 파일명으로 수정

# PDF를 문서로 변환 후 벡터DB 저장 (최초 실행 시만 필요)
with st.spinner("PDF 문서 로드 중..."):
    pdf_document = pdf_to_documents(pdf_path)
    smaller_documents = chunk_documents(pdf_document)
    save_to_vector_store(smaller_documents)

# PDF를 이미지로 변환해서 세션 상태로 임시 저장
with st.spinner("PDF 페이지를 이미지로 변환 중..."):
    images = convert_pdf_to_images(pdf_path)
    st.session_state.images = images


import subprocess
import time
import requests

# ngrok 실행 (백그라운드로)
subprocess.Popen(["ngrok", "http", "8501"])

# ngrok 연결 기다리기
time.sleep(5)

# ngrok 주소 가져오기
def get_ngrok_url():
    res = requests.get("http://127.0.0.1:4040/api/tunnels")
    tunnels = res.json()["tunnels"]
    for tunnel in tunnels:
        if tunnel["proto"] == "https":
            return tunnel["public_url"]
    return None

# 디스코드 전송
def send_to_discord(url):
    webhook_url = "https://discord.com/api/webhooks/..."  # 너의 webhook
    data = {"content": f"🟢 새로운 ngrok 주소가 생성되었습니다:\n{url}"}
    requests.post(webhook_url, json=data)

url = get_ngrok_url()
if url:
    send_to_discord(url)
    print("✅ 자동 전송 완료!")
else:
    print("❌ ngrok 주소를 못 찾았어요.")
