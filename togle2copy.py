import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import time


llm = ChatOpenAI(model="gpt-4o")

def simulate_finetuning():
    progress_container = st.empty()
    with progress_container.container():
        # 파인튜닝 중임을 알리는 큰 경고 메시지
        st.warning("⚠️ 파인튜닝이 진행 중입니다. 완료될 때까지 기다려주세요.")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(5):
            progress_bar.progress(i*25)
            status_text.text(f"파인튜닝 진행중... {i*25}%")
            time.sleep(1)
        
        status_text.text("파인튜닝이 완료되었습니다!")
        time.sleep(2)
    
    progress_container.empty()


class LlamaInference:
    def __init__(self, model_path):
        """모델과 토크나이저 초기화"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|end_of_text|>'
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('<|end_of_text|>')
        
        self.model.eval()

    def format_prompt(self, question, context,sentences):
        return f"""### You are a helpful assistant that provides accurate answers based on manuals and documentation.
You must:
- Only answer within the given context and sentences
- Never provide information outside of the provided documents
- Respond in natural Korean language
- Say "제공된 문서에서 해당 내용을 찾을 수 없습니다" if the information is not in the context
- Be polite and professional in your responses
### Question:
{question}

### Context:
{context}
### Sentences:
{sentences}

### Answer:"""

    def generate_answer(self, question, context, sentences):
        prompt = self.format_prompt(question, context, sentences)
        print(prompt)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        ).to(self.model.device)
        
        gen_config = {
        "do_sample": False,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 50,
        "max_length": 512,
        "min_length": 10,
        "no_repeat_ngram_size": 10,
        "num_beams": 10,
        "early_stopping": True,
        # "bad_words_ids": None,
        "pad_token_id": self.tokenizer.pad_token_id,
        "eos_token_id": self.tokenizer.convert_tokens_to_ids('<|end_of_text|>'),
        "use_cache": False
        }
        
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        return answer

@st.cache_resource
def init_korquad_model():
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    return LlamaInference(model_path)

def init_session_state():
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'retriever_context' not in st.session_state:
        st.session_state.retriever_context = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'use_korquad' not in st.session_state:
        st.session_state.use_korquad = False
    if 'enable_context' not in st.session_state:
        st.session_state.enable_context = True
    if 'is_finetuning' not in st.session_state:
        st.session_state.is_finetuning = False

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[".", ". ", "\n\n"],
        chunk_size=10,
        chunk_overlap=0
    )
    return text_splitter.split_text(raw_text)

def get_context(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n\n\n", "\n\n\n\n","\n"],
        chunk_size = 300,
        chunk_overlap=30
    )
    return text_splitter.split_text(raw_text)

def get_vectorstore(chunks,text_contexts, use_korquad=False):
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": "test.pdf"}
        ) for chunk in chunks
    ]
    
    documents_context = [
        Document(
            page_content=chunk,
            metadata={"source": "test.pdf"}
        ) for chunk in text_contexts
    ]
    
    # KorQuAD 모델 사용 시 다국어 임베딩 사용
    embedding_model = (
        HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')
        if use_korquad
        else OpenAIEmbeddings(model="text-embedding-3-large")
    )
    if use_korquad:
        database = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name='pdf',
            persist_directory='./chroma_hugging_pdf'
        )
        database_context = Chroma.from_documents(
            documents=documents_context,
            embedding=embedding_model,
            collection_name='pdf',
            persist_directory='./chroma_context_hugging_pdf'
        )
    else:
        database = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name='pdf',
            persist_directory='./chroma_pdf'
        )
        database_context = Chroma.from_documents(
            documents=documents_context,
            embedding=embedding_model,
            collection_name='pdf',
            persist_directory='./chroma_context_pdf'
        )

    return database.as_retriever(search_kwargs={"k": 5}), database_context.as_retriever(search_kwargs={"k": 1})

def get_ai_response(retriever, retriever_context, user_msg, korquad_model=None):
    if st.session_state.enable_context:
        retriever_context_docs = retriever_context.invoke(user_msg)
        retrieved_docs = retriever.invoke(user_msg)
        
        print(user_msg)
        formatted_docs = ",".join(doc.page_content for doc in retrieved_docs)
        # print(formatted_docs)
        # print(retriever_context_docs)
        if st.session_state.use_korquad:
            return korquad_model.generate_answer(user_msg,retriever_context_docs[0].page_content, formatted_docs)
        else:
            prompt = hub.pull("rlm/rag-prompt")
            user_prompt = prompt.invoke({
                "context": retriever_context_docs[0].page_content + formatted_docs , 
                "question": user_msg
            })
            print(user_prompt)
            # llm = ChatOpenAI(model="gpt-4o-mini")
            return llm.invoke(user_prompt)
    else:
        # 컨텍스트 없이 직접 질문
        if st.session_state.use_korquad:
            return korquad_model.generate_answer(user_msg, "No context provided.")
        else:
            prompt = hub.pull("rlm/rag-prompt")
            user_prompt = prompt.invoke({
                "context": "No context provided.", 
                "question": user_msg
            })

            # llm = ChatOpenAI(model="gpt-4o-mini")
            return llm.invoke(user_prompt)
def handle_model_toggle():
    # 채팅 기록 초기화
    st.chat_message("assistant").empty()
    st.session_state.messages = []
    
def main():
    load_dotenv()
    init_session_state()
    
    st.set_page_config(page_title="메뉴얼", page_icon=":book:")
    st.title(":book: 메뉴얼 챗봇")
    st.caption("메뉴얼에 관한 질문을 입력해주세요.")

    # KorQuAD 모델 초기화
    korquad_model = init_korquad_model() if st.session_state.use_korquad else None

    # 사이드바 설정
    with st.sidebar:
        st.subheader("Your documents")
        
        if st.button("파인튜닝 시작", disabled=st.session_state.is_finetuning):
            st.session_state.is_finetuning = True
            simulate_finetuning()
            st.session_state.is_finetuning = False
        
        
        # 스위치 컴포넌트들
        st.subheader("설정")
        use_korquad = st.toggle(
            "Llama-3.2-1B-Instruct 모델 사용",
            value=st.session_state.use_korquad,
            on_change = handle_model_toggle,
            help="OpenAI 대신 KorQuAD 모델을 사용합니다."
        )
        if use_korquad != st.session_state.use_korquad:
            st.session_state.use_korquad = use_korquad
            st.session_state.processed = False  # 모델 변경 시 문서 재처리 필요
            st.rerun()
        
        context_enabled = st.toggle(
            "문서 컨텍스트 사용",
            value=st.session_state.enable_context,
            help="답변 생성 시 문서 컨텍스트를 참조합니다."
        )
        st.session_state.enable_context = context_enabled
        
        # 파일 업로더
        docs = st.file_uploader(
            "Upload your PDF here and click on 'Process'",
            accept_multiple_files=True
        )
        
        if st.button("Process") and docs:
            with st.spinner("Processing documents..."):
                raw_text = get_pdf_text(docs)
                text_chunks = get_chunks(raw_text)
                text_contexts = get_context(raw_text)
                st.session_state.retriever, st.session_state.retriever_context = get_vectorstore(
                    text_chunks,text_contexts,
                    use_korquad=st.session_state.use_korquad
                )
                st.session_state.processed = True
                st.success("Documents processed successfully!")

    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 사용자 입력 처리
    if user_question := st.chat_input("질문을 입력해주세요."):
        # 문서가 처리되었는지 확인
        if not st.session_state.processed and st.session_state.enable_context:
            st.error("Please upload and process documents first!")
            return

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        # AI 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = get_ai_response(
                    st.session_state.retriever,
                    st.session_state.retriever_context,
                    user_question,
                    korquad_model
                )
                content = response.content if hasattr(response, 'content') else response
                st.write(content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": content}
                )





if __name__ == "__main__":
    main()