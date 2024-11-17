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
import os
from pathlib import Path
import finetunning.finetunning as finetunning
import finetunning.loraModel as loraModel
import threading
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
llm = ChatOpenAI(model="gpt-4o")


# loraModel에서 제공하는 함수들
@st.cache_resource
def model_init(BASE_MODEL, ADAPTER_MODEL):
    st.cache_resource.clear()  # 캐시 초기화
    torch.cuda.empty_cache()  # GPU 캐시 해제
    time.sleep(2)
    inferencer = loraModel.KorQuADLoRAInference(BASE_MODEL, ADAPTER_MODEL)
    return inferencer

def get_answer(inferencer, question):
    try:
        print(f"질문: {question}")
        answer = inferencer.generate_answer(question)
        return answer
    except Exception as e:
        return f"오류 발생: {e}"


def get_subfolders(base_path):
    """지정된 경로의 모든 하위 폴더를 반환합니다."""
    try:
        return [f.name for f in Path(base_path).iterdir() if f.is_dir()]
    except Exception as e:
        st.error(f"모델 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")
        return []

def simulate_finetuning(target_ratio=100, batch_size=4):
    progress_container = st.empty()
    with progress_container.container():
        st.warning("⚠️ 파인튜닝이 진행 중입니다. 완료될 때까지 기다려주세요.")
        st.cache_resource.clear()  # 캐시 초기화
        torch.cuda.empty_cache()  # GPU 캐시 해제
        time.sleep(2)
        # 진행 상황을 표시할 지표들
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_container = st.empty()
        pipeline = finetunning.pipline_get()
        
        pipeline_thread = threading.Thread(target=finetunning.run_pipeline_in_background, args=(pipeline,))
        pipeline_thread.start()

        # 실시간 진행률 확인
        while pipeline_thread.is_alive():
                # 진행률 업데이트 확인
                for progress in pipeline.get_training_progress():
                    print(f"훈련 진행률: {progress:.2f}%")
                    
                    progress_bar.progress(round(progress))
            
                    # 현재 학습 상태 표시
                    with metrics_container.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("현재 진행률", f"{progress:.2f}%")
                        with col2:
                            st.metric("배치 크기", f"{batch_size}")
                    
                    status_text.text(f"파인튜닝 진행중... {progress:.2f}%")
                    time.sleep(1)
                time.sleep(1)  # 진행률 업데이트 간격 조절
                
        pipeline_thread.join()  # 스레드 완료 대기
        
        status_text.text("파인튜닝이 완료되었습니다!")
        time.sleep(2)
        
        st.rerun()  # 페이지 새로고침
    
    progress_container.empty()


class LlamaInference:
    def __init__(self, model_path):
        """모델과 토크나이저 초기화"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        # model_id = 'google/gemma-2-2b-it' #9b 모델 사용시 비 활성화
        model_id = model_path #9b 모델 사용시 활성화     
        # 4-bit 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            quantization_config=quantization_config, #9b 모델을 양자화 해서 사용시 활성화
            trust_remote_code=True
        )

        # HuggingFace 파이프라인 먼저 생성
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03,
        )
        
        # LangChain 파이프라인 생성
        self.model = HuggingFacePipeline(pipeline=pipe)

    def format_prompt(self, question, context,sentences):
        return f"""<bos><start_of_turn>user
Context: {context}

Question: {question}

Sentences: {sentences}

Please answer based on the given context. If you can't find the answer, say "I don't know".
**Please answer in Korean**
<end_of_turn>
<start_of_turn>model
"""

    def generate_answer(self, question, context, sentences):
        prompt = self.format_prompt(question, context, sentences)
        response = self.model.invoke(prompt)
            # 응답에서 <start_of_turn>model 이후의 텍스트만 추출
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[1].strip()
        return response

@st.cache_resource
def init_gemma_model():
    # st.cache_resource.clear()  # 캐시 초기화
    torch.cuda.empty_cache()  # GPU 캐시 해제
    time.sleep(1)
    model_path = "google/gemma-2-9b-it"
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
    if 'finetuning_ratio' not in st.session_state:
        st.session_state.finetuning_ratio = 100
    if 'inferencer' not in st.session_state:
        st.session_state.inferencer = None

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
        
        formatted_docs = ",".join(doc.page_content for doc in retrieved_docs)
        
        if st.session_state.use_korquad:
            return korquad_model.generate_answer(user_msg,retriever_context_docs[0].page_content, formatted_docs)
        else:
            prompt = hub.pull("rlm/rag-prompt")
            user_prompt = prompt.invoke({
                "context": retriever_context_docs[0].page_content + formatted_docs , 
                "question": user_msg
            })
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
    st.session_state.inferencer = None
    st.chat_message("assistant").empty()
    st.session_state.messages = []

    
def main():
    load_dotenv()
    init_session_state()
    
    st.set_page_config(page_title="메뉴얼", page_icon=":book:")
    st.title(":book: 메뉴얼 챗봇")
    st.caption("메뉴얼에 관한 질문을 입력해주세요.")

    # KorQuAD 모델 초기화
    korquad_model = init_gemma_model() if st.session_state.use_korquad else None
    if (st.session_state.use_korquad):
        st.session_state.current_model_name = "Gemma-2-9b-it"
    elif st.session_state.inferencer:
        st.session_state.current_model_name = "Llama-3.2-1B-Instruct"
    else :
        st.session_state.current_model_name = "OpenAI GPT-4o"

    # 사이드바 설정
    with st.sidebar:
        
        st.subheader("Your documents")
        
        st.subheader("모델 선택")
        
        base_path = r"C:\Users\codeKim\Desktop\gemma2\loraData"  # 실제 사용할 경로로 변경
        
        folders = get_subfolders(base_path)
        
        # 선택박스와 적용 버튼을 같은 줄에 배치
        col1, col2 = st.columns([4, 1])
        
        with col1:
            selected_folder = st.selectbox(
                "모델 선택",
                options=folders,
                key="folder_select",
                label_visibility="collapsed"  # 라벨 숨기기

            )
        
        with col2:
            apply_button = st.button("적용", key="apply_folder")

        # 선택된 폴더를 loraModel 라이브러리에 넘기기
        if apply_button and selected_folder:
            selected_path = os.path.join(base_path, selected_folder)
            BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
            ADAPTER_MODEL = selected_path

            try:
                st.session_state.messages =[]
                st.session_state.inferencer = model_init(BASE_MODEL, ADAPTER_MODEL)
                st.session_state.current_model_name = selected_folder
                success_message = st.empty()
                success_message.success("모델이 성공적으로 로드되었습니다.")
                time.sleep(1)
                success_message.empty()
            except Exception as e:
                st.error(f"모델을 로드하는 중 오류가 발생했습니다: {str(e)}")

        # 현재 사용하는 모델 이름 표시
        if 'current_model_name' in st.session_state:
            st.write(f"**현재 선택된 모델:** {st.session_state.current_model_name}")
        else:
            st.write("현재 선택된 모델이 없습니다.")
        
        # if apply_button and selected_folder:
            # st.success(f"선택한 모델: {selected_folder}")
            # selected_path = os.path.join(base_path, selected_folder)
            # st.session_state.current_folder = selected_path
        # st.subheader("Your documents")
        
        finetuning_ratio = st.slider(
            "파인튜닝 비율 (%)",
            min_value=20,
            max_value=100,
            value=st.session_state.finetuning_ratio,
            step=20,
            help="모델 파인튜닝의 목표 비율을 설정합니다."
        )
        st.session_state.finetuning_ratio = finetuning_ratio
        
        if st.button("파인튜닝 시작", disabled=st.session_state.is_finetuning):
            st.session_state.is_finetuning = True
            simulate_finetuning(
                target_ratio=st.session_state.finetuning_ratio,
                # batch_size=st.session_state.batch_size
            )
            st.session_state.is_finetuning = False
        
        # if st.button("파인튜닝 시작", disabled=st.session_state.is_finetuning):
        #     st.session_state.is_finetuning = True
        #     simulate_finetuning()
        #     st.session_state.is_finetuning = False
        
        
        # 스위치 컴포넌트들
        st.subheader("설정")
        use_korquad = st.toggle(
            "Gemma-2-9b-it 모델 사용",
            value=st.session_state.use_korquad,
            on_change = handle_model_toggle,
            help="OpenAI 대신 Gemma-2-9b-it 모델을 사용합니다."
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
                st.session_state.messages = []
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


        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)


        if st.session_state.inferencer:
            with st.chat_message("assistant"):
               with st.spinner("LoRA 모델 응답 생성 중..."):
                    answer = get_answer(st.session_state.inferencer, user_question)
                    st.write(answer)
        else:
        # AI 응답 생성
                    # 문서가 처리되었는지 확인
            if not st.session_state.processed and st.session_state.enable_context:
                st.error("Please upload and process documents first!")
                return
        
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