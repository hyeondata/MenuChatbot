import os
import re
import json
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from transformers import TrainerCallback
from queue import Queue
import threading
import time

class ProgressCallback(TrainerCallback):
    def __init__(self, progress_queue):
        self.progress_queue = progress_queue

    def on_log(self, args, state, control, logs=None, **kwargs):
        """훈련 중 진행률을 큐로 전달하여 상위 레벨에서 접근 가능하게 함"""
        if logs is None:
            logs = {}

        current_step = state.global_step
        total_steps = state.max_steps
        progress_percentage = (current_step / total_steps) * 100
        self.progress_queue.put(progress_percentage)  # 큐에 진행률 추가


class QA_FineTuningPipeline:
    def __init__(self, model_path, pdf_paths, output_dir="results", lora_output_dir=r"./loraData"):
        load_dotenv()
        self.pdf_paths = pdf_paths
        self.model_path = model_path
        self.output_dir = output_dir
        self.lora_output_dir = lora_output_dir
        self.tokenizer = None
        self.model = None
        self.qa_chain = self.initialize_qa_chain()
        self.progress_queue = Queue()  # 진행률 큐 생성
    
    def initialize_qa_chain(self):
        # Load model and create QA prompt template for question generation
        qa_generation_template = """Based on the context below, generate 5 question-answer pairs. 
        All questions and answers must be written in Korean language.

        Context:
        {context}

        Please follow this output format:
        Q1: [질문1 - Korean question]
        A1: [답변1 - Korean answer]

        ...and so on

        Generation Rules:
        1. Questions must be clear and specific
        2. Answers must be strictly based on the given context only
        3. Include various levels of questions from simple fact-checking to in-depth analysis
        4. All answers should be in complete sentences
        5. Each QA pair should cover different aspects of the context
        6. ALL questions and answers MUST be written in Korean language
        7. Maintain formal/polite Korean language level (합쇼체 or 해요체)
        8. Ensure natural Korean expression rather than direct translation
        9. Use appropriate Korean particles and connectors for smooth flow
        10. Include relevant Korean-specific context when appropriate

        Generate the QA pairs following the above rules."""
        
        qa_prompt = PromptTemplate(input_variables=["context"], template=qa_generation_template)
        llm = ChatOpenAI(model='gpt-4o')
        return LLMChain(llm=llm, prompt=qa_prompt)

    def extract_text_from_pdf(self):
        """Extract text from PDF files."""
        text = ""
        for pdf_path in self.pdf_paths:
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def split_text_to_contexts(self, raw_text):
        """Split raw text into chunks for context."""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n\n\n", "\n\n\n\n","\n"],
            chunk_size=500,
            chunk_overlap=100
        )
        return text_splitter.split_text(raw_text)

    def generate_qa_pairs(self, context_text):
        """Generate QA pairs from a context using LLMChain."""
        try:
            return self.qa_chain.run(context=context_text)
        except Exception as e:
            print(f"Error generating QA pairs: {str(e)}")
            return ""

    def parse_qa_response(self, response_text):
        """Parse generated QA pairs into question and answer lists."""
        questions = re.findall(r'Q\d+:\s*(.*?)\s*(?=A\d+:|$)', response_text, re.DOTALL)
        answers = re.findall(r'A\d+:\s*(.*?)\s*(?=Q\d+:|$)', response_text, re.DOTALL)
        return [q.strip() for q in questions], [a.strip() for a in answers]

    def save_qa_dataset(self, questions, answers, base_filename="qa_dataset"):
        """Save parsed QA pairs to a JSON file."""
        json_data = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
        with open(f'{base_filename}.json', 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    def format_qa_dataset(self, json_file_path="qa_dataset.json"):
        """Load and format QA dataset for fine-tuning."""
        print("fileload")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        formatted_texts = []
        korQuAD_prompt = "### Question:\n{}\n\n### Answer:\n{}\n<|end_of_text|>"
        for item in qa_data:
            formatted_texts.append(korQuAD_prompt.format(item["question"], item["answer"]))
        return Dataset.from_dict({"text": formatted_texts})

    def initialize_model(self):
        """Load and configure model with LoRA for fine-tuning."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = '<|end_of_text|>'

        # Load and prepare model for quantized fine-tuning
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

    def fine_tune_model(self, formatted_dataset):
        """Fine-tune the model with the formatted dataset."""
        training_params = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=15,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            optim="paged_adamw_32bit",
            save_steps=10000,
            logging_steps=5,
            learning_rate=2e-4,
            weight_decay=0.001,
            fp16=True,
            bf16=False,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant",
            report_to="none"
        )

        # LoRA 설정을 위한 peft_config 생성
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # ProgressCallback 인스턴스에 큐 전달
        progress_callback = ProgressCallback(self.progress_queue)
        
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=formatted_dataset,
            tokenizer=self.tokenizer,
            args=training_params,
            peft_config=peft_config,  # peft_config 인자로 LoRA 설정 전달
            dataset_text_field="text",  # dataset_text_field 직접 전달
            max_seq_length=None,  # max_seq_length 직접 전달
            packing=False,
            callbacks=[progress_callback]
        )
        
        trainer.train()

        # Save the trained LoRA adapter
        self.model.save_pretrained(self.lora_output_dir+"/"+self.model_path.split("/")[-1])
    def get_training_progress(self):
        """진행률을 큐에서 꺼내 반환"""
        while not self.progress_queue.empty():
            progress = self.progress_queue.get()
            yield progress  # 진행률 반환

    def run_pipeline(self):
        """Run the entire pipeline for PDF processing, QA generation, and model fine-tuning."""
        # Step 1: Extract and split text
        raw_text = self.extract_text_from_pdf()
        text_contexts = self.split_text_to_contexts(raw_text)
        print(text_contexts[55])
        # Step 2: Generate QA pairs
        qa_pairs = self.generate_qa_pairs(text_contexts[55])  # Example with first context
        questions, answers = self.parse_qa_response(qa_pairs)

        # Step 3: Save QA dataset
        self.save_qa_dataset(questions, answers)

        # Step 4: Format QA dataset for training
        formatted_dataset = self.format_qa_dataset()

        # Step 5: Initialize and fine-tune model
        self.initialize_model()
        self.fine_tune_model(formatted_dataset)

# Example usage
def pipline_get():
    pipeline = QA_FineTuningPipeline(
        model_path="meta-llama/Llama-3.2-1B-Instruct",
        pdf_paths=[r"./test.pdf"]
    )
    return pipeline

# 파이프라인을 백그라운드에서 실행하는 함수
def run_pipeline_in_background(pipeline):
    pipeline.run_pipeline()
