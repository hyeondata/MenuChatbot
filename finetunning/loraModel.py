import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class KorQuADLoRAInference:
    def __init__(self, base_model_path, adapter_path):
        """LoRA 모델과 토크나이저 초기화"""
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map='auto',
            torch_dtype=torch.float16
        )
        self._configure_tokenizer()
        self.model.eval()

    def _configure_tokenizer(self):
        """토크나이저의 패딩 토큰 설정"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|end_of_text|>'
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids('<|end_of_text|>')

    def format_prompt(self, question):
        """프롬프트 포맷팅"""
        return f"""You are a helpful assistant that provides direct and concise answers.

IMPORTANT INSTRUCTIONS:
1. Provide ONLY the direct answer
2. DO NOT include any tags like 'Explanation:', 'Answer:', 'Note:'
3. DO NOT include any additional context or explanations
4. Respond in clear, complete sentences
5. Please answer in Korean
6. Keep the response simple and to the point
### Question: 
{question}

### Answer:"""

    def _get_gen_config(self):
        """텍스트 생성 구성 반환"""
        return {
            "do_sample": False,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 50,
            "max_length": 140,
            "min_length": 5,
            "no_repeat_ngram_size": 10,
            "num_beams": 10,
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.convert_tokens_to_ids('<|end_of_text|>'),
            "use_cache": False
        }

    def generate_answer(self, question):
        """질문에 대한 답변 생성"""
        prompt = self.format_prompt(question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self._get_gen_config())

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip().replace("### Answer:", ' ')


# 단일 예제 테스트 함수
def test_single_lora_example():
    BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    ADAPTER_MODEL = r"./loraData/Llama-3.2-1B-Instruct"
    
    inferencer = KorQuADLoRAInference(BASE_MODEL, ADAPTER_MODEL)
    
    question = "지문 인식률을 높이기 위해 어떤 손의 지문을 등록하는 것이 좋습니까?"  # 테스트 질문
    try:
        answer = inferencer.generate_answer(question)
        print(f"\n질문: {question}")
        print(f"답변: {answer}")
    except Exception as e:
        print(f"오류 발생: {e}")

# 테스트 실행

# def model_init(BASE_MODEL, ADAPTER_MODEL):
#     inferencer = KorQuADLoRAInference(BASE_MODEL, ADAPTER_MODEL)
#     return inferencer
    
# def get_answer(inferencer, question):
#     try:
#         print(f"질문: {question}")
#         answer = inferencer.generate_answer(question)
#         return answer
#     except Exception as e:
#         return f"오류 발생: {e}"


# test_single_lora_example()