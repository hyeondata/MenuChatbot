import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# TOKENIZERS_PARALLELISM 경고 제거
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class KorQuADLoRAInference:
    def __init__(self, base_model_path, adapter_path):
        """LoRA 모델과 토크나이저 초기화"""
        # 토크나이저 및 모델 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map="cpu",  # CPU 사용
            torch_dtype=torch.float32  # float32 사용
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map="cpu",  # CPU 사용
            torch_dtype=torch.float32  # float32 사용
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
            "do_sample": True,  # sampling 활성화
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
        ).to('cpu')  # CPU로 이동

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **self._get_gen_config())

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip().replace("### Answer:", ' ')