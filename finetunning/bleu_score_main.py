from mac_model_inference import KorQuADLoRAInference
from bleuScore import QAEvaluator
import json

def run_model_evaluation():
    """LoRA 모델 평가 실행"""
    BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
    ADAPTER_MODEL = r"../loraData/Llama-3.2-1B-Instruct"
    
    print("모델 로딩 중...")
    inferencer = KorQuADLoRAInference(BASE_MODEL, ADAPTER_MODEL)
    evaluator = QAEvaluator()
    
    print("\n답변 생성 중...")
    generated_answers = []
    for question in evaluator.test_questions:
        try:
            answer = inferencer.generate_answer(question)
            generated_answers.append(answer)
            print(f"질문: {question}")
            print(f"생성된 답변: {answer}\n")
        except Exception as e:
            print(f"답변 생성 중 오류 발생: {e}")
            generated_answers.append("")
    
    print("\nBLEU 평가 실행 중...")
    results = evaluator.run_evaluation(generated_answers)
    
    output_file = 'lora_evaluation_results.json'
    full_results = {
        'bleu_scores': results,
        'generated_answers': generated_answers,
        'reference_answers': evaluator.reference_answers,
        'test_questions': evaluator.test_questions
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    
    return results, generated_answers

def main():
    try:
        results, generated_answers = run_model_evaluation()
        
        print("\n=== 평가 요약 ===")
        print(f"전체 BLEU 점수: {results['corpus_bleu']:.4f}")
        print(f"평균 BLEU 점수: {results['average_score']:.4f}")
        
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")

if __name__ == "__main__":
    main()