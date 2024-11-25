import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import json

# NLTK 데이터 다운로드
nltk.download('punkt')
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.tokenize import word_tokenize
import json

# NLTK 데이터 다운로드
nltk.download('punkt')

class QAEvaluator:
    def __init__(self):
        self.test_questions = [
            '지문 등록을 시작하려면 설정 앱에서 어떤 메뉴를 선택해야 하나요?',
            '갤럭시 S10e 모델에서는 지문을 등록할 때 어디에 손가락을 올려야 하나요?',
            '지문 등록 과정에서 손가락을 몇 번 올렸다 내렸다 해야 하나요?',
            '등록된 지문을 확인하려면 어떤 절차를 따라야 하나요?',
            '지문 등록이 완료되면 어떤 버튼을 눌러야 하나요?'
        ]
        
        self.reference_answers = [
            '설정 앱을 실행한 후, 생체 인식 및 보안 메뉴에서 지문을 선택해야 합니다.',
            '갤럭시 S10e 모델에서는 전원 버튼에 손가락을 올려야 합니다.',
            '지문이 등록될 때까지 손가락을 인식 센서에 반복적으로 올렸다 내렸다 해야 합니다.',
            '설정 앱의 생체 인식 및 보안 메뉴에서 지문을 선택하고, 설정한 방식으로 잠금을 해제한 후 등록된 지문 확인하기를 누르고 지문 인식 센서에 손가락을 올려야 합니다.',
            '지문 등록이 완료되면 완료 버튼을 눌러야 합니다.'
        ]

    def evaluate_with_bleu(self, generated_answers):
        """BLEU 점수를 계산합니다."""
        reference_tokens = [[word_tokenize(ans.lower())] for ans in self.reference_answers]
        candidate_tokens = [word_tokenize(ans.lower()) for ans in generated_answers]
        
        corpus_score = corpus_bleu(reference_tokens, candidate_tokens)
        individual_scores = [
            sentence_bleu([ref[0]], cand) 
            for ref, cand in zip(reference_tokens, candidate_tokens)
        ]
        
        return {
            'corpus_bleu': corpus_score,
            'individual_scores': individual_scores,
            'average_score': sum(individual_scores) / len(individual_scores)
        }

    def run_evaluation(self, generated_answers):
        """평가를 실행하고 결과를 출력합니다."""
        bleu_scores = self.evaluate_with_bleu(generated_answers)
        
        print("\nEvaluation Results:")
        print(f"Corpus BLEU Score: {bleu_scores['corpus_bleu']:.4f}")
        print(f"Average BLEU Score: {bleu_scores['average_score']:.4f}")
        
        print("\nDetailed Comparisons:")
        for i in range(len(self.test_questions)):
            print(f"\nQuestion {i+1}: {self.test_questions[i]}")
            print(f"Reference Answer: {self.reference_answers[i]}")
            print(f"Generated Answer: {generated_answers[i]}")
            print(f"Individual BLEU Score: {bleu_scores['individual_scores'][i]:.4f}")
        
        return bleu_scores

# 테스트를 위한 예시 생성 답변 (실제 모델의 출력을 시뮬레이션)
generated_answers = [
    '설정 앱을 실행하고 생체 인식 및 보안 메뉴에서 지문을 선택합니다.',
    '갤럭시 S10e에서는 전원 버튼에 손가락을 올려야 합니다.',
    '지문이 등록될 때까지 손가락을 센서에 반복해서 올렸다 내려야 합니다.',
    '설정 앱의 생체 인식 및 보안에서 지문을 선택하고 잠금 해제 후 등록된 지문 확인하기를 누르고 센서에 손가락을 올립니다.',
    '지문 등록이 완료되면 완료 버튼을 누릅니다.'
]

# 평가 실행
if __name__ == "__main__":
    evaluator = QAEvaluator()
    results = evaluator.run_evaluation(generated_answers)