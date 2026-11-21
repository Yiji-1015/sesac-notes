# LLM으로 답 채점하기(LLM-as-a-Judge) 전체 코드 (Anthropic Python SDK, Messages API). 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 실행: 환경 변수 ANTHROPIC_API_KEY를 정한 뒤 python judge.py  (실제로 호출하면 비용이 들어요)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from typing import Literal                                          #@1
from pydantic import BaseModel, ConfigDict, Field                   #@1
import anthropic                                                    #@2
from sklearn.metrics import cohen_kappa_score                       #@5

# ① 채점 기준과 점수 계약
RUBRIC = """너는 학습 노트의 질문에 대한 답을 채점하는 평가자야.
[기준 답]과 비교해서 [채점할 답]에 1~5점을 줘.
5: 기준 답의 핵심을 모두 맞게 담음
4: 핵심은 맞지만 하나가 빠졌거나 덜 정확함
3: 절반쯤 맞고, 중요한 것이 빠짐
2: 대부분 빠졌거나 일부가 틀림
1: 틀렸거나 질문과 상관없음
오류 종류는 하나만 골라: 없음, 빠짐, 틀림, 질문과 무관
근거를 먼저 쓰고, 그 근거로 점수를 정해.
답이 길거나 자신 있게 쓰였다고 점수를 올리지 마. 기준 답에 없는 내용은 맞더라도 점수에 넣지 마."""

class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")                       # 계약에 없는 필드는 받지 않기
    rationale: str = Field(description="기준 답과 무엇이 같고 무엇이 다른지")
    error_type: Literal["없음", "빠짐", "틀림", "질문과 무관"]
    score: Literal[1, 2, 3, 4, 5]

ITEMS = [
    {"question": "교차검증은 왜 하나요?",
     "reference": "valid 한 번의 점수는 우연에 크게 흔들려서, 데이터를 여러 덩어리로 나눠 번갈아 재고 평균과 흔들림을 함께 봐요.",
     "answer": "valid 한 번으로는 점수가 많이 흔들려서, 다섯 덩어리로 나눠 다섯 번 재고 평균을 봐요."},
    {"question": "AdaBoost가 라벨 잡음에 약한 이유는?",
     "reference": "지수 손실을 줄여서, 크게 틀린 행의 무게가 지수적으로 커져요. 라벨이 잘못된 행에 무게를 몰아줘요.",
     "answer": "트리를 너무 많이 쌓아서 과적합하기 때문이에요."},
    {"question": "프롬프트 캐시는 무엇을 아끼나요?",
     "reference": "요청 앞부분이 이전 요청과 같으면, 그 부분의 입력 토큰을 싸게 읽어서 비용과 첫 토큰까지의 시간을 줄여요.",
     "answer": "프롬프트 캐시는 아주 중요한 기능으로, 모델이 만든 답을 저장해 두었다가 같은 질문이 오면 다시 보내 줘서 출력 토큰을 크게 아껴 줘요. 실무에서 꼭 써야 해요."},
]
schema = Verdict.model_json_schema()                                # 점수 계약을 JSON Schema로

# ② 요청 만들기: 고정 부분은 앞에, 바뀌는 부분은 뒤에
client = anthropic.Anthropic(max_retries=2)                         # 키는 환경 변수 ANTHROPIC_API_KEY에서 읽어요
MODEL = "claude-opus-5-5"

def judge_request(item, effort="low", max_tokens=2048):
    return dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral"}}],   # 모든 요청에 같은 채점 기준: 여기까지 캐시
        messages=[{"role": "user", "content": f"[질문]\n{item['question']}\n[기준 답]\n{item['reference']}\n[채점할 답]\n{item['answer']}"}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},   # 생각 깊이와 답의 모양
    )

def text_of(response):                                              # 생각 블록은 건너뛰고 글 블록만
    return "".join(b.text for b in response.content if b.type == "text")

first = client.messages.create(**judge_request(ITEMS[0]))
verdict = Verdict.model_validate_json(text_of(first))

# ③ 캐시로 비용 줄이기: 몇 건부터 이득인가
PRICE_IN, PRICE_OUT = 4.00, 20.00                                   # claude-opus-5-5, 100만 토큰당 달러
WRITE_5M, READ = 1.25, 0.05                                         # 5분 캐시 쓰기는 입력 단가의 1.25배, 읽기는 0.05배 (Opus 5.5)

def batch_cost(n, fixed, varying, out, cached):                     # n건 채점의 달러 비용
    if not cached:
        inputs = n * (fixed + varying)
    else:
        inputs = fixed * WRITE_5M + (n - 1) * fixed * READ + n * varying   # 첫 건은 쓰기, 나머지는 읽기
    return (inputs * PRICE_IN + n * out * PRICE_OUT) / 1_000_000

FIXED, VARYING, OUT = 3000, 300, 400                                # 가정: 채점 기준 3,000토큰, 문항 300토큰, 출력(생각 포함) 400토큰
costs = {n: (batch_cost(n, FIXED, VARYING, OUT, False), batch_cost(n, FIXED, VARYING, OUT, True)) for n in (1, 2, 10, 1000)}

# ④ 응답 읽기: 끝까지 못 쓴 답, 거절, 쓴 토큰 모으기
def judge(item):
    used, tries = [], []
    for max_tokens in (2048, 8192):                                 # 잘리면 한 번만 더 크게
        r = client.messages.create(**judge_request(item, max_tokens=max_tokens))
        used.append(r.usage)
        tries.append(r.stop_reason)
        if r.stop_reason == "refusal":                              # 안전 판단으로 거절: 점수 없음으로 남기기
            return None, used, tries
        if r.stop_reason == "max_tokens":                           # 생각하다가 한도에 닿아 JSON이 잘림
            continue
        return Verdict.model_validate_json(text_of(r)), used, tries
    return None, used, tries

def total_usage(used):                                              # 캐시 쓰기, 읽기까지 따로 더하기
    keys = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
    return {k: sum(getattr(u, k) or 0 for u in used) for k in keys}

results = [judge(it) for it in ITEMS]

# ⑤ 채점을 믿을 수 있나: 순서 바꿔 묻기, 사람 점수와 맞춰 보기
class Pick(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rationale: str
    winner: Literal["A", "B", "같음"]

def pick(question, reference, a, b):
    r = client.messages.create(model=MODEL, max_tokens=2048,
        system=[{"type": "text", "text": "두 답 중 [기준 답]에 더 가까운 쪽을 골라. 길이와 말투는 보지 마.", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": f"[질문]\n{question}\n[기준 답]\n{reference}\n[A]\n{a}\n[B]\n{b}"}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": Pick.model_json_schema()}})
    return Pick.model_validate_json(text_of(r)).winner

def pairwise(question, reference, x, y):                            # x를 A에 한 번, B에 한 번
    one = pick(question, reference, x, y)
    two = {"A": "B", "B": "A", "같음": "같음"}[pick(question, reference, y, x)]   # 두 번째 답을 첫 번째 기준으로 되돌리기
    return one, two, one == two

ITEM = ITEMS[1]
short_right = ITEM["reference"]
long_wrong = ITEMS[1]["answer"] + " 트리 수를 늘리면 train 오차가 0이 되고, 이것은 부스팅에서 아주 잘 알려진 현상이에요. 그래서 learning rate를 작게 하고 조기 종료를 꼭 써야 해요."
order_check = pairwise(ITEM["question"], ITEM["reference"], short_right, long_wrong)

human = [5, 4, 2, 1, 3, 5, 2, 4, 3, 1]                              # 가상의 사람 점수 10건
judged = [5, 5, 3, 1, 4, 5, 2, 5, 3, 2]                             # 같은 10건의 Judge 점수
exact = sum(h == j for h, j in zip(human, judged)) / len(human)
within1 = sum(abs(h - j) <= 1 for h, j in zip(human, judged)) / len(human)
kappa = cohen_kappa_score(human, judged, weights="quadratic")       # 우연히 맞을 몫을 뺀 일치도
lenient = sum(judged) / len(judged) - sum(human) / len(human)       # Judge가 평균 몇 점 후한가
