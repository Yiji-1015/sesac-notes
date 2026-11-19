# LLM 원리 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 모델: intfloat/multilingual-e5-small (Transformer 인코더 12층, 이 PC에 받아 둔 것)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import torch                                                     #@1
from transformers import AutoTokenizer                           #@1
from transformers import AutoModel                               #@2

MODEL = "intfloat/multilingual-e5-small"                         #@1

# ① 글을 토큰으로
tok = AutoTokenizer.from_pretrained(MODEL)
text = "서버를 실행하려면 어디서 해야 하나요?"
enc = tok(text, return_tensors="pt")
ids = enc["input_ids"][0]
tokens = tok.convert_ids_to_tokens(ids)

# ② 토큰을 벡터로 (Embedding)
model = AutoModel.from_pretrained(MODEL, attn_implementation="eager")
model.eval()
table = model.embeddings.word_embeddings          # 토큰 번호 → 벡터 표 (어휘 수 × 384)
vectors = table(ids)                              # (토큰 수, 384)

# ③ Attention을 손으로 계산하기
torch.manual_seed(0)
x = vectors[1:5].detach()                         # 앞 네 토큰만
d = 8
Wq, Wk, Wv = (torch.randn(384, d) for _ in range(3))  # 실제 모델에서는 학습으로 정해지는 행렬
Q, K, V = x @ Wq, x @ Wk, x @ Wv                  # 질문(Query), 열쇠(Key), 내용(Value)
scores = Q @ K.T / d ** 0.5                       # 토큰끼리의 관련 점수
weights = torch.softmax(scores, dim=-1)           # 줄마다 합이 1
mixed = weights @ V                               # 관련 있는 토큰의 내용을 섞은 새 표현

# ④ 뒤를 못 보게 막기 (Mask)
causal = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)  # 대각선 위 = 미래
weights_causal = torch.softmax(scores.masked_fill(causal, float("-inf")), dim=-1)

# ⑤ 실제 모델의 Attention 보기
with torch.no_grad():
    out = model(**enc, output_attentions=True)
attn = torch.stack(out.attentions)               # (층 12, 배치 1, head 12, 토큰, 토큰)

def looks_at(layer):
    a = attn[layer, 0].mean(0).clone()           # 그 층, head 12개 평균
    sink = a[:, 0].mean().item()                 # 시작 표시 <s>로 가는 몫
    a[:, 0] = a[:, -1] = 0                       # 표시용 토큰은 빼고
    return sink, [tokens[j] for j in a.argmax(-1)]  # 토큰마다 가장 많이 본 토큰

first_layer, last_layer = looks_at(0), looks_at(11)

# ⑥ 다음 토큰 고르기 (점수 → 확률)
candidates = ["서버", "폴더", "터미널", "브라우저"]
logits = torch.tensor([3.0, 2.2, 1.5, 0.3])      # 예시 점수: 모델이 다음 토큰 후보에 준 값이라고 가정
probs = {t: torch.softmax(logits / t, dim=-1) for t in (0.5, 1.0, 2.0)}  # temperature
greedy = candidates[int(logits.argmax())]
torch.manual_seed(0)
sampled = [candidates[int(torch.multinomial(probs[1.0], 1))] for _ in range(10)]
