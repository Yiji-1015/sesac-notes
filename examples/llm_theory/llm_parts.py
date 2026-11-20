# Transformer 구조 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 모델: intfloat/multilingual-e5-small (BERT 구조의 Transformer 인코더 12층, 이 PC에 받아 둔 것)
# 모델 안의 계산을 실제 가중치로 한 단계씩 다시 해 보고, 모델이 낸 값과 같은지 확인해요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import math                                                      #@3
import torch                                                     #@1
import torch.nn.functional as F                                  #@2
from transformers import AutoTokenizer                           #@1
from transformers import AutoModel                               #@2

MODEL = "intfloat/multilingual-e5-small"                         #@1

# ① 글을 토큰으로
tok = AutoTokenizer.from_pretrained(MODEL)
texts = ["배가 아파서 병원에 가요.", "배를 타고 섬에 가요.", "배를 깎아 먹었어요."]  # 같은 '배', 세 가지 뜻
enc = tok(texts, return_tensors="pt", padding=True)              # 짧은 문장 뒤는 채워서 한 묶음으로
ids, mask = enc["input_ids"], enc["attention_mask"]
tokens = [tok.convert_ids_to_tokens(row[m.bool()]) for row, m in zip(ids, mask)]
BAE = 1                                                          # 세 문장 모두 1번 자리가 '▁배'
PAIRS = [(0, 1), (0, 2), (1, 2)]                                 # 몸-탈것, 몸-과일, 탈것-과일

# ② 토큰 벡터에 위치 더하기
model = AutoModel.from_pretrained(MODEL, attn_implementation="eager").eval()
emb = model.embeddings
pos = torch.arange(ids.shape[1])                                 # 자리 번호 0, 1, 2, …
with torch.no_grad():
    word = emb.word_embeddings(ids)                              # 토큰 번호 → 벡터 (표에서 찾기)
    x0 = emb.LayerNorm(word + emb.position_embeddings(pos) + emb.token_type_embeddings(torch.zeros_like(ids)))
    same_embed = torch.allclose(x0, emb(ids), atol=1e-5)         # 모델 안의 계산과 같은가

def cos(a, b):                                                   # 두 벡터의 방향이 얼마나 같은가 (1이면 같음)
    return F.cosine_similarity(a, b, dim=-1).item()

bae_table = [cos(word[a, BAE], word[b, BAE]) for a, b in PAIRS]  # 표에서 찾은 '배' 벡터끼리

# ③ head 하나의 Attention
layer0 = model.encoder.layer[0]
H, Dh = model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads  # 12개, 32칸

def split(t):                                                    # (문장, 토큰, 384) → (문장, head, 토큰, 32)
    return t.view(t.shape[0], t.shape[1], H, Dh).transpose(1, 2)

pad = (1 - mask[:, None, None, :]) * -1e9                        # 채운 자리는 점수를 아주 낮게 → 보지 않음
with torch.no_grad():
    sa = layer0.attention.self
    Q, K, V = split(sa.query(x0)), split(sa.key(x0)), split(sa.value(x0))
    scores = Q @ K.transpose(-1, -2) / math.sqrt(Dh) + pad       # 모든 토큰 쌍의 맞음 점수
    weights = scores.softmax(-1)                                 # (문장, head, 토큰, 토큰), 줄마다 합 1
    out = model(**enc, output_attentions=True, output_hidden_states=True)
    same_attn = torch.allclose(weights, out.attentions[0], atol=1e-5)

# ④ 여러 head를 이어 붙이기
with torch.no_grad():
    heads = weights @ V                                          # head마다 섞은 내용 (문장, head, 토큰, 32)
    concat = heads.transpose(1, 2).reshape(x0.shape)             # 12개를 이어 붙여 다시 384칸
    attn_out = layer0.attention.output.dense(concat)             # 이어 붙인 것을 한 번 더 섞기
n0 = len(tokens[0])
looks = [tokens[0][j] for j in weights[0, :, BAE, :n0].argmax(-1)]  # head마다 문장 1의 '▁배'가 가장 많이 본 토큰

# ⑤ 층 하나 완성: 잔차 연결, LayerNorm, 작은 신경망
def encoder_layer(x, layer, block):                              # block: 보면 안 되는 자리에 더할 점수
    sa = layer.attention.self
    Q, K, V = split(sa.query(x)), split(sa.key(x)), split(sa.value(x))
    w = (Q @ K.transpose(-1, -2) / math.sqrt(Dh) + block).softmax(-1)
    a = layer.attention.output.dense((w @ V).transpose(1, 2).reshape(x.shape))
    h = layer.attention.output.LayerNorm(x + a)                  # 잔차 연결(입력을 그대로 더하기) + LayerNorm
    f = layer.output.dense(F.gelu(layer.intermediate.dense(h)))  # 토큰마다 384 → 1536 → 384
    return layer.output.LayerNorm(h + f)                         # 다시 잔차 연결 + LayerNorm

with torch.no_grad():
    x1 = encoder_layer(x0, layer0, pad)
    same_layer = torch.allclose(x1, out.hidden_states[1], atol=1e-4)

# ⑥ 12층 쌓기: 문맥이 섞인 벡터
with torch.no_grad():
    h = x0
    bae_by_layer = [[cos(h[a, BAE], h[b, BAE]) for a, b in PAIRS]]  # 0번은 ②의 벡터
    for layer in model.encoder.layer:
        h = encoder_layer(h, layer, pad)
        bae_by_layer.append([cos(h[a, BAE], h[b, BAE]) for a, b in PAIRS])
    same_model = torch.allclose(h, out.last_hidden_state, atol=1e-3)

# ⑦ 미래 가리기: 디코더는 뒤를 못 봐요
future = torch.triu(torch.ones(len(pos), len(pos), dtype=torch.bool), diagonal=1)  # 대각선 위 = 자기보다 뒤
with torch.no_grad():
    hc = x0
    for layer in model.encoder.layer:
        hc = encoder_layer(hc, layer, pad + future * -1e9)       # 같은 층에 가리기만 더하기
bae_causal = [cos(hc[a, BAE], hc[b, BAE]) for a, b in PAIRS]

# ⑧ 다음 토큰 고르기 (점수 → 확률)
candidates = ["섬", "바다", "병원", "학교"]                      # "배를 타고 ___에 가요"의 빈칸 후보
logits = torch.tensor([3.0, 2.2, 1.5, 0.3])                      # 예시 점수: 디코더가 후보에 준 값이라고 가정
probs = {t: torch.softmax(logits / t, dim=-1) for t in (0.5, 1.0, 2.0)}  # temperature
greedy = candidates[int(logits.argmax())]
torch.manual_seed(0)
sampled = [candidates[int(torch.multinomial(probs[1.0], 1))] for _ in range(10)]
