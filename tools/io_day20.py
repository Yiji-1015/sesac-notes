"""Day20 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의(Day20_20261113.pdf)의 코드 — Tokenizer(bert-base-multilingual-cased) · padding/truncation ·
cosine_similarity · SentenceTransformer(all-MiniLM-L6-v2) · TransformerEncoderLayer · 3×3 Attention ·
Softmax/Temperature — 를 그대로 CPU에서 돌리고, 강의 설명을 확인하는 작은 실행(마스크 유무 비교,
nn.Embedding·LM Head shape, Top-k/Top-p, 학습 vs 추론에서 가중치 변화)을 덧붙인다.
Tokenizer·문장 모델은 처음 실행할 때 Hugging Face Hub에서 내려받아 캐시에 둔다.
결과는 cmds/day20.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day20.py
"""
import html
import math
import os
import re
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
import transformers  # noqa: E402
import sentence_transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
IO = {}


def t2s(t, d=None):
    """tensor → 짧은 문자열 (줄바꿈 정리)"""
    if t.dim() == 0:
        v = t.item()
        return f"{v:.{d}f}" if (d is not None and isinstance(v, float)) else str(v)
    if d is not None:
        t = t.round(decimals=d)
    s = str(t)
    s = re.sub(r"\s+", " ", s).replace("tensor(", "").rstrip(")")
    s = re.sub(r"\[\s+", "[", s)
    return s


def lst(x):
    return "[" + ", ".join(f"{v:.2f}" for v in x) + "]"


# ------------------------------------------------------------------ 설치
IO["d20-pip-install"] = ("이 노트를 만든 가상환경에서 실제로 확인", [
    ("import torch, transformers, sentence_transformers",
     f"torch {torch.__version__} · transformers {transformers.__version__} · sentence-transformers {sentence_transformers.__version__}",
     "강의 기준(PyTorch 2.14 · Transformers 5.17 · sentence-transformers 6.1)보다 조금 새 버전 — 아래 결과는 이 버전 기준"),
])

# ------------------------------------------------------------------ Tokenizer
tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
text = "실습 파일은 개인 폴더에 보관합니다."
ids = tok(text)["input_ids"]
toks = tok.convert_ids_to_tokens(ids)
IO["d20-tokenizer"] = ("강의 코드 그대로 · bert-base-multilingual-cased", [
    ("print(tok.convert_ids_to_tokens(ids))", str(toks), "앞뒤 [CLS]·[SEP]는 Special Token · ##은 '앞 조각에 붙어 있음' 표시(WordPiece)"),
    ('print("ids:", ids)', "ids: " + str(ids), "Vocabulary 번호 — 119081이 101보다 '중요'하다는 뜻이 아니에요"),
    ('print("n_tokens:", len(ids))', f"n_tokens: {len(ids)}", f"원문은 공백 포함 {len(text)}글자 · 띄어쓰기 단어는 5개 — 셋 다 다른 숫자"),
    ("tok.vocab_size", f"{tok.vocab_size:,}", "이 Tokenizer가 아는 Token 종류 수 = ID 범위 0 ~ 119,546"),
])
DAY20_TOKENS = toks  # 그림에서 재사용할 수 있게

dec_full = tok.decode(ids)
dec_skip = tok.decode(ids, skip_special_tokens=True)
IO["d20-special-tokens"] = ("같은 Tokenizer로 실제 실행", [
    ("tok.all_special_tokens", str(tok.all_special_tokens), "BERT 계열 구분용 Token — GPT 계열은 이름·종류가 달라요(BOS·EOS 등)"),
    ("tok.all_special_ids", str(tok.all_special_ids), "[PAD]=0 · [UNK]=100 · [CLS]=101 · [SEP]=102 · [MASK]=103"),
    ('tok("실습", add_special_tokens=False)["input_ids"]', str(tok("실습", add_special_tokens=False)["input_ids"]),
     "Special Token을 빼면 '실' '##습' 두 개만"),
    ("tok.decode(ids)", dec_full, "ID → 문자열로 되돌림 · ## 조각은 다시 붙어요"),
    ("tok.decode(ids, skip_special_tokens=True)", dec_skip, "구분용 Token을 빼고 원문 복원"),
])

rows = []
for t, why in [("실습 파일은 개인 폴더에 보관합니다.", "한국어 · 글자 단위로 많이 쪼개짐"),
               ("Keep practice files in your personal folder.", "영어 · 단어 대부분이 한 Token, folder만 f ##old ##er"),
               ("def get_db(): yield db", "코드 · 기호 하나하나가 Token"),
               ("학습 파일", "강의 예시 — '학습 파일'도 4조각"),
               ("FastAPI와 PostgreSQL", "고유명사는 잘게 · 조사 '와'는 ##와로 붙음")]:
    i = tok(t)["input_ids"]
    rows.append((f'"{t}"', f"{len(i)}개 · " + " ".join(tok.convert_ids_to_tokens(i)), why))
IO["d20-compare-tokens"] = ("같은 Tokenizer · 실제 개수 (Special Token 2개 포함)", rows)

texts = ["AI", "FastAPI와 PostgreSQL"]
batch = tok(texts, padding=True, truncation=True, max_length=12, return_tensors="pt")
b4 = tok(texts, padding=True, truncation=True, max_length=4, return_tensors="pt")
bml = tok(texts, padding="max_length", truncation=True, max_length=12, return_tensors="pt")
IO["d20-batch-padding"] = ("강의 코드 그대로 + max_length·padding 바꿔 보기", [
    ('print(batch["input_ids"].shape)', str(batch["input_ids"].shape), "2문장 × 9 Token — 긴 문장(9개)에 맞춰짐, 12까지 채우지 않음"),
    ('print(batch["attention_mask"])', t2s(batch["attention_mask"]), "1 = 실제 Token · 0 = [PAD] (짧은 'AI' 뒤 6칸)"),
    ("첫 문장 Token", " ".join(tok.convert_ids_to_tokens(batch["input_ids"][0])), "PAD의 ID는 0"),
    ("max_length=4 로 바꾸면", f'{b4["input_ids"].shape} · ' + " | ".join(" ".join(tok.convert_ids_to_tokens(r)) for r in b4["input_ids"]),
     "두 번째 문장이 잘림 — PostgreSQL이 통째로 사라졌는데 오류는 없어요. [SEP]은 끝에 다시 붙음"),
    ('padding="max_length" 로 바꾸면', f'{bml["input_ids"].shape} · mask {t2s(bml["attention_mask"][1])}',
     "padding=True는 '배치 최장'까지, 'max_length'는 정한 길이까지 채움"),
])

# ------------------------------------------------------------------ Embedding
torch.manual_seed(0)
emb = nn.Embedding(num_embeddings=100, embedding_dim=4)
one = emb(torch.tensor(7))
sent = emb(torch.tensor([7, 3, 99, 3, 0]))
bat = emb(torch.randint(0, 100, (2, 5)))
IO["d20-embedding-lookup"] = ("작은 nn.Embedding(100, 4) · seed 0 · 실제 실행", [
    ("emb(torch.tensor(7)).shape", str(one.shape), "ID 하나(정수) → 4차원 벡터 하나"),
    ("emb(torch.tensor(7))", t2s(one.detach(), 2), "학습 전이라 무작위 값 — 학습하며 바뀌어요"),
    ("emb(torch.tensor([7, 3, 99, 3, 0])).shape", str(sent.shape), "Token 5개 × feature 4"),
    ("같은 ID 3의 두 행 비교", str(torch.equal(sent[1], sent[3])), "Embedding은 '표에서 행 꺼내기' — 위치가 달라도 같은 ID면 같은 벡터 (그래서 Position 정보가 따로 필요)"),
    ("emb(torch.randint(0, 100, (2, 5))).shape", str(bat.shape), "Batch 2 × Token 5 × feature 4 = [B, T, D]"),
    ("emb.weight.shape", str(emb.weight.shape), "Embedding Table = Vocabulary 크기 × 차원"),
])

a = torch.tensor([1.0, 0.0]); b = torch.tensor([0.8, 0.2]); c = torch.tensor([-1.0, 0.0])
cab = F.cosine_similarity(a, b, dim=0); cac = F.cosine_similarity(a, c, dim=0)
IO["d20-cosine"] = ("강의 코드 그대로 + 두 줄 더", [
    ("F.cosine_similarity(a, b, dim=0)", t2s(cab, 4), "방향이 거의 같음 (각도 약 14°)"),
    ("F.cosine_similarity(a, c, dim=0)", t2s(cac, 4), "정반대 방향"),
    ("a vs [0.0, 1.0]", t2s(F.cosine_similarity(a, torch.tensor([0.0, 1.0]), dim=0), 4), "직각 — 방향 관련 없음"),
    ("a vs [3.0, 0.0]", t2s(F.cosine_similarity(a, torch.tensor([3.0, 0.0]), dim=0), 4), "길이가 3배여도 1 — Cosine은 길이가 아니라 방향만 봐요"),
])

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
en = ["The file is in your project folder.", "Look for it in the project directory.", "The weather is sunny today."]
ev = model.encode(en)
es = model.similarity(ev, ev)
ko = ["실습 파일은 개인 폴더에 보관합니다.", "프로젝트 폴더에서 서버를 실행합니다.", "오류 메시지와 시도한 작업을 함께 적습니다."]
kv = model.encode(ko)
ks = model.similarity(kv, kv)
SIM = {"en": es, "ko": ks}
IO["d20-sentence-transformer"] = ("강의 코드 그대로 · all-MiniLM-L6-v2 (첫 실행 때 내려받음)", [
    ("print(vectors.shape)", str(ev.shape), "문장 3개 × 384차원 — 문장 하나가 벡터 하나"),
    ("similarity: 파일 위치 ↔ 폴더 찾기", f"{es[0,1]:.4f}", "단어가 달라도(folder/directory) 뜻이 가까워 높음"),
    ("similarity: 파일 위치 ↔ 날씨", f"{es[0,2]:.4f}", "관계없는 문장은 0 근처"),
    ("similarity 대각선", f"{es[0,0]:.4f} · {es[1,1]:.4f} · {es[2,2]:.4f}", "자기 자신과는 1"),
    ("같은 모델에 한국어 3문장 (프로젝트 사례 A·B·C)", f"A↔B {ks[0,1]:.4f} · A↔C {ks[0,2]:.4f} · B↔C {ks[1,2]:.4f}",
     "⚠️ 주제가 다른데 전부 0.86~0.89 — 영어용 모델이라 한국어 뜻을 구별 못 해요. 점수가 높다고 의미가 가까운 게 아니에요"),
])

# ------------------------------------------------------------------ Transformer Block
torch.manual_seed(42)
layer = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True, dropout=0.0)
x = torch.randn(2, 5, 8)
y = layer(x)
try:
    nn.TransformerEncoderLayer(d_model=8, nhead=3, batch_first=True)
    nh3 = "오류 없음"
except Exception as e:  # noqa: BLE001
    nh3 = f"{type(e).__name__}: {e}"
nparam = sum(p.numel() for p in layer.parameters())
IO["d20-encoder-layer"] = ("강의 코드 그대로 + 제약 확인", [
    ("print(x.shape, y.shape)", f"{x.shape} {y.shape}", "입력과 출력 shape가 같아요 — 그래서 Block을 여러 층 쌓을 수 있어요"),
    ("sum(p.numel() for p in layer.parameters())", f"{nparam}", "이 작은 층 하나의 학습 파라미터 수 — (2,5,8)의 숫자와는 다른 개념"),
    ("nhead=3 으로 바꾸면", nh3, "d_model(8)은 head 수로 나누어떨어져야 해요 — head마다 8÷2=4차원씩 나눠 가짐"),
    ("torch.equal(layer(x), layer(x))", str(torch.equal(layer(x), layer(x))), "dropout=0.0이라 같은 입력이면 같은 출력"),
])

torch.manual_seed(0)
V, D = 100, 8
embed = nn.Embedding(V, D)
block = nn.TransformerEncoderLayer(d_model=D, nhead=2, dim_feedforward=16, batch_first=True, dropout=0.0)
head = nn.Linear(D, V)
tid = torch.randint(0, V, (2, 5))
h0 = embed(tid); h1 = block(h0); lg = head(h1)
pr = torch.softmax(lg[:, -1, :], dim=-1)
IO["d20-shapes-lmhead"] = ("강의 shape 표를 작은 층으로 실제 실행 · seed 0", [
    ("token_ids.shape", str(tid.shape), "2문장 · 문장당 5 Token"),
    ("embed(token_ids).shape", str(h0.shape), "각 Token이 8차원 벡터"),
    ("block(h).shape", str(h1.shape), "Attention + FFN을 거쳐도 [B, T, D] 유지"),
    ("lm_head(h).shape", str(lg.shape), "위치마다 어휘 100개 각각의 점수(Logit)"),
    ("logits[:, -1, :].softmax(-1).sum(-1)", t2s(pr.sum(-1).detach(), 4), "다음 Token은 마지막 위치 점수로 — Softmax 뒤 합이 1"),
    ("logits[0, -1].min(), .max()", f"{lg[0,-1].min().item():.2f}, {lg[0,-1].max().item():.2f}", "Logit은 음수도 되는 점수 — 확률이 아니에요"),
])

# ------------------------------------------------------------------ Attention
q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
k, v = q.clone(), q.clone()
scores = q @ k.T / math.sqrt(2)
mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
weights = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
out = weights @ v
IO["d20-attention"] = ("강의 코드 그대로 + 중간값 출력", [
    ("print(scores.round(decimals=2))", t2s(scores, 2), "(3,3) 관계 점수 · 행=Query, 열=Key · 아직 확률 아님"),
    ("print(mask)", t2s(mask), "True = 가릴 칸 (대각선 위 = 미래)"),
    ("scores.masked_fill(mask, -inf)", t2s(scores.masked_fill(mask, float("-inf")), 2), "미래 칸을 -inf로 → Softmax 뒤 정확히 0"),
    ("print(weights.round(decimals=2))", t2s(weights, 2), "t1은 자기만 · t3는 [0.25, 0.25, 0.50]"),
    ("weights.sum(dim=-1)", t2s(weights.sum(-1)), "각 행의 합은 1"),
    ("weights @ v", t2s(out, 2), "t3 출력 = 0.25·[1,0] + 0.25·[0,1] + 0.5·[1,1] = [0.75, 0.75]"),
])
ATT = {"scores": scores, "weights": weights, "out": out}

nom = torch.softmax(scores, dim=-1)
IO["d20-attention-nomask"] = ("같은 q/k/v · 실제 실행", [
    ("torch.softmax(scores, dim=-1)  # Mask 없음", t2s(nom, 2), "t1이 미래 t3에 0.40을 줌 — 생성 모델이라면 정답 엿보기"),
    ("nomask_weights @ v", t2s(nom @ v, 2), "t1·t2 출력이 바뀜 · 마지막 t3 행은 원래 볼 수 있는 게 전부라 Mask 유무와 무관"),
    ("Mask 있음 t1 행 / 없음 t1 행", f"{lst(weights[0])} / {lst(nom[0])}", "Encoder(양방향)는 Mask 없이, Decoder(생성)는 Causal Mask로"),
])
ATT["nomask"] = nom

sd = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)
IO["d20-sdpa"] = ("같은 q/k/v · 실제 실행", [
    ("F.scaled_dot_product_attention(q, k, v, is_causal=True)", t2s(sd[0], 2), "직접 계산한 weights @ v와 같은 값 — √d_k 나누기·Mask·Softmax를 한 번에"),
    ("torch.allclose(sdpa, weights @ v)", str(torch.allclose(sd[0], out)), "같은 수식"),
    ("SDPA(attn_mask=mask) 와 SDPA(attn_mask=~mask) 중 is_causal과 같은 것",
     f"mask: {torch.allclose(F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=mask)[0], out)} · "
     f"~mask: {torch.allclose(F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=~mask)[0], out)}",
     "SDPA의 bool Mask는 True = 참여 — masked_fill용 Mask(True = 가림)를 그대로 주면 거꾸로 돼요. 뒤집어(~) 줘야 같음"),
])

# Padding Mask: PAD 자리 값을 바꿔도 실제 Token 출력은 그대로인가?
torch.manual_seed(42)
pl = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True, dropout=0.0).eval()
xp = torch.randn(1, 5, 8)
xp2 = xp.clone(); xp2[0, 3:] = torch.randn(2, 8) * 10          # PAD 자리(3·4번)에 엉뚱한 값
attn_mask = torch.tensor([[1, 1, 1, 0, 0]])                    # Tokenizer식: 1 = 실제
kpm = attn_mask == 0                                           # PyTorch식: True = 무시
with torch.no_grad():
    o_m1, o_m2 = pl(xp, src_key_padding_mask=kpm), pl(xp2, src_key_padding_mask=kpm)
    o_n1, o_n2 = pl(xp), pl(xp2)
IO["d20-padding-mask"] = ("작은 EncoderLayer · seed 42 · 실제 실행", [
    ("attention_mask (Tokenizer식)", t2s(attn_mask), "1 = 실제 Token · 0 = PAD"),
    ("src_key_padding_mask = attention_mask == 0", t2s(kpm), "PyTorch 이 인자는 반대로 True = 무시할 위치 — 그대로 넘기면 거꾸로 가려요"),
    ("PAD 자리 값을 바꿔도 실제 0~2번 출력이 같나? (Mask 있음)", str(torch.allclose(o_m1[0, :3], o_m2[0, :3], atol=1e-6)), "PAD를 안 보니 영향 없음"),
    ("같은 비교 (Mask 없음)", str(torch.allclose(o_n1[0, :3], o_n2[0, :3], atol=1e-6)), "PAD 자리 쓰레기 값이 실제 Token 표현에 섞여 들어감"),
])

# ------------------------------------------------------------------ Softmax · Temperature · 디코딩
logits = torch.tensor([2.0, 1.0, 0.0])
trows = []
TEMP = {}
for temp in [0.5, 1.0, 2.0]:
    probs = torch.softmax(logits / temp, dim=0)
    TEMP[temp] = probs
    trows.append((f"temp = {temp}", t2s(probs, 3), {0.5: "1등 후보에 몰림(0.867)", 1.0: "기본 Softmax", 2.0: "완만해짐 — 3등도 0.186"}[temp]))
trows.append(("logits.argmax()", t2s(logits.argmax()), "Temperature를 바꿔도 1등은 그대로 0번 — 순서는 안 바뀌고 '몰림 정도'만 바뀜"))
IO["d20-temperature"] = ("강의 코드 그대로", trows)

vocab = ["확인", "점검", "재시작", "삭제", "날씨"]
lg5 = torch.tensor([3.0, 2.5, 1.0, 0.5, -1.0])
p5 = torch.softmax(lg5, 0)
sp, si = p5.sort(descending=True)
cum = sp.cumsum(0)
k2 = sp[:2] / sp[:2].sum()
n_p = int((cum < 0.9).sum().item()) + 1
pp = sp[:n_p] / sp[:n_p].sum()
DEC = {"vocab": vocab, "logits": lg5, "p": p5}
IO["d20-decoding"] = ("장난감 어휘 5개 · 실제 계산", [
    ("softmax(logits)", ", ".join(f"{w} {x:.3f}" for w, x in zip(vocab, p5.tolist())), "다음 Token 후보 분포"),
    ("Greedy: probs.argmax()", vocab[int(p5.argmax())], "항상 1등 — 같은 입력이면 같은 답, 반복되기 쉬움"),
    ("Top-k (k=2) 후 다시 정규화", ", ".join(f"{vocab[i]} {x:.3f}" for i, x in zip(si[:2].tolist(), k2.tolist())), "상위 2개만 남겨 그 안에서 뽑기"),
    ("누적 확률 (큰 순)", t2s(cum, 3), "1등 0.543 → 2등까지 0.872 → 3등까지 0.946"),
    ("Top-p (p=0.9) 후 다시 정규화", ", ".join(f"{vocab[i]} {x:.3f}" for i, x in zip(si[:n_p].tolist(), pp.tolist())),
     f"누적이 0.9를 처음 넘는 곳까지 {n_p}개 — 분포가 퍼지면 개수가 늘어요"),
    ("torch.multinomial(probs, 1)", "실행할 때마다 달라질 수 있음", "샘플링은 무작위 — 재현하려면 seed 고정"),
])

# ------------------------------------------------------------------ 학습 vs 추론
torch.manual_seed(0)
lin = nn.Linear(4, 3)
xx = torch.randn(8, 4); yy = torch.randint(0, 3, (8,))
w0 = lin.weight.detach().clone()
lin.eval()
with torch.no_grad():
    o = lin(xx)
inf_same = torch.equal(w0, lin.weight)
inf_grad = lin.weight.grad
lin.train()
opt = torch.optim.SGD(lin.parameters(), lr=0.5)
loss0 = F.cross_entropy(lin(xx), yy)
opt.zero_grad(); loss0.backward(); opt.step()
loss1 = F.cross_entropy(lin(xx), yy)
IO["d20-train-vs-infer"] = ("작은 nn.Linear(4, 3) · seed 0 · 실제 실행", [
    ("추론: with torch.no_grad(): model(x)", f"출력 shape {tuple(o.shape)} · weight 그대로 {inf_same} · grad {inf_grad}", "가중치 고정 · 기울기 계산도 안 함"),
    ("학습: loss.backward(); optimizer.step()", f"weight[0,0] {w0[0,0]:.4f} → {lin.weight[0,0].item():.4f}", "가중치가 바뀜"),
    ("같은 x의 Loss (step 전 → 후)", f"{loss0.item():.4f} → {loss1.item():.4f}", "오차를 줄이는 쪽으로 한 걸음"),
])

# ------------------------------------------------------------------ 출력 · 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


if __name__ == "__main__":
    for cid, (title, rows) in IO.items():
        print(f"== {cid}")
        for i, o, n in rows:
            print(f"   {i}\n      → {o}   ({n})")
    frag = ROOT / "cmds" / "day20.html"
    if frag.exists():
        s = frag.read_text(encoding="utf-8")
        s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
        missing = []
        for cid, (title, rows) in IO.items():
            m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
            if not m:
                missing.append(cid); continue
            sec = m.group(0)
            kk = sec.find("</code></pre>\n")
            kk = kk + len("</code></pre>\n") if kk >= 0 else sec.find("</h3>\n") + len("</h3>\n")
            s = s[:m.start() + kk] + box(title, rows) + "\n" + s[m.start() + kk:]
        frag.write_text(s, encoding="utf-8")
        print("inserted", len(IO) - len(missing), "missing", missing)
