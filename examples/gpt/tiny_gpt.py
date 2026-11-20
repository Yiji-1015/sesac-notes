# 이 노트의 글로 작은 GPT를 처음부터 학습하고, 다음 토큰을 고르는 방법(디코딩)을 비교하고, 요약을 ROUGE로 재기 전체 코드.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/notes_corpus.jsonl (옛 Day 노트 19일치에서 사람이 쓴 문단만, 34만 자)
#         data/pages.jsonl (주제 페이지 26쪽의 '한 문장'과 본문 문장, ⑧의 요약용)
# 모델: GPT-2 구조를 아주 작게(4층, 128칸) 만들어 무작위에서 시작. 미리 학습된 가중치는 쓰지 않아요. CPU로 2분쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
from collections import Counter                                  #@1
from pathlib import Path                                         #@1
import torch                                                     #@3
import torch.nn.functional as F                                  #@8
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers  #@2
from transformers import AutoTokenizer, AutoModel                #@2
from transformers import GPT2Config, GPT2LMHeadModel             #@4

DATA = Path(__file__).resolve().parent / "data"                  #@1

# ① 말뭉치 읽고 정리하기
days = [json.loads(line) for line in (DATA / "notes_corpus.jsonl").open(encoding="utf-8")]
paras = [p for d in days for p in d["text"].splitlines()]       # 한 줄 = 한 문단
dup = Counter(paras)
unique = list(dict.fromkeys(paras))                              # 같은 문단은 처음 하나만 (순서 유지)
train_txt = [p for i, p in enumerate(unique) if i % 20 != 0]     # 문단 20개 중 19개는 학습
valid_txt = [p for i, p in enumerate(unique) if i % 20 == 0]     # 1개는 처음 보는 글로 남겨 두기

# ② Tokenizer 학습하기
def train_tokenizer(texts, vocab_size):
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.normalizer = normalizers.NFC()                            # 한글 자모가 따로 저장된 글자를 하나로
    tk.pre_tokenizer = pre_tokenizers.Metaspace()                # 띄어쓰기를 ▁로 바꿔 글자처럼 (SentencePiece 방식)
    tk.decoder = decoders.Metaspace()
    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<eos>"], min_frequency=2))
    return tk

tk = train_tokenizer(train_txt, 4000)
EOS = tk.token_to_id("<eos>")                                    # 문단 끝 표시
e5_tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")   # 비교용: 100개 언어 25만 어휘
valid_all = " ".join(valid_txt)
chars_per_token = {"우리 Tokenizer (4천 개)": len(valid_all) / len(tk.encode(valid_all).ids),
                   "e5 Tokenizer (25만 개)": len(valid_all) / len(e5_tok.tokenize(valid_all))}

# ③ 학습 데이터 만들기: 이어 붙이고 같은 길이로 자르기
BLOCK = 64                                                       # 모델이 한 번에 보는 토큰 수

def to_blocks(texts):
    ids = [i for t in texts for i in tk.encode(t).ids + [EOS]]  # 문단마다 끝에 <eos>를 붙여 한 줄로
    return torch.tensor(ids[: len(ids) // BLOCK * BLOCK]).view(-1, BLOCK)

X_train, X_valid = to_blocks(train_txt), to_blocks(valid_txt)

# ④ 작은 GPT 만들기 (무작위에서 시작)
config = GPT2Config(vocab_size=tk.get_vocab_size(), n_positions=BLOCK, n_embd=128, n_layer=4, n_head=4,
                    bos_token_id=EOS, eos_token_id=EOS)
torch.manual_seed(0)
gpt = GPT2LMHeadModel(config)                                    # 구조만 GPT-2, 가중치는 무작위

def valid_loss():
    gpt.eval()
    with torch.no_grad():
        return sum(gpt(b, labels=b).loss.item() for b in X_valid.split(64)) / len(X_valid.split(64))

start_loss = valid_loss()                                        # 아무것도 안 배웠을 때

# ⑤ 학습하기: 다음 토큰 맞히기
STEPS = 1000
opt = torch.optim.AdamW(gpt.parameters(), lr=2e-3, weight_decay=0.1)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)   # 처음엔 키우고 점점 줄이기
log = []
for step in range(1, STEPS + 1):
    gpt.train()
    batch = X_train[torch.randint(0, len(X_train), (32,))]      # 무작위 32줄
    loss = gpt(batch, labels=batch).loss                         # labels를 한 칸 밀어서 다음 토큰과 비교하는 건 모델이 해요
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
    opt.step()
    sched.step()
    if step % 200 == 0:
        log.append({"step": step, "train": loss.item(), "valid": valid_loss()})
gpt.eval()

# ⑥ 고르는 방법 1: 가장 그럴듯한 것 고르기 (greedy, beam)
def hangul_only(text):
    return all("가" <= ch <= "힣" or ch == " " for ch in text)

starts = Counter(" ".join(p.split()[:3]) for p in unique)       # 문단 시작 세 어절이 몇 번 나오나
prompts = [h for h in (" ".join(p.split()[:3]) for p in valid_txt) if hangul_only(h) and starts[h] == 1][:6]   # 처음 보는 문단의 앞 세 어절 (한글만, 상투적인 시작은 빼고)
torch.manual_seed(0)

def generate(prompt, **how):                                     # 앞부분을 주고 30토큰 이어 쓰기
    inp = torch.tensor([tk.encode(prompt).ids])
    out = gpt.generate(inp, max_new_tokens=30, pad_token_id=EOS, eos_token_id=None, **how)[0]
    return out[inp.shape[1]:].tolist()

def repeat_rate(ids):                                            # 이어 쓴 부분에서 앞에 이미 나온 3-토큰 묶음의 비율
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return sum(g in grams[:i] for i, g in enumerate(grams)) / len(grams)

def avg_nll(prompt, ids):                                        # 모델이 보기에 이어 쓴 토큰들이 얼마나 뜻밖인가 (낮을수록 그럴듯)
    full = torch.tensor([tk.encode(prompt).ids + ids])
    with torch.no_grad():
        logp = gpt(full).logits[0, :-1].log_softmax(-1)
    n = len(ids)
    return -logp[-n:].gather(1, full[0, -n:, None]).mean().item()

METHODS = {"greedy": dict(do_sample=False), "beam 5": dict(do_sample=False, num_beams=5),
           "beam 5 + 3-gram 반복 금지": dict(do_sample=False, num_beams=5, no_repeat_ngram_size=3)}

# ⑦ 고르는 방법 2: 확률대로 뽑기 (temperature, top-k, top-p)
METHODS.update({"sampling t=1.0": dict(do_sample=True, top_k=0, temperature=1.0), "sampling t=0.7": dict(do_sample=True, top_k=0, temperature=0.7),
                "top-k 20": dict(do_sample=True, top_k=20), "top-p 0.9": dict(do_sample=True, top_k=0, top_p=0.9)})
outputs = {name: [generate(p, **how) for p in prompts] for name, how in METHODS.items()}

def distinct2(seqs):                                             # 모든 이어 쓰기에서 서로 다른 2-토큰 묶음의 비율 (높을수록 다양)
    grams = [tuple(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)]
    return len(set(grams)) / len(grams)

decode_table = {name: {"반복": sum(map(repeat_rate, seqs)) / len(seqs), "다양성": distinct2(seqs),
                       "뜻밖의 정도": sum(avg_nll(p, s) for p, s in zip(prompts, seqs)) / len(seqs)} for name, seqs in outputs.items()}

# ⑧ 요약과 평가: 본문에서 대표 문장 고르기와 ROUGE
pages = [json.loads(line) for line in (DATA / "pages.jsonl").open(encoding="utf-8")]
e5 = AutoModel.from_pretrained("intfloat/multilingual-e5-small").eval()

def embed(texts):                                                # 문장 분류 쪽 ③과 같은 e5 문장 벡터
    with torch.no_grad():
        b = e5_tok(["passage: " + t for t in texts], padding=True, truncation=True, return_tensors="pt")
        m = b["attention_mask"][..., None].float()
        return F.normalize((e5(**b).last_hidden_state * m).sum(1) / m.sum(1), dim=-1)

def pieces(text):                                                # ROUGE용: e5 Tokenizer 조각 (▁만 있는 조각은 빼기)
    return [p.lstrip("▁") for p in e5_tok.tokenize(text) if p.lstrip("▁")]

def rouge(candidate, reference):                                 # ROUGE-1과 ROUGE-L의 재현율, 정밀도, F1
    c, r = pieces(candidate), pieces(reference)
    overlap = sum((Counter(c) & Counter(r)).values())            # 겹치는 조각 수 (ROUGE-1)
    lcs = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]        # 가장 긴 공통 부분 순서 (ROUGE-L)
    for i in range(len(c)):
        for j in range(len(r)):
            lcs[i + 1][j + 1] = lcs[i][j] + 1 if c[i] == r[j] else max(lcs[i][j + 1], lcs[i + 1][j])
    out = {}
    for name, hit in (("ROUGE-1", overlap), ("ROUGE-L", lcs[-1][-1])):
        rec, pre = hit / len(r), hit / len(c)
        out[name] = {"재현율": rec, "정밀도": pre, "F1": 2 * rec * pre / (rec + pre) if hit else 0.0}
    return out

summaries = []
for pg in pages:
    v = embed(pg["sentences"])
    k = int((v @ F.normalize(v.mean(0), dim=0)).argmax())        # 본문 전체의 평균과 가장 가까운 문장
    ref = embed([pg["summary"]])[0]
    summaries.append({"page": pg["page"], "첫 문장": pg["sentences"][0], "대표 문장": pg["sentences"][k], "정답": pg["summary"],
                      "첫 문장 뜻": float(v[0] @ ref), "대표 문장 뜻": float(v[k] @ ref)})   # 정답 요약과의 코사인 유사도
rouge_table = {way: {**{m: sum(rouge(s[way], s["정답"])[m]["F1"] for s in summaries) / len(summaries) for m in ("ROUGE-1", "ROUGE-L")},
                     "뜻 유사도": sum(s[way + " 뜻"] for s in summaries) / len(summaries)} for way in ("첫 문장", "대표 문장")}
