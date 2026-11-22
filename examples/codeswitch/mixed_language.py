# 한국어와 영어가 섞인 글(코드 스위칭)을 이 노트의 실제 문장으로 세어 보고, 다국어 임베딩 모델이 같은 개념을 한국어로 물을 때와 영어로 물을 때,
# 섞어서 물을 때 똑같이 찾는지 재 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: ../gpt/data/pages.jsonl (주제 페이지 본문 문장), ../../terms/day*.json (용어집: 한국어 용어, 영어 이름, 한국어 설명)
# 모델: intfloat/multilingual-e5-small (이 PC에 받아 둔 것). CPU로 30초쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                                     #@1
import re                                                                       #@1
from collections import Counter                                                 #@1
from pathlib import Path                                                        #@1
import numpy as np                                                              #@1
import torch                                                                    #@3
from transformers import AutoTokenizer, AutoModel                               #@2

# ① 노트 문장은 얼마나 섞여 있나
HERE = Path(__file__).resolve().parent
HANGUL, LATIN = re.compile(r"[가-힣]"), re.compile(r"[A-Za-z]")
pages = [json.loads(line) for line in (HERE.parent / "gpt" / "data" / "pages.jsonl").open(encoding="utf-8")]
sentences = [s for p in pages for s in p["sentences"]]
def word_kind(w):
    h, l = bool(HANGUL.search(w)), bool(LATIN.search(w))
    return "영어+한국어 조사/어미" if h and l else "영어" if l else "한국어" if h else None
kinds = [Counter(k for k in map(word_kind, s.split()) if k) for s in sentences]
mix_stats = {
    "문장 수": len(sentences),
    "영어 단어가 하나라도 있는 문장": sum(1 for c in kinds if c["영어"] or c["영어+한국어 조사/어미"]),
    "영어 단어에 조사가 붙은 문장 (예: RAG는)": sum(1 for c in kinds if c["영어+한국어 조사/어미"]),
    "한국어 단어만 있는 문장": sum(1 for c in kinds if not c["영어"] and not c["영어+한국어 조사/어미"]),
}
english_share = np.mean([(c["영어"] + c["영어+한국어 조사/어미"]) / sum(c.values()) for c in kinds if sum(c.values())])
attached = Counter(re.sub(r"^[A-Za-z0-9\-_.]+", "", w) for s in sentences for w in s.split() if word_kind(w) == "영어+한국어 조사/어미" and re.match(r"^[A-Za-z]", w))

# ② 용어집에서 한국어 이름, 영어 이름, 설명 짝 만들기
terms = []
seen = set()
for f in sorted((HERE.parent.parent / "terms").glob("day*.json")):
    for t in json.loads(f.read_text(encoding="utf-8")):
        ko, desc = t["term"], t["short"]
        if not HANGUL.search(ko) or LATIN.search(ko):
            continue
        en = next((c for c in [t.get("full", "")] + t.get("aliases", []) if c and LATIN.search(c) and not HANGUL.search(c) and (" " in c or len(c) > 5)), None)
        if not en or en.lower() in desc.lower() or ko in desc or ko in seen or en in seen:   # 설명에 이름이 그대로 있으면 너무 쉬워서 빼요
            continue
        seen |= {ko, en}
        terms.append((ko, en, desc))

MODEL = "intfloat/multilingual-e5-small"
tok = AutoTokenizer.from_pretrained(MODEL)
token_count = {"한국어 이름": float(np.mean([len(tok(k)["input_ids"]) - 2 for k, _, _ in terms])),
               "영어 이름": float(np.mean([len(tok(e)["input_ids"]) - 2 for _, e, _ in terms]))}
pieces = {w: tok.tokenize(w) for w in ("RAG는", "Chunking을", "임베딩을", "embedding을")}

# ③ 설명 찾기: 한국어로 물을 때 vs 영어로 물을 때
model = AutoModel.from_pretrained(MODEL).eval()
def embed(texts, prefix):                                        # e5는 질문에 "query: ", 문서에 "passage: "를 붙여요
    out = []
    for i in range(0, len(texts), 64):
        b = tok([prefix + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            h = model(**b).last_hidden_state
        e = (h * b["attention_mask"][..., None]).sum(1) / b["attention_mask"].sum(1, keepdim=True)
        out.append(torch.nn.functional.normalize(e, dim=1))
    return torch.cat(out)

passages = embed([d for _, _, d in terms], "passage: ")
def found(queries, k=5):                                         # 맞는 설명이 위 k개 안에 있나 (용어마다 참/거짓)
    top = (embed(queries, "query: ") @ passages.T).topk(k, dim=1).indices
    return (top == torch.arange(len(queries))[:, None]).any(dim=1).numpy()

hit = {"한국어 이름": found([k for k, _, _ in terms]), "영어 이름": found([e for _, e, _ in terms])}
chance = 5 / len(terms)

# ④ 두 언어 모두 찾나 (교차 언어 일관성)
ko_hit, en_hit = hit["한국어 이름"], hit["영어 이름"]
consistency = {"둘 다 찾음": int((ko_hit & en_hit).sum()), "한국어로만": int((ko_hit & ~en_hit).sum()),
               "영어로만": int((~ko_hit & en_hit).sum()), "둘 다 못 찾음": int((~ko_hit & ~en_hit).sum())}
only_en_examples = [terms[i][:2] for i in np.flatnonzero(~ko_hit & en_hit)[:3]]

# ⑤ 조금씩 섞어서 물으면
multi = [i for i, (k, e, _) in enumerate(terms) if len(k.split()) >= 2 and len(e.split()) >= 2]   # 두 단어 이상인 용어
def mix(i, ratio):                                               # 앞에서부터 ratio만큼 영어 단어로 바꾸기
    kw, ew = terms[i][0].split(), terms[i][1].split()
    return " ".join(ew[:round(ratio * len(ew))] + kw[round(ratio * len(kw)):])
def found_subset(queries, idx, k=5):
    top = (embed(queries, "query: ") @ passages.T).topk(k, dim=1).indices
    return (top == torch.tensor(idx)[:, None]).any(dim=1).numpy()
gradual = {r: float(found_subset([mix(i, r) for i in multi], multi).mean()) for r in (0.0, 0.5, 1.0)}
gradual_example = {r: mix(multi[0], r) for r in (0.0, 0.5, 1.0)}

# ⑥ 두 언어를 함께 적으면
hit["한국어 (영어)"] = found([f"{k} ({e})" for k, e, _ in terms])
both_rescued = int((hit["한국어 (영어)"] & ~ko_hit & ~en_hit).sum())
