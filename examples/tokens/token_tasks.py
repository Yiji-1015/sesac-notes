# 업무 메시지에서 사람, 회사, 날짜, 금액 찾기(개체명 인식)와 질문의 답 꺼내기(질문 답변) 전체 코드.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/ner.jsonl (문장 376개: train 240, valid 48, test 40은 처음 보는 말투와 이름, en_test 48은 영어)
#         data/qa.jsonl  (질문 1,111개: train 870, valid 145, test 96은 처음 보는 이름과 질문 말투)
# 모델: intfloat/multilingual-e5-small (이 PC에 받아 둔 것). 모두 CPU로 돌려요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
from pathlib import Path                                         #@1
import torch                                                     #@2
import torch.nn.functional as F                                  #@3
from transformers import AutoTokenizer, AutoModel                #@2

DATA = Path(__file__).resolve().parent / "data"                  #@1
MODEL = "intfloat/multilingual-e5-small"                         #@2

# ① 데이터 읽기
ner = [json.loads(line) for line in (DATA / "ner.jsonl").open(encoding="utf-8")]
split = {s: [r for r in ner if r["split"] == s] for s in ("train", "valid", "test", "en_test")}
KINDS = ["PER", "ORG", "DATE", "MONEY"]                          # 사람, 회사, 날짜, 금액
TAGS = ["O"] + [f"{p}-{k}" for k in KINDS for p in ("B", "I")]   # O: 개체 아님, B: 개체 시작, I: 개체 안

# ② 토큰과 라벨 맞추기
tok = AutoTokenizer.from_pretrained(MODEL)

def encode_ner(rows):                                            # 글자 위치로 된 개체를 토큰마다 라벨로
    b = tok([r["text"] for r in rows], padding=True, return_tensors="pt", return_offsets_mapping=True)
    labels = torch.full(b["input_ids"].shape, -100)              # -100: 손실과 채점에서 빼는 자리
    for i, r in enumerate(rows):
        for j, (s, e) in enumerate(b["offset_mapping"][i].tolist()):
            if e == 0:                                           # <s>, </s>, 채운 자리
                continue
            tag = "O"
            for a, z, kind in r["entities"]:
                if s < z and e > a:                              # 토큰의 글자 범위가 개체와 겹치면
                    tag = ("B-" if s <= a else "I-") + kind
            labels[i, j] = TAGS.index(tag)
    return b, labels

b0, l0 = encode_ner(split["train"][:1])

# ③ 토큰마다 분류하는 모델
class Tagger(torch.nn.Module):                                   # e5 몸통 + 토큰마다 384 → 9 분류 층
    def __init__(self):
        super().__init__()
        self.body = AutoModel.from_pretrained(MODEL)
        self.head = torch.nn.Linear(384, len(TAGS))

    def forward(self, b):
        return self.head(self.body(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state)

def fine_tune(model, rows, encode, loss_fn, epochs, train_layers=12):   # 문장 분류 쪽 ⑤와 같은 학습 반복
    torch.manual_seed(0)
    model.body.embeddings.requires_grad_(False)                  # 25만 어휘의 Embedding 표는 고정
    for layer in model.body.encoder.layer[:12 - train_layers]:
        layer.requires_grad_(False)
    opt = torch.optim.AdamW([{"params": [p for p in model.body.parameters() if p.requires_grad], "lr": 5e-5},
                             {"params": model.head.parameters(), "lr": 1e-3}])
    log = []
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(rows)).split(16):
            loss = loss_fn(model, *encode([rows[i] for i in idx]))
            opt.zero_grad()
            loss.backward()
            opt.step()
        log.append(loss.item())
    return model.eval(), log

def tag_loss(model, b, labels):                                  # 토큰마다 9종류 중 정답의 손실, -100 자리는 빼고
    return F.cross_entropy(model(b).flatten(0, 1), labels.flatten(), ignore_index=-100)

torch.manual_seed(0)
tagger, ner_log = fine_tune(Tagger(), split["train"], encode_ner, tag_loss, epochs=6)

# ④ 개체 단위로 채점하기
def tag_spans(tags):                                             # [0, 1, 2, 0, …] → {(시작 토큰, 끝 토큰, 종류), …}
    out, cur = set(), None
    for j, t in enumerate(list(tags) + [0]):
        name = TAGS[t] if t >= 0 else "O"
        if cur and (name == "O" or name.startswith("B-") or name[2:] != cur[2]):
            out.add(tuple(cur))
            cur = None
        if name.startswith("B-") or (name.startswith("I-") and cur is None):
            cur = [j, j, name[2:]]
        elif name.startswith("I-"):
            cur[1] = j
    return out

def predict_tags(rows):
    b, labels = encode_ner(rows)
    with torch.no_grad():
        pred = tagger(b).argmax(-1).masked_fill(labels == -100, -100)
    return b, labels, pred

def score_ner(rows):
    b, labels, pred = predict_tags(rows)
    tp = fp = fn = 0
    for g, p in zip(labels.tolist(), pred.tolist()):
        G, P = tag_spans(g), tag_spans(p)
        tp, fp, fn = tp + len(G & P), fp + len(P - G), fn + len(G - P)
    keep = labels != -100
    return {"token_acc": (pred[keep] == labels[keep]).float().mean().item(), "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}

ner_scores = {s: score_ner(split[s]) for s in ("valid", "test")}

# ⑤ 글자로 꺼내기: 끝 토큰에 붙은 조사 떼기
JOSA = ("님이", "님과", "님은", "님도", "에서", "에게", "까지", "부터", "으로", "에", "과", "와", "을", "를", "이", "가", "은", "는", "의", "로", "도")

def strip_josa(text):                                            # 한글로 끝나면 맨 뒤 조사 하나를 떼기
    for j in JOSA:
        if text.endswith(j) and len(text) - len(j) >= 2:
            return text[:-len(j)]
    return text

def extract(rows, clean=True):                                   # 문장마다 [(개체 글자, 종류), …]
    b, _, pred = predict_tags(rows)
    out = []
    for i, r in enumerate(rows):
        off = b["offset_mapping"][i].tolist()
        found = []
        for s, e, kind in sorted(tag_spans(pred[i].tolist())):
            text = r["text"][off[s][0]:off[e][1]].strip()
            found.append((strip_josa(text) if clean else text, kind))
        out.append(found)
    return out

def exact_rate(rows, clean=True):                                # 정답 개체 중 글자까지 똑같이 꺼낸 비율
    hit = total = 0
    for r, found in zip(rows, extract(rows, clean)):
        gold = [(r["text"][a:z], k) for a, z, k in r["entities"]]
        hit += sum(g in found for g in gold)
        total += len(gold)
    return hit / total

exact = {s: {"그대로": exact_rate(split[s], clean=False), "조사 떼기": exact_rate(split[s])} for s in ("valid", "test")}

# ⑥ 다른 언어로 옮기기: 한국어로만 배운 모델에 영어 문장
en_scores = score_ner(split["en_test"])
en_found = extract(split["en_test"][:4])

# ⑦ 질문 답변: 답의 시작과 끝 찾기
qa = [json.loads(line) for line in (DATA / "qa.jsonl").open(encoding="utf-8")]
qa_split = {s: [r for r in qa if r["split"] == s] for s in ("train", "valid", "test")}

def encode_qa(rows):                                             # (질문, 메시지) 한 쌍을 한 줄로, 답의 시작과 끝 토큰 번호
    b = tok([r["question"] for r in rows], [r["context"] for r in rows], padding=True, return_tensors="pt", return_offsets_mapping=True)
    starts, ends = [], []
    for i, r in enumerate(rows):
        st = en = 0                                              # 답이 없으면 0번(<s>)을 시작이자 끝으로
        if r.get("answer"):                                      # 답을 모르는 질문(⑧)은 학습에 안 쓰니 상관없음
            a, z = r["answer"]["start"], r["answer"]["start"] + len(r["answer"]["text"])
            for j, (s, e) in enumerate(b["offset_mapping"][i].tolist()):
                if b.sequence_ids(i)[j] == 1 and s < z and e > a:   # 메시지 쪽 토큰 중 답과 겹치는 것
                    st, en = st or j, j
        starts.append(st)
        ends.append(en)
    return b, torch.tensor(starts), torch.tensor(ends)

class Reader(torch.nn.Module):                                   # e5 몸통 + 토큰마다 (시작 점수, 끝 점수)
    def __init__(self):
        super().__init__()
        self.body = AutoModel.from_pretrained(MODEL)
        self.head = torch.nn.Linear(384, 2)

    def forward(self, b):
        logits = self.head(self.body(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state)
        logits = logits.masked_fill(b["attention_mask"][..., None] == 0, -1e4)   # 채운 자리는 답이 될 수 없음
        return logits[..., 0], logits[..., 1]

def span_loss(model, b, starts, ends):
    s, e = model(b)
    return (F.cross_entropy(s, starts) + F.cross_entropy(e, ends)) / 2

reader, qa_log = fine_tune(Reader(), qa_split["train"], encode_qa, span_loss, epochs=2)

def answer(rows, max_len=12):                                    # 시작 + 끝 점수가 가장 큰 범위, <s>가 더 크면 "답 없음"
    b, _, _ = encode_qa(rows)
    with torch.no_grad():
        s, e = reader(b)
    out = []
    for i, r in enumerate(rows):
        off, ctx = b["offset_mapping"][i].tolist(), [j for j, q in enumerate(b.sequence_ids(i)) if q == 1]
        score, a, z = max((s[i, a] + e[i, z], a, z) for a in ctx for z in ctx if a <= z < a + max_len)
        out.append(None if s[i, 0] + e[i, 0] > score else strip_josa(r["context"][off[a][0]:off[z][1]].strip()))
    return out

def score_qa(rows):
    pred = answer(rows)
    gold = [r["answer"]["text"] if r["answer"] else None for r in rows]
    has = [g is not None for g in gold]
    return {"exact": sum(p == g for p, g in zip(pred, gold)) / len(rows),
            "답 있는 질문": sum(p == g for p, g, h in zip(pred, gold, has) if h) / sum(has),
            "답 없는 질문": sum(p is None for p, h in zip(pred, has) if not h) / (len(has) - sum(has))}

qa_scores = {s: score_qa(qa_split[s]) for s in ("valid", "test")}

# ⑧ 찾아서 읽기: 메시지 여러 개에서 먼저 찾고, 찾은 메시지에서 답 꺼내기
search = [json.loads(line) for line in (DATA / "search.jsonl").open(encoding="utf-8")]
docs = [r["context"] for r in search]                            # 메시지 12개
retriever = AutoModel.from_pretrained(MODEL).eval()              # 미세조정하지 않은 원래 e5 (검색용)

def embed(texts, prefix):                                        # 문장 분류 쪽 ③과 같은 e5 문장 벡터
    with torch.no_grad():
        b = tok([prefix + t for t in texts], padding=True, return_tensors="pt")
        m = b["attention_mask"][..., None].float()
        return F.normalize((retriever(**b).last_hidden_state * m).sum(1) / m.sum(1), dim=-1)

doc_vec = embed(docs, "passage: ")
found = (embed([r["question"] for r in search], "query: ") @ doc_vec.T).argmax(1).tolist()   # 질문마다 가장 가까운 메시지
read = answer([{"question": r["question"], "context": docs[k]} for r, k in zip(search, found)])
search_hits = {"찾기": sum(docs[k] == r["context"] for r, k in zip(search, found)) / len(search),
               "찾고 읽기": sum(a == r["answer"] for r, a in zip(search, read)) / len(search)}
