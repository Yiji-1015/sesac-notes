window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 2
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 3
  },
  {
   "t": "from transformers import AutoTokenizer, AutoModel",
   "s": 2
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\"",
   "s": 1
  },
  {
   "t": "MODEL = \"intfloat/multilingual-e5-small\"",
   "s": 2
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 데이터 읽기",
   "s": 1
  },
  {
   "t": "ner = [json.loads(line) for line in (DATA / \"ner.jsonl\").open(encoding=\"utf-8\")]",
   "s": 1
  },
  {
   "t": "split = {s: [r for r in ner if r[\"split\"] == s] for s in (\"train\", \"valid\", \"test\", \"en_test\")}",
   "s": 1
  },
  {
   "t": "KINDS = [\"PER\", \"ORG\", \"DATE\", \"MONEY\"]  # 사람, 회사, 날짜, 금액",
   "s": 1
  },
  {
   "t": "TAGS = [\"O\"] + [f\"{p}-{k}\" for k in KINDS for p in (\"B\", \"I\")]  # O: 개체 아님, B: 개체 시작, I: 개체 안",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 토큰과 라벨 맞추기",
   "s": 2
  },
  {
   "t": "tok = AutoTokenizer.from_pretrained(MODEL)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def encode_ner(rows):  # 글자 위치로 된 개체를 토큰마다 라벨로",
   "s": 2
  },
  {
   "t": "    b = tok([r[\"text\"] for r in rows], padding=True, return_tensors=\"pt\", return_offsets_mapping=True)",
   "s": 2
  },
  {
   "t": "    labels = torch.full(b[\"input_ids\"].shape, -100)  # -100: 손실과 채점에서 빼는 자리",
   "s": 2
  },
  {
   "t": "    for i, r in enumerate(rows):",
   "s": 2
  },
  {
   "t": "        for j, (s, e) in enumerate(b[\"offset_mapping\"][i].tolist()):",
   "s": 2
  },
  {
   "t": "            if e == 0:  # <s>, </s>, 채운 자리",
   "s": 2
  },
  {
   "t": "                continue",
   "s": 2
  },
  {
   "t": "            tag = \"O\"",
   "s": 2
  },
  {
   "t": "            for a, z, kind in r[\"entities\"]:",
   "s": 2
  },
  {
   "t": "                if s < z and e > a:  # 토큰의 글자 범위가 개체와 겹치면",
   "s": 2
  },
  {
   "t": "                    tag = (\"B-\" if s <= a else \"I-\") + kind",
   "s": 2
  },
  {
   "t": "            labels[i, j] = TAGS.index(tag)",
   "s": 2
  },
  {
   "t": "    return b, labels",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "b0, l0 = encode_ner(split[\"train\"][:1])",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 토큰마다 분류하는 모델",
   "s": 3
  },
  {
   "t": "class Tagger(torch.nn.Module):  # e5 몸통 + 토큰마다 384 → 9 분류 층",
   "s": 3
  },
  {
   "t": "    def __init__(self):",
   "s": 3
  },
  {
   "t": "        super().__init__()",
   "s": 3
  },
  {
   "t": "        self.body = AutoModel.from_pretrained(MODEL)",
   "s": 3
  },
  {
   "t": "        self.head = torch.nn.Linear(384, len(TAGS))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "    def forward(self, b):",
   "s": 3
  },
  {
   "t": "        return self.head(self.body(input_ids=b[\"input_ids\"], attention_mask=b[\"attention_mask\"]).last_hidden_state)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def fine_tune(model, rows, encode, loss_fn, epochs, train_layers=12):  # 문장 분류 쪽 ⑤와 같은 학습 반복",
   "s": 3
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 3
  },
  {
   "t": "    model.body.embeddings.requires_grad_(False)  # 25만 어휘의 Embedding 표는 고정",
   "s": 3
  },
  {
   "t": "    for layer in model.body.encoder.layer[:12 - train_layers]:",
   "s": 3
  },
  {
   "t": "        layer.requires_grad_(False)",
   "s": 3
  },
  {
   "t": "    opt = torch.optim.AdamW([{\"params\": [p for p in model.body.parameters() if p.requires_grad], \"lr\": 5e-5},",
   "s": 3
  },
  {
   "t": "                             {\"params\": model.head.parameters(), \"lr\": 1e-3}])",
   "s": 3
  },
  {
   "t": "    log = []",
   "s": 3
  },
  {
   "t": "    for epoch in range(epochs):",
   "s": 3
  },
  {
   "t": "        model.train()",
   "s": 3
  },
  {
   "t": "        for idx in torch.randperm(len(rows)).split(16):",
   "s": 3
  },
  {
   "t": "            loss = loss_fn(model, *encode([rows[i] for i in idx]))",
   "s": 3
  },
  {
   "t": "            opt.zero_grad()",
   "s": 3
  },
  {
   "t": "            loss.backward()",
   "s": 3
  },
  {
   "t": "            opt.step()",
   "s": 3
  },
  {
   "t": "        log.append(loss.item())",
   "s": 3
  },
  {
   "t": "    return model.eval(), log",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def tag_loss(model, b, labels):  # 토큰마다 9종류 중 정답의 손실, -100 자리는 빼고",
   "s": 3
  },
  {
   "t": "    return F.cross_entropy(model(b).flatten(0, 1), labels.flatten(), ignore_index=-100)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 3
  },
  {
   "t": "tagger, ner_log = fine_tune(Tagger(), split[\"train\"], encode_ner, tag_loss, epochs=6)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 개체 단위로 채점하기",
   "s": 4
  },
  {
   "t": "def tag_spans(tags):  # [0, 1, 2, 0, …] → {(시작 토큰, 끝 토큰, 종류), …}",
   "s": 4
  },
  {
   "t": "    out, cur = set(), None",
   "s": 4
  },
  {
   "t": "    for j, t in enumerate(list(tags) + [0]):",
   "s": 4
  },
  {
   "t": "        name = TAGS[t] if t >= 0 else \"O\"",
   "s": 4
  },
  {
   "t": "        if cur and (name == \"O\" or name.startswith(\"B-\") or name[2:] != cur[2]):",
   "s": 4
  },
  {
   "t": "            out.add(tuple(cur))",
   "s": 4
  },
  {
   "t": "            cur = None",
   "s": 4
  },
  {
   "t": "        if name.startswith(\"B-\") or (name.startswith(\"I-\") and cur is None):",
   "s": 4
  },
  {
   "t": "            cur = [j, j, name[2:]]",
   "s": 4
  },
  {
   "t": "        elif name.startswith(\"I-\"):",
   "s": 4
  },
  {
   "t": "            cur[1] = j",
   "s": 4
  },
  {
   "t": "    return out",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def predict_tags(rows):",
   "s": 4
  },
  {
   "t": "    b, labels = encode_ner(rows)",
   "s": 4
  },
  {
   "t": "    with torch.no_grad():",
   "s": 4
  },
  {
   "t": "        pred = tagger(b).argmax(-1).masked_fill(labels == -100, -100)",
   "s": 4
  },
  {
   "t": "    return b, labels, pred",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def score_ner(rows):",
   "s": 4
  },
  {
   "t": "    b, labels, pred = predict_tags(rows)",
   "s": 4
  },
  {
   "t": "    tp = fp = fn = 0",
   "s": 4
  },
  {
   "t": "    for g, p in zip(labels.tolist(), pred.tolist()):",
   "s": 4
  },
  {
   "t": "        G, P = tag_spans(g), tag_spans(p)",
   "s": 4
  },
  {
   "t": "        tp, fp, fn = tp + len(G & P), fp + len(P - G), fn + len(G - P)",
   "s": 4
  },
  {
   "t": "    keep = labels != -100",
   "s": 4
  },
  {
   "t": "    return {\"token_acc\": (pred[keep] == labels[keep]).float().mean().item(), \"precision\": tp / max(tp + fp, 1),",
   "s": 4
  },
  {
   "t": "            \"recall\": tp / max(tp + fn, 1), \"f1\": 2 * tp / max(2 * tp + fp + fn, 1)}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "ner_scores = {s: score_ner(split[s]) for s in (\"valid\", \"test\")}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 글자로 꺼내기: 끝 토큰에 붙은 조사 떼기",
   "s": 5
  },
  {
   "t": "JOSA = (\"님이\", \"님과\", \"님은\", \"님도\", \"에서\", \"에게\", \"까지\", \"부터\", \"으로\", \"에\", \"과\", \"와\", \"을\", \"를\", \"이\", \"가\", \"은\", \"는\", \"의\", \"로\", \"도\")",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def strip_josa(text):  # 한글로 끝나면 맨 뒤 조사 하나를 떼기",
   "s": 5
  },
  {
   "t": "    for j in JOSA:",
   "s": 5
  },
  {
   "t": "        if text.endswith(j) and len(text) - len(j) >= 2:",
   "s": 5
  },
  {
   "t": "            return text[:-len(j)]",
   "s": 5
  },
  {
   "t": "    return text",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def extract(rows, clean=True):  # 문장마다 [(개체 글자, 종류), …]",
   "s": 5
  },
  {
   "t": "    b, _, pred = predict_tags(rows)",
   "s": 5
  },
  {
   "t": "    out = []",
   "s": 5
  },
  {
   "t": "    for i, r in enumerate(rows):",
   "s": 5
  },
  {
   "t": "        off = b[\"offset_mapping\"][i].tolist()",
   "s": 5
  },
  {
   "t": "        found = []",
   "s": 5
  },
  {
   "t": "        for s, e, kind in sorted(tag_spans(pred[i].tolist())):",
   "s": 5
  },
  {
   "t": "            text = r[\"text\"][off[s][0]:off[e][1]].strip()",
   "s": 5
  },
  {
   "t": "            found.append((strip_josa(text) if clean else text, kind))",
   "s": 5
  },
  {
   "t": "        out.append(found)",
   "s": 5
  },
  {
   "t": "    return out",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def exact_rate(rows, clean=True):  # 정답 개체 중 글자까지 똑같이 꺼낸 비율",
   "s": 5
  },
  {
   "t": "    hit = total = 0",
   "s": 5
  },
  {
   "t": "    for r, found in zip(rows, extract(rows, clean)):",
   "s": 5
  },
  {
   "t": "        gold = [(r[\"text\"][a:z], k) for a, z, k in r[\"entities\"]]",
   "s": 5
  },
  {
   "t": "        hit += sum(g in found for g in gold)",
   "s": 5
  },
  {
   "t": "        total += len(gold)",
   "s": 5
  },
  {
   "t": "    return hit / total",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "exact = {s: {\"그대로\": exact_rate(split[s], clean=False), \"조사 떼기\": exact_rate(split[s])} for s in (\"valid\", \"test\")}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 다른 언어로 옮기기: 한국어로만 배운 모델에 영어 문장",
   "s": 6
  },
  {
   "t": "en_scores = score_ner(split[\"en_test\"])",
   "s": 6
  },
  {
   "t": "en_found = extract(split[\"en_test\"][:4])",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 질문 답변: 답의 시작과 끝 찾기",
   "s": 7
  },
  {
   "t": "qa = [json.loads(line) for line in (DATA / \"qa.jsonl\").open(encoding=\"utf-8\")]",
   "s": 7
  },
  {
   "t": "qa_split = {s: [r for r in qa if r[\"split\"] == s] for s in (\"train\", \"valid\", \"test\")}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def encode_qa(rows):  # (질문, 메시지) 한 쌍을 한 줄로, 답의 시작과 끝 토큰 번호",
   "s": 7
  },
  {
   "t": "    b = tok([r[\"question\"] for r in rows], [r[\"context\"] for r in rows], padding=True, return_tensors=\"pt\", return_offsets_mapping=True)",
   "s": 7
  },
  {
   "t": "    starts, ends = [], []",
   "s": 7
  },
  {
   "t": "    for i, r in enumerate(rows):",
   "s": 7
  },
  {
   "t": "        st = en = 0  # 답이 없으면 0번(<s>)을 시작이자 끝으로",
   "s": 7
  },
  {
   "t": "        if r.get(\"answer\"):  # 답을 모르는 질문(⑧)은 학습에 안 쓰니 상관없음",
   "s": 7
  },
  {
   "t": "            a, z = r[\"answer\"][\"start\"], r[\"answer\"][\"start\"] + len(r[\"answer\"][\"text\"])",
   "s": 7
  },
  {
   "t": "            for j, (s, e) in enumerate(b[\"offset_mapping\"][i].tolist()):",
   "s": 7
  },
  {
   "t": "                if b.sequence_ids(i)[j] == 1 and s < z and e > a:  # 메시지 쪽 토큰 중 답과 겹치는 것",
   "s": 7
  },
  {
   "t": "                    st, en = st or j, j",
   "s": 7
  },
  {
   "t": "        starts.append(st)",
   "s": 7
  },
  {
   "t": "        ends.append(en)",
   "s": 7
  },
  {
   "t": "    return b, torch.tensor(starts), torch.tensor(ends)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "class Reader(torch.nn.Module):  # e5 몸통 + 토큰마다 (시작 점수, 끝 점수)",
   "s": 7
  },
  {
   "t": "    def __init__(self):",
   "s": 7
  },
  {
   "t": "        super().__init__()",
   "s": 7
  },
  {
   "t": "        self.body = AutoModel.from_pretrained(MODEL)",
   "s": 7
  },
  {
   "t": "        self.head = torch.nn.Linear(384, 2)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "    def forward(self, b):",
   "s": 7
  },
  {
   "t": "        logits = self.head(self.body(input_ids=b[\"input_ids\"], attention_mask=b[\"attention_mask\"]).last_hidden_state)",
   "s": 7
  },
  {
   "t": "        logits = logits.masked_fill(b[\"attention_mask\"][..., None] == 0, -1e4)  # 채운 자리는 답이 될 수 없음",
   "s": 7
  },
  {
   "t": "        return logits[..., 0], logits[..., 1]",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def span_loss(model, b, starts, ends):",
   "s": 7
  },
  {
   "t": "    s, e = model(b)",
   "s": 7
  },
  {
   "t": "    return (F.cross_entropy(s, starts) + F.cross_entropy(e, ends)) / 2",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "reader, qa_log = fine_tune(Reader(), qa_split[\"train\"], encode_qa, span_loss, epochs=2)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def answer(rows, max_len=12):  # 시작 + 끝 점수가 가장 큰 범위, <s>가 더 크면 \"답 없음\"",
   "s": 7
  },
  {
   "t": "    b, _, _ = encode_qa(rows)",
   "s": 7
  },
  {
   "t": "    with torch.no_grad():",
   "s": 7
  },
  {
   "t": "        s, e = reader(b)",
   "s": 7
  },
  {
   "t": "    out = []",
   "s": 7
  },
  {
   "t": "    for i, r in enumerate(rows):",
   "s": 7
  },
  {
   "t": "        off, ctx = b[\"offset_mapping\"][i].tolist(), [j for j, q in enumerate(b.sequence_ids(i)) if q == 1]",
   "s": 7
  },
  {
   "t": "        score, a, z = max((s[i, a] + e[i, z], a, z) for a in ctx for z in ctx if a <= z < a + max_len)",
   "s": 7
  },
  {
   "t": "        out.append(None if s[i, 0] + e[i, 0] > score else strip_josa(r[\"context\"][off[a][0]:off[z][1]].strip()))",
   "s": 7
  },
  {
   "t": "    return out",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def score_qa(rows):",
   "s": 7
  },
  {
   "t": "    pred = answer(rows)",
   "s": 7
  },
  {
   "t": "    gold = [r[\"answer\"][\"text\"] if r[\"answer\"] else None for r in rows]",
   "s": 7
  },
  {
   "t": "    has = [g is not None for g in gold]",
   "s": 7
  },
  {
   "t": "    return {\"exact\": sum(p == g for p, g in zip(pred, gold)) / len(rows),",
   "s": 7
  },
  {
   "t": "            \"답 있는 질문\": sum(p == g for p, g, h in zip(pred, gold, has) if h) / sum(has),",
   "s": 7
  },
  {
   "t": "            \"답 없는 질문\": sum(p is None for p, h in zip(pred, has) if not h) / (len(has) - sum(has))}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "qa_scores = {s: score_qa(qa_split[s]) for s in (\"valid\", \"test\")}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 찾아서 읽기: 메시지 여러 개에서 먼저 찾고, 찾은 메시지에서 답 꺼내기",
   "s": 8
  },
  {
   "t": "search = [json.loads(line) for line in (DATA / \"search.jsonl\").open(encoding=\"utf-8\")]",
   "s": 8
  },
  {
   "t": "docs = [r[\"context\"] for r in search]  # 메시지 12개",
   "s": 8
  },
  {
   "t": "retriever = AutoModel.from_pretrained(MODEL).eval()  # 미세조정하지 않은 원래 e5 (검색용)",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "def embed(texts, prefix):  # 문장 분류 쪽 ③과 같은 e5 문장 벡터",
   "s": 8
  },
  {
   "t": "    with torch.no_grad():",
   "s": 8
  },
  {
   "t": "        b = tok([prefix + t for t in texts], padding=True, return_tensors=\"pt\")",
   "s": 8
  },
  {
   "t": "        m = b[\"attention_mask\"][..., None].float()",
   "s": 8
  },
  {
   "t": "        return F.normalize((retriever(**b).last_hidden_state * m).sum(1) / m.sum(1), dim=-1)",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "doc_vec = embed(docs, \"passage: \")",
   "s": 8
  },
  {
   "t": "found = (embed([r[\"question\"] for r in search], \"query: \") @ doc_vec.T).argmax(1).tolist()  # 질문마다 가장 가까운 메시지",
   "s": 8
  },
  {
   "t": "read = answer([{\"question\": r[\"question\"], \"context\": docs[k]} for r, k in zip(search, found)])",
   "s": 8
  },
  {
   "t": "search_hits = {\"찾기\": sum(docs[k] == r[\"context\"] for r, k in zip(search, found)) / len(search),",
   "s": 8
  },
  {
   "t": "               \"찾고 읽기\": sum(a == r[\"answer\"] for r, a in zip(search, read)) / len(search)}",
   "s": 8
  }
 ],
 "names": {
  "1": "데이터 읽기",
  "2": "토큰과 라벨 맞추기",
  "3": "토큰마다 분류하는 모델",
  "4": "개체 단위로 채점하기",
  "5": "글자로 꺼내기: 끝 토큰에 붙은 조사 떼기",
  "6": "다른 언어로 옮기기: 한국어로만 배운 모델에 영어 문장",
  "7": "질문 답변: 답의 시작과 끝 찾기",
  "8": "찾아서 읽기: 메시지 여러 개에서 먼저 찾고, 찾은 메시지에서 답 꺼내기"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "data": {
   "title": "예제 데이터에서 몇 줄",
   "cols": [
    "파일",
    "내용"
   ],
   "rows": [
    [
     "ner.jsonl",
     "<b>[서지아]</b><small>사람</small>님이 <b>[1월 말]</b><small>날짜</small>에 <b>[하늘항공]</b><small>회사</small> 방문 예정입니다."
    ],
    [
     "ner.jsonl",
     "<b>[정우진]</b><small>사람</small>님이 <b>[2026년 3월 25일]</b><small>날짜</small>에 <b>[푸른제약]</b><small>회사</small> 방문 예정입니다."
    ],
    [
     "ner.jsonl (영어)",
     "<b>[Emily Stone]</b><small>사람</small> will visit <b>[Hanbit Trading]</b><small>회사</small> on <b>[March 5]</b><small>날짜</small>."
    ],
    [
     "qa.jsonl",
     "메시지: 윤지민님이 1월 말에 다온식품과 계약을 맺었습니다. 계약금 9억 원은 2026년 5월 6일까지 입금되고, 잔금 612만 원은 한예린님이 처리합니다.<br>질문: 계약은 언제 맺었나요? → 답: <b>1월 말</b>"
    ]
   ],
   "notes": [
    "개체명 인식(NER)은 문장 속 <b>어느 글자부터 어느 글자까지가 어떤 종류의 이름인지</b>를 찾아요. 문장 하나에 라벨 하나가 아니라, 글자 범위마다 라벨이 있어요.",
    "질문 답변은 메시지 안에서 <b>질문의 답이 되는 글자 범위</b>를 찾아요. 답이 메시지에 없으면 \"답 없음\"이라고 해야 해요."
   ]
  },
  "why": {
   "title": "이 쪽에서 만드는 것과 처음 보는 데이터에서의 결과",
   "cols": [
    "무엇을",
    "어떻게 쟀나",
    "결과",
    "단계"
   ],
   "rows": [
    [
     "한국어 메시지의 개체 찾기",
     "처음 보는 말투와 이름 40문장, 개체 단위 F1",
     "0.871",
     "③④"
    ],
    [
     "찾은 개체를 글자 그대로 꺼내기",
     "정답 개체와 글자까지 같은 비율",
     "0.70 → <b>0.89</b> (조사 떼기)",
     "⑤"
    ],
    [
     "영어 문장의 개체 찾기",
     "영어는 학습에 한 번도 안 씀, 개체 단위 F1",
     "0.742",
     "⑥"
    ],
    [
     "메시지에서 질문의 답 꺼내기",
     "처음 보는 이름과 질문 말투 96개, 정확히 일치",
     "0.802",
     "⑦"
    ],
    [
     "메시지 12개에서 찾아서 답하기",
     "검색 + 답 꺼내기, 정확히 일치",
     "1.00",
     "⑧"
    ]
   ],
   "notes": [
    "<a href=\"text-classification.html\">문장 분류</a>에서는 문장 벡터 하나로 답 하나를 냈어요. 이 쪽에서는 e5가 내는 <b>토큰마다의 벡터</b>를 하나하나 써요. 같은 몸통에 얹는 층만 바뀌어요.",
    "숫자보다 중요한 건 실패의 모양이에요. 한국어는 조사가 이름에 붙어서 토큰 경계와 이름 경계가 어긋나고(⑤), 다른 언어로 옮기면 경계가 한두 글자씩 틀려요(⑥)."
   ]
  },
  "s1": {
   "title": "① 데이터",
   "lines": [
    "train → 240문장",
    "valid → 48문장",
    "test → 40문장",
    "en_test → 48문장",
    "train 개체 수 → {'PER': 160, 'DATE': 140, 'ORG': 160, 'MONEY': 100}",
    [
     "split['train'][0]['text'] → 서지아님이 1월 말에 하늘항공 방문 예정입니다.",
     1
    ],
    [
     "split['train'][0]['entities'] → [[0, 3, 'PER'], [6, 10, 'DATE'], [12, 16, 'ORG']]",
     2,
     "newf"
    ],
    [
     "TAGS → ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-DATE', 'I-DATE', 'B-MONEY', 'I-MONEY']",
     3,
     "newf"
    ]
   ],
   "notes": [
    "메시지 문장 하나예요.",
    "개체마다 [시작 글자, 끝 글자, 종류]예요. 끝은 포함하지 않아요(파이썬 슬라이스처럼). <code>text[0:3]</code>이 사람 이름이에요.",
    "라벨 9개: 개체가 아니면 <code>O</code>, 개체의 첫 토큰은 <code>B-종류</code>, 나머지는 <code>I-종류</code>예요(BIO 방식). B와 I를 나누는 건, 같은 종류의 개체 두 개가 붙어 있을 때 어디서 나뉘는지 알기 위해서예요."
   ]
  },
  "s2": {
   "title": "② encode_ner(train 첫 문장)",
   "lines": [
    [
     "offset_mapping → [[0, 0], [0, 1], [1, 2], [2, 3], [3, 5], [6, 7], [7, 8], [9, 10]] …",
     1
    ],
    [
     "<s>/- ▁서/B-PER 지/I-PER 아/I-PER 님이/O ▁1/B-DATE 월/I-DATE ▁말/I-DATE 에/O ▁하늘/B-ORG 항공/I-ORG ▁방문/O ▁예정/O 입니다/O ./O </s>/-",
     2,
     "newf"
    ]
   ],
   "notes": [
    "토큰마다 원래 글자에서의 (시작, 끝) 위치예요. <code>&lt;s&gt;</code>와 <code>&lt;/s&gt;</code>는 (0, 0)이에요. 이 위치를 개체의 글자 범위와 겹쳐 보고 라벨을 정해요.",
    "토큰/라벨이에요. \"서지아\"는 토큰 셋(<code>▁서</code>, <code>지</code>, <code>아</code>)으로 나뉘어 B, I, I가 됐고, 표시용 토큰은 <code>-</code>(-100, 채점에서 빼는 자리)예요."
   ]
  },
  "t2": {
   "title": "같은 문장을 두 Tokenizer로 나누면",
   "cols": [
    "Tokenizer",
    "토큰"
   ],
   "rows": [
    [
     "e5 (SentencePiece, XLM-R 계열)",
     "▁서 지 아 님이 ▁1 월 ▁말 에 ▁하늘 항공 ▁방문 ▁예정 입니다 ."
    ],
    [
     "mBERT (WordPiece)",
     "서 ##지 ##아 ##님 ##이 1월 말 ##에 하 ##늘 ##항 ##공 방 ##문 예 ##정 ##입 ##니다 ."
    ]
   ],
   "notes": [
    "SentencePiece는 띄어쓰기도 글자로 보고 <code>▁</code>로 표시해서, 띄어쓰기가 없는 언어(중국어, 일본어)에도 똑같이 쓸 수 있어요. WordPiece는 먼저 띄어쓰기로 나눈 뒤 단어 안을 쪼개고, 이어지는 조각에 <code>##</code>을 붙여요.",
    "Tokenizer는 보통 네 단계예요: 글자 정리(유니코드 통일, 소문자 등), 1차로 나누기(띄어쓰기, 문장 부호), 조각 나누기(BPE, WordPiece, Unigram), 앞뒤 표시 붙이기(<code>&lt;s&gt;</code>). 모델마다 짝인 Tokenizer가 정해져 있어서 바꿔 쓸 수 없어요."
   ]
  },
  "s3": {
   "title": "③ 토큰마다 분류 (12층 미세조정, 6 epoch)",
   "lines": [
    [
     "tagger(b0).shape → (1, 16, 9)",
     1
    ],
    [
     "epoch마다 마지막 묶음 손실 → [1.376, 0.56, 0.283, 0.112, 0.067, 0.036]",
     2
    ],
    "고치는 가중치 → 21,444,873개"
   ],
   "notes": [
    "(문장 1, 토큰 16, 라벨 9)예요. 문장 분류는 문장마다 점수 3개였는데, 여기서는 토큰마다 점수 9개예요. 같은 384 → 9 층을 모든 토큰에 똑같이 써요.",
    "train 문장 240개로 손실이 꾸준히 내려가요. 문장 분류(⑤)에서는 12층 전부를 고치면 말투를 외워 버렸는데, 여기서는 문장 하나에 라벨이 토큰 수만큼 있어서 배울 신호가 훨씬 많아요. 12층을 다 고쳐도 처음 보는 말투에서 잘 맞혀요(④)."
   ]
  },
  "s4": {
   "title": "④ 채점 (토큰 정확도 vs 개체 단위)",
   "cols": [
    "데이터",
    "토큰 정확도",
    "개체 precision",
    "개체 recall",
    "개체 F1"
   ],
   "rows": [
    [
     "valid (같은 말투)",
     "1.000",
     "1.000",
     "1.000",
     "<b>1.000</b>"
    ],
    [
     "test (처음 보는 말투와 이름)",
     "0.951",
     "0.853",
     "0.890",
     "<b>0.871</b>"
    ]
   ],
   "notes": [
    "토큰 정확도가 더 높게 나와요. test 토큰의 55%가 <code>O</code>라서, 개체를 하나도 못 찾아도 정확도가 꽤 나와요. 그래서 NER은 <b>개체 단위</b>로 채점해요.",
    "개체 단위는 시작, 끝, 종류가 모두 맞아야 하나로 쳐요. \"한빛상사\"의 한 토큰만 틀려도 그 개체는 틀린 거예요. precision은 찾은 것 중 맞은 비율, recall은 있어야 할 것 중 찾은 비율이에요. seqeval이라는 라이브러리가 이 방식으로 재 줘요."
   ]
  },
  "s5": {
   "title": "⑤ 정답 개체를 글자까지 똑같이 꺼낸 비율",
   "cols": [
    "데이터",
    "토큰 단위 개체 F1 (④)",
    "글자 그대로",
    "끝 조사 떼기"
   ],
   "rows": [
    [
     "valid",
     "1.000",
     "0.830",
     "<b>0.991</b>"
    ],
    [
     "test",
     "0.871",
     "0.700",
     "<b>0.889</b>"
    ]
   ],
   "notes": [
    "valid는 토큰 단위로는 다 맞았는데(F1 1.0), 글자로 꺼내면 83%만 정답과 같아요. 예를 들어 test의 \"별빛미디어 쪽에서 7월 10일에 미팅 가능하냐고 물어봤어요.\"에서 <code>7월 10일에</code>가 꺼내졌어요.",
    "e5의 Tokenizer가 개체의 끝 글자와 조사를 한 토큰으로 묶는 경우가 있어서예요. 토큰 하나는 라벨 하나만 가지니, 그 토큰이 개체에 들어가면 조사까지 따라와요. 한국어처럼 조사가 붙는 언어에서 늘 생기는 일이에요.",
    "끝에서 흔한 조사를 하나 떼는 것만으로 0.83에서 0.99까지 올라요. 실제로는 형태소 분석기로 조사를 미리 나누거나, 글자 단위로 라벨을 붙이기도 해요."
   ]
  },
  "s6": {
   "title": "⑥ 한국어로만 배운 모델로 영어 문장 48개",
   "lines": [
    [
     "토큰 정확도 0.905, 개체 F1 0.742 (precision 0.695, recall 0.795)",
     1,
     "newf"
    ],
    "\"Emily Stone will visit Hanbit Trading on March 5.\" → [('Emily Stone', 'PER'), ('Hanbit Trading', 'ORG'), ('March 5.', 'DATE')]",
    "\"James Carter will visit Northwind Logistics on next Tuesday.\" → [('James Carter', 'PER'), ('Northwind Logistic', 'ORG'), ('next Tuesday', 'DATE')]",
    "\"Emily Stone will visit Hanbit Trading on November 20.\" → [('Emily Stone', 'PER'), ('Hanbit Trading', 'ORG'), ('November 20.', 'DATE')]",
    "\"Emily Stone will visit Northwind Logistics on November 20.\" → [('Emily Stone', 'PER'), ('Northwind Logistic', 'ORG'), ('November 20.', 'DATE')]"
   ],
   "notes": [
    "영어 문장은 학습에 한 번도 안 썼는데 개체를 0.74만큼 찾아요. e5는 100여 개 언어의 글로 함께 학습해서, \"사람 이름 자리\", \"날짜 자리\"를 언어와 상관없이 비슷한 벡터로 표현해요. 한 언어로 배운 걸 다른 언어에 쓰는 걸 <b>언어 간 전이</b>(zero-shot cross-lingual transfer)라고 해요.",
    "틀린 모양을 보면 경계가 한두 글자씩 어긋나요. 마침표가 붙거나 마지막 조각이 빠져요. 이걸 줄이려면 라벨이 있는 영어 문장을 조금 섞어 학습하는 게 보통의 다음 단계예요."
   ]
  },
  "f_span": {
   "title": "메시지의 토큰마다 \"답의 시작일 확률\"과 \"끝일 확률\"",
   "images": [
    [
     "img/tokens/span_probs.png",
     "test 질문 하나"
    ]
   ],
   "notes": [
    "모델은 토큰마다 점수 두 개(시작, 끝)를 내요. softmax를 하면 거의 한 토큰에 몰려요. 시작이 가장 높은 토큰부터 끝이 가장 높은 토큰까지가 답이에요. 여기서는 끝 토큰이 <code>일에</code>라서 잘라 낸 글은 <code>2026년 4월 16일에</code>이고, ⑤의 조사 떼기로 <code>2026년 4월 16일</code>이 됐어요."
   ],
   "chart": true
  },
  "s7": {
   "title": "⑦ 질문 답변 (12층 미세조정, 2 epoch)",
   "lines": [
    [
     "epoch마다 마지막 묶음 손실 → [0.548, 0.246]",
     1
    ],
    "valid: 정확히 일치 0.993 (답 있는 질문 0.992, 답 없는 질문 1.000)",
    "test: 정확히 일치 0.802 (답 있는 질문 0.736, 답 없는 질문 1.000)",
    [
     "test에서 가장 많이 틀린 질문 → \"비행기 표 가격이 얼마예요?\" 0/6, \"첫 출근 날짜는요?\" 0/6, \"택배 회사는 어디예요?\" 1/6",
     2,
     "bad"
    ]
   ],
   "notes": [
    "train 질문 870개로 2 epoch만 학습했어요.",
    "질문을 처음 보는 말투로 바꾸면 떨어져요. train에는 \"계약은 언제 맺었나요?\"만 있고 \"계약한 날짜가 언제예요?\"는 없었어요. 같은 뜻의 질문을 여러 말투로 넣어 학습해야 해요. 답 없는 질문은 모두 \"답 없음\"이라고 맞혔어요."
   ]
  },
  "s8": {
   "title": "⑧ 메시지 12개에서 찾고 읽기",
   "lines": [
    [
     "맞는 메시지를 찾은 비율 → 1.00",
     1
    ],
    [
     "찾고 읽어서 답까지 맞힌 비율 → 1.00",
     2,
     "newf"
    ],
    "\"세움건설과 맺은 계약의 잔금은 누가 처리하나요?\" → 조현우",
    "\"서지아님 출장의 항공권은 얼마였나요?\" → 753만 원",
    "\"한빛상사에서 산 노트북은 언제 배송되나요?\" → 8월 10일"
   ],
   "notes": [
    "질문에 든 회사나 사람 이름 덕분에, e5 문장 벡터만으로 12개 중 맞는 메시지를 찾았어요.",
    "찾은 메시지에 ⑦의 모델을 그대로 써서 답을 꺼냈어요. 검색(Retriever)과 읽기(Reader)를 따로 만들어 이어 붙인 구조예요."
   ]
  }
 },
 "flow": [
  [
   1,
   "데이터",
   "글자 범위 + 종류",
   "BIO 라벨 9개"
  ],
  [
   2,
   "라벨 맞추기",
   "offset_mapping",
   "토큰마다 라벨"
  ],
  [
   3,
   "토큰 분류",
   "384 → 9",
   "12층 미세조정"
  ],
  [
   4,
   "채점",
   "개체 단위",
   "F1 0.87"
  ],
  [
   5,
   "꺼내기",
   "조사 떼기",
   "0.70 → 0.89"
  ],
  [
   6,
   "영어로",
   "언어 간 전이",
   "F1 0.74"
  ],
  [
   7,
   "질문 답변",
   "시작, 끝",
   "0.80"
  ],
  [
   8,
   "찾고 읽기",
   "검색 + 읽기",
   "1.00"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, transformers 5.22.2, intfloat/multilingual-e5-small, CPU"
};
