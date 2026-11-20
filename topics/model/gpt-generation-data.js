window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "from collections import Counter",
   "s": 1
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 3
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 8
  },
  {
   "t": "from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers",
   "s": 2
  },
  {
   "t": "from transformers import AutoTokenizer, AutoModel",
   "s": 2
  },
  {
   "t": "from transformers import GPT2Config, GPT2LMHeadModel",
   "s": 4
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
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 말뭉치 읽고 정리하기",
   "s": 1
  },
  {
   "t": "days = [json.loads(line) for line in (DATA / \"notes_corpus.jsonl\").open(encoding=\"utf-8\")]",
   "s": 1
  },
  {
   "t": "paras = [p for d in days for p in d[\"text\"].splitlines()]  # 한 줄 = 한 문단",
   "s": 1
  },
  {
   "t": "dup = Counter(paras)",
   "s": 1
  },
  {
   "t": "unique = list(dict.fromkeys(paras))  # 같은 문단은 처음 하나만 (순서 유지)",
   "s": 1
  },
  {
   "t": "train_txt = [p for i, p in enumerate(unique) if i % 20 != 0]  # 문단 20개 중 19개는 학습",
   "s": 1
  },
  {
   "t": "valid_txt = [p for i, p in enumerate(unique) if i % 20 == 0]  # 1개는 처음 보는 글로 남겨 두기",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② Tokenizer 학습하기",
   "s": 2
  },
  {
   "t": "def train_tokenizer(texts, vocab_size):",
   "s": 2
  },
  {
   "t": "    tk = Tokenizer(models.BPE(unk_token=\"<unk>\"))",
   "s": 2
  },
  {
   "t": "    tk.normalizer = normalizers.NFC()  # 한글 자모가 따로 저장된 글자를 하나로",
   "s": 2
  },
  {
   "t": "    tk.pre_tokenizer = pre_tokenizers.Metaspace()  # 띄어쓰기를 ▁로 바꿔 글자처럼 (SentencePiece 방식)",
   "s": 2
  },
  {
   "t": "    tk.decoder = decoders.Metaspace()",
   "s": 2
  },
  {
   "t": "    tk.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[\"<unk>\", \"<eos>\"], min_frequency=2))",
   "s": 2
  },
  {
   "t": "    return tk",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "tk = train_tokenizer(train_txt, 4000)",
   "s": 2
  },
  {
   "t": "EOS = tk.token_to_id(\"<eos>\")  # 문단 끝 표시",
   "s": 2
  },
  {
   "t": "e5_tok = AutoTokenizer.from_pretrained(\"intfloat/multilingual-e5-small\")  # 비교용: 100개 언어 25만 어휘",
   "s": 2
  },
  {
   "t": "valid_all = \" \".join(valid_txt)",
   "s": 2
  },
  {
   "t": "chars_per_token = {\"우리 Tokenizer (4천 개)\": len(valid_all) / len(tk.encode(valid_all).ids),",
   "s": 2
  },
  {
   "t": "                   \"e5 Tokenizer (25만 개)\": len(valid_all) / len(e5_tok.tokenize(valid_all))}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 학습 데이터 만들기: 이어 붙이고 같은 길이로 자르기",
   "s": 3
  },
  {
   "t": "BLOCK = 64  # 모델이 한 번에 보는 토큰 수",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def to_blocks(texts):",
   "s": 3
  },
  {
   "t": "    ids = [i for t in texts for i in tk.encode(t).ids + [EOS]]  # 문단마다 끝에 <eos>를 붙여 한 줄로",
   "s": 3
  },
  {
   "t": "    return torch.tensor(ids[: len(ids) // BLOCK * BLOCK]).view(-1, BLOCK)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "X_train, X_valid = to_blocks(train_txt), to_blocks(valid_txt)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 작은 GPT 만들기 (무작위에서 시작)",
   "s": 4
  },
  {
   "t": "config = GPT2Config(vocab_size=tk.get_vocab_size(), n_positions=BLOCK, n_embd=128, n_layer=4, n_head=4,",
   "s": 4
  },
  {
   "t": "                    bos_token_id=EOS, eos_token_id=EOS)",
   "s": 4
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 4
  },
  {
   "t": "gpt = GPT2LMHeadModel(config)  # 구조만 GPT-2, 가중치는 무작위",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def valid_loss():",
   "s": 4
  },
  {
   "t": "    gpt.eval()",
   "s": 4
  },
  {
   "t": "    with torch.no_grad():",
   "s": 4
  },
  {
   "t": "        return sum(gpt(b, labels=b).loss.item() for b in X_valid.split(64)) / len(X_valid.split(64))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "start_loss = valid_loss()  # 아무것도 안 배웠을 때",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 학습하기: 다음 토큰 맞히기",
   "s": 5
  },
  {
   "t": "STEPS = 1000",
   "s": 5
  },
  {
   "t": "opt = torch.optim.AdamW(gpt.parameters(), lr=2e-3, weight_decay=0.1)",
   "s": 5
  },
  {
   "t": "sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)  # 처음엔 키우고 점점 줄이기",
   "s": 5
  },
  {
   "t": "log = []",
   "s": 5
  },
  {
   "t": "for step in range(1, STEPS + 1):",
   "s": 5
  },
  {
   "t": "    gpt.train()",
   "s": 5
  },
  {
   "t": "    batch = X_train[torch.randint(0, len(X_train), (32,))]  # 무작위 32줄",
   "s": 5
  },
  {
   "t": "    loss = gpt(batch, labels=batch).loss  # labels를 한 칸 밀어서 다음 토큰과 비교하는 건 모델이 해요",
   "s": 5
  },
  {
   "t": "    opt.zero_grad()",
   "s": 5
  },
  {
   "t": "    loss.backward()",
   "s": 5
  },
  {
   "t": "    torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)",
   "s": 5
  },
  {
   "t": "    opt.step()",
   "s": 5
  },
  {
   "t": "    sched.step()",
   "s": 5
  },
  {
   "t": "    if step % 200 == 0:",
   "s": 5
  },
  {
   "t": "        log.append({\"step\": step, \"train\": loss.item(), \"valid\": valid_loss()})",
   "s": 5
  },
  {
   "t": "gpt.eval()",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 고르는 방법 1: 가장 그럴듯한 것 고르기 (greedy, beam)",
   "s": 6
  },
  {
   "t": "def hangul_only(text):",
   "s": 6
  },
  {
   "t": "    return all(\"가\" <= ch <= \"힣\" or ch == \" \" for ch in text)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "starts = Counter(\" \".join(p.split()[:3]) for p in unique)  # 문단 시작 세 어절이 몇 번 나오나",
   "s": 6
  },
  {
   "t": "prompts = [h for h in (\" \".join(p.split()[:3]) for p in valid_txt) if hangul_only(h) and starts[h] == 1][:6]  # 처음 보는 문단의 앞 세 어절 (한글만, 상투적인 시작은 빼고)",
   "s": 6
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def generate(prompt, **how):  # 앞부분을 주고 30토큰 이어 쓰기",
   "s": 6
  },
  {
   "t": "    inp = torch.tensor([tk.encode(prompt).ids])",
   "s": 6
  },
  {
   "t": "    out = gpt.generate(inp, max_new_tokens=30, pad_token_id=EOS, eos_token_id=None, **how)[0]",
   "s": 6
  },
  {
   "t": "    return out[inp.shape[1]:].tolist()",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def repeat_rate(ids):  # 이어 쓴 부분에서 앞에 이미 나온 3-토큰 묶음의 비율",
   "s": 6
  },
  {
   "t": "    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]",
   "s": 6
  },
  {
   "t": "    return sum(g in grams[:i] for i, g in enumerate(grams)) / len(grams)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def avg_nll(prompt, ids):  # 모델이 보기에 이어 쓴 토큰들이 얼마나 뜻밖인가 (낮을수록 그럴듯)",
   "s": 6
  },
  {
   "t": "    full = torch.tensor([tk.encode(prompt).ids + ids])",
   "s": 6
  },
  {
   "t": "    with torch.no_grad():",
   "s": 6
  },
  {
   "t": "        logp = gpt(full).logits[0, :-1].log_softmax(-1)",
   "s": 6
  },
  {
   "t": "    n = len(ids)",
   "s": 6
  },
  {
   "t": "    return -logp[-n:].gather(1, full[0, -n:, None]).mean().item()",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "METHODS = {\"greedy\": dict(do_sample=False), \"beam 5\": dict(do_sample=False, num_beams=5),",
   "s": 6
  },
  {
   "t": "           \"beam 5 + 3-gram 반복 금지\": dict(do_sample=False, num_beams=5, no_repeat_ngram_size=3)}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 고르는 방법 2: 확률대로 뽑기 (temperature, top-k, top-p)",
   "s": 7
  },
  {
   "t": "METHODS.update({\"sampling t=1.0\": dict(do_sample=True, top_k=0, temperature=1.0), \"sampling t=0.7\": dict(do_sample=True, top_k=0, temperature=0.7),",
   "s": 7
  },
  {
   "t": "                \"top-k 20\": dict(do_sample=True, top_k=20), \"top-p 0.9\": dict(do_sample=True, top_k=0, top_p=0.9)})",
   "s": 7
  },
  {
   "t": "outputs = {name: [generate(p, **how) for p in prompts] for name, how in METHODS.items()}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def distinct2(seqs):  # 모든 이어 쓰기에서 서로 다른 2-토큰 묶음의 비율 (높을수록 다양)",
   "s": 7
  },
  {
   "t": "    grams = [tuple(s[i:i + 2]) for s in seqs for i in range(len(s) - 1)]",
   "s": 7
  },
  {
   "t": "    return len(set(grams)) / len(grams)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "decode_table = {name: {\"반복\": sum(map(repeat_rate, seqs)) / len(seqs), \"다양성\": distinct2(seqs),",
   "s": 7
  },
  {
   "t": "                       \"뜻밖의 정도\": sum(avg_nll(p, s) for p, s in zip(prompts, seqs)) / len(seqs)} for name, seqs in outputs.items()}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 요약과 평가: 본문에서 대표 문장 고르기와 ROUGE",
   "s": 8
  },
  {
   "t": "pages = [json.loads(line) for line in (DATA / \"pages.jsonl\").open(encoding=\"utf-8\")]",
   "s": 8
  },
  {
   "t": "e5 = AutoModel.from_pretrained(\"intfloat/multilingual-e5-small\").eval()",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "def embed(texts):  # 문장 분류 쪽 ③과 같은 e5 문장 벡터",
   "s": 8
  },
  {
   "t": "    with torch.no_grad():",
   "s": 8
  },
  {
   "t": "        b = e5_tok([\"passage: \" + t for t in texts], padding=True, truncation=True, return_tensors=\"pt\")",
   "s": 8
  },
  {
   "t": "        m = b[\"attention_mask\"][..., None].float()",
   "s": 8
  },
  {
   "t": "        return F.normalize((e5(**b).last_hidden_state * m).sum(1) / m.sum(1), dim=-1)",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "def pieces(text):  # ROUGE용: e5 Tokenizer 조각 (▁만 있는 조각은 빼기)",
   "s": 8
  },
  {
   "t": "    return [p.lstrip(\"▁\") for p in e5_tok.tokenize(text) if p.lstrip(\"▁\")]",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "def rouge(candidate, reference):  # ROUGE-1과 ROUGE-L의 재현율, 정밀도, F1",
   "s": 8
  },
  {
   "t": "    c, r = pieces(candidate), pieces(reference)",
   "s": 8
  },
  {
   "t": "    overlap = sum((Counter(c) & Counter(r)).values())  # 겹치는 조각 수 (ROUGE-1)",
   "s": 8
  },
  {
   "t": "    lcs = [[0] * (len(r) + 1) for _ in range(len(c) + 1)]  # 가장 긴 공통 부분 순서 (ROUGE-L)",
   "s": 8
  },
  {
   "t": "    for i in range(len(c)):",
   "s": 8
  },
  {
   "t": "        for j in range(len(r)):",
   "s": 8
  },
  {
   "t": "            lcs[i + 1][j + 1] = lcs[i][j] + 1 if c[i] == r[j] else max(lcs[i][j + 1], lcs[i + 1][j])",
   "s": 8
  },
  {
   "t": "    out = {}",
   "s": 8
  },
  {
   "t": "    for name, hit in ((\"ROUGE-1\", overlap), (\"ROUGE-L\", lcs[-1][-1])):",
   "s": 8
  },
  {
   "t": "        rec, pre = hit / len(r), hit / len(c)",
   "s": 8
  },
  {
   "t": "        out[name] = {\"재현율\": rec, \"정밀도\": pre, \"F1\": 2 * rec * pre / (rec + pre) if hit else 0.0}",
   "s": 8
  },
  {
   "t": "    return out",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "summaries = []",
   "s": 8
  },
  {
   "t": "for pg in pages:",
   "s": 8
  },
  {
   "t": "    v = embed(pg[\"sentences\"])",
   "s": 8
  },
  {
   "t": "    k = int((v @ F.normalize(v.mean(0), dim=0)).argmax())  # 본문 전체의 평균과 가장 가까운 문장",
   "s": 8
  },
  {
   "t": "    ref = embed([pg[\"summary\"]])[0]",
   "s": 8
  },
  {
   "t": "    summaries.append({\"page\": pg[\"page\"], \"첫 문장\": pg[\"sentences\"][0], \"대표 문장\": pg[\"sentences\"][k], \"정답\": pg[\"summary\"],",
   "s": 8
  },
  {
   "t": "                      \"첫 문장 뜻\": float(v[0] @ ref), \"대표 문장 뜻\": float(v[k] @ ref)})  # 정답 요약과의 코사인 유사도",
   "s": 8
  },
  {
   "t": "rouge_table = {way: {**{m: sum(rouge(s[way], s[\"정답\"])[m][\"F1\"] for s in summaries) / len(summaries) for m in (\"ROUGE-1\", \"ROUGE-L\")},",
   "s": 8
  },
  {
   "t": "                     \"뜻 유사도\": sum(s[way + \" 뜻\"] for s in summaries) / len(summaries)} for way in (\"첫 문장\", \"대표 문장\")}",
   "s": 8
  }
 ],
 "names": {
  "1": "말뭉치 읽고 정리하기",
  "2": "Tokenizer 학습하기",
  "3": "학습 데이터 만들기: 이어 붙이고 같은 길이로 자르기",
  "4": "작은 GPT 만들기 (무작위에서 시작)",
  "5": "학습하기: 다음 토큰 맞히기",
  "6": "고르는 방법 1: 가장 그럴듯한 것 고르기 (greedy, beam)",
  "7": "고르는 방법 2: 확률대로 뽑기 (temperature, top-k, top-p)",
  "8": "요약과 평가: 본문에서 대표 문장 고르기와 ROUGE"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "data": {
   "title": "notes_corpus.jsonl: 옛 Day 노트에서 사람이 쓴 문단만",
   "lines": [
    "Day 19일치, 문단 2,899개, 335,284자",
    "예: 큰 그림 같은 데이터로 숫자를 맞히거나, 종류를 맞히거나…",
    "예: 같은 데이터로도 질문을 두 가지로 만들 수 있어요. 예를 들어 AI 요청 기록이 있다면 \"이 요청은 몇 ms 걸릴까?\"(회귀)…",
    "예: 소요 시간 예측요청이 몇 ms 걸릴지 숫자로 (회귀)…"
   ],
   "notes": [
    "코드, 표, 주소, 마크다운 기호, 이모지는 빼고(<code>tools/make_gpt_corpus.py</code>), 가운뎃점은 노트 규칙대로 쉼표로 바꿨어요. 이 글만 보고 \"이 노트처럼 쓰는\" GPT를 처음부터 만들어요."
   ]
  },
  "why": {
   "title": "같은 GPT, 같은 앞부분 \"왜 비율 특성을\"에 이어 쓰게 해도 고르는 방법에 따라",
   "cols": [
    "고르는 방법",
    "이어 쓴 글 (30토큰)"
   ],
   "rows": [
    [
     "<b>greedy</b>",
     "한 번에 한 번 한 번도 안 눌러가 커져? 한 번 더 잘려? 한 번 더 잘려? 한 번 더"
    ],
    [
     "<b>beam 5</b>",
     "한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번"
    ],
    [
     "<b>top-p 0.9</b>",
     "내면 log()는 &quot;지금 쌓길 수천 lr에 0으로 바뀌지 다시 움직여요. 그래서 손실은 현장에 덜 출렁건만"
    ]
   ],
   "notes": [
    "greedy와 beam은 매번 \"가장 그럴듯한\" 토큰을 골라서 같은 말을 되풀이해요. 확률대로 뽑으면 반복은 거의 사라지지만 엉뚱한 말이 섞여요.",
    "모델이 똑같아도 다음 토큰을 <b>어떻게 고르느냐</b>(디코딩)에 따라 글이 완전히 달라져요. 이 쪽에서는 이 노트의 글로 작은 GPT를 직접 만들고, 고르는 방법을 숫자로 비교해요."
   ]
  },
  "s1": {
   "title": "① 말뭉치 정리",
   "lines": [
    [
     "문단 2,899개 → 같은 문단을 빼면 2,718개",
     1,
     "newf"
    ],
    "  19번 나온 문단: \"명세 한눈에 보는 참조표\"",
    "  19번 나온 문단: \"실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 \"",
    "  13번 나온 문단: \"실제로 돌려 보면 실제 PostgreSQL 18(연습용 DB)에서 실행\"",
    [
     "train 2,582문단, valid 136문단",
     2
    ]
   ],
   "notes": [
    "같은 문단이 181번 겹쳐 있었어요. 노트마다 붙인 상투적인 제목과 안내 문장이에요. 그대로 두면 모델이 이 문장들을 통째로 외워서, 글을 만들 때 자꾸 꺼내요. 큰 말뭉치를 만들 때도 중복 제거가 가장 먼저 하는 정리예요.",
    "문단 20개 중 하나를 valid로 남겨, 학습에 안 쓴 글에서 얼마나 맞히는지 재요."
   ]
  },
  "s2": {
   "title": "② Tokenizer",
   "lines": [
    [
     "어휘 수 → 4,000",
     1
    ],
    [
     "우리 Tokenizer → ['▁같은', '▁데이터로', '도', '▁질문을', '▁두', '▁가지', '로', '▁만들', '▁수', '▁있어요.']",
     2
    ],
    "e5 Tokenizer → ['▁같은', '▁데이터', '로', '도', '▁질문', '을', '▁두', '▁가지', '로', '▁만들', '▁수', '▁있어요', '.']",
    [
     "valid 글에서 토큰 하나가 평균 몇 글자인가 → 우리 Tokenizer (4천 개) 2.31, e5 Tokenizer (25만 개) 1.96",
     3,
     "newf"
    ]
   ],
   "notes": [
    "4,000개에서 <code>&lt;unk&gt;</code>, <code>&lt;eos&gt;</code> 두 개는 특별 토큰이에요.",
    "이 노트에 자주 나오는 \"데이터로\", \"질문을\", \"있어요.\"가 통째로 한 토큰이 됐어요. BPE는 가장 자주 붙어 나오는 조각 쌍을 하나로 합치기를 어휘 수가 찰 때까지 되풀이해서 어휘를 만들어요.",
    "어휘가 60분의 1인데 이 글을 더 짧게 나눠요(2.31자 대 1.96자). e5는 100개 언어에 어휘를 나눠 써야 하지만, 우리 것은 이 노트의 말투에만 맞췄기 때문이에요. 토큰이 적으면 같은 글을 더 짧게 넣을 수 있어 학습과 생성이 빨라져요."
   ]
  },
  "t2": {
   "title": "어휘 수를 바꾸면",
   "cols": [
    "vocab_size",
    "실제 어휘 수",
    "토큰당 글자 (valid)",
    "예문을 나누면"
   ],
   "rows": [
    [
     "1,000",
     "1,000",
     "1.03",
     "▁ 같 은 ▁ 데 이 터 로 도 ▁ 질 문 을 ▁ 두 ▁ 가 지 로 ▁ 만 들 ▁ 수 ▁ 있 어 요."
    ],
    [
     "<b>4,000</b> (지금)",
     "4,000",
     "2.31",
     "▁같은 ▁데이터로 도 ▁질문을 ▁두 ▁가지 로 ▁만들 ▁수 ▁있어요."
    ],
    [
     "16,000",
     "15,460",
     "3.07",
     "▁같은 ▁데이터로 도 ▁질문을 ▁두 ▁가지 로 ▁만들 ▁수 ▁있어요."
    ]
   ],
   "notes": [
    "어휘가 클수록 긴 조각을 통째로 갖고 있어서 토큰 수가 줄어요. 대신 Embedding 표(어휘 수 × 칸 수)가 커지고, 드물게 나오는 토큰은 학습할 기회가 적어요.",
    "실제 어휘가 정한 수보다 적을 수 있어요. 두 번 넘게 나온 쌍만 합치게 했는데(<code>min_frequency=2</code>), 34만 자에서는 합칠 쌍이 먼저 떨어져요."
   ]
  },
  "s3": {
   "title": "③ 학습용 묶음",
   "lines": [
    [
     "X_train.shape → (2083, 64), X_valid.shape → (114, 64)",
     1
    ],
    [
     "X_train[0][:8] → [2465, 2383, 276, 1286, 944, 9, 1657, 3855]",
     2
    ],
    "  = ['▁선형', '회귀', '는', '▁가중', '합', ',', '▁로', '지스틱']",
    [
     "입력 ['▁선형', '회귀', '는', '▁가중'] → 맞힐 것 ['회귀', '는', '▁가중', '합']",
     3,
     "newf"
    ]
   ],
   "notes": [
    "train 글 전체를 한 줄로 이어 붙인 뒤 64토큰씩 잘라 2,083줄을 만들었어요. 문단이 줄 중간에서 끝나도 괜찮아요. <code>&lt;eos&gt;</code>가 \"여기서 문단이 끝났다\"를 알려 줘요.",
    "한 줄은 토큰 번호 64개예요.",
    "정답은 따로 만들지 않아요. 같은 줄을 한 칸씩 밀면 \"지금까지 본 토큰 → 다음 토큰\" 쌍이 64개 생겨요. 라벨이 필요 없어서(자기 지도 학습) 글만 있으면 얼마든지 학습할 수 있어요."
   ]
  },
  "s4": {
   "title": "④ 무작위에서 시작한 작은 GPT",
   "lines": [
    [
     "가중치 수 → 1,313,536개 (그중 토큰과 위치 Embedding 520,192개)",
     1
    ],
    [
     "학습 전 valid 손실 → 8.319 (ln 4000 = 8.294)",
     2,
     "newf"
    ],
    "학습 전에 \"왜 비율 특성을\" 뒤를 이어 쓰면 → 려면망 오답곱 일이에요. 않아 사진은 후보를ab CNN 보여 설정을illu 대신 잡(F브러리으니 않아요."
   ],
   "notes": [
    "GPT-2의 가장 작은 공개 모델(약 1억 2천만 개)의 90분의 1쯤이에요. 층 4개, 칸 128개, head 4개예요.",
    "아무것도 모를 때 손실이 ln(4000)과 비슷해요. 토큰 4000개에 똑같은 확률(1/4000)을 주면 손실이 정확히 ln 4000이기 때문이에요. 학습이 시작점을 제대로 잡았는지 확인하는 기준이 돼요."
   ]
  },
  "s5": {
   "title": "⑤ 학습 기록 (1000 걸음, 한 걸음에 32줄)",
   "cols": [
    "걸음",
    "train 손실",
    "valid 손실",
    "valid perplexity"
   ],
   "rows": [
    [
     200,
     "6.174",
     "6.310",
     "550"
    ],
    [
     400,
     "5.262",
     "5.599",
     "270"
    ],
    [
     600,
     "4.605",
     "5.395",
     "220"
    ],
    [
     800,
     "4.300",
     "5.311",
     "203"
    ],
    [
     1000,
     "4.145",
     "5.294",
     "199"
    ]
   ],
   "notes": [
    "valid 손실이 8.32에서 5.29까지 내려갔어요. <b>perplexity</b>는 e^손실로, \"다음 토큰을 평균 몇 개 중에서 헷갈리나\"로 읽어요. 4000개 중에 고르던 게 199개쯤으로 줄었어요.",
    "train 손실(4.15)이 valid보다 낮아지고 있어요. 34만 자는 GPT에게 아주 작은 말뭉치라 이 이상 학습하면 외우기 시작해요. 실제 GPT는 수천억 토큰으로 배워요."
   ]
  },
  "f_loss": {
   "title": "학습 손실",
   "images": [
    [
     "img/gpt/loss_curve.png",
     ""
    ]
   ],
   "notes": [
    "처음 200걸음에 크게 떨어지고, 그 뒤로는 천천히 내려가요. 두 선이 벌어지는 만큼이 외운 정도예요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ \"왜 비율 특성을\" 뒤를 가장 그럴듯하게 이어 쓰면 (앞부분 6개 평균)",
   "cols": [
    "방법",
    "이어 쓴 글 (첫 앞부분)",
    "반복",
    "뜻밖의 정도"
   ],
   "rows": [
    [
     "<b>greedy</b>",
     "한 번에 한 번 한 번도 안 눌러가 커져? 한 번 더 잘려? 한 번 더 잘려? 한 번 더",
     "0.363",
     "2.309"
    ],
    [
     "<b>beam 5</b>",
     "한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번",
     "0.637",
     "1.260"
    ],
    [
     "<b>beam 5 + 3-gram 반복 금지</b>",
     "한 번에 한 번 한 번 눌러나요. 그림: 무엇을 보나: 작은 데모 CNN은 한 번의 한 번 갱신한 뒤 한 장의 평균",
     "0.000",
     "1.768"
    ]
   ],
   "notes": [
    "<b>반복</b>: 이어 쓴 글의 3-토큰 묶음 중 앞에서 이미 나온 것의 비율. <b>뜻밖의 정도</b>: 모델이 보기에 이어 쓴 토큰들이 평균 얼마나 뜻밖인가(토큰당 손실). 낮을수록 모델이 \"그럴듯하다\"고 보는 글이에요.",
    "beam은 greedy보다 더 그럴듯한 글(1.26)을 찾지만 반복도 더 많아요(0.64). 반복하는 글이 모델에게는 오히려 그럴듯하게 보여요. 한 번 나온 말을 다시 쓸 확률이 점점 커지는 고리에 빠져요.",
    "3-토큰 묶음이 두 번 나오지 못하게 막으면 반복은 0이 되지만, 같은 단어를 조금씩 바꿔 되풀이하는 모습은 남아요."
   ]
  },
  "s7": {
   "title": "⑦ 일곱 가지 방법 비교 (앞부분 6개, 30토큰씩)",
   "cols": [
    "방법",
    "\"왜 비율 특성을\" 뒤",
    "반복 ↓",
    "다양성 ↑",
    "뜻밖의 정도"
   ],
   "rows": [
    [
     "greedy",
     "한 번에 한 번 한 번도 안 눌러가 커져? 한 번 더 잘려? 한 번 더 잘려? 한 번 더",
     "0.363",
     "0.517",
     "2.309"
    ],
    [
     "beam 5",
     "한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번에 한 번",
     "0.637",
     "0.178",
     "1.260"
    ],
    [
     "beam 5 + 3-gram 반복 금지",
     "한 번에 한 번 한 번 눌러나요. 그림: 무엇을 보나: 작은 데모 CNN은 한 번의 한 번 갱신한 뒤 한 장의 평균",
     "0.000",
     "0.684",
     "1.768"
    ],
    [
     "<b>sampling t=1.0</b>",
     "따로 한 장곱 일이에요. 사진은 닿는 정하고, 다의 주도 거리로 숫자일수록 늘리면 하나로 정해졌나? 빠더&quot;는",
     "0.000",
     "0.994",
     "4.848"
    ],
    [
     "<b>sampling t=0.7</b>",
     "세 줄고 쪽으로 잘 골라 뭉개면 그릴 수 있어요. 그래서 &quot;순도 나여요. 그래서 클래스&quot;라, 그건(점선이",
     "0.036",
     "0.943",
     "3.334"
    ],
    [
     "<b>top-k 20</b>",
     "그대로 한 번의 클래스 하나여야 해요. 실제로 돌려 보면 예제 데이터 train train 안에서 한 번에 더하면 새 Pipeline을 학습하는 것. valid는 마지막 epoch",
     "0.000",
     "0.977",
     "3.201"
    ],
    [
     "<b>top-p 0.9</b>",
     "내면 log()는 &quot;지금 쌓길 수천 lr에 0으로 바뀌지 다시 움직여요. 그래서 손실은 현장에 덜 출렁건만",
     "0.000",
     "0.994",
     "4.148"
    ]
   ],
   "notes": [
    "<b>다양성</b>: 모든 이어 쓰기에서 2-토큰 묶음 중 서로 다른 것의 비율. 같은 말을 되풀이하면 낮아요.",
    "확률대로 뽑으면 반복이 거의 사라지고 다양성이 1 가까이 올라요. 대신 뜻밖의 정도가 크게 올라요(t=1.0에서 4.85). 확률이 낮은 엉뚱한 토큰도 가끔 뽑히고, 한 번 엉뚱해지면 그 뒤가 계속 흐트러져요.",
    "temperature를 낮추거나(t=0.7, 3.33) 상위 k개만 남기면(top-k 20, 3.20) 엉뚱한 꼬리가 잘려서 덜 뜻밖이면서 반복도 거의 없어요. 대화나 글쓰기에서 흔히 top-p 0.9 안팎, temperature 0.7 안팎에서 시작하는 이유예요.",
    "모델이 작아서 문장의 뜻은 어느 방법이든 엉성해요. 여기서 볼 건 방법마다의 <b>성질</b>이에요. 큰 모델에서도 같은 성질(beam의 반복, 샘플링의 엉뚱함)이 나타난다고 알려져 있어요."
   ]
  },
  "s8": {
   "title": "⑧ 주제 페이지 26쪽의 '한 문장'(사람이 쓴 요약)과 비교",
   "lines": [
    "예: embedding 쪽",
    "  정답 요약 → Embedding은 글을 뜻이 담긴 벡터로 바꾸는 거예요. 뜻이 비슷하면 벡터가 가까워서, 글자가 하나도 겹치지 않아도 질문에 맞는 자료를 찾을 수 있어요.",
    "  첫 문장 → \"개발용 웹 서비스는 어디서 돌리나요?\"라고 물었을 때 찾아야 할 자료는 \"프로젝트 폴더에서 서버를 실행합니다\"예요.",
    "  대표 문장 → 벡터용 인덱스를 만들어, 모든 자료와 비교하지 않고도 가까운 것을 빨리 찾아요.",
    "  첫 문장의 ROUGE-1 → 재현율 0.12, 정밀도 0.17, F1 0.14",
    "  대표 문장의 ROUGE-1 → 재현율 0.14, 정밀도 0.33, F1 0.20",
    "",
    [
     "26쪽 평균 ROUGE-1 F1 → 첫 문장 0.171, 대표 문장 0.183",
     1
    ],
    [
     "26쪽 평균 ROUGE-L F1 → 첫 문장 0.150, 대표 문장 0.148",
     2
    ],
    [
     "정답을 보고 ROUGE-1이 가장 높은 문장을 골라도 → 평균 0.426",
     3,
     "bad"
    ]
   ],
   "notes": [
    "본문에서 문장을 그대로 고르는 <b>추출 요약</b>은 첫 문장이든, 본문 전체와 뜻이 가장 가까운 대표 문장이든 ROUGE-1 F1이 0.17~0.18이에요. 사람이 쓴 요약과 겹치는 조각이 적고, 두 방법의 차이도 거의 없어요.",
    "ROUGE-L은 겹치는 조각이 <b>순서까지</b> 같은 만큼이에요.",
    "정답을 보고 가장 잘 겹치는 문장을 골라도 0.43에 그쳐요. 사람의 요약은 여러 문장의 뜻을 새 문장으로 묶어 쓴 <b>생성 요약</b>이라, 본문 어디에도 그런 문장이 없어요. 그래서 요약에는 글을 새로 쓰는 인코더-디코더 모델(BART, PEGASUS, T5)을 써요."
   ]
  }
 },
 "flow": [
  [
   1,
   "말뭉치",
   "중복 제거",
   "2,899 → 2,718문단"
  ],
  [
   2,
   "Tokenizer",
   "BPE 4천 개",
   "토큰당 2.31자"
  ],
  [
   3,
   "학습 묶음",
   "이어 붙여 자르기",
   "(2083, 64)"
  ],
  [
   4,
   "작은 GPT",
   "무작위 시작",
   "손실 8.32"
  ],
  [
   5,
   "학습",
   "다음 토큰",
   "손실 5.29"
  ],
  [
   6,
   "그럴듯하게",
   "greedy, beam",
   "반복"
  ],
  [
   7,
   "뽑기",
   "temperature, top-k, top-p",
   "다양성"
  ],
  [
   8,
   "요약 재기",
   "ROUGE",
   "추출의 한계"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, transformers 5.22.2, tokenizers 0.23.3, CPU (seed 0)"
};
