window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import re",
   "s": 1
  },
  {
   "t": "import inspect",
   "s": 1
  },
  {
   "t": "import importlib",
   "s": 1
  },
  {
   "t": "import zlib",
   "s": 4
  },
  {
   "t": "from collections import Counter, defaultdict",
   "s": 3
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "from tokenizers import Tokenizer, models, trainers, pre_tokenizers",
   "s": 5
  },
  {
   "t": "from scipy.stats import spearmanr",
   "s": 6
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 세 가지 말뭉치 모으기",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "ko = [p.strip() for line in (HERE.parent / \"gpt\" / \"data\" / \"notes_corpus.jsonl\").open(encoding=\"utf-8\")",
   "s": 1
  },
  {
   "t": "      for p in json.loads(line)[\"text\"].split(\"\\n\") if p.strip()]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "EN_MODULES = [\"os\", \"re\", \"json\", \"collections\", \"itertools\", \"functools\", \"pathlib\", \"typing\", \"string\", \"random\", \"statistics\", \"math\",",
   "s": 1
  },
  {
   "t": "              \"datetime\", \"logging\", \"argparse\", \"subprocess\", \"shutil\", \"csv\", \"sqlite3\", \"http.client\", \"urllib.request\", \"email.message\",",
   "s": 1
  },
  {
   "t": "              \"asyncio\", \"threading\", \"unittest\", \"decimal\", \"fractions\", \"heapq\", \"bisect\", \"textwrap\", \"difflib\", \"inspect\", \"ast\",",
   "s": 1
  },
  {
   "t": "              \"zipfile\", \"tarfile\", \"hashlib\", \"uuid\", \"socket\", \"ssl\", \"xml.etree.ElementTree\", \"pickle\", \"copy\", \"enum\", \"dataclasses\",",
   "s": 1
  },
  {
   "t": "              \"contextlib\", \"io\", \"tempfile\", \"calendar\", \"time\", \"codecs\", \"struct\", \"queue\", \"selectors\", \"signal\", \"base64\", \"mailbox\",",
   "s": 1
  },
  {
   "t": "              \"configparser\", \"doctest\", \"timeit\", \"weakref\", \"types\", \"operator\", \"turtle\", \"tkinter\", \"zlib\", \"lzma\"]",
   "s": 1
  },
  {
   "t": "def docs_of(mod):  # 모듈, 함수, 클래스, 메서드의 설명문을 문단으로",
   "s": 1
  },
  {
   "t": "    m = importlib.import_module(mod)",
   "s": 1
  },
  {
   "t": "    objs = [m] + [o for _, o in inspect.getmembers(m) if inspect.isfunction(o) or inspect.isclass(o) or inspect.isbuiltin(o)]",
   "s": 1
  },
  {
   "t": "    objs += [o for c in objs if inspect.isclass(c) and c.__module__ == m.__name__ for o in vars(c).values() if callable(o)]",
   "s": 1
  },
  {
   "t": "    return [\" \".join(p.split()) for o in objs for p in (inspect.getdoc(o) or \"\").split(\"\\n\\n\")]",
   "s": 1
  },
  {
   "t": "en = [p for mod in EN_MODULES for p in docs_of(mod)]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "CODE_MODULES = [\"json.decoder\", \"json.encoder\", \"textwrap\", \"difflib\", \"heapq\", \"bisect\", \"fractions\", \"statistics\", \"csv\", \"configparser\",",
   "s": 1
  },
  {
   "t": "                \"argparse\", \"dataclasses\", \"pathlib\", \"string\", \"calendar\", \"shutil\", \"glob\", \"fnmatch\", \"queue\", \"sched\", \"tempfile\",",
   "s": 1
  },
  {
   "t": "                \"pprint\", \"copy\", \"enum\", \"functools\", \"base64\", \"uuid\", \"random\", \"ipaddress\", \"gettext\"]",
   "s": 1
  },
  {
   "t": "def code_of(mod):  # 소스에서 설명문과 주석을 빼고 빈 줄로 나눈 덩어리",
   "s": 1
  },
  {
   "t": "    src = re.sub(r'\"\"\".*?\"\"\"|\\'\\'\\'.*?\\'\\'\\'', \"\", inspect.getsource(importlib.import_module(mod)), flags=re.S)",
   "s": 1
  },
  {
   "t": "    lines = [re.sub(r\"\\s+#.*$\", \"\", ln) for ln in src.splitlines() if not ln.lstrip().startswith(\"#\")]",
   "s": 1
  },
  {
   "t": "    return [b.strip(\"\\n\") for b in re.split(r\"\\n\\s*\\n\", \"\\n\".join(lines))]",
   "s": 1
  },
  {
   "t": "code = [b for mod in CODE_MODULES for b in code_of(mod)]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "corpus = {\"한국어\": ko, \"영어\": en, \"코드\": code}",
   "s": 1
  },
  {
   "t": "size = lambda docs: (len(docs), sum(map(len, docs)))  # noqa: E731  (문서 수, 글자 수)",
   "s": 1
  },
  {
   "t": "raw_size = {k: size(v) for k, v in corpus.items()}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 품질 거르기: 규칙 몇 개",
   "s": 2
  },
  {
   "t": "def bad_reason(text, kind):",
   "s": 2
  },
  {
   "t": "    if len(text) < 40:",
   "s": 2
  },
  {
   "t": "        return \"40자 미만\"",
   "s": 2
  },
  {
   "t": "    if kind != \"코드\":",
   "s": 2
  },
  {
   "t": "        letters = sum(ch.isalpha() for ch in text) / len(text)",
   "s": 2
  },
  {
   "t": "        if letters < 0.6:",
   "s": 2
  },
  {
   "t": "            return \"글자 비율 60% 미만\"  # 기호, 숫자, 표 조각 위주",
   "s": 2
  },
  {
   "t": "        words = text.split()",
   "s": 2
  },
  {
   "t": "        if len(words) >= 8 and len(set(words)) / len(words) < 0.5:",
   "s": 2
  },
  {
   "t": "            return \"같은 단어 반복\"",
   "s": 2
  },
  {
   "t": "    return None",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "dropped = Counter()",
   "s": 2
  },
  {
   "t": "drop_examples = {}",
   "s": 2
  },
  {
   "t": "for k in corpus:",
   "s": 2
  },
  {
   "t": "    keep = []",
   "s": 2
  },
  {
   "t": "    for t in corpus[k]:",
   "s": 2
  },
  {
   "t": "        r = bad_reason(t, k)",
   "s": 2
  },
  {
   "t": "        if r is None:",
   "s": 2
  },
  {
   "t": "            keep.append(t)",
   "s": 2
  },
  {
   "t": "        else:",
   "s": 2
  },
  {
   "t": "            dropped[(k, r)] += 1",
   "s": 2
  },
  {
   "t": "            drop_examples.setdefault((k, r), t)",
   "s": 2
  },
  {
   "t": "    corpus[k] = keep",
   "s": 2
  },
  {
   "t": "filtered_size = {k: size(v) for k, v in corpus.items()}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 똑같은 것 빼기",
   "s": 3
  },
  {
   "t": "top_dups = Counter(corpus[\"영어\"]).most_common(3)",
   "s": 3
  },
  {
   "t": "corpus = {k: list(dict.fromkeys(v)) for k, v in corpus.items()}  # 처음 나온 것만 남기기",
   "s": 3
  },
  {
   "t": "exact_size = {k: size(v) for k, v in corpus.items()}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 거의 같은 것 빼기: MinHash와 LSH",
   "s": 4
  },
  {
   "t": "PRIME, H, BANDS = 4294967311, 64, 16  # 해시 64개를 4개씩 16묶음으로",
   "s": 4
  },
  {
   "t": "rng = np.random.default_rng(0)",
   "s": 4
  },
  {
   "t": "ha, hb = rng.integers(1, 2**32, H, dtype=np.uint64), rng.integers(0, 2**32, H, dtype=np.uint64)",
   "s": 4
  },
  {
   "t": "def minhash(text, n=5):  # 단어 5개짜리 조각 집합의 \"지문\" 64칸",
   "s": 4
  },
  {
   "t": "    w = text.split()",
   "s": 4
  },
  {
   "t": "    sh = np.array([zlib.crc32(\" \".join(w[i:i + n]).encode()) for i in range(max(1, len(w) - n + 1))], dtype=np.uint64)",
   "s": 4
  },
  {
   "t": "    return ((np.outer(ha, sh) + hb[:, None]) % np.uint64(PRIME)).min(axis=1)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def near_duplicates(docs, threshold=0.8):",
   "s": 4
  },
  {
   "t": "    sig = np.array([minhash(t) for t in docs])",
   "s": 4
  },
  {
   "t": "    cand = set()",
   "s": 4
  },
  {
   "t": "    for b in range(BANDS):  # 한 묶음이라도 똑같으면 후보 (전부 비교하지 않기)",
   "s": 4
  },
  {
   "t": "        bucket = defaultdict(list)",
   "s": 4
  },
  {
   "t": "        for i, row in enumerate(sig[:, b * 4:(b + 1) * 4]):",
   "s": 4
  },
  {
   "t": "            bucket[row.tobytes()].append(i)",
   "s": 4
  },
  {
   "t": "        cand |= {(x, y) for v in bucket.values() for i, x in enumerate(v) for y in v[i + 1:]}",
   "s": 4
  },
  {
   "t": "    pairs = sorted((float((sig[i] == sig[j]).mean()), i, j) for i, j in cand)  # 지문이 같은 칸의 비율 ≈ 조각 집합의 Jaccard",
   "s": 4
  },
  {
   "t": "    return [(s, i, j) for s, i, j in pairs if s >= threshold], len(cand)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "en_exact = corpus[\"영어\"]",
   "s": 4
  },
  {
   "t": "near, n_cand = near_duplicates(en_exact)",
   "s": 4
  },
  {
   "t": "drop = {j for _, _, j in near}  # 짝마다 뒤의 것을 빼기",
   "s": 4
  },
  {
   "t": "corpus[\"영어\"] = [t for i, t in enumerate(en_exact) if i not in drop]",
   "s": 4
  },
  {
   "t": "clean_size = {k: size(v) for k, v in corpus.items()}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 토크나이저와 압축률",
   "s": 5
  },
  {
   "t": "split = {}",
   "s": 5
  },
  {
   "t": "for k, docs in corpus.items():  # 문서의 80%로 토크나이저를 배우고 20%로 재기",
   "s": 5
  },
  {
   "t": "    idx = np.random.default_rng(1).permutation(len(docs))",
   "s": 5
  },
  {
   "t": "    cut = int(len(docs) * 0.8)",
   "s": 5
  },
  {
   "t": "    split[k] = ([docs[i] for i in idx[:cut]], \"\\n\".join(docs[i] for i in idx[cut:]))",
   "s": 5
  },
  {
   "t": "DOMAINS = list(corpus)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def train_tokenizer(mix, budget=400_000, vocab=4000, seed=0):  # mix: 분야마다 비율, budget: 학습에 쓰는 글자 수",
   "s": 5
  },
  {
   "t": "    docs = []",
   "s": 5
  },
  {
   "t": "    for k, p in zip(DOMAINS, mix):",
   "s": 5
  },
  {
   "t": "        r = np.random.default_rng(seed)",
   "s": 5
  },
  {
   "t": "        total = 0",
   "s": 5
  },
  {
   "t": "        for i in r.permutation(len(split[k][0])):",
   "s": 5
  },
  {
   "t": "            if total >= budget * p:",
   "s": 5
  },
  {
   "t": "                break",
   "s": 5
  },
  {
   "t": "            docs.append(split[k][0][i])",
   "s": 5
  },
  {
   "t": "            total += len(split[k][0][i])",
   "s": 5
  },
  {
   "t": "    tok = Tokenizer(models.BPE())",
   "s": 5
  },
  {
   "t": "    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # 바이트 단위라 모르는 글자가 없어요",
   "s": 5
  },
  {
   "t": "    tok.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))",
   "s": 5
  },
  {
   "t": "    return tok",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def chars_per_token(tok):  # 압축률: 토큰 하나가 평균 몇 글자를 담나 (클수록 좋아요)",
   "s": 5
  },
  {
   "t": "    return np.array([len(split[k][1]) / len(tok.encode(split[k][1]).ids) for k in DOMAINS])",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "natural = np.array([clean_size[k][1] for k in DOMAINS], float)",
   "s": 5
  },
  {
   "t": "natural /= natural.sum()  # 가진 만큼 그대로 섞기",
   "s": 5
  },
  {
   "t": "tok_natural = train_tokenizer(natural)",
   "s": 5
  },
  {
   "t": "cpt_natural = chars_per_token(tok_natural)",
   "s": 5
  },
  {
   "t": "SAMPLE = \"토크나이저는 자주 나오는 글자 조각을 하나의 토큰으로 묶어요.\"",
   "s": 5
  },
  {
   "t": "tok_little_ko = train_tokenizer(np.array([0.02, 0.49, 0.49]))",
   "s": 5
  },
  {
   "t": "pieces = {\"가진 만큼 섞기\": tok_natural.encode(SAMPLE).tokens, \"한국어 2%\": tok_little_ko.encode(SAMPLE).tokens}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 섞는 비율을 바꾸면: 작은 대리 토크나이저로 먼저",
   "s": 6
  },
  {
   "t": "MIXES = np.random.default_rng(2).dirichlet(np.ones(3), 24)  # 비율 조합 24가지",
   "s": 6
  },
  {
   "t": "proxy = np.array([chars_per_token(train_tokenizer(m, budget=100_000, vocab=1000)) for m in MIXES])  # 작고 싸게",
   "s": 6
  },
  {
   "t": "full = np.array([chars_per_token(train_tokenizer(m)) for m in MIXES])  # 실제 크기",
   "s": 6
  },
  {
   "t": "rank_kept = {k: float(spearmanr(proxy[:, j], full[:, j])[0]) for j, k in enumerate(DOMAINS)}  # 순위가 그대로인가",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 회귀로 비율 고르기 (TReX 방식)",
   "s": 7
  },
  {
   "t": "TRAFFIC = np.array([0.6, 0.2, 0.2])  # 서비스에 들어올 글의 비율 (글자 기준): 한국어, 영어, 코드",
   "s": 7
  },
  {
   "t": "def features(m):",
   "s": 7
  },
  {
   "t": "    return np.column_stack([np.ones(len(m)), np.log(np.clip(m, 1e-3, 1))])",
   "s": 7
  },
  {
   "t": "coef = np.linalg.lstsq(features(MIXES), proxy, rcond=None)[0]  # 비율 → 분야마다 압축률 (대리 토크나이저 기준)",
   "s": 7
  },
  {
   "t": "def tokens_per_1k(cpt):  # 서비스 글 1,000자에 드는 토큰 수 (작을수록 좋아요)",
   "s": 7
  },
  {
   "t": "    return float((1000 * TRAFFIC / cpt).sum())",
   "s": 7
  },
  {
   "t": "grid = np.array([(a, b, 1 - a - b) for a in np.arange(0.02, 0.97, 0.02) for b in np.arange(0.02, 0.97, 0.02) if 1 - a - b >= 0.019])",
   "s": 7
  },
  {
   "t": "best_mix = grid[np.argmin([tokens_per_1k(c) for c in features(grid) @ coef])]",
   "s": 7
  },
  {
   "t": "candidates = {\"똑같이 3분의 1씩\": np.ones(3) / 3, \"가진 만큼\": natural, \"서비스 비율 그대로\": TRAFFIC, \"회귀가 고른 비율\": best_mix}",
   "s": 7
  },
  {
   "t": "verdict = {name: (m, chars_per_token(train_tokenizer(m))) for name, m in candidates.items()}  # 실제 크기로 확인",
   "s": 7
  }
 ],
 "names": {
  "1": "세 가지 말뭉치 모으기",
  "2": "품질 거르기: 규칙 몇 개",
  "3": "똑같은 것 빼기",
  "4": "거의 같은 것 빼기: MinHash와 LSH",
  "5": "토크나이저와 압축률",
  "6": "섞는 비율을 바꾸면: 작은 대리 토크나이저로 먼저",
  "7": "회귀로 비율 고르기 (TReX 방식)"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 모은 말뭉치",
   "cols": [
    "분야",
    "어디서",
    "문서(문단) 수",
    "글자 수"
   ],
   "rows": [
    [
     "한국어",
     "옛 Day 노트의 줄",
     "2,899",
     "335,284"
    ],
    [
     "영어",
     "표준 라이브러리 66개 모듈의 설명문",
     "12,364",
     "1,058,766"
    ],
    [
     "코드",
     "표준 라이브러리 30개 모듈의 소스",
     "3,158",
     "594,818"
    ]
   ],
   "notes": [
    "진짜 LLM은 웹 문서, PDF, 코드, 수학을 몇 조 토큰 모아요. 여기는 이 PC에 있는 글로 같은 일을 작게 해요. 영어가 한국어의 세 배쯤이에요."
   ]
  },
  "s2": {
   "title": "② 규칙으로 버린 것",
   "cols": [
    "분야",
    "규칙",
    "버린 수",
    "버린 것의 예"
   ],
   "rows": [
    [
     "한국어",
     "40자 미만",
     "504",
     "큰 그림 같은 데이터로 숫자를 맞히거나, 종류를 맞히거나"
    ],
    [
     "한국어",
     "글자 비율 60% 미만",
     "194",
     "왜 제곱해서 더하지? 잔차는 +도 있고 도 있어서 그냥 더하면 서로 지워져요(+10과 10의 합은 0). 제곱하면 모두 양수가 되고, 크게 빗나간 점일수록 훨씬 큰 벌점(10²=100, 30²=90…"
    ],
    [
     "영어",
     "40자 미만",
     "4,530",
     "(빈 문단)"
    ],
    [
     "영어",
     "글자 비율 60% 미만",
     "386",
     "E.g. for t = list[int], t.__origin__ is list and t.__args__ is (int,)."
    ],
    [
     "영어",
     "같은 단어 반복",
     "30",
     "The special sequences consist of &quot;\\\\&quot; and a character from the list below. If the ordinary character is not on…"
    ],
    [
     "코드",
     "40자 미만",
     "660",
     "import re"
    ]
   ],
   "notes": [
    "짧은 조각, 기호와 숫자 위주 조각, 같은 단어가 되풀이되는 조각을 버렸어요. 한국어는 2,899개에서 2,201개, 영어는 12,364개에서 7,418개가 남았어요.",
    "그런데 한국어에서 \"글자 비율 60% 미만\"으로 버린 194개 중에는 위 예처럼 식이 섞인 좋은 설명도 있어요. 한국어는 띄어쓰기와 문장 부호가 많아 같은 기준이 더 엄해요. 규칙은 분야마다 따로 맞추고, 버린 것을 꼭 직접 읽어 봐야 해요."
   ]
  },
  "t2": {
   "title": "한국어에서 글자 비율 기준을 바꾸면",
   "cols": [
    "기준 (이보다 낮으면 버림)",
    "버리는 한국어 문단 수 (40자 이상 중)"
   ],
   "rows": [
    [
     0.4,
     11
    ],
    [
     0.5,
     35
    ],
    [
     "<b>0.6</b> (지금)",
     194
    ],
    [
     0.7,
     1284
    ]
   ],
   "notes": [
    "기준을 조금만 올려도 버리는 수가 크게 늘어요. 몇 조 토큰짜리 말뭉치에서는 기준 하나가 수십억 토큰을 가르기 때문에, Olmo 3 같은 공개 모델은 기준마다 작은 모델을 학습해 보고 정해요."
   ]
  },
  "s3": {
   "title": "③ 똑같은 문단 빼기",
   "lines": [
    [
     "\"Initialize self. See help(type(self)) for accurate signature.\" → 213번",
     1,
     "newf"
    ],
    [
     "\"Create and return a new object. See help(type) for accurate signature.\" → 120번",
     2,
     null
    ],
    [
     "\"Common base class for all non-exit exceptions.\" → 41번",
     3,
     null
    ],
    [
     "영어 7,418개 → 5,067개, 한국어 2,201개 → 2,093개, 코드 2,498개 → 2,438개",
     4
    ]
   ],
   "notes": [
    "영어는 32%가 똑같은 문단이었어요. 클래스마다 물려받은 기본 설명문이 되풀이되기 때문이에요. 웹 문서도 머리말, 꼬리말, 복사한 글 때문에 이렇게 겹쳐요.",
    "같은 글을 여러 번 보면 모델이 그 글을 외우고(개인정보나 저작권 문제), 같은 계산량으로 볼 수 있는 새 글이 줄어요."
   ]
  },
  "s4": {
   "title": "④ MinHash가 찾은 거의 같은 짝 (영어)",
   "cols": [
    "닮은 정도",
    "앞 문단",
    "뒤 문단 (뺌)"
   ],
   "rows": [
    [
     "0.98",
     "Maximum of self and other. If one operand is a quiet NaN and the other is numeric, the numeric operand is returned.",
     "Minimum of self and other. If one operand is a quiet NaN and the other is numeric, the numeric operand is returned."
    ],
    [
     "0.98",
     "Every sequence of mouse-move-events on a turtle is preceded by a mouse-click event on that",
     "Every sequence of mouse-move-events on a turtle is preceded by a mouse-click event on that turtle."
    ],
    [
     "0.81",
     "dir_fd, effective_ids, and follow_symlinks may not be implemented on your platform. If they are unavailable, using them …",
     "dir_fd and follow_symlinks may not be implemented on your platform. If they are unavailable, using them will raise a Not…"
    ]
   ],
   "notes": [
    "영어 5,067개 문단의 모든 짝(12,834,711개)을 비교하는 대신, 지문의 한 묶음이라도 같은 476짝만 후보로 봤어요. 그중 닮은 정도 0.8 이상 25짝을 찾아 25개를 뺐어요.",
    "첫 줄을 보세요. Maximum, Minimum처럼 첫 단어만 달라 하는 일이 다른 설명인데, 나머지 글이 거의 같아서 지워졌어요. 거의 같은 것 빼기는 글의 모양만 봐요. 기준을 낮출수록 이런 손해가 늘어요."
   ]
  },
  "t4": {
   "title": "닮은 정도 기준을 바꾸면",
   "cols": [
    "기준",
    "찾은 짝",
    "빼는 문단"
   ],
   "rows": [
    [
     0.5,
     312,
     214
    ],
    [
     0.7,
     92,
     82
    ],
    [
     "<b>0.8</b> (지금)",
     25,
     25
    ],
    [
     0.9,
     9,
     9
    ]
   ],
   "notes": [
    "기준을 낮추면 더 많이 빼요. 많이 뺄수록 말뭉치는 깨끗해지지만 다른 뜻의 글도 같이 사라져요."
   ]
  },
  "s5": {
   "title": "⑤ 가진 만큼 섞은 토크나이저 (어휘 4,000개)",
   "lines": [
    [
     "섞은 비율: 한국어 19%, 영어 42%, 코드 39%",
     1
    ],
    [
     "토큰 하나에 담긴 글자 수: 한국어 1.75, 영어 3.34, 코드 3.37",
     2,
     "newf"
    ],
    [
     "\"토크나이저는 자주 나오는 글자 조각을 하나의 토큰으로 묶어요.\" → 25토큰",
     3
    ],
    [
     "한국어를 2%만 섞은 토크나이저 → 42토큰",
     4
    ]
   ],
   "notes": [
    "같은 어휘 크기라도 한국어는 토큰 하나에 1.75글자, 영어는 3.34글자를 담아요. 한국어 글은 토큰이 더 많이 들어서 API 비용이 더 들고, 같은 문맥 길이에 더 적게 들어가요.",
    "한국어를 조금만 섞으면 한글 한 글자도 바이트 조각으로 쪼개져요(아래 표의 �는 글자의 일부 바이트예요)."
   ]
  },
  "t5": {
   "title": "토큰으로 쪼갠 모양",
   "cols": [
    "토크나이저",
    "토큰"
   ],
   "rows": [
    [
     "가진 만큼 섞기",
     "토 | � | � | 나 | 이 | 저 | 는 |  자 | 주 |  나 | 오 | 는 |  글자 |  조각 | 을 |  하나 | 의 |  � | �� | � | � | 으로 |  묶 | 어요 | ."
    ],
    [
     "한국어 2%",
     "� | � | � | � | � | � | 나 | 이 | � | � | 는 |  � | � | � | � | � |  � | � | � | � | � | 는 |  글자 |  조 | � | � | 을 |  � | �� | �� | 의 |  � | � | � | � | � | � | 으로 |  � | � | 어요 | ."
    ]
   ],
   "notes": []
  },
  "t5b": {
   "title": "어휘 크기를 바꾸면",
   "cols": [
    "어휘 수",
    "한국어 글자/토큰",
    "영어 글자/토큰",
    "코드 글자/토큰"
   ],
   "rows": [
    [
     "1,000",
     "1.08",
     "2.30",
     "2.54"
    ],
    [
     "<b>4,000</b> (지금)",
     "1.75",
     "3.34",
     "3.37"
    ],
    [
     "16,000",
     "2.38",
     "4.05",
     "3.82"
    ]
   ],
   "notes": [
    "어휘가 클수록 긴 조각을 하나로 담아 압축률이 올라가요. 대신 임베딩 표가 커지고, 드문 토큰은 학습이 덜 돼요. 실제 LLM은 3만에서 25만 개를 써요."
   ]
  },
  "f_mix": {
   "title": "섞는 비율에 따른 압축률",
   "images": [
    [
     "img/pretrain/mixture_compression.png",
     ""
    ]
   ],
   "notes": [
    "점 하나가 비율 조합 하나예요(24가지). 그 분야를 많이 섞을수록 그 분야의 압축률이 올라가요. 비율이 아주 작은 쪽에서 가장 가파르게 변해요.",
    "속이 빈 점(작은 대리 토크나이저)은 값은 낮지만 모양이 같아요. 그래서 조합끼리의 순위가 거의 그대로예요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ 대리 토크나이저의 순위가 실제 크기에서도 그대로인가",
   "cols": [
    "분야",
    "순위 상관 (Spearman)"
   ],
   "rows": [
    [
     "한국어",
     "1.00"
    ],
    [
     "영어",
     "0.98"
    ],
    [
     "코드",
     "0.94"
    ]
   ],
   "notes": [
    "1이면 순위가 완전히 같아요. 작고 싼 토크나이저로 비율 조합을 많이 시험해 보고, 결과의 순위를 큰 토크나이저에도 믿고 쓸 수 있다는 뜻이에요(TReX 논문의 순위 불변성과 같은 결론)."
   ]
  },
  "s7": {
   "title": "⑦ 서비스 글(한국어 60%, 영어 20%, 코드 20%) 1,000자에 드는 토큰",
   "cols": [
    "섞은 비율",
    "한국어 : 영어 : 코드",
    "한국어 글자/토큰",
    "영어 글자/토큰",
    "코드 글자/토큰",
    "1,000자에 토큰"
   ],
   "rows": [
    [
     "똑같이 3분의 1씩",
     "0.33 : 0.33 : 0.33",
     "1.86",
     "3.17",
     "3.27",
     "447.0"
    ],
    [
     "가진 만큼",
     "0.19 : 0.42 : 0.39",
     "1.75",
     "3.34",
     "3.37",
     "461.9"
    ],
    [
     "서비스 비율 그대로",
     "0.60 : 0.20 : 0.20",
     "1.97",
     "2.92",
     "3.04",
     "438.6"
    ],
    [
     "회귀가 고른 비율",
     "0.76 : 0.10 : 0.14",
     "2.00",
     "2.73",
     "2.94",
     "440.5"
    ]
   ],
   "notes": [
    "가진 만큼 섞으면 461.9토큰, 서비스 비율대로 섞으면 438.6토큰이에요. 5%가 줄었어요. 같은 비용으로 그만큼 더 많은 글을 넣고 학습할 수 있어요.",
    "회귀가 고른 비율(0.76 : 0.10 : 0.14)은 440.5토큰으로, 서비스 비율 그대로(438.6)보다 조금 나빴어요. 회귀가 대리 토크나이저의 모양을 단순한 식(로그 비율의 직선)으로 근사하기 때문이에요. 여기처럼 분야가 셋뿐이면 \"서비스 비율대로\"가 이미 좋은 출발점이고, 회귀는 분야가 수십 개라 손으로 고를 수 없을 때 힘을 써요."
   ]
  },
  "t7": {
   "title": "서비스에 들어올 글의 비율이 다르면",
   "cols": [
    "서비스 비율 (한국어 : 영어 : 코드)",
    "회귀가 고른 학습 비율"
   ],
   "rows": [
    [
     "<b>0.6 : 0.2 : 0.2 (지금)</b>",
     "0.76 : 0.10 : 0.14"
    ],
    [
     "0.2 : 0.4 : 0.4",
     "0.18 : 0.42 : 0.40"
    ],
    [
     "0.34 : 0.33 : 0.33",
     "0.44 : 0.28 : 0.28"
    ]
   ],
   "notes": [
    "서비스에 한국어가 많으면 회귀는 그보다 더 한국어 쪽으로 기울여요(0.76 : 0.10 : 0.14). 한국어 압축률이 가장 낮아서, 한국어를 조금 더 잘 압축하는 게 전체 토큰을 가장 많이 줄이기 때문이에요. 한국어가 적은 서비스에서는 거의 서비스 비율 그대로를 골라요(0.18 : 0.42 : 0.40)."
   ]
  },
  "f_funnel": {
   "title": "거르고 빼면 얼마나 남나",
   "images": [
    [
     "img/pretrain/funnel.png",
     ""
    ]
   ],
   "notes": [
    "영어는 1,058,766자에서 629,654자로 41%가 줄었어요. 대부분 똑같은 설명문이었어요. 웹 문서를 모으는 실제 LLM도 이 단계에서 많은 양을 버려요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   1,
   "모으기",
   "한국어, 영어, 코드",
   "분야마다 크기"
  ],
  [
   2,
   "규칙으로 거르기",
   "짧음, 기호, 반복",
   "버린 것 읽기"
  ],
  [
   3,
   "똑같은 것 빼기",
   "dict.fromkeys",
   "외우기 방지"
  ],
  [
   4,
   "거의 같은 것",
   "MinHash, LSH",
   "닮은 정도 기준"
  ],
  [
   5,
   "토크나이저",
   "BPE, 압축률",
   "한국어는 토큰이 많이"
  ],
  [
   6,
   "섞는 비율",
   "대리 토크나이저",
   "순위 불변"
  ],
  [
   7,
   "비율 고르기",
   "회귀",
   "서비스 글 기준"
  ]
 ],
 "env": "Python 3.13.5 (표준 라이브러리가 영어와 코드 말뭉치), tokenizers 0.22.2, NumPy 2.5.2 (seed 0)"
};
