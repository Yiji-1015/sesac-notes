# LLM을 처음부터 학습하기 전에 하는 데이터 일(품질 거르기, 중복 빼기)과, 데이터를 섞는 비율이 토크나이저의 압축률을 어떻게 바꾸는지 재 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: 받을 파일 없이 이 PC에 있는 글로 만들어요. 한국어는 ../gpt/data/notes_corpus.jsonl(옛 Day 노트),
#         영어는 파이썬 표준 라이브러리의 설명문(docstring), 코드는 표준 라이브러리 소스에서 주석과 설명문을 뺀 것이에요.
# CPU로 1분쯤. 파이썬 버전이 다르면 영어와 코드 말뭉치가 조금 달라져서 숫자도 조금 달라져요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                                     #@1
import re                                                                       #@1
import inspect                                                                  #@1
import importlib                                                                #@1
import zlib                                                                     #@4
from collections import Counter, defaultdict                                    #@3
from pathlib import Path                                                        #@1
import numpy as np                                                              #@1
from tokenizers import Tokenizer, models, trainers, pre_tokenizers              #@5
from scipy.stats import spearmanr                                               #@6

# ① 세 가지 말뭉치 모으기
HERE = Path(__file__).resolve().parent
ko = [p.strip() for line in (HERE.parent / "gpt" / "data" / "notes_corpus.jsonl").open(encoding="utf-8")
      for p in json.loads(line)["text"].split("\n") if p.strip()]

EN_MODULES = ["os", "re", "json", "collections", "itertools", "functools", "pathlib", "typing", "string", "random", "statistics", "math",
              "datetime", "logging", "argparse", "subprocess", "shutil", "csv", "sqlite3", "http.client", "urllib.request", "email.message",
              "asyncio", "threading", "unittest", "decimal", "fractions", "heapq", "bisect", "textwrap", "difflib", "inspect", "ast",
              "zipfile", "tarfile", "hashlib", "uuid", "socket", "ssl", "xml.etree.ElementTree", "pickle", "copy", "enum", "dataclasses",
              "contextlib", "io", "tempfile", "calendar", "time", "codecs", "struct", "queue", "selectors", "signal", "base64", "mailbox",
              "configparser", "doctest", "timeit", "weakref", "types", "operator", "turtle", "tkinter", "zlib", "lzma"]
def docs_of(mod):                                                # 모듈, 함수, 클래스, 메서드의 설명문을 문단으로
    m = importlib.import_module(mod)
    objs = [m] + [o for _, o in inspect.getmembers(m) if inspect.isfunction(o) or inspect.isclass(o) or inspect.isbuiltin(o)]
    objs += [o for c in objs if inspect.isclass(c) and c.__module__ == m.__name__ for o in vars(c).values() if callable(o)]
    return [" ".join(p.split()) for o in objs for p in (inspect.getdoc(o) or "").split("\n\n")]
en = [p for mod in EN_MODULES for p in docs_of(mod)]

CODE_MODULES = ["json.decoder", "json.encoder", "textwrap", "difflib", "heapq", "bisect", "fractions", "statistics", "csv", "configparser",
                "argparse", "dataclasses", "pathlib", "string", "calendar", "shutil", "glob", "fnmatch", "queue", "sched", "tempfile",
                "pprint", "copy", "enum", "functools", "base64", "uuid", "random", "ipaddress", "gettext"]
def code_of(mod):                                                # 소스에서 설명문과 주석을 빼고 빈 줄로 나눈 덩어리
    src = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', "", inspect.getsource(importlib.import_module(mod)), flags=re.S)
    lines = [re.sub(r"\s+#.*$", "", ln) for ln in src.splitlines() if not ln.lstrip().startswith("#")]
    return [b.strip("\n") for b in re.split(r"\n\s*\n", "\n".join(lines))]
code = [b for mod in CODE_MODULES for b in code_of(mod)]

corpus = {"한국어": ko, "영어": en, "코드": code}
size = lambda docs: (len(docs), sum(map(len, docs)))             # noqa: E731  (문서 수, 글자 수)
raw_size = {k: size(v) for k, v in corpus.items()}

# ② 품질 거르기: 규칙 몇 개
def bad_reason(text, kind):
    if len(text) < 40:
        return "40자 미만"
    if kind != "코드":
        letters = sum(ch.isalpha() for ch in text) / len(text)
        if letters < 0.6:
            return "글자 비율 60% 미만"                            # 기호, 숫자, 표 조각 위주
        words = text.split()
        if len(words) >= 8 and len(set(words)) / len(words) < 0.5:
            return "같은 단어 반복"
    return None

dropped = Counter()
drop_examples = {}
for k in corpus:
    keep = []
    for t in corpus[k]:
        r = bad_reason(t, k)
        if r is None:
            keep.append(t)
        else:
            dropped[(k, r)] += 1
            drop_examples.setdefault((k, r), t)
    corpus[k] = keep
filtered_size = {k: size(v) for k, v in corpus.items()}

# ③ 똑같은 것 빼기
top_dups = Counter(corpus["영어"]).most_common(3)
corpus = {k: list(dict.fromkeys(v)) for k, v in corpus.items()}  # 처음 나온 것만 남기기
exact_size = {k: size(v) for k, v in corpus.items()}

# ④ 거의 같은 것 빼기: MinHash와 LSH
PRIME, H, BANDS = 4294967311, 64, 16                             # 해시 64개를 4개씩 16묶음으로
rng = np.random.default_rng(0)
ha, hb = rng.integers(1, 2**32, H, dtype=np.uint64), rng.integers(0, 2**32, H, dtype=np.uint64)
def minhash(text, n=5):                                          # 단어 5개짜리 조각 집합의 "지문" 64칸
    w = text.split()
    sh = np.array([zlib.crc32(" ".join(w[i:i + n]).encode()) for i in range(max(1, len(w) - n + 1))], dtype=np.uint64)
    return ((np.outer(ha, sh) + hb[:, None]) % np.uint64(PRIME)).min(axis=1)

def near_duplicates(docs, threshold=0.8):
    sig = np.array([minhash(t) for t in docs])
    cand = set()
    for b in range(BANDS):                                       # 한 묶음이라도 똑같으면 후보 (전부 비교하지 않기)
        bucket = defaultdict(list)
        for i, row in enumerate(sig[:, b * 4:(b + 1) * 4]):
            bucket[row.tobytes()].append(i)
        cand |= {(x, y) for v in bucket.values() for i, x in enumerate(v) for y in v[i + 1:]}
    pairs = sorted((float((sig[i] == sig[j]).mean()), i, j) for i, j in cand)   # 지문이 같은 칸의 비율 ≈ 조각 집합의 Jaccard
    return [(s, i, j) for s, i, j in pairs if s >= threshold], len(cand)

en_exact = corpus["영어"]
near, n_cand = near_duplicates(en_exact)
drop = {j for _, _, j in near}                                   # 짝마다 뒤의 것을 빼기
corpus["영어"] = [t for i, t in enumerate(en_exact) if i not in drop]
clean_size = {k: size(v) for k, v in corpus.items()}

# ⑤ 토크나이저와 압축률
split = {}
for k, docs in corpus.items():                                   # 문서의 80%로 토크나이저를 배우고 20%로 재기
    idx = np.random.default_rng(1).permutation(len(docs))
    cut = int(len(docs) * 0.8)
    split[k] = ([docs[i] for i in idx[:cut]], "\n".join(docs[i] for i in idx[cut:]))
DOMAINS = list(corpus)

def train_tokenizer(mix, budget=400_000, vocab=4000, seed=0):    # mix: 분야마다 비율, budget: 학습에 쓰는 글자 수
    docs = []
    for k, p in zip(DOMAINS, mix):
        r = np.random.default_rng(seed)
        total = 0
        for i in r.permutation(len(split[k][0])):
            if total >= budget * p:
                break
            docs.append(split[k][0][i])
            total += len(split[k][0][i])
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)   # 바이트 단위라 모르는 글자가 없어요
    tok.train_from_iterator(docs, trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    return tok

def chars_per_token(tok):                                        # 압축률: 토큰 하나가 평균 몇 글자를 담나 (클수록 좋아요)
    return np.array([len(split[k][1]) / len(tok.encode(split[k][1]).ids) for k in DOMAINS])

natural = np.array([clean_size[k][1] for k in DOMAINS], float)
natural /= natural.sum()                                         # 가진 만큼 그대로 섞기
tok_natural = train_tokenizer(natural)
cpt_natural = chars_per_token(tok_natural)
SAMPLE = "토크나이저는 자주 나오는 글자 조각을 하나의 토큰으로 묶어요."
tok_little_ko = train_tokenizer(np.array([0.02, 0.49, 0.49]))
pieces = {"가진 만큼 섞기": tok_natural.encode(SAMPLE).tokens, "한국어 2%": tok_little_ko.encode(SAMPLE).tokens}

# ⑥ 섞는 비율을 바꾸면: 작은 대리 토크나이저로 먼저
MIXES = np.random.default_rng(2).dirichlet(np.ones(3), 24)       # 비율 조합 24가지
proxy = np.array([chars_per_token(train_tokenizer(m, budget=100_000, vocab=1000)) for m in MIXES])   # 작고 싸게
full = np.array([chars_per_token(train_tokenizer(m)) for m in MIXES])                                  # 실제 크기
rank_kept = {k: float(spearmanr(proxy[:, j], full[:, j])[0]) for j, k in enumerate(DOMAINS)}          # 순위가 그대로인가

# ⑦ 회귀로 비율 고르기 (TReX 방식)
TRAFFIC = np.array([0.6, 0.2, 0.2])                              # 서비스에 들어올 글의 비율 (글자 기준): 한국어, 영어, 코드
def features(m):
    return np.column_stack([np.ones(len(m)), np.log(np.clip(m, 1e-3, 1))])
coef = np.linalg.lstsq(features(MIXES), proxy, rcond=None)[0]    # 비율 → 분야마다 압축률 (대리 토크나이저 기준)
def tokens_per_1k(cpt):                                          # 서비스 글 1,000자에 드는 토큰 수 (작을수록 좋아요)
    return float((1000 * TRAFFIC / cpt).sum())
grid = np.array([(a, b, 1 - a - b) for a in np.arange(0.02, 0.97, 0.02) for b in np.arange(0.02, 0.97, 0.02) if 1 - a - b >= 0.019])
best_mix = grid[np.argmin([tokens_per_1k(c) for c in features(grid) @ coef])]
candidates = {"똑같이 3분의 1씩": np.ones(3) / 3, "가진 만큼": natural, "서비스 비율 그대로": TRAFFIC, "회귀가 고른 비율": best_mix}
verdict = {name: (m, chars_per_token(train_tokenizer(m))) for name, m in candidates.items()}   # 실제 크기로 확인
