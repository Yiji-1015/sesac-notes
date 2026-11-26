# RAG 검색 고치기: 실패 나누기, 질문 다시 쓰기, 후보 수와 판 조건, 여러 질문으로 찾기, 재정렬. 이 파일 하나로 실행돼요.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 검색 대상: 가람 도서관 안내 조각 26개 (../ingest/out/chunks.json 12개 + ../rag/data/more_chunks.json 14개, 둘 다 이 노트용 가상 자료)
# 검색은 intfloat/multilingual-e5-small, 재정렬은 BAAI/bge-reranker-v2-m3를 CUDA GPU에서 돌려요. 둘 다 이 PC에 받아 둔 모델이에요.
# 질문 다시 쓰기는 원래 LLM이 하는 일인데, 이 예제는 미리 정한 답을 돌려주는 함수(fake_llm)로 대신해요. 키도 비용도 없어요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
import re                                                        #@3
import time                                                      #@7
from pathlib import Path                                         #@1
import numpy as np                                               #@1
import torch                                                     #@6
from sentence_transformers import SentenceTransformer            #@1
from sentence_transformers import CrossEncoder                   #@6

HERE = Path(__file__).resolve().parent                           #@1

# ① 도서관 조각을 벡터로 바꾸고 찾기
CATEGORY = {"guide.md": "이용", "hours.html": "이용", "room.md": "이용", "holds.md": "이용", "programs.md": "이용", "devices.md": "이용",
            "notice.txt": "공지", "old_notice.txt": "공지", "notice_2025.txt": "공지", "notice_2026.txt": "공지",
            "rules.pdf": "규정", "scan.pdf": "시설"}            # 파일마다 붙인 분류
raw = (json.loads((HERE.parent / "ingest" / "out" / "chunks.json").read_text(encoding="utf-8"))
       + json.loads((HERE.parent / "rag" / "data" / "more_chunks.json").read_text(encoding="utf-8")))
CHUNKS = [{"id": n, "file": c["file"], "title": c["heading"] or c["body"].split("\n")[0], "body": c["body"],
           "category": CATEGORY[c["file"]], "current": c.get("current", True)} for n, c in enumerate(raw, start=1)]
encoder = SentenceTransformer("intfloat/multilingual-e5-small")
vectors = encoder.encode([f"passage: {c['title']}\n{c['body']}" for c in CHUNKS], normalize_embeddings=True)

def search(query, category, k):                                  # 분류 안에서 질문과 가까운 조각 k개
    q = encoder.encode([f"query: {query}"], normalize_embeddings=True)[0]
    scores = vectors @ q
    order = [i for i in np.argsort(-scores) if CHUNKS[i]["category"] == category]
    return [{**CHUNKS[i], "score": round(float(scores[i]), 3)} for i in order[:k]]

first = search("대출 기간은 며칠인가요?", "이용", k=3)

# ② 정답을 정해 두고 실패 나누기
def find_id(title):
    return next(c["id"] for c in CHUNKS if c["title"] == title)

TESTS = [  # (앞선 질문, 질문, 분류, 정답 조각). 결과를 보기 전에 정해 둬요
    ("스터디룸은 어떻게 예약해?", "그럼 취소는?", "이용", find_id("예약 철회")),
    ("노트북도 빌릴 수 있어?", "몇 시간까지 쓸 수 있어?", "이용", find_id("노트북 대여")),
    ("", "상호대차 신청하면 며칠 걸려요?", "이용", find_id("다른 도서관 책 받아 보기")),
    ("", "예약해 놓고 안 가면 어떻게 돼요?", "이용", find_id("노쇼")),
    ("", "스터디룸 예약을 철회하려면 어떻게 하나요?", "이용", find_id("예약 철회")),
    ("", "책을 잃어버리면 어떻게 하나요?", "이용", find_id("분실과 훼손")),
]
CANDIDATES, CONTEXT = 10, 3                                      # 예시 설정: 후보 10개를 찾아 위 3개를 모델에게 줘요

def rank_of(gold, hits):                                         # 정답이 몇 위인지 (없으면 None)
    ids = [h["id"] for h in hits]
    return ids.index(gold) + 1 if gold in ids else None

def diagnose(rank):                                              # 실패가 어디서 났나
    if rank is None or rank > CANDIDATES:
        return "후보 밖"                                          # 검색이 못 찾음
    return "통과" if rank <= CONTEXT else "순위 밀림"              # 찾기는 했는데 위 3개에 못 듦

before = [search(q, cat, k=len(CHUNKS)) for _, q, cat, _ in TESTS]   # 분류 안 전체 순서
diagnosis = [diagnose(rank_of(gold, hits)) for hits, (_, _, _, gold) in zip(before, TESTS)]

# ③ 질문 다시 쓰기: 대화에서 빠진 대상 채우기
REWRITE_PROMPT = """앞선 질문을 보고, 대화를 모르는 사람도 알아듣는 검색용 질문 하나로 다시 쓰세요.
대화에 없는 대상, 날짜, 숫자, 조건은 더하지 말고, 기간과 부정 표현은 그대로 두세요.
무엇을 가리키는지 정할 수 없으면 needs_clarification을 true로 하세요.
JSON으로 답하세요: {{"search_query": "...", "needs_clarification": false}}

앞선 질문: {history}
현재 질문: {question}"""

CANNED = {  # 가짜 LLM이 돌려줄 미리 정한 답 (실제 모델이 아니에요). 표에 없으면 질문을 그대로 돌려줘요
    ("스터디룸은 어떻게 예약해?", "그럼 취소는?"): {"search_query": "스터디룸 예약을 취소하는 방법은?", "needs_clarification": False},
    ("노트북도 빌릴 수 있어?", "몇 시간까지 쓸 수 있어?"): {"search_query": "노트북은 몇 시간까지 빌려 쓸 수 있나요?", "needs_clarification": False},
    ("스터디룸은 하루에 몇 시간까지 쓸 수 있어?", "올해도 똑같아?"): {"search_query": "올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?", "needs_clarification": False},
    ("(없음)", "그거 바꿀 수 있어?"): {"search_query": "그거 바꿀 수 있어?", "needs_clarification": True},
}

def fake_llm(prompt):                                            # LLM API를 부를 자리. 받은 Prompt의 마지막 두 줄로 미리 정한 답을 찾아요
    history, question = re.search(r"앞선 질문: (.*)\n현재 질문: (.*)$", prompt).groups()
    return json.dumps(CANNED.get((history, question), {"search_query": question, "needs_clarification": False}), ensure_ascii=False)

TIME_WORDS = ("올해", "작년", "내년", "오늘", "내일")
def check_rewrite(history, question, rewritten):                 # 대화에 없던 숫자가 생겼나, 때를 가리키는 말이 빠졌나
    problems = [f"없던 숫자 {n}" for n in re.findall(r"\d+", rewritten) if n not in history + question]
    problems += [f"빠진 말 {w}" for w in TIME_WORDS if w in question and w not in rewritten]
    return problems

def rewrite(history, question):                                  # 검색용 질문을 따로 만들어요. None이면 되묻기 (검색하지 않음)
    prompt = REWRITE_PROMPT.format(history=history or "(없음)", question=question)
    answer = json.loads(fake_llm(prompt))
    if answer["needs_clarification"] or check_rewrite(history, question, answer["search_query"]):
        return None
    return answer["search_query"]

prompt_example = REWRITE_PROMPT.format(history=TESTS[0][0], question=TESTS[0][1])
rewritten = [rewrite(h, q) for h, q, _, _ in TESTS[:2]]
after = [search(s, cat, k=len(CHUNKS)) for s, (_, _, cat, _) in zip(rewritten, TESTS)]
year_query = rewrite("스터디룸은 하루에 몇 시간까지 쓸 수 있어?", "올해도 똑같아?")
unclear = rewrite("", "그거 바꿀 수 있어?")
checks = {t: check_rewrite("스터디룸은 하루에 몇 시간까지 쓸 수 있어?", "올해도 똑같아?", t)
          for t in ("올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?", "2025년 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?",
                    "스터디룸은 하루에 몇 시간까지 쓸 수 있나요?")}

# ④ 후보 모으기: 몇 개를, 어떤 조건으로
def search_now(query, category, k):                              # 지금 유효한 판만 남기고 k개
    return [h for h in search(query, category, len(CHUNKS)) if h["current"]][:k]

q3, gold3 = TESTS[2][1], TESTS[2][3]
k_sweep = {k: rank_of(gold3, search_now(q3, "이용", k)) is not None for k in (3, 5, 10, 20)}
both_years = search(year_query, "공지", k=2)                     # 판을 가리지 않으면
now_only = search_now(year_query, "공지", k=2)                   # 지금 유효한 판만

# ⑤ 여러 질문으로 찾기: 용어 넓히기와 합치기
TERMS = {"상호대차": "다른 도서관 책 받아 보기", "취소": "철회"}      # 도서관 용어집: 사람들이 쓰는 말 → 안내문이 쓰는 말

def expand(query):                                               # 원래 질문 + 용어집의 말로 바꾼 질문
    return [query] + [query.replace(a, b) for a, b in TERMS.items() if a in query]

def multi_search(queries, category, k):                          # 질문마다 같은 조건으로 찾고, 같은 조각은 한 번만
    merged, found_by = {}, {}
    for q in queries:
        for h in search_now(q, category, k):
            merged.setdefault(h["id"], h)
            found_by.setdefault(h["id"], []).append(q)
    return list(merged.values()), found_by

queries3 = expand(q3)
per_query = {q: search_now(q, "이용", 5) for q in queries3}
union3, found3 = multi_search(queries3, "이용", 5)

# ⑥ 다시 순위 매기기: Cross-Encoder
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU가 필요해요")                     # 조용히 CPU로 바꾸지 않아요
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0")   # 질문과 조각을 함께 읽는 모델

def rerank(query, hits):                                         # (질문, 본문) 짝마다 점수를 매겨 높은 순서로. 후보 밖 조각은 못 데려와요
    ranked = reranker.rank(query, [h["body"] for h in hits], batch_size=16)   # [{"corpus_id": 후보 번호, "score": 점수}, ...] 높은 순서
    return [{**hits[r["corpus_id"]], "rerank": float(r["score"])} for r in ranked]

q4, gold4 = TESTS[3][1], TESTS[3][3]
cands4 = search_now(q4, "이용", CANDIDATES)
ranked4 = rerank(q4, cands4)
logits4 = reranker.predict([(q4, h["body"]) for h in cands4], activation_fn=torch.nn.Identity())   # 0~1로 줄이기 전의 값
ranked3 = rerank(q3, union3)                                     # ⑤에서 합친 후보
year_ranked = rerank(year_query, both_years)                     # 2025년, 2026년 공지를 둘 다 넣으면

# ⑦ 하나씩 켜서 비교하기와 걸린 시간
def find(history, question, category, use_rewrite, use_expand, use_rerank):   # 질문 하나 → 순서가 정해진 후보 목록
    query = rewrite(history, question) if use_rewrite else question
    if query is None:
        return []                                                # 되묻기: 찾지 않아요
    hits, _ = multi_search(expand(query) if use_expand else [query], category, CANDIDATES)
    return rerank(query, hits) if use_rerank else hits

SETTINGS = {"기준": (False, False, False), "다시 쓰기만": (True, False, False), "재정렬만": (False, False, True),
            "다시 쓰기 + 재정렬": (True, False, True), "셋 다 (+ 넓히기)": (True, True, True)}   # (다시 쓰기, 넓히기, 재정렬)
compare = {name: [rank_of(gold, find(h, q, cat, *flags)) for h, q, cat, gold in TESTS] for name, flags in SETTINGS.items()}

def timed(fn, *args):                                            # 걸린 시간 (ms). GPU 계산은 끝날 때까지 기다렸다 재요
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(*args)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000

times = {"다시 쓰기": [], "검색": [], "재정렬": []}
for _ in range(20):
    query, t1 = timed(rewrite, TESTS[0][0], TESTS[0][1])
    hits, t2 = timed(lambda: multi_search(expand(query), "이용", CANDIDATES)[0])
    _, t3 = timed(rerank, query, hits)
    for name, ms in zip(times, (t1, t2, t3)):
        times[name].append(ms)
cost = {k: [timed(rerank, q4, search_now(q4, "이용", k))[1] for _ in range(20)] for k in (5, 10, 17)}
