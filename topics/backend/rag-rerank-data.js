window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import re",
   "s": 3
  },
  {
   "t": "import time",
   "s": 7
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
   "t": "import torch",
   "s": 6
  },
  {
   "t": "from sentence_transformers import SentenceTransformer",
   "s": 1
  },
  {
   "t": "from sentence_transformers import CrossEncoder",
   "s": 6
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 도서관 조각을 벡터로 바꾸고 찾기",
   "s": 1
  },
  {
   "t": "CATEGORY = {\"guide.md\": \"이용\", \"hours.html\": \"이용\", \"room.md\": \"이용\", \"holds.md\": \"이용\", \"programs.md\": \"이용\", \"devices.md\": \"이용\",",
   "s": 1
  },
  {
   "t": "            \"notice.txt\": \"공지\", \"old_notice.txt\": \"공지\", \"notice_2025.txt\": \"공지\", \"notice_2026.txt\": \"공지\",",
   "s": 1
  },
  {
   "t": "            \"rules.pdf\": \"규정\", \"scan.pdf\": \"시설\"}  # 파일마다 붙인 분류",
   "s": 1
  },
  {
   "t": "raw = (json.loads((HERE.parent / \"ingest\" / \"out\" / \"chunks.json\").read_text(encoding=\"utf-8\"))",
   "s": 1
  },
  {
   "t": "       + json.loads((HERE.parent / \"rag\" / \"data\" / \"more_chunks.json\").read_text(encoding=\"utf-8\")))",
   "s": 1
  },
  {
   "t": "CHUNKS = [{\"id\": n, \"file\": c[\"file\"], \"title\": c[\"heading\"] or c[\"body\"].split(\"\\n\")[0], \"body\": c[\"body\"],",
   "s": 1
  },
  {
   "t": "           \"category\": CATEGORY[c[\"file\"]], \"current\": c.get(\"current\", True)} for n, c in enumerate(raw, start=1)]",
   "s": 1
  },
  {
   "t": "encoder = SentenceTransformer(\"intfloat/multilingual-e5-small\")",
   "s": 1
  },
  {
   "t": "vectors = encoder.encode([f\"passage: {c['title']}\\n{c['body']}\" for c in CHUNKS], normalize_embeddings=True)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def search(query, category, k):  # 분류 안에서 질문과 가까운 조각 k개",
   "s": 1
  },
  {
   "t": "    q = encoder.encode([f\"query: {query}\"], normalize_embeddings=True)[0]",
   "s": 1
  },
  {
   "t": "    scores = vectors @ q",
   "s": 1
  },
  {
   "t": "    order = [i for i in np.argsort(-scores) if CHUNKS[i][\"category\"] == category]",
   "s": 1
  },
  {
   "t": "    return [{**CHUNKS[i], \"score\": round(float(scores[i]), 3)} for i in order[:k]]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "first = search(\"대출 기간은 며칠인가요?\", \"이용\", k=3)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 정답을 정해 두고 실패 나누기",
   "s": 2
  },
  {
   "t": "def find_id(title):",
   "s": 2
  },
  {
   "t": "    return next(c[\"id\"] for c in CHUNKS if c[\"title\"] == title)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "TESTS = [  # (앞선 질문, 질문, 분류, 정답 조각). 결과를 보기 전에 정해 둬요",
   "s": 2
  },
  {
   "t": "    (\"스터디룸은 어떻게 예약해?\", \"그럼 취소는?\", \"이용\", find_id(\"예약 철회\")),",
   "s": 2
  },
  {
   "t": "    (\"노트북도 빌릴 수 있어?\", \"몇 시간까지 쓸 수 있어?\", \"이용\", find_id(\"노트북 대여\")),",
   "s": 2
  },
  {
   "t": "    (\"\", \"상호대차 신청하면 며칠 걸려요?\", \"이용\", find_id(\"다른 도서관 책 받아 보기\")),",
   "s": 2
  },
  {
   "t": "    (\"\", \"예약해 놓고 안 가면 어떻게 돼요?\", \"이용\", find_id(\"노쇼\")),",
   "s": 2
  },
  {
   "t": "    (\"\", \"스터디룸 예약을 철회하려면 어떻게 하나요?\", \"이용\", find_id(\"예약 철회\")),",
   "s": 2
  },
  {
   "t": "    (\"\", \"책을 잃어버리면 어떻게 하나요?\", \"이용\", find_id(\"분실과 훼손\")),",
   "s": 2
  },
  {
   "t": "]",
   "s": 2
  },
  {
   "t": "CANDIDATES, CONTEXT = 10, 3  # 예시 설정: 후보 10개를 찾아 위 3개를 모델에게 줘요",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def rank_of(gold, hits):  # 정답이 몇 위인지 (없으면 None)",
   "s": 2
  },
  {
   "t": "    ids = [h[\"id\"] for h in hits]",
   "s": 2
  },
  {
   "t": "    return ids.index(gold) + 1 if gold in ids else None",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def diagnose(rank):  # 실패가 어디서 났나",
   "s": 2
  },
  {
   "t": "    if rank is None or rank > CANDIDATES:",
   "s": 2
  },
  {
   "t": "        return \"후보 밖\"  # 검색이 못 찾음",
   "s": 2
  },
  {
   "t": "    return \"통과\" if rank <= CONTEXT else \"순위 밀림\"  # 찾기는 했는데 위 3개에 못 듦",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "before = [search(q, cat, k=len(CHUNKS)) for _, q, cat, _ in TESTS]  # 분류 안 전체 순서",
   "s": 2
  },
  {
   "t": "diagnosis = [diagnose(rank_of(gold, hits)) for hits, (_, _, _, gold) in zip(before, TESTS)]",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 질문 다시 쓰기: 대화에서 빠진 대상 채우기",
   "s": 3
  },
  {
   "t": "REWRITE_PROMPT = \"\"\"앞선 질문을 보고, 대화를 모르는 사람도 알아듣는 검색용 질문 하나로 다시 쓰세요.",
   "s": 3
  },
  {
   "t": "대화에 없는 대상, 날짜, 숫자, 조건은 더하지 말고, 기간과 부정 표현은 그대로 두세요.",
   "s": 3
  },
  {
   "t": "무엇을 가리키는지 정할 수 없으면 needs_clarification을 true로 하세요.",
   "s": 3
  },
  {
   "t": "JSON으로 답하세요: {{\"search_query\": \"...\", \"needs_clarification\": false}}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "앞선 질문: {history}",
   "s": 3
  },
  {
   "t": "현재 질문: {question}\"\"\"",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "CANNED = {  # 가짜 LLM이 돌려줄 미리 정한 답 (실제 모델이 아니에요). 표에 없으면 질문을 그대로 돌려줘요",
   "s": 3
  },
  {
   "t": "    (\"스터디룸은 어떻게 예약해?\", \"그럼 취소는?\"): {\"search_query\": \"스터디룸 예약을 취소하는 방법은?\", \"needs_clarification\": False},",
   "s": 3
  },
  {
   "t": "    (\"노트북도 빌릴 수 있어?\", \"몇 시간까지 쓸 수 있어?\"): {\"search_query\": \"노트북은 몇 시간까지 빌려 쓸 수 있나요?\", \"needs_clarification\": False},",
   "s": 3
  },
  {
   "t": "    (\"스터디룸은 하루에 몇 시간까지 쓸 수 있어?\", \"올해도 똑같아?\"): {\"search_query\": \"올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\", \"needs_clarification\": False},",
   "s": 3
  },
  {
   "t": "    (\"(없음)\", \"그거 바꿀 수 있어?\"): {\"search_query\": \"그거 바꿀 수 있어?\", \"needs_clarification\": True},",
   "s": 3
  },
  {
   "t": "}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def fake_llm(prompt):  # LLM API를 부를 자리. 받은 Prompt의 마지막 두 줄로 미리 정한 답을 찾아요",
   "s": 3
  },
  {
   "t": "    history, question = re.search(r\"앞선 질문: (.*)\\n현재 질문: (.*)$\", prompt).groups()",
   "s": 3
  },
  {
   "t": "    return json.dumps(CANNED.get((history, question), {\"search_query\": question, \"needs_clarification\": False}), ensure_ascii=False)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "TIME_WORDS = (\"올해\", \"작년\", \"내년\", \"오늘\", \"내일\")",
   "s": 3
  },
  {
   "t": "def check_rewrite(history, question, rewritten):  # 대화에 없던 숫자가 생겼나, 때를 가리키는 말이 빠졌나",
   "s": 3
  },
  {
   "t": "    problems = [f\"없던 숫자 {n}\" for n in re.findall(r\"\\d+\", rewritten) if n not in history + question]",
   "s": 3
  },
  {
   "t": "    problems += [f\"빠진 말 {w}\" for w in TIME_WORDS if w in question and w not in rewritten]",
   "s": 3
  },
  {
   "t": "    return problems",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def rewrite(history, question):  # 검색용 질문을 따로 만들어요. None이면 되묻기 (검색하지 않음)",
   "s": 3
  },
  {
   "t": "    prompt = REWRITE_PROMPT.format(history=history or \"(없음)\", question=question)",
   "s": 3
  },
  {
   "t": "    answer = json.loads(fake_llm(prompt))",
   "s": 3
  },
  {
   "t": "    if answer[\"needs_clarification\"] or check_rewrite(history, question, answer[\"search_query\"]):",
   "s": 3
  },
  {
   "t": "        return None",
   "s": 3
  },
  {
   "t": "    return answer[\"search_query\"]",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "prompt_example = REWRITE_PROMPT.format(history=TESTS[0][0], question=TESTS[0][1])",
   "s": 3
  },
  {
   "t": "rewritten = [rewrite(h, q) for h, q, _, _ in TESTS[:2]]",
   "s": 3
  },
  {
   "t": "after = [search(s, cat, k=len(CHUNKS)) for s, (_, _, cat, _) in zip(rewritten, TESTS)]",
   "s": 3
  },
  {
   "t": "year_query = rewrite(\"스터디룸은 하루에 몇 시간까지 쓸 수 있어?\", \"올해도 똑같아?\")",
   "s": 3
  },
  {
   "t": "unclear = rewrite(\"\", \"그거 바꿀 수 있어?\")",
   "s": 3
  },
  {
   "t": "checks = {t: check_rewrite(\"스터디룸은 하루에 몇 시간까지 쓸 수 있어?\", \"올해도 똑같아?\", t)",
   "s": 3
  },
  {
   "t": "          for t in (\"올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\", \"2025년 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\",",
   "s": 3
  },
  {
   "t": "                    \"스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\")}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 후보 모으기: 몇 개를, 어떤 조건으로",
   "s": 4
  },
  {
   "t": "def search_now(query, category, k):  # 지금 유효한 판만 남기고 k개",
   "s": 4
  },
  {
   "t": "    return [h for h in search(query, category, len(CHUNKS)) if h[\"current\"]][:k]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "q3, gold3 = TESTS[2][1], TESTS[2][3]",
   "s": 4
  },
  {
   "t": "k_sweep = {k: rank_of(gold3, search_now(q3, \"이용\", k)) is not None for k in (3, 5, 10, 20)}",
   "s": 4
  },
  {
   "t": "both_years = search(year_query, \"공지\", k=2)  # 판을 가리지 않으면",
   "s": 4
  },
  {
   "t": "now_only = search_now(year_query, \"공지\", k=2)  # 지금 유효한 판만",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 여러 질문으로 찾기: 용어 넓히기와 합치기",
   "s": 5
  },
  {
   "t": "TERMS = {\"상호대차\": \"다른 도서관 책 받아 보기\", \"취소\": \"철회\"}  # 도서관 용어집: 사람들이 쓰는 말 → 안내문이 쓰는 말",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def expand(query):  # 원래 질문 + 용어집의 말로 바꾼 질문",
   "s": 5
  },
  {
   "t": "    return [query] + [query.replace(a, b) for a, b in TERMS.items() if a in query]",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def multi_search(queries, category, k):  # 질문마다 같은 조건으로 찾고, 같은 조각은 한 번만",
   "s": 5
  },
  {
   "t": "    merged, found_by = {}, {}",
   "s": 5
  },
  {
   "t": "    for q in queries:",
   "s": 5
  },
  {
   "t": "        for h in search_now(q, category, k):",
   "s": 5
  },
  {
   "t": "            merged.setdefault(h[\"id\"], h)",
   "s": 5
  },
  {
   "t": "            found_by.setdefault(h[\"id\"], []).append(q)",
   "s": 5
  },
  {
   "t": "    return list(merged.values()), found_by",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "queries3 = expand(q3)",
   "s": 5
  },
  {
   "t": "per_query = {q: search_now(q, \"이용\", 5) for q in queries3}",
   "s": 5
  },
  {
   "t": "union3, found3 = multi_search(queries3, \"이용\", 5)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 다시 순위 매기기: Cross-Encoder",
   "s": 6
  },
  {
   "t": "if not torch.cuda.is_available():",
   "s": 6
  },
  {
   "t": "    raise RuntimeError(\"CUDA GPU가 필요해요\")  # 조용히 CPU로 바꾸지 않아요",
   "s": 6
  },
  {
   "t": "reranker = CrossEncoder(\"BAAI/bge-reranker-v2-m3\", device=\"cuda:0\")  # 질문과 조각을 함께 읽는 모델",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def rerank(query, hits):  # (질문, 본문) 짝마다 점수를 매겨 높은 순서로. 후보 밖 조각은 못 데려와요",
   "s": 6
  },
  {
   "t": "    ranked = reranker.rank(query, [h[\"body\"] for h in hits], batch_size=16)  # [{\"corpus_id\": 후보 번호, \"score\": 점수}, ...] 높은 순서",
   "s": 6
  },
  {
   "t": "    return [{**hits[r[\"corpus_id\"]], \"rerank\": float(r[\"score\"])} for r in ranked]",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "q4, gold4 = TESTS[3][1], TESTS[3][3]",
   "s": 6
  },
  {
   "t": "cands4 = search_now(q4, \"이용\", CANDIDATES)",
   "s": 6
  },
  {
   "t": "ranked4 = rerank(q4, cands4)",
   "s": 6
  },
  {
   "t": "logits4 = reranker.predict([(q4, h[\"body\"]) for h in cands4], activation_fn=torch.nn.Identity())  # 0~1로 줄이기 전의 값",
   "s": 6
  },
  {
   "t": "ranked3 = rerank(q3, union3)  # ⑤에서 합친 후보",
   "s": 6
  },
  {
   "t": "year_ranked = rerank(year_query, both_years)  # 2025년, 2026년 공지를 둘 다 넣으면",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 하나씩 켜서 비교하기와 걸린 시간",
   "s": 7
  },
  {
   "t": "def find(history, question, category, use_rewrite, use_expand, use_rerank):  # 질문 하나 → 순서가 정해진 후보 목록",
   "s": 7
  },
  {
   "t": "    query = rewrite(history, question) if use_rewrite else question",
   "s": 7
  },
  {
   "t": "    if query is None:",
   "s": 7
  },
  {
   "t": "        return []  # 되묻기: 찾지 않아요",
   "s": 7
  },
  {
   "t": "    hits, _ = multi_search(expand(query) if use_expand else [query], category, CANDIDATES)",
   "s": 7
  },
  {
   "t": "    return rerank(query, hits) if use_rerank else hits",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "SETTINGS = {\"기준\": (False, False, False), \"다시 쓰기만\": (True, False, False), \"재정렬만\": (False, False, True),",
   "s": 7
  },
  {
   "t": "            \"다시 쓰기 + 재정렬\": (True, False, True), \"셋 다 (+ 넓히기)\": (True, True, True)}  # (다시 쓰기, 넓히기, 재정렬)",
   "s": 7
  },
  {
   "t": "compare = {name: [rank_of(gold, find(h, q, cat, *flags)) for h, q, cat, gold in TESTS] for name, flags in SETTINGS.items()}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def timed(fn, *args):  # 걸린 시간 (ms). GPU 계산은 끝날 때까지 기다렸다 재요",
   "s": 7
  },
  {
   "t": "    torch.cuda.synchronize()",
   "s": 7
  },
  {
   "t": "    start = time.perf_counter()",
   "s": 7
  },
  {
   "t": "    out = fn(*args)",
   "s": 7
  },
  {
   "t": "    torch.cuda.synchronize()",
   "s": 7
  },
  {
   "t": "    return out, (time.perf_counter() - start) * 1000",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "times = {\"다시 쓰기\": [], \"검색\": [], \"재정렬\": []}",
   "s": 7
  },
  {
   "t": "for _ in range(20):",
   "s": 7
  },
  {
   "t": "    query, t1 = timed(rewrite, TESTS[0][0], TESTS[0][1])",
   "s": 7
  },
  {
   "t": "    hits, t2 = timed(lambda: multi_search(expand(query), \"이용\", CANDIDATES)[0])",
   "s": 7
  },
  {
   "t": "    _, t3 = timed(rerank, query, hits)",
   "s": 7
  },
  {
   "t": "    for name, ms in zip(times, (t1, t2, t3)):",
   "s": 7
  },
  {
   "t": "        times[name].append(ms)",
   "s": 7
  },
  {
   "t": "cost = {k: [timed(rerank, q4, search_now(q4, \"이용\", k))[1] for _ in range(20)] for k in (5, 10, 17)}",
   "s": 7
  }
 ],
 "names": {
  "1": "도서관 조각을 벡터로 바꾸고 찾기",
  "2": "정답을 정해 두고 실패 나누기",
  "3": "질문 다시 쓰기: 대화에서 빠진 대상 채우기",
  "4": "후보 모으기: 몇 개를, 어떤 조건으로",
  "5": "여러 질문으로 찾기: 용어 넓히기와 합치기",
  "6": "다시 순위 매기기: Cross-Encoder",
  "7": "하나씩 켜서 비교하기와 걸린 시간"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① search(\"대출 기간은 며칠인가요?\", \"이용\", k=3)",
   "cols": [
    "순위",
    "조각",
    "유사도",
    "본문"
   ],
   "rows": [
    [
     1,
     "[1] 대출과 반납",
     "0.891",
     "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번…"
    ],
    [
     2,
     "[17] 대출 중인 책 예약",
     "0.830",
     "다른 사람이 빌려 간 책은 예약해 둘 수 있습니다. 책이 돌아오면 문자로 알려 …"
    ],
    [
     3,
     "[21] 독서 프로그램 신청",
     "0.796",
     "매달 첫째 주 월요일에 다음 달 독서 프로그램 신청을 받습니다. 먼저 신청한 순…"
    ]
   ],
   "notes": [
    "분류 '이용'의 조각 17개 중 질문과 가까운 순서로 3개예요. 1위가 대출 기간이 적힌 조각이에요.",
    "유사도는 두 벡터의 코사인 값이에요. 1에 가까울수록 뜻이 가깝다고 봐요. 다만 이 모델은 관련 없는 글끼리도 0.8 안팎이 나와서, 값 자체보다 <b>순서</b>를 봐요."
   ]
  },
  "t2": {
   "title": "② 질문을 그대로 검색 (후보 10개, 모델에게 줄 3개)",
   "cols": [
    "질문",
    "정답 조각",
    "정답 순위",
    "위 3개",
    "진단"
   ],
   "rows": [
    [
     "그럼 취소는?",
     "[14] 예약 철회",
     "5위",
     "[18] 도서 예약 취소<br>[22] 프로그램 신청 취소<br>[4] 분실과 훼손",
     "<b>순위 밀림</b>"
    ],
    [
     "몇 시간까지 쓸 수 있어?",
     "[23] 노트북 대여",
     "2위",
     "[13] 스터디룸 예약<br>[23] 노트북 대여<br>[1] 대출과 반납",
     "<b>통과</b>"
    ],
    [
     "상호대차 신청하면 며칠 걸려요?",
     "[20] 다른 도서관 책 받아 보기",
     "<b>후보 밖</b>",
     "[21] 독서 프로그램 신청<br>[1] 대출과 반납<br>[22] 프로그램 신청 취소",
     "<b>후보 밖</b>"
    ],
    [
     "예약해 놓고 안 가면 어떻게 돼요?",
     "[15] 노쇼",
     "5위",
     "[14] 예약 철회<br>[18] 도서 예약 취소<br>[17] 대출 중인 책 예약",
     "<b>순위 밀림</b>"
    ],
    [
     "스터디룸 예약을 철회하려면 어떻게 하나요?",
     "[14] 예약 철회",
     "1위",
     "[14] 예약 철회<br>[13] 스터디룸 예약<br>[15] 노쇼",
     "<b>통과</b>"
    ],
    [
     "책을 잃어버리면 어떻게 하나요?",
     "[3] 분실과 훼손",
     "1위",
     "[3] 분실과 훼손<br>[17] 대출 중인 책 예약<br>[18] 도서 예약 취소",
     "<b>통과</b>"
    ]
   ],
   "notes": [
    "'통과' 3개, '순위 밀림' 2개, '후보 밖' 1개예요. 사용자에게는 똑같이 '답이 틀림'으로 보여도, 순위 밀림과 후보 밖은 고칠 곳이 달라요.",
    "노트북 질문은 정답이 2위라 '통과'지만 1위는 [13] 스터디룸 예약이에요. 둘 다 '몇 시간까지'가 들어 있어서, 모델이 스터디룸의 시간을 노트북 답으로 쓸 수도 있어요. 순위 하나로는 다 안 보이는 실패예요.",
    "상호대차 질문은 정답이 11위라 후보 10개에도 못 들어와요. 이건 순서를 바꾸는 단계(⑥)로는 못 고쳐요."
   ]
  },
  "p3": {
   "title": "③ 가짜 LLM에게 보내는 Prompt (첫 질문)",
   "lines": [
    "앞선 질문을 보고, 대화를 모르는 사람도 알아듣는 검색용 질문 하나로 다시 쓰세요.",
    "대화에 없는 대상, 날짜, 숫자, 조건은 더하지 말고, 기간과 부정 표현은 그대로 두세요.",
    "무엇을 가리키는지 정할 수 없으면 needs_clarification을 true로 하세요.",
    "JSON으로 답하세요: {&quot;search_query&quot;: &quot;...&quot;, &quot;needs_clarification&quot;: false}",
    "",
    "앞선 질문: 스터디룸은 어떻게 예약해?",
    "현재 질문: 그럼 취소는?"
   ],
   "notes": []
  },
  "s3": {
   "title": "③ rewrite(앞선 질문, 질문)",
   "lines": [
    [
     "\"그럼 취소는?\" (앞선 질문 \"스터디룸은 어떻게 예약해?\") → \"스터디룸 예약을 취소하는 방법은?\": 정답 5위 → 1위",
     1,
     "newf"
    ],
    [
     "\"몇 시간까지 쓸 수 있어?\" (앞선 질문 \"노트북도 빌릴 수 있어?\") → \"노트북은 몇 시간까지 빌려 쓸 수 있나요?\": 정답 2위 → 1위",
     2,
     "newf"
    ],
    [
     "\"올해도 똑같아?\" (앞선 질문 \"스터디룸은 하루에 몇 시간까지 쓸 수 있어?\") → \"올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\"",
     3
    ],
    [
     "\"그거 바꿀 수 있어?\" (앞선 질문 없음) → None (되묻기)",
     4,
     "bad"
    ]
   ],
   "notes": [
    "앞선 질문의 '스터디룸 예약'을 채웠더니 다른 취소들을 제치고 정답이 1위가 됐어요.",
    "노트북 질문도 '노트북'이 들어가자 스터디룸 조각을 제쳤어요.",
    "'올해'는 그대로 뒀어요. 올해가 몇 년인지는 대화에 없으니 채우지 않아요(④에서 판 조건으로 풀어요).",
    "'그거'가 무엇인지 정할 수 없어요. 상상으로 채우지 않고 None을 돌려줘요. 이때는 검색하지 않고 사용자에게 되물어요."
   ]
  },
  "t3": {
   "title": "③ check_rewrite(앞선 질문, \"올해도 똑같아?\", 다시 쓴 질문)",
   "cols": [
    "다시 쓴 질문",
    "찾은 문제",
    "결과"
   ],
   "rows": [
    [
     "올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?",
     "없음",
     "통과"
    ],
    [
     "2025년 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?",
     "없던 숫자 2025, 빠진 말 올해",
     "<b>되묻기</b>"
    ],
    [
     "스터디룸은 하루에 몇 시간까지 쓸 수 있나요?",
     "빠진 말 올해",
     "<b>되묻기</b>"
    ]
   ],
   "notes": [
    "대화에 있는 말만 썼고 '올해'도 남겼어요. 통과예요.",
    "'올해'를 '2025년'으로 바꿨어요. 대화에 없던 숫자가 생기고 '올해'가 빠졌어요. 기간이 바뀌면 다른 공지를 찾게 돼요.",
    "'올해'를 빼면 연도와 상관없는 질문이 돼요.",
    "이 검사는 숫자와 때를 가리키는 말만 봐요. '수수료 없이'처럼 없는 조건을 붙이거나 '안 된다'를 '된다'로 뒤집는 건 못 잡아요. 그런 건 표본을 사람이 보거나 <a href=\"llm-judge.html\">LLM 채점</a>으로 따로 재요."
   ]
  },
  "t4": {
   "title": "④ \"상호대차 신청하면 며칠 걸려요?\": 후보 수를 바꾸면",
   "cols": [
    "후보 수",
    "정답이 후보에?",
    "⑥의 재정렬이 정답을 고를 수 있나"
   ],
   "rows": [
    [
     3,
     "없음",
     "<b>없음</b>"
    ],
    [
     5,
     "없음",
     "<b>없음</b>"
    ],
    [
     10,
     "없음",
     "<b>없음</b>"
    ],
    [
     20,
     "있음",
     "있음"
    ]
   ],
   "notes": [
    "정답은 '이용' 조각 17개 중 11위예요. 후보를 20개로 늘리면 들어오지만, '이용' 조각이 17개뿐이라 전부 가져온 거예요. 조각이 수만 개라면 후보를 늘려서 풀기 어려워요.",
    "후보를 늘리면 재정렬할 짝이 늘어 시간이 늘어요(⑦). 그래서 후보 수만 키우지 말고, 정답이 후보에 들어오게 질문을 고쳐요(⑤)."
   ]
  },
  "s4": {
   "title": "④ \"올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\" (분류 '공지', 위 2개)",
   "lines": [
    [
     "판을 가리지 않으면 search",
     0
    ],
    [
     "  [26] 2026년 스터디룸 운영 안내 유사도 0.907",
     1
    ],
    [
     "  [25] 2025년 스터디룸 운영 안내 유사도 0.905",
     1
    ],
    [
     "지금 유효한 판만 search_now",
     0
    ],
    [
     "  [26] 2026년 스터디룸 운영 안내 유사도 0.907",
     2,
     "newf"
    ],
    [
     "  [5] 겨울방학 운영 안내 유사도 0.820",
     2,
     "newf"
    ]
   ],
   "notes": [
    "두 해의 공지가 유사도 0.002 차이로 붙어 있어요. 문장 모양이 거의 같아서, 점수로는 어느 해가 지금인지 가릴 수 없어요. 둘 다 모델에게 가면 '2시간'과 '3시간'이 같이 들어가요.",
    "판 표시(current)로 거르면 2025년 공지가 빠지고 다른 공지가 그 자리에 와요. 이런 조건은 질문을 어떻게 바꾸든 <b>모든 검색에 똑같이</b> 붙여요(⑤의 multi_search도 search_now를 써요)."
   ]
  },
  "t5": {
   "title": "⑤ 질문마다 위 5개 (search_now)",
   "cols": [
    "질문",
    "찾은 조각"
   ],
   "rows": [
    [
     "상호대차 신청하면 며칠 걸려요?",
     "[21] 독서 프로그램 신청, [1] 대출과 반납, [22] 프로그램 신청 취소, [14] 예약 철회, [4] 분실과 훼손"
    ],
    [
     "다른 도서관 책 받아 보기 신청하면 며칠 걸려요?",
     "<b>[20] 다른 도서관 책 받아 보기</b>, [19] 희망 도서, [17] 대출 중인 책 예약, [23] 노트북 대여, [18] 도서 예약 취소"
    ]
   ],
   "notes": [
    "원래 질문으로는 정답이 위 5개에 없어요. 용어집으로 '상호대차'를 안내문의 말로 바꾼 질문에서는 정답이 1위예요.",
    "둘을 합치면 10개예요(두 질문이 함께 찾은 조각 0개). 같은 조각은 한 번만 남기고, 어느 질문이 찾았는지(found_by)를 기록해 둬요.",
    "합친 순서는 '먼저 찾은 순서'라 정답은 6번째예요. 모델에게 줄 3개 안에는 아직 못 들어요. 후보에 넣는 데까지가 여러 질문의 일이고, 순서는 ⑥에서 정해요."
   ]
  },
  "t6": {
   "title": "⑥ \"예약해 놓고 안 가면 어떻게 돼요?\": 후보 10개를 재정렬",
   "cols": [
    "조각",
    "e5 순위",
    "e5 유사도",
    "재정렬 순위",
    "재정렬 점수",
    "logit"
   ],
   "rows": [
    [
     "<b>[15] 노쇼</b>",
     "5위",
     "0.839",
     "1위",
     "0.7430",
     "1.06"
    ],
    [
     "[17] 대출 중인 책 예약",
     "3위",
     "0.849",
     "2위",
     "0.0777",
     "-2.47"
    ],
    [
     "[14] 예약 철회",
     "1위",
     "0.853",
     "3위",
     "0.0202",
     "-3.88"
    ],
    [
     "[22] 프로그램 신청 취소",
     "8위",
     "0.818",
     "4위",
     "0.0070",
     "-4.95"
    ],
    [
     "[2] 열람실",
     "6위",
     "0.832",
     "5위",
     "0.0010",
     "-6.89"
    ],
    [
     "[18] 도서 예약 취소",
     "2위",
     "0.850",
     "6위",
     "0.0005",
     "-7.69"
    ],
    [
     "[13] 스터디룸 예약",
     "7위",
     "0.821",
     "7위",
     "0.0001",
     "-9.68"
    ],
    [
     "[21] 독서 프로그램 신청",
     "9위",
     "0.815",
     "8위",
     "0.0000",
     "-9.97"
    ],
    [
     "[4] 분실과 훼손",
     "10위",
     "0.813",
     "9위",
     "0.0000",
     "-10.09"
    ],
    [
     "[16] 이용 규칙",
     "4위",
     "0.840",
     "10위",
     "0.0000",
     "-10.87"
    ]
   ],
   "notes": [
    "e5에서 5위였던 '노쇼'가 1위가 됐어요. e5 유사도로는 후보들이 0.813~0.853에 몰려 있었는데, 재정렬 점수는 1위만 0.743이고 나머지는 훨씬 낮아요. 질문과 본문을 같이 읽어서 '안 가면'과 '연락 없이 오지 않으면'이 같은 상황인 걸 알아봤어요.",
    "이 모델의 재정렬 점수는 logit(모델이 낸 원래 값)을 0~1로 줄인 거예요(sigmoid). 모델마다 달라서, logit을 그대로 내는 모델도 있어요. 1위의 logit 1.06이 0.743이 됐어요. 확률처럼 보여도 '맞을 확률'이 아니에요. 같은 질문 안에서 순서를 정하는 데만 써요.",
    "e5 유사도와 재정렬 점수는 다른 모델이 다른 방법으로 낸 값이라 숫자끼리 맞대어 비교하지 않아요."
   ]
  },
  "s6": {
   "title": "⑥ 다른 두 질문을 재정렬하면",
   "lines": [
    [
     "\"상호대차 신청하면 며칠 걸려요?\" + ⑤에서 합친 후보 10개 → 1위 [20] 다른 도서관 책 받아 보기 점수 0.0021, 2위 [4] 분실과 훼손 0.0014",
     1
    ],
    [
     "\"올해 스터디룸은 하루에 몇 시간까지 쓸 수 있나요?\" + 두 해의 공지 → [26] 2026년 스터디룸 운영 안내 0.9975, [25] 2025년 스터디룸 운영 안내 0.9959",
     2,
     "bad"
    ]
   ],
   "notes": [
    "정답이 후보에 들어오니 1위로 올랐어요. 그런데 점수가 0.0021로 아주 낮아요. 이 모델도 '상호대차'를 잘 몰라서, 다른 후보가 더 낮았을 뿐이에요. 점수가 낮은 1위를 그대로 믿지 말고, 기준 점수 아래면 '근거 부족'으로 다루는 방법도 있어요(기준은 평가 질문으로 정해요).",
    "두 공지 모두 거의 최고점이에요. 더 정밀한 모델도 어느 해가 지금인지는 몰라요. 최신 여부와 열람 권한은 점수가 아니라 ④의 조건으로 정해요."
   ]
  },
  "t7": {
   "title": "⑦ 정답 순위 (진하게: 모델에게 줄 3개 안)",
   "cols": [
    "설정",
    "그럼 취소는?",
    "몇 시간까지 쓸 수 있어?",
    "상호대차 신청하면 며칠 걸려요?",
    "예약해 놓고 안 가면 어떻게 돼요?",
    "스터디룸 예약을 철회하려면 어떻게 하나요?",
    "책을 잃어버리면 어떻게 하나요?",
    "3개 안"
   ],
   "rows": [
    [
     "기준",
     "5위",
     "<b>2위</b>",
     "<b>후보 밖</b>",
     "5위",
     "<b>1위</b>",
     "<b>1위</b>",
     "3/6"
    ],
    [
     "다시 쓰기만",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>후보 밖</b>",
     "5위",
     "<b>1위</b>",
     "<b>1위</b>",
     "4/6"
    ],
    [
     "재정렬만",
     "<b>3위</b>",
     "<b>2위</b>",
     "<b>후보 밖</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "5/6"
    ],
    [
     "다시 쓰기 + 재정렬",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>후보 밖</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "5/6"
    ],
    [
     "셋 다 (+ 넓히기)",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "<b>1위</b>",
     "6/6"
    ]
   ],
   "notes": [
    "다시 쓰기는 앞선 질문이 필요한 두 질문을, 재정렬은 '안 가면' 질문을, 넓히기는 상호대차 질문을 고쳐요. 하나씩 켜 봐야 무엇이 무엇을 고쳤는지 보여요.",
    "재정렬만 켜도 '그럼 취소는?'의 정답이 3위로 들어오지만, 질문 자체가 뜻이 비어 있어서 다시 쓰기를 켰을 때만 1위가 돼요.",
    "상호대차 질문은 넓히기 없이는 어떤 설정에서도 후보 밖이에요. 재정렬은 후보 밖 조각을 데려오지 못해요.",
    "질문 6개는 성질을 보이려고 고른 예예요. 실제로 켤지는 질문을 많이 모아 같은 방법으로 재 보고 정해요."
   ]
  },
  "t7b": {
   "title": "⑦ 걸린 시간 (\"그럼 취소는?\", 셋 다 켬, 20번 잰 가운데 값)",
   "cols": [
    "단계",
    "시간",
    "무엇이 드나"
   ],
   "rows": [
    [
     "다시 쓰기",
     "0.1 ms",
     "가짜 LLM이라 거의 0이에요. 실제로는 네트워크를 거쳐 모델이 글을 만드는 시간이 더해져요"
    ],
    [
     "검색",
     "14.6 ms",
     "질문 2개(원래 질문과 '철회'로 바꾼 질문)를 e5로 벡터로 만들고 26개와 비교 (GPU)"
    ],
    [
     "재정렬",
     "46.5 ms",
     "후보마다 (질문, 본문) 짝을 Cross-Encoder로 (GPU)"
    ],
    [
     "재정렬, 후보 5개",
     "32.4 ms",
     ""
    ],
    [
     "재정렬, 후보 10개",
     "46.9 ms",
     ""
    ],
    [
     "재정렬, 후보 17개",
     "103.6 ms",
     ""
    ]
   ],
   "notes": [
    "이 셋 중에는 재정렬이 가장 오래 걸려요. 후보 17개는 104 ms로, 5개나 10개(32 ms, 47 ms)보다 확 늘어요. 짝을 한 번에 16개씩(batch_size) 묶어 계산해서, 16개를 넘으면 한 묶음을 더 돌리거든요.",
    "실제 서비스에서는 다시 쓰기와 답변에 LLM 호출이 들어가서 보통 그쪽이 더 길어요. 그래서 짧고 뜻이 분명한 질문은 다시 쓰기를 건너뛰는 정책을 쓰기도 해요(건너뛴 사실은 기록해요).",
    "GPU 작업은 명령만 걸어 두고 바로 다음 줄로 넘어갈 수 있어요. 그래서 재기 전후에 <code>torch.cuda.synchronize()</code>로 끝날 때까지 기다려요. 시간은 실행할 때마다 조금씩 달라요(NVIDIA GeForce RTX 5080 Laptop GPU)."
   ]
  }
 },
 "flow": [
  [
   1,
   "찾기",
   "질문과 가까운 조각",
   "e5 벡터 검색"
  ],
  [
   2,
   "실패 나누기",
   "정답 순위로",
   "통과 3/6"
  ],
  [
   3,
   "다시 쓰기",
   "빠진 대상 채우기",
   "모르면 되묻기"
  ],
  [
   4,
   "후보 모으기",
   "후보 수, 판 조건",
   "후보 밖은 못 고름"
  ],
  [
   5,
   "여러 질문",
   "용어 넓히고 합치기",
   "같은 조각은 한 번"
  ],
  [
   6,
   "재정렬",
   "Cross-Encoder",
   "질문과 본문을 함께"
  ],
  [
   7,
   "비교",
   "하나씩 켜기, 시간",
   "3/6 → 6/6"
  ]
 ],
 "env": "Python 3.13.16, torch 2.14.0+cu130, sentence-transformers 6.3.0, transformers 5.22.2, NVIDIA GeForce RTX 5080 Laptop GPU, intfloat/multilingual-e5-small, BAAI/bge-reranker-v2-m3"
};
