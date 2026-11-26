window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import math",
   "s": 4
  },
  {
   "t": "import re",
   "s": 5
  },
  {
   "t": "import time",
   "s": 2
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 2
  },
  {
   "t": "import torch",
   "s": 2
  },
  {
   "t": "from sentence_transformers import SentenceTransformer",
   "s": 2
  },
  {
   "t": "from sentence_transformers import CrossEncoder",
   "s": 2
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
   "t": "DATA = HERE.parent / \"rag\" / \"data\"",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 정답 묶음 만들고 검사하기",
   "s": 1
  },
  {
   "t": "CHUNKS = (json.loads((HERE.parent / \"ingest\" / \"out\" / \"chunks.json\").read_text(encoding=\"utf-8\"))",
   "s": 1
  },
  {
   "t": "          + json.loads((DATA / \"more_chunks.json\").read_text(encoding=\"utf-8\")))",
   "s": 1
  },
  {
   "t": "TITLE = {c[\"chunk_id\"]: c[\"heading\"] or c[\"body\"].split(\"\\n\")[0] for c in CHUNKS}  # 조각 ID → 제목",
   "s": 1
  },
  {
   "t": "GOLD = json.loads((DATA / \"gold.json\").read_text(encoding=\"utf-8\"))",
   "s": 1
  },
  {
   "t": "seen = set()",
   "s": 1
  },
  {
   "t": "for row in GOLD[\"questions\"]:  # 정답 묶음의 기본 약속부터 검사",
   "s": 1
  },
  {
   "t": "    assert row[\"question_id\"] not in seen, \"질문 ID가 겹쳐요\"",
   "s": 1
  },
  {
   "t": "    seen.add(row[\"question_id\"])",
   "s": 1
  },
  {
   "t": "    assert set(row[\"gold\"]) <= set(TITLE), \"없는 조각 ID\"",
   "s": 1
  },
  {
   "t": "    assert all(g in (1, 2) for g in row[\"gold\"].values()), \"등급은 1이나 2\"",
   "s": 1
  },
  {
   "t": "    assert (row[\"expected_status\"] == \"insufficient_context\") == (not row[\"gold\"]), \"정답 조각이 없으면 '정보 부족'이 맞는 답\"",
   "s": 1
  },
  {
   "t": "gold_of = {row[\"question_id\"]: row[\"gold\"] for row in GOLD[\"questions\"]}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 평가할 검색을 돌리고 기록 남기기",
   "s": 2
  },
  {
   "t": "CATEGORY = {\"guide.md\": \"이용\", \"hours.html\": \"이용\", \"room.md\": \"이용\", \"holds.md\": \"이용\", \"programs.md\": \"이용\", \"devices.md\": \"이용\",",
   "s": 2
  },
  {
   "t": "            \"notice.txt\": \"공지\", \"old_notice.txt\": \"공지\", \"notice_2025.txt\": \"공지\", \"notice_2026.txt\": \"공지\",",
   "s": 2
  },
  {
   "t": "            \"rules.pdf\": \"규정\", \"scan.pdf\": \"시설\"}",
   "s": 2
  },
  {
   "t": "encoder = SentenceTransformer(\"intfloat/multilingual-e5-small\")",
   "s": 2
  },
  {
   "t": "vectors = encoder.encode([f\"passage: {TITLE[c['chunk_id']]}\\n{c['body']}\" for c in CHUNKS], normalize_embeddings=True)",
   "s": 2
  },
  {
   "t": "reranker = CrossEncoder(\"BAAI/bge-reranker-v2-m3\", device=\"cuda:0\")",
   "s": 2
  },
  {
   "t": "BODY = {c[\"chunk_id\"]: c[\"body\"] for c in CHUNKS}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def search(query, category, k=10):  # e5로 분류 안의 지금 유효한 조각 중 가까운 k개 (RAG 검색 고치기 ①④)",
   "s": 2
  },
  {
   "t": "    q = encoder.encode([f\"query: {query}\"], normalize_embeddings=True)[0]",
   "s": 2
  },
  {
   "t": "    order = [i for i in np.argsort(-(vectors @ q)) if CATEGORY[CHUNKS[i][\"file\"]] == category and CHUNKS[i].get(\"current\", True)]",
   "s": 2
  },
  {
   "t": "    return [CHUNKS[i][\"chunk_id\"] for i in order[:k]]",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def rerank(query, ids):  # 질문과 본문을 함께 읽어 다시 줄 세우기 (RAG 검색 고치기 ⑥)",
   "s": 2
  },
  {
   "t": "    ranked = reranker.rank(query, [BODY[i] for i in ids], batch_size=16)  # 점수 높은 순서의 [{\"corpus_id\", \"score\"}, ...]",
   "s": 2
  },
  {
   "t": "    return [ids[r[\"corpus_id\"]] for r in ranked]",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "REWRITTEN = {\"Q03\": \"스터디룸 예약을 취소하는 방법은?\", \"Q04\": \"노트북은 몇 시간까지 빌려 쓸 수 있나요?\"}  # 다시 쓰기 단계가 냈다고 칠 검색용 질문",
   "s": 2
  },
  {
   "t": "SETTINGS = {\"A e5만\": (False, False), \"B 재정렬\": (False, True), \"C 다시 쓰기 + 재정렬\": (True, True)}  # (다시 쓴 질문, 재정렬)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def run(row, setting):  # 평가할 대상을 한 번 실행하고 무엇이 나왔는지 기록해요. 정답은 넣지 않아요",
   "s": 2
  },
  {
   "t": "    use_rewrite, use_rerank = SETTINGS[setting]",
   "s": 2
  },
  {
   "t": "    query = REWRITTEN.get(row[\"question_id\"], row[\"question\"]) if use_rewrite else row[\"question\"]",
   "s": 2
  },
  {
   "t": "    torch.cuda.synchronize()",
   "s": 2
  },
  {
   "t": "    start = time.perf_counter()",
   "s": 2
  },
  {
   "t": "    ids = search(query, row[\"category\"])",
   "s": 2
  },
  {
   "t": "    if use_rerank:",
   "s": 2
  },
  {
   "t": "        ids = rerank(query, ids)",
   "s": 2
  },
  {
   "t": "    torch.cuda.synchronize()",
   "s": 2
  },
  {
   "t": "    return {\"question_id\": row[\"question_id\"], \"setting\": setting, \"query\": query, \"ranked\": ids, \"context\": ids[:3],",
   "s": 2
  },
  {
   "t": "            \"latency_ms\": round((time.perf_counter() - start) * 1000, 1)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "run(GOLD[\"questions\"][0], \"B 재정렬\")  # 처음 한 번은 GPU 준비 시간이 섞여서 버려요",
   "s": 2
  },
  {
   "t": "records = [run(row, s) for s in SETTINGS for row in GOLD[\"questions\"]]",
   "s": 2
  },
  {
   "t": "by_setting = {s: [r for r in records if r[\"setting\"] == s] for s in SETTINGS}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 찾았나: Hit@k, Recall@k, Precision@k",
   "s": 3
  },
  {
   "t": "def hit_at_k(ids, gold, k):  # 앞 k개에 정답이 하나라도 있으면 1",
   "s": 3
  },
  {
   "t": "    return None if not gold else int(bool(set(ids[:k]) & set(gold)))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def recall_at_k(ids, gold, k):  # 정답 중 몇 개를 찾았나 (분모: 정답 수)",
   "s": 3
  },
  {
   "t": "    return None if not gold else len(set(ids[:k]) & set(gold)) / len(gold)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def precision_at_k(ids, gold, k):  # 가져온 k개 중 몇 개가 정답인가 (분모: k)",
   "s": 3
  },
  {
   "t": "    return None if not gold else len(set(ids[:k]) & set(gold)) / k",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def mean(values):  # N/A(None)는 빼고 평균",
   "s": 3
  },
  {
   "t": "    vals = [v for v in values if v is not None]",
   "s": 3
  },
  {
   "t": "    return sum(vals) / len(vals) if vals else None",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "A = by_setting[\"A e5만\"]",
   "s": 3
  },
  {
   "t": "k_curve = {k: [mean(f(r[\"ranked\"], gold_of[r[\"question_id\"]], k) for r in A) for f in (hit_at_k, recall_at_k, precision_at_k)]",
   "s": 3
  },
  {
   "t": "           for k in (1, 3, 5, 10)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 앞에 뒀나: MRR과 nDCG",
   "s": 4
  },
  {
   "t": "def reciprocal_rank(ids, gold, k=10):  # 첫 정답이 r번째면 1/r, k 안에 없으면 0",
   "s": 4
  },
  {
   "t": "    if not gold:",
   "s": 4
  },
  {
   "t": "        return None",
   "s": 4
  },
  {
   "t": "    return next((1 / rank for rank, i in enumerate(ids[:k], start=1) if i in gold), 0.0)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def dcg(ids, grades, k):  # 등급이 높을수록 크게, 뒤로 갈수록 log로 덜 쳐요",
   "s": 4
  },
  {
   "t": "    return sum((2 ** grades.get(i, 0) - 1) / math.log2(rank + 1) for rank, i in enumerate(ids[:k], start=1))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def ndcg_at_k(ids, grades, k=10):  # 가장 좋은 순서로 놓았을 때의 DCG로 나눠 0~1로",
   "s": 4
  },
  {
   "t": "    best = dcg(sorted(grades, key=grades.get, reverse=True), grades, k)",
   "s": 4
  },
  {
   "t": "    return None if best == 0 else dcg(ids, grades, k) / best",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "order_scores = {(r[\"setting\"], r[\"question_id\"]): (reciprocal_rank(r[\"ranked\"], gold_of[r[\"question_id\"]]), ndcg_at_k(r[\"ranked\"], gold_of[r[\"question_id\"]]))",
   "s": 4
  },
  {
   "t": "                for r in records}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 답은 맞나: 상태, 근거, 핵심 사실",
   "s": 5
  },
  {
   "t": "REVIEWS = {r[\"question_id\"]: r for r in json.loads((DATA / \"answer_reviews.json\").read_text(encoding=\"utf-8\"))[\"reviews\"]}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def answer_of(qid):  # 평가할 서비스의 답. 검토표에 없는 질문은 '정보 부족'으로 답했다고 쳐요",
   "s": 5
  },
  {
   "t": "    review = REVIEWS.get(qid)",
   "s": 5
  },
  {
   "t": "    return (\"answered\", review[\"answer\"]) if review else (\"insufficient_context\", \"제공된 자료로는 답할 수 없습니다.\")",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def faithfulness(qid):  # 사람이 주장마다 판정한 표로: 근거로 확인된 주장 ÷ 전체 주장",
   "s": 5
  },
  {
   "t": "    review = REVIEWS.get(qid)",
   "s": 5
  },
  {
   "t": "    return None if review is None else sum(c[\"supported\"] for c in review[\"claims\"]) / len(review[\"claims\"])",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def number_rule(qid, context):  # 사람 대신 빠르게 해 보는 규칙: 주장 속 숫자가 넣은 조각에 모두 있나",
   "s": 5
  },
  {
   "t": "    text = \"\\n\".join(BODY[i] for i in context)",
   "s": 5
  },
  {
   "t": "    return [all(n in text for n in re.findall(r\"\\d+\", c[\"text\"])) for c in REVIEWS[qid][\"claims\"]]",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def fact_recall(answer, facts):  # 정답의 핵심 사실이 답에 글자로 들어 있나",
   "s": 5
  },
  {
   "t": "    return None if not facts else sum(f in answer for f in facts) / len(facts)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "C = {r[\"question_id\"]: r for r in by_setting[\"C 다시 쓰기 + 재정렬\"]}",
   "s": 5
  },
  {
   "t": "answers = {}",
   "s": 5
  },
  {
   "t": "for row in GOLD[\"questions\"]:",
   "s": 5
  },
  {
   "t": "    status, text = answer_of(row[\"question_id\"])",
   "s": 5
  },
  {
   "t": "    answers[row[\"question_id\"]] = {\"status\": status, \"answer\": text, \"status_ok\": status == row[\"expected_status\"],",
   "s": 5
  },
  {
   "t": "                                   \"faithfulness\": faithfulness(row[\"question_id\"]),",
   "s": 5
  },
  {
   "t": "                                   \"facts\": fact_recall(text, row[\"reference_facts\"]) if status == \"answered\" else None,",
   "s": 5
  },
  {
   "t": "                                   \"human\": [c[\"supported\"] for c in REVIEWS[row[\"question_id\"]][\"claims\"]] if row[\"question_id\"] in REVIEWS else None,",
   "s": 5
  },
  {
   "t": "                                   \"rule\": number_rule(row[\"question_id\"], C[row[\"question_id\"]][\"context\"]) if row[\"question_id\"] in REVIEWS else None}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 설정끼리 비교하고 고칠 곳 찾기",
   "s": 6
  },
  {
   "t": "def score(r):  # 실행 기록 하나 + 정답 → 검색 점수",
   "s": 6
  },
  {
   "t": "    g = gold_of[r[\"question_id\"]]",
   "s": 6
  },
  {
   "t": "    return {\"candidate_recall\": recall_at_k(r[\"ranked\"], g, 10), \"context_recall@3\": recall_at_k(r[\"context\"], g, 3),",
   "s": 6
  },
  {
   "t": "            \"mrr\": reciprocal_rank(r[\"ranked\"], g), \"ndcg\": ndcg_at_k(r[\"ranked\"], g)}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "scores = {(r[\"setting\"], r[\"question_id\"]): score(r) for r in records}",
   "s": 6
  },
  {
   "t": "summary = {s: {**{m: mean(scores[s, r[\"question_id\"]][m] for r in rows) for m in (\"candidate_recall\", \"context_recall@3\", \"mrr\", \"ndcg\")},",
   "s": 6
  },
  {
   "t": "               \"p50_ms\": float(np.percentile([r[\"latency_ms\"] for r in rows], 50)), \"p95_ms\": float(np.percentile([r[\"latency_ms\"] for r in rows], 95))}",
   "s": 6
  },
  {
   "t": "           for s, rows in by_setting.items()}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def changed(a, b, metric):  # 같은 질문끼리 짝지어 오른 것과 내린 것 (평균만 보면 나빠진 질문이 숨어요)",
   "s": 6
  },
  {
   "t": "    pairs = [(q, scores[a, q][metric], scores[b, q][metric]) for q in gold_of if scores[a, q][metric] is not None]",
   "s": 6
  },
  {
   "t": "    return {\"better\": [q for q, u, v in pairs if v > u + 1e-9], \"worse\": [q for q, u, v in pairs if v < u - 1e-9]}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "deltas = {s: changed(\"A e5만\", s, \"context_recall@3\") for s in (\"B 재정렬\", \"C 다시 쓰기 + 재정렬\")}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def failure_type(qid):  # C 설정에서, 앞 단계부터 차례로: 먼저 막힌 곳이 원인이에요",
   "s": 6
  },
  {
   "t": "    sc, ans = scores[\"C 다시 쓰기 + 재정렬\", qid], answers[qid]",
   "s": 6
  },
  {
   "t": "    if gold_of[qid] and not sc[\"candidate_recall\"]:",
   "s": 6
  },
  {
   "t": "        return \"후보 밖\"",
   "s": 6
  },
  {
   "t": "    if gold_of[qid] and not sc[\"context_recall@3\"]:",
   "s": 6
  },
  {
   "t": "        return \"모델에게 줄 3개에 못 듦\"",
   "s": 6
  },
  {
   "t": "    if not ans[\"status_ok\"]:",
   "s": 6
  },
  {
   "t": "        return \"답 상태 틀림\"",
   "s": 6
  },
  {
   "t": "    if ans[\"faithfulness\"] is not None and ans[\"faithfulness\"] < 1:",
   "s": 6
  },
  {
   "t": "        return \"근거 없는 주장\"",
   "s": 6
  },
  {
   "t": "    return \"통과\"",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "failures = {qid: failure_type(qid) for qid in gold_of}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 품질 기준: 다음 단계로 내보내기 전에",
   "s": 7
  },
  {
   "t": "GATE = {\"context_recall@3\": 0.8, \"mrr\": 0.7, \"status_ok\": 1.0, \"faithfulness\": 0.9}  # 예시 기준 (팀마다 정해요)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def quality_gate(setting):  # 기준을 모두 넘었나, 틀리면 곤란한 것은 평균이 아니라 0개로",
   "s": 7
  },
  {
   "t": "    values = {\"context_recall@3\": summary[setting][\"context_recall@3\"], \"mrr\": summary[setting][\"mrr\"],",
   "s": 7
  },
  {
   "t": "              \"status_ok\": mean(float(a[\"status_ok\"]) for a in answers.values()), \"faithfulness\": mean(a[\"faithfulness\"] for a in answers.values())}",
   "s": 7
  },
  {
   "t": "    checks = {m: (values[m], need, values[m] >= need) for m, need in GATE.items()}",
   "s": 7
  },
  {
   "t": "    unsupported = [q for q, a in answers.items() if a[\"faithfulness\"] is not None and a[\"faithfulness\"] < 1]",
   "s": 7
  },
  {
   "t": "    checks[\"근거 없는 주장이 있는 답\"] = (len(unsupported), 0, not unsupported)",
   "s": 7
  },
  {
   "t": "    checks[\"A보다 나빠진 질문\"] = (len(deltas[setting][\"worse\"]), 0, not deltas[setting][\"worse\"])",
   "s": 7
  },
  {
   "t": "    return all(ok for _, _, ok in checks.values()), checks, unsupported",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "gate = quality_gate(\"C 다시 쓰기 + 재정렬\")",
   "s": 7
  }
 ],
 "names": {
  "1": "정답 묶음 만들고 검사하기",
  "2": "평가할 검색을 돌리고 기록 남기기",
  "3": "찾았나: Hit@k, Recall@k, Precision@k",
  "4": "앞에 뒀나: MRR과 nDCG",
  "5": "답은 맞나: 상태, 근거, 핵심 사실",
  "6": "설정끼리 비교하고 고칠 곳 찾기",
  "7": "품질 기준: 다음 단계로 내보내기 전에"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 정답 묶음 (판 gold-2026-11-26, 질문 15개 중 일부)",
   "cols": [
    "질문 ID",
    "질문",
    "정답 조각 (등급)",
    "기대 상태",
    "핵심 사실"
   ],
   "rows": [
    [
     "Q01",
     "대출 기간은 며칠인가요?",
     "대출과 반납 (2)",
     "answered",
     "14일"
    ],
    [
     "Q03",
     "그럼 취소는?<br><small>앞선 질문: 스터디룸은 어떻게 예약해?</small>",
     "예약 철회 (2)",
     "answered",
     "1시간 전, 나의 신청"
    ],
    [
     "Q06",
     "예약해 놓고 안 가면 어떻게 돼요?",
     "노쇼 (2)<br>스터디룸 예약 (1)",
     "answered",
     "노쇼, 30일"
    ],
    [
     "Q14",
     "연체료는 하루에 얼마인가요?",
     "없음",
     "insufficient_context",
     "-"
    ]
   ],
   "notes": [
    "정답 조각이 둘인 질문이 6개 있어요. 바로 답하는 조각은 등급 2, 일부만 관련된 조각은 1이에요.",
    "Q14처럼 정답 조각이 없는 질문이 2개예요. 이런 질문은 '자료로는 답할 수 없다'고 멈추는 게 맞는 답이에요.",
    "핵심 사실은 맞는 답에 꼭 들어가야 할 말이에요(⑤의 정답성). 위의 검사(assert)는 이 묶음이 약속을 지키는지 실행할 때마다 확인해요.",
    "묶음에는 묶음의 판(gold-2026-11-26)과 조각의 판(garam-26-chunks)을 같이 적어 뒀어요. 조각을 다시 나누면 같은 조각 ID가 다른 내용을 가리키거나 사라질 수 있어서, 판이 다른 점수끼리는 나란히 비교하지 않아요."
   ]
  },
  "s2": {
   "title": "② run(Q03, \"C 다시 쓰기 + 재정렬\")가 남긴 기록",
   "lines": [
    [
     "question_id \"Q03\", setting \"C 다시 쓰기 + 재정렬\"",
     1
    ],
    [
     "query \"스터디룸 예약을 취소하는 방법은?\"",
     2
    ],
    [
     "ranked → 10개, 앞 3개: 예약 철회, 스터디룸 예약, 노쇼",
     3
    ],
    [
     "context → 예약 철회, 스터디룸 예약, 노쇼",
     4
    ],
    [
     "latency_ms 57.0",
     5
    ]
   ],
   "notes": [
    "어느 질문을 어느 설정으로 돌렸는지예요. 설정 셋 × 질문 15개 = 45줄이 쌓여요.",
    "실제로 검색한 질문이에요. C 설정은 다시 쓴 질문을 써요.",
    "검색(과 재정렬)이 돌려준 순서예요. ③④의 검색 점수는 이 목록으로 매겨요.",
    "그중 모델에게 줄 위 3개예요.",
    "이 실행에 걸린 시간이에요."
   ]
  },
  "t3": {
   "title": "③ A e5만: k를 바꾸면 (정답이 있는 질문 13개의 평균)",
   "cols": [
    "k",
    "Hit@k",
    "Recall@k",
    "Precision@k"
   ],
   "rows": [
    [
     1,
     "0.69",
     "0.50",
     "0.69"
    ],
    [
     3,
     "0.77",
     "0.73",
     "0.36"
    ],
    [
     5,
     "0.92",
     "0.85",
     "0.25"
    ],
    [
     10,
     "0.92",
     "0.92",
     "0.14"
    ]
   ],
   "notes": [
    "k를 늘리면 Hit와 Recall은 오르고(놓치는 게 줄어요) Precision은 내려요(정답 아닌 것도 같이 와요). k=10에서 Recall 0.92, Precision 0.14예요.",
    "k=1에서 Hit와 Precision이 같은 건, 하나만 볼 때는 '하나라도 맞나'와 '맞은 비율'이 같은 뜻이라서예요.",
    "정답 조각이 없는 질문 2개는 분모가 없어서 N/A로 빼고 평균했어요. 그 질문들은 ⑤에서 '정보 부족이라고 답했나'로 따로 재요."
   ]
  },
  "t3b": {
   "title": "③ A e5만: 같은 공식, 다른 목록",
   "cols": [
    "질문 ID",
    "질문",
    "Recall@10 (검색 후보)",
    "Recall@3 (모델에게 줄 것)"
   ],
   "rows": [
    [
     "Q03",
     "그럼 취소는?",
     "1.00",
     "0.00"
    ],
    [
     "Q06",
     "예약해 놓고 안 가면 어떻게 돼요?",
     "1.00",
     "0.00"
    ],
    [
     "Q07",
     "책을 잃어버리면 어떻게 하나요?",
     "1.00",
     "0.50"
    ]
   ],
   "notes": [
    "Q03, Q06, Q07은 검색이 정답을 모두 후보에 담았는데(Recall@10 = 1), 모델에게 줄 3개에는 다 못 들어갔어요. 앞 목록으로 재면 '검색이 찾았나', 뒤 목록으로 재면 '모델이 볼 수 있었나'예요. 표에는 어느 목록인지 k와 함께 꼭 적어요."
   ]
  },
  "t4": {
   "title": "④ A e5만 → B 재정렬: 값이 바뀐 질문만 (k=10)",
   "cols": [
    "질문 ID",
    "질문",
    "1/r (A)",
    "1/r (B)",
    "nDCG (A)",
    "nDCG (B)"
   ],
   "rows": [
    [
     "Q03",
     "그럼 취소는?",
     "0.20",
     "0.33",
     "0.39",
     "0.50"
    ],
    [
     "Q06",
     "예약해 놓고 안 가면 어떻게 돼요?",
     "0.20",
     "1.00",
     "0.41",
     "0.92"
    ],
    [
     "Q07",
     "책을 잃어버리면 어떻게 하나요?",
     "1.00",
     "1.00",
     "0.92",
     "0.96"
    ],
    [
     "Q09",
     "열람실은 주말에도 여나요?",
     "1.00",
     "1.00",
     "0.80",
     "1.00"
    ],
    [
     "Q10",
     "독서 프로그램 신청을 취소하려면 어떻게 하나요?",
     "1.00",
     "1.00",
     "1.00",
     "0.96"
    ]
   ],
   "notes": [
    "재정렬로 첫 정답이 앞으로 온 질문은 Q03, Q06예요. Q06은 첫 정답이 5위(1/5 = 0.20)에서 1위(1.00)가 됐어요.",
    "Q10은 첫 정답은 그대로 1위인데 nDCG가 조금 내려갔어요. 등급 1인 두 번째 정답의 자리가 밀렸거든요. MRR은 첫 정답만 보고, nDCG는 전체 순서와 등급을 봐서 이런 차이가 보여요.",
    "평균만 보면 이런 질문이 묻혀요. 질문별로 오르고 내린 것을 같이 봐요(⑥)."
   ]
  },
  "t5": {
   "title": "⑤ 답 점수 (답 문장은 검토표의 것, 근거는 C 설정에서 모델에게 줄 3개)",
   "cols": [
    "질문 ID",
    "답",
    "상태 맞나",
    "충실성",
    "핵심 사실"
   ],
   "rows": [
    [
     "Q04",
     "노트북은 한 번에 4시간까지 쓸 수 있고, 회원증 없이도 빌릴 수 있습니다.",
     "맞음",
     "0.50",
     "1.00"
    ],
    [
     "Q09",
     "열람실은 주말에 열지 않습니다.",
     "맞음",
     "1.00",
     "0.00"
    ],
    [
     "Q12",
     "흑백은 한 장에 100원, 컬러는 한 장에 200원입니다.",
     "맞음",
     "0.50",
     "0.00"
    ],
    [
     "Q14",
     "제공된 자료로는 답할 수 없습니다.",
     "맞음",
     "N/A",
     "N/A"
    ]
   ],
   "notes": [
    "Q04의 '회원증 없이도'는 근거에 없는 말이에요. 안내문은 '회원증을 맡기고 빌린다'고 해요. 주장 둘 중 하나라 충실성이 0.50이에요. 문장 하나로 맞다/틀리다를 매기면 맞는 절반 때문에 틀린 절반이 묻혀요.",
    "Q09는 '주말에 열지 않는다'로 맞게 답했는데 핵심 사실 점수가 0이에요. 정답에는 '휴실'로 적어 뒀는데 답이 다른 말을 썼거든요. 글자 대조는 같은 뜻을 다른 말로 쓰면 놓쳐요.",
    "Q12는 흑백 50원을 100원이라고 했어요. 숫자가 틀린 답이에요.",
    "Q14는 정답 조각이 없는 질문이라 '정보 부족'으로 멈췄으면 맞아요. 충실성과 핵심 사실은 잴 답이 없어 N/A예요."
   ]
  },
  "t5b": {
   "title": "⑤ 사람 판정 vs 빠른 숫자 규칙 (주장마다)",
   "cols": [
    "질문 ID",
    "주장",
    "사람",
    "숫자 규칙"
   ],
   "rows": [
    [
     "Q04",
     "노트북은 한 번에 4시간까지 쓸 수 있다",
     "확인",
     "통과"
    ],
    [
     "Q04",
     "회원증 없이도 빌릴 수 있다",
     "<b>근거 없음</b>",
     "통과"
    ],
    [
     "Q05",
     "신청한 뒤 보통 3일에서 5일 안에 도착한다",
     "확인",
     "통과"
    ],
    [
     "Q12",
     "흑백은 한 장에 100원",
     "<b>근거 없음</b>",
     "<b>걸림</b>"
    ],
    [
     "Q12",
     "컬러는 한 장에 200원",
     "확인",
     "통과"
    ]
   ],
   "notes": [
    "숫자 규칙은 Q12의 100원을 잡았어요(근거에 100이 없음). 하지만 숫자가 없는 Q04의 '회원증 없이도'는 못 잡아요.",
    "Q05는 C 설정의 근거 3개에 정답 조각이 없는데도 통과했어요. '3'과 '5'가 다른 조각(5권, 3일 안에)에 있었기 때문이에요. 숫자가 '어딘가에' 있는지만 봐서 생기는 헛통과예요.",
    "그래서 자동 규칙은 빠른 1차 거름으로만 쓰고, 사람 판정이나 <a href=\"llm-judge.html\">LLM 채점</a>과 갈리는 답을 표본으로 봐요."
   ]
  },
  "t6": {
   "title": "⑥ 설정마다 평균 (정답이 있는 질문만, 시간은 15개 모두)",
   "cols": [
    "설정",
    "후보 Recall@10",
    "Context Recall@3",
    "MRR",
    "nDCG@10",
    "시간 중앙값",
    "시간 95%"
   ],
   "rows": [
    [
     "A e5만",
     "0.92",
     "0.73",
     "0.76",
     "0.78",
     "7 ms",
     "28 ms"
    ],
    [
     "B 재정렬",
     "0.92",
     "0.88",
     "0.83",
     "0.84",
     "59 ms",
     "90 ms"
    ],
    [
     "C 다시 쓰기 + 재정렬",
     "0.92",
     "0.88",
     "0.92",
     "0.91",
     "52 ms",
     "61 ms"
    ]
   ],
   "notes": [
    "후보 Recall은 셋 다 0.92로 같아요. 재정렬과 다시 쓰기는 후보 안의 순서와 질문을 바꿨을 뿐, 후보 밖(Q05 상호대차)을 데려오지 못했어요.",
    "재정렬(B)은 Context Recall@3을 0.73에서 0.88로, 다시 쓰기까지 더한 C는 MRR을 0.92까지 올렸어요.",
    "대신 시간 중앙값이 7 ms에서 59 ms로 늘어요(GPU에서 후보 10개를 한 번 더 읽어요). 점수와 시간을 같이 보고 정해요.",
    "A에서 B로 바꿔 Context Recall@3이 오른 질문은 Q03, Q06, Q07이고, 내린 질문은 없어요."
   ]
  },
  "t6b": {
   "title": "⑥ C 다시 쓰기 + 재정렬에서 남은 실패 (앞 단계부터 확인해 먼저 막힌 곳)",
   "cols": [
    "질문 ID",
    "질문",
    "먼저 막힌 곳",
    "고칠 곳"
   ],
   "rows": [
    [
     "Q04",
     "몇 시간까지 쓸 수 있어?",
     "<b>근거 없는 주장</b>",
     "Prompt의 '자료만 근거로', 답 검사, 출처 확인"
    ],
    [
     "Q05",
     "상호대차 신청하면 며칠 걸려요?",
     "<b>후보 밖</b>",
     "질문 넓히기, 조각 나누기, 검색 범위 (<a href=\"rag-rerank.html\">RAG 검색 고치기</a> ⑤)"
    ],
    [
     "Q12",
     "복사는 한 장에 얼마예요?",
     "<b>근거 없는 주장</b>",
     "Prompt의 '자료만 근거로', 답 검사, 출처 확인"
    ]
   ],
   "notes": [
    "Q05는 답 문장이 맞게 나왔어도 '후보 밖'이에요. 이 예제의 답은 검토표에 고정돼 있어서 그래요. 실제 모델이었다면 근거 없이 답한 셈이니, 검색에서 먼저 막힌 걸 고쳐요.",
    "앞 단계부터 확인하는 이유는, 검색이 정답을 못 가져왔는데 답만 보고 Prompt를 고치면 헛수고가 되기 때문이에요."
   ]
  },
  "t7": {
   "title": "⑦ quality_gate(\"C 다시 쓰기 + 재정렬\") → 막힘",
   "cols": [
    "확인할 것",
    "값",
    "기준",
    "결과"
   ],
   "rows": [
    [
     "context_recall@3",
     "0.88",
     0.8,
     "통과"
    ],
    [
     "mrr",
     "0.92",
     0.7,
     "통과"
    ],
    [
     "status_ok",
     "1.00",
     1.0,
     "통과"
    ],
    [
     "faithfulness",
     "0.92",
     0.9,
     "통과"
    ],
    [
     "근거 없는 주장이 있는 답",
     2,
     0,
     "<b>막힘</b>"
    ],
    [
     "A보다 나빠진 질문",
     0,
     0,
     "통과"
    ]
   ],
   "notes": [
    "충실성 평균은 0.92로 기준 0.9을 넘었어요. 그래도 근거 없는 주장이 있는 답이 2개(Q04, Q12)라 막혔어요.",
    "틀리면 곤란한 것(지어낸 내용, 권한 위반)은 평균 기준이 아니라 '0개'로 둬요. 평균은 좋은 답 열한 개가 나쁜 답 둘을 가려 줘요.",
    "기준 숫자는 예시예요. 팀이 서비스의 위험도에 맞게 정하고, 바꾸면 이유와 날짜를 남겨요."
   ]
  }
 },
 "flow": [
  [
   1,
   "정답 묶음",
   "결과 전에 정하기",
   "질문 15개"
  ],
  [
   2,
   "실행 기록",
   "정답은 넣지 않고",
   "설정 3개"
  ],
  [
   3,
   "찾았나",
   "Hit, Recall, Precision",
   "어느 목록으로?"
  ],
  [
   4,
   "앞에 뒀나",
   "MRR, nDCG",
   "첫 정답, 전체 순서"
  ],
  [
   5,
   "답은 맞나",
   "상태, 충실성, 핵심 사실",
   "자동 검사의 한계"
  ],
  [
   6,
   "비교",
   "설정별, 질문별",
   "남은 실패 3"
  ],
  [
   7,
   "품질 기준",
   "평균과 0개 조건",
   "막힘"
  ]
 ],
 "env": "Python 3.13.16, torch 2.14.0+cu130, sentence-transformers 6.3.0, numpy 2.3.5, NVIDIA GeForce RTX 5080 Laptop GPU, intfloat/multilingual-e5-small, BAAI/bge-reranker-v2-m3"
};
