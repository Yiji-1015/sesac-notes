# RAG 평가: 결과를 보기 전에 정해 둔 정답 묶음으로 찾기와 답을 따로 재요. 이 파일 하나로 실행돼요.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 검색 대상: 가람 도서관 조각 26개 (../ingest/out/chunks.json + ../rag/data/more_chunks.json, 이 노트용 가상 자료)
# 정답 묶음: ../rag/data/gold.json (질문 15개), 답 검토표: ../rag/data/answer_reviews.json (사람이 주장마다 확인한 표)
# 검색은 intfloat/multilingual-e5-small, 재정렬은 BAAI/bge-reranker-v2-m3 (CUDA GPU). 둘 다 이 PC에 받아 둔 모델이에요.
# 평가할 서비스의 '답 문장'은 LLM을 부르지 않고 검토표에 적힌 답을 그대로 써요(일부러 틀린 답 둘이 섞여 있어요).
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
import math                                                      #@4
import re                                                        #@5
import time                                                      #@2
from pathlib import Path                                         #@1
import numpy as np                                               #@2
import torch                                                     #@2
from sentence_transformers import SentenceTransformer            #@2
from sentence_transformers import CrossEncoder                   #@2

HERE = Path(__file__).resolve().parent                           #@1
DATA = HERE.parent / "rag" / "data"                              #@1

# ① 정답 묶음 만들고 검사하기
CHUNKS = (json.loads((HERE.parent / "ingest" / "out" / "chunks.json").read_text(encoding="utf-8"))
          + json.loads((DATA / "more_chunks.json").read_text(encoding="utf-8")))
TITLE = {c["chunk_id"]: c["heading"] or c["body"].split("\n")[0] for c in CHUNKS}   # 조각 ID → 제목
GOLD = json.loads((DATA / "gold.json").read_text(encoding="utf-8"))
seen = set()
for row in GOLD["questions"]:                                    # 정답 묶음의 기본 약속부터 검사
    assert row["question_id"] not in seen, "질문 ID가 겹쳐요"
    seen.add(row["question_id"])
    assert set(row["gold"]) <= set(TITLE), "없는 조각 ID"
    assert all(g in (1, 2) for g in row["gold"].values()), "등급은 1이나 2"
    assert (row["expected_status"] == "insufficient_context") == (not row["gold"]), "정답 조각이 없으면 '정보 부족'이 맞는 답"
gold_of = {row["question_id"]: row["gold"] for row in GOLD["questions"]}

# ② 평가할 검색을 돌리고 기록 남기기
CATEGORY = {"guide.md": "이용", "hours.html": "이용", "room.md": "이용", "holds.md": "이용", "programs.md": "이용", "devices.md": "이용",
            "notice.txt": "공지", "old_notice.txt": "공지", "notice_2025.txt": "공지", "notice_2026.txt": "공지",
            "rules.pdf": "규정", "scan.pdf": "시설"}
encoder = SentenceTransformer("intfloat/multilingual-e5-small")
vectors = encoder.encode([f"passage: {TITLE[c['chunk_id']]}\n{c['body']}" for c in CHUNKS], normalize_embeddings=True)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda:0")
BODY = {c["chunk_id"]: c["body"] for c in CHUNKS}

def search(query, category, k=10):                               # e5로 분류 안의 지금 유효한 조각 중 가까운 k개 (RAG 검색 고치기 ①④)
    q = encoder.encode([f"query: {query}"], normalize_embeddings=True)[0]
    order = [i for i in np.argsort(-(vectors @ q)) if CATEGORY[CHUNKS[i]["file"]] == category and CHUNKS[i].get("current", True)]
    return [CHUNKS[i]["chunk_id"] for i in order[:k]]

def rerank(query, ids):                                          # 질문과 본문을 함께 읽어 다시 줄 세우기 (RAG 검색 고치기 ⑥)
    scores = reranker.predict([(query, BODY[i]) for i in ids], batch_size=16)
    return [i for _, i in sorted(zip(scores, ids), key=lambda p: -float(p[0]))]

REWRITTEN = {"Q03": "스터디룸 예약을 취소하는 방법은?", "Q04": "노트북은 몇 시간까지 빌려 쓸 수 있나요?"}   # 다시 쓰기 단계가 냈다고 칠 검색용 질문
SETTINGS = {"A e5만": (False, False), "B 재정렬": (False, True), "C 다시 쓰기 + 재정렬": (True, True)}   # (다시 쓴 질문, 재정렬)

def run(row, setting):                                           # 평가할 대상을 한 번 실행하고 무엇이 나왔는지 기록해요. 정답은 넣지 않아요
    use_rewrite, use_rerank = SETTINGS[setting]
    query = REWRITTEN.get(row["question_id"], row["question"]) if use_rewrite else row["question"]
    torch.cuda.synchronize()
    start = time.perf_counter()
    ids = search(query, row["category"])
    if use_rerank:
        ids = rerank(query, ids)
    torch.cuda.synchronize()
    return {"question_id": row["question_id"], "setting": setting, "query": query, "ranked": ids, "context": ids[:3],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1)}

run(GOLD["questions"][0], "B 재정렬")                            # 처음 한 번은 GPU 준비 시간이 섞여서 버려요
records = [run(row, s) for s in SETTINGS for row in GOLD["questions"]]
by_setting = {s: [r for r in records if r["setting"] == s] for s in SETTINGS}

# ③ 찾았나: Hit@k, Recall@k, Precision@k
def hit_at_k(ids, gold, k):                                      # 앞 k개에 정답이 하나라도 있으면 1
    return None if not gold else int(bool(set(ids[:k]) & set(gold)))

def recall_at_k(ids, gold, k):                                   # 정답 중 몇 개를 찾았나 (분모: 정답 수)
    return None if not gold else len(set(ids[:k]) & set(gold)) / len(gold)

def precision_at_k(ids, gold, k):                                # 가져온 k개 중 몇 개가 정답인가 (분모: k)
    return None if not gold else len(set(ids[:k]) & set(gold)) / k

def mean(values):                                                # N/A(None)는 빼고 평균
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None

A = by_setting["A e5만"]
k_curve = {k: [mean(f(r["ranked"], gold_of[r["question_id"]], k) for r in A) for f in (hit_at_k, recall_at_k, precision_at_k)]
           for k in (1, 3, 5, 10)}

# ④ 앞에 뒀나: MRR과 nDCG
def reciprocal_rank(ids, gold, k=10):                            # 첫 정답이 r번째면 1/r, k 안에 없으면 0
    if not gold:
        return None
    return next((1 / rank for rank, i in enumerate(ids[:k], start=1) if i in gold), 0.0)

def dcg(ids, grades, k):                                         # 등급이 높을수록 크게, 뒤로 갈수록 log로 덜 쳐요
    return sum((2 ** grades.get(i, 0) - 1) / math.log2(rank + 1) for rank, i in enumerate(ids[:k], start=1))

def ndcg_at_k(ids, grades, k=10):                                # 가장 좋은 순서로 놓았을 때의 DCG로 나눠 0~1로
    best = dcg(sorted(grades, key=grades.get, reverse=True), grades, k)
    return None if best == 0 else dcg(ids, grades, k) / best

order_scores = {(r["setting"], r["question_id"]): (reciprocal_rank(r["ranked"], gold_of[r["question_id"]]), ndcg_at_k(r["ranked"], gold_of[r["question_id"]]))
                for r in records}

# ⑤ 답은 맞나: 상태, 근거, 핵심 사실
REVIEWS = {r["question_id"]: r for r in json.loads((DATA / "answer_reviews.json").read_text(encoding="utf-8"))["reviews"]}

def answer_of(qid):                                              # 평가할 서비스의 답. 검토표에 없는 질문은 '정보 부족'으로 답했다고 쳐요
    review = REVIEWS.get(qid)
    return ("answered", review["answer"]) if review else ("insufficient_context", "제공된 자료로는 답할 수 없습니다.")

def faithfulness(qid):                                           # 사람이 주장마다 판정한 표로: 근거로 확인된 주장 ÷ 전체 주장
    review = REVIEWS.get(qid)
    return None if review is None else sum(c["supported"] for c in review["claims"]) / len(review["claims"])

def number_rule(qid, context):                                   # 사람 대신 빠르게 해 보는 규칙: 주장 속 숫자가 넣은 조각에 모두 있나
    text = "\n".join(BODY[i] for i in context)
    return [all(n in text for n in re.findall(r"\d+", c["text"])) for c in REVIEWS[qid]["claims"]]

def fact_recall(answer, facts):                                  # 정답의 핵심 사실이 답에 글자로 들어 있나
    return None if not facts else sum(f in answer for f in facts) / len(facts)

C = {r["question_id"]: r for r in by_setting["C 다시 쓰기 + 재정렬"]}
answers = {}
for row in GOLD["questions"]:
    status, text = answer_of(row["question_id"])
    answers[row["question_id"]] = {"status": status, "answer": text, "status_ok": status == row["expected_status"],
                                   "faithfulness": faithfulness(row["question_id"]),
                                   "facts": fact_recall(text, row["reference_facts"]) if status == "answered" else None,
                                   "human": [c["supported"] for c in REVIEWS[row["question_id"]]["claims"]] if row["question_id"] in REVIEWS else None,
                                   "rule": number_rule(row["question_id"], C[row["question_id"]]["context"]) if row["question_id"] in REVIEWS else None}

# ⑥ 설정끼리 비교하고 고칠 곳 찾기
def score(r):                                                    # 실행 기록 하나 + 정답 → 검색 점수
    g = gold_of[r["question_id"]]
    return {"candidate_recall": recall_at_k(r["ranked"], g, 10), "context_recall@3": recall_at_k(r["context"], g, 3),
            "mrr": reciprocal_rank(r["ranked"], g), "ndcg": ndcg_at_k(r["ranked"], g)}

scores = {(r["setting"], r["question_id"]): score(r) for r in records}
summary = {s: {**{m: mean(scores[s, r["question_id"]][m] for r in rows) for m in ("candidate_recall", "context_recall@3", "mrr", "ndcg")},
               "p50_ms": float(np.percentile([r["latency_ms"] for r in rows], 50)), "p95_ms": float(np.percentile([r["latency_ms"] for r in rows], 95))}
           for s, rows in by_setting.items()}

def changed(a, b, metric):                                       # 같은 질문끼리 짝지어 오른 것과 내린 것 (평균만 보면 나빠진 질문이 숨어요)
    pairs = [(q, scores[a, q][metric], scores[b, q][metric]) for q in gold_of if scores[a, q][metric] is not None]
    return {"better": [q for q, u, v in pairs if v > u + 1e-9], "worse": [q for q, u, v in pairs if v < u - 1e-9]}

deltas = {s: changed("A e5만", s, "context_recall@3") for s in ("B 재정렬", "C 다시 쓰기 + 재정렬")}

def failure_type(qid):                                           # C 설정에서, 앞 단계부터 차례로: 먼저 막힌 곳이 원인이에요
    sc, ans = scores["C 다시 쓰기 + 재정렬", qid], answers[qid]
    if gold_of[qid] and not sc["candidate_recall"]:
        return "후보 밖"
    if gold_of[qid] and not sc["context_recall@3"]:
        return "모델에게 줄 3개에 못 듦"
    if not ans["status_ok"]:
        return "답 상태 틀림"
    if ans["faithfulness"] is not None and ans["faithfulness"] < 1:
        return "근거 없는 주장"
    return "통과"

failures = {qid: failure_type(qid) for qid in gold_of}

# ⑦ 품질 기준: 다음 단계로 내보내기 전에
GATE = {"context_recall@3": 0.8, "mrr": 0.7, "status_ok": 1.0, "faithfulness": 0.9}   # 예시 기준 (팀마다 정해요)

def quality_gate(setting):                                       # 기준을 모두 넘었나, 틀리면 곤란한 것은 평균이 아니라 0개로
    values = {"context_recall@3": summary[setting]["context_recall@3"], "mrr": summary[setting]["mrr"],
              "status_ok": mean(float(a["status_ok"]) for a in answers.values()), "faithfulness": mean(a["faithfulness"] for a in answers.values())}
    checks = {m: (values[m], need, values[m] >= need) for m, need in GATE.items()}
    unsupported = [q for q, a in answers.items() if a["faithfulness"] is not None and a["faithfulness"] < 1]
    checks["근거 없는 주장이 있는 답"] = (len(unsupported), 0, not unsupported)
    checks["A보다 나빠진 질문"] = (len(deltas[setting]["worse"]), 0, not deltas[setting]["worse"])
    return all(ok for _, _, ok in checks.values()), checks, unsupported

gate = quality_gate("C 다시 쓰기 + 재정렬")
