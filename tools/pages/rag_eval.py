"""백엔드 > 검색과 RAG > RAG 평가. examples/rag/rag_service.py 의 ⑰~㉔ (①~⑧은 RAG 서비스, ⑨~⑯은 RAG 검색 고치기 쪽).
정답 묶음(examples/rag/data/gold.json, 질문 15개)과 답 검토표(answer_reviews.json)는 결과를 보기 전에 사람이 정한 것.
검색은 e5-small, 재정렬은 bge-reranker-v2-m3(CUDA GPU)를 실제로 돌리고, 답은 ChatOpenAI에 가짜 서버를 붙여 받는다(미리 정한 문장).
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/rag_eval.py
"""
import html
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = (ROOT / "examples" / "rag").resolve()
code = S.parse((EX / "rag_service.py").read_text(encoding="utf-8"))
ns = S.run(code, 24, EX, filename="rag_service.py")
import torch  # noqa: E402
V = {p: version(p) for p in ("langchain-core", "langchain-openai", "sentence-transformers", "torch")}
ENV = (f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items())
       + f", {torch.cuda.get_device_name(0)}, intfloat/multilingual-e5-small, BAAI/bge-reranker-v2-m3")
E, J = html.escape, K.josa
outs = {}
G = ns["GOLD"]
QS = G["questions"]
gold_of, rows_of = ns["gold_of"], ns["rows_of"]
f3 = lambda v: "N/A" if v is None else f"{v:.3f}"  # noqa: E731
f2 = lambda v: "N/A" if v is None else f"{v:.2f}"  # noqa: E731
HEAD = {c["chunk_id"]: (c["heading"] or c["file"]) for c in ns["CHUNKS"]}


def names(ids):
    return ", ".join(E(HEAD[i]) for i in ids) or "없음"


# ---------------- ⑰
no_answer = [q["question_id"] for q in QS if not q["gold"]]
S.claim(len(QS) == 15 and no_answer == ["Q14", "Q15"], "⑰ 질문 15개, 답 없는 질문 둘")
rows17 = [[q["question_id"], E(q["question"]) + (f"<br><small>앞선 질문: {E(q['history'][0])}</small>" if q["history"] else ""), q["category"],
           ", ".join(f"{E(HEAD[c])} ({g})" for c, g in q["gold"].items()) or "<b>없음</b>", q["expected_status"]] for q in QS]
outs["s17"] = S.table(f"⑰ 정답 묶음 {G['dataset_version']} (조각 {len(ns['CHUNKS'])}개 기준)", ["ID", "질문", "분류", "정답 조각 (등급)", "기대 상태"], rows17,
                      [f"질문 {len(QS)}개 중 {len(no_answer)}개({', '.join(no_answer)})는 자료에 답이 없는 질문이에요. 이런 질문에서 시스템이 '정보 부족'이라고 답하는지도 재야 해서 일부러 넣어요.",
                       "등급 2는 질문에 바로 답하는 조각, 1은 관련 있지만 답의 일부만 있는 조각이에요. 예를 들어 Q02(철회)에서 '예약 철회'는 2, '스터디룸 예약'은 1이에요. 적지 않은 조각은 0으로 봐요.",
                       "정답 조각은 실행마다 바뀌는 숫자 번호가 아니라 바뀌지 않는 chunk_id로 적었어요. 문서가 새 판으로 바뀌면 정답 묶음도 다시 검토해야 해서 corpus_version을 같이 적어요."])
rec = next(r for r in ns["base_records"] if r["question_id"] == "Q03")
outs["s17b"] = S.box("A 기준 설정으로 Q03을 한 번 실행한 기록 (run_target)", [f"question_id: {rec['question_id']}, setting: {rec['setting']}", f"search_query: \"{E(rec['search_query'])}\"",
                                                                    f"candidate_ids (10개): {E(names(rec['candidate_ids'][:4]))}, …", f"ranked_ids: 재정렬을 안 켜서 candidate_ids와 같은 순서",
                                                                    f"included_ids (넣은 {len(rec['included_ids'])}개): {E(names(rec['included_ids']))}", f"status: {rec['status']}, answer: \"{E(rec['answer'])}\"",
                                                                    f"latency_ms: {rec['latency_ms']}"])
S.claim(rec["ranked_ids"] == rec["candidate_ids"] and rec["status"] == "insufficient_context", "⑰ Q03 기준은 정보 부족")

# ---------------- ⑱
kc, cvc = ns["k_curve"], ns["cand_vs_ctx"]
S.claim(kc[1][1] < kc[3][1] < kc[5][1] <= kc[10][1] and kc[1][2] > kc[3][2] > kc[5][2] > kc[10][2], "⑱ k가 늘면 Recall은 오르고 Precision은 내린다")
outs["t18a"] = S.table("⑱ A 기준, 정답이 있는 질문 13개의 평균", ["k", "Hit@k", "Recall@k", "Precision@k"], [[k, f3(h), f3(r), f3(p)] for k, (h, r, p) in kc.items()],
                       [f"k를 1에서 10으로 늘리면 Recall은 {kc[1][1]:.2f}에서 {kc[10][1]:.2f}까지 오르고, Precision은 {kc[1][2]:.2f}에서 {kc[10][2]:.2f}까지 내려가요. 더 많이 가져오면 놓치는 건 줄지만, 가져온 것 중 정답 비율은 떨어져요.",
                        f"Hit@k는 정답 조각이 하나라도 있으면 1이라, 정답이 둘인 질문({', '.join(q['question_id'] for q in QS if len(q['gold']) == 2)})에서 하나만 찾아도 1이에요. 그래서 늘 Recall@k 이상이에요."])
diff = [q for q, (c, x) in cvc.items() if c is not None and c > x]
S.claim(set(diff) == {"Q03", "Q06", "Q07"}, "⑱ 후보에는 있는데 넣은 3개에서 빠진 질문은 Q03, Q06, Q07")
outs["t18b"] = S.table("후보 전체 Recall과 실제로 넣은 3개의 Recall (A 기준)", ["질문", "후보 Recall", "넣은 3개 Recall (Context Recall@3)", "넣은 조각"],
                       [[q, f2(c), f2(x), names(next(r["included_ids"] for r in ns["base_records"] if r["question_id"] == q))] for q, (c, x) in cvc.items() if c is not None and c != x],
                       [f"{', '.join(diff)}는 정답 조각이 후보 10개 안에는 있었는데, 넣을 3개를 고를 때 빠졌어요. 같은 Recall 공식이라도 <b>어느 목록으로 쟀는지</b>(candidate_ids인지 included_ids인지)를 꼭 적어야 해요.",
                        "후보 Recall은 높은데 Context Recall이 낮으면 검색을 넓힐 게 아니라 순서(재정렬)나 넣을 수를 봐요(⑲, ㉓)."])

# ---------------- ⑲
rp = ns["rank_pairs"]
same_cand = all(v["candidate_recall"][0] == v["candidate_recall"][1] for v in rp.values())
mrr_a, mrr_c = ns["mean"](v["mrr@10"][0] for v in rp.values()), ns["mean"](v["mrr@10"][1] for v in rp.values())
nd_worse = [q for q, v in rp.items() if v["ndcg@10"][0] is not None and v["ndcg@10"][1] < v["ndcg@10"][0] - 1e-9]
first_grade = lambda recs, q: gold_of[q].get(next(r["ranked_ids"][0] for r in recs if r["question_id"] == q), 0)  # noqa: E731
pos = lambda recs, q, g: next(r["ranked_ids"] for r in recs if r["question_id"] == q).index(g) + 1  # noqa: E731
S.claim(all(rp[q]["mrr@10"] == (1.0, 1.0) and any(pos(ns["rerank_records"], q, g) > pos(ns["base_records"], q, g) for g in gold_of[q]) for q in nd_worse), "⑲ nDCG가 내려간 질문은 첫 정답은 그대로, 다른 정답 조각이 뒤로")
S.claim(rp["Q09"]["mrr@10"] == (1.0, 1.0) and rp["Q09"]["ndcg@10"][1] == 1.0 and first_grade(ns["base_records"], "Q09") == 1 and first_grade(ns["rerank_records"], "Q09") == 2, "⑲ Q09는 MRR 그대로, 등급 2가 1위로 와서 nDCG 1")
S.claim(same_cand and mrr_c > mrr_a and rp["Q06"]["mrr@10"] == (0.2, 1.0) and nd_worse, "⑲ 재정렬은 후보 Recall은 그대로, MRR은 오르고, Q06은 0.2→1, nDCG가 내려간 질문도 있다")
outs["t19"] = S.table("⑲ 같은 후보를 재정렬만 했을 때 (A 기준 → C 재정렬만)", ["질문", "후보 Recall", "MRR@10", "nDCG@10"],
                      [[q, f"{f2(v['candidate_recall'][0])} → {f2(v['candidate_recall'][1])}", f"{f3(v['mrr@10'][0])} → <b>{f3(v['mrr@10'][1])}</b>" if v["mrr@10"][0] != v["mrr@10"][1] else f3(v["mrr@10"][0]),
                        f"{f3(v['ndcg@10'][0])} → <b>{f3(v['ndcg@10'][1])}</b>" if v["ndcg@10"][0] != v["ndcg@10"][1] else f3(v["ndcg@10"][0])] for q, v in rp.items() if v["mrr@10"][0] is not None],
                      [f"재정렬은 후보 안의 순서만 바꿔서 후보 Recall은 질문마다 그대로예요. 그런데 MRR 평균은 {mrr_a:.3f}에서 {mrr_c:.3f}로 올라요. Q06(노쇼)은 정답이 5위에서 1위로 올라와 0.2에서 1.0이 됐어요.",
                       "Q09는 MRR이 두 번 다 1이에요(첫 조각이 두 번 다 정답). 그런데 재정렬이 등급 2인 운영 시간 표를 등급 1인 열람실 안내 앞으로 올려서 nDCG가 1.0이 됐어요. 첫 정답만 보는 MRR은 못 보는 차이예요.",
                       f"{', '.join(nd_worse)}는 거꾸로 nDCG가 조금 내려갔어요. 1위는 그대로인데, 관련 없는 조각이 2위로 끼어들어 다른 정답 조각이 한 칸 뒤로 밀렸어요. 평균이 올라도 질문마다 보면 나빠진 곳이 있어요.",
                       "Q05(상호대차)는 정답 조각이 후보에 아예 없어서 순서를 아무리 바꿔도 0이에요. 후보 밖의 정답은 재정렬로 살릴 수 없어요."])

# ---------------- ⑳
ae = ns["answer_eval"]
claims_rows, agree, total = [], 0, 0
for q, v in ae.items():
    if v["human"] is None:
        continue
    for c, h, r in zip(ns["REVIEWS"][q]["claims"], v["human"], v["rule"]):
        total += 1
        agree += h == r
        if not h or not r:
            claims_rows.append([q, E(c["text"]), "확인됨" if h else "<b>근거 없음</b>", "통과" if r else "<b>걸림</b>"])
S.claim(ae["Q04"]["human"] == [True, False] and ae["Q04"]["rule"] == [True, True] and ae["Q12"]["human"][0] is False and ae["Q12"]["rule"][0] is False, "⑳ 숫자 규칙은 Q12는 잡고 Q04는 놓친다")
S.claim(ae["Q04"]["faithfulness"] == ae["Q12"]["faithfulness"] == 0.5 and ae["Q04"]["citation_ok"] and ae["Q12"]["citation_ok"], "⑳ Q04, Q12는 0.5, 출처는 맞음")
S.claim(ae["Q09"]["faithfulness"] == 1.0 and ae["Q09"]["facts"] == 0.0, "⑳ Q09는 충실하지만 핵심 사실 글자 대조는 0")
S.claim(ae["Q14"]["status_ok"] and ae["Q15"]["status_ok"] and not ae["Q05"]["status_ok"], "⑳ 답 없는 질문은 맞게 정보 부족, Q05는 틀림")
outs["t20"] = S.table("⑳ D(다시 쓰기 + 재정렬)의 답을 질문마다", ["질문", "상태", "기대와 같나", "출처가 넣은 조각 안인가", "충실성", "핵심 사실"],
                      [[q, v["status"], "예" if v["status_ok"] else "<b>아니요</b>", "예" if v["citation_ok"] else "<b>아니요</b>", f2(v["faithfulness"]), f2(v["facts"])] for q, v in ae.items()],
                      ["충실성(faithfulness)은 답을 주장 단위로 나눠, 넣은 조각에서 확인되는 주장의 비율이에요. Q04와 Q12는 주장 둘 중 하나가 근거 없어서 0.5예요. 출처 번호는 둘 다 맞아요(출처 검사 통과).",
                       "핵심 사실은 정답 묶음에 적어 둔 말이 답에 글자로 들어 있는지예요. Q09는 '휴실'을 '열지 않습니다'로 바꿔 쓴 맞는 답인데 0이에요. 글자 대조는 빠르지만 바꿔 말하기에 약해요.",
                       "Q14, Q15는 답이 없는 질문이라 정보 부족이 정답이에요. Q05는 답이 있는데 정답 조각을 못 찾아서(⑲) 정보 부족이라고 답했어요. 생성은 잘못한 게 없고 검색 실패가 상태 오류로 나타난 거예요."])
outs["t20b"] = S.table(f"사람 판정과 숫자 규칙이 갈리거나 근거 없는 주장 (주장 {total}개 중 {agree}개 일치)", ["질문", "주장", "사람", "숫자 규칙"], claims_rows,
                       ["숫자 규칙은 '주장 속 숫자가 넣은 조각에 모두 있나'만 봐요. Q12의 100원(자료는 50원)은 잡았지만, Q04의 '회원증 없이도'(자료는 '회원증을 맡기고')는 숫자가 없어서 놓쳤어요.",
                        "그래서 자동 규칙이나 <a href=\"llm-judge.html\">LLM 채점</a>은 사람 검토를 줄이는 보조로 쓰고, 둘이 갈리는 답은 사람이 봐요. 답 문장이 바뀌면 이전 판정은 쓰지 않고(N/A) 다시 검토해요."])

# ---------------- ㉑
rep = ns["report"]
lines = (EX / "out" / "eval_runs.jsonl").read_text(encoding="utf-8").splitlines()
S.claim(len(rep) == len(lines) == 15 * len(ns["SETTINGS_EVAL"]), "㉑ 질문 15 × 설정 5 = 75줄")
one = json.loads(lines[next(i for i, x in enumerate(rep) if x["question_id"] == "Q06" and x["setting"].startswith("C"))])
show = {k: one[k] for k in ("dataset_version", "flags", "question_id", "setting", "status", "candidate_recall", "context_recall@3", "mrr@10", "ndcg@10", "status_ok", "citation_ok", "faithfulness", "latency_ms")}
outs["s21"] = S.box(f"㉑ out/eval_runs.jsonl ({len(lines)}줄 중 C 재정렬만, Q06 한 줄에서 일부 칸)", [E(json.dumps(show, ensure_ascii=False))],
                    [f"질문 {len(QS)}개 × 설정 {len(ns['SETTINGS_EVAL'])}개 = {len(lines)}줄이에요. 한 줄에 무엇을 넣었고(질문, 설정, 판), 무엇이 나왔고(후보, 넣은 조각, 답), 몇 점인지가 다 있어서, 나중에 이 질문 하나만 다시 따라가 볼 수 있어요."])

# ---------------- ㉒
sm, dl = ns["summary"], ns["deltas"]
order = list(ns["SETTINGS_EVAL"])
best = max(order, key=lambda s: sm[s]["context_recall@3"])
S.claim(best.startswith("E") and all(not dl[s]["context_recall@3"]["worse"] for s in dl) and all(not dl[s]["mrr@10"]["worse"] for s in dl), "㉒ E가 가장 좋고, A보다 나빠진 질문이 없다")
S.claim(min(sm[s]["p50_ms"] for s in order if "재정렬" in s or s.startswith("E")) > 2 * sm["A 기준"]["p50_ms"], "㉒ 재정렬을 켜면 중앙값 시간이 2배 넘게")
changed = [x["question_id"] for x in ns["by_setting"]["B 다시 쓰기만"] if x["search_query"] != rows_of[x["question_id"]]["question"]]
S.claim(changed == ["Q03", "Q04"] and len(ns["SETTINGS_EVAL"]) == 5, "㉒ 다시 쓰기가 바꾼 건 Q03, Q04뿐")
S.claim(all(sm[s]["candidate_recall"] == sm["A 기준"]["candidate_recall"] for s in order[:4]) and sm["E D + 넓히기"]["candidate_recall"] > sm["D 다시 쓰기 + 재정렬"]["candidate_recall"], "㉒ 후보 Recall은 넓히기에서만 오른다")
outs["t22"] = S.table("㉒ 다섯 설정, 같은 질문 15개 (정답 없는 질문은 검색 점수에서 빼요)", ["설정", "후보 Recall", "넣은 3개 Recall", "MRR@10", "nDCG@10", "상태 맞음", "충실성"],
                      [[f"<b>{s}</b>" if s == best else s, f3(v["candidate_recall"]), f3(v["context_recall@3"]), f3(v["mrr@10"]), f3(v["ndcg@10"]), f"{v['status_ok']:.0%}", f3(v["faithfulness"])] for s, v in sm.items()],
                      [f"후보 Recall은 A~D가 모두 {sm['A 기준']['candidate_recall']:.3f}로 같고, 넓히기를 더한 E에서만 {sm['E D + 넓히기']['candidate_recall']:.3f}가 돼요. 다시 쓰기와 재정렬은 후보에 없는 정답을 데려오지 못하고, 넓히기가 Q05(상호대차 → '다른 도서관 책 받아 보기')를 후보에 넣었어요.",
                       "다시 쓰기는 기준 질문을 그대로 두고 후속 질문(Q03, Q04)만 바꿔서, 평균보다 질문별로 봐야 효과가 보여요(아래 표)."])
outs["t22c"] = S.table("질문 15개를 처리한 시간 (질문 하나당, GPU 동기화 후)", ["설정", "중앙값", "95% 지점"], [[s, f"{v['p50_ms']:.0f} ms", f"{v['p95_ms']:.0f} ms"] for s, v in sm.items()],
                       [f"재정렬을 켜면 중앙값이 {sm['A 기준']['p50_ms']:.0f}ms에서 {sm['C 재정렬만']['p50_ms']:.0f}ms쯤으로 늘어요(GPU에서 후보 10개를 한 번 더 읽어요). 점수가 오른 만큼 시간을 낼 만한지는 서비스 목표로 정해요. 단계별 시간은 <a href=\"rag-rerank.html\">RAG 검색 고치기</a> ⑮에 있어요.",
                        "95% 지점은 '스무 번 중 한 번은 이보다 느리다'는 뜻이에요. 평균이나 중앙값만 보면 가끔 느린 요청이 가려져요. 시간은 PC와 그때의 상태에 따라 달라서, 비교는 같은 PC에서 같은 때에 해요."])
outs["t22b"] = S.table("A 기준과 같은 질문끼리 짝지어 본 변화", ["설정", "Context Recall@3이 오른 질문", "MRR@10이 오른 질문", "나빠진 질문"],
                       [[s, ", ".join(d["context_recall@3"]["better"]) or "없음", ", ".join(d["mrr@10"]["better"]) or "없음",
                         ", ".join(sorted(set(d["context_recall@3"]["worse"] + d["mrr@10"]["worse"]))) or "없음"] for s, d in dl.items()],
                       ["같은 질문, 같은 정답, 같은 조각, 같은 k에서 한 번에 하나만 바꿨어요. B와 C를 따로 재 둬야 D에서 오른 것이 무엇 덕분인지 말할 수 있어요.",
                        "이건 고정된 질문으로 두 판을 짝지어 보는 오프라인 비교예요. 실제 사용자를 나눠 보내는 온라인 A/B 시험과는 달라요."])

# ---------------- ㉓
fl = ns["failures"]
TYPES = ["후보 밖", "넣을 3개에 못 듦", "답 상태 틀림", "근거 없는 주장", "통과"]
FIX = {"후보 밖": "검색 범위: 질문 다시 쓰기, 용어 넓히기, 분류, 조각 나누기", "넣을 3개에 못 듦": "순서: 재정렬, 넣을 수(context_k), 글자 예산",
       "답 상태 틀림": "정보 부족 규칙, Prompt (앞 단계가 원인이면 그쪽부터)", "근거 없는 주장": "생성: Prompt, 숫자와 조건 검사, 더 나은 모델", "통과": ""}
S.claim({q for q, t in fl["E D + 넓히기"].items() if t != "통과"} == {"Q04", "Q12"} and all(t == "근거 없는 주장" for q, t in fl["E D + 넓히기"].items() if t != "통과"), "㉓ E에서 남은 건 근거 없는 주장 둘")
outs["t23"] = S.table("㉓ 실패를 먼저 막힌 단계로 나누면 (질문 수)", ["설정"] + TYPES, [[s] + [", ".join(q for q, t in v.items() if t == ty) or "0" if ty != "통과" else str(sum(t == ty for t in v.values())) for ty in TYPES] for s, v in fl.items()],
                      ["앞 단계부터 차례로 봐요. 후보에 없으면 뒤에서 무엇을 해도 소용없고, 후보에 있어도 넣지 못하면 모델은 볼 수 없으니까요. 먼저 막힌 단계가 고칠 곳이에요.",
                       "A에서 다섯이던 실패가 E에서는 Q04, Q12 둘만 남아요. 둘 다 검색은 완벽한데 답에 근거 없는 주장이 섞인 경우예요. 검색을 더 고쳐도 이 둘은 안 고쳐져요."])
outs["t23b"] = S.table("실패 유형에서 고칠 곳으로", ["유형", "고칠 곳"], [[t, FIX[t]] for t in TYPES[:-1]] + [["시간이 늘어남", "후보 수, 질문 수, batch, 캐시, GPU와 DB 시간"], ["오류가 늘어남", "Timeout, 인증, DB 연결, GPU 메모리 (정보 부족과 다른 상태로 기록)"]])

# ---------------- ㉔
gd = ns["gate"]
okD, chD, unD = gd["D 다시 쓰기 + 재정렬"]
okE, chE, unE = gd["E D + 넓히기"]
S.claim(fl["D 다시 쓰기 + 재정렬"]["Q05"] == "후보 밖", "㉔ D의 Q05는 후보 밖")
S.claim(not okD and not chD["status_ok"][2] and not okE and [k for k, (_, _, ok) in chE.items() if not ok] == ["근거 없는 주장이 있는 답"] and chE["faithfulness"][2], "㉔ D는 상태와 근거에서, E는 근거 없는 주장 하나로만 막힘, 평균 충실성은 통과")
LABEL = {"context_recall@3": "Context Recall@3 평균", "mrr@10": "MRR@10 평균", "status_ok": "기대 상태와 같은 비율", "citation_ok": "출처가 넣은 조각 안인 비율", "faithfulness": "충실성 평균"}


def gate_rows(ch):
    out = []
    for k, (val, need, ok) in ch.items():
        v = f"{val}" if isinstance(val, int) else f3(val)
        n = f"{need}개" if isinstance(need, int) and k not in LABEL else f"{need} 이상"
        out.append([LABEL.get(k, k), v, n if k in LABEL else f"{need}개", "통과" if ok else "<b>막힘</b>"])
    return out


outs["t24"] = S.table("㉔ 예시 게이트로 E(D + 넓히기)를 재면", ["확인할 것", "값", "기준", "결과"], gate_rows(chE),
                      [f"<b>막혔어요.</b> 충실성 평균({chE['faithfulness'][0]:.3f})은 기준 {chE['faithfulness'][1]}을 넘지만, 근거 없는 주장이 든 답이 {len(unE)}개({', '.join(unE)}) 있어요. 평균만 보면 통과했을 거예요. 틀리면 곤란한 항목은 평균이 아니라 '0개'로 막아요.",
                       f"D(넓히기 없음)는 여기에 더해 기대 상태와 같은 비율이 {chD['status_ok'][0]:.0%}라 막혀요(Q05가 검색 실패로 정보 부족).",
                       "기준 숫자는 이 노트의 예시예요. 처음에는 절대 기준보다 '기준 판보다 나빠지지 않았나'를 먼저 보고, 질문이 쌓이면 팀이 기준을 정해요."])

flow = [[17, "정답 묶음", "질문과 정답 조각", f"{len(QS)}문항, 답 없음 {len(no_answer)}"], [18, "찾았나", "Hit, Recall, Precision", "후보 vs 넣은 3개"], [19, "앞에 뒀나", "MRR, nDCG", "재정렬의 효과"],
        [20, "답은 맞나", "충실성, 정답, 정보 부족", "사람 vs 규칙"], [21, "Runner", "실행 뒤 정답 붙이기", f"JSONL {len(lines)}줄"], [22, "비교", "하나씩 바꾸기", "질문별 변화와 시간"],
        [23, "고칠 곳", "먼저 막힌 단계", f"E: 남은 실패 {sum(t != '통과' for t in fl['E D + 넓히기'].values())}"], [24, "게이트", "Agent에 붙이기 전", "막힘: 근거 없는 주장"]]
S.save(ROOT / "topics" / "backend" / "rag-eval-data.js", {"code": code, "names": S.names(code), "range": [17, 24], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p><a href="rag-rerank.html">RAG 검색 고치기</a>에서 질문 다시 쓰기, 넓히기, 재정렬을 붙이고 질문 몇 개로 정답이 몇 등인지 봤어요. 그런데 \"좋아졌다\"고 말하려면 더 필요해요. 정해진 질문 묶음에서, 정해진 정답과 비교해, 숫자 하나로 모은 점수가 있어야 해요. 그래야 다음 변경이 무엇을 좋게 하고 무엇을 나쁘게 했는지 알 수 있어요.</p>
<p>RAG는 두 단계라서 따로 재요.</p>
{K.table(["단계", "묻는 것", "재는 것"], [["검색", "정답 조각을 찾았나, 앞에 뒀나", "Hit@k, Recall@k, Precision@k (⑱), MRR, nDCG (⑲)"], ["생성", "답이 넣은 근거와 맞나, 정답과 맞나, 답할 수 없을 때 멈췄나", "충실성, 정답성, 정보 부족 판단 (⑳)"]])}
<p>둘을 섞어 재면 원인을 못 찾아요. 검색이 정답을 못 가져왔는데 모델만 바꾸거나, 검색은 완벽한데 답이 지어낸 말을 하는데도 검색만 고치게 돼요. 이 쪽은 RAG 서비스와 검색 고치기에서 만든 흐름을 <b>그대로</b> 15개 질문으로 다섯 가지 설정에서 돌리고, 실행이 끝난 뒤에 정답을 붙여 점수를 매겨요. 검색과 재정렬은 실제 모델(e5, bge-reranker, GPU)이고, 답 문장은 가짜 서버가 돌려준 미리 정한 문장이에요(일부러 틀린 답 둘을 섞었어요).</p>
<p>LLM에게 채점을 맡기는 방법 자체는 <a href="llm-judge.html">LLM으로 답 채점하기</a>, 단계별 시간 재기는 <a href="rag-rerank.html">RAG 검색 고치기</a> ⑮에 있어서, 여기서는 그 결과를 평가 흐름에 어떻게 넣는지만 봐요.</p>
"""
whole = """<p>정답 묶음을 만들고(⑰), 검색 점수(⑱⑲)와 답 점수(⑳)를 정의한 뒤, 같은 Runner로 다섯 설정을 돌려(㉑) 비교하고(㉒), 실패를 고칠 곳으로 나누고(㉓), 다음 단계(Agent)에 붙여도 되는지 판단해요(㉔). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(17, "정답 묶음과 실행 기록", "결과를 보기 전에 질문마다 정답 조각과 기대 상태를 정해 두고, 한 번 실행할 때마다 단계별 결과를 같은 모양으로 기록해요.",
           "<p>평가의 기준은 <b>정답 묶음</b>(Gold Dataset)이에요. 질문마다 \"이 조각이면 답할 수 있다\"를 사람이 정해 둔 거예요. 검색 결과를 보고 정답을 고치면 점수가 의미를 잃어서, 결과를 보기 전에 정하고 판(dataset_version)을 붙여요.</p>"
           + K.table(["칸", "뜻"], [["question_id", "여러 실행에서 같은 질문을 잇는 ID"], ["question, history, category", "서비스에 넣을 값 (앞선 질문이 있으면 후속 질문)"], ["gold", "정답 조각과 등급. 2 = 바로 답함, 1 = 일부만 관련. 없으면 빈 묶음"],
                                  ["expected_status", "answered 또는 insufficient_context (답이 자료에 없는 질문)"], ["reference_facts", "정답 답에 꼭 들어갈 핵심 사실 (⑳의 정답성)"]])
           + "<p><b>관련 있다/없다(0, 1)</b>만 적으면 Recall과 MRR을 잴 수 있고, <b>등급(0, 1, 2)</b>까지 적으면 \"더 중요한 조각을 앞에 뒀나\"(nDCG)도 잴 수 있어요. 등급은 모델 점수가 아니라 사람이 정한 정답이에요. 실제로는 두 사람이 따로 판정하고, 갈린 것은 기준을 다시 맞춰 정해요.</p>"
           + K.out("s17")
           + "<p>실행 쪽은 <code>run_target</code> 하나예요. 서비스에 넣을 값만 받아서(<b>정답은 넣지 않아요</b>) RAG 검색 고치기의 검색을 그대로 부르고, 단계마다 나온 것을 기록해요. 점수는 이 기록을 받아 나중에 매겨요.</p>"
           + K.out("s17b")
           + K.code_table([("assert ...", "정답 묶음의 약속 검사: ID 중복, 없는 조각, 등급 범위, 정답이 없으면 정보 부족"), ("request_of(row, setting, flags)", "정답을 뺀 서비스 입력"),
                           ("run_target(req)", "한 번 실행하고 단계별 결과와 시간을 기록 (RunRecord)"), ("torch.cuda.synchronize()", "GPU 작업이 끝날 때까지 기다린 뒤 시간 재기")]), "s17"),
    K.step(18, "찾았나: Hit@k, Recall@k, Precision@k", "검색 결과 앞 k개가 정답 조각을 얼마나 담았는지 세요. 셋 다 같은 교집합을 쓰지만 분모가 달라요.",
           K.table(["지표", "식", "묻는 것"], [["Hit@k", "앞 k개에 정답이 하나라도 있으면 1, 없으면 0", "하나라도 걸렸나"], ["Recall@k", "|앞 k개 ∩ 정답| ÷ |정답|", "정답을 놓치지 않았나"],
                                                ["Precision@k", "|앞 k개 ∩ 정답| ÷ k", "가져온 것 중 정답이 얼마나 되나"]])
           + "<p>정답 조각이 없는 질문(답이 자료에 없음)은 분모가 0이라 검색 점수를 내지 않고 N/A로 빼요. 그 질문은 ⑳에서 \"정보 부족이라고 답했나\"로 따로 재요.</p>"
           + K.out("t18a")
           + "<p>같은 공식이라도 <b>어느 목록</b>에 적용하느냐가 중요해요. 검색이 돌려준 후보 전체(candidate_ids)로 재면 \"검색이 찾았나\"이고, 실제로 모델에게 넣은 조각(included_ids)으로 재면 \"모델이 볼 수 있었나\"예요.</p>"
           + K.out("t18b")
           + K.code_table([("set(ids[:k]) & set(gold)", "앞 k개와 정답의 교집합 (같은 조각은 한 번만)"), ("hit_at_k, recall_at_k, precision_at_k", "정답이 없으면 None(N/A)"), ("mean(values)", "N/A는 빼고 평균")]), "s18"),
    K.step(19, "앞에 뒀나: MRR과 nDCG", "정답을 찾았는지뿐 아니라 몇 번째에 뒀는지를 재요. 앞 몇 개만 모델에게 넣으니까 순서가 곧 품질이에요.",
           K.table(["지표", "식", "보는 것"], [["Reciprocal Rank", "첫 정답이 r번째면 1/r, k 안에 없으면 0", "첫 정답의 위치"], ["MRR", "질문들의 1/r 평균", "보통 몇 번째에 첫 정답이 오나"],
                                               ["DCG@k", "Σ (2^등급 − 1) ÷ log₂(순위 + 1)", "등급 높은 조각이 앞에 있을수록 커요"], ["nDCG@k", "DCG@k ÷ (가장 좋은 순서의 DCG@k)", "0~1로 맞춘 DCG"]])
           + "<p>log로 나누는 건, 뒤로 갈수록 사람도 모델도 덜 본다고 치는 할인이에요. 1위는 그대로, 3위는 절반쯤(log₂4 = 2로 나눔)만 쳐요. MRR은 첫 정답 하나만 보고, nDCG는 정답이 여럿이고 등급이 다를 때 전체 순서를 봐요.</p>"
           + K.out("t19")
           + K.code_table([("next((1 / rank for rank, i in enumerate(ids[:k], start=1) if i in gold), 0.0)", "첫 정답의 1/순위 (순위는 1부터)"), ("math.log2(rank + 1)", "순위 할인"),
                           ("sorted(grades, key=grades.get, reverse=True)", "등급이 높은 순서 = 가장 좋은 순서 (분모)")]), "s19"),
    K.step(20, "답은 맞나: 근거, 정답, 정보 부족", "답을 주장 단위로 나눠 넣은 근거로 확인되는지 보고, 정답의 핵심 사실이 들었는지, 답할 수 없을 때 멈췄는지 따로 재요.",
           K.table(["지표", "무엇과 무엇을 비교", "묻는 것"], [["충실성 (faithfulness)", "답의 주장 ↔ 넣은 근거", "근거에서 이 주장을 확인할 수 있나"], ["관련성 (relevance)", "질문 ↔ 답", "물은 것에 답했나"],
                                                       ["정답성 (correctness)", "답 ↔ 정답의 핵심 사실", "실제 정답과 숫자, 조건이 맞나"], ["출처 검사", "출처 번호 ↔ 넣은 조각", "넣은 조각만 출처로 댔나 (<a href=\"rag-service.html\">RAG 서비스</a> ⑥)"],
                                                       ["정보 부족 판단", "상태 ↔ 기대 상태", "답이 없는 질문에서 멈췄나"]])
           + "<p>충실성은 <b>주장 단위</b>로 재요. \"4시간까지 쓸 수 있고, 회원증 없이도 빌릴 수 있다\"에는 맞는 주장과 지어낸 주장이 섞여 있어서, 문장 하나로 맞다/틀리다를 매기면 틀린 절반이 묻혀요. 여기서는 사람이 주장마다 판정한 표(answer_reviews.json)로 재고, 빠른 자동 규칙(주장 속 숫자가 근거에 있나)을 옆에 놓고 비교해요.</p>"
           + K.out("t20") + K.out("t20b")
           + "<p>사람 평가표는 보통 질문 적합성, 근거 일치, 정답 사실, 출처, 안전한 보류(근거 없으면 멈춤)를 1~3점으로 매겨요. 숫자로 모으기 전에 기준표를 팀이 같이 정해요.</p>"
           + K.code_table([("faithfulness(record)", "확인된 주장 ÷ 전체 주장. 답이 없거나 답 문장이 바뀌면 N/A"), ("re.findall(r\"\\d+\", claim)", "주장 속 숫자"), ("fact_recall(record, facts)", "핵심 사실이 답에 글자로 들어 있는 비율")]), "s20"),
    K.step(21, "평가 Runner: 실행하고, 끝난 뒤에 정답을 붙여 점수 매기기", "실행(target)과 채점(evaluator)을 나눠 Runnable로 만들고, 질문 × 설정마다 한 줄씩 기록해요.",
           "<p>평가 Runner는 둘로 나눠요. <b>target</b>은 서비스를 실행해 기록만 남기고, <b>evaluator</b>는 그 기록과 정답을 받아 점수만 내요. 정답이 서비스 입력에 섞여 들어가면 시험지에 답을 적어 주는 셈이라, 실행이 끝난 뒤에만 붙여요. 둘 다 <code>RunnableLambda</code>라 다른 단계처럼 <code>invoke</code>로 불러요.</p>"
           + K.table(["", "실행 기록 (trace)", "평가 결과 (report)"], [["묻는 것", "어떻게 실행됐나", "잘 실행됐나"], ["담는 것", "단계별 입력, 출력, 시간, 오류", "정답, 기준과 비교한 점수"], ["주의", "키, 개인정보, 문서 원문을 함부로 남기지 않기", "판(dataset, corpus, 모델, Prompt)을 함께 적기"]])
           + K.out("s21")
           + "<p>LangSmith 같은 서비스는 실행 기록과 질문 묶음, 실험 비교를 화면으로 관리해 줘요. 필수는 아니고, 이 쪽처럼 JSONL 파일만으로도 같은 평가를 할 수 있어요. 외부 서비스로 기록을 보낼 때는 문서와 개인정보가 함께 나가지 않는지 먼저 확인해요.</p>"
           + K.code_table([("RunnableLambda(run_target), RunnableLambda(evaluate)", "실행과 채점을 Runnable로"), ("evaluator.invoke({\"record\": ..., \"gold\": ...})", "실행 뒤에 정답을 붙여 점수"),
                           ("json.dumps(line, ensure_ascii=False) + \"\\n\"", "한 줄에 한 기록 (JSONL)")]), "s21"),
    K.step(22, "하나씩 바꿔 비교하기", "같은 질문, 같은 정답, 같은 조각에서 한 번에 하나만 바꾼 설정들을 평균과 질문별 변화로 비교해요.",
           "<p>다시 쓰기와 재정렬을 한꺼번에 켜면 무엇 덕분에 좋아졌는지 말할 수 없어요. 그래서 A(기준)에서 하나씩 켠 B, C를 먼저 재고, 그다음 D, E를 재요. 질문, 정답, 조각, 분류, 후보 수와 넣을 수, 모델 판은 모두 같게 둬요.</p>"
           + K.out("t22") + K.out("t22b") + K.out("t22c")
           + K.code_table([("per_question_delta(a, b, metric)", "같은 질문끼리 짝지어 오른 것, 내린 것"), ("np.percentile(times, 95)", "시간의 95% 지점 (가끔 느린 요청)")]), "s22"),
    K.step(23, "낮은 점수에서 고칠 곳으로", "실패한 질문마다 가장 먼저 막힌 단계를 찾아, 그 단계를 고쳐요.",
           "<p>점수가 낮다고 모델부터 바꾸면 헛수고가 되기 쉬워요. 실패를 앞 단계부터 확인해 \"어디서 처음 막혔나\"로 나누면 고칠 곳이 보여요.</p>"
           + K.out("t23") + K.out("t23b")
           + K.code_table([("failure_type(x, row)", "후보 밖 → 넣지 못함 → 상태 틀림 → 근거 없는 주장 순서로 확인")]), "s23"),
    K.step(24, "품질 게이트: Agent에 붙이기 전에", "새 판을 다음 단계(Agent의 도구)로 내보내기 전에 넘어야 할 최소 조건을 코드로 확인해요.",
           "<p>다음 단계에서 Agent는 이 RAG를 <b>도구</b>로 불러요. Agent가 도구를 잘 골라도 도구가 틀린 자료를 돌려주면 소용없어요. 그래서 도구로 내보내기 전에 검색, 생성, 운영 조건을 확인하는 <b>게이트</b>를 둬요. 막히면 연결을 미루고, 막힌 질문과 판을 기록해요.</p>"
           + K.out("t24")
           + "<p>게이트에 넣을 만한 것: 검색(Recall, MRR이 기준 판보다 나빠지지 않음), 생성(근거 없는 주장 0, 정보 부족 판단), 출처(넣은 조각만), 권한(허락된 문서만 검색, 위반 0), 운영(시간, 오류 상태가 정보 부족과 구분되는지).</p>"
           + K.code_table([("GATE = {...}", "확인할 것과 기준 (예시)"), ("quality_gate(setting)", "(통과 여부, 항목별 값과 기준, 근거 없는 답 목록)")]), "s24"),
]
practice = [("정답 묶음은 작게 시작", "처음엔 10~20개면 충분해요. 실제로 틀린 질문을 모아 늘려 가요. 정답 묶음을 보고 Prompt나 재정렬을 고치기를 반복하면 그 묶음에만 맞춰져서, 가끔은 처음 보는 질문으로 따로 재요."),
            ("정답은 결과 전에, 판과 함께", "검색 결과를 보고 정답을 바꾸면 점수가 의미를 잃어요. dataset_version, corpus_version, 판정한 사람과 날짜를 적고, 바꾸면 이력을 남겨요."),
            ("어느 목록으로 쟀나", "후보 Recall과 Context Recall은 다른 질문에 답해요. 표에 어느 목록인지(k와 함께) 꼭 적어요."),
            ("평균만 보지 않기", "질문별로 오른 것과 내린 것을 같이 봐요. 틀리면 곤란한 항목(근거 없는 주장, 권한 위반)은 평균 대신 0개로 막아요."),
            ("자동 채점은 보조", "숫자 규칙, 글자 대조, <a href=\"llm-judge.html\">LLM 채점</a>은 빠르지만 각자 놓치는 게 있어요(⑳). 사람 판정과 갈리는 답을 표본으로 봐요."),
            ("시간도 같이 재기", "점수가 올라도 시간이 크게 늘면 쓸 수 없을 수 있어요. GPU를 쓰면 동기화한 뒤 재고, 중앙값과 95% 지점을 함께 봐요.")]
libs = [("⑰", "<code>json.loads</code>, <code>assert</code>, <code>torch.cuda.synchronize</code>", "정답 묶음 읽고 검사, 실행 기록"), ("⑱⑲", "<code>set</code>의 <code>&amp;</code>, <code>math.log2</code>", "검색 지표 (직접 계산)"),
        ("⑳", "<code>re.findall</code>", "주장 속 숫자 규칙"), ("㉑", "<code>RunnableLambda</code>, <code>invoke</code>, <code>json.dumps</code>", "Runner와 JSONL"), ("㉒", "<code>np.percentile</code>", "시간 분포")]
K.build(ROOT / "topics" / "backend" / "rag-eval.html", title="RAG 평가", h1="RAG 평가: 정답 묶음으로 검색과 답을 따로 재기", track="백엔드", crumb="백엔드 &gt; 검색과 RAG",
        one_line="RAG가 좋아졌는지는 <b>결과를 보기 전에 정해 둔 정답 묶음</b>으로 재요. 검색은 정답 조각을 찾았는지(<b>Recall@k</b>)와 앞에 뒀는지(<b>MRR, nDCG</b>)로, 답은 넣은 근거와 맞는지(<b>충실성</b>)와 답할 수 없을 때 멈췄는지로 <b>따로</b> 재요. 한 번에 하나씩 바꿔 질문별로 비교하고, 먼저 막힌 단계를 고치고, 최소 조건(<b>게이트</b>)을 넘어야 다음 단계에 붙여요.",
        why=why, whole=whole, steps=steps, code_file="examples/rag/rag_service.py", practice=practice, libs=libs,
        libs_note="<code>request_of</code>, <code>run_target</code>, <code>hit_at_k</code>, <code>recall_at_k</code>, <code>precision_at_k</code>, <code>mean</code>, <code>reciprocal_rank</code>, <code>dcg</code>, <code>ndcg_at_k</code>, <code>faithfulness</code>, <code>number_check</code>, <code>fact_recall</code>, <code>evaluate</code>, <code>per_question_delta</code>, <code>failure_type</code>, <code>quality_gate</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rag-rerank.html\">RAG 검색 고치기</a>", "여기서 재는 다시 쓰기, 넓히기, 재정렬"), ("<a href=\"rag-service.html\">RAG 서비스</a>", "출처 검사와 정보 부족 응답"),
                   ("<a href=\"llm-judge.html\">LLM으로 답 채점하기</a>", "충실성을 LLM에게 채점시킬 때"), ("<a href=\"../model/ml-metrics-clustering.html\">평가지표와 군집</a>", "분류의 정밀도와 재현율 (이름은 같고 대상이 달라요)")],
        sources=["다루는 범위: Day28 (2026-11-25) RAG Evaluation과 Debugging. 검색 실패와 생성 실패 나누기, Gold Dataset(binary와 graded relevance, 답 없는 질문), Hit@k, Recall@k, Precision@k, 후보와 Context Recall, MRR, nDCG, Faithfulness, Relevance, Correctness, LLM-as-a-Judge와 사람 평가표, LangChain 평가 Runner, Trace와 Report, LangSmith, 하나씩 바꾸는 비교, 실패 유형, Latency, JSONL 기록, Quality Gate. 수업의 출석, 회의실 예시 대신 이 노트의 가람 도서관 조각으로 같은 흐름을 다시 만들었어요",
                 "예제: <code>examples/rag/rag_service.py</code>의 ⑰~㉔. 정답 묶음 <code>examples/rag/data/gold.json</code>, 답 검토표 <code>answer_reviews.json</code> (이 노트용으로 만든 예)",
                 f"결과 값: <code>tools/pages/rag_eval.py</code>가 단계별로 실제 실행한 결과 ({ENV}). 답 문장은 가짜 서버의 미리 정한 값이고, 시간은 PC와 그때의 상태에 따라 달라져요",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/model_evaluation.html#ndcg-score\" target=\"_blank\">scikit-learn nDCG</a>, <a href=\"https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/\" target=\"_blank\">Ragas Faithfulness</a>, <a href=\"https://docs.langchain.com/langsmith/evaluate-rag-tutorial\" target=\"_blank\">LangSmith: Evaluate a RAG application</a>"])
print("ok")
