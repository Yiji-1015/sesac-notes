"""백엔드 > 검색과 RAG > RAG 평가. examples/rag_eval/evaluate.py 하나가 ①~⑦로 자란다 (다른 쪽 코드에 기대지 않음).
정답 묶음(examples/rag/data/gold.json)과 답 검토표(answer_reviews.json)로, e5 검색과 bge-reranker 재정렬(GPU)을 실제로 돌려 잰다.
답 문장은 LLM을 부르지 않고 검토표의 문장을 그대로 쓴다(일부러 틀린 답 둘).
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/rag_eval.py
"""
import html
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
EX = ROOT / "examples" / "rag_eval"
code = S.parse((EX / "evaluate.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, EX.resolve(), filename="evaluate.py")
torch = ns["torch"]
V = {p: version(p) for p in ("torch", "sentence-transformers", "numpy")}
GPU = torch.cuda.get_device_name(0)
ENV = f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items()) + f", {GPU}, intfloat/multilingual-e5-small, BAAI/bge-reranker-v2-m3"
E = html.escape
outs = {}

GOLD, T = ns["GOLD"], ns["TITLE"]
Q = GOLD["questions"]
ROW = {r["question_id"]: r for r in Q}
gold_of, SET = ns["gold_of"], list(ns["SETTINGS"])
A_, B_, C_ = SET


def f2(x):
    return "N/A" if x is None else f"{x:.2f}"


def tl(i):
    return E(T[i])


# ---------------- ①
n_ans = sum(r["expected_status"] == "answered" for r in Q)
n_ins = len(Q) - n_ans
S.claim(len(Q) == 15 and n_ins == 2, "① 15문항, 정보 부족 2")
graded = sum(1 for r in Q if 1 in r["gold"].values())
outs["s1"] = S.table(f"① 정답 묶음 (판 {E(GOLD['dataset_version'])}, 질문 {len(Q)}개 중 일부)", ["질문 ID", "질문", "정답 조각 (등급)", "기대 상태", "핵심 사실"],
                     [[r["question_id"], E(r["question"]) + (f"<br><small>앞선 질문: {E(r['history'][0])}</small>" if r["history"] else ""),
                       "<br>".join(f"{tl(i)} ({g})" for i, g in r["gold"].items()) or "없음", r["expected_status"], E(", ".join(r["reference_facts"])) or "-"]
                      for r in Q if r["question_id"] in ("Q01", "Q03", "Q06", "Q14")],
                     [f"정답 조각이 둘인 질문이 {graded}개 있어요. 바로 답하는 조각은 등급 2, 일부만 관련된 조각은 1이에요.",
                      f"Q14처럼 정답 조각이 없는 질문이 {n_ins}개예요. 이런 질문은 '자료로는 답할 수 없다'고 멈추는 게 맞는 답이에요.",
                      "핵심 사실은 맞는 답에 꼭 들어가야 할 말이에요(⑤의 정답성). 위의 검사(assert)는 이 묶음이 약속을 지키는지 실행할 때마다 확인해요.",
                      f"묶음에는 묶음의 판({E(GOLD['dataset_version'])})과 조각의 판({E(GOLD['corpus_version'])})을 같이 적어 뒀어요. 조각을 다시 나누면 같은 조각 ID가 다른 내용을 가리키거나 사라질 수 있어서, 판이 다른 점수끼리는 나란히 비교하지 않아요."])

# ---------------- ②
recs, bs = ns["records"], ns["by_setting"]
S.claim(len(recs) == len(Q) * 3, "② 질문 × 설정 3")
r3 = next(r for r in bs[C_] if r["question_id"] == "Q03")
outs["s2"] = S.box("② run(Q03, \"C 다시 쓰기 + 재정렬\")가 남긴 기록", [
    (f"question_id \"{r3['question_id']}\", setting \"{r3['setting']}\"", 1), (f"query \"{E(r3['query'])}\"", 2),
    (f"ranked → {len(r3['ranked'])}개, 앞 3개: {', '.join(tl(i) for i in r3['ranked'][:3])}", 3), (f"context → {', '.join(tl(i) for i in r3['context'])}", 4),
    (f"latency_ms {r3['latency_ms']}", 5)],
    ["어느 질문을 어느 설정으로 돌렸는지예요. 설정 셋 × 질문 15개 = 45줄이 쌓여요.",
     "실제로 검색한 질문이에요. C 설정은 다시 쓴 질문을 써요.",
     "검색(과 재정렬)이 돌려준 순서예요. ③④의 검색 점수는 이 목록으로 매겨요.",
     "그중 모델에게 줄 위 3개예요.",
     "이 실행에 걸린 시간이에요."])

# ---------------- ③
kc = ns["k_curve"]
S.claim(kc[10][0] == kc[10][1] and kc[1][2] == kc[1][0] and kc[10][2] < kc[1][2], "③ k가 커지면 Precision 내림")
outs["t3"] = S.table(f"③ {E(A_)}: k를 바꾸면 (정답이 있는 질문 {n_ans}개의 평균)", ["k", "Hit@k", "Recall@k", "Precision@k"],
                     [[k, f2(v[0]), f2(v[1]), f2(v[2])] for k, v in kc.items()],
                     [f"k를 늘리면 Hit와 Recall은 오르고(놓치는 게 줄어요) Precision은 내려요(정답 아닌 것도 같이 와요). k=10에서 Recall {f2(kc[10][1])}, Precision {f2(kc[10][2])}예요.",
                      "k=1에서 Hit와 Precision이 같은 건, 하나만 볼 때는 '하나라도 맞나'와 '맞은 비율'이 같은 뜻이라서예요.",
                      f"정답 조각이 없는 질문 {n_ins}개는 분모가 없어서 N/A로 빼고 평균했어요. 그 질문들은 ⑤에서 '정보 부족이라고 답했나'로 따로 재요."])
gap = [r["question_id"] for r in bs[A_] if ns["recall_at_k"](r["ranked"], gold_of[r["question_id"]], 10) == 1 and (ns["recall_at_k"](r["context"], gold_of[r["question_id"]], 3) or 0) < 1]
S.claim(set(gap) >= {"Q03", "Q06"}, "③ 후보엔 다 있는데 3개엔 못 든 질문")
outs["t3b"] = S.table(f"③ {E(A_)}: 같은 공식, 다른 목록", ["질문 ID", "질문", "Recall@10 (검색 후보)", "Recall@3 (모델에게 줄 것)"],
                      [[q, E(ROW[q]["question"]), f2(ns["recall_at_k"](r["ranked"], gold_of[q], 10)), f2(ns["recall_at_k"](r["context"], gold_of[q], 3))]
                       for q in gap for r in bs[A_] if r["question_id"] == q],
                      [f"{', '.join(gap)}은 검색이 정답을 모두 후보에 담았는데(Recall@10 = 1), 모델에게 줄 3개에는 다 못 들어갔어요. 앞 목록으로 재면 '검색이 찾았나', 뒤 목록으로 재면 '모델이 볼 수 있었나'예요. 표에는 어느 목록인지 k와 함께 꼭 적어요."])

# ---------------- ④
osc = ns["order_scores"]
rows4 = []
for q in gold_of:
    if not gold_of[q]:
        continue
    a, b = osc[A_, q], osc[B_, q]
    if a != b:
        rows4.append([q, E(ROW[q]["question"]), f2(a[0]), f2(b[0]), f2(a[1]), f2(b[1])])
up_mrr = [q for q in gold_of if gold_of[q] and osc[B_, q][0] > osc[A_, q][0] + 1e-9]
down_ndcg = [q for q in gold_of if gold_of[q] and osc[B_, q][1] < osc[A_, q][1] - 1e-9]
S.claim({"Q03", "Q06"} <= set(up_mrr) and down_ndcg, "④ 재정렬로 MRR 오른 질문, nDCG 내린 질문")
outs["t4"] = S.table(f"④ {E(A_)} → {E(B_)}: 값이 바뀐 질문만 (k=10)", ["질문 ID", "질문", "1/r (A)", "1/r (B)", "nDCG (A)", "nDCG (B)"], rows4,
                     [f"재정렬로 첫 정답이 앞으로 온 질문은 {', '.join(up_mrr)}예요. Q06은 첫 정답이 5위(1/5 = 0.20)에서 1위(1.00)가 됐어요.",
                      f"{', '.join(down_ndcg)}은 첫 정답은 그대로 1위인데 nDCG가 조금 내려갔어요. 등급 1인 두 번째 정답의 자리가 밀렸거든요. MRR은 첫 정답만 보고, nDCG는 전체 순서와 등급을 봐서 이런 차이가 보여요.",
                      "평균만 보면 이런 질문이 묻혀요. 질문별로 오르고 내린 것을 같이 봐요(⑥)."])

# ---------------- ⑤
ans = ns["answers"]
S.claim(ans["Q04"]["human"] == [True, False] and ans["Q04"]["rule"] == [True, True], "⑤ Q04: 사람은 잡고 규칙은 놓침")
S.claim(ans["Q12"]["human"] == ans["Q12"]["rule"] == [False, True], "⑤ Q12: 규칙도 잡음")
S.claim(ans["Q05"]["rule"] == [True] and gold_of["Q05"] and not set(ns["C"]["Q05"]["context"]) & set(gold_of["Q05"]), "⑤ Q05: 정답 조각이 없는데 숫자 규칙 통과")
S.claim(ans["Q09"]["facts"] == 0.0 and ans["Q09"]["faithfulness"] == 1.0, "⑤ Q09: 맞는 답인데 핵심 사실 글자 대조 0")
S.claim(all(a["status_ok"] for a in ans.values()), "⑤ 상태는 모두 맞음")
outs["t5"] = S.table("⑤ 답 점수 (답 문장은 검토표의 것, 근거는 C 설정에서 모델에게 줄 3개)", ["질문 ID", "답", "상태 맞나", "충실성", "핵심 사실"],
                     [[q, E(a["answer"]), "맞음" if a["status_ok"] else "<b>틀림</b>", f2(a["faithfulness"]), f2(a["facts"])] for q, a in ans.items() if q in ("Q04", "Q09", "Q12", "Q14")],
                     ["Q04의 '회원증 없이도'는 근거에 없는 말이에요. 안내문은 '회원증을 맡기고 빌린다'고 해요. 주장 둘 중 하나라 충실성이 0.50이에요. 문장 하나로 맞다/틀리다를 매기면 맞는 절반 때문에 틀린 절반이 묻혀요.",
                      "Q09는 '주말에 열지 않는다'로 맞게 답했는데 핵심 사실 점수가 0이에요. 정답에는 '휴실'로 적어 뒀는데 답이 다른 말을 썼거든요. 글자 대조는 같은 뜻을 다른 말로 쓰면 놓쳐요.",
                      "Q12는 흑백 50원을 100원이라고 했어요. 숫자가 틀린 답이에요.",
                      "Q14는 정답 조각이 없는 질문이라 '정보 부족'으로 멈췄으면 맞아요. 충실성과 핵심 사실은 잴 답이 없어 N/A예요."])
outs["t5b"] = S.table("⑤ 사람 판정 vs 빠른 숫자 규칙 (주장마다)", ["질문 ID", "주장", "사람", "숫자 규칙"],
                      [[q, E(c["text"]), "확인" if h else "<b>근거 없음</b>", "통과" if r else "<b>걸림</b>"]
                       for q in ("Q04", "Q05", "Q12") for c, h, r in zip(ns["REVIEWS"][q]["claims"], ans[q]["human"], ans[q]["rule"])],
                      ["숫자 규칙은 Q12의 100원을 잡았어요(근거에 100이 없음). 하지만 숫자가 없는 Q04의 '회원증 없이도'는 못 잡아요.",
                       "Q05는 C 설정의 근거 3개에 정답 조각이 없는데도 통과했어요. '3'과 '5'가 다른 조각(5권, 3일 안에)에 있었기 때문이에요. 숫자가 '어딘가에' 있는지만 봐서 생기는 헛통과예요.",
                       "그래서 자동 규칙은 빠른 1차 거름으로만 쓰고, 사람 판정이나 <a href=\"llm-judge.html\">LLM 채점</a>과 갈리는 답을 표본으로 봐요."])

# ---------------- ⑥
sm, dl, fl = ns["summary"], ns["deltas"], ns["failures"]
S.claim(sm[B_]["context_recall@3"] > sm[A_]["context_recall@3"] and sm[C_]["mrr"] > sm[B_]["mrr"] > sm[A_]["mrr"], "⑥ B, C가 A보다 높음")
S.claim(sm[A_]["candidate_recall"] == sm[B_]["candidate_recall"] == sm[C_]["candidate_recall"], "⑥ 후보 Recall은 셋이 같음")
S.claim(sm[B_]["p50_ms"] > 3 * sm[A_]["p50_ms"], "⑥ 재정렬이 시간을 여러 배로")
S.claim(not dl[B_]["worse"] and not dl[C_]["worse"], "⑥ Context Recall이 나빠진 질문 없음")
outs["t6"] = S.table("⑥ 설정마다 평균 (정답이 있는 질문만, 시간은 15개 모두)", ["설정", "후보 Recall@10", "Context Recall@3", "MRR", "nDCG@10", "시간 중앙값", "시간 95%"],
                     [[E(s), f2(v["candidate_recall"]), f2(v["context_recall@3"]), f2(v["mrr"]), f2(v["ndcg"]), f"{v['p50_ms']:.0f} ms", f"{v['p95_ms']:.0f} ms"] for s, v in sm.items()],
                     [f"후보 Recall은 셋 다 {f2(sm[A_]['candidate_recall'])}로 같아요. 재정렬과 다시 쓰기는 후보 안의 순서와 질문을 바꿨을 뿐, 후보 밖(Q05 상호대차)을 데려오지 못했어요.",
                      f"재정렬(B)은 Context Recall@3을 {f2(sm[A_]['context_recall@3'])}에서 {f2(sm[B_]['context_recall@3'])}로, 다시 쓰기까지 더한 C는 MRR을 {f2(sm[C_]['mrr'])}까지 올렸어요.",
                      f"대신 시간 중앙값이 {sm[A_]['p50_ms']:.0f} ms에서 {sm[B_]['p50_ms']:.0f} ms로 늘어요(GPU에서 후보 10개를 한 번 더 읽어요). 점수와 시간을 같이 보고 정해요.",
                      f"A에서 B로 바꿔 Context Recall@3이 오른 질문은 {', '.join(dl[B_]['better'])}이고, 내린 질문은 없어요."])
order = ["후보 밖", "모델에게 줄 3개에 못 듦", "답 상태 틀림", "근거 없는 주장"]
S.claim(fl["Q05"] == "후보 밖" and fl["Q04"] == fl["Q12"] == "근거 없는 주장" and sum(v != "통과" for v in fl.values()) == 3, "⑥ 실패 셋")
FIX = {"후보 밖": "질문 넓히기, 조각 나누기, 검색 범위 (<a href=\"rag-rerank.html\">RAG 검색 고치기</a> ⑤)", "모델에게 줄 3개에 못 듦": "재정렬, 넣을 수",
       "답 상태 틀림": "정보 부족 규칙, Prompt", "근거 없는 주장": "Prompt의 '자료만 근거로', 답 검사, 출처 확인"}
outs["t6b"] = S.table(f"⑥ {E(C_)}에서 남은 실패 (앞 단계부터 확인해 먼저 막힌 곳)", ["질문 ID", "질문", "먼저 막힌 곳", "고칠 곳"],
                      [[q, E(ROW[q]["question"]), f"<b>{t}</b>", FIX[t]] for q, t in fl.items() if t != "통과"],
                      ["Q05는 답 문장이 맞게 나왔어도 '후보 밖'이에요. 이 예제의 답은 검토표에 고정돼 있어서 그래요. 실제 모델이었다면 근거 없이 답한 셈이니, 검색에서 먼저 막힌 걸 고쳐요.",
                       "앞 단계부터 확인하는 이유는, 검색이 정답을 못 가져왔는데 답만 보고 Prompt를 고치면 헛수고가 되기 때문이에요."])

# ---------------- ⑦
ok, checks, uns = ns["gate"]
S.claim(not ok and checks["faithfulness"][2] and not checks["근거 없는 주장이 있는 답"][2] and set(uns) == {"Q04", "Q12"}, "⑦ 평균은 넘었는데 0개 조건에서 막힘")
outs["t7"] = S.table(f"⑦ quality_gate(\"{E(C_)}\") → {'통과' if ok else '막힘'}", ["확인할 것", "값", "기준", "결과"],
                     [[E(m), f2(v) if isinstance(v, float) else v, need, "통과" if good else "<b>막힘</b>"] for m, (v, need, good) in checks.items()],
                     [f"충실성 평균은 {f2(checks['faithfulness'][0])}로 기준 {checks['faithfulness'][1]}을 넘었어요. 그래도 근거 없는 주장이 있는 답이 {len(uns)}개({', '.join(uns)})라 막혔어요.",
                      "틀리면 곤란한 것(지어낸 내용, 권한 위반)은 평균 기준이 아니라 '0개'로 둬요. 평균은 좋은 답 열한 개가 나쁜 답 둘을 가려 줘요.",
                      "기준 숫자는 예시예요. 팀이 서비스의 위험도에 맞게 정하고, 바꾸면 이유와 날짜를 남겨요."])

flow = [[1, "정답 묶음", "결과 전에 정하기", f"질문 {len(Q)}개"], [2, "실행 기록", "정답은 넣지 않고", f"설정 {len(SET)}개"], [3, "찾았나", "Hit, Recall, Precision", "어느 목록으로?"],
        [4, "앞에 뒀나", "MRR, nDCG", "첫 정답, 전체 순서"], [5, "답은 맞나", "상태, 충실성, 핵심 사실", "자동 검사의 한계"],
        [6, "비교", "설정별, 질문별", f"남은 실패 {sum(v != '통과' for v in fl.values())}"], [7, "품질 기준", "평균과 0개 조건", "막힘" if not ok else "통과"]]
S.save(ROOT / "topics" / "backend" / "rag-eval-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>RAG는 질문과 관련된 조각을 <b>찾아서</b> 모델에게 주고, 그걸 근거로 <b>답하게</b> 하는 방식이에요(전체 흐름은 <a href="rag-service.html">RAG 서비스</a>). 무언가를 바꿔서 \"좋아졌다\"고 말하려면, 정해진 질문 묶음에서 정해진 정답과 비교한 점수가 있어야 해요. 그래야 다음 변경이 무엇을 좋게 하고 무엇을 나쁘게 했는지 알 수 있어요.</p>
<p>RAG는 두 단계라서 <b>따로</b> 재요.</p>
{K.table(["단계", "묻는 것", "재는 것"], [["찾기", "정답 조각을 찾았나, 앞에 뒀나", "Hit@k, Recall@k, Precision@k (③), MRR, nDCG (④)"], ["답하기", "답이 근거와 맞나, 정답과 맞나, 답할 수 없을 때 멈췄나", "충실성, 핵심 사실, 정보 부족 판단 (⑤)"]])}
<p>둘을 섞어 재면 원인을 못 찾아요. 검색이 정답을 못 가져왔는데 모델만 바꾸거나, 검색은 완벽한데 답이 지어낸 말을 하는데도 검색만 고치게 돼요.</p>
<p>무엇과 무엇을 비교하느냐로 보면 넷이에요(<a href="https://docs.langchain.com/langsmith/evaluate-rag-tutorial" target="_blank">LangSmith의 RAG 평가 튜토리얼</a>이 이렇게 나눠요). 미리 써 둔 정답 답이 꼭 있어야 하는 건 마지막 하나뿐이에요.</p>
{K.table(["비교", "묻는 것", "미리 정한 정답이 필요?", "이 쪽"], [["질문 ↔ 찾은 조각", "찾은 게 질문과 관련 있나", "이 쪽은 정답 조각을 정해 두고 순위로 재요. 정답 없이 LLM에게 채점시킬 수도 있어요", "③④"],
                                                        ["찾은 조각 ↔ 답", "근거에 있는 말만 했나 (충실성)", "아니요", "⑤"], ["질문 ↔ 답", "물은 것에 답했나 (관련성)", "아니요", "이 쪽에서는 재지 않음"],
                                                        ["답 ↔ 정답", "맞는 답인가 (정답성)", "예", "⑤ 핵심 사실"]])}
<p>순서도 튜토리얼과 같아요. 정답 묶음(dataset)을 만들고(①), 평가할 대상(target)을 돌리고(②), 채점 함수(evaluator)로 점수를 매겨요(③~⑤).</p>
<p>예제는 가상의 '가람 도서관' 안내문 조각 {len(T)}개에 대한 질문 {len(Q)}개예요. 질문마다 정답 조각을 미리 정해 뒀고, 그중 {n_ins}개는 답이 자료에 없는 질문이에요. 검색 설정 세 가지를 실제로 돌려요.</p>
{K.table(["설정", "하는 일"], [[E(A_), "질문을 그대로 e5로 찾아요"], [E(B_), "e5로 찾은 후보 10개를 Cross-Encoder로 다시 줄 세워요"], [E(C_), "앞선 질문이 있는 두 질문은 다시 쓴 질문으로 찾고, 재정렬도 해요"]])}
<p>검색과 재정렬 모델은 이 PC에서 실제로 돌려요. 다시 쓴 질문과 답 문장은 LLM을 부르지 않고 미리 정한 것을 써요(답 문장에는 일부러 틀린 답 둘을 섞었어요). 검색을 좋게 하는 방법 자체는 <a href="rag-rerank.html">RAG 검색 고치기</a>에 있고, 여기서는 그 결과를 <b>재는 법</b>만 봐요.</p>
"""
whole = """<p>정답 묶음을 만들고(①), 정답 없이 실행해 기록을 남기고(②), 검색 점수(③④)와 답 점수(⑤)를 매긴 뒤, 설정을 비교해 고칠 곳을 찾고(⑥), 다음 단계로 내보낼지 정해요(⑦). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "정답 묶음 만들고 검사하기", "결과를 보기 전에 질문마다 정답 조각과 기대 상태를 정해 두고, 그 묶음이 약속을 지키는지 코드로 확인해요.",
           "<p>평가의 기준은 <b>정답 묶음</b>(Gold Dataset)이에요. 질문마다 \"이 조각이면 답할 수 있다\"를 사람이 정해 둔 거예요. 검색 결과를 보고 정답을 고치면 점수가 의미를 잃어서, 결과를 보기 전에 정하고 판 이름(dataset_version)을 붙여요.</p>"
           + K.table(["칸", "뜻"], [["question_id", "여러 실행에서 같은 질문을 잇는 ID"], ["question, history, category", "서비스에 넣을 값 (앞선 질문이 있으면 후속 질문)"], ["gold", "정답 조각과 등급. 2 = 바로 답함, 1 = 일부만 관련. 없으면 빈 묶음"],
                                  ["expected_status", "answered 또는 insufficient_context (답이 자료에 없는 질문)"], ["reference_facts", "맞는 답에 꼭 들어갈 핵심 사실"]])
           + "<p>관련 있다/없다만 적으면 Recall과 MRR을 잴 수 있고, 등급까지 적으면 \"더 중요한 조각을 앞에 뒀나\"(nDCG)도 잴 수 있어요. 등급은 모델 점수가 아니라 사람이 정한 정답이에요. 실제로는 두 사람이 따로 판정하고, 갈린 것은 기준을 다시 맞춰 정해요.</p>"
           + K.code_table([("json.loads(path.read_text(encoding=\"utf-8\"))", "JSON 파일을 dict로"), ("assert 조건, \"메시지\"", "조건이 거짓이면 그 자리에서 멈추기 (묶음의 약속 검사)"),
                           ("set(row[\"gold\"]) <= set(TITLE)", "정답 조각이 모두 실제로 있는 조각인가 (부분집합)")]), "s1"),
    K.step(2, "평가할 검색을 돌리고 기록 남기기", "설정마다 질문을 넣어 검색을 실행하고, 무엇이 나왔는지 같은 모양으로 기록해요. 정답은 넣지 않아요.",
           "<p>평가는 <b>실행</b>과 <b>채점</b>을 나눠요. 실행하는 쪽(<code>run</code>)은 서비스에 넣을 값만 받고, 정답은 받지 않아요. 정답이 실행에 섞이면 시험지에 답을 적어 주는 셈이거든요. 점수는 실행이 끝난 기록을 받아 나중에 매겨요(③~⑥).</p>"
           "<p>검색은 e5로 찾고, B와 C는 Cross-Encoder로 다시 줄 세워요(둘 다 <a href=\"rag-rerank.html\">RAG 검색 고치기</a>에서 자세히 다뤄요). GPU는 계산을 걸어 두고 바로 다음 줄로 넘어갈 수 있어서, 시간은 <code>torch.cuda.synchronize()</code>로 끝날 때까지 기다린 뒤 재요. 처음 한 번은 GPU 준비 시간이 섞이니 버려요.</p>"
           + K.code_table([("SETTINGS = {이름: (다시 쓴 질문, 재정렬)}", "켜고 끄는 값 둘로 설정 셋"), ("run(row, setting)", "한 번 실행하고 기록(dict) 돌려주기. 정답은 안 봐요"),
                           ("torch.cuda.synchronize()", "GPU 계산이 끝날 때까지 기다리기"), ("[run(row, s) for s in SETTINGS for row in ...]", "설정 × 질문마다 한 번씩")]), "s2"),
    K.step(3, "찾았나: Hit@k, Recall@k, Precision@k", "검색 결과 앞 k개가 정답 조각을 얼마나 담았는지 세요. 셋 다 같은 교집합을 쓰지만 분모가 달라요.",
           K.table(["지표", "식", "묻는 것"], [["Hit@k", "앞 k개에 정답이 하나라도 있으면 1, 없으면 0", "하나라도 걸렸나"], ["Recall@k", "|앞 k개 ∩ 정답| ÷ |정답|", "정답을 놓치지 않았나"],
                                                ["Precision@k", "|앞 k개 ∩ 정답| ÷ k", "가져온 것 중 정답이 얼마나 되나"]])
           + "<p>정답 조각이 없는 질문은 분모가 0이라 검색 점수를 내지 않고 N/A로 빼요.</p>"
           + K.out("t3")
           + "<p>같은 공식이라도 <b>어느 목록</b>에 쓰느냐가 중요해요. 검색이 돌려준 후보 전체로 재면 \"검색이 찾았나\"이고, 실제로 모델에게 줄 몇 개로 재면 \"모델이 볼 수 있었나\"예요.</p>"
           + K.code_table([("set(ids[:k]) & set(gold)", "앞 k개와 정답의 교집합 (같은 조각은 한 번만)"), ("hit_at_k, recall_at_k, precision_at_k", "정답이 없으면 None(N/A)"), ("mean(values)", "N/A는 빼고 평균")]), "t3b"),
    K.step(4, "앞에 뒀나: MRR과 nDCG", "정답을 찾았는지뿐 아니라 몇 번째에 뒀는지를 재요. 앞 몇 개만 모델에게 주니까 순서가 곧 품질이에요.",
           K.table(["지표", "식", "보는 것"], [["Reciprocal Rank", "첫 정답이 r번째면 1/r, k 안에 없으면 0", "첫 정답의 위치"], ["MRR", "질문들의 1/r 평균", "보통 몇 번째에 첫 정답이 오나"],
                                               ["DCG@k", "Σ (2^등급 − 1) ÷ log₂(순위 + 1)", "등급 높은 조각이 앞에 있을수록 커요"], ["nDCG@k", "DCG@k ÷ (가장 좋은 순서의 DCG@k)", "0~1로 맞춘 DCG"]])
           + "<p>log로 나누는 건, 뒤로 갈수록 사람도 모델도 덜 본다고 치는 할인이에요. 1위는 그대로, 3위는 절반(log₂4 = 2로 나눔)만 쳐요. MRR은 첫 정답 하나만 보고, nDCG는 정답이 여럿이고 등급이 다를 때 전체 순서를 봐요.</p>"
           + K.code_table([("next((1 / rank for rank, i in enumerate(ids[:k], start=1) if i in gold), 0.0)", "첫 정답의 1/순위, 없으면 0 (순위는 1부터)"), ("math.log2(rank + 1)", "순위 할인"),
                           ("sorted(grades, key=grades.get, reverse=True)", "등급이 높은 순서 = 가장 좋은 순서 (분모)")]), "t4"),
    K.step(5, "답은 맞나: 상태, 근거, 핵심 사실", "답이 멈춰야 할 때 멈췄는지, 주장마다 근거로 확인되는지, 정답의 핵심 사실이 들었는지 따로 재요.",
           K.table(["지표", "무엇과 무엇을 비교", "묻는 것"], [["정보 부족 판단", "답의 상태 ↔ 기대 상태", "답이 없는 질문에서 멈췄나"], ["충실성 (faithfulness)", "답의 주장 ↔ 넣은 근거", "근거에서 이 주장을 확인할 수 있나"],
                                                       ["정답성 (correctness)", "답 ↔ 핵심 사실", "실제 정답과 숫자, 조건이 맞나"], ["관련성 (relevance)", "질문 ↔ 답", "물은 것에 답했나"]])
           + "<p>충실성은 <b>주장 단위</b>로 재요. 답을 \"~이다\" 하나하나로 나누고, 주장마다 근거에서 확인되는지 사람이 판정한 표(answer_reviews.json)로 세요. 옆에는 빠른 자동 규칙(주장 속 숫자가 근거에 있나)을 놓고 사람 판정과 비교해요.</p>"
           + K.out("t5")
           + K.code_table([("faithfulness(qid)", "확인된 주장 ÷ 전체 주장 (사람 판정표)"), ("re.findall(r\"\\d+\", c[\"text\"])", "주장 속 숫자 덩어리 모두"), ("fact_recall(answer, facts)", "핵심 사실이 답에 글자로 들어 있는 비율")]), "t5b"),
    K.step(6, "설정끼리 비교하고 고칠 곳 찾기", "같은 질문, 같은 정답에서 설정의 평균과 질문별 변화를 보고, 남은 실패를 먼저 막힌 단계로 나눠요.",
           "<p>비교할 때는 질문, 정답, 조각, 후보 수와 넣을 수, 모델 판을 모두 같게 두고 한 번에 하나만 바꿔요. 평균과 함께 질문별로 오른 것과 내린 것을 봐요.</p>"
           + K.out("t6")
           + "<p>점수가 낮다고 모델부터 바꾸면 헛수고가 되기 쉬워요. 실패한 질문마다 앞 단계부터 확인해 \"어디서 처음 막혔나\"로 나누면 고칠 곳이 보여요.</p>"
           + K.code_table([("changed(a, b, metric)", "같은 질문끼리 짝지어 오른 것, 내린 것"), ("np.percentile(times, 95)", "시간의 95% 지점 (가끔 느린 요청)"),
                           ("failure_type(qid)", "후보 밖 → 3개에 못 듦 → 상태 틀림 → 근거 없는 주장 순서로 확인")]), "t6b"),
    K.step(7, "품질 기준: 다음 단계로 내보내기 전에", "새 설정을 서비스나 Agent의 도구로 내보내기 전에 넘어야 할 최소 조건을 코드로 확인해요.",
           "<p>검색이 바뀐 판을 그대로 내보내면, 평균은 좋아도 몇 질문에서 나빠졌거나 지어낸 답이 섞여 나갈 수 있어요. 그래서 넘어야 할 조건(<b>품질 기준</b>, quality gate)을 정해 두고, 막히면 내보내기를 미루고 막힌 질문과 판을 기록해요.</p>"
           "<p>조건에는 두 종류가 있어요. 평균이 기준을 넘으면 되는 것(Recall, MRR)과, 평균이 아니라 <b>0개</b>여야 하는 것(근거 없는 주장, 권한 위반, 기준보다 나빠진 질문)이에요.</p>"
           + K.code_table([("GATE = {...}", "확인할 것과 기준 (예시)"), ("quality_gate(setting)", "(통과 여부, 항목별 (값, 기준, 결과), 근거 없는 답 목록)"),
                           ("all(ok for _, _, ok in checks.values())", "모든 항목을 넘었을 때만 통과")]), "t7"),
]
practice = [("정답 묶음은 작게 시작", "처음엔 10~20개면 충분해요. 실제로 틀린 질문을 모아 늘려 가요. 같은 묶음을 보며 고치기를 반복하면 그 묶음에만 맞춰져서, 가끔은 처음 보는 질문으로 따로 재요."),
            ("정답은 결과 전에, 판과 함께", "검색 결과를 보고 정답을 바꾸면 점수가 의미를 잃어요. 묶음의 판, 조각의 판, 판정한 사람과 날짜를 적고, 바꾸면 이력을 남겨요."),
            ("어느 목록으로 쟀나", "후보 Recall과 Context Recall은 다른 질문에 답해요. 표에 어느 목록인지 k와 함께 적어요."),
            ("평균만 보지 않기", "질문별로 오른 것과 내린 것을 같이 봐요. 틀리면 곤란한 항목은 평균 대신 0개로 막아요."),
            ("자동 채점은 보조", "숫자 규칙, 글자 대조, <a href=\"llm-judge.html\">LLM 채점</a>은 빠르지만 각자 놓치는 게 있어요(⑤). 사람 판정과 갈리는 답을 표본으로 봐요."),
            ("기록 남기기", "질문 × 설정마다 실행 기록과 점수를 한 줄씩 JSONL 파일로 남겨 두면 나중에 다시 비교할 수 있어요. LangSmith 같은 서비스는 이런 실행 기록과 비교를 화면으로 관리해 줘요. 외부로 보낼 때는 문서 원문과 개인정보가 같이 나가지 않는지 먼저 봐요."),
            ("시간도 같이 재기", "점수가 올라도 시간이 크게 늘면 쓸 수 없을 수 있어요. GPU를 쓰면 동기화한 뒤 재고, 중앙값과 95% 지점을 함께 봐요.")]
libs = [("①", "<code>json.loads</code>, <code>assert</code>", "정답 묶음 읽고 검사 (표준 라이브러리)"), ("②", "<code>SentenceTransformer</code>, <code>CrossEncoder</code>, <code>torch.cuda.synchronize</code>", "검색과 재정렬, 시간 재기"),
        ("③④", "<code>set</code>의 <code>&amp;</code>, <code>math.log2</code>", "검색 지표 (직접 계산)"), ("⑤", "<code>re.findall</code>", "주장 속 숫자 규칙"), ("②", "<code>CrossEncoder.rank</code>", "재정렬 (점수를 매겨 높은 순서로)"), ("⑥", "<code>np.percentile</code>", "시간 분포 (NumPy)")]
K.build(ROOT / "topics" / "backend" / "rag-eval.html", title="RAG 평가", h1="RAG 평가: 정답 묶음으로 검색과 답을 따로 재기", track="백엔드", crumb="백엔드 &gt; 검색과 RAG",
        one_line="RAG가 좋아졌는지는 <b>결과를 보기 전에 정해 둔 정답 묶음</b>으로 재요. 찾기는 정답 조각을 찾았는지(<b>Recall@k</b>)와 앞에 뒀는지(<b>MRR, nDCG</b>)로, 답은 근거와 맞는지(<b>충실성</b>)와 답할 수 없을 때 멈췄는지로 <b>따로</b> 재요. 한 번에 하나씩 바꿔 질문별로 비교하고, 먼저 막힌 단계를 고치고, 최소 조건(<b>품질 기준</b>)을 넘어야 다음 단계로 내보내요.",
        why=why, whole=whole, steps=steps, code_file="examples/rag_eval/evaluate.py", practice=practice, libs=libs,
        libs_note="<code>search</code>, <code>rerank</code>, <code>run</code>, <code>hit_at_k</code>, <code>recall_at_k</code>, <code>precision_at_k</code>, <code>mean</code>, <code>reciprocal_rank</code>, <code>dcg</code>, <code>ndcg_at_k</code>, <code>answer_of</code>, <code>faithfulness</code>, <code>number_rule</code>, <code>fact_recall</code>, <code>score</code>, <code>changed</code>, <code>failure_type</code>, <code>quality_gate</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rag-rerank.html\">RAG 검색 고치기</a>", "여기서 재는 다시 쓰기, 넓히기, 재정렬"), ("<a href=\"rag-service.html\">RAG 서비스</a>", "출처 검사와 정보 부족 응답"),
                   ("<a href=\"agent-tools.html\">AI Agent와 도구</a>", "기준을 넘은 RAG를 도구로 쓰기"), ("<a href=\"llm-judge.html\">LLM으로 답 채점하기</a>", "충실성을 LLM에게 채점시킬 때"),
                   ("<a href=\"../model/ml-metrics-clustering.html\">평가지표와 군집</a>", "분류의 정밀도와 재현율 (이름은 같고 대상이 달라요)"), ("<a href=\"../model/recsys.html\">추천</a>", "같은 지표(재현율@k, NDCG@10)를 추천 목록에")],
        sources=["다루는 범위: Day28 (2026-11-25) RAG Evaluation과 Debugging. 검색 실패와 생성 실패 나누기, Gold Dataset(binary와 graded relevance, 답 없는 질문), Hit@k, Recall@k, Precision@k, 후보와 Context Recall, MRR, nDCG, Faithfulness, Relevance, Correctness, LLM-as-a-Judge와 사람 평가표, 실행과 채점 나누기, 하나씩 바꾸는 비교, 실패 유형, Latency, Quality Gate. 실습 프로젝트의 Runner와 Trace 구조는 따라 하지 않고 '실제로 할 때'에 글로 적었어요",
                 "예제: <code>examples/rag_eval/evaluate.py</code> (이 파일 하나로 실행돼요). 정답 묶음 <code>examples/rag/data/gold.json</code>, 답 검토표 <code>answer_reviews.json</code>, 조각은 인덱싱 쪽과 <code>more_chunks.json</code> (모두 이 노트용 가상 자료)",
                 f"결과 값: <code>tools/pages/rag_eval.py</code>가 단계별로 실제 실행한 결과 ({ENV}). 다시 쓴 질문과 답 문장은 미리 정한 값이고, 시간은 PC와 그때의 상태에 따라 달라져요",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/model_evaluation.html#ndcg-score\" target=\"_blank\">scikit-learn nDCG</a>, <a href=\"https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/\" target=\"_blank\">Ragas Faithfulness</a>, <a href=\"https://docs.langchain.com/langsmith/evaluate-rag-tutorial\" target=\"_blank\">LangSmith: Evaluate a RAG application</a>"])
print("ok")
