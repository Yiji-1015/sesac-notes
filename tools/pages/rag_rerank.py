"""백엔드 > 검색과 RAG > RAG 검색 고치기. examples/rag/rag_service.py 의 ⑨~⑯ (①~⑧은 RAG 서비스 쪽).
검색 대상은 가람 도서관 조각 26개(인덱싱 쪽 12개 + tools/make_rag_more.py 로 만든 14개). 검색은 e5-small, 재정렬은
BAAI/bge-reranker-v2-m3 를 CUDA GPU에서 실제로 돌린다. 질문 다시 쓰기와 답변은 ChatOpenAI 에 가짜 서버를 붙여 받는다(미리 정한 값).
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/rag_rerank.py
"""
import html
import json
import statistics
import sys
import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "rag"
code = S.parse((EX / "rag_service.py").read_text(encoding="utf-8"))
ns = S.run(code, 16, EX.resolve(), filename="rag_service.py")
torch = ns["torch"]
V = {p: version(p) for p in ("torch", "sentence-transformers", "transformers", "langchain-core", "langchain-openai", "fastapi")}
GPU = torch.cuda.get_device_name(0)
VRAM = torch.cuda.max_memory_allocated() / 2**30
ENV = (f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items())
       + f", {GPU}, intfloat/multilingual-e5-small, BAAI/bge-reranker-v2-m3")
E, J = html.escape, K.josa
outs = {}

C = ns["CHUNKS"]
EVAL, CK, XK = ns["EVAL"], ns["CANDIDATE_K"], ns["CONTEXT_K"]
cid = ns["cid"]


def title(i):
    c = C[i - 1]
    return c["heading"] or c["file"]


def lab(i):
    return f"[{i}] {E(title(i))}"


def rk(r):
    return f"{r}위" if r else "<b>후보 밖</b>"


# ---------------- 왜: 예제 입력 전체
more = ns["MORE"]
S.claim(len(C) == 26 and len(more) == 14, "조각 26개 = 12 + 14")
why_chunks = K.table(["citation_id", "파일", "분류", "제목", "본문"],
                     [[c["citation_id"], c["file"], c["category"], E(c["heading"] or "(없음)"), E(c["body"]).replace("\n\n", " / ")] for c in more])
why_eval = K.table(["앞선 질문", "질문", "분류", "정답 조각"],
                   [[E(", ".join(h)) or "(없음)", E(q), cat, lab(g)] for h, q, cat, g in EVAL])
b = ns["baseline"]
S.claim(b[0]["rank"] > XK and title(b[0]["top"][0]) == "도서 예약 취소" and title(b[0]["top"][1]) == "프로그램 신청 취소", "왜: 그럼 취소는? 위 둘이 다른 취소")
why = f"""<p><a href="rag-service.html">RAG 서비스</a> 쪽에서 만든 서비스는 사용자의 질문을 <b>그대로</b> 검색해요. 그리고 찾은 후보 중 위에서 몇 개만 모델에게 줘요. 모델은 받은 조각만 근거로 답하니, 정답 조각이 그 몇 개 안에 들지 못하면 모델이 아무리 좋아도 답할 수 없어요.</p>
<p>그런데 실제 질문은 검색하기 좋은 모양이 아니에요. 대화 중에는 \"그럼 취소는?\"처럼 무엇의 취소인지 앞선 질문에만 있고, 사람들은 \"상호대차\"처럼 안내문과 다른 말을 써요. 또 검색 모델은 빨리 찾는 대신 순서가 조금씩 어긋나요. 이 쪽은 이런 실패를 <b>어디서 났는지 나눠</b> 고쳐요.</p>
<p>검색 대상은 <a href="rag-service.html">RAG 서비스</a>와 같은 가람 도서관 조각이에요. 원래 있던 12개(대출, 열람실, 분실, 회원 규정 등)에 같은 도서관의 안내 조각 {len(more)}개를 더해 {len(C)}개에서 찾아요. 더한 조각은 아래와 같아요.</p>
{why_chunks}
<p>2025년 공지는 2026년 공지가 나와서 <b>지금 유효한 판이 아니라고</b> 표시해 뒀어요(⑪에서 써요). 그리고 고치기 전에 질문 {len(EVAL)}개와 각각의 정답 조각을 먼저 정해 뒀어요. 결과를 본 뒤에 정답을 바꾸면 비교가 의미 없어지니까요.</p>
{why_eval}
<p>지금 서비스로 첫 질문을 그대로 찾으면, 위 {XK}개는 {lab(b[0]['top'][0])}, {lab(b[0]['top'][1])}, {lab(b[0]['top'][2])}{J(title(b[0]['top'][2]), '이에요', '예요')}. '취소'라는 말만 보고 다른 취소를 가져왔고, 정답인 {lab(EVAL[0][3])}는 {b[0]['rank']}위라 모델에게 가지 않아요.</p>
<p>고치는 곳은 실패한 자리에 따라 달라요.</p>
{K.table(["실패한 자리", "보이는 모습", "고치는 곳"], [["검색 전: 질문", "앞선 대화가 있어야 뜻이 통하는 질문", "질문 다시 쓰기(⑩)"],
                                         ["검색 전: 표현", "안내문과 다른 말을 써서 정답이 후보에도 안 옴", "용어 넓히기와 여러 질문(⑫)"],
                                         ["검색 범위", "후보 수가 모자라거나, 옛 판 문서가 섞임", "후보 수와 조건(⑪)"],
                                         ["검색 뒤: 순서", "정답이 후보에는 있는데 아래에 있음", "다시 순위 매기기(⑬)"]])}
<p>질문이 잘못되면 <b>찾기 전에</b>, 순서가 잘못되면 <b>찾은 뒤에</b> 고쳐요. 순서를 바꾸는 단계는 후보 안에서만 고를 수 있어서, 후보에 없는 정답은 앞 단계에서 찾아와야 해요.</p>
<p>코드는 RAG 서비스 쪽의 <code>examples/rag/rag_service.py</code>를 ⑨부터 이어서 키워요. ①~⑧에서 만든 검색(<code>search_hits</code>), Document 변환, 근거 고르기(<code>build_context</code>), 생성과 검사를 그대로 써요. 재정렬 모델은 이 PC의 GPU에서 실제로 돌리고, 질문 다시 쓰기와 답변은 RAG 서비스 쪽처럼 가짜 서버가 미리 정한 값을 돌려줘요.</p>
"""

# ---------------- ⑨
DIAG = ["순위 밀림", "통과", "후보 밖", "순위 밀림", "통과", "통과"]
S.claim([x["diagnosis"] for x in b] == DIAG, "⑨ 진단 순서")
S.claim(title(b[1]["top"][0]) == "스터디룸 예약" and b[1]["rank"] == 2, "⑨ 노트북 질문 1등은 스터디룸 예약")
outs["t9"] = S.table(f"⑨ 질문을 그대로 검색 (후보 {CK}개, 넣을 수 {XK}개)", ["질문", "정답 조각", "정답 등수", f"위 {XK}개", "진단"],
                     [[E(x["question"]), lab(g), rk(x["rank"]), "<br>".join(lab(i) for i in x["top"]), f"<b>{x['diagnosis']}</b>"] for x, (_, _, _, g) in zip(b, EVAL)],
                     [f"'통과'가 {DIAG.count('통과')}개, '순위 밀림'이 {DIAG.count('순위 밀림')}개, '후보 밖'이 {DIAG.count('후보 밖')}개예요. 사용자에게는 똑같이 '답이 틀림'으로 보여도, 순위 밀림과 후보 밖은 고칠 곳이 달라요.",
                      f"노트북 질문은 정답이 {b[1]['rank']}위라 '통과'지만, 1위는 {lab(b[1]['top'][0])}이에요. 둘 다 '몇 시간까지'가 들어 있어서, 모델이 스터디룸의 시간을 노트북 답으로 쓸 수도 있어요. 등수 하나로는 다 안 보이는 실패예요.",
                      f"상호대차 질문은 정답이 {b[2]['rank']}위라 후보 {CK}개에도 못 들어와요. 이건 순서를 바꾸는 단계로는 못 고쳐요."])

# ---------------- ⑩
rw, ar, yc, uc, chk = ns["rewritten"], ns["after_rewrite"], ns["year_case"], ns["unclear_case"], ns["rewrite_checks"]
S.claim(ar == [1, 1] and [s["status"] for s in rw] == ["ready", "ready"], "⑩ 다시 쓰면 둘 다 1위")
S.claim(yc["status"] == "ready" and uc["status"] == "needs_clarification", "⑩ 올해는 ready, 그거는 되묻기")
chk_vals = list(chk.values())
S.claim(chk_vals[0] == [] and len(chk_vals[1]) == 2 and len(chk_vals[2]) == 1, "⑩ 다시 쓰기 검사 0, 2, 1개")
rq = next(p for p in ns["sent"] if p.get("model") == "demo-small-model")
fmt = rq["text"]["format"]
outs["s10"] = S.box("⑩ rewrite_step(state)", [
    (f"가짜 서버가 받은 요청: model \"{rq['model']}\", 출력 형식 {fmt['type']} \"{fmt['name']}\", 역할 {[x['role'] for x in rq['input']]}", 1),
    (f"\"{E(EVAL[0][1])}\" (앞선 질문 \"{E(EVAL[0][0][0])}\") → search_query \"{E(rw[0]['search_query'])}\": 정답 {b[0]['rank']}위 → {ar[0]}위", 2, "newf"),
    (f"\"{E(EVAL[1][1])}\" (앞선 질문 \"{E(EVAL[1][0][0])}\") → search_query \"{E(rw[1]['search_query'])}\": 정답 {b[1]['rank']}위 → {ar[1]}위", 3, "newf"),
    (f"\"올해도 똑같아?\" → \"{E(yc['search_query'])}\", status {yc['status']}", 4),
    (f"\"그거 바꿀 수 있어?\" (앞선 질문 없음) → status {uc['status']}, search_query 없음", 5, "bad")],
    ["RAG 서비스 쪽 ⑥과 같은 방식이에요. 출력 모양(SearchQuery)을 JSON Schema로 붙여 보내요. 다시 쓰기는 짧은 일이라 작은 모델을 따로 쓰는 경우가 많아서 모델 객체도 따로 만들었어요.",
     "앞선 질문의 '스터디룸 예약'을 채웠더니 다른 취소들을 제치고 정답이 1위가 됐어요. 원래 질문 \"그럼 취소는?\"은 question에 그대로 있어요.",
     "노트북 질문도 '노트북'이 들어가자 스터디룸 조각을 제쳤어요.",
     "'올해'는 그대로 뒀어요. 올해가 몇 년인지는 대화에 없으니 채우지 않아요(⑪에서 판 조건으로 풀어요).",
     "'그거'가 무엇인지 앞선 질문이 없어서 정할 수 없어요. 상상으로 채우지 않고 되묻기 상태로 멈춰요. 이때는 검색도 하지 않아요(⑯)."])
outs["t10"] = S.table("⑩ check_rewrite(\"올해도 똑같아?\", [앞선 질문], 다시 쓴 질문)", ["다시 쓴 질문", "찾은 문제", "결과"],
                      [[E(t), E(", ".join(p)) or "없음", "<b>되묻기</b>" if p else "통과"] for t, p in chk.items()],
                      ["대화에 있는 말만 썼고 '올해'도 남겼어요. 통과예요.",
                       "'올해'를 '2025년'으로 바꿨어요. 대화에 없던 숫자가 생기고 '올해'가 빠졌어요. 기간이 바뀌면 다른 공지를 찾게 돼요.",
                       "'올해'를 빼면 연도와 상관없는 질문이 돼요.",
                       "이 검사는 숫자와 때를 가리키는 말만 봐요. '수수료 없이'처럼 없는 조건을 붙이거나 '안 된다'를 '된다'로 뒤집는 건 못 잡아요. 그런 건 표본을 사람이 보거나 <a href=\"llm-judge.html\">LLM 채점</a>으로 따로 재요."])

# ---------------- ⑪
ks, o3, av, no = ns["k_sweep"], ns["order3"], ns["all_versions"], ns["now_only"]
g3 = ns["gold3"]
S.claim(ks == {3: False, 5: False, 10: False, 20: True} and o3.index(g3) + 1 == b[2]["rank"], "⑪ 후보 20개에서만 정답")
c25, c26 = cid("notice_2025.txt"), cid("notice_2026.txt")
S.claim([h["citation_id"] for h in av] == [c26, c25] and abs(av[0]["score"] - av[1]["score"]) < 0.01, "⑪ 판을 안 가리면 2026, 2025 둘이 거의 같은 점수")
S.claim(c25 not in [h["citation_id"] for h in no] and no[0]["citation_id"] == c26, "⑪ 지금 판만이면 2025 빠짐")
n_use = sum(1 for c in C if c["category"] == "이용")
outs["t11"] = S.table(f"⑪ \"{E(ns['q3'])}\": 후보 수를 바꾸면", ["candidate_k", "정답이 후보에?", "재정렬이 정답을 고를 수 있나"],
                      [[k, "있음" if v else "없음", "있음" if v else "<b>없음</b>"] for k, v in ks.items()],
                      [f"정답은 분류 '이용'의 {n_use}개 중 {o3.index(g3) + 1}위예요. 후보를 20개로 늘리면 들어오지만, 그건 '이용' 조각이 {n_use}개뿐이라 전부 가져온 거예요. 조각이 수만 개라면 후보를 늘려서 해결하기 어려워요.",
                       "후보 수를 늘리면 재정렬할 짝이 늘어 시간이 늘어요(⑮). 그래서 후보 수만 키우지 말고, 정답이 후보에 들어오게 질문을 고쳐요(⑫)."])
outs["s11"] = S.box(f"⑪ \"{E(ns['year_q'])}\" (분류 '공지', 위 2개)",
                    [("판을 가리지 않으면 (search_hits)", 0), *[(f"  {lab(h['citation_id'])} 유사도 {h['score']:.3f}", 1) for h in av],
                     ("지금 유효한 판만 (search_now)", 0), *[(f"  {lab(h['citation_id'])} 유사도 {h['score']:.3f}", 2, "newf") for h in no]],
                    [f"2026년 공지와 2025년 공지의 유사도 차이가 {av[0]['score'] - av[1]['score']:.3f}뿐이에요. 문장 모양이 거의 같아서, 점수로는 어느 해가 지금인지 가릴 수 없어요. 둘 다 모델에게 가면 '2시간'과 '3시간'이 같이 들어가요.",
                     "판 표시(current)로 걸렀더니 2025년 공지가 빠지고 다른 공지가 그 자리에 왔어요. 이런 조건은 질문을 다시 쓰든 여러 질문으로 찾든 <b>모든 검색에 똑같이</b> 붙여요(⑫의 multi_search도 search_now를 써요)."])

# ---------------- ⑫
pq, u3, f3 = ns["per_query"], ns["union3"], ns["found3"]
q3list = ns["queries3"]
S.claim(len(q3list) == 2 and g3 not in pq[q3list[0]] and pq[q3list[1]][0] == g3, "⑫ 바꾼 질문에서 정답 1위")
u_ids = [h["citation_id"] for h in u3]
pos = u_ids.index(g3) + 1
S.claim(pos > XK, "⑫ 합친 목록에서도 정답이 위 3개 밖")
outs["t12"] = S.table("⑫ multi_search(queries, \"이용\", k=5): 질문마다 위 5개", ["질문", "찾은 조각 (위 5개)"],
                      [[E(q), ", ".join(("<b>" + lab(i) + "</b>") if i == g3 else lab(i) for i in ids)] for q, ids in pq.items()],
                      [f"원래 질문으로는 정답이 위 5개에 없어요. 용어집으로 '상호대차'를 안내문의 말로 바꾼 질문에서는 정답이 1위예요.",
                       f"둘을 chunk_id로 합치면 {len(u_ids)}개예요(두 질문이 함께 찾은 조각 {sum(1 for v in f3.values() if len(v) > 1)}개). 겹치는 조각은 한 번만 남기고, found_by에 어느 질문이 찾았는지 남겨요.",
                       f"합친 순서는 '먼저 찾은 순서'라 정답은 {pos}번째예요. 넣을 {XK}개 안에는 아직 못 들어요. 후보에 넣는 데까지가 여러 질문의 일이고, 순서는 ⑬에서 정해요."])

# ---------------- ⑬
r4, c4, lg = ns["ranked4"], ns["cands4"], ns["logits4"]
g4 = ns["gold4"]
e5_order = [d.metadata["citation_id"] for d in c4]
ce_order = [d.metadata["citation_id"] for d in r4]
S.claim(e5_order.index(g4) + 1 == b[3]["rank"] and ce_order[0] == g4, "⑬ 노쇼 e5 5위 → 1위")
sig = 1 / (1 + 2.718281828459045 ** -float(lg[e5_order.index(g4)]))
S.claim(abs(sig - r4[0].metadata["rerank_score"]) < 1e-3, "⑬ 점수 = sigmoid(logit)")
rows13 = []
for d in r4:
    i = d.metadata["citation_id"]
    j = e5_order.index(i)
    rows13.append([("<b>" + lab(i) + "</b>") if i == g4 else lab(i), f"{j + 1}위", f"{d.metadata['score']:.3f}", f"{ce_order.index(i) + 1}위", f"{d.metadata['rerank_score']:.4f}", f"{float(lg[j]):.2f}"])
g4_sc = r4[0].metadata["rerank_score"]
lg_g4 = f"{float(lg[e5_order.index(g4)]):.2f}"
S.claim(r4[1].metadata["rerank_score"] < 0.2 < g4_sc, "⑬ 1위만 높고 나머지는 훨씬 낮음")
outs["t13"] = S.table(f"⑬ \"{E(ns['q4'])}\": 후보 {len(c4)}개를 재정렬", ["조각", "e5 순위", "e5 유사도", "재정렬 순위", "재정렬 점수", "logit"], rows13,
                      [f"e5에서 {b[3]['rank']}위였던 '노쇼'가 1위가 됐어요. e5 유사도로는 후보들이 {min(d.metadata['score'] for d in c4):.3f}~{max(d.metadata['score'] for d in c4):.3f}에 몰려 있었는데, 재정렬 점수는 1위만 {g4_sc:.3f}이고 나머지는 훨씬 낮아요. 질문과 본문을 같이 읽어서 '안 가면'과 '연락 없이 오지 않으면'이 같은 상황인 걸 알아봤어요.",
                       f"재정렬 점수는 logit(모델이 낸 원래 값)을 0~1로 줄인 거예요(sigmoid). 1위의 logit {lg_g4}{J(lg_g4, '이', '가')} {g4_sc:.3f}{J(f'{g4_sc:.3f}', '이', '가')} 됐어요. 확률처럼 보여도 '맞을 확률'이 아니에요. 같은 질문 안에서 순서를 정하는 데만 써요.",
                       "e5 유사도와 재정렬 점수는 서로 다른 모델이 다른 방법으로 낸 값이라, 숫자를 맞대어 비교하지 않아요."])
r3, ys = ns["ranked3"], ns["year_scores"]
S.claim(r3[0].metadata["citation_id"] == g3 and r3[0].metadata["rerank_score"] < 0.01, "⑬ 상호대차 1위지만 점수 아주 낮음")
S.claim(ys[0].metadata["citation_id"] == c26 and ys[1].metadata["citation_id"] == c25 and ys[0].metadata["rerank_score"] - ys[1].metadata["rerank_score"] < 0.01, "⑬ 올해 공지 둘 다 높은 점수")
outs["s13"] = S.box("⑬ 다른 두 질문을 재정렬하면", [
    (f"\"{E(ns['q3'])}\" + ⑫에서 합친 후보 {len(u_ids)}개 → 1위 {lab(r3[0].metadata['citation_id'])} 점수 {r3[0].metadata['rerank_score']:.4f}, 2위 {lab(r3[1].metadata['citation_id'])} {r3[1].metadata['rerank_score']:.4f}", 1),
    (f"\"{E(ns['year_q'])}\" + 2025년, 2026년 공지 → {lab(ys[0].metadata['citation_id'])} {ys[0].metadata['rerank_score']:.4f}, {lab(ys[1].metadata['citation_id'])} {ys[1].metadata['rerank_score']:.4f}", 2, "bad")],
    [f"정답이 후보에 들어오니 1위로 올랐어요. 그런데 점수가 {r3[0].metadata['rerank_score']:.4f}로 아주 낮아요. 이 모델도 '상호대차'를 잘 몰라서, 다른 후보가 더 낮았을 뿐이에요. 점수가 낮은 1위를 그대로 믿지 말고, 기준 점수 아래면 정보 부족으로 다루는 방법도 있어요(기준은 평가 질문으로 정해요).",
     "두 공지 모두 거의 최고점이에요. 더 정밀한 모델도 어느 해가 지금인지는 몰라요. 최신 여부와 열람 권한은 점수가 아니라 ⑪의 조건으로 정해요."])

# ---------------- ⑭
s1 = ns["state1"]
inc1 = [d.metadata["citation_id"] for d in s1["included_docs"]]
S.claim(ns["calls_after"] - ns["calls_before"] == 1 and inc1[0] == EVAL[0][3], "⑭ invoke 때 한 번 호출, 정답이 맨 앞")
outs["s14"] = S.box("⑭ search_pipeline.invoke(state) 뒤의 state", [
    (f"question \"{E(s1['question'])}\"", 1), (f"search_query \"{E(s1['search_query'])}\"", 2, "newf"),
    (f"candidate_docs → {[d.metadata['citation_id'] for d in s1['candidate_docs']]}", 3, "newf"),
    (f"ranked_docs → {[d.metadata['citation_id'] for d in s1['ranked_docs']]}", 4, "newf"),
    (f"included_docs → {inc1}", 5, "newf"),
    (f"가짜 서버 호출 수: | 로 이은 직후 {ns['calls_before']}, invoke 뒤 {ns['calls_after']}", 6)],
    ["원래 질문은 처음 그대로예요. 답을 만들 때는 이걸 써요(⑯).",
     "다시 쓰기 단계가 새 이름으로 더했어요. 검색과 재정렬은 이걸 써요.",
     f"검색 단계의 후보 {len(s1['candidate_docs'])}개, e5 순서예요.",
     "재정렬 단계가 같은 조각을 순서만 바꿔 새 이름으로 남겼어요. 후보 목록을 덮어쓰지 않았으니 둘을 나란히 비교할 수 있어요.",
     f"근거 고르기가 위 {XK}개를 넣었어요. 출처로 댈 수 있는 번호는 이것뿐이에요.",
     "앞 단계까지 쓴 호출이 이미 있어서 0이 아니에요. | 로 잇기만 할 때는 늘지 않고, invoke에서 다시 쓰기 요청 1번이 늘었어요."])

# ---------------- ⑮
cmp_, SET = ns["compare"], ns["SETTINGS"]
qs = [e[1] for e in EVAL]
row = {name: [cmp_[name, q] for q in qs] for name in SET}
S.claim(row["기준"] == [b_["rank"] if b_["rank"] <= CK else None for b_ in b], "⑮ 기준 = ⑨")
S.claim(row["다시 쓰기만"][:2] == [1, 1] and row["다시 쓰기만"][3] == b[3]["rank"], "⑮ 다시 쓰기는 1, 2번만")
S.claim(row["재정렬만"][3] == 1 and 1 < row["재정렬만"][0] <= XK, "⑮ 재정렬만: 노쇼 1위, 취소는 3위 안")
S.claim(all(row[n][2] is None for n in SET if n != "셋 다 (+ 넓히기)") and row["셋 다 (+ 넓히기)"] == [1] * 6, "⑮ 상호대차는 넓히기만, 셋 다면 모두 1위")


def cell(r):
    if r is None:
        return "<b>후보 밖</b>"
    return f"<b>{r}위</b>" if r <= XK else f"{r}위"


hit = {n: sum(1 for r in v if r and r <= XK) for n, v in row.items()}
outs["t15"] = S.table(f"⑮ 정답 순위 (진하게: 넣을 {XK}개 안)", ["설정"] + [E(q) for q in qs] + [f"{XK}개 안"],
                      [[n] + [cell(r) for r in v] + [f"{hit[n]}/{len(qs)}"] for n, v in row.items()],
                      [f"다시 쓰기는 앞선 질문이 필요한 두 질문을, 재정렬은 '안 가면' 질문을, 넓히기는 상호대차 질문을 고쳐요. 하나씩 켜 봐야 무엇이 무엇을 고쳤는지 보여요. 한꺼번에 켜고 좋아졌다고만 하면 어느 단계가 필요한지 몰라요.",
                       f"재정렬만 켜도 '그럼 취소는?'의 정답이 {row['재정렬만'][0]}위로 들어오지만, 질문 자체가 뜻이 비어 있어서 다시 쓰기를 켰을 때만 1위가 돼요.",
                       "상호대차 질문은 넓히기 없이는 어떤 설정에서도 후보 밖이에요. 재정렬은 후보 밖 조각을 데려오지 못해요.",
                       "질문 6개는 성질을 보이려고 고른 예예요. 실제로 켤지는 질문을 많이 모아 같은 방법으로 재 보고 정해요."])
tm = defaultdict(list)
for n, ms in ns["timing"]:
    tm[n].append(ms)
med = {n: statistics.median(v) for n, v in tm.items()}
rc = defaultdict(list)
for n, ms in ns["rerank_cost"]:
    rc[n].append(ms)
rmed = {k: statistics.median(v) for k, v in rc.items()}
S.claim(med["재정렬"] == max(med.values()) and med["검색"] > med["근거 고르기"], "⑮ 재정렬이 가장 오래 걸림")
S.claim(rmed[5] < rmed[10] < rmed[17], "⑮ 후보가 많을수록 재정렬이 느림")
n_rep = len(tm["검색"])
n_q = len(ns["prepare_queries"](rw[0]["search_query"], ns["expand"](rw[0]["search_query"])))
S.claim(n_q == 2, "⑮ 넓히기로 질문 둘")
outs["t15b"] = S.table(f"⑮ 단계마다 걸린 시간 (\"{E(EVAL[0][1])}\", 넓히기 켬, {n_rep}번 잰 가운데 값)", ["단계", "시간", "무엇이 드나"],
                       [["다시 쓰기", f"{med['다시 쓰기']:.1f} ms", "가짜 서버라 실제 모델 호출(네트워크와 생성)은 빠져 있어요"],
                        ["검색", f"{med['검색']:.1f} ms", f"질문 {n_q}개(원래 질문과 '철회'로 바꾼 질문)를 e5로 벡터로 만들고 {len(C)}개와 비교 (GPU)"],
                        ["재정렬", f"{med['재정렬']:.1f} ms", "후보마다 Cross-Encoder 한 번 (GPU)"], ["근거 고르기", f"{med['근거 고르기']:.2f} ms", "파이썬 반복문"]]
                       + [[f"재정렬, 후보 {k}개", f"{v:.1f} ms", ""] for k, v in rmed.items()],
                       [f"재정렬이 검색의 {med['재정렬'] / med['검색']:.1f}배로 가장 오래 걸려요. 후보를 {min(rmed)}개에서 {max(rmed)}개로 늘리면 {rmed[min(rmed)]:.0f} ms에서 {rmed[max(rmed)]:.0f} ms가 돼요. 짝 하나하나를 모델에 통과시키니까요.",
                        "실제 서비스에서는 여기에 다시 쓰기와 답변의 LLM 호출 시간이 더해져요. 보통 이 둘이 가장 길어서, 짧고 뜻이 분명한 질문은 다시 쓰기를 건너뛰는 정책을 쓰기도 해요(건너뛴 사실은 기록해요).",
                        f"GPU 작업은 명령만 걸어 두고 바로 다음 줄로 넘어갈 수 있어요. 그래서 재기 전후에 <code>torch.cuda.synchronize()</code>로 끝날 때까지 기다려요. 시간은 실행할 때마다 조금씩 달라요({GPU})."])

# ---------------- ⑯
api = ns["api2"]
codes = [api[k][0] for k in api]
S.claim(codes == [200, 200, 200, 200, 422, 422, 502], "⑯ 200 넷, 422 둘, 502 하나")
S.claim([api[k][1].get("status") for k in list(api)[:4]] == ["answered", "needs_clarification", "answered", "insufficient_context"], "⑯ 상태 넷")


def summary(code_, body):
    if code_ == 200:
        if body["status"] == "needs_clarification":
            return f"status {body['status']}, message \"{body['message']}\""
        src = ", ".join(f"[{s['citation_id']}] {s['heading'] or s['file']}" for s in body["sources"]) or "없음"
        return f"status {body['status']}, search_query \"{body['search_query']}\", 출처 {src}"
    d = body["detail"]
    return f"detail \"{d}\"" if isinstance(d, str) else f"{d[0]['type']} at {'.'.join(map(str, d[0]['loc']))}"


outs["t16"] = S.table("⑯ TestClient로 POST /rag/enhanced", ["요청", "HTTP", "응답 요약"], [[E(k), f"<b>{c}</b>", E(summary(c, bd))] for k, (c, bd) in api.items()],
                      ["후속 질문은 다시 쓴 질문으로 찾고, 답은 원래 질문으로 만들어요. 출처는 RAG 서비스 쪽처럼 서버가 보관한 원문 위치를 붙여요.",
                       "가리키는 대상을 모르면 검색도 답도 하지 않고 되물어요. 실패가 아니라 정상 응답(200)이에요.",
                       "재정렬을 끄면 '노쇼' 조각이 넣을 3개 밖으로 밀려서 정보 부족이 돼요. 가짜 서버는 근거 구절이 자료에 있을 때만 답하게 만들어 둔 것이라, 실제 모델이라면 엉뚱한 조각으로 답을 지어낼 수도 있어요. 어느 쪽이든 근거가 빠진 게 원인이에요.",
                       "앞선 질문이 6개를 넘거나 후보 수가 20을 넘으면 입구에서 422로 막아요. 검색 함수가 받을 수 있는 범위와 요청 모델의 범위를 같게 맞춰 둬야, API는 통과했는데 안에서 터지는 일이 없어요.",
                       "재정렬 모델이 고장 나면(예: GPU 메모리 부족) 502와 어느 단계인지(rerank_error)를 알려요. 검색 순서로 몰래 바꿔 정상처럼 답하지 않아요. 그렇게 바꾸는 게 정책이라면 응답과 기록에 표시해요."])
lg16 = ns["enhanced_log"]
outs["s16"] = S.box("⑯ 서비스가 남긴 기록 (log)", [json.dumps(x, ensure_ascii=False) for x in lg16],
                    ["질문, 검색용 질문, 후보, 넣은 조각, 결과를 남겨 두면 '왜 정보 부족이었나'를 ⑨의 세 갈래로 나눠 볼 수 있어요. 키나 비밀번호, 긴 원문은 남기지 않아요."])

flow = [[9, "기준 재기", "질문 그대로 검색", f"통과 {DIAG.count('통과')}/{len(EVAL)}"], [10, "다시 쓰기", "빠진 대상 채우기", "question ≠ search_query"],
        [11, "후보 모으기", "후보 수, 판 조건", "후보 밖은 못 고름"], [12, "여러 질문", "용어 넓히고 합치기", "chunk_id로 합치기"],
        [13, "재정렬", "Cross-Encoder", "질문과 본문을 함께"], [14, "잇기", "Runnable과 state", "단계마다 새 이름"],
        [15, "비교", "하나씩 켜기, 시간", f"{hit['기준']}/6 → {hit['셋 다 (+ 넓히기)']}/6"], [16, "API", "POST /rag/enhanced", "되묻기, 정보 부족, 502"]]
S.save(ROOT / "topics" / "backend" / "rag-rerank-data.js", {"code": code, "names": S.names(code), "range": [9, 16], "outs": outs, "flow": flow, "env": ENV})

whole = f"""<p>검색 전(⑩), 검색(⑪⑫), 검색 뒤(⑬)를 하나씩 고치고, 단계로 이어(⑭) 비교한 뒤(⑮) API로 열어요(⑯). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>처음과 끝을 보면, 질문 {len(EVAL)}개 중 정답이 넣을 {XK}개 안에 든 질문이 {hit['기준']}개에서 {hit['셋 다 (+ 넓히기)']}개가 돼요. 대신 재정렬에 질문마다 약 {med['재정렬']:.0f} ms(이 PC의 GPU)가 더 들고, 다시 쓰기에는 LLM 호출이 하나 더 들어요.</p>"""

steps = [
    K.step(9, "조각을 더 넣고 실패 나누기", f"검색할 조각을 {len(C)}개로 늘리고, 정답을 미리 정한 질문 {len(EVAL)}개를 지금 서비스 그대로(질문을 그대로 검색) 돌려 정답이 몇 위인지 재요.",
           "<p>고치기 전에 <b>어디서</b> 실패하는지부터 나눠요. 후보 10개를 찾아 위 3개를 넣는 예시 설정이라면, 정답의 등수로 세 갈래가 돼요.</p>"
           + K.table(["진단", "정답 등수", "뜻", "고치는 곳"], [["통과", f"1~{XK}위", "모델에게 정답이 가요", "-"], ["순위 밀림", f"{XK + 1}~{CK}위", "찾기는 했는데 넣을 자리에 못 들어요", "검색 뒤에서 순서 바꾸기(⑬)"],
                                                         ["후보 밖", f"{CK}위 밖", "검색이 못 찾았어요", "검색 전(⑩⑫)이나 검색 범위(⑪)"]])
           + "<p>①의 <code>search_hits</code>는 바깥의 <code>CHUNKS</code>와 <code>vectors</code>를 그때그때 읽어요. 그래서 조각 목록에 더하고 벡터만 다시 만들면 검색 함수는 고치지 않아도 돼요. citation_id는 앞의 12개 다음부터 이어 붙여요.</p>"
           + K.code_table([("CHUNKS += MORE", "조각 목록 뒤에 더하기 (같은 리스트를 늘려요)"), ("rank_of(gold, ids)", "정답이 목록의 몇 번째인지, 없으면 None"),
                           ("diagnose(rank)", "등수로 통과, 순위 밀림, 후보 밖 나누기"), ("search_hits(question, category, k=len(CHUNKS))", "분류 안의 전체 순서 (등수를 끝까지 보려고)")]), "t9"),
    K.step(10, "질문 다시 쓰기: 대화에서 빠진 대상 채우기", "앞선 질문을 보고, 대화를 모르는 사람도 알아듣는 검색용 질문(standalone query)을 따로 만들어요. 원래 질문은 지우지 않아요.",
           "<p>\"그럼 취소는?\"을 그대로 벡터로 바꾸면 '취소'라는 뜻만 남아요. 무엇의 취소인지는 앞선 질문에 있는데, 검색 모델은 질문 하나만 봐요. 그래서 대화를 읽을 수 있는 LLM에게 <b>검색용 질문</b>을 따로 쓰게 해요(Query Rewrite).</p>"
           "<p>다시 쓰기의 규칙은 '채우되 만들지 않기'예요.</p>"
           "<ul><li>앞선 질문에서 확인되는 대상만 채워요.</li><li>대화에 없는 날짜, 숫자, 조건을 더하지 않고, 기간(올해, 2025년)과 부정(안 된다)은 그대로 둬요.</li><li>무엇을 가리키는지 정할 수 없으면 <code>needs_clarification</code>으로 되묻게 해요.</li></ul>"
           "<p>결과는 <code>search_query</code>라는 <b>새 이름</b>으로 적고, 사용자가 한 말은 <code>question</code>에 그대로 둬요. 답은 사용자가 실제로 한 질문에 해야 하고, 다시 쓰기가 틀렸을 때 둘을 나란히 봐야 원인을 찾을 수 있어요.</p>"
           + K.out("s10")
           + "<p>모델이 규칙을 어길 수도 있으니, 받은 다음 코드로 한 번 더 봐요. <code>check_rewrite</code>는 대화에 없던 숫자가 생겼는지, 원래 질문의 '올해' 같은 때를 가리키는 말이 빠졌는지 봐요.</p>"
           + K.code_table([("rewrite_model.with_structured_output(SearchQuery, method=\"json_schema\", strict=True)", "다시 쓴 질문과 되묻기 여부를 정해진 모양으로 받기"),
                           ("state[\"history\"][-6:]", "앞선 질문은 최근 6개만 (오래된 대화까지 넣으면 길고 엉뚱한 대상을 끌어와요)"),
                           ("REWRITES", "가짜 서버가 돌려줄 미리 정한 다시 쓰기 (실제 모델이 아니에요)"),
                           ("re.findall(r\"\\d+\", rewritten)", "다시 쓴 질문의 숫자 덩어리 모두 (\\d+는 숫자가 하나 이상 이어진 것)"),
                           ("{**state, \"search_query\": ...}", "앞 state를 그대로 두고 새 키를 더한 dict")]), "t10"),
    K.step(11, "후보 모으기: 몇 개를, 어떤 조건으로", "검색에서 몇 개를 가져올지(candidate_k)와, 모든 검색에 똑같이 붙일 조건(분류, 지금 유효한 판)을 정해요.",
           "<p>후보 수(<code>candidate_k</code>)와 넣을 수(<code>context_k</code>)는 달라요(<a href=\"rag-service.html\">RAG 서비스</a> ③). 이 쪽의 예시 설정은 후보 10개를 찾아 3개를 넣는 거예요. 후보를 넉넉히 가져와야 뒤에서 순서를 바꿀 여지가 생기는데, 다시 순위를 매기는 단계는 <b>후보 안에서만</b> 골라요. 후보에 없는 조각을 새로 데려오지 못해요.</p>"
           + K.out("t11")
           + "<p>조건도 함께 정해요. 분류(category)는 RAG 서비스 쪽부터 썼고, 여기에 <b>지금 유효한 판</b>인지를 더해요. 같은 안내가 해마다 새로 나오면 옛 공지와 새 공지가 거의 같은 문장이라, 점수로는 둘을 가리지 못하거든요.</p>"
           + K.code_table([("search_now(query, category, k)", "분류 안에서 current가 거짓인 조각을 빼고 k개"), ("CHUNKS[h[\"citation_id\"] - 1].get(\"current\", True)", "조각의 판 표시. 표시가 없는 옛 조각은 유효하다고 봐요"),
                           ("order3[:k]", "전체 순서에서 위 k개 = 후보 수가 k일 때의 후보")]), "s11"),
    K.step(12, "여러 질문으로 찾기: 용어 넓히기와 합치기", "사람들이 쓰는 말을 안내문이 쓰는 말로 바꾼 질문을 더 만들어 각각 찾고, 같은 조각은 하나로 합쳐요.",
           "<p>'상호대차'는 도서관 업무 용어인데, 안내문은 '다른 도서관 책 받아 보기'라고 써요. 검색 모델이 둘을 같은 뜻으로 묶지 못하면 정답이 후보에도 안 와요(⑨에서 11위). 질문에 다른 표현을 보태는 방법이 셋 있어요.</p>"
           + K.table(["방법", "하는 일", "검색 횟수"], [["다시 쓰기(⑩)", "빠진 대상을 채운 질문 하나", "1번"], ["넓히기 (Query Expansion)", "같은 뜻의 다른 말을 보태거나 바꾼 질문", "1번 또는 여러 번"],
                                                 ["여러 질문 (Multi-query)", "관점이나 표현이 다른 질문 여러 개로 각각 찾고 합치기", "질문 수만큼"]])
           + "<p>이 예제는 도서관 <b>용어집</b>으로 넓혀요. 사람들이 쓰는 말 → 안내문이 쓰는 말의 짝을 미리 적어 두고 바꿔요. LLM에게 여러 질문을 지어내게 할 수도 있지만, 뜻이 바뀐 질문이 섞이면 엉뚱한 후보만 늘어요. 규칙은 넷이에요.</p>"
           "<ul><li>원래 질문을 맨 앞에 둬요. 넓힌 질문이 틀려도 원래 질문의 후보는 남아요.</li><li>같은 질문은 한 번만, 개수에 상한(<code>limit</code>)을 둬요. 질문마다 검색 비용이 들어요.</li>"
           "<li>모든 질문에 <b>같은 조건</b>(분류, 판)을 붙여요. 질문마다 범위가 다르면 볼 수 없어야 할 조각이 섞여요.</li><li>합칠 때는 chunk_id로 한 번만 남기고, 어느 질문이 찾았는지(found_by)를 기록해요.</li></ul>"
           + K.code_table([("query.replace(term, alt)", "용어집의 말로 바꾼 질문"), ("[original, *variants]", "원래 질문을 맨 앞에 두고 나머지를 이어 붙인 목록"),
                           ("merged.setdefault(h[\"chunk_id\"], h)", "처음 본 조각만 넣고, 이미 있으면 그대로 두기"), ("found_by.setdefault(key, []).append(q)", "조각마다 찾은 질문 목록")]), "t12"),
    K.step(13, "다시 순위 매기기: Cross-Encoder", "후보마다 질문과 본문을 한 쌍으로 모델에 함께 넣어 관련 점수를 받고, 점수 순서로 다시 세워요(Reranking).",
           "<p>지금까지의 검색 모델 e5는 <b>Bi-Encoder</b>예요. 질문과 조각을 따로따로 벡터로 바꾸고 벡터끼리 비교해요. 조각 벡터는 미리 만들어 DB에 둘 수 있어서 수만 개에서도 빨라요. 대신 질문과 조각은 서로를 모른 채 각자 요약돼요.</p>"
           "<p><b>Cross-Encoder</b>는 질문과 조각을 <b>한 입력으로 이어 붙여</b> 같이 읽고 점수 하나를 내요. 두 글의 단어가 서로를 보며 계산되니 더 정밀해요. 대신 질문이 올 때마다 후보 하나당 모델을 한 번씩 돌려야 해서, 전체 조각이 아니라 <b>좁힌 후보에만</b> 써요.</p>"
           + K.table(["", "Bi-Encoder (e5, 검색)", "Cross-Encoder (재정렬)"], [["입력", "질문 하나, 조각 하나를 따로", "(질문, 조각) 한 쌍을 같이"], ["출력", "벡터 둘 → 유사도", "점수 하나"],
                                                                      ["미리 계산", "조각 벡터를 DB에 저장", "못 해요. 질문마다 새로"], ["쓰는 곳", "전체에서 후보 찾기", "후보 안에서 순서 정하기"]])
           + f"<p>이 예제는 한국어를 지원하는 <code>BAAI/bge-reranker-v2-m3</code>를 이 PC의 GPU({GPU})에서 돌려요. GPU가 없으면 조용히 CPU로 바꾸지 않고 멈춰요. 같은 코드라도 CPU에서는 훨씬 느려서, 모르고 바뀌면 응답 시간이 갑자기 늘어나요. 이 쪽 전체를 돌리는 동안 GPU 메모리는 최대 {VRAM:.1f}GB를 썼어요.</p>"
           + K.out("t13")
           + K.code_table([("CrossEncoder(\"BAAI/bge-reranker-v2-m3\", device=\"cuda:0\")", "재정렬 모델을 GPU에 올리기. 아직 점수는 안 냈어요"),
                           ("reranker.predict([(query, text), ...], batch_size=16)", "짝마다 점수. 한 번에 16짝씩 GPU로 (모자라면 줄여요)"),
                           ("activation_fn=torch.nn.Identity()", "0~1로 줄이지 않은 원래 값(logit)을 받기"),
                           ("math.isfinite(float(s))", "점수가 무한대나 NaN이 아닌지"), ("sorted(zip(scores, docs), key=..., reverse=True)", "점수와 조각을 짝지어 높은 순서로 (조각의 metadata는 그대로)")]), "s13"),
    K.step(14, "Runnable로 잇기: 단계마다 이름을 붙여 남기기", "다시 쓰기, 검색, 재정렬, 근거 고르기를 state dict를 주고받는 단계로 만들어 | 로 잇고, 단계마다 결과를 다른 이름으로 남겨요.",
           "<p>단계마다 dict 하나(<b>state</b>)를 받아, 앞의 내용은 그대로 두고 자기 결과만 <b>새 키</b>로 더해 넘겨요. 각 함수를 <code>RunnableLambda</code>로 감싸 <code>|</code>로 이으면 하나의 파이프라인이 돼요(Runnable은 <a href=\"rag-service.html\">RAG 서비스</a> ②).</p>"
           + K.table(["단계", "읽는 키", "더하는 키"], [["다시 쓰기", "history, question", "status, search_query"], ["검색", "search_query, category, candidate_k", "search_queries, candidate_docs, found_by"],
                                                 ["재정렬", "search_query, candidate_docs", "ranked_docs"], ["근거 고르기", "ranked_docs, context_k", "context, included_docs"]])
           + "<p>같은 <code>documents</code> 하나를 계속 덮어쓰면, 결과가 나빠졌을 때 어느 단계가 바꿨는지 알 수 없어요. 후보(candidate_docs)와 재정렬 결과(ranked_docs)를 따로 남겨야 ⑮처럼 비교하고, ⑯처럼 기록할 수 있어요.</p>"
           + K.code_table([("RunnableLambda(rewrite_step) | RunnableLambda(retrieve_step) | ...", "왼쪽 단계의 출력 dict가 오른쪽 단계의 입력이 되게 잇기. 아직 실행 안 됨"),
                           ("search_pipeline.invoke(state)", "이 줄에서 네 단계가 차례로 실행돼요"), ("{**state, \"ranked_docs\": ...}", "덮어쓰지 않고 새 키로 더하기")]), "s14"),
    K.step(15, "고치기 전후 비교: 하나씩 켜 보기와 걸린 시간", f"같은 질문 {len(EVAL)}개, 같은 조건에서 다시 쓰기, 넓히기, 재정렬을 하나씩 켜 보며 정답 순위를 비교하고, 단계마다 걸린 시간을 재요.",
           "<p>비교할 때 지킬 것이 셋 있어요. 질문과 정답 조각은 결과를 보기 전에 정해 두고 바꾸지 않아요. 조각, 분류, 후보 수, 넣을 수는 모든 설정에서 같게 둬요. 그리고 <b>한 번에 하나씩</b> 켜요.</p>"
           + K.out("t15")
           + "<p>좋아진 만큼 느려지기도 해요. 단계마다 시간을 따로 재 둬야 어디를 줄일지 보여요.</p>"
           + K.code_table([("SETTINGS", "(다시 쓰기, 넓히기, 재정렬) 켜고 끄기 조합"), ("torch.cuda.synchronize()", "GPU에 걸어 둔 계산이 다 끝날 때까지 기다리기"),
                           ("time.perf_counter()", "시간 재기용 정밀한 시계 (초)"), ("statistics.median", "여러 번 잰 값의 가운데 값 (튀는 값에 덜 흔들려요, 페이지 빌드에서 계산)")]), "t15b"),
    K.step(16, "서비스와 API로 열기", "단계를 서비스 함수 하나로 묶고, 되묻기, 정보 부족, 단계별 실패를 다른 응답으로 나눠 API로 열어요.",
           "<p><code>make_enhanced_service</code>는 RAG 서비스 쪽 ⑦의 흐름에 다시 쓰기와 재정렬을 끼운 거예요. 결과는 넷 중 하나예요.</p>"
           + K.table(["결과", "언제", "HTTP"], [["answered", "근거를 넣었고 검사를 통과한 답", "200"], ["insufficient_context", "근거가 없거나, 넣은 근거에 답이 없음", "200"],
                                             ["needs_clarification", "무엇을 가리키는지 몰라 되물음. 검색도 하지 않아요", "200"],
                                             ["rewrite_error, retrieval_error, rerank_error, generation_error", "그 단계가 실패. 정보 부족으로 바꾸지 않아요", "502"]])
           + "<p>요청 모델에는 앞선 질문(최대 6개), 후보 수(1~20), 넣을 수(1~3)와 함께 단계를 켜고 끄는 값(<code>use_rewrite</code>, <code>use_expand</code>, <code>use_rerank</code>)을 둬요. ⑮처럼 비교하거나, 재정렬 모델을 점검하는 동안 끌 때 써요.</p>"
           + K.out("t16")
           + K.code_table([("history: list[str] = Field(default_factory=list, max_length=6)", "앞선 질문은 없어도 되고, 최대 6개"),
                           ("make_enhanced_service(rerank_fn=broken_reranker)", "재정렬 함수만 바꿔 끼워 고장을 흉내 내기"),
                           ("generate_from_context(state[\"question\"], ...)", "답은 검색용 질문이 아니라 원래 질문으로"),
                           ("HTTPException(status_code=502, detail=str(exc))", "어느 단계가 실패했는지만 알려요")]), "s16"),
]
practice = [("무엇부터 고칠지", "기록(⑯)의 search_query, 후보, 넣은 조각을 보고 ⑨의 세 갈래(통과, 순위 밀림, 후보 밖)로 나눠요. 후보 밖이면 재정렬 모델을 바꿔도 소용없어요."),
            ("다시 쓰기의 비용", "질문마다 LLM 호출이 하나 늘어요. 앞선 질문이 없고 '그거', '그럼' 같은 말도 없으면 건너뛰는 정책을 쓰고, 건너뛴 사실을 기록해요. 다시 쓰기에는 작고 빠른 모델로 충분한 경우가 많아요."),
            ("다시 쓰기 검사의 한계", "숫자와 때를 가리키는 말은 코드로 잡지만, 없는 조건을 붙이거나 부정을 뒤집는 건 못 잡아요. 표본을 사람이 보거나 <a href=\"llm-judge.html\">LLM 채점</a>으로 재요."),
            ("용어집 관리", "넓히기는 업무 용어집(약어, 옛 이름, 제품 코드 → 정식 이름)이 가장 안전해요. 검색 기록에서 '후보 밖'이 난 질문의 말을 모아 용어집을 늘려요. 질문 수에는 상한을 둬요."),
            ("재정렬 모델 고르기", "한국어를 지원하는 모델인지 보고, 우리 문서와 질문으로 직접 재요. 점수가 낮은 1위(⑬의 상호대차)를 거를 기준값을 둘지도 평가 질문으로 정해요."),
            ("GPU 메모리와 시간", f"후보 수와 batch_size가 시간과 메모리를 정해요. 메모리가 모자라면(out of memory) 둘을 줄이거나 조각 길이를 줄여요. 이 PC에서는 이 쪽 전체에 최대 {VRAM:.1f}GB를 썼어요."),
            ("GPU가 없을 때", "CPU로 조용히 바꾸지 않아요. 응답 시간이 크게 바뀌니, 바꾼다면 정책으로 정하고 기록해요. 재정렬을 끄고 검색 순서로 답하는 것도 마찬가지예요."),
            ("판과 권한", "최신 여부와 열람 권한은 점수로 고를 수 없어요(⑪, ⑬). metadata 조건으로 검색 전에 거르고, 여러 질문으로 찾을 때도 모든 검색에 같은 조건을 붙여요."),
            ("좋아졌는지 재기", "질문을 많이 모아 정답 조각을 미리 정하고, 정답 등수를 숫자 하나로 모으는 지표(Recall@k, MRR 등)로 설정끼리 비교해요. 이 쪽의 6개는 성질을 보이려고 고른 예예요.")]
libs = [("⑩", "<code>ChatPromptTemplate</code>, <code>with_structured_output</code>, <code>ChatOpenAI</code>", "검색용 질문 받기 (langchain-core, langchain-openai)"),
        ("⑩", "<code>re.findall</code>", "숫자 덩어리 찾기 (표준 라이브러리)"),
        ("⑬", "<code>CrossEncoder</code>, <code>predict</code>", "질문과 조각 짝의 점수 (sentence-transformers)"),
        ("⑬", "<code>torch.cuda.is_available</code>, <code>torch.nn.Identity</code>", "GPU 확인, 원래 값(logit) 받기 (PyTorch)"),
        ("⑭", "<code>RunnableLambda</code>, <code>|</code>, <code>invoke</code>", "단계 잇기 (langchain-core)"),
        ("⑮", "<code>torch.cuda.synchronize</code>, <code>time.perf_counter</code>", "GPU 계산을 기다렸다 시간 재기"),
        ("⑯", "<code>Field(max_length=...)</code>, <code>HTTPException</code>, <code>TestClient</code>", "요청 범위, 실패 응답, API 시험 (Pydantic, FastAPI)")]
K.build(ROOT / "topics" / "backend" / "rag-rerank.html", title="RAG 검색 고치기", h1="RAG 검색 고치기: 질문 다시 쓰기와 재정렬", track="백엔드", crumb="백엔드 &gt; 검색과 RAG",
        one_line="RAG가 틀리는 원인은 모델보다 <b>검색</b>인 경우가 많아요. 대화 속 질문은 검색 전에 <b>다시 쓰고</b>, 안내문과 말이 다르면 <b>여러 질문으로</b> 찾고, 찾은 후보는 질문과 함께 읽는 모델로 <b>다시 순위를 매겨</b> 정답을 앞으로 올려요. 순위를 바꾸는 단계는 후보 안에서만 고를 수 있어서, 후보에 없는 정답은 앞 단계에서 찾아와야 해요.",
        why=why, whole=whole, steps=steps, code_file="examples/rag/rag_service.py", practice=practice, libs=libs,
        libs_note="<code>cid</code>, <code>rank_of</code>, <code>diagnose</code>, <code>fake_rewriter</code>, <code>check_rewrite</code>, <code>rewrite_step</code>, <code>search_now</code>, <code>expand</code>, <code>prepare_queries</code>, <code>multi_search</code>, <code>rerank</code>, <code>retrieve_step</code>, <code>rerank_step</code>, <code>context_step</code>, <code>run_search</code>, <code>timed</code>, <code>make_enhanced_service</code>, <code>create_enhanced_app</code>은 이 파일 안에서 정의한 함수예요. ①~⑧의 함수는 <a href=\"rag-service.html\">RAG 서비스</a> 쪽에 있어요.",
        next_rows=[("<a href=\"rag-service.html\">RAG 서비스</a>", "이 쪽이 이어 키우는 ①~⑧ (검색, 근거 고르기, 생성과 검사)"), ("<a href=\"embedding.html\">Embedding과 Vector Search</a>", "Bi-Encoder로 찾기"),
                   ("<a href=\"indexing.html\">문서 적재 3. 인덱싱</a>", "판 번호와 metadata"), ("<a href=\"llm-judge.html\">LLM으로 답 채점하기</a>", "다시 쓰기와 답을 따로 재기"),
                   ("<a href=\"prompt-contract.html\">Prompt와 출력 계약</a>", "구조화 출력")],
        sources=["다루는 범위: Day27 (2026-11-24) LangChain RAG, Query Rewrite, Expansion, Multi-query, Candidate Retrieval, Cross-Encoder Reranking, Before/After 비교와 Latency, 서비스 통합. 회의실 예약 예시 대신 이 노트의 가람 도서관 스터디룸으로 같은 흐름을 다시 만들었어요",
                 "예제: <code>examples/rag/rag_service.py</code> ⑨~⑯. 더한 조각은 <code>examples/rag/data/more_chunks.json</code>(<code>tools/make_rag_more.py</code>로 만든 가상 안내문)",
                 f"결과 값: <code>tools/pages/rag_rerank.py</code>가 단계별로 실제 실행한 결과 ({ENV}). 가짜 서버의 다시 쓰기와 답 문장은 미리 정한 값이고, 검색 순위, 재정렬 점수, 시간, HTTP 결과는 실행 값이에요. 시간은 실행할 때마다 조금씩 달라요",
                 "공식 문서: <a href=\"https://sbert.net/docs/cross_encoder/usage/usage.html\" target=\"_blank\">Sentence Transformers Cross-Encoder</a>, <a href=\"https://huggingface.co/BAAI/bge-reranker-v2-m3\" target=\"_blank\">bge-reranker-v2-m3</a>, <a href=\"https://docs.langchain.com/oss/python/langchain/retrieval\" target=\"_blank\">LangChain Retrieval</a>, <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html\" target=\"_blank\">torch.cuda.synchronize</a>"])
print("ok")
