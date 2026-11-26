"""백엔드 > 검색과 RAG > RAG 검색 고치기. examples/rag_rerank/search_fix.py 하나가 ①~⑦로 자란다 (다른 쪽 코드에 기대지 않음).
검색 대상은 가람 도서관 조각 26개(인덱싱 쪽 12개 + tools/make_rag_more.py 로 만든 14개). 검색은 e5-small, 재정렬은
BAAI/bge-reranker-v2-m3 를 CUDA GPU에서 실제로 돌린다. 질문 다시 쓰기는 미리 정한 답을 돌려주는 함수(fake_llm)로 대신한다.
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/rag_rerank.py
"""
import html
import statistics
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
EX = ROOT / "examples" / "rag_rerank"
code = S.parse((EX / "search_fix.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, EX.resolve(), filename="search_fix.py")
torch = ns["torch"]
V = {p: version(p) for p in ("torch", "sentence-transformers", "transformers")}
GPU = torch.cuda.get_device_name(0)
VRAM = torch.cuda.max_memory_allocated() / 2**30
ENV = f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items()) + f", {GPU}, intfloat/multilingual-e5-small, BAAI/bge-reranker-v2-m3"
E, J = html.escape, K.josa
outs = {}

C, T = ns["CHUNKS"], ns["TESTS"]
CK, XK = ns["CANDIDATES"], ns["CONTEXT"]
rank_of = ns["rank_of"]


def title(i):
    return C[i - 1]["title"]


def lab(i):
    return f"[{i}] {E(title(i))}"


def rk(r):
    return f"{r}위" if r else "<b>후보 밖</b>"


# ---------------- 왜
before, diag = ns["before"], ns["diagnosis"]
ranks0 = [rank_of(g, hits) for hits, (_, _, _, g) in zip(before, T)]
top0 = [h["id"] for h in before[0][:XK]]
S.claim(len(C) == 26, "조각 26개")
S.claim(ranks0[0] > XK and title(top0[0]) == "도서 예약 취소" and title(top0[1]) == "프로그램 신청 취소", "왜: 그럼 취소는? 위 둘이 다른 취소")
why_chunks = K.table(["번호", "분류", "제목", "본문"], [[c["id"], c["category"], E(c["title"]), E(c["body"]).replace("\n\n", " / ").replace("\n", " ")] for c in C])
why_tests = K.table(["앞선 질문", "질문", "분류", "정답 조각"], [[E(h) or "(없음)", E(q), cat, lab(g)] for h, q, cat, g in T])
why = f"""<p>RAG는 질문과 관련된 글 조각을 <b>찾아서</b> 모델에게 주고, 그 조각을 근거로 답하게 하는 방식이에요(처음부터 끝까지의 흐름은 <a href="rag-service.html">RAG 서비스</a>). 모델은 받은 조각만 보고 답하니, 정답이 든 조각을 못 찾으면 모델이 아무리 좋아도 답할 수 없어요. 이 쪽은 그중 <b>찾기</b>만 다뤄요.</p>
<p>보통은 후보를 넉넉히 찾은 뒤 위에서 몇 개만 모델에게 줘요. 이 쪽의 예시 설정은 후보 {CK}개를 찾아 위 {XK}개를 주는 거예요. 그런데 실제 질문은 찾기 좋은 모양이 아니에요.</p>
<ul><li>대화 중에는 \"그럼 취소는?\"처럼 <b>무엇의</b> 취소인지가 앞선 질문에만 있어요.</li>
<li>사람들은 \"상호대차\"처럼 안내문과 <b>다른 말</b>을 써요.</li>
<li>빨리 찾는 검색 모델은 <b>순서</b>가 조금씩 어긋나요.</li></ul>
<p>예제는 가상의 '가람 도서관' 안내문을 조각 {len(C)}개로 나눈 거예요. 분류(이용, 공지, 규정, 시설)는 파일마다 붙였고, 2025년 공지는 2026년 공지가 나와서 <b>지금 유효한 판이 아니라고</b> 표시해 뒀어요(④에서 써요).</p>
{why_chunks}
<p>고치기 전에 질문 {len(T)}개와 각각의 정답 조각을 먼저 정했어요. 결과를 본 뒤에 정답을 바꾸면 비교가 의미 없어지니까요.</p>
{why_tests}
<p>첫 질문을 그대로 찾으면 위 {XK}개는 {lab(top0[0])}, {lab(top0[1])}, {lab(top0[2])}{J(title(top0[2]), '이에요', '예요')}. '취소'라는 말만 보고 다른 취소를 가져왔고, 정답인 {lab(T[0][3])}는 {ranks0[0]}위라 모델에게 가지 않아요.</p>
<p>고치는 곳은 실패한 자리에 따라 달라요.</p>
{K.table(["실패한 자리", "보이는 모습", "고치는 곳"], [["검색 전: 질문", "앞선 대화가 있어야 뜻이 통하는 질문", "질문 다시 쓰기(③)"], ["검색 범위", "후보 수가 모자라거나, 옛 판이 섞임", "후보 수와 조건(④)"],
                                         ["검색 전: 표현", "안내문과 다른 말이라 정답이 후보에도 안 옴", "용어 넓히기와 여러 질문(⑤)"], ["검색 뒤: 순서", "정답이 후보에는 있는데 아래에 있음", "다시 순위 매기기(⑥)"]])}
<p>질문이 잘못되면 <b>찾기 전에</b>, 순서가 잘못되면 <b>찾은 뒤에</b> 고쳐요. 순서를 바꾸는 단계는 후보 안에서만 고를 수 있어서, 후보에 없는 정답은 앞 단계에서 찾아와야 해요.</p>
<p>코드는 파일 하나(<code>examples/rag_rerank/search_fix.py</code>)가 이 쪽에서 처음부터 끝까지 자라요. 검색 모델 e5와 재정렬 모델은 이 PC에서 실제로 돌리고, 질문 다시 쓰기에 쓸 LLM만 미리 정한 답을 돌려주는 함수로 대신해요.</p>
"""

# ---------------- ①
first = ns["first"]
S.claim(title(first[0]["id"]) == "대출과 반납", "① 1위가 대출과 반납")
n_use = sum(1 for c in C if c["category"] == "이용")
outs["s1"] = S.table("① search(\"대출 기간은 며칠인가요?\", \"이용\", k=3)", ["순위", "조각", "유사도", "본문"],
                     [[n + 1, lab(h["id"]), f"{h['score']:.3f}", E(h["body"][:44]) + "…"] for n, h in enumerate(first)],
                     [f"분류 '이용'의 조각 {n_use}개 중 질문과 가까운 순서로 3개예요. 1위가 대출 기간이 적힌 조각이에요.",
                      "유사도는 두 벡터의 코사인 값이에요. 1에 가까울수록 뜻이 가깝다고 봐요. 다만 이 모델은 관련 없는 글끼리도 0.8 안팎이 나와서, 값 자체보다 <b>순서</b>를 봐요."])

# ---------------- ②
DIAG = ["순위 밀림", "통과", "후보 밖", "순위 밀림", "통과", "통과"]
S.claim(diag == DIAG, "② 진단 순서")
S.claim(title(before[1][0]["id"]) == "스터디룸 예약" and ranks0[1] == 2, "② 노트북 질문 1위는 스터디룸 예약")
outs["t2"] = S.table(f"② 질문을 그대로 검색 (후보 {CK}개, 모델에게 줄 {XK}개)", ["질문", "정답 조각", "정답 순위", f"위 {XK}개", "진단"],
                     [[E(q), lab(g), rk(r if r and r <= CK else None), "<br>".join(lab(h["id"]) for h in hits[:XK]), f"<b>{d}</b>"]
                      for (_, q, _, g), hits, r, d in zip(T, before, ranks0, diag)],
                     [f"'통과' {DIAG.count('통과')}개, '순위 밀림' {DIAG.count('순위 밀림')}개, '후보 밖' {DIAG.count('후보 밖')}개예요. 사용자에게는 똑같이 '답이 틀림'으로 보여도, 순위 밀림과 후보 밖은 고칠 곳이 달라요.",
                      f"노트북 질문은 정답이 {ranks0[1]}위라 '통과'지만 1위는 {lab(before[1][0]['id'])}이에요. 둘 다 '몇 시간까지'가 들어 있어서, 모델이 스터디룸의 시간을 노트북 답으로 쓸 수도 있어요. 순위 하나로는 다 안 보이는 실패예요.",
                      f"상호대차 질문은 정답이 {ranks0[2]}위라 후보 {CK}개에도 못 들어와요. 이건 순서를 바꾸는 단계(⑥)로는 못 고쳐요."])

# ---------------- ③
rw, af, yq, chk = ns["rewritten"], ns["after"], ns["year_query"], ns["checks"]
ar = [rank_of(t[3], a) for t, a in zip(T, af)]
S.claim(ar == [1, 1] and ns["unclear"] is None and yq is not None, "③ 다시 쓰면 1위, 그거는 되묻기")
cv = list(chk.values())
S.claim(cv[0] == [] and len(cv[1]) == 2 and len(cv[2]) == 1, "③ 검사 0, 2, 1개")
outs["p3"] = S.box("③ 가짜 LLM에게 보내는 Prompt (첫 질문)", [E(x) for x in ns["prompt_example"].split("\n")])
outs["s3"] = S.box("③ rewrite(앞선 질문, 질문)", [
    (f"\"{E(T[0][1])}\" (앞선 질문 \"{E(T[0][0])}\") → \"{E(rw[0])}\": 정답 {ranks0[0]}위 → {ar[0]}위", 1, "newf"),
    (f"\"{E(T[1][1])}\" (앞선 질문 \"{E(T[1][0])}\") → \"{E(rw[1])}\": 정답 {ranks0[1]}위 → {ar[1]}위", 2, "newf"),
    (f"\"올해도 똑같아?\" (앞선 질문 \"스터디룸은 하루에 몇 시간까지 쓸 수 있어?\") → \"{E(yq)}\"", 3),
    ("\"그거 바꿀 수 있어?\" (앞선 질문 없음) → None (되묻기)", 4, "bad")],
    ["앞선 질문의 '스터디룸 예약'을 채웠더니 다른 취소들을 제치고 정답이 1위가 됐어요.",
     "노트북 질문도 '노트북'이 들어가자 스터디룸 조각을 제쳤어요.",
     "'올해'는 그대로 뒀어요. 올해가 몇 년인지는 대화에 없으니 채우지 않아요(④에서 판 조건으로 풀어요).",
     "'그거'가 무엇인지 정할 수 없어요. 상상으로 채우지 않고 None을 돌려줘요. 이때는 검색하지 않고 사용자에게 되물어요."])
outs["t3"] = S.table("③ check_rewrite(앞선 질문, \"올해도 똑같아?\", 다시 쓴 질문)", ["다시 쓴 질문", "찾은 문제", "결과"],
                     [[E(t), E(", ".join(p)) or "없음", "<b>되묻기</b>" if p else "통과"] for t, p in chk.items()],
                     ["대화에 있는 말만 썼고 '올해'도 남겼어요. 통과예요.",
                      "'올해'를 '2025년'으로 바꿨어요. 대화에 없던 숫자가 생기고 '올해'가 빠졌어요. 기간이 바뀌면 다른 공지를 찾게 돼요.",
                      "'올해'를 빼면 연도와 상관없는 질문이 돼요.",
                      "이 검사는 숫자와 때를 가리키는 말만 봐요. '수수료 없이'처럼 없는 조건을 붙이거나 '안 된다'를 '된다'로 뒤집는 건 못 잡아요. 그런 건 표본을 사람이 보거나 <a href=\"llm-judge.html\">LLM 채점</a>으로 따로 재요."])

# ---------------- ④
ks, by, no = ns["k_sweep"], ns["both_years"], ns["now_only"]
g3 = ns["gold3"]
c25 = next(c["id"] for c in C if c["file"] == "notice_2025.txt")
c26 = next(c["id"] for c in C if c["file"] == "notice_2026.txt")
S.claim(ks == {3: False, 5: False, 10: False, 20: True}, "④ 후보 20개에서만 정답")
S.claim([h["id"] for h in by] == [c26, c25] and abs(by[0]["score"] - by[1]["score"]) < 0.01, "④ 두 해 공지가 거의 같은 점수")
S.claim(c25 not in [h["id"] for h in no] and no[0]["id"] == c26, "④ 지금 판만이면 2025 빠짐")
outs["t4"] = S.table(f"④ \"{E(ns['q3'])}\": 후보 수를 바꾸면", ["후보 수", "정답이 후보에?", "⑥의 재정렬이 정답을 고를 수 있나"],
                     [[k, "있음" if v else "없음", "있음" if v else "<b>없음</b>"] for k, v in ks.items()],
                     [f"정답은 '이용' 조각 {n_use}개 중 {ranks0[2]}위예요. 후보를 20개로 늘리면 들어오지만, '이용' 조각이 {n_use}개뿐이라 전부 가져온 거예요. 조각이 수만 개라면 후보를 늘려서 풀기 어려워요.",
                      "후보를 늘리면 재정렬할 짝이 늘어 시간이 늘어요(⑦). 그래서 후보 수만 키우지 말고, 정답이 후보에 들어오게 질문을 고쳐요(⑤)."])
outs["s4"] = S.box(f"④ \"{E(yq)}\" (분류 '공지', 위 2개)",
                   [("판을 가리지 않으면 search", 0), *[(f"  {lab(h['id'])} 유사도 {h['score']:.3f}", 1) for h in by],
                    ("지금 유효한 판만 search_now", 0), *[(f"  {lab(h['id'])} 유사도 {h['score']:.3f}", 2, "newf") for h in no]],
                   [f"두 해의 공지가 유사도 {by[0]['score'] - by[1]['score']:.3f} 차이로 붙어 있어요. 문장 모양이 거의 같아서, 점수로는 어느 해가 지금인지 가릴 수 없어요. 둘 다 모델에게 가면 '2시간'과 '3시간'이 같이 들어가요.",
                    "판 표시(current)로 거르면 2025년 공지가 빠지고 다른 공지가 그 자리에 와요. 이런 조건은 질문을 어떻게 바꾸든 <b>모든 검색에 똑같이</b> 붙여요(⑤의 multi_search도 search_now를 써요)."])

# ---------------- ⑤
pq, u3, f3 = ns["per_query"], ns["union3"], ns["found3"]
qs3 = ns["queries3"]
S.claim(len(qs3) == 2 and g3 not in [h["id"] for h in pq[qs3[0]]] and pq[qs3[1]][0]["id"] == g3, "⑤ 바꾼 질문에서 정답 1위")
u_ids = [h["id"] for h in u3]
pos = u_ids.index(g3) + 1
S.claim(pos > XK, "⑤ 합친 목록에서도 정답이 위 3개 밖")
both_found = sum(1 for v in f3.values() if len(v) > 1)
outs["t5"] = S.table("⑤ 질문마다 위 5개 (search_now)", ["질문", "찾은 조각"],
                     [[E(q), ", ".join(("<b>" + lab(h["id"]) + "</b>") if h["id"] == g3 else lab(h["id"]) for h in hits)] for q, hits in pq.items()],
                     ["원래 질문으로는 정답이 위 5개에 없어요. 용어집으로 '상호대차'를 안내문의 말로 바꾼 질문에서는 정답이 1위예요.",
                      f"둘을 합치면 {len(u_ids)}개예요(두 질문이 함께 찾은 조각 {both_found}개). 같은 조각은 한 번만 남기고, 어느 질문이 찾았는지(found_by)를 기록해 둬요.",
                      f"합친 순서는 '먼저 찾은 순서'라 정답은 {pos}번째예요. 모델에게 줄 {XK}개 안에는 아직 못 들어요. 후보에 넣는 데까지가 여러 질문의 일이고, 순서는 ⑥에서 정해요."])

# ---------------- ⑥
r4, c4, lg = ns["ranked4"], ns["cands4"], ns["logits4"]
g4 = ns["gold4"]
e5_ids = [h["id"] for h in c4]
ce_ids = [h["id"] for h in r4]
g4_sc = r4[0]["rerank"]
lg_g4 = f"{float(lg[e5_ids.index(g4)]):.2f}"
S.claim(e5_ids.index(g4) + 1 == ranks0[3] and ce_ids[0] == g4, "⑥ 노쇼 e5 5위 → 1위")
S.claim(r4[1]["rerank"] < 0.2 < g4_sc, "⑥ 1위만 높고 나머지는 훨씬 낮음")
S.claim(abs(1 / (1 + 2.718281828459045 ** -float(lg_g4)) - g4_sc) < 2e-3, "⑥ 점수 = sigmoid(logit)")
rows6 = [[("<b>" + lab(h["id"]) + "</b>") if h["id"] == g4 else lab(h["id"]), f"{e5_ids.index(h['id']) + 1}위", f"{h['score']:.3f}",
          f"{n + 1}위", f"{h['rerank']:.4f}", f"{float(lg[e5_ids.index(h['id'])]):.2f}"] for n, h in enumerate(r4)]
outs["t6"] = S.table(f"⑥ \"{E(ns['q4'])}\": 후보 {len(c4)}개를 재정렬", ["조각", "e5 순위", "e5 유사도", "재정렬 순위", "재정렬 점수", "logit"], rows6,
                     [f"e5에서 {ranks0[3]}위였던 '노쇼'가 1위가 됐어요. e5 유사도로는 후보들이 {min(h['score'] for h in c4):.3f}~{max(h['score'] for h in c4):.3f}에 몰려 있었는데, 재정렬 점수는 1위만 {g4_sc:.3f}이고 나머지는 훨씬 낮아요. 질문과 본문을 같이 읽어서 '안 가면'과 '연락 없이 오지 않으면'이 같은 상황인 걸 알아봤어요.",
                      f"이 모델의 재정렬 점수는 logit(모델이 낸 원래 값)을 0~1로 줄인 거예요(sigmoid). 모델마다 달라서, logit을 그대로 내는 모델도 있어요. 1위의 logit {lg_g4}{J(lg_g4, '이', '가')} {g4_sc:.3f}{J(f'{g4_sc:.3f}', '이', '가')} 됐어요. 확률처럼 보여도 '맞을 확률'이 아니에요. 같은 질문 안에서 순서를 정하는 데만 써요.",
                      "e5 유사도와 재정렬 점수는 다른 모델이 다른 방법으로 낸 값이라 숫자끼리 맞대어 비교하지 않아요."])
r3, yr = ns["ranked3"], ns["year_ranked"]
S.claim(r3[0]["id"] == g3 and r3[0]["rerank"] < 0.01, "⑥ 상호대차 1위지만 점수 아주 낮음")
S.claim([h["id"] for h in yr] == [c26, c25] and yr[0]["rerank"] - yr[1]["rerank"] < 0.01, "⑥ 두 해 공지 모두 높은 점수")
outs["s6"] = S.box("⑥ 다른 두 질문을 재정렬하면", [
    (f"\"{E(ns['q3'])}\" + ⑤에서 합친 후보 {len(u_ids)}개 → 1위 {lab(r3[0]['id'])} 점수 {r3[0]['rerank']:.4f}, 2위 {lab(r3[1]['id'])} {r3[1]['rerank']:.4f}", 1),
    (f"\"{E(yq)}\" + 두 해의 공지 → {lab(yr[0]['id'])} {yr[0]['rerank']:.4f}, {lab(yr[1]['id'])} {yr[1]['rerank']:.4f}", 2, "bad")],
    [f"정답이 후보에 들어오니 1위로 올랐어요. 그런데 점수가 {r3[0]['rerank']:.4f}로 아주 낮아요. 이 모델도 '상호대차'를 잘 몰라서, 다른 후보가 더 낮았을 뿐이에요. 점수가 낮은 1위를 그대로 믿지 말고, 기준 점수 아래면 '근거 부족'으로 다루는 방법도 있어요(기준은 평가 질문으로 정해요).",
     "두 공지 모두 거의 최고점이에요. 더 정밀한 모델도 어느 해가 지금인지는 몰라요. 최신 여부와 열람 권한은 점수가 아니라 ④의 조건으로 정해요."])

# ---------------- ⑦
cmp_ = ns["compare"]
S.claim(cmp_["기준"] == [r if r <= CK else None for r in ranks0], "⑦ 기준 = ②")
S.claim(cmp_["다시 쓰기만"][:2] == [1, 1] and cmp_["다시 쓰기만"][3] == ranks0[3], "⑦ 다시 쓰기는 1, 2번만")
S.claim(cmp_["재정렬만"][3] == 1 and 1 < cmp_["재정렬만"][0] <= XK, "⑦ 재정렬만: 노쇼 1위, 취소는 3위 안")
S.claim(all(v[2] is None for n, v in cmp_.items() if n != "셋 다 (+ 넓히기)") and cmp_["셋 다 (+ 넓히기)"] == [1] * 6, "⑦ 상호대차는 넓히기만")
hit = {n: sum(1 for r in v if r and r <= XK) for n, v in cmp_.items()}


def cell(r):
    if r is None:
        return "<b>후보 밖</b>"
    return f"<b>{r}위</b>" if r <= XK else f"{r}위"


outs["t7"] = S.table(f"⑦ 정답 순위 (진하게: 모델에게 줄 {XK}개 안)", ["설정"] + [E(q) for _, q, _, _ in T] + [f"{XK}개 안"],
                     [[n] + [cell(r) for r in v] + [f"{hit[n]}/{len(T)}"] for n, v in cmp_.items()],
                     ["다시 쓰기는 앞선 질문이 필요한 두 질문을, 재정렬은 '안 가면' 질문을, 넓히기는 상호대차 질문을 고쳐요. 하나씩 켜 봐야 무엇이 무엇을 고쳤는지 보여요.",
                      f"재정렬만 켜도 '그럼 취소는?'의 정답이 {cmp_['재정렬만'][0]}위로 들어오지만, 질문 자체가 뜻이 비어 있어서 다시 쓰기를 켰을 때만 1위가 돼요.",
                      "상호대차 질문은 넓히기 없이는 어떤 설정에서도 후보 밖이에요. 재정렬은 후보 밖 조각을 데려오지 못해요.",
                      "질문 6개는 성질을 보이려고 고른 예예요. 실제로 켤지는 질문을 많이 모아 같은 방법으로 재 보고 정해요."])
med = {n: statistics.median(v) for n, v in ns["times"].items()}
cost = {k: statistics.median(v) for k, v in ns["cost"].items()}
S.claim(med["재정렬"] > med["검색"] > med["다시 쓰기"], "⑦ 재정렬 > 검색 > 가짜 다시 쓰기")
S.claim(cost[17] > 1.3 * max(cost[5], cost[10]), "⑦ 16개를 넘으면 크게 늘어남")
n_q = len(ns["expand"](ns["rewritten"][0]))
S.claim(n_q == 2, "⑦ 넓히기로 질문 둘")
outs["t7b"] = S.table(f"⑦ 걸린 시간 (\"{E(T[0][1])}\", 셋 다 켬, 20번 잰 가운데 값)", ["단계", "시간", "무엇이 드나"],
                      [["다시 쓰기", f"{med['다시 쓰기']:.1f} ms", "가짜 LLM이라 거의 0이에요. 실제로는 네트워크를 거쳐 모델이 글을 만드는 시간이 더해져요"],
                       ["검색", f"{med['검색']:.1f} ms", f"질문 {n_q}개(원래 질문과 '철회'로 바꾼 질문)를 e5로 벡터로 만들고 {len(C)}개와 비교 (GPU)"],
                       ["재정렬", f"{med['재정렬']:.1f} ms", "후보마다 (질문, 본문) 짝을 Cross-Encoder로 (GPU)"]]
                      + [[f"재정렬, 후보 {k}개", f"{v:.1f} ms", ""] for k, v in cost.items()],
                      [f"이 셋 중에는 재정렬이 가장 오래 걸려요. 후보 {max(cost)}개는 {cost[max(cost)]:.0f} ms로, 5개나 10개({cost[5]:.0f} ms, {cost[10]:.0f} ms)보다 확 늘어요. 짝을 한 번에 16개씩(batch_size) 묶어 계산해서, 16개를 넘으면 한 묶음을 더 돌리거든요.",
                       "실제 서비스에서는 다시 쓰기와 답변에 LLM 호출이 들어가서 보통 그쪽이 더 길어요. 그래서 짧고 뜻이 분명한 질문은 다시 쓰기를 건너뛰는 정책을 쓰기도 해요(건너뛴 사실은 기록해요).",
                       f"GPU 작업은 명령만 걸어 두고 바로 다음 줄로 넘어갈 수 있어요. 그래서 재기 전후에 <code>torch.cuda.synchronize()</code>로 끝날 때까지 기다려요. 시간은 실행할 때마다 조금씩 달라요({GPU})."])

flow = [[1, "찾기", "질문과 가까운 조각", "e5 벡터 검색"], [2, "실패 나누기", "정답 순위로", f"통과 {DIAG.count('통과')}/{len(T)}"],
        [3, "다시 쓰기", "빠진 대상 채우기", "모르면 되묻기"], [4, "후보 모으기", "후보 수, 판 조건", "후보 밖은 못 고름"],
        [5, "여러 질문", "용어 넓히고 합치기", "같은 조각은 한 번"], [6, "재정렬", "Cross-Encoder", "질문과 본문을 함께"],
        [7, "비교", "하나씩 켜기, 시간", f"{hit['기준']}/6 → {hit['셋 다 (+ 넓히기)']}/6"]]
S.save(ROOT / "topics" / "backend" / "rag-rerank-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

whole = f"""<p>찾고(①), 실패를 나눈 뒤(②), 검색 전(③), 검색 범위(④⑤), 검색 뒤(⑥)를 하나씩 고치고, 마지막에 하나씩 켜서 비교해요(⑦). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>처음과 끝을 보면, 질문 {len(T)}개 중 정답이 모델에게 줄 {XK}개 안에 든 질문이 {hit['기준']}개에서 {hit['셋 다 (+ 넓히기)']}개가 돼요. 대신 재정렬에 질문마다 약 {med['재정렬']:.0f} ms(이 PC의 GPU)가 더 들고, 다시 쓰기에 LLM 호출이 하나 더 들어요.</p>"""

steps = [
    K.step(1, "도서관 조각을 벡터로 바꾸고 찾기", f"조각 {len(C)}개를 e5로 벡터로 바꿔 두고, 질문이 오면 같은 분류 안에서 가까운 순서로 k개를 돌려줘요.",
           "<p>검색 모델 e5는 글을 384개의 숫자(벡터)로 바꿔요. 뜻이 비슷한 글은 벡터도 비슷한 방향을 가리켜서, 질문 벡터와 조각 벡터의 코사인 유사도로 가까운 순서를 매겨요(자세한 건 <a href=\"embedding.html\">Embedding과 Vector Search</a>). 조각 벡터는 미리 한 번만 만들어 두고, 질문이 올 때마다 질문 벡터만 새로 만들어요.</p>"
           "<p>e5는 질문 앞에 <code>query:</code>, 조각 앞에 <code>passage:</code>를 붙이라고 정해 둔 모델이에요. 둘을 섞으면 순서가 흔들려요.</p>"
           + K.code_table([("SentenceTransformer(\"intfloat/multilingual-e5-small\")", "검색 모델 불러오기 (이 PC에 받아 둔 것)"),
                           ("encoder.encode([...], normalize_embeddings=True)", "글 목록 → 벡터 목록. 길이를 1로 맞춰 두면 내적이 곧 코사인 유사도"),
                           ("vectors @ q", "모든 조각과 질문의 유사도를 한 번에"), ("np.argsort(-scores)", "유사도가 큰 것부터의 번호 순서")]), "s1"),
    K.step(2, "정답을 정해 두고 실패 나누기", f"질문 {len(T)}개를 그대로 검색해서 정답이 몇 위인지 보고, 실패를 세 갈래로 나눠요.",
           "<p>고치기 전에 <b>어디서</b> 실패하는지부터 나눠요. 후보 10개를 찾아 위 3개를 모델에게 주는 예시 설정이라면, 정답의 순위로 세 갈래가 돼요.</p>"
           + K.table(["진단", "정답 순위", "뜻", "고치는 곳"], [["통과", f"1~{XK}위", "모델에게 정답이 가요", "-"], ["순위 밀림", f"{XK + 1}~{CK}위", "찾기는 했는데 모델에게 줄 자리에 못 들어요", "검색 뒤에서 순서 바꾸기(⑥)"],
                                                         ["후보 밖", f"{CK}위 밖", "검색이 못 찾았어요", "검색 전(③⑤)이나 검색 범위(④)"]])
           + K.code_table([("search(q, cat, k=len(CHUNKS))", "분류 안의 전체 순서 (정답 순위를 끝까지 보려고)"), ("rank_of(gold, hits)", "정답이 몇 위인지, 없으면 None"),
                           ("diagnose(rank)", "순위로 통과, 순위 밀림, 후보 밖 나누기")]), "t2"),
    K.step(3, "질문 다시 쓰기: 대화에서 빠진 대상 채우기", "앞선 질문을 보고, 대화를 모르는 사람도 알아듣는 검색용 질문을 따로 만들어요. 원래 질문은 지우지 않아요.",
           "<p>\"그럼 취소는?\"을 그대로 벡터로 바꾸면 '취소'라는 뜻만 남아요. 무엇의 취소인지는 앞선 질문에 있는데, 검색 모델은 질문 하나만 봐요. 그래서 대화를 읽을 수 있는 LLM에게 <b>검색용 질문</b>을 따로 쓰게 해요(Query Rewrite). 이 예제에서는 LLM 자리에 미리 정한 답을 돌려주는 <code>fake_llm</code>을 뒀어요. 보내는 Prompt는 실제와 같아요.</p>"
           + K.out("p3")
           + "<p>규칙은 '채우되 만들지 않기'예요. 앞선 질문에서 확인되는 대상만 채우고, 대화에 없는 날짜, 숫자, 조건은 더하지 않아요. 무엇을 가리키는지 정할 수 없으면 되묻게 해요. 검색용 질문은 검색에만 쓰고, 답은 사용자가 실제로 한 질문에 해요. 그래서 원래 질문을 지우지 않고 둘을 같이 둬요.</p>"
           + K.out("s3")
           + "<p>모델이 규칙을 어길 수도 있으니, 받은 다음 코드로 한 번 더 봐요. <code>check_rewrite</code>는 대화에 없던 숫자가 생겼는지, 원래 질문의 '올해' 같은 말이 빠졌는지 봐요.</p>"
           + K.code_table([("REWRITE_PROMPT.format(history=..., question=...)", "Prompt 틀의 빈칸 채우기 ({{ }}는 글자 { } 그대로)"),
                           ("re.search(r\"앞선 질문: (.*)\\n현재 질문: (.*)$\", prompt)", "가짜 LLM이 Prompt 끝 두 줄에서 앞선 질문과 현재 질문을 꺼내기 (괄호 안이 꺼낼 부분)"),
                           ("json.loads(...)", "모델이 돌려준 JSON 글을 dict로"), ("re.findall(r\"\\d+\", rewritten)", "다시 쓴 질문 속 숫자 덩어리 모두 (\\d+는 숫자가 하나 이상 이어진 것)")]), "t3"),
    K.step(4, "후보 모으기: 몇 개를, 어떤 조건으로", "후보를 몇 개 가져올지 정하고, 모든 검색에 똑같이 붙일 조건(분류, 지금 유효한 판)을 정해요.",
           f"<p>후보 수와 모델에게 줄 수는 달라요. 후보를 넉넉히 가져와야 뒤에서 순서를 바꿀 여지가 생기는데, 순서를 바꾸는 단계(⑥)는 <b>후보 안에서만</b> 골라요. 후보에 없는 조각은 새로 데려오지 못해요.</p>"
           + K.out("t4")
           + "<p>조건도 함께 정해요. 분류에 더해 <b>지금 유효한 판</b>인지를 봐요. 같은 안내가 해마다 새로 나오면 옛 공지와 새 공지가 거의 같은 문장이라, 점수로는 둘을 가리지 못하거든요.</p>"
           + K.code_table([("search_now(query, category, k)", "판 표시(current)가 거짓인 조각을 빼고 k개"), ("rank_of(...) is not None", "정답이 후보에 있나")]), "s4"),
    K.step(5, "여러 질문으로 찾기: 용어 넓히기와 합치기", "사람들이 쓰는 말을 안내문이 쓰는 말로 바꾼 질문을 더 만들어 각각 찾고, 같은 조각은 하나로 합쳐요.",
           "<p>'상호대차'는 도서관 업무 용어인데, 안내문은 '다른 도서관 책 받아 보기'라고 써요. 검색 모델이 둘을 같은 뜻으로 묶지 못해서 정답이 11위였어요. 질문에 다른 표현을 보태는 방법이 셋 있어요.</p>"
           + K.table(["방법", "하는 일", "검색 횟수"], [["다시 쓰기(③)", "빠진 대상을 채운 질문 하나", "1번"], ["넓히기 (Query Expansion)", "같은 뜻의 다른 말을 보태거나 바꾼 질문", "1번 또는 여러 번"],
                                                 ["여러 질문 (Multi-query)", "표현이 다른 질문 여러 개로 각각 찾고 합치기", "질문 수만큼"]])
           + "<p>이 예제는 도서관 <b>용어집</b>으로 넓혀요. '사람들이 쓰는 말 → 안내문이 쓰는 말'의 짝을 적어 두고 바꿔요. LLM에게 여러 질문을 지어내게 할 수도 있지만, 뜻이 바뀐 질문이 섞이면 엉뚱한 후보만 늘어요.</p>"
           "<ul><li>원래 질문을 맨 앞에 둬요. 바꾼 질문이 틀려도 원래 질문의 후보는 남아요.</li><li>모든 질문에 <b>같은 조건</b>(④의 분류, 판)을 붙여요.</li>"
           "<li>같은 조각은 한 번만 남기고, 어느 질문이 찾았는지 기록해요. 질문 수만큼 검색 비용이 드니 개수에 상한을 둬요.</li></ul>"
           + K.code_table([("query.replace(a, b)", "용어집의 말로 바꾼 질문"), ("[query] + [...]", "원래 질문을 맨 앞에 두고 바꾼 질문을 이어 붙인 목록"),
                           ("merged.setdefault(h[\"id\"], h)", "처음 본 조각만 넣고, 이미 있으면 그대로 두기"), ("found_by.setdefault(id, []).append(q)", "조각마다 찾은 질문 목록")]), "t5"),
    K.step(6, "다시 순위 매기기: Cross-Encoder", "후보마다 질문과 본문을 한 쌍으로 모델에 함께 넣어 관련 점수를 받고, 점수 순서로 다시 세워요(Reranking).",
           "<p>①의 e5는 <b>Bi-Encoder</b>예요. 질문과 조각을 따로 벡터로 바꾸고 벡터끼리 비교해요. 조각 벡터를 미리 만들어 둘 수 있어서 수만 개에서도 빨라요. 대신 질문과 조각은 서로를 모른 채 각자 요약돼요.</p>"
           "<p><b>Cross-Encoder</b>는 질문과 조각을 <b>한 입력으로 이어 붙여</b> 같이 읽고 점수 하나를 내요. 두 글의 단어가 서로를 보며 계산되니 더 정밀해요. 대신 질문이 올 때마다 후보 하나당 모델을 한 번씩 돌려야 해서, 전체 조각이 아니라 <b>좁힌 후보에만</b> 써요.</p>"
           + K.table(["", "Bi-Encoder (e5, 찾기)", "Cross-Encoder (재정렬)"], [["입력", "질문 하나, 조각 하나를 따로", "(질문, 조각) 한 쌍을 같이"], ["출력", "벡터 둘 → 유사도", "점수 하나"],
                                                                     ["미리 계산", "조각 벡터를 저장해 둠", "못 해요. 질문마다 새로"], ["쓰는 곳", "전체에서 후보 찾기", "후보 안에서 순서 정하기"]])
           + "<p>Sentence Transformers 문서는 둘을 이렇게 함께 써요(<a href=\"https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html\" target=\"_blank\">Retrieve &amp; Re-Rank</a>). 빠른 Bi-Encoder로 <b>넉넉히</b> 찾고, 그중 위쪽만 느리지만 정밀한 Cross-Encoder로 <b>다시 줄 세워요</b>. 라이브러리에 <code>rank</code>가 있어서, 질문과 본문 목록을 주면 점수를 매겨 높은 순서로 돌려줘요.</p>"
           + f"<p>이 예제는 한국어를 지원하는 <code>BAAI/bge-reranker-v2-m3</code>를 이 PC의 GPU({GPU})에서 돌려요. GPU가 없으면 조용히 CPU로 바꾸지 않고 멈춰요. CPU에서는 훨씬 느려서, 모르고 바뀌면 응답 시간이 갑자기 늘어요. 이 쪽 전체를 돌리는 동안 GPU 메모리는 최대 {VRAM:.1f}GB를 썼어요.</p>"
           + K.out("t6")
           + K.code_table([("CrossEncoder(\"BAAI/bge-reranker-v2-m3\", device=\"cuda:0\")", "재정렬 모델을 GPU에 올리기. 아직 점수는 안 냈어요"),
                           ("reranker.rank(query, [본문, ...], batch_size=16)", "짝마다 점수를 매겨 높은 순서의 [{corpus_id, score}, ...]로. 한 번에 16짝씩 GPU로"),
                           ("hits[r[\"corpus_id\"]]", "corpus_id는 넘긴 목록에서의 위치. 원래 후보(출처 정보 포함)를 다시 찾아요"),
                           ("reranker.predict(pairs, activation_fn=torch.nn.Identity())", "정렬 없이 짝마다 점수만. 여기서는 0~1로 줄이기 전의 원래 값(logit)을 보려고 써요")]), "s6"),
    K.step(7, "하나씩 켜서 비교하기와 걸린 시간", f"같은 질문 {len(T)}개, 같은 조건에서 다시 쓰기, 넓히기, 재정렬을 하나씩 켜 보며 정답 순위를 비교하고, 단계마다 걸린 시간을 재요.",
           "<p>비교할 때 지킬 것이 셋 있어요. 질문과 정답 조각은 결과를 보기 전에 정해 두고 바꾸지 않아요. 조각, 분류, 후보 수, 모델에게 줄 수는 모든 설정에서 같게 둬요. 그리고 <b>한 번에 하나씩</b> 켜요. 한꺼번에 켜고 좋아졌다고만 하면 어느 단계가 필요한지 몰라요.</p>"
           + K.out("t7")
           + "<p>좋아진 만큼 느려지기도 해요. 단계마다 시간을 따로 재 둬야 어디를 줄일지 보여요.</p>"
           + K.code_table([("find(history, q, cat, 다시 쓰기, 넓히기, 재정렬)", "켜고 끄는 값 셋을 받아 순서가 정해진 후보 목록을 돌려줘요"),
                           ("torch.cuda.synchronize()", "GPU에 걸어 둔 계산이 다 끝날 때까지 기다리기"), ("time.perf_counter()", "시간 재기용 정밀한 시계 (초)")]), "t7b"),
]
practice = [("무엇부터 고칠지", "질문, 검색용 질문, 후보, 모델에게 준 조각을 기록해 두고 ②의 세 갈래로 나눠요. 후보 밖이면 재정렬 모델을 바꿔도 소용없어요."),
            ("다시 쓰기의 비용", "질문마다 LLM 호출이 하나 늘어요. 앞선 질문이 없고 '그거', '그럼' 같은 말도 없으면 건너뛰고, 건너뛴 사실을 기록해요. 다시 쓰기에는 작고 빠른 모델로 충분한 경우가 많아요."),
            ("다시 쓰기 검사의 한계", "숫자와 때를 가리키는 말은 코드로 잡지만, 없는 조건을 붙이거나 부정을 뒤집는 건 못 잡아요. 표본을 사람이 보거나 <a href=\"llm-judge.html\">LLM 채점</a>으로 재요."),
            ("용어집 관리", "넓히기는 업무 용어집(약어, 옛 이름, 제품 코드 → 정식 이름)이 가장 안전해요. '후보 밖'이 난 질문의 말을 모아 용어집을 늘려요."),
            ("재정렬 모델 고르기", "한국어를 지원하는지 보고, 우리 문서와 질문으로 직접 재요. 점수가 낮은 1위(⑥의 상호대차)를 거를 기준값을 둘지도 평가 질문으로 정해요."),
            ("GPU 메모리와 시간", f"후보 수와 batch_size가 시간과 메모리를 정해요. 메모리가 모자라면(out of memory) 둘을 줄이거나 조각 길이를 줄여요. 이 PC에서는 이 쪽 전체에 최대 {VRAM:.1f}GB를 썼어요."),
            ("고장 났을 때", "재정렬 모델이나 GPU가 고장 나면 조용히 검색 순서로 답하지 말고 실패로 알려요. 검색 순서로 답하는 게 정해 둔 정책이라면 응답과 기록에 그렇게 표시해요."),
            ("판과 권한", "최신 여부와 열람 권한은 점수로 고를 수 없어요(④, ⑥). 조건으로 검색 전에 거르고, 여러 질문으로 찾을 때도 모든 검색에 같은 조건을 붙여요."),
            ("서비스와 LangChain에 붙일 때", "다시 쓰기, 찾기, 재정렬은 각각 '입력 → 결과' 함수라서, LangChain을 쓴다면 함수마다 <code>RunnableLambda</code>로 감싸 <code>|</code>로 이으면 돼요(<a href=\"rag-service.html\">RAG 서비스</a> ②). 이때 검색용 질문과 원래 질문, 재정렬 전후 목록을 서로 다른 이름으로 남겨야 나중에 비교할 수 있어요."),
            ("좋아졌는지 재기", "질문을 많이 모아 정답 조각을 미리 정하고, 정답 순위를 숫자 하나로 모으는 지표(Recall@k, MRR 등)로 설정끼리 비교해요(<a href=\"rag-eval.html\">RAG 평가</a>).")]
libs = [("①", "<code>SentenceTransformer</code>, <code>encode</code>", "글을 벡터로 (sentence-transformers)"), ("①", "<code>np.argsort</code>", "큰 값부터의 순서 (NumPy)"),
        ("③", "<code>str.format</code>, <code>re.search</code>, <code>re.findall</code>, <code>json.loads</code>", "Prompt 채우기, 글에서 꺼내기, JSON 읽기 (표준 라이브러리)"),
        ("⑥", "<code>CrossEncoder</code>, <code>rank</code>, <code>predict</code>", "(질문, 조각) 짝의 점수와 정렬 (sentence-transformers)"),
        ("⑥", "<code>torch.cuda.is_available</code>, <code>torch.nn.Identity</code>", "GPU 확인, 원래 값(logit) 받기 (PyTorch)"),
        ("⑦", "<code>torch.cuda.synchronize</code>, <code>time.perf_counter</code>", "GPU 계산을 기다렸다 시간 재기")]
K.build(ROOT / "topics" / "backend" / "rag-rerank.html", title="RAG 검색 고치기", h1="RAG 검색 고치기: 질문 다시 쓰기와 재정렬", track="백엔드", crumb="백엔드 &gt; 검색과 RAG",
        one_line="RAG가 틀리는 원인은 모델보다 <b>찾기</b>인 경우가 많아요. 대화 속 질문은 찾기 전에 <b>다시 쓰고</b>, 안내문과 말이 다르면 <b>여러 질문으로</b> 찾고, 찾은 후보는 질문과 함께 읽는 모델로 <b>다시 순위를 매겨</b> 정답을 앞으로 올려요. 순위를 바꾸는 단계는 후보 안에서만 고를 수 있어서, 후보에 없는 정답은 앞 단계에서 찾아와야 해요.",
        why=why, whole=whole, steps=steps, code_file="examples/rag_rerank/search_fix.py", practice=practice, libs=libs,
        libs_note="<code>search</code>, <code>find_id</code>, <code>rank_of</code>, <code>diagnose</code>, <code>fake_llm</code>, <code>check_rewrite</code>, <code>rewrite</code>, <code>search_now</code>, <code>expand</code>, <code>multi_search</code>, <code>rerank</code>, <code>find</code>, <code>timed</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rag-service.html\">RAG 서비스</a>", "찾은 조각으로 출처가 붙은 답을 만드는 전체 흐름"), ("<a href=\"embedding.html\">Embedding과 Vector Search</a>", "①의 벡터 검색 (Bi-Encoder)"),
                   ("<a href=\"rag-eval.html\">RAG 평가</a>", "질문을 많이 모아 숫자로 재기"), ("<a href=\"llm-judge.html\">LLM으로 답 채점하기</a>", "다시 쓰기와 답을 따로 재기"),
                   ("<a href=\"indexing.html\">문서 적재 3. 인덱싱</a>", "판 번호와 metadata")],
        sources=["다루는 범위: Day27 (2026-11-24) LangChain RAG, Query Rewrite, Expansion, Multi-query, Candidate Retrieval, Cross-Encoder Reranking, Before/After 비교와 Latency. 회의실 예약 예시 대신 이 노트의 가람 도서관 스터디룸으로 같은 흐름을 다시 만들었어요. 서비스와 API 연결은 실습 프로젝트의 몫이라 이 쪽에서는 빼고 '실제로 할 때'에 적었어요",
                 "예제: <code>examples/rag_rerank/search_fix.py</code> (이 파일 하나로 실행돼요). 조각은 <code>examples/ingest/out/chunks.json</code>과 <code>examples/rag/data/more_chunks.json</code>(<code>tools/make_rag_more.py</code>로 만든 가상 안내문)",
                 f"결과 값: <code>tools/pages/rag_rerank.py</code>가 단계별로 실제 실행한 결과 ({ENV}). 가짜 LLM의 다시 쓰기는 미리 정한 값이고, 검색 순위, 재정렬 점수, 시간은 실행 값이에요. 시간은 실행할 때마다 조금씩 달라요",
                 "공식 문서: <a href=\"https://sbert.net/docs/cross_encoder/usage/usage.html\" target=\"_blank\">Sentence Transformers Cross-Encoder</a>, <a href=\"https://huggingface.co/BAAI/bge-reranker-v2-m3\" target=\"_blank\">bge-reranker-v2-m3</a>, <a href=\"https://huggingface.co/intfloat/multilingual-e5-small\" target=\"_blank\">multilingual-e5-small</a>, <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html\" target=\"_blank\">torch.cuda.synchronize</a>"])
print("ok")
