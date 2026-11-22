"""백엔드 > 여러 정답으로 채점하기. examples/eval_refs/multi_ref.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/multi_ref.py
"""
import sys
import shutil
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import html  # noqa: E402
import numpy as np  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "eval_refs"
code = S.parse((EX / "multi_ref.py").read_text(encoding="utf-8"))
_tmp = Path(tempfile.mkdtemp(prefix="stepcode_"))                   # ① 이 ../gpt/data 를 읽어서 두 폴더를 나란히 복사
work = _tmp / "eval_refs"
shutil.copytree(EX, work, ignore=shutil.ignore_patterns("__pycache__"))
(_tmp / "gpt" / "data").mkdir(parents=True)
shutil.copy(ROOT / "examples" / "gpt" / "data" / "pages.jsonl", _tmp / "gpt" / "data")
ns = S.run(code, 7, work, filename="multi_ref.py")
ENV = f"Python {sys.version.split()[0]}, NumPy {np.__version__} (seed 0, 모델과 API 없음)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "backend" / "img" / "multiref"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/multiref/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


def mark(a, b):                                                    # 원문과 다른 글자를 굵게
    import difflib
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        seg = html.escape(b[j1:j2]).replace(" ", "&nbsp;") if op != "equal" else html.escape(b[j1:j2])
        out.append(f"<b>{seg}</b>" if op != "equal" and seg else seg)
        if op == "delete":
            out.append("<b>⌃</b>")
    return "".join(out)


outs = {}
items, systems = ns["items"], ns["systems"]
NAMES = list(systems)
# ---------------- ①
it = items[0]
outs["s1"] = S.box("① 만든 채점 데이터", [(f"해요체로 끝나는 본문 문장 {len(ns['sentences']):,}개 중 문체를 바꿀 수 있는 {len(items)}개", 1),
                                        ("문장마다 똑같이 맞는 정답 셋: 해요체, 합니다체, 한다체", 2, "newf"), ("문장마다 오류 둘: 조사 바꾸기 또는 띄어쓰기 지우기", 3)],
                   ["답을 내는 \"고치기 시스템\" 다섯 개를 흉내 내요. 셋은 오류를 다 고쳤고 문체만 달라요. 하나는 오류 하나만 고쳤고, 하나는 아무것도 안 고쳤어요. 진짜 품질(고친 비율)을 알고 있으니 지표가 맞게 재는지 확인할 수 있어요."])
outs["t1"] = S.table("문장 하나의 예", ["", "문장 (굵은 곳이 정답 해요체와 다른 곳)"],
                     [["고칠 문장", mark(it["refs"][0], it["source"])], ["정답 1 (해요체)", html.escape(it["refs"][0])], ["정답 2 (합니다체)", mark(it["refs"][0], it["refs"][1])],
                      ["정답 3 (한다체)", mark(it["refs"][0], it["refs"][2])], ["하나만 고침", mark(it["refs"][0], it["half"])]], ["⌃는 빠진 띄어쓰기예요."])

# ---------------- ②
toy = ns["toy"]
S.claim(toy["가장 좋은 정답 하나"] == 5 and toy["정답마다 평균"] == 4.5 and toy["정답을 합친 횟수"] == 7, "② 강연 예와 같은 5, 4.5, 7")
outs["s2"] = S.table("② 강연의 예: 정답 둘로 1-gram 정밀도 재기", ["합치는 법", "맞은 단어 수", "정밀도"],
                     [[k, f"{toy[k]:g}", f"{toy[k]}/{toy['답의 단어 수']} = {toy[k] / toy['답의 단어 수']:.3f}"] for k in ("가장 좋은 정답 하나", "정답마다 평균", "정답을 합친 횟수")],
                     ["정답: \"the cat is on the mat\", \"a cat sits on the mat\". 답: \"the cat the cat on the mat\".",
                      "정답마다 따로 세면 the는 2번까지만 맞은 것으로 쳐요(잘라 세기). 그런데 두 정답의 횟수를 더하면 the가 3번까지 인정돼서, 말이 안 되는 답이 만점이 돼요. 합치는 방법 하나가 점수를 크게 바꿔요."])

# ---------------- ③
o1 = ns["one_ref"]
S.claim(o1["하나만 고침 (해요체)"] > o1["다 고침 (합니다체)"], "③ 정답 하나면 하나만 고친 답이 다 고친 합니다체보다 높음")
outs["s3"] = S.table("③ 정답 하나(해요체)로만 잰 BLEU (음절 단위, 문장 평균)", ["시스템", "진짜 고친 비율", "BLEU"],
                     [[n, f"{systems[n][1]:.0%}", f3(o1[n])] for n in NAMES],
                     [f"오류를 다 고친 합니다체({o1['다 고침 (합니다체)']:.3f})가 하나만 고친 해요체({o1['하나만 고침 (해요체)']:.3f})보다 낮아요. 문장 끝의 \"해요\"와 \"합니다\"가 다른 것이 고치지 못한 오류 하나보다 더 많은 글자 조각을 바꾸기 때문이에요.",
                      "BLEU는 \"정답과 얼마나 같은 조각을 썼나\"만 봐요. 정답이 하나면 그 정답과 다른 표현은 다 틀린 것이 돼요."])
wrows = []
for unit in ("char", "word"):
    sc = {n: float(np.mean([ns["single_bleu"](h, i["refs"][0], unit=unit) for h, i in zip(systems[n][0], items)])) for n in NAMES}
    wrows.append(["<b>음절</b> (지금)" if unit == "char" else "어절 (띄어쓰기로 나눔)", *[f3(sc[n]) for n in NAMES]])
outs["t3"] = S.table("조각을 음절이 아니라 어절로 자르면", ["단위", *NAMES], wrows,
                     ["어절 단위로 자르면 띄어쓰기 오류 하나가 어절 두 개를 통째로 바꿔서 오류에 훨씬 엄해져요. 그래서 이 데이터에서는 순서가 맞게 나와요. 한국어는 어절, 형태소, 음절 중 무엇으로 자르는지에 따라 같은 지표가 다르게 움직여요."])

# ---------------- ④
og = ns["one_ref_gleu"]
S.claim(o1["안 고침"] - og["안 고침"] > 0.05, "④ GLEU가 안 고친 답을 BLEU보다 크게 깎음")
outs["s4"] = S.table("④ 같은 정답 하나로 잰 BLEU와 GLEU", ["시스템", "BLEU", "GLEU", "차이"], [[n, f3(o1[n]), f3(og[n]), f3(og[n] - o1[n])] for n in NAMES],
                     [f"GLEU는 원문에 있던 조각 중 정답에는 없는 것(고쳐야 했던 곳)을 답이 그대로 쓰면 맞은 수에서 빼요. 그래서 안 고친 답이 {o1['안 고침']:.3f}에서 {og['안 고침']:.3f}로 크게 내려가요. 다 고친 답은 깎이지 않아요.",
                      "그래도 정답이 하나뿐이면 문체가 다른 답은 여전히 깎여요. 그건 정답을 늘려야 풀려요."])

# ---------------- ⑤
bd = ns["board"]
HOWS = ns["HOWS"]
S.claim(all(bd[("GLEU", h)]["다 고침 (합니다체)"] == 1.0 for h in ("가장 좋은 정답 하나", "정답을 합친 횟수")) and bd[("GLEU", "정답마다 평균")]["다 고침 (해요체)"] < 0.95,
        "⑤ 가장 좋은 하나와 합친 횟수는 다 고친 답이 1, 평균은 해요체를 0.95 아래로")
outs["s5"] = S.table("⑤ 정답 셋을 합치는 네 가지 방법 (GLEU)", ["시스템", *HOWS], [[n, *[f3(bd[('GLEU', h)][n]) for h in HOWS]] for n in NAMES],
                     ["\"가장 좋은 정답 하나\"와 \"정답을 합친 횟수\"는 다 고친 세 답에 모두 만점을 줘요. \"맞는 정답 중 하나와 맞으면 된다\"는 생각이에요.",
                      f"\"정답마다 평균\"은 다 고친 해요체에 {bd[('GLEU', '정답마다 평균')]['다 고침 (해요체)']:.3f}를 줘요. 나머지 두 정답(합니다체, 한다체)과 달라서 깎인 거예요. 정답들의 분포 전체에 가까울수록 좋다는 생각이라, 정답들이 서로 많이 다르면 맞는 답도 깎여요. softmax 가중 평균은 그 사이예요."])
outs["t5"] = S.table("BLEU로 같은 비교를 하면", ["시스템", *HOWS], [[n, *[f3(bd[('BLEU', h)][n]) for h in HOWS]] for n in NAMES],
                     ["순서는 GLEU와 같지만, 안 고친 답과 고친 답의 차이가 GLEU보다 작아요."])
fig, ax = plt.subplots(figsize=(9.6, 3.6))
x = np.arange(len(NAMES))
cols = [C["purple"], C["blue"], C["mint"], C["pink"]]
for j, h in enumerate(HOWS):
    ax.bar(x + (j - 1.5) * 0.2, [bd[("GLEU", h)][n] for n in NAMES], width=0.2, color=cols[j], label=h)
ax.set_xticks(x, [n.replace(" (", "\n(") for n in NAMES], fontsize=9)
ax.set(ylim=(0.6, 1.1), yticks=[0.6, 0.7, 0.8, 0.9, 1.0], ylabel="GLEU (문장 평균)", title="정답 셋을 합치는 방법에 따라")
ax.legend(fontsize=8, ncol=4, loc="upper center", frameon=False)
P_HOW = fig_save(fig, "aggregation")
outs["f_how"] = chart("합치는 방법별 점수", P_HOW, "", ["왼쪽 세 묶음은 모두 다 고친 답이에요. 보라(가장 좋은 하나)와 분홍(합친 횟수)은 셋 모두 1이고, 파랑(평균)과 민트(가중 평균)는 1보다 낮고 셋에 서로 다른 점수를 줘요."])

# ---------------- ⑥
ag = ns["agreement"]
S.claim(ag[("BLEU", "정답 하나 (해요체)")] < 0.8 and ag[("GLEU", "정답 하나 (해요체)")] > 0.95 and ag[("BLEU", "정답 셋")] == 1.0, "⑥ BLEU 정답 하나 0.8 아래, GLEU 0.95 위, 정답 셋은 1")
outs["s6"] = S.table("⑥ 문장마다, 더 많이 고친 답에 더 높은 점수를 줬나", ["지표", "정답", "맞게 줄 세운 비율"],
                     [[m, r, f"{v:.1%}"] for (m, r), v in ag.items()],
                     [f"같은 문장에 대한 두 답 중 진짜로 더 많이 고친 쪽을 지표도 더 높게 준 비율이에요. 문장 {len(items)}개 × 비교할 짝 9개를 셌어요.",
                      f"정답이 하나면 BLEU는 {ag[('BLEU', '정답 하나 (해요체)')]:.0%}만 맞혀요. GLEU는 정답이 하나여도 {ag[('GLEU', '정답 하나 (해요체)')]:.0%}이고, 정답이 셋이면 둘 다 모두 맞혀요(정답 셋은 \"가장 좋은 정답 하나\"로 합쳤어요).",
                      "이 데이터는 정답이 될 수 있는 표현을 우리가 다 알아서 쉬운 편이에요. 실제 정답 모음은 맞는 표현의 일부만 담고 있어서, 정답을 셋 모아도 이만큼 깔끔하지 않아요."])

# ---------------- ⑦
gr = ns["growth"]
S.claim(gr[("정답을 합친 횟수", 3)] > gr[("정답을 합친 횟수", 1)] + 0.05 and abs(gr[("정답마다 평균", 3)] - gr[("정답마다 평균", 1)]) < 1e-6, "⑦ 합친 횟수는 오르고 평균은 그대로")
outs["s7"] = S.table("⑦ 정답 수를 늘리면: \"하나만 고침\"의 GLEU", ["합치는 법", "정답 1개", "2개", "3개"],
                     [[h, *[f3(gr[(h, k)]) for k in (1, 2, 3)]] for h in ("가장 좋은 정답 하나", "정답마다 평균", "정답을 합친 횟수")],
                     ["정답 k개를 고르는 모든 경우의 평균이에요. \"하나만 고침\"은 해요체라서 해요체 정답이 들어 있을 때 점수가 높아요.",
                      f"\"정답을 합친 횟수\"는 정답이 늘수록 계속 올라가요({gr[('정답을 합친 횟수', 1)]:.3f} → {gr[('정답을 합친 횟수', 3)]:.3f}). 인정되는 조각이 늘어서예요. 강연의 실험(여러 언어의 맞춤법 고치기)에서도 합친 횟수가 정답을 더할수록 안정적으로 올랐어요.",
                      "\"정답마다 평균\"은 정답 수와 상관없이 같아요. 정답을 더 모아도 평균은 \"정답 하나의 기댓값\"이라서예요. 정답을 모으는 데 돈을 쓴다면 어떤 방법으로 합칠지부터 정해야 해요."])

flow = [[1, "데이터", "정답 셋, 오류 둘", "진짜 품질을 앎"], [2, "BLEU", "잘라 세기", "합치는 법이 점수를 바꿈"], [3, "정답 하나", "문체 차이", "틀린 순서"], [4, "GLEU", "안 고친 곳 벌점", "오류에 엄하게"],
        [5, "네 가지 합치기", "최대, 평균, 가중, 합침", "존재 대 분포"], [6, "줄 세우기", "짝마다 맞나", "정답 수의 힘"], [7, "정답 늘리기", "1, 2, 3개", "합친 횟수만 오름"]]
S.save(ROOT / "topics" / "backend" / "multi-ref-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p>요약, 번역, 맞춤법 고치기처럼 <b>맞는 답이 여러 개</b>인 일은 채점이 어려워요. 사람이 정답을 몇 개 써 두고 n-gram 지표(BLEU, ROUGE, GLEU)로 \"정답과 얼마나 같은 조각을 썼나\"를 재는 게 가장 싸고 빠른 방법인데, <b>정답이 하나뿐이면</b> 맞지만 다르게 쓴 답을 틀렸다고 해요.</p>"""
why += K.table(["", "정답 하나", "정답 여러 개"], [["맞지만 다르게 쓴 답", "깎임", "다른 정답과 맞으면 인정"], ["새로 정할 것", "없음", "정답별 점수를 어떻게 합칠까 (이 쪽 ⑤)"], ["비용", "정답 1개를 쓰는 사람 시간", "정답 수만큼"]])
why += "<p><a href=\"llm-judge.html\">LLM 채점</a>이 비싸거나 흔들릴 때, 또는 LLM 채점이 맞는지 확인할 기준이 필요할 때 이 지표들을 같이 써요. <a href=\"../model/gpt-generation.html\">작은 GPT</a>의 ROUGE 요약 채점도 같은 원리예요.</p>"
whole = """<p>데이터 → BLEU → 정답 하나로 재기 → GLEU → 정답 합치기 네 가지 → 문장마다 줄 세우기 → 정답 늘리기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/eval_refs/multi_ref.py</code> 하나예요. 지표를 라이브러리 없이 직접 짜서, 합치는 방법을 바꿔 끼울 수 있게 했어요.</p>"""
steps = [
    K.step(1, "정답 셋과 일부러 넣은 오류", "해요체 문장을 합니다체와 한다체로 바꿔 똑같이 맞는 정답 셋을 만들고, 오류 둘을 넣어 고칠 문장을 만들어요.",
           "<p>실제 맞춤법 고치기 데이터는 사람이 문장마다 고친 정답을 2~3개 써요. 고치는 방법이 여러 가지이기 때문이에요(\"음식이 안 막였습니다\"를 \"음식을 안 먹었습니다\"로도, \"음식이 안 맞았습니다\"로도 고칠 수 있어요). 여기서는 맞는 표현 차이를 문체로 흉내 냈어요.</p>"
           + K.out("t1")
           + K.code_table([("three_styles(s)", "문장 끝 \"해요.\"를 \"합니다.\", \"한다.\"로"), ("make_errors(s, k=2)", "조사 바꾸기(이↔가, 을↔를, 은↔는) 또는 띄어쓰기 지우기")]), "s1"),
    K.step(2, "BLEU: 정답과 같은 조각을 몇 개 썼나", "답의 n-gram(1~4개 연속 조각)이 정답에 몇 개 있는지 세고, 너무 짧은 답에 벌점을 줘요.",
           "<p><b>BLEU</b>는 n = 1~4마다 \"답의 조각 중 정답에도 있는 비율\"(정밀도)을 구해 기하 평균을 내요. 같은 조각을 되풀이해 점수를 따지 못하게, 정답에 있는 횟수까지만 맞은 것으로 쳐요(<b>잘라 세기</b>). 짧게 쓰면 정밀도가 쉽게 높아지니까 정답보다 짧으면 벌점(brevity penalty)을 곱해요.</p>"
           "<p>정답이 여러 개일 때 점수를 합치는 법은 크게 넷이에요. 정답마다 점수를 내서 <b>가장 높은 것</b>, <b>평균</b>, <b>높은 것에 무게를 더 준 평균</b>(softmax), 그리고 처음부터 <b>정답들의 조각 횟수를 더해</b> 한 정답처럼 쓰는 것이에요. 원래 BLEU는 마지막 방법을 써요.</p>"
           + K.code_table([("Counter(tuple(toks[i:i+n]) ...)", "n-gram 세기"), ("min(c, ref_counts[g])", "잘라 세기"), ("min(1, exp(1 − 정답 길이 / 답 길이))", "짧은 답 벌점")]), "s2"),
    K.step(3, "정답 하나로만 재면", "다섯 시스템의 답을 해요체 정답 하나로 BLEU를 재요.",
           "<p>한국어는 띄어쓰기 단위(어절)에 조사와 어미가 붙어서, 어절로 자르면 작은 차이도 어절 전체가 달라져요. 그래서 이 쪽은 기본으로 <b>음절(글자)</b> 단위로 잘라요. 형태소 분석기로 자르는 방법도 많이 써요.</p>"
           + K.try_("어절 단위로 자르면?", "t3"), "s3"),
    K.step(4, "GLEU: 고치지 않은 오류에 벌점", "원문에 있었지만 정답에는 없는 조각을 답이 그대로 쓰면, 맞은 수에서 빼요.",
           "<p>맞춤법 고치기에서는 답의 대부분이 원문과 같아요. 그래서 BLEU는 아무것도 안 고친 답에도 높은 점수를 줘요. <b>GLEU</b>는 \"원문에는 있는데 정답에는 없는 조각\"을 고쳐야 했던 곳으로 보고, 답에 그 조각이 남아 있으면 벌점을 줘요(구글의 GLEU와는 다른 지표예요).</p>"
           + K.code_table([("s - r 에 해당하는 조각", "원문에는 있고 정답에는 없는 조각 = 고쳐야 할 곳"), ("max(맞은 수 − 안 고친 수, 0)", "GLEU의 분자")]), "s4"),
    K.step(5, "정답 셋을 합치는 네 가지 방법", "정답 셋으로 GLEU와 BLEU를 네 가지 방법으로 합쳐 재요.",
           "<p>강연은 이 넷을 하나의 틀로 정리해요. 정답마다 점수 f₁, …, fₙ을 내고 연산자 Φ로 하나로 만들어요. 최대와 합친 횟수는 \"맞는 정답 중 <b>하나라도</b> 맞으면 된다\"(존재), 평균과 가중 평균은 \"정답들 <b>전체에</b> 얼마나 가깝나\"(분포)라는 다른 질문에 답해요. 어느 쪽이 맞는지는 일에 따라 달라요.</p>"
           + K.out("f_how")
           + K.code_table([("f.max()", "가장 좋은 정답 하나"), ("f.mean()", "정답마다 평균"), ("Σ softmax(τ f) × f", "높은 점수에 무게를 더 준 평균 (τ = 10)"), ("sum(정답들의 Counter)", "정답을 합친 횟수")])
           + K.try_("BLEU로 같은 비교를 하면?", "t5"), "s5"),
    K.step(6, "문장마다 맞게 줄 세우나", "같은 문장에 대한 두 답 중, 진짜로 더 많이 고친 쪽에 지표가 더 높은 점수를 줬는지 세요.",
           "<p>지표가 좋은지는 결국 <b>사람의 판단과 같은 순서를 매기나</b>로 봐요. 보통은 사람이 매긴 점수와의 상관을 재는데, 여기서는 진짜 품질을 알고 있어서 짝마다 맞혔는지 바로 셀 수 있어요.</p>"
           + K.code_table([("combinations(names, 2)", "같은 문장의 답 둘씩"), ("sc[better] > sc[worse]", "더 많이 고친 답이 더 높은 점수인가")]), "s6"),
    K.step(7, "정답 수를 늘리면", "정답을 1개, 2개, 3개로 늘려 가며 같은 답의 점수가 어떻게 바뀌는지 봐요.",
           "<p>정답을 더 모으는 데는 사람 시간이 들어요. 합치는 방법에 따라 정답을 늘린 효과가 다르니, 데이터를 모으기 전에 정해야 해요.</p>", "s7"),
]
practice = [("요약, 번역, 고치기를 채점한다", "정답을 2개 이상 모을 수 있으면 모으고, 합치는 방법을 문서에 적어 둬요. 숫자는 같은 방법끼리만 비교해요."),
            ("맞춤법, 문장 고치기", "BLEU보다 GLEU처럼 원문을 함께 보는 지표를 써요. 아무것도 안 고친 답의 점수를 꼭 기준선으로 같이 적어요."),
            ("한국어", "어절, 형태소, 음절 중 무엇으로 자르는지에 따라 점수가 달라요. 자르는 법도 같이 적어요. n-gram 지표가 모자라면 <a href=\"llm-judge.html\">LLM 채점</a>을 같이 써요.")]
libs = [("①~⑦", "<code>collections.Counter</code>, <code>itertools.combinations</code>, NumPy", "n-gram 세기, 짝 만들기, 평균")]
K.build(ROOT / "topics" / "backend" / "multi-ref.html", title="여러 정답으로 채점하기", h1="여러 정답으로 채점하기: BLEU, GLEU, 정답을 합치는 법", track="백엔드",
        crumb="백엔드 &gt; LLM 활용",
        one_line="맞는 답이 여러 개인 일을 <b>정답 하나</b>로 n-gram 채점하면 맞지만 다르게 쓴 답을 틀렸다고 해요. 정답을 여러 개 두고 점수를 <b>합치는 방법</b>(가장 좋은 하나, 평균, 가중 평균, 횟수 합치기)을 정해야 해요. 고치기 일에는 <b>안 고친 곳에 벌점</b>을 주는 GLEU가 더 맞아요.",
        why=why, whole=whole, steps=steps, code_file="examples/eval_refs/multi_ref.py", practice=practice, libs=libs,
        libs_note="<code>three_styles</code>, <code>make_errors</code>, <code>ngrams</code>, <code>clipped</code>, <code>bleu</code>, <code>gleu</code>, <code>multi</code>, <code>system_scores</code>, <code>pair_accuracy</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"llm-judge.html\">LLM으로 답 채점하기</a>", "같은 카테고리"), ("<a href=\"../model/gpt-generation.html\">작은 GPT와 디코딩</a>", "ROUGE로 요약 채점")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: Upstage 10Talks Lecture 05 \"A Formal Framework for Fluency-based Multi Reference Evaluation\"(n-gram 언어 모델, BLEU와 ROUGE와 GLEU, 정답을 합치는 네 가지 방법과 존재/분포 해석, 연산자 Φ로 정리한 틀, 여러 언어의 맞춤법 고치기 실험, 한국어 n-gram의 어려움). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/multi_ref.py</code>가 <code>examples/eval_refs/multi_ref.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
