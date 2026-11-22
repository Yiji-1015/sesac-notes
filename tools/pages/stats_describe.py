"""모델 > 데이터 분석 기초 1쪽: 요약하고 추정하기. examples/stats/describe_estimate.py 하나가 ①~⑥으로 자란다.
실행: python tools/pages/stats_describe.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy import stats  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "stats"
code = S.parse((EX / "describe_estimate.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="describe_estimate.py")
ENV = f"Python {sys.version.split()[0]}, pandas {pd.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__} (seed 0)"
f1 = lambda x: f"{float(x):,.1f}"  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "stats"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/stats/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
req = ns["req"]
# ---------------- ①
miss = ns["missing"]
S.claim(miss["elapsed_ms"] == 0 and miss["input_chars"] == 6, "① 처리 시간은 빈 칸 없음, 입력 길이는 6건 빔")
outs["s1"] = S.table("① 열마다 척도와 빈 칸", ["열", "척도", "빈 칸", "예시 값"],
                     [[f"<code>{c}</code>", ns["SCALES"][c], int(miss[c]), ", ".join(str(v) for v in req[c].dropna().unique()[:3])] for c in ns["SCALES"]],
                     ["<code>failed</code>는 0과 1이라 숫자처럼 보이지만, 크기가 아니라 \"실패인가\"라는 이름표예요(명목). 평균 0.3은 \"실패 비율 30%\"로만 읽어요.",
                      "<code>priority</code>는 low < normal < high 순서가 있지만, low와 normal의 차이가 normal과 high의 차이와 같다는 보장은 없어요(서열). 그래서 1, 2, 3으로 바꿔 평균 내는 건 조심해야 해요."])

# ---------------- ②
ce, sp, sh, shr = ns["center"], ns["spread"], ns["shape"], ns["shape_retry"]
S.claim(abs(sh["왜도"]) < 0.1 and shr["왜도"] > 1, "② 처리 시간은 거의 대칭, 재시도는 오른쪽 꼬리")
outs["s2"] = S.table("② 처리 시간(ms) 100건 요약", ["무엇", "값"],
                     [*[[k, f1(v)] for k, v in ce.items()], *[[k, f3(v) if k == "변동계수" else f1(v)] for k, v in sp.items()], *[[k, f3(v)] for k, v in sh.items()]],
                     [f"평균({ce['평균']:,.0f})과 중앙값({ce['중앙값']:,.0f})이 거의 같고 왜도가 0에 가까워요. 좌우가 대칭에 가까운 분포예요. 첨도가 음수({sh['첨도 (정규분포 = 0)']:.2f})라 정규분포보다 가운데가 덜 뾰족하고 꼬리가 짧아요.",
                      f"재시도 횟수는 왜도가 {shr['왜도']:.2f}로 오른쪽 꼬리가 길어요. 대부분 0이고 가끔 2, 3이에요. 이런 자료는 평균보다 중앙값이나 비율로 요약하는 게 더 정직해요."])
mg, ms = ns["means_growth"], ns["means_speed"]
outs["s2m"] = S.table("평균이 하나가 아니에요", ["상황", "산술평균", "맞는 평균"],
                      [["요청 수가 석 달 동안 1.10배, 1.50배, 0.80배", f3(mg["산술평균"]) + "배", f"기하평균 {mg['기하평균']:.3f}배"],
                       ["같은 거리를 60km/h로 가고 40km/h로 오기", f"{ms['산술평균']:.0f}km/h", f"조화평균 {ms['조화평균']:.0f}km/h"]],
                      [f"석 달 뒤 요청 수는 1.10 × 1.50 × 0.80 = {1.1 * 1.5 * 0.8:.2f}배예요. 매달 {mg['기하평균']:.3f}배씩 늘었다고 하면 {mg['기하평균'] ** 3:.2f}배로 맞지만, 산술평균 {mg['산술평균']:.3f}배로는 {mg['산술평균'] ** 3:.2f}배가 돼서 틀려요. 곱해서 쌓이는 비율은 기하평균이에요.",
                       "왕복 120km를 가는 데 2시간, 오는 데 3시간이면 5시간에 120km라 평균 24km/h × 2 = 48km/h예요. 시간 대신 거리를 같게 둔 비율은 조화평균이에요."])

# ---------------- ③
pg = ns["p_fail_given"]
S.claim(ns["dispersion"] > 1.5 and ns["observed"][0] > ns["expected"][0] + 10, "③ 재시도는 포아송보다 0이 훨씬 많고 분산이 평균보다 크다")
outs["s3"] = S.box("③ 확률로 읽기", [(f"P(실패) = {ns['p_fail']:.2f}", 1), (f"P(실패 | 우선순위) → " + ", ".join(f"{k} {v:.3f}" for k, v in pg.items()), 2, "newf"),
                                   (f"P(high 그리고 실패) = {ns['p_high_and_fail']:.2f}, P(high) × P(실패) = {ns['p_high'] * ns['p_fail']:.3f}", 3),
                                   (f"재시도 횟수: 평균 {ns['lam']:.2f}, 분산 ÷ 평균 = {ns['dispersion']:.2f}", 4, "newf")],
                   ["조건부 확률 P(A | B)는 \"B인 것들만 놓고 본 A의 비율\"이에요. high 요청의 실패율이 normal보다 높아 보이지만, 이 차이가 우연인지는 다음 쪽의 교차분석으로 따져요.",
                    f"두 사건이 독립이면 결합 확률이 각 확률의 곱과 같아야 해요. {ns['p_high_and_fail']:.2f}과 {ns['p_high'] * ns['p_fail']:.3f}은 가깝지만 똑같지는 않아요. 100건으로는 이 정도 차이가 흔해요.",
                    f"포아송 분포(일정한 비율로 드물게 일어나는 일의 횟수)는 평균과 분산이 같아요. 재시도는 분산이 평균의 {ns['dispersion']:.1f}배라 포아송이 아니에요. 대부분은 한 번도 재시도하지 않고, 하는 요청은 여러 번 하는 두 무리가 섞여 있어요."])
fig, axs = plt.subplots(1, 2, figsize=(12, 3.7))
axs[0].hist(req["elapsed_ms"], bins=15, color=OK, alpha=.85)
for k, col, ls in (("평균", FAIL, "-"), ("중앙값", C["blue"], "--")):
    axs[0].axvline(ce[k], color=col, lw=2, ls=ls, label=f"{k} {ce[k]:,.0f}")
axs[0].set(xlabel="처리 시간 (ms)", ylabel="건수", title="처리 시간 100건: 거의 대칭")
axs[0].legend(fontsize=9)
obs, exp = ns["observed"], ns["expected"]
x = np.arange(len(obs))
axs[1].bar(x - 0.2, obs.values, width=0.4, color=OK, label="실제")
axs[1].bar(x + 0.2, exp.values, width=0.4, color=C["gray"], label=f"포아송(λ = {ns['lam']:.2f})이라면")
axs[1].set_xticks(x, [str(i) for i in obs.index])
axs[1].set(xlabel="재시도 횟수", ylabel="건수", title="재시도: 0이 너무 많고 3도 많아요")
axs[1].legend(fontsize=9)
P_DIST = fig_save(fig, "distributions")
outs["f_dist"] = chart("두 열의 분포", P_DIST, "", ["왼쪽은 평균과 중앙값이 겹쳐요. 오른쪽은 실제(보라)가 0과 3에서 포아송(회색)보다 많고, 1에서 적어요."])

# ---------------- ④
dr, tm = ns["draws"], ns["true_mean"]
S.claim(abs(dr["단순 무작위"].mean() - tm) < 10 and dr["편한 대로"].mean() - tm > 300 and dr["층화"].std() < dr["단순 무작위"].std(), "④ 무작위는 치우침 없음, 편한 대로는 300 넘게 치우침, 층화가 덜 흔들림")
outs["s4"] = S.table(f"④ 1년치 10만 건(진짜 평균 {tm:,.1f}ms)에서 100건씩 2,000번 뽑아 평균 내기", ["뽑는 방법", "2,000번 평균의 평균", "치우침", "흔들림 (표준편차)"],
                     [[k, f1(v.mean()), f"{v.mean() - tm:+,.1f}", f1(v.std())] for k, v in dr.items()],
                     [f"편한 대로(research 요청함에서만) 뽑으면 아무리 여러 번 뽑아도 평균이 {dr['편한 대로'].mean() - tm:,.0f}ms쯤 높게 나와요. 건수를 늘려도 고쳐지지 않는 치우침이에요. 모집단을 대표하지 못하는 표본은 일반화하면 안 돼요.",
                      f"층화(종류 비율대로 나눠 각각 무작위)는 치우침이 없으면서 흔들림이 {dr['단순 무작위'].std():.0f}에서 {dr['층화'].std():.0f}로 줄었어요. 종류마다 처리 시간이 달라서, 종류 비율이 표본마다 달라지는 흔들림을 없앤 거예요."])
fig, ax = plt.subplots(figsize=(8, 3.6))
for (k, v), col in zip(dr.items(), (OK, FAIL, C["blue"])):
    ax.hist(v, bins=40, color=col, alpha=.55, label=k)
ax.axvline(tm, color=C["ink"], lw=2, label=f"진짜 평균 {tm:,.0f}")
ax.set(xlabel="100건 표본의 평균 처리 시간 (ms)", ylabel="2,000번 중 횟수", title="뽑는 방법에 따른 표본 평균")
ax.legend(fontsize=9)
P_SAMP = fig_save(fig, "sampling")
outs["f_samp"] = chart("표본을 뽑는 방법", P_SAMP, "", ["보라(단순 무작위)와 파랑(층화)은 진짜 평균 둘레에 있고, 파랑이 더 좁아요. 분홍(편한 대로)은 통째로 오른쪽에 있어요."])

# ---------------- ⑤
smn, se = ns["sample_means"], ns["se_theory"]
S.claim(all(abs(smn[n].std() / se[n] - 1) < 0.05 for n in ns["sizes"]) and stats.skew(smn[100]) < 0.3 < stats.skew(smn[1]), "⑤ 표본 평균의 표준편차가 σ/√n과 5% 안, n이 커질수록 대칭")
outs["s5"] = S.table("⑤ n건 표본의 평균을 5,000번 (모집단은 오른쪽 꼬리가 긴 분포)", ["n", "표본 평균들의 표준편차", "σ ÷ √n", "왜도"],
                     [[n, f1(smn[n].std()), f1(se[n]), f3(stats.skew(smn[n]))] for n in ns["sizes"]],
                     ["표본 평균이 흔들리는 정도는 σ ÷ √n(<b>표준오차</b>)과 거의 같아요. n을 4배로 늘려야 흔들림이 반으로 줄어요.",
                      f"모집단은 오른쪽 꼬리가 길어서(n = 1의 왜도 {stats.skew(smn[1]):.2f}) 정규분포가 아니지만, n = 30이면 평균의 분포가 거의 대칭이에요. 이게 <b>중심극한정리</b>예요. 그래서 평균에 대한 검정과 구간은 원래 분포가 정규가 아니어도 대개 쓸 수 있어요."])
fig, axs = plt.subplots(1, 4, figsize=(13, 3.2), sharey=False)
for ax, n in zip(axs, ns["sizes"]):
    v = smn[n]
    ax.hist(v, bins=50, color=OK, alpha=.8, density=True)
    xs = np.linspace(v.min(), v.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, tm, se[n]), color=FAIL, lw=1.8)
    ax.set(title=f"n = {n}", xlabel="표본 평균 (ms)")
    ax.set_yticks([])
P_CLT = fig_save(fig, "clt")
outs["f_clt"] = chart("표본 평균의 분포 (분홍: 평균 μ, 표준편차 σ/√n인 정규분포)", P_CLT, "", ["n = 1은 모집단 그대로라 오른쪽으로 늘어져 있어요. n이 커질수록 분홍 정규 곡선에 붙고 좁아져요."])

# ---------------- ⑥
cv = ns["coverage"]
S.claim(cv[(5, False)] < 0.9 and cv[(5, True)] > cv[(5, False)] and abs(cv[(100, True)] - 0.95) < 0.015, "⑥ n = 5에서 z 구간은 90% 아래, t가 더 맞고, n = 100이면 95% 근처")
lo, hi = ns["ci_ours"]
outs["s6"] = S.table("⑥ 95% 신뢰구간이 진짜 평균을 담은 비율 (2,000번)", ["n", "z로 (정규분포의 1.96)", "t로 (t분포의 값)"],
                     [[n, f"{cv[(n, False)]:.1%}", f"{cv[(n, True)]:.1%}"] for n in (5, 30, 100)],
                     [f"n = 5에서 z로 만든 \"95%\" 구간은 실제로 {cv[(5, False)]:.0%}만 맞아요. 표준편차까지 표본으로 짐작하니 더 흔들리는데, z는 그걸 모르고 구간을 좁게 잡아요. t분포는 그만큼 꼬리가 두꺼워서 더 넓게 잡아요.",
                      f"t로 만들어도 n = 5에서는 {cv[(5, True)]:.0%}로 95%에 못 미쳐요. t분포는 모집단이 정규분포라고 가정하는데, 이 모집단은 오른쪽 꼬리가 길어서예요. n이 커지면 둘 다 95%에 가까워져요.",
                      f"우리 100건의 평균 처리 시간에 대한 95% 신뢰구간은 {lo:,.0f}~{hi:,.0f}ms예요. \"이 방법으로 구간을 만들면 100번 중 95번은 진짜 평균을 담는다\"는 뜻이지, \"진짜 평균이 이 안에 있을 확률이 95%\"는 아니에요(진짜 평균은 정해진 하나의 값이라)."])
rng2 = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(8.6, 3.6))
miss_n = 0
for i in range(60):
    l_, h_ = ns["interval"](ns["pop_ms"][rng2.integers(ns["N"], size=30)])
    hit = l_ <= tm <= h_
    miss_n += not hit
    ax.plot([i, i], [l_, h_], color=OK if hit else FAIL, lw=2)
ax.axhline(tm, color=C["ink"], lw=1.5, label=f"진짜 평균 {tm:,.0f}")
ax.set(xlabel="표본 번호", ylabel="처리 시간 (ms)", title=f"n = 30 표본 60개의 95% 신뢰구간 (분홍 {miss_n}개가 빗나감)")
ax.legend(fontsize=9)
P_CI = fig_save(fig, "confidence_intervals")
outs["f_ci"] = chart("신뢰구간 60개", P_CI, "", ["구간은 표본마다 달라요. 그중 대부분(보라)은 진짜 평균을 지나고, 몇 개(분홍)는 빗나가요. 95%는 이 \"맞히는 비율\"에 대한 약속이에요."])

flow = [[1, "자료와 척도", "명목, 서열, 비율", "빈 칸"], [2, "요약", "가운데, 퍼짐, 모양", "평균의 종류"], [3, "확률", "조건부, 독립", "포아송 확인"],
        [4, "표본 뽑기", "무작위, 편한 대로, 층화", "치우침"], [5, "중심극한정리", "평균의 분포", "σ ÷ √n"], [6, "신뢰구간", "95%의 뜻", "z 대 t"]]
S.save(ROOT / "topics" / "model" / "stats-describe-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow, "env": ENV})

why = """<p>머신러닝 쪽에서는 데이터를 바로 모델에 넣었어요. 그 전에 할 일이 있어요. <b>자료를 정확히 요약하고</b>(기술통계), <b>가진 일부로 전체를 짐작할 때 얼마나 틀릴 수 있는지 아는 것</b>(추론통계)이에요. 실무 분석의 대부분은 모델이 아니라 이 단계와 데이터 정리에 시간을 써요.</p>"""
why += K.table(["", "기술통계 (이 쪽 ①~③)", "추론통계 (이 쪽 ④~⑥, 다음 쪽)"], [["질문", "가진 100건은 어떻게 생겼나", "1년치 전체는 어떨까, 차이는 우연일까"],
                                                                     ["도구", "평균, 분산, 분위수, 그림", "표본, 표준오차, 신뢰구간, 검정"], ["결과", "확실한 숫자", "틀릴 확률이 붙은 짐작"]])
why += "<p>이 단계는 데이터로 일하는 전체 흐름의 앞부분이에요. 모델은 그 뒤 일부이고, 실제 서비스에서는 모델 코드보다 그 주변(데이터 준비, 배포, 감시)이 훨씬 커요.</p>"
why += K.table(["단계", "하는 일", "이 사이트에서"], [["1. 문제 정의", "데이터로 풀 수 있는 질문인가, 무엇으로 성공을 재나", "이 쪽 ①, <a href=\"ml-metrics-clustering.html\">평가지표</a>"],
                                               ["2. 데이터 준비", "모으기, 정리하기, 탐색과 그림, 버전 관리", "이 쪽 전체, <a href=\"../backend/document-reading.html\">문서 읽기</a>"],
                                               ["3. 분석과 모델", "검정, 회귀, 머신러닝, 실험 기록", "<a href=\"stats-tests.html\">다음 쪽</a>, 머신러닝 기초"],
                                               ["4. 배포", "보고서와 대시보드(사람이 읽음) 또는 API(기계가 씀)", "<a href=\"../backend/fastapi.html\">FastAPI</a>"],
                                               ["5. 감시", "들어오는 데이터가 바뀌는지(데이터 드리프트), 성능이 떨어지는지", "-"], ["6. 다시 학습", "새 데이터로 고쳐서 끊김 없이 바꿔 넣기", "-"]])
why += "<p>역할도 나뉘어요. 데이터 엔지니어는 수집과 저장, 데이터 분석가는 요약과 그림과 검정, 데이터 과학자와 ML 엔지니어는 모델과 배포를 주로 맡아요.</p>"
why += "<p>예제는 머신러닝 기초와 같은 업무 요청 100건이에요. 전체(모집단)가 필요한 ④~⑥은 코드가 만든 가상의 1년치 10만 건에서 표본을 뽑아, 진짜 값과 비교해요.</p>"
whole = """<p>자료와 척도 → 요약 → 확률로 읽기 → 표본 뽑기 → 표본 평균의 성질 → 신뢰구간. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/stats/describe_estimate.py</code> 하나예요. 데이터는 머신러닝 예제의 <code>requests.csv</code>를 복사해 썼어요.</p>"""
steps = [
    K.step(1, "자료와 척도", "열마다 어떤 종류의 값인지(척도)부터 정해요. 척도에 따라 쓸 수 있는 계산과 그림이 달라요.",
           "<p>값을 잰 규칙을 <b>척도</b>라고 해요. 정보가 적은 쪽부터 넷이에요.</p>"
           + K.table(["척도", "뜻", "할 수 있는 계산", "예"], [["명목", "이름만 다름", "같다, 다르다, 세기", "작업 종류, 성별"], ["서열", "순서가 있음 (간격은 모름)", "+ 크다, 작다, 중앙값", "우선순위, 만족도 1~5"],
                                                          ["등간", "간격이 같음 (0이 \"없음\"이 아님)", "+ 더하기, 빼기, 평균", "섭씨 온도, 연도"], ["비율", "0이 진짜 \"없음\"", "+ 곱하기, 나누기 (2배)", "처리 시간, 입력 길이"]])
           + K.out("s1")
           + "<p>자료를 어디서 얻었는지도 기록해요. 내가 직접 모은 <b>1차 자료</b>와 남이 모아 둔 <b>2차 자료</b>, 지켜보기만 한 <b>관측 자료</b>와 조건을 내가 정해 준 <b>실험 자료</b>는 다음 쪽에서 말할 수 있는 결론이 달라요. 설문으로 모을 때는 한 문항에 한 가지만 묻고, 답을 유도하지 않고, 민감한 질문은 뒤에 둬요.</p>"
           + K.code_table([("req.isna().sum()", "열마다 빈 칸 수"), ("SCALES", "열 이름 → 척도 (사람이 정해요)")]), "s1"),
    K.step(2, "요약하기: 가운데, 퍼짐, 모양", "처리 시간 100건을 가운데(평균, 중앙값), 퍼짐(분산, 표준편차, 사분위 범위), 모양(왜도, 첨도)으로 요약해요.",
           "<p><b>가운데</b>: 평균은 모든 값을 쓰지만 튀는 값에 끌려가요. 중앙값은 정렬했을 때 가운데라 튀는 값에 덜 흔들려요. 위아래 10%를 빼고 평균 낸 절사 평균은 그 사이예요. 최빈값은 가장 많이 나온 값이라 명목 척도에도 써요.</p>"
           "<p><b>퍼짐</b>: 분산은 평균에서 떨어진 거리의 제곱 평균, 표준편차는 그 제곱근이라 원래 단위(ms)예요. 표본으로 계산할 때는 n − 1로 나눠요(평균을 표본에서 이미 한 번 정했으니 자유롭게 움직이는 값이 하나 줄어서). 사분위 범위는 가운데 50%의 폭이에요. 변동계수(표준편차 ÷ 평균)는 단위가 다른 것끼리 퍼짐을 비교할 때 써요.</p>"
           "<p><b>모양</b>: <b>왜도</b>는 치우침이에요. 양수면 오른쪽 꼬리가 길어요(평균 > 중앙값). <b>첨도</b>는 꼬리의 두께예요. 여기서는 정규분포를 0으로 둔 값이라, 양수면 튀는 값이 정규분포보다 자주 나와요.</p>"
           + K.out("s2") + K.out("s2m")
           + K.code_table([("stats.trim_mean(t, 0.1)", "위아래 10%씩 빼고 평균"), ("t.var()", "pandas는 기본으로 n − 1로 나눠요 (numpy는 n)"), ("stats.skew, stats.kurtosis", "왜도, 첨도 (정규분포 = 0)"),
                           ("stats.gmean, stats.hmean", "기하평균, 조화평균")]), None),
    K.step(3, "확률로 읽기: 조건부 확률, 독립, 분포", "실패 비율을 우선순위별로 나눠 보고, 재시도 횟수가 흔히 쓰는 분포(포아송)를 따르는지 확인해요.",
           "<p>한 사건의 확률 P(A), 둘이 함께 일어날 <b>결합 확률</b> P(A 그리고 B), B인 것들 중 A의 비율인 <b>조건부 확률</b> P(A | B) = P(A 그리고 B) ÷ P(B)가 있어요. A와 B가 <b>독립</b>이면 P(A 그리고 B) = P(A) × P(B)예요. 독립은 \"서로 영향이 없다\", 배반은 \"동시에 일어날 수 없다\"로 다른 말이에요. 조건부 확률을 뒤집는 식이 <a href=\"ml-svm-lda-pca.html\">SVM, 판별분석</a> ④의 베이즈 정리예요.</p>"
           "<p>값이 나오는 규칙을 식으로 적은 게 <b>확률분포</b>예요.</p>"
           + K.table(["분포", "어떤 값", "예"], [["이항", "n번 시도 중 성공 횟수", "요청 20건 중 실패 건수"], ["포아송", "일정한 비율로 드물게 일어나는 일의 횟수", "1분에 들어오는 요청 수"],
                                             ["정규", "가운데로 모인 종 모양", "측정 오차, ⑤의 표본 평균"], ["t, χ², F", "검정에 쓰는 분포 (다음 쪽)", "t검정, 교차분석, 분산분석"]])
           + K.out("s3") + K.out("f_dist")
           + K.code_table([("req.groupby(\"priority\")[\"failed\"].mean()", "우선순위마다 실패 비율 = P(실패 | 우선순위)"), ("stats.poisson.pmf(k, lam)", "포아송이면 값이 k일 확률"),
                           ("r.var() / r.mean()", "포아송이면 1 근처")]), None),
    K.step(4, "표본 뽑기: 무작위, 편한 대로, 층화", "가상의 1년치 10만 건(모집단)에서 100건씩 세 가지 방법으로 2,000번 뽑아, 표본 평균이 진짜 평균과 얼마나 다른지 봐요.",
           "<p>궁금한 전체가 <b>모집단</b>, 실제로 손에 쥔 일부가 <b>표본</b>이에요. 모집단의 값(<b>모수</b>, 예: 진짜 평균)은 모르고, 표본에서 계산한 값(<b>통계량</b>)으로 짐작해요.</p>"
           + K.table(["뽑는 방법", "어떻게", "특징"], [["단순 무작위", "모두 같은 확률로", "치우침 없음"], ["계통", "1,000번째마다", "간편, 주기가 있으면 위험"], ["층화", "종류별로 나눠 각각 무작위", "치우침 없고 덜 흔들림"],
                                                  ["군집", "부서 몇 개를 통째로", "싸지만 더 흔들림"], ["편한 대로, 판단, 할당", "손에 닿는 대로, 전문가가 골라, 비율만 맞춰", "확률을 몰라 치우침을 알 수 없음"]])
           + K.out("s4") + K.out("f_samp")
           + "<p><b>무작위 추출과 무작위 할당</b>은 다른 일이에요. 무작위로 <b>뽑아야</b> 결과를 모집단 전체로 넓힐 수 있고(일반화), 처치를 무작위로 <b>나눠 줘야</b> \"그 처치 때문\"이라고 말할 수 있어요(인과). 지켜보기만 한 자료에서는 둘 다 조심해야 해요.</p>"
           + K.code_table([("pop_ms", "가상의 모집단 10만 건 (종류마다 다른 처리 시간 × 로그정규 흔들림)"), ("simple_random, convenience, stratified", "세 가지 뽑는 방법")]), None),
    K.step(5, "표본 평균의 분포: 중심극한정리", "같은 모집단에서 n건 표본을 5,000번 뽑아, 표본 평균이 어떻게 퍼지는지 n을 바꿔 가며 봐요.",
           K.out("s5") + K.out("f_clt")
           + K.code_table([("pop_ms[rng.integers(N, size=n)].mean()", "n건을 무작위로 뽑아 평균"), ("pop_ms.std() / np.sqrt(n)", "표준오차 σ ÷ √n")]), None),
    K.step(6, "신뢰구간: 95%의 뜻", "n건 표본마다 95% 신뢰구간을 만들고, 그 구간이 진짜 평균을 담은 비율을 2,000번 세요. z와 t 두 가지로 만들어 비교해요.",
           "<p>하나의 값으로 짐작하는 게 <b>점추정</b>(표본 평균), 범위로 짐작하는 게 <b>구간추정</b>이에요. 평균의 95% <b>신뢰구간</b>은 <code>표본 평균 ± (기준값) × 표준오차</code>예요. 기준값은 정규분포면 1.96이고, 모집단 표준편차를 모르고 표본으로 짐작할 때는 그보다 큰 t분포의 값을 써요(n = 5면 2.78, n = 30이면 2.05).</p>"
           + K.out("s6") + K.out("f_ci")
           + "<p>분산에 대한 신뢰구간은 χ²분포로 만들어요. 그리고 구간을 반으로 좁히려면 ⑤처럼 표본이 4배 필요해요.</p>"
           + K.code_table([("stats.t.ppf((1 + level) / 2, n - 1)", "자유도 n − 1인 t분포에서 위쪽 2.5% 지점"), ("lo <= true_mean <= hi", "진짜 평균을 담았나 (실제 분석에서는 모르는 값)")]), "s6"),
]
practice = [("데이터가 깨끗하지 않다", "실무에서 데이터는 절대 깨끗하지 않고, 분석 시간의 대부분이 정리에 들어가요. 빈 칸, 단위, 중복, 이상한 값을 먼저 세어 보고 기록해요(①)."),
            ("평균만 보고하지 말기", "평균과 함께 중앙값, 표준편차나 사분위 범위, 건수를 같이 적어요. 치우친 자료(재시도 같은)는 그림을 먼저 봐요."),
            ("표본이 대표성이 있나", "누가 빠졌는지 먼저 따져요(④의 편한 대로). 설문 응답자, 로그에 남은 사용자, 남아 있는 고객은 이미 걸러진 집단이에요."),
            ("구간으로 말하기", "\"평균 1,278ms\"보다 \"95% 신뢰구간 1,150~1,405ms\"가 얼마나 믿을 만한지까지 알려 줘요.")]
libs = [("①~③", "<code>pandas</code>, <code>scipy.stats</code> (skew, kurtosis, gmean, hmean, poisson)", "요약, 분포"), ("④~⑥", "<code>numpy.random</code>, <code>scipy.stats.t</code>", "표본 뽑기, 신뢰구간")]
K.build(ROOT / "topics" / "model" / "stats-describe.html", title="요약하고 추정하기", h1="요약하고 추정하기: 기술통계와 신뢰구간", track="모델",
        crumb="모델 &gt; 데이터 분석 기초 1/2쪽 &nbsp;|&nbsp; 다음: <a href=\"stats-tests.html\" style=\"color:inherit\">가설 검정과 회귀 분석</a>",
        one_line="자료는 척도부터 정하고, <b>가운데, 퍼짐, 모양</b>으로 요약해요. 일부(표본)로 전체(모집단)를 짐작할 때는 <b>무작위로</b> 뽑아야 하고, 그 짐작의 흔들림은 <b>표준오차 σ ÷ √n</b>, 범위는 <b>신뢰구간</b>으로 말해요.",
        why=why, whole=whole, steps=steps, code_file="examples/stats/describe_estimate.py", practice=practice, libs=libs,
        libs_note="<code>simple_random</code>, <code>convenience</code>, <code>stratified</code>, <code>interval</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"stats-tests.html\">가설 검정과 회귀 분석</a>", "차이가 우연인지 따지기"), ("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "같은 데이터로 예측 모델"), ("<a href=\"ml-learning-theory.html\">학습 이론</a>", "Hoeffding, 표본 수와 믿음")],
        sources=["다루는 범위: 부트캠프 머신러닝 기초 앞에 두면 좋은 통계 기초예요. 수업에서 따로 다루지 않았어요",
                 "함께 정리한 자료: 경희대 특강(2025년 1월) \"통계 기초 및 가설 검정\"과 \"SPSS 실습\"(통계학의 기본 개념, 척도, 표본추출 방법, 중심성향과 산포, 왜도와 첨도, 확률의 종류, 이산과 연속 확률분포, 점추정과 구간추정, 1차와 2차 자료, 설문지 작성), \"Data Science\" 특강(데이터 과학의 불편한 진실, 관측과 실험 자료, 무작위 추출과 할당, 작업 흐름과 팀 역할), 경희대 Pattern Recognition and Machine Learning 1~2강(ML 생애주기 6단계와 직군, 파이썬 분석 도구). 값과 그림은 이 예제로 계산했어요",
                 "데이터: <code>examples/stats/data/requests.csv</code> (머신러닝 기초와 같은 가상의 업무 요청 100건)와 코드가 만든 가상의 1년치 10만 건",
                 "결과 값과 그림: <code>tools/pages/stats_describe.py</code>가 <code>examples/stats/describe_estimate.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
