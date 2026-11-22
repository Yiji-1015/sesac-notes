"""모델 > 데이터 분석 기초 2쪽: 가설 검정과 회귀 분석. examples/stats/tests_regression.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/stats_tests.py
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
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "stats"
code = S.parse((EX / "tests_regression.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="tests_regression.py")
ENV = f"Python {sys.version.split()[0]}, pandas {pd.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__} (seed 0)"
f1 = lambda x: f"{float(x):,.1f}"  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731


def pv(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


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


def coef_rows(coef):
    return [[f"<code>{i}</code>", f"{r['계수']:,.4f}", f"{r['표준오차']:,.4f}", f"{r['t']:.2f}", pv(r["p"])] for i, r in coef.iterrows()]


outs = {}
req = ns["req"]
# ---------------- ①
t1, pw = ns["type1"], ns["power"]
S.claim(all(0.035 < v < 0.065 for v in t1.values()), "① 차이가 없으면 5% 근처에서만 유의")
S.claim(pw[(10, 200)] < 0.15 and pw[(100, 400)] > 0.99 and ns["need_n"] > 100, "① 10건으로 200 차이는 15%도 못 찾고, 100건으로 400 차이는 99% 넘게 찾음")
outs["s1"] = S.table("① 표준편차 600인 두 집단을 n건씩 4,000번 비교 (α = 0.05)", ["n (집단당)", "진짜 차이 0: \"차이 있다\" 비율", "진짜 차이 200: 찾은 비율", "진짜 차이 400: 찾은 비율"],
                     [[n, f"{t1[n]:.1%}", f"{pw[(n, 200)]:.1%}", f"{pw[(n, 400)]:.1%}"] for n in (10, 30, 100)],
                     ["차이가 없을 때 \"있다\"고 잘못 말하는 비율(1종 오류)은 n과 상관없이 정한 α = 5% 근처예요. α는 우리가 고르는 값이에요.",
                      f"차이가 있을 때 찾아내는 비율이 <b>검정력</b>(1 − 2종 오류)이에요. 200의 차이는 10건씩이면 {pw[(10, 200)]:.0%}만 찾아요. 이 차이를 80%로 찾으려면 집단당 {ns['need_n']}건이 필요해요. 그래서 \"유의하지 않다\"는 \"차이가 없다\"가 아니라 \"이 건수로는 못 찾았다\"예요."])

# ---------------- ②
np_, lev = ns["normal_p"], ns["levene_p"]
S.claim(np_[0] > 0.05 and np_[1] < 0.05 and lev > 0.05 and ns["student"].pvalue < 1e-6 and ns["mann"].pvalue < 1e-5, "② 실패 쪽은 정규, 성공 쪽은 정규 아님, 등분산, 세 검정 모두 매우 유의")
fm, om = ns["failed_ms"], ns["ok_ms"]
outs["s2"] = S.table(f"② 실패 {len(fm)}건 대 성공 {len(om)}건의 처리 시간", ["확인", "통계량", "p", "읽기"],
                     [["평균 (실패 / 성공)", f"{fm.mean():,.0f} / {om.mean():,.0f}ms", "", f"차이 {fm.mean() - om.mean():,.0f}ms"],
                      ["정규성 (Shapiro-Wilk), 실패 쪽", "", pv(np_[0]), "정규분포와 다르다고 할 수 없음"],
                      ["정규성 (Shapiro-Wilk), 성공 쪽", "", pv(np_[1]), "정규분포와 다름"],
                      ["등분산 (Levene)", "", pv(lev), "분산이 다르다고 할 수 없음"],
                      ["t검정 (등분산 가정)", f"t = {ns['student'].statistic:.2f}", pv(ns["student"].pvalue), "평균이 다름"],
                      ["Welch t검정 (등분산 가정 안 함)", f"t = {ns['welch'].statistic:.2f}", pv(ns["welch"].pvalue), "평균이 다름"],
                      ["Mann-Whitney U (순위)", "", pv(ns["mann"].pvalue), "분포가 한쪽으로 밀려 있음"],
                      ["효과 크기 (Cohen의 d)", f"{ns['cohen_d']:.2f}", "", f"표준편차의 {ns['cohen_d']:.1f}배 차이, 큰 편"]],
                     ["성공 쪽은 정규성 검정에서 걸렸지만 70건이라 중심극한정리로 평균은 거의 정규로 다룰 수 있고, 순위만 보는 Mann-Whitney도 같은 결론이에요. 결론이 방법에 따라 바뀌지 않으면 믿을 만해요.",
                      "p값은 \"차이가 없다면 이만큼 큰 차이가 우연히 나올 확률\"이에요. 아주 작다는 건 차이가 있다는 뜻이지 <b>차이가 크다</b>는 뜻이 아니에요. 그래서 효과 크기를 같이 적어요. 다만 여기서는 실패한 요청이 재시도하며 오래 걸린 것이라, 오래 걸려서 실패했다고 거꾸로 읽으면 안 돼요."])

# ---------------- ③
an, tk = ns["anova"], ns["tukey"]
S.claim(an.pvalue > 0.05 and abs(ns["f_by_hand"] - an.statistic) < 1e-9 and (tk.pvalue > 0.05).all(), "③ 분산분석 유의하지 않음, 손 계산 F가 같고, 사후 분석도 모두 유의하지 않음")
names3 = list(ns["by_type"])
pairs = [(i, j) for i in range(len(names3)) for j in range(i + 1, len(names3))]
outs["s3"] = S.table("③ 작업 종류별 처리 시간 (일원 분산분석)", ["무엇", "값"],
                     [*[[f"{k} 평균 (n)", f"{v.mean():,.0f}ms ({len(v)})"] for k, v in ns["by_type"].items()],
                      ["집단 사이 제곱합 / 집단 안 제곱합", f"{ns['ss_between']:,.0f} / {ns['ss_within']:,.0f}"],
                      ["F = (사이 ÷ (k − 1)) ÷ (안 ÷ (n − k))", f"{ns['f_by_hand']:.3f} (scipy {an.statistic:.3f})"], ["p", pv(an.pvalue)], ["등분산 (Levene) p", pv(ns["levene_type"])],
                      *[[f"Tukey: {names3[i]} − {names3[j]}", f"{tk.statistic[i, j]:+,.0f}ms, p = {pv(tk.pvalue[i, j])}"] for i, j in pairs]],
                     [f"평균은 qa가 summary보다 {ns['by_type']['qa'].mean() - ns['by_type']['summary'].mean():,.0f}ms 길지만, 집단 안 흩어짐이 워낙 커서 F가 {an.statistic:.2f}에 그쳐요. 이 100건으로는 종류마다 다르다고 말할 수 없어요.",
                      "t검정을 세 쌍에 따로 하면 우연히 하나쯤 걸릴 확률이 커져요(⑦). 그래서 먼저 분산분석으로 \"어딘가 다른가\"를 보고, 다르면 Tukey 같은 사후 분석으로 어느 쌍인지 봐요. 여기서는 분산분석도 사후 분석도 모두 유의하지 않아요."])
fig, axs = plt.subplots(1, 2, figsize=(11.6, 3.6))
axs[0].boxplot([om, fm], tick_labels=["성공", "실패"], widths=.5, patch_artist=True, boxprops=dict(facecolor=C["soft"]), medianprops=dict(color=FAIL, lw=2))
axs[0].set(ylabel="처리 시간 (ms)", title="② 실패 여부: 뚜렷한 차이")
axs[1].boxplot(list(ns["by_type"].values()), tick_labels=names3, widths=.5, patch_artist=True, boxprops=dict(facecolor=C["soft"]), medianprops=dict(color=FAIL, lw=2))
axs[1].set(ylabel="처리 시간 (ms)", title="③ 작업 종류: 차이를 말할 수 없음")
P_BOX = fig_save(fig, "group_compare")
outs["f_box"] = chart("집단별 처리 시간 (상자: 가운데 50%, 분홍 선: 중앙값)", P_BOX, "", ["왼쪽은 두 상자가 거의 겹치지 않아요. 오른쪽은 세 상자가 크게 겹쳐요."])

# ---------------- ④
pe, sm = ns["pearson"], ns["spearman"]
ch = ns["chi2"]
simp, ov = ns["simpson"], ns["overall"]
S.claim(pe.pvalue < 0.01 and ch.pvalue > 0.05, "④ 입력 길이와 처리 시간 상관은 유의, 우선순위와 실패는 유의하지 않음")
S.claim(all(simp.loc[(simp["모델"] == "A") & (simp["요청"] == r_), "성공률"].iloc[0] > simp.loc[(simp["모델"] == "B") & (simp["요청"] == r_), "성공률"].iloc[0] for r_ in ("쉬움", "어려움")) and ov.loc["A", "성공률"] < ov.loc["B", "성공률"], "④ 심슨: 각 칸은 A가, 합치면 B가 높다")
outs["s4"] = S.table("④ 관계 보기", ["무엇", "값", "p"],
                     [["입력 길이와 처리 시간: Pearson r (곧은 관계)", f"{pe.statistic:.3f}", pv(pe.pvalue)], ["같은 둘: Spearman ρ (순위 관계)", f"{sm.statistic:.3f}", pv(sm.pvalue)],
                      ["우선순위 × 실패: χ² (교차분석)", f"{ch.statistic:.2f} (자유도 {ch.dof})", pv(ch.pvalue)]],
                     [f"상관 {pe.statistic:.2f}는 유의하지만 약해요. r²로 보면 처리 시간 흩어짐의 {pe.statistic ** 2:.0%}만 입력 길이로 설명돼요. 상관은 인과가 아니에요.",
                      "χ² 검정은 \"두 범주형 변수가 독립이라면 각 칸에 몇 건이 기대되나\"와 실제 건수를 비교해요. 앞 쪽 ③의 \"high의 실패율이 높아 보였던 것\"은 우연으로 설명될 만한 차이예요."])
outs["s4s"] = S.table("심슨의 역설: 가상의 두 모델", ["모델", "쉬운 요청", "어려운 요청", "합치면"],
                      [[m, *[f"{r_['성공']}/{r_['전체']} = {r_['성공률']:.1%}" for _, r_ in simp[simp["모델"] == m].iterrows()], f"{ov.loc[m, '성공']}/{ov.loc[m, '전체']} = {ov.loc[m, '성공률']:.1%}"] for m in ("A", "B")],
                      ["A는 쉬운 요청에서도 어려운 요청에서도 B보다 성공률이 높아요. 그런데 합치면 B가 훨씬 높아요. A가 어려운 요청을 훨씬 많이(300건) 맡았기 때문이에요.",
                       "어떤 요청을 맡았는지(혼란 변수)를 무시하고 합치면 결론이 뒤집혀요. 관측 자료로 비교할 때는 이런 변수로 나눠 보거나, 처음부터 무작위로 나눠 맡겨야(실험, A/B 검정) 해요."])

# ---------------- ⑤
c5, fit5, v5 = ns["coef5"], ns["fit5"], ns["vif5"]
S.claim(c5.loc["input_chars", "p"] < 0.05 and (c5.drop(["절편", "input_chars"])["p"] > 0.05).all() and fit5["F의 p"] > 0.05 and max(v5.values()) < 2 and abs(fit5["Durbin-Watson"] - 2) < 0.2, "⑤ 입력 길이만 유의, 회귀식 전체는 유의하지 않음, VIF < 2, DW ≈ 2")
outs["s5"] = S.table(f"⑤ 처리 시간 회귀 (n = {fit5['n']}, 빈 칸 있는 행 제외)", ["변수", "계수", "표준오차", "t", "p"], coef_rows(c5),
                     [f"R² = {fit5['R²']:.3f}, 수정 R² = {fit5['수정 R²']:.3f}, F = {fit5['F']:.2f} (p = {pv(fit5['F의 p'])}), Durbin-Watson = {fit5['Durbin-Watson']:.2f}, VIF 최대 {max(v5.values()):.2f}",
                      f"입력 길이 1자가 늘면 처리 시간이 {c5.loc['input_chars', '계수']:.2f}ms 늘어요(다른 변수가 같을 때). 이것만 유의해요. 작업 종류와 우선순위 계수는 0과 구별되지 않아요.",
                      f"R²가 {fit5['R²']:.2f}로, 이 변수들로는 처리 시간 흩어짐의 {fit5['R²']:.0%}만 설명돼요. 변수를 넣을수록 R²는 저절로 커지니, 변수 수를 깎아 준 수정 R²({fit5['수정 R²']:.3f})를 봐요. F 검정은 \"모든 계수가 0\"을 기각하지 못했어요(p = {pv(fit5['F의 p'])}). 입력 길이 하나는 유의한데 전체는 아닌 건, 쓸모없는 변수 넷이 함께 들어가 힘을 나눠 먹었기 때문이에요.",
                      "VIF가 모두 2 아래라 변수끼리 서로를 설명하는 문제(다중공선성)는 없어요. Durbin-Watson이 2 근처라 잔차가 이웃끼리 닮지도 않았어요."])

# ---------------- ⑥
md = ns["moderation"]
vr, vc = ns["vif_raw"], ns["vif_centered"]
inter = md["3단계 (+ 상호작용)"][0].loc["len_x_high"]
S.claim(inter["p"] > 0.05 and max(vr.values()) > 5 and max(vc.values()) < 1.1 and abs(ns["coef_mc"].loc["mc_len_x_high", "계수"] - inter["계수"]) < 1e-9, "⑥ 상호작용 유의하지 않음, 중심화 전 VIF > 5, 뒤 < 1.1, 상호작용 계수는 같음")
outs["s6"] = S.table("⑥ 우선순위 high가 입력 길이의 효과를 바꾸나 (위계적 회귀 3단계)", ["단계", "R²", "추가된 변수의 계수", "그 p"],
                     [[k, f3(f["R²"]), f"{c.iloc[-1]['계수']:,.4f} (<code>{c.index[-1]}</code>)", pv(c.iloc[-1]["p"])] for k, (c, f) in md.items()],
                     [f"3단계에서 상호작용항의 계수가 {inter['계수']:.3f}(high면 1자당 {-inter['계수']:.2f}ms 덜 늘어남)이지만 p = {inter['p']:.2f}라, 이 데이터로는 조절 효과가 있다고 말할 수 없어요.",
                      "조절 효과는 이렇게 단계를 나눠 넣어서, 마지막 단계의 상호작용항이 유의한지와 R²가 얼마나 늘었는지로 판단해요."])
outs["s6v"] = S.table("상호작용항을 넣었을 때의 VIF", ["변수", "그대로 곱했을 때", "평균 중심화 후"],
                      [[f"{a} / {b}", f"{vr[a]:.2f}", f"{vc[b]:.2f}"] for a, b in zip(vr, vc)],
                      ["그대로 곱한 상호작용항은 원래 두 변수와 많이 닮아서 VIF가 5를 넘어요. 각 변수에서 평균을 빼고 곱하면(평균 중심화) VIF가 1 근처로 내려가요.",
                       f"상호작용항의 계수와 p는 중심화 전후로 똑같아요({ns['coef_mc'].loc['mc_len_x_high', '계수']:.4f}). 바뀌는 건 나머지 두 변수의 계수 뜻이에요. 중심화하면 \"다른 변수가 평균일 때의 효과\"가 돼서 읽기 쉬워져요."])
fig, ax = plt.subplots(figsize=(7.8, 3.8))
c3 = md["3단계 (+ 상호작용)"][0]["계수"]
for flag, col, nm in ((0.0, OK, "high 아님"), (1.0, FAIL, "high")):
    sub = ns["reg"][ns["reg"]["is_high"] == flag]
    ax.scatter(sub["input_chars"], sub["elapsed_ms"], s=18, color=col, alpha=.6, label=f"{nm} ({len(sub)}건)")
    xs = np.linspace(ns["reg"]["input_chars"].min(), ns["reg"]["input_chars"].max(), 50)
    ax.plot(xs, c3["절편"] + c3["input_chars"] * xs + c3["is_high"] * flag + c3["len_x_high"] * xs * flag, color=col, lw=2.4)
ax.set(xlabel="입력 길이 (자)", ylabel="처리 시간 (ms)", title="3단계 회귀식: 두 기울기가 다르게 그려지지만 유의하지 않아요")
ax.legend(fontsize=9)
P_MOD = fig_save(fig, "moderation")
outs["f_mod"] = chart("우선순위에 따른 기울기", P_MOD, "", [f"분홍 선(high)이 보라 선보다 기울기가 완만해요. 하지만 high가 {int(ns['reg']['is_high'].sum())}건뿐이고 점이 크게 흩어져 있어서, 이 차이는 우연으로도 충분히 생겨요."])

# ---------------- ⑦
S.claim(0.6 < ns["any_raw"] < 0.7 and ns["any_bonf"] < 0.07, "⑦ 20개 중 하나라도 유의가 60~70%, Bonferroni면 7% 아래")
trials = ns["trials"]
outs["s7"] = S.box("⑦ 아무 차이 없는 지표 20개를 매번 검정 (2,000번)", [(f"20개 중 하나라도 p < 0.05 → {ns['any_raw']:.1%}", 1), (f"기준을 0.05 ÷ 20 = 0.0025로 (Bonferroni) → {ns['any_bonf']:.1%}", 2, "newf"),
                                                               f"한 번에 \"유의\"가 나온 지표 수의 평균 → {(trials < 0.05).sum(axis=1).mean():.2f}개"],
                   ["하나씩은 5%라도 20번 보면 1 − 0.95²⁰ ≈ 64%의 확률로 하나는 걸려요. 지표를 많이 보거나 데이터를 여러 방식으로 잘라 보다가 유의한 것만 보고하면, 없는 차이를 \"발견\"하게 돼요(p-해킹).",
                    "미리 정한 하나의 지표로 판단하거나, 검정 수만큼 기준을 엄격하게 해요. Bonferroni는 가장 단순하고 보수적인 방법이고, 덜 보수적인 Holm, 발견 중 틀린 비율을 조절하는 Benjamini-Hochberg도 있어요."])

flow = [[1, "검정의 뼈대", "1종 오류, 검정력", "몇 건이 필요한가"], [2, "두 집단", "t, Welch, 순위", "실패가 더 오래"], [3, "셋 이상", "분산분석, Tukey", "차이 없음"],
        [4, "관계", "상관, χ², 심슨", "합치면 뒤집힘"], [5, "회귀", "계수, R², F, VIF", "입력 길이만"], [6, "조절 효과", "상호작용, 중심화", "유의하지 않음"], [7, "여러 번 검정", "20개 중 하나", "Bonferroni"]]
S.save(ROOT / "topics" / "model" / "stats-tests-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p><a href=\"stats-describe.html\">1쪽</a>에서 표본으로 모집단을 짐작하면 흔들림이 있다는 걸 봤어요. 그러면 \"실패한 요청이 평균 750ms 더 걸렸다\"는 진짜 차이일까요, 이 100건에서만 우연히 나온 걸까요? 이걸 정해진 절차로 판단하는 게 <b>가설 검정</b>이고, 여러 변수의 영향을 한꺼번에 재는 게 <b>회귀 분석</b>이에요. 논문과 보고서의 통계가 거의 다 이 둘이에요.</p>"""
why += K.table(["알고 싶은 것", "독립변수", "종속변수", "방법 (이 쪽)"], [["두 집단 평균이 다른가", "명목 (2개)", "등간, 비율", "t검정 ②"], ["셋 이상 집단 평균이 다른가", "명목 (3개 이상)", "등간, 비율", "분산분석 ③"],
                                                                 ["두 숫자가 함께 움직이나", "등간, 비율", "등간, 비율", "상관 ④"], ["두 범주가 관련 있나", "명목, 서열", "명목, 서열", "교차분석 (χ²) ④"],
                                                                 ["여러 변수의 영향을 함께", "등간, 비율 (+ 더미)", "등간, 비율", "회귀 ⑤⑥"], ["결과가 0이나 1", "아무거나", "명목 (2개)", "로지스틱 회귀 (<a href=\"ml-regression-classification.html\">회귀와 분류</a> ⑤)"]])
whole = """<p>검정의 뼈대 → 두 집단 → 셋 이상 → 관계 → 회귀 → 조절 효과 → 여러 번 검정할 때. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/stats/tests_regression.py</code> 하나예요. 회귀는 numpy로 식을 직접 계산해서, SPSS나 statsmodels가 내는 표의 숫자가 어디서 나오는지 보여요.</p>"""
steps = [
    K.step(1, "검정의 뼈대: 1종 오류와 검정력", "정답을 아는 가상 실험으로, 차이가 없을 때 \"있다\"고 잘못 말하는 비율과 차이가 있을 때 찾아내는 비율을 재요.",
           "<p>가설 검정은 이렇게 해요.</p><ol><li><b>귀무가설</b> H₀(\"차이가 없다\")와 <b>대립가설</b> H₁(\"차이가 있다\")을 세워요.</li><li><b>유의수준</b> α(보통 0.05)를 미리 정해요.</li>"
           "<li>데이터로 검정통계량(예: t)을 계산하고, H₀가 맞다면 그만큼 극단적인 값이 나올 확률(<b>p값</b>)을 구해요.</li><li>p < α면 H₀를 기각해요(\"차이가 있다\"). 아니면 기각하지 못해요.</li></ol>"
           "<p>판단은 두 가지로 틀릴 수 있어요. 차이가 없는데 있다고 하는 <b>1종 오류</b>(확률 α)와, 차이가 있는데 못 찾는 <b>2종 오류</b>(확률 β)예요. 1 − β가 <b>검정력</b>이에요. \"한쪽으로만 다른가\"를 볼 때는 단측 검정, \"다른가\"를 볼 때는 양측 검정을 써요.</p>"
           + K.out("s1")
           + K.code_table([("experiment(n, diff)", "진짜 차이 diff인 두 집단에서 n건씩 뽑아 t검정을 4,000번, 유의한 비율"),
                           ("2 × ((z₀.₉₇₅ + z₀.₈) × σ ÷ 차이)²", "검정력 80%, α = 0.05로 그 차이를 찾는 데 필요한 집단당 건수 (정규 근사)")]), "s1"),
    K.step(2, "두 집단 비교: t검정", "실패한 요청과 성공한 요청의 처리 시간을 비교해요. 정규성과 등분산을 먼저 확인하고, t검정 둘과 순위 검정 하나로 같은 결론이 나오는지 봐요.",
           "<p><b>독립 표본 t검정</b>은 서로 다른 두 집단의 평균을 비교해요. t = (평균 차이) ÷ (차이의 표준오차)예요. 가정이 둘 있어요. 각 집단이 정규분포를 따르고(Shapiro-Wilk로 확인, 표본이 크면 중심극한정리로 느슨해져요), 분산이 같다(Levene으로 확인). 분산이 다르면 <b>Welch</b> t검정을 써요. 요즘은 처음부터 Welch를 쓰는 경우가 많아요. 정규성이 크게 어긋나면 값 대신 순위를 비교하는 Mann-Whitney U를 써요.</p>"
           "<p>같은 대상을 전과 후로 두 번 잰 경우(예: 같은 요청을 옛 서버와 새 서버로)는 <b>대응 표본 t검정</b>이에요. 한 사람 한 사람의 차이를 하나의 표본으로 봐서, 사람마다 다른 기본값이 지워져 더 민감해요.</p>"
           + K.out("s2")
           + K.code_table([("stats.shapiro, stats.levene", "정규성, 등분산 확인"), ("stats.ttest_ind(a, b, equal_var=False)", "Welch t검정"), ("stats.mannwhitneyu(a, b)", "순위로 비교"),
                           ("(평균 차이) ÷ 합친 표준편차", "Cohen의 d: 0.2 작음, 0.5 중간, 0.8 큼이 흔한 기준")]), "s2"),
    K.step(3, "셋 이상 비교: 분산분석과 사후 분석", "작업 종류 세 가지의 처리 시간이 다른지 일원 분산분석으로 보고, F를 손으로도 계산해요.",
           "<p><b>분산분석</b>(ANOVA)은 \"집단 평균들이 서로 떨어진 정도\"(집단 사이 분산)가 \"집단 안에서 흩어진 정도\"(집단 안 분산)에 비해 큰지를 F = 사이 ÷ 안으로 재요. 가정은 t검정과 같아요(정규성, 등분산, 독립). 요인이 하나면 일원, 둘이면 이원 분산분석이에요.</p>"
           "<p>F가 유의하면 \"어딘가 다르다\"까지만 알아요. 어느 쌍인지는 <b>사후 분석</b>으로 봐요. 등분산이면 Tukey나 Scheffé, 아니면 Games-Howell을 흔히 써요.</p>"
           + K.out("s3") + K.out("f_box")
           + K.code_table([("stats.f_oneway(*groups)", "일원 분산분석"), ("stats.tukey_hsd(*groups)", "Tukey 사후 분석: 모든 쌍의 차이와 p"), ("ss_between, ss_within", "F를 손으로: 제곱합을 자유도로 나눈 비")]), None),
    K.step(4, "관계 보기: 상관, 교차분석, 심슨의 역설", "입력 길이와 처리 시간의 상관, 우선순위와 실패의 교차분석을 하고, 합쳐 보면 결론이 뒤집히는 가상의 예를 봐요.",
           "<p><b>공분산</b>은 두 값이 함께 움직이는 정도, <b>상관계수</b>는 그걸 −1~1로 맞춘 거예요. Pearson은 곧은 관계를, Spearman은 순위가 함께 오르내리는지를 봐요(서열 척도나 휘어진 관계에). <b>교차분석</b>(χ² 검정)은 두 범주형 변수의 표에서 독립이라면 기대되는 건수와 실제 건수를 비교해요.</p>"
           + K.out("s4") + K.out("s4s")
           + K.code_table([("stats.pearsonr, stats.spearmanr", "상관계수와 p"), ("pd.crosstab(...)", "두 범주형 변수의 건수 표"), ("stats.chi2_contingency(table)", "χ² 독립성 검정 (기대 건수도 돌려줘요)")]), None),
    K.step(5, "회귀 분석 직접 계산: 계수, t, p, R², F, VIF, Durbin-Watson", "처리 시간을 입력 길이, 허용 도구 수, 작업 종류(더미), 우선순위 high로 설명하는 회귀를 numpy로 풀고, SPSS가 내는 표를 하나씩 만들어요.",
           "<p>회귀 계수는 <a href=\"ml-regression-classification.html\">회귀와 분류</a> ④의 최소제곱과 같아요. 통계에서는 여기에 \"이 계수가 0이 아니라고 할 수 있나\"를 붙여요. 계수의 표준오차로 t = 계수 ÷ 표준오차, 그 p를 구해요.</p>"
           + K.table(["표의 항목", "뜻", "흔한 기준"], [["R², 수정 R²", "종속변수 흩어짐 중 설명된 비율 (수정은 변수 수만큼 깎음)", "높을수록 좋지만 분야마다 다름"], ["F와 그 p", "\"모든 계수가 0\"을 기각하나", "p < 0.05"],
                                                    ["계수 (β)", "그 변수가 1 늘 때 종속변수 변화 (나머지가 같을 때)", "p < 0.05면 0과 다름"], ["VIF", "그 변수가 다른 변수들로 설명되는 정도", "10 (엄격하면 5) 아래"],
                                                    ["Durbin-Watson", "잔차가 이웃끼리 닮았나 (시간 순 자료에서 중요)", "2 근처"]])
           + "<p>작업 종류처럼 범주가 k개인 변수는 <b>더미 변수</b> k − 1개로 넣어요(하나는 기준). k개를 다 넣으면 서로 완전히 겹쳐서 풀리지 않아요. 변수에는 원인 역할의 <b>독립변수</b>, 결과인 <b>종속변수</b> 말고도, 중간을 잇는 <b>매개변수</b>, 관계의 세기를 바꾸는 <b>조절변수</b>(⑥), 영향을 빼고 보려고 넣는 <b>통제변수</b>가 있어요.</p>"
           + K.out("s5")
           + K.code_table([("np.linalg.lstsq(X, y)", "최소제곱 계수"), ("sigma2 * np.linalg.inv(X.T @ X)", "계수의 분산 행렬 (대각선의 제곱근이 표준오차)"), ("2 * stats.t.sf(|t|, n - p)", "양측 p"),
                           ("np.sum(np.diff(resid) ** 2) / (resid @ resid)", "Durbin-Watson"), ("1 / (1 - R²ⱼ)", "VIF: j번째 변수를 나머지로 회귀한 R²로")]), "s5"),
    K.step(6, "조절 효과: 상호작용과 평균 중심화", "우선순위 high가 \"입력 길이가 늘 때 처리 시간이 느는 정도\"를 바꾸는지 세 단계 회귀로 보고, 평균 중심화가 무엇을 바꾸는지 봐요.",
           "<p>X가 Y에 주는 영향이 Z에 따라 달라지면 Z를 <b>조절변수</b>라고 해요. 회귀에 X × Z(<b>상호작용항</b>)를 넣어 그 계수가 유의한지 봐요. 보통 세 단계로 넣어요. 1단계 X만, 2단계 X와 Z, 3단계 X, Z, X × Z. 3단계의 상호작용항이 유의하고 R²가 늘면 조절 효과가 있다고 해요.</p>"
           + K.table(["X의 계수", "상호작용 계수", "뜻"], [["+", "+", "Z가 클수록 X의 좋은 영향이 더 커짐"], ["+", "−", "Z가 클수록 X의 좋은 영향이 줄어듦"], ["−", "−", "Z가 클수록 X의 나쁜 영향이 더 커짐"], ["−", "+", "Z가 클수록 X의 나쁜 영향이 줄어듦"]])
           + K.out("s6") + K.out("s6v") + K.out("f_mod")
           + K.code_table([("reg[\"input_chars\"] * reg[\"is_high\"]", "상호작용항"), ("x - x.mean()", "평균 중심화"), ("ols(...) 세 번", "1, 2, 3단계 (위계적 회귀)")]), None),
    K.step(7, "여러 번 검정하면: 20개 중 하나는 걸려요", "아무 차이도 없는 지표 20개를 매번 t검정해서, 하나라도 \"유의\"가 나오는 비율을 재요.",
           K.out("s7")
           + "<p>머신러닝에서도 같아요. 하이퍼파라미터 후보를 많이 시험하고 가장 좋은 것만 보고하면 실제보다 좋아 보여요(<a href=\"ml-learning-theory.html\">학습 이론</a> ③). A/B 검정은 무작위로 나눠 맡기는(②의 관측 자료 문제를 없애는) 실험이지만, 지표를 여럿 보거나 결과를 보며 중간에 멈추면 이 문제가 그대로 생겨요.</p>"
           + K.code_table([("(trials < 0.05).any(axis=1)", "20개 중 하나라도 유의했나"), ("0.05 / 20", "Bonferroni: 기준을 검정 수로 나누기")]), "s7"),
]
practice = [("결과를 쓸 때", "검정 이름, 통계량, 자유도, p, 효과 크기, 표본 수를 함께 적어요. 예: Welch t(62.2) = 6.72, p < 0.001, d = 1.39, n = 30 대 70."),
            ("유의하지 않을 때", "\"차이가 없다\"가 아니라 \"이 자료로는 차이를 확인하지 못했다\"예요. 검정력이 충분했는지(①) 같이 봐요."),
            ("관측 자료로 원인 말하기", "실패와 처리 시간처럼 어느 쪽이 원인인지, 다른 변수(심슨의 역설)가 끼었는지 먼저 따져요. 인과를 말하려면 무작위 할당 실험이 가장 확실해요."),
            ("실무 분석의 대부분은", "딥러닝이 아니라 이런 검정과 일반화 선형 모형(선형 회귀, 로지스틱 회귀)으로 풀려요. 쓰는 사람은 방법보다 결론과 그 확실성을 궁금해해요."),
            ("SPSS로 할 때", "분석 → 평균 비교 → 독립표본 T검정에서 Levene 결과를 보고 \"등분산 가정됨/가정되지 않음\" 행을 골라 읽어요. 회귀는 선형 회귀 → 통계량에서 공선성 진단과 Durbin-Watson을 켜요.")]
libs = [("①~④⑦", "<code>scipy.stats</code> (ttest_ind, shapiro, levene, mannwhitneyu, f_oneway, tukey_hsd, pearsonr, spearmanr, chi2_contingency)", "검정"),
        ("⑤⑥", "<code>numpy.linalg.lstsq</code>, <code>numpy.linalg.inv</code>, <code>scipy.stats.t</code>, <code>scipy.stats.f</code>", "회귀를 식대로")]
K.build(ROOT / "topics" / "model" / "stats-tests.html", title="가설 검정과 회귀 분석", h1="가설 검정과 회귀 분석: 그 차이는 우연일까", track="모델",
        crumb="모델 &gt; 데이터 분석 기초 2/2쪽 &nbsp;|&nbsp; 이전: <a href=\"stats-describe.html\" style=\"color:inherit\">요약하고 추정하기</a>",
        one_line="\"차이가 없다면 이만큼의 차이가 우연히 나올 확률\"(<b>p값</b>)로 판단해요. 두 집단은 <b>t검정</b>, 셋 이상은 <b>분산분석</b>, 범주끼리는 <b>χ²</b>, 여러 변수는 <b>회귀</b>예요. p값과 함께 <b>효과 크기와 검정력</b>을 보고, 검정을 여러 번 하면 기준을 엄격하게 해요.",
        why=why, whole=whole, steps=steps, code_file="examples/stats/tests_regression.py", practice=practice, libs=libs,
        libs_note="<code>experiment</code>, <code>ols</code>, <code>vif</code>, <code>twenty_tests</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"stats-describe.html\">요약하고 추정하기</a>", "표준오차, 신뢰구간"), ("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "같은 데이터로 예측"), ("<a href=\"ml-learning-theory.html\">학습 이론</a>", "후보가 많으면 낙관적")],
        sources=["다루는 범위: 부트캠프 머신러닝 기초 앞에 두면 좋은 통계 분석이에요. 수업에서 따로 다루지 않았어요",
                 "함께 정리한 자료: 경희대 특강(2025년 1월) \"통계 기초 및 가설 검정\"과 \"SPSS 실습\"(가설 검정 절차, 1종과 2종 오류, 단측과 양측, 독립과 대응 표본 t검정, 정규성과 등분산 검정, 분산분석과 사후 분석, 상관과 교차분석, 분석 방법 고르기 표, 회귀 결과 읽기, 더미 변수, 조절 회귀 3단계와 평균 중심화, 조절 효과 해석 표), \"Data Science\" 특강(심슨의 역설, A/B 검정, 관측과 실험, 데이터 과학의 불편한 진실). 원래 실습 데이터(추천 정확도, 호텔 리뷰) 대신 이 노트의 업무 요청 데이터로 같은 분석을 다시 했어요",
                 "데이터: <code>examples/stats/data/requests.csv</code> (가상의 업무 요청 100건). ①⑦은 코드가 만든 가상 실험, ④의 심슨 예는 가상의 숫자",
                 "결과 값과 그림: <code>tools/pages/stats_tests.py</code>가 <code>examples/stats/tests_regression.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
