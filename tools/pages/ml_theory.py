"""모델 > 머신러닝 기초 5쪽: 학습 이론. examples/theory/learning_theory.py 하나가 ①~⑦로 자란다.
정답 규칙을 직접 정한 세상이라 진짜 오차를 정확히 계산할 수 있다.
실행: python tools/pages/ml_theory.py
"""
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "theory"
code = S.parse((EX / "learning_theory.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="learning_theory.py")
ENV = f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__} (seed 0)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "theory"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/theory/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
TRUE, ETA, DELTA = ns["TRUE"], ns["ETA"], ns["DELTA"]
pct = lambda x: f"{x:.0%}"  # noqa: E731

# ---------------- ① 정답을 아는 세상
Xd, yd = ns["X_demo"], ns["y_demo"]
clean = ns["inside"](Xd, TRUE).astype(int)
n_flip = int((clean != yd).sum())
S.claim(abs(ns["bayes"] - ETA) < 1e-12, "① 정답 직사각형의 진짜 오차는 잡음 비율과 같다")
S.claim(abs(ns["true_err_rect"](TRUE, eta=0)) < 1e-12, "① 잡음이 없으면 정답 직사각형의 오차는 0")
fig, ax = plt.subplots(figsize=(5.6, 5.2))
ax.add_patch(Rectangle((TRUE[0], TRUE[2]), TRUE[1] - TRUE[0], TRUE[3] - TRUE[2], fill=False, ec=C["ink"], lw=2))
for lab, col, nm in ((1, FAIL, "라벨 1"), (0, OK, "라벨 0")):
    k = yd == lab
    ax.scatter(Xd[k, 0], Xd[k, 1], s=22, color=col, label=nm, zorder=3)
fl = clean != yd
ax.scatter(Xd[fl, 0], Xd[fl, 1], s=110, facecolors="none", edgecolors=C["ink"], lw=1.2, label=f"뒤집힌 라벨 ({n_flip}건)", zorder=4)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x₁", ylabel="x₂", title="정답 직사각형(검은 선)과 뽑은 200건")
ax.set_aspect("equal")
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=False)
P_WORLD = fig_save(fig, "world")
outs["f_world"] = chart("이 쪽의 세상: [0, 1] × [0, 1]의 점, 정답 직사각형, 잡음 10%", P_WORLD, "",
                        [f"200건 중 {n_flip}건이 라벨이 뒤집혔어요(동그라미). 직사각형 밖인데 1이거나, 안인데 0인 점이에요.",
                         "어떤 모델도 이 뒤집힘은 미리 알 수 없어서, 아무리 잘해도 오차가 10% 아래로 내려가지 않아요."])
outs["s1"] = S.box("① 정답을 아는 세상", [(f"정답 직사각형 → x₁ {TRUE[0]}~{TRUE[1]}, x₂ {TRUE[2]}~{TRUE[3]} (넓이 {ns['area'](TRUE):.2f})", 1),
                                         (f"정답 직사각형 자체의 진짜 오차 → {ns['bayes']:.3f} (잡음 {ETA})", 2, "newf"),
                                         (f"200건을 뽑으면 라벨이 뒤집힌 것 → {n_flip}건", 3)],
                   ["진짜 오차는 \"처음 보는 점 하나를 뽑았을 때 틀릴 확률\"이에요. 실제 데이터에서는 모르는 값이지만, 여기서는 규칙을 알아서 넓이로 정확히 계산해요."])

# ---------------- ② Hoeffding
hf, R0 = ns["hoeff"], ns["R0"]
S.claim(all(v["outside"] <= DELTA for v in hf.values()), "② 막대 밖으로 나간 비율이 모두 δ 이하")
S.claim(all(v["outside"] < 0.01 for v in hf.values()), "② 실제로는 1%도 안 나간다 (상한이 넉넉하다)")
outs["s2"] = S.table(f"② 진짜 오차 {R0:.3f}인 모델 하나를 m건으로 2,000번 재면", ["m", "잰 오차의 표준편차", "Hoeffding 막대 ε (δ = 0.05)", "막대 밖으로 나간 비율", "실제 95%가 들어온 폭"],
                     [[m, f3(v["errs"].std()), f"±{v['eps']:.3f}", pct(v["outside"]) if v["outside"] else "0%", f"±{np.quantile(np.abs(v['errs'] - R0), 0.95):.3f}"] for m, v in hf.items()],
                     ["m을 5배로 늘리면 막대가 √5 ≈ 2.2배 좁아져요. 막대는 1 ÷ √m으로 줄어요.",
                      f"Hoeffding은 \"100번 중 5번 이하로만 막대 밖\"을 약속하는데, 실제로는 {pct(max(v['outside'] for v in hf.values()))} 남짓이었어요. 어떤 분포에서도 맞아야 해서 넉넉하게 잡은 상한이에요. 실제 95% 폭은 막대의 절반쯤이에요.",
                      "1~3쪽의 valid가 20건이었어요. m = 20이면 막대가 ±0.30이라, valid 한 번의 점수 차이 0.1은 우연과 구별이 안 돼요. 그래서 교차검증으로 여러 번 쟀어요."])
fig, axs = plt.subplots(1, 3, figsize=(12.6, 3.4), sharey=True)
for ax, (m, v) in zip(axs, hf.items()):
    step_ = 1 / m if m <= 100 else 0.01
    ax.hist(v["errs"], bins=np.arange(-step_ / 2, 0.6, step_), color=OK, alpha=.8)   # 잰 값(k ÷ m)이 막대 가운데에 오게
    ax.axvline(R0, color=C["ink"], lw=2, label=f"진짜 오차 {R0:.3f}")
    for s in (-1, 1):
        ax.axvline(R0 + s * v["eps"], color=FAIL, lw=1.6, ls="--", label="Hoeffding 막대" if s == 1 else None)
    ax.set(title=f"m = {m}", xlabel="m건으로 잰 오차", xlim=(0, 0.55))
axs[0].set_ylabel("2,000번 중 횟수")
fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=9, frameon=False)
P_HOEFF = fig_save(fig, "hoeffding")
outs["f_hoeff"] = chart("같은 모델을 m건으로 2,000번 잰 오차의 분포", P_HOEFF, "",
                        ["검은 선이 진짜 오차, 분홍 점선이 Hoeffding이 말하는 \"95% 이상은 이 안\"이에요. 건수가 적을수록 잰 값이 크게 흔들려요.",
                         "평균은 언제나 진짜 오차에 맞아요(잰 오차의 기댓값 = 진짜 오차). 문제는 한 번 잰 값이 얼마나 빗나가느냐예요."])

# ---------------- ③ 유한한 모델 묶음
fn = ns["finite"]
gm = {K_: v["gaps"].mean() for K_, v in fn.items()}
Ks = sorted(fn)
S.claim(abs(gm[1]) < 0.02 and all(gm[a] < gm[b] for a, b in zip(Ks, Ks[1:])), "③ 후보가 1개면 낙관이 거의 0이고, 후보가 늘수록 평균 낙관이 커진다")
S.claim(all((v["gaps"] > v["bound"]).mean() <= DELTA for v in fn.values()), "③ 상한을 넘은 비율이 δ 이하")
outs["s3"] = S.table(f"③ 후보 K개 중 {ns['M3']}건에서 가장 좋아 보이는 것을 골랐을 때 (200번)", ["후보 수 K", "진짜 − 보인 오차 (평균)", "95%가 이 안", "상한 √((log K + log(2/δ)) ÷ 2m)"],
                     [[f"{K_:,}", f"{v['gaps'].mean():+.3f}", f3(np.quantile(v["gaps"], 0.95)), f3(v["bound"])] for K_, v in fn.items()],
                     [f"후보가 1개면 고를 것이 없어서 낙관이 평균 {gm[1]:+.3f}, 거의 0이에요(②와 같아요). 후보가 1만 개면 평균 {gm[10000]:+.3f}만큼 실제보다 좋아 보이는 걸 골라요. 50건에서 운 좋게 잘 맞은 후보가 뽑히기 때문이에요.",
                      "상한은 log K로만 커져서, 후보가 1만 배 늘어도 상한은 2배가 안 돼요. 무작위 직사각형은 서로 비슷한 것이 많아서, 실제 낙관은 상한보다 더 천천히 커졌어요."])

# ---------------- ④ PAC: 직사각형
rc = ns["rect"]
ms = sorted(rc)
S.claim(all((v["R"] > v["bound"]).mean() <= DELTA for v in rc.values()), "④ 상한을 넘은 비율이 δ 이하")
S.claim(rc[10]["bound"] > 1, "④ m = 10이면 상한이 1보다 커서 쓸모없다")
r_ratio = rc[100]["R"].mean() / rc[1000]["R"].mean()
S.claim(7 < r_ratio < 13, "④ m을 10배 늘리면 평균 오차가 대략 10배 줄어든다")
outs["s4"] = S.table("④ 1인 점을 감싸는 가장 작은 직사각형 (잡음 없음, 1,000번)", ["m", "진짜 오차 평균", "95%가 이 안", "상한 (4 ÷ m) log(4/δ)"],
                     [[f"{m:,}", f3(v["R"].mean()), f3(np.quantile(v["R"], 0.95)), f3(v["bound"])] for m, v in rc.items()],
                     [f"m을 100에서 1,000으로 10배 늘리니 평균 오차가 {r_ratio:.1f}배 줄었어요. ②③의 1 ÷ √m보다 훨씬 빨라요. 오차 0인 모델이 묶음 안에 있고 잡음도 없으면 1 ÷ m으로 줄어요.",
                      f"m = 10이면 상한이 {rc[10]['bound']:.2f}로 1이 넘어서 아무것도 보장하지 않아요. 상한은 어떤 분포에서도 맞아야 해서 언제나 실제보다 커요."])
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.loglog(ms, [rc[m]["bound"] for m in ms], color=FAIL, lw=2, ls="--", marker="o", label="상한 (4/m) log(4/δ)")
ax.loglog(ms, [np.quantile(rc[m]["R"], 0.95) for m in ms], color=OK, lw=2.2, marker="o", label="실제 95% 지점")
ax.loglog(ms, [rc[m]["R"].mean() for m in ms], color=C["blue"], lw=2, marker="o", label="실제 평균")
ax.set(xlabel="m (log)", ylabel="진짜 오차 (log)", title="오차 0인 모델이 있을 때: 1 ÷ m으로 줄어요")
yt = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
ax.set_yticks(yt, [f"{t:g}" for t in yt])
ax.set_xticks(ms, [f"{m:,}" for m in ms])
ax.minorticks_off()
ax.legend(fontsize=9)
P_RECT = fig_save(fig, "rect_pac")
outs["f_rect"] = chart("직사각형 학습의 진짜 오차와 상한 (log 눈금)", P_RECT, "",
                       ["m이 30을 넘으면 세 선이 모두 기울기 −1인 곧은 선이 돼요. log 눈금에서 기울기 −1은 \"m이 10배면 10배 작아짐\", 곧 1 ÷ m이에요. m이 아주 작으면 1인 점이 거의 안 뽑혀서 직사각형이 거의 비어 있어요.",
                        f"상한(분홍)은 실제 95% 지점보다 {min(rc[m]['bound'] / np.quantile(rc[m]['R'], 0.95) for m in ms):.0f}~{max(rc[m]['bound'] / np.quantile(rc[m]['R'], 0.95) for m in ms):.0f}배 위에 있지만, 줄어드는 빠르기는 맞혀요. 이론 상한은 보통 이렇게 \"모양\"을 알려 줘요."])

# ---------------- ⑤ VC 차원
sh, gr = ns["shatter"], ns["growth"]
S.claim(sh["삼각형 3점"][0] == 8 and sh["정사각형 4점"][0] == 14 and sh["마름모 4점"][0] == 14, "⑤ 직선: 3점은 8가지 모두, 4점은 14가지뿐")
S.claim(sh["마름모 4점"][1] == 16 and sh["마름모+가운데 5점"][1] < 32, "⑤ 직사각형: 마름모 4점은 16가지 모두, 5점은 아님")
S.claim(all(a <= b <= c for a, b, c in gr.values()), "⑤ 실제 ≤ Sauer 상한 ≤ 2ⁿ")
S.claim(all(a == 2 * sum(math.comb(n - 1, k) for k in range(3)) for n, (a, _, _) in gr.items()), "⑤ 일반 위치의 점에서 직선이 만드는 라벨 수는 2 × Σ C(n−1, k), k ≤ 2와 정확히 같다")
outs["s5"] = S.table("⑤ 2ⁿ가지 라벨 중 하나의 모양으로 맞힐 수 있는 수", ["점", "직선 하나", "직사각형 하나", "전부 2ⁿ"],
                     [[k, f"<b>{a}</b>" if a == c else a, f"<b>{b}</b>" if b == c else b, c] for k, (a, b, c) in sh.items()],
                     ["굵은 숫자가 \"모든 라벨을 맞힘\"(산산조각 냄)이에요. 직선은 3점은 되지만 4점은 정사각형이든 마름모든 2가지가 안 돼요(대각선끼리 같은 라벨, XOR). 그래서 평면 직선의 VC 차원은 3이에요.",
                      "직사각형은 마름모 4점은 16가지 모두 되지만, 가운데에 한 점을 더하면 \"바깥 넷은 1, 가운데는 0\"을 못 만들어요. 4점을 감싸면 가운데도 들어오니까요. 평면 직사각형의 VC 차원은 4예요(5점은 어떻게 놓아도 안 돼요)."])
outs["t5"] = S.table("무작위 점 n개에서 직선이 만드는 라벨 수", ["n", "실제", "Sauer 상한 Σ C(n, i), i ≤ 3", "전부 2ⁿ"],
                     [[n, a, b, f"{c:,}"] for n, (a, b, c) in gr.items()],
                     [f"n = 10이면 1,024가지 중 {gr[10][0]}가지만 돼요. VC 차원이 d면 만들 수 있는 라벨 수가 2ⁿ이 아니라 n의 d제곱 정도로만 늘어요(Sauer 보조정리). 그래서 무한히 많은 직선이 있어도, n개 점 위에서는 유한한 묶음처럼 다룰 수 있어요.",
                      "실제 수는 점들이 한 줄에 세 개 놓이지 않으면 2 × (C(n−1, 0) + C(n−1, 1) + C(n−1, 2))와 정확히 같아요."])
fig, ax = plt.subplots(figsize=(7.2, 3.9))
nn = sorted(gr)
ax.semilogy(nn, [gr[n][2] for n in nn], color=C["gray"], lw=2, ls="--", marker="o", label="전부 2ⁿ")
ax.semilogy(nn, [gr[n][1] for n in nn], color=FAIL, lw=2, marker="o", label="Sauer 상한 (d = 3)")
ax.semilogy(nn, [gr[n][0] for n in nn], color=OK, lw=2.4, marker="o", label="직선이 실제로 만든 수")
ax.set(xlabel="점 수 n", ylabel="라벨 붙이는 방법의 수 (log)", title="VC 차원이 유한하면 2ⁿ이 아니라 다항식으로 늘어요")
ax.legend(fontsize=9)
P_GROWTH = fig_save(fig, "growth")
outs["f_growth"] = chart("직선이 n개 점에 붙일 수 있는 라벨 수", P_GROWTH, "",
                         ["회색(2ⁿ)은 log 눈금에서 곧은 선(지수)인데, 보라와 분홍은 휘어서 점점 덜 늘어요(다항식).",
                          "이 차이가 \"모델이 무한히 많아도 데이터가 늘면 믿을 수 있다\"의 근거예요."])

# ---------------- ⑥ Rademacher
rad = ns["rad"]
D_ = ns["DEPTHS"]
S.claim(all(rad[a] < rad[b] for a, b in zip(D_, D_[1:])), "⑥ 깊을수록 무작위 라벨을 더 잘 맞힌다")
S.claim(rad[12] > 0.8 and rad[1] < 0.15, "⑥ 깊이 1은 0.15 아래, 깊이 12는 0.8 위")
outs["s6"] = S.table("⑥ 동전 던지기로 만든 라벨을 트리가 얼마나 맞히나 (⑦의 200건, 30번 평균)", ["깊이", "Rademacher 복잡도 (추정)"],
                     [[d, f3(v)] for d, v in rad.items()],
                     [f"깊이 1은 {rad[1]:.2f}, 깊이 12는 {rad[12]:.2f}예요. 0이면 잡음을 전혀 못 따라가고, 1이면 어떤 잡음이든 다 외워요.",
                      "엄밀히는 \"가장 잘 맞히는 트리\"를 찾아야 하는데, scikit-learn 트리는 위에서부터 욕심껏 나눠서 가장 좋은 트리를 보장하지 않아요. 그래서 이 값은 진짜 값의 아래쪽 추정이에요."])

# ---------------- ⑦ 깊이 고르기
rows, pick = ns["rows"], ns["pick"]
true_best = pick["true"]
S.claim(abs(rows[1]["best"] - 0.3) < 0.01 and all(abs(rows[d]["best"] - ETA) < 0.005 for d in D_ if d >= 4), "⑦ 깊이 4부터는 가장 좋은 트리가 바닥(0.1)에 닿는다")
S.claim(pick["train"] == 12 and pick["valid"] == true_best and pick["cv"] != true_best and pick["penalty"] == 1, "⑦ train은 가장 깊게, valid는 정답, CV는 조금 깊게, 벌점은 가장 얕게 골랐다")
S.claim(all(rows[d]["penalty"] > 0.45 for d in D_), "⑦ 벌점 상한은 모든 깊이에서 0.45가 넘는다")
how_name = {"train": "train 오차가 가장 작은 깊이", "valid": "valid 50건 (150건으로 학습)", "cv": "5-fold 교차검증", "penalty": "train 오차 + Rademacher 벌점 (상한)", "true": "진짜 오차가 가장 작은 깊이 (정답)"}
outs["s7"] = S.table("⑦ 깊이마다 (200건으로 학습)", ["깊이", "train 오차", "진짜 오차", "그 깊이의 가장 좋은 트리", "근사 오차", "추정 오차", "5-fold 교차검증 오차"],
                     [[f"<b>{d}</b>" if d == true_best else d, f3(r["train"]), f"<b>{f3(r['true'])}</b>" if d == true_best else f3(r["true"]), f3(r["best"]), f3(r["best"] - ns["bayes"]), f3(r["true"] - r["best"]), f3(r["cv"])] for d, r in rows.items()],
                     [f"근사 오차 = (그 깊이의 가장 좋은 트리) − 바닥 {ns['bayes']:.1f}. 깊이 1~3은 직사각형을 다 표현하지 못해서 커요. 깊이 4부터는 0이에요(직사각형은 질문 4번으로 감쌀 수 있어요).",
                      f"추정 오차 = (200건으로 배운 트리) − (그 깊이의 가장 좋은 트리). 깊을수록 200건의 잡음까지 외워서 커져요. 깊이 4에서 {rows[4]['true'] - rows[4]['best']:.3f}, 깊이 12에서 {rows[12]['true'] - rows[12]['best']:.3f}예요.",
                      "train 오차는 깊이 12에서 0이지만 진짜 오차는 그때가 가장 커요."])
outs["s7pick"] = S.table("⑦ 고르는 방법마다 고른 깊이", ["방법", "고른 깊이", "그 깊이의 진짜 오차"],
                         [[how_name[h], f"<b>{pick[h]}</b>", f3(rows[pick[h]]["true"])] for h in ("true", "valid", "cv", "penalty", "train")],
                         [f"이번에는 valid 50건이 정답을 골랐고, 교차검증은 한 칸 깊은 {pick['cv']}를 골랐어요(진짜 오차 {rows[pick['cv']]['true']:.3f}). valid가 맞힌 건 운도 있어요. ②에서 50건의 막대는 ±{math.sqrt(math.log(2 / DELTA) / 100):.2f}이거든요.",
                          f"상한을 벌점으로 쓰면(구조적 위험 최소화) 깊이 {pick['penalty']}을 골랐어요. 벌점이 깊이마다 {min(rad.values()):.2f}~{max(rad.values()):.2f}이라 train 오차 차이를 다 덮어 버려요. 상한 값은 모든 깊이에서 0.45가 넘어서, 진짜 오차({min(r["true"] for r in rows.values()):.2f}~{max(r["true"] for r in rows.values()):.2f})의 상한으로는 맞지만 고르는 데는 너무 보수적이에요."])
fig, ax = plt.subplots(figsize=(8.4, 4.2))
xs = list(range(len(D_)))
ax.axhline(ns["bayes"], color=C["ink"], lw=1, ls=":", label="바닥 (잡음 0.1)")
ax.plot(xs, [rows[d]["best"] for d in D_], color=C["gray"], lw=2, marker="o", label="그 깊이의 가장 좋은 트리")
ax.plot(xs, [rows[d]["true"] for d in D_], color=OK, lw=2.6, marker="o", label="200건으로 배운 트리의 진짜 오차")
ax.plot(xs, [rows[d]["train"] for d in D_], color=FAIL, lw=2, marker="o", label="train 오차")
ax.plot(xs, [rows[d]["cv"] for d in D_], color=C["blue"], lw=1.8, ls="--", marker="o", label="5-fold 교차검증 오차")
ax.fill_between(xs, [ns["bayes"]] * len(xs), [rows[d]["best"] for d in D_], color=C["gray"], alpha=.12)
ax.fill_between(xs, [rows[d]["best"] for d in D_], [rows[d]["true"] for d in D_], color=OK, alpha=.12)
ax.text(0.15, 0.2, "근사 오차", fontsize=10, color=C["gray"])
ax.text(5.6, 0.15, "추정 오차", fontsize=10, color=OK)
ax.set_xticks(xs, [str(d) for d in D_])
ax.set(xlabel="트리 깊이", ylabel="오차", title="얕으면 근사 오차, 깊으면 추정 오차", ylim=(0, 0.42))
fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=9, frameon=False)
P_DEPTH = fig_save(fig, "depth_tradeoff")
outs["f_depth"] = chart("깊이에 따른 오차 쪼개기", P_DEPTH, "",
                        ["회색 칠(근사 오차)은 \"이 깊이로는 정답을 다 못 그린다\", 보라 칠(추정 오차)은 \"200건으로는 그 깊이의 가장 좋은 트리를 못 찾는다\"예요. 둘의 합이 가장 작은 곳이 고를 깊이예요.",
                         "train 오차(분홍)는 깊을수록 계속 내려가서 고르는 기준이 될 수 없어요. 교차검증(파랑 점선)은 진짜 오차(보라 선)를 꽤 잘 따라가요."])
outs["why"] = S.table("1~4쪽에서 \"그런가 보다\" 하고 넘어간 질문들", ["질문", "어디서", "이 쪽"],
                      [["valid 20건의 점수를 얼마나 믿을 수 있나", "2쪽 ⑫", "② Hoeffding 막대"],
                       ["후보를 많이 비교하면 왜 위험한가", "2쪽 ⑫, 3쪽 ⑰", "③ 후보 수와 낙관"],
                       ["모델이 무한히 많은데(직선, 트리) 왜 배울 수 있나", "1쪽 ⑤, 2쪽 ⑧", "⑤ VC 차원, ⑥ Rademacher"],
                       ["깊이, C, alpha는 왜 교차검증으로 고르나", "1~4쪽 전부", "⑦ 근사 오차와 추정 오차"]])

flow = [[1, "정답을 아는 세상", "직사각형 + 잡음 10%", "진짜 오차를 계산"], [2, "Hoeffding", "한 모델을 m건으로", "1 ÷ √m 막대"], [3, "후보가 많으면", "가장 좋아 보이는 것 고르기", "log K 만큼 낙관"],
        [4, "PAC", "오차 0인 모델이 있을 때", "1 ÷ m"], [5, "VC 차원", "모든 라벨을 맞히는 점 수", "직선 3, 직사각형 4"], [6, "Rademacher", "무작위 라벨 외우기", "깊을수록 1"],
        [7, "깊이 고르기", "근사 + 추정 오차", "교차검증"]]
S.save(ROOT / "topics" / "model" / "ml-learning-theory-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>1~4쪽에서 모델을 고를 때마다 "valid 20건은 흔들리니 교차검증으로", "깊을수록 train은 외운다", "후보는 교차검증으로 고른다"고 했어요. 경험으로는 맞는데, <b>왜</b> 그런지, 데이터가 몇 건이면 믿을 수 있는지는 따지지 않았어요.</p>
{K.out("why")}
<p>이걸 따지려면 "처음 보는 데이터에서의 오차"를 알아야 하는데, 실제 데이터에서는 그 값을 알 수 없어요. 그래서 이 쪽은 <b>정답 규칙을 직접 정한 작은 세상</b>에서 실험해요. 규칙을 아니까 진짜 오차를 정확히 계산할 수 있고, 이론이 말하는 상한이 맞는지 숫자로 확인할 수 있어요.</p>
"""
whole = """<p>세상 만들기 → 모델 하나를 재는 막대 → 여럿 중 고를 때 → 오차 0인 모델이 있을 때 → 모델이 무한히 많을 때(두 가지 재는 법) → 깊이 고르기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 새 파일 <code>examples/theory/learning_theory.py</code> 하나예요. 데이터 파일 없이 코드가 직접 점을 뽑아요. CPU로 10초 안에 끝나요.</p>"""
steps = [
    K.step(1, "정답을 아는 세상 만들기", "[0, 1] × [0, 1]에 점을 고르게 뿌리고, 정답 직사각형 안이면 1, 밖이면 0으로 정한 뒤 10%는 라벨을 뒤집어요.",
           "<p>학습 이론은 몇 가지 약속에서 시작해요.</p>"
           "<ul><li><b>i.i.d.</b>: 데이터는 어떤 정해진 분포에서 <b>서로 상관없이, 같은 방식으로</b> 뽑혀요(independent and identically distributed). train도 test도 같은 분포에서 나와요. 이 약속이 깨지면(예: 작년 데이터로 배워 올해에 쓰기) 아래 보장들도 깨져요.</li>"
           "<li><b>진짜 오차</b> R(h): 그 분포에서 새 점 하나를 뽑았을 때 모델 h가 틀릴 확률. 우리가 정말 알고 싶은 값이에요.</li>"
           "<li><b>보이는 오차</b> R̂(h): 가진 m건에서 틀린 비율. train 오차, valid 오차가 모두 이거예요.</li></ul>"
           "<p>라벨에 잡음이 있으면 같은 x에서도 y가 0일 때와 1일 때가 있어요. 그러면 어떤 모델도 넘지 못하는 바닥이 생겨요. x마다 더 흔한 답을 고르는 모델이 <b>베이즈 분류기</b>이고(4쪽 ④의 베이즈 정리로 고르는 것과 같아요), 그 오차가 <b>베이즈 오차</b>예요. 여기서는 정답 직사각형이 베이즈 분류기이고 베이즈 오차는 잡음 비율 0.1이에요.</p>"
           + K.out("f_world")
           + K.code_table([("rng.random((m, 2))", "[0, 1] × [0, 1]에 m개 점을 고르게"), ("np.where(flip, 1 - y, y)", "10% 확률로 라벨 뒤집기"),
                           ("true_err_rect(h)", "직사각형 모델의 진짜 오차를 넓이로: 잡음 + (1 − 2 × 잡음) × (정답과 다른 곳의 넓이)"),
                           ("GRID", "25만 칸 격자. 트리처럼 넓이로 계산하기 어려운 모델은 격자에서 정답과 다른 칸의 비율로 재요")]), "s1"),
    K.step(2, "한 모델을 m건으로 재면: Hoeffding 부등식", "미리 정해 둔 모델 하나의 오차를 m건으로 2,000번 재서, 잰 값이 진짜 오차에서 얼마나 벗어나는지 봐요.",
           "<p>모델 h를 데이터를 보기 <b>전에</b> 정해 뒀다면, m건에서 잰 오차는 \"동전을 m번 던져 앞면 비율을 재는 것\"과 같아요. 평균은 진짜 오차에 맞고, 흔들림만 문제예요. <b>Hoeffding 부등식</b>은 그 흔들림의 상한이에요.</p>"
           "<p><code>P(|R̂(h) − R(h)| ≥ ε) ≤ 2 exp(−2mε²)</code></p>"
           "<p>오른쪽을 δ로 두고 ε을 풀면, <b>1 − δ 이상의 확률로</b> <code>R(h) ≤ R̂(h) + √(log(2/δ) ÷ 2m)</code>예요. δ는 \"이 약속이 틀려도 되는 확률\"이에요. 막대는 1 ÷ √m으로 줄어서, 2배 좁히려면 데이터가 4배 필요해요.</p>"
           + K.out("s2") + K.out("f_hoeff")
           + K.code_table([("math.sqrt(math.log(2 / DELTA) / (2 * m))", "Hoeffding 막대 ε. δ = 0.05"), ("(np.abs(errs - R0) >= eps).mean()", "2,000번 중 막대 밖으로 나간 비율")]), "s2"),
    K.step(3, "후보가 많으면 고른 모델이 낙관적이 돼요: 유한한 모델 묶음", "무작위 직사각형 K개 중 50건에서 가장 좋아 보이는 것을 고르고, 그 모델의 진짜 오차와 보인 오차의 차이를 봐요.",
           "<p>②는 모델을 미리 정해 뒀을 때 얘기예요. 실제로는 데이터를 보고 <b>가장 좋아 보이는 모델</b>을 골라요. 그러면 \"운 좋게 이 m건에서 잘 맞은\" 모델이 뽑히기 쉬워서, 보인 오차가 진짜보다 낙관적이 돼요. 모델을 고르는 후보들의 모음을 <b>모델 묶음</b>(가설 집합, H)이라고 해요.</p>"
           "<p>후보가 K개면, 그중 하나라도 막대 밖으로 나갈 확률은 많아야 K × (하나가 나갈 확률)이에요(합집합 상한). 그래서 K개 <b>모두</b>에 대해 동시에 <code>R(h) ≤ R̂(h) + √((log K + log(2/δ)) ÷ 2m)</code>예요. 고른 모델도 그 K개 중 하나니까 이 막대 안에 있어요.</p>"
           "<p>log K로만 커진다는 게 핵심이에요. log₂ K는 \"후보 중 하나를 가리키는 데 필요한 bit 수\"라서, 막대는 <b>모델을 적는 데 드는 bit 수 ÷ 데이터 수</b>로 정해져요. 같은 성적이면 더 단순한 모델 묶음이 낫다는 <b>오컴의 면도날</b>이 여기서 나와요.</p>"
           + K.out("s3")
           + "<p>3쪽 ⑰에서 \"test는 마지막에 한 번만\"이라고 한 이유도 이거예요. test로 여러 모델을 비교해 고르면, test 점수가 ③처럼 낙관적인 값이 돼요.</p>"
           + K.code_table([("random_rects(K)", "무작위 직사각형 K개 (모서리를 고르게 뽑기)"), ("emp.argmin()", "50건에서 오차가 가장 작은 후보 (경험적 위험 최소화, ERM)"),
                           ("true_err_rect(H[k]) - emp[k]", "고른 후보의 진짜 오차 − 보인 오차")]), "s3"),
    K.step(4, "오차 0인 모델이 있을 때: 직사각형 PAC 학습", "잡음을 끄고, 1인 점을 모두 감싸는 가장 작은 직사각형을 배워요. m을 늘리며 진짜 오차가 얼마나 빨리 줄어드는지 봐요.",
           "<p><b>PAC 학습</b>(Probably Approximately Correct)은 \"학습할 수 있다\"를 이렇게 정해요. 어떤 ε, δ를 정해도, 데이터가 (1/ε, 1/δ의 다항식)건 이상이면 <b>1 − δ 이상의 확률로(Probably) 진짜 오차가 ε 이하(Approximately Correct)</b>인 모델을 내놓는 방법이 있다. 분포가 어떤 모양이든 상관없이요.</p>"
           "<p>직사각형은 PAC 학습이 돼요. \"1인 점을 감싸는 가장 작은 직사각형\"은 언제나 정답 안쪽에 있어서, 틀리는 곳은 정답의 네 변 근처 띠뿐이에요. 띠 하나에 확률 ε/4 이상이 있으면 m건 중 하나라도 거기 떨어질 확률이 높아서, <code>m ≥ (4/ε) log(4/δ)</code>면 충분해요.</p>"
           "<p>오차 0인 모델이 묶음 안에 있고 그걸 찾으면(<b>일관된</b> 경우), 상한이 √가 없는 <code>(log K + log(1/δ)) ÷ m</code> 꼴로 1 ÷ m만큼 빨리 줄어요. 잡음이 있거나 묶음 안에 정답이 없는 보통의 경우(②③)는 1 ÷ √m이에요. 이때는 \"묶음 안의 가장 좋은 모델에 ε만큼 가깝다\"를 목표로 삼아요(애그노스틱 PAC).</p>"
           + K.out("s4") + K.out("f_rect")
           + "<p>⚠️ 데이터가 충분해도 <b>계산</b>이 어려울 수 있어요. 예를 들어 \"조건 세 개 중 하나를 만족하면 1\" 같은 규칙 묶음(3-항 DNF)은 필요한 데이터 수는 적은데, 그 묶음 안에서 맞는 규칙을 찾는 계산이 너무 어려워요. 같은 규칙을 다른 꼴의 더 큰 묶음으로 바꿔 쓰면 쉽게 배워져요. 그래서 모델 묶음을 어떻게 잡느냐가 데이터 수만큼 중요해요.</p>"
           + K.code_table([("sample(m, eta=0)", "잡음 없는 세상"), ("(P[:, 0].min(), P[:, 0].max(), ...)", "1인 점들을 감싸는 가장 작은 직사각형"), ("4 / m * math.log(4 / DELTA)", "1 − δ 확률로 진짜 오차가 이 아래")]), "s4"),
    K.step(5, "무한한 모델 묶음: VC 차원", "직선과 직사각형이 점 몇 개까지 모든 라벨을 맞힐 수 있는지 하나하나 세어 봐요.",
           "<p>③의 막대에는 log K가 들어 있어요. 그런데 1쪽의 로지스틱(모든 직선)이나 2쪽의 트리는 후보가 무한히 많아요. log ∞라서 ③으로는 아무 말도 못 해요.</p>"
           "<p>실마리는 이거예요. 직선은 무한히 많아도, <b>점 n개에 붙이는 라벨</b>은 많아야 2ⁿ가지예요. 직선 두 개가 n개 점에 같은 라벨을 붙이면 그 데이터 위에서는 같은 모델이에요. 그래서 \"n개 점에 붙일 수 있는 라벨 수\"(<b>성장 함수</b>)가 ③의 K 역할을 해요.</p>"
           "<p>모델 묶음이 n개 점에 2ⁿ가지 라벨을 모두 붙일 수 있으면 그 점들을 <b>산산조각 낸다</b>(shatter)고 해요. 산산조각 낼 수 있는 가장 큰 n이 <b>VC 차원</b>이에요.</p>"
           + K.out("s5")
           + "<p>d차원 공간의 직선(초평면)은 VC 차원이 d + 1이라 계수 개수와 같아요. 하지만 늘 그렇지는 않아요. sin(ωx)는 계수가 ω 하나뿐인데 VC 차원이 무한이에요. ω를 아주 크게 하면 어떤 라벨이든 맞히도록 물결을 맞출 수 있거든요.</p>"
           "<p><b>Sauer 보조정리</b>: VC 차원이 d면 n개 점에 붙일 수 있는 라벨 수는 많아야 C(n, 0) + C(n, 1) + ... + C(n, d) ≤ (en/d)ᵈ예요. C(n, i)는 n개 중 i개를 고르는 방법의 수, e는 자연로그의 밑 2.718...이에요. 지수(2ⁿ)가 아니라 다항식이에요.</p>"
           + K.out("t5") + K.out("f_growth")
           + "<p>이걸 ③의 log K 자리에 넣으면 <code>R(h) ≤ R̂(h) + √(2d log(em/d) ÷ m) + √(log(1/δ) ÷ 2m)</code>예요. 결국 <b>데이터 수 ÷ VC 차원</b>(m/d)이 클수록 믿을 수 있어요. 거꾸로, 어떤 방법을 써도 VC 차원이 d면 적어도 d에 비례하는 데이터가 필요하다는 하한도 있어요. VC 차원이 무한이면 PAC 학습이 아예 안 돼요.</p>"
           "<p>4쪽 ⑬에서 \"스텀프 T개를 더한 모델의 VC 차원은 T log T쯤\"이라고 한 게 이 VC 차원이에요.</p>"
           + K.code_table([("linprog(..., A_ub=A, b_ub=-np.ones(n))", "y × (w₁x₁ + w₂x₂ + b) ≥ 1을 모두 만족하는 직선이 있는지 확인하는 선형 계획 문제"),
                           ("itertools.product([0, 1], repeat=n)", "n개 점의 2ⁿ가지 라벨을 하나씩"), ("math.comb(n, i)", "n개 중 i개 고르는 수 C(n, i)")]), "s5"),
    K.step(6, "무작위 라벨을 얼마나 외우나: Rademacher 복잡도", "동전 던지기로 만든 가짜 라벨을 깊이별 트리가 얼마나 맞히는지 재요. 이게 모델 묶음의 복잡도예요.",
           "<p>VC 차원은 \"최악의 점 배치\"를 기준으로 해서, 실제 데이터보다 비관적이에요. <b>Rademacher 복잡도</b>는 가진 데이터 위에서 바로 재요.</p>"
           "<p>방법은 간단해요. 각 점에 동전을 던져 +1, −1 가짜 라벨 σ를 붙이고, 모델 묶음에서 그 가짜 라벨을 가장 잘 맞히는 모델을 찾아요. \"맞은 비율 − 틀린 비율\"을 여러 번 평균 낸 값이에요. 가짜 라벨에는 배울 게 없으니, 이걸 잘 맞힌다는 건 <b>무엇이든 외울 수 있다</b>는 뜻이에요.</p>"
           + K.out("s6")
           + "<p>이걸로도 상한이 나와요. 1 − δ 이상의 확률로 묶음 안의 모든 모델에 대해 <code>R(h) ≤ R̂(h) + (Rademacher 복잡도) + √(log(1/δ) ÷ 2m)</code>예요. 4쪽 ⑬의 마진 상한도 이 꼴이었어요. Rademacher 복잡도는 √(2 log(성장 함수) ÷ m) 이하라서, VC 차원 상한보다 늘 좋거나 같아요.</p>"
           "<p>딥러닝 모델은 무작위 라벨도 거의 다 외울 만큼 크다는 실험이 알려져 있어요. Rademacher 복잡도가 1에 가깝다는 뜻이라, 이 상한만으로는 딥러닝이 왜 일반화되는지 설명이 안 돼요. 그래서 마진, 규제, 학습 방법의 성질을 함께 봐요.</p>"
           + K.code_table([("rng.choice([-1, 1], len(X))", "동전 던지기 가짜 라벨 σ"), ("DecisionTreeClassifier(max_depth=depth).fit(X, s)", "가짜 라벨을 가장 잘 맞히는 트리를 찾는 셈"),
                           ("(s * h).mean()", "맞으면 +1, 틀리면 −1의 평균 = 맞은 비율 − 틀린 비율")]), "s6"),
    K.step(7, "깊이 고르기: 근사 오차와 추정 오차", "200건으로 깊이 1~12인 트리를 배우고, 진짜 오차를 두 조각으로 나눈 뒤, 깊이를 고르는 방법 넷을 비교해요.",
           "<p>모델이 바닥(베이즈 오차)보다 얼마나 나쁜지는 두 조각으로 나뉘어요.</p>"
           "<p><code>진짜 오차 − 바닥 = (진짜 오차 − 묶음 안 가장 좋은 모델의 오차) + (묶음 안 가장 좋은 모델의 오차 − 바닥)</code></p>"
           "<ul><li><b>추정 오차</b>(앞): 가진 데이터로는 묶음 안의 가장 좋은 모델을 못 찾아서 생기는 차이. 묶음이 클수록, 데이터가 적을수록 커요. ②~⑥의 상한들이 다루는 게 이거예요.</li>"
           "<li><b>근사 오차</b>(뒤): 묶음 자체가 정답을 못 그려서 생기는 차이. 묶음이 클수록 작아져요. 분포를 모르면 잴 수 없어요.</li></ul>"
           "<p>1쪽 ⑥의 편향과 분산과 같은 이야기예요. 근사 오차가 편향, 추정 오차가 분산 쪽이에요. 여기서는 정답을 알아서 두 조각을 실제로 재요. \"그 깊이의 가장 좋은 트리\"는 10만 건으로 배워서 근사해요.</p>"
           + K.out("s7") + K.out("f_depth")
           + "<p><b>깊이를 고르는 방법</b>: 가장 단순한 건 train 오차가 가장 작은 걸 고르는 것(<b>경험적 위험 최소화</b>, ERM)인데, 늘 가장 깊은 트리를 골라요. <b>구조적 위험 최소화</b>(SRM)는 묶음을 \"깊이 1 ⊂ 깊이 2 ⊂ ...\"로 쌓아 두고, train 오차 + 복잡도 벌점(⑥의 상한)이 가장 작은 곳을 골라요. 1쪽 ④의 규제(Ridge, Lasso)도 같은 생각을 \"가중치 크기에 벌점\"으로 바꾼 거예요. 마지막은 데이터 일부를 떼어 재는 valid와 교차검증이에요.</p>"
           + K.out("s7pick")
           + "<p>이론으로도 교차검증은 \"조금 적은 데이터로 SRM을 한 것\"만큼 좋다는 보장이 있어요. k-fold의 k가 크면 매번 배우는 데이터가 전체에 가까워 치우침은 작지만, 배우는 데이터들이 서로 비슷해서 점수가 많이 흔들려요. k가 작으면 반대예요. 그래서 보통 5나 10을 써요. 4쪽 ⑫에서 본 힌지, 지수, 로그 손실은 0-1 손실 대신 줄이기 쉬운 손실(대리 손실)인데, 이것들을 줄여도 결국 0-1 오차가 바닥에 다가간다는 보장도 있어요.</p>"
           + K.code_table([("DecisionTreeClassifier(max_depth=d, min_samples_leaf=50).fit(X_big, y_big)", "10만 건으로 배운 \"그 깊이의 가장 좋은 트리\"의 근사"),
                           ("true_err_model(tree)", "격자에서 잰 진짜 오차"), ("cross_val_score(..., cv=cv7)", "5-fold 교차검증 정확도"),
                           ("rows[d][\"train\"] + rad[d] + math.sqrt(...)", "⑥의 Rademacher 상한을 벌점으로 (SRM)")]), "s7"),
]
practice = [("데이터가 몇 건이면 되나", "정해진 답은 없지만, 비교하고 싶은 점수 차이보다 ②의 막대(대략 1 ÷ √m)가 작아야 해요. 0.05 차이를 가리려면 수백 건이 필요해요."),
            ("후보를 많이 비교했다", "하이퍼파라미터 조합이 수백 개면 교차검증 점수도 ③처럼 낙관적이 돼요. 고를 때 쓴 점수와 따로, 마지막에 test로 한 번 재요."),
            ("이론 상한을 실제로 쓰나", "값 자체는 보통 1에 가깝거나 넘어서(⑦) 숫자로는 잘 안 써요. 무엇이 커지면 위험한지(후보 수, VC 차원, 데이터 수)를 읽는 데 써요."),
            ("train과 test의 분포가 다르다", "이 쪽의 모든 보장은 i.i.d. 가정 위에 있어요. 기간이나 출처가 다르면 test를 그 상황과 같게 따로 만들어요."),
            ("교차검증의 k", "5나 10이 보통이에요. 데이터가 아주 적으면 leave-one-out도 쓰지만, 계산이 m배 들고 점수가 많이 흔들려요.")]
libs = [("①", "<code>np.random.default_rng</code>, <code>rng.random</code>", "균등 분포에서 점 뽑기"), ("②③④", "<code>math.sqrt</code>, <code>math.log</code>", "상한 계산"),
        ("⑤", "<code>scipy.optimize.linprog</code>, <code>itertools.product</code>, <code>math.comb</code>", "직선으로 가를 수 있는지, 라벨 세기"),
        ("⑥⑦", "<code>DecisionTreeClassifier</code>, <code>StratifiedKFold</code>, <code>cross_val_score</code>", "트리, 교차검증 (scikit-learn)")]
K.build(ROOT / "topics" / "model" / "ml-learning-theory.html", title="학습 이론: 몇 건이면 믿을 수 있나", h1="학습 이론: 몇 건이면 믿을 수 있나", track="모델",
        crumb="모델 &gt; 머신러닝 기초 5/5쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-svm-lda-pca.html\" style=\"color:inherit\">SVM, 판별분석, 차원 축소</a>",
        one_line="가진 데이터에서의 오차와 처음 보는 데이터에서의 오차는 <b>데이터 수</b>와 <b>모델 묶음의 크기</b>(후보 수, VC 차원, Rademacher 복잡도)만큼 벌어질 수 있어요. 묶음이 크면 정답에 가까워지지만 데이터가 더 필요해서, 그 균형을 교차검증으로 찾아요.",
        why=why, whole=whole, steps=steps, code_file="examples/theory/learning_theory.py", practice=practice, libs=libs,
        libs_note="<code>inside</code>, <code>sample</code>, <code>area</code>, <code>true_err_rect</code>, <code>true_err_model</code>, <code>random_rects</code>, <code>line_can</code>, <code>rect_can</code>, <code>count_labelings</code>, <code>rademacher</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "교차검증을 실제 데이터에"), ("<a href=\"ml-svm-lda-pca.html\">SVM, 판별분석, 차원 축소</a>", "마진 상한, 대리 손실")],
        sources=["다루는 범위: 부트캠프 머신러닝 기초(Day05~08)를 이론 쪽으로 넓힌 쪽이에요",
                 "함께 정리한 자료: Mohri, Rostamizadeh, Talwalkar, <i>Foundations of Machine Learning</i> 2판 2장(PAC 학습, 축 정렬 직사각형, 유한 가설 집합의 일관된 경우와 아닌 경우, Hoeffding, 오컴의 면도날, 애그노스틱 PAC, 베이즈 오차와 잡음, 3-항 DNF와 k-CNF), 3장(Rademacher 복잡도, 성장 함수, VC 차원과 산산조각, Radon 정리, Sauer 보조정리, VC 상한과 하한), 4장(추정 오차와 근사 오차, ERM, SRM, 교차검증과 k-fold, 규제, 볼록 대리 손실). 값과 그림은 이 예제로 계산했어요",
                 "데이터: 코드가 직접 만든 가상의 점 (정답 직사각형, 잡음 10%)",
                 "결과 값과 그림: <code>tools/pages/ml_theory.py</code>가 <code>examples/theory/learning_theory.py</code>를 단계별로 실제 실행한 결과 (seed 0)",
                 "공식 문서: <a href=\"https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html\" target=\"_blank\">scipy.optimize.linprog</a>, <a href=\"https://scikit-learn.org/stable/modules/cross_validation.html\" target=\"_blank\">Cross-validation</a>"])
print("ok")
