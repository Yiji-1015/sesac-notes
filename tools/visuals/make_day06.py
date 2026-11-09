import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]   # 없는 기호(≈, ŷ)는 DejaVu로 대체
"""Day6 그림: 선형회귀 · 회귀지표 · 로지스틱 · KNN · 스케일링 · K 비교.

실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day06.py
수업 데이터(day6_compare)는 읽기만 한다.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_moons
from sklearn.metrics import f1_score, root_mean_squared_error

DAY = 6
OK, FAIL = C["purple"], C["pink"]          # 성공(0) / 실패(1) — 페이지 데모와 같은 색
DATA = "C:/practice_assignment/20261026/03_심화/day6_compare/data/"
NUM = ["input_chars", "allowed_tools"]
CAT = ["task_type", "priority"]
FEATURES = NUM + CAT


def load():
    df = pd.read_csv(DATA + "requests.csv").set_index("request_id")
    man = pd.read_csv(DATA + "split_manifest.csv")
    part = lambda s: df.loc[man[man.split == s].sort_values("position").request_id].reset_index()
    return part("train"), part("valid"), part("test")


def build(est, scale=True):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler() if scale else "passthrough")])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("pre", ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])),
                     ("model", est)])


tr, va, te = load()

# ---------------------------------------------------------------- ① 선형회귀: 직선 + 잔차
x = np.array([1, 2, 3, 4, 5]); y = np.array([52, 61, 69, 80, 89])
lr = LinearRegression().fit(x[:, None], y)
w, b = lr.coef_[0], lr.intercept_
fig, ax = plt.subplots(figsize=(6.4, 3.9))
xx = np.linspace(0, 5.4, 50)
ax.plot(xx, w * xx + b, color=C["purple"], lw=2.5, label=f"학습된 직선  ŷ = {w:.1f}x + {b:.1f}")
xe = np.linspace(5.4, 6.3, 10)
ax.plot(xe, w * xe + b, color=C["purple"], lw=2, ls="--", alpha=.6)
yhat = lr.predict(x[:, None])
for xi, yi, yh in zip(x, y, yhat):
    ax.plot([xi, xi], [yh, yi], color=C["pink"], lw=2.5)
    ax.text(xi + .08, (yi + yh) / 2, f"{yi - yh:+.1f}", color=C["pink"], fontsize=10, va="center")
ax.scatter(x, y, s=70, color=C["ink"], zorder=3, label="실제 점수")
ax.scatter([6], [w * 6 + b], s=90, marker="*", color=C["orange"], zorder=3)
ax.annotate("6시간 예측 98.1\n(직선 연장 = 외삽)", (6, w * 6 + b), (4.3, 99), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=C["orange"]), color=C["orange"])
ax.annotate("x=0일 때 높이 = 절편 b", (0, b), (0.35, 55), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=C["gray"]), color=C["gray"])
ax.text(2.2, 92, "분홍 세로선 = 잔차 (실제 - 예측)\n제곱해서 더한 값이 가장 작은 직선", fontsize=10, color=C["pink"])
ax.set(xlabel="공부시간 x (시간)", ylabel="점수 y", xlim=(-0.1, 6.5), ylim=(38, 104))
ax.legend(loc="lower right", fontsize=10)
save(fig, DAY, "linreg_residuals")

# ---------------------------------------------------------------- ① 실제 자료: 글자수 vs 소요시간
fig, ax = plt.subplots(figsize=(6.4, 3.7))
d = tr.dropna(subset=["input_chars"])
ax.scatter(d.input_chars, d.elapsed_ms, s=30, color=C["blue"], alpha=.8, label="train 요청 (결측 제외)")
l1 = LinearRegression().fit(d[["input_chars"]], d.elapsed_ms)
gx = np.linspace(150, 2450, 50)
ax.plot(gx, l1.predict(pd.DataFrame({"input_chars": gx})), color=C["purple"], lw=2.5,
        label=f"직선: 글자 1개당 {l1.coef_[0]:+.2f} ms")
ax.axhline(d.elapsed_ms.mean(), color=C["gray"], ls=":", lw=2, label="평균만 찍기 (기준선)")
ax.set(xlabel="input_chars (글자수)", ylabel="elapsed_ms (ms)")
ax.set_title("점들이 직선 주위에 넓게 흩어짐 → 글자수만으로는 신호가 약함", fontsize=11, fontweight="normal")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9.5, ncol=3)
save(fig, DAY, "real_chars_vs_ms")

# ---------------------------------------------------------------- ② MAE vs RMSE
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
pos = np.arange(3)
ax.bar(pos - .2, [0, 0, 100], .38, color=C["orange"], label="모델 A: 0, 0, 100")
ax.bar(pos + .2, [40, 40, 40], .38, color=C["mint"], label="모델 B: 40, 40, 40")
ax.set_xticks(pos, ["요청 1", "요청 2", "요청 3"])
ax.set(ylabel="|오차| (ms)", title="각 요청의 오차 크기", ylim=(0, 150))
ax.annotate("큰 실수 1번", (2 - .2, 100), (0.9, 112), arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax = axes[1]
vals = {"A": (100 / 3, np.sqrt(100 ** 2 / 3)), "B": (40, 40)}
pos = np.arange(2)
ba = ax.bar(pos - .2, [vals["A"][0], vals["B"][0]], .38, color=C["blue"], label="MAE")
br = ax.bar(pos + .2, [vals["A"][1], vals["B"][1]], .38, color=C["purple"], label="RMSE")
for bars in (ba, br):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5, f"{r.get_height():.1f}",
                ha="center", fontsize=10)
ax.set_xticks(pos, ["모델 A", "모델 B"])
ax.set(ylabel="ms", title="같은 오차, 다른 1등", ylim=(0, 82))
ax.legend(fontsize=10, loc="upper right", ncol=2)
ax.text(-.45, 66, "MAE로는 A가 작고\nRMSE로는 B가 작다", fontsize=10, color=C["ink"])
save(fig, DAY, "mae_vs_rmse")

# ---------------------------------------------------------------- ② R² 세 경우
rng = np.random.default_rng(0)
xs = np.linspace(0, 10, 14); ys = 3 * xs + 5 + rng.normal(0, 2.5, xs.size)
cases = [("좋은 모델", 3 * xs + 5, C["mint"]),
         ("평균만 찍기", np.full_like(xs, ys.mean()), C["gray"]),
         ("평균보다 나쁜 모델", -1.5 * xs + 40, C["red"])]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, (name, pred, col) in zip(axes, cases):
    r2 = 1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    for xi, yi, pi in zip(xs, ys, pred):
        ax.plot([xi, xi], [pi, yi], color=col, alpha=.55, lw=1.6)
    ax.plot(xs, pred, color=col, lw=2.5)
    ax.scatter(xs, ys, s=22, color=C["ink"], zorder=3)
    ax.set_title(f"{name}\nR² = {r2:.2f}", color=col if col != C["gray"] else C["ink"])
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
axes[2].text(3.6, 6, "오차(세로선)가\n평균 찍기보다 길다 → R² < 0", fontsize=10, color=C["red"])
axes[1].text(0.3, 33, "기준: R² = 0", fontsize=10, color=C["ink"])
save(fig, DAY, "r2_three_cases")

# ---------------------------------------------------------------- ③ 시그모이드
fig, ax = plt.subplots(figsize=(6.4, 3.6))
z = np.linspace(-6, 6, 300); p = 1 / (1 + np.exp(-z))
ax.axhspan(0.5, 1.02, color=FAIL, alpha=.07); ax.axhspan(-0.02, 0.5, color=OK, alpha=.07)
ax.plot(z, p, color=C["purple"], lw=3)
ax.axhline(0.5, color=C["gray"], ls="--", lw=1.5)
for zz, lab in [(-2, "z=-2 → p=0.119 → 0"), (0, "z=0 → p=0.500"), (2, "z=2 → p=0.881 → 1")]:
    pp = 1 / (1 + np.exp(-zz))
    ax.scatter([zz], [pp], s=70, color=C["ink"], zorder=3)
    off = {-2: (-5.8, 0.28), 0: (0.4, 0.40), 2: (2.4, 0.74)}[zz]
    ax.annotate(lab, (zz, pp), off, fontsize=10, arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.text(-5.8, 0.9, "p ≥ 0.5 → 실패(1)로 판단", color=FAIL, fontsize=10, fontweight="bold")
ax.text(3.0, 0.08, "p < 0.5 → 성공(0)", color=OK, fontsize=10, fontweight="bold")
ax.set(xlabel="선형 점수 z = w·x + b  (아무 값이나 가능)", ylabel="확률 p (0~1로 눌림)",
       ylim=(-0.02, 1.02), xlim=(-6, 6))
save(fig, DAY, "sigmoid")

# ---------------------------------------------------------------- ③ 로그 손실
fig, ax = plt.subplots(figsize=(6.2, 3.5))
pp = np.linspace(0.01, 1, 300)
ax.plot(pp, -np.log(pp), color=C["pink"], lw=3, label="실제 y=1일 때 손실 = -log(p)")
for v, txt, xy in [(0.9, "p=0.9 → 0.11\n(맞는 쪽에 자신 → 작은 벌점)", (0.5, 1.4)),
                   (0.1, "p=0.1 → 2.30\n(틀린 쪽에 자신 → 큰 벌점)", (0.17, 3.4))]:
    ax.scatter([v], [-np.log(v)], s=70, color=C["ink"], zorder=3)
    ax.annotate(txt, (v, -np.log(v)), xy, fontsize=10, arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.set(xlabel="모델이 준 실패 확률 p", ylabel="로그 손실", ylim=(0, 4.7), xlim=(0, 1.02))
ax.legend(loc="upper right", fontsize=10)
save(fig, DAY, "logloss")

# ---------------------------------------------------------------- ③ 혼동행렬
def cm_plot(ax, cm, title, names=("TN", "FP", "FN", "TP"),
            notes=("정상 통과", "괜히 경고", "놓침", "실패 발견")):
    cols = [[C["soft"], "#ffe3ef"], ["#ffe8cc", "#d3f9d8"]]
    for i in range(2):
        for j in range(2):
            k = i * 2 + j
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=cols[i][j], ec="white", lw=3))
            ax.text(j + .5, 1 - i + .62, f"{names[k]} {cm[i][j]}", ha="center", fontsize=16, fontweight="bold",
                    color=C["ink"])
            ax.text(j + .5, 1 - i + .3, notes[k], ha="center", fontsize=10, color=C["ink"])
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([.5, 1.5], ["예측 성공(0)", "예측 실패(1)"])
    ax.set_yticks([1.5, .5], ["실제 성공(0)", "실제 실패(1)"])
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title)


fig, ax = plt.subplots(figsize=(5, 4))
cm_plot(ax, [[10, 4], [4, 2]], "Logistic C=1 · valid 20건")
ax.text(1, -0.35, "F1 = 2·2 / (2·2 + 4 + 4) = 0.333   (TN은 F1에 안 들어감)", ha="center", fontsize=10)
save(fig, DAY, "confusion_logistic")

# ---------------------------------------------------------------- ③ 로지스틱 결정 경계 (2D 합성)
Xb, yb = make_moons(120, noise=.35, random_state=3)
fig, ax = plt.subplots(figsize=(6.2, 4))
lg = LogisticRegression().fit(Xb, yb)
g1, g2 = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.6, 2.1, 300))
P = lg.predict_proba(np.c_[g1.ravel(), g2.ravel()])[:, 1].reshape(g1.shape)
from matplotlib.colors import LinearSegmentedColormap
soft = LinearSegmentedColormap.from_list("soft", ["#efe9ff", "#ffffff", "#ffd3e4"])
cs = ax.contourf(g1, g2, P, levels=np.linspace(0, 1, 11), cmap=soft)
ax.contour(g1, g2, P, levels=[0.5], colors=[C["ink"]], linewidths=2.5)
ax.contour(g1, g2, P, levels=[0.3], colors=[C["orange"]], linewidths=2, linestyles="--")
ax.scatter(*Xb[yb == 0].T, color=OK, s=30, label="성공(0)", edgecolor=C["ink"], lw=.4)
ax.scatter(*Xb[yb == 1].T, color=FAIL, marker="s", s=30, label="실패(1)", edgecolor=C["ink"], lw=.4)
fig.colorbar(cs, ax=ax, label="실패 확률 p", ticks=[0, .5, 1])
ax.text(-1.9, -1.45, "검은 실선: p = 0.5 경계\n주황 점선: 임계값 0.3으로 내리면 경계가 이동", fontsize=9.5,
        color=C["ink"], bbox=dict(fc="white", ec="none", alpha=.8))
ax.set(xlabel="특성 1", ylabel="특성 2", title="로지스틱회귀는 '직선' 경계를 긋는다")
ax.legend(loc="upper left", fontsize=9, frameon=True)
save(fig, DAY, "logistic_boundary")

# ---------------------------------------------------------------- ④ KNN K=1 vs K=15
Xm, ym = make_moons(160, noise=.32, random_state=7)
cmap = ListedColormap(["#e5dbff", "#ffd6e7"])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
for ax, k in zip(axes, [1, 15]):
    m = KNeighborsClassifier(k).fit(Xm, ym)
    Z = m.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, Z, cmap=cmap, alpha=1)
    ax.contour(g1, g2, Z, levels=[.5], colors=[C["ink"]], linewidths=1.5)
    ax.scatter(*Xm[ym == 0].T, color=OK, s=20, edgecolor="white", lw=.5)
    ax.scatter(*Xm[ym == 1].T, color=FAIL, marker="s", s=20, edgecolor="white", lw=.5)
    acc = (m.predict(Xm) == ym).mean()
    ax.set_title(f"K = {k}   (train 정확도 {acc:.2f})")
    ax.set_xlabel("특성 1")
axes[0].set_ylabel("특성 2")
axes[0].text(-1.9, -1.45, "점 하나하나를 따라 섬처럼 구불구불\n→ 잡음까지 외움 (과적합)", fontsize=9.5,
             bbox=dict(fc="white", ec="none", alpha=.85))
axes[1].text(-1.9, -1.45, "15명에게 물어 다수결 → 매끄러운 경계\n(너무 크면 소수 패턴을 놓침)", fontsize=9.5,
             bbox=dict(fc="white", ec="none", alpha=.85))
save(fig, DAY, "knn_k1_vs_k15")

# ---------------------------------------------------------------- ④ 외삽: 선형 vs KNN
Xe = np.array([[1], [2], [3], [4]]); ye = np.array([10, 20, 30, 40])
lin = LinearRegression().fit(Xe, ye); kn = KNeighborsRegressor(2).fit(Xe, ye)
gx = np.linspace(0.5, 9, 300)[:, None]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axvspan(4.2, 9.2, color=C["gray"], alpha=.08)
ax.text(6.6, 3, "학습 자료가 없는 구간", ha="center", color=C["gray"], fontsize=10)
ax.plot(gx, lin.predict(gx), color=C["purple"], lw=2.5, label="선형회귀: 직선을 계속 연장")
ax.step(gx.ravel(), kn.predict(gx), where="mid", color=C["mint"], lw=2.5, label="KNN (K=2): 본 정답의 평균만")
ax.scatter(Xe, ye, s=70, color=C["ink"], zorder=3, label="학습 점")
ax.scatter([8, 8], [80, 35], s=110, marker="*", color=[C["purple"], C["mint"]], zorder=4)
ax.annotate("x=8 → 80", (8, 80), (6.2, 75), fontsize=10, color=C["purple"], arrowprops=dict(arrowstyle="->"))
ax.annotate("x=8 → 35\n(10~40 밖으로 못 나감)", (8, 35), (5.0, 15), fontsize=10, color=C["mint"],
            arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="x", ylabel="예측 y", ylim=(0, 92), xlim=(0.5, 9.2))
ax.legend(loc="upper left", fontsize=9.5)
save(fig, DAY, "extrapolation_linear_knn")

# ---------------------------------------------------------------- ⑤ 스케일링이 거리를 바꾼다
A, B, Q = np.array([1000, 1]), np.array([1400, 4]), np.array([1100, 4])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
ax = axes[0]
for pnt, lab, col, mk in [(A, "A (성공)", OK, "o"), (B, "B (실패)", FAIL, "s")]:
    ax.scatter(*pnt, s=120, color=col, marker=mk, zorder=3); ax.text(pnt[0] + (-95 if lab[0] == "A" else 15), pnt[1] + 25, lab, fontsize=11)
ax.scatter(*Q, s=220, marker="*", color=C["orange"], zorder=4); ax.text(Q[0] - 10, Q[1] + 40, "Q (새 요청)", fontsize=11)
ax.plot([Q[0], A[0]], [Q[1], A[1]], color=OK, ls="--"); ax.plot([Q[0], B[0]], [Q[1], B[1]], color=FAIL, ls="--")
ax.text(960, -110, "Q-A ≈ 100", color=OK, fontsize=11, fontweight="bold")
ax.text(1210, -70, "Q-B = 300", color=FAIL, fontsize=11, fontweight="bold")
ax.set_xlim(900, 1500); ax.set_ylim(-150, 450); ax.set_aspect("equal")
ax.set(xlabel="글자수", ylabel="도구수", title="원래 단위 (두 축 같은 눈금)")
ax.text(905, 400, "도구수 1→4 차이(3)는\n거의 안 보일 만큼 납작!", fontsize=10, va="top")
ax = axes[1]
mu, sd = np.array([1200, 2.5]), np.array([200, 1.5])
a, bb, q = (A - mu) / sd, (B - mu) / sd, (Q - mu) / sd
for pnt, lab, col, mk in [(a, "A (성공)", OK, "o"), (bb, "B (실패)", FAIL, "s")]:
    ax.scatter(*pnt, s=120, color=col, marker=mk, zorder=3); ax.text(pnt[0] + .07, pnt[1] - .25, lab, fontsize=11)
ax.scatter(*q, s=220, marker="*", color=C["orange"], zorder=4); ax.text(q[0] - .35, q[1] + .17, "Q", fontsize=11)
ax.plot([q[0], a[0]], [q[1], a[1]], color=OK, ls="--"); ax.plot([q[0], bb[0]], [q[1], bb[1]], color=FAIL, ls="--")
ax.text(-1.45, 0, "Q-A ≈ 2.06", color=OK, fontsize=11, fontweight="bold")
ax.text(0, 1.17, "Q-B = 1.5 (더 가까움!)", color=FAIL, fontsize=11, fontweight="bold")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
ax.set(xlabel="글자수 (표준화)", ylabel="도구수 (표준화)", title="표준화 후 (같은 눈금)")
save(fig, DAY, "scaling_distance")

# ---------------------------------------------------------------- ⑤ Standard vs MinMax 숫자선
fig, axes = plt.subplots(3, 1, figsize=(6.6, 3.6), sharex=False)
rows = [("원래 값", [100, 200, 300], 500, (0, 600)),
        ("StandardScaler", [-1.225, 0, 1.225], (500 - 200) / 81.65, (-2, 4.2)),
        ("MinMaxScaler", [0, .5, 1], 2.0, (-0.3, 2.3))]
for ax, (name, vals, new, lim) in zip(axes, rows):
    ax.axhline(0, color=C["gray"], lw=1)
    ax.scatter(vals, [0] * 3, s=90, color=C["purple"], zorder=3)
    ax.scatter([new], [0], s=110, color=C["orange"], marker="D", zorder=3)
    for v in vals:
        ax.text(v, .5, f"{v:g}", ha="center", fontsize=10)
    ax.text(new, .5, f"새 값 {new:.3g}", ha="center", fontsize=10, color=C["orange"])
    if name == "MinMaxScaler":
        ax.axvspan(0, 1, color=C["mint"], alpha=.12); ax.text(.5, -.8, "train 범위 0~1", ha="center", fontsize=9, color=C["mint"])
    ax.set_xlim(*lim); ax.set_ylim(-1.1, 1.1); ax.set_yticks([]); ax.grid(False)
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
    ax.set_xticks([])
axes[0].set_title("train 100·200·300으로 fit → 새 값 500을 transform", fontsize=11)
save(fig, DAY, "scaler_numberline")

# ---------------------------------------------------------------- ⑥ K별 train/valid F1 · RMSE (실제 계산)
Ks = [1, 3, 5, 7, 9, 11, 15]
trf, vaf, rm = [], [], []
for k in Ks:
    c = build(KNeighborsClassifier(k)).fit(tr[FEATURES], tr.failed)
    trf.append(f1_score(tr.failed, c.predict(tr[FEATURES]), zero_division=0))
    vaf.append(f1_score(va.failed, c.predict(va[FEATURES]), zero_division=0))
    r = build(KNeighborsRegressor(k)).fit(tr[FEATURES], tr.elapsed_ms)
    rm.append(root_mean_squared_error(va.elapsed_ms, r.predict(va[FEATURES])))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
ax = axes[0]
ax.plot(Ks, trf, "o-", color=C["gray"], lw=2, label="train F1")
ax.plot(Ks, vaf, "o-", color=C["purple"], lw=2.5, label="valid F1")
ax.annotate("K=1: train 1.000\n(자기 자신이 이웃)", (1, 1), (3.6, .93), fontsize=9.5, arrowprops=dict(arrowstyle="->"))
ax.annotate("valid 최고 K=3", (3, vaf[1]), (4.6, .7), fontsize=9.5, color=C["purple"], arrowprops=dict(arrowstyle="->", color=C["purple"]))
ax.annotate("K=9·15: 다수인 '성공'만\n예측 → 실패 F1 = 0", (15, 0.02), (9.6, .6), fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.set(xlabel="K (이웃 수)", ylabel="실패 클래스 F1", title="분류: 클수록 좋음", ylim=(-0.05, 1.1), xticks=Ks)
ax.legend(fontsize=9.5, loc="upper right")
ax = axes[1]
ax.plot(Ks, rm, "o-", color=C["mint"], lw=2.5, label="valid RMSE")
ax.annotate("K=15 최소 641.8 ms", (15, rm[-1]), (9.6, 790), fontsize=9.5, color=C["mint"], arrowprops=dict(arrowstyle="->", color=C["mint"]))
ax.axhline(658.75, color=C["gray"], ls=":", lw=1.8); ax.text(1, 665, "Dummy(평균) 658.8", fontsize=9, color=C["gray"])
ax.set(xlabel="K (이웃 수)", ylabel="ms", title="회귀: 작을수록 좋음", xticks=Ks)
ax.legend(fontsize=9.5, loc="upper right")
save(fig, DAY, "knn_k_search")

# ---------------------------------------------------------------- ⑥ 모델 비교: F1 + 오류 수
names = ["Dummy", "Logistic C=1", "KNN K=3"]
f1s = [0, .333, .545]; fps = [0, 4, 2]; fns = [6, 4, 3]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
ax = axes[0]
bars = ax.barh(names, f1s, color=[C["gray"], C["blue"], C["purple"]])
for bar, v in zip(bars, f1s):
    ax.text(v + .01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=10)
ax.set(xlim=(0, .7), xlabel="valid F1 (실패 클래스)", title="주 지표")
ax.invert_yaxis()
ax.text(.12, 0, "← Accuracy는 0.70인데 F1 = 0", va="center", fontsize=9.5, color=C["gray"])
ax = axes[1]
yy = np.arange(3)
ax.barh(yy - .2, fps, .38, color=C["orange"], label="FP 헛경고")
ax.barh(yy + .2, fns, .38, color=C["red"], label="FN 놓침")
ax.set_yticks(yy, names); ax.invert_yaxis()
ax.set(xlabel="건수 (valid 실패 6건 중)", title="어떤 오류를 냈나", xlim=(0, 7))
ax.legend(fontsize=9.5, loc="lower right")
save(fig, DAY, "model_compare")
print("done")
