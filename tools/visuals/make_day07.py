import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]   # 없는 기호는 DejaVu로 대체
"""Day7 그림: 트리 · Gini · 과적합 · Random Forest · 중요도 · 부스팅 · 교차검증.

실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day07.py
수업 데이터(day6_compare, Day7과 같은 계약)는 읽기만 한다.
"""
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.datasets import make_moons
from sklearn.metrics import f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from xgboost import XGBClassifier

DAY = 7
OK, FAIL = C["purple"], C["pink"]
DATA = "C:/practice_assignment/20261026/03_심화/day6_compare/data/"
NUM = ["input_chars", "allowed_tools"]
CAT = ["task_type", "priority"]
FEATURES = NUM + CAT


def load():
    df = pd.read_csv(DATA + "requests.csv").set_index("request_id")
    man = pd.read_csv(DATA + "split_manifest.csv")
    part = lambda s: df.loc[man[man.split == s].sort_values("position").request_id].reset_index()
    return part("train"), part("valid"), part("test")


def build(est, scale="none"):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler() if scale == "standard" else "passthrough")])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return Pipeline([("pre", ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])),
                     ("model", est)])


tr, va, te = load()
Xtr, ytr, Xva, yva = tr[FEATURES], tr.failed, va[FEATURES], va.failed
g1, g2 = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.6, 2.1, 300))
GRID = np.c_[g1.ravel(), g2.ravel()]
REGION = ListedColormap(["#e5dbff", "#ffd6e7"])


def boundary(ax, model, X, y, title):
    Z = model.predict(GRID).reshape(g1.shape)
    ax.contourf(g1, g2, Z, cmap=REGION)
    ax.contour(g1, g2, Z, levels=[.5], colors=[C["ink"]], linewidths=1.3)
    ax.scatter(*X[y == 0].T, color=OK, s=18, edgecolor="white", lw=.4)
    ax.scatter(*X[y == 1].T, color=FAIL, marker="s", s=18, edgecolor="white", lw=.4)
    ax.set_title(title, fontsize=12); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def box(ax, text, color=C["ink"]):
    ax.text(.02, .03, text, transform=ax.transAxes, fontsize=9.5, color=color,
            bbox=dict(fc="white", ec="none", alpha=.88))


# ---------------------------------------------------------------- ① Gini 곡선
fig, ax = plt.subplots(figsize=(6.2, 3.5))
p = np.linspace(0, 1, 200)
ax.plot(p, 1 - p ** 2 - (1 - p) ** 2, color=C["purple"], lw=3)
for pp, lab, xy in [(0, "성공 3 · 실패 0\nGini 0 (깨끗)", (0.08, .03)), (.5, "성공 3 · 실패 3\nGini 0.5 (최대로 섞임)", (.36, .3)),
                    (.75, "성공 1 · 실패 3\nGini 0.375", (.78, .43))]:
    gv = 1 - pp ** 2 - (1 - pp) ** 2
    ax.scatter([pp], [gv], s=70, color=C["ink"], zorder=3)
    ax.annotate(lab, (pp, gv), xy, fontsize=10, arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.set(xlabel="노드 안의 실패 비율", ylabel="Gini 불순도", ylim=(0, .6), xlim=(-.02, 1.02))
save(fig, DAY, "gini_curve")

# ---------------------------------------------------------------- ① 실제 트리 그림 (수업 기본 트리)
tree_pipe = build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)).fit(Xtr, ytr)
names = [n.split("__")[1] for n in tree_pipe["pre"].get_feature_names_out()]
fig, ax = plt.subplots(figsize=(10, 4.6))
plot_tree(tree_pipe["model"], feature_names=names, class_names=["성공", "실패"], filled=True, rounded=True,
          impurity=True, proportion=False, fontsize=9, ax=ax)
ax.set_title("수업 기본 트리 (max_depth=3, min_samples_leaf=3) · train 60건", fontsize=12)
save(fig, DAY, "tree_plot")

# ---------------------------------------------------------------- ① 회귀 트리 = 계단
X6 = np.array([[100], [200], [300], [700], [800], [900]]); y6 = np.array([100, 120, 140, 700, 720, 740])
rt = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X6, y6)
lin = LinearRegression().fit(X6, y6)
gx = np.linspace(0, 1300, 400)[:, None]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.axvspan(950, 1300, color=C["gray"], alpha=.08)
ax.plot(gx, lin.predict(gx), color=C["gray"], lw=1.8, ls="--", label="(비교) 선형회귀 직선")
ax.step(gx.ravel(), rt.predict(gx), where="mid", color=C["mint"], lw=3, label="회귀 트리 (깊이 1)")
ax.scatter(X6, y6, s=60, color=C["ink"], zorder=3, label="학습 점")
ax.axvline(500, color=C["purple"], ls=":", lw=2); ax.text(510, 360, "질문: 글자수 ≤ 500?", color=C["purple"], fontsize=10)
ax.text(120, 200, "왼쪽 잎 평균 120", color=C["mint"], fontsize=10)
ax.text(560, 770, "오른쪽 잎 평균 720", color=C["mint"], fontsize=10)
ax.annotate("1200도 그냥 720\n(외삽 안 함)", (1200, 720), (980, 480), fontsize=10, arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="글자수", ylabel="예측 (ms)", xlim=(0, 1300), ylim=(-50, 1100))
ax.legend(loc="upper left", fontsize=9)
save(fig, DAY, "tree_regression_step")

# ---------------------------------------------------------------- ② 깊이별 train/valid F1 (실제)
depths = [1, 2, 3, 4, 5, 6, None]
trs, vas, leaves = [], [], []
for d in depths:
    m = build(DecisionTreeClassifier(max_depth=d, min_samples_leaf=1, random_state=42)).fit(Xtr, ytr)
    trs.append(f1_score(ytr, m.predict(Xtr))); vas.append(f1_score(yva, m.predict(Xva), zero_division=0))
    leaves.append(m["model"].get_n_leaves())
print("depth", list(zip(depths, np.round(trs, 3), np.round(vas, 3), leaves)))
xs = np.arange(len(depths)); lab = [str(d) if d else "None" for d in depths]
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(xs, trs, "o-", color=C["gray"], lw=2.2, label="train F1")
ax.plot(xs, vas, "o-", color=C["purple"], lw=2.8, label="valid F1")
ax.fill_between(xs, vas, trs, color=C["pink"], alpha=.12)
ax.text(xs[-1] - .1, (trs[-1] + vas[-1]) / 2, "이 틈 =\n과적합 신호", ha="right", va="center", fontsize=10, color=C["pink"])
ax.annotate("깊이 1: 둘 다 낮음\n(과소적합)", (0, vas[0]), (0.3, .05), fontsize=9.5, arrowprops=dict(arrowstyle="->"))
ax.set_xticks(xs, [f"{l}\n(잎 {n})" for l, n in zip(lab, leaves)])
ax.set(xlabel="max_depth (괄호: 잎 개수)", ylabel="실패 클래스 F1", ylim=(0, 1.08))
ax.legend(loc="upper left", fontsize=10)
save(fig, DAY, "depth_overfit")

# ---------------------------------------------------------------- ② 깊이별 경계 (합성 2D)
Xm, ym = make_moons(160, noise=.32, random_state=7)
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
t2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xm, ym)
tn = DecisionTreeClassifier(random_state=0).fit(Xm, ym)
boundary(axes[0], t2, Xm, ym, f"max_depth=2 (잎 {t2.get_n_leaves()}개)")
boundary(axes[1], tn, Xm, ym, f"max_depth=None (잎 {tn.get_n_leaves()}개)")
box(axes[0], "질문 2번 → 가로·세로 선 몇 개로 크게 나눔\n(곡선 모양은 못 따라감)")
box(axes[1], "점 하나까지 따로 가두는 작은 직사각형들\n→ train 100% · 잡음까지 외움")
save(fig, DAY, "tree_boundary_depths")

# ---------------------------------------------------------------- ③ 숲 = 여러 트리의 평균 (1D 회귀)
rng = np.random.default_rng(1)
x1 = np.sort(rng.uniform(0, 6, 50)); y1 = np.sin(x1) + rng.normal(0, .35, x1.size)
gx = np.linspace(0, 6, 400)[:, None]
rf1 = RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0).fit(x1[:, None], y1)
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for i, est in enumerate(rf1.estimators_[:5]):
    ax.plot(gx, est.predict(gx), color=C["blue"], lw=1.1, alpha=.6, label="개별 트리 5개 (서로 다른 표본)" if i == 0 else None)
ax.plot(gx, rf1.predict(gx), color=C["purple"], lw=3, label="숲 = 트리 100개 평균")
ax.plot(gx, np.sin(gx), color=C["ink"], lw=1.5, ls="--", label="진짜 패턴 (sin)")
ax.scatter(x1, y1, s=16, color=C["gray"], zorder=3)
ax.set(xlabel="x", ylabel="y", ylim=(-2.1, 2.3))
ax.legend(fontsize=9, loc="lower left", ncol=1)
ax.text(2.6, 1.75, "파란 선(트리 1개씩)은 서로 다른 곳에서 튀고,\n보라 평균은 그 튐이 줄어 더 차분", fontsize=10)
save(fig, DAY, "rf_averaging")

# ---------------------------------------------------------------- ③ 트리 1개 vs 숲 경계
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
rfc = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0).fit(Xm, ym)
boundary(axes[0], tn, Xm, ym, "트리 1개 (깊이 제한 없음)")
boundary(axes[1], rfc, Xm, ym, "Random Forest (트리 100개 평균)")
box(axes[0], "각진 섬이 여기저기 → 잡음에 민감")
box(axes[1], "섬이 줄고 경계가 부드러워짐\n(트리마다 다른 실수가 평균에서 상쇄)")
save(fig, DAY, "rf_vs_tree_boundary")

# ---------------------------------------------------------------- ④ MDI vs 순열 중요도 (실제 Forest)
forest = build(RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, max_features="sqrt",
                                      random_state=42, n_jobs=1)).fit(Xtr, ytr)
mdi = pd.Series(forest["model"].feature_importances_,
                index=[n.split("__")[1] for n in forest["pre"].get_feature_names_out()]).sort_values()
perm = permutation_importance(forest, Xva, yva, scoring="f1", n_repeats=20, random_state=42, n_jobs=1)
pm = pd.DataFrame({"m": perm.importances_mean, "s": perm.importances_std}, index=FEATURES).sort_values("m")
print("mdi", mdi.round(4).to_dict()); print("perm", pm.round(4).to_dict())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
ax = axes[0]
ax.barh(mdi.index, mdi.values, color=C["blue"])
for i, v in enumerate(mdi.values): ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=9)
ax.set(title="MDI · train에서 분할에 쓰인 정도\n(변환 후 8열, 합 = 1)", xlabel="불순도 감소 기여", xlim=(0, .7))
ax = axes[1]
cols = [C["red"] if v < 0 else C["purple"] for v in pm.m]
ax.barh(pm.index, pm.m, xerr=pm.s, color=cols, capsize=4, error_kw=dict(ecolor=C["gray"]))
ax.axvline(0, color=C["ink"], lw=1)
for i, v in enumerate(pm.m): ax.text(v + (.01 if v >= 0 else -.01), i + .3, f"{v:+.3f}", va="center",
                                     ha="left" if v >= 0 else "right", fontsize=9)
ax.set(title="순열 중요도 · valid 20건에서 섞었을 때\nF1 감소 (원본 4열, 막대=20회 흔들림)", xlabel="F1 감소량")
save(fig, DAY, "importance_mdi_perm")

# ---------------------------------------------------------------- ⑤ 부스팅 한 단계 (4점 예)
y4 = np.array([100., 120., 200., 220.]); x4 = np.arange(1, 5)
init = np.full(4, y4.mean())
corr = DecisionTreeRegressor(max_depth=1, random_state=42).fit(x4[:, None], y4 - init).predict(x4[:, None])
upd = init + .5 * corr
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharey=True)
for ax, pred, title in [(axes[0], init, f"처음: 모두 평균 160  (MSE {np.mean((y4 - init) ** 2):.0f})"),
                        (axes[1], upd, f"보정 1회 후 (학습률 0.5)  (MSE {np.mean((y4 - upd) ** 2):.0f})")]:
    ax.bar(x4, y4, .55, color=C["soft"], ec=C["purple"], label="실제 y")
    ax.scatter(x4, pred, s=80, color=C["orange"], zorder=3, marker="D", label="예측")
    for xi, yi, pi in zip(x4, y4, pred):
        ax.annotate("", (xi, yi), (xi, pi), arrowprops=dict(arrowstyle="->", color=C["pink"], lw=2))
        ax.text(xi + .3, (yi + pi) / 2, f"{yi - pi:+.0f}", color=C["pink"], fontsize=10, va="center")
    ax.set_title(title, fontsize=11); ax.set_xticks(x4, [f"x={i}" for i in x4]); ax.set_ylim(0, 260)
axes[0].set_ylabel("ms"); axes[0].legend(loc="upper left", fontsize=9)
axes[1].text(0.6, 245, "분홍 화살표(잔차)가 모두 짧아졌다", fontsize=10, color=C["pink"])
save(fig, DAY, "boosting_one_step")

# ---------------------------------------------------------------- ⑤ 부스팅 회차가 쌓이면
lr_, preds, F = .3, {}, np.full_like(y1, y1.mean())
Fg = np.full(gx.shape[0], y1.mean())
for r in range(1, 81):
    t = DecisionTreeRegressor(max_depth=2, random_state=0).fit(x1[:, None], y1 - F)
    F = F + lr_ * t.predict(x1[:, None]); Fg = Fg + lr_ * t.predict(gx)
    if r in (1, 3, 10, 80): preds[r] = (Fg.copy(), np.mean((y1 - F) ** 2))
fig, axes = plt.subplots(1, 4, figsize=(11, 3.1), sharey=True)
for ax, (r, (fg, mse)) in zip(axes, preds.items()):
    ax.scatter(x1, y1, s=10, color=C["gray"])
    ax.plot(gx, fg, color=C["purple"], lw=2.5)
    ax.set_title(f"{r}회 보정 · train MSE {mse:.2f}", fontsize=10.5); ax.set_xticks([])
axes[0].set_ylabel("y")
axes[0].text(.2, -1.9, "처음엔 평균 근처의\n밋밋한 계단", fontsize=9)
axes[3].text(.2, -1.9, "회차가 쌓이면 잡음까지\n따라가기 시작", fontsize=9)
save(fig, DAY, "boosting_rounds")

# ---------------------------------------------------------------- ⑤ 조기 종료: train vs stop 손실
Xs, ys_ = make_moons(300, noise=.45, random_state=2)
idx = np.random.default_rng(0).permutation(300); a, b = idx[:200], idx[200:]
xg = XGBClassifier(n_estimators=400, learning_rate=.1, max_depth=4, subsample=.8, eval_metric="logloss",
                   tree_method="hist", device="cpu", random_state=42, n_jobs=1)
xg.fit(Xs[a], ys_[a], eval_set=[(Xs[a], ys_[a]), (Xs[b], ys_[b])], verbose=False)
ev = xg.evals_result(); trl, stl = ev["validation_0"]["logloss"], ev["validation_1"]["logloss"]
best = int(np.argmin(stl))
fig, ax = plt.subplots(figsize=(6.6, 3.6))
ax.plot(trl, color=C["gray"], lw=2.2, label="학습 자료 손실")
ax.plot(stl, color=C["purple"], lw=2.8, label="stop 자료 손실 (안 본 자료)")
ax.axvline(best, color=C["orange"], ls="--", lw=2)
ax.annotate(f"여기서 멈춤 (회차 {best})\nstop 손실이 가장 낮음", (best, stl[best]), (best + 40, .5), fontsize=10,
            color=C["orange"], arrowprops=dict(arrowstyle="->", color=C["orange"]))
ax.text(230, .36, "계속 돌리면 다시 올라감\n= 잡음을 보정하는 중", fontsize=9.5, color=C["purple"])
ax.set(xlabel="보정 회차 (트리 수)", ylabel="logloss (작을수록 좋음)", ylim=(0, .75))
ax.legend(loc="upper right", fontsize=9.5)
save(fig, DAY, "xgb_early_stopping")

# ---------------------------------------------------------------- ⑥ 5-fold 블록
fig, ax = plt.subplots(figsize=(7.4, 3.3))
for r in range(5):
    for c in range(5):
        ev_ = (r == c)
        ax.add_patch(plt.Rectangle((c * 1.05, 4 - r), 1, .8, color=C["pink"] if ev_ else C["soft"],
                                   ec=C["purple"] if not ev_ else C["pink"], lw=1))
        ax.text(c * 1.05 + .5, 4 - r + .4, "평가 12" if ev_ else "학습", ha="center", va="center", fontsize=10,
                color="white" if ev_ else C["ink"], fontweight="bold" if ev_ else "normal")
    ax.text(-.15, 4 - r + .4, f"{r + 1}회차", ha="right", va="center", fontsize=10)
    ax.text(5.35, 4 - r + .4, "→ 점수 " + ["①", "②", "③", "④", "⑤"][r], va="center", fontsize=10)
ax.text(2.6, 5.05, "train 60건을 12건씩 5조각", ha="center", fontsize=11, fontweight="bold")
ax.text(2.6, -.45, "모든 행이 딱 한 번씩 평가 칸에 들어감 → 점수 5개의 평균과 흩어짐을 같이 본다", ha="center", fontsize=9.5)
ax.set_xlim(-1.1, 6.9); ax.set_ylim(-.7, 5.3); ax.axis("off")
save(fig, DAY, "kfold_blocks")

# ---------------------------------------------------------------- ⑥ KFold vs StratifiedKFold: fold별 실패 수
yv = ytr.to_numpy()
fig, ax = plt.subplots(figsize=(6.6, 3.5))
w = .38
for k, (cvs, col, name) in enumerate([(KFold(5, shuffle=True, random_state=0), C["gray"], "KFold (그냥 섞어 자르기)"),
                                      (StratifiedKFold(5, shuffle=True, random_state=42), C["purple"], "StratifiedKFold")]):
    cnt = [yv[te_].sum() for _, te_ in cvs.split(np.zeros(60), yv)]
    bars = ax.bar(np.arange(5) + (k - .5) * w, cnt, w, color=col, label=name)
    for bb, v in zip(bars, cnt): ax.text(bb.get_x() + w / 2, v + .1, str(v), ha="center", fontsize=10)
ax.axhline(yv.mean() * 12, color=C["pink"], ls="--", lw=1.8)
ax.text(4.55, yv.mean() * 12 + .15, f"기대값 {yv.mean() * 12:.1f}", color=C["pink"], fontsize=9.5, ha="right")
ax.set_xticks(range(5), [f"fold {i + 1}" for i in range(5)])
ax.set(ylabel="평가 12건 중 실패 수", ylim=(0, 7.5), title="train 60건 · 실패 18건을 5조각으로")
ax.legend(fontsize=9.5, loc="upper left")
save(fig, DAY, "stratified_vs_kfold")

# ---------------------------------------------------------------- ⑥ 모델별 fold F1 점 + 평균
CV = StratifiedKFold(5, shuffle=True, random_state=42)
cands = {
    "Dummy": build(DummyClassifier(strategy="most_frequent"), "standard"),
    "Logistic": build(LogisticRegression(C=1.0, max_iter=1000), "standard"),
    "KNN3": build(KNeighborsClassifier(3), "standard"),
    "Tree": build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
    "Forest": build(RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, max_features="sqrt",
                                           random_state=42, n_jobs=1)),
    # XGB는 이 환경의 xgboost 버전에서 fold 점수가 강의 표(0.500)와 조금 달라 그림에서 뺐다
}
fig, ax = plt.subplots(figsize=(7.4, 3.9))
for i, (n, pipe) in enumerate(cands.items()):
    s = cross_validate(pipe, Xtr, ytr, cv=CV, scoring="f1", error_score="raise")["test_score"]
    print(n, s.round(3), round(s.mean(), 3), round(s.std(), 3))
    jitter = np.linspace(-.12, .12, 5)
    ax.scatter(i + jitter, s, s=34, color=C["blue"], alpha=.75, zorder=3, label="fold 점수 5개" if i == 0 else None)
    ax.errorbar(i + .32, s.mean(), yerr=s.std(), fmt="D", color=C["purple"], ms=7, capsize=5, lw=2,
                label="평균 ± 표준편차" if i == 0 else None)
    ax.text(i + .32, s.mean() + s.std() + .04, f"{s.mean():.3f}", ha="center", fontsize=9.5, color=C["purple"])
ax.set_xticks(range(len(cands)), list(cands))
ax.set(ylabel="fold 평가 F1", ylim=(-.05, 1.0), title="5-fold CV · 같은 fold로 모든 후보 비교")
ax.legend(fontsize=9.5, loc="upper right")
save(fig, DAY, "cv_fold_scores")
print("done")
