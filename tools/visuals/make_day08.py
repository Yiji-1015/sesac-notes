import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]   # 없는 기호는 DejaVu로 대체
"""Day8 그림: 분류 지표 · ROC/PR · 임계값 · RMSE 집계 · log1p · K-Means · 최종 평가.

실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day08.py
수업 데이터(day6_compare, Day8과 같은 계약)는 읽기만 한다.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, silhouette_score

DAY = 8
OK, FAIL = C["purple"], C["pink"]
DATA = "C:/practice_assignment/20261026/03_심화/day6_compare/data/"
NUM = ["input_chars", "allowed_tools"]


def load():
    df = pd.read_csv(DATA + "requests.csv").set_index("request_id")
    man = pd.read_csv(DATA + "split_manifest.csv")
    part = lambda s: df.loc[man[man.split == s].sort_values("position").request_id].reset_index()
    return part("train"), part("valid"), part("test")


tr, va, te = load()


def cm_axes(ax, cm, title, hl=None):
    names, notes = ("TN", "FP", "FN", "TP"), ("정상 통과", "오탐", "놓침", "실패 발견")
    for i in range(2):
        for j in range(2):
            k = i * 2 + j
            on = hl is None or (hl == "col" and j == 1) or (hl == "row" and i == 1)
            col = {"col": "#ffd6e7", "row": "#d0ebff", None: ["#f2ecff", "#ffe3ef", "#ffe8cc", "#d3f9d8"][k]}[hl] if on else "#f4f4f6"
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=col, ec="white", lw=3))
            ax.text(j + .5, 1 - i + .6, f"{names[k]} {cm[i][j]}", ha="center", fontsize=15, fontweight="bold",
                    color=C["ink"] if on else "#b9b4c6")
            ax.text(j + .5, 1 - i + .3, notes[k], ha="center", fontsize=10, color=C["ink"] if on else "#b9b4c6")
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.set_aspect("equal")
    ax.set_xticks([.5, 1.5], ["예측 0 · 성공", "예측 1 · 실패"]); ax.set_yticks([1.5, .5], ["실제 0 · 성공", "실제 1 · 실패"])
    ax.tick_params(length=0); ax.grid(False)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=12)


# ---------------------------------------------------------------- ① Precision vs Recall 분모
cm = [[10, 4], [4, 2]]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3))
cm_axes(axes[0], cm, "Precision = TP / (예측 1 열 전체)", "col")
cm_axes(axes[1], cm, "Recall = TP / (실제 1 행 전체)", "row")
axes[0].text(1, -.38, "\"실패\"라고 경고한 6건 중 진짜 2건 → 0.333", ha="center", fontsize=10)
axes[1].text(1, -.38, "진짜 실패 6건 중 찾은 2건 → 0.333", ha="center", fontsize=10)
save(fig, DAY, "precision_recall_denominator")

# ---------------------------------------------------------------- ① 조화평균 vs 산술평균
R = np.linspace(0.01, 1, 200)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(R, (1 + R) / 2, color=C["gray"], lw=2.5, ls="--", label="산술평균 (P+R)/2")
ax.plot(R, 2 * R / (1 + R), color=C["purple"], lw=3, label="F1 = 조화평균 2PR/(P+R)")
ax.scatter([.01, .01], [.505, 2 * .01 / 1.01], color=[C["gray"], C["purple"]], s=60, zorder=3)
ax.annotate("Recall 0.01이어도 0.505\n(그럴듯해 보임)", (.01, .505), (.12, .72), fontsize=10,
            arrowprops=dict(arrowstyle="->", color=C["gray"]))
ax.annotate("F1 ≈ 0.02\n(낮은 쪽에 끌려 내려감)", (.01, .02), (.2, .12), fontsize=10, color=C["purple"],
            arrowprops=dict(arrowstyle="->", color=C["purple"]))
ax.set(xlabel="Recall (Precision은 1.0으로 고정)", ylabel="점수", ylim=(0, 1.05), xlim=(0, 1.02))
ax.legend(loc="lower right", fontsize=10)
save(fig, DAY, "harmonic_mean")

# ---------------------------------------------------------------- ② 점수 분포 + ROC (페이지 데모와 같은 20건)
pos = np.array([0.28, 0.46, 0.58, 0.67, 0.81, 0.92])
neg = np.array([0.04, 0.09, 0.13, 0.18, 0.22, 0.26, 0.31, 0.36, 0.41, 0.44, 0.52, 0.55, 0.63, 0.74])
y = np.r_[np.ones(6), np.zeros(14)].astype(int); s = np.r_[pos, neg]
auc = roc_auc_score(y, s)
fpr, tpr, thr = roc_curve(y, s)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), gridspec_kw={"width_ratios": [1, 1]})
ax = axes[0]
x4 = np.array([0, 0, 1, 1]); p4 = np.array([.10, .40, .35, .80])
f4, t4, _ = roc_curve(x4, p4)
ax.fill_between(f4, t4, step="post", color=C["purple"], alpha=.15)
ax.step(f4, t4, where="post", color=C["purple"], lw=3)
ax.plot([0, 1], [0, 1], color=C["gray"], ls=":", lw=1.5)
ax.text(.55, .25, f"AUC = {roc_auc_score(x4, p4):.2f}\n(4쌍 중 3쌍 순서 맞음)", fontsize=11, color=C["purple"])
ax.text(.02, .92, "무작위 순서면 대각선 (AUC 0.5)", fontsize=9, color=C["gray"], rotation=0)
ax.set(title="코드 예시 4건 · 점수 0.10/0.40/0.35/0.80", xlabel="FPR (성공 중 잘못 경고)", ylabel="TPR = Recall (실패 중 찾음)",
       xlim=(-.02, 1.02), ylim=(-.02, 1.05)); ax.set_aspect("equal")
ax = axes[1]
ax.fill_between(fpr, tpr, step="post", color=C["pink"], alpha=.15)
ax.step(fpr, tpr, where="post", color=C["pink"], lw=3)
ax.plot([0, 1], [0, 1], color=C["gray"], ls=":", lw=1.5)
for t, col, off in [(.5, C["ink"], (.18, .55)), (.3, C["orange"], (.45, .78))]:
    tp_ = (pos > t).sum(); fp_ = (neg > t).sum()
    ax.scatter([fp_ / 14], [tp_ / 6], s=90, color=col, zorder=4)
    ax.annotate(f"t={t}: FP {fp_}, TP {tp_}", (fp_ / 14, tp_ / 6), ((.33, .42) if t == .5 else (.45, .68)),
                fontsize=10, color=col, arrowprops=dict(arrowstyle="->", color=col))
ax.text(.55, .12, f"AUC = {auc:.3f}", fontsize=12, color=C["pink"], fontweight="bold")
ax.set(title="슬라이더 데모 20건 (실패 6 · 성공 14)", xlabel="FPR", xlim=(-.02, 1.02), ylim=(-.02, 1.05)); ax.set_aspect("equal")
save(fig, DAY, "roc_auc")

# ---------------------------------------------------------------- ② 임계값에 따른 FP/FN/F1/비용 + AUC는 평평
ts = np.round(np.arange(.05, .951, .05), 2)
FP = np.array([(neg > t).sum() for t in ts]); FN = np.array([(pos <= t).sum() for t in ts]); TP = 6 - FN
F1 = 2 * TP / (2 * TP + FP + FN)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
ax = axes[0]
ax.plot(ts, FP, "o-", color=C["orange"], lw=2.2, ms=4, label="FP 오탐")
ax.plot(ts, FN, "o-", color=C["red"], lw=2.2, ms=4, label="FN 놓침")
ax.plot(ts, FP + 5 * FN, "-", color=C["ink"], lw=2, alpha=.7, label="비용 = FP + 5×FN (가정)")
best = ts[np.argmin(FP + 5 * FN)]
ax.axvline(best, color=C["mint"], ls="--"); ax.text(best + .02, 20, f"비용 최소 t={best}", color=C["mint"], fontsize=10)
ax.set(xlabel="임계값 t (p > t → 실패)", ylabel="건수 / 비용", title="선을 올리면 FP↓ FN↑", ylim=(0, 42))
ax.legend(fontsize=9, loc="upper left")
ax = axes[1]
ax.plot(ts, F1, "o-", color=C["purple"], lw=2.5, ms=4, label="F1")
ax.axhline(auc, color=C["pink"], lw=2.5, label=f"AUC {auc:.3f} (항상 그대로)")
ax.axvline(.5, color=C["gray"], ls=":", lw=1.5); ax.text(.51, .05, "기본 0.5", color=C["gray"], fontsize=9)
ax.set(xlabel="임계값 t", ylabel="점수", title="F1은 t 따라 변하고, AUC는 평평", ylim=(0, 1))
ax.legend(fontsize=9.5, loc="lower left")
save(fig, DAY, "threshold_sweep")

# ---------------------------------------------------------------- ② PR 곡선
pr, rc, _ = precision_recall_curve(y, s)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.step(rc, pr, where="post", color=C["blue"], lw=3)
ax.axhline(6 / 20, color=C["gray"], ls=":", lw=1.8); ax.text(.02, .32, "실패 비율 0.30 (아무렇게나 경고할 때의 Precision)", fontsize=9, color=C["gray"])
for t, col in [(.5, C["ink"]), (.3, C["orange"])]:
    tp_ = (pos > t).sum(); fp_ = (neg > t).sum()
    ax.scatter([tp_ / 6], [tp_ / (tp_ + fp_)], s=90, color=col, zorder=4)
    ax.annotate(f"t={t}: Recall {tp_ / 6:.2f}, Precision {tp_ / (tp_ + fp_):.2f}", (tp_ / 6, tp_ / (tp_ + fp_)),
                (.03 if t == .5 else .4, .52 if t == .5 else .14), fontsize=10, color=col, arrowprops=dict(arrowstyle="->", color=col))
ax.set(xlabel="Recall (실패 중 찾은 비율)", ylabel="Precision (경고 중 진짜)", xlim=(0, 1.03), ylim=(0, 1.05),
       title="같은 20건의 PR 곡선 · 오른쪽으로 갈수록 경고를 많이")
save(fig, DAY, "pr_curve")

# ---------------------------------------------------------------- ③ MAE vs RMSE (A·B)
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw={"width_ratios": [1.25, 1]})
ax = axes[0]
yy = np.array([100, 200, 300, 400]); a = np.array([100, 200, 300, 480]); b = yy + 30
xi = np.arange(4)
ax.bar(xi - .2, np.abs(a - yy), .38, color=C["orange"], label="A 오차: 0, 0, 0, 80")
ax.bar(xi + .2, np.abs(b - yy), .38, color=C["mint"], label="B 오차: 30, 30, 30, 30")
ax.set_xticks(xi, ["100ms", "200ms", "300ms", "400ms"]); ax.set(xlabel="실제값", ylabel="|오차| (ms)", ylim=(0, 115), title="요청별 오차")
ax.legend(fontsize=9, loc="upper left")
ax = axes[1]
m = ax.bar([-.2, .8], [20, 30], .38, color=C["blue"], label="MAE")
r = ax.bar([.2, 1.2], [40, 30], .38, color=C["purple"], label="RMSE")
for bb in list(m) + list(r): ax.text(bb.get_x() + .19, bb.get_height() + 1, f"{bb.get_height():.0f}", ha="center", fontsize=10)
ax.set_xticks([0, 1], ["A", "B"]); ax.set(ylabel="ms", ylim=(0, 52), title="MAE는 A, RMSE는 B가 승")
ax.legend(fontsize=9.5, loc="upper right", ncol=2)
save(fig, DAY, "mae_rmse_ab")

# ---------------------------------------------------------------- ③ fold 평균 vs 합쳐서 계산
fig, ax = plt.subplots(figsize=(6.2, 3.4))
vals = [2, 10, 6, np.sqrt((4 + 100) / 2)]
cols = [C["soft"], C["soft"], C["blue"], C["purple"]]
bars = ax.bar(["fold 1\nRMSE", "fold 2\nRMSE", "두 RMSE의\n평균", "오차를 합쳐\n한 번에 계산"], vals, color=cols, ec=C["purple"])
for bb, v in zip(bars, vals): ax.text(bb.get_x() + bb.get_width() / 2, v + .2, f"{v:.2f}", ha="center", fontsize=11)
ax.text(2.5, 9.6, "같은 오차인데\n계산 순서만 달라도 다른 값", ha="center", fontsize=10)
ax.set(ylabel="RMSE", ylim=(0, 12))
save(fig, DAY, "fold_mean_vs_pooled")

# ---------------------------------------------------------------- ④ log1p 곡선
xx = np.linspace(0, 1100, 500)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(xx, np.log1p(xx), color=C["purple"], lw=3)
for v in [0, 9, 99, 999]:
    ax.scatter([v], [np.log1p(v)], s=60, color=C["ink"], zorder=3)
    ax.text(v + 15, np.log1p(v) - .55, f"{v} → {np.log1p(v):.2f}", fontsize=10)
ax.plot([99, 999], [np.log1p(99)] * 2, color=C["pink"], ls="--", lw=1.8)
ax.plot([999, 999], [np.log1p(99), np.log1p(999)], color=C["pink"], ls="--", lw=1.8)
ax.text(420, 4.05, "가로로 900 차이", color=C["pink"], fontsize=10.5)
ax.text(1012, 4.95, "세로로\n겨우 2.3", color=C["pink"], fontsize=10.5)
ax.set(xlabel="글자수 x", ylabel="log1p(x) = log(1 + x)", ylim=(-.6, 7.8), xlim=(-30, 1180))
save(fig, DAY, "log1p_curve")

# ---------------------------------------------------------------- ④ log1p 전후 히스토그램 (치우친 합성 자료)
rng = np.random.default_rng(8)
raw = np.round(rng.lognormal(6.2, 0.9, 400))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
ax = axes[0]
ax.hist(raw, bins=40, color=C["orange"], ec="white")
ax.set(title="원래 값 · 오른쪽 꼬리가 길다", xlabel="글자수 (설명용 합성 400건)", ylabel="건수")
ax.annotate("몇 개의 아주 긴 요청", (raw.max() * .8, 2), (raw.max() * .45, 60), fontsize=10, arrowprops=dict(arrowstyle="->"))
ax = axes[1]
ax.hist(np.log1p(raw), bins=30, color=C["purple"], ec="white")
ax.set(title="log1p 후 · 가운데로 모이고 꼬리가 짧아짐", xlabel="log1p(글자수)")
save(fig, DAY, "log1p_hist")

# ---------------------------------------------------------------- ⑤ K-Means 반복 과정 (합성)
Xb, _ = make_blobs(150, centers=[[-3, -1], [2, 3], [3, -3]], cluster_std=1.1, random_state=4)
centers = np.array([[-1.5, 4.5], [0.5, 0.5], [-0.5, -4.5]])     # 일부러 나쁜 시작점
pal = [C["purple"], C["pink"], C["mint"]]
snaps = []
for it in range(5):
    lab = np.argmin(((Xb[:, None, :] - centers[None]) ** 2).sum(-1), 1)
    snaps.append((centers.copy(), lab.copy(), ((Xb - centers[lab]) ** 2).sum()))
    centers = np.array([Xb[lab == k].mean(0) for k in range(3)])
show = [0, 1, 2, 4]
fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharex=True, sharey=True)
for ax, i in zip(axes, show):
    cen, lab, ine = snaps[i]
    for k in range(3):
        ax.scatter(*Xb[lab == k].T, s=12, color=pal[k], alpha=.7)
    ax.scatter(*cen.T, s=260, marker="X", color=pal, edgecolor=C["ink"], lw=1.5, zorder=4)
    if i > 0:
        prev = snaps[i - 1][0] if i != 4 else snaps[2][0]
        for p0, p1 in [(a_, b_) for a_, b_ in zip(prev, cen) if np.hypot(*(a_ - b_)) > .15]:
            ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.4))
    ax.set_title(f"{'시작' if i == 0 else f'{i}회 반복 후'} · inertia {ine:.0f}", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].text(-7.5, -6.6, "X = 중심 (일부러 나쁜 곳에서 시작)", fontsize=9)
save(fig, DAY, "kmeans_steps")

# ---------------------------------------------------------------- ⑤ K 고르기: inertia vs silhouette (실제 train)
pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]).fit(tr[NUM])
Z = pre.transform(tr[NUM])
Ks = [2, 3, 4, 5, 6]; ins, sil = [], []
for k in Ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    ins.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
print("K", list(zip(Ks, np.round(ins, 3), np.round(sil, 3))))
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
ax = axes[0]
ax.plot(Ks, ins, "o-", color=C["blue"], lw=2.5)
for k, v in zip(Ks, ins): ax.text(k, v + 2, f"{v:.1f}", ha="center", fontsize=9.5)
ax.set(xlabel="K (군집 수)", ylabel="inertia (작을수록 촘촘)", title="inertia: K를 늘리면 계속 내려감", xticks=Ks, ylim=(0, max(ins) * 1.15))
ax = axes[1]
ax.plot(Ks, sil, "o-", color=C["purple"], lw=2.5)
bi = int(np.argmax(sil))
ax.scatter([Ks[bi]], [sil[bi]], s=160, facecolor="none", edgecolor=C["pink"], lw=2.5, zorder=4)
for k, v in zip(Ks, sil): ax.text(k, v + .008, f"{v:.3f}", ha="center", fontsize=9.5)
ax.set(xlabel="K (군집 수)", ylabel="silhouette (높을수록 잘 분리)", title=f"silhouette: K={Ks[bi]}에서 최고", xticks=Ks,
       ylim=(min(sil) - .05, max(sil) + .05))
save(fig, DAY, "kmeans_k_choice")

# ---------------------------------------------------------------- ⑤ 실제 train 군집 (표준화 공간) + 새 요청
km3 = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Z)
new = pd.DataFrame({"input_chars": [200., 2200.], "allowed_tools": [1., 5.]})
zn = pre.transform(new); pn = km3.predict(zn)
print("sizes", np.bincount(km3.labels_), "new", pn)
fig, ax = plt.subplots(figsize=(6.6, 4))
for k in range(3):
    ax.scatter(*Z[km3.labels_ == k].T, s=34, color=pal[k], alpha=.8, label=f"군집 {k} ({(km3.labels_ == k).sum()}건)")
ax.scatter(*km3.cluster_centers_.T, s=280, marker="X", color=pal, edgecolor=C["ink"], lw=1.5, zorder=4)
ax.scatter(*zn.T, s=240, marker="*", color=C["orange"], edgecolor=C["ink"], lw=.8, zorder=5)
for (zx, zy), lab_, txt in zip(zn, pn, ["새 요청 200자·도구1", "새 요청 2200자·도구5"]):
    ax.annotate(f"{txt}\n→ 군집 {lab_}", (zx, zy), (zx + (.25 if zx < 0 else -1.0), zy + (.35 if zy < 0 else -.35)), fontsize=9.5,
                arrowprops=dict(arrowstyle="->"))
ax.set(xlabel="글자수 (train 기준 표준화)", ylabel="도구수 (train 기준 표준화)", title="train 60건 · K=3 · X = 중심")
ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3)
save(fig, DAY, "kmeans_real")

# ---------------------------------------------------------------- ⑥ 후보 CV F1 vs valid F1 (강의 표 값)
cands = ["Dummy", "Logistic", "KNN3", "Tree", "Forest", "XGBoost", "Logistic\n+파생2"]
cvm = [0, .522, .336, .542, .280, .500, .551]; cvs = [0, .209, .176, .124, .254, .123, .190]
vf = [0, .333, .545, .333, .222, .182, .182]
fig, ax = plt.subplots(figsize=(8.6, 3.8))
xi = np.arange(len(cands))
bars = ax.bar(xi, cvm, .55, yerr=cvs, capsize=4, color=[C["soft"]] * 6 + [C["purple"]], ec=C["purple"],
              error_kw=dict(ecolor=C["gray"]), label="CV F1 평균 ± fold 표준편차 (선택 기준)")
ax.scatter(xi, vf, s=70, marker="D", color=C["orange"], zorder=4, label="valid F1 (진단용)")
ax.text(6, cvm[6] + cvs[6] + .04, "선택", ha="center", fontsize=10, color=C["purple"], fontweight="bold")
ax.set_xticks(xi, cands); ax.set(ylabel="F1", ylim=(0, 1), title="사전 선언한 7개 후보 · CV 평균 최대 규칙으로 선택")
ax.legend(fontsize=9.5, loc="upper left")
save(fig, DAY, "candidates_cv")

# ---------------------------------------------------------------- ⑥ 최종 test 혼동행렬
fig, ax = plt.subplots(figsize=(5, 4.1))
cm_axes(ax, [[12, 2], [3, 3]], "최종 test 20건 · Logistic_FE · t = 0.5")
ax.text(1, -.4, "Precision 0.600 · Recall 0.500 · F1 0.545 · AUC 0.786", ha="center", fontsize=10)
save(fig, DAY, "final_test_confusion")
print("done")
