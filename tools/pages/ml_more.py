"""모델 > 머신러닝 기초 4쪽: SVM, 판별분석, PCA. examples/ml/more_models.py 하나가 ①~⑦로 자란다.
1~3쪽(requests_model.py, tools/pages/ml_basics.py)과 같은 데이터, 같은 나누기, 같은 교차검증을 쓴다.
실행: python tools/pages/ml_more.py
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
import sklearn  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis  # noqa: E402
from sklearn.metrics import f1_score as F1, accuracy_score as ACC  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "ml"
code = S.parse((EX / "more_models.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="more_models.py")  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, pandas {pd.__version__}"
ns = run(7)
# 1~3쪽 값을 인용할 때도 손으로 옮겨 적지 않고, 그 쪽 예제(requests_model.py)를 ⑫까지 실제로 돌려서 가져와요
REF = S.run(S.parse((EX / "requests_model.py").read_text(encoding="utf-8")), 12, S.workdir(EX), filename="requests_model.py")
p1_f1 = REF["scores"]["f1"]           # 1쪽 ⑤ 로지스틱의 valid F1
xgb_cv = REF["cv_scores"]["xgboost"]  # 2쪽 ⑫ xgboost의 교차검증 F1 다섯 개

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "ml"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]
BG = ListedColormap(["#ece6ff", "#ffe0ee"])


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/ml/{name}.png"


def legend_below(fig, ax):
    # 왼쪽 아래 점 줄을 가리지 않게 범례를 그림 아래 바깥에 둬요
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(lab), fontsize=9, frameon=False)


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


# 그림용 2열 데이터: 숫자 두 열만 (빈 칸은 train 중앙값, train 평균과 표준편차로 크기 맞춤)
tr = ns["train"]
d2 = tr[["input_chars", "allowed_tools"]].fillna(tr[["input_chars", "allowed_tools"]].median())
X2 = ((d2 - d2.mean()) / d2.std(ddof=0)).to_numpy()
y2 = tr["failed"].to_numpy()
g1, g2 = np.meshgrid(np.linspace(-2.2, 2.4, 300), np.linspace(-1.8, 1.9, 300))
G = np.c_[g1.ravel(), g2.ravel()]


def scatter2(ax, sv=None):
    ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], s=24, color=OK, label="성공", zorder=3)
    ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], s=44, color=FAIL, marker="X", label="실패", zorder=3)
    if sv is not None:
        ax.scatter(X2[sv, 0], X2[sv, 1], s=130, facecolors="none", edgecolors=C["ink"], lw=1.2, label="서포트 벡터", zorder=4)
    ax.set(xlabel="input_chars (크기 맞춤)")
    ax.grid(False)


outs = {}
# ② 선형 SVM 그림
lin2 = SVC(kernel="linear", C=1.0).fit(X2, y2)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
F = lin2.decision_function(G).reshape(g1.shape)
ax.contourf(g1, g2, (F >= 0).astype(int), levels=[-.5, .5, 1.5], cmap=BG)
ax.contour(g1, g2, F, levels=[-1, 0, 1], colors=[C["gray"], C["ink"], C["gray"]], linestyles=["--", "-", "--"], linewidths=[1.2, 2, 1.2])
scatter2(ax, lin2.support_)
ax.set(ylabel="allowed_tools (크기 맞춤)", title=f"선형 SVM (C=1): 서포트 벡터 {len(lin2.support_)}개 / 60건")
legend_below(fig, ax)
P_LIN = fig_save(fig, "svm_linear_margin")
# ③ RBF gamma 그림
fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
for ax, g in zip(axs, (0.1, 1, 10)):
    m = SVC(kernel="rbf", C=10, gamma=g).fit(X2, y2)
    ax.contourf(g1, g2, m.predict(G).reshape(g1.shape), levels=[-.5, .5, 1.5], cmap=BG)
    scatter2(ax)
    ax.set(title=f"gamma = {g}  (train 정확도 {ACC(y2, m.predict(X2)):.2f})")
axs[0].set_ylabel("allowed_tools (크기 맞춤)")
legend_below(fig, axs[0])
P_RBF = fig_save(fig, "svm_rbf_gamma")
# ④ LDA 한 줄 사영
z, yt = ns["lda_z"], ns["y_train"].to_numpy()
fig, ax = plt.subplots(figsize=(8, 3.4))
bins = np.linspace(z.min() - .2, z.max() + .2, 18)
ax.hist(z[yt == 0], bins=bins, color=OK, alpha=.6, label=f"실제 성공 (평균 {z[yt == 0].mean():.2f})")
ax.hist(z[yt == 1], bins=bins, color=FAIL, alpha=.6, label=f"실제 실패 (평균 {z[yt == 1].mean():.2f})")
lda_m = ns["lda"].named_steps["model"]
ax.set(xlabel="LDA 판별 점수 (6열을 한 줄 위로 내린 값)", ylabel="train 건수", title="LDA: 두 무리의 평균은 멀게, 각 무리의 퍼짐은 좁게")
ax.legend(fontsize=9)
P_LDA = fig_save(fig, "lda_projection")
# ⑤ LDA vs QDA 경계 (숫자 두 열)
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (nm, m) in zip(axs, [("LDA: 공통 모양 → 직선 경계", LinearDiscriminantAnalysis()), ("QDA: 종류마다 다른 모양 → 곡선 경계", QuadraticDiscriminantAnalysis())]):
    m.fit(X2, y2)
    pr = m.predict_proba(G)[:, 1].reshape(g1.shape)
    ax.contourf(g1, g2, (pr >= 0.5).astype(int), levels=[-.5, .5, 1.5], cmap=BG)
    ax.contour(g1, g2, pr, levels=[0.5], colors=[C["ink"]], linewidths=2)
    scatter2(ax)
    ax.set(title=nm)
axs[0].set_ylabel("allowed_tools (크기 맞춤)")
legend_below(fig, axs[0])
P_QDA = fig_save(fig, "lda_vs_qda")
# ⑥ PCA
ratio = ns["ratio"]
cum = np.cumsum(ratio)
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 3.9))
idx = np.arange(1, len(ratio) + 1)
a.bar(idx, ratio, color=[FAIL if i <= ns["k70"] else C["soft"] for i in idx], edgecolor=OK)
a.plot(idx, cum, color=C["ink"], marker="o", lw=2, label="누적")
a.axhline(0.7, color=C["gray"], ls="--", lw=1.2)
a.text(5.2, 0.73, "70%", color=C["gray"], fontsize=10)
a.set(xlabel="주성분 번호", ylabel="담은 분산 비율", xticks=idx, ylim=(0, 1.08), title=f"주성분 {ns['k70']}개면 {cum[ns['k70'] - 1]:.0%}")
a.legend(fontsize=9, loc="center right")
P2 = ns["pca"].transform(ns["Z_train"])
b.scatter(P2[yt == 0, 0], P2[yt == 0, 1], s=26, color=OK, label="성공")
b.scatter(P2[yt == 1, 0], P2[yt == 1, 1], s=46, color=FAIL, marker="X", label="실패")
b.set(xlabel=f"주성분 1 ({ratio[0]:.0%})", ylabel=f"주성분 2 ({ratio[1]:.0%})", title="train 60건을 주성분 두 개 위에")
b.legend(fontsize=9)
P_PCA = fig_save(fig, "pca_variance_scatter")
load = pd.DataFrame(ns["pca"].components_[:2].T, index=[n.split("__", 1)[1] for n in ns["build"](ns["LogisticRegression"]()).named_steps["pre"].fit(ns["X_train"]).get_feature_names_out()], columns=["주성분 1", "주성분 2"])

# ---------------- 출력 상자
outs["s1"] = S.box("① 실행 결과", [f"train {len(ns['train'])}건, valid {len(ns['valid'])}건 (1쪽 ①과 같은 행)", (f"build(model) → 전처리 뒤 8열, build(model, drop=\"first\") → 6열", 1, "newf")],
                   ["<code>drop=\"first\"</code>는 <code>task_type</code>, <code>priority</code>의 원-핫 열을 종류마다 하나씩 빼요. 1쪽 ④에서 본 \"원-핫 세 열을 더하면 늘 1\"인 문제를 없애는 거예요. 열 사이 공분산의 역행렬을 쓰는 ④⑤의 판별분석에 필요해요."])
outs["s2"] = S.box("② 선형 SVM (8열, C=1)", [(f"도로 폭 2 ÷ ‖θ‖ = {ns['margin']:.3f}", 1), (f"서포트 벡터: 성공 {ns['n_sv'][0]}개, 실패 {ns['n_sv'][1]}개 (train 60건 중 {ns['n_sv'].sum()}개)", 2, "newf"),
                                                (f"valid → F1 {ns['lin_scores']['f1']:.2f}, recall {ns['lin_scores']['recall']:.2f}, accuracy {ns['lin_scores']['accuracy']:.2f}", 3)],
                   ["크기를 맞춘 8열 공간에서 도로 폭이에요. 숫자 자체보다 C를 바꿀 때 어떻게 변하는지가 중요해요 (🔁).",
                    f"{len(ns['train'])}건 중 {ns['n_sv'].sum()}개, 절반쯤이 서포트 벡터예요. 두 무리가 많이 겹쳐서, 도로 안이나 반대편에 들어간 점이 많다는 뜻이에요. 깨끗이 나뉘는 데이터라면 서포트 벡터는 몇 개뿐이에요.",
                    f"1쪽 로지스틱(F1 {p1_f1:.2f})보다 낮아요. 둘 다 곧은 경계라 비슷한 한계가 있어요."])
S.claim(0.4 <= ns["n_sv"].sum() / len(ns["train"]) <= 0.6, "② 서포트 벡터가 train의 절반쯤")
S.claim(ns["lin_scores"]["f1"] < p1_f1, "② 선형 SVM F1이 1쪽 로지스틱보다 낮다")
lin_rows, lin_n = [], {}
for Cv in (0.1, 1, 10):
    n = lin_n[Cv] = ns if Cv == 1 else run(2, [("SVC(kernel=\"linear\", C=1.0)", f"SVC(kernel=\"linear\", C={Cv})")])
    lin_rows.append([f"<b>{Cv}</b> (지금)" if Cv == 1 else Cv, f3(n["margin"]), int(n["n_sv"].sum()), f"{n['lin_scores']['f1']:.2f}"])
outs["t2"] = S.table("선형 SVM의 C를 바꾸면", ["C", "도로 폭", "서포트 벡터 수", "valid F1"], lin_rows,
                     [f"C가 작으면 도로 안으로 들어오는 점을 너그럽게 봐줘서, 도로가 넓어지고({lin_n[0.1]['margin']:.1f}) 서포트 벡터가 많아져요. 너무 너그러우면 경계를 거의 안 움직여서 모두 \"성공\"이라고 답해요 (F1 {lin_n[0.1]['lin_scores']['f1']:.0f}).",
                      "C를 키우면 틀린 점 하나하나를 비싸게 쳐서 도로가 좁아져요. 이 데이터에서는 C 1 이상이면 거의 같은 답에 머물러요. 1쪽 로지스틱의 C와 방향이 같아요: 작을수록 규제가 세요."])
S.claim(lin_n[0.1]["margin"] > lin_n[1]["margin"] and lin_n[0.1]["n_sv"].sum() > lin_n[1]["n_sv"].sum(), "C=0.1이면 도로가 넓고 서포트 벡터가 많다")
S.claim(lin_n[0.1]["svm_lin"].predict(ns["X_valid"]).sum() == 0, "C=0.1이면 valid를 모두 성공이라고 답한다")
S.claim(abs(lin_n[10]["lin_scores"]["f1"] - lin_n[1]["lin_scores"]["f1"]) < 0.05, "C 1 이상이면 거의 같은 답")
outs["s3"] = S.box("③ RBF 커널 SVM (8열, C=1, gamma=\"scale\")", [(f"기본 → F1 {ns['rbf_scores']['f1']:.2f}, recall {ns['rbf_scores']['recall']:.2f}, accuracy {ns['rbf_scores']['accuracy']:.2f}", 1),
                                                                (f"class_weight=\"balanced\" → F1 {ns['bal_scores']['f1']:.2f}, recall {ns['bal_scores']['recall']:.2f}, accuracy {ns['bal_scores']['accuracy']:.2f}", 2, "newf")],
                   [f"곡선 경계를 허용하니 선형({ns['lin_scores']['f1']:.2f})보다 F1이 올랐어요.",
                    f"실패가 30%뿐이라, 그냥 두면 SVM은 성공 쪽 오류를 줄이는 데 더 신경 써요. <code>balanced</code>는 실패 한 건을 틀리는 값을 성공 한 건보다 (70 ÷ 30)배 비싸게 쳐요. 그래서 실패를 더 잡고(recall {ns['bal_scores']['recall']:.2f}) 헛경보도 늘어요(정확도 {ns['bal_scores']['accuracy']:.2f}). 3쪽 ⑬에서 기준을 낮춘 것과 같은 방향이에요."])
S.claim(ns["rbf_scores"]["f1"] > ns["lin_scores"]["f1"], "③ RBF가 선형보다 F1이 높다")
S.claim(ns["bal_scores"]["recall"] > ns["rbf_scores"]["recall"] and ns["bal_scores"]["accuracy"] < ns["rbf_scores"]["accuracy"], "③ balanced는 recall이 오르고 정확도가 내린다")
g_rows, g_n = [], {}
for g in (0.01, 0.1, 1, 10):
    n = g_n[g] = run(3, [("svm_rbf = build(SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\"))", f"svm_rbf = build(SVC(kernel=\"rbf\", C=10, gamma={g}))")])
    m = n["svm_rbf"]
    g_rows.append([g, f3(ACC(n["y_train"], m.predict(n["X_train"]))), int(m.named_steps["model"].n_support_.sum()), f"{n['rbf_scores']['f1']:.2f}"])
g01_fail = int(g_n[0.01]["svm_rbf"].predict(ns["X_valid"]).sum())   # gamma 0.01이 valid에서 "실패"라고 답한 수
S.claim(g_n[0.01]["rbf_scores"]["recall"] == 0 and g01_fail > 0, "gamma 0.01은 실패를 하나도 못 맞히고, 실패라고 한 건 모두 헛경보")
outs["t3"] = S.table("RBF 커널의 gamma를 바꾸면 (C=10)", ["gamma", "train 정확도", "서포트 벡터 수", "valid F1"], g_rows,
                     [f"gamma가 작으면 점 하나의 영향이 멀리까지 퍼져서 경계가 아주 완만해요. 0.01은 거의 곧은 선이라 실패를 하나도 맞히지 못해요. \"실패\"라고 답한 {g01_fail}건은 모두 헛경보예요.",
                      f"gamma가 크면 점 하나의 영향이 자기 바로 옆에만 미쳐서, 점마다 작은 섬을 만들어요. 10은 train을 {float(g_rows[-1][1]):.2f} 맞히지만 valid에서는 실패를 하나도 못 잡아요. 외운 거예요(과적합). 서포트 벡터도 {g_rows[-1][2]}개로, 거의 모든 점이 경계를 정하는 데 쓰여요.",
                      "C와 gamma는 같이 움직이니, 실제로는 둘을 격자로 놓고 교차검증으로 골라요 (<code>GridSearchCV</code>)."])
S.claim(g_n[10]["rbf_scores"]["recall"] == 0 and float(g_rows[-1][1]) > 0.95, "gamma 10은 train을 거의 다 맞히고 valid 실패는 못 잡는다")
S.claim(g_rows[-1][2] >= 0.9 * len(ns["train"]), "gamma 10은 거의 모든 점이 서포트 벡터")
outs["s4"] = S.box("④ LDA (6열)", [(f"사전 확률 π → 성공 {ns['prior'][0]:.2f}, 실패 {ns['prior'][1]:.2f}", 1), (f"판별 점수 평균 → 성공 {z[yt == 0].mean():.2f}, 실패 {z[yt == 1].mean():.2f}", 2, "newf"),
                                    (f"valid → F1 {ns['lda_scores']['f1']:.2f}, recall {ns['lda_scores']['recall']:.2f}, accuracy {ns['lda_scores']['accuracy']:.2f}", 3)],
                   ["따로 정해 주지 않으면 train의 종류 비율을 그대로 써요. 아무 정보 없을 때 \"실패일 확률 30%\"에서 출발한다는 뜻이에요.",
                    "6열을 한 줄 위로 내렸을 때 두 무리의 평균이에요. 평균이 떨어져 있긴 하지만, 위 그림처럼 두 무리가 많이 겹쳐요.",
                    f"로지스틱({p1_f1:.2f})과 비슷해요. 둘 다 곧은 경계를 긋는데, LDA는 정규분포를 가정하고 로지스틱은 가정하지 않는다는 점만 달라요."])
outs["s5"] = S.box("⑤ QDA (6열, reg_param=0.1)", [(f"valid → F1 {ns['qda_scores']['f1']:.2f}, recall {ns['qda_scores']['recall']:.2f}, accuracy {ns['qda_scores']['accuracy']:.2f}", 1)],
                   ["LDA와 거의 같아요. 종류마다 퍼짐 모양을 따로 배워도 나아지지 않았어요. 실패가 18건뿐이라, 6열의 퍼짐 모양(공분산 21개 값)을 믿을 만하게 배우기에는 데이터가 부족해요."])
S.claim(abs(ns["lda_scores"]["f1"] - p1_f1) < 0.05, "④ LDA F1이 로지스틱과 비슷하다")
S.claim(abs(ns["qda_scores"]["f1"] - ns["lda_scores"]["f1"]) < 0.05, "⑤ QDA가 LDA와 거의 같다")
zero_pc = [i + 1 for i, r in enumerate(ratio) if r < 1e-9]   # 분산을 하나도 못 담은 주성분 번호
S.claim(zero_pc == [len(ratio) - 1, len(ratio)], "⑥ 마지막 두 주성분의 분산이 0 (원-핫 두 묶음)")
cum_txt = ", ".join(f"{c:.0%}" for c in cum)
outs["s6"] = S.box("⑥ PCA (크기 맞춘 8열)", [(f"주성분별 분산 비율 → {', '.join(f'{r:.3f}' for r in ratio)}", 1), (f"누적 → {cum_txt}", 2), (f"70%를 넘기는 가장 작은 개수 k70 = {ns['k70']}", 3, "newf"),
                                         (f"PCA {ns['k70']}개 → 로지스틱 → valid F1 {ns['pca_scores']['f1']:.2f}, recall {ns['pca_scores']['recall']:.2f}", 4)],
                   [f"마지막 두 주성분이 담은 분산이 0이에요. {len(ratio)}열 중 서로 다른 정보는 {len(ratio) - len(zero_pc)}개뿐이라서예요. 원-핫 세 열의 합이 늘 1이라 남는 방향이 두 개(task_type, priority) 생겨요. 1쪽 ④의 \"독립인 열 7개\"(절편 열을 포함한 9열 기준)와 같은 이야기예요.",
                    "주성분 하나하나가 담는 양이 고르게 줄어서, 몇 개로 확 압축되는 데이터는 아니에요.",
                    f"{len(ratio)}열을 {ns['k70']}열로 줄였어요.", f"valid 한 번은 {len(ratio)}열 로지스틱({p1_f1:.2f})보다 낮아요. 그런데 ⑦의 교차검증에서는 오히려 더 안정적이에요. 한 번의 점수로 판단하지 않는 이유예요."])
S.claim(ns["pca_scores"]["f1"] < p1_f1, "⑥ PCA+로지스틱 valid F1이 8열 로지스틱보다 낮다")
outs["s6l"] = S.table("주성분이 원래 열을 섞은 비율 (가중치, 처음 두 개)", ["원래 열", "주성분 1", "주성분 2"], [[f"<code>{i}</code>", f"{r['주성분 1']:+.2f}", f"{r['주성분 2']:+.2f}"] for i, r in load.iterrows()],
                      ["주성분은 원래 열에 가중치를 곱해 더한 새 열이에요. 가중치 절댓값이 큰 열이 그 주성분의 \"뜻\"이에요. 주성분 이름은 이 표를 보고 사람이 붙여요."])
cvs = ns["cv_scores"]
best_name = max(cvs, key=lambda k: cvs[k].mean())
pca_name = f"pca{ns['k70']}+logistic"
S.claim(best_name == "svm rbf 균형" and cvs[best_name].mean() > xgb_cv.mean() and cvs[best_name].std() > xgb_cv.std(), "⑦ 균형 SVM이 1등, xgboost보다 평균은 높고 표준편차는 크다")
S.claim(np.allclose(cvs["svm 선형"], cvs["lda"]) and int((~np.isclose(cvs["logistic"], cvs["lda"])).sum()) <= 1, "⑦ 선형 SVM과 LDA는 다섯 점수가 같고, 로지스틱은 거의 같다")
S.claim(cvs[pca_name].mean() > cvs["logistic"].mean() and cvs[pca_name].std() == min(v.std() for v in cvs.values()), "⑦ PCA+로지스틱이 로지스틱보다 평균이 높고 표준편차가 가장 작다")
outs["s7"] = S.table("⑦ 5-fold 교차검증 F1 (train+valid 80건, 2쪽 ⑫와 같은 나누기)", ["model", "다섯 번의 점수", "평균", "표준편차"],
                     [[f"<b>{m}</b>" if m == best_name else m, ", ".join(f3(x) for x in v), f"<b>{f3(v.mean())}</b>" if m == best_name else f3(v.mean()), f3(v.std())] for m, v in cvs.items()],
                     [f"<b>{best_name}</b>이 평균 {cvs[best_name].mean():.3f}로 가장 높아요. 2쪽 ⑫의 xgboost({xgb_cv.mean():.3f})보다도 높아요. 다만 표준편차는 {cvs[best_name].std():.3f}로 xgboost({xgb_cv.std():.3f})보다 커서, 덩어리마다 점수가 더 오르내려요. 모델 종류보다 \"드문 실패를 얼마나 중요하게 치느냐\"(class_weight)가 더 크게 작용했어요.",
                      "선형 SVM과 LDA는 다섯 점수가 완전히 같아요. 둘 다 곧은 경계를 긋고, 이 데이터에서는 같은 쪽으로 판단했어요. 로지스틱도 거의 같아요.",
                      f"PCA {ns['k70']}개 + 로지스틱은 평균이 8열 로지스틱보다 조금 높고, 표준편차가 가장 작아요. 열을 줄이니 덜 흔들렸어요.",
                      "⚠️ 3쪽 ⑰에서 test는 이미 한 번 썼어요. 이 결과를 보고 test로 다시 채점하면 test로 모델을 고르는 셈이라 하지 않아요. 실제 작업이라면 이 후보들을 ⑰ 전에 함께 비교했어야 해요."])
outs["f_lin"] = chart("숫자 두 열만 쓴 선형 SVM의 도로", P_LIN, "검은 선: 경계, 회색 점선: 도로 양쪽 끝",
                      ["그림으로 보려고 숫자 두 열만 쓴 모델이에요. ②의 실제 모델(8열)과 다르지만 생김새는 같아요.",
                       "동그라미 친 점이 서포트 벡터예요. 도로 끝선 위에 있거나, 도로 안 또는 반대편으로 넘어온 점이에요. <b>이 점들만</b> 경계를 정해요. 도로 바깥 멀리 있는 점은 지워도 경계가 그대로예요.",
                       "두 무리가 많이 섞여 있어서 도로 안에 점이 아주 많아요. 이런 데이터에서 SVM은 \"가장 넓은 도로\"보다 \"넘어온 점이 적은 도로\"를 찾는 쪽으로 기울어요 (C)."])
outs["f_rbf"] = chart("RBF 커널 SVM의 판단 지도 (숫자 두 열, C=10)", P_RBF, "분홍 바탕: \"실패\"라고 답하는 곳",
                      ["gamma 0.1은 매끈한 곡선 경계, 1은 실패가 모인 곳을 주머니처럼 감싸고, 10은 실패 점마다 작은 섬을 만들어요.",
                       "1쪽 KNN의 k와 방향이 반대예요. KNN은 k가 작을수록, RBF는 gamma가 클수록 외워요."])
outs["f_lda"] = chart("train 60건의 LDA 판별 점수", P_LDA, "", ["LDA가 찾은 방향으로 6열을 한 줄 위에 내렸어요. 이 방향은 \"두 평균 차이 ÷ 각 무리의 퍼짐\"이 가장 커지는 방향이에요.",
                                                               "실패(분홍)가 오른쪽에 몰려 있긴 하지만 성공과 넓게 겹쳐요. 겹친 부분은 어떤 경계로도 가를 수 없어서, 이 데이터에서는 어떤 선형 모델도 비슷한 한계에 부딪혀요."])
outs["f_qda"] = chart("숫자 두 열로 본 LDA와 QDA의 경계", P_QDA, "", ["LDA는 두 무리가 같은 모양으로 퍼져 있다고 가정해서 경계가 곧은 직선이에요.",
                                                                    "QDA는 실패 무리와 성공 무리의 퍼짐 모양을 따로 배워서, 경계가 휘어요. 대신 배울 값이 많아서 데이터가 적으면 엉뚱하게 휘기 쉬워요."])
outs["f_pca"] = chart("PCA: 주성분마다 담은 분산과, 처음 두 주성분 위의 train", P_PCA, "",
                      [f"왼쪽 막대가 주성분마다 담은 분산 비율, 선이 누적이에요. {ns['k70']}개에서 70%를 넘어요. {', '.join(map(str, zero_pc))}번째는 0이에요.",
                       f"오른쪽은 8열을 주성분 두 개로 줄여 평면에 그린 거예요. 두 개가 원래 분산의 {cum[1]:.0%}를 담아요. 실패와 성공이 섞여 있는데, PCA는 정답을 보지 않고 \"가장 많이 퍼진 방향\"만 찾기 때문이에요. 정답을 보고 \"가장 잘 갈리는 방향\"을 찾는 건 ④의 LDA예요."])
flow = [[1, "준비", "1쪽과 같은 나누기", "drop=first"], [2, "선형 SVM", "가장 넓은 도로", "서포트 벡터, C"], [3, "커널 SVM", "곡선 경계", "gamma, 균형"], [4, "LDA", "분포 가정, 베이즈", "한 줄로 사영"],
        [5, "QDA", "종류마다 다른 모양", "곡선 경계"], [6, "PCA", "정답 없이 열 줄이기", "분산 비율"], [7, "비교", "같은 교차검증", "균형 SVM이 1등"]]
S.save(ROOT / "topics" / "model" / "ml-svm-lda-pca-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>1~3쪽에서 선형 모델, KNN, 트리와 앙상블을 봤어요. 표 데이터 분류에는 오래 쓰여 온 모델이 셋 더 있어요. 서로 생각하는 방식이 달라요.</p>
{K.table(["모델", "한 문장", "1~3쪽과 닮은 점"], [["<b>SVM</b> (②③)", "두 무리 사이에 <b>가장 넓은 도로</b>를 내요", "로지스틱처럼 곧은 경계에서 시작해요. 커널을 쓰면 곡선 경계도 돼요"],
                                                 ["<b>LDA, QDA</b> (④⑤)", "무리마다 <b>어떻게 퍼져 있는지</b>를 배우고, 새 점이 어느 무리에서 나왔을 확률이 높은지 따져요", "확률을 내는 건 로지스틱과 같지만, 계산하는 길이 반대예요"],
                                                 ["<b>PCA</b> (⑥)", "정답 없이, 열을 <b>정보가 많은 몇 개로</b> 줄여요", "3쪽 K-Means처럼 비지도 학습이에요. 1쪽에서 말한 차원의 저주를 덜어요"]])}
<p>예제는 1~3쪽과 같은 업무 요청 100건, 같은 train/valid 나누기, 같은 교차검증이에요. 그래서 마지막 ⑦에서 2쪽 ⑫의 점수와 바로 비교할 수 있어요. 다만 3쪽 ⑰에서 test를 이미 한 번 썼기 때문에, 이 쪽에서는 test를 쓰지 않아요.</p>
"""
whole = """<p>같은 준비 위에 SVM 둘, 판별분석 둘, PCA 하나를 차례로 올리고, 마지막에 같은 교차검증으로 비교해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>이 쪽의 코드는 새 파일 <code>more_models.py</code>예요. 1~3쪽의 <code>requests_model.py</code>는 3쪽 ⑰의 최종 평가로 끝났거든요.</p>"""
steps = [
    K.step(1, "데이터와 전처리 준비", "1쪽 ①③과 같은 나누기와 전처리를 다시 만들어요. 원-핫에서 한 열씩 빼는 선택만 더해요.",
           "<p>모델을 비교하려면 데이터와 전처리가 같아야 해요. 그래서 1쪽과 같은 manifest로 나누고, 같은 <code>build</code>를 써요. 판별분석(④⑤)은 열 사이 공분산 행렬의 역행렬이 필요해서, 1쪽 ④에서 본 \"원-핫 열끼리 같은 정보\" 문제를 피하려고 종류마다 원-핫 한 열을 빼요.</p>"
           + K.code_table([("OneHotEncoder(..., drop=drop)", "<code>drop=\"first\"</code>면 종류마다 첫 값의 열을 빼요. 빠진 값은 \"나머지 열이 모두 0\"으로 표현돼요."),
                           ("score(m)", "valid에서 F1, recall, accuracy를 한 번에. 1쪽 ⑦의 <code>evaluate</code>와 같은 일")]), "s1"),
    K.step(2, "선형 SVM: 가장 넓은 도로 찾기", "두 무리를 가르는 곧은 경계 중, 양쪽 가장 가까운 점까지 거리가 가장 먼 것을 골라요.",
           "<p>두 무리를 가르는 직선은 하나가 아니에요. 로지스틱은 확률이 잘 맞는 선을 고르고, <b>SVM</b>(Support Vector Machine)은 선 양쪽에 <b>도로</b>를 깔았을 때 도로가 가장 넓어지는 선을 골라요. 도로가 넓을수록 새 점이 조금 비껴 와도 같은 쪽에 떨어져서, 처음 보는 데이터에 강하다는 생각이에요.</p>"
           "<p>식으로는 이래요. 실패를 +1, 성공을 −1로 두고(y), 경계는 <code>θᵀx + θ₀ = 0</code>, 도로의 양쪽 끝은 <code>θᵀx + θ₀ = +1</code>과 <code>−1</code>이에요. θᵀx는 열마다 θ와 x를 곱해 더한 값(내적)으로, 1쪽의 \"가중치를 곱해 더하기\"와 같아요. 두 점의 내적 xᵢᵀxⱼ도 같은 계산이고, 두 점이 같은 방향일수록 커요. 두 끝선 사이 거리는 <code>2 ÷ ‖θ‖</code>(‖θ‖는 가중치 벡터의 길이)라서, 도로를 넓히는 건 <b>‖θ‖를 작게</b> 하는 것과 같아요.</p>"
           "<p>그래서 SVM이 푸는 문제는 <code>‖θ‖² ÷ 2를 가장 작게, 단 모든 점이 y × (θᵀx + θ₀) ≥ 1</code> (모든 점이 자기 쪽 도로 끝선 바깥에). 이게 기본 SVM이에요.</p>"
           "<p><b>서포트 벡터</b>: 이 문제를 풀면, 가중치 θ가 <b>도로 끝선에 닿은 점들만의</b> 가중 합으로 나와요. 도로 바깥 멀리 있는 점은 무게가 0이라, 지우거나 옮겨도 경계가 안 변해요. 경계를 \"받치는\"(support) 점이라서 서포트 벡터예요.</p>"
           "<p><b>도로 안으로 넘어오는 점을 허락하기</b>: 실제 데이터는 깨끗이 안 갈려요(아래 그림처럼). 그래서 점마다 \"얼마나 넘어왔나\"(slack ξ)를 허락하고, 그 합에 <b>C</b>를 곱해 비용에 더해요. <code>‖θ‖² ÷ 2 + C × Σξ</code>. 넘어온 정도를 재는 <code>max(0, 1 − y × (θᵀx + θ₀))</code>를 <b>힌지 손실</b>이라고 해요. 도로 바깥에서는 0이고, 넘어올수록 곧게 커져요. C가 크면 넘어오는 점을 비싸게 쳐서 도로가 좁아지고(과적합 쪽), 작으면 넓어져요(과소적합 쪽).</p>"
           "<p><b>어떻게 푸나</b>: \"단 ~ 이어야\"라는 조건이 붙은 최소화는 <b>라그랑주 승수</b>로 풀어요. 점마다 승수 λᵢ ≥ 0을 붙여 조건을 비용 안에 넣고, θ 대신 λ를 찾는 <b>쌍대 문제</b>로 바꿔요. 풀고 나면 λᵢ > 0인 점이 정확히 서포트 벡터예요(KKT 조건 중 \"상보성\": 조건이 딱 맞아떨어지는 점만 승수가 0이 아니에요). 그리고 쌍대 문제에서는 데이터가 <b>두 점의 내적 xᵢᵀxⱼ</b>으로만 나와요. 이게 ③의 커널로 이어져요.</p>"
           + K.out("f_lin")
           + K.code_table([("SVC(kernel=\"linear\", C=1.0)", "곧은 경계 SVM. C는 넘어온 점의 값"), ("coef_[0]", "가중치 θ (선형 커널일 때만 있어요)"), ("np.linalg.norm(w)", "벡터의 길이 ‖θ‖"),
                           ("n_support_", "종류별 서포트 벡터 수")])
           + K.try_("C를 0.1, 1, 10으로 바꾸면 도로 폭과 서포트 벡터 수는?", "t2"), "s2"),
    K.step(3, "커널 SVM: 곧은 도로로 안 될 때", "점들을 더 높은 차원으로 옮겼다고 치고 거기서 곧은 도로를 내요. 원래 공간에서는 곡선 경계가 돼요.",
           "<p>평면에서 곧은 선으로 못 가르는 점들도, 새 열을 만들어 차원을 높이면 갈리기도 해요. 예를 들어 (x₁, x₂)에 x₁², x₂², x₁x₂를 더하면 원 모양 경계가 그 공간에서는 평면이 돼요. 문제는 열을 늘릴수록 계산이 폭발한다는 거예요.</p>"
           "<p><b>커널 트릭</b>: ②에서 본 것처럼 SVM은 데이터를 두 점의 내적으로만 써요. 그러니 새 공간의 좌표 Φ(x)를 실제로 만들지 않고, <b>새 공간에서의 내적만 바로 계산하는 함수</b> K(x, z) = Φ(x)ᵀΦ(z)가 있으면 돼요. 예: <code>K(x, z) = (xᵀz)²</code>은 2열 데이터를 (x₁², x₂², √2 x₁x₂) 3열로 옮긴 뒤 내적한 것과 같아요. 3열을 만들지 않고 원래 2열의 내적을 제곱만 했어요.</p>"
           + K.table(["커널", "K(x, z)", "경계"], [["linear", "xᵀz", "곧은 선 (②)"], ["poly", "(a × xᵀz + b)ᵈ", "d차 곡선. 보통 4차 미만"], ["sigmoid", "tanh(a × xᵀz + b)", "신경망과 비슷. tanh는 −1과 1 사이의 S자 곡선"],
                                                 ["<b>rbf</b> (가장 많이 써요)", "exp(−γ × ‖x − z‖²)", "두 점이 가까우면 1, 멀면 0. 아무 모양이나"]])
           + "<p>RBF의 <b>gamma(γ)</b>는 점 하나의 영향이 미치는 범위의 역수예요. 크면 좁게(점마다 섬, 과적합), 작으면 넓게(완만한 경계, 과소적합). <code>gamma=\"scale\"</code>은 1 ÷ (열 수 × 데이터 분산)으로 자동으로 정해요. 어떤 커널이 맞는지는 정해진 답이 없어서 교차검증으로 골라요.</p>"
           "<p>⚠️ 실패가 30%뿐이라, 그냥 두면 SVM은 많은 쪽(성공)을 덜 틀리는 경계를 그어요. <code>class_weight=\"balanced\"</code>는 종류마다 \"건수에 반비례하는 값\"을 매겨서 실패 한 건을 더 비싸게 쳐요.</p>"
           + K.out("f_rbf")
           + K.code_table([("SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\")", "RBF 커널 SVM"), ("class_weight=\"balanced\"", "드문 종류를 틀릴 때 값을 키우기. 로지스틱, 트리, 포레스트에도 있어요")])
           + K.try_("gamma를 0.01부터 10까지 바꾸면?", "t3"), "s3"),
    K.step(4, "LDA: 분포를 가정하고 경계 긋기", "무리마다 정규분포 모양으로 퍼져 있다고 보고, 새 점이 어느 무리에서 나왔을 확률이 더 높은지로 판단해요.",
           "<p>로지스틱은 \"입력 x가 주어졌을 때 실패일 확률\" P(실패 | x)를 <b>바로</b> 배워요(판별 모델). <b>LDA</b>(선형 판별분석)는 거꾸로 가요. 무리마다 \"실패한 요청은 x가 이렇게 퍼져 있다\", 곧 P(x | 실패)를 배우고(생성 모델), <b>베이즈 정리</b>로 뒤집어요.</p>"
           "<p><code>P(실패 | x) = P(x | 실패) × P(실패) ÷ P(x)</code>. P(실패)는 데이터를 보기 전 믿음인 <b>사전 확률</b>(train의 실패 비율 0.30), P(x | 실패)는 실패 무리 안에서 x가 나올 가능도, 결과 P(실패 | x)는 x를 보고 나서의 <b>사후 확률</b>이에요. P(x)는 실패, 성공 양쪽에 같아서 비교할 때는 빼도 돼요. 사후 확률이 더 큰 쪽으로 답해요.</p>"
           "<p>LDA의 가정은 둘이에요. ① 무리마다 x가 <b>다변량 정규분포</b>(평균 둘레로 종 모양, 여러 열이면 타원 모양)로 퍼져 있다. ② 두 무리의 <b>퍼짐 모양(공분산 행렬)이 같다</b>. 공분산 행렬은 열마다 얼마나 퍼졌는지와 두 열이 함께 움직이는 정도를 모은 표예요. 이 두 가정을 넣고 식을 정리하면 x의 제곱 항이 지워져서, 경계가 로지스틱처럼 <b>곧은 선</b>이 돼요.</p>"
           "<p><b>또 다른 눈</b>: LDA는 \"모든 점을 한 줄 위로 내렸을 때, 두 무리의 평균은 멀고 각 무리 안 퍼짐은 좁아지는 방향\"을 찾는 것과 같아요. 이 비율 (평균 차)² ÷ (퍼짐의 합)을 피셔 기준이라고 해요. 그래서 LDA는 분류기이면서 차원 축소이기도 해요. 종류가 c개면 c − 1개 축으로 줄여요(여기는 2종류라 1개).</p>"
           + K.table(["", "로지스틱 회귀", "LDA"], [["배우는 것", "P(실패 | x)를 바로", "P(x | 무리)와 P(무리)를 배워 뒤집기"], ["가정", "거의 없음", "정규분포, 같은 공분산"],
                                                  ["가정이 맞을 때", "", "데이터가 적어도 더 잘해요"], ["가정이 틀릴 때", "더 튼튼해요", ""], ["경계", "곧은 선", "곧은 선"]])
           + K.out("f_lda")
           + K.code_table([("LinearDiscriminantAnalysis()", "LDA. 기본은 사전 확률을 train 비율로"), ("priors_", "사전 확률"), ("lda.transform(X)", "한 줄(종류 수 − 1개 축) 위로 내린 판별 점수")]), "s4"),
    K.step(5, "QDA: 종류마다 다른 모양 허용", "LDA에서 \"퍼짐 모양이 같다\"는 가정을 빼요. 경계가 곡선이 돼요.",
           "<p>실패한 요청과 성공한 요청이 서로 다른 모양으로 퍼져 있을 수도 있어요. 예를 들어 실패는 입력 길이가 넓게 퍼지고, 성공은 좁게 모여 있는 식이요. <b>QDA</b>(이차 판별분석)는 무리마다 공분산을 따로 배워요. 그러면 정리한 식에 x의 제곱 항이 남아서 경계가 <b>이차 곡선</b>(원, 타원, 포물선 같은)이 돼요.</p>"
           "<p>대신 배울 값이 많아요. 열이 d개면 공분산 행렬 하나에 d × (d + 1) ÷ 2개 값이 있고, QDA는 이걸 무리마다 배워요. 6열이면 무리마다 21개예요. 데이터가 적으면 공분산이 불안정해져서, <code>reg_param</code>으로 공분산을 조금 둥글게(모든 방향으로 같은 퍼짐 쪽으로) 당겨요. Ridge처럼 규제예요.</p>"
           + K.table(["", "LDA", "QDA"], [["공분산", "두 무리가 같이 하나", "무리마다 따로"], ["경계", "직선", "곡선"], ["배울 값", "적음", "많음, 데이터가 많이 필요"], ["어울리는 때", "무리들의 퍼짐 모양이 비슷할 때", "퍼짐 모양이 확실히 다를 때"]])
           + K.out("f_qda")
           + K.code_table([("QuadraticDiscriminantAnalysis(reg_param=0.1)", "QDA. 공분산을 10%쯤 둥근 쪽으로 당겨요")]), "s5"),
    K.step(6, "PCA: 정답 없이 열 줄이기", "열들이 함께 움직이는 방향을 찾아, 데이터가 가장 많이 퍼진 방향 몇 개로 다시 표현해요.",
           "<p>열이 많으면 데이터가 공간에 듬성듬성해져서(1쪽의 차원의 저주) 모델이 배우기 어렵고, 계산도 늘고, 그림으로 볼 수도 없어요. 해결책은 셋이에요. 아는 것으로 중요한 열만 고르기, 규제(1쪽 ④), 그리고 <b>차원 축소</b>예요. 실제 데이터는 열이 많아도 그 안의 구조는 훨씬 적은 수의 방향으로 설명되는 경우가 많아요. 예: 28×28 손글씨 그림은 784열이지만, 실제로는 그보다 훨씬 작은 \"글자 모양 공간\" 위에 있어요.</p>"
           "<p><b>PCA</b>(주성분 분석)는 \"데이터를 한 방향으로 내렸을 때 분산(퍼짐)이 가장 큰 방향\"을 찾아요. 많이 퍼진 방향일수록 점들을 잘 구별해 주는 정보가 많다고 보는 거예요. 순서는 이래요.</p>"
           "<ol><li>열마다 평균을 빼서 데이터의 가운데를 원점으로 옮겨요. 단위가 다르면 크기도 맞춰요(여기는 ①의 <code>build</code>, 곧 1쪽 ③의 전처리로 이미 맞췄어요).</li>"
           "<li>열 사이 공분산 행렬 Σ를 구해요.</li>"
           "<li>Σ의 <b>고유벡터</b>를 구해요. 고유벡터 w는 Σ를 곱해도 방향은 그대로고 길이만 λ배 되는 방향이에요(Σw = λw). 이 λ가 <b>고유값</b>이고, 그 방향으로 내렸을 때의 분산과 같아요. \"단위 길이 방향 중 분산이 가장 큰 것\"을 라그랑주 승수로 풀면 바로 이 식이 나와요.</li>"
           "<li>고유값이 큰 순서로 1번, 2번 <b>주성분</b>이에요. 주성분끼리는 서로 직각이라 겹치는 정보가 없어요.</li>"
           "<li>앞에서 k개만 골라, 데이터를 그 방향들로 다시 표현해요(내적). 버린 방향의 정보는 잃어서, 원래대로 완벽히 되돌릴 수는 없어요.</li></ol>"
           "<p><b>몇 개?</b> 주성분 i가 담은 비율은 λᵢ ÷ (모든 λ의 합)이에요. 누적이 보통 70~90%를 넘는 가장 작은 k를 고르거나, 비율이 확 작아지는 곳(엘보)을 봐요.</p>"
           + K.out("f_pca") + K.out("s6l")
           + K.table(["", "PCA", "LDA (④)"], [["찾는 방향", "분산이 가장 큰 방향", "두 무리가 가장 잘 갈리는 방향"], ["정답", "안 봐요 (비지도)", "봐요 (지도)"], ["쓰는 곳", "열 줄이기, 잡음 덜기, 그림 그리기", "분류의 앞 단계"]])
           + "<p>PCA는 곧은 방향만 찾아서, 데이터가 휘어진 면 위에 있으면 잘 못 줄여요. 그럴 때는 점 사이 거리를 최대한 지키며 줄이는 방법들(MDS, t-SNE, UMAP)을 써요. PCA는 그중 거리를 곧은 방향으로만 지키는 특별한 경우예요.</p>"
           + K.code_table([("PCA().fit(Z)", "주성분을 모두 구해요"), ("explained_variance_ratio_", "주성분마다 담은 분산 비율 (λ ÷ 합)"), ("np.searchsorted(np.cumsum(r), 0.70) + 1", "누적이 처음 70% 이상이 되는 개수"),
                           ("PCA(n_components=k)", "앞에서 k개만 남기기. Pipeline 안에 두면 train으로만 방향을 배워요"), ("components_", "주성분마다 원래 열에 곱하는 가중치")]), "s6"),
    K.step(7, "같은 교차검증으로 비교하기", "2쪽 ⑫와 같은 80건, 같은 다섯 덩어리로 이 쪽 모델들을 재요.",
           "<p>valid 20건 한 번으로는 점수가 크게 흔들린다는 걸 2쪽에서 봤어요. 그래서 2쪽 ⑫와 같은 <code>StratifiedKFold(random_state=0)</code>로 나눠, 2쪽 표와 줄 단위로 비교할 수 있게 해요.</p>"
           + K.code_table([("cross_validate(m, X, y, cv=cv, scoring=\"f1\")", "다섯 번 배우고 재서 점수 목록. 전처리와 PCA도 매번 그 차례의 train으로만 다시 배워요")]), "s7"),
]
practice = [("SVM 앞에는 꼭 크기 맞추기", "SVM은 거리와 내적으로 판단해서, 큰 단위 열이 도로 방향을 다 정해 버려요. RBF는 특히 그래요."),
            ("데이터가 수만 건 넘으면 (SVM)", "커널 SVM은 모든 점 쌍의 커널 값을 다뤄서 건수의 제곱에서 세제곱만큼 느려져요. 곧은 경계로 충분하면 <code>LinearSVC</code>나 로지스틱을 써요."),
            ("SVM에서 확률이 필요하다", "SVM은 원래 \"어느 쪽인가\"와 경계까지 거리만 줘요. <code>probability=True</code>를 켜면 안에서 교차검증으로 확률을 맞춰 주지만 느려지고, 결과가 <code>predict</code>와 조금 다를 수 있어요."),
            ("종류가 셋 이상 (SVM)", "SVM은 원래 둘만 가르니까, 종류 쌍마다 분류기를 만들어 투표하거나(One-vs-One, <code>SVC</code> 기본) 종류마다 \"이것 대 나머지\" 분류기를 만들어요(One-vs-Rest)."),
            ("판별분석에 필요한 데이터 양", "보통 전체 건수가 열 수의 3배 이상, 가장 작은 종류의 건수가 열 수보다 많아야 공분산을 믿을 수 있어요. 이 예제는 실패 18건, 6열이라 LDA는 되지만 QDA는 빠듯해요."),
            ("LDA에 좋은 입력", "서로 닮지 않은(상관이 낮은) 열이 좋아요. 닮은 열이 있으면 공분산의 역행렬이 불안정해져요. ①의 <code>drop=\"first\"</code>가 그 이유예요."),
            ("PCA 전에", "크기 맞추기가 필수예요. 안 하면 단위가 큰 열 하나가 분산을 독차지해서 1번 주성분이 그 열 자체가 돼요."),
            ("PCA가 분류를 좋게 하나", "보장은 없어요. 분산이 큰 방향이 정답을 가르는 방향이라는 법은 없거든요. 열이 아주 많거나 서로 닮았을 때 흔들림을 줄이는 데 주로 도움이 돼요 (⑦에서 표준편차가 가장 작았어요). 늘 교차검증으로 확인해요."),
            ("그림으로 보고 싶다", "PCA 두 개로 먼저 보고, 휘어진 구조가 의심되면 t-SNE나 UMAP을 써요. 이 둘은 그림용이고, 축 값 자체에는 뜻이 없어요.")]
libs = [("①", "<code>OneHotEncoder(drop=\"first\")</code>", "같은 정보 열 빼기"), ("②③", "<code>SVC</code>, <code>coef_</code>, <code>n_support_</code>, <code>class_weight</code>", "SVM (scikit-learn)"),
        ("④⑤", "<code>LinearDiscriminantAnalysis</code>, <code>QuadraticDiscriminantAnalysis</code>, <code>transform</code>", "판별분석"), ("⑥", "<code>PCA</code>, <code>explained_variance_ratio_</code>, <code>components_</code>", "주성분 분석"),
        ("⑦", "<code>StratifiedKFold</code>, <code>cross_validate</code>", "교차검증")]
K.build(ROOT / "topics" / "model" / "ml-svm-lda-pca.html", title="SVM, 판별분석, 차원 축소", h1="SVM, 판별분석, 차원 축소: 도로, 분포, 방향으로 가르기", track="모델",
        crumb="모델 &gt; 머신러닝 기초 4/4쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-metrics-clustering.html\" style=\"color:inherit\">평가지표와 군집</a>",
        one_line="<b>SVM</b>은 두 무리 사이에 가장 넓은 도로를 내고, <b>LDA와 QDA</b>는 무리마다 퍼진 모양을 배워 확률로 판단하고, <b>PCA</b>는 정답 없이 데이터가 가장 많이 퍼진 방향 몇 개로 열을 줄여요.",
        why=why, whole=whole, steps=steps, code_file="examples/ml/more_models.py", practice=practice, libs=libs, libs_note="<code>part</code>, <code>build</code>, <code>score</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "로지스틱, 규제, 원-핫 열 문제"), ("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "⑦과 같은 교차검증 표"),
                   ("<a href=\"dl-tensor-training.html\">딥러닝 기초</a>", "신경망으로 넘어가기")],
        sources=["다루는 범위: 부트캠프 Day05~08의 머신러닝 기초를 넓힌 쪽이에요. 수업에서 따로 다루지 않은 모델이라, 그 Day의 예제를 그대로 이어 썼어요",
                 "함께 정리한 자료: 경희대 Pattern Recognition and Machine Learning 7강(판별분석), 9강(SVM), 11강(차원축소). 최대 마진, 서포트 벡터, 소프트 마진과 힌지 손실, 라그랑주 쌍대와 KKT, 커널 트릭과 커널 종류, 베이즈 정리, 생성 모델과 판별 모델, LDA의 가정과 피셔 기준, QDA, 차원의 저주, PCA의 순서와 고유값, 주성분 개수 고르기, PCA와 LDA 비교, MDS를 가져왔어요. 값과 그림은 모두 이 예제로 계산했어요",
                 "데이터: <code>examples/ml/data/requests.csv</code> (가상의 업무 요청 100건)와 <code>split_manifest.csv</code>. 1~3쪽과 같은 예제",
                 "결과 값과 그림: <code>tools/pages/ml_more.py</code>가 <code>examples/ml/more_models.py</code>를 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/svm.html\" target=\"_blank\">Support Vector Machines</a>, <a href=\"https://scikit-learn.org/stable/modules/lda_qda.html\" target=\"_blank\">Linear and Quadratic Discriminant Analysis</a>, <a href=\"https://scikit-learn.org/stable/modules/decomposition.html#pca\" target=\"_blank\">PCA</a>"])
print("ok")
