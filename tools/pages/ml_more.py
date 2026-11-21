"""모델 > 머신러닝 기초 4쪽: SVM, 판별분석, PCA, 커널 더 깊이, AdaBoost 더 깊이. examples/ml/more_models.py 하나가 ①~⑬으로 자란다.
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
ns = run(13)
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

# ---------------- ⑧ 커널이 되는 조건
eig = ns["eig"]
PDS_OK = ["linear", "poly (2차)", "rbf", "laplacian"]
NOT_K = ["sigmoid", "exp(−‖x−z‖³)"]
tol = 1e-9                                                          # 이보다 작은 음수는 컴퓨터 반올림으로 봐요
S.claim(ns["trick_gap"] < 1e-10, "⑧ 64열 내적과 8열 내적의 제곱이 반올림 수준에서 같다")
S.claim(all(eig[k].min() > -tol for k in PDS_OK) and all(eig[k].min() < -0.05 for k in NOT_K), "⑧ 네 커널은 고윳값이 모두 0 이상, sigmoid와 세제곱 거리는 뚜렷한 음수")
S.claim(ns["sig_scores"]["f1"] < ns["rbf_scores"]["f1"], "⑧ sigmoid SVM이 RBF보다 valid F1이 낮다")
n_neg = {k: int((v < -tol).sum()) for k, v in eig.items()}
outs["s8"] = S.box("⑧ (xᵀz)²를 두 길로 계산하기 (train 60건, 8열)", [(f"64열을 만들어 내적 vs 8열 내적의 제곱 → 가장 큰 차이 {ns['trick_gap']:.1e}", 1),
                                                                (f"sigmoid 커널 SVM → valid F1 {ns['sig_scores']['f1']:.2f} (③의 RBF는 {ns['rbf_scores']['f1']:.2f})", 2, "newf")],
                   [f"두 값의 차이가 컴퓨터 반올림 수준이에요. 열 {ns['Z'].shape[1]}개를 열 두 개씩 곱한 {ns['pairs'].shape[1]}열로 옮겨 내적한 것과, 원래 {ns['Z'].shape[1]}열의 내적을 제곱한 것이 같은 계산이에요. 열이 d개면 옮긴 공간은 d²열인데, 커널은 d열 계산 한 번으로 끝나요.",
                    "sklearn은 커널이 아닌 함수도 막지 않아서 학습은 끝나요. 다만 아래 표처럼 고윳값이 음수라 \"가장 좋은 답을 찾았다\"는 보장이 없어요. 이 데이터에서는 RBF보다 낮게 나왔어요."])
outs["t8"] = S.table("커널 행렬 60 × 60의 고윳값", ["함수", "가장 작은 고윳값", "음수 고윳값 수", "커널인가"],
                     [[f"<code>{k}</code>", f"{(max(eig[k].min(), 0.0) if eig[k].min() > -tol else eig[k].min()):+.4f}", n_neg[k], "커널" if k in PDS_OK else "<b>아님</b>"] for k in eig],
                     ["대칭 행렬의 고윳값이 모두 0 이상이면 양의 준정부호(PSD)예요. 어떤 점들을 골라도 커널 행렬이 PSD인 함수가 커널(PDS 커널)이에요. 0에 아주 가까운 음수(−0.000000000001 같은)는 컴퓨터 반올림이라 0으로 적었어요.",
                      f"sigmoid는 음수 고윳값이 {n_neg['sigmoid']}개예요. 여기서는 tanh(a × xᵀz + b)의 a = 1 ÷ {ns['Z'].shape[1]}, b = 1로 둘 다 양수인데도 그래요. 이름은 커널이지만 엄밀히는 커널이 아니에요.",
                      f"<code>exp(−‖x−z‖³)</code>는 RBF(<code>exp(−γ‖x−z‖²)</code>)에서 지수만 2를 3으로 바꿨는데 커널이 아니에요(음수 {n_neg['exp(−‖x−z‖³)']}개). 유클리드 거리의 p제곱은 p가 2 이하일 때만 커널을 만들어요. laplacian(맨해튼 거리를 쓴 exp(−γ × 거리))은 커널이에요."])

# ---------------- ⑨ 커널 SVM의 답
rbf_m = ns["rbf"]
n_sv9 = len(rbf_m.support_)
m60 = len(ns["y_train"])
loo_rate = len(ns["loo_wrong"]) / m60
S.claim(ns["hand_gap"] < 1e-10, "⑨ 손으로 만든 예측이 decision_function과 같다")
S.claim(0 < ns["n_at_C"] < n_sv9, "⑨ 서포트 벡터 중 일부는 α = C, 나머지는 0 < α < C")
S.claim(loo_rate <= n_sv9 / m60 and set(ns["loo_wrong"]) <= set(rbf_m.support_), "⑨ LOO 오류율이 서포트 벡터 비율 이하이고, 틀린 점은 모두 서포트 벡터")
S.claim(ns["Z"].var() < 1 and ns["g"] > 1 / ns["Z"].shape[1], "⑨ 전체 분산이 1보다 작아 gamma가 1 ÷ 열 수보다 크다")
outs["s9"] = S.box("⑨ ③의 RBF SVM을 손으로 다시 계산하기", [(f"gamma=\"scale\" → 1 ÷ ({ns['Z'].shape[1]}열 × 분산 {ns['Z'].var():.4f}) = {ns['g']:.4f}", 1),
                                                       (f"Σ αᵢyᵢ K(xᵢ, x) + b vs decision_function (valid 20건) → 가장 큰 차이 {ns['hand_gap']:.1e}", 2, "newf"),
                                                       (f"서포트 벡터 {n_sv9}개 / {m60}건 → α = C(넘어온 점) {ns['n_at_C']}개, 0 < α < C(도로 끝선 위) {n_sv9 - ns['n_at_C']}개", 3),
                                                       (f"leave-one-out → 틀린 건 {len(ns['loo_wrong'])}개 / {m60}건 (오류율 {loo_rate:.2f}), 서포트 벡터 비율 {n_sv9 / m60:.2f}", 4, "newf")],
                   [f"크기를 맞춘 숫자 열은 분산이 1이지만, 원-핫 열은 0과 1뿐이라 분산이 작아요. 그래서 전체 분산이 1보다 작고, gamma가 1 ÷ {ns['Z'].shape[1]}보다 커요.",
                    f"SVM이 배운 건 서포트 벡터 {n_sv9}개와 그 무게(<code>dual_coef_</code>), 절편 하나뿐이에요. 이걸로 새 점과 서포트 벡터 사이의 RBF 값을 무게 붙여 더하니, sklearn의 답과 반올림 수준까지 같아요.",
                    f"α가 상한 C에 닿은 {ns['n_at_C']}개는 도로 안이나 반대편으로 넘어온 점이에요. C는 \"넘어온 점 하나가 경계를 얼마나 세게 당길 수 있나\"의 한도예요. 나머지 {n_sv9 - ns['n_at_C']}개는 도로 끝선에 딱 닿아 있어요.",
                    f"틀린 {len(ns['loo_wrong'])}건은 모두 서포트 벡터예요. 서포트 벡터가 아닌 점은 빼고 다시 배워도 답이 그대로라 반드시 맞히거든요. 그래서 leave-one-out 오류율은 언제나 서포트 벡터 비율 이하예요. 여기서는 {loo_rate:.2f} ≤ {n_sv9 / m60:.2f}{K.josa(f'{n_sv9 / m60:.2f}', '이라', '라')} 맞지만 꽤 느슨해요. 두 무리가 많이 섞여 서포트 벡터가 많기 때문이에요."])

# ---------------- ⑩ 커널을 흉내 내는 열
rg = ns["rff_gap"]
Ds = sorted(rg)
drops = [rg[a][0] / rg[b][0] for a, b in zip(Ds, Ds[1:])]          # D를 10배 늘릴 때 평균 차이가 줄어든 배수
n_dev = len(ns["dev_X"])
n_same = int(round(ns["same_answer"] * n_dev))
v_diff = int((ns["svm_rff"].predict(ns["X_valid"]) != ns["svm_rbf"].predict(ns["X_valid"])).sum())
S.claim(all(rg[a][0] > rg[b][0] and rg[a][1] > rg[b][1] for a, b in zip(Ds, Ds[1:])), "⑩ D가 클수록 평균, 최대 차이가 모두 줄어든다")
S.claim(all(1.8 < d < 4.5 for d in drops), "⑩ 10배마다 2~4배쯤 줄어든다")
S.claim(ns["same_answer"] >= 0.9, "⑩ 80건 중 90% 넘게 같은 답")
big_m = 100_000
outs["s10"] = S.table("⑩ RBF 커널을 무작위 cos 열 D개로 흉내 내기 (train 60 × 60칸)", ["D (만든 열 수)", "평균 차이", "가장 큰 차이"],
                      [[f"{D:,}", f"{rg[D][0]:.3f}", f"{rg[D][1]:.3f}"] for D in Ds],
                      [f"D를 10배 늘릴 때마다 평균 차이가 {', '.join(f'{d:.1f}배' for d in drops)} 줄었어요. 무작위로 뽑은 방향의 평균이라, 차이는 대략 1 ÷ √D로 줄어요(10배면 √10 ≈ 3.2배).",
                       f"D = 1,000 열 위에 선형 SVM을 올리니 train과 valid {n_dev}건 중 {n_same}건에서 ③의 커널 SVM과 같은 답을 냈어요. 다른 답 {n_dev - n_same}건 중 {v_diff}건이 valid에 있어서 valid F1이 달라 보여요(③ {ns['rbf_scores']['f1']:.2f}, 흉내 낸 쪽 {ns['rff_scores']['f1']:.2f}). valid의 실패가 {int(ns['y_valid'].sum())}건뿐이라 답 하나에도 F1이 크게 움직여요.",
                       f"이 예제는 {m60}건이라 커널 행렬이 {m60 * m60:,}칸뿐이지만, {big_m:,}건이면 {big_m * big_m:,}칸이라 실수 하나에 8바이트씩 {big_m * big_m * 8 / 1e9:,.0f}GB가 필요해요. 흉내 낸 열은 건수 × D칸이라, D가 1,000이면 {big_m * 1000 * 8 / 1e9:.1f}GB로 줄어요."])
# ---------------- ⑪ AdaBoost를 직접
hist, al = ns["history"], ns["alphas"]
T_ALL = len(al)
Dl, heavy = ns["D_last"], ns["heavy"]
heavy_share = Dl[heavy].sum()
heavy_loo = [int(i) for i in heavy if i in set(ns["loo_wrong"])]
S.claim(all(abs(h["wrong_mass"] - 0.5) < 1e-9 for h in hist), "⑪ 무게를 고친 직후 틀린 행의 무게 합은 매번 정확히 ½")
S.claim(ns["same_as_sk"] == 1.0, "⑪ 직접 만든 AdaBoost와 scikit-learn의 valid 답이 모두 같다")
S.claim(all(h["eps"] < 0.5 for h in hist) and all(a > 0 for a in al), "⑪ 500번 모두 ε < ½, α > 0")
S.claim(abs(hist[0]["eps"] - 0.2) < 1e-9, "⑪ 첫 트리의 ε은 0.2")
S.claim(len(heavy_loo) >= 3 and heavy_share > 0.35, "⑪ 무게가 큰 5건이 무게의 35% 넘게 갖고, 그중 3건 이상이 ⑨의 leave-one-out에서 틀린 점")
outs["s11"] = S.box("⑪ AdaBoost를 직접 500번 (질문 하나짜리 트리, train 60건)",
                    [(f"1번째 트리 → ε {hist[0]['eps']:.3f}, α {al[0]:.3f}", 1),
                     (f"2번째 트리 → ε {hist[1]['eps']:.3f}, α {al[1]:.3f} / 3번째 → ε {hist[2]['eps']:.3f}, α {al[2]:.3f}", 2),
                     (f"무게를 고친 직후 틀린 행의 무게 합 → {T_ALL}번 모두 {hist[0]['wrong_mass']:.3f}", 3, "newf"),
                     (f"scikit-learn AdaBoost(트리 200개)와 valid 답이 같은 비율 → {ns['same_as_sk']:.2f}", 4),
                     (f"{T_ALL}번 뒤 무게가 가장 큰 5건 → {', '.join(f'{i}번 {Dl[i]:.3f}' for i in heavy)} (합 {heavy_share:.2f}, 처음 무게는 1건당 {1 / len(Dl):.3f})", 5, "newf")],
                    [f"첫 트리는 입력 길이로 나눠 {hist[0]['eps']:.0%}를 틀렸어요. α = ½ log((1 − 0.2) ÷ 0.2) = ½ log 4 = {al[0]:.3f}예요. 틀린 비율이 ½에 가까워질수록 α가 0에 가까워져요.",
                     "무게를 고치면 틀린 행 쪽 무게의 합이 언제나 정확히 ½이 돼요. 방금 고른 트리는 새 무게로 재면 동전 던지기와 같아서, 다음 트리는 그 트리가 놓친 것을 봐야 점수를 얻어요.",
                     "직접 짠 20줄과 scikit-learn이 같은 답을 내요. scikit-learn의 α는 log((1 − ε) ÷ ε)로 2배지만, 모든 α가 같은 배수라 부호(답)는 같아요.",
                     f"무게가 큰 5건이 전체의 {heavy_share:.0%}를 가져요. 그중 {', '.join(f'{i}번' for i in heavy_loo)}은 ⑨에서 leave-one-out으로 틀린 점이기도 해요. 두 모델이 똑같이 \"어렵다\"고 본 행이라, 라벨이 잘못됐거나 정말 예외인 요청일 수 있어요."])

# ---------------- ⑫ train 오차와 지수 손실
te, bd, bg, el = ns["train_err"], ns["bound"], ns["bound_gamma"], ns["exp_loss"]
fz, sz = ns["first_zero"], ns["stay_zero"]
gap_exp = float(np.abs(el - bd).max())
S.claim(gap_exp < 1e-12, "⑫ 지수 손실의 평균과 Z의 곱이 반올림 수준에서 같다")
S.claim(all(te[t] <= bd[t] <= bg[t] + 1e-12 for t in range(T_ALL)), "⑫ 매 T에서 train 오차 ≤ ΠZ ≤ exp(−2Σ(½−ε)²)")
S.claim(50 < fz <= sz < 200 and not (te[sz - 1:] > 0).any() and bd[-1] < 0.05, "⑫ train 오차는 트리 50~200개 사이에서 0이 되고 그 뒤로 계속 0, 500개에서 상한은 0.05 아래")
S.claim(hist[-1]["eps"] > 0.45, "⑫ 마지막 트리의 ε은 0.45보다 크다")
outs["s12"] = S.box("⑫ train 오차와 그 상한 (⑪의 500번)",
                    [(f"지수 손실의 평균 vs Z₁ × ... × Z_T → 가장 큰 차이 {gap_exp:.1e}", 1),
                     (f"트리 10개 → train 오차 {te[9]:.3f} ≤ ΠZ {bd[9]:.3f} ≤ exp(−2Σ(½−ε)²) {bg[9]:.3f}", 2),
                     (f"train 오차가 처음 0 → 트리 {fz}개 (그때 ΠZ {bd[fz - 1]:.3f}), 계속 0 → 트리 {sz}개부터", 3, "newf"),
                     (f"트리 {T_ALL}개 → ΠZ {bd[-1]:.3f}, 마지막 트리의 ε {hist[-1]['eps']:.3f}", 4)],
                    ["지수 손실의 평균과 Z의 곱은 이름만 다른 같은 값이에요. 그리고 train 오차는 언제나 그 아래에 있어요(아래 그림).",
                     f"상한이 줄어드는 속도는 트리마다 ε이 ½에서 얼마나 먼가로 정해져요. 뒤로 갈수록 ε이 {hist[-1]['eps']:.2f}쯤으로 ½에 가까워져서, 상한이 천천히 줄어요. 남은 어려운 행만 보다 보니 질문 하나로는 동전 던지기보다 조금 나은 정도만 맞히는 거예요."])
Tx = np.arange(1, T_ALL + 1)
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(Tx, bg, color=C["gray"], lw=1.6, ls="--", label="exp(−2 Σ (½ − ε)²)")
ax.plot(Tx, bd, color=OK, lw=2.4, label="지수 손실의 평균 = Z₁ × ... × Z_T")
ax.step(Tx, te, where="post", color=FAIL, lw=2.0, label="train 오차")
ax.axvline(fz, color=C["ink"], lw=.8, ls=":")
ax.annotate(f"트리 {fz}개에서 처음 0", (fz, 0.02), (fz + 60, 0.45), fontsize=10, color=C["ink"], arrowprops=dict(arrowstyle="->", color=C["ink"]))
ax.set(xlabel="트리 수 T", ylabel="값", title="train 오차는 지수 손실 아래에 있고, 지수 손실은 곱으로 줄어요", ylim=(0, 0.9))
ax.legend(fontsize=9, loc="upper right")
P_BOUND = fig_save(fig, "adaboost_bound")
outs["f_bound"] = chart("AdaBoost의 train 오차와 두 상한 (⑪의 500번)", P_BOUND, "",
                        ["분홍 계단이 실제 train 오차, 보라가 지수 손실의 평균이에요. 계단은 보라 선을 넘은 적이 없어요.",
                         f"계단은 트리 {fz}개에서 처음 0이 되고, 몇 번 한 건씩 다시 틀리다가 {sz}개부터는 계속 0이에요. 보라 선은 그 뒤에도 계속 내려가요. 틀린 행이 없어진 뒤에도 AdaBoost는 \"맞히긴 했지만 아슬아슬한\" 행을 계속 밀어내요. 이게 ⑬의 마진이에요."])
uu = np.linspace(-2, 2.5, 400)
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.plot(uu, (uu <= 0).astype(float), color=C["ink"], lw=2.2, label="0-1 손실 (틀리면 1)")
ax.plot(uu, np.exp(-uu), color=OK, lw=2.4, label="지수 손실 e^(−u) (AdaBoost)")
ax.plot(uu, np.log2(1 + np.exp(-uu)), color=C["blue"], lw=2.0, label="로그 손실 log₂(1 + e^(−u)) (로지스틱)")
ax.plot(uu, np.maximum(0, 1 - uu), color=FAIL, lw=2.0, ls="--", label="힌지 손실 max(0, 1 − u) (SVM)")
ax.set(xlabel="u = y × f(x)   (양수면 맞힘, 클수록 확실히 맞힘)", ylabel="손실", ylim=(0, 4.2), title="0-1 손실 대신 줄이는 매끄러운 손실들")
ax.legend(fontsize=9, loc="upper right")
P_LOSS = fig_save(fig, "margin_losses")
outs["f_loss"] = chart("한 행의 손실을 u = y × f(x)로 그리면", P_LOSS, "",
                       ["넷 다 u가 클수록(확실히 맞힐수록) 작아지고, 0-1 손실과 같거나 그 위에 있어요. 그래서 어느 것을 줄여도 틀린 수의 상한을 줄이는 셈이에요.",
                        "왼쪽(크게 틀린 쪽)을 보면 차이가 나요. 지수 손실은 u가 −2면 e² ≈ 7.4로 치솟아요. 로그 손실과 힌지 손실은 곧은 선에 가깝게만 늘어요. 그래서 AdaBoost는 크게 틀린 한 행에 무게를 몰아주고, 라벨이 잘못된 행에 약해요."])
dep_rows = []
for d in (1, 2, 3):
    n = ns if d == 1 else run(13, [("h = DecisionTreeClassifier(max_depth=1, random_state=0)", f"h = DecisionTreeClassifier(max_depth={d}, random_state=0)")])
    dep_rows.append((d, n["first_zero"], min(h["eps"] for h in n["history"]), n["margins"][-1].min(), n["valid_err"][-1]))
S.claim(dep_rows[0][1] > dep_rows[1][1] > dep_rows[2][1] and dep_rows[0][2] > dep_rows[2][2], "⑫ 트리가 깊을수록 ε이 작고 train 오차가 더 빨리 0이 된다")
S.claim(max(abs(r[4] - dep_rows[0][4]) for r in dep_rows) <= 0.05 + 1e-9, "⑫ 깊이를 바꿔도 valid 오차는 20건 중 한 건(0.05) 차이 안")
outs["t12"] = S.table("AdaBoost의 트리를 깊이 1, 2, 3으로 바꾸면 (500번)", ["깊이", "train 오차가 처음 0", "가장 작은 ε", "500번 뒤 가장 작은 마진", "500번 뒤 valid 오차"],
                      [[f"<b>{d}</b> (지금)" if d == 1 else d, f"트리 {z}개", f3(e), f3(mg_), f"{v:.2f}"] for d, z, e, mg_, v in dep_rows],
                      [f"트리가 깊을수록 한 번에 더 잘 맞혀서(ε이 작아서) 상한이 빨리 줄고, train 오차가 {dep_rows[0][1]}개 → {dep_rows[1][1]}개 → {dep_rows[2][1]}개 만에 0이 돼요.",
                       f"그런데 valid 오차는 {min(r[4] for r in dep_rows):.2f}~{max(r[4] for r in dep_rows):.2f}로, 20건 중 한 건 차이 안에서 오르내릴 뿐이에요. 기본 모델이 강할수록 train에 빨리 맞춰질 뿐, 처음 보는 데이터에 강해지는 건 아니에요. 그래서 AdaBoost는 보통 깊이 1~3의 약한 트리를 쓰고, 트리 수를 교차검증으로 정해요."])

# ---------------- ⑬ 마진
mg, ve = ns["margins"], ns["valid_err"]
mark = [10, 100, fz, T_ALL]
cvm = {T: v.mean() for T, v in ns["ada_cv"].items()}
best_T = max(cvm, key=cvm.get)
cv7 = {k: v.mean() for k, v in ns["cv_scores"].items()}
best7 = max(cv7, key=cv7.get)
S.claim(mg[9].min() < 0 < mg[fz - 1].min() < mg[-1].min(), "⑬ 가장 작은 마진이 음수에서 양수로 올라가고, 0이 된 뒤에도 커진다")
S.claim(np.median(mg[-1]) < np.median(mg[9]), "⑬ 마진의 가운데값은 오히려 줄었다")
S.claim(ve[-1] > ve[9] and best_T == 50, "⑬ valid 오차는 트리를 늘릴수록 나빠졌고, 교차검증은 50개가 가장 좋다")
S.claim(mg[-1].min() < 0.05, "⑬ 500번 뒤 가장 작은 마진도 0.05 아래")
outs["s13"] = S.table("⑬ 트리 수에 따른 L1 마진과 오차 (train 60건, valid 20건)", ["트리 수", "train 오차", "가장 작은 마진", "마진의 가운데값", "valid 오차"],
                      [[f"<b>{T}</b>" + (" (train 오차 첫 0)" if T == fz else ""), f3(te[T - 1]), f"{mg[T - 1].min():+.3f}", f"{np.median(mg[T - 1]):.3f}", f"{ve[T - 1]:.2f}"] for T in mark],
                      [f"가장 작은 마진은 {mg[9].min():+.3f}에서 {mg[-1].min():+.3f}로 올라갔어요. train 오차가 0이 된 뒤에도 계속 커져요. 대신 가운데값은 {np.median(mg[9]):.3f}에서 {np.median(mg[-1]):.3f}로 줄었어요. AdaBoost는 가장 아슬아슬한 행을 끌어올리느라, 이미 넉넉하던 행의 마진을 조금씩 내줘요.",
                       f"valid 오차는 오히려 {ve[9]:.2f}에서 {ve[-1]:.2f}로 나빠졌어요. valid가 20건이라 한 건이 0.05지만, 아래 교차검증도 같은 쪽이에요."])
outs["s13cv"] = S.table("AdaBoost 트리 수별 5-fold 교차검증 F1 (⑦과 같은 나누기)", ["트리 수", "다섯 번의 점수", "평균"],
                        [[f"<b>{T}</b>", ", ".join(f3(x) for x in v), f"<b>{f3(v.mean())}</b>" if T == best_T else f3(v.mean())] for T, v in ns["ada_cv"].items()],
                        [f"트리 {best_T}개가 가장 좋고({cvm[best_T]:.3f}), 그 뒤로는 내려가요. 2쪽 ⑫의 XGBoost는 {xgb_cv.mean():.3f}, ⑦에서 가장 좋았던 {best7}{K.josa(best7, '은', '는')} {cv7[best7]:.3f}예요.",
                         "건수가 60~80건이고 라벨이 깨끗하게 갈리지 않는 데이터라, 마진이 커지는 것보다 어려운 행에 맞추느라 생기는 과적합이 더 커요."])
fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.9))
cols = [C["gray"], C["blue"], FAIL, OK]
for T, col in zip(mark, cols):
    v = np.sort(mg[T - 1])
    axs[0].step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=2.2, label=f"트리 {T}개")
axs[0].axvline(0, color=C["ink"], lw=.8, ls=":")
axs[0].set(xlabel="L1 마진 y f(x) ÷ Σα", ylabel="그 값 이하인 행의 비율", title="마진의 누적 분포 (train 60건)", xlim=(-0.6, 1.02))
axs[0].legend(fontsize=9, loc="upper left")
axs[1].plot(Tx, mg.min(axis=1), color=OK, lw=2.2, label="가장 작은 마진")
axs[1].step(Tx, te, where="post", color=FAIL, lw=1.8, label="train 오차")
axs[1].step(Tx, ve, where="post", color=C["blue"], lw=1.8, label="valid 오차")
axs[1].axhline(0, color=C["ink"], lw=.8, ls=":")
axs[1].set(xlabel="트리 수 T", title="train은 좋아지고 마진은 커지는데, valid는 아니에요", ylim=(-0.35, 0.5))
axs[1].legend(fontsize=9, loc="lower right")
P_MARGIN = fig_save(fig, "adaboost_margins")
outs["f_margin"] = chart("트리를 늘리면 마진이 어떻게 바뀌나", P_MARGIN, "",
                         ["왼쪽: 선이 0보다 왼쪽에서 시작하면 틀린 행이 있다는 뜻이에요. 트리가 늘수록 선의 왼쪽 끝이 0을 넘어 오른쪽으로 옮겨 가요(가장 작은 마진이 커져요). 대신 선이 왼쪽으로 모여요(넉넉한 행이 줄어요).",
                          f"오른쪽: 가장 작은 마진(보라)은 오르내리면서도 올라가는데, valid 오차(파랑)는 내려가지 않아요. 마진 이론은 \"마진이 크면 상한이 작다\"는 보장이에요. 이 데이터처럼 마진이 {mg[-1].min():.2f} 남짓이면 상한 자체가 아주 커서 아무것도 보장해 주지 않아요."])

flow = [[1, "준비", "1쪽과 같은 나누기", "drop=first"], [2, "선형 SVM", "가장 넓은 도로", "서포트 벡터, C"], [3, "커널 SVM", "곡선 경계", "gamma, 균형"], [4, "LDA", "분포 가정, 베이즈", "한 줄로 사영"],
        [5, "QDA", "종류마다 다른 모양", "곡선 경계"], [6, "PCA", "정답 없이 열 줄이기", "분산 비율"], [7, "비교", "같은 교차검증", "균형 SVM이 1등"],
        [8, "커널의 조건", "고윳값이 0 이상", "sigmoid는 아님"], [9, "커널 SVM의 답", "서포트 벡터 위의 합", "leave-one-out"], [10, "많은 데이터", "무작위 cos 열", "RBFSampler"],
        [11, "AdaBoost 직접", "틀린 행에 무게", "α와 ε"], [12, "지수 손실", "오차의 상한", "곱으로 줄어요"], [13, "마진", "0이 된 뒤에도", "valid는 별개"]]
S.save(ROOT / "topics" / "model" / "ml-svm-lda-pca-data.js", {"code": code, "names": S.names(code), "range": [1, 13], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>1~3쪽에서 선형 모델, KNN, 트리와 앙상블을 봤어요. 표 데이터 분류에는 오래 쓰여 온 모델이 셋 더 있어요. 서로 생각하는 방식이 달라요.</p>
{K.table(["모델", "한 문장", "1~3쪽과 닮은 점"], [["<b>SVM</b> (②③)", "두 무리 사이에 <b>가장 넓은 도로</b>를 내요", "로지스틱처럼 곧은 경계에서 시작해요. 커널을 쓰면 곡선 경계도 돼요"],
                                                 ["<b>LDA, QDA</b> (④⑤)", "무리마다 <b>어떻게 퍼져 있는지</b>를 배우고, 새 점이 어느 무리에서 나왔을 확률이 높은지 따져요", "확률을 내는 건 로지스틱과 같지만, 계산하는 길이 반대예요"],
                                                 ["<b>PCA</b> (⑥)", "정답 없이, 열을 <b>정보가 많은 몇 개로</b> 줄여요", "3쪽 K-Means처럼 비지도 학습이에요. 1쪽에서 말한 차원의 저주를 덜어요"],
                                                 ["<b>커널 더 깊이</b> (⑧~⑩)", "③의 커널이 <b>언제 되고, 무엇을 배우고, 데이터가 많으면 어떻게 하는지</b> 봐요", "1쪽 KNN처럼 train 점들과의 유사도로 예측해요. 다만 서포트 벡터만, 무게를 붙여서요"],
                                                 ["<b>AdaBoost 더 깊이</b> (⑪~⑬)", "2쪽 ⑩의 부스팅을 <b>직접 짜고</b>, train 오차가 왜 빨리 줄고 그 뒤에 무엇이 바뀌는지 봐요", "SVM의 도로 폭처럼, AdaBoost에도 <b>마진</b>이 있어요"]])}
<p>예제는 1~3쪽과 같은 업무 요청 100건, 같은 train/valid 나누기, 같은 교차검증이에요. 그래서 마지막 ⑦에서 2쪽 ⑫의 점수와 바로 비교할 수 있어요. 다만 3쪽 ⑰에서 test를 이미 한 번 썼기 때문에, 이 쪽에서는 test를 쓰지 않아요.</p>
"""
whole = """<p>같은 준비 위에 SVM 둘, 판별분석 둘, PCA 하나를 차례로 올리고, 같은 교차검증으로 비교해요. ⑧~⑩은 ③의 커널을, ⑪~⑬은 2쪽 ⑩의 AdaBoost를 한 번 더 깊이 들여다봐요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
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
    K.step(8, "커널이 되는 조건: 커널 행렬의 고윳값", "아무 유사도 함수나 커널이 되는 건 아니에요. 점들끼리의 커널 값을 모은 표(커널 행렬)의 고윳값으로 확인해요.",
           "<p>③에서 커널 K(x, z)는 \"점을 새 공간으로 옮긴 뒤의 내적\"이라고 했어요. 먼저 그게 정말인지 숫자로 봐요. 2차 다항 커널 (xᵀz)²는, 열 두 개씩 곱한 새 열(x₁x₁, x₁x₂, ...)을 모두 만들어 내적한 것과 같아요. 아래 상자처럼 두 길의 값이 반올림 수준까지 같아요. RBF는 옮긴 공간의 열이 끝없이 많은 경우(무한 차원)에 해당해서, 열을 만드는 길로는 아예 계산할 수 없어요.</p>"
           + K.out("s8")
           + "<p>그러면 어떤 함수가 \"어떤 공간의 내적\"일까요? 답은 이래요. 어떤 점들 x₁, ..., xₘ을 골라도, 커널 값을 모은 m × m 표 <b>커널 행렬</b>(그람 행렬) [K(xᵢ, xⱼ)]가 대칭이고 <b>고윳값이 모두 0 이상</b>이면 돼요. 이런 행렬을 양의 준정부호(PSD), 이런 함수를 <b>PDS 커널</b>(양의 정부호 대칭 커널)이라고 해요. 이 조건만 맞으면 그 내적을 내는 공간(재생 커널 힐베르트 공간, RKHS)이 반드시 있어요. 공간을 몰라도 함수만 확인하면 되는 거예요.</p>"
           "<p>🤔 <b>왜 중요한가</b>: ②의 쌍대 문제에서 비용은 커널 행렬이 들어간 이차식이에요. 커널 행렬이 PSD면 비용이 그릇 모양(볼록)이라, 바닥이 하나고 SVM은 반드시 그 바닥을 찾아요. 음수 고윳값이 있으면 그 방향으로는 말안장처럼 내려가서, 찾은 답이 가장 좋은 답이라는 보장이 사라져요.</p>"
           + K.out("t8")
           + "<p><b>커널을 만드는 규칙</b>: 커널끼리 더해도, 곱해도, 양수를 곱해도 커널이에요. exp(커널)도 커널이에요. 그래서 exp(γ xᵀz)가 커널이고, 이걸 크기로 나눠 맞춘 것이 RBF예요. 거리 쪽에서 보면, 제곱 거리 ‖x − z‖²는 \"음의 정부호\"(NDS)라는 성질이 있고, NDS인 d에 대해 exp(−t × d)는 언제나 커널이에요. 거리의 p제곱은 p ≤ 2일 때만 NDS라서, 위 표의 세제곱은 커널이 안 돼요.</p>"
           "<p><b>숫자가 아닌 데이터</b>에도 커널을 정의할 수 있어요. 예를 들어 두 글에 함께 나오는 글자 두 개짜리 조각(bigram)의 개수를 곱해 더하면, \"bigram 개수 열\"의 내적이라 커널이에요. DNA 서열, 문장, 그래프를 SVM으로 다룰 때 이런 커널을 써요.</p>"
           + K.code_table([("polynomial_kernel(Z, degree=2)", "(γ xᵀz + c)ᵈ 를 모든 점 쌍에. 기본 γ = 1 ÷ 열 수, c = 1"), ("rbf_kernel(Z, gamma=g)", "exp(−γ‖x − z‖²)"), ("laplacian_kernel(Z)", "exp(−γ‖x − z‖₁), 거리의 1제곱"),
                           ("sigmoid_kernel(Z)", "tanh(γ xᵀz + c)"), ("euclidean_distances(Z)", "모든 점 쌍의 거리 ‖x − z‖"), ("np.linalg.eigvalsh(G)", "대칭 행렬의 고윳값을 작은 것부터"),
                           ("np.einsum(\"ni,nj->nij\", Z, Z)", "점마다 열 두 개씩 곱한 표")]), "s8"),
    K.step(9, "커널 SVM의 답: 서포트 벡터 위의 합", "커널 SVM이 배운 것은 서포트 벡터마다 붙은 무게뿐이에요. 그 무게로 예측을 손으로 다시 만들어요.",
           "<p>②의 선형 SVM은 가중치 θ를 직접 가졌어요. RBF는 옮긴 공간이 무한 차원이라 θ를 저장할 수 없어요. 대신 쌍대 문제의 답인 무게 αᵢ로 예측해요.</p>"
           "<p><code>h(x) = Σ αᵢ yᵢ K(xᵢ, x) + b</code></p>"
           "<p>새 점 x와 train 점 xᵢ의 커널 값(유사도)에 무게를 붙여 더하는 거예요. 1쪽 KNN이 가까운 점들의 답으로 투표한 것과 닮았는데, SVM은 α가 0이 아닌 서포트 벡터만 쓰고 점마다 무게가 달라요.</p>"
           "<p><b>Representer 정리</b>: \"함수의 크기(‖h‖) 규제 + train 점에서의 손실\"을 가장 작게 하는 함수는, 공간이 아무리 커도 언제나 <b>train 점 xᵢ마다 K(xᵢ, x)를 무게 붙여 더한 꼴</b>이에요. 그래서 무한 차원 공간에서 함수를 찾는 문제가 m개 무게를 찾는 문제로 줄어요. SVM뿐 아니라 커널 Ridge 회귀, 커널 로지스틱 회귀가 되는 것도 같은 이유예요.</p>"
           + K.out("s9")
           + "<p><b>α가 말해 주는 것</b>: α = 0이면 도로 바깥, 예측에 안 쓰여요. 0 < α < C면 도로 끝선에 딱 닿은 점이고, α = C면 도로 안이나 반대편으로 넘어온 점이에요. ②의 C는 이 α의 상한이에요.</p>"
           "<p><b>leave-one-out</b>은 한 건씩 빼고 배운 뒤 뺀 건을 맞혀 보기를 m번 하는 교차검증이에요. SVM에서는 \"leave-one-out 오류율 ≤ 서포트 벡터 비율\"이 늘 성립해요. 서포트 벡터가 적을수록(희소할수록) 처음 보는 데이터에 강하다는 근거예요.</p>"
           "<p><b>도로가 넓으면 왜 좋은가 (마진 이론)</b>: SVM이 처음 보는 데이터에서 틀릴 확률의 상한은, 열 수가 아니라 <b>도로 폭과 데이터 수</b>로 정해져요. 그래서 무한 차원인 RBF 공간에서도 넓은 도로를 찾는 게 의미가 있어요. 열이 많아서 생기는 과적합을 도로 폭이 막아 주는 셈이에요.</p>"
           + K.code_table([("dual_coef_", "서포트 벡터마다 αᵢ × yᵢ"), ("support_vectors_", "서포트 벡터의 좌표 (크기 맞춘 8열)"), ("intercept_", "절편 b"), ("decision_function(X)", "경계까지의 부호 있는 값. 0보다 크면 실패"),
                           ("cross_val_predict(m, X, y, cv=LeaveOneOut())", "한 건씩 빼고 배워서, 뺀 건마다 예측")]), "s9"),
    K.step(10, "데이터가 많을 때: 커널을 흉내 내는 열 만들기", "커널 행렬은 건수의 제곱만큼 커져요. 무작위 방향 D개로 만든 cos 열이 RBF 커널을 흉내 내게 하고, 그 위에서 선형 모델을 써요.",
           "<p>커널 SVM은 모든 점 쌍의 커널 값을 다뤄서, 건수가 늘면 메모리와 시간이 제곱 이상으로 늘어요(아래 표의 마지막 설명). 반대로 선형 모델은 건수에 비례해서만 늘어요. 그러면 \"커널과 비슷한 효과를 내는 열\"을 직접 만들어 선형 모델에 넣으면 어떨까요?</p>"
           "<p><b>랜덤 푸리에 특성</b>: RBF처럼 두 점의 차이 x − z만으로 정해지는 커널은, 무작위 방향 ω로 잰 cos 값들의 평균으로 쓸 수 있어요(Bochner 정리). 그래서 ω를 정규분포에서 D개 뽑아 열 <code>√(2/D) × cos(ωᵀx + b)</code>를 만들면(b는 무작위 위치), 두 점의 이 열끼리 내적이 RBF 값에 가까워져요. D가 클수록 정확하고, 차이는 대략 1 ÷ √D로 줄어요.</p>"
           + K.out("s10")
           + "<p>무작위로 뽑은 방향이라 <code>random_state</code>에 따라 값이 조금씩 달라요. 비슷한 방법으로 train 점 일부만 골라 커널 행렬을 근사하는 <code>Nystroem</code>도 있어요. 데이터가 수만 건 넘으면 이 둘 중 하나로 열을 만들고 <code>LinearSVC</code>나 <code>SGDClassifier</code> 같은 빠른 선형 모델을 올려요.</p>"
           + K.code_table([("RBFSampler(gamma=g, n_components=D, random_state=0)", "RBF 커널을 흉내 내는 cos 열 D개. gamma는 흉내 낼 RBF의 γ"), ("Pipeline([... (\"rff\", RBFSampler(...)), (\"model\", SVC(kernel=\"linear\"))])", "전처리 → 열 만들기 → 선형 SVM을 한 번에")]), "s10"),
    K.step(11, "AdaBoost를 직접: 틀린 행에 무게 주기", "질문 하나짜리 트리를 500번 고르면서, 매번 앞에서 틀린 행의 무게를 키워요. 20줄로 직접 짜고 scikit-learn과 맞춰 봐요.",
           "<p>2쪽 ⑩에서 AdaBoost를 \"앞에서 틀린 행에 가중치를 키워서, 그 행을 더 신경 쓰게\"라고 한 줄로 적었어요. 여기서는 그 한 줄을 직접 짜요.</p>"
           "<p><b>약한 학습기</b>: 질문 하나짜리 트리(깊이 1, <b>스텀프</b>)는 혼자서는 약해요. 부스팅이 기대하는 건 딱 하나, \"어떤 무게를 줘도 동전 던지기보다는 조금 낫다\"예요. 무게로 잰 틀린 비율 ε이 ½보다 작기만 하면 돼요. \"이런 약한 학습기를 여럿 묶으면 얼마든지 정확한 분류기를 만들 수 있나\"가 부스팅의 질문이었고, AdaBoost가 그 답이에요.</p>"
           "<p>정답을 +1(실패), −1(성공)로 두면 한 번의 차례는 이래요.</p>"
           "<ol><li>지금 무게 D로 스텀프 h를 하나 고르고, 무게로 잰 틀린 비율 ε을 재요.</li>"
           "<li>그 트리의 발언권 <code>α = ½ log((1 − ε) ÷ ε)</code>. ε이 작을수록 크고, ε = ½이면 0이에요.</li>"
           "<li>무게 고치기: 틀린 행(y × h(x) = −1)은 e^α배, 맞힌 행은 e^(−α)배 하고, 합이 1이 되게 Z로 나눠요.</li></ol>"
           "<p>T번 뒤의 점수는 <code>f(x) = α₁h₁(x) + ... + α_T h_T(x)</code>이고, 부호가 답이에요. 2쪽 ⑨의 포레스트는 트리마다 같은 무게로 투표했는데, AdaBoost는 트리마다 발언권이 달라요.</p>"
           + K.out("s11")
           + "<p>2쪽 ⑩의 Gradient Boosting은 \"남은 잔차를 맞히기\"였어요. AdaBoost는 \"틀린 행의 무게를 키우기\"예요. 달라 보이지만, ⑫에서 보듯 AdaBoost도 어떤 손실을 한 걸음씩 줄이는 방법이에요.</p>"
           + K.code_table([("np.where(y_train == 1, 1, -1)", "정답을 +1, −1로. 그래야 y × h(x)가 맞으면 +1, 틀리면 −1이 돼요"), ("DecisionTreeClassifier(max_depth=1)", "질문 하나짜리 트리 (스텀프)"),
                           ("fit(X, y, sample_weight=D)", "행마다 무게를 주고 학습. 무게가 큰 행을 틀리면 더 큰 손해로 쳐요"), ("D[p != y].sum()", "무게로 잰 틀린 비율 ε"),
                           ("np.cumsum(alphas[:, None] * H_train, axis=0)", "트리 1개, 2개, ..., 500개까지의 점수를 한 번에"), ("AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)", "같은 일을 하는 scikit-learn")]), "s11"),
    K.step(12, "train 오차가 빨리 줄어드는 이유: 지수 손실", "train 오차는 \"트리마다 정해지는 값 Z의 곱\" 아래에 있어요. AdaBoost는 매번 그 Z를 가장 작게 하는 α를 골라요.",
           "<p>⑪의 무게 고치기를 T번 이어서 풀어 쓰면, 행 i의 무게는 <code>e^(−yᵢ f(xᵢ)) ÷ (m × Z₁ × ... × Z_T)</code>예요(m은 행 수). 무게의 합은 1이니까 모든 행을 더하면 이렇게 돼요.</p>"
           "<p><code>(1/m) Σ e^(−yᵢ f(xᵢ)) = Z₁ × Z₂ × ... × Z_T</code></p>"
           "<p>왼쪽은 <b>지수 손실</b>의 평균이에요. 틀린 행은 y × f(x) ≤ 0이라 e^(−y f(x)) ≥ 1이고, 맞힌 행도 0보다 커요. 그래서 train 오차(틀린 행의 비율)는 언제나 지수 손실의 평균 아래예요.</p>"
           "<p>곧 <b>train 오차 ≤ Z₁ × ... × Z_T</b>이고, <code>Z_t = 2√(ε_t(1 − ε_t))</code>는 ε_t < ½이면 1보다 작아요. 1보다 작은 수를 계속 곱하니 상한이 곱으로(지수적으로) 줄어요. 더 정리하면 <code>≤ exp(−2 Σ (½ − ε_t)²)</code>이고, 모든 트리가 ε_t ≤ ½ − γ이면 <code>≤ exp(−2γ²T)</code>예요. γ가 조금이라도 0보다 크면, 트리를 늘리는 만큼 train 오차는 반드시 0으로 가요.</p>"
           + K.out("s12") + K.out("f_bound")
           + "<p><b>α는 왜 하필 ½ log((1 − ε) ÷ ε)인가</b>: Z = (1 − ε)e^(−α) + εe^α를 α로 미분해 0으로 두면 정확히 그 값이 나와요. 곧 AdaBoost는 매번 상한을 가장 많이 줄이는 발언권을 골라요.</p>"
           "<p><b>한 걸음씩 줄이는 방법이었다</b>: 모든 스텀프를 열로 놓고, 지수 손실 <code>(1/m) Σ e^(−y f(x))</code>를 \"한 번에 한 열의 계수만 고치는\" 방법(<b>좌표 하강</b>)으로 줄인다고 해 봐요. 매번 가장 많이 내려가는 열을 고르면 그게 ⑪에서 ε이 가장 작은 스텀프이고, 그 열을 얼마나 고칠지 가장 좋은 걸음 크기를 구하면 그게 α예요. AdaBoost는 지수 손실에 대한 좌표 하강과 정확히 같아요. 2쪽 ⑩의 Gradient Boosting이 \"잔차 = 기울기 반대 방향\"으로 손실을 줄였던 것과 같은 집안이에요.</p>"
           + K.out("f_loss")
           + "<p>지수 손실 대신 1쪽 ⑤의 로그 손실을 같은 방법으로 줄이면, 무게가 덜 공격적으로 커지는 부스팅이 돼요. scikit-learn의 <code>GradientBoostingClassifier</code>와 XGBoost가 분류에서 기본으로 쓰는 손실이 이 로그 손실이에요. 라벨에 잡음이 있으면 이쪽이 더 튼튼해요.</p>"
           + K.code_table([("np.exp(-ys * F_train).mean(axis=1)", "트리 수마다 지수 손실의 평균"), ("np.cumprod([h[\"Z\"] for h in history])", "Z₁, Z₁Z₂, Z₁Z₂Z₃, ... 누적 곱"), ("int(np.argmax(train_err == 0)) + 1", "train 오차가 처음 0이 된 트리 수"), ("np.flatnonzero(train_err > 0)[-1] + 2", "마지막으로 틀린 다음 트리 수. 그 뒤로는 계속 0")])
           + K.try_("스텀프 대신 깊이 2, 3인 트리를 쓰면?", "t12"), "s12"),
    K.step(13, "마진: train 오차가 0이 된 뒤에도 바뀌는 것", "점수를 발언권의 합으로 나눈 L1 마진이 트리를 늘릴수록 어떻게 바뀌는지 보고, valid와 교차검증 점수와 나란히 놓아요.",
           "<p>트리를 늘릴수록 모델은 복잡해져요. 모델 묶음이 얼마나 복잡한지 재는 값 하나가 <b>VC 차원</b>이에요. 그 묶음이 점 몇 개까지는 라벨을 어떻게 붙여도 다 맞힐 수 있는지를 세요. 스텀프 T개를 무게 붙여 더한 모델의 VC 차원은 대략 T log T에 비례해서 커져요. 그러면 트리를 늘릴수록 과적합해야 할 것 같은데, 실제로는 train 오차가 0이 된 뒤에도 test 오차가 계속 내려가는 일이 자주 보고됐어요.</p>"
           "<p>이걸 설명하는 게 <b>마진</b>이에요. ⑨에서 SVM은 도로가 넓을수록 처음 보는 데이터에 강하다고 했어요. AdaBoost에서는 점수를 발언권의 합으로 나눈 값이 그 역할을 해요.</p>"
           "<p><code>L1 마진 = y × f(x) ÷ (α₁ + ... + α_T)</code></p>"
           "<p>−1에서 1 사이이고, 양수면 맞힌 것, 클수록 \"트리들이 많이 동의해서\" 확실히 맞힌 거예요. 처음 보는 데이터에서 틀릴 확률의 상한은 \"마진이 ρ보다 작은 train 행의 비율 + (스텀프 묶음의 복잡도 ÷ ρ) + 데이터 수로 정해지는 항\"이에요. 여기에 <b>트리 수 T가 없어요</b>. 트리를 늘려도 마진이 커지면 상한은 오히려 작아질 수 있다는 거예요.</p>"
           + K.table(["", "SVM (②⑨)", "AdaBoost"], [["점수", "θᵀx + θ₀", "α₁h₁(x) + ... + α_T h_T(x)"], ["마진", "점수 ÷ ‖θ‖ (제곱합의 제곱근, L2)", "점수 ÷ Σα (절댓값의 합, L1)"],
                                                    ["마진을 키우는 방법", "가장 작은 마진을 직접 최대로", "트리를 늘리면 커지지만, 최대까지는 아니에요"]])
           + K.out("s13") + K.out("f_margin") + K.out("s13cv")
           + "<p><b>약한 학습 가정은 사실 강한 가정이에요</b>: \"어떤 무게를 줘도 ε ≤ ½ − γ인 스텀프가 있다\"는 조건을 게임으로 풀면(무게를 고르는 쪽과 스텀프를 고르는 쪽의 시합, 폰 노이만의 미니맥스 정리), train 데이터가 스텀프들의 무게 합으로 마진 2γ 이상 깨끗이 갈린다는 것과 같은 말이 돼요. 이 데이터는 ⑫에서 ε이 0.45 넘게 ½에 다가갔어요. 깨끗이 갈리지 않는 데이터라는 신호예요.</p>"
           "<p>그래서 실제로는 트리 수 T를 교차검증으로 정하거나 learning rate를 곱해 조금씩 더하고(2쪽 ⑩의 shrinkage), 발언권의 합 Σα에 벌점을 주기도 해요(L1 규제). 무게가 큰 행(⑪)은 이상치나 라벨 오류 후보로 따로 살펴봐요.</p>"
           + K.code_table([("ys * F_train / np.cumsum(alphas)[:, None]", "트리 수마다 행별 L1 마진"), ("cross_validate(build(AdaBoostClassifier(..., n_estimators=T)), ...)", "트리 수 T마다 ⑦과 같은 교차검증")]), "s13"),
]
practice = [("SVM 앞에는 꼭 크기 맞추기", "SVM은 거리와 내적으로 판단해서, 큰 단위 열이 도로 방향을 다 정해 버려요. RBF는 특히 그래요."),
            ("데이터가 수만 건 넘으면 (SVM)", "커널 SVM은 모든 점 쌍의 커널 값을 다뤄서 건수의 제곱에서 세제곱만큼 느려져요. 곧은 경계로 충분하면 <code>LinearSVC</code>나 로지스틱을 써요."),
            ("SVM에서 확률이 필요하다", "SVM은 원래 \"어느 쪽인가\"와 경계까지 거리만 줘요. <code>probability=True</code>를 켜면 안에서 교차검증으로 확률을 맞춰 주지만 느려지고, 결과가 <code>predict</code>와 조금 다를 수 있어요."),
            ("종류가 셋 이상 (SVM)", "SVM은 원래 둘만 가르니까, 종류 쌍마다 분류기를 만들어 투표하거나(One-vs-One, <code>SVC</code> 기본) 종류마다 \"이것 대 나머지\" 분류기를 만들어요(One-vs-Rest)."),
            ("판별분석에 필요한 데이터 양", "보통 전체 건수가 열 수의 3배 이상, 가장 작은 종류의 건수가 열 수보다 많아야 공분산을 믿을 수 있어요. 이 예제는 실패 18건, 6열이라 LDA는 되지만 QDA는 빠듯해요."),
            ("LDA에 좋은 입력", "서로 닮지 않은(상관이 낮은) 열이 좋아요. 닮은 열이 있으면 공분산의 역행렬이 불안정해져요. ①의 <code>drop=\"first\"</code>가 그 이유예요."),
            ("PCA 전에", "크기 맞추기가 필수예요. 안 하면 단위가 큰 열 하나가 분산을 독차지해서 1번 주성분이 그 열 자체가 돼요."),
            ("PCA가 분류를 좋게 하나", "보장은 없어요. 분산이 큰 방향이 정답을 가르는 방향이라는 법은 없거든요. 열이 아주 많거나 서로 닮았을 때 흔들림을 줄이는 데 주로 도움이 돼요 (⑦에서 표준편차가 가장 작았어요). 늘 교차검증으로 확인해요."),
            ("그림으로 보고 싶다", "PCA 두 개로 먼저 보고, 휘어진 구조가 의심되면 t-SNE나 UMAP을 써요. 이 둘은 그림용이고, 축 값 자체에는 뜻이 없어요."),
            ("커널을 직접 만들었다", "몇백 개 점으로 커널 행렬을 만들어 <code>np.linalg.eigvalsh</code>의 가장 작은 값이 0 이상인지 봐요(⑧). 이미 있는 커널을 더하거나 곱해서 만들면 늘 커널이에요."),
            ("커널 SVM이 너무 느리다", "건수가 수만을 넘으면 <code>RBFSampler</code>나 <code>Nystroem</code>으로 열을 만들고 선형 모델을 올려요(⑩). 정확도가 조금 내려갈 수 있으니 D를 늘려 가며 교차검증으로 확인해요."),
            ("AdaBoost의 트리 수", "train 오차가 0이 됐다고 멈추지 말고, 교차검증으로 정해요(⑬). <code>learning_rate</code>를 1보다 작게 두면 트리를 더 많이 쓰는 대신 덜 흔들려요."),
            ("라벨이 지저분한 데이터에 부스팅", "AdaBoost는 크게 틀린 행에 무게를 몰아줘서 라벨 오류에 약해요(⑫의 손실 그림). 로그 손실을 쓰는 <code>GradientBoostingClassifier</code>, XGBoost, LightGBM이 더 튼튼해요."),
            ("무게가 큰 행 살펴보기", "AdaBoost가 끝까지 무게를 몰아준 행(⑪)은 라벨 오류나 예외 사례일 때가 많아요. 지우기 전에 원본을 직접 확인해요.")]
libs = [("①", "<code>OneHotEncoder(drop=\"first\")</code>", "같은 정보 열 빼기"), ("②③", "<code>SVC</code>, <code>coef_</code>, <code>n_support_</code>, <code>class_weight</code>", "SVM (scikit-learn)"),
        ("④⑤", "<code>LinearDiscriminantAnalysis</code>, <code>QuadraticDiscriminantAnalysis</code>, <code>transform</code>", "판별분석"), ("⑥", "<code>PCA</code>, <code>explained_variance_ratio_</code>, <code>components_</code>", "주성분 분석"),
        ("⑦", "<code>StratifiedKFold</code>, <code>cross_validate</code>", "교차검증"),
        ("⑧", "<code>polynomial_kernel</code>, <code>rbf_kernel</code>, <code>laplacian_kernel</code>, <code>sigmoid_kernel</code>, <code>np.linalg.eigvalsh</code>", "커널 행렬과 고윳값"),
        ("⑨", "<code>dual_coef_</code>, <code>support_vectors_</code>, <code>LeaveOneOut</code>, <code>cross_val_predict</code>", "커널 SVM의 답, leave-one-out"), ("⑩", "<code>RBFSampler</code>", "커널 근사 (scikit-learn)"),
        ("⑪", "<code>DecisionTreeClassifier(max_depth=1)</code>, <code>sample_weight</code>, <code>AdaBoostClassifier</code>", "스텀프와 AdaBoost"), ("⑫⑬", "<code>np.cumsum</code>, <code>np.cumprod</code>", "트리 수마다 점수, 상한, 마진")]
K.build(ROOT / "topics" / "model" / "ml-svm-lda-pca.html", title="SVM, 판별분석, 차원 축소", h1="SVM, 판별분석, 차원 축소: 도로, 분포, 방향으로 가르기", track="모델",
        crumb="모델 &gt; 머신러닝 기초 4/5쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-metrics-clustering.html\" style=\"color:inherit\">평가지표와 군집</a> &nbsp;|&nbsp; 다음: <a href=\"ml-learning-theory.html\" style=\"color:inherit\">학습 이론</a>",
        one_line="<b>SVM</b>은 두 무리 사이에 가장 넓은 도로를 내고, <b>LDA와 QDA</b>는 무리마다 퍼진 모양을 배워 확률로 판단하고, <b>PCA</b>는 정답 없이 데이터가 가장 많이 퍼진 방향 몇 개로 열을 줄여요. 끝으로 <b>AdaBoost</b>를 직접 짜서, 부스팅에도 SVM 같은 마진이 있다는 걸 봐요.",
        why=why, whole=whole, steps=steps, code_file="examples/ml/more_models.py", practice=practice, libs=libs, libs_note="<code>part</code>, <code>build</code>, <code>score</code>, <code>adaboost</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "로지스틱, 규제, 원-핫 열 문제"), ("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "⑦과 같은 교차검증 표"),
                   ("<a href=\"dl-tensor-training.html\">딥러닝 기초</a>", "신경망으로 넘어가기")],
        sources=["다루는 범위: 부트캠프 Day05~08의 머신러닝 기초를 넓힌 쪽이에요. 수업에서 따로 다루지 않은 모델이라, 그 Day의 예제를 그대로 이어 썼어요",
                 "함께 정리한 자료: 경희대 Pattern Recognition and Machine Learning 7강(판별분석), 9강(SVM), 11강(차원축소). 최대 마진, 서포트 벡터, 소프트 마진과 힌지 손실, 라그랑주 쌍대와 KKT, 커널 트릭과 커널 종류, 베이즈 정리, 생성 모델과 판별 모델, LDA의 가정과 피셔 기준, QDA, 차원의 저주, PCA의 순서와 고유값, 주성분 개수 고르기, PCA와 LDA 비교, MDS를 가져왔어요. 값과 그림은 모두 이 예제로 계산했어요",
                 "함께 정리한 자료: Mohri, Rostamizadeh, Talwalkar, <i>Foundations of Machine Learning</i> 2판 5장(서포트 벡터의 두 종류, leave-one-out 보장, 마진 이론), 6장(PDS 커널과 커널 행렬, RKHS, 닫힘 성질, Representer 정리, NDS 커널, 시퀀스 커널, 랜덤 푸리에 특성), 7장(약학습, AdaBoost와 무게 고치기, 경험적 오차의 상한, 좌표 하강과의 관계, VC 차원 분석, L1 마진과 마진 상한, 게임 이론으로 본 약학습 가정, L1 규제, 잡음에 약한 점). ⑧~⑬의 값은 이 예제로 계산했어요",
                 "데이터: <code>examples/ml/data/requests.csv</code> (가상의 업무 요청 100건)와 <code>split_manifest.csv</code>. 1~3쪽과 같은 예제",
                 "결과 값과 그림: <code>tools/pages/ml_more.py</code>가 <code>examples/ml/more_models.py</code>를 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/svm.html\" target=\"_blank\">Support Vector Machines</a>, <a href=\"https://scikit-learn.org/stable/modules/metrics.html\" target=\"_blank\">Pairwise metrics and kernels</a>, <a href=\"https://scikit-learn.org/stable/modules/kernel_approximation.html\" target=\"_blank\">Kernel Approximation</a>, <a href=\"https://scikit-learn.org/stable/modules/lda_qda.html\" target=\"_blank\">Linear and Quadratic Discriminant Analysis</a>, <a href=\"https://scikit-learn.org/stable/modules/decomposition.html#pca\" target=\"_blank\">PCA</a>"])
print("ok")
