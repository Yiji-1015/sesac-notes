"""모델 > 머신러닝 기초 세 쪽을 만든다. examples/ml/requests_model.py 하나가 ①~⑰로 세 쪽에 걸쳐 자란다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/ml_basics.py
"""
import sys
import warnings
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "ml"
code = S.parse((EX / "requests_model.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="requests_model.py")  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, pandas {pd.__version__}"
DATA_SRC = "데이터: <code>examples/ml/data/requests.csv</code> (가상의 업무 요청 100건)와 <code>split_manifest.csv</code> (train 60, valid 20, test 20). 기존 노트와 같은 예제"
ns = run(17)
raw = ns["raw"]


def df_rows(df):
    return [[E(str(i))] + [E(str(v)) for v in r] for i, r in zip(df.index, df.values)]


# =============================== 1쪽 그림: 모두 이 예제를 실제로 계산해서 그린다 ===============================
sys.path.insert(0, str(ROOT / "tools" / "visuals"))
from vstyle import plt, C, setup  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.metrics import accuracy_score as _ACC, f1_score as _F1  # noqa: E402
setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "ml"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/ml/{name}.png"


tr_, va_ = ns["train"], ns["valid"]
pre_ = ns["reg"].named_steps["pre"]
Xd = np.c_[np.ones(len(tr_)), np.asarray(pre_.transform(ns["X_train"]), float)]  # 맨 앞 1은 절편 자리
yd = tr_["elapsed_ms"].to_numpy(float)
FIG = {}

# ④-1 최소제곱: 입력 길이 하나만 쓴 직선과 잔차, 그리고 8열 모델의 valid 실제 vs 예측
d1 = tr_.dropna(subset=["input_chars"])
one = LinearRegression().fit(d1[["input_chars"]], d1["elapsed_ms"])
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
xs = np.linspace(d1["input_chars"].min(), d1["input_chars"].max(), 50)
for x_, y_ in zip(d1["input_chars"], d1["elapsed_ms"]):
    a.plot([x_, x_], [one.predict([[x_]])[0], y_], color=FAIL, lw=1, alpha=.55)
a.scatter(d1["input_chars"], d1["elapsed_ms"], s=22, color=C["ink"], zorder=3, label="train 요청")
a.plot(xs, one.predict(xs[:, None]), color=OK, lw=2.5, label=f"제곱 오차가 가장 작은 직선 (기울기 {one.coef_[0]:.2f} ms/글자)")
a.set(xlabel="input_chars (입력 글자 수)", ylabel="elapsed_ms (처리 시간)", title="최소제곱법: 분홍 선 길이의 제곱 합이 가장 작게")
a.legend(fontsize=9, loc="upper left")
mn, mx = va_["elapsed_ms"].min(), va_["elapsed_ms"].max()
b.plot([mn, mx], [mn, mx], color=C["gray"], ls="--", lw=1.2, label="완벽한 예측 (실제 = 예측)")
b.axhline(yd.mean(), color=C["orange"], lw=1.5, label=f"평균으로 찍기 ({yd.mean():.0f} ms)")
b.scatter(va_["elapsed_ms"], ns["reg_pred"], s=36, color=OK, zorder=3, label="8열 선형 회귀의 예측")
b.set(xlabel="실제 처리 시간 (valid)", ylabel="예측 처리 시간", title=f"valid 20건: R² {ns['reg_r2']:.3f}")
b.legend(fontsize=9, loc="upper left")
FIG["ls"] = fig_save(fig, "least_squares")
one_r2 = float(np.corrcoef(d1["input_chars"], d1["elapsed_ms"])[0, 1] ** 2)

# ④-2 경사하강: 같은 비용을 learning rate 셋으로 줄여 보기
lam_max = float(np.linalg.eigvalsh(Xd.T @ Xd / len(yd)).max())
LRS = [0.003, 0.3, 1.25]
paths = {}
for lr in LRS:
    th, J = np.zeros(Xd.shape[1]), []
    for _ in range(60):
        r = Xd @ th - yd
        J.append(float(r @ r) / (2 * len(yd)))
        th = th - lr / len(yd) * (Xd.T @ r)  # 모든 θ를 같은 r로 한꺼번에 고친다 (동시 업데이트)
    paths[lr] = (J, th)
J_best = float(((Xd @ (np.linalg.pinv(Xd) @ yd) - yd) ** 2).mean() / 2)
fig, ax = plt.subplots(figsize=(8, 3.8))
for lr, col in zip(LRS, [C["gray"], OK, FAIL]):
    ax.plot(range(60), paths[lr][0], color=col, lw=2.2, marker="o" if lr == LRS[2] else None, ms=3, label=f"learning rate {lr}")
ax.axhline(J_best, color=C["mint"], ls="--", lw=1.3, label=f"공식으로 구한 바닥 ({J_best:,.0f})")
ax.set_yscale("log")
ax.set(xlabel="반복 횟수", ylabel="비용 J (MSE ÷ 2, 로그 눈금)", title="경사하강법: 한 걸음 크기에 따라", ylim=(J_best * 0.8, 2e7))
ax.legend(fontsize=9)
FIG["gd"] = fig_save(fig, "gradient_descent")

# ④ 🔁 규제 경로: alpha를 키울 때 가중치 8개가 어떻게 움직이나
alphas = np.logspace(-1, 3, 40)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
Xs, ys_ = ns["X_train"], tr_["elapsed_ms"]
for ax_, cls, nm in [(a, Ridge, "Ridge"), (b, Lasso, "Lasso")]:
    W = np.array([ns["build"](cls(alpha=al)).fit(Xs, ys_).named_steps["model"].coef_ for al in alphas])
    for j, col in zip(range(W.shape[1]), [C["purple"], C["pink"], C["mint"], C["orange"], C["blue"], C["yellow"], C["green"], C["gray"]]):
        lw_ = 2.6 if j == 0 else 1.4
        ax_.plot(alphas, W[:, j], lw=lw_, color=col, label=pre_.get_feature_names_out()[j].split("__", 1)[1])
    ax_.axhline(0, color=C["gray"], lw=.8)
    ax_.set_xscale("log")
    ax_.set_xticks([0.1, 1, 10, 100, 1000], ["0.1", "1", "10", "100", "1000"])
    ax_.minorticks_off()
    ax_.set(xlabel="alpha (벌점 세기, 로그 눈금)", title=f"{nm}: " + ("다 같이 천천히 0 쪽으로" if nm == "Ridge" else "하나씩 정확히 0이 된다"))
a.set_ylabel("배운 가중치")
b.legend(fontsize=8, loc="upper right")
FIG["path"] = fig_save(fig, "regularization_path")

# ⑤-1 시그모이드 위의 valid 20건
zz = np.linspace(min(-4.0, float(ns["z"].min()) - 0.3), 4, 200)
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(zz, 1 / (1 + np.exp(-zz)), color=C["ink"], lw=2, label="시그모이드 1 / (1 + e^(-z))")
ax.axhline(0.5, color=C["gray"], ls="--", lw=1)
ax.axvline(0, color=C["gray"], ls="--", lw=1)
yv = ns["y_valid"].to_numpy()
ax.scatter(ns["z"][yv == 0], ns["p_fail"][yv == 0], s=48, color=OK, zorder=3, label="실제 성공")
ax.scatter(ns["z"][yv == 1], ns["p_fail"][yv == 1], s=70, color=FAIL, marker="X", zorder=3, label="실제 실패")
ax.text(0.15, 0.06, "z ≥ 0 이면 \"실패\"라고 답해요", fontsize=10, color=C["gray"])
ax.set(xlabel="점수 z (가중치를 곱해 더한 값)", ylabel="실패 확률", title="valid 20건이 시그모이드 위 어디에 있나")
ax.legend(fontsize=9, loc="upper left")
FIG["sig"] = fig_save(fig, "sigmoid_valid")

# ⑤-2 로그 손실 곡선
pp = np.linspace(0.005, 0.995, 200)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(pp, -np.log(pp), color=FAIL, lw=2.4, label="실제 실패일 때: -log(p)")
ax.plot(pp, -np.log(1 - pp), color=OK, lw=2.4, label="실제 성공일 때: -log(1 - p)")
ax.plot(pp, (1 - pp) ** 2, color=C["gray"], lw=1.6, ls="--", label="(참고) 실제 실패일 때의 제곱 오차 (1 - p)²")
for p_ in (0.9, 0.1):
    ax.scatter([p_], [-np.log(p_)], color=FAIL, zorder=3)
    ax.annotate(f"p={p_} → {-np.log(p_):.2f}", (p_, -np.log(p_)), (p_ - 0.05 if p_ > 0.5 else p_ + 0.05, -np.log(p_) + 0.9), fontsize=10, color=FAIL, arrowprops=dict(arrowstyle="->", color=FAIL))
ax.set(xlabel="모델이 준 실패 확률 p", ylabel="비용", ylim=(0, 5), title="자신 있게 틀릴수록 비용이 크게 뛴다")
ax.legend(fontsize=9, loc="upper center")
FIG["ll"] = fig_save(fig, "log_loss_curve")

# ⑥-1 KNN 경계: 숫자 두 열만 써서 k=1, 5, 31
d2 = tr_[["input_chars", "allowed_tools"]].fillna(tr_[["input_chars", "allowed_tools"]].median())
mu, sd = d2.mean(), d2.std(ddof=0)
Z2 = ((d2 - mu) / sd).to_numpy()
y2 = tr_["failed"].to_numpy()
g1, g2 = np.meshgrid(np.linspace(-2.3, 2.3, 220), np.linspace(-2.0, 2.0, 220))
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
from matplotlib.colors import ListedColormap  # noqa: E402
for ax_, k in zip(axs, (1, 5, 31)):
    m2 = KNeighborsClassifier(n_neighbors=k).fit(Z2, y2)
    gg = m2.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax_.contourf(g1, g2, gg, levels=[-.5, .5, 1.5], cmap=ListedColormap(["#ece6ff", "#ffe0ee"]))
    ax_.scatter(Z2[y2 == 0, 0], Z2[y2 == 0, 1], s=22, color=OK, label="성공")
    ax_.scatter(Z2[y2 == 1, 0], Z2[y2 == 1, 1], s=40, color=FAIL, marker="X", label="실패")
    ax_.set(title=f"k = {k}  (train 정확도 {_ACC(y2, m2.predict(Z2)):.2f})", xlabel="input_chars (크기 맞춤)")
    ax_.grid(False)
axs[0].set_ylabel("allowed_tools (크기 맞춤)")
axs[0].legend(fontsize=9, loc="lower left")
FIG["knn"] = fig_save(fig, "knn_boundaries")

# ⑥-2 k에 따른 train과 valid 점수 (8열 모델 그대로)
krange = list(range(1, 32, 2))
tr_acc, va_f1 = [], []
for k in krange:
    mk = ns["build"](KNeighborsClassifier(n_neighbors=k)).fit(ns["X_train"], ns["y_train"])
    tr_acc.append(_ACC(ns["y_train"], mk.predict(ns["X_train"])))
    va_f1.append(_F1(ns["y_valid"], mk.predict(ns["X_valid"])))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(krange, tr_acc, color=C["ink"], marker="o", ms=4, lw=2, label="train 정확도 (본 데이터)")
ax.plot(krange, va_f1, color=FAIL, marker="o", ms=4, lw=2, label="valid F1 (처음 보는 데이터)")
ax.text(1.3, 0.12, "← 외움 (과적합)", fontsize=10, color=C["gray"])
ax.text(21, 0.12, "흔한 답만 (과소적합) →", fontsize=10, color=C["gray"])
ax.set(xlabel="k (볼 이웃 수)", ylabel="점수", ylim=(-0.03, 1.05), title="k가 작으면 외우고, 크면 뭉갠다")
ax.legend(fontsize=9, loc="center right")
FIG["kcurve"] = fig_save(fig, "knn_k_curve")

# =============================== 1쪽: 회귀와 분류 ===============================
outs = {}
outs["f_ls"] = S.images("입력 길이 하나로 그은 직선과, 8열 모델의 valid 예측", [(FIG["ls"], "왼쪽: train, 오른쪽: valid")],
                        [f"왼쪽은 설명을 위해 <code>input_chars</code> 한 열만 쓴 직선이에요. 직선과 점 사이 분홍 선이 오차(잔차)이고, 최소제곱법은 이 길이의 제곱 합이 가장 작은 직선을 골라요. 그래도 점이 넓게 흩어져서, 이 한 열로 설명되는 건 train에서도 R² {one_r2:.2f}뿐이에요.",
                         "오른쪽 점이 회색 점선(완벽한 예측)에 모이지 않고 주황 선(평균) 근처에 납작하게 깔려 있어요. 모델이 입력으로 시간 차이를 거의 못 가려낸다는 뜻이고, R²가 0 아래로 내려간 이유예요."])
it_03 = next(i for i, j in enumerate(paths[0.3][0]) if j <= J_best * 1.05)   # 0.3이 바닥 근처(5% 안)에 처음 닿은 반복
S.claim(it_03 <= 8 and paths[0.003][0][-1] > 2 * J_best and paths[1.25][0][-1] > paths[1.25][0][0] and 0.3 < 2 / lam_max < 1.25,
        "④ 경사하강: 0.3은 몇 번 만에 바닥, 0.003은 한참 위, 1.25는 발산")
outs["f_gd"] = S.images("같은 8열 회귀를 경사하강법으로 풀면", [(FIG["gd"], "반복할수록 비용이 줄어드는 모습")],
                        [f"0.3은 {it_03}번 만에 공식으로 구한 바닥(점선)에 거의 닿아요. 0.003은 60번을 가도 한참 위예요. 1.25는 걸음이 너무 커서 매번 바닥을 지나쳐 반대편 더 높은 곳에 떨어지고, 그래서 비용이 오히려 계속 커져요 (발산).",
                         f"이 데이터에서는 learning rate가 약 {2 / lam_max:.2f}를 넘으면 발산해요. 그래서 실제로는 0.001, 0.01, 0.1, 1처럼 열 배씩 바꿔 시험하고, 이 그림처럼 반복마다 비용이 줄어드는지 그려서 확인해요."])
outs["f_path"] = S.images("alpha를 0.1부터 1000까지 키우면 (가중치 8개)", [(FIG["path"], "굵은 선이 input_chars")],
                          ["Ridge는 선들이 함께 0으로 다가가지만 끝까지 닿지 않아요. Lasso는 선이 하나씩 0에 붙고, 마지막까지 남는 게 가장 쓸모 있는 <code>input_chars</code>예요."])
fn_z = np.sort(ns["z"][(ns["y_valid"].to_numpy() == 1) & (ns["z"] < 0)])        # 놓친 실패의 점수
pair = next(((a_ + b_) / 2 for a_, b_ in zip(fn_z, fn_z[1:]) if b_ - a_ < 0.05), None)  # 그림에서 겹쳐 보이는 두 점
overlap = f" (z ≈ {f'{pair:.1f}'.replace('-', '−')}의 두 개는 겹쳐 있어요)" if pair is not None else ""
outs["f_sig"] = S.images("⑤의 로지스틱 회귀가 valid 20건에 준 점수와 확률", [(FIG["sig"], "가로: 점수 z, 세로: 실패 확률")],
                         ["모든 점이 곡선 위에 있어요. 확률은 점수 z를 시그모이드에 넣은 값일 뿐이라서요.",
                          f"오른쪽 위(z ≥ 0)에 있는 점만 \"실패\"라고 답해요. 분홍 X가 왼쪽에 {len(fn_z)}개 있죠{overlap}. 그게 ⑤에서 놓친 실패 {ns['cm'][1][0]}건이에요. 보라 점도 오른쪽에 {ns['cm'][0][1]}개 있어요 (헛경보 {ns['cm'][0][1]}건)."])
outs["f_ll"] = S.images("로그 손실과 제곱 오차", [(FIG["ll"], "실제 실패인 요청 하나의 비용")],
                        ["실제 실패인데 p를 0.1로 줬다면 로그 손실은 2.30이지만, 제곱 오차는 0.81에서 멈춰요. 제곱 오차는 아무리 크게 틀려도 1을 넘지 않아서, 자신 있게 틀린 걸 충분히 벌주지 못해요."])
outs["f_knn"] = S.images("train 60건으로 그린 KNN의 판단 지도 (숫자 두 열만 사용)", [(FIG["knn"], "분홍 바탕: \"실패\"라고 답하는 곳")],
                         ["그림으로 보려고 <code>input_chars</code>와 <code>allowed_tools</code> 두 열만 쓴 모델이에요. ⑥의 실제 모델(8열)과는 다르지만 k의 효과는 같아요.",
                          "k=1은 실패 점 근처를 좁은 띠와 조각으로 감싸서 경계가 들쭉날쭉해요. train도 거의 다 맞혀요. 100%가 안 되는 건, 두 열만 보면 같은 자리인데 답이 다른 요청이 있어서예요 (8열 모델에서는 {K1_TRAIN}).",
                          "k=31은 오른쪽 아래 귀퉁이만 빼고 분홍 바탕이 사라져서, 거의 어디서든 \"성공\"이라고 답해요."])
best_k = krange[int(np.argmax(va_f1))]
zero_from = next(k for i, k in enumerate(krange) if all(v == 0 for v in va_f1[i:]))   # 이 k부터는 valid F1이 계속 0
S.claim(tr_acc[-1] < tr_acc[0], "⑥ k가 커질수록 train 정확도가 내려간다")
outs["f_k"] = S.images("k를 1부터 31까지 (8열 모델, ⑥과 같은 조건)", [(FIG["kcurve"], "")],
                       [f"검은 선(본 데이터)은 k=1에서 {tr_acc[0]:.1f}이고 k가 커질수록 내려가요. 분홍 선(처음 보는 데이터)은 k={best_k}에서 가장 높고, 오르내리다 k={zero_from}부터는 0에 붙어요. valid 20건에서 실패 한두 건이 맞고 틀리는 차이라 들쭉날쭉해요.",
                        "두 선의 간격이 클수록 외운 정도가 커요. 가장 좋은 k는 두 극단 사이 어딘가에 있고, 교차검증으로 찾아요 (2쪽)."])
for _k in ("f_ls", "f_gd", "f_path", "f_sig", "f_ll", "f_knn", "f_k"):
    outs[_k]["chart"] = True  # 그래프는 폭을 꽉 채워 보여 주기 (topics/steps.css의 .imgrow.chart)
outs["data"] = S.table("requests.csv 앞 5줄", ["request_id"] + list(raw.columns), df_rows(raw.head(5)),
                       ["<code>failed</code>(실패했나, 1이면 실패)와 <code>elapsed_ms</code>(처리 시간)를 맞히는 게 목표예요.",
                        "<code>elapsed_ms</code>, <code>retry_count</code>, <code>status</code>는 요청을 <b>처리한 뒤에야</b> 알 수 있는 값이라 입력으로 쓰면 안 돼요. 미래를 보고 맞히는 셈이라서요 (데이터 누수)."])
n1 = run(1)
outs["s1"] = S.box("① 실행 결과", [(f"train {len(n1['train'])}건, valid {len(n1['valid'])}건, test {len(n1['test'])}건", 1),
                                   (f"실패 비율 → train {n1['y_train'].mean():.2f}, valid {n1['y_valid'].mean():.2f}, test {n1['test']['failed'].mean():.2f}", 2),
                                   f"빈 칸 → input_chars {int(raw['input_chars'].isna().sum())}개, allowed_tools {int(raw['allowed_tools'].isna().sum())}개"],
                   ["나누는 방법을 파일(manifest)로 고정해 둬서, 몇 번을 돌려도 같은 행이 같은 쪽에 들어가요.", "세 쪽 모두 실패가 30%로 비슷하게 나뉘었어요. 빈 칸은 ③에서 채워요."])
outs["s2"] = S.box("② 기준선", [(f"분류: 언제나 \"성공\"이라고 답하기 → 정확도 {ns['base_acc']:.2f}", 1), (f"회귀: 언제나 train 평균 시간이라고 답하기 → MAE {ns['base_mae']:.0f} ms", 2)],
                   [f"아무것도 배우지 않은 모델이 정확도 {ns['base_acc']:.2f}이에요. 실패가 30%뿐이라서 \"다 성공\"이라고만 해도 70%는 맞거든요. 그러니 정확도 {ns['base_acc']:.2f}은 잘한 게 아니에요.", f"평균으로만 찍어도 평균 {ns['base_mae']:.0f} ms 틀려요. 모델은 이보다 나아야 의미가 있어요."])
pre = ns["build"](ns["LogisticRegression"]()).named_steps["pre"].fit(ns["X_train"])
outs["s3"] = S.box("③ 전처리만 따로 돌려 보면", [(f"입력 4열 → 전처리 뒤 {len(pre.get_feature_names_out())}열", 1), f"{list(pre.get_feature_names_out())}"],
                   ["숫자 두 열은 빈 칸을 중앙값으로 채우고 평균 0, 표준편차 1로 맞췄어요. 글자 두 열은 값마다 0/1 열로 펼쳤어요 (원-핫). 모델은 숫자만 받으니까요."])
short = lambda n: n.split("__", 1)[1]  # noqa: E731  "num__input_chars" → "input_chars"
coef = ns["reg_coef"]
top = coef.reindex(coef.abs().sort_values(ascending=False).index)
S.claim(ns["reg_rmse"] > ns["reg_mae"] and ns["reg_r2"] < 0, "④ RMSE가 MAE보다 크고, R²가 0보다 작다")
outs["s4"] = S.box("④ 선형 회귀로 처리 시간 예측 (valid)", [f"MAE → {ns['reg_mae']:.0f} ms  (기준선 {ns['base_mae']:.0f} ms)", (f"RMSE → {ns['reg_rmse']:.0f} ms", 1), (f"R² → {ns['reg_r2']:.3f}", 2, "bad"), "",
                                                         (f"배운 가중치: 절편 {ns['reg'].named_steps['model'].intercept_:.1f}, " + ", ".join(f"{short(i)} {v:+.1f}" for i, v in top.head(4).items()), 3)],
                   ["RMSE가 MAE보다 커요. 제곱을 거치면서 크게 빗나간 몇 건이 더 무겁게 잡혔다는 뜻이에요.",
                    "R²는 \"평균으로 찍는 것보다 얼마나 나은가\"예요. 1이면 완벽, 0이면 평균과 같고, <b>0보다 작으면 평균보다 못해요</b>. 이 네 입력만으로는 처리 시간을 거의 설명하지 못한다는 뜻이에요. 모델을 바꾸기 전에 \"입력이 충분한가\"부터 의심해야 하는 신호예요.",
                    f"숫자 열은 ③에서 표준편차 1로 맞췄으니, <code>input_chars</code> {coef['num__input_chars']:+.1f}은 \"입력 길이가 표준편차 하나만큼 길면 {coef['num__input_chars']:.0f} ms 더 걸린다고 배웠다\"는 뜻이에요. 절편은 숫자 열이 평균이고 원-핫 열이 모두 0일 때 식이 내는 값이에요."])
reg_line = "reg = build(LinearRegression())"
penal = []
for label, cls, a in [("선형 회귀", None, None), ("Ridge", "Ridge", 10), ("Ridge", "Ridge", 100), ("Lasso", "Lasso", 10), ("Lasso", "Lasso", 50)]:
    n = ns if cls is None else run(4, [(reg_line, f"from sklearn.linear_model import {cls}\nreg = build({cls}(alpha={a}))")])
    c = pd.Series(n["reg"].named_steps["model"].coef_, index=coef.index)
    penal.append([label, "-" if a is None else a, f"{c['num__input_chars']:.1f}", f"{c['num__allowed_tools']:.1f}", f"{c['cat__task_type_qa']:.1f}",
                  f"<b>{int((c.abs() < 1e-9).sum())}</b> / {len(c)}", f"{n['reg_mae']:.0f}", f3(n["reg_r2"])])
_zero = [int(r[5].split("</b>")[0].replace("<b>", "")) for r in penal]   # 줄마다 0인 가중치 수
S.claim(_zero[1] == _zero[2] == 0 and _zero[4] == len(coef) - 1 and float(penal[4][2]) != 0, "④ Ridge는 0이 없고, Lasso alpha 50은 input_chars 하나만 남는다")
S.claim(max(float(r[7]) for r in penal) < 0.05, "④ 규제를 줘도 R²가 나아지지 않는다")
outs["t4"] = S.table("가중치에 벌점을 주면 (valid)", ["모델", "alpha", "input_chars", "allowed_tools", "task_type_qa", "0인 가중치", "MAE", "R²"], penal,
                     ["Ridge는 alpha를 키울수록 <b>모든</b> 가중치를 조금씩 0 쪽으로 줄여요. 하나도 정확히 0이 되지는 않아요.",
                      "Lasso는 덜 쓸모 있는 가중치부터 <b>정확히 0</b>으로 만들어요. alpha 50에서는 <code>input_chars</code> 하나만 남았어요. 열을 골라 주는 효과예요.",
                      "그래도 R²는 나아지지 않았어요. 이 데이터의 문제는 \"가중치가 너무 커서\"(과적합)가 아니라 \"설명할 입력이 없어서\"라서요. 규제는 과적합을 고치는 도구예요. alpha는 데이터를 여러 번 나눠 재는 교차검증으로 골라요 (2쪽)."])
cm = ns["cm"]
z, pf, odds = ns["z"], ns["p_fail"], ns["odds"]
from sklearn.metrics import log_loss  # noqa: E402
rate = float(ns["y_train"].mean())
base_loss = log_loss(ns["y_valid"], np.full(len(ns["y_valid"]), rate))
first = ns["X_valid"].index[0]
odds_top = odds.reindex((np.log(odds)).abs().sort_values(ascending=False).index)
outs["s5"] = S.box("⑤ 로지스틱 회귀로 실패 예측 (valid 20건)", ["혼동 행렬 (행: 실제, 열: 예측)", f"            예측 성공  예측 실패", f"실제 성공      {cm[0][0]:>3}       {cm[0][1]:>3}", (f"실제 실패      {cm[1][0]:>3}       {cm[1][1]:>3}", 1), "",
                                                      (f"accuracy {ns['scores']['accuracy']:.2f}, precision {ns['scores']['precision']:.2f}, recall {ns['scores']['recall']:.2f}, F1 {ns['scores']['f1']:.2f}", 2), "",
                                                      (f"{first}: 점수 z {z[0]:.3f} → 시그모이드 {pf[0]:.3f} → 0.5보다 작아서 \"성공\"", 3, "newf"),
                                                      (f"로그 손실 → {ns['loss']:.3f}  (누구에게나 train 실패 비율 {rate:.2f}만 말하면 {base_loss:.3f})", 4, "bad"),
                                                      ("odds 배수: " + ", ".join(f"{short(i)} ×{v:.2f}" for i, v in odds_top.head(4).items()), 5, "newf")],
                   [f"실제 실패 {cm[1].sum()}건 중 {cm[1][1]}건만 잡았어요. {cm[1][0]}건은 놓쳤어요.", f"정확도는 기준선({ns['base_acc']:.2f})보다 낮지만, 기준선은 실패를 <b>하나도</b> 못 잡아요(F1 0). 실패를 찾는 게 목적이면 정확도보다 recall(실패 중 잡은 비율)과 F1을 봐야 해요.",
                    f"z는 음수라 확률이 0.5 아래예요. 직접 계산한 <code>p_fail</code>은 모델의 확률 함수 <code>predict_proba</code>가 내는 값과 같아요. 모델 안에서도 이 계산을 해요.",
                    "로그 손실이 \"모두에게 같은 확률 말하기\"보다 <b>커요</b>. 확률을 틀린 방향으로 자신 있게 낸 요청이 있다는 뜻이에요. 이 모델의 확률은 실패와 성공의 순서조차 잘 못 매긴다는 신호예요. 확률의 순서가 얼마나 맞는지는 3쪽에서 ROC-AUC라는 점수로 따로 재요.",
                    f"<code>input_chars</code>가 표준편차 하나만큼 길면 실패 odds가 {odds['num__input_chars']:.2f}배가 된다고 배웠어요. 1보다 작으면 실패 쪽을 줄이는 열이에요 (<code>priority_normal</code> ×{odds['cat__priority_normal']:.2f})."])
S.claim(ns["scores"]["accuracy"] < ns["base_acc"] and ns["loss"] > base_loss, "⑤ 정확도는 기준선보다 낮고, 로그 손실은 같은 확률 말하기보다 크다")
c_rows = []
for Cv in (0.01, 0.1, 1, 100):  # C는 그림 색 이름이라 Cv
    n = ns if Cv == 1 else run(5, [("logit = build(LogisticRegression(max_iter=1000))", f"logit = build(LogisticRegression(max_iter=1000, C={Cv}))")])
    w = n["logit"].named_steps["model"].coef_[0]
    c_rows.append([f"<b>{Cv}</b> (지금)" if Cv == 1 else Cv, f3(np.abs(w).sum()), f"{n['scores']['f1']:.2f}", f"{n['scores']['recall']:.2f}", f"{n['scores']['accuracy']:.2f}"])
S.claim(c_rows[0][2] == c_rows[1][2] == "0.00" and c_rows[0][4] == c_rows[1][4] == f"{ns['base_acc']:.2f}" and abs(float(c_rows[3][2]) - float(c_rows[2][2])) < 0.1, "⑤ C 0.1 이하는 F1 0, C 100도 크게 나아지지 않는다")
outs["t5"] = S.table("C를 바꾸면 (valid)", ["C", "가중치 절댓값 합", "F1", "recall", "accuracy"], c_rows,
                     ["C는 규제 세기의 <b>역수</b>예요 (C = 1/λ). 작을수록 벌점이 세서 가중치가 0 근처로 눌려요.",
                      "C 0.1 이하에서는 가중치가 너무 작아져서 모든 요청에 실패 확률 0.5 미만을 줘요. 그래서 기준선처럼 \"다 성공\"이라고 답하고 F1이 0이에요.",
                      "C를 100으로 풀어도 크게 나아지지 않아요. 기본값 1은 무난한 출발점이고, 고를 때는 교차검증으로 비교해요."])
from sklearn.metrics import accuracy_score as ACC, f1_score as F1  # noqa: E402
knn_line = "knn_scaled = build(KNeighborsClassifier(n_neighbors=5))"
k_rows = []
for label, arg in [("1", "n_neighbors=1"), ("3", "n_neighbors=3"), ("5 (지금)", "n_neighbors=5"), ("9", "n_neighbors=9"), ("31", "n_neighbors=31"),
                   ("5, 맨해튼 거리", "n_neighbors=5, p=1"), ("5, 가까울수록 큰 표", "n_neighbors=5, weights=\"distance\"")]:
    n = ns if arg == "n_neighbors=5" else run(6, [(knn_line, f"knn_scaled = build(KNeighborsClassifier({arg}))")])
    m = n["knn_scaled"]
    k_rows.append([f"<b>{label}</b>" if label.startswith("5 (") else label, f3(ACC(n["y_train"], m.predict(n["X_train"]))), f"{F1(n['y_valid'], m.predict(n['X_valid'])):.2f}", f"{ACC(n['y_valid'], m.predict(n['X_valid'])):.2f}"])
K1_TRAIN = k_rows[0][1]
outs["f_knn"]["notes"] = [n_.replace("{K1_TRAIN}", K1_TRAIN) for n_ in outs["f_knn"]["notes"]]
S.claim(k_rows[3][2] == k_rows[4][2] == "0.00" and float(k_rows[4][3]) == round(ns["base_acc"], 2), "⑥ k=9, 31은 F1 0이고 k=31 정확도는 기준선과 같다")
S.claim(max(k_rows[:5], key=lambda r: float(r[2]))[0] == "3", "⑥ valid에서는 k=3이 가장 좋다")
outs["t6"] = S.table("k와 거리를 바꾸면 (스케일 있음)", ["k와 설정", "train 정확도", "valid F1", "valid 정확도"], k_rows,
                     [f"k=1이면 train 정확도가 {k_rows[0][1]}이에요. train의 모든 요청은 자기 자신이 가장 가까운 이웃이라서, 외운 답을 그대로 말해요. \"가까울수록 큰 표\"도 같은 이유로 {k_rows[6][1]}이에요.",
                      f"k=9와 31에서는 실패를 하나도 맞히지 못해요 (F1 0). k=31이면 이웃 31개의 다수가 늘 성공이라 모든 요청에 \"성공\"이라고 답해요. 정확도 {k_rows[4][3]}이 기준선과 똑같은 이유예요. k가 작으면 과적합, 크면 과소적합이에요.",
                      "valid에서는 k=3이 가장 좋지만, valid가 20건이라 한두 건 차이예요. k도 교차검증으로 골라요 (2쪽)."])
ks, kr = F1(ns["y_valid"], ns["knn_scaled"].predict(ns["X_valid"])), F1(ns["y_valid"], ns["knn_raw"].predict(ns["X_valid"]))
outs["s6"] = S.box("⑥ KNN (k=5), 스케일 있음과 없음 (valid F1)", [f"스케일 있음 → {ks:.2f}", f"스케일 없음 → {kr:.2f}", f"input_chars 범위 {raw['input_chars'].min():.0f}~{raw['input_chars'].max():.0f}, allowed_tools 범위 {raw['allowed_tools'].min():.0f}~{raw['allowed_tools'].max():.0f}"],
                   ["스케일이 없으면 거리가 거의 <code>input_chars</code>(수백~수천)로만 정해지고, <code>allowed_tools</code>(1~5)는 무시돼요. 이 데이터에서는 그게 오히려 점수가 높게 나왔어요. 입력 중 <code>input_chars</code>가 가장 쓸모 있어서일 수도, valid가 20건뿐이라 우연일 수도 있어요. 한 번의 점수로 결론 내리지 않는 이유예요 (2쪽의 교차검증)."])
S.claim(kr > ks, "⑥ 스케일이 없는 쪽이 점수가 높다")
_b = ns["board"]
S.claim(_b["accuracy"].idxmax() == "dummy" and _b["f1"].idxmax() == "logistic", "⑦ 정확도는 dummy, F1은 logistic이 1등")
outs["s7"] = S.table("⑦ 같은 train, 같은 valid로 비교", ["model", "F1", "recall", "accuracy"], df_rows(ns["board"]),
                     ["정확도만 보면 dummy가 1등이에요. F1로 보면 logistic이 가장 나아요. <b>무엇을 기준으로 고르느냐</b>가 결과를 바꿔요.", "모두 같은 나누기, 같은 전처리 함수를 썼어요. 조건이 다르면 비교가 아니에요."])
flow1 = [[1, "읽고 나누기", "60 / 20 / 20", "누수 열 빼기"], [2, "기준선", "Dummy", f"정확도 {ns['base_acc']:.2f}"], [3, "Pipeline", "채우기, 맞추기", "원-핫"],
         [4, "회귀", "선형, 제곱 오차", "가중치, 규제"], [5, "분류", "로지스틱, 시그모이드", "확률, odds"], [6, "KNN", "거리, 다수결", "k, 스케일"], [7, "비교", "같은 조건", "기준이 결과를 바꿈"]]
S.save(ROOT / "topics" / "model" / "ml-regression-classification-data.js", {"code": [c for c in code if c["s"] <= 7], "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow1, "env": ENV})

why1 = f"""
<p>"이 요청은 실패할까?", "처리에 얼마나 걸릴까?"처럼, 지난 기록으로 앞으로를 맞히고 싶을 때가 있어요. 사람이 규칙을 일일이 정하는 대신 <b>기록에서 규칙을 배우게</b> 하는 게 머신러닝이에요. 표(행과 열) 데이터에서는 보통 두 가지를 해요.</p>
{K.table(["종류", "맞히는 것", "예"], [["<b>회귀</b>", "숫자", "처리 시간 1,520 ms"], ["<b>분류</b>", "정해진 종류 중 하나", "실패 / 성공"]])}
<p>예제는 업무 요청 100건이에요.</p>
{K.out("data")}
<p>모델을 만들기 전에 꼭 정해야 할 게 있어요. <b>무엇과 비교해서 잘했다고 할지</b>(기준선), <b>무엇으로 잴지</b>(지표), <b>어떤 데이터로 잴지</b>(나누기)예요. 이게 없으면 "정확도 70%"가 좋은 건지 나쁜 건지 알 수 없어요.</p>
<p>이 예제처럼 정답(<code>failed</code>, <code>elapsed_ms</code>)이 붙은 기록으로 배우는 걸 <b>지도 학습</b>이라고 해요. 정답 없이 데이터의 구조를 찾는 <b>비지도 학습</b>도 있고(3쪽의 K-Means), 정답 대신 행동의 결과로 받는 보상으로 배우는 <b>강화 학습</b>도 있어요. 이 쪽은 지도 학습만 다뤄요.</p>
<p>그리고 "기록에서 규칙을 배운다"는 말은 사실 네 가지를 정하는 일이에요. 이 쪽에서 쓰는 세 모델은 이 네 칸을 서로 다르게 채워요. 표의 <b>가중치</b>는 열마다 하나씩 곱하는 숫자, <b>k</b>는 몇 개의 이웃을 볼지예요. 낯선 말(MSE, 로그 손실)은 ④⑤에서 하나씩 풀어요.</p>
{K.table(["정할 것", "선형 회귀 (④)", "로지스틱 회귀 (⑤)", "KNN (⑥)"],
         [["<b>모양</b>: 입력으로 답을 어떻게 계산하나", "열마다 가중치를 곱해 더하기", "곱해 더한 점수를 0~1 확률로 바꾸기", "가장 가까운 k개의 다수결"],
          ["<b>비용</b>: 얼마나 틀렸나를 어떤 숫자로", "제곱 오차의 평균 (MSE)", "로그 손실", "없음"],
          ["<b>찾기</b>: 비용이 가장 작은 가중치를 어떻게", "공식으로 한 번에 계산", "조금씩 고쳐 가며 찾기", "찾지 않고 train을 저장만"],
          ["<b>예측</b>: 새 입력에", "같은 식에 넣기", "확률이 0.5 이상이면 실패", "저장한 train과 거리 재기"]])}
<p>모양과 비용이 정해지면 "찾기"는 도구가 해 줘요. 그래서 모델을 고른다는 건 대부분 <b>어떤 모양, 어떤 비용</b>을 쓸지 고르는 일이에요.</p>
"""
whole1 = """<p>나누고, 기준선을 정하고, 전처리를 묶고, 회귀와 분류를 해 보고, 같은 조건으로 비교해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p><code>requests_model.py</code> 하나가 이 쪽(①~⑦)에서 시작해 <a href="ml-trees-cv.html">2쪽</a>(⑧~⑫), <a href="ml-metrics-clustering.html">3쪽</a>(⑬~⑰)까지 자라요.</p>"""
steps1 = [
    K.step(1, "데이터 읽고 나누기", "요청을 받을 때 아는 네 열만 입력으로 고르고, 행을 train, valid, test로 나눠요.",
           "<p><b>train</b>으로 배우고, <b>valid</b>로 모델을 고르고, <b>test</b>는 마지막에 딱 한 번 성적을 매기는 데 써요. 같은 데이터로 배우고 채점하면, 답을 외운 모델도 100점이 나와요.</p>"
           "<p>얼마씩 나눌지는 데이터 양에 따라 달라요. 흔히 70/15/15쯤 나누고, 데이터가 아주 많으면 90/5/5처럼 valid와 test를 작게 둬도 충분해요. 이 예제는 100건뿐이라 채점용을 넉넉히 둔 60/20/20이에요.</p>"
           + K.code_table([("pd.read_csv(...).set_index(\"request_id\")", "표를 읽고, 요청 번호를 행 이름으로"), ("manifest[manifest[\"split\"] == name]", "이 쪽에 들어갈 행 번호만"), ("raw.loc[ids]", "그 번호의 행만 꺼내기")]), "s1"),
    K.step(2, "기준선 만들기", "아무것도 배우지 않는 모델의 점수를 먼저 재요.",
           K.code_table([("DummyClassifier(strategy=\"most_frequent\")", "가장 많은 답(성공)만 말하는 분류기"), ("DummyRegressor(strategy=\"mean\")", "train 평균만 말하는 회귀기"),
                         ("mean_absolute_error", "예측과 실제 차이의 평균 (MAE). 작을수록 좋아요.")]), "s2"),
    K.step(3, "전처리를 Pipeline으로 묶기", "빈 칸 채우기, 크기 맞추기, 글자를 숫자로 바꾸기를 모델과 한 덩어리로 묶어요.",
           "<p>전처리를 따로 하면 실수로 valid나 test까지 보고 기준(중앙값, 평균)을 정하기 쉬워요. 그러면 채점할 데이터를 미리 엿본 셈이에요. Pipeline으로 묶으면 <code>fit</code>할 때 <b>train만</b> 보고 기준을 정해요.</p>"
           + K.code_table([("SimpleImputer(strategy=\"median\")", "빈 칸을 중앙값으로"), ("StandardScaler()", "평균 0, 표준편차 1로"), ("OneHotEncoder(handle_unknown=\"ignore\")", "글자 값마다 0/1 열. 처음 보는 값은 모두 0"),
                           ("ColumnTransformer([...])", "열마다 다른 전처리"), ("Pipeline([(\"pre\", ...), (\"model\", ...)])", "전처리와 모델을 한 덩어리로"), ("clone(model)", "설정만 같은 새 모델. 같은 모델 객체를 여러 번 쓰지 않으려고")]), "s3"),
    K.step(4, "회귀: 처리 시간 예측", "선형 회귀로 처리 시간을 맞히고, 기준선과 비교해요.",
           "<p>선형 회귀의 모양은 <code>예측 = θ₀ + θ₁x₁ + θ₂x₂ + … + θₙxₙ</code>예요. x는 ③에서 만든 8개 열, θ(세타)는 열마다 하나씩 배우는 가중치, θ₀은 절편(모든 x가 0일 때의 값)이에요. \"회귀\"라는 이름은 키가 아주 큰 부모의 자녀 키가 평균 쪽으로 \"되돌아간다(regression)\"는 19세기 관찰에서 왔어요. 지금은 그냥 \"숫자를 맞히는 모델\"이라는 뜻으로 써요.</p>"
           "<p><b>왜 제곱 오차를 줄이나?</b> 차이를 그냥 더하면 +300과 −300이 서로 지워져요. 제곱하면 부호가 사라지고, 크게 빗나간 건 더 크게 벌을 받아요. 그리고 제곱 오차는 가중치에 대해 매끈한 그릇 모양이라, 바닥(가장 작은 점)을 공식 하나로 바로 구할 수 있어요. 이걸 <b>최소제곱법</b>이라고 하고, <code>LinearRegression</code>이 하는 일이 이거예요.</p>"
           + K.out("f_ls")
           + "<p>그 공식은 <code>θ = (XᵀX)⁻¹ Xᵀ y</code>예요 (<b>정규방정식</b>). X는 train 60행 × 9열 표(맨 앞에 절편 자리로 1을 붙인 열 + ③의 8열), y는 처리 시간 60개예요. Xᵀ는 X의 행과 열을 바꾼 것(전치), ⁻¹는 역행렬(곱하면 \"1\"이 되는 짝, 나눗셈 역할)이에요.</p>"
           + f"<p>⚠️ 그런데 이 예제에서는 XᵀX의 <b>역행렬이 없어요</b>. <code>task_type</code>의 원-핫 세 열은 어느 행에서나 더하면 1이라서, 절편 자리의 1 열과 같은 정보예요. <code>priority</code> 세 열도 마찬가지예요. 그래서 9열 중 서로 다른 정보는 {int(np.linalg.matrix_rank(Xd))}개뿐이고, <b>예측을 똑같이 내는 θ가 무수히 많아요</b>. <code>LinearRegression</code>은 그중 하나(가중치 크기가 가장 작은 것)를 골라 줘요. 그래서 아래 출력의 <code>task_type_qa {coef['cat__task_type_qa']:+.1f}</code>은 \"qa면 {coef['cat__task_type_qa']:.0f} ms 더\"가 아니라 \"다른 task_type보다 상대적으로 더\"로 읽어야 해요.</p>"
           + "<p><b>경사하강법</b>: 열이 수만 개면 정규방정식의 역행렬 계산(열 수의 세제곱에 비례)이 너무 무거워요. 그때는 그릇 바닥을 향해 조금씩 걸어 내려가요. 한 걸음은 <code>θⱼ ← θⱼ − α × (1/m) × Σ (예측 − 실제) × xⱼ</code>예요. m은 행 수(60), Σ는 모든 행에 대해 더하기, α(알파)는 한 걸음 크기(<b>learning rate</b>)예요. \"예측 − 실제\"가 크고 xⱼ도 큰 열일수록 θⱼ를 많이 고쳐요.</p>"
           "<p>⚠️ 한 걸음 안에서는 θ를 <b>모두 같은 예측으로</b> 고쳐야 해요(동시 업데이트). θ₀을 먼저 고치고 그 새 θ₀으로 다시 예측해서 θ₁을 고치면, 그릇의 기울기가 아닌 엉뚱한 방향으로 걷게 돼요. 신경망은 정규방정식 같은 공식이 없어서 늘 이 방법을 써요 (<a href=\"dl-tensor-training.html\">딥러닝 기초</a>).</p>"
           + K.out("f_gd")
           + K.table(["지표", "계산", "읽는 법"], [["MAE", "|실제 − 예측|의 평균", "평균 몇 ms 빗나가나. 큰 오차 몇 건에 덜 흔들려요."], ["MSE", "(실제 − 예측)²의 평균", "학습이 줄이는 값. 단위가 ms²라 읽기 어려워요."],
                                                   ["RMSE", "MSE의 제곱근", "다시 ms 단위. 큰 오차가 많을수록 MAE보다 커져요."], ["R²", "1 − (모델의 제곱 오차 합 ÷ 평균으로 찍은 제곱 오차 합)", "평균보다 얼마나 나은가. 0이면 평균과 같아요."]])
           + K.code_table([("LinearRegression()", "최소제곱법으로 가중치를 한 번에 구하는 기본 회귀"), ("r2_score", "결정 계수. 평균보다 얼마나 나은지"), ("root_mean_squared_error", "RMSE"),
                           ("reg.named_steps[\"model\"].coef_", "Pipeline 안의 모델이 배운 가중치. 순서는 <code>get_feature_names_out()</code>의 열 순서와 같아요.")])
           + "<p><b>규제</b>: 모델이 train에만 있는 우연한 잡음까지 외우면, train 점수는 높은데 처음 보는 데이터에서는 무너져요. 이걸 <b>과적합</b>이라고 해요. 선형 모델은 과적합할 때 가중치가 커지기 쉬워서, 비용에 \"가중치가 클수록 벌점\"을 더하기도 해요. 벌점이 가중치 제곱의 합(λ × Σθⱼ²)이면 <b>Ridge</b>(L2 규제), 절댓값의 합(λ × Σ|θⱼ|)이면 <b>Lasso</b>(L1 규제)예요. 절편 θ₀에는 벌점을 주지 않아요. 벌점 세기 λ(람다)를 scikit-learn에서는 <code>alpha</code>라고 불러요.</p>"
           + K.try_("LinearRegression을 Ridge나 Lasso로 바꾸고 alpha를 키우면 가중치는?", "t4", after=K.out("f_path")), "s4"),
    K.step(5, "분류: 실패 여부 예측", "로지스틱 회귀로 실패를 맞히고, 혼동 행렬과 여러 지표로 봐요.",
           "<p><b>왜 선형 회귀로 분류하지 않나?</b> 실패를 1, 성공을 0으로 두고 선형 회귀를 하면 예측이 1.4나 −0.3처럼 0~1 밖으로 나가요. 확률로 읽을 수 없어요. 로지스틱 회귀는 선형 회귀와 같은 방법으로 점수 <code>z = θ₀ + θ₁x₁ + …</code>를 만든 뒤, <b>시그모이드</b> <code>1 / (1 + e^(−z))</code>에 넣어요. e는 약 2.718인 상수예요. 시그모이드는 어떤 실수든 0과 1 사이로 접어 주는 S자 곡선이에요. z가 아주 크면 e^(−z)가 0에 가까워져 1이 되고, 아주 작으면 분모가 커져 0이 돼요.</p>"
           "<p>z가 0이면 e^0 = 1이라 확률이 정확히 1/2이에요. 그래서 \"확률 0.5 이상이면 실패\"는 \"z가 0 이상이면 실패\"와 같고, z = 0이 되는 곳이 실패와 성공을 가르는 경계예요. z가 가중치를 곱해 더한 값이니 경계는 곧은 직선(평면)이에요. 로지스틱 회귀를 <b>선형</b> 분류기라고 부르는 이유예요.</p>"
           + K.out("f_sig")
           + "<p><b>왜 비용이 제곱 오차가 아니라 로그 손실인가?</b> 시그모이드를 거친 확률에 제곱 오차를 쓰면, 비용이 그릇 모양이 아니라 울퉁불퉁해져서 조금씩 고쳐 가다 엉뚱한 골짜기에 멈출 수 있어요. 로그 손실은 정답 쪽 확률이 p일 때 <code>−log(p)</code>예요(log는 자연로그). 실제 실패면 −log(p), 실제 성공이면 −log(1 − p)이고, 둘을 한 줄로 쓰면 <code>−[y log(p) + (1 − y) log(1 − p)]</code>예요(y는 실패면 1, 성공이면 0). 정답에 0.9를 줬으면 0.11, 0.1을 줬으면 2.30으로, <b>자신 있게 틀릴수록</b> 비용이 크게 뛰어요. 이 비용은 그릇 모양이라 바닥을 놓치지 않아요.</p>"
           "<p>이 식은 아무렇게나 고른 게 아니에요. 로그 손실을 가장 작게 하는 가중치는 \"train의 정답들이 실제로 나올 확률\"을 가장 크게 하는 가중치와 같아요(<b>최대우도 추정</b>). 종류가 여럿일 때 같은 생각을 넓힌 게 교차 엔트로피예요. 그리고 이 비용을 미분하면 경사하강의 한 걸음이 ④와 <b>똑같은 모양</b> <code>(예측 − 실제) × xⱼ</code>이 나와요. 예측이 θᵀx 대신 시그모이드를 거친 확률이라는 것만 달라요.</p>"
           + K.out("f_ll")
           + "<p><b>가중치 읽기</b>: 실패 확률 p에 대해 p ÷ (1 − p)를 <b>odds</b>라고 해요. 0.2면 odds 0.25(1:4), 0.5면 1이에요. 로지스틱 회귀의 점수 z는 사실 log(odds)라서, 어떤 열이 1 커지면 odds가 e^θ배가 돼요. 확률이 몇 %p 오르는지는 출발점에 따라 달라요. 같은 odds 2배라도 10%에서는 약 18%로(+8%p), 90%에서는 약 95%로(+5%p) 올라요. 그래서 가중치는 \"odds 몇 배\"로 읽어요.</p>"
           + K.table(["지표", "뜻"], [["accuracy", "전체 중 맞힌 비율"], ["precision", "\"실패\"라고 한 것 중 진짜 실패 비율"], ["recall", "진짜 실패 중 잡아낸 비율"], ["F1", "precision과 recall의 조화 평균"]])
           + K.code_table([("LogisticRegression(max_iter=1000)", "확률을 내는 선형 분류기. 가중치는 정규방정식 같은 공식이 없어서 조금씩 고쳐 가며 찾아요. 최대 1000번까지. 찾는 방법(<code>solver</code>)의 기본값 <code>lbfgs</code>는 경사하강보다 영리하게 걸음 크기와 방향을 정해서, learning rate를 따로 고르지 않아도 돼요."),
                           ("confusion_matrix(y, pred, labels=[0, 1])", "실제와 예측의 2×2 표"),
                           ("logit.predict(X)", "실패(1) 또는 성공(0). 안에서는 확률이 0.5 이상인지 봐요. 확률 자체는 <code>predict_proba(X)</code>로 받아요."),
                           ("logit.decision_function(X)", "시그모이드에 넣기 전 점수 z"), ("1 / (1 + np.exp(-z))", "시그모이드. <code>np.exp</code>는 e의 거듭제곱"), ("log_loss(y, p)", "로그 손실의 평균"),
                           ("np.exp(...coef_[0])", "가중치 θ를 odds 배수 e^θ로. <code>coef_</code>는 종류별로 한 줄씩이라 이진 분류면 <code>[0]</code> 한 줄")])
           + "<p><code>LogisticRegression</code>에는 처음부터 Ridge와 같은 벌점이 들어 있어요. 세기는 <code>C</code>로 정해요.</p>"
           + K.try_("C를 0.01부터 100까지 바꾸면?", "t5"), "s5"),
    K.step(6, "KNN: 거리로 판단하는 모델", "가장 가까운 요청 5개의 결과로 투표해요. 거리를 쓰니 숫자 크기에 민감해요.",
           "<p>KNN은 가중치를 배우지 않아요. <code>fit</code>은 train을 저장하기만 하고, 예측할 때마다 새 요청과 저장한 요청 <b>전부</b>의 거리를 재서 가장 가까운 k개를 골라요. 그래서 배우는 건 빠르고 예측은 느려요. train이 백만 건이면 요청 하나에 거리를 백만 번 재요.</p>"
           "<p>거리는 보통 직선거리(<b>유클리드</b>, 열마다 차이를 제곱해 더하고 제곱근)를 써요. 차이의 절댓값을 그냥 더하는 <b>맨해튼</b> 거리(바둑판 길을 따라 걷는 거리)도 있어요. 둘을 묶은 일반식이 <b>민코프스키</b> 거리 <code>(Σ |차이|ᵖ)^(1/p)</code>로, p=1이면 맨해튼, p=2면 유클리드예요. 어느 쪽이든 <b>숫자가 큰 열이 거리를 거의 다 정해서</b>, ③의 크기 맞추기(스케일링)가 중요해요.</p>"
           "<p>k가 정하는 것: k가 작으면 바로 옆 한두 건에 휘둘려서 train의 잡음까지 따라가요(④에서 본 <b>과적합</b>). k가 크면 멀리 있는 요청까지 섞여서 결국 가장 흔한 답만 말해요(<b>과소적합</b>, 배울 걸 못 배운 상태). 동률을 피하려고 이진 분류에서는 k를 홀수로 두곤 해요.</p>"
           + K.out("f_knn")
           + "<p>이 줄다리기를 <b>편향과 분산</b>으로 설명하기도 해요. 처음 보는 데이터에서의 오차는 <code>편향² + 분산 + 줄일 수 없는 잡음</code>으로 나뉘어요.</p>"
           + K.table(["", "k 작음 (과적합 쪽)", "k 큼 (과소적합 쪽)"],
                     [["<b>편향</b>: 평균적으로 정답에서 얼마나 비껴 있나", "낮음. 데이터를 세밀하게 따라가요", "높음. 다 뭉개서 늘 비슷한 답"],
                      ["<b>분산</b>: train이 조금 바뀌면 답이 얼마나 흔들리나", "높음. 이웃 하나만 바뀌어도 답이 바뀌어요", "낮음"],
                      ["모델의 복잡도", "높음", "낮음"]])
           + "<p>선형 회귀에서는 열을 늘리면 복잡도가 오르고, 규제(alpha)를 세게 하면 내려가요. 모델마다 손잡이는 달라도, 그 손잡이로 이 둘 사이 균형점을 찾는 건 같아요.</p>"
           + K.code_table([("KNeighborsClassifier(n_neighbors=5)", "가까운 5개의 다수결. 기본 거리는 유클리드"), ("build(..., scale=False)", "크기 맞추기를 빼고 비교"),
                           ("p=1", "맨해튼 거리로 (기본 <code>p=2</code>는 유클리드)"), ("weights=\"distance\"", "가까운 이웃일수록 표를 크게")])
           + K.try_("k를 1부터 31까지, 거리 재는 법을 바꾸면?", "t6", after=K.out("f_k")), "s6"),
    K.step(7, "같은 조건으로 비교하기", "모델만 바꾸고 나머지(나누기, 전처리, 지표)는 같게 해서 표로 비교해요.",
           K.code_table([("evaluate({이름: 모델})", "모델마다 같은 train으로 배우고 같은 valid로 잰 표")]), "s7"),
]
practice1 = [("데이터가 적다", "valid 20건이면 한두 건만 바뀌어도 점수가 크게 흔들려요. 한 번의 점수로 고르지 말고 교차검증을 써요 (<a href=\"ml-trees-cv.html\">2쪽</a>)."),
             ("R²가 음수", "모델보다 입력부터 의심해요. 처리 시간을 정말 설명할 값(서버 상태, 요청 내용 등)이 입력에 있나요?"),
             ("실패가 드물다", "정확도 대신 recall, F1을 보고, 필요하면 확률 기준(0.5)을 바꿔요 (<a href=\"ml-metrics-clustering.html\">3쪽</a>)."),
             ("처리 후에 생기는 열", "<code>status</code>처럼 결과가 이미 담긴 열을 입력에 넣으면 점수가 비현실적으로 높게 나와요. 실제로 쓸 때는 그 값이 없어서 무너져요."),
             ("같은 정보를 담은 열이 둘", "예: 크기(m²)와 크기(평). ④의 원-핫 열처럼 가중치가 하나로 정해지지 않고 크게 흔들려요. 하나를 빼거나(원-핫이면 <code>OneHotEncoder(drop=\"first\")</code>로 한 열씩 빼기) Ridge를 써요. 예측만 쓸 거면 문제없지만, 가중치를 해석할 때는 꼭 확인해요."),
             ("learning rate를 고를 때", "0.001, 0.01, 0.1, 1처럼 열 배씩 바꿔 보고, ④의 그림처럼 반복마다 비용을 그려요. 비용이 오르면 줄이고, 너무 천천히 내려가면 키워요. \"직전보다 줄어든 양이 0.001보다 작으면 멈춘다\"처럼 멈출 조건도 정해요."),
             ("규제를 쓸 때", "크기 맞추기를 먼저 해요. 벌점은 가중치 크기에 붙는데, 단위가 큰 열은 가중치가 작게 나와서 벌을 덜 받아요. 단위가 공정하지 않으면 벌점도 공정하지 않아요."),
             ("종류가 셋 이상", "<code>LogisticRegression</code>은 종류마다 점수 z를 하나씩 만들고, e^z를 모두 더한 값으로 나눠 합이 1인 확률로 바꿔요 (softmax). 비용은 정답 종류의 −log(확률)이에요(교차 엔트로피). 따로 설정할 건 없고, <code>coef_</code>가 종류 수만큼 줄이 생겨요. 옛 방식인 One-vs-Rest(종류마다 \"이것 대 나머지\" 이진 분류기를 따로 만들고 가장 높은 확률을 고르기)는 종류 사이 경계가 애매한 곳에서 약해요. 혼동 행렬도 종류 수만큼 커져서(3종류면 3×3), 대각선이 맞힌 수예요."),
             ("KNN이 너무 느리다", "정확히 가장 가까운 k개 대신 \"거의 가장 가까운\" k개를 빨리 찾는 근사 검색을 써요. 비슷한 점끼리 같은 칸에 떨어지게 하는 해싱(LSH), 벡터를 짧은 코드로 압축하는 양자화(product quantization), 이웃끼리 그래프로 이어 두고 따라가는 HNSW 등이 있어요. HNSW는 <a href=\"../backend/embedding.html\">Embedding과 Vector Search</a>에서 실제로 써요."),
             ("열이 아주 많다 (KNN)", "열이 많아질수록 데이터가 공간에 듬성듬성해지고, 모든 점이 서로 비슷하게 멀어져서 \"가까운 이웃\"이 의미를 잃어요 (차원의 저주). 여러 열을 정보가 많은 몇 개로 줄이거나(차원 축소) 다른 모델을 써요.")]
libs1 = [("①", "<code>pd.read_csv()</code>, <code>.loc[]</code>", "읽기, 나누기 (pandas)"), ("②", "<code>DummyClassifier</code>, <code>DummyRegressor</code>", "기준선 (scikit-learn)"),
         ("③", "<code>Pipeline</code>, <code>ColumnTransformer</code>, <code>SimpleImputer</code>, <code>StandardScaler</code>, <code>OneHotEncoder</code>", "전처리 묶기"),
         ("④", "<code>LinearRegression</code>, <code>r2_score</code>, <code>root_mean_squared_error</code>, <code>coef_</code>", "회귀와 지표. 🔁에서 <code>Ridge</code>, <code>Lasso</code>"),
         ("⑤", "<code>LogisticRegression</code>, <code>decision_function</code>, <code>log_loss</code>, <code>confusion_matrix</code>, <code>f1_score</code>", "분류와 지표"), ("⑥", "<code>KNeighborsClassifier</code>", "KNN")]
K.build(ROOT / "topics" / "model" / "ml-regression-classification.html", title="회귀와 분류", h1="회귀와 분류: 표 데이터로 예측하기", track="모델", crumb="모델 &gt; 머신러닝 기초 1/5쪽 &nbsp;|&nbsp; 다음: <a href=\"ml-trees-cv.html\" style=\"color:inherit\">트리와 교차검증</a>",
        one_line="지난 기록에서 규칙을 배워 <b>숫자를 맞히면 회귀</b>, <b>종류를 맞히면 분류</b>예요. 모델보다 먼저 <b>기준선, 지표, 나누기</b>를 정해야 점수가 의미를 가져요.",
        why=why1, whole=whole1, steps=steps1, code_file="examples/ml/requests_model.py", practice=practice1, libs=libs1, libs_note="<code>part</code>, <code>build</code>, <code>evaluate</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "다른 모델과 믿을 만한 비교"), ("<a href=\"ml-metrics-clustering.html\">평가지표, 특성 설계, 군집</a>", "확률 기준, 비용, 최종 평가")],
        sources=["다루는 범위: Day05~06 (2026-10-23, 26) 문제 정의, 전처리 Pipeline, 회귀, 분류, KNN, 모델 비교",
                 "함께 정리한 자료: 경희대 Pattern Recognition and Machine Learning 4~6강 (KNN, 선형회귀, 로지스틱회귀). 지도 학습의 틀, 나누는 비율, 최소제곱법과 정규방정식, 경사하강법과 learning rate, 규제, 시그모이드와 로그 손실, 최대우도, odds, softmax, 거리와 k, 편향과 분산을 가져왔어요. 값과 그림은 모두 이 페이지 예제로 다시 계산했어요",
                 "그림: <code>tools/pages/ml_basics.py</code>가 이 예제를 실제로 계산해서 그린 것 (<code>topics/model/img/ml/</code>)",
                 DATA_SRC, "결과 값: <code>tools/pages/ml_basics.py</code>가 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/compose.html\" target=\"_blank\">Pipelines and composite estimators</a>, <a href=\"https://scikit-learn.org/stable/common_pitfalls.html\" target=\"_blank\">Common pitfalls</a>, <a href=\"https://scikit-learn.org/stable/modules/model_evaluation.html\" target=\"_blank\">Model evaluation</a>"])

# =============================== 2쪽: 트리, 앙상블, 교차검증 ===============================
outs = {}
from sklearn.tree import plot_tree, DecisionTreeClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import cross_validate as _CVAL  # noqa: E402
tm = ns["tree"].named_steps["model"]
tt = tm.tree_
tnames = [short(n) for n in ns["tree"].named_steps["pre"].get_feature_names_out()]
L0, R0 = tt.children_left[0], tt.children_right[0]
n0, nl, nr = int(tt.n_node_samples[0]), int(tt.n_node_samples[L0]), int(tt.n_node_samples[R0])
g0, gl, gr = float(tt.impurity[0]), float(tt.impurity[L0]), float(tt.impurity[R0])
g_after = (nl * gl + nr * gr) / n0
fl, fr = float(tt.value[L0][0][1]), float(tt.value[R0][0][1])  # 각 칸의 실패 비율
ent = lambda p: 0.0 if p in (0, 1) else float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))  # noqa: E731
p0 = float(tt.value[0][0][1])   # 뿌리의 실패 비율
S.claim("input_chars" in tnames[tt.feature[0]] and fr == 1.0 and gr == 0.0, "⑧ 첫 질문은 input_chars이고, 넘는 쪽은 모두 실패")
outs["s8g"] = S.box("⑧의 트리가 고른 첫 질문을 직접 계산하면", [
    (f"뿌리: train {n0}건, 실패 비율 {tt.value[0][0][1]:.2f} → Gini 1 − ({tt.value[0][0][0]:.2f}² + {tt.value[0][0][1]:.2f}²) = {g0:.3f}", 1),
    (f"질문: {tnames[tt.feature[0]]} ≤ {tt.threshold[0]:.1f} ?", 2, "newf"),
    (f"  예   → {nl}건, 실패 {fl:.2f} → Gini {gl:.3f}", 3), (f"  아니오 → {nr}건, 실패 {fr:.2f} → Gini {gr:.3f}", 3),
    (f"나눈 뒤 Gini (건수로 가중 평균) = ({nl}×{gl:.3f} + {nr}×{gr:.3f}) ÷ {n0} = {g_after:.3f}  → {g0 - g_after:.3f} 줄었어요", 4),
    (f"같은 질문을 엔트로피로 재면 {ent(p0):.3f} → {(nl * ent(fl) + nr * ent(fr)) / n0:.3f} (정보 이득 {ent(p0) - (nl * ent(fl) + nr * ent(fr)) / n0:.3f})", 5)],
    ["실패와 성공이 반반이면 Gini가 가장 크고(0.5), 한쪽만 있으면 0이에요. 30%면 0.42예요.",
     f"트리는 모든 열과 모든 자를 자리를 다 시도해 보고, 불순도가 가장 많이 줄어드는 질문 하나를 골라요. 입력 글자가 {int(tt.threshold[0])}자를 넘는 요청은 train에서 모두 실패였어요.",
     f"오른쪽 {nr}건은 모두 실패라 더 나눌 게 없어요(Gini 0, 끝 노드). 왼쪽 {nl}건은 다시 같은 방법으로 다음 질문을 찾아요.",
     "건수가 많은 칸이 더 무겁게 들어가요. 한 칸만 깨끗하고 나머지가 엉망인 질문이 좋은 점수를 받지 않게요.",
     "엔트로피로 재도 같은 방향이에요. 정보 이득은 \"질문 하나로 불확실성이 얼마나 줄었나\"예요."])
fig, ax = plt.subplots(figsize=(12, 5.2))
plot_tree(tm, feature_names=tnames, class_names=["성공", "실패"], filled=True, rounded=True, impurity=True, proportion=False, fontsize=9, ax=ax)
ax.set_title("⑧의 깊이 3 트리 (train 60건)")
FIG["tree"] = fig_save(fig, "tree_depth3")
pp2 = np.linspace(0, 1, 201)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(pp2, 1 - pp2 ** 2 - (1 - pp2) ** 2, color=OK, lw=2.4, label="Gini 1 - (p² + (1-p)²)")
ax.plot(pp2, [ent(p) for p in pp2], color=FAIL, lw=2.4, label="엔트로피 (bit)")
ax.plot(pp2, 1 - np.maximum(pp2, 1 - pp2), color=C["gray"], lw=1.4, ls="--", label="오분류율 1 - max(p, 1-p)")
ax.scatter([0.3], [g0], color=OK, zorder=3)
ax.annotate(f"뿌리 (p=0.30, Gini {g0:.2f})", (0.3, g0), (0.38, 0.12), fontsize=10, color=OK, arrowprops=dict(arrowstyle="->", color=OK))
ax.set(xlabel="한 칸 안의 실패 비율 p", ylabel="불순도", title="반반일수록 불순하고, 한쪽뿐이면 0")
ax.legend(fontsize=9, loc="upper right")
FIG["imp"] = fig_save(fig, "impurity_curves")
outs["f_tree"] = S.images("⑧의 트리를 그대로 그리면", [(FIG["tree"], "색이 진할수록 한쪽 답이 많은 칸")],
                          ["상자마다 질문, Gini, 건수(<code>samples</code>), 답별 건수(<code>value</code> = [성공, 실패])가 있어요. 맨 위가 뿌리, 맨 아래가 끝 노드예요. 끝 노드끼리는 겹치지 않아서, 어떤 요청이든 정확히 하나의 끝 노드에 떨어져요.",
                           "질문은 위에서부터 \"가장 많이 줄이는\" 순서예요. 그래서 위쪽에 자주 나오는 열이 판단에 많이 쓰인 열이에요 (⑪의 중요도)."])
outs["f_imp"] = S.images("불순도를 재는 세 가지 방법", [(FIG["imp"], "")],
                         ["셋 다 p=0.5에서 가장 크고 0과 1에서 0이에요. scikit-learn 기본은 Gini이고 <code>criterion=\"entropy\"</code>로 바꿀 수 있어요. 둘은 거의 같은 트리를 만들어요.",
                          "오분류율은 모양이 뾰족해서, 조금 나아진 질문을 잘 구별하지 못해요. 그래서 나눌 질문을 고를 때는 잘 쓰지 않아요."])
ccp_rows = []
for a in (0, 0.02, 0.03, 0.05):
    n = ns if a == 0 else run(8, [("deep = build(DecisionTreeClassifier(random_state=0), scale=False)", f"deep = build(DecisionTreeClassifier(random_state=0, ccp_alpha={a}), scale=False)")])
    m = n["deep"]
    ccp_rows.append([f"<b>{a}</b> (지금)" if a == 0 else a, m.named_steps["model"].get_n_leaves(), m.named_steps["model"].get_depth(), f3(_ACC(n["y_train"], m.predict(n["X_train"]))), f"{_F1(n['y_valid'], m.predict(n['X_valid'])):.2f}"])
S.claim(float(ccp_rows[-1][3]) < float(ccp_rows[0][3]) == 1.0 and float(ccp_rows[-1][4]) < float(ccp_rows[0][4]), "⑧ 가지치기하면 train 정확도가 1에서 내려오고 valid F1도 내려간다")
outs["t8"] = S.table("깊이 제한 없는 트리를 ccp_alpha로 가지치기하면", ["ccp_alpha", "끝 노드 수", "깊이", "train 정확도", "valid F1"], ccp_rows,
                     [f"alpha가 커질수록 \"끝 노드 하나 늘리는 값\"이 비싸져서, 덜 쓸모 있는 가지부터 잘려요. 끝 노드 {ccp_rows[0][1]}개가 {ccp_rows[-1][1]}개까지 줄었어요.",
                      f"train 정확도는 {ccp_rows[0][3]}에서 내려와요. 외운 걸 덜어내는 거예요. 이 valid 20건에서는 F1도 같이 내려갔지만, 20건이라 한 건 차이로도 흔들려요. alpha도 교차검증으로 골라요 (⑫)."])
dev_, cv_ = ns["dev"], ns["cv"]
rf_rows, rf_curve = [], []
for ne in (1, 5, 20, 200):
    sc = _CVAL(ns["build"](RandomForestClassifier(n_estimators=ne, max_depth=4, random_state=0), scale=False), dev_[ns["FEATURES"]], dev_["failed"], cv=cv_, scoring="f1")["test_score"]
    rf_rows.append([f"<b>{ne}</b> (지금)" if ne == 200 else ne, ", ".join(f3(x) for x in sc), f3(sc.mean()), f"<b>{f3(sc.std())}</b>"])
    rf_curve.append((ne, sc.mean(), sc.std()))
rf_sd = {ne: sd for ne, _, sd in rf_curve}
S.claim(rf_sd[200] < rf_sd[1] and min(rf_sd[5], rf_sd[20]) > rf_sd[1] and abs(rf_curve[0][1] - rf_curve[-1][1]) < 0.05,
        "⑨ 트리 1개보다 200개가 덜 흔들리고, 5개와 20개는 더 흔들리며, 평균은 비슷하다")
outs["t9"] = S.table("트리 개수를 바꾸면 (5-fold 교차검증 F1, ⑫와 같은 나누기)", ["트리 수", "다섯 번의 점수", "평균", "표준편차"], rf_rows,
                     [f"트리가 1개면 다섯 번의 점수가 크게 흔들려요(표준편차 {rf_sd[1]:.2f}). 200개면 {rf_sd[200]:.2f}로 줄어요. 평균은 비슷한데 <b>흔들림</b>이 줄었어요. 배깅이 하는 일이 이거예요: 편향은 그대로 두고 분산을 줄여요. 5개, 20개에서 오히려 더 흔들린 건, 트리가 적으면 어떤 트리가 뽑혔느냐는 운에 아직 크게 좌우되기 때문이에요.",
                      "트리를 늘려서 과적합이 생기지는 않아요. 계산만 늘어요. 보통 수백 개에서 점수가 더 안 변하는 지점을 써요."])
rng_ = np.random.default_rng(0)
boot_u = len(set(rng_.integers(0, len(ns["train"]), len(ns["train"])))) / len(ns["train"])
d1g = ns["train"].dropna(subset=["input_chars"]).sort_values("input_chars")
gbr = GradientBoostingRegressor(n_estimators=50, max_depth=1, learning_rate=0.3, random_state=0).fit(d1g[["input_chars"]], d1g["elapsed_ms"])
stg = list(gbr.staged_predict(d1g[["input_chars"]]))
fig, axs = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
for ax_, m_ in zip(axs, (1, 3, 10, 50)):
    pr = stg[m_ - 1]
    for x_, y_, p_ in zip(d1g["input_chars"], d1g["elapsed_ms"], pr):
        ax_.plot([x_, x_], [p_, y_], color=FAIL, lw=.8, alpha=.5)
    ax_.scatter(d1g["input_chars"], d1g["elapsed_ms"], s=12, color=C["ink"], zorder=3)
    ax_.step(d1g["input_chars"], pr, where="mid", color=OK, lw=2.2)
    ax_.set(title=f"트리 {m_}개 (MSE {((pr - d1g['elapsed_ms']) ** 2).mean():,.0f})", xlabel="input_chars")
axs[0].set_ylabel("elapsed_ms")
FIG["gb"] = fig_save(fig, "boosting_stages")
outs["f_gb"] = S.images("부스팅이 잔차를 이어서 고치는 모습 (설명용: 입력 길이 하나로 처리 시간 맞히기)", [(FIG["gb"], "보라 계단: 지금까지 더한 예측, 분홍 선: 남은 잔차")],
                        ["트리 하나는 질문이 하나뿐인 아주 작은 트리(깊이 1)예요. 첫 트리는 \"짧은 요청 대 긴 요청\" 두 칸으로만 나눠요.",
                         "다음 트리는 원래 정답이 아니라 <b>남은 잔차</b>(분홍 선)를 맞히도록 배우고, 그 결과에 learning rate 0.3을 곱해 더해요. 그래서 계단이 점점 잘게 쪼개지고, MSE가 줄어요.",
                         "50개쯤 가면 계단이 train 점의 잡음까지 따라가기 시작해요. 그래서 부스팅은 트리 수와 learning rate를 교차검증으로 함께 정하고, valid 점수가 나빠지면 멈춰요(조기 종료)."])
xg_rows = []
for lr in (0.01, 0.1, 1.0):
    n = ns if lr == 0.1 else run(12, [("learning_rate=0.1, random_state=0)", f"learning_rate={lr}, random_state=0)")])
    sc = n["cv_scores"]["xgboost"]
    xg_rows.append([f"<b>{lr}</b> (지금)" if lr == 0.1 else lr, ", ".join(f3(x) for x in sc), f3(sc.mean())])
_xm = {(0.1 if isinstance(r[0], str) else r[0]): float(r[2]) for r in xg_rows}   # learning_rate → 평균
S.claim(max(_xm, key=_xm.get) == 0.1 and max(_xm.values()) - min(_xm.values()) < 0.03, "⑩ learning_rate 0.1이 가장 높고, 차이는 0.03 안쪽")
outs["t10"] = S.table("XGBoost의 learning_rate를 바꾸면 (트리 200개, 교차검증 F1)", ["learning_rate", "다섯 번의 점수", "평균"], xg_rows,
                      ["0.01은 200개를 더해도 덜 배운 상태라 평균이 조금 낮아요. 1.0은 트리 하나하나가 잔차를 한 번에 다 고치려 해서, train의 잡음까지 따라가 다시 조금 낮아져요. 0.1이 가장 높았어요. 다만 차이가 0.03 안쪽이라 크지 않아요.",
                       "learning rate를 줄이면 트리 수를 늘려야 해요. \"작게, 많이\"가 보통 더 일반화가 잘 돼요 (shrinkage)."])
for _k in ("f_tree", "f_imp", "f_gb"):
    outs[_k]["chart"] = True
_cv3 = {m: ns["cv_scores"][m].mean() for m in ("logistic", "forest", "xgboost")}
S.claim(ns["board2"].loc[["logistic", "forest", "xgboost"], "f1"].idxmax() == "forest" and max(_cv3, key=_cv3.get) == "xgboost", "valid 한 번은 forest, 교차검증은 xgboost가 1등")
outs["why"] = S.table("1쪽 ⑦의 valid 점수 vs 같은 모델의 5-fold 교차검증 (F1)", ["model", "valid 20건 한 번", "교차검증 다섯 번", "평균"],
                      [[m, f3(ns["board2"].loc[m, "f1"]) if m in ns["board2"].index else "-", ", ".join(f3(x) for x in ns["cv_scores"][m]), f"<b>{f3(ns['cv_scores'][m].mean())}</b>"] for m in ["logistic", "forest", "xgboost"]],
                      [f"valid 한 번으로는 forest가 가장 좋아 보였는데, 다섯 번 나눠 재면 xgboost가 가장 좋아요. 그리고 같은 모델도 나누는 방법에 따라 {ns['cv_scores']['logistic'].min():.0f}에서 {ns['cv_scores']['logistic'].max():.1f}까지 흔들려요.", "데이터가 적을수록 \"한 번 잰 점수\"는 믿기 어려워요. 이 쪽에서는 표 데이터에 강한 트리 모델들과, 점수를 믿을 만하게 재는 교차검증을 다뤄요."])
ts = ns["tree_scores"]
S.claim(ts["제한 없음"][0] == 1.0 and ts["제한 없음"][1] > ts["max_depth=3"][1], "⑧ 깊이 제한이 없으면 train 100%, valid F1은 오히려 높다")
outs["s8"] = S.table("⑧ 결정 트리의 깊이", ["설정", "train 정확도", "valid F1"], [[k, f3(v[0]), f3(v[1])] for k, v in ts.items()],
                     ["깊이 제한이 없으면 train을 <b>100%</b> 맞혀요. train 60건을 다 외운 거예요. 외운 모델은 새 데이터에서 무너지기 쉬워서(과적합), 보통 깊이를 제한해요.", "이 데이터에서는 깊은 트리의 valid F1이 오히려 높게 나왔는데, valid가 20건이라 우연일 수 있어요. 교차검증(⑫)으로 다시 봐야 해요."])
outs["s10"] = S.table("⑩ 네 모델을 같은 valid로", ["model", "F1", "recall", "accuracy"], df_rows(ns["board2"]), ["valid 한 번의 결과예요. ⑫에서 교차검증으로 다시 비교해요."])
imp = ns["importance"]
S.claim(imp.index[0] == "num__input_chars", "⑪ input_chars가 가장 중요")
outs["s11"] = S.table("⑪ 랜덤 포레스트의 특성 중요도", ["전처리 뒤 열", "중요도"], [[f"<code>{E(i)}</code>", f3(v)] for i, v in imp.items()],
                      ["<code>input_chars</code>(입력 길이)가 판단에 가장 많이 쓰였어요. 1쪽 ⑥에서 KNN이 이 열만 본 게 오히려 잘 맞은 것과 같은 이야기예요.", "중요도는 \"모델이 많이 썼다\"는 뜻이지 \"원인이다\"는 뜻이 아니에요."])
outs["s12"] = S.table("⑫ 5-fold 교차검증 F1 (train+valid 80건)", ["model", "다섯 번의 점수", "평균", "표준편차"],
                      [[m, ", ".join(f3(x) for x in v), f"<b>{f3(v.mean())}</b>", f3(v.std())] for m, v in ns["cv_scores"].items() if m != "logistic+log1p"],
                      ["80건을 다섯 덩어리로 나눠, 넷으로 배우고 하나로 재기를 다섯 번 해요. 모든 행이 한 번씩 채점에 쓰여요.", "평균과 함께 표준편차(흔들림)도 봐요. 평균 차이가 흔들림보다 작으면 \"더 낫다\"고 말하기 어려워요."])
flow2 = [[8, "결정 트리", "질문을 이어 가기", "깊이 = 과적합"], [9, "랜덤 포레스트", "트리 여러 개 투표", "흔들림 줄이기"], [10, "XGBoost", "틀린 걸 이어서 고치기", "부스팅"], [11, "중요도", "어느 열을 썼나", "원인은 아님"], [12, "교차검증", "다섯 번 재기", "평균과 흔들림"]]
S.save(ROOT / "topics" / "model" / "ml-trees-cv-data.js", {"code": [c for c in code if c["s"] <= 12], "names": S.names(code), "range": [8, 12], "outs": outs, "flow": flow2, "env": ENV})
why2 = f"""<p><a href="ml-regression-classification.html">1쪽</a>에서는 선형 모델과 KNN을 valid 20건으로 비교했어요. 그런데 valid 한 번의 점수가 얼마나 믿을 만할까요? 1쪽의 모델과 이 쪽의 트리 모델들을 두 방법으로 재 봤어요. 하나는 1쪽처럼 valid 20건으로 한 번, 다른 하나는 <b>교차검증</b>이에요. 교차검증은 train+valid 80건을 다섯 덩어리로 나눠, 넷으로 배우고 남은 하나로 재기를 다섯 번 하는 방법이에요(5-fold, ⑫에서 자세히).</p>
{K.out("why")}"""
whole2 = """<p>트리 하나 → 트리 여럿의 투표(포레스트) → 틀린 걸 고쳐 가는 트리들(부스팅) → 어느 열을 썼나 → 믿을 만하게 재기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>1쪽의 코드(①~⑦)에 이어서 같은 파일이 자라요. 오른쪽 코드에서 1쪽 부분은 접혀 있어요.</p>"""
steps2 = [
    K.step(8, "결정 트리", f"\"입력 길이가 {int(tt.threshold[0])}자 이하인가?\" 같은 예/아니오 질문을 이어 가며 답을 정해요.",
           "<p>트리는 스무고개예요. 맨 위 <b>뿌리 노드</b>에서 질문 하나로 데이터를 둘로 나누고, 나뉜 <b>중간 노드</b>마다 또 질문을 해요. 더 묻지 않는 맨 아래 칸이 <b>끝 노드</b>(leaf)이고, 끝 노드에 모인 train의 다수 답(회귀면 평균)이 예측이에요.</p>"
           "<p>트리는 사람이 읽을 수 있는 규칙을 만들고, 정규분포 같은 통계 가정이 필요 없어요. 그리고 \"몇 이하인가\"만 묻기 때문에 숫자의 크기 차이에 영향을 받지 않아서 크기 맞추기가 필요 없어요 (<code>scale=False</code>).</p>"
           "<p><b>어떤 질문을 고르나?</b> 나눈 뒤 각 칸이 한 가지 답으로 \"깨끗해지는\" 질문이 좋아요. 깨끗하지 않은 정도를 <b>불순도</b>라고 하고, 보통 두 가지로 재요. 칸 안의 답 비율을 p₁, p₂, …라고 하면</p>"
           + K.table(["불순도", "식", "뜻"], [["<b>Gini</b>", "1 − Σ pₖ²", "칸에서 무작위로 하나 뽑아 무작위로 답을 붙이면 틀릴 확률"],
                                              ["<b>엔트로피</b>", "−Σ pₖ log₂ pₖ", "답을 알기 위해 평균 몇 번의 예/아니오(bit)가 더 필요한가. 1쪽 로그 손실과 같은 집안이에요"]])
           + "<p>나누기 전 불순도에서 나눈 뒤 불순도(칸 크기로 가중 평균)를 뺀 값이 <b>정보 이득</b>이에요. 트리는 이게 가장 큰 질문을 하나씩 골라요. 실제로 계산해 보면 이래요.</p>"
           + K.out("s8g") + K.out("f_imp")
           + K.table(["방법", "불순도", "나누는 방식", "회귀"], [["<b>CART</b> (scikit-learn이 쓰는 것)", "Gini", "언제나 둘로", "가능 (불순도 대신 제곱 오차)"],
                                                                ["C4.5", "정보 이득 비율", "값마다 여러 갈래", "불가"], ["CHAID", "카이제곱 검정 (두 범주가 서로 관련 있는지 재는 통계 검정)", "여러 갈래", "불가"]])
           + "<p>여러 갈래로 나누면 갈래가 많을수록 칸이 작아져 정보 이득이 저절로 커져요. C4.5는 그래서 정보 이득을 \"갈래가 얼마나 많은가\"로 나눈 <b>정보 이득 비율</b>을 써요.</p>"
           + K.code_table([("DecisionTreeClassifier(max_depth=3)", "질문을 최대 3번까지만. 깊을수록 train을 잘 외워요."), ("random_state=0", "같은 결과가 나오게 무작위를 고정"),
                           ("criterion=\"gini\"", "기본값. <code>\"entropy\"</code>로 바꿀 수 있어요.")])
           + "<p><b>과적합 막기</b>: 그냥 두면 트리는 끝 노드마다 한 건만 남을 때까지 나눠서 train을 다 외워요. 막는 방법은 둘이에요. 미리 멈추기(<code>max_depth</code> 깊이, <code>min_samples_leaf</code> 끝 노드 최소 건수, <code>max_leaf_nodes</code> 끝 노드 최대 수, <code>min_impurity_decrease</code> 최소 감소량)와, 다 키운 뒤 <b>가지치기</b>예요. 가지치기는 <code>오류 + α × 끝 노드 수</code>가 가장 작아지게 가지를 잘라요. 미리 멈추기는 지금 쓸모없어 보여도 나중에 좋은 질문으로 이어질 가지까지 막을 수 있어서, 가지치기가 보통 더 나아요.</p>"
           + K.out("f_tree")
           + K.try_("깊이 제한 없는 트리를 ccp_alpha로 가지치기하면?", "t8"), "s8"),
    K.step(9, "랜덤 포레스트", "조금씩 다른 데이터와 열로 트리를 200개 만들어 투표해요.",
           "<p>트리 하나는 데이터가 조금만 바뀌어도 첫 질문부터 달라져서 모양이 크게 바뀌어요(1쪽 ⑥의 <b>분산</b>이 커요). 그래서 여러 트리를 <b>서로 다르게</b> 만들어 다수결(회귀면 평균)을 내요. 이렇게 같은 종류 모델을 여럿 묶는 걸 <b>앙상블</b>이라고 해요.</p>"
           + f"<p>서로 다르게 만드는 방법은 둘이에요. ① <b>부트스트랩</b>: train에서 같은 크기만큼 <b>중복을 허용해</b> 뽑아요. 60건에서 60번 뽑으면 서로 다른 행은 약 63%뿐이에요 (실제로 한 번 뽑아 보니 {boot_u:.0%}). 트리마다 다른 행을 보게 돼요. 이걸 묶는 방식이 <b>배깅</b>(bootstrap aggregating)이에요. ② 질문을 고를 때마다 열도 일부만 무작위로 보여 줘요. 분류에서는 보통 전체 열 수의 제곱근이에요(8열이면 √8 ≈ 2.8이라 2개).</p>"
           "<p><b>왜 열까지 무작위로?</b> 트리 M개의 오차가 서로 완전히 독립이면, 평균 낸 오차의 분산은 1/M로 줄어요. 하지만 트리끼리 닮을수록(오차가 같은 방향이면) 줄어드는 양이 작아지고, 똑같은 트리 M개면 하나와 같아요. 이 예제에서는 <code>input_chars</code>가 워낙 강해서 모든 트리가 첫 질문으로 그걸 고르려 해요. 열을 가려야 트리들이 서로 달라져요.</p>"
           + K.code_table([("RandomForestClassifier(n_estimators=200, max_depth=4)", "깊이 4인 트리 200개"), ("max_features=\"sqrt\"", "기본값. 질문마다 열의 제곱근 개수만 보기"),
                           ("bootstrap=True", "기본값. 트리마다 중복 허용 추출")])
           + K.try_("트리를 1개, 5개, 20개, 200개로 바꾸면 교차검증 점수는?", "t9"), None),
    K.step(10, "XGBoost", "앞 트리가 틀린 부분을 다음 트리가 이어서 고치는 식으로 트리를 쌓아요 (부스팅).",
           "<p>배깅은 트리를 따로따로(병렬로) 만들어 평균 내요. <b>부스팅</b>은 차례로 만들어요. 앞 트리들이 틀린 부분을 다음 트리가 집중해서 배워요. 처음에는 일부러 약한 모델(작은 트리)로 시작해요. 너무 강한 모델로 시작하면 다음 트리가 고칠 게 없어요.</p>"
           + K.table(["방식", "다음 트리가 배우는 것"], [["AdaBoost", "앞에서 틀린 행에 가중치를 키워서, 그 행을 더 신경 쓰게"],
                                                     ["<b>Gradient Boosting</b>", "지금까지 예측의 <b>잔차</b>(정답 − 예측)를 그대로 맞히게"]])
           + "<p>AdaBoost를 직접 짜서 무게가 어떻게 바뀌는지, 트리를 늘리면 train 오차가 왜 빨리 0이 되는지, 그 뒤에 무엇이 바뀌는지는 <a href=\"ml-svm-lda-pca.html#s11\">4쪽 ⑪~⑬</a>에서 봐요.</p>"
           + "<p>잔차를 맞히는 게 왜 \"Gradient\"인가? 비용이 제곱 오차 ½(y − f)²이면, 예측 f에 대한 기울기는 f − y예요. 그러니 잔차 y − f는 <b>기울기의 반대 방향</b>, 곧 비용을 줄이는 방향이에요. 트리를 하나 더하는 게 1쪽 경사하강의 한 걸음과 같고, 여기서도 learning rate를 곱해 조금씩 더해요(shrinkage). 분류면 로그 손실의 기울기를 써요.</p>"
           + K.out("f_gb")
           + K.table(["", "배깅 (랜덤 포레스트)", "부스팅 (XGBoost 등)"], [["만드는 순서", "따로따로, 병렬", "차례로, 앞의 실수를 이어서"], ["주로 줄이는 것", "분산 (흔들림)", "편향 (덜 배운 것)"],
                                                                          ["과적합", "트리를 늘려도 잘 안 생겨요", "트리를 너무 많이 쌓으면 생겨요"], ["묶는 방법", "같은 무게로 투표, 평균", "learning rate를 곱해 더하기"]])
           + "<p><b>XGBoost</b>는 Gradient Boosting에 트리 크기 벌점(규제)을 넣고, 나눌 자리 찾기를 병렬로 빠르게 만든 구현이에요. 트리는 같은 깊이를 한 층씩 키워요. <b>LightGBM</b>은 비용을 가장 많이 줄이는 끝 노드 하나만 골라 더 깊게 키우고(그래서 더 빠르지만 과적합 조심), 잔차가 작은 행은 일부만 쓰고(GOSS), 동시에 0이 아닌 일이 없는 열들(원-핫처럼)을 하나로 묶어(EFB) 계산을 줄여요.</p>"
           + K.code_table([("XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1)", "작은 트리 200개를 차례로. 한 번에 조금씩(0.1) 고쳐요."), ("evaluate({...})", "1쪽의 비교 함수를 그대로 다시 써요.")])
           + K.try_("learning_rate를 0.01, 0.1, 1.0으로 바꾸면?", "t10"), "s10"),
    K.step(11, "특성 중요도", "포레스트가 판단할 때 어느 열을 많이 썼는지 봐요.",
           K.code_table([("named_steps[\"pre\"].get_feature_names_out()", "전처리 뒤 열 이름"), ("named_steps[\"model\"].feature_importances_", "열마다 중요도 (합이 1)")]), "s11"),
    K.step(12, "교차검증", "train+valid 80건을 다섯 덩어리로 나눠, 다섯 번 배우고 재서 평균을 내요.",
           K.code_table([("pd.concat([train, valid])", "고르는 데 쓸 80건. test는 3쪽 마지막까지 안 써요."), ("StratifiedKFold(n_splits=5, shuffle=True)", "덩어리마다 실패 비율이 비슷하게 나눠요."),
                         ("cross_validate(model, X, y, cv=cv, scoring=\"f1\")", "다섯 번 배우고 재서 점수 목록을 돌려줘요. 전처리도 매번 그 차례의 train으로만 다시 배워요.")]), "s12"),
]
practice2 = [("트리 깊이, 트리 개수는 어떻게 정하나", "후보 몇 개를 교차검증으로 비교해서 골라요. <code>GridSearchCV</code>는 후보 조합마다 교차검증을 돌려 가장 좋은 걸 골라 주는 도구예요. test로 고르면 안 돼요."),
             ("부스팅이 늘 이기나", "표 데이터에서 자주 강하지만, 데이터가 적으면 과적합하기 쉬워요. 교차검증 흔들림을 같이 봐요."),
             ("중요도를 원인으로 읽고 싶다", "중요도는 서로 비슷한 열끼리 나눠 갖기도 해서, 원인 분석에는 따로 실험이 필요해요."),
             ("숫자를 맞히는 트리 (회귀 트리)", "<code>DecisionTreeRegressor</code>. 불순도 대신 칸 안의 제곱 오차를 줄이는 질문을 고르고, 끝 노드의 평균을 예측해요. 그래서 예측이 계단 모양이고, train에 없던 범위(더 큰 값)는 맞히지 못해요. 잡음을 잘 따라가니 깊이를 꼭 제한해요."),
             ("트리가 약한 곳", "\"x가 1 늘면 y가 2 는다\" 같은 곧은 관계는 계단 여러 개로 흉내 내야 해서 선형 모델보다 서툴러요. 데이터가 조금 바뀌어도 트리 모양이 크게 바뀌어요. 둘 다 앙상블이 많이 덜어 줘요."),
             ("포레스트를 따로 검증하기 귀찮다", "부트스트랩에서 뽑히지 않은 약 37%의 행으로 그 트리를 채점할 수 있어요(OOB, out-of-bag). <code>oob_score=True</code>로 켜면 교차검증 없이 대략적인 점수를 줘요."),
             ("서로 다른 모델을 섞고 싶다", "<b>스태킹</b>: 로지스틱, 포레스트, XGBoost 같은 서로 다른 모델의 예측을 입력으로 받아, 마지막 모델(보통 로지스틱 회귀)이 \"누구 말을 얼마나 믿을지\"를 배워요. <code>StackingClassifier</code>. 안에서 교차검증 예측을 써서 누수를 막아요.")]
libs2 = [("⑧", "<code>DecisionTreeClassifier</code>", "결정 트리 (scikit-learn)"), ("⑨", "<code>RandomForestClassifier</code>", "랜덤 포레스트"), ("⑩", "<code>XGBClassifier</code>", "그래디언트 부스팅 (xgboost)"),
         ("⑪", "<code>feature_importances_</code>, <code>get_feature_names_out()</code>", "중요도"), ("⑫", "<code>StratifiedKFold</code>, <code>cross_validate</code>", "교차검증")]
K.build(ROOT / "topics" / "model" / "ml-trees-cv.html", title="트리와 앙상블, 교차검증", h1="트리와 앙상블, 교차검증: 더 나은 모델을 믿을 만하게 고르기", track="모델",
        crumb="모델 &gt; 머신러닝 기초 2/5쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-regression-classification.html\" style=\"color:inherit\">회귀와 분류</a> &nbsp;|&nbsp; 다음: <a href=\"ml-metrics-clustering.html\" style=\"color:inherit\">평가지표와 군집</a>",
        one_line="<b>트리</b>는 예/아니오 질문을 이어 가며 판단하고, 여러 트리를 묶으면(<b>앙상블</b>) 더 안정돼요. 어느 모델이 나은지는 한 번이 아니라 <b>교차검증</b>으로 여러 번 재서 판단해요.",
        why=why2, whole=whole2, steps=steps2, code_file="examples/ml/requests_model.py", practice=practice2, libs=libs2,
        next_rows=[("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "기준선, Pipeline, 지표"), ("<a href=\"ml-metrics-clustering.html\">평가지표, 특성 설계, 군집</a>", "확률 기준, 최종 평가")],
        sources=["다루는 범위: Day07 (2026-10-27) 트리, 앙상블, 특성 중요도, 과적합, 교차검증",
                 "함께 정리한 자료: 경희대 Pattern Recognition and Machine Learning 8강(의사결정나무), 12강(앙상블). 트리의 구조, Gini와 엔트로피, 정보 이득과 정보 이득 비율, CART와 C4.5, 멈추기와 가지치기, 회귀 트리, 부트스트랩과 배깅, AdaBoost와 Gradient Boosting, XGBoost와 LightGBM, 스태킹을 가져왔어요. 값과 그림은 이 예제로 다시 계산했어요",
                 DATA_SRC, "결과 값과 그림: <code>tools/pages/ml_basics.py</code>",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/tree.html\" target=\"_blank\">Decision Trees</a>, <a href=\"https://scikit-learn.org/stable/modules/ensemble.html\" target=\"_blank\">Ensembles</a>, <a href=\"https://scikit-learn.org/stable/modules/cross_validation.html\" target=\"_blank\">Cross-validation</a>, <a href=\"https://xgboost.readthedocs.io/\" target=\"_blank\">XGBoost</a>"])

# =============================== 3쪽: 평가지표, 특성 설계, 군집, 최종 ===============================
outs = {}
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.cluster import KMeans, DBSCAN  # noqa: E402
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster  # noqa: E402
fpr, tpr, thr = roc_curve(ns["y_valid"], ns["proba"])
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot([0, 1], [0, 1], color=C["gray"], ls="--", lw=1.2, label="동전 던지기 (AUC 0.5)")
ax.fill_between(fpr, tpr, step="post", color=C["soft"], alpha=.8)
ax.step(fpr, tpr, where="post", color=OK, lw=2.4, label=f"⑤의 로지스틱 (AUC {ns['auc']:.2f})")
p5 = (ns["proba"] >= 0.5)
fp5 = float(((p5 == 1) & (ns["y_valid"].to_numpy() == 0)).sum() / (ns["y_valid"] == 0).sum())
tp5 = float(((p5 == 1) & (ns["y_valid"].to_numpy() == 1)).sum() / (ns["y_valid"] == 1).sum())
ax.scatter([fp5], [tp5], s=80, color=FAIL, zorder=4)
ax.annotate(f"기준 0.5\n(헛경보율 {fp5:.2f}, 잡은 비율 {tp5:.2f})", (fp5, tp5), (0.02, 0.72), fontsize=10, color=FAIL, arrowprops=dict(arrowstyle="->", color=FAIL))
ax.set(xlabel="FPR: 실제 성공 중 \"실패\"라고 한 비율", ylabel="TPR (recall): 실제 실패 중 잡은 비율", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="기준을 1에서 0까지 내리며 그린 ROC 곡선")
ax.legend(fontsize=9, loc="lower right")
FIG["roc"] = fig_save(fig, "roc_curve")
S.claim(ns["auc"] < 0.65, "⑭ ROC 곡선이 대각선에 가깝다")
outs["f_roc"] = S.images("valid 20건으로 그린 ROC 곡선", [(FIG["roc"], "연보라 넓이가 AUC")],
                         ["기준을 아주 높게 잡으면 아무것도 \"실패\"라고 안 해서 (0, 0)이고, 0까지 내리면 다 \"실패\"라고 해서 (1, 1)이에요. 그 사이에서 기준을 내릴 때마다 위로(진짜 실패를 하나 더 잡음) 또는 오른쪽으로(헛경보 하나 더) 한 칸씩 가요.",
                          "완벽한 모델은 왼쪽 위 (0, 1)을 지나요. 이 모델은 대각선(동전 던지기)에 가깝게 붙어 있어요. 그래서 기준을 어디로 옮겨도 ⑬처럼 비용을 크게 줄이기 어려워요."])
Zc = ns["Z"]
ks_ = list(range(1, 9))
inert = [KMeans(n_clusters=k, n_init=10, random_state=0).fit(Zc).inertia_ for k in ks_]
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6))
a.plot(ks_, inert, color=OK, marker="o", lw=2.2)
a.set(xlabel="묶음 수 k", ylabel="SSE (inertia_)", title="엘보 방법: 꺾이는 곳 찾기")
sil = ns["silhouettes"]
b.bar(list(sil.keys()), list(sil.values()), color=[FAIL if v == max(sil.values()) else C["soft"] for v in sil.values()], edgecolor=OK)
b.axhline(0.5, color=C["gray"], ls="--", lw=1.2)
b.text(2, 0.52, "0.5 넘으면 \"잘 나뉘었다\"고 보는 기준", fontsize=9, color=C["gray"])
b.set(xlabel="묶음 수 k", ylabel="실루엣 평균", ylim=(0, 0.62), title="실루엣: 높을수록 잘 나뉨", xticks=list(sil.keys()))
FIG["elbow"] = fig_save(fig, "kmeans_elbow_silhouette")
kbest = max(sil, key=sil.get)
kmb = ns["clusters"][kbest]
lab = kmb.labels_
frate = pd.Series(ns["dev"]["failed"].to_numpy()).groupby(lab).mean()
fsize = np.bincount(lab)
fig, ax = plt.subplots(figsize=(7.2, 4.2))
cl_cols = [C["purple"], C["mint"], C["orange"], C["blue"], C["pink"]]
for c_ in range(kbest):
    msk = lab == c_
    ax.scatter(Zc[msk, 0], Zc[msk, 1] + np.random.default_rng(c_).normal(0, 0.04, msk.sum()), s=34, color=cl_cols[c_], alpha=.85, label=f"묶음 {c_}: {fsize[c_]}건, 실패 {frate[c_]:.0%}")
    ax.scatter(kmb.cluster_centers_[c_, 0], kmb.cluster_centers_[c_, 1], s=260, marker="*", color=cl_cols[c_], edgecolor=C["ink"], zorder=4)
ax.set(xlabel="input_chars (크기 맞춤)", ylabel="allowed_tools (크기 맞춤)", title=f"k={kbest} 묶음을 숫자 두 열 위에 (★ 중심)")
ax.legend(fontsize=9, loc="upper left")
FIG["kmeans"] = fig_save(fig, "kmeans_clusters")
Lw = linkage(Zc, method="ward")
fig, ax = plt.subplots(figsize=(11, 3.8))
dendrogram(Lw, ax=ax, no_labels=True, color_threshold=Lw[-(kbest - 1), 2] - 1e-9, above_threshold_color=C["gray"])
ax.axhline((Lw[-kbest, 2] + Lw[-(kbest - 1), 2]) / 2, color=FAIL, ls="--", lw=1.4)
ax.set(ylabel="합칠 때 늘어난 거리 (Ward)", title=f"계층적 군집: 아래에서부터 가까운 것끼리 합치기 (분홍 점선에서 자르면 {kbest}묶음)")
ax.grid(False)
FIG["dendro"] = fig_save(fig, "dendrogram")
ward_sizes = np.bincount(fcluster(Lw, kbest, "maxclust"))[1:].tolist()
db_rows = []
for eps in (0.5, 0.8, 1.0, 1.5):
    lb = DBSCAN(eps=eps, min_samples=5).fit_predict(Zc)
    db_rows.append([eps, len(set(lb) - {-1}), int((lb == -1).sum()), ", ".join(str(int((lb == c_).sum())) for c_ in sorted(set(lb) - {-1})) or "-"])
S.claim(db_rows[0][2] > len(Zc) / 2 and db_rows[-1][1] == 1 and db_rows[-1][2] < len(Zc) * 0.1, "⑯ 반경이 작으면 대부분 잡음, 크면 거의 한 묶음")
outs["t16"] = S.table("DBSCAN으로 같은 80건을 묶으면 (min_samples=5)", ["eps (이웃 반경)", "묶음 수", "잡음으로 뺀 건수", "묶음별 건수"], db_rows,
                      ["반경이 작으면 이웃이 5개 이상인 점(핵심점)이 거의 없어서 대부분이 잡음이에요. 반경을 키우면 묶음이 생기다가, 너무 크면 거의 전부가 한 묶음이 돼요.",
                       "어느 eps에서도 K-Means처럼 \"고르게 나뉜 몇 묶음\"이 나오지 않아요. 이 데이터에는 밀도가 뚜렷하게 다른 덩어리가 없다는 뜻이고, ⑯의 낮은 실루엣과 같은 이야기예요."])
S.claim(kbest == 3 and max(sil.values()) < 0.35, "⑯ 실루엣은 k=3에서 가장 높지만 0.5에 한참 못 미친다")
outs["f_elbow"] = S.images("k를 1부터 8까지 바꿔 본 K-Means", [(FIG["elbow"], "왼쪽: 묶음 안 거리 제곱 합, 오른쪽: 실루엣")],
                           ["SSE는 k를 늘리면 늘 줄어요(k = 80이면 0). 그래서 \"가장 작은 k\"가 아니라 <b>줄어드는 폭이 확 작아지는 곳</b>(팔꿈치)을 봐요. 여기서는 k=3 근처에서 기울기가 완만해져요.",
                            f"실루엣은 k={kbest}에서 가장 높지만 0.5 기준선에 한참 못 미쳐요. 둘 다 \"3개 정도가 그나마 낫지만 뚜렷한 덩어리는 아니다\"라고 말해요."])
outs["f_km"] = S.images(f"k={kbest} 묶음을 숫자 두 열 위에 그리면", [(FIG["kmeans"], "점은 위아래로 살짝 흩어 겹침을 줄였어요")],
                        ["K-Means는 8열 전부로 묶었지만, 그림은 그중 숫자 두 열만 보여 줘요. 그래서 경계가 이 평면에서 깔끔하게 안 보일 수 있어요.",
                         f"정답(<code>failed</code>)은 묶을 때 쓰지 않았는데, 묶음마다 실패 비율이 달라요. 실패가 {frate.min():.0%}인 묶음이 있어요. \"이런 요청은 거의 실패하지 않는다\"는 무리를 정답 없이 찾은 셈이에요. 묶음에 이름을 붙일 때는 이렇게 묶음별 평균을 봐요."])
outs["f_den"] = S.images("같은 80건을 계층적으로 묶은 덴드로그램", [(FIG["dendro"], "아래 끝 하나하나가 요청 한 건")],
                         [f"맨 아래에서는 80건이 각자 한 묶음이에요. 가장 가까운 둘부터 합치고, 합친 높이가 \"얼마나 멀리 있는 걸 합쳤나\"예요. 위로 갈수록 먼 것끼리 합쳐요.",
                          f"분홍 점선 높이에서 가로로 자르면 {kbest}묶음({', '.join(map(str, ward_sizes))}건)이 돼요. k를 미리 정하지 않고, 그림을 보고 긴 세로줄(합칠 때 거리가 크게 뛴 곳)을 자르는 게 요령이에요."])
for _k in ("f_roc", "f_elbow", "f_km", "f_den"):
    outs[_k]["chart"] = True
proba = ns["proba"]
missed = proba[(ns["y_valid"].to_numpy() == 1) & (proba < 0.5)]   # 0.5 기준에서 놓친 실패의 확률
outs["why"] = S.table("1쪽 로지스틱 모델이 valid 실패 6건에 준 확률", ["request_id", "실패 확률", "0.5 기준 판단"],
                      [[E(i), f3(p), "실패" if p >= 0.5 else "<span class=\"bad\">성공 (놓침)</span>"] for i, p, y in zip(ns["X_valid"].index, proba, ns["y_valid"]) if y == 1],
                      [f"모델은 사실 \"실패\" 아니면 \"성공\"이 아니라 <b>확률</b>을 내요. 0.5라는 기준선은 우리가 정한 거예요. {missed.min():.1f}~{missed.max():.1f}짜리 실패도 \"성공\"으로 넘어갔어요.", "놓친 실패가 큰 손해라면 기준을 낮춰야 해요. 이 쪽에서는 그 기준을 비용으로 정하고, 확률의 품질을 재고, 입력을 다듬고, 정답 없이 데이터를 묶어 보고, 마지막으로 test로 딱 한 번 채점해요."])
_c = ns["costs"]
S.claim(all(a_["fp"] >= b_["fp"] and a_["fn"] <= b_["fn"] for a_, b_ in zip(_c, _c[1:])), "⑬ 기준을 낮추면 헛경보는 늘고 놓침은 준다")
outs["s13"] = S.table("⑬ 기준별 오류와 비용 (FN 하나 = 5, FP 하나 = 1)", ["기준", "헛경보 FP", "놓침 FN", "비용"],
                      [[c["threshold"], c["fp"], c["fn"], f"<b>{c['cost']}</b>" if c["cost"] == min(x["cost"] for x in ns["costs"]) else c["cost"]] for c in ns["costs"]],
                      [f"기준을 낮추면 헛경보는 늘고 놓침은 줄어요. 비용은 기준 {min(ns['costs'], key=lambda c: c['cost'])['threshold']}에서 가장 작아요.", "비용 5:1은 설명용 가정이에요. 실제로는 \"실패를 놓치면 무슨 일이 생기나\"를 업무 담당자와 정해요. 기준은 valid로 정하고, test로 정하면 안 돼요."])
outs["s14"] = S.box("⑭ ROC-AUC", [(f"roc_auc_score(y_valid, proba) → {ns['auc']:.3f}", 1)],
                    [f"\"실패 하나와 성공 하나를 뽑았을 때, 실패 쪽에 더 높은 확률을 줄 확률\"이에요. 0.5면 동전 던지기, 1이면 완벽. 기준과 상관없이 확률의 순서만 봐요. {ns['auc']:.2f}이면 이 모델의 확률은 순서도 많이 틀려요."])
S.claim(ns["cv_scores"]["logistic+log1p"].mean() < ns["cv_scores"]["logistic"].mean(), "⑮ log1p를 넣으니 오히려 나빠졌다")
outs["s15"] = S.box("⑮ log1p 변환을 넣은 로지스틱의 교차검증 F1", [f"점수 → {[round(float(x), 3) for x in ns['cv_scores']['logistic+log1p']]}", (f"평균 → {ns['cv_scores']['logistic+log1p'].mean():.3f}  (변환 없는 logistic {ns['cv_scores']['logistic'].mean():.3f})", 1, "bad")],
                    ["치우친 숫자를 펴 주는 변환인데, 이 데이터에서는 오히려 나빠졌어요. 특성 설계는 \"해 보고 교차검증으로 확인\"이에요. 좋아 보이는 기법도 데이터에 따라 손해일 수 있어요."])
outs["s16"] = S.table("⑯ K-Means: 몇 개로 묶을 때 가장 잘 나뉘나", ["묶음 수 k", "실루엣 점수"], [[k, f"<b>{f3(v)}</b>" if v == max(ns["silhouettes"].values()) else f3(v)] for k, v in ns["silhouettes"].items()],
                      [f"정답(<code>failed</code>) 없이 입력만 보고 비슷한 요청끼리 묶었어요. 실루엣 점수는 \"내 묶음과는 가깝고 다른 묶음과는 먼가\"(−1~1)예요. k={max(ns['silhouettes'], key=ns['silhouettes'].get)}일 때 가장 높지만 {max(ns['silhouettes'].values()):.2f}로, 뚜렷하게 나뉘는 데이터는 아니에요."])
outs["s17"] = S.box("⑰ 최종 평가", [(f"교차검증 평균으로 고른 모델 → {ns['best']}", 1), (f"test 20건 (처음 사용) → F1 {ns['test_scores']['f1']:.3f}, recall {ns['test_scores']['recall']:.3f}, accuracy {ns['test_scores']['accuracy']:.2f}", 2)],
                    ["⑫와 ⑮의 교차검증 평균이 가장 높은 모델을 골라, train+valid 80건 전부로 다시 배웠어요.", "test는 이때 처음 썼어요. 이 숫자가 \"새 요청에서 기대할 성능\"이에요. 이걸 보고 다시 모델을 바꾸면 test도 고르는 데 쓴 셈이라, 더는 공정한 성적이 아니에요."])
flow3 = [[13, "기준과 비용", "0.5가 정답 아님", "FN, FP 비용"], [14, "ROC-AUC", "확률의 순서", "기준과 무관"], [15, "특성 설계", "log1p", "교차검증으로 확인"], [16, "군집", "K-Means", "정답 없이 묶기"], [17, "최종", "test 한 번", "고른 뒤 채점"]]
S.save(ROOT / "topics" / "model" / "ml-metrics-clustering-data.js", {"code": code, "names": S.names(code), "range": [13, 17], "outs": outs, "flow": flow3, "env": ENV})
why3 = f"""<p>지금까지는 모델이 \"실패\" 또는 \"성공\"이라고 답하는 것만 봤어요. 그런데 그 답은 확률에 0.5라는 기준을 그어 만든 거예요.</p>{K.out("why")}"""
whole3 = """<p>기준을 비용으로 정하고, 확률 자체의 품질을 재고, 입력을 다듬어 보고, 정답 없이 묶어 보고, 마지막에 test로 한 번 채점해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps3 = [
    K.step(13, "임계값과 오류 비용", "확률 기준을 여러 개 시험해서, 오류 비용이 가장 작은 기준을 찾아요.",
           K.code_table([("logit.predict_proba(X_valid)[:, 1]", "실패일 확률 (두 번째 열)"), ("(proba >= t).astype(int)", "기준 t 이상이면 실패(1)"), ("confusion_matrix(...).ravel()", "2×2 표를 tn, fp, fn, tp 네 값으로 펴기")]), "s13"),
    K.step(14, "ROC-AUC", "기준과 상관없이, 확률이 실패를 성공보다 높게 매기는지 재요.",
           "<p>⑬에서 본 것처럼 기준을 내리면 실패를 더 잡지만 헛경보도 늘어요. 이 둘을 한 그림에 그린 게 <b>ROC 곡선</b>이에요. 가로축은 <b>FPR</b>(실제 성공 중 \"실패\"라고 잘못 말한 비율, FP ÷ (FP + TN)), 세로축은 <b>TPR</b>(실제 실패 중 잡은 비율, TP ÷ (TP + FN), 1쪽의 recall과 같아요)이에요. 기준을 1에서 0까지 내리며 점을 찍어요.</p>"
           "<p>곡선 아래 넓이가 <b>AUC</b>예요. 넓이가 1이면 어떤 기준에서든 헛경보 없이 다 잡는 자리가 있다는 뜻이고, 0.5면 대각선, 곧 동전 던지기예요.</p>"
           + K.out("f_roc")
           + K.code_table([("roc_auc_score(y, proba)", "예측 \"답\"이 아니라 \"확률\"을 넣어요."), ("roc_curve(y, proba)", "기준마다 (FPR, TPR, 기준) 목록. 그림을 그릴 때 써요.")]), "s14"),
    K.step(15, "특성 설계: 치우친 값 펴기", "숫자 열에 log(1+x)를 씌워 큰 값을 줄여 본 뒤, 교차검증으로 나아졌는지 확인해요.",
           K.code_table([("FunctionTransformer(np.log1p)", "아무 함수나 전처리 단계로. <code>log1p</code>는 log(1+x)로, 0도 처리돼요."), ("feature_names_out=\"one-to-one\"", "열 이름을 그대로 유지")]), "s15"),
    K.step(16, "K-Means: 정답 없이 묶어 보기", "정답 없이 입력만으로 비슷한 요청끼리 k개 묶음을 만들고, 몇 개가 적당한지 봐요.",
           "<p>정답 없이 데이터의 구조를 찾는 걸 <b>비지도 학습</b>이라고 해요. 요청 종류를 나눠 보거나(고객 나누기, 뉴스 기사 묶기도 같은 일), 이상한 요청을 찾는 데 써요. 좋은 묶음은 <b>같은 묶음끼리는 가깝고, 다른 묶음과는 멀어요</b>.</p>"
           + K.table(["", "분류 (1쪽)", "군집"], [["학습", "지도 학습", "비지도 학습"], ["입력", "데이터와 정답", "데이터만"], ["목표", "정해진 종류 맞히기", "있는 줄 몰랐던 무리 찾기"]])
           + "<p><b>K-Means</b>는 \"각 점에서 자기 묶음 중심까지 거리 제곱의 합\"(SSE)을 가장 작게 하는 k개의 중심을 찾아요. 두 단계를 번갈아 반복해요.</p>"
           "<ol><li><b>배정</b>: 모든 점을 가장 가까운 중심의 묶음에 넣어요.</li><li><b>이동</b>: 중심을 그 묶음 점들의 평균 자리로 옮겨요.</li><li>배정이 더 바뀌지 않으면 멈춰요.</li></ol>"
           "<p>시작 중심을 어디에 두느냐에 따라 다른 답에 멈출 수 있어요. 그래서 시작을 여러 번 바꿔(<code>n_init=10</code>) SSE가 가장 작은 걸 고르고, 시작점도 서로 멀리 떨어지게 뽑아요(k-means++, 기본값). 한 번 반복할 때 계산은 (점 수 × k × 열 수)에 비례해서 큰 데이터에도 빨라요.</p>"
           "<p><b>k는 몇 개?</b> 두 가지를 같이 봐요. <b>엘보 방법</b>은 k를 늘려 가며 SSE가 줄어드는 폭이 확 작아지는 곳을 찾아요. <b>실루엣</b>은 점마다 a = 같은 묶음 점들까지 평균 거리, b = 가장 가까운 다른 묶음 점들까지 평균 거리로 <code>(b − a) ÷ max(a, b)</code>를 구해 평균 내요. 1에 가까울수록 잘 나뉘었고, 0 근처면 경계에 걸쳐 있고, 음수면 잘못 들어간 점이에요. k=2가 늘 높게 나오기 쉬워서 차순위 k도 같이 봐요.</p>"
           + K.out("f_elbow") + K.out("f_km")
           + "<p><b>계층적 군집</b>은 k를 미리 정하지 않아요. 모든 점을 한 묶음씩으로 시작해서, 가장 가까운 두 묶음을 합치기를 하나가 될 때까지 반복해요. 합친 순서를 그린 나무가 <b>덴드로그램</b>이고, 원하는 높이에서 잘라 묶음을 정해요. 묶음 사이 거리를 재는 방법이 여럿이에요.</p>"
           + K.table(["방법", "두 묶음 사이 거리"], [["single", "가장 가까운 두 점의 거리. 길쭉한 사슬 모양으로 이어지기 쉬워요"], ["complete", "가장 먼 두 점의 거리"], ["average", "모든 점 쌍 거리의 평균"],
                                                  ["centroid", "두 중심 사이 거리"], ["<b>Ward</b>", "합쳤을 때 묶음 안 SSE가 얼마나 늘어나나. K-Means와 같은 기준이라 비슷한 결과를 줘요"]])
           + K.out("f_den")
           + "<p><b>DBSCAN</b>은 \"점이 빽빽한 곳\"을 묶음으로 봐요. 반경 eps 안에 이웃이 <code>min_samples</code>개 이상인 점이 <b>핵심점</b>, 핵심점의 반경 안에 있지만 스스로는 이웃이 적은 점이 <b>경계점</b>, 둘 다 아니면 <b>잡음</b>이에요. 핵심점끼리 이어지면 한 묶음이에요. k를 정할 필요가 없고, 어떤 모양의 묶음이든 찾고, 잡음을 따로 빼 줘요. 대신 eps를 잘 골라야 해요(<code>min_samples</code>는 보통 열 수 + 1 정도에서 시작).</p>"
           + K.table(["", "K-Means", "DBSCAN"], [["묶음 모양", "중심 둘레의 둥근 덩어리", "아무 모양"], ["k", "정해 줘야 해요", "필요 없어요 (eps는 필요)"], ["잡음", "모든 점을 어딘가에 넣어요", "잡음으로 빼요"], ["크기와 밀도가 다른 묶음", "약해요", "강해요 (밀도 차이가 너무 크면 약해요)"]])
           + K.code_table([("KMeans(n_clusters=k, n_init=10)", "k개 중심을 찾아 가장 가까운 중심으로 묶어요. 시작점을 10번 바꿔 가장 좋은 것"), ("km.inertia_", "SSE (엘보 방법에 써요)"),
                           ("silhouette_score(Z, labels)", "묶음이 얼마나 잘 나뉘었나 (−1~1)")])
           + K.try_("K-Means 대신 DBSCAN으로 묶으면? eps를 바꿔 가며", "t16"), "s16"),
    K.step(17, "마지막 한 번: test로 최종 평가", "교차검증으로 고른 모델을 80건 전부로 다시 배우고, 아껴 둔 test로 한 번만 채점해요.",
           K.code_table([("max(cv_scores, key=...)", "교차검증 평균이 가장 높은 후보의 이름"), ("clone(candidates[best]).fit(dev...)", "같은 설정의 새 모델을 80건 전부로")]), "s17"),
]
practice3 = [("기준을 정할 때", "valid(또는 교차검증)로 정하고 test는 건드리지 않아요. 업무에서 FN, FP 비용을 먼저 물어요."),
             ("AUC는 높은데 F1이 낮다", "확률의 순서는 맞지만 기준이 안 맞는 거예요. ⑬처럼 기준을 조정해요."),
             ("묶음(군집)에 이름 붙이기", "K-Means는 숫자 번호만 줘요. 묶음마다 평균값을 보고 사람이 뜻을 붙여요. 크기 맞추기를 안 하면 큰 숫자 열로만 묶여요."),
             ("열끼리 서로 닮았다 (군집)", "유클리드 거리는 닮은 열 두 개를 따로 세서 그 방향을 두 번 쳐 줘요. 열 사이 상관을 빼고 재는 마할라노비스 거리를 쓰거나, 4쪽의 PCA로 먼저 정리해요."),
             ("데이터가 아주 크다 (계층적 군집)", "모든 점 쌍의 거리를 저장해서 점 수의 제곱만큼 메모리가 들어요. 수만 건이 넘으면 K-Means나 일부만 뽑아 계층적 군집을 하고, 그 결과를 K-Means의 시작 중심으로 써요."),
             ("eps를 정하기 어렵다 (DBSCAN)", "점마다 <code>min_samples</code>번째로 가까운 이웃까지 거리를 구해 정렬해 그리고, 확 꺾이는 곳을 eps로 써요. HDBSCAN은 eps 없이 여러 밀도를 한 번에 다뤄요.")]
libs3 = [("⑬", "<code>predict_proba()</code>, <code>confusion_matrix().ravel()</code>", "확률과 기준"), ("⑭", "<code>roc_auc_score</code>", "확률의 순서 품질"), ("⑮", "<code>FunctionTransformer</code>, <code>np.log1p</code>", "특성 변환"),
         ("⑯", "<code>KMeans</code>, <code>silhouette_score</code>", "군집"), ("⑰", "<code>clone()</code>, <code>fit()</code>, <code>predict()</code>", "최종 학습과 평가")]
K.build(ROOT / "topics" / "model" / "ml-metrics-clustering.html", title="평가지표, 특성 설계, 군집", h1="평가지표, 특성 설계, 군집: 기준을 정하고 마지막에 한 번 채점하기", track="모델",
        crumb="모델 &gt; 머신러닝 기초 3/5쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-trees-cv.html\" style=\"color:inherit\">트리와 교차검증</a> &nbsp;|&nbsp; 다음: <a href=\"ml-svm-lda-pca.html\" style=\"color:inherit\">SVM, 판별분석, 차원 축소</a>",
        one_line="분류 모델은 사실 <b>확률</b>을 내고, \"실패\"라고 부를 <b>기준은 오류 비용으로</b> 정해요. 모델과 기준을 다 고른 뒤에 <b>test로 딱 한 번</b> 채점해요.",
        why=why3, whole=whole3, steps=steps3, code_file="examples/ml/requests_model.py", practice=practice3, libs=libs3, libs_note="<code>cost_at</code>, <code>build_fe</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "후보 모델과 교차검증"), ("<a href=\"ml-svm-lda-pca.html\">SVM, 판별분석, 차원 축소</a>", "다른 모델 셋과 같은 교차검증 비교"), ("<a href=\"dl-tensor-training.html\">딥러닝 기초</a>", "신경망으로 넘어가기")],
        sources=["다루는 범위: Day08 (2026-10-28) 평가지표, 특성 설계, K-Means, 최종 평가",
                 "함께 정리한 자료: 경희대 Pattern Recognition and Machine Learning 4강(ROC 곡선), 10강(군집화). 분류와 군집의 차이, K-Means의 목적과 두 단계, 시작점 문제, 엘보 방법과 실루엣 식, 계층적 군집과 연결 방법, DBSCAN과 K-Means 비교, 마할라노비스 거리를 가져왔어요. 값과 그림은 이 예제로 다시 계산했어요",
                 DATA_SRC, "결과 값과 그림: <code>tools/pages/ml_basics.py</code>",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/classification_threshold.html\" target=\"_blank\">Tuning the decision threshold</a>, <a href=\"https://scikit-learn.org/stable/modules/clustering.html#k-means\" target=\"_blank\">K-means</a>"])
print("ok")
