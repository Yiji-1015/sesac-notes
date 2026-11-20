"""모델 > 딥러닝 기초 두 쪽. examples/dl/points_net.py 하나가 ①~⑪로 두 쪽에 걸쳐 자란다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/dl_basics.py
"""
import sys
import warnings
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "dl"
code = S.parse((EX / "points_net.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="points_net.py")  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CPU 학습 (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'})"
SRC_DATA = "데이터: <code>examples/dl/data/points.csv</code> (점 400개, 두 무리. 기존 노트와 같은 예제)"
ns = run(11)
df = ns["df"]

# =============================== 그림: 모두 이 예제를 실제로 계산해서 그린다 ===============================
import numpy as np  # noqa: E402
from torch import nn  # noqa: E402
sys.path.insert(0, str(ROOT / "tools" / "visuals"))
from vstyle import plt, C, setup  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "dl"
IMG.mkdir(parents=True, exist_ok=True)
C0, C1 = C["purple"], C["pink"]
BG = ListedColormap(["#ece6ff", "#ffe0ee"])


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/dl/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


Xtr, ytr = ns["X_train"].numpy(), ns["y_train"].numpy().ravel()
gx, gy = np.meshgrid(np.linspace(-1.6, 2.7, 260), np.linspace(-1.0, 1.6, 200))
G = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)


def pts(ax):
    ax.scatter(Xtr[ytr == 0, 0], Xtr[ytr == 0, 1], s=12, color=C0, label="무리 0")
    ax.scatter(Xtr[ytr == 1, 0], Xtr[ytr == 1, 1], s=12, color=C1, label="무리 1")
    ax.set(xlim=(-1.6, 2.7), ylim=(-1.0, 1.6), xlabel="x1")
    ax.grid(False)


def boundary(ax, model):
    with torch.no_grad():
        z = model(G).numpy().reshape(gx.shape)
    ax.contourf(gx, gy, (z > 0).astype(int), levels=[-.5, .5, 1.5], cmap=BG)
    ax.contour(gx, gy, z, levels=[0], colors=[C["ink"]], linewidths=1.8)


# ③ 은닉 칸 16개가 만든 꺾이는 선
net0 = ns["net"]
W0, b0 = net0[0].weight.detach().numpy(), net0[0].bias.detach().numpy()
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2))
boundary(a, ns["linear"])
pts(a)
a.set(title=f"② 선 하나 (valid {ns['linear_acc']:.3f})", ylabel="x2")
boundary(b, net0)
xs = np.linspace(-1.6, 2.7, 50)
for w, bb in zip(W0, b0):
    if abs(w[1]) > 1e-6:
        b.plot(xs, -(w[0] * xs + bb) / w[1], color=C["gray"], lw=.8, ls="--", alpha=.7)
    else:
        b.axvline(-bb / w[0], color=C["gray"], lw=.8, ls="--", alpha=.7)
pts(b)
b.set(title=f"⑤ 은닉 칸 16개 (valid {ns['hist'][-1]['valid_acc']:.3f}), 점선: 칸마다 꺾이는 선")
b.legend(fontsize=9, loc="lower right")
P_RELU = fig_save(fig, "relu_hinges")
# ③ 🔁 폭과 깊이
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402


def fresh_loader():  # 실험마다 ⑤와 똑같은 섞기 순서로 시작 (앞 실험에 영향받지 않게)
    return DataLoader(TensorDataset(ns["X_train"], ns["y_train"]), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))


def fresh_run(opt_fn, epochs=100, **kw):
    torch.manual_seed(0)
    m = ns["make_net"](**kw)
    return m, ns["train"](m, opt_fn(m.parameters()), epochs, loader=fresh_loader())


wd_rows, wd_models = [], []
for hidden, depth in [(2, 1), (4, 1), (16, 1), (64, 1), (4, 4), (16, 2)]:
    m, h = fresh_run(lambda p: torch.optim.Adam(p, lr=0.01), hidden=hidden, depth=depth)
    wd_models.append((hidden, depth, m, h[-1]["valid_acc"]))
    wd_rows.append([f"<b>{hidden}칸 × {depth}층</b> (지금)" if (hidden, depth) == (16, 1) else f"{hidden}칸 × {depth}층", sum(p.numel() for p in m.parameters()), f3(h[-1]["train_loss"]), f"{h[-1]['valid_acc']:.3f}"])
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
for ax, (hd, dp, m, va) in zip(axs, [w for w in wd_models if (w[0], w[1]) in ((2, 1), (4, 1), (16, 1), (4, 4))]):
    boundary(ax, m)
    pts(ax)
    ax.set(title=f"{hd}칸 × {dp}층, 가중치 {sum(p.numel() for p in m.parameters())}개 (valid {va:.3f})")
axs[0].set_ylabel("x2")
P_WD = fig_save(fig, "width_depth")
# ④ 초기화: 20층 ReLU 신경망에 무작위 입력을 넣으면
torch.manual_seed(0)
xin = torch.randn(256, 64)
init_std = {}
for name, fn in [("작게: 표준편차 0.01", lambda w: nn.init.normal_(w, 0, 0.01)), ("크게: 표준편차 1", lambda w: nn.init.normal_(w, 0, 1.0)), ("He: 표준편차 √(2 ÷ 64)", lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"))]:
    torch.manual_seed(0)
    h_, s_ = xin, []
    for _ in range(20):
        lin = nn.Linear(64, 64, bias=False)
        fn(lin.weight)
        h_ = torch.relu(lin(h_))
        s_.append(h_.std().item())
    init_std[name] = s_
_isv = list(init_std.values())
fac = {"small": (_isv[0][-1] / _isv[0][0]) ** (1 / 19), "big": (_isv[1][-1] / _isv[1][0]) ** (1 / 19)}
init_std_last = {"small": _isv[0][-1], "big": _isv[1][-1]}
fig, ax = plt.subplots(figsize=(8, 3.8))
for (name, s_), col in zip(init_std.items(), [C["blue"], C1, C0]):
    ax.plot(range(1, 21), s_, marker="o", ms=3, lw=2.2, color=col, label=name)
ax.set_yscale("log")
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set(xlabel="층 번호", ylabel="그 층 출력의 표준편차 (로그 눈금)", title="20층을 지나며 값이 사라지거나 폭발하거나")
ax.legend(fontsize=9)
P_INIT = fig_save(fig, "init_scale")
# ⑥⑦ 손실 곡선
n_mom = run(7, [('make = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[opt_name]', 'make = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam, "momentum": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9)}[opt_name]'),
                 ('for name in ("sgd", "adam")}', 'for name in ("sgd", "adam", "momentum")}')])  # ⑦에 모멘텀만 하나 더 (sgd, adam은 ⑦과 같은 값)
h_sgd60, h_adam60, h_mom = n_mom["opt_runs"]["sgd"], n_mom["opt_runs"]["adam"], n_mom["opt_runs"]["momentum"]
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8))
for (lr, h), col in zip(ns["lr_runs"].items(), [C["gray"], C0, C1]):
    a.plot([x["train_loss"] for x in h], color=col, lw=2, label=f"SGD, 학습률 {lr}")
a.set(xlabel="epoch", ylabel="train 손실", title="⑥ 학습률만 다르게", ylim=(0, 1.0))
a.legend(fontsize=9)
for (nm, h), col in zip([("SGD", h_sgd60), ("SGD + 모멘텀 0.9", h_mom), ("Adam", h_adam60)], [C["gray"], C["orange"], C0]):
    b.plot([x["train_loss"] for x in h], color=col, lw=2, label=nm)
b.set(xlabel="epoch", ylabel="train 손실", title="⑦ 같은 학습률 0.01, 고치는 방법만 다르게")
b.legend(fontsize=9)
P_OPT = fig_save(fig, "optimizer_curves")
# ⑧~⑩ 과적합 곡선
fig, ax = plt.subplots(figsize=(9, 3.9))
for nm, h, col in [("과적합 (⑧)", ns["over"], C1), ("규제 (⑨)", ns["regd"], C0)]:
    ax.plot([x["train_loss"] for x in h], color=col, lw=1.6, ls="--", label=f"{nm} train")
    ax.plot([x["valid_loss"] for x in h], color=col, lw=2.4, label=f"{nm} valid")
be0 = min(ns["early"], key=lambda x: x["valid_loss"])
ax.axvline(len(ns["early"]) - 1, color=C["mint"], lw=1.4, ls=":", label=f"조기 종료가 멈춘 곳 (epoch {len(ns['early'])})")
ax.scatter([be0["epoch"] - 1], [be0["valid_loss"]], color=C["mint"], s=60, zorder=4, label=f"가장 좋던 때 (epoch {be0['epoch']})")
ax.set(xlabel="epoch", ylabel="손실", ylim=(0, 2.0), title="train 40개로 큰 신경망: train은 0으로, valid는 위로")
ax.legend(fontsize=9, loc="upper left", ncol=2)
P_OVER = fig_save(fig, "overfit_curves")
w_over = torch.cat([p.detach().ravel() for n_, p in ns["big"].named_parameters() if "weight" in n_]).numpy()
w_reg = torch.cat([p.detach().ravel() for n_, p in ns["reg"].named_parameters() if "weight" in n_]).numpy()
fig, ax = plt.subplots(figsize=(8, 3.4))
bins = np.linspace(-1.2, 1.2, 61)
ax.hist(w_over, bins=bins, color=C1, alpha=.55, label=f"과적합 (⑧): 절댓값 평균 {np.abs(w_over).mean():.3f}")
ax.hist(w_reg, bins=bins, color=C0, alpha=.55, label=f"규제 (⑨): 절댓값 평균 {np.abs(w_reg).mean():.3f}")
ax.set_yscale("log")
ax.set(xlabel="가중치 값", ylabel="개수 (로그 눈금)", title="weight decay가 가중치를 0 쪽으로 모은 모습")
ax.legend(fontsize=9)
P_WH = fig_save(fig, "weight_hist")


def hist_rows(h, every):
    return [[x["epoch"], f3(x["train_loss"]), f3(x["valid_loss"]), f"{x['valid_acc']:.3f}"] for x in h if x["epoch"] == 1 or x["epoch"] % every == 0]


# =============================== 1쪽 ===============================
outs = {}
outs["data"] = S.table("points.csv 앞 5줄", ["id", "x1", "x2", "label", "split"], [[E(str(v)) for v in r] for r in df.head(5).values],
                       ["점마다 좌표 두 개(<code>x1</code>, <code>x2</code>)와 무리 번호(<code>label</code> 0 또는 1)가 있어요. train 240개로 배우고 valid 80개로 재요."])
n1 = run(1)
outs["s1"] = S.box("① 실행 결과", [(f"torch.cuda.is_available() → {n1['gpu']}", 1), (f"X_train.shape → {tuple(n1['X_train'].shape)}, dtype {n1['X_train'].dtype}", 2), f"y_train.shape → {tuple(n1['y_train'].shape)}"],
                   [("이 PC에는 GPU가 있어요. 그래도 이 예제는 작아서 CPU로 학습해요." if n1["gpu"] else "이 PC에는 쓸 수 있는 GPU가 없어서 False예요. 이 예제는 작아서 CPU로도 충분해요.") + " GPU는 숫자 계산을 아주 많이 동시에 할 때 빨라요.", "점 240개, 좌표 2개짜리 표가 Tensor가 됐어요. 신경망은 보통 32비트 실수로 계산해요."])
outs["s2"] = S.box("② 선 하나(nn.Linear)로 나누기, 500번 학습", [(f"valid 정확도 → {ns['linear_acc']:.3f}", 1)],
                   ["두 무리가 직선 하나로는 깔끔하게 나뉘지 않는 모양이라, 아무리 학습해도 80% 근처에서 멈춰요. 곡선 경계가 필요해요."])
outs["s3"] = S.box("③ make_net()", [str(ns["make_net"]()).replace("\n", " ")[:200], (f"가중치 수 → {ns['n_params']}개", 1)],
                   ["2 → 16 → 1. 가운데 층 16칸과 ReLU(음수는 0으로) 덕분에 경계를 구부릴 수 있어요. 가중치 65개 = (2×16 + 16) + (16×1 + 1)"])
g = ns["grad_first"]
outs["s4"] = S.box("④ 손실을 계산하고 backward() 한 번", [f"loss_before → {ns['loss_before']:.4f} (학습 전)", (f"net[0].weight.grad.shape → {tuple(g.shape)}", 1), f"grad 앞 2줄 → {[[round(float(v), 4) for v in row] for row in g[:2]]}"],
                   ["첫 층 가중치(16×2)와 같은 모양의 기울기가 생겼어요. 각 숫자는 \"이 가중치를 조금 키우면 손실이 얼마나 변하나\"예요. 이 기울기의 반대 방향으로 가중치를 조금씩 옮기는 게 학습이에요."])
outs["s5"] = S.table("⑤ Adam, 학습률 0.01, 100 epoch", ["epoch", "train 손실", "valid 손실", "valid 정확도"], hist_rows(ns["hist"], 20),
                     [f"손실이 줄면서 valid 정확도가 {ns['hist'][-1]['valid_acc']:.3f}까지 올라갔어요. 선 하나(②, {ns['linear_acc']:.3f})로는 못 넘던 벽을 넘었어요."])
outs["f_relu"] = chart("선 하나와 은닉 칸 16개가 배운 경계 (train 240개)", P_RELU, "검은 선: 경계, 분홍 바탕: 무리 1이라고 답하는 곳",
                       ["왼쪽 선 하나는 어떻게 학습해도 곧은 경계뿐이에요.",
                        "오른쪽 회색 점선 16개가 은닉 칸 16개예요. 칸마다 <code>w1×x1 + w2×x2 + b = 0</code>인 선이 하나씩 있고, 그 선 한쪽에서는 0(ReLU가 끔), 다른 쪽에서는 거리에 비례한 값을 내요. 마지막 층이 이 16개를 가중치를 곱해 더하니, 경계는 <b>여러 곧은 조각을 이어 붙인 꺾은선</b>이 돼요.",
                        "그래서 ReLU 신경망이 그리는 함수는 늘 \"조각마다 곧은\" 함수예요. 칸이 많을수록 조각이 많아져 더 매끈한 곡선을 흉내 내요."])
wd = {(hd, dp): (row[1], float(row[3])) for (hd, dp, _, _), row in zip(wd_models, wd_rows)}   # (칸, 층) → (가중치 수, valid 정확도)
S.claim(ns["linear_acc"] < wd[(2, 1)][1] < ns["linear_acc"] + 0.1 and min(wd[(16, 1)][1], wd[(64, 1)][1]) > wd[(2, 1)][1], "칸 2개는 선 하나보다 조금 낫고, 칸을 늘리면 오른다")
S.claim(wd[(4, 4)][1] > wd[(16, 1)][1] and abs(wd[(4, 4)][0] - wd[(16, 1)][0]) < 0.3 * wd[(16, 1)][0], "4칸 × 4층이 비슷한 가중치 수로 16칸 × 1층보다 정확하다")
S.claim(wd[(64, 1)][1] == wd[(16, 2)][1] == wd[(4, 4)][1], "64칸 × 1층, 16칸 × 2층, 4칸 × 4층의 정확도가 같다")
outs["t3"] = S.table("은닉 칸 수와 층 수를 바꾸면 (Adam, 학습률 0.01, 100 epoch)", ["모양", "가중치 수", "train 손실", "valid 정확도"], wd_rows,
                     [f"칸 2개면 꺾을 수 있는 선이 두 개뿐이라 선 하나({ns['linear_acc']:.2f})보다 조금 나은 정도예요. 4개, 16개, 64개로 늘리면 정확도가 올라가요.",
                      f"<b>4칸 × 4층</b>은 가중치가 {wd[(4, 4)][0]}개로 16칸 × 1층({wd[(16, 1)][0]}개)과 비슷한데 정확도가 더 높아요({wd[(4, 4)][1]:.3f} 대 {wd[(16, 1)][1]:.3f}). 층을 쌓으면 앞 층이 만든 꺾은 면을 다음 층이 다시 접어서, 같은 가중치 수로 더 많은 조각을 만들 수 있어요.",
                      f"64칸 × 1층, 16칸 × 2층은 가중치가 {wd[(64, 1)][0] // wd[(4, 4)][0]}~{wd[(16, 2)][0] // wd[(4, 4)][0]}배 많지만 4칸 × 4층과 같은 {wd[(4, 4)][1]:.3f}이에요. 이 데이터에는 이 정도면 충분해서 더 키워도 나아지지 않아요. 깊이와 폭은 문제의 어려움, 데이터 양, 학습이 잘 되는지를 함께 보고 정해요."])
outs["f_wd"] = chart("모양별 경계 (같은 학습 조건)", P_WD, "", ["칸이 적으면 경계가 몇 개의 곧은 조각뿐이에요. 4칸 × 4층은 칸이 적어도 층을 쌓아 휜 경계를 만들었어요."])
outs["f_init"] = chart("가중치를 어떤 크기로 시작하느냐 (64칸 × 20층, 무작위 입력 256개)", P_INIT, "",
                       ["점 예제가 아니라, 초기화의 효과만 보려고 무작위 입력을 20층에 통과시킨 실험이에요.",
                        f"가중치를 작게(0.01) 시작하면 층마다 값이 약 1/{1 / fac['small']:.0f}로 줄어서 20층 뒤에는 {init_std_last['small']:.0e}이 돼요. 거꾸로 흐르는 기울기도 같이 사라져서 앞쪽 층이 배우지 못해요(기울기 소실).",
                        f"크게(1) 시작하면 층마다 약 {fac['big']:.1f}배씩 커져 {init_std_last['big']:.0e}까지 폭발해요(기울기 폭발).",
                        "He 초기화는 표준편차를 √(2 ÷ 입력 칸 수)로 맞춰서, ReLU가 절반을 0으로 만드는 걸 감안해도 층마다 크기가 거의 유지돼요. PyTorch의 <code>nn.Linear</code>도 기본으로 이와 비슷한 방식(Kaiming 균등분포)으로 시작해요."])
flow1 = [[1, "Tensor", "표 → Tensor", "GPU 확인"], [2, "선 하나", "nn.Linear", "곡선을 못 그림"], [3, "신경망", "Linear, ReLU", f"가중치 {wd[(16, 1)][0]}개"], [4, "역전파", "backward()", "기울기"], [5, "학습 루프", "batch, epoch", f"valid {ns['hist'][-1]['valid_acc']:.3f}"]]
S.save(ROOT / "topics" / "model" / "dl-tensor-training-data.js", {"code": [c for c in code if c["s"] <= 5], "names": S.names(code), "range": [1, 5], "outs": outs, "flow": flow1, "env": ENV})
why1 = f"""<p>점들을 두 무리로 나누는 문제예요.</p>{K.out("data")}
<p>머신러닝 모델(<a href="ml-regression-classification.html">회귀와 분류</a>)은 사람이 준 입력 열을 그대로 써요. 경계가 복잡하면 사람이 \"x1의 제곱\", \"두 값의 곱\" 같은 새 열을 만들어 줘야 해요. <b>신경망</b>은 층을 쌓아 이런 표현을 <b>스스로 배워요</b>. 그래서 사진, 소리, 글처럼 사람이 열을 만들기 어려운 데이터에 강해요.</p>
<p>신경망도 <a href="ml-regression-classification.html">회귀와 분류</a>에서 본 지도 학습의 네 가지(모양, 비용, 찾기, 예측)를 그대로 따라요. 다른 건 <b>모양</b>이에요. 신경망은 \"곱해 더하고 꺾기\"를 여러 층 겹친 함수예요. 층 하나하나는 단순하지만, 겹치면 아주 복잡한 함수를 만들 수 있어요.</p>
<p>여기서는 파이썬 딥러닝 라이브러리 <b>PyTorch</b>로, 데이터를 숫자 배열(<b>Tensor</b>)로 바꾸고, 신경망을 만들고, 학습시키는 과정을 처음부터 따라가요.</p>"""
whole1 = """<p>데이터를 Tensor로 → 선 하나로는 안 되는 걸 확인 → 층을 쌓은 신경망 → 기울기 계산 → 반복 학습. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p><code>points_net.py</code> 하나가 이 쪽(①~⑤)에서 시작해 <a href="dl-training-stability.html">2쪽</a>(⑥~⑪)까지 자라요.</p>"""
steps1 = [
    K.step(1, "GPU 확인하고 Tensor로 바꾸기", "표를 PyTorch의 숫자 배열인 Tensor로 바꿔요.",
           "<p><b>Tensor</b>는 여러 차원의 숫자 배열이에요. numpy 배열과 비슷하지만, GPU에서 계산할 수 있고 기울기를 자동으로 계산할 수 있어요.</p>"
           + K.code_table([("torch.cuda.is_available()", "GPU를 쓸 수 있나"), ("torch.tensor(values, dtype=torch.float32)", "배열을 32비트 실수 Tensor로"),
                           (".unsqueeze(1)", "(240,) 모양을 (240, 1)로. 모델 출력 모양과 맞추려고"), (".to(device)", "Tensor를 CPU나 GPU로 옮기기. 모델과 데이터가 같은 곳에 있어야 계산돼요.")]), "s1"),
    K.step(2, "선 하나로 나눠 보기", "가장 단순한 모델 <code>nn.Linear(2, 1)</code>로 먼저 해 봐요. 기준선이에요.",
           K.code_table([("nn.Linear(2, 1)", "w1×x1 + w2×x2 + b. 경계는 언제나 직선"), ("binary_cross_entropy_with_logits", "두 종류 분류의 손실. 틀릴수록, 틀린 걸 확신할수록 커요. logit은 시그모이드를 거치기 전 점수예요 (④에서 자세히)."),
                         ("opt.zero_grad(); loss.backward(); opt.step()", "기울기 지우기 → 계산 → 가중치 옮기기. 학습 한 번"), ("with torch.no_grad()", "채점할 때는 기울기를 계산하지 않아 빨라요.")]), "s2"),
    K.step(3, "신경망 만들기", "Linear와 ReLU를 쌓아 경계를 구부릴 수 있는 모델을 만들어요.",
           "<p>가운데 층의 칸 하나하나를 <b>은닉 칸</b>(hidden unit)이라고 해요. 칸마다 입력에 가중치를 곱해 더한 뒤 <b>ReLU</b>를 거쳐요. ReLU는 음수를 0으로 바꾸는 함수라, 칸 하나는 \"어떤 직선의 한쪽에서는 꺼져 있고, 다른 쪽에서는 멀어질수록 커지는\" 꺾인 면을 만들어요. 마지막 층이 이 꺾인 면들을 가중치를 곱해 더하면, 곧은 조각들을 이어 붙인 꺾은 경계가 나와요(아래 그림).</p>"
           "<p><b>몇 칸이면 될까?</b> 은닉 칸만 충분하면 층 하나짜리 신경망으로도 어떤 연속 함수든 원하는 만큼 가깝게 흉내 낼 수 있다는 게 알려져 있어요(<b>보편 근사 정리</b>). 하지만 \"흉내 낼 수 있다\"일 뿐, 그 가중치를 학습으로 찾을 수 있는지, 처음 보는 데이터에서도 맞는지는 말해 주지 않아요. 그리고 필요한 칸 수가 엄청나게 많을 수 있어요.</p>"
           "<p><b>왜 층을 쌓나?</b> 층을 쌓으면 다음 층이 앞 층의 출력을 다시 꺾어요. 종이를 한 번 접으면 두 겹, 다시 접으면 네 겹이 되듯, 같은 가중치 수로 만들 수 있는 곧은 조각 수가 층마다 곱으로 늘어요. 그래서 깊은 신경망은 같은 크기의 얕은 신경망보다 복잡한 함수를 효율적으로 표현할 수 있어요. 폭(칸 수)과 깊이(층 수)는 사람이 정하는 설정(하이퍼파라미터)이에요.</p>"
           + K.out("f_relu")
           + K.code_table([("nn.Sequential(...)", "층을 순서대로 이어 붙인 모델"), ("nn.ReLU()", "음수는 0, 양수는 그대로. 이게 없으면 Linear를 몇 개 쌓아도 결국 직선이에요. Linear 두 개를 이으면 \"곱해 더하기\"를 두 번 한 것이라 다시 곱해 더하기 하나와 같거든요."),
                           ("nn.Dropout(dropout)", "2쪽 ⑨에서 써요. 0이면 아무것도 안 해요."), ("p.numel()", "가중치 Tensor 하나의 숫자 개수")])
           + K.try_("은닉 칸을 2, 4, 64개로, 층을 2개, 4개로 바꾸면?", "t3", after=K.out("f_wd")), "s3"),
    K.step(4, "손실과 역전파", "예측이 얼마나 틀렸는지(손실)를 계산하고, <code>backward()</code>로 모든 가중치의 기울기를 구해요.",
           "<p><b>손실은 어떻게 고르나?</b> \"정답이 어떤 확률분포에서 나온다고 볼까\"를 정하면 손실이 따라 나와요. 모델이 낸 값을 그 분포의 모양을 정하는 값(모수, 예를 들어 평균)으로 보고, train 정답이 나올 확률을 가장 크게 하는 것(최대우도)이 곧 −log(확률)의 평균을 가장 작게 하는 거예요. 숫자를 맞히면서 오차가 정규분포라고 보면 제곱 오차(MSE), 둘 중 하나면 베르누이 분포(동전처럼 둘 중 하나가 나오는 분포)라 이진 교차 엔트로피, 여럿 중 하나면 교차 엔트로피가 나와요. <a href=\"ml-regression-classification.html\">회귀와 분류</a> ⑤의 로그 손실이 바로 두 번째예요.</p>"
           "<p>모델의 마지막 출력은 시그모이드를 거치기 전 점수(<b>logit</b>)예요. <code>BCEWithLogitsLoss</code>는 시그모이드와 로그 손실을 한 번에 계산해서, 확률이 0이나 1에 아주 가까울 때 생기는 계산 오차를 피해요.</p>"
           "<p><b>역전파</b>: 신경망은 층이 여러 개라 \"앞쪽 가중치가 손실에 얼마나 영향을 줬나\"를 바로 알기 어려워요. 미분의 연쇄 법칙을 쓰면, 손실에 대한 앞 층의 기울기는 (뒤 층의 기울기) × (그 층 안의 기울기)예요. 그래서 앞으로 계산할 때(순전파) 층마다 중간값을 저장해 두고, 출력에서부터 거꾸로 한 층씩 곱해 내려가요. 같은 중간값을 여러 번 다시 쓰니 계산이 빨라요. PyTorch는 계산 과정을 그래프로 기억해 두었다가 <code>backward()</code> 한 번에 이걸 자동으로 해 줘요(<b>자동 미분</b>).</p>"
           "<p><b>가중치는 어떤 값에서 시작하나?</b> 층마다 \"곱해 더하기\"를 하니, 가중치가 조금만 작아도 값이 층마다 줄어들고, 조금만 커도 불어나요. 20층이면 그게 20번 곱해져요. 거꾸로 흐르는 기울기도 같은 곱을 거치니 같이 사라지거나(<b>기울기 소실</b>) 폭발해요(<b>기울기 폭발</b>). 그래서 입력 칸 수에 맞춰 시작 크기를 정해요. ReLU에는 <b>He 초기화</b>(표준편차 √(2 ÷ 입력 칸 수)), tanh 같은 대칭 함수에는 <b>Xavier 초기화</b>를 써요.</p>"
           + K.out("f_init")
           + K.code_table([("nn.BCEWithLogitsLoss()", "②와 같은 손실을 층(객체)으로"), ("loss.backward()", "손실에서 모든 가중치까지 기울기를 계산해 <code>.grad</code>에 넣어요."), ("net.zero_grad()", "기울기는 쌓이기 때문에 다음 계산 전에 지워요."),
                           ("nn.init.kaiming_normal_(w)", "He 초기화로 가중치 다시 채우기 (그림 실험에서 사용)")]), "s4"),
    K.step(5, "학습 루프", "데이터를 32개씩 나눠(batch) 가중치를 고치고, 이걸 전체 데이터로 100바퀴(epoch) 반복해요.",
           "<p>기울기를 train 240개 전부로 계산하면 정확하지만 느려요. 그래서 32개씩 꺼낸 묶음(<b>미니배치</b>)으로 기울기를 어림해서 바로 한 걸음 가요. 이게 <b>확률적 경사하강법</b>(SGD)이에요. 묶음마다 기울기가 조금씩 달라서 길이 흔들리는데, 이 흔들림이 오히려 좁은 골짜기에 갇히지 않게 도와주기도 해요. 묶음이 작을수록 한 걸음은 싸지만 더 흔들려요.</p>"
           + K.code_table([("DataLoader(..., batch_size=32, shuffle=True)", "32개씩 꺼내 주고, 바퀴마다 순서를 섞어요."), ("torch.optim.Adam(params, lr=0.01)", "기울기로 가중치를 옮기는 방법(Optimizer). 학습률(lr)은 한 번에 옮기는 크기. Adam이 어떻게 옮기는지는 2쪽 ⑦에서 봐요."),
                         ("model.train() / model.eval()", "학습 중과 채점 중을 구분해요. Dropout 같은 층이 다르게 동작해요."), ("history.append({...})", "바퀴마다 손실과 정확도를 기록. 2쪽에서 이 기록으로 문제를 진단해요.")]), "s5"),
]
practice1 = [("GPU로 학습한다", "모델과 데이터를 모두 <code>.to(\"cuda\")</code>로 옮겨요. 하나라도 CPU에 있으면 \"device가 다르다\" 오류가 나요. 결과는 GPU에서 매번 조금씩 다를 수 있어요."),
             ("Tensor 모양 오류", "대부분 <code>(N,)</code>과 <code>(N, 1)</code>처럼 모양이 어긋나서 생겨요. 의심되면 <code>.shape</code>를 찍어 봐요."),
             ("손실이 줄지 않는다", "학습률이 너무 작거나 크거나, <code>zero_grad()</code>를 빠뜨렸을 수 있어요. 2쪽 ⑥에서 학습률의 영향을 봐요."),
             ("손실이 nan이 된다", "기울기 폭발이에요. 학습률을 줄이고, 초기화를 확인하고, 그래도 안 되면 기울기 크기에 상한을 둬요 (<code>torch.nn.utils.clip_grad_norm_</code>)."),
             ("층을 아주 깊게 쌓고 싶다", "초기화만으로는 수십 층 넘게 안정시키기 어려워요. 층을 건너뛰는 길(잔차 연결)과 정규화 층을 같이 써요 (<a href=\"cnn.html\">CNN</a>의 ResNet)."),
             ("숫자를 맞히는 신경망", "마지막 층을 그대로 두고 <code>nn.MSELoss()</code>를 써요(오차를 정규분포로 본 것). 이상치가 많으면 <code>nn.L1Loss()</code>나 <code>nn.HuberLoss()</code>가 덜 흔들려요."),
             ("종류가 셋 이상", "마지막 층을 종류 수만큼 칸으로 두고 <code>nn.CrossEntropyLoss()</code>를 써요. 안에서 softmax를 같이 계산하니 모델에는 softmax를 넣지 않아요.")]
libs1 = [("①", "<code>torch.tensor()</code>, <code>.to()</code>, <code>.unsqueeze()</code>, <code>torch.cuda.is_available()</code>", "Tensor (torch)"), ("②③", "<code>nn.Linear</code>, <code>nn.ReLU</code>, <code>nn.Sequential</code>", "층과 모델 (torch.nn)"),
         ("④", "<code>nn.BCEWithLogitsLoss</code>, <code>loss.backward()</code>, <code>.grad</code>", "손실과 역전파"), ("⑤", "<code>DataLoader</code>, <code>TensorDataset</code>, <code>torch.optim.Adam</code>, <code>torch.no_grad()</code>", "학습 루프")]
K.build(ROOT / "topics" / "model" / "dl-tensor-training.html", title="Tensor와 신경망 학습", h1="Tensor와 신경망 학습: 층을 쌓아 경계를 배우기", track="모델",
        crumb="모델 &gt; 딥러닝 기초 1/2쪽 &nbsp;|&nbsp; 다음: <a href=\"dl-training-stability.html\" style=\"color:inherit\">학습을 안정시키기</a>",
        one_line="<b>신경망</b>은 Linear 층과 ReLU를 쌓아, 사람이 열을 만들어 주지 않아도 <b>복잡한 경계를 스스로 배워요</b>. 학습은 \"손실 계산 → 역전파로 기울기 → 가중치 조금 옮기기\"의 반복이에요.",
        why=why1, whole=whole1, steps=steps1, code_file="examples/dl/points_net.py", practice=practice1, libs=libs1, libs_note="<code>tensors</code>, <code>accuracy</code>, <code>make_net</code>, <code>train</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"dl-training-stability.html\">학습을 안정시키기</a>", "학습률, Optimizer, 과적합"), ("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "신경망 이전의 모델과 평가")],
        sources=["다루는 범위: Day09 (2026-10-29) GPU, Tensor, ANN, 역전파",
                 "함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 2~5장, 7장. 지도 학습의 틀, 얕은 신경망과 꺾은선 함수, 보편 근사 정리, 깊이의 효과, 최대우도로 손실 고르기, 역전파와 자동 미분, 초기화와 기울기 소실을 가져왔어요. 값과 그림은 이 예제로 다시 계산했어요",
                 SRC_DATA, "결과 값과 그림: <code>tools/pages/dl_basics.py</code>가 단계별로 실제 실행한 결과 (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/tutorials/beginner/basics/intro.html\" target=\"_blank\">PyTorch 기본 익히기</a>, <a href=\"https://docs.pytorch.org/docs/stable/nn.html\" target=\"_blank\">torch.nn</a>"])

# =============================== 2쪽 ===============================
outs = {}
_ov = ns["over"]
S.claim(_ov[-1]["train_loss"] < 0.01 and _ov[-1]["valid_loss"] > 2 * min(h_["valid_loss"] for h_ in _ov), "⑧ train 손실은 0, valid 손실은 계속 커진다")
outs["why"] = S.table("⑧에서 만든 큰 신경망 (학습 데이터 40개로 300 epoch)", ["epoch", "train 손실", "valid 손실", "valid 정확도"], hist_rows(ns["over"], 50),
                      ["train 손실은 0이 됐는데 valid 손실은 오히려 계속 커져요. 학습 데이터를 <b>외워 버린</b> 거예요 (과적합). 외운 모델은 새 데이터에서 틀린 답을 아주 확신해요.",
                       "학습이 잘되는지는 손실 하나로 알 수 없어요. 학습률, Optimizer, 모델 크기, 데이터 양이 모두 영향을 줘요. 이 쪽에서는 그걸 하나씩 실험해요."])
_lr = ns["lr_runs"]
S.claim(max(_lr, key=lambda k: _lr[k][-1]["valid_acc"]) == 0.1 and _lr[0.001][-1]["train_loss"] > 0.9 * _lr[0.001][0]["train_loss"] and _lr[10.0][-1]["train_loss"] > _lr[0.1][-1]["train_loss"],
        "⑥ 0.001은 거의 그대로, 10은 못 내려가고, 0.1이 가장 낫다")
_op = ns["opt_runs"]
S.claim(_op["adam"][9]["valid_acc"] > _op["sgd"][9]["valid_acc"] and _op["adam"][-1]["train_loss"] < _op["sgd"][-1]["train_loss"], "⑦ 같은 학습률에서 Adam이 SGD보다 빨리 내려간다")
S.claim(sum(p_.numel() for p_ in ns["big"].parameters()) > 30000, "⑧ 가중치가 3만 개 넘는다")
S.claim(abs(ns["regd"][-1]["valid_acc"] - ns["over"][-1]["valid_acc"]) < 0.05, "⑨ 규제해도 정확도 차이는 크지 않다")
outs["s6"] = S.table("⑥ 같은 신경망, SGD, 학습률만 다르게 (100 epoch)", ["학습률", "train 손실", "valid 정확도"], [[lr, f3(h[-1]["train_loss"]), f"{h[-1]['valid_acc']:.3f}"] for lr, h in ns["lr_runs"].items()],
                     ["0.001은 너무 조금씩 움직여서 100바퀴가 지나도 거의 그대로예요.", "10은 너무 크게 움직여서 최저점을 지나쳐 버려요. 0.1이 이 셋 중 가장 나아요. 적당한 학습률은 문제마다 달라서 몇 개를 시험해 봐요."])
outs["s7"] = S.table("⑦ 학습률 0.01로 SGD와 Adam (60 epoch)", ["Optimizer", "10 epoch valid 정확도", "60 epoch valid 정확도", "60 epoch train 손실"],
                     [[n, f"{h[9]['valid_acc']:.3f}", f"{h[-1]['valid_acc']:.3f}", f3(h[-1]["train_loss"])] for n, h in ns["opt_runs"].items()],
                     ["같은 학습률이어도 Adam이 훨씬 빨리 내려가요. Adam은 가중치마다 지금까지의 기울기를 보고 옮기는 크기를 알아서 조절해요. 그래서 처음 시작할 때 무난한 선택이에요."])
outs["s8"] = S.box("⑧ 과적합: 데이터 40개, 128칸 3층", [f"가중치 수 → {sum(p.numel() for p in ns['big'].parameters()):,}개", (f"300 epoch 뒤 train 손실 {ns['over'][-1]['train_loss']:.6f}, valid 손실 {ns['over'][-1]['valid_loss']:.3f}", 1, "bad")],
                   ["데이터 40개에 가중치 3만 개가 넘어요. 외우기에 충분하고도 남아요."])
outs["s9"] = S.table("⑨ 같은 조건에 Dropout 0.3 + weight decay", ["epoch", "train 손실", "valid 손실", "valid 정확도"], hist_rows(ns["regd"], 50),
                     [f"300 epoch 뒤 valid 손실이 {ns['over'][-1]['valid_loss']:.3f} → {ns['regd'][-1]['valid_loss']:.3f}로 줄었어요. 틀린 답에 대한 지나친 확신이 줄었다는 뜻이에요.",
                      "정확도 차이는 크지 않아요. 규제는 \"맞히는 개수\"보다 \"얼마나 믿을 만한 확률을 내나\"에 먼저 효과가 보이기도 해요."])
be = min(ns["early"], key=lambda x: x["valid_loss"])
_ta = ns["test_acc"]
S.claim(be["epoch"] <= 20, "⑩ 가장 좋았던 때가 아주 이르다")
S.claim(max(_ta, key=_ta.get) == "기본 신경망 (⑤)" and max(_ta["규제 (⑨)"], _ta["조기 종료 (⑩)"]) <= _ta["과적합 (⑧)"], "⑪ 기본 신경망이 가장 좋고, 규제와 조기 종료는 정확도를 올리지 못했다")
outs["s10"] = S.box("⑩ 조기 종료 (patience 20)", [(f"멈춘 epoch → {len(ns['early'])}", 1), (f"가장 좋았던 epoch → {be['epoch']} (valid 손실 {be['valid_loss']:.3f}, 정확도 {be['valid_acc']:.3f})", 2)],
                    ["valid 손실이 20번 연속 나아지지 않아서 멈췄어요. 300번을 다 돌지 않아 시간도 아껴요.", "마지막이 아니라 가장 좋았던 때의 가중치로 되돌렸어요. 데이터가 40개뿐이라 아주 이른 때가 가장 좋았어요."])
outs["s11"] = S.table("⑪ test 80개 정확도 (처음 사용)", ["모델", "test 정확도"], [[E(k), f"{v:.3f}"] for k, v in ns["test_acc"].items()],
                      ["데이터 240개를 다 쓴 기본 신경망이 가장 좋아요. 과적합을 막는 기법도 중요하지만, <b>데이터가 많은 것</b>을 이기기는 어려워요.", "⑧~⑩은 일부러 40개만 써서 과적합을 만든 실험이에요. 규제와 조기 종료는 손실(확신의 정도)을 안정시켰지만, 이 작은 test에서 정확도까지 올리진 못했어요."])
outs["f_opt"] = chart("epoch마다 train 손실", P_OPT, "", ["왼쪽: 학습률 0.001은 선이 거의 수평이고, 10은 손실이 내려가지 못하고 높은 곳에서 출렁여요. 0.1이 가장 빨리 내려가요.",
                                                     "오른쪽: 같은 학습률 0.01인데 고치는 방법만 바꿨어요. 모멘텀을 더하면 SGD보다 훨씬 빨리 내려가고, Adam은 그보다 더 빨라요."])
opt_rows = [[nm, f"{h[9]['valid_acc']:.3f}", f"{h[-1]['valid_acc']:.3f}", f3(h[-1]["train_loss"])] for nm, h in [("SGD", h_sgd60), ("SGD + 모멘텀 0.9", h_mom), ("Adam", h_adam60)]]
S.claim(h_adam60[-1]["train_loss"] < h_mom[-1]["train_loss"] < 0.8 * h_sgd60[-1]["train_loss"], "모멘텀이 SGD보다 train 손실을 줄이고, Adam은 그보다 더 낮다")
outs["t7"] = S.table("SGD에 모멘텀을 더하면 (학습률 0.01, 60 epoch)", ["Optimizer", "10 epoch valid 정확도", "60 epoch valid 정확도", "60 epoch train 손실"], opt_rows,
                     [f"모멘텀은 SGD와 Adam 사이예요. \"지난 걸음 방향을 90% 이어 가기\" 하나만 더했는데 train 손실이 {h_sgd60[-1]['train_loss']:.3f}에서 {h_mom[-1]['train_loss']:.3f}로 줄었어요.",
                      "Adam은 모멘텀(기울기의 평균)에 더해 가중치마다 기울기 크기의 평균으로 걸음을 나눠서, 이 문제처럼 학습률이 작게 잡혀 있어도 알아서 성큼 가요."])
outs["f_over"] = chart("⑧⑨⑩의 손실 곡선 (train 40개)", P_OVER, "점선: train, 실선: valid",
                       ["분홍 점선(과적합 train)은 0으로 붙는데, 분홍 실선(valid)은 어느 순간부터 계속 올라가요. 둘이 벌어지기 시작한 곳부터가 외우는 구간이에요.",
                        "보라(규제)도 train 손실은 0 근처까지 내려가지만, Dropout이 칸을 무작위로 꺼서 가끔 튀어요. valid는 크게 출렁이면서도 대부분 분홍 실선보다 아래에 있어요. 끝까지 계속 올라가지는 않아요.", f"민트 점이 조기 종료가 고른 가장 좋던 때예요(epoch {be['epoch']}). 데이터가 40개뿐이라 외우기 시작하는 게 아주 빨라요."])
outs["f_wh"] = chart("⑧과 ⑨가 배운 가중치의 분포", P_WH, "",
                     ["weight decay는 걸음마다 가중치를 조금씩 0 쪽으로 당겨요. 그래서 큰 가중치가 줄고 0 근처에 많이 모였어요. 가중치가 작으면 입력이 조금 바뀌어도 출력이 크게 흔들리지 않아, 덜 외운 매끈한 함수가 돼요."])
flow2 = [[6, "학습률", "0.001, 0.1, 10", "너무 작거나 크면"], [7, "Optimizer", "SGD vs Adam", "Adam이 빠름"], [8, "과적합", "40개로 외우기", "train 0, valid ↑"], [9, "규제", "Dropout, decay", "확신 줄이기"], [10, "조기 종료", "가장 좋던 때로", "patience"], [11, "test", "한 번", "데이터가 이김"]]
S.save(ROOT / "topics" / "model" / "dl-training-stability-data.js", {"code": code, "names": S.names(code), "range": [6, 11], "outs": outs, "flow": flow2, "env": ENV})
why2 = f"""<p><a href="dl-tensor-training.html">1쪽</a>에서는 학습이 순조로웠어요. 하지만 조건이 조금만 달라도 학습은 쉽게 어긋나요.</p>{K.out("why")}"""
whole2 = """<p>학습률과 Optimizer로 \"잘 내려가게\" 하고, 과적합을 일부러 만든 뒤 규제와 조기 종료로 다스리고, 마지막에 test로 한 번 채점해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>1쪽의 코드(①~⑤)에 이어서 같은 파일이 자라요.</p>"""
steps2 = [
    K.step(6, "학습률 비교", "같은 모델을 학습률만 바꿔 학습해요.",
           "<p>한 걸음은 <code>가중치 ← 가중치 − 학습률 × 기울기</code>예요(<a href=\"ml-regression-classification.html\">회귀와 분류</a> ④의 경사하강법과 같아요). 학습률이 작으면 안전하지만 한참 걸리고, 크면 골짜기 바닥을 지나쳐 반대편으로 튀어요. 신경망의 손실은 그릇 하나가 아니라 울퉁불퉁한 산이라, 알맞은 학습률은 실험으로 찾아요.</p>"
           "<p>학습률을 처음부터 끝까지 같게 두지 않기도 해요(<b>스케줄</b>). 처음 몇 바퀴는 작게 시작해 키우고(warmup), 그 뒤로 점점 줄여요(단계별로, 또는 코사인 곡선을 따라). 처음엔 크게 움직여 넓게 찾고, 끝에서는 작게 움직여 바닥에 자리 잡는 거예요.</p>"
           + K.code_table([("run(lr, opt_name=\"sgd\")", "seed를 같게 해서 학습률만 다른 조건을 만들어요."), ("torch.optim.SGD", "기울기 × 학습률만큼 그대로 옮기는 가장 기본 방법"),
                           ("torch.optim.lr_scheduler", "학습률 스케줄 (이 예제에서는 안 써요)")]), "s6"),
    K.step(7, "Optimizer 비교", "같은 학습률로 SGD와 Adam을 비교해요.",
           "<p>SGD는 미니배치마다 기울기가 흔들려서, 좁고 긴 골짜기에서는 양쪽 벽을 오가며 지그재그로 내려가요. <b>모멘텀</b>은 지난 걸음들의 방향을 굴러가는 공처럼 이어 가요. <code>속도 ← 0.9 × 속도 + 기울기</code>, <code>가중치 ← 가중치 − 학습률 × 속도</code>. 늘 같은 방향인 성분은 점점 빨라지고, 왔다 갔다 하는 성분은 서로 지워져요. Nesterov 모멘텀은 \"이미 갈 곳\"에서 기울기를 재서 조금 더 일찍 브레이크를 밟아요.</p>"
           "<p><b>Adam</b>은 두 가지를 기억해요. 기울기의 평균(모멘텀과 같은 것)과, 기울기 제곱의 평균(그 가중치가 평소 얼마나 크게 흔들리나)이에요. 그리고 평균을 크기로 나눠서 걸음을 정해요. 기울기가 늘 작은 가중치는 크게, 늘 큰 가중치는 작게 움직여서, 가중치마다 학습률을 따로 쓰는 셈이에요. 처음 몇 걸음은 평균이 0에서 시작해 작게 나오니 그만큼 보정해요.</p>"
           "<p>Optimizer 이름보다 학습률과 스케줄이 결과를 더 크게 바꾸는 경우가 많아요. 그리고 train 손실이 빨리 줄어드는 것과 처음 보는 데이터에서 잘하는 건 다른 문제예요.</p>"
           + K.out("f_opt")
           + K.code_table([("torch.optim.Adam", "기울기의 평균과 크기를 기억해서 가중치마다 보폭을 조절"), ("SGD(..., momentum=0.9)", "지난 방향을 90% 이어 가기 (그림과 🔁에서 사용)")])
           + K.try_("SGD에 모멘텀만 더하면?", "t7"), "s7"),
    K.step(8, "과적합 만들기", "데이터는 40개로 줄이고, 모델은 128칸 3층으로 크게 해서 300바퀴 돌려요.",
           "<p>처음 보는 데이터에서의 오차는 세 가지로 나눠 볼 수 있어요(<a href=\"ml-regression-classification.html\">회귀와 분류</a> ⑥의 편향과 분산). 데이터 자체의 잡음(어떤 모델로도 못 줄임), <b>편향</b>(모델이 너무 단순해서 생기는 오차), <b>분산</b>(train이 조금만 바뀌어도 모델이 크게 바뀌는 오차)이에요. 모델을 키우면 편향이 줄고 분산이 늘고, 데이터를 늘리면 분산이 줄어요. 이 쪽에서는 데이터를 40개로 줄이고 모델을 키워서 분산을 일부러 크게 만들어요.</p>"
           "<p>⚠️ 옛 교과서에는 \"모델이 커질수록 valid 오차가 U자로 다시 오른다\"고 나와요. 그런데 신경망을 데이터를 완전히 외울 수 있는 크기보다 <b>훨씬</b> 크게 만들면, valid 오차가 한 번 올랐다가 다시 내려가는 일이 자주 관찰돼요(<b>이중 하강</b>). 가중치가 데이터보다 많은데도 신경망이 일반화하는 이유는 아직 하나의 이론으로 다 설명되지 않아요. 확률적 경사하강의 흔들림이 넓고 평탄한 골짜기를 고르기 쉽다는 설명, 초기화와 잔차 연결(층을 건너뛰는 길, <a href=\"cnn.html\">CNN</a>의 ResNet)이 학습하기 쉬운 길을 만든다는 설명들이 함께 쓰여요.</p>"
           + K.code_table([("X_train[:40]", "학습 데이터를 앞 40개만"), ("make_net(hidden=128, depth=3)", "큰 모델")]), "s8"),
    K.step(9, "규제: Dropout과 weight decay", "학습 중에 일부 칸을 무작위로 끄고(Dropout), 가중치가 커지지 않게 벌점을 줘요(weight decay).",
           "<p>둘 다 모델이 몇몇 가중치나 칸에 기대어 답을 외우지 못하게 해요.</p>"
           "<p><b>weight decay</b>는 1쪽 회귀의 Ridge(L2 규제)와 같은 생각이에요. 가중치 제곱의 합에 벌점을 줘서 가중치를 작게 유지해요. 다만 Adam에서는 벌점을 손실에 더하는 방식과 가중치에서 직접 빼는 방식이 결과가 달라서, 직접 빼는 쪽을 따로 <code>AdamW</code>라고 불러요. 요즘은 AdamW를 많이 써요.</p>"
           "<p><b>Dropout</b>은 학습 때마다 칸을 무작위로 꺼서, 매번 조금씩 다른 작은 신경망을 학습시키는 셈이에요. 채점할 때는 다 켜니까, 수많은 작은 신경망을 평균 내는 앙상블(<a href=\"ml-trees-cv.html\">트리와 앙상블</a>)과 비슷한 효과가 나요. 특정 칸 하나에 기대는 것도 막아요.</p>"
           "<p>규제는 이 둘만이 아니에요. 입력이나 가중치에 일부러 잡음을 섞기, 그림을 뒤집고 자르는 <b>데이터 증강</b>(<a href=\"opencv-preprocessing.html\">OpenCV 전처리</a>), 큰 데이터로 미리 배운 모델에서 시작하는 <b>전이 학습</b>(<a href=\"transfer-learning.html\">전이학습</a>), 여러 모델 평균 내기도 모두 규제예요. 그리고 명시적인 벌점 없이도, SGD의 흔들림과 ⑩의 조기 종료 자체가 \"덜 복잡한 해\"를 고르게 하는 효과(암묵적 규제)가 있어요.</p>"
           + K.code_table([("nn.Dropout(0.3)", "학습 때 칸의 30%를 무작위로 0으로. 남은 칸은 1 ÷ 0.7배 키워서 합의 크기를 맞춰요. <code>eval()</code>에서는 꺼져요."), ("Adam(..., weight_decay=1e-3)", "가중치를 조금씩 0 쪽으로 당겨요.")])
           + K.out("f_wh"), "s9"),
    K.step(10, "조기 종료", "바퀴마다 valid 손실을 보고, 더 나아지지 않으면 멈춘 뒤 가장 좋았던 가중치로 되돌려요.",
           "<p>학습 초반에는 가중치가 아직 작고 함수가 매끈해요. 오래 학습할수록 train의 세세한 점까지 따라가요. 그러니 \"언제 멈추나\"가 곧 \"얼마나 복잡한 함수를 허락하나\"예요. 조기 종료는 벌점을 따로 넣지 않은 규제예요.</p>"
           + K.out("f_over")
           + K.code_table([("copy.deepcopy(model.state_dict())", "그때의 가중치를 통째로 복사해 둬요."), ("patience", "몇 번 연속 나아지지 않으면 멈출지"), ("model.load_state_dict(best_state)", "가장 좋았던 가중치로 되돌리기")]), "s10"),
    K.step(11, "마지막 한 번: test", "아껴 둔 test 80개로 모델들을 한 번 채점해요.", K.code_table([("tensors(\"test\")", "1쪽 ①의 함수로 test도 Tensor로")]), "s11"),
]
practice2 = [("학습률을 정할 때", "0.1, 0.01, 0.001처럼 10배씩 바꿔 몇 바퀴만 돌려 보고, 손실이 가장 잘 내려가는 쪽을 골라요. 학습 중에 점점 줄이는 방법(스케줄러)도 있어요."),
             ("BatchNorm", "층마다 값의 평균과 크기를 맞춰 학습을 안정시키는 층이에요 (<code>nn.BatchNorm1d</code>). Dropout처럼 <code>train()</code>과 <code>eval()</code>에서 다르게 동작해요."),
             ("과적합이 의심될 때", "train 손실과 valid 손실을 같이 그려 봐요. 둘이 벌어지기 시작하면 과적합이에요. 먼저 데이터를 늘릴 수 있는지 봐요."),
             ("미니배치 크기", "크면 기울기가 정확해 한 걸음이 안정되지만 메모리를 많이 쓰고, 흔들림이 줄어 오히려 덜 일반화하기도 해요. 배치를 k배 키우면 학습률도 비슷하게 키우는 게 출발점이에요."),
             ("test로 여러 번 고르면", "test 점수를 보고 설정을 바꾸고 다시 재기를 반복하면, test에도 맞춰 고른 셈이라 더는 독립적인 성적이 아니에요. 고르는 일은 valid나 교차검증으로 해요."),
             ("배포한 뒤 성능이 떨어진다", "실제로 들어오는 데이터가 train과 달라지는 일(데이터 드리프트)이 흔해요. test 점수는 \"그때 그 데이터\" 기준이라, 배포 뒤에도 입력 분포와 성능을 계속 지켜봐요."),
             ("큰 모델을 작게 쓰고 싶다", "학습한 뒤 영향이 작은 가중치를 지우거나(가지치기), 큰 모델의 출력을 작은 모델이 따라 배우게 해요(지식 증류). 배울 때 필요한 크기와 쓸 때 필요한 크기가 다를 수 있다는 뜻이에요.")]
libs2 = [("⑥⑦", "<code>torch.optim.SGD</code>, <code>torch.optim.Adam</code>", "Optimizer"), ("⑨", "<code>nn.Dropout</code>, <code>weight_decay=</code>", "규제"), ("⑩", "<code>state_dict()</code>, <code>load_state_dict()</code>, <code>copy.deepcopy</code>", "가중치 저장과 복원")]
K.build(ROOT / "topics" / "model" / "dl-training-stability.html", title="학습을 안정시키기", h1="학습을 안정시키기: 학습률, Optimizer, 과적합", track="모델",
        crumb="모델 &gt; 딥러닝 기초 2/2쪽 &nbsp;|&nbsp; 이전: <a href=\"dl-tensor-training.html\" style=\"color:inherit\">Tensor와 신경망 학습</a>",
        one_line="학습이 잘 내려가려면 <b>학습률과 Optimizer</b>가 맞아야 하고, 학습 데이터를 외우지 않으려면 <b>규제와 조기 종료</b>가 필요해요. 문제는 train과 valid 손실을 <b>같이</b> 봐야 보여요.",
        why=why2, whole=whole2, steps=steps2, code_file="examples/dl/points_net.py", practice=practice2, libs=libs2, libs_note="<code>run</code>, <code>train_early</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"dl-tensor-training.html\">Tensor와 신경망 학습</a>", "학습 루프의 기본"), ("<a href=\"cnn.html\">CNN과 이미지 분류</a>", "그림을 다루는 신경망")],
        sources=["다루는 범위: Day10 (2026-10-30) Optimizer, 학습률, Batch, Epoch, 과적합, Dropout, Early Stopping",
                 "함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 6장, 8~9장, 20장. 미니배치 SGD, 모멘텀과 Nesterov, Adam, 학습률 스케줄, 오차의 잡음, 편향, 분산, 이중 하강, weight decay와 AdamW, Dropout, 암묵적 규제, 증강과 전이 학습, 딥러닝이 일반화하는 이유, 가지치기와 증류를 가져왔어요. 값과 그림은 이 예제로 다시 계산했어요",
                 SRC_DATA, "결과 값과 그림: <code>tools/pages/dl_basics.py</code> (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/docs/stable/optim.html\" target=\"_blank\">torch.optim</a>, <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html\" target=\"_blank\">nn.Dropout</a>"])
print("ok")
