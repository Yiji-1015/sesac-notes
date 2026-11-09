"""Day10 (Optimizer·학습률·Batch·과적합·Dropout·BatchNorm·Weight Decay·Early Stopping) 그림 만들기.

실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day10.py
모든 계산은 CPU, seed 고정. 데이터는 강의처럼 2D 특성·2클래스(초승달 모양 toy 데이터).
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import make_moons

DAY = 10
CM = LinearSegmentedColormap.from_list("pm", [C["blue"], "#ffffff", C["pink"]])


def T(a, dt=torch.float32):
    return torch.tensor(a, dtype=dt)


def make_data(n_train, n_valid, noise, seed):
    X, y = make_moons(n_train + n_valid, noise=noise, random_state=seed)
    X = (X - X.mean(0)) / X.std(0)
    return (T(X[:n_train]), T(y[:n_train], torch.long), T(X[n_train:]), T(y[n_train:], torch.long))


class Net(nn.Module):
    def __init__(self, hidden=64, p=0.0, bn=False):
        super().__init__()
        layers, d = [], 2
        for _ in range(2):
            layers.append(nn.Linear(d, hidden))
            if bn:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if p > 0:
                layers.append(nn.Dropout(p))
            d = hidden
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run(model, opt, Xtr, ytr, Xva, yva, epochs, bs, seed=0, keep=(), per_iter=False):
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    hist = {"train": [], "valid": [], "iter": [], "states": {}}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            if per_iter:
                hist["iter"].append(loss.item())
        model.eval()
        with torch.no_grad():
            hist["train"].append(loss_fn(model(Xtr), ytr).item())
            hist["valid"].append(loss_fn(model(Xva), yva).item())
        if ep in keep:
            hist["states"][ep] = copy.deepcopy(model.state_dict())
    return hist


def boundary(ax, model, X, y, title, lim=2.6):
    xx, yy = np.meshgrid(np.linspace(-lim, lim, 200), np.linspace(-lim, lim, 200))
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(T(np.c_[xx.ravel(), yy.ravel()])), 1)[:, 1].numpy().reshape(xx.shape)
    ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap=CM, alpha=0.7)
    ax.contour(xx, yy, p, levels=[0.5], colors=C["ink"], linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=[C["blue"] if t == 0 else C["pink"] for t in y], s=22,
               edgecolors="white", linewidths=0.6)
    ax.set_title(title, fontsize=11.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


# ------------------------------------------------------------------
# 1) SGD vs Momentum vs Adam: 길쭉한 골짜기 위의 경로
# ------------------------------------------------------------------
def optim_paths():
    f = lambda w: 0.02 * w[0] ** 2 + 2.0 * w[1] ** 2
    cfgs = [("SGD (lr=0.45)", lambda p: torch.optim.SGD([p], lr=0.45), C["gray"]),
            ("SGD + momentum 0.8 (lr=0.45)", lambda p: torch.optim.SGD([p], lr=0.45, momentum=0.8), C["purple"]),
            ("Adam (lr=0.2)", lambda p: torch.optim.Adam([p], lr=0.2), C["mint"])]
    W1, W2 = np.meshgrid(np.linspace(-5.5, 2, 300), np.linspace(-2.2, 2.2, 200))
    Z = 0.02 * W1 ** 2 + 2 * W2 ** 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axes[0]
    ax.contour(W1, W2, Z, levels=[0.05, 0.2, 0.5, 1, 2, 3.5, 5, 7], colors="#cfc8e0", linewidths=1)
    steps = 40
    for name, mk, col in cfgs:
        p = torch.tensor([-5.0, 1.5], requires_grad=True)
        opt = mk(p); path, losses = [p.detach().clone().numpy()], [f(p).item()]
        for _ in range(steps):
            opt.zero_grad(); l = f(p); l.backward(); opt.step()
            path.append(p.detach().clone().numpy()); losses.append(f(p).item())
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", ms=3.5, lw=1.8, color=col, label=name)
        axes[1].plot(losses, lw=2.5, color=col, label=name)
    ax.plot(-5, 1.5, "s", color=C["ink"], ms=9); ax.text(-5.3, 1.75, "출발", fontsize=10)
    ax.plot(0, 0, "*", ms=17, color=C["yellow"], mec=C["ink"])
    ax.set_xlim(-5.5, 2); ax.set_ylim(-2.2, 2.2); ax.grid(False)
    ax.set_xlabel("w1 (완만한 방향)"); ax.set_ylabel("w2 (가파른 방향)")
    ax.set_title("같은 출발점, 40번 갱신 경로")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.9, frameon=True)
    ax.annotate("좌우로 튀느라\n앞으로는 느림", xy=(-4.2, -0.9), xytext=(-3.6, -2.0), fontsize=9.5, color=C["gray"],
                arrowprops=dict(arrowstyle="->", color=C["gray"]))
    a = axes[1]
    a.set_ylim(0, 5); a.set_xlabel("갱신 횟수 (step)"); a.set_ylabel("loss")
    a.set_title("loss가 줄어드는 속도")
    save(fig, DAY, "optim_paths")


# ------------------------------------------------------------------
# 2) 학습률 3개 비교 (실제 학습, AdamW, 40 epoch)
# ------------------------------------------------------------------
def lr_curves():
    Xtr, ytr, Xva, yva = make_data(240, 80, 0.25, 1)
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for lr, col, lab in [(0.0001, C["blue"], "lr=0.0001 (너무 작음)"), (0.01, C["mint"], "lr=0.01 (적당)"),
                         (0.5, C["red"], "lr=0.5 (너무 큼)")]:
        torch.manual_seed(7)
        m = Net()
        h = run(m, torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01), Xtr, ytr, Xva, yva, 40, 32, seed=7)
        ax.plot(range(1, 41), h["valid"], lw=2.5, color=col, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("valid loss")
    ax.set_ylim(0, 1.5); ax.text(5.5, 1.38, "위로 잘린 곳 = loss가 튀어 오름", fontsize=10, color=C["red"])
    ax.set_title("같은 seed·모델·데이터에서 lr만 바꾼 valid loss")
    ax.legend(fontsize=9.5, loc="upper right")
    save(fig, DAY, "lr_curves")


# ------------------------------------------------------------------
# 3) Scheduler: StepLR vs ReduceLROnPlateau (실제 torch 스케줄러 · 감시 값은 설명용 가상 곡선)
# ------------------------------------------------------------------
def scheduler_curves():
    E = 40
    ep = np.arange(1, E + 1)
    rng = np.random.default_rng(3)
    valid = 0.35 + 0.55 * np.exp(-ep / 5)
    valid[18:] = valid[17] + rng.normal(0, 0.004, E - 18)         # 18 epoch 이후 정체
    valid[26:] -= 0.04 * (1 - np.exp(-(ep[26:] - 26) / 3))       # lr 낮춘 뒤 조금 더 내려감
    valid[33:] = valid[32] + rng.normal(0, 0.003, E - 33)
    p = torch.nn.Parameter(torch.zeros(1))
    o1 = torch.optim.SGD([p], lr=0.01); s1 = torch.optim.lr_scheduler.StepLR(o1, step_size=15, gamma=0.5)
    o2 = torch.optim.SGD([p], lr=0.01)
    s2 = torch.optim.lr_scheduler.ReduceLROnPlateau(o2, mode="min", factor=0.5, patience=4)
    lr1, lr2 = [], []
    for e in range(E):
        lr1.append(o1.param_groups[0]["lr"]); lr2.append(o2.param_groups[0]["lr"])
        o1.step(); s1.step(); o2.step(); s2.step(valid[e])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    ax = axes[0]
    ax.plot(ep, valid, lw=2.5, color=C["purple"])
    ax.axvspan(18, 26, color=C["yellow"], alpha=0.15)
    ax.text(22, 0.75, "정체\n(plateau)", ha="center", fontsize=10.5, color=C["orange"])
    ax.set_xlabel("epoch"); ax.set_ylabel("valid loss"); ax.set_title("감시하는 값: valid loss (설명용 가상 곡선)")
    ax = axes[1]
    ax.step(ep, lr1, where="post", lw=2.5, color=C["gray"], label="StepLR: 15 epoch마다 x0.5 (달력처럼)")
    ax.step(ep, lr2, where="post", lw=2.5, color=C["pink"], label="ReduceLROnPlateau: 정체하면 x0.5")
    ax.axvspan(18, 26, color=C["yellow"], alpha=0.15)
    ax.set_ylim(0, 0.0115); ax.set_xlabel("epoch"); ax.set_ylabel("learning rate")
    ax.set_title("실제 torch 스케줄러가 정한 lr")
    ax.legend(fontsize=9.5, loc="lower left")
    save(fig, DAY, "scheduler_curves")


# ------------------------------------------------------------------
# 4) batch 크기: epoch당 iteration 수 + loss 흔들림 (실제 학습)
# ------------------------------------------------------------------
def batch_figs():
    Xtr, ytr, Xva, yva = make_data(240, 80, 0.25, 1)
    fig, ax = plt.subplots(figsize=(9, 2.9))
    for row, bs in enumerate([8, 32, 240]):
        n_it = int(np.ceil(240 / bs))
        for k in range(n_it):
            w = min(bs, 240 - k * bs)
            ax.add_patch(Rectangle((k * bs, -row), w - 0.6 if bs > 8 else w - 0.3, 0.7,
                                   color=[C["purple"], C["pink"]][k % 2], alpha=0.85))
        ax.text(-6, -row + 0.35, f"batch {bs}", ha="right", va="center", fontsize=11.5)
        ax.text(246, -row + 0.35, f"→ {n_it} iterations", va="center", fontsize=11.5, fontweight="bold",
                color=C["ink"])
    ax.set_xlim(-55, 320); ax.set_ylim(-2.3, 0.9); ax.axis("off")
    ax.set_title("train 240건 한 바퀴(1 epoch)를 몇 조각으로 자르나 · 조각 하나 = step 1번", fontsize=12)
    save(fig, DAY, "batch_iters")

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for bs, col in [(8, C["orange"]), (32, C["purple"]), (240, C["mint"])]:
        torch.manual_seed(7)
        m = Net()
        h = run(m, torch.optim.SGD(m.parameters(), lr=0.05), Xtr, ytr, Xva, yva, 15, bs, seed=7, per_iter=True)
        it = np.array(h["iter"]); x = np.arange(1, len(it) + 1) / np.ceil(240 / bs)
        ax.plot(x, it, lw=1.3 if bs == 8 else 2.2, color=col, alpha=0.85,
                label=f"batch {bs}  (epoch당 {int(np.ceil(240 / bs))} step)")
    ax.set_xlabel("epoch (학습한 데이터 양)"); ax.set_ylabel("그 batch의 train loss")
    ax.set_title("batch가 작을수록 step마다 loss가 들쭉날쭉")
    ax.legend(fontsize=9.5)
    save(fig, DAY, "batch_noise")


# ------------------------------------------------------------------
# 5) 과적합: train 60건 · hidden 64 · 규제 없음
# ------------------------------------------------------------------
def overfit_data():
    return make_data(60, 400, 0.35, 4)


def overfit_figs():
    Xtr, ytr, Xva, yva = overfit_data()
    E = 200
    torch.manual_seed(0)
    m = Net(64)
    h = run(m, torch.optim.Adam(m.parameters(), lr=0.01), Xtr, ytr, Xva, yva, E, 16, keep=set(range(1, E + 1)))
    best = int(np.argmin(h["valid"])) + 1

    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ep = np.arange(1, E + 1)
    ax.plot(ep, h["train"], lw=2.5, color=C["purple"], label="train loss")
    ax.plot(ep, h["valid"], lw=2.5, color=C["pink"], label="valid loss")
    ax.axvline(best, color=C["ink"], ls="--", lw=1)
    ax.plot(best, h["valid"][best - 1], "o", ms=10, color=C["yellow"], mec=C["ink"], zorder=5)
    ax.annotate(f"best epoch {best}\nvalid 최저점", xy=(best, h["valid"][best - 1]), xytext=(best + 25, 0.15),
                fontsize=10.5, arrowprops=dict(arrowstyle="->", color=C["ink"]))
    ax.annotate("train은 계속 ↓\nvalid는 다시 ↑ = 외우기 시작", xy=(150, h["valid"][149]), xytext=(95, h["valid"][149] + 0.35),
                fontsize=10.5, color=C["pink"], arrowprops=dict(arrowstyle="->", color=C["pink"]))
    ax.set_ylim(0, max(h["valid"]) * 1.25)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("train 60건 · hidden 64 · 규제 없음")
    ax.legend(loc="upper left")
    save(fig, DAY, "overfit_curves")

    # 경계 3장: 과소적합(선형) / best / 마지막
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
    torch.manual_seed(0)
    lin = nn.Linear(2, 2)
    run(lin, torch.optim.Adam(lin.parameters(), lr=0.01), Xtr, ytr, Xva, yva, 200, 16)
    boundary(axes[0], lin, Xtr.numpy(), ytr.numpy(), "Underfitting\n직선 하나라 모양을 못 따라감")
    m.load_state_dict(h["states"][best])
    boundary(axes[1], m, Xtr.numpy(), ytr.numpy(), f"Good fit (epoch {best})\n큰 흐름만 따라감")
    m.load_state_dict(h["states"][E])
    boundary(axes[2], m, Xtr.numpy(), ytr.numpy(), f"Overfitting (epoch {E})\n점 하나하나를 감싸려 구불구불")
    fig.suptitle("점 = train 60건 · 검은 선 = 결정 경계", fontsize=11, y=0.02)
    save(fig, DAY, "fit_regimes")
    return h, best


# ------------------------------------------------------------------
# 6) Dropout 네트워크 그림 (train 3번 · eval 1번)
# ------------------------------------------------------------------
def dropout_net():
    rng = np.random.default_rng(5)
    layers = [3, 6, 6, 2]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.4))
    titles = ["train · forward 1", "train · forward 2", "train · forward 3", "eval (전부 켜짐)"]
    for ai, ax in enumerate(axes):
        on = [np.ones(n, bool) for n in layers]
        if ai < 3:
            for li in (1, 2):
                on[li] = rng.random(layers[li]) >= 0.4
                if on[li].sum() == 0:
                    on[li][0] = True
        pos = [[(li, (n - 1) / 2 - k) for k in range(n)] for li, n in enumerate(layers)]
        for li in range(len(layers) - 1):
            for a, (x1, y1) in enumerate(pos[li]):
                for b, (x2, y2) in enumerate(pos[li + 1]):
                    live = on[li][a] and on[li + 1][b]
                    ax.plot([x1, x2], [y1, y2], color=C["purple"] if live else "#e6e1f0",
                            lw=0.9 if live else 0.5, alpha=0.7 if live else 1, zorder=1)
        for li, ps in enumerate(pos):
            for k, (x, y) in enumerate(ps):
                if on[li][k]:
                    ax.scatter(x, y, s=170, color=C["purple"] if li in (1, 2) else C["mint"], zorder=3, edgecolors="white")
                else:
                    ax.scatter(x, y, s=170, color="white", edgecolors=C["gray"], linewidths=1.2, zorder=3)
                    ax.text(x, y, "x", ha="center", va="center", fontsize=10, color=C["gray"], zorder=4)
        ax.set_title(titles[ai], fontsize=12, color=C["pink"] if ai == 3 else C["ink"])
        ax.set_xlim(-0.4, 3.4); ax.set_ylim(-3.2, 3.2); ax.axis("off")
    fig.suptitle("Dropout(p=0.4)을 은닉층 2개에 적용 · 흰 원(x) = 이번 forward에서 꺼진 뉴런", fontsize=11, y=0.02)
    save(fig, DAY, "dropout_net")


def dropout_effect():
    Xtr, ytr, Xva, yva = overfit_data()
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for p, col, lab in [(0.0, C["pink"], "Dropout 없음"), (0.3, C["mint"], "Dropout p=0.3")]:
        torch.manual_seed(0)
        m = Net(64, p=p)
        h = run(m, torch.optim.Adam(m.parameters(), lr=0.01), Xtr, ytr, Xva, yva, 200, 16)
        ax.plot(range(1, 201), h["valid"], lw=2.5, color=col, label=f"{lab} · valid")
        ax.plot(range(1, 201), h["train"], lw=1.5, ls="--", color=col, label=f"{lab} · train")
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (eval 모드로 측정)")
    ax.set_title("같은 데이터·seed, Dropout만 추가")
    ax.legend(fontsize=9.5, loc="center right")
    save(fig, DAY, "dropout_effect")


# ------------------------------------------------------------------
# 7) BatchNorm: 층을 지날 때 activation 분포
# ------------------------------------------------------------------
def batchnorm_hist():
    torch.manual_seed(0)
    x = torch.randn(512, 2) * torch.tensor([5.0, 0.3]) + torch.tensor([10.0, -2.0])  # 스케일이 엉망인 입력
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, bn, title in [(axes[0], False, "BatchNorm 없음"), (axes[1], True, "BatchNorm 있음 (Linear → BN → ReLU)")]:
        torch.manual_seed(1)
        layers = []
        d = 2
        for _ in range(4):
            layers.append(nn.Linear(d, 64)); d = 64
            if bn:
                layers.append(nn.BatchNorm1d(64))
            layers.append(nn.ReLU())
        net = nn.Sequential(*layers).train()
        h = x; outs = []
        with torch.no_grad():
            for L in layers:
                if isinstance(L, nn.ReLU):
                    outs.append(h.flatten().numpy())  # ReLU 직전 값
                h = L(h)
        for i, (o, col) in enumerate(zip(outs, [C["purple"], C["pink"], C["mint"], C["orange"]])):
            ax.hist(o, bins=80, range=(-10, 10), histtype="step", lw=2, color=col,
                    density=True, label=f"{i + 1}번째 층  평균 {o.mean():+.1f} · 표준편차 {o.std():.1f}")
        ax.set_title(title); ax.set_xlabel("ReLU에 들어가기 직전 값 (-10~10 구간만 표시)")
        ax.legend(fontsize=9, loc="upper left")
    axes[0].set_ylabel("비율")
    save(fig, DAY, "batchnorm_hist")


# ------------------------------------------------------------------
# 8) Weight decay: 가중치 히스토그램 (AdamW)
# ------------------------------------------------------------------
def wd_hist():
    Xtr, ytr, Xva, yva = overfit_data()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), gridspec_kw=dict(width_ratios=[1.3, 1]))
    models = {}
    for wd, col in [(0.0, C["pink"]), (0.5, C["mint"])]:
        torch.manual_seed(0)
        m = Net(64)
        h = run(m, torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=wd), Xtr, ytr, Xva, yva, 200, 16)
        w = m.net[2].weight.detach().flatten().numpy()
        axes[0].hist(w, bins=60, range=(-0.9, 0.9), alpha=0.55, color=col,
                     label=f"weight_decay={wd}\n|w| 평균 {np.abs(w).mean():.3f} · 마지막 valid loss {h['valid'][-1]:.2f}")
        models[wd] = m
    axes[0].set_xlabel("두 번째 Linear(64, 64)의 가중치 값"); axes[0].set_ylabel("개수")
    axes[0].set_title("200 epoch 뒤 가중치 분포 (AdamW)")
    axes[0].legend(fontsize=9, loc="upper right")
    boundary(axes[1], models[0.5], Xtr.numpy(), ytr.numpy(), "weight_decay=0.5, epoch 200의 결정 경계")
    save(fig, DAY, "wd_hist")


# ------------------------------------------------------------------
# 9) Early Stopping: patience 창 + best 복원
# ------------------------------------------------------------------
def early_stop(h):
    v = np.array(h["valid"]); t = np.array(h["train"])
    patience = 10
    best, best_ep, wait, stop = np.inf, 0, 0, None
    for e, val in enumerate(v, 1):
        if val < best:
            best, best_ep, wait = val, e, 0
        else:
            wait += 1
            if wait >= patience:
                stop = e; break
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ep = np.arange(1, len(v) + 1)
    ax.plot(ep, t, lw=1.8, color=C["purple"], alpha=0.5, label="train loss")
    ax.plot(ep[:stop], v[:stop], lw=2.8, color=C["pink"], label="valid loss (멈출 때까지)")
    ax.plot(ep[stop:], v[stop:], lw=1.8, ls=":", color=C["pink"], alpha=0.6, label="계속 돌렸다면")
    ax.axvspan(best_ep + 0.5, stop + 0.5, color=C["yellow"], alpha=0.2)
    ax.text((best_ep + stop) / 2 + 0.5, 0.62, f"patience={patience}\n개선 없이 기다림",
            ha="center", va="center", fontsize=10, color=C["orange"])
    ax.plot(best_ep, v[best_ep - 1], "o", ms=10, color=C["yellow"], mec=C["ink"], zorder=5)
    ax.axvline(stop, color=C["red"], lw=1.5)
    ax.text(stop + 1.5, 1.05, f"stop epoch {stop}", color=C["red"], fontsize=10.5)
    ax.annotate("", xy=(best_ep, 0.9), xytext=(stop, 0.9),
                arrowprops=dict(arrowstyle="->", color=C["ink"], lw=1.8))
    ax.text((best_ep + stop) / 2, 0.93, "load_state_dict(best_state)", ha="center", fontsize=9.5, color=C["ink"])
    ax.annotate(f"best epoch {best_ep}", xy=(best_ep, v[best_ep - 1]), xytext=(1, 0.62), fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=C["ink"]))
    ax.set_xlim(0, min(len(v), stop + 40)); ax.set_ylim(0, 1.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Early Stopping: 멈추는 시점과 되돌릴 시점은 다르다")
    ax.legend(loc="upper right", fontsize=9.5)
    save(fig, DAY, "early_stop")
    return best_ep, stop


if __name__ == "__main__":
    optim_paths()
    lr_curves()
    scheduler_curves()
    batch_figs()
    h, best = overfit_figs()
    print("best", best)
    dropout_net()
    dropout_effect()
    batchnorm_hist()
    wd_hist()
    print("early", early_stop(h))
