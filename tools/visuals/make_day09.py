"""Day9 (GPU·Tensor·Autograd·ANN·학습 루프) 그림 만들기.

실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day09.py
모든 계산은 CPU에서, seed 고정.
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()

import time
import numpy as np
import torch
from torch import nn
from matplotlib.patches import Rectangle, FancyBboxPatch
from sklearn.datasets import make_moons

DAY = 9
torch.manual_seed(0)
np.random.seed(0)


# ------------------------------------------------------------------
# 1) CPU 코어 vs GPU 연산 유닛: 결과 행렬 칸을 몇 "차례"에 나눠 계산하나
# ------------------------------------------------------------------
def cpu_gpu_lanes():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    n = 6  # 6x6 결과 행렬 = 36칸, 칸끼리 서로 독립
    rounds_cpu = np.arange(n * n).reshape(n, n) // 4  # 코어 4개 → 4칸씩 9차례
    cmap = plt.get_cmap("Purples")
    for ax, title, rounds, sub in [
        (axes[0], "CPU 코어 4개", rounds_cpu, "4칸씩 → 9차례에 걸쳐 계산"),
        (axes[1], "GPU 연산 유닛 36개 이상", np.zeros((n, n), int), "36칸을 → 1차례에 동시에"),
    ]:
        mx = max(rounds.max(), 1)
        for i in range(n):
            for j in range(n):
                r = rounds[i, j]
                col = cmap(0.35 + 0.6 * r / 8) if rounds.max() > 0 else C["mint"]
                ax.add_patch(Rectangle((j, n - 1 - i), 0.92, 0.92, color=col))
                ax.text(j + 0.46, n - 1 - i + 0.46, str(r + 1), ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")
        ax.set_xlim(-0.1, n); ax.set_ylim(-0.1, n); ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{title}\n{sub}", fontsize=12)
    fig.suptitle("결과 행렬 각 칸 = 서로 독립인 '곱하고 더하기' (숫자 = 몇 번째 차례에 계산되나)",
                 fontsize=11, color=C["ink"], y=0.02)
    save(fig, DAY, "cpu_gpu_lanes")


# ------------------------------------------------------------------
# 2) 행렬곱 시간: CPU 스레드 1개 vs 여러 개 (실측, CPU만)
# ------------------------------------------------------------------
def matmul_timing():
    sizes = [128, 256, 512, 768, 1024]
    max_threads = torch.get_num_threads()
    res = {}
    for th in sorted({1, max_threads}):
        torch.set_num_threads(th)
        ts = []
        for s in sizes:
            a, b = torch.randn(s, s), torch.randn(s, s)
            a @ b  # 워밍업
            reps = 5
            t0 = time.perf_counter()
            for _ in range(reps):
                a @ b
            ts.append((time.perf_counter() - t0) / reps * 1000)
        res[th] = ts
    torch.set_num_threads(max_threads)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(sizes, res[1], "o-", color=C["gray"], lw=2.5, label="CPU 스레드 1개 (한 줄로)")
    if max_threads > 1:
        ax.plot(sizes, res[max_threads], "o-", color=C["purple"], lw=2.5,
                label=f"CPU 스레드 {max_threads}개 (나눠서 동시에)")
        sp = res[1][-1] / res[max_threads][-1]
        ax.annotate(f"같은 계산인데\n약 {sp:.1f}배 빠름", xy=(sizes[-1], res[max_threads][-1]),
                    xytext=(420, res[1][-1] * 0.45), fontsize=11, color=C["purple"],
                    arrowprops=dict(arrowstyle="->", color=C["purple"]))
    ax.set_xlabel("정사각 행렬 크기 n  (n x n @ n x n)")
    ax.set_ylabel("1회 행렬곱 시간 (ms)")
    ax.set_title("이 PC의 CPU에서 실제로 잰 행렬곱 시간 (GPU 아님)")
    ax.legend(loc="upper left")
    save(fig, DAY, "matmul_timing")


# ------------------------------------------------------------------
# 3) Tensor 차원: 스칼라 → 벡터 → 행렬 → 3D → 4D [B,C,H,W]
# ------------------------------------------------------------------
def draw_grid(ax, x0, y0, rows, cols, s, face, edge="white", alpha=1.0):
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x0 + j * s, y0 + (rows - 1 - i) * s), s, s,
                                   facecolor=face, edgecolor=edge, lw=1.2, alpha=alpha))


def tensor_shapes():
    fig, ax = plt.subplots(figsize=(10, 3.6))
    s = 0.42
    # 0D
    draw_grid(ax, 0.3, 1.2, 1, 1, s, C["pink"])
    ax.text(0.3 + s / 2, 0.55, "0D 스칼라\nshape ()", ha="center", va="top", fontsize=10.5)
    # 1D
    draw_grid(ax, 1.4, 1.2, 1, 4, s, C["orange"])
    ax.text(1.4 + 2 * s, 0.55, "1D 벡터\nshape (4,)", ha="center", va="top", fontsize=10.5)
    # 2D
    draw_grid(ax, 3.6, 0.8, 3, 4, s, C["mint"])
    ax.text(3.6 + 2 * s, 0.55, "2D 행렬\nshape (3, 4)\n= (batch, feature)", ha="center", va="top", fontsize=10.5)
    # 3D: 2장 겹치기
    for k in range(2, -1, -1):
        draw_grid(ax, 6.0 + k * 0.18, 0.8 + k * 0.18, 3, 4, s, C["blue"], alpha=0.55 + 0.15 * (2 - k))
    ax.text(6.0 + 2 * s + 0.2, 0.55, "3D\nshape (3, 3, 4)\n예: (batch, seq, feature)", ha="center", va="top", fontsize=10.5)
    # 4D: batch 2개 x (채널 3장 x 3 x 3)
    chan_cols = [C["red"], C["green"], C["blue"]]
    for b in range(2):
        bx = 8.7 + b * 2.0
        for k in range(2, -1, -1):
            draw_grid(ax, bx + k * 0.16, 0.8 + k * 0.16, 3, 3, s, chan_cols[k], alpha=0.75)
        ax.text(bx + 0.75, 2.75, f"이미지 {b + 1}", ha="center", fontsize=10, color=C["ink"])
    ax.add_patch(FancyBboxPatch((8.55, 0.65), 3.85, 2.45, boxstyle="round,pad=0.05",
                                fill=False, ec=C["purple"], lw=1.5, ls="--"))
    ax.text(10.5, 0.55, "4D  shape (B=2, C=3, H=3, W=3)\nbatch 안에 이미지, 이미지 안에 RGB 채널",
            ha="center", va="top", fontsize=10.5, color=C["purple"])
    for x in [1.05, 3.25, 5.65, 8.25]:
        ax.annotate("", xy=(x + 0.25, 1.4), xytext=(x - 0.05, 1.4),
                    arrowprops=dict(arrowstyle="->", color=C["gray"], lw=1.5))
    ax.set_xlim(0, 12.6); ax.set_ylim(-0.45, 3.35); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("차원이 하나 늘 때마다 '같은 모양 상자'를 한 방향으로 더 쌓는다", fontsize=12.5)
    save(fig, DAY, "tensor_shapes")


# ------------------------------------------------------------------
# 4) Broadcasting: (2,3) + (3,) → (2,3)
# ------------------------------------------------------------------
def broadcasting():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([10, 20, 30])
    fig, ax = plt.subplots(figsize=(9, 3.0))
    s = 0.8

    def grid(x0, vals, face, ghost_rows=()):
        rows, cols = vals.shape
        for i in range(rows):
            for j in range(cols):
                ghost = i in ghost_rows
                ax.add_patch(Rectangle((x0 + j * s, (rows - 1 - i) * s), s, s, facecolor=face,
                                       edgecolor="white", lw=2, alpha=0.35 if ghost else 0.95,
                                       ls="--" if ghost else "-"))
                ax.text(x0 + j * s + s / 2, (rows - 1 - i) * s + s / 2, str(vals[i, j]), ha="center",
                        va="center", fontsize=13, color="white" if not ghost else C["ink"], fontweight="bold")

    grid(0, X, C["mint"])
    ax.text(1.2, -0.45, "X  (2, 3)", ha="center", fontsize=12)
    ax.text(2.75, 0.8, "+", fontsize=22, ha="center", va="center")
    grid(3.2, np.vstack([b, b]), C["orange"], ghost_rows=(1,))
    ax.text(4.4, -0.45, "bias (3,) → 행마다 복사한 것처럼", ha="center", fontsize=12)
    ax.text(5.95, 0.8, "=", fontsize=22, ha="center", va="center")
    grid(6.4, X + b, C["purple"])
    ax.text(7.6, -0.45, "결과 (2, 3)", ha="center", fontsize=12)
    ax.text(4.4, 1.85, "점선 줄은 실제로 메모리에 만들지 않아요", ha="center", fontsize=10, color=C["gray"])
    ax.set_xlim(-0.2, 9); ax.set_ylim(-0.8, 2.1); ax.set_aspect("equal"); ax.axis("off")
    save(fig, DAY, "broadcasting")


# ------------------------------------------------------------------
# 5) 경사하강: 같은 그릇 모양 loss에서 lr 3가지의 경로
# ------------------------------------------------------------------
def gd_lr_contour():
    f = lambda w1, w2: w1 ** 2 + 1.5 * w2 ** 2
    g = lambda w: np.array([2 * w[0], 3 * w[1]])
    W1, W2 = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-4, 4, 200))
    Z = f(W1, W2)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9), sharey=True)
    cases = [(0.02, 20, "lr = 0.02  너무 작음", C["blue"], "20번 갱신해도\n아직 멀었음"),
             (0.3, 20, "lr = 0.3  적당", C["mint"], "몇 번 만에 바닥 도착"),
             (0.7, 8, "lr = 0.7  너무 큼", C["red"], "바닥을 지나쳐 반대편으로,\n갈수록 더 멀리 튐 (발산)")]
    for ax, (lr, n, title, col, note) in zip(axes, cases):
        ax.contour(W1, W2, Z, levels=[0.5, 2, 5, 10, 18, 28, 40], colors="#cfc8e0", linewidths=1)
        w = np.array([-4.2, 1.8]); path = [w.copy()]
        for _ in range(n):
            w = w - lr * g(w); path.append(w.copy())
        p = np.array(path)
        ax.plot(p[:, 0], p[:, 1], "o-", color=col, ms=4.5, lw=1.6)
        for k in range(min(3, n)) if lr > 0.5 else []:
            ax.text(p[k + 1, 0] + 0.15, p[k + 1, 1], str(k + 1), fontsize=9, color=col)
        ax.plot(*p[0], "s", color=C["ink"], ms=8)
        ax.plot(0, 0, "*", color=C["yellow"], ms=16, mec=C["ink"])
        ax.set_title(title, color=col)
        ax.text(0.03, 0.03, note, transform=ax.transAxes, fontsize=10, color=col,
                bbox=dict(fc="white", ec="none", alpha=0.85))
        ax.set_xlabel("가중치 w1"); ax.grid(False)
        ax.set_xlim(-5, 5); ax.set_ylim(-4, 4)
    axes[0].set_ylabel("가중치 w2")
    axes[0].text(-4.4, 2.2, "출발", fontsize=10)
    axes[1].text(0.3, 0.35, "loss 최소", fontsize=10)
    fig.suptitle("등고선 = loss가 같은 높이 · 점 하나 = 갱신 한 번 · 같은 출발점, lr만 다름", fontsize=11, y=0.02)
    save(fig, DAY, "gd_lr_contour")


# ------------------------------------------------------------------
# 6) 활성화 함수 + "ReLU가 없으면 Linear를 쌓아도 직선"
# ------------------------------------------------------------------
def activations():
    x = torch.linspace(-4, 4, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.plot(x, torch.relu(x), lw=3, color=C["purple"], label="ReLU  max(0, x)")
    ax.plot(x, torch.sigmoid(x), lw=3, color=C["pink"], label="sigmoid  0~1로 눌러 줌")
    ax.plot(x, torch.tanh(x), lw=3, color=C["mint"], label="tanh  -1~1로 눌러 줌")
    ax.axhline(0, color=C["gray"], lw=0.8); ax.axvline(0, color=C["gray"], lw=0.8)
    ax.set_ylim(-1.3, 3); ax.set_xlabel("입력 z (가중합)"); ax.set_ylabel("출력")
    ax.set_title("활성화 함수: 직선을 '꺾어' 준다")
    ax.annotate("음수는 전부 0", xy=(-2.5, 0), xytext=(-3.9, 1.4), fontsize=10.5, color=C["purple"],
                arrowprops=dict(arrowstyle="->", color=C["purple"]))
    ax.legend(loc="upper left", fontsize=9.5)

    ax = axes[1]
    torch.manual_seed(3)
    l1, l2 = nn.Linear(1, 8), nn.Linear(8, 1)
    xx = x.unsqueeze(1)
    with torch.no_grad():
        no_relu = l2(l1(xx)).squeeze()
        with_relu = l2(torch.relu(l1(xx))).squeeze()
    ax.plot(x, no_relu, lw=3, color=C["gray"], label="Linear → Linear (ReLU 없음)")
    ax.plot(x, with_relu, lw=3, color=C["purple"], label="Linear → ReLU → Linear")
    ax.set_xlabel("입력 x"); ax.set_ylabel("모델 출력")
    ax.set_title("같은 가중치, ReLU만 넣었다 뺐다")
    ax.text(0.97, 0.80, "ReLU 없으면 아무리 쌓아도 직선 하나\nReLU가 있으면 꺾인 곡선", transform=ax.transAxes,
            ha="right", fontsize=10, color=C["ink"], bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.legend(loc="lower left", fontsize=9.5)
    save(fig, DAY, "activations")


# ------------------------------------------------------------------
# 7) logits → softmax 확률 → argmax
# ------------------------------------------------------------------
def softmax_bars():
    logits = torch.tensor([1.8, -0.4])
    probs = torch.softmax(logits, dim=0)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 0.7]))
    names = ["클래스 0", "클래스 1"]
    ax = axes[0]
    bars = ax.bar(names, logits, color=[C["purple"], C["gray"]], width=0.55)
    ax.axhline(0, color=C["ink"], lw=1)
    for b_, v in zip(bars, logits):
        ax.text(b_.get_x() + b_.get_width() / 2, v + (0.1 if v > 0 else -0.28), f"{v:.1f}", ha="center", fontsize=12)
    ax.set_title("① logits (모델 원시 점수)"); ax.set_ylim(-1, 2.4)
    ax.text(0.5, 0.92, "음수도 OK · 합이 1 아님", transform=ax.transAxes, ha="center", fontsize=10, color=C["gray"])
    ax = axes[1]
    bars = ax.bar(names, probs, color=[C["purple"], C["gray"]], width=0.55)
    for b_, v in zip(bars, probs):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=12)
    ax.set_ylim(0, 1.15); ax.set_title("② softmax 확률")
    ax.text(0.5, 0.92, f"0~1 사이 · 합 = {probs.sum():.2f}", transform=ax.transAxes, ha="center", fontsize=10, color=C["gray"])
    ax = axes[2]; ax.axis("off")
    ax.text(0.5, 0.62, "③ argmax", ha="center", fontsize=13, fontweight="bold", color=C["ink"])
    ax.text(0.5, 0.40, "예측 = 0", ha="center", fontsize=20, color=C["purple"], fontweight="bold")
    ax.text(0.5, 0.18, "가장 큰 것의 번호만", ha="center", fontsize=10, color=C["gray"])
    fig.text(0.355, 0.5, "→", fontsize=24, ha="center", va="center", color=C["gray"])
    fig.text(0.715, 0.5, "→", fontsize=24, ha="center", va="center", color=C["gray"])
    save(fig, DAY, "softmax_bars")


# ------------------------------------------------------------------
# 8) accuracy(계단) vs CrossEntropy loss(매끈한 경사)
# ------------------------------------------------------------------
def acc_vs_loss():
    m = np.linspace(-4, 4, 400)  # 정답 logit - 오답 logit
    ce = np.log1p(np.exp(-m))
    acc = (m > 0).astype(float)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(m, ce, lw=3, color=C["purple"], label="CrossEntropy loss (연속)")
    ax.step(m, acc, where="mid", lw=3, color=C["orange"], label="정확도 (맞음 1 / 틀림 0)")
    ax.axvline(0, color=C["gray"], lw=0.8, ls="--")
    ax.annotate("여기선 평평\n→ 기울기 0, 방향을 모름", xy=(2.5, 1.0), xytext=(1.0, 2.6), fontsize=10, color=C["orange"],
                arrowprops=dict(arrowstyle="->", color=C["orange"]))
    ax.annotate("loss는 늘 기울어 있어\n'조금 더 맞게' 갈 방향을 알려 줌", xy=(-1.5, np.log1p(np.exp(1.5))), xytext=(-3.9, 3.4),
                fontsize=10, color=C["purple"], arrowprops=dict(arrowstyle="->", color=C["purple"]))
    ax.set_xlabel("정답 클래스 점수 - 오답 클래스 점수")
    ax.set_ylabel("값"); ax.set_ylim(-0.1, 4.3)
    ax.set_title("왜 accuracy가 아니라 loss로 학습할까")
    ax.legend(loc="upper right", fontsize=9.5)
    save(fig, DAY, "acc_vs_loss")


# ------------------------------------------------------------------
# 9) DataLoader: 200개를 32개씩 → 7개 batch (마지막 8개)
# ------------------------------------------------------------------
def dataloader_batches():
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(200, generator=g).numpy()
    fig, ax = plt.subplots(figsize=(10, 2.6))
    cols = [C["purple"], C["pink"], C["mint"], C["orange"], C["blue"], C["yellow"], C["red"]]
    per_row = 40
    for k, i in enumerate(idx):
        b = k // 32
        r, c = divmod(i, per_row)  # 원래 위치에 색칠 → 섞인 모습
        ax.add_patch(Rectangle((c, -r), 0.9, 0.9, color=cols[b]))
    for b in range(7):
        n = 32 if b < 6 else 8
        ax.add_patch(Rectangle((b * 5.8, 1.3), 0.9, 0.9, color=cols[b]))
        ax.text(b * 5.8 + 1.2, 1.75, f"batch {b + 1}: {n}개", va="center", fontsize=10.5,
                fontweight="bold" if b == 6 else None, color=C["red"] if b == 6 else C["ink"])
    ax.set_xlim(-0.3, per_row + 0.3); ax.set_ylim(-4.4, 2.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("샘플 200개(칸 하나 = 샘플 하나)를 shuffle=True, batch_size=32로 나눈 모습", fontsize=11.5)
    save(fig, DAY, "dataloader_batches")


# ------------------------------------------------------------------
# 10) 2→8→2 ANN이 moons 데이터를 배워 가는 과정 + loss 곡선
# ------------------------------------------------------------------
def ann_training():
    X_np, y_np = make_moons(200, noise=0.2, random_state=0)
    X = torch.tensor(X_np, dtype=torch.float32); y = torch.tensor(y_np, dtype=torch.long)
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 220), np.linspace(-1.1, 1.6, 160))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    snaps_at = [0, 3, 20, 200]
    snaps, losses = {}, []

    def snap(ep):
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(grid), 1)[:, 1].numpy().reshape(xx.shape)
            acc = (model(X).argmax(1) == y).float().mean().item()
        snaps[ep] = (p, acc)

    snap(0)
    with torch.no_grad():
        losses.append(loss_fn(model(X), y).item())
    for ep in range(1, 201):
        model.train(); tot = 0
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        losses.append(tot / len(X))
        if ep in snaps_at:
            snap(ep)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("pm", [C["blue"], "#ffffff", C["pink"]])
    for ax, ep in zip(axes, snaps_at):
        p, acc = snaps[ep]
        ax.contourf(xx, yy, p, levels=np.linspace(0, 1, 11), cmap=cm, alpha=0.75)
        ax.contour(xx, yy, p, levels=[0.5], colors=C["ink"], linewidths=2)
        ax.scatter(X_np[:, 0], X_np[:, 1], c=[C["blue"] if t == 0 else C["pink"] for t in y_np], s=12,
                   edgecolors="white", linewidths=0.4)
        ax.set_title(f"epoch {ep}  ·  정확도 {acc:.0%}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("검은 선 = 결정 경계(두 클래스 확률이 반반인 곳) · 배경색 = 모델이 본 클래스 확률",
                 fontsize=11, y=0.02)
    save(fig, DAY, "ann_boundary")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(range(len(losses)), losses, lw=2.5, color=C["purple"])
    for ep in snaps_at:
        ax.plot(ep, losses[ep], "o", ms=9, color=C["pink"], mec="white", zorder=5)
        dx = -40 if ep == 200 else 6
        ax.annotate(f"epoch {ep}", xy=(ep, losses[ep]), xytext=(ep + dx, losses[ep] + 0.04), fontsize=10)
    ax.set_xlabel("epoch"); ax.set_ylabel("epoch 평균 train loss")
    ax.set_title("같은 학습의 loss 곡선: 처음엔 가파르게, 나중엔 천천히"); ax.set_ylim(0.1, 0.9)
    save(fig, DAY, "ann_loss")


if __name__ == "__main__":
    cpu_gpu_lanes()
    matmul_timing()
    tensor_shapes()
    broadcasting()
    gd_lr_contour()
    activations()
    softmax_bars()
    acc_vs_loss()
    dataloader_batches()
    ann_training()
