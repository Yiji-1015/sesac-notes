import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day11 (CNN) 그림 만들기.

실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day11.py
- 사전학습 가중치는 내려받지 않아요 (resnet18(weights=None)로 구조·파라미터 수만 셈).
- 예시 사진은 numpy로 그린 도형 + 실습 데이터(20261102/01_기본필수/data, 읽기 전용) 한 장.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

DAY = 11
rng = np.random.default_rng(0)
torch.manual_seed(0)
DATA = Path("C:/practice_assignment/20261102/01_기본필수/data")


# ---------------------------------------------------------------- 예시 이미지
def shapes_rgb(n=96):
    """빨간 사각형 + 초록 원 + 파란 삼각형 + 흰 대각선 (비대칭이라 뒤집기가 보임)."""
    img = np.zeros((n, n, 3), dtype=np.float32) + 0.08
    yy, xx = np.mgrid[0:n, 0:n]
    img[12:44, 10:42] = [0.95, 0.25, 0.25]                       # 사각형
    m = (yy - 66) ** 2 + (xx - 28) ** 2 < 17 ** 2
    img[m] = [0.2, 0.8, 0.35]                                    # 원
    tri = (yy > 50) & (yy < 86) & (xx > 52) & (xx - 52 < (yy - 50) * 1.1)
    img[tri] = [0.3, 0.5, 1.0]                                   # 삼각형
    for t in range(0, 34):                                       # 대각선
        img[8 + t, 56 + t:58 + t] = 1.0
    return img


def to_gray(img):
    return img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def conv2d_np(g, k, pad=1):
    t = torch.tensor(g)[None, None]
    w = torch.tensor(np.array(k, dtype=np.float32))[None, None]
    return F.conv2d(t, w, padding=pad)[0, 0].numpy()


IMG = shapes_rgb()
GRAY = to_gray(IMG)


def grid_ax(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- 1. NCHW 채널 분리
def fig_tensor_channels():
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.9))
    axes[0].imshow(IMG); axes[0].set_title("컬러 사진 [H, W, C]\n(화면·PIL·NumPy 순서)", fontsize=11)
    names = [("R 채널", "Reds"), ("G 채널", "Greens"), ("B 채널", "Blues")]
    for i, (nm, cm) in enumerate(names):
        axes[i + 1].imshow(IMG[:, :, i], cmap=cm, vmin=0, vmax=1)
        axes[i + 1].set_title(f"C={i}: {nm}\n96×96 숫자 판", fontsize=11)
    for a in axes:
        grid_ax(a)
    fig.suptitle("PyTorch는 이 세 장을 앞에 쌓아 [C=3, H=96, W=96] → 묶음이면 [N, 3, 96, 96]",
                 fontsize=11.5, y=0.02, color=C["purple"], fontweight="bold")
    save(fig, DAY, "tensor_channels")


# ---------------------------------------------------------------- 2. 슬라이딩 계산 한 칸
def fig_conv_sliding():
    x = np.zeros((6, 6), int); x[:, 2] = 1; x[4, 4] = 1
    k = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    out = np.array([[int((x[r:r + 3, c:c + 3] * k).sum()) for c in range(4)] for r in range(4)])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4), gridspec_kw={"width_ratios": [6, 3.2, 4]})
    a = axes[0]
    a.imshow(x, cmap="Purples", vmin=-0.3, vmax=1.6)
    for r in range(6):
        for c in range(6):
            a.text(c, r, x[r, c], ha="center", va="center", fontsize=12, color=C["ink"])
    a.add_patch(Rectangle((0.5, 0.5), 3, 3, fill=False, ec=C["pink"], lw=3.5))
    a.set_title("입력 6×6 (분홍 창 = 지금 보는 3×3)", fontsize=11)
    a = axes[1]
    a.imshow(k, cmap="Oranges", vmin=-0.3, vmax=1.6)
    for r in range(3):
        for c in range(3):
            a.text(c, r, k[r, c], ha="center", va="center", fontsize=13, fontweight="bold")
    a.set_title("Kernel 3×3\n(세로선 검출)", fontsize=11)
    a = axes[2]
    a.imshow(out, cmap="Greens", vmin=-0.5, vmax=3.5)
    for r in range(4):
        for c in range(4):
            a.text(c, r, out[r, c], ha="center", va="center", fontsize=13,
                   fontweight="bold" if out[r, c] == 3 else None)
    a.add_patch(Rectangle((0.5, 0.5), 1, 1, fill=False, ec=C["pink"], lw=3.5))
    a.set_title("Feature Map 4×4", fontsize=11)
    for a in axes:
        grid_ax(a)
    fig.text(0.5, -0.04, "분홍 창 안: 세로 가운데 줄 1+1+1 = 3 → 출력의 분홍 칸.  창을 한 칸씩 옮기며 같은 Kernel로 반복",
             ha="center", fontsize=11, color=C["ink"])
    save(fig, DAY, "conv_sliding")


# ---------------------------------------------------------------- 3. 진짜 Kernel 4종
KERNELS = [
    ("세로 경계", [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    ("가로 경계", [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    ("흐리기", [[1 / 9] * 3] * 3),
    ("선명하게", [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
]


def kernel_text(k):
    def f(v):
        if abs(v - 1 / 9) < 1e-6:
            return "1/9"
        return f"{int(v):d}"
    return "\n".join(" ".join(f"{f(v):>3}" for v in row) for row in k)


def fig_conv_kernels():
    fig, axes = plt.subplots(2, 5, figsize=(10, 3.9), gridspec_kw={"height_ratios": [1, 0.42], "hspace": 0.05})
    axes[0, 0].imshow(GRAY, cmap="gray", vmin=0, vmax=1); axes[0, 0].set_title("입력 (흑백)", fontsize=11)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5, "Kernel 숫자만\n바꾸면 →\n다른 Feature Map", ha="center", va="center",
                    fontsize=11, color=C["purple"], fontweight="bold", transform=axes[1, 0].transAxes)
    for i, (nm, k) in enumerate(KERNELS):
        fm = conv2d_np(GRAY, k)
        ax = axes[0, i + 1]
        if i < 2:
            lim = np.abs(fm).max()
            ax.imshow(fm, cmap="RdBu_r", vmin=-lim, vmax=lim)
        else:
            ax.imshow(np.clip(fm, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(nm, fontsize=12, fontweight="bold")
        kx = axes[1, i + 1]
        kx.axis("off")
        kx.text(0.5, 0.62, kernel_text(k), ha="center", va="center", family="monospace", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc=C["soft"], ec=C["purple"]), transform=kx.transAxes)
        kx.text(0.5, -0.12, "Kernel 3×3", ha="center", fontsize=10, color=C["gray"], transform=kx.transAxes)
    for a in axes[0]:
        grid_ax(a)
    fig.text(0.5, -0.02, "빨강 = +로 강하게 반응, 파랑 = -로 강하게 반응, 흰/회색 = 반응 없음 (경계 Kernel).  평평한 면은 0, 경계에서만 켜짐",
             ha="center", fontsize=10.5, color=C["ink"])
    save(fig, DAY, "conv_kernels")


# ---------------------------------------------------------------- 4. stride · padding 격자
def draw_grid(ax, n_in, pad, stride, title):
    N = n_in + 2 * pad
    out = (n_in + 2 * pad - 3) // stride + 1
    for r in range(N):
        for c in range(N):
            is_pad = r < pad or c < pad or r >= N - pad or c >= N - pad
            ax.add_patch(Rectangle((c, N - 1 - r), 1, 1, fc="#e9ecef" if is_pad else "#ffffff",
                                   ec="#adb5bd", lw=1, hatch="//" if is_pad else None))
            if is_pad:
                ax.text(c + 0.5, N - 1 - r + 0.5, "0", ha="center", va="center", fontsize=9, color=C["gray"])
    # 창 중심 표시
    for i in range(out):
        for j in range(out):
            cy, cx = i * stride + 1, j * stride + 1
            ax.add_patch(plt.Circle((cx + 0.5, N - 1 - cy + 0.5), 0.22, color=C["pink"], zorder=3))
    ax.add_patch(Rectangle((0, N - 3), 3, 3, fill=False, ec=C["purple"], lw=3, zorder=4))
    if out > 1:
        ax.add_patch(Rectangle((stride, N - 3), 3, 3, fill=False, ec=C["purple"], lw=2, ls="--", zorder=4))
    ax.set_xlim(-0.2, N + 0.2); ax.set_ylim(-0.2, N + 0.2); ax.set_aspect("equal")
    grid_ax(ax)
    ax.set_title(title + f"\n출력 {out}×{out}", fontsize=11.5)


def fig_stride_padding():
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.9))
    draw_grid(axes[0], 5, 0, 1, "padding 0, stride 1")
    draw_grid(axes[1], 5, 1, 1, "padding 1, stride 1")
    draw_grid(axes[2], 5, 1, 2, "padding 1, stride 2")
    fig.text(0.5, -0.03, "입력 5×5 · Kernel 3×3.  분홍 점 = 창 가운데가 설 수 있는 자리 = 출력 칸.  "
             "보라 실선 → 점선 = 창이 한 번 이동한 거리(stride)", ha="center", fontsize=10.5)
    save(fig, DAY, "stride_padding")


# ---------------------------------------------------------------- 5. 수용영역
def fig_receptive_field():
    layers = ["입력", "Conv3", "Pool2", "Conv3", "Pool2"]
    rf = [1]; j = 1
    for K, S in [(3, 1), (2, 2), (3, 1), (2, 2)]:
        rf.append(rf[-1] + (K - 1) * j); j *= S
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    n = 14
    a1.imshow(np.zeros((n, n)), cmap="gray_r", vmin=0, vmax=1)
    cols = [C["ink"], C["purple"], C["blue"], C["mint"], C["orange"]]
    cy = cx = 6.5
    for r, col, nm in zip(rf[::-1], cols[::-1], layers[::-1]):
        a1.add_patch(Rectangle((cx - r / 2, cy - r / 2), r, r, fc=col, alpha=0.25, ec=col, lw=2))
    for k in range(n + 1):
        a1.axhline(k - 0.5, color="#dee2e6", lw=0.6); a1.axvline(k - 0.5, color="#dee2e6", lw=0.6)
    grid_ax(a1)
    a1.set_title("출력 한 칸이 원본에서 보는 범위", fontsize=11.5)
    bars = a2.bar(range(5), rf, color=cols)
    for b, v in zip(bars, rf):
        a2.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v}칸", ha="center", fontsize=11, fontweight="bold")
    a2.set_xticks(range(5)); a2.set_xticklabels(["시작"] + [f"{l}\n후" for l in layers[1:]])
    a2.set_ylabel("수용영역 (한 변, 픽셀)"); a2.set_ylim(0, 12)
    a2.set_title("층을 지날수록 넓어짐: 1 → 3 → 4 → 8 → 10", fontsize=11.5)
    a2.annotate("Pool(stride 2) 뒤의 Conv는\n한 칸 이동 = 원본 2칸 →\n같은 3×3이어도 4칸씩 넓어짐",
                xy=(3, 8), xytext=(0.2, 8.6), fontsize=10, color=C["mint"],
                arrowprops=dict(arrowstyle="->", color=C["mint"]))
    save(fig, DAY, "receptive_field")


# ---------------------------------------------------------------- 6. MaxPool
def fig_maxpool():
    v = np.array([[1, 4, 0, 2], [2, 3, 1, 5], [6, 0, 2, 2], [1, 1, 3, 0]])
    out = v.reshape(2, 2, 2, 2).max(axis=(1, 3))
    blk = [C["purple"], C["pink"], C["mint"], C["orange"]]
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 0.62, 1.05, 1.05]})
    a = axes[0]
    for r in range(4):
        for c in range(4):
            b = (r // 2) * 2 + c // 2
            ismax = v[r, c] == out[r // 2, c // 2]
            a.add_patch(Rectangle((c, 3 - r), 1, 1, fc=blk[b], alpha=0.55 if ismax else 0.15, ec="white", lw=2))
            a.text(c + .5, 3 - r + .5, v[r, c], ha="center", va="center", fontsize=14,
                   fontweight="bold" if ismax else None)
    a.set_xlim(0, 4); a.set_ylim(0, 4); a.set_aspect("equal"); grid_ax(a)
    a.set_title("4×4 Feature Map", fontsize=11)
    a = axes[1]
    for r in range(2):
        for c in range(2):
            a.add_patch(Rectangle((c, 1 - r), 1, 1, fc=blk[r * 2 + c], alpha=0.55, ec="white", lw=2))
            a.text(c + .5, 1 - r + .5, out[r, c], ha="center", va="center", fontsize=14, fontweight="bold")
    a.set_xlim(0, 2); a.set_ylim(-1, 3); a.set_aspect("equal"); grid_ax(a)
    a.set_title("MaxPool 2×2\n→ 2×2", fontsize=11)
    # 진짜 Feature Map 축소
    fm = np.abs(conv2d_np(GRAY, KERNELS[0][1]))
    t = torch.tensor(fm)[None, None]
    p1 = F.max_pool2d(t, 2)[0, 0].numpy()
    axes[2].imshow(fm, cmap="magma"); axes[2].set_title(f"세로 경계 Feature Map\n{fm.shape[0]}×{fm.shape[1]}", fontsize=11)
    axes[3].imshow(p1, cmap="magma"); axes[3].set_title(f"MaxPool2d(2) 후\n{p1.shape[0]}×{p1.shape[1]}", fontsize=11)
    for a in axes[2:]:
        grid_ax(a)
    fig.text(0.5, -0.03, "색 블록마다 가장 큰 값 하나만 남김 → 칸 수 1/4, 하지만 \"여기 경계가 있다\"는 모양은 그대로",
             ha="center", fontsize=10.5)
    save(fig, DAY, "maxpool")


# ---------------------------------------------------------------- 7. 채널별 Feature Map (손으로 정한 Kernel)
def fig_feature_maps():
    vimg = np.asarray(Image.open(DATA / "train" / "vertical" / "train_vertical_000.png").convert("L"), np.float32) / 255
    himg = np.asarray(Image.open(DATA / "train" / "horizontal" / "train_horizontal_000.png").convert("L"), np.float32) / 255
    ks = [("세로 경계", KERNELS[0][1]), ("가로 경계", KERNELS[1][1]),
          ("대각 ↘", [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), ("흐리기", KERNELS[2][1])]
    fig, axes = plt.subplots(2, 5, figsize=(10, 5.0))
    for row, (im, nm) in enumerate([(vimg, "vertical 사진"), (himg, "horizontal 사진")]):
        axes[row, 0].imshow(im, cmap="gray"); axes[row, 0].set_title(nm, fontsize=11)
        for i, (kn, k) in enumerate(ks):
            fm = np.maximum(conv2d_np(im, k), 0)            # Conv → ReLU
            axes[row, i + 1].imshow(fm, cmap="viridis", vmin=0, vmax=max(fm.max(), 1e-6) if i == 3 else 3)
            mean = fm.mean()
            axes[row, i + 1].set_title(f"ch{i}: {kn}\n평균 {mean:.2f}", fontsize=10.5)
    for a in axes.flat:
        grid_ax(a)
    fig.text(0.5, -0.03, "수업 예제 데이터 두 장 · Conv(1→4채널, Kernel은 손으로 정함, 학습 X) → ReLU.  "
             "밝을수록 강한 반응. 채널 하나 = Feature Map 한 장", ha="center", fontsize=10)
    save(fig, DAY, "feature_maps")


# ---------------------------------------------------------------- 8. SmallCNN shape 흐름
def fig_shape_flow():
    stages = ["입력", "Conv1", "Pool1", "Conv2", "Pool2", "GAP"]
    hw = [64, 64, 32, 32, 16, 1]
    ch = [3, 16, 16, 32, 32, 32]
    fig, a1 = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(stages))
    a1.bar(x - 0.2, hw, 0.4, color=C["purple"], label="공간 크기 H (=W)")
    a1.bar(x + 0.2, ch, 0.4, color=C["orange"], label="채널 수 C")
    for i in range(len(stages)):
        a1.text(x[i] - 0.2, hw[i] + 1, hw[i], ha="center", fontsize=10, color=C["purple"], fontweight="bold")
        a1.text(x[i] + 0.2, ch[i] + 1, ch[i], ha="center", fontsize=10, color=C["orange"], fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(stages)
    a1.set_ylim(0, 75); a1.legend(loc="upper right", ncol=2)
    a1.set_title("SmallCNN: 공간은 줄고(어디에) 채널은 늘고(무엇이)", fontsize=12)
    save(fig, DAY, "shape_flow")


# ---------------------------------------------------------------- 9. BatchNorm 전/후
def fig_batchnorm():
    torch.manual_seed(1)
    x = torch.randn(64, 3, 16, 16)
    conv = nn.Conv2d(3, 3, 3, padding=1)
    with torch.no_grad():
        conv.weight.mul_(torch.tensor([1., 4., 0.5])[:, None, None, None])
        conv.bias.copy_(torch.tensor([3., -4., 0.5]))
        y = conv(x)
        bn = nn.BatchNorm2d(3).train()
        z = bn(y)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    cols = [C["purple"], C["pink"], C["mint"]]
    for c in range(3):
        for ax, t in [(axes[0], y), (axes[1], z)]:
            v = t[:, c].flatten().numpy()
            ax.hist(v, bins=60, range=(-12, 12), color=cols[c], alpha=0.5, label=f"채널 {c} (평균 {v.mean() + 0.0 if abs(v.mean()) >= 0.05 else 0.0:.1f}, 표준편차 {v.std():.1f})")
    axes[0].set_title("Conv 출력: 채널마다 위치·폭이 제각각", fontsize=11.5)
    axes[1].set_title("BatchNorm2d(3) 후: 모두 평균 0 · 폭 1 근처", fontsize=11.5)
    for a in axes:
        a.legend(fontsize=8.5, loc="upper left"); a.set_xlabel("값")
        a.set_ylim(0, a.get_ylim()[1] * 1.45)
    axes[0].set_ylabel("개수")
    save(fig, DAY, "batchnorm")


# ---------------------------------------------------------------- 10. CrossEntropy -log p
def fig_ce():
    p = np.linspace(0.005, 1, 400)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(p, -np.log(p), color=C["purple"], lw=2.5)
    for pv in [0.9, 0.5, 0.01]:
        ax.scatter([pv], [-np.log(pv)], color=C["pink"], zorder=3, s=40)
        ax.annotate(f"p={pv} → {-np.log(pv):.2f}", (pv, -np.log(pv)),
                    xytext=(pv + (0.04 if pv < 0.8 else -0.32), -np.log(pv) + 0.5), fontsize=10.5)
    ax.set_xlabel("모델이 정답 클래스에 준 확률 p"); ax.set_ylabel("손실 = -log p")
    ax.set_title("틀렸는데 자신만만할수록(p→0) 벌점 폭증", fontsize=12)
    ax.set_ylim(0, 5.6)
    save(fig, DAY, "ce_loss")


# ---------------------------------------------------------------- 11. 증강 격자
def fig_augmentation():
    from torchvision import transforms as T
    pil = Image.fromarray((IMG * 255).astype(np.uint8))
    torch.manual_seed(3)
    items = [("원본", pil),
             ("RandomHorizontalFlip", T.functional.hflip(pil)),
             ("RandomRotation(10)", T.functional.rotate(pil, 10)),
             ("RandomResizedCrop", T.functional.resized_crop(pil, 10, 14, 66, 66, [96, 96])),
             ("ColorJitter(밝기 -)", T.functional.adjust_brightness(pil, 0.6)),
             ("ColorJitter(밝기 +)", T.functional.adjust_brightness(pil, 1.4))]
    vimg = Image.open(DATA / "train" / "vertical" / "train_vertical_000.png").convert("RGB")
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for a, (nm, im) in zip(axes.flat[:6], items):
        a.imshow(im); a.set_title(nm, fontsize=10.5)
        for s in a.spines.values():
            s.set_visible(True); s.set_color(C["mint"]); s.set_linewidth(3)
    axes[1, 2].imshow(vimg); axes[1, 2].set_title("vertical 사진 (라벨 = vertical)", fontsize=10.5)
    axes[1, 3].imshow(vimg.rotate(90)); axes[1, 3].set_title("90도 회전 → 가로 줄무늬!", fontsize=10.5, color=C["red"])
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for s in axes[1, 3].spines.values():
        s.set_visible(True); s.set_color(C["red"]); s.set_linewidth(3)
    fig.text(0.5, -0.02, "초록 테두리 = 정답이 그대로인 증강 (OK).  빨간 테두리 = 정답을 바꿔 버리는 변환 (금지)",
             ha="center", fontsize=11)
    save(fig, DAY, "augmentation")


# ---------------------------------------------------------------- 12. 전이학습 Freeze + 파라미터 수
def fig_transfer():
    from torchvision.models import resnet18
    m = resnet18(weights=None)                 # 구조만 (다운로드 없음)
    m.fc = nn.Linear(512, 2)
    parts = [("conv1+bn1", list(m.conv1.parameters()) + list(m.bn1.parameters())),
             ("layer1", list(m.layer1.parameters())), ("layer2", list(m.layer2.parameters())),
             ("layer3", list(m.layer3.parameters())), ("layer4", list(m.layer4.parameters())),
             ("fc (Head)", list(m.fc.parameters()))]
    counts = [sum(p.numel() for p in ps) for _, ps in parts]
    names = [n for n, _ in parts]
    plans = {"Head-only": {5}, "Partial": {4, 5}, "전체 미세조정": {0, 1, 2, 3, 4, 5}}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.25, 1]})
    for row, (pn, tr) in enumerate(plans.items()):
        y = 2 - row
        for i in range(6):
            col = (C["pink"] if i == 5 else C["purple"]) if i in tr else "#ced4da"
            a1.add_patch(Rectangle((i, y), 0.92, 0.7, fc=col, ec="white"))
            a1.text(i + 0.46, y + 0.35, "학습" if i in tr else "고정", ha="center", va="center",
                    color="white" if i in tr else "#495057", fontsize=10, fontweight="bold")
        tot = sum(counts[i] for i in tr)
        a1.text(-0.1, y + 0.35, pn, ha="right", va="center", fontsize=11, fontweight="bold")
        a1.text(6.05, y + 0.35, f"{tot:,}개", ha="left", va="center", fontsize=10, color=C["purple"])
    for i, n in enumerate(["conv1\n+bn1", "layer1", "layer2", "layer3", "layer4", "fc\n(Head)"]):
        a1.text(i + 0.46, 2.85, n, ha="center", va="bottom", fontsize=9.5)
    a1.set_xlim(-2.3, 7.6); a1.set_ylim(-0.2, 3.4); a1.axis("off")
    a1.set_title("ResNet18 (K=2) · 회색 = requires_grad=False", fontsize=11.5)
    a2.barh(range(6)[::-1], counts, color=["#adb5bd"] * 4 + [C["purple"], C["pink"]])
    a2.set_xscale("log")
    for i, v in enumerate(counts):
        a2.text(v * 1.15, 5 - i, f"{v:,}", va="center", fontsize=9.5)
    a2.set_yticks(range(6)[::-1]); a2.set_yticklabels(names)
    a2.set_xlim(500, 8e7); a2.set_xlabel("파라미터 수 (로그 눈금)")
    a2.set_title("구간별 파라미터: 뒤쪽일수록 많음", fontsize=11.5)
    print("counts", counts, sum(counts))
    save(fig, DAY, "transfer_freeze")


# ---------------------------------------------------------------- 13. CPU·GPU 처리량 (실측)
class SmallCNN(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.pool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(32, k)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))


def measure(dev, bs, steps=8):
    torch.manual_seed(0)
    m = SmallCNN().to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x = torch.rand(bs, 3, 64, 64, device=dev); y = torch.randint(0, 2, (bs,), device=dev)
    lf = nn.CrossEntropyLoss()
    for _ in range(3):
        opt.zero_grad(); lf(m(x), y).backward(); opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(); lf(m(x), y).backward(); opt.step()
    if dev == "cuda":
        torch.cuda.synchronize()
    return bs * steps / (time.perf_counter() - t)


def fig_bench():
    torch.set_num_threads(max(1, torch.get_num_threads()))
    bss = [1, 4, 16, 64, 256]
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    res = {d: [measure(d, b) for b in bss] for d in devs}
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    for d, col in zip(devs, [C["orange"], C["mint"]]):
        ax.plot(bss, res[d], "o-", color=col, lw=2.2, label="CPU" if d == "cpu" else "GPU (CUDA)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(bss); ax.set_xticklabels(bss)
    ax.set_xlabel("batch size"); ax.set_ylabel("처리량 (images/s, 로그)")
    ax.set_title("SmallCNN 학습 step 실측: 작은 batch에선 GPU 이점이 작다", fontsize=11)
    ax.legend()
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
    ax.text(0.02, 0.97, f"64×64 입력 · 워밍업 3 + 측정 8 step\nGPU: {name}", transform=ax.transAxes,
            va="top", fontsize=8.5, color=C["gray"])
    print({d: [round(v) for v in r] for d, r in res.items()})
    save(fig, DAY, "bench_batch")


if __name__ == "__main__":
    fig_tensor_channels()
    fig_conv_sliding()
    fig_conv_kernels()
    fig_stride_padding()
    fig_receptive_field()
    fig_maxpool()
    fig_feature_maps()
    fig_shape_flow()
    fig_batchnorm()
    fig_ce()
    fig_augmentation()
    fig_transfer()
    fig_bench()
