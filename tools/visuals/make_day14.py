"""Day14 (전이학습·불균형·오류 분석·Grad-CAM) 학습 노트용 그림 생성.

실행:
    C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day14.py
- 원본 데이터(C:/practice_assignment/20261105/02_응용/data)는 읽기만 한다.
- 다운로드 없음: ResNet18은 weights=None(구조만)으로 파라미터 수만 센다.
- 오류 갤러리·Grad-CAM은 CPU에서 몇 초 학습한 '작은 데모 CNN' 결과다 (ResNet18 아님).
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()

import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score

DAY = 14
DATA = Path("C:/practice_assignment/20261105/02_응용/data")
CLASSES = ["circle", "square", "triangle"]
CCOL = [C["purple"], C["mint"], C["orange"]]
torch.set_num_threads(4)


def rd(path):
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"read failed: {path}")
    return img


def load_split(split):
    xs, ys, raw = [], [], []
    for k, c in enumerate(CLASSES):
        for f in sorted((DATA / split / c).glob("*.png")):
            bgr = rd(f)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raw.append(rgb)
            x = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            xs.append(x.transpose(2, 0, 1)); ys.append(k)
    return torch.tensor(np.stack(xs)), torch.tensor(ys), raw


def noaxis(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- ① 클래스 개수
def dataset_counts():
    splits = ["train", "valid", "test"]
    counts = {s: [len(list((DATA / s / c).glob("*.png"))) for c in CLASSES] for s in splits}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(splits)); w = 0.26
    for k, c in enumerate(CLASSES):
        vals = [counts[s][k] for s in splits]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=CCOL[k], label=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", fontsize=9.5)
    ax.set_xticks(x); ax.set_xticklabels([f"{s}\n(합 {sum(counts[s])})" for s in splits])
    ax.set_ylabel("이미지 수"); ax.set_ylim(0, 140)
    ax.set_title("수업 예제 데이터의 클래스별 개수", fontsize=12)
    ax.legend(ncol=3, loc="upper right")
    ax.annotate("train만 6 : 2 : 1로 불균형", xy=(0.26, 20), xytext=(0.55, 95), fontsize=10,
                color=C["red"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=C["red"]))
    save(fig, DAY, "dataset_counts")
    return counts


# ---------------------------------------------------------------- ② 학습 파라미터 수 (ResNet18 구조, 다운로드 없음)
def params_bars():
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 3)
    total = sum(p.numel() for p in m.parameters())
    head = sum(p.numel() for p in m.fc.parameters())
    l4 = sum(p.numel() for p in m.layer4.parameters())
    rows = [("Head-only\n(fc만)", head), ("Partial\n(layer4 + fc)", l4 + head), ("Full\n(전체)", total)]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(rows))[::-1]
    for yy, (name, v), col in zip(y, rows, [C["mint"], C["orange"], C["red"]]):
        ax.barh(yy, v, color=col, height=0.55)
        ax.text(v * 1.15, yy, f"{v:,}개  ({v / total * 100:.3f}%)" if v < total * 0.01 else f"{v:,}개  ({v / total * 100:.1f}%)",
                va="center", fontsize=10.5, fontweight="bold", color=C["ink"])
    ax.set_xscale("log"); ax.set_xlim(500, total * 30)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("학습되는 파라미터 수 (log 눈금: 한 칸 = 10배)")
    ax.set_title(f"ResNet18 + 3클래스 Head · 전체 {total:,}개 중 실제로 학습하는 개수", fontsize=11.5)
    save(fig, DAY, "params_bars")
    return head, l4, total


# ---------------------------------------------------------------- ③ 층별 학습률
def lr_groups():
    stages = ["A\nHead-only", "B\nlayer4 + Head", "C\n전체 + Head"]
    bb = [np.nan, 1e-4, 1e-5]; hd = [1e-3, 1e-3, 5e-4]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(3); w = 0.34
    ax.bar(x - w / 2, [0 if np.isnan(v) else v for v in bb], w, color=C["purple"], label="Backbone (열린 층) lr")
    ax.bar(x + w / 2, hd, w, color=C["pink"], label="새 Head lr")
    for i in range(3):
        if np.isnan(bb[i]):
            ax.text(x[i] - w / 2, 1.3e-6, "고정\n(학습 안 함)", ha="center", va="bottom", fontsize=9, color=C["gray"])
        else:
            ax.text(x[i] - w / 2, bb[i] * 1.2, f"{bb[i]:.0e}", ha="center", fontsize=9.5)
        ax.text(x[i] + w / 2, hd[i] * 1.2, f"{hd[i]:.0e}", ha="center", fontsize=9.5)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1e-2)
    ax.set_yticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]); ax.set_yticklabels(["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"])
    ax.minorticks_off()
    ax.set_xticks(x); ax.set_xticklabels(stages)
    ax.set_ylabel("학습률 (log 눈금)")
    ax.set_title("많이 열수록 Backbone 보폭은 더 작게 (강의 예시값)", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9.5)
    save(fig, DAY, "lr_groups")


# ---------------------------------------------------------------- 작은 데모 CNN
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),   # 16×16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # 8×8
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),                                         # 마지막 Conv (8×8)
        )
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        f = self.features(x)
        return self.head(f.mean(dim=(2, 3)))


def train_tiny(Xtr, ytr, Xva, yva, seed=14, epochs=40, lr=0.01, keep_curve=False):
    torch.manual_seed(seed)
    m = TinyCNN()
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, curve = float("inf"), None, []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtr), generator=g)
        tl = 0.0
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = crit(m(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tl += loss.item() * len(idx)
        m.eval()
        with torch.no_grad():
            vl = crit(m(Xva), yva).item()
        curve.append((tl / len(Xtr), vl))
        if vl < best:
            best, best_state, best_ep = vl, copy.deepcopy(m.state_dict()), ep
    m.load_state_dict(best_state); m.eval()
    return (m, curve, best_ep) if keep_curve else m


def best_epoch(curve, best_ep):
    tr = [c[0] for c in curve]; va = [c[1] for c in curve]
    ep = np.arange(1, len(curve) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(ep, tr, color=C["purple"], lw=2, label="train loss")
    ax.plot(ep, va, color=C["orange"], lw=2, label="valid loss")
    ax.scatter([best_ep + 1], [va[best_ep]], s=90, color=C["green"], zorder=5)
    ax.annotate(f"best = epoch {best_ep + 1}\n(valid loss {va[best_ep]:.3f}) → 이 순간을 저장",
                xy=(best_ep + 1, va[best_ep]), xytext=(best_ep + 1 + 3, max(va) * 0.75),
                color=C["green"], fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C["green"]))
    ax.annotate(f"마지막 epoch {len(ep)}\n(valid loss {va[-1]:.3f})", xy=(len(ep), va[-1]),
                xytext=(len(ep) - 12, max(va) * 0.45), color=C["red"], fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("작은 데모 CNN을 수업 예제 데이터로 15 epoch 학습한 실제 곡선 (seed 4)", fontsize=11.5)
    ax.legend(loc="upper left")
    save(fig, DAY, "best_epoch")


# ---------------------------------------------------------------- ④ Sampler: 배치 구성 (실제로 뽑아 셈)
def sampler_batches():
    from torch.utils.data import WeightedRandomSampler, RandomSampler
    targets = torch.tensor([0] * 120 + [1] * 40 + [2] * 20)
    counts = torch.bincount(targets)
    cw = (counts.sum() / counts); cw = cw / cw.mean()
    sw = (1.0 / counts.float())[targets]
    n_batches, bs = 300, 32

    def batch_counts(sampler):
        idx = torch.tensor(list(iter(sampler)))[: n_batches * bs].view(n_batches, bs)
        return np.stack([np.bincount(targets[b].numpy(), minlength=3) for b in idx])

    g1 = torch.Generator().manual_seed(14); g2 = torch.Generator().manual_seed(14)
    plain = batch_counts(RandomSampler(range(180), replacement=True, num_samples=n_batches * bs, generator=g1))
    weighted = batch_counts(WeightedRandomSampler(sw, num_samples=n_batches * bs, replacement=True, generator=g2))
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 1.25, 1.25]})
    a = axs[0]
    bars = a.bar(CLASSES, cw.numpy(), color=CCOL)
    for b, v in zip(bars, cw.numpy()):
        a.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.2f}", ha="center", fontsize=10)
    a.set_title("① class weight (평균 1)\n= 오답 벌점 배율", fontsize=11); a.set_ylim(0, 2.2)
    for a, data, title in [(axs[1], plain, "② 보통 무작위 추출\n배치 32장 안의 클래스 수"),
                           (axs[2], weighted, "③ WeightedRandomSampler\n배치 32장 안의 클래스 수")]:
        bp = a.boxplot([data[:, k] for k in range(3)], tick_labels=CLASSES, patch_artist=True, widths=0.5,
                       medianprops=dict(color=C["ink"]))
        for patch, col in zip(bp["boxes"], CCOL):
            patch.set_facecolor(col); patch.set_alpha(0.55)
        for k in range(3):
            mu = data[:, k].mean()
            a.text(k + 1.3, mu, f"평균 {mu:.1f}", va="center", fontsize=9.5, fontweight="bold", color=C["ink"],
                   bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
        a.set_ylim(-1, 32); a.set_title(title, fontsize=11)
    fig.text(0.5, -0.04, f"train 120 : 40 : 20을 가정하고 배치 {n_batches}개를 실제로 뽑아 센 결과 (seed 14) · 상자 = 배치마다 흔들리는 범위",
             ha="center", fontsize=9.5, color=C["gray"])
    save(fig, DAY, "sampler_batches")


# ---------------------------------------------------------------- ④⑤ Accuracy vs Macro F1
def metric_trap():
    cases = []
    y = [0] * 120 + [1] * 40 + [2] * 20; p = [0] * 180
    cases.append(("전부 circle로 찍기\n(120 : 40 : 20)", y, p))
    y = [0] * 4 + [1] * 2 + [2] * 1; p = [0] * 7
    cases.append(("전부 circle로 찍기\n(4 : 2 : 1 예시)", y, p))
    cm = np.array([[30, 0, 0], [8, 22, 0], [5, 4, 21]])
    y, p = cm_to_lists(cm)
    cases.append(("Sampler 합성 예시\n(30 : 30 : 30)", y, p))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(cases)); w = 0.26
    names = [("Accuracy", C["gray"]), ("Weighted F1", C["blue"]), ("Macro F1", C["pink"])]
    for k, (nm, col) in enumerate(names):
        vals = []
        for _, yt, yp in cases:
            if nm == "Accuracy":
                v = accuracy_score(yt, yp)
            else:
                v = f1_score(yt, yp, average=nm.split()[0].lower(), labels=[0, 1, 2], zero_division=0)
            vals.append(v)
        bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=nm)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylim(0, 1.12); ax.set_ylabel("점수")
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("소수 클래스를 하나도 못 맞혀도 Accuracy는 높다 · Macro F1만 실패를 드러냄", fontsize=11.5)
    save(fig, DAY, "metric_trap")


def cm_to_lists(cm):
    y, p = [], []
    for i in range(3):
        for j in range(3):
            y += [i] * int(cm[i, j]); p += [j] * int(cm[i, j])
    return y, p


# ---------------------------------------------------------------- ⑤ Confusion Matrix heatmap
def cm_heatmap():
    cm = np.array([[30, 0, 0], [8, 22, 0], [5, 4, 21]])
    norm = cm / cm.sum(1, keepdims=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, M, title, fmt in [(axs[0], cm, "개수 (raw)", "{:d}"), (axs[1], norm, "행 기준 비율 (= 클래스별 Recall)", "{:.2f}")]:
        ax.imshow(M, cmap="Purples", vmin=0, vmax=M.max())
        for i in range(3):
            for j in range(3):
                v = M[i, j]
                ax.text(j, i, fmt.format(int(v)) if fmt == "{:d}" else fmt.format(v), ha="center", va="center",
                        fontsize=13, fontweight="bold", color="white" if v > M.max() * 0.55 else C["ink"])
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("예측 (열)"); ax.set_ylabel("실제 (행)"); ax.grid(False)
        ax.set_title(title, fontsize=11.5)
        for s in ax.spines.values():
            s.set_visible(False)
        for k in range(3):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec=C["green"], lw=2.5))
    axs[0].add_patch(plt.Rectangle((-0.5, 0.5), 1, 2, fill=False, ec=C["red"], lw=2.5, ls="--"))
    axs[0].set_title("개수 (raw) · 빨간 점선 = circle로 몰린 오답 13개", fontsize=11.5)
    save(fig, DAY, "cm_heatmap")


def prf_bars():
    cm = np.array([[30, 0, 0], [8, 22, 0], [5, 4, 21]])
    y, p = cm_to_lists(cm)
    P, R, F, S = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(3); w = 0.26
    for k, (vals, nm, col) in enumerate([(P, "Precision", C["blue"]), (R, "Recall", C["orange"]), (F, "F1", C["pink"])]):
        bars = ax.bar(x + (k - 1) * w, vals, w, color=col, label=nm)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"{c}\n(support {s})" for c, s in zip(CLASSES, S)])
    ax.set_ylim(0, 1.3); ax.legend(ncol=3, loc="upper center")
    ax.annotate("Recall 1.00인데 Precision 0.70\n= 다 찾지만 '아무거나 circle'", xy=(-0.26, P[0]), xytext=(0.15, 0.32),
                fontsize=9.5, color=C["ink"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=C["ink"]),
                bbox=dict(fc="white", ec="none", alpha=0.9))
    ax.set_title(f"같은 행렬의 클래스별 지표 · Macro F1 = {F.mean():.3f}", fontsize=11.5)
    save(fig, DAY, "prf_bars")


def f1_harmonic():
    r = np.linspace(0, 1, 101); p = 1.0
    f1 = 2 * p * r / (p + r + 1e-12); am = (p + r) / 2
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(r, am, color=C["gray"], lw=2, ls="--", label="그냥 평균 (P+R)/2")
    ax.plot(r, f1, color=C["pink"], lw=2.6, label="F1 (조화평균)")
    ax.scatter([0], [0.5], color=C["gray"], zorder=5); ax.scatter([0], [0], color=C["pink"], zorder=5)
    ax.annotate("R=0이어도 0.5", xy=(0, 0.5), xytext=(0.2, 0.44), fontsize=10, color=C["gray"],
                arrowprops=dict(arrowstyle="->", color=C["gray"]))
    ax.annotate("F1은 0", xy=(0, 0), xytext=(0.12, 0.12), fontsize=10, color=C["pink"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["pink"]))
    ax.set_xlabel("Recall (Precision은 1.0으로 고정)"); ax.set_ylabel("요약 점수")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right"); ax.set_title("작은 쪽에 끌려가는 F1", fontsize=11.5)
    save(fig, DAY, "f1_harmonic")


# ---------------------------------------------------------------- ⑤ 오류 갤러리 (작은 데모 CNN)
def error_gallery(model, Xva, yva, raw_va):
    with torch.no_grad():
        probs = model(Xva).softmax(1)
    conf, pred = probs.max(1)
    wrong = (pred != yva).nonzero().flatten().tolist()
    right = (pred == yva).nonzero().flatten().tolist()
    wrong = sorted(wrong, key=lambda i: -conf[i].item())[:6]
    right_pick = []
    for k in range(3):
        cand = [i for i in right if yva[i] == k]
        if cand:
            right_pick.append(max(cand, key=lambda i: conf[i].item()))
    ncol = max(len(wrong), 3)
    fig, axs = plt.subplots(2, ncol, figsize=(1.75 * ncol + 0.4, 4.6))
    axs = np.array(axs).reshape(2, ncol)
    for c in range(ncol):
        for r in range(2):
            noaxis(axs[r, c])
    for c, i in enumerate(right_pick):
        ax = axs[0, c]; ax.imshow(raw_va[i])
        ax.set_title(f"T={CLASSES[yva[i]]}\nP={CLASSES[pred[i]]} {conf[i]:.2f}", fontsize=9.5, color=C["green"])
    for c, i in enumerate(wrong):
        ax = axs[1, c]; ax.imshow(raw_va[i])
        col = C["red"] if conf[i] > 0.8 else C["orange"]
        ax.set_title(f"T={CLASSES[yva[i]]}\nP={CLASSES[pred[i]]} {conf[i]:.2f}", fontsize=9.5, color=col)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(2.5)
    axs[0, 0].set_ylabel("맞힌 예", fontsize=11, color=C["green"]); axs[1, 0].set_ylabel("틀린 예", fontsize=11, color=C["red"])
    n_wrong = int((pred != yva).sum())
    fig.suptitle(f"작은 데모 CNN · valid 90장 중 오답 {n_wrong}장 (확신이 높은 순 최대 6장)", fontsize=11.5)
    save(fig, DAY, "error_gallery")
    return pred, conf, n_wrong


# ---------------------------------------------------------------- ⑥ Grad-CAM (작은 데모 CNN)
def gradcam(model, x, target):
    feats = {}
    layer = model.features[-2]          # 마지막 Conv

    def hook(mod, inp, out):
        feats["a"] = out
        out.retain_grad()
    h = layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(x[None])
            logits[0, target].backward()
    finally:
        h.remove()
    a = feats["a"][0].detach(); g = feats["a"].grad[0]
    alpha = g.mean(dim=(1, 2))
    cam = torch.relu((alpha[:, None, None] * a).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = torch.nn.functional.interpolate(cam[None, None], size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
    return cam.numpy(), logits.softmax(1)[0].detach()


def overlay(img32, cam):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    base = (img32 * 255).astype(np.uint8)
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)


def gradcam_demo(model, Xva, yva, pred):
    picks = []
    for k in range(3):
        cand = [i for i in range(len(yva)) if yva[i] == k and pred[i] == k]
        picks.append(cand[0])
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 7.4))
    for r, i in enumerate(picks):
        img = Xva[i].numpy().transpose(1, 2, 0)
        cam, pr = gradcam(model, Xva[i], int(pred[i]))
        for c, (im, t) in enumerate([(img, f"입력 32×32\n실제 {CLASSES[yva[i]]}"),
                                     (cam, f"Grad-CAM (target={CLASSES[pred[i]]})\n8×8 → 32×32 확대"),
                                     (overlay(img, cam), f"겹쳐 보기 · P={pr[pred[i]]:.2f}")]):
            ax = axs[r, c]
            ax.imshow(im, cmap="jet" if c == 1 else None, vmin=0, vmax=1, interpolation="bilinear")
            noaxis(ax); ax.set_title(t, fontsize=9.5)
    fig.suptitle("작은 데모 CNN(수업 예제 도형 데이터로 몇 초 학습)의 실제 Grad-CAM · ResNet18 결과 아님", fontsize=11)
    save(fig, DAY, "gradcam_demo")


def gradcam_targets(model, Xva, yva, pred):
    i = [k for k in range(len(yva)) if yva[k] == 1 and pred[k] == 1][0]
    img = Xva[i].numpy().transpose(1, 2, 0)
    fig, axs = plt.subplots(1, 4, figsize=(10, 3.0))
    axs[0].imshow(img); noaxis(axs[0]); axs[0].set_title("입력 (실제 square)", fontsize=10)
    for ax, t in zip(axs[1:], range(3)):
        cam, pr = gradcam(model, Xva[i], t)
        ax.imshow(overlay(img, cam)); noaxis(ax)
        ax.set_title(f"target = {CLASSES[t]}\n이 클래스 확률 {pr[t]:.2f}", fontsize=10,
                     color=C["green"] if t == 1 else C["ink"])
    fig.text(0.5, -0.03, "같은 사진이라도 '어느 클래스 점수의 근거?'를 바꾸면 지도가 달라짐 · 각 지도는 최댓값=1로 맞춰서 밝기 ≠ 확률",
             ha="center", fontsize=9.5, color=C["gray"])
    save(fig, DAY, "gradcam_targets")


# ---------------------------------------------------------------- ⑦ seed만 바꿔도
def seed_spread(Xtr, ytr, Xva, yva):
    seeds = [0, 1, 2, 3, 4, 14]
    scores = []
    for s in seeds:
        m = train_tiny(Xtr, ytr, Xva, yva, seed=s, epochs=15)
        with torch.no_grad():
            p = m(Xva).argmax(1)
        scores.append(f1_score(yva.numpy(), p.numpy(), average="macro"))
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.scatter(scores, np.zeros(len(scores)), s=120, color=C["purple"], zorder=5)
    order = np.argsort(scores)
    for rank, j in enumerate(order):
        s, v = seeds[j], scores[j]
        up = rank % 2 == 0
        ax.annotate(f"seed {s}\n{v:.3f}", (v, 0), xytext=(0, 14 if up else -16), textcoords="offset points",
                    ha="center", va="bottom" if up else "top", fontsize=9)
    lo, hi = min(scores), max(scores)
    ax.annotate("", xy=(lo, -0.75), xytext=(hi, -0.75), arrowprops=dict(arrowstyle="<->", color=C["red"], lw=1.6))
    ax.text((lo + hi) / 2, -0.85, f"차이 {hi - lo:.3f}", ha="center", va="top", color=C["red"], fontweight="bold")
    ax.set_ylim(-1.2, 1.0); ax.set_yticks([])
    ax.set_xlim(lo - 0.05, hi + 0.05)
    ax.set_xlabel("valid Macro F1")
    ax.set_title("데이터·모델·epoch(15)·lr 모두 같고 seed만 다를 때", fontsize=11.5)
    ax.spines["left"].set_visible(False)
    save(fig, DAY, "seed_spread")
    return dict(zip(seeds, scores))


if __name__ == "__main__":
    print(dataset_counts())
    print(params_bars())
    lr_groups()
    sampler_batches()
    metric_trap()
    cm_heatmap(); prf_bars(); f1_harmonic()
    Xtr, ytr, _ = load_split("train")
    Xva, yva, raw_va = load_split("valid")
    model, curve, bep = train_tiny(Xtr, ytr, Xva, yva, seed=4, epochs=15, keep_curve=True)
    best_epoch(curve, bep)
    pred, conf, nw = error_gallery(model, Xva, yva, raw_va)
    print("best epoch", bep + 1, "wrong", nw, "macroF1", f1_score(yva.numpy(), pred.numpy(), average="macro"))
    print(confusion_matrix(yva.numpy(), pred.numpy()))
    gradcam_demo(model, Xva, yva, pred)
    gradcam_targets(model, Xva, yva, pred)
    print(seed_spread(Xtr, ytr, Xva, yva))
