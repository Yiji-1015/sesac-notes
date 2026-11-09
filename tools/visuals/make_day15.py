import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day15 그림 만들기 (객체탐지 · IoU · mAP · YOLO 격자 · conf/NMS · 추적).

데이터: 수업용 합성 도형 데이터(160x128) + 직접 100 epoch 학습한 작은 데모 모델(YOLO26n).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day15.py
"""
import time
import csv
from pathlib import Path

import numpy as np
import cv2
from matplotlib.patches import Rectangle, FancyArrowPatch

RUN = Path(os.environ.get(
    "DAY15_RUN",
    r"C:/Users/pc_28/AppData/Local/Temp/claude/e384ea84ee9f536f/4c44c0e3-ebd2-483a-a969-2ab2aed04c8a/scratchpad/day15run"))
DATA = RUN / "data"
WEIGHTS = RUN / "runs/ep100/weights/best.pt"
NAMES = {0: "square", 1: "circle", 2: "triangle"}
CLS_COL = {0: C["red"], 1: C["green"], 2: C["blue"]}
W, H = 160, 128
UP = 4  # 작은 이미지를 nearest 로 4배 확대해서 표시

np.random.seed(0)


# ---------------------------------------------------------------- 공통 도우미
def show(ax, bgr, title=None):
    """BGR 이미지를 RGB로 바꾸고 nearest 로 확대해서 원본 좌표계(0~160, 0~128)로 표시."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    big = cv2.resize(rgb, (rgb.shape[1] * UP, rgb.shape[0] * UP), interpolation=cv2.INTER_NEAREST)
    ax.imshow(big, extent=(0, rgb.shape[1], rgb.shape[0], 0), interpolation="nearest")
    ax.set_xlim(0, rgb.shape[1]); ax.set_ylim(rgb.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#cfc8e0")
    if title:
        ax.set_title(title, fontsize=11.5)


def box(ax, b, color, lw=2.2, ls="-", label=None, fill=False, alpha=1.0, fs=9, lpos="top"):
    x1, y1, x2, y2 = b
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=fill, ec=color, fc=color if fill else "none",
                           lw=lw, ls=ls, alpha=alpha))
    if label:
        ty = y1 - 1.5 if lpos == "top" else y2 + 1.5
        va = "bottom" if lpos == "top" else "top"
        ax.text(x1, ty, label, color="white", fontsize=fs, fontweight="bold", va=va, ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.95), clip_on=False)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def yolo_labels(txt):
    out = []
    for line in Path(txt).read_text().split("\n"):
        if line.strip():
            c, cx, cy, w, h = map(float, line.split())
            out.append((int(c), ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)))
    return out


def nms(cands, thr):
    """cands: list of (cls, conf, box). 같은 클래스끼리만 지움."""
    keep = []
    for c in sorted(cands, key=lambda t: -t[1]):
        if all(not (k[0] == c[0] and iou(k[2], c[2]) >= thr) for k in keep):
            keep.append(c)
    return keep


def match(preds, gts, thr=0.5):
    """점수 높은 예측부터 같은 클래스 정답과 1:1 매칭. 반환: 예측별 'TP'/'FP', 남은 정답 index."""
    used, tags = set(), []
    for cl, cf, b in sorted(preds, key=lambda t: -t[1]):
        best, bi = 0, None
        for i, (gc, gb) in enumerate(gts):
            if gc == cl and i not in used:
                v = iou(b, gb)
                if v > best:
                    best, bi = v, i
        if bi is not None and best >= thr:
            used.add(bi); tags.append(((cl, cf, b), "TP", best))
        else:
            anyv = max([iou(b, gb) for gc, gb in gts if gc == cl] + [0])
            tags.append(((cl, cf, b), "FP", anyv))
    fn = [i for i in range(len(gts)) if i not in used]
    return tags, fn


scene = cv2.imread(str(DATA / "scene.png"))
GT_SCENE = [(int(r["class_id"]), tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
            for r in read_csv(DATA / "boxes_gt.csv")]
FIX = [(int(r["class_id"]), float(r["confidence"]), tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2")))
       for r in read_csv(DATA / "boxes_fixture.csv")]


# ======================================================== 1. 박스 좌표 세 가지
def fig_box_formats():
    x1, y1, x2, y2 = GT_SCENE[0][1]       # square: 12,18,52,58
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    fig, (ax, at) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})
    show(ax, scene, "같은 빨간 사각형 박스를 숫자로 적는 3가지 방법")
    box(ax, (x1, y1, x2, y2), C["purple"], lw=2.5)
    # 모서리 점
    for (px, py, t, ha, dy) in [(x1, y1, f"(x1, y1) = ({x1:.0f}, {y1:.0f})", "left", -3),
                                (x2, y2, f"(x2, y2) = ({x2:.0f}, {y2:.0f})", "left", 8)]:
        ax.plot(px, py, "o", ms=8, color=C["purple"], mec="white", mew=1.5)
        ax.text(px + (2 if ha == "left" else -2), py + dy, t, color=C["purple"], fontsize=10.5,
                fontweight="bold", ha=ha, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    # 중심과 w, h
    ax.plot(cx, cy, "o", ms=8, color=C["pink"], mec="white", mew=1.5)
    ax.text(cx, cy + 4, f"중심 ({cx:.0f}, {cy:.0f})", color=C["pink"], fontsize=10, fontweight="bold",
            va="top", ha="center", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.9, pad=1))
    ax.annotate("", xy=(x2, y2 + 14), xytext=(x1, y2 + 14), arrowprops=dict(arrowstyle="<->", color=C["mint"], lw=2))
    ax.text(cx, y2 + 16, f"w = {w:.0f}", color=C["mint"], ha="center", va="top", fontsize=10.5, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.annotate("", xy=(x2 + 6, y2), xytext=(x2 + 6, y1), arrowprops=dict(arrowstyle="<->", color=C["mint"], lw=2))
    ax.text(x2 + 8, cy - 8, f"h = {h:.0f}", color=C["mint"], va="center", fontsize=10.5, fontweight="bold",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    # 사진 크기
    ax.text(W - 2, H - 3, f"사진 W={W}, H={H}", ha="right", va="bottom", fontsize=9.5, color=C["ink"],
            bbox=dict(fc="white", ec="#cfc8e0", pad=2))

    at.axis("off"); at.set_xlim(0, 1); at.set_ylim(0, 1)
    rows = [
        ("xyxy  (모서리 두 점)", f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]", "그리기 · IoU 계산", C["purple"]),
        ("cxcywh  (중심 + 크기)", f"[{cx:.0f}, {cy:.0f}, {w:.0f}, {h:.0f}]", "모델이 예측하는 모양", C["pink"]),
        ("정규화  (÷W, ÷H)", f"[{cx / W:.3f}, {cy / H:.3f}, {w / W:.3f}, {h / H:.3f}]", "YOLO 라벨 txt", C["mint"]),
    ]
    for i, (name, val, use, col) in enumerate(rows):
        y = 0.93 - i * 0.29
        at.add_patch(Rectangle((0.0, y - 0.2), 1.0, 0.25, fc=col, alpha=0.09, ec=col, lw=1.4,
                               transform=at.transAxes))
        at.text(0.04, y, name, color=col, fontsize=12, fontweight="bold", va="center")
        at.text(0.04, y - 0.09, val, color=C["ink"], fontsize=12.5, family="Consolas", va="center")
        at.text(0.96, y - 0.09, use, color="#5b5270", fontsize=9.5, ha="right", va="center")
    at.text(0.0, 0.08, f"cx = 32/160 = 0.200   cy = 38/128 = 0.297\n"
                        f"x 쪽은 W(160)로, y 쪽은 H(128)로 나눈다", fontsize=9.5, color="#5b5270", va="top")
    save(fig, 15, "box_formats")


# ======================================================== 2. IoU 예시
def fig_iou_examples():
    gt = (30, 30, 70, 70)
    cases = []
    for target in (0.8, 0.5, 0.2):
        d = 40 * (1 - target) / (1 + target)       # 같은 크기, 오른쪽으로 d 만큼 이동
        cases.append(((30 + d, 30, 70 + d, 70), f"오른쪽으로 {d:.0f}px 밀림"))
    cases.append(((10, 10, 90, 90), "중심은 같고 크기만 2배"))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, (pb, desc) in zip(axs, cases):
        v = iou(gt, pb)
        ax.set_xlim(0, 100); ax.set_ylim(108, -8); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("#cfc8e0")
        ix1, iy1 = max(gt[0], pb[0]), max(gt[1], pb[1]); ix2, iy2 = min(gt[2], pb[2]), min(gt[3], pb[3])
        box(ax, gt, C["green"], fill=True, alpha=0.12, lw=0)
        box(ax, pb, C["blue"], fill=True, alpha=0.12, lw=0)
        if ix2 > ix1 and iy2 > iy1:
            box(ax, (ix1, iy1, ix2, iy2), C["yellow"], fill=True, alpha=0.65, lw=0)
        box(ax, gt, C["green"], lw=2.5, ls="--")
        box(ax, pb, C["blue"], lw=2.5)
        ok = v >= 0.5 - 1e-9
        ax.set_title(f"IoU = {v:.2f}", fontsize=15, color=C["green"] if ok else C["red"])
        ax.text(50, 106, desc, ha="center", va="bottom", fontsize=10, color=C["ink"])
        ax.text(50, -6, "기준 0.5 이상 → TP" if ok else "기준 0.5 미만 → FP", ha="center", va="top", fontsize=10.5,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=C["green"] if ok else C["red"], ec="none"))
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc="none", ec=C["green"], ls="--", lw=2, label="정답(GT)"),
                        Rectangle((0, 0), 1, 1, fc="none", ec=C["blue"], lw=2, label="예측"),
                        Rectangle((0, 0), 1, 1, fc=C["yellow"], alpha=0.65, label="겹친 부분")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=10.5)
    save(fig, 15, "iou_examples")


# ======================================================== 3. TP / FP / FN
def fig_tp_fp_fn():
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 5))
    sets = [("NMS 전: 예측 4개 그대로 채점", FIX), ("NMS 후: 중복 박스 정리 뒤 채점", nms(FIX, 0.7))]
    for ax, (title, preds) in zip(axs, sets):
        show(ax, scene, title)
        tags, fn = match(preds, GT_SCENE, 0.5)
        for gi, (gc, gb) in enumerate(GT_SCENE):
            box(ax, gb, C["ink"], lw=1.6, ls="--")
        for gi in fn:
            gb = GT_SCENE[gi][1]
            box(ax, gb, C["red"], lw=2.8, ls="--", label=f"FN 놓침 ({NAMES[GT_SCENE[gi][0]]})", lpos="bottom")
        for (cl, cf, b), t, v in tags:
            col = C["green"] if t == "TP" else C["orange"]
            why = ""
            if t == "FP":
                why = " 중복" if v >= 0.5 else " 엉뚱한 곳"
            lp = "bottom" if (cl == 0 and cf < 0.9) else "top"
            box(ax, b, col, lw=2.6, label=f"{t}{why} {NAMES[cl]} {cf:.2f}", lpos=lp, fs=8.5)
        ntp = sum(t == "TP" for _, t, _ in tags); nfp = len(tags) - ntp; nfn = len(fn)
        p, r = ntp / (ntp + nfp), ntp / (ntp + nfn)
        ax.text(W / 2, H + 4, f"TP {ntp} · FP {nfp} · FN {nfn}   →   Precision {ntp}/{ntp + nfp} = {p:.2f},  "
                              f"Recall {ntp}/{ntp + nfn} = {r:.2f}",
                ha="center", va="top", fontsize=10.5, color=C["ink"], fontweight="bold")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc="none", ec=C["ink"], ls="--", lw=1.6, label="정답(점선)"),
                        Rectangle((0, 0), 1, 1, fc="none", ec=C["green"], lw=2.5, label="TP 맞춤"),
                        Rectangle((0, 0), 1, 1, fc="none", ec=C["orange"], lw=2.5, label="FP 헛탐지"),
                        Rectangle((0, 0), 1, 1, fc="none", ec=C["red"], ls="--", lw=2.5, label="FN 놓친 정답")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08), fontsize=10.5)
    save(fig, 15, "tp_fp_fn")


# ======================================================== 4. PR 곡선과 AP
def fig_pr_curve():
    n_gt = 8
    confs = [0.97, 0.93, 0.90, 0.84, 0.80, 0.71, 0.66, 0.55, 0.48, 0.40, 0.31, 0.22]
    hits = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0]
    tp = np.cumsum(hits); fp = np.cumsum(1 - np.array(hits))
    prec = tp / (tp + fp); rec = tp / n_gt
    # all-point 보간 (오른쪽에서 왼쪽으로 최댓값)
    env = np.maximum.accumulate(prec[::-1])[::-1]
    r_pts = np.concatenate([[0], rec]); p_pts = np.concatenate([[env[0]], env])
    ap = float(np.sum((r_pts[1:] - r_pts[:-1]) * p_pts[1:]))

    fig, (ax, at) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.fill_between(r_pts, p_pts, step="pre", color=C["purple"], alpha=0.15, label=f"AP = 넓이 = {ap:.2f}")
    ax.step(r_pts, p_pts, where="pre", color=C["purple"], lw=2.5)
    ax.plot(rec, prec, "o", color=C["pink"], ms=6, zorder=3, label="conf 기준을 하나씩 낮출 때의 (R, P)")
    offs = {0: (14, -36), 4: (10, 22), 8: (16, 14), 11: (16, -26)}
    for i in (0, 4, 8, 11):
        ax.annotate(f"conf ≥ {confs[i]:.2f}\nP {prec[i]:.2f} · R {rec[i]:.2f}", (rec[i], prec[i]),
                    xytext=offs[i], textcoords="offset points", fontsize=9,
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=1),
                    arrowprops=dict(arrowstyle="-", color="#9a90b5"), color=C["ink"])
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.08)
    ax.set_xlabel("Recall (정답 중 찾은 비율)"); ax.set_ylabel("Precision (찾은 것 중 진짜 비율)")
    ax.set_title("한 클래스의 Precision-Recall 곡선과 AP")
    ax.annotate("conf를 낮출수록\n오른쪽(Recall↑)으로 가고\n아래(Precision↓)로 내려감", xy=(0.74, 0.45),
                xytext=(0.36, 0.17), fontsize=9.5, color=C["purple"],
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=1.5))
    ax.legend(loc="lower left", fontsize=9.5)

    at.axis("off"); at.set_xlim(0, 1); at.set_ylim(0, len(confs) + 1.5)
    at.text(0, len(confs) + 1, f"예측을 점수순으로 줄 세우기 (정답 {n_gt}개)", fontsize=11, fontweight="bold",
            color=C["ink"], va="center")
    for i, (cf, h) in enumerate(zip(confs, hits)):
        y = len(confs) - i
        col = C["green"] if h else C["orange"]
        at.text(0.02, y, f"{i + 1:>2}. conf {cf:.2f}", fontsize=10, va="center", family="Consolas")
        at.text(0.42, y, "TP" if h else "FP", fontsize=10, va="center", color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="none"))
        at.text(0.56, y, f"P {prec[i]:.2f}  R {rec[i]:.2f}", fontsize=9.5, va="center", color="#5b5270",
                family="Consolas")
    at.text(0, -0.2, "예시 숫자(가상 예측 12개). 위에서부터 하나씩 포함하며 P, R을 다시 계산.", fontsize=8.8,
            color="#5b5270", va="center")
    save(fig, 15, "pr_curve_ap")


# ======================================================== 5. mAP50 vs mAP50-95
def fig_map_thresholds():
    ious = [0.93, 0.89, 0.86, 0.81, 0.77, 0.74, 0.69, 0.64, 0.58, 0.53]   # 예측 10개가 각자 정답과 겹친 정도
    thrs = np.round(np.arange(0.50, 0.96, 0.05), 2)
    tp = np.array([sum(v >= t for v in ious) for t in thrs])
    score = tp / len(ious)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    # 왼쪽: 예측 10개의 IoU
    y = np.arange(len(ious))[::-1]
    a1.barh(y, ious, color=C["blue"], alpha=0.8, height=0.65)
    for yy, v in zip(y, ious):
        a1.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=9)
    a1.axvline(0.5, color=C["green"], ls="--", lw=2); a1.text(0.5, len(ious) - 0.3, "0.50", color=C["green"],
                                                          ha="center", fontsize=9.5, fontweight="bold")
    a1.axvline(0.75, color=C["orange"], ls="--", lw=2); a1.text(0.75, len(ious) - 0.3, "0.75", color=C["orange"],
                                                            ha="center", fontsize=9.5, fontweight="bold")
    a1.axvline(0.95, color=C["red"], ls="--", lw=2); a1.text(0.95, len(ious) - 0.3, "0.95", color=C["red"],
                                                         ha="center", fontsize=9.5, fontweight="bold")
    a1.set_xlim(0, 1.05); a1.set_yticks(y); a1.set_yticklabels([f"예측 {i + 1}" for i in range(len(ious))],
                                                               fontsize=9)
    a1.set_ylim(-0.6, len(ious) + 0.2)
    a1.set_xlabel("정답과의 IoU"); a1.set_title("같은 예측 10개 (모두 클래스는 맞음)")
    # 오른쪽: 기준별 TP 비율
    cols = [C["green"] if t == 0.5 else (C["red"] if t == 0.95 else C["purple"]) for t in thrs]
    a2.bar([f"{t:.2f}" for t in thrs], score, color=cols, alpha=0.85)
    for i, (s, n) in enumerate(zip(score, tp)):
        a2.text(i, s + 0.02, f"TP {n}", ha="center", fontsize=9, color=C["ink"])
    m = score.mean()
    a2.axhline(m, color=C["pink"], lw=2, ls="--")
    a2.text(9.45, m + 0.03, f"10개 평균 = {m:.2f}\n(mAP50-95 쪽)", color=C["pink"], ha="right", fontsize=10,
            fontweight="bold")
    a2.text(0, score[0] + 0.1, f"mAP50 쪽\n= {score[0]:.2f}", color=C["green"], ha="center", fontsize=10,
            fontweight="bold")
    a2.set_ylim(0, 1.3); a2.set_xlabel("평가 IoU 기준 (이 값 이상이어야 TP)")
    a2.set_ylabel("TP 비율 (단순화한 점수)")
    a2.set_title("기준을 빡빡하게 올리면 같은 예측도 TP → FP")
    save(fig, 15, "map50_vs_map50_95")


# ======================================================== 모델 관련 (실제 데모 모델)
def load_model():
    from ultralytics import YOLO
    return YOLO(str(WEIGHTS))


def raw_candidates(model, bgr):
    """후처리 전 원본 출력 420개: (cx, cy, w, h, s0, s1, s2)."""
    import torch
    net = model.model.eval().cpu()
    x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        out = net(x)[0][0].numpy()          # (7, 420)
    return out.T


def to_cands(raw):
    res = []
    for cx, cy, w, h, *s in raw:
        cl = int(np.argmax(s)); res.append((cl, float(s[cl]), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return res


SCALES = [(20, 16, 8, 0, 320), (10, 8, 16, 320, 400), (5, 4, 32, 400, 420)]


def fig_grid_scales(model, img_name="basic_val_005"):
    bgr = cv2.imread(str(DATA / f"images/val/{img_name}.png"))
    gts = yolo_labels(DATA / f"labels/val/{img_name}.txt")
    raw = raw_candidates(model, bgr)
    best = raw[:, 4:].max(1)
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
    labels = ["작은 물체 담당", "중간 물체 담당", "큰 물체 담당"]
    for ax, (gw, gh, st, a, b), lab in zip(axs, SCALES, labels):
        show(ax, bgr, f"{gw}×{gh} 격자 · 칸 {st}px · 후보 {b - a}개\n({lab})")
        sc = best[a:b].reshape(gh, gw)
        ov = np.zeros((gh, gw, 4)); ov[..., :3] = matplotlib_rgb(C["purple"])
        ov[..., 3] = np.clip(sc / best.max(), 0, 1) ** 0.5 * 0.8
        ax.imshow(ov, extent=(0, W, H, 0), interpolation="nearest", zorder=2)
        for gx in range(gw + 1):
            ax.axvline(gx * st, color="white", lw=1.2 if st > 8 else 0.6, alpha=0.95, zorder=3)
        for gy in range(gh + 1):
            ax.axhline(gy * st, color="white", lw=1.2 if st > 8 else 0.6, alpha=0.95, zorder=3)
        for gc, gb in gts:
            mx, my = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
            ax.add_patch(Rectangle((mx // st * st, my // st * st), st, st, fill=False, ec=C["yellow"], lw=2.5, zorder=4))
            ax.plot(mx, my, "x", color="black", ms=9, mew=2.5, zorder=5)
    fig.text(0.5, -0.02, "칸 하나 = 후보 하나.  보라색이 진할수록 그 칸의 점수가 높음(작은 데모 모델의 실제 출력).  "
                         "× = 정답 물체의 중심, 노란 칸 = 그 중심이 들어간 칸.   320 + 80 + 20 = 420개",
             ha="center", fontsize=10.5, color=C["ink"])
    save(fig, 15, "yolo_grid_scales")


def matplotlib_rgb(hexcol):
    h = hexcol.lstrip("#"); return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def fig_candidate_sim():
    """강의 요약의 420 → 11 → 4 흐름을 그대로 재현한 모의 그림 (점수는 손으로 정한 값)."""
    rng = np.random.default_rng(0)
    cands = []
    # 420칸 모두 후보 하나씩: 낮은 점수
    for gw, gh, st, a, b in SCALES:
        for gy in range(gh):
            for gx in range(gw):
                cx, cy = (gx + 0.5) * st, (gy + 0.5) * st
                near = min(np.hypot(cx - (gb[0] + gb[2]) / 2, cy - (gb[1] + gb[3]) / 2) for _, gb in GT_SCENE)
                s = rng.uniform(0.0, 0.03) + (0.2 * np.exp(-near / 10) if near < 30 else 0)
                cl = int(rng.integers(0, 3))
                cands.append((cl, float(min(s, 0.24)), (cx - st, cy - st, cx + st, cy + st)))
    # 물체 주변의 강한 후보 11개 (conf 0.25 이상)
    sq, ci, tr = GT_SCENE[0][1], GT_SCENE[1][1], GT_SCENE[2][1]
    sh = lambda bx, dx, dy, g=0: (bx[0] + dx - g, bx[1] + dy - g, bx[2] + dx + g, bx[3] + dy + g)
    strong = [(0, 0.85, sh(sq, 1, 0)), (0, 0.61, sh(sq, 0, 1)), (0, 0.42, sh(sq, 2, 1)), (0, 0.30, sh(sq, 1, -1, 1)),
              (1, 0.85, sh(ci, 0, 1)), (1, 0.52, sh(ci, 2, 0)), (1, 0.37, sh(ci, 9, 6)),
              (2, 0.89, sh(tr, 0, 0)), (2, 0.64, sh(tr, 1, 2)), (2, 0.45, sh(tr, -2, 0, 1)), (2, 0.28, sh(tr, 2, -1))]
    cands = cands[:420 - len(strong)] + strong
    passed = [c for c in cands if c[1] >= 0.25]
    final = nms(passed, 0.7)
    tags, fn = match(final, GT_SCENE, 0.5)
    print(f"[sim] {len(cands)} -> {len(passed)} -> {len(final)}", [(round(t[0][1], 2), t[1]) for t in tags])
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.7))
    show(axs[0], scene, f"① 격자 칸마다 후보 → {len(cands)}개")
    xs = [(c[2][0] + c[2][2]) / 2 for c in cands]; ys = [(c[2][1] + c[2][3]) / 2 for c in cands]
    axs[0].scatter(xs, ys, s=[4 + 260 * c[1] for c in cands], c=[C["purple"]] * len(cands), alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=3)
    axs[0].text(W / 2, H + 3, "점 = 후보 하나 · 클수록 점수 높음", ha="center", va="top", fontsize=9.5)
    show(axs[1], scene, f"② conf ≥ 0.25 → {len(passed)}개")
    for cl, cf, b in sorted(passed, key=lambda t: t[1]):
        box(axs[1], b, C["purple"], lw=1.6, alpha=0.85)
    axs[1].text(W / 2, H + 3, "같은 물체에 박스가 3~4개씩 겹쳐 있음", ha="center", va="top", fontsize=9.5)
    show(axs[2], scene, f"③ NMS(iou 0.7) → {len(final)}개 → 채점")
    for gc, gb in GT_SCENE:
        box(axs[2], gb, C["ink"], lw=1.3, ls="--")
    for (cl, cf, b), t, v in tags:
        box(axs[2], b, C["green"] if t == "TP" else C["orange"], lw=2.6,
            label=f"{t} {NAMES[cl]} {cf:.2f}", fs=8.5, lpos="bottom" if t == "FP" else "top")
    axs[2].text(W / 2, H + 3, "덜 겹친 중복(IoU < 0.7)은 살아남아 FP가 됨", ha="center", va="top", fontsize=9.5)
    for i in range(2):
        fig.text((i + 1) / 3, 0.5, "→", fontsize=26, ha="center", va="center", color=C["purple"])
    fig.text(0.5, -0.03, "모의 그림: 강의 예시(420 → 11 → 4, TP 3 · FP 1)의 흐름을 재현하려고 점수를 손으로 정함. 실제 모델 출력 아님.",
             ha="center", fontsize=9.8, color="#5b5270")
    save(fig, 15, "candidate_pipeline_sim")


def fig_candidate_pipeline(model, img_name="basic_val_004", conf=0.02, nms_iou=0.7):
    bgr = cv2.imread(str(DATA / f"images/val/{img_name}.png"))
    gts = yolo_labels(DATA / f"labels/val/{img_name}.txt")
    raw = raw_candidates(model, bgr)
    cands = to_cands(raw)
    passed = [c for c in cands if c[1] >= conf]
    final = nms(passed, nms_iou)
    # 공식 경로와 개수가 같은지 확인
    r = model.predict(bgr, conf=conf, iou=nms_iou, imgsz=160, verbose=False, device="cpu")[0]
    print(f"[pipeline] raw {len(cands)} -> conf>={conf}: {len(passed)} -> NMS: {len(final)}  (ultralytics: {len(r.boxes)})")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
    show(axs[0], bgr, f"① 모델이 낸 후보 {len(cands)}개")
    xs = [(c[2][0] + c[2][2]) / 2 for c in cands]; ys = [(c[2][1] + c[2][3]) / 2 for c in cands]
    axs[0].scatter(xs, ys, s=[6 + 400 * c[1] for c in cands], c=[CLS_COL[c[0]] for c in cands], alpha=0.55,
                   edgecolors="white", linewidths=0.4, zorder=3)
    axs[0].text(W / 2, H + 3, "점 = 후보 하나의 박스 중심 · 클수록 점수 높음", ha="center", va="top", fontsize=9.5)
    show(axs[1], bgr, f"② conf ≥ {conf} 만 남기면 {len(passed)}개")
    for cl, cf, b in sorted(passed, key=lambda t: t[1]):
        box(axs[1], b, CLS_COL[cl], lw=1.6, alpha=0.9)
    axs[1].text(W / 2, H + 3, "같은 물체에 박스가 여러 개 겹쳐 있음", ha="center", va="top", fontsize=9.5)
    show(axs[2], bgr, f"③ NMS(iou {nms_iou}) 후 {len(final)}개 = 최종 결과")
    for gc, gb in gts:
        box(axs[2], gb, C["ink"], lw=1.3, ls="--")
    seen = []
    for cl, cf, b in sorted(final, key=lambda t: -t[1]):
        lp = "bottom" if any(abs(b[0] - s[0]) < 25 and abs(b[1] - s[1]) < 10 for s in seen) else "top"
        seen.append(b)
        box(axs[2], b, CLS_COL[cl], lw=2.6, label=f"{NAMES[cl]} {cf:.2f}" if cf >= 0.1 else None, fs=8.5, lpos=lp)
    axs[2].text(W / 2, H + 3, "점선 = 정답 · 이름표는 0.1 이상만 · 덜 겹친 중복은 남음", ha="center", va="top", fontsize=9.5)
    for i in range(2):
        fig.text((i + 1) / 3, 0.5, "→", fontsize=26, ha="center", va="center", color=C["purple"])
    fig.text(0.5, -0.03, f"작은 데모 모델의 실제 출력. 점수가 낮은 모델이라 conf를 {conf}로 낮춰서 보여 줌 (박스 색 = 클래스).",
             ha="center", fontsize=9.8, color="#5b5270")
    save(fig, 15, "candidate_pipeline")
    return len(cands), len(passed), len(final)


def fig_nms_steps():
    cands = sorted(FIX, key=lambda t: -t[1])
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6))
    # step 0: 전체 후보
    show(axs[0], scene, "0. 후보를 점수순으로 줄 세움")
    for k, (cl, cf, b) in enumerate(cands):
        box(axs[0], b, C["purple"], lw=2, label=f"#{k + 1} {NAMES[cl]} {cf:.2f}", fs=8.5,
            lpos="bottom" if k in (1, 3) else "top")
    # step 1
    a = axs[1]; show(a, scene, "1. 1등(0.95) 남김 → 같은 클래스와 IoU 비교")
    v = iou(cands[0][2], cands[1][2])
    box(a, cands[1][2], C["gray"], lw=2, ls=":", label=f"0.85  IoU {v:.2f} ≥ 0.7 → 삭제", fs=8.5, lpos="bottom")
    b = cands[1][2]; a.plot([b[0], b[2]], [b[1], b[3]], color=C["red"], lw=2.5); a.plot([b[0], b[2]], [b[3], b[1]],
                                                                                         color=C["red"], lw=2.5)
    box(a, cands[0][2], C["green"], lw=3, label="남김 0.95", fs=8.5)
    for cl, cf, bb in cands[2:]:
        box(a, bb, C["purple"], lw=1.4, alpha=0.6)
    a.text(W / 2, H + 3, "아직 남은 후보(보라)는 다음 차례에 확인", ha="center", va="top", fontsize=9)
    # step 2
    a = axs[2]; show(a, scene, "2. 남은 것 중 1등(circle 0.80) 남김")
    box(a, cands[0][2], C["green"], lw=3, label="0.95")
    box(a, cands[2][2], C["green"], lw=3, label="남김 0.80", fs=8.5)
    box(a, cands[3][2], C["purple"], lw=1.4, alpha=0.6)
    a.text(W / 2, H + 3, "같은 클래스(circle) 다른 후보가 없어 지울 것 없음", ha="center", va="top", fontsize=9)
    # step 3
    a = axs[3]; show(a, scene, "3. 다음(triangle 0.40)도 남김 → 끝")
    for cl, cf, bb in [cands[0], cands[2], cands[3]]:
        box(a, bb, C["green"], lw=3, label=f"{cf:.2f}", fs=8.5, lpos="bottom" if cf < 0.5 else "top")
    a.text(W / 2, H + 3, "후보 4개 → 3개.  NMS는 '맞았나'는 모름(0.40은 엉뚱한 곳)", ha="center", va="top", fontsize=9)
    save(fig, 15, "nms_steps")


def fig_conf_compare(model, img_name="basic_val_005"):
    bgr = cv2.imread(str(DATA / f"images/val/{img_name}.png"))
    gts = yolo_labels(DATA / f"labels/val/{img_name}.txt")
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.7))
    for ax, cf in zip(axs, (0.05, 0.25, 0.5)):
        r = model.predict(bgr, conf=cf, imgsz=160, verbose=False, device="cpu")[0]
        preds = [(int(c), float(s), tuple(map(float, b))) for b, s, c in
                 zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())]
        tags, fn = match(preds, gts, 0.5)
        ntp = sum(t == "TP" for _, t, _ in tags)
        show(ax, bgr, f"conf = {cf}  →  박스 {len(preds)}개")
        for gc, gb in gts:
            box(ax, gb, C["ink"], lw=1.3, ls="--")
        seen = []
        for (cl, s, b), t, v in tags:
            lp = "bottom" if any(abs(b[0] - q[0]) < 25 and abs(b[1] - q[1]) < 10 for q in seen) else "top"
            seen.append(b)
            box(ax, b, C["green"] if t == "TP" else C["orange"], lw=2.6, label=f"{t} {NAMES[cl]} {s:.2f}", fs=8.5,
                lpos=lp)
        p = f"{ntp}/{len(preds)}" if preds else "-"
        ax.text(W / 2, H + 3, f"TP {ntp} · FP {len(preds) - ntp} · FN {len(fn)}   (P {p}, R {ntp}/{len(gts)})",
                ha="center", va="top", fontsize=10, fontweight="bold")
    fig.text(0.5, -0.04, "같은 사진·같은 모델. conf만 바꿈.  점선 = 정답 3개, 초록 = TP, 주황 = FP.  "
                         "(합성 도형으로 짧게 학습한 작은 데모 모델이라 점수가 전반적으로 낮음)",
             ha="center", fontsize=9.8, color="#5b5270")
    save(fig, 15, "conf_compare")


def fig_train_curves():
    rows = read_csv(RUN / "runs/ep100/results.csv")
    ep = [int(r["epoch"]) for r in rows]
    g = lambda k: [float(r[k]) for r in rows]
    m50, m5095 = g("metrics/mAP50(B)"), g("metrics/mAP50-95(B)")
    tl, vl = g("train/box_loss"), g("val/box_loss")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(ep, tl, color=C["purple"], lw=2, label="train box_loss")
    a1.plot(ep, vl, color=C["orange"], lw=2, label="val box_loss")
    a1.set_xlabel("epoch"); a1.set_title("loss: 낮아질수록 박스를 잘 맞춤"); a1.legend()
    a2.plot(ep, m50, color=C["green"], lw=2.2, label="mAP50")
    a2.plot(ep, m5095, color=C["pink"], lw=2.2, label="mAP50-95")
    bi = int(np.argmax(m5095))
    a2.plot(ep[bi], m5095[bi], "o", color=C["pink"], ms=9, mec="white", mew=2)
    a2.annotate(f"best.pt 후보\nepoch {ep[bi]}", (ep[bi], m5095[bi]), xytext=(-70, 35), textcoords="offset points",
                fontsize=9.5, arrowprops=dict(arrowstyle="->", color=C["ink"]))
    a2.annotate("", xy=(ep[-1] + 2, m50[-1]), xytext=(ep[-1] + 2, m5095[-1]),
                arrowprops=dict(arrowstyle="<->", color=C["ink"], lw=1.6))
    a2.text(ep[-1] - 1, (m50[-1] + m5095[-1]) / 2 + 0.02, "간격 = 박스가\n엉성한 정도", ha="right", va="center",
            fontsize=9.5, color=C["ink"], bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    a2.text(25, 0.05, "처음 ~50 epoch: 아직 하나도 못 찾음(0)", ha="center", fontsize=9, color="#5b5270")
    a2.set_ylim(0, 1); a2.set_xlabel("epoch"); a2.set_title("val 점수: mAP50이 항상 mAP50-95보다 위")
    a2.legend(loc="upper right")
    save(fig, 15, "train_curves")


def fig_track_frames():
    rows = read_csv(DATA / "track_ground_truth.csv")
    tr = {}
    for r in rows:
        tr.setdefault(int(r["gt_id"]), []).append((int(r["frame"]), int(r["class_id"]),
                                                   tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
    cap = cv2.VideoCapture(str(DATA / "demo.avi"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    idcol = {1: C["purple"], 2: C["orange"], 3: C["mint"]}
    pick = [0, 15, 29]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    # 마지막 칸: 30프레임 경로를 한 장에
    ax = axs[3]; show(ax, frames[29], "30프레임 경로 겹쳐 보기")
    for tid, seq in tr.items():
        pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for f, c, b in seq]
        ax.plot(*zip(*pts), "-o", color=idcol[tid], lw=2, ms=3.5, mec="white", mew=0.5)
        ax.annotate("", xy=pts[-1], xytext=pts[-4], arrowprops=dict(arrowstyle="-|>", color=idcol[tid], lw=2.5,
                                                                  mutation_scale=18))
        ax.text(pts[0][0], pts[0][1] - 6, f"ID {tid} 시작", color=idcol[tid], fontsize=8.5, fontweight="bold",
                ha="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    for ax, fi in zip(axs[:3], pick):
        show(ax, frames[fi], f"frame {fi}")
        for tid, seq in tr.items():
            pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for f, c, b in seq if f <= fi]
            if len(pts) > 1:
                ax.plot(*zip(*pts), color=idcol[tid], lw=2.2, alpha=0.9)
            cur = [(c, b) for f, c, b in seq if f == fi]
            if cur:
                c, b = cur[0]
                box(ax, b, idcol[tid], lw=2.4, label=f"ID {tid} · {NAMES[c]}", fs=8.5,
                    lpos="bottom" if b[1] < 14 else "top")
    fig.text(0.5, -0.03, "선 = 지금까지 지나온 경로.  프레임이 바뀌어도 같은 물체는 같은 ID(같은 색)를 유지해야 함.  "
                         "(영상의 정답 좌표로 그린 그림)", ha="center", fontsize=10, color=C["ink"])
    save(fig, 15, "track_frames")


def fig_speed(model):
    paths = sorted((DATA / "images/val").glob("*.png")) + sorted((DATA / "images/train").glob("*.png"))
    model.predict(str(paths[0]), imgsz=160, verbose=False, device="cpu")  # 워밍업
    rd, pre, inf, post = [], [], [], []
    for rep in range(3):
        for p in paths:
            t0 = time.perf_counter(); img = cv2.imread(str(p)); rd.append((time.perf_counter() - t0) * 1000)
            r = model.predict(img, imgsz=160, verbose=False, device="cpu")[0]
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
    vals = [np.median(rd), np.median(pre), np.median(inf), np.median(post)]
    names = ["읽기(imread)", "전처리", "모델 추론", "후처리(NMS)"]
    cols = [C["gray"], C["blue"], C["purple"], C["orange"]]
    print("[speed ms]", [round(v, 2) for v in vals])
    fig, ax = plt.subplots(figsize=(11, 3.3))
    left = 0
    for k, (v, n, c) in enumerate(zip(vals, names, cols)):
        ax.barh(0, v, left=left, color=c, height=0.5, edgecolor="white", lw=2)
        if v > sum(vals) * 0.3:
            ax.text(left + v / 2, 0, f"{n}\n{v:.1f}ms", ha="center", va="center", color="white", fontsize=9.5,
                    fontweight="bold")
        elif k == 1:   # 전처리: 아래쪽
            ax.annotate(f"{n} {v:.1f}ms", (left + v / 2, -0.25), xytext=(left + v / 2 + 0.3, -0.42),
                        color=c, fontsize=9, fontweight="bold", va="center",
                        arrowprops=dict(arrowstyle="-", color=c))
        else:
            ax.text(left + (0 if k == 0 else v / 2), 0.33, f"{n} {v:.1f}ms", ha="left" if k == 0 else "center",
                    va="bottom", color=c, fontsize=9, fontweight="bold")
        left += v
    tot = sum(vals)
    ax.barh(-0.75, vals[2], left=vals[0] + vals[1], color="none", edgecolor=C["purple"], hatch="///", height=0.4)
    ax.text(vals[0] + vals[1] + vals[2] + 0.4, -0.75,
            f"추론만 보면 {vals[2]:.1f}ms → 약 {1000 / vals[2]:.0f} FPS 처럼 보이지만", va="center", fontsize=10,
            color=C["purple"])
    ax.text(tot + 0.4, 0, f"합계 {tot:.1f}ms\n→ 실제 약 {1000 / tot:.0f} FPS", va="center", fontsize=10.5,
            fontweight="bold", color=C["ink"])
    ax.set_yticks([]); ax.set_xlim(0, tot * 1.6); ax.set_ylim(-1.2, 0.8)
    ax.set_xlabel("사진 1장 처리 시간 (ms, 중앙값)")
    ax.set_title("사진 1장이 결과가 되기까지 걸리는 시간 (CPU · 160px · 작은 데모 모델)")
    ax.grid(axis="y", visible=False)
    save(fig, 15, "speed_breakdown")


if __name__ == "__main__":
    fig_box_formats()
    fig_iou_examples()
    fig_tp_fp_fn()
    fig_pr_curve()
    fig_map_thresholds()
    fig_nms_steps()
    fig_train_curves()
    fig_track_frames()
    fig_candidate_sim()
    m = load_model()
    fig_grid_scales(m)
    fig_candidate_pipeline(m)
    fig_conf_compare(m)
    fig_speed(m)
