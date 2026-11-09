"""Day13 (OpenCV 전처리) 학습 노트용 그림 생성.

실행:
    C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day13.py
원본 데이터(C:/practice_assignment/20261104)는 읽기만 한다.
"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from matplotlib.patches import Rectangle

DAY = 13
DATA = Path("C:/practice_assignment/20261104/02_응용/data")
rng = np.random.default_rng(13)


def rd(path, flag=cv2.IMREAD_COLOR):
    """한글 경로에서도 읽히도록 fromfile + imdecode 사용."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)
    if img is None:
        raise RuntimeError(f"read failed: {path}")
    return img


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show(ax, img, title=None, cmap="gray", **kw):
    if img.ndim == 3:
        ax.imshow(rgb(img), **kw)
    else:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=255, **kw)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11)


scene = rd(DATA / "scene.png")                 # (240, 320, 3) BGR
noisy = rd(DATA / "noisy.png")
gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------- ① 픽셀 격자
def pixel_grid():
    red = (scene[:, :, 2] == 255) & (scene[:, :, 0] == 0) & (scene[:, :, 1] == 0)
    ys, xs = np.nonzero(red[:100, :100])
    n = 6
    y0, x0 = ys.max() - 2, xs.max() - 2       # 빨간 사각형 오른쪽 아래 모서리
    crop = scene[y0:y0 + n, x0:x0 + n]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    show(a1, scene, "원본 shape = (240, 320, 3)")
    a1.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), n, n, fill=False, ec=C["yellow"], lw=2.2))
    a1.annotate("여기 6×6칸을 확대 →", xy=(x0 + n, y0 + n), xytext=(75, 95),
                color="white", fontsize=10.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["yellow"], lw=2))
    a2.imshow(rgb(crop), interpolation="nearest")
    for y in range(n):
        for x in range(n):
            b, g, r = crop[y, x]
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            col = "white" if lum < 140 else C["ink"]
            a2.text(x, y, f"B {b}\nG {g}\nR {r}", ha="center", va="center", fontsize=7.2, color=col,
                    linespacing=1.15)
    a2.set_xticks(range(n)); a2.set_yticks(range(n))
    a2.set_xticklabels([str(x0 + i) for i in range(n)], fontsize=8)
    a2.set_yticklabels([str(y0 + i) for i in range(n)], fontsize=8)
    a2.set_xlabel("x (열, Width 방향)"); a2.set_ylabel("y (행, Height 방향)")
    a2.grid(False)
    a2.set_xticks(np.arange(-0.5, n), minor=True); a2.set_yticks(np.arange(-0.5, n), minor=True)
    a2.grid(which="minor", color="white", lw=1.2)
    a2.tick_params(which="minor", length=0)
    a2.set_title(f"img[{y0}:{y0+n}, {x0}:{x0+n}] · 칸마다 숫자 3개", fontsize=11)
    save(fig, DAY, "pixel_grid")


# ---------------------------------------------------------------- ① BGR 채널 분리
def bgr_channels():
    b, g, r = cv2.split(scene)
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    show(axs[0], scene, "원본 (컬러)")
    for ax, ch, name, col in zip(axs[1:], [b, g, r], ["B 채널 (파랑)", "G 채널 (초록)", "R 채널 (빨강)"],
                                 [C["blue"], C["green"], C["red"]]):
        show(ax, ch, name)
        ax.title.set_color(col)
    # 핵심 표시: 빨간 사각형은 R에서만 흰색
    for ax, txt in zip(axs[1:], ["빨강 칸 = 0", "빨강 칸 = 0", "빨강 칸 = 255"]):
        ax.add_patch(Rectangle((14, 14), 52, 42, fill=False, ec=C["yellow"], lw=2))
        ax.text(40, 74, txt, color=C["yellow"], fontsize=9, ha="center", fontweight="bold")
    fig.suptitle("cv2.split(img) → 채널마다 (240, 320) 흑백 지도. 밝을수록 그 색이 많다", fontsize=11.5, y=1.02)
    save(fig, DAY, "bgr_channels")


# ---------------------------------------------------------------- ② HSV: 그늘진 색 고르기
def hsv_lighting():
    h, w = gray.shape
    ramp = np.linspace(0.25, 1.0, w)[None, :, None]            # 왼쪽일수록 어둡게(그늘)
    shaded = np.clip(scene.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(shaded, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # BGR로 "주황"을 고정 범위로 잡기: 밝은 주황 값 근처만
    bgr_mask = cv2.inRange(shaded, (20, 140, 200), (90, 200, 255))
    hsv_mask = cv2.inRange(hsv, (10, 120, 40), (25, 255, 255))
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.4))
    show(axs[0, 0], shaded, "입력: 왼쪽에 그늘이 진 사진")
    show(axs[0, 1], H, "H (색 종류) · 0~179")
    show(axs[0, 2], V, "V (밝기) · 0~255")
    show(axs[1, 0], S, "S (진하기) · 0~255")
    show(axs[1, 1], bgr_mask, f"BGR 범위로 주황 찾기\n→ 흰 칸 {int((bgr_mask > 0).sum()):,}개")
    show(axs[1, 2], hsv_mask, f"HSV 범위(H 10~25)로 찾기\n→ 흰 칸 {int((hsv_mask > 0).sum()):,}개")
    axs[1, 1].title.set_color(C["red"]); axs[1, 2].title.set_color(C["green"])
    axs[0, 1].text(220, 230, "주황 사각형은 그늘이 져도\n같은 회색(약 H 18)", color=C["yellow"],
                   fontsize=8.5, ha="center", va="bottom", fontweight="bold")
    axs[0, 2].text(160, 230, "그늘은 V에만 나타남 (왼쪽이 어두움)", color=C["yellow"],
                   fontsize=8.5, ha="center", va="bottom", fontweight="bold")
    save(fig, DAY, "hsv_lighting")


# ---------------------------------------------------------------- ② 축소 보간 비교
def resize_shrink():
    n = 480
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r2 = (xx - n / 2) ** 2 + (yy - n / 2) ** 2
    zone = (127.5 + 127.5 * np.cos(r2 * np.pi / (n * 3.0))).astype(np.uint8)   # 바깥으로 갈수록 촘촘한 링
    out = 96
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.2))
    show(axs[0], zone, f"원본 {n}×{n}\n(바깥일수록 줄무늬가 촘촘)")
    for ax, (name, flag, col) in zip(axs[1:], [("INTER_NEAREST", cv2.INTER_NEAREST, C["red"]),
                                               ("INTER_LINEAR", cv2.INTER_LINEAR, C["orange"]),
                                               ("INTER_AREA", cv2.INTER_AREA, C["green"])]):
        small = cv2.resize(zone, (out, out), interpolation=flag)
        show(ax, small, f"{name}\n{out}×{out}로 축소", interpolation="nearest")
        ax.title.set_color(col)
    axs[1].text(48, 100, "가짜 무늬(물결)가 생김", color=C["red"], ha="center", va="top", fontsize=9.5,
                fontweight="bold")
    axs[3].text(48, 100, "면적 평균 → 회색으로 차분히", color=C["green"], ha="center", va="top",
                fontsize=9.5, fontweight="bold")
    save(fig, DAY, "resize_shrink")


# ---------------------------------------------------------------- ② Mask 확대: NEAREST vs LINEAR
def resize_mask():
    m = np.zeros((12, 12), np.uint8)
    m[3:9, 3:9] = 255
    big_n = cv2.resize(m, (96, 96), interpolation=cv2.INTER_NEAREST)
    big_l = cv2.resize(m, (96, 96), interpolation=cv2.INTER_LINEAR)
    un_n, un_l = np.unique(big_n), np.unique(big_l)
    fig = plt.figure(figsize=(10, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    a0, a1, a2, a3 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    show(a0, m, "원본 Mask 12×12\n값: 0, 255", interpolation="nearest")
    show(a1, big_n, f"NEAREST로 96×96\n값 종류 {len(un_n)}개", interpolation="nearest")
    show(a2, big_l, f"LINEAR로 96×96\n값 종류 {len(un_l)}개", interpolation="nearest")
    a1.title.set_color(C["green"]); a2.title.set_color(C["red"])
    a2.add_patch(Rectangle((60, 20), 20, 20, fill=False, ec=C["yellow"], lw=2))
    a2.text(70, 18, "경계에 회색", color=C["yellow"], ha="center", va="bottom", fontsize=9, fontweight="bold")
    row = 48
    a3.step(np.arange(96), big_n[row], where="mid", color=C["green"], lw=2.2, label="NEAREST")
    a3.plot(np.arange(96), big_l[row], color=C["red"], lw=2.2, label="LINEAR")
    a3.set_title(f"가운데 줄(y={row}) 값을 옆에서 본 모습", fontsize=11)
    a3.set_xlabel("x"); a3.set_ylabel("픽셀 값"); a3.set_ylim(-10, 290); a3.set_yticks([0, 128, 255])
    a3.legend(loc="center", fontsize=9)
    mid = int(((big_l > 0) & (big_l < 255)).sum())
    a3.text(48, 288, f"LINEAR: 0과 255 사이 값인 칸 {mid:,}개", ha="center", va="top", fontsize=9.5,
            color=C["red"], fontweight="bold")
    save(fig, DAY, "resize_mask")


# ---------------------------------------------------------------- ③ Blur Kernel 크기
def blur_kernels():
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    show(axs[0], scene, "원본")
    for ax, k in zip(axs[1:], [3, 7, 15]):
        show(ax, cv2.GaussianBlur(scene, (k, k), 0), f"GaussianBlur ({k}, {k})")
    for ax in axs:
        ax.add_patch(Rectangle((200, 10), 105, 45, fill=False, ec=C["yellow"], lw=1.8))
    for ax, note in zip(axs, ["줄무늬·경계 또렷", "거의 그대로", "줄무늬가 흐려지기 시작", "줄무늬가 회색으로 뭉개짐"]):
        ax.set_xlabel(note, fontsize=10, color=C["ink"])
    save(fig, DAY, "blur_kernels")


# ---------------------------------------------------------------- ③ 잡음 종류 × 필터
def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / mse)


def filter_noise():
    clean = gray
    gauss_noise = np.clip(clean + rng.normal(0, 22, clean.shape), 0, 255).astype(np.uint8)
    sp = clean.copy()
    u = rng.random(clean.shape)
    sp[u < 0.03] = 0
    sp[u > 0.97] = 255
    rows = [("고르게 퍼진 잡음\n(Gaussian noise)", gauss_noise), ("점 잡음\n(소금·후추)", sp)]
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.2))
    for r, (name, img) in enumerate(rows):
        outs = [("잡음 낀 입력", img),
                ("GaussianBlur (5,5)", cv2.GaussianBlur(img, (5, 5), 0)),
                ("medianBlur 5", cv2.medianBlur(img, 5)),
                ("bilateral d=9, 75, 75", cv2.bilateralFilter(img, 9, 75, 75))]
        scores = [psnr(o, clean) for _, o in outs]
        best = int(np.argmax(scores[1:])) + 1
        for c, ((t, o), sc) in enumerate(zip(outs, scores)):
            show(axs[r, c], o, f"{t}\n원본과의 PSNR {sc:.1f}dB")
            if c == best:
                axs[r, c].title.set_color(C["green"])
                for s in axs[r, c].spines.values():
                    s.set_visible(True); s.set_color(C["green"]); s.set_linewidth(3)
        axs[r, 0].set_ylabel(name, fontsize=11, rotation=0, ha="right", va="center", labelpad=12)
    fig.text(0.5, -0.02, "PSNR = 깨끗한 원본과 얼마나 비슷한가 (클수록 비슷) · 초록 테두리 = 그 줄에서 가장 비슷한 결과",
             ha="center", fontsize=10, color=C["gray"])
    save(fig, DAY, "filter_noise")


# ---------------------------------------------------------------- ③ Canny 기준값
def canny_thresholds():
    g = cv2.GaussianBlur(cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, (t1, t2, note) in zip(axs, [(20, 50, "낮게: 잡음 결까지 경계"),
                                        (70, 150, "강의 예시값"),
                                        (200, 400, "높게: 약한 경계를 놓침")]):
        e = cv2.Canny(g, t1, t2)
        show(ax, 255 - e, f"Canny({t1}, {t2}) · 경계 {int((e > 0).sum()):,}칸\n{note}")
    axs[2].text(160, 150, "원·사각형·색 칸의\n경계가 사라짐", color=C["red"], ha="center", va="center",
                fontsize=10, fontweight="bold")
    fig.text(0.5, -0.03, "보기 쉽게 반전: 검은 선 = 경계(255). 입력은 모두 같은 GaussianBlur(5,5) 결과",
             ha="center", fontsize=9.5, color=C["gray"])
    save(fig, DAY, "canny_thresholds")


def make_coins():
    """어두운 동전(물체) + 밝은 배경 + 잡음. 고정 시드."""
    r = np.random.default_rng(7)
    h, w = 200, 260
    img = np.full((h, w), 175, np.float32)
    for (cx, cy, rr) in [(60, 60, 34), (170, 70, 42), (110, 150, 30), (215, 160, 24)]:
        cv2.circle(img, (cx, cy), rr, 75, -1)
    return np.clip(img + r.normal(0, 16, img.shape), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- ④ 조명 vs 고정 T
def hist_lighting():
    obj = cv2.GaussianBlur(make_coins(), (5, 5), 0)
    dark = cv2.convertScaleAbs(obj, alpha=0.55, beta=0)
    bright = obj
    T = 128
    fig = plt.figure(figsize=(10.5, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])
    a0 = fig.add_subplot(gs[0, 0])
    for img, name, col in [(dark, "어두운 날 (×0.55)", C["purple"]), (bright, "보통 날", C["orange"])]:
        hst = np.bincount(img.ravel(), minlength=256)
        a0.fill_between(np.arange(256), hst, step="mid", alpha=0.45, color=col, label=name)
    a0.axvline(T, color=C["red"], lw=2, ls="--")
    a0.set_ylim(0, a0.get_ylim()[1] * 1.3)
    a0.text(T + 4, a0.get_ylim()[1] * 0.93, f"고정 T={T}", color=C["red"], fontweight="bold")
    a0.set_xlabel("밝기 (0~255)"); a0.set_ylabel("칸 수")
    a0.set_title("같은 장면, 조명만 다름 → 히스토그램이 통째로 이동", fontsize=11)
    a0.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    for k, (img, name) in enumerate([(bright, "보통 날"), (dark, "어두운 날")]):
        ax = fig.add_subplot(gs[0, k + 1])
        _, b = cv2.threshold(img, T, 255, cv2.THRESH_BINARY_INV)
        show(ax, b, f"{name} · BINARY_INV, T={T}\n물체(흰색) {int((b > 0).mean() * 100)}%")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#9a90b5")
    save(fig, DAY, "hist_lighting")


# ---------------------------------------------------------------- ④ Otsu
def otsu_hist():
    img = make_coins()
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    t_otsu, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hist = np.bincount(blur.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum(); v = np.arange(256)
    within = np.full(256, np.nan)
    for t in range(1, 255):
        w0, w1 = p[:t + 1].sum(), p[t + 1:].sum()
        if w0 < 1e-6 or w1 < 1e-6:
            continue
        m0 = (v[:t + 1] * p[:t + 1]).sum() / w0; m1 = (v[t + 1:] * p[t + 1:]).sum() / w1
        within[t] = (((v[:t + 1] - m0) ** 2) * p[:t + 1]).sum() + (((v[t + 1:] - m1) ** 2) * p[t + 1:]).sum()
    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.9, 1.6, 0.9])
    show(fig.add_subplot(gs[0, 0]), blur, "입력 (Blur 후)")
    a = fig.add_subplot(gs[0, 1])
    a.bar(v, hist, width=1.0, color=C["purple"], alpha=0.75, label="밝기 히스토그램")
    a.set_xlabel("밝기"); a.set_ylabel("칸 수")
    a2 = a.twinx()
    a2.plot(v, within, color=C["orange"], lw=2.2, label="무리 안 퍼짐 (작을수록 좋음)")
    a2.set_ylabel("무리 안 퍼짐", color=C["orange"]); a2.grid(False)
    a2.spines["right"].set_visible(True)
    a.axvline(t_otsu, color=C["red"], lw=2, ls="--")
    a.text(t_otsu - 4, hist.max() * 0.5, ha="right", s= f"Otsu T = {t_otsu:.0f}\n(퍼짐이 가장 작은 곳)",
           color=C["red"], fontweight="bold", fontsize=10, bbox=dict(fc="white", ec="none", alpha=0.85))
    a.text(35, hist.max() * 0.3, "물체\n무리", ha="center", color=C["ink"], fontsize=10)
    a.text(222, hist.max() * 0.3, "배경\n무리", ha="center", color=C["ink"], fontsize=10)
    a.set_title("두 봉우리 사이 골짜기에 T가 자동으로", fontsize=11)
    show(fig.add_subplot(gs[0, 2]), binary, f"BINARY_INV + OTSU\nretval = {t_otsu:.0f}")
    save(fig, DAY, "otsu_hist")


# ---------------------------------------------------------------- ④ Adaptive vs Global
def adaptive_vs_global():
    h, w = 180, 420
    page = np.full((h, w), 235, np.uint8)
    for i, line in enumerate(["OpenCV Threshold", "Adaptive vs Otsu", "SeSAC Day13 2026"]):
        cv2.putText(page, line, (14, 46 + i * 52), cv2.FONT_HERSHEY_SIMPLEX, 1.25, 40, 3, cv2.LINE_AA)
    light = np.linspace(0.18, 1.0, w)[None, :] * np.linspace(0.85, 1.0, h)[:, None]
    lit = np.clip(page * light + rng.normal(0, 3, page.shape), 0, 255).astype(np.uint8)
    _, fixed = cv2.threshold(lit, 128, 255, cv2.THRESH_BINARY)
    t, otsu = cv2.threshold(lit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(lit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    fig, axs = plt.subplots(2, 2, figsize=(10, 4.6))
    show(axs[0, 0], lit, "입력: 오른쪽에서만 빛이 들어온 문서")
    show(axs[0, 1], fixed, "Fixed T=128 → 왼쪽이 통째로 검정")
    show(axs[1, 0], otsu, f"Otsu (T={t:.0f}, 사진 전체에 하나) → 여전히 왼쪽 손실")
    show(axs[1, 1], adap, "Adaptive (blockSize=31, C=10) → 글자 전부 살아남")
    axs[0, 1].title.set_color(C["red"]); axs[1, 0].title.set_color(C["red"]); axs[1, 1].title.set_color(C["green"])
    save(fig, DAY, "adaptive_vs_global")


# ---------------------------------------------------------------- ④ 형태학: 칸 단위 그림
def morph_grid():
    m = np.zeros((21, 26), np.uint8)
    m[5:15, 3:11] = 255          # 물체 1
    m[9:11, 6:8] = 0             # 물체 안 작은 구멍
    m[6:14, 15:20] = 255         # 물체 2
    m[10, 15:20] = 0             # 가는 틈(끊김)
    m[18, 9] = 255; m[3, 23] = 255; m[18, 22] = 255   # 흰 먼지 3개 (테두리·물체와 떨어진 곳)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ops = [("원본 Binary", m),
           ("Erosion (침식)", cv2.erode(m, k)),
           ("Dilation (팽창)", cv2.dilate(m, k)),
           ("Opening (침식→팽창)", cv2.morphologyEx(m, cv2.MORPH_OPEN, k)),
           ("Closing (팽창→침식)", cv2.morphologyEx(m, cv2.MORPH_CLOSE, k))]
    notes = ["먼지 3 · 구멍 1 · 가는 틈 1", "먼지·가는 틈 사라지고\n물체도 한 겹 얇아짐", "구멍·틈 메워지고\n물체가 한 겹 두꺼워짐",
             "먼지만 사라지고\n물체 크기는 거의 그대로", "구멍·틈만 메워지고\n물체 크기는 거의 그대로"]
    fig, axs = plt.subplots(1, 5, figsize=(12.5, 3.4))
    for ax, (name, img), note in zip(axs, ops, notes):
        vis = np.zeros(m.shape + (3,), np.float32)
        vis[img > 0] = (1, 1, 1)
        vis[img == 0] = (0.16, 0.14, 0.25)
        removed = (m > 0) & (img == 0)
        added = (m == 0) & (img > 0)
        vis[removed] = np.array([224, 49, 49]) / 255
        vis[added] = np.array([15, 181, 166]) / 255
        ax.imshow(vis, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True); ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
        ax.grid(which="minor", color="#5b5270", lw=0.4); ax.grid(which="major", visible=False)
        ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel(note, fontsize=9, color=C["ink"])
    fig.text(0.5, -0.04, "3×3 RECT Kernel · 흰색 = 전경(물체) · 빨강 = 원본보다 깎인 칸 · 청록 = 원본보다 새로 생긴 칸",
             ha="center", fontsize=10, color=C["gray"])
    save(fig, DAY, "morph_grid")


# ---------------------------------------------------------------- ⑤ 영상 Frame 샘플링
def video_timeline():
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "sample.avi"
    shutil.copy(DATA / "sample.avi", src)        # 한글 경로 회피용 임시 복사(원본은 그대로)
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    shutil.rmtree(tmp, ignore_errors=True)
    n, step = len(frames), 5
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    event = range(12, 15)
    for i in range(n):
        sel = i % step == 0
        col = C["purple"] if sel else "#d9d2ea"
        ax.add_patch(Rectangle((i, 0), 0.85, 1, color=col))
        if i in event:
            ax.add_patch(Rectangle((i, 0), 0.85, 1, fill=False, ec=C["red"], lw=2))
    ax.set_xlim(-0.5, n + 0.5); ax.set_ylim(-0.9, 1.9)
    ax.set_yticks([]); ax.grid(False)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    ticks = np.arange(0, n + 1, fps)
    ax.set_xticks(ticks); ax.set_xticklabels([f"{t / fps:.0f}초\n(#{int(t)})" for t in ticks], fontsize=9)
    ax.text(0, 1.35, f"전체 {n}장 ÷ {fps:.0f}FPS = {n / fps:.0f}초 · 진보라 = step={step}로 처리하는 {len(range(0, n, step))}장",
            fontsize=10.5, color=C["ink"], fontweight="bold")
    ax.annotate("#12~14에만 있는 빠른 사건 → 한 장도 안 뽑힘", xy=(13.4, -0.05), xytext=(18, -0.75),
                color=C["red"], fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.6))
    for k, idx in enumerate([0, 5, 10, 13, 15, 20]):
        a = fig.add_subplot(gs[1, k])
        picked = idx % step == 0
        show(a, frames[idx], f"#{idx} · {idx / fps:.2f}초\n" +("처리함" if picked else "건너뜀"))
        a.title.set_color(C["purple"] if picked else C["red"])
    save(fig, DAY, "video_timeline")


# ---------------------------------------------------------------- ⑥ 증강: 라벨 뜻이 바뀌나
def aug_label():
    def canvas():
        return np.full((120, 120, 3), 245, np.uint8)
    cup = canvas()
    cv2.rectangle(cup, (32, 30), (78, 95), (60, 120, 230), -1)
    cv2.ellipse(cup, (80, 62), (16, 18), 0, -90, 90, (60, 120, 230), 6)
    letter = canvas()
    cv2.putText(letter, "b", (30, 98), cv2.FONT_HERSHEY_SIMPLEX, 3.6, (40, 35, 42), 9, cv2.LINE_AA)
    sign = canvas()
    cv2.circle(sign, (60, 60), 50, (230, 140, 40), -1)
    cv2.arrowedLine(sign, (88, 60), (30, 60), (255, 255, 255), 10, tipLength=0.45)
    items = [("컵 (사물 분류)", cup, "그대로 컵", True), ("글자 b (OCR)", letter, "b → d 로 바뀜", False),
             ("좌회전 표지", sign, "좌 → 우회전", False)]
    fig, axs = plt.subplots(2, 3, figsize=(8.4, 5.6))
    for c, (name, img, verdict, ok) in enumerate(items):
        show(axs[0, c], img, f"원본: {name}")
        show(axs[1, c], cv2.flip(img, 1), "flip(img, 1) 좌우 반전")
        axs[1, c].set_xlabel(("O  " if ok else "X  ") + verdict + (" · 써도 됨" if ok else " · 라벨 뜻이 바뀜"),
                             color=C["green"] if ok else C["red"], fontsize=10.5, fontweight="bold")
    save(fig, DAY, "aug_label")


# ---------------------------------------------------------------- ⑥ uint8 넘침
def brightness_overflow():
    x = np.arange(256, dtype=np.uint8)
    wrap = (x + np.uint8(30)).astype(np.uint8)              # uint8 그대로 더하기 → 넘치면 0부터
    clip = cv2.convertScaleAbs(x[None, :], alpha=1.0, beta=30)[0]
    fig = plt.figure(figsize=(10.5, 3.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.1, 1.3], hspace=0.9)
    a = fig.add_subplot(gs[:, 0])
    a.plot(x, x, color=C["gray"], lw=1.2, ls=":", label="원본 (그대로)")
    a.plot(x, wrap, color=C["red"], lw=2.2, label="img + 30 (uint8)")
    a.plot(x, clip, color=C["green"], lw=2.2, label="convertScaleAbs(beta=30)")
    a.annotate("250+30=280 → 24\n(256을 넘어 0부터 다시)", xy=(250, 24), xytext=(105, 100),
               color=C["red"], fontsize=9.5, fontweight="bold",
               arrowprops=dict(arrowstyle="->", color=C["red"]))
    a.annotate("255에서 멈춤", xy=(240, 255), xytext=(120, 225), color=C["green"], fontsize=9.5,
               fontweight="bold", arrowprops=dict(arrowstyle="->", color=C["green"]))
    a.set_xlabel("원래 밝기"); a.set_ylabel("바꾼 뒤 밝기"); a.set_xlim(0, 255); a.set_ylim(0, 270)
    a.legend(loc="upper left", fontsize=9)
    a.set_title("밝기 +30 하는 두 방법", fontsize=11)
    strip = lambda v: np.repeat(v[None, :], 18, 0)
    for r, (name, v) in enumerate([("원본 0 → 255", x), ("img + 30 · 오른쪽 끝이 갑자기 검게", wrap),
                                   ("convertScaleAbs · 밝은 곳은 흰색으로 유지", clip)]):
        ax = fig.add_subplot(gs[r, 1])
        show(ax, strip(v), None, aspect="auto")
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#cfc8e0")
        ax.set_title(name, fontsize=10, color=[C["ink"], C["red"], C["green"]][r], loc="left")
    save(fig, DAY, "brightness_overflow")


# ---------------------------------------------------------------- ⑦ 검증: shape는 같아도 정보가 사라질 수 있다
def verify_diff():
    b = np.zeros((60, 90), np.uint8)
    cv2.rectangle(b, (12, 14), (78, 48), 255, -1)
    defect_white = [(4, 5), (84, 54), (45, 6)]            # 진짜 작은 흰 결함(2×2) — 먼지와 구별 불가
    for (x, y) in defect_white:
        cv2.rectangle(b, (x, y), (x + 1, y + 1), 255, -1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    op = cv2.morphologyEx(b, cv2.MORPH_OPEN, k)
    lost = (b > 0) & (op == 0)
    vis = np.dstack([op, op, op]).astype(np.float32) / 255 * 0.85
    vis[lost] = np.array([224, 49, 49]) / 255
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
    show(axs[0], b, f"처리 전 Binary\nshape {b.shape} · 흰 칸 {int((b > 0).sum()):,}")
    show(axs[1], op, f"Opening 3×3 후\nshape {op.shape} · 흰 칸 {int((op > 0).sum()):,}")
    axs[2].imshow(vis, interpolation="nearest"); axs[2].set_xticks([]); axs[2].set_yticks([]); axs[2].grid(False)
    axs[2].set_title(f"차이 (빨강 = 사라진 칸 {int(lost.sum())}개)", fontsize=11, color=C["red"])
    for (x, y) in defect_white:
        axs[2].add_patch(plt.Circle((x + 0.5, y + 0.5), 3.5, fill=False, ec=C["red"], lw=1.8))
    fig.text(0.5, -0.04, "shape·dtype은 똑같이 통과 → 숫자 검증만으론 '작은 흰 결함 3개가 지워졌다'는 걸 못 봄",
             ha="center", fontsize=10, color=C["ink"])
    save(fig, DAY, "verify_diff")


if __name__ == "__main__":
    for fn in [pixel_grid, bgr_channels, hsv_lighting, resize_shrink, resize_mask, blur_kernels, filter_noise,
               canny_thresholds, hist_lighting, otsu_hist, adaptive_vs_global, morph_grid, video_timeline,
               aug_label, brightness_overflow, verify_diff]:
        fn()
