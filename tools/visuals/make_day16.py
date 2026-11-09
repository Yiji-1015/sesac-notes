import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day16 그림 만들기 (문서 전처리 · OCR 결과 · CER/평가 단위 · 구조화 · VLM 입력 크기 · 정밀도).

청구서는 PIL로 직접 그린 '가상 청구서'(강의 예시: 공급자 AI Lab, 발행일 2026-11-09, 합계 1,234,500원)이다.
이 PC에는 Tesseract 프로그램과 transformers가 없어서, OCR 박스·confidence·VLM 답은 실제 출력이 아니라
'그린 위치'와 '가상 값'으로 만든 설명용 그림이다(그림 안에도 표시).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day16.py
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon

np.random.seed(0)
KO = "C:/Windows/Fonts/malgun.ttf"
KO_B = "C:/Windows/Fonts/malgunbd.ttf"
MUTED = "#5b5270"


def font(size, bold=False):
    try:
        return ImageFont.truetype(KO_B if bold else KO, size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------------------ 가상 청구서 (1000 x 1200, 강의의 설명용 페이지 크기)
# (text, x, y, size, bold, field)  field = 업무 필드 이름 / "label" = 라벨 / None = 기타
ITEMS = [
    ("INVOICE", 80, 70, 64, True, None),
    ("가상 청구서 · No. INV-001", 84, 150, 28, False, None),
    ("공급자", 118, 262, 30, False, "label"),
    ("AI Lab", 330, 260, 32, True, "supplier"),
    ("발행일", 118, 205 - 0, 30, False, "label"),
    ("2026-11-09", 330, 203, 32, True, "issued_at"),
    ("품목", 118, 420, 28, True, None),
    ("수량", 560, 420, 28, True, None),
    ("금액", 760, 420, 28, True, None),
    ("GPU 서버 임대", 118, 490, 28, False, None),
    ("1", 580, 490, 28, False, None),
    ("900,000", 760, 490, 28, False, None),
    ("데이터 라벨링", 118, 550, 28, False, None),
    ("3", 580, 550, 28, False, None),
    ("222,273", 760, 550, 28, False, None),
    ("공급가액", 118, 680, 28, False, "label"),
    ("1,122,273원", 600, 680, 28, False, None),
    ("세액", 118, 735, 28, False, "label"),
    ("112,227원", 625, 735, 28, False, None),
    ("합계", 118, 822, 36, True, "label"),
    ("1,234,500원", 530, 818, 44, True, "total_amount"),
]


def make_invoice():
    W, H = 1000, 1200
    img = Image.new("RGB", (W, H), (252, 251, 246))
    d = ImageDraw.Draw(img)
    d.rectangle([60, 40, 940, 1150], outline=(60, 60, 70), width=3)
    for y in (400, 470, 610):
        d.line([100, y, 900, y], fill=(90, 90, 100), width=2)
    d.line([100, 800, 900, 800], fill=(40, 40, 50), width=3)
    d.text((118, 960), "입금 계좌: (가상) 000-0000-0000", font=font(24), fill=(110, 110, 110))
    words = []
    for text, x, y, size, bold, field in ITEMS:
        f = font(size, bold)
        if field == "total_amount":
            # 강의 예시 bbox [530,820,790,865] 안에 글자를 맞춰 그린다
            tb = d.textbbox((0, 0), text, font=f)
            x = 530 + (260 - (tb[2] - tb[0])) / 2 - tb[0]
            y = 820 + (45 - (tb[3] - tb[1])) / 2 - tb[1]
            d.text((x, y), text, font=f, fill=(20, 20, 25))
            words.append((text, field, (530, 820, 790, 865)))
            continue
        d.text((x, y), text, font=f, fill=(20, 20, 25))
        bb = d.textbbox((x, y), text, font=f)
        words.append((text, field, bb))
    return np.array(img)[:, :, ::-1].copy(), words      # BGR (OpenCV 순서)


INV, WORDS = make_invoice()


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------ "사진으로 찍은" 문서 만들기 (강의 p28 좌표)
SRC_CORNERS = np.float32([[92, 40], [820, 75], [860, 1080], [55, 1050]])   # 사진 속 문서 네 모서리
DST_CORNERS = np.float32([[0, 0], [699, 0], [699, 999], [0, 999]])           # 펼친 뒤 (700 x 1000)


def make_photo():
    h, w = INV.shape[:2]
    full = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    M = cv2.getPerspectiveTransform(full, SRC_CORNERS)
    pw, ph = 920, 1130
    photo = cv2.warpPerspective(INV, M, (pw, ph), borderValue=(88, 98, 112)).astype(np.float32)
    yy, xx = np.mgrid[0:ph, 0:pw]
    shade = 0.38 + 0.62 * np.clip((xx * 1.0 + 0.5 * yy) / (pw * 1.15), 0, 1)     # 왼쪽 위가 어두운 그림자
    photo = photo * shade[..., None] + np.random.normal(0, 6, photo.shape)
    return np.clip(photo, 0, 255).astype(np.uint8)


PHOTO = make_photo()
GRAY = cv2.cvtColor(PHOTO, cv2.COLOR_BGR2GRAY)
OTSU_T, OTSU = cv2.threshold(GRAY, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
ADAPT = cv2.adaptiveThreshold(GRAY, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)


def fig_preprocess_steps():
    M = cv2.getPerspectiveTransform(SRC_CORNERS, DST_CORNERS)
    warped = cv2.warpPerspective(ADAPT, M, (700, 1000), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    fig, axs = plt.subplots(1, 5, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.78]})
    panels = [
        (to_rgb(PHOTO), None, "① 촬영본 (BGR→RGB로 표시)", f"shape {PHOTO.shape}"),
        (GRAY, "gray", "② Gray", f"shape {GRAY.shape} · 채널 축 사라짐"),
        (OTSU, "gray", f"③ Otsu 전역 기준 T={OTSU_T:.0f}", "그림자 쪽이 통째로 검게"),
        (ADAPT, "gray", "④ Adaptive (31, 11)", "주변마다 기준 따로"),
        (warped, "gray", "⑤ Perspective 보정", "출력 (700, 1000) = (W, H)"),
    ]
    for ax, (im, cmap, title, sub) in zip(axs, panels):
        ax.imshow(im, cmap=cmap, vmin=0, vmax=255) if cmap else ax.imshow(im)
        ax.set_title(title, fontsize=12); ax.axis("off")
        ax.text(0.5, -0.05, sub, transform=ax.transAxes, ha="center", va="top", fontsize=10, color=MUTED)
    # 네 모서리 순서 표시
    poly = np.vstack([SRC_CORNERS, SRC_CORNERS[:1]])
    axs[0].plot(poly[:, 0], poly[:, 1], color=C["yellow"], lw=2, ls="--")
    for i, (x, y) in enumerate(SRC_CORNERS):
        axs[0].scatter([x], [y], s=70, color=C["yellow"], edgecolor="black", zorder=5)
        axs[0].text(x + (30 if i in (0, 3) else -30), y + (40 if i < 2 else -30), str(i + 1), color="black",
                    fontsize=12, fontweight="bold", ha="center",
                    bbox=dict(boxstyle="circle,pad=0.2", fc=C["yellow"], ec="none"))
    # 그림자 영역 표시
    axs[2].add_patch(Rectangle((70, 50), 300, 380, fill=False, ec=C["red"], lw=2.2, ls="--"))
    axs[2].text(300, 1010, "빨간 점선: 글자가\n배경과 함께 먹힘", color=C["red"], fontsize=10.5, ha="center", fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.85))
    axs[3].add_patch(Rectangle((70, 50), 300, 380, fill=False, ec=C["green"], lw=2.2, ls="--"))
    axs[3].text(300, 1010, "초록 점선: 같은 자리\n글자가 살아남음", color=C["green"], fontsize=10.5, ha="center", fontweight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.85))
    fig.text(0.5, -0.04, "가상 청구서를 기울이고 그림자·잡음을 넣어 만든 '촬영본'. 노란 1→2→3→4 = src 모서리 순서(시계 방향), "
                         "dst도 같은 순서로 [0,0]→[699,0]→[699,999]→[0,999]", ha="center", fontsize=10, color=MUTED)
    save(fig, 16, "preprocess_steps")


def fig_threshold_profile():
    blur = cv2.GaussianBlur(GRAY.astype(np.float32), (31, 31), 0)   # adaptive GAUSSIAN_C 와 같은 가중 평균(근사)
    # 그림자 쪽(x<450)에서 글자 획을 가장 많이 지나는 가로줄 고르기
    dark = (GRAY.astype(np.float32) < blur - 30)[:, 120:450].sum(1)
    row = 200 + int(np.argmax(dark[200:600]))
    prof = GRAY[row].astype(float)
    local_t = blur[row] - 11
    x = np.arange(len(prof))
    stroke_x = 120 + int(np.argmin(prof[120:450] - local_t[120:450]))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [0.62, 1.6]})
    ax = axs[0]
    ax.imshow(GRAY, cmap="gray", vmin=0, vmax=255); ax.axis("off")
    ax.axhline(row, color=C["pink"], lw=2.2)
    ax.text(10, row - 22, f"y = {row} 줄", color="white", fontsize=11, fontweight="bold",
            bbox=dict(fc=C["pink"], ec="none"))
    ax.set_title("Gray 촬영본에서 한 줄만 뽑아 보기")
    ax = axs[1]
    ax.plot(x, prof, color="#9aa0a6", lw=0.9, label="픽셀 밝기 (이 줄)")
    ax.axhline(OTSU_T, color=C["red"], lw=2.2, ls="--", label=f"Otsu 전역 기준 T={OTSU_T:.0f} (어디서나 같음)")
    ax.plot(x, local_t, color=C["green"], lw=2.4, label="Adaptive 기준 = 주변 31×31 가중평균 - 11")
    ax.set_xlim(0, len(prof)); ax.set_ylim(0, 255)
    ax.set_xlabel("x (왼쪽 = 그림자, 오른쪽 = 밝음)"); ax.set_ylabel("밝기 0~255")
    # 왼쪽 그림자 구간: 종이(배경)인데 T 아래
    edge = 60 + int(np.argmax(prof[60:] > 100))                          # 종이 왼쪽 끝 (어두운 배경 → 종이)
    bg_est = blur[row]
    cross = edge + int(np.argmax(bg_est[edge:] > OTSU_T))                 # 종이 밝기가 Otsu 선을 넘는 곳
    ax.axvspan(edge, cross, color=C["red"], alpha=0.08)
    ax.annotate("빨간 구간: 종이(배경)인데 Otsu 선보다 어두움\n→ 글자와 함께 통째로 검게 칠해짐",
                xy=((edge + cross) / 2, OTSU_T - 18), xytext=(edge, 12), fontsize=10.5, color=C["red"],
                fontweight="bold", bbox=dict(fc="white", ec="none", alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=C["red"]))
    ax.annotate("초록 선은 배경을 따라 오르내려서\n'주변보다 어두운가?'만 물음 → 글자만 검게", xy=(640, local_t[640]),
                xytext=(600, 18), fontsize=10.5, color=C["green"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["green"]))
    ax.annotate("뾰족하게 꺼진 곳 = 글자 획", xy=(stroke_x, prof[stroke_x]),
                xytext=(500, 75), fontsize=10.5, color=MUTED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.legend(loc="upper left", fontsize=9.6)
    ax.set_title("같은 줄에서 기준선 두 개 비교")
    save(fig, 16, "threshold_profile")


# ------------------------------------------------------------ OCR 결과 한 건 (설명용 · 가상 confidence)
CONF = {"INVOICE": 96.2, "가상": 91.0, "공급자": 94.5, "AI Lab": 93.1, "발행일": 95.0, "2026-11-09": 92.4,
        "합계": 97.3, "1,234,500원": 88.6, "1": 41.0, "3": 47.5, "세액": 90.2, "112,227원": 64.0}


def conf_color(c):
    return C["green"] if c >= 80 else (C["orange"] if c >= 50 else C["red"])


def fig_ocr_boxes():
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.2), gridspec_kw={"width_ratios": [1, 1.08]})
    ax = axs[0]
    ax.imshow(to_rgb(INV)); ax.axis("off")
    ax.set_title("단어(level 5)마다 Box + Text + Conf")
    rows = []
    for text, field, (x1, y1, x2, y2) in WORDS:
        c = CONF.get(text, 85.0 + (len(text) % 7))
        rows.append((text, c, x1, y1, x2 - x1, y2 - y1))
        ax.add_patch(Rectangle((x1 - 4, y1 - 4), x2 - x1 + 8, y2 - y1 + 8, fill=False, ec=conf_color(c), lw=2))
    # 합계 금액 bbox 강조
    tb = [b for t, f, b in WORDS if t == "1,234,500원"][0]
    ax.annotate(f"bbox xyxy = [{tb[0]},{tb[1]},{tb[2]},{tb[3]}]\nconf 88.6", xy=(tb[0], tb[3] + 4),
                xytext=(300, 1050), fontsize=11, color=C["pink"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["pink"], lw=1.8))
    b = [bb for tt, f, bb in WORDS if tt == "1"][0]
    ax.annotate("짧은 숫자 '1' conf 41\n(맞게 읽어도 낮을 수 있음)", xy=(b[2] + 6, (b[1] + b[3]) / 2),
                xytext=(600, 330), fontsize=9.8, color=C["red"], arrowprops=dict(arrowstyle="->", color=C["red"]))
    at = axs[1]; at.axis("off"); at.set_xlim(0, 1); at.set_ylim(0, 1)
    at.set_title("image_to_data(..., output_type=Output.DATAFRAME) 의 단어 행")
    cols = [("text", 0.02, "left"), ("conf", 0.40, "right"), ("left", 0.53, "right"), ("top", 0.66, "right"),
            ("width", 0.81, "right"), ("height", 0.97, "right")]
    for name, x, ha in cols:
        at.text(x, 0.93, name, ha=ha, family="Consolas", fontsize=11.5, fontweight="bold", color=C["ink"])
    at.plot([0.0, 1.0], [0.905, 0.905], color="#cfc8e0", lw=1)
    show = [r for r in rows if r[0] in ("INVOICE", "발행일", "2026-11-09", "공급자", "AI Lab", "1", "3",
                                        "112,227원", "합계", "1,234,500원")]
    for i, (t, c, l, tp, w, h) in enumerate(show):
        yy = 0.86 - i * 0.058
        vals = [t, f"{c:.1f}", l, tp, w, h]
        for (name, x, ha), v in zip(cols, vals):
            at.text(x, yy, str(v), ha=ha, fontsize=10.8, color=conf_color(c), fontweight="bold" if c < 50 else "normal")
    at.text(0.02, 0.86 - len(show) * 0.058, "...", fontsize=11, color=MUTED)
    at.add_patch(Rectangle((0.0, 0.02), 1.0, 0.2, fc=C["soft"], ec="#cfc8e0", transform=at.transAxes))
    at.text(0.03, 0.17, "review = words[words.conf.lt(50)]   → 검토 목록 (삭제가 아님)",
            fontsize=10.4, color=C["red"])
    at.text(0.03, 0.115, "left, top, width, height  → xyxy = [left, top, left+width, top+height]", family="Consolas",
            fontsize=10.2, color=C["ink"])
    at.text(0.03, 0.06, "초록 conf ≥ 80 · 주황 50~80 · 빨강 < 50", fontsize=10.2, color=MUTED)
    fig.text(0.5, 0.0, "설명용 그림: 박스는 글자를 그린 위치, conf는 가상 값입니다 (이 PC에서 Tesseract를 실행한 결과가 아님).",
             ha="center", fontsize=10, color=MUTED)
    save(fig, 16, "ocr_boxes")


# ------------------------------------------------------------ CER 정렬
def align(ref, hyp):
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = range(n + 1); D[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("D", ref[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", hyp[j - 1])); j -= 1
    return ops[::-1], D[n, m]


def fig_cer_alignment():
    cases = [("1234500", "12345O0", "금액 (강의 예시)"), ("발행일 2026-11-09", "발행일: 2O26-1l-9", "날짜 (예시)")]
    col = {"=": "#eef0f3", "S": C["orange"], "D": C["red"], "I": C["blue"]}
    fig, axs = plt.subplots(2, 1, figsize=(12, 5.2))
    for ax, (ref, hyp, name) in zip(axs, cases):
        ops, dist = align(ref, hyp)
        ax.set_xlim(-2.8, max(len(ops), 15) + 0.2); ax.set_ylim(-0.3, 2.75); ax.axis("off")
        ax.text(-0.3, 1.8, "정답", ha="right", va="center", fontsize=11.5, fontweight="bold")
        ax.text(-0.3, 0.9, "OCR", ha="right", va="center", fontsize=11.5, fontweight="bold")
        for k, (op, r, h) in enumerate(ops):
            fc = col[op]; tc = C["ink"] if op == "=" else "white"
            for y, ch in ((1.8, r), (0.9, h)):
                ax.add_patch(Rectangle((k + 0.05, y - 0.38), 0.9, 0.76, fc=fc, ec="white", lw=1))
                show = "_" if ch == " " else (ch if ch else "·")
                ax.text(k + 0.5, y, show, ha="center", va="center", fontsize=14, color=tc, fontweight="bold")
            if op != "=":
                ax.text(k + 0.5, 0.22, op, ha="center", va="center", fontsize=11, fontweight="bold", color=fc)
        S = sum(o[0] == "S" for o in ops); Dn = sum(o[0] == "D" for o in ops); I = sum(o[0] == "I" for o in ops)
        ax.text(-2.7, 2.55, name, ha="left", va="center", fontsize=11.5, fontweight="bold", color=C["purple"])
        ax.text(max(len(ops), 15) + 0.1, 2.55,
                f"S {S} + D {Dn} + I {I} = {dist}  ÷  정답 {len(ref)}글자  →  CER {dist / len(ref):.3f} ({dist / len(ref) * 100:.1f}%)",
                ha="right", va="center", fontsize=11.5, fontweight="bold", color=C["ink"])
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=col["S"], label="S 바꿔 읽음 (0→O, 1→l)"),
                        Rectangle((0, 0), 1, 1, fc=col["D"], label="D 빠뜨림"),
                        Rectangle((0, 0), 1, 1, fc=col["I"], label="I 끼워 넣음"),
                        Rectangle((0, 0), 1, 1, fc=col["="], ec="#cfc8e0", label="같음  ( _ = 띄어쓰기, · = 글자 없음)")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=10.5)
    save(fig, 16, "cer_alignment")


# ------------------------------------------------------------ 평가 단위 4가지 (강의 p9 가상 수치)
def fig_eval_units():
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.0))
    specs = [
        ("금액 문자열 CER", 1 / 7, "1 / 7 글자 틀림", "낮을수록 좋음", C["orange"]),
        ("필드 정확도 (3개)", 2 / 3, "2 / 3 필드 맞음", "높을수록 좋음", C["purple"]),
        ("JSON 문법 (응답 1건)", 1.0, "1 / 1 파싱 성공", "높을수록 좋음", C["mint"]),
    ]
    for ax, (title, v, sub, good, col) in zip(axs[:3], specs):
        ax.bar([0], [1], color="#eef0f3", width=0.55)
        ax.bar([0], [v], color=col, width=0.55)
        ax.text(0, v + 0.04, f"{v * 100:.1f}%", ha="center", fontsize=17, fontweight="bold", color=col)
        ax.set_ylim(0, 1.22); ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "50%", "100%"]); ax.set_title(title, fontsize=12.5)
        ax.text(0, -0.1, sub, ha="center", va="top", fontsize=10.5, color=C["ink"], transform=ax.transData)
        ax.text(0, -0.2, good, ha="center", va="top", fontsize=10, color=MUTED, transform=ax.transData)
        ax.grid(axis="x", visible=False)
    ax = axs[3]; ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("원문 근거 대조", fontsize=12.5)
    ax.add_patch(Rectangle((0.05, 0.52), 0.9, 0.3, fc="#fff4e6", ec=C["orange"], lw=2))
    ax.text(0.5, 0.72, "answer: 1,234,500원", ha="center", fontsize=11.5)
    ax.text(0.5, 0.6, "원문: 1,234,5O0원", ha="center", fontsize=11.5, color=C["red"])
    ax.text(0.5, 0.36, "금액 불일치 → 검토", ha="center", fontsize=14, fontweight="bold", color=C["red"])
    ax.text(0.5, 0.18, "점수가 아니라 '상태 + 이유'", ha="center", fontsize=10.5, color=MUTED)
    fig.suptitle("같은 문서 하나를 네 가지 잣대로 따로 채점 (강의의 가상 수치)", fontsize=13.5, fontweight="bold", y=1.03)
    save(fig, 16, "eval_units")


# ------------------------------------------------------------ Key-value 구조화
def fig_kv_extract():
    crop = to_rgb(INV[188:900, 60:960])
    oy, ox = 188, 60
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axs[0]; ax.imshow(crop); ax.axis("off")
    ax.set_title("라벨 Box 오른쪽에 있는 값 Box를 짝짓기")
    pairs = [("발행일", "2026-11-09", C["mint"]), ("공급자", "AI Lab", C["purple"]), ("합계", "1,234,500원", C["pink"])]
    bb = {t: b for t, f, b in WORDS}
    for lab, val, col in pairs:
        l, v = bb[lab], bb[val]
        for (x1, y1, x2, y2), ls in ((l, "--"), (v, "-")):
            ax.add_patch(Rectangle((x1 - ox - 5, y1 - oy - 5), x2 - x1 + 10, y2 - y1 + 10, fill=False, ec=col, lw=2.4, ls=ls))
        ax.add_patch(FancyArrowPatch((l[2] - ox + 8, (l[1] + l[3]) / 2 - oy), (v[0] - ox - 8, (v[1] + v[3]) / 2 - oy),
                                     arrowstyle="-|>", mutation_scale=16, color=col, lw=2))
    ax.text(20, 170, "점선 = 라벨(key) · 실선 = 값(value)", fontsize=10.5, color=MUTED,
            bbox=dict(fc="white", ec="none", alpha=0.85))
    at = axs[1]; at.axis("off"); at.set_xlim(0, 1); at.set_ylim(0, 1)
    at.set_title("fields.total_amount 한 칸에 담는 것")
    rows = [("raw", '"1,234,500원"', "OCR 원문 그대로", C["ink"]),
            ("value", "1234500", "프로그램이 쓸 정수 (쉼표·원 제거)", C["pink"]),
            ("currency", '"KRW"', "통화는 따로", C["ink"]),
            ("confidence", "88.6", "엔진 점수", C["ink"]),
            ("bbox", "[530, 820, 790, 865]", "원문 위치 (처리 이미지 xyxy)", C["ink"])]
    at.add_patch(Rectangle((0.0, 0.42), 1.0, 0.5, fc=C["soft"], ec="#cfc8e0"))
    for i, (k, v, note, col) in enumerate(rows):
        y = 0.86 - i * 0.092
        at.text(0.03, y, k, family="Consolas", fontsize=12, fontweight="bold", color=C["purple"])
        at.text(0.27, y, v, family="Malgun Gothic", fontsize=12, color=col, fontweight="bold")
        at.text(0.62, y, note, fontsize=10, color=MUTED)
    checks = [("필수 필드 있음?", "OK"), ("value 가 0 이상 정수?", "OK"),
              ("공급가액 + 세액 = 합계?", "1,122,273 + 112,227 = 1,234,500  OK"), ("→ 상태", "passed")]
    for i, (q, a) in enumerate(checks):
        y = 0.33 - i * 0.08
        at.text(0.03, y, q, fontsize=11, color=C["ink"], fontweight="bold" if i == 3 else "normal")
        at.text(0.45, y, a, fontsize=11, color=C["green"], fontweight="bold")
    save(fig, 16, "kv_extract")


# ------------------------------------------------------------ VLM 입력 크기: 전체 축소 vs Crop
def fig_crop_vs_resize():
    H_small = 224                                                       # 전체 페이지를 높이 224px로 축소
    sc = H_small / 1200
    small = cv2.resize(INV, (round(1000 * sc), H_small), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = 100, 660, 900, 780                                  # 공급가액·세액 두 줄 영역
    crop = INV[y1:y2, x1:x2]
    cw = 336
    crop_s = cv2.resize(crop, (cw, round(crop.shape[0] * cw / crop.shape[1])), interpolation=cv2.INTER_AREA)
    region_small = small[round(y1 * sc):round(y2 * sc), round(x1 * sc):round(x2 * sc)]
    # 같은 크기로 보여 주기 위해 둘 다 픽셀을 그대로 키움(NEAREST) → 실제 정보량 차이가 보임
    up_small = cv2.resize(region_small, (cw, round(region_small.shape[0] * cw / region_small.shape[1])),
                          interpolation=cv2.INTER_NEAREST)
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.55, 1.05, 1.2], height_ratios=[1, 1])
    a0 = fig.add_subplot(gs[:, 0]); a0.imshow(to_rgb(small)); a0.axis("off")
    a0.set_title(f"페이지 전체를\n{small.shape[1]}×{small.shape[0]}px로 축소", fontsize=12)
    a0.add_patch(Rectangle((x1 * sc, y1 * sc), (x2 - x1) * sc, (y2 - y1) * sc, fill=False, ec=C["red"], lw=2))
    a0.text(small.shape[1] / 2, H_small + 8, "28px 글자가 약 %dpx로" % round(28 * sc), ha="center", va="top",
            fontsize=10.5, color=C["red"], fontweight="bold")
    a1 = fig.add_subplot(gs[0, 1]); a1.imshow(to_rgb(up_small)); a1.axis("off")
    a1.set_title(f"A. 축소본의 빨간 영역 ({region_small.shape[1]}×{region_small.shape[0]}px를 확대)", fontsize=11.5)
    a2 = fig.add_subplot(gs[1, 1]); a2.imshow(to_rgb(crop_s)); a2.axis("off")
    a2.set_title(f"B. 원본에서 같은 영역만 Crop ({crop_s.shape[1]}×{crop_s.shape[0]}px)", fontsize=11.5)
    p_small = (small.shape[0] // 14) * (small.shape[1] // 14)
    p_crop = (crop_s.shape[0] // 14) * (crop_s.shape[1] // 14)
    a2.text(cw / 2, crop_s.shape[0] + 10, f"14px 조각 수: 축소 전체 {p_small}개 vs Crop {p_crop}개 - 더 적은 조각으로 더 또렷",
            ha="center", va="top", fontsize=10.2, color=C["green"], fontweight="bold")
    a3 = fig.add_subplot(gs[:, 2])
    sides = np.array([224, 336, 448, 672, 896, 1344])
    patches = (sides // 14) ** 2
    a3.plot(sides, patches, "o-", color=C["purple"], lw=2.2)
    for sx, p in zip(sides, patches):
        a3.text(sx, p + 350, f"{p:,}", ha="center", fontsize=9.8, color=C["ink"])
    a3.set_xlabel("정사각형 입력 한 변 (px)"); a3.set_ylabel("이미지 조각(패치) 수")
    a3.set_title("예시 계산: 14px 패치라면 해상도 2배 → 조각 4배")
    a3.annotate("한 변 ×2 → 조각 ×4\n(계산·메모리도 함께 증가)", xy=(1344, patches[-1]), xytext=(380, 7200),
                fontsize=10.5, color=C["pink"], fontweight="bold", arrowprops=dict(arrowstyle="->", color=C["pink"]))
    a3.set_ylim(0, patches[-1] * 1.15)
    fig.text(0.5, -0.02, "가상 청구서로 만든 그림. 패치 크기·조각 수는 모델마다 다르며 오른쪽은 원리를 보여 주는 예시 계산입니다.",
             ha="center", fontsize=10, color=MUTED)
    save(fig, 16, "crop_vs_resize")


# ------------------------------------------------------------ 정밀도별 크기 (예시)
def fig_precision_size():
    params = 2.5e6
    rows = [("FP32", 32, C["gray"], "기본 (학습할 때)"), ("FP16", 16, C["blue"], "절반 크기 · 대부분 정확도 거의 유지"),
            ("INT8", 8, C["purple"], "1/4 크기 · 정확도 변화 확인 필수")]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for i, (n, bits, c, note) in enumerate(rows):
        mb = params * bits / 8 / 1e6
        y = len(rows) - 1 - i
        ax.barh(y, mb, color=c, height=0.55)
        ax.text(mb + 0.15, y, f"{mb:.1f} MB  ·  숫자 하나 {bits}비트  ·  {note}", va="center", fontsize=10.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=12, fontweight="bold")
    ax.set_xlim(0, 16); ax.set_xlabel("가중치 파일 크기 (MB)")
    ax.set_title("같은 모델, 숫자를 몇 비트로 저장하나에 따라 크기가 달라짐 (예시 계산: 가중치 250만 개)")
    ax.grid(axis="y", visible=False)
    save(fig, 16, "precision_size")


if __name__ == "__main__":
    fig_preprocess_steps()
    fig_threshold_profile()
    fig_ocr_boxes()
    fig_cer_alignment()
    fig_eval_units()
    fig_kv_extract()
    fig_crop_vs_resize()
    fig_precision_size()
