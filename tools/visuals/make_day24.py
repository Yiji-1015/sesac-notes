import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day24 그림 만들기 (Document Parsing · Chunking · Indexing).

숫자 · 글자 조각은 tools/io_day24.py 를 먼저 실행해 %TEMP%/d24_results.json 에 저장된 실제 실행 결과를 읽는다
(예제 attendance.md 의 clean_text · 40/10 글자 창 · v1→v2 공개 전환 실험).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day24.py
"""
import json
import tempfile
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

R = json.loads((Path(tempfile.gettempdir()) / "d24_results.json").read_text(encoding="utf-8"))
MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=9.2, ls="-", lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec or C["purple"], lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, text=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, color=color or MUTED, lw=1.6, ls=ls))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, text, ha="center", va="bottom", fontsize=8.2, color=color or MUTED)


def fig_flow():
    fig, ax = canvas(13.2, 5.0, 13.2, 5.0)
    ax.text(6.6, 4.75, "원본 파일 하나가 '검색 가능한 Chunk'가 되기까지 — 단계마다 남기는 기록이 다름", ha="center",
            fontsize=10.6, fontweight="bold", color=C["ink"])
    steps = [("① 수집", "파일 목록\nSHA-256 · 크기", "#f8f9fa", C["gray"]),
             ("② Parsing", "형식별 읽기\n제목 · 표 · 페이지", "#e7f5ff", C["blue"]),
             ("③ Cleaning", "줄바꿈 · 공백 · NFC\n의미는 그대로", "#e6fcf5", C["mint"]),
             ("④ Chunking", "구조 + 토큰 한도\nOverlap", "#f3f0ff", C["purple"]),
             ("⑤ 검증", "Span · Coverage\n중복 · 빈 조각", "#fff9db", C["yellow"]),
             ("⑥ Metadata", "출처 · 버전 · 위치\n결정적 chunk_id", "#fff4e6", C["orange"]),
             ("⑦ 저장 · 공개", "staging → ready\n→ current 포인터", "#e7f5ff", C["blue"])]
    w, gap, y = 1.62, 0.22, 2.55
    xs = [0.15 + i * (w + gap) for i in range(len(steps))]
    for x, (h, b, fc, ec) in zip(xs, steps):
        rbox(ax, x, y, w, 1.45, f"{h}\n{b}", fc=fc, ec=ec, fs=8.9)
    for a_, b_ in zip(xs[:-1], xs[1:]):
        arrow(ax, a_ + w, y + 0.72, b_, y + 0.72)
    outs = ["Raw File\n(원본 보존)", "ParsedUnit\nraw_text", "CleanUnit\nclean_text", "Span\n[start, end)", "통과 / 격리", "Chunk 행", "검색 대상"]
    for x, o in zip(xs, outs):
        ax.text(x + w / 2, y - 0.38, o, ha="center", va="center", fontsize=8.3, color=MUTED, linespacing=1.25)
    # 보류 · 실패 가지
    rbox(ax, xs[0], 0.25, w, 0.9, "unsupported\nnot_found · empty", fc="#f1f3f5", ec=C["gray"], fs=8.4, ls="--", lw=1.3)
    rbox(ax, xs[1], 0.25, w, 0.9, "needs_ocr\n(스캔 PDF)", fc="#f1f3f5", ec=C["gray"], fs=8.4, ls="--", lw=1.3)
    rbox(ax, xs[4], 0.25, w, 0.9, "span_mismatch\ntoo_many_tokens", fc="#fff5f5", ec=C["red"], fs=8.4, ls="--", lw=1.3)
    rbox(ax, xs[6], 0.25, w, 0.9, "개수 불일치\n→ ROLLBACK · 이전 판 유지", fc="#fff5f5", ec=C["red"], fs=8.0, ls="--", lw=1.3)
    for i in (0, 1, 4, 6):
        arrow(ax, xs[i] + w / 2, y - 0.62, xs[i] + w / 2, 1.17, color=C["red"] if i in (4, 6) else C["gray"], ls=":")
    ax.text(xs[2] + w + gap / 2, 1.55, "보류 · 실패도 상태로 기록 — 입력 수 = 완료 + 보류 + 실패 + 건너뜀", ha="center", fontsize=8.8, color=MUTED)
    save(fig, 24, "flow")


def fig_windows():
    text = R["clean_md"]
    wins = R["md_windows"]
    n = len(text)
    fig, ax = plt.subplots(figsize=(12.4, 4.4))
    ax.set_xlim(-2, n + 2); ax.set_ylim(-0.6, len(wins) + 1.6); ax.axis("off")
    ax.add_patch(Rectangle((0, len(wins) + 0.55), n, 0.55, fc="#f3f0ff", ec=C["purple"], lw=1))
    ax.text(n / 2, len(wins) + 1.32, f"attendance.md 의 clean_text — {n}자 · 창 크기 40 · Overlap 10 (보폭 30)", ha="center",
            fontsize=10.4, fontweight="bold", color=C["ink"])
    # 조건 문장 위치 표시
    s = text.index("10분을 초과하면")
    e = text.index("결석으로 기록합니다", s) + len("결석으로 기록합니다")
    holders = [k for k, (a0, b0) in enumerate(wins) if a0 <= s and e <= b0]
    ax.add_patch(Rectangle((s, len(wins) + 0.55), e - s, 0.55, fc=C["orange"], alpha=0.55, ec="none"))
    ax.text((s + e) / 2, len(wins) + 0.82, "10분을 초과하면 결석…", ha="center", va="center", fontsize=8, color=C["ink"])
    for i, (a, b) in enumerate(wins):
        y = len(wins) - 1 - i
        whole = a <= s and e <= b
        ax.add_patch(Rectangle((a, y), b - a, 0.62, fc="#e6fcf5" if whole else "#f8f9fa", ec=C["mint"] if whole else C["gray"], lw=1.4))
        snippet = text[a:b].replace("\n", " / ")
        ax.text(a + 0.6, y + 0.31, f"[{a},{b})  {snippet[:17]}…" if len(snippet) > 17 else f"[{a},{b})  {snippet}",
                va="center", fontsize=8.1, color=C["ink"])
        if i:
            pa, pb = wins[i - 1]
            ax.add_patch(Rectangle((a, y + 0.62), pb - a, 0.1, fc=C["pink"], ec="none"))
    ax.text(0, -0.45, f"분홍 = 앞 조각과 겹친 10자 · 초록 = '10분을 초과하면 결석으로 기록합니다' 조건과 결과를 함께 담은 조각 — {len(wins)}개 중 {len(holders)}개 (마침표 · 다음 문장은 다음 조각으로 넘어감)",
            fontsize=8.8, color=MUTED)
    save(fig, 24, "windows")


def fig_publish():
    P = R["publish"]
    fig, ax = canvas(12.6, 3.6, 12.6, 3.6)
    ax.text(6.3, 3.35, "공개 포인터 전환 — 실제 실행(노트 전용 DB) 순서 그대로", ha="center", fontsize=10.4, fontweight="bold", color=C["ink"])
    events = [("v1 staging", "current = None\n(검색 대상 없음)", "#f8f9fa", C["gray"]),
              ("v1 publish", f"current = {P['v1']!r}", "#e6fcf5", C["mint"]),
              ("v2 staging\n(expected 3, 실제 2)", "current = 'v1' 유지", "#f8f9fa", C["gray"]),
              ("v2 publish 실패", "incomplete_version\n→ ROLLBACK\ncurrent = " + repr(P["after_fail"]), "#fff5f5", C["red"]),
              ("개수 맞춘 뒤\nv2 publish", f"current = {P['v2']!r}\nv1 Chunk 는 남지만 검색 제외", "#e6fcf5", C["mint"])]
    w, gap = 2.3, 0.22
    for i, (h, b, fc, ec) in enumerate(events):
        x = 0.15 + i * (w + gap)
        rbox(ax, x, 1.55, w, 1.3, h, fc=fc, ec=ec, fs=9.2)
        ax.text(x + w / 2, 1.0, b, ha="center", va="center", fontsize=8.6, color=C["ink"] if "current" in b else MUTED, linespacing=1.3)
        if i:
            arrow(ax, x - gap, 2.2, x, 2.2)
    ax.text(6.3, 0.15, "검색 SQL 은 c.source_version = d.current_version 인 Chunk 만 봄 → 어느 순간에도 '반쯤 바뀐 판'이 검색되지 않음",
            ha="center", fontsize=8.9, color=MUTED)
    save(fig, 24, "publish")


if __name__ == "__main__":
    fig_flow(); fig_windows(); fig_publish()


def fig_ocr():
    import cv2
    tmp = Path(tempfile.gettempdir())
    clean = cv2.imread(str(tmp / "d24_scan_clean.png"), cv2.IMREAD_GRAYSCALE)
    bad = cv2.imread(str(tmp / "d24_scan_bad.png"), cv2.IMREAD_GRAYSCALE)
    boxes = json.loads((tmp / "d24_bad_boxes.json").read_text(encoding="utf-8"))
    D = R["cases"]["D"]
    fig = plt.figure(figsize=(12.6, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.35, wspace=0.12)
    a1 = fig.add_subplot(gs[0, 0]); a1.imshow(clean, cmap="gray"); a1.axis("off")
    a1.set_title("① 스캔 이미지 (PDF 안에 든 그림 · 텍스트층 없음)", fontsize=10.5)
    a2 = fig.add_subplot(gs[0, 1]); a2.imshow(bad, cmap="gray"); a2.axis("off")
    a2.set_title("② 흐린 사본 (1/3 축소 · 1.8° 기울임 · 잡음) + 단어별 conf", fontsize=10.5)
    for t, c, (x, y, w, h) in boxes:
        col = C["red"] if c < 60 else ("#ff922b" if t == "『" else None)
        if col:
            a2.add_patch(Rectangle((x, y), w, h, fill=False, ec=col, lw=2))
            a2.text(x, y - 3, f"{t} {round(c)}", color=col, fontsize=8.5, fontweight="bold")
    a2.text(2, bad.shape[0] - 4, "빨강 = conf < 60 (검토 후보) · 주황 = 틀렸는데 conf 81", fontsize=8.4, color=C["ink"],
            bbox=dict(fc="white", ec="none", alpha=0.8))
    a3 = fig.add_subplot(gs[1, :])
    labels = [f"{r['img']}\n{r['lang']}{' · 띄어쓰기 보존' if r['pis'] else ''}" for r in D["runs"]]
    vals = [r["cer"] for r in D["runs"]]
    cols = [C["gray"], C["purple"], C["mint"], C["purple"], C["mint"], C["orange"]]
    bars = a3.bar(range(len(vals)), vals, color=cols)
    for b, v in zip(bars, vals):
        a3.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    a3.set_xticks(range(len(vals))); a3.set_xticklabels(labels, fontsize=8.3)
    a3.set_ylabel("CER (낮을수록 정확)")
    a3.set_title("③ 설정별 글자 오류율 — 정답 문장과 비교한 실제 값", fontsize=10.5)
    save(fig, 24, "ocr")


if __name__ == "__main__":
    fig_ocr()
