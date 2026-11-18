import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day23 그림 만들기 (Embedding · pgvector).

숫자가 들어간 그림(2차원 거리 비교 · ef_search 곡선 · 필터 반환 행 수)은 tools/io_day23.py 를 먼저 실행해
%TEMP%/d23_results.json 에 저장된 실제 실행 결과(NumPy · PostgreSQL 18.6 + pgvector 0.8.7)를 읽는다.
벡터는 2차원 예시 좌표와 덩어리 구조 임의 벡터이고, 모델 출력이 아니다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day23.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = json.loads((Path(tempfile.gettempdir()) / "d23_results.json").read_text(encoding="utf-8"))
MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=9.5, bold=False, ls="-", lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec or C["purple"], lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.7, ls="-", text=None, tfs=8.6, toff=(0, 0.08)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color or MUTED, lw=lw, ls=ls))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="bottom", fontsize=tfs, color=color or MUTED)


# ------------------------------------------------------------ 1. 큰 그림: 두 경로
def fig_flow():
    fig, ax = canvas(12.6, 5.2, 12.6, 5.2)
    ax.text(6.3, 4.95, "검색 = 미리 벡터로 저장해 둔 원문 중에서, 질문 벡터와 가까운 후보를 꺼내는 일", ha="center",
            fontsize=10.6, fontweight="bold", color=C["ink"])
    # 자료 준비 (위)
    ax.text(0.2, 4.35, "자료 준비 — 원문이 바뀔 때만", fontsize=9.6, color=C["mint"], fontweight="bold")
    top = [("원문 · ID\n검사", "#f8f9fa", C["gray"]), ("passage: 접두어\nEmbedding", "#e6fcf5", C["mint"]),
           ("벡터 검사\nshape · NaN · 길이", "#e6fcf5", C["mint"]), ("PostgreSQL\n원문 + vector(384)\n한 행 저장", "#e7f5ff", C["blue"])]
    xs = [0.2, 2.75, 5.3, 7.85]
    for x, (t, fc, ec) in zip(xs, top):
        rbox(ax, x, 3.0, 2.05, 1.2, t, fc=fc, ec=ec, fs=9)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 2.05, 3.6, b, 3.6)
    # 질문 처리 (아래)
    ax.text(0.2, 2.35, "질문 처리 — 요청마다", fontsize=9.6, color=C["purple"], fontweight="bold")
    bot = [("질문\n입력 검사", "#f8f9fa", C["gray"]), ("query: 접두어\nEmbedding", "#f3f0ff", C["purple"]),
           ("pgvector\n거리 계산 · Top-k\n(+ 필터 · 인덱스)", "#e7f5ff", C["blue"]), ("원문 후보\nID · 본문 · 거리", "#fff4e6", C["orange"]),
           ("관련성 검토\n→ Context", "#fff9db", C["yellow"])]
    xs2 = [0.2, 2.75, 5.3, 7.85, 10.4]
    for x, (t, fc, ec) in zip(xs2, bot):
        rbox(ax, x, 1.0, 2.05, 1.2, t, fc=fc, ec=ec, fs=9)
    for a, b in zip(xs2[:-1], xs2[1:]):
        arrow(ax, a + 2.05, 1.6, b, 1.6)
    arrow(ax, 8.875, 3.0, 6.6, 2.2, color=C["blue"], text="저장된 벡터와 비교", toff=(1.25, -0.1))
    ax.text(3.78, 2.48, "같은 모델 · 같은 정규화 계약", ha="center", fontsize=8.6, color=C["purple"])
    ax.plot([3.78, 3.78], [2.62, 2.98], color=C["purple"], lw=1.2, ls=":")
    rbox(ax, 2.75, 0.05, 9.7, 0.7, "점수가 가장 높은 후보도 '가장 가까운 벡터'일 뿐 — 답할 근거인지는 원문을 읽고 따로 판단",
         fc="#fff5f5", ec=C["red"], fs=9, ls="--", lw=1.4)
    save(fig, 23, "flow")


# ------------------------------------------------------------ 2. 2차원 세 척도
def fig_metrics():
    m = R["math"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.set_aspect("equal"); ax.set_xlim(-0.2, 1.35); ax.set_ylim(-0.25, 1.1)
    th = np.linspace(0, np.pi / 2, 60)
    ax.plot(np.cos(th), np.sin(th), color="#d0c8e8", lw=1, ls="--")
    for (x, y), name, col in [((1, 0), "q (1, 0)", C["purple"]), ((0.8, 0.6), "a (0.8, 0.6)", C["pink"])]:
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.text(x + 0.03, y + 0.03, name, color=col, fontsize=10, fontweight="bold")
    ax.plot([1, 0.8], [0, 0.6], color=C["orange"], lw=2.2, ls=":")
    ax.text(0.97, 0.3, f"L2 = {m['l2']}", color=C["orange"], fontsize=10, fontweight="bold")
    ang = np.linspace(0, np.arctan2(0.6, 0.8), 30)
    ax.plot(0.3 * np.cos(ang), 0.3 * np.sin(ang), color=C["mint"], lw=1.8)
    ax.text(0.33, 0.06, f"cos = {m['cos']}", color=C["mint"], fontsize=10, fontweight="bold")
    ax.set_title("길이 1인 두 벡터: Dot = Cosine")
    ax.text(0.0, -0.2, f"Dot = {m['dot']} · Cosine distance = 1 - {m['cos']} = {round(1 - m['cos'], 4)}", fontsize=9.4, color=MUTED)

    ax = axes[1]
    ax.set_aspect("equal"); ax.set_xlim(-0.5, 11); ax.set_ylim(-1, 11)
    ax.set_aspect("auto")
    for (x, y), name, col in [((1, 0), "q = U (1, 0)", C["purple"]), ((10, 10), "V (10, 10)", C["blue"])]:
        ax.annotate("", (x, y), (0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2))
        ax.text(x + 0.25, y - 0.6 if name.startswith("q") else y + 0.2, name, color=col, fontsize=10, fontweight="bold")
    ax.set_title("길이가 다르면: Dot과 Cosine의 1등이 갈림")
    ax.text(0.2, 10.2, f"Dot      q·U = 1   <   q·V = 10   → V가 1등", fontsize=9.6, color=C["ink"])
    ax.text(0.2, 9.3, f"Cosine  cos(q,U) = {m['cosU']}  >  cos(q,V) = {m['cosV']}   → U가 1등", fontsize=9.6, color=C["ink"])
    ax.text(0.2, 8.3, "정규화(길이 1로) 하면 두 척도의 순위가 같아짐", fontsize=9.4, color=C["mint"], fontweight="bold")
    save(fig, 23, "metrics")


# ------------------------------------------------------------ 3. pgvector 네 층
def fig_layers():
    fig, ax = canvas(12.6, 3.9, 12.6, 3.9)
    items = [("① 서버 파일", "vector.dll · vector.control\nvector--*.sql", "nmake … install\n(빌드 전용 x64 명령창)", "#e7f5ff", C["blue"]),
             ("② DB 활성화", "현재 DB의 pg_extension\n에 vector 0.8.7", "CREATE EXTENSION\nIF NOT EXISTS vector", "#e6fcf5", C["mint"]),
             ("③ Python 패키지", "pgvector(Python)\npsycopg · python-dotenv", "python -m pip install\npgvector", "#f3f0ff", C["purple"]),
             ("④ 연결 등록", "이 연결에서\nvector ↔ Vector 변환", "register_vector(conn)\n연결마다", "#fff4e6", C["orange"])]
    xs = [0.2, 3.3, 6.4, 9.5]
    for x, (h, what, how, fc, ec) in zip(xs, items):
        rbox(ax, x, 1.55, 2.9, 1.55, f"{h}\n{what}", fc=fc, ec=ec, fs=9.4)
        ax.text(x + 1.45, 1.15, how, ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.3)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 2.9, 2.32, b, 2.32)
    ax.text(6.3, 3.55, "네 층은 서로 다른 설치 — 하나를 했다고 다음 층이 생기지 않음", ha="center", fontsize=10.4, fontweight="bold", color=C["ink"])
    ax.text(6.3, 0.3, "pip install pgvector ≠ 서버 확장 설치 · CREATE EXTENSION 은 연결한 DB에만 · 새 연결엔 register_vector 다시",
            ha="center", fontsize=9, color=C["red"])
    save(fig, 23, "layers")


# ------------------------------------------------------------ 4. ANN: ef_search · 필터
def fig_ann():
    a = R["ann"]
    efs = [int(k) for k in a["sweep"]]
    rec = [a["sweep"][str(k)][0] for k in efs]
    ms = [a["sweep"][str(k)][1] for k in efs]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    ax = axes[0]
    ax.plot(efs, rec, "o-", color=C["purple"], lw=2.2, label="recall@10")
    for x, y in zip(efs, rec):
        ax.text(x, y - 0.045, f"{y:.3f}", ha="center", fontsize=8.6, color=C["purple"])
    ax.set_xscale("log"); ax.set_xticks(efs); ax.set_xticklabels([str(e) for e in efs])
    ax.set_ylim(0.8, 1.03); ax.set_xlabel("hnsw.ef_search"); ax.set_ylabel("recall@10 (Exact Top-10 중 찾은 비율)")
    ax2 = ax.twinx()
    ax2.plot(efs, ms, "s--", color=C["orange"], lw=1.6, label="ms/질의")
    ax2.set_ylabel("질의 1개 평균 시간 (ms, 이 PC)", color=C["orange"]); ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax.set_title(f"탐색 폭 ↑ → 놓침 ↓ · 시간 ↑  (임의 벡터 {a['N']:,}개)")
    ax = axes[1]
    off, on = a["rare_off"], a["rare_on"]
    bins = np.arange(0, 12) - 0.5
    ax.hist(off, bins=bins, color=C["red"], alpha=0.8, label=f"iterative_scan off (평균 {np.mean(off):.2f}행)")
    ax.hist(on, bins=bins, color=C["green"], alpha=0.8, label=f"relaxed_order (평균 {np.mean(on):.2f}행)")
    ax.axvline(10, color=MUTED, ls=":", lw=1.2); ax.text(9.6, 46, "k = 10", ha="right", fontsize=9, color=MUTED)
    ax.set_xlabel("반환된 행 수"); ax.set_ylabel("질의 수 (50개 중)"); ax.set_xticks(range(0, 11))
    ax.set_title("WHERE category='rare'(1%) + HNSW · ef_search 40")
    ax.legend(loc="upper center", fontsize=9)
    save(fig, 23, "ann")


if __name__ == "__main__":
    fig_flow(); fig_metrics(); fig_layers(); fig_ann()
