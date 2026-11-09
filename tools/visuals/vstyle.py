"""모든 Day 그림이 같은 느낌으로 나오게 하는 공통 스타일.

사용법 (각 make_dayNN.py 안에서):
    from vstyle import plt, C, save, setup
    setup()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ...
    save(fig, 15, "iou_examples")   # → notes/img/day15/iou_examples.png

실행은 수업 가상환경 파이썬으로:
    C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day15.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]          # SeSAC 폴더
IMG = ROOT / "notes" / "img"

# 사이트 색과 맞춘 팔레트
C = {
    "purple": "#7c4dff", "pink": "#ff5fa2", "mint": "#0fb5a6", "orange": "#ff922b",
    "blue": "#4dabf7", "yellow": "#fab005", "green": "#2b8a3e", "red": "#e03131",
    "gray": "#868e96", "ink": "#2a2340", "soft": "#f2ecff",
}
CYCLE = [C["purple"], C["pink"], C["mint"], C["orange"], C["blue"], C["yellow"]]


def setup():
    plt.rcParams.update({
        "font.family": "Malgun Gothic",
        "axes.unicode_minus": False,
        "figure.dpi": 100,
        "savefig.dpi": 160,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "axes.edgecolor": "#cfc8e0",
        "axes.labelcolor": C["ink"],
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#ece6f7",
        "grid.linewidth": 0.8,
        "xtick.color": "#5b5270",
        "ytick.color": "#5b5270",
        "legend.frameon": False,
        "font.size": 11,
    })


def save(fig, day, name):
    out = IMG / f"day{day:02d}"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", p.relative_to(ROOT))
    return p
