import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day20 그림 만들기 (LLM 생태계 · Token · Embedding · Transformer · Attention · 학습/추론).

숫자가 들어간 그림은 여기서 강의 코드를 실제로 다시 계산한 값이다
(Tokenizer bert-base-multilingual-cased 결과 · 3×3 Attention · cosine · Softmax/Temperature).
문맥 예산 막대는 강의 p33의 '산술 예시' 숫자를 그대로 그린 것이다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day20.py
"""
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=9.5, bold=False, color=None, ls="-", lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec or C["purple"], lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=color or C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.7, ls="-", text=None, tfs=8.6, toff=(0, 0.08)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color or MUTED, lw=lw, ls=ls))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="bottom", fontsize=tfs, color=color or MUTED)


# ------------------------------------------------------------ 1. 서비스 속 LLM 위치
def fig_system():
    fig, ax = canvas(12, 5.2, 12, 5.2)
    rbox(ax, 0.2, 2.0, 1.7, 1.1, "사용자\n“FastAPI에서\nPostgreSQL 연결이\n안 됩니다”", fc="#f8f9fa", ec=C["gray"], fs=8.6)
    rbox(ax, 2.4, 2.0, 1.8, 1.1, "FastAPI\n요청·응답·검증", fc="#e7f5ff", ec=C["blue"], fs=9.2)
    rbox(ax, 4.7, 2.0, 2.0, 1.1, "Agent 처리\n분기·상태·승인", fc="#fff4e6", ec=C["orange"], fs=9.2)
    rbox(ax, 7.3, 1.75, 2.4, 1.6, "LLM  ← 오늘\n문맥을 읽고\n다음 Token 생성\n(또는 행동 제안)", fc="#f3f0ff", ec=C["purple"], fs=9.6, bold=True, lw=2.6)
    rbox(ax, 7.3, 3.75, 2.4, 0.85, "RAG · 검색 근거\n원본 문서 / 벡터 인덱스", fc="#e6fcf5", ec=C["mint"], fs=8.6)
    rbox(ax, 4.7, 0.15, 2.0, 0.95, "Tool · 업무 시스템\n허용된 API · 권한·기록", fc="#fff0f6", ec=C["pink"], fs=8.6)
    rbox(ax, 10.2, 2.0, 1.6, 1.1, "응답\n설명·근거·\n확인 절차", fc="#f8f9fa", ec=C["gray"], fs=8.8)
    arrow(ax, 1.9, 2.55, 2.4, 2.55); arrow(ax, 4.2, 2.55, 4.7, 2.55); arrow(ax, 6.7, 2.55, 7.3, 2.55)
    arrow(ax, 8.5, 3.75, 8.5, 3.35, color=C["mint"], text="근거 → Context", toff=(0.75, -0.12))
    arrow(ax, 5.7, 2.0, 5.7, 1.1, color=C["pink"], text="실행은 Backend가\n권한 확인", toff=(-0.95, -0.25))
    arrow(ax, 9.7, 2.55, 10.2, 2.55)
    ax.text(6, 5.0, "LLM은 문장을 만드는 부분 · 근거 찾기(RAG)와 실제 실행(Tool)은 다른 부품", ha="center", fontsize=10.5, color=MUTED)
    save(fig, 20, "system_map")


# ------------------------------------------------------------ 2. 확대 지도 (Agent → Attention 계산)
def fig_zoom():
    fig, ax = canvas(11, 5.0, 11, 5.0)
    levels = [("01  AI Agent 서비스", "요청·도구·상태·권한을 조합", C["gray"]),
              ("02  LLM 호출", "텍스트 Context → 생성 결과", C["blue"]),
              ("03  Transformer 층 반복", "Block N개를 거치며 Token 표현이 바뀜", C["purple"]),
              ("04  Transformer Block", "Attention + FFN + Residual/Norm", C["pink"]),
              ("05  Self-Attention", "Token 사이 관계로 표현을 섞음", C["orange"]),
              ("06  Q / K / V · Mask · Softmax", "Score → Weight → Value 가중합", C["mint"])]
    for i, (t, sub, col) in enumerate(levels):
        x0, y0 = 0.25 + i * 0.42, 0.2 + i * 0.36
        w, h = 10.5 - i * 0.84, 4.6 - i * 0.72
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white" if i < 5 else "#e6fcf5", ec=col, lw=2.0))
        ax.text(x0 + 0.18, y0 + h - 0.24, t, fontsize=10, fontweight="bold", color=col, va="center")
        ax.text(x0 + w - 0.18, y0 + h - 0.24, sub, fontsize=8.8, color=MUTED, va="center", ha="right")
    ax.text(5.5, 0.62 + 5 * 0.36, "scores = Q · K.T / √d_k  →  Mask  →  softmax  →  weights · V", ha="center", va="center", fontsize=10, color=C["ink"])
    save(fig, 20, "zoom_map")


# ------------------------------------------------------------ 3. 문장 → 다음 Token (실제 Tokenizer 값)
def fig_pipeline():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    text = "실습 파일은 개인 폴더에 보관합니다."
    ids = tok(text)["input_ids"]; toks = tok.convert_ids_to_tokens(ids)
    fig, ax = canvas(12.5, 4.4, 12.5, 4.4)
    cols = [C["gray"], C["blue"], C["blue"], C["purple"], C["pink"], C["orange"], C["mint"]]
    steps = [("문자열", "“실습 파일은\n개인 폴더에\n보관합니다.”\n" + f"공백 포함 {len(text)}글자"),
             ("Tokenizer", "\n".join(" ".join(toks[i:i + 4]) for i in range(0, len(toks), 4))),
             ("Token ID", ", ".join(map(str, ids[:4])) + ",\n" + ", ".join(map(str, ids[4:8])) + ",\n… " + str(ids[-1]) + f"\n정수 {len(ids)}개"),
             ("Embedding", f"ID마다 표에서\n행 하나 꺼냄\n→ [{len(ids)}, D] 벡터"),
             ("Transformer", "Block × N\nAttention·FFN\n→ [T, D] 문맥 표현"),
             ("LM Head", "마지막 위치 →\n어휘 전체 점수\n(Logit, 확률 아님)"),
             ("Decode", "Softmax·선택\n→ 다음 Token\n문맥 끝에 붙여 반복")]
    w, gap = 1.62, 0.13
    for i, ((t, sub), col) in enumerate(zip(steps, cols)):
        x = 0.1 + i * (w + gap)
        rbox(ax, x, 3.25, w, 0.6, t, fc="white", ec=col, fs=10, bold=True, color=col)
        rbox(ax, x, 1.45, w, 1.6, sub, fc="#fbfaff", ec=col, fs=8.2 if i in (1, 2) else 8.6, lw=1.2)
        if i < len(steps) - 1:
            arrow(ax, x + w, 3.55, x + w + gap, 3.55, lw=1.4)
    ax.annotate("", xy=(0.9, 1.3), xytext=(11.75, 1.3),
                arrowprops=dict(arrowstyle="-|>", color=C["mint"], lw=1.6, ls="--", connectionstyle="arc3,rad=-0.07"))
    ax.text(6.25, 0.3, "선택한 Token을 입력 끝에 붙이고 EOS·길이 제한까지 반복 (Autoregressive)", ha="center", fontsize=9.6, color=C["mint"])
    ax.text(6.25, 4.15, "Token·ID 칸은 bert-base-multilingual-cased로 실제 실행한 값 · 뒤 단계는 개념", ha="center", fontsize=9.2, color=MUTED)
    save(fig, 20, "pipeline")


# ------------------------------------------------------------ 4. 문맥 예산 (강의 산술 예시)
def fig_budget():
    parts = [("System + 지시", 300, C["purple"]), ("이전 대화", 1200, C["blue"]), ("검색 근거", 2000, C["mint"]),
             ("User 질문", 150, C["orange"]), ("출력에 예약", 850, C["pink"])]
    total = sum(p[1] for p in parts)
    fig, ax = plt.subplots(figsize=(10, 2.5))
    left = 0
    for name, n, col in parts:
        ax.barh(0, n, left=left, color=col, height=0.5, edgecolor="white", lw=2)
        ax.text(left + n / 2, 0, f"{name}\n{n:,}", ha="center", va="center", fontsize=9 if n > 250 else 7.6, color="white", fontweight="bold")
        left += n
    ax.set_xlim(0, total); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(f"Token (합계 {total:,} — 강의의 산술 예시, 특정 모델의 한도 아님)")
    ax.set_title("한 요청의 Context Window를 나눠 쓰는 예")
    for s in ["left"]:
        ax.spines[s].set_visible(False)
    save(fig, 20, "context_budget")


# ------------------------------------------------------------ 5. Cosine 2D
def fig_cosine():
    vecs = {"a [1, 0]": (np.array([1.0, 0.0]), C["purple"]), "b [0.8, 0.2]": (np.array([0.8, 0.2]), C["mint"]),
            "c [-1, 0]": (np.array([-1.0, 0.0]), C["pink"]), "d [3, 0]": (np.array([3.0, 0.0]), C["blue"])}
    a = torch.tensor([1.0, 0.0])
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for name, (v, col) in vecs.items():
        if name.startswith("d"):
            ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, ls="--", alpha=0.6))
        else:
            ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
        cs = F.cosine_similarity(a, torch.tensor(v, dtype=torch.float32), dim=0).item()
        off = {"a": (0.02, -0.17), "b": (0.0, 0.08), "c": (-0.05, 0.08), "d": (-0.35, 0.1)}[name[0]]
        ax.text(v[0] + off[0], v[1] + off[1], f"{name}\ncos(a, ·) = {cs:.4f}" if name[0] != "a" else name, color=col, fontsize=9.2,
                ha="center", fontweight="bold")
    ax.set_xlim(-1.5, 3.4); ax.set_ylim(-0.45, 0.55); ax.set_aspect("equal")
    ax.axhline(0, color="#cfc8e0", lw=0.8); ax.axvline(0, color="#cfc8e0", lw=0.8)
    ax.set_title("Cosine은 방향만 본다 — 같은 방향이면 길이가 3배여도 1")
    save(fig, 20, "cosine_2d")


# ------------------------------------------------------------ 6. Attention: Mask 있음 vs 없음 (실제 계산)
def fig_attention():
    q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]]); k = v = q
    scores = q @ k.T / math.sqrt(2)
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1)
    wm = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
    wn = torch.softmax(scores, -1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1, 1, 1]})
    lab = ["t1", "t2", "t3"]
    for ax, M, title, cmap in [(axes[0], scores, "① scores = Q·K.T / √2 (Softmax 전)", "Purples"),
                               (axes[1], wm, "② Causal Mask + Softmax", "Greens"),
                               (axes[2], wn, "③ Mask 없이 Softmax", "Oranges")]:
        ax.imshow(M.numpy(), cmap=cmap, vmin=0, vmax=1.5 if M is scores else 1)
        for i in range(3):
            for j in range(3):
                masked = (M is wm) and bool(mask[i, j])
                ax.text(j, i, "0\n(가림)" if masked else f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10.5,
                        color="#adb5bd" if masked else ("white" if M[i, j] >= (1.2 if M is scores else 0.8) else C["ink"]),
                        fontweight="bold")
        ax.set_xticks(range(3), [f"Key {x}" for x in lab]); ax.set_yticks(range(3), [f"Query {x}" for x in lab])
        ax.grid(False); ax.set_title(title, fontsize=11)
        if M is not scores:
            o = (M @ v)
            ax.set_xlabel("출력 = weights·V → " + "  ".join(f"{x}[{r[0]:.2f}, {r[1]:.2f}]" for x, r in zip(lab, o)), fontsize=8.4)
    axes[0].set_xlabel("행 = Query(지금 위치) · 열 = Key(참고 대상)", fontsize=8.6)
    save(fig, 20, "attention_mask")


# ------------------------------------------------------------ 7. Transformer Block
def fig_block():
    fig, ax = canvas(12, 4.6, 12, 4.6)
    rbox(ax, 0.1, 1.8, 1.6, 1.0, "Token + Position\n벡터\n[B, T, D]", fc="#e7f5ff", ec=C["blue"], fs=9)
    ax.add_patch(FancyBboxPatch((2.1, 0.45), 6.9, 3.7, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#fbfaff", ec=C["purple"], lw=2, ls="--"))
    ax.text(5.55, 3.9, "Transformer Block  (× N층, 모델마다 순서 조금씩 다름)", ha="center", fontsize=10, color=C["purple"], fontweight="bold")
    rbox(ax, 2.4, 1.85, 1.5, 0.9, "Self-Attention\nToken 사이\n정보 교환", fc="white", ec=C["orange"], fs=8.8)
    rbox(ax, 4.25, 1.85, 1.25, 0.9, "Add & Norm\nx + f(x)", fc="white", ec=C["gray"], fs=8.8)
    rbox(ax, 5.85, 1.85, 1.5, 0.9, "Feed-Forward\nToken별\n비선형 변환", fc="white", ec=C["pink"], fs=8.8)
    rbox(ax, 7.65, 1.85, 1.15, 0.9, "Add & Norm", fc="white", ec=C["gray"], fs=8.8)
    arrow(ax, 1.7, 2.3, 2.4, 2.3); arrow(ax, 3.9, 2.3, 4.25, 2.3); arrow(ax, 5.5, 2.3, 5.85, 2.3); arrow(ax, 7.35, 2.3, 7.65, 2.3)
    # residual 우회선
    ax.annotate("", xy=(4.5, 2.75), xytext=(2.25, 2.75), arrowprops=dict(arrowstyle="-|>", color=C["mint"], lw=1.4, connectionstyle="arc3,rad=-0.55"))
    ax.annotate("", xy=(7.95, 2.75), xytext=(5.65, 2.75), arrowprops=dict(arrowstyle="-|>", color=C["mint"], lw=1.4, connectionstyle="arc3,rad=-0.55"))
    ax.text(5.55, 0.75, "초록 곡선 = Residual(원래 표현을 더해 주는 우회로) · 입력과 출력 shape는 같은 [B, T, D]", ha="center", fontsize=8.8, color=C["mint"])
    arrow(ax, 8.8, 2.3, 9.4, 2.3)
    rbox(ax, 9.4, 1.8, 1.15, 1.0, "LM Head\nLinear", fc="#fff4e6", ec=C["orange"], fs=9)
    arrow(ax, 10.55, 2.3, 10.85, 2.3)
    rbox(ax, 10.85, 1.6, 1.1, 1.4, "Logits\n[B, T, V]\n어휘 V개\n점수", fc="#e6fcf5", ec=C["mint"], fs=8.8)
    save(fig, 20, "transformer_block")


# ------------------------------------------------------------ 8. Temperature (실제 계산)
def fig_temperature():
    logits = torch.tensor([2.0, 1.0, 0.0])
    temps = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    xs = np.arange(3); w = 0.26
    for i, (t, col) in enumerate(zip(temps, [C["purple"], C["mint"], C["orange"]])):
        p = torch.softmax(logits / t, 0).numpy()
        ax.bar(xs + (i - 1) * w, p, w, color=col, label=f"temperature {t}")
        for x, y in zip(xs + (i - 1) * w, p):
            ax.text(x, y + 0.015, f"{y:.3f}", ha="center", fontsize=8.6)
    ax.set_xticks(xs, ["후보 0\n(logit 2.0)", "후보 1\n(logit 1.0)", "후보 2\n(logit 0.0)"])
    ax.set_ylabel("확률"); ax.set_ylim(0, 1); ax.legend(loc="upper right")
    ax.set_title("같은 Logit, 다른 Temperature — 순서는 그대로, 몰림 정도만 바뀜")
    save(fig, 20, "temperature")


# ------------------------------------------------------------ 9. 학습 vs 추론
def fig_train_infer():
    fig, ax = canvas(12, 4.0, 12, 4.0)
    ax.text(0.2, 3.7, "Training · 모델 개발", fontsize=11, fontweight="bold", color=C["pink"])
    ax.text(0.2, 1.9, "Inference · 서비스 응답", fontsize=11, fontweight="bold", color=C["mint"])
    tr = ["입력 Token\n→ Logit", "정답 Token과\nLoss 비교", "backward()\n기울기", "optimizer.step()\n가중치 θ 변경"]
    inf = ["입력 Token\n→ Logit", "Softmax·디코딩\n다음 Token 선택", "선택한 Token을\n입력 끝에 붙임", "반복 (θ 고정)\nKV Cache 가능"]
    for row, (items, col, y) in enumerate([(tr, C["pink"], 2.5), (inf, C["mint"], 0.7)]):
        for i, t in enumerate(items):
            x = 0.2 + i * 2.95
            rbox(ax, x, y, 2.45, 0.95, t, fc="white", ec=col, fs=9.2)
            if i < 3:
                arrow(ax, x + 2.45, y + 0.47, x + 2.95, y + 0.47)
    ax.annotate("", xy=(1.2, 0.7), xytext=(9.4, 0.7), arrowprops=dict(arrowstyle="-|>", color=C["mint"], lw=1.3, ls="--", connectionstyle="arc3,rad=-0.12"))
    save(fig, 20, "train_infer")


if __name__ == "__main__":
    fig_system(); fig_zoom(); fig_pipeline(); fig_budget(); fig_cosine(); fig_attention(); fig_block(); fig_temperature(); fig_train_infer()
    print("ok")
