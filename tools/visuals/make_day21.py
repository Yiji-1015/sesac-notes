import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day21 그림 만들기 (Prompt · Context Engineering).

검증 표(validation_grid)의 칸은 tools/io_day21.py 를 import 해서 교안 샘플 응답 14개를
실제로 parse_answer · final_rules · load_json_strict 에 넣은 결과다. 근거 칸만 교안이 정한
사람 검토 결과(잘못된 인용 1건)다. 예산 막대는 교안의 가상 Token 예산 숫자다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day21.py
"""
import contextlib
import io as _io

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
with contextlib.redirect_stdout(_io.StringIO()):
    import io_day21 as D  # noqa: E402

MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=9.5, bold=False, color=None, ls="-", lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec or C["purple"], lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=color or C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.7, ls="-", text=None, tfs=8.6, toff=(0, 0.08), cs="arc3"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color or MUTED, lw=lw, ls=ls,
                                 connectionstyle=cs))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="bottom", fontsize=tfs, color=color or MUTED)


# ------------------------------------------------------------ 1. 전체 흐름
def fig_flow():
    fig, ax = canvas(12.5, 4.6, 12.5, 4.6)
    ax.text(0.2, 4.3, "실행 흐름 — 요청 한 건마다", fontsize=10.5, fontweight="bold", color=C["ink"])
    xs = [0.2, 2.25, 4.3, 6.35, 8.4, 10.45]
    items = [("질문\n무엇을 알고 싶은가", "#f8f9fa", C["gray"]),
             ("Context\n필요한 자료 선택", "#e6fcf5", C["mint"]),
             ("Prompt\n지시·질문·자료 조립", "#f3f0ff", C["purple"]),
             ("LLM 응답\n검사할 결과 생성", "#fff0f6", C["pink"]),
             ("자동 검사\n문법·구조·참조·상태", "#e7f5ff", C["blue"]),
             ("근거 대조\n질문·답·원문을\n사람이 확인", "#fff4e6", C["orange"])]
    for x, (t, fc, ec) in zip(xs, items):
        rbox(ax, x, 2.35, 1.85, 1.35, t, fc=fc, ec=ec, fs=9.2)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 1.85, 3.02, b, 3.02)
    ax.text(2.25 + 2.95, 3.95, "입력 설계 (①~④)", ha="center", fontsize=9, color=C["purple"])
    ax.text(8.4 + 2.0, 3.95, "출력 확인 (⑤~⑦)", ha="center", fontsize=9, color=C["blue"])
    ax.text(0.2, 1.55, "평가 루프 — 여러 실행을 모아서 (⑧)", fontsize=10.5, fontweight="bold", color=C["ink"])
    rbox(ax, 4.3, 0.2, 4.0, 1.05, "A/B 평가\n여러 실행의 조건·응답·실패 기록을 모아 비교", fc="#fff9db", ec=C["yellow"], fs=9.2, ls="--")
    arrow(ax, 11.4, 2.35, 8.35, 0.75, color=C["yellow"], ls="--", text="기록 모으기", toff=(0.6, -0.05))
    arrow(ax, 4.3, 0.75, 5.2, 2.33, color=C["yellow"], ls="--", text="입력 개선 (새 버전)", toff=(-1.25, -0.1))
    ax.text(12.3, 0.25, "A/B는 요청마다 거치는 단계가 아니에요", ha="right", fontsize=8.6, color=MUTED)
    save(fig, 21, "flow")


# ------------------------------------------------------------ 2. 여섯 요소 + 변수 → Prompt
def fig_six():
    fig, ax = canvas(12, 4.6, 12, 4.6)
    rbox(ax, 0.2, 0.25, 4.6, 4.1, "", fc="#f3f0ff", ec=C["purple"])
    ax.text(2.5, 4.0, "고정 템플릿 (버전 관리)", ha="center", fontsize=10.5, fontweight="bold", color=C["purple"])
    rows = [("역할", "누구를 돕는가?"), ("작업", "무엇을 수행하는가?"), ("입력", "무엇을 읽는가?"),
            ("제약", "어디까지 답하는가?"), ("실패", "근거가 없으면?"), ("출력", "어떤 형식으로?")]
    for i, (k, v) in enumerate(rows):
        y = 3.35 - i * 0.52
        rbox(ax, 0.45, y, 1.0, 0.42, k, fc="#ffffff", ec=C["purple"], fs=9.4, bold=True)
        ax.text(1.65, y + 0.21, v, va="center", fontsize=9.4, color=C["ink"])
    rbox(ax, 5.55, 2.45, 2.9, 1.6, "요청마다 바뀌는 값\n\n$question  이번 질문\n$context_json  고른 지식", fc="#e6fcf5", ec=C["mint"], fs=9.3)
    rbox(ax, 5.55, 0.45, 2.9, 1.45, "$contract\n출력 계약 (버전으로 관리)", fc="#e7f5ff", ec=C["blue"], fs=9.3)
    ax.text(5.1, 2.25, "+", fontsize=22, ha="center", va="center", color=MUTED)
    rbox(ax, 9.4, 1.4, 2.4, 1.8, "완성된 Prompt\n\nsubstitute()가\n돌려준 문자열\n→ 모델에 전달", fc="#ffffff", ec=C["ink"], fs=9.4, bold=False, lw=2.2)
    arrow(ax, 8.45, 3.2, 9.4, 2.55); arrow(ax, 8.45, 1.15, 9.4, 2.0)
    save(fig, 21, "prompt_six")


# ------------------------------------------------------------ 3. Context 선택
def fig_context():
    fig, ax = canvas(12.5, 4.4, 12.5, 4.4)
    ax.text(1.55, 4.05, "DB · 저장된 지식", ha="center", fontsize=10, fontweight="bold", color=C["ink"])
    for i, r in enumerate(D.KNOWLEDGE):
        sel = r["id"] == 101
        rbox(ax, 0.2, 3.0 - i * 1.1, 2.7, 0.85, f"{r['id']}  {r['title']}\n{r['content']}", fs=8.4,
             fc="#e6fcf5" if sel else "#f8f9fa", ec=C["mint"] if sel else C["gray"], lw=2.4 if sel else 1.4)
    rbox(ax, 3.8, 3.0, 3.0, 1.05, "같은 자료 101에 질문 두 개\n① 어디에 보관하나요? (위치)\n② 몇 년 보관하나요? (기간)", fc="#fff4e6", ec=C["orange"], fs=8.6)
    arrow(ax, 5.3, 3.0, 5.3, 2.57)
    rbox(ax, 3.8, 1.35, 3.0, 1.2, "사람이 고른 ID로\nbuild_context(rows, [101])\n→ context_json\n'[{\"id\": 101, …}]'", fc="#f3f0ff", ec=C["purple"], fs=8.8)
    arrow(ax, 2.9, 3.42, 3.8, 1.95, color=C["mint"], lw=2.2)
    ax.text(5.3, 0.75, "102 · 103 = 이번에 전달 안 함\n(DB에 있다 ≠ 이번에 줬다)", ha="center", fontsize=8.6, color=MUTED)
    rbox(ax, 7.7, 1.35, 2.2, 1.2, "provided_ids\n[101]\n= 출처 검사 범위", fc="#e7f5ff", ec=C["blue"], fs=9.0)
    arrow(ax, 6.8, 1.95, 7.7, 1.95)
    rbox(ax, 10.3, 2.65, 2.0, 1.25, "위치를 물으면\nanswered\nsource_ids [101]", fc="#ebfbee", ec=C["green"], fs=8.8)
    rbox(ax, 10.3, 0.55, 2.0, 1.45, "기간을 물으면\ninsufficient_context\nsource_ids []\n부족: 보관 기간", fc="#fff5f5", ec=C["red"], fs=8.6)
    arrow(ax, 9.9, 2.2, 10.3, 3.2); arrow(ax, 9.9, 1.7, 10.3, 1.3)
    save(fig, 21, "context_select")


# ------------------------------------------------------------ 4. Token 예산
def fig_budget():
    total, ins, ex, out, mg, rem = D.RESULTS["budget"]
    fig, ax = plt.subplots(figsize=(11, 3.0))
    bars = [("기본 배정", [("지시·질문", ins, C["purple"]), ("예제", ex, C["pink"]), ("근거 자료", rem, C["mint"]),
                       ("출력용 공간", out, C["blue"]), ("여유", mg, C["gray"])]),
            ("예제 +1000", [("지시·질문", ins, C["purple"]), ("예제", ex + 1000, C["pink"]), ("근거 자료", rem - 1000, C["mint"]),
                          ("출력용 공간", out, C["blue"]), ("여유", mg, C["gray"])])]
    for yi, (name, segs) in enumerate(bars):
        y = 1 - yi
        left = 0
        for lab, v, col in segs:
            ax.barh(y, v, left=left, color=col, edgecolor="white", height=0.62)
            if v >= 500:
                ax.text(left + v / 2, y, f"{lab}\n{v:,}", ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
            else:
                ax.text(left + v / 2, y + 0.42, f"{lab} {v}", ha="center", va="bottom", fontsize=8, color=MUTED)
            left += v
    ax.set_yticks([1, 0]); ax.set_yticklabels([b[0] for b in bars], fontsize=9.5)
    ax.set_xlim(0, total); ax.set_ylim(-0.55, 1.75); ax.grid(False)
    ax.set_xticks([0, 2048, 4096, 6144, 8192]); ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlabel(f"Context Window {total:,} tokens (가상 예산 · 실제 측정값 아님)", fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, 21, "budget")


# ------------------------------------------------------------ 5. 검증 단계 표 (실제 결과)
def fig_validation():
    tab = D.RESULTS["fixtures"]
    stages = ["① 문법\nJSON", "② 구조\nSchema", "③ 규칙\n참조·상태", "④ 근거\n사람 검토"]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(0, 4 + 3.3); ax.set_ylim(len(tab) + 1.2, 0); ax.axis("off")
    for j, s in enumerate(stages):
        ax.text(3.3 + j + 0.5, 0.55, s, ha="center", va="center", fontsize=9.2, fontweight="bold", color=C["ink"])
    for i, t in enumerate(tab):
        y = i + 1.2
        ax.text(0.05, y + 0.5, t["label"], va="center", fontsize=9.2, color=C["ink"])
        ax.text(2.35, y + 0.5, f"ids {t['ids']}", va="center", fontsize=7.8, color=MUTED)
        code = t["final_strict"]
        stage_of = {"json_error": 0, "schema_error": 1}.get(code, 2 if code != "None" else 3)
        cells = []
        for j in range(4):
            if j < stage_of:
                cells.append(("통과", "#d3f9d8", C["green"]))
            elif j == stage_of and code != "None":
                cells.append((code, "#ffe3e3", C["red"]))
            elif j == 3 and code == "None":
                bad = t["key"] == "false_evidence"
                cells.append(("근거 불일치" if bad else "적합", "#ffe3e3" if bad else "#d3f9d8", C["red"] if bad else C["green"]))
            else:
                cells.append(("—", "#f1f3f5", C["gray"]))
        for j, (txt, fc, ec) in enumerate(cells):
            ax.add_patch(Rectangle((3.3 + j + 0.04, y + 0.08), 0.92, 0.84, fc=fc, ec="white", lw=1.5))
            ax.text(3.3 + j + 0.5, y + 0.5, txt, ha="center", va="center", fontsize=7.9 if len(txt) > 6 else 8.6, color=ec,
                    fontweight="bold" if txt not in ("—", "통과") else "normal")
    ax.text(0.05, len(tab) + 1.75, "①~③: io_day21.py로 실제 실행 (엄격 파서) · ④: 사람 검토 결과(예시) · — = 앞 단계 실패로 미수행",
            fontsize=8.4, color=MUTED, va="top")
    save(fig, 21, "validation_grid")


# ------------------------------------------------------------ 6. 재시도 판단
def fig_retry():
    fig, ax = canvas(12, 4.4, 12, 4.4)
    rbox(ax, 0.2, 1.7, 1.9, 1.0, "응답 받음\nattempt 1", fc="#fff0f6", ec=C["pink"], fs=9.3)
    rbox(ax, 2.7, 1.7, 2.0, 1.0, "validate\n자동 검사 3단계", fc="#e7f5ff", ec=C["blue"], fs=9.3)
    arrow(ax, 2.1, 2.2, 2.7, 2.2)
    rbox(ax, 5.6, 3.2, 3.0, 0.95, "통과 (정상 답 · 정상 정보 부족)\n→ 생성 종료 · 근거 검토로", fc="#ebfbee", ec=C["green"], fs=8.9)
    rbox(ax, 5.6, 1.75, 3.0, 0.95, "json_error · schema_error\n→ 수정 요청 1회 (attempt 2)", fc="#fff9db", ec=C["yellow"], fs=8.9)
    rbox(ax, 5.6, 0.3, 3.0, 0.95, "출처·상태 규칙 위반\n→ 재생성 없이 중단", fc="#fff5f5", ec=C["red"], fs=8.9)
    arrow(ax, 4.7, 2.45, 5.6, 3.65); arrow(ax, 4.7, 2.2, 5.6, 2.2); arrow(ax, 4.7, 1.95, 5.6, 0.8)
    rbox(ax, 9.3, 2.35, 2.5, 1.05, "다시 검사\n통과 → 근거 검토", fc="#ebfbee", ec=C["green"], fs=8.9)
    rbox(ax, 9.3, 0.9, 2.5, 1.05, "또 실패 → 중단\nreturn None, history", fc="#fff5f5", ec=C["red"], fs=8.9)
    arrow(ax, 8.6, 2.3, 9.3, 2.85); arrow(ax, 8.6, 2.1, 9.3, 1.45)
    ax.text(6.0, 4.3, "최초 응답 1회 + 형식 수정 1회 = 최대 2회 · 모든 시도를 history에 남김", ha="center", fontsize=9.6,
            fontweight="bold", color=C["ink"])
    save(fig, 21, "retry")


# ------------------------------------------------------------ 7. A/B × D/H
def fig_axes():
    fig, ax = canvas(11, 4.4, 11, 4.4)
    ax.text(5.5, 4.1, "두 축은 다른 질문에 답해요", ha="center", fontsize=10.5, fontweight="bold", color=C["ink"])
    ax.text(3.45, 3.55, "Prompt A (기준안)", ha="center", fontsize=9.6, color=C["purple"], fontweight="bold")
    ax.text(7.45, 3.55, "Prompt B (A + 근거 부족 규칙 1개)", ha="center", fontsize=9.6, color=C["pink"], fontweight="bold")
    ax.text(0.85, 2.55, "D 묶음\n개선용\nD01~D03", ha="center", va="center", fontsize=9.2, color=C["ink"])
    ax.text(0.85, 1.05, "H 묶음\n최종 확인용\nH01~H03", ha="center", va="center", fontsize=9.2, color=C["ink"])
    rbox(ax, 1.6, 1.95, 3.7, 1.2, "A로 D 질문 실행\n실패를 보고 B 가설을 세움", fc="#f3f0ff", ec=C["purple"], fs=9)
    rbox(ax, 5.6, 1.95, 3.7, 1.2, "B로 같은 D 질문 실행\n어떤 오류가 줄고 생겼나", fc="#fff0f6", ec=C["pink"], fs=9)
    rbox(ax, 1.6, 0.45, 3.7, 1.2, "버전 고정 후 H 실행\n(고치는 데 안 쓴 질문)", fc="#f8f9fa", ec=C["gray"], fs=9)
    rbox(ax, 5.6, 0.45, 3.7, 1.2, "버전 고정 후 H 실행\nH 보고 또 고치면 → 새 H 필요", fc="#f8f9fa", ec=C["gray"], fs=9)
    ax.text(10.6, 2.55, "← 고정:\n질문·문맥·\n모델·계약·\n예제·채점", ha="right", va="center", fontsize=8.4, color=MUTED)
    save(fig, 21, "ab_dh")


if __name__ == "__main__":
    for f in (fig_flow, fig_six, fig_context, fig_budget, fig_validation, fig_retry, fig_axes):
        f()
    print("done")
