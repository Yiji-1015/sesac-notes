import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day22 그림 만들기 (LLM API 클라이언트).

숫자가 들어간 그림(상태 층 표 · 스트림 표 · SDK 재시도 막대 · 전송 흐름 · 대기 곡선 · 비용 막대 · 캐시 판정)은
tools/io_day22.py 를 import 해서 강의 코드를 openai SDK + 가짜 서버(httpx.MockTransport)로 실제 실행한 결과다.
실제 OpenAI 서버에는 요청하지 않았다. 비용 단가·사용량은 교안의 가상값이다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day22.py
"""
import contextlib
import io as _io

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
with contextlib.redirect_stdout(_io.StringIO()):
    import io_day22 as D  # noqa: E402

MUTED = "#5b5270"
R = D.RESULTS


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


def cell(ax, x, y, w, h, text, fc, ec="#ffffff", fs=8.6, color=None, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color or C["ink"],
            fontweight="bold" if bold else "normal", linespacing=1.25)


OKC, BADC, SKIPC, WARN, HUMAN = "#d3f9d8", "#ffe3e3", "#f1f3f5", "#fff3bf", "#fff4e6"


# ------------------------------------------------------------ 1. 전체 흐름
def fig_flow():
    fig, ax = canvas(12.6, 4.4, 12.6, 4.4)
    xs = [0.2, 2.75, 5.3, 7.85, 10.4]
    items = [("질문 · 선택 지식\n+ 고정 Prompt\n(11/16 그대로)", "#f8f9fa", C["gray"]),
             ("LLM Client\n인증 · 모델 선택\n캐시 · 전송 · 재시도", "#f3f0ff", C["purple"]),
             ("외부 LLM API\nResponses\n(Streaming 가능)", "#fff0f6", C["pink"]),
             ("응답 검증\n완료 판정 → JSON·Schema\n→ 출처·상태 규칙", "#e7f5ff", C["blue"]),
             ("기록 · 근거 검토\nusage · 시간 · 비용\nsource_mode · 사람", "#fff4e6", C["orange"])]
    for x, (t, fc, ec) in zip(xs, items):
        rbox(ax, x, 2.0, 2.05, 1.5, t, fc=fc, ec=ec, fs=9.1)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + 2.05, 2.75, b, 2.75)
    ax.text(6.3, 4.05, "오늘 바뀌는 건 '응답을 받는 방법' — 사람이 복사하던 응답 → Python이 API로 수신", ha="center",
            fontsize=10.2, fontweight="bold", color=C["ink"])
    ax.text(3.78, 3.65, "오늘 새로: ③ Streaming ④ 재시도 ⑤ 측정 ⑥ 캐시 ⑦ 라우팅", ha="center", fontsize=8.6, color=C["purple"])
    rbox(ax, 2.75, 0.3, 7.15, 0.95, "검색(RAG) · 도구 실행(Agent)은 이후 — 모델의 JSON 응답은 도구를 실행해도 된다는 권한이 아님",
         fc="#fff9db", ec=C["yellow"], fs=9, ls="--")
    save(fig, 22, "flow")


# ------------------------------------------------------------ 2. 세 층의 상태 + 8 사례
def fig_layers():
    g = R["gate"]
    fig, ax = canvas(12.4, 5.6, 12.4, 5.6)
    ax.text(0.1, 5.3, "같은 'status'라도 세 층 — 교안 GATE_CASES 8개를 SDK Response로 만들어 실제로 판정", fontsize=10.2,
            fontweight="bold", color=C["ink"])
    cols = ["사례", "HTTP", "response.status", "completed_text\n오류", "run_status\n(앱 실행 결과)", "answer.status\n(업무)", "근거"]
    ws = [2.9, 0.9, 1.6, 1.7, 2.0, 1.8, 1.3]
    xs = [0.1]
    for w in ws[:-1]:
        xs.append(xs[-1] + w)
    top = 4.55
    for x, w, t in zip(xs, ws, cols):
        cell(ax, x, top, w, 0.6, t, "#ede7ff", fs=8.6, bold=True)
    h = 0.5
    for i, r in enumerate(g):
        y = top - (i + 1) * h
        cell(ax, xs[0], y, ws[0], h, r["label"], "#fafafa", fs=8.5)
        cell(ax, xs[1], y, ws[1], h, "200", OKC, fs=8.5)
        cell(ax, xs[2], y, ws[2], h, r["status"], OKC if r["status"] == "completed" else BADC, fs=8.5)
        cell(ax, xs[3], y, ws[3], h, "None" if r["gate"] is None else r["gate"], OKC if r["gate"] is None else BADC, fs=8.5)
        cell(ax, xs[4], y, ws[4], h, r["run_status"], OKC if r["run_status"] == "validated" else BADC, fs=8.5)
        cell(ax, xs[5], y, ws[5], h, r["ans_status"] or "—", OKC if r["ans_status"] else SKIPC, fs=8.3)
        hum = ("사람: 잘못된 인용" if r["key"] == "false_evidence" else "사람 검토 대기") if r["run_status"] == "validated" else "—"
        cell(ax, xs[6], y, ws[6], h, hum, BADC if r["key"] == "false_evidence" else (HUMAN if hum != "—" else SKIPC), fs=7.8)
    ax.text(0.1, 0.12, "HTTP 200 = 통신 완료 · response.status = 생성 완료 · answer.status = 업무 답변 상태. "
            "정보 부족(insufficient_context)은 정상 실행, '잘못된 인용'은 세 층을 다 통과해도 사람이 걸러요.",
            fontsize=8.6, color=MUTED)
    save(fig, 22, "status_layers")


# ------------------------------------------------------------ 3. 스트림 시나리오
def fig_stream():
    s = R["stream"]
    fig, ax = canvas(12.6, 5.0, 12.6, 5.0)
    ax.text(0.1, 4.72, "Streaming — SSE 바이트를 가짜 서버가 보내고, openai SDK가 해석한 이벤트 순서 (실행 결과)", fontsize=10.2,
            fontweight="bold", color=C["ink"])
    colors = {"response.created": ("#e7f5ff", "시작"), "response.output_text.delta": ("#f3f0ff", "Δ"),
              "response.output_text.done": ("#e6fcf5", "text\ndone"), "response.completed": ("#d3f9d8", "완료"),
              "response.incomplete": ("#fff3bf", "미완성"), "response.refusal.delta": ("#ffe3e3", "거절"),
              "error": ("#ffe3e3", "error")}
    h = 0.55
    for i, r in enumerate(s):
        y = 4.05 - i * h
        ax.text(0.1, y + h / 2 - 0.03, r["label"], fontsize=8.8, va="center", color=C["ink"])
        x = 3.2
        for t in r["seen"]:
            fc, lab = colors.get(t, ("#f1f3f5", t))
            ww = 0.42 if t == "response.output_text.delta" else 0.66
            cell(ax, x, y + 0.05, ww, h - 0.1, lab, fc, fs=7.2)
            x += ww + 0.04
        if r["exc"]:
            cell(ax, x, y + 0.05, 0.9, h - 0.1, "끊김\n" + r["exc"][:13], "#ffe3e3", fs=6.6)
            x += 0.94
        out = r["outcome"]
        fc = OKC if out == "validate 로" else BADC
        cell(ax, 9.2, y + 0.05, 1.6, h - 0.1, ("→ 검증기로" if out == "validate 로" else out), fc, fs=8.2, bold=True)
        cell(ax, 10.85, y + 0.05, 0.85, h - 0.1, f"Δ {r['parts']}개", "#fafafa", fs=7.8)
        cell(ax, 11.75, y + 0.05, 0.8, h - 0.1, "usage\n" + ("67" if r["usage_total"] else "None"), OKC if r["usage_total"] else SKIPC, fs=7.2)
    ax.text(9.2 + 0.8, 4.65 - 0.05, "결과", ha="center", fontsize=8.6, color=MUTED)
    ax.text(0.1, 0.1, "조각(Δ)이 보여도 완료 이벤트가 없으면 interrupted. 빈 Δ는 첫 텍스트 시간에 안 셈. "
            "usage는 최종 Response에서만 — 끊기면 None (0 아님).", fontsize=8.6, color=MUTED)
    save(fig, 22, "stream_cases")


# ------------------------------------------------------------ 4. SDK 재시도 막대
def fig_sdk_retry():
    e = [x for x in R["errors"] if "sends0" in x]
    to = {x["label"]: x["sends"] for x in R["errors"] if x["label"].startswith("timeout")}
    labels = [x["label"].replace(" ", "\n", 1) for x in e] + ["timeout"]
    s0 = [x["sends0"] for x in e] + [to["timeout mr=0"]]
    s2 = [x["sends2"] for x in e] + [to["timeout mr=2"]]
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    import numpy as np
    xx = np.arange(len(labels))
    b1 = ax.bar(xx - 0.2, s0, 0.38, color=C["purple"], label="max_retries=0 (수업 설정)")
    b2 = ax.bar(xx + 0.2, s2, 0.38, color=C["pink"], label="max_retries 기본값 2")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{int(b.get_height())}", ha="center", fontsize=9.5)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("가짜 서버가 받은 실제 전송 수"); ax.set_ylim(0, 3.7); ax.set_yticks([0, 1, 2, 3])
    ax.set_title("같은 오류, SDK가 몰래 보내는 횟수 — openai SDK + 가짜 서버 실행 결과")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.annotate("할당량 부족도\nSDK는 3번 보냄", xy=(4 + 0.2, 3), xytext=(4.5, 3.35), fontsize=8.6, color=C["red"],
                ha="center")
    save(fig, 22, "sdk_retries")


# ------------------------------------------------------------ 5. 전송 흐름 (bounded)
def fig_bounded():
    b = R["bounded"]
    fig, ax = canvas(12.6, 5.2, 12.6, 5.2)
    ax.text(0.1, 4.9, "생성 × 전송 — 각본대로 응답하는 가짜 서버로 send_bounded 실행 (생성 ≤ 2 · 생성당 전송 ≤ 2 · 전체 ≤ 4)",
            fontsize=10, fontweight="bold", color=C["ink"])
    h = 0.66
    for i, r in enumerate(b):
        y = 4.1 - i * h
        ax.text(0.1, y + h / 2 - 0.05, r["label"], fontsize=8.8, va="center", color=C["ink"])
        x = 3.35
        for g, s, m in r["log"]:
            if s == "-":
                ok = "passed" in m
                cell(ax, x, y + 0.06, 1.15, h - 0.14, ("검증 통과\n→ 사람 검토" if ok else m.replace("검증 → ", "검증 실패\n")),
                     OKC if ok else WARN, fs=7.2)
                x += 1.19
                continue
            short = ("429 대기" if "rate_limit" in m else "429 할당량\n중단" if "quota" in m else "401\n중단" if "Auth" in m
                     else "500" if "500" in m else "수신" if "수신" in m else m[:8])
            fc = OKC if short == "수신" else (WARN if short in ("429 대기", "500") else BADC)
            cell(ax, x, y + 0.06, 0.98, h - 0.14, f"생성{g}·전송{s}\n{short}", fc, fs=7.0)
            x += 1.02
        cell(ax, 11.0, y + 0.06, 1.55, h - 0.14, f"{r['final']}\n전송 {r['sends']}회", OKC if r["final"] == "validated" else BADC, fs=7.6, bold=True)
    ax.text(0.1, 0.15, "429 rate_limit은 Retry-After를 지키며 한 번 더, 할당량·인증은 1회로 중단, 형식 오류는 새 생성(수정 요청) 1회까지. "
            "앞선 실패도 모두 기록에 남겨요.", fontsize=8.5, color=MUTED)
    save(fig, 22, "bounded")


# ------------------------------------------------------------ 6. 대기 곡선
def fig_backoff():
    import numpy as np
    att = np.arange(1, 7)
    main_lo = [D.wait_main(a, None, 1e9, 1, 0.0) for a in att]
    main_hi = [D.wait_main(a, None, 1e9, 1, 0.25) for a in att]
    app_lo = [D.retry_delay(a, None, 1e9, 0.0) for a in att]
    app_hi = [D.retry_delay(a, None, 1e9, 0.5) for a in att]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.fill_between(att, main_lo, main_hi, color=C["purple"], alpha=0.25)
    ax.plot(att, main_lo, "o-", color=C["purple"], label="본문 wait-main: min(8, 2^(n-1)) + 지터 0~0.25")
    ax.fill_between(att, app_lo, app_hi, color=C["pink"], alpha=0.2)
    ax.plot(att, app_lo, "s--", color=C["pink"], label="부록 retry_delay: 2^(n-1) + 지터 0~0.5 (상한 없음)")
    for a, v in zip(att, app_lo):
        ax.text(a, v + 1.2, f"{v:g}", ha="center", fontsize=8.5, color=C["pink"])
    for a, v in zip(att, main_lo):
        ax.text(a + 0.12, v - 2.2, f"{v:g}", ha="center", fontsize=8.5, color=C["purple"])
    ax.set_xlabel("attempt (첫 실패 = 1)"); ax.set_ylabel("대기 초 (Retry-After 없음)")
    ax.set_title("Exponential backoff — 실패가 반복될수록 대기를 늘림 (강의 두 함수 실제 계산)")
    ax.set_ylim(-3, 37); ax.legend(fontsize=8.6, loc="upper left")
    save(fig, 22, "backoff")


# ------------------------------------------------------------ 7. 비용 막대
def fig_cost():
    cst = R["cost"]
    fig, ax = plt.subplots(figsize=(9.5, 2.9))
    left = 0
    cols = [C["blue"], C["mint"], C["orange"]]
    for (k, v), col in zip(cst.items(), cols):
        ax.barh([1], [v * 1e6], left=left, color=col, height=0.55)
        if v * 1e6 < 100:   # 좁은 칸은 막대 아래에
            ax.text(left + v * 1e6 / 2, 0.66, f"{k} ${v:.5f}", ha="center", va="top", fontsize=8.4, color=col, fontweight="bold")
        else:
            ax.text(left + v * 1e6 / 2, 1, f"{k}\n${v:.5f}".rstrip("0"), ha="center", va="center", fontsize=8.6, color="white", fontweight="bold")
        left += v * 1e6
    ax.text(left + 15, 1, f"= ${R['cost_total']:.5f}", va="center", fontsize=10, fontweight="bold", color=C["ink"])
    w = R["cost_wrong"] * 1e6
    ax.barh([0], [w], color=C["gray"], height=0.55, alpha=0.6)
    ax.text(w + 15, 0, f"= ${R['cost_wrong']:.5f}  ← 캐시 400을 일반 입력에도 포함", va="center", fontsize=9, color=C["red"])
    ax.set_yticks([1, 0]); ax.set_yticklabels(["올바른 계산\n(1000-400)·0.75 + 400·0.075 + 200·4.5", "잘못된 계산\n1000·0.75 + 400·0.075 + 200·4.5"], fontsize=8.6)
    ax.set_xlim(0, 2400); ax.set_xlabel("× 1/1,000,000 USD  (교안 가상 단가 · 입력 1000 · 캐시 400 · 출력 200)")
    ax.grid(axis="y", visible=False)
    ax.set_title("비용 추정 — 출력 200 토큰이 전체의 65%")
    save(fig, 22, "cost")


# ------------------------------------------------------------ 8. 캐시 세 종류 + 판정
def fig_cache():
    fig, ax = canvas(12.6, 5.6, 12.6, 5.6)
    ax.text(0.1, 5.3, "'캐시'는 세 종류 — 재사용하는 대상이 달라요", fontsize=10.4, fontweight="bold", color=C["ink"])
    rbox(ax, 0.1, 3.4, 3.9, 1.6, "Response Cache (우리 앱)\n승인된 최종 답변을 재사용\n유효한 hit → 새 API 전송 0회", fc="#e6fcf5", ec=C["mint"], fs=9.1)
    rbox(ax, 4.35, 3.4, 3.9, 1.6, "Prompt Caching (API 제공사)\n같은 입력 앞부분 계산 재사용\n요청·새 출력 생성은 계속\n→ usage.cached_tokens", fc="#f3f0ff", ec=C["purple"], fs=9.1)
    rbox(ax, 8.6, 3.4, 3.9, 1.6, "KV Cache (모델 내부)\n앞 토큰의 Key·Value 재사용\n다음 토큰 계산을 빠르게\n답변 저장소 아님", fc="#fff0f6", ec=C["pink"], fs=9.1)
    ax.text(0.1, 2.85, "read_cache 실제 실행 (TTL 60초 · 가상 시간)", fontsize=9.8, fontweight="bold", color=C["ink"])
    for i, (cond, res) in enumerate(R["cache"]):
        y = 2.25 - i * 0.48
        hit = not res.startswith("None")
        cell(ax, 0.1, y, 6.6, 0.44, cond, "#fafafa", fs=8.6)
        cell(ax, 6.75, y, 2.4, 0.44, "hit · 새 전송 0회" if hit else "miss", OKC if hit else BADC, fs=8.6, bold=True)
    ax.text(9.4, 1.6, "key = SHA-256(scope · model ·\nrequest · format · validator)\n\n같은 ID라도 본문이 바뀌면\n다른 key → miss",
            fontsize=8.8, color=MUTED, va="center")
    save(fig, 22, "cache")


if __name__ == "__main__":
    for f in (fig_flow, fig_layers, fig_stream, fig_sdk_retry, fig_bounded, fig_backoff, fig_cost, fig_cache):
        f()
    print("done")
