import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day17 그림 만들기 (FastAPI 지식 관리 API · CRUD · JSON 파일 저장).

숫자가 들어간 그림(시나리오 상태, Lock 실험)은 강의 코드를 그대로 돌려서 얻은 값이다.
- 시나리오 값: 강의 main.py(GET·POST·PUT·DELETE)를 FastAPI 0.162 TestClient로 실행해 확인
- Lock 실험: 이 스크립트가 직접 스레드 20개로 '읽기 → 추가 → 저장'을 동시에 돌려 센 결과
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day17.py
"""
import json
import tempfile
import threading
import time
from pathlib import Path

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=10, bold=False, color=None, ls="-", lw=1.8):
    ec = ec or C["purple"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color or C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, ls="-", text=None, tfs=9, toff=(0, 0.12)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color or MUTED, lw=lw, ls=ls))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="bottom",
                fontsize=tfs, color=color or MUTED)


# ------------------------------------------------------------ 1. 요청 한 건의 경로
def fig_request_flow():
    fig, ax = canvas(10.5, 4.4, 12.6, 5.4)
    steps = [
        ("클라이언트\n(/docs)", "GET /api/knowledge/2"),
        ("Uvicorn", "요청 받기\n앱 실행"),
        ("Route", "Method + Path\n맞는 함수 찾기"),
        ("타입 검증", "\"2\" → int 2\n바꿀 수 있나?"),
        ("처리 함수", "get_knowledge(2)\nid 같은 항목 찾기"),
        ("storage.py", "knowledge.json\n읽기"),
    ]
    xs = [0.2 + i * 2.08 for i in range(6)]
    for i, (name, sub) in enumerate(steps):
        rbox(ax, xs[i], 3.2, 1.8, 1.6, f"{name}\n\n{sub}", fc=C["soft"] if i else "#fff", fs=9)
        ax.text(xs[i] + 0.12, 4.62, f"{i + 1}", fontsize=10, fontweight="bold", color=C["purple"])
        if i < 5:
            arrow(ax, xs[i] + 1.8, 4.0, xs[i + 1], 4.0, color=C["purple"])
    # 결과 세 갈래
    rbox(ax, 8.5, 0.35, 3.9, 1.05, "200 + 지식 항목 JSON\n/api/knowledge/2", fc="#e6fcf5", ec=C["mint"], fs=9.5, bold=True)
    rbox(ax, 4.4, 0.35, 3.7, 1.05, "404 Knowledge not found\n/api/knowledge/9999", fc="#fff4e6", ec=C["orange"], fs=9.5, bold=True)
    rbox(ax, 0.3, 0.35, 3.7, 1.05, "422 int_parsing\n/api/knowledge/foo", fc="#fff0f6", ec=C["pink"], fs=9.5, bold=True)
    arrow(ax, xs[5] + 0.9, 3.2, 10.45, 1.4, color=C["mint"], text="찾음", toff=(0.45, 0))
    arrow(ax, xs[4] + 0.9, 3.2, 6.25, 1.4, color=C["orange"], text="끝까지 없음 → raise", toff=(1.85, -0.25))
    arrow(ax, xs[3] + 0.9, 3.2, 2.15, 1.4, color=C["pink"], text="숫자로 못 바꿈\n함수는 실행 안 됨", toff=(-1.75, -0.1))
    ax.set_title("GET 요청 한 건이 지나가는 길 — 어디서 멈추느냐가 상태 코드를 정한다", loc="left")
    save(fig, 17, "request_flow")


# ------------------------------------------------------------ 2. 플랫폼 안에서 오늘 만드는 부분
def fig_platform():
    fig, ax = canvas(9.5, 4.6, 10, 5.6)
    rbox(ax, 0.5, 4.55, 9, 0.75, "사용자 / 사내 업무 화면   (오늘은 /docs가 임시 클라이언트)", fc="#fff", fs=10)
    rbox(ax, 0.5, 3.35, 9, 0.8, "FastAPI — 서비스 입구: 요청 받기 · 입력 검증 · 함수 연결 · 응답", fc=C["soft"], fs=10.5, bold=True)
    cols = [("지식 관리 API\nCRUD · 원본 관리", "JSON 파일\n(오늘의 저장소)", True),
            ("RAG\n근거 검색 · LLM 답변", "Vector DB · 모델", False),
            ("AI Agent\n도구 선택 · 결과 확인", "Tool Registry", False)]
    for i, (top, bot, today) in enumerate(cols):
        x = 0.5 + i * 3.1
        ec = C["mint"] if today else C["gray"]
        fc = "#e6fcf5" if today else "#f8f9fa"
        ls = "-" if today else "--"
        rbox(ax, x, 1.75, 2.8, 1.15, top, fc=fc, ec=ec, ls=ls, fs=10, bold=today)
        rbox(ax, x, 0.3, 2.8, 0.95, bot, fc=fc, ec=ec, ls=ls, fs=9.5)
        arrow(ax, x + 1.4, 3.35, x + 1.4, 2.9, color=ec)
        arrow(ax, x + 1.4, 1.75, x + 1.4, 1.25, color=ec)
        ax.text(x + 2.72, 2.78, "오늘" if today else "이후", ha="right", va="top", fontsize=9,
                fontweight="bold", color=ec)
    arrow(ax, 5, 4.55, 5, 4.15)
    ax.set_title("AI 플랫폼 안에서 오늘 만드는 부분 (초록 실선) — RAG·Agent는 나중에 붙인다", loc="left")
    save(fig, 17, "platform_map")


# ------------------------------------------------------------ 3. URL 해부
def fig_url():
    fig, ax = canvas(10, 2.4, 10, 2.6)
    parts = [("http://", "통신 방식", C["gray"], 1.0),
             ("127.0.0.1", "내 컴퓨터\n(localhost)", C["blue"], 1.7),
             (":8000", "포트\n(서버 문 번호)", C["orange"], 1.0),
             ("/api/knowledge", "자원 경로\n(무엇을)", C["purple"], 2.5),
             ("/2", "Path 값\nknowledge_id", C["pink"], 0.8),
             ("?category=api", "Query\n(선택 조건)", C["mint"], 2.2)]
    x = 0.2
    for txt, lab, col, w in parts:
        rbox(ax, x, 1.35, w - 0.08, 0.75, txt, fc="#fff", ec=col, fs=11, bold=True, color=col)
        ax.text(x + (w - 0.08) / 2, 1.05, lab, ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.3)
        x += w
    ax.set_title("URL 하나에 들어 있는 정보 — Path는 '어느 것', Query는 '어떤 조건'", loc="left")
    save(fig, 17, "url_anatomy")


# ------------------------------------------------------------ 4. ID 정책: len+1 vs next_id
def fig_id_policy():
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.4))
    rows = [
        ("len(entries) + 1 로 번호 매기기", [[1, 2, 3], [1, 3], [1, 3, 3]], ["처음", "ID 2 삭제", "새로 등록"], True),
        ("next_id 를 따로 저장 (강의 방식)", [[1, 2, 3], [1, 3], [1, 3, 4]], ["처음 · next_id=4", "ID 2 삭제 · next_id=4 그대로", "새로 등록 → 4 · next_id=5"], False),
    ]
    for ax, (title, states, caps, bad) in zip(axes, rows):
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 2)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=11.5, color=C["red"] if bad else C["green"])
        for s, (ids, cap) in enumerate(zip(states, caps)):
            x0 = 0.3 + s * 4.0
            for k, v in enumerate(ids):
                dup = bad and s == 2 and v == 3
                new = s == 2 and k == len(ids) - 1
                fc = "#ffe3e3" if dup else ("#e6fcf5" if new else C["soft"])
                ec = C["red"] if dup else (C["mint"] if new else C["purple"])
                rbox(ax, x0 + k * 0.95, 0.75, 0.8, 0.8, str(v), fc=fc, ec=ec, fs=13, bold=True)
            ax.text(x0, 0.45, cap if not (bad and s == 2) else "새로 등록 → len 2 + 1 = 3 → 충돌!", fontsize=9,
                    color=C["red"] if (bad and s == 2) else MUTED, va="top")
            if s < 2:
                ax.annotate("", xy=(x0 + 3.75, 1.15), xytext=(x0 + 3.05, 1.15),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))
    save(fig, 17, "id_policy")


# ------------------------------------------------------------ 5. index vs id
def fig_index_vs_id():
    fig, ax = canvas(8.6, 2.9, 9, 3.0)
    ids = [1, 3, 4]
    for k, v in enumerate(ids):
        x = 1.6 + k * 1.6
        rbox(ax, x, 1.1, 1.3, 1.1, f"id {v}", fc=C["soft"], fs=13, bold=True)
        ax.text(x + 0.65, 0.85, f"index {k}", ha="center", va="top", fontsize=10.5, color=C["orange"], fontweight="bold")
    ax.text(0.15, 1.65, "entries", fontsize=11, va="center", color=MUTED)
    ax.text(6.6, 2.0, "id 4를 찾으려고 entries[4]\n→ IndexError (위치는 0~2뿐)", fontsize=10, color=C["red"], va="center")
    ax.text(6.6, 1.15, "id 1을 찾으려고 entries[1]\n→ id 3 항목이 나옴 (엉뚱한 지식)", fontsize=10, color=C["red"], va="center")
    ax.set_title("ID 2를 지운 뒤의 목록 — id(이름표)와 index(줄 선 순서)는 다른 숫자", loc="left")
    save(fig, 17, "index_vs_id")


# ------------------------------------------------------------ 6. 끝까지 추적 시나리오 (실제 실행 값)
def fig_scenario():
    # 강의 main.py를 TestClient로 돌려 확인한 값 (초기 3건 → POST → PUT → 재시작 → DELETE → 재시작)
    steps = ["① 시작", "② POST\n(id 4 받음)", "③ PUT 4\n본문 수정", "④ 서버\n재시작", "⑤ DELETE 4", "⑥ 다시\n재시작"]
    count = [3, 4, 4, 4, 3, 3]
    next_id = [4, 5, 5, 5, 5, 5]
    status = ["", "201", "200", "GET 4 → 200\n(수정본 그대로)", "204\nGET 4 → 404", "GET 4 → 404"]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    x = range(len(steps))
    ax.step(x, count, where="mid", color=C["purple"], lw=2.6, label="저장된 지식 항목 수")
    ax.step(x, next_id, where="mid", color=C["orange"], lw=2.6, ls="--", label="next_id (다음에 줄 번호)")
    for i in x:
        ax.plot(i, count[i], "o", color=C["purple"], ms=8)
        ax.plot(i, next_id[i], "s", color=C["orange"], ms=7)
        if status[i]:
            ax.text(i, 2.35, status[i], ha="center", va="top", fontsize=8.8, color=C["green"] if "404" not in status[i] else C["red"])
    ax.set_xticks(list(x), steps)
    ax.set_ylim(1.3, 6)
    ax.set_yticks([3, 4, 5])
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("지식 항목 하나를 끝까지 추적 — DELETE 후에도 next_id는 5로 남는다", loc="left")
    save(fig, 17, "scenario_state")


# ------------------------------------------------------------ 7. Lock 실험 (실제로 돌려서 센다)
def lock_experiment(n=20, use_lock=True, gap=0.01):
    tmp = Path(tempfile.mkdtemp()) / "knowledge.json"
    tmp.write_text(json.dumps({"next_id": 1, "entries": []}), encoding="utf-8")
    lock = threading.Lock()

    def create():
        def body():
            data = json.loads(tmp.read_text(encoding="utf-8"))      # 읽기
            time.sleep(gap)                                         # 처리 시간(틈을 일부러 넓힘)
            data["entries"].append({"id": data["next_id"]})         # 추가
            data["next_id"] += 1
            tmp.write_text(json.dumps(data), encoding="utf-8")      # 저장
        if use_lock:
            with lock:
                body()
        else:
            body()

    ts = [threading.Thread(target=create) for _ in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    d = json.loads(tmp.read_text(encoding="utf-8"))
    return len(d["entries"]), len({e["id"] for e in d["entries"]})


def fig_lock():
    n = 20
    runs_no = [lock_experiment(n, False) for _ in range(5)]
    runs_yes = [lock_experiment(n, True) for _ in range(5)]
    no = sorted(r[0] for r in runs_no)[2]
    yes = sorted(r[0] for r in runs_yes)[2]
    print("lock experiment (no lock, with lock):", runs_no, runs_yes)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.barh(["Lock 없이", "with LOCK:"], [no, yes], color=[C["pink"], C["mint"]], height=0.55)
    ax.axvline(n, color=C["ink"], ls="--", lw=1.4)
    ax.text(n - 0.3, 0.5, f"보낸 POST {n}건", ha="right", va="center", fontsize=9.5, color=C["ink"])
    for b, v in zip(bars, [no, yes]):
        ax.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v}건 저장", va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, n + 4)
    ax.set_xlabel("파일에 실제로 남은 지식 항목 수 (5번 반복 중 가운데 값)")
    ax.set_title("동시에 20건 등록 — 잠금이 없으면 서로의 저장을 덮어쓴다", loc="left")
    save(fig, 17, "lock_race")
    return no, yes


# ------------------------------------------------------------ 8. 임시 파일에 쓰고 교체
def fig_atomic_write():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, title, ok in [(axes[0], "바로 덮어쓰기: open(DATA, \"w\")", False),
                          (axes[1], "임시 파일에 쓰고 교체: .tmp → replace()", True)]:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 3.4)
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=11, color=C["green"] if ok else C["red"])
        if not ok:
            rbox(ax, 0.2, 1.7, 2.0, 1.0, "knowledge.json\n{\"next_id\": 5,\n \"entries\": [{\"id\"", fc="#ffe3e3", ec=C["red"], fs=8.8)
            ax.text(2.5, 2.2, "쓰는 도중\n오류·강제 종료", fontsize=10, color=C["red"], va="center")
            ax.text(0.2, 1.1, "원본이 이미 지워진 채 반쯤만 써짐\n→ 다음 실행 때 json.load()가 실패\n   (JSON decode error)", fontsize=9.5,
                    color=C["ink"], va="top")
        else:
            rbox(ax, 0.2, 1.7, 1.8, 1.0, "knowledge.json\n(원본 그대로)", fc="#e6fcf5", ec=C["mint"], fs=9.5)
            rbox(ax, 2.9, 1.7, 1.8, 1.0, "knowledge.tmp\n(여기에 먼저 씀)", fc=C["soft"], ec=C["purple"], fs=9.5)
            ax.annotate("", xy=(2.0, 2.2), xytext=(2.9, 2.2),
                        arrowprops=dict(arrowstyle="-|>", color=C["purple"], lw=2))
            ax.text(2.45, 2.75, "다 쓰면\nreplace()", ha="center", fontsize=9, color=C["purple"])
            ax.text(0.2, 1.1, "중간에 실패하면 .tmp만 망가지고 원본은 멀쩡\n→ 성공하면 한 번에 바꿔치기\n   finally에서 남은 .tmp 정리", fontsize=9.5,
                    color=C["ink"], va="top")
    save(fig, 17, "atomic_write")


# ------------------------------------------------------------ 9. 심화 구조: 역할별 파일
def fig_layers():
    fig, ax = canvas(10, 3.6, 12, 4.2)
    items = [("app/main.py", "앱 만들고\nRouter 등록", C["blue"]),
             ("routers/knowledge.py", "HTTP 담당\nMethod·Path·상태 코드", C["purple"]),
             ("repositories/\njson_knowledge.py", "저장 담당\n읽기·변경·저장·LOCK", C["orange"]),
             ("data/knowledge.json", "실제 데이터", C["mint"])]
    for i, (name, role, col) in enumerate(items):
        x = 0.2 + i * 3.0
        rbox(ax, x, 1.9, 2.5, 1.6, f"{name}\n\n{role}", fc="#fff", ec=col, fs=9.5)
        if i < 3:
            arrow(ax, x + 2.5, 2.7, x + 3.0, 2.7, color=col)
    rbox(ax, 3.2, 0.25, 5.5, 0.95, "app/schemas.py — 데이터 모양 (KnowledgeWrite · Knowledge)\nRouter가 입력 검증·응답 모양에 사용", fc=C["soft"], ec=C["pink"], fs=9.3)
    arrow(ax, 4.45, 1.2, 4.45, 1.9, color=C["pink"])
    ax.set_title("심화 구조 — 바뀌는 이유가 다른 코드를 다른 파일에 둔다", loc="left")
    save(fig, 17, "layers")


if __name__ == "__main__":
    fig_request_flow()
    fig_platform()
    fig_url()
    fig_id_policy()
    fig_index_vs_id()
    fig_scenario()
    print("lock:", fig_lock())
    fig_atomic_write()
    fig_layers()
    print("done")
