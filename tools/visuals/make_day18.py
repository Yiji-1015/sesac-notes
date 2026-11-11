import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day18 그림 만들기 (PostgreSQL 입문 · 지식 관리 DB).

표·결과 값은 강의 S0 데이터를 SQLite로 실제 실행해 얻은 것(tools/io_day18.py)과 같다.
ID 빈 번호 그림만 PostgreSQL 공식 문서(시퀀스 nextval은 ROLLBACK돼도 되돌아가지 않음)를 그린 개념도다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day18.py
"""
import sqlite3

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=10, bold=False, color=None, ls="-", lw=1.8, ha="center"):
    ec = ec or C["purple"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=lw, ls=ls))
    tx = x + w / 2 if ha == "center" else x + 0.15
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=color or C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.8, ls="-", text=None, tfs=9, toff=(0, 0.1), rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color or MUTED,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="bottom", fontsize=tfs,
                color=color or MUTED)


# ------------------------------------------------------------ S0를 실제로 만들어 값 읽기
def s0():
    con = sqlite3.connect(":memory:")
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("CREATE TABLE categories (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE)")
    con.execute("CREATE TABLE knowledge (id INTEGER PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL, "
                "category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE RESTRICT)")
    con.execute("INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference')")
    for t, c, cat in [("실습 파일 보관", "실습 파일은 개인 폴더에 보관합니다.", "learning"),
                      ("서버 실행", "프로젝트 폴더에서 서버를 실행합니다.", "environment"),
                      ("질문 작성", "오류 메시지와 시도한 작업을 함께 적습니다.", "learning")]:
        con.execute("INSERT INTO knowledge (title, content, category_id) SELECT ?, ?, id FROM categories WHERE name = ?", (t, c, cat))
    return con


# ------------------------------------------------------------ 1. 서비스 → 값 까지 확대
def fig_zoom():
    fig, ax = canvas(11.6, 3.7, 11.0, 3.9)
    steps = [
        ("AI Agent\n업무 서비스", "사용자에게\n답변·도구 제공", C["gray"], "-"),
        ("FastAPI", "요청·응답\n(11/10 · 11/12)", C["blue"], "-"),
        ("PostgreSQL 서버", "여러 DB를 관리하는\nDBMS", C["mint"], "-"),
        ("knowledge_lab\n· public", "Database·Schema\n이름 공간", C["mint"], "-"),
        ("categories\n· knowledge", "원본 지식을 담는\n두 테이블", C["mint"], "-"),
        ("한 행의\ncategory_id = 2", "값 하나가\n분류 2번을 참조", C["pink"], "-"),
    ]
    w = 1.62
    for i, (t, sub, col, ls) in enumerate(steps):
        x = 0.1 + i * 1.8
        rbox(ax, x, 1.75, w, 1.5, f"{t}\n\n{sub}", ec=col, fs=8.8, lw=2.2 if col == C["mint"] else 1.6)
        if i < 5:
            arrow(ax, x + w, 2.5, x + 1.8, 2.5)
    ax.add_patch(FancyBboxPatch((3.65, 1.55), 5.3, 1.9, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc="none", ec=C["mint"], lw=1.4, ls="--"))
    ax.text(6.3, 3.62, "오늘 직접 다루는 곳 (SQL)", ha="center", fontsize=10.5, color=C["mint"], fontweight="bold")
    rbox(ax, 0.1, 0.25, 5.2, 0.9, "검색·RAG (11/18~20)  ·  Agent 도구 실행 (이후)\n이 원본을 읽어 가는 별도 계층", ec=C["gray"], ls="--", fs=9, color=MUTED)
    arrow(ax, 5.35, 0.7, 6.0, 1.55, color=C["gray"], ls="--")
    ax.text(6.9, 0.55, "원본이 바뀌어도 검색 인덱스는 자동으로 안 바뀌어요", fontsize=9.5, color=MUTED)
    ax.set_title("서비스에서 값 하나까지 — 오늘은 초록 칸(DB 안쪽)을 SQL로 직접 만져요", loc="left")
    save(fig, 18, "zoom_map")


# ------------------------------------------------------------ 2. JSON 한 건 → 두 테이블 (ERD)
def fig_erd(con):
    fig, ax = canvas(11, 4.4, 11, 4.4)
    rbox(ax, 0.1, 0.9, 2.6, 2.6, '어제 (JSON 한 건)\n\n{"id": 2,\n "title": "서버 실행",\n "content": "...",\n "category": "environment"}',
         ec=C["gray"], fs=9.3, ha="left")
    ax.text(1.4, 0.6, "분류 이름이 지식마다 반복", ha="center", fontsize=9, color=MUTED)
    arrow(ax, 2.8, 2.2, 3.5, 2.2, text="나누기", tfs=9.5)
    cats = con.execute("SELECT id, name FROM categories ORDER BY id").fetchall()
    know = con.execute("SELECT id, title, category_id FROM knowledge ORDER BY id").fetchall()
    # categories
    rbox(ax, 3.6, 3.15, 2.4, 0.5, "categories (부모)", fc=C["soft"], ec=C["purple"], bold=True, fs=10)
    ax.text(3.75, 2.85, "id  BIGINT · PK · IDENTITY", fontsize=8.6, color=C["purple"])
    ax.text(3.75, 2.6, "name  VARCHAR(30) · NN · UNIQUE", fontsize=8.6, color=C["ink"])
    for i, (cid, name) in enumerate(cats):
        y = 2.15 - i * 0.38
        ax.text(3.85, y, f"{cid}", fontsize=10, color=C["purple"], fontweight="bold")
        ax.text(4.3, y, name, fontsize=10, color=C["ink"])
    # knowledge
    rbox(ax, 7.4, 3.15, 3.5, 0.5, "knowledge (자식)", fc="#e6fcf5", ec=C["mint"], bold=True, fs=10)
    ax.text(7.5, 2.9, "id  BIGINT · PK · IDENTITY  ·  title  VARCHAR(80) · NN", fontsize=8.3, color=C["ink"])
    ax.text(7.5, 2.66, "content  TEXT · NN  ·  category_id  BIGINT · NN · FK", fontsize=8.3, color=C["mint"])
    ax.text(7.6, 2.38, "category_id", fontsize=8.3, color=C["mint"], fontweight="bold")
    ax.text(8.65, 2.38, "id", fontsize=8.3, color=MUTED, fontweight="bold")
    ax.text(9.05, 2.38, "title", fontsize=8.3, color=MUTED, fontweight="bold")
    for i, (kid, title, cat) in enumerate(know):
        y = 2.0 - i * 0.38
        ax.text(7.95, y, f"{cat}", fontsize=10.5, color=C["mint"], fontweight="bold")
        ax.text(8.7, y, f"{kid}", fontsize=10, color=C["ink"], fontweight="bold")
        ax.text(9.05, y, title, fontsize=10, color=C["ink"])
        ty = 2.15 - (cat - 1) * 0.38
        arrow(ax, 7.8, y + 0.07, 5.75, ty + 0.07, color=C["mint"], lw=1.3)
    ax.text(6.7, 3.0, "1   :   0..N", ha="center", fontsize=11, color=C["ink"], fontweight="bold")
    ax.text(6.0, 0.55, "FK  knowledge.category_id → categories.id  (ON DELETE RESTRICT)", fontsize=9.5, color=C["mint"])
    ax.text(6.0, 0.22, "reference(3)를 가리키는 지식은 아직 0개 — 그래도 분류는 존재할 수 있어요", fontsize=9, color=MUTED)
    ax.set_title("JSON 한 건을 두 테이블로 — 분류 이름은 부모 표에 한 번만, 지식은 번호(FK)로 가리켜요", loc="left")
    save(fig, 18, "erd")


# ------------------------------------------------------------ 3. INNER JOIN vs LEFT JOIN vs COUNT
def fig_join(con):
    inner = con.execute("SELECT c.name, k.id, k.title FROM categories c JOIN knowledge k ON k.category_id = c.id ORDER BY c.id, k.id").fetchall()
    left = con.execute("SELECT c.name, k.id, k.title FROM categories c LEFT JOIN knowledge k ON k.category_id = c.id ORDER BY c.id, k.id").fetchall()
    cnt = con.execute("SELECT c.name, COUNT(*), COUNT(k.id) FROM categories c LEFT JOIN knowledge k ON k.category_id = c.id GROUP BY c.id ORDER BY c.id").fetchall()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    show = lambda v: "NULL" if v is None else str(v)
    for ax, rows, title, col in [(axes[0], inner, f"JOIN → {len(inner)}행", C["purple"]),
                                 (axes[1], left, f"LEFT JOIN → {len(left)}행", C["mint"])]:
        ax.axis("off"); ax.set_xlim(0, 4); ax.set_ylim(0, 5.2)
        ax.text(0.05, 4.75, title, fontsize=12, fontweight="bold", color=col)
        for j, h in enumerate(["c.name", "k.id", "k.title"]):
            ax.text([0.1, 1.55, 2.25][j], 4.1, h, fontsize=9.5, color=MUTED, fontweight="bold")
        for i, r in enumerate(rows):
            y = 3.45 - i * 0.62
            null = r[1] is None
            ax.add_patch(FancyBboxPatch((0.02, y - 0.2), 3.9, 0.48, boxstyle="round,pad=0.01,rounding_size=0.06",
                                        fc="#fff0f6" if null else "#ffffff", ec=C["pink"] if null else "#e3dcf2", lw=1.3))
            for j, v in enumerate(r):
                ax.text([0.1, 1.55, 2.25][j], y, show(v), fontsize=10, va="center",
                        color=C["pink"] if v is None else C["ink"], fontweight="bold" if v is None else "normal")
    axes[0].text(0.05, 0.75, "짝이 없는 reference는 빠짐", fontsize=9.5, color=MUTED)
    axes[1].text(0.05, 0.35, "왼쪽(분류)은 모두 남고\n짝이 없으면 오른쪽이 NULL", fontsize=9.5, color=C["pink"])
    ax = axes[2]
    names = [r[0] for r in cnt]
    xs = range(len(names))
    ax.bar([x - 0.18 for x in xs], [r[1] for r in cnt], width=0.34, color=C["orange"], label="COUNT(*)")
    ax.bar([x + 0.18 for x in xs], [r[2] for r in cnt], width=0.34, color=C["mint"], label="COUNT(k.id)")
    for x, r in zip(xs, cnt):
        ax.text(x - 0.18, r[1] + 0.05, str(r[1]), ha="center", fontsize=10, color=C["orange"], fontweight="bold")
        ax.text(x + 0.18, r[2] + 0.05, str(r[2]), ha="center", fontsize=10, color=C["mint"], fontweight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels(names); ax.set_ylim(0, 2.6); ax.set_yticks([0, 1, 2])
    ax.legend(loc="upper right", fontsize=9); ax.set_title("LEFT JOIN 뒤 분류별 개수", fontsize=11.5)
    fig.suptitle("같은 두 표, 다른 결과 — JOIN은 짝 있는 행만, LEFT JOIN은 왼쪽을 다 남겨요", x=0.01, ha="left",
                 fontweight="bold", fontsize=13)
    save(fig, 18, "join_kinds")


# ------------------------------------------------------------ 4. ROLLBACK vs COMMIT 타임라인
def fig_tx():
    fig, ax = canvas(11.5, 4.3, 11.5, 4.3)
    orig, tmp, new = "프로젝트 폴더에서" + chr(10) + "서버를 실행합니다.", "취소할 임시 안내입니다.", "실행 전에" + chr(10) + "가상환경을 확인합니다."
    ax.text(0.1, 3.95, "ROLLBACK 흐름", fontsize=11.5, fontweight="bold", color=C["orange"])
    xs = [0.1, 2.35, 4.6, 6.85, 9.1]
    lab = ["SELECT", "BEGIN", "UPDATE → SELECT", "ROLLBACK", "SELECT"]
    val = [orig, "", tmp, "", orig]
    for x, l, v in zip(xs, lab, val):
        ec = C["orange"] if l in ("BEGIN", "ROLLBACK") else C["purple"]
        rbox(ax, x, 2.6, 2.1, 1.1, l + (f"\n\n{v}" if v else "\n\n(작업 묶음 시작)" if l == "BEGIN" else "\n\n(미확정 변경 취소)"),
             ec=ec, fs=8.6, fc="#fff4e6" if v == tmp else "#ffffff")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + 2.1, 3.15, b, 3.15)
    ax.text(5.65, 2.3, "다른 연결에서 보면 아직 원래 본문 (실제 실행 확인)", ha="center", fontsize=8.8, color=MUTED)
    ax.text(0.1, 1.85, "COMMIT 흐름", fontsize=11.5, fontweight="bold", color=C["mint"])
    lab2 = ["BEGIN", "UPDATE … RETURNING", "COMMIT", "연결 닫고 재접속", "ROLLBACK?"]
    val2 = ["(작업 묶음 시작)", new, "(변경 확정)", new, "열린 묶음 없음 → 값 그대로"]
    for x, l, v in zip(xs, lab2, val2):
        ec = C["mint"] if l in ("BEGIN", "COMMIT") else (C["red"] if l == "ROLLBACK?" else C["purple"])
        rbox(ax, x, 0.5, 2.1, 1.1, f"{l}\n\n{v}", ec=ec, fs=8.6, fc="#e6fcf5" if v == new else "#ffffff")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, a + 2.1, 1.05, b, 1.05)
    ax.text(0.1, 0.18, "id = 2 한 행의 content 값 · 행 수와 ID는 두 흐름 모두 그대로", fontsize=9, color=MUTED)
    ax.set_title("BEGIN 뒤에는 둘 중 하나 — ROLLBACK(없던 일로) 또는 COMMIT(확정, 되돌릴 수 없음)", loc="left")
    save(fig, 18, "tx_timeline")


# ------------------------------------------------------------ 5. IDENTITY 빈 번호 (PostgreSQL 문서 개념도)
def fig_gap():
    fig, ax = canvas(11, 3.0, 11, 3.0)
    seq = [("1", "행 있음", C["mint"]), ("2", "행 있음", C["mint"]), ("3", "행 있음", C["mint"]),
           ("4", "INSERT 뒤\nROLLBACK", C["orange"]), ("5", "CHECK 위반으로\nINSERT 실패", C["red"]), ("6", "다음 INSERT\nRETURNING id = 6", C["purple"])]
    for i, (n, t, col) in enumerate(seq):
        x = 0.2 + i * 1.8
        filled = col == C["mint"] or col == C["purple"]
        ax.add_patch(FancyBboxPatch((x, 1.35), 1.4, 1.0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#ffffff" if filled else "#f8f9fa", ec=col, lw=2, ls="-" if filled else "--"))
        ax.text(x + 0.7, 1.95, n, ha="center", va="center", fontsize=20, fontweight="bold", color=col)
        ax.text(x + 0.7, 0.95, t, ha="center", va="center", fontsize=8.8, color=col, linespacing=1.3)
    ax.text(0.2, 0.25, "PostgreSQL: 시퀀스가 한 번 발급한 번호는 ROLLBACK·실패해도 다시 쓰지 않을 수 있어요 (공식 문서 개념도) "
            "→ 테이블엔 1, 2, 3, 6 · 정상이에요", fontsize=9.3, color=MUTED)
    ax.set_title("ID는 '몇 번째 행'이 아니에요 — 다음 번호를 예상하지 말고 RETURNING id를 읽어요", loc="left")
    save(fig, 18, "identity_gap")


if __name__ == "__main__":
    con = s0()
    fig_zoom(); fig_erd(con); fig_join(con); fig_tx(); fig_gap()
    print("ok")
