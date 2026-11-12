import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day19 그림 만들기 (FastAPI–PostgreSQL 연동 · ORM · Session · Alembic).

숫자가 들어간 그림(Session 상태)은 tools/io_day19.py 로 강의 코드를 실제 PostgreSQL 18에서 돌린 결과다
(add 직후 id None → flush 뒤 7 · 다른 연결 0행 → rollback → 다음 flush 8 → commit 뒤 다른 연결 1행).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day19.py
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MUTED = "#5b5270"


def canvas(w, h, xmax, ymax):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax); ax.axis("off")
    return fig, ax


def rbox(ax, x, y, w, h, text, fc="#ffffff", ec=None, fs=9.5, bold=False, color=None, ls="-", lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec or C["purple"], lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            color=color or C["ink"], linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=None, lw=1.7, ls="-", text=None, tfs=8.8, toff=(0, 0.08)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color or MUTED, lw=lw, ls=ls))
    if text:
        ax.text((x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1], text, ha="center", va="bottom", fontsize=tfs, color=color or MUTED)


# ------------------------------------------------------------ 1. POST 한 건의 길
def fig_flow():
    fig, ax = canvas(12, 4.6, 12, 4.6)
    steps = [("HTTP 요청", "POST /api/knowledge\nBody: title·content·\ncategory", C["gray"]),
             ("① Pydantic", "KnowledgeWrite\n길이·필수 검사", C["blue"]),
             ("② get_db", "요청 전용\nSession 준비", C["purple"]),
             ("③ transaction", "분류 찾기/만들기\n→ 지식 add\n→ flush (id 받음)", C["orange"]),
             ("④ COMMIT", "분류+지식\n함께 확정", C["mint"]),
             ("⑤ 응답", "201 + DB가\n발급한 id", C["mint"])]
    for i, (t, sub, col) in enumerate(steps):
        x = 0.1 + i * 1.98
        rbox(ax, x, 2.35, 1.72, 1.55, f"{t}\n\n{sub}", ec=col, fs=8.6, lw=2.1)
        if i < 5:
            arrow(ax, x + 1.72, 3.12, x + 1.98, 3.12)
    # 갈림길
    rbox(ax, 2.1, 0.35, 1.72, 0.95, "422\n함수 실행 전 거부\n(DB 안 감)", ec=C["pink"], fs=8.6, color=C["pink"])
    arrow(ax, 2.95, 2.35, 2.95, 1.32, color=C["pink"], text="길이·필수 위반", toff=(0.55, -0.1))
    rbox(ax, 6.05, 0.35, 1.9, 0.95, "ROLLBACK → 예외\n409 (알려진 UNIQUE·FK)\n500 (그 밖의 DB 오류)", ec=C["red"], fs=8.2, color=C["red"])
    arrow(ax, 7.0, 2.35, 7.0, 1.32, color=C["red"], text="DB 제약 위반", toff=(0.6, -0.1))
    rbox(ax, 9.9, 0.35, 1.95, 0.95, "다른 연결·pgAdmin에서\n같은 id 행이 보임", ec=C["mint"], fs=8.6, color=C["green"])
    arrow(ax, 10.87, 2.35, 10.87, 1.32, color=C["mint"])
    ax.text(0.1, 4.25, "Session은 ②에서 열리고 요청이 끝나면 close — close ≠ commit", fontsize=9.5, color=MUTED)
    ax.set_title("POST 한 건이 지나가는 길 — 201은 COMMIT이 성공한 뒤에만 나가요", loc="left")
    save(fig, 19, "request_flow")


# ------------------------------------------------------------ 2. 바깥 표현 vs 안쪽 표현
def fig_mapping():
    fig, ax = canvas(12, 4.1, 12, 4.1)
    rbox(ax, 0.1, 0.5, 3.0, 3.0, 'API (Pydantic)\n\nKnowledgeWrite / Read\n\n{"id": 5,\n "title": "실습 환경 안내",\n "content": "...",\n "category": "learning"}',
         ec=C["blue"], fs=9.2)
    rbox(ax, 4.6, 0.5, 3.0, 3.0, "ORM (SQLAlchemy)\n\nclass Knowledge(Base)\n\nid · title · content\ncategory_id = 1\n\nSession이 객체를 추적", ec=C["purple"], fs=9.2)
    rbox(ax, 8.6, 2.3, 3.3, 1.2, "public.categories\nid = 1 · name = 'learning'", ec=C["mint"], fs=9.5)
    rbox(ax, 8.6, 0.5, 3.3, 1.2, "public.knowledge\nid = 5 · category_id = 1", ec=C["mint"], fs=9.5)
    arrow(ax, 10.25, 1.72, 10.25, 2.28, color=C["mint"])
    arrow(ax, 3.15, 2.3, 4.55, 2.3, text="get_or_create\n_category\n문자열 → 번호", tfs=8.2, toff=(0, 0.08))
    arrow(ax, 4.55, 1.5, 3.15, 1.5, text="to_read\n번호 → 분류명", tfs=8.2, toff=(0, -0.62))
    arrow(ax, 7.7, 1.1, 8.55, 1.1, text="flush/commit", tfs=8.4)
    ax.text(10.4, 2.0, "FK", fontsize=9.5, color=C["mint"], fontweight="bold", va="center")
    ax.set_title("같은 지식, 세 가지 모양 — 바깥은 category 문자열, 안쪽은 category_id 번호", loc="left")
    save(fig, 19, "mapping")


# ------------------------------------------------------------ 3. Session 메서드 — 실제 실행 값
def fig_session():
    fig, ax = canvas(12, 3.7, 12, 3.7)
    cols = [("add(row)", "row.id = None", "다른 연결: —", "SQL 아직 안 보냄", C["gray"]),
            ("flush()", "row.id = 7", "다른 연결: 0행", "INSERT 보냄 · 미확정", C["orange"]),
            ("rollback()", "7번 행 취소", "다른 연결: 0행", "번호 7은 다시 안 씀", C["red"]),
            ("add → flush", "row2.id = 8", "다른 연결: 0행", "새 행은 8번부터", C["orange"]),
            ("commit()", "row2 확정", "다른 연결: 1행", "이제 모두에게 보임", C["mint"]),
            ("close()", "Session 정리", "값은 그대로", "저장 명령 아님", C["purple"])]
    for i, (t, a, b, c, col) in enumerate(cols):
        x = 0.1 + i * 1.98
        rbox(ax, x, 0.55, 1.75, 2.0, f"{a}\n{b}\n\n{c}", ec=col, fs=8.8, lw=2.1)
        ax.text(x + 0.875, 2.75, t, ha="center", fontsize=10, fontweight="bold", color=col)
        if i < 5:
            arrow(ax, x + 1.75, 1.78, x + 1.98, 1.78)
    ax.text(0.1, 0.18, "실제 PostgreSQL 18에서 강의 코드(SessionLocal)로 실행한 값 · 다른 연결 = 별도 연결로 같은 id를 SELECT", fontsize=9, color=MUTED)
    ax.set_title("add · flush · commit은 다른 일 — flush는 id를 받아도 아직 미확정이에요", loc="left")
    save(fig, 19, "session_states")


# ------------------------------------------------------------ 4. Migration 흐름
def fig_migration():
    fig, ax = canvas(12, 4.0, 12, 4.0)
    steps = [("빈 연습 DB", "alembic current\n(적용 없음)", C["gray"]),
             ("001_initial", "두 테이블 생성\ncurrent = 001", C["blue"]),
             ("기준 데이터", "가상 지식 1건\nCOMMIT", C["purple"]),
             ("002_source", "source_label 추가\n(NULL 허용)", C["mint"]),
             ("검증", "열 목록 · 기존 행\ncurrent = 002 (head)", C["mint"])]
    for i, (t, sub, col) in enumerate(steps):
        x = 0.1 + i * 2.4
        rbox(ax, x, 1.9, 2.05, 1.45, f"{t}\n\n{sub}", ec=col, fs=9, lw=2.1)
        if i < 4:
            arrow(ax, x + 2.05, 2.62, x + 2.4, 2.62)
    rbox(ax, 7.3, 0.2, 4.55, 1.2, "downgrade 001_initial → 다시 upgrade head\n열은 돌아오지만 source_label 값('사내 위키')은 NULL\n= drop_column은 데이터를 지워요", ec=C["red"], fs=8.8, color=C["red"])
    arrow(ax, 9.75, 1.9, 9.6, 1.42, color=C["red"])
    rbox(ax, 0.1, 0.2, 6.6, 1.2, "ORM Model에 열을 적는 것 ≠ DB 변경\nRevision(변경 기록)을 사람이 읽고 → 연습 DB에 upgrade → 실제 열·기존 행 확인\n주 프로젝트 DB · Migration 연습 DB · 테스트 DB는 서로 다른 DB", ec=C["gray"], fs=8.8, color=MUTED)
    ax.set_title("Alembic으로 구조를 한 단계씩 — 노트 전용 연습 DB에서 실제로 적용한 순서", loc="left")
    save(fig, 19, "migration_flow")


if __name__ == "__main__":
    fig_flow(); fig_mapping(); fig_session(); fig_migration(); print("ok")
