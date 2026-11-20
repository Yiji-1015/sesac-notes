"""주제 페이지 '백엔드 > 데이터베이스 > FastAPI와 PostgreSQL 연동'을 만든다.

examples/db_api/app.py 를 노트 전용 DB에서 단계별로 실제 실행한다 (TestClient, SQLAlchemy 로그 포함).
실행: NOTES_PG_ENV=<.env 경로> C:/practice_assignment/.venv/Scripts/python.exe tools/pages/db_api.py
"""
import logging
import os
import sys
from html import escape as E
from pathlib import Path
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import notes_pg  # noqa: E402
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import fastapi  # noqa: E402
import sqlalchemy  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import inspect, text  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "db_api"
code = S.parse((EX / "app.py").read_text(encoding="utf-8"))
C = notes_pg.CRED
outs = {}
NEW = {"title": "가상환경 만들기", "content": "python -m venv .venv 로 만듭니다.", "category": "environment"}


def at(k):
    notes_pg.fresh()
    os.environ["DATABASE_URL"] = (f"postgresql+psycopg://{quote_plus(C['DB_USER'])}:{quote_plus(C['DB_PASSWORD'])}"
                                  f"@{C.get('DB_HOST', 'localhost')}:{C.get('DB_PORT', 5432)}/{notes_pg.LAB}")
    ns = S.run(code, k, S.workdir(EX), filename="app.py")
    cl = TestClient(ns["app"], raise_server_exceptions=False) if "app" in ns else None
    return ns, cl


def jr(r):
    try:
        return r.json()
    except Exception:  # noqa: BLE001
        return r.text


class Grab(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, rec):
        msg = rec.getMessage()
        if not msg.startswith("[") and msg.strip():
            self.lines.append(" ".join(msg.split()))


try:
    # ---- ① ----
    ns, _ = at(1)
    outs["s1"] = S.box("① 실행 결과", [(f"engine.url → {ns['engine'].url.render_as_string(hide_password=True)}", 1), (f"SELECT 1 → {ns['db_ok']}", 2)],
                       ["주소의 비밀번호 자리는 <code>***</code>로 가려져 보여요. 실제 값은 환경 변수에만 있어요.", "DB가 1을 돌려줬으니 연결이 됐어요."])
    ns["engine"].dispose()

    # ---- ② ----
    ns, _ = at(2)
    insp = inspect(ns["engine"])
    rows = []
    for t in insp.get_table_names():
        for col in insp.get_columns(t):
            rows.append([t, f"<code>{col['name']}</code>", E(str(col["type"])), "가능" if col["nullable"] else "불가"])
    with ns["engine"].connect() as c:
        cats = c.execute(text("SELECT id, name FROM categories ORDER BY id")).fetchall()
        nk = c.execute(text("SELECT count(*) FROM knowledge")).scalar()
    outs["s2"] = S.table("create_all()이 DB에 만든 테이블", ["테이블", "열", "타입", "NULL"], rows,
                         ["클래스에 적은 <code>Mapped[int]</code>, <code>String(30)</code>가 DB 타입 <code>INTEGER</code>, <code>VARCHAR(30)</code>로 바뀌었어요. <code>Mapped[str]</code>처럼 <code>Optional</code>이 아니면 NULL 불가가 돼요.",
                          f"시작 데이터: 분류 {[tuple(r) for r in cats]}, 지식 {nk}개"])
    ns["engine"].dispose()

    # ---- ③ ----
    ns, _ = at(3)
    eng = ns["engine"]
    before = eng.pool.checkedout()
    gen = ns["get_session"]()
    s = next(gen)
    s.execute(text("SELECT 1"))
    during = eng.pool.checkedout()
    gen.close()
    after = eng.pool.checkedout()
    outs["s3"] = S.box("get_session()을 요청 하나처럼 열고 닫으면", [f"요청 전   빌려 간 연결 → {before}", (f"요청 중   빌려 간 연결 → {during}", 1), (f"요청 끝   빌려 간 연결 → {after}", 2)],
                       ["Session이 처음 SQL을 보낼 때 연결 풀에서 연결 하나를 빌려 가요.", "요청이 끝나 <code>with</code> 블록을 나가면 Session이 닫히고 연결이 풀로 돌아와요. 요청마다 이렇게 열고 닫아서, 연결이 새거나 다른 요청과 섞이지 않아요."])
    eng.dispose()

    # ---- ④ ----
    ns, cl = at(4)
    rows = []
    for u in ["/knowledge", "/knowledge/2", "/knowledge/99"]:
        r = cl.get(u)
        b = jr(r)
        rows.append([f"<code>GET {u}</code>", f"<b>{r.status_code}</b>", f"<code>{E(str(b)[:110])}</code>"])
    outs["s4"] = S.table("④ 실행 결과", ["요청", "상태", "응답"], rows,
                         ["API 쪽 모습은 <a href=\"fastapi.html\">파일 버전</a>과 같아요. 바뀐 건 함수 안에서 파일 대신 Session으로 DB를 읽는다는 것뿐이에요."])
    ns["engine"].dispose()

    # ---- ⑤ ----
    ns, cl = at(5)
    g = Grab()
    lg = logging.getLogger("sqlalchemy.engine.Engine")
    lg.setLevel(logging.INFO)
    lg.addHandler(g)
    r = cl.post("/knowledge", json=NEW)
    lg.removeHandler(g)
    lg.setLevel(logging.WARNING)
    sql = [ln for ln in g.lines if ln.split(" ")[0] in ("BEGIN", "SELECT", "INSERT", "COMMIT", "ROLLBACK", "UPDATE", "DELETE")]
    explain = {"BEGIN": "트랜잭션 시작. Session이 처음 SQL을 보낼 때 알아서 시작해요.",
               "SELECT categories": "분류 이름으로 번호 찾기 (<code>select(Category).where(...)</code>가 바뀐 것)",
               "INSERT": "<code>commit()</code>이 보낸 저장. <code>RETURNING</code>으로 DB가 붙인 번호를 받아요.",
               "COMMIT": "확정",
               "SELECT knowledge": "<code>refresh(k)</code>가 저장된 행을 다시 읽어요. 그래서 바로 앞에 새 <code>BEGIN</code>이 있어요.",
               "ROLLBACK": "요청이 끝나 Session을 닫을 때, 읽기만 한 트랜잭션을 정리해요. 바뀐 건 없어요."}
    notes = []
    for ln in sql:
        key = next((k for k in explain if ln.startswith(k)), None)
        notes.append(explain.get(key, ""))
    outs["sql"] = S.box("POST /knowledge 한 번에 실제로 DB로 간 SQL (SQLAlchemy 로그)", [(ln[:150], i + 1) for i, ln in enumerate(sql)], notes)
    r2 = cl.post("/knowledge", json={**NEW, "title": "없는 분류", "category": "nothing"})
    r3 = cl.post("/knowledge", json=NEW)
    S.claim(r.status_code == 201 and r2.status_code == 422 and r3.status_code == 500, "⑤ 201, 422, 그리고 같은 제목은 500")
    outs["s5"] = S.box("⑤ 실행 결과", [(f"POST {NEW['title']} → {r.status_code} {jr(r)}", 1), (f"없는 분류 → {r2.status_code} {jr(r2)}", 2), (f"같은 제목 한 번 더 → {r3.status_code} {jr(r3)}", 3, "bad")],
                       [f"201과 함께 DB가 붙인 번호 {jr(r)['id']}가 왔어요.", "분류 이름이 틀리면 저장 전에 422로 돌려보내요. 이 검사는 우리가 직접 짰어요.",
                        "같은 제목은 DB의 <code>UNIQUE</code> 규칙에 걸려 <code>IntegrityError</code>가 났는데, 아무도 잡지 않아서 <b>500(서버 내부 오류)</b>이 됐어요. 서버가 고장 난 게 아니라 요청이 규칙을 어긴 건데도요. ⑥에서 고쳐요."])
    ns["engine"].dispose()

    # ---- ⑥ ----
    ns, cl = at(6)
    cl.post("/knowledge", json=NEW)
    r = cl.post("/knowledge", json=NEW)
    n = len(cl.get("/knowledge").json())
    r4 = cl.post("/knowledge", json={**NEW, "title": "다음 요청"})
    first_id = jr(r4)["id"] - 2   # 첫 POST가 받은 번호 (실패한 INSERT가 하나를 받았다가 버림)
    S.claim(r.status_code == 409 and n == first_id and jr(r4)["id"] == n + 2, "⑥ 409, 실패한 요청은 남지 않고, 다음 번호는 하나를 건너뛴다")
    outs["s6"] = S.box("같은 제목을 두 번 POST", [(f"두 번째 → {r.status_code} {jr(r)}", 1), (f"지식 수 → {n}", 2), (f"이어서 다른 POST → {r4.status_code}, id {jr(r4).get('id')}", 3)],
                       ["이제 409(충돌)로, 무엇이 문제인지 알려 줘요.", f"실패한 요청은 아무것도 남기지 않았어요 (시작 {n - 1}개 + 첫 POST 1개).",
                        f"실패한 요청의 Session은 닫히면서 트랜잭션이 취소돼서, 다음 요청에 영향이 없어요. 번호는 {jr(r4)['id']}이에요. 실패한 INSERT가 번호 {jr(r4)['id'] - 1}를 받았다가 버렸기 때문이에요."])
    ns["engine"].dispose()

    # ---- ⑦ ----
    ns, cl = at(7)
    r1 = cl.put("/knowledge/2", json={"title": "서버 실행", "content": "uvicorn app:app --reload 로 실행합니다.", "category": "environment"})
    r2 = cl.delete("/knowledge/3")
    r3 = cl.delete("/knowledge/3")
    ids = [k["id"] for k in cl.get("/knowledge").json()]
    outs["s7"] = S.box("⑦ 실행 결과", [(f"PUT /knowledge/2 → {r1.status_code} {jr(r1)}", 1), f"DELETE /knowledge/3 → {r2.status_code}", f"DELETE /knowledge/3 다시 → {r3.status_code} {jr(r3)}", (f"남은 id → {ids}", 2)],
                       ["<code>session.get()</code>으로 찾은 객체의 속성을 바꾸고 <code>commit()</code>하면 SQLAlchemy가 바뀐 속성만 <code>UPDATE</code>로 보내요.", "다섯 개 주소가 모두 DB로 동작해요."])
    ns["engine"].dispose()
finally:
    notes_pg.drop()

flow = [[1, "연결", "Engine", "DATABASE_URL"], [2, "ORM", "클래스 = 테이블", "create_all"], [3, "Session", "요청마다 하나", "Depends"],
        [4, "조회", "select()", "session.get"], [5, "등록", "add, commit", "201, 422"], [6, "오류", "IntegrityError", "500 → 409"], [7, "수정, 삭제", "PUT, DELETE", "200, 204, 404"]]
S.save(ROOT / "topics" / "backend" / "fastapi-postgres-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow,
                                                                   "env": f"Python {sys.version.split()[0]}, FastAPI {fastapi.__version__}, SQLAlchemy {sqlalchemy.__version__}, PostgreSQL (노트 전용 DB)"})

why = f"""
<p><a href="fastapi.html">API 서버</a>는 지식을 JSON 파일에 저장했고, <a href="postgres.html">PostgreSQL 기초</a>에서는 SQL로 DB를 다뤘어요. 이제 둘을 이어서, API가 받은 요청을 DB에 저장해요.</p>
<p>파일 읽기, 쓰기를 SQL로 바꾸기만 하면 될 것 같지만, 실제로는 세 가지를 새로 정해야 해요.</p>
{K.table(["무엇을", "왜", "어디서"], [["DB 연결을 언제 열고 닫나", "요청마다 연결을 새로 만들면 느리고, 닫는 걸 잊으면 연결이 바닥나요.", "③ Session"],
                                     ["파이썬 객체와 테이블 행을 어떻게 오가나", "SQL 문자열과 결과 튜플을 손으로 다루면 실수가 잦아요.", "② ORM"],
                                     ["DB 오류를 어떤 응답으로 바꾸나", "규칙 위반을 그냥 두면 모두 500(서버 고장)으로 보여요.", "⑥ 오류 처리"]])}
<p>여기서는 <b>SQLAlchemy</b>를 써요. 파이썬 클래스로 테이블을 적으면 SQL을 대신 만들어 주는 도구(<b>ORM</b>, Object Relational Mapping)이고, 연결과 트랜잭션도 관리해 줘요.</p>
"""
whole = f"""
<p>요청이 오면 FastAPI가 Session을 하나 열어 함수에 넘기고, 함수는 파이썬 객체로 DB를 다루고, SQLAlchemy가 그걸 SQL로 바꿔 보내요. 응답을 보내면 Session을 닫아요. 상자를 누르면 그 단계로 가요.</p>
<div class="flow" id="flow"></div>
<p>실제로 <code>POST /knowledge</code> 하나를 보냈을 때 DB로 간 SQL이에요. 파이썬 코드에는 SQL이 한 줄도 없는데, 이만큼이 오갔어요.</p>
{K.out("sql")}
<p>결과는 모두 노트 전용 연습 DB를 새로 만들어 <code>app.py</code>를 단계별로 실행하고 <code>TestClient</code>로 요청을 보낸 값이에요.</p>
"""
steps = [
    K.step(1, "DB에 연결하기", "접속 주소를 환경 변수에서 읽어 <b>Engine</b>(연결을 관리하는 객체)을 만들어요.",
           "<p>접속 주소는 <code>postgresql+psycopg://사용자:비밀번호@주소:포트/DB이름</code> 꼴이에요. <code>+psycopg</code>는 \"psycopg로 접속하라\"는 뜻이에요. 비밀번호가 들어 있어서 코드 대신 환경 변수 <code>DATABASE_URL</code>에 둬요.</p>"
           + K.code_table([('os.environ["DATABASE_URL"]', "환경 변수. 없으면 <code>KeyError</code>로 바로 멈춰서, 엉뚱한 DB에 붙는 일을 막아요."),
                           ("create_engine(url)", "Engine을 만들어요. 이때는 아직 접속하지 않고, 필요할 때 연결을 빌려 줘요(연결 풀)."),
                           ('c.execute(text("SELECT 1")).scalar()', "연결이 되는지 가장 간단한 SQL로 확인. <code>text()</code>는 SQL 글을 그대로 보낼 때 써요.")]), "s1"),
    K.step(2, "테이블을 파이썬 클래스로 (ORM)", "클래스 하나가 테이블 하나, 속성 하나가 열 하나예요.",
           K.code_table([("class Base(DeclarativeBase)", "모든 테이블 클래스의 부모. 테이블 목록(<code>metadata</code>)을 모아 둬요."),
                         ('__tablename__ = "knowledge"', "DB의 테이블 이름"),
                         ("Mapped[int] = mapped_column(primary_key=True)", "정수 열, 기본 키. 번호는 DB가 붙여요."),
                         ("mapped_column(String(80), unique=True)", "80자 글, 중복 금지"),
                         ('ForeignKey("categories.id")', "외래 키. <code>categories</code>의 <code>id</code>만 가리킬 수 있어요."),
                         ("Base.metadata.create_all(engine)", "클래스대로 테이블을 만들어요. <b>이미 있으면 건드리지 않아서</b>, 열을 추가하는 것 같은 변경은 못 해요. 그래서 실제로는 Alembic을 써요 (실제로 할 때).")]), "s2"),
    K.step(3, "요청마다 Session 하나", "요청 하나가 Session 하나를 쓰고, 요청이 끝나면 닫히게 해요.",
           "<p><b>Session</b>은 \"이번 작업에서 읽고 바꾼 것\"을 모아 두었다가 한 트랜잭션으로 DB에 보내는 작업 공간이에요. 요청끼리 Session을 같이 쓰면 한 요청의 실패가 다른 요청까지 취소해요. 그래서 요청마다 새로 만들어요.</p>"
           + K.code_table([("sessionmaker(engine)", "Session을 찍어 내는 틀"),
                           ("def get_session(): with SessionLocal() as session: yield session", "<code>yield</code>로 Session을 넘겨주고, 요청이 끝나면 돌아와서 <code>with</code>를 나가며 닫아요."),
                           ("session: Session = Depends(get_session)", "FastAPI가 요청마다 <code>get_session()</code>을 불러 그 결과를 이 인자로 넣어 줘요 (의존성 주입).")]), "s3"),
    K.step(4, "조회하기", "<code>select()</code>로 목록을, <code>session.get()</code>으로 번호 하나를 읽어요.",
           K.code_table([("select(Knowledge).order_by(Knowledge.id)", "<code>SELECT ... FROM knowledge ORDER BY id</code>를 파이썬으로 적은 것"),
                         ("session.scalars(...)", "결과를 <code>Knowledge</code> 객체로 하나씩"),
                         ("session.get(Knowledge, knowledge_id)", "기본 키로 하나 찾기. 없으면 <code>None</code> → 404"),
                         ("to_dict(k)", "객체를 JSON으로 보낼 dict로 바꿔요.")]), "s4"),
    K.step(5, "등록하기", "분류 이름을 번호로 바꾸고, 객체를 만들어 <code>add</code>, <code>commit</code>해요.",
           "<p>입력 모양 검사는 Pydantic(<code>KnowledgeCreate</code>)이, 저장 규칙은 DB가 맡아요. 둘은 다른 일이에요. Pydantic은 \"제목이 글이고 1~80자인가\"를 보고, DB는 \"같은 제목이 이미 있나\", \"분류 번호가 있나\"를 봐요.</p>"
           + K.code_table([("select(Category).where(Category.name == payload.category)", "분류 이름으로 분류 찾기. 없으면 422"),
                           ("session.add(k)", "새 객체를 Session에 올려요. 아직 DB에는 안 갔어요."),
                           ("session.commit()", "모아 둔 변경을 <code>INSERT</code>로 보내고 확정해요."),
                           ("session.refresh(k)", "DB가 붙인 번호 같은 값을 객체에 다시 읽어 와요.")]), "s5"),
    K.step(6, "DB 오류를 알맞은 응답으로", "<code>IntegrityError</code>(DB 규칙 위반)가 나면 500 대신 409(충돌)로 답해요.",
           "<p>⑤에서 같은 제목을 두 번 보내면 500이 났어요. 500은 \"서버가 고장 났다\"는 뜻이라, 클라이언트는 다시 보내면 될지, 입력을 고쳐야 할지 알 수 없어요. 원인별로 다른 응답을 줘야 해요.</p>"
           + K.table(["원인", "응답"], [["입력 모양이 틀림 (Pydantic)", "422"], ["찾는 번호가 없음", "404"], ["DB 규칙과 충돌 (같은 제목)", "409"], ["진짜 서버 문제 (DB 꺼짐 등)", "500"]])
           + K.code_table([("@app.exception_handler(IntegrityError)", "어느 주소에서든 이 오류가 나면 이 함수가 응답을 만들어요."),
                           ("JSONResponse(status_code=409, content=...)", "상태 코드와 본문을 직접 정한 응답")]), "s6"),
    K.step(7, "고치고 지우기", "찾은 객체의 속성을 바꾸거나 <code>session.delete()</code>로 지우고 확정해요.",
           K.code_table([("k.title, k.content, k.category_id = ...", "객체 속성만 바꾸면, <code>commit()</code> 때 바뀐 열만 <code>UPDATE</code>돼요."),
                         ("session.delete(k)", "지울 객체로 표시하고, <code>commit()</code> 때 <code>DELETE</code>")]), "s7"),
]
practice = [
    ("테이블 구조를 바꿔야 한다", "<code>create_all()</code>은 이미 있는 테이블을 고치지 않아요. <b>Alembic</b>으로 \"무엇을 어떻게 바꿨는지\"를 마이그레이션 파일로 남기고 <code>alembic upgrade head</code>로 적용해요. DB 구조의 Git 같은 거예요. 자세한 내용은 <a href=\"../../notes/day19.html\">옛 Day19 노트</a>를 보세요."),
    ("응답이 이상하다", "가장 먼저 <b>어느 DB에 붙었는지</b> 확인해요 (①의 <code>engine.url</code>). 연습 DB와 실제 DB를 헷갈리는 경우가 많아요."),
    ("ORM 모델과 Pydantic 모델", "ORM 모델(<code>Knowledge</code>)은 DB 행의 모양, Pydantic 모델(<code>KnowledgeCreate</code>)은 요청의 모양이에요. 등록할 때는 <code>id</code>가 없고 분류를 이름으로 받는 것처럼 둘은 달라요. 하나로 합치지 않아요."),
    ("요청이 많아진다", "Engine의 연결 풀 크기(<code>pool_size</code>)만큼만 동시에 DB를 써요. 요청이 끝나면 연결을 돌려주는 ③이 중요한 이유예요."),
]
libs = [("①", "<code>create_engine(url)</code>, <code>engine.connect()</code>, <code>text()</code>", "연결 (SQLAlchemy)"),
        ("②", "<code>DeclarativeBase</code>, <code>Mapped</code>, <code>mapped_column()</code>, <code>ForeignKey</code>, <code>metadata.create_all()</code>", "ORM 모델과 테이블"),
        ("③", "<code>sessionmaker()</code>, <code>Depends()</code>", "요청마다 Session (SQLAlchemy, FastAPI)"),
        ("④~⑦", "<code>select()</code>, <code>session.scalars()</code>, <code>session.get()</code>, <code>add()</code>, <code>commit()</code>, <code>refresh()</code>, <code>delete()</code>", "읽고 쓰기 (SQLAlchemy)"),
        ("⑥", "<code>IntegrityError</code>, <code>@app.exception_handler</code>, <code>JSONResponse</code>", "DB 오류를 응답으로")]
K.build(ROOT / "topics" / "backend" / "fastapi-postgres.html", title="FastAPI와 PostgreSQL 연동", h1="FastAPI와 PostgreSQL 연동: API 요청을 DB 행으로", track="백엔드",
        crumb="백엔드 &gt; 데이터베이스",
        one_line="API가 받은 요청을 DB에 저장하려면 <b>연결(Session)</b>, <b>객체와 행의 변환(ORM)</b>, <b>DB 오류를 응답으로 바꾸기</b>를 정해야 해요. SQLAlchemy가 앞의 둘을 맡아요.",
        why=why, whole=whole, steps=steps, code_file="examples/db_api/app.py", practice=practice, libs=libs,
        libs_note="<code>get_session</code>, <code>to_dict</code>와 각 주소의 함수는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"fastapi.html\">FastAPI로 지식 API 만들기</a>", "같은 API의 파일 버전"), ("<a href=\"postgres.html\">PostgreSQL 기초</a>", "SQL과 제약 조건, 트랜잭션"),
                   ("<a href=\"indexing.html\">인덱싱</a>", "같은 DB에 벡터를 넣고 검색하기")],
        sources=["다루는 범위: Day19 (2026-11-12) FastAPI와 PostgreSQL 연동 (Alembic은 실제로 할 때에만)",
                 "예제: <code>examples/db_api/app.py</code> (기존 노트의 분류, 지식 예제)",
                 "결과 값: <code>tools/pages/db_api.py</code>가 노트 전용 DB를 만들어 단계별로 실제 실행하고 지운 결과",
                 "공식 문서: <a href=\"https://docs.sqlalchemy.org/en/20/orm/quickstart.html\" target=\"_blank\">SQLAlchemy ORM Quick Start</a>, <a href=\"https://fastapi.tiangolo.com/tutorial/sql-databases/\" target=\"_blank\">FastAPI SQL Databases</a>, <a href=\"https://fastapi.tiangolo.com/tutorial/handling-errors/\" target=\"_blank\">Handling Errors</a>, <a href=\"https://alembic.sqlalchemy.org/en/latest/tutorial.html\" target=\"_blank\">Alembic Tutorial</a>"])
print("ok")
