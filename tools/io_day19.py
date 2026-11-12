"""Day19(FastAPI–PostgreSQL 연동) 명령어 사전의 '입력 → 출력' 상자를 실제 PostgreSQL 실행 결과로 만든다.

- 접속 정보: 환경변수 NOTES_PG_ENV 가 가리키는 .env (DB_HOST/DB_PORT/DB_USER/DB_PASSWORD) — 비밀번호는 어디에도 출력하지 않는다.
- 노트 전용 DB 두 개(sesac_notes_lab, sesac_notes_migration)를 새로 만들어 쓰고 끝나면 지운다. 수업용 DB는 건드리지 않는다.
- 강의 코드(settings.py · db.py · models.py · main.py · Alembic 002_source)를 임시 폴더에 그대로 만들고
  TestClient · uvicorn(빈 포트) · alembic 명령으로 실제로 돌린다.
실행 (fastapi·sqlalchemy·psycopg·alembic·pydantic-settings 가 깔린 python):
  NOTES_PG_ENV=<.env 경로> python tools/io_day19.py
"""
import html
import importlib
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LAB, MIG = "sesac_notes_lab", "sesac_notes_migration"


def read_env(p):
    out = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


CRED = read_env(os.environ["NOTES_PG_ENV"])
SECRET = CRED["DB_PASSWORD"]


def scrub(s):
    s = str(s)
    return s.replace(SECRET, "********") if SECRET else s


from sqlalchemy import URL, create_engine, text  # noqa: E402


def url_for(db):
    return URL.create("postgresql+psycopg", username=CRED["DB_USER"], password=SECRET,
                      host=CRED.get("DB_HOST", "localhost"), port=int(CRED.get("DB_PORT", 5432)), database=db)


admin = create_engine(url_for("postgres"), isolation_level="AUTOCOMMIT", hide_parameters=True)


def recreate(db):
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{db}"'))
        c.execute(text(f'CREATE DATABASE "{db}"'))


def drop(db):
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)'))


recreate(LAB); recreate(MIG)

DDL = """
CREATE TABLE categories (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  name VARCHAR(30) NOT NULL UNIQUE,
  CONSTRAINT categories_name_length CHECK (char_length(name) BETWEEN 1 AND 30));
CREATE TABLE knowledge (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  title VARCHAR(80) NOT NULL, content TEXT NOT NULL, category_id BIGINT NOT NULL,
  CONSTRAINT knowledge_title_length CHECK (char_length(title) BETWEEN 1 AND 80),
  CONSTRAINT knowledge_content_length CHECK (char_length(content) BETWEEN 1 AND 2000),
  CONSTRAINT knowledge_category_fk FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE RESTRICT);
INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference');
INSERT INTO knowledge (title, content, category_id) SELECT '실습 파일 보관', '실습 파일은 개인 폴더에 보관합니다.', id FROM categories WHERE name = 'learning';
INSERT INTO knowledge (title, content, category_id) SELECT '서버 실행', '프로젝트 폴더에서 서버를 실행합니다.', id FROM categories WHERE name = 'environment';
INSERT INTO knowledge (title, content, category_id) SELECT '질문 작성', '오류 메시지와 시도한 작업을 함께 적습니다.', id FROM categories WHERE name = 'learning';
"""
lab_engine = create_engine(url_for(LAB), hide_parameters=True)
with lab_engine.begin() as c:
    for stmt in [s for s in DDL.split(";") if s.strip()]:
        c.execute(text(stmt))


def sql(q, **kw):
    """노트 SQL을 새 연결로 실행해 '열 | 열' 글자로 (pgAdmin에서 확인하는 것과 같은 역할)."""
    with lab_engine.connect() as c:
        r = c.execute(text(q), kw)
        cols = list(r.keys()); rows = r.all()
    if not rows:
        return " | ".join(cols) + "\n(0행)"
    return "\n".join([" | ".join(cols)] + [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows])


# ---------------------------------------------------------------- 강의 코드 (PDF 그대로, 부록 A02~A09)
PROJ = Path(tempfile.mkdtemp())
(PROJ / ".env").write_text(
    f"DB_HOST={CRED.get('DB_HOST', 'localhost')}\nDB_PORT={CRED.get('DB_PORT', 5432)}\nDB_NAME={LAB}\n"
    f"DB_USER={CRED['DB_USER']}\nDB_PASSWORD='{SECRET}'\n", encoding="utf-8")
FILES = {
    "settings.py": '''from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings, SettingsConfigDict,
)
class Settings(BaseSettings):
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str
    db_user: str
    db_password: SecretStr
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_file_encoding="utf-8",
    )
settings = Settings()
''',
    "db.py": '''from contextlib import contextmanager
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker
from settings import settings
url = URL.create("postgresql+psycopg",
    username=settings.db_user,
    password=settings.db_password.get_secret_value(),
    host=settings.db_host, port=settings.db_port,
    database=settings.db_name)
engine = create_engine(url, pool_pre_ping=True,
    hide_parameters=True,
    connect_args={"connect_timeout": 5})
SessionLocal = sessionmaker(bind=engine,
    autoflush=False, expire_on_commit=False)
def get_db():
    with SessionLocal() as db:
        yield db
@contextmanager
def transaction(db):
    try:
        yield             # with 안의 조회·변경·flush
        db.commit()       # 정상일 때 한 번 확정
    except Exception:
        db.rollback()     # 오류가 나면 미확정 작업 취소
        raise             # 성공 응답으로 숨기지 않는다.
''',
    "models.py": '''from sqlalchemy import (
    BigInteger, String, Text, Identity,
    ForeignKey, CheckConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column,
)
class Base(DeclarativeBase):
    pass
class Category(Base):
    __tablename__ = "categories"
    __table_args__ = (
        CheckConstraint(
            "char_length(name) BETWEEN 1 AND 30",
            name="categories_name_length"),
        {"schema": "public"},
    )
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True),
        primary_key=True)
    name: Mapped[str] = mapped_column(
        String(30), nullable=False, unique=True)
class Knowledge(Base):
    __tablename__ = "knowledge"
    __table_args__ = (
        CheckConstraint(
            "char_length(title) BETWEEN 1 AND 80",
            name="knowledge_title_length",
        ),
        CheckConstraint(
            "char_length(content) BETWEEN 1 AND 2000",
            name="knowledge_content_length",
        ),
        {"schema": "public"},
    )
    id: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True),
        primary_key=True)
    title: Mapped[str] = mapped_column(
        String(80), nullable=False)
    content: Mapped[str] = mapped_column(
        Text, nullable=False)
    category_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("public.categories.id",
            name="knowledge_category_fk",
            ondelete="RESTRICT"), nullable=False)
''',
    "main.py": '''from typing import Annotated
from fastapi import (
    FastAPI, Depends, HTTPException, Response, Request,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, DBAPIError, OperationalError
from sqlalchemy.orm import Session
from db import get_db, transaction
from models import Knowledge, Category
app = FastAPI(title="Knowledge API · PostgreSQL")
DbSession = Annotated[Session, Depends(get_db)]
@app.get("/health")
def health():
    return {"status": "ok"}
class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)
class KnowledgeRead(KnowledgeWrite):
    id: int
def to_read(row: Knowledge, category: str):
    return KnowledgeRead(
        id=row.id, title=row.title,
        content=row.content, category=category,
    )
def get_or_create_category(db, name):
    category = db.scalar(
        select(Category).where(Category.name == name)
    )
    if category is None:
        category = Category(name=name)
        db.add(category)
        db.flush()  # category.id 확보, 아직 미확정
    return category
def knowledge_or_404(db, knowledge_id):
    row = db.get(Knowledge, knowledge_id)
    if row is None:
        raise HTTPException(404, "Knowledge not found")
    return row
@app.get("/api/knowledge",
         response_model=list[KnowledgeRead])
def list_knowledge(db: DbSession,
                   category: str | None = None):
    stmt = select(Knowledge, Category.name).join(
        Category, Knowledge.category_id == Category.id)
    if category is not None:
        stmt = stmt.where(Category.name == category)
    stmt = stmt.order_by(Knowledge.id)
    return [to_read(row, name)
            for row, name in db.execute(stmt).all()]
@app.get("/api/knowledge/{knowledge_id}",
         response_model=KnowledgeRead)
def read_knowledge(knowledge_id: int, db: DbSession):
    row = knowledge_or_404(db, knowledge_id)
    category = db.get(Category, row.category_id)
    return to_read(row, category.name)
@app.post("/api/knowledge", status_code=201,
          response_model=KnowledgeRead)
def create_knowledge(payload: KnowledgeWrite,
                     db: DbSession):
    with transaction(db):
        category = get_or_create_category(
            db, payload.category)
        row = Knowledge(
            title=payload.title, content=payload.content,
            category_id=category.id)
        db.add(row)
        db.flush()
        result = to_read(row, category.name)
    return result
@app.put("/api/knowledge/{knowledge_id}",
         response_model=KnowledgeRead)
def update_knowledge(knowledge_id: int,
                     payload: KnowledgeWrite,
                     db: DbSession):
    with transaction(db):
        row = knowledge_or_404(db, knowledge_id)
        category = get_or_create_category(
            db, payload.category)
        row.title = payload.title
        row.content = payload.content
        row.category_id = category.id
        db.flush()
        result = to_read(row, category.name)
    return result
@app.delete("/api/knowledge/{knowledge_id}",
            status_code=204)
def delete_knowledge(knowledge_id: int,
                     db: DbSession):
    with transaction(db):
        row = knowledge_or_404(db, knowledge_id)
        db.delete(row)
    return Response(status_code=204)
@app.exception_handler(IntegrityError)
async def integrity_error(request: Request, exc):
    code = getattr(exc.orig, "sqlstate", None)
    diag = getattr(exc.orig, "diag", None)
    name = getattr(diag, "constraint_name", None)
    known = {("23505", "categories_name_key"),
             ("23503", "knowledge_category_fk")}
    status = 409 if (code, name) in known else 500
    detail = ("Data conflict" if status == 409 else
              "Database operation failed")
    return JSONResponse(status_code=status,
                        content={"detail": detail})
@app.exception_handler(DBAPIError)
async def other_db_error(request: Request, exc):
    code = getattr(exc.orig, "sqlstate", None)
    unavailable = (
        isinstance(exc, OperationalError)
        and (str(code or "").startswith("08")
             or bool(exc.connection_invalidated))
    )
    status = 503 if unavailable else 500
    detail = ("Database unavailable" if unavailable
              else "Database operation failed")
    return JSONResponse(status_code=status,
                        content={"detail": detail})
''',
    "check_database.py": '''from sqlalchemy import text
from db import engine
with engine.connect() as conn:
    info = conn.execute(text(
        "SELECT current_database(), current_user"
    )).one()
    count = conn.execute(text(
        "SELECT count(*) FROM public.knowledge"
    )).scalar_one()
print("DB:", info[0], "USER:", info[1])
print("knowledge count:", count)
''',
}
for name, src in FILES.items():
    (PROJ / name).write_text(src, encoding="utf-8")
sys.path.insert(0, str(PROJ))
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

IO = {}
TITLE = "강의 코드를 실제 PostgreSQL 18(노트 전용 DB)에서 실행"
USER = CRED["DB_USER"]


def who(s):
    return scrub(s)


# ---------------------------------------------------------------- 설치 확인
vers = {}
for m, pipname in [("sqlalchemy", "sqlalchemy"), ("psycopg", "psycopg"), ("alembic", "alembic"), ("pydantic_settings", "pydantic-settings"), ("fastapi", "fastapi")]:
    vers[pipname] = importlib.import_module(m).__version__
import psycopg  # noqa: E402
IO["d19-pip-install"] = ("이 노트를 만든 가상환경에서 실제로 확인", [
    ("python -m pip show sqlalchemy psycopg alembic", f"sqlalchemy {vers['sqlalchemy']} · psycopg {vers['psycopg']} · alembic {vers['alembic']}", "강의 예시는 sqlalchemy 2.0 계열 — 이 PC는 2.1이라 노트 결과도 2.1 기준"),
    ("pydantic-settings · fastapi", f"{vers['pydantic-settings']} · {vers['fastapi']}", ""),
    ("import psycopg; psycopg.pq.__impl__", psycopg.pq.__impl__, "[binary]로 깔면 C 확장(binary)으로 PostgreSQL과 통신 — 따로 libpq를 설치할 필요가 없어요"),
])

# ---------------------------------------------------------------- settings · URL
import settings as settings_mod  # noqa: E402
st = settings_mod.settings
from sqlalchemy.engine import make_url  # noqa: E402
fake = "p@ss%w0rd"
u_ok = URL.create("postgresql+psycopg", username="app", password=fake, host="localhost", port=5432, database="knowledge_lab")
try:
    bad = make_url(f"postgresql+psycopg://app:{fake}@localhost:5432/knowledge_lab")
    bad_s = f"host={bad.host!r} · password={bad.password!r}"
except Exception as e:
    bad_s = f"오류: {type(e).__name__}: {e}"
IO["d19-settings"] = (TITLE, [
    ("settings.db_name", repr(st.db_name), ".env의 DB_NAME — 연결 대상"),
    ("settings.db_port", repr(st.db_port), "문자열 '5432'가 int로 바뀜 (타입 힌트대로)"),
    ("settings.db_password", repr(st.db_password), "SecretStr — print해도 별표. 파일 암호화는 아님"),
    ("print(settings)", who(repr(st)).replace("Settings(", "Settings(")[:160], "실수로 출력해도 비밀번호는 가려짐"),
    ("settings.db_password.get_secret_value()", "(실제 비밀번호 문자열 — 여기엔 안 보여 줌)", "URL.create에 넘길 때만 꺼냄"),
])
IO["d19-url-create"] = ("가짜 비밀번호 'p@ss%w0rd'로 실제 실행", [
    ('URL.create(..., password="p@ss%w0rd", host="localhost", ...)', f"host={u_ok.host!r} · password={u_ok.password!r}", "구성요소를 따로 넘겨서 @·%가 있어도 그대로"),
    ("str(url)", str(u_ok), "문자열로 바꾸면 비밀번호는 ***"),
    ('f"postgresql+psycopg://app:{pw}@localhost:5432/..."  (문자열 조립)', bad_s, "비밀번호 속 @를 주소 구분자로 읽어 호스트가 깨져요"),
])

# ---------------------------------------------------------------- check_database.py
r = subprocess.run([PY, "check_database.py"], cwd=PROJ, capture_output=True, text=True, env=ENV, encoding="utf-8")
IO["d19-check-database"] = (TITLE, [
    ("python check_database.py", who(r.stdout.strip() or r.stderr.strip()[-300:]), "연결 대상 DB 이름과 실제 표의 행 수"),
    ("pgAdmin: SELECT count(*) FROM public.knowledge;", sql("SELECT count(*) FROM public.knowledge"), "같은 DB면 같은 숫자"),
])

# ---------------------------------------------------------------- Engine · Session
import db as dbmod  # noqa: E402
import models  # noqa: E402
from sqlalchemy import select, func  # noqa: E402
e = dbmod.engine
s1, s2 = dbmod.SessionLocal(), dbmod.SessionLocal()
g = dbmod.get_db(); gdb = next(g)
in_tx_before = gdb.in_transaction()
gdb.execute(text("SELECT 1"))
in_tx_after = gdb.in_transaction()
try:
    next(g)
except StopIteration:
    pass
IO["d19-engine-session"] = (TITLE, [
    ("type(engine.pool).__name__", type(e.pool).__name__, "연결을 모아 두고 빌려주는 Pool"),
    ("engine.url  (출력)", who(str(e.url)), "비밀번호는 ***로 표시"),
    ("SessionLocal() is SessionLocal()", str(s1 is s2), "부를 때마다 새 Session — 요청마다 하나씩"),
    ("get_db() 첫 next → db.in_transaction()", str(in_tx_before), "Session을 만들기만 해선 DB 트랜잭션이 아직 없음"),
    ("db.execute(text('SELECT 1')) 뒤 in_transaction()", str(in_tx_after), "첫 SQL 때 트랜잭션이 자동으로 시작"),
    ("요청 끝 (generator 종료) → with SessionLocal() 블록 종료", "Session.close()", "close는 확정(commit)이 아니에요"),
])
s1.close(); s2.close()

# ---------------------------------------------------------------- ORM Model → DDL
from sqlalchemy.schema import CreateTable  # noqa: E402
from sqlalchemy.dialects import postgresql  # noqa: E402
ddl_cat = str(CreateTable(models.Category.__table__).compile(dialect=postgresql.dialect())).strip()
ddl_know = str(CreateTable(models.Knowledge.__table__).compile(dialect=postgresql.dialect())).strip()
ddl_cat = re.sub(r"\n\s*\n", "\n", ddl_cat); ddl_know = re.sub(r"\n\s*\n", "\n", ddl_know)
IO["d19-model-category"] = ("Model이 '만약 CREATE한다면' 낼 DDL — 실제로 실행하지는 않음 (CreateTable(...).compile)", [
    ("CreateTable(Category.__table__)", ddl_cat, "어제 손으로 쓴 DDL과 같은 구조"),
    ("Category.__table__.fullname", models.Category.__table__.fullname, '{"schema": "public"}'),
])
IO["d19-model-knowledge"] = ("Model이 '만약 CREATE한다면' 낼 DDL — 실제로 실행하지는 않음", [
    ("CreateTable(Knowledge.__table__)", ddl_know, "CHECK 2개 + FK(RESTRICT) 이름까지 어제와 같아요"),
    ("[c.name for c in Knowledge.__table__.columns]", str([c.name for c in models.Knowledge.__table__.columns]), "DB 쪽 열 이름 — API의 category 문자열은 없음"),
])

# ---------------------------------------------------------------- API (TestClient + 실제 DB)
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
client = TestClient(main.app, raise_server_exceptions=False)


def j(resp, cut=260):
    body = resp.text if resp.content else "(빈 본문)"
    return f"{resp.status_code} {body}"[:cut]


lst = client.get("/api/knowledge")
IO["d19-list"] = (TITLE, [
    ("GET /api/knowledge", j(lst, 400), "SQL로 넣은 S0 3건이 API로 보임 — id 오름차순"),
    ("GET /api/knowledge?category=learning", j(client.get("/api/knowledge", params={"category": "learning"}), 300), "분류 이름으로 거르기 (부록 A07)"),
    ("GET /api/knowledge?category=reference", j(client.get("/api/knowledge", params={"category": "reference"})), "맞는 지식이 없으면 200 + []"),
])

bad_put = client.put("/api/knowledge/1", json={"content": "본문만"})
extra = client.post("/api/knowledge", json={"title": "추가 필드", "content": "extra 테스트", "category": "learning", "author": "누군가"})
extra_id = extra.json().get("id")
IO["d19-schemas"] = (TITLE, [
    ('PUT /api/knowledge/1  {"content": "본문만"}', j(bad_put, 330), "세 필드 모두 필수 → 422 (함수 실행 전)"),
    ('POST {..., "author": "누군가"}  (모르는 필드)', j(extra, 200), "Pydantic 기본: 모르는 필드는 조용히 무시"),
    ("응답 키", str(list(extra.json().keys())), "응답은 이 네 필드만 — category_id는 안 보임"),
])
client.delete(f"/api/knowledge/{extra_id}")

cats_before = sql("SELECT id, name FROM public.categories ORDER BY id")
post = client.post("/api/knowledge", json={"title": "실습 환경 안내", "content": "실행 전에 가상환경을 확인합니다.", "category": "learning"})
pid = post.json()["id"]
post_new = client.post("/api/knowledge", json={"title": "자주 묻는 질문", "content": "포트가 겹치면 8001을 씁니다.", "category": "faq"})
nid = post_new.json()["id"]
IO["d19-post"] = (TITLE, [
    ('POST {"title": "실습 환경 안내", ..., "category": "learning"}', j(post, 300), "201 — id는 DB가 발급"),
    (f"pgAdmin: SELECT k.id, k.title, c.name ... WHERE k.id = {pid}", sql("SELECT k.id, k.title, c.name AS category FROM public.knowledge k JOIN public.categories c ON c.id = k.category_id WHERE k.id = :i", i=pid), "같은 id가 실제 DB 행으로 — API와 SQL이 같은 DB"),
    ('POST {..., "category": "faq"}  (없던 분류)', j(post_new, 300), "분류를 먼저 만들고 지식을 같은 트랜잭션으로"),
    ("분류 표 (POST 전 → 후)", cats_before.split("\n", 1)[1].replace("\n", " · ") + "  →  " + sql("SELECT id, name FROM public.categories ORDER BY id").split("\n", 1)[1].replace("\n", " · "), "faq가 새로 생겼어요"),
])
IO["d19-get-or-create"] = (TITLE, [
    ('"learning"으로 POST 두 번째', sql("SELECT count(*) AS learning_rows FROM public.categories WHERE name = 'learning'"), "이미 있는 분류는 그 ID를 재사용 — 분류가 늘지 않음"),
    ("새 지식의 category_id", sql("SELECT id, category_id FROM public.knowledge WHERE id IN (:a, :b) ORDER BY id", a=pid, b=nid), "외부는 문자열, 안쪽은 번호(FK)"),
])

IO["d19-detail"] = (TITLE, [
    (f"GET /api/knowledge/{pid}", j(client.get(f"/api/knowledge/{pid}"), 300), "db.get(Knowledge, id)로 찾고 분류명을 붙임"),
    ("GET /api/knowledge/99999", j(client.get("/api/knowledge/99999")), "없는 정수 ID → 404 (어제와 같은 약속)"),
    ("GET /api/knowledge/abc", j(client.get("/api/knowledge/abc"), 230), "정수가 아님 → 422 (함수 실행 전)"),
])

cnt0 = sql("SELECT count(*) FROM public.knowledge")
put = client.put(f"/api/knowledge/{pid}", json={"title": "실습 환경 안내", "content": "실행 전에 .venv 활성화를 확인합니다.", "category": "environment"})
put404 = client.put("/api/knowledge/99999", json={"title": "없음", "content": "없는 ID", "category": "brand-new"})
IO["d19-put"] = (TITLE, [
    (f"PUT /api/knowledge/{pid}  (본문·분류 변경)", j(put, 300), "같은 id · 세 필드 교체"),
    (f"pgAdmin: WHERE k.id = {pid}", sql("SELECT k.id, k.content, c.name AS category FROM public.knowledge k JOIN public.categories c ON c.id = k.category_id WHERE k.id = :i", i=pid), "DB 행도 바뀜"),
    ("지식 수 (PUT 전 → 후)", cnt0.split("\n")[1] + " → " + sql("SELECT count(*) FROM public.knowledge").split("\n")[1], "수정은 개수를 바꾸지 않아요"),
    ('PUT /api/knowledge/99999  {"category": "brand-new"}', j(put404), "404"),
    ("SELECT ... WHERE name = 'brand-new'", sql("SELECT id, name FROM public.categories WHERE name = 'brand-new'"), "없는 지식을 먼저 확인해서 새 분류도 안 생김"),
])

dl = client.delete(f"/api/knowledge/{nid}")
IO["d19-delete"] = (TITLE, [
    (f"DELETE /api/knowledge/{nid}", f"{dl.status_code} · 본문 길이 {len(dl.content)}", "204는 본문이 없어요 ({}도 null도 아님)"),
    (f"GET /api/knowledge/{nid}", j(client.get(f"/api/knowledge/{nid}")), "이후 상세 조회는 404"),
    (f"pgAdmin: WHERE id = {nid}", sql("SELECT id FROM public.knowledge WHERE id = :i", i=nid), "DB에서도 0행"),
    ("SELECT name FROM categories WHERE name = 'faq'", sql("SELECT name FROM public.categories WHERE name = 'faq'"), "부모 분류는 남아요 (빈 분류 삭제는 별도 기능)"),
])

# ---------------------------------------------------------------- Session 메서드 (add · flush · commit · rollback · refresh)
from sqlalchemy.exc import IntegrityError, PendingRollbackError  # noqa: E402
K, C = models.Knowledge, models.Category
with dbmod.SessionLocal() as db:
    cat_id = db.scalar(select(C.id).where(C.name == "learning"))
    row = K(title="세션 실험", content="flush와 commit 비교", category_id=cat_id)
    db.add(row)
    id_after_add = row.id
    db.flush()
    id_after_flush = row.id
    other_flush = sql("SELECT count(*) AS 다른_연결에서 FROM public.knowledge WHERE id = :i", i=id_after_flush)
    db.rollback()
    other_rb = sql("SELECT count(*) AS 다른_연결에서 FROM public.knowledge WHERE id = :i", i=id_after_flush)
    row2 = K(title="세션 실험 2", content="이번엔 commit", category_id=cat_id)
    db.add(row2); db.flush(); id2 = row2.id
    db.commit()
    other_cm = sql("SELECT count(*) AS 다른_연결에서 FROM public.knowledge WHERE id = :i", i=id2)
    with lab_engine.begin() as c:
        c.execute(text("UPDATE public.knowledge SET content = 'SQL로 직접 바꿈' WHERE id = :i"), {"i": id2})
    stale = row2.content
    db.refresh(row2)
    fresh = row2.content
IO["d19-session-methods"] = (TITLE, [
    ("db.add(row) 직후 row.id", repr(id_after_add), "등록만 — 아직 SQL을 안 보냄"),
    ("db.flush() 뒤 row.id", repr(id_after_flush), "INSERT를 보내 DB가 id 발급 · 아직 미확정"),
    ("그 순간 다른 연결에서 SELECT", other_flush, "미확정이라 안 보여요"),
    ("db.rollback() 뒤 다른 연결", other_rb, "취소 — 행 없음"),
    ("새로 add → flush → row2.id", repr(id2), f"{id_after_flush}번은 rollback돼도 다시 쓰지 않아요 (시퀀스는 되감기지 않음)"),
    ("db.commit() 뒤 다른 연결", other_cm, "확정 — 이제 보여요"),
    ("다른 곳에서 SQL로 본문 수정 → row2.content", repr(stale), "expire_on_commit=False라 객체는 예전 값을 들고 있음"),
    ("db.refresh(row2) → row2.content", repr(fresh), "DB 값을 다시 읽음 (저장 명령 아님)"),
])
with lab_engine.begin() as c:
    c.execute(text("DELETE FROM public.knowledge WHERE id = :i"), {"i": id2})

# ---------------------------------------------------------------- transaction 도우미 · 원자성
cats_n0 = sql("SELECT count(*) AS categories FROM public.categories")
err_atomic = ""
with dbmod.SessionLocal() as db:
    try:
        with dbmod.transaction(db):
            cat = main.get_or_create_category(db, "atomic-test")
            flushed_cat_id = cat.id
            db.add(K(title="빈 본문", content="", category_id=cat.id))
            db.flush()
    except IntegrityError as exc:
        err_atomic = f"IntegrityError · sqlstate {exc.orig.sqlstate} · {exc.orig.diag.constraint_name}"
# 잘못 나눈 경우: 분류를 먼저 commit
with dbmod.SessionLocal() as db:
    with dbmod.transaction(db):
        main.get_or_create_category(db, "split-commit")
    try:
        with dbmod.transaction(db):
            cid2 = db.scalar(select(C.id).where(C.name == "split-commit"))
            db.add(K(title="빈 본문", content="", category_id=cid2)); db.flush()
    except IntegrityError:
        pass
IO["d19-transaction"] = (TITLE, [
    ("with transaction(db): 새 분류 'atomic-test' flush", f"category.id = {flushed_cat_id}", "id는 받았지만 미확정"),
    ("  같은 with 안에서 content='' 지식 flush", err_atomic, "CHECK 위반 → 도우미가 rollback 후 예외를 다시 던짐"),
    ("SELECT ... WHERE name = 'atomic-test'", sql("SELECT id, name FROM public.categories WHERE name = 'atomic-test'"), "분류도 같이 취소 — 반쪽 저장 없음"),
    ("❌ 분류를 먼저 commit하고 지식은 따로", sql("SELECT id, name FROM public.categories WHERE name = 'split-commit'"), "지식은 실패했는데 새 분류만 남았어요"),
])

# ---------------------------------------------------------------- 오류 처리기 (409 · 500 · 503)
from fastapi.routing import APIRoute  # noqa: E402
IO_err = []
with dbmod.SessionLocal() as db:
    for label, obj in [("분류 'learning' 중복 INSERT", C(name="learning")),
                       ("없는 category_id = 99999", K(title="t", content="c", category_id=99999)),
                       ("content = '' (CHECK)", K(title="t", content="", category_id=cat_id))]:
        try:
            db.add(obj); db.flush()
        except IntegrityError as exc:
            code = exc.orig.sqlstate; name = exc.orig.diag.constraint_name
            status = 409 if (code, name) in {("23505", "categories_name_key"), ("23503", "knowledge_category_fk")} else 500
            IO_err.append((label, f"sqlstate {code} · constraint_name {name}", f"→ 처리기 응답 {status}"))
            db.rollback()
IO["d19-error-handler"] = (TITLE + " · 실제 예외의 진단값", IO_err + [
    ("지식 제목 중복 POST", j(client.post("/api/knowledge", json={"title": "서버 실행", "content": "제목만 같음", "category": "environment"}), 120), "제목은 UNIQUE가 아니라 409가 아니라 201"),
])
# 접속 실패: 존재하지 않는 포트
from sqlalchemy.exc import OperationalError  # noqa: E402
bad_engine = create_engine(URL.create("postgresql+psycopg", username="x", password="x", host="127.0.0.1", port=1, database="x"),
                           connect_args={"connect_timeout": 2})
try:
    with bad_engine.connect():
        pass
except OperationalError as exc:
    code = getattr(exc.orig, "sqlstate", None)
    unavailable = str(code or "").startswith("08") or bool(exc.connection_invalidated)
    IO["d19-connectivity"] = ("빈 포트(127.0.0.1:1)로 실제 접속 시도", [
        ("create_engine(...port=1).connect()", f"{type(exc).__name__} · sqlstate {code!r} · connection_invalidated {exc.connection_invalidated}", "서버가 없으면 DB에 닿지도 못해서 SQLSTATE가 없어요"),
        ("강의 처리기 규칙으로 판정", "503" if unavailable else "500 (Database operation failed)", "규칙이 08* 코드나 연결 무효화만 503으로 봐서, '서버 꺼짐'은 이 규칙으론 500이 돼요 — 환경별 검증이 필요한 이유"),
    ])

# ---------------------------------------------------------------- 같은 Session에서 실패 후 rollback (A11)
with dbmod.SessionLocal() as db:
    try:
        db.add(C(name="learning")); db.flush()
    except IntegrityError as exc:
        code = exc.orig.sqlstate
        try:
            db.scalar(select(func.count(C.id)))
            no_rb = "조회됨"
        except PendingRollbackError as e2:
            no_rb = f"PendingRollbackError"
        db.rollback()
    count_after = db.scalar(select(func.count(C.id)))
IO["d19-same-session"] = (TITLE, [
    ("db.add(Category(name='learning')); db.flush()", f"IntegrityError · exc.orig.sqlstate = {code}", "UNIQUE 위반 (기대한 23505)"),
    ("rollback 없이 같은 db로 조회", no_rb, "실패한 Session은 rollback 전엔 다음 SQL을 거부"),
    ("db.rollback() 후 같은 db로 count", f"category count: {count_after}", "같은 Session이 회복 — 기존 분류 그대로"),
])

# ---------------------------------------------------------------- 테스트 (A14 · p53) : 다른 DB로 get_db 교체
from sqlalchemy.orm import sessionmaker  # noqa: E402
TestSessionLocal = sessionmaker(bind=lab_engine, autoflush=False, expire_on_commit=False)


def override_get_db():
    with TestSessionLocal() as db:
        yield db


main.app.dependency_overrides[dbmod.get_db] = override_get_db
checks = []
try:
    with TestClient(main.app) as tc:
        response = tc.post("/api/knowledge", json={"title": "시험용 안내", "content": "공개 가능한 가상 문장입니다.", "category": "learning"})
        checks.append(("response.status_code == 201", response.status_code == 201))
        created = response.json()
        with TestSessionLocal() as db:
            row = db.get(K, created["id"])
            checks.append(("row is not None", row is not None))
            checks.append(('row.content == created["content"]', row.content == created["content"]))
            cat = db.get(C, row.category_id)
            checks.append(('category.name == created["category"]', cat.name == created["category"]))
            db.delete(row); db.commit()
finally:
    main.app.dependency_overrides.pop(dbmod.get_db, None)
IO["d19-test-db"] = ("dependency_overrides로 get_db를 교체해 실제 실행 (노트에선 같은 노트 전용 DB 사용)", [
    (a, str(b), "") for a, b in checks
] + [("finally: dependency_overrides.pop(get_db)", str(main.app.dependency_overrides), "끝나면 원래 get_db로 복원")])


# ---------------------------------------------------------------- uvicorn 실행 · 재시작 후 보존
def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def run_server(port):
    p = subprocess.Popen([PY, "-m", "uvicorn", "main:app", "--port", str(port)], cwd=PROJ, env=ENV,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); return p
        except Exception:
            time.sleep(0.25)
    return p


def get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
        return r.status, r.read().decode("utf-8")


port = free_port()
srv = run_server(port)
st1, body1 = get(port, f"/api/knowledge/{pid}")
st_docs, _ = get(port, "/docs")
srv.terminate(); srv.wait(timeout=10)
srv = run_server(port)
st2, body2 = get(port, f"/api/knowledge/{pid}")
srv.terminate(); srv.wait(timeout=10)
IO["d19-uvicorn"] = (f"임시 폴더의 강의 코드를 빈 포트({port})로 띄워 실행 (평소엔 8001 등)", [
    ("python -m uvicorn main:app --port <포트>  →  GET /docs", f"{st_docs}", "자동 문서가 열림"),
    (f"GET /api/knowledge/{pid}", f"{st1} {body1}"[:200], "PUT으로 바꾼 본문"),
    ("서버 종료 → 같은 명령으로 재시작 → 같은 GET", f"{st2} {body2}"[:200], "DB에 있으니 재시작해도 그대로"),
])

# ---------------------------------------------------------------- Alembic (별도 연습 DB)
MD = PROJ / "migration_demo"
(MD / "migrations" / "versions").mkdir(parents=True)
(MD / ".env.migration").write_text((PROJ / ".env").read_text(encoding="utf-8").replace(f"DB_NAME={LAB}", f"DB_NAME={MIG}"), encoding="utf-8")
(MD / "alembic.ini").write_text("""[alembic]
script_location = migrations

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console
qualname =

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
""", encoding="utf-8")
(MD / "migrations" / "script.py.mako").write_text('''"""${message}"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
def upgrade():
    ${upgrades if upgrades else "pass"}
def downgrade():
    ${downgrades if downgrades else "pass"}
''', encoding="utf-8")
(MD / "migrations" / "env.py").write_text(f'''from pathlib import Path
import sys
from logging.config import fileConfig
from alembic import context
from sqlalchemy import create_engine, pool, text, URL
fileConfig(context.config.config_file_name)   # 기본 템플릿처럼 alembic.ini의 로그 설정 사용
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
env = {{}}
for ln in (Path(__file__).resolve().parents[1] / ".env.migration").read_text(encoding="utf-8").splitlines():
    if "=" in ln:
        k, v = ln.split("=", 1); env[k] = v.strip().strip("'\\"")
demo_url = URL.create("postgresql+psycopg", username=env["DB_USER"], password=env["DB_PASSWORD"],
                      host=env["DB_HOST"], port=int(env["DB_PORT"]), database=env["DB_NAME"])
import models_mig
target_metadata = models_mig.Base.metadata
connectable = create_engine(demo_url,
                            poolclass=pool.NullPool)
with connectable.connect() as connection:
    actual = connection.execute(text(
        "SELECT current_database()"
    )).scalar_one()
    connection.rollback()  # 점검 SELECT의 tx 종료
    if actual != "{MIG}":
        raise RuntimeError("연습 DB 대상 확인 필요")
    context.configure(connection=connection,
                      target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()
''', encoding="utf-8")
(MD / "migrations" / "versions" / "001_initial.py").write_text('''from alembic import op
import sqlalchemy as sa
revision = "001_initial"
down_revision = None
def upgrade():
    op.create_table("categories",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("name", sa.String(30), nullable=False, unique=True),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 30", name="categories_name_length"),
        schema="public")
    op.create_table("knowledge",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), primary_key=True),
        sa.Column("title", sa.String(80), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("category_id", sa.BigInteger, sa.ForeignKey("public.categories.id", name="knowledge_category_fk", ondelete="RESTRICT"), nullable=False),
        sa.CheckConstraint("char_length(title) BETWEEN 1 AND 80", name="knowledge_title_length"),
        sa.CheckConstraint("char_length(content) BETWEEN 1 AND 2000", name="knowledge_content_length"),
        schema="public")
def downgrade():
    op.drop_table("knowledge", schema="public")
    op.drop_table("categories", schema="public")
''', encoding="utf-8")
(MD / "migrations" / "versions" / "002_source.py").write_text('''from alembic import op
import sqlalchemy as sa
revision = "002_source"
down_revision = "001_initial"
def upgrade():
    op.add_column("knowledge",
        sa.Column("source_label", sa.String(120),
                  nullable=True), schema="public")
def downgrade():
    op.drop_column("knowledge", "source_label",
                   schema="public")
''', encoding="utf-8")
models_src = FILES["models.py"].replace('''            ondelete="RESTRICT"), nullable=False)
''', '''            ondelete="RESTRICT"), nullable=False)
    source_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
''')
(PROJ / "models_mig.py").write_text(models_src, encoding="utf-8")
mig_engine = create_engine(url_for(MIG), hide_parameters=True)


def alembic(*args):
    r = subprocess.run([PY, "-m", "alembic", *args], cwd=MD, capture_output=True, text=True, env=ENV, encoding="utf-8")
    out = (r.stdout + r.stderr).strip()
    out = "\n".join(l for l in out.splitlines() if "Will assume transactional DDL" not in l and "Context impl" not in l)
    out = out.replace(str(MD) + "\\", "")
    return scrub(out) or "(출력 없음)"


def msql(q):
    with mig_engine.connect() as c:
        r = c.execute(text(q)); cols = list(r.keys()); rows = r.all()
    if not rows:
        return " | ".join(cols) + "\n(0행)"
    return "\n".join([" | ".join(cols)] + [" | ".join("NULL" if v is None else str(v) for v in row) for row in rows])


COLS = "SELECT column_name, is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = 'knowledge' ORDER BY ordinal_position"
a_cur0 = alembic("current")
a_hist = alembic("history")
a_up1 = alembic("upgrade", "001_initial")
a_cur1 = alembic("current")
with mig_engine.begin() as c:
    c.execute(text("INSERT INTO public.categories (name) VALUES ('learning')"))
    c.execute(text("INSERT INTO public.knowledge (title, content, category_id) SELECT '가상 안내', '마이그레이션 전 데이터', id FROM public.categories"))
cols1 = msql(COLS)
a_up2 = alembic("upgrade", "head")
a_cur2 = alembic("current")
cols2 = msql(COLS)
rows2 = msql("SELECT id, title, source_label FROM public.knowledge")
with mig_engine.begin() as c:
    c.execute(text("UPDATE public.knowledge SET source_label = '사내 위키'"))
a_down = alembic("downgrade", "001_initial")
cols3 = msql(COLS)
a_cur3 = alembic("current")
a_up3 = alembic("upgrade", "head")
rows3 = msql("SELECT id, title, source_label FROM public.knowledge")
IO["d19-alembic-commands"] = (f"강의 002_source를 노트 전용 연습 DB({MIG})에 실제 적용", [
    ("python -m alembic current  (처음)", a_cur0, "아직 적용된 Revision 없음"),
    ("python -m alembic history", a_hist, "001 → 002 순서"),
    ("python -m alembic upgrade 001_initial", a_up1, "두 테이블 생성"),
    ("python -m alembic current", a_cur1, ""),
    ("(가상 지식 1건 COMMIT 후) 열 목록", cols1, "아직 source_label 없음"),
    ("python -m alembic upgrade head", a_up2, ""),
    ("python -m alembic current", a_cur2, "head = 마지막 Revision"),
    ("information_schema.columns", cols2, "source_label 추가 · NULL 허용"),
    ("SELECT id, title, source_label FROM knowledge", rows2, "기존 행은 그대로, 새 열은 NULL"),
])
IO["d19-revision"] = (f"노트 전용 연습 DB({MIG})에서 되돌림까지 실제 실행", [
    ("UPDATE ... SET source_label = '사내 위키' → downgrade 001_initial", a_down, "002의 downgrade() = drop_column"),
    ("열 목록", cols3, "source_label 열이 사라짐"),
    ("다시 upgrade head → SELECT", rows3, "열은 돌아왔지만 '사내 위키' 값은 복구되지 않아요"),
])

# autogenerate: Model에서 열 이름을 바꾸면?
(PROJ / "models_mig.py").write_text(models_src.replace("source_label", "source_note"), encoding="utf-8")
a_auto = alembic("revision", "--autogenerate", "-m", "rename_source")
gen = sorted((MD / "migrations" / "versions").glob("*rename_source*.py"))
gen_ops = ""
if gen:
    src = gen[0].read_text(encoding="utf-8")
    up = src.split("def downgrade")[0]
    gen_ops = "\n".join(l.strip() for l in up.splitlines() if "op." in l and not l.strip().startswith("#"))
a_stamp_cur = alembic("current")
IO["d19-autogenerate"] = ("Model의 source_label → source_note 로 이름만 바꾸고 autogenerate 실제 실행", [
    ("python -m alembic revision --autogenerate -m rename_source", a_auto, "초안 파일이 생김 — 아직 DB엔 아무 일도 없음"),
    ("생성된 초안의 op 줄", gen_ops, "'이름 바꾸기' 의도가 열 추가 + 열 삭제로 나와요 → 그대로 적용하면 값이 사라짐. FK를 지웠다 다시 만드는 줄도 섞였어요(public 스키마 표기 차이) — 초안은 꼭 읽고 고쳐요"),
    ("python -m alembic current (초안 생성 후)", a_stamp_cur, "초안만으론 DB 이력이 안 바뀜"),
])

# ---------------------------------------------------------------- 정리
engine_dispose = [lab_engine.dispose(), mig_engine.dispose(), dbmod.engine.dispose()]
drop(LAB); drop(MIG)
with admin.connect() as c:
    left = c.execute(text("SELECT datname FROM pg_database WHERE datname IN (:a, :b)"), {"a": LAB, "b": MIG}).all()
admin.dispose()
shutil.rmtree(PROJ, ignore_errors=True)
print("cleanup: 남은 노트 DB", left, "· 임시 폴더 삭제", not PROJ.exists())


# ---------------------------------------------------------------- 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(who(o))}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


for cid, (title, rows) in IO.items():
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i}\n      → {who(o)}".replace("\n", "\n        ") + f"   ({n})")

frag = ROOT / "cmds" / "day19.html"
if frag.exists():
    s = frag.read_text(encoding="utf-8")
    s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
    missing = []
    for cid, (title, rows) in IO.items():
        m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
        if not m:
            missing.append(cid); continue
        sec = m.group(0)
        k = sec.find("</code></pre>\n")
        k = k + len("</code></pre>\n") if k >= 0 else sec.find("</table>\n") + len("</table>\n")
        s = s[:m.start() + k] + box(title, rows) + "\n" + s[m.start() + k:]
    assert SECRET not in s
    frag.write_text(s, encoding="utf-8")
    print("inserted", len(IO) - len(missing), "missing", missing)
