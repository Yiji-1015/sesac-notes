# 백엔드 외전. 실행 방법은 같은 폴더의 README.md에 있어요.
from pathlib import Path
import json
import os

# ① 실행 설정을 한곳에서 읽기
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
config = dotenv_values(ROOT / ".env")
DB_FILE = Path(os.environ.get("SERVICE_DEMO_DB", config.get("SERVICE_DEMO_DB") or "out/service.sqlite3"))
if not DB_FILE.is_absolute():
    DB_FILE = ROOT / DB_FILE
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

# ② DB 준비와 검증을 서버에서 분리하기
from sqlalchemy import create_engine, event, select, func, ForeignKey, String, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

engine = create_engine(
    "sqlite:///" + DB_FILE.as_posix(),
    connect_args={"check_same_thread": False},
    pool_size=2, max_overflow=0, pool_timeout=2,
)

@event.listens_for(engine, "connect")
def enable_foreign_keys(connection, _):
    connection.execute("PRAGMA foreign_keys=ON")

class Base(DeclarativeBase):
    pass

class Note(Base):
    __tablename__ = "notes"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(80), unique=True)
    content: Mapped[str]
    revision: Mapped[int] = mapped_column(default=1)

class Change(Base):
    __tablename__ = "changes"
    id: Mapped[int] = mapped_column(primary_key=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"))
    action: Mapped[str]

SessionLocal = sessionmaker(engine, expire_on_commit=False, autoflush=False)

def prepare():
    Base.metadata.create_all(engine)
    with SessionLocal.begin() as db:
        if not db.scalar(select(func.count()).select_from(Note)):
            note = Note(title="환경 준비", content="같은 Python으로 설치하고 실행해요.")
            db.add(note)
            db.flush()
            db.add(Change(note_id=note.id, action="created"))
    return verify()

def verify():
    with SessionLocal() as db:
        return {
            "notes": db.scalar(select(func.count()).select_from(Note)),
            "changes": db.scalar(select(func.count()).select_from(Change)),
            "orphans": db.scalar(select(func.count()).select_from(Change).where(
                ~Change.note_id.in_(select(Note.id)))),
        }

# ③ 요청마다 Session을 빌리고 돌려주기
def get_db():
    with SessionLocal() as db:
        yield db

# ④ 입력 계약과 저장 담당을 정하기
from pydantic import BaseModel, ConfigDict, Field

class NoteWrite(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)

class NoteRead(NoteWrite):
    id: int
    revision: int

class DomainError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message

def as_dict(note):
    return {"id": note.id, "title": note.title,
            "content": note.content, "revision": note.revision}

def require_note(db, note_id):
    note = db.get(Note, note_id)
    if note is None:
        raise DomainError(404, "NOTE_NOT_FOUND", "해당 노트가 없어요.")
    return note

def record_change(db, note, action):
    db.add(Change(note_id=note.id, action=action))

def insert_note(db, payload):
    note = Note(**payload.model_dump())
    db.add(note)
    db.flush()
    return note

def replace_note(db, note, payload):
    note.title, note.content = payload.title, payload.content
    note.revision += 1

def remove_note_row(db, note):
    db.delete(note)

# ⑤ Service가 트랜잭션을 책임지기
from sqlalchemy.exc import IntegrityError

def create_note(db, payload):
    try:
        with db.begin():
            note = insert_note(db, payload)
            record_change(db, note, "created")
            result = as_dict(note)
        return result
    except IntegrityError as exc:
        # 이 SQLite 예제에서 확인한 제목 UNIQUE 위반만 업무 오류로 바꿔요.
        if "UNIQUE constraint failed: notes.title" not in str(exc.orig):
            raise
        raise DomainError(409, "TITLE_CONFLICT", "같은 제목이 이미 있어요.") from exc

def update_note(db, note_id, payload):
    try:
        with db.begin():
            note = require_note(db, note_id)
            replace_note(db, note, payload)
            record_change(db, note, "updated")
            result = as_dict(note)
        return result
    except IntegrityError as exc:
        if "UNIQUE constraint failed: notes.title" not in str(exc.orig):
            raise
        raise DomainError(409, "TITLE_CONFLICT", "같은 제목이 이미 있어요.") from exc

def delete_note(db, note_id):
    with db.begin():
        remove_note_row(db, require_note(db, note_id))

# ⑥ Router와 Health를 앱에 연결하기
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Depends, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        engine.dispose()

app = FastAPI(title="백엔드 외전: 작은 노트 서비스", lifespan=lifespan)
router = APIRouter(prefix="/notes", tags=["notes"])

@app.exception_handler(DomainError)
async def domain_error(request: Request, exc: DomainError):
    return JSONResponse(status_code=exc.status, content={
        "error": {"code": exc.code, "message": exc.message}})

@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "error": {"code": "INVALID_INPUT", "message": "입력 필드를 확인해 주세요."}})

@app.get("/live")
def live():
    return {"status": "ok"}

@app.get("/health")
def health():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            db.execute(select(Note.id).limit(1))
            db.execute(select(Change.id).limit(1))
        return {"status": "ok", "db": "ready"}
    except SQLAlchemyError:
        raise DomainError(503, "DB_UNAVAILABLE", "DB 연결과 테이블 준비를 확인해 주세요.")

@router.get("", response_model=list[NoteRead])
def list_notes(db=Depends(get_db)):
    return [as_dict(note) for note in db.scalars(select(Note).order_by(Note.id))]

@router.get("/{note_id}", response_model=NoteRead)
def read_note(note_id: int, db=Depends(get_db)):
    return as_dict(require_note(db, note_id))

@router.post("", status_code=201, response_model=NoteRead)
def post_note(payload: NoteWrite, db=Depends(get_db)):
    return create_note(db, payload)

@router.put("/{note_id}", response_model=NoteRead)
def put_note(note_id: int, payload: NoteWrite, db=Depends(get_db)):
    return update_note(db, note_id, payload)

@router.delete("/{note_id}", status_code=204)
def remove_note(note_id: int, db=Depends(get_db)):
    delete_note(db, note_id)
    return Response(status_code=204)

app.include_router(router)

# ⑦ 클라이언트의 검증 순서를 파일로 공유하기
def export_postman():
    def request(name, method, path, expected, checks=(), body=None, before=()):
        req = {"method": method, "url": "{{base_url}}" + path}
        if body is not None:
            req["header"] = [{"key": "Content-Type", "value": "application/json"}]
            req["body"] = {"mode": "raw", "raw": json.dumps(body, ensure_ascii=False),
                           "options": {"raw": {"language": "json"}}}
        scripts = [{"listen": "test", "script": {"type": "text/javascript", "exec": [
            f'pm.test("HTTP {expected}", () => pm.response.to.have.status({expected}));',
            *checks]}}]
        if before:
            scripts.insert(0, {"listen": "prerequest", "script": {
                "type": "text/javascript", "exec": list(before)}})
        return {"name": name, "request": req, "event": scripts}

    payload = {"title": "DB 세션을 서비스에 연결하기", "content": "요청마다 Session을 열고 닫아요."}
    changed = {**payload, "content": "Service가 저장과 기록을 함께 확정해요."}
    remember = [
        'pm.test("입력값과 응답 일치", () => {',
        '  const b = JSON.parse(pm.variables.replaceIn(pm.request.body.raw));',
        '  const r = pm.response.json();',
        '  pm.expect(r.title).to.eql(b.title); pm.expect(r.content).to.eql(b.content);',
        '});',
        'if (pm.response.code === 201 || pm.response.code === 200) {',
        '  pm.environment.set("created_id", String(pm.response.json().id));',
        '  pm.environment.set("expected_note", JSON.stringify(pm.response.json()));',
        '}']
    same = ['pm.test("다른 요청으로 다시 읽은 저장값 일치", () => {',
            '  pm.expect(pm.response.json()).to.eql(JSON.parse(pm.environment.get("expected_note")));',
            '});']
    has_id = ['if (!pm.environment.get("created_id")) { throw new Error("먼저 등록 요청을 성공시켜 주세요."); }']
    items = [
        request("01 DB 준비 확인", "GET", "/health", 200, [
            'pm.test("DB ready", () => pm.expect(pm.response.json().db).to.eql("ready"));']),
        request("02 시작 상태 기록", "GET", "/notes", 200, [
            'pm.test("시작 데이터 배열", () => pm.expect(pm.response.json()).to.be.an("array"));',
            'pm.environment.set("baseline_count", String(pm.response.json().length));'], before=[
            'if (pm.environment.get("created_id")) { throw new Error("이전 생성 ID를 먼저 확인하고 정리해 주세요."); }']),
        request("03 등록하고 ID 저장", "POST", "/notes", 201, remember + [
            'pm.test("첫 revision", () => pm.expect(pm.response.json().revision).to.eql(1));'], payload),
        request("04 저장값 재조회", "GET", "/notes/{{created_id}}", 200, same, before=has_id),
        request("05 중복 제목 거부", "POST", "/notes", 409, [
            'pm.test("충돌 코드", () => pm.expect(pm.response.json().error.code).to.eql("TITLE_CONFLICT"));'], payload),
        request("06 잘못된 수정 거부", "PUT", "/notes/{{created_id}}", 422, [
            'pm.test("입력 오류 코드", () => pm.expect(pm.response.json().error.code).to.eql("INVALID_INPUT"));'],
            {"title": payload["title"]}, before=has_id),
        request("07 실패 후 원래 값 유지", "GET", "/notes/{{created_id}}", 200, same, before=has_id),
        request("08 정상 수정", "PUT", "/notes/{{created_id}}", 200, remember + [
            'pm.test("revision 증가", () => pm.expect(pm.response.json().revision).to.eql(2));'], changed, before=has_id),
        request("09 수정값 재조회", "GET", "/notes/{{created_id}}", 200, same, before=has_id),
        request("10 이번 생성 데이터 삭제", "DELETE", "/notes/{{created_id}}", 204, [
            'pm.test("본문 없음", () => pm.expect(pm.response.text()).to.eql(""));'], before=has_id),
        request("11 삭제 후 404 확인", "GET", "/notes/{{created_id}}", 404, [
            'pm.test("미존재 코드", () => pm.expect(pm.response.json().error.code).to.eql("NOTE_NOT_FOUND"));',
            'if (pm.response.code === 404) { pm.environment.unset("created_id"); pm.environment.unset("expected_note"); }'], before=has_id),
        request("12 시작 수량 복귀", "GET", "/notes", 200, [
            'pm.test("시작 수량과 일치", () => pm.expect(pm.response.json().length).to.eql(Number(pm.environment.get("baseline_count"))));']),
    ]
    collection = {"info": {"name": "백엔드 외전 - 노트 서비스", "schema":
        "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"}, "item": items}
    environment = {"id": "c8416540-f1da-4ed8-a57a-2fa6b3aa71e7", "name": "노트 서비스 - Local",
        "values": [{"key": key, "value": value, "enabled": True, "type": "default"}
                   for key, value in [("base_url", "http://127.0.0.1:8015"), ("created_id", ""),
                                      ("expected_note", ""), ("baseline_count", "")]],
        "_postman_variable_scope": "environment"}
    for name, data in [("service.postman_collection.json", collection),
                       ("service.postman_environment.json", environment)]:
        (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return collection, environment

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "verify", "postman", "serve"])
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8015)
    elif args.command == "postman":
        export_postman()
        print("Postman Collection / Environment JSON 생성 완료")
    else:
        print(json.dumps(prepare() if args.command == "prepare" else verify(), ensure_ascii=False))
