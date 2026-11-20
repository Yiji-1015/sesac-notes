window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import os",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 실행 설정을 한곳에서 읽기",
   "s": 1
  },
  {
   "t": "from dotenv import dotenv_values",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "ROOT = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "config = dotenv_values(ROOT / \".env\")",
   "s": 1
  },
  {
   "t": "DB_FILE = Path(os.environ.get(\"SERVICE_DEMO_DB\", config.get(\"SERVICE_DEMO_DB\") or \"out/service.sqlite3\"))",
   "s": 1
  },
  {
   "t": "if not DB_FILE.is_absolute():",
   "s": 1
  },
  {
   "t": "    DB_FILE = ROOT / DB_FILE",
   "s": 1
  },
  {
   "t": "DB_FILE.parent.mkdir(parents=True, exist_ok=True)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② DB 준비와 검증을 서버에서 분리하기",
   "s": 2
  },
  {
   "t": "from sqlalchemy import create_engine, event, select, func, ForeignKey, String, text",
   "s": 2
  },
  {
   "t": "from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "engine = create_engine(",
   "s": 2
  },
  {
   "t": "    \"sqlite:///\" + DB_FILE.as_posix(),",
   "s": 2
  },
  {
   "t": "    connect_args={\"check_same_thread\": False},",
   "s": 2
  },
  {
   "t": "    pool_size=2, max_overflow=0, pool_timeout=2,",
   "s": 2
  },
  {
   "t": ")",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "@event.listens_for(engine, \"connect\")",
   "s": 2
  },
  {
   "t": "def enable_foreign_keys(connection, _):",
   "s": 2
  },
  {
   "t": "    connection.execute(\"PRAGMA foreign_keys=ON\")",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "class Base(DeclarativeBase):",
   "s": 2
  },
  {
   "t": "    pass",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "class Note(Base):",
   "s": 2
  },
  {
   "t": "    __tablename__ = \"notes\"",
   "s": 2
  },
  {
   "t": "    id: Mapped[int] = mapped_column(primary_key=True)",
   "s": 2
  },
  {
   "t": "    title: Mapped[str] = mapped_column(String(80), unique=True)",
   "s": 2
  },
  {
   "t": "    content: Mapped[str]",
   "s": 2
  },
  {
   "t": "    revision: Mapped[int] = mapped_column(default=1)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "class Change(Base):",
   "s": 2
  },
  {
   "t": "    __tablename__ = \"changes\"",
   "s": 2
  },
  {
   "t": "    id: Mapped[int] = mapped_column(primary_key=True)",
   "s": 2
  },
  {
   "t": "    note_id: Mapped[int] = mapped_column(ForeignKey(\"notes.id\", ondelete=\"CASCADE\"))",
   "s": 2
  },
  {
   "t": "    action: Mapped[str]",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "SessionLocal = sessionmaker(engine, expire_on_commit=False, autoflush=False)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def prepare():",
   "s": 2
  },
  {
   "t": "    Base.metadata.create_all(engine)",
   "s": 2
  },
  {
   "t": "    with SessionLocal.begin() as db:",
   "s": 2
  },
  {
   "t": "        if not db.scalar(select(func.count()).select_from(Note)):",
   "s": 2
  },
  {
   "t": "            note = Note(title=\"환경 준비\", content=\"같은 Python으로 설치하고 실행해요.\")",
   "s": 2
  },
  {
   "t": "            db.add(note)",
   "s": 2
  },
  {
   "t": "            db.flush()",
   "s": 2
  },
  {
   "t": "            db.add(Change(note_id=note.id, action=\"created\"))",
   "s": 2
  },
  {
   "t": "    return verify()",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def verify():",
   "s": 2
  },
  {
   "t": "    with SessionLocal() as db:",
   "s": 2
  },
  {
   "t": "        return {",
   "s": 2
  },
  {
   "t": "            \"notes\": db.scalar(select(func.count()).select_from(Note)),",
   "s": 2
  },
  {
   "t": "            \"changes\": db.scalar(select(func.count()).select_from(Change)),",
   "s": 2
  },
  {
   "t": "            \"orphans\": db.scalar(select(func.count()).select_from(Change).where(",
   "s": 2
  },
  {
   "t": "                ~Change.note_id.in_(select(Note.id)))),",
   "s": 2
  },
  {
   "t": "        }",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 요청마다 Session을 빌리고 돌려주기",
   "s": 3
  },
  {
   "t": "def get_db():",
   "s": 3
  },
  {
   "t": "    with SessionLocal() as db:",
   "s": 3
  },
  {
   "t": "        yield db",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 입력 계약과 저장 담당을 정하기",
   "s": 4
  },
  {
   "t": "from pydantic import BaseModel, ConfigDict, Field",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "class NoteWrite(BaseModel):",
   "s": 4
  },
  {
   "t": "    model_config = ConfigDict(str_strip_whitespace=True, extra=\"forbid\")",
   "s": 4
  },
  {
   "t": "    title: str = Field(min_length=1, max_length=80)",
   "s": 4
  },
  {
   "t": "    content: str = Field(min_length=1, max_length=2000)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "class NoteRead(NoteWrite):",
   "s": 4
  },
  {
   "t": "    id: int",
   "s": 4
  },
  {
   "t": "    revision: int",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "class DomainError(Exception):",
   "s": 4
  },
  {
   "t": "    def __init__(self, status, code, message):",
   "s": 4
  },
  {
   "t": "        super().__init__(message)",
   "s": 4
  },
  {
   "t": "        self.status, self.code, self.message = status, code, message",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def as_dict(note):",
   "s": 4
  },
  {
   "t": "    return {\"id\": note.id, \"title\": note.title,",
   "s": 4
  },
  {
   "t": "            \"content\": note.content, \"revision\": note.revision}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def require_note(db, note_id):",
   "s": 4
  },
  {
   "t": "    note = db.get(Note, note_id)",
   "s": 4
  },
  {
   "t": "    if note is None:",
   "s": 4
  },
  {
   "t": "        raise DomainError(404, \"NOTE_NOT_FOUND\", \"해당 노트가 없어요.\")",
   "s": 4
  },
  {
   "t": "    return note",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def record_change(db, note, action):",
   "s": 4
  },
  {
   "t": "    db.add(Change(note_id=note.id, action=action))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def insert_note(db, payload):",
   "s": 4
  },
  {
   "t": "    note = Note(**payload.model_dump())",
   "s": 4
  },
  {
   "t": "    db.add(note)",
   "s": 4
  },
  {
   "t": "    db.flush()",
   "s": 4
  },
  {
   "t": "    return note",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def replace_note(db, note, payload):",
   "s": 4
  },
  {
   "t": "    note.title, note.content = payload.title, payload.content",
   "s": 4
  },
  {
   "t": "    note.revision += 1",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def remove_note_row(db, note):",
   "s": 4
  },
  {
   "t": "    db.delete(note)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ Service가 트랜잭션을 책임지기",
   "s": 5
  },
  {
   "t": "from sqlalchemy.exc import IntegrityError",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def create_note(db, payload):",
   "s": 5
  },
  {
   "t": "    try:",
   "s": 5
  },
  {
   "t": "        with db.begin():",
   "s": 5
  },
  {
   "t": "            note = insert_note(db, payload)",
   "s": 5
  },
  {
   "t": "            record_change(db, note, \"created\")",
   "s": 5
  },
  {
   "t": "            result = as_dict(note)",
   "s": 5
  },
  {
   "t": "        return result",
   "s": 5
  },
  {
   "t": "    except IntegrityError as exc:",
   "s": 5
  },
  {
   "t": "        # 이 SQLite 예제에서 확인한 제목 UNIQUE 위반만 업무 오류로 바꿔요.",
   "s": 5
  },
  {
   "t": "        if \"UNIQUE constraint failed: notes.title\" not in str(exc.orig):",
   "s": 5
  },
  {
   "t": "            raise",
   "s": 5
  },
  {
   "t": "        raise DomainError(409, \"TITLE_CONFLICT\", \"같은 제목이 이미 있어요.\") from exc",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def update_note(db, note_id, payload):",
   "s": 5
  },
  {
   "t": "    try:",
   "s": 5
  },
  {
   "t": "        with db.begin():",
   "s": 5
  },
  {
   "t": "            note = require_note(db, note_id)",
   "s": 5
  },
  {
   "t": "            replace_note(db, note, payload)",
   "s": 5
  },
  {
   "t": "            record_change(db, note, \"updated\")",
   "s": 5
  },
  {
   "t": "            result = as_dict(note)",
   "s": 5
  },
  {
   "t": "        return result",
   "s": 5
  },
  {
   "t": "    except IntegrityError as exc:",
   "s": 5
  },
  {
   "t": "        if \"UNIQUE constraint failed: notes.title\" not in str(exc.orig):",
   "s": 5
  },
  {
   "t": "            raise",
   "s": 5
  },
  {
   "t": "        raise DomainError(409, \"TITLE_CONFLICT\", \"같은 제목이 이미 있어요.\") from exc",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def delete_note(db, note_id):",
   "s": 5
  },
  {
   "t": "    with db.begin():",
   "s": 5
  },
  {
   "t": "        remove_note_row(db, require_note(db, note_id))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ Router와 Health를 앱에 연결하기",
   "s": 6
  },
  {
   "t": "from contextlib import asynccontextmanager",
   "s": 6
  },
  {
   "t": "from fastapi import FastAPI, APIRouter, Depends, Request, Response",
   "s": 6
  },
  {
   "t": "from fastapi.exceptions import RequestValidationError",
   "s": 6
  },
  {
   "t": "from fastapi.responses import JSONResponse",
   "s": 6
  },
  {
   "t": "from sqlalchemy.exc import SQLAlchemyError",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@asynccontextmanager",
   "s": 6
  },
  {
   "t": "async def lifespan(app):",
   "s": 6
  },
  {
   "t": "    try:",
   "s": 6
  },
  {
   "t": "        yield",
   "s": 6
  },
  {
   "t": "    finally:",
   "s": 6
  },
  {
   "t": "        engine.dispose()",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "app = FastAPI(title=\"백엔드 외전: 작은 노트 서비스\", lifespan=lifespan)",
   "s": 6
  },
  {
   "t": "router = APIRouter(prefix=\"/notes\", tags=[\"notes\"])",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@app.exception_handler(DomainError)",
   "s": 6
  },
  {
   "t": "async def domain_error(request: Request, exc: DomainError):",
   "s": 6
  },
  {
   "t": "    return JSONResponse(status_code=exc.status, content={",
   "s": 6
  },
  {
   "t": "        \"error\": {\"code\": exc.code, \"message\": exc.message}})",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@app.exception_handler(RequestValidationError)",
   "s": 6
  },
  {
   "t": "async def invalid_input(request: Request, exc: RequestValidationError):",
   "s": 6
  },
  {
   "t": "    return JSONResponse(status_code=422, content={",
   "s": 6
  },
  {
   "t": "        \"error\": {\"code\": \"INVALID_INPUT\", \"message\": \"입력 필드를 확인해 주세요.\"}})",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@app.get(\"/live\")",
   "s": 6
  },
  {
   "t": "def live():",
   "s": 6
  },
  {
   "t": "    return {\"status\": \"ok\"}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@app.get(\"/health\")",
   "s": 6
  },
  {
   "t": "def health():",
   "s": 6
  },
  {
   "t": "    try:",
   "s": 6
  },
  {
   "t": "        with SessionLocal() as db:",
   "s": 6
  },
  {
   "t": "            db.execute(text(\"SELECT 1\"))",
   "s": 6
  },
  {
   "t": "            db.execute(select(Note.id).limit(1))",
   "s": 6
  },
  {
   "t": "            db.execute(select(Change.id).limit(1))",
   "s": 6
  },
  {
   "t": "        return {\"status\": \"ok\", \"db\": \"ready\"}",
   "s": 6
  },
  {
   "t": "    except SQLAlchemyError:",
   "s": 6
  },
  {
   "t": "        raise DomainError(503, \"DB_UNAVAILABLE\", \"DB 연결과 테이블 준비를 확인해 주세요.\")",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@router.get(\"\", response_model=list[NoteRead])",
   "s": 6
  },
  {
   "t": "def list_notes(db=Depends(get_db)):",
   "s": 6
  },
  {
   "t": "    return [as_dict(note) for note in db.scalars(select(Note).order_by(Note.id))]",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@router.get(\"/{note_id}\", response_model=NoteRead)",
   "s": 6
  },
  {
   "t": "def read_note(note_id: int, db=Depends(get_db)):",
   "s": 6
  },
  {
   "t": "    return as_dict(require_note(db, note_id))",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@router.post(\"\", status_code=201, response_model=NoteRead)",
   "s": 6
  },
  {
   "t": "def post_note(payload: NoteWrite, db=Depends(get_db)):",
   "s": 6
  },
  {
   "t": "    return create_note(db, payload)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@router.put(\"/{note_id}\", response_model=NoteRead)",
   "s": 6
  },
  {
   "t": "def put_note(note_id: int, payload: NoteWrite, db=Depends(get_db)):",
   "s": 6
  },
  {
   "t": "    return update_note(db, note_id, payload)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "@router.delete(\"/{note_id}\", status_code=204)",
   "s": 6
  },
  {
   "t": "def remove_note(note_id: int, db=Depends(get_db)):",
   "s": 6
  },
  {
   "t": "    delete_note(db, note_id)",
   "s": 6
  },
  {
   "t": "    return Response(status_code=204)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "app.include_router(router)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 클라이언트의 검증 순서를 파일로 공유하기",
   "s": 7
  },
  {
   "t": "def export_postman():",
   "s": 7
  },
  {
   "t": "    def request(name, method, path, expected, checks=(), body=None, before=()):",
   "s": 7
  },
  {
   "t": "        req = {\"method\": method, \"url\": \"{{base_url}}\" + path}",
   "s": 7
  },
  {
   "t": "        if body is not None:",
   "s": 7
  },
  {
   "t": "            req[\"header\"] = [{\"key\": \"Content-Type\", \"value\": \"application/json\"}]",
   "s": 7
  },
  {
   "t": "            req[\"body\"] = {\"mode\": \"raw\", \"raw\": json.dumps(body, ensure_ascii=False),",
   "s": 7
  },
  {
   "t": "                           \"options\": {\"raw\": {\"language\": \"json\"}}}",
   "s": 7
  },
  {
   "t": "        scripts = [{\"listen\": \"test\", \"script\": {\"type\": \"text/javascript\", \"exec\": [",
   "s": 7
  },
  {
   "t": "            f'pm.test(\"HTTP {expected}\", () => pm.response.to.have.status({expected}));',",
   "s": 7
  },
  {
   "t": "            *checks]}}]",
   "s": 7
  },
  {
   "t": "        if before:",
   "s": 7
  },
  {
   "t": "            scripts.insert(0, {\"listen\": \"prerequest\", \"script\": {",
   "s": 7
  },
  {
   "t": "                \"type\": \"text/javascript\", \"exec\": list(before)}})",
   "s": 7
  },
  {
   "t": "        return {\"name\": name, \"request\": req, \"event\": scripts}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "    payload = {\"title\": \"DB 세션을 서비스에 연결하기\", \"content\": \"요청마다 Session을 열고 닫아요.\"}",
   "s": 7
  },
  {
   "t": "    changed = {**payload, \"content\": \"Service가 저장과 기록을 함께 확정해요.\"}",
   "s": 7
  },
  {
   "t": "    remember = [",
   "s": 7
  },
  {
   "t": "        'pm.test(\"입력값과 응답 일치\", () => {',",
   "s": 7
  },
  {
   "t": "        '  const b = JSON.parse(pm.variables.replaceIn(pm.request.body.raw));',",
   "s": 7
  },
  {
   "t": "        '  const r = pm.response.json();',",
   "s": 7
  },
  {
   "t": "        '  pm.expect(r.title).to.eql(b.title); pm.expect(r.content).to.eql(b.content);',",
   "s": 7
  },
  {
   "t": "        '});',",
   "s": 7
  },
  {
   "t": "        'if (pm.response.code === 201 || pm.response.code === 200) {',",
   "s": 7
  },
  {
   "t": "        '  pm.environment.set(\"created_id\", String(pm.response.json().id));',",
   "s": 7
  },
  {
   "t": "        '  pm.environment.set(\"expected_note\", JSON.stringify(pm.response.json()));',",
   "s": 7
  },
  {
   "t": "        '}']",
   "s": 7
  },
  {
   "t": "    same = ['pm.test(\"다른 요청으로 다시 읽은 저장값 일치\", () => {',",
   "s": 7
  },
  {
   "t": "            '  pm.expect(pm.response.json()).to.eql(JSON.parse(pm.environment.get(\"expected_note\")));',",
   "s": 7
  },
  {
   "t": "            '});']",
   "s": 7
  },
  {
   "t": "    has_id = ['if (!pm.environment.get(\"created_id\")) { throw new Error(\"먼저 등록 요청을 성공시켜 주세요.\"); }']",
   "s": 7
  },
  {
   "t": "    items = [",
   "s": 7
  },
  {
   "t": "        request(\"01 DB 준비 확인\", \"GET\", \"/health\", 200, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"DB ready\", () => pm.expect(pm.response.json().db).to.eql(\"ready\"));']),",
   "s": 7
  },
  {
   "t": "        request(\"02 시작 상태 기록\", \"GET\", \"/notes\", 200, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"시작 데이터 배열\", () => pm.expect(pm.response.json()).to.be.an(\"array\"));',",
   "s": 7
  },
  {
   "t": "            'pm.environment.set(\"baseline_count\", String(pm.response.json().length));'], before=[",
   "s": 7
  },
  {
   "t": "            'if (pm.environment.get(\"created_id\")) { throw new Error(\"이전 생성 ID를 먼저 확인하고 정리해 주세요.\"); }']),",
   "s": 7
  },
  {
   "t": "        request(\"03 등록하고 ID 저장\", \"POST\", \"/notes\", 201, remember + [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"첫 revision\", () => pm.expect(pm.response.json().revision).to.eql(1));'], payload),",
   "s": 7
  },
  {
   "t": "        request(\"04 저장값 재조회\", \"GET\", \"/notes/{{created_id}}\", 200, same, before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"05 중복 제목 거부\", \"POST\", \"/notes\", 409, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"충돌 코드\", () => pm.expect(pm.response.json().error.code).to.eql(\"TITLE_CONFLICT\"));'], payload),",
   "s": 7
  },
  {
   "t": "        request(\"06 잘못된 수정 거부\", \"PUT\", \"/notes/{{created_id}}\", 422, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"입력 오류 코드\", () => pm.expect(pm.response.json().error.code).to.eql(\"INVALID_INPUT\"));'],",
   "s": 7
  },
  {
   "t": "            {\"title\": payload[\"title\"]}, before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"07 실패 후 원래 값 유지\", \"GET\", \"/notes/{{created_id}}\", 200, same, before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"08 정상 수정\", \"PUT\", \"/notes/{{created_id}}\", 200, remember + [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"revision 증가\", () => pm.expect(pm.response.json().revision).to.eql(2));'], changed, before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"09 수정값 재조회\", \"GET\", \"/notes/{{created_id}}\", 200, same, before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"10 이번 생성 데이터 삭제\", \"DELETE\", \"/notes/{{created_id}}\", 204, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"본문 없음\", () => pm.expect(pm.response.text()).to.eql(\"\"));'], before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"11 삭제 후 404 확인\", \"GET\", \"/notes/{{created_id}}\", 404, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"미존재 코드\", () => pm.expect(pm.response.json().error.code).to.eql(\"NOTE_NOT_FOUND\"));',",
   "s": 7
  },
  {
   "t": "            'if (pm.response.code === 404) { pm.environment.unset(\"created_id\"); pm.environment.unset(\"expected_note\"); }'], before=has_id),",
   "s": 7
  },
  {
   "t": "        request(\"12 시작 수량 복귀\", \"GET\", \"/notes\", 200, [",
   "s": 7
  },
  {
   "t": "            'pm.test(\"시작 수량과 일치\", () => pm.expect(pm.response.json().length).to.eql(Number(pm.environment.get(\"baseline_count\"))));']),",
   "s": 7
  },
  {
   "t": "    ]",
   "s": 7
  },
  {
   "t": "    collection = {\"info\": {\"name\": \"백엔드 외전 - 노트 서비스\", \"schema\":",
   "s": 7
  },
  {
   "t": "        \"https://schema.getpostman.com/json/collection/v2.1.0/collection.json\"}, \"item\": items}",
   "s": 7
  },
  {
   "t": "    environment = {\"id\": \"c8416540-f1da-4ed8-a57a-2fa6b3aa71e7\", \"name\": \"노트 서비스 - Local\",",
   "s": 7
  },
  {
   "t": "        \"values\": [{\"key\": key, \"value\": value, \"enabled\": True, \"type\": \"default\"}",
   "s": 7
  },
  {
   "t": "                   for key, value in [(\"base_url\", \"http://127.0.0.1:8015\"), (\"created_id\", \"\"),",
   "s": 7
  },
  {
   "t": "                                      (\"expected_note\", \"\"), (\"baseline_count\", \"\")]],",
   "s": 7
  },
  {
   "t": "        \"_postman_variable_scope\": \"environment\"}",
   "s": 7
  },
  {
   "t": "    for name, data in [(\"service.postman_collection.json\", collection),",
   "s": 7
  },
  {
   "t": "                       (\"service.postman_environment.json\", environment)]:",
   "s": 7
  },
  {
   "t": "        (ROOT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + \"\\n\", encoding=\"utf-8\")",
   "s": 7
  },
  {
   "t": "    return collection, environment",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "if __name__ == \"__main__\":",
   "s": 7
  },
  {
   "t": "    import argparse",
   "s": 7
  },
  {
   "t": "    parser = argparse.ArgumentParser()",
   "s": 7
  },
  {
   "t": "    parser.add_argument(\"command\", choices=[\"prepare\", \"verify\", \"postman\", \"serve\"])",
   "s": 7
  },
  {
   "t": "    args = parser.parse_args()",
   "s": 7
  },
  {
   "t": "    if args.command == \"serve\":",
   "s": 7
  },
  {
   "t": "        import uvicorn",
   "s": 7
  },
  {
   "t": "        uvicorn.run(app, host=\"127.0.0.1\", port=8015)",
   "s": 7
  },
  {
   "t": "    elif args.command == \"postman\":",
   "s": 7
  },
  {
   "t": "        export_postman()",
   "s": 7
  },
  {
   "t": "        print(\"Postman Collection / Environment JSON 생성 완료\")",
   "s": 7
  },
  {
   "t": "    else:",
   "s": 7
  },
  {
   "t": "        print(json.dumps(prepare() if args.command == \"prepare\" else verify(), ensure_ascii=False))",
   "s": 7
  }
 ],
 "names": {
  "1": "실행 설정을 한곳에서 읽기",
  "2": "DB 준비와 검증을 서버에서 분리하기",
  "3": "요청마다 Session을 빌리고 돌려주기",
  "4": "입력 계약과 저장 담당을 정하기",
  "5": "Service가 트랜잭션을 책임지기",
  "6": "Router와 Health를 앱에 연결하기",
  "7": "클라이언트의 검증 순서를 파일로 공유하기"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "임시 폴더에서 설정 읽기",
   "lines": [
    [
     "DB 파일: out/service.sqlite3",
     1
    ]
   ],
   "notes": [
    "이번 실행의 service.py가 있는 임시 폴더 아래예요. 실제 사용자 DB를 읽거나 바꾸지 않아요."
   ]
  },
  "s2": {
   "title": "준비 후 실제 DB를 다시 조회",
   "lines": [
    [
     "{\"notes\": 1, \"changes\": 1, \"orphans\": 0}",
     1
    ],
    [
     "prepare를 한 번 더 실행: {\"notes\": 1, \"changes\": 1, \"orphans\": 0}",
     2
    ]
   ],
   "notes": [
    "노트 1건, 연결된 변경 기록 1건이에요. orphans는 부모 노트가 없는 기록 수예요.",
    "기존 데이터를 지우지 않고, 빈 테이블에만 시작 데이터를 넣었어요. 스키마 변경 기능은 아니에요."
   ]
  },
  "s3": {
   "title": "두 요청의 Session 수명 실험",
   "lines": [
    [
     "서로 다른 Session인가: True",
     1
    ],
    [
     "둘 다 SELECT한 동안 빌린 연결: 2",
     2
    ],
    [
     "둘 다 닫은 뒤 빌린 연결: 0",
     3
    ]
   ],
   "notes": [
    "Session은 요청마다 새 객체예요.",
    "이 실험은 두 Session을 동시에 열어 두었어요. 실제 HTTP 동시 부하 시험은 아니에요.",
    "연결을 풀에 돌려줬어요. DB 서버의 연결 자체를 매번 끊는다는 뜻은 아니에요."
   ]
  },
  "autobegin": {
   "title": "SELECT를 먼저 넣으면?",
   "lines": [
    [
     "SELECT 뒤 in_transaction(): True",
     1
    ],
    [
     "이어서 begin(): A transaction is already begun on this Session.",
     2
    ],
    [
     "rollback 뒤 새 begin에서 SELECT: 1",
     3
    ]
   ],
   "notes": [
    "SQLAlchemy는 SELECT에도 트랜잭션을 자동으로 시작해요. 이를 autobegin이라고 해요.",
    "같은 Session에 새 begin을 겹쳐 시작해서 실패했어요.",
    "실험은 rollback으로 끝냈어요. 실제 get_db에서는 조회를 빼고 Service에 트랜잭션 시작을 맡겨요."
   ]
  },
  "s4": {
   "title": "HTTP에 연결하기 전 입력 계약만 검사",
   "lines": [
    [
     "{\"title\": \"DB 세션을 서비스에 연결하기\", \"content\": \"요청마다 Session을 열고 닫아요.\"}",
     1
    ],
    [
     "content를 빼면: ValidationError (content)",
     2
    ]
   ],
   "notes": [
    "양끝 공백을 제거한 입력이에요. ORM 객체나 DB 행은 아직 만들지 않았어요.",
    "이 검사는 DB 없이 가능해요. 다음 단계의 저장 성공을 대신하지는 않아요."
   ]
  },
  "why_state": {
   "title": "같은 기록 실패를 서로 다른 저장 순서에 넣은 결과",
   "cols": [
    "저장 방식",
    "실패 후 남은 새 노트",
    "실패 후 남은 새 기록"
   ],
   "rows": [
    [
     "노트를 먼저 확정하고 기록을 따로 저장",
     1,
     0
    ],
    [
     "노트와 기록을 하나로 묶어서 저장",
     0,
     0
    ]
   ],
   "notes": [
    "같은 노트 입력을 사용하고 기록 함수에 의도적으로 예외를 냈어요. 각 실험 뒤 실제 DB를 다시 조회했어요."
   ]
  },
  "s5": {
   "title": "실제 INSERT 뒤 두 번째 작업에 예외 주입",
   "lines": [
    [
     "트랜잭션 안에서 조회한 새 제목: DB 세션을 서비스에 연결하기",
     1
    ],
    [
     "실패 전 수량: {\"notes\": 1, \"changes\": 1, \"orphans\": 0}",
     2
    ],
    [
     "실패 후 다른 Session의 수량: {\"notes\": 1, \"changes\": 1, \"orphans\": 0}",
     3
    ],
    [
     "다음 정상 등록의 응답: {\"id\": 2, \"title\": \"DB 세션을 서비스에 연결하기\", \"content\": \"요청마다 Session을 열고 닫아요.\", \"revision\": 1}",
     4
    ]
   ],
   "notes": [
    "flush 후 같은 Session에서는 새 노트를 읽을 수 있었어요. 아직 commit은 하지 않았어요.",
    "시작 데이터예요.",
    "변경 기록 함수에 의도적으로 RuntimeError를 냈어요. with db.begin()이 노트 INSERT까지 롤백했어요.",
    "실패한 작업이 반쯤 남지 않았고 다음 요청도 정상 저장됐어요. 실제 DB 장애나 PostgreSQL 동시성 시험은 아니에요."
   ]
  },
  "health": {
   "title": "같은 서버, DB 준비 전후",
   "lines": [
    [
     "준비 전 GET /live: 200 {\"status\": \"ok\"}",
     1
    ],
    [
     "준비 전 GET /health: 503 {\"error\": {\"code\": \"DB_UNAVAILABLE\", \"message\": \"DB 연결과 테이블 준비를 확인해 주세요.\"}}",
     2
    ],
    [
     "prepare 뒤 GET /health: 200 {\"status\": \"ok\", \"db\": \"ready\"}",
     3
    ]
   ],
   "notes": [
    "프로세스가 요청에 답했어요.",
    "DB 파일은 열리지만 테이블이 없어서 서비스를 사용할 준비가 안 됐어요.",
    "연결과 두 테이블 조회까지 확인했어요. 데이터 전체 무결성이나 LLM, 검색 품질을 보증하는 검사는 아니에요."
   ]
  },
  "s6": {
   "title": "같은 노트에 서로 다른 HTTP 요청을 보낸 결과",
   "cols": [
    "보낸 요청",
    "실제 상태",
    "확인"
   ],
   "rows": [
    [
     "POST 등록",
     201,
     "{&quot;title&quot;: &quot;DB 세션을 서비스에 연결하기&quot;, &quot;content&quot;: &quot;요청마다 Session을 열고 닫아요.&quot;, &quot;id&quot;: 2, &quot;revision&quot;: 1}"
    ],
    [
     "같은 제목 POST",
     409,
     "TITLE_CONFLICT"
    ],
    [
     "content 없는 PUT",
     422,
     "INVALID_INPUT"
    ],
    [
     "실패 뒤 GET",
     200,
     "최초 POST 응답과 모든 필드가 같음"
    ],
    [
     "정상 PUT 뒤 GET",
     200,
     "{&quot;title&quot;: &quot;DB 세션을 서비스에 연결하기&quot;, &quot;content&quot;: &quot;Service가 저장과 기록을 함께 확정해요.&quot;, &quot;id&quot;: 2, &quot;revision&quot;: 2}"
    ],
    [
     "DELETE",
     204,
     "본문 0바이트"
    ],
    [
     "삭제 뒤 GET",
     404,
     "NOTE_NOT_FOUND"
    ]
   ],
   "notes": [
    "시작 {\"notes\": 1, \"changes\": 1, \"orphans\": 0}에서 종료 {\"notes\": 1, \"changes\": 1, \"orphans\": 0}로 돌아왔어요. 이번 노트의 변경 기록도 외래 키 CASCADE로 지웠어요."
   ]
  },
  "s7": {
   "title": "실제로 생성한 클라이언트 파일",
   "lines": [
    [
     "Collection: 요청 12개, pm.test 블록 26개",
     1
    ],
    [
     "Environment: {\"base_url\": \"http://127.0.0.1:8015\", \"created_id\": \"\", \"expected_note\": \"\", \"baseline_count\": \"\"}",
     2
    ]
   ],
   "notes": [
    "파일 안의 요청과 검사 개수예요. 파일을 만들었다는 사실만으로 API 검증을 통과한 것은 아니에요.",
    "base_url은 서버 주소이고, 나머지는 실행 중 채울 값이에요. 서버의 .env와는 다른 설정이에요."
   ]
  },
  "newman": {
   "title": "실제 Uvicorn 서버에 같은 Collection 실행",
   "lines": [
    [
     "요청 12개, 검사 26개, 실패 0개",
     1
    ],
    [
     "완료 뒤 DB 직접 조회: {\"notes\": 1, \"changes\": 1, \"orphans\": 0}",
     2
    ]
   ],
   "notes": [
    "Newman으로 HTTP 주소에 실제 요청을 보냈어요. Postman 데스크톱 UI를 실행한 결과는 아니에요.",
    "클라이언트 검사에 더해 별도 프로세스에서 DB를 다시 읽었어요. PostgreSQL, 외부 모델, 동시 부하 검증은 포함하지 않아요."
   ]
  }
 },
 "flow": [
  [
   1,
   "설정",
   ".env",
   "실행 조건"
  ],
  [
   2,
   "DB 준비",
   "prepare",
   "스키마와 시작 데이터"
  ],
  [
   3,
   "요청 자원",
   "Session",
   "생성하고 반납"
  ],
  [
   4,
   "역할",
   "계약과 저장",
   "입력과 DB"
  ],
  [
   5,
   "업무",
   "트랜잭션",
   "함께 확정하거나 취소"
  ],
  [
   6,
   "앱",
   "Router / Health",
   "HTTP로 연결"
  ],
  [
   7,
   "검증",
   "Postman",
   "저장값과 정리"
  ]
 ],
 "env": "Python 3.13.16, fastapi 0.162.2, SQLAlchemy 2.1.8, pydantic 2.14.3, httpx 0.28.1, uvicorn 0.54.0, SQLite (임시 파일)"
};
