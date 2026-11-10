"""Day17 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 코드(단일 파일 main.py · storage.py · app/ 패키지 · Template · 테스트)를 임시 폴더에 그대로 만들고
TestClient · uvicorn(빈 포트에 잠깐) · pytest 로 실제로 돌려 본 값만 넣는다.
결과는 cmds/day17.html 각 항목의 첫 코드 블록 바로 뒤에(코드가 없으면 첫 표 뒤에) <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: (fastapi · pydantic · httpx · jinja2 · uvicorn · pytest 가 깔린 python) tools/io_day17.py
"""
import html
import importlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp())
PY = sys.executable
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

SEED = {"next_id": 4, "entries": [
    {"id": 1, "title": "가상환경", "category": "setup", "content": "프로젝트별 패키지를 분리한다."},
    {"id": 2, "title": "서버 실행", "category": "api", "content": "uvicorn으로 개발 서버를 켠다."},
    {"id": 3, "title": "입력 오류", "category": "api", "content": "422이면 입력값을 확인한다."},
]}
NEW = {"title": "실습 환경 안내", "content": "Python 3.10 이상을 사용한다.", "category": "setup"}
UPD = {**NEW, "content": "프로젝트별 가상환경을 사용한다."}

# ---------------------------------------------------------------- 강의 코드 (노트와 같은 내용)
STORAGE_SRC = '''import json
from pathlib import Path
from threading import Lock

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "knowledge.json"
LOCK = Lock()

def read_data():
    with DATA.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_data(data):
    temp = DATA.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp.replace(DATA)
    finally:
        temp.unlink(missing_ok=True)
'''

MAIN_SRC = '''from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from storage import BASE, LOCK, read_data, write_data

class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

class Knowledge(KnowledgeWrite):
    id: int

app = FastAPI(title="Knowledge API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/knowledge")
def list_knowledge(category: str | None = None):
    with LOCK:
        data = read_data()
    result = []
    for entry in data["entries"]:
        if category is None or entry["category"] == category:
            result.append(entry)
    return result

@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        if entry["id"] == knowledge_id:
            return entry
    raise HTTPException(404, "Knowledge not found")

@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    with LOCK:
        data = read_data()
        entry = payload.model_dump()
        entry["id"] = data["next_id"]
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    return entry

@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                updated = payload.model_dump()
                updated["id"] = knowledge_id
                data["entries"][i] = updated
                write_data(data)
                return updated
    raise HTTPException(404, "Knowledge not found")

@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                data["entries"].pop(i)
                write_data(data)
                return Response(status_code=204)
    raise HTTPException(404, "Knowledge not found")
'''

# q 검색 버전 (목록 Route만 교체)
QMAIN_SRC = '''from fastapi import FastAPI, Query
from storage import LOCK, read_data

app = FastAPI(title="Knowledge API")

@app.get("/api/knowledge")
def list_knowledge(q: str = Query("", max_length=80)):
    with LOCK:
        data = read_data()
    result = []
    for entry in data["entries"]:
        if q in entry["title"] or q in entry["content"]:
            result.append(entry)
    return result
'''

# Template 버전 (main.py 에 Route 하나 추가)
WEB_SRC = MAIN_SRC + '''
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory=BASE / "templates")

@app.get("/web/knowledge", response_class=HTMLResponse)
def knowledge_page(request: Request):
    with LOCK:
        data = read_data()
    return templates.TemplateResponse(
        request=request, name="knowledge.html",
        context={"entries": data["entries"]})
'''
TEMPLATE_SRC = '''<!-- templates/knowledge.html -->
<ul>
{% for entry in entries %}
  <li>{{ entry.title }}</li>
{% endfor %}
</ul>
'''

# 역할별로 나눈 패키지 (fastapi_knowledge/)
PKG = {
    "app/__init__.py": "",
    "app/routers/__init__.py": "",
    "app/repositories/__init__.py": "",
    "app/schemas.py": '''from pydantic import BaseModel, Field

class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

class Knowledge(KnowledgeWrite):
    id: int
''',
    "app/repositories/json_knowledge.py": '''import json
from pathlib import Path
from threading import Lock

BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "knowledge.json"
LOCK = Lock()

def read_data():
    with DATA.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_data(data):
    temp = DATA.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp.replace(DATA)
    finally:
        temp.unlink(missing_ok=True)

def list_all():
    with LOCK:
        return read_data()["entries"]

def get(knowledge_id):
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        if entry["id"] == knowledge_id:
            return entry
    return None

def create(fields: dict) -> dict:
    with LOCK:
        data = read_data()
        entry = dict(fields)
        entry["id"] = data["next_id"]
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    return entry

def update(knowledge_id, fields):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                updated = dict(fields)
                updated["id"] = knowledge_id
                data["entries"][i] = updated
                write_data(data)
                return updated
    return None

def delete(knowledge_id):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                data["entries"].pop(i)
                write_data(data)
                return True
    return False
''',
    "app/routers/knowledge.py": '''from fastapi import APIRouter, HTTPException, Response
from app.schemas import Knowledge, KnowledgeWrite
from app.repositories import json_knowledge as repo

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

@router.get("")
def list_knowledge():
    return repo.list_all()

@router.get("/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    entry = repo.get(knowledge_id)
    if entry is None:
        raise HTTPException(404, "Knowledge not found")
    return entry

@router.post("", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    return repo.create(payload.model_dump())

@router.put("/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    updated = repo.update(knowledge_id, payload.model_dump())
    if updated is None:
        raise HTTPException(404, "Knowledge not found")
    return updated

@router.delete("/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    if not repo.delete(knowledge_id):
        raise HTTPException(404, "Knowledge not found")
    return Response(status_code=204)
''',
    "app/main.py": '''from fastapi import FastAPI
from app.routers.knowledge import router

app = FastAPI(title="Knowledge API")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
''',
    "tests/test_knowledge.py": '''from fastapi.testclient import TestClient
from app.main import app
from app.repositories import json_knowledge as repo

def test_missing(tmp_path, monkeypatch):
    test_file = tmp_path / "knowledge.json"
    test_file.write_text('{"next_id":1,"entries":[]}', encoding="utf-8")
    monkeypatch.setattr(repo, "DATA", test_file)
    before = test_file.read_bytes()
    with TestClient(app) as client:
        r = client.delete("/api/knowledge/9999")
    assert r.status_code == 404
    assert test_file.read_bytes() == before
''',
}

# ---------------------------------------------------------------- 임시 프로젝트 만들기
SINGLE = TMP / "single"           # main.py 하나짜리
(SINGLE / "data").mkdir(parents=True)
(SINGLE / "templates").mkdir()
(SINGLE / "storage.py").write_text(STORAGE_SRC, encoding="utf-8")
(SINGLE / "main.py").write_text(MAIN_SRC, encoding="utf-8")
(SINGLE / "main_q.py").write_text(QMAIN_SRC, encoding="utf-8")
(SINGLE / "main_web.py").write_text(WEB_SRC, encoding="utf-8")
(SINGLE / "templates" / "knowledge.html").write_text(TEMPLATE_SRC, encoding="utf-8")
DATA = SINGLE / "data" / "knowledge.json"

PROJ = TMP / "fastapi_knowledge"  # 역할별로 나눈 구조
for rel, src in PKG.items():
    p = PROJ / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(src, encoding="utf-8")
(PROJ / "data").mkdir()
PDATA = PROJ / "data" / "knowledge.json"


def seed(path=DATA, obj=SEED):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def fstate(path=DATA):
    d = json.loads(path.read_text(encoding="utf-8"))
    return f"항목 {len(d['entries'])}개 {[e['id'] for e in d['entries']]} · next_id {d['next_id']}"


def short(s, n=110):
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + " …"


def resp(r, n=110):
    body = r.text if r.content else "(빈 본문)"
    return f"{r.status_code} {short(body, n)}"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def kill_tree(p):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    else:
        p.terminate()
    try:
        out, err = p.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
    return (out or b"").decode("utf-8", "replace") + (err or b"").decode("utf-8", "replace")


def get(url):
    with urllib.request.urlopen(url, timeout=2) as r:
        return r.status, r.headers.get("content-type"), r.read().decode("utf-8")


sys.path.insert(0, str(SINGLE))
seed()
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

OPEN = []


def client(app):
    """요청마다 이벤트 루프를 새로 만들지 않도록 with TestClient(app) 상태로 열어 둔다."""
    for attempt in range(5):                   # 이 PC에서 가끔 socketpair가 WinError 10014로 실패해서 다시 시도
        cl = TestClient(app)
        try:
            cl.__enter__()
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(0.5)
    OPEN.append(cl)
    return cl


storage = importlib.import_module("storage")
main = importlib.import_module("main")
c = client(main.app)
IO = {}
SKIPPED = []

# 1. main.py — 첫 JSON 응답
r = c.get("/health")
IO["d17-main-py"] = ("강의 main.py를 TestClient로 호출", [
    ('client.get("/health")', resp(r), "dict를 돌려주면 JSON으로 응답 · 상태 코드 200"),
    ('r.headers["content-type"]', r.headers["content-type"], "FastAPI가 JSON으로 바꿔 보냄"),
    ("app.title", repr(main.app.title), "/docs 맨 위에 보이는 이름"),
    ("[r.path for r in app.routes]",
     repr([rt.path for rt in main.app.routes if rt.path in ("/openapi.json", "/docs", "/redoc", "/health")]),
     "@app.get 한 줄로 /health 등록 + 문서 주소는 자동으로 생김"),
])

# 2. 서버 실행 — uvicorn을 빈 포트에 잠깐 띄우기
seed()
ok = False
for attempt in range(10):                      # 이 PC에서 가끔 Worker가 socketpair(WinError 10014)로 죽어서 다시 시도
    port = free_port()
    srv_log = open(TMP / "uvicorn.log", "wb")  # PIPE 대신 파일 — 자식 프로세스가 파이프를 붙잡아 멈추는 일 방지
    srv = subprocess.Popen([PY, "-m", "uvicorn", "main:app", "--reload", "--port", str(port)],
                           cwd=SINGLE, env=ENV, stdout=srv_log, stderr=subprocess.STDOUT)
    for _ in range(80):
        try:
            st_h, ct_h, body_h = get(f"http://127.0.0.1:{port}/health")
            ok = True
            break
        except Exception:
            if "Traceback" in (TMP / "uvicorn.log").read_text(encoding="utf-8", errors="replace"):
                break
            time.sleep(0.25)
    if ok:
        break
    kill_tree(srv)
    srv_log.close()
if ok:
    st_d, ct_d, body_d = get(f"http://127.0.0.1:{port}/docs")
    for _ in range(4):                         # WinError 10014(이 PC의 socketpair 문제)로 먼저 죽으면 다시
        dup = subprocess.run([PY, "-m", "uvicorn", "main:app", "--port", str(port)],
                             cwd=SINGLE, env=ENV, capture_output=True, timeout=30)
        dup_log = (dup.stdout + dup.stderr).decode("utf-8", "replace")
        dup_err = [ln for ln in dup_log.splitlines() if "10048" in ln]
        if dup_err:
            break
    kill_tree(srv)
    srv_log.close()
    log = (TMP / "uvicorn.log").read_text(encoding="utf-8", errors="replace")
    running = next((ln.strip() for ln in log.splitlines() if "Uvicorn running on" in ln), "")
    reloader = next((ln.strip() for ln in log.splitlines() if "reloader process" in ln.lower()), "")
    rows = [
        (f"python -m uvicorn main:app --reload --port {port}", short(re.sub(r"^INFO:\s+", "", running), 90),
         "포트만 빈 번호로 바꿔 실행 — 기본은 8000"),
    ]
    if reloader:
        rows.append(("(같은 터미널 로그)", short(re.sub(r"^INFO:\s+", "", reloader), 90), "--reload: 파일 변경을 지켜보는 감시 프로세스가 따로 뜸"))
    rows += [
        (f"브라우저 http://127.0.0.1:{port}/health", f"{st_h} {body_h}", "주소창 = GET 요청"),
        (f"브라우저 http://127.0.0.1:{port}/docs", f"{st_d} {ct_d} · Swagger UI 페이지", "자동 API 문서 (HTML)"),
    ]
    if dup_err:
        rows.append(("같은 포트로 uvicorn 한 번 더 실행", short(re.sub(r"^ERROR:\s+", "", dup_err[-1].strip()).split(": [winerror")[0], 120),
                     f"종료 코드 {dup.returncode} — 서버는 한 개만!"))
    else:
        SKIPPED.append("d17-서버-실행: '같은 포트 두 번' 줄 (10048 오류를 재현 못 함)")
    IO["d17-서버-실행"] = ("임시 폴더에 main.py를 만들고 빈 포트로 잠깐 띄운 뒤 Ctrl+C 대신 프로세스를 종료", rows)
else:
    kill_tree(srv)
    srv_log.close()
    SKIPPED.append("d17-서버-실행 (uvicorn이 뜨지 않음)")

# 3. /docs — 문서(openapi.json)와 실제 응답 비교
spec = c.get("/openapi.json").json()
paths = {p: sorted(m.keys()) for p, m in spec["paths"].items()}
doc_get_id = sorted(spec["paths"]["/api/knowledge/{knowledge_id}"]["get"]["responses"].keys())
doc_post = sorted(spec["paths"]["/api/knowledge"]["post"]["responses"].keys())
seed()
r_ex = c.post("/api/knowledge", json=NEW)
IO["d17-docs"] = ("/docs 화면이 읽어 가는 openapi.json · Execute와 같은 요청", [
    ('client.get("/docs")', f"{c.get('/docs').status_code} {c.get('/docs').headers['content-type']}", "사람이 보는 화면 (HTML)"),
    ('client.get("/openapi.json").json()["paths"]', short(json.dumps(paths, ensure_ascii=False), 140), "Route마다 열 수 있는 Method 목록 = /docs의 줄들"),
    ('… ["/api/knowledge/{knowledge_id}"]["get"]["responses"]', repr(doc_get_id), "문서상 Responses에는 404가 없음!"),
    ('client.get("/api/knowledge/9999")', resp(c.get("/api/knowledge/9999")), "그래도 실제 Server response는 404 — 진짜는 이쪽"),
    ('… ["/api/knowledge"]["post"]["responses"]', repr(doc_post), "response_model · status_code에서 자동으로 만든 예시"),
    ("Execute = POST /api/knowledge (Body 세 필드)", resp(r_ex), "Try it out → Execute가 보내는 진짜 요청"),
])

# 4. storage.py
seed()
d = storage.read_data()
storage.write_data(d)
saved = DATA.read_text(encoding="utf-8")
before = DATA.read_bytes()
try:
    storage.write_data({"next_id": 4, "entries": [{"id": 1, "bad": {1, 2}}]})  # set은 JSON으로 못 바꿈
    fail = "에러 없음"
except Exception as e:  # noqa: BLE001
    fail = f"{type(e).__name__}: {e}"
IO["d17-storage-py"] = ("임시 폴더의 storage.py + data/knowledge.json (처음 3건)", [
    ("storage.BASE.name, storage.DATA.relative_to(BASE)", f"{storage.BASE.name!r}, {storage.DATA.relative_to(storage.BASE).as_posix()!r}", "storage.py가 있는 폴더 기준 경로"),
    ("data = read_data(); type(data), list(data)", f"{type(d).__name__}, {list(d)}", "JSON 글자 → 파이썬 dict"),
    ('data["entries"][0]["title"]', repr(d["entries"][0]["title"]), "utf-8로 읽어 한글 그대로"),
    ("write_data(data) 후 파일 2~3번째 줄", short(" | ".join(saved.splitlines()[1:3]), 80), "indent=2 줄바꿈"),
    ('"가상환경" in 파일 내용', repr("가상환경" in saved), "ensure_ascii=False → 한글 그대로 저장"),
    ('json.dumps("가상환경")  (ensure_ascii 기본값)', json.dumps("가상환경"), "기본값이면 \\uXXXX 코드로 바뀜"),
    ('DATA.with_suffix(".tmp").exists()', repr(DATA.with_suffix(".tmp").exists()), "성공 후 임시 파일은 안 남음"),
    ("write_data({… set 값 …})  (저장 실패)", short(fail, 70), "json.dump가 중간에 실패"),
    ("실패 후 원본 bytes == 실패 전", repr(DATA.read_bytes() == before), "임시 파일에서 실패해서 원본은 그대로"),
    ('실패 후 DATA.with_suffix(".tmp").exists()', repr(DATA.with_suffix(".tmp").exists()), "finally에서 반쯤 쓴 임시 파일 정리"),
])

# 5. with LOCK — 20개 스레드 동시 등록
def race(mode, n=20):
    """mode: "write" = 저장 한 줄만 잠금, "all" = 읽기~저장 전체를 with LOCK."""
    seed(obj={"next_id": 1, "entries": []})
    errors = []

    def body(k):
        data = storage.read_data()
        time.sleep(0.01)                 # 읽기와 저장 사이 틈
        data["entries"].append({"id": data["next_id"], "who": k})
        data["next_id"] += 1
        if mode == "write":
            with storage.LOCK:
                storage.write_data(data)
        else:
            storage.write_data(data)

    def work(k):
        try:
            if mode == "all":
                with storage.LOCK:
                    body(k)
            else:
                body(k)
        except Exception as e:  # noqa: BLE001
            errors.append(type(e).__name__)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    d = storage.read_data()
    err = f" · 오류 {len(errors)}건({', '.join(sorted(set(errors)))})" if errors else ""
    return f"{n}건 보냄 → 파일에 {len(d['entries'])}건 · next_id {d['next_id']}{err}"


writeonly = race("write")
withlock = race("all")
inside = None
with storage.LOCK:
    inside = storage.LOCK.locked()
IO["d17-with-lock"] = ("빈 파일에서 스레드 20개가 동시에 '읽기 → 0.01초 → 추가 → 저장'", [
    ("read_data() … with LOCK: write_data()  (저장만 잠금)", writeonly, "모두 빈 파일을 읽은 채 차례로 저장 → 서로 덮어씀"),
    ("with LOCK: read_data() … write_data()", withlock, "앞 요청이 끝날 때까지 기다리니 전부 남음"),
    ("with LOCK: 안에서 LOCK.locked()", repr(inside), "누가 잡고 있는 중"),
    ("블록을 나온 뒤 LOCK.locked()", repr(storage.LOCK.locked()), "with가 끝나면 자동으로 풀림"),
])

# 6. 목록 조회
seed()
b0 = DATA.read_bytes()
r = c.get("/api/knowledge")
same = DATA.read_bytes() == b0
seed(obj={"next_id": 1, "entries": []})
r_empty = c.get("/api/knowledge")
seed()
IO["d17-목록-조회"] = ("처음 3건 파일", [
    ('client.get("/api/knowledge")', f"{r.status_code} · {len(r.json())}건 · id {[e['id'] for e in r.json()]}", "list → JSON 배열"),
    ('r.json()[0]', short(json.dumps(r.json()[0], ensure_ascii=False), 100), "파일의 항목 그대로"),
    ("GET 전후 파일 bytes 같음?", repr(same), "GET은 읽기만 함"),
    ('빈 파일에서 client.get("/api/knowledge")', resp(r_empty), "없는 건 오류가 아니라 빈 배열"),
])

# 7. 상세 조회 + 404
IO["d17-상세-조회-404"] = ("처음 3건 파일", [
    ('client.get("/api/knowledge/2")', resp(c.get("/api/knowledge/2")), "찾으면 200 + 그 항목"),
    ('client.get("/api/knowledge/9999")', resp(c.get("/api/knowledge/9999")), "Route는 맞는데 그 ID가 없음"),
    ('client.get("/api/knowledge/foo")', resp(c.get("/api/knowledge/foo"), 90), "int로 못 바꿔 함수 실행 전에 422"),
    ('client.get("/nope")', resp(c.get("/nope")), "Route 자체가 없음 — detail 문구가 다름"),
])

# 8. Query로 거르기 (+ q 검색 버전)
mq = importlib.import_module("main_q")
cq = client(mq.app)
ids = lambda rr: f"{rr.status_code} · id {[e['id'] for e in rr.json()]}" if rr.status_code == 200 else resp(rr, 90)  # noqa: E731
IO["d17-query로-목록-거르기"] = ("처음 3건 파일 (category: setup 1건, api 2건)", [
    ('client.get("/api/knowledge")', ids(c.get("/api/knowledge")), "category 생략 → None → 전체"),
    ('client.get("/api/knowledge?category=api")', ids(c.get("/api/knowledge", params={"category": "api"})), "같은 분류만"),
    ('client.get("/api/knowledge?category=zzz")', ids(c.get("/api/knowledge", params={"category": "zzz"})), "맞는 게 없으면 빈 배열"),
    ('q 버전: get("/api/knowledge?q=서버")', ids(cq.get("/api/knowledge", params={"q": "서버"})), "제목 '서버 실행'에 글자가 들어 있음"),
    ('q 버전: get("…?q=uvicorn")', ids(cq.get("/api/knowledge", params={"q": "uvicorn"})), "본문에서 찾음"),
    ('q 버전: get("…?q=Uvicorn")', ids(cq.get("/api/knowledge", params={"q": "Uvicorn"})), "대소문자가 다르면 못 찾음"),
    ('q 버전: get("…?q=개발 서버 켜는 법")', ids(cq.get("/api/knowledge", params={"q": "개발 서버 켜는 법"})), "뜻이 같아도 글자가 그대로 없으면 0건"),
    ('q 버전: get("…?q=" + "가" * 81)', resp(cq.get("/api/knowledge", params={"q": "가" * 81}), 80), "max_length=80 넘으면 422"),
])

# 9. Pydantic
KW, K = main.KnowledgeWrite, main.Knowledge


def verr(**kw):
    try:
        KW(**kw)
        return "통과"
    except ValidationError as e:
        er = e.errors()[0]
        return f"ValidationError: type={er['type']!r}, loc={er['loc']!r}"


IO["d17-pydantic으로-입력-모양-정하기"] = ("main.py의 KnowledgeWrite · Knowledge를 직접 만들어 보기 — NEW = " + json.dumps(NEW, ensure_ascii=False), [
    ("KnowledgeWrite(**NEW).model_dump()", short(json.dumps(KW(**NEW).model_dump(), ensure_ascii=False), 100), "검사를 통과한 값 → dict"),
    ('KnowledgeWrite(**NEW, id=1).model_dump()', repr(sorted(KW(**NEW, id=1).model_dump())), "보낸 id는 버려짐 — 번호는 서버가 정함"),
    ('KnowledgeWrite(title="", …)', verr(**{**NEW, "title": ""}), "min_length=1 위반"),
    ('KnowledgeWrite(title="가" * 81, …)', verr(**{**NEW, "title": "가" * 81}), "max_length=80 위반"),
    ('KnowledgeWrite(title=123, …)', verr(**{**NEW, "title": 123}), "숫자를 문자열로 바꿔 주지 않음"),
    ('KnowledgeWrite(content=…, category=…)  (title 없음)', verr(content="x", category="setup"), "기본값이 없으니 필수"),
    ("Knowledge(**NEW, id=4).model_dump()", repr(list(K(**NEW, id=4).model_dump())), "응답 모델 = 입력 세 필드 + id"),
])

# 10. 등록 함수
seed()
n0 = len(c.get("/api/knowledge").json())
r1 = c.post("/api/knowledge", json=NEW)
st1 = fstate()
r1get = c.get(f"/api/knowledge/{r1.json()['id']}")
n1 = len(c.get("/api/knowledge").json())
r2 = c.post("/api/knowledge", json=NEW)
IO["d17-등록-함수"] = ("처음 3건 파일에 POST — NEW = " + json.dumps(NEW, ensure_ascii=False), [
    ('client.post("/api/knowledge", json=NEW)', resp(r1), "201 Created + 새 id"),
    ("sorted(r.json())", repr(sorted(r1.json())), "response_model 덕분에 next_id는 응답에 안 섞임"),
    (f'client.get("/api/knowledge/{r1.json()["id"]}")', f"{r1get.status_code}", "받은 ID로 다시 조회"),
    ("목록 개수", f"{n0} → {n1}", ""),
    ("파일 직접 확인", st1, "파일에도 저장됨"),
    ("같은 POST 한 번 더", f"{r2.status_code} · id {r2.json()['id']} · {fstate()}", "POST는 보낼 때마다 새로 만듦"),
])
seed()

# 11. 422
seed()
c.post("/api/knowledge", json=NEW)          # 노트와 같이 4건 · next_id 5 에서 시작
st_before = fstate()
b_before = DATA.read_bytes()
e1 = c.post("/api/knowledge", json={"content": "x", "category": "setup"})
e2 = c.post("/api/knowledge", json={**NEW, "content": ""})
e3 = c.post("/api/knowledge", content='{"title": "a",}'.encode(), headers={"content-type": "application/json"})


def d0(rr):
    er = rr.json()["detail"][0]
    return f"{rr.status_code} type={er['type']!r} loc={er['loc']!r}"


IO["d17-422-잘못된-입력은-저장-전에-막힘"] = (f"항목 4건 · next_id 5 파일에 잘못된 POST 세 번", [
    ("시작 상태", st_before, ""),
    ('post(json={"content": "x", "category": "setup"})', d0(e1), "title 빠짐"),
    ('post(json={**NEW, "content": ""})', d0(e2), "빈 문자열 — min_length=1"),
    ("post(content='{\"title\": \"a\",}')", d0(e3), "JSON 문법 오류 (끝에 쉼표)"),
    ('e.json()["detail"][0]["msg"]  (content "")', e2.json()["detail"][0]["msg"], "고칠 곳을 알려 주는 문장"),
    ("세 번 뒤 파일", f"{fstate()} · bytes 같음 {DATA.read_bytes() == b_before}", "함수 본문 전에 막혀 저장 코드가 안 돎"),
])

# 12. 수정 함수 (4건 · next_id 5 상태에서)
p1 = c.put("/api/knowledge/4", json=UPD)
s1 = fstate()
bb = DATA.read_bytes()
p2 = c.put("/api/knowledge/4", json=UPD)
same2 = DATA.read_bytes() == bb
p3 = c.put("/api/knowledge/4", json={"title": "a", "content": "b"})
p4 = c.put("/api/knowledge/9999", json=UPD)
s4 = fstate()
p5 = c.patch("/api/knowledge/2", json={"title": "x"})
IO["d17-수정-함수"] = ("항목 4건(id 4 = 실습 환경 안내) · next_id 5 — UPD = NEW에서 content만 " + json.dumps(UPD["content"], ensure_ascii=False), [
    ('client.put("/api/knowledge/4", json=UPD)', resp(p1), "그 자리만 교체"),
    ("PUT 뒤 파일", s1, "개수·next_id 그대로 (append 아님)"),
    ("같은 PUT 다시 → 파일 bytes 같음?", f"{p2.status_code} · {same2}", "몇 번 보내도 결과가 같음"),
    ('put("/api/knowledge/4", json={title, content})', d0(p3), "PUT은 세 필드 다 — category 빠지면 422"),
    ('put("/api/knowledge/9999", json=UPD)', resp(p4), "새로 만들지 않고 404"),
    ("404 뒤 파일", s4, "새 항목 안 생김"),
    ('client.patch("/api/knowledge/2", …)', resp(p5), "PATCH Route는 없음 → 405"),
])

# 13. 삭제 함수
dl1 = c.delete("/api/knowledge/4")
s_del = fstate()
try:
    dl1.json()
    js = "값 있음"
except Exception as e:  # noqa: BLE001
    js = type(e).__name__
g4 = c.get("/api/knowledge/4")
dl2 = c.delete("/api/knowledge/4")
pn = c.post("/api/knowledge", json=NEW)
IO["d17-삭제-함수"] = ("항목 4건 · next_id 5 에서 id 4 삭제", [
    ('client.delete("/api/knowledge/4")', f"{dl1.status_code} · r.content = {dl1.content!r}", "본문 없는 성공"),
    ("삭제 뒤 파일", s_del, "그 한 건만 빠지고 next_id는 그대로"),
    ("r.json()  (204 응답에)", js, "빈 본문은 JSON으로 못 읽음 → 상태 코드로 판단"),
    ('client.get("/api/knowledge/4")', resp(g4), "재조회 404"),
    ('client.delete("/api/knowledge/4")  (또)', resp(dl2), "이미 없음"),
    ("삭제 뒤 새로 POST", f"{pn.status_code} · id {pn.json()['id']}", "지운 4를 다시 쓰지 않고 5"),
])

# ---------------------------------------------------------------- 역할별 구조
seed(PDATA)
sys.path.insert(0, str(PROJ))
pmain = importlib.import_module("app.main")
prouter = importlib.import_module("app.routers.knowledge")
repo = importlib.import_module("app.repositories.json_knowledge")
pc = client(pmain.app)

# 14. app/main.py
bad = subprocess.run([PY, "-m", "uvicorn", "app.main:app", "--port", str(free_port())],
                     cwd=PROJ / "app", env=ENV, capture_output=True, timeout=30)
bad_log = (bad.stdout + bad.stderr).decode("utf-8", "replace")
bad_line = next((ln.strip() for ln in bad_log.splitlines() if "Could not import" in ln or "ModuleNotFoundError" in ln),
                f"종료 코드 {bad.returncode}")
routes14 = {p: sorted(m) for p, m in pc.get("/openapi.json").json()["paths"].items()}
IO["d17-app-main-py"] = ("임시 fastapi_knowledge/ 폴더 (app/ 패키지 + data/knowledge.json 3건)", [
    ("from app.main import app  (프로젝트 최상위에서)", "성공", "app 폴더를 패키지로 찾음"),
    ('client.get("/openapi.json").json()["paths"]', short(json.dumps(routes14), 140), "include_router로 Router의 Route들이 붙음"),
    ('client.get("/health")', resp(pc.get("/health")), "앱만 대답하면 200"),
    ('client.get("/api/knowledge")', f"{pc.get('/api/knowledge').status_code} · {len(pc.get('/api/knowledge').json())}건", "Router 쪽 Route도 동작"),
    ("app/ 안에서 python -m uvicorn app.main:app", short(bad_line, 110), f"종료 코드 {bad.returncode} — 실행 위치는 최상위"),
])

# 15. APIRouter
import fastapi  # noqa: E402

dupapp = fastapi.FastAPI()


@dupapp.post("/api/knowledge")
def old_create():
    return {"from": "main.py에 남은 옛 코드"}


dupapp.include_router(prouter.router)
dc = client(dupapp)
seed(PDATA)
pspec = pc.get("/openapi.json").json()
IO["d17-apirouter"] = ("app/routers/knowledge.py의 router", [
    ("router.prefix, router.tags", f"{prouter.router.prefix!r}, {prouter.router.tags!r}", ""),
    ("[r.path for r in router.routes]", repr(sorted({rt.path for rt in prouter.router.routes})), 'prefix + "" / prefix + "/{knowledge_id}"'),
    ('openapi["paths"]["/api/knowledge"]["post"]["tags"]', repr(pspec["paths"]["/api/knowledge"]["post"]["tags"]), "/docs에서 묶어 보여 줄 그룹 이름"),
    ('client.post("/api/knowledge", json=NEW)', resp(pc.post("/api/knowledge", json=NEW)), "Router → repo.create → 201"),
    ("옛 @app.post를 지우지 않고 include_router → POST", resp(dc.post("/api/knowledge", json=NEW)), "먼저 등록된 옛 함수가 받음 (201도 아님)"),
])

# 16. Repository
seed(PDATA)
e_new = repo.create({"title": "실습 환경 안내", "content": "Python 3.10 이상을 사용한다.", "category": "setup"})
here = Path(repo.__file__).resolve()
IO["d17-repository-저장-로직-그대로-옮기기"] = ("app/repositories/json_knowledge.py", [
    ("Path(__file__).resolve().parents[0].name", repr(here.parents[0].name), ""),
    ("Path(__file__).resolve().parents[1].name", repr(here.parents[1].name), ""),
    ("Path(__file__).resolve().parents[2].name", repr(here.parents[2].name), "프로젝트 루트"),
    ("repo.DATA.relative_to(repo.BASE)", repr(repo.DATA.relative_to(repo.BASE).as_posix()), "루트 아래 data 파일을 그대로 가리킴"),
    ('repo.create({title, content, category})', short(json.dumps(e_new, ensure_ascii=False), 100), "HTTP 없이 저장만 — 순서는 main.py 때와 같음"),
    ("파일 확인", fstate(PDATA), "id 4 · next_id 5"),
])

# 17. Template
seed()
mweb = importlib.import_module("main_web")
wc = client(mweb.app)
rw = wc.get("/web/knowledge")
lis = re.findall(r"<li>(.*?)</li>", rw.text)
rj = wc.get("/api/knowledge")
IO["d17-같은-데이터를-html-화면으로-template"] = ("main.py + templates/knowledge.html, 처음 3건 파일", [
    ('client.get("/web/knowledge").headers["content-type"]', f"{rw.status_code} {rw.headers['content-type']}", "사람이 보는 화면"),
    ("응답 본문의 <li> 내용", repr(lis), "{% for %}가 entries 3건만큼 반복"),
    ('client.get("/api/knowledge").headers["content-type"]', f"{rj.status_code} {rj.headers['content-type']}", "같은 파일, 프로그램용 JSON"),
    ('client.get("/api/knowledge").json()[0]["title"]', repr(rj.json()[0]["title"]), "값을 바로 꺼내 쓸 수 있음"),
])

# 18. 자동 테스트
seed(PDATA)
real_before = PDATA.read_bytes()
pt = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=PROJ, env=ENV, capture_output=True, timeout=120)
pt_out = pt.stdout.decode("utf-8", "replace").strip().splitlines()
pt_last = pt_out[-1] if pt_out else ""
pt_last = re.sub(r"\s+in [\d.]+s", " in …s", pt_last)
# monkeypatch 없이 같은 요청이면 실제 파일을 씀 — 대신 실패 요청은 파일을 바꾸지 않음을 직접 확인
IO["d17-자동-테스트-실제-지식-파일은-건드리지-않게"] = ("임시 fastapi_knowledge/ 에 tests/test_knowledge.py를 만들고 실행", [
    ("python -m pytest -q  (프로젝트 최상위)", pt_last, f"종료 코드 {pt.returncode} — 테스트 통과"),
    ("테스트 뒤 실제 data/knowledge.json bytes 같음?", repr(PDATA.read_bytes() == real_before), "monkeypatch로 임시 파일만 썼음"),
    ("실제 파일", fstate(PDATA), "처음 3건 그대로"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day17.html"
s = frag.read_text(encoding="utf-8")
s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
for cid, (title, rows) in IO.items():
    m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
    assert m, cid
    sec = m.group(0)
    k = sec.find("</code></pre>\n")
    if k >= 0:
        k += len("</code></pre>\n")
    else:                                   # 코드가 없는 항목은 첫 표 뒤에
        k = sec.find("</table>\n")
        assert k >= 0, cid
        k += len("</table>\n")
    pos = m.start() + k
    s = s[:pos] + box(title, rows) + "\n" + s[pos:]
frag.write_text(s, encoding="utf-8")
for cid, (title, rows) in IO.items():
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i}  →  {o}   ({n})")
print("SKIPPED:", SKIPPED)
for cl in OPEN:
    cl.__exit__(None, None, None)
shutil.rmtree(TMP, ignore_errors=True)
