"""백엔드 외전. 예제를 실제 실행하고 페이지를 재생성한다. --newman은 HTTP 전체 검증도 실행."""
import argparse
from contextlib import contextmanager
from html import escape
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import pagekit as K
import stepcode as S

EXAMPLE = ROOT / "examples/service_flow"
CODE = S.parse((EXAMPLE / "service.py").read_text(encoding="utf-8"))
outs = {}

@contextmanager
def at(stage, prepared=True):
    with tempfile.TemporaryDirectory(prefix="sesac_service_flow_") as temp:
        wd = Path(temp)
        ns = S.run(CODE, stage, wd, filename="service.py")
        try:
            if prepared and "prepare" in ns:
                ns["prepare"]()
            yield ns, wd
        finally:
            if "engine" in ns:
                ns["engine"].dispose()

def j(value):
    return json.dumps(value, ensure_ascii=False)

def generate(newman=False):
    from fastapi.testclient import TestClient
    from pydantic import ValidationError
    from sqlalchemy.exc import InvalidRequestError

    with at(1) as (ns, wd):
        S.claim(ns["DB_FILE"] == wd / "out/service.sqlite3", "상대 DB 경로는 예제 파일 기준")
        outs["s1"] = S.box("임시 폴더에서 설정 읽기", [("DB 파일: out/service.sqlite3", 1)],
            ["이번 실행의 service.py가 있는 임시 폴더 아래예요. 실제 사용자 DB를 읽거나 바꾸지 않아요."])

    with at(2, prepared=False) as (ns, _):
        first = ns["prepare"]()
        second = ns["prepare"]()
        S.claim(first == second == {"notes": 1, "changes": 1, "orphans": 0}, "prepare 재실행은 행을 늘리지 않는다")
        outs["s2"] = S.box("준비 후 실제 DB를 다시 조회", [(j(first), 1), ("prepare를 한 번 더 실행: " + j(second), 2)],
            [f"노트 {first['notes']}건, 연결된 변경 기록 {first['changes']}건이에요. orphans는 부모 노트가 없는 기록 수예요.",
             "기존 데이터를 지우지 않고, 빈 테이블에만 시작 데이터를 넣었어요. 스키마 변경 기능은 아니에요."])

    with at(3) as (ns, _):
        a, b = ns["get_db"](), ns["get_db"]()
        sa, sb = next(a), next(b)
        different = sa is not sb
        sa.execute(ns["text"]("SELECT 1"))
        sb.execute(ns["text"]("SELECT 1"))
        borrowed = ns["engine"].pool.checkedout()
        a.close()
        b.close()
        returned = ns["engine"].pool.checkedout()
        S.claim(different and borrowed == 2 and returned == 0, "Session 분리와 연결 반납")
        outs["s3"] = S.box("두 요청의 Session 수명 실험", [
            (f"서로 다른 Session인가: {different}", 1),
            (f"둘 다 SELECT한 동안 빌린 연결: {borrowed}", 2),
            (f"둘 다 닫은 뒤 빌린 연결: {returned}", 3)],
            ["Session은 요청마다 새 객체예요.", "이 실험은 두 Session을 동시에 열어 두었어요. 실제 HTTP 동시 부하 시험은 아니에요.",
             "연결을 풀에 돌려줬어요. DB 서버의 연결 자체를 매번 끊는다는 뜻은 아니에요."])
        with ns["SessionLocal"]() as db:
            db.execute(ns["text"]("SELECT 1"))
            active = db.in_transaction()
            try:
                with db.begin():
                    pass
            except InvalidRequestError as exc:
                error = str(exc)
            db.rollback()
            with db.begin():
                recovered = db.scalar(ns["text"]("SELECT 1"))
        S.claim(active and recovered == 1 and "already begun" in error, "조회로 시작된 트랜잭션과 begin 충돌")
        outs["autobegin"] = S.box("SELECT를 먼저 넣으면?", [
            (f"SELECT 뒤 in_transaction(): {active}", 1), (f"이어서 begin(): {error}", 2),
            (f"rollback 뒤 새 begin에서 SELECT: {recovered}", 3)],
            ["SQLAlchemy는 SELECT에도 트랜잭션을 자동으로 시작해요. 이를 autobegin이라고 해요.",
             "같은 Session에 새 begin을 겹쳐 시작해서 실패했어요.",
             "실험은 rollback으로 끝냈어요. 실제 get_db에서는 조회를 빼고 Service에 트랜잭션 시작을 맡겨요."])

    with at(4) as (ns, _):
        payload = ns["NoteWrite"](title="  DB 세션을 서비스에 연결하기  ", content="요청마다 Session을 열고 닫아요.")
        try:
            ns["NoteWrite"](title="DB 세션을 서비스에 연결하기")
        except ValidationError as exc:
            missing = exc.errors()[0]["loc"][0]
        S.claim(missing == "content" and not payload.title.startswith(" "), "입력 계약 검사")
        outs["s4"] = S.box("HTTP에 연결하기 전 입력 계약만 검사", [
            (j(payload.model_dump()), 1), (f"content를 빼면: ValidationError ({missing})", 2)],
            ["양끝 공백을 제거한 입력이에요. ORM 객체나 DB 행은 아직 만들지 않았어요.",
             "이 검사는 DB 없이 가능해요. 다음 단계의 저장 성공을 대신하지는 않아요."])

    payload_data = {"title": "DB 세션을 서비스에 연결하기", "content": "요청마다 Session을 열고 닫아요."}
    with at(5) as (ns, _):
        baseline = ns["verify"]()
        inserted = []
        original = ns["record_change"]
        def fail_change(db, note, action):
            inserted.append(db.get(ns["Note"], note.id).title)
            raise RuntimeError("변경 기록 저장 실패를 재현")
        # 먼저 잘못 나눈 두 트랜잭션에서 실제로 일부 저장이 남는지 확인해요.
        with ns["SessionLocal"].begin() as db:
            bad_id = ns["insert_note"](db, ns["NoteWrite"](**payload_data)).id
        try:
            with ns["SessionLocal"].begin() as db:
                fail_change(db, ns["require_note"](db, bad_id), "created")
        except RuntimeError:
            pass
        partial = ns["verify"]()
        with ns["SessionLocal"]() as db:
            ns["delete_note"](db, bad_id)
        inserted.clear()
        ns["record_change"] = fail_change
        with ns["SessionLocal"]() as db:
            try:
                ns["create_note"](db, ns["NoteWrite"](**payload_data))
            except RuntimeError:
                after_error = db.in_transaction()
        ns["record_change"] = original
        after = ns["verify"]()
        with ns["SessionLocal"]() as db:
            created = ns["create_note"](db, ns["NoteWrite"](**payload_data))
        S.claim(inserted == [payload_data["title"]] and baseline == after and not after_error,
                "노트 INSERT 뒤 기록 실패는 전체 롤백")
        S.claim(partial["notes"] == baseline["notes"] + 1 and partial["changes"] == baseline["changes"],
                "노트만 먼저 확정하면 기록 실패 뒤 일부 저장이 남는다")
        outs["why_state"] = S.table("같은 기록 실패를 서로 다른 저장 순서에 넣은 결과",
            ["저장 방식", "실패 후 남은 새 노트", "실패 후 남은 새 기록"], [
                ["노트를 먼저 확정하고 기록을 따로 저장", partial["notes"] - baseline["notes"], partial["changes"] - baseline["changes"]],
                ["노트와 기록을 하나로 묶어서 저장", after["notes"] - baseline["notes"], after["changes"] - baseline["changes"]]],
            ["같은 노트 입력을 사용하고 기록 함수에 의도적으로 예외를 냈어요. 각 실험 뒤 실제 DB를 다시 조회했어요."])
        S.claim(ns["verify"]()["notes"] == baseline["notes"] + 1, "실패 뒤 정상 요청도 저장 가능")
        outs["s5"] = S.box("실제 INSERT 뒤 두 번째 작업에 예외 주입", [
            (f"트랜잭션 안에서 조회한 새 제목: {inserted[0]}", 1),
            (f"실패 전 수량: {j(baseline)}", 2), (f"실패 후 다른 Session의 수량: {j(after)}", 3),
            (f"다음 정상 등록의 응답: {j(created)}", 4)],
            ["flush 후 같은 Session에서는 새 노트를 읽을 수 있었어요. 아직 commit은 하지 않았어요.",
             "시작 데이터예요.", "변경 기록 함수에 의도적으로 RuntimeError를 냈어요. with db.begin()이 노트 INSERT까지 롤백했어요.",
             "실패한 작업이 반쯤 남지 않았고 다음 요청도 정상 저장됐어요. 실제 DB 장애나 PostgreSQL 동시성 시험은 아니에요."])

    with at(6, prepared=False) as (ns, _):
        with TestClient(ns["app"]) as client:
            live = client.get("/live")
            unready = client.get("/health")
            ns["prepare"]()
            ready = client.get("/health")
            start = ns["verify"]()
            first = client.post("/notes", json=payload_data)
            note_id = first.json()["id"]
            duplicate = client.post("/notes", json=payload_data)
            invalid = client.put(f"/notes/{note_id}", json={"title": payload_data["title"]})
            unchanged = client.get(f"/notes/{note_id}")
            changed_data = {**payload_data, "content": "Service가 저장과 기록을 함께 확정해요."}
            put = client.put(f"/notes/{note_id}", json=changed_data)
            read = client.get(f"/notes/{note_id}")
            deleted = client.delete(f"/notes/{note_id}")
            missing = client.get(f"/notes/{note_id}")
            end = ns["verify"]()
        S.claim(live.status_code == 200 and unready.status_code == 503 and ready.status_code == 200, "Health 준비 단계 구분")
        S.claim(first.status_code == 201 and duplicate.status_code == 409 and invalid.status_code == 422,
                "정상 저장, 중복 거부, 잘못된 수정 거부")
        S.claim(unchanged.json() == first.json() and put.json() == read.json()
                and read.json()["revision"] == first.json()["revision"] + 1, "실패 후 값 유지와 수정 후 재조회")
        S.claim(deleted.status_code == 204 and deleted.content == b"" and missing.status_code == 404
                and start == end, "삭제와 관계 데이터 정리, 시작 수량 복귀")
        outs["health"] = S.box("같은 서버, DB 준비 전후", [
            (f"준비 전 GET /live: {live.status_code} {j(live.json())}", 1),
            (f"준비 전 GET /health: {unready.status_code} {j(unready.json())}", 2),
            (f"prepare 뒤 GET /health: {ready.status_code} {j(ready.json())}", 3)],
            ["프로세스가 요청에 답했어요.", "DB 파일은 열리지만 테이블이 없어서 서비스를 사용할 준비가 안 됐어요.",
             "연결과 두 테이블 조회까지 확인했어요. 데이터 전체 무결성이나 LLM, 검색 품질을 보증하는 검사는 아니에요."])
        outs["s6"] = S.table("같은 노트에 서로 다른 HTTP 요청을 보낸 결과", ["보낸 요청", "실제 상태", "확인"], [
            ["POST 등록", first.status_code, escape(j(first.json()))],
            ["같은 제목 POST", duplicate.status_code, duplicate.json()["error"]["code"]],
            ["content 없는 PUT", invalid.status_code, invalid.json()["error"]["code"]],
            ["실패 뒤 GET", unchanged.status_code, "최초 POST 응답과 모든 필드가 같음"],
            ["정상 PUT 뒤 GET", read.status_code, escape(j(read.json()))],
            ["DELETE", deleted.status_code, f"본문 {len(deleted.content)}바이트"],
            ["삭제 뒤 GET", missing.status_code, missing.json()["error"]["code"]]],
            [f"시작 {j(start)}에서 종료 {j(end)}로 돌아왔어요. 이번 노트의 변경 기록도 외래 키 CASCADE로 지웠어요."])

    with at(7) as (ns, wd):
        collection, environment = ns["export_postman"]()
        for filename in ("service.postman_collection.json", "service.postman_environment.json"):
            shutil.copyfile(wd / filename, EXAMPLE / filename)
        test_count = sum(line.count("pm.test(") for item in collection["item"]
                         for ev in item["event"] for line in ev["script"]["exec"])
        outs["s7"] = S.box("실제로 생성한 클라이언트 파일", [
            (f"Collection: 요청 {len(collection['item'])}개, pm.test 블록 {test_count}개", 1),
            ("Environment: " + j({v["key"]: v["value"] for v in environment["values"]}), 2)],
            ["파일 안의 요청과 검사 개수예요. 파일을 만들었다는 사실만으로 API 검증을 통과한 것은 아니에요.",
             "base_url은 서버 주소이고, 나머지는 실행 중 채울 값이에요. 서버의 .env와는 다른 설정이에요."])
        if newman:
            outs["newman"] = run_newman(wd, start, ns)
        else:
            outs["newman"] = S.box("검증 범위", ["이 빌드는 Python TestClient로 API와 실제 SQLite 저장값을 확인했어요."],
                ["실제 TCP 서버와 Collection 검사도 실행하려면 빌드 명령에 --newman을 붙여요."])

    write_page()

def run_newman(wd, expected, ns):
    """파일, TCP 서버, Postman 스크립트를 실제로 함께 실행한다."""
    import httpx
    import uvicorn
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    shutil.copyfile(EXAMPLE / "service.py", wd / "service.py")
    server = uvicorn.Server(uvicorn.Config(ns["app"], host="127.0.0.1", port=port, log_level="critical"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{port}"
        with httpx.Client(timeout=1, trust_env=False) as client:
            for _ in range(60):
                try:
                    if client.get(url + "/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.25)
            else:
                raise RuntimeError("검증용 Uvicorn 서버가 준비되지 않았어요.")
        npx = shutil.which("npx.cmd") or shutil.which("npx")
        if not npx:
            raise RuntimeError("--newman 검증에는 Node.js의 npx가 필요해요.")
        report_path = wd / "newman.json"
        result = subprocess.run([npx, "--yes", "newman", "run", "service.postman_collection.json", "-e",
            "service.postman_environment.json", "--env-var", f"base_url={url}", "--bail", "--reporters", "json",
            "--reporter-json-export", str(report_path)], cwd=wd, capture_output=True, text=True, encoding="utf-8", timeout=180)
        if result.returncode:
            raise RuntimeError("Newman 실패: " + result.stdout[-2000:] + result.stderr[-2000:])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        run = report["run"]
        stats = run["stats"]
        S.claim(not run["failures"] and stats["requests"]["failed"] == 0 and stats["assertions"]["failed"] == 0,
                "실제 HTTP Collection 검사에 실패 없음")
        S.claim(stats["requests"]["total"] == 12 and stats["assertions"]["pending"] == 0, "요청과 검사를 건너뛰지 않음")
        exported = json.loads(subprocess.check_output([sys.executable, "service.py", "verify"], cwd=wd).decode("utf-8"))
        S.claim(exported == expected, "실제 서버 검증 후 시작 DB 상태 복귀")
        state = {v["key"]: v["value"] for v in report["environment"]["values"]}
        S.claim(not state.get("created_id") and not state.get("expected_note"), "생성 ID와 응답 snapshot 정리")
        print(f"Newman: {stats['requests']['total']} requests, {stats['assertions']['total']} assertions, 0 failures")
        return S.box("실제 Uvicorn 서버에 같은 Collection 실행", [
            (f"요청 {stats['requests']['total']}개, 검사 {stats['assertions']['total']}개, 실패 {len(run['failures'])}개", 1),
            (f"완료 뒤 DB 직접 조회: {j(exported)}", 2)],
            ["Newman으로 HTTP 주소에 실제 요청을 보냈어요. Postman 데스크톱 UI를 실행한 결과는 아니에요.",
             "클라이언트 검사에 더해 별도 프로세스에서 DB를 다시 읽었어요. PostgreSQL, 외부 모델, 동시 부하 검증은 포함하지 않아요."])
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        S.claim(not thread.is_alive(), "검증 서버가 종료되어 DB 파일을 반납")

def write_page():
    S.save(ROOT / "topics/backend/service-flow-data.js", {
        "code": CODE, "names": S.names(CODE), "range": [1, 7], "outs": outs,
        "flow": [[1, "설정", ".env", "실행 조건"], [2, "DB 준비", "prepare", "스키마와 시작 데이터"],
                 [3, "요청 자원", "Session", "생성하고 반납"], [4, "역할", "계약과 저장", "입력과 DB"],
                 [5, "업무", "트랜잭션", "함께 확정하거나 취소"], [6, "앱", "Router / Health", "HTTP로 연결"],
                 [7, "검증", "Postman", "저장값과 정리"]],
        "env": "Python " + sys.version.split()[0] + ", " + ", ".join(f"{p} {version(p)}" for p in
            ["fastapi", "SQLAlchemy", "pydantic", "httpx", "uvicorn"]) + ", SQLite (임시 파일)",
    })
    # 본문은 아래에서 같은 pagekit 틀로 조립해요.
    build_article()

def build_article():
    why = """
<p><a href="fastapi.html">FastAPI 입문</a>에서는 HTTP 입구를, <a href="fastapi-postgres.html">DB 연동</a>에서는 ORM과 Session을 배웠어요. 이번에는 기능 하나를 구현한 뒤 <b>다른 사람이 같은 조건에서 실행하고, 저장 결과를 확인하고, 실패한 작업을 정리할 때까지</b> 연결해요.</p>
<p>예제는 작은 노트 서비스예요. 시작 데이터로 <b>환경 준비</b> 노트가 있고, 다음 노트를 등록한 뒤 내용을 수정하고 마지막에 삭제해요. 이 입력 하나를 끝까지 사용해요.</p>
<pre><code>{"title": "DB 세션을 서비스에 연결하기", "content": "요청마다 Session을 열고 닫아요."}</code></pre>
<p>요청 JSON은 저장할 노트 내용이에요. <b>Postman JSON</b>은 이 요청을 어디로 보내고 무엇을 검사할지 담은 클라이언트 파일이에요. 이름에 JSON이 들어가도 서로 다른 역할이에요.</p>
<p>저장은 두 곳에 해요. <code>notes</code>에는 노트, <code>changes</code>에는 그 노트를 만들거나 바꿨다는 기록을 넣어요. <code>changes.note_id</code>는 <code>notes.id</code>를 가리키는 외래 키예요. 이 둘을 함께 확정하지 않으면 노트만 저장되고 기록은 빠질 수 있어요.</p>
""" + K.out("why_state") + """
<p>앞 방식은 기록에 실패했는데도 노트가 남았어요. 뒤 방식은 함께 취소했어요. 이렇게 DB 작업을 함께 확정하거나 함께 취소할 묶음을 <b>트랜잭션</b>이라고 해요. <b>기능의 응답뿐 아니라 저장 상태도 봐야 하는 이유</b>예요. 실패를 낸 위치와 정리 방식은 ⑤에서 다시 봐요.</p>
""" + K.table(["기존 페이지의 중심", "이 외전에서 연결할 부분"], [
        ['<a href="fastapi.html">CRUD와 상태 코드</a>', '입력 계약, 업무 규칙, 저장 담당, Router를 앱에 연결하는 순서'],
        ['<a href="fastapi-postgres.html#s3">요청마다 Session 하나</a>', 'Engine과 Session의 수명, 트랜잭션 주인, SELECT 뒤 begin이 실패하는 이유'],
        ['<a href="postgres.html">SQL과 무결성</a>', '준비, 초기 데이터, 검증의 역할과 테스트 전후 상태'],
        ['<a href="llm-client.html">외부 API 호출</a>', '무료 검증과 실제 외부 연동의 검증 범위를 구분하기'],
    ])
    whole = """
<p>서비스에는 <b>실행하기 전에 준비하는 흐름</b>과 <b>실행 중 요청을 처리하는 흐름</b>이 있어요. 위에서부터 파일을 복사하는 순서만 외우면, 에러가 어느 흐름에서 났는지 놓치기 쉬워요.</p>
""" + K.table(["언제", "흐름", "끝에서 확인하는 것"], [
        ["실행 전", "같은 Python 확인 → 설정 읽기 → DB 구조 준비 → 시작 데이터 → 검증", "올바른 DB와 시작 상태"],
        ["서버 시작", "앱 생성 → 오류 처리기와 Router 등록 → 서버 수신", "live와 health의 차이"],
        ["요청 처리", "Postman → Router → Service → 저장 함수 → DB → HTTP 응답", "상태 코드와 실제 저장값"],
        ["검증 종료", "등록 → 재조회 → 실패 요청 → 재조회 → 수정 → 재조회 → 삭제 → 미존재 확인", "이번 생성 데이터 정리와 시작 수량 복귀"],
    ]) + """
<div class="flow" id="flow"></div>
<p>아래 예제는 <b>PostgreSQL을 설치하지 않아도 흐름을 재현하도록 별도 SQLite 파일을 써요.</b> Session과 트랜잭션을 연결하는 원리는 비교할 수 있지만, PostgreSQL의 연결 장애, 격리 수준, pgvector, 동시 수정은 이 실험으로 확인하지 않았어요.</p>
<p>핵심 코드는 <code>examples/service_flow/service.py</code> 하나예요. 오른쪽에서 단계별로 자라고, 같은 폴더의 <a href="../../examples/service_flow/service.py" download>전체 Python 파일</a>로도 받을 수 있어요. 설정과 검증용 파일은 <a href="../../examples/service_flow/README.md">실행 안내</a>에 있어요.</p>
<details><summary>직접 실행하는 순서 (PowerShell)</summary>
<p>저장소를 내려받은 뒤 <code>examples/service_flow</code> 폴더에서 실행해요. 처음에만 가상환경을 만들고 설정 템플릿을 복사해요. 이미 편집한 <code>.env</code>를 덮어쓰지 않아요.</p>
<pre><code>python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\\.venv\\Scripts\\python.exe service.py prepare
.\\.venv\\Scripts\\python.exe service.py serve</code></pre>
<p>브라우저에서 <code>http://127.0.0.1:8015/docs</code>를 열어요. 마지막 명령은 서버를 계속 실행하므로, 다른 명령은 두 번째 터미널에서 보내요. 서버 종료는 <code>Ctrl+C</code>예요.</p></details>
""" + K.out("health")

    steps = [
        K.step(1, "실행 설정을 한곳에서 읽기", "서버가 사용할 설정을 읽고, DB 파일의 위치를 한곳에서 결정해요.", """
<p><b>환경 변수</b>는 프로세스가 실행될 때 받는 이름과 값이에요. <code>.env</code>는 그 값을 파일로 적는 관례이며, Python이나 FastAPI가 모든 <code>.env</code>를 저절로 읽는 것은 아니에요. 이 예제는 <code>dotenv_values()</code>로 직접 읽어요.</p>
<p><b>왜?</b> 터미널을 다른 폴더에서 열었다고 새 DB가 생기면, 같은 노트를 저장했는데도 다른 터미널에서는 보이지 않아요. 그래서 상대 DB 경로는 현재 터미널 위치가 아니라 <code>service.py</code>가 있는 폴더에 붙여요.</p>
""" + K.code_table([
            ('ROOT = Path(__file__).resolve().parent', '현재 파일의 위치예요. <code>__file__</code>은 실행 중인 Python 파일 경로예요.'),
            ('config = dotenv_values(ROOT / ".env")', '설정 파일을 dict로 읽어요. 다른 프로세스나 Postman 설정은 바꾸지 않아요.'),
            ('os.environ.get("SERVICE_DEMO_DB", ...)', '이미 프로세스에 전달된 값이 있으면 우선 사용해요. 파일 설정, 기본 경로 순으로 내려가요.'),
        ]) + K.table(["이름", "누가 읽나", "담는 것"], [
            ['<code>.env.example</code>', '사람', '공유할 설정 이름과 예시. 실제 접속 비밀번호 대신 빈 값이나 안내 값'],
            ['<code>.env</code>', '서버의 설정 로더', '이 예제는 SQLite 경로. PostgreSQL 서비스라면 접속 URL, 모델 설정 등'],
            ['Postman Environment', 'Postman', '<code>base_url</code>, 응답에서 받은 생성 ID, 비교할 입력값'],
        ]) + """
<p>Postman의 <code>base_url</code>을 바꿔도 서버의 DB 접속 설정은 바뀌지 않아요. 반대로 서버의 <code>.env</code>를 고쳐도 실행 중인 프로세스가 다시 읽지 않으면 그대로예요. 이 예제는 시작 시 한 번 읽으므로 <b>서버를 재시작</b>해요.</p>
<p>패키지 설치와 서버 실행에 같은 Python을 써요. <code>python -c "import sys; print(sys.executable)"</code>로 경로를 확인하고, <code>python -m pip</code>로 그 Python에 설치해요. <code>requirements.txt</code>가 있다는 것과 설치가 끝났다는 것은 달라요.</p>
""", "s1"),
        K.step(2, "DB 준비와 검증을 서버에서 분리하기", "테이블과 시작 데이터를 준비한 뒤, 실제 저장 상태를 별도 명령으로 읽어요.", """
<p><b>스키마</b>는 테이블, 열, 제약처럼 DB의 구조예요. <b>시드 데이터</b>는 처음 기능을 확인하기 위해 넣는 시작 데이터예요. <b>무결성</b>은 외래 키 같은 관계와 규칙이 맞는 상태를 말해요.</p>
<p><b>왜?</b> 서버가 켜질 때마다 시작 데이터를 넣거나 지우면 재시작만 해도 검증 기준이 바뀌어요. <code>prepare</code>는 준비, <code>serve</code>는 요청 처리, <code>verify</code>는 상태 조회로 나눠요. 서버를 켠 사실만으로 DB 준비가 끝났다고 판단하지 않아요.</p>
""" + K.table(["역할", "일반 프로젝트의 파일 예", "이 예제"], [
            ['준비 점검', '<code>preflight.sql</code>', 'DB 파일 위치와 준비 전 Health로 구분'],
            ['구조 만들기', '<code>schema.sql</code> 또는 마이그레이션', '<code>Base.metadata.create_all()</code>'],
            ['시작 데이터', '<code>seed.sql</code>', '빈 DB에 환경 준비 노트와 생성 기록'],
            ['결과 검증', '<code>verify.sql</code>', '<code>verify()</code>로 노트, 기록, 고아 기록 수 조회'],
        ]) + K.code_table([
            ('ForeignKey("notes.id", ondelete="CASCADE")', '부모 노트를 지우면 연결된 변경 기록도 DB에서 지워요. SQLite는 연결마다 외래 키 적용을 켰어요.'),
            ('with SessionLocal.begin() as db:', '준비 작업용 Session과 트랜잭션을 열고, 정상 종료면 commit, 예외면 rollback해요.'),
            ('if not db.scalar(select(func.count()).select_from(Note)):', '노트가 하나도 없을 때만 시작 데이터를 넣어요. 데이터 초기화 명령은 아니에요.'),
        ]) + """
<p><code>create_all()</code>은 이미 있는 테이블에 새 열을 보충하지 않아요. 기존 구조를 바꾸려면 <b>마이그레이션</b>, 즉 구조 변경 이력을 적용하는 도구가 필요해요. <a href="fastapi-postgres.html#practice">DB 연동의 Alembic 설명</a>과 연결해서 보면 돼요.</p>
<p>PostgreSQL에서는 먼저 접속한 대상부터 확인해요. 다음은 그때 쓸 조회문이며, <b>이 페이지에서 실행한 결과는 아니에요.</b></p>
<pre><code>SELECT current_database(), current_user;
SELECT extname FROM pg_extension WHERE extname = 'vector';</code></pre>
<p>Python의 <code>pgvector</code> 패키지와 PostgreSQL 서버의 <code>vector</code> 확장은 설치 위치가 달라요. 확장이 활성화됐다는 것, 벡터가 적재됐다는 것, 현재 원문 버전과 벡터가 맞는다는 것도 각각 따로 확인해야 해요.</p>
""", "s2"),
        K.step(3, "요청마다 Session을 빌리고 돌려주기", "요청마다 독립적인 작업 공간을 주고, 사용이 끝나면 DB 연결을 돌려줘요.", """
<p><b>왜?</b> Session을 전역 변수 하나로 공유하면 동시에 들어온 요청이 같은 작업 공간을 건드려요. 반대로 Engine을 요청마다 만들면 연결을 재사용하는 풀이 계속 생겨요. <b>오래 공유할 것과 짧게 쓸 것을 구분</b>해야 해요.</p>
""" + K.table(["객체", "역할", "이 예제의 수명"], [
            ['<code>Engine</code>', 'DB 연결을 빌려주는 관리자. 실제 연결을 재사용하는 연결 풀을 가짐', '서버 프로세스 동안 공유'],
            ['<code>sessionmaker</code>', '같은 설정으로 Session을 만드는 공장', '프로세스 동안 공유'],
            ['<code>Session</code>', '읽은 객체와 변경 사항을 관리하는 작업 공간', 'HTTP 요청마다 새로 생성'],
            ['트랜잭션', '함께 확정하거나 함께 취소할 DB 작업의 범위', '쓰기 Service의 업무 한 묶음'],
        ]) + """
<p>여기서 Session은 로그인 정보를 보관하는 브라우저 세션이 아니에요. Session 객체를 만들었다고 바로 DB 연결을 점유하는 것도 아니며, 실제 SQL이 필요할 때 연결을 빌려요.</p>
""" + K.code_table([
            ('with SessionLocal() as db: yield db', '<code>yield</code>에서 Session을 호출한 쪽에 넘겨요. 의존성 처리가 끝나면 돌아와 <code>with</code>를 나가며 닫아요.'),
            ('db=Depends(get_db)', 'FastAPI가 이 요청에 필요한 Session을 준비해 인자에 넣어요. 이를 의존성 주입이라고 해요.'),
            ('pool_size=2, max_overflow=0, pool_timeout=2', '이 SQLite 예제의 설정값이에요. 풀의 연결을 모두 빌린 동안 새 대여는 지정 시간까지 기다려요. 운영 권장값은 아니에요.'),
        ]) + """
<p><code>with SessionLocal()</code>은 <b>닫기를 보장</b>해요. 성공한 쓰기를 알아서 commit해 주는 표현은 아니에요. 쓰기 확정은 ⑤의 <code>with db.begin()</code>이 맡아요. 닫을 때 미완료 트랜잭션은 취소되고 연결이 반환돼요.</p>
""" + K.try_("get_db에서 연결 확인용 SELECT를 먼저 실행하면?", "autobegin", after="""
<p>이미 시작된 트랜잭션 안에 또 <code>begin()</code>을 넣으면 충돌해요. Health는 자기 Session을 따로 쓰고, 일반 요청의 <code>get_db</code>는 Session을 전달하는 일만 맡겨요. 충돌을 숨기려고 무조건 <code>commit()</code>하거나 중첩 트랜잭션으로 감싸면 업무 경계가 더 흐려져요.</p>"""), "s3"),
        K.step(4, "입력 계약과 저장 담당을 정하기", "요청의 모양과 DB의 모양을 나누고, 저장 담당 함수에 트랜잭션 확정을 맡기지 않아요.", """
<p><b>계약</b>은 어떤 주소로 어떤 입력을 보내면 어떤 응답이 오는지에 대한 약속이에요. <b>Router</b>는 HTTP 입구, <b>Service</b>는 업무 순서와 규칙, <b>Repository</b>는 데이터를 읽고 쓰는 담당이에요. 폴더 이름 자체가 기능을 만드는 것은 아니고, <b>책임의 경계</b>를 드러내는 이름이에요.</p>
<p><b>왜?</b> 저장 함수가 혼자 commit하면 Service가 다른 작업과 묶어서 취소할 수 없어요. Router에 SQL과 업무 규칙이 다 있으면 HTTP 없이 그 규칙만 검사하기도 어려워져요.</p>
""" + K.table(["보통 나누는 위치", "하는 일", "예제의 이름"], [
            ['<code>core/config.py</code>', '환경 설정', '<code>config</code>, <code>DB_FILE</code>'],
            ['<code>models/</code>', 'ORM 모델, DB 행의 구조', '<code>Note</code>, <code>Change</code>'],
            ['<code>schemas/</code>', 'HTTP 입력과 출력의 구조', '<code>NoteWrite</code>, <code>NoteRead</code>'],
            ['<code>repositories/</code>', '조회, INSERT, 변경 기록. commit은 하지 않음', '<code>require_note</code>, <code>insert_note</code>, <code>record_change</code>'],
            ['<code>services/</code>', '노트 저장과 기록을 업무 하나로 묶음', '⑤의 <code>create_note</code>, <code>update_note</code>'],
            ['<code>api/</code>', 'HTTP 입력을 Service에 전달', '⑥의 <code>post_note</code>, <code>put_note</code>'],
            ['<code>main.py</code>', '앱과 Router, 오류 처리기를 연결', '⑥의 <code>app.include_router(router)</code>'],
        ]) + """
<p>이 페이지는 코드를 오가며 찾지 않도록 위 역할을 <b>파일 하나의 함수와 클래스로</b> 보여 줘요. 프로젝트가 커지면 위 위치로 옮길 수 있지만, 파일 개수를 늘리는 것이 목표는 아니에요.</p>
""" + K.code_table([
            ('class NoteWrite(BaseModel):', '클라이언트가 보내는 title, content의 자료형과 길이를 검사해요. 요청에는 DB가 만들 id가 없어요.'),
            ('class NoteRead(NoteWrite): id: int; revision: int', '응답에는 저장된 번호와 수정 버전을 더해요. 이 코드에서는 실제로 줄을 나눠 선언했어요.'),
            ('extra="forbid"', '계약에 없는 필드를 조용히 무시하지 않고 입력 오류로 알려요.'),
            ('db.flush()', '모아 둔 INSERT를 DB로 보내 번호와 제약 위반을 확인해요. 아직 commit한 것은 아니에요.'),
        ]) + """
<p>Pydantic은 <code>content</code>가 빠졌는지 검사할 수 있지만 같은 제목이 DB에 있는지는 몰라요. 입력 검사를 통과한 것과 DB 저장 성공은 별개의 단계예요.</p>
""", "s4"),
        K.step(5, "Service가 트랜잭션을 책임지기", "노트와 변경 기록을 함께 확정하고, 하나라도 실패하면 둘 다 취소해요.", """
<p><b>왜?</b> HTTP 201을 먼저 돌려주고 나서 commit에 실패하면 사용자는 저장됐다고 믿는데 DB에는 없어요. 반대로 노트만 먼저 commit하면 변경 기록 실패를 되돌릴 수 없어요. <b>쓰기 작업을 묶고 commit이 성공한 다음 응답</b>해야 해요.</p>
""" + K.code_table([
            ('with db.begin():', '쓰기 업무의 시작과 끝이에요. 정상적으로 블록을 나가면 commit, 예외로 나가면 rollback해요.'),
            ('note = insert_note(db, payload)', '실제 INSERT와 flush를 해요. 아직 다른 연결에서 확정된 저장값으로 볼 단계가 아니에요.'),
            ('record_change(db, note, "created")', '같은 Session의 같은 트랜잭션에 기록을 추가해요.'),
            ('return result', 'begin 블록 바깥에 있어요. commit이 실패하면 이 줄에 도착하지 않아요.'),
        ]) + """
<p>수정도 <code>with db.begin()</code> 안에서 먼저 노트를 찾아요. 조회를 블록 밖으로 빼면 ③의 autobegin 때문에 이미 시작된 트랜잭션에 또 begin하려고 할 수 있어요. <code>expire_on_commit=False</code>는 commit 뒤에도 읽은 속성을 유지하는 설정이며, 자동 commit 설정이 아니에요.</p>
<p>제목 중복은 DB의 <code>UNIQUE</code> 제약으로 막고, 확인한 제목 충돌만 <code>409 / TITLE_CONFLICT</code>로 바꿔요. 모든 DB 오류를 중복이라고 처리하면 연결 장애와 프로그램 오류까지 입력 탓으로 보이게 돼요. PostgreSQL에서는 오류 코드와 제약 이름으로 원인을 분류해요.</p>
<p>아래 실험은 노트 INSERT가 끝난 다음 변경 기록 함수에서 예외를 냈어요. 그 뒤 <b>별도의 Session</b>으로 실제 DB 수량을 확인하고 정상 요청도 다시 실행했어요.</p>
""" + K.out("s5") + """
<p>이 예제의 <code>changes</code>는 동작 확인용 변경 기록이에요. 노트 삭제 시 같이 지우므로, 삭제 이력도 영구 보존해야 하는 감사 로그의 설계와는 목적이 달라요.</p>
<p>LLM처럼 오래 걸리는 외부 요청까지 DB 트랜잭션에 넣으면 대기하는 동안 연결을 붙잡을 수 있어요. 원문과 버전을 짧게 읽고 Session을 닫은 뒤 외부 계산을 수행하고, 저장할 때 <code>revision</code>이 그대로인지 다시 확인하는 식으로 경계를 설계해요. 그 사이 수정과 충돌하는 정책은 별도 구현과 동시성 검증이 필요해요.</p>
"""),
        K.step(6, "Router와 Health를 앱에 연결하기", "업무 함수를 HTTP 입구에 붙이고, 프로세스 응답과 서비스 준비 상태를 나눠 확인해요.", """
<p><b>왜?</b> 함수 파일이 있어도 앱에 Router를 등록하지 않으면 그 주소로 요청할 수 없어요. 서버가 응답해도 필요한 테이블이 없다면 저장 기능은 쓸 수 없어요.</p>
""" + K.code_table([
            ('router = APIRouter(prefix="/notes")', '관련 주소를 묶어요. 실제 코드의 tags는 자동 문서에서 묶어 보여 주는 이름이에요.'),
            ('app.include_router(router)', '위에서 정의한 주소 묶음을 앱에 등록해요. 함수 선언만으로 별도 앱에 자동 연결되지는 않아요.'),
            ('@app.exception_handler(DomainError)', '업무 오류를 같은 모양의 JSON과 상태 코드로 바꿔요.'),
            ('async def lifespan(app): ... finally: engine.dispose()', '앱 시작과 종료에 맞춰 공유 자원을 관리하는 수명 주기 함수예요. 종료 때 연결 풀을 정리해요.'),
        ]) + """
<p><code>GET /live</code>는 앱이 응답하는지만 봐요. <code>GET /health</code>는 자기 Session으로 DB 연결과 노트, 변경 기록 테이블 조회까지 확인해요. 여기서는 이 주소를 <b>readiness</b>, 즉 요청을 처리할 준비 점검으로 사용해요. Health라는 이름만으로 어디까지 검사하는지 정해지는 것은 아니에요.</p>
""" + K.out("health") + """
<p>HTTP <code>409</code>는 제목 충돌, <code>422</code>는 입력 계약 위반, <code>404</code>는 노트 없음이에요. 오류 본문은 <code>{"error": {"code": "...", "message": "..."}}</code>로 맞췄어요. FastAPI 기본 입력 오류의 본문 모양은 다르므로 <code>RequestValidationError</code> 처리기도 직접 등록했어요.</p>
<p>실제 서버 실행은 <code>service.py serve</code> 명령이에요. 일반 패키지 프로젝트에서는 <code>python -m app.main</code>처럼 패키지를 기준으로 실행하면 import 경로를 일정하게 잡을 수 있어요. <code>__init__.py</code>는 일반 Python 패키지를 표시하는 빈 파일로 둘 수 있고, <code>if __name__ == "__main__"</code> 아래 코드는 직접 실행했을 때만 동작해요.</p>
<p>새 주소가 <code>/docs</code>에 없다면 먼저 Router 등록, 실행 중인 앱, 재시작 여부를 봐요. 다른 서버가 같은 포트에서 돌고 있으면 새 코드와 전혀 다른 응답을 받을 수 있어요.</p>
""", "s6"),
        K.step(7, "클라이언트의 검증 순서를 파일로 공유하기", "Postman 파일 두 개로 같은 요청 순서를 재현하고, 생성 ID를 다음 요청으로 넘겨요.", """
<p><b>Postman</b>은 HTTP 요청을 보내고 응답을 검사하는 클라이언트예요. <b>Collection</b>은 요청 URL, 방식, Body와 검사 스크립트를 묶은 파일이고, <b>Environment</b>는 그 요청에 채울 변수의 묶음이에요.</p>
<p><b>왜?</b> 매번 URL과 JSON을 손으로 입력하면 생성 ID를 놓치거나, 잘못된 서버에 보내거나, 실패한 수정을 성공으로 착각해요. 요청과 기대 결과를 파일로 남기면 다른 사람도 같은 순서로 확인할 수 있어요. 이 파일이 서버나 DB를 대신 만들어 주는 것은 아니에요.</p>
<p><a href="../../examples/service_flow/service.postman_collection.json" download>Collection JSON 다운로드</a> / <a href="../../examples/service_flow/service.postman_environment.json" download>Environment JSON 다운로드</a>. 두 파일을 Postman의 Import로 읽고 <b>노트 서비스 - Local</b> 환경을 선택해요. Collection Runner에서 위에서 아래 순서로 실행해요.</p>
""" + K.table(["파일 속 위치", "뜻", "이 예제"], [
            ['Collection의 <code>item</code>', '순서대로 실행할 요청 목록', '준비 확인부터 삭제 후 확인까지'],
            ['<code>request.method</code>, <code>request.url</code>', '요청 방식과 주소', '<code>POST {{base_url}}/notes</code>'],
            ['<code>request.body.raw</code>', '보낼 JSON을 담은 문자열', 'title, content가 있는 노트 입력'],
            ['<code>event</code>의 <code>listen: "test"</code>', '응답 뒤 실행할 검사 스크립트', 'Postman 화면의 Scripts → Post-response'],
            ['Environment의 <code>values</code>', '이름과 값의 목록', '<code>base_url</code>, <code>created_id</code> 등'],
        ]) + """
<p>URL의 <code>{{base_url}}</code>은 보내기 전에 선택된 변수 값으로 치환돼요. 다른 범위에 같은 이름의 변수가 있으면 우선순위 때문에 기대와 다른 값을 쓸 수 있으므로, 이 예제는 변수를 선택한 Environment에 모아요. 실제 전송 URL에 중괄호가 남아 있다면 환경 선택과 값부터 확인해요.</p>
<p>등록 응답에서 받은 <code>id</code>를 저장하고, 이후 요청은 <code>{{created_id}}</code>를 써요. ID를 특정 숫자로 예상하지 않아요. <b>snapshot</b>은 비교용으로 남겨 둔 값의 사본이에요. 여기서는 등록 또는 수정 응답을 <code>expected_note</code>에 저장하고 새 GET 응답과 비교해요. POST와 PUT 응답 자체도 보낸 입력과 따로 비교해요.</p>
<pre><code>pm.environment.set("created_id", String(pm.response.json().id));
pm.environment.set("expected_note", JSON.stringify(pm.response.json()));
pm.test("저장값 재조회", () =&gt; {
  pm.expect(pm.response.json()).to.eql(
    JSON.parse(pm.environment.get("expected_note"))
  );
});</code></pre>
<p>위 코드는 핵심을 모아 보여 준 조각이에요. 실제 파일에서는 저장 스크립트는 POST와 PUT 뒤에, 비교 스크립트는 GET 뒤에 있어요. <code>pm.test</code>는 검사 이름과 함수를 등록하고, <code>pm.expect</code>는 실제 값과 기대 값을 비교해요. <code>to.eql</code>은 객체 내부 필드까지 비교해요.</p>
""" + K.table(["순서", "검증 의도", "다음 요청에 남기는 것"], [
            ['준비 확인, 목록 조회', 'DB 준비와 시작 수량 확인', '<code>baseline_count</code>'],
            ['등록, 재조회', '입력값이 실제 저장됐는지', '<code>created_id</code>, <code>expected_note</code>'],
            ['중복 등록, 잘못된 수정, 재조회', '409와 422가 정확히 나오고 원래 값은 유지되는지', '기존 snapshot 유지'],
            ['정상 수정, 재조회', '내용과 revision이 실제 바뀌었는지', '새 snapshot'],
            ['삭제, 삭제 후 조회', '204의 빈 본문과 404 확인', '404 확인 뒤 생성 ID와 snapshot 제거'],
            ['마지막 목록 조회', '시작 노트 수로 돌아왔는지', '이번 생성 데이터 정리 완료'],
        ]) + """
<p>삭제 직후에는 ID를 비우지 않아요. 그 ID로 GET을 보내 404까지 확인한 다음 비워요. 중간에 실패하면 남은 ID를 먼저 조회해 상태를 파악해요. <code>seed.sql</code>을 다시 실행하는 것은 검증용 데이터 정리와 같은 일이 아니에요.</p>
""" + K.out("s7") + """
<p><b>Newman</b>은 v2.1 형식의 Postman Collection을 터미널에서 실행하는 도구예요. 이 예제의 두 파일은 그 형식으로 만들었어요. 서버를 실행한 상태에서 같은 폴더의 두 번째 터미널로 실행할 수 있어요.</p>
<pre><code>npx --yes newman run service.postman_collection.json -e service.postman_environment.json --bail</code></pre>
<p><code>--bail</code>은 실패 시 진행을 멈추게 해요. 중간 요청만 빼서 실행하거나 검사를 건너뛴 결과를 전체 성공으로 기록하지 않아요. Postman의 Test Results와 실패한 요청의 응답 본문을 함께 읽어요.</p>
""" + K.out("newman") + """
<p>마지막으로 <code>service.py verify</code>로 DB 수량과 고아 기록도 직접 확인해요. 상태 코드와 JSON 검사, 실제 저장값, 관계 무결성은 서로 다른 증거예요. 검색 관련성이나 자연어 요약의 근거까지 자동 형식 검사만으로 확인했다고 말할 수는 없어요.</p>
"""),
    ]
    path = K.build(ROOT / "topics/backend/service-flow.html", title="백엔드 외전: 로직 밖에서 서비스를 완성하기",
        h1="로직 밖에서 서비스를 완성하기", track="백엔드", crumb="외전 / 실행 준비에서 저장 검증과 정리까지",
        one_line="서비스 구현은 <b>기능 코드에 실행 조건, 자원 수명, 저장 책임, 검증 순서를 연결</b>해서 같은 결과를 다시 확인할 수 있게 만드는 일이에요.",
        why=why, whole=whole, steps=steps, code_file="examples/service_flow/service.py",
        practice=[
            ["패키지를 설치했는데 import 실패", "설치와 실행의 <code>sys.executable</code>을 비교해요. 같은 Python으로 <code>-m pip</code>와 서버를 실행해요."],
            ["설정을 바꿨는데 응답이 그대로", "어느 파일의 설정인지 먼저 나눠요. 서버 .env는 서버 재시작, Postman 값은 선택한 환경과 실제 전송 URL을 확인해요."],
            ["transaction already begun", "같은 Session에서 먼저 실행된 SELECT가 있는지 봐요. get_db는 전달과 정리만, 쓰기 Service는 첫 조회부터 begin 안에서 처리해요."],
            ["rollback 뒤 같은 Session을 계속 사용", "실패 상태를 끝내지 않으면 다음 SQL도 실패할 수 있어요. 이 예제는 begin 문맥이 예외 시 rollback하고 요청 종료 시 Session을 닫아요."],
            ["요청이 많아지면 연결이 부족", "Session 반납 누락과 긴 트랜잭션부터 확인해요. 풀은 프로세스마다 있으므로 worker 수까지 합쳐 DB 연결 예산을 잡아요."],
            ["백그라운드 작업에서 DB 사용", "요청의 Session을 넘겨 공유하지 않아요. 작업이 자기 Session과 트랜잭션을 만들고 닫도록 해요."],
            ["Health는 성공인데 검색이 안 됨", "Health가 실제로 검사한 범위를 봐요. 확장 활성화, 벡터 수량, 모델명, 원문과 인덱스 revision 일치, 검색 관련성은 따로 확인해요."],
            ["Postman 요청을 반복할수록 데이터가 늘어남", "이전 created_id와 응답 이력을 먼저 찾아요. 정리가 확인되지 않았는데 POST부터 다시 보내지 않아요."],
            ["실패 응답을 받았는데 일부 저장됐는지 의심", "새 요청으로 다시 읽고 관련 테이블도 조회해요. 입력 거부 시험과 INSERT 뒤 장애를 낸 롤백 시험은 검증 범위가 달라요."],
            ["LLM이 없는 테스트는 통과", "입력 계약, DB 연동, 실제 모델 호출, 답변 품질을 구분해 기록해요. 모의 응답 성공은 실제 외부 연동 성공의 증거가 아니에요."],
            ["날짜 문자열은 다른데 같은 시각인가", "시간대가 있는 값을 같은 시각으로 변환해 비교해요. +09:00 입력과 UTC 응답은 문자열 모양이 달라도 같은 순간일 수 있어요."],
            ["PostgreSQL로 바꾸려면", "접속 URL과 드라이버, 스키마 준비, 오류 분류를 바꾸고 실제 DB에서 다시 검증해요. SQLite의 UNIQUE 오류 문자열과 연결 설정을 그대로 옮기지 않아요."],
        ], libs=[
            ["①", "<code>Path</code>, <code>dotenv_values()</code>, <code>os.environ.get()</code>", "파일 위치와 서버 설정 읽기"],
            ["②~③", "<code>create_engine()</code>, <code>sessionmaker()</code>, <code>ForeignKey()</code>", "DB 연결, 요청의 작업 공간과 관계 규칙"],
            ["④", "<code>BaseModel</code>, <code>Field()</code>, <code>ConfigDict()</code>", "Pydantic 입력 계약"],
            ["⑤", "<code>begin()</code>, <code>flush()</code>", "SQLAlchemy의 업무 트랜잭션"],
            ["⑥", "<code>FastAPI()</code>, <code>APIRouter()</code>, <code>Depends()</code>, <code>JSONResponse()</code>", "HTTP 입구와 공통 응답"],
            ["⑥", "<code>asynccontextmanager</code>, <code>engine.dispose()</code>", "앱 종료 시 공유 자원 정리"],
            ["⑦", "<code>json.dumps()</code>, <code>pm.test()</code>, <code>pm.environment.set()</code>", "파일 생성은 Python, 응답 검사는 Postman JavaScript"],
        ], libs_note="결과 수집은 빌드 스크립트에서 TestClient와 별도 SQLite 파일을 사용해요. --newman 옵션은 실제 Uvicorn 서버와 Newman까지 실행해요.",
        next_rows=[
            ['<a href="fastapi-postgres.html">FastAPI와 PostgreSQL 연동</a>', '실제 PostgreSQL의 ORM, Session과 DB 오류를 더 자세히 봐요.'],
            ['<a href="indexing.html">인덱싱</a>', '원문과 벡터 버전을 맞추고 검색 가능한 상태로 공개하는 흐름이에요.'],
            ['<a href="llm-client.html">LLM API 클라이언트</a>', '외부 API의 오류, 시간 제한, 출력 계약과 연결해요.'],
        ], sources=[
            '정리의 출발점: Day25 (2026-11-20), 민준튜터가 정리하는 프로젝트 구현 흐름 익히는 비법 안내서. 원본 HTML은 공개 저장소에 포함하지 않았어요.',
            '가상의 입력과 독립 예제: <a href="../../examples/service_flow/service.py">service.py</a>. 원본 프로젝트 코드, DB 이름, 시작 데이터와 요청 수를 복제하지 않았어요.',
            '실행 결과와 장애 주입: <code>tools/pages/service_flow.py</code>. 실제 PostgreSQL, 외부 모델, 동시 부하는 이 페이지의 실행 검증 범위에 없어요.',
            '공식 문서: <a href="https://docs.sqlalchemy.org/en/21/orm/session_basics.html">SQLAlchemy Session Basics</a>, <a href="https://fastapi.tiangolo.com/tutorial/dependencies/dependencies-with-yield/">FastAPI Dependencies with yield</a>, <a href="https://fastapi.tiangolo.com/advanced/events/">Lifespan Events</a>.',
            '공식 문서: <a href="https://learning.postman.com/docs/use/send-requests/variables/variables/">Postman Variables</a>, <a href="https://learning.postman.com/docs/tests-and-scripts/write-scripts/test-scripts/">Post-response Scripts</a>, <a href="https://learning.postman.com/docs/reference/newman-cli/command-line-integration-with-newman/">Newman CLI</a>.',
            '공식 문서: <a href="https://www.postgresql.org/docs/current/functions-info.html">PostgreSQL 접속 정보 함수</a>, <a href="https://www.postgresql.org/docs/current/catalog-pg-extension.html">설치된 확장 카탈로그</a>.',
        ])
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--newman", action="store_true")
    generate(parser.parse_args().newman)
