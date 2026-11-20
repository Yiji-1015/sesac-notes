"""주제 페이지 '백엔드 > API 서버 > FastAPI로 지식 API 만들기'를 만든다.

examples/api/main.py 를 단계별로 실제 실행하고(TestClient), 그 결과로 페이지와 -data.js 를 쓴다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/api_server.py
"""
import json
import sys
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import fastapi  # noqa: E402
import pydantic  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "api"
code = S.parse((EX / "main.py").read_text(encoding="utf-8"))
NEW = {"title": "실습 환경 안내", "category": "setup", "content": "Python 3.10 이상을 사용한다."}
outs, flow = {}, []


def app_at(k, wd=None):
    wd = wd or S.workdir(EX)
    ns = S.run(code, k, wd, filename="main.py")
    return ns, TestClient(ns["app"], raise_server_exceptions=False), wd


def j(x, n=None):
    s = json.dumps(x, ensure_ascii=False)
    return s if n is None or len(s) <= n else s[:n] + "…"


def call(c, method, url, body=None):
    r = c.request(method, url, json=body)
    try:
        b = r.json()
    except Exception:  # noqa: BLE001
        b = r.text
    return r.status_code, b


# ---- 왜: 파일을 직접 고치면 vs API로 보내면 -----------------------------------
ns, c, wd = app_at(6)
data = json.loads((wd / "data" / "knowledge.json").read_text(encoding="utf-8"))
data["entries"].append({"id": 99, "title": "카테고리 빠짐", "content": "누군가 파일을 직접 고쳤다"})
(wd / "data" / "knowledge.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
st, body = call(c, "GET", "/knowledge?category=api")
ns2, c2, wd2 = app_at(6)
st2, body2 = call(c2, "POST", "/knowledge", {"title": "카테고리 빠짐", "content": "API로 보냈다"})
after = len(json.loads((wd2 / "data" / "knowledge.json").read_text(encoding="utf-8"))["entries"])
outs["why_file"] = S.box("파일에 category 없는 항목을 직접 넣은 뒤 GET /knowledge?category=api", [
    (f"상태 코드 {st}", 1, "bad"), f"응답 본문: {body if isinstance(body, str) else j(body)}"],
    ["파일은 아무 모양이든 받아 줘서, 잘못된 항목이 그대로 저장됐어요. 그걸 읽는 순간 서버 코드가 <code>e[\"category\"]</code>에서 멈춰서 <b>모든 요청</b>이 500(서버 내부 오류)이 돼요."])
outs["why_api"] = S.box("같은 내용을 API로 POST", [
    (f"상태 코드 {st2}", 1), f"응답: {j(body2['detail'][0], 140)}", (f"파일의 항목 수: {after}개 (그대로)", 2)],
    ["API는 저장하기 <b>전에</b> 입력 모양을 검사해서 422(입력 오류)로 돌려보내요. 무엇이 틀렸는지도 알려 줘요 (<code>category</code> 빠짐).",
     "잘못된 항목은 파일에 들어가지 않았어요."])

# ---- ① ------------------------------------------------------------------------
ns, c, wd = app_at(1)
st, body = call(c, "GET", "/health")
paths = sorted(c.get("/openapi.json").json()["paths"])
outs["s1"] = S.box("TestClient로 GET /health", [(f"상태 코드 {st}", 1), (f"응답 {j(body)}", 2), f"GET /docs → {c.get('/docs').status_code} (자동 문서 화면)", f"openapi.json의 경로 → {paths}"],
                   ["200은 \"성공\"이라는 뜻이에요.", "함수가 돌려준 dict가 JSON으로 바뀌어 왔어요. 바꾸는 일은 FastAPI가 해요."])

# ---- ② ------------------------------------------------------------------------
ns, c, wd = app_at(2)
d = ns["read_data"]()
outs["s2"] = S.box("read_data()", [(f"next_id → {d['next_id']}", 1), (f"entries → {len(d['entries'])}개", 2), f"entries[0] → {j(d['entries'][0])}"],
                   ["다음에 등록할 항목이 받을 번호. 지운 번호를 다시 쓰지 않게 따로 세요.", "아직 API 경로는 /health 하나라서, 이 데이터를 밖에서 볼 방법은 없어요. ③에서 열어요."])

# ---- ③ ------------------------------------------------------------------------
ns, c, wd = app_at(3)
rows = []
for m, u in [("GET", "/knowledge"), ("GET", "/knowledge?category=api"), ("GET", "/knowledge/2"), ("GET", "/knowledge/99")]:
    st, b = call(c, m, u)
    rows.append([f"<code>{m} {E(u)}</code>", f"<b>{st}</b>", f"<code>{E(j(b, 120))}</code>"])
outs["s3"] = S.table("③ 실행 결과: 요청 네 개", ["요청", "상태", "응답"], rows,
                     ["<code>?category=api</code>처럼 <code>?</code> 뒤에 붙는 값은 함수 인자 <code>category</code>로 들어와요 (쿼리 매개변수).",
                      "<code>/knowledge/2</code>의 <code>2</code>는 경로 안의 <code>{knowledge_id}</code>로 들어와요 (경로 매개변수). 타입이 <code>int</code>라서 FastAPI가 숫자로 바꿔 줘요.",
                      "없는 번호는 <code>HTTPException(404)</code>로 \"없음\"을 알려요. 빈 응답이나 200을 주면 클라이언트가 \"있는데 비었다\"로 오해해요."])
st, b = call(c, "GET", "/knowledge/abc")
outs["t3"] = S.box("GET /knowledge/abc", [(f"상태 코드 {st}", 1), f"응답: {j(b['detail'][0], 160)}"],
                   ["경로의 <code>abc</code>를 <code>int</code>로 바꿀 수 없어서, 함수가 불리기도 전에 FastAPI가 422로 돌려보냈어요. 타입을 적어 두는 것만으로 검사가 생겨요."])

# ---- ④ ------------------------------------------------------------------------
ns, c, wd = app_at(4)
st, b = call(c, "POST", "/knowledge", NEW)
f = json.loads((wd / "data" / "knowledge.json").read_text(encoding="utf-8"))
outs["s4"] = S.box(f"POST /knowledge  본문 {j(NEW)}", [(f"상태 코드 {st}", 1), (f"응답 {j(b)}", 2), (f"파일: next_id {f['next_id']}, 항목 {len(f['entries'])}개", 3)],
                   ["201은 \"새로 만들었다\"는 뜻이에요. <code>status_code=201</code>로 정했어요.", "보낸 세 필드에 서버가 정한 <code>id</code>가 붙었어요. 번호는 클라이언트가 정하지 않아요.", "파일에도 저장됐어요."])
bad = [("title이 빈 글", {**NEW, "title": ""}), ("category 빠짐", {"title": "x", "content": "y"}), ("title이 숫자", {**NEW, "title": 123})]
rows = []
for name, body in bad:
    st, b = call(c, "POST", "/knowledge", body)
    e0 = b["detail"][0]
    rows.append([E(name), f"<b>{st}</b>", f"<code>{E(str(e0['loc']))}</code>", E(e0["msg"])])
f2 = json.loads((wd / "data" / "knowledge.json").read_text(encoding="utf-8"))
S.claim(all(r_[1] == "<b>422</b>" for r_ in rows), "④ 잘못된 POST는 세 번 모두 422")
outs["t4"] = S.table("잘못된 POST 세 번", ["보낸 것", "상태", "어디가 (loc)", "왜 (msg)"], rows,
                     [f"세 번 모두 422였고, 파일 항목 수는 {len(f2['entries'])}개 그대로예요. 검사에 걸리면 <code>create_knowledge</code> 함수 자체가 불리지 않아요."])

# ---- ⑤ ------------------------------------------------------------------------
ns, c, wd = app_at(5)
call(c, "POST", "/knowledge", NEW)
upd = {**NEW, "content": "프로젝트별 가상환경을 사용한다."}
st, b = call(c, "PUT", "/knowledge/4", upd)
st2, b2 = call(c, "PUT", "/knowledge/99", upd)
outs["s5"] = S.box("PUT /knowledge/4 (content만 바꿔서 전체를 다시 보냄)", [(f"상태 코드 {st}", 1), f"응답 {j(b)}", (f"PUT /knowledge/99 → {st2} {j(b2)}", 2)],
                   ["PUT은 \"이 번호의 항목을 이 내용으로 통째로 바꿔라\"예요. 바꾸지 않는 필드도 다 보내요. 입력 검사는 ④의 <code>KnowledgeWrite</code>를 그대로 써요.", "없는 번호는 조회와 똑같이 404예요."])

# ---- ⑥ ------------------------------------------------------------------------
ns, c, wd = app_at(6)
call(c, "POST", "/knowledge", NEW)
r = c.delete("/knowledge/4")
r2 = c.delete("/knowledge/4")
ids = [e["id"] for e in c.get("/knowledge").json()]
st3, b3 = call(c, "POST", "/knowledge", NEW)
S.claim(r.status_code == 204 and r2.status_code == 404 and 4 not in ids and b3["id"] > 4,"⑥ 지우면 204, 다시 지우면 404, 새 항목은 지운 번호를 다시 쓰지 않는다")
outs["s6"] = S.box("DELETE /knowledge/4 를 두 번", [(f"첫 번째 → {r.status_code}, 본문 {len(r.content)}바이트", 1), (f"두 번째 → {r2.status_code} {r2.json()}", 2),
                    f"남은 id → {ids}", (f"다시 등록하면 받는 id → {b3['id']}", 3)],
                   ["204는 \"성공했고 돌려줄 내용은 없다\"예요. 그래서 본문이 비어 있어요.", "이미 지운 번호라 404예요.", f"4번은 지웠지만 <code>next_id</code>는 {b3['id']}라서 새 항목은 {b3['id']}번이에요. 지운 번호를 다시 쓰면, 옛 4번을 가리키던 링크가 엉뚱한 항목을 열게 돼요."])
codes = [["<code>GET /knowledge</code>", "200", "—", "—"], ["<code>GET /knowledge/{id}</code>", "200", "404 (없음), 422 (숫자 아님)", "③"],
         ["<code>POST /knowledge</code>", "201", "422 (입력 모양 틀림)", "④"], ["<code>PUT /knowledge/{id}</code>", "200", "404, 422", "⑤"],
         ["<code>DELETE /knowledge/{id}</code>", "204", "404", "⑥"]]
outs["codes"] = S.table("이 API의 상태 코드 한눈에", ["요청", "성공", "실패", "나온 곳"], codes)

flow = [[1, "앱", "/health", "FastAPI()"], [2, "저장소", "JSON 파일", "읽기, 쓰기"], [3, "조회", "GET", "목록, 상세, 404"],
        [4, "등록", "POST", "Pydantic 검사, 201"], [5, "수정", "PUT", "통째로 바꾸기"], [6, "삭제", "DELETE", "204, 번호 재사용 안 함"]]

page = {"code": [c_ for c_ in code], "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
        "env": f"Python {sys.version.split()[0]}, FastAPI {fastapi.__version__}, Pydantic {pydantic.__version__}, TestClient(httpx)"}
S.save(ROOT / "topics" / "backend" / "fastapi-data.js", page)

# ============================== 페이지 글 ==============================
why = f"""
<p><b>API</b>는 다른 프로그램이 이 서비스의 기능을 쓰는 <b>정해진 입구</b>예요. 웹 화면, 다른 서버, AI Agent가 모두 같은 입구로 "지식 목록 줘", "이거 저장해 줘"를 요청해요. 요청과 응답은 보통 <b>HTTP</b>(웹 주소로 요청을 보내고 답을 받는 규칙)로 주고받아요.</p>
<p>예제는 지식 항목(제목, 분류, 내용)을 JSON 파일에 저장하는 작은 서비스예요. 파일은 이렇게 생겼어요.</p>
{K.table(["id", "title", "category", "content"], [[1, "가상환경", "setup", "프로젝트별 패키지를 분리한다."], [2, "서버 실행", "api", "uvicorn으로 개발 서버를 켠다."], [3, "입력 오류", "api", "422이면 입력값을 확인한다."]])}
<p>그냥 각 프로그램이 이 파일을 직접 열어 읽고 쓰면 안 될까요? 누군가 <code>category</code>를 빠뜨린 항목을 파일에 직접 넣으면 이렇게 돼요.</p>
<div class="twocol">{K.out("why_file")}{K.out("why_api")}</div>
<p>입구를 하나로 모으면, <b>들어오는 모든 데이터를 같은 규칙으로 검사</b>하고, 저장 방식이 파일에서 DB로 바뀌어도 바깥 프로그램은 몰라도 돼요. 여기서는 파이썬 웹 프레임워크 <b>FastAPI</b>로 이 입구를 만들어요.</p>
<p><b>그래서 목표는</b> 지식을 조회, 등록, 수정, 삭제하는 입구를 만들고, 잘못된 요청은 저장 전에 막고, 결과를 알맞은 <b>상태 코드</b>(요청이 어떻게 됐는지 알려 주는 세 자리 숫자)로 알려 주는 거예요.</p>
"""
whole = f"""
<p>요청 하나가 들어오면 FastAPI가 <b>주소와 방식</b>(GET, POST 등)으로 함수를 찾고, 입력을 검사한 뒤 함수를 불러요. 함수는 파일을 읽고 쓰고, 돌려준 값은 JSON 응답이 돼요. 상자를 누르면 그 단계로 가요.</p>
<div class="flow" id="flow"></div>
{K.out("codes")}
<p>아래에서 ①부터 ⑥까지, <code>main.py</code> 파일 하나가 자라는 모습을 따라가요. 결과는 모두 FastAPI의 <code>TestClient</code>(서버를 띄우지 않고 요청을 보내 보는 도구)로 실제로 보낸 요청이에요.</p>
"""
steps = [
    K.step(1, "앱 만들기", "<code>FastAPI()</code>로 앱을 만들고, 살아 있는지 확인하는 <code>/health</code> 주소 하나를 열어요.",
           "<p>함수 위에 <code>@app.get(\"/health\")</code>를 붙이면 \"<code>GET /health</code> 요청이 오면 이 함수를 불러라\"라는 뜻이 돼요. 이렇게 함수를 주소에 연결하는 걸 <b>라우팅</b>이라고 해요.</p>"
           + K.code_table([("app = FastAPI(title=...)", "앱 하나. 모든 주소가 여기에 등록돼요. <code>title</code>은 자동 문서에 보이는 이름"),
                           ('@app.get("/health")', "GET 방식의 <code>/health</code> 요청을 아래 함수에 연결"),
                           ('return {"status": "ok"}', "dict를 돌려주면 FastAPI가 JSON 응답으로 바꿔요."),
                           ("uvicorn main:app --reload", "실제로 서버를 띄우는 명령. <code>main.py</code> 안의 <code>app</code>을 실행하고, 코드가 바뀌면 다시 불러요(개발용). 띄운 뒤 <code>/docs</code>에서 모든 주소를 눌러 볼 수 있어요.")]), "s1"),
    K.step(2, "파일 저장소", "JSON 파일을 통째로 읽고 쓰는 함수 두 개를 만들어요.",
           "<p>파일 모양은 <code>{\"next_id\": 4, \"entries\": [...]}</code>예요. 항목 목록과 함께 \"다음 번호\"를 저장해 둬요.</p>"
           + K.code_table([('Path(__file__).resolve().parent / "data"', "이 파일(<code>main.py</code>)이 있는 폴더 기준 경로. 어디서 실행해도 같은 파일을 가리켜요."),
                           ('read_text(encoding="utf-8")', "한글이 들어 있으니 인코딩을 꼭 적어요."),
                           ("json.dumps(data, ensure_ascii=False, indent=2)", "한글을 그대로, 사람이 읽기 좋게 들여 써서 저장")]), "s2"),
    K.step(3, "조회하기 (GET)", "목록 주소와 번호 하나 주소를 만들어요. 없는 번호는 404로 알려요.",
           "<p>주소에는 값을 두 가지 방법으로 실을 수 있어요. 주소 안에 넣는 <b>경로 매개변수</b>(<code>/knowledge/2</code>)와, <code>?</code> 뒤에 붙이는 <b>쿼리 매개변수</b>(<code>?category=api</code>)예요. 보통 \"무엇을\"은 경로에, \"어떻게 거를지\"는 쿼리에 둬요.</p>"
           + K.code_table([('@app.get("/knowledge/{knowledge_id}")', "중괄호 자리가 경로 매개변수. 같은 이름의 함수 인자로 들어와요."),
                           ("knowledge_id: int", "타입을 적으면 FastAPI가 숫자로 바꾸고, 못 바꾸면 422로 돌려보내요."),
                           ("category: str | None = None", "기본값이 있는 인자는 쿼리 매개변수. 안 보내면 <code>None</code>"),
                           ('raise HTTPException(status_code=404, detail=...)', "함수를 멈추고 404 응답을 보내요. <code>detail</code>이 응답 본문에 들어가요.")])
           + K.try_("<code>GET /knowledge/abc</code>처럼 숫자가 아닌 번호를 보내면?", "t3"), "s3"),
    K.step(4, "입력 모양 정하고 등록하기 (POST)", "Pydantic으로 \"등록할 때 보내야 하는 모양\"을 정하고, 통과한 것만 저장해요.",
           "<p><b>Pydantic</b>은 \"이 데이터는 이런 필드와 타입이어야 한다\"를 클래스로 적어 두면 검사해 주는 라이브러리예요. 함수 인자 타입을 이 클래스로 두면, FastAPI가 요청 본문(JSON)을 이 모양으로 검사한 뒤에만 함수를 불러요.</p>"
           + K.code_table([("class KnowledgeWrite(BaseModel)", "등록과 수정 때 받을 모양. <code>id</code>는 없어요. 서버가 정하니까요."),
                           ("Field(min_length=1, max_length=80)", "빈 글이나 너무 긴 글을 막아요."),
                           ("payload: KnowledgeWrite", "요청 본문을 이 모양으로 검사해서 객체로 받아요."),
                           ("payload.model_dump()", "객체를 다시 dict로. 저장하려고요."),
                           ('{"id": data["next_id"], **payload.model_dump()}', "서버가 정한 번호에 받은 필드를 펼쳐 붙여요."),
                           ('@app.post("/knowledge", status_code=201)', "성공하면 200 대신 201(새로 만듦)")])
           + K.try_("<code>title</code>을 빈 글로, <code>category</code>를 빼고, <code>title</code>을 숫자로 보내면?", "t4"), "s4"),
    K.step(5, "고치기 (PUT)", "번호로 항목을 찾아 받은 내용으로 통째로 바꿔요.",
           "<p>검사는 ④에서 만든 <code>KnowledgeWrite</code>를 그대로 다시 써요. 입력 규칙이 한 곳에만 있어서, 규칙을 바꿀 때 한 곳만 고치면 돼요.</p>"
           + K.code_table([("e.update(payload.model_dump())", "찾은 항목 dict의 값을 받은 값으로 덮어써요. <code>id</code>는 그대로예요.")]), "s5"),
    K.step(6, "지우기 (DELETE)", "번호가 다른 항목만 남겨서 저장해요. 지운 게 없으면 404예요.",
           K.code_table([("[e for e in ... if e[\"id\"] != knowledge_id]", "지울 번호만 빼고 새 목록"),
                         ("len(kept) == len(data[\"entries\"])", "길이가 그대로면 지운 게 없다는 뜻 → 404"),
                         ("status_code=204", "성공했지만 돌려줄 본문이 없다는 뜻. 함수가 아무것도 돌려주지 않아요.")]), "s6"),
]
practice = [
    ("두 요청이 동시에 저장한다", "둘 다 파일을 읽고, 각자 하나를 추가하고, 각자 저장하면 먼저 저장한 쪽이 사라져요. <code>threading.Lock</code>으로 \"읽기부터 저장까지\"를 한 번에 하나만 하게 막을 수 있지만, 서버가 여러 개면 소용없어요. 이게 DB를 쓰는 가장 큰 이유예요 (<a href=\"postgres.html\">PostgreSQL</a>)."),
    ("항목이 많아진다", "지금은 요청마다 파일 전체를 읽고 써요. 항목이 수만 개가 되면 느려져요. 이것도 DB가 필요한 이유예요."),
    ("주소 순서가 겹친다", "<code>/knowledge/search</code>와 <code>/knowledge/{knowledge_id}</code>가 둘 다 있으면 먼저 등록한 쪽이 이겨요. 고정 주소를 먼저 등록해요."),
    ("422 응답을 사용자에게 보여 준다", "<code>detail</code>의 <code>loc</code>(어디)와 <code>msg</code>(왜)를 그대로 보여 주면 무엇을 고칠지 알 수 있어요."),
    ("<code>--reload</code>", "코드가 바뀔 때마다 서버를 다시 띄우는 개발용 옵션이에요. 실제 운영에서는 빼요."),
]
libs = [
    ("①", "<code>FastAPI()</code>, <code>@app.get</code>", "앱 만들기, 주소에 함수 연결 (fastapi)"),
    ("②", "<code>Path.read_text()</code>, <code>json.loads()</code>, <code>json.dumps()</code>", "파일 저장소 (표준 라이브러리)"),
    ("③", "<code>HTTPException(status_code, detail)</code>", "실패 응답 보내기 (fastapi)"),
    ("④", "<code>BaseModel</code>, <code>Field()</code>, <code>model_dump()</code>, <code>@app.post(status_code=201)</code>", "입력 모양 검사 (pydantic), 등록"),
    ("⑤ ⑥", "<code>@app.put</code>, <code>@app.delete(status_code=204)</code>", "수정, 삭제"),
    ("확인", "<code>TestClient(app).get()</code>, <code>.post(json=...)</code>", "서버 없이 요청 보내 보기 (fastapi.testclient)"),
]
K.build(ROOT / "topics" / "backend" / "fastapi.html",
        title="FastAPI로 지식 API 만들기", h1="FastAPI로 지식 API 만들기", track="백엔드",
        crumb="백엔드 &gt; API 서버",
        one_line="<b>API 서버</b>는 다른 프로그램이 데이터를 읽고 쓰는 <b>하나의 정해진 입구</b>예요. FastAPI로 주소마다 함수를 연결하고, 들어오는 입력은 <b>저장하기 전에 검사</b>해요.",
        why=why, whole=whole, steps=steps, code_file="examples/api/main.py", practice=practice, libs=libs,
        libs_note="<code>read_data</code>, <code>write_data</code>와 각 주소의 함수는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"postgres.html\">PostgreSQL 입문</a>", "파일 대신 DB에 저장하기"),
                   ("<a href=\"fastapi-postgres.html\">FastAPI와 PostgreSQL 연동</a>", "이 API를 DB와 연결하기")],
        sources=["다루는 범위: Day17 (2026-11-10) FastAPI 지식 관리 API",
                 "예제: <code>examples/api/main.py</code>, <code>examples/api/data/knowledge.json</code> (기존 노트의 예제 항목 세 개)",
                 "결과 값: <code>tools/pages/api_server.py</code>가 단계별로 실제 실행해 만든 <code>topics/backend/fastapi-data.js</code>",
                 "공식 문서: <a href=\"https://fastapi.tiangolo.com/tutorial/\" target=\"_blank\">FastAPI Tutorial</a>, <a href=\"https://fastapi.tiangolo.com/tutorial/testing/\" target=\"_blank\">Testing</a>, <a href=\"https://docs.pydantic.dev/latest/concepts/fields/\" target=\"_blank\">Pydantic Field</a>, <a href=\"https://developer.mozilla.org/ko/docs/Web/HTTP/Status\" target=\"_blank\">HTTP 상태 코드</a>"])
print("ok")
