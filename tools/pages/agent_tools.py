"""백엔드 > Agent > AI Agent와 도구. examples/agent_tools/agent.py 하나가 ①~⑦로 자란다 (다른 쪽 코드에 기대지 않음).
도구 둘(규정 검색: e5-small, 예약 조회: SQLite 메모리 DB)은 실제로 실행하고, 모델 자리는 미리 정한 계획대로 도구를 부르는 가짜 함수다.
DB 서버나 키가 필요 없다.
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/agent_tools.py
"""
import html
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "agent_tools"
code = S.parse((EX / "agent.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, EX.resolve(), filename="agent.py")
V = {p: version(p) for p in ("langchain-core", "pydantic", "sentence-transformers")}
ENV = f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items()) + f", SQLite {ns['sqlite3'].sqlite_version}, intfloat/multilingual-e5-small"
E = html.escape
outs = {}


def js(x):
    return E(json.dumps(x, ensure_ascii=False))


# ---------------- ①
ss, sd = ns["search_schema"], ns["search_direct"]
S.claim(sd["passages"][0]["title"] == "예약 철회" and set(ss["properties"]) == {"query", "category"} and ss["required"] == ["query"], "① 1위 예약 철회, 입력 둘")
outs["s1"] = S.box("① search_rules가 모델에게 보이는 모양 (tool_call_schema)", [
    (f"title (도구 이름) \"{ss['title']}\"", 1), (f"description \"{ss['description']}\"", 2),
    (f"properties.query {json.dumps(ss['properties']['query'], ensure_ascii=False)}, 꼭 채워야 함", 3),
    (f"properties.category {json.dumps(ss['properties']['category'], ensure_ascii=False)}", 4),
    (f"직접 실행 search_rules.invoke({{\"query\": \"스터디룸 예약 철회 방법\"}}) → passages {[(p['id'], p['title']) for p in sd['passages']]}", 5, "newf")],
    ["함수 이름이 도구 이름이 돼요. 모델은 이 이름으로 불러요.",
     "docstring이 설명이 돼요. 모델은 이 설명을 읽고 언제 이 도구를 쓸지 골라요. 그래서 '무엇은 못 한다'까지 적었어요.",
     "타입 힌트가 입력 모양이 돼요. query는 글자 하나예요.",
     "Literal로 넷 중 하나만 받게 했어요(enum). 기본값이 있어서 안 채워도 돼요.",
     "모델 없이 도구만 직접 실행해 볼 수도 있어요. 결과는 글(JSON)로 돌려줘요. 모델은 결과를 글로 받거든요."])

# ---------------- ②
db, ls = ns["db_checks"], ns["lookup_schema"]
S.claim([v["status"] for v in db.values()] == ["ok", "not_found_or_forbidden", "ok", "not_found_or_forbidden"], "② 본인 것만 보임")
S.claim(list(ls["properties"]) == ["reservation_id"] and ls["properties"]["reservation_id"].get("minimum") == 1, "② member_id는 숨고 최소 1")
S.claim("readonly" in (ns["write_blocked"] or ""), "② 읽기 전용 연결에서 쓰기 막힘")
outs["t2"] = S.table("② read_reservation(예약 번호, 회원)", ["누가, 무엇을", "결과"],
                     [[E(k), js(v)] for k, v in db.items()] + [["읽기 전용 연결로 UPDATE", f"<b>{E(ns['write_blocked'])}</b>"]],
                     ["m-001은 자기 예약 101을 읽어요.",
                      "m-001이 m-002의 202를 물으면 '없거나 볼 수 없음'이에요. 없는 번호 999와 같은 답이에요. '남의 예약이라 못 보여 준다'고 하면 그 번호가 있다는 걸 알려 주게 되거든요.",
                      "WHERE에 member_id를 넣었기 때문에 이렇게 돼요. 프로그램 쪽에서 결과를 골라내는 게 아니라 DB에서부터 본인 것만 읽어요.",
                      "이 연결은 읽기 전용(PRAGMA query_only)이라 바꾸는 SQL이 실패해요. 실수로든 모델의 요청으로든 예약이 바뀌지 않아요."])
outs["s2"] = S.box("② lookup_reservation이 모델에게 보이는 모양", [json.dumps(ls, ensure_ascii=False)],
                   ["함수에는 member_id 인수가 있지만 InjectedToolArg로 표시해서 모델에게는 안 보여요. 모델은 예약 번호만 채우고, 누구인지는 실행할 때 앱이 로그인 정보로 넣어요(③). 모델이 member_id를 정하게 하면 \"m-002의 예약을 보여 줘\" 한 줄로 남의 정보를 볼 수 있어요.",
                    "Field(ge=1, strict=True)라서 0 이하나 글자 \"101\"은 실행 전에 막혀요(⑤)."])

# ---------------- ③
fst, tm, fin = ns["first"], ns["tool_msgs"], ns["final"]
S.claim([c["name"] for c in fst["tool_calls"]] == ["search_rules", "lookup_reservation"] and "[출처 14]" in fin["content"] and "예약됨" in fin["content"], "③ 한 바퀴")
outs["s3"] = S.box(f"③ \"{ns['Q_BOTH']}\" 한 바퀴", [
    ("1) 모델 → 앱: tool_calls " + json.dumps([{"name": c["name"], "args": c["args"]} for c in fst["tool_calls"]], ensure_ascii=False), 1, "newf"),
    *[(f"2) 앱 → 모델: role \"tool\", name \"{m['name']}\", content {m['content'][:70]}…", 2) for m in tm],
    (f"3) 모델 → 사용자: \"{fin['content']}\"", 3, "newf")],
    ["모델은 직접 실행하지 않고 '이 도구를 이 인수로 불러 달라'고 요청만 해요. 질문 하나에 도구 둘을 한꺼번에 요청했어요. 요청마다 id가 붙어요.",
     "앱이 허용된 도구인지, 입력이 맞는지 보고 실행해요. 결과는 어느 요청의 결과인지(tool_call_id)와 함께 role \"tool\" 메시지로 돌려줘요. 예약 조회에는 앱이 로그인한 회원(m-001)을 넣었어요.",
     "모델은 결과를 읽고 답을 써요. 규정 쪽 답에는 검색 결과의 조각 번호를 출처로 붙였어요(⑥에서 확인해요)."])

# ---------------- ④
tr = ns["traces"]
tools_of = {q: [e["tool"] for e in t[2]] for q, t in tr.items()}
S.claim(list(tools_of.values()) == [["search_rules"], ["lookup_reservation"], ["search_rules", "lookup_reservation"], []], "④ 질문마다 고른 도구")
outs["t4"] = S.table("④ run_agent(질문) (회원 m-001)", ["질문", "메시지 흐름", "부른 도구", "답"],
                     [[E(q), " → ".join(m["role"] + ("(도구 요청)" if m.get("tool_calls") else "") for m in t[1]), ", ".join(tools_of[q]) or "없음", E(t[0])] for q, t in tr.items()],
                     ["질문에 따라 도구를 하나, 둘, 또는 하나도 안 불러요. 순서가 정해진 RAG 서비스와 다른 점이 이거예요. 다음에 무엇을 할지 모델이 골라요.",
                      "도구가 필요 없는 질문은 user → assistant로 바로 끝나요.",
                      "이 예제의 가짜 모델은 질문마다 미리 정한 계획(PLANS)대로 부르고, 도구 결과의 첫 문장을 옮겨 답해요. 그래서 이 표의 답 문장은 모델이 똑똑한지가 아니라 요청, 실행, 결과가 어떻게 흐르는지를 보여 줘요."])

# ---------------- ⑤
g = ns["guards"]
S.claim([v["outcome"] for v in g.values()] == ["답함", "답함", "답함", "중단", "답함"], "⑤ 반복만 중단")
S.claim([v["tool_results"][0].get("error") or v["tool_results"][0].get("status") for v in g.values() if v["outcome"] == "답함"]
        == ["invalid_args", "not_found_or_forbidden", "unknown_tool", "db_unavailable"], "⑤ 실패가 서로 다른 이름")
WHERE = {"잘못된 입력 (-1)": "실행 전 입력 검사", "남의 예약 (202)": "DB 조회 조건 (WHERE member_id)", "없는 도구": "허용 목록 (TOOLS)", "같은 도구를 계속": "반복 상한 (max_steps=4)", "DB 장애": "DB 연결"}
outs["t5"] = S.table("⑤ 가짜 모델이 이상한 요청을 하면 (회원 m-001)", ["상황", "막은 곳", "도구 결과", "끝"],
                     [[E(k), WHERE[k], js(v["tool_results"][0]) if v.get("tool_results") else "-", f"<b>{v['outcome']}</b>: " + E(v.get("answer") or v.get("error"))] for k, v in g.items()],
                     ["잘못된 입력과 없는 도구는 실행하지 않고, 무엇이 틀렸는지를 도구 결과로 모델에게 돌려줘요. 실제 모델은 이걸 보고 다시 시도할 수 있어요.",
                      "남의 예약은 오류가 아니라 정상 결과(없거나 볼 수 없음)예요.",
                      "같은 도구를 계속 부르면 4번에서 멈추고 오류(tool_call_limit)를 올려요. 상한이 없으면 비용과 시간이 끝없이 들어요.",
                      "DB 장애는 '예약 없음'이 아니라 db_unavailable이에요. 고장을 '없음'으로 바꾸면 사용자에게 틀린 답을 주게 돼요.",
                      "실패마다 이름이 달라야 다시 시도할지, 멈출지, 사용자에게 뭐라고 할지 정할 수 있어요."])

# ---------------- ⑥
ck, fc = ns["checked"], ns["fake_cite"]
q0 = ns["ASKS"][0]
S.claim(all(v["ok"] for v in ck.values()) and not fc["ok"], "⑥ 정상은 통과, 지어낸 출처는 막힘")
outs["s6"] = S.box("⑥ check_answer(답, 이번 실행 기록)", [
    (f"\"{q0}\" → 답의 출처 {ck[q0]['cited']}, 검색이 돌려준 조각 {ck[q0]['returned']} → ok {ck[q0]['ok']}", 1),
    (f"\"1시간 전까지 철회해요. [출처 99]\" → 출처 {fc['cited']}, 돌려준 조각 {fc['returned']} → ok {fc['ok']}, {', '.join(fc['problems'])}", 2, "bad")],
    ["답에 적힌 출처 번호가 이번 실행에서 search_rules가 실제로 돌려준 조각 안에 있어요.",
     "모델이 출처 번호를 지어내면(99번 조각은 이번에 돌려준 적이 없어요) 걸려요. 모델의 마지막 문장은 그대로 믿지 않고, 앱이 따로 남긴 실행 기록(ledger)과 대조한 뒤 내보내요."])

# ---------------- ⑦
rg, ac = ns["regression"], ns["after_change"]
S.claim(all(all(r[k] for k in ("tools_ok", "answer_ok", "check_ok")) for r in rg), "⑦ 처음엔 모두 통과")
bad = [r["name"] for r in ac if not all(r[k] for k in ("tools_ok", "answer_ok", "check_ok"))]
S.claim(bad == ["둘 다"], "⑦ 바꾼 뒤 '둘 다'만 실패")


def mark(v):
    return "통과" if v else "<b>실패</b>"


outs["t7"] = S.table("⑦ 회귀 시나리오: 처음 → 모델을 바꾼 뒤", ["시나리오", "질문", "회원", "도구", "답", "출처 검사", "바꾼 뒤"],
                     [[E(n), E(q), m, mark(r["tools_ok"]), mark(r["answer_ok"]), mark(r["check_ok"]),
                       "통과" if all(a[k] for k in ("tools_ok", "answer_ok", "check_ok")) else f"<b>실패</b> (도구 {mark(a['tools_ok'])}, 답 {mark(a['answer_ok'])})"]
                      for (n, q, m, _, _), r, a in zip(ns["SCENARIOS"], rg, ac)],
                     ["처음에는 여섯 시나리오가 모두 통과해요. 남의 예약(m-001이 202)은 안 보이고, 주인(m-002)에게는 보여요.",
                      "모델을 바꿨더니 '둘 다' 질문에서 예약 조회를 빼먹는다고 쳐 보면(PLANS를 고침), 그 시나리오만 실패로 바뀌어요. 답에 규정 설명은 있어서 얼핏 그럴듯한데, '불려야 할 도구'를 함께 확인하니 바로 걸려요.",
                      "모델, 도구 설명, 규칙(system prompt)을 바꿀 때마다 같은 시나리오를 돌려요. 실제 모델은 같은 질문에도 고르는 도구가 흔들릴 수 있어서 여러 번 돌려 봐요."])

flow = [[1, "검색 도구", "이름, 설명, 입력 모양", "docstring이 설명"], [2, "DB 도구", "읽기 전용, 본인 것만", "member_id는 숨김"], [3, "한 바퀴", "요청 → 실행 → 답", "모델은 요청만"],
        [4, "반복", "답이 나올 때까지", "도구 0~2개"], [5, "안전장치", "입력, 남의 것, 없는 도구, 반복, 장애", "실패마다 다른 이름"],
        [6, "답 확인", "출처를 실행 기록과", "지어낸 출처 막기"], [7, "회귀", "같은 시나리오로", f"바꾼 뒤 실패 {len(bad)}"]]
S.save(ROOT / "topics" / "backend" / "agent-tools-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

rows_db = ns["admin"].execute("SELECT * FROM reservations ORDER BY reservation_id").fetchall()
why = f"""<p>RAG 서비스는 순서가 정해져 있어요. 질문이 오면 찾고, 고르고, 답해요. 그런데 \"스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘\" 같은 질문은 문서 검색만으로는 답할 수 없어요. 규정은 안내문에 있지만, 내 예약의 지금 상태는 DB에 있으니까요. 질문에 따라 <b>무엇을 할지 골라야</b> 해요.</p>
<p><b>AI Agent</b>는 모델이 질문과 쓸 수 있는 도구 목록을 보고 다음 행동(도구 호출 또는 답)을 고르고, 앱이 그 행동을 실제로 해서 결과를 다시 모델에게 주는 구조예요. 역할이 셋으로 나뉘어요.</p>
{K.table(["누가", "하는 일", "하지 않는 일"], [["모델", "도구 이름과 인수를 <b>요청</b>, 결과를 읽고 답 쓰기", "DB 접속, 실제 실행, 권한 판단"], ["도구 (Tool)", "검색, 조회 같은 실제 일을 하는 함수. 이름, 설명, 입력 모양, 결과 모양을 가져요", "다른 도구 부르기"],
                                    ["하네스 (모델을 둘러싼 앱 코드)", "허용된 도구인지와 입력 확인, 실행, 결과 전달, 반복 상한, 오류 처리", "무엇을 부를지 정하기"]])}
<p><a href="https://docs.langchain.com/oss/python/langchain/agents" target="_blank">LangChain 문서</a>도 Agent를 \"도구를 부르는 모델을 일이 끝날 때까지 반복해서 돌리는 것\"으로 정의하고, 모델 바깥의 규칙, 도구, 반복과 오류 처리를 묶어 <b>하네스(harness)</b>라고 불러요. 이 쪽 코드에서는 하네스를 \"앱\"이라고도 써요.</p>
<p>예제는 가상의 '가람 도서관'이에요. 도구는 둘이에요. 안내문 조각 26개에서 찾는 <code>search_rules</code>와, 회원의 스터디룸 예약을 읽는 <code>lookup_reservation</code>이에요. 예약 표에는 가상의 회원 둘(m-001, m-002)의 예약 {len(rows_db)}개가 있어요.</p>
{K.table(["예약 번호", "회원", "방", "날짜", "시작", "상태"], [[r[0], r[1], r[2], r[3], f"{r[4]}시", r[5]] for r in rows_db])}
<p>도구는 실제로 실행해요(검색은 e5 모델, 예약은 SQLite DB). 모델 자리만 미리 정한 계획대로 도구를 부르는 가짜 함수예요. 그래서 이 쪽의 결과는 \"모델이 똑똑한가\"가 아니라 <b>요청, 실행, 권한, 오류가 어떻게 흐르나</b>를 보여 줘요.</p>
"""
whole = """<p>도구 둘을 만들고(①②), 한 바퀴를 손으로 돌려 본 뒤(③) 반복으로 묶고(④), 이상한 요청을 막고(⑤), 답을 확인하고(⑥), 시나리오로 시험해요(⑦). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "규정 검색을 도구로: 이름, 설명, 입력 모양, 결과", "함수 하나에 @tool을 붙이면 도구가 돼요. 모델은 이 도구의 이름, 설명, 입력 모양만 보고 부를지 정해요.",
           "<p>도구는 그냥 함수가 아니라 <b>약속</b>이에요. 모델에게는 이름, 설명, 입력 모양이 보이고, 앱에게는 결과 모양이 약속이에요. LangChain의 <code>@tool</code>은 함수 이름, docstring, 타입 힌트에서 이 약속을 만들어 줘요. 그래서 타입 힌트는 꼭 써야 해요. 입력 모양을 거기서 만들거든요(<a href=\"https://docs.langchain.com/oss/python/langchain/tools\" target=\"_blank\">LangChain Tools</a>).</p>"
           "<p>검색 자체는 e5로 질문과 가까운 조각을 찾는 거예요(자세한 건 <a href=\"rag-rerank.html\">RAG 검색 고치기</a> ①). 이 도구는 답을 쓰지 않고 <b>근거 조각</b>만 돌려줘요. 답은 모델이 써요.</p>"
           + K.code_table([("@tool", "함수를 LangChain 도구로. 이름은 함수 이름, 설명은 docstring, 입력 모양은 타입 힌트에서"), ("Literal[\"이용\", \"공지\", \"규정\", \"시설\"]", "넷 중 하나만 받기"),
                           ("search_rules.tool_call_schema.model_json_schema()", "모델에게 보이는 입력 모양 (JSON Schema)"), ("search_rules.invoke({\"query\": ...})", "모델 없이 도구만 직접 실행")]), "s1"),
    K.step(2, "예약 조회를 도구로: 읽기 전용, 본인 것만", "예약 상태처럼 계속 바뀌는 정보는 DB에서 읽어요. 누구의 것을 읽을지는 모델이 아니라 앱이 정해요.",
           "<p>DB 도구에서 지킬 것은 셋이에요.</p>"
           "<ol><li>모델에게 SQL을 쓰게 하지 않아요. 개발자가 쓴 SQL에 <b>값만</b> <code>?</code> 자리로 넘겨요. 값을 문장에 이어 붙이면 \"101 OR 1=1\" 같은 입력으로 다른 행까지 읽힐 수 있어요.</li>"
           "<li>누구의 예약인지(member_id)는 로그인한 사람으로 앱이 정하고, WHERE에 꼭 넣어요.</li><li>읽기만 하는 연결로 열어요.</li></ol>"
           + K.out("t2")
           + "<p>예제는 설치가 필요 없는 SQLite로 해요. PostgreSQL이라면 SELECT만 허락된 계정으로 접속하고 <code>conn.read_only = True</code>로 같은 일을 해요(<a href=\"postgres.html\">PostgreSQL</a>).</p>"
           + K.code_table([("sqlite3.connect(\"file:garam?mode=memory&cache=shared\", uri=True)", "메모리 안의 DB. 같은 이름으로 연 연결끼리 공유해요"),
                           ("conn.execute(\"PRAGMA query_only = ON\")", "이 연결을 읽기 전용으로"), ("conn.execute(SQL, (reservation_id, member_id))", "값을 ? 자리로 넘기기"),
                           ("Annotated[str, InjectedToolArg]", "모델에게 안 보이고, 실행할 때 앱이 넣는 인수"), ("Field(ge=1, strict=True)", "1 이상의 정수만 (글자 \"101\"도 거절)")]), "s2"),
    K.step(3, "도구 호출 한 바퀴: 모델이 요청하고, 앱이 실행하고, 결과로 답해요", "모델이 도구 호출을 요청하고, 앱이 실행해 결과를 붙여 다시 물으면, 모델이 답을 써요. 이 한 바퀴가 Agent의 기본이에요.",
           "<p>메시지는 역할(role)이 셋이에요. tool calling을 지원하는 모델 API가 공통으로 쓰는 모양이에요.</p>"
           + K.table(["role", "누가 만드나", "담는 것"], [["user", "사용자", "질문"], ["assistant", "모델", "답 글, 또는 <code>tool_calls</code> (도구 이름, 인수, id)"],
                                                 ["tool", "앱", "도구 결과와 <code>tool_call_id</code> (어느 요청의 결과인지)"]])
           + "<p>앱의 <code>execute</code>는 세 가지를 확인한 뒤 실행해요. 허용 목록에 있는 도구인지, 입력이 약속한 모양인지, 그리고 예약 조회라면 회원을 로그인 정보로 넣어요. 모델이 무엇을 요청하든 실제로 일어나는 일은 앱이 정해요.</p>"
           + K.code_table([("{\"role\": \"assistant\", \"tool_calls\": [{\"id\", \"name\", \"args\"}]}", "모델의 도구 호출 요청"), ("TOOLS[call[\"name\"]].invoke(args)", "허용된 도구만 실행"),
                           ("{\"role\": \"tool\", \"tool_call_id\": ..., \"content\": ...}", "결과를 요청 id와 짝지어 돌려주기")]), "s3"),
    K.step(4, "반복으로 묶기: 답이 나올 때까지", "모델 → 도구 → 모델을, 모델이 도구 호출 없이 답할 때까지 되풀이해요. 몇 번까지 돌지는 앱이 정해요.",
           "<p>③의 한 바퀴를 <code>run_agent</code>로 묶어요. 모델의 응답에 도구 요청이 있으면 실행해서 결과를 붙이고 다시 묻고, 요청이 없으면 그게 답이에요. 도구 결과를 보고 또 다른 도구가 필요해지면 여러 바퀴를 돌아요. 끝없이 돌지 않게 <code>max_steps</code>로 상한을 둬요.</p>"
           "<p>도구를 부를 때마다 앱은 따로 기록(ledger)을 남겨요. 모델이 쓴 메시지와 별개로, 앱이 실제로 무엇을 실행했고 무엇이 나왔는지예요(⑥에서 써요).</p>"
           + K.code_table([("for _ in range(max_steps)", "최대 max_steps번까지만"), ("if not reply.get(\"tool_calls\")", "도구 요청이 없으면 답이 나온 것"), ("ledger.append({\"tool\": ..., \"result\": ...})", "앱이 실제로 실행한 기록")]), "t4"),
    K.step(5, "안전장치: 잘못된 입력, 남의 것, 없는 도구, 반복, DB 장애", "모델이 이상한 요청을 해도 앱에서 걸러지고, 끝없이 돌지 않고, 고장은 고장으로 올라오는지 봐요.",
           "<p>모델은 틀릴 수 있어요. 인수를 잘못 채우거나, 없는 도구 이름을 지어내거나, 같은 도구를 계속 부르기도 해요. 그래서 막는 곳을 모델이 아니라 앱(③의 <code>execute</code>, ④의 상한, ②의 DB 조건)에 둬요. 여기서는 가짜 모델의 다음 응답을 미리 정해(<code>NEXT</code>) 이상한 요청을 일부러 만들어요.</p>"
           "<p>도구 실패는 하나로 뭉치면 안 돼요. 어디서 실패했는지에 따라 다시 시도할지, 멈출지, 사용자에게 뭐라고 할지가 달라요.</p>"
           + K.code_table([("except ValidationError: return {\"error\": \"invalid_args\"}", "입력 모양이 틀리면 실행 전에 막기"), ("if call[\"name\"] not in TOOLS", "허용 목록에 없는 도구는 실행하지 않기"),
                           ("raise RuntimeError(\"tool_call_limit\")", "상한을 넘으면 멈추기"), ("except sqlite3.OperationalError: raise RuntimeError(\"db_unavailable\")", "DB 장애는 '없음'이 아니라 오류로")]), "t5"),
    K.step(6, "답 확인하기: 출처와 도구 실패", "Agent의 마지막 답을 그대로 믿지 않고, 답에 적힌 출처가 이번 실행에서 실제로 돌려받은 조각인지, 도구가 실패하지 않았는지 대조해요.",
           "<p>마지막 답은 모델이 쓴 글이에요. 출처 번호도 모델이 적은 거라 틀리거나 지어낼 수 있어요. 앱은 ④에서 따로 남긴 실행 기록을 갖고 있으니, 답을 내보내기 전에 둘을 대조해요. 확인을 못 넘긴 답은 내보내지 않아요.</p>"
           + K.code_table([("re.findall(r\"\\[출처 (\\d+)\\]\", answer)", "답에 적힌 [출처 n]의 번호들"), ("cited <= returned", "적힌 번호가 모두 이번에 돌려받은 조각 안에 있나 (부분집합)")]), "s6"),
    K.step(7, "회귀 시나리오: 바꿀 때마다 같은 질문으로", "정상, 복합, 남의 것, 도구가 필요 없는 질문을 시나리오로 정해 두고, 부른 도구와 답과 출처를 함께 확인해요.",
           "<p>모델이나 도구 설명, 규칙을 바꾸면 예전에 되던 질문이 안 될 수 있어요. 그래서 시나리오마다 <b>불려야 할 도구</b>와 <b>답에 있어야 할 말</b>을 정해 두고, 바꿀 때마다 다시 돌려요(회귀 테스트).</p>"
           + K.code_table([("SCENARIOS = [(이름, 질문, 회원, 불려야 할 도구, 답에 있어야 할 말), ...]", "시나리오 표"), ("run_regression()", "시나리오마다 도구, 답, 출처 검사 통과 여부"),
                           ("PLANS[Q_BOTH] = PLANS[Q_BOTH][:1]", "모델이 바뀌어 도구 하나를 빼먹는 상황 흉내")]), "t7"),
]
practice = [("도구 설명이 곧 길잡이", "모델은 이름, 설명, 입력 모양만 보고 도구를 고르니까, 설명에 '무엇은 못 한다'까지 적어요. 비슷한 도구가 둘이면 설명으로 경계를 분명히 해요."),
            ("신원은 서버에서", "user_id, member_id 같은 값은 모델이 채우는 인수로 받지 않아요. LangChain에서는 InjectedToolArg나 create_agent의 실행 정보(ToolRuntime)로 앱이 넣어요."),
            ("DB 도구는 읽기부터", "처음엔 SELECT만 허락된 계정, 정해 둔 SQL, 값 바인딩으로 시작해요. 바꾸는 도구(예약 취소 같은)는 사용자에게 한 번 더 확인받는 단계를 따로 둬요."),
            ("반복 상한은 필수", "입력 오류나 없는 도구는 모델에게 돌아가 다시 시도하게 되니까, 실행 전체와 도구별 호출 수에 상한을 걸어요. 시간과 비용 상한도 따로 둬요."),
            ("정보 부족과 고장을 나누기", "'자료에 없음', '남의 것이라 안 보임', 'DB 장애', '로그인 안 됨'은 다른 상태예요. 고장을 '없음'으로 바꾸면 사용자에게 틀린 답을 주게 돼요."),
            ("LangChain create_agent", "④의 반복(모델 호출, 도구 실행, 결과 전달, 끝날 때까지)은 <code>create_agent(model, tools, system_prompt)</code>가 대신 돌려줘요. 반복 상한은 middleware(<code>ToolCallLimitMiddleware</code>)로 붙여요. 다만 권한 검사와 답 확인은 해 주지 않아서 여전히 도구 안과 앱(⑤⑥)에서 해요."),
            ("API로 열 때", "로그인으로 회원을 정하고(요청 본문에서 받지 않아요), 확인을 못 넘긴 답이나 도구 장애는 정상 답(200)과 다른 응답(예: 502)으로 돌려줘요."),
            ("기록은 최소한으로", "실행 기록에는 단계와 시간, 보안 기록에는 누가 어떤 도구를 썼는지만 남기고, 토큰, 키, 개인정보, 문서 원문은 남기지 않아요."),
            ("다음 주제", "여러 단계와 상태를 그래프로 다루는 LangGraph, 문서 속 지시로 모델을 조종하려는 Prompt Injection 같은 보안, 도구를 표준 방식으로 붙이는 MCP로 이어져요.")]
libs = [("①", "<code>@tool</code>, <code>tool_call_schema</code>, <code>invoke</code>", "도구 만들기와 실행 (langchain-core)"), ("①", "<code>SentenceTransformer</code>, <code>encode</code>", "규정 검색 (sentence-transformers)"),
        ("②", "<code>sqlite3.connect</code>, <code>PRAGMA query_only</code>, <code>execute(SQL, 값)</code>", "읽기 전용 DB 조회 (표준 라이브러리)"),
        ("②", "<code>InjectedToolArg</code>, <code>Field</code>, <code>Annotated</code>", "숨긴 인수와 입력 범위 (langchain-core, Pydantic)"),
        ("⑤", "<code>ValidationError</code>", "입력 모양 오류 (Pydantic)"), ("⑥", "<code>re.findall</code>", "답의 출처 번호 꺼내기")]
K.build(ROOT / "topics" / "backend" / "agent-tools.html", title="AI Agent와 도구", h1="AI Agent와 도구: 모델은 고르고, 앱이 확인하고 실행한다", track="백엔드", crumb="백엔드 &gt; Agent",
        one_line="<b>AI Agent</b>는 도구를 부르는 모델을 <b>일이 끝날 때까지 반복해서</b> 돌리는 구조예요. 모델은 무엇을 할지 <b>요청만</b> 하고, 모델을 둘러싼 코드(<b>하네스</b>)가 허용된 도구인지, 입력이 맞는지, 누구의 정보를 읽을지, 몇 번까지 돌지, 답의 출처가 맞는지를 정하고 확인해요. Agent = 모델 + 하네스예요.",
        why=why, whole=whole, steps=steps, code_file="examples/agent_tools/agent.py", practice=practice, libs=libs,
        libs_note="<code>search_rules</code>, <code>read_reservation</code>, <code>lookup_reservation</code>, <code>say</code>, <code>fake_model</code>, <code>execute</code>, <code>run_agent</code>, <code>scripted</code>, <code>ask_tool</code>, <code>check_answer</code>, <code>run_regression</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rag-service.html\">RAG 서비스</a>", "순서가 정해진 흐름 (Agent와 비교)"), ("<a href=\"rag-rerank.html\">RAG 검색 고치기</a>", "규정 검색 도구 안의 검색"),
                   ("<a href=\"rag-eval.html\">RAG 평가</a>", "검색 도구를 붙이기 전에 재기"), ("<a href=\"postgres.html\">PostgreSQL</a>", "실제 DB에서 읽기 전용 계정과 값 바인딩"),
                   ("<a href=\"llm-client.html\">LLM API 클라이언트</a>", "실제 모델 API를 부를 때")],
        sources=["다루는 범위: Day29 (2026-11-26) AI Agent와 Tool Calling. Agent와 정해진 흐름의 차이, 모델과 도구와 실행 환경의 역할, @tool과 도구 설명, 입력 모양, 메시지와 tool_calls, Agent 루프, 읽기 전용 DB 도구와 본인 것만 보기, 신원은 서버에서, 반복 상한, 실패 나누기, 답의 출처 확인, 회귀 시나리오, create_agent와 middleware. 실습 프로젝트의 PostgreSQL 연결, FastAPI 연결, RAG 서비스 재사용은 따라 하지 않고 이 쪽 하나로 실행되게 다시 만들었어요",
                 "예제: <code>examples/agent_tools/agent.py</code> (이 파일 하나로 실행돼요). 조각은 <code>examples/ingest/out/chunks.json</code>과 <code>examples/rag/data/more_chunks.json</code>, 예약 표는 파일 안의 가상 자료",
                 f"결과 값: <code>tools/pages/agent_tools.py</code>가 단계별로 실제 실행한 결과 ({ENV}). 모델의 도구 선택과 답 문장은 가짜 함수의 규칙이고, 도구 실행, DB, 검사 결과는 실행 값이에요",
                 "공식 문서: <a href=\"https://docs.langchain.com/oss/python/langchain/tools\" target=\"_blank\">LangChain Tools</a>, <a href=\"https://docs.langchain.com/oss/python/langchain/agents\" target=\"_blank\">LangChain Agents</a>, <a href=\"https://docs.python.org/3/library/sqlite3.html\" target=\"_blank\">sqlite3</a>, <a href=\"https://www.sqlite.org/pragma.html#pragma_query_only\" target=\"_blank\">PRAGMA query_only</a>"])
print("ok")
