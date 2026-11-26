# AI Agent와 도구: 모델이 도구를 고르고, 앱이 확인하고 실행해요. 이 파일 하나로 실행돼요.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 도구 둘: 규정 검색(가람 도서관 조각 26개, e5-small) + 스터디룸 예약 조회(SQLite 메모리 DB, 가상의 회원 둘).
# 모델 자리는 미리 정한 계획대로 도구를 부르고, 도구 결과를 옮겨 답하는 가짜 함수(fake_model)예요. 키도 비용도 없어요.
# 메시지는 tool calling API가 쓰는 모양(role, content, tool_calls)의 dict예요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
import re                                                        #@6
import sqlite3                                                   #@2
from pathlib import Path                                         #@1
from typing import Annotated, Literal                            #@1
import numpy as np                                               #@1
from langchain_core.tools import InjectedToolArg, tool           #@1
from pydantic import Field, ValidationError                      #@2
from sentence_transformers import SentenceTransformer            #@1

HERE = Path(__file__).resolve().parent                           #@1

# ① 규정 검색을 도구로: 이름, 설명, 입력 모양, 결과
CATEGORY = {"guide.md": "이용", "hours.html": "이용", "room.md": "이용", "holds.md": "이용", "programs.md": "이용", "devices.md": "이용",
            "notice.txt": "공지", "old_notice.txt": "공지", "notice_2025.txt": "공지", "notice_2026.txt": "공지",
            "rules.pdf": "규정", "scan.pdf": "시설"}
raw = (json.loads((HERE.parent / "ingest" / "out" / "chunks.json").read_text(encoding="utf-8"))
       + json.loads((HERE.parent / "rag" / "data" / "more_chunks.json").read_text(encoding="utf-8")))
CHUNKS = [{"id": n, "title": c["heading"] or c["body"].split("\n")[0], "body": c["body"], "category": CATEGORY[c["file"]],
           "current": c.get("current", True)} for n, c in enumerate(raw, start=1)]
encoder = SentenceTransformer("intfloat/multilingual-e5-small")
vectors = encoder.encode([f"passage: {c['title']}\n{c['body']}" for c in CHUNKS], normalize_embeddings=True)

@tool
def search_rules(query: str, category: Literal["이용", "공지", "규정", "시설"] = "이용") -> str:
    """도서관 이용 규정과 안내문에서 질문과 가까운 조각 2개를 찾아요. 회원 개인의 예약 상태는 찾지 못해요."""
    q = encoder.encode([f"query: {query}"], normalize_embeddings=True)[0]
    order = [i for i in np.argsort(-(vectors @ q)) if CHUNKS[i]["category"] == category and CHUNKS[i]["current"]][:2]
    return json.dumps({"status": "ok", "passages": [{"id": CHUNKS[i]["id"], "title": CHUNKS[i]["title"], "text": CHUNKS[i]["body"]} for i in order]},
                      ensure_ascii=False)

search_schema = search_rules.tool_call_schema.model_json_schema()   # 모델에게 보이는 이름, 설명, 입력 모양
search_direct = json.loads(search_rules.invoke({"query": "스터디룸 예약 철회 방법"}))

# ② 예약 조회를 도구로: 읽기 전용, 본인 것만
DB_URI = "file:garam?mode=memory&cache=shared"                   # 예시 DB (메모리 안). 실제로는 PostgreSQL 같은 DB 서버
admin = sqlite3.connect(DB_URI, uri=True)                        # 관리자 쪽 연결: 표를 만들고 채워요 (이 연결이 열려 있는 동안 DB가 남아요)
admin.execute("CREATE TABLE reservations (reservation_id INTEGER PRIMARY KEY, member_id TEXT, room TEXT, use_date TEXT, start_hour INTEGER, status TEXT)")
admin.executemany("INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)", [
    (101, "m-001", "스터디룸 A", "2026-11-28", 14, "예약됨"), (102, "m-001", "스터디룸 B", "2026-11-21", 10, "철회함"),
    (202, "m-002", "스터디룸 A", "2026-11-28", 16, "예약됨")])  # 가상의 회원 둘
admin.commit()

def read_reservation(reservation_id, member_id):                 # 로그인한 회원 본인의 예약만 읽어요
    try:
        conn = sqlite3.connect(DB_URI, uri=True)
        conn.execute("PRAGMA query_only = ON")                   # 이 연결로는 바꾸는 SQL이 실패해요
        row = conn.execute("SELECT reservation_id, room, use_date, start_hour, status FROM reservations "
                           "WHERE reservation_id = ? AND member_id = ?", (reservation_id, member_id)).fetchone()   # 값은 ? 자리로만
        conn.close()
    except sqlite3.OperationalError as exc:
        raise RuntimeError("db_unavailable") from exc            # DB 장애는 '없음'이 아니라 오류
    if row is None:                                              # 없는 번호와 남의 예약을 구별해 알려 주지 않아요
        return {"status": "not_found_or_forbidden", "reservation_id": reservation_id}
    return {"status": "ok", "reservation_id": row[0], "room": row[1], "use_date": row[2], "start_hour": row[3], "reservation_status": row[4]}

@tool
def lookup_reservation(reservation_id: Annotated[int, Field(ge=1, strict=True, description="조회할 스터디룸 예약 번호 (양의 정수)")],
                       member_id: Annotated[str, InjectedToolArg]) -> str:
    """로그인한 회원 본인의 스터디룸 예약 상태를 예약 번호로 읽어요. 이용 규정은 찾지 못해요."""
    return json.dumps(read_reservation(reservation_id, member_id), ensure_ascii=False)

lookup_schema = lookup_reservation.tool_call_schema.model_json_schema()   # member_id는 모델에게 안 보여요
db_checks = {name: read_reservation(rid, mid) for name, rid, mid in (("m-001의 101", 101, "m-001"), ("m-001의 202 (남의 예약)", 202, "m-001"),
                                                                     ("m-002의 202", 202, "m-002"), ("m-001의 999 (없는 번호)", 999, "m-001"))}
reader = sqlite3.connect(DB_URI, uri=True)
reader.execute("PRAGMA query_only = ON")
try:
    reader.execute("UPDATE reservations SET status = '철회함' WHERE reservation_id = 101")
    write_blocked = None
except sqlite3.OperationalError as exc:
    write_blocked = str(exc)                                     # 읽기 전용 연결에서 바꾸려 하면

# ③ 도구 호출 한 바퀴: 모델이 요청하고, 앱이 실행하고, 결과로 답해요
TOOLS = {"search_rules": search_rules, "lookup_reservation": lookup_reservation}   # 앱이 허용한 도구
PLANS = {  # 가짜 모델이 질문마다 요청할 도구 호출 (실제 모델은 도구 설명을 읽고 스스로 골라요)
    "스터디룸 예약을 철회하려면 어떻게 해?": [("search_rules", {"query": "스터디룸 예약 철회 방법"})],
    "내 101번 예약 어떻게 됐어?": [("lookup_reservation", {"reservation_id": 101})],
    "스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘": [("search_rules", {"query": "스터디룸 예약 철회 방법"}), ("lookup_reservation", {"reservation_id": 101})],
    "202번 예약 상태 알려줘": [("lookup_reservation", {"reservation_id": 202})],
}
NEXT = []                                                        # ⑤에서 가짜 모델의 다음 응답을 미리 정할 때 써요

def say(result):                                                 # 가짜 모델이 도구 결과 하나를 답 문장으로 옮기는 방법
    if "passages" in result:
        top = result["passages"][0]
        return f"{top['text'].split('. ')[0].rstrip('.')}. [출처 {top['id']}]"
    if result.get("status") == "ok":
        return f"{result['reservation_id']}번 예약은 {result['room']} {result['use_date']} {result['start_hour']}시, 상태는 {result['reservation_status']}이에요."
    if result.get("status") == "not_found_or_forbidden":
        return f"{result['reservation_id']}번 예약은 찾을 수 없어요."
    return f"조회하지 못했어요({result.get('error')})."

def fake_model(messages):                                        # 도구 결과가 아직 없으면 호출을 요청하고, 있으면 그걸로 답을 써요
    if NEXT:
        return NEXT.pop(0)
    results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
    if results:
        return {"role": "assistant", "content": " ".join(say(r) for r in results)}
    calls = PLANS.get(messages[0]["content"], [])
    if not calls:
        return {"role": "assistant", "content": "도서관 규정이나 내 예약을 물어봐 주세요."}
    return {"role": "assistant", "tool_calls": [{"id": f"call_{n}", "name": name, "args": args} for n, (name, args) in enumerate(calls)]}

def execute(call, member_id):                                    # 앱이 확인하고 실행해요. 모델은 직접 실행하지 못해요
    if call["name"] not in TOOLS:
        return {"error": "unknown_tool"}                         # 허용 목록에 없는 도구는 실행하지 않아요
    args = dict(call["args"])
    if call["name"] == "lookup_reservation":
        args["member_id"] = member_id                            # 누구인지는 모델이 아니라 서버(로그인)가 정해요
    try:
        return json.loads(TOOLS[call["name"]].invoke(args))
    except ValidationError:
        return {"error": "invalid_args"}                         # 입력 모양이 틀리면 실행 전에 막아요
    except RuntimeError as exc:
        return {"error": str(exc)}

Q_BOTH = "스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘"
messages = [{"role": "user", "content": Q_BOTH}]
first = fake_model(messages)                                     # 1) 모델: 도구를 불러 달라고 요청
tool_msgs = [{"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": json.dumps(execute(c, "m-001"), ensure_ascii=False)}
             for c in first["tool_calls"]]                       # 2) 앱: 확인하고 실행
final = fake_model(messages + [first] + tool_msgs)               # 3) 모델: 결과를 보고 답

# ④ 반복으로 묶기: 답이 나올 때까지
def run_agent(question, member_id="m-001", max_steps=4):         # 모델 → 도구 → 모델 …을 답이 나올 때까지, 최대 max_steps번
    messages, ledger = [{"role": "user", "content": question}], []
    for _ in range(max_steps):
        reply = fake_model(messages)
        messages.append(reply)
        if not reply.get("tool_calls"):
            return reply["content"], messages, ledger
        for call in reply["tool_calls"]:
            result = execute(call, member_id)
            ledger.append({"tool": call["name"], "result": result})   # 서버가 따로 남기는 도구 실행 기록 (⑥에서 써요)
            messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"], "content": json.dumps(result, ensure_ascii=False)})
    raise RuntimeError("tool_call_limit")                        # 계속 도구만 부르면 멈춰요

ASKS = ["스터디룸 예약을 철회하려면 어떻게 해?", "내 101번 예약 어떻게 됐어?", Q_BOTH, "안녕"]
traces = {q: run_agent(q) for q in ASKS}

# ⑤ 안전장치: 잘못된 입력, 남의 것, 없는 도구, 반복, DB 장애
def scripted(replies, question="예약 확인해 줘", member_id="m-001"):   # 가짜 모델의 응답을 미리 정해 두고 실행
    NEXT.extend(replies)
    try:
        answer, _, ledger = run_agent(question, member_id)
        return {"outcome": "답함", "answer": answer, "tool_results": [e["result"] for e in ledger]}
    except RuntimeError as exc:
        return {"outcome": "중단", "error": str(exc)}
    finally:
        NEXT.clear()

def ask_tool(name, args):                                        # 모델이 도구 하나를 요청하는 응답
    return {"role": "assistant", "tool_calls": [{"id": "call_x", "name": name, "args": args}]}

guards = {"잘못된 입력 (-1)": scripted([ask_tool("lookup_reservation", {"reservation_id": -1})]),
          "남의 예약 (202)": scripted([ask_tool("lookup_reservation", {"reservation_id": 202})]),
          "없는 도구": scripted([ask_tool("delete_reservation", {"reservation_id": 101})]),
          "같은 도구를 계속": scripted([ask_tool("lookup_reservation", {"reservation_id": 101})] * 6)}
DB_URI, real_uri = "file:missing.db?mode=ro", DB_URI             # DB가 멈춘 상황 (열 수 없는 파일)
guards["DB 장애"] = scripted([ask_tool("lookup_reservation", {"reservation_id": 101})])
DB_URI = real_uri

# ⑥ 답 확인하기: 출처와 도구 실패
def check_answer(answer, ledger):                                # 답에 붙은 출처 번호가 이번 실행의 검색 결과 안에 있나, 도구가 실패했나
    cited = {int(n) for n in re.findall(r"\[출처 (\d+)\]", answer)}
    returned = {p["id"] for e in ledger if e["tool"] == "search_rules" for p in e["result"].get("passages", [])}
    problems = [f"확인 안 된 출처 {sorted(cited - returned)}"] if not cited <= returned else []
    problems += [f"도구 실패 {e['result']['error']}" for e in ledger if "error" in e["result"]]
    return {"cited": sorted(cited), "returned": sorted(returned), "ok": not problems, "problems": problems}

checked = {q: check_answer(answer, ledger) for q, (answer, _, ledger) in traces.items()}
fake_cite = check_answer("1시간 전까지 철회해요. [출처 99]", traces[ASKS[0]][2])   # 모델이 출처를 지어내면

# ⑦ 회귀 시나리오: 바꿀 때마다 같은 질문으로
SCENARIOS = [  # (이름, 질문, 회원, 불려야 할 도구, 답에 있어야 할 말)
    ("규정만", "스터디룸 예약을 철회하려면 어떻게 해?", "m-001", ["search_rules"], "[출처"),
    ("내 예약만", "내 101번 예약 어떻게 됐어?", "m-001", ["lookup_reservation"], "예약됨"),
    ("둘 다", Q_BOTH, "m-001", ["search_rules", "lookup_reservation"], "예약됨"),
    ("남의 예약", "202번 예약 상태 알려줘", "m-001", ["lookup_reservation"], "찾을 수 없어요"),
    ("주인은 보임", "202번 예약 상태 알려줘", "m-002", ["lookup_reservation"], "예약됨"),
    ("도구가 필요 없음", "안녕", "m-001", [], "물어봐"),
]
def run_regression():                                           # 시나리오마다 도구, 답, 출처 검사를 통과했나
    rows = []
    for name, q, member, need, must in SCENARIOS:
        answer, _, ledger = run_agent(q, member)
        rows.append({"name": name, "tools_ok": [e["tool"] for e in ledger] == need, "answer_ok": must in answer,
                     "check_ok": check_answer(answer, ledger)["ok"]})
    return rows

regression = run_regression()
PLANS[Q_BOTH] = PLANS[Q_BOTH][:1]                                # 모델을 바꿨더니 '둘 다' 질문에서 예약 조회를 빼먹는다고 치면
after_change = run_regression()
