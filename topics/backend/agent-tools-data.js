window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import re",
   "s": 6
  },
  {
   "t": "import sqlite3",
   "s": 2
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "from typing import Annotated, Literal",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "from langchain_core.tools import InjectedToolArg, tool",
   "s": 1
  },
  {
   "t": "from pydantic import Field, ValidationError",
   "s": 2
  },
  {
   "t": "from sentence_transformers import SentenceTransformer",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 규정 검색을 도구로: 이름, 설명, 입력 모양, 결과",
   "s": 1
  },
  {
   "t": "CATEGORY = {\"guide.md\": \"이용\", \"hours.html\": \"이용\", \"room.md\": \"이용\", \"holds.md\": \"이용\", \"programs.md\": \"이용\", \"devices.md\": \"이용\",",
   "s": 1
  },
  {
   "t": "            \"notice.txt\": \"공지\", \"old_notice.txt\": \"공지\", \"notice_2025.txt\": \"공지\", \"notice_2026.txt\": \"공지\",",
   "s": 1
  },
  {
   "t": "            \"rules.pdf\": \"규정\", \"scan.pdf\": \"시설\"}",
   "s": 1
  },
  {
   "t": "raw = (json.loads((HERE.parent / \"ingest\" / \"out\" / \"chunks.json\").read_text(encoding=\"utf-8\"))",
   "s": 1
  },
  {
   "t": "       + json.loads((HERE.parent / \"rag\" / \"data\" / \"more_chunks.json\").read_text(encoding=\"utf-8\")))",
   "s": 1
  },
  {
   "t": "CHUNKS = [{\"id\": n, \"title\": c[\"heading\"] or c[\"body\"].split(\"\\n\")[0], \"body\": c[\"body\"], \"category\": CATEGORY[c[\"file\"]],",
   "s": 1
  },
  {
   "t": "           \"current\": c.get(\"current\", True)} for n, c in enumerate(raw, start=1)]",
   "s": 1
  },
  {
   "t": "encoder = SentenceTransformer(\"intfloat/multilingual-e5-small\")",
   "s": 1
  },
  {
   "t": "vectors = encoder.encode([f\"passage: {c['title']}\\n{c['body']}\" for c in CHUNKS], normalize_embeddings=True)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "@tool",
   "s": 1
  },
  {
   "t": "def search_rules(query: str, category: Literal[\"이용\", \"공지\", \"규정\", \"시설\"] = \"이용\") -> str:",
   "s": 1
  },
  {
   "t": "    \"\"\"도서관 이용 규정과 안내문에서 질문과 가까운 조각 2개를 찾아요. 회원 개인의 예약 상태는 찾지 못해요.\"\"\"",
   "s": 1
  },
  {
   "t": "    q = encoder.encode([f\"query: {query}\"], normalize_embeddings=True)[0]",
   "s": 1
  },
  {
   "t": "    order = [i for i in np.argsort(-(vectors @ q)) if CHUNKS[i][\"category\"] == category and CHUNKS[i][\"current\"]][:2]",
   "s": 1
  },
  {
   "t": "    return json.dumps({\"status\": \"ok\", \"passages\": [{\"id\": CHUNKS[i][\"id\"], \"title\": CHUNKS[i][\"title\"], \"text\": CHUNKS[i][\"body\"]} for i in order]},",
   "s": 1
  },
  {
   "t": "                      ensure_ascii=False)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "search_schema = search_rules.tool_call_schema.model_json_schema()  # 모델에게 보이는 이름, 설명, 입력 모양",
   "s": 1
  },
  {
   "t": "search_direct = json.loads(search_rules.invoke({\"query\": \"스터디룸 예약 철회 방법\"}))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 예약 조회를 도구로: 읽기 전용, 본인 것만",
   "s": 2
  },
  {
   "t": "DB_URI = \"file:garam?mode=memory&cache=shared\"  # 예시 DB (메모리 안). 실제로는 PostgreSQL 같은 DB 서버",
   "s": 2
  },
  {
   "t": "admin = sqlite3.connect(DB_URI, uri=True)  # 관리자 쪽 연결: 표를 만들고 채워요 (이 연결이 열려 있는 동안 DB가 남아요)",
   "s": 2
  },
  {
   "t": "admin.execute(\"CREATE TABLE reservations (reservation_id INTEGER PRIMARY KEY, member_id TEXT, room TEXT, use_date TEXT, start_hour INTEGER, status TEXT)\")",
   "s": 2
  },
  {
   "t": "admin.executemany(\"INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)\", [",
   "s": 2
  },
  {
   "t": "    (101, \"m-001\", \"스터디룸 A\", \"2026-11-28\", 14, \"예약됨\"), (102, \"m-001\", \"스터디룸 B\", \"2026-11-21\", 10, \"철회함\"),",
   "s": 2
  },
  {
   "t": "    (202, \"m-002\", \"스터디룸 A\", \"2026-11-28\", 16, \"예약됨\")])  # 가상의 회원 둘",
   "s": 2
  },
  {
   "t": "admin.commit()",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def read_reservation(reservation_id, member_id):  # 로그인한 회원 본인의 예약만 읽어요",
   "s": 2
  },
  {
   "t": "    try:",
   "s": 2
  },
  {
   "t": "        conn = sqlite3.connect(DB_URI, uri=True)",
   "s": 2
  },
  {
   "t": "        conn.execute(\"PRAGMA query_only = ON\")  # 이 연결로는 바꾸는 SQL이 실패해요",
   "s": 2
  },
  {
   "t": "        row = conn.execute(\"SELECT reservation_id, room, use_date, start_hour, status FROM reservations \"",
   "s": 2
  },
  {
   "t": "                           \"WHERE reservation_id = ? AND member_id = ?\", (reservation_id, member_id)).fetchone()  # 값은 ? 자리로만",
   "s": 2
  },
  {
   "t": "        conn.close()",
   "s": 2
  },
  {
   "t": "    except sqlite3.OperationalError as exc:",
   "s": 2
  },
  {
   "t": "        raise RuntimeError(\"db_unavailable\") from exc  # DB 장애는 '없음'이 아니라 오류",
   "s": 2
  },
  {
   "t": "    if row is None:  # 없는 번호와 남의 예약을 구별해 알려 주지 않아요",
   "s": 2
  },
  {
   "t": "        return {\"status\": \"not_found_or_forbidden\", \"reservation_id\": reservation_id}",
   "s": 2
  },
  {
   "t": "    return {\"status\": \"ok\", \"reservation_id\": row[0], \"room\": row[1], \"use_date\": row[2], \"start_hour\": row[3], \"reservation_status\": row[4]}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "@tool",
   "s": 2
  },
  {
   "t": "def lookup_reservation(reservation_id: Annotated[int, Field(ge=1, strict=True, description=\"조회할 스터디룸 예약 번호 (양의 정수)\")],",
   "s": 2
  },
  {
   "t": "                       member_id: Annotated[str, InjectedToolArg]) -> str:",
   "s": 2
  },
  {
   "t": "    \"\"\"로그인한 회원 본인의 스터디룸 예약 상태를 예약 번호로 읽어요. 이용 규정은 찾지 못해요.\"\"\"",
   "s": 2
  },
  {
   "t": "    return json.dumps(read_reservation(reservation_id, member_id), ensure_ascii=False)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "lookup_schema = lookup_reservation.tool_call_schema.model_json_schema()  # member_id는 모델에게 안 보여요",
   "s": 2
  },
  {
   "t": "db_checks = {name: read_reservation(rid, mid) for name, rid, mid in ((\"m-001의 101\", 101, \"m-001\"), (\"m-001의 202 (남의 예약)\", 202, \"m-001\"),",
   "s": 2
  },
  {
   "t": "                                                                     (\"m-002의 202\", 202, \"m-002\"), (\"m-001의 999 (없는 번호)\", 999, \"m-001\"))}",
   "s": 2
  },
  {
   "t": "reader = sqlite3.connect(DB_URI, uri=True)",
   "s": 2
  },
  {
   "t": "reader.execute(\"PRAGMA query_only = ON\")",
   "s": 2
  },
  {
   "t": "try:",
   "s": 2
  },
  {
   "t": "    reader.execute(\"UPDATE reservations SET status = '철회함' WHERE reservation_id = 101\")",
   "s": 2
  },
  {
   "t": "    write_blocked = None",
   "s": 2
  },
  {
   "t": "except sqlite3.OperationalError as exc:",
   "s": 2
  },
  {
   "t": "    write_blocked = str(exc)  # 읽기 전용 연결에서 바꾸려 하면",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 도구 호출 한 바퀴: 모델이 요청하고, 앱이 실행하고, 결과로 답해요",
   "s": 3
  },
  {
   "t": "TOOLS = {\"search_rules\": search_rules, \"lookup_reservation\": lookup_reservation}  # 앱이 허용한 도구",
   "s": 3
  },
  {
   "t": "PLANS = {  # 가짜 모델이 질문마다 요청할 도구 호출 (실제 모델은 도구 설명을 읽고 스스로 골라요)",
   "s": 3
  },
  {
   "t": "    \"스터디룸 예약을 철회하려면 어떻게 해?\": [(\"search_rules\", {\"query\": \"스터디룸 예약 철회 방법\"})],",
   "s": 3
  },
  {
   "t": "    \"내 101번 예약 어떻게 됐어?\": [(\"lookup_reservation\", {\"reservation_id\": 101})],",
   "s": 3
  },
  {
   "t": "    \"스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘\": [(\"search_rules\", {\"query\": \"스터디룸 예약 철회 방법\"}), (\"lookup_reservation\", {\"reservation_id\": 101})],",
   "s": 3
  },
  {
   "t": "    \"202번 예약 상태 알려줘\": [(\"lookup_reservation\", {\"reservation_id\": 202})],",
   "s": 3
  },
  {
   "t": "}",
   "s": 3
  },
  {
   "t": "NEXT = []  # ⑤에서 가짜 모델의 다음 응답을 미리 정할 때 써요",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def say(result):  # 가짜 모델이 도구 결과 하나를 답 문장으로 옮기는 방법",
   "s": 3
  },
  {
   "t": "    if \"passages\" in result:",
   "s": 3
  },
  {
   "t": "        top = result[\"passages\"][0]",
   "s": 3
  },
  {
   "t": "        return f\"{top['text'].split('. ')[0].rstrip('.')}. [출처 {top['id']}]\"",
   "s": 3
  },
  {
   "t": "    if result.get(\"status\") == \"ok\":",
   "s": 3
  },
  {
   "t": "        return f\"{result['reservation_id']}번 예약은 {result['room']} {result['use_date']} {result['start_hour']}시, 상태는 {result['reservation_status']}이에요.\"",
   "s": 3
  },
  {
   "t": "    if result.get(\"status\") == \"not_found_or_forbidden\":",
   "s": 3
  },
  {
   "t": "        return f\"{result['reservation_id']}번 예약은 찾을 수 없어요.\"",
   "s": 3
  },
  {
   "t": "    return f\"조회하지 못했어요({result.get('error')}).\"",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def fake_model(messages):  # 도구 결과가 아직 없으면 호출을 요청하고, 있으면 그걸로 답을 써요",
   "s": 3
  },
  {
   "t": "    if NEXT:",
   "s": 3
  },
  {
   "t": "        return NEXT.pop(0)",
   "s": 3
  },
  {
   "t": "    results = [json.loads(m[\"content\"]) for m in messages if m[\"role\"] == \"tool\"]",
   "s": 3
  },
  {
   "t": "    if results:",
   "s": 3
  },
  {
   "t": "        return {\"role\": \"assistant\", \"content\": \" \".join(say(r) for r in results)}",
   "s": 3
  },
  {
   "t": "    calls = PLANS.get(messages[0][\"content\"], [])",
   "s": 3
  },
  {
   "t": "    if not calls:",
   "s": 3
  },
  {
   "t": "        return {\"role\": \"assistant\", \"content\": \"도서관 규정이나 내 예약을 물어봐 주세요.\"}",
   "s": 3
  },
  {
   "t": "    return {\"role\": \"assistant\", \"tool_calls\": [{\"id\": f\"call_{n}\", \"name\": name, \"args\": args} for n, (name, args) in enumerate(calls)]}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def execute(call, member_id):  # 앱이 확인하고 실행해요. 모델은 직접 실행하지 못해요",
   "s": 3
  },
  {
   "t": "    if call[\"name\"] not in TOOLS:",
   "s": 3
  },
  {
   "t": "        return {\"error\": \"unknown_tool\"}  # 허용 목록에 없는 도구는 실행하지 않아요",
   "s": 3
  },
  {
   "t": "    args = dict(call[\"args\"])",
   "s": 3
  },
  {
   "t": "    if call[\"name\"] == \"lookup_reservation\":",
   "s": 3
  },
  {
   "t": "        args[\"member_id\"] = member_id  # 누구인지는 모델이 아니라 서버(로그인)가 정해요",
   "s": 3
  },
  {
   "t": "    try:",
   "s": 3
  },
  {
   "t": "        return json.loads(TOOLS[call[\"name\"]].invoke(args))",
   "s": 3
  },
  {
   "t": "    except ValidationError:",
   "s": 3
  },
  {
   "t": "        return {\"error\": \"invalid_args\"}  # 입력 모양이 틀리면 실행 전에 막아요",
   "s": 3
  },
  {
   "t": "    except RuntimeError as exc:",
   "s": 3
  },
  {
   "t": "        return {\"error\": str(exc)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "Q_BOTH = \"스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘\"",
   "s": 3
  },
  {
   "t": "messages = [{\"role\": \"user\", \"content\": Q_BOTH}]",
   "s": 3
  },
  {
   "t": "first = fake_model(messages)  # 1) 모델: 도구를 불러 달라고 요청",
   "s": 3
  },
  {
   "t": "tool_msgs = [{\"role\": \"tool\", \"tool_call_id\": c[\"id\"], \"name\": c[\"name\"], \"content\": json.dumps(execute(c, \"m-001\"), ensure_ascii=False)}",
   "s": 3
  },
  {
   "t": "             for c in first[\"tool_calls\"]]  # 2) 앱: 확인하고 실행",
   "s": 3
  },
  {
   "t": "final = fake_model(messages + [first] + tool_msgs)  # 3) 모델: 결과를 보고 답",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 반복으로 묶기: 답이 나올 때까지",
   "s": 4
  },
  {
   "t": "def run_agent(question, member_id=\"m-001\", max_steps=4):  # 모델 → 도구 → 모델 …을 답이 나올 때까지, 최대 max_steps번",
   "s": 4
  },
  {
   "t": "    messages, ledger = [{\"role\": \"user\", \"content\": question}], []",
   "s": 4
  },
  {
   "t": "    for _ in range(max_steps):",
   "s": 4
  },
  {
   "t": "        reply = fake_model(messages)",
   "s": 4
  },
  {
   "t": "        messages.append(reply)",
   "s": 4
  },
  {
   "t": "        if not reply.get(\"tool_calls\"):",
   "s": 4
  },
  {
   "t": "            return reply[\"content\"], messages, ledger",
   "s": 4
  },
  {
   "t": "        for call in reply[\"tool_calls\"]:",
   "s": 4
  },
  {
   "t": "            result = execute(call, member_id)",
   "s": 4
  },
  {
   "t": "            ledger.append({\"tool\": call[\"name\"], \"result\": result})  # 서버가 따로 남기는 도구 실행 기록 (⑥에서 써요)",
   "s": 4
  },
  {
   "t": "            messages.append({\"role\": \"tool\", \"tool_call_id\": call[\"id\"], \"name\": call[\"name\"], \"content\": json.dumps(result, ensure_ascii=False)})",
   "s": 4
  },
  {
   "t": "    raise RuntimeError(\"tool_call_limit\")  # 계속 도구만 부르면 멈춰요",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "ASKS = [\"스터디룸 예약을 철회하려면 어떻게 해?\", \"내 101번 예약 어떻게 됐어?\", Q_BOTH, \"안녕\"]",
   "s": 4
  },
  {
   "t": "traces = {q: run_agent(q) for q in ASKS}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 안전장치: 잘못된 입력, 남의 것, 없는 도구, 반복, DB 장애",
   "s": 5
  },
  {
   "t": "def scripted(replies, question=\"예약 확인해 줘\", member_id=\"m-001\"):  # 가짜 모델의 응답을 미리 정해 두고 실행",
   "s": 5
  },
  {
   "t": "    NEXT.extend(replies)",
   "s": 5
  },
  {
   "t": "    try:",
   "s": 5
  },
  {
   "t": "        answer, _, ledger = run_agent(question, member_id)",
   "s": 5
  },
  {
   "t": "        return {\"outcome\": \"답함\", \"answer\": answer, \"tool_results\": [e[\"result\"] for e in ledger]}",
   "s": 5
  },
  {
   "t": "    except RuntimeError as exc:",
   "s": 5
  },
  {
   "t": "        return {\"outcome\": \"중단\", \"error\": str(exc)}",
   "s": 5
  },
  {
   "t": "    finally:",
   "s": 5
  },
  {
   "t": "        NEXT.clear()",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def ask_tool(name, args):  # 모델이 도구 하나를 요청하는 응답",
   "s": 5
  },
  {
   "t": "    return {\"role\": \"assistant\", \"tool_calls\": [{\"id\": \"call_x\", \"name\": name, \"args\": args}]}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "guards = {\"잘못된 입력 (-1)\": scripted([ask_tool(\"lookup_reservation\", {\"reservation_id\": -1})]),",
   "s": 5
  },
  {
   "t": "          \"남의 예약 (202)\": scripted([ask_tool(\"lookup_reservation\", {\"reservation_id\": 202})]),",
   "s": 5
  },
  {
   "t": "          \"없는 도구\": scripted([ask_tool(\"delete_reservation\", {\"reservation_id\": 101})]),",
   "s": 5
  },
  {
   "t": "          \"같은 도구를 계속\": scripted([ask_tool(\"lookup_reservation\", {\"reservation_id\": 101})] * 6)}",
   "s": 5
  },
  {
   "t": "DB_URI, real_uri = \"file:missing.db?mode=ro\", DB_URI  # DB가 멈춘 상황 (열 수 없는 파일)",
   "s": 5
  },
  {
   "t": "guards[\"DB 장애\"] = scripted([ask_tool(\"lookup_reservation\", {\"reservation_id\": 101})])",
   "s": 5
  },
  {
   "t": "DB_URI = real_uri",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 답 확인하기: 출처와 도구 실패",
   "s": 6
  },
  {
   "t": "def check_answer(answer, ledger):  # 답에 붙은 출처 번호가 이번 실행의 검색 결과 안에 있나, 도구가 실패했나",
   "s": 6
  },
  {
   "t": "    cited = {int(n) for n in re.findall(r\"\\[출처 (\\d+)\\]\", answer)}",
   "s": 6
  },
  {
   "t": "    returned = {p[\"id\"] for e in ledger if e[\"tool\"] == \"search_rules\" for p in e[\"result\"].get(\"passages\", [])}",
   "s": 6
  },
  {
   "t": "    problems = [f\"확인 안 된 출처 {sorted(cited - returned)}\"] if not cited <= returned else []",
   "s": 6
  },
  {
   "t": "    problems += [f\"도구 실패 {e['result']['error']}\" for e in ledger if \"error\" in e[\"result\"]]",
   "s": 6
  },
  {
   "t": "    return {\"cited\": sorted(cited), \"returned\": sorted(returned), \"ok\": not problems, \"problems\": problems}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "checked = {q: check_answer(answer, ledger) for q, (answer, _, ledger) in traces.items()}",
   "s": 6
  },
  {
   "t": "fake_cite = check_answer(\"1시간 전까지 철회해요. [출처 99]\", traces[ASKS[0]][2])  # 모델이 출처를 지어내면",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 회귀 시나리오: 바꿀 때마다 같은 질문으로",
   "s": 7
  },
  {
   "t": "SCENARIOS = [  # (이름, 질문, 회원, 불려야 할 도구, 답에 있어야 할 말)",
   "s": 7
  },
  {
   "t": "    (\"규정만\", \"스터디룸 예약을 철회하려면 어떻게 해?\", \"m-001\", [\"search_rules\"], \"[출처\"),",
   "s": 7
  },
  {
   "t": "    (\"내 예약만\", \"내 101번 예약 어떻게 됐어?\", \"m-001\", [\"lookup_reservation\"], \"예약됨\"),",
   "s": 7
  },
  {
   "t": "    (\"둘 다\", Q_BOTH, \"m-001\", [\"search_rules\", \"lookup_reservation\"], \"예약됨\"),",
   "s": 7
  },
  {
   "t": "    (\"남의 예약\", \"202번 예약 상태 알려줘\", \"m-001\", [\"lookup_reservation\"], \"찾을 수 없어요\"),",
   "s": 7
  },
  {
   "t": "    (\"주인은 보임\", \"202번 예약 상태 알려줘\", \"m-002\", [\"lookup_reservation\"], \"예약됨\"),",
   "s": 7
  },
  {
   "t": "    (\"도구가 필요 없음\", \"안녕\", \"m-001\", [], \"물어봐\"),",
   "s": 7
  },
  {
   "t": "]",
   "s": 7
  },
  {
   "t": "def run_regression():  # 시나리오마다 도구, 답, 출처 검사를 통과했나",
   "s": 7
  },
  {
   "t": "    rows = []",
   "s": 7
  },
  {
   "t": "    for name, q, member, need, must in SCENARIOS:",
   "s": 7
  },
  {
   "t": "        answer, _, ledger = run_agent(q, member)",
   "s": 7
  },
  {
   "t": "        rows.append({\"name\": name, \"tools_ok\": [e[\"tool\"] for e in ledger] == need, \"answer_ok\": must in answer,",
   "s": 7
  },
  {
   "t": "                     \"check_ok\": check_answer(answer, ledger)[\"ok\"]})",
   "s": 7
  },
  {
   "t": "    return rows",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "regression = run_regression()",
   "s": 7
  },
  {
   "t": "PLANS[Q_BOTH] = PLANS[Q_BOTH][:1]  # 모델을 바꿨더니 '둘 다' 질문에서 예약 조회를 빼먹는다고 치면",
   "s": 7
  },
  {
   "t": "after_change = run_regression()",
   "s": 7
  }
 ],
 "names": {
  "1": "규정 검색을 도구로: 이름, 설명, 입력 모양, 결과",
  "2": "예약 조회를 도구로: 읽기 전용, 본인 것만",
  "3": "도구 호출 한 바퀴: 모델이 요청하고, 앱이 실행하고, 결과로 답해요",
  "4": "반복으로 묶기: 답이 나올 때까지",
  "5": "안전장치: 잘못된 입력, 남의 것, 없는 도구, 반복, DB 장애",
  "6": "답 확인하기: 출처와 도구 실패",
  "7": "회귀 시나리오: 바꿀 때마다 같은 질문으로"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① search_rules가 모델에게 보이는 모양 (tool_call_schema)",
   "lines": [
    [
     "title (도구 이름) \"search_rules\"",
     1
    ],
    [
     "description \"도서관 이용 규정과 안내문에서 질문과 가까운 조각 2개를 찾아요. 회원 개인의 예약 상태는 찾지 못해요.\"",
     2
    ],
    [
     "properties.query {\"title\": \"Query\", \"type\": \"string\"}, 꼭 채워야 함",
     3
    ],
    [
     "properties.category {\"default\": \"이용\", \"enum\": [\"이용\", \"공지\", \"규정\", \"시설\"], \"title\": \"Category\", \"type\": \"string\"}",
     4
    ],
    [
     "직접 실행 search_rules.invoke({\"query\": \"스터디룸 예약 철회 방법\"}) → passages [(14, '예약 철회'), (13, '스터디룸 예약')]",
     5,
     "newf"
    ]
   ],
   "notes": [
    "함수 이름이 도구 이름이 돼요. 모델은 이 이름으로 불러요.",
    "docstring이 설명이 돼요. 모델은 이 설명을 읽고 언제 이 도구를 쓸지 골라요. 그래서 '무엇은 못 한다'까지 적었어요.",
    "타입 힌트가 입력 모양이 돼요. query는 글자 하나예요.",
    "Literal로 넷 중 하나만 받게 했어요(enum). 기본값이 있어서 안 채워도 돼요.",
    "모델 없이 도구만 직접 실행해 볼 수도 있어요. 결과는 글(JSON)로 돌려줘요. 모델은 결과를 글로 받거든요."
   ]
  },
  "t2": {
   "title": "② read_reservation(예약 번호, 회원)",
   "cols": [
    "누가, 무엇을",
    "결과"
   ],
   "rows": [
    [
     "m-001의 101",
     "{&quot;status&quot;: &quot;ok&quot;, &quot;reservation_id&quot;: 101, &quot;room&quot;: &quot;스터디룸 A&quot;, &quot;use_date&quot;: &quot;2026-11-28&quot;, &quot;start_hour&quot;: 14, &quot;reservation_status&quot;: &quot;예약됨&quot;}"
    ],
    [
     "m-001의 202 (남의 예약)",
     "{&quot;status&quot;: &quot;not_found_or_forbidden&quot;, &quot;reservation_id&quot;: 202}"
    ],
    [
     "m-002의 202",
     "{&quot;status&quot;: &quot;ok&quot;, &quot;reservation_id&quot;: 202, &quot;room&quot;: &quot;스터디룸 A&quot;, &quot;use_date&quot;: &quot;2026-11-28&quot;, &quot;start_hour&quot;: 16, &quot;reservation_status&quot;: &quot;예약됨&quot;}"
    ],
    [
     "m-001의 999 (없는 번호)",
     "{&quot;status&quot;: &quot;not_found_or_forbidden&quot;, &quot;reservation_id&quot;: 999}"
    ],
    [
     "읽기 전용 연결로 UPDATE",
     "<b>attempt to write a readonly database</b>"
    ]
   ],
   "notes": [
    "m-001은 자기 예약 101을 읽어요.",
    "m-001이 m-002의 202를 물으면 '없거나 볼 수 없음'이에요. 없는 번호 999와 같은 답이에요. '남의 예약이라 못 보여 준다'고 하면 그 번호가 있다는 걸 알려 주게 되거든요.",
    "WHERE에 member_id를 넣었기 때문에 이렇게 돼요. 프로그램 쪽에서 결과를 골라내는 게 아니라 DB에서부터 본인 것만 읽어요.",
    "이 연결은 읽기 전용(PRAGMA query_only)이라 바꾸는 SQL이 실패해요. 실수로든 모델의 요청으로든 예약이 바뀌지 않아요."
   ]
  },
  "s2": {
   "title": "② lookup_reservation이 모델에게 보이는 모양",
   "lines": [
    "{\"description\": \"로그인한 회원 본인의 스터디룸 예약 상태를 예약 번호로 읽어요. 이용 규정은 찾지 못해요.\", \"properties\": {\"reservation_id\": {\"description\": \"조회할 스터디룸 예약 번호 (양의 정수)\", \"minimum\": 1, \"title\": \"Reservation Id\", \"type\": \"integer\"}}, \"required\": [\"reservation_id\"], \"title\": \"lookup_reservation\", \"type\": \"object\"}"
   ],
   "notes": [
    "함수에는 member_id 인수가 있지만 InjectedToolArg로 표시해서 모델에게는 안 보여요. 모델은 예약 번호만 채우고, 누구인지는 실행할 때 앱이 로그인 정보로 넣어요(③). 모델이 member_id를 정하게 하면 \"m-002의 예약을 보여 줘\" 한 줄로 남의 정보를 볼 수 있어요.",
    "Field(ge=1, strict=True)라서 0 이하나 글자 \"101\"은 실행 전에 막혀요(⑤)."
   ]
  },
  "s3": {
   "title": "③ \"스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘\" 한 바퀴",
   "lines": [
    [
     "1) 모델 → 앱: tool_calls [{\"name\": \"search_rules\", \"args\": {\"query\": \"스터디룸 예약 철회 방법\"}}, {\"name\": \"lookup_reservation\", \"args\": {\"reservation_id\": 101}}]",
     1,
     "newf"
    ],
    [
     "2) 앱 → 모델: role \"tool\", name \"search_rules\", content {\"status\": \"ok\", \"passages\": [{\"id\": 14, \"title\": \"예약 철회\", \"text\": \"예약…",
     2
    ],
    [
     "2) 앱 → 모델: role \"tool\", name \"lookup_reservation\", content {\"status\": \"ok\", \"reservation_id\": 101, \"room\": \"스터디룸 A\", \"use_date\": …",
     2
    ],
    [
     "3) 모델 → 사용자: \"예약한 스터디룸을 쓰지 않게 되면 이용 시작 1시간 전까지 홈페이지의 '나의 신청'에서 철회합니다. [출처 14] 101번 예약은 스터디룸 A 2026-11-28 14시, 상태는 예약됨이에요.\"",
     3,
     "newf"
    ]
   ],
   "notes": [
    "모델은 직접 실행하지 않고 '이 도구를 이 인수로 불러 달라'고 요청만 해요. 질문 하나에 도구 둘을 한꺼번에 요청했어요. 요청마다 id가 붙어요.",
    "앱이 허용된 도구인지, 입력이 맞는지 보고 실행해요. 결과는 어느 요청의 결과인지(tool_call_id)와 함께 role \"tool\" 메시지로 돌려줘요. 예약 조회에는 앱이 로그인한 회원(m-001)을 넣었어요.",
    "모델은 결과를 읽고 답을 써요. 규정 쪽 답에는 검색 결과의 조각 번호를 출처로 붙였어요(⑥에서 확인해요)."
   ]
  },
  "t4": {
   "title": "④ run_agent(질문) (회원 m-001)",
   "cols": [
    "질문",
    "메시지 흐름",
    "부른 도구",
    "답"
   ],
   "rows": [
    [
     "스터디룸 예약을 철회하려면 어떻게 해?",
     "user → assistant(도구 요청) → tool → assistant",
     "search_rules",
     "예약한 스터디룸을 쓰지 않게 되면 이용 시작 1시간 전까지 홈페이지의 &#x27;나의 신청&#x27;에서 철회합니다. [출처 14]"
    ],
    [
     "내 101번 예약 어떻게 됐어?",
     "user → assistant(도구 요청) → tool → assistant",
     "lookup_reservation",
     "101번 예약은 스터디룸 A 2026-11-28 14시, 상태는 예약됨이에요."
    ],
    [
     "스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘",
     "user → assistant(도구 요청) → tool → tool → assistant",
     "search_rules, lookup_reservation",
     "예약한 스터디룸을 쓰지 않게 되면 이용 시작 1시간 전까지 홈페이지의 &#x27;나의 신청&#x27;에서 철회합니다. [출처 14] 101번 예약은 스터디룸 A 2026-11-28 14시, 상태는 예약됨이에요."
    ],
    [
     "안녕",
     "user → assistant",
     "없음",
     "도서관 규정이나 내 예약을 물어봐 주세요."
    ]
   ],
   "notes": [
    "질문에 따라 도구를 하나, 둘, 또는 하나도 안 불러요. 순서가 정해진 RAG 서비스와 다른 점이 이거예요. 다음에 무엇을 할지 모델이 골라요.",
    "도구가 필요 없는 질문은 user → assistant로 바로 끝나요.",
    "이 예제의 가짜 모델은 질문마다 미리 정한 계획(PLANS)대로 부르고, 도구 결과의 첫 문장을 옮겨 답해요. 그래서 이 표의 답 문장은 모델이 똑똑한지가 아니라 요청, 실행, 결과가 어떻게 흐르는지를 보여 줘요."
   ]
  },
  "t5": {
   "title": "⑤ 가짜 모델이 이상한 요청을 하면 (회원 m-001)",
   "cols": [
    "상황",
    "막은 곳",
    "도구 결과",
    "끝"
   ],
   "rows": [
    [
     "잘못된 입력 (-1)",
     "실행 전 입력 검사",
     "{&quot;error&quot;: &quot;invalid_args&quot;}",
     "<b>답함</b>: 조회하지 못했어요(invalid_args)."
    ],
    [
     "남의 예약 (202)",
     "DB 조회 조건 (WHERE member_id)",
     "{&quot;status&quot;: &quot;not_found_or_forbidden&quot;, &quot;reservation_id&quot;: 202}",
     "<b>답함</b>: 202번 예약은 찾을 수 없어요."
    ],
    [
     "없는 도구",
     "허용 목록 (TOOLS)",
     "{&quot;error&quot;: &quot;unknown_tool&quot;}",
     "<b>답함</b>: 조회하지 못했어요(unknown_tool)."
    ],
    [
     "같은 도구를 계속",
     "반복 상한 (max_steps=4)",
     "-",
     "<b>중단</b>: tool_call_limit"
    ],
    [
     "DB 장애",
     "DB 연결",
     "{&quot;error&quot;: &quot;db_unavailable&quot;}",
     "<b>답함</b>: 조회하지 못했어요(db_unavailable)."
    ]
   ],
   "notes": [
    "잘못된 입력과 없는 도구는 실행하지 않고, 무엇이 틀렸는지를 도구 결과로 모델에게 돌려줘요. 실제 모델은 이걸 보고 다시 시도할 수 있어요.",
    "남의 예약은 오류가 아니라 정상 결과(없거나 볼 수 없음)예요.",
    "같은 도구를 계속 부르면 4번에서 멈추고 오류(tool_call_limit)를 올려요. 상한이 없으면 비용과 시간이 끝없이 들어요.",
    "DB 장애는 '예약 없음'이 아니라 db_unavailable이에요. 고장을 '없음'으로 바꾸면 사용자에게 틀린 답을 주게 돼요.",
    "실패마다 이름이 달라야 다시 시도할지, 멈출지, 사용자에게 뭐라고 할지 정할 수 있어요."
   ]
  },
  "s6": {
   "title": "⑥ check_answer(답, 이번 실행 기록)",
   "lines": [
    [
     "\"스터디룸 예약을 철회하려면 어떻게 해?\" → 답의 출처 [14], 검색이 돌려준 조각 [13, 14] → ok True",
     1
    ],
    [
     "\"1시간 전까지 철회해요. [출처 99]\" → 출처 [99], 돌려준 조각 [13, 14] → ok False, 확인 안 된 출처 [99]",
     2,
     "bad"
    ]
   ],
   "notes": [
    "답에 적힌 출처 번호가 이번 실행에서 search_rules가 실제로 돌려준 조각 안에 있어요.",
    "모델이 출처 번호를 지어내면(99번 조각은 이번에 돌려준 적이 없어요) 걸려요. 모델의 마지막 문장은 그대로 믿지 않고, 앱이 따로 남긴 실행 기록(ledger)과 대조한 뒤 내보내요."
   ]
  },
  "t7": {
   "title": "⑦ 회귀 시나리오: 처음 → 모델을 바꾼 뒤",
   "cols": [
    "시나리오",
    "질문",
    "회원",
    "도구",
    "답",
    "출처 검사",
    "바꾼 뒤"
   ],
   "rows": [
    [
     "규정만",
     "스터디룸 예약을 철회하려면 어떻게 해?",
     "m-001",
     "통과",
     "통과",
     "통과",
     "통과"
    ],
    [
     "내 예약만",
     "내 101번 예약 어떻게 됐어?",
     "m-001",
     "통과",
     "통과",
     "통과",
     "통과"
    ],
    [
     "둘 다",
     "스터디룸 예약 철회 규정이랑 내 101번 예약 상태 알려줘",
     "m-001",
     "통과",
     "통과",
     "통과",
     "<b>실패</b> (도구 <b>실패</b>, 답 <b>실패</b>)"
    ],
    [
     "남의 예약",
     "202번 예약 상태 알려줘",
     "m-001",
     "통과",
     "통과",
     "통과",
     "통과"
    ],
    [
     "주인은 보임",
     "202번 예약 상태 알려줘",
     "m-002",
     "통과",
     "통과",
     "통과",
     "통과"
    ],
    [
     "도구가 필요 없음",
     "안녕",
     "m-001",
     "통과",
     "통과",
     "통과",
     "통과"
    ]
   ],
   "notes": [
    "처음에는 여섯 시나리오가 모두 통과해요. 남의 예약(m-001이 202)은 안 보이고, 주인(m-002)에게는 보여요.",
    "모델을 바꿨더니 '둘 다' 질문에서 예약 조회를 빼먹는다고 쳐 보면(PLANS를 고침), 그 시나리오만 실패로 바뀌어요. 답에 규정 설명은 있어서 얼핏 그럴듯한데, '불려야 할 도구'를 함께 확인하니 바로 걸려요.",
    "모델, 도구 설명, 규칙(system prompt)을 바꿀 때마다 같은 시나리오를 돌려요. 실제 모델은 같은 질문에도 고르는 도구가 흔들릴 수 있어서 여러 번 돌려 봐요."
   ]
  }
 },
 "flow": [
  [
   1,
   "검색 도구",
   "이름, 설명, 입력 모양",
   "docstring이 설명"
  ],
  [
   2,
   "DB 도구",
   "읽기 전용, 본인 것만",
   "member_id는 숨김"
  ],
  [
   3,
   "한 바퀴",
   "요청 → 실행 → 답",
   "모델은 요청만"
  ],
  [
   4,
   "반복",
   "답이 나올 때까지",
   "도구 0~2개"
  ],
  [
   5,
   "안전장치",
   "입력, 남의 것, 없는 도구, 반복, 장애",
   "실패마다 다른 이름"
  ],
  [
   6,
   "답 확인",
   "출처를 실행 기록과",
   "지어낸 출처 막기"
  ],
  [
   7,
   "회귀",
   "같은 시나리오로",
   "바꾼 뒤 실패 1"
  ]
 ],
 "env": "Python 3.13.16, langchain-core 1.8.4, pydantic 2.13.4, sentence-transformers 6.3.0, SQLite 3.50.4, intfloat/multilingual-e5-small"
};
