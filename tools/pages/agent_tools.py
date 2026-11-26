"""백엔드 > Agent > AI Agent와 도구. examples/rag/rag_service.py 의 ㉕~㉜ (①~㉔는 RAG 서비스, 검색 고치기, RAG 평가 쪽).
rag_search 도구는 ⑯의 RAG 서비스(e5, bge-reranker GPU)를 실제로 부르고, lookup_reservation 도구는 노트 전용 PostgreSQL DB(sesac_notes_lab)를
실제로 읽는다(만들고, 쓰고, 지운다. 비밀번호는 출력하지 않음). Agent의 판단(어떤 도구를 부를지, 답 문장)은 ChatOpenAI에 붙인 가짜 서버가 한다.
실행: NOTES_PG_ENV=<.env 경로> HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/agent_tools.py
"""
import html
import json
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import notes_pg  # noqa: E402
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = (ROOT / "examples" / "rag").resolve()
code = S.parse((EX / "rag_service.py").read_text(encoding="utf-8"))
notes_pg.fresh()
try:
    ns = S.run(code, 32, EX, filename="rag_service.py")
    with notes_pg.psycopg.connect() as _c:
        PG = _c.execute("SHOW server_version").fetchone()[0]
finally:
    notes_pg.drop()
V = {p: version(p) for p in ("langchain", "langgraph", "langchain-core", "langchain-openai", "psycopg", "fastapi")}
ENV = f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items()) + f", PostgreSQL {PG.split()[0]}"
E = html.escape
outs = {}


def short(s, n=70):
    s = str(s)
    return E(s[:n] + ("…" if len(s) > n else ""))


# ---------------- ㉕
rs, rd = ns["rag_schema"], ns["rag_direct"]
q1, q2 = list(rd)
S.claim(list(rs["properties"]) == ["query", "category"] and rs["required"] == ["query"] and rd[q1]["status"] == "answered" and rd[q2]["status"] == "insufficient_context", "㉕ 입력 둘, 하나는 답, 하나는 정보 부족")
outs["s25"] = S.box("㉕ 모델에게 보이는 rag_search (tool_call_schema)", [(f"name: rag_search", 1), (f"description: {E(rs['description'])}", 2),
                                                                     (f"parameters: query (글자, 필수), category ({', '.join(rs['properties']['category']['enum'])} 중 하나, 기본 {rs['properties']['category']['default']})", 3, "newf"),
                                                                     (f"rag_search.invoke(\"{E(q1)}\") → status {rd[q1]['status']}, 출처 {[s['citation_id'] for s in rd[q1]['sources']]} ({E(rd[q1]['sources'][0]['where'])})", 4),
                                                                     (f"rag_search.invoke(\"{E(q2)}\", category=\"규정\") → status {rd[q2]['status']}, 출처 {rd[q2]['sources']}", 5)],
                   ["모델이 고를 때 보는 이름이에요. 짧고 하는 일이 드러나게 지어요.",
                    "설명은 \"언제 쓰는지\"와 \"무엇은 못 하는지\"를 같이 적어요. 이 설명을 보고 모델이 개인 예약 질문에 이 도구를 부르지 않게 돼요. 함수의 docstring이 그대로 설명이 돼요.",
                    "타입 힌트에서 입력 모양(JSON Schema)이 만들어져요. <code>Literal</code>로 분류를 넷으로 묶어 두면 모델이 다른 값을 넣을 수 없어요.",
                    "안쪽은 <a href=\"rag-rerank.html\">RAG 검색 고치기</a> ⑯의 서비스 그대로예요. 결과는 서버가 확인한 출처만 담아 정해진 모양(status, answer, sources)으로 바꿔 돌려줘요.",
                    "자료에 답이 없으면 정보 부족이에요. 이건 정상 결과고, 서비스가 고장 나면 이 모양으로 바꾸지 않고 오류를 그대로 올려요."])

# ---------------- ㉖
dc, ls, ba = ns["db_checks"], ns["lookup_schema"], ns["bad_args"]
S.claim(dc["m-001의 101"]["status"] == "ok" and dc["m-001의 202 (남의 예약)"]["status"] == dc["m-001의 999 (없는 번호)"]["status"] == "not_found_or_forbidden" and dc["m-002의 202"]["status"] == "ok", "㉖ 본인 것만 보이고 남의 것과 없는 번호는 같은 결과")
S.claim(list(ls["properties"]) == ["reservation_id"] and ns["write_blocked"] == "ReadOnlySqlTransaction" and set(ba.values()) == {"greater_than_equal", "int_type"}, "㉖ runtime은 안 보이고, 쓰기는 막히고, 이상한 값은 걸린다")
outs["t26"] = S.table("㉖ read_reservation(번호, 회원) (노트 전용 PostgreSQL)", ["누가", "무엇을", "결과"],
                      [[k.split("의 ")[0], k.split("의 ")[1], short(json.dumps(v, ensure_ascii=False), 90)] for k, v in dc.items()],
                      ["SQL의 WHERE에 <code>member_id = %s</code>가 있어서, 같은 202번이라도 주인(m-002)에게만 보여요.",
                       "남의 예약과 없는 번호가 같은 결과(not_found_or_forbidden)예요. \"그 번호는 다른 사람 것\"이라고 알려 주는 것도 정보가 새는 거라 구별하지 않아요."])
outs["s26"] = S.box("모델에게 보이는 것과 막히는 것", [(f"lookup_reservation의 입력 모양 → {list(ls['properties'])}만 (runtime은 없음)", 1, "newf"),
                                                 *[(f"reservation_id={E(k)} → {v}", 2) for k, v in ba.items()],
                                                 (f"읽기 전용 연결로 UPDATE → {ns['write_blocked']}", 3, "newf")],
                   ["함수 인수의 <code>runtime: ToolRuntime[Member]</code>는 Agent가 실행할 때 서버가 채워 넣는 자리라, 모델에게 보이는 입력 모양에 없어요. 그래서 모델은 \"누구의 예약인지\"를 정할 수 없어요.",
                    "Pydantic의 <code>Field(ge=1, strict=True)</code>가 음수와 글자를 SQL에 닿기 전에 막아요. \"101 OR 1=1\" 같은 글자는 정수가 아니라 걸려요. 값이 통과해도 SQL에는 <code>%s</code> 자리로만 넘겨서, 글자가 SQL 문장으로 읽히지 않아요.",
                    "연결을 읽기 전용으로 열어 두면 도구 안의 실수로도 바꿀 수 없어요. 실제 서비스에서는 SELECT만 허락된 DB 계정을 따로 써요."])

# ---------------- ㉗
first, tmsgs, final = ns["first"], ns["tool_msgs"], ns["final"]
S.claim([c["name"] for c in first.tool_calls] == ["rag_search", "lookup_reservation"] and [m.tool_call_id for m in tmsgs] == [c["id"] for c in first.tool_calls] and ns["second_input"].count("function_call_output") == 2 and "[출처 14]" in final.text, "㉗ 두 호출, ID로 짝, 결과 둘을 붙여 다시, 답에 출처")
outs["s27"] = S.box("㉗ 손으로 한 바퀴", [(f"bind_tools 뒤 요청에 실린 tools → {[t['name'] for t in ns['tools_sent']]}", 1),
                                        (f"1) 모델의 응답 AIMessage.tool_calls → " + "; ".join(f"{c['name']}({short(json.dumps(c['args'], ensure_ascii=False), 60)}) id={c['id']}" for c in first.tool_calls), 2, "newf"),
                                        *[(f"2) 서버가 실행 → ToolMessage(name={m.name}, tool_call_id={m.tool_call_id}, content={short(m.content, 50)})", 3) for m in tmsgs],
                                        (f"3) 다시 보낸 요청의 input → {ns['second_input']}", 4), (f"4) 최종 답 → \"{E(final.text)}\"", 5)],
                   ["<code>bind_tools</code>는 도구의 이름, 설명, 입력 모양을 요청에 실어 보내기만 해요. 도구를 실행하지는 않아요.",
                    "모델은 글 대신 \"이 도구를 이 인수로 불러 달라\"는 요청을 돌려줘요. 한 번에 둘을 요청할 수도 있어요. 이건 요청일 뿐이라 아직 아무것도 조회되지 않았어요.",
                    "실행은 우리 서버가 해요. 허용 목록에 있는 도구인지 보고, 예약 조회에는 모델이 준 인수가 아니라 서버가 아는 회원(m-001)을 넣어요. 결과는 요청의 id를 붙인 ToolMessage가 돼서, 여러 호출이어도 어느 결과가 어느 요청 것인지 짝이 맞아요.",
                    "앞의 대화, 모델의 호출 요청, 도구 결과를 모두 붙여 모델을 다시 불러요. 모델 쪽에서 보면 대화가 이어지는 거예요.",
                    "도구 결과를 읽은 모델이 답을 써요(여기서는 가짜 서버가 결과를 이어 붙여요). 이 \"요청 → 실행 → 결과 → 다시 판단\"을 답이 나올 때까지 되풀이하는 것이 Agent 루프예요."])

# ---------------- ㉘
tr = ns["traces"]
asks = ns["ASKS"]
S.claim([len(tr[q]["tools"]) for q in asks] == [1, 1, 2, 0] and tr[asks[2]]["kinds"].count("ToolMessage") == 2 and tr[asks[3]]["kinds"] == ["HumanMessage", "AIMessage"], "㉘ 도구 1, 1, 2, 0개")
outs["t28"] = S.table("㉘ agent.invoke(질문, context=Member(\"m-001\")) 네 번", ["질문", "메시지 순서", "부른 도구", "최종 답 (final_text)"],
                      [[E(q), " → ".join(k.replace("Message", "") for k in tr[q]["kinds"]), ", ".join(tr[q]["tools"]) or "없음", short(tr[q]["answer"], 90)] for q in asks],
                      ["규정만 물으면 rag_search 하나, 내 예약만 물으면 lookup_reservation 하나, 둘 다 물으면 둘, 인사에는 도구 없이 바로 답했어요. 몇 개를 부를지는 모델이 정해서, \"이 질문엔 꼭 두 번\"이라고 가정하면 안 돼요.",
                       "결과는 답 글 하나가 아니라 메시지 목록이에요. 맨 뒤의 \"도구 호출이 없는 AIMessage\"가 답이에요. 중간의 ToolMessage는 실행 기록이라, 출처 확인(㉚)에 써요.",
                       "둘을 한 번에 요청하면 두 도구가 함께 실행돼서, 끝나는 순서는 매번 다를 수 있어요."])

# ---------------- ㉙
g = ns["guards"]
S.claim(g["잘못된 인수 (-1)"]["outcome"] == "답함" and "greater than or equal to 1" in g["잘못된 인수 (-1)"]["tool_results"][0] and "not a valid tool" in g["없는 도구"]["tool_results"][0]
        and g["남의 예약 (202)"]["answer"].endswith("찾을 수 없어요.") and g["같은 도구를 계속"]["error"] == "ToolCallLimitExceededError" and "5/4" in g["같은 도구를 계속"]["detail"] and g["DB 장애"]["outcome"] == "중단", "㉙ 다섯 경우가 글과 같다")
G_NOTE = {"잘못된 인수 (-1)": "입력 모양 검사에서 걸려, 오류 문장이 ToolMessage로 모델에게 돌아감", "남의 예약 (202)": "정상 실행, SQL이 본인 것만 읽어서 결과 없음",
          "없는 도구": "등록되지 않은 이름이라 실행 안 함, 오류 문장이 모델에게 돌아감", "같은 도구를 계속": "다섯 번째 호출에서 상한(4번)에 걸려 실행 전체가 멈춤", "DB 장애": "도구 안의 예외가 그대로 올라와 실행이 멈춤"}
outs["t29"] = S.table("㉙ 가짜 모델이 이상한 호출을 하면", ["경우", "무슨 일이 일어났나", "결과 (모델이 본 것 또는 오류)"],
                      [[k, G_NOTE[k], f"<b>{v['outcome']}</b>: " + short(v.get("tool_results", [v.get("detail", "")])[0] if v["outcome"] == "답함" else f"{v['error']}: {v['detail']}", 80)] for k, v in g.items()],
                      ["입력 오류와 없는 도구는 create_agent가 실행을 멈추지 않고 \"고쳐서 다시 해 보라\"는 문장을 모델에게 돌려줘요. 모델이 다시 시도할 수 있게 하려는 거예요. 그래서 반복 상한이 꼭 있어야 해요.",
                       "<code>ToolCallLimitMiddleware(run_limit=4, exit_behavior=\"error\")</code>는 한 번의 실행에서 도구 호출이 4번을 넘으면 예외로 멈춰요. 같은 조회를 끝없이 되풀이해 비용이 쌓이는 걸 막아요. 도구별 상한(<code>tool_name=</code>)도 따로 걸 수 있어요.",
                       "남의 예약은 실패가 아니라 정상 결과예요. 권한은 Prompt로 \"남의 것은 보지 마\"라고 부탁하는 게 아니라 서버의 SQL에서 막아요.",
                       "DB 장애는 \"예약이 없음\"으로 바꾸지 않고 오류로 올려요. 사용자에게 \"예약이 없어요\"라고 잘못 말하는 것보다 \"지금 조회할 수 없어요\"가 맞으니까요(㉜에서 502)."])

# ---------------- ㉚
ck, fk, au = ns["checks"], ns["fake_cite"], ns["audit"]
S.claim(all(v["ok"] for v in ck.values()) and not fk["ok"] and fk["problems"] == ["확인 안 된 출처 [99]"] and all(set(a) == {"request_id", "actor", "tool", "status"} for a in au), "㉚ 진짜 답은 통과, 지어낸 출처는 걸림, 기록에는 네 칸만")
outs["s30"] = S.box("㉚ validate_agent_answer와 보안 기록", [*[(f"\"{short(q, 30)}\" → 답의 출처 {v['cited']}, 이번 실행에서 확인된 출처 {v['verified']}, 통과 {v['ok']}", 1) for q, v in ck.items()],
                                                            (f"출처를 지어낸 답 \"… [출처 99]\" → {fk['problems'][0]}", 2, "newf"),
                                                            *[(f"보안 기록 → {E(json.dumps(a, ensure_ascii=False))}", 3) for a in au]],
                   ["답에 붙은 출처 번호를 모델의 말이 아니라 <b>이번 실행에서 rag_search가 실제로 돌려준 번호</b>(서버가 따로 남긴 TOOL_LEDGER)와 대조해요. <a href=\"rag-service.html\">RAG 서비스</a> ⑥의 \"넣은 조각만 출처로\"를 Agent에 옮긴 거예요.",
                    "모델이 그럴듯한 출처 번호를 지어내도 이번 실행 기록에 없으면 걸려요. 내용이 그 출처와 맞는지(충실성)는 <a href=\"rag-eval.html\">RAG 평가</a> ⑳처럼 따로 재요.",
                    "실행 기록(trace)은 개발할 때 \"어떻게 실행됐나\"를 보려는 것이고, 보안 기록(audit)은 \"누가 어떤 도구로 무엇에 접근했나\"를 남기는 거예요. 보안 기록에는 회원 ID를 해시로 바꾸고, 질문 원문, 토큰, 키는 남기지 않아요."])

# ---------------- ㉛
reg = ns["regression"]
S.claim(len(reg) == 7 and all(r["tools_ok"] and r["answer_ok"] and r["citations_ok"] for r in reg), "㉛ 일곱 시나리오 모두 통과")
ok = lambda b: "통과" if b else "<b>실패</b>"  # noqa: E731
outs["t31"] = S.table("㉛ 회귀 테스트 일곱 개", ["시나리오", "부른 도구", "도구 선택", "답 내용", "출처 확인"], [[r["name"], ", ".join(r["tools"]) or "없음", ok(r["tools_ok"]), ok(r["answer_ok"]), ok(r["citations_ok"])] for r in reg],
                      ["Agent는 답 문장만 보면 안 되고, <b>어떤 도구를 불렀는지</b>까지 봐야 해요. 규정 질문에 DB만 보고 답했다면 문장이 그럴듯해도 실패예요.",
                       "Prompt, 도구 설명, 모델을 바꿀 때마다 같은 시나리오를 다시 돌려요. 이 쪽은 가짜 모델이라 늘 통과하지만, 실제 모델은 판마다 도구 선택이 달라질 수 있어서 이 표가 가장 먼저 깨지는 곳이에요. 검색과 답의 품질은 <a href=\"rag-eval.html\">RAG 평가</a>의 지표로 따로 재요."])

# ---------------- ㉜
api = ns["agent_api"]
S.claim([api[k][0] for k in api] == [401, 200, 200, 422, 422, 502] and api["본문에 member_id"][1]["detail"][0]["type"] == "extra_forbidden", "㉜ 401, 200, 200, 422, 422, 502")


def api_sum(c, b):
    if c == 200:
        return f"tools {b['tools']}, sources {b['sources']}, answer \"{short(b['answer'], 40)}\""
    d = b["detail"]
    return f"detail \"{d}\"" if isinstance(d, str) else f"{d[0]['type']} at {'.'.join(map(str, d[0]['loc']))}"


outs["t32"] = S.table("㉜ TestClient로 POST /agent/ask", ["요청", "HTTP", "응답 요약"], [[k, f"<b>{c}</b>", api_sum(c, b)] for k, (c, b) in api.items()],
                      ["회원은 요청 본문이 아니라 인증 단계(여기서는 X-Member-Token 헤더를 토큰표에서 찾는 예시)에서 정해요. 로그인이 없으면 Agent를 부르기도 전에 401이에요.",
                       "본문에 member_id를 넣어 남의 예약을 보려 하면, 요청 모양에 없는 칸이라 422로 막혀요(extra=\"forbid\"). 신원을 본문에서 받지 않는 게 핵심이에요.",
                       "DB가 멈추면 502예요. 반복 상한에 걸리거나 출처 확인을 못 넘긴 답도 502로 막아서, 확인되지 않은 답이 사용자에게 가지 않게 해요."])

flow = [[25, "RAG 도구", "@tool rag_search", "설명과 입력 모양"], [26, "DB 도구", "읽기 전용, 본인 것만", "ToolRuntime"], [27, "한 바퀴", "요청 → 실행 → 결과", "tool_calls, ToolMessage"],
        [28, "create_agent", "루프를 맡기기", "도구 0, 1, 2개"], [29, "안전장치", "인수, 권한, 반복, 장애", "상한 4번"], [30, "결과 확인", "출처 대조, 보안 기록", "지어낸 출처 차단"],
        [31, "회귀 테스트", "시나리오 7개", "도구 선택까지"], [32, "API", "인증된 신원", "401, 422, 502"]]
S.save(ROOT / "topics" / "backend" / "agent-tools-data.js", {"code": code, "names": S.names(code), "range": [25, 32], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>지금까지의 RAG는 순서가 정해져 있었어요. 질문이 오면 <b>늘</b> 검색하고, 근거를 고르고, 답해요. 그런데 \"스터디룸 철회 규정이랑 <b>내</b> 101번 예약 상태\"를 물으면, 규정은 문서에 있고 내 예약은 DB에 있어요. 질문마다 필요한 자료가 다르면 \"무엇을 조회할지\"부터 정해야 해요.</p>
{K.table(["방식", "다음 단계를 누가 정하나", "맞는 때"], [["고정 RAG", "개발자가 검색 → 답 순서를 고정", "늘 같은 종류의 질문 (규정 문의)"], ["규칙 Workflow", "if/else 같은 정해 둔 조건", "절차가 분명하고 안정성이 가장 중요할 때"],
                                                ["<b>Tool Calling Agent</b>", "모델이 도구를 부를지, 무엇을 부를지, 언제 답할지 제안", "질문마다 필요한 자료와 순서가 다를 때"]])}
<p><b>AI Agent</b>는 모델이 상황과 쓸 수 있는 도구 목록을 보고 다음 행동(도구 호출 또는 답)을 고르고, 실행 환경이 그 행동을 실제로 해서 결과를 다시 모델에게 주는 구조예요. 역할이 셋으로 나뉘어요.</p>
{K.table(["누가", "하는 일", "하지 않는 일"], [["모델", "도구 이름과 인수를 <b>제안</b>, 결과를 읽고 답 쓰기", "DB 접속, 실제 실행, 권한 판단"], ["도구 (Tool)", "검색, 조회 같은 실제 일을 하는 함수. 이름, 설명, 입력 모양, 결과 약속을 가져요", "다른 도구 부르기"],
                                    ["실행 환경 (Harness)", "도구 실행, 결과 전달, 반복 상한, 오류 처리", "무엇을 부를지 정하기"]])}
<p>이 쪽은 RAG 서비스, 검색 고치기, RAG 평가에서 키운 RAG를 <b>그대로</b> 도구 하나(rag_search)로 감싸고, PostgreSQL의 예약 표를 읽는 도구(lookup_reservation)를 더해, <code>create_agent</code>로 묶어요. 도구 실행, DB, RAG, 검사, API는 모두 실제로 돌리고, 모델의 판단(무엇을 부를지, 답 문장)만 가짜 서버가 정해 둔 규칙으로 해요. 그래서 이 쪽의 숫자는 \"모델이 똑똑한가\"가 아니라 <b>요청, 실행, 권한, 오류가 어떻게 흐르나</b>에 대한 거예요.</p>
"""
whole = """<p>도구 둘을 만들고(㉕㉖), Agent 루프를 손으로 한 바퀴 돌려 본 뒤(㉗), create_agent에 맡기고(㉘), 이상한 호출을 막고(㉙), 답을 서버에서 확인하고(㉚), 시나리오로 시험하고(㉛), API로 열어요(㉜). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(25, "RAG를 도구로: 이름, 설명, 입력 모양, 결과 약속", "이미 있는 RAG 서비스를 @tool 함수 하나로 감싸요. 모델은 이 도구의 이름, 설명, 입력 모양만 보고 부를지 정해요.",
           "<p>도구는 그냥 함수 이름 목록이 아니라 <b>약속</b>이에요. 모델에게는 이름, 설명, 입력 모양이 보이고, 서버에게는 결과 모양이 약속이에요.</p>"
           + K.table(["약속", "이 도구에서", "왜"], [["이름", "rag_search", "모델이 고르는 이름"], ["설명", "docstring", "언제 쓰고 무엇은 못 하는지. 모델의 선택을 좌우해요"], ["입력 모양", "query (필수), category (넷 중 하나)", "이상한 값이 서비스에 닿기 전에 걸러요"],
                                              ["결과 약속", "answered / insufficient_context, 그리고 오류는 그대로", "정보 부족과 고장을 섞지 않아요"]])
           + K.out("s25")
           + "<p>RAG를 도구로 만드는 방법은 둘이에요. 이 쪽처럼 RAG가 답까지 써서 돌려주는 방식(Answer Tool)은 이미 있는 서비스를 그대로 쓰기 쉬워요. 대신 Agent 모델이 그 답을 다시 쓰면 LLM을 두 번 부르게 돼요. 근거와 출처만 돌려주고 답은 Agent가 쓰게 하는 방식(Evidence Tool)도 있어요. 시간, 비용, 출처 관리 중 무엇이 중요한지로 골라요.</p>"
           + K.code_table([("@tool", "함수를 LangChain 도구로. 이름은 함수 이름, 설명은 docstring, 입력 모양은 타입 힌트에서"), ("rag_search.tool_call_schema.model_json_schema()", "모델에게 보이는 입력 모양"),
                           ("rag_search.invoke({\"query\": ..., \"category\": ...})", "모델 없이 도구만 직접 실행")]), "s25"),
    K.step(26, "DB를 도구로: 읽기 전용, 본인 것만", "예약 상태처럼 계속 바뀌는 정보는 DB에서 읽어요. 누구의 것을 읽을지는 모델이 아니라 서버가 정해요.",
           "<p>규정은 문서에 있지만, \"내 예약이 지금 어떤 상태인가\"는 DB의 최신 값이에요. RAG가 대신할 수 없어요. 그래서 두 번째 도구는 PostgreSQL의 예약 표를 읽어요.</p>"
           "<p>DB 도구에서 지킬 것은 셋이에요. ① 모델에게 SQL을 쓰게 하지 않고, 개발자가 쓴 SQL에 <b>값만</b> <code>%s</code>로 넘겨요. ② 누구의 예약인지(member_id)는 로그인한 서버 값으로 정하고, WHERE에 꼭 넣어요. ③ 읽기만 하는 연결과 계정으로 열어요.</p>"
           + K.out("t26")
           + "<p>LangChain에서는 <code>ToolRuntime</code>으로 ②를 해요. 도구 함수에 <code>runtime: ToolRuntime[Member]</code> 인수를 두면, Agent가 실행할 때 서버가 넘긴 값(<code>context=Member(...)</code>)이 들어오고, 모델에게 보이는 입력 모양에서는 빠져요.</p>"
           + K.out("s26")
           + K.code_table([("conn.execute(SQL, (reservation_id, member_id))", "값을 %s 자리로 넘기기 (SQL 문장에 이어 붙이지 않아요)"), ("conn.read_only = True", "이 연결의 트랜잭션을 읽기 전용으로"),
                           ("ToolRuntime[Member]", "서버가 채워 주는 실행 정보. runtime.context.member_id"), ("@tool(args_schema=LookupArgs)", "입력 모양을 Pydantic 모델로 (범위, strict)")]), "s26"),
    K.step(27, "Tool Calling 한 바퀴를 손으로: 요청 → 실행 → 결과 → 답", "모델이 도구 호출을 요청하고, 서버가 실행해 결과를 붙여 다시 물으면, 모델이 답을 써요. 이 한 바퀴가 Agent 루프예요.",
           "<p>메시지가 네 종류예요.</p>"
           + K.table(["메시지", "누가 만드나", "담는 것"], [["SystemMessage", "개발자", "역할, 규칙"], ["HumanMessage", "사용자", "질문"], ["AIMessage", "모델", "답 글 또는 <code>tool_calls</code> (도구 이름, 인수, id)"],
                                                 ["ToolMessage", "실행 환경", "도구 결과와 <code>tool_call_id</code> (어느 요청의 결과인지)"]])
           + K.out("s27")
           + K.code_table([("model.bind_tools([...])", "도구 목록을 요청에 싣는 모델 (실행은 안 해요)"), ("ai_message.tool_calls", "[{name, args, id}, ...]"), ("ToolMessage(content, tool_call_id, name)", "도구 결과를 요청 id와 짝지어 돌려주기")]), "s27"),
    K.step(28, "create_agent로 묶기", "모델, 도구, 규칙을 넘기면 ㉗의 루프(요청, 실행, 결과 전달, 끝날 때까지 반복)를 대신 돌려줘요.",
           "<p><code>create_agent(model, tools, system_prompt, context_schema, middleware)</code>가 Agent를 만들어요. <code>agent.invoke({\"messages\": [...]}, context=Member(...))</code>로 부르면 모델이 도구 호출을 멈추고 답할 때까지 루프를 돌아요. 권한 검사나 결과 확인까지 해 주지는 않아서, 그건 여전히 도구 안과 서버(㉙㉚)에서 해요.</p>"
           + K.out("t28")
           + K.code_table([("create_agent(model=..., tools=[...], system_prompt=..., context_schema=Member, middleware=[...])", "Agent 만들기"), ("agent.invoke({\"messages\": [...]}, context=Member(\"m-001\"))", "질문 하나 실행. 결과는 {\"messages\": [...]}"),
                           ("final_text(messages)", "맨 뒤의 도구 호출 없는 AIMessage 글")]), "s28"),
    K.step(29, "안전장치: 잘못된 인수, 남의 것, 없는 도구, 반복, DB 장애", "모델이 이상한 호출을 해도 서버에서 걸러지고, 끝없이 돌지 않고, 고장은 고장으로 올라오는지 봐요.",
           "<p>도구 실패는 하나로 뭉치면 안 돼요. 어디서 실패했는지에 따라 다시 시도할지, 멈출지, 사용자에게 뭐라고 할지가 달라요.</p>"
           + K.out("t29")
           + K.code_table([("ToolCallLimitMiddleware(run_limit=4, exit_behavior=\"error\")", "한 번 실행에 도구 호출 4번까지"), ("ToolCallLimitExceededError", "상한을 넘었을 때의 예외")]), "s29"),
    K.step(30, "서버에서 결과 확인하기: 출처와 기록", "Agent의 마지막 답을 그대로 믿지 않고, 이번 실행에서 실제로 확인된 출처와 대조하고, 보안 기록을 남겨요.",
           K.out("s30")
           + K.code_table([("validate_agent_answer(answer, ledger)", "답의 [출처 n]이 이번 실행에서 rag_search가 돌려준 번호 안에 있나"), ("audit_record(request_id, member_id, ledger)", "누가(해시), 어떤 도구를, 결과가 어땠나"),
                           ("hashlib.sha256(...).hexdigest()[:10]", "회원 ID를 그대로 남기지 않기")]), "s30"),
    K.step(31, "회귀 테스트: 바꿀 때마다 같은 시나리오로", "정상, 복합, 남의 것, 자료에 없는 질문, 도구가 필요 없는 질문을 시나리오로 정해 두고, 도구 선택과 답과 출처를 함께 확인해요.",
           K.out("t31")
           + "<p>실제 모델로 돌릴 때는 같은 시나리오를 여러 번 돌려 도구 선택이 흔들리는지도 봐요. 이 표에 DB 장애, 반복 상한(㉙)도 함께 넣어 두면, 오류 처리가 바뀌었을 때도 바로 알 수 있어요.</p>"
           + K.code_table([("SCENARIOS = [(이름, 질문, 회원, 호출돼야 할 도구, 통과 조건), ...]", "시나리오 표"), ("{c[\"name\"] for m in msgs if isinstance(m, AIMessage) for c in m.tool_calls}", "이번 실행에서 부른 도구들")]), "s31"),
    K.step(32, "API로 열기", "FastAPI가 로그인한 회원을 정하고, Agent를 부르고, 확인을 못 넘긴 답은 내보내지 않아요.",
           "<p>Agent를 API로 열 때의 순서예요. ① 인증으로 회원을 정해요(본문에서 받지 않아요). ② 질문 모양을 검사해요. ③ <code>agent.invoke(..., context=Member(회원))</code>로 실행해요. ④ 답의 출처를 확인해요(㉚). ⑤ 실패를 HTTP 코드로 나눠요.</p>"
           + K.out("t32")
           + "<p>여기서 다음으로 이어지는 주제는 여러 단계와 상태를 그래프로 다루는 LangGraph, Prompt Injection 같은 보안, 도구를 표준 방식으로 붙이는 MCP예요.</p>"
           + K.code_table([("Depends(current_member)", "인증 함수를 거쳐 회원을 받기"), ("Header(default=None)", "요청 헤더 값 읽기"), ("model_config = ConfigDict(extra=\"forbid\")", "모양에 없는 칸(member_id)이 오면 422")]), "s32"),
]
practice = [("도구 설명이 곧 라우팅", "모델은 이름, 설명, 입력 모양만 보고 고르니까, 설명에 \"무엇은 못 한다\"까지 적어요. 비슷한 도구가 둘이면 설명으로 경계를 분명히 해요."),
            ("신원은 서버에서", "user_id, member_id 같은 값은 도구 인수로 받지 않고 ToolRuntime이나 서버 변수로 넣어요. 모델 인수로 받으면 프롬프트 한 줄로 남의 정보를 볼 수 있어요."),
            ("DB 도구는 읽기부터", "처음엔 SELECT만 허락된 계정, 정해 둔 SQL, 값 바인딩으로 시작해요. 바꾸는 도구(예약 취소 같은)는 사용자 확인 단계를 따로 둬요."),
            ("반복 상한은 필수", "입력 오류나 없는 도구는 모델에게 돌아가 다시 시도하게 되니까, 실행 전체와 도구별 호출 수에 상한을 걸어요. 시간과 비용 상한도 따로 둬요."),
            ("정보 부족과 고장을 나누기", "\"자료에 없음\", \"남의 것이라 안 보임\", \"DB 장애\", \"인증 실패\"는 다른 상태예요. 고장을 '없음'으로 바꾸면 사용자에게 틀린 답을 주게 돼요."),
            ("답을 그대로 믿지 않기", "Agent의 마지막 문장은 모델이 쓴 글이에요. 출처, 부른 도구, 도구 결과를 서버의 실행 기록과 대조한 뒤 내보내요."),
            ("기록은 최소한으로", "실행 기록에는 단계와 시간, 보안 기록에는 누가 어떤 도구를 썼는지만 남기고, 토큰, 키, 개인정보, 문서 원문은 남기지 않아요.")]
libs = [("㉕", "<code>@tool</code>, <code>tool_call_schema</code>, <code>invoke</code>", "도구 만들기 (langchain)"), ("㉖", "<code>psycopg.connect</code>, <code>read_only</code>, <code>ToolRuntime</code>, <code>Field</code>", "읽기 전용 DB 도구"),
        ("㉗", "<code>bind_tools</code>, <code>tool_calls</code>, <code>ToolMessage</code>", "Tool Calling 메시지"), ("㉘", "<code>create_agent</code>, <code>agent.invoke</code>", "Agent (langchain, langgraph 위에서 동작)"),
        ("㉙", "<code>ToolCallLimitMiddleware</code>, <code>ToolCallLimitExceededError</code>", "반복 상한"), ("㉚", "<code>hashlib.sha256</code>, <code>re.findall</code>", "출처 대조, 보안 기록"),
        ("㉜", "<code>Depends</code>, <code>Header</code>, <code>HTTPException</code>, <code>TestClient</code>", "API")]
K.build(ROOT / "topics" / "backend" / "agent-tools.html", title="AI Agent와 도구", h1="AI Agent와 도구: RAG와 DB를 골라 부르게 하기", track="백엔드", crumb="백엔드 &gt; Agent",
        one_line="<b>Agent</b>는 모델이 질문을 보고 <b>어떤 도구를 부를지</b> 제안하고, 서버가 그 도구를 실제로 실행해 결과를 돌려주면, 모델이 다시 판단해 답하는 구조예요. 모델은 <b>제안만</b> 하고, 실행, 권한, 반복 상한, 출처 확인은 <b>서버</b>가 해요. 이 쪽은 앞에서 만든 RAG를 도구 하나로, 예약 DB 조회를 또 하나의 도구로 만들어 묶어요.",
        why=why, whole=whole, steps=steps, code_file="examples/rag/rag_service.py", practice=practice, libs=libs,
        libs_note="<code>normalize_rag</code>, <code>rag_search</code>, <code>read_reservation</code>, <code>lookup_reservation</code>, <code>fake_agent_reply</code>, <code>fake_agent_server</code>, <code>dispatch</code>, <code>run_agent</code>, <code>final_text</code>, <code>call_with</code>, <code>validate_agent_answer</code>, <code>audit_record</code>, <code>current_member</code>, <code>create_agent_app</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rag-service.html\">RAG 서비스</a>", "rag_search 안쪽의 검색, 근거, 출처 검사"), ("<a href=\"rag-eval.html\">RAG 평가</a>", "도구가 돌려주는 답의 품질 재기"),
                   ("<a href=\"postgres.html\">PostgreSQL 기초</a>", "표, 조회, 트랜잭션"), ("<a href=\"llm-client.html\">LLM API 클라이언트</a>", "모델 호출, 구조화 출력, 실패")],
        sources=["다루는 범위: Day29 (2026-11-26) AI Agent Fundamentals와 Tool Calling. 고정 RAG, Workflow, Agent 비교, Agent 루프, Tool Calling 메시지(tool_calls, ToolMessage, bind_tools), @tool과 입력 모양, RAG를 도구로(Answer Tool과 Evidence Tool), PostgreSQL 읽기 전용 도구와 ToolRuntime, create_agent, 신뢰성(입력, 권한, 반복 상한, 실패 구분, 결과 확인, trace와 audit), 회귀 테스트, FastAPI 연결. 수업의 과제 제출 상태 대신 이 노트의 스터디룸 예약으로 같은 흐름을 다시 만들었어요",
                 "예제: <code>examples/rag/rag_service.py</code>의 ㉕~㉜. 예약 표는 예시 데이터(가상의 회원 둘)이고, 노트 전용 DB에 만들었다가 빌드가 끝나면 지워요",
                 f"결과 값: <code>tools/pages/agent_tools.py</code>가 단계별로 실제 실행한 결과 ({ENV}). Agent의 도구 선택과 답 문장은 가짜 서버의 규칙이에요",
                 "공식 문서: <a href=\"https://docs.langchain.com/oss/python/langchain/agents\" target=\"_blank\">LangChain Agents</a>, <a href=\"https://docs.langchain.com/oss/python/langchain/tools\" target=\"_blank\">Tools</a>, <a href=\"https://docs.langchain.com/oss/python/langchain/middleware\" target=\"_blank\">Middleware</a>, <a href=\"https://www.psycopg.org/psycopg3/docs/basic/params.html\" target=\"_blank\">psycopg: Passing parameters</a>"])
print("ok")
