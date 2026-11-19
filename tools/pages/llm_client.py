"""주제 페이지 '백엔드 > LLM 활용 > LLM API 클라이언트'를 만든다.

examples/llm/client.py 를 단계별로 실제 실행한다. OpenAI 서버에는 요청을 보내지 않고(키, 비용 없음),
openai SDK 에 httpx.MockTransport(가짜 서버)를 붙여 SDK 가 실제로 만드는 요청, 예외, 재시도 횟수, 스트림 해석을 관찰한다.
응답 글과 토큰 수는 기존 노트에 실린 실제 호출 기록 값이다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/llm_client.py
"""
import json
import os
import sys
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import httpx  # noqa: E402
import openai  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "llm"
code = S.parse((EX / "client.py").read_text(encoding="utf-8"))
os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key"
os.environ["OPENAI_MODEL"] = "gpt-5.4-mini"
TEXT = "API는 서로 다른 프로그램들이 정해진 규칙으로 데이터를 주고받고 기능을 이용하게 해 주는 “중간 다리”입니다."
USAGE = {"input_tokens": 31, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 36,
         "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 67}
PARSED = {"term": "API", "one_line": "프로그램끼리 정해진 규칙으로 기능과 데이터를 주고받는 창구", "level": "easy"}
outs, sent = {}, []


def body(text):
    return {"id": "resp_demo01", "object": "response", "created_at": 0, "model": os.environ["OPENAI_MODEL"], "status": "completed",
            "output": [{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}],
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": USAGE, "incomplete_details": None}


class Stream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


def sse(text):
    parts = ["API는 ", "서로 다른 프로그램들이 ", "정해진 규칙으로 ", "데이터를 주고받고 ", "기능을 이용하게 해 주는 ", "“중간 다리”입니다."]
    assert "".join(parts) == text
    part = {"type": "output_text", "text": "", "annotations": []}
    item = {"type": "message", "id": "msg_1", "status": "in_progress", "role": "assistant", "content": []}
    ev = [{"type": "response.created", "response": {**body(""), "status": "in_progress", "output": []}},
          {"type": "response.output_item.added", "output_index": 0, "item": item},
          {"type": "response.content_part.added", "item_id": "msg_1", "output_index": 0, "content_index": 0, "part": part}]
    ev += [{"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": p, "logprobs": []} for p in parts]
    done_part = {**part, "text": text}
    ev += [{"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0, "content_index": 0, "text": text, "logprobs": []},
           {"type": "response.content_part.done", "item_id": "msg_1", "output_index": 0, "content_index": 0, "part": done_part},
           {"type": "response.output_item.done", "output_index": 0, "item": {**item, "status": "completed", "content": [done_part]}},
           {"type": "response.completed", "response": body(text)}]
    return [f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': i}, ensure_ascii=False)}\n\n".encode() for i, e in enumerate(ev)]


def ok_handler(req):
    b = json.loads(req.content)
    sent.append({"method": req.method, "url": str(req.url), "body": b, "auth": req.headers.get("authorization", "")[:10] + "…"})
    if b.get("stream"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream(sse(TEXT)))
    if b.get("text", {}).get("format", {}).get("type") == "json_schema":
        return httpx.Response(200, json=body(json.dumps(PARSED, ensure_ascii=False)))
    return httpx.Response(200, json=body(TEXT))


def fake(handler):
    return {"FAKE_HTTP": httpx.Client(transport=httpx.MockTransport(handler))}


PATCH = [("OpenAI(timeout=30.0, max_retries=2)", "OpenAI(timeout=30.0, max_retries=2, http_client=FAKE_HTTP)")]
run = lambda k, handler=ok_handler: S.run(code, k, S.workdir(EX), PATCH, filename="client.py", env=fake(handler))  # noqa: E731

# ---- ① ----
ns = run(1)
cl = ns["client"]
outs["s1"] = S.box("① 실행 결과", [(f"client.base_url → {cl.base_url}", 1), f"client.timeout → {cl.timeout}", (f"client.max_retries → {cl.max_retries}", 2), f"키가 설정됐나 → {bool(cl.api_key)}"],
                   ["따로 정하지 않으면 OpenAI 서버 주소를 써요.", "실패하면 SDK가 <b>알아서</b> 최대 2번 더 보내요. ④에서 실제 횟수를 봐요."])

# ---- ② ----
sent.clear()
ns = run(2)
s0 = sent[0]
r = ns["response"]
outs["s2"] = S.box("SDK가 실제로 만든 요청과 받은 응답", [(f"{s0['method']} {s0['url']}", 1), f"Authorization: {s0['auth']}", (f"본문 → {json.dumps(s0['body'], ensure_ascii=False)}", 2), "",
                    (f"response.output_text → {ns['answer']}", 3), f"response.usage → 입력 {r.usage.input_tokens}토큰, 출력 {r.usage.output_tokens}토큰"],
                   ["Responses API의 주소예요. 모델 이름은 주소가 아니라 본문에 들어가요.", "<code>create()</code>에 준 인자가 그대로 JSON 본문이 됐어요. 키는 본문이 아니라 머리글(<code>Authorization</code>)로 가요.",
                    "응답 안의 여러 조각에서 글만 모아 주는 편의 속성이에요."])

# ---- ③ ----
sent.clear()
ns = run(3)
fmt = [s for s in sent if s["body"].get("text")][0]["body"]["text"]["format"]
outs["s3"] = S.box("responses.parse(text_format=Explain)", [(f"요청 본문의 text.format → type={fmt['type']}, name={fmt['name']}, strict={fmt['strict']}", 1),
                    (f"schema.required → {fmt['schema']['required']}", 2), f"schema.properties.level → {json.dumps(fmt['schema']['properties']['level'], ensure_ascii=False)}", "",
                    (f"output_parsed → {ns['parsed']!r}", 3)],
                   ["SDK가 Pydantic 클래스를 JSON 스키마로 바꿔 요청에 실었어요. <code>strict</code>라서 모델은 이 모양으로만 답해요.", "필드 셋 모두 필수예요.",
                    "돌아온 JSON 글을 SDK가 <code>Explain</code> 객체로 바꿔 줬어요. 내용이 맞는지(출처 등)는 여전히 따로 검사해요 (<a href=\"prompt-contract.html\">Prompt와 출력 계약</a>)."])

# ---- ④ ----
cases = [("키가 틀림 (401)", 401, "invalid_api_key"), ("너무 자주 보냄 (429)", 429, "rate_limit_exceeded"), ("할당량 다 씀 (429)", 429, "insufficient_quota"),
         ("서버 오류 (500)", 500, "server_error"), ("응답이 늦음 (시간 초과)", None, None)]
rows = []
ns = run(4)
for label, st, cd in cases:
    cnt = [0]

    def h(req, st=st, cd=cd, cnt=cnt):
        cnt[0] += 1
        if st is None:
            raise httpx.ReadTimeout("모의 시간 초과", request=req)
        return httpx.Response(st, json={"error": {"message": "모의 오류", "type": "t", "code": cd, "param": None}}, headers={"retry-after-ms": "5"})
    ns["client"] = openai.OpenAI(timeout=30.0, max_retries=2, http_client=httpx.Client(transport=httpx.MockTransport(h)))
    kind, val = ns["ask"]("질문")
    rows.append([E(label), f"<code>{kind}, {E(str(val))}</code>", f"<b>{cnt[0]}번</b>"])
outs["s4"] = S.table("ask()를 상황별 가짜 서버에 보내면 (max_retries=2)", ["상황", "ask()가 돌려준 값", "SDK가 실제로 보낸 횟수"], rows,
                     ["401은 다시 보내도 같아서 SDK도 한 번만 보내요. 429, 500, 시간 초과는 SDK가 <b>앱 모르게 3번</b>까지 보내요.",
                      "두 429는 예외 종류(<code>RateLimitError</code>)가 같아요. \"너무 자주\"는 기다리면 풀리지만 \"할당량 다 씀\"은 결제를 확인해야 해서, <code>e.code</code>로 갈라야 해요. 그런데 SDK는 할당량 문제도 3번 보냈어요. 비용과 시간이 아깝다면 <code>max_retries</code>를 줄이고 직접 판단해요."])

# ---- ⑤ ----
sent.clear()
ns = run(5)
outs["s5"] = S.box("responses.stream()으로 받은 조각", [(f"pieces → {ns['pieces']}", 1), f"\"\".join(pieces) → {''.join(ns['pieces'])}", (f"final.output_text == 이어 붙인 글 → {ns['final'].output_text == ''.join(ns['pieces'])}", 2)],
                   [f"답이 {len(ns['pieces'])}조각으로 나눠 도착했어요. 조각이 올 때마다 화면에 붙이면, 사용자는 다 만들어질 때까지 기다리지 않아도 돼요.", "다 받은 뒤에는 전체 응답(사용량 포함)도 얻을 수 있어요."])

# ---- ⑥ ----
ns = run(6)
outs["s6"] = S.box("사용량과 비용 (예시 단가)", [f"입력 {ns['usage'].input_tokens}토큰 × $0.75 / 100만", f"출력 {ns['usage'].output_tokens}토큰 × $4.50 / 100만", (f"cost → ${ns['cost']}", 1)],
                   ["한 번 호출에 약 0.0002달러예요. 적어 보여도 하루 만 번이면 2달러, 재시도가 붙으면 더 늘어요. 그래서 호출마다 토큰과 비용을 기록해요. 단가는 예시예요. 실제 단가는 모델마다 다르고 바뀌니 공식 가격표를 봐요.<br><code>Decimal</code>은 돈 계산에서 소수 오차가 쌓이지 않게 쓰는 정확한 소수예요."])

flow = [[1, "클라이언트", "OpenAI()", "키는 환경 변수"], [2, "보내기", "responses.create", "output_text"], [3, "모양 정하기", "responses.parse", "Pydantic → 스키마"],
        [4, "실패", "예외 종류별", "자동 재시도 3번"], [5, "스트리밍", "조각으로 받기", "delta 이벤트"], [6, "비용", "usage", "토큰 × 단가"]]
S.save(ROOT / "topics" / "backend" / "llm-client-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                             "env": f"Python {sys.version.split()[0]}, openai {openai.__version__}, httpx {httpx.__version__}. 실제 OpenAI 서버 대신 httpx.MockTransport 가짜 서버"})

why = """
<p>LLM은 보통 내 컴퓨터가 아니라 회사의 서버에서 돌아가요. 우리 프로그램은 인터넷으로 <b>요청</b>을 보내고 <b>응답</b>을 받아요. 이 요청을 직접 HTTP로 짜도 되지만, 키를 붙이고, 응답 JSON을 풀고, 실패하면 다시 보내는 일을 매번 하게 돼요. 그래서 회사가 주는 <b>SDK</b>(그 API를 쓰기 쉽게 감싼 라이브러리)를 써요. 여기서는 OpenAI의 파이썬 SDK와 <b>Responses API</b>를 써요.</p>
<p>그런데 SDK를 쓰더라도 앱이 직접 정해야 하는 게 있어요.</p>
""" + K.table(["무엇을", "왜"], [["키를 어디에 두나", "키가 코드에 들어가면 공유할 때 같이 새요. 남이 쓰면 비용이 내게 청구돼요."],
                                 ["답을 어떤 모양으로 받나", "글로 받으면 프로그램이 쓰기 어려워요."],
                                 ["실패하면 어떻게 하나", "키 문제, 너무 잦은 요청, 할당량, 서버 오류, 시간 초과는 각각 대처가 달라요."],
                                 ["얼마나 썼나", "호출마다 토큰 수만큼 돈이 들어요."]]) + """
<p><b>그래서 목표는</b> 안전하게 클라이언트를 만들고, 답을 정해진 모양으로 받고, 실패를 종류별로 다루고, 사용량을 기록하는 거예요.</p>
"""
whole = """
<p>클라이언트 하나를 만들어 두고, 보내기, 모양 정해 받기, 실패 다루기, 조각으로 받기, 비용 기록을 차례로 붙여요. 상자를 누르면 그 단계로 가요.</p>
<div class="flow" id="flow"></div>
<p><b>이 페이지는 실제 OpenAI 서버에 요청을 보내지 않아요</b> (키와 비용 없음). SDK에 가짜 서버를 붙여서, SDK가 <b>실제로 만드는 요청</b>, <b>실제로 던지는 예외</b>, <b>실제로 보내는 횟수</b>를 관찰했어요. 응답 글과 토큰 수(입력 31, 출력 36)는 실제 호출 기록에서 가져온 값이에요.</p>
"""
steps = [
    K.step(1, "클라이언트 만들기", "키와 모델 이름은 환경 변수에서 읽고, 시간 제한과 재시도 횟수를 정해요.",
           K.code_table([("os.environ[\"OPENAI_MODEL\"]", "모델 이름도 환경 변수로. 코드를 안 고치고 모델을 바꿀 수 있어요."),
                         ("OpenAI(timeout=30.0, max_retries=2)", "키는 인자로 주지 않으면 환경 변수 <code>OPENAI_API_KEY</code>에서 읽어요. 30초 안에 답이 없으면 실패로 보고, 실패하면 2번까지 다시 보내요.")]), "s1"),
    K.step(2, "질문 보내고 답 읽기", "<code>responses.create()</code>로 보내고 <code>output_text</code>로 답 글을 읽어요.",
           K.code_table([("model=MODEL", "어느 모델에게 물을지"), ("instructions=...", "모델이 지킬 지시 (역할, 말투 등)"), ("input=question", "사용자 질문"),
                         ("response.output_text", "응답 안의 글 조각을 모은 것"), ("response.usage", "이번 호출에 쓴 토큰 수. 비용 계산에 써요 (⑥).")]), "s2"),
    K.step(3, "정해진 모양으로 받기 (구조화 출력)", "Pydantic 클래스를 주면, 모델이 그 JSON 모양으로만 답하고 SDK가 객체로 바꿔 줘요.",
           K.code_table([("class Explain(BaseModel)", "받고 싶은 모양"), ("client.responses.parse(..., text_format=Explain)", "SDK가 클래스를 JSON 스키마로 바꿔 요청에 싣고, 답을 다시 객체로 바꿔요."),
                         (".output_parsed", "바뀐 <code>Explain</code> 객체")]), "s3"),
    K.step(4, "실패를 종류별로 다루기", "SDK가 던지는 예외 종류로 \"멈출지, 나중에 다시 할지\"를 정해요.",
           K.code_table([("openai.AuthenticationError", "401. 키가 틀렸거나 없음. 다시 보내도 같아요."), ("openai.RateLimitError", "429. 너무 자주 보냈거나(<code>rate_limit_exceeded</code>) 할당량을 다 썼거나(<code>insufficient_quota</code>)"),
                         ("e.code", "같은 예외 안에서 원인을 가르는 값"), ("openai.APITimeoutError", "정한 시간 안에 답이 없음"), ("openai.APIStatusError", "그 밖의 HTTP 오류. 위의 예외들도 이걸 물려받아서, 맨 마지막에 둬요.")]), "s4"),
    K.step(5, "나오는 대로 받기 (스트리밍)", "답이 다 만들어지길 기다리지 않고, 만들어지는 조각을 바로바로 받아요.",
           "<p>긴 답은 다 만들어지는 데 몇 초가 걸려요. 스트리밍은 서버가 조각을 보내는 대로 받아서, 사용자가 글이 써지는 걸 바로 볼 수 있게 해요.</p>"
           + K.code_table([("with client.responses.stream(...) as stream", "스트리밍 연결을 열고, 블록이 끝나면 닫아요."), ("event.type == \"response.output_text.delta\"", "글 조각이 왔다는 이벤트. 다른 이벤트(시작, 끝 등)도 와요."),
                           ("event.delta", "새로 온 글 조각"), ("stream.get_final_response()", "다 받은 뒤의 전체 응답")]), "s5"),
    K.step(6, "사용량과 비용 기록하기", "응답의 토큰 수에 단가를 곱해 이번 호출의 비용을 계산해요.",
           K.code_table([("usage.input_tokens, usage.output_tokens", "보낸 토큰과 받은 토큰. 보통 받는 쪽이 더 비싸요."), ("Decimal(\"0.75\")", "정확한 소수. <code>0.1 + 0.2</code> 같은 실수 오차가 없어요.")]), "s6"),
]
practice = [
    ("키를 저장한다", "<code>.env</code> 파일에 두고 <code>python-dotenv</code>로 읽거나, 운영 환경의 비밀 저장소를 써요. <code>.env</code>는 꼭 <code>.gitignore</code>에 넣어요."),
    ("같은 질문이 반복된다", "질문, 모델, 지시가 같으면 답을 저장해 두고 다시 쓰면(캐시) 비용이 줄어요. 캐시 열쇠에 Prompt 판 번호도 넣어서, 틀이 바뀌면 새로 묻게 해요."),
    ("재시도를 직접 정한다", "<code>max_retries=0</code>으로 SDK 재시도를 끄고, <code>e.code</code>와 응답의 <code>retry-after</code> 시간을 보고 직접 기다렸다 보내요. 전체 시간 한도를 정해 두지 않으면 사용자가 무한정 기다려요."),
    ("모델을 고른다", "간단한 질문은 싸고 빠른 모델, 어려운 질문은 비싼 모델로 보내는 식으로 나눌 수 있어요. 모델 이름을 환경 변수로 둔 이유예요."),
]
libs = [("①", "<code>OpenAI(timeout=, max_retries=)</code>", "클라이언트 (openai)"), ("②", "<code>client.responses.create()</code>, <code>output_text</code>, <code>usage</code>", "보내고 받기"),
        ("③", "<code>client.responses.parse(text_format=)</code>, <code>output_parsed</code>", "구조화 출력 (openai + pydantic)"),
        ("④", "<code>AuthenticationError</code>, <code>RateLimitError</code>, <code>APITimeoutError</code>, <code>APIStatusError</code>", "예외 종류"),
        ("⑤", "<code>client.responses.stream()</code>, <code>event.delta</code>, <code>get_final_response()</code>", "스트리밍"), ("⑥", "<code>decimal.Decimal</code>", "정확한 비용 계산 (표준 라이브러리)")]
K.build(ROOT / "topics" / "backend" / "llm-client.html", title="LLM API 클라이언트", h1="LLM API 클라이언트: 모델을 안전하게 부르기", track="백엔드",
        crumb="백엔드 &gt; LLM 활용",
        one_line="LLM은 원격 서버에 있어서, SDK로 요청을 보내고 응답을 받아요. 앱은 <b>키를 안전하게 두고</b>, <b>답을 정해진 모양으로 받고</b>, <b>실패를 종류별로 다루고</b>, <b>비용을 기록</b>해야 해요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm/client.py", practice=practice, libs=libs,
        libs_note="<code>ask</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"prompt-contract.html\">Prompt와 출력 계약</a>", "받은 답을 검사하기"), ("<a href=\"indexing.html\">인덱싱</a>", "질문에 맞는 근거 자료 찾기")],
        sources=["다루는 범위: Day22 (2026-11-17) LLM API 클라이언트", "예제: <code>examples/llm/client.py</code>. 응답 글과 토큰 수는 기존 노트에 실린 실제 호출 기록",
                 "결과 값: <code>tools/pages/llm_client.py</code>가 가짜 서버를 붙여 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://platform.openai.com/docs/api-reference/responses\" target=\"_blank\">Responses API</a>, <a href=\"https://github.com/openai/openai-python\" target=\"_blank\">openai-python</a>, <a href=\"https://platform.openai.com/docs/guides/structured-outputs\" target=\"_blank\">Structured Outputs</a>, <a href=\"https://platform.openai.com/docs/guides/streaming-responses\" target=\"_blank\">Streaming</a>"])
print("ok")
