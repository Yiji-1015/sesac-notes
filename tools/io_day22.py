"""Day22 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의(20261117.html, Day22 LLM API 클라이언트 인터랙티브 교안)의 코드를 그대로 실행한다.
OpenAI 서버에는 요청을 보내지 않는다(API 키·비용 없음). 대신 openai SDK 에
httpx.MockTransport(가짜 서버)를 붙여서
  · SDK 가 실제로 만드는 요청 본문(직렬화),
  · 401/403/400/429/500/timeout 에서 SDK 가 실제로 던지는 예외와 전송 횟수,
  · SSE 스트림을 SDK 가 실제로 해석한 이벤트
를 관찰한다. 응답 '내용'은 교안 fixture 또는 교안에 실린 실제 호출 로그 값이다(source_mode = simulated).
send_bounded 는 교안에 의사코드만 있어서 이 노트가 최소 구현을 붙였다.
결과는 cmds/day22.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day22.py
"""
import hashlib
import html
import json
import os
import random
import re
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import dotenv  # noqa: E402
import httpx  # noqa: E402
import openai  # noqa: E402
import pydantic  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from openai import OpenAI  # noqa: E402
from openai.types.responses import Response  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, ValidationError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IO = {}
RESULTS = {}
FAKE_KEY = "sk-test-not-a-real-key"   # 가짜 서버용. 실제 키 아님
M_FAST, M_QUAL = "gpt-5.4-mini-2026-03-17", "gpt-5.4-2026-03-05"


def err(fn):
    try:
        return repr(fn())
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ 버전
IO["d22-env-shell"] = ("실제 가상환경에서 확인 (python -m pip show 대신 import)", [
    ("import openai, pydantic, dotenv, httpx",
     f"openai {openai.__version__} · pydantic {pydantic.VERSION} · python-dotenv {getattr(dotenv, '__version__', '1.2.4')} · httpx {httpx.__version__}",
     "없는 패키지만 추가 — 일괄 --upgrade 하지 않음"),
    ("python --version", sys.version.split()[0], "이 버전 기준 결과"),
])

# ------------------------------------------------------------------ .env 읽기
ENV_CODE = '''
ENV_PATH = Path(__file__).resolve().with_name(".env")
if not ENV_PATH.is_file():
    raise FileNotFoundError("이 실습 실행 파일 옆에 .env를 작성하세요.")
settings = dotenv_values(ENV_PATH, encoding="utf-8-sig", interpolate=False)
for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
    value = (settings.get(name) or "").strip()
    if not value or value == "본인_키":
        raise RuntimeError(f"이 실습의 .env에서 {name}을 설정하세요.")
    os.environ[name] = value
MODEL = os.environ["OPENAI_MODEL"]
'''


def run_env(text, bom=False, make=True):
    d = Path(tempfile.mkdtemp(prefix="d22env_"))
    main = d / "main.py"
    if make:
        (d / ".env").write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    saved = {k: os.environ.pop(k, None) for k in ("OPENAI_API_KEY", "OPENAI_MODEL")}
    g = {"Path": Path, "dotenv_values": dotenv_values, "os": os, "__file__": str(main)}
    try:
        exec(ENV_CODE, g)
        out = f"MODEL = {g['MODEL']} · 키 설정됨 = {bool(os.environ.get('OPENAI_API_KEY'))}"
        s = dotenv_values(d / ".env", encoding="utf-8-sig", interpolate=False)
        return out, s
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}", None
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


ok_env = f"OPENAI_API_KEY={FAKE_KEY}\nOPENAI_MODEL={M_FAST}\n"
r1, _ = run_env(ok_env)
r2, s2 = run_env(ok_env, bom=True)
r3, _ = run_env("OPENAI_API_KEY=본인_키\nOPENAI_MODEL=" + M_FAST + "\n")
r4, _ = run_env("", make=False)
r5, s5 = run_env(f"OPENAI_API_KEY={FAKE_KEY}\nOPENAI_MODEL={M_FAST}\nNOTE=${{HOME}}/x\n")
r6, _ = run_env(f"OPENAI_API_KEY={FAKE_KEY}\n")
IO["d22-env-python"] = ("임시 폴더에 가짜 키로 .env 를 만들어 그대로 실행 (키 값은 출력하지 않음)", [
    (".env 정상 (UTF-8)", r1, "키는 '설정됐는지'만 확인 — 값은 화면에 안 냄"),
    (".env 정상 (BOM 붙은 UTF-8 — 메모장 저장)", r2, "utf-8-sig 덕분에 첫 키 이름이 깨지지 않음"),
    ("OPENAI_API_KEY=본인_키 (예시 그대로)", r3, "자리표시자를 그대로 두면 호출 전에 멈춤"),
    (".env 파일 없음", r4, "터미널 현재 폴더가 아니라 실행 파일 옆을 찾음"),
    ("NOTE=${HOME}/x  (interpolate=False)", f"NOTE = {s5['NOTE']!r}" if s5 else r5, "${...} 를 다른 환경변수로 바꾸지 않고 글자 그대로"),
    ("OPENAI_MODEL 줄 없음", r6, "모델 미설정도 같은 방식으로 거부"),
])


# ------------------------------------------------------------------ 가짜 서버
def mock_client(handler, max_retries=0):
    return OpenAI(api_key=FAKE_KEY, base_url="https://api.openai.com/v1", timeout=30.0,
                  max_retries=max_retries, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def resp_body(text=None, status="completed", refusal=None, usage=None, model=M_FAST, rid="resp_demo01",
              incomplete=None):
    content = []
    if refusal is not None:
        content.append({"type": "refusal", "refusal": refusal})
    if text is not None:
        content.append({"type": "output_text", "text": text, "annotations": []})
    b = {"id": rid, "object": "response", "created_at": 0, "model": model, "status": status,
         "output": [{"type": "message", "id": "msg_1", "status": status if status != "completed" else "completed",
                     "role": "assistant", "content": content}],
         "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "usage": usage,
         "incomplete_details": incomplete}
    return b


INSTR = json.load(open(Path(tempfile.gettempdir()) / "d22src" / "data.json", encoding="utf-8"))["DAY22_DATA"]["instructor_observation"] \
    if (Path(tempfile.gettempdir()) / "d22src" / "data.json").exists() else None
LOG_TEXT = INSTR["output_text"] if INSTR else "API는 서로 다른 프로그램들이 정해진 규칙으로 데이터를 주고받고 기능을 이용하게 해 주는 “중간 다리”입니다."
LOG_USAGE = {"input_tokens": 31, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 36,
             "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 67}

sent = []


def h_first(req):
    sent.append({"method": req.method, "url": str(req.url), "body": json.loads(req.content),
                 "auth": req.headers.get("authorization", "")[:7] + "…"})
    return httpx.Response(200, json=resp_body(LOG_TEXT, usage=LOG_USAGE), headers={"x-request-id": "req_demo01"})


client = mock_client(h_first)
MODEL = M_FAST
response = client.responses.create(
    model=MODEL, instructions="한국어 한 문장으로 답하세요.", input="API의 역할을 쉽게 설명해 주세요.",
    reasoning={"effort": "none"}, max_output_tokens=1200, store=False)
first_body = sent[-1]
IO["d22-first"] = ("openai SDK + 가짜 서버 · 위쪽 2줄은 SDK 가 실제로 보낸 요청, 아래는 실제 호출 로그 값을 넣은 응답", [
    ("(SDK가 보낸 주소)", f"{first_body['method']} {first_body['url']}", "responses.create = POST /v1/responses"),
    ("(SDK가 보낸 JSON 본문)", json.dumps(first_body["body"], ensure_ascii=False), "키는 본문이 아니라 Authorization 헤더(Bearer …)로 감"),
    ('print("응답의 모델:", response.model)', f"응답의 모델: {response.model}", "요청한 모델과 같은지 따로 확인"),
    ('print("생성 상태:", response.status)', f"생성 상태: {response.status}", "생성 완료 — 내용이 맞다는 뜻은 아님"),
    ('print("답변:", response.output_text)', f"답변: {response.output_text}", "원본 자료에 실린 실제 호출 로그의 문장"),
    ('print("토큰 사용량:", response.usage.model_dump())', "토큰 사용량: " + str(response.usage.model_dump()), "31 + 36 = 67 · 가짜 응답에 없던 칸은 SDK 가 None 으로 채움 (실제 호출 로그에는 cache_write_tokens 0 이 있었음)"),
    ("response.id · response._request_id", f"{response.id} · {response._request_id}", "생성 응답 ID ≠ HTTP 요청 추적 ID ≠ 우리 run_id"),
])

# ------------------------------------------------------------------ 요청 만들기
KNOWLEDGE = [
    {"id": 101, "title": "실습 파일 보관", "content": "실습 파일은 개인 폴더에 보관합니다."},
    {"id": 102, "title": "서버 실행", "content": "프로젝트 폴더에서 서버를 실행합니다."},
    {"id": 103, "title": "질문 작성", "content": "오류 메시지와 시도한 작업을 함께 적습니다."},
]
instructions_text = "제공 지식으로만 답하고 status·answer·source_ids·missing_information 네 필드 JSON을 반환하세요."
question = "오류 질문에 무엇을 적나요?"
selected = [KNOWLEDGE[2]]
payload = {"question": question, "knowledge": selected}
user_input = json.dumps(payload, ensure_ascii=False)
provided_ids = {item["id"] for item in selected}
request = {
    "model": MODEL, "instructions": instructions_text,
    "input": [{"role": "user", "content": user_input}],
    "reasoning": {"effort": "none"}, "max_output_tokens": 1200, "store": False,
}
IO["d22-request"] = ("H01 질문 + 지식 103 (instructions_text 는 짧게 쓴 예)", [
    ("print(user_input)", user_input, "DB 전체가 아니라 고른 지식 1개만"),
    ("print(provided_ids)", str(provided_ids), "이번 요청의 출처 허용 범위 = {103}"),
    ("print(sorted(request))", str(sorted(request)), "이 키들이 그대로 responses.create 의 인자 이름"),
    ('print(request["input"][0]["role"])', request["input"][0]["role"], "role = 메시지 역할, content = 실제 내용"),
])


# ------------------------------------------------------------------ WireAnswer
class WireAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient_context"]
    answer: str
    source_ids: list[int]
    missing_information: str


output_format = {"type": "json_schema", "name": "knowledge_answer", "strict": True,
                 "schema": WireAnswer.model_json_schema()}
sch = output_format["schema"]
IO["d22-wire"] = ("실제 실행", [
    ('WireAnswer.model_json_schema()["required"]', str(sch["required"]), "기본값 없는 네 필드 = 모두 필수"),
    ('…["additionalProperties"]', str(sch["additionalProperties"]), 'extra="forbid" → 추가 필드 금지'),
    ('…["properties"]["status"]', json.dumps(sch["properties"]["status"], ensure_ascii=False), "Literal → enum (허용 값 목록)"),
    ('…["properties"]["source_ids"]', json.dumps(sch["properties"]["source_ids"], ensure_ascii=False), "정수 목록 — 길이·양수 제한은 여기에 없음"),
    ('WireAnswer.model_validate({..., "source_ids": ["101"]}).source_ids',
     err(lambda: WireAnswer.model_validate({"status": "answered", "answer": "a", "source_ids": ["101"], "missing_information": ""}).source_ids),
     'Wire 는 strict 가 아니라 "101" → 101 로 바꿔 줌 — 로컬 검증은 KnowledgeAnswer 가 맡는 이유'),
])

sent.clear()
client2 = mock_client(lambda req: (sent.append(json.loads(req.content)),
                                   httpx.Response(200, json=resp_body("{}", usage=LOG_USAGE)))[1])
client2.responses.create(**request, text={"format": output_format})
tb = sent[-1]
IO["d22-create"] = ("SDK + 가짜 서버 · SDK 가 실제로 보낸 본문에서 확인", [
    ('body["text"]["format"]["type"], ["name"], ["strict"]',
     f'{tb["text"]["format"]["type"]}, {tb["text"]["format"]["name"]}, {tb["text"]["format"]["strict"]}',
     "출력 형식은 text.format 에 들어감"),
    ('sorted(body["text"]["format"]["schema"]["properties"])', str(sorted(tb["text"]["format"]["schema"]["properties"])), "WireAnswer 의 네 필드가 그대로 전송"),
    ("sorted(body)", str(sorted(tb)), "**request 로 펼친 키 + text"),
    ('client.responses.create(**request, text=..., model="x")', err(lambda: client2.responses.create(**request, text={"format": output_format}, model="x")),
     "같은 키를 두 번 주면 Python 이 바로 거부 — request 안에 text 를 중복으로 넣지 말라는 이유"),
])


# ------------------------------------------------------------------ 완료 판정
def completed_text(response):
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    return None, "refusal"
    if response.status != "completed":
        return None, response.status
    if not response.output_text.strip():
        return None, "empty_output"
    return response.output_text, None


INSUFFICIENT = "제공된 지식으로는 답할 수 없습니다."


def answer_(t, ids):
    return {"status": "answered", "answer": t, "source_ids": ids, "missing_information": ""}


def insufficient_(m):
    return {"status": "insufficient_context", "answer": INSUFFICIENT, "source_ids": [], "missing_information": m}


def rawj(x):
    return json.dumps(x, ensure_ascii=False, indent=2)


SRV = "프로젝트 폴더에서 서버를 실행합니다."
GATE_CASES = {   # 교안 GATE_CASES: (이름, response.status, 거절?, 원문, provided_ids)
    "good": ("정상 답변", "completed", False, rawj(answer_(SRV, [102])), [102]),
    "missing": ("정보 부족 응답(정상 처리)", "completed", False, rawj(insufficient_("보관 기간")), [101]),
    "refusal": ("API의 요청 거절", "completed", True, "", [102]),
    "incomplete": ("불완전 응답", "incomplete", False, '{"status":"answered","answer":', [102]),
    "empty": ("완료 + 빈 텍스트", "completed", False, "", [102]),
    "broken": ("완료 + JSON 오류", "completed", False, '{"status":"answered"', [102]),
    "unknown": ("문맥 밖 출처 999", "completed", False, rawj(answer_(SRV, [999])), [102]),
    "false_evidence": ("존재하는 ID, 잘못된 인용", "completed", False, rawj(answer_(SRV, [101])), [101, 102, 103]),
}


def gate_response(st, refused, raw):
    b = resp_body(None if refused else raw, status=st, refusal="이 요청은 도와드릴 수 없습니다." if refused else None,
                  incomplete={"reason": "max_output_tokens"} if st == "incomplete" else None)
    return Response.model_validate(b)


# 로컬 계약 (교안 local-contract 그대로)
SourceID = Annotated[int, Field(strict=True, gt=0)]


class KnowledgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1, max_length=500)
    source_ids: list[SourceID] = Field(max_length=3)
    missing_information: str = Field(max_length=200)


def check_rules(obj, provided_ids):   # 교안 rules 그대로 (16일 check_rules + final_rules)
    ids = obj.source_ids
    if len(ids) != len(set(ids)):
        return "duplicate_source"
    if not set(ids) <= set(provided_ids):
        return "unknown_source"
    if obj.status == "answered":
        if not ids or obj.missing_information != "":
            return "status_rule"
    elif ids or not obj.missing_information.strip():
        return "status_rule"
    if not obj.answer.strip():
        return "blank_answer"
    if obj.status == "insufficient_context":
        if obj.answer != "제공된 지식으로는 답할 수 없습니다.":
            return "status_rule"
    return None


def reject_constant(value):
    raise ValueError("비표준 수치")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("중복 JSON 키")
        result[key] = value
    return result


def load_json_strict(raw):   # 16일 정의
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)


def validate_answer(raw, provided_ids, load_json=json.loads):   # 16일 정의 (규칙은 위 check_rules)
    try:
        data = load_json(raw)
    except ValueError:
        return None, "json_error"
    try:
        obj = KnowledgeAnswer.model_validate(data)
    except ValidationError:
        return None, "schema_error"
    e = check_rules(obj, provided_ids)
    return (None, e) if e else (obj, None)


def inspect_answer(raw, provided_ids):   # 교안 validator-bridge 그대로
    obj, error = validate_answer(raw, provided_ids, load_json=load_json_strict)
    return {"auto_status": "passed_needs_human_review" if error is None else "failed",
            "answer_object": obj if error is None else None, "error": error}


def run_record(response, provided_ids):   # 교안 inspect 그대로
    raw, gate_error = completed_text(response)
    if gate_error is not None:
        run_status = "refused" if gate_error == "refusal" else gate_error
        return {"run_status": run_status, "answer_object": None, "evidence_ok": None}
    report = inspect_answer(raw, provided_ids)
    return {"run_status": "validated" if report["error"] is None else report["error"],
            "answer_object": report["answer_object"], "evidence_ok": None}


gate_rows, rec_rows, gate_table = [], [], []
for k, (label, st, refused, raw, ids) in GATE_CASES.items():
    resp = gate_response(st, refused, raw)
    g = completed_text(resp)
    rec = run_record(resp, ids)
    ao = rec["answer_object"]
    gate_table.append({"key": k, "label": label, "status": st, "gate": g[1], "run_status": rec["run_status"],
                       "ans_status": ao.status if ao else None})
    gate_rows.append((f"{label}  (response.status={st})", f"(… , {g[1]!r})" if g[1] else "(원문, None)",
                      {"good": "통과 → 검사기로", "missing": "통과 — 정보 부족도 '완료된 응답'",
                       "refusal": "거절 콘텐츠 먼저 확인", "incomplete": "텍스트가 있어도 미완성은 막음",
                       "empty": "완료인데 비었음", "broken": "통과 — JSON 오류는 다음 단계가 잡음",
                       "unknown": "통과 — 출처 범위는 다음 단계가 잡음", "false_evidence": "통과 — 근거는 사람이"}[k]))
    rec_rows.append((label, f'run_status={rec["run_status"]!r}, answer.status={(ao.status if ao else None)!r}, evidence_ok=None',
                     {"good": "자동 검사 통과 = 사람 검토 대기", "missing": "정상 실행 + 업무상 정보 부족",
                      "refusal": "앱 상태 refused (정보 부족으로 위장 안 함)", "incomplete": "부분 JSON 사용 안 함",
                      "empty": "검사기로 못 넘김", "broken": "16일 검사기가 json_error",
                      "unknown": "16일 검사기가 unknown_source",
                      "false_evidence": "자동으로는 validated — 102를 실행 위치 근거로 101을 인용한 걸 못 잡음"}[k]))
RESULTS["gate"] = gate_table
IO["d22-gate"] = ("교안 GATE_CASES 8개를 openai SDK 의 실제 Response 객체로 만들어 강의 함수 그대로", gate_rows)
IO["d22-inspect"] = ("같은 8개 · completed_text → inspect_answer(16일 검사기) → 실행 기록", rec_rows)

# 로컬 계약 · 규칙 (교안 FIXTURES 14개 중 대표)
A = "개인 폴더에 보관합니다."


def kv(d):
    return err(lambda: KnowledgeAnswer.model_validate(d).source_ids)


IO["d22-local-contract"] = ("KnowledgeAnswer.model_validate 에 dict 직접", [
    ("source_ids=[101]", kv(answer_(A, [101])), "통과"),
    ('source_ids=["101"]', kv(answer_(A, ["101"])).split("\n")[0] + " …", "문자열 ID 거부 (Wire 와 다름)"),
    ("source_ids=[True]", kv(answer_(A, [True])).split("\n")[0] + " …", "불리언도 정수로 안 바꿈"),
    ("source_ids=[0]", kv(answer_(A, [0])).split("\n")[0] + " …", "gt=0 → 양의 정수만"),
    ("source_ids=[101,102,103,104]", kv(answer_(A, [101, 102, 103, 104])).split("\n")[0] + " …", "예시 설정: 최대 3개"),
    ('answer="가"×501', kv(answer_("가" * 501, [101])).split("\n")[0] + " …", "예시 설정: 1~500자"),
])


def rule(d, ids):
    return repr(check_rules(KnowledgeAnswer.model_validate(d), ids))


IO["d22-rules"] = ("교안 rules 그대로 · obj 는 KnowledgeAnswer 를 통과한 객체", [
    ("[102] 인용, provided_ids={102}", rule(answer_(SRV, [102]), {102}), "통과 (근거 검토는 별도)"),
    ("[101] 인용, provided_ids={102}  (101 은 DB 에 있음)", rule(answer_(A, [101]), {102}), "DB 에 있어도 이번에 안 보냈으면 범위 밖"),
    ("[102,102]", rule(answer_(SRV, [102, 102]), {102}), "중복이 먼저 걸림"),
    ("answered 인데 source_ids=[]", rule(answer_(SRV, []), {102}), "답했으면 출처가 있어야"),
    ("insufficient_context 인데 [101]", rule({**insufficient_("보관 위치"), "source_ids": [101]}, {101}), "정보 부족이면 출처 비움"),
    ('insufficient_context, answer="몰라요"', rule({**insufficient_("보관 기간"), "answer": "몰라요"}, {101}), "정보 부족 안내문은 고정 문장"),
    ('answer="   "', rule(answer_("   ", [101]), {101}), "공백만은 거부 (min_length=1 은 통과함)"),
])

IO["d22-validator-bridge"] = ("교안 inspect_answer 그대로 · 16일 load_json_strict 사용", [
    ("inspect_answer(정상 [102], {102})", str({k: (v if k != "answer_object" else "KnowledgeAnswer(...)") for k, v in inspect_answer(rawj(answer_(SRV, [102])), {102}).items()}), "통과여도 '사람 검토 필요'"),
    ('inspect_answer(키 중복 "answer" 두 번, {101})',
     str(inspect_answer('{"status":"answered","answer":"앞의 값","answer":"개인 폴더에 보관합니다.","source_ids":[101],"missing_information":""}', {101})),
     "strict 파서 → json_error (기본 json.loads 라면 조용히 통과)"),
    ("inspect_answer(NaN, {101})", str(inspect_answer('{"status":"answered","answer":"a","source_ids":[NaN],"missing_information":""}', {101})), "비표준 NaN 도 파싱 단계에서 거부"),
])

# ------------------------------------------------------------------ Streaming (SSE)


def sse(events, cut_after=None):
    """이벤트 목록을 SSE 바이트로. cut_after 이면 그 개수 뒤에 연결을 끊는다."""
    chunks = []
    for i, e in enumerate(events):
        chunks.append(f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': i}, ensure_ascii=False)}\n\n".encode())

    def gen():
        for i, c in enumerate(chunks):
            if cut_after is not None and i == cut_after:
                raise httpx.ReadError("연결 끊김(모의)")
            yield c
    return gen


class Stream(httpx.SyncByteStream):
    def __init__(self, g):
        self.g = g

    def __iter__(self):
        yield from self.g()


def d(txt):
    return {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0,
            "delta": txt, "logprobs": []}


def done(kind, body):
    return {"type": kind, "response": body}


GOOD_RAW = json.dumps(answer_(SRV, [102]), ensure_ascii=False)
pieces = ['{"status":', '"answered",', '"answer":"프로젝트 폴더에서 서버를 실행합니다.",', '"source_ids":[102],"missing_information":""}']
created = {"type": "response.created", "response": resp_body(None, status="in_progress")}
STREAMS = {
    "completed": ("delta 여러 개 → completed", [created] + [d(p) for p in pieces]
                  + [{"type": "response.output_text.done", "item_id": "msg_1", "output_index": 0, "content_index": 0, "text": GOOD_RAW, "logprobs": []},
                     done("response.completed", resp_body(GOOD_RAW, usage=LOG_USAGE))], None),
    "cut": ("delta → 연결 끊김", [created] + [d(p) for p in pieces[:2]] + [d(p) for p in pieces[2:]], 3),
    "incomplete": ("delta → incomplete", [created] + [d(p) for p in pieces[:2]]
                   + [done("response.incomplete", resp_body(pieces[0] + pieces[1], status="incomplete", incomplete={"reason": "max_output_tokens"}, usage=LOG_USAGE))], None),
    "refusal": ("refusal → completed", [created, {"type": "response.refusal.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "도와드릴 수 없습니다."},
                                       done("response.completed", resp_body(None, refusal="도와드릴 수 없습니다.", usage=LOG_USAGE))], None),
    "empty": ("빈 delta → completed", [created, d(""), d("")] + [d(p) for p in pieces] + [done("response.completed", resp_body(GOOD_RAW, usage=LOG_USAGE))], None),
    "nodone": ("완료 이벤트 없이 종료", [created] + [d(p) for p in pieces], None),
    "error": ("error 이벤트", [created, d(pieces[0]), {"type": "error", "code": "server_error", "message": "모의 오류", "param": None}], None),
}


def run_stream(events, cut):
    cl = mock_client(lambda req: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream(sse(events, cut))))
    from time import perf_counter
    t0 = perf_counter()
    first_ms, final = None, None
    stream_error = None
    parts = []
    seen = []
    exc = None
    try:
        with cl.responses.create(**request, text={"format": output_format}, stream=True) as events_:
            for event in events_:
                seen.append(event.type)
                if event.type == "response.output_text.delta":
                    if event.delta and first_ms is None:
                        first_ms = (perf_counter() - t0) * 1000
                    parts.append(event.delta)
                elif event.type == "response.completed":
                    final = event.response
                elif event.type in {"response.failed", "response.incomplete"}:
                    final = event.response
                elif event.type == "error":
                    stream_error = "stream_error"
    except Exception as e:  # noqa: BLE001  — 교안: 바깥 호출부가 담당
        exc = type(e).__name__
        stream_error = stream_error or "stream_interrupted"
    if final is None or stream_error is not None:
        outcome = "interrupted"
        gate = None
        final_view = "interrupted"
    else:
        final_view = f"final.status={final.status!r} · output_text {len(final.output_text)}자"
        raw, gate = completed_text(final)
        outcome = "validate 로" if gate is None else gate
    usage = None if final is None or final.usage is None else final.usage.total_tokens
    return {"seen": seen, "parts": len(parts), "first_text": first_ms is not None, "outcome": outcome, "final_view": final_view,
            "exc": exc, "usage_total": usage, "preview": "".join(parts)[:30]}


st_rows, st_table = [], []
for k, (label, evs, cut) in STREAMS.items():
    r = run_stream(evs, cut)
    st_table.append({"key": k, "label": label, **r})
    st_rows.append((label,
                    f'조각 {r["parts"]}개 · first_ms 관측={r["first_text"]} · {r["final_view"]} · usage 합계={r["usage_total"]}'
                    + (f' · 예외 {r["exc"]}' if r["exc"] else ""),
                    {"completed": "완료 + 원문 있음 → json.loads → model_validate 로", "cut": "보인 조각이 있어도 최종 Response·usage 없음",
                     "incomplete": "끝났지만 미완성 — 부분 JSON 버림", "refusal": "status 는 completed 인데 텍스트 0자 — 거절 콘텐츠라서 (status 만 보면 안 되는 이유)",
                     "empty": "빈 조각은 첫 텍스트로 안 셈", "nodone": "조각은 다 왔어도 완료 확인 못 함",
                     "error": "error 이벤트 → interrupted"}[k]))
RESULTS["stream"] = st_table
IO["d22-stream"] = ("SSE 바이트를 가짜 서버가 보내고, SDK 가 실제로 해석한 이벤트로 위 루프 그대로 (시간 값은 의미 없어 관측 여부만)", st_rows)

# ------------------------------------------------------------------ 오류와 SDK 재시도


def err_handler(status, code, headers=None, counter=None):
    def h(req):
        counter[0] += 1
        return httpx.Response(status, json={"error": {"message": "모의 오류", "type": "t", "code": code, "param": None}}, headers=headers or {})
    return h


ERR_CASES = [("401 invalid_api_key", 401, "invalid_api_key", None), ("403 permission_denied", 403, "permission_denied", None),
             ("400 invalid_request", 400, "invalid_request", None), ("429 rate_limit_exceeded", 429, "rate_limit_exceeded", {"retry-after-ms": "5"}),
             ("429 insufficient_quota", 429, "insufficient_quota", {"retry-after-ms": "5"}), ("500 server_error", 500, "server_error", {"retry-after-ms": "5"})]
err_rows, err_table = [], []
for label, stt, code, hd in ERR_CASES:
    res = {}
    for mr in (0, 2):
        n = [0]
        cl = mock_client(err_handler(stt, code, hd, n), max_retries=mr)
        try:
            cl.responses.create(model=MODEL, input="q")
        except openai.APIError as e:
            res[mr] = (type(e).__name__, getattr(e, "code", None), n[0])
    err_table.append({"label": label, "exc": res[0][0], "code": res[0][1], "sends0": res[0][2], "sends2": res[2][2]})
    err_rows.append((label, f"{res[0][0]} · e.code={res[0][1]!r} · 전송 {res[0][2]}회 (max_retries=0) / {res[2][2]}회 (SDK 기본 2)",
                     {401: "인증 — 중단", 403: "권한 — 중단", 400: "요청 수정 후 별도 실행",
                      500: "SDK 기본값이면 앱 모르게 3번 전송"}.get(stt, "대기 후 제한 재시도 가능" if code == "rate_limit_exceeded" else
                                                         "같은 RateLimitError 인데 SDK 기본값은 할당량 문제도 3번 보냄 — e.code 로 갈라야")))
n = [0]


def h_to(req):
    n[0] += 1
    raise httpx.ReadTimeout("모의 timeout", request=req)


for mr in (0, 2):
    n[0] = 0
    try:
        mock_client(h_to, max_retries=mr).responses.create(model=MODEL, input="q")
    except openai.APIError as e:
        err_table.append({"label": f"timeout mr={mr}", "exc": type(e).__name__, "sends": n[0]})
        if mr == 0:
            t0n = (type(e).__name__, n[0])
        else:
            t2n = n[0]
err_rows.append(("timeout (응답이 안 옴)", f"{t0n[0]} · 전송 {t0n[1]}회 (max_retries=0) / {t2n}회 (SDK 기본 2)", "서버가 이미 생성했을 수도 — 자동 재전송 안 함"))
err_rows.append(("issubclass(openai.RateLimitError, openai.APIStatusError)", str(issubclass(openai.RateLimitError, openai.APIStatusError)),
                 "모든 HTTP 오류 예외의 부모 = APIStatusError"))
err_rows.append(("issubclass(openai.APITimeoutError, openai.APIStatusError)", str(issubclass(openai.APITimeoutError, openai.APIStatusError)),
                 "timeout 은 HTTP 상태가 없음 → APIConnectionError 쪽"))
RESULTS["errors"] = err_table
IO["d22-ownership"] = ("openai SDK + 가짜 서버가 오류만 돌려줌 · 같은 오류를 max_retries=0 과 SDK 기본값(2)으로 각각 보냄", err_rows)


# ------------------------------------------------------------------ 대기 판단
MAX_SENDS_TOTAL = 4


def wait_main(attempt, retry_after_s, remaining_s, sends_used, jitter):
    fallback = min(8.0, 2 ** (attempt - 1))
    base = (retry_after_s if retry_after_s is not None else fallback)
    wait_s = base + jitter
    if wait_s > remaining_s:
        return "deferred"
    if sends_used >= MAX_SENDS_TOTAL:
        return "attempt_limit"
    return round(wait_s, 2)


def wm(*a):
    lo, hi = wait_main(*a, 0.0), wait_main(*a, 0.25)
    return f"{lo} ~ {hi}" if lo != hi else str(lo)


IO["d22-wait-main"] = ("교안 발췌를 함수로 감싸 실행 · 지터는 random.uniform(0, 0.25) 의 양 끝(0, 0.25)을 넣어 범위로", [
    ("attempt=1, Retry-After 없음, 남은 30초, 전송 1회", wm(1, None, 30, 1), "1초(2⁰) + 지터"),
    ("attempt=3, Retry-After 없음", wm(3, None, 30, 1), "4초(2²) + 지터"),
    ("attempt=6, Retry-After 없음", wm(6, None, 60, 1), "2⁵=32 이지만 min(8.0, …) 으로 8초가 상한"),
    ("Retry-After=10, 남은 3초", wm(1, 10, 3, 1), "서버 최소 대기를 줄여 보내지 않고 미룸"),
    ("Retry-After=2, 남은 30초, 이미 4회 전송", wm(1, 2, 30, 4), "시간은 되지만 총 전송 상한"),
    ("Retry-After=0.5, 남은 0.6초", f"{wait_main(1, 0.5, 0.6, 1, 0.0)} / {wait_main(1, 0.5, 0.6, 1, 0.25)}", "지터에 따라 결과가 갈림 — 경계에서는 예산 여유를 둘 것"),
])


def retry_delay(attempt, retry_after_s, remaining_s, jitter):   # 교안 wait-appendix (uniform 대신 jitter 인자)
    hinted = 0.0 if retry_after_s is None else retry_after_s
    if hinted < 0 or remaining_s <= 0:
        return None
    exponential = 2 ** (attempt - 1)
    wait = max(hinted, exponential) + jitter
    if wait >= remaining_s:
        return None
    return wait


def rd(*a):
    lo, hi = retry_delay(*a, 0.0), retry_delay(*a, 0.5)
    return f"{lo} ~ {hi}" if lo != hi else str(lo)


IO["d22-wait-appendix"] = ("교안 retry_delay · 지터는 uniform(0, 0.5) 양 끝", [
    ("retry_delay(1, None, 30)", rd(1, None, 30), "헤더 없으면 2⁰=1초"),
    ("retry_delay(3, 1, 30)", rd(3, 1, 30), "max(서버 1초, 지수 4초) = 4초 — 더 긴 쪽"),
    ("retry_delay(1, 10, 30)", rd(1, 10, 30), "서버가 10초라면 10초"),
    ("retry_delay(1, 10, 3)", rd(1, 10, 3), "None = deferred (짧게 잘라 보내지 않음)"),
    ("retry_delay(1, -5, 30)", rd(1, -5, 30), "음수 헤더는 잘못된 값 → 중단"),
    ("retry_delay(6, None, 100)", rd(6, None, 100), "본문 예시와 달리 상한 8초가 없음 — 교안이 '상수가 다른 별도 예시'라고 한 부분"),
])


# ------------------------------------------------------------------ 전체 흐름 (send_bounded 최소 구현)
class Budget:
    def __init__(self):
        self.used = 0


def send_bounded(client, request, budget, log, gen_no, remaining_s=30.0):
    """노트가 붙인 최소 구현: 생성 1번 안에서 전송 최대 2회, 전체 4회."""
    for send_attempt in (1, 2):
        if budget.used >= MAX_SENDS_TOTAL:
            return SimpleNamespace(transport_failed=True, response=None, reason="attempt_limit")
        budget.used += 1
        try:
            resp = client.responses.create(**request)
            log.append((gen_no, send_attempt, "응답 수신"))
            return SimpleNamespace(transport_failed=False, response=resp, reason=None)
        except openai.RateLimitError as e:
            if e.code != "rate_limit_exceeded":
                log.append((gen_no, send_attempt, f"429 {e.code} → 중단"))
                return SimpleNamespace(transport_failed=True, response=None, reason=e.code)
            ra = e.response.headers.get("retry-after")
            w = wait_main(send_attempt, float(ra) if ra else None, remaining_s, budget.used, 0.0)
            log.append((gen_no, send_attempt, f"429 rate_limit_exceeded → 대기 판단 {w}"))
            if not isinstance(w, float) or send_attempt == 2:
                return SimpleNamespace(transport_failed=True, response=None, reason=w if isinstance(w, str) else "rate_limited")
        except openai.InternalServerError:
            log.append((gen_no, send_attempt, "500 → 예산 안에서 1회 재시도" if send_attempt == 1 else "500 → 생성당 2회 상한, 중단"))
            if send_attempt == 2:
                return SimpleNamespace(transport_failed=True, response=None, reason="server_error")
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError) as e:
            log.append((gen_no, send_attempt, f"{type(e).__name__} → 중단"))
            return SimpleNamespace(transport_failed=True, response=None, reason=type(e).__name__)
        except openai.APITimeoutError:
            log.append((gen_no, send_attempt, "timeout → 자동 재전송 안 함"))
            return SimpleNamespace(transport_failed=True, response=None, reason="timeout")
    return SimpleNamespace(transport_failed=True, response=None, reason="send_limit")


def scripted_server(script):
    it = iter(script)

    def h(req):
        kind = next(it)
        if kind == "429":
            return httpx.Response(429, json={"error": {"message": "x", "type": "requests", "code": "rate_limit_exceeded", "param": None}}, headers={"retry-after": "1"})
        if kind == "quota":
            return httpx.Response(429, json={"error": {"message": "x", "type": "insufficient_quota", "code": "insufficient_quota", "param": None}})
        if kind == "401":
            return httpx.Response(401, json={"error": {"message": "x", "type": "t", "code": "invalid_api_key", "param": None}})
        if kind == "500":
            return httpx.Response(500, json={"error": {"message": "x", "type": "t", "code": "server_error", "param": None}})
        raw = {"good": rawj(answer_(SRV, [102])), "broken": '{"status":"answered"'}[kind]
        return httpx.Response(200, json=resp_body(raw, usage=LOG_USAGE))
    return h


def run_bounded(script):
    cl = mock_client(scripted_server(script))
    budget, log = Budget(), []
    req = dict(request)
    generation = 0
    final = None
    while generation < 2 and budget.used < 4:
        generation += 1
        outcome = send_bounded(cl, req, budget, log, generation)
        if outcome.transport_failed:
            final = outcome.reason
            break
        raw, ge = completed_text(outcome.response)
        rep = inspect_answer(raw, {102}) if ge is None else {"error": ge}
        log.append((generation, "-", f"검증 → {rep['error'] or 'passed_needs_human_review'}"))
        if rep["error"] is None:
            final = "validated"
            break
        if rep["error"] not in ("json_error", "schema_error") or generation == 2:
            final = rep["error"]
            break
        req = {**req, "instructions": req["instructions"] + " (형식 수정: 이전 응답이 JSON 형식 오류)"}
    return final, budget.used, log


BOUNDED = {"auth": ("401 → 중단", ["401"]), "rate": ("429 일시적 제한 → 정상", ["429", "good"]),
           "quota": ("429 할당량 → 중단", ["quota"]), "server": ("500 → 500 → 중단", ["500", "500"]),
           "repair": ("429 + 형식 오류 + 수정 → 정상", ["429", "broken", "429", "good"]),
           "repair_fail": ("형식 오류 → 형식 오류", ["broken", "broken"])}
b_rows, b_table = [], []
_sleep = 0
for k, (label, script) in BOUNDED.items():
    fin, used, log = run_bounded(script)
    b_table.append({"key": k, "label": label, "final": fin, "sends": used, "log": log})
    b_rows.append((label, f"최종 {fin!r} · 실제 전송 {used}회 · 기록 " + " / ".join(f"생성{g}·전송{s}: {m}" for g, s, m in log),
                   {"auth": "1회로 끝", "rate": "이전 429 도 기록에 남김", "quota": "같은 429 라도 기다리지 않음",
                    "server": "생성당 전송 2회 상한", "repair": "4회 = 전체 상한에 딱 맞음",
                    "repair_fail": "형식 수정은 1회까지 — 3번째 생성은 없음"}[k]))
RESULTS["bounded"] = b_table
IO["d22-bounded"] = ("교안 의사코드 + 노트의 send_bounded 최소 구현 · SDK + 가짜 서버가 각본대로 응답 (sleep 은 실제로 하지 않고 판단만 기록)", b_rows)


# ------------------------------------------------------------------ usage
def read_usage(u):   # 교안 usage 그대로
    return None if u is None else {
        "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
        "cached_tokens": getattr(u.input_tokens_details, "cached_tokens", None),
        "reasoning_tokens": getattr(u.output_tokens_details, "reasoning_tokens", None)}


def usage_obj(raw):
    if raw is None:
        return None
    # SDK 가 실제 응답을 읽는 방식(관대한 construct)으로 — 가짜 서버 경유
    cl = mock_client(lambda req: httpx.Response(200, json=resp_body("x", usage=raw)))
    return cl.responses.create(model=MODEL, input="q").usage


USAGE = {"normal": ("일반 응답", {"input_tokens": 1000, "output_tokens": 200, "input_tokens_details": {"cached_tokens": 400}, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1200}),
         "zero": ("cached_tokens 가 실제 0", {"input_tokens": 1000, "output_tokens": 200, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 1200}),
         "missing": ("캐시 세부 필드가 없음", {"input_tokens": 1000, "output_tokens": 200, "input_tokens_details": None, "output_tokens_details": None, "total_tokens": 1200}),
         "timeout": ("Timeout 으로 usage 미수신", None)}
u_rows = []
for k, (label, raw) in USAGE.items():
    u_rows.append((label, str(read_usage(usage_obj(raw))),
                   {"normal": "cached 400 은 input 1000 안의 일부", "zero": "0 은 '관측된 0'",
                    "missing": "필드 없음 → None (0 이라고 단정 안 함)", "timeout": "usage 자체가 None → 비용 미확인"}[k]))
mis = usage_obj(USAGE["missing"][1])
u_rows.append(("(잘못된 방법) getattr(...) or 0", str((getattr(mis.input_tokens_details, "cached_tokens", None) or 0)), "모르는 값이 0 으로 둔갑 — or 0 을 쓰지 않는 이유"))
IO["d22-usage"] = ("가상 usage 사례를 SDK Response 로 파싱해 실행", u_rows)


# ------------------------------------------------------------------ 비용
def est(i, c, o, p_in=Decimal("0.75"), p_c=Decimal("0.075"), p_o=Decimal("4.50")):
    if None in (i, c, o):
        return None
    return ((i - c) * p_in + c * p_c + o * p_o) / Decimal(1_000_000)


wrong = (1000 * Decimal("0.75") + 400 * Decimal("0.075") + 200 * Decimal("4.50")) / Decimal(1_000_000)
parts_ = {"일반 입력": 600 * Decimal("0.75") / Decimal(1_000_000), "캐시 입력": 400 * Decimal("0.075") / Decimal(1_000_000),
          "출력": 200 * Decimal("4.50") / Decimal(1_000_000)}
RESULTS["cost"] = {k: float(v) for k, v in parts_.items()}
RESULTS["cost_total"] = float(est(1000, 400, 200))
RESULTS["cost_wrong"] = float(wrong)
IO["d22-cost"] = ("실제 실행 (단가·사용량은 가상값)", [
    ("print(estimated_usd)  # 1000 · 400 · 200", str(est(1000, 400, 200)), "손 계산 0.00138 과 같음"),
    ("항목별", " + ".join(f"{k} {v.normalize()}" for k, v in parts_.items()), "출력 200 토큰이 전체의 65% — 출력 단가가 6배"),
    ("cached_tokens = 0 이면", str(est(1000, 0, 200)), "캐시 400 이 0.000270 아낀 셈"),
    ("(잘못) 입력 1000 전체 + 캐시 400 을 또 더함", str(wrong), "캐시 입력을 두 번 셈 → 과대 추정"),
    ("output_tokens = 2000 이면", str(est(1000, 400, 2000)), "출력이 늘면 비용 대부분이 출력"),
    ("usage = None (timeout)", str(est(None, None, None)), "estimated_usd=None — 0 으로 쓰지 않음"),
    ("0.1 + 0.2  (float)", repr(0.1 + 0.2), "금액에 Decimal 을 쓰는 이유"),
])


# ------------------------------------------------------------------ cache key
def cache_key(req, fmt, scope="classroom-demo", validator="day21-v1"):
    cache_basis = {"scope": scope, "model": req["model"], "request": req, "format": fmt, "validator_version": validator}
    canonical = json.dumps(cache_basis, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


k0 = cache_key(request, output_format)
req_reordered = dict(reversed(list(request.items())))
changed_kb = {**request, "input": [{"role": "user", "content": json.dumps({"question": question, "knowledge": [{**KNOWLEDGE[2], "content": "오류 메시지, 시도한 작업, 실행 환경을 함께 적습니다."}]}, ensure_ascii=False)}]}
changed_model = {**request, "model": M_QUAL}
ck = {"기준": k0, "키 순서만 바꿈": cache_key(req_reordered, output_format), "지식 103 본문 수정 (ID 같음)": cache_key(changed_kb, output_format),
      "모델 → quality": cache_key(changed_model, output_format), "scope 다른 조직": cache_key(request, output_format, scope="other-org"),
      "validator_version day22-v1": cache_key(request, output_format, validator="day22-v1")}
RESULTS["cache_keys"] = ck
IO["d22-cache-key"] = ("실제 실행 · 예시 요청 기준 (앞 16글자만 표시)", [
    (k, v[:16] + "…", "기준과 " + ("같음" if v == k0 else "다름") + {"기준": " — 64글자 16진수", "키 순서만 바꿈": " — sort_keys 덕분",
                                                                  "지식 103 본문 수정 (ID 같음)": " — 같은 ID 라도 본문이 바뀌면 다른 요청",
                                                                  "모델 → quality": " — 모델을 바꾸면 캐시도 새로",
                                                                  "scope 다른 조직": " — 권한 범위가 다르면 재사용 금지",
                                                                  "validator_version day22-v1": " — 검사 규칙이 바뀌어도 새로"}[k])
    for k, v in ck.items()])


# ------------------------------------------------------------------ cache read
def read_cache(cache, key, now, provided_ids):   # 교안 cache-read 그대로
    entry = cache.get(key)
    if entry is None or now >= entry["expires_at"]:
        return None
    if not entry["human_approved"]:
        return None
    report = inspect_answer(entry["raw"], provided_ids)
    if report["auto_status"] != "passed_needs_human_review":
        return None
    return {"answer_object": report["answer_object"], "source_mode": "app_cache",
            "origin_run_id": entry["origin_run_id"], "new_api_send_count": 0}


H01_RAW = rawj(answer_("오류 메시지와 시도한 작업을 함께 적습니다.", [103]))
cache = {k0: {"raw": H01_RAW, "expires_at": 60, "human_approved": True, "origin_run_id": "demo-001"}}
cache_pending = {k0: {**cache[k0], "human_approved": False}}


def rc(c, key, now, ids):
    r = read_cache(c, key, now, ids)
    return "None (miss)" if r is None else str({**r, "answer_object": "KnowledgeAnswer(...)"})


cache_cases = [("저장 0초 · 현재 30초 · 승인됨 · provided {103}", rc(cache, k0, 30, {103}), "hit — 새 API 전송 0회"),
               ("현재 60초 (expires_at=60)", rc(cache, k0, 60, {103}), "now >= expires_at → 만료"),
               ("승인 전 (human_approved=False)", rc(cache_pending, k0, 30, {103}), "사람 검토 전 답은 재사용 안 함"),
               ("같은 key, 이번 provided {102}", rc(cache, k0, 30, {102}), "현재 문맥으로 재검사 → 103 이 범위 밖 → miss"),
               ("지식 본문 수정 → 다른 key", rc(cache, ck["지식 103 본문 수정 (ID 같음)"], 30, {103}), "key 자체가 달라 항목 없음")]
RESULTS["cache"] = [(a, b) for a, b, _ in cache_cases]
IO["d22-cache-read"] = ("교안 read_cache 그대로 · TTL 60초 가상 시간", cache_cases)


# ------------------------------------------------------------------ routing
MODELS = {"fast": M_FAST, "quality": M_QUAL}


def choose_model(mode):
    if mode not in MODELS:
        raise ValueError("지원하지 않는 실행 모드")
    return MODELS[mode], f"explicit_mode:{mode}"


IO["d22-routing"] = ("교안 코드 그대로", [
    ('choose_model("fast")', repr(choose_model("fast")), "확인된 기본 실습 모델 + 선택 이유"),
    ('choose_model("quality")', repr(choose_model("quality")), "비교 후보 — 생성은 미확인 (not_run)"),
    ('choose_model("turbo")', err(lambda: choose_model("turbo")), "모르는 모드는 호출 전에 거부"),
    ('choose_model("Fast")', err(lambda: choose_model("Fast")), "대소문자도 그대로 비교"),
    ("cache_key(fast 요청) == cache_key(quality 요청)", str(cache_key(request, output_format) == cache_key(changed_model, output_format)), "모델을 바꾸면 key 도 다시 계산"),
])

# ------------------------------------------------------------------ 모델 목록 (가짜 서버)
listing = {"object": "list", "data": [{"id": M_FAST, "object": "model", "created": 0, "owned_by": "system"},
                                      {"id": "gpt-5.4-nano", "object": "model", "created": 0, "owned_by": "system"}]}
out = []
with OpenAI(api_key=FAKE_KEY, base_url="https://api.openai.com/v1", timeout=30.0, max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=listing)))) as c_:
    listed_ids = {item.id for item in c_.models.list()}
    for model_id in dict.fromkeys([MODEL, M_QUAL, MODEL]):
        out.append(f"{model_id} {'목록에 있음' if model_id in listed_ids else '목록에서 확인되지 않음'}")
n403 = [0]
with OpenAI(api_key=FAKE_KEY, max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(err_handler(403, "permission_denied", None, n403)))) as c_:
    try:
        c_.models.list()
    except openai.APIError as exc:
        e403 = f"목록 조회 실패: {type(exc).__name__} {getattr(exc, 'status_code', None)}"
IO["d22-model-check"] = ("SDK + 가짜 서버 · 목록은 2개짜리 가짜 (실제 계정 목록 아님)", [
    ("for model_id in dict.fromkeys([MODEL, 비교후보, MODEL])", " / ".join(out), "dict.fromkeys = 순서 유지 중복 제거 (3개 → 2줄)"),
    ("목록 조회가 403", e403, "원문 오류·키 대신 예외 이름과 상태만"),
])

# 강사 로그 산수
IO["d22-instructor"] = ("실제 호출 로그의 숫자로 계산", [
    ("31 + 36", str(31 + 36), "total_tokens 67 과 일치 — cached·reasoning 은 이미 각 합 안의 일부라 더하지 않음"),
    ("est(31, 0, 36)  # 가상 단가 적용", str(est(31, 0, 36)), "호출 1회의 추정치 (가상 단가 · 청구액 아님)"),
])


# ------------------------------------------------------------------ 출력 · 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


if __name__ == "__main__":
    for _cid in ["d22-gate", "d22-rules", "d22-validator-bridge", "d22-inspect", "d22-wait-main",
                 "d22-wait-appendix", "d22-bounded", "d22-cache-read", "d22-routing"]:
        IO.pop(_cid, None)  # 명령어 사전에서 뺀 항목 (강의가 직접 만든 함수)
    for cid, (title, rows) in IO.items():
        print(f"== {cid}")
        for i, o, n in rows:
            print(f"   {i}\n      → {o}   ({n})")
    RESULTS["output_format"] = output_format
    RESULTS["k0"] = k0
    (Path(tempfile.gettempdir()) / "d22_results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, default=str), encoding="utf-8")
    frag = ROOT / "cmds" / "day22.html"
    if frag.exists():
        s = frag.read_text(encoding="utf-8")
        s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
        missing = []
        for cid, (title, rows) in IO.items():
            m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
            if not m:
                missing.append(cid)
                continue
            sec = m.group(0)
            kk = sec.find("</code></pre>\n")
            kk = kk + len("</code></pre>\n") if kk >= 0 else sec.find("</h3>\n") + len("</h3>\n")
            s = s[:m.start() + kk] + box(title, rows) + "\n" + s[m.start() + kk:]
        frag.write_text(s, encoding="utf-8")
        print("inserted", len(IO) - len(missing), "missing", missing)
