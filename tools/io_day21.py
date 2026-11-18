"""Day21 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의(20261116.html, Day21 Prompt·Context Engineering 인터랙티브 교안)의 코드 —
string.Template · build_context · json 직렬화 · SHA-256 · KnowledgeAnswer(Pydantic v2) ·
parse_answer · check_rules · final_rules · run_with_retry — 를 그대로 실행한다.
검사 입력은 교안 데모에 들어 있는 '교사가 작성한 샘플 응답(fixture)' 14개이고,
모델은 호출하지 않는다(교안도 모델을 부르지 않음). run_with_retry 의 generate 는
fixture 를 순서대로 돌려주는 가짜 함수다.
결과는 cmds/day21.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day21.py
"""
import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Annotated, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pydantic  # noqa: E402
import pytest  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, ValidationError  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IO = {}
RESULTS = {}   # 그림·실습 검증용


def err(fn):
    try:
        return repr(fn())
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


# ------------------------------------------------------------------ 버전
IO["d21-pip-install"] = ("실제 가상환경에서 확인", [
    ("import pydantic, pytest", f"pydantic {pydantic.VERSION} · pytest {pytest.__version__}",
     "string · json · hashlib · copy 는 Python 표준 라이브러리라 설치 없음"),
    ("python --version", sys.version.split()[0], "이 버전 기준 결과"),
])

# ------------------------------------------------------------------ 교안 고정 자료
KNOWLEDGE = [
    {"id": 101, "title": "실습 파일 보관", "content": "실습 파일은 개인 폴더에 보관합니다."},
    {"id": 102, "title": "서버 실행", "content": "프로젝트 폴더에서 서버를 실행합니다."},
    {"id": 103, "title": "질문 작성", "content": "오류 메시지와 시도한 작업을 함께 적습니다."},
]

# ------------------------------------------------------------------ Prompt 문자열
prompt = """[작업] 제공된 자료로 질문에 답하세요.
[질문] 파일은 어디에 보관하나요?
[자료] 101: 실습 파일은 개인 폴더에 보관합니다.
[조건] 자료에 없는 사실은 추측하지 마세요."""
IO["d21-prompt-basic"] = ("실제 실행", [
    ("print(prompt)", prompt.replace("\n", " ⏎ "), "줄바꿈(⏎)까지 그대로 — 모델에 보낼 입력 문자열일 뿐"),
    ("type(prompt).__name__, len(prompt)", f"{type(prompt).__name__}, {len(prompt)}",
     "그냥 str — print는 모델을 부르지 않아요"),
])

# ------------------------------------------------------------------ $ 자리표시자
question = "파일은 어디에 보관하나요?"
template = Template("질문: $question")
p1 = template.substitute(question=question)
IO["d21-placeholder"] = ("실제 실행 + 빠진 값 확인", [
    ('Template("질문: $question").substitute(question=question)', p1, "$question 자리에 값이 들어감"),
    ('Template("기호: $$").substitute()', Template("기호: $$").substitute(), "$ 글자 자체는 $$ 로"),
    ('Template("질문: ${question}입니다").substitute(question="위치")',
     Template("질문: ${question}입니다").substitute(question="위치"), "${이름}은 뒤에 글자가 바로 붙을 때"),
    ("template.substitute()  # 값 없이", err(lambda: template.substitute()), "빠진 값은 조용히 넘어가지 않고 KeyError"),
    ('Template(\'{"id": 101}\').substitute()', Template('{"id": 101}').substitute(),
     "JSON의 { }는 자리표시자가 아님 — 그래서 f-string 대신 Template을 골랐어요"),
])

# ------------------------------------------------------------------ 세 값 대입 템플릿
context_json = '[{"id":101,"content":"실습 파일은 개인 폴더에 보관합니다."}]'
contract_text = "status, answer, source_ids, missing_information의 JSON"
T = Template("""[작업]
제공 지식을 참고해 질문에 답하세요.
[질문]
$question
[지식 데이터]
$context_json
[출력]
$contract
""")
p20 = T.substitute(question=question, context_json=context_json, contract=contract_text)
IO["d21-template"] = ("실제 실행", [
    ("print(prompt)", p20.strip().replace("\n", " ⏎ "), "고정 문장은 그대로, $ 세 곳만 바뀜"),
    ("T.substitute(question=question, context_json=context_json)",
     err(lambda: T.substitute(question=question, context_json=context_json)), "contract를 빼면 KeyError: 'contract'"),
    ('T.substitute(question="서버는 어디에서 실행하나요?", …)',
     T.substitute(question="서버는 어디에서 실행하나요?", context_json=context_json, contract=contract_text).split("\n")[3],
     "같은 틀 · 다른 질문 → 질문 줄만 달라짐"),
])

# ------------------------------------------------------------------ 여섯 요소
parts = {
    "역할": "초보자의 실습 도움말 작성자",
    "작업": "질문에 한국어 2문장 이내로 답하기",
    "입력": "사용자 질문과 선택된 지식",
    "제약": "제공한 근거 범위에서만 답하기",
    "실패": "없는 사실은 추측하지 않고 정보 부족 표시",
    "출력": "status, answer, source_ids, missing_information의 JSON",
}
instruction = "\n\n".join(f"[{name}]\n{text}" for name, text in parts.items())
IO["d21-prompt-six"] = ("실제 실행", [
    ("print(instruction)  # 처음 두 구역", " ⏎ ".join(instruction.split("\n")[:5]), "구역 제목 + 내용, 빈 줄로 구분"),
    ("len(parts), instruction.count('[')", f"{len(parts)}, {instruction.count('[')}", "여섯 구역"),
])

# ------------------------------------------------------------------ Zero / Few-shot
zero_shot = """[지시] 제공 자료로 답하고 출처 ID를 표시하세요.
[자료] 101: 실습 파일은 개인 폴더에 보관합니다.
[질문] 파일은 어디에 보관하나요?
[출력] answer와 source_ids를 반환하세요."""
examples = """[예제 자료] 901: 대여 장소는 안내 데스크입니다.
[예제 질문 1] 어디서 대여하나요?
[바람직한 처리 1] 답변 가능 / 안내 데스크 / 출처 [901]
[예제 질문 2] 대여료는 얼마인가요?
[바람직한 처리 2] 정보 부족 / 필요한 정보: 대여료 / 출처 []"""
current_request = """[현재 자료] 101: 실습 파일은 개인 폴더에 보관합니다.
[현재 질문] 파일은 어디에 보관하나요?"""
few_shot = examples + "\n\n" + current_request
IO["d21-zero-few-shot"] = ("실제 실행", [
    ("zero_shot.count('[예제')", str(zero_shot.count("[예제")), "Zero-shot = 입력→출력 예제 0개 (자료·지시는 있음)"),
    ("few_shot.count('[예제 질문')", str(few_shot.count("[예제 질문")), "예제 2개: 답변 가능 1 · 정보 부족 1"),
    ("len(zero_shot), len(few_shot)", f"{len(zero_shot)}, {len(few_shot)}", "예제만큼 입력이 길어짐 = Context 공간을 씀"),
    ('"901" in current_request', str("901" in current_request), "현재 요청 구역에는 예제 ID가 없어야 해요"),
])

# ------------------------------------------------------------------ 스냅샷
rows = [{"id": 101, "content": "실습 파일은 개인 폴더에 보관합니다."}]
snapshot = deepcopy(rows)
shallow = list(rows)
rows[0]["content"] = "(DB가 나중에 바뀜)"
IO["d21-snapshot"] = ("실제 실행 + 원본을 바꿔 보기", [
    ('rows[0]["content"] = "(DB가 나중에 바뀜)" 후 snapshot[0]["content"]', snapshot[0]["content"], "deepcopy 스냅샷은 그대로"),
    ("같은 조건에서 shallow = list(rows) 였다면", shallow[0]["content"], "얕은 복사는 안쪽 dict를 공유 → 같이 바뀜"),
    ("snapshot is rows", str(snapshot is rows), "다른 객체"),
])

# ------------------------------------------------------------------ 직렬화
selected_rows = [{"id": 101, "content": "실습 파일은 개인 폴더에 보관합니다."}]
cj = json.dumps(selected_rows, ensure_ascii=False)
restored = json.loads(cj)
IO["d21-serialization"] = ("실제 실행", [
    ("type(context_json).__name__", type(cj).__name__, "직렬화 결과는 문자열"),
    ("print(context_json)", cj, "한글이 그대로 보임"),
    ("json.dumps(selected_rows)  # ensure_ascii 기본값", json.dumps(selected_rows)[:60] + "…", "기본은 \\uXXXX로 바뀜 — 사람이 읽기 어려움"),
    ("type(restored).__name__, restored == selected_rows", f"{type(restored).__name__}, {restored == selected_rows}", "역직렬화하면 다시 list · 내용 같음"),
])

# ------------------------------------------------------------------ build_context


def build_context(rows, selected_ids):
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("중복 ID를 확인하세요")
    ids = list(dict.fromkeys(selected_ids))
    if any(i not in by_id for i in ids):
        raise ValueError("선택한 ID가 없습니다")
    chosen = [by_id[i] for i in ids]
    return json.dumps(chosen, ensure_ascii=False)


small = [{"id": r["id"], "content": r["content"]} for r in KNOWLEDGE]
c102 = build_context(small, [102])
c_order = build_context(small, [102, 101, 102])
IO["d21-build-context"] = ("강의 함수 그대로 · 지식 101~103 (교안 고정 자료)", [
    ("build_context(knowledge, [102])", c102, "고른 1건만 — 101·103은 이번 출처 범위 밖"),
    ("build_context(knowledge, [102, 101, 102])  # ID만",
     str([r["id"] for r in json.loads(c_order)]), "선택 목록 중복은 제거 · 처음 나온 순서 유지(정렬 안 함)"),
    ("build_context(knowledge, [999])", err(lambda: build_context(small, [999])), "없는 ID를 조용히 빼지 않음"),
    ("build_context(knowledge + [knowledge[0]], [101])", err(lambda: build_context(small + [small[0]], [101])),
     "원본 rows 중복은 오류 — 선택 목록 중복과 다르게 처리"),
    ("build_context(knowledge, [])", build_context(small, []), "아무것도 안 골라도 오류는 아님 → 빈 문맥 '[]'"),
])

# ------------------------------------------------------------------ 예산
total, instruction_and_question, examples_tok, reserved_output, margin = 8192, 800, 600, 1600, 256
remaining = total - instruction_and_question - examples_tok - reserved_output - margin
rem2 = total - instruction_and_question - (examples_tok + 1000) - reserved_output - margin
rem3 = total - instruction_and_question - 6000 - reserved_output - margin
IO["d21-budget"] = ("실제 실행 · 숫자는 가상 예산 (실제 Tokenizer 측정값 아님)", [
    ("print(remaining)", str(remaining), "8192 − 800 − 600 − 1600 − 256"),
    ("examples = 1600 (예제 +1000)", str(rem2), "예제를 늘린 만큼 근거 공간이 줄어요"),
    ("examples = 6000", f"{rem3} → 예산 초과 메시지", "음수 = 근거를 하나도 못 넣음"),
])
RESULTS["budget"] = (total, instruction_and_question, examples_tok, reserved_output, margin, remaining)

# ------------------------------------------------------------------ 해시


def sha(rows_):
    t = json.dumps(rows_, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return t, hashlib.sha256(t.encode("utf-8")).hexdigest()


st, h1 = sha(selected_rows)
_, h_keys = sha([{"content": "실습 파일은 개인 폴더에 보관합니다.", "id": 101}])
_, h_order = sha([small[1], small[0]])
_, h_order2 = sha([small[0], small[1]])
_, h_edit = sha([{"id": 101, "content": "실습 파일은 개인 폴더에 보관합니다!"}])
IO["d21-hash"] = ("실제 실행 + 바꿔 보기", [
    ("snapshot_text", st, "sort_keys · 공백 없는 구분자 → 표현을 하나로 고정"),
    ("context_sha256", h1, "16진수 64글자"),
    ("dict 키 순서만 바꾼 같은 내용", f"{h_keys[:16]}… 같음={h_keys == h1}", "sort_keys=True 덕분에 같은 해시"),
    ("목록 순서 [102, 101] vs [101, 102]", f"같음={h_order == h_order2}", "목록 순서는 내용으로 취급 → 다른 해시"),
    ("마침표 하나를 '!'로", f"{h_edit[:16]}… 같음={h_edit == h1}", "한 글자만 달라도 완전히 다른 값"),
])

# ------------------------------------------------------------------ 실행 기록 · 지표
planned, executed, json_passed, human_judged, human_ok = 3, 2, 2, 1, 1
json_rate = json_passed / planned
human_rate = human_ok / human_judged if human_judged else None
IO["d21-metrics"] = ("실제 실행 · 분모 설명용 가상 기록 (실제 모델 성능 아님)", [
    ('print("JSON:", json_rate, "미실행:", planned - executed)', f"JSON: {json_rate} 미실행: {planned - executed}", "분모는 계획한 3건 — 2/3"),
    ('print("근거 적합률:", human_rate, "근거 미평가:", executed - human_judged)',
     f"근거 적합률: {human_rate} 근거 미평가: {executed - human_judged}", "1/1 = 100%지만 3건 중 1건만 본 결과"),
    ("human_judged = 0 이면 human_rate", repr((lambda hj: human_ok / hj if hj else None)(0)), "평가 안 했으면 0%가 아니라 None"),
])

# ------------------------------------------------------------------ JSON
raw = '{"answer": "개인 폴더에 보관합니다.", "source_ids": [101]}'
data = json.loads(raw)
IO["d21-json"] = ("실제 실행 + 자료형 비교", [
    ('print(data["answer"])', data["answer"], "JSON 텍스트 → dict"),
    ('type(data["source_ids"][0]).__name__', type(data["source_ids"][0]).__name__, "101 → int"),
    ("json.loads('[\"101\", true, 101.0]')  # 각 자료형",
     ", ".join(type(v).__name__ for v in json.loads('["101", true, 101.0]')), '"101" → str · true → bool · 101.0 → float'),
    ("json.loads(\"{'answer': 1}\")", err(lambda: json.loads("{'answer': 1}")), "작은따옴표는 JSON 문법이 아님"),
    ('json.loads(\'{"answer": "3년"}\')', repr(json.loads('{"answer": "3년"}')), "문법은 통과 — 사실인지와는 무관"),
])

# ------------------------------------------------------------------ Pydantic 계약
SourceID = Annotated[int, Field(strict=True, gt=0)]


class KnowledgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1, max_length=500)
    source_ids: list[SourceID] = Field(max_length=3)
    missing_information: str = Field(max_length=200)


schema = KnowledgeAnswer.model_json_schema()
good = {"status": "answered", "answer": "개인 폴더에 보관합니다.", "source_ids": [101], "missing_information": ""}
obj = KnowledgeAnswer.model_validate(good)
IO["d21-schema"] = ("실제 실행 · model_json_schema() 실제 출력", [
    ("schema['required']", str(schema["required"]), "네 필드 모두 필수"),
    ("schema['additionalProperties']", str(schema["additionalProperties"]), 'extra="forbid" → 추가 필드 금지'),
    ("schema['properties']['status']", json.dumps(schema["properties"]["status"], ensure_ascii=False), "Literal → enum"),
    ("schema['properties']['source_ids']", json.dumps(schema["properties"]["source_ids"], ensure_ascii=False),
     "gt=0 → exclusiveMinimum · max_length=3 → maxItems"),
    ("schema['properties']['answer']", json.dumps(schema["properties"]["answer"], ensure_ascii=False), "길이 제약이 명세로"),
])


def vfail(d):
    try:
        KnowledgeAnswer.model_validate(d)
        return "통과"
    except ValidationError as e:
        x = e.errors()[0]
        return f"{x['type']} @ {'.'.join(map(str, x['loc']))}"


IO["d21-pydantic"] = ("KnowledgeAnswer · model_validate 실제 결과", [
    ("KnowledgeAnswer.model_validate(정상 dict)", f"{type(obj).__name__}(status='{obj.status}', source_ids={obj.source_ids})", "객체가 만들어짐"),
    ('source_ids = ["101"]', vfail({**good, "source_ids": ["101"]}), "strict — 문자열을 정수로 바꿔 주지 않음"),
    ("source_ids = [True]", vfail({**good, "source_ids": [True]}), "bool도 int로 안 받음"),
    ("source_ids = [101.0]", vfail({**good, "source_ids": [101.0]}), "101.0은 float — strict에서는 거부"),
    ("source_ids = [0]", vfail({**good, "source_ids": [0]}), "gt=0 → 0 이하 거부"),
    ("source_ids = [999]", vfail({**good, "source_ids": [999]}), "구조상은 통과 — 이번에 준 ID인지는 다른 검사"),
    ("source_ids = [101, 102, 103, 104]", vfail({**good, "source_ids": [101, 102, 103, 104]}), "최대 3개"),
    ('confidence = 0.9 추가', vfail({**good, "confidence": 0.9}), "선언 안 한 필드 거부"),
    ('status = "unknown"', vfail({**good, "status": "unknown"}), "허용 상태 2개뿐"),
    ('answer = "   "', vfail({**good, "answer": "   "}), "공백 3칸도 길이 3 → 구조 검사는 통과 (뒤 규칙이 막음)"),
])
_lax = type("Lax", (BaseModel,), {"__annotations__": {"source_ids": list[int]}})
IO["d21-pydantic"][1].append(('strict 없이 list[int] 에 ["101"]', str(_lax.model_validate({"source_ids": ["101"]}).source_ids),
                              "strict를 빼면 '101'이 101로 조용히 바뀜 — 그래서 strict"))

# ------------------------------------------------------------------ 교안 fixture 14개
INSUFFICIENT = "제공된 지식으로는 답할 수 없습니다."


def answer_(text, ids):
    return {"status": "answered", "answer": text, "source_ids": ids, "missing_information": ""}


def insufficient_(missing):
    return {"status": "insufficient_context", "answer": INSUFFICIENT, "source_ids": [], "missing_information": missing}


def rawj(x):
    return json.dumps(x, ensure_ascii=False, indent=2)


A = "개인 폴더에 보관합니다."
FIXTURES = {
    "good": ("정상 답변", [101], rawj(answer_(A, [101]))),
    "missing": ("정상 정보 부족", [101], rawj(insufficient_("보관 기간"))),
    "broken": ("닫는 괄호 누락", [101], '{"status":"answered","answer":"개인 폴더에 보관합니다.","source_ids":[101],"missing_information":""'),
    "string_id": ('문자열 "101"', [101], rawj(answer_(A, ["101"]))),
    "bool_id": ("불리언 true", [101], rawj(answer_(A, [True]))),
    "float_id": ("실수 표기 101.0", [101], '{"status":"answered","answer":"개인 폴더에 보관합니다.","source_ids":[101.0],"missing_information":""}'),
    "unknown": ("제공하지 않은 999", [101], rawj(answer_(A, [999]))),
    "duplicate": ("출처 ID 중복", [101], rawj(answer_(A, [101, 101]))),
    "extra": ("추가 필드", [101], rawj({**answer_(A, [101]), "confidence": 0.9})),
    "status": ("상태 조합 위반", [101], rawj({**insufficient_("보관 위치"), "source_ids": [101]})),
    "blank": ("공백만 있는 answer", [101], rawj(answer_("   ", [101]))),
    "false_evidence": ("ID는 있으나 잘못된 인용", [101, 102, 103], rawj(answer_("프로젝트 폴더에서 서버를 실행합니다.", [101]))),
    "dupkey": ("JSON 키 중복", [101], '{"status":"answered","answer":"앞의 값","answer":"개인 폴더에 보관합니다.","source_ids":[101],"missing_information":""}'),
    "nan": ("비표준 NaN", [101], '{"status":"answered","answer":"개인 폴더에 보관합니다.","source_ids":[NaN],"missing_information":""}'),
}


def parse_answer(raw, load_json=json.loads):
    try:
        data = load_json(raw)
    except ValueError:
        return None, "json_error"
    try:
        obj = KnowledgeAnswer.model_validate(data)
    except ValidationError:
        return None, "schema_error"
    return obj, None


def check_rules(obj, provided_ids):
    ids = obj.source_ids
    if len(ids) != len(set(ids)):
        return "duplicate_source"
    if not set(ids) <= set(provided_ids):
        return "unknown_source"
    if obj.status == "answered":
        if not ids or obj.missing_information != "":
            return "status_rule"
    else:
        if ids or not obj.missing_information.strip():
            return "status_rule"
    return None


def final_rules(obj, provided_ids):
    error = check_rules(obj, provided_ids)
    if error:
        return error
    if not obj.answer.strip():
        return "blank_answer"
    if obj.status == "insufficient_context":
        if obj.answer != INSUFFICIENT:
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


def load_json_strict(raw):
    return json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)


def validate_answer(raw, provided_ids, load_json=json.loads):
    obj, error = parse_answer(raw, load_json=load_json)
    if error is not None:
        return None, error
    error = final_rules(obj, provided_ids)
    if error is not None:
        return None, error
    return obj, None


def show(x):
    return "None" if x is None else x


rows_parse, rows_rules, table = [], [], []
for key, (label, ids, r) in FIXTURES.items():
    o1, e1 = parse_answer(r)
    o2, e2 = parse_answer(r, load_json=load_json_strict)
    fin = final_rules(o1, ids) if o1 else "—"
    fin_s = final_rules(o2, ids) if o2 else "—"
    ov, ev = validate_answer(r, ids, load_json_strict)
    table.append({"key": key, "label": label, "ids": ids, "parse": show(e1), "parse_strict": show(e2),
                  "rules": show(fin), "final_strict": show(ev)})
    rows_parse.append((f"{label}", f"{'obj' if o1 else 'None'}, {show(e1)}", ""))
RESULTS["fixtures"] = table

note_parse = {
    "good": "문법·구조 통과 (근거는 아직)", "missing": "정보 부족도 같은 틀로 통과",
    "broken": "괄호 하나 → 문법 단계에서 멈춤", "string_id": 'strict라 "101" 거부',
    "bool_id": "true 거부", "float_id": "101.0 거부", "unknown": "999도 구조상은 정상 정수",
    "duplicate": "중복은 구조로는 못 막음", "extra": "추가 필드 거부", "status": "자료형은 다 맞음",
    "blank": "공백도 길이 3", "false_evidence": "구조 완벽", "dupkey": "기본 json.loads는 마지막 값으로 덮어씀 → 통과",
    "nan": "기본 json.loads는 NaN을 float로 읽음 → 구조 단계에서 거부",
}
IO["d21-parse-answer"] = ("json.loads → model_validate 두 단계 · 입력은 샘플 응답(fixture) 14개", [
    (lab, f"{obj_s}", note_parse[k]) for (lab, obj_s, _), k in zip(rows_parse, FIXTURES)
])

IO["d21-strict-json"] = ("콜백 두 개를 json.loads에 지정 · 같은 fixture", [
    ("JSON 키 중복 — json.loads", repr(json.loads(FIXTURES["dupkey"][2])["answer"]), "앞의 값이 조용히 사라짐"),
    ("JSON 키 중복 — 콜백 지정", err(lambda: load_json_strict(FIXTURES["dupkey"][2])), "중복을 오류로"),
    ("NaN — json.loads", repr(json.loads(FIXTURES["nan"][2])["source_ids"]), "Python 기본 파서는 NaN을 받아 줌(표준 JSON 아님)"),
    ("NaN — 콜백 지정", err(lambda: load_json_strict(FIXTURES["nan"][2])), "비표준 수치 거부"),
    ("키 중복 · 콜백 지정 → 위 try/except", str(parse_answer(FIXTURES["dupkey"][2], load_json_strict)[1]),
     "기본이면 통과(None) → 엄격이면 json_error"),
])

IO["d21-check-rules"] = ("강의 check_rules · final_rules 그대로 · 구조 검사를 통과한 fixture만", [
    (f"{FIXTURES[k][0]} · provided_ids={FIXTURES[k][1]}",
     str(next(t["rules"] for t in table if t["key"] == k)), n)
    for k, n in [
        ("good", "None = 자동 규칙 통과 (의미 검증 완료 아님)"),
        ("missing", "정보 부족 + 출처 [] + 필요한 정보 → 정상"),
        ("unknown", "999는 이번에 준 ID가 아님"),
        ("duplicate", "[101, 101]"),
        ("status", "정보 부족인데 출처 [101]"),
        ("blank", "공백만 있는 답"),
        ("false_evidence", "⚠️ 통과 — 101은 이번에 준 ID라서. 하지만 101은 '파일 보관' 자료"),
        ("dupkey", "기본 파서로 읽었다면 여기까지 통과"),
    ]
])

stage_count = {}
for t in table:
    stage_count[t["final_strict"]] = stage_count.get(t["final_strict"], 0) + 1
IO["d21-validate"] = ("validate_answer(raw, provided_ids, load_json_strict) · fixture 14개 실제 결과", [
    ("오류 코드별 개수", ", ".join(f"{k} {v}" for k, v in sorted(stage_count.items(), key=lambda kv: -kv[1])),
     "자동 검사를 통과한 None은 3개"),
    ("통과한 3개", ", ".join(t["label"] for t in table if t["final_strict"] == "None"),
     "그중 '잘못된 인용'은 사람 근거 검토에서만 걸러져요"),
])

# ------------------------------------------------------------------ pytest
test_src = f'''import json, pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Annotated, Literal
SourceID = Annotated[int, Field(strict=True, gt=0)]
class KnowledgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1, max_length=500)
    source_ids: list[SourceID] = Field(max_length=3)
    missing_information: str = Field(max_length=200)
@pytest.mark.parametrize("raw, expected", [
    ('{{"status":', json.JSONDecodeError),
    ('{{}}', ValidationError),
])
def test_invalid_outputs(raw, expected):
    with pytest.raises(expected):
        KnowledgeAnswer.model_validate(json.loads(raw))
'''
with tempfile.TemporaryDirectory() as td:
    tp = Path(td) / "test_answer.py"
    tp.write_text(test_src, encoding="utf-8")
    pr = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tp)],
                        capture_output=True, text=True, encoding="utf-8", cwd=td)
    last = [ln for ln in pr.stdout.strip().splitlines() if ln.strip()][-1]
    ids_out = subprocess.run([sys.executable, "-m", "pytest", "-q", "--collect-only", "-p", "no:cacheprovider", str(tp)],
                             capture_output=True, text=True, encoding="utf-8", cwd=td).stdout
    ids_list = [ln.split("::")[-1] for ln in ids_out.splitlines() if "::" in ln]
IO["d21-pytest"] = ("실제 pytest 실행", [
    ("python -m pytest -q test_answer.py", re.sub(r" in [\d.]+s", " in …s", last), "같은 함수에 실패 입력 2개를 대조"),
    ("--collect-only", ", ".join(ids_list), "parametrize가 케이스마다 테스트를 하나씩 만듦"),
])

# ------------------------------------------------------------------ 수정 요청
error_summary = "source_ids의 항목은 문자열이 아닌 정수여야 합니다."
repair_instruction = """[수정 요청]
같은 질문과 같은 지식으로 다시 출력하세요.
출력 계약은 변경하지 않습니다.
[확인된 오류]
""" + error_summary + """
[주의]
부족한 사실은 새로 만들지 마세요.
코드 블록 없이 JSON 객체 하나만 반환하세요.
출처는 이번에 제공한 ID만 사용하세요.
"""
IO["d21-repair"] = ("실제 실행", [
    ("print(repair_instruction)  # 줄 수", f"{len(repair_instruction.strip().splitlines())}줄", "오류 요약 한 줄 + 고정 문구"),
    ("'Traceback' in repair_instruction", str("Traceback" in repair_instruction), "전체 오류 기록·경로·비밀 값은 넣지 않음"),
])

# ------------------------------------------------------------------ 재시도


def run_with_retry(generate, prompt, validate):
    history = []
    for attempt in (1, 2):
        raw = generate(prompt)
        obj, error = validate(raw)
        history.append({"attempt": attempt, "error": error})
        if error is None:
            return obj, history
        if error not in {"json_error", "schema_error"}:
            break
        if attempt == 2:
            break
        prompt = repair_prompt(prompt, error)
    return None, history


def repair_prompt(prompt, error):   # 이 노트에서 붙인 최소 구현 (교안은 이름만 연결)
    return prompt + "\n\n" + repair_instruction


def scripted(*keys):
    seq = iter([FIXTURES[k][2] for k in keys])
    calls = []

    def generate(p):
        calls.append(p)
        return next(seq)
    return generate, calls


def scenario(keys, ids=(101,)):
    gen, calls = scripted(*keys)
    o, h = run_with_retry(gen, "PROMPT", lambda r: validate_answer(r, list(ids), load_json_strict))
    return ("obj" if o else "None"), h, len(calls), ("[수정 요청]" in calls[-1])


sc = {}
for name, keys in [("ok", ["good"]), ("missing", ["missing"]), ("fix", ["broken", "good"]),
                   ("twice", ["broken", "string_id"]), ("rule", ["unknown", "good"])]:
    sc[name] = scenario(keys)
RESULTS["retry"] = sc


def hs(h):
    return str(h).replace("'", '"').replace('"error": None', '"error": None')


IO["d21-retry"] = ("강의 run_with_retry 그대로 · generate = fixture를 차례로 돌려주는 가짜 함수 (모델 호출 없음)", [
    ("응답: 정상", f"{sc['ok'][0]}, 호출 {sc['ok'][2]}회", "1회에 통과 → 근거 검토로"),
    ("응답: 정상 정보 부족", f"{sc['missing'][0]}, 호출 {sc['missing'][2]}회", "정보 부족도 정상 종료 — 다시 묻지 않음"),
    ("응답: 괄호 누락 → 정상", f"{sc['fix'][0]}, {hs(sc['fix'][1])}", f"형식 오류 → 수정 요청 1회 (2번째 입력에 수정 요청 포함={sc['fix'][3]})"),
    ('응답: 괄호 누락 → 문자열 "101"', f"{sc['twice'][0]}, {hs(sc['twice'][1])}", "두 번 다 형식 실패 → None (최대 2회)"),
    ("응답: 제공하지 않은 999 (다음 응답은 정상이었더라도)", f"{sc['rule'][0]}, 호출 {sc['rule'][2]}회, {hs(sc['rule'][1])}",
     "규칙 위반은 자동 재생성 안 함 — 바로 중단"),
])

history = [{"attempt": 1, "error": "json_error"}, {"attempt": 2, "error": None}]
IO["d21-retry-log"] = ("실제 실행", [
    ('print("첫 응답:", first_attempt["error"])', f'첫 응답: {history[0]["error"]}', "첫 실패를 지우지 않음"),
    ('print("수정 시도 수:", len(repair_attempts))', f"수정 시도 수: {len(history[1:])}", "수정 성공과 첫 응답 품질을 따로 셈"),
])


# ------------------------------------------------------------------ 출력 · 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


if __name__ == "__main__":
    for _cid in ["d21-build-context", "d21-check-rules", "d21-validate", "d21-retry"]:
        IO.pop(_cid, None)  # 명령어 사전에서 뺀 항목 (강의가 직접 만든 함수)
    for cid, (title, rows) in IO.items():
        print(f"== {cid}")
        for i, o, n in rows:
            print(f"   {i}\n      → {o}   ({n})")
    print("== FIXTURE TABLE")
    for t in table:
        print("  ", t)
    (Path(tempfile.gettempdir()) / "d21_results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, default=str), encoding="utf-8")
    frag = ROOT / "cmds" / "day21.html"
    if frag.exists():
        s = frag.read_text(encoding="utf-8")
        s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
        missing = []
        for cid, (title, rows) in IO.items():
            m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
            if not m:
                missing.append(cid); continue
            sec = m.group(0)
            kk = sec.find("</code></pre>\n")
            kk = kk + len("</code></pre>\n") if kk >= 0 else sec.find("</h3>\n") + len("</h3>\n")
            s = s[:m.start() + kk] + box(title, rows) + "\n" + s[m.start() + kk:]
        frag.write_text(s, encoding="utf-8")
        print("inserted", len(IO) - len(missing), "missing", missing)
