# Prompt와 출력 계약 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 모델을 부르는 부분은 ask() 하나예요. 여기서는 모델 대신 samples.json 의 미리 정한 응답을 돌려줘요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                        #@1
from pathlib import Path                                           #@1
import hashlib                                                     #@2
from string import Template                                        #@2
from typing import Annotated, Literal                              #@3
from pydantic import BaseModel, ConfigDict, Field                  #@3
from pydantic import ValidationError                               #@4

HERE = Path(__file__).resolve().parent                             #@1

# ① 근거 자료 고르기
KNOWLEDGE = [
    {"id": 101, "title": "실습 파일 보관", "content": "실습 파일은 개인 폴더에 보관합니다."},
    {"id": 102, "title": "서버 실행", "content": "프로젝트 폴더에서 서버를 실행합니다."},
    {"id": 103, "title": "질문 작성", "content": "오류 메시지와 시도한 작업을 함께 적습니다."},
]

def build_context(rows, selected_ids):
    by_id = {r["id"]: r for r in rows}
    ids = list(dict.fromkeys(selected_ids))  # 순서는 지키고 중복만 빼기
    if any(i not in by_id for i in ids):
        raise ValueError("선택한 ID가 없습니다")
    return json.dumps([{"id": i, "content": by_id[i]["content"]} for i in ids], ensure_ascii=False)

question = "파일은 어디에 보관하나요?"
provided_ids = [101]
context = build_context(KNOWLEDGE, provided_ids)

# ② Prompt 틀 채우기
TEMPLATE = Template("""[역할] 너는 실습 안내 도우미야.
[작업] 아래 자료만 근거로 질문에 답해.
[질문] $question
[자료] $context
[규칙] 자료에 없으면 추측하지 말고 status를 "insufficient_context"로 해.
[출력] JSON 하나만: {"status", "answer", "source_ids", "missing_information"}""")
prompt = TEMPLATE.substitute(question=question, context=context)
prompt_version = hashlib.sha256(TEMPLATE.template.encode("utf-8")).hexdigest()[:8]

# ③ 출력 계약 정하기
SourceID = Annotated[int, Field(strict=True, gt=0)]

class KnowledgeAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["answered", "insufficient_context"]
    answer: str = Field(min_length=1, max_length=500)
    source_ids: list[SourceID] = Field(max_length=3)
    missing_information: str = Field(max_length=200)

# ④ 응답 읽고 모양 검사하기
def parse_answer(raw):
    try:
        data = json.loads(raw)
    except ValueError:
        return None, "json_error"
    try:
        return KnowledgeAnswer.model_validate(data), None
    except ValidationError:
        return None, "schema_error"

# ⑤ 내용 규칙 검사하기
def check_rules(obj, provided_ids):
    ids = obj.source_ids
    if len(ids) != len(set(ids)):
        return "duplicate_source"
    if not set(ids) <= set(provided_ids):
        return "unknown_source"  # 주지 않은 자료를 근거로 댔다
    if obj.status == "answered" and (not ids or obj.missing_information):
        return "status_rule"
    if obj.status == "insufficient_context" and (ids or not obj.missing_information.strip()):
        return "status_rule"
    return None

def validate(raw):
    obj, error = parse_answer(raw)
    return (obj, error) if error else (obj, check_rules(obj, provided_ids))

# ⑥ 모양이 틀렸을 때만 한 번 다시 묻기
SAMPLES = json.loads((HERE / "samples.json").read_text(encoding="utf-8"))

def ask(prompt, script=iter(["broken", "good"])):  # 모델 대신: 미리 정한 응답을 순서대로
    return SAMPLES[next(script)]["raw"]

def run_with_retry(prompt):
    history = []
    for attempt in (1, 2):
        obj, error = validate(ask(prompt))
        history.append({"attempt": attempt, "error": error})
        if error is None:
            return obj, history
        if error not in {"json_error", "schema_error"}:
            break  # 내용 규칙 위반은 다시 물어도 같은 근거로 같은 답이 나오기 쉬워요
        prompt += f"\n\n[다시] 직전 응답이 {error}였어. JSON 형식만 지켜서 다시 답해."
    return None, history

final, history = run_with_retry(prompt)
