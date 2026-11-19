window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import hashlib",
   "s": 2
  },
  {
   "t": "from string import Template",
   "s": 2
  },
  {
   "t": "from typing import Annotated, Literal",
   "s": 3
  },
  {
   "t": "from pydantic import BaseModel, ConfigDict, Field",
   "s": 3
  },
  {
   "t": "from pydantic import ValidationError",
   "s": 4
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
   "t": "# ① 근거 자료 고르기",
   "s": 1
  },
  {
   "t": "KNOWLEDGE = [",
   "s": 1
  },
  {
   "t": "    {\"id\": 101, \"title\": \"실습 파일 보관\", \"content\": \"실습 파일은 개인 폴더에 보관합니다.\"},",
   "s": 1
  },
  {
   "t": "    {\"id\": 102, \"title\": \"서버 실행\", \"content\": \"프로젝트 폴더에서 서버를 실행합니다.\"},",
   "s": 1
  },
  {
   "t": "    {\"id\": 103, \"title\": \"질문 작성\", \"content\": \"오류 메시지와 시도한 작업을 함께 적습니다.\"},",
   "s": 1
  },
  {
   "t": "]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def build_context(rows, selected_ids):",
   "s": 1
  },
  {
   "t": "    by_id = {r[\"id\"]: r for r in rows}",
   "s": 1
  },
  {
   "t": "    ids = list(dict.fromkeys(selected_ids))  # 순서는 지키고 중복만 빼기",
   "s": 1
  },
  {
   "t": "    if any(i not in by_id for i in ids):",
   "s": 1
  },
  {
   "t": "        raise ValueError(\"선택한 ID가 없습니다\")",
   "s": 1
  },
  {
   "t": "    return json.dumps([{\"id\": i, \"content\": by_id[i][\"content\"]} for i in ids], ensure_ascii=False)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "question = \"파일은 어디에 보관하나요?\"",
   "s": 1
  },
  {
   "t": "provided_ids = [101]",
   "s": 1
  },
  {
   "t": "context = build_context(KNOWLEDGE, provided_ids)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② Prompt 틀 채우기",
   "s": 2
  },
  {
   "t": "TEMPLATE = Template(\"\"\"[역할] 너는 실습 안내 도우미야.",
   "s": 2
  },
  {
   "t": "[작업] 아래 자료만 근거로 질문에 답해.",
   "s": 2
  },
  {
   "t": "[질문] $question",
   "s": 2
  },
  {
   "t": "[자료] $context",
   "s": 2
  },
  {
   "t": "[규칙] 자료에 없으면 추측하지 말고 status를 \"insufficient_context\"로 해.",
   "s": 2
  },
  {
   "t": "[출력] JSON 하나만: {\"status\", \"answer\", \"source_ids\", \"missing_information\"}\"\"\")",
   "s": 2
  },
  {
   "t": "prompt = TEMPLATE.substitute(question=question, context=context)",
   "s": 2
  },
  {
   "t": "prompt_version = hashlib.sha256(TEMPLATE.template.encode(\"utf-8\")).hexdigest()[:8]",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 출력 계약 정하기",
   "s": 3
  },
  {
   "t": "SourceID = Annotated[int, Field(strict=True, gt=0)]",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "class KnowledgeAnswer(BaseModel):",
   "s": 3
  },
  {
   "t": "    model_config = ConfigDict(extra=\"forbid\", strict=True)",
   "s": 3
  },
  {
   "t": "    status: Literal[\"answered\", \"insufficient_context\"]",
   "s": 3
  },
  {
   "t": "    answer: str = Field(min_length=1, max_length=500)",
   "s": 3
  },
  {
   "t": "    source_ids: list[SourceID] = Field(max_length=3)",
   "s": 3
  },
  {
   "t": "    missing_information: str = Field(max_length=200)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 응답 읽고 모양 검사하기",
   "s": 4
  },
  {
   "t": "def parse_answer(raw):",
   "s": 4
  },
  {
   "t": "    try:",
   "s": 4
  },
  {
   "t": "        data = json.loads(raw)",
   "s": 4
  },
  {
   "t": "    except ValueError:",
   "s": 4
  },
  {
   "t": "        return None, \"json_error\"",
   "s": 4
  },
  {
   "t": "    try:",
   "s": 4
  },
  {
   "t": "        return KnowledgeAnswer.model_validate(data), None",
   "s": 4
  },
  {
   "t": "    except ValidationError:",
   "s": 4
  },
  {
   "t": "        return None, \"schema_error\"",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 내용 규칙 검사하기",
   "s": 5
  },
  {
   "t": "def check_rules(obj, provided_ids):",
   "s": 5
  },
  {
   "t": "    ids = obj.source_ids",
   "s": 5
  },
  {
   "t": "    if len(ids) != len(set(ids)):",
   "s": 5
  },
  {
   "t": "        return \"duplicate_source\"",
   "s": 5
  },
  {
   "t": "    if not set(ids) <= set(provided_ids):",
   "s": 5
  },
  {
   "t": "        return \"unknown_source\"  # 주지 않은 자료를 근거로 댔다",
   "s": 5
  },
  {
   "t": "    if obj.status == \"answered\" and (not ids or obj.missing_information):",
   "s": 5
  },
  {
   "t": "        return \"status_rule\"",
   "s": 5
  },
  {
   "t": "    if obj.status == \"insufficient_context\" and (ids or not obj.missing_information.strip()):",
   "s": 5
  },
  {
   "t": "        return \"status_rule\"",
   "s": 5
  },
  {
   "t": "    return None",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def validate(raw):",
   "s": 5
  },
  {
   "t": "    obj, error = parse_answer(raw)",
   "s": 5
  },
  {
   "t": "    return (obj, error) if error else (obj, check_rules(obj, provided_ids))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 모양이 틀렸을 때만 한 번 다시 묻기",
   "s": 6
  },
  {
   "t": "SAMPLES = json.loads((HERE / \"samples.json\").read_text(encoding=\"utf-8\"))",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def ask(prompt, script=iter([\"broken\", \"good\"])):  # 모델 대신: 미리 정한 응답을 순서대로",
   "s": 6
  },
  {
   "t": "    return SAMPLES[next(script)][\"raw\"]",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def run_with_retry(prompt):",
   "s": 6
  },
  {
   "t": "    history = []",
   "s": 6
  },
  {
   "t": "    for attempt in (1, 2):",
   "s": 6
  },
  {
   "t": "        obj, error = validate(ask(prompt))",
   "s": 6
  },
  {
   "t": "        history.append({\"attempt\": attempt, \"error\": error})",
   "s": 6
  },
  {
   "t": "        if error is None:",
   "s": 6
  },
  {
   "t": "            return obj, history",
   "s": 6
  },
  {
   "t": "        if error not in {\"json_error\", \"schema_error\"}:",
   "s": 6
  },
  {
   "t": "            break  # 내용 규칙 위반은 다시 물어도 같은 근거로 같은 답이 나오기 쉬워요",
   "s": 6
  },
  {
   "t": "        prompt += f\"\\n\\n[다시] 직전 응답이 {error}였어. JSON 형식만 지켜서 다시 답해.\"",
   "s": 6
  },
  {
   "t": "    return None, history",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "final, history = run_with_retry(prompt)",
   "s": 6
  }
 ],
 "names": {
  "1": "근거 자료 고르기",
  "2": "Prompt 틀 채우기",
  "3": "출력 계약 정하기",
  "4": "응답 읽고 모양 검사하기",
  "5": "내용 규칙 검사하기",
  "6": "모양이 틀렸을 때만 한 번 다시 묻기"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "why": {
   "title": "모델이 낼 수 있는 응답 여덟 가지를 검사하면",
   "cols": [
    "응답",
    "모델이 보낸 글",
    "결과",
    "걸린 곳"
   ],
   "rows": [
    [
     "정상 답변",
     "<code>{&quot;status&quot;: &quot;answered&quot;, &quot;answer&quot;: &quot;개인 폴더에 보관합니다.&quot;, &quot;source_ids&quot;: [101],…</code>",
     "<b>통과</b>",
     "통과"
    ],
    [
     "정보 부족을 제대로 알림",
     "<code>{&quot;status&quot;: &quot;insufficient_context&quot;, &quot;answer&quot;: &quot;제공된 자료로는 답할 수 없습니다.&quot;, &quot;s…</code>",
     "<b>통과</b>",
     "통과"
    ],
    [
     "닫는 괄호 빠짐",
     "<code>{&quot;status&quot;:&quot;answered&quot;,&quot;answer&quot;:&quot;개인 폴더에 보관합니다.&quot;,&quot;source_ids&quot;:[101],&quot;miss…</code>",
     "<span class=\"bad\">json_error</span>",
     "④ JSON 읽기"
    ],
    [
     "출처 번호를 글자 &quot;101&quot;로",
     "<code>{&quot;status&quot;: &quot;answered&quot;, &quot;answer&quot;: &quot;개인 폴더에 보관합니다.&quot;, &quot;source_ids&quot;: [&quot;101&quot;…</code>",
     "<span class=\"bad\">schema_error</span>",
     "④ 모양 검사"
    ],
    [
     "없는 필드 confidence 추가",
     "<code>{&quot;status&quot;: &quot;answered&quot;, &quot;answer&quot;: &quot;개인 폴더에 보관합니다.&quot;, &quot;source_ids&quot;: [101],…</code>",
     "<span class=\"bad\">schema_error</span>",
     "④ 모양 검사"
    ],
    [
     "주지 않은 자료 999를 출처로",
     "<code>{&quot;status&quot;: &quot;answered&quot;, &quot;answer&quot;: &quot;개인 폴더에 보관합니다.&quot;, &quot;source_ids&quot;: [999],…</code>",
     "<span class=\"bad\">unknown_source</span>",
     "⑤ 내용 규칙"
    ],
    [
     "출처 번호 중복",
     "<code>{&quot;status&quot;: &quot;answered&quot;, &quot;answer&quot;: &quot;개인 폴더에 보관합니다.&quot;, &quot;source_ids&quot;: [101, …</code>",
     "<span class=\"bad\">duplicate_source</span>",
     "⑤ 내용 규칙"
    ],
    [
     "정보 부족이라면서 출처를 댐",
     "<code>{&quot;status&quot;: &quot;insufficient_context&quot;, &quot;answer&quot;: &quot;제공된 자료로는 답할 수 없습니다.&quot;, &quot;s…</code>",
     "<span class=\"bad\">status_rule</span>",
     "⑤ 내용 규칙"
    ]
   ],
   "notes": [
    "모델의 답은 언제나 <b>그냥 글자</b>예요. 괄호 하나 빠지거나, 번호를 글자로 쓰거나, 주지 않은 자료를 근거로 대는 일이 실제로 생겨요. 위 응답들은 그런 실수를 흉내 내 미리 만든 샘플이에요.",
    "검사 없이 화면에 보여 주면 이 중 여섯 개가 사용자에게 그대로 나가요."
   ]
  },
  "s1": {
   "title": "build_context(KNOWLEDGE, [101])",
   "lines": [
    [
     "context → [{\"id\": 101, \"content\": \"실습 파일은 개인 폴더에 보관합니다.\"}]",
     1
    ],
    "question → 파일은 어디에 보관하나요?"
   ],
   "notes": [
    "세 자료 중 고른 101번만, <b>번호와 함께</b> 넣었어요. 번호가 있어야 모델이 \"어느 자료를 근거로 했는지\" 답할 수 있어요."
   ]
  },
  "t1": {
   "title": "고르는 번호가 이상하면",
   "lines": [
    "build_context(..., [102, 101, 102]) → [{\"id\": 102, \"content\": \"프로젝트 폴더에서 서버를 실행합니다.\"}, {\"id\": 101, \"content\": \"실습 파일은 개인 폴더에 보관합니다.\"}]",
    [
     "build_context(..., [101, 999]) → ValueError: 선택한 ID가 없습니다",
     1
    ]
   ],
   "notes": [
    "중복은 순서를 지키며 빼고, 없는 번호는 Prompt를 만들기 전에 멈춰요. 잘못된 자료로 모델을 부르면 비용만 쓰고 답도 틀려요."
   ]
  },
  "s2": {
   "title": "prompt (모델에 보낼 글 전체)",
   "lines": [
    "[역할] 너는 실습 안내 도우미야.",
    "[작업] 아래 자료만 근거로 질문에 답해.",
    "[질문] 파일은 어디에 보관하나요?",
    "[자료] [{\"id\": 101, \"content\": \"실습 파일은 개인 폴더에 보관합니다.\"}]",
    "[규칙] 자료에 없으면 추측하지 말고 status를 \"insufficient_context\"로 해.",
    "[출력] JSON 하나만: {\"status\", \"answer\", \"source_ids\", \"missing_information\"}",
    "",
    [
     "prompt_version → f72107e9",
     1
    ]
   ],
   "notes": [
    "틀(<code>TEMPLATE</code>)의 지문 앞 8자리. 틀을 한 글자라도 고치면 바뀌어서, 나중에 \"이 답은 어느 판의 Prompt로 받았나\"를 기록할 수 있어요."
   ]
  },
  "t2": {
   "title": "틀에 \"친절한\" 한 단어를 넣으면",
   "lines": [
    "고치기 전 → f72107e9",
    [
     "고친 뒤   → 01c129b7",
     1
    ]
   ],
   "notes": [
    "완전히 다른 값이에요. Prompt를 바꿔 결과가 좋아졌는지 비교할 때(A/B), 판 번호로 어느 쪽 결과인지 구분해요."
   ]
  },
  "s3": {
   "title": "KnowledgeAnswer.model_json_schema() 중 일부",
   "lines": [
    [
     "required → ['status', 'answer', 'source_ids', 'missing_information']",
     1
    ],
    [
     "additionalProperties → False",
     2
    ],
    [
     "status → {\"enum\": [\"answered\", \"insufficient_context\"], \"title\": \"Status\", \"type\": \"string\"}",
     3
    ],
    "source_ids → {\"items\": {\"exclusiveMinimum\": 0, \"type\": \"integer\"}, \"maxItems\": 3, \"title\": \"Source Ids\", \"type\": \"array\"}"
   ],
   "notes": [
    "네 필드 모두 꼭 있어야 해요.",
    "<code>extra=\"forbid\"</code> 때문에 다른 필드는 금지예요.",
    "<code>Literal</code>은 \"이 두 값 중 하나\"가 돼요. 이 스키마를 그대로 모델에게 주면 그 모양으로 답하게 할 수도 있어요 (<a href=\"llm-client.html\">LLM API 클라이언트</a>의 구조화 출력)."
   ]
  },
  "s4": {
   "title": "parse_answer() 실행 결과",
   "cols": [
    "응답",
    "결과",
    "Pydantic이 알려 준 이유"
   ],
   "rows": [
    [
     "정상 답변",
     "<b>통과</b>",
     ""
    ],
    [
     "닫는 괄호 빠짐",
     "<span class=\"bad\">json_error</span>",
     ""
    ],
    [
     "출처 번호를 글자 &quot;101&quot;로",
     "<span class=\"bad\">schema_error</span>",
     "(&#x27;source_ids&#x27;, 0): Input should be a valid integer"
    ],
    [
     "없는 필드 confidence 추가",
     "<span class=\"bad\">schema_error</span>",
     "(&#x27;confidence&#x27;,): Extra inputs are not permitted"
    ]
   ],
   "notes": [
    "\"101\"(글자)과 101(숫자)은 다른 값이에요. <code>strict=True</code>라서 글자를 숫자로 몰래 바꿔 주지 않고 거부해요.",
    "오류를 <code>json_error</code>, <code>schema_error</code>처럼 <b>종류별 이름</b>으로 돌려줘서, ⑥에서 다시 물을지 정할 수 있어요."
   ]
  },
  "s5": {
   "title": "모양 검사를 통과한 응답에 check_rules()",
   "cols": [
    "응답",
    "source_ids",
    "status",
    "결과"
   ],
   "rows": [
    [
     "정상 답변",
     "<code>[101]</code>",
     "answered",
     "<b>통과</b>"
    ],
    [
     "정보 부족을 제대로 알림",
     "<code>[]</code>",
     "insufficient_context",
     "<b>통과</b>"
    ],
    [
     "주지 않은 자료 999를 출처로",
     "<code>[999]</code>",
     "answered",
     "<span class=\"bad\">unknown_source</span>"
    ],
    [
     "출처 번호 중복",
     "<code>[101, 101]</code>",
     "answered",
     "<span class=\"bad\">duplicate_source</span>"
    ],
    [
     "정보 부족이라면서 출처를 댐",
     "<code>[101]</code>",
     "insufficient_context",
     "<span class=\"bad\">status_rule</span>"
    ]
   ],
   "notes": [
    "모두 모양은 맞아서 ④는 통과했어요. 하지만 \"주지 않은 999번을 근거로 댄\" 답은 그럴듯해 보여도 출처를 지어낸 거예요. 이런 건 <b>내용 규칙</b>으로만 잡혀요.",
    "규칙의 기준은 ①에서 실제로 넣은 번호(<code>provided_ids</code>)예요."
   ]
  },
  "s6": {
   "title": "run_with_retry(prompt)  (첫 응답: 괄호 빠짐, 두 번째: 정상)",
   "lines": [
    [
     "history → [{'attempt': 1, 'error': 'json_error'}, {'attempt': 2, 'error': None}]",
     1
    ],
    [
     "final → KnowledgeAnswer(status='answered', answer='개인 폴더에 보관합니다.', source_ids=[101], missing_information='')",
     2
    ]
   ],
   "notes": [
    "첫 응답은 JSON이 깨져서(<code>json_error</code>) 무엇이 틀렸는지 덧붙여 한 번 더 물었어요.",
    "두 번째 응답은 모든 검사를 통과했어요."
   ]
  },
  "t6": {
   "title": "첫 응답이 '주지 않은 자료 999를 출처로'였다면",
   "lines": [
    [
     "history → [{'attempt': 1, 'error': 'unknown_source'}]",
     1
    ],
    "final → None"
   ],
   "notes": [
    "내용 규칙 위반은 다시 묻지 않고 바로 멈췄어요. 같은 자료로 다시 물으면 같은 착각을 하기 쉬워서, 이런 건 사람이 보거나 자료를 바꿔야 해요. 몇 번이고 다시 물으면 비용만 늘어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "자료",
   "골라 넣기",
   "번호와 함께"
  ],
  [
   2,
   "Prompt",
   "틀 채우기",
   "판 번호"
  ],
  [
   3,
   "계약",
   "Pydantic",
   "답의 모양"
  ],
  [
   4,
   "모양 검사",
   "JSON, 스키마",
   "json_error, schema_error"
  ],
  [
   5,
   "내용 검사",
   "출처, 상태",
   "unknown_source 등"
  ],
  [
   6,
   "다시 묻기",
   "모양만 1번",
   "내용 위반은 멈춤"
  ]
 ],
 "env": "Python 3.13.16, Pydantic 2.14.3. 모델은 부르지 않고 samples.json 의 미리 정한 응답 사용"
};
