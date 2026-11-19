window.PAGE = {
 "code": [
  {
   "t": "import os",
   "s": 1
  },
  {
   "t": "from openai import OpenAI",
   "s": 1
  },
  {
   "t": "from typing import Literal",
   "s": 3
  },
  {
   "t": "from pydantic import BaseModel, Field",
   "s": 3
  },
  {
   "t": "import openai",
   "s": 4
  },
  {
   "t": "from decimal import Decimal",
   "s": 6
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 클라이언트 만들기",
   "s": 1
  },
  {
   "t": "MODEL = os.environ[\"OPENAI_MODEL\"]",
   "s": 1
  },
  {
   "t": "client = OpenAI(timeout=30.0, max_retries=2)  # 키는 환경 변수 OPENAI_API_KEY 에서 읽어요",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 질문 보내고 답 읽기",
   "s": 2
  },
  {
   "t": "question = \"API를 한 문장으로 설명해 줘.\"",
   "s": 2
  },
  {
   "t": "response = client.responses.create(",
   "s": 2
  },
  {
   "t": "    model=MODEL,",
   "s": 2
  },
  {
   "t": "    instructions=\"초보자에게 한국어로 짧게 설명해.\",",
   "s": 2
  },
  {
   "t": "    input=question,",
   "s": 2
  },
  {
   "t": ")",
   "s": 2
  },
  {
   "t": "answer = response.output_text",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 정해진 모양으로 받기 (구조화 출력)",
   "s": 3
  },
  {
   "t": "class Explain(BaseModel):",
   "s": 3
  },
  {
   "t": "    term: str",
   "s": 3
  },
  {
   "t": "    one_line: str = Field(max_length=120)",
   "s": 3
  },
  {
   "t": "    level: Literal[\"easy\", \"normal\"]",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "parsed = client.responses.parse(",
   "s": 3
  },
  {
   "t": "    model=MODEL,",
   "s": 3
  },
  {
   "t": "    input=question,",
   "s": 3
  },
  {
   "t": "    text_format=Explain,",
   "s": 3
  },
  {
   "t": ").output_parsed",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 실패를 종류별로 다루기",
   "s": 4
  },
  {
   "t": "def ask(text):",
   "s": 4
  },
  {
   "t": "    try:",
   "s": 4
  },
  {
   "t": "        r = client.responses.create(model=MODEL, input=text)",
   "s": 4
  },
  {
   "t": "        return \"ok\", r.output_text",
   "s": 4
  },
  {
   "t": "    except openai.AuthenticationError:",
   "s": 4
  },
  {
   "t": "        return \"stop\", \"키를 확인하세요\"  # 다시 보내도 같은 결과",
   "s": 4
  },
  {
   "t": "    except openai.RateLimitError as e:",
   "s": 4
  },
  {
   "t": "        return \"later\", e.code  # rate_limit_exceeded 면 잠시 뒤, insufficient_quota 면 결제 확인",
   "s": 4
  },
  {
   "t": "    except openai.APITimeoutError:",
   "s": 4
  },
  {
   "t": "        return \"later\", \"timeout\"",
   "s": 4
  },
  {
   "t": "    except openai.APIStatusError as e:",
   "s": 4
  },
  {
   "t": "        return \"error\", e.status_code",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 나오는 대로 받기 (스트리밍)",
   "s": 5
  },
  {
   "t": "pieces = []",
   "s": 5
  },
  {
   "t": "with client.responses.stream(model=MODEL, input=question) as stream:",
   "s": 5
  },
  {
   "t": "    for event in stream:",
   "s": 5
  },
  {
   "t": "        if event.type == \"response.output_text.delta\":",
   "s": 5
  },
  {
   "t": "            pieces.append(event.delta)  # 화면에 바로 이어 붙여 보여 줄 수 있어요",
   "s": 5
  },
  {
   "t": "    final = stream.get_final_response()",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 사용량과 비용 기록하기",
   "s": 6
  },
  {
   "t": "PRICE = {\"input\": Decimal(\"0.75\"), \"output\": Decimal(\"4.50\")}  # 예시 단가: 100만 토큰당 달러",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "usage = response.usage",
   "s": 6
  },
  {
   "t": "cost = (usage.input_tokens * PRICE[\"input\"] + usage.output_tokens * PRICE[\"output\"]) / 1_000_000",
   "s": 6
  }
 ],
 "names": {
  "1": "클라이언트 만들기",
  "2": "질문 보내고 답 읽기",
  "3": "정해진 모양으로 받기 (구조화 출력)",
  "4": "실패를 종류별로 다루기",
  "5": "나오는 대로 받기 (스트리밍)",
  "6": "사용량과 비용 기록하기"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    [
     "client.base_url → https://api.openai.com/v1/",
     1
    ],
    "client.timeout → 30.0",
    [
     "client.max_retries → 2",
     2
    ],
    "키가 설정됐나 → True"
   ],
   "notes": [
    "따로 정하지 않으면 OpenAI 서버 주소를 써요.",
    "실패하면 SDK가 <b>알아서</b> 최대 2번 더 보내요. ④에서 실제 횟수를 봐요."
   ]
  },
  "s2": {
   "title": "SDK가 실제로 만든 요청과 받은 응답",
   "lines": [
    [
     "POST https://api.openai.com/v1/responses",
     1
    ],
    "Authorization: Bearer sk-…",
    [
     "본문 → {\"model\": \"gpt-5.4-mini\", \"input\": \"API를 한 문장으로 설명해 줘.\", \"instructions\": \"초보자에게 한국어로 짧게 설명해.\"}",
     2
    ],
    "",
    [
     "response.output_text → API는 서로 다른 프로그램들이 정해진 규칙으로 데이터를 주고받고 기능을 이용하게 해 주는 “중간 다리”입니다.",
     3
    ],
    "response.usage → 입력 31토큰, 출력 36토큰"
   ],
   "notes": [
    "Responses API의 주소예요. 모델 이름은 주소가 아니라 본문에 들어가요.",
    "<code>create()</code>에 준 인자가 그대로 JSON 본문이 됐어요. 키는 본문이 아니라 머리글(<code>Authorization</code>)로 가요.",
    "응답 안의 여러 조각에서 글만 모아 주는 편의 속성이에요."
   ]
  },
  "s3": {
   "title": "responses.parse(text_format=Explain)",
   "lines": [
    [
     "요청 본문의 text.format → type=json_schema, name=Explain, strict=True",
     1
    ],
    [
     "schema.required → ['term', 'one_line', 'level']",
     2
    ],
    "schema.properties.level → {\"enum\": [\"easy\", \"normal\"], \"title\": \"Level\", \"type\": \"string\"}",
    "",
    [
     "output_parsed → Explain(term='API', one_line='프로그램끼리 정해진 규칙으로 기능과 데이터를 주고받는 창구', level='easy')",
     3
    ]
   ],
   "notes": [
    "SDK가 Pydantic 클래스를 JSON 스키마로 바꿔 요청에 실었어요. <code>strict</code>라서 모델은 이 모양으로만 답해요.",
    "필드 셋 모두 필수예요.",
    "돌아온 JSON 글을 SDK가 <code>Explain</code> 객체로 바꿔 줬어요. 내용이 맞는지(출처 등)는 여전히 따로 검사해요 (<a href=\"prompt-contract.html\">Prompt와 출력 계약</a>)."
   ]
  },
  "s4": {
   "title": "ask()를 상황별 가짜 서버에 보내면 (max_retries=2)",
   "cols": [
    "상황",
    "ask()가 돌려준 값",
    "SDK가 실제로 보낸 횟수"
   ],
   "rows": [
    [
     "키가 틀림 (401)",
     "<code>stop, 키를 확인하세요</code>",
     "<b>1번</b>"
    ],
    [
     "너무 자주 보냄 (429)",
     "<code>later, rate_limit_exceeded</code>",
     "<b>3번</b>"
    ],
    [
     "할당량 다 씀 (429)",
     "<code>later, insufficient_quota</code>",
     "<b>3번</b>"
    ],
    [
     "서버 오류 (500)",
     "<code>error, 500</code>",
     "<b>3번</b>"
    ],
    [
     "응답이 늦음 (시간 초과)",
     "<code>later, timeout</code>",
     "<b>3번</b>"
    ]
   ],
   "notes": [
    "401은 다시 보내도 같아서 SDK도 한 번만 보내요. 429, 500, 시간 초과는 SDK가 <b>앱 모르게 3번</b>까지 보내요.",
    "두 429는 예외 종류(<code>RateLimitError</code>)가 같아요. \"너무 자주\"는 기다리면 풀리지만 \"할당량 다 씀\"은 결제를 확인해야 해서, <code>e.code</code>로 갈라야 해요. 그런데 SDK는 할당량 문제도 3번 보냈어요. 비용과 시간이 아깝다면 <code>max_retries</code>를 줄이고 직접 판단해요."
   ]
  },
  "s5": {
   "title": "responses.stream()으로 받은 조각",
   "lines": [
    [
     "pieces → ['API는 ', '서로 다른 프로그램들이 ', '정해진 규칙으로 ', '데이터를 주고받고 ', '기능을 이용하게 해 주는 ', '“중간 다리”입니다.']",
     1
    ],
    "\"\".join(pieces) → API는 서로 다른 프로그램들이 정해진 규칙으로 데이터를 주고받고 기능을 이용하게 해 주는 “중간 다리”입니다.",
    [
     "final.output_text == 이어 붙인 글 → True",
     2
    ]
   ],
   "notes": [
    "답이 6조각으로 나눠 도착했어요. 조각이 올 때마다 화면에 붙이면, 사용자는 다 만들어질 때까지 기다리지 않아도 돼요.",
    "다 받은 뒤에는 전체 응답(사용량 포함)도 얻을 수 있어요."
   ]
  },
  "s6": {
   "title": "사용량과 비용 (예시 단가)",
   "lines": [
    "입력 31토큰 × $0.75 / 100만",
    "출력 36토큰 × $4.50 / 100만",
    [
     "cost → $0.00018525",
     1
    ]
   ],
   "notes": [
    "한 번 호출에 약 0.0002달러예요. 적어 보여도 하루 만 번이면 2달러, 재시도가 붙으면 더 늘어요. 그래서 호출마다 토큰과 비용을 기록해요. 단가는 예시예요. 실제 단가는 모델마다 다르고 바뀌니 공식 가격표를 봐요.<br><code>Decimal</code>은 돈 계산에서 소수 오차가 쌓이지 않게 쓰는 정확한 소수예요."
   ]
  }
 },
 "flow": [
  [
   1,
   "클라이언트",
   "OpenAI()",
   "키는 환경 변수"
  ],
  [
   2,
   "보내기",
   "responses.create",
   "output_text"
  ],
  [
   3,
   "모양 정하기",
   "responses.parse",
   "Pydantic → 스키마"
  ],
  [
   4,
   "실패",
   "예외 종류별",
   "자동 재시도 3번"
  ],
  [
   5,
   "스트리밍",
   "조각으로 받기",
   "delta 이벤트"
  ],
  [
   6,
   "비용",
   "usage",
   "토큰 × 단가"
  ]
 ],
 "env": "Python 3.13.16, openai 3.52.0, httpx 0.28.1. 실제 OpenAI 서버 대신 httpx.MockTransport 가짜 서버"
};
