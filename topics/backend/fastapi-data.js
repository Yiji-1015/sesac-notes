window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 2
  },
  {
   "t": "from pathlib import Path",
   "s": 2
  },
  {
   "t": "from fastapi import FastAPI",
   "s": 1
  },
  {
   "t": "from fastapi import HTTPException",
   "s": 3
  },
  {
   "t": "from pydantic import BaseModel, Field",
   "s": 4
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 앱 만들기",
   "s": 1
  },
  {
   "t": "app = FastAPI(title=\"지식 관리 API\")",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "@app.get(\"/health\")",
   "s": 1
  },
  {
   "t": "def health():",
   "s": 1
  },
  {
   "t": "    return {\"status\": \"ok\"}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 파일 저장소",
   "s": 2
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\" / \"knowledge.json\"",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def read_data():",
   "s": 2
  },
  {
   "t": "    return json.loads(DATA.read_text(encoding=\"utf-8\"))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def write_data(data):",
   "s": 2
  },
  {
   "t": "    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding=\"utf-8\")",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 조회하기 (GET)",
   "s": 3
  },
  {
   "t": "@app.get(\"/knowledge\")",
   "s": 3
  },
  {
   "t": "def list_knowledge(category: str | None = None):",
   "s": 3
  },
  {
   "t": "    entries = read_data()[\"entries\"]",
   "s": 3
  },
  {
   "t": "    if category:",
   "s": 3
  },
  {
   "t": "        entries = [e for e in entries if e[\"category\"] == category]",
   "s": 3
  },
  {
   "t": "    return entries",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "@app.get(\"/knowledge/{knowledge_id}\")",
   "s": 3
  },
  {
   "t": "def get_knowledge(knowledge_id: int):",
   "s": 3
  },
  {
   "t": "    for e in read_data()[\"entries\"]:",
   "s": 3
  },
  {
   "t": "        if e[\"id\"] == knowledge_id:",
   "s": 3
  },
  {
   "t": "            return e",
   "s": 3
  },
  {
   "t": "    raise HTTPException(status_code=404, detail=\"knowledge not found\")",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 입력 모양 정하고 등록하기 (POST)",
   "s": 4
  },
  {
   "t": "class KnowledgeWrite(BaseModel):",
   "s": 4
  },
  {
   "t": "    title: str = Field(min_length=1, max_length=80)",
   "s": 4
  },
  {
   "t": "    category: str = Field(min_length=1, max_length=30)",
   "s": 4
  },
  {
   "t": "    content: str = Field(min_length=1)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "@app.post(\"/knowledge\", status_code=201)",
   "s": 4
  },
  {
   "t": "def create_knowledge(payload: KnowledgeWrite):",
   "s": 4
  },
  {
   "t": "    data = read_data()",
   "s": 4
  },
  {
   "t": "    entry = {\"id\": data[\"next_id\"], **payload.model_dump()}",
   "s": 4
  },
  {
   "t": "    data[\"entries\"].append(entry)",
   "s": 4
  },
  {
   "t": "    data[\"next_id\"] += 1",
   "s": 4
  },
  {
   "t": "    write_data(data)",
   "s": 4
  },
  {
   "t": "    return entry",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 고치기 (PUT)",
   "s": 5
  },
  {
   "t": "@app.put(\"/knowledge/{knowledge_id}\")",
   "s": 5
  },
  {
   "t": "def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):",
   "s": 5
  },
  {
   "t": "    data = read_data()",
   "s": 5
  },
  {
   "t": "    for e in data[\"entries\"]:",
   "s": 5
  },
  {
   "t": "        if e[\"id\"] == knowledge_id:",
   "s": 5
  },
  {
   "t": "            e.update(payload.model_dump())",
   "s": 5
  },
  {
   "t": "            write_data(data)",
   "s": 5
  },
  {
   "t": "            return e",
   "s": 5
  },
  {
   "t": "    raise HTTPException(status_code=404, detail=\"knowledge not found\")",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 지우기 (DELETE)",
   "s": 6
  },
  {
   "t": "@app.delete(\"/knowledge/{knowledge_id}\", status_code=204)",
   "s": 6
  },
  {
   "t": "def delete_knowledge(knowledge_id: int):",
   "s": 6
  },
  {
   "t": "    data = read_data()",
   "s": 6
  },
  {
   "t": "    kept = [e for e in data[\"entries\"] if e[\"id\"] != knowledge_id]",
   "s": 6
  },
  {
   "t": "    if len(kept) == len(data[\"entries\"]):",
   "s": 6
  },
  {
   "t": "        raise HTTPException(status_code=404, detail=\"knowledge not found\")",
   "s": 6
  },
  {
   "t": "    data[\"entries\"] = kept",
   "s": 6
  },
  {
   "t": "    write_data(data)",
   "s": 6
  }
 ],
 "names": {
  "1": "앱 만들기",
  "2": "파일 저장소",
  "3": "조회하기 (GET)",
  "4": "입력 모양 정하고 등록하기 (POST)",
  "5": "고치기 (PUT)",
  "6": "지우기 (DELETE)"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "why_file": {
   "title": "파일에 category 없는 항목을 직접 넣은 뒤 GET /knowledge?category=api",
   "lines": [
    [
     "상태 코드 500",
     1,
     "bad"
    ],
    "응답 본문: Internal Server Error"
   ],
   "notes": [
    "파일은 아무 모양이든 받아 줘서, 잘못된 항목이 그대로 저장됐어요. 그걸 읽는 순간 서버 코드가 <code>e[\"category\"]</code>에서 멈춰서 <b>모든 요청</b>이 500(서버 내부 오류)이 돼요."
   ]
  },
  "why_api": {
   "title": "같은 내용을 API로 POST",
   "lines": [
    [
     "상태 코드 422",
     1
    ],
    "응답: {\"type\": \"missing\", \"loc\": [\"body\", \"category\"], \"msg\": \"Field required\", \"input\": {\"title\": \"카테고리 빠짐\", \"content\": \"API로 보냈다\"}}",
    [
     "파일의 항목 수: 3개 (그대로)",
     2
    ]
   ],
   "notes": [
    "API는 저장하기 <b>전에</b> 입력 모양을 검사해서 422(입력 오류)로 돌려보내요. 무엇이 틀렸는지도 알려 줘요 (<code>category</code> 빠짐).",
    "잘못된 항목은 파일에 들어가지 않았어요."
   ]
  },
  "s1": {
   "title": "TestClient로 GET /health",
   "lines": [
    [
     "상태 코드 200",
     1
    ],
    [
     "응답 {\"status\": \"ok\"}",
     2
    ],
    "GET /docs → 200 (자동 문서 화면)",
    "openapi.json의 경로 → ['/health']"
   ],
   "notes": [
    "200은 \"성공\"이라는 뜻이에요.",
    "함수가 돌려준 dict가 JSON으로 바뀌어 왔어요. 바꾸는 일은 FastAPI가 해요."
   ]
  },
  "s2": {
   "title": "read_data()",
   "lines": [
    [
     "next_id → 4",
     1
    ],
    [
     "entries → 3개",
     2
    ],
    "entries[0] → {\"id\": 1, \"title\": \"가상환경\", \"category\": \"setup\", \"content\": \"프로젝트별 패키지를 분리한다.\"}"
   ],
   "notes": [
    "다음에 등록할 항목이 받을 번호. 지운 번호를 다시 쓰지 않게 따로 세요.",
    "아직 API 경로는 /health 하나라서, 이 데이터를 밖에서 볼 방법은 없어요. ③에서 열어요."
   ]
  },
  "s3": {
   "title": "③ 실행 결과: 요청 네 개",
   "cols": [
    "요청",
    "상태",
    "응답"
   ],
   "rows": [
    [
     "<code>GET /knowledge</code>",
     "<b>200</b>",
     "<code>[{&quot;id&quot;: 1, &quot;title&quot;: &quot;가상환경&quot;, &quot;category&quot;: &quot;setup&quot;, &quot;content&quot;: &quot;프로젝트별 패키지를 분리한다.&quot;}, {&quot;id&quot;: 2, &quot;title&quot;: &quot;서버 실행&quot;, &quot;category&quot;:…</code>"
    ],
    [
     "<code>GET /knowledge?category=api</code>",
     "<b>200</b>",
     "<code>[{&quot;id&quot;: 2, &quot;title&quot;: &quot;서버 실행&quot;, &quot;category&quot;: &quot;api&quot;, &quot;content&quot;: &quot;uvicorn으로 개발 서버를 켠다.&quot;}, {&quot;id&quot;: 3, &quot;title&quot;: &quot;입력 오류&quot;, &quot;categor…</code>"
    ],
    [
     "<code>GET /knowledge/2</code>",
     "<b>200</b>",
     "<code>{&quot;id&quot;: 2, &quot;title&quot;: &quot;서버 실행&quot;, &quot;category&quot;: &quot;api&quot;, &quot;content&quot;: &quot;uvicorn으로 개발 서버를 켠다.&quot;}</code>"
    ],
    [
     "<code>GET /knowledge/99</code>",
     "<b>404</b>",
     "<code>{&quot;detail&quot;: &quot;knowledge not found&quot;}</code>"
    ]
   ],
   "notes": [
    "<code>?category=api</code>처럼 <code>?</code> 뒤에 붙는 값은 함수 인자 <code>category</code>로 들어와요 (쿼리 매개변수).",
    "<code>/knowledge/2</code>의 <code>2</code>는 경로 안의 <code>{knowledge_id}</code>로 들어와요 (경로 매개변수). 타입이 <code>int</code>라서 FastAPI가 숫자로 바꿔 줘요.",
    "없는 번호는 <code>HTTPException(404)</code>로 \"없음\"을 알려요. 빈 응답이나 200을 주면 클라이언트가 \"있는데 비었다\"로 오해해요."
   ]
  },
  "t3": {
   "title": "GET /knowledge/abc",
   "lines": [
    [
     "상태 코드 422",
     1
    ],
    "응답: {\"type\": \"int_parsing\", \"loc\": [\"path\", \"knowledge_id\"], \"msg\": \"Input should be a valid integer, unable to parse string as an integer\", \"input\": \"abc\"}"
   ],
   "notes": [
    "경로의 <code>abc</code>를 <code>int</code>로 바꿀 수 없어서, 함수가 불리기도 전에 FastAPI가 422로 돌려보냈어요. 타입을 적어 두는 것만으로 검사가 생겨요."
   ]
  },
  "s4": {
   "title": "POST /knowledge  본문 {\"title\": \"실습 환경 안내\", \"category\": \"setup\", \"content\": \"Python 3.10 이상을 사용한다.\"}",
   "lines": [
    [
     "상태 코드 201",
     1
    ],
    [
     "응답 {\"id\": 4, \"title\": \"실습 환경 안내\", \"category\": \"setup\", \"content\": \"Python 3.10 이상을 사용한다.\"}",
     2
    ],
    [
     "파일: next_id 5, 항목 4개",
     3
    ]
   ],
   "notes": [
    "201은 \"새로 만들었다\"는 뜻이에요. <code>status_code=201</code>로 정했어요.",
    "보낸 세 필드에 서버가 정한 <code>id</code>가 붙었어요. 번호는 클라이언트가 정하지 않아요.",
    "파일에도 저장됐어요."
   ]
  },
  "t4": {
   "title": "잘못된 POST 세 번",
   "cols": [
    "보낸 것",
    "상태",
    "어디가 (loc)",
    "왜 (msg)"
   ],
   "rows": [
    [
     "title이 빈 글",
     "<b>422</b>",
     "<code>[&#x27;body&#x27;, &#x27;title&#x27;]</code>",
     "String should have at least 1 character"
    ],
    [
     "category 빠짐",
     "<b>422</b>",
     "<code>[&#x27;body&#x27;, &#x27;category&#x27;]</code>",
     "Field required"
    ],
    [
     "title이 숫자",
     "<b>422</b>",
     "<code>[&#x27;body&#x27;, &#x27;title&#x27;]</code>",
     "Input should be a valid string"
    ]
   ],
   "notes": [
    "세 번 모두 422였고, 파일 항목 수는 4개 그대로예요. 검사에 걸리면 <code>create_knowledge</code> 함수 자체가 불리지 않아요."
   ]
  },
  "s5": {
   "title": "PUT /knowledge/4 (content만 바꿔서 전체를 다시 보냄)",
   "lines": [
    [
     "상태 코드 200",
     1
    ],
    "응답 {\"id\": 4, \"title\": \"실습 환경 안내\", \"category\": \"setup\", \"content\": \"프로젝트별 가상환경을 사용한다.\"}",
    [
     "PUT /knowledge/99 → 404 {\"detail\": \"knowledge not found\"}",
     2
    ]
   ],
   "notes": [
    "PUT은 \"이 번호의 항목을 이 내용으로 통째로 바꿔라\"예요. 바꾸지 않는 필드도 다 보내요. 입력 검사는 ④의 <code>KnowledgeWrite</code>를 그대로 써요.",
    "없는 번호는 조회와 똑같이 404예요."
   ]
  },
  "s6": {
   "title": "DELETE /knowledge/4 를 두 번",
   "lines": [
    [
     "첫 번째 → 204, 본문 0바이트",
     1
    ],
    [
     "두 번째 → 404 {'detail': 'knowledge not found'}",
     2
    ],
    "남은 id → [1, 2, 3]",
    [
     "다시 등록하면 받는 id → 5",
     3
    ]
   ],
   "notes": [
    "204는 \"성공했고 돌려줄 내용은 없다\"예요. 그래서 본문이 비어 있어요.",
    "이미 지운 번호라 404예요.",
    "4번은 지웠지만 <code>next_id</code>는 5라서 새 항목은 5번이에요. 지운 번호를 다시 쓰면, 옛 4번을 가리키던 링크가 엉뚱한 항목을 열게 돼요."
   ]
  },
  "codes": {
   "title": "이 API의 상태 코드 한눈에",
   "cols": [
    "요청",
    "성공",
    "실패",
    "나온 곳"
   ],
   "rows": [
    [
     "<code>GET /knowledge</code>",
     "200",
     "—",
     "—"
    ],
    [
     "<code>GET /knowledge/{id}</code>",
     "200",
     "404 (없음), 422 (숫자 아님)",
     "③"
    ],
    [
     "<code>POST /knowledge</code>",
     "201",
     "422 (입력 모양 틀림)",
     "④"
    ],
    [
     "<code>PUT /knowledge/{id}</code>",
     "200",
     "404, 422",
     "⑤"
    ],
    [
     "<code>DELETE /knowledge/{id}</code>",
     "204",
     "404",
     "⑥"
    ]
   ],
   "notes": []
  }
 },
 "flow": [
  [
   1,
   "앱",
   "/health",
   "FastAPI()"
  ],
  [
   2,
   "저장소",
   "JSON 파일",
   "읽기, 쓰기"
  ],
  [
   3,
   "조회",
   "GET",
   "목록, 상세, 404"
  ],
  [
   4,
   "등록",
   "POST",
   "Pydantic 검사, 201"
  ],
  [
   5,
   "수정",
   "PUT",
   "통째로 바꾸기"
  ],
  [
   6,
   "삭제",
   "DELETE",
   "204, 번호 재사용 안 함"
  ]
 ],
 "env": "Python 3.13.16, FastAPI 0.162.2, Pydantic 2.14.3, TestClient(httpx)"
};
