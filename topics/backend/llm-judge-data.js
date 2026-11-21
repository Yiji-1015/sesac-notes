window.PAGE = {
 "code": [
  {
   "t": "from typing import Literal",
   "s": 1
  },
  {
   "t": "from pydantic import BaseModel, ConfigDict, Field",
   "s": 1
  },
  {
   "t": "import anthropic",
   "s": 2
  },
  {
   "t": "from sklearn.metrics import cohen_kappa_score",
   "s": 5
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 채점 기준과 점수 계약",
   "s": 1
  },
  {
   "t": "RUBRIC = \"\"\"너는 학습 노트의 질문에 대한 답을 채점하는 평가자야.",
   "s": 1
  },
  {
   "t": "[기준 답]과 비교해서 [채점할 답]에 1~5점을 줘.",
   "s": 1
  },
  {
   "t": "5: 기준 답의 핵심을 모두 맞게 담음",
   "s": 1
  },
  {
   "t": "4: 핵심은 맞지만 하나가 빠졌거나 덜 정확함",
   "s": 1
  },
  {
   "t": "3: 절반쯤 맞고, 중요한 것이 빠짐",
   "s": 1
  },
  {
   "t": "2: 대부분 빠졌거나 일부가 틀림",
   "s": 1
  },
  {
   "t": "1: 틀렸거나 질문과 상관없음",
   "s": 1
  },
  {
   "t": "오류 종류는 하나만 골라: 없음, 빠짐, 틀림, 질문과 무관",
   "s": 1
  },
  {
   "t": "근거를 먼저 쓰고, 그 근거로 점수를 정해.",
   "s": 1
  },
  {
   "t": "답이 길거나 자신 있게 쓰였다고 점수를 올리지 마. 기준 답에 없는 내용은 맞더라도 점수에 넣지 마.\"\"\"",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "class Verdict(BaseModel):",
   "s": 1
  },
  {
   "t": "    model_config = ConfigDict(extra=\"forbid\")  # 계약에 없는 필드는 받지 않기",
   "s": 1
  },
  {
   "t": "    rationale: str = Field(description=\"기준 답과 무엇이 같고 무엇이 다른지\")",
   "s": 1
  },
  {
   "t": "    error_type: Literal[\"없음\", \"빠짐\", \"틀림\", \"질문과 무관\"]",
   "s": 1
  },
  {
   "t": "    score: Literal[1, 2, 3, 4, 5]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "ITEMS = [",
   "s": 1
  },
  {
   "t": "    {\"question\": \"교차검증은 왜 하나요?\",",
   "s": 1
  },
  {
   "t": "     \"reference\": \"valid 한 번의 점수는 우연에 크게 흔들려서, 데이터를 여러 덩어리로 나눠 번갈아 재고 평균과 흔들림을 함께 봐요.\",",
   "s": 1
  },
  {
   "t": "     \"answer\": \"valid 한 번으로는 점수가 많이 흔들려서, 다섯 덩어리로 나눠 다섯 번 재고 평균을 봐요.\"},",
   "s": 1
  },
  {
   "t": "    {\"question\": \"AdaBoost가 라벨 잡음에 약한 이유는?\",",
   "s": 1
  },
  {
   "t": "     \"reference\": \"지수 손실을 줄여서, 크게 틀린 행의 무게가 지수적으로 커져요. 라벨이 잘못된 행에 무게를 몰아줘요.\",",
   "s": 1
  },
  {
   "t": "     \"answer\": \"트리를 너무 많이 쌓아서 과적합하기 때문이에요.\"},",
   "s": 1
  },
  {
   "t": "    {\"question\": \"프롬프트 캐시는 무엇을 아끼나요?\",",
   "s": 1
  },
  {
   "t": "     \"reference\": \"요청 앞부분이 이전 요청과 같으면, 그 부분의 입력 토큰을 싸게 읽어서 비용과 첫 토큰까지의 시간을 줄여요.\",",
   "s": 1
  },
  {
   "t": "     \"answer\": \"프롬프트 캐시는 아주 중요한 기능으로, 모델이 만든 답을 저장해 두었다가 같은 질문이 오면 다시 보내 줘서 출력 토큰을 크게 아껴 줘요. 실무에서 꼭 써야 해요.\"},",
   "s": 1
  },
  {
   "t": "]",
   "s": 1
  },
  {
   "t": "schema = Verdict.model_json_schema()  # 점수 계약을 JSON Schema로",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 요청 만들기: 고정 부분은 앞에, 바뀌는 부분은 뒤에",
   "s": 2
  },
  {
   "t": "client = anthropic.Anthropic(max_retries=2)  # 키는 환경 변수 ANTHROPIC_API_KEY에서 읽어요",
   "s": 2
  },
  {
   "t": "MODEL = \"claude-opus-5-5\"",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def judge_request(item, effort=\"low\", max_tokens=2048):",
   "s": 2
  },
  {
   "t": "    return dict(",
   "s": 2
  },
  {
   "t": "        model=MODEL,",
   "s": 2
  },
  {
   "t": "        max_tokens=max_tokens,",
   "s": 2
  },
  {
   "t": "        system=[{\"type\": \"text\", \"text\": RUBRIC, \"cache_control\": {\"type\": \"ephemeral\"}}],  # 모든 요청에 같은 채점 기준: 여기까지 캐시",
   "s": 2
  },
  {
   "t": "        messages=[{\"role\": \"user\", \"content\": f\"[질문]\\n{item['question']}\\n[기준 답]\\n{item['reference']}\\n[채점할 답]\\n{item['answer']}\"}],",
   "s": 2
  },
  {
   "t": "        output_config={\"effort\": effort, \"format\": {\"type\": \"json_schema\", \"schema\": schema}},  # 생각 깊이와 답의 모양",
   "s": 2
  },
  {
   "t": "    )",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def text_of(response):  # 생각 블록은 건너뛰고 글 블록만",
   "s": 2
  },
  {
   "t": "    return \"\".join(b.text for b in response.content if b.type == \"text\")",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "first = client.messages.create(**judge_request(ITEMS[0]))",
   "s": 2
  },
  {
   "t": "verdict = Verdict.model_validate_json(text_of(first))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 캐시로 비용 줄이기: 몇 건부터 이득인가",
   "s": 3
  },
  {
   "t": "PRICE_IN, PRICE_OUT = 4.00, 20.00  # claude-opus-5-5, 100만 토큰당 달러",
   "s": 3
  },
  {
   "t": "WRITE_5M, READ = 1.25, 0.05  # 5분 캐시 쓰기는 입력 단가의 1.25배, 읽기는 0.05배 (Opus 5.5)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def batch_cost(n, fixed, varying, out, cached):  # n건 채점의 달러 비용",
   "s": 3
  },
  {
   "t": "    if not cached:",
   "s": 3
  },
  {
   "t": "        inputs = n * (fixed + varying)",
   "s": 3
  },
  {
   "t": "    else:",
   "s": 3
  },
  {
   "t": "        inputs = fixed * WRITE_5M + (n - 1) * fixed * READ + n * varying  # 첫 건은 쓰기, 나머지는 읽기",
   "s": 3
  },
  {
   "t": "    return (inputs * PRICE_IN + n * out * PRICE_OUT) / 1_000_000",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "FIXED, VARYING, OUT = 3000, 300, 400  # 가정: 채점 기준 3,000토큰, 문항 300토큰, 출력(생각 포함) 400토큰",
   "s": 3
  },
  {
   "t": "costs = {n: (batch_cost(n, FIXED, VARYING, OUT, False), batch_cost(n, FIXED, VARYING, OUT, True)) for n in (1, 2, 10, 1000)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 응답 읽기: 끝까지 못 쓴 답, 거절, 쓴 토큰 모으기",
   "s": 4
  },
  {
   "t": "def judge(item):",
   "s": 4
  },
  {
   "t": "    used, tries = [], []",
   "s": 4
  },
  {
   "t": "    for max_tokens in (2048, 8192):  # 잘리면 한 번만 더 크게",
   "s": 4
  },
  {
   "t": "        r = client.messages.create(**judge_request(item, max_tokens=max_tokens))",
   "s": 4
  },
  {
   "t": "        used.append(r.usage)",
   "s": 4
  },
  {
   "t": "        tries.append(r.stop_reason)",
   "s": 4
  },
  {
   "t": "        if r.stop_reason == \"refusal\":  # 안전 판단으로 거절: 점수 없음으로 남기기",
   "s": 4
  },
  {
   "t": "            return None, used, tries",
   "s": 4
  },
  {
   "t": "        if r.stop_reason == \"max_tokens\":  # 생각하다가 한도에 닿아 JSON이 잘림",
   "s": 4
  },
  {
   "t": "            continue",
   "s": 4
  },
  {
   "t": "        return Verdict.model_validate_json(text_of(r)), used, tries",
   "s": 4
  },
  {
   "t": "    return None, used, tries",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def total_usage(used):  # 캐시 쓰기, 읽기까지 따로 더하기",
   "s": 4
  },
  {
   "t": "    keys = (\"input_tokens\", \"cache_creation_input_tokens\", \"cache_read_input_tokens\", \"output_tokens\")",
   "s": 4
  },
  {
   "t": "    return {k: sum(getattr(u, k) or 0 for u in used) for k in keys}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "results = [judge(it) for it in ITEMS]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 채점을 믿을 수 있나: 순서 바꿔 묻기, 사람 점수와 맞춰 보기",
   "s": 5
  },
  {
   "t": "class Pick(BaseModel):",
   "s": 5
  },
  {
   "t": "    model_config = ConfigDict(extra=\"forbid\")",
   "s": 5
  },
  {
   "t": "    rationale: str",
   "s": 5
  },
  {
   "t": "    winner: Literal[\"A\", \"B\", \"같음\"]",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def pick(question, reference, a, b):",
   "s": 5
  },
  {
   "t": "    r = client.messages.create(model=MODEL, max_tokens=2048,",
   "s": 5
  },
  {
   "t": "        system=[{\"type\": \"text\", \"text\": \"두 답 중 [기준 답]에 더 가까운 쪽을 골라. 길이와 말투는 보지 마.\", \"cache_control\": {\"type\": \"ephemeral\"}}],",
   "s": 5
  },
  {
   "t": "        messages=[{\"role\": \"user\", \"content\": f\"[질문]\\n{question}\\n[기준 답]\\n{reference}\\n[A]\\n{a}\\n[B]\\n{b}\"}],",
   "s": 5
  },
  {
   "t": "        output_config={\"effort\": \"low\", \"format\": {\"type\": \"json_schema\", \"schema\": Pick.model_json_schema()}})",
   "s": 5
  },
  {
   "t": "    return Pick.model_validate_json(text_of(r)).winner",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def pairwise(question, reference, x, y):  # x를 A에 한 번, B에 한 번",
   "s": 5
  },
  {
   "t": "    one = pick(question, reference, x, y)",
   "s": 5
  },
  {
   "t": "    two = {\"A\": \"B\", \"B\": \"A\", \"같음\": \"같음\"}[pick(question, reference, y, x)]  # 두 번째 답을 첫 번째 기준으로 되돌리기",
   "s": 5
  },
  {
   "t": "    return one, two, one == two",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "ITEM = ITEMS[1]",
   "s": 5
  },
  {
   "t": "short_right = ITEM[\"reference\"]",
   "s": 5
  },
  {
   "t": "long_wrong = ITEMS[1][\"answer\"] + \" 트리 수를 늘리면 train 오차가 0이 되고, 이것은 부스팅에서 아주 잘 알려진 현상이에요. 그래서 learning rate를 작게 하고 조기 종료를 꼭 써야 해요.\"",
   "s": 5
  },
  {
   "t": "order_check = pairwise(ITEM[\"question\"], ITEM[\"reference\"], short_right, long_wrong)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "human = [5, 4, 2, 1, 3, 5, 2, 4, 3, 1]  # 가상의 사람 점수 10건",
   "s": 5
  },
  {
   "t": "judged = [5, 5, 3, 1, 4, 5, 2, 5, 3, 2]  # 같은 10건의 Judge 점수",
   "s": 5
  },
  {
   "t": "exact = sum(h == j for h, j in zip(human, judged)) / len(human)",
   "s": 5
  },
  {
   "t": "within1 = sum(abs(h - j) <= 1 for h, j in zip(human, judged)) / len(human)",
   "s": 5
  },
  {
   "t": "kappa = cohen_kappa_score(human, judged, weights=\"quadratic\")  # 우연히 맞을 몫을 뺀 일치도",
   "s": 5
  },
  {
   "t": "lenient = sum(judged) / len(judged) - sum(human) / len(human)  # Judge가 평균 몇 점 후한가",
   "s": 5
  }
 ],
 "names": {
  "1": "채점 기준과 점수 계약",
  "2": "요청 만들기: 고정 부분은 앞에, 바뀌는 부분은 뒤에",
  "3": "캐시로 비용 줄이기: 몇 건부터 이득인가",
  "4": "응답 읽기: 끝까지 못 쓴 답, 거절, 쓴 토큰 모으기",
  "5": "채점을 믿을 수 있나: 순서 바꿔 묻기, 사람 점수와 맞춰 보기"
 },
 "range": [
  1,
  5
 ],
 "outs": {
  "s1": {
   "title": "Pydantic 클래스가 만든 JSON Schema (줄임)",
   "lines": [
    [
     "properties 순서 → rationale, error_type, score",
     1
    ],
    [
     "error_type → enum ['없음', '빠짐', '틀림', '질문과 무관']",
     2
    ],
    [
     "score → enum [1, 2, 3, 4, 5]",
     3
    ],
    [
     "additionalProperties → False",
     4
    ],
    "required → ['rationale', 'error_type', 'score']"
   ],
   "notes": [
    "필드 순서가 곧 모델이 쓰는 순서예요. 근거(rationale)를 먼저 쓰게 해야, 점수를 먼저 정하고 근거를 끼워 맞추는 일이 줄어요.",
    "점수와 오류 종류는 정해진 값 중에서만 고르게 했어요(<code>enum</code>). \"4점\", \"4.5\" 같은 답이 아예 나올 수 없어요.",
    "<code>extra=\"forbid\"</code>가 <code>additionalProperties: false</code>가 돼요. 모델이 <code>confidence</code> 같은 필드를 덧붙이지 못해요."
   ]
  },
  "s2": {
   "title": "SDK가 실제로 보낸 요청 (가짜 서버에서 본 것)",
   "lines": [
    [
     "POST https://api.anthropic.com/v1/messages",
     1
    ],
    [
     "x-api-key: sk-ant-tes…, anthropic-version: 2023-06-01",
     2
    ],
    [
     "본문 → {\"max_tokens\": 2048, \"messages\": [{\"role\": \"user\", \"content\": \"[질문]\\n교차검증은 왜 하나요?\\n[기준 답]…\"}], \"model\": \"claude-opus-5-5\", \"output_config\": {\"effort\": \"low\", \"format\": {\"type\": \"json_schema\", \"schema\": \"①의 JSON Schema\"}}, \"system\": [{\"type\": \"text\", \"text\": \"너는 학습 노트의 질문에 대한 답을 채점하는 평가자…\", \"cache_control\": {\"type\": \"ephemeral\"}}]}",
     3
    ],
    "",
    [
     "세 문항의 system이 모두 같은가 → True, messages는 모두 다른가 → True",
     4,
     "newf"
    ],
    [
     "응답 content 블록 → ['thinking', 'text']",
     5
    ],
    [
     "Verdict로 읽은 결과 → score 4, error_type 빠짐",
     6
    ]
   ],
   "notes": [
    "키는 본문이 아니라 머리글(<code>x-api-key</code>)로 가요. 어떤 API 판으로 말하는지는 <code>anthropic-version</code>으로 정해요.",
    "채점 기준은 문항마다 같아서 <code>system</code>에 두고 <code>cache_control</code>을 붙였어요. 바뀌는 질문과 답은 그 뒤 <code>messages</code>에 와요. 캐시는 앞에서부터 글자 하나까지 같은 부분만 다시 써요.",
    "Claude Opus 5.5는 생각(thinking)이 늘 켜져 있어서, 응답 맨 앞에 생각 블록이 와요(내용은 기본으로 비어 있어요). 그래서 <code>content[0].text</code>가 아니라 글 블록만 골라 읽어요.",
    "판정 내용은 가짜 서버가 돌려준 고정 값이에요. 이 단계에서 볼 것은 요청의 모양이에요."
   ]
  },
  "s3": {
   "title": "③ 채점 기준 3,000토큰 + 문항 300토큰, 출력 400토큰일 때 (claude-opus-5-5 단가, 가정한 토큰 수)",
   "cols": [
    "문항 수",
    "캐시 없이",
    "캐시 쓰고",
    "줄어든 비율"
   ],
   "rows": [
    [
     "1",
     "$0.0212",
     "$0.0242",
     "오히려 14% 더 비쌈"
    ],
    [
     "2",
     "$0.0424",
     "$0.0340",
     "20%"
    ],
    [
     "10",
     "$0.2120",
     "$0.1124",
     "47%"
    ],
    [
     "1,000",
     "$21.2000",
     "$9.8144",
     "54%"
    ]
   ],
   "notes": [
    "1건뿐이면 캐시 쓰기 값(입력 단가의 1.25배)만 더 내서 손해예요. 같은 기준으로 2건만 채점해도 이득이에요.",
    "1,000건이면 전체 비용이 54% 줄어요. 입력 비용만 보면 86% 줄지만, 출력(생각 포함) 비용은 캐시와 상관없어서 그대로예요.",
    "토큰 수는 가정한 값이에요. 실제 수는 <code>client.messages.count_tokens(...)</code>로 세거나, 응답의 <code>usage</code>로 확인해요."
   ]
  },
  "s4": {
   "title": "④ 세 문항을 채점한 기록 (가짜 서버)",
   "cols": [
    "질문",
    "stop_reason",
    "score",
    "error_type",
    "입력 / 출력 토큰 (합)"
   ],
   "rows": [
    [
     "교차검증은 왜 하나요?",
     "end_turn",
     4,
     "빠짐",
     "600 / 180"
    ],
    [
     "AdaBoost가 라벨 잡음에 약한 이유는?",
     "max_tokens → end_turn",
     1,
     "틀림",
     "1,200 / 2,228"
    ],
    [
     "프롬프트 캐시는 무엇을 아끼나요?",
     "end_turn",
     1,
     "틀림",
     "600 / 180"
    ]
   ],
   "notes": [
    "두 번째 문항은 가짜 서버가 일부러 <code>max_tokens</code>로 끊었어요. 생각에 토큰을 쓰다 한도에 닿으면, 글 블록의 JSON이 중간에 잘려요. 그대로 읽으면 Pydantic 오류가 나니까, <code>stop_reason</code>을 먼저 보고 한도를 키워 한 번만 다시 물었어요.",
    "두 번 부른 토큰은 둘 다 돈이 나가요. 그래서 성공한 호출만이 아니라 모든 호출의 <code>usage</code>를 더해요. 캐시 쓰기와 읽기는 단가가 달라서 칸을 따로 더해요."
   ]
  },
  "s5": {
   "title": "⑤ 순서를 바꿔 두 번 묻기 (가짜 Judge는 늘 A를 고르게 해 둠)",
   "lines": [
    [
     "짧고 맞는 답을 A에 → A",
     1
    ],
    [
     "짧고 맞는 답을 B에 → 첫 번째 기준으로 되돌리면 B",
     2
    ],
    [
     "두 번의 판정이 같은가 → False",
     3,
     "newf"
    ]
   ],
   "notes": [
    "판정이 순서에 따라 뒤집히면 그 비교는 버리거나 \"같음\"으로 둬요. 실제 Judge는 이렇게 극단적이진 않지만, 첫 번째나 두 번째 자리를 조금씩 더 좋아하는 버릇이 알려져 있어요."
   ]
  },
  "s5k": {
   "title": "가상의 10건으로 사람 점수와 Judge 점수 맞춰 보기",
   "cols": [
    "지표",
    "값",
    "뜻"
   ],
   "rows": [
    [
     "완전 일치",
     "50%",
     "점수가 똑같은 비율"
    ],
    [
     "1점 안",
     "100%",
     "차이가 1점 이하인 비율"
    ],
    [
     "가중 kappa",
     "0.884",
     "우연히 맞을 몫을 빼고, 크게 어긋날수록 더 깎은 일치도. 1이면 완전 일치, 0이면 우연 수준"
    ],
    [
     "평균 차이 (Judge − 사람)",
     "+0.5점",
     "Judge가 평균적으로 후한지(+) 박한지(−)"
    ]
   ],
   "notes": [
    "점수는 사람이 미리 적어 둔 가상의 값이에요. 숫자로 무엇을 보는지 익히려는 표예요.",
    "완전 일치는 50%뿐이지만 1점 안에서는 모두 맞고, 가중 kappa가 0.88로 높아요. 대신 Judge가 평균 0.5점 후해요. 순위(어느 답이 더 나은가)는 믿어도 되지만, \"평균 4점이면 합격\" 같은 절대 기준은 사람 점수로 다시 맞춰야 한다는 뜻이에요."
   ]
  }
 },
 "flow": [
  [
   1,
   "채점 기준",
   "루브릭 + 점수 계약",
   "근거 먼저, enum"
  ],
  [
   2,
   "요청",
   "고정은 앞, 바뀌는 건 뒤",
   "cache_control, effort"
  ],
  [
   3,
   "비용",
   "캐시 쓰기 vs 읽기",
   "2건부터 이득"
  ],
  [
   4,
   "응답 읽기",
   "잘림, 거절, 사용량",
   "stop_reason 먼저"
  ],
  [
   5,
   "믿을 수 있나",
   "순서 바꿔 묻기",
   "사람 점수와 kappa"
  ]
 ],
 "env": "Python 3.13.5, anthropic 0.122.0, pydantic 2.13.4, httpx 0.28.1 (가짜 서버)"
};
