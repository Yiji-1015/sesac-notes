window.PAGE = {
 "code": [
  {
   "t": "import hashlib",
   "s": 2
  },
  {
   "t": "import json",
   "s": 3
  },
  {
   "t": "import re",
   "s": 2
  },
  {
   "t": "import statistics",
   "s": 1
  },
  {
   "t": "import sys",
   "s": 4
  },
  {
   "t": "import tempfile",
   "s": 4
  },
  {
   "t": "import time",
   "s": 1
  },
  {
   "t": "import unicodedata",
   "s": 2
  },
  {
   "t": "from importlib.metadata import version",
   "s": 6
  },
  {
   "t": "from pathlib import Path",
   "s": 4
  },
  {
   "t": "from typing import Literal",
   "s": 3
  },
  {
   "t": "import httpx",
   "s": 5
  },
  {
   "t": "from dotenv import dotenv_values",
   "s": 4
  },
  {
   "t": "from pydantic import BaseModel, ConfigDict, ValidationError",
   "s": 3
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 6
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 결과에 어떻게 얻었는지 적기",
   "s": 1
  },
  {
   "t": "def measure(fn, mode):  # 값과 시간에 '어떻게 얻었나'를 함께 적어요",
   "s": 1
  },
  {
   "t": "    start = time.perf_counter()",
   "s": 1
  },
  {
   "t": "    value = fn()",
   "s": 1
  },
  {
   "t": "    return {\"value\": value, \"ms\": (time.perf_counter() - start) * 1000, \"execution_mode\": mode}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def live_work():  # 실제로 계산",
   "s": 1
  },
  {
   "t": "    return sum(i * i for i in range(300_000))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def fixture_work():  # 시험용으로 미리 적어 둔 값 (계산하지 않아요)",
   "s": 1
  },
  {
   "t": "    return 8999955000050000",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "runs = [measure(live_work, \"live\") for _ in range(5)] + [measure(fixture_work, \"fixture\") for _ in range(5)]",
   "s": 1
  },
  {
   "t": "mixed_ms = statistics.mean(r[\"ms\"] for r in runs)  # 모드를 안 보고 평균을 내면",
   "s": 1
  },
  {
   "t": "live_ms = statistics.mean(r[\"ms\"] for r in runs if r[\"execution_mode\"] == \"live\")",
   "s": 1
  },
  {
   "t": "same_value = len({r[\"value\"] for r in runs}) == 1  # 값만 보면 둘 다 '정답'이에요",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 원본을 남기고, 바꾼 것은 따로",
   "s": 2
  },
  {
   "t": "original = \"스터디룸   예약은 이용 시작 1시간 전까지 철회할 수 있어요.　노쇼가 2번이면 30일 동안 예약할 수 없어요.\"",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def clean_buggy(text):  # 공백을 정리하려다 숫자까지 지운 정리 함수 (흔한 실수)",
   "s": 2
  },
  {
   "t": "    return re.sub(r\"[^가-힣\\s.]\", \"\", unicodedata.normalize(\"NFC\", text)).split()",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def clean(text):",
   "s": 2
  },
  {
   "t": "    return \" \".join(unicodedata.normalize(\"NFC\", text).split())",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "overwritten = {\"text\": original}",
   "s": 2
  },
  {
   "t": "overwritten[\"text\"] = \" \".join(clean_buggy(overwritten[\"text\"]))  # 같은 칸에 덮어쓰기",
   "s": 2
  },
  {
   "t": "kept = {\"original\": original, \"original_sha256\": hashlib.sha256(original.encode()).hexdigest()[:12],",
   "s": 2
  },
  {
   "t": "        \"cleaned\": \" \".join(clean_buggy(original))}  # 원본과 바꾼 것을 따로",
   "s": 2
  },
  {
   "t": "lost_numbers = sorted(set(re.findall(r\"\\d+\", kept[\"original\"])) - set(re.findall(r\"\\d+\", kept[\"cleaned\"])))  # 원본이 있어야 비교할 수 있어요",
   "s": 2
  },
  {
   "t": "fixed = clean(original)",
   "s": 2
  },
  {
   "t": "fixed_lost = sorted(set(re.findall(r\"\\d+\", original)) - set(re.findall(r\"\\d+\", fixed)))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 내 입력은 엄격하게, 남의 응답은 필요한 것만 엄격하게",
   "s": 3
  },
  {
   "t": "class ReserveRequest(BaseModel):  # 우리 API가 받는 요청: 모르는 칸은 거부",
   "s": 3
  },
  {
   "t": "    model_config = ConfigDict(extra=\"forbid\", strict=True)",
   "s": 3
  },
  {
   "t": "    room: str",
   "s": 3
  },
  {
   "t": "    hours: int",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "checks = {}",
   "s": 3
  },
  {
   "t": "for name, body in {\"정상\": {\"room\": \"A\", \"hours\": 2}, \"오타 칸 houre\": {\"room\": \"A\", \"houre\": 2}, \"글자 \\\"2\\\"\": {\"room\": \"A\", \"hours\": \"2\"}}.items():",
   "s": 3
  },
  {
   "t": "    try:",
   "s": 3
  },
  {
   "t": "        ReserveRequest.model_validate(body)",
   "s": 3
  },
  {
   "t": "        checks[name] = \"통과\"",
   "s": 3
  },
  {
   "t": "    except ValidationError as exc:",
   "s": 3
  },
  {
   "t": "        checks[name] = \", \".join(e[\"type\"] for e in exc.errors())",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "class PartnerStrict(BaseModel):  # 남의 API 응답을 우리 요청처럼 엄격하게 읽으면",
   "s": 3
  },
  {
   "t": "    model_config = ConfigDict(extra=\"forbid\")",
   "s": 3
  },
  {
   "t": "    status: Literal[\"confirmed\", \"rejected\"]",
   "s": 3
  },
  {
   "t": "    reservation_id: int",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "class PartnerTolerant(BaseModel):  # 필요한 칸만 엄격하게, 나머지는 흘려보내기",
   "s": 3
  },
  {
   "t": "    model_config = ConfigDict(extra=\"ignore\")",
   "s": 3
  },
  {
   "t": "    status: Literal[\"confirmed\", \"rejected\"]",
   "s": 3
  },
  {
   "t": "    reservation_id: int",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "reply_v1 = {\"status\": \"confirmed\", \"reservation_id\": 101}",
   "s": 3
  },
  {
   "t": "reply_v2 = {\"status\": \"confirmed\", \"reservation_id\": 101, \"estimated_wait_min\": 5}  # 상대가 칸을 하나 늘림",
   "s": 3
  },
  {
   "t": "reply_broken = {\"status\": \"confirmed\"}  # 꼭 필요한 칸이 빠짐",
   "s": 3
  },
  {
   "t": "partner = {}",
   "s": 3
  },
  {
   "t": "for name, reply in {\"v1\": reply_v1, \"v2 (칸 추가)\": reply_v2, \"필수 칸 빠짐\": reply_broken}.items():",
   "s": 3
  },
  {
   "t": "    for model in (PartnerStrict, PartnerTolerant):",
   "s": 3
  },
  {
   "t": "        try:",
   "s": 3
  },
  {
   "t": "            model.model_validate(reply)",
   "s": 3
  },
  {
   "t": "            partner[name, model.__name__] = \"통과\"",
   "s": 3
  },
  {
   "t": "        except ValidationError as exc:",
   "s": 3
  },
  {
   "t": "            partner[name, model.__name__] = exc.errors()[0][\"type\"]",
   "s": 3
  },
  {
   "t": "quiet_default = reply_broken.get(\"reservation_id\", 0)  # .get()은 없는 값을 가짜 0으로 채워요",
   "s": 3
  },
  {
   "t": "try:",
   "s": 3
  },
  {
   "t": "    reply_broken[\"reservation_id\"]",
   "s": 3
  },
  {
   "t": "    loud = \"읽힘\"",
   "s": 3
  },
  {
   "t": "except KeyError:",
   "s": 3
  },
  {
   "t": "    loud = \"KeyError\"  # []는 없으면 바로 멈춰요",
   "s": 3
  },
  {
   "t": "review = {\"evidence_ok\": None}  # 아직 검사 안 함",
   "s": 3
  },
  {
   "t": "none_as_fail = not review[\"evidence_ok\"]  # True: '검사 실패'로 잘못 분류",
   "s": 3
  },
  {
   "t": "none_kept = review[\"evidence_ok\"] is False  # False: '실패'만 실패로",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 설정은 누가 어느 파일을 읽는지부터",
   "s": 4
  },
  {
   "t": "tmp = Path(tempfile.mkdtemp())",
   "s": 4
  },
  {
   "t": "(tmp / \"url_style.env\").write_text(\"DATABASE_URL=postgresql://reader:example-pass@localhost:5432/library\\n\", encoding=\"utf-8\")",
   "s": 4
  },
  {
   "t": "(tmp / \"key_style.env\").write_text(\"DB_HOST=localhost\\nDB_USER=reader\\nDB_PASSWORD=CHANGE_ME\\n\", encoding=\"utf-8\")",
   "s": 4
  },
  {
   "t": "NEEDED = (\"DB_HOST\", \"DB_USER\", \"DB_PASSWORD\")  # 이 프로그램이 읽는 키",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def load_settings(path):  # 무엇을 읽었는지 보고하되 값은 가려요",
   "s": 4
  },
  {
   "t": "    values = dotenv_values(path)",
   "s": 4
  },
  {
   "t": "    return {\"file\": path.name, \"keys\": sorted(values), \"missing\": [k for k in NEEDED if not values.get(k)],",
   "s": 4
  },
  {
   "t": "            \"placeholder\": [k for k, v in values.items() if v in (\"CHANGE_ME\", \"your_api_key\", \"\")],",
   "s": 4
  },
  {
   "t": "            \"shown\": {k: (\"***\" if any(s in k for s in (\"PASSWORD\", \"URL\", \"KEY\", \"TOKEN\")) else v) for k, v in values.items()}}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "settings = [load_settings(tmp / \"url_style.env\"), load_settings(tmp / \"key_style.env\")]",
   "s": 4
  },
  {
   "t": "interpreter = {\"name\": Path(sys.executable).name, \"in_venv\": sys.prefix != sys.base_prefix}  # 설치와 실행에 같은 파이썬을 쓰나",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 재시도는 멱등할 때만",
   "s": 5
  },
  {
   "t": "def make_server():  # 예약을 저장한 뒤, 첫 응답만 늦게 보내 클라이언트가 시간 초과로 보게 하는 가짜 서버",
   "s": 5
  },
  {
   "t": "    state = {\"reservations\": [], \"by_key\": {}, \"calls\": 0}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "    def handle(request):",
   "s": 5
  },
  {
   "t": "        state[\"calls\"] += 1",
   "s": 5
  },
  {
   "t": "        key = request.headers.get(\"Idempotency-Key\")",
   "s": 5
  },
  {
   "t": "        if key and key in state[\"by_key\"]:  # 같은 키의 요청은 새로 만들지 않고 처음 결과를 돌려줘요",
   "s": 5
  },
  {
   "t": "            return httpx.Response(201, json=state[\"by_key\"][key])",
   "s": 5
  },
  {
   "t": "        rid = 100 + len(state[\"reservations\"]) + 1",
   "s": 5
  },
  {
   "t": "        state[\"reservations\"].append(rid)",
   "s": 5
  },
  {
   "t": "        if key:",
   "s": 5
  },
  {
   "t": "            state[\"by_key\"][key] = {\"reservation_id\": rid}",
   "s": 5
  },
  {
   "t": "        if state[\"calls\"] == 1:",
   "s": 5
  },
  {
   "t": "            raise httpx.ReadTimeout(\"응답이 늦어요\", request=request)  # 저장은 됐는데 응답이 안 옴",
   "s": 5
  },
  {
   "t": "        return httpx.Response(201, json={\"reservation_id\": rid})",
   "s": 5
  },
  {
   "t": "    return state, httpx.Client(transport=httpx.MockTransport(handle))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def reserve_with_retry(client, headers):  # 시간 초과면 한 번 더 보내기",
   "s": 5
  },
  {
   "t": "    for attempt in (1, 2, 3):",
   "s": 5
  },
  {
   "t": "        try:",
   "s": 5
  },
  {
   "t": "            return client.post(\"https://library.example/reservations\", json={\"room\": \"A\", \"hours\": 2}, headers=headers).json(), attempt",
   "s": 5
  },
  {
   "t": "        except httpx.ReadTimeout:",
   "s": 5
  },
  {
   "t": "            continue",
   "s": 5
  },
  {
   "t": "    raise RuntimeError(\"세 번 모두 실패\")",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "plain_state, plain_client = make_server()",
   "s": 5
  },
  {
   "t": "plain_result = reserve_with_retry(plain_client, {})",
   "s": 5
  },
  {
   "t": "keyed_state, keyed_client = make_server()",
   "s": 5
  },
  {
   "t": "keyed_result = reserve_with_retry(keyed_client, {\"Idempotency-Key\": \"req-7f3a\"})",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 판을 적어 두기",
   "s": 6
  },
  {
   "t": "def env_record(packages):  # 이번 실행의 판: 파이썬과 라이브러리",
   "s": 6
  },
  {
   "t": "    return {\"python\": sys.version.split()[0], **{p: version(p) for p in packages}}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "now = env_record([\"pydantic\", \"httpx\", \"python-dotenv\"])",
   "s": 6
  },
  {
   "t": "PKGS = \"pydantic|openai|langchain-core|langchain-openai|transformers|PyTorch|torch|scikit-learn|sentence-transformers|fastapi|pandas|numpy|httpx\"",
   "s": 6
  },
  {
   "t": "recorded = {}  # 이 노트의 쪽들이 빌드될 때 적어 둔 판",
   "s": 6
  },
  {
   "t": "for f in sorted((HERE.parent.parent / \"topics\").glob(\"*/*-data.js\")):",
   "s": 6
  },
  {
   "t": "    m = re.search(r'\"env\":\\s*\"([^\"]*)\"', f.read_text(encoding=\"utf-8\"))",
   "s": 6
  },
  {
   "t": "    for pkg, ver in re.findall(rf\"\\b({PKGS})\\s+(\\d[\\w.+]*\\w)\", m.group(1) if m else \"\"):",
   "s": 6
  },
  {
   "t": "        recorded.setdefault(pkg, {}).setdefault(ver, []).append(f.stem.replace(\"-data\", \"\"))",
   "s": 6
  },
  {
   "t": "drift = {pkg: vers for pkg, vers in recorded.items() if len(vers) > 1}  # 같은 라이브러리가 쪽마다 다른 판",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 규칙이 정한 것과 모델이 정한 것",
   "s": 7
  },
  {
   "t": "def keyword_route(text):  # 키워드 규칙으로 도구 고르기 (if문이에요, 모델이 아니에요)",
   "s": 7
  },
  {
   "t": "    tools = []",
   "s": 7
  },
  {
   "t": "    if any(w in text for w in (\"규정\", \"어떻게\")):",
   "s": 7
  },
  {
   "t": "        tools.append(\"rag_search\")",
   "s": 7
  },
  {
   "t": "    if re.search(r\"\\d+번\", text):",
   "s": 7
  },
  {
   "t": "        tools.append(\"lookup_reservation\")",
   "s": 7
  },
  {
   "t": "    return {\"tools\": tools, \"decided_by\": \"rule\"}  # 누가 정했는지 함께 적어요",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "EXPECTED = {\"내 101번 예약 상태 알려줘\": [\"lookup_reservation\"], \"스터디룸 철회 규정이 뭐예요?\": [\"rag_search\"],",
   "s": 7
  },
  {
   "t": "            \"이용 안내 3번 항목이 무슨 뜻이에요?\": [\"rag_search\"], \"오늘 하루 어떻게 지냈어?\": []}  # 사람이 정한 기대",
   "s": 7
  },
  {
   "t": "routes = {q: {**keyword_route(q), \"expected\": e} for q, e in EXPECTED.items()}",
   "s": 7
  }
 ],
 "names": {
  "1": "결과에 어떻게 얻었는지 적기",
  "2": "원본을 남기고, 바꾼 것은 따로",
  "3": "내 입력은 엄격하게, 남의 응답은 필요한 것만 엄격하게",
  "4": "설정은 누가 어느 파일을 읽는지부터",
  "5": "재시도는 멱등할 때만",
  "6": "판을 적어 두기",
  "7": "규칙이 정한 것과 모델이 정한 것"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 같은 계산을 다섯 번은 실제로, 다섯 번은 미리 적어 둔 값으로",
   "cols": [
    "모드",
    "값",
    "시간 (평균)"
   ],
   "rows": [
    [
     "live (실제 계산)",
     "8,999,955,000,050,000",
     "9.078 ms"
    ],
    [
     "fixture (미리 적어 둔 값)",
     "8,999,955,000,050,000",
     "0.0001 ms"
    ],
    [
     "<b>모드를 안 보고 섞으면</b>",
     "",
     "<b>4.539 ms</b>"
    ]
   ],
   "notes": [
    "값은 똑같아서, 값만 검사하면 둘 다 '정답'이에요. 시험용 값(fixture)으로도 검사가 통과하는 건 정상이에요. 문제는 그 결과를 실제 결과와 구별할 수 없을 때예요.",
    "모드를 안 보고 시간 평균을 내면 4.54ms로, 실제(9.08ms)의 절반쯤으로 나와요. 결과마다 <code>execution_mode</code>를 적어 두면 실제만 골라 낼 수 있어요. 아예 실행하지 않았다면 <code>not_run</code>으로 남기고, 0이나 빈 값으로 채우지 않아요."
   ]
  },
  "s2": {
   "title": "② 공백을 정리하려다 숫자까지 지운 함수",
   "lines": [
    [
     "원본 → \"스터디룸   예약은 이용 시작 1시간 전까지 철회할 수 있어요.　노쇼가 2번이면 30일 동안 예약할 수 없어요.\"",
     1
    ],
    [
     "덮어쓴 기록 → {\"text\": \"스터디룸 예약은 이용 시작 시간 전까지 철회할 수 있어요. 노쇼가 번이면 일 동안 예약할 수 없어요.\"}",
     2
    ],
    [
     "따로 둔 기록 → original_sha256 cc75f13287e1, cleaned \"스터디룸 예약은 이용 시작 시간 전까지 철회할 수 있어요. 노쇼가 번이면 일 동안 예약할 수 없어요.\"",
     3,
     "newf"
    ],
    [
     "원본과 비교해서 사라진 숫자 → ['1', '2', '30']",
     4,
     "newf"
    ],
    [
     "고친 정리 함수 → \"스터디룸 예약은 이용 시작 1시간 전까지 철회할 수 있어요. 노쇼가 2번이면 30일 동안 예약할 수 없어요.\", 사라진 숫자 []",
     5
    ]
   ],
   "notes": [
    "공백이 여러 개이고, 전각 공백(\\u3000)도 섞인 원문이에요.",
    "같은 칸에 덮어쓰면 남은 건 '1시간'이 '시간'으로, '30일'이 '일'로 바뀐 글뿐이에요. 무엇이 사라졌는지 알 방법이 없어요.",
    "원본과 정리본을 다른 칸에 두고, 원본의 해시(내용으로 만든 짧은 지문)도 적어 둬요. 나중에 원본이 바뀌었는지도 해시로 알 수 있어요.",
    "원본이 있으니 '숫자가 사라졌다'를 기계적으로 잡을 수 있어요. 규정 문서에서 숫자가 사라지면 답이 통째로 틀려져요.",
    "정리 함수를 고친 뒤에도 같은 비교로 확인해요. 바꾸는 단계마다 '바꾸기 전'을 남겨 두면, 어느 단계가 뜻을 바꿨는지 거꾸로 따라갈 수 있어요."
   ]
  },
  "s3a": {
   "title": "③ 우리 API가 받는 요청 (extra=\"forbid\", strict=True)",
   "cols": [
    "요청",
    "결과"
   ],
   "rows": [
    [
     "정상",
     "통과"
    ],
    [
     "오타 칸 houre",
     "missing, extra_forbidden"
    ],
    [
     "글자 &quot;2&quot;",
     "int_type"
    ]
   ],
   "notes": [
    "오타 난 칸 houre는 조용히 버려지지 않고 거부돼요(extra_forbidden). 함께 진짜 칸 hours가 없다는 것(missing)도 알려 줘요. 버렸다면 \"2시간으로 설정했는데 안 먹는다\"가 됐을 거예요."
   ]
  },
  "s3b": {
   "title": "남의 API 응답을 읽을 때",
   "cols": [
    "응답",
    "엄격하게 (forbid)",
    "필요한 것만 (ignore)"
   ],
   "rows": [
    [
     "v1",
     "통과",
     "통과"
    ],
    [
     "v2 (칸 추가)",
     "extra_forbidden",
     "통과"
    ],
    [
     "필수 칸 빠짐",
     "missing",
     "missing"
    ]
   ],
   "notes": [
    "상대가 응답에 칸을 하나 늘렸을 뿐인데, 우리 요청처럼 엄격하게 읽으면 그날부터 서비스가 깨져요. 남의 응답은 <b>필요한 칸만</b> 엄격하게 읽고 나머지는 흘려보내요.",
    "그래도 꼭 필요한 칸(reservation_id)이 빠지면 둘 다 멈춰요. 느슨하게 읽는 것과 빠진 걸 채워 넣는 건 달라요."
   ]
  },
  "s3": {
   "title": "없는 값과 아직 모르는 값",
   "lines": [
    [
     "reply.get(\"reservation_id\", 0) → 0",
     1
    ],
    [
     "reply[\"reservation_id\"] → KeyError",
     2,
     "newf"
    ],
    [
     "evidence_ok = None 에서 not evidence_ok → True",
     3
    ],
    [
     "evidence_ok is False → False",
     4,
     "newf"
    ]
   ],
   "notes": [
    "<code>.get()</code>에 기본값을 주면 없는 예약 번호가 가짜 0으로 채워져, 0번 예약이 있는 것처럼 다음 단계로 넘어가요.",
    "꼭 있어야 하는 값은 <code>[]</code>로 읽어서 없으면 바로 멈추게 해요.",
    "None은 '아직 검사 안 함'인데, <code>not</code>으로 보면 '검사 실패'와 같아져요.",
    "<code>is False</code>로 실패만 실패로 보고, None은 '검토 대기'로 따로 다뤄요."
   ]
  },
  "s4": {
   "title": "④ 같은 DB 설정을 두 방식으로 적은 .env를, 키 셋(DB_HOST, DB_USER, DB_PASSWORD)을 읽는 프로그램이 읽으면",
   "cols": [
    "파일",
    "읽은 키 (값은 가림)",
    "없는 키",
    "예시값 그대로"
   ],
   "rows": [
    [
     "url_style.env",
     "{&quot;DATABASE_URL&quot;: &quot;***&quot;}",
     "DB_HOST, DB_USER, DB_PASSWORD",
     "없음"
    ],
    [
     "key_style.env",
     "{&quot;DB_HOST&quot;: &quot;localhost&quot;, &quot;DB_USER&quot;: &quot;reader&quot;, &quot;DB_PASSWORD&quot;: &quot;***&quot;}",
     "없음",
     "DB_PASSWORD"
    ]
   ],
   "notes": [
    "URL 한 줄로 적은 파일은 정보가 다 들어 있는데도, 이 프로그램이 찾는 키가 하나도 없어요. 다른 프로젝트의 .env를 복사해 왔을 때 \"값을 넣었는데 설정하라고 나온다\"의 정체예요.",
    "키 방식은 키는 다 있지만 비밀번호가 예시값(CHANGE_ME) 그대로예요. 시작할 때 이걸 잡아 멈추면, DB 접속 실패라는 엉뚱한 증상으로 헤매지 않아요.",
    "무엇을 읽었는지 출력할 때는 비밀값을 가려요(***). 지금 실행 중인 파이썬은 python.exe이고 가상환경 안이에요(True). 패키지를 설치한 파이썬과 실행하는 파이썬이 같은지 확인하는 게 \"설치했는데 import가 안 된다\"의 첫 점검이에요."
   ]
  },
  "s5": {
   "title": "⑤ 첫 응답만 늦는 서버에 예약을 보내고, 시간 초과면 다시 보내기",
   "cols": [
    "클라이언트",
    "보낸 횟수",
    "받은 답",
    "서버에 생긴 예약"
   ],
   "rows": [
    [
     "재시도만",
     2,
     "{&quot;reservation_id&quot;: 102}",
     "101, 102"
    ],
    [
     "재시도 + Idempotency-Key",
     2,
     "{&quot;reservation_id&quot;: 101}",
     "101"
    ]
   ],
   "notes": [
    "첫 요청은 서버가 예약을 저장했는데 응답만 늦었어요. 클라이언트는 실패로 보고 다시 보냈고, 서버는 새 요청으로 보고 또 저장했어요. 예약이 두 개 생겼는데, 사용자는 하나만 했다고 알아요.",
    "요청마다 고유한 키(Idempotency-Key)를 붙이고 서버가 같은 키를 기억하면, 다시 보내도 처음 결과를 돌려줘요. 이렇게 몇 번을 보내도 결과가 같은 성질을 <b>멱등</b>이라고 해요.",
    "조회는 원래 멱등이라 재시도해도 괜찮지만, 예약, 결제, 메일 보내기처럼 무언가를 바꾸는 요청은 멱등하게 만들기 전에는 자동 재시도를 켜지 않아요. SDK의 기본 재시도(예: <code>max_retries</code>)가 이미 켜져 있는지도 확인해요."
   ]
  },
  "s6": {
   "title": "⑥ 이 노트의 쪽들이 빌드될 때 적어 둔 판 (판이 둘 이상인 라이브러리 5개)",
   "cols": [
    "라이브러리",
    "판: 그 판으로 빌드한 쪽 수"
   ],
   "rows": [
    [
     "PyTorch",
     "2.13.0+cpu (6쪽), 2.14.0+cu130 (13쪽)"
    ],
    [
     "pandas",
     "2.2.3 (3쪽), 2.3.3 (3쪽)"
    ],
    [
     "pydantic",
     "2.13.4 (4쪽), 2.14.3 (1쪽)"
    ],
    [
     "scikit-learn",
     "1.8.0 (4쪽), 1.9.0 (3쪽)"
    ],
    [
     "transformers",
     "5.15.0 (1쪽), 5.22.2 (7쪽)"
    ]
   ],
   "notes": [
    "같은 노트 안에서도 라이브러리 5개가 쪽마다 다른 판으로 빌드됐어요. PC를 옮기거나 패키지를 새로 설치하는 사이에 판이 바뀌었기 때문이에요. 쪽마다 '실행 환경' 줄이 있어서 이걸 알 수 있어요.",
    "판이 바뀌면 같은 코드도 결과가 달라질 수 있어요. 실제로 <a href=\"../backend/rag-eval.html#s1\">RAG 평가 ①</a>의 정답 묶음에는 묶음의 판과 조각의 판을 같이 적어 둬요. 조각을 다시 나누면 같은 조각 ID가 다른 내용을 가리키거나 사라질 수 있어서, 판이 다른 점수끼리는 나란히 비교하지 않아요.",
    "그래서 결과에는 판을 같이 적고, 프로젝트에는 판을 고정한 목록(requirements.txt에 ==로, 또는 lock 파일)을 둬요. 이번 실행의 판: {&quot;python&quot;: &quot;3.13.16&quot;, &quot;pydantic&quot;: &quot;2.13.4&quot;, &quot;httpx&quot;: &quot;0.28.1&quot;, &quot;python-dotenv&quot;: &quot;1.2.2&quot;}"
   ]
  },
  "s7": {
   "title": "⑦ 키워드 규칙으로 도구 고르기 (decided_by: rule)",
   "cols": [
    "질문",
    "규칙이 고른 도구",
    "기대",
    "맞나"
   ],
   "rows": [
    [
     "내 101번 예약 상태 알려줘",
     "lookup_reservation",
     "lookup_reservation",
     "맞음"
    ],
    [
     "스터디룸 철회 규정이 뭐예요?",
     "rag_search",
     "rag_search",
     "맞음"
    ],
    [
     "이용 안내 3번 항목이 무슨 뜻이에요?",
     "lookup_reservation",
     "rag_search",
     "<b>틀림</b>"
    ],
    [
     "오늘 하루 어떻게 지냈어?",
     "rag_search",
     "없음",
     "<b>틀림</b>"
    ]
   ],
   "notes": [
    "'3번 항목'의 '번' 때문에 예약 조회로, '어떻게 지냈어'의 '어떻게' 때문에 규정 검색으로 갔어요. 키워드 규칙은 빠르고 예측 가능하지만, 말이 조금만 달라도 틀려요.",
    "이 결정은 if문이 내렸어요. 기록에 decided_by를 남기지 않으면, 이 실수를 '모델이 도구를 잘못 골랐다'로 오해하고 Prompt를 고치게 돼요. 반대로 모델이 규칙과 다르게 골랐을 때 그걸 버그로 보기도 해요.",
    "<a href=\"../backend/agent-tools.html#s4\">AI Agent와 도구 ④</a>의 '모델 판단'도 사실 가짜 함수(PLANS)의 이런 규칙이에요. 그 쪽이 이걸 처음부터 밝혀 둔 이유예요."
   ]
  }
 },
 "flow": [
  [
   1,
   "실행 모드",
   "live, fixture, not_run",
   "섞으면 평균이 반"
  ],
  [
   2,
   "원본 보존",
   "따로 두고 비교",
   "사라진 숫자 1, 2, 30"
  ],
  [
   3,
   "계약",
   "내 입력 엄격, 남의 응답 필요한 것만",
   "칸 하나에 깨짐"
  ],
  [
   4,
   "설정",
   "누가 어느 키를 읽나",
   "URL 방식은 키 0개"
  ],
  [
   5,
   "재시도",
   "멱등할 때만",
   "예약 두 개"
  ],
  [
   6,
   "판",
   "결과에 적기",
   "판이 갈린 라이브러리 5"
  ],
  [
   7,
   "누가 정했나",
   "규칙 vs 모델",
   "규칙이 2개 틀림"
  ]
 ],
 "env": "Python 3.13.16, pydantic 2.13.4, httpx 0.28.1, python-dotenv 1.2.2"
};
