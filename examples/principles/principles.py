# 만들 때 지킬 것: 원칙을 일부러 어겨 보고 무엇이 망가지는지 보는 작은 예제들. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 단계마다 서로 독립이에요. 도서관 스터디룸 예약이라는 가상의 상황을 써요(실제 서비스나 DB에 연결하지 않아요).
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import hashlib                                                   #@2
import json                                                      #@3
import re                                                        #@2
import statistics                                                #@1
import sys                                                       #@4
import tempfile                                                  #@4
import time                                                      #@1
import unicodedata                                               #@2
from importlib.metadata import version                           #@6
from pathlib import Path                                         #@4
from typing import Literal                                       #@3
import httpx                                                     #@5
from dotenv import dotenv_values                                 #@4
from pydantic import BaseModel, ConfigDict, ValidationError      #@3

HERE = Path(__file__).resolve().parent                           #@6

# ① 결과에 어떻게 얻었는지 적기
def measure(fn, mode):                                           # 값과 시간에 '어떻게 얻었나'를 함께 적어요
    start = time.perf_counter()
    value = fn()
    return {"value": value, "ms": (time.perf_counter() - start) * 1000, "execution_mode": mode}

def live_work():                                                 # 실제로 계산
    return sum(i * i for i in range(300_000))

def fixture_work():                                              # 시험용으로 미리 적어 둔 값 (계산하지 않아요)
    return 8999955000050000

runs = [measure(live_work, "live") for _ in range(5)] + [measure(fixture_work, "fixture") for _ in range(5)]
mixed_ms = statistics.mean(r["ms"] for r in runs)                # 모드를 안 보고 평균을 내면
live_ms = statistics.mean(r["ms"] for r in runs if r["execution_mode"] == "live")
same_value = len({r["value"] for r in runs}) == 1                # 값만 보면 둘 다 '정답'이에요

# ② 원본을 남기고, 바꾼 것은 따로
original = "스터디룸   예약은 이용 시작 1시간 전까지 철회할 수 있어요.　노쇼가 2번이면 30일 동안 예약할 수 없어요."

def clean_buggy(text):                                           # 공백을 정리하려다 숫자까지 지운 정리 함수 (흔한 실수)
    return re.sub(r"[^가-힣\s.]", "", unicodedata.normalize("NFC", text)).split()

def clean(text):
    return " ".join(unicodedata.normalize("NFC", text).split())

overwritten = {"text": original}
overwritten["text"] = " ".join(clean_buggy(overwritten["text"]))   # 같은 칸에 덮어쓰기
kept = {"original": original, "original_sha256": hashlib.sha256(original.encode()).hexdigest()[:12],
        "cleaned": " ".join(clean_buggy(original))}              # 원본과 바꾼 것을 따로
lost_numbers = sorted(set(re.findall(r"\d+", kept["original"])) - set(re.findall(r"\d+", kept["cleaned"])))   # 원본이 있어야 비교할 수 있어요
fixed = clean(original)
fixed_lost = sorted(set(re.findall(r"\d+", original)) - set(re.findall(r"\d+", fixed)))

# ③ 내 입력은 엄격하게, 남의 응답은 필요한 것만 엄격하게
class ReserveRequest(BaseModel):                                 # 우리 API가 받는 요청: 모르는 칸은 거부
    model_config = ConfigDict(extra="forbid", strict=True)
    room: str
    hours: int

checks = {}
for name, body in {"정상": {"room": "A", "hours": 2}, "오타 칸 houre": {"room": "A", "houre": 2}, "글자 \"2\"": {"room": "A", "hours": "2"}}.items():
    try:
        ReserveRequest.model_validate(body)
        checks[name] = "통과"
    except ValidationError as exc:
        checks[name] = ", ".join(e["type"] for e in exc.errors())

class PartnerStrict(BaseModel):                                  # 남의 API 응답을 우리 요청처럼 엄격하게 읽으면
    model_config = ConfigDict(extra="forbid")
    status: Literal["confirmed", "rejected"]
    reservation_id: int

class PartnerTolerant(BaseModel):                                # 필요한 칸만 엄격하게, 나머지는 흘려보내기
    model_config = ConfigDict(extra="ignore")
    status: Literal["confirmed", "rejected"]
    reservation_id: int

reply_v1 = {"status": "confirmed", "reservation_id": 101}
reply_v2 = {"status": "confirmed", "reservation_id": 101, "estimated_wait_min": 5}   # 상대가 칸을 하나 늘림
reply_broken = {"status": "confirmed"}                           # 꼭 필요한 칸이 빠짐
partner = {}
for name, reply in {"v1": reply_v1, "v2 (칸 추가)": reply_v2, "필수 칸 빠짐": reply_broken}.items():
    for model in (PartnerStrict, PartnerTolerant):
        try:
            model.model_validate(reply)
            partner[name, model.__name__] = "통과"
        except ValidationError as exc:
            partner[name, model.__name__] = exc.errors()[0]["type"]
quiet_default = reply_broken.get("reservation_id", 0)            # .get()은 없는 값을 가짜 0으로 채워요
try:
    reply_broken["reservation_id"]
    loud = "읽힘"
except KeyError:
    loud = "KeyError"                                            # []는 없으면 바로 멈춰요
review = {"evidence_ok": None}                                   # 아직 검사 안 함
none_as_fail = not review["evidence_ok"]                         # True: '검사 실패'로 잘못 분류
none_kept = review["evidence_ok"] is False                       # False: '실패'만 실패로

# ④ 설정은 누가 어느 파일을 읽는지부터
tmp = Path(tempfile.mkdtemp())
(tmp / "url_style.env").write_text("DATABASE_URL=postgresql://reader:example-pass@localhost:5432/library\n", encoding="utf-8")
(tmp / "key_style.env").write_text("DB_HOST=localhost\nDB_USER=reader\nDB_PASSWORD=CHANGE_ME\n", encoding="utf-8")
NEEDED = ("DB_HOST", "DB_USER", "DB_PASSWORD")                   # 이 프로그램이 읽는 키

def load_settings(path):                                         # 무엇을 읽었는지 보고하되 값은 가려요
    values = dotenv_values(path)
    return {"file": path.name, "keys": sorted(values), "missing": [k for k in NEEDED if not values.get(k)],
            "placeholder": [k for k, v in values.items() if v in ("CHANGE_ME", "your_api_key", "")],
            "shown": {k: ("***" if any(s in k for s in ("PASSWORD", "URL", "KEY", "TOKEN")) else v) for k, v in values.items()}}

settings = [load_settings(tmp / "url_style.env"), load_settings(tmp / "key_style.env")]
interpreter = {"name": Path(sys.executable).name, "in_venv": sys.prefix != sys.base_prefix}   # 설치와 실행에 같은 파이썬을 쓰나

# ⑤ 재시도는 멱등할 때만
def make_server():                                               # 예약을 저장한 뒤, 첫 응답만 늦게 보내 클라이언트가 시간 초과로 보게 하는 가짜 서버
    state = {"reservations": [], "by_key": {}, "calls": 0}

    def handle(request):
        state["calls"] += 1
        key = request.headers.get("Idempotency-Key")
        if key and key in state["by_key"]:                       # 같은 키의 요청은 새로 만들지 않고 처음 결과를 돌려줘요
            return httpx.Response(201, json=state["by_key"][key])
        rid = 100 + len(state["reservations"]) + 1
        state["reservations"].append(rid)
        if key:
            state["by_key"][key] = {"reservation_id": rid}
        if state["calls"] == 1:
            raise httpx.ReadTimeout("응답이 늦어요", request=request)   # 저장은 됐는데 응답이 안 옴
        return httpx.Response(201, json={"reservation_id": rid})
    return state, httpx.Client(transport=httpx.MockTransport(handle))

def reserve_with_retry(client, headers):                         # 시간 초과면 한 번 더 보내기
    for attempt in (1, 2, 3):
        try:
            return client.post("https://library.example/reservations", json={"room": "A", "hours": 2}, headers=headers).json(), attempt
        except httpx.ReadTimeout:
            continue
    raise RuntimeError("세 번 모두 실패")

plain_state, plain_client = make_server()
plain_result = reserve_with_retry(plain_client, {})
keyed_state, keyed_client = make_server()
keyed_result = reserve_with_retry(keyed_client, {"Idempotency-Key": "req-7f3a"})

# ⑥ 판을 적어 두기
def env_record(packages):                                        # 이번 실행의 판: 파이썬과 라이브러리
    return {"python": sys.version.split()[0], **{p: version(p) for p in packages}}

now = env_record(["pydantic", "httpx", "python-dotenv"])
PKGS = "pydantic|openai|langchain-core|langchain-openai|transformers|PyTorch|torch|scikit-learn|sentence-transformers|fastapi|pandas|numpy|httpx"
recorded = {}                                                    # 이 노트의 쪽들이 빌드될 때 적어 둔 판
for f in sorted((HERE.parent.parent / "topics").glob("*/*-data.js")):
    m = re.search(r'"env":\s*"([^"]*)"', f.read_text(encoding="utf-8"))
    for pkg, ver in re.findall(rf"\b({PKGS})\s+(\d[\w.+]*\w)", m.group(1) if m else ""):
        recorded.setdefault(pkg, {}).setdefault(ver, []).append(f.stem.replace("-data", ""))
drift = {pkg: vers for pkg, vers in recorded.items() if len(vers) > 1}   # 같은 라이브러리가 쪽마다 다른 판

# ⑦ 규칙이 정한 것과 모델이 정한 것
def keyword_route(text):                                         # 키워드 규칙으로 도구 고르기 (if문이에요, 모델이 아니에요)
    tools = []
    if any(w in text for w in ("규정", "어떻게")):
        tools.append("rag_search")
    if re.search(r"\d+번", text):
        tools.append("lookup_reservation")
    return {"tools": tools, "decided_by": "rule"}                # 누가 정했는지 함께 적어요

EXPECTED = {"내 101번 예약 상태 알려줘": ["lookup_reservation"], "스터디룸 철회 규정이 뭐예요?": ["rag_search"],
            "이용 안내 3번 항목이 무슨 뜻이에요?": ["rag_search"], "오늘 하루 어떻게 지냈어?": []}   # 사람이 정한 기대
routes = {q: {**keyword_route(q), "expected": e} for q, e in EXPECTED.items()}
