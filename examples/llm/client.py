# LLM API 클라이언트 전체 코드 (OpenAI Python SDK, Responses API). 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 실행: 환경 변수 OPENAI_API_KEY, OPENAI_MODEL 을 정한 뒤 python client.py  (실제로 호출하면 비용이 들어요)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import os                                                          #@1
from openai import OpenAI                                          #@1
from typing import Literal                                         #@3
from pydantic import BaseModel, Field                              #@3
import openai                                                      #@4
from decimal import Decimal                                        #@6

# ① 클라이언트 만들기
MODEL = os.environ["OPENAI_MODEL"]
client = OpenAI(timeout=30.0, max_retries=2)  # 키는 환경 변수 OPENAI_API_KEY 에서 읽어요

# ② 질문 보내고 답 읽기
question = "API를 한 문장으로 설명해 줘."
response = client.responses.create(
    model=MODEL,
    instructions="초보자에게 한국어로 짧게 설명해.",
    input=question,
)
answer = response.output_text

# ③ 정해진 모양으로 받기 (구조화 출력)
class Explain(BaseModel):
    term: str
    one_line: str = Field(max_length=120)
    level: Literal["easy", "normal"]

parsed = client.responses.parse(
    model=MODEL,
    input=question,
    text_format=Explain,
).output_parsed

# ④ 실패를 종류별로 다루기
def ask(text):
    try:
        r = client.responses.create(model=MODEL, input=text)
        return "ok", r.output_text
    except openai.AuthenticationError:
        return "stop", "키를 확인하세요"  # 다시 보내도 같은 결과
    except openai.RateLimitError as e:
        return "later", e.code  # rate_limit_exceeded 면 잠시 뒤, insufficient_quota 면 결제 확인
    except openai.APITimeoutError:
        return "later", "timeout"
    except openai.APIStatusError as e:
        return "error", e.status_code

# ⑤ 나오는 대로 받기 (스트리밍)
pieces = []
with client.responses.stream(model=MODEL, input=question) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            pieces.append(event.delta)  # 화면에 바로 이어 붙여 보여 줄 수 있어요
    final = stream.get_final_response()

# ⑥ 사용량과 비용 기록하기
PRICE = {"input": Decimal("0.75"), "output": Decimal("4.50")}  # 예시 단가: 100만 토큰당 달러

usage = response.usage
cost = (usage.input_tokens * PRICE["input"] + usage.output_tokens * PRICE["output"]) / 1_000_000
