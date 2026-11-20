"""주제 페이지 '백엔드 > LLM 활용 > Prompt와 출력 계약'을 만든다.

examples/llm/answer_check.py 를 단계별로 실제 실행한다. 모델은 부르지 않고 samples.json 의 미리 정한 응답을 쓴다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/prompt_contract.py
"""
import json
import sys
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import pydantic  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "llm"
code = S.parse((EX / "answer_check.py").read_text(encoding="utf-8"))
SAMPLES = json.loads((EX / "samples.json").read_text(encoding="utf-8"))
outs = {}
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="answer_check.py")  # noqa: E731

ns = run(6)
rows = []
for key, s in SAMPLES.items():
    obj, err = ns["validate"](s["raw"])
    where = {"json_error": "④ JSON 읽기", "schema_error": "④ 모양 검사", None: "통과"}.get(err, "⑤ 내용 규칙")
    rows.append([E(s["label"]), f"<code>{E(s['raw'][:70])}{'…' if len(s['raw']) > 70 else ''}</code>",
                 "<b>통과</b>" if err is None else f"<span class=\"bad\">{err}</span>", where])
outs["why"] = S.table("모델이 낼 수 있는 응답 여덟 가지를 검사하면", ["응답", "모델이 보낸 글", "결과", "걸린 곳"], rows,
                      ["모델의 답은 언제나 <b>그냥 글자</b>예요. 괄호 하나 빠지거나, 번호를 글자로 쓰거나, 주지 않은 자료를 근거로 대는 일이 실제로 생겨요. 위 응답들은 그런 실수를 흉내 내 미리 만든 샘플이에요.",
                       "검사 없이 화면에 보여 주면 이 중 여섯 개가 사용자에게 그대로 나가요."])

ns = run(1)
outs["s1"] = S.box("build_context(KNOWLEDGE, [101])", [(f"context → {ns['context']}", 1), f"question → {ns['question']}"],
                   ["세 자료 중 고른 101번만, <b>번호와 함께</b> 넣었어요. 번호가 있어야 모델이 \"어느 자료를 근거로 했는지\" 답할 수 있어요."])
try:
    ns["build_context"](ns["KNOWLEDGE"], [101, 999])
    e1 = "오류 없음"
except ValueError as e:
    e1 = f"ValueError: {e}"
outs["t1"] = S.box("고르는 번호가 이상하면", [f"build_context(..., [102, 101, 102]) → {ns['build_context'](ns['KNOWLEDGE'], [102, 101, 102])}", (f"build_context(..., [101, 999]) → {e1}", 1)],
                   ["중복은 순서를 지키며 빼고, 없는 번호는 Prompt를 만들기 전에 멈춰요. 잘못된 자료로 모델을 부르면 비용만 쓰고 답도 틀려요."])

ns = run(2)
outs["s2"] = S.box("prompt (모델에 보낼 글 전체)", ns["prompt"].splitlines() + ["", (f"prompt_version → {ns['prompt_version']}", 1)],
                   ["틀(<code>TEMPLATE</code>)의 지문 앞 8자리. 틀을 한 글자라도 고치면 바뀌어서, 나중에 \"이 답은 어느 판의 Prompt로 받았나\"를 기록할 수 있어요."])
ns2 = run(2, patch=[("[역할] 너는 실습 안내 도우미야.", "[역할] 너는 친절한 실습 안내 도우미야.")])
outs["t2"] = S.box("틀에 \"친절한\" 한 단어를 넣으면", [f"고치기 전 → {ns['prompt_version']}", (f"고친 뒤   → {ns2['prompt_version']}", 1)],
                   ["완전히 다른 값이에요. Prompt를 바꿔 결과가 좋아졌는지 비교할 때(A/B), 판 번호로 어느 쪽 결과인지 구분해요."])

ns = run(3)
sch = ns["KnowledgeAnswer"].model_json_schema()
outs["s3"] = S.box("KnowledgeAnswer.model_json_schema() 중 일부", [(f"required → {sch['required']}", 1), (f"additionalProperties → {sch['additionalProperties']}", 2),
                    (f"status → {json.dumps(sch['properties']['status'], ensure_ascii=False)}", 3), f"source_ids → {json.dumps(sch['properties']['source_ids'], ensure_ascii=False)}"],
                   ["네 필드 모두 꼭 있어야 해요.", "<code>extra=\"forbid\"</code> 때문에 다른 필드는 금지예요.", "<code>Literal</code>은 \"이 두 값 중 하나\"가 돼요. 이 스키마를 그대로 모델에게 주면 그 모양으로 답하게 할 수도 있어요 (<a href=\"llm-client.html\">LLM API 클라이언트</a>의 구조화 출력)."])

ns = run(4)
rows = []
for key in ["good", "broken", "string_id", "extra"]:
    raw = SAMPLES[key]["raw"]
    obj, err = ns["parse_answer"](raw)
    detail = ""
    if err == "schema_error":
        try:
            ns["KnowledgeAnswer"].model_validate(json.loads(raw))
        except ns["ValidationError"] as e:
            x = e.errors()[0]
            detail = f"{x['loc']}: {x['msg']}"
    rows.append([E(SAMPLES[key]["label"]), "<b>통과</b>" if err is None else f"<span class=\"bad\">{err}</span>", E(detail)])
outs["s4"] = S.table("parse_answer() 실행 결과", ["응답", "결과", "Pydantic이 알려 준 이유"], rows,
                     ["\"101\"(글자)과 101(숫자)은 다른 값이에요. <code>strict=True</code>라서 글자를 숫자로 몰래 바꿔 주지 않고 거부해요.", "오류를 <code>json_error</code>, <code>schema_error</code>처럼 <b>종류별 이름</b>으로 돌려줘서, ⑥에서 다시 물을지 정할 수 있어요."])

ns = run(5)
rows = []
for key in ["good", "missing", "unknown", "duplicate", "status"]:
    obj, err = ns["parse_answer"](SAMPLES[key]["raw"])
    r = ns["check_rules"](obj, ns["provided_ids"])
    rows.append([E(SAMPLES[key]["label"]), "<code>" + E(str(obj.source_ids)) + "</code>", E(obj.status), "<b>통과</b>" if r is None else f"<span class=\"bad\">{r}</span>"])
S.claim(all(r_[3] != "<b>통과</b>" for r_ in rows if "999" in r_[1]), "⑤ 999를 근거로 댄 답은 내용 규칙에 걸린다")
outs["s5"] = S.table("모양 검사를 통과한 응답에 check_rules()", ["응답", "source_ids", "status", "결과"], rows,
                     ["모두 모양은 맞아서 ④는 통과했어요. 하지만 \"주지 않은 999번을 근거로 댄\" 답은 그럴듯해 보여도 출처를 지어낸 거예요. 이런 건 <b>내용 규칙</b>으로만 잡혀요.",
                      "규칙의 기준은 ①에서 실제로 넣은 번호(<code>provided_ids</code>)예요."])

ns = run(6)
S.claim(len(ns["history"]) == 2 and "json_error" in str(ns["history"][0]), "⑥ 첫 응답은 json_error로 한 번 더 묻고, 두 번째에 통과")
outs["s6"] = S.box("run_with_retry(prompt)  (첫 응답: 괄호 빠짐, 두 번째: 정상)", [(f"history → {ns['history']}", 1), (f"final → {ns['final']!r}", 2)],
                   ["첫 응답은 JSON이 깨져서(<code>json_error</code>) 무엇이 틀렸는지 덧붙여 한 번 더 물었어요.", "두 번째 응답은 모든 검사를 통과했어요."])
ns2 = run(6, patch=[('script=iter(["broken", "good"])', 'script=iter(["unknown", "good"])')])
S.claim(len(ns2["history"]) == 1, "내용 규칙 위반은 다시 묻지 않고 바로 멈춘다")
outs["t6"] = S.box("첫 응답이 '주지 않은 자료 999를 출처로'였다면", [(f"history → {ns2['history']}", 1), f"final → {ns2['final']!r}"],
                   ["내용 규칙 위반은 다시 묻지 않고 바로 멈췄어요. 같은 자료로 다시 물으면 같은 착각을 하기 쉬워서, 이런 건 사람이 보거나 자료를 바꿔야 해요. 몇 번이고 다시 물으면 비용만 늘어요."])

flow = [[1, "자료", "골라 넣기", "번호와 함께"], [2, "Prompt", "틀 채우기", "판 번호"], [3, "계약", "Pydantic", "답의 모양"],
        [4, "모양 검사", "JSON, 스키마", "json_error, schema_error"], [5, "내용 검사", "출처, 상태", "unknown_source 등"], [6, "다시 묻기", "모양만 1번", "내용 위반은 멈춤"]]
S.save(ROOT / "topics" / "backend" / "prompt-contract-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                                  "env": f"Python {sys.version.split()[0]}, Pydantic {pydantic.__version__}. 모델은 부르지 않고 samples.json 의 미리 정한 응답 사용"})

why = f"""
<p>LLM(대규모 언어 모델)에게 질문을 보내면 답이 <b>글</b>로 와요. 사람이 읽기엔 충분하지만, 서비스는 그 답을 화면에 보여 주고, 근거 자료를 링크하고, 모르는 질문이면 담당자에게 넘겨야 해요. 그러려면 답이 <b>정해진 모양</b>이어야 하고, 그 모양을 지켰는지 <b>프로그램이 검사</b>할 수 있어야 해요.</p>
<p>예제는 "파일은 어디에 보관하나요?"라는 질문에 자료 101번("실습 파일은 개인 폴더에 보관합니다.")을 근거로 답하게 해요. 답은 이런 JSON으로 받기로 약속해요.</p>
<pre><code>{{"status": "answered", "answer": "개인 폴더에 보관합니다.", "source_ids": [101], "missing_information": ""}}</code></pre>
<p>그런데 모델은 약속을 어길 수 있어요. 여러 가지로 어긴 응답을 실제 검사 코드에 넣어 보면:</p>
{K.out("why")}
<p><b>그래서 할 일은</b> 모델에게 줄 글(<b>Prompt</b>)과 근거 자료(<b>Context</b>)를 정해진 틀로 만들고, 받은 답을 <b>모양 검사</b>와 <b>내용 검사</b> 두 겹으로 거르는 거예요.</p>
"""
whole = f"""
<p>자료를 고르고, Prompt를 채워 모델에게 보내고, 돌아온 글을 두 번 검사해요. 모양이 틀렸을 때만 한 번 다시 물어요. 상자를 누르면 그 단계로 가요.</p>
<div class="flow" id="flow"></div>
<p>이 페이지는 <b>모델을 실제로 부르지 않아요</b>. 모델에게 보내는 부분은 <code>ask()</code> 함수 하나뿐이고, 여기서는 미리 정한 응답(<code>samples.json</code>)을 돌려줘요. 검사 코드는 모두 실제로 실행한 결과예요. 실제 모델을 부르는 방법은 <a href="llm-client.html">LLM API 클라이언트</a>에서 다뤄요.</p>
"""
steps = [
    K.step(1, "근거 자료 고르기", "질문에 쓸 자료만 골라, 번호와 함께 JSON 글로 만들어요.",
           "<p>모델은 Prompt에 들어 있는 것만 근거로 쓸 수 있어요. 자료를 다 넣으면 길어지고 엉뚱한 자료를 섞어 쓰기 쉬워서, 질문에 맞는 것만 골라 넣어요. 이렇게 모델에게 주는 배경 자료를 <b>Context</b>라고 해요. 어떤 자료를 고를지는 보통 검색이 정해요 (<a href=\"indexing.html\">인덱싱</a>).</p>"
           + K.code_table([("dict.fromkeys(selected_ids)", "순서를 지키면서 중복을 빼는 관용구"),
                           ("json.dumps([...], ensure_ascii=False)", "번호와 내용을 JSON 글로. 모델이 \"101번 자료\"라고 정확히 가리킬 수 있어요.")])
           + K.try_("같은 번호를 두 번 고르거나, 없는 번호를 고르면?", "t1"), "s1"),
    K.step(2, "Prompt 틀 채우기", "역할, 작업, 질문, 자료, 규칙, 출력 모양을 정해진 틀에 채워요.",
           "<p>Prompt를 그때그때 손으로 쓰면 빠뜨리는 게 생기고, 바꿨을 때 무엇이 달라졌는지 알기 어려워요. 그래서 틀을 하나 정해 두고 빈칸만 채워요. 틀에는 <b>무엇을 하라</b>뿐 아니라 <b>모르면 어떻게 하라</b>와 <b>어떤 모양으로 답하라</b>까지 적어요.</p>"
           + K.code_table([("Template(\"... $question ...\")", "<code>$이름</code> 자리를 나중에 채우는 틀 (표준 라이브러리 string)"),
                           ("TEMPLATE.substitute(question=..., context=...)", "빈칸을 채워요. 빠진 이름이 있으면 오류가 나서 실수를 막아요."),
                           ("hashlib.sha256(...).hexdigest()[:8]", "틀의 지문. 틀이 바뀌면 값이 바뀌어요.")])
           + K.try_("틀에 \"친절한\" 한 단어만 넣으면 판 번호는?", "t2"), "s2"),
    K.step(3, "출력 계약 정하기", "받을 답의 모양을 Pydantic 클래스로 적어요. 이게 모델과의 <b>계약</b>이에요.",
           K.code_table([("Literal[\"answered\", \"insufficient_context\"]", "이 두 값만 허용"),
                         ("Annotated[int, Field(strict=True, gt=0)]", "0보다 큰 정수만. <code>strict</code>라서 글자 \"101\"은 안 돼요."),
                         ("Field(max_length=3)", "목록에 3개까지"),
                         ("ConfigDict(extra=\"forbid\", strict=True)", "정하지 않은 필드는 금지, 타입은 바꿔 주지 않고 엄격하게"),
                         ("model_json_schema()", "이 계약을 JSON 스키마로. 모델에게 줄 수도 있어요.")]), "s3"),
    K.step(4, "응답 읽고 모양 검사하기", "받은 글을 JSON으로 읽고, 계약 모양에 맞는지 검사해요. 틀리면 이유를 이름으로 돌려줘요.",
           K.code_table([("json.loads(raw)", "글을 JSON으로. 괄호가 빠지면 <code>ValueError</code>"),
                         ("KnowledgeAnswer.model_validate(data)", "계약 모양에 맞는지 검사. 틀리면 <code>ValidationError</code>"),
                         ("return obj, None / return None, \"json_error\"", "결과와 오류 이름을 함께 돌려줘요. 예외를 그대로 던지지 않아서, 부르는 쪽이 종류별로 처리할 수 있어요.")]), "s4"),
    K.step(5, "내용 규칙 검사하기", "모양이 맞아도 내용이 약속을 어겼는지 확인해요.",
           "<p>모양 검사는 \"번호가 정수인가\"까지만 알아요. \"그 번호가 <b>내가 준 자료</b>인가\"는 이번 요청에서 무엇을 줬는지 알아야 판단할 수 있어서, 따로 검사해요.</p>"
           + K.code_table([("set(ids) <= set(provided_ids)", "답의 출처가 모두 내가 준 번호 안에 있나 (부분집합)"),
                           ("len(ids) != len(set(ids))", "같은 번호를 두 번 댔나"),
                           ("status == \"answered\" and (not ids or ...)", "답했다면 출처가 있어야 하고, 모른다면 출처가 없고 무엇이 부족한지 적어야 해요.")]), "s5"),
    K.step(6, "모양이 틀렸을 때만 한 번 다시 묻기", "JSON이 깨졌거나 모양이 틀렸으면, 무엇이 틀렸는지 덧붙여 한 번만 다시 물어요.",
           K.code_table([("for attempt in (1, 2)", "최대 두 번. 횟수를 정해 두지 않으면 비용이 끝없이 늘 수 있어요."),
                         ("if error not in {\"json_error\", \"schema_error\"}: break", "내용 규칙 위반은 다시 묻지 않아요."),
                         ("ask(prompt)", "모델에게 보내는 자리. 여기서는 미리 정한 응답을 순서대로 돌려줘요.")])
           + K.try_("첫 응답이 '주지 않은 자료 999를 출처로'였다면?", "t6"), "s6"),
]
practice = [
    ("모델에게 처음부터 모양을 지키게 하고 싶다", "API의 구조화 출력 기능에 ③의 스키마를 넘기면, 모델이 그 모양으로만 답하게 할 수 있어요 (<a href=\"llm-client.html\">LLM API 클라이언트</a>). 그래도 ⑤의 내용 검사는 따로 해요. 모양을 지킨다고 출처가 맞는 건 아니니까요."),
    ("Prompt를 고쳐 본다", "한 번에 한 가지만 바꾸고, ②의 판 번호와 함께 결과를 기록해서 비교해요. 여러 개를 한꺼번에 바꾸면 무엇 때문에 좋아졌는지 몰라요."),
    ("예시를 넣고 싶다 (Few-shot)", "Prompt에 \"이런 질문엔 이렇게\" 예시를 몇 개 넣으면 모양을 더 잘 지켜요. 예시 속 답을 현재 질문에 그대로 베끼지 않게, 예시는 다른 주제로 골라요."),
    ("검사에 걸린 답", "사용자에게는 \"답을 만들지 못했어요\"처럼 안전한 문구를 보여 주고, 원래 응답과 오류 이름은 기록해 둬요. 어떤 실수가 자주 나는지 보고 Prompt를 고쳐요."),
]
libs = [("①", "<code>json.dumps()</code>, <code>dict.fromkeys()</code>", "자료를 JSON 글로 (표준 라이브러리)"),
        ("②", "<code>string.Template</code>, <code>substitute()</code>, <code>hashlib.sha256()</code>", "틀 채우기, 판 번호"),
        ("③", "<code>BaseModel</code>, <code>Field()</code>, <code>ConfigDict()</code>, <code>Literal</code>, <code>Annotated</code>, <code>model_json_schema()</code>", "출력 계약 (pydantic, typing)"),
        ("④", "<code>json.loads()</code>, <code>model_validate()</code>, <code>ValidationError</code>, <code>e.errors()</code>", "모양 검사")]
K.build(ROOT / "topics" / "backend" / "prompt-contract.html", title="Prompt와 출력 계약", h1="Prompt와 출력 계약: 모델의 답을 믿을 수 있게 만들기", track="백엔드",
        crumb="백엔드 &gt; LLM 활용",
        one_line="모델의 답은 언제나 그냥 글이에요. 그래서 <b>Prompt와 근거 자료는 정해진 틀로</b> 만들고, 받은 답은 <b>모양 검사</b>와 <b>내용 검사</b>를 거친 것만 써요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm/answer_check.py", practice=practice, libs=libs,
        libs_note="<code>build_context</code>, <code>parse_answer</code>, <code>check_rules</code>, <code>validate</code>, <code>ask</code>, <code>run_with_retry</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"llm-client.html\">LLM API 클라이언트</a>", "실제 모델을 부르고, 오류와 비용을 다루기"), ("<a href=\"indexing.html\">인덱싱</a>", "①에서 넣을 자료를 검색으로 고르기")],
        sources=["다루는 범위: Day21 (2026-11-16) Prompt, Context Engineering", "예제: <code>examples/llm/answer_check.py</code>, <code>examples/llm/samples.json</code> (기존 노트의 자료 101~103과 응답 유형)",
                 "결과 값: <code>tools/pages/prompt_contract.py</code>가 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://docs.python.org/3/library/string.html#template-strings\" target=\"_blank\">string.Template</a>, <a href=\"https://docs.pydantic.dev/latest/concepts/strict_mode/\" target=\"_blank\">Pydantic strict mode</a>, <a href=\"https://platform.openai.com/docs/guides/prompt-engineering\" target=\"_blank\">OpenAI Prompt engineering</a>"])
print("ok")
