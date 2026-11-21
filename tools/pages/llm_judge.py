"""주제 페이지 '백엔드 > LLM 활용 > LLM으로 답 채점하기(LLM-as-a-Judge)'를 만든다.

examples/llm/judge.py 를 단계별로 실제 실행한다. Anthropic 서버에는 요청을 보내지 않고(키, 비용 없음),
anthropic SDK 에 httpx.MockTransport(가짜 서버)를 붙여 SDK 가 실제로 만드는 요청 본문과, 응답 처리 코드의 동작을 관찰한다.
가짜 서버가 돌려주는 판정, 토큰 수는 코드 경로를 보이려고 넣은 값이라 실제 모델의 판단이 아니다.
실행: python tools/pages/llm_judge.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import anthropic  # noqa: E402
import httpx  # noqa: E402
import pydantic  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "llm"
code = S.parse((EX / "judge.py").read_text(encoding="utf-8"))
ENV = f"Python {sys.version.split()[0]}, anthropic {anthropic.__version__}, pydantic {pydantic.__version__}, httpx {httpx.__version__} (가짜 서버)"
E = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
outs, sent = {}, []

# 가짜 서버의 판정 (코드 경로를 보이기 위한 고정 값)
FAKE = {"교차검증은": {"rationale": "흔들림을 줄이려 여러 번 재고 평균을 본다는 핵심이 같아요. 흔들림(표준편차)도 함께 본다는 점은 빠졌어요.", "error_type": "빠짐", "score": 4},
        "AdaBoost가": {"rationale": "기준 답은 지수 손실과 무게 쏠림이 이유인데, 답은 트리 수와 과적합을 말해요. 핵심이 틀렸어요.", "error_type": "틀림", "score": 1},
        "프롬프트 캐시는": {"rationale": "기준 답은 입력 앞부분을 싸게 읽는 것인데, 답은 출력을 저장해 다시 보낸다고 해서 틀렸어요. 길이와 말투는 점수에 넣지 않았어요.", "error_type": "틀림", "score": 1}}
calls = {}


def message(text, stop="end_turn", n_in=600, n_out=180):
    return {"id": f"msg_{len(sent):03d}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "thinking", "thinking": "", "signature": "sig-demo"}, {"type": "text", "text": text}],
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": n_in, "output_tokens": n_out, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}}


def handler(req):
    b = json.loads(req.content)
    sent.append({"url": str(req.url), "headers": {k: v for k, v in req.headers.items() if k in ("x-api-key", "anthropic-version", "content-type")}, "body": b})
    sysm = b["system"][0]["text"]
    user = b["messages"][0]["content"]
    if sysm.startswith("두 답 중"):                                  # 순서 버릇을 심은 가짜 Judge: 언제나 A
        return httpx.Response(200, json=message(json.dumps({"rationale": "A가 더 가까워요.", "winner": "A"}, ensure_ascii=False)))
    key = next(k for k in FAKE if f"[질문]\n{k}" in user)
    calls[key] = calls.get(key, 0) + 1
    full = json.dumps(FAKE[key], ensure_ascii=False)
    if key == "AdaBoost가" and b["max_tokens"] == 2048:   # ④에서 한 번은 생각하다 한도에 닿게
        return httpx.Response(200, json=message(full[:40], stop="max_tokens", n_out=2048))
    return httpx.Response(200, json=message(full))


FAKE_HTTP = httpx.Client(transport=httpx.MockTransport(handler))
PATCH = [("anthropic.Anthropic(max_retries=2)", "anthropic.Anthropic(max_retries=2, api_key=\"sk-ant-test-not-a-real-key\", http_client=FAKE_HTTP)")]
run = lambda k: S.run(code, k, S.workdir(EX), PATCH if k >= 2 else None, filename="judge.py", env={"FAKE_HTTP": FAKE_HTTP})  # noqa: E731

# ---- ① ----
ns = run(1)
sch = ns["schema"]
props = sch["properties"]
S.claim(sch.get("additionalProperties") is False and list(props) == ["rationale", "error_type", "score"], "① 계약에 없는 필드 금지, 근거가 점수보다 먼저")
S.claim(props["score"]["enum"] == [1, 2, 3, 4, 5], "① 점수는 1~5 중 하나")
outs["s1"] = S.box("Pydantic 클래스가 만든 JSON Schema (줄임)",
                   [(f"properties 순서 → {', '.join(props)}", 1), (f"error_type → enum {props['error_type']['enum']}", 2),
                    (f"score → enum {props['score']['enum']}", 3), (f"additionalProperties → {sch['additionalProperties']}", 4), f"required → {sch['required']}"],
                   ["필드 순서가 곧 모델이 쓰는 순서예요. 근거(rationale)를 먼저 쓰게 해야, 점수를 먼저 정하고 근거를 끼워 맞추는 일이 줄어요.",
                    "점수와 오류 종류는 정해진 값 중에서만 고르게 했어요(<code>enum</code>). \"4점\", \"4.5\" 같은 답이 아예 나올 수 없어요.",
                    "<code>extra=\"forbid\"</code>가 <code>additionalProperties: false</code>가 돼요. 모델이 <code>confidence</code> 같은 필드를 덧붙이지 못해요."])

# ---- ② ----
sent.clear()
calls.clear()
ns = run(2)
s0 = sent[0]
bdy = s0["body"]
same_sys = len({json.dumps(ns["judge_request"](it)["system"], ensure_ascii=False) for it in ns["ITEMS"]}) == 1
diff_msg = len({json.dumps(ns["judge_request"](it)["messages"], ensure_ascii=False) for it in ns["ITEMS"]}) == len(ns["ITEMS"])
blocks = [b.type for b in ns["first"].content]
S.claim(same_sys and diff_msg, "② 세 문항의 system은 글자 하나까지 같고, messages만 다르다")
S.claim("temperature" not in bdy and bdy["output_config"]["effort"] == "low" and blocks == ["thinking", "text"], "② temperature 없음, effort low, 응답은 생각 블록 다음 글 블록")
body_view = {**bdy, "system": [{**bdy["system"][0], "text": bdy["system"][0]["text"][:28] + "…"}],
             "messages": [{"role": "user", "content": bdy["messages"][0]["content"][:24] + "…"}],
             "output_config": {"effort": bdy["output_config"]["effort"], "format": {"type": "json_schema", "schema": "①의 JSON Schema"}}}
outs["s2"] = S.box("SDK가 실제로 보낸 요청 (가짜 서버에서 본 것)",
                   [(f"POST {s0['url']}", 1), (f"x-api-key: {s0['headers']['x-api-key'][:10]}…, anthropic-version: {s0['headers']['anthropic-version']}", 2),
                    (f"본문 → {json.dumps(body_view, ensure_ascii=False)}", 3), "",
                    (f"세 문항의 system이 모두 같은가 → {same_sys}, messages는 모두 다른가 → {diff_msg}", 4, "newf"),
                    (f"응답 content 블록 → {blocks}", 5), (f"Verdict로 읽은 결과 → score {ns['verdict'].score}, error_type {ns['verdict'].error_type}", 6)],
                   ["키는 본문이 아니라 머리글(<code>x-api-key</code>)로 가요. 어떤 API 판으로 말하는지는 <code>anthropic-version</code>으로 정해요.",
                    "채점 기준은 문항마다 같아서 <code>system</code>에 두고 <code>cache_control</code>을 붙였어요. 바뀌는 질문과 답은 그 뒤 <code>messages</code>에 와요. 캐시는 앞에서부터 글자 하나까지 같은 부분만 다시 써요.",
                    "Claude Opus 5.5는 생각(thinking)이 늘 켜져 있어서, 응답 맨 앞에 생각 블록이 와요(내용은 기본으로 비어 있어요). 그래서 <code>content[0].text</code>가 아니라 글 블록만 골라 읽어요.",
                    "판정 내용은 가짜 서버가 돌려준 고정 값이에요. 이 단계에서 볼 것은 요청의 모양이에요."])

# ---- ③ ----
ns = run(3)
cs = ns["costs"]
S.claim(cs[1][1] > cs[1][0] and cs[2][1] < cs[2][0], "③ 1건이면 캐시가 손해, 2건부터 이득")
save1000 = 1 - cs[1000][1] / cs[1000][0]
in_only = lambda n, c: c - n * ns["OUT"] * ns["PRICE_OUT"] / 1e6  # noqa: E731
save_in = 1 - in_only(1000, cs[1000][1]) / in_only(1000, cs[1000][0])
outs["s3"] = S.table(f"③ 채점 기준 {ns['FIXED']:,}토큰 + 문항 {ns['VARYING']}토큰, 출력 {ns['OUT']}토큰일 때 (claude-opus-5-5 단가, 가정한 토큰 수)",
                     ["문항 수", "캐시 없이", "캐시 쓰고", "줄어든 비율"],
                     [[f"{n:,}", f"${a:,.4f}", f"${b:,.4f}", f"{1 - b / a:.0%}" if b < a else f"오히려 {b / a - 1:.0%} 더 비쌈"] for n, (a, b) in cs.items()],
                     [f"1건뿐이면 캐시 쓰기 값(입력 단가의 {ns['WRITE_5M']}배)만 더 내서 손해예요. 같은 기준으로 2건만 채점해도 이득이에요.",
                      f"1,000건이면 전체 비용이 {save1000:.0%} 줄어요. 입력 비용만 보면 {save_in:.0%} 줄지만, 출력(생각 포함) 비용은 캐시와 상관없어서 그대로예요.",
                      "토큰 수는 가정한 값이에요. 실제 수는 <code>client.messages.count_tokens(...)</code>로 세거나, 응답의 <code>usage</code>로 확인해요."])

# ---- ④ ----
sent.clear()
calls.clear()
ns = run(4)
res = ns["results"]
rows4 = []
for it, (v, used, tries) in zip(ns["ITEMS"], res):
    tu = ns["total_usage"](used)
    rows4.append([E(it["question"]), " → ".join(tries), v.score if v else "없음", v.error_type if v else "-", f"{tu['input_tokens']:,} / {tu['output_tokens']:,}"])
S.claim(res[1][2] == ["max_tokens", "end_turn"] and res[1][0] is not None and all(r[2] == ["end_turn"] for i, r in enumerate(res) if i != 1), "④ 두 번째 문항만 한 번 잘렸다가 더 큰 한도로 성공")
outs["s4"] = S.table("④ 세 문항을 채점한 기록 (가짜 서버)", ["질문", "stop_reason", "score", "error_type", "입력 / 출력 토큰 (합)"], rows4,
                     ["두 번째 문항은 가짜 서버가 일부러 <code>max_tokens</code>로 끊었어요. 생각에 토큰을 쓰다 한도에 닿으면, 글 블록의 JSON이 중간에 잘려요. 그대로 읽으면 Pydantic 오류가 나니까, <code>stop_reason</code>을 먼저 보고 한도를 키워 한 번만 다시 물었어요.",
                      "두 번 부른 토큰은 둘 다 돈이 나가요. 그래서 성공한 호출만이 아니라 모든 호출의 <code>usage</code>를 더해요. 캐시 쓰기와 읽기는 단가가 달라서 칸을 따로 더해요."])

# ---- ⑤ ----
sent.clear()
ns = run(5)
one, two, same = ns["order_check"]
S.claim(not same and one == "A" and two == "B", "⑤ 늘 A를 고르는 가짜 Judge를 순서 바꿔 묻기가 잡아낸다")
S.claim(0.8 < ns["kappa"] < 1 and ns["exact"] == 0.5 and ns["within1"] == 1.0 and ns["lenient"] > 0, "⑤ 일치 0.5, 1점 안 1.0, 가중 kappa 0.8~1, Judge가 후하다")
outs["s5"] = S.box("⑤ 순서를 바꿔 두 번 묻기 (가짜 Judge는 늘 A를 고르게 해 둠)",
                   [(f"짧고 맞는 답을 A에 → {one}", 1), (f"짧고 맞는 답을 B에 → 첫 번째 기준으로 되돌리면 {two}", 2), (f"두 번의 판정이 같은가 → {same}", 3, "newf")],
                   ["판정이 순서에 따라 뒤집히면 그 비교는 버리거나 \"같음\"으로 둬요. 실제 Judge는 이렇게 극단적이진 않지만, 첫 번째나 두 번째 자리를 조금씩 더 좋아하는 버릇이 알려져 있어요."])
outs["s5k"] = S.table("가상의 10건으로 사람 점수와 Judge 점수 맞춰 보기", ["지표", "값", "뜻"],
                      [["완전 일치", f"{ns['exact']:.0%}", "점수가 똑같은 비율"], ["1점 안", f"{ns['within1']:.0%}", "차이가 1점 이하인 비율"],
                       ["가중 kappa", f"{ns['kappa']:.3f}", "우연히 맞을 몫을 빼고, 크게 어긋날수록 더 깎은 일치도. 1이면 완전 일치, 0이면 우연 수준"],
                       ["평균 차이 (Judge − 사람)", f"{ns['lenient']:+.1f}점", "Judge가 평균적으로 후한지(+) 박한지(−)"]],
                      ["점수는 사람이 미리 적어 둔 가상의 값이에요. 숫자로 무엇을 보는지 익히려는 표예요.",
                       f"완전 일치는 {ns['exact']:.0%}뿐이지만 1점 안에서는 모두 맞고, 가중 kappa가 {ns['kappa']:.2f}로 높아요. 대신 Judge가 평균 {ns['lenient']:.1f}점 후해요. 순위(어느 답이 더 나은가)는 믿어도 되지만, \"평균 4점이면 합격\" 같은 절대 기준은 사람 점수로 다시 맞춰야 한다는 뜻이에요."])

flow = [[1, "채점 기준", "루브릭 + 점수 계약", "근거 먼저, enum"], [2, "요청", "고정은 앞, 바뀌는 건 뒤", "cache_control, effort"],
        [3, "비용", "캐시 쓰기 vs 읽기", "2건부터 이득"], [4, "응답 읽기", "잘림, 거절, 사용량", "stop_reason 먼저"], [5, "믿을 수 있나", "순서 바꿔 묻기", "사람 점수와 kappa"]]
S.save(ROOT / "topics" / "backend" / "llm-judge-data.js", {"code": code, "names": S.names(code), "range": [1, 5], "outs": outs, "flow": flow, "env": ENV})

why = """<p>LLM이 만든 답이 좋은지 어떻게 알까요? 사람이 다 읽으면 정확하지만 느리고 비싸요. <a href=\"../model/gpt-generation.html\">작은 GPT와 디코딩</a> ⑧의 ROUGE처럼 기준 답과 겹치는 단어를 세면 빠르지만, 뜻은 같은데 말이 다른 답을 깎아요. 그 사이에 있는 방법이 <b>LLM을 채점자로 쓰는 것</b>(LLM-as-a-Judge)이에요.</p>
<p>채점자도 LLM이라서 <a href="prompt-contract.html">Prompt와 출력 계약</a>, <a href="llm-client.html">LLM API 클라이언트</a>에서 본 것을 그대로 써요. 이 쪽에서는 거기에 채점만의 문제를 더해요.</p>"""
why += K.table(["문제", "이 쪽"], [["채점 기준을 어떻게 적고, 점수를 어떤 모양으로 받나", "①"], ["같은 기준으로 수천 건을 채점할 때 비용", "②③ 프롬프트 캐시"],
                                    ["생각하다 잘린 답, 거절", "④"], ["채점자의 버릇(순서, 길이, 후함)과 사람 점수와의 일치", "⑤"]])
whole = """<p>채점 기준을 점수 계약으로 → 고정 부분을 캐시하는 요청 → 비용 계산 → 응답 읽기 → 채점을 믿을 수 있는지 확인. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>예제는 Anthropic Python SDK로 Claude Opus 5.5를 불러요. 이 노트의 값은 <b>가짜 서버</b>를 SDK에 붙여 실제로 실행한 결과예요. SDK가 만드는 요청과 응답 처리 코드는 진짜지만, 판정 내용과 토큰 수는 가짜 서버가 돌려준 고정 값이라 모델의 실제 판단이 아니에요.</p>"""
steps = [
    K.step(1, "채점 기준과 점수 계약", "무엇을 몇 점으로 볼지 글로 적고(루브릭), 받을 답의 모양을 Pydantic 클래스로 정해요.",
           "<p><b>루브릭</b>은 점수마다 무엇을 뜻하는지 적은 채점 기준표예요. \"좋으면 5점\"처럼 막연하면, 같은 답도 부를 때마다 점수가 달라져요. 점수마다 구체적인 조건을 적고, 기준 답을 함께 줘서 \"무엇과 비교할지\"를 정해요.</p>"
           "<p>채점 방식은 크게 둘이에요. 답 하나에 점수를 매기는 <b>점수 매기기</b>(이 쪽 ①~④)와, 두 답 중 나은 쪽을 고르는 <b>둘 중 고르기</b>(⑤)예요. 둘 중 고르기가 더 안정적이지만, 답이 n개면 비교가 n²에 가깝게 늘어요.</p>"
           "<p>받는 모양은 <a href=\"prompt-contract.html\">Prompt와 출력 계약</a> ③처럼 Pydantic으로 적어요. 채점에서는 두 가지를 더 신경 써요.</p>"
           "<ul><li><b>근거를 점수보다 먼저</b>: 모델은 앞에서부터 차례로 써요. 점수를 먼저 쓰면 그 뒤의 근거가 점수에 맞춰져요.</li>"
           "<li><b>고를 수 있는 값을 정해 두기</b>: 점수와 오류 종류를 정해진 값 중에서만 고르게 하면, 모아서 세기 쉬워요.</li></ul>"
           + K.code_table([("class Verdict(BaseModel)", "받을 판정의 모양"), ("Literal[1, 2, 3, 4, 5]", "점수는 이 다섯 값 중 하나"), ("ConfigDict(extra=\"forbid\")", "계약에 없는 필드는 거절"),
                           ("Verdict.model_json_schema()", "클래스를 API에 줄 JSON Schema로 바꾸기")]), "s1"),
    K.step(2, "요청 만들기: 고정 부분은 앞에, 바뀌는 부분은 뒤에", "채점 기준은 system에 두고 캐시 표시를 붙이고, 문항은 messages에 넣어요. 답의 모양과 생각 깊이는 output_config로 정해요.",
           "<p><b>프롬프트 캐시</b>: 요청의 <b>앞부분</b>이 이전 요청과 글자 하나까지 같으면, 서버가 그 부분을 다시 계산하지 않고 저장해 둔 결과를 읽어요. 읽은 토큰은 훨씬 싸고, 첫 글자가 나오기까지의 시간도 줄어요. 요청은 <code>tools</code> → <code>system</code> → <code>messages</code> 순서로 이어 붙여지니까, 안 바뀌는 것을 앞에 두고 <code>cache_control</code>로 \"여기까지 저장\"을 표시해요.</p>"
           "<p>⚠️ 캐시가 깨지는 흔한 이유: 채점 기준에 오늘 날짜나 요청 번호를 넣기, 예시 순서를 매번 섞기, 키 순서가 바뀌는 JSON. 그리고 앞부분이 너무 짧으면(Claude Opus 5.5는 512토큰 미만) 표시를 해도 저장되지 않아요. 오류도 나지 않으니 응답의 <code>usage.cache_read_input_tokens</code>가 0인지로 확인해요.</p>"
           "<p><b>output_config</b>에는 두 가지를 넣어요. <code>format</code>은 ①의 JSON Schema로 답의 모양을 강제하고, <code>effort</code>는 생각을 얼마나 깊이 할지예요(<code>low</code>~<code>max</code>). 채점은 기준이 분명하면 <code>low</code>로도 충분한 경우가 많아서, 몇십 건으로 사람 점수와 비교해 보고 정해요.</p>"
           "<p>예전 예제는 같은 답이 나오게 <code>temperature=0</code>을 주곤 했는데, Claude Opus 5.5 같은 최신 Claude 모델은 <code>temperature</code>를 받지 않아요(보내면 오류). 판정이 흔들리는지는 같은 문항을 몇 번 다시 채점해서 직접 재요.</p>"
           + K.code_table([("anthropic.Anthropic(max_retries=2)", "키는 환경 변수 <code>ANTHROPIC_API_KEY</code>에서. 실패하면 2번까지 다시"),
                           ("system=[{..., \"cache_control\": {\"type\": \"ephemeral\"}}]", "여기까지를 5분 동안 저장. <code>\"ttl\": \"1h\"</code>면 1시간"),
                           ("output_config={\"effort\": ..., \"format\": {\"type\": \"json_schema\", ...}}", "생각 깊이와 답의 모양"),
                           ("b.type == \"text\"", "생각(thinking) 블록은 건너뛰고 글 블록만 읽기")]), "s2"),
    K.step(3, "캐시로 비용 줄이기: 몇 건부터 이득인가", "캐시에 저장할 때는 조금 더 내고, 읽을 때는 훨씬 적게 내요. 문항 수에 따라 전체 비용을 계산해요.",
           "<p>캐시 값은 세 가지예요. 저장하지 않은 입력은 보통 단가, 처음 저장할 때는 입력 단가의 1.25배(5분) 또는 2배(1시간), 저장된 것을 읽을 때는 훨씬 싸요(Claude Opus 5.5는 0.05배). 출력 토큰은 캐시와 상관없어요.</p>"
           "<p>읽을 때마다 저장 시간이 다시 5분으로 늘어나요. 그래서 같은 기준으로 5분 안쪽 간격으로 계속 채점하면 5분짜리로 충분하고, 몇십 분씩 쉬었다 하면 1시간짜리가 나아요.</p>"
           + K.code_table([("fixed * WRITE_5M + (n - 1) * fixed * READ", "첫 건은 저장 값, 나머지 n − 1건은 읽기 값"), ("n * varying", "바뀌는 문항 부분은 매번 보통 단가")]), "s3"),
    K.step(4, "응답 읽기: 잘린 답, 거절, 쓴 토큰 모으기", "stop_reason을 먼저 보고, 잘렸으면 한도를 키워 한 번만 다시 묻고, 모든 호출의 사용량을 더해요.",
           "<p>답의 모양이 맞는지 검사하고 틀리면 한 번 다시 묻는 건 <a href=\"prompt-contract.html\">Prompt와 출력 계약</a> ④⑥에서 봤어요. <code>format</code>으로 모양을 강제하면 그런 일은 드물어요. 대신 채점에서는 다른 두 경우를 먼저 봐요.</p>"
           "<ul><li><code>stop_reason == \"max_tokens\"</code>: 생각에 토큰을 쓰다 한도에 닿아 JSON이 중간에 끊겼어요. 한도를 키워 다시 물어요.</li>"
           "<li><code>stop_reason == \"refusal\"</code>: 안전 판단으로 답하지 않았어요. 다시 물어도 같기 쉬워서, 점수 없음으로 남기고 사람이 봐요.</li></ul>"
           + K.code_table([("for max_tokens in (2048, 8192)", "처음엔 2,048, 잘리면 8,192로 한 번만 더"), ("r.stop_reason", "왜 멈췄나: end_turn(다 씀), max_tokens(한도), refusal(거절)"),
                           ("total_usage(used)", "모든 호출의 입력, 캐시 쓰기, 캐시 읽기, 출력 토큰을 따로 더하기")]), "s4"),
    K.step(5, "채점을 믿을 수 있나: 순서 바꿔 묻기, 사람 점수와 맞춰 보기", "두 답의 자리를 바꿔 한 번 더 물어 판정이 뒤집히는지 보고, 사람이 매긴 점수와 얼마나 맞는지 숫자로 재요.",
           "<p>LLM 채점자에게는 알려진 버릇이 있어요.</p>"
           + K.table(["버릇", "모습", "줄이는 방법"], [["자리 버릇", "같은 두 답도 먼저(또는 나중에) 나온 쪽을 더 고름", "순서를 바꿔 두 번 묻고, 뒤집히면 \"같음\""],
                                                    ["길이 버릇", "길고 자신 있게 쓴 답을 더 좋게 봄", "루브릭에 \"길이는 보지 마\"를 넣고, 기준 답과 비교하게"],
                                                    ["자기 편", "자기와 같은 모델이 쓴 답을 더 좋게 봄", "채점 모델을 답한 모델과 다르게, 또는 여러 모델의 판정을 모으기"],
                                                    ["후함", "점수가 전체로 높게 몰림", "사람 점수 몇십 건과 맞춰 평균 차이를 보고 기준을 다시 잡기"]])
           + K.out("s5")
           + "<p><b>사람 점수와 맞춰 보기</b>: 채점자를 쓰기 전에 몇십 건을 사람이 먼저 채점하고, 같은 문항을 채점자에게 매겨 비교해요. 완전 일치만 보면 4점과 5점의 작은 차이도 틀린 걸로 쳐서 너무 박하고, 그냥 상관계수를 보면 \"늘 1점씩 후함\"을 못 잡아요. 그래서 <b>가중 kappa</b>(우연히 맞을 몫을 빼고, 크게 어긋날수록 더 깎은 일치도)와 평균 차이를 함께 봐요.</p>"
           + K.out("s5k")
           + "<p>사람끼리도 점수가 다 같지는 않아요. 사람 두 명의 kappa를 먼저 재서, 채점자가 \"사람끼리 맞는 만큼\" 맞으면 쓸 만하다고 봐요. 기준 답이 하나뿐이면 다르게 말한 좋은 답이 깎이니까, 기준 답을 여러 개 주기도 해요. 그리고 채점자를 쓴 뒤에도 일부는 계속 사람이 다시 봐요(사람이 고리 안에 남기).</p>"
           + K.code_table([("pairwise(question, reference, x, y)", "x를 A 자리에 한 번, B 자리에 한 번 두고 묻기"), ("{\"A\": \"B\", \"B\": \"A\", ...}[...]", "두 번째 판정을 첫 번째 자리 기준으로 되돌리기"),
                           ("cohen_kappa_score(human, judged, weights=\"quadratic\")", "차이의 제곱만큼 깎는 가중 kappa (scikit-learn)")]), "s5"),
]
practice = [("채점 기준이 길다", "예시 답과 점수를 몇 개 넣으면 좋아지지만 길어져요. 고정 부분이라 캐시되니, 길이보다 기준이 바뀌지 않게 하는 게 중요해요."),
            ("수천 건을 한꺼번에 채점", "급하지 않으면 Message Batches로 보내면 값이 절반이에요. 응답은 순서가 섞여 오니 <code>custom_id</code>로 맞춰요."),
            ("채점 모델 고르기", "사람 점수 몇십 건으로 모델과 effort를 바꿔 가며 kappa를 재서, 맞는 것 중 가장 싼 조합을 골라요."),
            ("점수가 문항마다 흔들린다", "같은 문항을 3번 채점해 다르게 나온 비율을 재요. 많으면 루브릭의 점수 경계를 더 구체적으로 적어요."),
            ("기준 답이 없다", "두 답 중 고르기로 바꾸거나, 루브릭에 꼭 들어가야 할 항목을 하나씩 적고 \"있다 / 없다\"로 체크하게 해요.")]
libs = [("①⑤", "<code>pydantic.BaseModel</code>, <code>Literal</code>, <code>ConfigDict(extra=\"forbid\")</code>, <code>model_json_schema()</code>", "점수 계약"),
        ("②④⑤", "<code>anthropic.Anthropic</code>, <code>client.messages.create</code>, <code>output_config</code>, <code>cache_control</code>, <code>stop_reason</code>, <code>usage</code>", "Anthropic Python SDK"),
        ("⑤", "<code>sklearn.metrics.cohen_kappa_score</code>", "가중 kappa")]
K.build(ROOT / "topics" / "backend" / "llm-judge.html", title="LLM으로 답 채점하기", h1="LLM으로 답 채점하기: LLM-as-a-Judge", track="백엔드",
        crumb="백엔드 &gt; LLM 활용",
        one_line="채점 기준(루브릭)과 점수 모양을 정하고, 고정된 기준은 <b>캐시</b>해서 수천 건을 싸게 채점해요. 채점자도 버릇이 있어서, <b>순서를 바꿔 묻고</b> <b>사람 점수와 맞춰 봐야</b> 믿을 수 있어요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm/judge.py", practice=practice, libs=libs,
        libs_note="<code>judge_request</code>, <code>text_of</code>, <code>batch_cost</code>, <code>judge</code>, <code>total_usage</code>, <code>pick</code>, <code>pairwise</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"prompt-contract.html\">Prompt와 출력 계약</a>", "모양 검사와 한 번 다시 묻기"), ("<a href=\"llm-client.html\">LLM API 클라이언트</a>", "재시도, 스트리밍, 비용 기록"),
                   ("<a href=\"../model/llm-architectures.html\">요즘 LLM의 부품</a>", "서버 쪽 KV 캐시와 prefix caching")],
        sources=["다루는 범위: 부트캠프 LLM 활용(Day21, 22)을 평가 쪽으로 넓힌 쪽이에요",
                 "함께 정리한 자료: 개인 위키의 LLM-as-a-Judge 정리(구조화 출력, 채점 기준 캐시, effort, 형식이 깨졌을 때 복구, 평가와 사람 검토). 예제 코드는 이 노트용으로 새로 썼어요",
                 "예제: <code>examples/llm/judge.py</code>. 결과 값: <code>tools/pages/llm_judge.py</code>가 가짜 서버를 SDK에 붙여 단계별로 실제 실행한 결과. 판정 내용과 토큰 수는 가짜 서버의 고정 값",
                 "공식 문서: <a href=\"https://platform.claude.com/docs/en/build-with-claude/structured-outputs\" target=\"_blank\">Structured outputs</a>, <a href=\"https://platform.claude.com/docs/en/build-with-claude/prompt-caching\" target=\"_blank\">Prompt caching</a>, <a href=\"https://platform.claude.com/docs/en/build-with-claude/effort\" target=\"_blank\">Effort</a>, <a href=\"https://github.com/anthropics/anthropic-sdk-python\" target=\"_blank\">anthropic-sdk-python</a>"])
print("ok")
