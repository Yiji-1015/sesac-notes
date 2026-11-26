"""백엔드 > 검색과 RAG > RAG 서비스. examples/rag/rag_service.py 하나가 ①~⑦로 자란다 (다른 쪽 코드에 기대지 않음).
검색은 e5-small(로컬 캐시)로 가람 도서관 조각 12개(../ingest/out/chunks.json)에서 하고, 답을 쓰는 모델 자리는 미리 정한 답을 돌려주는 함수다.
OpenAI에는 요청을 보내지 않는다(키, 비용 없음). 답 문장은 미리 정한 것이고, 검색, 근거 고르기, 검사, 서비스 결과는 실제 실행 값이다.
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/rag_service.py
"""
import html
import sys
import warnings
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "rag"
code = S.parse((EX / "rag_service.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, EX.resolve(), filename="rag_service.py")
V = {p: version(p) for p in ("langchain-core", "pydantic", "sentence-transformers")}
ENV = f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {v}" for p, v in V.items()) + ", intfloat/multilingual-e5-small"
E, J = html.escape, K.josa
outs = {}
C = ns["CHUNKS"]


def first_line(text, n=40):
    line = next((x for x in text.splitlines() if x.strip()), "")
    return line[:n] + ("…" if len(line) > n else "")


def name(i):
    c = C[i - 1]
    return c["heading"] or first_line(c["body"], 12)


# ---------------- ①
hits = ns["hits"]
S.claim(hits[0]["heading"] == "대출과 반납", "① 1위가 대출과 반납")
n_use = sum(1 for c in C if c["category"] == "이용")
outs["s1"] = S.table("① search(\"대출 기간은 며칠인가요?\", \"이용\", k=3)", ["id", "원문 위치", "제목", "유사도", "본문 첫 줄"],
                     [[h["id"], f"{h['file']}#{E(h['where'])}", E(h["heading"] or "(없음)"), f"{h['score']:.3f}", E(first_line(h["body"]))] for h in hits],
                     [f"분류가 '이용'인 조각 {n_use}개 중에서 가까운 순서로 {len(hits)}개예요. 1위가 질문에 맞는 '대출과 반납'이에요.",
                      "결과 한 건에는 본문과 함께 <b>어느 파일의 어디</b>에서 왔는지가 같이 있어요. 나중에 출처를 보여 주려면 처음부터 같이 다녀야 해요. id는 답에서 출처를 가리킬 짧은 번호예요."])

# ---------------- ②
hand = [d.metadata["id"] for d in ns["by_hand"]]
chain = [d.metadata["id"] for d in ns["documents"]]
d0 = ns["documents"][0]
S.claim(hand == chain == [h["id"] for h in hits], "② 직접 부르기와 | 로 이은 결과가 같다")
outs["s2"] = S.box("② 같은 일을 두 방식으로", [(f"함수를 차례로 직접 → id {hand}", 1), (f"RunnableLambda | RunnableLambda 의 invoke(req) → id {chain}", 2, "newf"),
                                          (f"documents[0] → {type(d0).__name__}(page_content=\"{first_line(d0.page_content, 24)}\", metadata 키 {list(d0.metadata)})", 3, "newf")],
                   ["보통의 파이썬이에요. 앞 함수의 결과를 받아 다음 함수에 넣어요.",
                    "<code>|</code>로 이으면 앞 단계의 출력이 다음 단계의 입력으로 자동으로 넘어가요. 결과가 똑같아요. 하는 일이 달라지는 게 아니라, 모든 단계를 같은 방식(invoke)으로 부르고 이을 수 있게 되는 거예요.",
                    "본문은 page_content에, 출처와 점수는 metadata에 담아요. LangChain의 검색 단계들은 이 Document 모양으로 주고받아요."])

# ---------------- ③
ctx, pid, bud = ns["context"], ns["provided_ids"], ns["budget"]
S.claim(bud[(2, 600)] == [1, 4] and len(bud[(3, 600)]) == 3 and len(bud[(3, 250)]) < 3 and bud[(2, 100)] == [], "③ 예산과 개수에 따라 2개, 3개, 3개 미만, 0개")
S.claim(len(set(bud[(3, 600)])) == len(bud[(3, 600)]), "③ 같은 조각을 두 번 넣지 않는다")
outs["s3"] = S.box(f"③ build_context(documents, max_items=2, max_chars=600) → {len(ctx)}자, 넣은 id {sorted(pid)}", ctx.split("\n"))
outs["t3"] = S.table("③ 후보 3개(맨 앞 후보를 한 번 더 섞은 4개)에 개수와 글자 예산을 바꾸면", ["넣을 수", "글자 예산", "실제로 넣은 id"],
                     [[k, c, str(v) if v else "<b>없음 (빈 자료)</b>"] for (k, c), v in bud.items()],
                     ["맨 앞 후보가 두 번 들어와도 id로 한 번만 넣어요.",
                      "글자 예산 250이면 세 번째 조각이 들어가면 넘쳐서 통째로 빼요. 조각을 중간에서 자르면 '14일이며, 한 번 연장하면'처럼 조건이 잘려 뜻이 바뀔 수 있거든요.",
                      "예산 100이면 어떤 조각도 통째로 들어가지 않아 자료가 비어요. 이건 '근거 부족'이지 '검색 실패'가 아니에요(⑦에서 둘을 다르게 처리해요)."])

# ---------------- ④
msgs = ns["messages"]
S.claim([m.type for m in msgs] == ["system", "human"], "④ system 하나, human 하나")
outs["s4"] = S.box("④ prompt.invoke({question, context}) → 메시지 (모델은 아직 안 불러요)", [(f"{m.type} ({len(m.content)}자): {m.content.splitlines()[0][:60]}…", i + 1) for i, m in enumerate(msgs)],
                   ["바뀌지 않는 규칙은 system 메시지에 둬요. 질문과 자료가 매번 바뀌어도 규칙은 그대로예요.",
                    "질문과 자료는 human 메시지에, '질문:'과 '자료:'로 경계를 나눠 넣어요. 자료 안에 '앞의 규칙은 무시하라' 같은 문장이 있어도 그건 자료일 뿐, 따를 지시가 아니라고 규칙에 적어 뒀어요."])

# ---------------- ⑤
msg = ns["message"]
S.claim(type(msg).__name__ == "AIMessage" and '"status": "answered"' in msg.content, "⑤ AIMessage로 JSON 답")
outs["s5"] = S.box("⑤ generation.invoke({question, context}) → 모델의 답", [(f"돌려받은 것 → {type(msg).__name__}", 1), (f"message.content → {msg.content}", 2, "newf")],
                   ["LangChain의 Chat Model은 답을 AIMessage로 돌려줘요. 가짜 함수도 같은 모양으로 돌려주게 해서, 나중에 진짜 모델로 바꿔 끼워도 뒤 단계는 그대로예요.",
                    "규칙대로 JSON 하나로 답했어요. 여기서는 가짜 함수가 미리 정한 문장이고, 출처 번호만 실제로 받은 자료에서 '대출 기간은 14일' 구절이 든 조각을 찾아 붙였어요. 이 글은 아직 검사하지 않았어요(⑥)."])

# ---------------- ⑥
oc = ns["outcome"]
expect = {"정상": ("통과", "answered"), "없는 출처 번호": ("막힘", "source_not_provided"), "출처는 맞는데 내용이 틀림": ("통과", "answered"),
          "status와 출처가 안 맞음": ("막힘", "answered_without_source"), "숫자 대신 글자 ID": ("막힘", "int_type"), "깨진 JSON": ("막힘", "json_invalid")}
S.claim(oc == expect, "⑥ 경우마다 결과가 글과 같다")
REASON = {"정상": "모든 검사 통과", "없는 출처 번호": "넣지 않은 번호 99", "출처는 맞는데 내용이 틀림": "번호는 넣은 것(자료는 14일, 답은 21일)", "status와 출처가 안 맞음": "answered인데 출처가 없음",
          "숫자 대신 글자 ID": "\"1\"은 strict에서 정수가 아님", "깨진 JSON": "JSON이 중간에 끝남"}
outs["t6"] = S.table("⑥ 같은 질문과 자료에 모델이 여섯 가지로 답하면", ["모델의 답", "무엇이 다른가", "check 결과"],
                     [[k, REASON[k], f"<b>{v[0]}</b>: <code>{v[1]}</code>"] for k, v in oc.items()],
                     ["'출처는 맞는데 내용이 틀림'이 <b>통과</b>해요. 코드가 보는 건 모양, 번호, status 조합뿐이에요. 출처를 맞게 댔는지(citation)와 답이 그 출처 내용과 맞는지(grounding)는 다른 검사예요. 뒤쪽은 숫자, 조건, 예외를 원문과 대조해야 해서 사람이 보거나 <a href=\"llm-judge.html\">LLM 채점</a>을 따로 붙여요.",
                      "모양 검사는 Pydantic이 해요. 깨진 JSON(json_invalid)과 글자 \"1\"(int_type)은 내용을 보기도 전에 막혀요.",
                      "어떤 실패도 '정보 부족' 답으로 바꾸지 않아요. 정보 부족은 정상적인 답이고, 실패는 실패예요(⑦)."])

# ---------------- ⑦
tr, res, fl = ns["trace"], ns["results"], ns["failures"]
S.claim([t["status"] for t in tr] == ["answered"] * 3 + ["insufficient_context"] * 3, "⑦ 셋은 답, 셋은 정보 부족")
S.claim([t["model_calls"] for t in tr] == [1, 1, 1, 1, 1, 0] and tr[5]["included"] == [], "⑦ 빈 자료만 모델을 안 부른다")
S.claim(all({s["id"] for s in r["sources"]} <= set(t["included"]) for r, t in zip(res, tr)), "⑦ 출처는 넣은 조각 안에서만")
S.claim(tr[3]["included"] and tr[3]["status"] == "insufficient_context", "⑦ 연체료는 관련 조각이 있어도 정보 부족")
S.claim(fl == {"검색 시스템 멈춤": "retrieval_error", "모델이 넣지 않은 출처를 댐": "generation_error"}, "⑦ 실패 둘은 오류로")
rows7 = []
for (q, cat), t, r in zip(ns["QUESTIONS"] + [("대출 기간은 며칠인가요? (글자 예산 100)", "이용")], tr, res):
    src = ", ".join(f"[{s['id']}] {s['file']}" for s in r["sources"]) or "없음"
    rows7.append([E(q), cat, str(t["candidates"]), str(t["included"]) if t["included"] else "없음", f"<b>{t['status']}</b>", src, t["model_calls"]])
outs["t7"] = S.table("⑦ service(req): 질문 여섯 개 (후보 3개, 넣을 수 2개)", ["질문", "분류", "찾은 후보", "넣은 조각", "결과", "돌려준 출처", "모델 호출"], rows7,
                     [f"연체료 질문은 '규정'에서 연체와 관련된 조각 {tr[3]['included']}를 넣었지만 금액이 없어서 정보 부족이에요. 관련 문서가 있다는 것과 답할 수 있다는 건 달라요.",
                      "글자 예산을 100으로 줄이면 넣을 조각이 없어서 모델을 부르지 않고 바로 정보 부족을 돌려줘요(호출 0).",
                      "돌려준 출처는 모델이 쓴 파일명이 아니라, 검사를 통과한 번호로 서버가 보관해 둔 원문 위치를 찾아 붙인 거예요."])
outs["t7b"] = S.table("⑦ 고장 났을 때", ["상황", "결과"], [[E(k), f"<b>RagError(\"{v}\")</b>"] for k, v in fl.items()],
                      ["검색이 멈추면 '찾은 게 없음'으로 바꾸지 않고 retrieval_error로 올려요. 바꾸면 장애가 '자료에 없음'처럼 보여요.",
                       "모델이 넣지 않은 출처를 대면 ⑥의 검사에서 막혀 generation_error예요. 검사를 못 넘긴 답은 내보내지 않아요.",
                       "API로 열 때는 정보 부족은 정상 응답(200), RagError는 뒤에서 일하는 쪽의 실패(502)로 나눠 돌려줘요."])

flow = [[1, "찾기", "질문 + 분류 → 조각", f"후보 {len(hits)}개"], [2, "LangChain", "Document, Runnable", "| 와 invoke"], [3, "근거 고르기", "개수, 글자 예산", f"넣은 {sorted(pid)}"],
        [4, "Prompt", "규칙 + 질문 + 자료", "system, human"], [5, "모델", "AIMessage", "가짜 함수"], [6, "검사", "모양, 출처, status", "citation ≠ grounding"],
        [7, "서비스", "찾기 → 답 → 출처", "정보 부족과 실패 구분"]]
S.save(ROOT / "topics" / "backend" / "rag-service-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>LLM은 우리 회사의 최신 규정이나 내부 문서를 모르고, 학습이 끝난 뒤 바뀐 내용도 몰라요. 회사 기준으로 답하게 하는 방법은 크게 셋이에요.</p>
{K.table(["방법", "하는 일", "맞는 때", "모델 가중치"], [["문서 통째로 넣기", "정해 둔 문서를 Prompt에 그대로 넣어 답하게", "문서가 적고 짧을 때", "그대로"],
                                               ["<b>RAG</b>", "질문마다 관련 조각을 <b>찾아서</b> 넣고, 그걸 근거로 답하게", "문서가 많고 자주 바뀌고, 출처가 필요할 때", "그대로"],
                                               ["미세조정", "예시로 더 학습시켜 말투, 형식, 작업 방식을 바꾸기", "형식과 행동을 바꿀 때. 새 사실을 넣는 데는 비싸고 출처를 못 대요", "바뀜"]])}
<p>RAG(Retrieval-Augmented Generation)는 <b>찾고(Retrieval)</b>, 모델 입력에 <b>보태고(Augmented)</b>, 그걸 근거로 <b>답하게(Generation)</b> 해요. 이 쪽은 질문이 들어와 조각을 찾고, 자료를 골라 모델에게 묻고, 답을 검사해 출처와 함께 돌려주기까지를 한 파일로 만들어요.</p>
<p><b>LangChain</b>은 이 연결을 돕는 라이브러리예요. 같은 규칙에 질문과 자료만 바꿔 끼우는 Prompt 틀, 회사마다 다른 모델 API를 같은 방식으로 부르는 Chat Model, 단계를 <code>|</code>로 잇고 <code>invoke</code>로 실행하는 Runnable을 줘요. 다만 무엇을 찾고, 무엇을 넣고, 답이 맞는지는 정해 주지 않아요.</p>
<p>예제의 검색 대상은 가상의 '가람 도서관' 안내문을 나눈 조각 {len(C)}개예요(<a href="indexing.html">인덱싱</a> 쪽에서 만든 것). 검색은 e5 모델로 실제로 하고, 답을 쓰는 모델 자리는 미리 정한 답을 돌려주는 함수예요. 그래서 키도 비용도 없이 돌아가고, 근거 고르기와 검사와 서비스의 동작은 실제 실행 값이에요.</p>
{K.table(["id", "파일", "분류", "제목", "본문 첫 줄"], [[c["id"], c["file"], c["category"], E(c["heading"] or "(없음)"), E(first_line(c["body"], 36))] for c in C])}
"""
whole = """<p>한 질문이 답이 되기까지 일곱 단계예요. ①~⑥에서 부품을 하나씩 만들고, ⑦에서 서비스 함수 하나로 묶어요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "질문과 가까운 조각 찾기", "질문과 분류로 가까운 조각을 k개 찾아요. 결과 한 건에는 본문과 함께, 나중에 출처를 다시 찾아갈 정보를 담아요.",
           "<p>조각마다 e5로 만든 벡터를 미리 두고, 질문이 오면 질문 벡터와 가까운 순서로 골라요(자세한 건 <a href=\"embedding.html\">Embedding과 Vector Search</a>). 분류(category)는 검색 범위예요. 같은 질문이라도 '이용' 안에서 찾는지 '공지' 안에서 찾는지에 따라 후보가 달라져요.</p>"
           + K.code_table([("encoder.encode([\"query: ...\"], normalize_embeddings=True)", "질문을 벡터로 (e5는 질문 앞에 query:, 조각 앞에 passage:)"), ("vectors @ q", "모든 조각과의 코사인 유사도 (정규화했으니 내적)"),
                           ("search(question, category, k)", "분류 안에서 가까운 k개를 dict 목록으로")]), "s1"),
    K.step(2, "LangChain으로 잇기: Document와 Runnable", "함수를 LangChain의 실행 단위(Runnable)로 감싸면, 모든 단계를 invoke 하나로 부르고 | 로 이을 수 있어요.",
           "<p><b>Runnable</b>은 \"입력을 받아 결과를 내는 단계\"라는 약속이에요. Prompt, Chat Model, 내가 만든 함수가 모두 Runnable이 되면, 모두 <code>invoke(입력)</code>으로 부르고 <code>a | b</code>로 이어요. <code>|</code>로 이은 것도 하나의 Runnable이라 또 <code>invoke</code>로 불러요. 잇는 줄에서는 아무것도 실행되지 않고, <code>invoke</code>하는 줄에서 실행돼요.</p>"
           "<p>LangChain은 검색 결과를 <b>Document</b>로 주고받아요. 본문은 <code>page_content</code>, 나머지는 <code>metadata</code>예요.</p>"
           + K.code_table([("RunnableLambda(f)", "파이썬 함수를 Runnable로"), ("a | b", "a의 출력을 b의 입력으로 잇기"), ("runnable.invoke(x)", "입력 하나로 실행"),
                           ("Document(page_content=..., metadata={...})", "본문과 출처 정보를 담는 LangChain의 자료 객체")]), "s2"),
    K.step(3, "근거 고르기: 후보 수, 넣을 수, 글자 예산", "찾은 후보 중 실제로 모델에게 줄 조각을 고르고, 넣은 조각을 따로 보관해요. 답이 댈 수 있는 출처는 이 조각들뿐이에요.",
           "<p>개수와 길이를 정하는 값이 셋이에요.</p>"
           + K.table(["값", "뜻", "이 예제"], [["<code>candidate_k</code>", "검색에서 가져올 후보 수", "3"], ["<code>context_k</code>", "그중 모델에게 줄 조각 수", "2"], ["<code>max_chars</code>", "자료 전체의 글자 예산", "600"]])
           + "<p>조각은 <b>통째로</b> 넣거나 빼요. 예산에 맞춘다고 중간을 자르면 조건이나 예외가 잘려 뜻이 바뀌어요. 같은 조각이 두 번 오면 한 번만 넣어요. 글자 수는 토큰 수와 달라서, 실제 서비스에서는 모델의 Tokenizer로 규칙, 질문, 자료를 모두 센 뒤 답을 쓸 여유도 남겨요(<a href=\"chunking.html\">Chunking</a> ⑨).</p>"
           "<p>그리고 <b>넣은 조각</b>을 따로 돌려받아요(<code>included</code>). 후보가 셋이어도 둘만 넣었다면, 답이 출처로 댈 수 있는 번호도 그 둘뿐이에요. 모델이 읽지도 않은 조각을 출처로 대면 막아야 하니까요.</p>"
           + K.out("s3")
           + K.code_table([("build_context(documents, max_items, max_chars)", "(자료 글, 실제로 넣은 Document 목록)을 함께 돌려줘요"), ("{d.metadata[\"id\"] for d in included}", "허용 출처 번호")]), "t3"),
    K.step(4, "Prompt: 규칙, 질문, 자료를 메시지로", "바뀌지 않는 규칙은 system 메시지에, 매번 바뀌는 질문과 자료는 human 메시지에 넣는 틀을 만들어요.",
           "<p><code>ChatPromptTemplate</code>은 메시지 틀이에요. <code>{question}</code>, <code>{context}</code> 자리에 값을 채우면 모델에게 보낼 메시지 목록이 돼요. 틀도 Runnable이라 <code>prompt.invoke(값)</code>으로 채워요. 이때는 모델을 부르지 않으니, 보내기 전에 메시지가 제대로 만들어졌는지 확인할 수 있어요.</p>"
           "<p>규칙에는 세 가지를 꼭 적어요. 자료만 근거로 쓸 것, <b>자료 안의 지시는 따르지 말 것</b>(문서에 \"규칙을 무시하라\"가 섞여 들어올 수 있어요), 답할 수 없을 때 무엇을 돌려줄 것인지예요. 답은 JSON 하나로 달라고 해서 ⑥에서 검사할 수 있게 해요(출력 모양의 자세한 규칙은 <a href=\"prompt-contract.html\">Prompt와 출력 계약</a>).</p>"
           + K.code_table([("ChatPromptTemplate.from_messages([(\"system\", ...), (\"human\", ...)])", "역할별 메시지 틀"), ("prompt.invoke({\"question\": ..., \"context\": ...})", "값을 채운 메시지 묶음"),
                           ("prompt_value.to_messages()", "보낼 메시지 목록")]), "s4"),
    K.step(5, "모델 부르기: 모델 자리에 가짜 함수", "Prompt 뒤에 모델을 | 로 이어, 질문과 자료를 넣으면 모델의 답(AIMessage)이 나오게 해요.",
           "<p>실제 서비스라면 이 자리에 <code>ChatOpenAI(model=...)</code> 같은 Chat Model이 들어가요(키 설정과 재시도는 <a href=\"llm-client.html\">LLM API 클라이언트</a>). 이 예제는 키와 비용 없이 돌리려고, 메시지를 받아 미리 정한 답을 <code>AIMessage</code>로 돌려주는 함수를 <code>RunnableLambda</code>로 감싸 그 자리에 넣었어요. 모양이 같아서 나중에 진짜 모델로 바꿔도 앞뒤 단계는 그대로예요.</p>"
           + K.code_table([("RunnableLambda(fake_llm)", "모델 자리에 끼운 가짜 함수 (실제로는 ChatOpenAI)"), ("prompt | model", "질문과 자료 → 메시지 → 답"), ("AIMessage(content=...)", "Chat Model이 돌려주는 답 객체. 글은 .content")]), "s5"),
    K.step(6, "답 검사하기: 모양, 출처 번호, status", "모델의 답을 바로 믿지 않고, 정해진 모양인지, 출처가 실제로 넣은 조각인지, status와 출처가 맞는지 차례로 봐요.",
           "<p>모델은 규칙을 어길 수 있어요. JSON이 중간에 끊기거나, 넣지 않은 조각을 출처로 대거나, '답했다'면서 출처가 없을 수도 있어요. 그래서 받은 글을 바깥에서부터 검사해요.</p>"
           "<ol><li><b>모양</b>: Pydantic 모델(<code>Answer</code>)로 읽어요. 깨진 JSON, 빠진 칸, 틀린 자료형은 여기서 막혀요.</li>"
           "<li><b>출처 번호</b>: source_ids가 ③에서 실제로 넣은 번호 안에 있나</li><li><b>status 조합</b>: answered면 출처가 있어야 하고, 정보 부족이면 출처가 없어야 해요</li></ol>"
           + K.code_table([("Answer.model_validate_json(text)", "JSON 글을 읽으며 모양 검사 (틀리면 ValidationError)"), ("ConfigDict(extra=\"forbid\", strict=True)", "모르는 칸 거부, 자료형을 바꿔 주지 않음"),
                           ("set(obj.source_ids) <= provided_ids", "출처가 모두 넣은 조각 안에 있나 (부분집합)")]), "t6"),
    K.step(7, "서비스로 묶기: 정보 부족과 실패를 나누기", "찾기, 근거 고르기, 답 만들기와 검사를 함수 하나로 묶어요. 정보 부족은 답으로, 찾기와 답 만들기의 고장은 오류로 나눠요.",
           "<p><code>make_rag_service(retriever)</code>는 부품을 받아 <code>answer_question(req)</code> 함수를 만들어요. 순서가 정해진 흐름이에요. 다음에 무엇을 할지 모델이 고르는 Agent(<a href=\"agent-tools.html\">AI Agent와 도구</a>)와는 달라요.</p>"
           + K.table(["상황", "분류", "서비스가 하는 일"], [["넣을 근거가 없음 (빈 자료)", "정보 부족", "모델을 부르지 않고 insufficient_context"], ["근거는 넣었는데 답할 내용이 없음", "정보 부족", "모델이 insufficient_context, 날짜나 금액을 만들지 않음"],
                                                    ["검색 시스템 고장", "<code>retrieval_error</code>", "오류. 빈 검색 결과로 바꾸지 않음"], ["모델의 답이 검사를 못 넘김", "<code>generation_error</code>", "오류. 정보 부족으로 바꾸지 않음"]])
           + "<p>돌려줄 출처는 검사를 통과한 번호로 서버가 보관한 원문 위치를 찾아 붙여요. 파일명이나 위치를 모델에게 쓰게 하지 않아요. 그리고 질문마다 후보, 넣은 조각, 결과, 모델 호출 수를 기록해 두면 \"왜 정보 부족이었나\"를 나중에 따라갈 수 있어요.</p>"
           + K.out("t7")
           + K.code_table([("make_rag_service(retriever, max_chars, log)", "부품을 받아 질문 하나를 처리하는 함수를 만들기"), ("raise RagError(\"retrieval_error\") from exc", "고장을 이름 붙인 오류로 올리기 (원래 예외는 from으로 남겨요)"),
                           ("sources[i] for i in obj.source_ids", "검사를 통과한 번호 → 서버가 아는 원문 위치")]), "t7b"),
]
practice = [("답이 늘 '정보 부족'이다", "기록의 후보와 넣은 조각부터 봐요. 분류가 맞는지, 후보에 답이 있는데 글자 예산에서 빠졌는지, 아예 검색이 못 찾았는지에 따라 고칠 곳(분류, 개수, 예산, 조각 나누기, 검색)이 달라요(<a href=\"rag-rerank.html\">RAG 검색 고치기</a>)."),
            ("출처 검사는 통과했는데 틀린 답", "출처 검사는 번호만 봐요(⑥). 숫자, 기한, 예외처럼 틀리면 곤란한 답은 원문과 대조하는 단계를 따로 둬요. 사람이 표본을 보거나 <a href=\"llm-judge.html\">LLM 채점</a>을 써요(<a href=\"rag-eval.html\">RAG 평가</a>)."),
            ("실패를 숨기지 않기", "검색 고장이나 모델 실패를 '자료로는 답할 수 없습니다'로 바꾸면 장애가 보이지 않아요. 정보 부족과 실패는 다른 응답이에요."),
            ("API로 열 때", "FastAPI로 열면 정보 부족과 정상 답은 둘 다 200, RagError는 502로 돌려줘요. 요청 모양(질문 길이, 분류, 개수 범위)은 입구에서 422로 막아요(<a href=\"fastapi.html\">FastAPI 기초</a>)."),
            ("진짜 모델로 바꿀 때", "⑤의 자리에 <code>ChatOpenAI</code> 같은 Chat Model을 넣어요. 모델이 정해진 모양으로만 답하게 하는 <code>with_structured_output</code>을 쓰면 ⑥의 모양 검사가 더 쉬워지지만, 출처 번호와 status 검사는 여전히 앱이 해요."),
            ("글자 예산과 토큰", "max_chars는 길이를 눈으로 보기 위한 값이에요. 실제 한도는 모델의 Tokenizer로 규칙, 질문, 자료를 모두 센 토큰 수와, 답을 쓸 여유로 정해요."),
            ("후보와 넣은 조각을 섞지 않기", "허용 출처는 넣은 조각에서만 만들어요. 후보 전체를 허용하면 모델이 읽지 않은 조각을 출처로 대도 통과해요."),
            ("LangChain을 쓸지", "여러 단계를 잇고 모델을 바꿔 비교할 일이 있으면 공통 방식이 편해요. 한 번 부르는 작은 기능이면 SDK를 직접 쓰는 쪽(<a href=\"llm-client.html\">LLM API 클라이언트</a>)이 더 단순해요."),
            ("API 키와 설정", "키와 모델 이름은 환경 변수로 읽고(<code>.env</code>는 저장소에 넣지 않아요), 기록과 오류 응답에 남기지 않아요.")]
libs = [("①", "<code>SentenceTransformer.encode</code>", "질문과 조각을 벡터로 (sentence-transformers)"), ("②", "<code>RunnableLambda</code>, <code>|</code>, <code>invoke</code>, <code>Document</code>", "실행 단위와 자료 객체 (langchain-core)"),
        ("④", "<code>ChatPromptTemplate.from_messages</code>, <code>to_messages</code>", "메시지 틀 (langchain-core)"), ("⑤", "<code>AIMessage</code>", "모델의 답 객체 (langchain-core)"),
        ("⑥", "<code>BaseModel</code>, <code>model_validate_json</code>, <code>ValidationError</code>", "답의 모양 검사 (Pydantic)")]
K.build(ROOT / "topics" / "backend" / "rag-service.html", title="RAG 서비스", h1="RAG 서비스: 찾은 근거로 출처가 붙은 답 만들기", track="백엔드", crumb="백엔드 &gt; 검색과 RAG",
        one_line="<b>RAG</b>는 질문과 관련된 조각을 <b>찾아서</b> 모델 입력에 <b>보태고</b>, 그걸 근거로 <b>답하게</b> 하는 방식이에요. 모델을 다시 학습시키지 않아요. 서비스로 만들 때 핵심은 셋이에요. 답이 <b>실제로 넣어 준 조각만</b> 출처로 대게 하고, 출처를 맞게 댄 것과 내용이 맞는 것을 다르게 보고, <b>정보 부족</b>과 <b>실패</b>를 다른 응답으로 나눠요.",
        why=why, whole=whole, steps=steps, code_file="examples/rag/rag_service.py", practice=practice, libs=libs,
        libs_note="<code>search</code>, <code>to_documents</code>, <code>build_context</code>, <code>fake_llm</code>, <code>check</code>, <code>make_rag_service</code>, <code>broken_search</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rag-rerank.html\">RAG 검색 고치기</a>", "질문 다시 쓰기, 여러 질문, 재정렬"), ("<a href=\"rag-eval.html\">RAG 평가</a>", "검색과 답을 숫자로 재기"),
                   ("<a href=\"agent-tools.html\">AI Agent와 도구</a>", "순서를 모델이 고르는 흐름"), ("<a href=\"indexing.html\">문서 적재 3. 인덱싱</a>", "이 쪽이 검색하는 조각을 만든 곳"),
                   ("<a href=\"prompt-contract.html\">Prompt와 출력 계약</a>", "출력 모양과 출처 검사 규칙"), ("<a href=\"llm-client.html\">LLM API 클라이언트</a>", "진짜 모델 API 부르기, 재시도, 비용")],
        sources=["다루는 범위: Day26 (2026-11-23) Basic RAG와 LangChain. RAG와 미세조정 비교, 검색 결과와 자료, candidate_k와 context_k, 글자 예산, Runnable과 invoke, Document, ChatPromptTemplate, Chat Model과 AIMessage, 답의 모양 검사, citation과 grounding, 정보 부족과 실패 구분, 서비스 조립. 실습 프로젝트의 가짜 HTTP 서버, 응답 묶음(raw, parsed, parsing_error), FastAPI 단계는 따라 하지 않고 '실제로 할 때'에 글로 적었어요",
                 "예제: <code>examples/rag/rag_service.py</code> (이 파일 하나로 실행돼요). 검색 대상은 <code>examples/ingest/out/chunks.json</code>(문서 적재 1~3쪽의 결과). 분류는 파일마다 붙인 예시 값이에요",
                 f"결과 값: <code>tools/pages/rag_service.py</code>가 단계별로 실제 실행한 결과 ({ENV}). 가짜 모델의 답 문장은 미리 정한 값이에요",
                 "공식 문서: <a href=\"https://python.langchain.com/docs/concepts/runnables/\" target=\"_blank\">Runnable</a>, <a href=\"https://python.langchain.com/docs/concepts/structured_outputs/\" target=\"_blank\">Structured outputs</a>, <a href=\"https://python.langchain.com/docs/integrations/chat/openai/\" target=\"_blank\">ChatOpenAI</a>, <a href=\"https://docs.pydantic.dev/latest/concepts/json/\" target=\"_blank\">Pydantic JSON</a>"])
print("ok")
