"""모델 > 순서와 언어 모델 > LLM 원리 (Token, Embedding, Attention, Mask, 다음 토큰).
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/llm_principles.py
"""
import os
import sys
import warnings
from html import escape as E
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "llm_theory"
code = S.parse((EX / "llm_parts.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="llm_parts.py")
tk = ns["tokens"]
r2 = lambda t: [[round(float(v), 2) for v in row] for row in t]  # noqa: E731
outs = {}
outs["why"] = S.table("질문 하나가 LLM 안에서 거치는 길", ["단계", "무엇이 되나", "이 쪽"], [["글", f"\"{E(ns['text'])}\"", ""], ["토큰", f"{len(tk)}개 조각", "①"], ["벡터", f"토큰마다 {ns['vectors'].shape[1]}칸", "②"],
                                                                                    ["Attention", "다른 토큰을 참고해 고친 벡터", "③④⑤"], ["다음 토큰", "후보마다 확률", "⑥"]],
                      ["LLM은 이 과정을 한 토큰 만들 때마다 반복해요. 만든 토큰을 입력 끝에 붙이고 다시 처음부터요."])
outs["s1"] = S.box("① tok(text)", [(f"tokens → {tk}", 1), f"ids → {ns['ids'].tolist()}", (f"글자 {len(ns['text'])}개 → 토큰 {len(tk)}개 (표시용 2개 포함)", 2)],
                   ["<code>▁</code>는 \"앞에 띄어쓰기가 있던 조각\"이에요. \"서버를\"은 <code>▁서버</code>와 <code>를</code>로 나뉘었어요. 자주 나오는 말은 통째로, 드문 말은 잘게 쪼개요.", "<code>&lt;s&gt;</code>, <code>&lt;/s&gt;</code>는 시작과 끝을 알리는 표시예요. LLM의 입력 한도(Context Window)와 비용은 글자가 아니라 이 토큰 수로 세요."])
outs["s2"] = S.box("② Embedding 표", [(f"word_embeddings → {tuple(ns['table'].weight.shape)}", 1), (f"vectors.shape → {tuple(ns['vectors'].shape)}", 2), f"'▁서버' 벡터 앞 5칸 → {[round(float(v), 4) for v in ns['vectors'][1][:5]]}"],
                   [f"어휘 {ns['table'].weight.shape[0]:,}개 각각에 {ns['table'].weight.shape[1]}칸 벡터가 하나씩 있는 큰 표예요. 학습으로 정해진 값이에요.", "토큰 번호로 표에서 줄을 찾아오는 것뿐이에요. 이 단계에서는 아직 앞뒤 문맥을 몰라요. \"배\"가 과일인지 배인지 정하는 건 다음 단계(Attention)예요."])
w = ns["weights"]
sub = tk[1:5]
outs["s3"] = S.table("③ 앞 네 토큰의 Attention 가중치 (줄: 보는 쪽, 칸: 보이는 쪽)", [""] + [E(t) for t in sub], [[E(sub[i])] + [f"{v:.2f}" for v in row] for i, row in enumerate(r2(w))],
                     ["줄마다 합이 1이에요. 예를 들어 첫 줄은 \"<code>▁서버</code>\"가 새 표현을 만들 때 각 토큰의 내용을 몇 퍼센트씩 섞는지예요.", "여기서는 Q, K, V를 만드는 행렬을 무작위로 정해서 숫자 자체에 뜻은 없어요. 실제 모델은 이 행렬을 학습해서, 관련 있는 토큰에 가중치가 몰리게 돼요 (⑤)."])
wc = ns["weights_causal"]
outs["s4"] = S.table("④ 미래를 가린 뒤의 가중치", [""] + [E(t) for t in sub], [[E(sub[i])] + [("<b>0.00</b>" if j > i else f"{v:.2f}") for j, v in enumerate(row)] for i, row in enumerate(r2(wc))],
                     ["대각선 위(자기보다 뒤의 토큰)가 모두 0이 됐어요. 첫 토큰은 자기 자신만 봐요.", "다음 토큰을 맞히는 LLM(GPT 같은)은 학습할 때 답을 미리 보면 안 돼서 이렇게 가려요. 반대로 이 예제의 e5 같은 인코더는 문장 전체를 앞뒤로 다 봐요."])
fs, fl = ns["first_layer"]
ls, ll = ns["last_layer"]
outs["s5"] = S.table("⑤ 실제 e5 모델: 토큰마다 가장 많이 본 토큰 (head 12개 평균, 표시용 토큰 제외)", ["토큰", "1층에서", "12층에서"],
                     [[E(tk[i]), E(fl[i]), E(ll[i])] for i in range(1, len(tk) - 1)],
                     [f"attentions → 층 {ns['attn'].shape[0]}개, 층마다 head {ns['attn'].shape[2]}개, {ns['attn'].shape[3]}×{ns['attn'].shape[4]} 가중치",
                      "1층에서는 대부분 <b>바로 앞 토큰</b>을 봐요 (\"를\" → \"서버\", \"하려면\" → \"실행\"). 단어 조각을 이어 붙이는 일이에요.",
                      f"12층에서는 모두 \"<code>▁서버</code>\"를 봐요. 문장의 핵심 대상에 정보가 모였어요. 그리고 가중치의 {ls:.0%}는 시작 표시 <code>&lt;s&gt;</code>로 가요. 딱히 볼 곳이 없을 때 쉬는 자리처럼 쓰여요."])
pr = ns["probs"]
outs["s6"] = S.table("⑥ 후보 네 개의 확률 (점수는 예시)", ["후보", "점수(logit)"] + [f"temperature {t}" for t in pr], [[E(c), f"{float(ns['logits'][i]):.1f}"] + [f"{float(pr[t][i]):.2f}" for t in pr] for i, c in enumerate(ns["candidates"])],
                     [f"가장 높은 것만 고르면(greedy) 언제나 \"{ns['greedy']}\"예요.", f"확률대로 뽑으면(temperature 1.0, 10번) → {ns['sampled']}. 같은 질문에도 답이 조금씩 달라지는 이유예요.",
                      "temperature가 낮으면 1등에 몰리고(0.5에서 0.80), 높으면 고르게 퍼져요(2.0에서 0.42). 정확해야 하면 낮게, 다양해야 하면 높게 써요."])
flow = [[1, "토큰", "▁서버, 를, …", "한도와 비용의 단위"], [2, "Embedding", "번호 → 벡터", "아직 문맥 모름"], [3, "Attention", "Q, K, V", "가중치 합 1"], [4, "Mask", "미래 가리기", "GPT식 학습"], [5, "실제 모델", "12층", "앞 토큰 → 핵심어"], [6, "다음 토큰", "softmax", "temperature"]]
S.save(ROOT / "topics" / "model" / "llm-principles-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                             "env": f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, intfloat/multilingual-e5-small"})
why = f"""<p>LLM(대규모 언어 모델)은 <b>지금까지의 글을 보고 다음 토큰을 맞히는</b> 모델이에요. 그걸 수없이 반복해서 문장을 만들어요. 똑똑해 보이는 답도 결국 \"다음에 올 조각의 확률\"을 고른 결과예요. 그래서 그럴듯하지만 틀린 답(환각)도 나와요.</p>
{K.out("why")}
<p>이 쪽에서는 이 길을 한 단계씩, 이 PC에 있는 실제 Transformer 모델(<code>multilingual-e5-small</code>, 12층)로 따라가요. 다만 이 모델은 문장을 벡터로 바꾸는 인코더라 다음 토큰을 만들지는 않아요. ⑥은 원리만 예시 점수로 계산해요.</p>"""
whole = """<p>글 → 토큰 → 벡터 → Attention으로 문맥 섞기 → (미래 가리기) → 다음 토큰 확률. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "글을 토큰으로", "Tokenizer가 글을 모델이 아는 조각(토큰)으로 나누고 번호를 붙여요.",
           K.code_table([("AutoTokenizer.from_pretrained(MODEL)", "그 모델의 Tokenizer"), ("tok(text, return_tensors=\"pt\")", "글을 토큰 번호 Tensor로"), ("convert_ids_to_tokens(ids)", "번호를 다시 조각 글자로")]), "s1"),
    K.step(2, "토큰을 벡터로 (Embedding)", "토큰 번호마다 정해진 벡터를 표에서 찾아와요.",
           K.code_table([("model.embeddings.word_embeddings", "토큰 번호 → 벡터 표 (<code>nn.Embedding</code>)"), ("table(ids)", "번호들로 줄을 찾아 (토큰 수, 384)")]), "s2"),
    K.step(3, "Attention을 손으로 계산하기", "토큰마다 Query, Key, Value를 만들고, Query와 Key가 잘 맞을수록 그 토큰의 Value를 많이 섞어요.",
           "<p>도서관에 빗대면, <b>Query</b>는 \"내가 찾는 것\", <b>Key</b>는 \"책 등의 제목\", <b>Value</b>는 \"책 내용\"이에요. 찾는 것과 제목이 잘 맞는 책의 내용을 많이 가져와요.</p>"
           + K.code_table([("Q, K, V = x @ Wq, x @ Wk, x @ Wv", "같은 벡터를 세 행렬로 바꿔 세 역할로"), ("Q @ K.T / d ** 0.5", "모든 토큰 쌍의 맞음 점수. √d로 나눠 너무 커지지 않게"),
                           ("torch.softmax(scores, dim=-1)", "줄마다 합이 1인 가중치"), ("weights @ V", "가중치만큼 내용을 섞은 새 표현")]), "s3"),
    K.step(4, "뒤를 못 보게 막기 (Mask)", "자기보다 뒤의 토큰 점수를 −∞로 바꿔, softmax 뒤 가중치가 0이 되게 해요.",
           K.code_table([("torch.triu(..., diagonal=1)", "대각선 위쪽만 True. 미래 자리"), ("scores.masked_fill(causal, float(\"-inf\"))", "미래 자리 점수를 −∞로")]), "s4"),
    K.step(5, "실제 모델의 Attention 보기", "e5 모델에 문장을 넣고, 12개 층의 Attention 가중치를 꺼내 봐요.",
           "<p>Transformer는 ③의 Attention(과 작은 신경망)을 한 층으로 해서 여러 층 쌓은 모델이에요. 층마다 head 여러 개가 서로 다른 관계를 봐요.</p>"
           + K.code_table([("AutoModel.from_pretrained(MODEL, attn_implementation=\"eager\")", "가중치를 꺼낼 수 있는 계산 방식으로 모델 불러오기"), ("model(**enc, output_attentions=True)", "층마다 Attention 가중치도 돌려줘요."),
                           ("torch.stack(out.attentions)", "(층, 배치, head, 토큰, 토큰)"), ("a.argmax(-1)", "줄마다 가장 큰 가중치의 위치")]), "s5"),
    K.step(6, "다음 토큰 고르기", "모델이 후보마다 준 점수를 softmax로 확률로 바꾸고, 그중 하나를 골라요.",
           K.code_table([("torch.softmax(logits / t, dim=-1)", "temperature t로 나눈 뒤 확률로"), ("logits.argmax()", "가장 높은 것 (greedy)"), ("torch.multinomial(probs, 1)", "확률대로 하나 뽑기 (sampling)")]), "s6"),
]
practice = [("Context Window", "모델이 한 번에 볼 수 있는 토큰 수의 한도예요. 대화가 길어지면 앞부분이 잘려서 \"잊어버린\" 것처럼 보여요. 그래서 필요한 자료만 골라 넣어요 (<a href=\"../backend/prompt-contract.html\">Prompt와 출력 계약</a>)."),
            ("한국어는 토큰이 많다", "같은 뜻이라도 영어보다 토큰이 많이 나오는 편이라 비용과 한도에 영향을 줘요. 실제 토큰 수는 그 모델의 Tokenizer로 세요."),
            ("환각", "모델은 \"사실인가\"가 아니라 \"그럴듯한가\"로 다음 토큰을 골라요. 근거 자료를 주고(RAG), 답을 검사하는 이유예요."),
            ("학습과 추론", "학습은 엄청난 글로 가중치를 정하는 일이고, 우리가 API로 쓰는 건 정해진 가중치로 다음 토큰을 계산하는 추론이에요.")]
libs = [("①", "<code>AutoTokenizer</code>, <code>convert_ids_to_tokens()</code>", "Tokenizer (transformers)"), ("②⑤", "<code>AutoModel</code>, <code>word_embeddings</code>, <code>output_attentions=True</code>", "모델 안 들여다보기"),
        ("③④⑥", "<code>torch.softmax</code>, <code>torch.triu</code>, <code>masked_fill</code>, <code>torch.multinomial</code>", "Attention과 고르기 계산 (torch)")]
K.build(ROOT / "topics" / "model" / "llm-principles.html", title="LLM 원리", h1="LLM 원리: Token, Embedding, Attention, 다음 토큰", track="모델", crumb="모델 &gt; 순서와 언어 모델",
        one_line="LLM은 글을 <b>토큰</b>으로 나누고, <b>벡터</b>로 바꾸고, <b>Attention</b>으로 다른 토큰을 참고해 문맥을 섞은 뒤, <b>다음 토큰의 확률</b>을 계산해 하나씩 이어 붙여요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm_theory/llm_parts.py", practice=practice, libs=libs, libs_note="<code>looks_at</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"sequence-models.html\">순서 모델</a>", "RNN과 Attention의 출발점"), ("<a href=\"../backend/llm-client.html\">LLM API 클라이언트</a>", "실제 LLM 부르기"), ("<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>", "같은 e5 모델로 뜻 검색")],
        sources=["다루는 범위: Day20 (2026-11-13) LLM 생태계, Token, Context, Transformer, Attention, 학습과 추론", "모델: 이 PC에 받아 둔 <code>intfloat/multilingual-e5-small</code>. ③의 Q, K, V 행렬과 ⑥의 점수는 설명용 예시 값",
                 "결과 값: <code>tools/pages/llm_principles.py</code>가 단계별로 실제 실행한 결과", "공식 문서: <a href=\"https://huggingface.co/docs/transformers/main_classes/tokenizer\" target=\"_blank\">Tokenizer</a>, <a href=\"https://arxiv.org/abs/1706.03762\" target=\"_blank\">Attention Is All You Need</a>"])
print("ok")
