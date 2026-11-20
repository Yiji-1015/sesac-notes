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
DATA_JS = ROOT / "topics" / "model" / "llm-principles-data.js"
try:
    ns = S.run(code, 6, S.workdir(EX), filename="llm_parts.py")
    REUSE = False
except OSError:  # e5 모델을 받아 두지 않은 PC: 모델 실행 값(①~⑥)은 이전에 실제로 실행해 저장한 값을 그대로 쓴다
    import json
    prev = json.loads(DATA_JS.read_text(encoding="utf-8")[len("window.PAGE = "):].rstrip().rstrip(";"))
    REUSE = True
    print("e5 모델이 없어 ①~⑥ 출력은 기존 data.js 값을 재사용합니다.")
r2 = lambda t: [[round(float(v), 2) for v in row] for row in t]  # noqa: E731
outs = {}
if REUSE:
    outs = {k: v for k, v in prev["outs"].items() if k in ("why", "s1", "s2", "s3", "s4", "s5", "s6")}
    ENV_PREV = prev["env"]
else:
    tk = ns["tokens"]
    outs["why"] = S.table("질문 하나가 LLM 안에서 거치는 길", ["단계", "무엇이 되나", "이 쪽"], [["글", f"\"{E(ns['text'])}\"", ""], ["토큰", f"{len(tk)}개 조각", "①"], ["벡터", f"토큰마다 {ns['vectors'].shape[1]}칸", "②"],
                                                                                        ["Attention", "다른 토큰을 참고해 고친 벡터", "③④⑤"], ["다음 토큰", "후보마다 확률", "⑥"]],
                          ["LLM은 이 과정을 한 토큰 만들 때마다 반복해요. 만든 토큰을 입력 끝에 붙이고 다시 처음부터요."])
    S.claim(any(a_ == "▁서버" and b_ == "를" for a_, b_ in zip(tk, tk[1:])), "① 서버를이 ▁서버와 를로 나뉜다")
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
    mid = range(1, len(tk) - 1)   # 시작, 끝 표시를 뺀 토큰 자리
    S.claim(sum(fl[i] == tk[i - 1] for i in mid) > len(mid) / 2 and fl[tk.index("를")] == "▁서버" and fl[tk.index("하려면")].lstrip("▁") == "실행",
            "⑤ 1층은 대부분 바로 앞 토큰을 본다 (를 → 서버, 하려면 → 실행)")
    S.claim(all(ll[i] == "▁서버" for i in mid), "⑤ 12층에서는 모두 ▁서버를 본다")
    outs["s5"] = S.table("⑤ 실제 e5 모델: 토큰마다 가장 많이 본 토큰 (head 12개 평균, 표시용 토큰 제외)", ["토큰", "1층에서", "12층에서"],
                         [[E(tk[i]), E(fl[i]), E(ll[i])] for i in range(1, len(tk) - 1)],
                         [f"attentions → 층 {ns['attn'].shape[0]}개, 층마다 head {ns['attn'].shape[2]}개, {ns['attn'].shape[3]}×{ns['attn'].shape[4]} 가중치",
                          "1층에서는 대부분 <b>바로 앞 토큰</b>을 봐요 (\"를\" → \"서버\", \"하려면\" → \"실행\"). 단어 조각을 이어 붙이는 일이에요.",
                          f"12층에서는 모두 \"<code>▁서버</code>\"를 봐요. 문장의 핵심 대상에 정보가 모였어요. 그리고 가중치의 {ls:.0%}는 시작 표시 <code>&lt;s&gt;</code>로 가요. 딱히 볼 곳이 없을 때 쉬는 자리처럼 쓰여요."])
    pr = ns["probs"]
    top = int(ns["logits"].argmax())   # 점수 1등 후보
    outs["s6"] = S.table("⑥ 후보 네 개의 확률 (점수는 예시)", ["후보", "점수(logit)"] + [f"temperature {t}" for t in pr], [[E(c), f"{float(ns['logits'][i]):.1f}"] + [f"{float(pr[t][i]):.2f}" for t in pr] for i, c in enumerate(ns["candidates"])],
                         [f"가장 높은 것만 고르면(greedy) 언제나 \"{ns['greedy']}\"예요.", f"확률대로 뽑으면(temperature 1.0, 10번) → {ns['sampled']}. 같은 질문에도 답이 조금씩 달라지는 이유예요.",
                          f"temperature가 낮으면 1등에 몰리고(0.5에서 {float(pr[0.5][top]):.2f}), 높으면 고르게 퍼져요(2.0에서 {float(pr[2.0][top]):.2f}). 정확해야 하면 낮게, 다양해야 하면 높게 써요."])
flow = [[1, "토큰", "▁서버, 를, …", "한도와 비용의 단위"], [2, "Embedding", "번호 → 벡터", "아직 문맥 모름"], [3, "Attention", "Q, K, V", "가중치 합 1"], [4, "Mask", "미래 가리기", "GPT식 학습"], [5, "실제 모델", "12층", "앞 토큰 → 핵심어"], [6, "다음 토큰", "softmax", "temperature"]]
# ---------------- Transformer 구조 실험 (모델 없이 torch로 직접 계산)
import numpy as np  # noqa: E402
sys.path.insert(0, str(ROOT / "tools" / "visuals"))
from vstyle import plt, C, setup  # noqa: E402
setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "llm"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/llm/{name}.png"


def attend(x, Wq, Wk, Wv, d):
    w_ = torch.softmax((x @ Wq) @ (x @ Wk).T / d ** 0.5, dim=-1)
    return w_ @ (x @ Wv)


# 1) 순서를 섞어도 Attention은 같은 값을 순서만 바꿔 돌려준다
torch.manual_seed(0)
xs = torch.randn(5, 16)
Wq_, Wk_, Wv_ = (torch.randn(16, 8) for _ in range(3))
perm = torch.tensor([3, 0, 4, 1, 2])
o_plain, o_perm = attend(xs, Wq_, Wk_, Wv_, 8), attend(xs[perm], Wq_, Wk_, Wv_, 8)
same_perm = bool(torch.allclose(o_plain[perm], o_perm, atol=1e-5))
pe_dim, n_pos = 64, 50
pos = torch.arange(n_pos)[:, None].float()
freq = torch.exp(-np.log(10000.0) * torch.arange(0, pe_dim, 2).float() / pe_dim)
PE = torch.zeros(n_pos, pe_dim)
PE[:, 0::2], PE[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
o_pe, o_pe_perm = attend(xs + PE[:5, :16], Wq_, Wk_, Wv_, 8), attend(xs[perm] + PE[:5, :16], Wq_, Wk_, Wv_, 8)
same_pe = bool(torch.allclose(o_pe[perm], o_pe_perm, atol=1e-5))
outs["t3"] = S.table("토큰 5개의 순서를 3, 0, 4, 1, 2로 섞어서 같은 Attention에 넣으면", ["입력", "결과가 \"순서만 바뀐 같은 값\"인가"],
                     [["벡터만", f"<b>{'예' if same_perm else '아니오'}</b>"], ["벡터 + 위치 벡터", f"<b>{'예' if same_pe else '아니오'}</b>"]],
                     ["Attention은 토큰들을 \"집합\"으로 봐요. 순서를 섞으면 결과도 똑같이 섞일 뿐, 값은 그대로예요. \"개가 사람을 물었다\"와 \"사람이 개를 물었다\"를 구별하지 못한다는 뜻이에요.",
                      "자리마다 다른 위치 벡터를 더하면 같은 토큰도 자리에 따라 다른 벡터가 돼서, 순서가 결과에 들어가요. 그래서 Transformer는 처음에 위치 정보를 꼭 더해요."])
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.5, 1]})
im = a.imshow(PE.numpy().T, aspect="auto", cmap="PuOr_r", vmin=-1, vmax=1)
a.set(xlabel="자리 (0부터)", ylabel="벡터의 칸 (0~63)", title="사인, 코사인 위치 벡터: 칸마다 주기가 달라요")
a.grid(False)
fig.colorbar(im, ax=a, fraction=0.03)
sim = (PE @ PE[10]) / (PE.norm(dim=1) * PE[10].norm())
b.plot(range(n_pos), sim.numpy(), color=C["purple"], lw=2.2)
b.axvline(10, color=C["gray"], ls="--", lw=1)
b.set(xlabel="자리", ylabel="10번 자리와의 코사인 유사도", title="가까운 자리일수록 비슷한 벡터")
P_PE = fig_save(fig, "positional_encoding")
outs["f_pe"] = S.images("자리 50개의 위치 벡터 (64칸)", [(P_PE, "")],
                        ["앞쪽 칸은 사인, 코사인이 빠르게 바뀌고 뒤쪽 칸은 아주 천천히 바뀌어요. 시계의 초침, 분침, 시침처럼 여러 주기를 겹쳐서 자리마다 다른 무늬를 만들어요.",
                         "오른쪽: 10번 자리와 가까운 자리일수록 벡터가 비슷해요. 그래서 모델이 \"얼마나 떨어져 있나\"를 쉽게 알 수 있어요."])
outs["f_pe"]["chart"] = True
# 2) 세 가지 가리기 모양
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
n_ = 6
full = np.ones((n_, n_))
causal_m = np.tril(np.ones((n_, n_)))
cross = np.ones((4, n_))
for ax, (m_, ttl, xl, yl) in zip(axs, [(full, "인코더 (BERT, e5)\n앞뒤를 다 봐요", "보이는 토큰", "보는 토큰"), (causal_m, "디코더 (GPT)\n자기와 앞만 봐요", "보이는 토큰", "보는 토큰"),
                                        (cross, "인코더-디코더의 교차 Attention (T5)\n만드는 쪽이 입력 전체를 봐요", "입력 문장의 토큰", "만드는 문장의 토큰")]):
    ax.imshow(m_, cmap="Purples", vmin=0, vmax=1.3)
    ax.set(title=ttl, xlabel=xl, ylabel=yl, xticks=range(m_.shape[1]), yticks=range(m_.shape[0]))
    ax.grid(False)
P_MASK = fig_save(fig, "attention_masks")
outs["f_mask"] = S.images("세 가지 Transformer가 \"누가 누구를 보나\"", [(P_MASK, "진한 칸: 볼 수 있음, 흰 칸: 가림")],
                          ["④에서 만든 미래 가리기가 가운데 모양이에요. 같은 Attention 계산에 가리는 모양만 바꿔서 세 종류의 모델이 나와요."])
outs["f_mask"]["chart"] = True
# 3) 길이에 따른 Attention 점수표 크기
cost_rows = []
for n_tok in (512, 4096, 32768, 131072):
    cells = n_tok * n_tok
    cost_rows.append([f"{n_tok:,}", f"{cells:,}", f"{cells * 12 * 12 * 2 / 2 ** 30:,.1f} GB"])
outs["t5"] = S.table("토큰 수에 따른 Attention 점수표 크기 (e5-small처럼 12층 × head 12개, 숫자 하나 2바이트로 계산)", ["토큰 수", "head 하나의 점수 칸 수 (n × n)", "전부 저장하면"], cost_rows,
                     ["토큰이 8배가 되면 점수표는 64배가 돼요. 모든 토큰 쌍을 비교하니 계산과 메모리가 토큰 수의 제곱으로 늘어요.",
                      "그래서 긴 문맥을 다루는 모델은 점수표를 한꺼번에 만들지 않고 조각씩 계산하거나(FlashAttention), 가까운 토큰만 보거나, 앞 토큰들의 Key와 Value를 저장해 다시 쓰는(KV 캐시) 방법을 써요."])
S.save(DATA_JS, {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                 "env": ENV_PREV if REUSE else f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, intfloat/multilingual-e5-small"})
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
           "<p>\"잘 맞음\"은 Query와 Key의 내적(칸마다 곱해 더한 값)으로 재요. 같은 방향일수록 커요. 그런데 벡터 칸 수 d가 크면 내적도 커져서(칸마다 곱한 값을 d개 더하니까), softmax가 1등 하나에 거의 다 몰려 버리고 기울기가 사라져요. 그래서 √d로 나눠 크기를 맞춰요.</p>"
           "<p><b>여러 head</b>: 한 번의 Attention은 한 가지 관계만 잘 잡아요. 그래서 Q, K, V 행렬을 여러 벌 두고(head) 따로 계산한 뒤 결과를 이어 붙여요. 어떤 head는 바로 앞 토큰을, 어떤 head는 주어를, 어떤 head는 따옴표 짝을 보는 식으로 나뉘어요. ⑤에서는 실제 모델의 head 12개를 평균 내서 봐요.</p>"
           "<p><b>순서는 어디에?</b> 이 계산에는 \"몇 번째 토큰인가\"가 전혀 없어요. 아래 🔁처럼 순서를 섞어도 결과가 똑같이 섞일 뿐이에요. 그래서 Transformer는 처음에 자리마다 다른 <b>위치 벡터</b>를 토큰 벡터에 더해요. 처음 Transformer는 사인, 코사인으로 만든 고정 벡터를 썼고, 요즘은 학습하는 위치 벡터나, Query와 Key를 자리에 따라 돌려서 \"얼마나 떨어졌나\"를 넣는 RoPE를 많이 써요.</p>"
           + K.out("f_pe")
           + K.code_table([("Q, K, V = x @ Wq, x @ Wk, x @ Wv", "같은 벡터를 세 행렬로 바꿔 세 역할로"), ("Q @ K.T / d ** 0.5", "모든 토큰 쌍의 맞음 점수. √d로 나눠 너무 커지지 않게"),
                           ("torch.softmax(scores, dim=-1)", "줄마다 합이 1인 가중치"), ("weights @ V", "가중치만큼 내용을 섞은 새 표현")])
           + K.try_("토큰 순서를 섞어서 넣으면 결과가 달라질까?", "t3"), "s3"),
    K.step(4, "뒤를 못 보게 막기 (Mask)", "자기보다 뒤의 토큰 점수를 −∞로 바꿔, softmax 뒤 가중치가 0이 되게 해요.",
           "<p>같은 Attention에 <b>가리는 모양</b>만 바꾸면 Transformer의 세 갈래가 나와요. 무엇을 맞히며 미리 배우는지(사전학습 목표)도 같이 달라요.</p>"
           + K.table(["갈래", "보는 범위", "미리 배우는 방법", "잘하는 일", "예"],
                     [["<b>인코더</b>", "앞뒤 전부", "문장 중간을 가리고(마스크) 가린 토큰 맞히기", "문장 이해, 분류, 문장 벡터(검색)", "BERT, e5"],
                      ["<b>디코더</b>", "자기와 앞만", "다음 토큰 맞히기", "글 만들기, 대화", "GPT, 대부분의 LLM"],
                      ["<b>인코더-디코더</b>", "인코더는 전부, 디코더는 앞만 + 입력 전체", "입력을 받아 다른 글 만들기", "번역, 요약", "T5, 처음 Transformer"]])
           + K.out("f_mask")
           + K.code_table([("torch.triu(..., diagonal=1)", "대각선 위쪽만 True. 미래 자리"), ("scores.masked_fill(causal, float(\"-inf\"))", "미래 자리 점수를 −∞로")]), "s4"),
    K.step(5, "실제 모델의 Attention 보기", "e5 모델에 문장을 넣고, 12개 층의 Attention 가중치를 꺼내 봐요.",
           "<p>Transformer는 ③의 Attention(과 작은 신경망)을 한 층으로 해서 여러 층 쌓은 모델이에요. 층마다 head 여러 개가 서로 다른 관계를 봐요.</p>"
           "<p>층 하나는 이렇게 생겼어요. ① 여러 head의 Attention으로 다른 토큰의 정보를 섞고, ② 입력을 그대로 더한 뒤(잔차 연결) LayerNorm(토큰 하나의 벡터 안에서 값들을 평균 0, 표준편차 1로 맞추기)으로 크기를 맞추고, ③ 토큰마다 따로 2층짜리 작은 신경망(보통 칸 수를 4배로 넓혔다가 줄여요)을 통과시키고, ④ 다시 잔차 연결과 LayerNorm을 해요. Attention은 \"토큰끼리 정보 나누기\", 작은 신경망은 \"토큰마다 생각하기\"를 맡아요. 잔차 연결과 정규화는 <a href=\"cnn.html\">CNN</a> ⑦의 ResNet과 같은 이유로 넣어요. 그래야 수십 층을 쌓아도 학습돼요.</p>"
           "<p>그림도 같은 방식으로 다룰 수 있어요. 그림을 16×16 조각으로 잘라 조각 하나를 토큰처럼 쓰면 <b>Vision Transformer</b>(ViT)예요. CNN의 \"가까운 픽셀끼리\" 가정이 없어서 데이터가 아주 많을 때 강해요.</p>"
           + K.code_table([("AutoModel.from_pretrained(MODEL, attn_implementation=\"eager\")", "가중치를 꺼낼 수 있는 계산 방식으로 모델 불러오기"), ("model(**enc, output_attentions=True)", "층마다 Attention 가중치도 돌려줘요."),
                           ("torch.stack(out.attentions)", "(층, 배치, head, 토큰, 토큰)"), ("a.argmax(-1)", "줄마다 가장 큰 가중치의 위치")])
           + K.try_("토큰이 512개, 4천 개, 13만 개면 Attention 점수표는 얼마나 클까?", "t5"), "s5"),
    K.step(6, "다음 토큰 고르기", "모델이 후보마다 준 점수를 softmax로 확률로 바꾸고, 그중 하나를 골라요.",
           K.code_table([("torch.softmax(logits / t, dim=-1)", "temperature t로 나눈 뒤 확률로"), ("logits.argmax()", "가장 높은 것 (greedy)"), ("torch.multinomial(probs, 1)", "확률대로 하나 뽑기 (sampling)")]), "s6"),
]
practice = [("Context Window", "모델이 한 번에 볼 수 있는 토큰 수의 한도예요. 대화가 길어지면 앞부분이 잘려서 \"잊어버린\" 것처럼 보여요. 그래서 필요한 자료만 골라 넣어요 (<a href=\"../backend/prompt-contract.html\">Prompt와 출력 계약</a>)."),
            ("한국어는 토큰이 많다", "같은 뜻이라도 영어보다 토큰이 많이 나오는 편이라 비용과 한도에 영향을 줘요. 실제 토큰 수는 그 모델의 Tokenizer로 세요."),
            ("환각", "모델은 \"사실인가\"가 아니라 \"그럴듯한가\"로 다음 토큰을 골라요. 근거 자료를 주고(RAG), 답을 검사하는 이유예요."),
            ("학습과 추론", "학습은 엄청난 글로 가중치를 정하는 일이고, 우리가 API로 쓰는 건 정해진 가중치로 다음 토큰을 계산하는 추론이에요."),
            ("긴 문맥이 느리고 비싸다", "Attention이 토큰 수의 제곱으로 늘어서예요(⑤의 🔁). 필요한 부분만 넣고(RAG), 같은 앞부분을 반복해서 보낼 때는 캐시를 쓰는 API 기능을 활용해요."),
            ("글 만들기와 검색, 어떤 모델?", "글을 만들려면 디코더(GPT식 LLM), 문장을 벡터로 바꿔 비슷한 걸 찾으려면 인코더(e5 같은 Embedding 모델)를 써요. 갈래가 달라서 서로 대신 쓰기 어려워요 (④의 표)."),
            ("큰 모델의 점수는 어디서 왔나", "사전학습은 다음 토큰(또는 가린 토큰) 맞히기 하나로 엄청난 글을 배우는 거예요. 지시를 따르는 대화형 LLM은 그 뒤에 질문과 좋은 답 쌍으로 더 배우고(미세조정), 사람의 선호로 다듬어요.")]
libs = [("①", "<code>AutoTokenizer</code>, <code>convert_ids_to_tokens()</code>", "Tokenizer (transformers)"), ("②⑤", "<code>AutoModel</code>, <code>word_embeddings</code>, <code>output_attentions=True</code>", "모델 안 들여다보기"),
        ("③④⑥", "<code>torch.softmax</code>, <code>torch.triu</code>, <code>masked_fill</code>, <code>torch.multinomial</code>", "Attention과 고르기 계산 (torch)")]
K.build(ROOT / "topics" / "model" / "llm-principles.html", title="LLM 원리", h1="LLM 원리: Token, Embedding, Attention, 다음 토큰", track="모델", crumb="모델 &gt; 순서와 언어 모델",
        one_line="LLM은 글을 <b>토큰</b>으로 나누고, <b>벡터</b>로 바꾸고, <b>Attention</b>으로 다른 토큰을 참고해 문맥을 섞은 뒤, <b>다음 토큰의 확률</b>을 계산해 하나씩 이어 붙여요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm_theory/llm_parts.py", practice=practice, libs=libs, libs_note="<code>looks_at</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"sequence-models.html\">순서 모델</a>", "RNN과 Attention의 출발점"), ("<a href=\"../backend/llm-client.html\">LLM API 클라이언트</a>", "실제 LLM 부르기"), ("<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>", "같은 e5 모델로 뜻 검색")],
        sources=["다루는 범위: Day20 (2026-11-13) LLM 생태계, Token, Context, Transformer, Attention, 학습과 추론",
                 "함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 12장(Transformer). √d로 나누는 이유, 여러 head, 위치 정보, 인코더, 디코더, 인코더-디코더와 사전학습 목표, Transformer 층의 구조, Vision Transformer, 긴 문맥의 비용을 가져왔어요. 위치 벡터, 순서 섞기, 가리기 모양 그림, 점수표 크기는 이 쪽에서 직접 계산했어요",
                 "모델: <code>intfloat/multilingual-e5-small</code>. ③의 Q, K, V 행렬과 ⑥의 점수는 설명용 예시 값. ①~⑥ 출력은 이 모델을 받아 둔 PC에서 실제로 실행한 값이에요",
                 "결과 값: <code>tools/pages/llm_principles.py</code>가 단계별로 실제 실행한 결과", "공식 문서: <a href=\"https://huggingface.co/docs/transformers/main_classes/tokenizer\" target=\"_blank\">Tokenizer</a>, <a href=\"https://arxiv.org/abs/1706.03762\" target=\"_blank\">Attention Is All You Need</a>"])
print("ok")
