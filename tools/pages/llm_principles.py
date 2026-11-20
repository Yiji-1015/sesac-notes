"""모델 > 순서와 언어 모델 > Transformer 구조. examples/llm_theory/llm_parts.py 하나가 ①~⑧로 자란다.
e5-small의 실제 가중치로 모델 안의 계산을 한 단계씩 다시 하고, 모델이 낸 값과 같은지 확인한다.
실행: HF_HUB_OFFLINE=1 python tools/pages/llm_principles.py
e5 모델을 받아 두지 않은 PC에서는 아무 파일도 바꾸지 않고 끝낸다 (기존 페이지를 그대로 둠).
"""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import html  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "llm_theory"
code = S.parse((EX / "llm_parts.py").read_text(encoding="utf-8"))
try:
    ns = S.run(code, 8, S.workdir(EX), filename="llm_parts.py")
except OSError:
    print("e5 모델(intfloat/multilingual-e5-small)이 없어서 이 쪽은 다시 만들지 않아요. 기존 파일을 그대로 둬요.")
    sys.exit(0)

from vstyle import plt, C, setup  # noqa: E402

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "llm"
IMG.mkdir(parents=True, exist_ok=True)
E = html.escape
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, intfloat/multilingual-e5-small, CPU"


def josa(word, with_final, without_final):
    """받침이 있으면 앞의 조사, 없으면 뒤의 조사. 숫자는 읽는 소리로 (0 영, 1 일, 3 삼, 6 육, 7 칠, 8 팔은 받침 있음)."""
    ch = str(word).strip(" '\".,")[-1]
    if "가" <= ch <= "힣":
        return with_final if (ord(ch) - 0xAC00) % 28 else without_final
    return with_final if ch in "013678" else without_final


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/llm/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


tk, BAE, PAIRS = ns["tokens"], ns["BAE"], ns["PAIRS"]
n0 = ns["n0"]
pair_name = {(0, 1): "몸-탈것", (0, 2): "몸-과일", (1, 2): "탈것-과일"}
pn = [pair_name[p] for p in PAIRS]
last = ns["bae_by_layer"][-1]
cos_txt = lambda vals: ", ".join(f3(v) for v in vals)  # noqa: E731
outs = {}

# ---------------- 왜
S.claim(all(abs(v - 1) < 1e-6 for v in ns["bae_table"]) and max(last) < 0.97, "표의 '배' 벡터는 같고, 12층 뒤에는 문장마다 달라진다")
outs["why"] = S.box("세 문장의 '▁배' 벡터끼리 코사인 유사도 (1이면 방향이 같음)", [f"짝 → {', '.join(pn)}", (f"표에서 찾아온 벡터 → {cos_txt(ns['bae_table'])}", 1, "bad"), (f"Transformer 12층을 지난 뒤 → {cos_txt(last)}", 2, "newf")],
                    ["토큰 번호로 표에서 찾아온 벡터는 문장과 상관없이 늘 같아요. \"배가 아파서\"의 배도, \"배를 타고\"의 배도 같은 벡터에서 출발해요.",
                     "12층을 지나면 같은 토큰이 문장마다 다른 벡터가 돼요. 주변 토큰의 정보가 섞였기 때문이에요. 이 쪽은 그 섞는 과정을 한 단계씩 따라가요."])

# ---------------- ①
outs["s1"] = S.box("① 토큰과 번호", [*(f"문장 {i + 1} → {t}" for i, t in enumerate(tk)), (f"ids.shape → {tuple(ns['ids'].shape)}", 1), (f"mask[1] → {ns['mask'][1].tolist()}", 2, "newf")],
                   ["<code>▁</code>는 \"앞에 띄어쓰기가 있던 조각\"이에요. 자주 나오는 말은 통째로, 드문 말은 잘게 나뉘어요. 세 문장 모두 1번 자리가 <code>▁배</code>예요. 0번 <code>&lt;s&gt;</code>와 끝의 <code>&lt;/s&gt;</code>는 시작과 끝 표시예요.",
                    f"세 문장의 토큰 수가 달라서({', '.join(str(len(t)) for t in tk)}개), 짧은 문장 뒤를 채워 (문장 수, 가장 긴 길이) 표 하나로 만들었어요. <code>mask</code>의 0이 채운 자리예요. <a href=\"sequence-models.html\">순서 모델</a> ①의 padding과 같은 방법이에요."])

# ---------------- ② 위치 벡터 그림 (학습된 위치 벡터)
P = ns["emb"].position_embeddings.weight.detach()[:64]
Pn = torch.nn.functional.normalize(P, dim=-1)
simP = (Pn @ Pn.T).numpy()
fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1.1, 1]})
im = a.imshow(simP, cmap="PuOr_r", vmin=-1, vmax=1)
a.set(xlabel="자리", ylabel="자리", title="e5가 배운 위치 벡터끼리의 코사인 유사도 (0~63번)")
a.grid(False)
fig.colorbar(im, ax=a, fraction=0.04)
b.plot(range(64), simP[10], color=C["purple"], lw=2.2)
b.axvline(10, color=C["gray"], ls="--", lw=1)
b.set(xlabel="자리", ylabel="10번 자리와의 코사인 유사도", title="가까운 자리일수록 비슷해요")
P_POS = fig_save(fig, "position_similarity")
near, far = float(np.mean([simP[10, 9], simP[10, 11]])), float(np.mean(simP[10, 40:64]))
S.claim(near > far + 0.2, "② 가까운 자리의 위치 벡터가 먼 자리보다 훨씬 비슷하다")
outs["f_pos"] = chart("e5의 위치 벡터 (학습으로 정해진 값)", P_POS, "",
                      [f"아무도 \"가까운 자리는 비슷하게\"라고 정해 주지 않았는데, 학습 결과 이웃한 자리끼리 비슷해졌어요. 10번 자리와 9, 11번은 평균 {near:.2f}, 40~63번과는 평균 {far:.2f}{josa(f'{far:.2f}', '이에요', '예요')}.",
                       "그래서 모델은 위치 벡터를 보고 \"몇 번째인가\"뿐 아니라 \"얼마나 떨어져 있나\"도 알 수 있어요."])
pe = ns["emb"].position_embeddings.weight
outs["s2"] = S.box("② Embedding", [(f"word.shape → {tuple(ns['word'].shape)}", 1), (f"position_embeddings → {tuple(pe.shape)}", 2), (f"x0와 모델의 emb(ids)가 같은가 → {ns['same_embed']}", 3, "newf"),
                                      f"표에서 찾은 '▁배' 벡터끼리 코사인 ({', '.join(pn)}) → {cos_txt(ns['bae_table'])}"],
                   [f"(문장 3, 토큰 {ns['ids'].shape[1]}, 384칸). 토큰마다 384칸 벡터 하나예요. 어휘가 {ns['emb'].word_embeddings.weight.shape[0]:,}개라 표의 크기는 {ns['emb'].word_embeddings.weight.shape[0]:,} × 384예요.",
                    f"자리마다 위치 벡터도 하나씩 있어요. 자리가 {pe.shape[0]}개뿐이라, 이 모델은 토큰을 {pe.shape[0]}개까지만 받을 수 있어요. 입력 길이 한도(Context Window)가 여기서 정해져요.",
                    "직접 더하고 LayerNorm한 값이 모델 안의 계산과 같아요. 모델이 하는 일을 그대로 따라 했다는 확인이에요."])

# ---------------- ③
w = ns["weights"]
look_tok = ns["looks"]
specials = {"<s>", "</s>", "."}
h_sel = next(i for i, t in enumerate(look_tok) if t not in specials)   # '▁배'가 내용 토큰을 가장 많이 본 head
row = w[0, h_sel, BAE, :n0]
top3 = torch.topk(row, 3)
top_tok = tk[0][int(top3.indices[0])]
S.claim(top_tok == "▁병원", "③ 고른 head에서 '배'가 가장 많이 본 토큰은 '병원'")
outs["s3"] = S.box(f"③ 첫 층의 Attention (head {h_sel})", [(f"Q.shape → {tuple(ns['Q'].shape)}", 1), f"weights.shape → {tuple(w.shape)}, 줄마다 합 → {[round(float(v), 3) for v in w[0, 0].sum(-1)[:3]]} …",
                                                         (f"손으로 계산한 weights와 모델의 attentions[0]이 같은가 → {ns['same_attn']}", 2, "newf"),
                                                         (f"문장 1, head {h_sel}에서 '▁배'가 많이 본 토큰 → " + ", ".join(f"{tk[0][int(j)]} {float(v):.2f}" for v, j in zip(top3.values, top3.indices)), 3)],
                   ["(문장 3, head 12, 토큰 12, 32칸). 384칸을 32칸씩 12조각으로 나눠 head마다 따로 계산해요 (④).",
                    "e5의 실제 Query, Key, Value 행렬로 계산한 가중치가 모델이 낸 가중치와 같아요.",
                    f"이 head에서는 '▁배'가 <code>{E(top_tok)}</code>{josa(top_tok, '을', '를')} 가장 많이 봤어요({float(top3.values[0]):.2f}). 다른 토큰보다 아주 크지는 않지만, '배'의 뜻을 정하는 데 쓸 만한 단어예요. 다른 head는 다른 곳을 봐요 (④)."])
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(w[0, h_sel, :n0, :n0].numpy(), cmap="Purples", vmin=0, vmax=1)
ax.set_xticks(range(n0), tk[0], rotation=60, fontsize=9)
ax.set_yticks(range(n0), tk[0], fontsize=9)
ax.set(xlabel="보이는 토큰 (Key)", ylabel="보는 토큰 (Query)", title=f"문장 1, 첫 층 head {h_sel}의 가중치")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04)
outs["f_head"] = chart("한 head의 Attention 가중치 (줄마다 합 1)", fig_save(fig, "attention_one_head"), "진할수록 많이 봄", ["줄 하나가 \"이 토큰이 새 벡터를 만들 때 어느 토큰을 몇 %씩 섞나\"예요."])
nos = S.run(code, 3, S.workdir(EX), [("/ math.sqrt(Dh) + pad", "+ pad")], filename="llm_parts.py")
sharp = lambda ww: float(ww[0, :, :n0, :n0].max(-1).values.mean())  # noqa: E731
s_yes, s_no = sharp(w), sharp(nos["weights"])
S.claim(s_no > s_yes, "√d로 나누지 않으면 가중치가 1등에 더 몰린다")
outs["t3"] = S.table("√d(= √32 ≈ 5.7)로 나누지 않으면 (문장 1, 첫 층 head 12개)", ["점수 계산", "줄마다 가장 큰 가중치의 평균", "모델과 같은가"],
                     [["<b>QKᵀ ÷ √32</b> (모델)", f3(s_yes), str(ns["same_attn"])], ["QKᵀ 그대로", f3(s_no), str(nos["same_attn"])]],
                     [f"나누지 않으면 점수 차이가 커져서 softmax가 1등 하나에 더 몰려요({s_yes:.2f} → {s_no:.2f}). 한 토큰만 보고 나머지는 거의 무시하게 돼요.",
                      "몰린 softmax는 기울기도 거의 0이라 학습이 잘 안 돼요. 칸 수 d가 클수록 내적이 커지니, √d로 나눠 크기를 맞춰요."])

# ---------------- ④
fig, axs = plt.subplots(2, 6, figsize=(15, 5.6))
for hh, ax in enumerate(axs.ravel()):
    ax.imshow(w[0, hh, :n0, :n0].numpy(), cmap="Purples", vmin=0, vmax=1)
    ax.set(title=f"head {hh}", xticks=[], yticks=[])
    ax.grid(False)
outs["f_heads"] = chart("첫 층 head 12개의 가중치 (문장 1, 줄: 보는 토큰, 칸: 보이는 토큰)", fig_save(fig, "attention_heads"), "",
                        ["head마다 무늬가 달라요. 대각선 바로 옆(앞이나 뒤 토큰)을 보는 head, 첫 칸이나 마지막 칸(<code>&lt;s&gt;</code>, <code>&lt;/s&gt;</code>)에 몰리는 head, 넓게 퍼진 head가 있어요.",
                         "시작과 끝 표시에 몰리는 건 \"딱히 볼 곳이 없을 때 쉬는 자리\"처럼 쓰는 거예요. 큰 모델에서도 흔히 보여요."])
n_distinct = len(set(look_tok))
S.claim(n_distinct >= 4 and h_sel is not None, "④ head마다 '배'가 많이 본 토큰이 다르다")
outs["s4"] = S.box("④ 여러 head", [(f"heads.shape → {tuple(ns['heads'].shape)}", 1), (f"concat.shape → {tuple(ns['concat'].shape)}", 2), f"attn_out.shape → {tuple(ns['attn_out'].shape)}",
                                    (f"head마다 문장 1의 '▁배'가 가장 많이 본 토큰 → {look_tok}", 3, "newf")],
                   ["head마다 32칸짜리 결과가 나와요.", "12개를 이어 붙이면 다시 384칸이에요. 그 뒤 384 × 384 행렬 하나로 한 번 더 섞어요(<code>attention.output.dense</code>).",
                    f"같은 '▁배'인데 head마다 본 곳이 {n_distinct}가지로 달라요. 한 head가 한 가지 관계만 잘 잡으니, 여러 벌을 두고 나눠 맡겨요."])

# ---------------- ⑤
L0 = ns["layer0"]
n_attn = sum(p.numel() for m in (L0.attention.self.query, L0.attention.self.key, L0.attention.self.value, L0.attention.output.dense) for p in m.parameters())
n_ffn = sum(p.numel() for m in (L0.intermediate.dense, L0.output.dense) for p in m.parameters())
diff1 = float((ns["x1"] - ns["out"].hidden_states[1]).abs().max())
S.claim(ns["same_layer"] and n_ffn > n_attn, "⑤ 손으로 만든 층이 모델과 같고, FFN 가중치가 Attention보다 많다")
outs["s5"] = S.box("⑤ 첫 층 전체", [(f"x1.shape → {tuple(ns['x1'].shape)}", 1), (f"모델의 hidden_states[1]과 같은가 → {ns['same_layer']} (가장 큰 차이 {diff1:.1e})", 2, "newf"),
                                    (f"층 하나의 가중치: Attention {n_attn:,}개, 작은 신경망 {n_ffn:,}개", 3)],
                   ["입력과 같은 모양이에요. 그래서 같은 층을 몇 번이고 쌓을 수 있어요 (⑥).", "Attention, 잔차 연결, LayerNorm, 작은 신경망까지 직접 이어 붙인 결과가 모델의 첫 층 출력과 같아요. 차이는 소수 계산의 반올림 수준이에요.",
                    f"Attention(Q, K, V, 출력 행렬 네 개)보다 토큰마다 통과하는 작은 신경망의 가중치가 {n_ffn / n_attn:.0f}배 많아요. 모델이 배운 지식의 상당 부분이 이 작은 신경망에 들어 있다고 봐요."])
enc_layer, word, emb = ns["encoder_layer"], ns["word"], ns["emb"]
perm = torch.tensor([0] + list(range(n0 - 2, 0, -1)) + [n0 - 1])   # 시작, 끝 표시는 두고 가운데를 거꾸로
pad1 = ns["pad"][0:1, :, :, :n0]
with torch.no_grad():
    typ = emb.token_type_embeddings(torch.zeros(1, n0, dtype=torch.long))
    no_pos = emb.LayerNorm(word[0:1, :n0] + typ)
    a_np = enc_layer(no_pos, L0, pad1)[0, perm]
    b_np = enc_layer(no_pos[:, perm], L0, pad1)[0]
    with_pos = emb.LayerNorm(word[0:1, :n0] + emb.position_embeddings(torch.arange(n0)) + typ)
    perm_pos = emb.LayerNorm(word[0:1, perm] + emb.position_embeddings(torch.arange(n0)) + typ)   # 섞은 순서에 위치를 새로 붙임
    a_p = enc_layer(with_pos, L0, pad1)[0, perm]
    b_p = enc_layer(perm_pos, L0, pad1)[0]
same_np, same_p = bool(torch.allclose(a_np, b_np, atol=1e-4)), bool(torch.allclose(a_p, b_p, atol=1e-4))
S.claim(same_np and not same_p, "위치 벡터가 없으면 순서를 섞어도 결과가 그대로 섞일 뿐이고, 있으면 달라진다")
sent1 = "".join(t.replace("▁", " ") for t in tk[0][1:-1]).strip()
outs["t5"] = S.table(f"문장 1의 가운데 토큰 순서를 거꾸로 해서 같은 층에 넣으면", ["입력", "결과가 \"순서만 바뀐 같은 값\"인가"],
                     [["토큰 벡터만", f"<b>{'예' if same_np else '아니오'}</b>"], ["토큰 벡터 + 위치 벡터 (②)", f"<b>{'예' if same_p else '아니오'}</b>"]],
                     [f"위치 벡터가 없으면 \"{sent1}\"{josa(sent1, '을', '를')} 거꾸로 넣어도 토큰마다 같은 벡터가 나와요. Attention과 작은 신경망 어디에도 \"몇 번째\"가 없어서, 층은 토큰들을 순서 없는 집합으로 봐요.",
                      "위치 벡터를 더하면 같은 토큰도 자리에 따라 다른 입력이 돼서, 순서가 결과에 들어가요. ②에서 위치를 더하는 이유예요."])

# ---------------- ⑥
by = np.array(ns["bae_by_layer"])
fig, ax = plt.subplots(figsize=(8.2, 3.9))
for j, (nm, col) in enumerate(zip(pn, [C["purple"], C["pink"], C["mint"]])):
    ax.plot(range(len(by)), by[:, j], marker="o", ms=4, lw=2.2, color=col, label=nm)
ax.set(xlabel="지난 층 수 (0 = ②의 벡터)", ylabel="'배' 벡터끼리 코사인 유사도", title="층을 지나며 같은 토큰이 문장마다 달라져요", xticks=range(len(by)))
ax.legend(fontsize=9)
P_LAY = fig_save(fig, "bae_by_layer")
low_layer = int(by.min(1).argmin())
outs["f_layers"] = chart("세 문장의 '▁배' 벡터, 층마다", P_LAY, "", [f"0층(표에서 찾은 벡터 + 위치)에서는 셋 다 1이에요. 층을 지나며 내려가서(가장 낮은 곳은 {low_layer}층), 12층에서는 {cos_txt(last)}{josa(cos_txt(last), '이에요', '예요')}.",
                                                                    "벡터가 달라졌다는 것까지가 이 그림이 보여 주는 거예요. 그 차이가 \"몸, 탈것, 과일\"을 잘 가르는지는 따로 재야 해요. 그 벡터 위에 분류용 층을 얹어 학습시키는 게 미세조정이에요."])
diffL = float((ns["h"] - ns["out"].last_hidden_state).abs().max())
outs["s6"] = S.box("⑥ 12층", [(f"모델의 last_hidden_state와 같은가 → {ns['same_model']} (가장 큰 차이 {diffL:.1e})", 1, "newf"), *(f"{k}층 뒤 '▁배' 코사인 ({', '.join(pn)}) → {cos_txt(ns['bae_by_layer'][k])}" for k in (0, 1, 6, 12))],
                   ["⑤의 층 함수를 12번 부른 결과가 모델의 마지막 출력과 같아요. Transformer 인코더는 결국 이 층의 반복이에요.", "층마다 주변 토큰의 정보가 조금씩 더 섞여요."])
# 🔁 길이와 비용: 실제 시간 재기
torch.manual_seed(0)
sa0 = L0.attention.self
cost = []
for n_tok in (256, 1024, 2048):
    xx = torch.randn(1, n_tok, 384)
    zero = torch.zeros(1, 1, 1, n_tok)

    def attn_part():
        Q_, K_, V_ = ns["split"](sa0.query(xx)), ns["split"](sa0.key(xx)), ns["split"](sa0.value(xx))
        return L0.attention.output.dense(((Q_ @ K_.transpose(-1, -2) / 32 ** 0.5 + zero).softmax(-1) @ V_).transpose(1, 2).reshape(xx.shape))

    def ffn_part():
        return L0.output.dense(torch.nn.functional.gelu(L0.intermediate.dense(xx)))

    t_ = {}
    with torch.no_grad():
        for nm, fn in (("attn", attn_part), ("ffn", ffn_part)):
            fn()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            t_[nm] = min(ts) * 1000
    cost.append((n_tok, t_["attn"], t_["ffn"], 12 * n_tok * n_tok * 4 / 2 ** 20))
share = [a_ / (a_ + f_) for _, a_, f_, _ in cost]
S.claim(share[-1] > share[0], "토큰이 길수록 시간 중 Attention의 몫이 커진다")
outs["t6"] = S.table("토큰 수를 늘리며 첫 층 하나의 시간 재기 (CPU, 세 번 중 가장 빠른 값)", ["토큰 수", "Attention", "작은 신경망", "Attention의 몫", "점수표 메모리 (head 12개)"],
                     [[f"{n:,}", f"{a_:.1f} ms", f"{f_:.1f} ms", f"<b>{s_:.0%}</b>", f"{mb:,.0f} MB"] for (n, a_, f_, mb), s_ in zip(cost, share)],
                     [f"작은 신경망은 토큰마다 따로 계산해서 토큰 수에 비례해 늘어요. Attention은 모든 토큰 쌍의 점수표를 만들어서 토큰 수의 제곱으로 늘어요. 그래서 짧을 때는 작은 신경망이, 길어질수록 Attention이 시간을 차지해요({share[0]:.0%} → {share[-1]:.0%}).",
                      "점수표는 토큰이 4배가 되면 16배가 돼요. 이 모델은 위치 벡터가 512개라 실제로는 512토큰까지만 받지만, 긴 문맥을 다루는 모델에서는 이게 가장 큰 비용이에요. 시간은 PC마다 달라요."])

# ---------------- ⑦
with torch.no_grad():
    pos2_bi = ns["cos"](ns["h"][1, 2], ns["h"][2, 2])
    pos2_ca = ns["cos"](ns["hc"][1, 2], ns["hc"][2, 2])
S.claim(all(abs(v - 1) < 1e-5 for v in ns["bae_causal"]) and abs(pos2_ca - 1) < 1e-5 and pos2_bi < 0.99, "⑦ 미래를 가리면 '배'와 문장 2, 3의 '를'은 끝까지 같다")
outs["s7"] = S.box("⑦ 미래를 가리고 12층", [(f"'▁배' 코사인 ({', '.join(pn)}) → {cos_txt(ns['bae_causal'])}", 1, "bad"),
                                          (f"문장 2, 3의 2번 자리 '를' → 가리면 {pos2_ca:.3f}, 안 가리면 {pos2_bi:.3f}", 2)],
                   ["뒤를 못 보니 '▁배'는 앞의 <code>&lt;s&gt;</code>와 자기 자신만 봐요. 세 문장 모두 앞부분이 같아서 12층을 지나도 벡터가 똑같아요. \"아파서\"인지 \"타고\"인지 아직 모르는 거예요.",
                    "문장 2, 3은 \"배를\"까지 같아서 '를'도 똑같아요. 디코더는 뒤 단어로 앞 단어의 뜻을 고칠 수 없어서, 다음 토큰을 맞힐 때마다 앞까지의 정보로만 판단해요. 인코더는 앞뒤를 다 봐서 문장 이해에 강하고, 디코더는 글을 앞에서부터 만드는 데 맞아요."])
# 세 가지 가리기 모양 그림
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
n_ = 6
for ax, (m_, ttl, xl, yl) in zip(axs, [(np.ones((n_, n_)), "인코더 (BERT, e5)\n앞뒤를 다 봐요", "보이는 토큰", "보는 토큰"), (np.tril(np.ones((n_, n_))), "디코더 (GPT)\n자기와 앞만 봐요", "보이는 토큰", "보는 토큰"),
                                      (np.ones((4, n_)), "인코더-디코더의 교차 Attention (T5)\n만드는 쪽이 입력 전체를 봐요", "입력 문장의 토큰", "만드는 문장의 토큰")]):
    ax.imshow(m_, cmap="Purples", vmin=0, vmax=1.3)
    ax.set(title=ttl, xlabel=xl, ylabel=yl, xticks=range(m_.shape[1]), yticks=range(m_.shape[0]))
    ax.grid(False)
outs["f_mask"] = chart("세 갈래의 Transformer가 \"누가 누구를 보나\"", fig_save(fig, "attention_masks"), "진한 칸: 볼 수 있음, 흰 칸: 가림",
                       ["⑦에서 더한 미래 가리기가 가운데 모양이에요. 같은 층에 가리는 모양만 바꿔서 세 갈래가 나와요."])

# ---------------- ⑧
pr = ns["probs"]
top = int(ns["logits"].argmax())
outs["s8"] = S.table("⑧ 빈칸 후보 네 개의 확률 (점수는 예시)", ["후보", "점수(logit)"] + [f"temperature {t}" for t in pr],
                     [[E(c), f"{float(ns['logits'][i]):.1f}"] + [f"{float(pr[t][i]):.2f}" for t in pr] for i, c in enumerate(ns["candidates"])],
                     [f"가장 높은 것만 고르면(greedy) 언제나 \"{ns['greedy']}\"{josa(ns['greedy'], '이에요', '예요')}.", f"확률대로 뽑으면(temperature 1.0, 10번) → {ns['sampled']}. 같은 질문에도 답이 조금씩 달라지는 이유예요.",
                      f"temperature가 낮으면 1등에 몰리고(0.5에서 {float(pr[0.5][top]):.2f}), 높으면 고르게 퍼져요(2.0에서 {float(pr[2.0][top]):.2f}). 정확해야 하면 낮게, 다양해야 하면 높게 써요."])

flow = [[1, "토큰", "세 문장, 패딩", "▁배는 1번 자리"], [2, "Embedding", "표 + 위치", "'배'는 아직 같음"], [3, "head 하나", "Q, K, V, √d", "모델과 같음"], [4, "여러 head", "12 × 32칸", "보는 곳이 다름"],
        [5, "층 하나", "잔차, LayerNorm, FFN", "모델과 같음"], [6, "12층", "층 반복", "'배'가 달라짐"], [7, "미래 가리기", "디코더", "'배'가 그대로"], [8, "다음 토큰", "softmax", "temperature"]]
S.save(ROOT / "topics" / "model" / "llm-principles-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>글을 다루는 모델은 두 가지를 해내야 해요. 토큰마다 <b>주변 문맥을 반영한 벡터</b>를 만드는 것, 그리고 그걸 <b>빠르게</b> 하는 것이에요.</p>
<p>예전 방식은 <a href="sequence-models.html">RNN</a>이었어요. 번역이라면 인코더 RNN이 문장을 앞에서부터 읽어 <b>벡터 하나</b>에 눌러 담고, 디코더 RNN이 그 벡터에서 번역문을 만들었어요. 긴 문장도 벡터 하나에 담으니 앞부분 정보가 흐려지고(병목), 한 토큰씩 차례로 읽어야 해서 느렸어요. 그래서 디코더가 단어를 만들 때마다 인코더의 <b>모든</b> 토큰을 다시 보고 관련 있는 것에 가중치를 주는 <b>Attention</b>이 붙었어요. Transformer는 여기서 RNN을 아예 빼고, 토큰들이 서로를 Attention으로 보게 해서 모든 토큰을 한꺼번에 계산해요.</p>
<p>이 쪽의 예제는 같은 \"배\"가 세 가지 뜻으로 쓰인 세 문장이에요.</p>
{K.table(["문장", "'배'의 뜻"], [[E(t), m_] for t, m_ in zip(ns["texts"], ["몸", "탈것", "과일"])])}
<p>이 PC에 있는 실제 Transformer(<code>multilingual-e5-small</code>, 12층)에 넣고, 세 문장의 '배' 벡터를 비교했어요.</p>
{K.out("why")}"""
whole = """<p>글 → 토큰 → 토큰 벡터 + 위치 → (Attention → 잔차와 LayerNorm → 작은 신경망 → 잔차와 LayerNorm) × 12층 → 문맥이 섞인 벡터. 같은 층에 미래 가리기를 더하면 디코더가 되고, 디코더는 마지막 벡터로 다음 토큰을 골라요. ③⑤⑥에서는 모델 안의 계산을 직접 해 보고, 모델이 낸 값과 같은지 확인해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "글을 토큰으로", "Tokenizer가 글을 모델이 아는 조각(토큰)으로 나누고 번호를 붙여요.",
           "<p><b>왜 조각으로?</b> 글자 하나씩 나누면 토큰이 너무 많아지고 글자 하나에는 뜻이 거의 없어요. 단어 단위로 나누면 어휘가 끝없이 늘고, 처음 보는 단어(신조어, 오타)는 아예 다룰 수 없어요. 그래서 자주 나오는 말은 통째로, 드문 말은 잘게 쪼개는 <b>부분단어</b> 방식을 써요. e5는 SentencePiece라는 Tokenizer로 25만 개 어휘를 만들었어요.</p>"
           + K.code_table([("AutoTokenizer.from_pretrained(MODEL)", "그 모델과 짝인 Tokenizer"), ("tok(texts, return_tensors=\"pt\", padding=True)", "여러 문장을 번호 Tensor 하나로. 짧은 문장 뒤는 채우고 <code>attention_mask</code>에 0으로 표시"),
                           ("convert_ids_to_tokens(row[m.bool()])", "채운 자리를 빼고 번호를 조각 글자로")]), "s1"),
    K.step(2, "토큰 벡터에 위치 더하기", "토큰 번호로 표에서 벡터를 찾고, 자리마다 정해진 위치 벡터를 더한 뒤 LayerNorm으로 크기를 맞춰요.",
           "<p><b>왜 위치를 더하나?</b> 다음 단계의 Attention은 토큰들을 순서 없이 봐요(⑤의 🔁에서 확인). \"개가 사람을 물었다\"와 \"사람이 개를 물었다\"를 가르려면 자리 정보를 벡터에 넣어 줘야 해요. 처음 Transformer는 사인, 코사인으로 만든 고정 벡터를, BERT 계열인 e5는 <b>학습한 위치 벡터</b>를 더해요. 요즘 LLM은 Query와 Key를 자리에 따라 돌려서 \"얼마나 떨어졌나\"를 넣는 RoPE를 많이 써요.</p>"
           "<p><b>LayerNorm</b>은 토큰 벡터 하나 안의 384개 값을 평균 0, 표준편차 1로 맞춘 뒤, 학습한 배율과 이동값을 곱하고 더하는 거예요. 값의 크기가 층마다 들쭉날쭉하지 않게 해요.</p>"
           + K.out("f_pos")
           + K.code_table([("emb.word_embeddings(ids)", "토큰 번호 → 384칸 벡터 (<code>nn.Embedding</code>, 표에서 줄 찾기)"), ("emb.position_embeddings(pos)", "자리 번호 → 384칸 위치 벡터"),
                           ("emb.token_type_embeddings(...)", "문장 두 개를 이어 넣을 때 앞뒤를 가르는 벡터. 여기서는 모두 0번"), ("torch.allclose(x0, emb(ids), atol=1e-5)", "두 Tensor가 아주 작은 차이 안에서 같은가")]), "s2"),
    K.step(3, "head 하나의 Attention", "토큰마다 Query, Key, Value를 만들고, Query와 Key가 잘 맞을수록 그 토큰의 Value를 많이 섞어요. e5 첫 층의 실제 행렬로 계산해요.",
           "<p>도서관에 빗대면 <b>Query</b>는 \"내가 찾는 것\", <b>Key</b>는 \"책 등의 제목\", <b>Value</b>는 \"책 내용\"이에요. 같은 토큰 벡터를 학습한 세 행렬에 곱해 세 역할을 만들어요. \"잘 맞음\"은 Query와 Key의 내적(칸마다 곱해 더한 값)으로 재고, softmax로 줄마다 합이 1인 가중치로 바꾼 뒤, 그 가중치로 Value들을 섞어요. 식으로 쓰면 <code>softmax(QKᵀ ÷ √d) V</code>예요.</p>"
           "<p><b>왜 √d로 나누나?</b> 칸 수 d가 크면 내적은 d개를 더한 값이라 커져요. 그러면 softmax가 1등 하나에 거의 다 몰리고 기울기가 사라져요. 아래 🔁에서 실제로 비교해요. 채운 자리(<code>mask</code> 0)의 점수에는 아주 큰 음수를 더해서, softmax 뒤 가중치가 0이 되게 해요.</p>"
           + K.out("f_head")
           + K.code_table([("sa.query(x0)", "토큰 벡터에 Query 행렬을 곱하기 (<code>nn.Linear</code>, 384 → 384)"), ("split(t)", "384칸을 head 12개 × 32칸으로 나누기 (<code>view</code>, <code>transpose</code>)"),
                           ("Q @ K.transpose(-1, -2) / math.sqrt(Dh)", "모든 토큰 쌍의 맞음 점수"), ("(1 - mask[:, None, None, :]) * -1e9", "채운 자리만 아주 큰 음수"),
                           ("model(**enc, output_attentions=True, output_hidden_states=True)", "모델이 층마다 낸 가중치와 벡터도 같이 받기 (비교용)")])
           + K.try_("√d로 나누지 않으면?", "t3"), "s3"),
    K.step(4, "여러 head를 이어 붙이기", "head 12개가 각자 섞은 32칸 결과를 이어 붙여 384칸으로 되돌리고, 행렬 하나로 한 번 더 섞어요.",
           "<p><b>왜 여러 head?</b> Attention 한 번은 가중치 줄 하나로 섞으니, 한 가지 관계만 잘 잡아요. 그래서 384칸을 12조각으로 나눠 head마다 다른 Q, K, V로 따로 보게 해요. 어떤 head는 바로 앞 토큰을, 어떤 head는 문장 끝을, 어떤 head는 뜻이 이어진 단어를 봐요. 계산량은 head 하나로 384칸 전체를 볼 때와 같아요.</p>"
           + K.out("f_heads")
           + K.code_table([("weights @ V", "head마다 가중치로 Value 섞기"), ("heads.transpose(1, 2).reshape(x0.shape)", "(문장, head, 토큰, 32) → (문장, 토큰, 384)로 이어 붙이기"),
                           ("layer0.attention.output.dense(concat)", "이어 붙인 결과를 384 × 384 행렬로 한 번 더 섞기")]), "s4"),
    K.step(5, "층 하나 완성: 잔차 연결, LayerNorm, 작은 신경망", "Attention 결과를 입력에 더하고(잔차) LayerNorm, 토큰마다 작은 신경망을 통과시키고 다시 잔차와 LayerNorm을 해요. 이걸 함수 하나로 묶어요.",
           "<p>층은 두 부분이에요. <b>Attention</b>은 \"토큰끼리 정보 나누기\", <b>작은 신경망</b>(Feed Forward, 384 → 1536 → 384, 가운데 GELU)은 \"토큰마다 따로 생각하기\"예요. GELU는 ReLU처럼 음수를 줄이지만 0 근처에서 부드럽게 꺾이는 활성화 함수예요.</p>"
           "<p><b>왜 잔차 연결?</b> 층의 결과를 입력에 <b>더해서</b> 내보내면, 층은 입력을 통째로 새로 만들 필요 없이 \"얼마나 고칠지\"만 배우면 되고, 기울기도 더하기 길로 앞 층까지 잘 내려가요. <a href=\"cnn.html\">CNN</a> ⑦의 ResNet과 같은 이유예요. e5와 BERT는 더한 <b>뒤에</b> LayerNorm을 하고(Post-LN), GPT-2 이후의 LLM 대부분은 Attention과 작은 신경망에 들어가기 <b>전에</b> 해요(Pre-LN). 층을 아주 깊게 쌓을 때 학습이 더 안정적이라서요.</p>"
           "<p>함수의 <code>block</code>은 \"보면 안 되는 자리에 더할 점수\"예요. ③의 <code>pad</code>를 넣으면 채운 자리를 안 보고, ⑦에서는 여기에 미래 가리기를 더해 넣어요.</p>"
           + K.code_table([("def encoder_layer(x, layer, block)", "③④의 계산에 잔차, LayerNorm, 작은 신경망을 붙인 층 하나"), ("layer.attention.output.LayerNorm(x + a)", "잔차 연결 뒤 LayerNorm"),
                           ("F.gelu(layer.intermediate.dense(h))", "384 → 1536으로 넓히고 GELU"), ("layer.output.dense(...)", "1536 → 384로 되돌리기")])
           + K.try_("토큰 순서를 섞어서 넣으면 결과가 달라질까?", "t5"), "s5"),
    K.step(6, "12층 쌓기: 문맥이 섞인 벡터", "⑤의 층 함수를 e5의 12개 층 가중치로 차례로 불러요. 층마다 세 문장의 '▁배' 벡터를 비교해요.",
           "<p>층은 입력과 출력의 모양이 같아서 몇 번이고 쌓을 수 있어요. 층마다 가중치는 따로예요. 층을 지날 때마다 각 토큰 벡터에 주변 토큰의 정보가 더 섞여서, 마지막에는 \"이 문장 안에서의 이 토큰\"을 나타내는 벡터가 돼요. 이걸 <b>문맥이 섞인 Embedding</b>이라고 해요. 이 벡터들을 평균 내면 문장 하나의 벡터가 되고(<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>), 첫 토큰의 벡터 위에 작은 층을 얹으면 문장 분류기가 돼요.</p>"
           "<p>그림도 같은 방식으로 다룰 수 있어요. 그림을 16×16 조각으로 잘라 조각 하나를 토큰처럼 넣으면 <b>Vision Transformer</b>(ViT)예요.</p>"
           + K.out("f_layers")
           + K.code_table([("for layer in model.encoder.layer:", "e5의 12개 층을 차례로"), ("torch.allclose(h, out.last_hidden_state, atol=1e-3)", "12층을 다 지난 결과가 모델의 마지막 출력과 같은가")])
           + K.try_("토큰이 길어지면 어디가 느려질까?", "t6"), "s6"),
    K.step(7, "미래 가리기: 디코더는 뒤를 못 봐요", "점수표에서 자기보다 뒤의 자리를 모두 가리고, 같은 12층을 다시 통과시켜요.",
           "<p>글을 앞에서부터 한 토큰씩 만드는 모델은, 학습할 때 정답(뒤 토큰)을 미리 보면 안 돼요. 그래서 자기보다 뒤의 점수에 아주 큰 음수를 더해 가중치를 0으로 만들어요. 계산은 ⑤의 층 그대로이고, <code>block</code>에 가리기만 더했어요.</p>"
           "<p>가리는 모양에 따라 Transformer는 세 갈래로 나뉘고, 미리 배우는 방법(사전학습 목표)도 달라요.</p>"
           + K.table(["갈래", "보는 범위", "미리 배우는 방법", "잘하는 일", "예"],
                     [["<b>인코더</b>", "앞뒤 전부", "문장 중간 토큰을 가리고 맞히기 (MLM)", "문장 이해: 분류, 개체명 찾기, 질문의 답 찾기, 문장 벡터", "BERT, RoBERTa, ELECTRA, e5"],
                      ["<b>디코더</b>", "자기와 앞만", "다음 토큰 맞히기 (CLM)", "글 만들기, 대화", "GPT 계열, 대부분의 LLM"],
                      ["<b>인코더-디코더</b>", "인코더는 전부, 디코더는 앞만 + 인코더 출력 전체", "입력을 받아 다른 글 만들기, 망가뜨린 글 되살리기", "번역, 요약", "T5, BART, 처음 Transformer"]])
           + "<p>인코더-디코더의 디코더에는 층마다 Attention이 하나 더 있어요. Query는 만드는 문장에서, Key와 Value는 인코더 출력에서 가져오는 <b>교차 Attention</b>이에요. RNN 번역기의 \"디코더가 인코더 전체를 다시 본다\"가 여기에 남아 있어요.</p>"
           + K.out("f_mask")
           + K.code_table([("torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)", "대각선 위만 True. 자기보다 뒤의 자리"), ("pad + future * -1e9", "채운 자리와 미래 자리 모두 아주 큰 음수")]), "s7"),
    K.step(8, "다음 토큰 고르기 (점수 → 확률)", "디코더는 마지막 벡터를 어휘 수만큼의 점수로 바꾸고, softmax로 확률을 만든 뒤 하나를 골라요.",
           "<p>디코더의 마지막 층 위에는 \"384칸 → 어휘 수\" 행렬이 하나 있어서, 다음 토큰 후보마다 점수(logit)를 내요. 고른 토큰을 입력 끝에 붙이고 다시 계산하기를 반복하면 글이 돼요. e5는 인코더라 이 행렬이 없어서, 여기서는 예시 점수로 고르는 부분만 계산해요.</p>"
           "<p><b>temperature</b>는 점수를 나누는 값이에요. 작으면 차이가 커져 1등에 몰리고, 크면 고르게 퍼져요.</p>"
           + K.code_table([("torch.softmax(logits / t, dim=-1)", "temperature t로 나눈 뒤 확률로"), ("logits.argmax()", "가장 높은 것 (greedy)"), ("torch.multinomial(probs, 1)", "확률대로 하나 뽑기 (sampling)")]), "s8"),
]
practice = [("Hugging Face 라이브러리", "<code>transformers</code>(모델), <code>tokenizers</code>(Tokenizer), <code>datasets</code>(데이터 읽기, 전처리), <code>accelerate</code>(여러 GPU 학습)가 짝으로 쓰여요. 모델과 데이터는 Hub에서 이름으로 받아요. 결과만 필요하면 <code>pipeline(\"text-classification\")</code>처럼 한 줄로도 써요."),
            ("Context Window", "모델이 한 번에 볼 수 있는 토큰 수의 한도예요. 학습한 위치 벡터를 쓰는 모델은 그 개수(e5는 512)가 한도예요. 넘으면 잘라야 해서, 필요한 자료만 골라 넣어요 (<a href=\"../backend/prompt-contract.html\">Prompt와 출력 계약</a>)."),
            ("한국어는 토큰이 많다", "같은 뜻이라도 영어보다 토큰이 많이 나오는 편이라 비용과 한도에 영향을 줘요. 실제 토큰 수는 그 모델의 Tokenizer로 세요."),
            ("긴 문맥이 느리고 비싸다", "점수표가 토큰 수의 제곱으로 늘어서예요(⑥의 🔁). 점수표를 조각씩 계산하거나(FlashAttention), 가까운 토큰과 몇몇 전역 토큰만 보거나(희소 Attention, Longformer), 점수표 없이 선형으로 근사하거나(선형 Attention), 앞 토큰들의 Key와 Value를 저장해 다시 써요(KV 캐시)."),
            ("모델을 키우면 얼마나 좋아지나", "모델 크기, 데이터 양, 계산량을 늘리면 손실이 거듭제곱 꼴로 꾸준히 줄어든다는 관찰이 있어요(스케일링 법칙). 셋 중 하나만 키우면 금방 한계에 닿아서, 함께 키워요."),
            ("글이 아닌 데이터", "그림은 조각을 토큰으로(ViT), 소리는 짧은 구간을 토큰으로(wav2vec 2.0), 문서는 글자와 위치를 함께(LayoutLM) 넣어요. 그림과 글을 같은 벡터 공간에 맞춰 배우면(CLIP) 글로 그림을 찾거나 처음 보는 종류도 분류할 수 있어요."),
            ("환각", "디코더는 \"사실인가\"가 아니라 \"그럴듯한가\"로 다음 토큰을 골라요. 근거 자료를 주고(RAG), 답을 검사하는 이유예요."),
            ("글 만들기와 검색, 어떤 모델?", "글을 만들려면 디코더, 문장을 벡터로 바꿔 비슷한 걸 찾거나 분류하려면 인코더를 써요(⑦의 표).")]
libs = [("①", "<code>AutoTokenizer</code>, <code>padding=True</code>, <code>convert_ids_to_tokens()</code>", "Tokenizer (transformers)"),
        ("②", "<code>AutoModel</code>, <code>word_embeddings</code>, <code>position_embeddings</code>, <code>LayerNorm</code>", "모델의 Embedding 부분"),
        ("③④⑤", "<code>query</code>, <code>key</code>, <code>value</code>, <code>attention.output.dense</code>, <code>intermediate.dense</code>, <code>F.gelu</code>", "층 하나의 실제 가중치"),
        ("③⑥", "<code>output_attentions=True</code>, <code>output_hidden_states=True</code>, <code>torch.allclose</code>", "모델 값과 비교"),
        ("⑦⑧", "<code>torch.triu</code>, <code>torch.softmax</code>, <code>torch.multinomial</code>", "가리기와 고르기 (torch)")]
K.build(ROOT / "topics" / "model" / "llm-principles.html", title="Transformer 구조", h1="Transformer 구조: 토큰 벡터에 문맥이 섞이기까지", track="모델", crumb="모델 &gt; 순서와 언어 모델",
        one_line="Transformer는 토큰 벡터에 위치를 더한 뒤, <b>Attention</b>(토큰끼리 정보 나누기)과 <b>작은 신경망</b>(토큰마다 생각하기)을 잔차 연결과 LayerNorm으로 묶은 층을 여러 번 쌓아요. 뒤를 가리면 글을 만드는 디코더가 돼요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm_theory/llm_parts.py", practice=practice, libs=libs,
        libs_note="<code>cos</code>, <code>split</code>, <code>encoder_layer</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"sequence-models.html\">순서 모델</a>", "RNN과 Attention의 출발점"), ("<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>", "같은 e5로 문장 벡터 만들기"),
                   ("<a href=\"../backend/llm-client.html\">LLM API 클라이언트</a>", "실제 디코더 LLM 부르기")],
        sources=["다루는 범위: Day20 (2026-11-13) LLM 생태계, Token, Context, Transformer, Attention, 학습과 추론",
                 "함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 12장(Transformer). Lewis Tunstall 외, <i>Natural Language Processing with Transformers</i> 1장(인코더-디코더와 Attention의 출발, Hugging Face 생태계), 3장(Scaled Dot-Product Attention, 여러 head, Feed Forward, Post-LN과 Pre-LN, 위치 Embedding, 디코더와 교차 Attention, 세 갈래 모델), 11장(스케일링 법칙, 희소 Attention과 선형 Attention, ViT, CLIP, wav2vec 2.0, LayoutLM)",
                 "모델: <code>intfloat/multilingual-e5-small</code>. ⑧의 점수는 설명용 예시 값이고, 나머지 출력과 그림은 이 모델로 실제로 계산한 값이에요",
                 "결과 값과 그림: <code>tools/pages/llm_principles.py</code>가 <code>examples/llm_theory/llm_parts.py</code>를 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://huggingface.co/docs/transformers/main_classes/tokenizer\" target=\"_blank\">Tokenizer</a>, <a href=\"https://huggingface.co/docs/transformers/model_doc/bert\" target=\"_blank\">BERT 구조</a>, <a href=\"https://arxiv.org/abs/1706.03762\" target=\"_blank\">Attention Is All You Need</a>"])
print("ok")
