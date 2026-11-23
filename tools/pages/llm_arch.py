"""모델 > 순서 모델과 Transformer > 요즘 LLM의 부품. examples/llm_arch/modern_llm.py 하나가 ①~⑧로 자란다.
모델은 받지 않고 설정(config)만으로 무작위 가중치 모델을 만들어 모양, 크기, 계산 방식을 본다 (CPU, 1분 안쪽).
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/llm_arch.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "llm_arch"
code = S.parse((EX / "modern_llm.py").read_text(encoding="utf-8"))
ns = S.run(code, 8, S.workdir(EX), filename="modern_llm.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, CPU 스레드 4개, 가중치는 무작위(seed 0)"
J = K.josa

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "llm_arch"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/llm_arch/{name}.png"


outs = {}
P, T = ns["parts"], ns["total"]
g0, l0 = ns["g0"], ns["l0"]

# ---------------- ①
rows = [[k, f"{P['GPT-2'][k]:,}", f"{P['Llama'][k]:,}"] for k in P["GPT-2"]] + [["<b>모두</b>", f"<b>{T['GPT-2']:,}</b>", f"<b>{T['Llama']:,}</b>"]]
diff = T["Llama"] - T["GPT-2"]
lm, wpe = P["Llama"]["출력 층"], P["GPT-2"]["위치 표"]
S.claim(abs(P["GPT-2"]["Attention (층마다)"] / P["Llama"]["Attention (층마다)"] - 1) < 0.01 and abs(P["GPT-2"]["작은 신경망 (층마다)"] / P["Llama"]["작은 신경망 (층마다)"] - 1) < 0.01,
        "① Attention과 작은 신경망은 1% 안쪽으로 같은 크기")
S.claim(P["Llama"]["위치 표"] == 0 and P["GPT-2"]["출력 층"] == 0 and abs(diff - (lm - wpe)) < 0.01 * diff, "① 차이는 거의 출력 층과 위치 표에서")
outs["s1"] = S.table("① 가중치 수 (벡터 256칸, 4층, head 8개, 어휘 8,000)", ["부품", "GPT-2", "Llama"], rows,
                     [f"Attention과 작은 신경망은 1% 안쪽으로 같은 크기예요. 같은 크기로 맞춰서, 남은 차이는 부품의 모양뿐이에요.",
                      f"Llama에는 위치 표가 없어요. 위치를 표로 더하지 않고 Query와 Key를 돌려서 넣어요(④).",
                      f"Llama가 {diff:,}개 더 많은 건 출력 층({lm:,}개) 때문이에요. GPT-2는 출력 층이 Token Embedding 표를 같이 쓰고(tied), 대신 위치 표({wpe:,}개)가 있어요. 요즘 큰 모델은 대부분 따로 두고, 작은 모델은 같이 쓰기도 해요."])

# ---------------- ②
x = ns["x"]
gap_lr = (ns["ln"] - ns["rms"]).abs().max().item()
gap_hand = (ns["rms"] - ns["rms_by_hand"]).abs().max().item()
gap_c = (ns["ln"] - ns["rms_centered"]).abs().max().item()
eps_ln, eps_rms = ns["gpt2"].config.layer_norm_epsilon, ns["llama"].config.rms_norm_eps
S.claim(gap_lr > 1 and gap_hand < 1e-5 and gap_c < 1e-2 and gap_c < gap_lr / 100, "② 평균이 0이 아니면 크게 다르고, 손 계산은 같고, 평균을 빼면 거의 같다")
outs["s2"] = S.box(f"② 토큰 벡터 3개 (칸 평균 약 {x.mean().item():.1f})에 두 정규화", [(f"LayerNorm vs RMSNorm → 가장 큰 차이 {gap_lr:.2f}", 1),
                                                                         (f"RMSNorm 손 계산 x ÷ √(평균(x²) + ε) vs 모듈 → {gap_hand:.1e}", 2, "newf"),
                                                                         (f"평균을 먼저 뺀 RMSNorm vs LayerNorm → {gap_c:.1e}", 3)],
                   ["입력의 평균이 0에서 멀면 두 결과가 많이 달라요. RMSNorm은 평균을 빼지 않으니까요.",
                    "한 줄짜리 식이 모듈과 같은 값이에요. 처음 γ(칸마다 곱하는 값)는 모두 1이라 식에서 뺐어요.",
                    f"평균을 먼저 빼 주면 거의 같아져요. 남은 작은 차이는 두 모듈의 ε(LayerNorm {eps_ln:g}, RMSNorm {eps_rms:g})이 달라서예요. 학습된 모델 안에서는 벡터 평균이 0 근처라, 평균을 빼는 계산을 생략해도 성능이 거의 같아요."])

# ---------------- ③
inner_g, inner_l = g0.mlp.c_fc.weight.shape[1], l0.mlp.gate_proj.out_features
S.claim(ns["swiglu_gap"] < 1e-6, "③ 손 계산과 모듈이 같다")
outs["s3"] = S.box("③ 작은 신경망 두 가지", [(f"GPT-2: 256 → {inner_g:,} → 256, 행렬 2개 → 가중치 {P['GPT-2']['작은 신경망 (층마다)']:,}개", 1),
                                          (f"Llama: 256 → {inner_l} (gate, up 두 갈래) → 256, 행렬 3개 → 가중치 {P['Llama']['작은 신경망 (층마다)']:,}개", 2, "newf"),
                                          (f"down(SiLU(gate(x)) × up(x)) 손 계산 vs 모듈 → 가장 큰 차이 {ns['swiglu_gap']:.1e}", 3)],
                   [f"안쪽을 벡터의 4배({inner_g:,}칸)로 넓혔다가 다시 줄여요.",
                    f"행렬이 하나 늘어서, 안쪽을 4배 대신 약 8/3배({inner_l}칸)로 줄여 가중치 수를 맞췄어요.",
                    "모듈 안에서 하는 일이 정확히 이 한 줄이에요."])

# ---------------- ④
rp = ns["rope"]
S.claim(abs(rp[(5, 3)] - rp[(105, 103)]) < 1e-3 and abs(rp[(5, 3)] - rp[(2005, 2003)]) < 1e-3 and abs(rp[(5, 3)] - rp[(5, 4)]) > 0.01 and abs(rp[(5, 3)] - rp[(5, 0)]) > 0.01,
        "④ 거리가 같으면 점수가 소수점 셋째 자리까지 같고, 거리가 다르면 다르다")
outs["s4"] = S.table("④ Query와 Key를 자리만큼 돌린 뒤의 내적 (head 하나, 32칸)", ["Query 자리", "Key 자리", "거리", "점수"],
                     [[i, j, i - j, f"{v:+.4f}"] for (i, j), v in rp.items()],
                     ["거리가 2로 같으면 자리가 5든 2,005든 점수가 같아요(소수점 셋째 자리까지). 거리가 1이나 5로 바뀌면 점수도 바뀌어요.",
                      f"GPT-2의 위치 표는 {ns['max_pos']:,}줄이라, 2,005번째 자리는 넣을 수조차 없어요. RoPE는 표가 없어서 어떤 자리든 돌릴 수 있어요. 다만 학습 때 본 길이보다 훨씬 길면 처음 보는 각도라 성능이 떨어져서, 각도를 늘려 맞추는 방법(YaRN, NTK, LongRoPE)을 써요."])

# ---------------- ⑤
kvd = ns["kv"]
names5 = {8: "MHA", 2: "GQA", 1: "MQA"}
S.claim(kvd[2]["bytes"] * 4 == kvd[8]["bytes"] and kvd[1]["bytes"] * 8 == kvd[8]["bytes"], "⑤ GQA(2)는 1/4, MQA는 1/8")
outs["s5"] = S.table("⑤ 300토큰을 넣은 뒤의 KV 캐시 (작은 모델, 4층, 32비트)", ["K, V head 수", "이름", "층 하나의 K 모양", "캐시 크기", "Attention 가중치 (층마다)"],
                     [[h, f"<b>{names5[h]}</b>", str(kvd[h]["shape"]), f"{kvd[h]['bytes'] / 1e6:.2f} MB", f"{kvd[h]['attn']:,}"] for h in (8, 2, 1)],
                     [f"모양은 (문장 수, K head 수, 토큰 수, head 크기)예요. Query head는 셋 다 8개인데, K와 V를 저장하는 head만 8, 2, 1로 줄였어요. 캐시가 {kvd[8]['bytes'] / kvd[2]['bytes']:.0f}분의 1, {kvd[8]['bytes'] / kvd[1]['bytes']:.0f}분의 1이 돼요.",
                      f"K, V를 만드는 행렬도 작아져서 Attention 가중치가 {kvd[8]['attn']:,}개에서 {kvd[1]['attn']:,}개까지 줄어요."])
big = ns["big"]
w8b = 8e9 * 2 / 2 ** 30
outs["t5"] = S.table("8B급 설정이면 (32층, Query head 32개, head 크기 128, 16비트, 문맥 8,192토큰)", ["방식", "K, V head 수", "요청 하나의 KV 캐시"],
                     [["MHA", 32, f"{big['MHA'] / 2 ** 30:.2f} GB"], ["GQA", 8, f"{big['GQA'] / 2 ** 30:.2f} GB"], ["MQA", 1, f"{big['MQA'] / 2 ** 30:.3f} GB"]],
                     [f"가중치(80억 개 × 2바이트)가 약 {w8b:.0f}GB인데, MHA라면 8,192토큰 요청 {int(w8b // (big['MHA'] / 2 ** 30)) + 1}개만 동시에 받아도 KV 캐시가 가중치보다 커져요. GQA면 {int(w8b // (big['GQA'] / 2 ** 30)) + 1}개부터예요.",
                      "MQA가 가장 작지만 품질이 떨어지기 쉬워서, 2024년 이후 새 모델은 GQA가 기본이에요."])

# ---------------- ⑥
tp, td = ns["t_prefill"], ns["t_decode"]
per_tok = tp / 200
gen = ns["gen"]
g_on, g_off = min(gen[True]), min(gen[False])
rg = ns["reuse_gap"]
S.claim(td / per_tok > 5, "⑥ 토큰 하나당 decode가 prefill보다 5배 넘게 비싸다")
S.claim(g_off / g_on > 2, "⑥ 캐시 없이 만들면 2배 넘게 느리다")
S.claim(max(rg) < 1e-4, "⑥ 앞부분 캐시를 다시 써도 처음부터 계산한 값과 같다")
outs["s6"] = S.box("⑥ GQA 모델로 재기 (CPU, 다섯 번 중 가장 빠른 값)", [(f"prefill: 입력 200토큰 한 번에 → {tp * 1000:.1f} ms (토큰 하나당 {per_tok * 1000:.3f} ms)", 1),
                                                                 (f"decode: 캐시를 읽으며 다음 1토큰 → {td * 1000:.2f} ms", 2),
                                                                 (f"50토큰 생성: 캐시 씀 {g_on:.2f}초, 캐시 안 씀 {g_off:.2f}초", 3, "newf"),
                                                                 (f"앞부분 200토큰 캐시 + 새 20토큰 vs 처음부터 220토큰 → 가장 큰 차이 {max(rg):.0e}", 4, "newf")],
                   ["200토큰을 한꺼번에 행렬 곱으로 묶어 계산해요. 계산 장치를 꽉 채워 쓰는 단계라, 계산량이 시간을 정해요.",
                    f"토큰 하나만 계산하는데도 prefill의 토큰 하나당 시간보다 {td / per_tok:.0f}배 걸려요. 1토큰을 위해서도 가중치 전체와 캐시를 메모리에서 한 번씩 읽어야 해서, 계산보다 메모리 읽기가 시간을 정해요.",
                    f"캐시가 없으면 새 토큰마다 앞 토큰 전체를 다시 계산해서 {g_off / g_on:.1f}배 느려요. 글이 길수록 차이가 더 벌어져요.",
                    "앞 토큰의 K, V는 뒤에 무엇이 오든 변하지 않아서, 같은 앞부분의 캐시를 복사해 다시 써도 값이 같아요(남은 차이는 계산 순서에 따른 반올림). 요청마다 220토큰 대신 20토큰만 계산했어요."])

# ---------------- ⑦
cl, far = ns["cache_len"], ns["far"]
W = ns["mistral"].config.sliding_window
reach = ns["LAYERS"] * (W - 1)
S.claim(all(cl[n][0] == n for n in cl) and cl[50][1] == 50 and cl[100][1] == W - 1 and cl[300][1] == W - 1, "⑦ 전체는 길이만큼, 윈도우는 W−1에서 멈춘다")
S.claim(far["전체"][299] > 0.01 and far["윈도우 64"][100] > 0.01 and far["윈도우 64"][299] == 0 and 100 < reach < 299, "⑦ 윈도우 모델은 299번째가 그대로, 100번째는 바뀐다")
outs["s7"] = S.table(f"⑦ 층 하나의 KV 캐시 길이 (윈도우 W = {W})", ["입력 토큰 수", "전체 Attention", "슬라이딩 윈도우"], [[n, a, b] for n, (a, b) in cl.items()],
                     [f"전체 Attention은 넣은 만큼 쌓이고, 윈도우 모델은 {W - 1}개에서 멈춰요. 새 토큰이 들어오면 가장 오래된 것을 지워요."])
outs["s7b"] = S.table("맨 앞 토큰 하나만 바꿨을 때 출력이 변한 정도 (가장 큰 차이)", ["모델", "100번째 출력", "299번째 출력"],
                      [[k, f"{v[100]:.3f}", f"{v[299]:.3f}"] for k, v in far.items()],
                      [f"윈도우 모델에서 299번째 출력은 맨 앞 토큰을 전혀 몰라요. 한 층에서 {W - 1}칸씩, {ns['LAYERS']}층을 거치면 최대 {reach}칸 앞까지만 닿거든요. 100번째는 그 안이라 바뀌어요.",
                       "그래서 실제 모델은 윈도우 층과 전체 Attention 층을 번갈아 두거나(Gemma 2, GPT-OSS), 윈도우를 수천 토큰으로 넓게 잡아요."])
ctx = np.array([1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
gb = lambda h, t: 2 * 32 * h * 128 * t * 2 / 2 ** 30  # noqa: E731
fig, ax = plt.subplots(figsize=(8.4, 4.4))
for lab, ys, col, ls in (("MHA (K, V head 32)", gb(32, ctx), C["pink"], "-"), ("GQA (8)", gb(8, ctx), C["purple"], "-"), ("MQA (1)", gb(1, ctx), C["gray"], "-"),
                         ("GQA (8) + 윈도우 4,096", gb(8, np.minimum(ctx, 4096)), C["purple"], "--")):
    ax.plot(ctx, ys, marker="o", color=col, ls=ls, lw=2, label=lab)
ax.axhline(w8b, color=C["ink"], lw=1, ls=":")
ax.text(1100, w8b * 1.12, f"가중치 약 {w8b:.0f}GB", fontsize=9, color=C["ink"])
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.minorticks_off()
ax.set_xticks(ctx, [f"{c // 1024}K" for c in ctx])
yt = [0.0625, 0.25, 1, 4, 16, 64]
ax.set_yticks(yt, [f"{v:g}" for v in yt])
ax.set(xlabel="문맥 길이 (토큰)", ylabel="요청 하나의 KV 캐시 (GB, 로그 눈금)", title="8B급 설정 (32층, head 크기 128, 16비트)")
ax.legend(fontsize=9, loc="lower right")
outs["f7"] = S.images("문맥 길이에 따른 KV 캐시", [(fig_save(fig, "kv_cache_by_context"), "")],
                      [f"MHA로 문맥 128K면 요청 하나의 캐시가 {gb(32, 131072):.0f}GB예요. GQA로 4분의 1, 윈도우를 더하면 4,096토큰부터 더 늘지 않아요(점선). 다만 윈도우만 쓰면 먼 토큰을 직접 못 봐서, 실제로는 전체 Attention 층과 섞어요."])
outs["f7"]["chart"] = True

# ---------------- ⑧
pe, mt, ma = ns["per_expert"], ns["moe_total"], ns["moe_active"]
load = ns["load"].tolist()
S.claim(pe == P["Llama"]["작은 신경망 (층마다)"], "⑧ 전문가 하나 = Llama의 작은 신경망 하나")
S.claim(sum(load) == 400 and max(load) >= 2 * min(load), "⑧ 무작위 라우터인데도 가장 많은 전문가가 가장 적은 것의 2배 넘게")
outs["s8"] = S.box("⑧ 전문가 8개 중 2개 (작은 모델, 4층)", [(f"전문가 하나 → {pe:,}개 (③의 SwiGLU 하나와 같은 크기)", 1),
                                                      (f"전체 가중치 {mt:,}개, 토큰 하나가 거치는 가중치 {ma:,}개 ({ma / mt:.0%})", 2, "newf"),
                                                      (f"첫 층에서 전문가마다 맡은 토큰 수 (200토큰 × 2) → {', '.join(map(str, load))}", 3, "newf")],
                   ["작은 신경망을 같은 모양으로 8벌 만들었어요.",
                    f"Llama({T['Llama']:,}개)의 약 {mt / T['Llama']:.0f}배 크기지만, 토큰 하나의 계산은 전문가 2개 몫이라 {ma / T['Llama']:.1f}배쯤이에요. 대신 메모리에는 8벌이 모두 올라가 있어야 해요.",
                    f"학습 전 무작위 라우터인데도 {min(load)}개에서 {max(load)}개까지 고르지 않아요. 학습하면 잘하는 전문가로 더 몰리기 쉬워서, 몰림을 벌하는 보조 손실(Mixtral)이나 점수에 편향을 더해 맞추는 방법(DeepSeek V3)을 써요."])

flow = [[1, "같은 크기로", "GPT-2와 Llama", f"{T['GPT-2'] / 1e6:.1f}M vs {T['Llama'] / 1e6:.1f}M"], [2, "정규화", "LayerNorm → RMSNorm", "평균 빼기 생략"],
        [3, "작은 신경망", "GELU → SwiGLU", "행렬 3개, 8/3배"], [4, "위치", "표 → RoPE", "거리만 남는 점수"], [5, "Attention", "MHA → GQA → MQA", "KV 1/4, 1/8"],
        [6, "KV 캐시", "prefill과 decode", f"캐시 없으면 {g_off / g_on:.1f}배"], [7, "슬라이딩 윈도우", "앞의 W개만", f"캐시 {W - 1}에서 멈춤"], [8, "MoE", "8개 중 2개", f"활성 {ma / mt:.0%}"]]
S.save(ROOT / "topics" / "model" / "llm-architectures-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow, "env": ENV})

lineage = K.table(["흐름", "시작", "핵심", "대표 모델"], [
    ["1. Llama 세트", "2023.02", "RMSNorm, SwiGLU, RoPE, GQA 네 가지가 기본이 됨", "Llama 1~3, Mistral, Cohere Command R"],
    ["2. 학습 안정화 실험", "2024.02", "정규화를 늘리고(QK Norm, 블록당 여러 개), 점수 상한(softcapping)", "Gemma 1~4"],
    ["3. 작고 효율적으로", "2024.04", "같은 부품으로 작은 모델, 긴 문맥", "Qwen 2~3, OLMo, Phi, SmolLM"],
    ["4. 분야, 언어별", "2024", "코드, 한국어, 중국어 특화", "StarCoder2, InternLM, GLM-4, HyperCLOVA X SEED, ExaOne"],
    ["5. MoE", "2023.12", "전문가 여럿 중 일부만 계산 (⑧)", "Mixtral, DeepSeek V2~V4, Qwen MoE, Solar Open"],
    ["6. Transformer 너머", "2023.12", "Attention 대신 고정 크기 순환 상태 (SSM, RNN)", "Mamba 1, 2, RWKV v7"],
    ["7. 하이브리드", "2024.03", "Attention 층과 SSM, 선형 Attention 층을 섞기", "Jamba, Nemotron-H, Qwen3 Next"],
    ["8. 복합 구조", "2025.04", "위 흐름을 층마다 다르게 섞기", "Llama 4, DeepSeek V4, GPT-OSS, Kimi K2"]])
why = f"""<p><a href="llm-principles.html">Transformer 구조</a>에서 GPT 같은 디코더 블록을 봤어요. 2023년 Llama 1이 공개된 뒤 가중치를 공개한 LLM이 수십 종 나왔는데, 골격은 거의 그대로예요. 디코더 블록을 쌓고, 다음 토큰을 맞히며 배워요. 바뀐 건 <b>블록 안의 부품</b>이에요. 바꾼 이유는 크게 셋이에요.</p>
{K.table(["이유", "바뀐 부품"], [["깊게 쌓아도 학습이 흔들리지 않게", "RMSNorm(②), Pre-norm, QK Norm"], ["같은 계산으로 더 잘 배우게, 더 길게 쓰게", "SwiGLU(③), RoPE(④)"],
                                  ["서비스할 때 메모리와 시간을 줄이게", "GQA(⑤), KV 캐시 재사용(⑥), 슬라이딩 윈도우(⑦), MoE(⑧)"]])}
<p>특히 세 번째가 요즘 설계를 이끌어요. 긴 글을 여러 사람이 동시에 쓰면 모델 가중치보다 <b>KV 캐시</b>가 메모리를 더 차지해서, 캐시를 줄이는 방법이 모델 구조를 정해요. 흐름을 크게 나누면 이래요.</p>
{lineage}
<p>이 쪽은 1번 흐름의 네 부품(②~⑤)과, 서비스 문제를 푸는 KV 캐시(⑥), 슬라이딩 윈도우(⑦), MoE(⑧)를 작은 모델로 직접 만들어 봐요. 모델은 받지 않고 <code>transformers</code>의 설정만으로 만들어서 가중치는 무작위예요. 그래서 이 쪽의 숫자는 <b>모양, 크기, 계산 방식</b>에 대한 것이고, 어느 모델이 더 똑똑한지는 아니에요.</p>
"""
whole = """<p>GPT-2와 Llama를 같은 크기로 만들어 나란히 두고(①), 부품을 하나씩 바꿔 봐요(②~⑤). 그다음 생성할 때의 메모리와 시간을 다뤄요(⑥~⑧). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "같은 크기의 GPT-2와 Llama", "벡터 256칸, 4층, head 8개로 두 모델을 설정만으로 만들고, 가중치가 어느 부품에 있는지 세요.",
           "<p><code>GPT2Config</code>, <code>LlamaConfig</code>에 크기만 적어 모델 클래스에 넣으면, 아무것도 받지 않고 무작위 가중치 모델이 생겨요. 실제 Llama, Mistral, Mixtral도 같은 클래스에 큰 설정과 학습된 가중치를 넣은 거예요. 크기를 같게 맞추면 부품의 차이만 남아요.</p>"
           + K.out("s1")
           + K.code_table([("GPT2Config(n_embd, n_layer, n_head, ...)", "GPT-2 모양의 설정. 이름이 옛 방식(n_embd = 벡터 칸 수)"), ("LlamaConfig(hidden_size, num_hidden_layers, ...)", "Llama 모양의 설정. Mistral, Mixtral, Qwen 설정도 이름이 거의 같아요"),
                           ("LlamaForCausalLM(config)", "설정으로 무작위 가중치 모델 만들기 (다음 토큰 맞히기용 출력 층 포함)"), ("model.model.layers[0]", "첫 번째 블록. GPT-2는 model.transformer.h[0]")]), "s1"),
    K.step(2, "정규화: LayerNorm → RMSNorm", "LayerNorm은 평균을 빼고 표준편차로 나누고, RMSNorm은 평균은 두고 크기(RMS)로만 나눠요.",
           "<p>층을 깊이 쌓으면 벡터 크기가 층마다 커지거나 작아져서 학습이 흔들려요. 그래서 블록마다 벡터 크기를 맞춰요(<a href=\"llm-principles.html\">Transformer 구조</a> ⑤의 LayerNorm).</p>"
           + K.table(["", "식", "배우는 값"], [["LayerNorm", "(x − 평균) ÷ √(분산 + ε) × γ + β", "γ, β (칸마다)"], ["RMSNorm", "x ÷ √(평균(x²) + ε) × γ", "γ (칸마다)"]])
           + "<p>RMSNorm은 평균을 구하고 빼는 계산과 β를 없앴어요. 계산이 가벼운데 성능은 거의 같아서 Llama 1 이후 기본이 됐어요.</p>"
           + K.out("s2")
           + "<p><b>어디에 두나</b>: 원조 Transformer와 BERT, e5는 블록 출력 쪽에서 정규화해요(post-norm, <code>norm(x + layer(x))</code>). GPT-2와 Llama는 입력 쪽에 둬요(pre-norm, <code>x + layer(norm(x))</code>). 잔차 길이 정규화를 거치지 않아서, 깊게 쌓아도 학습이 안정적이에요. 아주 깊은 MoE 모델에서는 post-norm 계열을 다시 섞기도 해요.</p>"
           "<p><b>QK Norm</b>: Attention 점수를 내기 전에 Query와 Key에도 RMSNorm을 걸어요. 점수가 너무 커져서 softmax가 한 토큰에 쏠리는 걸 막아요. 2024~25년 새 모델(Qwen 3, OLMo 2, Gemma 3)에 빠르게 퍼졌어요.</p>"
           + K.code_table([("block.input_layernorm(x)", "Llama 블록의 Attention 앞 RMSNorm"), ("block.ln_1(x)", "GPT-2 블록의 Attention 앞 LayerNorm"), ("config.rms_norm_eps", "RMSNorm의 ε")]), "s2"),
    K.step(3, "작은 신경망: GELU → SwiGLU", "한 갈래가 다른 갈래의 '문'이 되는 작은 신경망이에요. 행렬이 셋이라 안쪽을 좁혀 크기를 맞춰요.",
           "<p>블록마다 Attention 뒤에 토큰마다 따로 계산하는 작은 신경망이 있어요. GPT-2는 <code>down(GELU(up(x)))</code>예요. SwiGLU는 갈래를 둘로 나눠, 한 갈래(gate)를 SiLU에 넣은 값을 다른 갈래(up)에 칸마다 곱해요.</p>"
           "<p><code>down(SiLU(gate(x)) × up(x))</code>, SiLU(z) = z × sigmoid(z)</p>"
           "<p>gate 갈래가 칸마다 \"얼마나 통과시킬지\"를 입력에 따라 정하는 셈이에요. 같은 가중치 수로 더 빨리, 더 잘 배운다는 결과가 많아서 2023년 이후 표준이 됐어요. 최근에는 값이 폭주하지 않게 범위를 자르는(clamp) 변형도 써요.</p>"
           + K.out("s3")
           + K.code_table([("mlp.gate_proj, mlp.up_proj, mlp.down_proj", "SwiGLU의 행렬 셋"), ("F.silu(z)", "z × sigmoid(z)")]), "s3"),
    K.step(4, "위치: 위치 표 → RoPE", "자리마다 배운 벡터를 더하는 대신, Query와 Key를 자리만큼 돌려요. 그러면 점수가 두 토큰의 거리에만 달려요.",
           "<p>GPT-2는 자리마다 배운 벡터(위치 표의 한 줄)를 토큰 벡터에 더해요(<a href=\"llm-principles.html\">Transformer 구조</a> ②). 표의 줄 수가 곧 넣을 수 있는 가장 긴 길이예요.</p>"
           "<p><b>RoPE</b>(Rotary Position Embedding)는 Query와 Key 벡터를 두 칸씩 짝지어, 짝마다 <b>자리 × 정해진 각도</b>만큼 돌려요. 짝마다 도는 속도가 달라서 가까운 거리와 먼 거리를 함께 구별해요. Query를 i만큼, Key를 j만큼 돌린 뒤 내적하면 두 회전이 상쇄되고 <b>i − j만큼의 회전</b>만 남아요. 그래서 점수가 절대 자리가 아니라 거리에 달려요.</p>"
           + K.out("s4")
           + "<p>위치를 Value가 아니라 Query와 Key에만 넣고, 한 번 돌린 Key는 뒤 토큰이 와도 그대로라 KV 캐시(⑥)와도 잘 맞아요. 일부 칸만 돌리고 나머지는 뜻을 담게 두는 Partial RoPE, 층이나 Attention 종류마다 각도를 달리 하는 변형도 있어요.</p>"
           + K.code_table([("model.model.rotary_emb(x, position_ids)", "자리마다 cos, sin 값"), ("apply_rotary_pos_emb(q, k, cos, sin)", "Query와 Key를 돌리기 (transformers의 Llama 코드 안 함수)")]), "s4"),
    K.step(5, "Attention: K, V head를 나눠 쓰기 (MHA → GQA → MQA)", "head마다 따로 갖던 K, V를 여러 head가 같이 써요. 점수 계산 방식은 같고, 저장할 K, V가 줄어요.",
           "<p>다음 토큰을 하나 만들 때마다 앞 토큰 전체의 K, V가 필요해서, 생성할 때는 이걸 쌓아 둬요(KV 캐시, ⑥). 그래서 K, V의 양이 곧 메모리예요.</p>"
           + K.table(["방식", "Query head", "K, V head", "예"], [["MHA", "8", "8 (head마다)", "GPT-2, Llama 1"], ["<b>GQA</b>", "8", "2 (4개 head가 하나씩 같이)", "Llama 2 70B, Llama 3, Mistral, 요즘 대부분"], ["MQA", "8", "1 (모두 하나)", "PaLM, Falcon"]])
           + K.out("s5") + K.out("t5")
           + "<p>더 줄이는 방법도 있어요. DeepSeek V2의 <b>MLA</b>는 K, V를 작은 잠재 벡터로 압축해 저장했다가 쓸 때 펼쳐요. 슬라이딩 윈도우(⑦)는 오래된 토큰을 지워요.</p>"
           + K.code_table([("LlamaConfig(..., num_key_value_heads=2)", "K, V head 수. Query head 수와 같으면 MHA, 1이면 MQA"), ("model(ids, use_cache=True).past_key_values", "넣은 토큰들의 KV 캐시"),
                           ("cache.layers[0].keys.shape", "(문장 수, K, V head 수, 토큰 수, head 크기)")]), "s5"),
    K.step(6, "KV 캐시: prefill과 decode, 앞부분 다시 쓰기", "입력을 한 번에 계산해 캐시를 만드는 prefill과, 캐시를 읽으며 한 토큰씩 만드는 decode로 나뉘어요. 같은 앞부분의 캐시는 다시 쓸 수 있어요.",
           "<p>디코더는 뒤를 못 보니까(<a href=\"llm-principles.html\">Transformer 구조</a> ⑦), 앞 토큰의 K, V는 뒤에 무엇이 붙어도 변하지 않아요. 그래서 한 번 계산한 K, V를 쌓아 두면, 다음 토큰은 새 토큰 하나만 계산하면 돼요. 생성은 두 단계로 나뉘어요.</p>"
           + K.table(["단계", "하는 일", "시간을 정하는 것", "사용자가 느끼는 것"], [["<b>prefill</b>", "입력 전체를 한꺼번에 계산해 캐시 만들기", "계산량", "첫 토큰까지 걸리는 시간 (TTFT)"],
                                                                         ["<b>decode</b>", "캐시를 읽으며 한 토큰씩 만들기", "메모리 읽기 (가중치와 캐시)", "토큰이 나오는 속도"]])
           + K.out("s6")
           + "<p><b>서빙 엔진이 하는 일</b>: vLLM 같은 서빙 엔진은 캐시를 고정 크기 블록(예: 16토큰, 128토큰)으로 나눠 메모리에 흩어 담아요(PagedAttention). 블록 내용의 해시로 같은 앞부분을 찾아, 다른 요청이 계산한 캐시를 그대로 다시 써요(prefix caching). 시스템 Prompt, 긴 문서, 여러 턴 대화처럼 앞부분이 되풀이될수록 첫 토큰까지 시간이 크게 줄어요.</p>"
           "<p><b>캐시를 밖으로 옮기기(offloading)</b>: GPU 메모리가 모자라면 캐시를 CPU 메모리, 디스크, 네트워크 저장소로 옮겨 뒀다가 필요할 때 가져와요. 다시 계산하는 것보다 읽어 오는 게 빠를 때만 이득이라, 짧은 Prompt는 그냥 다시 계산하는 게 나아요. 사용자가 많고 같은 앞부분을 많이 공유할수록 효과가 커요.</p>"
           + K.code_table([("model(ids, use_cache=True)", "prefill: 캐시를 만들어 past_key_values로 돌려줘요"), ("model(next_ids, past_key_values=cache, use_cache=True)", "decode: 캐시 뒤에 이어서 계산 (캐시에 새 토큰이 붙어요)"),
                           ("cache.crop(-1)", "캐시 끝에서 1토큰 지우기"), ("copy.deepcopy(cache)", "같은 앞부분 캐시를 요청마다 복사해 쓰기"), ("model.generate(..., use_cache=False)", "캐시 없이 생성 (비교용)")]), "s6"),
    K.step(7, "슬라이딩 윈도우: 앞의 W개만 보기", "토큰마다 바로 앞 W개만 봐요. 캐시는 W개에서 더 늘지 않고, 먼 토큰은 층을 거쳐 간접적으로만 닿아요.",
           "<p>전체 Attention은 점수 계산이 길이의 제곱으로, KV 캐시가 길이에 비례해서 늘어요. <b>슬라이딩 윈도우</b>는 토큰마다 바로 앞 W개만 보게 해서 둘 다 W에 묶어요(Mistral 7B가 퍼뜨렸어요).</p>"
           + K.out("s7") + K.out("s7b") + K.out("f7")
           + K.code_table([("MistralConfig(..., sliding_window=64)", "앞 64개만 보는 Attention"), ("cache.layers[0]", "윈도우 모델에서는 오래된 토큰을 지우는 캐시 층")]), "s7"),
    K.step(8, "MoE: 전문가 8개 중 2개만", "작은 신경망을 여러 벌(전문가) 두고, 토큰마다 라우터가 고른 몇 개만 계산해요. 가중치는 많아도 토큰 하나의 계산은 작아요.",
           "<p>모델을 키우면 더 잘하지만 토큰마다 계산도 커져요. <b>MoE</b>(Mixture of Experts)는 블록의 작은 신경망(③의 SwiGLU)을 전문가 N벌로 늘리고, 작은 Linear 하나(라우터)가 토큰마다 전문가 점수를 매겨 <b>상위 k개만</b> 계산해서 가중 합해요. Mixtral은 8벌 중 2개예요.</p>"
           + K.out("s8")
           + "<p>요즘은 모든 토큰이 늘 거치는 <b>공유 전문가</b>를 하나 두고 나머지를 잘게 많이 나누기도 해요(Qwen MoE, DeepSeek, Solar Open). 라우터 점수도 softmax 대신 전문가마다 따로 매기는 sigmoid로 바뀌는 추세예요. 학습이 까다로운 대신, 전체 크기에 비해 계산이 작아서 큰 모델일수록 MoE가 많아요.</p>"
           + K.code_table([("MixtralConfig(..., num_local_experts=8, num_experts_per_tok=2)", "전문가 수와 토큰마다 고르는 수"), ("model(ids, output_router_logits=True).router_logits", "층마다 라우터 점수 (토큰 수 × 전문가 수)"),
                           ("router.topk(2, dim=-1).indices", "토큰마다 고른 전문가 번호"), ("torch.bincount(x, minlength=8)", "번호마다 개수 세기")]), "s8"),
]
practice = [("새 모델의 설정 읽기", "<code>config.json</code>에서 <code>num_key_value_heads</code>(GQA), <code>sliding_window</code>, <code>num_local_experts</code>와 <code>num_experts_per_tok</code>(MoE), <code>rope_scaling</code>(긴 문맥), <code>tie_word_embeddings</code>를 보면 이 쪽의 부품 중 무엇을 썼는지 알 수 있어요."),
            ("서빙 메모리 어림하기", "KV 캐시 = 2 × 층 수 × K, V head 수 × head 크기 × 토큰 수 × 바이트(16비트면 2)예요. 여기에 동시 요청 수를 곱하고 가중치 크기를 더해요(⑤)."),
            ("첫 토큰이 느리다", "prefill이 길어서예요. 공통 앞부분을 앞에 모아 prefix caching이 걸리게 하고, 필요 없는 문맥을 줄여요(⑥)."),
            ("토큰이 천천히 나온다", "decode는 메모리 읽기가 병목이라, 가중치를 작게(양자화, <a href=\"model-compression.html\">모델 가볍게</a>), KV 캐시를 작게(GQA, MLA), 여러 요청을 묶어 한 번에 읽어요."),
            ("더 빨리 생성하기 (추측 디코딩)", "작은 모델(drafter)이 토큰 여러 개를 먼저 제안하고, 큰 모델이 한 번의 계산으로 검사해 맞는 데까지 받아들여요. 틀리면 큰 모델의 답을 쓰니 결과는 그대로예요. 모델 안에 여러 토큰을 미리 맞히는 머리(MTP)를 같이 학습해 두기도 해요."),
            ("Attention 없는 모델", "Mamba, RWKV는 Attention 대신 고정 크기의 순환 상태로 문맥을 요약해서, 길이가 늘어도 메모리가 일정해요. 대신 앞의 내용을 정확히 꺼내는 데 약해서, Attention 층과 섞은 하이브리드(Jamba, Nemotron-H, Qwen3 Next)가 늘고 있어요.")]
libs = [("①", "<code>GPT2Config</code>, <code>LlamaConfig</code>, <code>LlamaForCausalLM</code>", "설정으로 모델 만들기 (transformers)"), ("②③", "<code>input_layernorm</code>, <code>gate_proj</code>, <code>up_proj</code>, <code>down_proj</code>, <code>F.silu</code>", "블록 안 부품"),
        ("④", "<code>rotary_emb</code>, <code>apply_rotary_pos_emb</code>", "RoPE"), ("⑤⑥", "<code>num_key_value_heads</code>, <code>past_key_values</code>, <code>use_cache</code>, <code>crop</code>, <code>generate</code>", "GQA, KV 캐시"),
        ("⑦", "<code>MistralConfig(sliding_window=...)</code>", "슬라이딩 윈도우"), ("⑧", "<code>MixtralConfig</code>, <code>output_router_logits</code>, <code>topk</code>, <code>bincount</code>", "MoE")]
K.build(ROOT / "topics" / "model" / "llm-architectures.html", title="요즘 LLM의 부품", h1="요즘 LLM의 부품: GPT-2의 블록은 어떻게 바뀌었나", track="모델", crumb="모델 &gt; 순서 모델과 Transformer",
        one_line="2023년 Llama 이후의 LLM도 골격은 <b>디코더 블록을 쌓은 Transformer</b>예요. 블록 안에서는 정규화가 <b>RMSNorm</b>, 작은 신경망이 <b>SwiGLU</b>, 위치가 <b>RoPE</b>, Attention이 K, V를 나눠 쓰는 <b>GQA</b>로 바뀌었어요. 그리고 생성할 때의 메모리와 시간을 줄이려고 <b>KV 캐시</b>를 다시 쓰고, <b>슬라이딩 윈도우</b>와 <b>MoE</b>를 붙여요.",
        why=why, whole=whole, steps=steps, code_file="examples/llm_arch/modern_llm.py", practice=practice, libs=libs,
        libs_note="<code>count</code>, <code>rope_score</code>, <code>kv_bytes</code>, <code>best_of</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"llm-principles.html\">Transformer 구조</a>", "이 쪽의 출발점인 원래 블록"), ("<a href=\"gpt-generation.html\">작은 GPT와 디코딩</a>", "디코더를 직접 학습하고 다음 토큰 고르기"),
                   ("<a href=\"model-compression.html\">모델 가볍게</a>", "가중치를 int8로 줄이기"), ("<a href=\"../backend/llm-client.html\">LLM API 클라이언트</a>", "서빙된 LLM을 부르는 쪽")],
        sources=["함께 정리한 자료: 신정규(래블업), \"LLM internals: 언어 모델의 계보와 알고리즘 진화\" 발표 (2026-05). 8가지 흐름, Norm, Attention, 위치, 활성 함수, MoE, KV 캐시, 추측 디코딩의 흐름을 가져왔어요. vLLM 사용자 모임(래블업) 노트에서 prefill과 decode, KV 캐시 offloading, prefix caching을 가져왔어요. 수업에서 따로 다루지 않은 주제라 예제는 새로 만들었어요",
                 "결과 값과 그림: <code>tools/pages/llm_arch.py</code>가 <code>examples/llm_arch/modern_llm.py</code>를 단계별로 실제 실행한 결과 (CPU, 가중치는 무작위). ⑤의 8B급 설정과 그림은 같은 식으로 계산한 값",
                 "공식 문서: <a href=\"https://huggingface.co/docs/transformers/model_doc/llama\" target=\"_blank\">Llama</a>, <a href=\"https://huggingface.co/docs/transformers/model_doc/mistral\" target=\"_blank\">Mistral</a>, <a href=\"https://huggingface.co/docs/transformers/model_doc/mixtral\" target=\"_blank\">Mixtral</a>, <a href=\"https://huggingface.co/docs/transformers/cache_explanation\" target=\"_blank\">Caching</a>, <a href=\"https://docs.vllm.ai/en/latest/\" target=\"_blank\">vLLM</a>"])
print("ok")
