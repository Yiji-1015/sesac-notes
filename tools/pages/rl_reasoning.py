"""모델 > 강화학습 > 추론 모델과 검증 가능한 보상. examples/reasoning_rl/verifiable_rl.py 하나가 ①~⑥으로 자란다.
실행: python tools/pages/rl_reasoning.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import copy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "reasoning_rl"
code = S.parse((EX / "verifiable_rl.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="verifiable_rl.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__} (CPU 1스레드, seed 0)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "rl"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/rl/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
runs = ns["runs"]
final = {k: l["test"][-1] for k, (m, l) in runs.items()}
acc, base, train, test = ns["accuracy"], ns["base"], ns["train"], ns["test"]
# ---------------- ①
st = ns["sft_table"]
S.claim(st[700]["생각하고 답하기"][1] > 0.95 and st[700]["바로 답하기"][1] < 0.5, "① 풀이 700개면 생각하는 모델은 처음 보는 문제도 95% 넘게, 바로 답하기는 50% 아래")
outs["s1"] = S.table("① 풀이를 따라 배운 뒤 (SFT, 200에폭)", ["학습한 풀이 수", "생각하고 답하기: 배운 문제", "처음 보는 문제", "바로 답하기: 배운 문제", "처음 보는 문제"],
                     [[n, *[f3(v[name][i]) for name in ("생각하고 답하기", "바로 답하기") for i in (0, 1)]] for n, v in st.items()],
                     [f"두 모델 모두 배운 문제는 다 맞혀요. 차이는 처음 보는 300문제에서 나요. 풀이 700개를 배우면 생각하는 모델은 {st[700]['생각하고 답하기'][1]:.2f}, 바로 답하는 모델은 {st[700]['바로 답하기'][1]:.2f}예요.",
                      "바로 답하는 모델은 (a, b, c) 1,000가지 조합을 하나하나 외워야 해요. 생각하는 모델은 \"a + b\" 100가지와 \"중간값 + c\" 100가지만 배우면 되고, 처음 보는 문제도 아는 두 단계로 나눠 풀어요. 중간 단계를 적는 게 추론 모델이 \"생각\"을 길게 쓰는 이유 중 하나예요."])

# ---------------- ②
bp = ns["base_pass"]
S.claim(bp[64] > 3 * bp[1], "② pass@64가 pass@1의 세 배 넘게")
outs["s2"] = S.box("② 출발점: 풀이 100개를 30에폭만 배운 모델", [(f"가장 그럴듯한 답으로 채점: 학습 문제 {acc(base, train):.3f}, 처음 보는 문제 {acc(base, test):.3f}", 1),
                                                          (f"답을 뽑아서 채점 (처음 보는 문제): 1번 {bp[1]:.3f}, 8번 중 하나라도 {bp[8]:.3f}, 64번 중 하나라도 {bp[64]:.3f}", 2, "newf")],
                   [f"한 번에 맞히는 비율은 낮지만, 여러 번 뽑으면 {bp[64]:.0%}의 문제에서 한 번은 맞는 답이 나와요(pass@k). 강화학습은 이렇게 가끔 나오는 맞는 답을 더 자주 나오게 만드는 일이에요. 한 번도 안 나오는 답은 강화할 수 없어요."])

# ---------------- ③
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
steps = lambda l: [min(i * 100, 1999) for i in range(len(l["test"]))]  # noqa: E731
style = {"REINFORCE": (C["gray"], "-"), "GRPO": (C["purple"], "-"), "GRPO + 단계별 보상": (C["blue"], "-"), "가끔 맞히는 문제만": (C["orange"], "--"),
         "SFT 없이 GRPO": (C["pink"], "-"), "SFT 없이 GRPO + 단계별 보상": (C["blue"], "-")}
for ax, names, title in ((axs[0], ["REINFORCE", "GRPO", "GRPO + 단계별 보상", "가끔 맞히는 문제만"], "SFT 모델에서 출발"), (axs[1], ["SFT 없이 GRPO", "SFT 없이 GRPO + 단계별 보상"], "아무것도 안 배운 모델에서 출발")):
    for n in names:
        col, ls = style[n]
        ax.plot(steps(runs[n][1]), runs[n][1]["test"], color=col, ls=ls, lw=2.4, label=n.replace("SFT 없이 ", ""))
    ax.set(xlabel="강화학습 걸음", title=title)
    ax.legend(fontsize=8, loc="upper left")
axs[0].set_ylabel("처음 보는 문제 정확도")
P_CURVE = fig_save(fig, "reasoning_curves")
S.claim(final["GRPO"] > final["REINFORCE"] and final["GRPO"] > acc(base, test) + 0.1, "③ GRPO가 REINFORCE보다 높고 출발점보다 0.1 넘게")
outs["s3"] = S.table("③ 맞았는지만 보고 2,000걸음 (문제 32개 × 답 8개씩)", ["방법", "처음 보는 문제 정확도"],
                     [["출발점 (SFT)", f3(acc(base, test))], ["REINFORCE (보상 그대로)", f3(final["REINFORCE"])], ["GRPO (같은 문제의 답끼리 비교)", f3(final["GRPO"])]],
                     [f"둘 다 올라가지만 GRPO가 더 높아요({final['REINFORCE']:.2f} → {final['GRPO']:.2f}). REINFORCE는 맞은 답을 모두 같은 크기로 밀어 줘요. GRPO는 같은 문제에 대한 답 8개의 평균을 빼고 표준편차로 나눠서, 그 문제에서 평균보다 나은 답만 밀고 못한 답은 눌러요. 쉬운 문제의 흔한 정답보다 어려운 문제의 드문 정답에 더 큰 무게가 가요.",
                      f"그런데 둘 다 {final['GRPO']:.2f} 근처에서 멈춰요. 다음 단계에서 이유를 봐요."])
outs["f_curve"] = chart("강화학습 걸음마다 처음 보는 문제 정확도", P_CURVE, "",
                        ["왼쪽: SFT 모델에서 출발한 네 가지. 모두 몇백 걸음 만에 오르고 멈춰요.",
                         f"오른쪽: 아무것도 배우지 않은 모델에서 출발하면 답만 채점할 때(분홍)는 오히려 떨어지고({final['SFT 없이 GRPO']:.2f}), 생각 단계도 채점하면(파랑) {final['SFT 없이 GRPO + 단계별 보상']:.2f}까지 올라요."])
g_rows = []
for G in (4, 8, 16):
    v = final["GRPO"] if G == 8 else ns["rl"](copy.deepcopy(base), group=G)[1]["test"][-1]
    g_rows.append([f"<b>{G}</b> (지금)" if G == 8 else G, f3(v)])
outs["t3"] = S.table("문제마다 뽑는 답의 수(그룹 크기)를 바꾸면", ["그룹 크기", "처음 보는 문제 정확도"], g_rows,
                     ["그룹이 크면 \"가끔 맞는\" 답이 그룹 안에 하나라도 들어올 확률이 커서 신호가 더 자주 나와요. 대신 답을 뽑는 계산이 그만큼 늘어요. 추론 모델에서는 답 하나가 수천 토큰이라 이게 가장 큰 비용이에요."])

# ---------------- ④
gl = runs["GRPO"][1]
allr, allw, ng = gl["모두 맞음"], gl["모두 틀림"], gl["그룹"]
ns_solved, ns_total = ns["never_then_solved"]
S.claim((allr + allw) / ng > 0.5 and ns_solved <= ns_total * 0.2, "④ 신호 없는 그룹이 절반 넘고, 64번에 못 맞히던 문제는 20% 이하만 풀림")
fig, ax = plt.subplots(figsize=(8.4, 3.0))
names4 = ["REINFORCE", "GRPO", "GRPO + 단계별 보상", "가끔 맞히는 문제만", "SFT 없이 GRPO", "SFT 없이 GRPO + 단계별 보상"]
left = np.zeros(len(names4))
for key, col, lab in (("모두 틀림", C["pink"], "모두 틀림 (신호 없음)"), ("섞임", C["purple"], "맞고 틀린 답이 섞임 (배움)"), ("모두 맞음", C["gray"], "모두 맞음 (신호 없음)")):
    vals = np.array([(runs[n][1]["그룹"] - runs[n][1]["모두 맞음"] - runs[n][1]["모두 틀림"]) if key == "섞임" else runs[n][1][key] for n in names4]) / np.array([runs[n][1]["그룹"] for n in names4])
    ax.barh(names4[::-1], vals[::-1], left=left[::-1], color=col, label=lab)
    left += vals
ax.set(xlabel="그룹 비율", xlim=(0, 1), title="그룹 하나(같은 문제의 답 8개)가 배울 신호를 줬나")
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
P_SIG = fig_save(fig, "reasoning_signal")
outs["s4"] = S.box("④ GRPO 2,000걸음 동안의 그룹 64,000개", [(f"8개 모두 맞음: {allr:,}개 ({allr / ng:.0%})", 1), (f"8개 모두 틀림: {allw:,}개 ({allw / ng:.0%})", 2, "newf"),
                                                       (f"출발점이 64번 뽑아도 못 맞힌 처음 보는 문제 {ns_total}개 중, 강화학습 뒤 맞힌 것: {ns_solved}개", 3)],
                   [f"GRPO는 같은 문제의 답끼리 비교하니까, 8개가 모두 맞거나 모두 틀리면 평균을 뺀 값이 다 0이 돼서 배울 게 없어요. 그런 그룹이 {(allr + allw) / ng:.0%}예요. 뽑은 답의 대부분을 버린 셈이에요.",
                    "\"모두 틀림\"이 특히 문제예요. 정답이 한 번도 안 나오는 문제는 아무리 많이 풀려도 배우지 못해요(보상이 드묾). 그래서 강화학습은 출발 모델이 가끔이라도 맞히는 범위를 넓히기보다 그 안에서 더 자주 맞히게 만드는 쪽이에요."])
outs["f_sig"] = chart("그룹마다 신호가 있었나", P_SIG, "",
                      [f"보라가 신호를 준 그룹이에요. 아무것도 안 배운 모델에서 답만 채점하면(아래서 둘째) 보라가 {1 - (runs['SFT 없이 GRPO'][1]['모두 맞음'] + runs['SFT 없이 GRPO'][1]['모두 틀림']) / runs['SFT 없이 GRPO'][1]['그룹']:.0%}나 되는데도 정확도는 떨어져요. 찍어도 10%는 맞으니 맞은 답의 대부분이 운이고, 그때의 틀린 생각까지 같이 밀어 주기 때문이에요. 신호가 있다고 다 쓸모 있는 신호는 아니에요.",
                       "생각 단계도 채점하면(맨 아래) 분홍이 크게 줄고, 보라가 진짜로 맞은 생각을 밀어 줘요."])

# ---------------- ⑤
S.claim(final["SFT 없이 GRPO + 단계별 보상"] > final["SFT 없이 GRPO"] + 0.3 and abs(final["GRPO + 단계별 보상"] - final["GRPO"]) < 0.05, "⑤ 처음부터면 단계별 보상이 크게, SFT에서는 거의 같음")
outs["s5"] = S.table("⑤ 답만 채점 vs 생각 단계도 채점", ["출발점", "답만 채점", "생각 단계도 채점", "출발점 정확도"],
                     [["SFT 모델", f3(final["GRPO"]), f3(final["GRPO + 단계별 보상"]), f3(acc(base, test))], ["아무것도 안 배운 모델", f3(final["SFT 없이 GRPO"]), f3(final["SFT 없이 GRPO + 단계별 보상"]), "약 0.1"]],
                     [f"아무것도 안 배운 모델은 답이 맞을 확률이 10%뿐이고, 맞아도 생각이 맞아서인지 운인지 몰라요. 답만 채점하면 {final['SFT 없이 GRPO']:.2f}로 오히려 떨어져요. 생각 단계에도 점수를 주면 어느 단계가 틀렸는지 알려 줘서 {final['SFT 없이 GRPO + 단계별 보상']:.2f}까지 올라요.",
                      f"SFT 모델에서 출발하면 단계별 보상의 차이가 거의 없어요({final['GRPO']:.3f} vs {final['GRPO + 단계별 보상']:.3f}). 생각 단계는 SFT에서 이미 꽤 배워서, 남은 문제는 \"한 번도 안 나오는 답\"이기 때문이에요. 단계별 보상은 신호가 가장 모자랄 때 가장 힘이 커요.",
                      "실제 추론 모델에서는 생각 단계마다 맞았는지 알기 어려워요. 그래서 단계를 채점하는 모델(과정 보상 모델)을 따로 학습하거나, 정답이 확인되는 수학과 코드부터 시작해요."])

# ---------------- ⑥
S.claim(final["가끔 맞히는 문제만"] < final["GRPO"], "⑥ 미리 고른 문제만 쓰면 오히려 낮음")
fl = runs["가끔 맞히는 문제만"][1]
outs["s6"] = S.table("⑥ 출발점이 가끔 맞히는 문제만 골라 쓰면", ["", "전체 학습 문제", "가끔 맞히는 문제만"],
                     [["문제 수", len(train), len(ns["useful"])], ["고르는 데 뽑은 답", "0", f"{ns['filter_cost']:,}"], ["처음 보는 문제 정확도", f3(final["GRPO"]), f3(final["가끔 맞히는 문제만"])],
                      ["모두 맞은 그룹 비율", f"{gl['모두 맞음'] / ng:.0%}", f"{fl['모두 맞음'] / fl['그룹']:.0%}"]],
                     [f"출발점으로 학습 문제마다 8번씩 풀어 보고, 0번도 8번도 아닌 {len(ns['useful'])}문제만 남겼어요. 신호가 나올 문제에만 계산을 쓰자는 생각이에요(강연에서는 이렇게 고른 데이터로 뽑는 답 수를 8.55배 줄였다고 해요).",
                      f"여기서는 오히려 낮아졌어요({final['GRPO']:.3f} → {final['가끔 맞히는 문제만']:.3f}). 고른 문제는 몇백 걸음 만에 다 맞히게 돼서 \"모두 맞음\" 그룹이 {fl['모두 맞음'] / fl['그룹']:.0%}로 늘었고, 처음에 다 틀리던 문제는 아예 빠져서 다시 볼 기회가 없었어요. 어려운 정도는 학습하면서 계속 바뀌어서, 한 번 고른 목록은 금방 낡아요.",
                      "그래서 실제로는 학습 도중에 배치마다 \"다 맞거나 다 틀린 그룹\"을 빼고 다시 뽑거나(동적 샘플링), 쉬운 것부터 어려운 것으로 단계를 나눠 가며(커리큘럼) 다시 골라요."])

flow = [[1, "생각하고 답하기", "중간 단계 적기", "처음 보는 문제"], [2, "출발점", "pass@k", "가끔 나오는 정답"], [3, "검증 가능한 보상", "REINFORCE, GRPO", "그룹 안에서 비교"],
        [4, "신호 없는 그룹", "다 맞음, 다 틀림", "드문 보상"], [5, "단계별 보상", "생각도 채점", "처음부터일 때"], [6, "문제 고르기", "어려운 정도", "목록이 낡음"]]
S.save(ROOT / "topics" / "model" / "rl-reasoning-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow, "env": ENV})

why = """<p>o1, DeepSeek-R1, Olmo 3 Think 같은 <b>추론 모델</b>(LRM)은 답하기 전에 긴 \"생각\"을 써요. 이 능력은 보통 두 단계로 만들어요. 먼저 잘 푸는 모델이 쓴 풀이를 따라 배우고(<b>생각 SFT</b>), 그다음 정답을 기계로 확인할 수 있는 문제(수학, 코드)에서 <b>맞았는지만 보상으로 주는 강화학습</b>을 해요. 사람이 매번 점수를 매기지 않아도 돼서 크게 돌릴 수 있어요.</p>"""
why += K.table(["", "사람 선호로 학습 (RLHF)", "검증 가능한 보상 (RLVR)"], [["보상", "사람 선호를 흉내 낸 보상 모델의 점수", "채점기: 정답이면 1, 아니면 0"], ["잘 맞는 일", "말투, 도움 됨, 안전", "수학, 코드, 형식이 정해진 답"],
                                                                     ["어려운 점", "보상 모델을 속이기", "정답이 드물게 나옴, 답이 길어 비쌈"]])
why += "<p>이 쪽은 숫자 세 개를 더하는 아주 작은 문제로 이 과정을 그대로 재현해요. <a href=\"rl-deep.html\">심층 강화학습</a>의 REINFORCE, PPO를 알면 더 쉬워요.</p>"
whole = """<p>생각하고 답하기 → 출발점과 pass@k → 맞았는지만 보고 강화학습 → 신호 없는 그룹 → 단계별 보상 → 문제 고르기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/reasoning_rl/verifiable_rl.py</code> 하나예요. 모델이 작아서 CPU로 1분이면 돌아요.</p>"""
steps_html = [
    K.step(1, "생각하고 답하기", "(a + b + c)의 일의 자리를 바로 답하는 모델과, (a + b)를 먼저 적고 답하는 모델을 같은 풀이로 가르쳐 처음 보는 문제에서 비교해요.",
           "<p>\"생각\"은 답 전에 쓰는 중간 토큰이에요. 여기서는 중간값 하나지만, 추론 모델은 수천 토큰을 써요. 생각을 길게 쓰려면 긴 문맥을 다룰 수 있어야 해서, Olmo 3는 강화학습이 잘 되려면 데이터보다 구조(GQA, RoPE 설정)가 먼저였다고 해요(<a href=\"llm-architectures.html\">요즘 LLM의 부품</a>).</p>"
           + K.code_table([("Thinker.first(p)", "a, b → 중간 생각 (a + b의 일의 자리)"), ("Thinker.second(mid, p)", "중간 생각, c → 답"), ("sft(m, p, epochs)", "선생님 풀이(생각 + 답)를 따라 하기")]), "s1"),
    K.step(2, "출발점과 pass@k", "풀이 100개를 조금만 배운 모델을 강화학습의 출발점으로 두고, 답을 여러 번 뽑으면 얼마나 맞히는지 재요.",
           "<p><b>pass@k</b>는 답을 k번 뽑아 한 번이라도 맞으면 푼 것으로 치는 지표예요. pass@1과 pass@k의 차이가 \"강화학습이 끌어올릴 수 있는 여지\"예요.</p>"
           "<p>Olmo 3는 생각 SFT 다음에 <b>생각 DPO</b>도 해요. 같은 문제에 큰 모델과 작은 모델이 쓴 풀이를 짝지어 \"큰 모델 쪽이 낫다\"를 배우게 해요. 둘의 차이가 일정하면 둘 다 완벽하지 않아도 배울 신호가 된다는 생각이에요.</p>"
           + K.code_table([("p.repeat_interleave(k, dim=0)", "문제마다 k번"), ("hit.any(dim=1)", "한 번이라도 맞았나")]), "s2"),
    K.step(3, "맞았는지만 보고 강화학습: GRPO", "문제마다 답을 8개 뽑아 채점하고, 같은 문제의 답끼리 비교한 값으로 정책을 고쳐요.",
           "<p><b>GRPO</b>(Group Relative Policy Optimization)는 가치 함수(비평가) 없이, 같은 문제에 대한 답 여러 개의 평균을 기준값으로 써요. 장점 = (보상 − 그룹 평균) ÷ 그룹 표준편차. PPO처럼 비평가 신경망을 따로 학습하지 않아도 돼서 큰 LLM에서 많이 써요. 실제 구현은 여기에 PPO식 잘라내기(clipping)를 더해요.</p>"
           + K.out("f_curve")
           + K.code_table([("(a == answer(p)).float()", "검증 가능한 보상: 맞으면 1"), ("(r − r.mean(1)) / r.std(1)", "GRPO: 같은 문제의 답끼리 비교"), ("−(adv × logp).mean()", "정책 경사 손실 (생각과 답의 log 확률 합)")])
           + K.try_("문제마다 뽑는 답을 4, 8, 16개로 바꾸면?", "t3"), "s3"),
    K.step(4, "신호가 없는 그룹", "강화학습 동안 그룹 중 \"모두 맞음\"과 \"모두 틀림\"이 얼마나 되는지 세고, 출발점이 못 풀던 문제를 풀게 됐는지 봐요.",
           "<p>추론 모델의 강화학습이 어려운 가장 큰 이유가 이거예요. 답 하나가 길어서 뽑는 데 오래 걸리는데, 정답 신호는 드물어요. Olmo 3 강연도 \"생각 모드로 답 공간을 탐색 → 긴 추론 시간 → 정답 신호가 드묾 → 학습이 비효율적이고 불안정\"이라고 정리해요.</p>"
           + K.out("f_sig"), "s4"),
    K.step(5, "생각 단계도 채점하면", "중간 생각이 맞으면 0.5, 답이 맞으면 0.5를 주고, SFT 모델과 아무것도 안 배운 모델에서 각각 출발해 봐요.",
           "<p>답만 채점하는 걸 <b>결과 보상</b>, 단계마다 채점하는 걸 <b>과정(단계별) 보상</b>이라고 해요. 결과 보상은 만들기 쉽지만, 긴 풀이 중 어디가 틀렸는지 알려 주지 않아요. 생각 경로 여러 개를 나무처럼 펼쳐 탐색하는 방법(MCTS)도 단계별 점수가 있어야 쓸 수 있어요.</p>"
           + K.code_table([("0.5 × (mid == thought(p)) + 0.5 × (a == answer(p))", "생각과 답을 반씩 채점")]), "s5"),
    K.step(6, "어려운 정도로 문제 고르기", "출발점이 학습 문제마다 8번 풀어 보고, 가끔 맞히는 문제만 남겨 강화학습해요.",
           "<p>다 맞히는 문제와 다 틀리는 문제는 GRPO에 신호를 주지 않으니, 처음부터 빼면 계산을 아낄 수 있어요. 강연의 검색 에이전트(LNS)도 \"먼저 올바르게 행동하기 → 의미 있는 검색 → 정답 정확도\"로 단계를 나누고, 어려운 정도로 데이터를 골라 뽑는 답을 크게 줄였어요.</p>"
           + K.code_table([("rate = 8번 중 맞은 비율", "문제마다 어려운 정도"), ("train[(rate > 0) & (rate < 1)]", "가끔 맞히는 문제만")]), "s6"),
]
practice = [("작은 모델에 추론을 가르친다", "강화학습보다 좋은 풀이로 SFT부터 해요. 출발 모델이 가끔이라도 맞혀야 강화학습이 시작돼요. 출발점의 pass@k를 먼저 재 보면 강화학습으로 얼마나 오를지 짐작할 수 있어요."),
            ("강화학습을 돌린다", "\"모두 맞음\", \"모두 틀림\" 그룹 비율을 함께 기록해요. 이 비율이 높으면 문제 난이도나 그룹 크기를 바꿀 때예요. 정답률만 보면 왜 멈췄는지 알 수 없어요."),
            ("보상을 설계한다", "기계로 확인되는 보상(정답, 테스트 통과, 형식)부터 써요. 중간 단계를 채점할 수 있으면 신호가 드문 초반에 큰 도움이 돼요.")]
libs = [("①~⑥", "<code>torch</code>, <code>torch.distributions.Categorical</code>", "작은 정책 신경망, 답 뽑기와 log 확률")]
K.build(ROOT / "topics" / "model" / "rl-reasoning.html", title="추론 모델과 검증 가능한 보상", h1="추론 모델과 검증 가능한 보상: 생각을 가르치고, 맞았는지로 강화하기", track="모델",
        crumb="모델 &gt; 강화학습",
        one_line="중간 단계를 적게 하면(<b>생각</b>) 처음 보는 문제도 아는 단계로 나눠 풀어요. 풀이를 따라 배운 뒤(<b>생각 SFT</b>), 정답을 기계로 확인하는 보상으로 <b>GRPO</b>를 돌려요. 그런데 같은 문제의 답이 <b>모두 맞거나 모두 틀리면 신호가 없어서</b>, 출발 모델이 가끔이라도 맞히는 범위 안에서만 배워요. 생각 단계도 채점하면(<b>단계별 보상</b>) 그 한계가 줄어요.",
        why=why, whole=whole, steps=steps_html, code_file="examples/reasoning_rl/verifiable_rl.py", practice=practice, libs=libs,
        libs_note="<code>Thinker</code>, <code>Direct</code>, <code>generate</code>, <code>accuracy</code>, <code>sft</code>, <code>pass_at_k</code>, <code>rl</code>은 이 파일 안에서 정의했어요.",
        next_rows=[("<a href=\"rl-deep.html\">심층 강화학습</a>", "REINFORCE, 베이스라인, PPO"), ("<a href=\"rl-basics.html\">강화학습 기초</a>", "같은 카테고리"), ("<a href=\"llm-architectures.html\">요즘 LLM의 부품</a>", "긴 문맥, GQA")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: Upstage 10Talks Lecture 04 \"Lessons from Olmo 3\" 3부(강화학습의 어려움, 구조와 긴 문맥, 생각 SFT, 생각 DPO와 차이 학습), Lecture 09 \"Vertical Domain Adaptation for Smaller Reasoning Agent\"(추론 모델의 학습 흐름, 단계별 보상, 구조적 탐색, 단계별 커리큘럼 강화학습과 어려운 정도로 고르기). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/rl_reasoning.py</code>가 <code>examples/reasoning_rl/verifiable_rl.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
