"""모델 > 평생 학습과 머신 언러닝. examples/continual/remember_forget.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/continual.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "continual"
code = S.parse((EX / "remember_forget.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="remember_forget.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, scikit-learn {sklearn.__version__} (CPU 1스레드, seed 0)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "continual"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/continual/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
acc = ns["acc"]
# ---------------- ①
st = ns["start"]
S.claim(st[0] > 0.95 and st[1] < 0.2, "① 일 A는 95% 넘게, 픽셀을 섞은 일 B는 거의 못 맞힘")
outs["s1"] = S.box("① 일 A(원래 숫자)만 배운 모델", [(f"학습 {len(ns['y_tr'])}장, 평가 {len(ns['y_te'])}장", 1), (f"일 A 정확도 → {st[0]:.3f}", 2), (f"일 B(픽셀을 섞은 같은 숫자) 정확도 → {st[1]:.3f}", 3, "newf")],
                   ["같은 숫자라도 픽셀 자리를 섞으면 모델에게는 전혀 다른 일이에요. 찍는 수준(0.1)에도 못 미쳐요. 이제 이 모델에게 일 B를 더 가르쳐요."])
fig, axs = plt.subplots(1, 2, figsize=(6.2, 3.0))
axs[0].imshow(ns["X_tr"][0].reshape(8, 8), cmap="Purples")
axs[0].set_title(f"일 A: 숫자 {ns['y_tr'][0]}")
axs[1].imshow(ns["X_tr"][0, ns["shuffle"]].reshape(8, 8), cmap="Purples")
axs[1].set_title(f"일 B: 같은 그림, 픽셀 섞음")
for a in axs:
    a.axis("off")
P_TASK = fig_save(fig, "two_tasks")
outs["f_task"] = chart("두 가지 일", P_TASK, "", ["오른쪽은 왼쪽 그림의 64픽셀을 정해진 순서로 섞은 거예요. 사람 눈에는 아무것도 아니지만, 숫자는 그대로라 같은 답을 내야 해요."])

# ---------------- ②~⑤
bd = ns["board"]
S.claim(bd["그냥 이어서"][0] < 0.4 and bd["리플레이 100장"][0] > 0.85 and bd["EWC λ = 1000"][0] > 0.85 and bd["처음부터 둘 다"][0] > 0.95, "② 그냥 이어서면 A가 0.4 아래, 리플레이와 EWC는 0.85 위, 함께는 0.95 위")
outs["s2"] = S.table("②~⑤ 일 A를 배운 뒤 일 B를 30에폭 배우면", ["방법", "일 A (예전 일)", "일 B (새 일)"], [[k, f3(a), f3(b)] for k, (a, b) in bd.items()],
                     [f"그냥 이어서 배우면 일 B는 {bd['그냥 이어서'][1]:.2f}로 잘하게 되지만, 일 A는 {bd['일 A만'][0]:.2f}에서 {bd['그냥 이어서'][0]:.2f}로 무너져요. 같은 가중치를 새 일에 맞춰 덮어쓰기 때문이에요(파국적 망각).",
                      f"일 A의 학습 데이터 {len(ns['y_tr'])}장 중 100장만 남겨 섞어도 {bd['리플레이 100장'][0]:.2f}를 지켜요. EWC는 예전 데이터를 하나도 안 쓰고 {bd['EWC λ = 1000'][0]:.2f}를 지키지만, 대신 일 B가 {bd['EWC λ = 1000'][1]:.2f}로 조금 덜 배워져요.",
                      "두 일을 처음부터 함께 배우는 게 상한이에요. 하지만 실제로는 예전 데이터를 다 가지고 있지 못하거나(개인정보, 저장 비용), 매번 처음부터 다시 배우기엔 너무 비싸서 평생 학습이 필요해요."])
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
ep = np.arange(1, len(ns["naive_curve"]) + 1)
for curve, nm, col in ((ns["naive_curve"], "그냥 이어서", FAIL), (ns["replay_curve"], "리플레이 100장", OK), (ns["ewc_curve"], "EWC λ = 1000", C["blue"])):
    c = np.array(curve)
    axs[0].plot(ep, c[:, 0], color=col, lw=2.4, label=nm)
    axs[1].plot(ep, c[:, 1], color=col, lw=2.4, label=nm)
axs[0].axhline(st[0], color=C["ink"], lw=1, ls=":", label="일 B를 배우기 전")
axs[0].set(xlabel="일 B를 배운 에폭", ylabel="정확도", title="예전 일 A", ylim=(0, 1.02))
axs[1].set(xlabel="일 B를 배운 에폭", ylabel="정확도", title="새 일 B", ylim=(0, 1.02))
axs[0].legend(fontsize=9, loc="lower left")
P_CURVE = fig_save(fig, "forgetting_curve")
outs["f_curve"] = chart("일 B를 배우는 동안 일 A는", P_CURVE, "",
                        [f"분홍(그냥 이어서)은 첫 에폭에 이미 일 A가 {ns['naive_curve'][0][0]:.2f}로 떨어지고, 일 B를 더 배울수록 계속 떨어져 {ns['naive_curve'][-1][0]:.2f}까지 가요.",
                         "보라(리플레이)와 파랑(EWC)은 일 A를 0.9 근처에서 지켜요. 오른쪽을 보면 일 B는 분홍과 보라가 몇 에폭 만에 거의 다 배우고, 파랑은 가중치가 묶인 만큼 조금 낮게 머물러요."])
rp_rows = []
for k in (20, 100, 300):
    m, _ = ns["replay_run"](k) if k != 100 else (ns["replay100"], None)
    rp_rows.append([f"<b>{k}</b> (지금)" if k == 100 else k, f3(acc(m, ns["A_test"])), f3(acc(m, ns["B_test"]))])
outs["t3"] = S.table("리플레이로 남겨 두는 장 수를 바꾸면", ["남긴 장 수", "일 A", "일 B"], rp_rows,
                     ["20장(일 A의 2%)만 남겨도 그냥 이어서보다 훨씬 덜 잊어요. 많이 남길수록 더 잘 지키지만, 저장과 계산이 늘어요."])
ewc_rows = []
for lam in (10, 100, 1000, 10000):
    m = ns["ewc1000"] if lam == 1000 else ns["train"](ns["copy"].deepcopy(ns["base"]), ns["B"], penalty=ns["ewc"](lam))[0]
    ewc_rows.append([f"<b>{lam}</b> (지금)" if lam == 1000 else lam, f3(acc(m, ns["A_test"])), f3(acc(m, ns["B_test"]))])
S.claim(float(ewc_rows[0][1]) < float(ewc_rows[2][1]) and float(ewc_rows[-1][2]) < float(ewc_rows[0][2]), "④ λ가 크면 A는 지키고 B는 덜 배움")
outs["t4"] = S.table("EWC의 λ를 바꾸면", ["λ", "일 A", "일 B"], ewc_rows,
                     ["λ가 작으면 그냥 이어서 배우기에 가깝고, 크면 가중치가 일 A 때 자리에 묶여 일 B를 덜 배워요. 예전 일과 새 일 사이의 저울이에요(안정성 대 가소성)."])

# ---------------- ⑥⑦
ul = ns["unlearn"]
S.claim(ul["처음부터 다시"]["지울 숫자 (test)"] == 0 and ul["남은 것으로 3에폭 더"]["지울 숫자 (test)"] > 0.3 and ul["경사 상승 30걸음"]["지울 숫자 (test)"] < 0.05 and ul["작업 벡터 빼기 (α = 4)"]["지울 숫자 (test)"] < 0.05, "⑥⑦ 다시 학습은 0, 남은 것으로 더는 30% 넘게 남고, 경사 상승과 작업 벡터는 5% 아래")
cols6 = list(next(iter(ul.values())))
outs["s6"] = S.table(f"⑥⑦ 숫자 {ns['FORGET']}을 지우기", ["방법", *cols6], [[k, *[f3(v[c]) for c in cols6]] for k, v in ul.items()],
                     [f"처음부터 지울 데이터 없이 다시 배운 모델이 정답이에요. 숫자 {ns['FORGET']}을 전혀 못 맞히고({ul['처음부터 다시']['지울 숫자 (test)']:.2f}) 나머지는 그대로예요. 하지만 큰 모델은 다시 배우는 데 몇 주와 큰돈이 들어요.",
                      f"남은 데이터로 조금 더 배우기만 하면 숫자 {ns['FORGET']}을 아직 {ul['남은 것으로 3에폭 더']['지울 숫자 (test)']:.0%} 맞혀요. 안 보여 주는 것만으로는 빨리 잊지 않아요.",
                      f"경사 상승(지울 데이터의 손실을 일부러 키우기)은 30걸음 만에 0으로 만들었지만, 나머지가 {ul['원래 모델']['나머지 (test)']:.3f}에서 {ul['경사 상승 30걸음']['나머지 (test)']:.3f}로 조금 떨어졌어요. 손실을 키우는 건 끝이 없어서, 너무 오래 하면 모델 전체가 망가져요.",
                      f"작업 벡터 빼기는 \"숫자 {ns['FORGET']}만 더 배운 모델 − 원래 모델\"을 그 방향으로 가는 화살표로 보고, 반대로 α배 움직여요. 학습 없이 덧셈 한 번이에요."])
neg_rows = [[f"<b>{a}</b> (지금)" if a == 4.0 else a, *[f3(ns["check"](ns["negate"](a))[c]) for c in cols6]] for a in (1.0, 2.0, 4.0, 8.0)]
outs["t7"] = S.table("작업 벡터를 몇 배 빼나 (α)", ["α", *cols6], neg_rows,
                     ["조금 빼면 덜 잊고, 많이 빼면 다 잊지만 나머지 숫자까지 흔들려요. 지울 데이터에 대한 성능과 나머지 성능 사이에서 α를 골라요."])
fig, ax = plt.subplots(figsize=(8.4, 3.6))
names6 = list(ul)
xs = np.arange(len(names6))
ax.bar(xs - 0.2, [ul[n]["지울 숫자 (test)"] for n in names6], width=0.4, color=FAIL, label=f"숫자 {ns['FORGET']} (지워야 함)")
ax.bar(xs + 0.2, [ul[n]["나머지 (test)"] for n in names6], width=0.4, color=OK, label="나머지 숫자 (지켜야 함)")
ax.set_xticks(xs, [n.replace(" (α = 4)", "\n(α = 4)").replace(" 30걸음", "\n30걸음").replace("으로 3에폭 더", "으로\n3에폭 더") for n in names6], fontsize=9)
ax.set(ylabel="test 정확도", title="잊게 하기: 분홍은 낮을수록, 보라는 높을수록 좋아요", ylim=(0, 1.05))
ax.legend(fontsize=9, loc="center right")
P_UL = fig_save(fig, "unlearning")
outs["f_ul"] = chart("네 가지 잊게 하기", P_UL, "", ["\"처음부터 다시\"와 같은 모양(분홍 0, 보라 그대로)에 가까울수록 잘 잊은 거예요."])

flow = [[1, "두 가지 일", "원래, 픽셀 섞음", "A만 배움"], [2, "그냥 이어서", "덮어쓰기", "파국적 망각"], [3, "리플레이", "예전 것 조금", "장 수"], [4, "EWC", "중요한 가중치 묶기", "λ"],
        [5, "비교", "함께 배우기가 상한", "안정성 대 가소성"], [6, "지워 주세요", "다시 학습이 정답", "안 보여 주기는 느림"], [7, "빠르게 잊기", "경사 상승, 작업 벡터", "α"]]
S.save(ROOT / "topics" / "model" / "continual-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p>모델은 한 번 배우고 끝나지 않아요. 새 데이터와 새 일이 계속 생기고(<b>평생 학습</b>, 지속 학습), 반대로 \"제 데이터를 지워 주세요\"처럼 배운 것의 일부를 없애야 할 때도 있어요(<b>머신 언러닝</b>). 둘 다 \"처음부터 다시 학습\"하면 정확하지만 큰 모델에서는 너무 비싸요.</p>"""
why += K.table(["", "평생 학습", "머신 언러닝"], [["원하는 것", "새로 배우면서 예전 것은 지키기", "골라서 잊고 나머지는 지키기"], ["어려운 점", "새 일을 배우면 예전 것을 덮어씀 (파국적 망각)", "안 보여 주기만으로는 잘 안 잊음, 잊었는지 확인도 어려움"],
                                             ["정답 (비쌈)", "모든 데이터로 처음부터", "지울 데이터를 빼고 처음부터"], ["이 쪽", "②~⑤", "⑥⑦"]])
why += "<p>이 쪽은 작은 손글씨 숫자 모델로 두 문제를 직접 재현해요. 개인정보를 외운 LLM에서 그 정보만 지우거나, 자율주행 모델에 새 도시의 데이터를 더하는 일이 같은 문제의 큰 버전이에요.</p>"
whole = """<p>두 가지 일 → 그냥 이어서 → 리플레이 → EWC → 비교 → 지워 달라는 요청 → 빠르게 잊기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/continual/remember_forget.py</code> 하나예요. 데이터는 scikit-learn에 들어 있어서 받을 파일이 없어요.</p>"""
steps = [
    K.step(1, "숫자와 두 번째 일", "8 × 8 손글씨 숫자로 일 A를 배우고, 같은 숫자의 픽셀을 섞은 일 B를 준비해요.",
           "<p>평생 학습에는 몇 가지 꼴이 있어요. 같은 답을 다른 입력에서 맞히는 <b>도메인 증가</b>(여기처럼, 또는 낮 사진 다음 밤 사진), 새 클래스가 계속 늘어나는 <b>클래스 증가</b>, 일마다 출력 머리가 따로 있는 <b>작업 증가</b>예요. 클래스 증가가 가장 어려워요.</p>"
           + K.out("f_task")
           + K.code_table([("load_digits()", "scikit-learn에 들어 있는 손글씨 숫자 1,797장"), ("X_tr[:, shuffle]", "64픽셀을 정해진 순서로 섞은 일 B"), ("nn.Linear(64, 128) → ReLU → nn.Linear(128, 10)", "작은 신경망 하나를 두 일이 함께 써요")]), "s1"),
    K.step(2, "그냥 이어서 배우면: 파국적 망각", "일 A를 배운 모델에게 일 B만 30에폭 가르치고, 에폭마다 일 A와 일 B 정확도를 재요.",
           "<p>경사하강은 지금 보는 데이터의 손실만 줄여요. 일 B를 배우는 동안 일 A의 데이터는 없으니, 일 A에 중요했던 가중치도 거리낌 없이 바뀌어요. 그래서 예전 일을 갑자기 크게 잊어요. 이걸 <b>파국적 망각</b>이라고 해요. <a href=\"rl-deep.html\">심층 강화학습</a>에서 경험 재생이 필요했던 이유도 같아요.</p>"
           + K.code_table([("train(copy.deepcopy(base), B, watch=(A_test, B_test))", "일 A 모델을 복사해 일 B로만 학습, 에폭마다 두 일을 채점")]), None),
    K.step(3, "예전 것을 조금 다시 보기: 리플레이", "일 A의 학습 데이터에서 100장만 남겨 두고, 일 B를 배울 때 배치마다 같은 수만큼 섞어요.",
           "<p>가장 단순하고 강한 방법이에요. 예전 데이터를 조금 저장해 두고(기억 버퍼) 새 데이터와 섞어 배워요. 예전 데이터를 저장할 수 없으면, 생성 모델로 예전 데이터를 닮은 가짜를 만들어 섞기도 해요(생성 리플레이).</p>"
           + K.code_table([("torch.randperm(...)[:k]", "일 A에서 k장 고르기"), ("torch.cat([x, replay[0][r]])", "배치마다 같은 수의 예전 데이터를 섞기")])
           + K.try_("남겨 두는 장 수를 20, 100, 300으로 바꾸면?", "t3"), None),
    K.step(4, "중요한 가중치는 덜 움직이게: EWC", "일 A에서 가중치마다 중요도(Fisher 정보)를 재 두고, 일 B를 배울 때 중요한 가중치가 멀리 가면 벌점을 줘요.",
           "<p>모든 가중치가 일 A에 똑같이 중요하지는 않아요. 조금만 바꿔도 일 A의 출력이 크게 흔들리는 가중치가 중요한 거예요. 그 정도를 재는 게 <b>Fisher 정보</b>예요. 모델이 내는 분포에서 답을 뽑아 그 log 확률의 기울기를 제곱해 평균 내요.</p>"
           "<p><b>EWC</b>(Elastic Weight Consolidation)는 일 B의 손실에 <code>λ/2 × Σ Fᵢ (θᵢ − θᵢ_A)²</code>를 더해요. 중요한 가중치는 고무줄로 일 A 때 자리에 묶이고, 덜 중요한 가중치는 일 B를 위해 자유롭게 움직여요. 예전 데이터를 저장하지 않아도 되는 게 장점이에요.</p>"
           "<p>이 밖에 예전 모델의 출력을 새 모델이 따라 하게 하는 LwF(<a href=\"model-compression.html\">모델 가볍게</a>의 지식 정제와 같은 생각), 일마다 작은 전문가를 두고 입력에 따라 골라 쓰는 방법(전문가 혼합)도 있어요.</p>"
           + K.code_table([("fisher_diag(m, data)", "가중치마다 기울기 제곱의 평균"), ("lam / 2 * Σ F × (θ − θ_A)²", "EWC 벌점: 중요한 가중치가 멀리 갈수록 커요")])
           + K.try_("λ를 10부터 10,000까지 바꾸면?", "t4"), None),
    K.step(5, "같이 놓고 비교하기", "다섯 가지 방법을 일 A, 일 B 정확도로 나란히 놓고, 일 B를 배우는 동안의 변화를 그림으로 봐요.",
           K.out("s2") + K.out("f_curve")
           + K.code_table([("joint", "두 일을 처음부터 함께 배운 모델 (상한)")]), None),
    K.step(6, "숫자 3을 지워 주세요: 처음부터 다시, 남은 것만 더", "일 A 모델에서 숫자 3에 대한 지식만 없애요. 지울 데이터 없이 처음부터 다시 배운 모델을 정답으로 두고 비교해요.",
           "<p>개인정보 삭제 요청(잊힐 권리), 저작권 문제, 잘못된 데이터, 위험한 지식처럼 모델에서 일부를 빼야 할 때가 있어요. 확실한 방법은 지울 데이터를 빼고 처음부터 다시 배우는 거예요. 머신 언러닝의 목표는 이 모델과 <b>구별되지 않는</b> 모델을 훨씬 싸게 얻는 거예요.</p>"
           "<p>잘 잊었는지는 세 가지로 봐요. 지울 데이터에서 성능이 다시 배운 모델만큼 떨어졌나, 나머지 성능은 지켰나, 그리고 모델의 반응으로 \"이 데이터가 학습에 쓰였는지\"를 알아낼 수 없나(멤버십 추론)예요. 여기서는 클래스 하나를 통째로 지워서 앞의 둘로 충분해요. 특정 사람의 몇 건만 지우는 경우는 성능 차이가 작아서 세 번째가 중요해져요.</p>"
           + K.code_table([("y_tr != FORGET", "숫자 3이 아닌 데이터 = 남길 데이터"), ("check(m)", "지울 숫자의 test 정확도, 나머지의 test 정확도, 지울 숫자의 학습 데이터 정확도")]), None),
    K.step(7, "빠르게 잊게 하기: 경사 상승과 작업 벡터 빼기", "지울 데이터의 손실을 일부러 키우는 방법과, \"지울 데이터 쪽으로 가는 가중치 방향\"을 빼는 방법을 비교해요.",
           "<p><b>경사 상승</b>: 지울 데이터에서는 손실을 줄이는 대신 키워요. 남길 데이터의 손실도 같이 줄여서 나머지가 무너지지 않게 잡아 줘요. LLM에서는 이걸 기본으로, 손실이 끝없이 커지지 않게 묶은 NPO 같은 변형을 써요.</p>"
           "<p><b>작업 벡터 빼기</b>: 지울 데이터만 더 배운 모델에서 원래 모델을 빼면 \"그 지식 쪽으로 가는 방향\"이 나와요(작업 벡터). 이걸 원래 모델에서 빼면 그 지식이 약해져요. 학습 없이 가중치 덧셈뿐이라 빠르고, 여러 설정으로 만든 작업 벡터들에서 부호가 같은 칸만 모아 빼면(NegMerge) 더 안정적이라는 연구도 있어요. LLM에서는 Fisher 정보로 지울 지식과 관련된 가중치만 골라 손대기도 해요.</p>"
           + K.out("s6") + K.out("f_ul")
           + K.code_table([("-cross_entropy(forget) + cross_entropy(retain)", "지울 것은 손실을 키우고, 남길 것은 줄이기"), ("p1 - p0", "작업 벡터: 지울 데이터로 더 배운 가중치 − 원래 가중치"),
                           ("p -= alpha * tv", "그 방향의 반대로 α배")])
           + K.try_("작업 벡터를 1, 2, 4, 8배 빼면?", "t7"), None),
]
practice = [("새 데이터가 계속 들어온다", "가장 먼저 리플레이를 써요. 예전 데이터 일부를 남기기 어렵다면 EWC 같은 규제나, 일마다 작은 어댑터(LoRA)를 따로 두는 방법을 봐요."),
            ("잊혀야 하는 데이터가 생겼다", "법적 요구라면 결국 다시 학습하는 게 가장 확실해요. 빠른 방법은 다시 학습할 때까지의 응급 처치로 쓰고, 지울 데이터, 남길 데이터, 멤버십 추론 셋으로 확인해요."),
            ("애초에 덜 외우게", "학습할 때 개인정보를 걸러내고, 중복을 없애고, 차분 프라이버시 같은 방법을 쓰면 나중에 지울 일이 줄어요.")]
libs = [("①", "<code>sklearn.datasets.load_digits</code>, <code>train_test_split</code>", "데이터"), ("②~⑦", "<code>torch</code>, <code>copy.deepcopy</code>", "학습, 모델 복사, 가중치 연산")]
K.build(ROOT / "topics" / "model" / "continual.html", title="평생 학습과 머신 언러닝", h1="평생 학습과 머신 언러닝: 잊지 않게, 잊게 하기", track="모델",
        crumb="모델 &gt; 추천과 평생 학습",
        one_line="새 일을 배우면 예전 일을 덮어써서 <b>파국적으로 잊어요</b>. 예전 데이터를 조금 섞거나(<b>리플레이</b>) 중요한 가중치를 묶으면(<b>EWC</b>) 덜 잊어요. 반대로 일부를 <b>골라 잊게</b> 할 때는 처음부터 다시 배운 모델을 정답으로 두고, <b>경사 상승</b>이나 <b>작업 벡터 빼기</b>로 싸게 흉내 내요.",
        why=why, whole=whole, steps=steps, code_file="examples/continual/remember_forget.py", practice=practice, libs=libs,
        libs_note="<code>model</code>, <code>acc</code>, <code>train</code>, <code>replay_run</code>, <code>fisher_diag</code>, <code>ewc</code>, <code>check</code>, <code>gradient_ascent</code>, <code>negate</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"dl-training-stability.html\">학습을 안정시키기</a>", "규제, weight decay"), ("<a href=\"model-compression.html\">모델 가볍게</a>", "지식 정제"), ("<a href=\"recsys.html\">추천 시스템</a>", "같은 카테고리")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: Upstage 10Talks Lecture 03 \"Towards Sustainable Machine Learning: Continual Learning and Machine Unlearning\"(파국적 망각, 다시 학습과 미세조정과 평생 학습의 한계, 일마다 전문가를 두고 분포로 골라 쓰기, Fisher 정보로 지울 지식의 가중치 찾기와 그 보정, LoRA 근사, 작업 벡터 빼기와 부호 일관성으로 합치기, TOFU 같은 언러닝 평가). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/continual.py</code>가 <code>examples/continual/remember_forget.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
