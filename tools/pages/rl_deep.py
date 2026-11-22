"""모델 > 강화학습 2쪽: 심층 강화학습. examples/rl/deep_rl.py 하나가 ①~⑤로 자란다.
실행: python tools/pages/rl_deep.py  (CPU로 6분쯤)
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
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "rl"
code = S.parse((EX / "deep_rl.py").read_text(encoding="utf-8"))
ns = S.run(code, 5, S.workdir(EX), filename="deep_rl.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, NumPy {np.__version__}, CPU 1스레드 (seed 0, 1, 2)"
f1 = lambda x: f"{float(x):.1f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "rl"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/rl/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


def at(curves, xs):                                             # 씨앗별 곡선을 같은 걸음 수에서 읽어 평균
    return np.array([[np.interp(x, c[:, 0], c[:, 1]) for x in xs] for c in curves]).mean(axis=0)


outs = {}
# ---------------- ①
L = np.array(ns["lengths"])
S.claim(10 < L.mean() < 30, "① 무작위로 밀면 평균 10~30걸음")
outs["s1"] = S.box("① 무작위로 밀면 (1,000판)", [(f"평균 {L.mean():.1f}걸음, 가운데값 {np.median(L):.0f}걸음, 가장 오래 {L.max()}걸음 (최대 200)", 1)],
                   ["상태는 수레 위치, 속도, 막대 각도, 각속도의 실수 4개예요. 1쪽처럼 칸을 셀 수 없어서 Q를 표로 적을 수 없어요. 비슷한 상태끼리 값을 나눠 쓰는 함수가 필요해요."])

# ---------------- ② DQN
ab = ns["ablation"]
names = {(False, False): "둘 다 없음", (False, True): "타깃 네트워크만", (True, False): "경험 재생만", (True, True): "둘 다 (DQN)"}
xs = [5000, 10000, 15000, 20000]
tab = {k: at(v, xs) for k, v in ab.items()}
last5 = {k: [c[-5:, 1].mean() for c in v] for k, v in ab.items()}
S.claim(tab[(True, False)][-1] > 190 and tab[(False, False)][-1] < 120, "② 경험 재생만 쓰면 2만 걸음에 190 넘고, 둘 다 없으면 120 아래")
S.claim(tab[(True, True)][-1] < tab[(True, False)][-1], "② 이 문제에서는 타깃 네트워크를 더하면 오히려 낮다")
outs["s2"] = S.table("② 네 가지 구성, 씨앗 3개 평균 (막대를 버틴 걸음 수, 최대 200)", ["구성", "5천 걸음", "1만", "1만 5천", "2만", "씨앗별 마지막 5판"],
                     [[names[k], *[f1(x) for x in tab[k]], ", ".join(f1(x) for x in last5[k])] for k in ab],
                     ["책의 실험(작은 격자 게임)에서는 두 장치가 <b>함께</b> 있어야 안정됐어요. 이 막대 세우기에서는 <b>경험 재생이 결정적</b>이었어요. 둘 다 쓴 DQN은 7,500걸음쯤까지 경험 재생만 쓴 쪽과 비슷하게 올라갔다가, 그 뒤 150 근처로 내려앉았어요.",
                      "타깃 네트워크는 500걸음마다만 따라와서, 그 사이에 배운 것이 목표 값에 늦게 반영돼요. 이 작은 문제에서는 그 늦음이 득보다 실이었던 것으로 보여요. 씨앗 3개, 2만 걸음의 결과라 일반화하긴 어렵고, 어느 장치가 필요한지는 문제마다 달라서 이렇게 하나씩 빼 보며 확인해요(ablation).",
                      "타깃 네트워크만 쓴 경우 1만 5천 걸음쯤 크게 무너졌다가 다시 올라왔어요. 방금 겪은 몇 걸음에만 맞춰 배우면, 한 번 배운 것을 쉽게 잊어요(파국적 망각)."])
fig, ax = plt.subplots(figsize=(8.4, 3.9))
grid = np.linspace(1000, 20000, 60)
for (k, col) in zip(ab, (C["gray"], C["blue"], OK, FAIL)):
    ys = np.array([np.interp(grid, c[:, 0], c[:, 1]) for c in ab[k]])
    ax.plot(grid, ys.mean(0), color=col, lw=2.4, label=names[k])
    ax.fill_between(grid, ys.min(0), ys.max(0), color=col, alpha=.10)
ax.set(xlabel="환경과 주고받은 걸음 수", ylabel="버틴 걸음 수 (학습 중, 최근 판)", title="DQN의 두 장치를 하나씩 빼 보기 (씨앗 3개, 칠한 곳은 최소~최대)", ylim=(0, 210))
ax.legend(fontsize=9, loc="lower right")
P_DQN = fig_save(fig, "dqn_ablation")
outs["f_dqn"] = chart("경험 재생과 타깃 네트워크", P_DQN, "", ["보라(경험 재생만)는 8천 걸음쯤부터 200 근처에 붙어요. 회색(둘 다 없음)과 파랑(타깃만)은 오르내림이 커요.",
                                                         "y축은 ε-greedy로 탐색하면서 버틴 걸음 수라, 탐색을 끄면 조금 더 높아요."])

# ---------------- ③ REINFORCE와 베이스라인
sp = ns["spread"]
S.claim(sp[True][0] < sp[False][0] / 2, "③ 베이스라인을 빼면 기울기 흔들림이 절반 아래")
outs["s3"] = S.table("③ 같은 정책에서 기울기를 30번 재면 (8판씩 200걸음)", ["", "흔들림 (분산의 합)", "평균 기울기의 크기"],
                     [["수익 G 그대로", f"{sp[False][0]:.4f}", f"{sp[False][1]:.3f}"], ["G − 평균 (베이스라인)", f"{sp[True][0]:.4f}", f"{sp[True][1]:.3f}"]],
                     [f"베이스라인을 빼니 흔들림이 {sp[False][0] / sp[True][0]:.1f}배 줄었고, 평균 방향의 크기는 거의 같아요. 같은 방향을 덜 흔들리게 재는 거예요.",
                      "수익이 모두 양수(버틴 걸음마다 +1)라 베이스라인이 없으면 모든 행동의 확률을 올리려 하고, 그중 \"조금 더\" 좋았던 행동만 덜 올라가는 식으로 배워요. 평균을 빼면 평균보다 나은 행동은 올리고, 못한 행동은 내려요."])

# ---------------- ④⑤ A2C, PPO
pg = ns["pg_runs"]
pnames = {"reinforce": "REINFORCE", "reinforce_b": "REINFORCE + 베이스라인", "a2c": "A2C (5걸음)", "ppo": "PPO"}
xs2 = [10000, 20000, 40000, 60000]
ptab = {k: at(v, xs2) for k, v in pg.items()}
S.claim(ptab["ppo"][0] > 100 and ptab["ppo"][-1] > 190 and all(ptab["ppo"][0] > ptab[k][0] for k in pg if k != "ppo"), "⑤ PPO가 1만 걸음에 100 넘고 6만에 190 넘으며, 1만 걸음에서 가장 높다")
lam_rows = []
for lam in (0.0, 0.95, 1.0):
    if lam == ns["A2C_LAMBDA"]:
        lam_rows.append((lam, ptab["a2c"]))
        continue
    ns["A2C_LAMBDA"] = lam
    lam_rows.append((lam, at([ns["train_pg"]("a2c", sd) for sd in range(3)], xs2)))
ns["A2C_LAMBDA"] = 1.0
S.claim(lam_rows[0][1][-1] < 100 and lam_rows[2][1][-1] > 150, "④ 한 걸음 TD(λ = 0)는 6만 걸음에 100 아래, 5걸음 수익(λ = 1)은 150 위")
outs["s5"] = S.table("④⑤ 정책 경사 네 가지, 씨앗 3개 평균 (8판을 함께 굴림)", ["방법", "1만 걸음", "2만", "4만", "6만", "한 번 고칠 때 쓰는 걸음"],
                     [[pnames[k], *[f1(x) for x in ptab[k]], {"reinforce": "200 × 8판, 1번", "reinforce_b": "200 × 8판, 1번", "a2c": "5 × 8판, 1번", "ppo": "128 × 8판, 4번 다시"}[k]] for k in pg],
                     ["PPO가 가장 빨리, 가장 높이 올라가요. 같은 기록으로 네 번 다시 배워서 한 걸음의 경험을 더 알뜰하게 써요.",
                      "REINFORCE는 판이 끝나야 배워서 처음엔 느리지만 6만 걸음에는 180을 넘었어요. 이 문제에서는 평균을 빼는 베이스라인이 학습 곡선을 크게 바꾸지는 않았어요(③에서 흔들림은 줄었는데). 판 길이가 짧을 때부터 모든 판이 비슷하게 짧아서 평균만 빼도 신호가 크지 않기 때문이에요.",
                      "A2C는 5걸음마다 바로 고쳐서 빠르지만 오르내림이 커요. 씨앗마다 차이도 커서, 씨앗 3개 평균은 숫자 하나하나보다 흐름으로 읽어요."])
outs["t4"] = S.table("A2C의 이득(advantage)을 몇 걸음 보고 재나 (GAE의 λ)", ["λ", "1만 걸음", "2만", "4만", "6만"],
                     [[f"<b>{lam}</b> (지금)" if lam == 1.0 else lam, *[f1(x) for x in v]] for lam, v in lam_rows],
                     ["λ = 0은 한 걸음 TD 오차만 써요. 비평가(critic)가 아직 엉터리일 때 그 엉터리 값에 기대서, 6만 걸음 뒤에도 100을 못 넘었어요.",
                      "λ = 1은 5걸음 동안 실제로 받은 보상을 쓰고 맨 끝에만 비평가 값을 빌려요. 덜 기대는 만큼 처음엔 흔들려도 결국 잘 배웠어요. 그 사이 값(0.95)은 둘을 섞은 거예요."])
fig, ax = plt.subplots(figsize=(8.4, 3.9))
g2 = np.linspace(2000, 60000, 80)
for k, col in zip(pg, (C["gray"], C["blue"], FAIL, OK)):
    ys = np.array([np.interp(g2, c[:, 0], c[:, 1]) for c in pg[k]])
    ax.plot(g2, ys.mean(0), color=col, lw=2.4, label=pnames[k])
    ax.fill_between(g2, ys.min(0), ys.max(0), color=col, alpha=.08)
ax.set(xlabel="환경과 주고받은 걸음 수 (8판 합)", ylabel="버틴 걸음 수 (최근 20판)", title="정책 경사 네 가지 (씨앗 3개, 칠한 곳은 최소~최대)", ylim=(0, 210))
ax.legend(fontsize=9, loc="lower right")
P_PG = fig_save(fig, "policy_gradient")
outs["f_pg"] = chart("정책 경사 방법 비교", P_PG, "", ["보라(PPO)가 1만 걸음 안에 120을 넘고, 3만 걸음 뒤로는 180~200 사이에 머물러요.",
                                                   "DQN(②)은 한 판씩 굴렸고 이쪽은 8판을 함께 굴려서, 같은 걸음 수라도 고치는 횟수와 방식이 달라요. 두 그림을 숫자로 바로 비교하지는 않아요."])

flow = [[1, "막대 세우기", "실수 상태 4개", "무작위면 17걸음"], [2, "DQN", "경험 재생 + 타깃", "하나씩 빼 보기"], [3, "REINFORCE", "정책을 직접", "베이스라인"],
        [4, "A2C", "비평가가 돕기", "λ로 섞기"], [5, "PPO", "조금씩만 바꾸기", "가장 빠름"]]
S.save(ROOT / "topics" / "model" / "rl-deep-data.js", {"code": code, "names": S.names(code), "range": [1, 5], "outs": outs, "flow": flow, "env": ENV})

why = """<p><a href=\"rl-basics.html\">1쪽</a>에서는 칸이 32개라 Q를 표 한 장에 다 적었어요. 그런데 로봇의 관절 각도, 게임 화면, 바둑판처럼 상태가 실수이거나 경우의 수가 천문학적이면 표를 만들 수 없어요. 만들더라도 한 번도 안 가 본 칸은 아무 값도 모르고, \"비슷한 칸이면 값도 비슷하겠지\"라는 걸 쓸 수 없어요.</p>
<p>그래서 Q나 정책을 <b>신경망</b>으로 근사해요. 상태를 넣으면 행동마다의 값(또는 확률)이 나오는 함수예요. 딥러닝의 일반화 덕분에 처음 보는 상태도 다룰 수 있지만, 1쪽에는 없던 문제가 새로 생겨요.</p>"""
why += K.table(["", "가치 기반 (DQN)", "정책 기반 (REINFORCE, A2C, PPO)"], [["신경망이 내는 것", "행동마다의 Q", "행동마다의 확률"], ["행동 고르기", "Q가 가장 큰 것 (+ ε 탐색)", "확률대로 뽑기 (탐색이 저절로)"],
                                                                     ["다룰 수 있는 행동", "몇 개 중 고르기", "몇 개 중 고르기, 연속 값(핸들 각도)도"], ["지난 경험 다시 쓰기", "잘 돼요 (off-policy)", "거의 못 해요 (on-policy). PPO가 몇 번만"]])
whole = """<p>환경 → 가치를 신경망으로(DQN) → 정책을 신경망으로(REINFORCE) → 비평가 붙이기(A2C) → 조금씩만 바꾸기(PPO). 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/rl/deep_rl.py</code> 하나예요. 환경은 수레 위 막대 세우기라는 고전 제어 문제를 물리식으로 직접 짰어요(강화학습 연습에 오래 써 온 문제예요). CPU로 다 돌리면 5분쯤 걸려요.</p>"""
steps = [
    K.step(1, "막대 세우기 환경", "수레를 왼쪽이나 오른쪽으로 밀어 막대가 12도 넘게 기울지 않게 해요. 버틴 걸음마다 +1, 200걸음이면 성공이에요.",
           "<p>0.02초마다 수레에 ±10N의 힘을 줘요. 다음 상태는 뉴턴 역학 식으로 계산해요. 에이전트는 이 식을 모르고, 상태 4개와 보상만 봐요.</p>"
           "<p>8판을 한꺼번에 굴리는 이유는 속도예요. 정책 경사 방법(③~⑤)은 여러 판의 경험을 모아 한 번에 고치는데, numpy로 8판을 한 번에 계산하면 따로 굴리는 것보다 훨씬 빨라요.</p>"
           + K.code_table([("class Pole", "n판을 한꺼번에 굴리는 환경"), ("self.s", "(판 수, 4): 수레 위치, 수레 속도, 막대 각도, 막대 각속도"),
                           ("fell = (np.abs(x) > 2.4) | (np.abs(th) > 12 * math.pi / 180)", "수레가 2.4m 밖으로 나가거나 막대가 12도 넘게 기울면 실패"),
                           ("done = fell | (self.t >= 200)", "실패했거나 200걸음을 채우면 끝")]), "s1"),
    K.step(2, "Q를 신경망으로: 경험 재생과 타깃 네트워크 (DQN)", "1쪽의 Q-learning을 신경망으로 바꾸고, 학습을 안정시키는 장치 둘을 하나씩 빼며 효과를 재요.",
           "<p>1쪽 ⑥의 Q-learning 목표 <code>r + γ maxₐ' Q(s', a')</code>를 그대로 정답처럼 두고, 신경망 Q(s, a; θ)를 제곱 오차로 학습시켜요. 이렇게만 하면 두 가지가 어긋나요.</p>"
           "<ul><li><b>움직이는 과녁</b>: 목표 값을 계산하는 것도 같은 신경망이라, 한 번 고칠 때마다 목표도 같이 움직여요. 표와 달리 신경망은 한 상태를 고치면 비슷한 다른 상태 값도 같이 바뀌어서 더 심해요.</li>"
           "<li><b>붙어 있는 경험</b>: 방금 겪은 몇 걸음은 서로 아주 비슷해요(바로 앞 상태에서 이어지니까). 그걸로만 배우면 최근 상황에 맞춰 버리고, 전에 배운 걸 잊어요. 머신러닝이 가정하는 i.i.d.가 깨지는 거예요.</li></ul>"
           "<p><b>DQN</b>은 둘을 이렇게 막아요. <b>경험 재생</b>: 겪은 (s, a, r, s')을 버퍼에 쌓아 두고, 무작위로 64개씩 뽑아 배워요. 붙어 있던 경험이 섞이고, 한 경험을 여러 번 써요. <b>타깃 네트워크</b>: 목표 값은 가끔만(여기선 500걸음마다) 복사해 오는 별도 신경망으로 계산해요. 그동안은 과녁이 멈춰 있어요.</p>"
           + K.out("s2") + K.out("f_dqn")
           + "<p>⚠️ <b>죽음의 삼중주</b>: 남이 한 행동으로 배우기(off-policy) + 함수 근사 + 짐작으로 짐작 고치기(부트스트래핑)가 함께 있으면 값이 발산할 수 있다고 알려져 있어요. DQN은 셋을 다 가지고 있어서, 위 장치들로 위험을 줄이는 거예요.</p>"
           "<p><b>더 나아간 DQN들</b>: max가 우연히 큰 값을 골라 Q를 부풀리는 문제를, 고르기는 메인 신경망으로 하고 값은 타깃 신경망으로 재서 줄인 <b>Double DQN</b>. Q를 \"상태의 가치 V + 행동의 이득 A\"로 나눠 배우는 <b>Dueling DQN</b>. 오차가 큰 경험을 더 자주 뽑는 우선순위 재생. 이것들을 합친 Rainbow가 있어요.</p>"
           + K.code_table([("buf[n % len(buf)] = ...", "꽉 차면 가장 오래된 것부터 덮어쓰는 버퍼"), ("rng.integers(min(n, len(buf)), size=batch)", "버퍼에서 무작위로 64개"),
                           ("(q_target if target else q)(...).max(dim=1)", "목표 값을 타깃 신경망으로"), ("y = r + γ × max × (1 − 실패)", "넘어졌으면 그 뒤 가치는 0"),
                           ("q_target.load_state_dict(q.state_dict())", "500걸음마다 타깃을 메인으로 맞추기")]), None),
    K.step(3, "정책을 직접 배우기: REINFORCE와 베이스라인", "행동 확률을 내는 신경망을 두고, 수익이 컸던 행동의 확률을 올리는 쪽으로 고쳐요.",
           "<p>가치 기반은 \"값을 배운 뒤 가장 큰 행동 고르기\"였어요. <b>정책 경사</b>는 정책 π(a | s; θ) 자체를 배워요. 신경망이 행동마다 점수를 내고 softmax로 확률을 만들어요.</p>"
           "<p>목표는 기대 수익 J(θ)를 키우는 거예요. <b>정책 경사 정리</b>에 따르면 그 기울기는 <code>E[ Σₜ ∇ log π(aₜ | sₜ) × Gₜ ]</code>예요. \"그 행동의 log 확률을 올리는 방향 × 그 뒤에 받은 수익\"의 평균이에요. 환경의 식(전이 확률)이 들어 있지 않아서, 해 본 기록만으로 계산돼요(log 확률의 미분을 쓰는 이 요령을 우도비 요령이라고 해요).</p>"
           "<p><b>REINFORCE</b>는 판을 끝까지 해 본 뒤 실제 수익 Gₜ로 이 식을 계산해요(몬테카를로). 정직하지만 판마다 수익이 크게 흔들려서 기울기도 흔들려요. 그래서 수익에서 <b>베이스라인</b> b를 빼요. <code>∇ log π × (Gₜ − b)</code>. b가 행동에 따라 달라지지 않으면 평균 방향은 그대로고 흔들림만 줄어요.</p>"
           + K.out("s3")
           + K.code_table([("torch.distributions.Categorical(logits=policy(s))", "점수 → 확률 → 뽑기"), ("returns_to_go(R, D)", "뒤에서부터 Gₜ = rₜ + γGₜ₊₁"),
                           ("-(log_prob(A) * w).mean()", "이 값을 줄이면 log π × w가 커져요 (기울기 상승)"), ("G - G.mean()", "가장 단순한 베이스라인: 평균 수익")]), "s3"),
    K.step(4, "비평가 붙이기: 액터-크리틱 (A2C)", "가치를 재는 신경망(비평가)을 하나 더 두고, 판이 끝나기 전에 몇 걸음마다 정책(행위자)을 고쳐요.",
           "<p>가장 좋은 베이스라인은 그 상태의 가치 V(s)예요. 그러면 <code>Gₜ − V(sₜ)</code>는 \"그 행동이 평소보다 얼마나 나았나\", 곧 <b>이득</b>(advantage) A(s, a)가 돼요. V를 배우는 신경망을 <b>비평가</b>(critic), 정책 신경망을 <b>행위자</b>(actor)라고 해요.</p>"
           "<p>Gₜ를 판이 끝날 때까지 기다리지 않고, 1쪽 ⑤의 TD처럼 <code>rₜ + γV(sₜ₊₁)</code>로 바꾸면 매 걸음 배울 수 있어요. 대신 비평가가 틀리면 그만큼 치우쳐요. 그 사이를 <b>GAE</b>가 λ로 조절해요. λ = 0은 한 걸음 TD, λ = 1은 n걸음 동안 실제 보상을 쓰는 거예요.</p>"
           "<p><b>A2C</b>(Advantage Actor-Critic)는 여러 판을 동시에 굴려 n걸음씩 모은 뒤 한 번 고쳐요. 여러 판의 서로 다른 상황이 섞여서, DQN의 경험 재생과 비슷하게 붙어 있는 경험 문제를 줄여요.</p>"
           + K.code_table([("advantages(R, D, V, V_last, lam)", "TD 오차 δ = r + γV(s') − V(s)를 λ로 섞어 뒤에서부터 더하기"), ("A2C_LAMBDA = 1.0", "5걸음 실제 보상 + 5걸음 뒤 비평가 값"),
                           ("0.5 * ((target_v - critic(S)) ** 2).mean()", "비평가는 목표 값과의 제곱 오차로")])
           + K.try_("이득을 한 걸음 TD(λ = 0)로 재면?", "t4"), None),
    K.step(5, "조금씩만 바꾸기: PPO", "같은 기록으로 여러 번 배우되, 새 정책이 옛 정책보다 20% 넘게 달라진 몫은 점수에 안 쳐 줘요.",
           "<p>정책 경사는 지금 정책으로 모은 기록으로만 정확해요(on-policy). 한 번 고치고 나면 그 기록은 \"옛 정책의 것\"이라 다시 쓰기 어려워요. 그렇다고 한 번에 크게 고치면 정책이 망가져서, 다음 기록이 엉망이 되고 회복이 안 돼요.</p>"
           "<p><b>PPO</b>는 새 정책과 옛 정책의 확률 비 <code>ρ = π_new(a|s) ÷ π_old(a|s)</code>를 곱해서 옛 기록으로도 새 정책을 평가해요(중요도 가중). 그리고 ρ가 0.8~1.2를 벗어나 이득이 커지는 쪽은 잘라내요(clip). 그래서 같은 기록으로 여러 번(여기선 4번) 배워도 정책이 한 번에 멀리 가지 않아요.</p>"
           + K.out("s5") + K.out("f_pg")
           + "<p><b>연속 행동</b>: 핸들 각도처럼 실수를 고를 때는 신경망이 정규분포의 평균과 표준편차를 내고 거기서 뽑아요. 정책 경사 식은 그대로 써요. 가위바위보처럼 \"무작위로 섞는 게 최적\"인 상황도 확률을 직접 배우는 정책 기반만 표현할 수 있어요(<a href=\"rl-games.html\">3쪽</a>).</p>"
           "<p><b>일부만 보일 때</b>: 화면 한 장으로는 속도를 모르듯, 상태를 다 못 보면 최근 몇 개를 겹쳐 넣거나 RNN(순환 신경망)으로 지금까지 본 것을 기억하게 해요.</p>"
           "<p><b>오프라인 강화학습</b>: 환경에서 새로 해 볼 수 없고(의료, 자율주행), 이미 모아 둔 기록만 있을 때예요. 기록에 없던 행동의 Q를 부풀리기 쉬워서, 기록에 있는 행동 근처에 머물게 하는 장치를 더해요.</p>"
           + K.code_table([("ratio = torch.exp(log_prob_new - LP_old)", "새 정책 ÷ 옛 정책"), ("ratio.clamp(0.8, 1.2)", "20% 넘게 바뀐 몫 잘라내기"),
                           ("torch.min(ratio * adv, clipped * adv)", "둘 중 작은 쪽 = 비관적으로"), ("for _ in range(4)", "같은 기록으로 4번 다시 배우기")]), "s5"),
]
practice = [("어느 알고리즘부터", "행동이 몇 개 중 고르기면 DQN 계열이나 PPO, 연속 값이면 PPO나 SAC(Soft Actor-Critic) 같은 액터-크리틱부터 써요. PPO는 하이퍼파라미터에 덜 민감해서 첫 시도로 자주 써요."),
            ("학습이 갑자기 무너진다", "②의 파랑처럼 흔해요. 학습률을 낮추고, 경험 재생 버퍼를 키우고, 정책 경사는 한 번에 바뀌는 크기를 줄여요(PPO의 clip)."),
            ("보상을 표준화한다", "보상 크기가 크거나 들쭉날쭉하면 수익이나 이득을 평균 0, 표준편차 1로 맞춰요(⑤의 PPO가 이득에 그렇게 해요)."),
            ("결과 보고", "씨앗 3~10개 평균과 퍼짐을, x축은 환경 걸음 수로. 학습 중(탐색 포함) 점수와 탐색을 끈 평가 점수는 구분해서 적어요.")]
libs = [("①", "<code>numpy</code>", "물리식으로 만든 환경, 8판 동시 계산"), ("②~⑤", "<code>torch.nn</code>, <code>torch.optim.Adam</code>", "Q, 정책, 비평가 신경망"),
        ("③~⑤", "<code>torch.distributions.Categorical</code>", "점수 → 확률 → 뽑기, log 확률")]
K.build(ROOT / "topics" / "model" / "rl-deep.html", title="심층 강화학습", h1="심층 강화학습: DQN에서 PPO까지", track="모델",
        crumb="모델 &gt; 강화학습 2/4쪽 &nbsp;|&nbsp; 이전: <a href=\"rl-basics.html\" style=\"color:inherit\">강화학습 기초</a> &nbsp;|&nbsp; 다음: <a href=\"rl-games.html\" style=\"color:inherit\">게임과 균형</a>",
        one_line="상태가 너무 많으면 Q나 정책을 <b>신경망</b>으로 근사해요. 가치 쪽(<b>DQN</b>)은 경험 재생과 타깃 네트워크로, 정책 쪽은 <b>베이스라인</b>, <b>비평가</b>(A2C), <b>조금씩만 바꾸기</b>(PPO)로 학습을 안정시켜요.",
        why=why, whole=whole, steps=steps, code_file="examples/rl/deep_rl.py", practice=practice, libs=libs,
        libs_note="<code>Pole</code>은 클래스, <code>mlp</code>, <code>dqn</code>, <code>rollout</code>, <code>returns_to_go</code>, <code>grad_spread</code>, <code>advantages</code>, <code>train_pg</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rl-basics.html\">강화학습 기초</a>", "표로 하는 Q-learning, TD"), ("<a href=\"dl-training-stability.html\">학습을 안정시키기</a>", "학습률, Adam, 흔들림"),
                   ("<a href=\"rl-multiagent.html\">다중 에이전트 강화학습</a>", "여럿이 함께 배울 때")],
        sources=["다루는 범위: 부트캠프 딥러닝 기초를 강화학습 쪽으로 넓힌 쪽이에요. 수업에서 따로 다루지 않았어요",
                 "함께 정리한 자료: Albrecht, Christianos, Schäfer, <i>Multi-Agent Reinforcement Learning</i> 7장(함수 근사가 필요한 이유), 8장(움직이는 과녁, 죽음의 삼중주, 경험 재생, 타깃 네트워크, DQN과 그 하나씩 빼기 실험, Double, Dueling, 정책 경사 정리, REINFORCE, 베이스라인, 액터-크리틱, A2C, PPO, 여러 판 동시 학습, 관측과 이력). Prince, <i>Understanding Deep Learning</i> 19장 4~7절(적합 Q-러닝, Double Q-러닝, 우도비 요령, 베이스라인, 오프라인 강화학습). 막대 세우기 환경과 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/rl_deep.py</code>가 <code>examples/rl/deep_rl.py</code>를 단계별로 실제 실행한 결과 (씨앗 0, 1, 2)"])
print("ok")
