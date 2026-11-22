"""모델 > 강화학습 4쪽: 다중 에이전트 강화학습. examples/rl/multi_agent.py 하나가 ①~④로 자란다.
실행: python tools/pages/rl_multiagent.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "rl"
code = S.parse((EX / "multi_agent.py").read_text(encoding="utf-8"))
ns = S.run(code, 4, S.workdir(EX), filename="multi_agent.py")
ENV = f"Python {sys.version.split()[0]}, NumPy {np.__version__}"
f1 = lambda x: f"{float(x):.1f}"  # noqa: E731
f2 = lambda x: f"{float(x):.2f}"  # noqa: E731

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


outs = {}
CL = ns["CLIMB"]
# ---------------- ①
cb = ns["climb"]
S.claim(cb["central"][(0, 0)] == 100 and cb["independent"].get((0, 0), 0) == 0 and cb["vdn"].get((0, 0), 0) == 0, "① 중앙은 100번 모두 (0, 0), 각자와 가치 분해는 한 번도 못 찾음")
kn = {"independent": "각자 배우기 (독립 Q-learning)", "vdn": "가치 분해 (VDN, Q₁ + Q₂)", "central": "함께 배우기 (결합 행동 하나의 Q)"}
outs["s1"] = S.table("① 오르막 게임을 5,000판씩, 씨앗 100개 (마지막에 고르는 결합 행동)", ["방식", "고른 결합 행동 (횟수)", "그때 보상"],
                     [[kn[k], ", ".join(f"({a}, {b}) {n}번" for (a, b), n in cb[k].most_common()), ", ".join(str(CL[a, b]) for (a, b), _ in cb[k].most_common())] for k in kn],
                     ["가장 좋은 (0, 0)의 11은 바로 옆에 −30이 붙어 있어요. 각자 배우면 \"내가 0을 냈을 때\"의 평균을 보는데, 상대가 아직 탐색 중이라 자주 1이나 2를 내니 0의 평균이 −30에 끌려 내려가요. 그래서 둘 다 안전한 2(평균이 0 아래로 안 내려가는)로 가요.",
                      "결합 행동 9칸 각각의 값을 따로 배우면 (0, 0)이 11이라는 걸 그대로 알아요. 대신 에이전트가 늘면 결합 행동 수가 곱으로 늘어요(④).",
                      "가치 분해 Q₁(a₁) + Q₂(a₂)도 실패했어요. 더하기로는 \"(0, 0)은 11인데 (0, 1)은 −30\"처럼 둘의 조합에 따라 크게 뒤집히는 값을 표현할 수 없어요."])

# ---------------- ②
qt, uni, p2 = ns["q_trace"], ns["uniform"], ns["partner_on_2"]
S.claim(p2[1] > p2[2] and qt[-1, 1] < qt[-1, 2], "② 상대가 거의 2만 낼 때 행동 1이 진짜로 더 좋은데, 기억하는 Q(1)은 Q(2)보다 낮다")
outs["s2"] = S.table("② 0번 에이전트가 기억하는 Q와, 상대 정책에 따른 진짜 기대 보상", ["", "행동 0", "행동 1", "행동 2"],
                     [["상대가 셋을 1/3씩 낼 때 (진짜)", *[f2(x) for x in uni]], ["상대가 거의 2만 낼 때 (진짜)", *[f2(x) for x in p2]],
                      ["5,000판 뒤 기억하는 Q", *[f2(x) for x in qt[-1]]]],
                     [f"상대가 거의 2만 내는 지금, 내가 1을 내면 {p2[1]:.2f}로 2({p2[2]:.2f})보다 나아요(행 1, 열 2의 6점). 그런데 내 Q(1)은 {qt[-1, 1]:.2f}에 머물러 있어요. 상대가 아무거나 내던 시절에 배운 값이고, 그 뒤로 1을 거의 안 내서 고칠 기회가 없었어요.",
                      "상대의 정책이 바뀌면, 나에게는 <b>환경이 바뀐 것</b>과 같아요. 1쪽의 Q-learning은 환경이 바뀌지 않는다(정상성)고 가정해서 이런 수렴 보장이 깨져요. 이게 <b>비정상성</b>, 또는 움직이는 과녁 문제예요."])
fig, ax = plt.subplots(figsize=(8.4, 3.8))
for k, col in enumerate((OK, FAIL, C["blue"])):
    ax.plot(qt[:, k], color=col, lw=2, label=f"Q(행동 {k})")
ax.axvline(3000, color=C["gray"], lw=1, ls=":")
ax.text(3050, 8, "탐색이 0.05까지 줄어든 때", fontsize=9, color=C["gray"])
ax.set(xlabel="판 수", ylabel="0번 에이전트의 Q", title="상대가 배우는 동안 내 Q가 따라 움직여요 (독립 Q-learning, 씨앗 0)", ylim=(-20, 12))
ax.legend(fontsize=9, loc="lower left")
P_Q = fig_save(fig, "independent_q_trace")
outs["f_q"] = chart("움직이는 과녁", P_Q, "", ["처음엔 상대가 아무거나 내서 Q(0)과 Q(1)이 −30에 자주 끌려 떨어져요. 상대가 2에 자리 잡은 뒤에는 Q(2)만 5로 올라가고, 거의 안 고르는 Q(0), Q(1)은 가끔 탐색할 때만 계단처럼 고쳐져서, 상대가 바뀐 지금의 진짜 값을 따라가지 못해요."])

# ---------------- ③
gr = ns["grid"]
gn = {"central": "중앙 (결합 행동 36개)", "independent": "각자 (독립 Q-learning)", "vdn": "가치 분해 (VDN)"}
fin = {k: [int(c[-1, 1]) for c in v] for k, v in gr.items()}
S.claim(np.mean(fin["central"]) < np.mean(fin["independent"]) and all(x <= 11 for v in fin.values() for x in v), "③ 중앙이 가장 짧고, 셋 다 5번 모두 11걸음 안에 성공")
marks = [5, 10, 15, len(gr["central"][0])]
outs["s3"] = S.table(f"③ 상자를 함께 들기까지 걸린 걸음 수 (탐색 없이 평가, 씨앗 5개 평균, 최단 {ns['shortest']}걸음)", ["방식", *[f"{m * 200:,}판" for m in marks], "씨앗별 마지막"],
                     [[gn[k], *[f1(np.mean([c[m - 1, 1] for c in v])) for m in marks], ", ".join(str(x) for x in fin[k])] for k, v in gr.items()],
                     [f"61은 60걸음 안에 못 든 판이에요. 둘뿐이면 결합 행동이 36개라 중앙 학습도 감당할 만해서, 가장 빨리, 가장 짧게(평균 {np.mean(fin['central']):.1f}걸음) 배웠어요.",
                      f"각자 배우기와 가치 분해도 6,000판 뒤에는 모두 성공했지만 몇 걸음 돌아가요(평균 {np.mean(fin['independent']):.1f}, {np.mean(fin['vdn']):.1f}). 이 문제에서는 가치 분해가 각자 배우기보다 낫지 않았어요. 가치 분해가 이득을 보는 건 에이전트가 많아 중앙 학습을 못 쓰고, 팀 보상만 있어서 \"누구 덕인지\" 나눠야 할 때예요."])
fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
for k, col in zip(gr, (OK, C["blue"], FAIL)):
    xs = np.linspace(0, 1, len(gr[k][0]))
    ys = np.array([c[:, 1] for c in gr[k]])
    axs[0].plot(np.arange(1, len(xs) + 1) * 200, ys.mean(0), color=col, lw=2.4, label=gn[k])
axs[0].axhline(ns["shortest"], color=C["ink"], lw=1, ls=":", label="최단 6걸음")
axs[0].set(xlabel="판 수", ylabel="평가 판에서 걸린 걸음 (61 = 실패)", title="함께 들기를 배우는 속도 (씨앗 5개 평균)", ylim=(0, 64))
axs[0].legend(fontsize=9)
N, BOX = ns["N"], ns["BOX"]
ax = axs[1]
ax.add_patch(Rectangle((BOX[1] - .5, BOX[0] - .5), 1, 1, color="#ffe8cc"))
ax.text(BOX[1], BOX[0], "상자", ha="center", va="center", fontsize=10)
for (r, c), nm, col in zip(ns["STARTS"], ("로봇 1", "로봇 2"), (OK, FAIL)):
    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color=col, alpha=.25))
    ax.text(c, r, nm, ha="center", va="center", fontsize=9)
for (dr, dc) in ns["MOVES"][:4]:
    ax.add_patch(Rectangle((BOX[1] + dc - .5, BOX[0] + dr - .5), 1, 1, fill=False, ec=C["green"], lw=1.6, ls="--"))
ax.set(xlim=(-.5, N - .5), ylim=(N - .5, -.5), title="6 × 6 격자 (초록 점선: 들 수 있는 칸)")
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.grid(False)
P_G = fig_save(fig, "lift_box")
outs["f_grid"] = chart("둘이 함께 드는 상자", P_G, "",
                       ["두 로봇이 상자 옆 칸(초록 점선)에 서서 같은 걸음에 함께 \"들기\"를 해야 +1을 받고 끝나요. 한쪽만 들려고 하면 −0.3이에요. 보상은 팀이 함께 받아요(공통 보상).",
                        "보라(중앙)가 1,000판 안쪽에 최단 근처로 내려와요. 파랑과 분홍은 3,000판 근처부터 10걸음 안쪽으로 내려오고, 그 뒤에도 가끔 다시 길어져요."])

# ---------------- ④
sz = ns["sizes"]
S.claim(sz[5]["중앙 (결합 행동)"] / sz[5]["각자 또는 가치 분해"] > 200, "④ 다섯이면 중앙 표가 200배 넘게 크다")
outs["s4"] = S.table("④ 로봇 수에 따른 Q 표의 칸 수 (한 로봇이 설 수 있는 칸 35개, 행동 6개)", ["로봇 수", "결합 행동 수", "중앙 Q 표", "각자 또는 가치 분해 Q 표", "배수"],
                     [[n, f"{6 ** n:,}", f"{v['중앙 (결합 행동)']:,}", f"{v['각자 또는 가치 분해']:,}", f"{v['중앙 (결합 행동)'] / v['각자 또는 가치 분해']:,.0f}배"] for n, v in sz.items()],
                     ["상태(모두의 위치 조합)는 둘 다 로봇 수에 따라 거듭제곱으로 늘어요. 그래서 실제로는 표 대신 신경망을 써요(2쪽). 차이는 행동 쪽이에요. 중앙은 결합 행동이 6ⁿ개라 n이 하나 늘 때마다 6배, 각자는 n × 6개라 거의 그대로예요.",
                      "다섯이면 결합 행동이 7,776개예요. 신경망이 행동마다 값을 하나씩 내야 하니, 출력이 7,776개인 신경망을 만들고 그중 최대를 찾아야 해요. 각자 배우기나 가치 분해는 로봇마다 6개만 내면 돼요."])

flow = [[1, "오르막 게임", "각자 vs 함께", "−30이 무서워서"], [2, "움직이는 과녁", "상대가 배우는 중", "옛 Q에 갇힘"], [3, "함께 들기", "중앙, 각자, 분해", "둘이면 중앙"], [4, "규모", "6ⁿ 대 n × 6", "분해가 필요"]]
S.save(ROOT / "topics" / "model" / "rl-multiagent-data.js", {"code": code, "names": S.names(code), "range": [1, 4], "outs": outs, "flow": flow, "env": ENV})

why = """<p>창고 로봇 수십 대, 도로 위 자율주행차들, 게임 속 한 팀처럼 <b>여럿이 같은 환경에서 동시에 배우는</b> 경우가 많아요. 1~2쪽의 강화학습을 그대로 각자 돌리면 될 것 같지만, 상대도 배우면서 바뀌어서 \"환경은 그대로\"라는 가정이 깨져요. <a href=\"rl-games.html\">3쪽</a>의 게임 이론이 \"무엇이 해인가\"를 정했다면, 이 쪽은 그 해를 <b>강화학습으로 어떻게 배우나</b>예요.</p>"""
why += K.table(["어려움", "뜻", "이 쪽"], [["비정상성", "다른 에이전트가 배우는 동안 내 입장에서는 환경이 계속 바뀜", "②"], ["균형 선택", "좋은 균형과 안전한 균형 중 어디로 갈지", "①"],
                                       ["신용 할당", "팀 보상을 받았을 때 누구의 행동 덕인지", "③, 가치 분해"], ["규모", "에이전트가 늘면 결합 행동이 곱으로 늘어남", "④"]])
whole = """<p>상태 없는 협력 게임에서 세 방식 비교 → 비정상성 들여다보기 → 상태가 있는 협력 문제 → 규모. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/rl/multi_agent.py</code> 하나예요. 표 Q-learning만 써서 30초 안쪽에 끝나요. 신경망으로 넓힌 알고리즘들은 각 단계 끝과 마지막의 표에 정리했어요.</p>"""
steps = [
    K.step(1, "각자 배우기 vs 함께 배우기: 오르막 게임", "둘이 같은 보상을 받는 게임에서 셋을 비교해요. 각자 자기 Q만 배우기, 결합 행동 하나의 Q를 배우기, 팀 Q를 둘의 합으로 나눠 배우기.",
           "<p>다중 에이전트 문제는 3쪽의 게임 모델(행렬 게임, 확률적 게임, POSG)에 학습 목표(어떤 해)를 붙인 거예요. 여기서는 모두가 같은 보상을 받는 협력 게임이라 목표가 분명해요. 팀 보상을 최대로요.</p>"
           "<p>가장 쉬운 두 방법은 단일 에이전트 강화학습으로 바꾸는 거예요. <b>중앙 학습</b>: 모두를 한 에이전트로 보고 결합 행동 (a₁, a₂)를 고르게 해요. <b>독립 학습</b>: 각자 다른 에이전트를 환경의 일부로 보고 자기 Q만 배워요. 독립 학습은 단순하고 에이전트 수에 잘 버티지만, ②의 비정상성을 무시해요.</p>"
           + K.table(["행동 (나 \\ 상대)", "0", "1", "2"], [[str(i), *[str(v) for v in CL[i]]] for i in range(3)])
           + K.out("s1")
           + "<p>각자 배우기가 안전한 쪽으로 빠지는 이 현상은 <b>균형 선택</b>의 한 예예요. (0, 0)과 (2, 2)는 둘 다 균형에 가깝지만, 모두가 조금씩 탐색하는 동안에는 위험이 적은 쪽이 더 좋아 보여요.</p>"
           + K.code_table([("Qj[a1, a2]", "중앙: 결합 행동 9칸마다 값"), ("Q1[a1] += alpha * (r - Q1[a1])", "각자: 상대가 없는 것처럼"), ("td = r - (Q1[a1] + Q2[a2])", "가치 분해: 두 Q의 합이 팀 보상을 맞히게")]), None),
    K.step(2, "움직이는 과녁: 상대가 배우는 동안 내 Q가 흔들려요", "①의 각자 배우기에서 한 에이전트의 Q가 판마다 어떻게 바뀌는지 그리고, 상대 정책에 따른 진짜 값과 비교해요.",
           K.out("s2") + K.out("f_q")
           + "<p>이걸 줄이는 방법들: 학습 중에만 상대의 행동이나 정책을 함께 보는 것(아래의 CTDE), 상대의 정책을 모델로 만들어 그에 대한 최적 응답을 배우는 것(3쪽의 가상 플레이를 신경망으로 넓힌 에이전트 모델링), 2쪽의 경험 재생 버퍼를 짧게 두는 것(옛 상대 정책의 경험이 덜 섞이게)이 있어요.</p>", None),
    K.step(3, "상태가 있는 협력: 둘이 함께 드는 상자", "6 × 6 격자에서 두 로봇이 상자 옆으로 걸어가 같은 걸음에 함께 들어야 해요. ①의 세 방식을 상태가 있는 문제로 비교해요.",
           "<p>상태가 있으면 3쪽의 확률적 게임이에요. 각 방식의 Q 목표는 1쪽 Q-learning과 같은 꼴이에요.</p>"
           + K.table(["방식", "배우는 것", "목표 값", "실행할 때"], [["중앙", "Q(s, a₁, a₂)", "r + γ max over (a₁', a₂') Q", "한 곳에서 결합 행동을 정해 나눠 줌"],
                                                             ["각자", "Q₁(s, a₁), Q₂(s, a₂)", "각자 r + γ max Qᵢ", "각자 자기 Q로"],
                                                             ["가치 분해 (VDN)", "Q₁(s, a₁) + Q₂(s, a₂) = 팀 Q", "r + γ (max Q₁ + max Q₂)", "각자 자기 Q로"]])
           + K.out("s3") + K.out("f_grid")
           + "<p><b>학습과 실행을 나누기 (CTDE)</b>: 실행할 때는 각 로봇이 자기 관측만으로 움직여야 하는 경우가 많아요(통신이 없거나 느림). 하지만 학습은 시뮬레이터에서 하니, 학습할 때만은 모두의 정보를 써도 돼요. 이걸 중앙 학습, 분산 실행(CTDE)이라고 해요. VDN이 그 예예요. 학습은 팀 Q 하나로 하고, 실행은 각자 Qᵢ의 max로 해요.</p>"
           "<p><b>가치 분해가 성립하는 조건 (IGM)</b>: \"각자 자기 Qᵢ를 최대로 하는 행동들을 모으면 팀 Q를 최대로 하는 결합 행동이 된다\"가 맞아야 각자 실행해도 팀에 최선이에요. 더하기(VDN)는 이걸 늘 만족하지만 표현력이 약해서 ①의 오르막 게임을 못 나타내요. <b>QMIX</b>는 더하기 대신 \"각 Qᵢ가 커지면 팀 Q도 커지는\"(단조) 신경망으로 섞어서 더 많은 경우를 나타내요. QTRAN, QPLEX는 그 제약도 더 풀었어요.</p>"
           "<p><b>정책 기반에서는</b>: 행위자는 각자 자기 관측만 보고, 비평가는 학습 중에 모두의 관측과 행동을 보는 <b>중앙 비평가</b>를 써요(MADDPG, MAPPO). <b>COMA</b>는 \"나만 다른 행동을 했다면 팀 Q가 얼마였나\"를 빼서 내 몫의 이득을 재요(반사실 베이스라인). 팀 보상을 누구 덕인지 나누는 <b>신용 할당</b>의 한 방법이에요.</p>"
           + K.code_table([("sid(p1, p2)", "두 로봇의 위치 조합 → 상태 번호 (35 × 35 = 1,225개)"), ("step(p1, p2, a1, a2)", "같은 걸음에 둘 다 상자 옆에서 \"들기\"면 +1"),
                           ("Q1[s, a1] += alpha / 2 * td", "가치 분해에서 두 Q를 반씩 고쳐 합이 α만큼 움직이게")]), None),
    K.step(4, "에이전트가 늘면: 표의 크기", "로봇 수가 2에서 5로 늘 때, 결합 행동을 다루는 중앙 방식과 각자 다루는 방식의 표 크기를 계산해요.",
           K.out("s4")
           + "<p><b>파라미터 공유</b>: 로봇들이 모두 같은 종류면 신경망 하나를 함께 쓰고, 입력에 \"몇 번 로봇인지\"를 붙여요. 배울 가중치가 로봇 수와 상관없이 하나라서 많은 에이전트로 늘리기 쉬워요. 경험을 서로 나눠 쓰는 방법(SEAC)도 있어요.</p>"
           "<p><b>경쟁에서는 셀프 플레이</b>: 바둑이나 스타크래프트처럼 두 편이 겨루면, 자기 자신(또는 지난 버전들)과 두면서 배워요. AlphaZero는 몬테카를로 트리 탐색(MCTS)으로 수를 미리 읽고, 그 결과로 정책과 가치 신경망을 가르치는 셀프 플레이를 했어요. 자기와만 두면 가위바위보처럼 한 전략에만 맞춰지는 순환에 빠질 수 있어서, 여러 버전의 정책을 모아 그 모음에 대한 최적 응답을 계속 더하는 <b>PSRO</b>(AlphaStar가 쓴 리그 방식의 바탕)도 있어요.</p>", "s4"),
]
practice = [("어느 방법부터", "에이전트가 둘셋이고 시뮬레이터가 있으면 중앙 학습이나 중앙 비평가(MAPPO)부터. 많고 같은 종류면 파라미터를 공유한 독립 PPO(IPPO)가 의외로 강한 기준선이에요."),
            ("실험 환경", "협력: 레벨 기반 채집(LBF), 창고 로봇(RWARE), 스타크래프트 다중 에이전트 챌린지(SMAC), 하나비, 오버쿡드. 섞임: 다중 에이전트 파티클 환경(MPE), Melting Pot. 모음: PettingZoo, OpenSpiel."),
            ("평가와 보고", "씨앗 여러 개의 평균과 퍼짐, x축은 환경 걸음 수. 알고리즘마다 하이퍼파라미터를 같은 예산으로 찾아야 공정해요. 보상을 표준화하고, 관측이 일부만 보이면 RNN이나 최근 관측 겹치기를 써요."),
            ("팀 보상만 있다", "각자 배우기는 \"내가 잘해서인지 동료 덕인지\"를 몰라 게을러지기 쉬워요(게으른 에이전트). 가치 분해나 COMA 같은 신용 할당을 써요.")]
libs = [("①~④", "<code>numpy</code>, <code>collections.Counter</code>", "표 Q-learning, 결과 세기")]
K.build(ROOT / "topics" / "model" / "rl-multiagent.html", title="다중 에이전트 강화학습", h1="다중 에이전트 강화학습: 여럿이 함께 배우기", track="모델",
        crumb="모델 &gt; 강화학습 4/4쪽 &nbsp;|&nbsp; 이전: <a href=\"rl-games.html\" style=\"color:inherit\">게임과 균형</a>",
        one_line="여럿이 동시에 배우면 서로가 서로의 환경이라 <b>움직이는 과녁</b>이 돼요. 모두를 하나로 묶으면(<b>중앙</b>) 정확하지만 결합 행동이 곱으로 늘고, <b>각자</b> 배우면 가볍지만 손발이 안 맞아요. 그 사이에 학습만 함께 하는 <b>CTDE</b>와 <b>가치 분해</b>가 있어요.",
        why=why, whole=whole, steps=steps, code_file="examples/rl/multi_agent.py", practice=practice, libs=libs,
        libs_note="<code>eps_at</code>, <code>matrix_learn</code>, <code>q_if_partner</code>, <code>sid</code>, <code>next_to_box</code>, <code>step</code>, <code>grid_learn</code>, <code>table_sizes</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rl-games.html\">게임과 균형</a>", "목표가 되는 해"), ("<a href=\"rl-deep.html\">심층 강화학습</a>", "표 대신 신경망 (IDQN, IPPO의 바탕)"), ("<a href=\"rl-basics.html\">강화학습 기초</a>", "Q-learning")],
        sources=["다루는 범위: 강화학습을 여러 에이전트로 넓힌 쪽이에요. 수업에서 따로 다루지 않았어요",
                 "함께 정리한 자료: Albrecht, Christianos, Schäfer, <i>Multi-Agent Reinforcement Learning</i> 1장(다중 에이전트 시스템, 응용, 어려움), 5장(학습 과정, 수렴의 종류, 중앙 학습과 독립 학습, 비정상성, 균형 선택, 신용 할당, 규모, 셀프 플레이), 6장(결합 행동 학습, 에이전트 모델링), 9장(CTE, DTE, CTDE, IDQN과 IPPO, 중앙 비평가, MADDPG, COMA, IGM, VDN, QMIX, QTRAN과 QPLEX, 파라미터 공유와 경험 공유, MCTS와 AlphaZero, PSRO와 AlphaStar), 10장(평가와 실험 설계), 11장(실험 환경). 오르막 게임의 보상 표는 이 분야에서 널리 쓰는 예를 따랐고, 격자 문제와 실험은 이 노트용으로 새로 만들었어요",
                 "결과 값과 그림: <code>tools/pages/rl_multiagent.py</code>가 <code>examples/rl/multi_agent.py</code>를 단계별로 실제 실행한 결과 (씨앗 0~99, 격자는 0~4)"])
print("ok")
