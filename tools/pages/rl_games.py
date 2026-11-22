"""모델 > 강화학습 3쪽: 게임과 균형. examples/rl/matrix_games.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/rl_games.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "rl"
code = S.parse((EX / "matrix_games.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="matrix_games.py")
ENV = f"Python {sys.version.split()[0]}, NumPy {np.__version__}, SciPy {scipy.__version__} (seed 0)"
f2 = lambda x: f"{float(x):.2f}"  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

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


def matrix_html(R1, R2, acts):
    head = "<tr><th></th>" + "".join(f"<th>{a}</th>" for a in acts) + "</tr>"
    rows = "".join(f"<tr><th>{acts[i]}</th>" + "".join(f"<td>{R1[i, j]}, {R2[i, j]}</td>" for j in range(len(acts))) + "</tr>" for i in range(len(acts)))
    return f"<table class=\"mini\">{head}{rows}</table>"


G = ns["GAMES"]
outs = {}
# ---------------- ①
outs["s1"] = S.table("① 네 게임 (칸마다 행 플레이어, 열 플레이어의 보상)", ["게임", "보상 표 (행 = 나, 열 = 상대)", "보상 관계"],
                     [[n, matrix_html(R1, R2, acts), ns["kind"][n]] for n, (R1, R2, acts) in G.items()],
                     ["죄수의 딜레마는 둘 다 협력하면 각 −1이지만, 상대가 협력할 때 배신하면 0, 둘 다 배신하면 각 −3이에요. 치킨은 둘 다 직진하면 부딪혀 0, 한쪽이 피하면 직진한 쪽이 7이에요.",
                      "조정 게임은 둘의 보상이 늘 같아요(공통 보상). 가위바위보는 한쪽이 얻는 만큼 다른 쪽이 잃어요(제로섬). 나머지는 그 사이(일반합)예요."])

# ---------------- ②
pure = ns["pure"]
S.claim(pure["죄수의 딜레마"] == [(1, 1)] and len(pure["조정 게임"]) == 2 and len(pure["치킨"]) == 2 and pure["가위바위보"] == [], "② 순수 내시 균형: 죄수 1, 조정 2, 치킨 2, 가위바위보 0")
outs["s2"] = S.table("② 서로 최적 응답인 칸 (순수 내시 균형)", ["게임", "찾은 칸", "그때 보상"],
                     [[n, ", ".join(f"({G[n][2][a]}, {G[n][2][b]})" for a, b in pure[n]) or "없음", ", ".join(f"({G[n][0][a, b]}, {G[n][1][a, b]})" for a, b in pure[n]) or "-"] for n in G],
                     ["죄수의 딜레마의 균형은 (배신, 배신)뿐이에요. 둘 다 협력하면 둘 다 더 좋은데(−1 > −3), 상대가 협력하든 배신하든 나는 배신하는 게 늘 1~2점 더 나아서 거기서 못 벗어나요.",
                      "조정 게임과 치킨은 균형이 둘이에요. 어느 쪽으로 갈지 정해 주는 게 없어요(균형 선택 문제). 가위바위보는 한 가지만 내는 균형이 없어요. 무엇을 내든 상대가 이기는 걸 내면 되니까요."])

# ---------------- ③
mx = ns["mixed"]
S.claim(abs(mx["치킨"][0] - 1 / 3) < 1e-9 and np.allclose(ns["rps_policy"], 1 / 3, atol=1e-6) and abs(ns["rps_value"]) < 1e-9, "③ 치킨 혼합 균형 1/3, 가위바위보 미니맥스는 균등에 값 0")
S.claim(ns["tilt_value"] > 0 and ns["tilt_policy"][1] > ns["tilt_policy"][0], "③ 가위가 보를 2점으로 이기면 값이 양수, 바위 확률이 가위보다 크다")
tp = ns["tilt_policy"]
outs["s3"] = S.box("③ 섞어서 고르는 균형", [(f"조정 게임 → 둘 다 A를 {mx['조정 게임'][0]:.2f} 확률로 (보상 각 {10 * 0.5:.0f})", 1),
                                           (f"치킨 → 둘 다 직진을 {mx['치킨'][0]:.3f} 확률로 (보상 각 {ns['chicken_points']['내시 (섞어서)'][0]:.3f})", 2),
                                           (f"가위바위보 미니맥스 → 가위, 바위, 보 = {', '.join(f3(x) for x in ns['rps_policy'])}, 게임 값 {abs(ns['rps_value']):.3f}", 3, "newf"),
                                           (f"가위가 보를 2점으로 이기는 가위바위보 → {', '.join(f3(x) for x in tp)}, 게임 값 {ns['tilt_value']:.4f}", 4, "newf")],
                   ["혼합 균형에서는 상대가 내 두 행동을 똑같이 좋게 느끼게 확률을 맞춰요. 그래야 상대도 섞을 이유가 생겨요. 치킨에서 1/3씩 직진하면 기대 보상이 4.67로, 한쪽이 양보하는 순수 균형의 7, 2보다 공평하지만 합은 작아요.",
                    f"가위가 보를 2점으로 이기게 바꾸면, 균형에서 가위를 오히려 덜 내요({tp[0]:.3f}). 상대가 보를 덜 내니까요. 대신 그 가위를 막는 바위를 더 내요({tp[1]:.3f}). 행 플레이어가 먼저 두는 이점이 생겨 값이 0보다 커졌어요."])

# ---------------- ④
ce, u1, u2 = ns["ce_welfare"]
S.claim(abs(u1 - 5.25) < 1e-6 and abs(u2 - 5.25) < 1e-6 and ce[(0, 0)] < 1e-9, "④ 치킨의 합 최대 상관 균형은 각 5.25, (직진, 직진)은 0")
outs["s4"] = S.table("④ 치킨에서 신호를 받아 함께 정하기 (상관 균형, 둘의 합이 가장 크게)", ["추천", "확률"],
                     [[f"({G['치킨'][2][a]}, {G['치킨'][2][b]})", f3(v)] for (a, b), v in ce.items()],
                     [f"신호등처럼 \"누가 직진할지\"를 공정한 제비뽑기로 알려 주면, 각자 받은 추천만 보고 따르는 게 가장 좋아요. 이때 기대 보상은 둘 다 {u1:.2f}로, 섞어서 고르는 내시 균형(4.67)보다 둘 다 높아요.",
                      "\"피함\"을 추천받으면 상대는 1/3 확률로 직진, 2/3 확률로 피함이에요. 그대로 피하면 2 × 1/3 + 6 × 2/3 = 4.67, 직진으로 바꾸면 0 × 1/3 + 7 × 2/3 = 4.67로 같아서 바꿀 이유가 없어요(그래서 이 확률이 합 최대예요)."])
cp = ns["chicken_points"]
fig, ax = plt.subplots(figsize=(6.4, 5.2))
pts = np.array([[G["치킨"][0][a, b], G["치킨"][1][a, b]] for a in range(2) for b in range(2)])
hull = ConvexHull(pts)
poly = pts[hull.vertices]
ax.fill(poly[:, 0], poly[:, 1], color=C["soft"], alpha=.9, label="섞어서 낼 수 있는 보상 쌍")
ax.plot([7, 6, 2], [2, 6, 7], color=OK, lw=2.4, label="파레토 최적 (둘 중 하나를 깎아야만 더 나아짐)")
marks = {"내시 (직진, 피함)": ("o", FAIL), "내시 (피함, 직진)": ("o", FAIL), "내시 (섞어서)": ("s", FAIL), "상관 균형 (합 최대)": ("*", C["blue"]), "(피함, 피함)": ("D", C["gray"])}
for k, (x, y) in cp.items():
    m, col = marks[k]
    ax.scatter([x], [y], marker=m, s=160 if m == "*" else 70, color=col, zorder=3)
    ax.annotate(f"{k}\n({x:.2f}, {y:.2f})" if isinstance(x, float) else f"{k}\n({x}, {y})", (x, y), (x + 0.25, y - 0.6 if k != "(피함, 피함)" else y + 0.3), fontsize=8.5)
ax.set(xlabel="행 플레이어의 기대 보상", ylabel="열 플레이어의 기대 보상", title="치킨의 보상 지도", xlim=(-0.5, 9.5), ylim=(-0.5, 8.5))
ax.legend(fontsize=8, loc="lower left")
P_CH = fig_save(fig, "chicken_map")
outs["f_chicken"] = chart("치킨에서 해들이 놓인 곳", P_CH, "",
                          ["보라 선 위의 점은 한쪽을 손해 보게 하지 않고는 더 나아질 수 없는 결과(파레토 최적)예요. (피함, 피함)의 (6, 6)은 파레토 최적이지만 균형은 아니에요. 서로 피할 때 내가 직진으로 바꾸면 7을 받으니까요.",
                           "섞어서 고르는 내시 균형(4.67, 4.67)은 보라 선에서 한참 안쪽이에요. 상관 균형(5.25, 5.25)은 그보다 바깥이지만 여전히 선에 못 미쳐요. 균형은 \"서로 바꿀 이유가 없다\"일 뿐 \"가장 좋다\"가 아니에요."])

# ---------------- ⑤
fp, rmp, rmr, rmr_ch, rmj = ns["fp_rps"], ns["rm_policy"], ns["rm_regret"], ns["rm_regret_ch"], ns["rm_joint_ch"]
S.claim(np.abs(fp[-1] - 1 / 3).max() < 0.01 and np.abs(rmp - 1 / 3).max() < 0.01 and rmr[-1] < 0.01, "⑤ 가상 플레이와 후회 매칭의 평균이 1/3로 모이고, 후회는 0.01 아래")
S.claim(rmj[0, 1] > 0.99, "⑤ 치킨에서 후회 매칭은 (직진, 피함)에 붙었다")
outs["s5"] = S.table("⑤ 가위바위보를 반복하며 배운 행 플레이어의 평균 정책", ["방법", "가위", "바위", "보", "평균 후회"],
                     [["가상 플레이 (5,000판)", *[f3(x) for x in fp[-1]], "-"], ["후회 매칭 (20,000판)", *[f3(x) for x in rmp], f"{rmr[-1]:.4f}"]],
                     ["둘 다 균형(1/3씩)에 모였어요. 다만 매판 실제로 내는 것은 계속 바뀌어요. 가상 플레이는 매번 하나만 내고, 그 하나가 가위 → 바위 → 보로 돌아요. 모이는 것은 <b>지금까지 낸 비율</b>이에요.",
                      f"<b>평균 후회</b>는 \"지금까지 한 가지만 계속 냈다면 얼마나 더 받았을까\"의 판당 값이에요. 100판에 {rmr[99]:.3f}, 1,000판에 {rmr[999]:.3f}, 20,000판에 {rmr[-1]:.4f}로 0에 다가가요(후회 없음).",
                      f"같은 후회 매칭을 치킨에서 돌리면 둘이 (직진, 피함)에 붙었어요(결합 빈도 {rmj[0, 1]:.3f}). 후회 매칭의 결합 빈도는 거친 상관 균형으로 모인다는 보장이 있는데, 순수 내시 균형도 그 안에 들어 있어서 거기 머문 거예요. 어느 균형에 붙을지는 처음 몇 판의 운이 정해요."])

# ---------------- ⑥
ip, wp = ns["iga_path"], ns["wolf_path"]
S.claim(np.abs(ip[-3000:] - 0.5).max() > 0.3 and np.abs(wp[-1] - 0.5).max() < 0.01, "⑥ IGA는 균형 주위를 맴돌고 WoLF는 0.5로 모인다")
outs["s6"] = S.box("⑥ 동전 맞히기를 기울기로 배우기 (20,000걸음)", [(f"그냥 기울기(IGA) → 마지막 (p, q) = ({ip[-1, 0]:.3f}, {ip[-1, 1]:.3f}), 마지막 3,000걸음 동안 0.5에서 최대 {np.abs(ip[-3000:] - 0.5).max():.2f} 떨어짐", 1),
                                                         (f"WoLF → 마지막 (p, q) = ({wp[-1, 0]:.3f}, {wp[-1, 1]:.3f})", 2, "newf")],
                   ["그냥 기울기를 따르면 서로를 쫓아다니며 균형 (0.5, 0.5) 둘레를 계속 돌아요. 걸음을 띄엄띄엄 밟아서 원이 조금씩 바깥으로 퍼지기까지 해요. 상대가 앞면을 많이 내면 나도 앞면을 늘리고, 그러면 상대는 뒷면으로 돌아서요.",
                    "WoLF(이기고 있으면 천천히, 지고 있으면 빨리)는 같은 기울기에 걸음 크기만 바꿔요. 이길 때 천천히 움직이니 원이 조금씩 안쪽으로 말려 균형에 모여요."])
fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.9))
t = np.arange(1, len(fp) + 1)
for k, (nm, col) in enumerate(zip(G["가위바위보"][2], (OK, FAIL, C["blue"]))):
    axs[0].plot(t, fp[:, k], color=col, lw=2, label=nm)
axs[0].axhline(1 / 3, color=C["ink"], lw=.8, ls=":")
axs[0].set(xscale="log", xlabel="판 수 (log)", ylabel="지금까지 낸 비율", title="가상 플레이 (가위바위보)", ylim=(0.2, 0.5))
axs[0].legend(fontsize=8)
axs[1].plot(np.arange(1, len(rmr) + 1), rmr, color=OK, lw=2, label="가위바위보")
axs[1].plot(np.arange(1, len(rmr_ch) + 1), rmr_ch, color=FAIL, lw=2, label="치킨")
axs[1].set(xscale="log", yscale="log", xlabel="판 수 (log)", ylabel="평균 후회 (log)", title="후회 매칭: 후회가 0으로")
yt = [0.0001, 0.001, 0.01, 0.1, 1]
axs[1].set_yticks(yt, [f"{v:g}" for v in yt])
axs[1].legend(fontsize=8)
axs[2].plot(ip[:, 0], ip[:, 1], color=C["gray"], lw=1, label="그냥 기울기 (IGA)")
axs[2].plot(wp[:, 0], wp[:, 1], color=OK, lw=1.4, label="WoLF")
axs[2].scatter([0.2], [0.6], color=C["ink"], zorder=3, s=20)
axs[2].scatter([0.5], [0.5], marker="*", color=FAIL, s=150, zorder=3, label="균형")
axs[2].set(xlabel="행 플레이어가 앞면을 낼 확률 p", ylabel="열 플레이어의 q", title="동전 맞히기 (출발 0.2, 0.6)", xlim=(0, 1), ylim=(0, 1))
axs[2].set_aspect("equal")
fig.legend(*axs[2].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.83, 0.0), ncol=3, fontsize=8, frameon=False)
P_LEARN = fig_save(fig, "game_learning")
outs["f_learn"] = chart("반복하며 균형에 다가가는 세 가지 방법", P_LEARN, "",
                        ["왼쪽: 비율이 출렁이다가 1/3로 모여요. 가운데: 가위바위보(보라)는 출렁이며 대략 1 ÷ √판 수로 줄고, 치킨(분홍)은 한 칸에 붙은 뒤로 1 ÷ 판 수로 곧게 줄어요.",
                         "오른쪽: 회색은 균형 둘레를 큰 원으로 돌며 조금씩 바깥으로 퍼져서 굵은 띠가 됐어요. 보라는 안쪽으로 말려들어 별(균형)에 닿아요."])

# ---------------- ⑦
lg = ns["league"]
avg = {a: np.mean(list(v.values())) for a, v in lg.items()}
order = sorted(avg, key=avg.get, reverse=True)
S.claim(order[0] in ("원한", "맞대응") and order[1] in ("원한", "맞대응") and lg["늘 배신"]["늘 협력"] == 0, "⑦ 원한과 맞대응이 1, 2등, 늘 배신은 늘 협력을 상대로 0점")
STRAT = list(lg)
outs["s7"] = S.table("⑦ 반복 죄수의 딜레마 100번 대결 (행 쪽의 판당 평균 보상, 자기 자신과도 겨룸)", ["전략 \\ 상대", *STRAT, "평균"],
                     [[a, *[f2(lg[a][b]) for b in STRAT], f"<b>{f2(avg[a])}</b>"] for a in order],
                     [f"늘 배신은 한 판씩 보면 언제나 이득이지만, 평균은 {f2(avg['늘 배신'])}로 맞대응({f2(avg['맞대응'])})과 원한({f2(avg['원한'])})보다 낮아요. 맞대응과 원한은 서로, 그리고 늘 협력과 만나면 내내 협력해서 −1을 받고, 늘 배신에게는 첫 판만 당하고 그 뒤로 막아요.",
                      "한 번만 하는 게임에서는 배신이 유일한 균형이었는데(②), 같은 상대와 계속 만나면 \"배신하면 다음부터 보복\"이라는 약속이 협력을 지켜 줘요."])

flow = [[1, "게임 표", "넷 + 동전", "협력, 경쟁, 섞임"], [2, "순수 내시", "서로 최적 응답", "없거나 여럿"], [3, "섞어서", "혼합 균형, 미니맥스", "선형 계획"],
        [4, "상관 균형", "신호로 함께", "파레토와 후생"], [5, "반복 학습", "가상 플레이, 후회", "평균이 모임"], [6, "기울기", "IGA 대 WoLF", "맴돌기 대 모이기"], [7, "반복 게임", "죄수의 딜레마 대결", "맞대응"]]
S.save(ROOT / "topics" / "model" / "rl-games-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p><a href=\"rl-basics.html\">1쪽</a>과 <a href=\"rl-deep.html\">2쪽</a>에서 환경은 \"규칙대로 반응하는 세상\"이었어요. 그런데 상대도 배우는 에이전트라면 어떨까요? 내가 최선을 다해도 결과는 상대가 무엇을 하느냐에 달려 있고, 상대도 나를 보고 바뀌어요. 그래서 \"가장 좋은 정책\"이라는 말부터 다시 정해야 해요.</p>
<p>그 기준을 오래 연구해 온 분야가 <b>게임 이론</b>이에요. 이 쪽에서는 한 번씩 동시에 고르는 가장 작은 게임(행렬 게임)으로 해의 개념들을 계산해 보고, 반복하면서 그 해에 다가가는 학습 방법을 봐요. 상태가 있는 게임에서 여럿이 배우는 건 <a href=\"rl-multiagent.html\">4쪽</a>에서 다뤄요.</p>"""
whole = """<p>게임 표 → 서로 최적 응답(내시) → 섞어서 고르기 → 신호로 함께 정하기 → 반복하며 배우기(세 가지) → 같은 상대와 계속 만나기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/rl/matrix_games.py</code> 하나예요. 몇 초면 끝나요.</p>"""
steps = [
    K.step(1, "게임을 표로: 행렬 게임", "두 플레이어가 동시에 행동 하나를 고르면, 두 행동의 조합에 따라 각자 보상을 받아요. 그 보상을 표 두 장에 적어요.",
           "<p>가장 작은 게임은 <b>정규형 게임</b>이에요. 플레이어들이 동시에 한 번 고르고 끝나요. 둘이면 보상을 행렬로 적어서 <b>행렬 게임</b>이라고도 해요. 같은 게임을 되풀이하면 반복 게임(⑦), 상태가 바뀌어 가며 여러 번 고르면 확률적 게임(마르코프 게임)이에요. 확률적 게임은 1쪽의 MDP를 여럿으로 늘린 것이고, 상태를 다 못 보면 부분 관측 확률적 게임(POSG)이에요.</p>"
           "<p>보상 관계로 게임을 나눠요. 모두 같은 보상을 받으면 <b>공통 보상</b>(협력), 한쪽 이득이 다른 쪽 손해면 <b>제로섬</b>(경쟁), 그 밖은 <b>일반합</b>이에요.</p>"
           + K.code_table([("GAMES", "게임 이름 → (행 플레이어 보상 표, 열 플레이어 보상 표, 행동 이름)"), ("np.array_equal(R1, -R2)", "합이 늘 0이면 제로섬")]), "s1"),
    K.step(2, "최적 응답과 순수 내시 균형", "상대의 행동을 고정했을 때 내가 가장 많이 받는 행동이 최적 응답이에요. 둘 다 서로에게 최적 응답인 칸을 찾아요.",
           "<p>상대의 정책이 정해져 있으면 내 문제는 1쪽처럼 \"가장 많이 받기\"예요. 그 답이 <b>최적 응답</b>이에요. 모두가 서로에게 최적 응답이라 아무도 혼자 바꿀 이유가 없는 상태가 <b>내시 균형</b>이에요. 표에서는 \"그 열에서 내 보상이 최대이고, 그 행에서 상대 보상이 최대\"인 칸이에요.</p>"
           + K.code_table([("R1[a, b] == R1[:, b].max()", "상대가 b일 때 a가 내 최적 응답인가"), ("R2[a, b] == R2[a, :].max()", "내가 a일 때 b가 상대의 최적 응답인가")]), "s2"),
    K.step(3, "섞어서 고르기: 혼합 균형과 미니맥스", "행동을 확률로 섞는 정책까지 넣으면 모든 유한 게임에 균형이 있어요. 2 × 2는 식으로, 제로섬은 선형 계획으로 풀어요.",
           "<p>행동을 확률로 섞는 정책을 <b>혼합 정책</b>이라고 해요. 내시는 혼합 정책까지 넣으면 <b>모든 유한 게임에 균형이 적어도 하나</b> 있다는 걸 보였어요. 가위바위보의 유일한 균형은 셋을 1/3씩 섞는 거예요.</p>"
           "<p>2인 <b>제로섬</b> 게임에서는 균형이 특별한 꼴이에요. \"상대가 최악으로 나와도 내가 보장할 수 있는 값을 최대로\"(maxmin)와 \"상대에게 허락하는 값을 최소로\"(minmax)가 같아지고, 그 값이 <b>게임 값</b>이에요(미니맥스 정리). 균형이 여럿이어도 값은 하나예요. 이건 선형 계획으로 풀 수 있어요. 변수는 내 행동 확률과 보장 값 v, 조건은 \"상대의 어떤 행동에 대해서도 내 기대 보상 ≥ v\"예요.</p>"
           + K.code_table([("mixed_2x2(R1, R2)", "상대가 두 행동을 똑같이 좋게 느끼게 하는 내 확률"), ("minimax(R)", "선형 계획: v 최대, 단 모든 열에 대해 Σ xᵢ R[i, j] ≥ v"),
                           ("linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)", "scipy의 선형 계획. c를 최소로 하니 −v를 넣어요")]), "s3"),
    K.step(4, "신호를 받아 함께 정하기: 상관 균형, 파레토, 사회 후생", "누가 무엇을 할지 공정한 제비뽑기로 추천해 주면, 혼자 섞을 때보다 둘 다 나아질 수 있어요. 선형 계획으로 둘의 합이 가장 큰 것을 찾아요.",
           "<p>내시 균형은 각자 <b>따로</b> 섞어요. 신호등처럼 바깥에서 결합 행동을 하나 뽑아 각자에게 자기 몫만 알려 주고, 받은 추천을 따르는 게 늘 최적이면 <b>상관 균형</b>이에요. 내시 균형은 모두 상관 균형이고(따로 뽑는 특별한 경우), 상관 균형은 선형 계획으로 바로 계산돼요. 조건을 더 느슨하게 해서 \"추천을 보기 전에 따를지 정한다\"로 바꾸면 <b>거친 상관 균형</b>이에요(⑤의 후회 매칭이 모이는 곳).</p>"
           + K.out("s4") + K.out("f_chicken")
           + "<p>균형이 여럿일 때 고르는 기준도 있어요. <b>파레토 최적</b>: 누구 하나를 손해 보게 하지 않고는 더 나아질 수 없는 결과. <b>사회 후생</b>: 모두의 보상 합. <b>공정성</b>: 보상의 곱처럼 고르게 받을수록 커지는 값. 여기서는 사회 후생을 최대로 하는 상관 균형을 골랐어요.</p>"
           + K.code_table([("correlated(R1, R2, objective)", "결합 행동마다 확률 x를 두고, 추천을 바꾸면 손해라는 조건 아래 objective를 최대로"),
                           ("itertools.permutations(range(k), 2)", "(추천받은 행동, 바꿀 행동)의 모든 쌍"), ("ConvexHull", "섞어서 낼 수 있는 보상 쌍의 영역 (그림용)")]), None),
    K.step(5, "반복하며 배우기: 가상 플레이와 후회 매칭", "같은 게임을 수천 번 반복하며, 상대의 지금까지 비율에 맞서거나(가상 플레이) 지난 아쉬움을 줄이는 쪽으로(후회 매칭) 행동을 바꿔요.",
           "<p>균형을 계산하려면 상대의 보상 표까지 알아야 해요. 실제로는 모르니까, 반복하면서 배워요.</p>"
           "<p><b>가상 플레이</b>: 상대가 지금까지 낸 비율이 상대의 정책이라고 믿고, 그에 대한 최적 응답을 내요. 상대를 모델로 만들어 맞서는 방법(에이전트 모델링)의 가장 단순한 꼴이에요. 제로섬과 2 × 2 게임 등에서는 비율이 균형으로 모인다는 보장이 있어요.</p>"
           "<p><b>후회 매칭</b>: 매판 \"그때 다른 행동을 냈으면 얼마나 더 받았나\"(후회)를 행동마다 더해 두고, 후회가 큰 행동을 그 비율만큼 자주 내요. 그러면 평균 후회가 0으로 가고(후회 없음), 모두가 이렇게 하면 결합 행동의 빈도가 거친 상관 균형으로 모여요. 포커 AI가 쓰는 방법들이 이 생각에서 나왔어요.</p>"
           + K.out("s5")
           + K.code_table([("np.argmax(R1 @ (seen[1] / seen[1].sum()))", "상대의 지금까지 비율에 대한 최적 응답"), ("regret[0] += R1[:, b] - R1[a, b]", "모든 행동에 대해 \"그걸 냈으면\" − \"실제로 받은 것\""),
                           ("p = np.maximum(r, 0) / 합", "양수인 후회만, 그 크기에 비례해 고르기")]), None),
    K.step(6, "기울기로 배우기: IGA와 WoLF", "두 플레이어가 각자 자기 기대 보상의 기울기를 따라 확률을 조금씩 바꿔요. 그냥 따르면 맴돌고, 걸음 크기를 바꾸면 모여요.",
           "<p>2쪽의 정책 경사처럼, 각자 자기 정책을 기대 보상이 커지는 방향으로 조금씩 움직여요(무한소 기울기 상승, IGA). 문제는 상대도 동시에 움직인다는 거예요. 내가 올라가는 방향이 상대 때문에 계속 바뀌어요.</p>"
           "<p><b>WoLF</b>(Win or Learn Fast)는 \"지금 균형 정책보다 많이 받고 있으면(이기는 중) 천천히, 덜 받고 있으면(지는 중) 빨리\" 움직여요. 이기는 동안 천천히 움직이면 상대가 따라잡을 시간이 생겨서, 쫓고 쫓기는 원이 안쪽으로 말려요. 실제로는 균형 정책을 모르니 지금까지의 평균 정책과 비교하는 WoLF-PHC를 써요.</p>"
           + K.out("s6") + K.out("f_learn")
           + K.code_table([("dp = q * 2 - (1 - q) * 2", "행 플레이어 기대 보상 (2p − 1)(2q − 1)의 p에 대한 기울기"), ("lr * (0.25 if mine > 0 else 1.0)", "이기는 중이면 4분의 1 걸음")]), "s6"),
    K.step(7, "같은 게임을 되풀이하면: 반복 죄수의 딜레마", "다섯 가지 전략끼리 100번씩 죄수의 딜레마를 두고, 판당 평균 보상을 비교해요.",
           "<p>한 번만 하면 배신이 유일한 균형이었어요. 같은 상대와 계속 만나면 지난 판을 보고 행동을 정할 수 있어요. <b>맞대응</b>(첫 판은 협력, 그다음엔 상대가 지난번에 한 대로)은 협력에는 협력으로, 배신에는 배신으로 답해요.</p>"
           + K.out("s7")
           + "<p>미래를 충분히 중요하게 보면(γ가 1에 가까우면) \"각자 보복당하지 않을 만큼은 받는\" 어떤 보상 조합도 균형이 될 수 있다는 결과(포크 정리)가 있어요. 협력이 균형이 될 수 있다는 좋은 소식이지만, 균형이 너무 많아져서 어느 것을 고를지는 더 어려워져요.</p>"
           "<p>균형이라는 해에는 한계도 있어요. 가장 좋은 결과가 아닐 수 있고(②의 죄수의 딜레마), 여러 개라 어느 것인지 정해 주지 않고(균형 선택), 균형에서 벗어난 상황에서 어떻게 할지는 말해 주지 않아요. 그래서 다중 에이전트 학습에서는 균형 말고도 파레토 최적, 후생, 후회 없음 같은 목표를 함께 써요.</p>"
           + K.code_table([("STRATEGIES", "전략 이름 → (내 지난 행동들, 상대 지난 행동들)을 받아 이번 행동을 내는 함수"), ("theirs[-1] if theirs else 0", "맞대응: 상대가 지난번에 한 대로"),
                           ("1 if 1 in theirs else 0", "원한: 한 번이라도 배신당하면 끝까지 배신")]), "s7"),
]
practice = [("균형을 직접 계산할 수 있나", "행동이 수십 개인 2인 게임의 내시 균형은 계산이 어렵다고 알려져 있어요(효율적인 방법이 있다는 보장이 없음). 제로섬과 상관 균형은 선형 계획이라 커도 풀려요."),
            ("어떤 해를 목표로", "협력 게임은 모두의 보상 합(공통 보상이면 그 값)을 최대로, 경쟁 게임은 미니맥스, 섞인 게임은 문제에 맞게 균형 + 파레토, 후생, 공정성을 함께 봐요."),
            ("상대가 사람이나 다른 회사의 AI", "균형은 상대도 합리적이라고 가정해요. 실제 상대가 그렇지 않으면, 상대를 관찰해 모델로 만들고 그에 대한 최적 응답을 찾는 쪽(⑤의 가상 플레이 같은)이 나을 수 있어요.")]
libs = [("①②⑤⑥⑦", "<code>numpy</code>, <code>itertools</code>", "보상 표, 반복 학습"), ("③④", "<code>scipy.optimize.linprog</code>", "미니맥스, 상관 균형 선형 계획")]
K.build(ROOT / "topics" / "model" / "rl-games.html", title="게임과 균형", h1="게임과 균형: 상대도 배울 때의 \"가장 좋은 정책\"", track="모델",
        crumb="모델 &gt; 강화학습 3/4쪽 &nbsp;|&nbsp; 이전: <a href=\"rl-deep.html\" style=\"color:inherit\">심층 강화학습</a> &nbsp;|&nbsp; 다음: <a href=\"rl-multiagent.html\" style=\"color:inherit\">다중 에이전트 강화학습</a>",
        one_line="상대가 있으면 \"가장 좋은 정책\" 대신 <b>균형</b>을 찾아요. 아무도 혼자 바꿀 이유가 없는 <b>내시 균형</b>, 제로섬의 <b>미니맥스</b>, 신호로 함께 정하는 <b>상관 균형</b>이 있고, 반복하며 배우는 방법(가상 플레이, 후회 매칭, WoLF)이 그 균형에 다가가요.",
        why=why, whole=whole, steps=steps, code_file="examples/rl/matrix_games.py", practice=practice, libs=libs,
        libs_note="<code>pure_nash</code>, <code>mixed_2x2</code>, <code>minimax</code>, <code>correlated</code>, <code>fictitious_play</code>, <code>regret_matching</code>, <code>gradient_play</code>, <code>match</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rl-multiagent.html\">다중 에이전트 강화학습</a>", "상태가 있는 게임에서 여럿이 배우기"), ("<a href=\"ml-svm-lda-pca.html\">SVM, 판별분석, 차원 축소</a> ⑬", "AdaBoost와 미니맥스 정리"),
                   ("<a href=\"ml-learning-theory.html\">학습 이론</a>", "선형 계획, 보장")],
        sources=["다루는 범위: 강화학습을 여러 에이전트로 넓히기 위한 게임 이론 기초. 수업에서 따로 다루지 않았어요",
                 "함께 정리한 자료: Albrecht, Christianos, Schäfer, <i>Multi-Agent Reinforcement Learning</i> 3장(정규형 게임, 반복 게임, 맞대응, 확률적 게임, POSG, 공통 보상과 제로섬과 일반합), 4장(최적 응답, 미니맥스와 선형 계획, 내시 균형, ε-내시, 상관 균형과 거친 상관 균형, 균형의 한계, 파레토 최적, 사회 후생과 공정성, 후회 없음, 포크 정리), 6장(가상 플레이, IGA, WoLF, 후회 매칭). 게임의 보상 숫자 일부(치킨, 죄수의 딜레마)는 책의 예를 따랐고, 계산과 그림은 이 노트용으로 새로 했어요",
                 "결과 값과 그림: <code>tools/pages/rl_games.py</code>가 <code>examples/rl/matrix_games.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
