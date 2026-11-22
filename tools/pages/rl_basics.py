"""모델 > 강화학습 1쪽: 강화학습 기초. examples/rl/grid_world.py 하나가 ①~⑥으로 자란다.
실행: python tools/pages/rl_basics.py
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
code = S.parse((EX / "grid_world.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="grid_world.py")  # noqa: E731
ns = run(6)
ENV = f"Python {sys.version.split()[0]}, NumPy {np.__version__} (seed 0)"
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


ID, START, GOAL, STAIRS, END, H, W = ns["ID"], ns["START"], ns["GOAL"], ns["STAIRS"], ns["END"], ns["H"], ns["W"]
sid = ID[START]
outs = {}


def draw_grid(ax, values=None, policy=None, title="", path=None, path_color=OK, vmin=None, vmax=None):
    if values is not None:
        grid = np.array([[values[ID[(r, c)]] for c in range(W)] for r in range(H)])
        mask = np.array([[(r, c) in END for c in range(W)] for r in range(H)])
        ax.imshow(np.ma.array(grid, mask=mask), cmap="Purples", vmin=vmin, vmax=vmax)
    for (r, c) in STAIRS:
        ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#ffd6e7"))
        ax.text(c, r, "계단", ha="center", va="center", fontsize=9, color=FAIL)
    ax.add_patch(Rectangle((GOAL[1] - .5, GOAL[0] - .5), 1, 1, color="#d3f9d8"))
    ax.text(GOAL[1], GOAL[0], "목표", ha="center", va="center", fontsize=9, color=C["green"])
    ax.text(START[1], START[0] + .32, "출발", ha="center", va="center", fontsize=8, color=C["ink"])
    if policy is not None:
        for (r, c) in ns["CELLS"]:
            if (r, c) in END:
                continue
            a = int(policy[ID[(r, c)]].argmax())
            dr, dc = ns["MOVES"][a]
            ax.arrow(c - dc * .22, r - dr * .22, dc * .36, dr * .36, head_width=.14, color=C["ink"], length_includes_head=True)
    if path is not None:
        ax.plot([p[1] for p in path], [p[0] for p in path], color=path_color, lw=3, alpha=.8)
    ax.set_xticks(range(W))
    ax.set_yticks(range(H))
    ax.set_xlim(-.5, W - .5)
    ax.set_ylim(H - .5, -.5)
    ax.set_title(title, fontsize=11)
    ax.grid(False)


def cells(pi, limit=20):
    s, out = START, [START]
    while s not in END and len(out) < limit:
        s = ns["move"](s, int(pi[ID[s]].argmax()))
        out.append(s)
    return out


# ---------------- ① 환경
games = ns["random_games"]
lens = [len(g) for g in games]
fall = np.mean([g[-1] == ns["FALL_R"] for g in games])
reach = np.mean([g[-1] == ns["GOAL_R"] for g in games])
S.claim(fall > 0.95 and reach < 0.03, "① 무작위로 걸으면 95% 넘게 계단에 떨어진다")
outs["s1"] = S.box("① 네 방향을 똑같이 고르는 로봇으로 1,000판", [(f"칸 {ns['nS']}개 (그중 끝나는 칸 {len(END)}개), 행동 {ns['nA']}개", 1),
                                                         (f"한 판 평균 걸음 → {np.mean(lens):.1f}", 2), (f"계단에 떨어진 판 → {fall:.1%}, 목표에 닿은 판 → {reach:.1%}", 3, "newf")],
                   ["출발 칸 바로 오른쪽이 계단이라, 아무렇게나 걸으면 거의 다 떨어져요. 무엇을 해야 하는지 정답은 아무도 안 알려 주고, 판이 끝나야 보상으로만 알 수 있어요."])

# ---------------- ② 가치 함수
Vr = ns["V_random"]
S.claim(np.abs(Vr - ns["V_random_sweep"]).max() < 1e-6, "② 연립방정식과 되풀이 덮어쓰기가 같은 값")
S.claim(abs(ns["mc_random"] - Vr[sid]) < 0.5, "② 1,000판 평균이 계산한 값에 0.5 안으로 가깝다")
outs["s2"] = S.box("② 무작위 정책의 가치 (γ = 0.95, 미끄러짐 없음)", [(f"연립방정식으로 푼 V(출발) → {Vr[sid]:.3f}", 1),
                                                                (f"같은 식을 되풀이해 덮어쓰기 → {ns['sweeps_random']}번 만에 같은 값 (가장 큰 차이 {np.abs(Vr - ns['V_random_sweep']).max():.0e})", 2),
                                                                (f"①의 1,000판에서 직접 잰 할인 수익 평균 → {ns['mc_random']:.3f}", 3, "newf")],
                   ["계산한 값과 직접 해 본 평균이 거의 같아요. 가치 함수는 \"이 정책으로 여기서 시작하면 평균 얼마를 받나\"예요.",
                    "출발 칸의 가치가 −20 가까이 되는 건, 거의 늘 한두 걸음 만에 계단(−20)에 떨어지기 때문이에요."])

# ---------------- ③ 동적 계획법
gaps = np.array(ns["gaps"])
ratios = gaps[4:12] / gaps[3:11]
S.claim(np.all(ratios <= ns["GAMMA"] + 1e-9), "③ 매 번 최대 변화가 γ배 이하로 줄어든다")
S.claim(np.array_equal(ns["pi_star"], ns["pi_pi"]) and ns["rounds"] <= 5, "③ 정책 반복이 몇 번 만에 같은 정책")
route95 = ns["route"](ns["pi_star"])
S.claim(route95.startswith("↑↑"), "③ 미끄러짐 10%면 계단에서 두 줄 떨어진 길")
bg = ns["by_gamma"]
S.claim(set(ns["route"](bg[0.5][1])) == {"←"}, "③ γ = 0.5면 움직이지 않는 게 최적")
outs["s3"] = S.box("③ 미끄러짐 10%인 환경에서 최적 정책 찾기 (γ = 0.95)", [(f"가치 반복 → {len(gaps)}번 만에 변화가 1e−8 아래", 1),
                                                                   (f"한 번 덮어쓸 때마다 최대 변화가 줄어든 배수 → {', '.join(f2(x) for x in ratios[:5])} (γ = 0.95 이하)", 2, "newf"),
                                                                   (f"정책 반복 → 평가와 개선 {ns['rounds']}번 만에 같은 정책, 두 방법의 가치 차이 {np.abs(ns['V_star'] - ns['V_pi']).max():.0e}", 3),
                                                                   (f"출발점에서 따라간 길 → {route95}, V*(출발) = {ns['V_star'][sid]:.3f}", 4, "newf")],
                   ["벨만 최적 방정식으로 한 번 덮어쓸 때마다 \"지금 값과 진짜 값의 가장 큰 차이\"가 γ배 이하로 줄어요. 그래서 반드시 하나의 답으로 모여요.",
                    "미끄러지면 옆 칸으로 가니까, 계단 바로 위 줄을 걸으면 10%의 반, 곧 5%씩 떨어질 위험이 있어요. 그래서 두 줄 위로 돌아가요. 한 걸음 더 걷는 −1이 떨어지는 −20의 5%보다 싸요."])
outs["t3"] = S.table("할인율 γ를 바꾸면 (미끄러짐 10%)", ["γ", "출발점에서 따라간 길", "V*(출발)"],
                     [[f"<b>{g}</b> (지금)" if g == 0.95 else g, ns["route"](pi) if len(set(ns["route"](pi))) > 1 else "← (벽에 부딪혀 제자리)", f3(V[sid])] for g, (V, pi) in bg.items()],
                     ["γ = 0.5면 10걸음 뒤의 +10은 0.5¹⁰ ≈ 0.001배로 줄어들어 거의 0이에요. 그래서 목표까지 가는 동안의 −1들이 더 커 보여서, 아예 움직이지 않고 벽에 부딪히는 게(매 걸음 −1이지만 끝나지도 않는) 최적이 돼요.",
                      "γ는 \"나중 보상을 얼마나 믿을지\"를 정하는 학습 목표의 일부예요. 학습 방법의 손잡이가 아니라 문제 정의라서, 바꾸면 최적 정책 자체가 바뀌어요."])
fig, axs = plt.subplots(1, 2, figsize=(12.6, 3.8))
Vs = ns["V_star"]
inner = [Vs[ID[s]] for s in ns["CELLS"] if s not in END]
draw_grid(axs[0], values=Vs, policy=ns["pi_star"], title="미끄러짐 10%: 최적 가치(진할수록 큼)와 정책", vmin=min(inner), vmax=max(inner))
P0, R0 = ns["P0"], ns["R0"]
V0, pi0, _ = ns["value_iteration"](P0, R0)
inner0 = [V0[ID[s]] for s in ns["CELLS"] if s not in END]
draw_grid(axs[1], values=V0, policy=pi0, title="미끄러짐 없음: 최적 가치와 정책", vmin=min(inner0), vmax=max(inner0))
P_VSTAR = fig_save(fig, "grid_optimal")
outs["f_vstar"] = chart("가치 반복으로 찾은 최적 정책 (γ = 0.95)", P_VSTAR, "",
                        ["화살표가 각 칸에서 고를 방향, 색이 그 칸의 최적 가치예요. 목표에 가까울수록 진해요.",
                         f"미끄러짐이 있으면 계단 위 두 줄로 돌아가고(왼쪽), 없으면 계단 바로 위 줄로 가요(오른쪽, V*(출발) = {V0[sid]:.3f})."])
fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.semilogy(range(1, len(gaps) + 1), gaps, color=OK, lw=2.4, marker="o", ms=3, label="한 번 덮어쓸 때의 가장 큰 변화")
ax.semilogy(range(1, len(gaps) + 1), gaps[0] * ns["GAMMA"] ** np.arange(len(gaps)), color=FAIL, lw=1.6, ls="--", label="처음 변화 × γᵏ")
ax.set(xlabel="덮어쓴 횟수 k", ylabel="가장 큰 변화 (log)", title="가치 반복은 γ배씩 줄어들며 하나의 답으로 모여요")
yt = [10.0 ** e for e in range(-9, 3, 2)]
ax.set_yticks(yt, [f"{t:g}" for t in yt])
ax.minorticks_off()
ax.legend(fontsize=9)
P_GAPS = fig_save(fig, "value_iteration_gaps")
outs["f_gaps"] = chart("가치 반복의 수렴", P_GAPS, "", ["처음 몇 번은 목표의 소식이 칸에서 칸으로 퍼지는 중이라 변화가 커요. 소식이 다 퍼진 뒤에는 γ 기울기보다 빠르게 줄어요."])

# ---------------- ④ 밴딧
bd = ns["bandit"]
last = {e: (r[-100:].mean(), o[-100:].mean()) for e, (r, o) in bd.items()}
S.claim(last[0.1][0] > last[0.0][0] and last[0.1][0] > last[0.3][0] and last[0.0][1] < 0.45, "④ ε = 0.1이 가장 좋고, 탐색 없이는 최적 머신을 45%도 못 고른다")
outs["s4"] = S.table("④ 슬롯머신 10대, 1,000번 당기기 (2,000판 평균)", ["ε", "마지막 100번 평균 보상", "마지막 100번 최적 머신을 고른 비율", "1,000번 전체 평균 보상"],
                     [[f"<b>{e}</b>", f3(r), f"{o:.1%}", f3(bd[e][0].mean())] for e, (r, o) in last.items()],
                     [f"ε = 0(늘 지금 가장 좋아 보이는 것만)은 처음에 운 좋게 괜찮았던 머신에 갇혀서 최적 머신을 {last[0.0][1]:.0%}만 골라요.",
                      f"ε = 0.1이 가장 좋았어요. ε = 0.3은 최적 머신을 알아낸 뒤에도 30%를 아무 데나 써서 평균 보상이 다시 낮아요. ε = 0.01은 천천히 찾지만, 더 오래 하면 결국 0.1을 앞질러요(덜 낭비하니까)."])
fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.6))
cols = {0.0: C["gray"], 0.01: C["blue"], 0.1: OK, 0.3: FAIL}
kern = np.ones(20) / 20
for e, (r, o) in bd.items():
    axs[0].plot(np.convolve(r, kern, mode="valid"), color=cols[e], lw=2, label=f"ε = {e}")
    axs[1].plot(np.convolve(o, kern, mode="valid"), color=cols[e], lw=2, label=f"ε = {e}")
axs[0].set(xlabel="당긴 횟수", ylabel="평균 보상 (20번 이동 평균)", title="받은 보상")
axs[1].set(xlabel="당긴 횟수", ylabel="최적 머신을 고른 비율", title="최적 머신을 고른 비율", ylim=(0, 1))
axs[1].legend(fontsize=9, loc="lower right")
P_BANDIT = fig_save(fig, "bandit")
outs["f_bandit"] = chart("탐색 비율 ε에 따른 차이 (2,000판 평균)", P_BANDIT, "", ["회색(ε = 0)은 빨리 올라가다 멈춰요. 더 좋은 머신이 있는지 확인하지 않기 때문이에요."])

# ---------------- ⑤ MC vs TD
pe = ns["pred_err"]
S.claim(pe[-1, 0] < pe[-1, 1] and pe[9, 0] < pe[9, 1], "⑤ 이 문제의 출발 칸에서는 몬테카를로가 TD(0)보다 오차가 작다")
alpha_rows = []
for al in (0.02, 0.05, 0.1, 0.2, 0.4):
    if al == 0.05:
        e = pe
    else:
        n = run(5, [("def mc_vs_td(episodes=300, alpha=0.05, seed=0):", f"def mc_vs_td(episodes=300, alpha={al}, seed=0):")])
        e = n["pred_err"]
    alpha_rows.append((al, e))
S.claim(all(e[-1, 1] > e[-1, 0] for _, e in alpha_rows), "⑤ 어떤 α에서도 300판 뒤 TD 오차가 MC보다 크다")
outs["s5"] = S.table(f"⑤ 정책 하나의 V(출발) 재기 (진짜 값 {ns['V_true'][sid]:.3f}, 20번 반복 평균 오차)", ["판 수", "몬테카를로", "TD(0), α = 0.05"],
                     [[n_, f3(pe[n_ - 1, 0]), f3(pe[n_ - 1, 1])] for n_ in (10, 50, 150, 300)],
                     ["책에서는 TD가 흔들림이 적어 보통 더 빨리 맞는다고 해요. 그런데 이 문제의 출발 칸에서는 몬테카를로가 계속 앞섰어요.",
                      "출발 칸은 매 판 지나가니까 몬테카를로는 판마다 진짜 수익 표본을 하나씩 얻어 정확히 평균을 내요. TD(0)은 목표의 +10 소식이 칸에서 칸으로 한 걸음씩 거꾸로 전해져야 해서, 처음 수십 판은 출발 칸의 값이 0 근처에 머물러요."])
outs["t5"] = S.table("TD(0)의 α를 바꾸면 (출발 칸 오차, 20번 반복 평균)", ["α", "50판 뒤", "150판 뒤", "300판 뒤", "몬테카를로 300판 뒤"],
                     [[f"<b>{a}</b> (지금)" if a == 0.05 else a, f3(e[49, 1]), f3(e[149, 1]), f3(e[-1, 1]), f3(e[-1, 0])] for a, e in alpha_rows],
                     ["α가 작으면 소식이 느리게 퍼지고, 크면 빨리 퍼지지만 한 판 한 판에 너무 흔들려서 오차가 다시 커져요. 0.1 근처가 가장 나았지만 몬테카를로는 못 이겼어요.",
                      "TD의 장점은 판이 아주 길거나 끝나지 않는 문제, 칸끼리 값을 나눠 쓰는 문제에서 커요. 어느 쪽이 나은지는 문제마다 달라서, 둘을 섞는 TD(λ)도 있어요."])
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(range(1, len(pe) + 1), pe[:, 0], color=OK, lw=2.4, label="몬테카를로 (판이 끝나면 평균)")
ax.plot(range(1, len(pe) + 1), pe[:, 1], color=FAIL, lw=2.4, label="TD(0), α = 0.05 (한 걸음마다)")
ax.set(xlabel="판 수", ylabel="V(출발) 오차", title="이 문제의 출발 칸에서는 몬테카를로가 빨라요", ylim=(0, None))
ax.legend(fontsize=9)
P_PRED = fig_save(fig, "mc_vs_td")
outs["f_pred"] = chart("몬테카를로와 TD(0)로 가치 재기", P_PRED, "", ["TD(0)은 처음 수십 판 동안 0 근처에서 출발해, 목표의 소식이 출발 칸까지 전해질 때까지 오차가 커요."])

# ---------------- ⑥ SARSA vs Q-learning
rs = ns["runs"]
Vopt0 = ns["V_opt0"]
paths = {}
for kind in rs:
    cnt = {}
    for Q, c in rs[kind]:
        p = ns["route"](ns["greedy"](Q))
        cnt[p] = cnt.get(p, 0) + 1
    paths[kind] = cnt
final = {k: np.mean([c[-1, 2] for _, c in v]) for k, v in rs.items()}
online = {k: np.mean([c[-200:, 1].mean() for _, c in v]) for k, v in rs.items()}
near = lambda k: sum(n for p, n in paths[k].items() if p.startswith("↑→→"))  # noqa: E731
S.claim(near("q") == 10 and near("sarsa") <= 3 and abs(final["q"] - Vopt0) < 1e-6, "⑥ Q-learning은 10번 모두 계단 바로 위 최단 길, SARSA는 대부분 돌아간다")
outs["s6"] = S.table("⑥ 미끄러짐 없는 환경, ε = 0.1, α = 0.5, 1,000판 (10번 반복)", ["", "SARSA", "Q-learning"],
                     [["greedy 정책의 출발점 가치 (최적 {:.3f})".format(Vopt0), f3(final["sarsa"]), f3(final["q"])],
                      ["계단 바로 위 최단 길을 배운 횟수", f"{near('sarsa')} / 10", f"{near('q')} / 10"],
                      ["학습 중 마지막 200판의 평균 수익", f2(online["sarsa"]), f2(online["q"])],
                      ["1,000판 동안 걸은 걸음 수 (평균)", f"{np.mean([c[-1, 0] for _, c in rs['sarsa']]):,.0f}", f"{np.mean([c[-1, 0] for _, c in rs['q']]):,.0f}"]],
                     [f"Q-learning은 10번 모두 최적 길(↑→→→→→→→↓)을 배웠어요. SARSA는 대부분 계단에서 한 줄 이상 떨어진 길을 배웠어요({near('sarsa')}번만 최단 길).",
                      "SARSA는 \"다음에 실제로 고를 행동\"의 값으로 배워요. 학습 중에는 10% 확률로 아무 방향이나 가니까, 계단 바로 위에서는 그 무작위 한 걸음에 떨어질 수 있다는 걸 값에 넣어요. 그래서 학습 중의 자기 모습에 맞는 안전한 길을 배워요.",
                      "Q-learning은 \"다음 칸에서 가장 좋은 행동\"의 값으로 배워요. 무작위 한 걸음은 계산에 안 넣고, 나중에 greedy로만 움직일 때의 최적 길을 배워요.",
                      f"학습 중 평균 수익은 이번에는 Q-learning이 더 높았어요({online['q']:.2f} 대 {online['sarsa']:.2f}). SARSA의 돌아가는 길이 더 길고, 아직 덜 배운 판도 있었기 때문이에요. \"SARSA가 학습 중에는 더 안전하다\"는 건 계단에서 잃는 점수가 길을 돌아가는 비용보다 훨씬 클 때 두드러져요."])
fig, axs = plt.subplots(1, 2, figsize=(12.6, 3.8))
for kind, col, nm in (("sarsa", FAIL, "SARSA"), ("q", OK, "Q-learning")):
    grid_x = np.linspace(0, 9000, 91)
    ys = [np.interp(grid_x, c[:, 0], c[:, 2]) for _, c in rs[kind]]
    axs[0].plot(grid_x, np.mean(ys, axis=0), color=col, lw=2.4, label=nm)
    axs[0].fill_between(grid_x, np.percentile(ys, 25, axis=0), np.percentile(ys, 75, axis=0), color=col, alpha=.15)
axs[0].axhline(Vopt0, color=C["ink"], lw=1, ls=":", label="최적")
axs[0].set(xlabel="누적 걸음 수", ylabel="지금 greedy 정책의 V(출발)", title="학습 곡선 (10번 평균, 칠한 곳은 가운데 50%)", ylim=(-22, 2))
axs[0].legend(fontsize=9, loc="lower right")
common_s = max(paths["sarsa"], key=paths["sarsa"].get)
qs = next(Q for Q, _ in rs["q"])
qpath = cells(ns["greedy"](qs))
spath = None
for Q, _ in rs["sarsa"]:
    if ns["route"](ns["greedy"](Q)) == common_s:
        spath = cells(ns["greedy"](Q))
        break
draw_grid(axs[1], title="배운 길: Q-learning(보라)과 SARSA의 가장 흔한 길(분홍)")
axs[1].plot([p[1] for p in qpath], [p[0] for p in qpath], color=OK, lw=3.4, alpha=.85)
axs[1].plot([p[1] + .08 for p in spath], [p[0] - .08 for p in spath], color=FAIL, lw=3.4, alpha=.85)
P_TD = fig_save(fig, "sarsa_vs_q")
q_reach = max(c[np.flatnonzero(np.abs(c[:, 2] - Vopt0) > 1e-6).max() + 1, 0] for _, c in rs["q"])   # 그 뒤로 계속 최적인 걸음 수
S.claim(q_reach < 3000, "⑥ Q-learning은 3,000걸음 안에 최적")
outs["f_td"] = chart("SARSA와 Q-learning이 배운 것", P_TD, "",
                     ["왼쪽 x축은 판 수가 아니라 <b>누적 걸음 수</b>예요. 판 수로 그리면 한 판에 오래 헤맨 쪽이 경험을 더 많이 모은 걸 놓쳐요.",
                      f"Q-learning은 10번 모두 {q_reach:,.0f}걸음 안쪽부터 계속 최적이고, SARSA는 돌아가는 길에 머물러 최적보다 낮게 오르내려요."])
outs["t6"] = S.table("SARSA의 greedy 길 10개", ["길", "횟수"], [[p, n] for p, n in sorted(paths["sarsa"].items(), key=lambda x: -x[1])],
                     ["↓만 계속되는 한 개는 출발 칸에서 아래 벽에 부딪히는 쪽을 골라 1,000판 안에 아직 덜 배운 경우예요. ε을 서서히 0으로 줄이면 SARSA도 결국 최단 길로 가요."])

flow = [[1, "환경", "4 × 8 격자, 계단", "보상 −1, −20, +10"], [2, "가치 함수", "벨만 방정식", "연립방정식"], [3, "동적 계획법", "모델을 알 때", "γ배씩 수렴"],
        [4, "탐색과 활용", "슬롯머신 10대", "ε = 0.1"], [5, "MC와 TD", "해 보면서 가치 재기", "여기선 MC"], [6, "SARSA와 Q", "모델 없이 정책 배우기", "안전 대 최단"]]
S.save(ROOT / "topics" / "model" / "rl-basics-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow, "env": ENV})

why = """<p>1~4쪽의 머신러닝은 \"입력과 정답\"을 주고 배웠어요. 그런데 로봇이 길을 찾거나, 게임을 하거나, 재고를 언제 채울지 정할 때는 <b>정답 행동</b>을 아무도 몰라요. 해 보고 나서 결과가 좋았는지 나빴는지만 알 수 있고, 그것도 한참 뒤에야 알아요.</p>"""
why += K.table(["", "지도학습", "강화학습"], [["배우는 재료", "입력과 정답", "해 본 행동과 받은 보상"], ["피드백", "바로, 정답으로", "늦게, 점수로만 (무엇이 정답인지는 안 알려 줌)"],
                                         ["데이터", "미리 모아 둔 표", "내가 고른 행동에 따라 다음 데이터가 달라짐"], ["어려운 점", "과적합", "탐색과 활용, 늦은 보상의 공을 누구에게 줄지(시간 크레딧 할당)"]])
why += "<p>이 쪽은 칸이 32개뿐인 작은 격자로, 모든 값을 정확히 계산할 수 있는 상태에서 강화학습의 기본 도구를 하나씩 봐요. 칸이 수만 개를 넘어 표로 다 못 적는 경우는 <a href=\"rl-deep.html\">2쪽 심층 강화학습</a>에서 다뤄요.</p>"
whole = """<p>환경 → 가치 함수 → 모델을 알 때(동적 계획법) → 탐색 → 해 보면서 가치 재기 → 해 보면서 정책 배우기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/rl/grid_world.py</code> 하나예요. 환경도 코드가 직접 만들어서 받을 파일이 없어요.</p>"""
steps = [
    K.step(1, "격자 환경 만들기", "칸(상태), 방향(행동), 한 걸음의 결과(다음 칸과 보상)를 정해요. 이것이 마르코프 결정 과정(MDP)이에요.",
           "<p>강화학습에는 둘이 있어요. 결정하는 쪽이 <b>에이전트</b>(여기서는 로봇), 나머지 전부가 <b>환경</b>이에요. 매 시점 에이전트는 지금 <b>상태</b>(어느 칸인지)를 보고 <b>행동</b>(방향)을 고르고, 환경은 다음 상태와 <b>보상</b>(점수)을 돌려줘요. 시작부터 끝나는 칸까지가 한 <b>판</b>(에피소드)이에요.</p>"
           "<p>이걸 수학으로 적은 게 <b>MDP</b>(마르코프 결정 과정)예요. 상태 집합, 행동 집합, 전이 확률 P(s' | s, a), 보상 R, 시작 분포로 이뤄져요. \"마르코프\"는 <b>다음 일이 지금 상태와 행동에만 달려 있고, 그 전에 어떻게 왔는지는 상관없다</b>는 가정이에요. 그래서 지금 칸만 보고 결정해도 충분해요.</p>"
           "<p>지금 상태를 다 볼 수 없고 일부만 보는 경우(카메라에 가린 곳이 있는 로봇)는 부분 관측 MDP(POMDP)라고 해요. 그때는 지금까지 본 것들을 기억해서 상태를 짐작해야 해요.</p>"
           + K.code_table([("MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]", "위, 오른쪽, 아래, 왼쪽"), ("outcomes(s, a, slip)", "칸 s에서 a로 갈 때 생길 수 있는 결과 (확률, 다음 칸, 보상)"),
                           ("make_mdp(slip)", "모든 칸과 행동의 결과를 표 두 개로: P[s, a, s'], R[s, a]"), ("RANDOM", "네 방향을 1/4씩 고르는 정책"), ("play(policy, slip)", "정책대로 한 판 해 보기")]), "s1"),
    K.step(2, "가치 함수와 벨만 방정식", "\"이 정책으로 여기서 시작하면 평균 얼마를 받나\"를 칸마다 계산해요. 계산 식이 자기 자신을 다시 부르는 꼴이라 연립방정식으로 풀려요.",
           "<p>한 판의 점수는 받은 보상을 더한 <b>수익</b>이에요. 다만 나중 보상은 덜 쳐서 <code>G = r₀ + γr₁ + γ²r₂ + ...</code>로 더해요. γ(0~1)는 <b>할인율</b>이에요. 1에 가까우면 먼 미래까지 보고, 0에 가까우면 당장만 봐요. \"매 걸음 1 − γ 확률로 세상이 끝날 수 있다\"고 읽어도 같아요. γ < 1이면 끝나지 않는 문제에서도 수익이 무한대가 되지 않아요.</p>"
           "<p><b>정책</b> π(a | s)는 칸마다 어느 행동을 얼마의 확률로 고를지예요. 정책 π를 따를 때 칸 s에서 시작한 평균 수익이 <b>상태 가치</b> V^π(s), 칸 s에서 행동 a를 먼저 하고 그 뒤 π를 따를 때가 <b>행동 가치</b> Q^π(s, a)예요.</p>"
           "<p>수익은 \"이번 보상 + γ × 다음부터의 수익\"이라서, 가치도 같은 꼴로 쓸 수 있어요. 이게 <b>벨만 방정식</b>이에요.</p>"
           "<p><code>V^π(s) = Σₐ π(a|s) Σₛ' P(s'|s, a) [R + γ V^π(s')]</code></p>"
           "<p>칸이 32개면 미지수 32개짜리 일차 연립방정식이라 한 번에 풀려요. 또는 오른쪽을 계산해 왼쪽에 덮어쓰기를 되풀이해도 같은 답으로 모여요.</p>"
           + K.code_table([("discounted(rewards)", "한 판의 할인 수익"), ("np.einsum(\"sa,sat->st\", policy, P)", "정책을 반영한 칸→칸 이동 확률"),
                           ("np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)", "V = Rπ + γ Pπ V를 V에 대해 풀기"), ("evaluate_by_sweeps", "같은 식을 덮어쓰며 되풀이 (반복 정책 평가)")]), "s2"),
    K.step(3, "모델을 알 때: 가치 반복과 정책 반복", "전이 확률과 보상을 다 알면, 벨만 최적 방정식을 되풀이해서 최적 정책을 바로 계산할 수 있어요(동적 계획법).",
           "<p>가장 좋은 정책의 가치 V*는 \"가장 좋은 행동을 고른다\"를 넣은 <b>벨만 최적 방정식</b>을 만족해요.</p>"
           "<p><code>V*(s) = maxₐ Σₛ' P(s'|s, a) [R + γ V*(s')]</code></p>"
           "<p>max 때문에 일차식이 아니라 한 번에는 못 풀어요. 대신 오른쪽을 계산해 덮어쓰기를 되풀이하면(<b>가치 반복</b>) 매번 진짜 값과의 차이가 γ배 이하로 줄어서 반드시 V*로 모여요(수축 사상). V*를 알면 칸마다 가장 큰 Q*(s, a)의 행동을 고르면 그게 최적 정책이에요. MDP에는 늘 확률 없이 하나로 정해진 최적 정책이 있어요.</p>"
           "<p><b>정책 반복</b>은 \"지금 정책의 가치를 정확히 계산(②) → 그 가치로 가장 좋은 행동만 고르게 고치기\"를 바뀌지 않을 때까지 되풀이해요. 고친 정책은 늘 원래보다 나쁘지 않아서(정책 개선 정리) 몇 번 만에 끝나요. 같은 문제를 선형 계획 문제로 풀 수도 있어요.</p>"
           + K.out("f_vstar") + K.out("f_gaps")
           + "<p>⚠️ 동적 계획법은 P와 R을 <b>다 알아야</b> 해요. 실제 로봇이나 게임은 \"이 행동을 하면 무슨 일이 몇 % 확률로 일어나나\"를 모르는 경우가 대부분이라, ⑤⑥처럼 해 보면서 배워요.</p>"
           + K.code_table([("Q = R + gamma * P @ V", "모든 칸, 모든 행동의 Q를 한 번에"), ("Q.max(axis=1)", "벨만 최적 방정식의 max"), ("greedy(Q)", "Q가 가장 큰 행동만 고르는 정책"),
                           ("np.array_equal(new_pi, pi)", "정책이 더 안 바뀌면 끝")])
           + K.try_("할인율 γ를 0.5, 0.99로 바꾸면?", "t3"), "s3"),
    K.step(4, "탐색과 활용: 슬롯머신 10대", "평균을 모르는 슬롯머신 10대를 1,000번 당기며, 지금 가장 좋아 보이는 것만 당길지 가끔 다른 것도 해 볼지 비교해요.",
           "<p>모델을 모르면 해 봐야 알아요. 그런데 지금까지 가장 좋았던 행동만 하면(<b>활용</b>), 한 번도 제대로 안 해 본 더 좋은 행동을 영영 못 찾아요. 반대로 새로운 것만 해 보면(<b>탐색</b>) 아는 좋은 행동을 안 써서 손해예요. 이게 <b>탐색과 활용의 딜레마</b>예요.</p>"
           "<p>상태가 하나뿐이고 매번 바로 끝나는 가장 단순한 강화학습이 <b>멀티암드 밴딧</b>(팔이 여러 개인 슬롯머신)이에요. 가장 단순한 해법은 <b>ε-greedy</b>예요. ε 확률로 아무거나, 나머지는 지금 평균이 가장 높은 것을 골라요.</p>"
           + K.out("s4") + K.out("f_bandit")
           + "<p>더 똑똑한 탐색도 있어요. 덜 해 본 행동에 \"아직 모르니까\" 덤을 얹어 주는 UCB, 평균의 분포를 두고 뽑아 보는 톰슨 샘플링 같은 방법이에요. 큰 문제에서는 ε을 처음에 크게 두고 서서히 줄이는 게 흔해요.</p>"
           + K.code_table([("rng.random(RUNS) < eps", "ε 확률로 탐색"), ("est += (r - est) / cnt", "지금까지의 평균을 한 건씩 고치기 (표본 평균)")]), "s4"),
    K.step(5, "해 보면서 가치 재기: 몬테카를로와 TD(0)", "정해진 정책으로 300판을 해 보며 V(출발)를 두 방법으로 재고, ③의 진짜 값과 비교해요.",
           "<p><b>몬테카를로</b>(MC): 판이 끝날 때까지 기다렸다가, 실제로 받은 수익의 평균을 가치로 써요. 정직하지만 판이 끝나야 배우고, 판마다 수익이 크게 흔들려요.</p>"
           "<p><b>TD(0)</b>(시간차 학습): 한 걸음만 보고 바로 고쳐요. 다음 칸의 \"지금 짐작한 가치\"를 빌려 써서 <code>V(s) ← V(s) + α [r + γV(s') − V(s)]</code>예요. 괄호 안이 <b>TD 오차</b>예요. 짐작으로 짐작을 고치는 걸 <b>부트스트래핑</b>이라고 해요. 판이 끝나기 전에도 배우고 덜 흔들리지만, 처음 짐작이 틀리면 그 틀림이 퍼지는 데 시간이 걸려요.</p>"
           + K.out("s5") + K.out("f_pred")
           + K.try_("TD(0)의 α를 0.02부터 0.4까지 바꾸면?", "t5")
           + "<p>α를 처음엔 크게, 갈수록 작게(예: 1 ÷ 방문 횟수) 두면 TD도 진짜 값으로 모인다는 보장이 있어요(확률적 근사 조건: α의 합은 무한, α²의 합은 유한). 몬테카를로의 1 ÷ 횟수 평균이 바로 그런 α예요.</p>"
           + K.code_table([("V_td[s] += alpha * (target - V_td[s])", "TD(0): 한 걸음 뒤에 바로 고치기"), ("for s_t, r_t in reversed(steps)", "몬테카를로: 판이 끝난 뒤 뒤에서부터 수익 G를 쌓기"),
                           ("V_mc[s] += (G - V_mc[s]) / n_mc[s]", "방문할 때마다 평균 고치기")]), "s5"),
    K.step(6, "모델 없이 정책 배우기: SARSA와 Q-learning", "미끄러짐 없는 환경에서, 행동 가치 Q를 한 걸음마다 고쳐 가며 최적 정책을 찾아요. 두 방법은 \"다음 값\"을 무엇으로 보느냐만 달라요.",
           "<p>⑤는 정해진 정책의 가치만 쟀어요. 정책을 <b>고치려면</b> 행동마다 값을 알아야 해서 Q(s, a)를 배워요. 행동은 ε-greedy로 고르고, 한 걸음마다 Q를 TD로 고쳐요.</p>"
           + K.table(["", "SARSA", "Q-learning"], [["목표 값", "r + γ Q(s', <b>실제로 고른 a'</b>)", "r + γ <b>max</b>ₐ' Q(s', a')"], ["이름", "(s, a, r, s', a') 다섯 개로 배워서", "Q를 배워서"],
                                                  ["무엇을 배우나", "지금 행동하는 정책(ε 포함)의 가치 (<b>on-policy</b>)", "greedy 최적 정책의 가치 (<b>off-policy</b>)"],
                                                  ["행동하는 정책", "배우는 정책과 같아야 해요", "아무 정책이나 돼요. 남이 한 기록으로도 배울 수 있어요"]])
           + K.out("s6") + K.out("f_td") + K.out("t6")
           + "<p><b>학습 곡선</b>은 강화학습을 평가하는 기본 그림이에요. x축은 판 수가 아니라 환경과 주고받은 <b>누적 걸음 수</b>로 두고, y축은 그때까지 배운 정책을 탐색 없이(greedy) 따로 돌린 <b>평가 수익</b>으로 둬요. 무작위 씨앗을 여러 개로 반복해 평균과 퍼짐을 같이 그려요. 할인 수익 말고 할인 없는 총점, 판 길이, 승률을 같이 보기도 해요.</p>"
           "<p>칸이 수십 개라 Q를 표 하나에 다 적었어요. 칸이 수만, 수억 개면(바둑, 게임 화면) 표를 못 만들고, 비슷한 칸끼리 값을 나눠 쓸 수도 없어요. 그래서 Q를 신경망으로 근사해요. 그게 <a href=\"rl-deep.html\">2쪽 심층 강화학습</a>이에요.</p>"
           + K.code_table([("pick(s)", "ε-greedy로 행동 고르기 (동점이면 무작위)"), ("r + GAMMA * Q[ID[s2], a2]", "SARSA의 목표"), ("r + GAMMA * Q[ID[s2]].max()", "Q-learning의 목표"),
                           ("evaluate(greedy(Q), P, R)[ID[START]]", "지금 Q로 만든 greedy 정책의 진짜 가치 (평가용, 학습에는 안 씀)")]), "s6"),
]
practice = [("γ는 어떻게 정하나", "학습 목표의 일부예요. \"얼마나 먼 미래까지 중요한가\"로 문제에 맞게 정하고, 학습 방법을 비교할 때는 같은 γ로 맞춰요(③에서 바꾸면 최적 정책이 달라졌어요)."),
            ("보상 설계", "목표 +10, 한 걸음 −1처럼 정하면 \"빨리 목표로\"를 배워요. 보상을 잘못 주면 엉뚱한 행동(③의 γ = 0.5처럼 가만히 있기)을 배우니, 학습된 행동을 꼭 직접 돌려 봐요."),
            ("ε은 얼마로", "처음엔 크게(0.5~1) 시작해 0.05 근처까지 줄이는 게 흔해요. 고정 ε이면 ⑥의 SARSA처럼 그 ε에 맞는 조심스러운 정책을 배워요."),
            ("결과가 들쭉날쭉하다", "강화학습은 씨앗에 따라 결과가 크게 달라요. 씨앗 5~10개로 반복하고 평균과 퍼짐을 같이 보고해요(⑥의 칠한 띠).")]
libs = [("①~⑥", "<code>numpy</code>", "표, 연립방정식(<code>np.linalg.solve</code>), 무작위(<code>default_rng</code>)")]
K.build(ROOT / "topics" / "model" / "rl-basics.html", title="강화학습 기초", h1="강화학습 기초: 해 보면서 배우기", track="모델",
        crumb="모델 &gt; 강화학습 1/4쪽 &nbsp;|&nbsp; 다음: <a href=\"rl-deep.html\" style=\"color:inherit\">심층 강화학습</a>",
        one_line="정답 없이 <b>해 보고 받은 보상</b>으로 배워요. 칸마다 \"앞으로 평균 얼마를 받나\"(<b>가치</b>)를 벨만 방정식으로 계산하거나(모델을 알 때), 한 걸음마다 고쳐 가며 배워요(<b>TD</b>: SARSA, Q-learning). 언제나 <b>탐색과 활용</b> 사이에서 줄타기를 해요.",
        why=why, whole=whole, steps=steps, code_file="examples/rl/grid_world.py", practice=practice, libs=libs,
        libs_note="<code>move</code>, <code>outcomes</code>, <code>make_mdp</code>, <code>play</code>, <code>discounted</code>, <code>evaluate</code>, <code>evaluate_by_sweeps</code>, <code>greedy</code>, <code>value_iteration</code>, <code>policy_iteration</code>, <code>route</code>, <code>soft</code>, <code>mc_vs_td</code>, <code>td_control</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"rl-deep.html\">심층 강화학습</a>", "Q와 정책을 신경망으로"), ("<a href=\"rl-games.html\">게임과 균형</a>", "상대가 있을 때"), ("<a href=\"ml-learning-theory.html\">학습 이론</a>", "탐색 없이 데이터가 주어질 때의 보장")],
        sources=["다루는 범위: 부트캠프 머신러닝, 딥러닝 기초를 넓힌 쪽이에요. 수업에서 따로 다루지 않았어요",
                 "함께 정리한 자료: Albrecht, Christianos, Schäfer, <i>Multi-Agent Reinforcement Learning: Foundations and Modern Approaches</i> 2장(MDP, 할인 수익, 벨만 방정식, 수축 사상, 정책 반복과 가치 반복, SARSA와 Q-learning, ε-greedy, 학습 곡선). Prince, <i>Understanding Deep Learning</i> 19장 1~3절(마르코프 과정, POMDP, 몬테카를로, TD). Mohri 외, <i>Foundations of Machine Learning</i> 17장(정책 평가, 선형 계획, 확률적 근사, TD(λ)). 격자 환경과 슬롯머신은 이 노트용으로 새로 만들었어요",
                 "결과 값과 그림: <code>tools/pages/rl_basics.py</code>가 <code>examples/rl/grid_world.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
