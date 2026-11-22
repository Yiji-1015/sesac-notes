# 여럿이 함께 배우는 강화학습(MARL)을 작은 협력 문제로 직접 해 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 문제 ①②: 둘이 동시에 고르는 협력 행렬 게임. 문제 ③: 6 × 6 격자에서 무거운 상자를 둘이 함께 들기.
# 데이터 파일 없이 코드가 직접 만든 환경이에요. CPU로 1분쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from collections import Counter                                                 #@1
import numpy as np                                                              #@1

# ① 각자 배우기 vs 함께 배우기: 오르막 게임
CLIMB = np.array([[11, -30, 0],                                 # 둘이 받는 보상이 같은(공통 보상) 협력 게임
                  [-30, 7, 6],                                  # (0, 0)이 11로 가장 좋지만, 한쪽만 0을 내면 −30
                  [0, 0, 5]])

def eps_at(t, horizon=3000):                                    # 처음엔 거의 무작위, 3,000판에 걸쳐 0.05까지
    return max(0.05, 1 - t / horizon)

def matrix_learn(kind, seed, rounds=5000, alpha=0.1, log=False):
    rng = np.random.default_rng(seed)
    Q1, Q2, Qj, trace = np.zeros(3), np.zeros(3), np.zeros((3, 3)), []
    for t in range(rounds):
        if rng.random() < eps_at(t):                            # 둘 다 탐색 (둘이 따로 동전을 던져도 결과는 비슷해요)
            a1, a2 = rng.integers(3), rng.integers(3)
        elif kind == "central":                                 # 결합 행동 9개 중 가장 좋은 것
            a1, a2 = np.unravel_index(np.argmax(Qj + rng.random((3, 3)) * 1e-9), (3, 3))
        else:                                                   # 각자 자기 Q로
            a1, a2 = int(np.argmax(Q1 + rng.random(3) * 1e-9)), int(np.argmax(Q2 + rng.random(3) * 1e-9))
        r = CLIMB[a1, a2]
        if kind == "central":
            Qj[a1, a2] += alpha * (r - Qj[a1, a2])
        elif kind == "independent":                             # 각자 상대가 없는 것처럼 배우기
            Q1[a1] += alpha * (r - Q1[a1])
            Q2[a2] += alpha * (r - Q2[a2])
        else:                                                   # vdn: Q1 + Q2가 보상을 맞히게
            td = r - (Q1[a1] + Q2[a2])
            Q1[a1] += alpha / 2 * td
            Q2[a2] += alpha / 2 * td
        if log:
            trace.append(Q1.copy())
    final = np.unravel_index(np.argmax(Qj), (3, 3)) if kind == "central" else (int(np.argmax(Q1)), int(np.argmax(Q2)))
    return tuple(int(x) for x in final), np.array(trace)

climb = {k: Counter(matrix_learn(k, s)[0] for s in range(100)) for k in ("independent", "vdn", "central")}

# ② 움직이는 과녁: 상대가 배우는 동안 내 Q가 흔들려요
_, q_trace = matrix_learn("independent", seed=0, log=True)      # 0번 에이전트의 Q 세 개가 시간에 따라 어떻게 바뀌나
def q_if_partner(policy2):                                      # 상대가 policy2로 고정돼 있다면 내 행동의 진짜 기대 보상
    return CLIMB @ policy2
uniform = q_if_partner(np.ones(3) / 3)                          # 상대가 아무거나 낼 때
partner_on_2 = q_if_partner(np.array([0.025, 0.025, 0.95]))     # 상대가 거의 2만 낼 때 (ε = 0.05)

# ③ 상태가 있는 협력: 둘이 함께 드는 상자
N, BOX = 6, (2, 3)
STARTS = ((5, 0), (5, 5))
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]              # 위, 오른쪽, 아래, 왼쪽, 가만히
LIFT, NA = 5, 6                                                 # 행동 5번은 "들기"
CELLS = [(r, c) for r in range(N) for c in range(N) if (r, c) != BOX]
CID = {c: i for i, c in enumerate(CELLS)}
NS = len(CELLS) ** 2                                            # 두 로봇의 위치 조합이 상태

def sid(p1, p2):
    return CID[p1] * len(CELLS) + CID[p2]

def next_to_box(p):
    return abs(p[0] - BOX[0]) + abs(p[1] - BOX[1]) == 1

def step(p1, p2, a1, a2):                                       # → 다음 위치 둘, 보상, 끝났나
    if a1 == LIFT and a2 == LIFT and next_to_box(p1) and next_to_box(p2):
        return p1, p2, 1.0, True                                # 둘이 상자 옆에서 같이 들면 성공
    r = -0.3 if (a1 == LIFT) != (a2 == LIFT) else 0.0           # 혼자 들려고 하면 허리 다쳐요
    moved = []
    for p, a in ((p1, a1), (p2, a2)):
        if a < LIFT:
            q = (p[0] + MOVES[a][0], p[1] + MOVES[a][1])
            if 0 <= q[0] < N and 0 <= q[1] < N and q != BOX:
                p = q
        moved.append(p)
    if moved[0] == moved[1]:                                    # 같은 칸으로 가면 부딪혀서 둘 다 제자리
        moved = [p1, p2]
    return moved[0], moved[1], r, False

GAMMA, MAX_T = 0.95, 60
def grid_learn(kind, seed, episodes=6000, alpha=0.2):
    rng = np.random.default_rng(seed)
    Qj = np.zeros((NS, NA * NA)) if kind == "central" else None   # 중앙: 결합 행동 36개
    Q1, Q2 = np.zeros((NS, NA)), np.zeros((NS, NA))             # 각자: 행동 6개
    def act(s, greedy):
        if not greedy and rng.random() < eps_now:
            return int(rng.integers(NA)), int(rng.integers(NA))
        if kind == "central":
            return divmod(int(np.argmax(Qj[s] + rng.random(NA * NA) * 1e-9)), NA)
        return int(np.argmax(Q1[s] + rng.random(NA) * 1e-9)), int(np.argmax(Q2[s] + rng.random(NA) * 1e-9))
    steps, curve = 0, []
    for ep in range(episodes):
        eps_now = max(0.05, 1 - ep / (episodes * 0.6))
        (p1, p2), t, done = STARTS, 0, False
        while not done and t < MAX_T:
            s = sid(p1, p2)
            a1, a2 = act(s, greedy=False)
            q1, q2, r, done = step(p1, p2, a1, a2)
            s2 = sid(q1, q2)
            if kind == "central":
                y = r + (0 if done else GAMMA * Qj[s2].max())
                Qj[s, a1 * NA + a2] += alpha * (y - Qj[s, a1 * NA + a2])
            elif kind == "independent":
                Q1[s, a1] += alpha * (r + (0 if done else GAMMA * Q1[s2].max()) - Q1[s, a1])
                Q2[s, a2] += alpha * (r + (0 if done else GAMMA * Q2[s2].max()) - Q2[s, a2])
            else:                                               # vdn: 팀 Q = Q1 + Q2, 각자 자기 max로 고르면 팀 max
                y = r + (0 if done else GAMMA * (Q1[s2].max() + Q2[s2].max()))
                td = y - (Q1[s, a1] + Q2[s, a2])
                Q1[s, a1] += alpha / 2 * td                     # 합이 한 번에 α만큼 움직이게 반씩
                Q2[s, a2] += alpha / 2 * td
            p1, p2, t, steps = q1, q2, t + 1, steps + 1
        if ep % 200 == 199:                                     # 탐색 없이 한 판 해 보기 (평가)
            (p1, p2), t, done = STARTS, 0, False
            while not done and t < MAX_T:
                a1, a2 = act(sid(p1, p2), greedy=True)
                p1, p2, _, done = step(p1, p2, a1, a2)
                t += 1
            curve.append((steps, t if done else MAX_T + 1))
    return np.array(curve)

grid = {k: [grid_learn(k, s) for s in range(5)] for k in ("central", "independent", "vdn")}
shortest = 6                                                    # 두 로봇이 상자 옆까지 걷고(5걸음) 함께 들기(1걸음)

# ④ 에이전트가 늘면: 표의 크기
def table_sizes(n_agents, positions=35, actions=6):             # 위치 조합 상태 × 행동 칸 수
    states = positions ** n_agents
    return {"중앙 (결합 행동)": states * actions ** n_agents, "각자 또는 가치 분해": n_agents * states * actions}

sizes = {n: table_sizes(n) for n in (2, 3, 4, 5)}
