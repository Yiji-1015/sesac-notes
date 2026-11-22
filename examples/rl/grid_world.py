# 격자 위 배달 로봇으로 강화학습 기초를 직접 해 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 환경: 4 × 8 칸. 왼쪽 아래에서 출발해 오른쪽 아래 목표로 가요. 아래 줄 가운데 6칸은 계단이라 떨어지면 끝나요.
# 데이터 파일 없이 코드가 직접 만든 환경이에요. CPU로 1분 안쪽.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import numpy as np                                                              #@1

# ① 격자 환경 만들기
H, W = 4, 8
START, GOAL = (3, 0), (3, 7)
STAIRS = {(3, c) for c in range(1, 7)}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]                  # 위, 오른쪽, 아래, 왼쪽
ARROW = "↑→↓←"
STEP_R, FALL_R, GOAL_R = -1.0, -20.0, 10.0                  # 한 걸음 −1, 계단 −20, 목표 +10
CELLS = [(r, c) for r in range(H) for c in range(W)]
ID = {s: i for i, s in enumerate(CELLS)}
nS, nA = len(CELLS), len(MOVES)
END = STAIRS | {GOAL}                                       # 들어가면 끝나는 칸

def move(s, d):                                             # 벽에 부딪히면 제자리
    r, c = s[0] + MOVES[d][0], s[1] + MOVES[d][1]
    return (r, c) if 0 <= r < H and 0 <= c < W else s

def outcomes(s, a, slip):                                   # (확률, 다음 칸, 보상) 목록
    if s in END:
        return [(1.0, s, 0.0)]
    probs = {}
    for d, p in ((a, 1 - slip), ((a + 1) % 4, slip / 2), ((a + 3) % 4, slip / 2)):   # 미끄러지면 옆으로
        probs[move(s, d)] = probs.get(move(s, d), 0) + p
    return [(p, s2, FALL_R if s2 in STAIRS else GOAL_R if s2 == GOAL else STEP_R) for s2, p in probs.items() if p > 0]

def make_mdp(slip):                                         # 전이 확률 P[s, a, s']과 기대 보상 R[s, a]
    P, R = np.zeros((nS, nA, nS)), np.zeros((nS, nA))
    for s in CELLS:
        for a in range(nA):
            for p, s2, r in outcomes(s, a, slip):
                if s not in END:
                    P[ID[s], a, ID[s2]] += p                # 끝난 칸에서는 더 가지 않아요
                R[ID[s], a] += p * r
    return P, R

rng = np.random.default_rng(0)
def play(policy, slip, max_steps=200):                      # 정책대로 한 판 해 보기 → (보상 목록, 걸음 수)
    s, rewards = START, []
    while s not in END and len(rewards) < max_steps:
        a = rng.choice(nA, p=policy[ID[s]])
        o = outcomes(s, a, slip)
        _, s, r = o[rng.choice(len(o), p=[x[0] for x in o])]
        rewards.append(r)
    return rewards

RANDOM = np.full((nS, nA), 1 / nA)                          # 네 방향을 똑같이 고르는 정책
P0, R0 = make_mdp(slip=0.0)
random_games = [play(RANDOM, 0.0) for _ in range(1000)]

# ② 가치 함수와 벨만 방정식
GAMMA = 0.95
def discounted(rewards, gamma=GAMMA):                       # r₀ + γr₁ + γ²r₂ + ...
    return sum(r * gamma ** t for t, r in enumerate(rewards))

def evaluate(policy, P, R, gamma=GAMMA):                    # 벨만 방정식을 연립방정식으로 풀기: V = Rπ + γ Pπ V
    P_pi = np.einsum("sa,sat->st", policy, P)
    R_pi = (policy * R).sum(axis=1)
    return np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)

def evaluate_by_sweeps(policy, P, R, gamma=GAMMA, tol=1e-8):   # 같은 식을 되풀이해서 덮어쓰기
    V, sweeps = np.zeros(nS), 0
    while True:
        V_new = (policy * (R + gamma * P @ V)).sum(axis=1)
        sweeps += 1
        if np.abs(V_new - V).max() < tol:
            return V_new, sweeps
        V = V_new

V_random = evaluate(RANDOM, P0, R0)
V_random_sweep, sweeps_random = evaluate_by_sweeps(RANDOM, P0, R0)
mc_random = np.mean([discounted(g) for g in random_games])  # 1,000판의 할인 수익 평균

# ③ 동적 계획법: 가치 반복과 정책 반복
SLIP = 0.1
P1, R1 = make_mdp(slip=SLIP)
def greedy(Q):                                              # 행동 가치가 가장 큰 쪽으로 확률 1
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), Q.argmax(axis=1)] = 1.0
    return pi

def value_iteration(P, R, gamma=GAMMA, tol=1e-8):
    V, gaps = np.zeros(nS), []
    while True:
        Q = R + gamma * P @ V                               # Q(s, a) = R(s, a) + γ Σ P(s'|s, a) V(s')
        V_new = Q.max(axis=1)                               # 벨만 최적 방정식
        gaps.append(np.abs(V_new - V).max())
        V = V_new
        if gaps[-1] < tol:
            return V, greedy(Q), gaps

def policy_iteration(P, R, gamma=GAMMA):
    pi, rounds = RANDOM.copy(), 0
    while True:
        V = evaluate(pi, P, R, gamma)                       # 평가
        new_pi = greedy(R + gamma * P @ V)                  # 개선
        rounds += 1
        if np.array_equal(new_pi, pi):
            return V, pi, rounds
        pi = new_pi

V_star, pi_star, gaps = value_iteration(P1, R1)
V_pi, pi_pi, rounds = policy_iteration(P1, R1)
by_gamma = {g: value_iteration(P1, R1, gamma=g)[:2] for g in (0.5, 0.95, 0.99)}

def route(pi, limit=20):                                    # 출발점에서 정책을 따라간 화살표 (미끄러짐 없이)
    s, path = START, ""
    while s not in END and len(path) < limit:
        a = int(pi[ID[s]].argmax())
        path += ARROW[a]
        s = move(s, a)
    return path

# ④ 탐색과 활용: 슬롯머신 10대
K, STEPS, RUNS = 10, 1000, 2000
means = rng.normal(0, 1, (RUNS, K))                         # 판마다 다른 슬롯머신 10대 (평균은 숨겨져 있어요)
best = means.argmax(axis=1)
bandit = {}
for eps in (0.0, 0.01, 0.1, 0.3):
    est, cnt = np.zeros((RUNS, K)), np.zeros((RUNS, K))
    reward, optimal = np.zeros(STEPS), np.zeros(STEPS)
    for t in range(STEPS):
        explore = rng.random(RUNS) < eps
        a = np.where(explore, rng.integers(K, size=RUNS), (est + rng.random((RUNS, K)) * 1e-9).argmax(axis=1))   # 동점이면 무작위
        r = rng.normal(means[np.arange(RUNS), a], 1)
        cnt[np.arange(RUNS), a] += 1
        est[np.arange(RUNS), a] += (r - est[np.arange(RUNS), a]) / cnt[np.arange(RUNS), a]   # 표본 평균을 한 건씩 갱신
        reward[t], optimal[t] = r.mean(), (a == best).mean()
    bandit[eps] = (reward, optimal)

# ⑤ 해 보면서 가치 재기: 몬테카를로와 TD(0)
def soft(pi, eps):                                          # ε 확률로 아무 방향
    return (1 - eps) * pi + eps / nA

FIXED = soft(pi_star, 0.2)                                  # 잴 정책: ③의 최적 정책에 무작위 20%
V_true = evaluate(FIXED, P1, R1)
def mc_vs_td(episodes=300, alpha=0.05, seed=0):
    global rng
    rng = np.random.default_rng(seed)
    V_mc, n_mc, V_td, err = np.zeros(nS), np.zeros(nS), np.zeros(nS), []
    for _ in range(episodes):
        s, steps = START, []
        while s not in END and len(steps) < 200:
            a = rng.choice(nA, p=FIXED[ID[s]])
            o = outcomes(s, a, SLIP)
            _, s2, r = o[rng.choice(len(o), p=[x[0] for x in o])]
            target = r + (0 if s2 in END else GAMMA * V_td[ID[s2]])   # TD(0): 한 걸음 보고 바로 고치기
            V_td[ID[s]] += alpha * (target - V_td[ID[s]])
            steps.append((s, r))
            s = s2
        G = 0.0
        for s_t, r_t in reversed(steps):                    # 몬테카를로: 판이 끝난 뒤 뒤에서부터 수익 계산
            G = r_t + GAMMA * G
            n_mc[ID[s_t]] += 1
            V_mc[ID[s_t]] += (G - V_mc[ID[s_t]]) / n_mc[ID[s_t]]
        err.append((abs(V_mc[ID[START]] - V_true[ID[START]]), abs(V_td[ID[START]] - V_true[ID[START]])))
    return np.array(err)

pred_err = np.mean([mc_vs_td(seed=s) for s in range(20)], axis=0)   # 20번 반복 평균

# ⑥ 모델 없이 정책 배우기: SARSA와 Q-learning
def td_control(kind, episodes=1000, eps=0.1, alpha=0.5, seed=0, slip=0.0):
    global rng
    rng = np.random.default_rng(seed)
    P, R = make_mdp(slip)
    Q, steps_so_far, curve = np.zeros((nS, nA)), 0, []
    def pick(s):                                            # ε-greedy
        return int(rng.integers(nA)) if rng.random() < eps else int((Q[ID[s]] + rng.random(nA) * 1e-9).argmax())
    for ep in range(episodes):
        s, a, G = START, pick(START), 0.0
        while s not in END and steps_so_far < 10 ** 7:
            o = outcomes(s, a, slip)
            _, s2, r = o[rng.choice(len(o), p=[x[0] for x in o])]
            a2 = pick(s2)
            if s2 in END:
                target = r
            elif kind == "sarsa":
                target = r + GAMMA * Q[ID[s2], a2]          # 다음에 실제로 고른 행동의 값
            else:
                target = r + GAMMA * Q[ID[s2]].max()        # 다음 칸에서 가장 좋은 행동의 값
            Q[ID[s], a] += alpha * (target - Q[ID[s], a])
            s, a, G, steps_so_far = s2, a2, G + r, steps_so_far + 1
        curve.append((steps_so_far, G, evaluate(greedy(Q), P, R)[ID[START]]))   # (누적 걸음, 이번 판 수익, 지금 greedy 정책의 진짜 가치)
    return Q, np.array(curve)

runs = {kind: [td_control(kind, seed=s) for s in range(10)] for kind in ("sarsa", "q")}
V_opt0 = value_iteration(P0, R0)[0][ID[START]]              # 미끄러짐 없는 환경의 최적 가치
