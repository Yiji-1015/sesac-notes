window.PAGE = {
 "code": [
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 격자 환경 만들기",
   "s": 1
  },
  {
   "t": "H, W = 4, 8",
   "s": 1
  },
  {
   "t": "START, GOAL = (3, 0), (3, 7)",
   "s": 1
  },
  {
   "t": "STAIRS = {(3, c) for c in range(1, 7)}",
   "s": 1
  },
  {
   "t": "MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # 위, 오른쪽, 아래, 왼쪽",
   "s": 1
  },
  {
   "t": "ARROW = \"↑→↓←\"",
   "s": 1
  },
  {
   "t": "STEP_R, FALL_R, GOAL_R = -1.0, -20.0, 10.0  # 한 걸음 −1, 계단 −20, 목표 +10",
   "s": 1
  },
  {
   "t": "CELLS = [(r, c) for r in range(H) for c in range(W)]",
   "s": 1
  },
  {
   "t": "ID = {s: i for i, s in enumerate(CELLS)}",
   "s": 1
  },
  {
   "t": "nS, nA = len(CELLS), len(MOVES)",
   "s": 1
  },
  {
   "t": "END = STAIRS | {GOAL}  # 들어가면 끝나는 칸",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def move(s, d):  # 벽에 부딪히면 제자리",
   "s": 1
  },
  {
   "t": "    r, c = s[0] + MOVES[d][0], s[1] + MOVES[d][1]",
   "s": 1
  },
  {
   "t": "    return (r, c) if 0 <= r < H and 0 <= c < W else s",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def outcomes(s, a, slip):  # (확률, 다음 칸, 보상) 목록",
   "s": 1
  },
  {
   "t": "    if s in END:",
   "s": 1
  },
  {
   "t": "        return [(1.0, s, 0.0)]",
   "s": 1
  },
  {
   "t": "    probs = {}",
   "s": 1
  },
  {
   "t": "    for d, p in ((a, 1 - slip), ((a + 1) % 4, slip / 2), ((a + 3) % 4, slip / 2)):  # 미끄러지면 옆으로",
   "s": 1
  },
  {
   "t": "        probs[move(s, d)] = probs.get(move(s, d), 0) + p",
   "s": 1
  },
  {
   "t": "    return [(p, s2, FALL_R if s2 in STAIRS else GOAL_R if s2 == GOAL else STEP_R) for s2, p in probs.items() if p > 0]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def make_mdp(slip):  # 전이 확률 P[s, a, s']과 기대 보상 R[s, a]",
   "s": 1
  },
  {
   "t": "    P, R = np.zeros((nS, nA, nS)), np.zeros((nS, nA))",
   "s": 1
  },
  {
   "t": "    for s in CELLS:",
   "s": 1
  },
  {
   "t": "        for a in range(nA):",
   "s": 1
  },
  {
   "t": "            for p, s2, r in outcomes(s, a, slip):",
   "s": 1
  },
  {
   "t": "                if s not in END:",
   "s": 1
  },
  {
   "t": "                    P[ID[s], a, ID[s2]] += p  # 끝난 칸에서는 더 가지 않아요",
   "s": 1
  },
  {
   "t": "                R[ID[s], a] += p * r",
   "s": 1
  },
  {
   "t": "    return P, R",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "rng = np.random.default_rng(0)",
   "s": 1
  },
  {
   "t": "def play(policy, slip, max_steps=200):  # 정책대로 한 판 해 보기 → (보상 목록, 걸음 수)",
   "s": 1
  },
  {
   "t": "    s, rewards = START, []",
   "s": 1
  },
  {
   "t": "    while s not in END and len(rewards) < max_steps:",
   "s": 1
  },
  {
   "t": "        a = rng.choice(nA, p=policy[ID[s]])",
   "s": 1
  },
  {
   "t": "        o = outcomes(s, a, slip)",
   "s": 1
  },
  {
   "t": "        _, s, r = o[rng.choice(len(o), p=[x[0] for x in o])]",
   "s": 1
  },
  {
   "t": "        rewards.append(r)",
   "s": 1
  },
  {
   "t": "    return rewards",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "RANDOM = np.full((nS, nA), 1 / nA)  # 네 방향을 똑같이 고르는 정책",
   "s": 1
  },
  {
   "t": "P0, R0 = make_mdp(slip=0.0)",
   "s": 1
  },
  {
   "t": "random_games = [play(RANDOM, 0.0) for _ in range(1000)]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 가치 함수와 벨만 방정식",
   "s": 2
  },
  {
   "t": "GAMMA = 0.95",
   "s": 2
  },
  {
   "t": "def discounted(rewards, gamma=GAMMA):  # r₀ + γr₁ + γ²r₂ + ...",
   "s": 2
  },
  {
   "t": "    return sum(r * gamma ** t for t, r in enumerate(rewards))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def evaluate(policy, P, R, gamma=GAMMA):  # 벨만 방정식을 연립방정식으로 풀기: V = Rπ + γ Pπ V",
   "s": 2
  },
  {
   "t": "    P_pi = np.einsum(\"sa,sat->st\", policy, P)",
   "s": 2
  },
  {
   "t": "    R_pi = (policy * R).sum(axis=1)",
   "s": 2
  },
  {
   "t": "    return np.linalg.solve(np.eye(nS) - gamma * P_pi, R_pi)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def evaluate_by_sweeps(policy, P, R, gamma=GAMMA, tol=1e-8):  # 같은 식을 되풀이해서 덮어쓰기",
   "s": 2
  },
  {
   "t": "    V, sweeps = np.zeros(nS), 0",
   "s": 2
  },
  {
   "t": "    while True:",
   "s": 2
  },
  {
   "t": "        V_new = (policy * (R + gamma * P @ V)).sum(axis=1)",
   "s": 2
  },
  {
   "t": "        sweeps += 1",
   "s": 2
  },
  {
   "t": "        if np.abs(V_new - V).max() < tol:",
   "s": 2
  },
  {
   "t": "            return V_new, sweeps",
   "s": 2
  },
  {
   "t": "        V = V_new",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "V_random = evaluate(RANDOM, P0, R0)",
   "s": 2
  },
  {
   "t": "V_random_sweep, sweeps_random = evaluate_by_sweeps(RANDOM, P0, R0)",
   "s": 2
  },
  {
   "t": "mc_random = np.mean([discounted(g) for g in random_games])  # 1,000판의 할인 수익 평균",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 동적 계획법: 가치 반복과 정책 반복",
   "s": 3
  },
  {
   "t": "SLIP = 0.1",
   "s": 3
  },
  {
   "t": "P1, R1 = make_mdp(slip=SLIP)",
   "s": 3
  },
  {
   "t": "def greedy(Q):  # 행동 가치가 가장 큰 쪽으로 확률 1",
   "s": 3
  },
  {
   "t": "    pi = np.zeros((nS, nA))",
   "s": 3
  },
  {
   "t": "    pi[np.arange(nS), Q.argmax(axis=1)] = 1.0",
   "s": 3
  },
  {
   "t": "    return pi",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def value_iteration(P, R, gamma=GAMMA, tol=1e-8):",
   "s": 3
  },
  {
   "t": "    V, gaps = np.zeros(nS), []",
   "s": 3
  },
  {
   "t": "    while True:",
   "s": 3
  },
  {
   "t": "        Q = R + gamma * P @ V  # Q(s, a) = R(s, a) + γ Σ P(s'|s, a) V(s')",
   "s": 3
  },
  {
   "t": "        V_new = Q.max(axis=1)  # 벨만 최적 방정식",
   "s": 3
  },
  {
   "t": "        gaps.append(np.abs(V_new - V).max())",
   "s": 3
  },
  {
   "t": "        V = V_new",
   "s": 3
  },
  {
   "t": "        if gaps[-1] < tol:",
   "s": 3
  },
  {
   "t": "            return V, greedy(Q), gaps",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def policy_iteration(P, R, gamma=GAMMA):",
   "s": 3
  },
  {
   "t": "    pi, rounds = RANDOM.copy(), 0",
   "s": 3
  },
  {
   "t": "    while True:",
   "s": 3
  },
  {
   "t": "        V = evaluate(pi, P, R, gamma)  # 평가",
   "s": 3
  },
  {
   "t": "        new_pi = greedy(R + gamma * P @ V)  # 개선",
   "s": 3
  },
  {
   "t": "        rounds += 1",
   "s": 3
  },
  {
   "t": "        if np.array_equal(new_pi, pi):",
   "s": 3
  },
  {
   "t": "            return V, pi, rounds",
   "s": 3
  },
  {
   "t": "        pi = new_pi",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "V_star, pi_star, gaps = value_iteration(P1, R1)",
   "s": 3
  },
  {
   "t": "V_pi, pi_pi, rounds = policy_iteration(P1, R1)",
   "s": 3
  },
  {
   "t": "by_gamma = {g: value_iteration(P1, R1, gamma=g)[:2] for g in (0.5, 0.95, 0.99)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def route(pi, limit=20):  # 출발점에서 정책을 따라간 화살표 (미끄러짐 없이)",
   "s": 3
  },
  {
   "t": "    s, path = START, \"\"",
   "s": 3
  },
  {
   "t": "    while s not in END and len(path) < limit:",
   "s": 3
  },
  {
   "t": "        a = int(pi[ID[s]].argmax())",
   "s": 3
  },
  {
   "t": "        path += ARROW[a]",
   "s": 3
  },
  {
   "t": "        s = move(s, a)",
   "s": 3
  },
  {
   "t": "    return path",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 탐색과 활용: 슬롯머신 10대",
   "s": 4
  },
  {
   "t": "K, STEPS, RUNS = 10, 1000, 2000",
   "s": 4
  },
  {
   "t": "means = rng.normal(0, 1, (RUNS, K))  # 판마다 다른 슬롯머신 10대 (평균은 숨겨져 있어요)",
   "s": 4
  },
  {
   "t": "best = means.argmax(axis=1)",
   "s": 4
  },
  {
   "t": "bandit = {}",
   "s": 4
  },
  {
   "t": "for eps in (0.0, 0.01, 0.1, 0.3):",
   "s": 4
  },
  {
   "t": "    est, cnt = np.zeros((RUNS, K)), np.zeros((RUNS, K))",
   "s": 4
  },
  {
   "t": "    reward, optimal = np.zeros(STEPS), np.zeros(STEPS)",
   "s": 4
  },
  {
   "t": "    for t in range(STEPS):",
   "s": 4
  },
  {
   "t": "        explore = rng.random(RUNS) < eps",
   "s": 4
  },
  {
   "t": "        a = np.where(explore, rng.integers(K, size=RUNS), (est + rng.random((RUNS, K)) * 1e-9).argmax(axis=1))  # 동점이면 무작위",
   "s": 4
  },
  {
   "t": "        r = rng.normal(means[np.arange(RUNS), a], 1)",
   "s": 4
  },
  {
   "t": "        cnt[np.arange(RUNS), a] += 1",
   "s": 4
  },
  {
   "t": "        est[np.arange(RUNS), a] += (r - est[np.arange(RUNS), a]) / cnt[np.arange(RUNS), a]  # 표본 평균을 한 건씩 갱신",
   "s": 4
  },
  {
   "t": "        reward[t], optimal[t] = r.mean(), (a == best).mean()",
   "s": 4
  },
  {
   "t": "    bandit[eps] = (reward, optimal)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 해 보면서 가치 재기: 몬테카를로와 TD(0)",
   "s": 5
  },
  {
   "t": "def soft(pi, eps):  # ε 확률로 아무 방향",
   "s": 5
  },
  {
   "t": "    return (1 - eps) * pi + eps / nA",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "FIXED = soft(pi_star, 0.2)  # 잴 정책: ③의 최적 정책에 무작위 20%",
   "s": 5
  },
  {
   "t": "V_true = evaluate(FIXED, P1, R1)",
   "s": 5
  },
  {
   "t": "def mc_vs_td(episodes=300, alpha=0.05, seed=0):",
   "s": 5
  },
  {
   "t": "    global rng",
   "s": 5
  },
  {
   "t": "    rng = np.random.default_rng(seed)",
   "s": 5
  },
  {
   "t": "    V_mc, n_mc, V_td, err = np.zeros(nS), np.zeros(nS), np.zeros(nS), []",
   "s": 5
  },
  {
   "t": "    for _ in range(episodes):",
   "s": 5
  },
  {
   "t": "        s, steps = START, []",
   "s": 5
  },
  {
   "t": "        while s not in END and len(steps) < 200:",
   "s": 5
  },
  {
   "t": "            a = rng.choice(nA, p=FIXED[ID[s]])",
   "s": 5
  },
  {
   "t": "            o = outcomes(s, a, SLIP)",
   "s": 5
  },
  {
   "t": "            _, s2, r = o[rng.choice(len(o), p=[x[0] for x in o])]",
   "s": 5
  },
  {
   "t": "            target = r + (0 if s2 in END else GAMMA * V_td[ID[s2]])  # TD(0): 한 걸음 보고 바로 고치기",
   "s": 5
  },
  {
   "t": "            V_td[ID[s]] += alpha * (target - V_td[ID[s]])",
   "s": 5
  },
  {
   "t": "            steps.append((s, r))",
   "s": 5
  },
  {
   "t": "            s = s2",
   "s": 5
  },
  {
   "t": "        G = 0.0",
   "s": 5
  },
  {
   "t": "        for s_t, r_t in reversed(steps):  # 몬테카를로: 판이 끝난 뒤 뒤에서부터 수익 계산",
   "s": 5
  },
  {
   "t": "            G = r_t + GAMMA * G",
   "s": 5
  },
  {
   "t": "            n_mc[ID[s_t]] += 1",
   "s": 5
  },
  {
   "t": "            V_mc[ID[s_t]] += (G - V_mc[ID[s_t]]) / n_mc[ID[s_t]]",
   "s": 5
  },
  {
   "t": "        err.append((abs(V_mc[ID[START]] - V_true[ID[START]]), abs(V_td[ID[START]] - V_true[ID[START]])))",
   "s": 5
  },
  {
   "t": "    return np.array(err)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "pred_err = np.mean([mc_vs_td(seed=s) for s in range(20)], axis=0)  # 20번 반복 평균",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 모델 없이 정책 배우기: SARSA와 Q-learning",
   "s": 6
  },
  {
   "t": "def td_control(kind, episodes=1000, eps=0.1, alpha=0.5, seed=0, slip=0.0):",
   "s": 6
  },
  {
   "t": "    global rng",
   "s": 6
  },
  {
   "t": "    rng = np.random.default_rng(seed)",
   "s": 6
  },
  {
   "t": "    P, R = make_mdp(slip)",
   "s": 6
  },
  {
   "t": "    Q, steps_so_far, curve = np.zeros((nS, nA)), 0, []",
   "s": 6
  },
  {
   "t": "    def pick(s):  # ε-greedy",
   "s": 6
  },
  {
   "t": "        return int(rng.integers(nA)) if rng.random() < eps else int((Q[ID[s]] + rng.random(nA) * 1e-9).argmax())",
   "s": 6
  },
  {
   "t": "    for ep in range(episodes):",
   "s": 6
  },
  {
   "t": "        s, a, G = START, pick(START), 0.0",
   "s": 6
  },
  {
   "t": "        while s not in END and steps_so_far < 10 ** 7:",
   "s": 6
  },
  {
   "t": "            o = outcomes(s, a, slip)",
   "s": 6
  },
  {
   "t": "            _, s2, r = o[rng.choice(len(o), p=[x[0] for x in o])]",
   "s": 6
  },
  {
   "t": "            a2 = pick(s2)",
   "s": 6
  },
  {
   "t": "            if s2 in END:",
   "s": 6
  },
  {
   "t": "                target = r",
   "s": 6
  },
  {
   "t": "            elif kind == \"sarsa\":",
   "s": 6
  },
  {
   "t": "                target = r + GAMMA * Q[ID[s2], a2]  # 다음에 실제로 고른 행동의 값",
   "s": 6
  },
  {
   "t": "            else:",
   "s": 6
  },
  {
   "t": "                target = r + GAMMA * Q[ID[s2]].max()  # 다음 칸에서 가장 좋은 행동의 값",
   "s": 6
  },
  {
   "t": "            Q[ID[s], a] += alpha * (target - Q[ID[s], a])",
   "s": 6
  },
  {
   "t": "            s, a, G, steps_so_far = s2, a2, G + r, steps_so_far + 1",
   "s": 6
  },
  {
   "t": "        curve.append((steps_so_far, G, evaluate(greedy(Q), P, R)[ID[START]]))  # (누적 걸음, 이번 판 수익, 지금 greedy 정책의 진짜 가치)",
   "s": 6
  },
  {
   "t": "    return Q, np.array(curve)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "runs = {kind: [td_control(kind, seed=s) for s in range(10)] for kind in (\"sarsa\", \"q\")}",
   "s": 6
  },
  {
   "t": "V_opt0 = value_iteration(P0, R0)[0][ID[START]]  # 미끄러짐 없는 환경의 최적 가치",
   "s": 6
  }
 ],
 "names": {
  "1": "격자 환경 만들기",
  "2": "가치 함수와 벨만 방정식",
  "3": "동적 계획법: 가치 반복과 정책 반복",
  "4": "탐색과 활용: 슬롯머신 10대",
  "5": "해 보면서 가치 재기: 몬테카를로와 TD(0)",
  "6": "모델 없이 정책 배우기: SARSA와 Q-learning"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "s1": {
   "title": "① 네 방향을 똑같이 고르는 로봇으로 1,000판",
   "lines": [
    [
     "칸 32개 (그중 끝나는 칸 7개), 행동 4개",
     1
    ],
    [
     "한 판 평균 걸음 → 11.0",
     2
    ],
    [
     "계단에 떨어진 판 → 98.8%, 목표에 닿은 판 → 1.2%",
     3,
     "newf"
    ]
   ],
   "notes": [
    "출발 칸 바로 오른쪽이 계단이라, 아무렇게나 걸으면 거의 다 떨어져요. 무엇을 해야 하는지 정답은 아무도 안 알려 주고, 판이 끝나야 보상으로만 알 수 있어요."
   ]
  },
  "s2": {
   "title": "② 무작위 정책의 가치 (γ = 0.95, 미끄러짐 없음)",
   "lines": [
    [
     "연립방정식으로 푼 V(출발) → -19.958",
     1
    ],
    [
     "같은 식을 되풀이해 덮어쓰기 → 196번 만에 같은 값 (가장 큰 차이 9e-08)",
     2
    ],
    [
     "①의 1,000판에서 직접 잰 할인 수익 평균 → -19.919",
     3,
     "newf"
    ]
   ],
   "notes": [
    "계산한 값과 직접 해 본 평균이 거의 같아요. 가치 함수는 \"이 정책으로 여기서 시작하면 평균 얼마를 받나\"예요.",
    "출발 칸의 가치가 −20 가까이 되는 건, 거의 늘 한두 걸음 만에 계단(−20)에 떨어지기 때문이에요."
   ]
  },
  "s3": {
   "title": "③ 미끄러짐 10%인 환경에서 최적 정책 찾기 (γ = 0.95)",
   "lines": [
    [
     "가치 반복 → 35번 만에 변화가 1e−8 아래",
     1
    ],
    [
     "한 번 덮어쓸 때마다 최대 변화가 줄어든 배수 → 0.90, 0.89, 0.89, 0.89, 0.89 (γ = 0.95 이하)",
     2,
     "newf"
    ],
    [
     "정책 반복 → 평가와 개선 3번 만에 같은 정책, 두 방법의 가치 차이 4e-09",
     3
    ],
    [
     "출발점에서 따라간 길 → ↑↑→→→→→→→↓↓, V*(출발) = -4.469",
     4,
     "newf"
    ]
   ],
   "notes": [
    "벨만 최적 방정식으로 한 번 덮어쓸 때마다 \"지금 값과 진짜 값의 가장 큰 차이\"가 γ배 이하로 줄어요. 그래서 반드시 하나의 답으로 모여요.",
    "미끄러지면 옆 칸으로 가니까, 계단 바로 위 줄을 걸으면 10%의 반, 곧 5%씩 떨어질 위험이 있어요. 그래서 두 줄 위로 돌아가요. 한 걸음 더 걷는 −1이 떨어지는 −20의 5%보다 싸요."
   ]
  },
  "t3": {
   "title": "할인율 γ를 바꾸면 (미끄러짐 10%)",
   "cols": [
    "γ",
    "출발점에서 따라간 길",
    "V*(출발)"
   ],
   "rows": [
    [
     0.5,
     "← (벽에 부딪혀 제자리)",
     "-1.999"
    ],
    [
     "<b>0.95</b> (지금)",
     "↑↑→→→→→→→↓↓",
     "-4.469"
    ],
    [
     0.99,
     "↑↑→→→→→→→↓↓",
     "-3.329"
    ]
   ],
   "notes": [
    "γ = 0.5면 10걸음 뒤의 +10은 0.5¹⁰ ≈ 0.001배로 줄어들어 거의 0이에요. 그래서 목표까지 가는 동안의 −1들이 더 커 보여서, 아예 움직이지 않고 벽에 부딪히는 게(매 걸음 −1이지만 끝나지도 않는) 최적이 돼요.",
    "γ는 \"나중 보상을 얼마나 믿을지\"를 정하는 학습 목표의 일부예요. 학습 방법의 손잡이가 아니라 문제 정의라서, 바꾸면 최적 정책 자체가 바뀌어요."
   ]
  },
  "f_vstar": {
   "title": "가치 반복으로 찾은 최적 정책 (γ = 0.95)",
   "images": [
    [
     "img/rl/grid_optimal.png",
     ""
    ]
   ],
   "notes": [
    "화살표가 각 칸에서 고를 방향, 색이 그 칸의 최적 가치예요. 목표에 가까울수록 진해요.",
    "미끄러짐이 있으면 계단 위 두 줄로 돌아가고(왼쪽), 없으면 계단 바로 위 줄로 가요(오른쪽, V*(출발) = -0.097)."
   ],
   "chart": true
  },
  "f_gaps": {
   "title": "가치 반복의 수렴",
   "images": [
    [
     "img/rl/value_iteration_gaps.png",
     ""
    ]
   ],
   "notes": [
    "처음 몇 번은 목표의 소식이 칸에서 칸으로 퍼지는 중이라 변화가 커요. 소식이 다 퍼진 뒤에는 γ 기울기보다 빠르게 줄어요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ 슬롯머신 10대, 1,000번 당기기 (2,000판 평균)",
   "cols": [
    "ε",
    "마지막 100번 평균 보상",
    "마지막 100번 최적 머신을 고른 비율",
    "1,000번 전체 평균 보상"
   ],
   "rows": [
    [
     "<b>0.0</b>",
     "1.035",
     "36.1%",
     "1.027"
    ],
    [
     "<b>0.01</b>",
     "1.320",
     "59.4%",
     "1.198"
    ],
    [
     "<b>0.1</b>",
     "1.381",
     "80.1%",
     "1.316"
    ],
    [
     "<b>0.3</b>",
     "1.089",
     "67.5%",
     "1.061"
    ]
   ],
   "notes": [
    "ε = 0(늘 지금 가장 좋아 보이는 것만)은 처음에 운 좋게 괜찮았던 머신에 갇혀서 최적 머신을 36%만 골라요.",
    "ε = 0.1이 가장 좋았어요. ε = 0.3은 최적 머신을 알아낸 뒤에도 30%를 아무 데나 써서 평균 보상이 다시 낮아요. ε = 0.01은 천천히 찾지만, 더 오래 하면 결국 0.1을 앞질러요(덜 낭비하니까)."
   ]
  },
  "f_bandit": {
   "title": "탐색 비율 ε에 따른 차이 (2,000판 평균)",
   "images": [
    [
     "img/rl/bandit.png",
     ""
    ]
   ],
   "notes": [
    "회색(ε = 0)은 빨리 올라가다 멈춰요. 더 좋은 머신이 있는지 확인하지 않기 때문이에요."
   ],
   "chart": true
  },
  "s5": {
   "title": "⑤ 정책 하나의 V(출발) 재기 (진짜 값 -8.137, 20번 반복 평균 오차)",
   "cols": [
    "판 수",
    "몬테카를로",
    "TD(0), α = 0.05"
   ],
   "rows": [
    [
     10,
     "1.569",
     "6.624"
    ],
    [
     50,
     "0.654",
     "3.547"
    ],
    [
     150,
     "0.421",
     "1.041"
    ],
    [
     300,
     "0.316",
     "0.465"
    ]
   ],
   "notes": [
    "책에서는 TD가 흔들림이 적어 보통 더 빨리 맞는다고 해요. 그런데 이 문제의 출발 칸에서는 몬테카를로가 계속 앞섰어요.",
    "출발 칸은 매 판 지나가니까 몬테카를로는 판마다 진짜 수익 표본을 하나씩 얻어 정확히 평균을 내요. TD(0)은 목표의 +10 소식이 칸에서 칸으로 한 걸음씩 거꾸로 전해져야 해서, 처음 수십 판은 출발 칸의 값이 0 근처에 머물러요."
   ]
  },
  "t5": {
   "title": "TD(0)의 α를 바꾸면 (출발 칸 오차, 20번 반복 평균)",
   "cols": [
    "α",
    "50판 뒤",
    "150판 뒤",
    "300판 뒤",
    "몬테카를로 300판 뒤"
   ],
   "rows": [
    [
     0.02,
     "5.662",
     "3.318",
     "1.582",
     "0.316"
    ],
    [
     "<b>0.05</b> (지금)",
     "3.547",
     "1.041",
     "0.465",
     "0.316"
    ],
    [
     0.1,
     "1.596",
     "0.901",
     "0.558",
     "0.316"
    ],
    [
     0.2,
     "1.216",
     "1.383",
     "0.882",
     "0.316"
    ],
    [
     0.4,
     "1.913",
     "2.025",
     "1.813",
     "0.316"
    ]
   ],
   "notes": [
    "α가 작으면 소식이 느리게 퍼지고, 크면 빨리 퍼지지만 한 판 한 판에 너무 흔들려서 오차가 다시 커져요. 0.1 근처가 가장 나았지만 몬테카를로는 못 이겼어요.",
    "TD의 장점은 판이 아주 길거나 끝나지 않는 문제, 칸끼리 값을 나눠 쓰는 문제에서 커요. 어느 쪽이 나은지는 문제마다 달라서, 둘을 섞는 TD(λ)도 있어요."
   ]
  },
  "f_pred": {
   "title": "몬테카를로와 TD(0)로 가치 재기",
   "images": [
    [
     "img/rl/mc_vs_td.png",
     ""
    ]
   ],
   "notes": [
    "TD(0)은 처음 수십 판 동안 0 근처에서 출발해, 목표의 소식이 출발 칸까지 전해질 때까지 오차가 커요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ 미끄러짐 없는 환경, ε = 0.1, α = 0.5, 1,000판 (10번 반복)",
   "cols": [
    "",
    "SARSA",
    "Q-learning"
   ],
   "rows": [
    [
     "greedy 정책의 출발점 가치 (최적 -0.097)",
     "-4.885",
     "-0.097"
    ],
    [
     "계단 바로 위 최단 길을 배운 횟수",
     "0 / 10",
     "10 / 10"
    ],
    [
     "학습 중 마지막 200판의 평균 수익",
     "-4.30",
     "-3.36"
    ],
    [
     "1,000판 동안 걸은 걸음 수 (평균)",
     "14,089",
     "9,654"
    ]
   ],
   "notes": [
    "Q-learning은 10번 모두 최적 길(↑→→→→→→→↓)을 배웠어요. SARSA는 대부분 계단에서 한 줄 이상 떨어진 길을 배웠어요(0번만 최단 길).",
    "SARSA는 \"다음에 실제로 고를 행동\"의 값으로 배워요. 학습 중에는 10% 확률로 아무 방향이나 가니까, 계단 바로 위에서는 그 무작위 한 걸음에 떨어질 수 있다는 걸 값에 넣어요. 그래서 학습 중의 자기 모습에 맞는 안전한 길을 배워요.",
    "Q-learning은 \"다음 칸에서 가장 좋은 행동\"의 값으로 배워요. 무작위 한 걸음은 계산에 안 넣고, 나중에 greedy로만 움직일 때의 최적 길을 배워요.",
    "학습 중 평균 수익은 이번에는 Q-learning이 더 높았어요(-3.36 대 -4.30). SARSA의 돌아가는 길이 더 길고, 아직 덜 배운 판도 있었기 때문이에요. \"SARSA가 학습 중에는 더 안전하다\"는 건 계단에서 잃는 점수가 길을 돌아가는 비용보다 훨씬 클 때 두드러져요."
   ]
  },
  "f_td": {
   "title": "SARSA와 Q-learning이 배운 것",
   "images": [
    [
     "img/rl/sarsa_vs_q.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽 x축은 판 수가 아니라 <b>누적 걸음 수</b>예요. 판 수로 그리면 한 판에 오래 헤맨 쪽이 경험을 더 많이 모은 걸 놓쳐요.",
    "Q-learning은 10번 모두 2,511걸음 안쪽부터 계속 최적이고, SARSA는 돌아가는 길에 머물러 최적보다 낮게 오르내려요."
   ],
   "chart": true
  },
  "t6": {
   "title": "SARSA의 greedy 길 10개",
   "cols": [
    "길",
    "횟수"
   ],
   "rows": [
    [
     "↑↑→→→→→→→↓↓",
     2
    ],
    [
     "↑↑→→→→→→↓→↓",
     1
    ],
    [
     "↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓",
     1
    ],
    [
     "↑→↑→→→→↑→↓→↓↓",
     1
    ],
    [
     "↑↑→↑→→→→→↓→↓↓",
     1
    ],
    [
     "↑→↑↑→→→→↓→→↓↓",
     1
    ],
    [
     "↑→↑→↑→→→→→↓↓↓",
     1
    ],
    [
     "↑↑→→↑→↓→→→→↓↓",
     1
    ],
    [
     "↑↑↑→→→→→→↓→↓↓",
     1
    ]
   ],
   "notes": [
    "↓만 계속되는 한 개는 출발 칸에서 아래 벽에 부딪히는 쪽을 골라 1,000판 안에 아직 덜 배운 경우예요. ε을 서서히 0으로 줄이면 SARSA도 결국 최단 길로 가요."
   ]
  }
 },
 "flow": [
  [
   1,
   "환경",
   "4 × 8 격자, 계단",
   "보상 −1, −20, +10"
  ],
  [
   2,
   "가치 함수",
   "벨만 방정식",
   "연립방정식"
  ],
  [
   3,
   "동적 계획법",
   "모델을 알 때",
   "γ배씩 수렴"
  ],
  [
   4,
   "탐색과 활용",
   "슬롯머신 10대",
   "ε = 0.1"
  ],
  [
   5,
   "MC와 TD",
   "해 보면서 가치 재기",
   "여기선 MC"
  ],
  [
   6,
   "SARSA와 Q",
   "모델 없이 정책 배우기",
   "안전 대 최단"
  ]
 ],
 "env": "Python 3.13.5, NumPy 2.5.2 (seed 0)"
};
