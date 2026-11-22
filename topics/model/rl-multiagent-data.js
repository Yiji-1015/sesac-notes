window.PAGE = {
 "code": [
  {
   "t": "from collections import Counter",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 각자 배우기 vs 함께 배우기: 오르막 게임",
   "s": 1
  },
  {
   "t": "CLIMB = np.array([[11, -30, 0],  # 둘이 받는 보상이 같은(공통 보상) 협력 게임",
   "s": 1
  },
  {
   "t": "                  [-30, 7, 6],  # (0, 0)이 11로 가장 좋지만, 한쪽만 0을 내면 −30",
   "s": 1
  },
  {
   "t": "                  [0, 0, 5]])",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def eps_at(t, horizon=3000):  # 처음엔 거의 무작위, 3,000판에 걸쳐 0.05까지",
   "s": 1
  },
  {
   "t": "    return max(0.05, 1 - t / horizon)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def matrix_learn(kind, seed, rounds=5000, alpha=0.1, log=False):",
   "s": 1
  },
  {
   "t": "    rng = np.random.default_rng(seed)",
   "s": 1
  },
  {
   "t": "    Q1, Q2, Qj, trace = np.zeros(3), np.zeros(3), np.zeros((3, 3)), []",
   "s": 1
  },
  {
   "t": "    for t in range(rounds):",
   "s": 1
  },
  {
   "t": "        if rng.random() < eps_at(t):  # 둘 다 탐색 (둘이 따로 동전을 던져도 결과는 비슷해요)",
   "s": 1
  },
  {
   "t": "            a1, a2 = rng.integers(3), rng.integers(3)",
   "s": 1
  },
  {
   "t": "        elif kind == \"central\":  # 결합 행동 9개 중 가장 좋은 것",
   "s": 1
  },
  {
   "t": "            a1, a2 = np.unravel_index(np.argmax(Qj + rng.random((3, 3)) * 1e-9), (3, 3))",
   "s": 1
  },
  {
   "t": "        else:  # 각자 자기 Q로",
   "s": 1
  },
  {
   "t": "            a1, a2 = int(np.argmax(Q1 + rng.random(3) * 1e-9)), int(np.argmax(Q2 + rng.random(3) * 1e-9))",
   "s": 1
  },
  {
   "t": "        r = CLIMB[a1, a2]",
   "s": 1
  },
  {
   "t": "        if kind == \"central\":",
   "s": 1
  },
  {
   "t": "            Qj[a1, a2] += alpha * (r - Qj[a1, a2])",
   "s": 1
  },
  {
   "t": "        elif kind == \"independent\":  # 각자 상대가 없는 것처럼 배우기",
   "s": 1
  },
  {
   "t": "            Q1[a1] += alpha * (r - Q1[a1])",
   "s": 1
  },
  {
   "t": "            Q2[a2] += alpha * (r - Q2[a2])",
   "s": 1
  },
  {
   "t": "        else:  # vdn: Q1 + Q2가 보상을 맞히게",
   "s": 1
  },
  {
   "t": "            td = r - (Q1[a1] + Q2[a2])",
   "s": 1
  },
  {
   "t": "            Q1[a1] += alpha / 2 * td",
   "s": 1
  },
  {
   "t": "            Q2[a2] += alpha / 2 * td",
   "s": 1
  },
  {
   "t": "        if log:",
   "s": 1
  },
  {
   "t": "            trace.append(Q1.copy())",
   "s": 1
  },
  {
   "t": "    final = np.unravel_index(np.argmax(Qj), (3, 3)) if kind == \"central\" else (int(np.argmax(Q1)), int(np.argmax(Q2)))",
   "s": 1
  },
  {
   "t": "    return tuple(int(x) for x in final), np.array(trace)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "climb = {k: Counter(matrix_learn(k, s)[0] for s in range(100)) for k in (\"independent\", \"vdn\", \"central\")}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 움직이는 과녁: 상대가 배우는 동안 내 Q가 흔들려요",
   "s": 2
  },
  {
   "t": "_, q_trace = matrix_learn(\"independent\", seed=0, log=True)  # 0번 에이전트의 Q 세 개가 시간에 따라 어떻게 바뀌나",
   "s": 2
  },
  {
   "t": "def q_if_partner(policy2):  # 상대가 policy2로 고정돼 있다면 내 행동의 진짜 기대 보상",
   "s": 2
  },
  {
   "t": "    return CLIMB @ policy2",
   "s": 2
  },
  {
   "t": "uniform = q_if_partner(np.ones(3) / 3)  # 상대가 아무거나 낼 때",
   "s": 2
  },
  {
   "t": "partner_on_2 = q_if_partner(np.array([0.025, 0.025, 0.95]))  # 상대가 거의 2만 낼 때 (ε = 0.05)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 상태가 있는 협력: 둘이 함께 드는 상자",
   "s": 3
  },
  {
   "t": "N, BOX = 6, (2, 3)",
   "s": 3
  },
  {
   "t": "STARTS = ((5, 0), (5, 5))",
   "s": 3
  },
  {
   "t": "MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]  # 위, 오른쪽, 아래, 왼쪽, 가만히",
   "s": 3
  },
  {
   "t": "LIFT, NA = 5, 6  # 행동 5번은 \"들기\"",
   "s": 3
  },
  {
   "t": "CELLS = [(r, c) for r in range(N) for c in range(N) if (r, c) != BOX]",
   "s": 3
  },
  {
   "t": "CID = {c: i for i, c in enumerate(CELLS)}",
   "s": 3
  },
  {
   "t": "NS = len(CELLS) ** 2  # 두 로봇의 위치 조합이 상태",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def sid(p1, p2):",
   "s": 3
  },
  {
   "t": "    return CID[p1] * len(CELLS) + CID[p2]",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def next_to_box(p):",
   "s": 3
  },
  {
   "t": "    return abs(p[0] - BOX[0]) + abs(p[1] - BOX[1]) == 1",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def step(p1, p2, a1, a2):  # → 다음 위치 둘, 보상, 끝났나",
   "s": 3
  },
  {
   "t": "    if a1 == LIFT and a2 == LIFT and next_to_box(p1) and next_to_box(p2):",
   "s": 3
  },
  {
   "t": "        return p1, p2, 1.0, True  # 둘이 상자 옆에서 같이 들면 성공",
   "s": 3
  },
  {
   "t": "    r = -0.3 if (a1 == LIFT) != (a2 == LIFT) else 0.0  # 혼자 들려고 하면 허리 다쳐요",
   "s": 3
  },
  {
   "t": "    moved = []",
   "s": 3
  },
  {
   "t": "    for p, a in ((p1, a1), (p2, a2)):",
   "s": 3
  },
  {
   "t": "        if a < LIFT:",
   "s": 3
  },
  {
   "t": "            q = (p[0] + MOVES[a][0], p[1] + MOVES[a][1])",
   "s": 3
  },
  {
   "t": "            if 0 <= q[0] < N and 0 <= q[1] < N and q != BOX:",
   "s": 3
  },
  {
   "t": "                p = q",
   "s": 3
  },
  {
   "t": "        moved.append(p)",
   "s": 3
  },
  {
   "t": "    if moved[0] == moved[1]:  # 같은 칸으로 가면 부딪혀서 둘 다 제자리",
   "s": 3
  },
  {
   "t": "        moved = [p1, p2]",
   "s": 3
  },
  {
   "t": "    return moved[0], moved[1], r, False",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "GAMMA, MAX_T = 0.95, 60",
   "s": 3
  },
  {
   "t": "def grid_learn(kind, seed, episodes=6000, alpha=0.2):",
   "s": 3
  },
  {
   "t": "    rng = np.random.default_rng(seed)",
   "s": 3
  },
  {
   "t": "    Qj = np.zeros((NS, NA * NA)) if kind == \"central\" else None  # 중앙: 결합 행동 36개",
   "s": 3
  },
  {
   "t": "    Q1, Q2 = np.zeros((NS, NA)), np.zeros((NS, NA))  # 각자: 행동 6개",
   "s": 3
  },
  {
   "t": "    def act(s, greedy):",
   "s": 3
  },
  {
   "t": "        if not greedy and rng.random() < eps_now:",
   "s": 3
  },
  {
   "t": "            return int(rng.integers(NA)), int(rng.integers(NA))",
   "s": 3
  },
  {
   "t": "        if kind == \"central\":",
   "s": 3
  },
  {
   "t": "            return divmod(int(np.argmax(Qj[s] + rng.random(NA * NA) * 1e-9)), NA)",
   "s": 3
  },
  {
   "t": "        return int(np.argmax(Q1[s] + rng.random(NA) * 1e-9)), int(np.argmax(Q2[s] + rng.random(NA) * 1e-9))",
   "s": 3
  },
  {
   "t": "    steps, curve = 0, []",
   "s": 3
  },
  {
   "t": "    for ep in range(episodes):",
   "s": 3
  },
  {
   "t": "        eps_now = max(0.05, 1 - ep / (episodes * 0.6))",
   "s": 3
  },
  {
   "t": "        (p1, p2), t, done = STARTS, 0, False",
   "s": 3
  },
  {
   "t": "        while not done and t < MAX_T:",
   "s": 3
  },
  {
   "t": "            s = sid(p1, p2)",
   "s": 3
  },
  {
   "t": "            a1, a2 = act(s, greedy=False)",
   "s": 3
  },
  {
   "t": "            q1, q2, r, done = step(p1, p2, a1, a2)",
   "s": 3
  },
  {
   "t": "            s2 = sid(q1, q2)",
   "s": 3
  },
  {
   "t": "            if kind == \"central\":",
   "s": 3
  },
  {
   "t": "                y = r + (0 if done else GAMMA * Qj[s2].max())",
   "s": 3
  },
  {
   "t": "                Qj[s, a1 * NA + a2] += alpha * (y - Qj[s, a1 * NA + a2])",
   "s": 3
  },
  {
   "t": "            elif kind == \"independent\":",
   "s": 3
  },
  {
   "t": "                Q1[s, a1] += alpha * (r + (0 if done else GAMMA * Q1[s2].max()) - Q1[s, a1])",
   "s": 3
  },
  {
   "t": "                Q2[s, a2] += alpha * (r + (0 if done else GAMMA * Q2[s2].max()) - Q2[s, a2])",
   "s": 3
  },
  {
   "t": "            else:  # vdn: 팀 Q = Q1 + Q2, 각자 자기 max로 고르면 팀 max",
   "s": 3
  },
  {
   "t": "                y = r + (0 if done else GAMMA * (Q1[s2].max() + Q2[s2].max()))",
   "s": 3
  },
  {
   "t": "                td = y - (Q1[s, a1] + Q2[s, a2])",
   "s": 3
  },
  {
   "t": "                Q1[s, a1] += alpha / 2 * td  # 합이 한 번에 α만큼 움직이게 반씩",
   "s": 3
  },
  {
   "t": "                Q2[s, a2] += alpha / 2 * td",
   "s": 3
  },
  {
   "t": "            p1, p2, t, steps = q1, q2, t + 1, steps + 1",
   "s": 3
  },
  {
   "t": "        if ep % 200 == 199:  # 탐색 없이 한 판 해 보기 (평가)",
   "s": 3
  },
  {
   "t": "            (p1, p2), t, done = STARTS, 0, False",
   "s": 3
  },
  {
   "t": "            while not done and t < MAX_T:",
   "s": 3
  },
  {
   "t": "                a1, a2 = act(sid(p1, p2), greedy=True)",
   "s": 3
  },
  {
   "t": "                p1, p2, _, done = step(p1, p2, a1, a2)",
   "s": 3
  },
  {
   "t": "                t += 1",
   "s": 3
  },
  {
   "t": "            curve.append((steps, t if done else MAX_T + 1))",
   "s": 3
  },
  {
   "t": "    return np.array(curve)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "grid = {k: [grid_learn(k, s) for s in range(5)] for k in (\"central\", \"independent\", \"vdn\")}",
   "s": 3
  },
  {
   "t": "shortest = 6  # 두 로봇이 상자 옆까지 걷고(5걸음) 함께 들기(1걸음)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 에이전트가 늘면: 표의 크기",
   "s": 4
  },
  {
   "t": "def table_sizes(n_agents, positions=35, actions=6):  # 위치 조합 상태 × 행동 칸 수",
   "s": 4
  },
  {
   "t": "    states = positions ** n_agents",
   "s": 4
  },
  {
   "t": "    return {\"중앙 (결합 행동)\": states * actions ** n_agents, \"각자 또는 가치 분해\": n_agents * states * actions}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "sizes = {n: table_sizes(n) for n in (2, 3, 4, 5)}",
   "s": 4
  }
 ],
 "names": {
  "1": "각자 배우기 vs 함께 배우기: 오르막 게임",
  "2": "움직이는 과녁: 상대가 배우는 동안 내 Q가 흔들려요",
  "3": "상태가 있는 협력: 둘이 함께 드는 상자",
  "4": "에이전트가 늘면: 표의 크기"
 },
 "range": [
  1,
  4
 ],
 "outs": {
  "s1": {
   "title": "① 오르막 게임을 5,000판씩, 씨앗 100개 (마지막에 고르는 결합 행동)",
   "cols": [
    "방식",
    "고른 결합 행동 (횟수)",
    "그때 보상"
   ],
   "rows": [
    [
     "각자 배우기 (독립 Q-learning)",
     "(2, 2) 100번",
     "5"
    ],
    [
     "가치 분해 (VDN, Q₁ + Q₂)",
     "(2, 2) 99번, (1, 2) 1번",
     "5, 6"
    ],
    [
     "함께 배우기 (결합 행동 하나의 Q)",
     "(0, 0) 100번",
     "11"
    ]
   ],
   "notes": [
    "가장 좋은 (0, 0)의 11은 바로 옆에 −30이 붙어 있어요. 각자 배우면 \"내가 0을 냈을 때\"의 평균을 보는데, 상대가 아직 탐색 중이라 자주 1이나 2를 내니 0의 평균이 −30에 끌려 내려가요. 그래서 둘 다 안전한 2(평균이 0 아래로 안 내려가는)로 가요.",
    "결합 행동 9칸 각각의 값을 따로 배우면 (0, 0)이 11이라는 걸 그대로 알아요. 대신 에이전트가 늘면 결합 행동 수가 곱으로 늘어요(④).",
    "가치 분해 Q₁(a₁) + Q₂(a₂)도 실패했어요. 더하기로는 \"(0, 0)은 11인데 (0, 1)은 −30\"처럼 둘의 조합에 따라 크게 뒤집히는 값을 표현할 수 없어요."
   ]
  },
  "s2": {
   "title": "② 0번 에이전트가 기억하는 Q와, 상대 정책에 따른 진짜 기대 보상",
   "cols": [
    "",
    "행동 0",
    "행동 1",
    "행동 2"
   ],
   "rows": [
    [
     "상대가 셋을 1/3씩 낼 때 (진짜)",
     "-6.33",
     "-5.67",
     "1.67"
    ],
    [
     "상대가 거의 2만 낼 때 (진짜)",
     "-0.47",
     "5.12",
     "4.75"
    ],
    [
     "5,000판 뒤 기억하는 Q",
     "-11.28",
     "-2.06",
     "5.00"
    ]
   ],
   "notes": [
    "상대가 거의 2만 내는 지금, 내가 1을 내면 5.12로 2(4.75)보다 나아요(행 1, 열 2의 6점). 그런데 내 Q(1)은 -2.06에 머물러 있어요. 상대가 아무거나 내던 시절에 배운 값이고, 그 뒤로 1을 거의 안 내서 고칠 기회가 없었어요.",
    "상대의 정책이 바뀌면, 나에게는 <b>환경이 바뀐 것</b>과 같아요. 1쪽의 Q-learning은 환경이 바뀌지 않는다(정상성)고 가정해서 이런 수렴 보장이 깨져요. 이게 <b>비정상성</b>, 또는 움직이는 과녁 문제예요."
   ]
  },
  "f_q": {
   "title": "움직이는 과녁",
   "images": [
    [
     "img/rl/independent_q_trace.png",
     ""
    ]
   ],
   "notes": [
    "처음엔 상대가 아무거나 내서 Q(0)과 Q(1)이 −30에 자주 끌려 떨어져요. 상대가 2에 자리 잡은 뒤에는 Q(2)만 5로 올라가고, 거의 안 고르는 Q(0), Q(1)은 가끔 탐색할 때만 계단처럼 고쳐져서, 상대가 바뀐 지금의 진짜 값을 따라가지 못해요."
   ],
   "chart": true
  },
  "s3": {
   "title": "③ 상자를 함께 들기까지 걸린 걸음 수 (탐색 없이 평가, 씨앗 5개 평균, 최단 6걸음)",
   "cols": [
    "방식",
    "1,000판",
    "2,000판",
    "3,000판",
    "6,000판",
    "씨앗별 마지막"
   ],
   "rows": [
    [
     "중앙 (결합 행동 36개)",
     "6.6",
     "6.2",
     "6.2",
     "6.2",
     "6, 7, 6, 6, 6"
    ],
    [
     "각자 (독립 Q-learning)",
     "28.8",
     "18.6",
     "9.2",
     "7.8",
     "10, 6, 8, 7, 8"
    ],
    [
     "가치 분해 (VDN)",
     "50.6",
     "39.8",
     "19.6",
     "9.4",
     "10, 8, 10, 8, 11"
    ]
   ],
   "notes": [
    "61은 60걸음 안에 못 든 판이에요. 둘뿐이면 결합 행동이 36개라 중앙 학습도 감당할 만해서, 가장 빨리, 가장 짧게(평균 6.2걸음) 배웠어요.",
    "각자 배우기와 가치 분해도 6,000판 뒤에는 모두 성공했지만 몇 걸음 돌아가요(평균 7.8, 9.4). 이 문제에서는 가치 분해가 각자 배우기보다 낫지 않았어요. 가치 분해가 이득을 보는 건 에이전트가 많아 중앙 학습을 못 쓰고, 팀 보상만 있어서 \"누구 덕인지\" 나눠야 할 때예요."
   ]
  },
  "f_grid": {
   "title": "둘이 함께 드는 상자",
   "images": [
    [
     "img/rl/lift_box.png",
     ""
    ]
   ],
   "notes": [
    "두 로봇이 상자 옆 칸(초록 점선)에 서서 같은 걸음에 함께 \"들기\"를 해야 +1을 받고 끝나요. 한쪽만 들려고 하면 −0.3이에요. 보상은 팀이 함께 받아요(공통 보상).",
    "보라(중앙)가 1,000판 안쪽에 최단 근처로 내려와요. 파랑과 분홍은 3,000판 근처부터 10걸음 안쪽으로 내려오고, 그 뒤에도 가끔 다시 길어져요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ 로봇 수에 따른 Q 표의 칸 수 (한 로봇이 설 수 있는 칸 35개, 행동 6개)",
   "cols": [
    "로봇 수",
    "결합 행동 수",
    "중앙 Q 표",
    "각자 또는 가치 분해 Q 표",
    "배수"
   ],
   "rows": [
    [
     2,
     "36",
     "44,100",
     "14,700",
     "3배"
    ],
    [
     3,
     "216",
     "9,261,000",
     "771,750",
     "12배"
    ],
    [
     4,
     "1,296",
     "1,944,810,000",
     "36,015,000",
     "54배"
    ],
    [
     5,
     "7,776",
     "408,410,100,000",
     "1,575,656,250",
     "259배"
    ]
   ],
   "notes": [
    "상태(모두의 위치 조합)는 둘 다 로봇 수에 따라 거듭제곱으로 늘어요. 그래서 실제로는 표 대신 신경망을 써요(2쪽). 차이는 행동 쪽이에요. 중앙은 결합 행동이 6ⁿ개라 n이 하나 늘 때마다 6배, 각자는 n × 6개라 거의 그대로예요.",
    "다섯이면 결합 행동이 7,776개예요. 신경망이 행동마다 값을 하나씩 내야 하니, 출력이 7,776개인 신경망을 만들고 그중 최대를 찾아야 해요. 각자 배우기나 가치 분해는 로봇마다 6개만 내면 돼요."
   ]
  }
 },
 "flow": [
  [
   1,
   "오르막 게임",
   "각자 vs 함께",
   "−30이 무서워서"
  ],
  [
   2,
   "움직이는 과녁",
   "상대가 배우는 중",
   "옛 Q에 갇힘"
  ],
  [
   3,
   "함께 들기",
   "중앙, 각자, 분해",
   "둘이면 중앙"
  ],
  [
   4,
   "규모",
   "6ⁿ 대 n × 6",
   "분해가 필요"
  ]
 ],
 "env": "Python 3.13.5, NumPy 2.5.2"
};
