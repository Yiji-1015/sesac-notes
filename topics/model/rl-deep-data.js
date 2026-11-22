window.PAGE = {
 "code": [
  {
   "t": "import math",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 2
  },
  {
   "t": "from torch import nn",
   "s": 2
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 막대 세우기 환경",
   "s": 1
  },
  {
   "t": "class Pole:  # n판을 한꺼번에 굴려요",
   "s": 1
  },
  {
   "t": "    def __init__(self, n, seed):",
   "s": 1
  },
  {
   "t": "        self.n, self.rng = n, np.random.default_rng(seed)",
   "s": 1
  },
  {
   "t": "        self.s = self.rng.uniform(-0.05, 0.05, (n, 4))  # 수레 위치, 수레 속도, 막대 각도, 막대 각속도",
   "s": 1
  },
  {
   "t": "        self.t = np.zeros(n, dtype=int)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "    def step(self, a):  # a: 0 = 왼쪽으로 밀기, 1 = 오른쪽으로 밀기",
   "s": 1
  },
  {
   "t": "        x, v, th, w = self.s.T",
   "s": 1
  },
  {
   "t": "        force = np.where(a == 1, 10.0, -10.0)",
   "s": 1
  },
  {
   "t": "        cos, sin = np.cos(th), np.sin(th)",
   "s": 1
  },
  {
   "t": "        tmp = (force + 0.05 * w ** 2 * sin) / 1.1  # 수레 1kg, 막대 0.1kg, 막대 반 길이 0.5m",
   "s": 1
  },
  {
   "t": "        w_acc = (9.8 * sin - cos * tmp) / (0.5 * (4 / 3 - 0.1 * cos ** 2 / 1.1))",
   "s": 1
  },
  {
   "t": "        x_acc = tmp - 0.05 * w_acc * cos / 1.1",
   "s": 1
  },
  {
   "t": "        x, v, th, w = x + 0.02 * v, v + 0.02 * x_acc, th + 0.02 * w, w + 0.02 * w_acc  # 0.02초씩",
   "s": 1
  },
  {
   "t": "        self.s, self.t = np.stack([x, v, th, w], axis=1), self.t + 1",
   "s": 1
  },
  {
   "t": "        fell = (np.abs(x) > 2.4) | (np.abs(th) > 12 * math.pi / 180)  # 벗어나거나 12도 넘게 기울면 실패",
   "s": 1
  },
  {
   "t": "        done = fell | (self.t >= 200)  # 200걸음 버티면 성공으로 끝",
   "s": 1
  },
  {
   "t": "        s_next = self.s.copy()",
   "s": 1
  },
  {
   "t": "        k = np.flatnonzero(done)",
   "s": 1
  },
  {
   "t": "        self.s[k], self.t[k] = self.rng.uniform(-0.05, 0.05, (len(k), 4)), 0  # 끝난 판은 새로 시작",
   "s": 1
  },
  {
   "t": "        return s_next, np.ones(self.n), fell, done",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "env = Pole(1000, seed=0)",
   "s": 1
  },
  {
   "t": "lengths, alive = [], np.zeros(1000, dtype=int)",
   "s": 1
  },
  {
   "t": "while len(lengths) < 1000:  # 무작위로 밀면 몇 걸음 버티나",
   "s": 1
  },
  {
   "t": "    _, _, _, done = env.step(env.rng.integers(2, size=1000))",
   "s": 1
  },
  {
   "t": "    alive += 1",
   "s": 1
  },
  {
   "t": "    lengths += alive[done].tolist()",
   "s": 1
  },
  {
   "t": "    alive[done] = 0",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② Q를 신경망으로: 경험 재생과 타깃 네트워크 (DQN)",
   "s": 2
  },
  {
   "t": "torch.set_num_threads(1)",
   "s": 2
  },
  {
   "t": "def mlp(n_out):  # 상태 4개 → 은닉 64 → 64 → 출력",
   "s": 2
  },
  {
   "t": "    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "GAMMA = 0.99",
   "s": 2
  },
  {
   "t": "def dqn(seed, steps=20000, replay=True, target=True, batch=64, sync=500):",
   "s": 2
  },
  {
   "t": "    torch.manual_seed(seed)",
   "s": 2
  },
  {
   "t": "    rng, env = np.random.default_rng(seed), Pole(1, seed)",
   "s": 2
  },
  {
   "t": "    q, q_target = mlp(2), mlp(2)",
   "s": 2
  },
  {
   "t": "    q_target.load_state_dict(q.state_dict())",
   "s": 2
  },
  {
   "t": "    opt = torch.optim.Adam(q.parameters(), lr=1e-3)",
   "s": 2
  },
  {
   "t": "    buf, n = np.zeros((50_000, 11), dtype=np.float32), 0  # 재생 버퍼: (s 4칸, a, r, s' 4칸, 실패 여부)",
   "s": 2
  },
  {
   "t": "    s, length, curve = env.s[0].copy(), 0, []",
   "s": 2
  },
  {
   "t": "    for t in range(steps):",
   "s": 2
  },
  {
   "t": "        eps = max(0.05, 1 - t / 5000)  # 처음엔 거의 무작위, 5,000걸음에 걸쳐 0.05까지",
   "s": 2
  },
  {
   "t": "        a = int(rng.integers(2)) if rng.random() < eps else int(q(torch.tensor(s, dtype=torch.float32)).argmax())",
   "s": 2
  },
  {
   "t": "        s2, r, fell, done = env.step(np.array([a]))",
   "s": 2
  },
  {
   "t": "        buf[n % len(buf)] = np.r_[s, a, r[0], s2[0], fell[0]]",
   "s": 2
  },
  {
   "t": "        n, length = n + 1, length + 1",
   "s": 2
  },
  {
   "t": "        if done[0]:",
   "s": 2
  },
  {
   "t": "            curve.append((t + 1, length))",
   "s": 2
  },
  {
   "t": "            s, length = env.s[0].copy(), 0",
   "s": 2
  },
  {
   "t": "        else:",
   "s": 2
  },
  {
   "t": "            s = s2[0]",
   "s": 2
  },
  {
   "t": "        if n >= batch:",
   "s": 2
  },
  {
   "t": "            rows = rng.integers(min(n, len(buf)), size=batch) if replay else np.array([(n - 1) % len(buf)])  # 재생 없으면 방금 것 하나만",
   "s": 2
  },
  {
   "t": "            b = torch.tensor(buf[rows])",
   "s": 2
  },
  {
   "t": "            with torch.no_grad():",
   "s": 2
  },
  {
   "t": "                nxt = (q_target if target else q)(b[:, 6:10]).max(dim=1).values",
   "s": 2
  },
  {
   "t": "                y = b[:, 5] + GAMMA * nxt * (1 - b[:, 10])  # 넘어진 뒤의 가치는 0",
   "s": 2
  },
  {
   "t": "            pred = q(b[:, :4]).gather(1, b[:, 4:5].long()).squeeze(1)",
   "s": 2
  },
  {
   "t": "            loss = ((y - pred) ** 2).mean()",
   "s": 2
  },
  {
   "t": "            opt.zero_grad()",
   "s": 2
  },
  {
   "t": "            loss.backward()",
   "s": 2
  },
  {
   "t": "            opt.step()",
   "s": 2
  },
  {
   "t": "        if target and t % sync == 0:",
   "s": 2
  },
  {
   "t": "            q_target.load_state_dict(q.state_dict())  # 타깃은 가끔만 따라오게",
   "s": 2
  },
  {
   "t": "    return np.array(curve)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "ablation = {(rp, tg): [dqn(sd, replay=rp, target=tg) for sd in range(3)] for rp in (False, True) for tg in (False, True)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 정책을 직접 배우기: REINFORCE와 베이스라인",
   "s": 3
  },
  {
   "t": "def rollout(policy, env, T):  # T걸음 동안 n판을 굴려 기록 모으기",
   "s": 3
  },
  {
   "t": "    S, A, R, D, LP = [], [], [], [], []",
   "s": 3
  },
  {
   "t": "    for _ in range(T):",
   "s": 3
  },
  {
   "t": "        s = torch.tensor(env.s, dtype=torch.float32)",
   "s": 3
  },
  {
   "t": "        with torch.no_grad():",
   "s": 3
  },
  {
   "t": "            dist = torch.distributions.Categorical(logits=policy(s))",
   "s": 3
  },
  {
   "t": "            a = dist.sample()",
   "s": 3
  },
  {
   "t": "        _, r, _, done = env.step(a.numpy())",
   "s": 3
  },
  {
   "t": "        S.append(s), A.append(a), R.append(r), D.append(done), LP.append(dist.log_prob(a))",
   "s": 3
  },
  {
   "t": "    f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)  # noqa: E731",
   "s": 3
  },
  {
   "t": "    return torch.stack(S), torch.stack(A), f(R), f(D), torch.stack(LP)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def returns_to_go(R, D, last=None):  # 뒤에서부터 G_t = r_t + γ G_{t+1}, 판이 끝나면 끊기",
   "s": 3
  },
  {
   "t": "    G, run = torch.zeros_like(R), torch.zeros(R.shape[1]) if last is None else last",
   "s": 3
  },
  {
   "t": "    for t in reversed(range(len(R))):",
   "s": 3
  },
  {
   "t": "        run = R[t] + GAMMA * run * (1 - D[t])",
   "s": 3
  },
  {
   "t": "        G[t] = run",
   "s": 3
  },
  {
   "t": "    return G",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def grad_spread(baseline, batches=30, seed=0):  # 같은 정책에서 기울기를 30번 재서 얼마나 흔들리나",
   "s": 3
  },
  {
   "t": "    torch.manual_seed(seed)",
   "s": 3
  },
  {
   "t": "    policy, env, grads = mlp(2), Pole(8, seed), []",
   "s": 3
  },
  {
   "t": "    for _ in range(batches):",
   "s": 3
  },
  {
   "t": "        S, A, R, D, _ = rollout(policy, env, 200)",
   "s": 3
  },
  {
   "t": "        G = returns_to_go(R, D)",
   "s": 3
  },
  {
   "t": "        w = G - G.mean() if baseline else G  # 베이스라인: 평균 수익을 빼기",
   "s": 3
  },
  {
   "t": "        loss = -(torch.distributions.Categorical(logits=policy(S)).log_prob(A) * w).mean()",
   "s": 3
  },
  {
   "t": "        policy.zero_grad()",
   "s": 3
  },
  {
   "t": "        loss.backward()",
   "s": 3
  },
  {
   "t": "        grads.append(torch.cat([p.grad.flatten() for p in policy.parameters()]).clone())",
   "s": 3
  },
  {
   "t": "    g = torch.stack(grads)",
   "s": 3
  },
  {
   "t": "    return g.var(dim=0).sum().item(), g.mean(dim=0).norm().item()  # 흔들림(분산의 합), 평균 방향의 크기",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "spread = {b: grad_spread(b) for b in (False, True)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 몇 걸음마다 배우기: 액터-크리틱 (A2C)",
   "s": 4
  },
  {
   "t": "A2C_LAMBDA = 1.0  # 1이면 5걸음 동안 실제로 받은 보상 + 5걸음 뒤의 가치, 0이면 한 걸음 TD",
   "s": 4
  },
  {
   "t": "def advantages(R, D, V, V_last, lam):  # GAE: TD 오차를 λ로 섞어 더하기 (λ = 0이면 한 걸음 TD)",
   "s": 4
  },
  {
   "t": "    adv, gae = torch.zeros_like(R), torch.zeros(R.shape[1])",
   "s": 4
  },
  {
   "t": "    for t in reversed(range(len(R))):",
   "s": 4
  },
  {
   "t": "        nxt = V_last if t == len(R) - 1 else V[t + 1]",
   "s": 4
  },
  {
   "t": "        delta = R[t] + GAMMA * nxt * (1 - D[t]) - V[t]  # TD 오차",
   "s": 4
  },
  {
   "t": "        gae = delta + GAMMA * lam * (1 - D[t]) * gae",
   "s": 4
  },
  {
   "t": "        adv[t] = gae",
   "s": 4
  },
  {
   "t": "    return adv",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 너무 크게 바뀌지 않게: PPO",
   "s": 5
  },
  {
   "t": "def train_pg(kind, seed, steps=60000, n_env=8, lr=None):",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(seed)",
   "s": 5
  },
  {
   "t": "    env, policy, critic = Pole(n_env, seed), mlp(2), mlp(1)",
   "s": 5
  },
  {
   "t": "    T = {\"reinforce\": 200, \"reinforce_b\": 200, \"a2c\": 5, \"ppo\": 128}[kind]",
   "s": 5
  },
  {
   "t": "    lr = lr or {\"reinforce\": 3e-3, \"reinforce_b\": 3e-3, \"a2c\": 1e-3, \"ppo\": 1e-3}[kind]",
   "s": 5
  },
  {
   "t": "    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=lr)",
   "s": 5
  },
  {
   "t": "    total, lengths, alive, curve = 0, [], np.zeros(n_env), []",
   "s": 5
  },
  {
   "t": "    while total < steps:",
   "s": 5
  },
  {
   "t": "        S, A, R, D, LP_old = rollout(policy, env, T)",
   "s": 5
  },
  {
   "t": "        for t in range(T):  # 판 길이 기록",
   "s": 5
  },
  {
   "t": "            alive += 1",
   "s": 5
  },
  {
   "t": "            lengths += alive[D[t].numpy() > 0].tolist()",
   "s": 5
  },
  {
   "t": "            alive[D[t].numpy() > 0] = 0",
   "s": 5
  },
  {
   "t": "        total += T * n_env",
   "s": 5
  },
  {
   "t": "        if kind in (\"reinforce\", \"reinforce_b\"):",
   "s": 5
  },
  {
   "t": "            G = returns_to_go(R, D)",
   "s": 5
  },
  {
   "t": "            w = G - G.mean() if kind == \"reinforce_b\" else G",
   "s": 5
  },
  {
   "t": "            loss = -(torch.distributions.Categorical(logits=policy(S)).log_prob(A) * w).mean()",
   "s": 5
  },
  {
   "t": "        else:",
   "s": 5
  },
  {
   "t": "            with torch.no_grad():",
   "s": 5
  },
  {
   "t": "                V, V_last = critic(S).squeeze(-1), critic(torch.tensor(env.s, dtype=torch.float32)).squeeze(-1)",
   "s": 5
  },
  {
   "t": "            adv = advantages(R, D, V, V_last, lam=A2C_LAMBDA if kind == \"a2c\" else 0.95)",
   "s": 5
  },
  {
   "t": "            target_v = adv + V",
   "s": 5
  },
  {
   "t": "        if kind == \"a2c\":",
   "s": 5
  },
  {
   "t": "            loss = -(torch.distributions.Categorical(logits=policy(S)).log_prob(A) * adv).mean() + 0.5 * ((target_v - critic(S).squeeze(-1)) ** 2).mean()",
   "s": 5
  },
  {
   "t": "        if kind == \"ppo\":",
   "s": 5
  },
  {
   "t": "            S, A, LP_old, adv, target_v = S.reshape(-1, 4), A.reshape(-1), LP_old.reshape(-1), adv.reshape(-1), target_v.reshape(-1)",
   "s": 5
  },
  {
   "t": "            adv = (adv - adv.mean()) / (adv.std() + 1e-8)",
   "s": 5
  },
  {
   "t": "            for _ in range(4):  # 같은 기록으로 4번 다시 배우기",
   "s": 5
  },
  {
   "t": "                for mb in torch.randperm(len(A)).split(256):",
   "s": 5
  },
  {
   "t": "                    ratio = torch.exp(torch.distributions.Categorical(logits=policy(S[mb])).log_prob(A[mb]) - LP_old[mb])  # 새 정책 ÷ 옛 정책",
   "s": 5
  },
  {
   "t": "                    clipped = torch.min(ratio * adv[mb], ratio.clamp(0.8, 1.2) * adv[mb])  # 20% 넘게 바뀐 몫은 안 쳐 주기",
   "s": 5
  },
  {
   "t": "                    loss = -clipped.mean() + 0.5 * ((target_v[mb] - critic(S[mb]).squeeze(-1)) ** 2).mean()",
   "s": 5
  },
  {
   "t": "                    opt.zero_grad()",
   "s": 5
  },
  {
   "t": "                    loss.backward()",
   "s": 5
  },
  {
   "t": "                    opt.step()",
   "s": 5
  },
  {
   "t": "        else:",
   "s": 5
  },
  {
   "t": "            opt.zero_grad()",
   "s": 5
  },
  {
   "t": "            loss.backward()",
   "s": 5
  },
  {
   "t": "            opt.step()",
   "s": 5
  },
  {
   "t": "        curve.append((total, np.mean(lengths[-20:]) if lengths else 0.0))",
   "s": 5
  },
  {
   "t": "    return np.array(curve)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "pg_runs = {k: [train_pg(k, sd) for sd in range(3)] for k in (\"reinforce\", \"reinforce_b\", \"a2c\", \"ppo\")}",
   "s": 5
  }
 ],
 "names": {
  "1": "막대 세우기 환경",
  "2": "Q를 신경망으로: 경험 재생과 타깃 네트워크 (DQN)",
  "3": "정책을 직접 배우기: REINFORCE와 베이스라인",
  "4": "몇 걸음마다 배우기: 액터-크리틱 (A2C)",
  "5": "너무 크게 바뀌지 않게: PPO"
 },
 "range": [
  1,
  5
 ],
 "outs": {
  "s1": {
   "title": "① 무작위로 밀면 (1,000판)",
   "lines": [
    [
     "평균 17.2걸음, 가운데값 16걸음, 가장 오래 30걸음 (최대 200)",
     1
    ]
   ],
   "notes": [
    "상태는 수레 위치, 속도, 막대 각도, 각속도의 실수 4개예요. 1쪽처럼 칸을 셀 수 없어서 Q를 표로 적을 수 없어요. 비슷한 상태끼리 값을 나눠 쓰는 함수가 필요해요."
   ]
  },
  "s2": {
   "title": "② 네 가지 구성, 씨앗 3개 평균 (막대를 버틴 걸음 수, 최대 200)",
   "cols": [
    "구성",
    "5천 걸음",
    "1만",
    "1만 5천",
    "2만",
    "씨앗별 마지막 5판"
   ],
   "rows": [
    [
     "둘 다 없음",
     "88.3",
     "130.0",
     "100.2",
     "79.0",
     "36.4, 159.0, 60.2"
    ],
    [
     "타깃 네트워크만",
     "68.8",
     "44.0",
     "34.5",
     "160.7",
     "136.8, 144.4, 155.6"
    ],
    [
     "경험 재생만",
     "170.7",
     "196.8",
     "191.3",
     "200.0",
     "193.6, 200.0, 200.0"
    ],
    [
     "둘 다 (DQN)",
     "187.3",
     "141.6",
     "143.2",
     "151.7",
     "161.4, 168.0, 127.2"
    ]
   ],
   "notes": [
    "책의 실험(작은 격자 게임)에서는 두 장치가 <b>함께</b> 있어야 안정됐어요. 이 막대 세우기에서는 <b>경험 재생이 결정적</b>이었어요. 둘 다 쓴 DQN은 7,500걸음쯤까지 경험 재생만 쓴 쪽과 비슷하게 올라갔다가, 그 뒤 150 근처로 내려앉았어요.",
    "타깃 네트워크는 500걸음마다만 따라와서, 그 사이에 배운 것이 목표 값에 늦게 반영돼요. 이 작은 문제에서는 그 늦음이 득보다 실이었던 것으로 보여요. 씨앗 3개, 2만 걸음의 결과라 일반화하긴 어렵고, 어느 장치가 필요한지는 문제마다 달라서 이렇게 하나씩 빼 보며 확인해요(ablation).",
    "타깃 네트워크만 쓴 경우 1만 5천 걸음쯤 크게 무너졌다가 다시 올라왔어요. 방금 겪은 몇 걸음에만 맞춰 배우면, 한 번 배운 것을 쉽게 잊어요(파국적 망각)."
   ]
  },
  "f_dqn": {
   "title": "경험 재생과 타깃 네트워크",
   "images": [
    [
     "img/rl/dqn_ablation.png",
     ""
    ]
   ],
   "notes": [
    "보라(경험 재생만)는 8천 걸음쯤부터 200 근처에 붙어요. 회색(둘 다 없음)과 파랑(타깃만)은 오르내림이 커요.",
    "y축은 ε-greedy로 탐색하면서 버틴 걸음 수라, 탐색을 끄면 조금 더 높아요."
   ],
   "chart": true
  },
  "s3": {
   "title": "③ 같은 정책에서 기울기를 30번 재면 (8판씩 200걸음)",
   "cols": [
    "",
    "흔들림 (분산의 합)",
    "평균 기울기의 크기"
   ],
   "rows": [
    [
     "수익 G 그대로",
     "0.0962",
     "0.685"
    ],
    [
     "G − 평균 (베이스라인)",
     "0.0385",
     "0.721"
    ]
   ],
   "notes": [
    "베이스라인을 빼니 흔들림이 2.5배 줄었고, 평균 방향의 크기는 거의 같아요. 같은 방향을 덜 흔들리게 재는 거예요.",
    "수익이 모두 양수(버틴 걸음마다 +1)라 베이스라인이 없으면 모든 행동의 확률을 올리려 하고, 그중 \"조금 더\" 좋았던 행동만 덜 올라가는 식으로 배워요. 평균을 빼면 평균보다 나은 행동은 올리고, 못한 행동은 내려요."
   ]
  },
  "s5": {
   "title": "④⑤ 정책 경사 네 가지, 씨앗 3개 평균 (8판을 함께 굴림)",
   "cols": [
    "방법",
    "1만 걸음",
    "2만",
    "4만",
    "6만",
    "한 번 고칠 때 쓰는 걸음"
   ],
   "rows": [
    [
     "REINFORCE",
     "30.0",
     "45.9",
     "119.2",
     "183.1",
     "200 × 8판, 1번"
    ],
    [
     "REINFORCE + 베이스라인",
     "26.1",
     "51.5",
     "138.6",
     "181.3",
     "200 × 8판, 1번"
    ],
    [
     "A2C (5걸음)",
     "27.2",
     "154.9",
     "106.8",
     "182.4",
     "5 × 8판, 1번"
    ],
    [
     "PPO",
     "128.2",
     "138.0",
     "191.7",
     "198.4",
     "128 × 8판, 4번 다시"
    ]
   ],
   "notes": [
    "PPO가 가장 빨리, 가장 높이 올라가요. 같은 기록으로 네 번 다시 배워서 한 걸음의 경험을 더 알뜰하게 써요.",
    "REINFORCE는 판이 끝나야 배워서 처음엔 느리지만 6만 걸음에는 180을 넘었어요. 이 문제에서는 평균을 빼는 베이스라인이 학습 곡선을 크게 바꾸지는 않았어요(③에서 흔들림은 줄었는데). 판 길이가 짧을 때부터 모든 판이 비슷하게 짧아서 평균만 빼도 신호가 크지 않기 때문이에요.",
    "A2C는 5걸음마다 바로 고쳐서 빠르지만 오르내림이 커요. 씨앗마다 차이도 커서, 씨앗 3개 평균은 숫자 하나하나보다 흐름으로 읽어요."
   ]
  },
  "t4": {
   "title": "A2C의 이득(advantage)을 몇 걸음 보고 재나 (GAE의 λ)",
   "cols": [
    "λ",
    "1만 걸음",
    "2만",
    "4만",
    "6만"
   ],
   "rows": [
    [
     0.0,
     "18.0",
     "88.4",
     "87.2",
     "58.5"
    ],
    [
     0.95,
     "23.1",
     "119.8",
     "131.0",
     "168.3"
    ],
    [
     "<b>1.0</b> (지금)",
     "27.2",
     "154.9",
     "106.8",
     "182.4"
    ]
   ],
   "notes": [
    "λ = 0은 한 걸음 TD 오차만 써요. 비평가(critic)가 아직 엉터리일 때 그 엉터리 값에 기대서, 6만 걸음 뒤에도 100을 못 넘었어요.",
    "λ = 1은 5걸음 동안 실제로 받은 보상을 쓰고 맨 끝에만 비평가 값을 빌려요. 덜 기대는 만큼 처음엔 흔들려도 결국 잘 배웠어요. 그 사이 값(0.95)은 둘을 섞은 거예요."
   ]
  },
  "f_pg": {
   "title": "정책 경사 방법 비교",
   "images": [
    [
     "img/rl/policy_gradient.png",
     ""
    ]
   ],
   "notes": [
    "보라(PPO)가 1만 걸음 안에 120을 넘고, 3만 걸음 뒤로는 180~200 사이에 머물러요.",
    "DQN(②)은 한 판씩 굴렸고 이쪽은 8판을 함께 굴려서, 같은 걸음 수라도 고치는 횟수와 방식이 달라요. 두 그림을 숫자로 바로 비교하지는 않아요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   1,
   "막대 세우기",
   "실수 상태 4개",
   "무작위면 17걸음"
  ],
  [
   2,
   "DQN",
   "경험 재생 + 타깃",
   "하나씩 빼 보기"
  ],
  [
   3,
   "REINFORCE",
   "정책을 직접",
   "베이스라인"
  ],
  [
   4,
   "A2C",
   "비평가가 돕기",
   "λ로 섞기"
  ],
  [
   5,
   "PPO",
   "조금씩만 바꾸기",
   "가장 빠름"
  ]
 ],
 "env": "Python 3.13.5, PyTorch 2.13.0+cpu, NumPy 2.5.2, CPU 1스레드 (seed 0, 1, 2)"
};
