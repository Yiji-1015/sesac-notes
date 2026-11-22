# 수레 위 막대 세우기로 심층 강화학습(DQN, REINFORCE, A2C, PPO)을 직접 해 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 환경: 고전 제어 문제를 물리식으로 직접 짰어요. 수레를 왼쪽이나 오른쪽으로 밀어 막대가 넘어지지 않게 해요. 버틴 걸음마다 +1, 200걸음이면 끝.
# CPU로 5분쯤 걸려요(DQN 열두 번, 정책 경사 열두 번).
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import math                                                                     #@1
import numpy as np                                                              #@1
import torch                                                                    #@2
from torch import nn                                                            #@2

# ① 막대 세우기 환경
class Pole:                                                     # n판을 한꺼번에 굴려요
    def __init__(self, n, seed):
        self.n, self.rng = n, np.random.default_rng(seed)
        self.s = self.rng.uniform(-0.05, 0.05, (n, 4))          # 수레 위치, 수레 속도, 막대 각도, 막대 각속도
        self.t = np.zeros(n, dtype=int)

    def step(self, a):                                          # a: 0 = 왼쪽으로 밀기, 1 = 오른쪽으로 밀기
        x, v, th, w = self.s.T
        force = np.where(a == 1, 10.0, -10.0)
        cos, sin = np.cos(th), np.sin(th)
        tmp = (force + 0.05 * w ** 2 * sin) / 1.1               # 수레 1kg, 막대 0.1kg, 막대 반 길이 0.5m
        w_acc = (9.8 * sin - cos * tmp) / (0.5 * (4 / 3 - 0.1 * cos ** 2 / 1.1))
        x_acc = tmp - 0.05 * w_acc * cos / 1.1
        x, v, th, w = x + 0.02 * v, v + 0.02 * x_acc, th + 0.02 * w, w + 0.02 * w_acc   # 0.02초씩
        self.s, self.t = np.stack([x, v, th, w], axis=1), self.t + 1
        fell = (np.abs(x) > 2.4) | (np.abs(th) > 12 * math.pi / 180)   # 벗어나거나 12도 넘게 기울면 실패
        done = fell | (self.t >= 200)                           # 200걸음 버티면 성공으로 끝
        s_next = self.s.copy()
        k = np.flatnonzero(done)
        self.s[k], self.t[k] = self.rng.uniform(-0.05, 0.05, (len(k), 4)), 0   # 끝난 판은 새로 시작
        return s_next, np.ones(self.n), fell, done

env = Pole(1000, seed=0)
lengths, alive = [], np.zeros(1000, dtype=int)
while len(lengths) < 1000:                                      # 무작위로 밀면 몇 걸음 버티나
    _, _, _, done = env.step(env.rng.integers(2, size=1000))
    alive += 1
    lengths += alive[done].tolist()
    alive[done] = 0

# ② Q를 신경망으로: 경험 재생과 타깃 네트워크 (DQN)
torch.set_num_threads(1)
def mlp(n_out):                                                 # 상태 4개 → 은닉 64 → 64 → 출력
    return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_out))

GAMMA = 0.99
def dqn(seed, steps=20000, replay=True, target=True, batch=64, sync=500):
    torch.manual_seed(seed)
    rng, env = np.random.default_rng(seed), Pole(1, seed)
    q, q_target = mlp(2), mlp(2)
    q_target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf, n = np.zeros((50_000, 11), dtype=np.float32), 0        # 재생 버퍼: (s 4칸, a, r, s' 4칸, 실패 여부)
    s, length, curve = env.s[0].copy(), 0, []
    for t in range(steps):
        eps = max(0.05, 1 - t / 5000)                           # 처음엔 거의 무작위, 5,000걸음에 걸쳐 0.05까지
        a = int(rng.integers(2)) if rng.random() < eps else int(q(torch.tensor(s, dtype=torch.float32)).argmax())
        s2, r, fell, done = env.step(np.array([a]))
        buf[n % len(buf)] = np.r_[s, a, r[0], s2[0], fell[0]]
        n, length = n + 1, length + 1
        if done[0]:
            curve.append((t + 1, length))
            s, length = env.s[0].copy(), 0
        else:
            s = s2[0]
        if n >= batch:
            rows = rng.integers(min(n, len(buf)), size=batch) if replay else np.array([(n - 1) % len(buf)])   # 재생 없으면 방금 것 하나만
            b = torch.tensor(buf[rows])
            with torch.no_grad():
                nxt = (q_target if target else q)(b[:, 6:10]).max(dim=1).values
                y = b[:, 5] + GAMMA * nxt * (1 - b[:, 10])      # 넘어진 뒤의 가치는 0
            pred = q(b[:, :4]).gather(1, b[:, 4:5].long()).squeeze(1)
            loss = ((y - pred) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if target and t % sync == 0:
            q_target.load_state_dict(q.state_dict())            # 타깃은 가끔만 따라오게
    return np.array(curve)

ablation = {(rp, tg): [dqn(sd, replay=rp, target=tg) for sd in range(3)] for rp in (False, True) for tg in (False, True)}

# ③ 정책을 직접 배우기: REINFORCE와 베이스라인
def rollout(policy, env, T):                                    # T걸음 동안 n판을 굴려 기록 모으기
    S, A, R, D, LP = [], [], [], [], []
    for _ in range(T):
        s = torch.tensor(env.s, dtype=torch.float32)
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=policy(s))
            a = dist.sample()
        _, r, _, done = env.step(a.numpy())
        S.append(s), A.append(a), R.append(r), D.append(done), LP.append(dist.log_prob(a))
    f = lambda x: torch.tensor(np.array(x), dtype=torch.float32)  # noqa: E731
    return torch.stack(S), torch.stack(A), f(R), f(D), torch.stack(LP)

def returns_to_go(R, D, last=None):                             # 뒤에서부터 G_t = r_t + γ G_{t+1}, 판이 끝나면 끊기
    G, run = torch.zeros_like(R), torch.zeros(R.shape[1]) if last is None else last
    for t in reversed(range(len(R))):
        run = R[t] + GAMMA * run * (1 - D[t])
        G[t] = run
    return G

def grad_spread(baseline, batches=30, seed=0):                  # 같은 정책에서 기울기를 30번 재서 얼마나 흔들리나
    torch.manual_seed(seed)
    policy, env, grads = mlp(2), Pole(8, seed), []
    for _ in range(batches):
        S, A, R, D, _ = rollout(policy, env, 200)
        G = returns_to_go(R, D)
        w = G - G.mean() if baseline else G                     # 베이스라인: 평균 수익을 빼기
        loss = -(torch.distributions.Categorical(logits=policy(S)).log_prob(A) * w).mean()
        policy.zero_grad()
        loss.backward()
        grads.append(torch.cat([p.grad.flatten() for p in policy.parameters()]).clone())
    g = torch.stack(grads)
    return g.var(dim=0).sum().item(), g.mean(dim=0).norm().item()   # 흔들림(분산의 합), 평균 방향의 크기

spread = {b: grad_spread(b) for b in (False, True)}

# ④ 몇 걸음마다 배우기: 액터-크리틱 (A2C)
A2C_LAMBDA = 1.0                                                # 1이면 5걸음 동안 실제로 받은 보상 + 5걸음 뒤의 가치, 0이면 한 걸음 TD
def advantages(R, D, V, V_last, lam):                           # GAE: TD 오차를 λ로 섞어 더하기 (λ = 0이면 한 걸음 TD)
    adv, gae = torch.zeros_like(R), torch.zeros(R.shape[1])
    for t in reversed(range(len(R))):
        nxt = V_last if t == len(R) - 1 else V[t + 1]
        delta = R[t] + GAMMA * nxt * (1 - D[t]) - V[t]          # TD 오차
        gae = delta + GAMMA * lam * (1 - D[t]) * gae
        adv[t] = gae
    return adv

# ⑤ 너무 크게 바뀌지 않게: PPO
def train_pg(kind, seed, steps=60000, n_env=8, lr=None):
    torch.manual_seed(seed)
    env, policy, critic = Pole(n_env, seed), mlp(2), mlp(1)
    T = {"reinforce": 200, "reinforce_b": 200, "a2c": 5, "ppo": 128}[kind]
    lr = lr or {"reinforce": 3e-3, "reinforce_b": 3e-3, "a2c": 1e-3, "ppo": 1e-3}[kind]
    opt = torch.optim.Adam(list(policy.parameters()) + list(critic.parameters()), lr=lr)
    total, lengths, alive, curve = 0, [], np.zeros(n_env), []
    while total < steps:
        S, A, R, D, LP_old = rollout(policy, env, T)
        for t in range(T):                                      # 판 길이 기록
            alive += 1
            lengths += alive[D[t].numpy() > 0].tolist()
            alive[D[t].numpy() > 0] = 0
        total += T * n_env
        if kind in ("reinforce", "reinforce_b"):
            G = returns_to_go(R, D)
            w = G - G.mean() if kind == "reinforce_b" else G
            loss = -(torch.distributions.Categorical(logits=policy(S)).log_prob(A) * w).mean()
        else:
            with torch.no_grad():
                V, V_last = critic(S).squeeze(-1), critic(torch.tensor(env.s, dtype=torch.float32)).squeeze(-1)
            adv = advantages(R, D, V, V_last, lam=A2C_LAMBDA if kind == "a2c" else 0.95)
            target_v = adv + V
        if kind == "a2c":
            loss = -(torch.distributions.Categorical(logits=policy(S)).log_prob(A) * adv).mean() + 0.5 * ((target_v - critic(S).squeeze(-1)) ** 2).mean()
        if kind == "ppo":
            S, A, LP_old, adv, target_v = S.reshape(-1, 4), A.reshape(-1), LP_old.reshape(-1), adv.reshape(-1), target_v.reshape(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            for _ in range(4):                                  # 같은 기록으로 4번 다시 배우기
                for mb in torch.randperm(len(A)).split(256):
                    ratio = torch.exp(torch.distributions.Categorical(logits=policy(S[mb])).log_prob(A[mb]) - LP_old[mb])   # 새 정책 ÷ 옛 정책
                    clipped = torch.min(ratio * adv[mb], ratio.clamp(0.8, 1.2) * adv[mb])   # 20% 넘게 바뀐 몫은 안 쳐 주기
                    loss = -clipped.mean() + 0.5 * ((target_v[mb] - critic(S[mb]).squeeze(-1)) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()
        curve.append((total, np.mean(lengths[-20:]) if lengths else 0.0))
    return np.array(curve)

pg_runs = {k: [train_pg(k, sd) for sd in range(3)] for k in ("reinforce", "reinforce_b", "a2c", "ppo")}
