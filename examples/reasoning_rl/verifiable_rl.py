# 정답을 기계로 확인할 수 있는 문제에서 "생각 단계"를 가르치고(SFT), 맞았는지만 알려 주는 보상으로 강화학습(GRPO)을 하는 전체 코드.
# 추론 모델(LRM) 학습의 축소판이에요. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 문제: 0~9 숫자 셋 a, b, c를 주면 (a + b + c)의 일의 자리를 답하기. "생각"은 (a + b)의 일의 자리를 먼저 적는 것이에요.
# 데이터: 코드가 만들어요(1,000문제 중 700개로 학습, 300개는 처음 보는 문제). CPU로 1분쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import copy                                                                     #@2
import itertools                                                                #@1
import torch                                                                    #@1
from torch import nn                                                            #@1

# ① 문제, 채점기, 두 가지 모델
torch.set_num_threads(1)
problems = torch.tensor(list(itertools.product(range(10), repeat=3)))   # (a, b, c) 1,000개
order = torch.randperm(1000, generator=torch.Generator().manual_seed(0))
train, test = problems[order[:700]], problems[order[700:]]
answer = lambda p: p.sum(dim=1) % 10                             # noqa: E731  채점기: 맞았는지 기계로 확인
thought = lambda p: (p[:, 0] + p[:, 1]) % 10                     # noqa: E731  맞는 중간 생각
onehot = lambda x: nn.functional.one_hot(x, 10).float()          # noqa: E731

class Thinker(nn.Module):                                        # 먼저 (a + b)를 적고, 그것과 c로 답하기
    def __init__(self):
        super().__init__()
        self.think = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
        self.say = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
    def first(self, p):
        return self.think(torch.cat([onehot(p[:, 0]), onehot(p[:, 1])], dim=1))
    def second(self, mid, p):
        return self.say(torch.cat([onehot(mid), onehot(p[:, 2])], dim=1))

class Direct(nn.Module):                                         # 생각 없이 a, b, c에서 바로 답하기
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(30, 128), nn.ReLU(), nn.Linear(128, 10))
    def forward(self, p):
        return self.net(torch.cat([onehot(p[:, 0]), onehot(p[:, 1]), onehot(p[:, 2])], dim=1))

def generate(m, p, greedy=False):                                # 답 (생각, 답, log 확률)
    if isinstance(m, Direct):
        d = torch.distributions.Categorical(logits=m(p))
        a = d.probs.argmax(1) if greedy else d.sample()
        return None, a, d.log_prob(a)
    d1 = torch.distributions.Categorical(logits=m.first(p))
    mid = d1.probs.argmax(1) if greedy else d1.sample()
    d2 = torch.distributions.Categorical(logits=m.second(mid, p))
    a = d2.probs.argmax(1) if greedy else d2.sample()
    return mid, a, d1.log_prob(mid) + d2.log_prob(a)

def accuracy(m, p):                                              # 가장 그럴듯한 답 하나로 채점
    with torch.no_grad():
        return (generate(m, p, greedy=True)[1] == answer(p)).float().mean().item()

def sft(m, p, epochs):                                           # 선생님이 쓴 풀이(생각 + 답)를 따라 하기
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(epochs):
        if isinstance(m, Direct):
            loss = nn.functional.cross_entropy(m(p), answer(p))
        else:
            loss = nn.functional.cross_entropy(m.first(p), thought(p)) + nn.functional.cross_entropy(m.second(thought(p), p), answer(p))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m

def sft_compare(n, epochs=200):
    out = {}
    for name, cls in (("생각하고 답하기", Thinker), ("바로 답하기", Direct)):
        torch.manual_seed(0)
        m = sft(cls(), train[:n], epochs)
        out[name] = (accuracy(m, train[:n]), accuracy(m, test))
    return out

sft_table = {n: sft_compare(n) for n in (30, 100, 700)}

# ② 출발점: 풀이 100개로 조금만 배운 모델
torch.manual_seed(0)
base = sft(Thinker(), train[:100], epochs=30)
def pass_at_k(m, p, k):                                          # k번 뽑아서 한 번이라도 맞으면 맞은 문제
    with torch.no_grad():
        rep = p.repeat_interleave(k, dim=0)
        hit = (generate(m, rep)[1] == answer(rep)).view(-1, k)
    return hit.any(dim=1)
torch.manual_seed(0)
base_pass = {k: pass_at_k(base, test, k).float().mean().item() for k in (1, 8, 64)}

# ③ 맞았는지만 보고 강화학습: REINFORCE와 GRPO
def rl(m, steps=2000, group=8, batch=32, lr=3e-3, method="GRPO", step_reward=False, pool=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    pool = train if pool is None else pool
    log = {"test": [], "모두 맞음": 0, "모두 틀림": 0, "그룹": 0, "뽑은 답": 0}
    for step in range(steps):
        p = pool[torch.randint(len(pool), (batch,), generator=g)].repeat_interleave(group, dim=0)   # 문제마다 답을 group개
        mid, a, logp = generate(m, p)
        reward = (a == answer(p)).float()                        # 검증 가능한 보상: 맞으면 1, 틀리면 0
        if step_reward:
            reward = 0.5 * (mid == thought(p)).float() + 0.5 * reward   # 생각 단계도 채점
        r = reward.view(-1, group)
        if method == "GRPO":
            adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-6)   # 같은 문제의 답끼리 비교
        else:
            adv = r                                              # REINFORCE: 보상을 그대로
        log["모두 맞음"] += int((r.min(dim=1).values == 1).sum())
        log["모두 틀림"] += int((r.max(dim=1).values == 0).sum())
        log["그룹"] += batch
        log["뽑은 답"] += len(p)
        loss = -(adv.reshape(-1).detach() * logp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            log["test"].append(accuracy(m, test))
    return m, log

runs = {}
for method in ("REINFORCE", "GRPO"):
    runs[method] = rl(copy.deepcopy(base), method=method)

# ④ 신호가 없는 그룹: 다 맞거나 다 틀리면 배울 게 없어요
grpo_model, grpo_log = runs["GRPO"]
torch.manual_seed(0)
never = ~pass_at_k(base, test, 64)                               # 출발점이 64번 뽑아도 못 맞힌 문제
solved_after = (generate(grpo_model, test, greedy=True)[1] == answer(test))
never_then_solved = (never & solved_after).sum().item(), never.sum().item()

# ⑤ 생각 단계도 채점하면 (단계별 보상)
torch.manual_seed(1)
blank = Thinker()                                                # SFT 없이 처음부터
runs["GRPO + 단계별 보상"] = rl(copy.deepcopy(base), step_reward=True)
runs["SFT 없이 GRPO"] = rl(copy.deepcopy(blank))
runs["SFT 없이 GRPO + 단계별 보상"] = rl(copy.deepcopy(blank), step_reward=True)

# ⑥ 어려운 정도로 문제 고르기
torch.manual_seed(0)
with torch.no_grad():
    rep = train.repeat_interleave(8, dim=0)
    rate = (generate(base, rep)[1] == answer(rep)).float().view(-1, 8).mean(dim=1)   # 학습 문제마다 8번 풀어 본 정답률
useful = train[(rate > 0) & (rate < 1)]                          # 가끔 맞히는 문제만 (신호가 나오는 문제)
filter_cost = len(rep)
runs["가끔 맞히는 문제만"] = rl(copy.deepcopy(base), pool=useful)
