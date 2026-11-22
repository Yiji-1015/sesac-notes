window.PAGE = {
 "code": [
  {
   "t": "import copy",
   "s": 2
  },
  {
   "t": "import itertools",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 1
  },
  {
   "t": "from torch import nn",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 문제, 채점기, 두 가지 모델",
   "s": 1
  },
  {
   "t": "torch.set_num_threads(1)",
   "s": 1
  },
  {
   "t": "problems = torch.tensor(list(itertools.product(range(10), repeat=3)))  # (a, b, c) 1,000개",
   "s": 1
  },
  {
   "t": "order = torch.randperm(1000, generator=torch.Generator().manual_seed(0))",
   "s": 1
  },
  {
   "t": "train, test = problems[order[:700]], problems[order[700:]]",
   "s": 1
  },
  {
   "t": "answer = lambda p: p.sum(dim=1) % 10  # noqa: E731  채점기: 맞았는지 기계로 확인",
   "s": 1
  },
  {
   "t": "thought = lambda p: (p[:, 0] + p[:, 1]) % 10  # noqa: E731  맞는 중간 생각",
   "s": 1
  },
  {
   "t": "onehot = lambda x: nn.functional.one_hot(x, 10).float()  # noqa: E731",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "class Thinker(nn.Module):  # 먼저 (a + b)를 적고, 그것과 c로 답하기",
   "s": 1
  },
  {
   "t": "    def __init__(self):",
   "s": 1
  },
  {
   "t": "        super().__init__()",
   "s": 1
  },
  {
   "t": "        self.think = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))",
   "s": 1
  },
  {
   "t": "        self.say = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))",
   "s": 1
  },
  {
   "t": "    def first(self, p):",
   "s": 1
  },
  {
   "t": "        return self.think(torch.cat([onehot(p[:, 0]), onehot(p[:, 1])], dim=1))",
   "s": 1
  },
  {
   "t": "    def second(self, mid, p):",
   "s": 1
  },
  {
   "t": "        return self.say(torch.cat([onehot(mid), onehot(p[:, 2])], dim=1))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "class Direct(nn.Module):  # 생각 없이 a, b, c에서 바로 답하기",
   "s": 1
  },
  {
   "t": "    def __init__(self):",
   "s": 1
  },
  {
   "t": "        super().__init__()",
   "s": 1
  },
  {
   "t": "        self.net = nn.Sequential(nn.Linear(30, 128), nn.ReLU(), nn.Linear(128, 10))",
   "s": 1
  },
  {
   "t": "    def forward(self, p):",
   "s": 1
  },
  {
   "t": "        return self.net(torch.cat([onehot(p[:, 0]), onehot(p[:, 1]), onehot(p[:, 2])], dim=1))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def generate(m, p, greedy=False):  # 답 (생각, 답, log 확률)",
   "s": 1
  },
  {
   "t": "    if isinstance(m, Direct):",
   "s": 1
  },
  {
   "t": "        d = torch.distributions.Categorical(logits=m(p))",
   "s": 1
  },
  {
   "t": "        a = d.probs.argmax(1) if greedy else d.sample()",
   "s": 1
  },
  {
   "t": "        return None, a, d.log_prob(a)",
   "s": 1
  },
  {
   "t": "    d1 = torch.distributions.Categorical(logits=m.first(p))",
   "s": 1
  },
  {
   "t": "    mid = d1.probs.argmax(1) if greedy else d1.sample()",
   "s": 1
  },
  {
   "t": "    d2 = torch.distributions.Categorical(logits=m.second(mid, p))",
   "s": 1
  },
  {
   "t": "    a = d2.probs.argmax(1) if greedy else d2.sample()",
   "s": 1
  },
  {
   "t": "    return mid, a, d1.log_prob(mid) + d2.log_prob(a)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def accuracy(m, p):  # 가장 그럴듯한 답 하나로 채점",
   "s": 1
  },
  {
   "t": "    with torch.no_grad():",
   "s": 1
  },
  {
   "t": "        return (generate(m, p, greedy=True)[1] == answer(p)).float().mean().item()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def sft(m, p, epochs):  # 선생님이 쓴 풀이(생각 + 답)를 따라 하기",
   "s": 1
  },
  {
   "t": "    opt = torch.optim.Adam(m.parameters(), lr=1e-2)",
   "s": 1
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 1
  },
  {
   "t": "        if isinstance(m, Direct):",
   "s": 1
  },
  {
   "t": "            loss = nn.functional.cross_entropy(m(p), answer(p))",
   "s": 1
  },
  {
   "t": "        else:",
   "s": 1
  },
  {
   "t": "            loss = nn.functional.cross_entropy(m.first(p), thought(p)) + nn.functional.cross_entropy(m.second(thought(p), p), answer(p))",
   "s": 1
  },
  {
   "t": "        opt.zero_grad()",
   "s": 1
  },
  {
   "t": "        loss.backward()",
   "s": 1
  },
  {
   "t": "        opt.step()",
   "s": 1
  },
  {
   "t": "    return m",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def sft_compare(n, epochs=200):",
   "s": 1
  },
  {
   "t": "    out = {}",
   "s": 1
  },
  {
   "t": "    for name, cls in ((\"생각하고 답하기\", Thinker), (\"바로 답하기\", Direct)):",
   "s": 1
  },
  {
   "t": "        torch.manual_seed(0)",
   "s": 1
  },
  {
   "t": "        m = sft(cls(), train[:n], epochs)",
   "s": 1
  },
  {
   "t": "        out[name] = (accuracy(m, train[:n]), accuracy(m, test))",
   "s": 1
  },
  {
   "t": "    return out",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "sft_table = {n: sft_compare(n) for n in (30, 100, 700)}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 출발점: 풀이 100개로 조금만 배운 모델",
   "s": 2
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 2
  },
  {
   "t": "base = sft(Thinker(), train[:100], epochs=30)",
   "s": 2
  },
  {
   "t": "def pass_at_k(m, p, k):  # k번 뽑아서 한 번이라도 맞으면 맞은 문제",
   "s": 2
  },
  {
   "t": "    with torch.no_grad():",
   "s": 2
  },
  {
   "t": "        rep = p.repeat_interleave(k, dim=0)",
   "s": 2
  },
  {
   "t": "        hit = (generate(m, rep)[1] == answer(rep)).view(-1, k)",
   "s": 2
  },
  {
   "t": "    return hit.any(dim=1)",
   "s": 2
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 2
  },
  {
   "t": "base_pass = {k: pass_at_k(base, test, k).float().mean().item() for k in (1, 8, 64)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 맞았는지만 보고 강화학습: REINFORCE와 GRPO",
   "s": 3
  },
  {
   "t": "def rl(m, steps=2000, group=8, batch=32, lr=3e-3, method=\"GRPO\", step_reward=False, pool=None, seed=0):",
   "s": 3
  },
  {
   "t": "    g = torch.Generator().manual_seed(seed)",
   "s": 3
  },
  {
   "t": "    opt = torch.optim.Adam(m.parameters(), lr=lr)",
   "s": 3
  },
  {
   "t": "    pool = train if pool is None else pool",
   "s": 3
  },
  {
   "t": "    log = {\"test\": [], \"모두 맞음\": 0, \"모두 틀림\": 0, \"그룹\": 0, \"뽑은 답\": 0}",
   "s": 3
  },
  {
   "t": "    for step in range(steps):",
   "s": 3
  },
  {
   "t": "        p = pool[torch.randint(len(pool), (batch,), generator=g)].repeat_interleave(group, dim=0)  # 문제마다 답을 group개",
   "s": 3
  },
  {
   "t": "        mid, a, logp = generate(m, p)",
   "s": 3
  },
  {
   "t": "        reward = (a == answer(p)).float()  # 검증 가능한 보상: 맞으면 1, 틀리면 0",
   "s": 3
  },
  {
   "t": "        if step_reward:",
   "s": 3
  },
  {
   "t": "            reward = 0.5 * (mid == thought(p)).float() + 0.5 * reward  # 생각 단계도 채점",
   "s": 3
  },
  {
   "t": "        r = reward.view(-1, group)",
   "s": 3
  },
  {
   "t": "        if method == \"GRPO\":",
   "s": 3
  },
  {
   "t": "            adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-6)  # 같은 문제의 답끼리 비교",
   "s": 3
  },
  {
   "t": "        else:",
   "s": 3
  },
  {
   "t": "            adv = r  # REINFORCE: 보상을 그대로",
   "s": 3
  },
  {
   "t": "        log[\"모두 맞음\"] += int((r.min(dim=1).values == 1).sum())",
   "s": 3
  },
  {
   "t": "        log[\"모두 틀림\"] += int((r.max(dim=1).values == 0).sum())",
   "s": 3
  },
  {
   "t": "        log[\"그룹\"] += batch",
   "s": 3
  },
  {
   "t": "        log[\"뽑은 답\"] += len(p)",
   "s": 3
  },
  {
   "t": "        loss = -(adv.reshape(-1).detach() * logp).mean()",
   "s": 3
  },
  {
   "t": "        opt.zero_grad()",
   "s": 3
  },
  {
   "t": "        loss.backward()",
   "s": 3
  },
  {
   "t": "        opt.step()",
   "s": 3
  },
  {
   "t": "        if step % 100 == 0 or step == steps - 1:",
   "s": 3
  },
  {
   "t": "            log[\"test\"].append(accuracy(m, test))",
   "s": 3
  },
  {
   "t": "    return m, log",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "runs = {}",
   "s": 3
  },
  {
   "t": "for method in (\"REINFORCE\", \"GRPO\"):",
   "s": 3
  },
  {
   "t": "    runs[method] = rl(copy.deepcopy(base), method=method)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 신호가 없는 그룹: 다 맞거나 다 틀리면 배울 게 없어요",
   "s": 4
  },
  {
   "t": "grpo_model, grpo_log = runs[\"GRPO\"]",
   "s": 4
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 4
  },
  {
   "t": "never = ~pass_at_k(base, test, 64)  # 출발점이 64번 뽑아도 못 맞힌 문제",
   "s": 4
  },
  {
   "t": "solved_after = (generate(grpo_model, test, greedy=True)[1] == answer(test))",
   "s": 4
  },
  {
   "t": "never_then_solved = (never & solved_after).sum().item(), never.sum().item()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 생각 단계도 채점하면 (단계별 보상)",
   "s": 5
  },
  {
   "t": "torch.manual_seed(1)",
   "s": 5
  },
  {
   "t": "blank = Thinker()  # SFT 없이 처음부터",
   "s": 5
  },
  {
   "t": "runs[\"GRPO + 단계별 보상\"] = rl(copy.deepcopy(base), step_reward=True)",
   "s": 5
  },
  {
   "t": "runs[\"SFT 없이 GRPO\"] = rl(copy.deepcopy(blank))",
   "s": 5
  },
  {
   "t": "runs[\"SFT 없이 GRPO + 단계별 보상\"] = rl(copy.deepcopy(blank), step_reward=True)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 어려운 정도로 문제 고르기",
   "s": 6
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    rep = train.repeat_interleave(8, dim=0)",
   "s": 6
  },
  {
   "t": "    rate = (generate(base, rep)[1] == answer(rep)).float().view(-1, 8).mean(dim=1)  # 학습 문제마다 8번 풀어 본 정답률",
   "s": 6
  },
  {
   "t": "useful = train[(rate > 0) & (rate < 1)]  # 가끔 맞히는 문제만 (신호가 나오는 문제)",
   "s": 6
  },
  {
   "t": "filter_cost = len(rep)",
   "s": 6
  },
  {
   "t": "runs[\"가끔 맞히는 문제만\"] = rl(copy.deepcopy(base), pool=useful)",
   "s": 6
  }
 ],
 "names": {
  "1": "문제, 채점기, 두 가지 모델",
  "2": "출발점: 풀이 100개로 조금만 배운 모델",
  "3": "맞았는지만 보고 강화학습: REINFORCE와 GRPO",
  "4": "신호가 없는 그룹: 다 맞거나 다 틀리면 배울 게 없어요",
  "5": "생각 단계도 채점하면 (단계별 보상)",
  "6": "어려운 정도로 문제 고르기"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "s1": {
   "title": "① 풀이를 따라 배운 뒤 (SFT, 200에폭)",
   "cols": [
    "학습한 풀이 수",
    "생각하고 답하기: 배운 문제",
    "처음 보는 문제",
    "바로 답하기: 배운 문제",
    "처음 보는 문제"
   ],
   "rows": [
    [
     30,
     "1.000",
     "0.127",
     "1.000",
     "0.057"
    ],
    [
     100,
     "1.000",
     "0.413",
     "1.000",
     "0.027"
    ],
    [
     700,
     "1.000",
     "1.000",
     "1.000",
     "0.220"
    ]
   ],
   "notes": [
    "두 모델 모두 배운 문제는 다 맞혀요. 차이는 처음 보는 300문제에서 나요. 풀이 700개를 배우면 생각하는 모델은 1.00, 바로 답하는 모델은 0.22예요.",
    "바로 답하는 모델은 (a, b, c) 1,000가지 조합을 하나하나 외워야 해요. 생각하는 모델은 \"a + b\" 100가지와 \"중간값 + c\" 100가지만 배우면 되고, 처음 보는 문제도 아는 두 단계로 나눠 풀어요. 중간 단계를 적는 게 추론 모델이 \"생각\"을 길게 쓰는 이유 중 하나예요."
   ]
  },
  "s2": {
   "title": "② 출발점: 풀이 100개를 30에폭만 배운 모델",
   "lines": [
    [
     "가장 그럴듯한 답으로 채점: 학습 문제 0.189, 처음 보는 문제 0.160",
     1
    ],
    [
     "답을 뽑아서 채점 (처음 보는 문제): 1번 0.117, 8번 중 하나라도 0.463, 64번 중 하나라도 0.867",
     2,
     "newf"
    ]
   ],
   "notes": [
    "한 번에 맞히는 비율은 낮지만, 여러 번 뽑으면 87%의 문제에서 한 번은 맞는 답이 나와요(pass@k). 강화학습은 이렇게 가끔 나오는 맞는 답을 더 자주 나오게 만드는 일이에요. 한 번도 안 나오는 답은 강화할 수 없어요."
   ]
  },
  "s3": {
   "title": "③ 맞았는지만 보고 2,000걸음 (문제 32개 × 답 8개씩)",
   "cols": [
    "방법",
    "처음 보는 문제 정확도"
   ],
   "rows": [
    [
     "출발점 (SFT)",
     "0.160"
    ],
    [
     "REINFORCE (보상 그대로)",
     "0.280"
    ],
    [
     "GRPO (같은 문제의 답끼리 비교)",
     "0.353"
    ]
   ],
   "notes": [
    "둘 다 올라가지만 GRPO가 더 높아요(0.28 → 0.35). REINFORCE는 맞은 답을 모두 같은 크기로 밀어 줘요. GRPO는 같은 문제에 대한 답 8개의 평균을 빼고 표준편차로 나눠서, 그 문제에서 평균보다 나은 답만 밀고 못한 답은 눌러요. 쉬운 문제의 흔한 정답보다 어려운 문제의 드문 정답에 더 큰 무게가 가요.",
    "그런데 둘 다 0.35 근처에서 멈춰요. 다음 단계에서 이유를 봐요."
   ]
  },
  "f_curve": {
   "title": "강화학습 걸음마다 처음 보는 문제 정확도",
   "images": [
    [
     "img/rl/reasoning_curves.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽: SFT 모델에서 출발한 네 가지. 모두 몇백 걸음 만에 오르고 멈춰요.",
    "오른쪽: 아무것도 배우지 않은 모델에서 출발하면 답만 채점할 때(분홍)는 오히려 떨어지고(0.02), 생각 단계도 채점하면(파랑) 0.62까지 올라요."
   ],
   "chart": true
  },
  "t3": {
   "title": "문제마다 뽑는 답의 수(그룹 크기)를 바꾸면",
   "cols": [
    "그룹 크기",
    "처음 보는 문제 정확도"
   ],
   "rows": [
    [
     4,
     "0.267"
    ],
    [
     "<b>8</b> (지금)",
     "0.353"
    ],
    [
     16,
     "0.400"
    ]
   ],
   "notes": [
    "그룹이 크면 \"가끔 맞는\" 답이 그룹 안에 하나라도 들어올 확률이 커서 신호가 더 자주 나와요. 대신 답을 뽑는 계산이 그만큼 늘어요. 추론 모델에서는 답 하나가 수천 토큰이라 이게 가장 큰 비용이에요."
   ]
  },
  "s4": {
   "title": "④ GRPO 2,000걸음 동안의 그룹 64,000개",
   "lines": [
    [
     "8개 모두 맞음: 16,745개 (26%)",
     1
    ],
    [
     "8개 모두 틀림: 32,386개 (51%)",
     2,
     "newf"
    ],
    [
     "출발점이 64번 뽑아도 못 맞힌 처음 보는 문제 37개 중, 강화학습 뒤 맞힌 것: 1개",
     3
    ]
   ],
   "notes": [
    "GRPO는 같은 문제의 답끼리 비교하니까, 8개가 모두 맞거나 모두 틀리면 평균을 뺀 값이 다 0이 돼서 배울 게 없어요. 그런 그룹이 77%예요. 뽑은 답의 대부분을 버린 셈이에요.",
    "\"모두 틀림\"이 특히 문제예요. 정답이 한 번도 안 나오는 문제는 아무리 많이 풀려도 배우지 못해요(보상이 드묾). 그래서 강화학습은 출발 모델이 가끔이라도 맞히는 범위를 넓히기보다 그 안에서 더 자주 맞히게 만드는 쪽이에요."
   ]
  },
  "f_sig": {
   "title": "그룹마다 신호가 있었나",
   "images": [
    [
     "img/rl/reasoning_signal.png",
     ""
    ]
   ],
   "notes": [
    "보라가 신호를 준 그룹이에요. 아무것도 안 배운 모델에서 답만 채점하면(아래서 둘째) 보라가 50%나 되는데도 정확도는 떨어져요. 찍어도 10%는 맞으니 맞은 답의 대부분이 운이고, 그때의 틀린 생각까지 같이 밀어 주기 때문이에요. 신호가 있다고 다 쓸모 있는 신호는 아니에요.",
    "생각 단계도 채점하면(맨 아래) 분홍이 크게 줄고, 보라가 진짜로 맞은 생각을 밀어 줘요."
   ],
   "chart": true
  },
  "s5": {
   "title": "⑤ 답만 채점 vs 생각 단계도 채점",
   "cols": [
    "출발점",
    "답만 채점",
    "생각 단계도 채점",
    "출발점 정확도"
   ],
   "rows": [
    [
     "SFT 모델",
     "0.353",
     "0.357",
     "0.160"
    ],
    [
     "아무것도 안 배운 모델",
     "0.023",
     "0.623",
     "약 0.1"
    ]
   ],
   "notes": [
    "아무것도 안 배운 모델은 답이 맞을 확률이 10%뿐이고, 맞아도 생각이 맞아서인지 운인지 몰라요. 답만 채점하면 0.02로 오히려 떨어져요. 생각 단계에도 점수를 주면 어느 단계가 틀렸는지 알려 줘서 0.62까지 올라요.",
    "SFT 모델에서 출발하면 단계별 보상의 차이가 거의 없어요(0.353 vs 0.357). 생각 단계는 SFT에서 이미 꽤 배워서, 남은 문제는 \"한 번도 안 나오는 답\"이기 때문이에요. 단계별 보상은 신호가 가장 모자랄 때 가장 힘이 커요.",
    "실제 추론 모델에서는 생각 단계마다 맞았는지 알기 어려워요. 그래서 단계를 채점하는 모델(과정 보상 모델)을 따로 학습하거나, 정답이 확인되는 수학과 코드부터 시작해요."
   ]
  },
  "s6": {
   "title": "⑥ 출발점이 가끔 맞히는 문제만 골라 쓰면",
   "cols": [
    "",
    "전체 학습 문제",
    "가끔 맞히는 문제만"
   ],
   "rows": [
    [
     "문제 수",
     700,
     345
    ],
    [
     "고르는 데 뽑은 답",
     "0",
     "5,600"
    ],
    [
     "처음 보는 문제 정확도",
     "0.353",
     "0.253"
    ],
    [
     "모두 맞은 그룹 비율",
     "26%",
     "46%"
    ]
   ],
   "notes": [
    "출발점으로 학습 문제마다 8번씩 풀어 보고, 0번도 8번도 아닌 345문제만 남겼어요. 신호가 나올 문제에만 계산을 쓰자는 생각이에요(강연에서는 이렇게 고른 데이터로 뽑는 답 수를 8.55배 줄였다고 해요).",
    "여기서는 오히려 낮아졌어요(0.353 → 0.253). 고른 문제는 몇백 걸음 만에 다 맞히게 돼서 \"모두 맞음\" 그룹이 46%로 늘었고, 처음에 다 틀리던 문제는 아예 빠져서 다시 볼 기회가 없었어요. 어려운 정도는 학습하면서 계속 바뀌어서, 한 번 고른 목록은 금방 낡아요.",
    "그래서 실제로는 학습 도중에 배치마다 \"다 맞거나 다 틀린 그룹\"을 빼고 다시 뽑거나(동적 샘플링), 쉬운 것부터 어려운 것으로 단계를 나눠 가며(커리큘럼) 다시 골라요."
   ]
  }
 },
 "flow": [
  [
   1,
   "생각하고 답하기",
   "중간 단계 적기",
   "처음 보는 문제"
  ],
  [
   2,
   "출발점",
   "pass@k",
   "가끔 나오는 정답"
  ],
  [
   3,
   "검증 가능한 보상",
   "REINFORCE, GRPO",
   "그룹 안에서 비교"
  ],
  [
   4,
   "신호 없는 그룹",
   "다 맞음, 다 틀림",
   "드문 보상"
  ],
  [
   5,
   "단계별 보상",
   "생각도 채점",
   "처음부터일 때"
  ],
  [
   6,
   "문제 고르기",
   "어려운 정도",
   "목록이 낡음"
  ]
 ],
 "env": "Python 3.13.5, PyTorch 2.13.0+cpu (CPU 1스레드, seed 0)"
};
