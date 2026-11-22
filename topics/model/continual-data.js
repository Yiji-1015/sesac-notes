window.PAGE = {
 "code": [
  {
   "t": "import copy",
   "s": 2
  },
  {
   "t": "import numpy as np",
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
   "t": "from sklearn.datasets import load_digits",
   "s": 1
  },
  {
   "t": "from sklearn.model_selection import train_test_split",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 숫자와 두 번째 일",
   "s": 1
  },
  {
   "t": "torch.set_num_threads(1)",
   "s": 1
  },
  {
   "t": "digits = load_digits()",
   "s": 1
  },
  {
   "t": "X, y = (digits.data / 16).astype(np.float32), digits.target",
   "s": 1
  },
  {
   "t": "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)",
   "s": 1
  },
  {
   "t": "shuffle = np.random.default_rng(0).permutation(64)  # 두 번째 일: 64픽셀의 자리를 섞기 (답은 같은 숫자)",
   "s": 1
  },
  {
   "t": "T = lambda a, b: (torch.tensor(a), torch.tensor(b))  # noqa: E731",
   "s": 1
  },
  {
   "t": "A, A_test = T(X_tr, y_tr), T(X_te, y_te)  # 일 A: 원래 숫자",
   "s": 1
  },
  {
   "t": "B, B_test = T(X_tr[:, shuffle], y_tr), T(X_te[:, shuffle], y_te)  # 일 B: 픽셀을 섞은 숫자",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def model():",
   "s": 1
  },
  {
   "t": "    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def acc(m, data):",
   "s": 1
  },
  {
   "t": "    with torch.no_grad():",
   "s": 1
  },
  {
   "t": "        return (m(data[0]).argmax(dim=1) == data[1]).float().mean().item()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def train(m, data, epochs=30, lr=1e-2, penalty=None, replay=None, watch=None, seed=0):",
   "s": 1
  },
  {
   "t": "    g = torch.Generator().manual_seed(seed)",
   "s": 1
  },
  {
   "t": "    opt = torch.optim.Adam(m.parameters(), lr=lr)",
   "s": 1
  },
  {
   "t": "    curve = []",
   "s": 1
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 1
  },
  {
   "t": "        for b in torch.randperm(len(data[1]), generator=g).split(64):",
   "s": 1
  },
  {
   "t": "            x, t = data[0][b], data[1][b]",
   "s": 1
  },
  {
   "t": "            if replay is not None:  # 예전 일의 기록 몇 장을 섞어 함께 배우기",
   "s": 1
  },
  {
   "t": "                r = torch.randint(len(replay[1]), (len(b),), generator=g)",
   "s": 1
  },
  {
   "t": "                x, t = torch.cat([x, replay[0][r]]), torch.cat([t, replay[1][r]])",
   "s": 1
  },
  {
   "t": "            loss = nn.functional.cross_entropy(m(x), t)",
   "s": 1
  },
  {
   "t": "            if penalty is not None:",
   "s": 1
  },
  {
   "t": "                loss = loss + penalty(m)",
   "s": 1
  },
  {
   "t": "            opt.zero_grad()",
   "s": 1
  },
  {
   "t": "            loss.backward()",
   "s": 1
  },
  {
   "t": "            opt.step()",
   "s": 1
  },
  {
   "t": "        if watch is not None:",
   "s": 1
  },
  {
   "t": "            curve.append(tuple(acc(m, w) for w in watch))  # 에폭마다 일 A, 일 B 정확도",
   "s": 1
  },
  {
   "t": "    return m, curve",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 1
  },
  {
   "t": "base, _ = train(model(), A)  # 일 A만 배운 모델",
   "s": 1
  },
  {
   "t": "start = (acc(base, A_test), acc(base, B_test))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 그냥 이어서 배우면: 파국적 망각",
   "s": 2
  },
  {
   "t": "naive, naive_curve = train(copy.deepcopy(base), B, watch=(A_test, B_test))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 예전 것을 조금 다시 보기 (리플레이)",
   "s": 3
  },
  {
   "t": "def replay_run(k, seed=1):",
   "s": 3
  },
  {
   "t": "    keep = torch.randperm(len(A[1]), generator=torch.Generator().manual_seed(seed))[:k]  # 일 A에서 k장만 남겨 두기",
   "s": 3
  },
  {
   "t": "    return train(copy.deepcopy(base), B, replay=(A[0][keep], A[1][keep]), watch=(A_test, B_test))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "replay100, replay_curve = replay_run(100)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 중요한 가중치는 덜 움직이게: EWC",
   "s": 4
  },
  {
   "t": "def fisher_diag(m, data):  # 가중치마다 \"이 일에 얼마나 중요한가\" (Fisher 정보의 대각선)",
   "s": 4
  },
  {
   "t": "    F = [torch.zeros_like(p) for p in m.parameters()]",
   "s": 4
  },
  {
   "t": "    for i in range(len(data[1])):",
   "s": 4
  },
  {
   "t": "        m.zero_grad()",
   "s": 4
  },
  {
   "t": "        logp = nn.functional.log_softmax(m(data[0][i:i + 1]), dim=1)",
   "s": 4
  },
  {
   "t": "        k = torch.multinomial(logp.exp(), 1)[0]  # 모델이 내는 분포에서 답 하나를 뽑아",
   "s": 4
  },
  {
   "t": "        logp[0, k].sum().backward()  # 그 log 확률의 기울기 제곱",
   "s": 4
  },
  {
   "t": "        for f, p in zip(F, m.parameters()):",
   "s": 4
  },
  {
   "t": "            f += p.grad.detach() ** 2",
   "s": 4
  },
  {
   "t": "    return [f / len(data[1]) for f in F]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "F_A = fisher_diag(base, A)",
   "s": 4
  },
  {
   "t": "theta_A = [p.detach().clone() for p in base.parameters()]",
   "s": 4
  },
  {
   "t": "def ewc(lam):",
   "s": 4
  },
  {
   "t": "    return lambda m: lam / 2 * sum((f * (p - p0) ** 2).sum() for f, p, p0 in zip(F_A, m.parameters(), theta_A))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "ewc1000, ewc_curve = train(copy.deepcopy(base), B, penalty=ewc(1000), watch=(A_test, B_test))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 같은 비용으로 비교: 처음부터 둘 다",
   "s": 5
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "joint, _ = train(model(), (torch.cat([A[0], B[0]]), torch.cat([A[1], B[1]])))  # 두 일을 처음부터 함께 (상한)",
   "s": 5
  },
  {
   "t": "board = {name: (acc(m, A_test), acc(m, B_test)) for name, m in",
   "s": 5
  },
  {
   "t": "         ((\"일 A만\", base), (\"그냥 이어서\", naive), (\"리플레이 100장\", replay100), (\"EWC λ = 1000\", ewc1000), (\"처음부터 둘 다\", joint))}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 숫자 3을 지워 주세요: 처음부터 다시, 남은 것만 더",
   "s": 6
  },
  {
   "t": "FORGET = 3",
   "s": 6
  },
  {
   "t": "keep_tr, keep_te = y_tr != FORGET, y_te != FORGET",
   "s": 6
  },
  {
   "t": "retain, forget = (A[0][keep_tr], A[1][keep_tr]), (A[0][~keep_tr], A[1][~keep_tr])  # 남길 데이터, 지울 데이터",
   "s": 6
  },
  {
   "t": "retain_test, forget_test = (A_test[0][keep_te], A_test[1][keep_te]), (A_test[0][~keep_te], A_test[1][~keep_te])",
   "s": 6
  },
  {
   "t": "def check(m):  # 지울 숫자를 얼마나 맞히나, 나머지는 그대로인가",
   "s": 6
  },
  {
   "t": "    return {\"지울 숫자 (test)\": acc(m, forget_test), \"나머지 (test)\": acc(m, retain_test), \"지울 숫자 (학습에 쓴 것)\": acc(m, forget)}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "unlearn = {\"원래 모델\": check(base)}",
   "s": 6
  },
  {
   "t": "retrained, _ = train(model(), retain)  # 정답: 처음부터 지울 데이터 없이",
   "s": 6
  },
  {
   "t": "unlearn[\"처음부터 다시\"] = check(retrained)",
   "s": 6
  },
  {
   "t": "more_retain, _ = train(copy.deepcopy(base), retain, epochs=3)  # 남은 데이터로 조금 더",
   "s": 6
  },
  {
   "t": "unlearn[\"남은 것으로 3에폭 더\"] = check(more_retain)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 빠르게 잊게 하기: 경사 상승과 작업 벡터 빼기",
   "s": 7
  },
  {
   "t": "def gradient_ascent(m, steps=30, lr=1e-3, seed=0):  # 지울 데이터의 손실은 키우고, 남길 데이터의 손실은 줄이기",
   "s": 7
  },
  {
   "t": "    g = torch.Generator().manual_seed(seed)",
   "s": 7
  },
  {
   "t": "    opt = torch.optim.Adam(m.parameters(), lr=lr)",
   "s": 7
  },
  {
   "t": "    for _ in range(steps):",
   "s": 7
  },
  {
   "t": "        f = torch.randint(len(forget[1]), (32,), generator=g)",
   "s": 7
  },
  {
   "t": "        r = torch.randint(len(retain[1]), (64,), generator=g)",
   "s": 7
  },
  {
   "t": "        loss = -nn.functional.cross_entropy(m(forget[0][f]), forget[1][f]) + nn.functional.cross_entropy(m(retain[0][r]), retain[1][r])",
   "s": 7
  },
  {
   "t": "        opt.zero_grad()",
   "s": 7
  },
  {
   "t": "        loss.backward()",
   "s": 7
  },
  {
   "t": "        opt.step()",
   "s": 7
  },
  {
   "t": "    return m",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "unlearn[\"경사 상승 30걸음\"] = check(gradient_ascent(copy.deepcopy(base)))",
   "s": 7
  },
  {
   "t": "tuned, _ = train(copy.deepcopy(base), forget, epochs=10, lr=1e-3)  # 지울 데이터만 더 배운 모델",
   "s": 7
  },
  {
   "t": "task_vector = [p1.detach() - p0.detach() for p0, p1 in zip(base.parameters(), tuned.parameters())]  # 그 차이 = \"숫자 3 쪽으로 가는 방향\"",
   "s": 7
  },
  {
   "t": "def negate(alpha):",
   "s": 7
  },
  {
   "t": "    m = copy.deepcopy(base)",
   "s": 7
  },
  {
   "t": "    with torch.no_grad():",
   "s": 7
  },
  {
   "t": "        for p, tv in zip(m.parameters(), task_vector):",
   "s": 7
  },
  {
   "t": "            p -= alpha * tv  # 그 방향의 반대로",
   "s": 7
  },
  {
   "t": "    return m",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "unlearn[\"작업 벡터 빼기 (α = 4)\"] = check(negate(4.0))",
   "s": 7
  }
 ],
 "names": {
  "1": "숫자와 두 번째 일",
  "2": "그냥 이어서 배우면: 파국적 망각",
  "3": "예전 것을 조금 다시 보기 (리플레이)",
  "4": "중요한 가중치는 덜 움직이게: EWC",
  "5": "같은 비용으로 비교: 처음부터 둘 다",
  "6": "숫자 3을 지워 주세요: 처음부터 다시, 남은 것만 더",
  "7": "빠르게 잊게 하기: 경사 상승과 작업 벡터 빼기"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 일 A(원래 숫자)만 배운 모델",
   "lines": [
    [
     "학습 1257장, 평가 540장",
     1
    ],
    [
     "일 A 정확도 → 0.978",
     2
    ],
    [
     "일 B(픽셀을 섞은 같은 숫자) 정확도 → 0.044",
     3,
     "newf"
    ]
   ],
   "notes": [
    "같은 숫자라도 픽셀 자리를 섞으면 모델에게는 전혀 다른 일이에요. 찍는 수준(0.1)에도 못 미쳐요. 이제 이 모델에게 일 B를 더 가르쳐요."
   ]
  },
  "f_task": {
   "title": "두 가지 일",
   "images": [
    [
     "img/continual/two_tasks.png",
     ""
    ]
   ],
   "notes": [
    "오른쪽은 왼쪽 그림의 64픽셀을 정해진 순서로 섞은 거예요. 사람 눈에는 아무것도 아니지만, 숫자는 그대로라 같은 답을 내야 해요."
   ],
   "chart": true
  },
  "s2": {
   "title": "②~⑤ 일 A를 배운 뒤 일 B를 30에폭 배우면",
   "cols": [
    "방법",
    "일 A (예전 일)",
    "일 B (새 일)"
   ],
   "rows": [
    [
     "일 A만",
     "0.978",
     "0.044"
    ],
    [
     "그냥 이어서",
     "0.231",
     "0.976"
    ],
    [
     "리플레이 100장",
     "0.896",
     "0.980"
    ],
    [
     "EWC λ = 1000",
     "0.900",
     "0.922"
    ],
    [
     "처음부터 둘 다",
     "0.976",
     "0.976"
    ]
   ],
   "notes": [
    "그냥 이어서 배우면 일 B는 0.98로 잘하게 되지만, 일 A는 0.98에서 0.23로 무너져요. 같은 가중치를 새 일에 맞춰 덮어쓰기 때문이에요(파국적 망각).",
    "일 A의 학습 데이터 1257장 중 100장만 남겨 섞어도 0.90를 지켜요. EWC는 예전 데이터를 하나도 안 쓰고 0.90를 지키지만, 대신 일 B가 0.92로 조금 덜 배워져요.",
    "두 일을 처음부터 함께 배우는 게 상한이에요. 하지만 실제로는 예전 데이터를 다 가지고 있지 못하거나(개인정보, 저장 비용), 매번 처음부터 다시 배우기엔 너무 비싸서 평생 학습이 필요해요."
   ]
  },
  "f_curve": {
   "title": "일 B를 배우는 동안 일 A는",
   "images": [
    [
     "img/continual/forgetting_curve.png",
     ""
    ]
   ],
   "notes": [
    "분홍(그냥 이어서)은 첫 에폭에 이미 일 A가 0.80로 떨어지고, 일 B를 더 배울수록 계속 떨어져 0.23까지 가요.",
    "보라(리플레이)와 파랑(EWC)은 일 A를 0.9 근처에서 지켜요. 오른쪽을 보면 일 B는 분홍과 보라가 몇 에폭 만에 거의 다 배우고, 파랑은 가중치가 묶인 만큼 조금 낮게 머물러요."
   ],
   "chart": true
  },
  "t3": {
   "title": "리플레이로 남겨 두는 장 수를 바꾸면",
   "cols": [
    "남긴 장 수",
    "일 A",
    "일 B"
   ],
   "rows": [
    [
     20,
     "0.607",
     "0.978"
    ],
    [
     "<b>100</b> (지금)",
     "0.896",
     "0.980"
    ],
    [
     300,
     "0.931",
     "0.972"
    ]
   ],
   "notes": [
    "20장(일 A의 2%)만 남겨도 그냥 이어서보다 훨씬 덜 잊어요. 많이 남길수록 더 잘 지키지만, 저장과 계산이 늘어요."
   ]
  },
  "t4": {
   "title": "EWC의 λ를 바꾸면",
   "cols": [
    "λ",
    "일 A",
    "일 B"
   ],
   "rows": [
    [
     10,
     "0.609",
     "0.961"
    ],
    [
     100,
     "0.835",
     "0.959"
    ],
    [
     "<b>1000</b> (지금)",
     "0.900",
     "0.922"
    ],
    [
     10000,
     "0.902",
     "0.915"
    ]
   ],
   "notes": [
    "λ가 작으면 그냥 이어서 배우기에 가깝고, 크면 가중치가 일 A 때 자리에 묶여 일 B를 덜 배워요. 예전 일과 새 일 사이의 저울이에요(안정성 대 가소성)."
   ]
  },
  "s6": {
   "title": "⑥⑦ 숫자 3을 지우기",
   "cols": [
    "방법",
    "지울 숫자 (test)",
    "나머지 (test)",
    "지울 숫자 (학습에 쓴 것)"
   ],
   "rows": [
    [
     "원래 모델",
     "0.982",
     "0.977",
     "1.000"
    ],
    [
     "처음부터 다시",
     "0.000",
     "0.977",
     "0.000"
    ],
    [
     "남은 것으로 3에폭 더",
     "0.600",
     "0.975",
     "0.602"
    ],
    [
     "경사 상승 30걸음",
     "0.000",
     "0.965",
     "0.008"
    ],
    [
     "작업 벡터 빼기 (α = 4)",
     "0.000",
     "0.959",
     "0.000"
    ]
   ],
   "notes": [
    "처음부터 지울 데이터 없이 다시 배운 모델이 정답이에요. 숫자 3을 전혀 못 맞히고(0.00) 나머지는 그대로예요. 하지만 큰 모델은 다시 배우는 데 몇 주와 큰돈이 들어요.",
    "남은 데이터로 조금 더 배우기만 하면 숫자 3을 아직 60% 맞혀요. 안 보여 주는 것만으로는 빨리 잊지 않아요.",
    "경사 상승(지울 데이터의 손실을 일부러 키우기)은 30걸음 만에 0으로 만들었지만, 나머지가 0.977에서 0.965로 조금 떨어졌어요. 손실을 키우는 건 끝이 없어서, 너무 오래 하면 모델 전체가 망가져요.",
    "작업 벡터 빼기는 \"숫자 3만 더 배운 모델 − 원래 모델\"을 그 방향으로 가는 화살표로 보고, 반대로 α배 움직여요. 학습 없이 덧셈 한 번이에요."
   ]
  },
  "t7": {
   "title": "작업 벡터를 몇 배 빼나 (α)",
   "cols": [
    "α",
    "지울 숫자 (test)",
    "나머지 (test)",
    "지울 숫자 (학습에 쓴 것)"
   ],
   "rows": [
    [
     1.0,
     "0.782",
     "0.984",
     "0.805"
    ],
    [
     2.0,
     "0.382",
     "0.986",
     "0.422"
    ],
    [
     "<b>4.0</b> (지금)",
     "0.000",
     "0.959",
     "0.000"
    ],
    [
     8.0,
     "0.000",
     "0.761",
     "0.000"
    ]
   ],
   "notes": [
    "조금 빼면 덜 잊고, 많이 빼면 다 잊지만 나머지 숫자까지 흔들려요. 지울 데이터에 대한 성능과 나머지 성능 사이에서 α를 골라요."
   ]
  },
  "f_ul": {
   "title": "네 가지 잊게 하기",
   "images": [
    [
     "img/continual/unlearning.png",
     ""
    ]
   ],
   "notes": [
    "\"처음부터 다시\"와 같은 모양(분홍 0, 보라 그대로)에 가까울수록 잘 잊은 거예요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   1,
   "두 가지 일",
   "원래, 픽셀 섞음",
   "A만 배움"
  ],
  [
   2,
   "그냥 이어서",
   "덮어쓰기",
   "파국적 망각"
  ],
  [
   3,
   "리플레이",
   "예전 것 조금",
   "장 수"
  ],
  [
   4,
   "EWC",
   "중요한 가중치 묶기",
   "λ"
  ],
  [
   5,
   "비교",
   "함께 배우기가 상한",
   "안정성 대 가소성"
  ],
  [
   6,
   "지워 주세요",
   "다시 학습이 정답",
   "안 보여 주기는 느림"
  ],
  [
   7,
   "빠르게 잊기",
   "경사 상승, 작업 벡터",
   "α"
  ]
 ],
 "env": "Python 3.13.5, PyTorch 2.13.0+cpu, scikit-learn 1.9.0 (CPU 1스레드, seed 0)"
};
