window.PAGE = {
 "code": [
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 4
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 이용 기록 만들고 나누기",
   "s": 1
  },
  {
   "t": "rng = np.random.default_rng(0)",
   "s": 1
  },
  {
   "t": "U, I, G = 600, 400, 8  # 사용자, 아이템, 장르 수",
   "s": 1
  },
  {
   "t": "genre = rng.integers(G, size=I)  # 아이템마다 장르 하나",
   "s": 1
  },
  {
   "t": "popularity = rng.pareto(1.2, I) + 1  # 소수의 아이템이 아주 인기 있게 (멱법칙)",
   "s": 1
  },
  {
   "t": "taste = rng.dirichlet(np.full(G, 0.3), size=U)  # 사용자마다 장르 취향 (몇 장르에 몰림)",
   "s": 1
  },
  {
   "t": "true_pref = taste[:, genre] * popularity[None, :] ** 0.6  # 진짜 선호 (실제로는 모르는 값)",
   "s": 1
  },
  {
   "t": "true_prob = true_pref / true_pref.sum(axis=1, keepdims=True)",
   "s": 1
  },
  {
   "t": "R = np.zeros((U, I), dtype=bool)  # 본 적 있으면 True (암묵적 피드백)",
   "s": 1
  },
  {
   "t": "for u in range(U):",
   "s": 1
  },
  {
   "t": "    R[u, rng.choice(I, size=rng.poisson(25) + 5, replace=False, p=true_prob[u])] = True",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "train, test = R.copy(), np.zeros_like(R)  # 사용자마다 기록의 30%를 숨겨 정답으로",
   "s": 1
  },
  {
   "t": "for u in range(U):",
   "s": 1
  },
  {
   "t": "    seen = rng.permutation(np.flatnonzero(R[u]))",
   "s": 1
  },
  {
   "t": "    hide = seen[:round(len(seen) * 0.3)]",
   "s": 1
  },
  {
   "t": "    test[u, hide], train[u, hide] = True, False",
   "s": 1
  },
  {
   "t": "X = train.astype(np.float32)",
   "s": 1
  },
  {
   "t": "density = R.mean()",
   "s": 1
  },
  {
   "t": "top10_share = np.sort(R.sum(axis=0))[::-1][: I // 10].sum() / R.sum()  # 상위 10% 아이템이 차지한 기록 비율",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 평가 지표와 인기순 기준선",
   "s": 2
  },
  {
   "t": "K = 10",
   "s": 2
  },
  {
   "t": "DISCOUNT = 1 / np.log2(np.arange(2, K + 2))  # 위에 있을수록 크게 쳐 주는 무게",
   "s": 2
  },
  {
   "t": "def evaluate(scores):",
   "s": 2
  },
  {
   "t": "    s = scores.astype(float).copy()",
   "s": 2
  },
  {
   "t": "    s[train] = -np.inf  # 이미 본 것은 추천하지 않기",
   "s": 2
  },
  {
   "t": "    top = np.argsort(-s, axis=1)[:, :K]",
   "s": 2
  },
  {
   "t": "    out = {\"정밀도@10\": [], \"재현율@10\": [], \"NDCG@10\": []}",
   "s": 2
  },
  {
   "t": "    for u in range(U):",
   "s": 2
  },
  {
   "t": "        if not test[u].any():",
   "s": 2
  },
  {
   "t": "            continue",
   "s": 2
  },
  {
   "t": "        hit = test[u, top[u]]",
   "s": 2
  },
  {
   "t": "        out[\"정밀도@10\"].append(hit.mean())  # 추천 10개 중 맞힌 비율",
   "s": 2
  },
  {
   "t": "        out[\"재현율@10\"].append(hit.sum() / test[u].sum())  # 숨긴 것 중 찾아낸 비율",
   "s": 2
  },
  {
   "t": "        out[\"NDCG@10\"].append((hit * DISCOUNT).sum() / DISCOUNT[: min(test[u].sum(), K)].sum())  # 순서까지",
   "s": 2
  },
  {
   "t": "    res = {k: float(np.mean(v)) for k, v in out.items()}",
   "s": 2
  },
  {
   "t": "    res[\"커버리지\"] = len(np.unique(top)) / I  # 한 번이라도 추천된 아이템 비율",
   "s": 2
  },
  {
   "t": "    return res",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "results = {\"인기순\": evaluate(np.tile(X.sum(axis=0), (U, 1))), \"진짜 선호 (천장)\": evaluate(true_pref)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 이웃으로 추천하기: 아이템 기반 협업 필터링",
   "s": 3
  },
  {
   "t": "def cosine(A):  # 열끼리 코사인 유사도",
   "s": 3
  },
  {
   "t": "    n = np.linalg.norm(A, axis=0) + 1e-9",
   "s": 3
  },
  {
   "t": "    S = (A.T @ A) / n[:, None] / n[None, :]",
   "s": 3
  },
  {
   "t": "    np.fill_diagonal(S, 0)",
   "s": 3
  },
  {
   "t": "    return S",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "item_sim = cosine(X)  # 같은 사람이 함께 본 아이템일수록 닮음",
   "s": 3
  },
  {
   "t": "results[\"아이템 이웃\"] = evaluate(X @ item_sim)",
   "s": 3
  },
  {
   "t": "user_sim = cosine(X.T)",
   "s": 3
  },
  {
   "t": "results[\"사용자 이웃\"] = evaluate(user_sim @ X)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 숨은 취향 찾기: 행렬 분해 (BPR)",
   "s": 4
  },
  {
   "t": "torch.set_num_threads(1)",
   "s": 4
  },
  {
   "t": "def bpr_mf(dim=64, epochs=30, lr=0.01, reg=0.05, seed=0):",
   "s": 4
  },
  {
   "t": "    torch.manual_seed(seed)",
   "s": 4
  },
  {
   "t": "    g = np.random.default_rng(seed)",
   "s": 4
  },
  {
   "t": "    P = torch.nn.Parameter(torch.randn(U, dim) * 0.1)  # 사용자 벡터",
   "s": 4
  },
  {
   "t": "    Q = torch.nn.Parameter(torch.randn(I, dim) * 0.1)  # 아이템 벡터",
   "s": 4
  },
  {
   "t": "    opt = torch.optim.Adam([P, Q], lr=lr)",
   "s": 4
  },
  {
   "t": "    users, items = np.nonzero(train)",
   "s": 4
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 4
  },
  {
   "t": "        for b in np.array_split(g.permutation(len(users)), len(users) // 1024):",
   "s": 4
  },
  {
   "t": "            u, i = torch.tensor(users[b]), torch.tensor(items[b])",
   "s": 4
  },
  {
   "t": "            j = torch.tensor(g.integers(I, size=len(b)))  # 안 본 것으로 치는 아이템 (음성 샘플)",
   "s": 4
  },
  {
   "t": "            x = (P[u] * (Q[i] - Q[j])).sum(dim=1)  # 본 것 점수 − 안 본 것 점수",
   "s": 4
  },
  {
   "t": "            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * (P[u].pow(2).sum() + Q[i].pow(2).sum() + Q[j].pow(2).sum()) / len(b)",
   "s": 4
  },
  {
   "t": "            opt.zero_grad()",
   "s": 4
  },
  {
   "t": "            loss.backward()",
   "s": 4
  },
  {
   "t": "            opt.step()",
   "s": 4
  },
  {
   "t": "    return (P @ Q.T).detach().numpy()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "results[\"행렬 분해 (BPR)\"] = evaluate(bpr_mf())",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 닫힌 식 하나로: EASE",
   "s": 5
  },
  {
   "t": "def ease(lam=200.0):  # 아이템 × 아이템 가중치 B를 한 번에 풀기 (대각선은 0)",
   "s": 5
  },
  {
   "t": "    P = np.linalg.inv(X.T @ X + lam * np.eye(I))",
   "s": 5
  },
  {
   "t": "    B = -P / np.diag(P)[None, :]",
   "s": 5
  },
  {
   "t": "    np.fill_diagonal(B, 0)",
   "s": 5
  },
  {
   "t": "    return X @ B",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "results[\"EASE\"] = evaluate(ease())",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 이용 기록을 그래프로: LightGCN",
   "s": 6
  },
  {
   "t": "def lightgcn(dim=64, layers=2, epochs=30, lr=0.01, reg=0.01, seed=0):",
   "s": 6
  },
  {
   "t": "    torch.manual_seed(seed)",
   "s": 6
  },
  {
   "t": "    g = np.random.default_rng(seed)",
   "s": 6
  },
  {
   "t": "    A = np.zeros((U + I, U + I), dtype=np.float32)  # 사용자와 아이템을 노드로, 본 기록을 선으로",
   "s": 6
  },
  {
   "t": "    A[:U, U:], A[U:, :U] = X, X.T",
   "s": 6
  },
  {
   "t": "    deg = A.sum(axis=1)",
   "s": 6
  },
  {
   "t": "    d = np.where(deg > 0, deg ** -0.5, 0)",
   "s": 6
  },
  {
   "t": "    A_norm = torch.tensor(d[:, None] * A * d[None, :])  # GNN 쪽의 정규화한 인접 행렬과 같아요",
   "s": 6
  },
  {
   "t": "    E = torch.nn.Parameter(torch.randn(U + I, dim) * 0.1)",
   "s": 6
  },
  {
   "t": "    opt = torch.optim.Adam([E], lr=lr)",
   "s": 6
  },
  {
   "t": "    def embed():  # 이웃 벡터를 layers번 모아 평균 (가중치, 활성화 없음)",
   "s": 6
  },
  {
   "t": "        h, out = E, [E]",
   "s": 6
  },
  {
   "t": "        for _ in range(layers):",
   "s": 6
  },
  {
   "t": "            h = A_norm @ h",
   "s": 6
  },
  {
   "t": "            out.append(h)",
   "s": 6
  },
  {
   "t": "        return torch.stack(out).mean(dim=0)",
   "s": 6
  },
  {
   "t": "    users, items = np.nonzero(train)",
   "s": 6
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 6
  },
  {
   "t": "        for b in np.array_split(g.permutation(len(users)), len(users) // 1024):",
   "s": 6
  },
  {
   "t": "            F = embed()",
   "s": 6
  },
  {
   "t": "            u, i, j = torch.tensor(users[b]), torch.tensor(items[b]) + U, torch.tensor(g.integers(I, size=len(b))) + U",
   "s": 6
  },
  {
   "t": "            x = (F[u] * (F[i] - F[j])).sum(dim=1)",
   "s": 6
  },
  {
   "t": "            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * (E[u].pow(2).sum() + E[i].pow(2).sum() + E[j].pow(2).sum()) / len(b)",
   "s": 6
  },
  {
   "t": "            opt.zero_grad()",
   "s": 6
  },
  {
   "t": "            loss.backward()",
   "s": 6
  },
  {
   "t": "            opt.step()",
   "s": 6
  },
  {
   "t": "    F = embed().detach().numpy()",
   "s": 6
  },
  {
   "t": "    return F[:U] @ F[U:].T",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "results[\"LightGCN\"] = evaluate(lightgcn())",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 기록이 적은 사용자와 인기 편향",
   "s": 7
  },
  {
   "t": "activity = train.sum(axis=1)",
   "s": 7
  },
  {
   "t": "lo, hi = np.quantile(activity, [1 / 3, 2 / 3]).astype(int)  # 사용자 수가 비슷하게 셋으로",
   "s": 7
  },
  {
   "t": "groups = {f\"기록 {lo}개 이하\": activity <= lo, f\"{lo + 1}~{hi}개\": (activity > lo) & (activity <= hi), f\"{hi + 1}개 이상\": activity > hi}",
   "s": 7
  },
  {
   "t": "def ndcg_by_group(scores):",
   "s": 7
  },
  {
   "t": "    s = scores.astype(float).copy()",
   "s": 7
  },
  {
   "t": "    s[train] = -np.inf",
   "s": 7
  },
  {
   "t": "    top = np.argsort(-s, axis=1)[:, :K]",
   "s": 7
  },
  {
   "t": "    nd = np.array([(test[u, top[u]] * DISCOUNT).sum() / DISCOUNT[: max(1, min(test[u].sum(), K))].sum() for u in range(U)])",
   "s": 7
  },
  {
   "t": "    return {name: float(nd[m].mean()) for name, m in groups.items()}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "by_group = {\"인기순\": ndcg_by_group(np.tile(X.sum(axis=0), (U, 1))), \"아이템 이웃\": ndcg_by_group(X @ item_sim), \"EASE\": ndcg_by_group(ease())}",
   "s": 7
  },
  {
   "t": "head = np.argsort(-X.sum(axis=0))[: I // 10]  # 인기 상위 10% 아이템",
   "s": 7
  },
  {
   "t": "def head_share(scores):  # 추천 목록 중 인기 상위 10% 아이템의 비율",
   "s": 7
  },
  {
   "t": "    s = scores.astype(float).copy()",
   "s": 7
  },
  {
   "t": "    s[train] = -np.inf",
   "s": 7
  },
  {
   "t": "    top = np.argsort(-s, axis=1)[:, :K]",
   "s": 7
  },
  {
   "t": "    return float(np.isin(top, head).mean())",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "popularity_bias = {\"인기순\": head_share(np.tile(X.sum(axis=0), (U, 1))), \"아이템 이웃\": head_share(X @ item_sim), \"EASE\": head_share(ease()),",
   "s": 7
  },
  {
   "t": "                   \"행렬 분해 (BPR)\": head_share(bpr_mf()), \"진짜 선호\": head_share(true_pref)}",
   "s": 7
  }
 ],
 "names": {
  "1": "이용 기록 만들고 나누기",
  "2": "평가 지표와 인기순 기준선",
  "3": "이웃으로 추천하기: 아이템 기반 협업 필터링",
  "4": "숨은 취향 찾기: 행렬 분해 (BPR)",
  "5": "닫힌 식 하나로: EASE",
  "6": "이용 기록을 그래프로: LightGCN",
  "7": "기록이 적은 사용자와 인기 편향"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 만든 이용 기록",
   "lines": [
    [
     "사용자 600명 × 아이템 400개 = 240,000칸 중 본 기록 18,137칸 (7.6%)",
     1
    ],
    [
     "인기 상위 10% 아이템(40개)이 전체 기록의 24%",
     2,
     "newf"
    ],
    [
     "사용자 한 명당 학습용 기록 평균 21.2개, 숨긴 정답 평균 9.1개",
     3
    ]
   ],
   "notes": [
    "표의 92%가 비어 있어요. 추천은 이 빈칸 중 \"볼 만한 것\"을 고르는 문제예요. 빈칸은 \"싫어한다\"가 아니라 \"아직 모른다\"예요.",
    "몇몇 아이템에 기록이 몰려 있어요(긴 꼬리). 그래서 인기순만으로도 어느 정도 맞히고, 대부분의 아이템은 기록이 적어 배우기 어려워요."
   ]
  },
  "f_pop": {
   "title": "아이템별 인기",
   "images": [
    [
     "img/recsys/long_tail.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽 몇 개가 아주 높고 오른쪽으로 길게 낮아져요. 실제 서비스(음악, 영화, 채용 공고)도 대개 이 모양이에요."
   ],
   "chart": true
  },
  "s2": {
   "title": "② 기준선 둘",
   "cols": [
    "방법",
    "정밀도@10",
    "재현율@10",
    "NDCG@10",
    "커버리지"
   ],
   "rows": [
    [
     "인기순",
     "0.102",
     "0.113",
     "0.134",
     "0.048"
    ],
    [
     "진짜 선호 (천장)",
     "0.237",
     "0.261",
     "0.310",
     "0.403"
    ]
   ],
   "notes": [
    "인기순은 거의 모두에게 같은 10개를 추천해서, 400개 중 19개만 추천 목록에 나와요(커버리지).",
    "진짜 선호로 순서를 매겨도 정밀도가 0.24예요. 사람은 좋아할 만한 것 중 일부만 우연히 보니까, 숨긴 것을 다 맞히는 건 불가능해요. 이 값이 이 데이터에서 어떤 모델도 넘기 어려운 천장이에요."
   ]
  },
  "s3": {
   "title": "③~⑥ 다섯 모델 (사용자 600명 평균)",
   "cols": [
    "방법",
    "정밀도@10",
    "재현율@10",
    "NDCG@10",
    "커버리지"
   ],
   "rows": [
    [
     "아이템 이웃",
     "0.200",
     "0.220",
     "0.262",
     "0.632"
    ],
    [
     "사용자 이웃",
     "0.175",
     "0.193",
     "0.217",
     "0.287"
    ],
    [
     "행렬 분해 (BPR)",
     "0.179",
     "0.198",
     "0.218",
     "0.718"
    ],
    [
     "EASE",
     "0.198",
     "0.218",
     "0.261",
     "0.590"
    ],
    [
     "LightGCN",
     "0.197",
     "0.217",
     "0.253",
     "0.647"
    ],
    [
     "진짜 선호 (천장)",
     "0.237",
     "0.261",
     "0.310",
     "0.403"
    ]
   ],
   "notes": [
    "가장 단순한 아이템 이웃이 NDCG 1등이에요. 닫힌 식 하나로 푸는 EASE와 그래프 모델 LightGCN이 거의 같고, 학습으로 사용자 벡터를 찾는 행렬 분해는 그보다 낮아요.",
    "추천 연구에서 \"새 신경망 모델이 잘 다듬은 단순한 기준선을 생각보다 못 이긴다\"는 보고가 꾸준히 나와요. 새 모델을 들이기 전에 인기순, 아이템 이웃, EASE를 먼저 같은 조건으로 재 봐요."
   ]
  },
  "t4": {
   "title": "행렬 분해의 규제 세기 reg를 바꾸면",
   "cols": [
    "reg",
    "NDCG@10",
    "커버리지"
   ],
   "rows": [
    [
     0.01,
     "0.186",
     "0.968"
    ],
    [
     "<b>0.05</b> (지금)",
     "0.218",
     "0.718"
    ],
    [
     0.1,
     "0.104",
     "0.295"
    ]
   ],
   "notes": [
    "규제가 약하면(0.01) 각 사용자의 학습 기록 20개 남짓에 벡터 64칸을 맞추느라 외워 버려요. 너무 세면(0.1) 모두를 비슷한 벡터로 눌러 인기순에 가까워지고 커버리지가 줄어요. 학습형 모델은 이런 손잡이를 잘 맞춰야 기준선 근처에 와요."
   ]
  },
  "t5": {
   "title": "EASE의 λ를 바꾸면",
   "cols": [
    "λ",
    "NDCG@10",
    "커버리지"
   ],
   "rows": [
    [
     50.0,
     "0.239",
     "0.838"
    ],
    [
     "<b>200.0</b> (지금)",
     "0.261",
     "0.590"
    ],
    [
     500.0,
     "0.252",
     "0.465"
    ]
   ],
   "notes": [
    "λ는 아이템끼리의 가중치 B를 작게 누르는 규제예요. 작으면 드문 아이템 쌍까지 믿어서 커버리지는 넓지만 덜 맞고, 크면 인기 있는 쪽으로 몰려요."
   ]
  },
  "t6": {
   "title": "LightGCN의 층 수를 바꾸면 (0층 = 그래프 없이 행렬 분해)",
   "cols": [
    "층",
    "NDCG@10",
    "커버리지"
   ],
   "rows": [
    [
     0,
     "0.186",
     "0.968"
    ],
    [
     1,
     "0.232",
     "0.887"
    ],
    [
     "<b>2</b> (지금)",
     "0.253",
     "0.647"
    ],
    [
     3,
     "0.263",
     "0.575"
    ]
   ],
   "notes": [
    "0층은 이웃을 전혀 모으지 않는 행렬 분해라, 위 표의 reg 0.01과 같은 값이에요. 한두 층 이웃 벡터를 모아 평균 내면 \"나와 같은 걸 본 사람이 본 것\"(2홉)이 들어와 점수가 크게 올라요. 3층까지는 계속 올라서 아이템 이웃과 같은 수준이 됐어요(0.263).",
    "층을 아주 많이 쌓으면 GNN 쪽에서 본 과평활처럼 모든 벡터가 비슷해질 수 있어서, 보통 2~4층을 써요."
   ]
  },
  "f_cmp": {
   "title": "모델 비교",
   "images": [
    [
     "img/recsys/model_compare.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽은 이웃, EASE, LightGCN이 비슷하게 높아요. 오른쪽은 행렬 분해가 가장 다양하게 추천하지만 덜 맞혀요. 인기순은 가장 좁아요."
   ],
   "chart": true
  },
  "s7": {
   "title": "⑦ 학습 기록 수로 사용자를 셋으로 나눠 본 NDCG@10",
   "cols": [
    "방법",
    "기록 20개 이하 (272명)",
    "21~22개 (142명)",
    "23개 이상 (186명)"
   ],
   "rows": [
    [
     "인기순",
     "0.132",
     "0.139",
     "0.134"
    ],
    [
     "아이템 이웃",
     "0.240",
     "0.267",
     "0.291"
    ],
    [
     "EASE",
     "0.236",
     "0.270",
     "0.291"
    ]
   ],
   "notes": [
    "기록이 많은 사용자일수록 잘 맞혀요. 기록이 하나도 없는 새 사용자(콜드 스타트)는 협업 필터링으로는 아예 추천할 수 없어서, 가입할 때 고른 관심사나 인기순, 아이템 내용(장르, 설명)으로 시작해요."
   ]
  },
  "s7b": {
   "title": "추천 목록 중 인기 상위 10% 아이템이 차지한 비율",
   "cols": [
    "방법",
    "비율"
   ],
   "rows": [
    [
     "인기순",
     "100%"
    ],
    [
     "아이템 이웃",
     "59%"
    ],
    [
     "EASE",
     "63%"
    ],
    [
     "행렬 분해 (BPR)",
     "35%"
    ],
    [
     "진짜 선호",
     "55%"
    ]
   ],
   "notes": [
    "진짜 선호대로 골라도 55%는 인기 아이템이에요(인기 있는 데는 이유가 있으니까). 아이템 이웃과 EASE는 그보다 조금 더 인기 쪽으로 기울어요(인기 편향). 인기 있는 것이 더 추천되고, 그래서 더 인기 있어지는 고리가 생길 수 있어요.",
    "행렬 분해는 반대로 인기 아이템을 덜 추천했어요. 커버리지는 넓지만, 그만큼 정답을 덜 맞혔어요(③). 다양성과 정확도 사이에서 어디에 설지는 서비스가 정해요."
   ]
  }
 },
 "flow": [
  [
   1,
   "이용 기록",
   "600 × 400, 92% 빈칸",
   "긴 꼬리"
  ],
  [
   2,
   "지표와 기준선",
   "정밀도, 재현율, NDCG",
   "인기순, 천장"
  ],
  [
   3,
   "이웃",
   "함께 본 것",
   "아이템 이웃 1등"
  ],
  [
   4,
   "행렬 분해",
   "숨은 취향 벡터",
   "BPR"
  ],
  [
   5,
   "EASE",
   "닫힌 식",
   "λ 하나"
  ],
  [
   6,
   "LightGCN",
   "그래프로 2홉",
   "층 수"
  ],
  [
   7,
   "편향",
   "기록 적은 사람, 인기",
   "콜드 스타트"
  ]
 ],
 "env": "Python 3.13.5, NumPy 2.5.2, PyTorch 2.13.0+cpu (CPU 1스레드, seed 0)"
};
