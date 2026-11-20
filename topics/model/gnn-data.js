window.PAGE = {
 "code": [
  {
   "t": "import networkx as nx",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 1
  },
  {
   "t": "from torch import nn",
   "s": 2
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 그래프를 Tensor로",
   "s": 1
  },
  {
   "t": "g = nx.karate_club_graph()",
   "s": 1
  },
  {
   "t": "n = g.number_of_nodes()",
   "s": 1
  },
  {
   "t": "A = torch.zeros(n, n)",
   "s": 1
  },
  {
   "t": "for u, v in g.edges():",
   "s": 1
  },
  {
   "t": "    A[u, v] = A[v, u] = 1.0  # 친구면 1 (방향 없음)",
   "s": 1
  },
  {
   "t": "y = torch.tensor([0 if g.nodes[i][\"club\"] == \"Mr. Hi\" else 1 for i in range(n)])",
   "s": 1
  },
  {
   "t": "X = torch.eye(n)  # 회원마다 \"자기 이름표\" 한 칸. 그 밖의 정보는 없음",
   "s": 1
  },
  {
   "t": "train_idx = torch.tensor([0, 1, 32, 33])  # 소속을 알려 주는 4명 (두 파의 대표와 그 옆 사람)",
   "s": 1
  },
  {
   "t": "test_mask = torch.ones(n, dtype=torch.bool)",
   "s": 1
  },
  {
   "t": "test_mask[train_idx] = False  # 나머지 30명으로 채점",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def acc(pred):",
   "s": 1
  },
  {
   "t": "    return (pred[test_mask] == y[test_mask]).float().mean().item()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def fit(model, steps=200, lr=0.01):",
   "s": 1
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 1
  },
  {
   "t": "    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)",
   "s": 1
  },
  {
   "t": "    for _ in range(steps):",
   "s": 1
  },
  {
   "t": "        loss = F.cross_entropy(model(X)[train_idx], y[train_idx])  # 아는 4명으로만 손실",
   "s": 1
  },
  {
   "t": "        opt.zero_grad(); loss.backward(); opt.step()",
   "s": 1
  },
  {
   "t": "    model.eval()",
   "s": 1
  },
  {
   "t": "    with torch.no_grad():",
   "s": 1
  },
  {
   "t": "        return acc(model(X).argmax(1))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 기준선: 연결을 무시한 신경망",
   "s": 2
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 2
  },
  {
   "t": "mlp = nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Linear(16, 2))",
   "s": 2
  },
  {
   "t": "mlp_acc = fit(mlp)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 이웃 정보 모으기를 손으로",
   "s": 3
  },
  {
   "t": "A_hat = A + torch.eye(n)  # 자기 자신도 이웃으로",
   "s": 3
  },
  {
   "t": "deg = A_hat.sum(1)",
   "s": 3
  },
  {
   "t": "A_norm = A_hat / torch.sqrt(deg[:, None] * deg[None, :])  # 이웃이 많은 사람의 몫을 줄여 크기 맞추기",
   "s": 3
  },
  {
   "t": "known = torch.zeros(n, 2)",
   "s": 3
  },
  {
   "t": "known[train_idx, y[train_idx]] = 1.0  # 아는 4명만 한 칸에 1",
   "s": 3
  },
  {
   "t": "spread = {k: torch.linalg.matrix_power(A_norm, k) @ known for k in (1, 2, 3)}  # k번 이웃을 거쳐 퍼뜨리기",
   "s": 3
  },
  {
   "t": "spread_acc = {k: acc(s.argmax(1)) for k, s in spread.items()}",
   "s": 3
  },
  {
   "t": "reached = {k: int((s.sum(1) > 0).sum()) for k, s in spread.items()}  # 정보가 닿은 사람 수",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ GCN: 이웃을 모으는 층을 학습하기",
   "s": 4
  },
  {
   "t": "class GCN(nn.Module):",
   "s": 4
  },
  {
   "t": "    def __init__(self, sizes, agg=\"norm\"):",
   "s": 4
  },
  {
   "t": "        super().__init__()",
   "s": 4
  },
  {
   "t": "        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))",
   "s": 4
  },
  {
   "t": "        self.agg = agg",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "    def gather(self, h):  # 이웃(과 자기)의 값을 모으기",
   "s": 4
  },
  {
   "t": "        if self.agg == \"norm\":",
   "s": 4
  },
  {
   "t": "            return A_norm @ h",
   "s": 4
  },
  {
   "t": "        if self.agg == \"mean\":",
   "s": 4
  },
  {
   "t": "            return (A_hat @ h) / A_hat.sum(1, keepdim=True)",
   "s": 4
  },
  {
   "t": "        if self.agg == \"sum\":",
   "s": 4
  },
  {
   "t": "            return A_hat @ h",
   "s": 4
  },
  {
   "t": "        return torch.stack([h[A_hat[i] > 0].max(0).values for i in range(n)])  # max",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "    def forward(self, x):",
   "s": 4
  },
  {
   "t": "        for i, lin in enumerate(self.layers):",
   "s": 4
  },
  {
   "t": "            x = self.gather(lin(x))  # 모든 사람에게 같은 가중치로 바꾼 뒤 이웃끼리 모으기",
   "s": 4
  },
  {
   "t": "            if i < len(self.layers) - 1:",
   "s": 4
  },
  {
   "t": "                x = F.relu(x)",
   "s": 4
  },
  {
   "t": "        return x",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 4
  },
  {
   "t": "gcn = GCN([n, 16, 2])",
   "s": 4
  },
  {
   "t": "gcn_acc = fit(gcn)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 모으는 방법과 층 수 바꾸기",
   "s": 5
  },
  {
   "t": "agg_acc = {}",
   "s": 5
  },
  {
   "t": "for agg in (\"norm\", \"mean\", \"sum\", \"max\"):",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "    agg_acc[agg] = fit(GCN([n, 16, 2], agg))",
   "s": 5
  },
  {
   "t": "depth_acc = {}",
   "s": 5
  },
  {
   "t": "for depth in (2, 4, 8, 16):",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "    depth_acc[depth] = fit(GCN([n] + [16] * (depth - 1) + [2]))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def avg_cosine(h):  # 모든 두 사람 사이 코사인 유사도의 평균. 1에 가까울수록 다 비슷해진 것",
   "s": 5
  },
  {
   "t": "    u = F.normalize(h, dim=1)",
   "s": 5
  },
  {
   "t": "    s = u @ u.T",
   "s": 5
  },
  {
   "t": "    return ((s.sum() - n) / (n * (n - 1))).item()",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "smooth = {k: avg_cosine(torch.linalg.matrix_power(A_norm, k) @ X) for k in (0, 1, 2, 4, 8, 16, 32)}  # 모으기만 k번",
   "s": 5
  }
 ],
 "names": {
  "1": "그래프를 Tensor로",
  "2": "기준선: 연결을 무시한 신경망",
  "3": "이웃 정보 모으기를 손으로",
  "4": "GCN: 이웃을 모으는 층을 학습하기",
  "5": "모으는 방법과 층 수 바꾸기"
 },
 "range": [
  1,
  5
 ],
 "outs": {
  "data": {
   "title": "데이터: 회원 34명과 친구 관계 78개",
   "images": [
    [
     "img/gnn/graph.png",
     "보라: 사범(0번) 쪽, 분홍: 회장(33번) 쪽"
    ]
   ],
   "notes": [
    "1970년대 미국의 한 가라테 클럽이 사범과 회장의 다툼으로 둘로 갈라졌어요. 선은 \"클럽 밖에서도 만나는 사이\"예요. 이 그래프는 그래프 연구에서 가장 많이 쓰는 예제예요.",
    "4명(굵은 테두리: 0, 1번과 32, 33번)의 소속만 알려 주고, 나머지 30명이 어느 쪽으로 갔는지 맞혀요. 알려 주는 정보는 \"누가 누구와 친한가\"뿐이에요."
   ],
   "chart": true
  },
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    "A.shape → (34, 34), 친구 관계 78개 (A에는 양쪽으로 두 번)",
    "X.shape → (34, 34) (단위행렬)",
    [
     "y → 두 쪽 [17, 17]명, 알려 준 4명의 소속 [0, 0, 1, 1]",
     1
    ]
   ],
   "notes": [
    "두 파가 17명씩이라, 아무렇게나 찍으면 정확도가 0.5예요."
   ]
  },
  "s2": {
   "title": "② 연결을 무시한 신경망",
   "lines": [
    [
     "나머지 30명 정확도 → 0.50",
     1,
     "bad"
    ]
   ],
   "notes": [
    "입력이 \"자기 이름표\"뿐이라, 처음 보는 30명의 이름표는 학습에 한 번도 나오지 않았어요. 신경망은 이 사람들에 대해 아무것도 모르니 반반으로 찍어요. 이 문제의 정보는 전부 <b>연결</b>에 있어요."
   ]
  },
  "f_spread": {
   "title": "알려 준 4명의 소속을 친구 관계를 따라 퍼뜨리면",
   "images": [
    [
     "img/gnn/spread.png",
     "회색: 아직 정보가 안 닿음, 진할수록 회장 쪽, 주황 원: 틀린 사람"
    ]
   ],
   "notes": [
    "1번 퍼뜨리면 4명의 친구까지만 정보가 닿아요. 2번이면 모두에게 닿고, 30명 중 29명을 맞혀요. 학습한 것은 하나도 없어요. \"친구끼리는 같은 쪽으로 갔을 것\"이라는 가정 하나로 맞힌 거예요.",
    "틀린 사람은 양쪽 모두와 친한 경계에 있는 사람이에요."
   ],
   "chart": true
  },
  "s3": {
   "title": "③ 모으기만 k번 반복한 결과 (학습 없음)",
   "cols": [
    "k",
    "정보가 닿은 사람",
    "나머지 30명 정확도"
   ],
   "rows": [
    [
     1,
     "31명",
     "0.90"
    ],
    [
     2,
     "34명",
     "0.97"
    ],
    [
     3,
     "34명",
     "0.97"
    ]
   ],
   "notes": [
    "A_norm을 한 번 곱할 때마다 \"친구의 친구\"까지 한 칸씩 더 퍼져요. 이 그래프는 누구든 2칸이면 4명 중 누군가에게 닿아요."
   ]
  },
  "f_gcn": {
   "title": "2층 GCN이 4명만 보고 배운 결과",
   "images": [
    [
     "img/gnn/gcn.png",
     ""
    ]
   ],
   "notes": [
    "오른쪽: 학습에 쓴 건 4명의 정답뿐인데, 34명 모두가 두 무리로 갈라졌어요. 층마다 이웃끼리 값을 섞기 때문에, 친한 사람들이 비슷한 점수를 받아요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ GCN (34 → 16 → 2, Adam 200번)",
   "lines": [
    "가중치 수 → 594개",
    [
     "나머지 30명 정확도 → 0.97",
     1
    ]
   ],
   "notes": [
    "③의 손으로 퍼뜨리기와 같은 0.97가 나왔어요. 이 그래프에서는 \"친구끼리 같은 쪽\"이라는 가정이 거의 다 맞아서, 퍼뜨리기만으로도 충분해요. GCN은 여기에 \"어떤 정보를 어떻게 바꿔서 퍼뜨릴지\"를 배우는 가중치를 더한 거예요. 사람마다 나이, 수련 기간 같은 정보가 있다면 그것까지 섞어서 배울 수 있어요."
   ]
  },
  "t5a": {
   "title": "모으는 방법만 바꾸면",
   "cols": [
    "모으기",
    "나머지 30명 정확도"
   ],
   "rows": [
    [
     "<b>정규화한 합 (④)</b>",
     "0.97"
    ],
    [
     "평균",
     "0.97"
    ],
    [
     "그냥 합",
     "0.97"
    ],
    [
     "최댓값",
     "0.93"
    ]
   ],
   "notes": [
    "이 작은 그래프에서는 거의 차이가 없어요. 최댓값만 1명 더 틀렸어요.",
    "차이는 친구 수가 크게 다를 때 드러나요. 그냥 합은 친구가 많은 사람의 값이 커지고, 평균은 친구가 1명이든 100명이든 같은 크기예요. 대신 평균은 \"친구가 몇 명인가\" 자체를 잃어요. 최댓값은 \"이웃 중에 이런 사람이 하나라도 있나\"를 잘 잡아요."
   ]
  },
  "t5b": {
   "title": "층 수를 바꾸면 (칸 16개)",
   "cols": [
    "층 수",
    "나머지 30명 정확도"
   ],
   "rows": [
    [
     "<b>2</b>",
     "0.97"
    ],
    [
     4,
     "0.97"
    ],
    [
     8,
     "0.50"
    ],
    [
     16,
     "0.50"
    ]
   ],
   "notes": [
    "8층부터 반반으로 찍어요. 층마다 이웃과 섞으니 8층이면 8칸 떨어진 사람까지 섞이는데, 이 그래프는 누구든 5칸 안에 닿아서 모두가 비슷한 값이 돼 버려요(아래 그림). 층이 깊어 학습 자체도 어려워져요."
   ]
  },
  "f_smooth": {
   "title": "이름표 벡터를 A_norm으로 k번 섞은 뒤, 사람끼리 얼마나 비슷한가",
   "images": [
    [
     "img/gnn/oversmoothing.png",
     ""
    ]
   ],
   "notes": [
    "처음(k=0)에는 이름표가 모두 달라서 0이에요. 4번이면 0.61, 16번이면 0.97, 32번이면 0.999로 1에 붙어요. 누가 누구인지 구별할 정보가 사라지는 거예요.",
    "그래서 GCN은 보통 2~3층만 쌓아요. 더 깊이 가려면 <a href=\"cnn.html\">CNN</a>처럼 잔차 연결을 넣거나, 층마다 자기 원래 값을 일부 남겨요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   1,
   "그래프 → Tensor",
   "인접 행렬 A",
   "34명, 78관계"
  ],
  [
   2,
   "연결 무시",
   "이름표만",
   "0.50"
  ],
  [
   3,
   "퍼뜨리기",
   "A_norm 곱하기",
   "학습 없이 0.97"
  ],
  [
   4,
   "GCN",
   "바꾸고 모으기",
   "4명으로 학습"
  ],
  [
   5,
   "모으기와 깊이",
   "평균, 합, 최댓값",
   "과평활"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, networkx 3.7, CPU (seed 0)"
};
