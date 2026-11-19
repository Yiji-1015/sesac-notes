window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import pandas as pd",
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
   "t": "from torch.utils.data import DataLoader, TensorDataset",
   "s": 5
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\" / \"points.csv\"",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① GPU 확인하고 Tensor로 바꾸기",
   "s": 1
  },
  {
   "t": "gpu = torch.cuda.is_available()",
   "s": 1
  },
  {
   "t": "device = \"cpu\"  # 이 예제는 작아서 CPU로 충분하고, 결과가 매번 똑같이 나와요",
   "s": 1
  },
  {
   "t": "df = pd.read_csv(DATA)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def tensors(split):",
   "s": 1
  },
  {
   "t": "    part = df[df[\"split\"] == split]",
   "s": 1
  },
  {
   "t": "    X = torch.tensor(part[[\"x1\", \"x2\"]].values, dtype=torch.float32)",
   "s": 1
  },
  {
   "t": "    y = torch.tensor(part[\"label\"].values, dtype=torch.float32).unsqueeze(1)  # (N,) → (N, 1)",
   "s": 1
  },
  {
   "t": "    return X.to(device), y.to(device)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "X_train, y_train = tensors(\"train\")",
   "s": 1
  },
  {
   "t": "X_valid, y_valid = tensors(\"valid\")",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 선 하나로 나눠 보기",
   "s": 2
  },
  {
   "t": "def accuracy(model, X, y):",
   "s": 2
  },
  {
   "t": "    with torch.no_grad():  # 채점할 때는 기울기 계산이 필요 없어요",
   "s": 2
  },
  {
   "t": "        return ((model(X) > 0).float() == y).float().mean().item()",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 2
  },
  {
   "t": "linear = nn.Linear(2, 1)  # w1*x1 + w2*x2 + b",
   "s": 2
  },
  {
   "t": "opt = torch.optim.SGD(linear.parameters(), lr=0.1)",
   "s": 2
  },
  {
   "t": "for _ in range(500):",
   "s": 2
  },
  {
   "t": "    loss = nn.functional.binary_cross_entropy_with_logits(linear(X_train), y_train)",
   "s": 2
  },
  {
   "t": "    opt.zero_grad()",
   "s": 2
  },
  {
   "t": "    loss.backward()",
   "s": 2
  },
  {
   "t": "    opt.step()",
   "s": 2
  },
  {
   "t": "linear_acc = accuracy(linear, X_valid, y_valid)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 신경망 만들기",
   "s": 3
  },
  {
   "t": "def make_net(hidden=16, depth=1, dropout=0.0):",
   "s": 3
  },
  {
   "t": "    layers, size = [], 2",
   "s": 3
  },
  {
   "t": "    for _ in range(depth):",
   "s": 3
  },
  {
   "t": "        layers += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout)]",
   "s": 3
  },
  {
   "t": "        size = hidden",
   "s": 3
  },
  {
   "t": "    return nn.Sequential(*layers, nn.Linear(size, 1)).to(device)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 3
  },
  {
   "t": "net = make_net()",
   "s": 3
  },
  {
   "t": "n_params = sum(p.numel() for p in net.parameters())",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 손실과 역전파 한 번",
   "s": 4
  },
  {
   "t": "loss_fn = nn.BCEWithLogitsLoss()",
   "s": 4
  },
  {
   "t": "loss = loss_fn(net(X_train), y_train)",
   "s": 4
  },
  {
   "t": "loss_before = loss.item()  # 학습 전 손실",
   "s": 4
  },
  {
   "t": "loss.backward()  # 모든 가중치에 대해 \"손실을 줄이려면 어느 쪽으로\" 를 계산",
   "s": 4
  },
  {
   "t": "grad_first = net[0].weight.grad.clone()",
   "s": 4
  },
  {
   "t": "net.zero_grad()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 학습 루프",
   "s": 5
  },
  {
   "t": "loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,",
   "s": 5
  },
  {
   "t": "                    generator=torch.Generator().manual_seed(0))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def train(model, optimizer, epochs=100, loader=loader):",
   "s": 5
  },
  {
   "t": "    X_seen, y_seen = loader.dataset.tensors  # 실제로 배우는 데 쓴 데이터",
   "s": 5
  },
  {
   "t": "    history = []",
   "s": 5
  },
  {
   "t": "    for epoch in range(epochs):",
   "s": 5
  },
  {
   "t": "        model.train()  # Dropout 같은 학습용 동작 켜기",
   "s": 5
  },
  {
   "t": "        for xb, yb in loader:",
   "s": 5
  },
  {
   "t": "            loss = loss_fn(model(xb), yb)",
   "s": 5
  },
  {
   "t": "            optimizer.zero_grad()",
   "s": 5
  },
  {
   "t": "            loss.backward()",
   "s": 5
  },
  {
   "t": "            optimizer.step()",
   "s": 5
  },
  {
   "t": "        model.eval()  # 채점할 때는 끄기",
   "s": 5
  },
  {
   "t": "        with torch.no_grad():",
   "s": 5
  },
  {
   "t": "            history.append({\"epoch\": epoch + 1, \"train_loss\": loss_fn(model(X_seen), y_seen).item(),",
   "s": 5
  },
  {
   "t": "                            \"valid_loss\": loss_fn(model(X_valid), y_valid).item(),",
   "s": 5
  },
  {
   "t": "                            \"valid_acc\": accuracy(model, X_valid, y_valid)})",
   "s": 5
  },
  {
   "t": "    return history",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "net = make_net()",
   "s": 5
  },
  {
   "t": "hist = train(net, torch.optim.Adam(net.parameters(), lr=0.01))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  }
 ],
 "names": {
  "1": "GPU 확인하고 Tensor로 바꾸기",
  "2": "선 하나로 나눠 보기",
  "3": "신경망 만들기",
  "4": "손실과 역전파 한 번",
  "5": "학습 루프",
  "6": "학습률 비교",
  "7": "Optimizer 비교",
  "8": "과적합 만들기",
  "9": "규제: Dropout과 weight decay",
  "10": "조기 종료: valid 손실이 가장 낮았던 때로 돌아가기",
  "11": "마지막 한 번: test"
 },
 "range": [
  1,
  5
 ],
 "outs": {
  "data": {
   "title": "points.csv 앞 5줄",
   "cols": [
    "id",
    "x1",
    "x2",
    "label",
    "split"
   ],
   "rows": [
    [
     "D10_0110",
     "0.0748258",
     "0.9682801",
     "0",
     "train"
    ],
    [
     "D10_0192",
     "-0.1642391",
     "1.4372999",
     "0",
     "test"
    ],
    [
     "D10_0354",
     "1.9158348",
     "0.1583332",
     "1",
     "valid"
    ],
    [
     "D10_0087",
     "-0.5322674",
     "0.7706648",
     "0",
     "train"
    ],
    [
     "D10_0296",
     "1.8406929",
     "0.268489",
     "1",
     "train"
    ]
   ],
   "notes": [
    "점마다 좌표 두 개(<code>x1</code>, <code>x2</code>)와 무리 번호(<code>label</code> 0 또는 1)가 있어요. train 240개로 배우고 valid 80개로 재요."
   ]
  },
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    [
     "torch.cuda.is_available() → True",
     1
    ],
    [
     "X_train.shape → (240, 2), dtype torch.float32",
     2
    ],
    "y_train.shape → (240, 1)"
   ],
   "notes": [
    "이 PC에는 GPU가 있어요. 그래도 이 예제는 작아서 CPU로 학습해요. GPU는 숫자 계산을 아주 많이 동시에 할 때 빨라요.",
    "점 240개, 좌표 2개짜리 표가 Tensor가 됐어요. 신경망은 보통 32비트 실수로 계산해요."
   ]
  },
  "s2": {
   "title": "② 선 하나(nn.Linear)로 나누기, 500번 학습",
   "lines": [
    [
     "valid 정확도 → 0.812",
     1
    ]
   ],
   "notes": [
    "두 무리가 직선 하나로는 깔끔하게 나뉘지 않는 모양이라, 아무리 학습해도 80% 근처에서 멈춰요. 곡선 경계가 필요해요."
   ]
  },
  "s3": {
   "title": "③ make_net()",
   "lines": [
    "Sequential(   (0): Linear(in_features=2, out_features=16, bias=True)   (1): ReLU()   (2): Dropout(p=0.0, inplace=False)   (3): Linear(in_features=16, out_features=1, bias=True) )",
    [
     "가중치 수 → 65개",
     1
    ]
   ],
   "notes": [
    "2 → 16 → 1. 가운데 층 16칸과 ReLU(음수는 0으로) 덕분에 경계를 구부릴 수 있어요. 가중치 65개 = (2×16 + 16) + (16×1 + 1)"
   ]
  },
  "s4": {
   "title": "④ 손실을 계산하고 backward() 한 번",
   "lines": [
    "loss_before → 0.6799 (학습 전)",
    [
     "net[0].weight.grad.shape → (16, 2)",
     1
    ],
    "grad 앞 2줄 → [[-0.0002, -0.0009], [0.0003, -0.0001]]"
   ],
   "notes": [
    "첫 층 가중치(16×2)와 같은 모양의 기울기가 생겼어요. 각 숫자는 \"이 가중치를 조금 키우면 손실이 얼마나 변하나\"예요. 이 기울기의 반대 방향으로 가중치를 조금씩 옮기는 게 학습이에요."
   ]
  },
  "s5": {
   "title": "⑤ Adam, 학습률 0.01, 100 epoch",
   "cols": [
    "epoch",
    "train 손실",
    "valid 손실",
    "valid 정확도"
   ],
   "rows": [
    [
     1,
     "0.615",
     "0.628",
     "0.762"
    ],
    [
     20,
     "0.222",
     "0.231",
     "0.925"
    ],
    [
     40,
     "0.149",
     "0.147",
     "0.950"
    ],
    [
     60,
     "0.106",
     "0.097",
     "0.975"
    ],
    [
     80,
     "0.080",
     "0.072",
     "0.975"
    ],
    [
     100,
     "0.065",
     "0.055",
     "0.975"
    ]
   ],
   "notes": [
    "손실이 줄면서 valid 정확도가 0.975까지 올라갔어요. 선 하나(②, 0.812)로는 못 넘던 벽을 넘었어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "Tensor",
   "표 → Tensor",
   "GPU 확인"
  ],
  [
   2,
   "선 하나",
   "nn.Linear",
   "곡선을 못 그림"
  ],
  [
   3,
   "신경망",
   "Linear, ReLU",
   "가중치 65개"
  ],
  [
   4,
   "역전파",
   "backward()",
   "기울기"
  ],
  [
   5,
   "학습 루프",
   "batch, epoch",
   "valid 0.975"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, CPU 학습 (GPU: NVIDIA GeForce RTX 5080 Laptop GPU)"
};
