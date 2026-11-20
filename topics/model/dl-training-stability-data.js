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
   "t": "import copy",
   "s": 10
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
  },
  {
   "t": "# ⑥ 학습률 비교",
   "s": 6
  },
  {
   "t": "def run(lr, opt_name=\"sgd\", epochs=100, **net_kw):",
   "s": 6
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "    model = make_net(**net_kw)",
   "s": 6
  },
  {
   "t": "    make = {\"sgd\": torch.optim.SGD, \"adam\": torch.optim.Adam}[opt_name]",
   "s": 6
  },
  {
   "t": "    return model, train(model, make(model.parameters(), lr=lr), epochs)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "lr_runs = {lr: run(lr)[1] for lr in (0.001, 0.1, 10.0)}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ Optimizer 비교",
   "s": 7
  },
  {
   "t": "opt_runs = {name: run(0.01, name, epochs=60)[1] for name in (\"sgd\", \"adam\")}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 과적합 만들기",
   "s": 8
  },
  {
   "t": "small = DataLoader(TensorDataset(X_train[:40], y_train[:40]), batch_size=8, shuffle=True,",
   "s": 8
  },
  {
   "t": "                   generator=torch.Generator().manual_seed(0))  # 학습 데이터를 40개로 줄이기",
   "s": 8
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 8
  },
  {
   "t": "big = make_net(hidden=128, depth=3)",
   "s": 8
  },
  {
   "t": "over = train(big, torch.optim.Adam(big.parameters(), lr=0.01), epochs=300, loader=small)",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "# ⑨ 규제: Dropout과 weight decay",
   "s": 9
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 9
  },
  {
   "t": "reg = make_net(hidden=128, depth=3, dropout=0.3)",
   "s": 9
  },
  {
   "t": "regd = train(reg, torch.optim.Adam(reg.parameters(), lr=0.01, weight_decay=1e-3), epochs=300, loader=small)",
   "s": 9
  },
  {
   "t": "",
   "s": 9
  },
  {
   "t": "# ⑩ 조기 종료: valid 손실이 가장 낮았던 때로 돌아가기",
   "s": 10
  },
  {
   "t": "def train_early(model, optimizer, patience=20, max_epochs=300, loader=small):",
   "s": 10
  },
  {
   "t": "    best, best_state, wait, history = float(\"inf\"), None, 0, []",
   "s": 10
  },
  {
   "t": "    for epoch in range(1, max_epochs + 1):",
   "s": 10
  },
  {
   "t": "        h = {**train(model, optimizer, 1, loader)[0], \"epoch\": epoch}",
   "s": 10
  },
  {
   "t": "        history.append(h)",
   "s": 10
  },
  {
   "t": "        if h[\"valid_loss\"] < best:",
   "s": 10
  },
  {
   "t": "            best, best_state, wait = h[\"valid_loss\"], copy.deepcopy(model.state_dict()), 0",
   "s": 10
  },
  {
   "t": "        else:",
   "s": 10
  },
  {
   "t": "            wait += 1",
   "s": 10
  },
  {
   "t": "            if wait >= patience:  # patience 번 연속 나아지지 않으면 멈춤",
   "s": 10
  },
  {
   "t": "                break",
   "s": 10
  },
  {
   "t": "    model.load_state_dict(best_state)",
   "s": 10
  },
  {
   "t": "    return history",
   "s": 10
  },
  {
   "t": "",
   "s": 10
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 10
  },
  {
   "t": "early_net = make_net(hidden=128, depth=3)",
   "s": 10
  },
  {
   "t": "early = train_early(early_net, torch.optim.Adam(early_net.parameters(), lr=0.01))",
   "s": 10
  },
  {
   "t": "",
   "s": 10
  },
  {
   "t": "# ⑪ 마지막 한 번: test",
   "s": 11
  },
  {
   "t": "X_test, y_test = tensors(\"test\")",
   "s": 11
  },
  {
   "t": "test_acc = {\"기본 신경망 (⑤)\": accuracy(net, X_test, y_test), \"과적합 (⑧)\": accuracy(big, X_test, y_test),",
   "s": 11
  },
  {
   "t": "            \"규제 (⑨)\": accuracy(reg, X_test, y_test), \"조기 종료 (⑩)\": accuracy(early_net, X_test, y_test)}",
   "s": 11
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
  6,
  11
 ],
 "outs": {
  "why": {
   "title": "⑧에서 만든 큰 신경망 (학습 데이터 40개로 300 epoch)",
   "cols": [
    "epoch",
    "train 손실",
    "valid 손실",
    "valid 정확도"
   ],
   "rows": [
    [
     1,
     "0.404",
     "0.442",
     "0.762"
    ],
    [
     50,
     "0.000",
     "0.739",
     "0.913"
    ],
    [
     100,
     "0.000",
     "0.776",
     "0.925"
    ],
    [
     150,
     "0.000",
     "0.797",
     "0.925"
    ],
    [
     200,
     "0.000",
     "0.813",
     "0.925"
    ],
    [
     250,
     "0.000",
     "0.828",
     "0.925"
    ],
    [
     300,
     "0.000",
     "0.841",
     "0.913"
    ]
   ],
   "notes": [
    "train 손실은 0이 됐는데 valid 손실은 오히려 계속 커져요. 학습 데이터를 <b>외워 버린</b> 거예요 (과적합). 외운 모델은 새 데이터에서 틀린 답을 아주 확신해요.",
    "학습이 잘되는지는 손실 하나로 알 수 없어요. 학습률, Optimizer, 모델 크기, 데이터 양이 모두 영향을 줘요. 이 쪽에서는 그걸 하나씩 실험해요."
   ]
  },
  "s6": {
   "title": "⑥ 같은 신경망, SGD, 학습률만 다르게 (100 epoch)",
   "cols": [
    "학습률",
    "train 손실",
    "valid 정확도"
   ],
   "rows": [
    [
     0.001,
     "0.632",
     "0.775"
    ],
    [
     0.1,
     "0.249",
     "0.875"
    ],
    [
     10.0,
     "0.479",
     "0.762"
    ]
   ],
   "notes": [
    "0.001은 너무 조금씩 움직여서 100바퀴가 지나도 거의 그대로예요.",
    "10은 너무 크게 움직여서 최저점을 지나쳐 버려요. 0.1이 이 셋 중 가장 나아요. 적당한 학습률은 문제마다 달라서 몇 개를 시험해 봐요."
   ]
  },
  "s7": {
   "title": "⑦ 학습률 0.01로 SGD와 Adam (60 epoch)",
   "cols": [
    "Optimizer",
    "10 epoch valid 정확도",
    "60 epoch valid 정확도",
    "60 epoch train 손실"
   ],
   "rows": [
    [
     "sgd",
     "0.762",
     "0.750",
     "0.442"
    ],
    [
     "adam",
     "0.850",
     "0.975",
     "0.107"
    ]
   ],
   "notes": [
    "같은 학습률이어도 Adam이 훨씬 빨리 내려가요. Adam은 가중치마다 지금까지의 기울기를 보고 옮기는 크기를 알아서 조절해요. 그래서 처음 시작할 때 무난한 선택이에요."
   ]
  },
  "s8": {
   "title": "⑧ 과적합: 데이터 40개, 128칸 3층",
   "lines": [
    "가중치 수 → 33,537개",
    [
     "300 epoch 뒤 train 손실 0.000001, valid 손실 0.841",
     1,
     "bad"
    ]
   ],
   "notes": [
    "데이터 40개에 가중치 3만 개가 넘어요. 외우기에 충분하고도 남아요."
   ]
  },
  "s9": {
   "title": "⑨ 같은 조건에 Dropout 0.3 + weight decay",
   "cols": [
    "epoch",
    "train 손실",
    "valid 손실",
    "valid 정확도"
   ],
   "rows": [
    [
     1,
     "0.358",
     "0.490",
     "0.762"
    ],
    [
     50,
     "0.011",
     "0.455",
     "0.913"
    ],
    [
     100,
     "0.002",
     "0.399",
     "0.925"
    ],
    [
     150,
     "0.001",
     "0.662",
     "0.925"
    ],
    [
     200,
     "0.001",
     "0.418",
     "0.925"
    ],
    [
     250,
     "0.002",
     "0.459",
     "0.938"
    ],
    [
     300,
     "0.001",
     "0.439",
     "0.938"
    ]
   ],
   "notes": [
    "300 epoch 뒤 valid 손실이 0.841 → 0.439로 줄었어요. 틀린 답에 대한 지나친 확신이 줄었다는 뜻이에요.",
    "정확도 차이는 크지 않아요. 규제는 \"맞히는 개수\"보다 \"얼마나 믿을 만한 확률을 내나\"에 먼저 효과가 보이기도 해요."
   ]
  },
  "s10": {
   "title": "⑩ 조기 종료 (patience 20)",
   "lines": [
    [
     "멈춘 epoch → 24",
     1
    ],
    [
     "가장 좋았던 epoch → 4 (valid 손실 0.321, 정확도 0.825)",
     2
    ]
   ],
   "notes": [
    "valid 손실이 20번 연속 나아지지 않아서 멈췄어요. 300번을 다 돌지 않아 시간도 아껴요.",
    "마지막이 아니라 가장 좋았던 때의 가중치로 되돌렸어요. 데이터가 40개뿐이라 아주 이른 때가 가장 좋았어요."
   ]
  },
  "s11": {
   "title": "⑪ test 80개 정확도 (처음 사용)",
   "cols": [
    "모델",
    "test 정확도"
   ],
   "rows": [
    [
     "기본 신경망 (⑤)",
     "0.950"
    ],
    [
     "과적합 (⑧)",
     "0.938"
    ],
    [
     "규제 (⑨)",
     "0.900"
    ],
    [
     "조기 종료 (⑩)",
     "0.850"
    ]
   ],
   "notes": [
    "데이터 240개를 다 쓴 기본 신경망이 가장 좋아요. 과적합을 막는 기법도 중요하지만, <b>데이터가 많은 것</b>을 이기기는 어려워요.",
    "⑧~⑩은 일부러 40개만 써서 과적합을 만든 실험이에요. 규제와 조기 종료는 손실(확신의 정도)을 안정시켰지만, 이 작은 test에서 정확도까지 올리진 못했어요."
   ]
  },
  "f_opt": {
   "title": "epoch마다 train 손실",
   "images": [
    [
     "img/dl/optimizer_curves.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽: 학습률 0.001은 선이 거의 수평이고, 10은 손실이 내려가지 못하고 높은 곳에서 출렁여요. 0.1이 가장 빨리 내려가요.",
    "오른쪽: 같은 학습률 0.01인데 고치는 방법만 바꿨어요. 모멘텀을 더하면 SGD보다 훨씬 빨리 내려가고, Adam은 그보다 더 빨라요."
   ],
   "chart": true
  },
  "t7": {
   "title": "SGD에 모멘텀을 더하면 (학습률 0.01, 60 epoch)",
   "cols": [
    "Optimizer",
    "10 epoch valid 정확도",
    "60 epoch valid 정확도",
    "60 epoch train 손실"
   ],
   "rows": [
    [
     "SGD",
     "0.762",
     "0.750",
     "0.442"
    ],
    [
     "SGD + 모멘텀 0.9",
     "0.738",
     "0.812",
     "0.288"
    ],
    [
     "Adam",
     "0.850",
     "0.975",
     "0.107"
    ]
   ],
   "notes": [
    "모멘텀은 SGD와 Adam 사이예요. \"지난 걸음 방향을 90% 이어 가기\" 하나만 더했는데 train 손실이 0.442에서 0.288로 줄었어요.",
    "Adam은 모멘텀(기울기의 평균)에 더해 가중치마다 기울기 크기의 평균으로 걸음을 나눠서, 이 문제처럼 학습률이 작게 잡혀 있어도 알아서 성큼 가요."
   ]
  },
  "f_over": {
   "title": "⑧⑨⑩의 손실 곡선 (train 40개)",
   "images": [
    [
     "img/dl/overfit_curves.png",
     "점선: train, 실선: valid"
    ]
   ],
   "notes": [
    "분홍 점선(과적합 train)은 0으로 붙는데, 분홍 실선(valid)은 어느 순간부터 계속 올라가요. 둘이 벌어지기 시작한 곳부터가 외우는 구간이에요.",
    "보라(규제)도 train 손실은 0 근처까지 내려가지만, Dropout이 칸을 무작위로 꺼서 가끔 튀어요. valid는 크게 출렁이면서도 대부분 분홍 실선보다 아래에 있어요. 끝까지 계속 올라가지는 않아요.",
    "민트 점이 조기 종료가 고른 가장 좋던 때예요(epoch 4). 데이터가 40개뿐이라 외우기 시작하는 게 아주 빨라요."
   ],
   "chart": true
  },
  "f_wh": {
   "title": "⑧과 ⑨가 배운 가중치의 분포",
   "images": [
    [
     "img/dl/weight_hist.png",
     ""
    ]
   ],
   "notes": [
    "weight decay는 걸음마다 가중치를 조금씩 0 쪽으로 당겨요. 그래서 큰 가중치가 줄고 0 근처에 많이 모였어요. 가중치가 작으면 입력이 조금 바뀌어도 출력이 크게 흔들리지 않아, 덜 외운 매끈한 함수가 돼요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   6,
   "학습률",
   "0.001, 0.1, 10",
   "너무 작거나 크면"
  ],
  [
   7,
   "Optimizer",
   "SGD vs Adam",
   "Adam이 빠름"
  ],
  [
   8,
   "과적합",
   "40개로 외우기",
   "train 0, valid ↑"
  ],
  [
   9,
   "규제",
   "Dropout, decay",
   "확신 줄이기"
  ],
  [
   10,
   "조기 종료",
   "가장 좋던 때로",
   "patience"
  ],
  [
   11,
   "test",
   "한 번",
   "데이터가 이김"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, CPU 학습 (GPU: NVIDIA GeForce RTX 5080 Laptop GPU)"
};
