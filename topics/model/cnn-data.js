window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
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
   "t": "from torchvision import datasets, transforms",
   "s": 1
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 3
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\"",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 그림을 Tensor로",
   "s": 1
  },
  {
   "t": "to_tensor = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])  # 흑백 한 장, 0~1",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def load(split):",
   "s": 1
  },
  {
   "t": "    ds = datasets.ImageFolder(DATA / split, transform=to_tensor)  # 폴더 이름이 곧 정답",
   "s": 1
  },
  {
   "t": "    X = torch.stack([x for x, _ in ds])",
   "s": 1
  },
  {
   "t": "    y = torch.tensor([t for _, t in ds])",
   "s": 1
  },
  {
   "t": "    return X, y, ds.classes",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "X_train, y_train, classes = load(\"train\")",
   "s": 1
  },
  {
   "t": "X_valid, y_valid, _ = load(\"valid\")",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 픽셀을 그냥 펼치면 (MLP)",
   "s": 2
  },
  {
   "t": "def make_mlp():",
   "s": 2
  },
  {
   "t": "    return nn.Sequential(nn.Flatten(), nn.Linear(64 * 64, 32), nn.ReLU(), nn.Linear(32, 2))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 합성곱 한 번: 작은 무늬 찾기",
   "s": 3
  },
  {
   "t": "horizontal_kernel = torch.tensor([[-1., -1., -1.], [2., 2., 2.], [-1., -1., -1.]])  # 가로 선에 크게 반응",
   "s": 3
  },
  {
   "t": "vertical_kernel = horizontal_kernel.T  # 세로 선에 크게 반응",
   "s": 3
  },
  {
   "t": "kernels = torch.stack([horizontal_kernel, vertical_kernel])[:, None]  # (2, 1, 3, 3)",
   "s": 3
  },
  {
   "t": "sample = torch.stack([X_train[y_train == 0][0], X_train[y_train == 1][0]])  # 가로 한 장, 세로 한 장",
   "s": 3
  },
  {
   "t": "maps = F.relu(F.conv2d(sample, kernels, padding=1))  # (2장, 2무늬, 64, 64)",
   "s": 3
  },
  {
   "t": "response = maps.mean(dim=(2, 3))  # 그림마다 무늬별 평균 반응",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ CNN 만들기",
   "s": 4
  },
  {
   "t": "def make_cnn():",
   "s": 4
  },
  {
   "t": "    return nn.Sequential(",
   "s": 4
  },
  {
   "t": "        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64 → 32",
   "s": 4
  },
  {
   "t": "        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),",
   "s": 4
  },
  {
   "t": "        nn.AdaptiveAvgPool2d(1), nn.Flatten(),  # 어디에 있든 평균 → 16개 값",
   "s": 4
  },
  {
   "t": "        nn.Linear(16, 2))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def count(model):",
   "s": 4
  },
  {
   "t": "    return sum(p.numel() for p in model.parameters())",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 학습하고 비교하기",
   "s": 5
  },
  {
   "t": "def fit(model, epochs=60):",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "    opt = torch.optim.Adam(model.parameters(), lr=0.01)",
   "s": 5
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 5
  },
  {
   "t": "        loss = F.cross_entropy(model(X_train), y_train)",
   "s": 5
  },
  {
   "t": "        opt.zero_grad()",
   "s": 5
  },
  {
   "t": "        loss.backward()",
   "s": 5
  },
  {
   "t": "        opt.step()",
   "s": 5
  },
  {
   "t": "    return model",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def acc(model, X, y):",
   "s": 5
  },
  {
   "t": "    with torch.no_grad():",
   "s": 5
  },
  {
   "t": "        return (model(X).argmax(1) == y).float().mean().item()",
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
   "t": "mlp = fit(make_mlp())",
   "s": 5
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "cnn = fit(make_cnn())",
   "s": 5
  },
  {
   "t": "scores = {\"MLP\": acc(mlp, X_valid, y_valid), \"CNN\": acc(cnn, X_valid, y_valid)}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 선의 위치가 바뀌면",
   "s": 6
  },
  {
   "t": "shifted = {k: torch.roll(X_valid, shifts=(k, k), dims=(2, 3)) for k in (0, 8, 16)}  # k칸씩 아래, 오른쪽으로",
   "s": 6
  },
  {
   "t": "shift_scores = {k: {\"MLP\": acc(mlp, Xs, y_valid), \"CNN\": acc(cnn, Xs, y_valid)} for k, Xs in shifted.items()}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 깊게 쌓기: 잔차 연결과 BatchNorm",
   "s": 7
  },
  {
   "t": "class Block(nn.Module):  # 3×3 합성곱 하나짜리 블록",
   "s": 7
  },
  {
   "t": "    def __init__(self, ch, residual, bn):",
   "s": 7
  },
  {
   "t": "        super().__init__()",
   "s": 7
  },
  {
   "t": "        self.conv = nn.Conv2d(ch, ch, 3, padding=1)",
   "s": 7
  },
  {
   "t": "        self.bn = nn.BatchNorm2d(ch) if bn else nn.Identity()  # 끄면 아무것도 안 하는 층",
   "s": 7
  },
  {
   "t": "        self.residual = residual",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "    def forward(self, x):",
   "s": 7
  },
  {
   "t": "        h = self.bn(self.conv(x))",
   "s": 7
  },
  {
   "t": "        return F.relu(x + h) if self.residual else F.relu(h)  # 잔차 연결: 입력을 그대로 더하기",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "def make_deep(depth, residual, bn):",
   "s": 7
  },
  {
   "t": "    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),",
   "s": 7
  },
  {
   "t": "                         *[Block(8, residual, bn) for _ in range(depth)],",
   "s": 7
  },
  {
   "t": "                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "DEPTH = 16",
   "s": 7
  },
  {
   "t": "deep_runs = {}",
   "s": 7
  },
  {
   "t": "for name, (residual, bn) in {\"그냥 쌓기\": (False, False), \"잔차 연결\": (True, False), \"잔차 + BatchNorm\": (True, True)}.items():",
   "s": 7
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 7
  },
  {
   "t": "    m = make_deep(DEPTH, residual, bn)",
   "s": 7
  },
  {
   "t": "    F.cross_entropy(m(X_train), y_train).backward()",
   "s": 7
  },
  {
   "t": "    grad0 = m[0].weight.grad.norm().item()  # 학습 전, 맨 앞 층까지 내려온 기울기 크기",
   "s": 7
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 7
  },
  {
   "t": "    m = fit(make_deep(DEPTH, residual, bn)).eval()",
   "s": 7
  },
  {
   "t": "    with torch.no_grad():",
   "s": 7
  },
  {
   "t": "        deep_runs[name] = {\"grad0\": grad0, \"train_loss\": F.cross_entropy(m(X_train), y_train).item(),",
   "s": 7
  },
  {
   "t": "                           \"valid_acc\": acc(m, X_valid, y_valid)}",
   "s": 7
  }
 ],
 "names": {
  "1": "그림을 Tensor로",
  "2": "픽셀을 그냥 펼치면 (MLP)",
  "3": "합성곱 한 번: 작은 무늬 찾기",
  "4": "CNN 만들기",
  "5": "학습하고 비교하기",
  "6": "선의 위치가 바뀌면",
  "7": "깊게 쌓기: 잔차 연결과 BatchNorm"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "data": {
   "title": "train 그림 예시 (64×64)",
   "images": [
    [
     "img/cnn/h0.png",
     "horizontal"
    ],
    [
     "img/cnn/h1.png",
     "horizontal"
    ],
    [
     "img/cnn/h2.png",
     "horizontal"
    ],
    [
     "img/cnn/v0.png",
     "vertical"
    ],
    [
     "img/cnn/v1.png",
     "vertical"
    ],
    [
     "img/cnn/v2.png",
     "vertical"
    ]
   ],
   "notes": [
    "가로줄 그림과 세로줄 그림이에요. 줄의 위치와 두께가 그림마다 달라요. train 60장, valid 20장, test 20장이에요."
   ]
  },
  "why": {
   "title": "두 모델 비교 (valid 20장)",
   "cols": [
    "모델",
    "가중치 수",
    "정확도",
    "선을 8칸 옮기면"
   ],
   "rows": [
    [
     "픽셀을 펼친 MLP (②)",
     "131,170",
     "0.90",
     "<span class=\"bad\">0.15</span>"
    ],
    [
     "CNN (④)",
     "1,282",
     "<b>1.00</b>",
     "<b>1.00</b>"
    ]
   ],
   "notes": [
    "CNN은 가중치가 <b>100분의 1</b>인데도 더 잘 맞히고, 선의 위치가 바뀌어도 그대로 맞혀요. MLP는 \"몇 번째 픽셀이 밝은가\"를 외워서, 위치가 바뀌면 무너져요."
   ]
  },
  "s1": {
   "title": "① load(\"train\")",
   "lines": [
    [
     "X_train.shape → (60, 1, 64, 64)",
     1
    ],
    "classes → ['horizontal', 'vertical']",
    "픽셀 값 범위 → 0.1 ~ 0.9"
   ],
   "notes": [
    "(그림 수, 채널, 세로, 가로) 순서예요. 흑백이라 채널이 1이에요. 컬러면 3(R, G, B)이에요."
   ]
  },
  "s2": {
   "title": "② make_mlp()",
   "lines": [
    "Sequential(   (0): Flatten(start_dim=1, end_dim=-1)   (1): Linear(in_features=4096, out_features=32, bias=True)   (2): ReLU()   (3): Linear(in_features=32, out_features=2",
    [
     "가중치 수 → 131,170",
     1
    ]
   ],
   "notes": [
    "64×64 = 4,096개 픽셀 하나하나에 따로 가중치가 붙어요. 그림의 \"모양\"이 아니라 \"자리\"를 배워요."
   ]
  },
  "s3": {
   "title": "③ 무늬 두 개를 그림 두 장에 대 보면 (밝을수록 강한 반응)",
   "images": [
    [
     "img/cnn/s_h.png",
     "가로줄 그림"
    ],
    [
     "img/cnn/m_hh.png",
     "가로 무늬 반응"
    ],
    [
     "img/cnn/m_hv.png",
     "세로 무늬 반응"
    ],
    [
     "img/cnn/s_v.png",
     "세로줄 그림"
    ],
    [
     "img/cnn/m_vh.png",
     "가로 무늬 반응"
    ],
    [
     "img/cnn/m_vv.png",
     "세로 무늬 반응"
    ]
   ],
   "notes": [
    "평균 반응: 가로줄 그림 → 가로 무늬 0.167, 세로 무늬 0.038 / 세로줄 그림 → 가로 무늬 0.044, 세로 무늬 0.209",
    "3×3 무늬 하나를 그림 전체에 밀며 대 보니, 맞는 방향의 선이 있는 자리만 밝게 켜졌어요. <b>어디에 있든</b> 같은 무늬로 찾아요. 이게 합성곱이에요. CNN은 이런 무늬를 사람이 정하지 않고 학습으로 찾아요."
   ]
  },
  "s4": {
   "title": "④ make_cnn()",
   "lines": [
    "Sequential(   (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   (1): ReLU()   (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)   (3): Conv2d(8, 16, kernel_size=(3, 3), ",
    [
     "가중치 수 → 1,282",
     1
    ]
   ],
   "notes": [
    "3×3 무늬 8개, 그 위에 다시 무늬 16개만 배우면 돼요. 무늬 하나를 그림 모든 자리에 같이 쓰니까(가중치 공유) 가중치가 아주 적어요."
   ]
  },
  "s5": {
   "title": "⑤ 같은 조건(Adam, 60번)으로 학습, valid 정확도",
   "lines": [
    "MLP → 0.90",
    [
     "CNN → 1.00",
     1
    ]
   ],
   "notes": [
    "두 모델 다 꽤 맞히지만, 차이는 ⑥에서 크게 드러나요."
   ]
  },
  "s6": {
   "title": "⑥ valid 그림을 k칸씩 옮긴 뒤 정확도",
   "cols": [
    "옮긴 칸 수",
    "MLP",
    "CNN"
   ],
   "rows": [
    [
     0,
     "0.90",
     "<b>1.00</b>"
    ],
    [
     8,
     "0.15",
     "<b>1.00</b>"
    ],
    [
     16,
     "0.65",
     "<b>1.00</b>"
    ]
   ],
   "notes": [
    "CNN은 무늬를 어디서 찾든 같고, 마지막에 <code>AdaptiveAvgPool2d</code>로 위치를 평균 내버려서 옮겨도 그대로예요. MLP는 학습 때 본 자리가 아니면 엉뚱하게 답해요."
   ]
  },
  "s6img": {
   "title": "valid 첫 그림을 옮긴 모습",
   "images": [
    [
     "img/cnn/shift0.png",
     "0칸"
    ],
    [
     "img/cnn/shift8.png",
     "8칸"
    ],
    [
     "img/cnn/shift16.png",
     "16칸"
    ]
   ],
   "notes": []
  },
  "t3": {
   "title": "③의 가로 무늬를 설정만 바꿔 대 보면 (입력 64×64)",
   "cols": [
    "설정",
    "출력 크기",
    "크기 계산",
    "무늬가 덮는 넓이"
   ],
   "rows": [
    [
     "padding=0",
     "(62, 62)",
     "(64 − 3 + 2×0) ÷ 1 + 1 = 62",
     "3×3"
    ],
    [
     "<b>padding=1</b> (③)",
     "(64, 64)",
     "(64 − 3 + 2×1) ÷ 1 + 1 = 64",
     "3×3"
    ],
    [
     "padding=1, stride=2",
     "(32, 32)",
     "(64 − 3 + 2×1) ÷ 2 + 1 = 32",
     "3×3"
    ],
    [
     "padding=2, dilation=2",
     "(64, 64)",
     "(64 − 5 + 2×2) ÷ 1 + 1 = 64",
     "5×5"
    ]
   ],
   "notes": [
    "둘레에 0을 두르지 않으면(padding=0) 무늬가 그림 밖으로 못 나가서 가장자리 한 줄씩 잃어요(62×62).",
    "stride=2는 두 칸씩 건너뛰며 대서 크기가 절반이 돼요. 풀링 없이 크기를 줄이는 방법이에요.",
    "dilation=2는 무늬 칸 사이를 한 칸씩 띄워서, 가중치는 9개 그대로인데 5×5 넓이를 봐요. 넓은 맥락이 필요할 때 써요."
   ]
  },
  "f_k": {
   "title": "④의 CNN이 학습으로 찾은 첫 층 무늬 8개와 그 반응",
   "images": [
    [
     "img/cnn/learned_kernels.png",
     "위: 무늬(주황 +, 보라 −), 가운데와 아래: 무늬마다 반응 지도"
    ]
   ],
   "notes": [
    "③에서는 가로, 세로 무늬를 사람이 정했어요. ④의 CNN은 무늬 8개를 정답만 보고 스스로 찾았어요.",
    "무늬마다 반응하는 그림이 달라요. 어떤 무늬는 가로줄 그림의 선 자리에서만, 어떤 무늬는 세로줄 그림에서만 밝게 켜져요. 둘째 층은 이 8장을 입력으로 받아 다시 무늬를 찾아요."
   ],
   "chart": true
  },
  "s7": {
   "title": "⑦ 3×3 합성곱 블록 16개를 쌓으면 (Adam, 60번)",
   "cols": [
    "쌓는 방식",
    "학습 전 맨 앞 층 기울기",
    "학습 뒤 train 손실",
    "valid 정확도"
   ],
   "rows": [
    [
     "그냥 쌓기",
     "1.4e-11",
     "0.693",
     "0.50"
    ],
    [
     "잔차 연결",
     "3.5e-02",
     "0.329",
     "1.00"
    ],
    [
     "잔차 + BatchNorm",
     "6.2e-01",
     "0.000",
     "1.00"
    ]
   ],
   "notes": [
    "그냥 쌓으면 맨 앞 층까지 내려오는 기울기가 1e-11로 사실상 0이에요. 앞쪽 층이 전혀 바뀌지 않아서 train 손실이 log 2 = 0.693에 그대로 머물러요. 어떤 그림이든 반반으로 찍는다는 뜻이에요. 층이 2개뿐인 ④는 문제없이 배웠는데, 깊게 쌓자마자 배우지 못해요.",
    "잔차 연결 하나만 넣었는데 기울기가 살아나서 valid가 맞기 시작해요. 블록마다 입력을 그대로 더해 주는 길이 있어서, 기울기가 그 길로 곧장 앞쪽까지 내려와요.",
    "BatchNorm까지 넣으면 train 손실이 0에 가까워져요. 층마다 값의 크기를 다시 맞춰 주니 더 깊이 쌓아도 안정돼요(🔁)."
   ]
  },
  "f_g": {
   "title": "층마다 기울기 크기 (학습 전, 블록 16개)",
   "images": [
    [
     "img/cnn/residual_grads.png",
     ""
    ]
   ],
   "notes": [
    "분홍(그냥 쌓기)은 뒤에서 앞으로 갈수록 층마다 일정 비율로 줄어서, 맨 앞에서는 10⁻¹¹ 근처까지 떨어져요. <a href=\"dl-tensor-training.html\">딥러닝 기초</a> ④의 기울기 소실이 실제 CNN에서 일어난 거예요.",
    "민트(잔차)와 보라(잔차 + BatchNorm)는 앞쪽까지 크기가 거의 그대로예요."
   ],
   "chart": true
  },
  "t7": {
   "title": "블록을 30개로 늘리면",
   "cols": [
    "쌓는 방식",
    "학습 전 맨 앞 층 기울기",
    "학습 뒤 train 손실",
    "valid 정확도"
   ],
   "rows": [
    [
     "그냥 쌓기",
     "4.0e-18",
     "0.693",
     "0.50"
    ],
    [
     "잔차 연결",
     "8.4e+00",
     "0.661",
     "0.55"
    ],
    [
     "잔차 + BatchNorm",
     "1.2e+00",
     "0.000",
     "1.00"
    ]
   ],
   "notes": [
    "그냥 쌓기는 기울기가 4e-18로 더 작아졌어요.",
    "잔차 연결만 있으면 이번엔 반대로 기울기가 8.4로 커져서 학습이 흔들리고 손실이 거의 줄지 않아요. 블록마다 입력에 무언가를 더하니, 30번 더하면 값이 계속 불어나요.",
    "BatchNorm이 층마다 크기를 되돌려 주면 30층도 문제없어요. 그래서 ResNet은 잔차 연결과 BatchNorm을 늘 같이 써요."
   ]
  }
 },
 "flow": [
  [
   1,
   "Tensor",
   "(N, 1, 64, 64)",
   "ImageFolder"
  ],
  [
   2,
   "MLP",
   "픽셀 펼치기",
   "13만 가중치"
  ],
  [
   3,
   "합성곱",
   "3×3 무늬",
   "어디서든 찾기"
  ],
  [
   4,
   "CNN",
   "Conv, Pool",
   "1,282 가중치"
  ],
  [
   5,
   "학습",
   "같은 조건",
   "비교"
  ],
  [
   6,
   "옮기기",
   "위치 바꾸기",
   "CNN만 버팀"
  ],
  [
   7,
   "깊게",
   "잔차 연결",
   "BatchNorm"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, CPU (seed 0)"
};
