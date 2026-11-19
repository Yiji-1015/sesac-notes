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
  }
 ],
 "names": {
  "1": "그림을 Tensor로",
  "2": "픽셀을 그냥 펼치면 (MLP)",
  "3": "합성곱 한 번: 작은 무늬 찾기",
  "4": "CNN 만들기",
  "5": "학습하고 비교하기",
  "6": "선의 위치가 바뀌면"
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
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, CPU (seed 0)"
};
