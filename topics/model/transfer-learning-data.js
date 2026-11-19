window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "from collections import Counter",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 3
  },
  {
   "t": "import copy",
   "s": 5
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
   "s": 2
  },
  {
   "t": "from torchvision import datasets, transforms",
   "s": 1
  },
  {
   "t": "from sklearn.metrics import f1_score",
   "s": 2
  },
  {
   "t": "from sklearn.metrics import classification_report, confusion_matrix",
   "s": 6
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
   "t": "# ① 데이터 읽고 개수 세기",
   "s": 1
  },
  {
   "t": "prep = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])  # 빌려 올 모델이 배운 입력 모양",
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
   "t": "    ds = datasets.ImageFolder(DATA / split, transform=prep)",
   "s": 1
  },
  {
   "t": "    return torch.stack([x for x, _ in ds]), torch.tensor(ds.targets), ds.classes",
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
   "t": "counts = {classes[k]: v for k, v in sorted(Counter(y_train.tolist()).items())}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 처음부터 학습하기",
   "s": 2
  },
  {
   "t": "def make_backbone():",
   "s": 2
  },
  {
   "t": "    return nn.Sequential(",
   "s": 2
  },
  {
   "t": "        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),",
   "s": 2
  },
  {
   "t": "        nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),",
   "s": 2
  },
  {
   "t": "        nn.AdaptiveAvgPool2d(4), nn.Flatten())  # 특징 16 × 4 × 4 = 256개",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def make_model():",
   "s": 2
  },
  {
   "t": "    return nn.Sequential(make_backbone(), nn.Linear(256, 3))  # [0] 특징 뽑기(Backbone), [1] 분류(Head)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def fit(model, epochs=30, lr=0.01, weight=None):",
   "s": 2
  },
  {
   "t": "    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)",
   "s": 2
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 2
  },
  {
   "t": "        model.train()",
   "s": 2
  },
  {
   "t": "        for i in torch.randperm(len(X_train), generator=torch.Generator().manual_seed(0)).split(32):",
   "s": 2
  },
  {
   "t": "            loss = F.cross_entropy(model(X_train[i]), y_train[i], weight=weight)",
   "s": 2
  },
  {
   "t": "            opt.zero_grad()",
   "s": 2
  },
  {
   "t": "            loss.backward()",
   "s": 2
  },
  {
   "t": "            opt.step()",
   "s": 2
  },
  {
   "t": "    return model",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def predict(model, X):",
   "s": 2
  },
  {
   "t": "    model.eval()",
   "s": 2
  },
  {
   "t": "    with torch.no_grad():",
   "s": 2
  },
  {
   "t": "        return model(X).argmax(1)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def report(model):",
   "s": 2
  },
  {
   "t": "    p = predict(model, X_valid)",
   "s": 2
  },
  {
   "t": "    return {\"accuracy\": (p == y_valid).float().mean().item(), \"macro_f1\": f1_score(y_valid, p, average=\"macro\")}",
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
   "t": "scratch = fit(make_model(), epochs=5)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 미리 배운 Backbone 빌려 오기",
   "s": 3
  },
  {
   "t": "source = np.load(DATA / \"source_backbone.npz\")",
   "s": 3
  },
  {
   "t": "pretrained = make_backbone()",
   "s": 3
  },
  {
   "t": "pretrained.load_state_dict({k: torch.tensor(source[k]) for k in source.files})",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ Backbone은 고정하고 Head만 학습하기",
   "s": 4
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 4
  },
  {
   "t": "frozen = make_model()",
   "s": 4
  },
  {
   "t": "frozen[0].load_state_dict(pretrained.state_dict())",
   "s": 4
  },
  {
   "t": "for p in frozen[0].parameters():",
   "s": 4
  },
  {
   "t": "    p.requires_grad = False  # Backbone 가중치는 바꾸지 않음",
   "s": 4
  },
  {
   "t": "trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)",
   "s": 4
  },
  {
   "t": "fit(frozen, epochs=5)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 전체를 조금씩 고치기 (미세조정)",
   "s": 5
  },
  {
   "t": "finetuned = copy.deepcopy(frozen)  # ④의 모델은 그대로 두고 복사본을",
   "s": 5
  },
  {
   "t": "for p in finetuned[0].parameters():",
   "s": 5
  },
  {
   "t": "    p.requires_grad = True",
   "s": 5
  },
  {
   "t": "fit(finetuned, epochs=5, lr=0.001)  # 이미 좋은 가중치라 작은 학습률로",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 종류별로 보기",
   "s": 6
  },
  {
   "t": "pred = predict(finetuned, X_valid)",
   "s": 6
  },
  {
   "t": "per_class = classification_report(y_valid, pred, target_names=classes, output_dict=True, zero_division=0)",
   "s": 6
  },
  {
   "t": "cm = confusion_matrix(y_valid, pred)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 적은 종류에 무게 주기",
   "s": 7
  },
  {
   "t": "weight = torch.tensor([len(y_train) / (3 * n) for n in counts.values()], dtype=torch.float32)  # 적을수록 크게",
   "s": 7
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 7
  },
  {
   "t": "weighted = make_model()",
   "s": 7
  },
  {
   "t": "weighted[0].load_state_dict(pretrained.state_dict())",
   "s": 7
  },
  {
   "t": "fit(weighted, epochs=5, lr=0.001, weight=weight)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 마지막 한 번: test",
   "s": 8
  },
  {
   "t": "X_test, y_test, _ = load(\"test\")",
   "s": 8
  },
  {
   "t": "test_scores = {name: f1_score(y_test, predict(m, X_test), average=\"macro\")",
   "s": 8
  },
  {
   "t": "               for name, m in {\"처음부터\": scratch, \"Head만\": frozen, \"미세조정\": finetuned, \"미세조정 + 무게\": weighted}.items()}",
   "s": 8
  }
 ],
 "names": {
  "1": "데이터 읽고 개수 세기",
  "2": "처음부터 학습하기",
  "3": "미리 배운 Backbone 빌려 오기",
  "4": "Backbone은 고정하고 Head만 학습하기",
  "5": "전체를 조금씩 고치기 (미세조정)",
  "6": "종류별로 보기",
  "7": "적은 종류에 무게 주기",
  "8": "마지막 한 번: test"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "data": {
   "title": "train 그림 예시 (32×32로 맞춤)",
   "images": [
    [
     "img/transfer/ex00.png",
     "circle"
    ],
    [
     "img/transfer/ex01.png",
     "circle"
    ],
    [
     "img/transfer/ex10.png",
     "square"
    ],
    [
     "img/transfer/ex11.png",
     "square"
    ],
    [
     "img/transfer/ex20.png",
     "triangle"
    ],
    [
     "img/transfer/ex21.png",
     "triangle"
    ]
   ],
   "notes": [
    "train 개수: circle 120장, square 40장, triangle 20장. 원이 세모의 6배예요. valid는 종류마다 30장씩이에요."
   ]
  },
  "why": {
   "title": "같은 구조, 같은 학습 횟수(5번)로 비교 (valid 90장)",
   "cols": [
    "방법",
    "정확도",
    "macro F1"
   ],
   "rows": [
    [
     "처음부터 학습 (②)",
     "0.656",
     "<span class=\"bad\">0.649</span>"
    ],
    [
     "미리 배운 Backbone + Head만 학습 (④)",
     "1.000",
     "<b>1.000</b>"
    ]
   ],
   "notes": [
    "그림 180장, 5번 학습으로는 처음부터 배우기 어려워요. 다른 도형 그림으로 <b>이미 배운 특징 뽑기 부분</b>을 빌려 오면, 분류하는 마지막 층만 배워도 거의 다 맞혀요."
   ]
  },
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    [
     "X_train.shape → (180, 3, 32, 32)",
     1
    ],
    "classes → ['circle', 'square', 'triangle']",
    [
     "counts → {'circle': 120, 'square': 40, 'triangle': 20}",
     2
    ]
   ],
   "notes": [
    "40×40 그림을 32×32로 줄였어요. 빌려 올 모델이 그 크기로 배웠기 때문이에요. 입력 모양은 빌려 온 모델과 맞춰야 해요.",
    "종류마다 개수가 크게 달라요 (불균형). ⑥, ⑦에서 다시 봐요."
   ]
  },
  "s2": {
   "title": "② 처음부터 5번 학습",
   "lines": [
    "가중치 수 → 2,187개",
    [
     "valid → 정확도 0.656, macro F1 0.649",
     1
    ]
   ],
   "notes": [
    "무작위 가중치에서 시작해서, 선과 모서리 같은 기본 특징부터 배워야 해요. 데이터와 학습 횟수가 적으면 거기까지 가지 못해요."
   ]
  },
  "s3": {
   "title": "③ source_backbone.npz",
   "lines": [
    "0.weight → (8, 3, 3, 3)",
    "1.weight → (8,)",
    "1.bias → (8,)",
    "1.running_mean → (8,)",
    "1.running_var → (8,)",
    "4.weight → (16, 8, 3, 3)",
    [
     "load_state_dict → 성공 (이름과 모양이 모두 맞음)",
     1
    ]
   ],
   "notes": [
    "다른 도형 그림 300장으로 미리 학습해 둔 Backbone의 가중치예요 (이 그림들은 우리 train에 없어요). 층 번호와 모양이 우리 <code>make_backbone()</code>과 똑같아서 그대로 끼울 수 있어요."
   ]
  },
  "s4": {
   "title": "④ Backbone 고정, Head만 5번 학습",
   "lines": [
    [
     "학습하는 가중치 → 771개 / 전체 2,187개",
     1
    ],
    [
     "valid → 정확도 1.000, macro F1 1.000",
     2
    ]
   ],
   "notes": [
    "Head(256 → 3)만 배우니 학습할 게 3분의 1로 줄었어요.",
    "그런데도 ②보다 훨씬 잘 맞혀요. 빌려 온 Backbone이 이미 \"도형을 구분하는 특징\"을 뽑아 주니까요."
   ]
  },
  "s5": {
   "title": "⑤ 전체를 작은 학습률(0.001)로 5번 더",
   "lines": [
    [
     "valid → 정확도 0.978, macro F1 0.978",
     1
    ]
   ],
   "notes": [
    "④(1.000)보다 오히려 조금 떨어졌어요. 그림 180장으로 Backbone까지 고치면, 빌려 온 좋은 특징이 우리 데이터에 맞춰 흐트러질 수 있어요. 데이터가 적을 때는 Head만 배우는 게 나을 때가 많아요. 원래 데이터와 우리 데이터가 많이 다르고 데이터가 넉넉할수록 미세조정의 효과가 커요."
   ]
  },
  "s6": {
   "title": "⑥ 종류별 성적 (valid)",
   "cols": [
    "종류",
    "precision",
    "recall",
    "valid 장수"
   ],
   "rows": [
    [
     "circle",
     "0.94",
     "1.00",
     30
    ],
    [
     "square",
     "1.00",
     "0.97",
     30
    ],
    [
     "triangle",
     "1.00",
     "0.97",
     30
    ]
   ],
   "notes": [
    "혼동 행렬 (행: 실제, 열: 예측) → [[30, 0, 0], [1, 29, 0], [1, 0, 29]]",
    "틀린 2장은 모두 \"원\"으로 잘못 답했어요. train에 원이 가장 많아서 애매하면 원 쪽으로 기우는 거예요. 정확도 하나로는 이런 치우침이 안 보여서 종류별로 봐요. macro F1은 종류마다 F1을 구해 <b>똑같은 무게로</b> 평균 낸 값이라 적은 종류도 공평하게 반영돼요."
   ]
  },
  "s6img": {
   "title": "틀린 그림",
   "images": [
    [
     "img/transfer/wrong52.png",
     "정답 square → 예측 circle"
    ],
    [
     "img/transfer/wrong62.png",
     "정답 triangle → 예측 circle"
    ]
   ],
   "notes": [
    "틀린 그림을 직접 보면 원인을 짐작할 수 있어요. 모양이 작거나, 모서리가 뭉개졌거나, 다른 종류와 닮았는지요."
   ]
  },
  "s7": {
   "title": "⑦ 종류별 무게를 주고 미세조정",
   "lines": [
    [
     "weight → [0.5, 1.5, 3.0]  (circle, square, triangle)",
     1
    ],
    [
     "valid → 정확도 0.989, macro F1 0.989",
     2
    ]
   ],
   "notes": [
    "적은 종류일수록 틀렸을 때 손실을 크게 쳐요. 세모 하나를 틀리면 원 하나를 틀린 것의 6배로요.",
    "⑤(0.978)와 비교하면 valid에서 0.989로 조금 올랐어요."
   ]
  },
  "s8": {
   "title": "⑧ test 30장 macro F1 (처음 사용)",
   "cols": [
    "방법",
    "macro F1"
   ],
   "rows": [
    [
     "처음부터",
     "0.663"
    ],
    [
     "Head만",
     "0.967"
    ],
    [
     "미세조정",
     "0.967"
    ],
    [
     "미세조정 + 무게",
     "0.967"
    ]
   ],
   "notes": [
    "빌려 온 Backbone을 쓴 방법들이 처음부터 배운 것보다 크게 나아요. 그 사이의 차이(Head만, 미세조정, 무게)는 30장 중 한두 장이라 우연일 수 있어요. 이 데이터로 확실히 말할 수 있는 건 \"빌려 오면 훨씬 낫다\"까지예요."
   ]
  }
 },
 "flow": [
  [
   1,
   "데이터",
   "180장",
   "불균형"
  ],
  [
   2,
   "처음부터",
   "5번 학습",
   "부족함"
  ],
  [
   3,
   "빌려 오기",
   "npz → Backbone",
   "모양이 같아야"
  ],
  [
   4,
   "Head만",
   "Backbone 고정",
   "가중치 1/3"
  ],
  [
   5,
   "미세조정",
   "작은 학습률",
   "전체 조금"
  ],
  [
   6,
   "종류별",
   "macro F1",
   "틀린 그림"
  ],
  [
   7,
   "무게",
   "적은 종류 ↑",
   "손실 무게"
  ],
  [
   8,
   "test",
   "한 번",
   "비교"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, CPU (seed 0)"
};
