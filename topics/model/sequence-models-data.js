window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
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
   "t": "from torch.nn.utils.rnn import pad_sequence",
   "s": 1
  },
  {
   "t": "from torch.nn.utils.rnn import pack_padded_sequence",
   "s": 4
  },
  {
   "t": "import pandas as pd",
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
   "t": "# ① 길이가 다른 기록을 한 묶음으로",
   "s": 1
  },
  {
   "t": "records = json.loads((DATA / \"sequences.json\").read_text(encoding=\"utf-8\"))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def batch(split):",
   "s": 1
  },
  {
   "t": "    rows = [r for r in records if r[\"split\"] == split]",
   "s": 1
  },
  {
   "t": "    seqs = [torch.tensor(r[\"values\"], dtype=torch.float32) for r in rows]",
   "s": 1
  },
  {
   "t": "    X = pad_sequence(seqs, batch_first=True)  # 짧은 기록 뒤를 0으로 채워 길이 맞추기",
   "s": 1
  },
  {
   "t": "    lengths = torch.tensor([len(s) for s in seqs])",
   "s": 1
  },
  {
   "t": "    mask = torch.arange(X.shape[1])[None, :] < lengths[:, None]  # 진짜 값이 있는 자리만 True",
   "s": 1
  },
  {
   "t": "    y = torch.tensor([r[\"label\"] for r in rows], dtype=torch.float32)",
   "s": 1
  },
  {
   "t": "    return X, lengths, mask, y",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "X_train, len_train, mask_train, y_train = batch(\"train\")",
   "s": 1
  },
  {
   "t": "X_valid, len_valid, mask_valid, y_valid = batch(\"valid\")",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def fit(model, epochs=200, lr=0.01):",
   "s": 1
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 1
  },
  {
   "t": "    opt = torch.optim.Adam(model.parameters(), lr=lr)",
   "s": 1
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 1
  },
  {
   "t": "        loss = nn.functional.binary_cross_entropy_with_logits(model(X_train, len_train, mask_train), y_train)",
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
   "t": "    with torch.no_grad():",
   "s": 1
  },
  {
   "t": "        return ((model(X_valid, len_valid, mask_valid) > 0).float() == y_valid).float().mean().item()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 순서를 무시하면: 평균만 보기",
   "s": 2
  },
  {
   "t": "class MeanModel(nn.Module):",
   "s": 2
  },
  {
   "t": "    def __init__(self):",
   "s": 2
  },
  {
   "t": "        super().__init__()",
   "s": 2
  },
  {
   "t": "        self.out = nn.Linear(3, 1)",
   "s": 2
  },
  {
   "t": "    def forward(self, X, lengths, mask):",
   "s": 2
  },
  {
   "t": "        mean = (X * mask[..., None]).sum(1) / lengths[:, None]  # 채운 0은 빼고 평균",
   "s": 2
  },
  {
   "t": "        return self.out(mean).squeeze(1)",
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
   "t": "acc_mean = fit(MeanModel())",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ RNN: 앞에서부터 차례로 읽기",
   "s": 3
  },
  {
   "t": "class RNNModel(nn.Module):",
   "s": 3
  },
  {
   "t": "    def __init__(self, cell=nn.RNN, hidden=16):",
   "s": 3
  },
  {
   "t": "        super().__init__()",
   "s": 3
  },
  {
   "t": "        self.rnn = cell(3, hidden, batch_first=True)",
   "s": 3
  },
  {
   "t": "        self.out = nn.Linear(hidden, 1)",
   "s": 3
  },
  {
   "t": "    def forward(self, X, lengths, mask):",
   "s": 3
  },
  {
   "t": "        h, _ = self.rnn(X)  # 시점마다의 기억 (batch, 길이, hidden)",
   "s": 3
  },
  {
   "t": "        last = h[torch.arange(len(X)), lengths - 1]  # 진짜 마지막 시점의 기억",
   "s": 3
  },
  {
   "t": "        return self.out(last).squeeze(1)",
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
   "t": "acc_rnn = fit(RNNModel(nn.RNN))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ LSTM: 오래 기억하는 RNN",
   "s": 4
  },
  {
   "t": "class LSTMModel(RNNModel):",
   "s": 4
  },
  {
   "t": "    def forward(self, X, lengths, mask):",
   "s": 4
  },
  {
   "t": "        packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)",
   "s": 4
  },
  {
   "t": "        _, (h_n, _) = self.rnn(packed)  # 채운 0은 읽지 않고, 마지막 기억만",
   "s": 4
  },
  {
   "t": "        return self.out(h_n[-1]).squeeze(1)",
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
   "t": "acc_lstm = fit(LSTMModel(nn.LSTM))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ Attention: 어느 시점이 중요한지 점수 매겨 섞기",
   "s": 5
  },
  {
   "t": "class AttnModel(nn.Module):",
   "s": 5
  },
  {
   "t": "    def __init__(self, hidden=16):",
   "s": 5
  },
  {
   "t": "        super().__init__()",
   "s": 5
  },
  {
   "t": "        self.enc = nn.Linear(3, hidden)",
   "s": 5
  },
  {
   "t": "        self.score = nn.Linear(hidden, 1)",
   "s": 5
  },
  {
   "t": "        self.out = nn.Linear(hidden, 1)",
   "s": 5
  },
  {
   "t": "    def forward(self, X, lengths, mask, return_weights=False):",
   "s": 5
  },
  {
   "t": "        h = torch.tanh(self.enc(X))",
   "s": 5
  },
  {
   "t": "        s = self.score(h).squeeze(-1).masked_fill(~mask, float(\"-inf\"))  # 채운 자리는 점수 -∞",
   "s": 5
  },
  {
   "t": "        w = torch.softmax(s, dim=1)  # 시점별 가중치, 합이 1",
   "s": 5
  },
  {
   "t": "        y = self.out((w[..., None] * h).sum(1)).squeeze(1)",
   "s": 5
  },
  {
   "t": "        return (y, w) if return_weights else y",
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
   "t": "attn = AttnModel()",
   "s": 5
  },
  {
   "t": "acc_attn = fit(attn)",
   "s": 5
  },
  {
   "t": "with torch.no_grad():",
   "s": 5
  },
  {
   "t": "    _, weights = attn(X_valid[:1], len_valid[:1], mask_valid[:1], return_weights=True)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ Autoencoder: 잘 복원되지 않으면 이상",
   "s": 6
  },
  {
   "t": "series = pd.read_csv(DATA / \"series.csv\")",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def windows(split, size=12):",
   "s": 6
  },
  {
   "t": "    v = torch.tensor(series[series[\"split\"] == split][\"value\"].values, dtype=torch.float32)",
   "s": 6
  },
  {
   "t": "    return v.unfold(0, size, 1)  # 길이 12짜리 창을 한 칸씩 밀며",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "W_train, W_valid, W_test = windows(\"train\"), windows(\"valid\"), windows(\"test\")",
   "s": 6
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "ae = nn.Sequential(nn.Linear(12, 4), nn.Tanh(), nn.Linear(4, 12))  # 12 → 4로 줄였다가 → 12로 되살리기",
   "s": 6
  },
  {
   "t": "opt = torch.optim.Adam(ae.parameters(), lr=0.01)",
   "s": 6
  },
  {
   "t": "for _ in range(500):",
   "s": 6
  },
  {
   "t": "    loss = ((ae(W_train) - W_train) ** 2).mean()",
   "s": 6
  },
  {
   "t": "    opt.zero_grad()",
   "s": 6
  },
  {
   "t": "    loss.backward()",
   "s": 6
  },
  {
   "t": "    opt.step()",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    err_train = ((ae(W_train) - W_train) ** 2).mean(1)",
   "s": 6
  },
  {
   "t": "    err_valid = ((ae(W_valid) - W_valid) ** 2).mean(1)",
   "s": 6
  },
  {
   "t": "    err_test = ((ae(W_test) - W_test) ** 2).mean(1)",
   "s": 6
  },
  {
   "t": "limit = err_train.mean() + 3 * err_train.std()  # train 복원 오차의 평균 + 표준편차 3배",
   "s": 6
  },
  {
   "t": "alerts = {\"valid\": (err_valid > limit).nonzero().flatten().tolist(), \"test\": (err_test > limit).nonzero().flatten().tolist()}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "series_odd = series[series[\"split\"] == \"test\"][\"value\"].values.copy()",
   "s": 6
  },
  {
   "t": "series_odd[35] += 1.0  # 확인용: test 35번째 시점에 일부러 튀는 값 하나",
   "s": 6
  },
  {
   "t": "W_odd = torch.tensor(series_odd, dtype=torch.float32).unfold(0, 12, 1)",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    err_odd = ((ae(W_odd) - W_odd) ** 2).mean(1)",
   "s": 6
  },
  {
   "t": "alerts_odd = (err_odd > limit).nonzero().flatten().tolist()  # 그 시점을 품은 창들",
   "s": 6
  }
 ],
 "names": {
  "1": "길이가 다른 기록을 한 묶음으로",
  "2": "순서를 무시하면: 평균만 보기",
  "3": "RNN: 앞에서부터 차례로 읽기",
  "4": "LSTM: 오래 기억하는 RNN",
  "5": "Attention: 어느 시점이 중요한지 점수 매겨 섞기",
  "6": "Autoencoder: 잘 복원되지 않으면 이상"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "data": {
   "title": "sequences.json의 첫 기록",
   "lines": [
    "id train_000, split train, label 0, 길이 8",
    "  시점 1: [0.993, -0.382, -0.17]",
    "  시점 2: [0.764, -0.311, 0.152]",
    "  시점 3: [0.505, -0.292, -0.089]",
    "  시점 4: [0.113, -0.123, -0.151]",
    "  …"
   ],
   "notes": [
    "기록마다 시점 수가 6~12로 다르고, 시점마다 센서 값이 3개예요. <code>label</code>(0 또는 1)을 맞히는 게 목표예요."
   ]
  },
  "why": {
   "title": "valid 20개 정확도",
   "cols": [
    "모델",
    "순서를 보나",
    "정확도"
   ],
   "rows": [
    [
     "평균만 보기 (②)",
     "아니요",
     "<span class=\"bad\">0.45</span>"
    ],
    [
     "RNN (③)",
     "예",
     "<b>1.00</b>"
    ],
    [
     "LSTM (④)",
     "예",
     "<b>1.00</b>"
    ],
    [
     "Attention (⑤)",
     "예",
     "<b>1.00</b>"
    ]
   ],
   "notes": [
    "값들의 평균만 보면 동전 던지기 수준이에요. 두 종류는 같은 값들을 <b>다른 순서</b>로 갖고 있어서, 순서를 봐야 구분돼요."
   ]
  },
  "s1": {
   "title": "① batch(\"train\")",
   "lines": [
    [
     "X_train.shape → (60, 12, 3)",
     1
    ],
    "len_train 앞 5개 → [8, 6, 12, 11, 7]",
    [
     "mask_train[0] → [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]",
     2
    ]
   ],
   "notes": [
    "기록 60개를 가장 긴 길이 12에 맞춰 (60, 12, 3) 하나의 Tensor로 만들었어요. 짧은 기록의 뒤는 0으로 채웠어요 (padding).",
    "첫 기록은 길이 8이라 앞 8칸만 진짜(1)이고 뒤 4칸은 채운 자리(0)예요. 채운 0을 진짜 값으로 읽으면 안 돼서 이 표시를 같이 들고 다녀요."
   ]
  },
  "s2": {
   "title": "② 평균만 보는 모델",
   "lines": [
    [
     "valid 정확도 → 0.45",
     1
    ]
   ],
   "notes": [
    "채운 0은 빼고 진짜 값들만 평균했어요. 그래도 순서 정보가 사라져서 구분하지 못해요."
   ]
  },
  "s3": {
   "title": "③ RNN",
   "lines": [
    [
     "valid 정확도 → 1.00",
     1
    ]
   ],
   "notes": [
    "시점을 하나씩 읽으며 \"지금까지 본 것의 요약\"(hidden state)을 갱신하고, 진짜 마지막 시점의 요약으로 판단했어요."
   ]
  },
  "s4": {
   "title": "④ LSTM",
   "lines": [
    [
     "valid 정확도 → 1.00",
     1
    ]
   ],
   "notes": [
    "이 데이터는 짧아서 RNN도 충분했어요. 기록이 길어질수록 RNN은 앞쪽을 잊기 쉬워서 LSTM이 유리해져요."
   ]
  },
  "s5": {
   "title": "⑤ Attention, valid 첫 기록(길이 6)의 시점별 가중치",
   "lines": [
    [
     "정확도 → 1.00",
     1
    ],
    [
     "가중치 → [0.516, 0.377, 0.039, 0.01, 0.006, 0.053, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",
     2
    ],
    "합 → 1.000"
   ],
   "notes": [
    "순서대로 읽는 대신, 모든 시점을 한꺼번에 보고 \"어느 시점이 중요한가\"를 점수로 매겨 섞었어요.",
    "이 기록에서는 앞의 두 시점에 가중치가 몰렸어요. 길이 6 뒤의 채운 자리는 정확히 0이에요 (점수를 −∞로 막아서). 가중치를 보면 모델이 어디를 봤는지 알 수 있어요."
   ]
  },
  "s6": {
   "title": "⑥ Autoencoder 복원 오차",
   "lines": [
    "창 크기 12, train 창 169개",
    [
     "이상 기준 (train 평균 + 표준편차 3배) → 0.0025",
     1
    ],
    "valid, test에서 기준을 넘은 창 → {'valid': [], 'test': []}",
    [
     "test 35번째 시점에 일부러 +1.0을 넣으면 → 걸린 창 [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]  (최대 오차 0.138)",
     2,
     "bad"
    ]
   ],
   "notes": [
    "정상 데이터만으로 \"12칸을 4칸으로 줄였다가 되살리는\" 법을 배웠어요. 정상 패턴은 잘 되살려서 오차가 작아요.",
    "튀는 값이 섞인 창은 배운 적 없는 모양이라 잘 되살리지 못해 오차가 기준의 수십 배가 됐어요. 그 시점을 품은 창 12개가 모두 걸렸어요. 정답(이상 표시) 없이도 이상을 찾을 수 있어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "묶기",
   "padding, mask",
   "길이가 달라도"
  ],
  [
   2,
   "평균",
   "순서 무시",
   "0.45"
  ],
  [
   3,
   "RNN",
   "차례로 읽기",
   "마지막 기억"
  ],
  [
   4,
   "LSTM",
   "오래 기억",
   "pack"
  ],
  [
   5,
   "Attention",
   "중요한 시점",
   "가중치 합 1"
  ],
  [
   6,
   "Autoencoder",
   "줄였다 되살리기",
   "오차 = 이상"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, CPU (seed 0)"
};
