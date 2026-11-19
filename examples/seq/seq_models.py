# 순서 모델 전체 코드 (PyTorch). 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/sequences.json (센서 기록 100개, 길이 6~12, 시점마다 값 3개, label 0/1)
#         data/series.csv (시간 300칸짜리 값 하나, train 180 / valid 60 / test 60)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
from pathlib import Path                                         #@1
import torch                                                     #@1
from torch import nn                                             #@1
from torch.nn.utils.rnn import pad_sequence                      #@1
from torch.nn.utils.rnn import pack_padded_sequence              #@4
import pandas as pd                                              #@6

DATA = Path(__file__).resolve().parent / "data"                  #@1

# ① 길이가 다른 기록을 한 묶음으로
records = json.loads((DATA / "sequences.json").read_text(encoding="utf-8"))

def batch(split):
    rows = [r for r in records if r["split"] == split]
    seqs = [torch.tensor(r["values"], dtype=torch.float32) for r in rows]
    X = pad_sequence(seqs, batch_first=True)          # 짧은 기록 뒤를 0으로 채워 길이 맞추기
    lengths = torch.tensor([len(s) for s in seqs])
    mask = torch.arange(X.shape[1])[None, :] < lengths[:, None]  # 진짜 값이 있는 자리만 True
    y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
    return X, lengths, mask, y

X_train, len_train, mask_train, y_train = batch("train")
X_valid, len_valid, mask_valid, y_valid = batch("valid")

def fit(model, epochs=200, lr=0.01):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        loss = nn.functional.binary_cross_entropy_with_logits(model(X_train, len_train, mask_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((model(X_valid, len_valid, mask_valid) > 0).float() == y_valid).float().mean().item()

# ② 순서를 무시하면: 평균만 보기
class MeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(3, 1)
    def forward(self, X, lengths, mask):
        mean = (X * mask[..., None]).sum(1) / lengths[:, None]  # 채운 0은 빼고 평균
        return self.out(mean).squeeze(1)

torch.manual_seed(0)
acc_mean = fit(MeanModel())

# ③ RNN: 앞에서부터 차례로 읽기
class RNNModel(nn.Module):
    def __init__(self, cell=nn.RNN, hidden=16):
        super().__init__()
        self.rnn = cell(3, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)
    def forward(self, X, lengths, mask):
        h, _ = self.rnn(X)                                  # 시점마다의 기억 (batch, 길이, hidden)
        last = h[torch.arange(len(X)), lengths - 1]         # 진짜 마지막 시점의 기억
        return self.out(last).squeeze(1)

torch.manual_seed(0)
acc_rnn = fit(RNNModel(nn.RNN))

# ④ LSTM: 오래 기억하는 RNN
class LSTMModel(RNNModel):
    def forward(self, X, lengths, mask):
        packed = pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.rnn(packed)                      # 채운 0은 읽지 않고, 마지막 기억만
        return self.out(h_n[-1]).squeeze(1)

torch.manual_seed(0)
acc_lstm = fit(LSTMModel(nn.LSTM))

# ⑤ Attention: 어느 시점이 중요한지 점수 매겨 섞기
class AttnModel(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.enc = nn.Linear(3, hidden)
        self.score = nn.Linear(hidden, 1)
        self.out = nn.Linear(hidden, 1)
    def forward(self, X, lengths, mask, return_weights=False):
        h = torch.tanh(self.enc(X))
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))  # 채운 자리는 점수 -∞
        w = torch.softmax(s, dim=1)                         # 시점별 가중치, 합이 1
        y = self.out((w[..., None] * h).sum(1)).squeeze(1)
        return (y, w) if return_weights else y

torch.manual_seed(0)
attn = AttnModel()
acc_attn = fit(attn)
with torch.no_grad():
    _, weights = attn(X_valid[:1], len_valid[:1], mask_valid[:1], return_weights=True)

# ⑥ Autoencoder: 잘 복원되지 않으면 이상
series = pd.read_csv(DATA / "series.csv")

def windows(split, size=12):
    v = torch.tensor(series[series["split"] == split]["value"].values, dtype=torch.float32)
    return v.unfold(0, size, 1)  # 길이 12짜리 창을 한 칸씩 밀며

W_train, W_valid, W_test = windows("train"), windows("valid"), windows("test")
torch.manual_seed(0)
ae = nn.Sequential(nn.Linear(12, 4), nn.Tanh(), nn.Linear(4, 12))  # 12 → 4로 줄였다가 → 12로 되살리기
opt = torch.optim.Adam(ae.parameters(), lr=0.01)
for _ in range(500):
    loss = ((ae(W_train) - W_train) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    err_train = ((ae(W_train) - W_train) ** 2).mean(1)
    err_valid = ((ae(W_valid) - W_valid) ** 2).mean(1)
    err_test = ((ae(W_test) - W_test) ** 2).mean(1)
limit = err_train.mean() + 3 * err_train.std()  # train 복원 오차의 평균 + 표준편차 3배
alerts = {"valid": (err_valid > limit).nonzero().flatten().tolist(), "test": (err_test > limit).nonzero().flatten().tolist()}

series_odd = series[series["split"] == "test"]["value"].values.copy()
series_odd[35] += 1.0  # 확인용: test 35번째 시점에 일부러 튀는 값 하나
W_odd = torch.tensor(series_odd, dtype=torch.float32).unfold(0, 12, 1)
with torch.no_grad():
    err_odd = ((ae(W_odd) - W_odd) ** 2).mean(1)
alerts_odd = (err_odd > limit).nonzero().flatten().tolist()  # 그 시점을 품은 창들
