"""Day12 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261103/02_응용/data: sequences.json · series.csv)는 임시 폴더에 복사해서 읽기만 한다.
나머지는 seed를 고정한 작은 난수 Tensor로 강의 예제 모양을 그대로 따라 한다.
결과는 cmds/day12.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day12.py
"""
import csv
import html
import json
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", message="LSTM with projections")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261103/02_응용/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")
DATA = TMP / "data"


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"


def sh(t):
    return str(list(t.shape))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def mat(t, fmt="{:g}"):
    return "\n".join(" ".join(fmt.format(v) for v in row) for row in t.tolist())


IO = {}
SEQ = json.loads((DATA / "sequences.json").read_text(encoding="utf-8"))
train_seq = [d for d in SEQ if d["split"] == "train"]

# 1. mask / masked_fill / pad_sequence
seed(0)
x = torch.randn(4, 6, 3)
lengths = torch.tensor([6, 4, 5, 3])
mask = torch.arange(6)[None, :] >= lengths[:, None]
xm = x.masked_fill(mask.unsqueeze(-1), 0.0)
seqs = [torch.ones(4, 3), torch.ones(2, 3) * 2, torch.ones(5, 3) * 3]
padded = pad_sequence(seqs, batch_first=True)
valid = (~(torch.arange(5)[None, :] >= torch.tensor([4, 2, 5])[:, None])).unsqueeze(-1)
mean = (padded * valid).sum(1) / valid.sum(1)
real = [torch.tensor(d["values"]) for d in train_seq[:4]]
real_pad = pad_sequence(real, batch_first=True)
IO["d12-pad-sequence"] = ("torch.randn(4, 6, 3) (seed 0) · 강의 sequences.json train 앞 4개", [
    ("mask = arange(6)[None, :] >= lengths[:, None]  (lengths [6, 4, 5, 3])", mat(mask.int()), "1 = Padding 칸 (실제 길이 뒤쪽)"),
    ("x.masked_fill(mask.unsqueeze(-1), 0.0)[1]", mat(xm[1], "{:.2f}"), "샘플 1은 길이 4 → 뒤 2줄이 0"),
    ("x.masked_fill(mask, 0.0)  ← unsqueeze 빼면", "RuntimeError (shape [4, 6] vs [4, 6, 3])", "Mask가 마지막 F축까지 안 늘어남"),
    ("pad_sequence(seqs, batch_first=True).shape", sh(padded), "길이 4·2·5 → 가장 긴 5에 맞춤"),
    ("pad_sequence(seqs).shape  ← batch_first 기본 False", sh(pad_sequence(seqs)), "[T, B, F] — 순서가 바뀜"),
    ("padded[1, :, 0]", " ".join(f"{v:g}" for v in padded[1, :, 0].tolist()), "길이 2짜리 뒤를 0으로 채움"),
    ("(padded * valid).sum(1) / valid.sum(1)  [:, 0]", " ".join(f"{v:g}" for v in mean[:, 0].tolist()), "실제 칸만 평균 → 1 · 2 · 3 그대로"),
    ("padded.mean(1)[:, 0]  ← Mask 없이 평균", " ".join(f"{v:g}" for v in padded.mean(1)[:, 0].tolist()), "Padding 0이 섞여 짧은 샘플일수록 작아짐"),
    ("강의 데이터: [d['length'] for d in train[:4]]", repr([d["length"] for d in train_seq[:4]]), ""),
    ("pad_sequence(강의 4개, batch_first=True).shape", sh(real_pad), "[B=4, T=최대 12, F=3]"),
])
try:
    x.masked_fill(mask, 0.0)
except RuntimeError:
    pass
else:
    raise SystemExit("masked_fill 오류가 안 났음")

# 2. unfold
rows = list(csv.DictReader((DATA / "series.csv").open(encoding="utf-8")))
train_s = torch.tensor([float(r["value"]) for r in rows if r["split"] == "train"])
window = train_s.unfold(0, 13, 1)
xw = window[:, :12].unsqueeze(-1)
yw = window[:, 12:13]
toy = torch.arange(20.0)
tw = toy.unfold(0, 5, 1)
toy[0] = 100.0
IO["d12-unfold"] = ("강의 series.csv의 train 구간(180개) · torch.arange(20)", [
    ("train_s.shape", sh(train_s), "split == 'train' 인 값만 (시간 순서 그대로)"),
    ("window = train_s.unfold(0, 13, 1); window.shape", sh(window), "180 − 13 + 1 = 168개 창"),
    ("window[:, :12].unsqueeze(-1).shape", sh(xw), "[N, T=12, F=1]"),
    ("window[:, 12:13].shape", sh(yw), "[N, 1] 정답"),
    ("torch.equal(window[1, :12], train_s[1:13])", str(torch.equal(window[1, :12], train_s[1:13])), "창은 1칸씩 밀림"),
    ("torch.arange(20.).unfold(0, 5, 1).shape", sh(tw), "20 − 5 + 1 = 16개"),
    ("torch.arange(20.).unfold(0, 5, 2).shape", sh(torch.arange(20.0).unfold(0, 5, 2)), "step 2 → (20 − 5) // 2 + 1 = 8개"),
    ("원본 toy[0] = 100 으로 바꾼 뒤 tw[0]", " ".join(f"{v:g}" for v in tw[0].tolist()), "view라 창도 같이 바뀜"),
])

# 3. pack_padded_sequence
seed(0)
real_len = torch.tensor([d["length"] for d in train_seq[:4]])
rnn = nn.RNN(3, 8, batch_first=True)
packed = pack_padded_sequence(real_pad, real_len, batch_first=True, enforce_sorted=False)
out_packed, h_n = rnn(packed)
output, out_lengths = pad_packed_sequence(out_packed, batch_first=True)
_, h_pad = rnn(real_pad)
out_pad, _ = rnn(real_pad)
i_short = int(real_len.argmin())
last_real = output[torch.arange(4), real_len - 1]
IO["d12-pack-padded-sequence"] = (f"강의 sequences.json train 앞 4개 (길이 {real_len.tolist()}, F=3) · nn.RNN(3, 8) seed 0", [
    ("packed.data.shape", sh(packed.data), f"Σ길이 {int(real_len.sum())} 칸만 담음 (Padding 칸 없음)"),
    ("packed.batch_sizes[:3], [-3:]", f"{packed.batch_sizes[:3].tolist()}, {packed.batch_sizes[-3:].tolist()}", "시점마다 아직 살아 있는 샘플 수"),
    ("output.shape, out_lengths", f"{sh(output)}, {out_lengths.tolist()}", "다시 [B, T, H]로 펼침"),
    ("torch.allclose(h_n[-1], output[range(4), lengths-1])", str(torch.allclose(h_n[-1], last_real, atol=1e-6)), "Pack한 h_n = 샘플마다 실제 마지막 시점"),
    (f"Pack 없이 rnn(padded) → h_n 차이 (길이 {int(real_len[i_short])}짜리)", f"{(h_pad[-1, i_short] - h_n[-1, i_short]).abs().max():.3f}", "Pack 안 하면 Padding 0을 더 읽은 뒤의 상태"),
    ("pad_packed_sequence(…, total_length=15)[0].shape", sh(pad_packed_sequence(out_packed, batch_first=True, total_length=15)[0]), "원래 T에 맞춰 펴기"),
    ("pack_padded_sequence(padded, lengths.cuda(), …)", "RuntimeError (lengths는 CPU)" if DEV.type == "cuda" else "(GPU 없음)", "lengths는 CPU Tensor/정수 목록이어야 함"),
])
if DEV.type == "cuda":
    try:
        pack_padded_sequence(real_pad, real_len.cuda(), batch_first=True, enforce_sorted=False)
    except RuntimeError:
        pass
    else:
        raise SystemExit("GPU lengths 오류가 안 났음")


# 4. RNN
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(3, 8, batch_first=True)
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])


seed(0)
x = torch.randn(4, 6, 3)
clf = RNNClassifier()
o, hn = clf.rnn(x)
r2 = nn.RNN(3, 8, num_layers=2, batch_first=True)
rb = nn.RNN(3, 8, bidirectional=True, batch_first=True)
ob, hb = rb(x)
IO["d12-nn-rnn"] = ("torch.randn(4, 6, 3) (seed 0) = [B=4, T=6, F=3]", [
    ("output, h_n = nn.RNN(3, 8, batch_first=True)(x)", f"output {sh(o)}, h_n {sh(hn)}", "output은 모든 시점, h_n은 마지막 상태만"),
    ("torch.allclose(h_n[-1], output[:, -1])", str(torch.allclose(hn[-1], o[:, -1])), "1층·단방향·Padding 없음 → 같은 값"),
    ("RNNClassifier()(x).shape", sh(clf(x)), "[B, 2] 점수"),
    ("sum(p.numel() for p in nn.RNN(3, 8).parameters())", str(n_params(clf.rnn)), "W_x 8×3 + W_h 8×8 + bias 8 × 2 = 104"),
    ("num_layers=2 → h_n.shape", sh(r2(x)[1]), "첫 축 = 층 수 → h_n[-1]이 마지막 층"),
    ("bidirectional=True → output.shape, h_n.shape", f"{sh(ob)}, {sh(hb)}", "마지막 축 2H=16, h_n 첫 축 D=2"),
    ("nn.RNN(3, 8)(x)  ← batch_first 기본 False", sh(nn.RNN(3, 8)(x)[0]), "오류 없이 [T=4, B=6]으로 읽어 버림 — 출력 모양까지 같아서 눈치채기 어려움!"),
])


# 5. LSTM
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1])


seed(0)
fx = torch.randn(4, 12, 1)
fm = LSTMForecaster()
lo, (lh, lc) = fm.lstm(fx)
valid_s = torch.tensor([float(r["value"]) for r in rows if r["split"] == "valid"])
vw = valid_s.unfold(0, 13, 1)
vx_, vy_ = vw[:, :12].unsqueeze(-1), vw[:, 12:13]
crit = nn.MSELoss()
with torch.no_grad():
    before = crit(fm(vx_), vy_).item()
opt = torch.optim.Adam(fm.parameters(), lr=1e-2)
for _ in range(100):
    opt.zero_grad()
    loss = crit(fm(xw), yw)
    loss.backward()
    opt.step()
with torch.no_grad():
    after = crit(fm(vx_), vy_).item()
    naive = crit(vx_[:, -1], vy_).item()
IO["d12-nn-lstm"] = ("torch.randn(4, 12, 1) (seed 0) · 강의 series.csv 창(train 168개 · valid 48개)으로 100 step 학습", [
    ("output, (h_n, c_n) = nn.LSTM(1, 16, batch_first=True)(x)", f"output {sh(lo)}, h_n {sh(lh)}, c_n {sh(lc)}", "상태가 두 개(h, c)"),
    ("LSTMForecaster()(x).shape", sh(fm(fx)), "[B, 1] 다음 값 하나"),
    ("sum(p.numel() for p in nn.LSTM(1, 16).parameters())", f"{n_params(fm.lstm):,}", "Gate 4개 × (16·1 + 16·16 + 16 + 16) = 1,216 — RNN의 4배"),
    ("nn.LSTM(1, 16, proj_size=8) → h_n, c_n", " , ".join(sh(t) for t in nn.LSTM(1, 16, proj_size=8, batch_first=True)(fx)[1]), "h만 8로 줄고 c는 16 그대로"),
    ("valid MSE (학습 전)", f"{before:.4f}", "무작위 가중치"),
    ("valid MSE (train 창으로 Adam 100 step 후)", f"{after:.4f}", "다음 값을 맞히는 쪽으로 줄어듦"),
    ("valid MSE (그냥 '마지막 값 그대로' 예측)", f"{naive:.4f}", "비교용 기준선 — 모델이 이걸 이겨야 의미가 있음"),
])

# 6. Autoencoder
seed(0)
ae = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2), nn.ReLU(), nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 8))
xa = torch.randn(5, 8)
recon = ae(xa)
loss = nn.MSELoss()(recon, xa)
score = ((recon - xa) ** 2).mean(dim=1)
IO["d12-autoencoder-forward와-재구성-오차"] = ("torch.randn(5, 8) (seed 0), 학습 전 Autoencoder", [
    ("recon.shape, score.shape", f"{sh(recon)}, {sh(score)}", "복원은 입력과 같은 모양, 점수는 샘플마다 1개"),
    ("model[:4](x).shape", sh(ae[:4](xa)), "앞 4개 층만 = Encoder → Latent z 2칸"),
    ("loss.item()", f"{loss.item():.4f}", "Batch 전체 평균 하나"),
    ("score", " ".join(f"{v:.3f}" for v in score.tolist()), "샘플별 오류 — 어느 샘플이 큰지 보임"),
    ("torch.isclose(score.mean(), loss)", str(bool(torch.isclose(score.mean(), loss))), "샘플별 점수의 평균 = 학습용 loss"),
    ("sum(p.numel() for p in model.parameters())", str(n_params(ae)), "36 + 10 + 12 + 40"),
])

# 7. reduction="none"
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    bc_val = nn.MSELoss()(recon, xa[:1]).item()
bc_msg = f"{bc_val:.4f} + UserWarning" if any("target size" in str(c.message) for c in caught) else f"{bc_val:.4f} (경고 없음)"
none = nn.MSELoss(reduction="none")(recon, xa)
error = none.mean(dim=1)
IO["d12-reduction-none"] = ("'Autoencoder forward' 항목의 recon, x 그대로", [
    ('nn.MSELoss(reduction="none")(recon, x).shape', sh(none), "칸마다 제곱 오차 [B, F]"),
    (".mean(dim=1).shape", sh(error), "샘플마다 하나 [B]"),
    ('nn.MSELoss()(recon, x)  (기본 "mean")', f"{nn.MSELoss()(recon, xa).item():.4f}", "숫자 하나"),
    ('nn.MSELoss(reduction="sum")(recon, x)', f"{nn.MSELoss(reduction='sum')(recon, xa).item():.4f}", "= mean × 40칸"),
    ('nn.MSELoss(reduction="none")(recon, x).mean(dim=0).shape', sh(none.mean(dim=0)), "dim=0이면 특징별 오차 [F]"),
    ("nn.MSELoss()(recon, x[:1])  ← target [1, 8]", bc_msg, "에러 없이 Broadcasting으로 계산돼 버림 — 경고를 놓치면 엉뚱한 값"),
])

# 8. TensorDataset / Denoising / 기준값
seed(0)
z = torch.randn(300, 2)
mix = torch.randn(2, 8)
x_all = z @ mix + 0.05 * torch.randn(300, 8)
x_train, x_valid = x_all[:200], x_all[200:].clone()
x_valid[:3] += torch.tensor([3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0])  # 이상 샘플 3개를 일부러 넣음
loader = DataLoader(TensorDataset(x_train), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
first = next(iter(loader))
seed(0)
model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2), nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 8))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(60):
    for (xb,) in loader:
        noisy = xb + 0.1 * torch.randn_like(xb)
        loss = nn.MSELoss()(model(noisy), xb)
        opt.zero_grad()
        loss.backward()
        opt.step()
with torch.no_grad():
    valid_err = nn.MSELoss(reduction="none")(model(x_valid), x_valid).mean(dim=1)
threshold = torch.quantile(valid_err, 0.99)
is_candidate = valid_err > threshold
top = valid_err.topk(4).indices.tolist()
assert set(top[:3]) == {0, 1, 2} and int(is_candidate.sum()) == 1, (top, is_candidate.sum())
IO["d12-입력-정답-batch-denoising-기준값"] = ("seed 0 합성 데이터: 2차원 잠재값을 8칸으로 늘린 train 200개 · valid 100개(앞 3개는 일부러 이상값), Denoising AE 60 epoch", [
    ("type(batch), len(batch)  (next(iter(loader)))", f"{type(first).__name__}, {len(first)}", "Tensor 하나만 넣어서 한 칸짜리 묶음"),
    ("(xb,) = batch; xb.shape", sh(first[0]), "정답 Tensor가 따로 없음 — x 자신"),
    ("마지막 train loss (노이즈 입력 → 깨끗한 정답)", f"{loss.item():.4f}", ""),
    ("valid_err.shape", sh(valid_err), "샘플마다 점수"),
    ("threshold = torch.quantile(valid_err, 0.99)", f"{threshold.item():.4f}", "valid 오류의 상위 1% 경계"),
    ("is_candidate.sum()", str(int(is_candidate.sum())), "이상값을 3개 넣었는데 1개만 후보! 99% 규칙은 '100개 중 약 1개'를 고르는 규칙이라서"),
    ("valid_err.topk(4).indices", repr(top), "점수 순위로는 이상값 0·1·2가 1~3등 — 모델은 잘 구분, 놓친 건 기준값 규칙"),
    ("정상 valid 오류 중앙값 vs 이상 3개 평균", f"{valid_err[3:].median():.4f} vs {valid_err[:3].mean():.3f}", "이상값은 재구성이 훨씬 나쁨"),
])

# 9. MultiheadAttention
seed(0)
xq = torch.randn(2, 5, 16)
attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
out, w = attn(xq, xq, xq)
kpm = torch.tensor([[False] * 5, [False, False, False, True, True]])
out_m, w_m = attn(xq, xq, xq, key_padding_mask=kpm)
out_n, w_n = attn(xq, xq, xq, key_padding_mask=kpm, need_weights=False)
causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), 1)
_, w_c = attn(xq, xq, xq, attn_mask=causal)
IO["d12-nn-multiheadattention"] = ("torch.randn(2, 5, 16) (seed 0) = [B=2, T=5, E=16]", [
    ("output, weights = attn(x, x, x)", f"output {sh(out)}, weights {sh(w)}", "weights = [B, Query T, Key T] (Head 평균)"),
    ("weights.sum(-1)[0]", " ".join(f"{v:.3f}" for v in w.sum(-1)[0].tolist()), "각 행은 확률 → 합 1"),
    ("key_padding_mask (샘플 1의 뒤 2칸 True) → weights[1, 0]", " ".join(f"{v:.3f}" for v in w_m[1, 0].tolist()), "막은 Key 3·4에는 0"),
    ("need_weights=False → weights", repr(w_n), "가중치를 안 돌려줌"),
    ("attn_mask = triu(ones(5, 5), 1) → weights[0]", mat(w_c[0], "{:.2f}"), "Causal: 자기보다 뒤(미래)는 0"),
    ("sum(p.numel() for p in attn.parameters())", f"{n_params(attn):,}", "Q·K·V 투영 16×48 + 48, 출력 16×16 + 16"),
    ("nn.MultiheadAttention(embed_dim=10, num_heads=4)", "AssertionError", "10은 4로 안 나누어떨어짐"),
])
try:
    nn.MultiheadAttention(embed_dim=10, num_heads=4)
except AssertionError:
    pass
else:
    raise SystemExit("embed_dim 오류가 안 났음")

# 10. TransformerEncoderLayer
seed(0)
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.1, batch_first=True)
xt = torch.randn(2, 5, 16)
encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
big = nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True)
layer.train()
d_train = (layer(xt) - layer(xt)).abs().max().item()
layer.eval()
d_eval = (layer(xt) - layer(xt)).abs().max().item()
IO["d12-nn-transformerencoderlayer"] = ("torch.randn(2, 5, 16) (seed 0)", [
    ("layer(x).shape", sh(layer(xt)), "입력과 같은 [B, T, E]"),
    ("encoder(x).shape  (num_layers=2)", sh(encoder(xt)), "몇 층을 쌓아도 Shape 유지"),
    ("Layer 파라미터 (dim_feedforward=32)", f"{n_params(layer):,}", "Attention 1,088 + FFN 1,072 + LayerNorm 64"),
    ("Layer 파라미터 (dim_feedforward 기본 2048)", f"{n_params(big):,}", "FFN이 대부분 — d_model 16엔 너무 큼"),
    ("Encoder 2층 파라미터", f"{n_params(encoder):,}", "Layer를 복제해서 정확히 2배"),
    ("train 모드: layer(x) 두 번의 차이", f"{d_train:.3f}", "dropout 0.1이 매번 다르게 꺼서 결과가 흔들림"),
    ("eval 모드: layer(x) 두 번의 차이", f"{d_eval:.3f}", "dropout 꺼짐 → 결정적"),
])


# 11. Position + Mask + masked_mean
def masked_mean(x, padding_mask):
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("실제 길이는 1 이상")
    valid = (~padding_mask).unsqueeze(-1)
    summed = x.masked_fill(~valid, 0).sum(dim=1)
    return summed / valid.sum(dim=1)


class TransformerClassifier(nn.Module):
    def __init__(self, max_len=64, d_model=16, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x, padding_mask):
        pos = torch.arange(x.size(1), device=x.device)
        h = x + self.pos(pos) if self.use_pos else x
        z = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.head(masked_mean(z, padding_mask))


seed(0)
tc = TransformerClassifier().eval()
xs = torch.randn(3, 6, 16)
pm = torch.arange(6)[None, :] >= torch.tensor([6, 4, 2])[:, None]
x99 = xs.masked_fill(pm.unsqueeze(-1), 99.0)
with torch.no_grad():
    a = tc(xs, pm)
    b = tc(x99, pm)
    perm = torch.tensor([5, 4, 3, 2, 1, 0])
    full = torch.zeros(3, 6, dtype=torch.bool)
    p_on = (tc(xs, full) - tc(xs[:, perm], full)).abs().max().item()
    seed(0)
    nopos = TransformerClassifier(use_pos=False).eval()
    p_off = (nopos(xs, full) - nopos(xs[:, perm], full)).abs().max().item()
    z_out = tc.encoder(xs + tc.pos(torch.arange(6)), src_key_padding_mask=pm)
    z_out99 = tc.encoder(x99 + tc.pos(torch.arange(6)), src_key_padding_mask=pm)
try:
    masked_mean(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool))
    err = "오류 없음"
except ValueError as e:
    err = f"ValueError: {e}"
IO["d12-position-embedding-padding-mask-실제-위치만-평균"] = ("torch.randn(3, 6, 16) (seed 0), 길이 [6, 4, 2], 학습 전 모델 · eval", [
    ("model(x, padding_mask).shape", sh(a), "[B, 2] logits"),
    ("Padding 칸을 99로 바꾼 뒤 출력 차이", f"{(a - b).abs().max():.2e}", "≈ 0 → Mask가 잘 걸림 (Padding 불변)"),
    ("encoder 출력의 Padding 위치는?  (99로 바꾼 뒤 차이)", f"{(z_out - z_out99)[pm].abs().max():.2f}", "Padding 위치 출력은 남아 있고 값도 변함 → masked_mean으로 빼야 함"),
    ("Position 있음: 순서를 뒤집은 뒤 출력 차이", f"{p_on:.3f}", "순서가 바뀌면 결과도 바뀜"),
    ("Position 없음: 순서를 뒤집은 뒤 출력 차이", f"{p_off:.2e}", "≈ 0 → 순서를 모르는 모델"),
    ("masked_mean(z, 전부 True인 mask)", err, "0 ÷ 0 = NaN을 미리 막음"),
    ("model.pos(torch.arange(65))  ← max_len 64 초과", "IndexError", "없는 위치 번호"),
])
try:
    tc.pos(torch.arange(65))
except IndexError:
    pass
else:
    raise SystemExit("max_len 오류가 안 났음")


# 12. 시간 재기
class LSTMClf(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x, lengths):
        p = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(p)
        return self.fc(h[-1])


def bench(device, T, B=64):
    seed(0)
    model = LSTMClf().to(device).eval()
    x = torch.randn(B, T, 3, device=device)
    lengths = torch.randint(T // 2, T + 1, (B,))
    with torch.inference_mode():
        for _ in range(2):
            model(x, lengths)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
        start = time.perf_counter()
        for _ in range(20):
            model(x, lengths)
        if x.device.type == "cuda":
            torch.cuda.synchronize(x.device)
        seconds = time.perf_counter() - start
    peak = f"{torch.cuda.max_memory_allocated(x.device) / 1024**2:.1f} MiB" if x.device.type == "cuda" else "N/A"
    return len(x) * 20 / seconds, peak


rows = []
for T in (50, 400):
    c, cm = bench(torch.device("cpu"), T)
    rows.append((f"CPU · LSTM(3→32), B=64, T={T}", f"{c:,.0f} samples/s · peak {cm}", "CPU 메모리는 이 방식으로 안 잼"))
    if DEV.type == "cuda":
        g, gm = bench(DEV, T)
        rows.append((f"GPU · 같은 조건 T={T}", f"{g:,.0f} samples/s · peak {gm}", f"CPU의 {g / c:.1f}배"))
IO["d12-한-batch-순전파를-반복해-시간-재기"] = (f"seed 0 난수 [64, T, 3] · 길이는 T/2~T · pack 포함 · 워밍업 2회 후 20회 · {GPU} — 이 PC 기준 숫자예요", rows + [
    ("(해석)", "T가 8배 길어지면 처리량이 크게 줄어듦", "LSTM은 시점을 하나씩 차례로 계산해서 T에 비례해 느려짐. 이 작은 LSTM(H=32)·B=64에선 GPU가 CPU와 비슷하거나 오히려 느릴 때도 있어요 — 시점마다 작은 계산을 띄우는 오버헤드가 커서"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day12.html"
s = frag.read_text(encoding="utf-8")
s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
for cid, (title, rows) in IO.items():
    m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</code></pre>\n', s, re.S)
    assert m, cid
    s = s[:m.end()] + box(title, rows) + "\n" + s[m.end():]
frag.write_text(s, encoding="utf-8")
for cid, (title, rows) in IO.items():
    print(f"== {cid}  [{title}]")
    for i, o, n in rows:
        print(f"   {i}  →  {o}   ({n})")
shutil.rmtree(TMP, ignore_errors=True)
