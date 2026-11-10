"""Day11 'PyTorch 한 장 정리'(치트시트) 사전 항목 6개의 '입력 → 출력' 상자.

FashionMNIST는 내려받지 않고, 같은 모양(1×28×28, 10클래스)의 합성 데이터로 실제 실행한다.
상자 class 는 "io io-cheat" (io_day11.py 가 지우는 class="io" 와 구분).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day11_cheat.py
"""
import html
import io as _io
import re
import sys
import tempfile
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
torch.manual_seed(11)
IO = {}
device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        return self.layers(x)


class MyDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_data(n):
    """클래스 k면 k번째 가로 띠(약 3줄)가 밝은 28×28 합성 사진 — 배울 수 있는 규칙이 있음."""
    y = torch.randint(0, 10, (n,))
    X = torch.rand(n, 1, 28, 28) * 0.3
    for i in range(n):
        r = int(y[i]) * 2 + 4
        X[i, 0, r:r + 3, :] += 0.7
    return X.clamp(0, 1), y


# ---- 1. 전체 흐름
Xtr, ytr = make_data(1280)
Xte, yte = make_data(256)
train_dl = DataLoader(MyDS(Xtr, ytr), batch_size=64, shuffle=True)
test_dl = DataLoader(MyDS(Xte, yte), batch_size=64)
model = Net().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-1)


def test(m):
    m.eval(); correct = 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            correct += (m(X).argmax(1) == y).sum().item()
    return correct / len(test_dl.dataset)


acc0 = test(model)
hist = []
for epoch in range(3):
    model.train(); tot = 0
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        loss = loss_fn(model(X), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot += loss.item() * len(X)
    hist.append((round(tot / len(train_dl.dataset), 3), round(test(model), 3)))
steps_fm = len(DataLoader(range(60000), batch_size=64))
IO["d11-cheat-flow"] = ("FashionMNIST와 같은 모양의 합성 데이터 (학습 1,280 · 평가 256장, 클래스마다 밝은 띠 위치가 다름) · lr 0.1", [
    ("device", device, "이 PC는 NVIDIA GPU가 있어서 cuda (PC마다 달라요)"),
    ("len(DataLoader(60,000개, batch_size=64))", str(steps_fm), "FashionMNIST 한 epoch = 938번 step"),
    ("len(train_dl)  (1,280장)", str(len(train_dl)), "1,280 ÷ 64 = 20 묶음"),
    ("학습 전 test(model)", f"{acc0:.3f}", "학습 전이라 찍기 수준 (이번 무작위 시작값에선 기대값 0.1보다도 낮게 나왔어요)"),
    ("epoch 1·2·3 (학습 손실, 평가 정확도)", str(hist), "손실은 계속 줄고, 정확도는 첫 epoch부터 1.0 — 규칙이 쉬운 합성 데이터라서예요(실제 FashionMNIST는 이렇게 안 돼요)"),
])

# ---- 2. shape
x = torch.rand(64, 1, 28, 28)
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
pred = torch.randn(64, 10)
IO["d11-tensor-shape"] = ("무작위 배치 (64, 1, 28, 28) · 2×3 텐서", [
    ("x.reshape(64, -1).shape", str(tuple(x.reshape(64, -1).shape)), "-1 자리 = 1×28×28 = 784 자동 계산"),
    ("x.flatten(1).shape", str(tuple(x.flatten(1).shape)), "배치 축은 두고 나머지만 펴기"),
    ("x.flatten().shape", str(tuple(x.flatten().shape)), "0번부터 펴면 배치까지 한 줄로 — 보통 실수"),
    ("t.sum(dim=0)", str(t.sum(dim=0).tolist()), "줄 축(0)이 사라짐"),
    ("t.sum(dim=1)", str(t.sum(dim=1).tolist()), "칸 축(1)이 사라짐"),
    ("t.sum(dim=1, keepdim=True).shape", str(tuple(t.sum(dim=1, keepdim=True).shape)), "축을 길이 1로 남김"),
    ("pred.argmax(dim=1).shape, [:5]", f"{tuple(pred.argmax(dim=1).shape)}, {pred.argmax(dim=1)[:5].tolist()}", "사진마다 클래스 번호 하나"),
])
try:
    x.reshape(100, -1)
    rs_err = "에러 없음"
except RuntimeError as e:
    rs_err = str(e).split("\n")[0][:70]
IO["d11-tensor-shape"][1].append(("x.reshape(100, -1)", rs_err, "50,176개를 100줄로 못 나눠요"))

# ---- 3. Dataset
ds = MyDS(*make_data(150))
dl = DataLoader(ds, batch_size=64, shuffle=True)
Xb, yb = next(iter(dl))
sizes = [len(b[1]) for b in dl]
IO["d11-dataset-class"] = ("합성 사진 150장", [
    ("len(ds)", str(len(ds)), "__len__ 이 돌려준 값"),
    ("ds[0][0].shape, ds[0][1]", f"{tuple(ds[0][0].shape)}, {int(ds[0][1])}", "__getitem__(0): 샘플 1개"),
    ("X, y = next(iter(dl)) → shape", f"{tuple(Xb.shape)}, {tuple(yb.shape)}", "64개가 앞에 붙은 배치"),
    ("len(dl), 배치 크기들", f"{len(dl)}, {sizes}", "150 = 64 + 64 + 22 → 마지막은 남는 만큼"),
    ("y.dtype", str(yb.dtype), "CrossEntropyLoss가 원하는 정수(long)"),
])

# ---- 4. Net
net = Net()
n_params = sum(p.numel() for p in net.parameters())
IO["d11-net-module"] = ("무작위 배치 (64, 1, 28, 28)", [
    ("model(x).shape", str(tuple(net(x).shape)), "클래스 10개 점수(로짓)"),
    ("[p.shape for p in model.parameters()]", str([tuple(p.shape) for p in net.parameters()]), "Linear마다 weight (out, in) + bias (out,)"),
    ("sum(p.numel() for p in model.parameters())", f"{n_params:,}", "784×512 + 512 + 512×10 + 10"),
    ("model.layers[0](x).shape", str(tuple(net.layers[0](x).shape)), "Flatten 지난 직후"),
])

# ---- 5. 손실 · 최적화 + 흔한 실수
torch.manual_seed(5)
logits = torch.tensor([[4.0, 0.5, -1.0], [0.2, 3.5, 0.1]])
yt = torch.tensor([0, 1])
ce = nn.CrossEntropyLoss()
good = ce(logits, yt).item()
bad = ce(logits.softmax(dim=1), yt).item()
perfect_bad = ce(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), yt).item()
lin = nn.Linear(3, 1)
xx = torch.ones(4, 3); tt = torch.zeros(4, 1)
g = []
for _ in range(3):
    nn.MSELoss()(lin(xx), tt).backward()
    g.append(round(lin.weight.grad.norm().item(), 3))
lin2 = nn.Linear(3, 1); opt = optim.SGD(lin2.parameters(), lr=0.1)
w0 = lin2.weight.detach().clone()
l2 = nn.MSELoss()(lin2(xx), tt); opt.zero_grad(); l2.backward()
gw = lin2.weight.grad.clone(); opt.step()
sgd_ok = torch.allclose(lin2.weight.detach(), w0 - 0.1 * gw)
dev_msg = "GPU가 없어 확인 못 함"
if torch.cuda.is_available():
    try:
        nn.Linear(3, 1).to("cuda")(torch.ones(1, 3))
        dev_msg = "에러 없음"
    except RuntimeError as e:
        dev_msg = str(e).split("\n")[0][:80]
IO["d11-loss-optim"] = ("작은 예제 (로짓 2개 · Linear(3, 1))", [
    ("CrossEntropyLoss()(로짓, y)", f"{good:.3f}", "정답 점수가 가장 커서 손실이 작아요"),
    ("CrossEntropyLoss()(softmax(로짓), y)  ← 실수", f"{bad:.3f}", "같은 예측인데 손실이 커져요 — softmax가 두 번 들어가 차이가 뭉개짐"),
    ("softmax 값이 완벽한 [1, 0, 0]이어도", f"{perfect_bad:.3f}", "0이 못 돼요 → 아무리 맞혀도 손실이 이 아래로 안 내려감"),
    ("zero_grad 없이 backward 3번 → 기울기 크기", str(g), "더해져서 1배·2배·3배로 쌓여요"),
    ("SGD step 후 p == p − lr × p.grad ?", str(sgd_ok), "SGD 한 번 = 정확히 이 식"),
    ("모델은 cuda, 입력은 cpu → model(x)", dev_msg, "장치가 다르면 바로 에러"),
])

# ---- 6. test · state_dict
acc_eval = test(model)
nb, nd = len(test_dl), len(test_dl.dataset)
sd = model.state_dict()
buf = _io.BytesIO(); torch.save(sd, buf); size_kb = buf.tell() / 1024
tmp = Path(tempfile.mkdtemp()) / "model.pth"
torch.save(sd, tmp)
m2 = Net().to(device); res = m2.load_state_dict(torch.load(tmp, weights_only=True))
same = test(m2) == acc_eval
drop = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 1)); xi = torch.ones(1, 4)
drop.train(); a1, a2 = drop(xi).item(), drop(xi).item()
drop.eval(); b1, b2 = drop(xi).item(), drop(xi).item()
IO["d11-test-save"] = ("위 '전체 흐름'에서 3 epoch 학습한 모델 · 평가 256장", [
    ("test(model)", f"{acc_eval:.3f}", "eval() + no_grad() 로 잰 정확도"),
    ("len(test_dl), len(test_dl.dataset)", f"{nb}, {nd}", "배치 수로 나누면 정확도가 64배 커져요"),
    ("list(model.state_dict().keys())", str(list(sd.keys())), "층 번호.weight / .bias"),
    ("state_dict 저장 크기", f"{size_kb:,.0f} KB", "가중치 407,050개 × 4바이트"),
    ("새 Net()에 load_state_dict → test", f"{res} · 같은 정확도: {same}", "구조는 코드로, 값만 채워 넣기"),
    ("Dropout 모델 train() 두 번 / eval() 두 번", f"{a1:.3f}, {a2:.3f} / {b1:.3f}, {b2:.3f}", "eval()을 빠뜨리면 평가할 때마다 값이 흔들려요"),
])


# ---------------------------------------------------------------- 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io io-cheat">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


f = ROOT / "cmds" / "day11.html"
s = f.read_text(encoding="utf-8")
for cid, (title, rows) in IO.items():
    sec = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
    assert sec, cid
    body = re.sub(r'<div class="io io-cheat">.*?\n</div>\n', '', sec.group(0), flags=re.S)
    m = re.search(r'</code></pre>\n', body)
    body = body[:m.end()] + box(title, rows) + "\n" + body[m.end():]
    s = s[:sec.start()] + body + s[sec.end():]
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i} → {o}   ({n})")
f.write_text(s, encoding="utf-8")
