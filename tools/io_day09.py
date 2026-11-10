"""Day9 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

데이터는 노트 코드와 같은 seed 고정 합성 Tensor(torch.randn)를 쓴다 — 파일을 읽거나 쓰지 않는다.
결과는 cmds/day09.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day09.py
"""
import html
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")


def r(v, nd=4):
    """출력값을 짧은 문자열로."""
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return f"{v.item():.{nd}g}"
        return np.array2string(v.detach().cpu().numpy(), precision=nd, separator=", ", suppress_small=True)
    if isinstance(v, float):
        return f"{v:.{nd}g}"
    return repr(v)


def err(fn):
    try:
        fn()
        return "오류 없음"
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()[0]
        return f"{type(e).__name__}: {msg[:90]}{'…' if len(msg) > 90 else ''}"


IO = {}
PC = "이 PC 기준 — 다른 PC에선 값이 달라요"

# 1. nvidia-smi
try:
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    name, drv, used, total, util = [s.strip() for s in q.splitlines()[0].split(",")]
    head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    m = re.search(r"CUDA Version:\s*([\d.]+)", head)
    smi_cuda = m.group(1) if m else "(표시 없음)"
    IO["d9-nvidia-smi"] = (PC, [
        ("GPU Name", name, "실제 장치 모델"),
        ("Driver Version", drv, "설치된 그래픽 드라이버"),
        ("CUDA Version (표 오른쪽 위)", smi_cuda, "드라이버가 지원하는 상한 — PyTorch 빌드 CUDA와 비교해 봐요"),
        ("Memory-Usage", f"{used} / {total}", "지금 쓰는 VRAM / 전체 VRAM"),
        ("GPU-Util", util, "실행 순간 GPU가 얼마나 바쁜지"),
    ])
except Exception as e:  # noqa: BLE001
    IO["d9-nvidia-smi"] = (PC, [("nvidia-smi", f"실행 실패: {e}", "드라이버가 없거나 PATH에 없음")])

# 2. PyTorch GPU 확인
rows = [
    ("torch.__version__", torch.__version__, "+cu130 = CUDA 13.0용 빌드"),
    ("torch.version.cuda", repr(torch.version.cuda), "CPU 빌드였다면 None"),
    ("torch.cuda.is_available()", repr(CUDA), "지금 이 환경에서 실제로 쓸 수 있나"),
]
if CUDA:
    p = torch.cuda.get_device_properties(0)
    rows += [
        ("torch.cuda.get_device_name(0)", repr(torch.cuda.get_device_name(0)), "nvidia-smi의 GPU Name과 같아요"),
        ("round(props.total_memory / 1024**3, 1)", repr(round(p.total_memory / 1024**3, 1)), "바이트 → GB (GiB)"),
        ("torch.cuda.device_count()", repr(torch.cuda.device_count()), "GPU 1개 → 번호는 0만 유효"),
    ]
IO["d9-pytorch에서-gpu-확인"] = (PC, rows)

# 3. CPU·GPU 패턴
torch.manual_seed(0)
m3 = nn.Linear(2, 1)
before = next(m3.parameters()).device
m3 = m3.to(device)
X3 = torch.randn(4, 2)
IO["d9-cpu-gpu-둘-다-되는-코드-패턴"] = (PC + " · nn.Linear(2, 1), X = torch.randn(4, 2)", [
    ('torch.device("cuda" if torch.cuda.is_available() else "cpu")', repr(device), "GPU가 보여서 cuda가 골라짐 (없으면 cpu)"),
    ("next(model.parameters()).device  (to 전 → 후)", f"{before} → {next(m3.parameters()).device}", "모델은 to() 한 줄로 파라미터가 전부 이동"),
    ("X.to(device).device", str(X3.to(device).device), "X 자체는 그대로 cpu — 다시 담아야 해요"),
    ("model(X.to(device)).shape", str(tuple(m3(X3.to(device)).shape)), "모델과 입력이 같은 장치라 계산됨"),
])

# 4. torch.tensor
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.long)
a = np.array([1.0, 2.0])
t = torch.tensor(a)
a[0] = 99
src = torch.ones(2, requires_grad=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    wrapped = torch.tensor(src)
IO["d9-torch-tensor"] = ("노트 코드 그대로", [
    ("x.shape, x.dtype, x.device", f"{tuple(x.shape)}, {x.dtype}, {x.device}", "2×2 실수, CPU에 만들어짐"),
    ("y.dtype", str(y.dtype), "torch.long = int64 (클래스 번호)"),
    ("torch.tensor([0, 1]).dtype   (dtype 생략)", str(torch.tensor([0, 1]).dtype), "정수 list면 알아서 int64"),
    ("torch.tensor([0.5, 1]).dtype", str(torch.tensor([0.5, 1]).dtype), "실수가 섞이면 float32"),
    ("a = np.array([1., 2.]); t = torch.tensor(a); a[0] = 99 → t", f"{r(t)} (dtype {t.dtype})", "값을 복사해서 원본을 바꿔도 그대로. NumPy float64는 float64로 옴"),
    ("torch.tensor(src).requires_grad   (src는 requires_grad=True)", f"{wrapped.requires_grad} (경고 {len(w)}개)", "다시 감싸면 그래프가 끊기고 UserWarning이 떠요"),
])

# 5. randn / zeros / ones
torch.manual_seed(42)
a5 = torch.randn(2, 3)
torch.manual_seed(42)
a5b = torch.randn(2, 3)
big = torch.randn(100_000)
IO["d9-난수-0-1로-채우기"] = ("torch.manual_seed(42) 직후", [
    ("torch.randn(2, 3)", r(a5), "seed 42에서 이 PC가 실제로 낸 값"),
    ("seed 다시 42 → torch.randn(2, 3) 와 같나?", repr(torch.equal(a5, a5b)), "같은 seed면 같은 난수"),
    ("torch.zeros(2, 3).sum().item()", repr(torch.zeros(2, 3).sum().item()), ""),
    ("torch.ones(2, 3).mean().item()", repr(torch.ones(2, 3).mean().item()), ""),
    ("x = torch.randn(100000) → x.mean(), x.std()", f"{big.mean().item():.4f}, {big.std().item():.4f}", "많이 뽑으면 평균 ≈ 0, 표준편차 ≈ 1"),
])

# 6. .to(device)
torch.manual_seed(0)
X6 = torch.randn(4, 2)
X6g = X6.to(device)
m6 = nn.Linear(2, 1).to(device)
out6 = m6(X6g)
IO["d9-to-device"] = (PC + " · X = torch.randn(4, 2)", [
    ("X.to(device) 후 X.device", str(X6.device), "원본은 안 바뀜 — 새 Tensor를 돌려줌"),
    ("X = X.to(device) → X.device", str(X6g.device), "다시 담아야 바뀜"),
    ("output = model(X) → output.device", str(out6.device), "입력·모델과 같은 장치"),
    ('X.to("cpu") is X', repr(X6.to("cpu") is X6), "이미 cpu면 복사 없이 자기 자신 (copy=False)"),
    ("X.to(torch.float64).dtype", str(X6.to(torch.float64).dtype), "dtype만 바꾸기도 가능"),
    ("cuda 모델에 cpu Tensor 넣기  model(X_cpu)", err(lambda: m6(X6)) if CUDA else "(GPU 없음)", "장치를 섞으면 바로 오류"),
])

# 7. backward
x7 = torch.tensor(3.0, requires_grad=True)
y7 = x7**2 + 2 * x7 + 1
ret = y7.backward()
g1 = x7.grad.item()
(x7**2 + 2 * x7 + 1).backward()
g2 = x7.grad.item()
IO["d9-backward"] = ("x = 3, y = x² + 2x + 1", [
    ("y.item()", repr(y7.item()), "9 + 6 + 1"),
    ("y.backward() 의 반환값", repr(ret), "값은 안 돌려주고 .grad를 채움"),
    ("x.grad.item()", repr(g1), "dy/dx = 2x + 2 = 8"),
    ("다시 순전파 + backward → x.grad.item()", repr(g2), "zero_grad 없이 또 하면 8 + 8 = 16 (누적!)"),
    ("y.requires_grad, y.grad_fn", f"{y7.requires_grad}, {type(y7.grad_fn).__name__}", "y는 x로부터 계산된 값이라 그래프 노드"),
])

# 8. 손으로 5번 갱신
def manual(lr, steps=5, zero=True):
    w = torch.tensor(0.0, requires_grad=True)
    target = torch.tensor(4.0)
    path = []
    for _ in range(steps):
        loss = (w - target) ** 2
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
        if zero:
            w.grad.zero_()
        path.append(round(w.item(), 4))
    return path


IO["d9-손으로-5번-갱신하기"] = ("loss = (w − 4)², w = 0에서 시작", [
    ("lr = 0.1, 5 step의 w", " → ".join(map(str, manual(0.1))), "남은 거리가 매번 ×0.8"),
    ("lr = 0.01", " → ".join(map(str, manual(0.01))), "너무 작으면 거의 안 움직임"),
    ("lr = 1.1", " → ".join(map(str, manual(1.1))), "1보다 크면 4를 넘어가며 점점 멀어짐(발산)"),
    ("lr = 0.1인데 w.grad.zero_() 빠뜨림", " → ".join(map(str, manual(0.1, zero=False))), "기울기가 쌓여 4를 지나쳐 버림"),
])

# 9. no_grad / detach
torch.manual_seed(0)
m9 = nn.Linear(2, 2)
X9 = torch.randn(4, 2)
lg_train = m9(X9)
m9.eval()
with torch.no_grad():
    lg = m9(X9)
probs = torch.softmax(lg_train, dim=1).detach().cpu()
IO["d9-torch-no-grad"] = ("nn.Linear(2, 2), X = torch.randn(4, 2)", [
    ("model(X).requires_grad  (그냥 호출)", repr(lg_train.requires_grad), "파라미터가 학습 대상이라 기록됨"),
    ("with torch.no_grad(): model(X).requires_grad", repr(lg.requires_grad), "블록 안에선 기록 안 함"),
    ("softmax(logits).detach().requires_grad", repr(probs.requires_grad), "그래프에서 떼어 낸 값"),
    ("probs.sum(dim=1)", r(probs.sum(dim=1)), "softmax라 각 행 합이 1"),
    ("logits.numpy()  (detach 없이)", err(lambda: lg_train.numpy()), "기록 중인 Tensor는 NumPy로 못 바꿈 → detach 필요"),
])

# 10. Sequential ANN
torch.manual_seed(0)
m10 = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
X10 = torch.randn(4, 2)
lg10 = m10(X10)
IO["d9-2-8-2-ann-만들기"] = ("torch.manual_seed(0), X = torch.randn(4, 2)", [
    ("logits.shape", str(tuple(lg10.shape)), "(batch 4, class 2)"),
    ("logits[0]", r(lg10[0]), "확률이 아니라 원시 점수 — 음수도 나오고 합이 1도 아님"),
    ("model[0].weight.shape, model[0].bias.shape", f"{tuple(m10[0].weight.shape)}, {tuple(m10[0].bias.shape)}", "weight는 (out, in)"),
    ("sum(p.numel() for p in model.parameters())", repr(sum(p.numel() for p in m10.parameters())), "2×8+8 + 8×2+2"),
    ("(model[1](model[0](X)) >= 0).all()", repr(bool((m10[1](m10[0](X10)) >= 0).all())), "ReLU 뒤엔 음수가 없음"),
    ("model(torch.randn(4, 3))", err(lambda: m10(torch.randn(4, 3))), "입력 마지막 차원이 in_features(2)와 다르면 오류"),
])

# 11. nn.Module
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


class ListANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 8), nn.Linear(8, 2)]  # 그냥 list


class NoSuper(nn.Module):
    def __init__(self):
        self.net = nn.Linear(2, 2)


torch.manual_seed(0)
m11 = SimpleANN()
IO["d9-nn-module"] = ("SimpleANN() 과 일부러 틀린 두 클래스", [
    ("model(X).shape   (X: (4, 2))", str(tuple(m11(X10).shape)), "model(X)가 forward를 불러 줌"),
    ("[n for n, _ in model.named_parameters()]", repr([n for n, _ in m11.named_parameters()]), "self.net 아래 층이 자동 등록"),
    ("층을 파이썬 list에 넣은 모델의 파라미터 수", repr(sum(p.numel() for p in ListANN().parameters())), "0개! 등록이 안 돼 Optimizer가 못 봐요"),
    ("super().__init__() 빼먹은 모델 만들기", err(lambda: NoSuper()), "층을 등록할 장부가 없어 오류"),
])

# 12. CrossEntropyLoss
torch.manual_seed(0)
m12 = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
X12 = torch.randn(4, 2)
y12 = torch.tensor([0, 1, 1, 0], dtype=torch.long)
lg12 = m12(X12)
ce = nn.CrossEntropyLoss()
IO["d9-nn-crossentropyloss"] = ("학습 전 2→8→2 모델(seed 0), X = randn(4, 2), y = [0, 1, 1, 0]", [
    ("loss_fn(logits, y)", r(ce(lg12, y12)), "학습 전이라 ln 2 ≈ 0.693 근처 (반반 찍는 수준)"),
    ("nn.CrossEntropyLoss(reduction='none')(logits, y)", r(nn.CrossEntropyLoss(reduction="none")(lg12, y12)), "샘플별 loss 4개, 평균이 위 값"),
    ("loss_fn(torch.tensor([[10., 0.]]), torch.tensor([0]))", r(ce(torch.tensor([[10.0, 0.0]]), torch.tensor([0]))), "정답 점수가 크게 앞서면 거의 0"),
    ("loss_fn(torch.tensor([[10., 0.]]), torch.tensor([1]))", r(ce(torch.tensor([[10.0, 0.0]]), torch.tensor([1]))), "확신하고 틀리면 아주 큼"),
    ("loss_fn(torch.softmax(torch.tensor([[10., 0.]]), 1), torch.tensor([0]))", r(ce(torch.softmax(torch.tensor([[10.0, 0.0]]), 1), torch.tensor([0]))),
     "softmax를 먼저 하면 확실히 맞혀도 0.31에서 더 못 내려감 (원래는 0.00005)"),
    ("loss_fn(logits, y.float())", err(lambda: ce(lg12, y12.float())), "(batch,) 정답을 float로 주면 오류 — long이어야 해요"),
])

# 13. TensorDataset / DataLoader
torch.manual_seed(42)
X13 = torch.randn(200, 2)
y13 = ((X13[:, 0] + 0.8 * X13[:, 1]) > 0).long()
ds = TensorDataset(X13, y13)
ld = DataLoader(ds, batch_size=32, shuffle=True)
xb, yb = next(iter(ld))
sizes = [len(b[0]) for b in ld]
IO["d9-tensordataset"] = ("torch.manual_seed(42), X = randn(200, 2)", [
    ("len(TensorDataset(X, y))", repr(len(ds)), "샘플 수"),
    ("ds[0]", f"({r(ds[0][0])}, {r(ds[0][1])})", "같은 번호의 X와 y가 짝으로 나옴"),
    ("y.float().mean()", r(y13.float().mean()), "클래스 1 비율 — 대략 반반"),
    ("xb.shape, yb.shape, yb.dtype", f"{tuple(xb.shape)}, {tuple(yb.shape)}, {yb.dtype}", "batch 하나"),
    ("len(loader), 각 batch 크기", f"{len(ld)}, {sizes}", "200 = 32×6 + 8 → 마지막은 8개"),
    ("len(DataLoader(ds, batch_size=32, drop_last=True))", repr(len(DataLoader(ds, batch_size=32, drop_last=True))), "작은 마지막 batch를 버림"),
])

# 14. 준비 + batch 반복 (노트 코드 그대로, 이 PC의 device)
torch.manual_seed(42)
X = torch.randn(200, 2)
y = ((X[:, 0] + 0.8 * X[:, 1]) > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
w_before = model[0].weight.detach().clone()
model.eval()
with torch.no_grad():
    acc0 = (model(X.to(device)).argmax(1).cpu() == y).float().mean().item()
steps = 0
first_loss = None
for epoch in range(80):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        optimizer.step()
        steps += 1
last_loss = loss.item()
model.eval()
with torch.no_grad():
    pred = model(X.to(device)).argmax(dim=1)
    acc = (pred.cpu() == y).float().mean().item()
IO["d9-준비-batch-반복"] = (f"노트 코드 그대로 (seed 42, 직선 경계 데이터 200개, device={device})", [
    ("학습 전 정확도", f"{acc0:.3f}", "무작위 가중치라 찍기 수준"),
    ("첫 batch loss → 마지막 batch loss", f"{first_loss:.4f} → {last_loss:.4f}", "batch마다 step하며 줄어듦"),
    ("optimizer.step() 호출 수", repr(steps), "7 batch × 80 epoch"),
    ("model[0].weight 가 바뀐 양 (평균 절댓값)", f"{(model[0].weight.detach().cpu() - w_before.cpu()).abs().mean().item():.4f}", "step이 실제로 가중치를 움직임"),
    ("acc (train 데이터 전체)", f"{acc:.3f}", "직선으로 갈리는 쉬운 데이터라 200개 모두 맞힘 — 학습한 데이터로 잰 값이라 일반화 성능은 아님"),
    ("pred.device", str(pred.device), "비교 전에 .cpu()로 정답과 장치를 맞춤"),
])

# 15. epoch 평균 loss
torch.manual_seed(42)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)


def train_one_epoch(model, loader, loss_fn, optimizer, device, log):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        log.append((loss.item(), len(xb)))
    return total_loss / len(loader.dataset)


log = []
ep1 = train_one_epoch(model, loader, loss_fn, optimizer, device, log)
naive = sum(l for l, _ in log) / len(log)
ep2 = train_one_epoch(model, loader, loss_fn, optimizer, device, [])
IO["d9-epoch-평균-loss를-제대로-내기"] = ("같은 데이터 200개, batch 32 (마지막 batch 8개)", [
    ("epoch 1 batch 크기", repr([n for _, n in log]), ""),
    ("epoch 1 마지막 batch loss", f"{log[-1][0]:.4f}", "8개짜리 batch의 평균"),
    ("train_one_epoch(...)  (샘플 가중 평균)", f"{ep1:.4f}", "샘플 하나하나의 평균"),
    ("batch loss들을 그냥 평균", f"{naive:.4f}", "8개 batch가 32개와 같은 무게라 살짝 다름"),
    ("epoch 2 train_one_epoch(...)", f"{ep2:.4f}", "다음 epoch엔 더 내려감"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day09.html"
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
