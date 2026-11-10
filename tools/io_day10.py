"""Day10 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261030/02_응용/data/points.csv, train 240 · valid 80 · test 80)는
임시 폴더에 복사해서 읽기만 한다. 학습은 재현을 위해 CPU, seed 고정.
결과는 cmds/day10.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day10.py
"""
import copy
import html
import re
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261030/02_응용/data/points.csv")
TMP = Path(tempfile.mkdtemp())
shutil.copy(SRC, TMP / "points.csv")
df = pd.read_csv(TMP / "points.csv")


def split(name):
    d = df[df["split"] == name]
    return (torch.tensor(d[["x1", "x2"]].values, dtype=torch.float32),
            torch.tensor(d["label"].values, dtype=torch.long))


X_tr, y_tr = split("train")
X_va, y_va = split("valid")
train_ds, valid_ds = TensorDataset(X_tr, y_tr), TensorDataset(X_va, y_va)
loss_fn = nn.CrossEntropyLoss()
DATA = "points.csv (train 240 · valid 80)"


def Net(width=32, dropout=0.0, bn=False):
    layers = [nn.Linear(2, width)]
    if bn:
        layers.append(nn.BatchNorm1d(width))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width)]
    if bn:
        layers.append(nn.BatchNorm1d(width))
    layers += [nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 2)]
    return nn.Sequential(*layers)


def loader(ds=train_ds, seed=43, bs=32):
    return DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed))


def train_one_epoch(model, ld, optimizer):
    model.train()
    tot = 0.0
    for xb, yb in ld:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        tot += loss.item() * len(xb)
    return tot / len(ld.dataset)


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        lg = model(X)
        return loss_fn(lg, y).item(), (lg.argmax(1) == y).float().mean().item()


def run(make_opt, epochs=20, seed=42, net_kw=None, ld=None):
    torch.manual_seed(seed)
    model = Net(**(net_kw or {}))
    opt = make_opt(model)
    ld = ld or loader()
    hist = {"train_loss": [], "valid_loss": [], "valid_acc": []}
    for _ in range(epochs):
        hist["train_loss"].append(train_one_epoch(model, ld, opt))
        vl, va = evaluate(model, X_va, y_va)
        hist["valid_loss"].append(vl)
        hist["valid_acc"].append(va)
    return model, opt, hist


IO = {}
f4 = lambda v: f"{v:.4f}"

# 1. Optimizer 바꿔 끼우기
res = {}
for name, mk in [
    ("SGD(lr=0.05, momentum=0)", lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.0)),
    ("SGD(lr=0.05, momentum=0.9)", lambda m: torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)),
    ("AdamW(lr=0.01, weight_decay=0.01)", lambda m: torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.01)),
]:
    model, opt, h = run(mk)
    res[name] = (h, opt)
rows = [(n, f"valid loss {f4(h['valid_loss'][-1])} · acc {h['valid_acc'][-1]:.3f}  (epoch 1 train loss {f4(h['train_loss'][0])})", "")
        for n, (h, _) in res.items()]
rows[0] = (*rows[0][:2], "momentum 없으면 20 epoch에 아직 덜 내려감")
rows[1] = (*rows[1][:2], "이전 방향을 누적해 같은 lr로도 훨씬 빨리 내려감")
rows[2] = (*rows[2][:2], "파라미터마다 보폭을 조절 — 이 데이터·설정에선 이렇게 나왔을 뿐, 항상 1등은 아님")
adam = res["AdamW(lr=0.01, weight_decay=0.01)"][1]
st = adam.state[adam.param_groups[0]["params"][0]]
rows.append(("sorted(optimizer.state[첫 weight].keys())  (AdamW)", repr(sorted(st.keys())), "step마다 쌓은 이동평균 — state_dict()에 같이 저장돼요"))
IO["d10-optimizer만-바꿔-끼우기"] = (DATA + " · 2→32→32→2, seed 42, 같은 batch 순서, 20 epoch 뒤 valid", rows)

# 2. 학습률 후보
rows = []
notes = {0.0001: "너무 작아 40 epoch 동안 거의 제자리", 0.01: "가장 낮은 valid loss, 마지막까지 안정",
         0.5: "최저값은 0.01과 거의 같지만 마지막엔 다시 올라감 — 큰 보폭으로 흔들림. 이 쉬운 데이터라 버틴 것"}
best = {}
for lr in [0.0001, 0.01, 0.5]:
    _, _, h = run(lambda m, lr=lr: torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01), epochs=40, seed=7)
    best[lr] = min(h["valid_loss"])
    rows.append((f"lr={lr} → min(history['valid_loss'])", f"{f4(best[lr])}  (마지막 epoch {f4(h['valid_loss'][-1])})", ""))
order = sorted(best, key=best.get)
for i, lr in enumerate([0.0001, 0.01, 0.5]):
    rank = order.index(lr) + 1
    rows[i] = (*rows[i][:2], f"{rank}등 — {notes[lr]}")
rows.append(("valid 기준으로 고른 lr", repr(order[0]), "test는 고르는 데 쓰지 않아요"))
IO["d10-학습률-후보-비교"] = (DATA + " · AdamW, 후보마다 seed 7로 새 모델, 40 epoch", rows)

# 3. ReduceLROnPlateau
opt = torch.optim.AdamW(Net().parameters(), lr=0.01)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-4)
seq = []
for v in [1.0] * 12:
    sch.step(v)
    seq.append(opt.param_groups[0]["lr"])
torch.manual_seed(42)
model = Net()
opt = torch.optim.AdamW(model.parameters(), lr=0.01)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4, min_lr=1e-4)
ld = loader()
changes, prev = [], 0.01
for ep in range(1, 61):
    train_one_epoch(model, ld, opt)
    vl, _ = evaluate(model, X_va, y_va)
    sch.step(vl)
    cur = opt.param_groups[0]["lr"]
    if cur != prev:
        changes.append(f"ep{ep}: {cur:g}")
        prev = cur
opt2 = torch.optim.AdamW(Net().parameters(), lr=0.01)
sch2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2, mode="min")
IO["d10-reducelronplateau"] = ("가짜 loss 1.0을 계속 넣은 경우 · " + DATA + "로 60 epoch 실제 학습한 경우", [
    ("valid_loss = 1.0 을 12번 step → 각 step 뒤 lr", ", ".join(f"{v:g}" for v in seq),
     "처음 값이 best, 그 뒤 5번째 '개선 없음'(patience 4 초과)에서 ×0.5, 다시 5번 뒤 또 ×0.5"),
    ("실제 학습: lr이 바뀐 epoch", ", ".join(changes) or "(바뀐 적 없음)", "valid loss가 정체할 때만 깎임"),
    ("optimizer.param_groups[0]['lr']  (학습 끝)", f"{opt.param_groups[0]['lr']:g}", "60 epoch 동안 세 번 절반씩 — min_lr(1e-4)까지는 아직 여유"),
    ("scheduler.step()  (값 없이)", "", ""),  # 아래에서 채움
])


def _try(s):
    try:
        s.step()
        return None
    except Exception as e:  # noqa: BLE001
        return e


e = _try(sch2)
IO["d10-reducelronplateau"][1][3] = ("scheduler.step()  (값 없이)",
                                     "오류 없음" if e is None else f"{type(e).__name__}: {str(e).splitlines()[0][:80]}",
                                     "감시할 값(metrics)이 필수 인자라 바로 오류")

# 4. DataLoader
tl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
vl_ = DataLoader(valid_ds, batch_size=32, shuffle=False)
torch.manual_seed(0)
first1 = next(iter(tl))[0][0]
first2 = next(iter(tl))[0][0]
v1 = next(iter(vl_))[0][0]
v2 = next(iter(vl_))[0][0]
IO["d10-dataloader"] = (DATA, [
    ("len(train_loader), 각 batch 크기", f"{len(tl)}, {[len(b[0]) for b in tl]}", "240 = 32×7 + 16 → 8 iteration / epoch"),
    ("len(DataLoader(train_dataset, 32, drop_last=True))", repr(len(DataLoader(train_ds, batch_size=32, drop_last=True))), "16개짜리 마지막 batch를 버림 → 그 16개는 그 epoch에 안 씀"),
    ("len(valid_loader), 각 batch 크기", f"{len(vl_)}, {[len(b[0]) for b in vl_]}", "80 = 32×2 + 16"),
    ("train: epoch 두 번의 첫 샘플이 같나?", repr(torch.equal(first1, first2)), "shuffle=True라 epoch마다 순서가 바뀜"),
    ("valid: 두 번 돌린 첫 샘플이 같나?", repr(torch.equal(v1, v2)), "shuffle=False라 순서 그대로"),
])

# 5. zero_grad / step
torch.manual_seed(42)
model = Net()
opt = torch.optim.SGD(model.parameters(), lr=0.05)
xb, yb = next(iter(loader()))
w0 = model[0].weight
opt.zero_grad(set_to_none=True)
g_before = w0.grad
loss = loss_fn(model(xb), yb)
loss.backward()
gshape = tuple(w0.grad.shape)
wcopy = w0.detach().clone()
opt.step()
after = loss_fn(model(xb), yb).item()
opt.zero_grad(set_to_none=False)
IO["d10-optimizer-zero-grad"] = (DATA + "의 batch 하나(32개), SGD lr=0.05", [
    ("zero_grad(set_to_none=True) 후 weight.grad", repr(g_before), "처음엔 기울기 자체가 없음"),
    ("loss.backward() 후 weight.grad.shape", repr(gshape), "파라미터와 같은 shape로 채워짐"),
    ("optimizer.step() 의 반환값 · weight 변화량(평균 절댓값)", f"None · {(w0.detach() - wcopy).abs().mean().item():.5f}", "step은 값을 돌려주지 않고 파라미터를 직접 고침"),
    ("같은 batch의 loss: step 전 → 후", f"{f4(loss.item())} → {f4(after)}", "한 번 갱신하니 그 batch loss가 줄어듦"),
    ("zero_grad(set_to_none=False) 후 weight.grad", f"{w0.grad.abs().sum().item()} (shape {tuple(w0.grad.shape)})", "None 대신 0으로 채운 Tensor"),
])

# 6. history — 작은 train(60건)에서 과적합 관찰
X60, y60 = X_tr[:60], y_tr[:60]
ld60 = loader(TensorDataset(X60, y60))
_, _, h = run(lambda m: torch.optim.AdamW(m.parameters(), lr=0.01, weight_decay=0.0), epochs=200, net_kw={"width": 64}, ld=ld60)
best_epoch = h["valid_loss"].index(min(h["valid_loss"])) + 1
IO["d10-epoch별-기록-남기기"] = ("points.csv train 앞 60건만 학습 · valid 80 · 2→64→64→2, AdamW lr=0.01, 200 epoch", [
    ("len(history['valid_loss'])", repr(len(h["valid_loss"])), "epoch마다 하나씩"),
    ("best_epoch", repr(best_epoch), "valid loss가 가장 낮았던 epoch"),
    ("min(history['valid_loss']) → 마지막 valid loss", f"{f4(min(h['valid_loss']))} → {f4(h['valid_loss'][-1])}", "best 뒤로 valid가 조금 다시 올라감 → 마지막 모델 ≠ 최선 모델 (경계가 쉬운 데이터라 과적합은 약하게만 보여요)"),
    ("train loss: 1 epoch → 마지막", f"{f4(h['train_loss'][0])} → {f4(h['train_loss'][-1])}", "train은 0에 붙도록 계속 내려감 — 60건을 외워 가는 중"),
    ("valid acc: best_epoch → 마지막", f"{h['valid_acc'][best_epoch - 1]:.3f} → {h['valid_acc'][-1]:.3f}", "loss는 올라도 정확도는 그대로 — 둘이 다른 말을 할 수 있어서 따로 기록"),
    ("pd.DataFrame(history).shape", repr(pd.DataFrame(h).shape), "CSV로 저장할 표: epoch × 기록 항목"),
])

# 7. manual_seed
torch.manual_seed(42)
a = Net()[0].weight.detach().clone()
torch.manual_seed(42)
b = Net()[0].weight.detach().clone()
c = Net()[0].weight.detach().clone()
o1 = [int(i) for i in next(iter(DataLoader(range(10), batch_size=10, shuffle=True, generator=torch.Generator().manual_seed(43))))]
o2 = [int(i) for i in next(iter(DataLoader(range(10), batch_size=10, shuffle=True, generator=torch.Generator().manual_seed(43))))]
IO["d10-torch-manual-seed"] = ("Net() 첫 층 weight · range(10) 섞기", [
    ("seed 42 → Net(), 다시 seed 42 → Net(): weight 같나?", repr(torch.equal(a, b)), "같은 seed면 같은 초기 가중치"),
    ("seed 다시 안 걸고 한 번 더 Net()", repr(torch.equal(b, c)), "난수 순서가 이어져서 다른 값"),
    ("Net()[0].weight[0]  (seed 42)", str([round(v, 4) for v in a[0].tolist()]), "이 PC·버전에서 실제로 나온 값"),
    ("DataLoader(..., generator=Generator().manual_seed(43)) 순서 ×2", f"{o1} / {o2}", "shuffle 순서는 Generator seed로 따로 고정"),
])

# 8. Dropout
torch.manual_seed(0)
dp = nn.Dropout(0.3)
ones = torch.ones(1, 10)
dtr = dp(ones)
big = dp(torch.ones(100_000))
dp.eval()
dev = dp(ones)
IO["d10-nn-dropout-p"] = ("nn.Dropout(p=0.3), 입력 torch.ones(1, 10)", [
    ("train 모드: dropout(x)", str([round(v, 4) for v in dtr[0].tolist()]), "꺼진 칸은 0, 남은 칸은 1/(1−0.3) ≈ 1.4286배"),
    ("train 모드: 10만 칸 중 0이 된 비율", f"{(big == 0).float().mean().item():.4f}", "정확히 30%가 아니라 확률적으로 약 30%"),
    ("train 모드: 10만 칸 평균", f"{big.mean().item():.4f}", "키워 준 덕에 평균이 1 근처로 유지"),
    ("eval 모드: dropout(x)", str([round(v, 4) for v in dev[0].tolist()]), "그대로 통과(identity)"),
    ("dropout(x).shape", repr(tuple(dtr.shape)), "입력 shape와 같음"),
])

# 9. BatchNorm1d
torch.manual_seed(0)
bn = nn.BatchNorm1d(4)
xb4 = torch.randn(32, 4) * 3 + 5
o = bn(xb4)
rm = bn.running_mean.clone()
bn.eval()
oe = bn(xb4)
bn.train()
IO["d10-nn-batchnorm1d"] = ("nn.BatchNorm1d(4), 입력 randn(32, 4) × 3 + 5 (평균 ≈ 5, 표준편차 ≈ 3)", [
    ("train: out.mean(0)", str([round(v, 3) for v in o.mean(0).tolist()]), "batch 통계로 정규화 → 특성마다 평균 0"),
    ("train: out.std(0)", str([round(v, 3) for v in o.std(0).tolist()]), "≈ 1 (std()는 n−1로 나눠서 1보다 살짝 큼)"),
    ("bn.running_mean  (batch 한 번 본 뒤)", str([round(v, 3) for v in rm.tolist()]), "0에서 출발해 momentum 0.1만큼만 batch 평균 쪽으로"),
    ("eval: out.mean(0)", str([round(v, 3) for v in oe.mean(0).tolist()]), "eval은 running 통계를 써서 아직 평균 0이 아님"),
    ("", "", ""),  # 아래 emsg로 채움
    ("", "", ""),
    ("sorted(bn.state_dict().keys())", repr(sorted(bn.state_dict().keys())), "학습 파라미터 + running 통계가 함께 저장"),
])


def emsg(fn):
    try:
        fn()
        return "오류 없음"
    except Exception as ex:  # noqa: BLE001
        m = str(ex).strip().splitlines()[0]
        return f"{type(ex).__name__}: {m[:80]}{'…' if len(m) > 80 else ''}"


IO["d10-nn-batchnorm1d"][1][4] = ("train 모드에서 bn(torch.randn(1, 4))", emsg(lambda: bn(torch.randn(1, 4))), "batch가 1이면 분산을 못 구해 오류")
IO["d10-nn-batchnorm1d"][1][5] = ("nn.BatchNorm1d(64)에 (32, 32) 입력", emsg(lambda: nn.BatchNorm1d(64)(torch.randn(32, 32))), "num_features가 앞 Linear out_features와 달라서 오류")

# 10. train / eval
torch.manual_seed(0)
m = Net(dropout=0.3)
x5 = X_va[:5]
m.train()
t1, t2 = m(x5), m(x5)
ret = m.eval()
with torch.no_grad():
    e1, e2 = m(x5), m(x5)
lg_eval_grad = m(x5)
IO["d10-model-train"] = ("Net(dropout=0.3), valid 앞 5건", [
    ("model.train() → model(x) 두 번 같나?", repr(torch.equal(t1, t2)), "Dropout이 매번 다른 칸을 꺼서 달라짐"),
    ("model.eval() → model(x) 두 번 같나?", repr(torch.equal(e1, e2)), "Dropout 꺼짐 → 같은 입력 = 같은 출력"),
    ("model.eval() is model", repr(ret is m), "self를 돌려줘서 이어 쓸 수 있음"),
    ("model.training, model[3].training", f"{m.training}, {m[3].training}", "하위 층(Dropout)까지 한꺼번에 바뀜"),
    ("eval()만 하고 no_grad 없이: model(x).requires_grad", repr(lg_eval_grad.requires_grad), "eval은 기울기 기록을 끄지 않음 → no_grad는 따로"),
])

# 11. param groups
torch.manual_seed(0)
model = Net(width=64, bn=True)
decay, no_decay = [], []
for layer in model.modules():
    for name, p in layer.named_parameters(recurse=False):
        if name == "bias" or isinstance(layer, nn.BatchNorm1d):
            no_decay.append(p)
        else:
            decay.append(p)
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=0.001)
IO["d10-파라미터-그룹으로-weight-decay-다르게-주기"] = ("Net(width=64, bn=True): Linear 3개 + BatchNorm1d 2개", [
    ("len(decay), len(no_decay)", f"{len(decay)}, {len(no_decay)}", "Linear weight 3개 / Linear bias 3 + BN weight·bias 4"),
    ("[tuple(p.shape) for p in decay]", repr([tuple(p.shape) for p in decay]), "입력에 곱해지는 가중치만"),
    ("len(optimizer.param_groups)", repr(len(optimizer.param_groups)), "그룹 2개"),
    ("[g['weight_decay'] for g in optimizer.param_groups]", repr([g["weight_decay"] for g in optimizer.param_groups]), "그룹마다 다른 감쇠"),
    ("[g['lr'] for g in optimizer.param_groups]", repr([g["lr"] for g in optimizer.param_groups]), "그룹에 안 적은 lr은 바깥 기본값을 같이 씀"),
    ("파라미터 총수 = decay + no_decay ?", repr(sum(p.numel() for p in decay + no_decay) == sum(p.numel() for p in model.parameters())), "빠진 파라미터 없음"),
])


# 12. Early stopping
def early(losses, min_delta, patience):
    best_loss = reference = float("inf")
    wait, best_ep = 0, None
    for ep, v in enumerate(losses, 1):
        if v < best_loss:
            best_loss, best_ep = v, ep
        if v < reference - min_delta:
            reference, wait = v, 0
        else:
            wait += 1
            if wait >= patience:
                return ep, best_ep, best_loss
    return len(losses), best_ep, best_loss


s_ep, b_ep, b_l = early([1.00, 0.80, 0.79, 0.78, 0.70], 0.05, 2)
torch.manual_seed(42)
model = Net(width=64)
opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
best_loss = reference = float("inf")
wait, best_state, best_ep, stop = 0, None, 0, 200
alias = model.state_dict()
for ep in range(1, 201):
    train_one_epoch(model, ld60, opt)
    v, _ = evaluate(model, X_va, y_va)
    if v < best_loss:
        best_loss, best_ep = v, ep
        best_state = copy.deepcopy(model.state_dict())
        shallow = model.state_dict()
        shallow_w = shallow["0.weight"].clone()
    if v < reference - 0.001:
        reference, wait = v, 0
    else:
        wait += 1
        if wait >= 10:
            stop = ep
            break
last_v, _ = evaluate(model, X_va, y_va)
moved = not torch.equal(shallow["0.weight"], shallow_w)
model.load_state_dict(best_state)
rest_v, _ = evaluate(model, X_va, y_va)
IO["d10-early-stopping-직접-짜기"] = ("교재 숫자 예 · train 앞 60건 + valid 80, AdamW lr=0.01, patience 10, min_delta 0.001", [
    ("valid loss 1.00, .80, .79, .78, .70 / min_delta=.05, patience=2", f"멈춘 epoch {s_ep}, best epoch {b_ep} (loss {b_l})",
     ".79·.78은 충분한 개선이 아니라 대기 1·2 → 4에서 멈춤, best는 .78 (.70은 보지도 못함)"),
    ("실제 학습: best epoch → 멈춘 epoch", f"{best_ep} → {stop}", "best 뒤 10 epoch 동안 0.001 넘게 좋아지지 않아 중단"),
    ("멈춘 시점 valid loss", f4(last_v), "그냥 쓰면 best보다 나쁜 모델"),
    ("model.load_state_dict(best_state) 후 valid loss", f4(rest_v), f"best epoch의 값({f4(best_loss)})으로 정확히 돌아옴"),
    ("deepcopy 없이 담은 state_dict()가 학습 후 바뀌었나?", repr(moved), "그냥 담으면 지금 가중치를 가리켜서 같이 바뀜 → deepcopy 필요"),
])

# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day10.html"
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
