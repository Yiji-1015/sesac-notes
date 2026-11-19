# 점 두 무리 나누기 (PyTorch) 전체 코드. 노트 두 쪽의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
#   1쪽 Tensor와 신경망 학습: ①~⑤   2쪽 학습을 안정시키기: ⑥~⑪
# 데이터: data/points.csv (점 400개, x1, x2, label 0/1, split train 240 / valid 80 / test 80)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import pandas as pd                                              #@1
import torch                                                     #@1
from torch import nn                                             #@2
from torch.utils.data import DataLoader, TensorDataset           #@5
import copy                                                      #@10

DATA = Path(__file__).resolve().parent / "data" / "points.csv"   #@1

# ① GPU 확인하고 Tensor로 바꾸기
gpu = torch.cuda.is_available()
device = "cpu"  # 이 예제는 작아서 CPU로 충분하고, 결과가 매번 똑같이 나와요
df = pd.read_csv(DATA)

def tensors(split):
    part = df[df["split"] == split]
    X = torch.tensor(part[["x1", "x2"]].values, dtype=torch.float32)
    y = torch.tensor(part["label"].values, dtype=torch.float32).unsqueeze(1)  # (N,) → (N, 1)
    return X.to(device), y.to(device)

X_train, y_train = tensors("train")
X_valid, y_valid = tensors("valid")

# ② 선 하나로 나눠 보기
def accuracy(model, X, y):
    with torch.no_grad():  # 채점할 때는 기울기 계산이 필요 없어요
        return ((model(X) > 0).float() == y).float().mean().item()

torch.manual_seed(0)
linear = nn.Linear(2, 1)  # w1*x1 + w2*x2 + b
opt = torch.optim.SGD(linear.parameters(), lr=0.1)
for _ in range(500):
    loss = nn.functional.binary_cross_entropy_with_logits(linear(X_train), y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
linear_acc = accuracy(linear, X_valid, y_valid)

# ③ 신경망 만들기
def make_net(hidden=16, depth=1, dropout=0.0):
    layers, size = [], 2
    for _ in range(depth):
        layers += [nn.Linear(size, hidden), nn.ReLU(), nn.Dropout(dropout)]
        size = hidden
    return nn.Sequential(*layers, nn.Linear(size, 1)).to(device)

torch.manual_seed(0)
net = make_net()
n_params = sum(p.numel() for p in net.parameters())

# ④ 손실과 역전파 한 번
loss_fn = nn.BCEWithLogitsLoss()
loss = loss_fn(net(X_train), y_train)
loss_before = loss.item()  # 학습 전 손실
loss.backward()  # 모든 가중치에 대해 "손실을 줄이려면 어느 쪽으로" 를 계산
grad_first = net[0].weight.grad.clone()
net.zero_grad()

# ⑤ 학습 루프
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(0))

def train(model, optimizer, epochs=100, loader=loader):
    X_seen, y_seen = loader.dataset.tensors  # 실제로 배우는 데 쓴 데이터
    history = []
    for epoch in range(epochs):
        model.train()  # Dropout 같은 학습용 동작 켜기
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()   # 채점할 때는 끄기
        with torch.no_grad():
            history.append({"epoch": epoch + 1, "train_loss": loss_fn(model(X_seen), y_seen).item(),
                            "valid_loss": loss_fn(model(X_valid), y_valid).item(),
                            "valid_acc": accuracy(model, X_valid, y_valid)})
    return history

torch.manual_seed(0)
net = make_net()
hist = train(net, torch.optim.Adam(net.parameters(), lr=0.01))

# ⑥ 학습률 비교
def run(lr, opt_name="sgd", epochs=100, **net_kw):
    torch.manual_seed(0)
    model = make_net(**net_kw)
    make = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[opt_name]
    return model, train(model, make(model.parameters(), lr=lr), epochs)

lr_runs = {lr: run(lr)[1] for lr in (0.001, 0.1, 10.0)}

# ⑦ Optimizer 비교
opt_runs = {name: run(0.01, name, epochs=60)[1] for name in ("sgd", "adam")}

# ⑧ 과적합 만들기
small = DataLoader(TensorDataset(X_train[:40], y_train[:40]), batch_size=8, shuffle=True,
                   generator=torch.Generator().manual_seed(0))  # 학습 데이터를 40개로 줄이기
torch.manual_seed(0)
big = make_net(hidden=128, depth=3)
over = train(big, torch.optim.Adam(big.parameters(), lr=0.01), epochs=300, loader=small)

# ⑨ 규제: Dropout과 weight decay
torch.manual_seed(0)
reg = make_net(hidden=128, depth=3, dropout=0.3)
regd = train(reg, torch.optim.Adam(reg.parameters(), lr=0.01, weight_decay=1e-3), epochs=300, loader=small)

# ⑩ 조기 종료: valid 손실이 가장 낮았던 때로 돌아가기
def train_early(model, optimizer, patience=20, max_epochs=300, loader=small):
    best, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, max_epochs + 1):
        h = {**train(model, optimizer, 1, loader)[0], "epoch": epoch}
        history.append(h)
        if h["valid_loss"] < best:
            best, best_state, wait = h["valid_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:  # patience 번 연속 나아지지 않으면 멈춤
                break
    model.load_state_dict(best_state)
    return history

torch.manual_seed(0)
early_net = make_net(hidden=128, depth=3)
early = train_early(early_net, torch.optim.Adam(early_net.parameters(), lr=0.01))

# ⑪ 마지막 한 번: test
X_test, y_test = tensors("test")
test_acc = {"기본 신경망 (⑤)": accuracy(net, X_test, y_test), "과적합 (⑧)": accuracy(big, X_test, y_test),
            "규제 (⑨)": accuracy(reg, X_test, y_test), "조기 종료 (⑩)": accuracy(early_net, X_test, y_test)}
