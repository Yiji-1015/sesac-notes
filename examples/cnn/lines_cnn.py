# 가로줄, 세로줄 그림 분류 (CNN) 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/train, data/valid, data/test 아래 horizontal/, vertical/ 폴더 (64×64 그림, train 60 / valid 20 / test 20)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import torch                                                     #@1
from torch import nn                                             #@1
from torchvision import datasets, transforms                     #@1
import torch.nn.functional as F                                  #@3

DATA = Path(__file__).resolve().parent / "data"                  #@1

# ① 그림을 Tensor로
to_tensor = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])  # 흑백 한 장, 0~1

def load(split):
    ds = datasets.ImageFolder(DATA / split, transform=to_tensor)  # 폴더 이름이 곧 정답
    X = torch.stack([x for x, _ in ds])
    y = torch.tensor([t for _, t in ds])
    return X, y, ds.classes

X_train, y_train, classes = load("train")
X_valid, y_valid, _ = load("valid")

# ② 픽셀을 그냥 펼치면 (MLP)
def make_mlp():
    return nn.Sequential(nn.Flatten(), nn.Linear(64 * 64, 32), nn.ReLU(), nn.Linear(32, 2))

# ③ 합성곱 한 번: 작은 무늬 찾기
horizontal_kernel = torch.tensor([[-1., -1., -1.], [2., 2., 2.], [-1., -1., -1.]])  # 가로 선에 크게 반응
vertical_kernel = horizontal_kernel.T                                                 # 세로 선에 크게 반응
kernels = torch.stack([horizontal_kernel, vertical_kernel])[:, None]                  # (2, 1, 3, 3)
sample = torch.stack([X_train[y_train == 0][0], X_train[y_train == 1][0]])           # 가로 한 장, 세로 한 장
maps = F.relu(F.conv2d(sample, kernels, padding=1))                                  # (2장, 2무늬, 64, 64)
response = maps.mean(dim=(2, 3))                                                     # 그림마다 무늬별 평균 반응

# ④ CNN 만들기
def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 → 32
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),                       # 어디에 있든 평균 → 16개 값
        nn.Linear(16, 2))

def count(model):
    return sum(p.numel() for p in model.parameters())

# ⑤ 학습하고 비교하기
def fit(model, epochs=60):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(epochs):
        loss = F.cross_entropy(model(X_train), y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

def acc(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()

torch.manual_seed(0)
mlp = fit(make_mlp())
torch.manual_seed(0)
cnn = fit(make_cnn())
scores = {"MLP": acc(mlp, X_valid, y_valid), "CNN": acc(cnn, X_valid, y_valid)}

# ⑥ 선의 위치가 바뀌면
shifted = {k: torch.roll(X_valid, shifts=(k, k), dims=(2, 3)) for k in (0, 8, 16)}  # k칸씩 아래, 오른쪽으로
shift_scores = {k: {"MLP": acc(mlp, Xs, y_valid), "CNN": acc(cnn, Xs, y_valid)} for k, Xs in shifted.items()}

# ⑦ 깊게 쌓기: 잔차 연결과 BatchNorm
class Block(nn.Module):  # 3×3 합성곱 하나짜리 블록
    def __init__(self, ch, residual, bn):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(ch) if bn else nn.Identity()  # 끄면 아무것도 안 하는 층
        self.residual = residual

    def forward(self, x):
        h = self.bn(self.conv(x))
        return F.relu(x + h) if self.residual else F.relu(h)  # 잔차 연결: 입력을 그대로 더하기

def make_deep(depth, residual, bn):
    return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         *[Block(8, residual, bn) for _ in range(depth)],
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 2))

DEPTH = 16
deep_runs = {}
for name, (residual, bn) in {"그냥 쌓기": (False, False), "잔차 연결": (True, False), "잔차 + BatchNorm": (True, True)}.items():
    torch.manual_seed(0)
    m = make_deep(DEPTH, residual, bn)
    F.cross_entropy(m(X_train), y_train).backward()
    grad0 = m[0].weight.grad.norm().item()  # 학습 전, 맨 앞 층까지 내려온 기울기 크기
    torch.manual_seed(0)
    m = fit(make_deep(DEPTH, residual, bn)).eval()
    with torch.no_grad():
        deep_runs[name] = {"grad0": grad0, "train_loss": F.cross_entropy(m(X_train), y_train).item(),
                           "valid_acc": acc(m, X_valid, y_valid)}
