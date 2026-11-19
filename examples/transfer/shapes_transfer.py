# 도형 분류 전이학습 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/train, valid, test 아래 circle/, square/, triangle/ (40×40 컬러 그림)
#         data/source_backbone.npz  다른 도형 그림 300장으로 미리 학습해 둔 작은 CNN의 가중치
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
from collections import Counter                                  #@1
import numpy as np                                               #@3
import copy                                                      #@5
import torch                                                     #@1
from torch import nn                                             #@2
import torch.nn.functional as F                                  #@2
from torchvision import datasets, transforms                     #@1
from sklearn.metrics import f1_score                             #@2
from sklearn.metrics import classification_report, confusion_matrix  #@6

DATA = Path(__file__).resolve().parent / "data"                  #@1

# ① 데이터 읽고 개수 세기
prep = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])  # 빌려 올 모델이 배운 입력 모양

def load(split):
    ds = datasets.ImageFolder(DATA / split, transform=prep)
    return torch.stack([x for x, _ in ds]), torch.tensor(ds.targets), ds.classes

X_train, y_train, classes = load("train")
X_valid, y_valid, _ = load("valid")
counts = {classes[k]: v for k, v in sorted(Counter(y_train.tolist()).items())}

# ② 처음부터 학습하기
def make_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten())                   # 특징 16 × 4 × 4 = 256개

def make_model():
    return nn.Sequential(make_backbone(), nn.Linear(256, 3))     # [0] 특징 뽑기(Backbone), [1] 분류(Head)

def fit(model, epochs=30, lr=0.01, weight=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        model.train()
        for i in torch.randperm(len(X_train), generator=torch.Generator().manual_seed(0)).split(32):
            loss = F.cross_entropy(model(X_train[i]), y_train[i], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X).argmax(1)

def report(model):
    p = predict(model, X_valid)
    return {"accuracy": (p == y_valid).float().mean().item(), "macro_f1": f1_score(y_valid, p, average="macro")}

torch.manual_seed(0)
scratch = fit(make_model(), epochs=5)

# ③ 미리 배운 Backbone 빌려 오기
source = np.load(DATA / "source_backbone.npz")
pretrained = make_backbone()
pretrained.load_state_dict({k: torch.tensor(source[k]) for k in source.files})

# ④ Backbone은 고정하고 Head만 학습하기
torch.manual_seed(0)
frozen = make_model()
frozen[0].load_state_dict(pretrained.state_dict())
for p in frozen[0].parameters():
    p.requires_grad = False                                  # Backbone 가중치는 바꾸지 않음
trainable = sum(p.numel() for p in frozen.parameters() if p.requires_grad)
fit(frozen, epochs=5)

# ⑤ 전체를 조금씩 고치기 (미세조정)
finetuned = copy.deepcopy(frozen)                            # ④의 모델은 그대로 두고 복사본을
for p in finetuned[0].parameters():
    p.requires_grad = True
fit(finetuned, epochs=5, lr=0.001)                           # 이미 좋은 가중치라 작은 학습률로

# ⑥ 종류별로 보기
pred = predict(finetuned, X_valid)
per_class = classification_report(y_valid, pred, target_names=classes, output_dict=True, zero_division=0)
cm = confusion_matrix(y_valid, pred)

# ⑦ 적은 종류에 무게 주기
weight = torch.tensor([len(y_train) / (3 * n) for n in counts.values()], dtype=torch.float32)  # 적을수록 크게
torch.manual_seed(0)
weighted = make_model()
weighted[0].load_state_dict(pretrained.state_dict())
fit(weighted, epochs=5, lr=0.001, weight=weight)

# ⑧ 마지막 한 번: test
X_test, y_test, _ = load("test")
test_scores = {name: f1_score(y_test, predict(m, X_test), average="macro")
               for name, m in {"처음부터": scratch, "Head만": frozen, "미세조정": finetuned, "미세조정 + 무게": weighted}.items()}
