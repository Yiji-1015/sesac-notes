"""Day11 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261102/02_응용/data, 64×64 가로줄/세로줄 PNG)는 임시 폴더에 복사해서 읽기만 한다.
ResNet18 사전학습 가중치는 이 PC에 캐시가 없어서 받지 않고 weights=None(구조만)으로 돌린다.
결과는 cmds/day11.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day11.py
"""
import copy
import html
import random
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261102/02_응용/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")
DATA = TMP / "data"


def seed(s=0):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = torch.cuda.get_device_name(0) if DEV.type == "cuda" else "CPU"


def sh(t):
    return str(list(t.shape))


def n_params(m, only_trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not only_trainable))


IO = {}

# 1. permute / unsqueeze
seed(0)
images = torch.rand(4, 3, 32, 32)
img = images[0]
hwc = img.permute(1, 2, 0)
nchw = img.unsqueeze(0)
png = sorted((DATA / "train" / "horizontal").glob("*.png"))[0]
from PIL import Image  # noqa: E402

pil = Image.open(png).convert("RGB")
t = transforms.ToTensor()(pil)
IO["d11-축-순서-바꾸기-개수-축-붙이기"] = (f"torch.rand 난수(seed 0) · 강의 데이터 train/horizontal/{png.name}", [
    ("images = torch.rand(4, 3, 32, 32); images.shape", sh(images), "[N, C, H, W] = 4장 · 3채널 · 32×32"),
    ("img = images[0]; img.shape", sh(img), "한 장을 꺼내면 N축이 빠짐"),
    ("img.permute(1, 2, 0).shape", sh(hwc), "[H, W, C] — plt.imshow가 원하는 순서"),
    ("hwc[5, 7, 1] == img[1, 5, 7]", str(bool(hwc[5, 7, 1] == img[1, 5, 7])), "값은 그대로, 읽는 순서만 바뀜"),
    ("img.permute(1, 2, 0).is_contiguous()", str(hwc.is_contiguous()), "메모리는 안 옮기고 보는 방식만 바꾼 view"),
    ("img.unsqueeze(0).shape", sh(nchw), "앞에 개수 축 1을 붙임 → 모델에 한 장 넣을 때"),
    ("np.array(Image.open(png)).shape", str(np.array(pil).shape), "PIL/NumPy 사진은 [H, W, C] · 0~255"),
    ("transforms.ToTensor()(pil).shape, min, max", f"{sh(t)}, {t.min():.2f}, {t.max():.2f}", "ToTensor가 [C, H, W] · 0~1로 바꿔 줌"),
])

# 2. Conv2d
seed(0)
x = torch.rand(4, 3, 32, 32)
conv = nn.Conv2d(3, 8, 3, padding=1)
inp = torch.zeros(1, 1, 5, 5)
inp[0, 0, :, 1] = 1
kernel = torch.zeros(1, 1, 3, 3)
kernel[0, 0, :, 1] = 1
fc = F.conv2d(inp, kernel)[0, 0]
IO["d11-nn-conv2d"] = ("torch.rand(4, 3, 32, 32) (seed 0) · 5×5 세로선 그림", [
    ("nn.Conv2d(3, 8, 3, padding=1)(x).shape", sh(conv(x)), "채널 3 → 8, 크기 유지(padding=1)"),
    ("conv.weight.shape, conv.bias.shape", f"{sh(conv.weight)}, {sh(conv.bias)}", "[out, in, K, K] — Kernel 8장, 각각 두께 3"),
    ("sum(p.numel() for p in conv.parameters())", str(n_params(conv)), "8·3·3·3 + 8 = 224"),
    ("nn.Conv2d(3, 8, 3, padding=0)(x).shape", sh(nn.Conv2d(3, 8, 3)(x)), "padding 0이면 가장자리 2칸 줄어듦"),
    ("nn.Conv2d(3, 8, 3, stride=2, padding=1)(x).shape", sh(nn.Conv2d(3, 8, 3, stride=2, padding=1)(x)), "stride 2면 가로·세로 절반"),
    ("F.conv2d(inp, kernel)[0, 0]", "\n".join(" ".join(f"{v:g}" for v in row) for row in fc.tolist()), "padding 없이 5×5 → 3×3. 세로선(입력 1열)이 Kernel 가운데 열과 겹치는 출력 0열만 1×3 = 3"),
    ("nn.Conv2d(1, 8, 3)(x)  ← in_channels 틀림", "RuntimeError (채널 수 불일치)", "Kernel 두께 1인데 입력은 3채널"),
])
try:
    nn.Conv2d(1, 8, 3)(x)
except RuntimeError:
    pass
else:
    raise SystemExit("in_channels 오류가 안 났음")


# 3. SmallCNN
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(x)


seed(0)
model = SmallCNN(num_classes=2)
h = torch.rand(8, 3, 64, 64)
rows = []
for i, layer in enumerate(model.features):
    h = layer(h)
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d)):
        rows.append((f"features[{i}] {type(layer).__name__}", sh(h), f"파라미터 {n_params(layer):,}"))
p = model.pool(h)
rows += [
    ("model.pool(h).shape", sh(p), "채널마다 평균 하나"),
    ("torch.flatten(p, 1).shape", sh(torch.flatten(p, 1)), "[N, 32]"),
    ("torch.flatten(p).shape  ← 1을 빼면", sh(torch.flatten(p)), "8장이 한 줄로 섞여 버림"),
    ("model(torch.rand(8, 3, 64, 64)).shape", sh(model(torch.rand(8, 3, 64, 64))), "[N, K] 점수(logit)"),
    ("model(torch.rand(2, 3, 224, 224)).shape", sh(model(torch.rand(2, 3, 224, 224))), "크기가 달라도 Head 입력은 32 → 그대로 동작"),
    ("sum(p.numel() for p in model.parameters())", f"{n_params(model):,}", "448 + 4,640 + 66"),
]
IO["d11-smallcnn-만들기"] = ("torch.rand(8, 3, 64, 64) (seed 0)를 층마다 통과", rows)

# 4. BatchNorm2d
seed(0)
block = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
xb = torch.rand(8, 3, 64, 64)
bn = block[1]
conv_out = block[0](xb)
block.train()
bn_out = bn(conv_out)
rm_after = bn.running_mean[:3].tolist()
cm = conv_out.mean(dim=(0, 2, 3))[:3].tolist()
y_train_a = block(xb)[0]
y_train_b = block(torch.cat([xb[:1], torch.rand(7, 3, 64, 64) * 3]))[0]
block.eval()
y_eval_a = block(xb)[0]
y_eval_b = block(torch.cat([xb[:1], torch.rand(7, 3, 64, 64) * 3]))[0]
IO["d11-nn-batchnorm2d"] = ("torch.rand(8, 3, 64, 64) (seed 0)", [
    ("block(x).shape", sh(block(xb)), "Shape 그대로"),
    ("sum(p.numel() for p in bn.parameters())", str(n_params(bn)), "채널마다 γ·β 하나씩 = 16 × 2"),
    ("conv 출력 채널 평균 [:3]", " ".join(f"{v:.3f}" for v in cm), "BN 전: 채널마다 제각각"),
    ("train 모드 BN 출력의 채널 평균·표준편차", f"평균 ≈ {bn_out.mean(dim=(0, 2, 3)).abs().max():.1e}, std ≈ {bn_out.std(dim=(0, 2, 3)).mean():.3f}", "모든 채널이 평균 0 · 표준편차 1로"),
    ("bn.running_mean[:3]  (forward 한 번 뒤)", " ".join(f"{v:.3f}" for v in rm_after), "0에서 batch 평균 쪽으로 10%(momentum 0.1)만 이동"),
    ("train: 같은 사진, batch 구성만 바꿔 출력 차이", f"{(y_train_a - y_train_b).abs().max():.3f}", "batch 통계를 써서 같은 사진도 결과가 달라짐"),
    ("eval: 같은 사진, batch 구성만 바꿔 출력 차이", f"{(y_eval_a - y_eval_b).abs().max():.3f}", "저장 통계를 써서 언제나 같은 결과"),
])

# 5. ImageFolder
seed(0)
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_ds = datasets.ImageFolder(str(DATA / "train"), transform=transform)
valid_ds = datasets.ImageFolder(str(DATA / "valid"), transform=transform)
g = torch.Generator().manual_seed(0)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False)
images_b, labels_b = next(iter(train_loader))
raw = datasets.ImageFolder(str(DATA / "train"))
s0 = train_ds.samples[0]
IO["d11-imagefolder"] = ("강의 데이터 data/train (horizontal 30 · vertical 30) · data/valid (10 · 10)", [
    ("train_ds.class_to_idx", repr(train_ds.class_to_idx), "폴더 이름 알파벳 순서로 번호"),
    ("len(train_ds), len(valid_ds)", f"{len(train_ds)}, {len(valid_ds)}", ""),
    ("train_ds.samples[0]", f"('…/train/horizontal/{Path(s0[0]).name}', {s0[1]})", "(파일 경로, 라벨)"),
    ("type(ImageFolder('data/train')[0][0])  ← transform 없음", type(raw[0][0]).__name__, "PIL 이미지 그대로 → batch로 못 묶음"),
    ("images, labels = next(iter(train_loader))", f"{sh(images_b)}, {sh(labels_b)}", "batch_size=32 → 32장씩"),
    ("images.min(), images.max()", f"{images_b.min():.2f}, {images_b.max():.2f}", "ToTensor로 0~1 실수"),
    ("labels[:12]", " ".join(str(v) for v in labels_b[:12].tolist()), "shuffle=True라 두 클래스가 섞여 나옴"),
    ("len(train_loader)", str(len(train_loader)), "60장 ÷ 32 → 마지막 batch는 28장"),
])

# 6. 학습 루프
seed(0)
model = SmallCNN(num_classes=2).to(DEV)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
best_loss = float("inf")
hist = []
g.manual_seed(0)
t0 = time.perf_counter()
for epoch in range(1, 9):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEV), labels.to(DEV)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    vl_sum, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in valid_loader:
            images, labels = images.to(DEV), labels.to(DEV)
            logits = model(images)
            vl_sum += loss_fn(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    vl, va = vl_sum / n, correct / n
    hist.append((epoch, vl, va))
    if vl < best_loss:
        best_loss, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
alias = model.state_dict()  # deepcopy 없이 담은 것
before = alias["fc.bias"].clone()
with torch.no_grad():
    model.fc.bias += 1.0
alias_changed = not torch.equal(before, alias["fc.bias"])
deep_changed = not torch.equal(best_state["fc.bias"], model.state_dict()["fc.bias"])
model.load_state_dict(best_state)
ce = lambda lg, y: f"{nn.CrossEntropyLoss()(torch.tensor([lg]), torch.tensor([y])).item():.3f}"
IO["d11-손실-학습-루프-valid-평가-best-저장"] = (f"강의 데이터 train 60장으로 SmallCNN 8 epoch (seed 0, {GPU}) — 숫자는 작은 데이터라 흔들릴 수 있어요", [
    ("nn.CrossEntropyLoss()([[0., 0.]], [0])", ce([0.0, 0.0], 0), "반반 확신 → −log 0.5"),
    ("nn.CrossEntropyLoss()([[3., 0.]], [0])", ce([3.0, 0.0], 0), "정답에 자신 있게 → 벌점 작음"),
    ("nn.CrossEntropyLoss()([[0., 3.]], [0])", ce([0.0, 3.0], 0), "자신 있게 틀림 → 벌점 큼"),
    *[(f"epoch {e}: valid loss · acc", f"{v:.3f} · {a:.2f}", "") for e, v, a in hist],
    ("best_epoch, best_loss", f"{best_epoch}, {best_loss:.3f}", "valid loss가 가장 낮았던 순간"),
    ("(해석)", f"acc는 epoch {next(e for e, _, a in hist if a == 1.0)}부터 1.00, loss는 0.69 근처",
     "방향은 다 맞혔지만 확률은 아직 반반에 가까움 — lr 1e-3에 step 16번뿐이라 확신이 덜 자람. 정확도만 보면 이걸 놓쳐요"),
    ("state_dict()를 그대로 담고 나서 학습이 바뀌면", f"담아 둔 값도 바뀜: {alias_changed}", "같은 Tensor를 가리킴"),
    ("copy.deepcopy(state_dict())로 담았으면", f"담아 둔 값도 바뀜: {not deep_changed}", "복사본이라 그대로 남음"),
])
# 위 마지막 줄 확인 (deepcopy 쪽은 안 바뀌어야 정상)
assert deep_changed and alias_changed
train_time = time.perf_counter() - t0

# 7. 저장 / 불러오기
ck = TMP / "best_model.pt"
torch.save({"state_dict": best_state, "class_to_idx": train_ds.class_to_idx, "image_size": 64,
            "normalize": None, "best_epoch": best_epoch}, ck)
ckpt = torch.load(ck, weights_only=True)
restored = SmallCNN(num_classes=len(ckpt["class_to_idx"])).to(DEV)
restored.load_state_dict(ckpt["state_dict"])
restored.eval()
vx, _ = next(iter(valid_loader))
vx = vx.to(DEV)
with torch.inference_mode():
    same = torch.allclose(model(vx), restored(vx))
try:
    nn.Sequential(nn.Linear(2, 2)).load_state_dict(ckpt["state_dict"])
    bad = "오류 없음"
except RuntimeError as e:
    bad = "RuntimeError: " + str(e).splitlines()[0][:60] + "…"
IO["d11-저장할-땐-읽는-법-까지-함께"] = ("'손실 · 학습 루프' 항목에서 학습한 best SmallCNN을 임시 폴더에 저장", [
    ("torch.save({...}, \"best_model.pt\") 파일 크기", f"{ck.stat().st_size / 1024:.1f} KB", "파라미터 5,154개 × 4byte ≈ 20KB + 정보"),
    ("list(ckpt.keys())", repr(list(ckpt.keys())), "숫자 + 읽는 법"),
    ("ckpt[\"class_to_idx\"], ckpt[\"image_size\"]", f"{ckpt['class_to_idx']}, {ckpt['image_size']}", "0번이 horizontal인 걸 파일만 보고 앎"),
    ("torch.allclose(model(x), restored(x))", str(same), "불러온 모델이 같은 답을 냄"),
    ("다른 구조에 load_state_dict", bad, "층 이름·크기가 안 맞으면 오류"),
])

# 8. ResNet18 Head-only (가중치 없이 구조만)
seed(0)
weights = ResNet18_Weights.DEFAULT
tf = weights.transforms()
rn = resnet18(weights=None)
total_before = n_params(rn)
for p_ in rn.parameters():
    p_.requires_grad = False
rn.fc = nn.Linear(rn.fc.in_features, 2)
trainable = n_params(rn, True)
wrong = resnet18(weights=None)
wrong.fc = nn.Linear(wrong.fc.in_features, 2)
for p_ in wrong.parameters():
    p_.requires_grad = False
rn.eval()
rn.fc.train()
bn1_before = rn.bn1.running_mean.clone()
rn(torch.rand(2, 3, 224, 224))
bn1_eval_same = torch.equal(bn1_before, rn.bn1.running_mean)
rn.train()
rn(torch.rand(2, 3, 224, 224))
bn1_train_same = torch.equal(bn1_before, rn.bn1.running_mean)
t224 = tf(pil)
IO["d11-resnet18-head-only-전이학습"] = ("ResNet18 — 사전학습 가중치 없이 구조만(weights=None, 다운로드 안 함) · 강의 PNG 한 장", [
    ("weights.transforms()", "crop 224 · resize 256 · bilinear\nmean [0.485, 0.456, 0.406]\nstd [0.229, 0.224, 0.225]", "ImageNet 학습 때와 같은 입력 규칙"),
    ("transform(64×64 PNG).shape", sh(t224), "64×64가 224×224로 커짐"),
    ("transform(PNG).mean(), .std()", f"{t224.mean():.2f}, {t224.std():.2f}", "Normalize 때문에 0~1이 아님 (음수도 나옴)"),
    ("resnet18() 전체 파라미터", f"{total_before:,}", "Head(1000클래스) 포함"),
    ("model.fc.in_features", str(rn.fc.in_features), "Head에 들어오는 특징 수"),
    ("얼린 뒤 fc 교체 → 학습 파라미터 수", f"{trainable:,}", "512 × 2 + 2 = Head만"),
    ("fc 교체 뒤에 얼리면 → 학습 파라미터 수", f"{n_params(wrong, True):,}", "순서 실수: 새 Head까지 얼어 배울 게 0개"),
    ("model.eval(); model.fc.train() 후 forward → bn1 저장 통계", "그대로" if bn1_eval_same else "바뀜", "얼린 층의 BN 통계 유지"),
    ("model.train() 후 forward → bn1 저장 통계", "그대로" if bn1_train_same else "바뀜", "requires_grad=False여도 BN 통계는 바뀜!"),
])

# 9. Partial
for p_ in rn.layer4.parameters():
    p_.requires_grad = True
opt = torch.optim.AdamW([
    {"params": rn.layer4.parameters(), "lr": 1e-4},
    {"params": rn.fc.parameters(), "lr": 1e-3},
])
IO["d11-partial-미세조정-layer4-head-lr-다르게"] = ("Head-only로 만든 ResNet18 구조(weights=None, 구조만)에서 layer4를 열었을 때", [
    ("sum(p.numel() for p in model.layer4.parameters())", f"{n_params(rn.layer4):,}", "마지막 블록 하나에 전체의 약 3/4이 몰려 있음"),
    ("학습 파라미터 수 (layer4 + fc)", f"{n_params(rn, True):,}", f"전체 {n_params(rn):,}의 {n_params(rn, True) / n_params(rn) * 100:.1f}%"),
    ("[g['lr'] for g in optimizer.param_groups]", repr([gg["lr"] for gg in opt.param_groups]), "그룹마다 다른 lr"),
    ("[len(g['params']) for g in optimizer.param_groups]", repr([len(gg["params"]) for gg in opt.param_groups]), "layer4 Tensor 15개(Conv·BN), fc 2개(weight·bias)"),
])

# 10. 시간 재기
def bench(device, loader, reps=3):
    seed(0)
    m = SmallCNN(num_classes=2).to(device)
    o = torch.optim.AdamW(m.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    data = [(a, b) for a, b in loader]

    def one_epoch():
        seen = 0
        for a, b in data:
            a, b = a.to(device), b.to(device)
            o.zero_grad()
            lf(m(a), b).backward()
            o.step()
            seen += len(b)
        return seen

    one_epoch()  # 워밍업
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    seen = sum(one_epoch() for _ in range(reps))
    if device.type == "cuda":
        torch.cuda.synchronize()
    sec = time.perf_counter() - start
    mem = f"{torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB" if device.type == "cuda" else "N/A"
    return seen / sec, mem


def no_sync(device, a):
    m = SmallCNN(2).to(device)
    a = a.to(device)
    m(a)
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(20):
        m(a)
    t_ns = time.perf_counter() - s
    torch.cuda.synchronize()
    t_s = time.perf_counter() - s
    return t_ns * 1000, t_s * 1000


cpu_ips, cpu_mem = bench(torch.device("cpu"), train_loader)
rows = [("CPU · 강의 train 60장, batch 32", f"{cpu_ips:,.0f} images/s · 메모리 {cpu_mem}", "CPU에선 VRAM 값 없음")]
if DEV.type == "cuda":
    gpu_ips, gpu_mem = bench(DEV, train_loader)
    rows.append((f"GPU({GPU}) · 같은 조건", f"{gpu_ips:,.0f} images/s · peak {gpu_mem}", f"CPU의 약 {gpu_ips / cpu_ips:.1f}배 — 60장·batch 32라 GPU가 덜 바쁨"))
    big = DataLoader(torch.utils.data.TensorDataset(torch.rand(1024, 3, 64, 64), torch.randint(0, 2, (1024,))), batch_size=256)
    c2, _ = bench(torch.device("cpu"), big, reps=1)
    g2, gm2 = bench(DEV, big, reps=1)
    rows += [
        ("CPU · 난수 1024장, batch 256", f"{c2:,.0f} images/s", ""),
        ("GPU · 난수 1024장, batch 256", f"{g2:,.0f} images/s · peak {gm2}", f"batch가 크면 GPU가 약 {g2 / c2:.0f}배"),
    ]
    a, b_ = no_sync(DEV, torch.rand(256, 3, 64, 64))
    rows.append(("forward 20번: synchronize 없이 / 있게", f"{a:.1f} ms / {b_:.1f} ms", "없으면 GPU가 끝나기 전에 시계를 멈춤"))
IO["d11-한-epoch-시간-재기"] = ("SmallCNN 학습 step, 워밍업 1회 후 측정 — 이 PC 기준이라 다른 장치에선 숫자가 달라요", rows)


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day11.html"
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
