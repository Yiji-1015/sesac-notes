"""Day14 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261105/02_응용/data: circle/square/triangle 40×40 PNG,
train 120·40·20 / valid 30·30·30 / test 10·10·10)는 임시 폴더에 복사해서 읽기만 한다.
ResNet18 사전학습 가중치는 이 PC에 캐시가 없어서 받지 않고 weights=None(구조만)으로 돌린다.
결과는 cmds/day14.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day14.py
"""
import copy
import html
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

sys.stdout.reconfigure(encoding="utf-8")
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="A single label was found")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261105/02_응용/data")
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


def sh(t):
    return str(list(t.shape))


def n_params(m, only_trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not only_trainable))


def fmt(v, d=3):
    return " ".join(f"{x:.{d}f}" for x in v)


IO = {}
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8  # 강의 코드의 range(8)

# 1. ImageFolder
train = ImageFolder(str(DATA / "train"))
valid = ImageFolder(str(DATA / "valid"))
test = ImageFolder(str(DATA / "test"))
assert train.class_to_idx == valid.class_to_idx == test.class_to_idx
IO["d14-imagefolder"] = ("강의 데이터 data/train · valid · test", [
    ("train.class_to_idx", repr(train.class_to_idx), "폴더 이름 알파벳 순서"),
    ("train.class_to_idx == valid.class_to_idx == test.class_to_idx", "True", "분할마다 번호가 같아야 비교가 됨"),
    *[(f"Counter({n}.targets)", repr(dict(sorted(Counter(ds.targets).items()))), note)
      for n, ds, note in [("train", train, "6 : 2 : 1로 기울어 있음"), ("valid", valid, "균형"), ("test", test, "균형")]],
    ("train.samples[0]", f"('…/train/circle/{Path(train.samples[0][0]).name}', {train.samples[0][1]})", "(파일 경로, 라벨)"),
    ("train[0][0].size, train[0][0].mode", f"{train[0][0].size}, {train[0][0].mode!r}", "transform이 없어서 PIL 이미지 그대로"),
    ("전부 circle이라고 찍었을 때 train 정확도", f"{120 / 180:.3f}", "모델이 아무것도 안 배워도 66.7%"),
])

# 2. GroupShuffleSplit
rng = np.random.default_rng(0)
rows = pd.DataFrame({"group_id": np.repeat(np.arange(20), 3), "aug": np.tile(["원본", "flip", "rot"], 20)})
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, va_idx = next(gss.split(rows, groups=rows.group_id))
both = set(rows.group_id[tr_idx]) & set(rows.group_id[va_idx])
rtr, rva = train_test_split(np.arange(len(rows)), test_size=0.25, random_state=42)
both_r = set(rows.group_id[rtr]) & set(rows.group_id[rva])
IO["d14-groupshufflesplit"] = ("합성 표: 원본 20장 × (원본·flip·rot) = 60행, group_id = 원본 번호", [
    ("len(tr_idx), len(va_idx)", f"{len(tr_idx)}, {len(va_idx)}", "그룹 20개 중 25%인 5그룹 = 15행이 valid"),
    ("sorted(set(rows.group_id[va_idx]))", repr(sorted(int(g) for g in set(rows.group_id[va_idx]))), "valid로 간 원본 번호"),
    ("set(train 그룹) & set(valid 그룹)", repr(both) if both else "set()", "겹치는 원본 없음 → 누수 없음"),
    ("train_test_split(이미지 단위, test_size=0.25)로 나누면 겹치는 원본 수", str(len(both_r)), "같은 원본의 변형이 양쪽에 → valid 점수가 부풀어요"),
])

# 3. 사전학습 모델 + Head 교체 (가중치 없이 구조만)
seed(0)
weights = ResNet18_Weights.DEFAULT
num_classes = 3
model = resnet18(weights=None)
fc_before = repr(model.fc)
for p in model.parameters():
    p.requires_grad = False
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
preprocess = weights.transforms()
pil = train[0][0]
t = preprocess(pil)
wrong = resnet18(weights=None)
wrong.fc = nn.Linear(512, num_classes)
for p in wrong.parameters():
    p.requires_grad = False
with torch.no_grad():
    out = model.eval()(t.unsqueeze(0))
IO["d14-사전학습-모델-불러오기-head-교체"] = ("ResNet18 — 사전학습 가중치 없이 구조만(weights=None, 다운로드 안 함) · 강의 train circle 한 장(40×40)", [
    ("weights", f"{weights}", "DEFAULT가 가리키는 실제 이름"),
    ("model.fc  (교체 전)", fc_before, "ImageNet 1000클래스 Head"),
    ("model.fc.in_features", str(in_features), ""),
    ("model.fc  (교체 후)", repr(model.fc), "우리 3클래스 Head"),
    ("model.fc.weight.requires_grad", str(model.fc.weight.requires_grad), "얼린 뒤에 만들어서 학습 대상"),
    ("preprocess", f"resize {preprocess.resize_size} → crop {preprocess.crop_size}\nmean {preprocess.mean}\nstd {preprocess.std}", "평가용 입력 규칙"),
    ("preprocess(40×40 PIL).shape", sh(t), "작은 사진도 224×224로 키움"),
    ("model(x).shape", sh(out), "[1, 3] 점수"),
    ("순서 실수: fc를 먼저 바꾸고 전체를 얼림 → 학습 파라미터", str(n_params(wrong, True)), "0개! 에러도 안 남"),
])

# 4. 파라미터 세기
trainable = n_params(model, True)
total = n_params(model)
names = [n for n, p in model.named_parameters() if p.requires_grad]
IO["d14-학습-가능한-파라미터-수-세기"] = ("Head 교체한 ResNet18 구조 (3클래스, weights=None이어도 개수는 같음)", [
    ("trainable", f"{trainable:,}", "512 × 3 + 3"),
    ("total", f"{total:,}", f"전체의 {trainable / total * 100:.3f}%만 학습"),
    ("[name for name, p in model.named_parameters() if p.requires_grad]", repr(names), "열린 층 이름"),
    ("순서 실수 모델의 trainable", str(n_params(wrong, True)), "이 숫자가 0이면 바로 의심"),
])

# 5. 층 열기 · 층별 lr
for p in model.layer4.parameters():
    p.requires_grad = True
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
old_opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
try:
    torch.optim.AdamW([{"params": model.fc.parameters()}, {"params": model.fc.parameters()}])
    dup = "오류 없음"
except ValueError as e:
    dup = "ValueError: " + str(e)[:55] + "…"
IO["d14-층-열기-층별-학습률"] = ("Head 교체한 ResNet18 구조(weights=None)에서 layer4를 연 뒤", [
    ("trainable (layer4 + fc)", f"{n_params(model, True):,}", f"전체의 {n_params(model, True) / total * 100:.1f}% — 갑자기 수천 배"),
    ("[(len(g['params']), g['lr'], g['weight_decay']) for g in optimizer.param_groups]",
     repr([(len(g["params"]), g["lr"], g["weight_decay"]) for g in optimizer.param_groups]), "그룹마다 Tensor 수 · lr · weight_decay"),
    ("층을 열기 전에 만든 optimizer가 관리하는 Tensor 수", str(sum(len(g["params"]) for g in old_opt.param_groups)), "fc 2개뿐 → layer4는 열어도 안 바뀜"),
    ("같은 파라미터를 두 그룹에 넣으면", dup, "중복은 에러"),
])

# ---- 강의 데이터를 Tensor로 (40×40 그대로)
to_t = transforms.ToTensor()


def stack(ds):
    X = torch.stack([to_t(img) for img, _ in ds])
    y = torch.tensor(ds.targets)
    return X, y


X_train, y_train = stack(train)
X_valid, y_valid = stack(valid)
paths = [Path(p).name for p, _ in valid.samples]
class_names = train.classes


def make_model():
    seed(0)
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten())
    head = nn.Linear(32, 3)
    return backbone, head, nn.Sequential(backbone, head)


def warm_bn(backbone):
    # 얼린 Backbone이라도 BN 기억값은 한 번 채워 둔다(사전학습 모델 흉내) — 학습 전 train 데이터로 1회 forward
    backbone.train()
    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = None  # 누적 평균
    with torch.no_grad():
        backbone(X_train)


def train_head(criterion, epochs=8, lr=0.01, loader=None):
    backbone, head, model = make_model()
    warm_bn(backbone)
    for p in backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    best_loss, hist = float("inf"), []
    for epoch in range(epochs):
        model.train()
        backbone.eval()
        batches = [(X_train, y_train)] if loader is None else loader
        for xb, yb in batches:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            valid_loss = nn.CrossEntropyLoss()(model(X_valid), y_valid).item()
        hist.append(valid_loss)
        if valid_loss < best_loss:
            best_loss, best_epoch, best_state = valid_loss, epoch, copy.deepcopy(model.state_dict())
    last = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_valid).argmax(1)
    return model, hist, best_epoch, best_loss, pred, last


# 6. 학습 루프 + best 복원
crit = nn.CrossEntropyLoss()
model6, hist, best_epoch, best_loss, pred6, last_state = train_head(crit, epochs=EPOCHS, lr=0.01)
tmp_m = make_model()[2]
tmp_m.load_state_dict(last_state)
tmp_m.eval()
with torch.no_grad():
    last_loss = crit(tmp_m(X_valid), y_valid).item()
n_tr = sum(p.numel() for p in model6.parameters() if p.requires_grad)
rec6 = recall_score(y_valid, pred6, average=None, labels=[0, 1, 2], zero_division=0)
IO["d14-학습-루프-best-복원"] = (f"강의 train 180장(40×40) 통째로 한 batch · 작은 CNN Backbone(seed 0 무작위, 얼림) + Linear Head · lr 0.01 · {EPOCHS} epoch",
    [("학습 파라미터 수 (Head만)", str(n_tr), "32 × 3 + 3")]
    + [(f"epoch {e}: valid loss", f"{v:.4f}", "") for e, v in enumerate(hist) ]
    + [
        ("best_epoch, best_loss", f"{best_epoch}, {best_loss:.4f}", "가장 좋았던 순간"),
        ("마지막 epoch 모델의 valid loss", f"{last_loss:.4f}", "복원 안 하면 이 모델이 남음"),
        ("best 복원 후 클래스별 valid Recall (circle·square·triangle)", fmt(rec6), "전부 circle로 답함"),
        ("(해석 — 예상과 다르게 나온 결과)", "valid loss가 epoch 0부터 오르기만 해서 best = 처음",
         "무작위로 얼린 특징으론 모양을 못 가르고, Head가 train의 6 : 2 : 1 비율만 배워 '일단 circle'로 기움 → 균형 잡힌 valid에선 loss가 올라감. "
         "best 복원이 '나빠진 마지막 모델'을 막아 준 예이자, 사전학습 Backbone이 왜 필요한지 보여 주는 예"),
    ])

# 7. class weight
counts = torch.bincount(y_train).float()
class_weight = counts.sum() / counts
class_weight = class_weight / class_weight.mean()
loss_w = nn.CrossEntropyLoss(weight=class_weight)
logits0 = torch.zeros(3, 3)
y3 = torch.tensor([0, 1, 2])
per = nn.CrossEntropyLoss(reduction="none")(torch.tensor([[2.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]]), y3)
w_mean = loss_w(torch.tensor([[2.0, 0, 0], [2.0, 0, 0], [2.0, 0, 0]]), y3)
model_w, _, _, _, pred_w, _ = train_head(loss_w, epochs=EPOCHS, lr=0.01)
rec_w = recall_score(y_valid, pred_w, average=None, labels=[0, 1, 2], zero_division=0)
IO["d14-nn-crossentropyloss-weight"] = (f"강의 train 개수 [120, 40, 20] · 'best 복원' 항목과 같은 모델·seed·{EPOCHS} epoch에 weight만 추가", [
    ("torch.bincount(train targets)", repr(counts.int().tolist()), ""),
    ("class_weight", fmt(class_weight.tolist(), 2), "역빈도를 평균 1로 — 적은 클래스일수록 큼"),
    ("(counts * class_weight).tolist()", fmt((counts * class_weight).tolist(), 0), "클래스마다 '개수 × 무게'가 같아짐"),
    ("세 샘플 모두 logit [2, 0, 0], 정답 0·1·2 → 샘플별 loss", fmt(per.tolist()), "circle만 맞힘"),
    ("weight 없음 (평균)", f"{per.mean():.3f}", ""),
    ("weight 있음", f"{w_mean:.3f}", "Σ wᵢ·lossᵢ ÷ Σ wᵢ — 틀린 square·triangle 쪽 벌점이 더 큼"),
    ("valid Recall (weight 없음)", fmt(rec6), "'best 복원' 항목 결과"),
    ("valid Recall (weight 있음)", fmt(rec_w),
     f"square·triangle은 0 → {rec_w[1]:.2f}·{rec_w[2]:.2f}로 올랐지만 circle은 1.00 → {rec_w[0]:.2f}로 떨어짐 — 공짜가 아님. 이 작은 실험의 결과일 뿐 일반 법칙은 아니에요"),
])

# 8. WeightedRandomSampler
targets = torch.tensor(train.targets)
cnt = torch.bincount(targets)
cw = 1.0 / cnt.float()
sample_weight = cw[targets]
gen = torch.Generator().manual_seed(0)
sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True, generator=gen)
drawn = list(sampler)
drawn_cls = Counter(int(targets[i]) for i in drawn)
tri_idx = [i for i in drawn if targets[i] == 2]
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler)
plain = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))


def batch_stats(ld, epochs=50):
    per = []
    for _ in range(epochs):
        for _, yy in ld:
            if len(yy) == 32:
                per.append(torch.bincount(yy, minlength=3).tolist())
    a = np.array(per)
    return a.mean(0), a.min(0), a.max(0)


s_mean, s_min, s_max = batch_stats(loader)
p_mean, p_min, p_max = batch_stats(plain)
gen.manual_seed(0)
bad = list(WeightedRandomSampler(cw, num_samples=10, replacement=True, generator=torch.Generator().manual_seed(0)))
try:
    DataLoader(TensorDataset(X_train, y_train), batch_size=32, sampler=sampler, shuffle=True)
    both_err = "오류 없음"
except ValueError as e:
    both_err = "ValueError: " + str(e)[:50] + "…"
gen.manual_seed(0)
_, _, _, _, pred_s, _ = train_head(nn.CrossEntropyLoss(), epochs=EPOCHS, lr=0.01, loader=loader)
rec_s = recall_score(y_valid, pred_s, average=None, labels=[0, 1, 2], zero_division=0)
IO["d14-weightedrandomsampler"] = ("강의 train 180장 (120 · 40 · 20), generator seed 0", [
    ("sample_weight[:2], sample_weight[-2:]", f"{fmt(sample_weight[:2].tolist(), 4)}, {fmt(sample_weight[-2:].tolist(), 4)}", "circle 1/120 · triangle 1/20"),
    ("한 epoch(180번) 뽑은 클래스 수", repr(dict(sorted(drawn_cls.items()))), "거의 1/3씩 (정확히 같지는 않음)"),
    ("triangle 뽑힌 횟수 / 서로 다른 triangle 장수", f"{len(tri_idx)} / {len(set(tri_idx))}",
     f"복원 추출: {len(set(tri_idx))}장이 평균 {len(tri_idx) / len(set(tri_idx)):.1f}번씩 뽑히고 {20 - len(set(tri_idx))}장은 이번 epoch에 한 번도 안 뽑힘"),
    ("batch(32장)당 클래스 수 평균 [최소~최대]  sampler, 50 epoch",
     " · ".join(f"{m:.1f} [{lo}~{hi}]" for m, lo, hi in zip(s_mean, s_min, s_max)), "평균은 거의 1/3씩, 하지만 batch마다 흔들림"),
    ("같은 통계  shuffle=True",
     " · ".join(f"{m:.1f} [{lo}~{hi}]" for m, lo, hi in zip(p_mean, p_min, p_max)), "보통 추출은 평균 6 : 2 : 1 그대로"),
    ("클래스 weight 3개를 그대로 넣으면 → 뽑힌 인덱스", repr(bad), "0·1·2번 '샘플'만 뽑힘 — 샘플별 weight가 필요"),
    ("DataLoader(…, sampler=sampler, shuffle=True)", both_err, "둘은 같이 못 씀"),
    (f"valid Recall (sampler, batch 32 · {EPOCHS} epoch)", fmt(rec_s), f"Head-only {fmt(rec6, 2)} · class weight {fmt(rec_w, 2)}와 비교 — 같은 모델·seed의 작은 실험 결과예요"),
])

# 9. confusion_matrix
labels = list(range(len(class_names)))
y_true = y_valid.numpy()


def cm_rows(pred, tag):
    cm = confusion_matrix(y_true, pred, labels=labels)
    rep = classification_report(y_true, pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    return [
        (f"confusion_matrix ({tag})", "\n".join(" ".join(f"{v:3d}" for v in r) for r in cm), "행 = 실제, 열 = 예측 (circle · square · triangle)"),
        (f"report['macro avg']['f1-score'] ({tag})", f"{rep['macro avg']['f1-score']:.3f}", ""),
        (f"f1_score(average='weighted') · accuracy ({tag})", f"{f1_score(y_true, pred, average='weighted'):.3f} · {rep['accuracy']:.3f}", ""),
    ]


all_circle = np.zeros_like(y_true)
rows9 = cm_rows(pred6.numpy(), "Head-only") + cm_rows(pred_w.numpy(), "class weight") + [
    ("confusion_matrix(y_true[:30], 전부 0)  ← labels 없이 (circle만 있는 부분)", repr(confusion_matrix(y_true[:30], all_circle[:30]).tolist()), "1×1 표 — 없는 클래스가 표에서 사라짐"),
    ("같은 입력에 labels=[0, 1, 2]", "\n".join(" ".join(f"{v:3d}" for v in r) for r in confusion_matrix(y_true[:30], all_circle[:30], labels=labels)), "3×3으로 고정"),
]
IO["d14-confusion-matrix"] = ("강의 valid 90장 (30 · 30 · 30) — 'best 복원'(Head-only)·'Class weight' 항목에서 학습한 두 모델의 예측", rows9)

# 10. 오답 모으기
model_w.eval()
with torch.no_grad():
    logits = model_w(X_valid)
    probs = logits.softmax(dim=1)
    conf, pred = probs.max(dim=1)
wrong_i = (pred != y_valid).nonzero().flatten()
rows10 = [
    ("probs[0], probs[0].sum()", f"{fmt(probs[0].tolist())}, {probs[0].sum():.3f}", "클래스별 확률, 합 1"),
    ("len(wrong)", f"{len(wrong_i)} / {len(y_valid)}", "틀린 valid 수"),
    ("Counter(T→P) 오답 유형", repr({f"{class_names[a]}→{class_names[b]}": c for (a, b), c in Counter(zip(y_valid[wrong_i].tolist(), pred[wrong_i].tolist())).most_common()}), "어느 쪽으로 틀리나"),
    ("오답 확신도 conf[wrong] 최소 · 중앙 · 최대", f"{conf[wrong_i].min():.2f} · {conf[wrong_i].median():.2f} · {conf[wrong_i].max():.2f}", f"0.9 이상이면 '자신 있게 틀린' 위험 오답 — 여기선 최대 {conf[wrong_i].max():.2f}라 그런 오답은 없고, 모델이 전반적으로 확신이 낮음(3클래스 반반이 0.33)"),
]
for k, i in enumerate(wrong_i[conf[wrong_i].argsort(descending=True)][:3].tolist()):
    rows10.append((f"paths[{i}], T, P, conf", f"{paths[i]}  T={class_names[y_valid[i]]}  P={class_names[pred[i]]}  {conf[i]:.2f}",
                   "가장 자신 있게 틀린 오답부터 — 이 사진을 직접 열어 봐야 '왜'가 보여요" if k == 0 else ""))
IO["d14-오답-이미지를-확률과-함께-모으기"] = ("강의 valid 90장 — 'Class weight' 항목에서 학습한 모델(Head-only는 전부 circle이라 오답 분석이 단조로워서)", rows10)

# 11. Hook (ResNet18 구조, 가중치 없이)
seed(0)
rn = resnet18(weights=None)
rn.fc = nn.Linear(512, 3)
rn.eval()
activation, gradient, handles = {}, {}, []
calls = {"fwd": 0, "grad": 0}


def save_grad(grad):
    calls["grad"] += 1
    gradient["value"] = grad.detach()


def save_act(module, inputs, output):
    calls["fwd"] += 1
    activation["value"] = output.detach()
    handles.append(output.register_hook(save_grad))


image = preprocess(pil).unsqueeze(0).detach().requires_grad_(True)
target_layer = rn.layer4[-1]
handles.append(target_layer.register_forward_hook(save_act))
with torch.enable_grad():
    rn.zero_grad(set_to_none=True)
    logits = rn(image)
    target_class = int(logits.argmax(1))
    logits[0, target_class].backward()
calls_snap = dict(calls)
n_hooks_before = len(target_layer._forward_hooks)
for h in handles:
    h.remove()
with torch.no_grad():
    try:
        h2 = target_layer.register_forward_hook(save_act)
        rn(image)
        nograd = "Tensor.register_hook 오류 없음"
    except RuntimeError as e:
        nograd = "RuntimeError: " + str(e)[:45] + "…"
    finally:
        h2.remove()
IO["d14-hook으로-activation-gradient-기록"] = ("ResNet18 구조(weights=None, 3클래스 Head) · 강의 circle 한 장을 preprocess → [1, 3, 224, 224]", [
    ("activation['value'].shape", sh(activation["value"]), "layer4[-1] 출력: 512채널 · 7×7 지도"),
    ("gradient['value'].shape", sh(gradient["value"]), "같은 모양의 gradient"),
    ("forward hook · tensor hook 호출 횟수", f"{calls_snap['fwd']} · {calls_snap['grad']}", "Forward 1번, backward 1번"),
    ("h.remove() 전/후 target_layer._forward_hooks 수", f"{n_hooks_before} → {len(target_layer._forward_hooks)}", "지워야 다음 forward에 안 쌓임"),
    ("no_grad() 안에서 output.register_hook", nograd, "gradient를 안 만드는 곳에선 hook을 못 검"),
    ("image.grad.shape", sh(image.grad), "입력에도 gradient가 생김 (requires_grad_(True))"),
])

# 12. Heatmap
grads = gradient["value"][0]
acts = activation["value"][0]
alpha = grads.mean(dim=(1, 2))
cam0 = (alpha[:, None, None] * acts).sum(0)
cam = torch.relu(cam0)
cam = cam / (cam.max() + 1e-8)
up = torch.nn.functional.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
IO["d14-채널-weight와-heatmap-계산"] = ("'Hook' 항목의 activation·gradient 그대로 — 가중치가 무작위라 지도 모양 자체엔 의미가 없어요(계산 과정 확인용)", [
    ("alpha.shape", sh(alpha), "채널마다 중요도 숫자 하나"),
    ("(alpha * acts).sum(0).shape", sh(cam0), "7×7 지도"),
    ("relu 전 음수 칸 비율", f"{(cam0 < 0).float().mean() * 100:.0f}%", "점수를 낮추는 쪽 칸 — relu가 지움"),
    ("cam.min(), cam.max()  (정규화 후)", f"{cam.min():.2f}, {cam.max():.2f}", "가장 밝은 곳 = 1"),
    ("interpolate(…, size=(224, 224)).shape", sh(up), "원본 크기로 확대"),
    ("확대 전후 서로 다른 값 개수", f"{cam.unique().numel()} → {up.unique().numel()}", "부드럽게 이어 붙였을 뿐, 49칸 정보가 늘어나진 않음"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day14.html"
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
