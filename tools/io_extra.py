"""이론 페이지에서 나중에 옮겨 온 명령어 8개의 '입력 → 출력' 상자 (tools/move_leftover_code.py 로 옮긴 것).

상자 class 는 "io io-extra" — io_dayNN.py 가 자기 상자(class="io")를 지우고 다시 넣을 때 건드리지 않게 구분한다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_extra.py
"""
import html
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
IO = {}

# ---- Day6 KNN 스케일링
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X = np.array([[1000, 1], [1400, 4]]); y = np.array([0, 1]); q = np.array([[1100, 4]])
raw = KNeighborsClassifier(n_neighbors=1).fit(X, y)
s = StandardScaler().fit(X)
sc = KNeighborsClassifier(n_neighbors=1).fit(s.transform(X), y)
IO[(6, "d6-knn-standardscaler")] = ("노트의 A·B·Q 세 요청 (글자수, 도구수)", [
    ("raw.predict([[1100, 4]])", str(raw.predict(q).tolist()), "원래 단위: 글자수 차이가 거리를 지배 → A(0)"),
    ("s.mean_, s.scale_", f"{s.mean_.tolist()}, {s.scale_.tolist()}", "train 두 행의 평균·표준편차를 기억"),
    ("s.transform([[1100, 4]])", np.array2string(s.transform(q).round(3)), "Q도 같은 규칙으로 변환"),
    ("scaled.predict(s.transform(q))", str(sc.predict(s.transform(q)).tolist()), "표준화 후엔 도구수 차이도 반영 → B(1)"),
])
X1 = np.array([[100], [200], [300]])
a = StandardScaler().fit(X1); b = MinMaxScaler().fit(X1)
IO[(6, "d6-standardscaler")] = ("train 값 100·200·300", [
    ("StandardScaler().fit(X).transform(X)", str(a.transform(X1).ravel().round(3).tolist()), f"평균 {a.mean_[0]:g}, 표준편차 {a.scale_[0]:.1f}"),
    ("MinMaxScaler().fit(X).transform(X)", str(b.transform(X1).ravel().tolist()), "최소 100 → 0, 최대 300 → 1"),
    ("b.transform([[500]])", str(b.transform([[500]]).ravel().tolist()), "train 범위 밖이라 1을 넘어요"),
    ("a.transform([[500]])", str(a.transform([[500]]).ravel().round(3).tolist()), "표준화도 범위 제한이 없어요"),
])

# ---- Day7 부스팅 한 회차
from sklearn.tree import DecisionTreeRegressor

X7 = np.array([[1], [2], [3], [4]]); y7 = np.array([100., 120., 200., 220.])
init = np.full(4, y7.mean()); res = y7 - init
corr = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X7, res)
upd = init + 0.5 * corr.predict(X7)
IO[(7, "d7-decisiontreeregressor")] = ("노트의 4개 예제", [
    ("residual = y - initial", str(res.tolist()), "처음 예측 160에서 얼마나 틀렸나"),
    ("correction.predict(X)", str(corr.predict(X7).tolist()), "깊이 1 트리: x ≤ 2.5면 −50, 아니면 +50"),
    ("initial + 0.5 * correction.predict(X)", str(upd.tolist()), "보정을 절반(학습률 0.5)만 반영"),
    ("MSE 전 → 후", f"{np.mean((y7 - init) ** 2):g} → {np.mean((y7 - upd) ** 2):g}", "한 회차 만에 오차가 줄어요"),
    ("initial + 1.0 * correction.predict(X)", str((init + corr.predict(X7)).tolist()), f"학습률 1이면 MSE {np.mean((y7 - init - corr.predict(X7)) ** 2):g} — 이 4개엔 더 맞지만 잡음까지 따라갈 위험"),
])

# ---- Day8 MAE · RMSE
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

yt = [100, 200, 300, 400]; pa = [100, 200, 300, 480]; pb = [130, 230, 330, 430]
IO[(8, "d8-mean-absolute-error")] = ("실제 100·200·300·400", [
    ("mean_absolute_error(y, A)", f"{mean_absolute_error(yt, pa):g}", "A 오차 (0, 0, 0, 80) → 평균 20"),
    ("root_mean_squared_error(y, A)", f"{root_mean_squared_error(yt, pa):g}", "80이 제곱돼 6400 → 큰 오차에 민감"),
    ("mean_absolute_error(y, B)", f"{mean_absolute_error(yt, pb):g}", "B 오차는 고르게 30"),
    ("root_mean_squared_error(y, B)", f"{root_mean_squared_error(yt, pb):g}", "고른 오차는 MAE = RMSE"),
])

# ---- Day9 train / eval / no_grad
torch.manual_seed(9)
m = nn.Sequential(nn.Linear(4, 8), nn.Dropout(p=0.5), nn.Linear(8, 1))
x = torch.ones(1, 4)
m.train(); t1, t2 = m(x).item(), m(x).item()
m.eval(); e1, e2 = m(x).item(), m(x).item()
out_eval = m(x)
with torch.no_grad():
    out_ng = m(x)
IO[(9, "d9-model-train")] = ("Dropout(0.5)가 든 작은 모델 · 같은 입력 두 번", [
    ("model.train(); model(x) 두 번", f"{t1:.4f}, {t2:.4f}", "학습 모드: Dropout이 매번 다른 칸을 꺼서 값이 달라요"),
    ("model.eval(); model(x) 두 번", f"{e1:.4f}, {e2:.4f}", "평가 모드: 매번 같아요"),
    ("model.eval() 후 model(x).requires_grad", str(out_eval.requires_grad), "eval()만으로는 기울기 기록이 안 꺼져요!"),
    ("with torch.no_grad(): model(x).requires_grad", str(out_ng.requires_grad), "기울기 기록은 no_grad가 끔"),
])

# ---- Day11 transforms
from PIL import Image
from torchvision import transforms

img = Image.fromarray((np.indices((80, 100)).sum(0) % 255).astype(np.uint8)).convert("RGB")
train_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
                               transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.15),
                               transforms.ToTensor()])
valid_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
torch.manual_seed(11)
ta, tb = train_tf(img), train_tf(img)
va, vb = valid_tf(img), valid_tf(img)
IO[(11, "d11-transforms-compose")] = ("100×80 합성 그라데이션 사진 한 장", [
    ("train_tf(img).shape, dtype", f"{tuple(ta.shape)}, {ta.dtype}", "[C, H, W], 0~1 실수 (ToTensor)"),
    ("train_tf(img) 두 번이 같나?", str(bool(torch.equal(ta, tb))), "부를 때마다 무작위로 다르게 변형"),
    ("valid_tf(img) 두 번이 같나?", str(bool(torch.equal(va, vb))), "평가용은 항상 똑같이"),
    ("valid_tf(img).min(), max()", f"{va.min().item():.3f}, {va.max().item():.3f}", "ToTensor가 0~255를 0~1로"),
])

# ---- Day13 Frame 건너뛰기
SRC = Path("C:/practice_assignment/20261104/02_응용/data/sample.avi")
tmp = Path(tempfile.mkdtemp()); shutil.copy(SRC, tmp / "sample.avi")
cap = cv2.VideoCapture(str(tmp / "sample.avi"))
step, idx, done = 5, 0, []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if idx % step == 0:
        done.append(idx)
    idx += 1
cap.release(); shutil.rmtree(tmp, ignore_errors=True)
IO[(13, "d13-frame-idx-step")] = ("data/sample.avi (48 Frame, 12 FPS)", [
    ("읽은 Frame 수", str(idx), ""),
    ("step = 5 로 처리한 Frame 번호", str(done), f"{len(done)}장만 처리 — 계산량 약 1/{step}"),
    ("처리 간격(초)", f"{step / 12:.3f}", "12 FPS에서 5장 = 약 0.42초마다 한 번 → 그 사이 빠른 사건은 놓칠 수 있어요"),
])

# ---- Day14 seed · save · load
torch.manual_seed(14); r1 = torch.rand(3)
torch.manual_seed(14); r2 = torch.rand(3)
r3 = torch.rand(3)
tmpd = Path(tempfile.mkdtemp())
net = nn.Linear(4, 3)
torch.save({"weights": net.state_dict(), "classes": ["circle", "square", "triangle"], "lr": 0.01,
            "epochs": 8, "seed": 14, "test_used": False}, tmpd / "model.pt")
saved = torch.load(tmpd / "model.pt", map_location="cpu", weights_only=True)
net2 = nn.Linear(4, 3); res_load = net2.load_state_dict(saved["weights"])
same = all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), net2.state_dict().values()))
shutil.rmtree(tmpd, ignore_errors=True)
IO[(14, "d14-torch-manual-seed")] = ("작은 Linear(4, 3) 모델로", [
    ("torch.manual_seed(14); torch.rand(3)", np.array2string(r1.numpy(), precision=4), ""),
    ("다시 manual_seed(14); torch.rand(3)", np.array2string(r2.numpy(), precision=4), "seed가 같으면 똑같은 난수"),
    ("이어서 torch.rand(3) 한 번 더", np.array2string(r3.numpy(), precision=4), "seed를 다시 안 정하면 다음 난수로 넘어가요"),
    ("list(saved.keys())", str(list(saved.keys())), "가중치 + 만든 방법을 함께 저장"),
    ("model.load_state_dict(saved['weights'])", str(res_load), "missing·unexpected 둘 다 비면 정상"),
    ("저장 전후 가중치가 같나?", str(same), ""),
])


# ---------------------------------------------------------------- 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io io-extra">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


for (day, cid), (title, rows) in IO.items():
    f = ROOT / "cmds" / f"day{day:02d}.html"
    s = f.read_text(encoding="utf-8")
    sec = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
    assert sec, cid
    body = re.sub(r'<div class="io io-extra">.*?\n</div>\n', '', sec.group(0), flags=re.S)
    m = re.search(r'</code></pre>\n|</table>\n', body)
    body = body[:m.end()] + box(title, rows) + "\n" + body[m.end():]
    s = s[:sec.start()] + body + s[sec.end():]
    f.write_text(s, encoding="utf-8")
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i} → {o}   ({n})")
