"""Day13 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 실습 데이터(C:/practice_assignment/20261104/02_응용/data)는 임시 폴더에 복사해서 읽기만 한다.
결과는 cmds/day13.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day13.py
"""
import html
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261104/02_응용/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")
OUT = TMP / "outputs"
OUT.mkdir()


def rd(name, flag=cv2.IMREAD_COLOR):
    return cv2.imread(str(TMP / "data" / name), flag)


def r(v):
    """출력값을 사람이 읽기 좋은 짧은 문자열로."""
    if isinstance(v, np.ndarray) and v.size > 12:
        return f"array shape={v.shape} dtype={v.dtype}"
    if isinstance(v, np.ndarray):
        return np.array2string(v, separator=" ")
    if isinstance(v, float):
        return f"{v:g}"
    return repr(v)


IO = {}
img = rd("scene.png")
H, W = img.shape[:2]
red = np.argwhere((img[:, :, 2] == 255) & (img[:, :, 1] == 0) & (img[:, :, 0] == 0))
ry, rx = (int(v) for v in red[len(red) // 2])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. imread / imwrite
rgba = rd("rgba.png", cv2.IMREAD_UNCHANGED)
IO["d13-cv2-imread"] = ("data/scene.png 기준", [
    ('img = cv2.imread("data/scene.png")', f"img.shape → {img.shape}, dtype → {img.dtype}", "세로 240 · 가로 320 · 채널 3(BGR)"),
    ("img.min(), img.max()", f"({img.min()}, {img.max()})", "uint8이라 0~255 안"),
    ('cv2.imread("data/scene.png", cv2.IMREAD_GRAYSCALE).shape', r(rd("scene.png", cv2.IMREAD_GRAYSCALE).shape), "채널 축이 없어짐"),
    ('cv2.imread("data/rgba.png", cv2.IMREAD_UNCHANGED).shape', r(rgba.shape), "투명도(Alpha) 포함 4채널"),
    ('cv2.imread("data/rgba.png").shape', r(rd("rgba.png").shape), "기본 COLOR는 Alpha를 버림"),
    ('cv2.imread("data/없는파일.png")', r(cv2.imread(str(TMP / "data" / "nofile.png"))), "에러가 아니라 None!"),
    ('cv2.imwrite("outputs/copy.png", img)', r(cv2.imwrite(str(OUT / "copy.png"), img)), "성공하면 True"),
    ('cv2.imwrite("outputs/없는폴더/a.png", img)', r(cv2.imwrite(str(OUT / "nodir" / "a.png"), img)), "폴더가 없으면 False (에러 아님)"),
])

# 2. imshow / waitKey — GUI라 값만
IO["d13-cv2-imshow"] = ("창을 띄우는 명령이라 반환값만 정리", [
    ('cv2.imshow("result", img)', "None", "창에 그리기만 하고 돌려주는 값 없음"),
    ("cv2.waitKey(0)", "누른 키의 번호 (예: q → " + str(ord("q")) + ", Esc → 27)", "키 입력 전까지 멈춤"),
    ("cv2.waitKey(1)", "1ms 안에 키가 없으면 -1", "영상 반복에서 매 Frame 잠깐만 기다릴 때"),
])

# 3. split / merge
b, g, rr = cv2.split(img)
merged = cv2.merge([b, g, rr])
rgbm = cv2.merge([rr, g, b])
IO["d13-cv2-split"] = (f"빨간 사각형 안의 한 칸 img[{ry}, {rx}]", [
    (f"img[{ry}, {rx}]", r(img[ry, rx]), "B 0 · G 0 · R 255"),
    ("b, g, r = cv2.split(img)", f"b.shape → {b.shape}", "채널마다 2차원 [H, W]"),
    (f"b[{ry}, {rx}], g[{ry}, {rx}], r[{ry}, {rx}]", f"({b[ry, rx]}, {g[ry, rx]}, {rr[ry, rx]})", ""),
    ("(cv2.merge([b, g, r]) == img).all()", r(bool((merged == img).all())), "넣은 순서 그대로면 원본과 같음"),
    (f"cv2.merge([r, g, b])[{ry}, {rx}]", r(rgbm[ry, rx]), "순서를 바꾸면 RGB 배열"),
])

# 4. cvtColor
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
IO["d13-cv2-cvtcolor"] = (f"같은 빨간 칸 [{ry}, {rx}]", [
    ("cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape", r(gray.shape), "채널 축이 사라짐"),
    (f"gray[{ry}, {rx}]", r(int(gray[ry, rx])), "0.299×255 ≈ 76 (빨강만 있으니 R 무게만)"),
    (f"cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[{ry}, {rx}]", r(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[ry, rx]), "B와 R 자리만 바뀜"),
    (f"hsv[{ry}, {rx}]", r(hsv[ry, rx]), "H 0(빨강) · S 255(진함) · V 255(밝음)"),
    ("cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)" + f"[{ry}, {rx}]", r(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)[ry, rx]), "같은 밝기를 3번 복사 — 색은 안 돌아옴"),
])

# 5. resize
mask = rd("mask_truth.png", cv2.IMREAD_GRAYSCALE)
up_n = cv2.resize(mask, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
up_l = cv2.resize(mask, None, fx=8, fy=8, interpolation=cv2.INTER_LINEAR)
IO["d13-cv2-resize"] = ("scene.png (240, 320, 3) · mask_truth.png", [
    ("cv2.resize(img, dsize=(224, 224)).shape", r(cv2.resize(img, (224, 224)).shape), "4:3 → 1:1로 찌그러짐"),
    ("cv2.resize(img, dsize=(160, 120)).shape", r(cv2.resize(img, (160, 120)).shape), "dsize는 (W, H), shape는 (H, W)!"),
    ("cv2.resize(img, None, fx=0.5, fy=0.5).shape", r(cv2.resize(img, None, fx=0.5, fy=0.5).shape), "비율 유지하며 절반"),
    ("np.unique(mask)", r(np.unique(mask)), f"원본 마스크 {mask.shape}: 값 2종류"),
    ("len(np.unique(resize(mask, fx=8, INTER_NEAREST)))", r(len(np.unique(up_n))), "값 종류 그대로"),
    ("len(np.unique(resize(mask, fx=8, INTER_LINEAR)))", r(len(np.unique(up_l))), "경계에 중간값이 생김 → 마스크엔 NEAREST"),
])

# 6. blur
noisy = rd("noisy_gray.png", cv2.IMREAD_GRAYSCALE)
clean = rd("clean_gray.png", cv2.IMREAD_GRAYSCALE)


def err(a):
    return f"{np.abs(a.astype(int) - clean.astype(int)).mean():.2f}"


IO["d13-cv2-gaussianblur"] = ("noisy_gray.png를 깨끗한 원본(clean_gray.png)과 비교 — 평균 오차(작을수록 원본에 가까움)", [
    ("noisy (필터 전)", err(noisy), "소금·후추처럼 튀는 점 잡음"),
    ("cv2.GaussianBlur(noisy, (5, 5), 0)", err(cv2.GaussianBlur(noisy, (5, 5), 0)), "오히려 커짐 — 튀는 점을 주변에 번지게 하고 경계도 흐림"),
    ("cv2.medianBlur(noisy, 5)", err(cv2.medianBlur(noisy, 5)), "가장 작음 — 정렬한 가운데 값이라 튀는 점이 그냥 버려짐"),
    ("cv2.bilateralFilter(noisy, 9, 75, 75)", err(cv2.bilateralFilter(noisy, 9, 75, 75)), "조금 커짐 — 튀는 점도 '경계'로 보고 지켜 버림"),
    ("출력 shape · dtype", f"{cv2.medianBlur(noisy, 5).shape} · {cv2.medianBlur(noisy, 5).dtype}", "입력과 같음"),
])

# 7. Canny
def edge_count(src, t1=70, t2=150):
    return int((cv2.Canny(src, t1, t2) > 0).sum())


blur5 = cv2.GaussianBlur(noisy, (5, 5), 0)
IO["d13-cv2-canny"] = ("noisy_gray.png — 경계(흰색)로 표시된 칸 수", [
    ("np.unique(cv2.Canny(blur, 70, 150))", r(np.unique(cv2.Canny(blur5, 70, 150))), "경계 255 / 아님 0 — 1채널"),
    ("Canny(noisy, 70, 150) 경계 칸 수", r(edge_count(noisy)), "Blur 없이: 잡음까지 경계로"),
    ("Canny(GaussianBlur(noisy, (5,5), 0), 70, 150)", r(edge_count(blur5)), "Blur 후: 잡음 경계가 줄어듦"),
    ("Canny(blur, 30, 80)", r(edge_count(blur5, 30, 80)), "기준을 낮추면 경계 ↑"),
    ("Canny(blur, 150, 300)", r(edge_count(blur5, 150, 300)), "기준을 높이면 경계 ↓"),
])

# 8. threshold
ret1, bin1 = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
gb = cv2.GaussianBlur(gray, (5, 5), 0)
ret2, bin2 = cv2.threshold(gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
pct = lambda a: f"{(a > 0).mean() * 100:.1f}%"
IO["d13-cv2-threshold"] = ("scene.png의 gray — 흰색(255)이 된 칸 비율", [
    ("cv2.threshold(gray, 140, 255, THRESH_BINARY)", f"retval={ret1:g}, 흰 칸 {pct(bin1)}", "기준을 내가 정함 → retval도 140"),
    ("np.unique(binary)", r(np.unique(bin1)), "0과 255 두 값만"),
    ("cv2.threshold(blur, 0, 255, BINARY + OTSU)", f"retval={ret2:g}, 흰 칸 {pct(bin2)}", "0은 무시되고 Otsu가 고른 기준이 retval"),
    ("cv2.adaptiveThreshold(gray, 255, GAUSSIAN_C, BINARY, 11, 2)", f"흰 칸 {pct(ad)}", "칸마다 주변 기준 — 반환값은 이미지 하나"),
])

# 9. morphology
k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
obj = cv2.cvtColor(rd("objects.png"), cv2.COLOR_BGR2GRAY)
_, ob = cv2.threshold(obj, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
op = cv2.morphologyEx(ob, cv2.MORPH_OPEN, k5)
cl = cv2.morphologyEx(op, cv2.MORPH_CLOSE, k5)
ncc = lambda a: cv2.connectedComponents(a)[0] - 1
IO["d13-cv2-getstructuringelement"] = ("objects.png를 Otsu로 이진화한 뒤 — 흰 덩어리 개수(connectedComponents)", [
    ("cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5))", "\n".join(" ".join(str(v) for v in row) for row in k5), "0/1로 된 원 모양 Kernel"),
    ("binary 의 흰 덩어리 수", r(ncc(ob)), "작은 잡음 점까지 덩어리로 셈"),
    ("morphologyEx(binary, MORPH_OPEN, kernel)", r(ncc(op)), "Opening: 작은 흰 점 제거"),
    ("morphologyEx(opened, MORPH_CLOSE, kernel)", r(ncc(cl)), "Closing: 물체 속 작은 구멍 메움"),
    ("흰 칸 수 binary → opened → closed", f"{int((ob > 0).sum())} → {int((op > 0).sum())} → {int((cl > 0).sum())}", "열기는 점을 지우고, 닫기는 틈을 메워 다시 늘어남"),
])

# 10. video
cap = cv2.VideoCapture(str(TMP / "data" / "sample.avi"))
opened = cap.isOpened()
fps = cap.get(cv2.CAP_PROP_FPS)
nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, fr = cap.read()
size = (160, 120)
wr = cv2.VideoWriter(str(OUT / "edge.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
wopen = wr.isOpened()
n = 0
while ok:
    s = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA)
    wr.write(cv2.cvtColor(cv2.Canny(cv2.GaussianBlur(s, (3, 3), 0), 80, 160), cv2.COLOR_GRAY2BGR))
    n += 1
    ok, fr = cap.read()
last_ok = ok
cap.release()
wr.release()
c2 = cv2.VideoCapture(str(OUT / "edge.avi"))
n2 = int(c2.get(cv2.CAP_PROP_FRAME_COUNT))
_, f2 = c2.read()
c2.release()
bad = cv2.VideoWriter(str(OUT / "bad.avi"), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
bad.write(np.zeros((240, 320, 3), np.uint8))  # frameSize와 다른 크기
bad.release()
c3 = cv2.VideoCapture(str(OUT / "bad.avi"))
n3 = int(c3.get(cv2.CAP_PROP_FRAME_COUNT)) if c3.isOpened() else 0
c3.release()
IO["d13-cv2-videocapture"] = ("data/sample.avi", [
    ("cap.isOpened()", r(opened), "열기 성공"),
    ("cap.get(cv2.CAP_PROP_FPS)", r(float(fps)), "초당 Frame 수"),
    ("cap.get(cv2.CAP_PROP_FRAME_COUNT)", r(nfr), "전체 Frame 수"),
    ("ok, frame = cap.read()  → frame.shape", f"True, {f2.shape if False else rd('scene.png').shape if False else ''}".strip(", ") or "", ""),
    ("반복으로 처리·저장한 Frame 수", r(n), "마지막 다음 read()는 ok=False → 반복 끝"),
    ("writer.isOpened()", r(wopen), "MJPG + .avi 조합 OK"),
    ("저장한 edge.avi 다시 읽기: Frame 수 · shape", f"{n2} · {f2.shape}", "(160, 120)으로 줄여 3채널로 저장됨"),
    ("frameSize=(160,120)인데 (320,240) Frame을 write", f"저장된 Frame 수 {n3}", "에러 없이 조용히 실패!"),
])
# frame.shape 줄은 첫 Frame shape로 다시 채움
cap = cv2.VideoCapture(str(TMP / "data" / "sample.avi"))
_, fr0 = cap.read()
cap.release()
IO["d13-cv2-videocapture"][1][3] = ("ok, frame = cap.read()", f"ok → True, frame.shape → {fr0.shape}", "Frame 한 장 = 이미지 배열")

# 11. flip
def red_box(a):
    m = (a[:, :, 2] == 255) & (a[:, :, 1] == 0) & (a[:, :, 0] == 0)
    ys, xs = np.where(m)
    return f"x {xs.min()}~{xs.max()}, y {ys.min()}~{ys.max()}"


IO["d13-cv2-flip"] = (f"빨간 사각형이 있는 자리 (가로 {W}, 세로 {H})", [
    ("img (원본)", red_box(img), ""),
    ("cv2.flip(img, 1)", red_box(cv2.flip(img, 1)), f"좌우: x → {W - 1} - x"),
    ("cv2.flip(img, 0)", red_box(cv2.flip(img, 0)), f"상하: y → {H - 1} - y"),
    ("cv2.flip(img, -1)", red_box(cv2.flip(img, -1)), "둘 다"),
    ("cv2.flip(img, 1).shape", r(cv2.flip(img, 1).shape), "크기는 그대로"),
])

# 12. rotate
M = cv2.getRotationMatrix2D((W / 2, H / 2), 10, 1.0)
rot_c = cv2.warpAffine(img, M, (W, H))
rot_r = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)
IO["d13-getrotationmatrix2d"] = ("scene.png, 가운데 기준 10° 회전", [
    ("cv2.getRotationMatrix2D((160, 120), 10, 1.0)", np.array2string(M, precision=3, suppress_small=True), "2×3 행렬: [cos sin 이동x; -sin cos 이동y]"),
    ("warpAffine(img, M, (320, 240)).shape", r(rot_c.shape), "dsize 그대로 → 모서리가 잘림"),
    ("기본(BORDER_CONSTANT) 왼쪽 위 칸 rot[0, 0]", r(rot_c[0, 0]), "빈 곳은 검정(0)으로 채움"),
    ("BORDER_REFLECT 왼쪽 위 칸 rot[0, 0]", r(rot_r[0, 0]), "빈 곳을 거울처럼 비친 값으로"),
])

# 13. brightness / crop
a = np.array([[100, 250]], np.uint8)
crop = img[H // 10: H - H // 10, W // 10: W - W // 10]
IO["d13-밝기-crop-blur"] = ("값 [100, 250]인 uint8 두 칸 · scene.png", [
    ("a + 30   (NumPy 그냥 더하기)", r(a + np.uint8(30)), "250+30=280 → 256을 넘겨 24로 돌아감!"),
    ("cv2.add(a, 30)", r(cv2.add(a, np.array([[30, 30]], np.uint8))), "255에서 멈춤(포화)"),
    ("cv2.convertScaleAbs(a, alpha=1.0, beta=30)", r(cv2.convertScaleAbs(a, alpha=1.0, beta=30)), "255에서 멈춤"),
    ("img[h//10 : h-h//10, w//10 : w-w//10].shape", r(crop.shape), "위아래·좌우 10%씩 잘라냄"),
    ("cv2.resize(crop, (w, h)).shape", r(cv2.resize(crop, (W, H)).shape), "다시 원래 크기로 → 확대 효과"),
])

# 14. batch
names = [p.name for p in sorted((TMP / "data" / "images").glob("*.png"))]
od = OUT / "images"
od.mkdir()
for p in sorted((TMP / "data" / "images").glob("*.png")):
    im = cv2.imread(str(p))
    cv2.imwrite(str(od / p.name), cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (320, 240)))
first = sorted((TMP / "data" / "images").glob("*.png"))[0]
IO["d13-sorted-path-glob"] = ("data/images 폴더", [
    ('[p.name for p in sorted(Path("data/images").glob("*.png"))]', r(names), "이름 순서로 고정"),
    (f'cv2.imread("data/images/{first.name}").shape', r(cv2.imread(str(first)).shape), "원본"),
    (f'cv2.imread("outputs/images/{first.name}", 0).shape', r(cv2.imread(str(od / first.name), 0).shape), "Gray + (320, 240)으로 저장된 것 확인"),
    ("len(list(Path(\"outputs/images\").glob(\"*.png\")))", r(len(list(od.glob("*.png")))), "입력 개수와 같아야 함"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day13.html"
s = frag.read_text(encoding="utf-8")
s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
for cid, (title, rows) in IO.items():
    m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</code></pre>\n', s, re.S)
    assert m, cid
    s = s[:m.end()] + box(title, rows) + "\n" + s[m.end():]
frag.write_text(s, encoding="utf-8")
for cid, (title, rows) in IO.items():
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i}  →  {o}   ({n})")
shutil.rmtree(TMP, ignore_errors=True)
