"""Day15 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

데이터: 수업 합성 도형 데이터(C:/practice_assignment/20261106/01_기본필수/data, 160x128, 3클래스)를
임시 폴더에 복사해서 쓴다(원본은 읽기만). 모델은 make_day15.py 와 같은 작은 데모 모델
(DAY15_RUN/runs/ep100/weights/best.pt, 합성 도형으로 100 epoch 학습한 YOLO26n)을 쓴다.
모델이 없으면 모델이 필요한 항목은 건너뛴다. 큰 가중치는 내려받지 않는다.
결과는 cmds/day15.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day15.py
"""
import csv
import html
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert, box_iou

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261106/01_기본필수/data")
RUN = Path(os.environ.get(
    "DAY15_RUN",
    r"C:/Users/pc_28/AppData/Local/Temp/claude/e384ea84ee9f536f/4c44c0e3-ebd2-483a-a969-2ab2aed04c8a/scratchpad/day15run"))
WEIGHTS_SRC = RUN / "runs/ep100/weights/best.pt"
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")
DATA = TMP / "data"
OUT = TMP / "outputs"
OUT.mkdir()
HAVE_MODEL = WEIGHTS_SRC.exists()
if HAVE_MODEL:
    shutil.copy(WEIGHTS_SRC, TMP / "best.pt")
WEIGHTS = TMP / "best.pt"
os.chdir(TMP)
# dataset.yaml 의 path: . 는 '현재 폴더' 기준이라, 복사본 폴더를 절대 경로로 적은 yaml을 따로 만든다
YAML = TMP / "dataset.yaml"
YAML.write_text((DATA / "dataset.yaml").read_text(encoding="utf-8").replace("path: .", f"path: {DATA.as_posix()}"),
                encoding="utf-8")


def read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def arr(a, prec=2):
    """ultralytics가 numpy 출력 형식을 바꿔 두므로 직접 반올림해서 리스트로 보여 준다."""
    a = np.asarray(a)
    if a.dtype.kind in "iub":
        return repr(a.tolist())
    return repr(np.round(a.astype(float), prec).tolist()).replace(".0,", ",").replace(".0]", "]")


IO = {}

# 1. 박스 그리기
gt = read_csv(DATA / "boxes_gt.csv")
img = cv2.imread(str(DATA / "scene.png"))
before = img[18, 12].copy()
x1, y1, x2, y2 = (int(gt[0][k]) for k in ("x1", "y1", "x2", "y2"))
ret = cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
ret2 = cv2.putText(img, "square", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
mid = img[(y1 + y2) // 2, (x1 + x2) // 2]
ok = cv2.imwrite(str(OUT / "detected.png"), img)
IO["d15-박스-그리기"] = (f"data/scene.png {img.shape} · boxes_gt.csv 첫 박스 ({x1}, {y1}, {x2}, {y2})", [
    (f"img[{y1}, {x1}]  (그리기 전)", arr(before), "원래 그 자리 색"),
    (f"cv2.rectangle(img, ({x1}, {y1}), ({x2}, {y2}), color=(255, 0, 0), thickness=2)", f"img[{y1}, {x1}] → {arr(img[y1, x1])}",
     "B=255 → 파랑 선이 그려짐 (빨강 아님!)"),
    ("ret is img", repr(ret is img), "새 그림을 만드는 게 아니라 원본 배열에 바로 그려요"),
    (f"img[{(y1 + y2) // 2}, {(x1 + x2) // 2}]  (박스 한가운데)", arr(mid), "선만 그리니 안쪽 색은 그대로"),
    ('cv2.putText(img, "square", (x1, y1 - 8), ...).shape', repr(ret2.shape), "글자도 같은 배열에 덧그림"),
    ('cv2.imwrite("outputs/detected.png", img)', repr(ok), "저장 성공"),
])

# 2. box_convert
boxes = torch.tensor([[12, 18, 52, 58]], dtype=torch.float32)
cxcywh = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
labels = cxcywh / torch.tensor([160, 128, 160, 128])
xywh = box_convert(boxes, in_fmt="xyxy", out_fmt="xywh")
back = box_convert(cxcywh, in_fmt="cxcywh", out_fmt="xyxy")
IO["d15-box-convert"] = ("scene.png(가로 160 · 세로 128)의 square 정답 박스 [12, 18, 52, 58]", [
    ('box_convert(boxes, "xyxy", "cxcywh")', arr(cxcywh.numpy(), 1), "중심 (32, 38) · 폭 40 · 높이 40"),
    ('box_convert(boxes, "xyxy", "xywh")', arr(xywh.numpy(), 1), "왼쪽 위 (12, 18) + 크기"),
    ("cxcywh / [160, 128, 160, 128]", arr(labels.numpy(), 4), "YOLO 라벨 한 줄의 숫자 4개 (0~1)"),
    ('box_convert(cxcywh, "cxcywh", "xyxy")', arr(back.numpy(), 1), "되돌리면 원래 좌표"),
])

# 3. box_iou
fx = read_csv(DATA / "boxes_fixture.csv")
pred_boxes = torch.tensor([[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in fx])
gt_boxes = torch.tensor([[float(r[k]) for k in ("x1", "y1", "x2", "y2")] for r in gt])
iou = box_iou(pred_boxes, gt_boxes)
matched = iou >= 0.5
IO["d15-box-iou"] = (f"boxes_fixture.csv 예측 {len(fx)}개 × boxes_gt.csv 정답 {len(gt)}개 (손으로 정한 연습값)", [
    ("box_iou(pred_boxes, gt_boxes).shape", repr(tuple(iou.shape)), "[예측 수, 정답 수]"),
    ("box_iou(pred_boxes, gt_boxes)", arr(iou.numpy(), 3), "행 = 예측, 열 = 정답"),
    ("iou >= 0.5", arr(matched.numpy().astype(int)), "예측 0·1이 둘 다 정답 0과 겹침 → 하나만 TP, 나머지는 중복 FP"),
    ("예측 3 (90,6,110,22) 줄", arr(iou[3].numpy(), 3), "어떤 정답과도 안 겹침 → FP, 정답 2는 못 찾음 → FN"),
])

# ---------------------------------------------------------------- 모델이 필요한 항목
if HAVE_MODEL:
    from ultralytics import YOLO

    # 4. model.val
    model = YOLO(str(WEIGHTS))
    m = model.val(data=str(YAML), split="val", imgsz=160, conf=0.001, iou=0.7,
                  device="cpu", plots=False, verbose=False, project=str(OUT), name="val")
    m2 = YOLO(str(WEIGHTS)).val(data=str(YAML), split="val", imgsz=160, conf=0.5, iou=0.7,
                                device="cpu", plots=False, verbose=False, project=str(OUT), name="val2")
    IO["d15-model-val"] = ("데모 모델 best.pt(train 12장으로 100 epoch 학습한 작은 모델) · dataset.yaml 의 val 6장(정답 18개) · imgsz=160", [
        ("metrics.box.map50", f"{m.box.map50:.3f}", "IoU 0.5 기준 평균 AP — 작은 데모라 높지 않아요"),
        ("metrics.box.map", f"{m.box.map:.3f}", "IoU 0.5~0.95 평균 → 박스를 꼼꼼히 맞춰야 해서 더 낮음"),
        ("metrics.box.mp, metrics.box.mr", f"({m.box.mp:.3f}, {m.box.mr:.3f})", "낸 박스는 대체로 맞지만(P↑) 놓치는 게 많음(R↓)"),
        ("metrics.box.maps", arr(m.box.maps, 3), "클래스별 mAP50-95 (square, circle, triangle) — circle은 하나도 못 맞힘"),
        ("conf=0.5로 val 했을 때 metrics.box.map", f"{m2.box.map:.3f}",
         "이 모델은 0.5를 넘는 후보가 거의 없어 PR 곡선이 통째로 잘림 → 검증은 conf=0.001"),
    ])

    # 5. YOLO() / train
    empty = YOLO("yolo26n.yaml")
    nparam = lambda mm: sum(p.numel() for p in mm.model.parameters())
    n_best, n_empty_p = nparam(YOLO(str(WEIGHTS))), nparam(empty)   # 추론 전에 세기(추론하면 층이 합쳐짐)
    tr_model = YOLO("yolo26n.yaml")
    tr_model.train(data=str(YAML), epochs=1, imgsz=160, batch=4, device="cpu", workers=0,
                   plots=False, verbose=False, project=str(OUT), name="train", seed=0, val=True)
    tdir = OUT / "train"
    made = sorted(p.relative_to(tdir).as_posix() for p in tdir.rglob("*") if p.is_file())
    pv = cv2.imread(str(DATA / "images/val/basic_val_000.png"))
    n_trained = len(model.predict(pv, imgsz=160, device="cpu", verbose=False)[0].boxes)
    n_empty = len(empty.predict(pv, imgsz=160, device="cpu", verbose=False)[0].boxes)
    IO["d15-yolo"] = ("데모 모델 best.pt vs 빈 뇌 yolo26n.yaml · val 사진 basic_val_000.png(정답 3개)", [
        ('YOLO("best.pt").task, .names', f"{model.task!r}, {model.names}", "탐지 모델 · 클래스 이름이 들어 있음"),
        ('YOLO("best.pt") / YOLO("yolo26n.yaml") 파라미터 수', f"{n_best:,} / {n_empty_p:,}",
         "같은 구조 — .yaml은 기본 80클래스라 출력 층만 조금 큼. 차이는 결국 '가중치 값'"),
        ('YOLO("best.pt").predict(사진) 박스 수', repr(n_trained), "학습된 뇌: 3개 중 1개만 — 12장으로 배운 작은 데모라 아직 약해요"),
        ('YOLO("yolo26n.yaml").predict(사진) 박스 수', repr(n_empty), "무작위 가중치라 점수가 conf 0.25를 못 넘음"),
        ('YOLO("yolo26n.yaml").train(data=..., epochs=1, imgsz=160, device="cpu", workers=0, plots=False) 후 생긴 파일', " · ".join(made),
         "1 epoch만 돌려도 weights/best.pt · last.pt · results.csv가 생김 (plots=False라 그림은 없음)"),
    ])

    # 6. predict
    vdir = DATA / "images/val"
    results = model.predict(source=str(vdir), conf=0.25, iou=0.7, imgsz=160, save=True, device="cpu",
                            verbose=False, project=str(OUT), name="predict")
    per = [len(r.boxes) for r in results]
    low = [len(r.boxes) for r in model.predict(source=str(vdir), conf=0.01, iou=0.7, imgsz=160, device="cpu", verbose=False)]
    tight = [len(r.boxes) for r in model.predict(source=str(vdir), conf=0.01, iou=0.1, imgsz=160, device="cpu", verbose=False)]
    ngt = [len([l for l in (DATA / "labels/val" / (Path(r.path).stem + ".txt")).read_text().split("\n") if l.strip()])
           for r in results]
    saved = sorted(p.name for p in (OUT / "predict").glob("*"))
    IO["d15-model-predict"] = ("데모 모델 best.pt · source=data/images/val (6장) · imgsz=160", [
        ("len(results)", repr(len(results)), "사진 1장 = result 1개"),
        ("사진별 정답 박스 수 (라벨 파일)", repr(ngt), "비교 기준"),
        ("conf=0.25, iou=0.7 → 사진별 박스 수", repr(per), "정답보다 적음 — 데모 모델 점수가 대부분 0.25 아래라 버려짐"),
        ("conf=0.01, iou=0.7 → 사진별 박스 수", repr(low), "기준을 낮추면 점수 낮은 후보까지 남아 박스가 늘어남(정답보다 많아지기도)"),
        ("conf=0.01, iou=0.1 → 사진별 박스 수", repr(tight), "NMS 기준을 빡빡하게 하면 조금만 겹쳐도 지워져 다시 줄어듦"),
        ("save=True 로 저장된 파일", f"{len(saved)}개 ({saved[0]} …)", "박스 그린 사진이 runs/.../predict 폴더에"),
    ])

    # 7. result.boxes
    r0 = results[0]
    bx = r0.boxes
    c0 = int(bx.cls[0])
    lab = np.loadtxt(DATA / "labels/val" / (Path(r0.path).stem + ".txt"), ndmin=2)
    lab_c = lab[lab[:, 0] == c0][0]
    empty_r = model.predict(np.full((128, 160, 3), 255, np.uint8), imgsz=160, device="cpu", verbose=False)[0]
    IO["d15-result-boxes"] = (f"model.predict(source=val, conf=0.25) 결과의 첫 사진 {Path(r0.path).name} — 박스 {len(bx)}개", [
        ("result.boxes.xyxy.cpu().numpy()", arr(bx.xyxy.cpu().numpy(), 1), "픽셀 좌표 [N, 4]"),
        ("result.boxes.conf.cpu().numpy()", arr(bx.conf.cpu().numpy(), 3), "점수"),
        ("result.boxes.cls.cpu().numpy().astype(int)", arr(bx.cls.cpu().numpy().astype(int)), "클래스 번호"),
        ("result.names", repr(r0.names), "번호 → 이름"),
        ("result.boxes.xywhn.cpu().numpy()", arr(bx.xywhn.cpu().numpy(), 3), "정규화 중심 좌표"),
        (f"정답 라벨 파일에서 같은 클래스({c0}) 줄의 cx cy w h", arr(lab_c[1:], 3), "위 xywhn과 비슷 → 라벨과 같은 형식 (박스는 조금 작게 잡음)"),
        ("result.plot().shape, dtype", f"{r0.plot().shape}, {r0.plot().dtype}", "박스 그린 BGR 이미지 → cv2.imwrite로 저장"),
        ("하얀 빈 그림의 len(result.boxes)", repr(len(empty_r.boxes)), "탐지 0개 — 에러가 아니라 정상"),
    ])

    # 8. track
    rows = read_csv(DATA / "track_ground_truth.csv")
    ngt_ids = len({r["gt_id"] for r in rows})
    tm = YOLO(str(WEIGHTS))
    ids_per = []
    for res in tm.track(source=str(DATA / "demo.avi"), persist=True, tracker="botsort.yaml", stream=True,
                        imgsz=160, device="cpu", verbose=False):
        ids_per.append(res.boxes.id.int().tolist() if res.boxes.id is not None else None)
    flat = [i for x in ids_per if x for i in x]
    nnone = sum(x is None for x in ids_per)
    tm2 = YOLO(str(WEIGHTS))
    ids_bt = set()
    for res in tm2.track(source=str(DATA / "demo.avi"), persist=True, tracker="bytetrack.yaml", stream=True,
                         imgsz=160, device="cpu", verbose=False):
        if res.boxes.id is not None:
            ids_bt |= set(res.boxes.id.int().tolist())
    # 같은 객체로 두 번째 영상(같은 파일)을 이어서 추적하면 ID가 이어서 커짐
    again = set()
    for res in tm2.track(source=str(DATA / "demo.avi"), persist=True, tracker="bytetrack.yaml", stream=True,
                         imgsz=160, device="cpu", verbose=False):
        if res.boxes.id is not None:
            again |= set(res.boxes.id.int().tolist())
    first = next(i for i, x in enumerate(ids_per) if x)
    IO["d15-model-track"] = (f"데모 모델 best.pt · data/demo.avi ({len(ids_per)}프레임, 정답 물체 {ngt_ids}개) · imgsz=160", [
        ("frame 0 의 result.boxes.id", repr(ids_per[0]), "첫 프레임은 ID 없음 — 박스 점수가 새 Track 시작 기준(0.25) 근처로 낮아서"),
        ("ID가 처음 붙은 프레임", f"frame {first} → {ids_per[first]}", "점수가 오른 뒤에야 Track이 시작됨"),
        ("frame 29 의 result.boxes.id.int().tolist()", repr(ids_per[-1]), "2·6번은 끝까지 같은 번호 — 3번은 이 프레임에서 탐지를 놓쳐 빠짐"),
        ('botsort: 나온 ID 종류 · id가 None인 프레임 수', f"{sorted(set(flat))} · {nnone}",
         "번호가 1부터가 아님(확정 못 된 후보도 번호를 씀). 탐지를 놓친 프레임은 이을 박스가 없음"),
        ('tracker="bytetrack.yaml" 의 ID 종류', repr(sorted(ids_bt)), "이 영상에선 두 추적기 결과가 같았어요"),
        ("같은 YOLO 객체로 영상을 한 번 더 track", repr(sorted(again)), "기억이 남아 번호가 이어서 커짐 → 다른 영상엔 새 YOLO 객체"),
    ])
else:
    print("!! 데모 모델이 없어 model.val / YOLO / predict / result.boxes / track 상자는 건너뜀:", WEIGHTS_SRC)


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day15.html"
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
os.chdir(ROOT)
shutil.rmtree(TMP, ignore_errors=True)
