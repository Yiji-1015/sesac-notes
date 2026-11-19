# 객체 탐지의 박스 다루기 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/labels/*/*.txt   YOLO 형식 라벨 (종류, 중심 x, 중심 y, 너비, 높이 — 0~1 비율)
#         data/boxes_gt.csv      scene.png의 정답 박스 3개
#         data/boxes_fixture.csv 원리 연습용으로 손으로 정한 예측 박스 4개 (YOLO 예측 아님)
#         data/track_ground_truth.csv  영상 30프레임의 물체 위치와 ID
# 좌표는 모두 [x1, y1, x2, y2) 픽셀 (오른쪽, 아래 경계는 포함 안 함)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import pandas as pd                                              #@1
import torch                                                     #@4
from torchvision.ops import nms                                  #@4

DATA = Path(__file__).resolve().parent / "data"                  #@1
NAMES = {0: "square", 1: "circle", 2: "triangle"}                #@1

# ① YOLO 라벨을 픽셀 박스로
def yolo_to_xyxy(cx, cy, w, h, img_w=160, img_h=128):
    return (round((cx - w / 2) * img_w), round((cy - h / 2) * img_h),
            round((cx + w / 2) * img_w), round((cy + h / 2) * img_h))

label_file = DATA / "labels" / "train" / "basic_train_000.txt"
label_rows = [line.split() for line in label_file.read_text().splitlines()]
pixel_boxes = [(NAMES[int(c)], yolo_to_xyxy(*map(float, rest))) for c, *rest in label_rows]

# ② 겹친 정도 재기 (IoU)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union

gt = pd.read_csv(DATA / "boxes_gt.csv", encoding="utf-8-sig")
pred = pd.read_csv(DATA / "boxes_fixture.csv", encoding="utf-8-sig")
box = lambda r: (r.x1, r.y1, r.x2, r.y2)  # noqa: E731
ious = [[round(iou(box(p), box(g)), 3) for g in gt.itertuples()] for p in pred.itertuples()]

# ③ 예측과 정답 짝짓기
def match(pred, gt, thr=0.5):
    used, rows = set(), []
    for p in pred.sort_values("confidence", ascending=False).itertuples():  # 확신 높은 예측부터
        best, best_iou = None, thr
        for j, g in enumerate(gt.itertuples()):
            if j in used or g.class_id != p.class_id:
                continue
            v = iou(box(p), box(g))
            if v >= best_iou:
                best, best_iou = j, v
        if best is None:
            rows.append((p.confidence, NAMES[p.class_id], "FP"))   # 짝이 없는 예측 = 헛탐지
        else:
            used.add(best)
            rows.append((p.confidence, NAMES[p.class_id], "TP"))
    fn = len(gt) - len(used)                                       # 짝을 못 찾은 정답 = 놓침
    tp = sum(r[2] == "TP" for r in rows)
    return rows, {"TP": tp, "FP": len(rows) - tp, "FN": fn,
                  "precision": tp / len(rows), "recall": tp / len(gt)}

matched, counts = match(pred, gt)

# ④ 겹친 후보 정리하기 (NMS)
boxes = torch.tensor(pred[["x1", "y1", "x2", "y2"]].values, dtype=torch.float32)
scores = torch.tensor(pred["confidence"].values, dtype=torch.float32)
keep = nms(boxes, scores, iou_threshold=0.5).tolist()            # 실제로는 종류별로 따로 해요
after_nms = pred.iloc[keep]
matched_nms, counts_nms = match(after_nms, gt)

# ⑤ 확신 기준을 바꾸며 보기 (Precision-Recall)
pr_table = [{"기준": t, **match(after_nms[after_nms.confidence >= t], gt)[1]} for t in (0.9, 0.7, 0.3)]

# ⑥ 영상에서 같은 물체 이어 붙이기 (추적)
track = pd.read_csv(DATA / "track_ground_truth.csv", encoding="utf-8-sig")

def iou_tracker(frames, thr=0.3):
    tracks, next_id, out = {}, 1, []                     # tracks: ID → 마지막 박스
    for f in sorted(frames.frame.unique()):
        dets = frames[frames.frame == f].sample(frac=1, random_state=int(f))  # 탐지 결과에는 순서도 ID도 없어요
        new = {}
        for d in dets.itertuples():
            cand = [(iou(box(d), b), tid) for tid, b in tracks.items() if tid not in new]
            v, tid = max(cand, default=(0, None))
            if v < thr:                                  # 이어 붙일 곳이 없으면 새 ID
                tid, next_id = next_id, next_id + 1
            new[tid] = box(d)
            out.append((f, d.gt_id, tid))
        tracks = new
    return pd.DataFrame(out, columns=["frame", "gt_id", "track_id"])

def id_switches(result):
    return int(sum(g.track_id.nunique() - 1 for _, g in result.groupby("gt_id")))  # 한 물체에 붙은 ID가 몇 번 바뀌었나

tracked = iou_tracker(track)
switches = {gap: id_switches(iou_tracker(track[track.frame % gap == 0]))   # gap 프레임에 한 번만 탐지하면
            for gap in (1, 5, 10, 20)}
