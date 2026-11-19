window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import pandas as pd",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 4
  },
  {
   "t": "from torchvision.ops import nms",
   "s": 4
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\"",
   "s": 1
  },
  {
   "t": "NAMES = {0: \"square\", 1: \"circle\", 2: \"triangle\"}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① YOLO 라벨을 픽셀 박스로",
   "s": 1
  },
  {
   "t": "def yolo_to_xyxy(cx, cy, w, h, img_w=160, img_h=128):",
   "s": 1
  },
  {
   "t": "    return (round((cx - w / 2) * img_w), round((cy - h / 2) * img_h),",
   "s": 1
  },
  {
   "t": "            round((cx + w / 2) * img_w), round((cy + h / 2) * img_h))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "label_file = DATA / \"labels\" / \"train\" / \"basic_train_000.txt\"",
   "s": 1
  },
  {
   "t": "label_rows = [line.split() for line in label_file.read_text().splitlines()]",
   "s": 1
  },
  {
   "t": "pixel_boxes = [(NAMES[int(c)], yolo_to_xyxy(*map(float, rest))) for c, *rest in label_rows]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 겹친 정도 재기 (IoU)",
   "s": 2
  },
  {
   "t": "def iou(a, b):",
   "s": 2
  },
  {
   "t": "    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))",
   "s": 2
  },
  {
   "t": "    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))",
   "s": 2
  },
  {
   "t": "    inter = ix * iy",
   "s": 2
  },
  {
   "t": "    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter",
   "s": 2
  },
  {
   "t": "    return inter / union",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "gt = pd.read_csv(DATA / \"boxes_gt.csv\", encoding=\"utf-8-sig\")",
   "s": 2
  },
  {
   "t": "pred = pd.read_csv(DATA / \"boxes_fixture.csv\", encoding=\"utf-8-sig\")",
   "s": 2
  },
  {
   "t": "box = lambda r: (r.x1, r.y1, r.x2, r.y2)  # noqa: E731",
   "s": 2
  },
  {
   "t": "ious = [[round(iou(box(p), box(g)), 3) for g in gt.itertuples()] for p in pred.itertuples()]",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 예측과 정답 짝짓기",
   "s": 3
  },
  {
   "t": "def match(pred, gt, thr=0.5):",
   "s": 3
  },
  {
   "t": "    used, rows = set(), []",
   "s": 3
  },
  {
   "t": "    for p in pred.sort_values(\"confidence\", ascending=False).itertuples():  # 확신 높은 예측부터",
   "s": 3
  },
  {
   "t": "        best, best_iou = None, thr",
   "s": 3
  },
  {
   "t": "        for j, g in enumerate(gt.itertuples()):",
   "s": 3
  },
  {
   "t": "            if j in used or g.class_id != p.class_id:",
   "s": 3
  },
  {
   "t": "                continue",
   "s": 3
  },
  {
   "t": "            v = iou(box(p), box(g))",
   "s": 3
  },
  {
   "t": "            if v >= best_iou:",
   "s": 3
  },
  {
   "t": "                best, best_iou = j, v",
   "s": 3
  },
  {
   "t": "        if best is None:",
   "s": 3
  },
  {
   "t": "            rows.append((p.confidence, NAMES[p.class_id], \"FP\"))  # 짝이 없는 예측 = 헛탐지",
   "s": 3
  },
  {
   "t": "        else:",
   "s": 3
  },
  {
   "t": "            used.add(best)",
   "s": 3
  },
  {
   "t": "            rows.append((p.confidence, NAMES[p.class_id], \"TP\"))",
   "s": 3
  },
  {
   "t": "    fn = len(gt) - len(used)  # 짝을 못 찾은 정답 = 놓침",
   "s": 3
  },
  {
   "t": "    tp = sum(r[2] == \"TP\" for r in rows)",
   "s": 3
  },
  {
   "t": "    return rows, {\"TP\": tp, \"FP\": len(rows) - tp, \"FN\": fn,",
   "s": 3
  },
  {
   "t": "                  \"precision\": tp / len(rows), \"recall\": tp / len(gt)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "matched, counts = match(pred, gt)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 겹친 후보 정리하기 (NMS)",
   "s": 4
  },
  {
   "t": "boxes = torch.tensor(pred[[\"x1\", \"y1\", \"x2\", \"y2\"]].values, dtype=torch.float32)",
   "s": 4
  },
  {
   "t": "scores = torch.tensor(pred[\"confidence\"].values, dtype=torch.float32)",
   "s": 4
  },
  {
   "t": "keep = nms(boxes, scores, iou_threshold=0.5).tolist()  # 실제로는 종류별로 따로 해요",
   "s": 4
  },
  {
   "t": "after_nms = pred.iloc[keep]",
   "s": 4
  },
  {
   "t": "matched_nms, counts_nms = match(after_nms, gt)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 확신 기준을 바꾸며 보기 (Precision-Recall)",
   "s": 5
  },
  {
   "t": "pr_table = [{\"기준\": t, **match(after_nms[after_nms.confidence >= t], gt)[1]} for t in (0.9, 0.7, 0.3)]",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 영상에서 같은 물체 이어 붙이기 (추적)",
   "s": 6
  },
  {
   "t": "track = pd.read_csv(DATA / \"track_ground_truth.csv\", encoding=\"utf-8-sig\")",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def iou_tracker(frames, thr=0.3):",
   "s": 6
  },
  {
   "t": "    tracks, next_id, out = {}, 1, []  # tracks: ID → 마지막 박스",
   "s": 6
  },
  {
   "t": "    for f in sorted(frames.frame.unique()):",
   "s": 6
  },
  {
   "t": "        dets = frames[frames.frame == f].sample(frac=1, random_state=int(f))  # 탐지 결과에는 순서도 ID도 없어요",
   "s": 6
  },
  {
   "t": "        new = {}",
   "s": 6
  },
  {
   "t": "        for d in dets.itertuples():",
   "s": 6
  },
  {
   "t": "            cand = [(iou(box(d), b), tid) for tid, b in tracks.items() if tid not in new]",
   "s": 6
  },
  {
   "t": "            v, tid = max(cand, default=(0, None))",
   "s": 6
  },
  {
   "t": "            if v < thr:  # 이어 붙일 곳이 없으면 새 ID",
   "s": 6
  },
  {
   "t": "                tid, next_id = next_id, next_id + 1",
   "s": 6
  },
  {
   "t": "            new[tid] = box(d)",
   "s": 6
  },
  {
   "t": "            out.append((f, d.gt_id, tid))",
   "s": 6
  },
  {
   "t": "        tracks = new",
   "s": 6
  },
  {
   "t": "    return pd.DataFrame(out, columns=[\"frame\", \"gt_id\", \"track_id\"])",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def id_switches(result):",
   "s": 6
  },
  {
   "t": "    return int(sum(g.track_id.nunique() - 1 for _, g in result.groupby(\"gt_id\")))  # 한 물체에 붙은 ID가 몇 번 바뀌었나",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "tracked = iou_tracker(track)",
   "s": 6
  },
  {
   "t": "switches = {gap: id_switches(iou_tracker(track[track.frame % gap == 0]))  # gap 프레임에 한 번만 탐지하면",
   "s": 6
  },
  {
   "t": "            for gap in (1, 5, 10, 20)}",
   "s": 6
  }
 ],
 "names": {
  "1": "YOLO 라벨을 픽셀 박스로",
  "2": "겹친 정도 재기 (IoU)",
  "3": "예측과 정답 짝짓기",
  "4": "겹친 후보 정리하기 (NMS)",
  "5": "확신 기준을 바꾸며 보기 (Precision-Recall)",
  "6": "영상에서 같은 물체 이어 붙이기 (추적)"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "why": {
   "title": "scene.png: 정답(초록)과 예측 후보(빨강)",
   "images": [
    [
     "img/detection/scene_all.png",
     "정답 3개, 예측 4개"
    ]
   ],
   "notes": [
    "분류는 그림 하나에 답 하나지만, 탐지는 그림 하나에 <b>여러 물체</b>의 <b>종류와 위치(박스)</b>를 답해요. 그래서 \"맞았다\"를 정하는 것부터 달라요.",
    "예측 박스가 정답과 조금 어긋나 있고, 한 물체에 두 개가 겹쳐 있고, 엉뚱한 곳에 하나가 있어요. 이걸 어떻게 채점하고 정리할지가 이 쪽의 내용이에요."
   ]
  },
  "s1": {
   "title": "① basic_train_000.txt → 픽셀 박스 (그림 160×128)",
   "lines": [
    "0 0.79687500 0.33984375 0.19375000 0.24218750",
    "1 0.18750000 0.25000000 0.16250000 0.20312500",
    "2 0.48125000 0.64062500 0.21250000 0.26562500",
    "",
    "square → (112, 28, 143, 59)",
    "circle → (17, 19, 43, 45)",
    "triangle → (60, 65, 94, 99)"
   ],
   "notes": [
    "YOLO 라벨은 줄마다 \"종류 번호, 박스 중심 x, 중심 y, 너비, 높이\"이고, 위치는 그림 크기에 대한 <b>비율(0~1)</b>이에요. 그림 크기를 바꿔도 라벨을 고칠 필요가 없어요.",
    "그리거나 IoU를 잴 때는 왼쪽 위(x1, y1), 오른쪽 아래(x2, y2) 픽셀로 바꿔요. 가로 세로를 바꿔 곱하면 박스가 엉뚱한 곳에 그려져요."
   ]
  },
  "s1img": {
   "title": "라벨을 그림 위에",
   "images": [
    [
     "img/detection/label.png",
     "basic_train_000.png"
    ]
   ],
   "notes": []
  },
  "s2": {
   "title": "② 예측 4개 × 정답 3개의 IoU",
   "cols": [
    "예측 (점수, 종류)",
    "정답 square",
    "정답 circle",
    "정답 triangle"
   ],
   "rows": [
    [
     "0.95, square",
     "<b>0.902</b>",
     "0.0",
     "0.0"
    ],
    [
     "0.85, square",
     "<b>0.816</b>",
     "0.0",
     "0.0"
    ],
    [
     "0.8, circle",
     "0.0",
     "<b>1.0</b>",
     "0.0"
    ],
    [
     "0.4, triangle",
     "0.0",
     "0.0",
     "0.0"
    ]
   ],
   "notes": [
    "IoU = 겹친 넓이 ÷ 합친 넓이. 1이면 완전히 같고 0이면 안 겹쳐요. 보통 0.5 이상이면 \"같은 물체를 찾았다\"고 봐요.",
    "점수 0.4짜리 세모 예측은 어느 정답과도 겹치지 않아요."
   ]
  },
  "s3": {
   "title": "③ match(pred, gt)  (IoU 0.5, 같은 종류, 점수 높은 것부터)",
   "lines": [
    "점수 0.95, square → TP",
    "점수 0.85, square → FP",
    "점수 0.8, circle → TP",
    "점수 0.4, triangle → FP",
    "",
    [
     "TP 2, FP 2, FN 1 → precision 0.50, recall 0.67",
     1
    ]
   ],
   "notes": [
    "두 번째 네모(0.85)는 정답과 IoU 0.82로 잘 겹치는데도 FP예요. 그 정답은 이미 0.95짜리가 가져갔기 때문이에요. <b>정답 하나에는 예측 하나만</b> 짝지어요. 세모 정답은 아무도 못 찾아서 FN이에요."
   ]
  },
  "s4": {
   "title": "④ nms(boxes, scores, iou_threshold=0.5)",
   "lines": [
    [
     "keep → [0, 2, 3]  (예측 4개 중 3개 남음)",
     1
    ],
    "점수 0.95, square → TP",
    "점수 0.8, circle → TP",
    "점수 0.4, triangle → FP",
    [
     "precision 0.50 → 0.67, recall 0.67 그대로",
     2
    ]
   ],
   "notes": [
    "점수가 가장 높은 박스를 남기고, 그것과 IoU 0.5 이상 겹치는 낮은 점수 박스를 지웠어요. 같은 물체에 대한 중복 후보예요.",
    "중복이 사라져 헛탐지가 줄었어요. 놓친 물체는 NMS로 찾아지지 않아요."
   ]
  },
  "s4img": {
   "title": "NMS 전후",
   "images": [
    [
     "img/detection/nms_before.png",
     "전: 후보 4개"
    ],
    [
     "img/detection/nms_after.png",
     "후: 3개"
    ]
   ],
   "notes": []
  },
  "s5": {
   "title": "⑤ 점수 기준을 바꾸면 (NMS 뒤)",
   "cols": [
    "점수 기준",
    "TP",
    "FP",
    "FN",
    "precision",
    "recall"
   ],
   "rows": [
    [
     0.9,
     1,
     0,
     2,
     "1.00",
     "0.33"
    ],
    [
     0.7,
     2,
     0,
     1,
     "1.00",
     "0.67"
    ],
    [
     0.3,
     2,
     1,
     1,
     "0.67",
     "0.67"
    ]
   ],
   "notes": [
    "기준을 높이면 확실한 것만 남아 precision이 올라가고, 낮추면 더 많이 찾지만 헛탐지도 늘어요. 기준을 여러 개로 바꿔 가며 그린 precision-recall 곡선의 아래 넓이가 <b>AP</b>이고, 종류별 AP의 평균이 <b>mAP</b>예요.",
    "mAP50은 IoU 0.5 기준, mAP50-95는 IoU 0.5~0.95를 평균해서 박스가 얼마나 정확한지까지 따져요."
   ]
  },
  "s6": {
   "title": "⑥ IoU 추적기 (물체 3개, 30프레임)",
   "cols": [
    "몇 프레임마다 탐지",
    "물체가 그사이 움직인 거리",
    "ID가 바뀐 횟수"
   ],
   "rows": [
    [
     1,
     "약 1픽셀",
     "0"
    ],
    [
     5,
     "약 5픽셀",
     "0"
    ],
    [
     10,
     "약 10픽셀",
     "0"
    ],
    [
     20,
     "약 20픽셀",
     "<b>1</b>"
    ]
   ],
   "notes": [
    "탐지기는 프레임마다 박스만 줘요. \"이 박스가 아까 그 물체인가\"는 직전 박스와 IoU가 가장 큰 것끼리 이어 붙여 정해요.",
    "물체가 프레임마다 약 1픽셀씩 움직여서, 10프레임 간격까지는 잘 이어졌어요. 20프레임 간격에서는 박스가 많이 어긋나 IoU가 기준(0.3)보다 작아져서, 같은 물체에 새 ID가 붙었어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "라벨",
   "비율 → 픽셀",
   "중심, 너비, 높이"
  ],
  [
   2,
   "IoU",
   "겹친 / 합친",
   "0.5 이상이면 찾음"
  ],
  [
   3,
   "짝짓기",
   "TP, FP, FN",
   "정답 하나에 하나"
  ],
  [
   4,
   "NMS",
   "중복 지우기",
   "점수 높은 것만"
  ],
  [
   5,
   "기준",
   "precision ↔ recall",
   "AP, mAP"
  ],
  [
   6,
   "추적",
   "IoU로 잇기",
   "ID 바뀜"
  ]
 ],
 "env": "Python 3.13.16, torchvision 0.29.0+cu130 (nms), OpenCV 4.13.0"
};
