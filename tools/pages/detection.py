"""모델 > 이미지 > 객체 탐지 (박스, IoU, 짝짓기, NMS, Precision-Recall, 추적).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/detection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "detection"
IMG = ROOT / "topics" / "model" / "img" / "detection"
IMG.mkdir(parents=True, exist_ok=True)
code = S.parse((EX / "boxes.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="boxes.py")
D = EX / "data"


def draw(path, boxes, name, scale=2):
    im = cv2.imread(str(path))
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    for (x1, y1, x2, y2), color, text in boxes:
        cv2.rectangle(im, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale) - 1, int(y2 * scale) - 1), color, 2)
        if text:
            cv2.putText(im, text, (int(x1 * scale), max(10, int(y1 * scale) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.imwrite(str(IMG / name), im)
    return f"img/detection/{name}"


G, R, B = (0, 200, 0), (0, 0, 255), (255, 120, 0)
gt, pred = ns["gt"], ns["pred"]
gtb = [((g.x1, g.y1, g.x2, g.y2), G, "") for g in gt.itertuples()]
predb = [((p.x1, p.y1, p.x2, p.y2), R, f"{p.confidence}") for p in pred.itertuples()]
outs = {}
outs["why"] = S.images("scene.png: 정답(초록)과 예측 후보(빨강)", [(draw(D / "scene.png", gtb + predb, "scene_all.png"), "정답 3개, 예측 4개")],
                       ["분류는 그림 하나에 답 하나지만, 탐지는 그림 하나에 <b>여러 물체</b>의 <b>종류와 위치(박스)</b>를 답해요. 그래서 \"맞았다\"를 정하는 것부터 달라요.",
                        "예측 박스가 정답과 조금 어긋나 있고, 한 물체에 두 개가 겹쳐 있고, 엉뚱한 곳에 하나가 있어요. 이걸 어떻게 채점하고 정리할지가 이 쪽의 내용이에요."])
lab = ns["label_rows"]
outs["s1"] = S.box("① basic_train_000.txt → 픽셀 박스 (그림 160×128)", [*(" ".join(r) for r in lab), "", *(f"{n} → {b}" for n, b in ns["pixel_boxes"])],
                   ["YOLO 라벨은 줄마다 \"종류 번호, 박스 중심 x, 중심 y, 너비, 높이\"이고, 위치는 그림 크기에 대한 <b>비율(0~1)</b>이에요. 그림 크기를 바꿔도 라벨을 고칠 필요가 없어요.", "그리거나 IoU를 잴 때는 왼쪽 위(x1, y1), 오른쪽 아래(x2, y2) 픽셀로 바꿔요. 가로 세로를 바꿔 곱하면 박스가 엉뚱한 곳에 그려져요."])
outs["s1img"] = S.images("라벨을 그림 위에", [(draw(D / "images" / "train" / "basic_train_000.png", [(b, G, n) for n, b in ns["pixel_boxes"]], "label.png"), "basic_train_000.png")])
outs["s2"] = S.table("② 예측 4개 × 정답 3개의 IoU", ["예측 (점수, 종류)"] + [f"정답 {ns['NAMES'][g.class_id]}" for g in gt.itertuples()],
                     [[f"{p.confidence}, {ns['NAMES'][p.class_id]}"] + [f"<b>{v}</b>" if v >= 0.5 else str(v) for v in row] for p, row in zip(pred.itertuples(), ns["ious"])],
                     ["IoU = 겹친 넓이 ÷ 합친 넓이. 1이면 완전히 같고 0이면 안 겹쳐요. 보통 0.5 이상이면 \"같은 물체를 찾았다\"고 봐요.", "점수 0.4짜리 세모 예측은 어느 정답과도 겹치지 않아요."])
c = ns["counts"]
_conf = [p.confidence for p in pred.itertuples()]
dup_iou = max(ns["ious"][_conf.index(0.85)])   # 두 번째 네모 예측이 정답과 겹친 정도
S.claim(max(ns["ious"][_conf.index(0.4)]) == 0, "② 0.4짜리 세모 예측은 어느 정답과도 안 겹친다")
S.claim(any(s_ == 0.85 and "FP" in str(r_) for s_, _, r_ in ns["matched"]) and c["FN"] == 1, "③ 0.85짜리 네모는 잘 겹치는데도 FP, 세모 정답은 FN")
outs["s3"] = S.box("③ match(pred, gt)  (IoU 0.5, 같은 종류, 점수 높은 것부터)", [*(f"점수 {s}, {n} → {r}" for s, n, r in ns["matched"]), "", (f"TP {c['TP']}, FP {c['FP']}, FN {c['FN']} → precision {c['precision']:.2f}, recall {c['recall']:.2f}", 1)],
                   [f"두 번째 네모(0.85)는 정답과 IoU {dup_iou:.2f}로 잘 겹치는데도 FP예요. 그 정답은 이미 0.95짜리가 가져갔기 때문이에요. <b>정답 하나에는 예측 하나만</b> 짝지어요. 세모 정답은 아무도 못 찾아서 FN이에요."])
cn = ns["counts_nms"]
S.claim(cn["recall"] == c["recall"] and cn["precision"] > c["precision"], "④ NMS로 precision은 오르고 recall은 그대로")
outs["s4"] = S.box("④ nms(boxes, scores, iou_threshold=0.5)", [(f"keep → {ns['keep']}  (예측 {len(pred)}개 중 {len(ns['keep'])}개 남음)", 1), *(f"점수 {s}, {n} → {r}" for s, n, r in ns["matched_nms"]),
                                                                  (f"precision {c['precision']:.2f} → {cn['precision']:.2f}, recall {cn['recall']:.2f} 그대로", 2)],
                   ["점수가 가장 높은 박스를 남기고, 그것과 IoU 0.5 이상 겹치는 낮은 점수 박스를 지웠어요. 같은 물체에 대한 중복 후보예요.", "중복이 사라져 헛탐지가 줄었어요. 놓친 물체는 NMS로 찾아지지 않아요."])
outs["s4img"] = S.images("NMS 전후", [(draw(D / "scene.png", gtb + predb, "nms_before.png"), f"전: 후보 {len(pred)}개"), (draw(D / "scene.png", gtb + [((p.x1, p.y1, p.x2, p.y2), R, f"{p.confidence}") for p in ns["after_nms"].itertuples()], "nms_after.png"), f"후: {len(ns['after_nms'])}개")])
outs["s5"] = S.table("⑤ 점수 기준을 바꾸면 (NMS 뒤)", ["점수 기준", "TP", "FP", "FN", "precision", "recall"], [[r["기준"], r["TP"], r["FP"], r["FN"], f"{r['precision']:.2f}", f"{r['recall']:.2f}"] for r in ns["pr_table"]],
                     ["기준을 높이면 확실한 것만 남아 precision이 올라가고, 낮추면 더 많이 찾지만 헛탐지도 늘어요. 기준을 여러 개로 바꿔 가며 그린 precision-recall 곡선의 아래 넓이가 <b>AP</b>이고, 종류별 AP의 평균이 <b>mAP</b>예요.",
                      "mAP50은 IoU 0.5 기준, mAP50-95는 IoU 0.5~0.95를 평균해서 박스가 얼마나 정확한지까지 따져요."])
sw = ns["switches"]
S.claim(all(v == 0 for g_, v in sw.items() if g_ <= 10) and sw[20] > 0, "⑥ 10프레임 간격까지는 잘 이어지고, 20프레임 간격에서 ID가 바뀐다")
outs["s6"] = S.table("⑥ IoU 추적기 (물체 3개, 30프레임)", ["몇 프레임마다 탐지", "물체가 그사이 움직인 거리", "ID가 바뀐 횟수"], [[g, f"약 {g}픽셀", f"<b>{v}</b>" if v else "0"] for g, v in sw.items()],
                     ["탐지기는 프레임마다 박스만 줘요. \"이 박스가 아까 그 물체인가\"는 직전 박스와 IoU가 가장 큰 것끼리 이어 붙여 정해요.", "물체가 프레임마다 약 1픽셀씩 움직여서, 10프레임 간격까지는 잘 이어졌어요. 20프레임 간격에서는 박스가 많이 어긋나 IoU가 기준(0.3)보다 작아져서, 같은 물체에 새 ID가 붙었어요."])
flow = [[1, "라벨", "비율 → 픽셀", "중심, 너비, 높이"], [2, "IoU", "겹친 / 합친", "0.5 이상이면 찾음"], [3, "짝짓기", "TP, FP, FN", "정답 하나에 하나"], [4, "NMS", "중복 지우기", "점수 높은 것만"], [5, "기준", "precision ↔ recall", "AP, mAP"], [6, "추적", "IoU로 잇기", "ID 바뀜"]]
S.save(ROOT / "topics" / "model" / "object-detection-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                              "env": f"Python {sys.version.split()[0]}, torchvision {torchvision.__version__} (nms), OpenCV {cv2.__version__}"})
why = f"""<p>사진 속에 무엇이 <b>어디에</b> 있는지 찾는 일을 <b>객체 탐지</b>라고 해요. 결과는 물체마다 종류, 점수(확신), 박스예요.</p>{K.out("why")}
<p>YOLO 같은 탐지 모델은 그림 한 장에서 수많은 후보 박스를 한 번에 내요. 그 뒤에는 언제나 같은 일이 따라와요. 박스를 좌표로 다루고, 정답과 얼마나 겹치는지 재고, 중복을 지우고, 점수 기준을 정하고, 영상이면 프레임끼리 이어요. 이 쪽에서는 모델 없이 이 과정을 직접 계산해요.</p>"""
whole = """<p>라벨 읽기 → 겹친 정도 재기 → 정답과 짝짓기 → 중복 지우기 → 점수 기준 정하기 → 프레임끼리 잇기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "YOLO 라벨을 픽셀 박스로", "비율로 적힌 중심, 너비, 높이를 픽셀 좌표 (x1, y1, x2, y2)로 바꿔요.",
           K.code_table([("(cx - w / 2) * img_w", "왼쪽 끝 x. 중심에서 너비 절반을 빼고 그림 너비를 곱해요."), ("label_file.read_text().splitlines()", "물체 하나가 한 줄. 빈 파일이면 물체 0개라는 뜻이에요.")]) + K.out("s1img"), "s1"),
    K.step(2, "겹친 정도 재기 (IoU)", "두 박스의 겹친 넓이를 합친 넓이로 나눠요.",
           K.code_table([("min(a[2], b[2]) - max(a[0], b[0])", "겹친 부분의 가로. 음수면 안 겹친 것이라 0으로"), ("inter / union", "합친 넓이 = 두 넓이의 합 − 겹친 넓이")]), "s2"),
    K.step(3, "예측과 정답 짝짓기", "점수 높은 예측부터, 같은 종류이면서 IoU 0.5 이상인 정답 중 가장 많이 겹치는 것과 짝지어요.",
           K.table(["결과", "뜻"], [["TP", "정답과 짝지어진 예측 (맞게 찾음)"], ["FP", "짝이 없는 예측 (헛탐지, 중복)"], ["FN", "짝을 못 찾은 정답 (놓침)"]])
           + K.code_table([("sort_values(\"confidence\", ascending=False)", "확신 높은 예측이 먼저 정답을 가져가요."), ("used", "이미 짝지어진 정답 번호")]), "s3"),
    K.step(4, "겹친 후보 정리하기 (NMS)", "같은 물체에 겹쳐 나온 후보 중 점수가 가장 높은 것만 남겨요.",
           K.code_table([("nms(boxes, scores, iou_threshold=0.5)", "남길 박스 번호를 점수 순으로 (torchvision)"), ("pred.iloc[keep]", "남은 예측만")]) + K.out("s4img"), "s4"),
    K.step(5, "확신 기준을 바꾸며 보기", "점수 기준을 바꿔 가며 precision과 recall이 어떻게 변하는지 봐요.", K.code_table([("after_nms[after_nms.confidence >= t]", "기준 t 이상인 예측만")]), "s5"),
    K.step(6, "영상에서 같은 물체 이어 붙이기 (추적)", "프레임마다 나온 박스를 직전 프레임의 박스와 IoU로 이어서 같은 ID를 붙여요.",
           K.code_table([("frames.sample(frac=1)", "탐지 결과처럼 순서를 섞어서, 순서에 기대지 않게"), ("max(cand)", "직전 박스 중 IoU가 가장 큰 것"), ("if v < thr: 새 ID", "이어 붙일 곳이 없으면 새 물체로"), ("track_id.nunique() - 1", "한 물체(정답 ID)에 붙은 ID 종류 − 1 = ID가 바뀐 횟수")]), "s6"),
]
practice = [("YOLO로 직접 학습한다", "Ultralytics의 <code>YOLO(\"yolov8n.pt\").train(data=\"dataset.yaml\", epochs=…)</code>로 학습하고 <code>model.predict()</code>, <code>model.val()</code>로 박스와 mAP를 받아요. 미리 학습된 가중치를 처음 한 번 내려받아요. 자세한 명령은 <a href=\"../../notes/day15.html\">옛 Day15 노트</a>를 보세요."),
            ("NMS는 종류별로", "다른 종류의 물체끼리는 겹쳐도 지우면 안 돼서, 실제로는 종류별로 NMS를 해요 (<code>torchvision.ops.batched_nms</code>)."),
            ("작은 물체", "몇 픽셀만 어긋나도 IoU가 크게 떨어져요. mAP50-95가 낮으면 작은 물체의 박스 정확도를 의심해요."),
            ("추적이 끊긴다", "빠른 물체, 가려짐, 낮은 프레임률에서 끊겨요. 움직임을 예측하는 추적기(ByteTrack, BoT-SORT 등)를 쓰면 더 잘 이어져요.")]
libs = [("①", "<code>Path.read_text()</code>, <code>str.split()</code>", "라벨 읽기"), ("②③⑤", "<code>pandas.read_csv</code>, <code>sort_values</code>, 직접 짠 <code>iou</code>, <code>match</code>", "채점"), ("④", "<code>torchvision.ops.nms</code>", "중복 지우기"), ("⑥", "<code>DataFrame.sample</code>, <code>groupby</code>, <code>nunique</code>", "추적 점검")]
K.build(ROOT / "topics" / "model" / "object-detection.html", title="객체 탐지", h1="객체 탐지: 박스를 채점하고 정리하고 잇기", track="모델", crumb="모델 &gt; 이미지",
        one_line="객체 탐지는 물체마다 <b>종류, 점수, 박스</b>를 답해요. 박스는 <b>IoU</b>로 채점하고, 겹친 후보는 <b>NMS</b>로 지우고, 점수 기준으로 precision과 recall을 조절하고, 영상에서는 프레임끼리 박스를 <b>이어서</b> 추적해요.",
        why=why, whole=whole, steps=steps, code_file="examples/detection/boxes.py", practice=practice, libs=libs, libs_note="<code>yolo_to_xyxy</code>, <code>iou</code>, <code>match</code>, <code>iou_tracker</code>, <code>id_switches</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"transfer-learning.html\">전이학습과 오류 분석</a>", "그림 분류와 종류별 평가"), ("<a href=\"ocr-documents.html\">OCR과 문서 구조화</a>", "글자 박스를 읽어 값으로")],
        sources=["다루는 범위: Day15 (2026-11-06) 객체 탐지, IoU, mAP, NMS, YOLO 학습과 추론, 추적", "데이터: <code>examples/detection/data/</code> (합성 도형 그림과 라벨. 예측 박스는 원리 연습용으로 손으로 정한 값이고 YOLO 예측이 아니에요)",
                 "결과 값: <code>tools/pages/detection.py</code>. YOLO 모델 학습은 이 쪽에서 실행하지 않았어요", "공식 문서: <a href=\"https://docs.ultralytics.com/datasets/detect/\" target=\"_blank\">Ultralytics 탐지 데이터 형식</a>, <a href=\"https://docs.ultralytics.com/guides/yolo-performance-metrics/\" target=\"_blank\">YOLO 성능 지표</a>, <a href=\"https://docs.pytorch.org/vision/stable/generated/torchvision.ops.nms.html\" target=\"_blank\">torchvision.ops.nms</a>"])
print("ok")
