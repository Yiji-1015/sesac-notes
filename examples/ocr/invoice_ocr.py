# 청구서 OCR과 구조화 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/doc01.png ~ doc05.png  합성 영어 청구서 (실제 사람, 거래 아님)
#         data/lowres_raw.png, rotated_raw.png, shadow_raw.png  doc03을 작게, 기울게, 그림자 지게 찍은 것
#         data/lesson.json  문서마다 정답 글(reference_text)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                      #@1
from pathlib import Path                                         #@1
import cv2                                                       #@1
import pytesseract                                               #@1
import numpy as np                                               #@3
import re                                                        #@5

DATA = Path(__file__).resolve().parent / "data"                  #@1
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  #@1  이 PC의 설치 위치
CONFIG = "--psm 6"  # 그림 전체를 글 한 덩어리로 보기                #@1

# ① 그대로 읽기
lesson = json.loads((DATA / "lesson.json").read_text(encoding="utf-8"))
truth = {k: v["reference_text"] for k, v in lesson["documents"].items()}
doc01 = cv2.imread(str(DATA / "doc01.png"))
text01 = pytesseract.image_to_string(doc01, lang="eng", config=CONFIG)

# ② 얼마나 틀렸나 재기 (CER)
def cer(pred, ref):
    pred, ref = " ".join(pred.split()), " ".join(ref.split())  # 줄바꿈, 공백 차이는 빼고
    d = list(range(len(ref) + 1))                    # 편집 거리: 몇 글자를 고치면 같아지나
    for i, p in enumerate(pred, 1):
        prev, d[0] = d[0], i
        for j, r in enumerate(ref, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (p != r))
    return d[len(ref)] / len(ref)

cer01 = cer(text01, truth["doc01"])

# ③ 나쁜 사진을 다듬고 읽기
def upscale(img):                                    # 작은 글자 → 3배로 키우기
    return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

def deskew(img):                                     # 기울어진 글줄 → 반듯하게
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = np.column_stack(np.where(gray < 128))[:, ::-1].astype(np.float32)  # 글자 픽셀 좌표 (x, y)
    angle = cv2.minAreaRect(ink)[2]
    angle = angle - 90 if angle > 45 else angle
    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))

def unshadow(img):                                   # 그림자 → 배경 밝기로 나눠 평평하게
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)
    return cv2.normalize(cv2.divide(gray, background, scale=255), None, 0, 255, cv2.NORM_MINMAX)

fixes = {"lowres": upscale, "rotated": deskew, "shadow": unshadow}
bad = {}
for name, fix in fixes.items():
    raw = cv2.imread(str(DATA / f"{name}_raw.png"))
    before = pytesseract.image_to_string(raw, lang="eng", config=CONFIG)
    after = pytesseract.image_to_string(fix(raw), lang="eng", config=CONFIG)
    bad[name] = {"before": before, "after": after,
                 "cer_before": cer(before, truth["doc03"]), "cer_after": cer(after, truth["doc03"])}

# ④ 단어마다 위치와 확신 받기
data = pytesseract.image_to_data(doc01, lang="eng", config=CONFIG, output_type=pytesseract.Output.DICT)
words = [{"text": t, "conf": round(float(c), 1), "box": (x, y, w, h)}
         for t, c, x, y, w, h in zip(data["text"], data["conf"], data["left"], data["top"], data["width"], data["height"])
         if t.strip()]
unsure = [w for w in words if w["conf"] < 90]

# ⑤ 필드 뽑기 (구조화)
def parse(text):
    fields = {"doc_id": re.search(r"INVOICE\s+(\S+)", text),
              "date": re.search(r"DATE:\s*(\d{4}-\d{2}-\d{2})", text),
              "supplier": re.search(r"SUPPLIER:\s*(.+)", text),
              "total": re.search(r"TOTAL:\s*(\d+)", text)}
    out = {k: (m.group(1).strip() if m else None) for k, m in fields.items()}
    out["items"] = [{"item": a, "qty": int(b), "price": int(c), "amount": int(d)}
                    for a, b, c, d in re.findall(r"^([A-Z]+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", text, re.M)]
    return out

parsed = {k: parse(pytesseract.image_to_string(cv2.imread(str(DATA / f"{k}.png")), lang="eng", config=CONFIG))
          for k in truth}

# ⑥ 규칙으로 검산하기
def check(doc):
    problems = []
    if not doc["doc_id"] or not re.fullmatch(r"DOC\d{2}", doc["doc_id"]):
        problems.append(f"doc_id 모양 이상: {doc['doc_id']}")
    for it in doc["items"]:
        if it["qty"] * it["price"] != it["amount"]:
            problems.append(f"{it['item']}: {it['qty']}×{it['price']} ≠ {it['amount']}")
    if doc["total"] is None or sum(it["amount"] for it in doc["items"]) != int(doc["total"]):
        problems.append(f"합계 불일치: 항목 합 {sum(it['amount'] for it in doc['items'])}, TOTAL {doc['total']}")
    return problems or ["통과"]

checks = {k: check(v) for k, v in parsed.items()}
