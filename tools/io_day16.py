"""Day16 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

데이터: 수업 자료 C:/practice_assignment/20261109/02_intermediate/data (합성 영어 청구서 doc01~05,
perspective.png · shadow.png · lowres.png, lesson.json)를 임시 폴더에 복사해서 읽기만 한다.
이 PC에는 Tesseract 프로그램 · jiwer · transformers · onnx · TensorRT(trtexec)가 없다.
 - image_to_data: 실제 호출은 TesseractNotFoundError → lesson.json 에 '기록된 실제 Tesseract 5.5.0 결과'
   (단어·conf·bbox)를 DataFrame으로 만들어 강의 코드의 거르기만 실제로 실행한다.
 - jiwer.cer: 같은 정의(편집 거리 ÷ 정답 글자 수)인 강의의 cer() 함수로 대신 계산한다.
 - pipeline / AutoProcessor / torch.onnx.export / trtexec: 실행할 수 없어서 상자를 넣지 않는다.
결과는 cmds/day16.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day16.py
"""
import html
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261109/02_intermediate/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")
DATA = TMP / "data"
OUT = TMP / "outputs"
OUT.mkdir()
L = json.loads((DATA / "lesson.json").read_text(encoding="utf-8"))
DOC3 = L["documents"]["doc03"]
OCR3 = DOC3["ocr"]["text"]


def rd(name, flag=cv2.IMREAD_COLOR):
    return cv2.imread(str(DATA / name), flag)


def pct(mask):
    return f"{mask.mean() * 100:.1f}%"


def err(fn):
    try:
        return repr(fn())
    except Exception as e:  # noqa: BLE001 — 실제로 난 예외를 그대로 보여 준다
        msg = str(e)
        m = re.search(r"\(-\d+:[^)]*\)[^\n]*", msg)          # OpenCV 에러는 긴 경로 대신 핵심 줄만
        return f"{type(e).__name__}: {(m.group(0) if m else msg.splitlines()[0])[:110]}"


IO = {}

# 1. imread / cvtColor
image = rd("doc03.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
IO["d16-cv2-imread"] = ("수업 청구서 data/doc03.png", [
    ('image = cv2.imread("doc03.png", cv2.IMREAD_COLOR)', f"image.shape → {image.shape}", "(H, W, 3) — 세로 700 · 가로 1000 · BGR"),
    ("cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape", repr(gray.shape), "채널 축이 사라짐"),
    ('cv2.imread("doc03.png", cv2.IMREAD_GRAYSCALE).shape', repr(rd("doc03.png", cv2.IMREAD_GRAYSCALE).shape), "처음부터 1채널로 읽기"),
    ('cv2.imread("없는파일.png")', repr(cv2.imread(str(DATA / "nofile.png"))), "에러가 아니라 None → 바로 is None 검사"),
    ('cv2.imwrite("outputs/gray.png", gray)', repr(cv2.imwrite(str(OUT / "gray.png"), gray)), "성공하면 True"),
    ('cv2.imwrite("없는폴더/gray.png", gray)', repr(cv2.imwrite(str(OUT / "nodir" / "gray.png"), gray)), "실패해도 예외 없이 False"),
])

# 2. threshold / adaptiveThreshold — 왼쪽이 그림자로 어두운 shadow.png
sh = cv2.cvtColor(rd("shadow.png"), cv2.COLOR_BGR2GRAY)
otsu_t, binary = cv2.threshold(sh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(sh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
_, no_otsu = cv2.threshold(sh, 0, 255, cv2.THRESH_BINARY)
left, right = slice(0, 300), slice(700, 1000)
IO["d16-cv2-threshold"] = ("data/shadow.png (왼쪽으로 갈수록 그림자로 어두운 청구서) — 검은색(0)이 된 칸 비율", [
    ("gray 평균 밝기: 왼쪽 300px / 오른쪽 300px", f"{sh[:, left].mean():.0f} / {sh[:, right].mean():.0f}", "같은 흰 종이인데 왼쪽이 훨씬 어두움"),
    ("otsu_t  (THRESH_BINARY | THRESH_OTSU)", f"{otsu_t:g}", "0을 넣었지만 Otsu가 고른 기준이 돌아옴"),
    ("binary(Otsu) 검은 칸: 왼쪽 / 오른쪽", f"{pct(binary[:, left] == 0)} / {pct(binary[:, right] == 0)}",
     "기준 하나라 그림자 쪽은 종이까지 통째로 검게 → 글자가 묻힘"),
    ("adaptive(31, 11) 검은 칸: 왼쪽 / 오른쪽", f"{pct(adaptive[:, left] == 0)} / {pct(adaptive[:, right] == 0)}",
     "칸마다 주변 평균을 기준으로 → 그림자 속에서도 글자만 검게"),
    ("OTSU 없이 threshold(gray, 0, 255, BINARY) 검은 칸", pct(no_otsu == 0), "0이 진짜 기준이 돼서 전부 흰색"),
])

# 3. getPerspectiveTransform / warpPerspective — lesson.json 의 모서리
photo = rd("perspective.png")
corners = np.float32(L["perspective"]["corners"])
dst = np.float32([[0, 0], [999, 0], [999, 699], [0, 699]])
M = cv2.getPerspectiveTransform(corners, dst)
warped = cv2.warpPerspective(photo, M, (1000, 700), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def diff(a):
    return f"{np.abs(a.astype(int) - image.astype(int)).mean():.2f}"


rolled = cv2.warpPerspective(photo, cv2.getPerspectiveTransform(np.roll(corners, 1, axis=0), dst), (1000, 700),
                             borderValue=(255, 255, 255))
swap = corners[[0, 3, 2, 1]]
swapped = cv2.warpPerspective(photo, cv2.getPerspectiveTransform(swap, dst), (1000, 700), borderValue=(255, 255, 255))
squashed = cv2.warpPerspective(photo, M, (700, 1000), borderValue=(255, 255, 255))
IO["d16-cv2-getperspectivetransform"] = (
    f"data/perspective.png {photo.shape} · lesson.json 모서리 {L['perspective']['corners']} → 정면 doc03.png와 비교(평균 픽셀 차, 작을수록 같음)", [
        ("M = cv2.getPerspectiveTransform(src, dst)", repr(np.round(M, 3).tolist()), "3×3 행렬 — 마지막 칸은 1로 고정"),
        ("warpPerspective(photo, M, (1000, 700)).shape", repr(warped.shape), "dsize는 (W, H) → shape는 (700, 1000, 3)"),
        ("보정본 vs 정면 원본 doc03.png 평균 차", diff(warped), "거의 같음 — 펴기 성공"),
        ("모서리 순서를 한 칸 밀었을 때 평균 차", diff(rolled), "문서가 90° 돌아가며 찌그러짐"),
        ("2번 ↔ 4번 모서리를 맞바꿨을 때 평균 차", diff(swapped), "대각선으로 뒤집힌(가로·세로가 바뀐) 문서"),
        ("dsize를 (700, 1000)으로 바꿔 넣으면 .shape", repr(squashed.shape), "세로로 긴 캔버스 — dst(999, 699)와 안 맞음"),
        ("그 캔버스의 아래 300줄 중 흰 칸 비율", pct(squashed[700:] == 255),
         "문서는 위 700줄에만(오른쪽 300px는 잘림), 아래엔 원본 사진의 문서 밖 배경이 딸려 옴"),
    ])

# 4. perspectiveTransform / resize
tok = next(t for t in DOC3["ocr"]["tokens"] if t["text"] == "TOTAL:")
pt = np.float32([[[tok["bbox"][0], tok["bbox"][1]]]])
back = cv2.perspectiveTransform(pt, np.linalg.inv(M))
again = cv2.perspectiveTransform(back, M)
wrong = cv2.perspectiveTransform(pt, M)
small = rd("lowres.png")
big = cv2.resize(small, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# 명령어 사전 코드의 src/dst 그대로 계산해 보기
M_doc = cv2.getPerspectiveTransform(np.float32([[92, 40], [820, 75], [860, 1080], [55, 1050]]),
                                    np.float32([[0, 0], [699, 0], [699, 999], [0, 999]]))
back_doc = cv2.perspectiveTransform(np.float32([[[40, 370]]]), np.linalg.inv(M_doc))
IO["d16-cv2-perspectivetransform"] = (
    f"getPerspectiveTransform 항목의 M(perspective.png) · doc03 OCR 기록에서 TOTAL: 단어의 왼쪽 위 점 {tuple(tok['bbox'][:2])} · data/lowres.png", [
        (f"perspectiveTransform([[[{tok['bbox'][0]}, {tok['bbox'][1]}]]], inv(M))", repr(np.round(back.astype(float), 1).tolist()), "보정본 점 → 비스듬한 원본 사진의 점"),
        ("perspectiveTransform(back, M)", repr(np.round(again.astype(float), 1).tolist()), "다시 M으로 보내면 원래 점 — 역행렬이 맞다는 확인"),
        ("perspectiveTransform(점, M)  (inv 없이)", repr(np.round(wrong.astype(float), 1).tolist()), "한 번 더 '펴는' 셈이라 엉뚱한 곳"),
        ("perspectiveTransform(np.float32([[56, 501]]), inv(M))  — shape (1, 2)",
         err(lambda: cv2.perspectiveTransform(np.float32([[56, 501]]), np.linalg.inv(M)).tolist()),
         "(N, 1, 2)가 아니면 OpenCV 에러"),
        ("이 사전 코드의 src·dst로 [[[40, 370]]] 되돌리기", repr(np.round(back_doc.astype(float), 1).tolist()),
         "코드 주석의 [[[64.7 349.6]]]과 달라요 — 주석 값은 이 src 숫자로는 재현되지 않아요"),
        ("cv2.resize(lowres, None, fx=2, fy=2, interpolation=INTER_CUBIC).shape", f"{small.shape} → {big.shape}", "가로·세로 2배"),
    ])

# 5. image_to_data — 실제 호출은 실패, 기록된 실제 Tesseract 결과로 거르기만 실행
import pytesseract  # noqa: E402
from pytesseract import Output  # noqa: E402

call = err(lambda: pytesseract.image_to_data(gray, lang="eng", config="--oem 3 --psm 6", output_type=Output.DATAFRAME))
v = L["variants"]["lowres_raw"]
ocr = pd.DataFrame([{"level": 5, "text": t["text"], "conf": round(t["conf"], 1), "left": t["bbox"][0], "top": t["bbox"][1],
                     "width": t["bbox"][2] - t["bbox"][0], "height": t["bbox"][3] - t["bbox"][1]} for t in v["ocr"]["tokens"]])
words = ocr[ocr.level.eq(5) & ocr.text.str.strip().ne("")].copy()
review = words[words.conf.lt(50)]
first = words.iloc[0]
d3 = pd.DataFrame([{"text": t["text"], "conf": t["conf"]} for t in DOC3["ocr"]["tokens"]])
IO["d16-pytesseract-image-to-data"] = (
    "이 PC엔 Tesseract가 없어서, 수업 lesson.json에 기록된 실제 Tesseract 5.5.0(eng, psm 6) 결과로 표를 만들고 거르기만 실행 — lowres_raw.png(작은 사진) 기준", [
        ("pytesseract.image_to_data(gray, ...)  (이 PC)", call, "엔진 프로그램이 따로 설치돼 있어야 해요"),
        ("len(words)", repr(len(words)), "단어(level 5) 행 수"),
        ("words.iloc[0][['text', 'conf', 'left', 'top', 'width', 'height']]",
         repr([first.text, float(first.conf), int(first.left), int(first.top), int(first.width), int(first.height)]),
         "맞게 읽은 INVOICE인데 conf가 낮음 → 점수 ≠ 정답 여부"),
        ("review = words[words.conf.lt(50)] → text 목록", repr(review.text.tolist()), f"{len(review)}개가 검토 목록으로"),
        ("[left, top, left+width, top+height]  (첫 단어)",
         repr([int(first.left), int(first.top), int(first.left + first.width), int(first.top + first.height)]), "xyxy로 바꾸기"),
        ("정면 깨끗한 doc03.png 기록: conf 최소값 · 50 미만 개수",
         f"{d3.conf.min():.1f} · {int(d3.conf.lt(50).sum())}", "같은 엔진이라도 사진 품질에 따라 점수 분포가 달라짐"),
    ])


# 6. CER — 강의 함수 그대로
def levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = old
    return dp[-1]


def cer(reference: str, prediction: str) -> float:
    if not reference:
        raise ValueError("빈 정답 문서는 별도 평가합니다")
    errors = levenshtein(reference, prediction)
    return errors / len(reference)


ref3 = DOC3["reference_text"]
var = L["variants"]
IO["d16-cer-계산-함수"] = ("강의 예시 문자열 + lesson.json의 정답 글자 vs 기록된 실제 Tesseract 결과 (doc03 계열)", [
    ('levenshtein("1,234,500원", "1,234,5O0원")', repr(levenshtein("1,234,500원", "1,234,5O0원")), "0 → O 치환(S) 1번"),
    ('cer("1,234,500원", "1,234,5O0원")', f"{cer('1,234,500원', '1,234,5O0원'):.3f}", "1 ÷ 정답 10글자"),
    ('levenshtein("33000", "3300")  ·  ("33000", "330000")', f"{levenshtein('33000', '3300')} · {levenshtein('33000', '330000')}", "빠뜨림(D) 1 · 끼워 넣음(I) 1"),
    ("cer(정답, doc03 OCR)  — 정면 깨끗한 문서", f"{cer(ref3, OCR3):.3f}", "DOC03 → DOCO3 한 글자만 틀림"),
    ("cer(정답, rotated_raw OCR)", f"{cer(var['rotated_raw']['reference_text'], var['rotated_raw']['ocr']['text']):.3f}", "기울어진 사진 그대로"),
    ("cer(정답, lowres_raw / lowres_corrected OCR)",
     f"{cer(var['lowres_raw']['reference_text'], var['lowres_raw']['ocr']['text']):.3f} / "
     f"{cer(var['lowres_corrected']['reference_text'], var['lowres_corrected']['ocr']['text']):.3f}", "작은 사진 → 보정하면 줄어듦"),
    ("cer(정답, shadow_raw OCR)  — OCR 결과가 빈 문자열", f"{cer(var['shadow_raw']['reference_text'], var['shadow_raw']['ocr']['text']):.3f}",
     "전부 빠뜨림 = 1.0"),
    ('cer("", "abc")', err(lambda: cer("", "abc")), "빈 정답은 0으로 나누지 않고 따로 집계"),
])

# 7. jiwer.cer — 미설치, 같은 정의로 대신
psm3 = next(r for r in L["psm_records"] if r["psm"] == 3)
truth = " ".join(psm3["reference"].split("\n"))
read = " ".join(psm3["text"].split())
lr = var["lowres_raw"]
t_lr, r_lr = " ".join(lr["reference_text"].split("\n")), " ".join(lr["ocr"]["text"].split())
IO["d16-jiwer-cer"] = ("jiwer가 이 PC에 없어 같은 정의(편집 거리 ÷ 정답 글자 수)인 CER 계산 함수의 cer()로 계산 — doc01 psm 3 기록 · lowres_raw 기록", [
    ("import jiwer", err(lambda: __import__("jiwer")), "pip install jiwer 필요"),
    ("cer(정답 원문, psm 3 OCR 원문)  — 정리 없이", f"{cer(psm3['reference'], psm3['text']):.4f}", "OCR이 넣은 빈 줄(\\n) 하나도 오류로 셈"),
    ('cer(" ".join(lines), " ".join(text.split()))', f"{cer(truth, read):.4f}", "양쪽을 같은 규칙으로 정리 → 진짜 글자 오류만 남음"),
    ("cer(truth, read)  — lowres_raw", f"{cer(t_lr, r_lr):.4f}", "(정답, 예측) 순서"),
    ("cer(read, truth)  — 순서를 바꾸면", f"{cer(r_lr, t_lr):.4f}", "분모가 OCR 글자 수로 바뀌어 값이 달라짐"),
])

# 8. re.search — 사전 코드 그대로 + 실제 OCR 문자열
text = "발행일 2026-11-09   합계 1,234,500원"
date_pat = r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})"
money_pat = (r"합계\s*" r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*원")


def parse(t):
    dm, am = re.search(date_pat, t), re.search(money_pat, t)
    issued_at, total = None, None
    if dm:
        try:
            issued_at = date(*map(int, dm.groups())).isoformat()
        except ValueError:
            pass
    if am:
        total = int(am.group(1).replace(",", ""))
    return issued_at, total


dm = re.search(date_pat, text)
amounts = re.findall(r"^[A-Z]+ \d+ \d+ (\d+)$", OCR3, re.MULTILINE)
IO["d16-re-search"] = ("사전 코드의 text 문자열 + 수업 doc03의 실제 Tesseract 기록 문자열", [
    ("re.search(date_pat, text).groups()", repr(dm.groups()), "괄호 세 개 = 그룹 세 개 (아직 글자)"),
    ("print(issued_at, total)", " ".join(map(str, parse(text))), "날짜 검사 통과 · 쉼표 뗀 정수"),
    ('같은 코드에 "발행일 2026-13-45 합계 5000원"', repr(parse("발행일 2026-13-45 합계 5000원")), "모양은 날짜라 패턴엔 걸리지만 date()가 거부 → None"),
    ('re.search(money_pat, "합계: 미정")', repr(re.search(money_pat, "합계: 미정")), "못 찾으면 None → .group() 전에 검사"),
    ('re.sub(r"[^0-9]", "", "1,234,500원")', repr(re.sub(r"[^0-9]", "", "1,234,500원")), "숫자 아닌 글자를 지운 새 문자열"),
    (r're.search(r"^TOTAL: (\d+)$", ocr_text, re.MULTILINE).group(1)', repr(re.search(r"^TOTAL: (\d+)$", OCR3, re.MULTILINE).group(1)), "라벨 + 줄 단위 → 정확"),
    (r're.search(r"^TOTAL: (\d+)$", ocr_text)  (MULTILINE 없이)', repr(re.search(r"^TOTAL: (\d+)$", OCR3)), "^ $가 글 전체의 처음·끝이라 못 찾음"),
    (r're.search(r"(\d+)", ocr_text).group(1)', repr(re.search(r"(\d+)", OCR3).group(1)), "첫 숫자 = 맨 윗줄 DOCO3의 3 — 라벨이 없으면 엉뚱한 값"),
    (r're.findall(r"^[A-Z]+ \d+ \d+ (\d+)$", ocr_text, re.MULTILINE)', repr(amounts), f"품목 합 {sum(map(int, amounts))} = 합계 → total_mismatch 없음"),
])

# 9. json.dump
record = {
    "document_id": "invoice_001", "page": 1, "image_size": [1000, 1200], "coordinate_space": "processed_px_xyxy",
    "fields": {
        "issued_at": {"value": "2026-11-09", "raw": "2026. 11. 9", "confidence": 92.4, "bbox": [118, 205, 310, 244]},
        "supplier": {"value": "AI Lab", "raw": "AI Lab", "confidence": 93.1, "bbox": [118, 260, 310, 299]},
        "total_amount": {"value": 1234500, "raw": "1,234,500원", "confidence": 88.6, "bbox": [530, 820, 790, 865]},
    },
}
fp = OUT / "invoice_001.json"
with open(fp, "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
lines = fp.read_text(encoding="utf-8").split("\n")
raw_line = next(x.strip() for x in lines if "1,234,500" in x)
IO["d16-json-dump"] = ("사전 코드의 record를 그대로 저장", [
    ('json.dump(record, f, ensure_ascii=False, indent=2) 후 파일', f"{len(lines)}줄 · {fp.stat().st_size} bytes", "들여쓰기로 한 key가 한 줄씩"),
    ("파일 안 total_amount 의 raw 줄", raw_line, "한글 그대로"),
    ('json.dumps(record["fields"]["total_amount"]["raw"])  (ensure_ascii 기본 True)', json.dumps(record["fields"]["total_amount"]["raw"]),
     "원 → \\uc6d0 — 값은 같지만 사람이 못 읽음"),
    ("json.dumps(record) 길이: indent 없음 / indent=2", f"{len(json.dumps(record, ensure_ascii=False))} / {len(json.dumps(record, ensure_ascii=False, indent=2))}", "한 줄로 붙으면 작지만 읽기 힘듦"),
    ("json.load(open(...)) == record", repr(json.loads(fp.read_text(encoding="utf-8")) == record), "다시 읽으면 원래 dict와 같음"),
])


# 10. validate_invoice — 사전 코드 그대로
def validate_invoice(doc: dict) -> list[str]:
    if not isinstance(doc, dict):
        return ["입력은 객체여야 합니다"]
    errors = []
    if not doc.get("document_id"):
        errors.append("document_id 누락")
    fields = doc.get("fields", {})
    if not isinstance(fields, dict):
        return errors + ["fields는 객체여야 합니다"]
    for name in ["issued_at", "supplier", "total_amount"]:
        item = fields.get(name)
        value = item.get("value") if isinstance(item, dict) else None
        if value is None or value == "":
            errors.append(f"필수 필드 누락: {name}")
        elif name == "total_amount":
            if type(value) is not int or value < 0:
                errors.append("합계는 0 이상의 정수여야 합니다")
    return errors


def variant(**kw):
    d = json.loads(json.dumps(record))
    for k, v in kw.items():
        if k == "drop":
            del d["fields"][v]
        else:
            d["fields"]["total_amount"]["value"] = v
    return d


IO["d16-validate-invoice"] = ("json.dump 항목의 record와 일부러 망가뜨린 사본들", [
    ("validate_invoice(record)", repr(validate_invoice(record)), "빈 목록 = 통과"),
    ('total_amount value = 0', repr(validate_invoice(variant(total=0))), "0원은 진짜 값 — 누락이 아님"),
    ('total_amount value = "1234500"  (글자)', repr(validate_invoice(variant(total="1234500"))), "정수가 아님"),
    ("total_amount value = True", repr(validate_invoice(variant(total=True))), "type() is int 검사라 bool도 걸러짐"),
    ('supplier 필드 삭제', repr(validate_invoice(variant(drop="supplier"))), "필수 필드 검사"),
    ('{"fields": "x"}', repr(validate_invoice({"fields": "x"})), "문제를 모아서 한꺼번에"),
    ("validate_invoice([record])", repr(validate_invoice([record])), "dict가 아니면 바로 거절 (.get 에러 없이)"),
])


# 13. json.loads 최소 검사 — 사전 코드 그대로를 함수로
def check(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    missing = {"answer", "evidence", "page", "bbox"} - data.keys()
    box = data.get("bbox")
    valid_box = isinstance(box, list) and len(box) == 4
    if valid_box:
        valid_box = all(type(v) in (int, float) for v in box)
    if valid_box:
        x1, y1, x2, y2 = box
        valid_box = 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1200
    return sorted(missing), valid_box


good = '{"answer":"1,234,500원","evidence":"1,234,500원","page":1,"bbox":[530,820,790,865]}'
IO["d16-json-loads"] = ("사전 코드의 raw와 망가진 응답들 → (누락 key, Box 범위 OK?)", [
    ("사전 코드의 raw", repr(check(good)), "구조는 통과"),
    ('\'{"answer":"1,234,500원","evid\'  (잘린 답)', repr(check(good[:30])), "JSONDecodeError → {} → 전부 누락"),
    ("'합계는 1,234,500원입니다'  (말로 답함)", repr(check("합계는 1,234,500원입니다")), "JSON이 아님"),
    ("'[530, 820, 790, 865]'  (리스트)", repr(check("[530, 820, 790, 865]")), "문법은 맞지만 객체가 아님"),
    ('bbox = ["530","820","790","865"]', repr(check(good.replace("[530,820,790,865]", '["530","820","790","865"]'))), "좌표가 글자"),
    ("bbox = [790,820,530,865]  (x1 > x2)", repr(check(good.replace("[530,820,790,865]", "[790,820,530,865]"))), "뒤집힌 Box"),
    ("bbox = [530,820,1790,865]", repr(check(good.replace("790", "1790", 1).replace('"1,234,5', '"1,234,5'))), "1000×1200 밖"),
])

# 14. 결합 규칙 — OCR 원문은 doc03 기록에서 정규식으로, VLM 답은 Day16 본문 표에 기록된 값
raw = {
    "합계": re.search(r"^TOTAL: (\d+)$", OCR3, re.MULTILINE).group(1),
    "공급자": re.search(r"^SUPPLIER: (.+)$", OCR3, re.MULTILINE).group(1),
    "발행일": re.search(r"^DATE: (\S+)$", OCR3, re.MULTILINE).group(1),
    "세금": None, "할인액": None, "공급자 전화번호": None,
}
vlm = {"합계": "33000", "공급자": "ALPHA STUDIO", "발행일": "2026-11-03", "세금": "0.", "할인액": "33000", "공급자 전화번호": "ALPHA STUDIO"}
final = {k: (raw[k] if vlm[k] == raw[k] else None) for k in vlm}
rows14 = [(f'answer={vlm[k]!r} vs raw={raw[k]!r}', f"final = {final[k]!r}", "✅ 채택" if final[k] is not None else "⏸ 보류 — 문서에 근거 없음")
          for k in vlm]
rows14.append(("max_new_tokens=3으로 잘린 답 '33' vs raw '33000'", f"final = {('33000' if '33' == raw['합계'] else None)!r}", "에러 없이 잘린 답도 대조에서 걸러짐"))
rows14.append(("채택 수 · 보류 수", f"{sum(v is not None for v in final.values())} · {sum(v is None for v in final.values())}", "보류는 정답으로 세지 않아요"))
IO["d16-가장-단순한-결합-규칙"] = ("VLM 답 6개는 Day16 본문 표에 기록된 SmolVLM 결과(이 PC에선 VLM 미실행) · OCR 원문(raw)은 수업 doc03의 실제 Tesseract 기록에서 ^TOTAL: · ^SUPPLIER: · ^DATE: 정규식으로 뽑음 (문서에 없는 항목은 None)", rows14)


# 17. dataclass — 사전 코드 그대로
@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass
class OCRResult:
    full_text: str
    words: list[OCRWord]
    engine: str


@dataclass
class VLMResult:
    answer: str | None
    evidence: str | None
    raw_text: str
    model_id: str


def run_ocr(image) -> OCRResult:
    raise NotImplementedError("OCR 어댑터를 연결하세요")


w = OCRWord(text=tok["text"], confidence=round(tok["conf"], 1), bbox=tuple(tok["bbox"]))
res = OCRResult(full_text=OCR3, words=[OCRWord(t["text"], round(t["conf"], 1), tuple(t["bbox"])) for t in DOC3["ocr"]["tokens"]],
                engine="tesseract 5.5.0 (기록)")
IO["d16-dataclass"] = ("수업 doc03의 기록된 Tesseract 단어로 결과 상자 만들기", [
    ("OCRWord(text, confidence, bbox)  — TOTAL: 단어", repr(w), "출력(__repr__)을 자동으로 만들어 줌"),
    ("len(res.words), res.words[-3].text", f"{len(res.words)}, {res.words[-3].text!r}", "엔진이 달라도 같은 모양으로 주고받기"),
    ("OCRWord('a', 1.0, (0, 0, 1, 1)) == OCRWord('a', 1.0, (0, 0, 1, 1))", repr(OCRWord("a", 1.0, (0, 0, 1, 1)) == OCRWord("a", 1.0, (0, 0, 1, 1))), "값이 같으면 같다(__eq__ 자동)"),
    ('OCRWord(text="a", conf=1.0, bbox=(0, 0, 1, 1))  (이름 오타)', err(lambda: OCRWord(text="a", conf=1.0, bbox=(0, 0, 1, 1))), "dict와 달리 오타를 바로 알려 줌"),
    ('VLMResult(answer=None, evidence=None, raw_text="0.", model_id="SmolVLM-256M")',
     repr(VLMResult(answer=None, evidence=None, raw_text="0.", model_id="SmolVLM-256M")), "근거 없으면 None + 원문은 따로"),
    ("run_ocr(image)", err(lambda: run_ocr(image)), "아직 어댑터가 없다는 걸 분명히"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day16.html"
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
