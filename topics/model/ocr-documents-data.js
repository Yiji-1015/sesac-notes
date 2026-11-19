window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import cv2",
   "s": 1
  },
  {
   "t": "import pytesseract",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 3
  },
  {
   "t": "import re",
   "s": 5
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
   "t": "pytesseract.pytesseract.tesseract_cmd = r\"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"  #@1  이 PC의 설치 위치",
   "s": 1
  },
  {
   "t": "CONFIG = \"--psm 6\"  # 그림 전체를 글 한 덩어리로 보기",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 그대로 읽기",
   "s": 1
  },
  {
   "t": "lesson = json.loads((DATA / \"lesson.json\").read_text(encoding=\"utf-8\"))",
   "s": 1
  },
  {
   "t": "truth = {k: v[\"reference_text\"] for k, v in lesson[\"documents\"].items()}",
   "s": 1
  },
  {
   "t": "doc01 = cv2.imread(str(DATA / \"doc01.png\"))",
   "s": 1
  },
  {
   "t": "text01 = pytesseract.image_to_string(doc01, lang=\"eng\", config=CONFIG)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 얼마나 틀렸나 재기 (CER)",
   "s": 2
  },
  {
   "t": "def cer(pred, ref):",
   "s": 2
  },
  {
   "t": "    pred, ref = \" \".join(pred.split()), \" \".join(ref.split())  # 줄바꿈, 공백 차이는 빼고",
   "s": 2
  },
  {
   "t": "    d = list(range(len(ref) + 1))  # 편집 거리: 몇 글자를 고치면 같아지나",
   "s": 2
  },
  {
   "t": "    for i, p in enumerate(pred, 1):",
   "s": 2
  },
  {
   "t": "        prev, d[0] = d[0], i",
   "s": 2
  },
  {
   "t": "        for j, r in enumerate(ref, 1):",
   "s": 2
  },
  {
   "t": "            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (p != r))",
   "s": 2
  },
  {
   "t": "    return d[len(ref)] / len(ref)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "cer01 = cer(text01, truth[\"doc01\"])",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 나쁜 사진을 다듬고 읽기",
   "s": 3
  },
  {
   "t": "def upscale(img):  # 작은 글자 → 3배로 키우기",
   "s": 3
  },
  {
   "t": "    return cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def deskew(img):  # 기울어진 글줄 → 반듯하게",
   "s": 3
  },
  {
   "t": "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
   "s": 3
  },
  {
   "t": "    ink = np.column_stack(np.where(gray < 128))[:, ::-1].astype(np.float32)  # 글자 픽셀 좌표 (x, y)",
   "s": 3
  },
  {
   "t": "    angle = cv2.minAreaRect(ink)[2]",
   "s": 3
  },
  {
   "t": "    angle = angle - 90 if angle > 45 else angle",
   "s": 3
  },
  {
   "t": "    M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)",
   "s": 3
  },
  {
   "t": "    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255, 255, 255))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def unshadow(img):  # 그림자 → 배경 밝기로 나눠 평평하게",
   "s": 3
  },
  {
   "t": "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)",
   "s": 3
  },
  {
   "t": "    background = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 31)",
   "s": 3
  },
  {
   "t": "    return cv2.normalize(cv2.divide(gray, background, scale=255), None, 0, 255, cv2.NORM_MINMAX)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "fixes = {\"lowres\": upscale, \"rotated\": deskew, \"shadow\": unshadow}",
   "s": 3
  },
  {
   "t": "bad = {}",
   "s": 3
  },
  {
   "t": "for name, fix in fixes.items():",
   "s": 3
  },
  {
   "t": "    raw = cv2.imread(str(DATA / f\"{name}_raw.png\"))",
   "s": 3
  },
  {
   "t": "    before = pytesseract.image_to_string(raw, lang=\"eng\", config=CONFIG)",
   "s": 3
  },
  {
   "t": "    after = pytesseract.image_to_string(fix(raw), lang=\"eng\", config=CONFIG)",
   "s": 3
  },
  {
   "t": "    bad[name] = {\"before\": before, \"after\": after,",
   "s": 3
  },
  {
   "t": "                 \"cer_before\": cer(before, truth[\"doc03\"]), \"cer_after\": cer(after, truth[\"doc03\"])}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 단어마다 위치와 확신 받기",
   "s": 4
  },
  {
   "t": "data = pytesseract.image_to_data(doc01, lang=\"eng\", config=CONFIG, output_type=pytesseract.Output.DICT)",
   "s": 4
  },
  {
   "t": "words = [{\"text\": t, \"conf\": round(float(c), 1), \"box\": (x, y, w, h)}",
   "s": 4
  },
  {
   "t": "         for t, c, x, y, w, h in zip(data[\"text\"], data[\"conf\"], data[\"left\"], data[\"top\"], data[\"width\"], data[\"height\"])",
   "s": 4
  },
  {
   "t": "         if t.strip()]",
   "s": 4
  },
  {
   "t": "unsure = [w for w in words if w[\"conf\"] < 90]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 필드 뽑기 (구조화)",
   "s": 5
  },
  {
   "t": "def parse(text):",
   "s": 5
  },
  {
   "t": "    fields = {\"doc_id\": re.search(r\"INVOICE\\s+(\\S+)\", text),",
   "s": 5
  },
  {
   "t": "              \"date\": re.search(r\"DATE:\\s*(\\d{4}-\\d{2}-\\d{2})\", text),",
   "s": 5
  },
  {
   "t": "              \"supplier\": re.search(r\"SUPPLIER:\\s*(.+)\", text),",
   "s": 5
  },
  {
   "t": "              \"total\": re.search(r\"TOTAL:\\s*(\\d+)\", text)}",
   "s": 5
  },
  {
   "t": "    out = {k: (m.group(1).strip() if m else None) for k, m in fields.items()}",
   "s": 5
  },
  {
   "t": "    out[\"items\"] = [{\"item\": a, \"qty\": int(b), \"price\": int(c), \"amount\": int(d)}",
   "s": 5
  },
  {
   "t": "                    for a, b, c, d in re.findall(r\"^([A-Z]+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s*$\", text, re.M)]",
   "s": 5
  },
  {
   "t": "    return out",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "parsed = {k: parse(pytesseract.image_to_string(cv2.imread(str(DATA / f\"{k}.png\")), lang=\"eng\", config=CONFIG))",
   "s": 5
  },
  {
   "t": "          for k in truth}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 규칙으로 검산하기",
   "s": 6
  },
  {
   "t": "def check(doc):",
   "s": 6
  },
  {
   "t": "    problems = []",
   "s": 6
  },
  {
   "t": "    if not doc[\"doc_id\"] or not re.fullmatch(r\"DOC\\d{2}\", doc[\"doc_id\"]):",
   "s": 6
  },
  {
   "t": "        problems.append(f\"doc_id 모양 이상: {doc['doc_id']}\")",
   "s": 6
  },
  {
   "t": "    for it in doc[\"items\"]:",
   "s": 6
  },
  {
   "t": "        if it[\"qty\"] * it[\"price\"] != it[\"amount\"]:",
   "s": 6
  },
  {
   "t": "            problems.append(f\"{it['item']}: {it['qty']}×{it['price']} ≠ {it['amount']}\")",
   "s": 6
  },
  {
   "t": "    if doc[\"total\"] is None or sum(it[\"amount\"] for it in doc[\"items\"]) != int(doc[\"total\"]):",
   "s": 6
  },
  {
   "t": "        problems.append(f\"합계 불일치: 항목 합 {sum(it['amount'] for it in doc['items'])}, TOTAL {doc['total']}\")",
   "s": 6
  },
  {
   "t": "    return problems or [\"통과\"]",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "checks = {k: check(v) for k, v in parsed.items()}",
   "s": 6
  }
 ],
 "names": {
  "1": "그대로 읽기",
  "2": "얼마나 틀렸나 재기 (CER)",
  "3": "나쁜 사진을 다듬고 읽기",
  "4": "단어마다 위치와 확신 받기",
  "5": "필드 뽑기 (구조화)",
  "6": "규칙으로 검산하기"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "why": {
   "title": "같은 청구서(doc03)를 세 가지로 찍은 사진",
   "images": [
    [
     "img/ocr/doc03.png",
     "깨끗한 원본"
    ],
    [
     "img/ocr/lowres_raw.png",
     "lowres: 글자 오류율 10%"
    ],
    [
     "img/ocr/rotated_raw.png",
     "rotated: 글자 오류율 3%"
    ],
    [
     "img/ocr/shadow_raw.png",
     "shadow: 글자 오류율 100%"
    ]
   ],
   "notes": [
    "사람 눈에는 넷 다 읽히는데, OCR 결과는 크게 달라요. 그림자 진 사진은 <b>한 글자도</b> 읽지 못했어요 (오류율 100%).",
    "그리고 다 읽었다 해도 글자 덩어리일 뿐이에요. 업무에 쓰려면 \"날짜는?\", \"합계는?\"처럼 <b>값</b>으로 꺼내고, 맞는지 <b>검산</b>해야 해요."
   ]
  },
  "s1": {
   "title": "① doc01.png를 그대로 OCR",
   "lines": [
    "INVOICE DOCO1",
    "",
    "DATE: 2026-11-01",
    "SUPPLIER: BLUE CAFE",
    "ITEM QTY PRICE AMOUNT",
    "TEA 2 3000 6000",
    "COFFEE 3 5000 15000",
    "CAKE 1 6000 6000",
    "TOTAL: 27000",
    "CURRENCY: KRW",
    "",
    [
     "정답 첫 줄 → INVOICE DOC01",
     1
    ]
   ],
   "notes": [
    "깨끗한 문서라 거의 다 맞았지만, 첫 줄의 <code>DOC01</code>을 <code>DOCO1</code>로 읽었어요. 숫자 0과 영어 O는 글꼴에 따라 거의 똑같이 생겨서 OCR이 자주 헷갈려요."
   ]
  },
  "s2": {
   "title": "② cer(text01, 정답)",
   "lines": [
    [
     "CER → 0.007",
     1
    ]
   ],
   "notes": [
    "글자 오류율(CER) = 정답과 같아지려면 고쳐야 하는 글자 수 ÷ 정답 글자 수. 0.007이면 140글자 중 1글자 정도예요. 한 글자지만, 하필 문서 번호예요."
   ]
  },
  "s3": {
   "title": "③ 다듬기 전후의 글자 오류율 (정답: doc03)",
   "cols": [
    "사진",
    "그대로",
    "다듬은 뒤",
    "그대로 읽은 앞부분"
   ],
   "rows": [
    [
     "lowres",
     "0.101",
     "<b>0.006</b>",
     "<code>INVOICE DoCe3 ave: 2026-11-03 SUPPLIER: </code>"
    ],
    [
     "rotated",
     "0.025",
     "<b>0.006</b>",
     "<code>INVOICE poco3 DATE: 2026-11-03 SUPPLIER:</code>"
    ],
    [
     "shadow",
     "1.000",
     "<b>0.006</b>",
     "<code>(빈 글)</code>"
    ]
   ],
   "notes": [
    "세 사진 모두 다듬은 뒤에는 깨끗한 원본과 같은 수준(0.006)이 됐어요. 문제마다 맞는 처방이 달라요: 작으면 키우고, 기울면 바로 세우고, 그림자는 배경 밝기로 나눠 없애요."
   ]
  },
  "s3img": {
   "title": "다듬은 뒤",
   "images": [
    [
     "img/ocr/lowres_fixed.png",
     "lowres"
    ],
    [
     "img/ocr/rotated_fixed.png",
     "rotated"
    ],
    [
     "img/ocr/shadow_fixed.png",
     "shadow"
    ]
   ],
   "notes": []
  },
  "s4": {
   "title": "④ image_to_data: 단어 27개 중 확신 90 미만",
   "cols": [
    "단어",
    "확신(conf)",
    "위치 (x, y, 가로, 세로)"
   ],
   "rows": [
    [
     "<code>DOCO1</code>",
     69.0,
     "(221, 47, 98, 25)"
    ]
   ],
   "notes": [
    "단어마다 글자, 확신(0~100), 박스를 받았어요. 확신이 낮은 단어는 딱 하나, 앞에서 틀린 <code>DOCO1</code>이에요. OCR도 스스로 자신 없어 했어요. 확신이 낮은 단어만 사람에게 보여 주면 확인할 양이 크게 줄어요."
   ]
  },
  "s5": {
   "title": "⑤ parse(doc01의 OCR 글)",
   "lines": [
    "{",
    " \"doc_id\": \"DOCO1\",",
    " \"date\": \"2026-11-01\",",
    " \"supplier\": \"BLUE CAFE\",",
    " \"total\": \"27000\",",
    " \"items\": [",
    "  {",
    "   \"item\": \"TEA\",",
    "   \"qty\": 2,",
    "  …"
   ],
   "notes": [
    "글자 덩어리에서 \"DATE: 뒤의 날짜\", \"TOTAL: 뒤의 숫자\"처럼 <b>표시 글자 옆의 값</b>을 정규식으로 꺼냈어요. 항목 줄은 \"영어 단어 + 숫자 셋\" 모양인 줄만 골랐어요."
   ]
  },
  "s6": {
   "title": "⑥ 다섯 문서 검산 결과",
   "cols": [
    "문서",
    "날짜",
    "합계",
    "항목 수",
    "검산"
   ],
   "rows": [
    [
     "doc01",
     "2026-11-01",
     "27000",
     3,
     "<span class=\"bad\">doc_id 모양 이상: DOCO1</span>"
    ],
    [
     "doc02",
     "2026-11-02",
     "18000",
     3,
     "<span class=\"bad\">doc_id 모양 이상: DOCO2</span>"
    ],
    [
     "doc03",
     "2026-11-03",
     "33000",
     3,
     "<span class=\"bad\">doc_id 모양 이상: DOCO3</span>"
    ],
    [
     "doc04",
     "2026-11-04",
     "12500",
     3,
     "<span class=\"bad\">doc_id 모양 이상: DOCO4</span>"
    ],
    [
     "doc05",
     "2026-11-05",
     "24000",
     3,
     "<span class=\"bad\">doc_id 모양 이상: DOCO5</span>"
    ]
   ],
   "notes": [
    "금액(수량 × 단가)과 합계는 다섯 문서 모두 맞아요. 숫자 읽기는 정확했어요.",
    "다섯 문서 모두 문서 번호에서 걸렸어요. \"DOC + 숫자 두 자리\"여야 하는데 O가 섞였거든요. 이런 문서는 자동으로 넘기지 않고 사람 확인으로 보내요. 숫자 자리의 O를 0으로 바꾸는 규칙을 둘 수도 있지만, 바꾼 사실을 기록해 둬요."
   ]
  }
 },
 "flow": [
  [
   1,
   "그대로 읽기",
   "image_to_string",
   "O와 0"
  ],
  [
   2,
   "오류율",
   "CER",
   "편집 거리"
  ],
  [
   3,
   "다듬기",
   "키우기, 바로 세우기, 그림자",
   "100% → 0.6%"
  ],
  [
   4,
   "단어와 확신",
   "image_to_data",
   "낮은 것만 확인"
  ],
  [
   5,
   "구조화",
   "정규식",
   "값으로"
  ],
  [
   6,
   "검산",
   "규칙",
   "통과 / 사람 확인"
  ]
 ],
 "env": "Python 3.13.16, Tesseract 5.4.0.20240606 (eng), pytesseract, OpenCV 4.13.0"
};
