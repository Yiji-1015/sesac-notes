window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import cv2",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 2
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
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 그림 읽기",
   "s": 1
  },
  {
   "t": "img = cv2.imread(str(DATA / \"noisy.png\"))  # 숫자 배열 (세로, 가로, 채널). 채널 순서는 B, G, R",
   "s": 1
  },
  {
   "t": "red_patch = img[40, 40].tolist()  # 왼쪽 위 빨간 네모 안의 한 점",
   "s": 1
  },
  {
   "t": "red_patch_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[40, 40].tolist()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 밝기와 색으로 고르기",
   "s": 2
  },
  {
   "t": "gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 채널 3개 → 밝기 하나",
   "s": 2
  },
  {
   "t": "hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 색상(H), 채도(S), 밝기(V)",
   "s": 2
  },
  {
   "t": "cyan = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))  # 하늘색 범위만 255",
   "s": 2
  },
  {
   "t": "cyan_area = int((cyan > 0).sum())",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 잡음 줄이기",
   "s": 3
  },
  {
   "t": "clean = cv2.imread(str(DATA / \"clean_gray.png\"), cv2.IMREAD_GRAYSCALE)",
   "s": 3
  },
  {
   "t": "gauss = cv2.GaussianBlur(gray, (5, 5), 0)  # 주변 5×5를 가까울수록 크게 반영한 평균",
   "s": 3
  },
  {
   "t": "median = cv2.medianBlur(gray, 5)  # 주변 5×5의 가운데 값",
   "s": 3
  },
  {
   "t": "psnr = {\"원본\": cv2.PSNR(gray, clean), \"Gaussian\": cv2.PSNR(gauss, clean), \"Median\": cv2.PSNR(median, clean)}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 이진화: 하나의 기준 vs 자리마다 다른 기준",
   "s": 4
  },
  {
   "t": "otsu_t, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 그림 전체에 기준 하나",
   "s": 4
  },
  {
   "t": "adaptive = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,",
   "s": 4
  },
  {
   "t": "                                 cv2.THRESH_BINARY, 31, 5)  # 주변 31×31마다 기준",
   "s": 4
  },
  {
   "t": "white_left_right = {name: (round(float((b[:, :160] > 0).mean()), 2), round(float((b[:, 160:] > 0).mean()), 2))",
   "s": 4
  },
  {
   "t": "                    for name, b in {\"Otsu\": otsu, \"adaptive\": adaptive}.items()}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 형태학: 작은 점 지우고 구멍 메우기",
   "s": 5
  },
  {
   "t": "objects = cv2.imread(str(DATA / \"objects.png\"), cv2.IMREAD_GRAYSCALE)",
   "s": 5
  },
  {
   "t": "truth = cv2.imread(str(DATA / \"mask_truth.png\"), cv2.IMREAD_GRAYSCALE) > 0",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def iou(mask):",
   "s": 5
  },
  {
   "t": "    return float((mask & truth).sum() / (mask | truth).sum())  # 겹친 넓이 / 합친 넓이",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))",
   "s": 5
  },
  {
   "t": "opened = cv2.morphologyEx(objects, cv2.MORPH_OPEN, kernel)  # 깎았다 불리기 → 작은 점 사라짐",
   "s": 5
  },
  {
   "t": "closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # 불렸다 깎기 → 작은 구멍 메워짐",
   "s": 5
  },
  {
   "t": "morph = {name: {\"iou\": iou(m > 0), \"pieces\": cv2.connectedComponents(m)[0] - 1}",
   "s": 5
  },
  {
   "t": "         for name, m in {\"처음\": objects, \"열기\": opened, \"열기 + 닫기\": closed}.items()}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 물체 세고 모양 보기",
   "s": 6
  },
  {
   "t": "contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)",
   "s": 6
  },
  {
   "t": "shapes = []",
   "s": 6
  },
  {
   "t": "for c in contours:",
   "s": 6
  },
  {
   "t": "    x, y, w, h = cv2.boundingRect(c)",
   "s": 6
  },
  {
   "t": "    corners = len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))  # 윤곽을 꼭짓점 몇 개로 줄일 수 있나",
   "s": 6
  },
  {
   "t": "    shapes.append({\"x\": x, \"y\": y, \"w\": w, \"h\": h, \"area\": int(cv2.contourArea(c)), \"corners\": corners})",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 증강: 같은 그림을 조금씩 바꿔 늘리기",
   "s": 7
  },
  {
   "t": "shape = cv2.imread(str(DATA / \"images\" / \"train_01.png\"))",
   "s": 7
  },
  {
   "t": "augmented = {\"좌우 뒤집기\": cv2.flip(shape, 1),",
   "s": 7
  },
  {
   "t": "             \"15도 회전\": cv2.warpAffine(shape, cv2.getRotationMatrix2D((160, 120), 15, 1.0), (320, 240),",
   "s": 7
  },
  {
   "t": "                                       borderMode=cv2.BORDER_REFLECT),",
   "s": 7
  },
  {
   "t": "             \"밝게\": cv2.convertScaleAbs(shape, alpha=1.0, beta=40)}",
   "s": 7
  }
 ],
 "names": {
  "1": "그림 읽기",
  "2": "밝기와 색으로 고르기",
  "3": "잡음 줄이기",
  "4": "이진화: 하나의 기준 vs 자리마다 다른 기준",
  "5": "형태학: 작은 점 지우고 구멍 메우기",
  "6": "물체 세고 모양 보기",
  "7": "증강: 같은 그림을 조금씩 바꿔 늘리기"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "why": {
   "title": "noisy.png 한 장에 들어 있는 문제들",
   "images": [
    [
     "img/opencv/noisy.png",
     "noisy.png (320×240)"
    ]
   ],
   "notes": [
    "<b>잡음</b>: 흰 점, 검은 점이 흩뿌려져 있어요 (소금후추 잡음).",
    "<b>조명</b>: 왼쪽은 어둡고 오른쪽은 밝아요. 같은 회색도 자리마다 다른 뜻이에요.",
    "<b>색</b>: 하늘색 원, 주황 네모처럼 색으로 구분되는 물체가 있어요."
   ]
  },
  "s1": {
   "title": "① 빨간 네모 안의 한 점 img[40, 40]",
   "lines": [
    [
     "cv2.imread 그대로 → [0, 0, 255]",
     1
    ],
    "RGB로 바꾸면 → [255, 0, 0]",
    "img.shape → (240, 320, 3), dtype uint8"
   ],
   "notes": [
    "빨간색인데 마지막 값이 255예요. OpenCV는 채널을 <b>B, G, R</b> 순서로 읽어요. matplotlib처럼 RGB를 기대하는 도구에 그대로 넘기면 빨강과 파랑이 바뀌어 보여요."
   ]
  },
  "s2": {
   "title": "② 밝기(흑백)와 색(HSV)",
   "images": [
    [
     "img/opencv/gray.png",
     "흑백"
    ],
    [
     "img/opencv/cyan.png",
     "하늘색만 (inRange)"
    ]
   ],
   "notes": [
    "흑백에서는 하늘색 원과 배경이 둘 다 비슷한 회색이라 밝기로는 가르기 어려워요. HSV의 색상 범위로 고르면 원만 남아요 (6,398 픽셀)."
   ]
  },
  "s3": {
   "title": "③ 잡음 줄이기 (깨끗한 흑백과의 PSNR, 클수록 가까움)",
   "images": [
    [
     "img/opencv/g_raw.png",
     "원본 21.8 dB"
    ],
    [
     "img/opencv/g_gauss.png",
     "Gaussian 26.6 dB"
    ],
    [
     "img/opencv/g_median.png",
     "Median 32.2 dB"
    ]
   ],
   "notes": [
    "Gaussian은 점을 주변과 <b>섞어서</b> 흐릿한 얼룩으로 남겨요. Median은 주변의 가운데 값을 골라서, 튀는 점 하나는 아예 <b>무시</b>돼요. 소금후추 잡음에는 Median이 훨씬 잘 맞아요 (32.2 dB)."
   ]
  },
  "s4": {
   "title": "④ 이진화 (Otsu 기준 116)",
   "images": [
    [
     "img/opencv/otsu.png",
     "Otsu (기준 하나)"
    ],
    [
     "img/opencv/adaptive.png",
     "adaptive (자리마다 기준)"
    ]
   ],
   "notes": [
    "흰 픽셀 비율 (왼쪽 절반, 오른쪽 절반): Otsu (0.25, 0.86), adaptive (0.81, 0.83)",
    "기준이 하나면 조명이 밝은 오른쪽은 거의 다 흰색, 어두운 왼쪽은 거의 다 검은색이 돼요. 물체가 아니라 <b>조명</b>을 갈랐어요. adaptive는 주변 31×31의 평균을 기준으로 삼아서 조명과 상관없이 테두리와 선이 살아나요."
   ]
  },
  "s5": {
   "title": "⑤ objects.png에 열기, 닫기",
   "images": [
    [
     "img/opencv/obj.png",
     "처음: 조각 8개"
    ],
    [
     "img/opencv/open.png",
     "열기: 조각 3개"
    ],
    [
     "img/opencv/close.png",
     "열기 + 닫기: 조각 3개"
    ]
   ],
   "notes": [
    "처음에는 도형 3개 말고도 작은 점 5개가 따로 세어졌어요. 열기로 점이 사라져 3개가 됐어요. 도형 안의 작은 구멍은 닫기로 메웠어요.",
    "정답 마스크와 겹친 정도(IoU): 처음 0.9979 → 열기 + 닫기 0.9987. 넓이로는 차이가 작지만, <b>개수를 셀 때</b>는 큰 차이예요 (8개 → 3개)."
   ]
  },
  "s6": {
   "title": "⑥ 윤곽마다 크기와 꼭짓점 수",
   "cols": [
    "위치 (x, y)",
    "가로×세로",
    "넓이",
    "꼭짓점"
   ],
   "rows": [
    [
     "(165, 175)",
     "101×31",
     2994,
     "<b>4</b>"
    ],
    [
     "(161, 65)",
     "79×81",
     4910,
     "<b>8</b>"
    ],
    [
     "(35, 50)",
     "66×91",
     5844,
     "<b>4</b>"
    ]
   ],
   "notes": [
    "네모 두 개는 꼭짓점 4개, 원은 8개로 줄었어요 (원은 많을수록 원). 꼭짓점 수로 모양을 대략 나눌 수 있어요."
   ]
  },
  "s6img": {
   "title": "찾은 윤곽 (빨강)",
   "images": [
    [
     "img/opencv/contours.png",
     "findContours"
    ]
   ],
   "notes": []
  },
  "s7": {
   "title": "⑦ train_01.png를 조금씩 바꾸기",
   "images": [
    [
     "img/opencv/aug0.png",
     "원래"
    ],
    [
     "img/opencv/aug1.png",
     "좌우 뒤집기"
    ],
    [
     "img/opencv/aug2.png",
     "15도 회전"
    ],
    [
     "img/opencv/aug3.png",
     "밝게"
    ]
   ],
   "notes": [
    "같은 도형을 뒤집거나 돌리거나 밝게 해도 정답은 그대로예요. 그림을 이렇게 늘려 학습하면 모델이 위치, 방향, 밝기 변화에 강해져요. 단, 정답이 바뀌는 변형(예: 글자를 좌우로 뒤집기)은 안 돼요."
   ]
  }
 },
 "flow": [
  [
   1,
   "읽기",
   "BGR 배열",
   "채널 순서 주의"
  ],
  [
   2,
   "밝기와 색",
   "GRAY, HSV",
   "색으로 고르기"
  ],
  [
   3,
   "잡음",
   "Gaussian, Median",
   "PSNR"
  ],
  [
   4,
   "이진화",
   "Otsu, adaptive",
   "조명 문제"
  ],
  [
   5,
   "형태학",
   "열기, 닫기",
   "점과 구멍"
  ],
  [
   6,
   "세기",
   "윤곽",
   "꼭짓점"
  ],
  [
   7,
   "증강",
   "뒤집기, 회전",
   "데이터 늘리기"
  ]
 ],
 "env": "Python 3.13.16, OpenCV 4.13.0"
};
