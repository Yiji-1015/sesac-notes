# OpenCV 이미지 전처리 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/noisy.png      색 도형 사진 (왼쪽 어둡고 오른쪽 밝은 조명, 소금후추 잡음)
#         data/clean_gray.png 같은 장면의 잡음 없는 흑백 (비교용)
#         data/objects.png    흑백 도형 마스크 (작은 점 잡음과 구멍이 섞임), data/mask_truth.png 정답 마스크
#         data/images/train_01.png 증강 예시
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import cv2                                                       #@1
import numpy as np                                               #@2

DATA = Path(__file__).resolve().parent / "data"                  #@1

# ① 그림 읽기
img = cv2.imread(str(DATA / "noisy.png"))        # 숫자 배열 (세로, 가로, 채널). 채널 순서는 B, G, R
red_patch = img[40, 40].tolist()                 # 왼쪽 위 빨간 네모 안의 한 점
red_patch_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[40, 40].tolist()

# ② 밝기와 색으로 고르기
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 채널 3개 → 밝기 하나
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)       # 색상(H), 채도(S), 밝기(V)
cyan = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))  # 하늘색 범위만 255
cyan_area = int((cyan > 0).sum())

# ③ 잡음 줄이기
clean = cv2.imread(str(DATA / "clean_gray.png"), cv2.IMREAD_GRAYSCALE)
gauss = cv2.GaussianBlur(gray, (5, 5), 0)        # 주변 5×5를 가까울수록 크게 반영한 평균
median = cv2.medianBlur(gray, 5)                 # 주변 5×5의 가운데 값
psnr = {"원본": cv2.PSNR(gray, clean), "Gaussian": cv2.PSNR(gauss, clean), "Median": cv2.PSNR(median, clean)}

# ④ 이진화: 하나의 기준 vs 자리마다 다른 기준
otsu_t, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 그림 전체에 기준 하나
adaptive = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 5)                        # 주변 31×31마다 기준
white_left_right = {name: (round(float((b[:, :160] > 0).mean()), 2), round(float((b[:, 160:] > 0).mean()), 2))
                    for name, b in {"Otsu": otsu, "adaptive": adaptive}.items()}

# ⑤ 형태학: 작은 점 지우고 구멍 메우기
objects = cv2.imread(str(DATA / "objects.png"), cv2.IMREAD_GRAYSCALE)
truth = cv2.imread(str(DATA / "mask_truth.png"), cv2.IMREAD_GRAYSCALE) > 0

def iou(mask):
    return float((mask & truth).sum() / (mask | truth).sum())  # 겹친 넓이 / 합친 넓이

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(objects, cv2.MORPH_OPEN, kernel)   # 깎았다 불리기 → 작은 점 사라짐
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)   # 불렸다 깎기 → 작은 구멍 메워짐
morph = {name: {"iou": iou(m > 0), "pieces": cv2.connectedComponents(m)[0] - 1}
         for name, m in {"처음": objects, "열기": opened, "열기 + 닫기": closed}.items()}

# ⑥ 물체 세고 모양 보기
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
shapes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    corners = len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True))  # 윤곽을 꼭짓점 몇 개로 줄일 수 있나
    shapes.append({"x": x, "y": y, "w": w, "h": h, "area": int(cv2.contourArea(c)), "corners": corners})

# ⑦ 증강: 같은 그림을 조금씩 바꿔 늘리기
shape = cv2.imread(str(DATA / "images" / "train_01.png"))
augmented = {"좌우 뒤집기": cv2.flip(shape, 1),
             "15도 회전": cv2.warpAffine(shape, cv2.getRotationMatrix2D((160, 120), 15, 1.0), (320, 240),
                                       borderMode=cv2.BORDER_REFLECT),
             "밝게": cv2.convertScaleAbs(shape, alpha=1.0, beta=40)}
