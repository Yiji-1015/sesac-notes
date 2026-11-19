# OpenCV 이미지 전처리 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/noisy.png (잡음 섞인 도형 사진), data/clean_gray.png (같은 장면의 깨끗한 흑백),
#         data/mask_truth.png (도형 자리 정답), data/images/train_01.png (증강 예시)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import cv2                                                       #@1
import numpy as np                                               #@4

DATA = Path(__file__).resolve().parent / "data"                  #@1

# ① 그림 읽기
img = cv2.imread(str(DATA / "noisy.png"))        # 숫자 배열 (세로, 가로, 채널). 채널 순서는 B, G, R
pixel_bgr = img[120, 160].tolist()
pixel_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[120, 160].tolist()

# ② 흑백으로
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 채널 3개 → 밝기 하나
clean = cv2.imread(str(DATA / "clean_gray.png"), cv2.IMREAD_GRAYSCALE)

# ③ 잡음 줄이기
gauss = cv2.GaussianBlur(gray, (5, 5), 0)        # 주변 5×5를 가까울수록 크게 반영한 평균
median = cv2.medianBlur(gray, 5)                 # 주변 5×5의 가운데 값
psnr = {"원본": cv2.PSNR(gray, clean), "Gaussian": cv2.PSNR(gauss, clean), "Median": cv2.PSNR(median, clean)}

# ④ 이진화: 도형과 배경 나누기
truth = cv2.imread(str(DATA / "mask_truth.png"), cv2.IMREAD_GRAYSCALE) > 0

def iou(mask):
    return (mask & truth).sum() / (mask | truth).sum()  # 겹친 넓이 / 합친 넓이

_, fixed = cv2.threshold(median, 127, 255, cv2.THRESH_BINARY_INV)           # 기준을 직접 127로
otsu_t, otsu = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 기준을 그림이 정하게
iou_bin = {"고정 127": iou(fixed > 0), f"Otsu {otsu_t:.0f}": iou(otsu > 0)}

# ⑤ 형태학: 작은 점 지우고 구멍 메우기
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)     # 깎았다 불리기 → 작은 점 사라짐
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)  # 불렸다 깎기 → 작은 구멍 메워짐
iou_morph = {"Otsu만": iou(otsu > 0), "열기": iou(opened > 0), "열기 + 닫기": iou(closed > 0)}

# ⑥ 물체 세기
n, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
areas = sorted(stats[1:, cv2.CC_STAT_AREA].tolist(), reverse=True)  # 0번은 배경
objects = [a for a in areas if a >= 100]         # 너무 작은 덩어리는 잡음으로 보고 빼기

# ⑦ 증강: 같은 그림을 조금씩 바꿔 늘리기
shape = cv2.imread(str(DATA / "images" / "train_01.png"))
augmented = {"좌우 뒤집기": cv2.flip(shape, 1),
             "15도 회전": cv2.warpAffine(shape, cv2.getRotationMatrix2D((160, 120), 15, 1.0), (320, 240), borderMode=cv2.BORDER_REFLECT),
             "밝게": cv2.convertScaleAbs(shape, alpha=1.0, beta=40)}
