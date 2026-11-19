"""모델 > 이미지 > OpenCV 이미지 전처리. 그림 출력은 topics/model/img/opencv/ 에 저장한다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/opencv_pre.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import cv2  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "opencv"
IMG = ROOT / "topics" / "model" / "img" / "opencv"
IMG.mkdir(parents=True, exist_ok=True)
code = S.parse((EX / "preprocess.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="preprocess.py")


def save(a, name, w=200):
    h = int(a.shape[0] * w / a.shape[1])
    cv2.imwrite(str(IMG / name), cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST))
    return f"img/opencv/{name}"


outs = {}
outs["why"] = S.images("noisy.png 한 장에 들어 있는 문제들", [(save(ns["img"], "noisy.png", 320), "noisy.png (320×240)")],
                       ["<b>잡음</b>: 흰 점, 검은 점이 흩뿌려져 있어요 (소금후추 잡음).", "<b>조명</b>: 왼쪽은 어둡고 오른쪽은 밝아요. 같은 회색도 자리마다 다른 뜻이에요.", "<b>색</b>: 하늘색 원, 주황 네모처럼 색으로 구분되는 물체가 있어요."])
p = ns["red_patch"]
outs["s1"] = S.box("① 빨간 네모 안의 한 점 img[40, 40]", [(f"cv2.imread 그대로 → {p}", 1), f"RGB로 바꾸면 → {ns['red_patch_rgb']}", f"img.shape → {ns['img'].shape}, dtype {ns['img'].dtype}"],
                   ["빨간색인데 마지막 값이 255예요. OpenCV는 채널을 <b>B, G, R</b> 순서로 읽어요. matplotlib처럼 RGB를 기대하는 도구에 그대로 넘기면 빨강과 파랑이 바뀌어 보여요."])
outs["s2"] = S.images("② 밝기(흑백)와 색(HSV)", [(save(ns["gray"], "gray.png"), "흑백"), (save(ns["cyan"], "cyan.png"), "하늘색만 (inRange)")],
                      [f"흑백에서는 하늘색 원과 배경이 둘 다 비슷한 회색이라 밝기로는 가르기 어려워요. HSV의 색상 범위로 고르면 원만 남아요 ({ns['cyan_area']:,} 픽셀)."])
ps = ns["psnr"]
outs["s3"] = S.images("③ 잡음 줄이기 (깨끗한 흑백과의 PSNR, 클수록 가까움)", [(save(ns["gray"], "g_raw.png"), f"원본 {ps['원본']:.1f} dB"), (save(ns["gauss"], "g_gauss.png"), f"Gaussian {ps['Gaussian']:.1f} dB"), (save(ns["median"], "g_median.png"), f"Median {ps['Median']:.1f} dB")],
                      ["Gaussian은 점을 주변과 <b>섞어서</b> 흐릿한 얼룩으로 남겨요. Median은 주변의 가운데 값을 골라서, 튀는 점 하나는 아예 <b>무시</b>돼요. 소금후추 잡음에는 Median이 훨씬 잘 맞아요 (32.2 dB)."])
wl = ns["white_left_right"]
outs["s4"] = S.images(f"④ 이진화 (Otsu 기준 {ns['otsu_t']:.0f})", [(save(ns["otsu"], "otsu.png"), "Otsu (기준 하나)"), (save(ns["adaptive"], "adaptive.png"), "adaptive (자리마다 기준)")],
                      [f"흰 픽셀 비율 (왼쪽 절반, 오른쪽 절반): Otsu {wl['Otsu']}, adaptive {wl['adaptive']}",
                       "기준이 하나면 조명이 밝은 오른쪽은 거의 다 흰색, 어두운 왼쪽은 거의 다 검은색이 돼요. 물체가 아니라 <b>조명</b>을 갈랐어요. adaptive는 주변 31×31의 평균을 기준으로 삼아서 조명과 상관없이 테두리와 선이 살아나요."])
mo = ns["morph"]
outs["s5"] = S.images("⑤ objects.png에 열기, 닫기", [(save(ns["objects"], "obj.png"), f"처음: 조각 {mo['처음']['pieces']}개"), (save(ns["opened"], "open.png"), f"열기: 조각 {mo['열기']['pieces']}개"), (save(ns["closed"], "close.png"), f"열기 + 닫기: 조각 {mo['열기 + 닫기']['pieces']}개")],
                      ["처음에는 도형 3개 말고도 작은 점 5개가 따로 세어졌어요. 열기로 점이 사라져 3개가 됐어요. 도형 안의 작은 구멍은 닫기로 메웠어요.",
                       f"정답 마스크와 겹친 정도(IoU): 처음 {mo['처음']['iou']:.4f} → 열기 + 닫기 {mo['열기 + 닫기']['iou']:.4f}. 넓이로는 차이가 작지만, <b>개수를 셀 때</b>는 큰 차이예요 (8개 → 3개)."])
vis = cv2.cvtColor(ns["closed"], cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, ns["contours"], -1, (0, 0, 255), 2)
outs["s6"] = S.table("⑥ 윤곽마다 크기와 꼭짓점 수", ["위치 (x, y)", "가로×세로", "넓이", "꼭짓점"], [[f"({s['x']}, {s['y']})", f"{s['w']}×{s['h']}", s["area"], f"<b>{s['corners']}</b>"] for s in ns["shapes"]],
                     ["네모 두 개는 꼭짓점 4개, 원은 8개로 줄었어요 (원은 많을수록 원). 꼭짓점 수로 모양을 대략 나눌 수 있어요."])
outs["s6img"] = S.images("찾은 윤곽 (빨강)", [(save(vis, "contours.png"), "findContours")])
aug = ns["augmented"]
outs["s7"] = S.images("⑦ train_01.png를 조금씩 바꾸기", [(save(ns["shape"], "aug0.png"), "원래")] + [(save(a, f"aug{i + 1}.png"), k) for i, (k, a) in enumerate(aug.items())],
                      ["같은 도형을 뒤집거나 돌리거나 밝게 해도 정답은 그대로예요. 그림을 이렇게 늘려 학습하면 모델이 위치, 방향, 밝기 변화에 강해져요. 단, 정답이 바뀌는 변형(예: 글자를 좌우로 뒤집기)은 안 돼요."])
flow = [[1, "읽기", "BGR 배열", "채널 순서 주의"], [2, "밝기와 색", "GRAY, HSV", "색으로 고르기"], [3, "잡음", "Gaussian, Median", "PSNR"], [4, "이진화", "Otsu, adaptive", "조명 문제"], [5, "형태학", "열기, 닫기", "점과 구멍"], [6, "세기", "윤곽", "꼭짓점"], [7, "증강", "뒤집기, 회전", "데이터 늘리기"]]
S.save(ROOT / "topics" / "model" / "opencv-preprocessing-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": f"Python {sys.version.split()[0]}, OpenCV {cv2.__version__}"})
why = f"""<p>모델에 사진을 넣기 전에, 사진은 대개 그대로 쓰기 어려운 상태예요. 이 한 장에 흔한 문제가 다 들어 있어요.</p>{K.out("why")}
<p><b>OpenCV</b>는 그림을 숫자 배열로 다루는 라이브러리예요. 잡음을 줄이고, 물체와 배경을 나누고, 모양을 세고, 그림을 늘리는 <b>전처리</b>를 맡아요. 전처리를 잘하면 모델이 배울 것이 줄고, 모델 없이 규칙만으로 풀리는 문제도 많아요.</p>"""
whole = """<p>읽기 → 밝기와 색으로 고르기 → 잡음 줄이기 → 흑백 두 값으로 나누기 → 모양 다듬기 → 세기 → 늘리기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "그림 읽기", "그림 파일을 (세로, 가로, 채널) 숫자 배열로 읽어요.",
           K.code_table([("cv2.imread(path)", "컬러 그림을 BGR 순서의 uint8(0~255) 배열로. 파일이 없으면 오류 대신 <code>None</code>을 돌려줘서 꼭 확인해요."), ("img[40, 40]", "(세로 40, 가로 40) 자리의 픽셀. 좌표 순서가 (y, x)예요."), ("cv2.cvtColor(img, cv2.COLOR_BGR2RGB)", "채널 순서 바꾸기")]), "s1"),
    K.step(2, "밝기와 색으로 고르기", "흑백으로 바꾸거나, HSV로 바꿔 특정 색 범위만 골라요.",
           "<p><b>HSV</b>는 색을 색상(Hue, 무슨 색), 채도(Saturation, 얼마나 진한지), 밝기(Value)로 나눠요. 조명이 바뀌면 RGB 세 값이 다 바뀌지만, 색상은 비교적 그대로라 색으로 고르기 쉬워요.</p>"
           + K.code_table([("cv2.COLOR_BGR2GRAY", "밝기 하나로"), ("cv2.COLOR_BGR2HSV", "색상, 채도, 밝기로. OpenCV의 색상은 0~179"), ("cv2.inRange(hsv, low, high)", "세 값이 모두 범위 안이면 255, 아니면 0")]), "s2"),
    K.step(3, "잡음 줄이기", "주변 픽셀을 보고 값을 고쳐서 튀는 점을 줄여요.",
           K.code_table([("cv2.GaussianBlur(gray, (5, 5), 0)", "5×5 주변의 가중 평균. 0이면 퍼짐 정도를 크기에 맞춰 자동"), ("cv2.medianBlur(gray, 5)", "5×5 주변의 가운데 값"), ("cv2.PSNR(a, b)", "두 그림이 얼마나 비슷한지 (dB). 클수록 비슷")]), "s3"),
    K.step(4, "이진화: 기준 하나 vs 자리마다 다른 기준", "픽셀을 기준보다 밝으면 흰색, 어두우면 검은색으로 나눠요.",
           K.code_table([("cv2.threshold(..., cv2.THRESH_BINARY + cv2.THRESH_OTSU)", "Otsu가 그림 전체를 보고 두 무리를 가장 잘 나누는 기준을 하나 정해요. 앞 값이 그 기준"),
                         ("cv2.adaptiveThreshold(img, 255, GAUSSIAN_C, BINARY, 31, 5)", "자리마다 주변 31×31의 가중 평균에서 5를 뺀 값을 기준으로")]), "s4"),
    K.step(5, "형태학: 작은 점 지우고 구멍 메우기", "흰 영역을 깎고 불리는 연산으로 마스크를 다듬어요.",
           K.code_table([("cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))", "깎고 불릴 때 쓸 5×5 원 모양"), ("cv2.MORPH_OPEN", "열기 = 깎기 후 불리기. 깎을 때 사라진 작은 점은 다시 안 생겨요."), ("cv2.MORPH_CLOSE", "닫기 = 불리기 후 깎기. 불릴 때 메워진 작은 구멍은 다시 안 열려요."), ("cv2.connectedComponents(m)", "이어진 흰 덩어리 수 (배경 포함이라 1을 빼요)")]), "s5"),
    K.step(6, "물체 세고 모양 보기", "흰 덩어리의 윤곽을 찾아, 위치, 넓이, 꼭짓점 수를 재요.",
           K.code_table([("cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)", "바깥 윤곽만, 직선 구간은 끝점만 남겨서"), ("cv2.boundingRect(c)", "윤곽을 감싸는 네모 (x, y, 가로, 세로)"),
                         ("cv2.approxPolyDP(c, 0.02 * 둘레, True)", "윤곽을 둘레의 2% 오차 안에서 꼭짓점이 적은 다각형으로 줄이기")]) + K.out("s6img"), "s6"),
    K.step(7, "증강: 같은 그림을 조금씩 바꿔 늘리기", "뒤집기, 회전, 밝기 바꾸기로 학습용 그림을 늘려요.",
           K.code_table([("cv2.flip(img, 1)", "좌우 뒤집기 (0이면 상하)"), ("cv2.getRotationMatrix2D(중심, 각도, 배율)", "회전을 나타내는 2×3 행렬"), ("cv2.warpAffine(img, M, (가로, 세로), borderMode=BORDER_REFLECT)", "행렬대로 옮기고, 빈 가장자리는 거울처럼 채우기"),
                         ("cv2.convertScaleAbs(img, alpha=1.0, beta=40)", "모든 픽셀에 40을 더하고 0~255로 자르기")]), "s7"),
]
practice = [("어떤 잡음인지 먼저 본다", "점처럼 튀는 잡음은 Median, 고르게 자글자글한 잡음은 Gaussian이 잘 맞아요. 줄이면 선과 모서리도 같이 흐려지니 커널 크기를 작게 시작해요."),
            ("조명이 고르지 않다", "기준 하나(Otsu)는 조명을 갈라요. adaptive 이진화를 쓰거나, 배경을 크게 흐린 그림을 빼서 조명을 먼저 평평하게 해요."),
            ("모델에 넣을 그림", "크기를 맞추고(<code>cv2.resize</code>), BGR을 RGB로 바꾸고, 0~1이나 평균 0으로 맞춰요. 학습과 실제 사용에서 전처리가 <b>똑같아야</b> 해요."),
            ("영상", "<code>cv2.VideoCapture</code>로 한 장씩 꺼내 같은 전처리를 해요. 자세한 내용은 <a href=\"../../notes/day13.html\">옛 Day13 노트</a>를 보세요.")]
libs = [("①②", "<code>cv2.imread</code>, <code>cv2.cvtColor</code>, <code>cv2.inRange</code>", "읽기, 색 바꾸기, 색 고르기"), ("③", "<code>cv2.GaussianBlur</code>, <code>cv2.medianBlur</code>, <code>cv2.PSNR</code>", "잡음"),
        ("④", "<code>cv2.threshold</code>, <code>cv2.adaptiveThreshold</code>", "이진화"), ("⑤⑥", "<code>cv2.morphologyEx</code>, <code>cv2.connectedComponents</code>, <code>cv2.findContours</code>, <code>cv2.approxPolyDP</code>, <code>cv2.boundingRect</code>", "형태학, 윤곽"),
        ("⑦", "<code>cv2.flip</code>, <code>cv2.warpAffine</code>, <code>cv2.getRotationMatrix2D</code>, <code>cv2.convertScaleAbs</code>", "증강")]
K.build(ROOT / "topics" / "model" / "opencv-preprocessing.html", title="OpenCV 이미지 전처리", h1="OpenCV 이미지 전처리: 잡음, 조명, 모양 다루기", track="모델", crumb="모델 &gt; 이미지",
        one_line="사진은 잡음, 고르지 않은 조명, 섞인 물체 때문에 그대로 쓰기 어려워요. <b>OpenCV</b>로 잡음을 줄이고, 물체와 배경을 나누고, 모양을 세고, 그림을 늘려요.",
        why=why, whole=whole, steps=steps, code_file="examples/opencv/preprocess.py", practice=practice, libs=libs, libs_note="<code>iou</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"cnn.html\">CNN과 이미지 분류</a>", "다듬은 그림을 모델로 분류하기"), ("<a href=\"ocr-documents.html\">OCR과 문서 구조화</a>", "글자를 읽기 전 문서 그림 다듬기")],
        sources=["다루는 범위: Day13 (2026-11-04) 이미지 표현, 색공간, 필터, 형태학, 증강", "데이터: <code>examples/opencv/data/</code> (수업용으로 만든 합성 도형 그림. 기존 노트와 같은 예제)", "결과 값: <code>tools/pages/opencv_pre.py</code>",
                 "공식 문서: <a href=\"https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html\" target=\"_blank\">OpenCV-Python Tutorials</a>"])
print("ok")
