# SeSAC Vision·멀티모달 AI 부트캠프 학습 노트 — 전체 본문

사이트: https://yiji-1015.github.io/sesac-notes/  ·  저장소: https://github.com/Yiji-1015/sesac-notes
강의 PDF를 바탕으로 수강생이 다시 정리한 노트입니다(강의 원본 아님). 각 Day = 3줄 요약 → 필요한 패키지 → 이론별 설명(🤔 왜?) + 핵심 코드 한두 줄. 코드 전체·인자 표·실제 실행 입출력은 각 Day 본문 뒤 '🔧 DayN 명령어'(사이트의 commands.html 명령어 사전)에 있습니다.


---

# Day6 · 회귀·분류·KNN·모델 비교
_2026-10-26(월) · 선형회귀 · 로지스틱회귀 · KNN · 스케일링 · 공정한 비교_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day06.html

📌 딱 3줄 요약

- 회귀는 숫자(몇 ms 걸릴까?), 분류는 종류(실패할까?)를 맞히는 문제. 지표도 따로 쓴다 (RMSE ↔ F1).

- 선형회귀는 가중합, 로지스틱회귀는 확률, KNN은 가까운 이웃으로 예측한다.

- 모델 비교는 같은 분할 · 같은 입력 · 같은 지표에서 한 조건만 바꿔서. 고르는 건 valid, test는 끝까지 보관. 고른 Pipeline은 검사하고 저장한다.

🔧 Day6 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| scikit-learn | 모델(선형·로지스틱·KNN), 전처리, 지표, Pipeline 전부 | from sklearn.linear_model import LinearRegression 등 |
| numpy | 숫자 배열 계산 (np.array, np.exp) | import numpy as np |
| pandas | 표(DataFrame)로 데이터 읽고 비교표 만들기 | import pandas as pd |
| joblib | 학습한 Pipeline을 파일로 저장·불러오기 (scikit-learn 설치 시 함께 깔림) | import joblib |
| (표준 라이브러리) | pathlib·tempfile — 저장 경로·임시 폴더 (설치 불필요) | from pathlib import Path, import tempfile |

### 🔧 설치 · 확인

```
python -m pip install numpy pandas scikit-learn     # 새 환경이면 강사가 검증한 requirements로 준비
python --version
python -m pip show numpy pandas scikit-learn
python -c "import sklearn; print(sklearn.__version__)"
.venv\Scripts\python.exe -m pip check               # Windows: 프로젝트 인터프리터로 의존성 충돌 확인
```

💡 pip 이름과 import 이름이 달라요: scikit-learn으로 설치하고 코드에서는 sklearn으로 불러요.

⚠️ 기존 가상환경을 유지하고, 수업 직전에 무조건 최신 버전으로 올리지 말 것. 강의 자료는 Python 3.13.5 · NumPy 2.3.5 · pandas 2.2.3 · scikit-learn 1.8.0에서 검증됐어요(PDF 6쪽). VS Code 인터프리터가 프로젝트의 .venv인지 먼저 확인하세요.

🤔 왜? — 라이브러리 버전이 바뀌면 함수 이름·기본값이 달라질 수 있어요(예: root_mean_squared_error는 최근 버전에 생긴 함수). 같은 코드가 같은 숫자를 내려면 검증된 환경을 그대로 쓰는 게 안전해요.

## 이거 어디에 써?

같은 데이터로도 질문을 두 가지로 만들 수 있어요. 예를 들어 AI 요청 기록이 있다면 "이 요청은 몇 ms 걸릴까?"(회귀)와 "이 요청은 실패할까?"(분류)를 따로 물을 수 있죠.

⏱️

소요 시간 예측요청이 몇 ms 걸릴지 숫자로 (회귀)

🚨

실패 위험 경고실행 전에 실패 확률 계산 (분류)

📧

스팸 · 이탈스팸 여부, 고객 이탈 가능성

🏠

비슷한 사례 찾기가장 비슷한 과거 사례로 판단 (KNN)

|  | 회귀 | 분류 |
|---|---|---|
| 질문 | 실행에 몇 ms 걸릴까? | 실행이 실패할까? |
| 정답 y | 연속적인 숫자 (elapsed_ms) | 0=성공, 1=실패 (failed) |
| 모델 후보 | LinearRegression · KNN 회귀 | LogisticRegression · KNN 분류 |
| 주 지표 | RMSE (큰 오차를 중시) | 실패 클래스의 F1 |

⚠️ 누수 주의 — elapsed_ms는 회귀의 정답은 될 수 있지만, "실행 전" 예측의 입력 X로 넣으면 안 돼요. 실행이 끝나야 알 수 있는 값이니까요. ID·실행 후 정보(status, retry_count)도 입력에서 빼요.

🤔 왜? — 모델은 실제로 쓰일 때 예측 시점에 존재하는 값만 받을 수 있어요. 실행 후에야 생기는 값을 학습에 넣으면 연습 점수는 높게 나와도, 실전에서는 그 값이 없어서 쓸 수 없는 모델이 돼요. 또 회귀·분류의 지표가 다른 이유는 정답의 모양이 달라서예요. 숫자는 "얼마나 빗나갔나(오차 크기)"를, 종류는 "맞혔나·틀렸나(어떤 오류인가)"를 세야 하거든요.

💡 강의의 공통 데이터는 수업용 합성 데이터 100건(일부 결측 포함)이고, 분할은 train 60 · valid 20 · test 20이에요. 아래 숫자들은 "이 작은 자료에서 나온 결과"일 뿐, 실서비스 성능의 근거가 아니에요.

## ① 선형회귀 — 입력의 가중합으로 숫자를 예측

공부시간(x)으로 점수(y)를 맞힌다고 해 봐요. 점들 사이에 가장 잘 맞는 직선 하나를 긋는 게 선형회귀예요.

```
예측 ŷ = w × x + b
  w : 기울기  — x가 1 늘 때 예측이 얼마나 변하나
  b : 절편    — x = 0일 때의 예측값

입력이 여러 개면  ŷ = w₁x₁ + w₂x₂ + … + b
```

잔차 = 실제값 − 예측값 (점과 직선 사이의 세로 거리). fit은 이 잔차를 제곱해 더한 값(잔차제곱합)이 가장 작아지는 w와 b를 찾아요.

🤔 왜 제곱해서 더하지? — 잔차는 +도 있고 −도 있어서 그냥 더하면 서로 지워져요(+10과 −10의 합은 0). 제곱하면 모두 양수가 되고, 크게 빗나간 점일수록 훨씬 큰 벌점(10²=100, 30²=900)을 받아서 직선이 큰 실수를 줄이는 쪽으로 움직여요.

[그림: 공부시간과 점수 5개 점, 학습된 직선, 점과 직선 사이의 잔차 세로선] 무엇을 보나: 검은 점 5개(실제 점수)와 보라 직선(ŷ = 9.3x + 42.3) 사이의 분홍 세로선이 잔차예요. 점선은 6시간까지 직선을 그대로 연장한 예측(98.1)이에요. 🤔 왜? fit은 분홍 선들의 길이를 제곱해 더한 값이 가장 작아지도록 직선을 기울이고 올려요. 그래서 모든 잔차가 ±1 근처로 작아요. 점선 부분엔 학습 점이 없으니, 직선이 거기서도 맞는다는 보장은 없어요(외삽).

💡 학습 오차가 작아도 새 자료에 잘 맞는지는 따로 평가해야 해요. 그래서 train/valid/test를 나눠요.

🤔 왜? — train 점수는 "본 문제로 다시 시험 보기"와 같아요. 이미 정답을 보고 맞춘 직선이니 잘 맞는 게 당연하죠. 처음 보는 문제(valid)에서도 잘 맞아야 진짜 규칙을 배운 거예요.

⚠️ 계수 해석 주의

- 계수는 "다른 입력을 고정했을 때의 변화"일 뿐, 원인·결과의 증명이 아니에요.

- 단위가 다른 입력끼리 계수 크기만 보고 중요도를 비교하지 않아요.

- 학습 범위 밖을 예측하는 외삽은 믿기 어려워요.

🤔 왜? — 계수는 "같이 움직이는 정도"를 잰 것이라, 숨은 다른 요인 때문에 함께 움직였을 수도 있어요(→ 원인 증명 아님). 단위가 다르면 계수 크기도 단위를 따라가요: 글자수를 "글자"로 재면 계수가 작고 "천 글자"로 재면 1000배 커지죠. 외삽이 위험한 건 직선이 관측한 범위 밖에서도 그대로 이어진다는 보장이 없기 때문이에요.

🔧 LinearRegression() 명령어 사전에서 자세히 →

```
model.fit(X, y)
print("6시간:", model.predict([[6]]).round(1)) # [98.1]
```

## ② 회귀 지표와 기준선

데이터는 역할별로 나눠요: train(전처리·모델 학습) → valid(같은 규칙으로 예측해서 후보 비교) → test(최종 확인까지 보관).

[그림: ] 무엇을 보나: 100건이 세 칸으로 나뉘고, 칸마다 하는 일이 하나씩 정해져 있어요. 오른쪽으로 갈수록 "덜 만진" 자료예요. 🤔 왜? 점수는 "그 자료를 얼마나 안 봤나"만큼만 믿을 수 있어요. valid는 고르는 데 여러 번 봤으니 살짝 낙관적이고, test는 한 번도 안 봤으니 가장 정직한 마지막 확인이 돼요.

| 지표 | 계산 핵심 | 해석 |
|---|---|---|
| MAE | \|실제 − 예측\|의 평균 | 정답과 같은 단위 · 작을수록 좋음 |
| MSE | (실제 − 예측)²의 평균 | 단위가 제곱됨 · 큰 오차에 더 큰 벌점 |
| RMSE | MSE의 제곱근 | 정답과 같은 단위 · 큰 오차에 민감 |
| R² | 1 − 잔차제곱합 / 전체제곱합 | 최대 1 · 음수도 가능 · 오류 비율이 아님 |

🤔 왜 train / valid / test를 나눠 쓰지? — train은 전처리와 계수를 배우는 곳, valid는 같은 규칙으로 예측해 후보를 고르는 곳, test는 고르는 데 한 번도 안 써서 마지막에 정직하게 확인하는 곳이에요. 역할을 섞으면 "고른 데이터로 채점"하게 돼서 점수가 실제보다 좋아 보여요. 좋은 학습 점수보다 "같은 valid에서 기준선보다 무엇이 나아졌나"를 봐요.

💡 소요 시간을 ms로 예측하면 MAE·RMSE도 ms예요(RMSE 640 → 640ms). "정확도 90%" 같은 말로 바꾸면 안 돼요.

🤔 왜 RMSE는 제곱했다가 다시 루트를 씌우지? — 제곱(MSE)으로 큰 오차에 큰 벌점을 주고 싶지만, 그러면 단위가 ms²가 돼서 읽기 어려워요. 루트를 씌우면 벌점 성질은 남기고 단위만 ms로 되돌릴 수 있어요.

### MAE와 RMSE가 다른 선택을 할 때

| 모델 | 오차 | MAE | RMSE |
|---|---|---|---|
| A | 0, 0, 100 (큰 실수 1번) | 33.33 | 57.74 |
| B | 40, 40, 40 (고르게 조금씩) | 40.00 | 40.00 |

MAE로 보면 A, RMSE로 보면 B가 나아요. "큰 실패 한 번"을 더 피해야 하는 업무라면 RMSE 관점이 필요해요. 그래서 주 지표는 미리 정하고, 둘 다 기록해요.

🤔 왜 결과가 뒤집히지? — MAE는 오차를 그대로 평균: A = (0+0+100)/3 = 33.33, B = 40. RMSE는 제곱해서 평균: A = √(100²/3) = √3333 ≈ 57.74, B = √(40²) = 40. 제곱하면 100은 10000이 되어 혼자 튀어 오르니까, 큰 실수 한 번이 있는 A가 RMSE에서 불리해져요.

[그림: 모델 A와 B의 요청별 오차 막대와 MAE, RMSE 비교 막대] 무엇을 보나: 왼쪽은 요청별 오차(A는 한 번 100, B는 고르게 40), 오른쪽은 같은 오차로 계산한 MAE·RMSE예요. MAE로는 A(33.3)가, RMSE로는 B(40.0)가 이겨요. 🤔 왜? RMSE는 오차를 제곱한 뒤 평균을 내니 100이 10000으로 부풀어 A를 57.7까지 끌어올려요. 같은 예측인데도 지표가 "큰 실수 한 번"을 얼마나 무겁게 보느냐에 따라 1등이 바뀌는 거예요.

### R²가 음수라는 뜻

R²의 기준은 "평가 자료의 y 평균을 항상 예측하는 경우"예요. 내 모델의 제곱 오차가 그 기준보다 크면 R²는 0보다 작아져요. 코드가 틀렸다는 뜻이 아니라, 평균만 찍는 것보다도 못하다는 뜻이에요.

🤔 왜 음수가 나올 수 있지? — R² = 1 − (내 모델의 잔차제곱합) / (평균만 찍었을 때의 제곱합)이에요. 내 모델의 오차가 "평균 찍기"보다 크면 분수가 1보다 커지고, 1에서 빼면 음수가 돼요. 위쪽 한계는 오차 0일 때의 1이지만 아래쪽은 한계가 없어요. 단, 이 기준의 평균은 평가 중인 y(valid)의 평균이라 계산 기준일 뿐, 학습 때 미리 알 수 있는 값이 아니에요.

[그림: 좋은 모델, 평균만 찍기, 평균보다 나쁜 모델의 R² 비교 세 그림] 무엇을 보나: 같은 점들에 세 가지 예측선을 그렸어요. 세로선(오차)이 짧으면 R²가 1에 가깝고(0.93), 평균선이면 정확히 0, 엉뚱한 선이면 음수(-3.66)예요. 🤔 왜? R²는 "평균만 찍기"의 오차를 기준(분모)으로 삼아요. 오른쪽처럼 세로선이 가운데 그림보다 길어지면 분수가 1을 넘어서 1에서 빼면 음수가 돼요. 계산 실수가 아니라 "평균보다 못하다"는 진단이에요.

💡 기준선(baseline) — DummyRegressor(strategy="mean")는 train의 평균만 계속 예측하는 "아무것도 안 배운 모델"이에요. 새 모델은 최소한 이것보다는 나아야 의미가 있어요. Dummy는 train 평균을 배우고 R²는 valid 평균을 기준으로 하니, Dummy의 valid R²가 꼭 0은 아니에요(이 자료에선 −0.162).

🔧 회귀 지표 명령어 사전에서 자세히 →

```
print("MAE:", round(mean_absolute_error(y_true, y_pred), 2))       # 33.33
print("RMSE:", round(root_mean_squared_error(y_true, y_pred), 2))  # 38.3
```

[그림: 수업 데이터 train의 글자수와 소요시간 산점도, 회귀 직선과 평균선] 무엇을 보나: 수업 데이터 train에서 글자수(가로)와 실제 소요시간(세로)을 찍은 그림이에요. 보라 직선은 위로 조금 기울었지만 점들이 위아래로 아주 넓게 흩어져 있어요. 🤔 왜? 직선과 평균선(점선)이 점들을 설명하는 정도가 거의 비슷하죠. 입력만으로 소요시간을 맞힐 신호가 약하니, valid에서 선형회귀의 R²가 -0.097처럼 0 근처·음수로 나오는 게 이상한 일이 아니에요.

## ③ 로지스틱회귀 — 이름은 회귀지만 분류 모델

입력으로 실패(1)일 가능성을 0~1 사이 확률로 추정하고, 그 확률을 기준(임계값)과 비교해 클래스를 정해요.

입력 x→ 선형 점수 z = w₁x₁ + … + b→ 시그모이드 p = 1 / (1 + e−z)→ 판단 규칙 p ≥ 0.5 → 1

```
z = -2.0  →  p = 0.119  →  0 (성공)
z =  0.0  →  p = 0.500  →  1 (규칙이 p ≥ 0.5이면)
z =  2.0  →  p = 0.881  →  1 (실패)
```

🤔 왜 시그모이드를 거치지? — 선형 점수 z는 −∞~+∞ 어떤 값이든 나올 수 있어서 "확률"로 읽을 수 없어요. 시그모이드는 어떤 z든 0~1 사이로 눌러 담아요. z가 클수록 1에, 작을수록 0에 가까워지고, z=0이면 정확히 e⁰=1이라 p = 1/(1+1) = 0.5가 돼요. 그래서 "z가 커질수록 실패 확률이 커진다"고 읽을 수 있어요.

[그림: 선형 점수 z를 0과 1 사이 확률로 바꾸는 시그모이드 곡선과 0.5 기준선] 무엇을 보나: 가로축 z는 아무 값이나 될 수 있지만, 보라 S자 곡선을 거치면 세로축 p가 항상 0~1 안에 들어와요. 회색 점선(0.5)보다 위는 분홍(실패), 아래는 보라(성공)로 판단해요. 🤔 왜? z가 아주 크거나 작으면 곡선이 1·0에 붙어 거의 안 변하고, z=0 근처에서 가장 가파르게 바뀌어요. 그래서 z=0이 정확히 p=0.5, 즉 "반반"인 경계가 돼요.

⚠️ 확률 0.8 ≠ "80% 확실". 모델이 추정한 값일 뿐이라 따로 검증이 필요해요. 그리고 확률(predict_proba)과 최종 클래스(predict)는 서로 다른 정보예요.

💡 손실 ≠ 평가 지표 — 로지스틱회귀는 학습할 때 로그 손실(정답에 낮은 확률을 주면 커지는 값)과 규제를 줄여요. 모델을 고를 때는 따로 정한 지표(여기선 F1)를 써요. 학습이 F1을 직접 올리는 건 아니에요. 기본 구현에는 계수가 지나치게 커지는 걸 막는 규제가 들어 있어요(C로 조절).

🤔 왜 로그 손실을 쓰지? — 실제 y=1인데 p=0.9를 주면 손실은 −log(0.9) ≈ 0.11, p=0.1을 주면 −log(0.1) ≈ 2.30이에요. 틀린 쪽에 자신 있게 말할수록 벌점이 폭발적으로 커져서, 모델이 확률을 조심스럽게 맞추도록 만들어요. F1은 0/1로 끊은 뒤 세는 값이라 계수를 조금씩 고치는 학습에 쓰기 어려워요(조금 바꿔도 값이 계단처럼 안 변함).

[그림: 실제 정답이 1일 때 예측 확률에 따른 로그 손실 곡선] 무엇을 보나: 실제로 실패(y=1)인 요청에 모델이 준 확률 p별 벌점이에요. p=0.9면 0.11로 작지만, p=0.1이면 2.30으로 커지고, p가 0에 가까워질수록 곡선이 하늘로 치솟아요. 🤔 왜? -log(p)는 p→0에서 무한대로 가요. "틀린 쪽에 자신만만"한 답을 강하게 벌주니까, 모델은 확신이 없을 땐 0.5 근처로 조심스럽게 말하도록 학습돼요.

### 혼동행렬 — 오류 두 종류를 나눠 읽기

실패(1)를 양성으로 정해요. 행 = 실제, 열 = 예측.

|  | 예측 성공(0) | 예측 실패(1) |
|---|---|---|
| 실제 성공(0) | TN 10 | FP 4 — 괜히 경고 (불필요한 점검) |
| 실제 실패(1) | FN 4 — 놓침 (문제를 못 잡음) | TP 2 |

[그림: TN 10, FP 4, FN 4, TP 2인 혼동행렬 네 칸 그림] 무엇을 보나: 행은 실제, 열은 예측이에요. 대각선(TN 10, TP 2)이 맞힌 것, 나머지 두 칸이 서로 다른 두 종류의 실수(FP 괜히 경고 4, FN 놓침 4)예요. 🤔 왜? Accuracy는 대각선 합(12)만 보지만, 실무에선 "어느 칸에서 틀렸나"가 비용을 정해요. F1 공식엔 왼쪽 위 TN 칸이 아예 없어서, 흔한 성공을 많이 맞혀도 F1은 올라가지 않아요.

| 지표 | 계산 | 위 예시 |
|---|---|---|
| Accuracy | (TP + TN) / 전체 | 12 / 20 = 0.600 |
| Precision | TP / (TP + FP) — 경고한 것 중 진짜 실패 | 2 / 6 = 0.333 |
| Recall | TP / (TP + FN) — 진짜 실패 중 찾은 것 | 2 / 6 = 0.333 |
| F1 | 2TP / (2TP + FP + FN) | 4 / 12 = 0.333 |

⚠️ 전부 "성공"이라고 찍어도 Accuracy는 14/20 = 0.70. 하지만 실패 6건을 하나도 못 찾아요. 그래서 실패 클래스의 F1을 주 지표로 써요.

🤔 왜 Accuracy가 속일까? — Accuracy는 "맞힌 개수 / 전체"라서 많은 쪽(성공 14건)만 맞혀도 높게 나와요. 우리가 진짜 찾고 싶은 건 드문 실패인데, 그걸 전부 놓쳐도 0.70이죠. 반면 F1 = 2TP / (2TP + FP + FN)에는 TN이 아예 안 들어가서, "실패를 얼마나 잘 잡고(Recall), 잡았다고 한 게 얼마나 진짜였나(Precision)"만 봐요. 단, 어떤 오류(FP·FN)가 더 비싼지는 F1이 알려 주지 않으니 오류 수도 함께 적어요.

🔧 LogisticRegression · predict_proba 명령어 사전에서 자세히 →

```
clf = build(LogisticRegression(C=1.0, max_iter=1000))   # build = 전처리+모델 Pipeline (⑤ 참고)
p = clf.predict_proba(X_valid)[:, pos]                  # 실패 확률
```

🔧 임계값 바꾸기 명령어 사전에서 자세히 →

```
pred = (p >= threshold).astype(int)                 # True/False → 1/0
tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
```

아래 슬라이더로 임계값을 움직여 보세요. 명령어 사전 코드의 6건(실제 [0, 0, 0, 1, 1, 1], 확률 [0.10, 0.40, 0.70, 0.35, 0.60, 0.90])을 그대로 씁니다.

[그림: 2차원 합성 자료에서 로지스틱회귀의 확률 배경과 0.5, 0.3 결정 경계] 무엇을 보나: 2개 특성으로 만든 설명용 자료예요. 배경색이 진할수록 실패 확률이 높고, 검은 직선이 p=0.5 경계, 주황 점선이 임계값을 0.3으로 내렸을 때의 경계예요. 🤔 왜? 모델(배경색)은 그대로인데 판단 선만 왼쪽 위로 밀려나요. 그 사이에 있던 분홍 네모(실패)가 새로 잡혀 FN이 줄지만, 같은 띠에 있던 보라 점(성공)도 함께 "실패"로 불려 FP가 늘어요. 로지스틱회귀는 z가 특성의 가중합이라 경계가 항상 직선이에요.

## ④ KNN — 가까운 학습 사례를 보고 예측

KNN(K-최근접 이웃)은 "나랑 제일 비슷한 K개는 뭐였지?"를 보는 모델이에요.

새 입력→ 거리 계산 모든 학습 자료와→ K개 이웃 가장 가까운 것→ 분류: 투표 회귀: 평균

예: 학습 x = [1, 2, 4, 7, 9], 라벨 = [0, 0, 1, 1, 1], 새 값 3.2, K=3 → 이웃은 4, 2, 1 (거리 0.8, 1.2, 2.2), 라벨 1, 0, 0 → 투표로 0.

💡 KNN의 fit은 학습 자료를 기억만 하고, 진짜 일은 predict 때(이웃 찾기) 해요. 선형회귀처럼 학습되는 계수 w는 없지만, 학습 사례를 저장한 "fitted 상태"는 있어요. 정답 y는 거리 계산에 넣지 않아요.

🤔 왜 이웃으로 예측이 되지? — "입력이 비슷하면 결과도 비슷할 것"이라는 가정이에요. 그래서 KNN은 무엇을 '비슷하다'고 보느냐(어떤 특성을 넣었나, 단위는 어떤가, 거리를 어떻게 재나)에 따라 이웃이 바뀌고 예측도 바뀌어요. 위 예에서 3.2는 1·2(라벨 0)와 4(라벨 1) 사이인데, 가까운 3개 중 2개가 0이라 0이 돼요.

아래에서 ⭐ 새 요청을 끌어 보세요. 가로축(글자수)은 수백~수천, 세로축(허용 도구수)은 1~5라 단위가 크게 달라요. 표준화를 켜고 끄면 이웃이 바뀌는 걸 볼 수 있어요.

🔧 KNeighborsClassifier · KNeighborsRegressor 명령어 사전에서 자세히 →

```
knn = KNeighborsClassifier(n_neighbors=3).fit(X, [0, 0, 1, 1, 1])
distance, index = knn.kneighbors([[3.2]])     # 이웃까지 거리, 학습 행 번호
```

[그림: 학습 범위 밖 x=8에서 선형회귀는 80, KNN은 35를 예측하는 그림] 무엇을 보나: 학습 점은 x=1~4뿐이에요(회색 띠 밖). 보라 직선은 회색 띠 안으로 쭉 올라가 x=8에서 80, 초록 계단(KNN)은 x=4 이후 35에서 평평하게 멈춰요. 🤔 왜? KNN 예측은 "가까운 이웃 정답의 평균"이라 본 적 있는 값(10~40) 사이에 갇혀요. 선형회귀는 기울기를 끝없이 이어 가요. 회색 구간엔 정답 자료가 없으니 둘 중 누가 맞는지 판단할 근거가 없어요.

[그림: 같은 자료에서 KNN K=1과 K=15의 결정 영역 비교] 무엇을 보나: 같은 합성 자료에 K=1(왼쪽)과 K=15(오른쪽)의 판단 영역을 칠했어요. 왼쪽은 경계가 구불구불하고 반대색 점 하나마다 작은 섬이 생겼고, 오른쪽은 매끄러워요. 🤔 왜? K=1은 가장 가까운 점 하나만 따라 하니 잡음 점까지 영역을 만들어 train 정확도가 1.00이에요(외운 것). K=15는 15표 다수결이라 튀는 점 하나로는 판정이 안 바뀌어 경계가 부드러워져요. 너무 크게 하면 반대로 소수 클래스의 작은 무리를 덮어 버려요.

## ⑤ 스케일링과 Pipeline

KNN은 거리로 판단하니까 단위가 결과를 바꿔요.

|  | 글자수 | 허용 도구수 | 클래스 |
|---|---|---|---|
| 학습 A | 1000 | 1 | 0 |
| 학습 B | 1400 | 4 | 1 |
| 새 요청 Q | 1100 | 4 | ? |

원래 단위로는 글자수 차이(100 vs 300)가 거리를 지배해서 Q는 A에 가까워요. train 기준으로 표준화하면 도구수 차이도 반영돼 B에 가까워져요.

🤔 왜 스케일이 KNN을 바꾸지? — 거리는 각 특성 차이를 제곱해서 더해요. 원래 단위로 Q–A = √(100² + 3²) ≈ 100, Q–B = √(300² + 0²) = 300 → 도구수 3개 차이는 글자수 100 앞에서 티도 안 나요. 표준화(글자수 평균 1200·표준편차 200, 도구수 평균 2.5·표준편차 1.5)를 하면 Q=(−0.5, 1), A=(−1, −1), B=(1, 1)이 되어 Q–A ≈ 2.06, Q–B = 1.5 → 이제 B가 더 가까워요. 단위가 큰 특성이 거리를 독차지하지 못하게 "같은 눈금"으로 맞추는 게 스케일링이에요.

[그림: 원래 단위와 표준화 후 A, B, Q의 위치와 거리 비교] 무엇을 보나: 두 축을 같은 눈금으로 그렸어요. 왼쪽(원래 단위)에선 도구수 1→4 차이가 너무 작아 세 점이 한 줄로 납작해지고 Q는 A에 붙어 있어요. 오른쪽(표준화)에선 세로 차이가 살아나 Q가 B에 더 가까워져요. 🤔 왜? 거리는 축마다 차이를 제곱해 더하는데, 글자수 차이(100~300)에 비하면 도구수 차이(3)는 티끌이에요. 표준화로 두 축을 "표준편차 몇 개만큼"이라는 같은 단위로 바꾸면 도구수도 거리에 제 몫을 하게 돼요.

🔧 KNN 스케일링 전후 비교 StandardScaler 명령어 사전에서 자세히 →

```
X = np.array([[1000, 1], [1400, 4]])
y = np.array([0, 1])
```

| 용어 | 무엇을 바꾸나 | 대표 예 |
|---|---|---|
| 표준화 | 각 특성에서 평균을 빼고 표준편차로 나눔 | StandardScaler |
| 최소–최대 스케일링 | train의 최소·최대로 0~1 범위로 | MinMaxScaler |
| 벡터 정규화 | 각 행의 길이를 1로 | Normalizer |
| 규제 (Regularization) | 큰 계수에 벌점을 주어 학습을 제어 (스케일링 아님!) | 로지스틱회귀의 C |

⚠️ 헷갈리기 쉬운 것

- StandardScaler는 데이터를 정규분포로 만들지 않아요. 위치와 크기만 바꾸는 선형 변환.

- MinMaxScaler로 바꿔도 새 값은 0~1 밖일 수 있어요 (train 범위 100~300에서 500 → 2.0).

- 스케일링은 단위의 과도한 영향을 줄이는 것이지, 점수 상승을 보장하지 않아요.

🔧 StandardScaler · MinMaxScaler 값 확인 명령어 사전에서 자세히 →

```
X = np.array([[100], [200], [300]])
a = StandardScaler().fit(X)
```

🤔 왜 이런 숫자가 나오지? — 표준화: 평균 200, 표준편차 약 81.6 → (100−200)/81.6 ≈ −1.225. MinMax: (x − 최소 100) / (최대 300 − 최소 100) → 500이면 400/200 = 2.0. 최소·최대는 train에서 정한 값을 고정해서 쓰니, 그 범위를 넘는 새 값은 0~1 밖으로 나가요. 둘 다 "빼고 나누기"만 하는 선형 변환이라 분포 모양(치우침 등)은 그대로예요.

[그림: 원래 값, StandardScaler, MinMaxScaler 결과를 숫자선에 비교] 무엇을 보나: train 100·200·300(보라)과 새 값 500(주황 마름모)을 세 가지 눈금에 올렸어요. 세 줄 모두 점들의 간격 비율과 순서는 그대로이고, 위치와 눈금 크기만 바뀌었어요. 🤔 왜? 두 스케일러 모두 "빼고 나누기"만 하는 선형 변환이라 모양을 바꾸지 않아요. MinMax의 0~1(초록 띠)은 train 최소·최대로 정한 범위라, 그보다 큰 새 값 500은 띠 밖의 2.0으로 나가요.

### 수치형·범주형은 각자의 길로

입력 X 4개 열→ 수치 2열 중앙값 대체 → 스케일링 범주 2열 최빈값 대체 → One-hot→ 합치기 8개 특성→ 모델 회귀 / 분류

Pipeline이 해 주는 일: fit(train) = 전처리 fit_transform → 모델 fit, predict(valid) = 전처리 transform만 → 모델 predict. 그래서 valid·test의 통계가 학습에 섞이지 않아요.

🤔 왜 train으로만 fit하지? — 중앙값·평균·표준편차·범주 목록도 "데이터에서 배운 값"이에요. valid·test까지 넣고 계산하면 미래 자료의 정보가 학습에 새어 들어가서, 실전보다 좋은 점수가 나와요. 실서비스에선 새 요청 하나만 들어오니, 그때 쓸 수 있는 건 train에서 배운 규칙뿐이에요. 수치·범주를 나누는 이유는 필요한 처리가 달라서예요: 숫자는 중앙값으로 채우고 눈금을 맞추고, 범주는 최빈값으로 채우고 0/1 열로 펼쳐야 거리·계수 계산에 쓸 수 있어요.

[그림: ] 무엇을 보나: 입력 4열이 수치 길(위)과 범주 길(아래)로 갈라져 각자 처리된 뒤 8열로 합쳐져 모델에 들어가요. 바깥 점선 상자 전체가 하나의 Pipeline 객체예요. 🤔 왜? 숫자는 "채우고 눈금 맞추기", 범주는 "채우고 0/1로 펼치기"가 필요해서 길이 달라요. 그리고 이 모든 상자를 한 객체로 묶어 두면 fit은 train에서만, predict는 transform만 하도록 자동으로 지켜져서 valid·test 정보가 새어 들어갈 틈이 없어요.

🔧 ColumnTransformer + Pipeline 명령어 사전에서 자세히 →

```
pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
model = build(KNeighborsClassifier(n_neighbors=5), scale="standard")
```

## ⑥ 하이퍼파라미터와 공정한 모델 비교

| 구분 | 예 | 어떻게 정하나 |
|---|---|---|
| 학습 파라미터 | 선형·로지스틱회귀의 계수, 절편 | train에 fit해서 자동으로 |
| 하이퍼파라미터 | K, weights, p, C | 사람이 후보를 정하고 valid로 비교 |
| 전처리 선택 | Standard / MinMax / 없음 | 실험 조건으로 적고 비교 |
| 판단 정책 | 실패 확률의 임계값 | 오류 비용과 valid를 보고 결정 |

🤔 왜 하이퍼파라미터는 valid로 고르지? — 계수는 train의 오차를 줄이도록 자동으로 정해지지만, K나 C는 "어떻게 배울지"를 정하는 설정이라 train만 보면 늘 train에 가장 바짝 붙는 값(예: K=1)이 이겨요. 그건 외우기일 뿐이니, 안 본 valid에서 비교해야 공정해요.

실험 전에 적어 둘 것: 고정 조건(같은 train 60/valid 20, 같은 4열, 중앙값·표준화 + 최빈값·원핫, weights=uniform · p=2) · 후보(K = 1, 3, 5, 9, 15) · 선택 규칙(분류는 valid F1 최대, 회귀는 valid RMSE 최소, 동점이면 작은 K).

🤔 왜 미리 적어 두지? — 결과를 본 뒤 후보나 규칙을 바꾸면, 우연히 좋게 나온 조합을 고르기 쉬워요. 같은 valid를 여러 번 들여다볼수록 선택 편향이 생겨요. 그리고 K를 바꿀 때 split까지 바꾸면 점수 차이가 "K 때문인지, 데이터 구성 때문인지" 분리할 수 없어요.

| K | 분류 train F1 | 분류 valid F1 | 회귀 valid RMSE (ms) |
|---|---|---|---|
| 1 | 1.000 | 0.400 | 944.68 |
| 3 | 0.571 | 0.545 | 795.22 |
| 5 | 0.462 | 0.200 | 744.99 |
| 9 | 0.519 | 0.000 | 707.65 |
| 15 | 0.364 | 0.000 | 641.79 |

💡 분류는 K=3, 회귀는 K=15. 같은 데이터라도 목표와 지표가 다르면 최적 K도 달라요. 분류의 K를 회귀에 복사하지 않아요. 그리고 RMSE 640과 F1 0.545는 서로 비교하거나 평균 낼 수 없는 값이에요.

🤔 왜 K마다 이런 패턴이 나오지?

- K=1 train 1.000 / valid 0.400 — train에선 자기 자신이 이웃이라 만점. 새 자료에선 이웃 하나의 잡음을 그대로 따라가서 떨어져요(과적합 신호).

- K=9·15 분류 valid 0.000 — 실패는 소수(valid 20건 중 6건)라, 이웃을 많이 모을수록 다수인 "성공" 표가 이겨서 실패를 한 번도 예측하지 못해요. 그래서 실패 F1이 0이 돼요.

- 회귀는 K가 클수록 RMSE↓ — 많은 이웃의 평균은 극단적인 값을 덜 내는 "무난한" 예측이 돼서 큰 오차가 줄어요. RMSE는 큰 오차에 민감하니 유리해져요. 목표가 다르면 최적 K도 달라지는 이유예요.

- 실제 곡선은 매끄럽거나 단조롭지 않을 수 있어요(train F1도 K=5 0.462 → K=9 0.519로 오르내림). K=3은 이 고정 실험 안의 잠정 선택이에요.

[그림: K별 분류 train·valid F1과 회귀 valid RMSE 그래프] 무엇을 보나: 수업 데이터로 K를 바꿔 가며 직접 계산한 결과예요(위 표 + K=7·11 추가). 왼쪽 분류는 회색 train이 K=1에서 1.0이지만 보라 valid는 K=3에서 최고, 오른쪽 회귀 RMSE는 K가 클수록 계속 내려가요. 🤔 왜? K=1은 자기 자신을 이웃으로 삼아 train만 만점이에요(두 선의 큰 격차 = 과적합 신호). 분류는 K가 커지면 다수인 "성공" 표에 묻혀 실패 F1이 0으로 떨어지고, 회귀는 많은 이웃을 평균할수록 극단 예측이 줄어 RMSE가 낮아져요. 그래서 목표마다 최적 K가 달라요. 곡선이 들쭉날쭉한 건 valid가 20건뿐이라서예요.

### 지표로 진단하기

| 이렇게 나오면 | 의심할 것 |
|---|---|
| train은 좋은데 valid는 크게 나쁨 (K=1: 1.000 vs 0.400) | 과적합 |
| 설정을 조금만 바꿔도 valid가 크게 흔들림 | 표본이 작음 (valid 20건에서 1건 = Accuracy 0.05) |
| 점수가 낮음 | 곧 과적합은 아님 → 누수·라벨·분할, 신호 부족, 분포 차이 먼저 확인 |
| 새 모델이 Dummy와 비슷함 | 입력에 쓸 만한 신호가 부족할 수 있음 (먼저 누수·라벨·분할 문제부터 확인) |

🤔 왜 낮은 점수 = 과적합이 아니지? — 과적합은 "train은 좋은데 valid만 나쁜" 상태예요. train도 valid도 둘 다 나쁘면 외운 게 아니라 배울 신호가 부족하거나 분포가 다르거나 표본이 적은 쪽을 먼저 의심해요. valid 20건에선 1건만 달라져도 Accuracy가 0.05 움직이니 작은 차이에 확정적인 의미를 두지 않아요.

🔧 후보 반복 비교 · 기준선 명령어 사전에서 자세히 →

```
model = build(KNeighborsClassifier(n_neighbors=k))   # 매번 새 Pipeline
va = f1_score(y_valid, model.predict(X_valid), zero_division=0)
```

[그림: Dummy, Logistic, KNN3의 valid F1과 FP, FN 건수 비교 막대] 무엇을 보나: 왼쪽은 주 지표(실패 F1), 오른쪽은 각 모델이 낸 오류 수예요. Dummy는 헛경고(FP)가 0이지만 실패 6건을 전부 놓쳐요(FN 6). 🤔 왜? Dummy는 "전부 성공"만 말하니 경고를 안 해서 FP가 0이고, Accuracy도 0.70으로 그럴듯해요. 하지만 F1이 0이라 막대가 없죠. KNN3는 F1이 가장 높지만 여전히 실패 3건을 놓쳐요 — 그래서 점수와 함께 오류 수도 적어요.

## ⑦ 자동 검사 · 저장과 복원

고른 Pipeline이 "에러 없이 돌았다"만으로는 부족해요. 예측이 학습 통계를 바꾸지 않는지, 입력 열이 약속대로인지 확인하고, 저장한 뒤 다시 불러도 같은 예측이 나오는지 대조해요.

🤔 왜 검사하지? — Pipeline 안의 imputer는 train 중앙값을 statistics_에 기억해요. 만약 predict 때 이 값이 바뀐다면 valid 정보가 규칙에 섞였다는 뜻(누수)이에요. 저장·복원 대조는 "내 컴퓨터에서 나온 결과를 나중에도, 다른 사람도 똑같이 재현할 수 있나"를 확인하는 거예요.

[그림: ] 무엇을 보나: 두 줄 모두 "실행 전 값을 따로 떼어 두고 → 무언가 한 뒤 → 다시 비교"하는 같은 모양이에요. 🤔 왜? "에러 없이 돌았다"는 아무것도 증명하지 않아요. 바뀌면 안 되는 값(학습 통계, 예측 결과)을 전·후로 직접 대조해야 누수나 저장 실수를 숫자로 잡을 수 있어요.

🔧 학습 통계 불변 검사 명령어 사전에서 자세히 →

```
before = imp.statistics_.copy()                 # 중앙값을 따로 복사해 보관
print("통계 불변:", np.array_equal(before, after))                    # True
```

🔧 저장 · 복원 후 예측 대조 명령어 사전에서 자세히 →

```
joblib.dump(model, path)                    # 전처리 + 모델을 통째로 저장
restored = joblib.load(path)                # 본인이 만든 파일만 읽기
```

## 원본 자료

- 강의: 20261026월_머신러닝_회귀_분류_KNN_모델비교_Day6 (2).pdf — ① 선형회귀 10~14쪽 · ② 회귀 지표 15~23쪽 · ③ 로지스틱·분류 지표 24~35쪽 · ④ KNN 36~43쪽 · ⑤ 스케일링·Pipeline 44~54쪽 · ⑥ 비교 55~69쪽 · ⑦ 검사·저장 72~73, 77쪽

- 공식 문서: scikit-learn · Linear models · Nearest neighbors · Metrics and scoring · Common pitfalls

## 🔧 Day6 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day6)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install numpy pandas scikit-learn |
| ① 선형회귀 | LinearRegression(), .fit(), .coef_, .intercept_ |
| ② 회귀 지표 · 기준선 | mean_absolute_error, root_mean_squared_error, r2_score, DummyRegressor |
| ③ 로지스틱회귀 · 분류 지표 | LogisticRegression(C=, max_iter=), predict_proba, confusion_matrix, f1_score |
| ④ KNN | KNeighborsClassifier / KNeighborsRegressor(n_neighbors=, weights=, p=), kneighbors() |
| ⑤ 스케일링 · Pipeline | StandardScaler, MinMaxScaler, ColumnTransformer, Pipeline |
| ⑥ 하이퍼파라미터 · 공정한 비교 | for k in [...] 반복 비교, DummyClassifier, zero_division=0 |
| ⑦ 자동 검사 · 저장/복원 | statistics_, np.array_equal, joblib.dump / joblib.load |

### LinearRegression()

```
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5]])     # 공부시간 (2차원: 행 5 × 열 1)
y = np.array([52, 61, 69, 80, 89])          # 점수

model = LinearRegression()
model.fit(X, y)
print("기울기:", model.coef_.round(1))        # [9.3]
print("절편:", round(model.intercept_, 1))    # 42.3
print("6시간:", model.predict([[6]]).round(1)) # [98.1]
```

▶ 실제로 돌려 보면 노트의 공부시간 예제 5행

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| X.shape | (5, 1) | 행 5(사례) × 열 1(특성) — 2차원 |
| model.coef_ | [9.30] | 1시간당 약 9.3점 — 특성마다 하나라 배열 |
| model.intercept_ | 42.30 | x=0일 때 높이 42.3 |
| model.predict([[6]]) | [98.10] | 9.3 × 6 + 42.3 |
| model.predict([6]) | ValueError: Expected 2D array, got 1D array instead | 1차원을 넣으면 에러 — [[6]]이어야 함 |
| hasattr(LinearRegression(), "coef_") | False | fit 전엔 _ 속성이 없음 |

| 인자 / 속성 | 뜻 | 주의 |
|---|---|---|
| fit(X, y) | 계수와 절편을 학습 | X는 항상 2차원 (행 × 열). 왜? 행 = 사례, 열 = 특성이라는 약속이라 특성이 1개여도 "열 1개짜리 표"로 줘야 해요 |
| coef_ / intercept_ | 학습된 기울기 / 절편 | 끝에 _가 붙은 건 "학습 후 생기는 값" (fit 전에는 없음) |
| predict([[6]]) | 새 입력 1행 × 1열을 예측 | [6]이 아니라 [[6]]. 6시간 → 9.3×6 + 42.3 ≈ 98.1 |

🤔 왜 기울기가 9.3이지? — 점수 [52, 61, 69, 80, 89]는 1시간 늘 때마다 대략 9~11점씩 올라요. 잔차제곱합이 가장 작아지도록 평균적인 증가폭을 고르면 9.3, 그 직선이 x=0에서 만나는 높이가 42.3이에요.

### 회귀 지표

```
from sklearn.metrics import (mean_absolute_error,
                             root_mean_squared_error, r2_score)
from sklearn.dummy import DummyRegressor

y_true = [100, 200, 300]
y_pred = [120, 180, 360]
print("MAE:", round(mean_absolute_error(y_true, y_pred), 2))       # 33.33
print("RMSE:", round(root_mean_squared_error(y_true, y_pred), 2))  # 38.3
print("R2:", round(r2_score(y_true, y_pred), 2))                   # 0.78

base = DummyRegressor(strategy="mean").fit(X_train, yr_train)      # 기준선
```

▶ 실제로 돌려 보면 작은 배열 3개 + 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| mean_absolute_error(y_true, y_pred) | 33.33 | (20+20+60)/3 |
| root_mean_squared_error(y_true, y_pred) | 38.30 | 큰 오차 60이 제곱돼 MAE보다 큼 |
| r2_score(y_true, y_pred) | 0.78 | 평균선보다 오차 제곱합이 78% 작음 |
| r2_score(y_pred, y_true) (순서 바꿈) | 0.86 | R²는 첫 인자의 평균이 기준이라 값이 달라짐 |
| Linear: MAE · RMSE · R² (valid) | 551.8 · 640.0 · -0.097 | R²가 음수 — 입력만으론 소요시간 신호가 약함 |
| DummyRegressor(mean): 같은 지표 | 541.3 · 658.7 · -0.162 | train 평균만 말해도 MAE는 Linear보다 작음 |
| base.predict(X_valid)[:3] | [1235.2, 1235.2, 1235.2] | 모두 train 평균 1235.2ms |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| y_true, y_pred | 순서가 맞는 실제값 / 예측값 | 두 인자를 바꿔 넣으면 R²가 달라짐 — R²는 y_true의 평균을 기준으로 삼기 때문 (MAE·RMSE는 차이의 크기만 보니 같음) |
| strategy="mean" | train y 평균을 예측 | "median"이면 중앙값 — 중앙값은 극단값에 덜 끌려가서 MAE 기준선으로 더 맞음 |

작은 배열이 아니라 공통 자료(전처리 + 선형회귀)로 valid 20건을 예측하면 결과는 MAE 551.8 · RMSE 640.0 · R² −0.097이에요(reg.fit(X_train, yr_train) → reg.predict(X_valid), fit은 train에만).

⚠️ 큰 오류는 지우지 말고 먼저 입력·단위·누수 여부를 확인해요. valid를 보고 임의로 행을 삭제하면 비교가 무너져요. 잔차(실제 − 예측)가 양수면 실제가 더 오래 걸린 것, 음수면 실제보다 오래 걸린다고 예측한 것이에요.

🤔 왜 지우면 안 되지? — valid를 보고 "틀린 행"을 빼면 점수는 좋아지지만, 그건 모델이 나아진 게 아니라 시험지를 고친 것이에요. 실서비스에서도 그런 요청은 또 들어오니까, 왜 틀렸는지(입력 오류? 단위? 누수?)를 먼저 밝혀야 해요.

### LogisticRegression · predict_proba

```
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

clf = build(LogisticRegression(C=1.0, max_iter=1000))   # build = 전처리+모델 Pipeline (⑤ 스케일링과 Pipeline 참고)
clf.fit(X_train, y_train)

classes = clf.named_steps["model"].classes_             # [0, 1]
pos = list(classes).index(1)                            # 실패(1)가 몇 번째 열?
p = clf.predict_proba(X_valid)[:, pos]                  # 실패 확률
pred = clf.predict(X_valid)                             # 0/1 클래스

print(confusion_matrix(y_valid, pred, labels=[0, 1]))   # [[10 4] [4 2]]
print("Accuracy:", accuracy_score(y_valid, pred))
print("F1:", f1_score(y_valid, pred))
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| clf.named_steps["model"].classes_ | [0, 1] | predict_proba 열 순서 → 실패(1)는 1번 열 |
| clf.predict_proba(X_valid).shape | (20, 2) | 행마다 [P(0), P(1)], 합은 1 |
| p[:4] (실패 확률) | [0.141, 0.185, 0.601, 0.745] | 확신 정도가 서로 다름 |
| clf.predict(X_valid)[:4] | [0, 0, 1, 1] | 확률을 0.5에서 자른 결과 |
| confusion_matrix(y_valid, pred, labels=[0, 1]) | [[10, 4], [4, 2]] | [[TN FP] [FN TP]] = [[10 4] [4 2]] |
| accuracy_score · f1_score | 0.600 · 0.333 | Accuracy는 Dummy(0.70)보다도 낮음 |
| C=0.01일 때 \|계수\| 합 vs C=1 | 0.205 vs 3.496 | C가 작을수록 규제↑ → 계수가 눌림 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| C=1.0 | 규제 강도의 역수 | 작게 → 규제↑(계수가 작게 눌려 단순한 모델, 과적합↓·과소적합 위험↑), 크게 → 규제↓(train에 더 바짝 맞춤). "역수"라서 C가 작을수록 규제가 세요 |
| max_iter=1000 | 최적화 반복 상한 | 너무 작으면 답을 찾다 멈춰 수렴 경고가 남. 크게 해도 반복 횟수 상한일 뿐이라 일반화 성능이 좋아진다는 보장은 없음 |
| classes_ | predict_proba 열의 클래스 순서 | 열 순서는 classes_를 따르니, 두 번째 열이 항상 원하는 클래스라고 가정하지 말고 index(1)로 찾기 |
| labels=[0, 1] | 혼동행렬 행·열 순서 고정 | 안 주면 데이터에 나타난 값으로 정해져서, 한 클래스가 없을 때 표 모양이 달라질 수 있음 |
| f1_score(..) | F1 계산 | 기본 양성 클래스는 1(=실패) — 그래서 실패 기준 F1이 나옴 |

🤔 왜 predict와 predict_proba를 따로 보지? — 앞 4개 확률이 [0.141, 0.185, 0.601, 0.745]일 때 predict는 [0, 0, 1, 1]이에요. 0.601과 0.745는 둘 다 "1"이지만 확신 정도가 달라요. 확률은 정보가 더 많고, 클래스는 그 확률을 0.5에서 잘라 버린 결과라서 임계값을 바꾸고 싶을 땐 확률이 필요해요.

### 임계값 바꾸기

```
truth = np.array([0, 0, 0, 1, 1, 1])
p     = np.array([0.10, 0.40, 0.70, 0.35, 0.60, 0.90])
for threshold in [0.5, 0.3]:
    pred = (p >= threshold).astype(int)                 # True/False → 1/0
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    print(f"t={threshold}: FP={fp}, FN={fn}, TP={tp}")
# t=0.5: FP=1, FN=1, TP=2
# t=0.3: FP=2, FN=0, TP=3   ← 놓침은 줄고 헛경고는 늘었다
```

▶ 실제로 돌려 보면 노트의 6건 (실제 3:3)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| pred = (p >= 0.5).astype(int) | [0, 0, 1, 0, 1, 1] → FP=1, FN=1, TP=2 | 기본 선 |
| pred = (p >= 0.3).astype(int) | [0, 1, 1, 1, 1, 1] → FP=2, FN=0, TP=3 | 선을 내리면 FN↓ · FP↑ |
| pred = (p >= 0.8).astype(int) | [0, 0, 0, 0, 0, 1] → FP=0, FN=2, TP=1 | 선을 올리면 FP↓ · FN↑ |

💡 임계값을 낮추면 FN(놓침)은 줄지만 FP(오탐)는 늘 수 있어요. 공짜 개선은 없어요. 임계값도 valid에서만 정해요.

🤔 왜 하나가 줄면 하나가 늘지? — 임계값은 "이 확률 이상이면 실패라고 부른다"는 선이에요. 선을 0.5 → 0.3으로 내리면 실제 실패였던 0.35가 새로 잡혀서 FN이 줄지만, 실제 성공인데 0.40이던 것도 같이 넘어와서 FP가 늘어요. 모델은 그대로고 판단 규칙만 바뀐 거예요. 그래서 어느 쪽 오류가 더 비싼지(놓침 vs 헛경고)를 먼저 정해야 해요.

이 코드의 6건으로 임계값을 직접 움직여 보는 슬라이더는 Day6 ③ 로지스틱회귀 본문에 있어요.

### KNeighborsClassifier · KNeighborsRegressor

```
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

X = np.array([[1], [2], [4], [7], [9]])
knn = KNeighborsClassifier(n_neighbors=3).fit(X, [0, 0, 1, 1, 1])
distance, index = knn.kneighbors([[3.2]])     # 이웃까지 거리, 학습 행 번호
print(X[index[0], 0].tolist(), distance[0].round(1).tolist())   # [4, 2, 1] [0.8, 1.2, 2.2]
print(knn.predict([[3.2]]))                    # [0]

reg = KNeighborsRegressor(n_neighbors=3).fit(X, [10, 20, 40, 70, 90])
print(reg.predict([[3.2]]))                    # [23.33]  = (40+20+10)/3
w = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(X, [10, 20, 40, 70, 90])
print(w.predict([[3.2]]))                      # [28.06]  가까운 40에 더 큰 가중치
```

▶ 실제로 돌려 보면 노트의 x=[1, 2, 4, 7, 9] 예제

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| knn.kneighbors([[3.2]]) | 거리 [0.8, 1.2, 2.2], 행 [2, 1, 0] | x=4, 2, 1 순서로 가까움 |
| knn.predict([[3.2]]) | [0] | 이웃 정답 1·0·0 → 다수결 0 |
| knn.predict_proba([[3.2]]) | [0.667, 0.333] | 3표 중 0이 2표 |
| KNeighborsRegressor(3).predict([[3.2]]) | [23.33] | (40+20+10)/3 |
| weights="distance" 일 때 | [28.06] | 가까운 40의 목소리가 커짐 |
| x=8: LinearRegression vs KNN(K=2) | 80.0 vs 35.0 | 직선은 연장, KNN은 본 정답 범위에 갇힘 |
| KNeighborsClassifier(n_neighbors=6).fit(5행).predict | ValueError: Expected n_neighbors <= n_samples_fit | K는 학습 행 수 이하여야 함 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| n_neighbors (K) | 참고할 이웃 수 | 작게 → 이웃 한두 개의 잡음에 그대로 휘둘림, 크게 → 멀리 있는 사례까지 투표해서 소수 클래스의 국소 패턴을 놓칠 수 있음. train 행 수 이하여야 함(없는 이웃은 못 고르니까) |
| weights | "uniform"(똑같이, 기본값) / "distance"(거리의 역수만큼 가중) | distance면 가까운 이웃 말이 더 커짐 → 회귀 예측이 23.33 → 28.06. 항상 더 낫지는 않음 |
| p | 2 = 유클리드(직선, 기본값) / 1 = 맨해튼(가로+세로) | 거리 재는 법이 바뀌니 "누가 가장 가까운가"가 달라질 수 있음 (기본 metric=minkowski일 때) |
| kneighbors() | 선택된 이웃의 거리·학습 행 인덱스 반환 | 왜 그 예측이 나왔는지 설명할 때 |

🤔 왜 거리 가중이면 28.06이지? — 이웃 40·20·10의 거리가 0.8·1.2·2.2라 가중치는 1/거리 = 1.25·0.83·0.45예요. (40×1.25 + 20×0.83 + 10×0.45) ÷ (1.25+0.83+0.45) ≈ 71.2 ÷ 2.54 ≈ 28.06. 가장 가까운 40의 목소리가 커져서 균일 평균 23.33보다 올라가요. 회귀 예측의 단위는 정답 y의 단위를 그대로 따라요.

⚠️ 범위 밖 예측 — 학습 x=[1,2,3,4], y=[10,20,30,40]에서 x=8을 물으면 선형회귀는 직선을 연장해 80, KNN(K=2)은 가까운 3·4의 평균 35. 어느 쪽도 외삽 정답을 보장하지 않아요.

```
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
X = [[1], [2], [3], [4]]
y = [10, 20, 30, 40]
linear = LinearRegression().fit(X, y)
knn = KNeighborsRegressor(n_neighbors=2).fit(X, y)
print("x=8 선형:", linear.predict([[8]]).round(1).tolist())   # [80.0]
print("x=8 KNN:", knn.predict([[8]]).round(1).tolist())       # [35.0]
```

🤔 왜 둘이 이렇게 다르지? — 선형회귀는 찾은 직선(y = 10x)을 범위 밖까지 그대로 연장해요. KNN은 이미 본 정답들의 평균만 낼 수 있어서, 아무리 먼 x를 넣어도 학습 정답 범위(10~40) 밖의 값을 절대 못 내요. 어느 쪽이 맞는지는 데이터가 없는 구간이라 알 수 없어요.

⚠️ K=1이면 train 점수가 거의 완벽하게 나와요. 그래서 꼭 valid로 확인해요. 자료가 커지면 예측할 때마다 이웃을 찾느라 메모리·시간도 커지고, 선택한 모든 특성이 거리에 참여하니 불필요한 특성·단위·범주 가중에 주의해요.

🤔 왜 K=1의 train 점수는 믿으면 안 되지? — train 자료로 예측하면 가장 가까운 이웃이 자기 자신(거리 0)이라 자기 정답을 그대로 베껴요. 시험 문제와 정답지가 같은 셈이라 1.000이 나와도 아무것도 증명하지 않아요.

### ColumnTransformer + Pipeline

```
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

NUM = ["input_chars", "allowed_tools"]       # 수치형
CAT = ["task_type", "priority"]              # 범주형

from sklearn.base import clone
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}

def build(estimator, scale="standard"):
    scaler = SCALERS[scale]() if scale in SCALERS else "passthrough"   # "none" → 스케일링 안 함
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", scaler)])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("pre", pre), ("model", clone(estimator))])   # clone = 학습 안 된 새 복사본

model = build(KNeighborsClassifier(n_neighbors=5), scale="standard")
model.fit(X_train, y_train)          # 전처리도 train으로만 학습
pred = model.predict(X_valid)        # valid는 transform → predict만
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| X_train.isna().sum().sum() | 12 | 원본 train에 결측이 있음 |
| model["pre"].transform(X_train).shape | (60, 8) | 수치 2 + 원핫 6 = 8열 |
| model["pre"].transform(X_valid).shape | (20, 8) | valid도 같은 8열 |
| np.isnan(Z_train).sum() | 0 | 결측이 train 중앙값·최빈값으로 채워짐 |
| get_feature_names_out()[:3] | ['num__input_chars', 'num__allowed_tools', 'cat__task_type_qa'] | num__/cat__ 접두어로 출처 표시 |
| Z_train[:, :2].mean(0) · std(0) | [0.00, 0.00] · [1.00, 1.00] | standard: train 기준 평균 0 · 표준편차 1 |
| K=5 valid F1: none · standard · minmax | 0.400 · 0.200 · 0.400 | 표준화했는데 오히려 낮음 — 노트 설명대로 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| strategy="median" | 수치 결측을 train 중앙값으로 | "mean"이면 평균 (이상치에 더 흔들림) |
| strategy="most_frequent" | 범주 결측을 최빈값으로 | 범주는 평균·중앙값을 낼 수 없으니 가장 흔한 값으로 채움 |
| handle_unknown="ignore" | train에 없던 새 범주는 해당 블록 전부 0 | 기본값("error")이면 새 범주에서 에러 — 실서비스에선 처음 보는 범주가 언제든 올 수 있어서 ignore로 둠 |
| sparse_output=False | 결과를 보기 쉬운 일반(밀집) 배열로 | 교육용 작은 배열이라 False. 큰 데이터면 True(희소 행렬)가 메모리 절약 |
| scale | "standard" / "minmax" / "none"("passthrough" = 그대로 통과) | 스케일러만 바꿔서 공정하게 비교 (K=5 고정 시 valid F1: none 0.400 · standard 0.200 · minmax 0.400) |

변환 결과 shape는 train (60, 8), valid (20, 8) — 수치 2열 + 범주 2열을 원핫으로 펼친 6열 = 8개 특성이에요. valid도 같은 8열 구조를 유지해야 모델이 받을 수 있어요.

🤔 왜 표준화했는데 F1이 오히려 떨어졌지? — 스케일링은 "단위가 큰 특성의 독주를 막는 것"이지 점수 상승 약속이 아니에요. 이 자료에선 원핫 범주(0/1)도 거리에 참여하는데, 수치 열의 눈금이 바뀌면 수치 vs 범주의 영향력 균형이 바뀌어 이웃이 달라져요. 게다가 valid가 20건이라 이웃 몇 개만 바뀌어도 점수가 크게 흔들려요. 그래서 유리한 결과만 남기지 말고 바꾼 조건과 모든 후보 점수를 기록해요. (다음 K 실험은 standard를 고정 — 그게 전체 조합의 최적이라는 뜻은 아님)

⚠️ Pipeline을 써도 fit에 valid·test를 넣으면 누수예요. 누수 방지는 객체 이름이 아니라 학습에 어떤 데이터를 넣느냐의 문제예요. 후보마다 새 Pipeline을 만들고, 학습된 객체를 다른 실험에 재사용하지 않아요.

### 후보 반복 비교 · 기준선

```
from sklearn.dummy import DummyClassifier
import pandas as pd

for k in [1, 3, 5, 9, 15]:                              # 후보는 미리 정해 둔다
    model = build(KNeighborsClassifier(n_neighbors=k))   # 매번 새 Pipeline
    model.fit(X_train, y_train)
    tr = f1_score(y_train, model.predict(X_train), zero_division=0)
    va = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    print(f"k={k:2d}: train={tr:.3f}, valid={va:.3f}")

candidates = {"Dummy": DummyClassifier(strategy="most_frequent"),
              "Logistic": LogisticRegression(C=1.0, max_iter=1000),
              "KNN_k3": KNeighborsClassifier(n_neighbors=3)}
rows = []
for name, est in candidates.items():
    pred = build(est).fit(X_train, y_train).predict(X_valid)
    rows.append({"model": name, "F1": f1_score(y_valid, pred, zero_division=0)})
print(pd.DataFrame(rows))

# 회귀도 같은 방식: DummyRegressor(strategy="mean") · LinearRegression() · KNeighborsRegressor(n_neighbors=15)
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| for k in [1, 3, 5, 9, 15] → train/valid F1 | k=1: 1.000/0.400 · k=3: 0.571/0.545 · k=5: 0.462/0.200 | K=1은 train 1.000 — 자기 자신이 이웃 |
| (계속) | k=9: 0.519/0.000 · k=15: 0.364/0.000 | K가 크면 실패를 한 번도 안 불러 F1 0 |
| Dummy: Accuracy · F1 · FP/FN | 0.70 · 0.000 · 0/6 | 다 성공이라 해도 Accuracy 0.70 |
| Logistic: 같은 지표 | 0.60 · 0.333 · 4/4 | 헛경고 4 · 놓침 4 |
| KNN_k3: 같은 지표 | 0.75 · 0.545 · 2/3 | F1 최고지만 실패 3건을 놓침 |
| f1_score(y_valid, [0]*20) (zero_division 없이) | 0.0 · 경고 없음 | F1 = 2TP/(2TP+FP+FN)의 분모는 FN=6이라 0이 아님 → 이 버전(sklearn 1.8)에선 F1 경고가 안 떠요 |
| precision_score(y_valid, [0]*20) | 0.0 · 경고 UndefinedMetricWarning | Precision 분모(TP+FP)는 진짜 0 → 여기서 경고. zero_division=0은 이 규칙을 명시하는 것 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| zero_division=0 | 양성 예측이 0건이라 Precision 분모가 0이면 0으로 기록 | 안 주면 경고가 뜸. K=9·15처럼 실패를 한 번도 예측하지 않을 때 필요 |
| strategy="most_frequent" | 가장 많은 클래스만 예측하는 기준선 | Accuracy 0.70이지만 F1 0 — "아무것도 안 배워도 이만큼은 나온다"는 바닥선 |
| build(est) 매번 새로 | 후보마다 전처리·모델 객체를 새로 생성 | 학습된 객체를 재사용하면 앞 실험의 fitted 상태가 덮어써지거나 섞여서 비교가 무너짐 |

| 분류 모델 | Accuracy | Precision | Recall | F1 | FP / FN |
|---|---|---|---|---|---|
| Dummy | 0.700 | 0.000 | 0.000 | 0.000 | 0 / 6 |
| Logistic · C=1 | 0.600 | 0.333 | 0.333 | 0.333 | 4 / 4 |
| KNN · K=3 | 0.750 | 0.600 | 0.500 | 0.545 | 2 / 3 |

| 회귀 모델 | MAE (ms) | RMSE (ms) | R² |
|---|---|---|---|
| Dummy · train 평균 | 541.26 | 658.75 | −0.162 |
| LinearRegression | 551.80 | 640.00 | −0.097 |
| KNN · K=15 | 530.06 | 641.79 | −0.103 |

⚠️ RMSE는 Linear, MAE는 KNN15가 작아요. 주 지표를 먼저 정해야 하는 이유예요. 1.79ms 차이는 과장하지 않고, valid 최고 점수는 "후보 선택에 쓴 점수"라 최종 성능이 아니에요. test는 모델·평가 정책을 확정한 뒤 한 번만.

🤔 왜 valid 최고 점수를 최종 성능이라 부르면 안 되지? — 여러 후보 중 가장 높게 나온 것을 골랐으니, 그 점수엔 "운 좋게 이 20건에 잘 맞은 몫"이 섞여 있어요(선택 편향). 그래서 고르는 데 쓰지 않은 test로 마지막에 한 번 확인해요. 또 KNN3도 실패 6건 중 3건을 놓치니, "잠정 선택"이지 운영 승인이 아니에요.

💡 재현성 — seed만 적으면 끝이 아니에요. 원본 데이터 버전과 행 순서, 실제 분할 ID, 바꾼 조건, train·valid 점수와 오류 수를 함께 기록해요. 모델을 트리로 바꿔도 원본·입력 계약·분할 ID·선택 지표·test 보관 정책은 그대로 남겨요.

### 학습 통계 불변 검사

```
model = build(KNeighborsClassifier(n_neighbors=3))
model.fit(X_train, y_train)
imp = model["pre"].named_transformers_["num"]["imputer"]
before = imp.statistics_.copy()                 # 중앙값을 따로 복사해 보관
model.predict(X_valid)
after = imp.statistics_
print("통계 불변:", np.array_equal(before, after))                    # True
print("입력 열 일치:", list(X_train.columns) == FEATURES)              # True
print("금지 열 없음:", not set(FORBIDDEN) & set(X_train.columns))      # True
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| imp.statistics_ | [1315.0, 3.0] | train 중앙값: 글자수 · 도구 수 (valid 중앙값 1123.0과 다름) |
| np.array_equal(before, after) (predict 후) | True | predict는 다시 학습하지 않음 |
| before is imp.statistics_ · alias is imp.statistics_ | False · True | .copy() 없이 담으면 같은 배열을 가리킴 |
| list(X_train.columns) == FEATURES | True | 입력 4열 순서까지 일치 |
| not set(FORBIDDEN) & set(X_train.columns) | True | 정답·결과 열이 입력에 없음 |

### 저장 · 복원 후 예측 대조

```
from pathlib import Path
import tempfile, joblib

model = build(KNeighborsClassifier(n_neighbors=3))
model.fit(X_train, y_train)
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "selected_pipeline.joblib"
    joblib.dump(model, path)                    # 전처리 + 모델을 통째로 저장
    restored = joblib.load(path)                # 본인이 만든 파일만 읽기
    before = model.predict(X_valid)
    after = restored.predict(X_valid)
    print("복원 예측 동일:", np.array_equal(before, after))   # True
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 · KNN K=3

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| joblib.dump(model, path) | 파일 약 13 KB | KNN은 train 60행을 통째로 들고 있음 |
| type(restored).__name__, list(restored.named_steps) | Pipeline, ['pre', 'model'] | 전처리 + 모델이 함께 복원 |
| np.array_equal(model.predict(X_valid), restored.predict(X_valid)) | True | 같은 환경에서 예측 20개 일치 |

| 인자 / 함수 | 뜻 | 바꾸면? |
|---|---|---|
| statistics_.copy() | 학습된 중앙값을 별도로 보관 | .copy() 없이 담으면 같은 배열을 가리켜서, 값이 제자리에서 바뀌면 before도 같이 바뀌어 비교가 의미 없어짐 |
| np.array_equal(a, b) | 두 배열의 모양·값이 모두 같은지 | 하나라도 다르면 False |
| joblib.dump(model, path) | 전처리와 모델을 함께 저장 | 모델만 저장하면 새 입력에 같은 전처리를 다시 못 붙임 |
| joblib.load(path) | 저장한 Pipeline 복원 | 남이 준 파일은 load 금지 — 불러오는 순간 안의 코드가 실행될 수 있어요. 같은 환경(버전)에서 대조 |
| tempfile.TemporaryDirectory() | with 블록이 끝나면 지워지는 임시 폴더 | 시연·검사용. 실제로 보관하려면 고정 경로에 저장 |

💡 경계 상황도 미리 정해 두고 확인해요: 수치 결측·새 범주가 와도 8개 특성 유지, 필수 열이 빠지면 학습/예측 전에 명확한 오류(임의 값 생성 금지), K > train 행 수면 실행 전 설정 오류(1 ≤ K ≤ 60). "에러가 안 났다"가 아니라 기대 출력·단위·행 수를 먼저 정하는 게 테스트예요.

### KNN 스케일링 전후 비교 StandardScaler

```
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
X = np.array([[1000, 1], [1400, 4]])
y = np.array([0, 1])
q = np.array([[1100, 4]])
raw = KNeighborsClassifier(n_neighbors=1).fit(X, y)
s = StandardScaler().fit(X)                       # train 평균·표준편차 저장
scaled = KNeighborsClassifier(n_neighbors=1)
scaled.fit(s.transform(X), y)
print("원단위:", raw.predict(q).tolist())          # [0]
print("표준화:", scaled.predict(s.transform(q)).tolist())   # [1]  같은 규칙으로 q도 변환
```

▶ 실제로 돌려 보면 노트의 A·B·Q 세 요청 (글자수, 도구수)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| raw.predict([[1100, 4]]) | [0] | 원래 단위: 글자수 차이가 거리를 지배 → A(0) |
| s.mean_, s.scale_ | [1200.0, 2.5], [200.0, 1.5] | train 두 행의 평균·표준편차를 기억 |
| s.transform([[1100, 4]]) | [[-0.5 1. ]] | Q도 같은 규칙으로 변환 |
| scaled.predict(s.transform(q)) | [1] | 표준화 후엔 도구수 차이도 반영 → B(1) |

### StandardScaler · MinMaxScaler 값 확인

```
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
X = np.array([[100], [200], [300]])
a = StandardScaler().fit(X)
b = MinMaxScaler().fit(X)
print("표준화:", a.transform(X).ravel().round(3).tolist())   # [-1.225, 0.0, 1.225]
print("MinMax:", b.transform(X).ravel().tolist())            # [0.0, 0.5, 1.0]
print("새값500:", b.transform([[500]]).ravel().tolist())     # [2.0]
```

▶ 실제로 돌려 보면 train 값 100·200·300

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| StandardScaler().fit(X).transform(X) | [-1.225, 0.0, 1.225] | 평균 200, 표준편차 81.6 |
| MinMaxScaler().fit(X).transform(X) | [0.0, 0.5, 1.0] | 최소 100 → 0, 최대 300 → 1 |
| b.transform([[500]]) | [2.0] | train 범위 밖이라 1을 넘어요 |
| a.transform([[500]]) | [3.674] | 표준화도 범위 제한이 없어요 |

## 📖 Day6 용어 (76개)
- **머신러닝** (Machine Learning (기계학습)): 사람이 규칙을 직접 짜는 대신, 데이터(입력과 정답)를 보여 주고 컴퓨터가 규칙을 스스로 찾게 하는 방법. 그렇게 학습된 결과물을 '모델'이라고 불러요.
- **회귀** (Regression (회귀)): 정답이 연속적인 숫자인 예측 문제. 예: 이 요청은 몇 ms 걸릴까? 지표는 MAE·RMSE처럼 '얼마나 빗나갔나'를 재요.
- **분류** (Classification (분류)): 정답이 정해진 종류(클래스) 중 하나인 예측 문제. 예: 이 요청은 실패(1)할까 성공(0)할까? 지표는 F1처럼 '맞혔나·어떤 오류인가'를 세요.
- **특성** (Feature (특성, 입력 열)): 모델에 넣는 입력 정보 한 가지. 표에서 열 하나가 특성 하나예요(예: 글자수, 허용 도구수). 행 = 사례, 열 = 특성.
- **입력 X** (Input X (입력 특성 표)): 모델이 보고 판단하는 특성들의 표(2차원: 행 × 열). 예측 시점에 실제로 알 수 있는 값만 넣어야 해요.
- **정답 y** (Target y (정답, 타깃)): 모델이 맞혀야 하는 값. 회귀에선 숫자(elapsed_ms), 분류에선 클래스(failed: 0/1).
- **라벨** (Label (정답 라벨)): 사람이 미리 붙여 둔 정답 값. 분류에서는 각 사례가 어느 클래스인지(예: 0=성공, 1=실패)를 뜻해요.
- **클래스** (Class (클래스, 범주 정답)): 분류 문제에서 고를 수 있는 정답의 종류. 이 수업에선 0(성공)과 1(실패) 두 클래스예요. classes_는 모델이 기억한 클래스 순서.
- **fit** (fit (학습)): 데이터를 보고 모델(또는 전처리)이 필요한 값을 배우는 단계. 예: 선형회귀는 계수·절편을, 스케일러는 평균·표준편차를 배워요. train으로만 해야 해요.
- **predict** (predict (예측)): fit으로 배운 규칙을 새 입력에 적용해 답을 내는 단계. 예측할 때는 다시 학습하지 않아요.
- **train/valid/test** (Train / Validation / Test split (학습·검증·시험 분할)): 데이터를 역할별로 나눈 것. train은 학습, valid는 후보 비교·선택, test는 끝까지 보관했다가 마지막에 한 번만 확인. 역할을 섞으면 점수가 실제보다 좋아 보여요.
- **기준선** (Baseline (기준선 모델)): 아무것도 배우지 않은 가장 단순한 예측(예: train 평균만 예측, 가장 많은 클래스만 예측). 새 모델은 최소한 이것보다 나아야 의미가 있어요.
- **선형회귀** (Linear Regression (선형회귀)): 입력의 가중합 ŷ = w₁x₁ + … + b로 숫자를 예측하는 모델. 잔차제곱합이 가장 작아지는 w와 b를 찾아요.
- **가중합** (Weighted sum (가중합)): 각 입력에 가중치(계수)를 곱해 모두 더한 값. w₁x₁ + w₂x₂ + … + b.
- **계수** (Coefficient (계수, 기울기 w)): 입력이 1 늘 때 예측이 얼마나 변하는지를 나타내는 학습된 숫자. 다른 입력을 고정했을 때의 변화일 뿐, 원인·결과의 증명은 아니에요.
- **절편** (Intercept (절편 b)): 모든 입력이 0일 때의 예측값. 직선이 세로축과 만나는 높이예요.
- **잔차** (Residual (잔차)): 실제값 − 예측값. 점과 직선 사이의 세로 거리. 양수면 실제가 예측보다 크다는 뜻.
- **잔차제곱합** (Sum of Squared Residuals (잔차제곱합)): 잔차를 각각 제곱해서 모두 더한 값. +와 −가 서로 지워지지 않고, 크게 빗나간 점일수록 훨씬 큰 벌점을 받아요.
- **외삽** (Extrapolation (외삽)): 학습 데이터 범위 밖의 값을 예측하는 것. 직선이 범위 밖에서도 그대로 이어진다는 보장이 없어 믿기 어려워요.
- **MAE** (Mean Absolute Error (평균 절대 오차)): |실제 − 예측|의 평균. 정답과 같은 단위(예: ms)이고 작을수록 좋아요.
- **MSE** (Mean Squared Error (평균 제곱 오차)): (실제 − 예측)²의 평균. 큰 오차에 더 큰 벌점을 주지만 단위가 제곱(ms²)이 돼요.
- **RMSE** (Root Mean Squared Error (평균 제곱근 오차)): MSE에 제곱근을 씌운 값. 정답과 같은 단위로 돌아오면서 큰 오차에 민감한 성질은 남아요. 이 수업 회귀의 주 지표.
- **R²** (Coefficient of determination (결정계수)): 1 − 잔차제곱합 ÷ 전체제곱합. 평가 자료 y의 평균만 찍는 것보다 얼마나 나은지. 최대 1, 평균보다 못하면 음수도 나와요. 오류 비율이 아니에요.
- **로지스틱회귀** (Logistic Regression (로지스틱회귀)): 이름은 회귀지만 분류 모델. 선형 점수 z를 시그모이드로 0~1 확률로 바꾸고, 임계값과 비교해 클래스를 정해요.
- **시그모이드** (Sigmoid function (시그모이드 함수)): 어떤 숫자 z든 0~1 사이로 눌러 담는 함수. p = 1 / (1 + e⁻ᶻ). z=0이면 0.5.
- **predict_proba** (predict_proba (클래스별 확률 예측)): 각 클래스일 확률을 돌려주는 함수. 열 순서는 classes_를 따라요. predict는 이 확률을 임계값에서 잘라 0/1로 바꾼 결과예요.
- **임계값** (Threshold (판단 기준값)): 확률이 이 값 이상(또는 초과)이면 양성(실패)이라고 부르는 선. 낮추면 놓침(FN)은 줄고 헛경고(FP)는 늘어요. valid에서만 정해요.
- **로그 손실** (Log loss (로그 손실)): 분류 모델이 학습할 때 줄이는 값. 정답에 낮은 확률을 줄수록(틀린 쪽에 자신만만할수록) 벌점이 폭발적으로 커져요. 평가 지표(F1)와는 별개예요.
- **규제** (Regularization (규제)): 계수가 지나치게 커지지 않게 벌점을 줘서 학습을 제어하는 것. 로지스틱회귀에선 C로 조절하며, C가 작을수록 규제가 세요. 스케일링과는 다른 개념.
- **수렴** (Convergence (수렴)): 학습이 반복하며 더 나아지지 않는 안정된 답에 도달하는 것. max_iter(반복 상한)가 너무 작으면 도달 전에 멈춰 수렴 경고가 나요.
- **양성** (Positive class (양성 클래스)): 찾고 싶은 쪽으로 정한 클래스. 이 수업에선 실패(1)가 양성이에요. Precision·Recall·F1은 양성을 기준으로 계산돼요.
- **혼동행렬** (Confusion Matrix (혼동행렬)): 실제(행) × 예측(열)으로 맞힘·틀림을 네 칸(TN, FP, FN, TP)에 세어 둔 표. 어떤 종류의 오류인지 나눠 볼 수 있어요.
- **TP** (True Positive (맞춘 양성)): 실제 양성을 양성이라고 맞힌 경우(예: 실제 실패를 실패로 예측). 객체탐지에서는 정답 박스와 짝지어진 예측을 뜻해요.
- **TN** (True Negative (맞춘 음성)): 실제 음성을 음성이라고 맞힌 경우(예: 실제 성공을 성공으로 예측). F1에는 들어가지 않아요.
- **FP** (False Positive (헛경고·오탐)): 실제는 음성인데 양성이라고 한 경우(괜한 경고). 객체탐지에서는 정답과 짝이 없는 예측.
- **FN** (False Negative (놓침)): 실제는 양성인데 음성이라고 한 경우. 찾아야 할 것을 못 찾음.
- **Accuracy** (Accuracy (정확도)): 전체 중 맞힌 비율. (TP + TN) ÷ 전체. 한쪽 클래스가 많으면 그쪽만 찍어도 높게 나와 속기 쉬워요.
- **Precision** (Precision (정밀도)): 양성이라고 한 것 중 진짜 양성인 비율. TP ÷ (TP + FP). 헛경고가 늘면 낮아져요.
- **Recall** (Recall (재현율)): 실제 양성 중 모델이 찾아낸 비율. TP ÷ (TP + FN). 놓침이 늘면 낮아져요.
- **F1** (F1 score): Precision과 Recall의 조화평균. 2TP ÷ (2TP + FP + FN). 둘 중 하나라도 낮으면 같이 낮아지고 TN은 반영하지 않아요. 이 수업 분류의 주 지표.
- **zero_division** (zero_division (0으로 나누기 처리)): 양성 예측이 0건이라 Precision 분모가 0일 때 어떤 값으로 기록할지 정하는 인자. 0으로 두면 경고 없이 0을 기록해요.
- **KNN** (K-Nearest Neighbors (K-최근접 이웃)): 새 입력과 가장 가까운 학습 사례 K개를 찾아, 분류면 다수결 투표·회귀면 평균으로 예측하는 모델. fit은 사례를 기억만 하고 일은 predict 때 해요.
- **n_neighbors** (n_neighbors (K, 참고할 이웃 수)): KNN이 투표·평균에 쓸 이웃 수. 작으면 잡음에 휘둘리고, 크면 소수 클래스의 패턴을 놓칠 수 있어요.
- **유클리드 거리** (Euclidean distance (직선 거리)): 두 점 사이의 직선 거리. 각 특성 차이를 제곱해 더한 뒤 제곱근. KNN의 기본 거리(p=2).
- **맨해튼 거리** (Manhattan distance (가로+세로 거리)): 각 특성 차이의 절댓값을 그냥 더한 거리(p=1).
- **스케일링** (Feature Scaling (특성 스케일링)): 단위가 다른 특성들의 눈금을 비슷하게 맞추는 전처리. 거리로 판단하는 KNN에선 단위가 큰 특성이 거리를 독차지하지 못하게 해 줘요. 점수 상승을 보장하진 않아요.
- **표준화** (Standardization (표준화)): 각 특성에서 train 평균을 빼고 train 표준편차로 나누는 변환. 결과는 평균 0·표준편차 1 근처. 정규분포로 바꿔 주는 건 아니에요.
- **최소–최대 스케일링** (Min-Max Scaling): train의 최솟값·최댓값으로 (x − 최소) ÷ (최대 − 최소)를 계산해 0~1로 맞추는 변환. train 범위 밖의 새 값은 0~1을 벗어날 수 있어요.
- **벡터 정규화** (Vector normalization (행 단위 정규화)): 각 행(사례)의 벡터 길이를 1로 맞추는 변환. 열 단위인 표준화·MinMax와 방향이 달라요.
- **표준편차** (Standard deviation (표준편차)): 값들이 평균에서 평균적으로 얼마나 퍼져 있는지 나타내는 숫자. 클수록 들쭉날쭉.
- **전처리** (Preprocessing (전처리)): 모델에 넣기 전에 데이터를 다듬는 단계(결측 채우기, 스케일링, 원-핫 인코딩 등). 전처리가 배우는 값도 train에서만 배워야 해요.
- **결측** (Missing value (결측값)): 데이터가 비어 있는 칸. 모델 대부분은 빈칸을 못 받아서 채우거나(대체) 처리해야 해요.
- **결측 대체** (Imputation (결측 대체)): 빈칸을 정한 규칙의 값으로 채우는 것. 수치는 train 중앙값, 범주는 train 최빈값으로 채웠어요.
- **중앙값** (Median (중앙값)): 값을 크기순으로 줄 세웠을 때 가운데 값. 극단값에 덜 끌려가요.
- **최빈값** (Mode (최빈값)): 가장 자주 나온 값. 평균을 낼 수 없는 범주형 결측을 채울 때 써요.
- **수치형** (Numerical feature (수치형 특성)): 크기 비교와 계산이 되는 숫자 특성(예: 글자수, 허용 도구수).
- **범주형** (Categorical feature (범주형 특성)): 종류를 나타내는 특성(예: task_type, priority). 숫자로 바로 계산할 수 없어 원-핫 인코딩 등으로 바꿔 넣어요.
- **원-핫 인코딩** (One-hot encoding (원-핫 인코딩)): 범주 하나를 '그 범주면 1, 아니면 0'인 열 여러 개로 펼치는 변환. 이렇게 생긴 열을 더미 열이라고도 해요.
- **handle_unknown** (handle_unknown="ignore"): train에 없던 새 범주가 들어왔을 때 오류 대신 그 범주 블록을 모두 0으로 처리하게 하는 OneHotEncoder 설정.
- **Pipeline** (Pipeline (전처리 + 모델 묶음)): 전처리 단계와 모델을 한 줄로 묶은 객체. fit(train)하면 전처리·모델을 train으로만 배우고, predict(valid)하면 배운 규칙으로 transform → predict만 해서 누수를 막아 줘요.
- **ColumnTransformer** (ColumnTransformer (열별 변환기)): 열 묶음마다 다른 전처리를 적용하고 결과를 합쳐 주는 도구. 수치 열은 중앙값·스케일링, 범주 열은 최빈값·원-핫으로.
- **transform** (transform / fit_transform): transform은 이미 배운 규칙(평균·중앙값 등)으로 데이터를 바꾸기만 하는 것. fit_transform은 배우기와 바꾸기를 한 번에(train에만).
- **누수** (Data Leakage (데이터 누수)): 검증·테스트용 정보나 예측 시점에 알 수 없는 값(예: 실행 후에야 생기는 elapsed_ms)이 학습에 섞여 들어가 점수가 실제보다 좋게 나오는 문제.
- **하이퍼파라미터** (Hyperparameter (하이퍼파라미터)): 학습 전에 사람이 정하는 설정값(예: K, weights, C, max_depth). train으로 자동 결정되는 계수와 달리 후보를 정해 valid·교차검증으로 비교해 골라요.
- **학습 파라미터** (Learned parameter (학습 파라미터)): fit하면서 train 데이터로 자동으로 정해지는 값(예: 선형·로지스틱회귀의 계수와 절편).
- **과적합** (Overfitting (과적합)): 학습 자료의 우연한 잡음까지 외워서 train 점수만 높고 새 자료(valid) 점수는 낮은 상태. 두 점수 격차가 핵심 신호예요.
- **과소적합** (Underfitting (과소적합)): 모델이 너무 단순하거나 제약이 세서 필요한 규칙도 못 배운 상태. train과 valid 점수가 둘 다 낮아요.
- **일반화** (Generalization (일반화)): 학습에 쓰지 않은 새 자료에서도 잘 맞히는 능력. 모델 평가의 진짜 목표예요.
- **선택 편향** (Selection bias (선택 편향)): 같은 평가 자료를 여러 번 보며 가장 좋게 나온 후보를 고르면, 그 점수에 '운 좋게 맞은 몫'이 섞여 실제보다 낙관적이 되는 현상.
- **재현성** (Reproducibility (재현성)): 같은 데이터·코드·설정이면 언제 누가 돌려도 같은 결과가 나오는 성질. seed(난수 고정)뿐 아니라 데이터 버전·분할 ID·조건을 함께 기록해요.
- **joblib** (joblib (모델 저장·복원 도구)): 학습한 Pipeline을 파일로 저장(dump)하고 다시 불러오는(load) 라이브러리. 불러오면 안의 코드가 실행될 수 있어 직접 만든 파일만 load해요.
- **scikit-learn** (scikit-learn (파이썬 머신러닝 라이브러리)): 모델·전처리·지표·Pipeline을 fit/predict 같은 같은 방식으로 제공하는 라이브러리. pip로는 scikit-learn, 코드에선 sklearn으로 불러요.
- **numpy** (NumPy (숫자 배열 계산 라이브러리)): 숫자 배열(np.array)을 빠르게 계산하는 파이썬 라이브러리.
- **pandas** (pandas (표 데이터 라이브러리)): 엑셀 같은 표(DataFrame)로 데이터를 읽고 다루는 파이썬 라이브러리.
- **가상환경** (Virtual environment (가상환경, .venv)): 프로젝트마다 따로 쓰는 파이썬과 패키지 묶음. 버전이 섞이지 않게 해 같은 코드가 같은 결과를 내도록 도와요.
- **합성 데이터** (Synthetic data (합성 데이터)): 실제 기록이 아니라 수업용으로 만들어 낸 가짜 데이터. 여기서 나온 점수는 실서비스 성능의 근거가 아니에요.

---

# Day7 · 트리·앙상블·교차검증
_2026-10-27(화) · Decision Tree · Random Forest · XGBoost · 특성 중요도 · 5-fold CV_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day07.html

📌 딱 3줄 요약

- 트리는 "글자수 ≤ 500?" 같은 질문으로 자료를 나눠 예측하고, 깊어질수록 과적합 위험이 커진다.

- Random Forest는 서로 다른 트리를 동시에 여러 개 만들어 평균, XGBoost는 앞의 부족한 점을 순서대로 보정한다.

- 교차검증은 train을 5조각으로 돌려 가며 평가해 한 번의 분할에 덜 흔들리게 한다. 전처리도 fold 안에서 다시 학습!

🔧 Day7 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| scikit-learn | 트리·랜덤포레스트·교차검증·중요도·Pipeline | from sklearn.tree import DecisionTreeClassifier 등 |
| xgboost | 그래디언트 부스팅 트리 (sklearn처럼 fit/predict) | from xgboost import XGBClassifier, XGBRegressor |
| numpy, pandas | 배열 계산, 비교표·중요도표 만들기 | import numpy as np, import pandas as pd |
| joblib | 학습한 Pipeline 저장·복원 (scikit-learn과 함께 설치됨) | import joblib |

### 🔧 설치 · 확인

```
python --version
python -m pip show scikit-learn xgboost
python -m pip install "xgboost==3.1.3"        # xgboost가 없을 때만 (강의 검증 버전 예시)
python -c "import xgboost; print(xgboost.__version__)"
python -m pip check
```

💡 오늘은 GPU 불필요. XGBoost도 tree_method="hist", device="cpu", n_jobs=1로 CPU에서 돌려요. CUDA 설치는 선행 조건이 아니에요.

⚠️ 기존 .venv를 그대로 쓰고 xgboost만 추가하세요. 강의 실행 검증 환경: Python 3.13.5 · scikit-learn 1.8.0 · XGBoost 3.1.3 (PDF 6쪽).

🤔 왜 n_jobs=1로 하지? — 병렬로 돌리면 빠르지만, 작업 순서가 매번 조금씩 달라져 결과 재현이 어려워질 수 있고 노트북 CPU를 모두 잡아먹어요. 수업에선 같은 숫자가 다시 나오는 것이 속도보다 중요해서 1로 고정해요.

## 이거 어디에 써?

트리 계열은 표(엑셀 같은 데이터)에서 가장 많이 쓰는 모델이에요. 특히 "왜 이렇게 예측했어?"를 조건문으로 설명할 수 있다는 게 장점이에요.

🌳

설명 가능한 규칙"글자수 > 500이면 실패 위험" 같은 조건

💳

신용 · 이상거래표 데이터 분류에 강한 앙상블

📊

무엇을 봤나?특성 중요도로 모델이 의존한 열 확인

🔁

믿을 만한 비교교차검증으로 운에 덜 흔들리는 점수

|  | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|
| 모양 | 트리 1개 | 트리 여러 개 (독립) | 트리 여러 개 (순차) |
| 결합 | — | 확률·수치 평균 | 기본 점수에 보정값 누적 |
| 장점 | 규칙을 따라가기 쉬움 | 다양성으로 흔들림 감소 | 손실을 차근차근 줄임 |

💡 데이터 계약은 Day6 그대로예요: 입력 X 4열(input_chars·allowed_tools·task_type·priority), 분류 정답 failed(주 지표: 실패 클래스 F1), 회귀 정답 elapsed_ms(RMSE, ms), 금지 입력(failed·elapsed_ms·retry_count·status·request_id), 고정 분할 train 60 · valid 20 · test 20. 수업용 합성 요청 100건이라 실서비스 성능을 대표하지 않아요.

🤔 왜 모델을 바꿔도 분할은 그대로 두지? — 분할까지 바꾸면 점수 차이가 "모델이 달라서"인지 "시험지가 달라서"인지 구분할 수 없어요. 같은 ID·같은 순서를 유지해야 모델만의 차이를 볼 수 있어요.

## ① Decision Tree — 질문으로 나눠 예측

한 번에 한 특성의 조건을 검사해 자료를 둘로 나눠요. 질문은 사람이 쓰는 게 아니라 학습 자료로 자동 선택돼요.

| 용어 | 뜻 |
|---|---|
| 루트 | 첫 질문 |
| 내부 노드 | 다음 질문들 |
| 잎(leaf) | 최종 예측 위치 — 분류는 잎의 클래스 분포, 회귀는 잎의 y 평균 |

어떤 질문이 좋은가? — 나눈 뒤에 덜 섞여 있을수록 좋은 질문이에요. 섞인 정도를 Gini 불순도로 재요.

```
Gini = 1 − (성공 비율)² − (실패 비율)²
  성공 3 · 실패 3  →  1 − 0.5² − 0.5² = 0.5   (최대로 섞임)
  성공 3 · 실패 0  →  1 − 1² − 0²     = 0     (깨끗함)

분할 후 Gini = 왼쪽 Gini × (왼쪽 비율) + 오른쪽 Gini × (오른쪽 비율)
감소량 = 부모 Gini − 분할 후 Gini   ← 가장 많이 줄이는 질문을 고른다

예) 후보 A "≤ 250": 왼쪽 2건(0,0) Gini 0 · 오른쪽 4건(0,1,1,1) Gini 0.375
    → 분할 후 = 0×2/6 + 0.375×4/6 = 0.25,  감소량 0.5 − 0.25 = 0.25
    후보 B "≤ 500": 왼쪽 3건(0,0,0) · 오른쪽 3건(1,1,1) 모두 Gini 0 → 감소량 0.5  ✅ 선택
```

🤔 왜 "덜 섞인" 쪽이 좋은 질문이지? — 잎이 한 클래스로만 차 있으면 그 잎에 떨어진 새 입력을 자신 있게 예측할 수 있어요. 반반 섞인 잎은 동전 던지기와 같죠. Gini는 "그 노드에서 무작위로 두 개를 뽑았을 때 클래스가 다를 확률"이라, 0이면 깨끗하고 0.5(이진 분류 최대)면 최악이에요. 분할 후 Gini를 건수 비율로 가중하는 이유는 큰 쪽 노드가 더 많은 자료를 대표하기 때문이에요.

[그림: 노드 안의 실패 비율에 따른 Gini 불순도 곡선] 무엇을 보나: 가로축은 한 노드 안의 실패 비율, 세로축은 Gini예요. 양 끝(한 클래스만 있음)에서 0, 정확히 반반(0.5)일 때 꼭대기 0.5인 언덕 모양이에요. 🤔 왜? Gini = 1 - p² - (1-p)²는 "무작위로 두 개 뽑았을 때 클래스가 다를 확률"이라, 한쪽으로 쏠릴수록 같은 걸 뽑기 쉬워 0으로 떨어져요. 좋은 질문은 자식 노드들을 이 언덕의 양 끝 쪽으로 밀어내는 질문이에요.

아래 6개 요청의 기준선을 움직여 보세요. 점을 클릭하면 성공 ↔ 실패가 바뀌어요.

⚠️ Gini는 노드가 얼마나 섞였는지 재는 값이지 시험 점수나 회귀 오차가 아니에요. 그리고 트리는 매 단계에서 가장 좋은 질문을 고를 뿐, 전체 최적을 한 번에 찾는 건 아니에요.

🤔 왜 전체 최적이 아니지? — 가능한 모든 트리 모양을 다 비교하는 건 경우의 수가 너무 많아요. 그래서 트리는 "지금 이 노드에서 가장 많이 줄이는 질문"을 고르고 다음으로 넘어가요(탐욕적 방법). 빠르지만, 지금은 덜 좋아 보여도 나중에 더 좋은 길을 여는 질문은 놓칠 수 있어요.

[그림: 수업 데이터로 학습한 깊이 3 결정트리 구조] 무엇을 보나: 수업 데이터 train 60건으로 학습한 기본 트리(max_depth=3, min_samples_leaf=3)를 그대로 그렸어요. 상자마다 질문·gini·samples(행 수)·value([성공, 실패] 수)가 있고, 주황은 성공 쪽, 파랑은 실패 쪽, 색이 진할수록 한 클래스로 깨끗해요. 🤔 왜? 루트에서 "input_chars ≤ 1866" 한 번에 실패만 6건인 잎(gini 0)을 떼어 냈어요. 그 질문이 그 순간 Gini를 가장 많이 줄였기 때문이에요(탐욕적 선택). 잎이 4개라 위 표의 get_n_leaves() = 4와 일치하고, 각 잎의 클래스 비율이 곧 predict_proba 값이에요(예: 오른쪽 아래 잎 실패 5/7 ≈ 0.71).

🔧 DecisionTreeClassifier · DecisionTreeRegressor 명령어 사전에서 자세히 →

```
tree = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, [0, 0, 0, 1, 1, 1])
reg = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, [100, 120, 140, 700, 720, 740])
```

[그림: 회귀 트리의 계단 모양 예측과 선형회귀 직선 비교] 무엇을 보나: 위 코드의 6개 점에 깊이 1 회귀 트리를 맞춘 결과예요. 초록 예측은 500에서 한 번 뛰는 계단이고, 계단 높이는 각 잎의 평균(120, 720)이에요. 회색 점선은 비교용 직선이에요. 🤔 왜? 트리는 "≤ 500?"으로 잎을 고른 뒤 그 잎의 평균만 말해요. 그래서 오른쪽 회색 구간(학습 범위 밖) 1200을 넣어도 직선처럼 올라가지 않고 720에 머물러요.

## ② 복잡도와 과적합

깊은 트리는 학습 자료의 작은 차이, 우연한 잡음까지 나눠 버려요. train 점수만 높고 새 자료 점수가 낮으면 과적합을 의심해요.

| max_depth | train F1 | valid F1 |
|---|---|---|
| 1 | 0.500 | 0.286 |
| 3 | 0.710 | 0.333 |
| 5 | 0.882 | 0.500 |
| None (제한 없음) | 1.000 | 0.500 |

(min_samples_leaf=1로 둔 진단 실험) 제한 없는 트리는 train을 완벽히 외웠지만 valid는 깊이 5보다 나아지지 않았어요. 한 번의 분할만 보고 "최적 깊이"를 단정하진 않아요. 이 표는 깊이별 실험 결과이지, epoch에 따른 학습 곡선이나 미래 성능 보장이 아니에요.

🤔 왜 깊어질수록 train만 좋아지지? — 질문을 계속 허락하면 트리는 결국 학습 행 하나하나를 각자 잎에 가둘 수 있어요. 그러면 train은 전부 맞히지만(1.000), 그 질문들 중 상당수는 우연한 잡음을 가르는 질문이라 새 자료에선 쓸모가 없어요. train–valid 격차는 과적합의 단서예요. 다만 valid가 20건이라 표본 크기·분포 차이도 함께 살펴요.

[그림: max_depth별 train F1과 valid F1 비교 그래프] 무엇을 보나: 수업 데이터로 깊이를 바꿔 직접 계산한 결과예요(위 표 + 깊이 2·4·6 추가). 회색 train은 깊어질수록 1.0까지 오르지만 보라 valid는 0.5에서 멈추고, 분홍 틈이 계속 벌어져요. 🤔 왜? 깊이가 늘면 잎 개수가 2 → 14로 늘어나 train 행을 더 잘게 가둘 수 있어요. 새로 생긴 잎들이 잡음을 가르는 데 쓰이면 train만 좋아지고 valid는 그대로라, 그 틈이 과적합의 신호예요.

[그림: 합성 2차원 자료에서 깊이 2 트리와 제한 없는 트리의 결정 영역] 무엇을 보나: 같은 합성 자료에서 깊이 2(왼쪽)와 제한 없음(오른쪽)의 판단 영역이에요. 트리 경계는 늘 가로·세로 선으로만 만들어져요. 🤔 왜? 트리 질문은 "한 특성 ≤ 기준값?"이라 한 번에 한 축만 잘라요. 제한을 풀면 반대색 점 하나까지 따로 가두는 가늘고 작은 직사각형이 생기는데, 그게 "잡음까지 외운" 모습이에요.

💡 트리는 거리 기반이 아니라서 스케일링이 필수가 아니에요. 그래도 결측 대체(중앙값·최빈값)와 One-hot은 같은 경로로 유지해요 → 변환 결과 (60, 8) = 수치 2 + 범주 더미 6. (일부 구현은 결측을 직접 처리하지만, 오늘은 비교를 위해 명시적인 대체 경로를 고정해요.)

🤔 왜 트리는 스케일이 상관없지? — 트리는 "글자수 ≤ 500?"처럼 한 특성 안에서 크다/작다만 비교해요. 글자수를 1000으로 나눠도 "≤ 0.5?"로 질문만 바뀔 뿐 같은 행들이 같은 쪽으로 가요. 여러 특성을 섞어 거리를 재는 KNN과 달리 단위끼리 경쟁할 일이 없어요.

🔧 복잡도 제한 인자 명령어 사전에서 자세히 →

```
model = build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42), scale="none")
model.fit(X_train, y_train)
```

## ③ Random Forest — 서로 다른 트리들의 평균

train 자료→ 트리마다 다른 자료 중복 허용 추출→ 분할마다 특성 후보도 랜덤 max_features→ 결합 분류: 확률 평균 · 회귀: 예측 평균

같은 실수를 반복하는 트리 100개보다, 서로 다른 관점을 가진 트리를 묶는 게 핵심이에요.

🤔 왜 "서로 달라야" 하지? — 트리 100개가 모두 똑같다면 평균을 내도 트리 1개와 같아요. 트리마다 다른 표본(중복 허용 추출)과 분할마다 다른 특성 후보를 주면 각 트리의 실수가 서로 다른 곳에서 나요. 그러면 평균을 낼 때 제각각인 실수들이 서로 상쇄되어, 한 트리가 잡음에 휘둘리는 정도(흔들림)가 줄어들어요.

[그림: 1차원 회귀에서 개별 트리 5개와 숲 평균의 예측 곡선] 무엇을 보나: 같은 점들에 대해, 서로 다른 표본으로 학습한 트리 5개(얇은 파랑)와 트리 100개의 평균(굵은 보라)을 겹쳐 그렸어요. 점선은 진짜 패턴이에요. 🤔 왜? 트리마다 본 표본이 달라 튀는 위치가 서로 달라요. 평균을 내면 한 트리만 튄 곳은 다른 트리들이 눌러 줘서 보라 선이 덜 출렁이고 점선에 더 가깝게 붙어요. 트리들이 다 같은 곳에서 튄다면 평균을 내도 줄어들지 않겠죠 — 그래서 "서로 달라야" 해요.

| 가상 트리 3개의 실패 확률 | 계산 | 결과 |
|---|---|---|
| 0.51 · 0.51 · 0.01 | 클래스만 다수결: 실패 2표 | 실패(1) |
| 확률을 먼저 평균: (0.51+0.51+0.01)/3 = 0.343 | 성공(0) ← sklearn 방식 |  |

🤔 왜 결과가 다르지? — 다수결은 0.51과 0.99를 똑같이 "실패 1표"로 세지만, 확률 평균은 얼마나 확신했는지까지 반영해요. 두 트리는 겨우 0.51(거의 반반)인데 한 트리는 0.01(강한 성공)이라, 평균하면 0.343 < 0.5 → 성공이에요. 위 확률 3개는 설명용 가상 값이에요. 아래에서 직접 바꿔 보세요.

⚠️ Random Forest의 다음 트리는 앞 트리의 잔차를 배우지 않아요 (그건 부스팅). 그리고 n_estimators=100은 깊이 100이 아니라 트리 개수예요. 회귀 숲은 각 트리의 숫자 예측을 평균해요.

🔧 RandomForestClassifier · RandomForestRegressor 명령어 사전에서 자세히 →

```
rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=1)
forest = build(rf).fit(X_train, y_train)
```

[그림: 합성 자료에서 트리 1개와 랜덤포레스트의 결정 영역 비교] 무엇을 보나: 넉넉한 합성 자료(160건)에서 깊은 트리 1개(왼쪽)와 숲(오른쪽)의 판단 영역이에요. 왼쪽의 각진 섬·가는 띠가 오른쪽에선 대부분 사라지고 경계가 부드러워졌어요. 🤔 왜? 개별 트리의 섬은 표본마다 다른 곳에 생기니, 평균을 내면 대부분 다수 의견에 묻혀 사라져요. 다만 이건 자료가 충분할 때의 모습이에요. 위의 수업 자료처럼 60건에 실패가 소수면 평균이 실패 확률을 0.5 아래로 깎아 오히려 F1이 떨어질 수 있어요.

## ④ 특성 중요도 — 모델이 무엇에 기댔나

|  | MDI (불순도 기반) | 순열 중요도 (Permutation) |
|---|---|---|
| 아이디어 | 각 분기에서 줄어든 불순도 × 그 분기가 다룬 자료 비중을 특성별로 합산 → 여러 트리 평균 | 한 열을 행끼리 섞었을 때 점수가 얼마나 떨어지나 (기준 F1 − 섞은 F1, 여러 번 반복) |
| 계산 자료 | 학습 자료 | 평가 자료 (valid 등) |
| 재학습 | — | 안 함 (학습된 모델 고정) |
| 값 | 합 = 1 (분기 없으면 모두 0) | 점수 감소량 · 음수도 가능 |
| 대상 열 | 변환 후 8개 열 (더미 포함) | 원본 4개 열 (범주 더미 묶음째) |

🤔 왜 열을 섞으면 중요도가 되지? — 한 열의 값을 행끼리 뒤섞으면 그 열과 정답 사이의 연결만 끊기고, 값의 분포(평균·범위)는 그대로예요. 모델이 그 열에 기대고 있었다면 점수가 뚝 떨어지고, 안 쓰고 있었다면 거의 안 변해요. 모델을 다시 학습하지 않으니 "이 학습된 모델이 무엇에 기대나"를 보는 진단이에요. 범주 열은 원본 열째로 섞어서 그 열의 더미 묶음 전체를 함께 평가해요.

[그림: ] 무엇을 보나: 오른쪽 표에선 분홍 칸(task_type)만 행끼리 뒤섞였고, 글자수·priority는 그대로예요. 같은 학습된 모델로 두 표를 예측해 F1 차이를 재요. 🤔 왜? 섞어도 그 열의 값 종류·비율은 똑같고 정답과의 짝만 끊겨요. 모델이 그 열에 기대고 있었다면 짝이 끊긴 순간 점수가 떨어지고, 안 쓰고 있었다면 거의 안 변하죠. 그 낙폭이 "이 모델이 그 열에 얼마나 기댔나"예요.

⚠️ 중요도 해석에서 피할 결론

- 중요 ≠ 원인. 영향의 방향(+/−)도 알려주지 않아요.

- 값 종류가 많은 특성은 분할 후보가 많아 MDI가 높게 보일 수 있어요.

- 서로 비슷한(상관된) 입력은 기여를 나눠 가질 수 있어요.

- 순위를 보고 평가셋에 맞춰 열을 반복 삭제하면 선택 편향이 커져요. 열 선택도 학습 fold 안에서.

- valid 20행처럼 작으면 순위·부호가 불안정해요. 음수가 나와도 바로 삭제하지 않아요.

- 현재 Forest의 valid F1은 0.222예요. 예측을 잘 못하는 모델의 중요도를 업무 우선순위로 단정하지 않아요.

🤔 왜 이런 함정이 생기지? — MDI는 "분할에 얼마나 쓰였나"를 세는데, 값 종류가 많은 열(글자수처럼)은 자를 수 있는 기준선 후보가 많아서 우연히라도 불순도를 줄이는 분할을 찾기 쉬워요. 상관된 두 열은 같은 정보를 담고 있어서 트리가 어느 쪽을 써도 되니 기여가 둘로 나뉘어 각각 작아 보여요. 그리고 중요도는 "얼마나 기댔나"의 크기만 재지, 값이 클수록 실패가 늘었는지 줄었는지(방향)나 원인 관계는 알려 주지 않아요.

🔧 feature_importances_ · permutation_importance 명령어 사전에서 자세히 →

```
mdi = pd.Series(forest["model"].feature_importances_, index=names)
perm = permutation_importance(forest, X_valid, y_valid, scoring="f1", n_repeats=20, random_state=42, n_jobs=1)
```

[그림: 랜덤포레스트의 MDI 중요도와 순열 중요도 막대 비교] 무엇을 보나: 수업 데이터 Forest로 직접 계산한 두 중요도예요(위 코드 출력과 같은 값). 왼쪽 MDI는 input_chars가 압도적 1위, 오른쪽 순열 중요도는 task_type이 1위이고 priority는 빨간 음수예요. 회색 가로 막대는 20번 섞을 때마다의 흔들림(std)이에요. 🤔 왜? MDI는 train에서 "분할에 많이 쓰였나"(값 종류가 많은 글자수가 유리), 순열은 valid에서 "섞으면 F1이 떨어지나"를 재요. 오른쪽 흔들림 막대가 0을 가로지를 만큼 길죠 — valid 20건이라 순위·부호가 운에 크게 좌우된다는 뜻이라, 음수라고 바로 열을 지우지 않아요.

## ⑤ XGBoost — 부족한 부분을 순서대로 보정

처음 예측 예: 정답 평균→ 보정 트리 지금 틀린 방향을 학습→ 예측 갱신 이전 + 학습률 × 보정→ 반복 순차적으로

```
y        = [100, 120, 200, 220]
처음     = [160, 160, 160, 160]         (평균)          MSE 2600
잔차     = [-60, -40,  40,  60]         ← 깊이1 트리가 이걸 학습
보정 후  = 처음 + 0.5 × 보정 = [135, 135, 185, 185]      MSE 725
```

🔧 부스팅 한 회차 직접 계산 DecisionTreeRegressor 명령어 사전에서 자세히 →

```
X = np.array([[1], [2], [3], [4]])
y = np.array([100., 120., 200., 220.])
```

🤔 왜 MSE가 2600 → 725로 줄지? — 처음엔 모두 평균 160이라 오차가 (−60, −40, 40, 60), 제곱 평균 2600. 깊이 1 트리는 잔차를 "x ≤ 2.5 → −50, 아니면 +50"(각 쪽 잔차 평균)으로 배워요. 학습률 0.5라 그 절반(±25)만 더하니 135·135·185·185, 새 오차 (−35, −15, 15, 35)의 제곱 평균은 725. 틀린 방향으로 조금씩 당겨 주는 게 부스팅이에요.

[그림: 부스팅 한 회차 전후의 예측과 잔차 화살표 비교] 무엇을 보나: 막대가 실제 y, 주황 마름모가 예측, 분홍 화살표가 잔차(실제 - 예측)예요. 왼쪽(모두 160)보다 오른쪽(보정 1회 후)에서 화살표가 모두 짧아졌어요. 🤔 왜? 깊이 1 보정 트리가 "x ≤ 2.5면 -50, 아니면 +50"이라는 잔차의 방향을 배웠고, 학습률 0.5만큼(±25)만 예측을 그쪽으로 옮겼어요. 잔차가 60 → 35, 40 → 15로 줄어 제곱 평균이 2600 → 725가 돼요.

🤔 왜 보정을 절반(학습률)만 반영하지? — 보정 트리도 작은 자료에서 배운 것이라 잡음이 섞여 있어요. 한 번에 100% 믿고 반영하면 그 잡음까지 따라가 과적합되기 쉬워요. 조금씩 여러 번 고치면 여러 트리의 의견이 쌓여 더 안정적이에요. 대신 학습률이 작으면 회차가 더 필요해요.

[그림: 부스팅 회차 1, 3, 10, 80에서의 예측 곡선 변화] 무엇을 보나: 합성 자료에 얕은 트리(깊이 2)를 학습률 0.3으로 계속 더해 가는 모습이에요. 1회차는 밋밋한 계단, 10회차는 패턴을 잘 따르고, 80회차는 점 하나하나를 쫓아 뾰족뾰족해져요. 🤔 왜? 회차마다 "아직 남은 잔차"를 조금씩 고치니 train MSE는 계속 줄어요(0.31 → 0.00). 그런데 큰 패턴을 다 배운 뒤엔 남은 잔차가 대부분 잡음이라, 그걸 고치는 건 외우기예요. 그래서 학습률·회차·조기 종료로 "어디서 멈출지"를 조절해요.

💡 "잔차를 그대로 배운다"는 건 제곱오차일 때의 직관이에요. 분류에서는 손실의 기울기 정보를 써요. XGBoost는 여기에 트리 복잡도·잎 값에 대한 규제를 더한 구현이에요.

| 관점 | Random Forest (Bagging) | XGBoost (Boosting) |
|---|---|---|
| 학습 관계 | 트리를 독립적으로 | 현재 예측을 보정하며 순차 추가 |
| 다양성·보정 | 표본·특성 후보의 무작위성 | 손실의 기울기·곡률에 따른 보정 |
| 결합 | 평균 | 기본 점수에 보정값 누적 |
| 핵심 조절 | 트리 수·깊이·잎 크기·특성 수 | 깊이·학습률·회차·표본/특성 비율·규제 |

[그림: ] 무엇을 보나: 위(Random Forest)는 트리들이 나란히 서 있고 서로 화살표가 없어요. 아래(XGBoost)는 트리가 한 줄로 이어지고 각 트리가 앞 결과를 받아요. 🤔 왜? 숲은 "다양한 의견의 평균"으로 흔들림을 줄이려는 것이라 트리끼리 몰라야 하고, 부스팅은 "아직 틀린 부분을 고치는 것"이라 앞 트리의 잔차를 알아야 해요. 그래서 숲의 트리는 동시에 만들 수 있지만, 부스팅은 순서대로만 만들 수 있어요.

🔧 XGBClassifier · XGBRegressor 명령어 사전에서 자세히 →

```
xgb = XGBClassifier(n_estimators=80, learning_rate=0.05, max_depth=2, …, objective="binary:logistic", tree_method="hist")
model = build(xgb).fit(X_train, y_train)           # sklearn Pipeline에 그대로 연결
```

[그림: XGBoost 회차별 학습 손실과 stop 자료 손실 곡선] 무엇을 보나: 설명용 합성 자료로 XGBoost를 400회 돌리며 학습 자료(회색)와 따로 떼어 둔 stop 자료(보라)의 logloss를 그렸어요. 회색은 끝까지 내려가지만 보라는 60회 근처 바닥을 찍고 다시 올라가요. 🤔 왜? 바닥 이후의 회차는 학습 자료의 잡음만 고치고 있어서 안 본 자료에선 오히려 손해예요. 조기 종료는 보라 선이 일정 회차(예: 15회) 동안 더 내려가지 않으면 멈추고 그 바닥 회차를 고르는 규칙이에요. (수업 시연의 최적 회차 64는 다른 자료라 이 그림과 숫자가 같을 필요는 없어요.)

## ⑥ 교차검증 — 한 번의 분할에 덜 의존하기

valid 20건 한 번만 보면 운이 크게 작용해요. train 60건을 5조각(12건씩)으로 나눠, 돌아가며 1조각으로 평가하고 4조각(48건)으로 학습해요.

🤔 왜 한 번 나누기 대신 교차검증이지? — 한 번의 valid는 "어떤 20건이 우연히 뽑혔나"에 점수가 크게 좌우돼요(1건 = F1이 크게 출렁). 5번 돌리면 train의 모든 행이 한 번씩 평가에 쓰이고, 점수 5개의 평균과 흩어짐을 함께 볼 수 있어 운에 덜 흔들려요. 대신 계산은 5배예요. 그래도 CV 역시 후보 선택에 쓴 점수라, 평균이 높다는 것만으로 최종 성능을 확정하진 않아요.

| 회차 | 조각1 | 조각2 | 조각3 | 조각4 | 조각5 |
|---|---|---|---|---|---|
| 1 | 평가 | 학습 | 학습 | 학습 | 학습 |
| 2 | 학습 | 평가 | 학습 | 학습 | 학습 |
| 3 | 학습 | 학습 | 평가 | 학습 | 학습 |
| 4 | 학습 | 학습 | 학습 | 평가 | 학습 |
| 5 | 학습 | 학습 | 학습 | 학습 | 평가 |

[그림: 5-fold 교차검증에서 회차마다 평가 조각이 바뀌는 블록 그림] 무엇을 보나: 한 줄이 한 회차예요. 분홍 칸(평가 12건)이 대각선을 따라 한 칸씩 옮겨 가고, 나머지 4칸(48건)으로 학습해요. 🤔 왜? 대각선이라는 건 모든 조각이 정확히 한 번씩 평가된다는 뜻이에요. 그래서 "어떤 12건이 평가에 뽑혔나"의 운이 5번에 걸쳐 평균되고, 점수 5개가 얼마나 흩어지는지도 볼 수 있어요.

| 자료 | 오늘의 역할 |
|---|---|
| train 60 | 5-fold로 후보 비교 (각 fold에서 전처리도 학습) |
| valid 20 | 이미 쓴 진단 자료 — 오류·기존 결과 비교 |
| test 20 | 오늘도 보관, 최종 선택 후 확인 |

⚠️ 틀린 CV vs 올바른 CV ❌ train 60 전체에 전처리 fit → 변환된 배열을 5-fold로 나눔 (평가 행의 중앙값이 이미 학습에 섞임) ✅ 아직 fit 안 한 Pipeline 통째를 cross_validate에 전달 → fold마다 학습 48건으로 전처리·모델 fit → 평가 12건은 transform·predict만

🤔 왜 fold 안에서 전처리를 다시 학습하지? — fold의 평가 12건은 그 회차에선 "처음 보는 자료" 역할이에요. 60건 전체로 중앙값을 먼저 구해 두면 그 12건의 값이 이미 중앙값에 섞여 있어서, 평가 자료를 살짝 미리 본 셈이 돼요. Pipeline을 통째로 넘기면 sklearn이 회차마다 48건으로만 fit하니 이 누수가 자동으로 막혀요. train 60 안에서도 학습 48과 평가 12는 분리해야 해요.

[그림: ] 무엇을 보나: 두 줄은 같은 상자를 쓰지만 "전처리 fit"이 나눔보다 앞이냐 뒤냐만 달라요. 🤔 왜? 위 줄은 60건 전체로 중앙값을 구한 뒤 나누니, 각 회차의 평가 12건 값이 이미 학습 규칙에 들어가 있어요. 아래 줄처럼 Pipeline을 통째로 넘기면 회차마다 학습 48건으로만 fit해서, 평가 12건이 정말 "처음 보는 자료"가 돼요.

🔧 StratifiedKFold · KFold · cross_validate 명령어 사전에서 자세히 →

```
CLS_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # 분류: 실패 비율 유지
cv = cross_validate(build(rf), X_train, y_train, cv=CLS_CV, scoring="f1", return_train_score=True, …)
```

[그림: KFold와 StratifiedKFold의 fold별 실패 건수 비교] 무엇을 보나: 수업 train(60건 중 실패 18건)을 두 방법으로 5조각 냈을 때, 각 평가 조각에 들어간 실패 수예요. 회색 KFold는 2~5건으로 들쭉날쭉, 보라 Stratified는 3~4건으로 기대값(3.6) 근처에 모여요. 🤔 왜? 그냥 섞어 자르면 우연히 실패가 2건뿐인 fold가 생기고, 그 fold의 F1은 한두 건에 크게 흔들려요. Stratified는 클래스별로 나눠 담아 모든 fold의 실패 비율을 원본(30%)에 맞추니 fold끼리 비교가 공정해져요.

[그림: 모델별 5개 fold F1 점수와 평균 및 표준편차] 무엇을 보나: 같은 5개 fold로 계산한 후보별 fold 점수(파란 점 5개)와 평균 ± 표준편차(보라)예요. 값은 아래 표와 같아요(XGB는 이 PC의 실행에서 fold 점수가 표와 조금 달라 그림에서 뺐어요). Forest는 점이 0부터 0.667까지 흩어져 막대가 가장 길어요. 🤔 왜? 평가 조각마다 실패가 3~4건뿐이라 한두 건을 맞히느냐로 F1이 크게 출렁여요. 평균만 보면 Tree(0.542)와 Logistic(0.522)이 달라 보이지만, 점들의 흩어짐이 그 차이보다 훨씬 크죠 — 그래서 평균·편차·fold 점수를 함께 남겨요.

### 지표로 진단하기

| 분류 모델 | CV F1 평균 ± 편차 | valid F1 | valid FP / FN |
|---|---|---|---|
| Dummy | 0.000 ± 0.000 | 0.000 | 0 / 6 |
| Logistic | 0.522 ± 0.209 | 0.333 | 4 / 4 |
| KNN3 | 0.336 ± 0.176 | 0.545 | 2 / 3 |
| Tree | 0.542 ± 0.124 | 0.333 | 4 / 4 |
| Forest | 0.280 ± 0.254 | 0.222 | 2 / 5 |
| XGB | 0.500 ± 0.123 | 0.182 | 4 / 5 |

| 회귀 모델 | CV RMSE 평균 ± 편차 (ms) | valid RMSE | valid MAE |
|---|---|---|---|
| Dummy | 606.18 ± 100.84 | 658.75 | 541.26 |
| Linear | 639.83 ± 57.72 | 640.00 | 551.80 |
| KNN15 | 598.85 ± 91.21 | 641.79 | 530.06 |
| Tree | 599.63 ± 96.12 | 719.00 | 632.03 |
| Forest | 624.01 ± 96.97 | 690.37 | 600.34 |
| XGB | 655.79 ± 92.17 | 700.02 | 609.34 |

오늘의 잠정 기준은 분류 CV 평균 F1 최대 → Tree(0.542), 회귀 CV 평균 RMSE 최소 → KNN15(598.85ms). 반대 증거도 함께 적어요: valid 최고는 KNN3(0.545)·Linear(640.00ms), Tree는 valid 실패 6건 중 4건을 놓침. 비교 시 전처리는 모델에 맞춘 학습 절차 전체(선형/KNN은 수치 standard, 트리 계열은 스케일링 없음)를 비교해요.

🤔 왜 CV 1등과 valid 1등이 다르지? — CV는 train 60 안의 5개 조각 평균, valid는 다른 20건 한 세트예요. 둘 다 작은 자료라 어느 행이 평가에 들어갔느냐에 따라 순위가 바뀔 만큼 차이가 작아요. 그래서 기준(오늘은 CV)을 미리 정해 두고, 그 기준이 Day6의 valid 기준에서 바뀌었다는 사실과 반대 증거를 모두 보고서에 남겨요.

| 이렇게 나오면 | 의심할 것 |
|---|---|
| train F1 = 1.000, valid·CV는 낮음 | 과적합 → 깊이·잎 크기·회차 제한 |
| fold마다 점수가 0 ~ 0.667로 크게 흔들림 | 평가 자료(12건) 구성에 민감 → 평균 하나로 가리지 말 것 |
| CV 1등과 valid 1등이 다름 | 소표본 · 순위 변동 → 유리한 점수만 골라 보고하지 않기 |
| CV RMSE 차이 0.78ms (KNN15 598.85 vs Tree 599.63) | 사실상 동률 → 단순한 후보 우선, 추가 검증 |

💡 후보와 지표는 먼저 정해요. CV를 해도 같은 자료를 반복 조회하면 선택 편향은 남아요. 동점이면 더 단순한 모델을 우선해요.

🤔 왜 동점이면 단순한 쪽? — 점수가 거의 같다면, 더 복잡한 모델의 "약간의 우세"는 작은 자료의 우연일 가능성이 커요. 단순한 모델은 설명·재현·유지가 쉽고, 새 자료에서 크게 무너질 위험도 상대적으로 작아요.

## ⑦ 입력 검사와 저장 — 새 범주 vs 누락 열

새 요청이 들어올 때 두 상황은 다르게 다뤄요. 처음 보는 범주(예: task_type="translation")는 학습된 규칙대로 처리해 8열을 유지하고, 필수 열이 빠진 입력(예: priority 없음)은 예측 전에 명확한 오류로 막아요.

🤔 왜 다르게 다루지? — 새 범주는 실서비스에서 자연스럽게 생기는 일이라 handle_unknown="ignore"로 "그 범주 더미는 전부 0"이라고 처리할 수 있어요. 하지만 열 자체가 없으면 모델이 판단할 근거가 사라진 것이라, 임의 값으로 자동 추측해서 채우면 틀린 예측을 조용히 내보내게 돼요. 그래서 시끄럽게 멈추는 쪽이 안전해요.

[그림: ] 무엇을 보나: 위는 "값이 낯선" 입력, 아래는 "열이 사라진" 입력이에요. 위는 끝까지 흘러가 예측이 나오고, 아래는 중간에 일부러 멈춰요. 🤔 왜? 새 범주는 열 구조(8열)를 지킨 채 "아는 범주 아님 = 더미 모두 0"으로 표현할 수 있지만, 열이 없으면 판단 근거 자체가 없어요. 그걸 조용히 채워 예측하면 틀린 답을 숨기게 되니 시끄럽게 멈추는 쪽이 안전해요.

🔧 새 범주 · 통계 불변 검사 명령어 사전에서 자세히 →

```
new["task_type"], new["priority"] = "translation", "urgent"   # 학습에 없던 범주
print(model["pre"].transform(new).shape)       # (2, 8)  ← 기존 8열 유지
```

🔧 저장 · 복원 대조 명령어 사전에서 자세히 →

```
joblib.dump(model, path)
restored = joblib.load(path)              # 자신이 방금 만든 파일만
```

## 원본 자료

- 강의: 20261027화_머신러닝_트리_앙상블_교차검증_Day7 (3).pdf — ① 트리 9~16쪽 · ② 과적합 17~24쪽 · ③ Random Forest 25~33쪽 · ④ 특성 중요도 34~43쪽 · ⑤ XGBoost 44~55쪽 · ⑥ 교차검증 56~70쪽 · ⑦ 검사·저장 72~73, 77쪽

- 공식 문서: scikit-learn · Decision Trees · Ensemble methods · Permutation importance · Cross-validation · XGBoost Parameters

## 🔧 Day7 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day7)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install scikit-learn xgboost |
| ① Decision Tree | DecisionTreeClassifier / DecisionTreeRegressor, tree_.threshold, get_depth() |
| ② 복잡도 · 과적합 | max_depth=, min_samples_leaf=, min_samples_split=, ccp_alpha= |
| ③ Random Forest | RandomForestClassifier(n_estimators=, max_features=), estimators_ |
| ④ 특성 중요도 | feature_importances_, get_feature_names_out(), permutation_importance() |
| ⑤ XGBoost | XGBClassifier / XGBRegressor(learning_rate=, objective=, tree_method=) |
| ⑥ 교차검증 | StratifiedKFold, KFold, cross_validate(scoring=, return_train_score=) |
| ⑦ 입력 검사 · 저장 | statistics_, transform(new).shape, joblib.dump / joblib.load |

### DecisionTreeClassifier · DecisionTreeRegressor

```
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

X = np.array([[100], [200], [300], [700], [800], [900]])
tree = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, [0, 0, 0, 1, 1, 1])
print(tree.tree_.threshold[0])                    # 500.0  ← 찾은 질문
print(tree.predict([[250], [850]]))               # [0 1]
print(tree.predict_proba([[250], [850]]))         # 잎의 클래스 비율
print(tree.get_depth(), tree.get_n_leaves())      # 1 2

reg = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, [100, 120, 140, 700, 720, 740])
print(reg.predict([[250], [850], [1200]]))        # [120. 720. 720.]  ← 잎 평균
```

▶ 실제로 돌려 보면 노트의 x=[100 … 900] 6점

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| tree.tree_.threshold[0] | 500 | 300과 700의 중간값 |
| tree.predict([[250], [850]]) | [0, 1] | ≤ 500 → 0, > 500 → 1 |
| tree.predict_proba([[250], [850]]) | [[1.0, 0.0], [0.0, 1.0]] | 각 잎이 한 클래스뿐이라 1.0 — 학습 비율일 뿐 |
| tree.get_depth(), tree.get_n_leaves() | (1, 2) | 질문 1개 · 잎 2개 |
| reg.predict([[250], [850], [1200]]) | [120.0, 720.0, 720.0] | 잎 평균 120 / 720 — 1200도 720 (외삽 안 함) |
| 정답이 0·1 번갈아(잡음) + max_depth=None: 깊이 · 잎 | 5 · 6 | 제한이 없으면 6점을 다 외울 때까지 나눔 |

| 인자 / 속성 | 뜻 | 바꾸면? |
|---|---|---|
| max_depth | 질문을 최대 몇 단계까지 | 1이면 질문 1개·잎 2개. None이면 잎이 깨끗해질 때까지 계속 나눔 → 과적합 위험 |
| random_state | 난수 고정 (특성 후보 순서·동률 처리 등) | 안 고정하면 같은 점수의 분할 중 무엇을 고를지가 실행마다 달라질 수 있음 → 결과 재현용 |
| tree_.threshold[0] | 루트 질문의 기준값 (500.0) | 300과 700 사이의 중간값을 기준으로 잡아요 — 두 그룹을 가르는 가장 무난한 선이라서 |
| get_depth() / get_n_leaves() | 실제 깊이 / 잎 개수 | 복잡도 확인용 |
| DecisionTreeRegressor | 연속값 트리 (기본 기준 squared_error) | 잎의 y 평균으로 예측: 왼쪽 (100+120+140)/3 = 120, 오른쪽 720 |

💡 fit에서 질문을 찾고, predict는 찾은 질문을 따라 잎으로 내려갈 뿐이에요(예측 때 다시 학습하지 않음). 회귀 트리는 학습 범위 밖(1200)도 마지막 잎 평균(720)을 내요 — 선형처럼 직선을 연장하지 않아요.

🤔 왜 잎에서 "평균"을 내지? — 회귀 트리는 제곱오차를 줄이는 기준(squared_error)으로 나누는데, 한 묶음의 숫자들과의 제곱오차를 가장 작게 만드는 하나의 값이 바로 평균이에요. 그리고 1200이 들어와도 "글자수 > 500?" → 예 → 오른쪽 잎으로 갈 뿐이라, 잎 바깥의 새 값을 만들어 내지 못해요(외삽 안 함).

⚠️ 한 잎에 실패만 있어서 확률 1이 나와도, 그건 그 잎의 학습 비율이지 새 입력의 결과를 보장하는 게 아니에요.

🤔 왜 확률 1을 믿으면 안 되지? — predict_proba는 "이 잎에 떨어진 학습 행 중 실패 비율"이에요. 잎에 학습 행이 3개뿐이면 3/3 = 1.0이 쉽게 나오지만, 겨우 3건으로 본 비율이라 새 입력에선 얼마든지 달라질 수 있어요.

### 복잡도 제한 인자

```
model = build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42), scale="none")
model.fit(X_train, y_train)
est = model["model"]                       # Pipeline에서 트리만 꺼내기
print(est.get_depth(), est.get_n_leaves()) # 3 4
# train F1 0.710 · valid F1 0.333 · FP=4 · FN=4

for depth in [1, 2, 3, 5, None]:           # 다른 조건은 고정, 깊이만 바꿈
    m = build(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1,
                                     random_state=42), scale="none").fit(X_train, y_train)
    print(depth, m["model"].get_n_leaves(),
          f1_score(y_train, m.predict(X_train)), f1_score(y_valid, m.predict(X_valid)))
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| est.get_depth(), est.get_n_leaves() | (3, 4) | 깊이 3이면 잎 최대 8개지만 실제 4개 — 더 나눌 이득이 없는 노드에선 멈춤 |
| train F1 · valid F1 · FP · FN | 0.710 · 0.333 · 4 · 4 | train과 valid 격차가 큼 |
| leaf_=1, depth 1 / 2 / 3 | 잎 2: 0.500→0.286 / 잎 3: 0.500→0.286 / 잎 4: 0.710→0.333 | train → valid F1 |
| leaf_=1, depth 5 / None | 잎 9: 0.882→0.500 / 잎 14: 1.000→0.500 | 깊을수록 train은 1에 가까워지지만 valid는 따라오지 않음 |
| min_samples_leaf=3 각 잎 행 수 | [6, 7, 21, 26] | 모든 잎이 3행 이상. 이 자료에선 leaf=1로 풀어도 depth 3 트리가 같게 나와(잎 4 · .710→.333) 이 제약이 실제로 바꾼 건 없음 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| max_depth | 최대 질문 깊이 | 너무 작으면 필요한 규칙도 못 배움 (과소적합, 깊이 1: train .500), 너무 크면 잡음까지 외움 |
| min_samples_leaf | 각 잎에 필요한 최소 학습 행 수 | 키우면 "행 1~2개짜리 잎"을 못 만들어 작은 조각(잡음) 분할이 억제됨. "잎 5개"라는 뜻 아님! |
| min_samples_split | 노드를 더 나누는 데 필요한 최소 행 수 | "나눌지 말지"의 조건이라, 나눈 뒤 각 자식 잎의 크기 조건과는 다름 |
| ccp_alpha | 복잡도 비용을 고려한 사후 가지치기 | 잎 하나 늘 때마다 alpha만큼 벌점 → 클수록 불순도를 크게 줄이지 못하는 가지가 잘려 더 단순해짐 |

🤔 왜 복잡도를 학습 전에 정하지? — 트리는 내버려 두면 train을 다 외울 때까지 자라요. 깊이·잎 크기 같은 제약은 "얼마나 자세히 외울 수 있게 할지"를 미리 정하는 하이퍼파라미터라, 같은 분할·같은 지표에서 후보를 비교해 골라요. 오늘의 기본 비교 트리는 max_depth=3, min_samples_leaf=3, random_state=42예요.

💡 build(모델)은 Day6 ⑤에서 만든 "전처리 + 모델" Pipeline 함수예요. 트리·숲·XGB는 수치 스케일링 없이(scale="none") 결측 대체 + One-hot만 거쳐요(아래 코드에서 build(rf)처럼 줄여 적어도 트리 계열은 스케일링 없음으로 보면 돼요). model["model"], model["pre"]처럼 단계 이름으로 꺼낼 수 있어요.

⚠️ valid에서 틀린 행만 지우면 성능 개선이 아니라 평가 자료를 결과에 맞춘 부당한 비교예요.

### RandomForestClassifier · RandomForestRegressor

```
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                            max_features="sqrt", random_state=42, n_jobs=1)
forest = build(rf).fit(X_train, y_train)
print(len(forest["model"].estimators_))      # 100  ← 학습된 개별 트리 목록
# train F1 0.800 · valid F1 0.222 · FP=2 · FN=5

rfr = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2,
                            max_features=1.0, random_state=42, n_jobs=1)
forest_r = build(rfr).fit(X_train, yr_train)
# valid MAE 600.34 ms · RMSE 690.37 ms · R² −0.276
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(forest["model"].estimators_) | 100 | 학습된 개별 트리 100개 |
| forest["model"].estimators_[0].get_depth() | 4 | 개별 트리도 max_depth=4 안 |
| train F1 · valid F1 · FP · FN | 0.800 · 0.222 · 2 · 5 | 단일 Tree(valid .333)보다 낮음 — 노트 설명대로 |
| valid 실패 확률 > 0.5 인 행 수 | 3 / 20 | 평균 확률이 0.5 아래로 깎여 실패를 적게 부름 |
| RandomForestRegressor: valid | MAE 600.34 · RMSE 690.37 · R² -0.276 | 회귀도 Dummy보다 나쁨 |

| 인자 | 수업 값 | 뜻 | 바꾸면? |
|---|---|---|---|
| n_estimators | 100 | 숲의 트리 수 | 늘리면 평균이 안정적이지만 계산↑. 같은 실수를 하는 트리가 늘 뿐이면 성능 개선은 보장 안 됨 |
| max_depth, min_samples_leaf | 4, 2 | 개별 트리의 복잡도 | 트리 수와는 다른 조절 — "몇 명에게 묻나" vs "한 명이 얼마나 자세히 아나" |
| max_features | 분류 "sqrt" / 회귀 1.0 | 분할마다 살펴볼 특성 후보 수·비율 | 작을수록 트리마다 다른 특성을 보게 되어 더 다양(상관↓), 대신 각 트리는 약해짐. 1.0이면 전체 특성을 후보로 |
| random_state / n_jobs | 42 / 1 | 난수 재현 / 병렬 작업 수 | random_state를 안 고정하면 추출·특성 후보가 매번 달라져 점수도 달라짐. n_jobs=-1이면 CPU 전부 |

💡 이번 자료에서는 단일 Tree(train 0.710 / valid 0.333)보다 Forest(train 0.800 / valid 0.222)가 낮았어요. 앙상블이라는 이름이나 트리 수가 아니라 같은 평가 조건의 결과와 오류로 판단해요.

🤔 왜 숲이 더 나빴을까? (가능한 이유) — 숲의 장점은 "트리마다 다른 실수를 평균으로 줄이는 것"인데, 자료가 60건뿐이고 실패가 소수라 각 트리가 볼 실패 사례가 더 적어요. 확률을 평균하면 실패 확률이 0.5 아래로 무난하게 깎이기 쉬워 실패를 덜 예측하게 되고(FN 5), F1이 떨어졌어요. 앙상블은 흔들림을 줄여 줄 뿐, 없는 신호를 만들어 주진 못해요.

### feature_importances_ · permutation_importance

```
import pandas as pd
from sklearn.inspection import permutation_importance

names = forest["pre"].get_feature_names_out()            # 전처리 후 실제 열 이름·순서
mdi = pd.Series(forest["model"].feature_importances_, index=names)
print(mdi.sort_values(ascending=False))                  # input_chars .5643, allowed_tools .1670 …
print("합:", round(mdi.sum(), 4))                         # 1.0

perm = permutation_importance(forest, X_valid, y_valid,  # Pipeline 통째 + 원본 4열
                              scoring="f1", n_repeats=20, random_state=42, n_jobs=1)
importance = pd.DataFrame({"feature": FEATURES,
                           "F1_drop_mean": perm.importances_mean,
                           "repeat_std": perm.importances_std})
print(importance)    # 기준 F1 0.222 · chars .0649 · tools .0478 · task_type .1396 · priority −.0619
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 · 위 Forest

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(names), names[:2] | 8, ['num__input_chars', 'num__allowed_tools'] | 원핫으로 4열 → 8열 |
| mdi 상위 2개 | input_chars 0.5643 · allowed_tools 0.1670 | train에서 분할에 많이 쓰인 열 |
| mdi.sum() | 1.0000 | MDI는 합이 1로 정규화 |
| perm.importances_mean (원본 4열) | input_chars +0.0649 · allowed_tools +0.0478 · task_type +0.1396 · priority -0.0619 | valid F1 감소량 — 1위가 task_type으로 MDI와 다름 |
| perm.importances_std | 0.123 · 0.128 · 0.114 · 0.150 | 평균만큼 큰 흔들림 — valid 20건이라 |

🤔 왜 두 방법의 순위가 다르지? — MDI는 학습 자료에서 "분할에 많이 쓰인" 열(input_chars 1위)을, 순열 중요도는 valid 20건에서 "섞었을 때 F1이 떨어진" 열(task_type 1위)을 봐요. 보는 자료도 질문도 달라서 순위가 다를 수 있어요. priority가 음수(−.0619)라는 건 섞었더니 오히려 F1이 올랐다는 뜻인데, 20행짜리 평가에선 우연히 그럴 수 있어요 → 바로 삭제할 근거도, 인과 효과도 아니에요.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| get_feature_names_out() | 전처리 뒤 실제 열 이름·순서 (8개) | 안 쓰면 중요도 숫자가 어느 열 것인지 모름 — One-hot으로 열이 늘어나 원본 4열과 순서가 달라지니까 |
| feature_importances_ | 학습 불순도 기반 기여 (MDI) | 이 숲에선 합이 1. 분기가 전혀 없는 모델이면 모두 0일 수 있음 |
| scoring="f1" | 실패 클래스 F1의 감소량을 봄 | 지표가 바뀌면 "무엇이 중요한가"의 기준이 바뀌니 중요도도 바뀜 |
| n_repeats=20 | 20번 섞어서 평균·변동 기록 | 섞는 방식이 무작위라 한 번만 하면 운에 좌우됨 → 늘리면 안정적, 느려짐 |
| importances_mean / _std | 감소량 평균 / 반복 간 표준편차 | std는 "섞을 때마다의 흔들림"이지 신뢰구간이 아님 |

### XGBClassifier · XGBRegressor

```
from xgboost import XGBClassifier, XGBRegressor

xgb = XGBClassifier(n_estimators=80, learning_rate=0.05, max_depth=2,
                    min_child_weight=1, subsample=0.8, colsample_bytree=1.0,
                    reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", device="cpu", random_state=42, n_jobs=1)
model = build(xgb).fit(X_train, y_train)           # sklearn Pipeline에 그대로 연결
# valid F1 0.182 · Precision 0.200 · Recall 0.167 · FP=4 · FN=5

xgr = XGBRegressor(n_estimators=80, learning_rate=0.05, max_depth=2,
                   min_child_weight=1, subsample=0.8, colsample_bytree=1.0,
                   reg_lambda=1.0, objective="reg:squarederror", eval_metric="rmse",
                   tree_method="hist", device="cpu", random_state=42, n_jobs=1)
model_r = build(xgr).fit(X_train, yr_train)        # X·분할 ID는 동일, 정답 y만 elapsed_ms로
# valid MAE 609.34 ms · RMSE 700.02 ms · R² −0.312
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model["model"].get_booster().num_boosted_rounds() | 80 | 보정 회차 = n_estimators |
| valid F1 · Precision · Recall · FP · FN | 0.200 · 0.250 · 0.167 · 3 · 5 | 노트·강의 결과(F1 .182 · P .200 · FP 4)와 조금 달라요 — 같은 버전(XGBoost 3.1.3)인데 이 PC에선 valid 1행의 판정이 바뀜. 그래도 Tree(.333)·Forest(.222)보다 낮은 건 같음 |
| train F1 | 0.774 | 강의 결과와 같음 · Forest(.800)보다 조금 낮음 |
| predict_proba(X_valid)[:, 1] 범위 · 0.5 초과 행 수 | 0.039 ~ 0.661 · 4건 | 실제 실패는 6건인데 실패라 부르는 행이 적음 |
| XGBRegressor: valid | MAE 610.20 · RMSE 700.41 · R² -0.313 | 노트(MAE 609.34 · RMSE 700.02)와 소수점 차이 — 위와 같은 환경 차이. R²는 여전히 음수 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| n_estimators | 보정 회차 (트리 수) · 수업 80 | 회차마다 train 오차를 더 줄이니, 너무 많으면 잡음까지 보정해 과적합 위험 (RF의 트리 수와 성격이 다름) |
| learning_rate | 한 회차 보정을 얼마나 반영할지 · 0.05 | 줄이면 한 번에 조금만 고쳐 같은 회차에선 보정이 부족할 수 있음 → 회차·복잡도와 함께 비교 |
| max_depth | 각 보정 트리의 깊이 · 2 | 보정 트리는 "조금씩 고치는" 역할이라 얕게 둬요. 깊으면 한 회차가 너무 많이 외움 |
| min_child_weight | 자식 노드에 필요한 Hessian 합의 하한 · 1 | min_samples_leaf(행 수)와 다름 — 분류(logloss)에선 행마다 Hessian이 p(1−p) ≤ 0.25라, 5라는 값이 "5행"이 아니라 훨씬 많은 행을 뜻할 수 있음 |
| subsample / colsample_bytree | 회차마다 쓸 행 / 트리별 특성 비율 · 0.8 / 1.0 | 낮추면 회차마다 다른 일부만 보니 다양성↑·과적합↓, 너무 낮으면 정보 부족 |
| reg_lambda | 잎 가중치 L2 규제 · 1.0 | 키우면 잎 값이 작게 눌려 한 번의 보정이 보수적이 됨 |
| objective | "binary:logistic" 이진분류 / "reg:squarederror" 회귀 | 학습이 줄이는 손실 자체가 바뀜 — 문제 유형에 맞춰야 함 |
| eval_metric | 학습 진단용 손실 (logloss, rmse) | 모델 선택 지표(F1)와 별개. logloss가 좋아져도 F1이 좋아진다는 보장은 없음 |
| tree_method / device / n_jobs | "hist" 방식 / "cpu" / 1 | hist는 값을 구간으로 묶어 빠르게 분할 탐색. "cuda"는 GPU 필요 |

🤔 왜 "Hessian 합"이지? — XGBoost는 손실의 기울기(gradient)와 곡률(Hessian)로 보정값을 계산해요. 제곱오차에선 행마다 Hessian이 1이라 "합 = 행 수"처럼 보이지만, logloss에선 확률이 0이나 1에 가까운(이미 확신하는) 행일수록 Hessian이 작아요. 그래서 min_child_weight는 "보정할 정보가 충분히 모였나"를 재는 조건이에요.

💡 조기 종료(early stopping) — 손실 개선이 일정 회차(예: 15회) 동안 없으면 멈추는 방법. 강의 시연에선 train 60을 inner 42 / stop 18로 나눠 최적 회차 64, 실제 학습 79회(15회 개선 없어 종료)였고, 기본 비교의 XGB80과는 별도 실험으로 기록해요. 멈출 지점을 고르는 stop 자료도 모델 선택에 쓰이므로 train 안에서 따로 떼어 쓰고, valid·test를 넣지 않아요. XGBClassifier(early_stopping_rounds=15) + fit(..., eval_set=[(Z_stop, y_stop)]) → model.best_iteration(0부터 셈)으로 확인해요.

🤔 왜 조기 종료가 과적합을 막지? — 회차가 늘수록 train 손실은 계속 줄지만, 어느 순간부터는 잡음을 보정하기 시작해 안 본 자료(stop)의 손실이 더 이상 줄지 않아요. 그 지점에서 멈추면 "배울 만큼만" 배운 모델이 돼요. 단, stop 자료로 멈출 곳을 골랐으니 그 자료의 점수는 이미 낙관적이에요 → valid·test와 분리.

| 모델 | 과적합을 제어하는 것 | 진단할 증거 |
|---|---|---|
| Decision Tree | 깊이·잎 크기·가지치기 | train/valid 격차 · 잎 개수 |
| Random Forest | 개별 트리 복잡도·특성 후보·트리 수 | 반복 결과 · 계산량 · fold별 편차 |
| XGBoost | 깊이·규제·학습률·회차·부분 표본 | 손실 변화 · stop 자료 · 과도한 학습량 |

⚠️ 이번 자료에서 XGB의 valid F1은 0.182였어요. 학습률을 낮추거나 규제를 강하게 해도 항상 좋아지지 않아요. 규제·단순화도 과소적합과의 균형을 비교로 확인해요.

### StratifiedKFold · KFold · cross_validate

```
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate

CLS_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # 분류: 실패 비율 유지
REG_CV = KFold(n_splits=5, shuffle=True, random_state=42)             # 회귀: 연속 y

cv = cross_validate(build(rf), X_train, y_train, cv=CLS_CV, scoring="f1",
                    return_train_score=True, return_estimator=True,
                    n_jobs=1, error_score="raise")
print(cv["test_score"])                 # fold별 F1  [.333 .000 .667 .400 .000]
print(cv["test_score"].mean(), cv["test_score"].std(ddof=0))   # 평균 .280, 편차 .254

print(cv["train_score"].mean())         # train 평균 — fold 점수와의 격차로 과적합 진단

cvr = cross_validate(build(xgr), X_train, yr_train, cv=REG_CV,
                     scoring="neg_root_mean_squared_error")
rmse = -cvr["test_score"]               # 부호 되돌리기 → ms

# 각 fold의 중앙값이 그 fold의 학습 48건에서 왔는지 검사 (E16)
cv = cross_validate(build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42),
                          scale="none"),
                    X_train, y_train, cv=CLS_CV, scoring="f1",
                    return_estimator=True, error_score="raise")
for (a, b), m in zip(CLS_CV.split(X_train, y_train), cv["estimator"]):   # a=학습 행, b=평가 행
    actual = m["pre"].named_transformers_["num"]["imputer"].statistics_
    expected = X_train.iloc[a][NUM].median()
    assert np.allclose(actual, expected)    # 근사 일치
    assert not set(a) & set(b)              # 학습·평가 교집합 0
print("전 fold 중앙값·분할 검사 통과")
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 (fold 학습 48 / 평가 12)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv["test_score"] (Forest F1) | [0.333, 0.000, 0.667, 0.400, 0.000] | 실패 몇 건으로 0 ~ 0.667까지 출렁 |
| mean() · std(ddof=0) | 0.280 · 0.254 | 편차가 평균과 비슷할 만큼 큼 — 평균 하나로 요약하면 안 되는 이유 |
| cv["train_score"].mean() | 0.753 | fold 평가 점수와 큰 격차 → 과적합 신호 |
| -cvr["test_score"] (XGB 회귀 RMSE) | [756.8, 614.7, 721.2, 668.3, 493.4] | 부호를 되돌려 ms 단위로 |
| 평가 fold의 실패 수: Stratified vs KFold | [4, 4, 4, 3, 3] vs [5, 5, 4, 2, 2] | Stratified는 3~4건으로 고르게 |
| fold별 중앙값(글자수) · 검사 통과 | [1309.0, 1303.0, 1315.0, 1315.0, 1338.0] · True | fold마다 자기 학습 48건의 중앙값 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| n_splits=5 | 몇 조각으로 나눌지 | 늘리면 회차마다 학습 자료↑(평가 조각은 작아져 점수가 더 출렁), 계산↑ |
| shuffle=True, random_state | 섞어서 나누기 + 재현 | 안 섞으면 원본 행 순서대로 잘려 한쪽에 특정 유형이 몰릴 수 있음. 모든 후보가 같은 fold를 써야 공정 |
| scoring | fold 평가 지표 | 회귀는 "neg_root_mean_squared_error" — sklearn은 "클수록 좋은 점수"로 통일하기 때문에 RMSE에 −를 붙여 돌려줌 → 다시 −를 붙여 ms로 |
| return_train_score=True | 학습 점수도 함께 | train–fold 격차로 과적합 진단 |
| return_estimator=True | fold별 학습된 Pipeline 반환 | fold마다 중앙값이 학습 48건에서 왔는지 검사 가능 |
| error_score="raise" | fold 실패 시 바로 에러 | 기본값은 NaN으로 조용히 넘어가서, 실패한 fold를 모르고 평균을 낼 수 있음 |
| std(ddof=0) | 5개 점수의 기술적 편차 | "점수가 얼마나 흩어졌나"의 기록일 뿐 95% 신뢰구간이 아님 |

🤔 왜 분류는 StratifiedKFold지? — 실패는 소수라 그냥 무작위로 12건씩 자르면 어떤 fold엔 실패가 거의 없을 수 있어요. 그러면 그 fold의 F1은 계산 자체가 불안정해져요. Stratified는 모든 fold의 실패 비율을 원본과 비슷하게 맞춰 줘요. 회귀의 연속 정답은 "비율"을 맞출 클래스가 없으니 그냥 KFold를 써요.

🤔 왜 fold F1이 0 ~ 0.667로 크게 흔들리지? — 평가 조각이 12건, 그중 실패는 몇 건뿐이라 실패 한두 개를 맞히느냐 마느냐로 F1이 0과 0.667 사이를 오가요. 그래서 평균 하나(.280)로 가리지 말고 fold 점수·평균·편차(.254)를 함께 남겨요.

⚠️ cross_validate의 test_score는 "fold 평가 점수"라는 이름일 뿐, 보관한 test 20이 아니에요. 그리고 평균 ± 표준편차는 95% 신뢰구간이 아니에요.

| 상황 | 분할 방법 | 주의 |
|---|---|---|
| 분류 · 독립 요청 | StratifiedKFold(5, shuffle=True) | 실패 비율 유지 |
| 회귀 · 독립 요청 | KFold(5, shuffle=True) | 연속 정답은 그대로 stratify 안 함 |
| 같은 요청의 여러 단계 | 그룹 단위 분할 | 같은 ID가 학습·평가를 가로지르지 않게 |
| 미래 요청 예측 | 시간 순서 분할 | 미래 정보가 학습에 들어가지 않게 |

### 새 범주 · 통계 불변 검사

```
model = build(new_classifier("Forest")).fit(X_train, y_train)
imp = model["pre"].named_transformers_["num"]["imputer"]
before = imp.statistics_.copy()
new = X_valid.iloc[:2].copy()
new["task_type"], new["priority"] = "translation", "urgent"   # 학습에 없던 범주
model.predict(require_features(new))          # require_features: 필수 열이 없으면 오류
assert np.array_equal(before, imp.statistics_) # 예측해도 학습 통계 불변
print(model["pre"].transform(new).shape)       # (2, 8)  ← 기존 8열 유지
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 · valid 앞 2행의 범주를 바꿈

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model.predict(require_features(new)) | [0, 0] | 처음 보는 범주여도 예측은 나옴 |
| model["pre"].transform(new).shape | (2, 8) | 기존 8열 유지 |
| 원핫 6열 부분 Zn[:, 2:] | [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]] | handle_unknown="ignore" → 범주 칸이 전부 0 |
| np.array_equal(before, imp.statistics_) | True | 예측해도 학습 통계 불변 |
| require_features(new.drop(columns="priority")) | ValueError: 필수 열 누락: ['priority'] | 필수 열 누락은 예측 전에 명확한 오류 |
| 검사 없이 model.predict(…priority 없음) | ValueError: columns are missing: {'priority'} | 검사가 없어도 여기선 에러 — 다만 메시지가 덜 친절함 |

### 저장 · 복원 대조

```
from pathlib import Path
import tempfile, joblib
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "selected_pipeline.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)              # 자신이 방금 만든 파일만
    same = np.array_equal(model.predict(X_valid), restored.predict(X_valid))
    print("복원 예측 동일:", same)             # True
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 · Forest

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| joblib.dump(model, path) | 파일 약 175 KB | 트리 100개를 통째로 저장 |
| np.array_equal(model.predict(X_valid), restored.predict(X_valid)) | True | 예측 20개 일치 |
| np.allclose(predict_proba 두 개) | True | 확률까지 일치 |

| 인자 / 함수 | 뜻 | 바꾸면? |
|---|---|---|
| require_features(df) | 강의 코드의 사용자 정의 함수 — 필수 4열 확인 | 없으면 누락 열이 imputer까지 가서 엉뚱한 오류가 나거나 조용히 잘못 처리될 수 있음 |
| transform(new).shape | 변환 후 열 수 확인 | 8이 아니면 학습 때와 다른 입력 구조라 모델이 받을 수 없음 |
| joblib.load | 직접 만든 신뢰하는 파일만 | 버전·입력 스키마도 함께 기록. 다른 환경에서의 호환은 보장 안 됨 |

💡 "오류 없이 끝났다"와 "요구사항대로 동작했다"는 달라요. 정상 valid 20(20개 예측·ID 순서), 결측·미지 범주(8열·통계 불변), 필수 열 누락(명확한 오류), CV 한 fold(학습 48·평가 12·교집합 0), 저장 모델(복원 후 예측 일치)을 따로 확인해요. 저장·복원의 자세한 설명은 Day6 ⑦.

### 부스팅 한 회차 직접 계산 DecisionTreeRegressor

```
X = np.array([[1], [2], [3], [4]])
y = np.array([100., 120., 200., 220.])
initial = np.full(4, y.mean())                       # 처음 예측: 모두 160
residual = y - initial                               # 제곱오차에서의 보정 목표
correction = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, residual)
updated = initial + 0.5 * correction.predict(X)      # 학습률 0.5 = 보정량의 절반만
print(updated.tolist())                              # [135.0, 135.0, 185.0, 185.0]
print(np.mean((y - initial)**2), np.mean((y - updated)**2))   # 2600.0 725.0
```

▶ 실제로 돌려 보면 노트의 4개 예제

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| residual = y - initial | [-60.0, -40.0, 40.0, 60.0] | 처음 예측 160에서 얼마나 틀렸나 |
| correction.predict(X) | [-50.0, -50.0, 50.0, 50.0] | 깊이 1 트리: x ≤ 2.5면 −50, 아니면 +50 |
| initial + 0.5 * correction.predict(X) | [135.0, 135.0, 185.0, 185.0] | 보정을 절반(학습률 0.5)만 반영 |
| MSE 전 → 후 | 2600 → 725 | 한 회차 만에 오차가 줄어요 |
| initial + 1.0 * correction.predict(X) | [110.0, 110.0, 210.0, 210.0] | 학습률 1이면 MSE 100 — 이 4개엔 더 맞지만 잡음까지 따라갈 위험 |

## 📖 Day7 용어 (47개)
- **과적합** (Overfitting (과적합)): 학습 자료의 우연한 잡음까지 외워서 train 점수만 높고 새 자료(valid) 점수는 낮은 상태. 두 점수 격차가 핵심 신호예요.
- **Decision Tree** (Decision Tree (결정 트리)): "글자수 ≤ 500?" 같은 예/아니오 질문을 이어 가며 자료를 나누고, 마지막 잎에서 예측하는 모델. 질문은 학습 자료로 자동 선택돼요.
- **루트** (Root node (루트 노드)): 트리의 맨 위, 첫 번째 질문이 있는 곳.
- **내부 노드** (Internal node (내부 노드)): 루트 다음에 이어지는 중간 질문들. 자료를 다시 둘로 나눠요.
- **잎** (Leaf node (잎 노드)): 더 이상 나누지 않는 트리의 끝. 분류는 그 잎에 떨어진 학습 행의 클래스 비율, 회귀는 y 평균으로 예측해요.
- **Gini 불순도** (Gini impurity (지니 불순도)): 한 노드에 클래스가 얼마나 섞였는지 재는 값. 1 − Σ(각 클래스 비율)². 0이면 한 클래스만(깨끗), 이진 분류에선 0.5가 가장 섞인 상태. 트리는 이 값을 가장 많이 줄이는 질문을 골라요.
- **불순도** (Impurity (불순도)): 노드 안에 서로 다른 정답이 섞인 정도. 낮을수록 그 노드의 예측을 자신 있게 할 수 있어요. 시험 점수나 오차가 아니에요.
- **탐욕적 방법** (Greedy method (탐욕적 방법)): 전체 최적을 한 번에 찾지 않고, 매 단계에서 '지금 가장 좋아 보이는' 선택을 하고 넘어가는 방식. 빠르지만 나중에 더 좋은 길은 놓칠 수 있어요.
- **max_depth** (max_depth (최대 깊이)): 트리가 질문을 최대 몇 단계까지 할 수 있는지. 작으면 과소적합, None(제한 없음)이면 잎이 깨끗해질 때까지 나눠 과적합 위험.
- **min_samples_leaf** (min_samples_leaf (잎의 최소 행 수)): 각 잎에 있어야 하는 최소 학습 행 수. 키우면 행 1~2개짜리 잎(잡음 분할)을 막아요. '잎 개수'가 아니에요.
- **min_samples_split** (min_samples_split (분할 최소 행 수)): 노드를 더 나누려면 그 노드에 최소 몇 행이 있어야 하는지. '나눌지 말지' 조건이에요.
- **가지치기** (Cost-Complexity Pruning (비용-복잡도 가지치기)): 다 키운 트리에서 불순도를 별로 못 줄이는 가지를 잘라 단순하게 만드는 것. ccp_alpha가 클수록 더 많이 잘려요.
- **seed** (random_state (난수 고정값)): 무작위 과정(추출, 특성 후보, 분할 섞기 등)을 매번 같게 만드는 번호. 결과를 재현하려고 고정해요(예: 42).
- **앙상블** (Ensemble (앙상블)): 여러 모델(여기선 트리 여러 개)의 예측을 묶어 하나의 답을 내는 방법. 대표 방식이 배깅(Random Forest)과 부스팅(XGBoost). 항상 점수가 오르는 건 아니에요.
- **Random Forest** (Random Forest (랜덤 포레스트)): 트리마다 다른 표본(중복 허용 추출)과 분할마다 다른 특성 후보를 줘서 서로 다른 트리를 독립적으로 여러 개 만들고, 분류는 확률·회귀는 예측을 평균하는 앙상블.
- **Bagging** (Bootstrap Aggregating (배깅)): 중복 허용 추출로 만든 서로 다른 표본에서 모델을 각각 독립적으로 학습하고 결과를 평균하는 앙상블 방식. Random Forest가 대표.
- **중복 허용 추출** (Bootstrap sampling (중복 허용 추출)): 자료에서 하나 뽑고 다시 넣고를 반복해 같은 크기의 표본을 만드는 방법. 어떤 행은 여러 번, 어떤 행은 한 번도 안 뽑혀서 트리마다 보는 자료가 달라져요.
- **n_estimators** (n_estimators (트리 개수 / 보정 회차)): Random Forest에선 숲의 트리 수, XGBoost에선 보정 트리를 추가하는 회차. 깊이가 아니에요.
- **max_features** (max_features (분할마다 볼 특성 후보 수)): Random Forest가 분할할 때마다 무작위로 살펴볼 특성의 수·비율. 작을수록 트리끼리 더 달라지지만 각 트리는 약해져요.
- **estimators_** (estimators_ (학습된 개별 트리 목록)): Random Forest 안에 학습된 트리들을 담은 목록. n_estimators=100이면 100개.
- **n_jobs** (n_jobs (병렬 작업 수)): 계산에 쓸 CPU 작업 수. -1이면 전부. 수업에선 재현성을 위해 1로 고정해요.
- **특성 중요도** (Feature importance (특성 중요도)): 학습된 모델이 어떤 특성(열)에 얼마나 기댔는지 나타낸 값. '중요 = 원인'이 아니고, 영향의 방향(+/−)도 알려 주지 않아요.
- **MDI** (Mean Decrease in Impurity (불순도 감소 기반 중요도)): 각 분기에서 줄어든 불순도 × 그 분기가 다룬 자료 비중을 특성별로 더한 값(트리들 평균). 학습 자료 기준이고 합은 1. 값 종류가 많은 특성이 높게 나오기 쉬워요.
- **순열 중요도** (Permutation importance (순열 중요도)): 평가 자료에서 한 열의 값을 행끼리 뒤섞었을 때 점수가 얼마나 떨어지는지로 중요도를 재는 방법. 다시 학습하지 않으며 음수도 나올 수 있어요.
- **get_feature_names_out** (get_feature_names_out() (변환 후 열 이름)): 전처리를 거친 뒤 실제 열 이름과 순서를 돌려주는 함수. 원-핫으로 열이 늘어나니 중요도가 어느 열 것인지 알려면 필요해요.
- **더미** (Dummy variable (더미 열)): 원-핫 인코딩으로 생긴 0/1 열. 범주 2열이 더미 6열로 펼쳐졌어요.
- **부스팅** (Boosting (부스팅)): 앞 모델이 틀린 부분을 다음 모델이 보정하도록 트리를 순서대로 추가하는 앙상블 방식. 기본 점수에 보정값을 조금씩 누적해요.
- **XGBoost** (eXtreme Gradient Boosting): 손실을 줄이는 트리를 순서대로 추가하는 그래디언트 부스팅 구현. 트리 복잡도·잎 값에 규제를 걸어 과적합을 제어하고, sklearn처럼 fit/predict로 Pipeline에 연결돼요.
- **그래디언트 부스팅** (Gradient Boosting (그래디언트 부스팅)): 손실의 기울기(gradient) 정보를 따라 '지금 틀린 방향'을 다음 트리가 배우게 하는 부스팅. 제곱오차에선 그 방향이 곧 잔차예요.
- **학습률** (Learning rate (학습률)): 한 회차의 보정을 얼마나 반영할지 정하는 비율(예: 0.05). 작으면 조금씩 안정적으로 고치지만 회차가 더 필요해요.
- **Hessian** (Hessian (손실의 곡률, 2차 미분)): 손실이 얼마나 휘어 있는지 나타내는 값. XGBoost는 기울기와 함께 이걸로 보정값을 계산하고, min_child_weight는 자식 노드의 Hessian 합 하한이에요.
- **min_child_weight** (min_child_weight (자식 노드 최소 Hessian 합)): XGBoost에서 노드를 나눌 때 자식에 보정할 정보(Hessian 합)가 이만큼은 모여야 한다는 조건. 행 수(min_samples_leaf)와는 달라요.
- **subsample** (subsample / colsample_bytree (행·특성 부분 표본 비율)): XGBoost가 회차마다 쓸 행의 비율 / 트리마다 쓸 특성의 비율. 낮추면 다양성↑·과적합↓, 너무 낮으면 정보 부족.
- **L2 규제** (L2 regularization (L2 규제)): 값의 제곱에 벌점을 줘 크게 튀지 않게 하는 규제. XGBoost의 reg_lambda는 잎 값에 걸어 한 번의 보정을 보수적으로 만들어요.
- **objective** (objective (학습 목표 손실)): XGBoost가 학습하며 줄일 손실의 종류. 이진분류는 binary:logistic, 회귀는 reg:squarederror. 문제 유형에 맞춰야 해요.
- **eval_metric** (eval_metric (학습 진단 지표)): 학습 중 진행을 확인하는 지표(logloss, rmse). 모델 선택 지표(F1)와는 별개예요.
- **hist** (tree_method="hist" (히스토그램 방식)): 값을 구간으로 묶어 분할 기준을 빠르게 찾는 XGBoost 트리 생성 방식. CPU에서 돌아가요.
- **GPU** (Graphics Processing Unit (그래픽 처리 장치)): 대량 계산을 병렬로 빠르게 하는 장치. CUDA는 NVIDIA GPU를 쓰기 위한 도구. 이번 수업은 CPU만으로 충분해요.
- **조기 종료** (Early stopping (조기 종료)): 따로 떼어 둔 stop 자료의 손실이 정한 회차(예: 15회) 동안 나아지지 않으면 학습을 멈추는 방법. 잡음까지 배우기 전에 멈춰 과적합을 줄여요.
- **교차검증** (Cross-validation (교차검증)): train을 여러 조각(fold)으로 나눠, 돌아가며 한 조각으로 평가하고 나머지로 학습하는 방법. 한 번의 분할 운에 덜 흔들리고, 점수 평균과 흩어짐을 함께 봐요. 전처리도 fold마다 다시 학습해야 해요.
- **fold** (Fold (교차검증 조각)): 교차검증에서 나눈 한 조각. 5-fold면 train 60건을 12건씩 5조각으로 나눠 각 회차에 48건 학습, 12건 평가.
- **StratifiedKFold** (Stratified K-Fold (층화 K겹 분할)): 모든 fold의 클래스 비율(예: 실패 비율)을 원본과 비슷하게 맞춰 나누는 분할. 소수 클래스가 있는 분류에 써요.
- **KFold** (K-Fold (K겹 분할)): 자료를 K조각으로 그냥 나누는 교차검증 분할. 클래스가 없는 회귀(연속 정답)에 써요. shuffle=True면 섞어서 나눠요.
- **neg_root_mean_squared_error** (Negative RMSE (음수 RMSE 점수)): sklearn이 모든 점수를 '클수록 좋다'로 통일하려고 RMSE에 −를 붙인 것. 보고할 땐 다시 −를 붙여 원래 RMSE(ms)로 되돌려요.
- **그룹 단위 분할** (Group split (그룹 단위 분할)): 같은 고객·요청 등 한 그룹의 행이 학습과 평가 양쪽에 동시에 들어가지 않게 그룹째로 나누는 방법.
- **시간 순서 분할** (Time-ordered split (시간 순서 분할)): 과거 자료로 학습하고 그 뒤의 자료로 평가하도록 시간 순서대로 나누는 방법. 미래 정보가 학습에 들어가지 않게 해요.
- **신뢰구간** (Confidence interval (신뢰구간)): 진짜 값이 들어 있을 것으로 추정하는 통계적 범위. fold 점수의 평균 ± 표준편차는 흩어짐 기록일 뿐 신뢰구간이 아니에요.

---

# Day8 · 평가지표·특성설계·K-Means
_2026-10-28(수) · Precision/Recall/F1 · ROC-AUC · 교차검증 · Feature Engineering · K-Means · 최종 평가_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day08.html

📌 딱 3줄 요약

- 지표는 분모가 다르다: Precision = 경고 중 진짜, Recall = 진짜 중 찾은 것. AUC는 순위 평가, 임계값은 운영 규칙.

- 새 특성(log, 비율)과 K-Means도 Pipeline 안에서, 학습 자료로만 fit해야 누수가 없다.

- 선택은 train CV로 → 설정을 잠그고 → 개발 80건으로 재학습 → test 20건은 딱 한 번 평가.

🔧 Day8 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| scikit-learn | 지표·ROC·교차검증·FunctionTransformer·KMeans | from sklearn.metrics import roc_auc_score 등 |
| numpy | log1p 같은 행별 계산, 배열 비교 | import numpy as np |
| pandas | DataFrame에 새 특성 열 추가, 결과 CSV | import pandas as pd |
| xgboost (Day7에서 설치) | 후보 모델 중 하나로 계속 비교 | from xgboost import XGBClassifier |
| matplotlib (선택) | ROC 곡선·군집 그림 그리기 | import matplotlib.pyplot as plt |
| 표준 라이브러리 | 설정 파일 저장, 경로 | import json, from pathlib import Path |

### 🔧 설치 · 확인

```
python --version
python -m pip show scikit-learn xgboost
python -m pip show numpy pandas matplotlib
python -c "import sys; print(sys.executable)"     # 지금 쓰는 파이썬이 .venv인지 확인
```

⚠️ 전날 환경(.venv) 그대로. 설치가 필요한 경우에만 강사 검증 버전(Python 3.13.5 · NumPy 2.3.5 · pandas 2.2.3 · scikit-learn 1.8.0 · XGBoost 3.1.3 · Matplotlib 3.10.8)에 맞춰요. 오늘은 GPU가 필요 없고, CUDA·PyTorch를 임의로 업데이트하지 않아요. 최종 제출에는 실제 설치 버전을 기록해요.

🤔 왜 sys.executable을 확인하지? — 컴퓨터에 파이썬이 여러 개 있으면, 패키지를 깔은 파이썬과 코드를 돌리는 파이썬이 다를 수 있어요. 지금 실행 중인 파이썬의 경로가 프로젝트 .venv 안인지 보면 "설치했는데 import가 안 돼요" 문제를 바로 잡을 수 있어요.

💡 오늘의 평가 계획: train 60은 5-fold로 후보 비교(전처리도 fold 안에서) · valid 20은 개발 중 진단(최종 독립 점수 아님) · 설정 확정 후 새 Pipeline을 개발 80건으로 재학습 · test 20은 고정 버전으로 한 번 평가. 기본 경로는 실패 클래스 F1로 선택, 임계값은 처음부터 0.5로 고정해요.

## 이거 어디에 써?

"좋은 모델"의 기준은 업무마다 달라요. 실패를 놓치면 큰일인지, 괜한 경고가 비싼지에 따라 볼 지표가 바뀌어요.

🏥

놓치면 큰일질병·장애 탐지 → Recall 우선

📨

경고가 비쌈사람이 확인해야 하는 알림 → Precision

🧩

정답 없이 묶기비슷한 요청·고객끼리 그룹 (K-Means)

🔒

최종 보고선택을 잠그고 test로 한 번 확인

| 상황 | 우선 검토 | 같이 확인할 것 |
|---|---|---|
| 실패를 놓치면 손실이 크다 | Recall | 오탐 수 · 처리 인력 · Precision |
| 경고마다 사람의 확인 비용이 든다 | Precision | 놓친 실제 실패 수 · Recall |
| 두 오류를 함께 줄이고 싶다 | F1 | FP · FN 건수와 실제 비용 |
| 실행 시간(숫자)을 예측한다 | MAE · RMSE | 단위(ms) · 큰 잔차 · 자료 범위 |

🤔 왜 지표를 먼저 정하지? — 실패를 놓치는 것(FN)과 괜한 경고(FP)는 서로 다른 비용이에요. 어떤 실수가 더 아픈지 정하지 않고 결과부터 보면, 점수가 좋게 나온 지표를 골라 "잘했다"고 말하게 돼요. 지표 선택은 정답 하나를 찾는 게 아니라 실패의 비용과 운영 조건을 적는 일이에요.

## ① 분류 지표 — 분모가 다르다

양성 = 실패(1). 행 = 실제, 열 = 예측. (설명용 20건)

|  | 예측 0 · 성공 | 예측 1 · 실패 |
|---|---|---|
| 실제 0 · 성공 | TN 10 · 정상 통과 | FP 4 · 오탐 |
| 실제 1 · 실패 | FN 4 · 놓침 | TP 2 · 실패 발견 |

| 지표 | 공식 | 질문 | 예시 |
|---|---|---|---|
| Accuracy | (TP + TN) / 전체 | 전체 중 몇 개 맞혔나? | 12/20 = 0.600 |
| Precision | TP / (TP + FP) | "실패"라고 경고한 것 중 진짜는? (오탐↑ → 낮아짐) | 2/6 = 0.333 |
| Recall | TP / (TP + FN) | 진짜 실패 중 몇 개 찾았나? (놓침↑ → 낮아짐) | 2/6 = 0.333 |
| F1 | 2·P·R / (P + R) = 2TP / (2TP + FP + FN) | 둘의 균형 (조화평균) | 4/12 = 0.333 |

💡 Precision과 Recall이 둘 다 0.333이어도 의미가 달라요. "경고 6건"과 "실제 실패 6건"은 서로 다른 집합이에요. F1은 조화평균이라 한쪽이 낮으면 같이 낮아지고, TN은 반영하지 않아요.

⚠️ 정확도 함정 — 실제 성공 14 · 실패 6인데 전부 "성공"으로 찍으면 Accuracy = 0.70, Recall = 0, F1 = 0. "60% 정확도"와 "실패 60% 발견"은 전혀 다른 말이에요.

🤔 왜 분모가 중요하지? — Precision의 분모는 "내가 실패라고 말한 것"(TP+FP)이라 헛경고가 늘면 떨어지고, Recall의 분모는 "실제로 실패였던 것"(TP+FN)이라 놓침이 늘면 떨어져요. 같은 TP 2건이라도 어느 집합으로 나누느냐에 따라 질문 자체가 달라요. Accuracy는 분모가 전체(20)라, 많은 쪽(성공 14)만 맞혀도 점수가 올라가서 드문 실패를 찾는 문제에선 속기 쉬워요.

[그림: 혼동행렬에서 Precision은 예측 1 열, Recall은 실제 1 행을 분모로 쓰는 그림] 무엇을 보나: 같은 혼동행렬을 두 번 그리고, 각 지표가 쓰는 칸만 색칠했어요. Precision은 세로 열(예측 1 = 경고한 것: FP+TP), Recall은 가로 행(실제 1 = 진짜 실패: FN+TP)이에요. 🤔 왜? 둘 다 분자는 TP 2로 같지만, 나누는 집합이 "내가 경고한 6건"이냐 "실제 실패 6건"이냐로 달라요. 그래서 FP가 늘면 왼쪽만, FN이 늘면 오른쪽만 나빠져요. 회색 TN은 어느 쪽에도 안 들어가요.

🤔 왜 F1은 산술평균이 아니라 조화평균이지? — Precision 1.0, Recall 0.01인 모델(경고 1번만 해서 맞힘)을 산술평균하면 0.505로 그럴듯해 보여요. 조화평균은 2×1×0.01/(1+0.01) ≈ 0.02예요. 한쪽이 낮으면 같이 낮아지게 만들어서, 한 지표만 극단적으로 올리는 꼼수를 막아요.

[그림: Precision을 1로 고정했을 때 Recall에 따른 산술평균과 F1 비교] 무엇을 보나: Precision을 1.0에 고정하고 Recall만 바꿨을 때의 두 평균이에요. 회색 점선(산술평균)은 Recall이 0이어도 0.5에서 시작하지만, 보라 F1은 0에서 시작해요. 🤔 왜? 조화평균은 작은 값에 강하게 끌려가요. 한쪽 지표가 0에 가까우면 다른 쪽이 아무리 높아도 F1이 바닥이라, "경고 한 번만 해서 Precision 1.0 만들기" 같은 꼼수가 안 통해요. 둘이 같아질수록(오른쪽 끝) 두 선이 만나요.

🔧 혼동행렬과 네 지표 명령어 사전에서 자세히 →

```
cm = confusion_matrix(y, pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()                         # 이 순서로 풀림!
```

## ② ROC-AUC와 임계값

| 모델이 낸 값 | 운영에서 내릴 판단 |
|---|---|
| predict_proba → 클래스별 추정 확률. 실패(1) 열은 classes_로 확인 | 확률이 임계값을 넘으면 실패. 기본 정책: p > 0.5이면 1, p = 0.5이면 0 |
| ROC-AUC에는 이 연속 점수를 넣음 | 임계값은 모델·자료와 함께 평가 전에 고정할 설정 |

ROC 곡선: 임계값을 1 → 0으로 내리면서 찍은 점들.

```
가로축 FPR = FP / (FP + TN)   성공 요청 중 잘못 경고한 비율
세로축 TPR = TP / (TP + FN)   실제 실패 중 찾아낸 비율 (= Recall)
AUC   = 곡선 아래 넓이 = "실패 1건 · 성공 1건을 뽑았을 때 실패 점수가 더 높을 확률"
```

| 실제 | 점수 | 비교 |
|---|---|---|
| 성공 0 | 0.10 | 실패 0.35, 0.80보다 낮음 ✅✅ |
| 성공 0 | 0.40 | 실패 0.35보다 높음 ❌, 0.80보다 낮음 ✅ |
| 실패 1 | 0.35 | 양성·음성 2×2 = 4쌍 중 3쌍에서 양성 점수가 높음 → AUC = 0.75 |
| 실패 1 | 0.80 |  |

🤔 왜 AUC는 "순위" 평가지? — AUC는 실패·성공을 한 쌍씩 짝지어 "실패 쪽 점수가 더 높은가"만 세요(동률 없음 기준). 점수의 크기가 아니라 순서만 보니까, 임계값을 어디에 두든 순서는 그대로라 AUC도 그대로예요. 그래서 0/1 클래스를 넣으면 순서 정보가 사라져 의미가 없고, 연속 점수(predict_proba의 실패 열)를 넣어야 해요. 또 짝을 지으려면 두 클래스가 모두 있어야 하니, 한 클래스뿐인 자료에선 계산할 수 없어요(0으로 위장하지 말고 None + 실제 양성 수 기록).

[그림: 코드 예시 4건과 슬라이더 20건의 ROC 곡선과 AUC 면적] 무엇을 보나: 왼쪽은 아래 코드의 4건, 오른쪽은 아래 슬라이더의 20건으로 그린 ROC 곡선이에요. 칠해진 넓이가 AUC(0.75, 0.821)이고, 점선 대각선은 무작위 순서(0.5)예요. 오른쪽 점 두 개는 임계값 0.5·0.3이 곡선 위 어디에 서 있는지 보여 줘요. 🤔 왜? 곡선은 임계값을 1에서 0으로 내리며 "실패 하나 잡을 때마다 위로, 성공 하나 잘못 경고할 때마다 오른쪽으로" 한 칸씩 걸은 발자국이에요. 점수 순서만으로 걸음이 정해지니, 임계값은 곡선 위의 한 점을 고르는 것일 뿐 곡선(AUC)을 바꾸지 않아요.

🤔 왜 확률 0.8 ≠ "80% 실패"지? — 모델이 낸 확률은 "추정치"라서, 실제로 0.8짜리 요청들 중 80%가 실패하는지는 따로 검증(확률 보정)해야 알아요. 기본 정책은 p > 0.5면 실패, 정확히 0.5면 성공으로 미리 적어 두어 경계에서 헷갈리지 않게 해요.

아래 슬라이더로 임계값을 바꿔 보세요. AUC는 그대로인데 FP·FN·F1·비용은 바뀌어요. (설명용 가상 점수 20건 · 실패 6건 · 비용 = FP + 5×FN은 가정)

🔧 roc_curve · roc_auc_score · 임계값 명령어 사전에서 자세히 →

```
fpr, tpr, thresholds = roc_curve(y, p)          # FPR [0 0 .5 .5 1], TPR [0 .5 .5 1 1]
print(roc_auc_score(y, p))                      # 0.75
```

[그림: 임계값에 따른 FP, FN, 비용, F1과 변하지 않는 AUC] 무엇을 보나: 슬라이더와 같은 20건으로 임계값을 0.05씩 옮긴 결과예요. 왼쪽: 선을 올리면 주황 FP는 내려가고 빨간 FN은 올라가며, 가정한 비용(FP + 5×FN)은 t=0.25에서 가장 작아요. 오른쪽: F1은 t마다 오르내리지만 분홍 AUC는 수평선이에요. 🤔 왜? FN 한 건을 FP 다섯 건과 같은 값으로 쳤으니, 비용 곡선은 FN이 생기기 시작하는 지점 바로 앞에서 바닥을 쳐요. 비용 비율을 바꾸면 그 바닥 위치도 옮겨 가요. 임계값은 이렇게 오류 비용으로 정하는 운영 규칙이고, valid에서만 골라요.

[그림: 같은 20건 점수의 Precision-Recall 곡선과 임계값 0.5, 0.3의 위치] 무엇을 보나: 같은 20건을 Recall(가로)-Precision(세로)로 다시 그린 PR 곡선이에요. 오른쪽으로 갈수록 경고를 많이 해서 실패를 더 찾지만 Precision이 내려가요. 점선 0.30은 아무렇게나 경고할 때의 Precision(실패 비율)이에요. 🤔 왜? ROC의 가로축(FPR)은 흔한 성공 14건으로 나누니 오탐이 몇 건 늘어도 티가 덜 나요. PR은 "경고한 것 중 진짜"를 직접 보여 줘서, 실패가 드문 불균형 자료에서 경고의 질을 더 솔직하게 드러내요. t=0.3은 실패를 5/6 찾는 대신 경고의 62%가 오탐이에요.

## ③ RMSE와 교차검증 점수 집계

```
RMSE = √ mean((y − ŷ)²)      오차 → 제곱 → 평균 → 제곱근 · 단위는 정답과 같음(ms)
```

| 모델 | 예측 (실제 100, 200, 300, 400) | MAE | RMSE |
|---|---|---|---|
| A | 100, 200, 300, 480 (한 번 크게) | 20 ms | 40 ms |
| B | 130, 230, 330, 430 (고르게) | 30 ms | 30 ms |

💡 MAE는 평균적인 절대 오차, RMSE는 큰 오차에 더 민감해요. 평균만 보지 말고 큰 잔차의 실제값·예측값·ID를 확인해요. 회귀 점수와 분류 F1은 합쳐서 평균 내지 않아요.

🔧 MAE · RMSE 계산 mean_absolute_error · root_mean_squared_error 명령어 사전에서 자세히 →

```
for name, pred in [("A", a), ("B", b)]:
    print(name, mean_absolute_error(y, pred), root_mean_squared_error(y, pred))
```

🤔 왜 A는 MAE로 이기고 RMSE로 지지? — A의 오차는 (0, 0, 0, 80): MAE = 80/4 = 20, RMSE = √(80²/4) = √1600 = 40. B는 (30, 30, 30, 30): 둘 다 30. 제곱하면 큰 오차 80이 6400으로 커져 혼자 평균을 끌어올리니, "한 번 크게 틀리는" A가 RMSE에서 불리해요.

[그림: 모델 A와 B의 요청별 오차와 MAE, RMSE 비교] 무엇을 보나: 왼쪽은 실제값별 오차예요. A는 마지막 요청에서만 80 틀렸고, B는 매번 30씩 틀렸어요. 오른쪽에서 MAE는 A(20) < B(30), RMSE는 A(40) > B(30)로 순위가 뒤집혀요. 🤔 왜? RMSE는 제곱 평균이라 80이 6400으로 부풀어 혼자 평균을 끌어올려요. "가끔 크게 틀리는" 모델과 "늘 조금씩 틀리는" 모델 중 무엇이 더 나쁜지는 업무가 정할 일이라, 주 지표를 미리 고르고 큰 잔차의 행을 따로 확인해요.

| 집계 방법 | 뜻 | 주의 |
|---|---|---|
| fold 점수 평균 ± 표준편차 | 예: F1 0.522 ± 0.209 → 분할에 따라 흔들렸다는 진단 근거 | 95% 신뢰구간이 아님 |
| OOF 예측을 합쳐 한 번 계산 | 각 행을 그 행을 학습하지 않은 모델로 예측해 모음 | fold 평균과 다를 수 있음 (RMSE 2·10 → 평균 6, 합친 값 약 7.21) |

🤔 왜 평균 6인데 합치면 7.21이지? — RMSE는 "제곱 평균 → 루트"라 직선이 아니에요(비선형). 같은 크기 두 fold의 제곱오차 평균은 2²=4, 10²=100이라, 합치면 √((4+100)/2) = √52 ≈ 7.21. 반면 루트를 씌운 뒤 평균 내면 (2+10)/2 = 6. F1도 분모가 fold마다 달라서 마찬가지예요. 그래서 무엇을 어떻게 평균 냈는지 적고, 서로 다른 계산의 숫자를 같은 열에 섞지 않아요.

[그림: fold RMSE 2와 10의 평균 6과 합쳐 계산한 7.21 비교 막대] 무엇을 보나: 같은 두 fold(RMSE 2, 10)를 두 방식으로 요약했어요. 각 fold의 RMSE를 평균하면 6.00, 두 fold의 제곱오차를 먼저 합친 뒤 한 번에 RMSE를 내면 7.21이에요. 🤔 왜? 루트(√)는 큰 값을 덜 키우는 곡선이라 "루트 먼저 → 평균"과 "평균 먼저 → 루트"의 결과가 달라요. 둘 다 틀린 계산이 아니라 질문이 다른 계산이니, 보고서에 어떤 방식인지 적고 한 열에 섞지 않아요.

🤔 왜 "평균 ± 표준편차"가 신뢰구간이 아니지? — 5개 fold는 같은 60건에서 학습 자료를 서로 많이 나눠 쓰기 때문에 독립적인 5번의 실험이 아니에요. 표준편차는 "분할에 따라 이만큼 흔들렸다"는 진단 기록이지, 진짜 성능이 그 범위 안에 있을 확률을 말해 주지 않아요.

| 자료 구조 | 검토할 분할 | 추가 검사 |
|---|---|---|
| 독립 요청 · 분류 | StratifiedKFold | fold별 양성 수 · ID 교집합 |
| 같은 고객·요청의 반복 행 | GroupKFold 등 | 같은 그룹이 학습/평가 양쪽에 없는가 |
| 시간 순서 · 미래 예측 | TimeSeriesSplit 등 | 미래 자료가 과거 학습에 들어오지 않는가 |
| 고른 최고 점수의 낙관성 | 별도 최종 test / 필요하면 nested CV | 선택용 자료와 최종 평가의 역할 구분 |

💡 분할 방법은 점수가 잘 나오는 걸로 고르지 않고, 실제로 예측할 상황에 맞춰 정해요.

🤔 왜 상황마다 분할이 다르지? — 평가는 "실전과 같은 조건의 시험"이어야 해요. 같은 고객의 행이 학습·평가에 나뉘어 들어가면, 모델이 그 고객을 이미 "외운" 상태로 시험을 보게 돼요(그룹 분할 필요). 미래를 예측하는데 미래 자료로 학습하면 실전에선 불가능한 정보를 쓰는 셈이에요(시간 분할 필요).

🔧 cross_validate로 여러 지표 · 음수 RMSE 명령어 사전에서 자세히 →

```
scores = cross_validate(pipe, X_train, y_train, cv=CLS_FOLDS, scoring={"f1": "f1", "auc": "roc_auc"}, …)
rmse = -reg["test_score"]                            # 부호 되돌리기 → 평균 639.83 ms (KNN15는 598.85 ms)
```

## ④ 특성 설계 (Feature Engineering)

원본 입력을 의미 있게 변환·조합해서, 모델이 쓰기 좋은 표현을 새로 만들어 줘요.

가설→ 변환 정의→ 입력/출력 검사→ 고정 조건 비교 같은 모델 · 같은 CV→ 해석의 한계

| 특성 | 정의 | 주의 |
|---|---|---|
| log_input_chars | log(1 + 글자수) | 0이면 0 · 음수 금지 · NaN은 NaN 유지 (큰 값 차이를 완만하게) |
| chars_per_slot | 글자수 / (1 + 허용 도구수) | +1은 분모 0 방지용 설계 선택 · "도구당 실행시간"이 아님 |
| 원본 4열 | 수치 2 + 범주 2 그대로 | 추가만 하고 삭제하지 않음 |
| ❌ 금지 | 실행시간/글자수, retry_count 등 | 실행 후 정보로 만들면 시점 누수 |

| 글자수 / 도구수 | log_input_chars | chars_per_slot |
|---|---|---|
| 0 / 0 | 0 | 0 |
| 99 / 2 | 4.605 | 33 |
| 999 / 4 | 6.908 | 199.8 |
| NaN | NaN 유지 | NaN 유지 |

🤔 왜 log를 씌우지? — 글자수는 수백~수천처럼 범위가 넓고 큰 값 쪽으로 길게 늘어져요. log를 씌우면 99 → 4.605, 999 → 6.908처럼 10배 차이가 약 2.3 차이로 완만해져서, 몇 개의 아주 긴 요청이 계수·거리를 휘두르지 않게 돼요. log1p(= log(1+x))를 쓰는 이유는 글자수 0에서도 log(1)=0으로 안전하기 때문이에요(그냥 log(0)은 −∞).

[그림: log1p 함수 곡선과 0, 9, 99, 999의 변환값] 무엇을 보나: log1p 곡선 위에 0·9·99·999를 찍었어요. 99에서 999까지 가로로 900을 가도 세로로는 4.61 → 6.91, 겨우 2.3만 올라가요. 🤔 왜? 로그는 "몇 배인가"를 "몇 칸인가"로 바꿔요. 10배씩 커질 때마다 약 2.3씩 같은 간격으로 올라가니, 큰 값끼리의 차이는 눌리고 작은 값 쪽 차이는 살아나요. x=0에서도 log(1)=0이라 안전하고요.

[그림: 오른쪽 꼬리가 긴 분포가 log1p 후 가운데로 모이는 히스토그램] 무엇을 보나: 오른쪽 꼬리가 긴 설명용 합성 글자수 400건(왼쪽)이 log1p 후(오른쪽) 가운데로 모인 종 모양에 가까워졌어요. 🤔 왜? 아주 긴 몇 개 요청(왼쪽 그림 오른쪽 끝)이 log 눈금에선 다른 값들과 가까워져서, 계수·거리를 혼자 휘두르지 못해요. 참고로 수업 자료의 글자수는 203~2381로 이렇게 심하게 치우치진 않아서 log 효과가 크지 않을 수 있어요 — 그래서 같은 조건 비교로 확인해요.

🤔 왜 비율 특성을 만들지? 왜 +1? — "글자수가 많은데 쓸 수 있는 도구는 적다" 같은 두 입력의 관계는 원본 열 하나씩만 봐서는 선형 모델이 잘 못 잡아요. 비율로 만들어 주면 그 관계를 한 열로 직접 보여 줄 수 있어요. 분모에 1을 더한 건 도구 수가 0일 때 0으로 나누는 오류(무한대)를 막기 위한 명시적 설계 선택이에요. 성능이 좋아질지는 같은 조건 비교로 따로 확인해요.

| 행별로 계산 가능한 규칙 | 자료에서 추정하는 변환 |
|---|---|
| log1p(글자수), 글자수 / (1 + 도구수) — 한 행의 사전 입력만 사용 | 중앙값·평균·표준편차, 범주 목록, 특성 선택, PCA, K-Means |
| 그래도 Pipeline에 넣어 학습·추론 규칙을 일치시킴 | 반드시 학습 자료(fold 학습 부분)에서만 fit |

⚠️ 정답을 쓰지 않는 비지도 변환이라도 평가 자료로 fit하면 누수예요. 수치 결측을 "전체 자료 중앙값"으로 먼저 채우는 것도 안 돼요. 정답을 이용하는 인코딩·특성 선택은 더 엄격한 누수 방지 절차가 필요해요.

🤔 왜 행별 규칙도 Pipeline에 넣지? — log1p나 비율은 한 행만 보고 계산하니 누수는 없어요. 그래도 Pipeline 밖에서 손으로 만들면, 나중에 새 요청을 예측할 때 똑같이 만드는 걸 잊거나 식이 조금 달라질 수 있어요. Pipeline 안에 두면 학습 때와 추론 때 같은 규칙이 자동으로 적용돼요. 반대로 중앙값·표준편차·범주 목록·K-Means 중심처럼 자료에서 배우는 값은 반드시 학습 부분에서만 fit해야 해요.

[그림: ] 무엇을 보나: 새 특성을 만드는 분홍 상자가 Pipeline 맨 앞 단계로 들어가 있어요. 4열 → 6열 → 10열로 늘어나며 모델까지 한 줄로 흘러가요. 🤔 왜? 분홍 상자는 한 행만 보고 계산하니 누수 걱정은 없지만, Pipeline 안에 있어야 새 요청을 예측할 때도 똑같은 식이 자동으로 붙어요. 보라 상자(중앙값·평균·표준편차)는 자료에서 배우는 값이라, fold마다 학습 부분으로만 fit되는 것도 이 구조 덕분이에요.

🔧 np.log1p · FunctionTransformer 명령어 사전에서 자세히 →

```
out["log_input_chars"] = np.log1p(out["input_chars"])
("features", FunctionTransformer(add_features, validate=False)),
```

## ⑤ K-Means — 정답 없이 비슷한 것끼리 묶기

|  | KNN (Day6) | K-Means |
|---|---|---|
| 종류 | 지도학습 (정답 있음) | 비지도학습 (정답 없음) |
| K의 뜻 | 참고할 이웃 수 | 만들 군집 개수 |
| 하는 일 | 가까운 학습 사례의 라벨로 예측 | 군집 중심을 학습, 새 입력은 가까운 중심에 할당 |

초기화 K개 중심 선택 (k-means++)→ 할당 각 점 → 가장 가까운 중심→ 갱신 중심 = 소속 점들의 평균→ 반복 안정되면 종료

목표는 각 점에서 소속 중심까지 제곱거리의 합(inertia)을 줄이는 것. 초기값에 따라 결과가 달라질 수 있어서 여러 번 시도(n_init)해요.

🤔 왜 "할당 → 평균으로 이동"을 반복하면 되지? — 중심이 정해져 있으면 각 점을 가장 가까운 중심에 붙이는 게 제곱거리를 가장 줄이는 방법이고, 묶음이 정해져 있으면 그 묶음의 평균이 제곱거리 합을 가장 작게 만드는 중심이에요. 두 단계 모두 inertia를 줄이거나 그대로 두니 언젠가 멈춰요. 다만 처음 중심을 어디 두었느냐에 따라 "그럭저럭 괜찮은 곳"에서 멈출 수 있어서, 여러 번 시작해 보고 가장 좋은 걸 고르는 거예요(n_init).

[그림: K-Means가 반복하며 중심이 군집 가운데로 이동하는 네 단계] 무엇을 보나: 합성 자료에서 일부러 나쁜 곳에 중심 3개(X)를 놓고 시작했어요. 점 색은 "지금 가장 가까운 중심"이고, 화살표는 중심이 소속 점들의 평균 위치로 옮겨 간 길이에요. 위쪽 숫자 inertia가 1950 → 348로 줄어요. 🤔 왜? 할당 단계에선 각 점이 더 가까운 중심으로 갈아타고, 갱신 단계에선 중심이 자기 점들의 한가운데로 옮겨요. 두 단계 모두 제곱거리 합을 줄이기만 하니, 몇 번 만에 더 움직일 곳이 없어 멈춰요. 시작점이 나쁘면 엉뚱한 곳에서 멈출 수도 있어 n_init으로 여러 번 시작해요.

⚠️ 군집 0·1·2는 순서나 성공/실패 의미가 없는 이름표예요. 0과 1을 바꿔도 같은 묶음. "군집 2가 더 위험하다" 같은 해석은 금지.

🤔 왜 번호에 의미가 없지? — K-Means는 정답 y를 전혀 보지 않고 "가까운 것끼리"만 묶어요. 번호는 계산 순서상 붙은 이름표라 0과 1을 바꿔도 묶음은 똑같아요. 성공/실패나 위험도는 학습에 들어간 적이 없으니, 번호에서 그런 의미를 읽을 수 없어요.

| 평가 | 뜻 | 주의 |
|---|---|---|
| inertia | 군집 안의 거리 (작을수록 촘촘) | K를 늘리면 거의 항상 작아짐 → 최솟값만으로 K 결정 금지 |
| silhouette | 같은 군집과의 거리 vs 다른 군집과의 거리 (−1 ~ 1, 높을수록 분리 좋음) | 군집 2개 이상 · 표본 수 미만 필요. 높아도 업무적 의미는 별도 |

| K | inertia | silhouette |
|---|---|---|
| 2 | 69.864 | 0.392 |
| 3 | 40.430 | 0.453 |
| 4 | 30.374 | 0.445 |

🤔 왜 inertia만으로 K를 고르면 안 되지? — 군집을 늘리면 각 점이 더 가까운 중심을 갖게 되니 inertia는 거의 항상 줄어요(K = 점 개수면 0). 그래서 "작을수록 좋다"로 고르면 늘 큰 K가 이겨요. silhouette는 "내 군집과는 가깝고 옆 군집과는 먼가"를 같이 보니, 쓸데없이 쪼갠 군집은 옆 군집과도 가까워져 점수가 떨어져요. 이 자료에선 K=4가 inertia는 작지만 silhouette는 K=3(0.453)이 가장 높아서 K=3으로 설명해요. 이건 이 train 60·이 전처리의 교육용 결과이지, 새 자료에서도 K=3이 최선이라는 뜻은 아니에요.

[그림: K별 inertia 감소 곡선과 silhouette 점수] 무엇을 보나: 수업 train 60건(결측 대체 + 표준화)으로 K=2~6을 직접 계산했어요. K=2·3·4 값은 위 표와 같아요. 왼쪽 inertia는 K를 늘릴수록 계속 내려가고, 오른쪽 silhouette는 K=3(0.453)에서 꼭대기를 찍은 뒤 0.44대에 머물러요. 🤔 왜? 중심이 많을수록 모든 점이 더 가까운 중심을 가지니 inertia는 당연히 줄어요 — 그래서 "작을수록 좋다"로 고르면 늘 큰 K가 이겨요. silhouette는 "내 군집과 가깝고 옆 군집과 먼가"를 보니, 한 덩어리를 억지로 쪼개면 옆 군집과도 가까워져 더 오르지 않아요.

K=3 결과: 군집별 개수 24 / 19 / 17, inertia 40.430. 중심(원래 단위로 읽기): 군집0 약 1542자·도구 4.12 / 군집1 약 1430자·도구 1.37 / 군집2 약 445자·도구 2.71. 도구 수가 소수인 건 군집 내부 평균이라서예요. 중심은 결측 대체 후 좌표의 평균이라 "실제 사용한 도구 수"나 "성공률"로 해석하지 않아요.

[그림: 수업 train을 표준화 공간에서 K=3으로 묶은 산점도와 새 요청 두 개의 할당] 무엇을 보나: 표준화된 글자수(가로)·도구수(세로) 공간의 train 60건이에요. 색은 군집, 큰 X는 중심, 주황 별은 새 요청 2개(200자·도구1, 2200자·도구5)예요. 각각 가장 가까운 중심인 군집 2와 0에 할당돼요. 🤔 왜? 도구수는 1~5 정수라 점들이 가로줄로 늘어서요. 군집은 대략 "짧은 글"(초록), "길고 도구 적음"(분홍), "길고 도구 많음"(보라)으로 나뉘었는데, 이건 거리로만 묶은 결과라 성공/실패 의미는 없어요. 새 요청은 predict로 기존 중심에만 붙이고 중심은 그대로예요.

🔧 KMeans · silhouette_score 명령어 사전에서 자세히 →

```
km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels = km.fit_predict(X)              # y를 넣지 않는다!  → [0 0 0 1 1 1]
```

## ⑥ 최종 평가 흐름 — 잠그고, 재학습하고, 한 번 평가

선택 train60 CV · valid는 진단→ 동결 모델·특성·임계값 0.5 기록→ 재학습 새 Pipeline을 개발 80건으로→ 최종 보고 test20 transform · predict

| 후보 (사전 선언 7개) | CV F1 | fold 표준편차 | valid F1 |
|---|---|---|---|
| Dummy | 0.000 | 0.000 | 0.000 |
| Logistic | 0.522 | 0.209 | 0.333 |
| KNN3 | 0.336 | 0.176 | 0.545 |
| Tree | 0.542 | 0.124 | 0.333 |
| Forest | 0.280 | 0.254 | 0.222 |
| XGBoost | 0.500 | 0.123 | 0.182 |
| Logistic + 파생 2개 | 0.551 | 0.190 | 0.182 |

[그림: 7개 후보의 CV F1 평균과 표준편차 막대, valid F1 점] 무엇을 보나: 위 표를 그림으로 옮겼어요. 막대가 CV F1 평균(선택 기준), 회색 수염이 fold 표준편차, 주황 마름모가 valid F1(진단용)이에요. 진한 보라 막대가 규칙대로 선택된 Logistic+파생2예요. 🤔 왜? 1등 막대(0.551)와 2등 Tree(0.542)의 차이는 수염 길이(약 0.12~0.19)보다 훨씬 작아요. 그리고 주황 마름모가 막대와 제각각 위치해 있죠 — 작은 자료라 평가 세트마다 순위가 바뀐다는 뜻이에요. 그래서 "규칙에 따른 선택"이지 "확실히 우월"이라고 쓰지 않아요.

🤔 왜 test를 열기 전에 잠그지? — test 점수를 본 뒤에 모델·특성·임계값을 바꾸면, test는 "고르는 데 쓴 자료"가 되어 버려요. 그럼 그 점수는 valid처럼 낙관적인 값이 되고, 진짜 처음 보는 자료에서의 성능을 알 방법이 사라져요. 그래서 무엇(후보·특성) · 어떻게(CV F1 최대, 동점 규칙) · 판단(임계값 0.5)을 먼저 파일로 남기고 test는 딱 한 번만 써요. valid도 이미 개발에 쓴 자료라 최종 독립 평가처럼 포장하지 않아요.

[그림: ] 무엇을 보나: 가로 막대 한 줄이 한 단계이고, 칸 너비가 자료 건수예요. 회색 🔒 test 20은 1~3단계 내내 잠겨 있다가 4단계에서만 진하게 열려요. 🤔 왜? 고르는 데 쓴 자료의 점수는 낙관적이라, 한 번도 안 쓴 자료가 하나는 남아 있어야 정직한 마지막 점수를 얻어요. 2단계에서 설정을 파일로 먼저 잠그는 건, test를 본 뒤 "조금만 바꿔 보자"는 유혹을 막는 장치예요.

CV F1 최대 규칙으로 Logistic_FE 선택(Tree와의 약 0.009 차이를 확실한 우월성이라 주장하지 않음) → test 20건 결과: Acc 0.750 · Precision 0.600 · Recall 0.500 · F1 0.545 · AUC 0.786 (TN12 FP2 FN3 TP3, 실패 3/6 발견).

[그림: 최종 test 20건 혼동행렬 TN 12, FP 2, FN 3, TP 3] 무엇을 보나: 잠근 설정(Logistic_FE, 임계값 0.5)으로 test 20건을 한 번 예측한 결과예요. 실제 실패 6건(아래 줄) 중 3건만 찾고(TP 3), 3건은 놓쳤어요(FN 3). 🤔 왜? AUC 0.786은 순위가 꽤 괜찮다는 뜻이지만, 놓친 3건은 확률이 0.5를 넘지 못했어요. 이 표가 이 모델의 최종 정직한 성적표라서, 마음에 안 든다고 임계값·모델을 바꿔 다시 재면 안 돼요 — 그 순간 test는 "고른 자료"가 돼요.

🤔 왜 마지막에 80건으로 다시 학습하지? — 선택이 끝나면 valid 20건은 더 이상 "고르는 데" 쓸 일이 없어요. 그 정보를 버리기 아까우니 train+valid 80건으로 같은 설정의 새 Pipeline을 학습해 최종 모델로 써요. 이때 clone으로 학습 상태 없는 새 객체를 만드는 이유는, 기존에 60건으로 학습된 상태(중앙값 등)가 섞이지 않게 하려는 거예요. 재학습 뒤 수치 중앙값이 1278 / 3, log 7.154, ratio 309.8로 바뀌는 건 정상이고, 80건 재학습 후의 과거 valid 성능은 더 이상 독립 검증 점수가 아니에요.

🔧 설정 저장 · clone 재학습 · 저장/복원 명령어 사전에서 자세히 →

```
final = clone(candidates()[selected]).fit(X_dev, y_dev)  # 학습 상태 없는 새 Pipeline
joblib.dump(final, "final_pipeline.joblib")
```

🔧 새 입력 검사 (새 범주 vs 필수 열 누락) 명령어 사전에서 자세히 →

```
p = failure_scores(fitted, checked)          # 새 범주가 있어도 (2, 10)으로 변환 · 확률은 유한값
check_input(new_requests.drop(columns="priority"))         # ← ValueError가 나야 정상 (실패 시험)
```

### 지표로 진단하기

| 이렇게 나오면 | 의심할 것 |
|---|---|
| Accuracy 높은데 Recall 0 | 다수 클래스만 찍는 모델 → 실제 양성 수 확인 |
| AUC는 좋은데 F1 낮음 | 순위는 괜찮지만 임계값이 운영에 안 맞음 |
| CV는 오르고 valid는 떨어짐 (0.522→0.551 / 0.333→0.182) | 소표본 변동 → 개선 확정 금지 |
| K를 늘릴수록 inertia만 계속 감소 | 당연한 현상 → silhouette·해석과 함께 |
| 새 범주 입력 / 필수 열 누락 | 새 범주는 handle_unknown="ignore"로 열 수 유지, 열 누락은 예측 전에 ValueError로 막기 |

## 원본 자료

- 강의: 20261028수_머신러닝_평가지표_특성설계_KMeans_최종평가_Day8 (2).pdf — ① 분류 지표 9~19쪽 · ② ROC-AUC·임계값 20~29쪽 · ③ RMSE·CV 집계 30~38쪽 · ④ 특성 설계 39~47쪽 · ⑤ K-Means 48~60쪽 · ⑥ 최종 평가 61~69, 73~77쪽

- 공식 문서: scikit-learn · Metrics and scoring · Decision threshold · FunctionTransformer · Clustering · K-Means · Model persistence

## 🔧 Day8 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day8)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install numpy pandas scikit-learn |
| ① 분류 지표 | confusion_matrix(labels=), precision_score, recall_score, f1_score(zero_division=) |
| ② ROC-AUC · 임계값 | predict_proba, roc_curve, roc_auc_score, (p > t).astype(int) |
| ③ RMSE · CV 점수 집계 | root_mean_squared_error, cross_validate(scoring={...}) |
| ④ 특성 설계 | np.log1p, FunctionTransformer(validate=False), pipe[:-1] |
| ⑤ K-Means | KMeans(n_clusters=, n_init=), inertia_, silhouette_score, predict |
| ⑥ 최종 평가 흐름 | clone(), pd.concat, json.dumps, joblib.dump/load, 입력 검사(ValueError) |

### 혼동행렬과 네 지표

```
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

cm = confusion_matrix(y, pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()                         # 이 순서로 풀림!
print(accuracy_score(y, pred))
print(precision_score(y, pred, zero_division=0))
print(recall_score(y, pred, zero_division=0))
print(f1_score(y, pred, zero_division=0))       # [[10 4] [4 2]] · Acc 0.600 · P/R/F1 0.333

# 전부 "성공"으로 찍는 경우 (E02)
y = np.array([0]*14 + [1]*6)
pred = np.zeros(20, dtype=int)
print(accuracy_score(y, pred), recall_score(y, pred, zero_division=0),
      f1_score(y, pred, zero_division=0))       # 0.70 0.0 0.0  → 실패 발견 0/6
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 · Logistic C=1의 valid 예측

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| confusion_matrix(y, pred, labels=[0, 1]) | [[10, 4], [4, 2]] | 행 = 실제, 열 = 예측 |
| tn, fp, fn, tp = cm.ravel() | 10, 4, 4, 2 | 이 순서로 풀림 |
| accuracy · precision · recall · f1 | 0.600 · 0.333 · 0.333 · 0.333 | P = 2/(2+4), R = 2/(2+4) — 분모가 달라도 여기선 FP = FN이라 같음 |
| f1_score(y, pred, pos_label=0) | 0.714 | 성공을 양성으로 보면 전혀 다른 숫자 |
| 전부 0 예측: accuracy · recall · f1 | 0.70 · 0.0 · 0.0 | 실패 6건을 하나도 못 찾아도 Accuracy 0.70 |
| confusion_matrix([0,0,0], [0,0,0]) (labels 없이) | [[3]] | 1×1 — ravel()로 4개를 못 풂 |
| confusion_matrix([0,0,0], [0,0,0], labels=[0, 1]) | [[3, 0], [0, 0]] | labels를 주면 항상 2×2 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| labels=[0, 1] | 행·열 순서 고정 | 안 주면 데이터에 나온 값으로 정해져서, 한 클래스만 있는 자료에선 1×1이 되어 ravel() 풀기가 깨짐 → 지정하면 2×2 유지 |
| .ravel() | 2×2를 TN, FP, FN, TP 순서로 펼침 | 행=실제·열=예측 순서라 이 순서. 이름을 바꿔 받으면 지표가 통째로 틀려짐 |
| zero_division=0 | 분모가 0(양성 예측 0건 등)이면 0으로 기록 | 안 주면 경고와 함께 0. "계산 불가"를 0으로 기록하는 규칙이니 "양성 예측 0건"이라는 상황도 함께 적기 |
| pos_label (기본 1) | 어느 클래스를 양성으로 볼지 | 실패=1이 아니면 직접 지정. 바꾸면 Precision·Recall·F1이 "성공"을 기준으로 다시 계산됨 |

### roc_curve · roc_auc_score · 임계값

```
from sklearn.metrics import roc_curve, roc_auc_score

y = np.array([0, 0, 1, 1])
p = np.array([0.10, 0.40, 0.35, 0.80])          # 실패(1)의 확률 = predict_proba(X)[:, 실패열]
fpr, tpr, thresholds = roc_curve(y, p)          # FPR [0 0 .5 .5 1], TPR [0 .5 .5 1 1]
print(roc_auc_score(y, p))                      # 0.75

for t in [0.5, 0.3]:
    pred = (p > t).astype(int)                   # 같은 점수, 다른 판단 기준
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(t, fp, fn, fp + 5 * fn)               # t=0.5: FP0 FN1 비용5 (F1 .667) / t=0.3: FP1 FN0 비용1 (F1 .800)
# 비용 5:1은 설명용 가정이지 실제 업무 근거가 아님 · AUC는 계속 0.75
```

▶ 실제로 돌려 보면 노트의 4건 (실제 [0, 0, 1, 1])

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| fpr, tpr = roc_curve(y, p)[:2] | [0.0, 0.0, 0.5, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0, 1.0] | 왼쪽 아래 (0,0)에서 오른쪽 위 (1,1)까지 |
| thresholds | [inf, 0.80, 0.40, 0.35, 0.10] | 첫 값 inf = '아무도 양성 아님' 시작점 |
| roc_auc_score(y, p) | 0.75 | 실패·성공 4쌍 중 3쌍에서 실패 점수가 더 높음 (0.35 < 0.40만 역전) |
| t=0.5: (p > t) → FP, FN, 비용 fp+5·fn, F1 | [0, 0, 0, 1] → 0, 1, 5, 0.667 | 기본 선: 0.35인 실패를 놓침 |
| t=0.3: (p > t) → FP, FN, 비용 fp+5·fn, F1 | [0, 1, 1, 1] → 1, 0, 1, 0.800 | 선을 내리면 놓침 0 · 헛경고 1 |
| roc_auc_score(y, p) (임계값과 무관) | 0.75 | 순위만 보니 t를 바꿔도 그대로 |
| roc_auc_score(y, (p > 0.5).astype(int)) · ROC 점 개수 | 0.75 · 3개 | 이 4건에선 우연히 같은 0.75(동점을 0.5로 셈)지만, 곡선 점이 5개 → 3개로 줄어 순위 정보를 잃음 |
| roc_auc_score([0, 0, 0], [.1, .2, .3]) | nan · 경고 UndefinedMetricWarning | 한 클래스뿐이면 정의 불가 — sklearn 1.8은 에러 대신 nan + 경고를 돌려줘요. 조용히 넘어가지 않게 직접 검사해 None으로 기록 |

| 인자 | 뜻 | 주의 / 바꾸면? |
|---|---|---|
| roc_curve(y, score) | FPR · TPR · 임계값 배열 | score는 0/1 클래스가 아니라 연속 점수 — 클래스를 넣으면 곡선 점이 몇 개뿐이 됨 |
| roc_auc_score(y, score) | 양성 점수의 순위 평가 | y에 두 클래스가 모두 있어야 계산 가능 (0으로 위장 금지) |
| (p > t) | 임계값 규칙 | t를 낮추면 경고↑ → FN↓·FP↑. 임계값 탐색은 test에서 금지 — test로 고르면 그 test는 더 이상 독립 평가가 아님 |

🤔 왜 t=0.3에서 비용이 줄었지? — 실패 0.35가 0.5 선에선 "성공"(FN)이었는데 0.3 선에선 "실패"로 잡혀요. 대신 성공 0.40도 넘어와 FP가 1 생겨요. 가정한 비용이 FN 1건 = 5, FP 1건 = 1이라, 비싼 놓침 하나를 싼 헛경고 하나와 바꾼 셈이라 5 → 1이 됐어요. 비용 가정이 바뀌면 답도 바뀌어요.

⚠️ AUC만 보고 끝내지 않기

- AUC 0.786은 "실패 78.6% 발견"이 아니에요. 발견 비율은 Recall.

- AUC는 특정 임계값의 F1이나 확률이 얼마나 잘 맞는지(보정)와 별개예요.

- 불균형 자료에서는 실제 양성 수, PR 관점도 함께 봐요.

- 임계값만 바꾸면 점수 순위는 같으니 AUC는 그대로, FP·FN·F1이 바뀌어요.

### cross_validate로 여러 지표 · 음수 RMSE

```
from sklearn.model_selection import cross_validate

scores = cross_validate(pipe, X_train, y_train, cv=CLS_FOLDS,
                        scoring={"f1": "f1", "auc": "roc_auc"},   # 한 번에 두 지표
                        return_train_score=True, error_score="raise", n_jobs=1)
print(scores["test_f1"], scores["test_f1"].mean())   # fold F1 [.286 .667 .857 .400 .400], 평균 .522 (편차 .209)
print(scores["test_auc"].mean())                     # 평균 AUC .805

reg = cross_validate(build(LinearRegression()), X_train, yr_train, cv=REG_FOLDS,
                     scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise")
rmse = -reg["test_score"]                            # 부호 되돌리기 → 평균 639.83 ms (KNN15는 598.85 ms)
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20 · 5-fold (학습 48 / 평가 12)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| sorted(k for k in scores if k.startswith("test")) | ['test_auc', 'test_f1'] | scoring 딕셔너리 키 → test_키 |
| scores["test_f1"] | [0.286, 0.667, 0.857, 0.400, 0.400] | fold마다 크게 다름 |
| mean() · std(ddof=0) | 0.522 · 0.209 | fold 점수 범위(.286~.857)가 넓어 평균 하나로는 부족 |
| scores["test_auc"].mean() · scores["train_f1"].mean() | 0.805 · 0.607 | AUC는 순위, F1은 0.5 선 기준 — 다른 질문 |
| reg["test_score"] | [-596.6, -628.6, -715.7, -694.9, -563.3] | neg_ → 음수로 나옴 (클수록 좋게 통일) |
| rmse = -reg["test_score"]; rmse.mean() | 639.83 | Linear 평균 RMSE (ms) |
| KNN15: -test_score 평균 | 598.85 | CV에선 KNN15가 Linear보다 작음 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| scoring={"f1": ..., "auc": ...} | 같은 fold에서 여러 지표 | 결과 키가 test_f1, test_auc로 생김. 따로 돌리면 학습을 두 번 해서 느리고, fold가 같은지도 따로 챙겨야 함 |
| "neg_root_mean_squared_error" | "클수록 좋다" 관례 때문에 음수 | −500이 −700보다 좋음 (RMSE 500 < 700). sklearn이 모든 점수를 "최대화"로 통일해서 생긴 규칙 → 보고할 땐 부호를 되돌림 |
| return_train_score=True | train 점수도 함께 | train과 fold 평가의 격차로 과적합 진단 |
| error_score="raise" | 실패를 조용히 숨기지 않음 | 기본값이면 실패 fold가 NaN으로 들어가 평균이 이상해져도 모를 수 있음 |

⚠️ test_f1의 "test"는 fold 평가 12건이지 보관한 test 20이 아니에요. CV 평균이 조금 높다고 통계적으로 우수하다고 단정하지 않아요.

### np.log1p · FunctionTransformer

```
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

def add_features(X):
    out = X.copy()                                              # 원본은 건드리지 않음
    out["log_input_chars"] = np.log1p(out["input_chars"])
    out["chars_per_slot"] = out["input_chars"] / (1.0 + out["allowed_tools"])
    return out

pipe = Pipeline([
    ("features", FunctionTransformer(add_features, validate=False)),
    ("pre", make_pre(engineered=True)),         # 결측 대체·표준화·One-hot (수치 4열로 늘어남)
    ("model", LogisticRegression(C=1, max_iter=1000))])
pipe.fit(X_train, y_train)
print(pipe[:-1].transform(X_train).shape)       # (60, 10) = 수치 4 + 원핫 6
print(pipe[:-1].transform(X_valid).shape)       # (20, 10)  NaN·inf 없음

# 파생 특성의 효과만 분리해서 비교 (같은 Logistic C=1 · 같은 5-fold · 같은 전처리)
for name in ["Logistic", "Logistic_FE"]:
    pipe = candidates()[name]
    cv = cross_validate(pipe, X_train, y_train, cv=CLS_FOLDS, scoring="f1", n_jobs=1)
    pipe.fit(X_train, y_train)
    print(name, cv["test_score"].mean(), f1_score(y_valid, pipe.predict(X_valid)))

# 각 fold의 중앙값이 그 fold 학습 48건(파생 특성 포함)에서 왔는지 검사
cv = cross_validate(pipe, X_train, y_train, cv=CLS_FOLDS, scoring="f1", return_estimator=True)
for (fit_idx, _), fitted in zip(CLS_FOLDS, cv["estimator"]):
    expected = add_features(X_train.iloc[fit_idx])[NUM + DERIVED].median()
    actual = fitted["pre"].named_transformers_["num"]["imputer"].statistics_
    assert np.allclose(expected.to_numpy(), actual)   # 글자수 중앙값 1309 / 1303 / 1315 / 1315 / 1338
```

▶ 실제로 돌려 보면 강의 requests.csv · train 60 / valid 20

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| np.log1p(0) · np.log(0) | 0 · -inf | 글자수 0이어도 log1p는 안전 |
| np.log1p(np.nan) | nan | 결측은 결측으로 남김 → 뒤의 imputer가 채움 |
| pipe[:-1].transform(X_train).shape | (60, 10) | 수치 4(원본 2 + 파생 2) + 원핫 6 |
| pipe[:-1].transform(X_valid).shape · NaN/inf 개수 | (20, 10) · 0 | valid도 같은 10열, 모두 유한값 |
| add_features(X) 뒤 원본 X 열 수 | 4 | X.copy() 덕분에 원본은 4열 그대로 |
| Logistic: CV F1 · std · valid F1 | 0.522 · 0.209 · 0.333 | 기존 |
| Logistic_FE: CV F1 · std · valid F1 | 0.551 · 0.190 · 0.182 | CV는 조금 오르고 valid는 떨어짐 — 노트 표와 같음 |
| fold별 글자수 중앙값 · 검사 통과 | [1309, 1303, 1315, 1315, 1338] · True | fold마다 자기 학습 48건에서 계산 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| np.log1p(x) | log(1 + x) — x=0이어도 안전 | np.log는 0에서 −inf. NaN은 NaN으로 남겨 뒤의 imputer가 학습 규칙대로 채우게 함 |
| X.copy() | 원본 DataFrame 보호 | 빼면 넘겨받은 원본에 열이 추가돼, 다음 실험·fold에서 이미 열이 있는 자료를 쓰게 됨 |
| FunctionTransformer(func) | 직접 만든 함수를 Pipeline 단계로 | 함수는 모듈에 이름 있는 함수로 (저장·복원용) |
| validate=False | DataFrame 입력을 그대로 유지 | True면 배열로 바뀌어 out["input_chars"] 같은 열 이름 사용 불가 |
| pipe[:-1] | 마지막(모델) 전까지만 실행 | 변환 결과 shape 확인용 |
| make_pre(engineered=True) | 강의 코드의 사용자 정의 함수 — 수치 4열(원본 2 + 파생 2) 경로 | False면 원본 수치 2열만 → (60, 8) |

| Logistic C=1 | 기존 | 파생 2개 추가 |
|---|---|---|
| CV F1 | 0.522 | 0.551 |
| CV 표준편차 | 0.209 | 0.190 |
| valid F1 | 0.333 | 0.182 |

💡 CV는 0.029 올랐지만 변동이 크고 valid는 떨어졌어요. 특성을 추가했다고 항상 좋아지지 않으니, 결과와 한계를 모두 보고해요.

🤔 왜 CV는 오르고 valid는 떨어질 수 있지? — CV 0.029 상승은 fold 표준편차(약 0.19)에 비하면 아주 작은 차이라 우연일 수 있어요. valid는 실패 6건뿐이라 한 건만 달라져도 F1이 크게 움직여요. 둘 다 작은 자료의 "흔들림" 범위 안이라, 어느 쪽 숫자만 골라 "개선"이나 "악화"로 확정하지 않아요. 그래서 비교할 땐 모델·fold·전처리는 고정하고 파생 특성만 바꿔요.

### KMeans · silhouette_score

```
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X = np.array([[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]], dtype=float)
km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels = km.fit_predict(X)              # y를 넣지 않는다!  → [0 0 0 1 1 1]
print(km.cluster_centers_, km.inertia_) # (1.333,1.333) (8.333,8.333) · 2.667

cluster_pipe = Pipeline([                # 실제 자료: 거리 계산 전에 결측 대체 + 표준화
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("cluster", KMeans(n_clusters=3, n_init=10, random_state=42))])
cluster_pipe.fit(X_train[NUM])           # train 수치 2열만
print(np.bincount(cluster_pipe["cluster"].labels_))      # [24 19 17]

Z = cluster_pipe[:-1].transform(X_train[NUM])            # 같은 표준화 공간에서
print(silhouette_score(Z, cluster_pipe["cluster"].labels_))   # K=3: 0.453

new = pd.DataFrame({"input_chars": [200., 2200.], "allowed_tools": [1., 5.]})
before = cluster_pipe["cluster"].cluster_centers_.copy()
print(cluster_pipe.predict(new))         # [2 0]  새 요청 → 기존 중심에 할당 (fit_predict 금지)
print(np.array_equal(before, cluster_pipe["cluster"].cluster_centers_))   # True  중심 불변
```

▶ 실제로 돌려 보면 노트의 6점 + 강의 requests.csv · train 60 / valid 20의 수치 2열

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| km.fit_predict(X) | [0, 0, 0, 1, 1, 1] | y 없이 가까운 것끼리 |
| km.cluster_centers_ · km.inertia_ | [[1.333, 1.333], [8.333, 8.333]] · 2.667 | 각 묶음의 평균점 · 중심까지 제곱거리 합 |
| np.bincount(cluster_pipe["cluster"].labels_) | [24, 19, 17] | train 60건이 24·19·17로 |
| 중심을 원래 단위로 (scaler.inverse_transform) | (1542자, 도구 4.1) / (1430자, 도구 1.4) / (445자, 도구 2.7) | 긴 요청·도구 많음 / 긴 요청·도구 적음 / 짧은 요청 |
| K=2 · 3 · 4: silhouette | 0.392 · 0.453 · 0.445 | K=3이 가장 높음 |
| K=2 · 3 · 4: inertia | 69.9 · 40.4 · 30.4 | K를 늘리면 거의 항상 줄어듦 → inertia만으로 K를 못 고름 |
| silhouette를 원래 단위 X로 채점하면 | 0.173 | 같은 군집인데 점수가 달라짐 — 채점은 Z 공간에서 |
| cluster_pipe.predict(new) | [2, 0] | 200자·도구1 → 짧은 군집, 2200자·도구5 → 긴·도구 많음 |
| np.array_equal(before, 중심) | True | predict는 중심을 안 바꿈 |

| 인자 / 속성 | 뜻 | 바꾸면? |
|---|---|---|
| n_clusters | 군집 개수 K | 늘리면 inertia는 거의 항상 감소 → 후보(2·3·4)를 미리 제한하고 silhouette·해석과 함께 비교 |
| n_init=10 | 초기화를 10번 시도해 inertia가 가장 작은 것 | 줄이면 빠르지만 운 나쁜 시작점의 해에 갇힐 수 있음 |
| random_state | 초기화(k-means++) 재현 | 안 고정하면 실행마다 군집 번호·결과가 바뀔 수 있음 |
| labels_ / cluster_centers_ / inertia_ | 학습 자료의 군집 번호 / 중심 / 제곱거리 합 | 중심은 표준화 공간 좌표 → 원래 단위로 읽으려면 되돌려 해석 |
| silhouette_score(Z, labels) | Z 공간에서 분리도 (−1 ~ 1) | 클러스터링한 것과 같은 표준화 공간의 Z를 넣어야 함 — 원래 단위를 넣으면 다른 거리로 채점하게 됨 |
| predict(new) | 새 입력을 기존 중심에 할당 | fit_predict(new)는 새 자료로 중심을 다시 학습해 버려 기존 군집과 번호·의미가 달라짐 |

🤔 왜 1~9짜리 예제는 표준화 없이 되는데 실제 자료는 필요하지? — 원리 예제는 두 축 단위가 같아요. 실제 요청은 글자수(수백~수천)와 도구 수(1~5)라, 표준화 없이 거리를 재면 KNN 때처럼 글자수만 보고 묶어요. 그래서 거리 계산 전에 train 중앙값으로 결측을 채우고 train 기준으로 표준화해요. 정답 y는 쓰지 않아요.

⚠️ K-Means는 거리 기반이라 표준화가 필요해요(글자수 vs 도구수 단위 차이). 정답을 안 쓴다고 전체 100건으로 fit해도 되는 게 아니에요 — 예측 모델 평가와 연결된다면 train만 써요.

### 설정 저장 · clone 재학습 · 저장/복원

```
import json, joblib
import pandas as pd
from pathlib import Path
from sklearn.base import clone

Path("evaluation_lock.json").write_text(                 # 선택 내용을 먼저 잠근다
    json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

X_dev = pd.concat([X_train, X_valid])                    # 개발 80건
y_dev = pd.concat([y_train, y_valid])
final = clone(candidates()[selected]).fit(X_dev, y_dev)  # 학습 상태 없는 새 Pipeline

p = final.predict_proba(X_test)[:, list(final.classes_).index(1)]
pred = (p > protocol["threshold"]).astype(int)          # 임계값 0.5 고정
# test 결과를 본 뒤 모델·특성·임계값을 바꾸지 않는다

joblib.dump(final, "final_pipeline.joblib")
restored = joblib.load("final_pipeline.joblib")         # 본인이 만든 파일만
assert np.allclose(final.predict_proba(X_check), restored.predict_proba(X_check))
```

▶ 실제로 돌려 보면 강의 requests.csv · 개발 80(train+valid) → test 20 · 선택 모델 Logistic_FE

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| json.dumps({"rule": "점수 > 임계값"}) | {"rule": "\uc810\uc218 > \uc784\uacc4\uac12"} | ensure_ascii 기본 True → 한글이 \uXXXX |
| json.dumps(..., ensure_ascii=False) | {"rule": "점수 > 임계값"} | 사람이 읽을 수 있게 |
| clone(학습된 Pipeline).predict(X) | NotFittedError | clone은 설정만 복사 — 학습 상태는 버림 |
| X_dev.shape · y_dev.mean() | (80, 4) · 0.30 | train 60 + valid 20, 실패 비율 30% |
| 개발 80 재학습 후 글자수 중앙값 | 1278 | train 60 때(1315)와 달라지는 건 정상 |
| test: F1 · AUC · TN FP FN TP | 0.545 · 0.786 · 12 2 3 3 | 실패 6건 중 3건 발견 (Recall 0.5) |
| np.allclose(final.predict_proba, restored.predict_proba) | True | add_features가 이름 있는 모듈 함수라 복원 성공 |

| 명령 | 뜻 | 주의 |
|---|---|---|
| json.dumps(..., ensure_ascii=False, indent=2) | 설정을 사람이 읽기 좋은 JSON으로 | 한글이 \uXXXX로 깨지지 않게 |
| clone(estimator) | 설정만 같고 학습은 안 된 새 객체 | 이미 학습된 객체를 재사용하지 않음 |
| pd.concat([A, B]) | train + valid를 고정 순서로 결합 | 설정 확정 뒤에만 |
| joblib.dump / load | 변환 함수 포함 Pipeline 저장·복원 | add_features는 모듈(feature_ops.py)에 정의해 import해야 복원됨. 본인이 만든 파일만 load |
| failure_scores(model, X) | 강의 코드의 사용자 정의 함수 — classes_에서 실패(1) 열을 골라 확률 반환 | 열 번호를 1로 박아 두면 클래스 순서가 다를 때 성공 확률을 쓰게 됨 |

🤔 왜 함수를 모듈에 이름 붙여 둬야 하지? — joblib은 Pipeline을 저장할 때 add_features의 코드 자체가 아니라 "어느 모듈의 어떤 이름의 함수"라는 주소만 적어 둬요. 노트북 셀이나 lambda로 만든 함수는 새 프로세스에서 그 주소를 찾을 수 없어서 복원이 실패해요. 그래서 feature_ops.py에 정의하고 import해서 쓰고, 코드·버전·임계값도 함께 보관해요.

### 새 입력 검사 (새 범주 vs 필수 열 누락)

```
checked = check_input(new_requests)          # 강의 코드의 사용자 정의 검사 함수
num = fitted["pre"].named_transformers_["num"]
before = num["imputer"].statistics_.copy()
p = failure_scores(fitted, checked)          # 새 범주가 있어도 (2, 10)으로 변환 · 확률은 유한값
assert np.array_equal(before, num["imputer"].statistics_)   # 통계 불변
check_input(new_requests.drop(columns="priority"))         # ← ValueError가 나야 정상 (실패 시험)
```

▶ 실제로 돌려 보면 valid 앞 2행 · 1행 글자수를 결측으로, task_type을 처음 보는 'translation'으로

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| fitted[:-1].transform(checked).shape | (2, 10) | 결측·새 범주가 있어도 10열 |
| np.isfinite(Z).all() | True | NaN/inf 없음 — 결측은 학습 중앙값으로 |
| failure_scores(fitted, checked) | [0.397, 0.259] | 확률 2개, 유한값 |
| np.array_equal(before, num["imputer"].statistics_) | True | 예측해도 통계 불변 |
| check_input(new_requests.drop(columns="priority")) | ValueError: 필수 열 누락: ['priority'] | 필수 열 누락 → 예측 전에 멈춤 (실패해야 정상) |
| roc_auc_score(실패 0건인 y, 점수) | nan · 경고 UndefinedMetricWarning → 기록 None, 실제 양성 0 | nan을 그대로 평균에 섞지 않고, 클래스 수를 먼저 확인해 None + 양성 수로 기록 |

| 상황 | 기대 결과 | 판정 근거 |
|---|---|---|
| 정상 입력 20건 | 확률·예측 각각 20개 | ID·행 순서 유지, 지표 재계산 일치 |
| 수치 결측 · 새 범주 | 원래 경로로 변환·예측 | 10열 유지 · NaN/inf 없음 · 통계 불변 |
| priority 필수 열 누락 | 예측 전에 ValueError | 임의 값으로 조용히 보완하지 않음 |
| 한 클래스뿐인 평가 | AUC 계산 불가 표시 | None/null + 실제 양성 수 기록 |
| 동결 뒤 설정 변경 | 새 버전 또는 평가 중단 | 기존 최종 성능과 구분해 기록 |

🤔 왜 "실패해야 정상"인 시험도 만들지? — 코드가 멈추지 않았다는 것과 정한 정책대로 동작했다는 건 달라요. 필수 열이 빠졌는데도 조용히 예측을 내놓는다면, 그건 "잘 돌아간 것"이 아니라 틀린 답을 숨긴 것이에요. 그래서 기대 결과(오류가 나야 함)를 먼저 적고 그대로 되는지 확인해요.

⚠️ 최종 평가의 경계

- test F1이 낮다고 KNN으로 바꿔 다시 평가하면 test를 선택에 쓴 것. 기존 점수는 남기고 새 독립 평가 자료가 필요해요.

- 80건 재학습 후 중앙값이 바뀌는 건 정상 (허용한 자료로 새로 fit했으니까).

- 같은 모델·같은 입력으로 결과 파일만 다시 만드는 건 누수가 아니라 재현 확인.

- test 20건 · 실패 6건의 소표본, 합성 데이터라는 한계를 같이 적어요. 실패 6건 중 3건을 놓쳤고, 이 결과만으로 실서비스 배포 가능성을 보장하지 않아요.

- CV 점수와 최종 test 점수를 같은 성능이라고 합치지 않아요.

🤔 왜 AUC 0.786인데 실패는 절반만 찾았지? — AUC는 "실패·성공 쌍에서 실패 점수가 더 높은 비율"이라 순위가 꽤 괜찮다는 뜻이에요. 하지만 실제로 경고를 낼지는 임계값 0.5가 정해요. 순위상 위쪽에 있어도 확률이 0.5를 못 넘은 실패는 놓쳐요(Recall 0.5). 좋은 순위와 좋은 운영 기준은 따로 점검해야 하는 이유예요.

### MAE · RMSE 계산 mean_absolute_error · root_mean_squared_error

```
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
y = [100, 200, 300, 400]
a = [100, 200, 300, 480]
b = [130, 230, 330, 430]
for name, pred in [("A", a), ("B", b)]:
    print(name, mean_absolute_error(y, pred), root_mean_squared_error(y, pred))
# A 20.0 40.0  /  B 30.0 30.0
```

▶ 실제로 돌려 보면 실제 100·200·300·400

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| mean_absolute_error(y, A) | 20 | A 오차 (0, 0, 0, 80) → 평균 20 |
| root_mean_squared_error(y, A) | 40 | 80이 제곱돼 6400 → 큰 오차에 민감 |
| mean_absolute_error(y, B) | 30 | B 오차는 고르게 30 |
| root_mean_squared_error(y, B) | 30 | 고른 오차는 MAE = RMSE |

## 📖 Day8 용어 (36개)
- **Precision** (Precision (정밀도)): 양성이라고 한 것 중 진짜 양성인 비율. TP ÷ (TP + FP). 헛경고가 늘면 낮아져요.
- **Recall** (Recall (재현율)): 실제 양성 중 모델이 찾아낸 비율. TP ÷ (TP + FN). 놓침이 늘면 낮아져요.
- **F1** (F1 score): Precision과 Recall의 조화평균. 2TP ÷ (2TP + FP + FN). 둘 중 하나라도 낮으면 같이 낮아지고 TN은 반영하지 않아요. 이 수업 분류의 주 지표.
- **조화평균** (Harmonic mean (조화평균)): 2·P·R ÷ (P + R)처럼 계산하는 평균. 두 값 중 하나라도 낮으면 결과도 낮아져서, 한 지표만 극단적으로 올리는 꼼수를 막아요.
- **pos_label** (pos_label (양성 클래스 지정)): Precision·Recall·F1을 어느 클래스 기준으로 계산할지 정하는 인자. 기본값은 1(=실패).
- **ROC 곡선** (Receiver Operating Characteristic curve (ROC 곡선)): 임계값을 1 → 0으로 내리면서 가로축 FPR, 세로축 TPR을 찍어 이은 곡선. 임계값 하나가 아니라 모든 임계값에서의 성능을 한눈에 보여 줘요.
- **AUC** (Area Under the ROC Curve (ROC 곡선 아래 넓이)): 양성 1건·음성 1건을 뽑았을 때 양성 쪽 점수가 더 높을 확률. 점수의 '순위'만 평가하므로 임계값을 바꿔도 그대로예요. 0.5는 무작위 수준, 1은 완벽. 발견 비율(Recall)이 아니에요.
- **FPR** (False Positive Rate (거짓 양성 비율)): 실제 음성 중 잘못 양성이라고 한 비율. FP ÷ (FP + TN). ROC 곡선의 가로축.
- **TPR** (True Positive Rate (참 양성 비율)): 실제 양성 중 찾아낸 비율. TP ÷ (TP + FN). Recall과 같아요. ROC 곡선의 세로축.
- **확률 보정** (Probability calibration (확률 보정)): 모델이 낸 확률이 실제 빈도와 맞는지(0.8짜리들 중 정말 80%가 양성인지) 확인하고 맞추는 것. 확률 0.8이 곧 '80% 확실'은 아니에요.
- **불균형 자료** (Imbalanced data (클래스 불균형 자료)): 한 클래스가 다른 클래스보다 훨씬 많은 자료(예: 성공 14 · 실패 6). Accuracy가 속기 쉬워 실제 양성 수, Precision·Recall 관점도 함께 봐요.
- **OOF** (Out-of-fold prediction (폴드 밖 예측)): 교차검증에서 각 행을 그 행을 학습하지 않은 fold 모델로 예측해 모은 것. 모아서 한 번에 계산한 점수는 fold 점수 평균과 다를 수 있어요.
- **nested CV** (Nested cross-validation (중첩 교차검증)): 후보를 고르는 교차검증을 바깥 교차검증 안에 한 번 더 넣어, 고른 최고 점수의 낙관성까지 평가하는 방법.
- **특성 설계** (Feature Engineering (특성 설계)): 원본 입력을 변환·조합해 모델이 쓰기 좋은 새 특성을 만드는 일(예: log 변환, 비율). 예측 시점에 알 수 있는 값만 써야 하고, 효과는 같은 조건 비교로 확인해요.
- **파생 특성** (Derived feature (파생 특성)): 기존 열로 계산해 새로 만든 특성. 예: log_input_chars = log(1 + 글자수), chars_per_slot = 글자수 ÷ (1 + 허용 도구수).
- **log1p** (log(1 + x) 변환): x에 1을 더한 뒤 로그를 씌우는 함수. 값이 0이어도 안전하고(log(1)=0), 99 → 4.6, 999 → 6.9처럼 큰 값의 차이를 완만하게 줄여요.
- **FunctionTransformer** (FunctionTransformer (함수 변환 단계)): 직접 만든 함수(예: add_features)를 Pipeline의 한 단계로 넣어 주는 도구. 학습·예측 때 같은 변환이 자동으로 적용돼요.
- **지도학습** (Supervised learning (지도학습)): 입력과 정답(y)을 함께 주고 정답을 맞히도록 학습하는 방식. 회귀·분류, KNN이 여기에 속해요.
- **비지도학습** (Unsupervised learning (비지도학습)): 정답(y) 없이 입력만 보고 데이터 속 구조(비슷한 것끼리의 묶음 등)를 찾는 방식. K-Means가 대표.
- **K-Means** (K-Means clustering (K-평균 군집화)): K개 중심을 정해 각 점을 가장 가까운 중심에 할당하고, 중심을 소속 점들의 평균으로 옮기는 과정을 반복해 데이터를 K개 군집으로 묶는 비지도학습. 거리 기반이라 표준화가 필요해요.
- **군집** (Cluster (군집)): 서로 가까운(비슷한) 데이터끼리 묶인 그룹. 군집 번호 0·1·2는 순서나 성공/실패 의미가 없는 이름표예요.
- **군집 중심** (Cluster center / Centroid (군집 중심)): 한 군집에 속한 점들의 평균 위치. 새 입력은 가장 가까운 중심의 군집에 할당돼요.
- **k-means++** (k-means++ (초기 중심 선택법)): 처음 K개 중심을 서로 멀리 떨어지게 뽑아 더 나은 결과를 얻기 쉽게 하는 K-Means의 초기화 방법.
- **n_clusters** (n_clusters (군집 개수 K)): K-Means가 만들 군집의 개수. KNN의 K(이웃 수)와는 뜻이 달라요.
- **n_init** (n_init (초기화 시도 횟수)): 시작 중심을 바꿔 K-Means를 몇 번 돌릴지. 그중 inertia가 가장 작은 결과를 써요. 운 나쁜 시작점에 갇히는 걸 줄여요.
- **inertia** (Inertia (군집 내 제곱거리 합)): 각 점에서 소속 군집 중심까지 거리를 제곱해 모두 더한 값. 작을수록 촘촘하지만 K를 늘리면 거의 항상 줄어서 이것만으로 K를 고르면 안 돼요.
- **silhouette** (Silhouette score (실루엣 점수)): 내 군집과는 가깝고 옆 군집과는 먼지를 −1~1로 나타낸 점수. 높을수록 군집이 잘 나뉜 것. 업무적 의미는 따로 해석해야 해요.
- **fit_predict** (fit_predict (학습 + 할당 한 번에)): 주어진 자료로 군집 중심을 학습하고 바로 군집 번호를 돌려주는 함수. 새 입력에 쓰면 중심을 다시 배워 버리니, 새 입력은 predict로 기존 중심에 할당해요.
- **PCA** (Principal Component Analysis (주성분 분석)): 여러 특성을 정보 손실이 적게 더 적은 개수의 새 축으로 압축하는 비지도 변환. 자료에서 배우는 변환이라 학습 자료로만 fit해야 해요.
- **특성 선택** (Feature selection (특성 선택)): 여러 특성 중 쓸 열만 골라내는 것. 평가 자료를 보고 고르면 누수·선택 편향이 생기니 학습 fold 안에서 해야 해요.
- **동결** (Freeze / Lock (설정 동결)): test를 열기 전에 모델·특성·임계값·선택 규칙을 파일로 기록해 더 이상 바꾸지 않는 것. test 점수를 보고 바꾸면 test가 선택용 자료가 돼 버려요.
- **재학습** (Refit (재학습)): 선택을 마친 뒤 같은 설정의 새 Pipeline을 train+valid(개발 80건)로 다시 학습하는 것. 그 뒤 test 20건으로 딱 한 번 평가해요.
- **clone** (sklearn.base.clone (학습 안 된 복사본)): 설정은 같지만 학습 상태는 없는 새 모델 객체를 만드는 함수. 이전에 학습한 값(중앙값 등)이 섞이지 않게 해요.
- **JSON** (JavaScript Object Notation): 설정이나 결과를 사람과 프로그램이 모두 읽기 쉬운 텍스트로 저장하는 형식. ensure_ascii=False면 한글이 깨지지 않아요.
- **ValueError** (ValueError (값 오류)): 입력 값이 약속과 다를 때 파이썬이 내는 오류. 필수 열이 빠진 입력은 예측 전에 이 오류로 분명하게 막아요.
- **matplotlib** (Matplotlib (그래프 라이브러리)): ROC 곡선, 군집 그림 같은 그래프를 그리는 파이썬 라이브러리.

---

# Day9 · GPU·Tensor·ANN
_2026-10-29(목) · GPU·CUDA · Tensor · Autograd · ANN · 학습 루프_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day09.html

📌 딱 3줄 요약

- 딥러닝은 큰 숫자 표(행렬) 계산을 엄청 반복해요. 그래서 GPU를 쓰고, torch.cuda.is_available()로 진짜 쓸 수 있는지 확인해요.

- 데이터는 전부 Tensor예요. 값만 보지 말고 shape · dtype · device를 같이 읽어야 해요.

- 학습은 zero_grad → forward → loss → backward → step 다섯 단계를 batch마다 반복하는 거예요. 기울기는 쌓이니까 매번 비우고, 평가할 땐 model.eval() + torch.no_grad()를 같이 써요.

🔧 Day9 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| torch | Tensor 계산, 자동 미분, 신경망 층, Optimizer까지 딥러닝의 모든 것 | import torch, from torch import nn |
| (torch 안에 포함) | 데이터를 batch로 묶어 공급 | from torch.utils.data import TensorDataset, DataLoader |

이날 강의 코드는 torch 하나로 다 돼요. 가상환경(venv)을 만들고 그 안에 설치해요.

### 🔧 설치 · 확인

```
python --version
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# torch 설치 명령은 https://pytorch.org/get-started/locally/ 에서 골라 복사
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"   # True면 GPU 사용 가능
```

⚠️ torch는 pytorch.org의 Start Locally 선택기에서 OS · Pip · Python · Compute Platform(CUDA 버전 또는 CPU)을 고른 뒤 나온 명령으로 설치해요. 그냥 pip install torch만 치면 내 GPU의 CUDA와 안 맞는 빌드가 깔릴 수 있어요. 이미 잘 돌아가는 환경이면 버전 숫자만 보고 다시 깔지 마세요.

🤔 왜 설치 명령을 외우지 않아요? — OS · Python · 드라이버 · CUDA 조합에 따라 맞는 torch 빌드가 달라서, 설치 명령도 달라져요. 그래서 강의일에 공식 선택기가 알려 주는 조합을 그대로 쓰고, 마지막엔 torch.cuda.is_available()로 실제로 되는지 확인해요. GPU가 없거나 False여도 CPU로 같은 코드를 돌릴 수 있어요.

## 이거 어디에 써?

Day8까지의 머신러닝은 사람이 특성을 만들어서 모델에 넣었어요. 딥러닝은 층(layer)을 여러 개 쌓아서 모델이 스스로 쓸 만한 표현을 배우게 해요. 사진·글·소리처럼 사람이 특성을 일일이 정하기 어려운 데이터에 강해요.

🖼️

이미지 분류사진이 고양이인지 강아지인지

💬

문장 이해리뷰가 긍정인지 부정인지

📈

표 데이터 예측특성 몇 개로 클래스 맞히기 (오늘의 2→8→2 ANN)

⚡

GPU 학습같은 계산을 동시에 많이 처리해 학습 시간 단축

|  | CPU | GPU |
|---|---|---|
| 생김새 | 소수의 강한 코어 | 아주 많은 연산 유닛 |
| 잘하는 일 | 복잡한 분기, 파일 읽기, 전처리, 프로그램 흐름 | 행렬곱·합성곱처럼 같은 모양의 계산을 대량으로 |
| 비유 | 복잡한 일을 빨리 하는 소수 정예 | 단순 계산을 동시에 하는 대규모 팀 |

⚠️ GPU가 항상 빠른 건 아니에요. CPU 메모리(RAM) → GPU 메모리(VRAM)로 옮기는 데도 시간이 들어요. 작은 계산을 자주 옮기면 오히려 느릴 수 있어요. 결론은 "내 데이터·모델·batch로 직접 재 보기". 그리고 "CPU는 직렬, GPU는 병렬"은 지나친 단순화예요. CPU도 병렬 처리를 해요.

딥러닝의 핵심 계산은 행렬 곱셈이에요. 결과의 칸마다 곱하고 더하는데, 각 칸은 서로 독립이라 동시에 계산할 수 있어요.

```
[1 2]   [5 6]   [19 22]
[3 4] × [7 8] = [43 50]      예: 19 = 1×5 + 2×7
```

[그림: 6x6 결과 행렬을 CPU 4코어는 9차례, GPU는 1차례에 계산하는 모식도] 무엇을 보나: 같은 6×6 결과 행렬(36칸). 숫자는 그 칸이 몇 번째 차례에 계산되는지예요. 왼쪽(코어 4개)은 4칸씩 9차례, 오른쪽(유닛이 아주 많음)은 36칸을 한 차례에 끝내요. 🤔 왜? 결과 칸 하나 = 행 하나와 열 하나를 곱해 더한 값이라 다른 칸을 기다릴 필요가 없어요(독립). 그래서 일꾼이 많을수록 동시에 끝낼 수 있어요. (모식도예요 — 실제 CPU 코어 수·GPU 유닛 수와 시간은 장치마다 달라요.)

🤔 왜 딥러닝엔 GPU가 유리해? — 학습 한 번은 입력 → 곱하고 더해 예측 → 오차 계산 → 가중치 조금 수정이에요. 이걸 "수많은 데이터 × 수많은 가중치 × 여러 epoch" 만큼 반복하니 연산량이 급격히 커져요. 그런데 결과 행렬의 칸 하나하나는 서로 독립이라 동시에 계산할 수 있어요. GPU는 단순한 계산을 동시에 처리하는 연산 유닛이 아주 많아서, 이렇게 "같은 모양의 독립 계산이 충분히 많을 때" 처리량(Throughput)이 커져요. 반대로 작업이 적거나 조건 분기가 많으면 그 장점이 안 살아요.

[그림: CPU에서 스레드 1개와 여러 개로 행렬곱 시간을 잰 그래프] 무엇을 보나: 이 노트를 만든 PC의 CPU에서만 잰 행렬곱 시간이에요(GPU 측정 아님). 회색 = 스레드 1개로 한 줄씩, 보라 = 같은 CPU의 여러 스레드로 나눠 동시에. 행렬이 커질수록 회색은 가파르게 늘고, 보라는 여러 배 빨라요. 🤔 왜? n이 2배가 되면 곱셈 수는 약 8배(n³)라 계산량이 급격히 커져요. 칸끼리 독립이라 일꾼을 늘리면 그만큼 나눠 처리할 수 있어요 — GPU는 이 아이디어를 수천 개 유닛으로 밀어붙인 장치예요. 작은 행렬(왼쪽 끝)에선 차이가 거의 없죠? 일이 적으면 나누는 이득도 작아요.

데이터 준비 CPU가 파일 읽기·전처리→ 순전파 GPU가 행렬 계산→ 오차·역전파 GPU가 기울기 계산→ 가중치 갱신 GPU에서 값 수정

이 과정을 수백·수천 batch에 반복하니까 GPU 병렬 연산의 이점이 누적돼요. 파일 읽기·전처리 같은 순차 작업은 여전히 CPU 몫이에요.

## ① GPU·CUDA 환경 — "GPU가 있다" ≠ "PyTorch가 GPU를 쓴다"

GPU 학습은 네 층이 모두 맞아야 돼요. 하나만 어긋나도 GPU를 못 써요.

NVIDIA GPU 연산 장치 · VRAM→ GPU Driver OS와 GPU 연결→ CUDA Runtime GPU 연산 API→ PyTorch Tensor · Autograd · 신경망

🤔 왜 "GPU가 있다"만으론 부족해? — 위 네 층이 차례로 연결돼야 PyTorch 코드가 GPU에서 돌아요. GPU가 꽂혀 있어도 드라이버가 안 맞거나, PyTorch가 CPU 전용 빌드면 중간에서 끊겨요. 한 층이 정상이어도 다른 층이 안 맞으면 GPU 학습이 안 될 수 있어서, 확인도 GPU 확인 → 드라이버·CUDA 확인 → PyTorch 설치 → 실행 가능 여부 확인 순서로 따로따로 해요.

🔧 nvidia-smi (터미널) 명령어 사전에서 자세히 →

```
nvidia-smi
```

🔧 PyTorch에서 GPU 확인 명령어 사전에서 자세히 →

```
print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

🔧 CPU·GPU 둘 다 되는 코드 패턴 명령어 사전에서 자세히 →

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

[그림: ] 무엇을 보나: RAM(왼쪽)과 VRAM(오른쪽)은 따로 떨어진 두 창고예요. .to(device)는 화살표처럼 한쪽 창고에서 다른 쪽으로 복사해요. 🤔 왜? GPU는 자기 창고(VRAM)에 있는 숫자만 바로 계산할 수 있어요. 그래서 model·xb·yb를 모두 같은 쪽에 둬야 하고, 복사에는 시간이 드니 batch 단위로 한 번에 옮겨요.

## ② Tensor — 딥러닝의 공통 데이터 상자

Tensor는 차원을 가진 숫자 배열이에요. NumPy 배열이랑 비슷한데, GPU에 올릴 수 있고 미분도 자동으로 해 줘요.

| 차원 | 예 | shape |
|---|---|---|
| 0D 스칼라 | 3.14 | () |
| 1D 벡터 | [1, 2, 3] | (3,) |
| 2D 행렬 | [[1,2],[3,4]] | (2, 2) |
| 3D 이상 | batch · 이미지 · 문장 | shape로 의미를 읽어요 |

### 딥러닝에서 자주 보는 shape

| 대상 | shape | 뜻 |
|---|---|---|
| 표 데이터 | (batch, feature) | 여러 행의 특성 |
| 분류 정답 | (batch,) | 샘플마다 클래스 번호 하나 |
| 분류 logits | (batch, class) | 클래스별 원시 점수 |
| 흑백 이미지 | (batch, channel, height, width) | CNN 입력 |
| 문장 임베딩 | (batch, sequence, feature) | RNN·Transformer 입력 |

[그림: 스칼라, 벡터, 행렬, 3D, 4D 텐서를 블록으로 쌓은 그림] 무엇을 보나: 칸 하나 = 숫자 하나. 0D(점) → 1D(한 줄) → 2D(표) → 3D(표를 겹겹이) → 4D(겹친 묶음을 또 여러 개). 오른쪽 점선 상자가 이미지 batch (B, C, H, W): 이미지 2장, 각 장은 빨강·초록·파랑 채널 3장, 채널 하나는 3×3 픽셀. 🤔 왜? shape의 숫자 하나하나가 "이 방향으로 몇 개 쌓였나"예요. 그래서 shape만 읽어도 "이미지 몇 장 · 채널 몇 개 · 크기 얼마"를 알 수 있고, 층이 기대하는 방향과 어긋나면 오류가 나요.

💡 Tensor를 볼 땐 네 가지를 같이 봐요: 값 · shape(모양) · dtype(자료형) · device(어디에 있나), 그리고 미분 기록 여부 requires_grad.

🤔 왜 값만 보면 안 돼? — 같은 숫자라도 shape가 (batch, feature)인지 (batch,)인지에 따라 "여러 행의 특성"인지 "샘플마다 정답 하나"인지 뜻이 달라져요. 층은 정해진 shape만 받고(예: Linear는 마지막 차원), 손실함수는 정해진 dtype을 요구하고, 연산은 같은 device끼리만 돼요. 그래서 오류의 대부분이 이 네 가지 중 하나가 어긋나서 생겨요.

🔧 Tensor 만들기 torch.tensor() 명령어 사전에서 자세히 →

```
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.long)
```

🔧 난수 · 0 · 1로 채우기 명령어 사전에서 자세히 →

```
a = torch.randn(2, 3)                       # 평균 0, 분산 1 정규분포 난수
b = torch.zeros(2, 3)                       # 0으로 채움
```

### Broadcasting과 shape 바꾸기

X가 (2, 3)이고 bias가 (3,)이면, X + bias는 bias를 행마다 자동으로 늘려서 더해 줘요(결과 (2, 3)). 이걸 broadcasting이라고 해요. 편하지만, 의도와 다른 shape끼리도 계산돼 버릴 수 있으니 shape부터 확인해요.

[그림: (2,3) 행렬에 (3,) 벡터를 더하면 벡터가 행마다 복사된 것처럼 더해지는 그림] 무엇을 보나: bias [10, 20, 30]은 한 줄뿐인데, 아래 점선 줄처럼 행마다 복사된 것처럼 더해져 결과가 (2, 3)이 돼요. 🤔 왜? 마지막 차원(3)이 서로 같으니 PyTorch가 "모자란 행 방향으로 늘려도 된다"고 판단해요. 실제 복사는 하지 않아 메모리는 아껴요. 반대로 의도치 않은 shape끼리도 이렇게 늘어나 버릴 수 있으니 shape 확인이 중요해요.

| 연산 | 하는 일 | 언제 / 주의 |
|---|---|---|
| x.reshape(…) | 원소 개수는 그대로, 모양만 바꿔 봄 | 원소 수가 안 맞으면 오류 |
| x.view(…) | 메모리 배치가 허락하면 같은 데이터를 다른 모양으로 봄 | 메모리 배치가 안 맞으면 오류 → 그땐 reshape |
| x.transpose(0, 1) | 두 차원의 순서를 바꿈 | 행↔열 바꾸기 |
| x.unsqueeze(0) / x.squeeze() | 크기 1인 차원을 추가 / 제거 | 샘플 1개를 (1, feature) batch로 만들 때 |

🤔 왜 shape를 바꿔야 해? — 층과 함수는 정해진 모양을 기대해요. 예를 들어 모델은 (batch, feature)를 받는데 샘플 하나는 (feature,)라서 unsqueeze(0)로 batch 차원을 붙여 줘야 해요. broadcasting은 작은 Tensor를 규칙에 따라 자동 확장하는 거라, 모양이 의도와 달라도 오류 없이 계산될 수 있어요. 그래서 shape부터 확인하는 습관이 중요해요.

🔧 장치 옮기기 .to(device) 명령어 사전에서 자세히 →

```
X = torch.randn(4, 2).to(device)
model = nn.Linear(2, 1).to(device)
```

## ③ Autograd와 경사하강 — 미분은 PyTorch가 대신

PyTorch는 계산을 하면서 "무엇과 무엇을 곱하고 더했는지"를 계산 그래프로 기록해요. 마지막에 backward()를 부르면 그 기록을 거꾸로 따라가며(연쇄법칙, chain rule) 기울기를 자동으로 구해요.

```
y = x² + 2x + 1       →   dy/dx = 2x + 2
x = 3 이면  y = 16,  dy/dx = 8
```

🤔 왜 그래프로 기록해? — 신경망은 곱셈·덧셈·ReLU 같은 작은 연산이 수천 개 이어진 거예요. 각 연산의 미분은 쉬우니까, "어떤 순서로 연결됐는지"만 기억해 두면 마지막 값에서 입력·파라미터 방향으로 연쇄법칙을 한 칸씩 곱해 가며 모든 기울기를 한 번에 구할 수 있어요. 사람이 식을 손으로 미분할 필요가 없어지는 거예요.

[그림: ] 무엇을 보나: y = x² + 2x + 1을 x = 3에서 계산한 계산 그래프. 보라 실선 = 순전파(값 9, 6, 16이 흘러감), 분홍 점선 = y.backward()(기울기가 거꾸로 흘러감). 🤔 왜? 연산 하나하나의 미분은 쉬워요(x²은 2x, 2·x는 2, +는 1). 거꾸로 가며 이걸 곱하고(연쇄법칙), 길이 갈라졌다 합쳐지면 더하면 끝이에요. 그래서 x.grad = 6 + 2 = 8로, 코드 출력 16.0 8.0과 같아요.

| 개념 | 한마디로 |
|---|---|
| requires_grad=True | 이 Tensor의 연산을 Autograd가 추적해요. 학습 파라미터는 보통 True |
| .grad | backward() 후 기울기가 저장되는 곳. 누적돼요 |
| torch.no_grad() | 이 블록 안 연산은 그래프에 기록하지 않음 (평가·예측·수동 갱신) |
| tensor.detach() | 현재 그래프에서 떼어 낸 Tensor |

🔧 자동 미분 backward() 명령어 사전에서 자세히 →

```
x = torch.tensor(3.0, requires_grad=True)
y.backward()
```

기울기 누적 체험 — 직접 눌러 보세요. loss = 2·w라서 backward 한 번마다 기울기 2가 생겨요.

### 경사하강법 (Gradient Descent)

손실을 줄이려면 기울기의 반대 방향으로 조금씩 움직이면 돼요. 얼마나 크게 움직일지가 학습률(learning rate, lr)이에요.

```
parameter = parameter − learning_rate × gradient
```

🤔 왜 "반대" 방향이야? — 기울기(gradient)는 "w를 늘리면 loss가 얼마나 커지나"예요. 기울기가 양수면 w를 줄여야 loss가 작아지고, 음수면 늘려야 해요. 그래서 기울기에 마이너스를 붙여 움직여요. 기울기가 클수록(목표에서 멀수록) 많이, 가까워지면 조금씩 움직여요.

목표값 4에 w를 맞추는 문제 loss = (w − 4)²로 직접 해 봐요. 학습률을 바꾸면 이동 모습이 달라져요.

💡 lr = 0.1이면 w가 0.8 → 1.44 → 1.952 → 2.3616 → 2.6893으로 차근차근 4에 다가가요. 너무 작으면 거의 안 움직이고, 1보다 크면 4를 지나쳐 점점 멀어져요(발산).

[그림: 가중치 2개짜리 loss 등고선 위에서 학습률 3가지의 경사하강 경로] 무엇을 보나: 위 체험을 가중치 2개로 늘린 버전이에요. 위에서 내려다본 그릇 모양 loss(등고선), ■에서 출발해 ★(최소)로 가는 길. 왼쪽은 20걸음에도 반도 못 가고, 가운데는 몇 걸음 만에 도착, 오른쪽은 1→2→3 번호처럼 ★를 지나쳐 반대편으로 튀며 점점 멀어져요. 🤔 왜? 한 걸음 = lr × 기울기. 벽이 가파른 방향(세로)은 기울기가 커서, lr이 크면 그 방향 보폭이 골짜기 폭보다 커져 맞은편 벽으로 넘어가요. 넘어간 곳은 더 가팔라 다음 보폭은 더 커지고… 그래서 발산해요.

🤔 왜 lr이 크면 발산해? — 이 문제에서 기울기는 2(w − 4)라서 한 번 갱신하면 "4까지 남은 거리"가 × (1 − 2·lr)배가 돼요. lr = 0.1이면 ×0.8 → 매번 20%씩 줄어요. lr = 0.8이면 ×(−0.6) → 부호가 바뀌어(4를 넘어감) 왔다 갔다 하지만 크기는 줄어요. lr = 1이면 ×(−1) → 같은 거리로 영원히 왕복. lr > 1이면 |배수| > 1이라 넘어갈 때마다 더 멀어져요. 반대로 lr이 아주 작으면 배수가 1에 가까워 거의 안 움직여요.

🔧 손으로 5번 갱신하기 명령어 사전에서 자세히 →

```
loss.backward()               # 1. 기울기 계산
w -= lr * w.grad          # 2. 기록 없이 값 갱신
```

🔧 torch.no_grad() vs tensor.detach() 명령어 사전에서 자세히 →

```
with torch.no_grad():              # 블록 안 연산은 그래프에 기록 안 함
    logits = model(X)
```

## ④ ANN — 가중합 + 비선형을 층으로 쌓기

인공 뉴런 하나는 입력마다 가중치를 곱해 더하고 편향(bias)을 더한 다음(z = w₁x₁ + w₂x₂ + b), 활성화 함수(ReLU)를 통과시켜요. 이런 뉴런을 여러 개 묶은 게 층, 층을 쌓은 게 ANN이에요.

입력층 특성 2개 · (batch, 2)→ Linear(2, 8) + ReLU 은닉 표현 8개→ Linear(8, 2) 클래스 logits 2개→ argmax 예측 0 또는 1

[그림: ] 무엇을 보나: 오늘 만드는 2→8→2 ANN. 동그라미 = 뉴런(숫자 하나), 선 = 가중치 하나. 은닉층 뉴런 안의 꺾인 표시가 ReLU예요. 🤔 왜? Linear 층은 "앞 층 모든 뉴런 × 뒤 층 모든 뉴런"을 선으로 잇기 때문에 가중치가 in × out개(2×8 = 16)이고, 뉴런마다 bias가 하나씩 붙어요. 그래서 numel() 합이 42가 나와요.

| 단계 | 코드 | 입력 shape | 출력 shape |
|---|---|---|---|
| Linear 1 | nn.Linear(2, 8) | (batch, 2) | (batch, 8) |
| ReLU | nn.ReLU() | (batch, 8) | (batch, 8) |
| Linear 2 | nn.Linear(8, 2) | (batch, 8) | (batch, 2) |
| 예측 | argmax(dim=1) | (batch, 2) | (batch,) |

🤔 왜 ReLU가 필요해? — 음수를 0으로 바꾸는 단순한 함수지만, 이게 없으면 Linear를 아무리 쌓아도 결국 Linear 하나랑 똑같아요(선형 변환 두 번 = 선형 변환 한 번). 층 사이에 비선형성을 넣어야 단순한 선형 변환의 반복을 넘어선 표현을 배울 수 있어요.

[그림: ReLU, sigmoid, tanh 곡선과 ReLU 유무에 따른 모델 출력 비교] 무엇을 보나: 왼쪽은 활성화 함수 세 가지 모양. 오른쪽은 같은 가중치로 만든 1→8→1 모델에서 ReLU만 넣었다 뺐을 때의 출력이에요. 회색(ReLU 없음)은 곧은 직선, 보라(ReLU 있음)는 여러 번 꺾인 선이에요. 🤔 왜? 직선에 직선을 이어 붙여도 직선이에요(Linear 두 번 = Linear 한 번). ReLU가 음수를 0으로 잘라 내면서 뉴런마다 다른 위치에서 "꺾임"이 생기고, 꺾임이 많을수록 복잡한 경계를 그릴 수 있어요.

🤔 왜 batch 차원은 그대로야? — Linear는 입력의 마지막 차원(feature)에만 계산을 해요. 샘플마다 같은 가중치로 따로 계산하니까 샘플 수(batch)는 유지되고, feature 수만 2 → 8 → 2로 바뀌어요. 마지막에 argmax(dim=1)는 클래스 차원을 없애고 "가장 큰 점수의 번호"만 남겨서 (batch,)가 돼요.

🔧 2→8→2 ANN 만들기 명령어 사전에서 자세히 →

```
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
logits = model(X)                 # 순전파: 각 층의 forward를 순서대로 호출
```

🔧 직접 모델 클래스 만들기 nn.Module 명령어 사전에서 자세히 →

```
class SimpleANN(nn.Module):
logits = model(X)                          # model(X) 하면 forward가 실행됨
```

## ⑤ 손실함수와 데이터 공급

### logits · 확률 · 클래스는 다른 값

| 이름 | 예 | 특징 |
|---|---|---|
| logits | [1.8, -0.4] | 모델이 낸 원시 점수. 합이 1일 필요 없음 |
| softmax 확률 | [0.90, 0.10] | 클래스별 가능성. 합이 1 |
| 예측 클래스 | argmax → 0 | 점수가 가장 큰 클래스 번호 |

[그림: logits 1.8, -0.4가 softmax로 0.90, 0.10이 되고 argmax로 0이 되는 막대그래프] 무엇을 보나: 위 표의 숫자 그대로예요. ① logits [1.8, -0.4]는 음수도 있고 합도 1이 아니에요 → ② softmax를 거치면 [0.90, 0.10], 모두 0~1이고 합이 1 → ③ argmax는 큰 쪽 번호 0만 남겨요. 🤔 왜? softmax는 점수마다 e점수를 하고 전체 합으로 나눠요. 그래서 순서(누가 더 큰가)는 그대로 두고 "비율"로 바꿔요. 점수 차 2.2가 확률 차 0.90 vs 0.10으로 벌어지는 것도 지수 때문이에요.

손실(loss)은 "정답에서 얼마나 멀리 있나"를 숫자 하나로 만든 거예요. 학습은 이 숫자를 줄이는 방향으로 가중치를 고쳐요. 정확도(accuracy)는 사람이 보는 성적표이고, loss와는 다른 지표예요.

🤔 왜 accuracy가 아니라 loss로 학습해? — accuracy는 "맞았다/틀렸다"만 세서, 가중치를 살짝 바꿔도 대부분 그대로예요(기울기가 0). loss는 "정답 클래스 점수가 얼마나 높은가"를 연속적인 숫자로 보여 줘서, 조금 좋아져도 조금 줄어들어요. 그래서 미분해서 방향을 정하는 목적함수로 쓸 수 있어요.

[그림: 정답과 오답 점수 차이에 따른 정확도 계단과 CrossEntropy 곡선] 무엇을 보나: 가로축은 "정답 클래스 점수 - 오답 클래스 점수". 주황 정확도는 0에서 1로 계단처럼 뛰고 나머지 구간은 평평해요. 보라 loss는 어디서나 매끈하게 기울어 있어요. 🤔 왜? 경사하강은 "기울기"를 보고 움직여요. 평평한 계단에선 기울기가 0이라 어느 쪽으로 가야 할지 몰라요. loss는 정답 점수를 조금만 올려도 조금 줄어드니, 늘 "이쪽으로 가면 좋아진다"를 알려 줘요.

🔧 분류 손실 nn.CrossEntropyLoss 명령어 사전에서 자세히 →

```
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, y)                      # 스칼라, 작을수록 정답에 가까움
```

### Dataset과 DataLoader

TensorDataset X와 y를 같은 번호로 묶음→ DataLoader batch_size만큼 꺼냄, 섞기→ mini-batch xb (32, 2) · yb (32,)→ 모델 학습 반복

200개를 32개씩 나누면 32 × 6 = 192, 마지막 batch는 8개예요. 마지막 batch는 작을 수 있어요. batch마다 forward · backward · step을 한 번씩 해요.

[그림: 샘플 200개를 색으로 batch 7개에 나눠 표시한 그림] 무엇을 보나: 칸 200개 = 원래 순서대로 놓은 샘플 200개, 색 = 그 샘플이 들어간 batch. 같은 색이 한데 모여 있지 않고 여기저기 흩어져 있어요. 빨강(batch 7)은 8칸뿐이에요. 🤔 왜? shuffle=True라 매 epoch 순서를 섞은 뒤 앞에서부터 32개씩 잘라요. 그래서 각 batch는 데이터 전체를 골고루 닮은 작은 표본이 되고, 200 = 32×6 + 8이라 마지막만 작아요.

🤔 왜 데이터를 batch로 나눠? — 전체를 한 번에 넣으면 메모리가 부족할 수 있고, 한 번 다 보고 나서야 겨우 한 번 갱신해요. 하나씩 넣으면 갱신은 많지만 기울기가 너무 들쭉날쭉하고 GPU 병렬성을 못 살려요. 그 사이인 mini-batch가 속도 · 메모리 · 기울기 흔들림의 균형점이에요(자세한 비교는 다음 수업). Dataset은 "자료가 어떻게 생겼나(구조)", DataLoader는 "어떻게 꺼내 줄까(반복 · batch · shuffle)"로 역할을 나눠요.

🔧 TensorDataset · DataLoader 명령어 사전에서 자세히 →

```
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
```

## ⑥ 학습 루프 — 다섯 단계를 batch마다

1. zero_grad 지난 기울기 비우기→ 2. forward 예측→ 3. loss 오차 계산→ 4. backward 기울기 계산→ 5. step 가중치 갱신

[그림: ] 무엇을 보나: 학습 루프 다섯 단계가 원을 이뤄요. 한 바퀴가 batch 하나(iteration 1번), 모든 batch를 돌면 epoch 1번. 각 칸 아래 작은 글씨가 그 단계에서 실제로 바뀌는 것이에요. 🤔 왜? 기울기(.grad)는 backward 때 더해지기만 하고, 가중치는 step 때만 바뀌어요. 그래서 다음 바퀴 시작 전에 비우고(zero_grad), 새 가중치로 다시 예측해야 "이번 batch"의 기울기로만 고칠 수 있어요.

⚠️ 순서를 바꾸면 지난 기울기가 섞이거나, 기울기 없이 갱신하는 문제가 생겨요. loss.backward() 직후엔 기울기만 계산됐을 뿐이고, 값이 실제로 바뀌는 건 optimizer.step()에서예요.

🤔 왜 꼭 이 순서야? — ① 지난 batch 기울기가 남아 있으면 섞이니까 먼저 비우고(zero_grad), ② 지금 가중치로 예측하고(forward), ③ 정답과 비교해 loss를 만들고, ④ 그 loss로 기울기를 구하고(backward), ⑤ 그 기울기로 가중치를 바꿔요(step). step이 backward보다 먼저 오면 기울기 없이(또는 지난 기울기로) 갱신하게 돼요. 이 5단계만 지키면 대부분의 기본 학습 루프를 읽을 수 있어요.

🔧 준비 + batch 반복 명령어 사전에서 자세히 →

```
loss.backward()
optimizer.step()
```

[그림: 2-8-2 ANN이 epoch 0, 3, 20, 200에서 그린 결정 경계] 무엇을 보나: 위 코드와 같은 2→8→2 모델·SGD(lr=0.1)·batch 32를, 직선으로 못 가르는 초승달 모양 데이터(200개)에 학습시킨 실제 결과예요. 검은 선이 결정 경계. epoch 0엔 아무렇게나, 3에선 대충 직선, 20에선 비스듬히, 200에선 꺾인 선으로 두 달을 갈라요. 🤔 왜? 처음 가중치는 무작위라 경계도 무작위예요. batch마다 step을 반복하며 틀린 점 쪽으로 경계가 조금씩 밀려나고, ReLU 뉴런 8개가 만든 꺾임을 써서 직선 하나로는 안 되는 모양까지 따라가요.

[그림: 같은 학습의 epoch별 평균 train loss 곡선] 무엇을 보나: 바로 위 학습의 epoch 평균 loss. 분홍 점이 위 그림 4장의 시점이에요. 처음 20 epoch에 크게 떨어지고, 그 뒤로는 천천히 내려가요. 🤔 왜? 처음엔 크게 틀린 점이 많아 기울기가 크고 한 번에 많이 고쳐져요. 경계가 자리를 잡으면 남은 건 경계 근처의 어려운 점들이라 기울기가 작아지고 개선도 느려져요. 곡선이 살짝 들쭉날쭉한 건 batch마다 다른 샘플을 보기 때문이에요.

🔧 epoch 평균 loss를 제대로 내기 명령어 사전에서 자세히 →

```
total_loss += loss.item() * len(xb)       # batch 평균 × batch 크기
return total_loss / len(loader.dataset)        # 샘플 수로 가중한 평균 loss
```

🔧 model.train() · model.eval() · torch.no_grad() 명령어 사전에서 자세히 →

```
model.eval()                 # 평가 동작으로 (기울기는 그대로)
with torch.no_grad():        # 기울기 기록 끄기
```

💡 학습 곡선의 loss가 매 epoch 꼭 줄어드는 건 아니에요. 전체 추세를 보고, loss와 accuracy는 따로 읽어요.

## 증상으로 진단

| 이런 일이 생기면 | 원인 | 확인 |
|---|---|---|
| GPU가 있는데 is_available()가 False | CPU용 torch 빌드, 다른 가상환경, 드라이버 문제 | torch.version.cuda가 None인지, 선택기로 재설치 |
| "Expected all tensors to be on the same device" | 모델은 cuda, 입력은 cpu (또는 반대) | .to(device) 결과를 변수에 다시 담았나 |
| shape 오류 (mat1 and mat2 …) | 입력 마지막 차원 ≠ in_features | X.shape와 nn.Linear(in, out) |
| CrossEntropyLoss target 오류 | 정답 dtype이 float | 클래스 번호는 torch.long |
| loss가 이상하게 튐 | zero_grad() 빠뜨려 기울기 누적 | step 전후 .grad 비교 |
| 평가 메모리가 큼 | no_grad() 없이 평가 | 평가 블록을 with torch.no_grad():로 |

## 원본 자료

- 강의: Day9_GPU_Tensor_ANN_20261029_v2.pdf — GPU 개념 p6–16 · 환경 점검 p19–28 · Tensor p31–42 · Autograd p45–53 · ANN p56–64 · 손실·DataLoader p67–74 · 학습 루프 p77–84 · 프로젝트 코드 p92–93 · 오류 상황 p100

- 명령어 이름 확인: PyTorch_치트시트.pdf (1쪽)

- 공식 문서: PyTorch Start Locally (설치 선택기) · CUDA semantics · torch.nn · nvidia-smi

## 🔧 Day9 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day9)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install torch (공식 선택기 명령) |
| ① GPU·CUDA 환경 | nvidia-smi, torch.cuda.is_available(), torch.device() |
| ② Tensor | torch.tensor(), torch.randn(), reshape / view / unsqueeze, .to(device) |
| ③ Autograd · 경사하강 | requires_grad=True, loss.backward(), .grad.zero_(), torch.no_grad(), detach() |
| ④ ANN (인공신경망) | nn.Linear, nn.ReLU, nn.Sequential, nn.Module |
| ⑤ 손실함수 · 데이터 공급 | nn.CrossEntropyLoss, TensorDataset, DataLoader |
| ⑥ 학습 루프 | torch.optim.SGD, optimizer.zero_grad(), optimizer.step(), model.train() / model.eval() |

### nvidia-smi (터미널)

```
nvidia-smi
```

▶ 실제로 돌려 보면 이 PC 기준 — 다른 PC에선 값이 달라요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| GPU Name | NVIDIA GeForce RTX 5080 Laptop GPU | 실제 장치 모델 |
| Driver Version | 592.27 | 설치된 그래픽 드라이버 |
| CUDA Version (표 오른쪽 위) | 13.1 | 드라이버가 지원하는 상한 — PyTorch 빌드 CUDA와 비교해 봐요 |
| Memory-Usage | 0 MiB / 16303 MiB | 지금 쓰는 VRAM / 전체 VRAM |
| GPU-Util | 0 % | 실행 순간 GPU가 얼마나 바쁜지 |

| 출력 항목 | 뜻 |
|---|---|
| GPU Name | 실제 장치 모델 (예: RTX …) |
| Driver / KMD Version | 설치된 그래픽 드라이버 버전 |
| CUDA (UMD) Version | 드라이버가 지원하는 상한일 수 있음. 설치된 CUDA Toolkit과 항상 같진 않아요 (출력 형식은 버전마다 조금 달라요) |
| Memory-Usage | 사용 중인 VRAM / 전체 VRAM (예: 0 / 8192 MiB) |
| GPU-Util | 지금 GPU가 얼마나 바쁜지 (%) |
| Processes | GPU를 쓰고 있는 프로그램 목록 |

⚠️ nvidia-smi가 잘 나와도 PyTorch CUDA가 True라는 보장은 없어요. PyTorch 빌드(CPU용인지 CUDA용인지), 가상환경, 드라이버를 모두 확인해야 해요.

### PyTorch에서 GPU 확인

```
import torch
print("PyTorch:", torch.__version__)
print("빌드 CUDA:", torch.version.cuda)          # CPU 빌드면 None
print("CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():                    # 꼭 확인한 뒤에 조회!
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM(GB):", round(props.total_memory / 1024**3, 1))
```

▶ 실제로 돌려 보면 이 PC 기준 — 다른 PC에선 값이 달라요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| torch.__version__ | 2.14.0+cu130 | +cu130 = CUDA 13.0용 빌드 |
| torch.version.cuda | '13.0' | CPU 빌드였다면 None |
| torch.cuda.is_available() | True | 지금 이 환경에서 실제로 쓸 수 있나 |
| torch.cuda.get_device_name(0) | 'NVIDIA GeForce RTX 5080 Laptop GPU' | nvidia-smi의 GPU Name과 같아요 |
| round(props.total_memory / 1024**3, 1) | 15.9 | 바이트 → GB (GiB) |
| torch.cuda.device_count() | 1 | GPU 1개 → 번호는 0만 유효 |

| 함수 / 인자 | 뜻 | 바꾸면? / 주의 |
|---|---|---|
| torch.cuda.is_available() | CUDA 런타임과 보이는 장치를 지금 PyTorch가 쓸 수 있나? → True/False | 인자 없음. 필수 인자도 없어요 |
| torch.cuda.get_device_name(device=None) | GPU 이름 문자열(str) | GPU가 없는데 부르면 오류 → 그래서 available 확인 뒤 호출 |
| torch.cuda.get_device_properties(device=None) | 메모리(total_memory) 등 속성 객체 | 바이트 단위라 / 1024**3 하면 GB |
| device 인자 | int · str · torch.device 가능 | None이면 현재 장치. 보통 0(첫 번째 GPU), 장치 번호가 다를 수도 있어요 |

🤔 왜 torch.version.cuda도 따로 봐요? — 이건 "이 PyTorch가 어떤 CUDA로 빌드됐나"예요. CPU 빌드면 None이라, GPU가 있는데 False가 나오는 이유를 바로 알 수 있어요. 반면 is_available()은 "지금 이 환경에서 실제로 쓸 수 있나"라서 둘은 다른 질문이에요.

### CPU·GPU 둘 다 되는 코드 패턴

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
X = X.to(device)
```

▶ 실제로 돌려 보면 이 PC 기준 — 다른 PC에선 값이 달라요 · nn.Linear(2, 1), X = torch.randn(4, 2)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| torch.device("cuda" if torch.cuda.is_available() else "cpu") | device(type='cuda') | GPU가 보여서 cuda가 골라짐 (없으면 cpu) |
| next(model.parameters()).device (to 전 → 후) | cpu → cuda:0 | 모델은 to() 한 줄로 파라미터가 전부 이동 |
| X.to(device).device | cuda:0 | X 자체는 그대로 cpu — 다시 담아야 해요 |
| model(X.to(device)).shape | (4, 1) | 모델과 입력이 같은 장치라 계산됨 |

🤔 왜 이 패턴을 써요? — "cuda"를 코드에 박아 두면 GPU 없는 PC에서 바로 오류가 나요. is_available() 결과로 장치를 고르면 같은 코드가 GPU 있는 곳에선 GPU로, 없는 곳에선 CPU로 돌아가요. False가 나와도 학습은 할 수 있어요.

### Tensor 만들기 torch.tensor()

```
import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
y = torch.tensor([0, 1], dtype=torch.long)
print(x.shape, x.dtype, x.device)   # torch.Size([2, 2]) torch.float32 cpu
print(y.shape, y.dtype)             # torch.Size([2]) torch.int64
```

▶ 실제로 돌려 보면 노트 코드 그대로

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| x.shape, x.dtype, x.device | (2, 2), torch.float32, cpu | 2×2 실수, CPU에 만들어짐 |
| y.dtype | torch.int64 | torch.long = int64 (클래스 번호) |
| torch.tensor([0, 1]).dtype (dtype 생략) | torch.int64 | 정수 list면 알아서 int64 |
| torch.tensor([0.5, 1]).dtype | torch.float32 | 실수가 섞이면 float32 |
| a = np.array([1., 2.]); t = torch.tensor(a); a[0] = 99 → t | [1., 2.] (dtype torch.float64) | 값을 복사해서 원본을 바꿔도 그대로. NumPy float64는 float64로 옴 |
| torch.tensor(src).requires_grad (src는 requires_grad=True) | False (경고 1개) | 다시 감싸면 그래프가 끊기고 UserWarning이 떠요 |

torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False)

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| data (필수) | list, tuple, NumPy 배열, 스칼라 | 값을 복사해 새 Tensor를 만듦 → 원본을 바꿔도 Tensor는 안 바뀜 |
| dtype | torch.float32, torch.long … | 입력 특성은 보통 float, 분류 클래스 번호는 long. 틀리면 손실함수에서 오류 |
| device | "cpu" / "cuda" | 처음부터 GPU에 만들 수 있음 (나중에 옮기는 복사가 줄어요) |
| requires_grad | 연산을 미분용으로 기록할지 (기본 False) | True면 Autograd가 추적 (③ Autograd). 기록하는 만큼 메모리를 더 씀 |
| pin_memory | CPU 고정 메모리에 만들지 | GPU로 보내는 복사가 빨라질 수 있음 |

🤔 왜 x는 float, y는 long? — 입력 특성은 곱하고 더하고 미분해야 하니 실수(float)여야 해요. 분류 정답은 "몇 번째 클래스인가"라는 번호(index)라서 정수(long)예요. CrossEntropyLoss가 class index target을 long으로 받기 때문에 맞춰 줘요.

⚠️ 이미 있는 Tensor를 torch.tensor(t)로 다시 감싸면 복사 + 그래프 분리(detach)가 일어나요. 의도에 따라 clone() / detach() / torch.as_tensor()를 구분해서 써요.

### 난수 · 0 · 1로 채우기

```
a = torch.randn(2, 3)                       # 평균 0, 분산 1 정규분포 난수
b = torch.zeros(2, 3)                       # 0으로 채움
c = torch.ones(2, 3, dtype=torch.float32)   # 1로 채움
torch.manual_seed(42)                       # 난수 재현용 seed 고정
print(a.shape, b.sum().item(), c.mean().item())   # torch.Size([2, 3]) 0.0 1.0
```

▶ 실제로 돌려 보면 torch.manual_seed(42) 직후

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| torch.randn(2, 3) | [[ 0.3367, 0.1288, 0.2345], [ 0.2303, -1.1229, -0.1863]] | seed 42에서 이 PC가 실제로 낸 값 |
| seed 다시 42 → torch.randn(2, 3) 와 같나? | True | 같은 seed면 같은 난수 |
| torch.zeros(2, 3).sum().item() | 0.0 |  |
| torch.ones(2, 3).mean().item() | 1.0 |  |
| x = torch.randn(100000) → x.mean(), x.std() | 0.0069, 1.0027 | 많이 뽑으면 평균 ≈ 0, 표준편차 ≈ 1 |

torch.randn(*size, *, generator=None, dtype=None, device=None, requires_grad=False)

| 함수 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| size (필수) | 만들 shape. 2, 3 또는 (2, 3) | shape만 바뀌고 값은 여전히 표준정규 난수 |
| generator | 재현용 난수 생성기 | 같은 generator·seed면 같은 난수가 나옴 |
| dtype · device · requires_grad | torch.tensor와 같은 뜻 |  |
| torch.zeros / torch.ones | 0 / 1로 채운 Tensor | 모델 입력·초기 상태·테스트 데이터 만들 때 |
| .item() | 원소 1개짜리 Tensor → 파이썬 숫자 | 출력·기록할 때 씀 |

🤔 왜 seed를 고정해요? — 난수는 실행할 때마다 달라서, 결과가 바뀐 게 코드를 고쳐서인지 운 때문인지 구분이 안 돼요. seed를 고정하면 같은 환경에서 같은 난수가 나와 비교와 재현이 쉬워져요. 단, 무작위 데이터는 실제 업무 데이터와 다르다는 점은 기억해요.

### 장치 옮기기 .to(device)

```
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.randn(4, 2).to(device)
model = nn.Linear(2, 1).to(device)
output = model(X)
print(X.device, output.device)       # 둘이 같아야 함
```

▶ 실제로 돌려 보면 이 PC 기준 — 다른 PC에선 값이 달라요 · X = torch.randn(4, 2)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| X.to(device) 후 X.device | cpu | 원본은 안 바뀜 — 새 Tensor를 돌려줌 |
| X = X.to(device) → X.device | cuda:0 | 다시 담아야 바뀜 |
| output = model(X) → output.device | cuda:0 | 입력·모델과 같은 장치 |
| X.to("cpu") is X | True | 이미 cpu면 복사 없이 자기 자신 (copy=False) |
| X.to(torch.float64).dtype | torch.float64 | dtype만 바꾸기도 가능 |
| cuda 모델에 cpu Tensor 넣기 model(X_cpu) | RuntimeError: Expected all tensors to be on the same device, but got mat1 is on cpu, different from othe… | 장치를 섞으면 바로 오류 |

tensor.to(device=None, dtype=None, non_blocking=False, copy=False) — 명시적 필수 인자는 없지만, 보통 device나 dtype 중 하나를 적어요.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| device | cpu / cuda | 그 장치로 복사한 Tensor를 반환 (원본은 그대로) |
| dtype | 자료형 변경 (예: torch.float32) | 장치는 두고 자료형만 바꿀 수 있음 |
| non_blocking | 가능하면 비동기 복사 요청 | True면 복사하는 동안 다른 일을 할 수 있음 (pin_memory와 함께) |
| copy | 조건이 같아도 복사할지 | False(기본)면 이미 같은 장치·dtype일 때 원본을 그대로 돌려줄 수 있음 |

⚠️ X.to(device)만 쓰고 끝내면 안 돼요. 새 Tensor를 돌려주는 거라서 X = X.to(device)처럼 다시 담아야 해요. 그리고 모델 · 입력 · 정답이 모두 같은 device에 있어야 계산돼요. CPU Tensor와 CUDA 모델을 섞으면 오류가 나요.

🤔 왜 같은 device여야 해? — CPU 메모리(RAM)와 GPU 메모리(VRAM)는 서로 다른 공간이에요. GPU는 VRAM에 있는 값만 바로 계산할 수 있어서, 가중치(모델)는 VRAM에 있고 입력은 RAM에 있으면 곱셈 자체를 할 수 없어요. .to()는 한쪽에서 다른 쪽으로 복사하는 거라 시간도 들어요. 그래서 소량을 자주 옮기기보다 batch 단위로 한 번에 옮겨요.

### 자동 미분 backward()

```
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(y.item(), x.grad.item())    # 16.0 8.0
```

▶ 실제로 돌려 보면 x = 3, y = x² + 2x + 1

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| y.item() | 16.0 | 9 + 6 + 1 |
| y.backward() 의 반환값 | None | 값은 안 돌려주고 .grad를 채움 |
| x.grad.item() | 8.0 | dy/dx = 2x + 2 = 8 |
| 다시 순전파 + backward → x.grad.item() | 16.0 | zero_grad 없이 또 하면 8 + 8 = 16 (누적!) |
| y.requires_grad, y.grad_fn | True, AddBackward0 | y는 x로부터 계산된 값이라 그래프 노드 |

함수 형태: torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, inputs=None) — 보통은 loss.backward() 메서드로 불러요.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| (없음) | 스칼라 loss면 loss.backward()만 부르면 됨. 반환값은 None, 결과는 leaf Tensor·Parameter의 .grad에 | 값이 아니라 .grad가 바뀌어요 |
| gradient / grad_tensors | 출력이 스칼라가 아닐 때 시작 기울기 | 비스칼라 출력에서 안 주면 오류 |
| retain_graph | 그래프를 지우지 않고 남겨 둘지 | True면 같은 그래프로 한 번 더 backward 가능 (메모리↑) |
| create_graph | 고차 미분용 그래프를 만들지 | 기울기의 기울기가 필요할 때만 |
| inputs | 특정 입력에 대해서만 기울기 계산 | 나머지 .grad는 건드리지 않음 |

⚠️ 기울기는 누적돼요. 새로 순전파한 뒤 backward()를 또 부르면 이전 기울기에 더해져요(2 → 4). 그래서 일반 학습 루프에선 매 step마다 기울기를 비워요(zero_grad).

backward를 여러 번 눌러 기울기가 쌓이는 모습은 Day9 ③ Autograd의 누적 체험에서 직접 해 볼 수 있어요.

🤔 왜 PyTorch는 기울기를 덮어쓰지 않고 더해? — 일부러 여러 번 backward해서 기울기를 모은 뒤 한 번에 갱신하고 싶은 경우(예: GPU 메모리가 부족해서 작은 batch 여러 개를 합칠 때)가 있어서예요. 하지만 일반 학습에선 지금 batch의 기울기만 써야 하니, 안 비우면 지난 batch 기울기까지 섞여 갱신이 엉뚱하게 커져요.

### 손으로 5번 갱신하기

```
w = torch.tensor(0.0, requires_grad=True)
target = torch.tensor(4.0)
lr = 0.1
for step in range(5):
    loss = (w - target) ** 2
    loss.backward()               # 1. 기울기 계산
    with torch.no_grad():
        w -= lr * w.grad          # 2. 기록 없이 값 갱신
    w.grad.zero_()                # 3. 기울기 비우기
    print(step + 1, round(w.item(), 4))
```

▶ 실제로 돌려 보면 loss = (w − 4)², w = 0에서 시작

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| lr = 0.1, 5 step의 w | 0.8 → 1.44 → 1.952 → 2.3616 → 2.6893 | 남은 거리가 매번 ×0.8 |
| lr = 0.01 | 0.08 → 0.1584 → 0.2352 → 0.3105 → 0.3843 | 너무 작으면 거의 안 움직임 |
| lr = 1.1 | 8.8 → -1.76 → 10.912 → -4.2944 → 13.9533 | 1보다 크면 4를 넘어가며 점점 멀어짐(발산) |
| lr = 0.1인데 w.grad.zero_() 빠뜨림 | 0.8 → 2.24 → 4.032 → 5.8176 → 7.2397 | 기울기가 쌓여 4를 지나쳐 버림 |

| 줄 | 왜? |
|---|---|
| with torch.no_grad(): | 갱신하는 연산 자체가 다음 계산 그래프에 기록되지 않도록 |
| w.grad.zero_() | 안 비우면 다음 step 기울기에 이번 것이 더해짐 |
| lr | 작게 → 느림, 크게 → 진동·발산 가능 |

### torch.no_grad() vs tensor.detach()

```
model.eval()
with torch.no_grad():              # 블록 안 연산은 그래프에 기록 안 함
    logits = model(X)
probs = torch.softmax(logits, dim=1).detach().cpu()   # 그래프에서 떼고 CPU로
```

▶ 실제로 돌려 보면 nn.Linear(2, 2), X = torch.randn(4, 2)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model(X).requires_grad (그냥 호출) | True | 파라미터가 학습 대상이라 기록됨 |
| with torch.no_grad(): model(X).requires_grad | False | 블록 안에선 기록 안 함 |
| softmax(logits).detach().requires_grad | False | 그래프에서 떼어 낸 값 |
| probs.sum(dim=1) | [1., 1., 1., 1.] | softmax라 각 행 합이 1 |
| logits.numpy() (detach 없이) | RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead. | 기록 중인 Tensor는 NumPy로 못 바꿈 → detach 필요 |

| 명령 | 하는 일 | 언제? |
|---|---|---|
| with torch.no_grad(): | 블록 안 연산을 gradient 그래프에 기록하지 않음 | 평가 · 예측 · 수동 파라미터 갱신 |
| tensor.detach() | 현재 그래프에서 분리한 Tensor(view)를 반환 | 값만 꺼내 쓰고 싶을 때 (기록·출력·NumPy 변환) |

🤔 왜 갱신은 no_grad 안에서 해? — w -= lr * w.grad도 연산이라, 기록을 켜 두면 "갱신하는 과정" 자체가 다음 계산 그래프에 섞여 들어가요. 파라미터 값 자체를 Autograd가 다시 추적하지 않도록 갱신 구간을 분리하는 거예요. 반대로 학습 중 필요한 기울기까지 실수로 끊지 않도록 no_grad/detach는 위치를 잘 골라야 해요.

### 2→8→2 ANN 만들기

```
import torch
from torch import nn
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
X = torch.randn(4, 2)
logits = model(X)                 # 순전파: 각 층의 forward를 순서대로 호출
print(logits.shape)               # torch.Size([4, 2]) ← 확률이 아니라 logits
print(sum(p.numel() for p in model.parameters()))   # 42
```

▶ 실제로 돌려 보면 torch.manual_seed(0), X = torch.randn(4, 2)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| logits.shape | (4, 2) | (batch 4, class 2) |
| logits[0] | [ 0.0862, -0.0312] | 확률이 아니라 원시 점수 — 음수도 나오고 합이 1도 아님 |
| model[0].weight.shape, model[0].bias.shape | (8, 2), (8,) | weight는 (out, in) |
| sum(p.numel() for p in model.parameters()) | 42 | 2×8+8 + 8×2+2 |
| (model[1](model[0](X)) >= 0).all() | True | ReLU 뒤엔 음수가 없음 |
| model(torch.randn(4, 3)) | RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x3 and 2x8) | 입력 마지막 차원이 in_features(2)와 다르면 오류 |

nn.Linear(in_features, out_features, bias=True, device=None, dtype=None) · nn.ReLU(inplace=False) · nn.Sequential(*modules)

| 명령 / 인자 | 뜻 | 바꾸면? / 주의 |
|---|---|---|
| in_features (필수) | 입력 마지막 차원 크기 | 입력의 마지막 차원과 다르면 shape 오류 |
| out_features (필수) | 출력 feature 수 (이 층의 뉴런 수) | 키우면 표현력·파라미터 수↑. weight shape는 (out, in) → Linear(2,8)은 (8, 2) |
| bias=True | 편향 b 사용 여부 (기본 True) | False면 bias 파라미터가 없어짐. bias shape는 (out_features,) |
| device · dtype | 파라미터를 만들 장치·자료형 | 보통 생략하고 나중에 model.to(device) |
| nn.ReLU(inplace=False) | 음수를 0으로 (필수 인자 없음) | inplace=True면 입력을 직접 덮어써 메모리를 아끼지만 원본이 사라짐. 출력 shape는 입력과 같음 |
| nn.Sequential(*modules) | 층을 순서대로 연결해 모델 하나로 | 넣은 순서가 곧 모델 구조. 마지막 모듈 출력을 돌려줌. OrderedDict로 이름도 붙일 수 있음 |
| model.parameters() | 학습할 Parameter 반복자 (Optimizer에 넘김) | named_parameters()는 이름까지 |
| p.numel() | 파라미터 안 숫자 개수 | 2×8+8 + 8×2+2 = 42 |

🤔 왜 마지막에 softmax를 안 붙여? — 출력은 확률이 아니라 logits(원시 점수)예요. ⑤ 손실함수에서 쓸 CrossEntropyLoss가 안에서 softmax까지 함께 계산하기 때문에, 모델에서 또 softmax를 하면 두 번 하게 돼요.

### 직접 모델 클래스 만들기 nn.Module

```
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()                 # 꼭 호출해야 층이 등록됨
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
    def forward(self, x):                  # 입력 (batch, 2) → 출력 (batch, 2)
        return self.net(x)

model = SimpleANN()
logits = model(X)                          # model(X) 하면 forward가 실행됨
```

▶ 실제로 돌려 보면 SimpleANN() 과 일부러 틀린 두 클래스

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model(X).shape (X: (4, 2)) | (4, 2) | model(X)가 forward를 불러 줌 |
| [n for n, _ in model.named_parameters()] | ['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias'] | self.net 아래 층이 자동 등록 |
| 층을 파이썬 list에 넣은 모델의 파라미터 수 | 0 | 0개! 등록이 안 돼 Optimizer가 못 봐요 |
| super().__init__() 빼먹은 모델 만들기 | AttributeError: cannot assign module before Module.__init__() call | 층을 등록할 장부가 없어 오류 |

| 구성 | 뜻 | 빠뜨리면? |
|---|---|---|
| nn.Module 상속 | 모든 신경망의 기본 클래스. 하위 Module·Parameter 등록, device 이동, train/eval 모드 관리 | parameters()·to()·eval()을 못 씀 |
| super().__init__() | 부모 초기화 | 층을 등록할 장부가 안 만들어져 오류 |
| forward(x) | 순전파 계산을 정의 | model(X) 호출 시 실행할 내용이 없음 |
| model.to(device) | 안의 층·파라미터가 한꺼번에 이동 | 하나씩 옮길 필요 없음 |

🤔 왜 nn.Module을 상속해? — self.net = …처럼 속성에 층을 넣으면 Module이 그걸 자동으로 "내 하위 층"으로 등록해요. 덕분에 model.parameters() 한 줄로 모든 가중치를 Optimizer에 넘기고, model.to(device) 한 줄로 전부 옮기고, model.eval() 한 줄로 전부 평가 모드로 바꿀 수 있어요.

⚠️ 등록된 파라미터만 Optimizer가 갱신해요. 층을 파이썬 list에 그냥 넣으면 등록이 안 되니 nn.ModuleList 같은 등록 구조를 써요.

### 분류 손실 nn.CrossEntropyLoss

```
loss_fn = nn.CrossEntropyLoss()
logits = model(X)                              # (batch, class), softmax 전!
y = torch.tensor([0, 1, 1, 0], dtype=torch.long)   # (batch,), long
loss = loss_fn(logits, y)                      # 스칼라, 작을수록 정답에 가까움
```

▶ 실제로 돌려 보면 학습 전 2→8→2 모델(seed 0), X = randn(4, 2), y = [0, 1, 1, 0]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| loss_fn(logits, y) | 0.6972 | 학습 전이라 ln 2 ≈ 0.693 근처 (반반 찍는 수준) |
| nn.CrossEntropyLoss(reduction='none')(logits, y) | [0.6362, 0.7467, 0.6825, 0.7234] | 샘플별 loss 4개, 평균이 위 값 |
| loss_fn(torch.tensor([[10., 0.]]), torch.tensor([0])) | 4.542e-05 | 정답 점수가 크게 앞서면 거의 0 |
| loss_fn(torch.tensor([[10., 0.]]), torch.tensor([1])) | 10 | 확신하고 틀리면 아주 큼 |
| loss_fn(torch.softmax(torch.tensor([[10., 0.]]), 1), torch.tensor([0])) | 0.3133 | softmax를 먼저 하면 확실히 맞혀도 0.31에서 더 못 내려감 (원래는 0.00005) |
| loss_fn(logits, y.float()) | RuntimeError: expected target dtype to be Long or Byte, but got Float | (batch,) 정답을 float로 주면 오류 — long이어야 해요 |

nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction="mean", label_smoothing=0.0) — 생성자 필수 인자는 없고, 호출할 때 (logits, target)을 넣어요.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| weight | 클래스별 가중치 | 적은 클래스를 틀릴 때 loss가 더 커져 → 그 클래스를 더 신경 써서 학습 |
| ignore_index | 무시할 클래스 번호 (기본 -100) | 그 번호가 정답인 샘플은 loss·기울기에서 빠짐 |
| reduction | "mean"(기본) / "sum" / "none" | mean이면 스칼라(batch 평균), sum이면 합이라 batch가 클수록 커짐, none이면 샘플별 loss |
| label_smoothing | 정답을 살짝 부드럽게 (기본 0.0) | 키우면 정답에 100% 확신하도록 몰아붙이지 않음 |

⚠️ CrossEntropyLoss에는 softmax 하기 전 logits를 그대로 넣어요(안에서 softmax를 함께 계산해요). 정답은 클래스 번호, shape (batch,), dtype long. 그래서 모델 마지막에 softmax를 안 붙여요.

🤔 왜 softmax를 먼저 하면 안 돼? — 손실 함수가 안에서 softmax를 또 하니까, 미리 하면 softmax를 두 번 하게 돼요. 확률이 0~1 사이로 한 번 눌린 값에 다시 softmax를 하면 클래스 간 차이가 작아져서, 정답을 잘 맞혀도 loss가 잘 안 줄고 기울기도 약해져요. logits를 그대로 넣으면 안에서 한 번에 안정적으로 계산해요.

### TensorDataset · DataLoader

```
from torch.utils.data import TensorDataset, DataLoader
X = torch.randn(200, 2)
y = ((X[:, 0] + 0.8 * X[:, 1]) > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)          # torch.Size([32, 2]) torch.Size([32])
```

▶ 실제로 돌려 보면 torch.manual_seed(42), X = randn(200, 2)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(TensorDataset(X, y)) | 200 | 샘플 수 |
| ds[0] | ([1.9269, 1.4873], 1) | 같은 번호의 X와 y가 짝으로 나옴 |
| y.float().mean() | 0.52 | 클래스 1 비율 — 대략 반반 |
| xb.shape, yb.shape, yb.dtype | (32, 2), (32,), torch.int64 | batch 하나 |
| len(loader), 각 batch 크기 | 7, [32, 32, 32, 32, 32, 32, 8] | 200 = 32×6 + 8 → 마지막은 8개 |
| len(DataLoader(ds, batch_size=32, drop_last=True)) | 6 | 작은 마지막 batch를 버림 |

DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, drop_last=False, ...)

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| dataset (필수) | 꺼낼 데이터 묶음 (TensorDataset(X, y)는 X와 y를 같은 번호로 묶고, len = 샘플 수) |  |
| batch_size | 한 번에 꺼낼 샘플 수 (기본 1) | 키우면 step 수↓·메모리↑·기울기 안정, 줄이면 그 반대 (Day10 ③ Batch) |
| shuffle | epoch마다 순서 섞기 (기본 False) | train은 True — 매번 같은 순서로 보면 그 순서 패턴까지 배울 수 있어서 섞어 줘요 |
| num_workers | 데이터 읽는 보조 프로세스 수 (기본 0) | 늘리면 로딩이 빨라질 수 있지만, Windows에선 실행 진입점(if __name__ == "__main__":) 구조에 주의 → 처음엔 0 |
| pin_memory | GPU 전송용 고정 메모리 | True면 CPU→GPU 복사가 빨라질 수 있음 |
| drop_last | 마지막 작은 batch 버리기 (기본 False) | True면 모든 batch 크기가 같아지지만 그 epoch엔 일부 샘플을 안 씀 |

### 준비 + batch 반복

```
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X = torch.randn(200, 2)
y = ((X[:, 0] + 0.8 * X[:, 1]) > 0).long()
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(80):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)   # batch도 같은 device로
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():                           # 평가: 기록 끄기
    pred = model(X.to(device)).argmax(dim=1)
    acc = (pred.cpu() == y).float().mean().item()
```

▶ 실제로 돌려 보면 노트 코드 그대로 (seed 42, 직선 경계 데이터 200개, device=cuda)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 학습 전 정확도 | 0.515 | 무작위 가중치라 찍기 수준 |
| 첫 batch loss → 마지막 batch loss | 0.7407 → 0.0518 | batch마다 step하며 줄어듦 |
| optimizer.step() 호출 수 | 560 | 7 batch × 80 epoch |
| model[0].weight 가 바뀐 양 (평균 절댓값) | 0.5693 | step이 실제로 가중치를 움직임 |
| acc (train 데이터 전체) | 1.000 | 직선으로 갈리는 쉬운 데이터라 200개 모두 맞힘 — 학습한 데이터로 잰 값이라 일반화 성능은 아님 |
| pred.device | cuda:0 | 비교 전에 .cpu()로 정답과 장치를 맞춤 |

torch.optim.SGD(params, lr=0.001, momentum=0, weight_decay=0, ...) · optimizer.zero_grad(set_to_none=True) · optimizer.step(closure=None)

| 명령 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| params (필수) | 갱신할 Parameter, 보통 model.parameters() | 여기 안 넘긴 파라미터는 학습해도 안 바뀜 |
| lr | 학습률 (기본 0.001, 그래도 명시 권장) | 너무 크면 발산, 너무 작으면 느림 (③ 경사하강 lr 체험 참고). 강의 예시는 0.1 |
| momentum, weight_decay, nesterov | 이전 방향 누적, 큰 가중치 억제 (기본 0) | Day10 ① Optimizer에서 본격 비교 |
| optimizer.zero_grad(set_to_none=True) | 지난 기울기 비우기 | 기본은 .grad를 None으로, False면 0으로 채움 |
| optimizer.step(closure=None) | 현재 .grad로 한 번 갱신 (반환 None) | closure는 대부분 생략 |

### epoch 평균 loss를 제대로 내기

```
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)       # batch 평균 × batch 크기
    return total_loss / len(loader.dataset)        # 샘플 수로 가중한 평균 loss
```

▶ 실제로 돌려 보면 같은 데이터 200개, batch 32 (마지막 batch 8개)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| epoch 1 batch 크기 | [32, 32, 32, 32, 32, 32, 8] |  |
| epoch 1 마지막 batch loss | 0.4449 | 8개짜리 batch의 평균 |
| train_one_epoch(...) (샘플 가중 평균) | 0.5334 | 샘플 하나하나의 평균 |
| batch loss들을 그냥 평균 | 0.5239 | 8개 batch가 32개와 같은 무게라 살짝 다름 |
| epoch 2 train_one_epoch(...) | 0.4788 | 다음 epoch엔 더 내려감 |

🤔 왜 × len(xb)를 해? — loss는 batch 안의 평균이에요(reduction="mean"). 마지막 batch는 8개처럼 작을 수 있어서, batch 평균들을 그냥 평균 내면 8개짜리 batch가 32개짜리와 같은 무게를 갖게 돼요. 샘플 수를 곱해 합친 뒤 전체 샘플 수로 나누면 "샘플 하나하나의 평균"이 정확히 나와요. 함수 하나가 "epoch 1회 학습"이라는 한 가지 역할만 하게 나누면 읽기도 쉬워요.

### model.train() · model.eval() · torch.no_grad()

| 명령 | 하는 일 | 안 하는 일 |
|---|---|---|
| model.train() | Dropout·BatchNorm 같은 층을 학습 동작으로 | 기울기를 켜고 끄진 않음 |
| model.eval() | 평가 동작으로 전환 | 기울기 계산을 끄지 않음! |
| with torch.no_grad(): | 그래프 기록을 끔 → 메모리 절약 | 모드는 안 바꿈 |

▶ 실제로 돌려 보면 Dropout(0.5)가 든 작은 모델 · 같은 입력 두 번

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model.train(); model(x) 두 번 | 0.4557, 0.5273 | 학습 모드: Dropout이 매번 다른 칸을 꺼서 값이 달라요 |
| model.eval(); model(x) 두 번 | 0.4728, 0.4728 | 평가 모드: 매번 같아요 |
| model.eval() 후 model(x).requires_grad | True | eval()만으로는 기울기 기록이 안 꺼져요! |
| with torch.no_grad(): model(x).requires_grad | False | 기울기 기록은 no_grad가 끔 |

💡 평가할 땐 model.eval() + torch.no_grad()를 세트로 쓰는 습관을 들여요. GPU 결과를 NumPy로 쓰려면 probs.detach().cpu()처럼 CPU로 옮겨요.

🤔 왜 둘 다 필요해? — 둘은 다른 스위치예요. eval()은 Dropout·BatchNorm 같은 층이 학습용 동작(무작위로 끄기, batch 통계 쓰기)을 멈추게 해서 예측이 매번 같고 공정하게 만들어요. no_grad()는 평가 땐 backward를 안 하니 그래프를 안 만들어 메모리·시간을 아껴요. no_grad를 빠뜨려도 결과 숫자는 같을 수 있지만 불필요한 그래프가 생겨요. 오늘 모델엔 Dropout·BatchNorm이 없지만, 습관을 지금 들이면 다음 수업에서 바로 효과를 봐요.

## 📖 Day9 용어 (70개)
- **특성** (Feature (특성, 입력 열)): 모델에 넣는 입력 정보 한 가지. 표에서 열 하나가 특성 하나예요(예: 글자수, 허용 도구수). 행 = 사례, 열 = 특성.
- **Accuracy** (Accuracy (정확도)): 전체 중 맞힌 비율. (TP + TN) ÷ 전체. 한쪽 클래스가 많으면 그쪽만 찍어도 높게 나와 속기 쉬워요.
- **가상환경** (Virtual environment (가상환경, .venv)): 프로젝트마다 따로 쓰는 파이썬과 패키지 묶음. 버전이 섞이지 않게 해 같은 코드가 같은 결과를 내도록 도와요.
- **seed** (random_state (난수 고정값)): 무작위 과정(추출, 특성 후보, 분할 섞기 등)을 매번 같게 만드는 번호. 결과를 재현하려고 고정해요(예: 42).
- **학습률** (Learning rate (학습률)): 한 회차의 보정을 얼마나 반영할지 정하는 비율(예: 0.05). 작으면 조금씩 안정적으로 고치지만 회차가 더 필요해요.
- **GPU** (Graphics Processing Unit (그래픽 처리 장치)): 대량 계산을 병렬로 빠르게 하는 장치. CUDA는 NVIDIA GPU를 쓰기 위한 도구. 이번 수업은 CPU만으로 충분해요.
- **CPU** (Central Processing Unit (중앙 처리 장치)): 컴퓨터의 기본 두뇌. 소수의 강한 코어로 복잡한 분기·파일 읽기·전처리 같은 일을 잘해요. 딥러닝에서도 데이터 준비는 CPU 몫이에요.
- **CUDA** (Compute Unified Device Architecture): NVIDIA GPU에서 계산을 시킬 수 있게 해 주는 연산 플랫폼·API. PyTorch가 GPU를 쓰려면 CUDA용으로 빌드된 torch가 필요해요.
- **VRAM** (Video RAM (GPU 메모리)): GPU에 붙어 있는 전용 메모리. GPU는 VRAM에 올라온 값만 바로 계산할 수 있어서, 모델과 데이터를 여기로 옮겨야 해요.
- **RAM** (Random Access Memory (주 메모리)): CPU가 쓰는 컴퓨터의 주 메모리. VRAM과는 서로 다른 공간이라, CPU의 Tensor를 GPU에서 쓰려면 복사(.to)해야 해요.
- **드라이버** (GPU Driver (그래픽 드라이버)): 운영체제(OS)가 GPU를 쓸 수 있게 연결해 주는 프로그램. 버전이 안 맞으면 GPU가 있어도 PyTorch가 못 써요.
- **nvidia-smi** (NVIDIA System Management Interface): 터미널에서 GPU 이름·드라이버 버전·VRAM 사용량·사용률을 보여 주는 명령. 잘 나와도 PyTorch가 GPU를 쓸 수 있다는 보장은 없어요.
- **PyTorch**: Tensor 계산, 자동 미분(Autograd), 신경망 층, Optimizer까지 들어 있는 딥러닝 라이브러리. 코드에선 import torch 로 불러요.
- **병렬 처리** (Parallel processing): 서로 독립인 계산 여러 개를 동시에 처리하는 것. 행렬 곱의 각 칸은 서로 독립이라 GPU가 동시에 계산할 수 있어요.
- **처리량** (Throughput (처리량)): 정해진 시간에 처리하는 일의 양. GPU는 같은 모양의 독립 계산이 충분히 많을 때 처리량이 커져요.
- **행렬** (Matrix (행렬)): 숫자를 행과 열로 늘어놓은 2차원 표. 딥러닝의 핵심 계산은 행렬 곱셈(곱하고 더하기)이에요. 0D는 스칼라(숫자 하나), 1D는 벡터(숫자 한 줄)예요.
- **Tensor** (Tensor (텐서)): 차원을 가진 숫자 배열. 딥러닝의 모든 데이터·가중치가 Tensor예요. NumPy 배열과 비슷하지만 GPU에 올릴 수 있고 자동 미분이 돼요.
- **차원** (Dimension (차원)): Tensor의 축 개수. 0D 스칼라(숫자 하나), 1D 벡터, 2D 행렬, 3D 이상은 batch·이미지·문장 같은 데이터예요.
- **shape** (Shape (모양)): Tensor가 축마다 몇 칸인지 적은 것. 예: (batch, feature) = (32, 2)는 샘플 32개 × 특성 2개. 오류 대부분이 shape가 안 맞아서 생겨요.
- **dtype** (Data type (자료형)): Tensor 안 숫자의 종류. 입력 특성은 보통 실수 torch.float32, 분류 정답(클래스 번호)은 정수 torch.long 이에요.
- **device** (Device (장치)): Tensor·모델이 있는 곳: "cpu" 또는 "cuda"(GPU). 모델·입력·정답이 모두 같은 device에 있어야 계산돼요.
- **Broadcasting**: 모양이 다른 Tensor끼리 계산할 때 작은 쪽을 규칙에 따라 자동으로 늘려 맞추는 것. 편하지만 의도와 다른 shape도 오류 없이 계산될 수 있어요.
- **Autograd** (Automatic differentiation (자동 미분)): PyTorch가 계산 과정을 기록해 두었다가 backward() 한 번으로 모든 기울기를 자동으로 구해 주는 기능.
- **계산 그래프** (Computational graph): 어떤 값끼리 무엇을 곱하고 더했는지 연산 순서를 기록한 연결도. backward()가 이걸 거꾸로 따라가며 기울기를 구해요.
- **연쇄법칙** (Chain rule (연쇄법칙)): 이어진 연산의 미분은 각 단계 미분을 곱해서 구한다는 규칙. 역전파가 이 규칙으로 기울기를 한 칸씩 곱해 가요.
- **requires_grad** (requires_grad (기울기 추적 여부)): True면 이 Tensor로 하는 연산을 Autograd가 기록해 기울기를 구해요. 학습할 파라미터는 보통 True, 기본값은 False.
- **기울기** (Gradient (기울기)): 파라미터를 조금 늘리면 loss가 얼마나 커지는지 나타내는 값. backward() 후 .grad 에 저장되고, 지우지 않으면 누적돼요.
- **경사하강법** (Gradient Descent (경사 하강법)): 기울기의 반대 방향으로 조금씩 움직여 loss를 줄이는 방법. 파라미터 = 파라미터 − 학습률 × 기울기.
- **발산** (Divergence (발산)): 학습률이 너무 커서 갱신할 때마다 목표를 지나쳐 loss가 점점 커지는 상태. 심하면 NaN이 돼요.
- **torch.no_grad()**: 이 블록 안의 연산은 계산 그래프에 기록하지 않아요. 평가·예측·수동 갱신 때 써서 메모리와 시간을 아껴요.
- **detach()** (Tensor.detach()): Tensor를 계산 그래프에서 떼어 낸 것. 값만 꺼내 출력·기록·NumPy 변환할 때 써요.
- **뉴런** ((Artificial) Neuron (인공 뉴런)): 입력마다 가중치를 곱해 더하고 편향을 더한 뒤(z = w₁x₁ + w₂x₂ + b) 활성화 함수를 통과시키는 계산 단위.
- **가중치** (Weight (가중치)): 입력에 곱해지는 숫자. 학습이란 결국 이 가중치들을 loss가 줄어드는 쪽으로 조금씩 고치는 일이에요.
- **편향** (Bias (편향)): 가중합에 더하는 상수 b. 출력 위치를 위아래로 옮겨 주는 학습 파라미터예요.
- **파라미터** (Parameter (모델 파라미터)): 학습으로 값이 바뀌는 숫자들 — 가중치와 편향. model.parameters()로 모아 Optimizer에 넘겨요. 2→8→2 ANN은 42개예요.
- **층** (Layer (층)): 뉴런 여러 개를 묶은 계산 단위. 층을 여러 개 쌓을수록 모델이 더 복잡한 표현을 배울 수 있어요.
- **Linear 층** (nn.Linear (완전연결 층)): 입력의 마지막 차원에 가중치를 곱하고 편향을 더하는 층. nn.Linear(2, 8)은 특성 2개를 8개로 바꿔요.
- **활성화 함수** (Activation function (활성화 함수)): 층의 출력에 비선형성을 넣는 함수. 없으면 Linear를 아무리 쌓아도 Linear 하나와 같아요. 대표가 ReLU예요.
- **ReLU** (Rectified Linear Unit): 음수는 0으로, 양수는 그대로 내보내는 가장 흔한 활성화 함수. 단순하지만 층 사이에 비선형성을 넣어 줘요.
- **비선형성** (Nonlinearity (비선형성)): 직선(곱하고 더하기)만으로는 표현 못 하는 굽은 관계. 활성화 함수가 넣어 줘야 신경망이 복잡한 패턴을 배워요.
- **ANN** (Artificial Neural Network (인공신경망)): 인공 뉴런을 묶은 층을 여러 개 쌓은 모델. 오늘은 입력 2 → 은닉 8 → 출력 2 구조를 만들었어요.
- **nn.Sequential** (torch.nn.Sequential): 층을 넣은 순서대로 연결해 모델 하나로 만드는 상자. 넣은 순서가 곧 모델 구조예요.
- **nn.Module** (torch.nn.Module): PyTorch의 모든 신경망이 상속하는 기본 클래스. 안에 넣은 층·파라미터를 등록해서 parameters()·to()·eval()을 한 번에 쓸 수 있게 해요.
- **logit** (Logits (원시 점수)): 모델이 마지막 층에서 낸, softmax 하기 전의 클래스별 점수. 합이 1일 필요가 없고 확률이 아니에요.
- **softmax** (Softmax): logits를 0~1 사이, 합이 1인 확률로 바꾸는 함수. CrossEntropyLoss가 안에서 해 주니 모델 끝에는 붙이지 않아요.
- **argmax** (Argument of the maximum): 가장 큰 값이 있는 위치(번호)를 돌려주는 함수. 분류에서는 점수가 가장 큰 클래스 번호 = 예측 클래스예요.
- **손실 함수** (Loss function (손실 함수)): 모델 예측이 정답에서 얼마나 먼지를 숫자 하나(loss)로 만드는 함수. 학습은 이 숫자를 줄이는 방향으로 가중치를 고쳐요.
- **CrossEntropyLoss** (Cross-Entropy Loss (교차 엔트로피 손실)): 분류용 손실 함수. softmax 전 logits와 클래스 번호(long, shape (batch,))를 받아, 정답 클래스 확률이 낮을수록 loss를 크게 줘요.
- **목적함수** (Objective function (목적함수)): 학습이 줄이거나(또는 키우려고) 하는 기준 함수. 딥러닝에선 미분할 수 있는 loss를 목적함수로 써요.
- **순전파** (Forward pass (순전파)): 입력을 층마다 차례로 통과시켜 예측(logits)을 만드는 계산. model(X)를 부르면 forward가 실행돼요.
- **역전파** (Backpropagation (역전파)): loss에서 거꾸로 거슬러 올라가며 모든 파라미터의 기울기를 구하는 계산. PyTorch에선 loss.backward() 한 줄이에요.
- **zero_grad** (optimizer.zero_grad()): 지난 기울기를 비우는 명령. 기울기는 누적되기 때문에 매 step 처음에 꼭 불러야 이번 batch 기울기만 남아요.
- **Optimizer** (Optimizer (최적화 도구)): 구한 기울기를 "어떤 규칙으로" 파라미터 갱신에 반영할지 정하는 객체. optimizer.step()이 실제로 값을 바꿔요. SGD·AdamW 등이 있어요.
- **iteration** (Iteration / step (반복 1회)): batch 하나로 forward → loss → backward → optimizer.step()까지 한 번 갱신한 것. epoch당 iteration 수 = ⌈샘플 수 ÷ batch_size⌉.
- **batch** (Batch (배치)): 한 번에 모델에 넣어 계산하는 샘플 묶음. 이 묶음마다 기울기를 구해 가중치를 한 번 고쳐요.
- **mini-batch** (Mini-batch (미니배치)): 전체 데이터를 작게 나눈 batch. 전체를 한 번에 넣는 것과 하나씩 넣는 것 사이에서 속도·메모리·기울기 흔들림의 균형을 맞춰요.
- **batch size** (Batch size (배치 크기)): batch 하나에 들어가는 샘플 수. 키우면 기울기가 안정되고 step 수가 줄지만 메모리를 더 써요.
- **epoch** (Epoch (에폭)): train 데이터 전체를 한 바퀴 다 본 단위. 200개를 batch 32로 나누면 7번 갱신하면 1 epoch이에요.
- **Dataset** (Dataset / TensorDataset): 샘플을 번호로 꺼낼 수 있게 묶은 자료 구조. TensorDataset(X, y)는 X와 y를 같은 번호끼리 짝지어 묶어요.
- **DataLoader** (torch.utils.data.DataLoader): Dataset에서 batch_size만큼씩 꺼내 주는 도구. shuffle로 epoch마다 순서를 섞을 수 있어요.
- **shuffle** (Shuffle (섞기)): epoch마다 데이터 순서를 무작위로 섞는 옵션. train은 True(순서 패턴을 외우지 않게), valid/test는 False.
- **학습 루프** (Training loop (학습 루프)): batch마다 zero_grad → forward → loss → backward → step 다섯 단계를 반복하는 코드 구조.
- **model.train()** (Training mode (학습 모드)): Dropout·BatchNorm 같은 층을 학습용 동작으로 바꾸는 스위치. 기울기 계산을 켜고 끄는 것과는 상관없어요.
- **model.eval()** (Evaluation mode (평가 모드)): Dropout을 끄고 BatchNorm이 저장된 통계를 쓰게 바꾸는 스위치. 기울기 기록은 안 끄니 평가 땐 torch.no_grad()와 함께 써요.
- **Dropout** (Dropout (드롭아웃)): 학습할 때 뉴런 출력 일부를 확률 p로 무작위로 0으로 끄는 층. 특정 뉴런에만 기대지 못하게 해 과적합을 줄여요. 평가 땐 꺼져요(그대로 통과).
- **BatchNorm** (Batch Normalization (배치 정규화)): 층의 중간값을 현재 batch의 평균·분산으로 정리(정규화)해 학습을 안정시키는 층. 평가 땐 학습 중 모아 둔 running 통계를 써요.
- **CNN** (Convolutional Neural Network (합성곱 신경망)): 이미지처럼 가로·세로 구조가 있는 데이터에 쓰는 신경망. 입력 shape는 보통 (batch, channel, height, width)예요.
- **RNN** (Recurrent Neural Network (순환 신경망)): 문장·시계열처럼 순서가 있는 데이터를 앞에서부터 차례로 읽는 신경망.
- **Transformer**: 문장 속 단어들이 서로 얼마나 관련 있는지(attention)를 계산해 처리하는 신경망 구조. 요즘 언어 모델의 기본 구조예요.
- **임베딩** (Embedding (임베딩)): 단어 같은 기호를 의미를 담은 숫자 벡터로 바꾼 것. 문장은 (batch, sequence, feature) shape가 돼요.

---

# Day10 · Optimizer·Regularization (학습 안정화)
_2026-10-30(금) · Optimizer · 학습률 · Batch·Epoch · 과적합 · Dropout · BatchNorm · Early Stopping_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day10.html

📌 딱 3줄 요약

- Optimizer는 기울기를 "어떤 규칙으로" 반영할지, 학습률은 "얼마나 크게" 움직일지를 정해요.

- train loss는 줄고 valid loss가 다시 오르면 과적합. 학습 곡선(epoch별 기록)으로 진단해요.

- Dropout · BatchNorm · Weight Decay · Early Stopping은 목적이 다른 도구예요. 한 번에 하나씩 바꿔서 같은 조건으로 비교해요.

🔧 Day10 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| torch | 모델·Optimizer(SGD·AdamW)·Scheduler·Dropout·BatchNorm·DataLoader 전부 (Day9에서 설치) | import torch, from torch import nn, from torch.utils.data import DataLoader |
| (파이썬 기본) | best 모델 상태를 통째로 복사 | import copy |
| pandas (선택) | epoch별 기록(history)을 CSV 표로 저장·비교 | import pandas as pd |
| matplotlib (선택) | train·valid 학습 곡선 그래프 그리기 | import matplotlib.pyplot as plt |

### 🔧 설치 · 확인

```
python -m pip install pandas matplotlib            # 기록 CSV · 학습 곡선 그림용
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # Day9 환경 그대로인지
```

⚠️ torch는 Day9에 맞춰 둔 CPU/CUDA 빌드를 그대로 유지해요. 새로 깔아야 한다면 pytorch.org 선택기에서 내 환경에 맞는 명령을 써요. 그냥 pip install torch로 덮어쓰면 GPU 빌드가 CPU 빌드로 바뀔 수 있어요.

🤔 왜 새 패키지가 거의 없어? — 오늘 배우는 Optimizer · Scheduler · Dropout · BatchNorm은 모두 torch 안에 들어 있어요. 오늘은 새 도구보다 같은 학습 루프에서 조건을 하나씩 바꿔 비교하는 날이라, 기록을 표와 그래프로 남길 도구만 더하면 돼요.

## 이거 어디에 써?

Day9에서 학습 루프를 "돌리는" 법을 배웠다면, 오늘은 잘 학습되는 조건을 찾는 법이에요. 현업에선 "학습이 끝났다"보다 "무엇을 근거로 이 모델을 골랐나"를 설명해야 해요.

🐢

학습이 너무 느릴 때학습률·Optimizer 점검

〰️

loss가 출렁일 때lr을 낮추거나 batch 키우기

📉

새 데이터에 약할 때Dropout·Weight Decay·Early Stopping

💾

GPU 메모리 부족batch 크기 줄이기

| 데이터 | 쓰는 곳 | 한마디로 |
|---|---|---|
| train | 기울기 계산, 가중치 갱신 | 배우는 자료 |
| valid | epoch · 구조 · 규제 · 설정 고르기 | 고르는 자료 |
| test | 모든 선택이 끝난 뒤 최종 평가 한 번 | 보관하는 자료 |

[그림: ] 무엇을 보나: 강의 예시의 400건 분할(train 240 · valid 80 · test 80)과 각자의 일. 아래 고리처럼 train으로 배우고 → valid로 채점을 epoch마다 반복하며 설정을 고르고, test는 맨 끝에 한 번만 열어요. 🤔 왜? 채점에 쓴 자료는 결국 "고르는 데" 쓰인 자료예요. valid를 보고 수십 번 고르면 valid에 살짝 맞춰지니, 한 번도 안 본 test가 따로 있어야 진짜 실력을 잴 수 있어요.

⚠️ valid loss를 보고 epoch를 고르는 건 OK, test loss를 보고 고르면 안 돼요. 그러면 test 점수가 실제보다 좋게 나와요.

🤔 왜 train 성적만 보면 안 돼? — train은 모델이 직접 보고 가중치를 맞춘 자료라, 거기서 잘하는 건 "외웠다"일 수도 있어요. 우리가 원하는 건 새 요청에서도 잘하는 모델이라, 학습에 안 쓴 valid로 확인해야 진짜 실력을 알 수 있어요. test는 마지막 시험 한 번을 위해 끝까지 보관해요. 장치를 GPU로 바꿔도 이 원칙은 그대로예요.

## ① Optimizer — 기울기를 갱신으로 바꾸는 규칙

기울기 계산 backward→ 갱신 규칙 적용 Optimizer→ 파라미터 변경 step→ 다음 forward

같은 기울기라도 Optimizer · 학습률 · momentum · weight decay 설정에 따라 이동 경로가 달라져요.

| 이름 | 아이디어 |
|---|---|
| SGD | 지금 기울기 반대 방향으로 lr만큼 이동. 단순하고 해석 쉬움 |
| Momentum | 이전 이동 방향을 함께 써요. 공이 굴러가듯 일관된 방향은 가속, 좌우 흔들림은 줄임 |
| Adam | 기울기의 평균(1차)과 제곱 평균(2차)을 기억해 파라미터마다 갱신 크기를 따로 조절 |
| AdamW | Adam에서 weight decay를 적응적 갱신과 분리해 적용한 버전 |

🤔 왜 Optimizer마다 경로가 달라? — Gradient Descent는 "loss가 커지는 방향의 반대로 간다"는 기본 원칙이고, Optimizer는 그 기울기를 어떻게 누적 · 보정 · 적용할지 정하는 객체예요. Momentum은 지난 이동량을 더해서, 좁은 골짜기에서 좌우로 튀는 성분은 서로 상쇄되고(흔들림↓) 계속 같은 쪽을 가리키는 성분은 쌓여요(가속). Adam은 기울기의 평균(1차)과 제곱 평균(2차)을 기억해서, 기울기가 늘 큰 파라미터는 보폭을 줄이고 작은 파라미터는 키워요. 그래서 파라미터마다 보폭이 자동으로 맞춰져 초반이 빠른 경우가 많아요.

[그림: 길쭉한 골짜기 loss 위에서 SGD, momentum, Adam의 40번 갱신 경로와 loss 감소] 무엇을 보나: 가로로 길고 세로로 가파른 골짜기 loss(등고선)에서 같은 출발점(■)으로 40번 갱신한 실제 torch Optimizer 경로예요. 회색 SGD는 위아래로 튀다가 ★까지 절반도 못 갔고, 보라 momentum(같은 lr)은 출렁이면서도 ★에 도착, 초록 Adam은 대각선으로 곧장 내려가요. 🤔 왜? SGD는 가파른 세로 방향 기울기가 커서 lr을 더 키우면 발산하고, 완만한 가로 방향은 기울기가 작아 한 걸음이 짧아요. momentum은 매번 같은 쪽(가로)을 가리키는 이동을 쌓아서 가속하고, 위아래로 번갈아 바뀌는 이동은 서로 상쇄해요. Adam은 방향마다 기울기 크기로 나눠서 보폭을 맞추니 두 방향이 비슷한 속도로 내려가요. (이 그림 하나로 "항상 Adam이 이긴다"는 아니에요 — 오른쪽처럼 끝에서 살짝 넘어갔다 돌아오기도 해요.)

| 관점 | SGD + Momentum | AdamW |
|---|---|---|
| 쓰는 정보 | 현재 기울기 + 이동량 | 기울기 평균 + 제곱 평균 |
| 주요 설정 | lr · momentum · weight_decay | lr · betas · eps · weight_decay |
| 장점 | 구조 단순 · 해석 쉬움 | 초기 수렴이 빠른 경우가 많음 |

⚠️ "AdamW가 항상 이긴다"는 없어요. 초반이 빨라도 일반화가 꼭 더 좋은 건 아니에요. 비교할 땐 같은 데이터 분할 · 모델 · seed · epoch로 하고, 두 Optimizer는 잘 맞는 lr 범위도 달라요. Optimizer가 잘못된 데이터 분할이나 lr을 대신 고쳐 주지도 않아요.

🔧 Optimizer만 바꿔 끼우기 명령어 사전에서 자세히 →

```
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
```

## ② 학습률과 Scheduler

| 학습률 | 모습 | 학습 곡선 |
|---|---|---|
| 너무 작음 | 작은 이동만 반복 | loss가 천천히 줄고, 정한 epoch 안에 덜 배움 |
| 적절함 | 안정적으로 접근 | loss가 빠르게 줄고 valid도 안정적으로 개선 |
| 너무 큼 | 최적점을 지나침 | loss가 진동하거나 발산, NaN |

예시(같은 조건 40 epoch): lr=0.0001은 안정적이지만 느리고(최소 valid loss ≈ 0.53), lr=0.01은 빠르게 줄고(≈ 0.12), lr=0.5는 초반에 크게 흔들려요. 좋은 학습률은 Optimizer · batch 크기 · 모델 · 정규화와 함께 정해져요. 정답 하나가 아니라 후보 범위를 정해 곡선으로 비교하는 하이퍼파라미터예요.

[그림: AdamW에서 학습률 0.0001, 0.01, 0.5로 학습한 valid loss 곡선] 무엇을 보나: 초승달 모양 toy 데이터(train 240 · valid 80)에 강의와 같은 2→64→64→2 모델, AdamW, 40 epoch, seed 7로 lr만 바꿔 학습한 실제 valid loss예요. 파랑(0.0001)은 매끈하지만 40 epoch 내내 천천히 내려가고, 초록(0.01)은 2~3 epoch 만에 낮아져 안정, 빨강(0.5)은 초반에 그래프 밖으로 튀어 오르고 끝까지 출렁여요. 🤔 왜? 보폭 = lr × 기울기. 너무 작으면 방향은 맞아도 정한 시간 안에 못 가고, 너무 크면 바닥을 지나쳐 반대 벽으로 튀어요. 빨강도 가끔 낮은 값을 찍지만 매 epoch 크게 흔들려서 "어느 epoch를 고를지" 믿기 어려워요. 그래서 최솟값 하나보다 곡선 모양을 같이 봐요.

🤔 왜 너무 크면 진동·발산하고 너무 작으면 느려? — 갱신량은 lr × 기울기예요. lr이 크면 한 번에 너무 멀리 뛰어 최적점을 지나쳐 반대편으로 가고, 거기서 또 크게 뛰면서 loss가 출렁이거나 점점 커져요(NaN까지). lr이 작으면 방향은 맞아도 보폭이 작아 정한 epoch 안에 도착을 못 해요. (Day9의 경사하강 체험에서 직접 볼 수 있어요.)

### 학습률 문제를 의심할 수 있는 신호

| 보이는 신호 | 의심할 것 | 왜? |
|---|---|---|
| 초반부터 loss가 NaN 또는 매우 큰 값 | lr 과대 · 입력 스케일 · loss 구현 | 너무 큰 보폭이나 너무 큰 입력값이 숫자를 폭발시킴 |
| train loss가 거의 안 줄음 | lr 과소 · 기울기 단절 · Parameter 등록 여부 | 보폭이 작거나, 기울기가 파라미터까지 안 오거나, Optimizer가 그 파라미터를 모름 |
| loss가 위아래로 크게 진동 | lr 과대 · batch 너무 작음 · 데이터 이상 | 보폭이 크거나, 작은 batch의 기울기가 들쭉날쭉 |
| 초반 개선 후 후반 정체 | Scheduler 또는 Early Stopping 후보 | 처음 lr이 끝부분의 미세 조정엔 너무 큼 |

🔧 학습률 후보 비교 명령어 사전에서 자세히 →

```
for lr in [0.0001, 0.01, 0.5]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
```

### Scheduler — 학습 중에 lr을 바꾸기

현재 lr로 학습→ valid 지표 확인→ 정체했나?→ lr 낮추고 계속

Optimizer는 파라미터를 갱신하고, Scheduler는 Optimizer가 쓸 lr을 조정해요. ReduceLROnPlateau는 valid 지표가 한동안 좋아지지 않으면(plateau = 평평한 고원) lr을 깎아요.

[그림: 정체 구간이 있는 valid loss와 StepLR, ReduceLROnPlateau가 정한 학습률 계단] 무엇을 보나: 왼쪽은 설명용으로 만든 valid loss(18 epoch쯤부터 노란 구간에서 정체). 오른쪽은 그 값을 넣어 실제 torch 스케줄러가 정한 lr이에요. 회색 StepLR은 valid와 상관없이 15·30 epoch에 달력처럼 반으로, 분홍 ReduceLROnPlateau(factor=0.5, patience=4)는 정체가 4 epoch 넘게 이어진 뒤에야 반으로 깎아요. 그 뒤 왼쪽 곡선이 조금 더 내려가요. 🤔 왜? 바닥 근처에선 큰 보폭이 바닥을 계속 넘나들어 더 못 내려가요. 보폭을 줄이면 더 깊이 들어갈 수 있어요. Plateau 방식은 "정체했나"를 valid 값으로 판단하니 scheduler.step(valid_loss)처럼 값을 꼭 넣어야 해요.

🔧 ReduceLROnPlateau 명령어 사전에서 자세히 →

```
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-4)
scheduler.step(valid_loss)                     # valid 후, 감시할 값을 넣어서!
```

## ③ Batch · Iteration · Epoch

| 용어 | 뜻 |
|---|---|
| Batch size | 한 번의 forward/backward에 쓰는 샘플 수 |
| Iteration | batch 하나로 optimizer.step()까지 한 횟수 |
| Epoch | train 데이터 전체를 한 바퀴 다 본 단위 |

```
epoch당 iteration 수 = ⌈ 샘플 수 ÷ batch_size ⌉      (drop_last=False일 때)
예: 240건, batch_size=32 → 32×7 = 224건 + 마지막 16건 → 8 iterations
```

[그림: 240건을 batch 8, 32, 240으로 자르면 30, 8, 1 iterations가 되는 그림] 무엇을 보나: 막대 하나 = train 240건 한 바퀴(1 epoch). 색이 바뀌는 조각 하나가 batch 하나 = optimizer.step() 한 번이에요. batch 32 줄의 마지막 조각은 16건이라 짧아요. 🤔 왜? iteration 수 = ⌈240 ÷ batch_size⌉. batch가 작을수록 같은 데이터를 더 잘게 쪼개 더 자주 갱신하고, 크면 한 번에 많이 보고 적게 갱신해요.

|  | 작은 batch | 큰 batch |
|---|---|---|
| 기울기 | 흔들림 큼 | 안정적 |
| 메모리 | 적게 씀 | 많이 씀 (GPU 메모리 부족 주의) |
| epoch당 step 수 | 많음 | 적음 (한 step 계산량은 큼) |

🤔 왜 batch가 작으면 기울기가 흔들려? — batch 기울기는 "전체 데이터 기울기"를 그 batch 샘플들로 어림한 값이에요. 샘플이 적으면 어떤 샘플이 뽑혔냐에 따라 평균이 크게 달라져요(여론조사 표본이 적으면 결과가 들쭉날쭉한 것과 같아요). batch가 크면 어림이 정확해져 안정적이지만, 한 step에 계산할 양과 GPU 메모리가 커지고 epoch당 갱신 횟수는 줄어요.

[그림: batch 8, 32, 240으로 학습할 때 step마다 기록한 train loss] 무엇을 보나: 같은 모델·SGD(lr=0.05)·seed로 batch 크기만 바꿔 15 epoch 학습하며 step마다 그 batch의 loss를 찍었어요. 주황(8)은 위아래로 크게 요동, 보라(32)는 중간, 초록(240 = 전체)은 매끈한 한 줄이에요. 대신 초록은 epoch당 1번만 갱신해서 15 epoch 뒤에도 가장 덜 내려갔어요. 🤔 왜? batch loss·기울기는 "전체의 어림값"이라 샘플 8개로 낸 평균은 어떤 8개가 뽑혔냐에 따라 크게 달라요. 흔들림과 갱신 횟수는 맞바꾸는 관계예요 — 그래서 batch 크기도 lr과 함께 곡선으로 비교해요.

| 교재 설명 예시 · batch size | iteration / epoch | 최소 valid loss | 마지막 valid acc | 해석 |
|---|---|---|---|---|
| 8 | 30 | 0.102 | 0.925 | step 많음 · gradient 변동 큼 |
| 32 | 8 | 0.117 | 0.925 | 중간 크기 |
| 240 | 1 | 0.110 | 0.962 | full-batch · step 1회 |

(240건, 같은 model·optimizer·lr·epoch, batch_size 하나만 변경) 이 작은 예에선 차이가 크지 않았어요. 데이터·모델마다 달라지니 직접 기록해서 비교해요.

🔧 DataLoader 다시 보기 명령어 사전에서 자세히 →

```
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
```

🔧 optimizer.zero_grad() · optimizer.step() 명령어 사전에서 자세히 →

```
optimizer.zero_grad(set_to_none=True)    # 이전 iteration의 gradient 제거
optimizer.step(closure=None)             # 현재 .grad와 optimizer 상태로 갱신
```

## ④ 과적합과 학습 곡선

과적합(Overfitting) = train만 너무 외워서 새 데이터에 약해진 상태예요. 최종 점수 하나로는 안 보이고, epoch별 train·valid 곡선을 같이 봐야 보여요.

|  | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| 모습 | train도 나쁨 | train·valid 함께 개선 | train 개선 · valid 악화 |
| 비유 | 공부를 덜 함 | 이해하며 공부함 | 문제집 답만 외움 |

예시: train 60건, hidden 64, 규제 없음 → valid loss 최저점은 약 13 epoch. 그 뒤 train loss는 계속 줄지만 valid loss는 다시 커져요. 마지막 epoch가 항상 최선은 아니에요. 모델 크기, epoch 수, 데이터 양, 노이즈, 규제 여부가 모두 과적합에 영향을 줘요.

[그림: train 60건 규제 없는 모델의 train loss와 valid loss, best epoch 15 표시] 무엇을 보나: 강의 설정처럼 train 60건 · hidden 64 · 규제 없음(Adam, lr=0.01)으로 200 epoch 학습한 실제 곡선. 보라 train loss는 끝까지 0을 향해 내려가는데, 분홍 valid loss는 15 epoch에서 바닥을 찍고 그 뒤 계속 올라가요. 🤔 왜? 15 epoch까지는 두 달 모양이라는 "공통 규칙"을 배워 둘 다 좋아져요. 그 뒤로는 train 60건에만 있는 노이즈 점까지 맞추느라 train loss만 더 깎이고, 새 자료(valid)에선 오히려 틀려요. 마지막 epoch(200)가 최선이 아닌 이유예요.

🤔 왜 train은 계속 좋아지는데 valid는 나빠져? — 모델은 train loss를 줄이는 방향으로만 움직여요. 처음엔 데이터에 공통인 큰 규칙을 배우니 둘 다 좋아지지만, 그걸 다 배우고 나면 train에만 있는 우연한 패턴과 노이즈까지 외워서 train loss를 더 깎아요. 그 외운 내용은 새 자료(valid)엔 없으니 오히려 틀리게 돼요. 모델이 크거나(외울 공간이 많음), 데이터가 적거나(외우기 쉬움), 오래 학습할수록(외울 시간이 많음) 잘 생겨요. 그래서 마지막 숫자 하나가 아니라 epoch별 곡선을 봐야 언제부터 외우기 시작했는지 보여요.

[그림: Underfitting 직선 경계, best epoch의 매끈한 경계, epoch 200의 구불구불한 경계] 무엇을 보나: 같은 train 60점 위의 결정 경계 세 개. 왼쪽 선형 모델은 직선이라 달 모양을 못 따라가고(Underfitting), 가운데 best epoch 15는 큰 흐름만, 오른쪽 epoch 200은 반대편에 박힌 몇몇 점까지 가늘게 감싸려고 경계가 튀어나와요(Overfitting). 🤔 왜? 오른쪽의 튀어나온 부분은 노이즈 점 한두 개를 맞추려고 생긴 거라, 새 데이터가 그 근처에 오면 오히려 틀려요. 위 곡선에서 epoch 15 뒤 valid loss가 오르는 게 바로 이 모양 변화예요.

| 곡선 패턴 | train loss | valid loss | 가능한 원인 | 우선 점검 |
|---|---|---|---|---|
| 둘 다 높음 | 높거나 천천히 감소 | 높거나 천천히 감소 | Underfitting · lr 과소 | 모델 · lr · 학습 시간 |
| train↓ valid↑ | 계속 감소 | 최저점 후 증가 | Overfitting | Early Stopping · 규제 · 데이터 |
| 둘 다 진동 | 큰 폭 변동 | 큰 폭 변동 | lr 과대 · batch 과소 | lr · batch · 입력 스케일 |
| NaN 발생 | NaN | NaN | 수치 불안정 · 잘못된 loss | 입력 · lr · gradient norm |

🔧 epoch별 기록 남기기 명령어 사전에서 자세히 →

```
history["valid_loss"].append(valid_loss)
best_epoch = history["valid_loss"].index(min(history["valid_loss"])) + 1
```

🔧 재현성 torch.manual_seed() 명령어 사전에서 자세히 →

```
torch.manual_seed(42)     # PyTorch 난수 순서 고정
```

## ⑤ Dropout과 BatchNorm — train/eval에서 다르게 동작하는 층

둘 다 모델 안에 들어가는 층이에요. 흔한 배치 예: Linear → BatchNorm → ReLU → Dropout → 다음 층

[그림: ] 무엇을 보나: 은닉층 하나의 흔한 순서와, 모드 스위치에 따라 달라지는 두 칸(분홍 배경). train에선 BatchNorm이 지금 batch 통계를 쓰고 Dropout이 일부를 끄지만, eval에선 저장된 통계를 쓰고 아무것도 안 꺼요. 🤔 왜? 학습 때는 "흔들어서" 외우기를 막거나 batch마다 분포를 맞추는 게 도움이 되지만, 평가·서비스에선 같은 입력이면 같은 답이 나와야 해요. 그래서 model.eval() 한 줄로 이 두 층만 바꿔요.

### Dropout — 일부 연결을 잠깐 끄기

입력 activation→ 무작위 mask 확률 p로 0→ 남은 값 키우기 × 1/(1−p)→ 다음 층

학습할 때 뉴런 일부를 무작위로 꺼서, 모델이 특정 특징 하나에만 기대지 않게 해요. 남은 값을 1/(1−p)배 키워서 평균 크기를 맞춰요(scale 보정). 평가(eval) 땐 아무것도 안 끄고 그대로 통과시켜요.

[그림: Dropout으로 forward마다 다른 은닉 뉴런이 꺼진 신경망 세 개와 전부 켜진 eval 신경망] 무엇을 보나: 같은 네트워크를 train 모드로 세 번 forward 했을 때 꺼진 뉴런(흰 원 x)과 끊긴 선(연한 회색). 매번 다른 뉴런이 꺼져요. 맨 오른쪽 eval에선 전부 켜져요. 🤔 왜? 어떤 뉴런이 꺼질지 모르니 모델은 "이 뉴런 하나만 믿자"를 못 해요. 매번 조금씩 다른 작은 네트워크로 학습하는 셈이라 정보가 여러 뉴런에 나눠 담기고, eval에선 그 작은 네트워크들을 평균 낸 것처럼 전부 써요.

🤔 왜 일부를 끄면 과적합이 줄어? — 매 forward마다 어떤 뉴런이 꺼질지 모르니, 모델은 "이 뉴런 하나만 믿으면 된다"는 식으로 특정 특징에 지나치게 의존할 수 없어요. 여러 뉴런에 정보를 나눠 담게 되고, 그만큼 train에만 있는 우연한 조합을 외우기 어려워져요. 효과는 valid 결과로 확인해요. 🤔 왜 남은 값을 키워? — 평가 땐 전부 켜니까, 학습 때 일부만 켜진 상태와 다음 층이 받는 값의 평균 크기가 같아야 해요. 그래서 학습 때 남은 값을 1/(1−p)배 해 둬요. 아래 체험에서 출력 평균이 입력 평균 근처에 머무는 걸 보세요.

🔧 nn.Dropout(p) 명령어 사전에서 자세히 →

```
nn.Linear(2, 64), nn.ReLU(), nn.Dropout(p),
```

[그림: Dropout 없음과 p=0.3의 train, valid loss 곡선 비교] 무엇을 보나: 위 과적합 실험(train 60건)에 nn.Dropout(0.3)만 추가했어요(seed·데이터·Optimizer 같음, 측정은 eval 모드). 초록 valid loss가 분홍보다 천천히 올라가고, 초록 train loss는 0에 덜 붙어요. 하지만 초록도 결국은 올라가요. 🤔 왜? 일부를 끄면 train을 완벽히 외우기 어려워져서(train loss가 덜 내려감) 외우는 속도가 늦어져요. 다만 데이터가 60건뿐이면 Dropout 하나로는 다 막지 못해요 — 그래서 아래 Weight Decay·Early Stopping과 함께 쓰고, 효과는 늘 이렇게 곡선으로 확인해요.

### BatchNorm — batch 통계로 중간값을 정리

batch activation→ 평균·분산 계산→ 정규화→ 학습되는 scale·shift

층을 지날 때마다 값의 분포가 들쭉날쭉하면 학습이 불안정해요. BatchNorm은 현재 batch의 평균·분산으로 값을 정리해서 학습을 원활하게 해요. 규제 효과가 조금 있을 수 있지만 본래 목적은 중간 activation의 분포 안정화예요. 학습 중엔 현재 batch 통계를 쓰면서 running_mean·running_var(누적 통계)도 갱신해요.

🤔 왜 정규화하면 학습이 원활해져? — 앞 층 가중치가 바뀔 때마다 다음 층이 받는 값의 크기·중심이 계속 움직이면, 다음 층은 "움직이는 과녁"을 맞추느라 같은 lr로도 불안정해져요. 값을 평균 0·분산 1 근처로 정리해 두면 층마다 받는 입력이 비슷한 범위에 머물러 학습이 덜 흔들려요. 그 뒤 학습되는 scale·shift가 있어서 필요하면 모델이 다시 원하는 범위로 바꿀 수 있어요. 🤔 왜 eval 땐 running 통계를 써? — 평가·서비스에선 샘플이 1개만 들어올 수도 있고, 같은 입력이면 같은 답이 나와야 해요. batch 통계를 쓰면 함께 들어온 다른 샘플에 따라 결과가 바뀌니까, 학습 동안 모아 둔 고정 통계를 써요.

[그림: BatchNorm 없을 때와 있을 때 층별 activation 분포 히스토그램] 무엇을 보나: 스케일이 엉망인 입력(한 특성은 평균 10·표준편차 5)을 4층 네트워크(학습 전)에 넣고 각 층의 ReLU 직전 값 분포를 그렸어요. 왼쪽(BN 없음)은 1층에선 -10~10 넘게 퍼졌다가 4층으로 갈수록 0 근처로 쪼그라들어요(표준편차 4.9 → 0.3). 오른쪽(BN 있음)은 네 층 모두 평균 0 · 표준편차 1로 같은 모양이에요. 🤔 왜? BN이 층마다 batch 평균을 빼고 표준편차로 나누니까, 앞에서 무슨 일이 있었든 다음 층은 늘 비슷한 범위의 값을 받아요. 층마다 범위가 제각각이면 같은 lr이 어떤 층엔 너무 크고 어떤 층엔 너무 작아 학습이 흔들려요.

|  | model.train() | model.eval() |
|---|---|---|
| Dropout | 켜짐 (무작위로 끔) | 꺼짐 (그대로 통과) |
| BatchNorm | 현재 batch 평균·분산 사용, running 통계 갱신 | 저장된 running_mean·running_var 사용, 갱신 없음 |

🔧 nn.BatchNorm1d() 명령어 사전에서 자세히 →

```
nn.Linear(2, 64), nn.BatchNorm1d(64), nn.ReLU(),
```

🔧 model.train() · model.eval() 명령어 사전에서 자세히 →

```
model.train()                 # 학습 모드 (= train(True))
model.eval()                  # 평가 모드 (= train(False))
```

## ⑥ Weight Decay와 Early Stopping

### Weight Decay — 큰 가중치에 벌점

가중치가 지나치게 커지는 걸 막아서 모델이 데이터를 과하게 외우지 않게 해요.

|  | weight_decay 동작 |
|---|---|
| SGD | 기울기에 L2 형태의 항을 더해서 갱신 |
| AdamW | 적응적 갱신과 분리해서 파라미터를 직접 감쇠 |

⚠️ weight_decay가 너무 크면 train 성능도 안 오르는 Underfitting이 생겨요.

[그림: weight decay 0과 0.5로 학습한 가중치 히스토그램과 결정 경계] 무엇을 보나: 위 과적합 실험을 AdamW로 weight_decay만 0 vs 0.5로 바꿔 200 epoch 학습한 뒤, 두 번째 Linear(64, 64) 가중치 4096개의 분포예요. 초록(0.5)은 0 근처에 뾰족하게 몰리고 평균 크기 |w|가 0.217 → 0.102로 절반이 됐어요. 마지막 valid loss도 1.76 → 0.49. 오른쪽 경계는 epoch 200인데도 과적합 그림보다 덜 구불구불해요. 🤔 왜? AdamW는 매 step 가중치를 조금씩 0 쪽으로 끌어당겨요. 그래서 꼭 필요한 가중치만 크게 남고 나머지는 작아져, 점 하나하나를 감싸는 뾰족한 경계를 만들기 어려워져요. (0.5는 차이를 보이려고 일부러 크게 준 값이에요 — 실제로는 후보를 valid로 비교해 골라요.)

🤔 왜 가중치가 작으면 과적합이 줄어? — train의 노이즈까지 딱 맞추려면 몇몇 가중치가 아주 커져서 입력이 조금만 바뀌어도 출력이 크게 휙휙 바뀌는 "뾰족한" 함수가 돼요. 큰 가중치에 비용을 주면 모델은 꼭 필요한 만큼만 가중치를 키우게 되고, 더 완만한(단순한) 함수를 고르게 돼요. 반대로 비용이 너무 크면 필요한 가중치까지 못 키워서 train조차 못 맞춰요. 🤔 왜 AdamW는 "분리"해? — Adam은 파라미터마다 보폭을 자동 조절하는데, weight decay를 기울기에 섞어 넣으면 그 감쇠도 같이 조절돼 버려서 파라미터마다 감쇠 세기가 들쭉날쭉해져요. AdamW는 감쇠를 적응적 갱신과 따로 적용해서 의도한 만큼 고르게 줄여요.

🔧 파라미터 그룹으로 weight decay 다르게 주기 명령어 사전에서 자세히 →

```
{"params": decay, "weight_decay": 0.01},
{"params": no_decay, "weight_decay": 0.0},
```

### Early Stopping — 좋아지지 않으면 멈추고 best로 되돌리기

| 변수 | 뜻 |
|---|---|
| best | 지금까지 가장 좋은 valid 지표 |
| patience | 개선 없이 기다려 줄 epoch 수 |
| min_delta | "개선"으로 인정할 최소 변화량 |
| best_state | 가장 좋았던 시점의 model.state_dict() 복사본 |

예시 (patience=3, min_delta=0): valid loss 0.65 → 0.50 → 0.42 → 0.45 → 0.47 → 0.46. 3 epoch에 best 저장, 4·5·6에서 대기 1·2·3회 → 6 epoch에 중단 → 3 epoch 상태로 복원.

[그림: valid loss 곡선 위에 best epoch 15, patience 10 대기 구간, stop epoch 25, best로 되돌리는 화살표] 무엇을 보나: 위 과적합 곡선에 patience=10을 적용한 모습. 15 epoch에서 best(노란 점) 저장 → 노란 구간 10 epoch 동안 더 좋아지지 않음 → 25 epoch에 멈춤(빨간 선) → 화살표처럼 15 epoch 상태로 되돌려요. 점선은 멈추지 않았다면 갔을 길(valid가 계속 나빠짐)이에요. 🤔 왜? "더 이상 안 좋아진다"는 기다려 봐야 알 수 있어서, 멈춘 순간(25)의 모델은 이미 best보다 10 epoch 더 외운 상태예요. 그래서 멈추기와 되돌리기(load_state_dict(best_state))가 세트예요. 강의 예시(best 13 · stop 25)와 같은 구조예요.

🤔 왜 멈추고, 왜 되돌려? — 과적합이 시작되면 학습을 더 할수록 valid가 나빠지니까, 개선이 멈춘 시점에서 끝내면 시간도 아끼고 외우기도 막아요. 그런데 "개선이 멈췄다"는 건 patience만큼 기다려 본 뒤에야 알 수 있어서, 멈춘 순간의 모델은 이미 best보다 몇 epoch 더 학습된(조금 나빠진) 상태예요. 그래서 중단 여부 판단과 별개로, best 시점의 state_dict를 저장해 뒀다가 복원해요. 감시는 train이 아니라 valid로 해요 — train은 학습에 직접 쓰여서 거의 항상 계속 좋아지니 기준이 될 수 없어요.

설명용 가상 곡선에서 patience와 min_delta를 바꿔 보세요. (실제 실험값이 아니에요)

🔧 Early Stopping 직접 짜기 명령어 사전에서 자세히 →

```
best_state = copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)                   # 멈춘 시점이 아니라 best로!
```

### 네 기법 비교

| 기법 | 무엇을 바꾸나 | train/eval 차이 | 주요 설정 | 주의 |
|---|---|---|---|---|
| Dropout | activation 일부를 무작위로 0 | 있음 | p | p 과대 → Underfitting |
| BatchNorm | 중간 activation 정규화 | 있음 | num_features, momentum | 작은 batch, 모드 전환 |
| Weight Decay | 파라미터 크기 억제 | 없음 | weight_decay | 과대 → train 성능 저하 |
| Early Stopping | 학습 epoch 수 제한 | 선택 과정 | patience, min_delta | best state 복원 필수 |

## 증상으로 진단

먼저 물어볼 것: ① train과 valid 중 어디가 문제? ② 초반부터인가, 후반부터인가? ③ loss와 accuracy가 같은 결론인가? ④ 한 설정만 바꿨나?

| 관찰한 곡선 | 가능한 원인 | 먼저 해 볼 것 | 추가 확인 |
|---|---|---|---|
| train·valid 둘 다 높음 / 천천히 감소 | Underfitting, lr 과소 | 모델 용량·lr·epoch 늘리기 | 데이터·loss·기울기 |
| train↓ valid는 최저점 후 ↑ | Overfitting | Early Stopping · Dropout · Weight Decay | 데이터 양, 라벨 노이즈 |
| 둘 다 크게 진동 | lr 과대, batch 과소 | lr 낮추기 · batch 키우기 | 입력 정규화(스케일) |
| 초반 개선 후 정체 | plateau | Scheduler · 다른 Optimizer | 기울기 크기, 활성화 |
| loss가 NaN / Inf | 수치 불안정 | lr 낮추기 · 입력·loss 검사 | gradient clipping, dtype |
| train loss가 거의 안 줄음 | lr 과소, 기울기 끊김 | lr 키우기 | 파라미터가 Optimizer에 등록됐나 |
| GPU 메모리 부족(OOM) | batch가 너무 큼 | batch 줄이기 | gradient accumulation |
| 실행마다 valid 결과가 다름 | eval 누락, seed, shuffle | model.eval() 확인 | test를 반복 확인하진 않기 |
| BatchNorm 결과가 이상함 | batch 너무 작음, 모드 착각 | batch 크기·train/eval 확인 | drop_last |

## 원본 자료

- 강의: Day10_Optimizer_Regularization_20261030_v2.pdf — 큰 그림 p2–11 · Optimizer p17–25 · 학습률·Scheduler p27–35 · Batch·Epoch p38–45 · 과적합·학습 곡선 p48–54 · Dropout·모드 p57–64 · BatchNorm·Weight Decay p67–74 · Early Stopping p77–81 · 오류 처방 p82·p86–87

- 공식 문서: torch.optim (SGD · AdamW · lr_scheduler) · nn.Dropout · nn.BatchNorm1d · DataLoader

## 🔧 Day10 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day10)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | torch (Day9 환경 유지) · 기록·그래프용 pandas, matplotlib |
| ① Optimizer | torch.optim.SGD(momentum=), torch.optim.AdamW() |
| ② 학습률 · Scheduler | lr=, ReduceLROnPlateau, scheduler.step(valid_loss) |
| ③ Batch · Iteration · Epoch | DataLoader(batch_size=, shuffle=, drop_last=), optimizer.zero_grad() / step() |
| ④ 과적합 · 학습 곡선 | history 기록, torch.manual_seed() |
| ⑤ Dropout · BatchNorm | nn.Dropout(p), nn.BatchNorm1d(), model.train() / model.eval() |
| ⑥ Weight Decay · Early Stopping | weight_decay=, copy.deepcopy(model.state_dict()), load_state_dict() |

### Optimizer만 바꿔 끼우기

```
model = Net()
loss_fn = nn.CrossEntropyLoss()

# 선택 1
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
# 선택 2
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

for xb, yb in train_loader:          # 학습 루프는 그대로!
    optimizer.zero_grad()
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optimizer.step()
```

▶ 실제로 돌려 보면 points.csv (train 240 · valid 80) · 2→32→32→2, seed 42, 같은 batch 순서, 20 epoch 뒤 valid

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SGD(lr=0.05, momentum=0) | valid loss 0.3324 · acc 0.863 (epoch 1 train loss 0.6777) | momentum 없으면 20 epoch에 아직 덜 내려감 |
| SGD(lr=0.05, momentum=0.9) | valid loss 0.0791 · acc 0.988 (epoch 1 train loss 0.6589) | 이전 방향을 누적해 같은 lr로도 훨씬 빨리 내려감 |
| AdamW(lr=0.01, weight_decay=0.01) | valid loss 0.0438 · acc 0.988 (epoch 1 train loss 0.5556) | 파라미터마다 보폭을 조절 — 이 데이터·설정에선 이렇게 나왔을 뿐, 항상 1등은 아님 |
| sorted(optimizer.state[첫 weight].keys()) (AdamW) | ['exp_avg', 'exp_avg_sq', 'step'] | step마다 쌓은 이동평균 — state_dict()에 같이 저장돼요 |

교재 설명 예시(같은 split · 40 epochs): SGD valid acc ≈ 0.913, AdamW ≈ 0.925. 초반 수렴 속도와 후반 변동이 달랐지만, 이 그래프 하나로 "항상 AdamW가 우수하다"고 결론 내리지 않아요.

torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False, ...) torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False, ...)

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| params (필수) | 갱신할 Parameter iterable, 보통 model.parameters() | 여기 없는 파라미터는 안 바뀜 |
| lr | 학습률 (SGD·AdamW 기본 0.001) | 한 번에 움직이는 기본 크기 → ② 학습률과 Scheduler 참고 |
| momentum (SGD) | 이전 방향 누적 비율 (기본 0) | 0.9 정도면 흔들림↓, 같은 방향 가속. 0이면 순수 SGD |
| dampening · nesterov (SGD) | momentum 버퍼에 더하는 현재 기울기를 줄이는 계수 / Nesterov 방식 사용 | 보통 기본값 그대로 |
| betas (AdamW) | 두 이동평균(1차·2차) 계수 (기본 (0.9, 0.999)) | 값이 클수록 과거를 오래 기억해 변화가 느리고 부드러움 |
| eps (AdamW) | 0으로 나누기 방지 (기본 1e-8) | 제곱 평균이 0에 가까울 때 보폭이 폭발하지 않게 |
| amsgrad · maximize | 변형 알고리즘 사용 / 최대화 문제 여부 | 보통 False |
| weight_decay | 큰 가중치 억제 (SGD 기본 0, AdamW 기본 0.01) | 키우면 가중치가 작게 유지 → 과적합↓, 너무 크면 Underfitting (⑥ Weight Decay) |

🤔 왜 Optimizer만 바꿔 끼우고 나머지는 그대로 둬? — 결과가 달라졌을 때 그게 Optimizer 덕인지, 데이터 분할·seed·epoch가 바뀐 탓인지 구분하려면 바꾼 조건이 하나여야 해요. 학습 루프 구조(zero_grad → forward → loss → backward → step)는 Optimizer와 상관없이 똑같아서 객체만 교체하면 돼요.

💡 AdamW는 step마다 내부 상태(이동평균)를 갱신하고, 그 상태도 optimizer.state_dict()에 들어 있어요. 학습을 이어서 하려면 이것도 저장해요.

### 학습률 후보 비교

```
for lr in [0.0001, 0.01, 0.5]:
    torch.manual_seed(7)                       # 고정: seed
    model = Net()                              # 고정: 모델
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = train(model, optimizer, train_loader, valid_data, epochs=40)
    print(lr, min(history["valid_loss"]))      # 바꾼 건 lr 하나뿐
```

▶ 실제로 돌려 보면 points.csv (train 240 · valid 80) · AdamW, 후보마다 seed 7로 새 모델, 40 epoch

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| lr=0.0001 → min(history['valid_loss']) | 0.5696 (마지막 epoch 0.5696) | 3등 — 너무 작아 40 epoch 동안 거의 제자리 |
| lr=0.01 → min(history['valid_loss']) | 0.0223 (마지막 epoch 0.0255) | 1등 — 가장 낮은 valid loss, 마지막까지 안정 |
| lr=0.5 → min(history['valid_loss']) | 0.0230 (마지막 epoch 0.0608) | 2등 — 최저값은 0.01과 거의 같지만 마지막엔 다시 올라감 — 큰 보폭으로 흔들림. 이 쉬운 데이터라 버틴 것 |
| valid 기준으로 고른 lr | 0.01 | test는 고르는 데 쓰지 않아요 |

🤔 왜 후보마다 seed를 다시 고정하고 모델을 새로 만들어? — 같은 초기 가중치에서 출발해야 차이가 오직 lr에서 왔다고 말할 수 있어요. 이전 후보로 학습된 모델을 이어 쓰면 출발점이 달라져 비교가 안 돼요. 그리고 고르는 기준은 valid예요. test는 후보 선택에 쓰지 않아요.

### ReduceLROnPlateau

```
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-4)

for epoch in range(epochs):
    train_one_epoch(model, train_loader, optimizer)
    valid_loss = evaluate_loss(model, valid_loader)
    scheduler.step(valid_loss)                     # valid 후, 감시할 값을 넣어서!
    current_lr = optimizer.param_groups[0]["lr"]   # 지금 lr 확인
```

▶ 실제로 돌려 보면 가짜 loss 1.0을 계속 넣은 경우 · points.csv (train 240 · valid 80)로 60 epoch 실제 학습한 경우

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| valid_loss = 1.0 을 12번 step → 각 step 뒤 lr | 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0025, 0.0025 | 처음 값이 best, 그 뒤 5번째 '개선 없음'(patience 4 초과)에서 ×0.5, 다시 5번 뒤 또 ×0.5 |
| 실제 학습: lr이 바뀐 epoch | ep38: 0.005, ep51: 0.0025, ep59: 0.00125 | valid loss가 정체할 때만 깎임 |
| optimizer.param_groups[0]['lr'] (학습 끝) | 0.00125 | 60 epoch 동안 세 번 절반씩 — min_lr(1e-4)까지는 아직 여유 |
| scheduler.step() (값 없이) | TypeError: ReduceLROnPlateau.step() missing 1 required positional argument: 'metrics' | 감시할 값(metrics)이 필수 인자라 바로 오류 |

torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-8)

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| optimizer (필수) | 감쌀(lr을 조정할) Optimizer 객체 |  |
| mode | "min" = 값이 줄어야 좋음(loss), "max" = 커야 좋음(accuracy·F1) | 감시 값과 방향이 반대면 좋아지고 있는데도 "정체"로 보고 lr을 깎아요 |
| factor | lr에 곱할 배수 (기본 0.1) | 0.5면 0.01 → 0.005 → 0.0025. 작을수록 한 번에 크게 깎음 |
| patience | 개선 없이 참아 줄 epoch 수 (기본 10) | 작으면 잠깐 흔들려도 빨리 깎고, 크면 오래 기다림 |
| threshold | "의미 있는 개선"으로 칠 최소 변화 (기본 1e-4) | 키우면 아주 작은 개선은 개선으로 안 쳐요 |
| cooldown, min_lr, eps | 깎은 뒤 쉬는 epoch / lr 하한 / 이보다 작은 lr 변화는 적용 안 함 | min_lr 아래로는 안 내려가요 |

| 교재 예시 구간 | validation 상황 | learning rate | 해석 |
|---|---|---|---|
| 1~18 epoch | 개선 또는 patience 이내 | 0.0100 | 기본 학습률 유지 |
| 19~23 epoch | 의미 있는 개선 없음 | 0.0050 | factor=0.5 적용 |
| 24~25 epoch | 다시 정체 | 0.0025 | 두 번째 감소 |

⚠️ ReduceLROnPlateau는 scheduler.step()에 감시할 값(보통 valid loss)을 꼭 넣어야 해요. 값 없이 부르면 잘못된 사용이에요.

🤔 왜 학습 도중에 lr을 낮춰? — 처음엔 큰 보폭으로 빨리 내려가는 게 좋지만, 바닥 근처에선 같은 보폭이 오히려 바닥을 계속 넘나들게 해서 더 안 내려가요(정체). 이때 보폭을 줄이면 더 세밀하게 바닥으로 들어갈 수 있어요. "정체했는지"는 valid 결과를 봐야 알 수 있으니, valid 계산 후 그 값을 넣어 step()을 불러요.

### DataLoader 다시 보기

```
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
```

▶ 실제로 돌려 보면 points.csv (train 240 · valid 80)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(train_loader), 각 batch 크기 | 8, [32, 32, 32, 32, 32, 32, 32, 16] | 240 = 32×7 + 16 → 8 iteration / epoch |
| len(DataLoader(train_dataset, 32, drop_last=True)) | 7 | 16개짜리 마지막 batch를 버림 → 그 16개는 그 epoch에 안 씀 |
| len(valid_loader), 각 batch 크기 | 3, [32, 32, 16] | 80 = 32×2 + 16 |
| train: epoch 두 번의 첫 샘플이 같나? | False | shuffle=True라 epoch마다 순서가 바뀜 |
| valid: 두 번 돌린 첫 샘플이 같나? | True | shuffle=False라 순서 그대로 |

DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, ...)

| 인자 | 뜻 | 바꾸면? / 언제? |
|---|---|---|
| dataset (필수) | Dataset 객체 (또는 호환 dataset) |  |
| batch_size | batch당 샘플 수 (기본 1) | 키우면 iteration↓·메모리↑·기울기 안정, 줄이면 반대 |
| shuffle=True | epoch마다 순서 섞기 | train에서 — 매번 같은 순서면 순서 패턴의 영향을 받으니 섞어요 |
| shuffle=False | 순서 유지 (기본) | valid/test에서 — 평가 결과와 ID·출력 순서가 매번 같게 유지돼요 |
| drop_last=True | 마지막 작은 batch 버림 | BatchNorm에서 작은 batch가 문제일 때. 대신 그 epoch에 일부 샘플은 안 씀 |
| num_workers, pin_memory | 로딩 subprocess 수, CUDA 전송용 pinned memory | 늘리면 로딩이 빨라질 수 있지만 Windows는 multiprocessing 시작 방식 확인 → 0부터 |
| sampler, collate_fn | 뽑는 방식 / 샘플 합치는 방식 | sampler 쓰면 shuffle과 충돌 (둘 다 "순서"를 정하려 해서) |

### optimizer.zero_grad() · optimizer.step()

```
optimizer.zero_grad(set_to_none=True)    # 이전 iteration의 gradient 제거
loss = loss_fn(model(xb), yb)
loss.backward()
optimizer.step(closure=None)             # 현재 .grad와 optimizer 상태로 갱신
```

▶ 실제로 돌려 보면 points.csv (train 240 · valid 80)의 batch 하나(32개), SGD lr=0.05

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| zero_grad(set_to_none=True) 후 weight.grad | None | 처음엔 기울기 자체가 없음 |
| loss.backward() 후 weight.grad.shape | (32, 2) | 파라미터와 같은 shape로 채워짐 |
| optimizer.step() 의 반환값 · weight 변화량(평균 절댓값) | None · 0.00032 | step은 값을 돌려주지 않고 파라미터를 직접 고침 |
| 같은 batch의 loss: step 전 → 후 | 0.6991 → 0.6916 | 한 번 갱신하니 그 batch loss가 줄어듦 |
| zero_grad(set_to_none=False) 후 weight.grad | 0.0 (shape (32, 2)) | None 대신 0으로 채운 Tensor |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| set_to_none | 기본 True → .grad를 None으로 | False면 0 Tensor로 채움. None과 0은 일부 동작이 달라요(None이면 그 파라미터는 "기울기 없음"으로 취급) |
| closure | loss를 다시 계산하는 함수 | 일부 특수 Optimizer용, 대부분 생략 |

🤔 왜 순서가 언제나 zero_grad → forward → loss → backward → step? — iteration 하나는 "이번 batch의 기울기로 한 번 갱신"이에요. 지난 기울기를 먼저 지워야 이번 것만 남아요. 일부러 여러 batch의 기울기를 모으고 싶을 때(누적 학습, 예: GPU 메모리 부족 시 gradient accumulation)만 zero_grad 위치를 의도적으로 바꿔요.

### epoch별 기록 남기기

```
history = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(...)
    valid_loss, valid_acc = evaluate(...)          # 같은 eval 조건으로
    history["train_loss"].append(train_loss)
    history["valid_loss"].append(valid_loss)
    history["train_acc"].append(train_acc)
    history["valid_acc"].append(valid_acc)

best_epoch = history["valid_loss"].index(min(history["valid_loss"])) + 1
pd.DataFrame(history).to_csv("history.csv", index=False)
plt.plot(history["train_loss"], label="train"); plt.plot(history["valid_loss"], label="valid"); plt.legend()
```

▶ 실제로 돌려 보면 points.csv train 앞 60건만 학습 · valid 80 · 2→64→64→2, AdamW lr=0.01, 200 epoch

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(history['valid_loss']) | 200 | epoch마다 하나씩 |
| best_epoch | 150 | valid loss가 가장 낮았던 epoch |
| min(history['valid_loss']) → 마지막 valid loss | 0.0313 → 0.0333 | best 뒤로 valid가 조금 다시 올라감 → 마지막 모델 ≠ 최선 모델 (경계가 쉬운 데이터라 과적합은 약하게만 보여요) |
| train loss: 1 epoch → 마지막 | 0.7091 → 0.0009 | train은 0에 붙도록 계속 내려감 — 60건을 외워 가는 중 |
| valid acc: best_epoch → 마지막 | 0.988 → 0.988 | loss는 올라도 정확도는 그대로 — 둘이 다른 말을 할 수 있어서 따로 기록 |
| pd.DataFrame(history).shape | (200, 3) | CSV로 저장할 표: epoch × 기록 항목 |

| 기록 요령 | 왜? |
|---|---|
| loss와 metric(정확도)을 따로 저장 | 둘이 다른 결론을 낼 수 있음 |
| epoch 번호, lr도 같이 | Scheduler가 언제 lr을 바꿨는지 확인 |
| CSV·그래프로 저장 | 다시 실행한 결과와 비교 |
| train·valid를 같은 epoch 말, 같은 eval 조건으로 | 한쪽만 Dropout이 켜진 상태면 공정한 비교가 아님 |
| 최저 valid loss epoch 계산 | 마지막 모델 ≠ 최선 모델 (예: best_epoch = 13, last_epoch = 80) |

### 재현성 torch.manual_seed()

```
torch.manual_seed(42)     # PyTorch 난수 순서 고정
```

▶ 실제로 돌려 보면 Net() 첫 층 weight · range(10) 섞기

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| seed 42 → Net(), 다시 seed 42 → Net(): weight 같나? | True | 같은 seed면 같은 초기 가중치 |
| seed 다시 안 걸고 한 번 더 Net() | False | 난수 순서가 이어져서 다른 값 |
| Net()[0].weight[0] (seed 42) | [0.5406, 0.5869] | 이 PC·버전에서 실제로 나온 값 |
| DataLoader(..., generator=Generator().manual_seed(43)) 순서 ×2 | [5, 0, 7, 6, 4, 3, 2, 8, 9, 1] / [5, 0, 7, 6, 4, 3, 2, 8, 9, 1] | shuffle 순서는 Generator seed로 따로 고정 |

⚠️ seed를 고정해도 완전히 같은 결과가 안 나올 수 있어요. DataLoader shuffle에는 별도 Generator나 worker seed가 영향을 주고, GPU 연산·병렬 처리·알고리즘 선택 때문에 완전한 결정론을 얻으려면 속도를 손해 볼 수 있어요. 그래서 보고서에 seed · 버전 · device · batch size · epoch · optimizer 설정을 같이 적어요.

🤔 왜 seed를 고정해? — 초기 가중치, Dropout mask, 섞는 순서가 다 난수예요. seed가 매번 다르면 lr을 바꿔서 좋아진 건지 운이 좋았던 건지 알 수 없어요. 같은 seed로 조건을 하나씩 바꿔야 차이의 원인을 말할 수 있어요.

### nn.Dropout(p)

```
class DropoutNet(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(p),
            nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x)            # (batch, 2) → (batch, 64) → (batch, 64) → (batch, 2)
```

▶ 실제로 돌려 보면 nn.Dropout(p=0.3), 입력 torch.ones(1, 10)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| train 모드: dropout(x) | [0.0, 0.0, 1.4286, 0.0, 1.4286, 0.0, 1.4286, 1.4286, 1.4286, 1.4286] | 꺼진 칸은 0, 남은 칸은 1/(1−0.3) ≈ 1.4286배 |
| train 모드: 10만 칸 중 0이 된 비율 | 0.3007 | 정확히 30%가 아니라 확률적으로 약 30% |
| train 모드: 10만 칸 평균 | 0.9990 | 키워 준 덕에 평균이 1 근처로 유지 |
| eval 모드: dropout(x) | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | 그대로 통과(identity) |
| dropout(x).shape | (1, 10) | 입력 shape와 같음 |

torch.nn.Dropout(p=0.5, inplace=False) — 필수 인자 없음. 입력과 출력 shape가 같고, eval 모드에선 identity(그대로 통과).

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| p | 원소 하나를 0으로 만들 확률 (기본 0.5) | p=0.3이어도 매번 정확히 30%가 꺼지는 건 아님(확률). 키우면 규제가 강해지지만, 너무 크면 정보가 너무 많이 사라져 train도 안 오르는 Underfitting (특히 작은 데이터) |
| inplace | 입력을 직접 수정할지 (기본 False) | True면 메모리를 조금 아끼지만 원본 값이 사라짐 |

⚠️ Dropout에서 흔한 실수: 평가 전에 model.eval()을 안 불러 예측이 실행마다 달라짐 · p를 너무 크게 · Dropout 추가하면서 다른 조건까지 같이 바꿔 효과를 구분 못 함.

### nn.BatchNorm1d()

```
self.net = nn.Sequential(
    nn.Linear(2, 64), nn.BatchNorm1d(64), nn.ReLU(),
    nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
    nn.Linear(64, 2))
```

▶ 실제로 돌려 보면 nn.BatchNorm1d(4), 입력 randn(32, 4) × 3 + 5 (평균 ≈ 5, 표준편차 ≈ 3)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| train: out.mean(0) | [0.0, -0.0, -0.0, 0.0] | batch 통계로 정규화 → 특성마다 평균 0 |
| train: out.std(0) | [1.016, 1.016, 1.016, 1.016] | ≈ 1 (std()는 n−1로 나눠서 1보다 살짝 큼) |
| bn.running_mean (batch 한 번 본 뒤) | [0.44, 0.509, 0.563, 0.551] | 0에서 출발해 momentum 0.1만큼만 batch 평균 쪽으로 |
| eval: out.mean(0) | [3.007, 2.97, 3.98, 3.716] | eval은 running 통계를 써서 아직 평균 0이 아님 |
| train 모드에서 bn(torch.randn(1, 4)) | ValueError: Expected more than 1 value per channel when training, got input size torch.Size(… | batch가 1이면 분산을 못 구해 오류 |
| nn.BatchNorm1d(64)에 (32, 32) 입력 | RuntimeError: running_mean should contain 32 elements not 64 | num_features가 앞 Linear out_features와 달라서 오류 |
| sorted(bn.state_dict().keys()) | ['bias', 'num_batches_tracked', 'running_mean', 'running_var', 'weight'] | 학습 파라미터 + running 통계가 함께 저장 |

torch.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None) — 흔한 배치: Linear → BatchNorm1d → ReLU

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| num_features (필수) | 입력 특성 수 C. 2D 입력이면 (N, C) | 앞 Linear의 out_features와 같아야 함. 다르면 shape 오류 |
| eps | 분모에 더하는 작은 값 (기본 1e-5) | 분산이 0에 가까워도 0으로 나누지 않게 하는 안전장치 |
| momentum | running 통계 갱신 계수 (기본 0.1) | 키우면 최근 batch를 더 많이 반영. Optimizer momentum과 다른 것 |
| affine | 학습되는 scale·bias 사용 (기본 True) | False면 정규화만 하고 다시 조절할 수 없음 |
| track_running_stats | running 통계를 저장할지 (기본 True) | False면 eval에서도 batch 통계를 씀 |

💡 BatchNorm은 학습되는 weight·bias와 running_mean·running_var를 함께 상태(state_dict)로 저장해요. Dropout과 같이 쓸 때는 목적과 순서를 명시하고 실험으로 확인해요.

⚠️ BatchNorm은 batch 통계를 쓰니까 batch_size=1로 학습하면 문제가 생길 수 있어요. 작은 batch는 통계가 불안정하니 batch 크기와 같이 해석해요.

### model.train() · model.eval()

```
model.train()                 # 학습 모드 (= train(True))
...
model.eval()                  # 평가 모드 (= train(False))
with torch.no_grad():         # 기울기 기록 끄기는 따로!
    logits = model(x_valid)
```

▶ 실제로 돌려 보면 Net(dropout=0.3), valid 앞 5건

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model.train() → model(x) 두 번 같나? | False | Dropout이 매번 다른 칸을 꺼서 달라짐 |
| model.eval() → model(x) 두 번 같나? | True | Dropout 꺼짐 → 같은 입력 = 같은 출력 |
| model.eval() is model | True | self를 돌려줘서 이어 쓸 수 있음 |
| model.training, model[3].training | False, False | 하위 층(Dropout)까지 한꺼번에 바뀜 |
| eval()만 하고 no_grad 없이: model(x).requires_grad | True | eval은 기울기 기록을 끄지 않음 → no_grad는 따로 |

| 명령 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| model.train(mode=True) | Module(과 하위 층 전부)을 training mode로 | mode=False면 eval과 같은 상태 |
| model.eval() | evaluation mode로 (train(False)와 동일) | Dropout 비활성, BatchNorm은 running 통계 사용 |
| torch.no_grad() / torch.inference_mode() | gradient 기록 끄기 | eval()과는 별개 스위치 → 평가 땐 함께 |

💡 eval()은 Dropout·BatchNorm의 동작만 바꿔요. 기울기 계산을 끄는 건 torch.no_grad() 또는 torch.inference_mode()예요. 둘 다 self를 돌려줘서 model.eval()처럼 바로 이어 쓸 수 있어요.

🤔 왜 eval()을 꼭 불러? — 안 부르면 평가 중에도 Dropout이 무작위로 뉴런을 끄고, BatchNorm이 평가 batch 통계를 써요(running 통계도 바뀌어요). 그러면 같은 입력의 예측이 실행마다 달라지고, 설정끼리 valid 성능 비교도 불안정해져요. 평가가 끝나고 다시 학습할 땐 model.train()으로 돌려놔요.

### 파라미터 그룹으로 weight decay 다르게 주기

```
decay, no_decay = [], []
for layer in model.modules():
    for name, p in layer.named_parameters(recurse=False):
        if name == "bias" or isinstance(layer, nn.BatchNorm1d):
            no_decay.append(p)          # bias·BatchNorm은 감쇠 제외
        else:
            decay.append(p)

optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": 0.01},
    {"params": no_decay, "weight_decay": 0.0},
], lr=0.001)
print(len(optimizer.param_groups))      # 2
```

▶ 실제로 돌려 보면 Net(width=64, bn=True): Linear 3개 + BatchNorm1d 2개

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(decay), len(no_decay) | 3, 7 | Linear weight 3개 / Linear bias 3 + BN weight·bias 4 |
| [tuple(p.shape) for p in decay] | [(64, 2), (64, 64), (2, 64)] | 입력에 곱해지는 가중치만 |
| len(optimizer.param_groups) | 2 | 그룹 2개 |
| [g['weight_decay'] for g in optimizer.param_groups] | [0.01, 0.0] | 그룹마다 다른 감쇠 |
| [g['lr'] for g in optimizer.param_groups] | [0.001, 0.001] | 그룹에 안 적은 lr은 바깥 기본값을 같이 씀 |
| 파라미터 총수 = decay + no_decay ? | True | 빠진 파라미터 없음 |

💡 초급 단계에선 전체에 같은 weight_decay를 주는 것부터 시작해도 돼요. 그룹 나누기는 다음 단계 기술이에요. 모듈 유형(isinstance)으로 BatchNorm을 판별하니 Sequential의 숫자 이름(0, 1 …)에도 그대로 적용돼요.

🤔 왜 bias·BatchNorm은 감쇠에서 빼기도 해? — weight decay의 목적은 입력에 곱해지는 가중치가 지나치게 커지는 걸 막는 거예요. bias는 출력 위치만 옮기고, BatchNorm의 scale·shift는 정규화 뒤 범위를 되돌리는 역할이라 0쪽으로 끌어당기면 오히려 그 역할을 방해할 수 있어요.

### Early Stopping 직접 짜기

```
import copy
best_loss = reference = float("inf")
wait = 0
for epoch in range(max_epochs):
    train_one_epoch(...)
    valid_loss = evaluate_loss(...)
    if valid_loss < best_loss:                      # 조금이라도 좋아지면 best 저장
        best_loss = valid_loss
        best_state = copy.deepcopy(model.state_dict())
    if valid_loss < reference - min_delta:          # 충분히 좋아졌나? (patience 기준)
        reference, wait = valid_loss, 0
    else:
        wait += 1
        if wait >= patience:
            break
model.load_state_dict(best_state)                   # 멈춘 시점이 아니라 best로!
```

▶ 실제로 돌려 보면 교재 숫자 예 · train 앞 60건 + valid 80, AdamW lr=0.01, patience 10, min_delta 0.001

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| valid loss 1.00, .80, .79, .78, .70 / min_delta=.05, patience=2 | 멈춘 epoch 4, best epoch 4 (loss 0.78) | .79·.78은 충분한 개선이 아니라 대기 1·2 → 4에서 멈춤, best는 .78 (.70은 보지도 못함) |
| 실제 학습: best epoch → 멈춘 epoch | 69 → 79 | best 뒤 10 epoch 동안 0.001 넘게 좋아지지 않아 중단 |
| 멈춘 시점 valid loss | 0.0276 | 그냥 쓰면 best보다 나쁜 모델 |
| model.load_state_dict(best_state) 후 valid loss | 0.0265 | best epoch의 값(0.0265)으로 정확히 돌아옴 |
| deepcopy 없이 담은 state_dict()가 학습 후 바뀌었나? | True | 그냥 담으면 지금 가중치를 가리켜서 같이 바뀜 → deepcopy 필요 |

| 명령 | 뜻 | 주의 |
|---|---|---|
| model.state_dict() | 모델의 모든 가중치(·BN 통계) 묶음 | 그냥 담으면 학습하면서 같이 바뀜 |
| copy.deepcopy(…) | 그 순간 값을 통째로 복사 | best 저장엔 꼭 deepcopy |
| model.load_state_dict(best_state) | 저장한 상태로 되돌림 | 멈춘 뒤 반드시 호출 |
| patience | 개선 없이 기다려 줄 epoch 수 | 작으면 잠깐 흔들려도 일찍 멈춤(덜 학습 위험), 크면 오래 더 돌아 시간 낭비 |
| min_delta | 개선으로 인정할 최소 변화량 | 키우면 아주 작은 개선은 "개선 아님"으로 쳐서 더 빨리 멈춤 |

🤔 왜 best_loss와 reference를 따로 둬? — 역할이 달라서예요. best_loss는 "가장 좋았던 모델"을 저장하려고 작은 개선도 다 기록하고, reference는 "기다림을 리셋할 만큼 충분히 좋아졌나"를 min_delta 기준으로 판단해요. 교재 예: loss 1.00 → .80 → .79 → .78, min_delta=.05, patience=2 → .79와 .78은 .80보다 0.05 이상 좋아진 게 아니라 대기 1·2회 → 4번째에 중단. 하지만 best는 .78로 저장돼 있어요. 🤔 왜 copy.deepcopy? — state_dict()는 지금 학습 중인 가중치를 가리키는 묶음이라, 그냥 담아 두면 다음 epoch 학습 때 같이 바뀌어 버려요. 그 순간 값을 통째로 복사해야 best가 보존돼요.

💡 Scheduler와 같이 쓸 땐 순서가 train → valid loss → scheduler.step(valid_loss) → best 저장·patience 확인. 보통 Early Stopping의 patience를 Scheduler보다 길게 둬서, lr을 낮춘 뒤 다시 좋아질 기회를 줘요. (교재 예시: best epoch 13, stop epoch 25 → 최종 모델은 13 epoch 상태)

🤔 왜 Early Stopping patience가 더 길어야 해? — 둘 다 "정체"를 보고 반응해요. Early Stopping이 먼저 끝나 버리면 Scheduler가 lr을 낮춰 볼 기회가 없어요. Scheduler가 먼저 lr을 줄이고, 그래도 안 좋아질 때 Early Stopping이 멈추게 하려면 기다리는 시간이 더 길어야 해요.

## 📖 Day10 용어 (42개)
- **규제** (Regularization (규제)): 계수가 지나치게 커지지 않게 벌점을 줘서 학습을 제어하는 것. 로지스틱회귀에선 C로 조절하며, C가 작을수록 규제가 세요. 스케일링과는 다른 개념.
- **수렴** (Convergence (수렴)): 학습이 반복하며 더 나아지지 않는 안정된 답에 도달하는 것. max_iter(반복 상한)가 너무 작으면 도달 전에 멈춰 수렴 경고가 나요.
- **F1** (F1 score): Precision과 Recall의 조화평균. 2TP ÷ (2TP + FP + FN). 둘 중 하나라도 낮으면 같이 낮아지고 TN은 반영하지 않아요. 이 수업 분류의 주 지표.
- **결측** (Missing value (결측값)): 데이터가 비어 있는 칸. 모델 대부분은 빈칸을 못 받아서 채우거나(대체) 처리해야 해요.
- **하이퍼파라미터** (Hyperparameter (하이퍼파라미터)): 학습 전에 사람이 정하는 설정값(예: K, weights, C, max_depth). train으로 자동 결정되는 계수와 달리 후보를 정해 valid·교차검증으로 비교해 골라요.
- **과적합** (Overfitting (과적합)): 학습 자료의 우연한 잡음까지 외워서 train 점수만 높고 새 자료(valid) 점수는 낮은 상태. 두 점수 격차가 핵심 신호예요.
- **과소적합** (Underfitting (과소적합)): 모델이 너무 단순하거나 제약이 세서 필요한 규칙도 못 배운 상태. train과 valid 점수가 둘 다 낮아요.
- **일반화** (Generalization (일반화)): 학습에 쓰지 않은 새 자료에서도 잘 맞히는 능력. 모델 평가의 진짜 목표예요.
- **학습률** (Learning rate (학습률)): 한 회차의 보정을 얼마나 반영할지 정하는 비율(예: 0.05). 작으면 조금씩 안정적으로 고치지만 회차가 더 필요해요.
- **L2 규제** (L2 regularization (L2 규제)): 값의 제곱에 벌점을 줘 크게 튀지 않게 하는 규제. XGBoost의 reg_lambda는 잎 값에 걸어 한 번의 보정을 보수적으로 만들어요.
- **조기 종료** (Early stopping (조기 종료)): 따로 떼어 둔 stop 자료의 손실이 정한 회차(예: 15회) 동안 나아지지 않으면 학습을 멈추는 방법. 잡음까지 배우기 전에 멈춰 과적합을 줄여요.
- **Optimizer** (Optimizer (최적화 도구)): 구한 기울기를 "어떤 규칙으로" 파라미터 갱신에 반영할지 정하는 객체. optimizer.step()이 실제로 값을 바꿔요. SGD·AdamW 등이 있어요.
- **Dropout** (Dropout (드롭아웃)): 학습할 때 뉴런 출력 일부를 확률 p로 무작위로 0으로 끄는 층. 특정 뉴런에만 기대지 못하게 해 과적합을 줄여요. 평가 땐 꺼져요(그대로 통과).
- **BatchNorm** (Batch Normalization (배치 정규화)): 층의 중간값을 현재 batch의 평균·분산으로 정리(정규화)해 학습을 안정시키는 층. 평가 땐 학습 중 모아 둔 running 통계를 써요.
- **SGD** (Stochastic Gradient Descent (확률적 경사 하강법)): 지금 batch의 기울기 반대 방향으로 lr만큼 이동하는 가장 기본 Optimizer. 단순하고 해석하기 쉬워요.
- **Momentum** (Momentum (모멘텀, 관성)): 이전 이동 방향을 함께 쓰는 방법. 같은 방향은 쌓여서 가속하고, 좌우로 튀는 흔들림은 서로 상쇄돼 줄어요. (BatchNorm의 momentum과는 다른 뜻)
- **Adam** (Adaptive Moment Estimation): 기울기의 평균(1차)과 제곱 평균(2차)을 기억해서 파라미터마다 갱신 보폭을 따로 조절하는 Optimizer. 초반 수렴이 빠른 경우가 많아요.
- **AdamW** (Adam with decoupled Weight decay): Adam에서 weight decay를 적응적 갱신과 분리해 적용한 버전. 감쇠를 의도한 만큼 고르게 줘요. torch 기본 weight_decay는 0.01.
- **eps** (Epsilon (아주 작은 값)): 0으로 나누는 사고를 막으려고 분모에 더하는 아주 작은 수(예: 1e-8). AdamW·BatchNorm에 있어요.
- **Scheduler** (Learning rate scheduler (학습률 스케줄러)): 학습 도중에 Optimizer가 쓸 학습률(lr)을 바꿔 주는 도구. Optimizer는 파라미터를, Scheduler는 lr을 조정해요.
- **ReduceLROnPlateau** (Reduce Learning Rate On Plateau): valid 지표가 patience만큼 좋아지지 않으면 lr에 factor를 곱해 깎는 Scheduler. step(valid_loss)처럼 감시할 값을 꼭 넣어요.
- **plateau** (Plateau (고원, 정체 구간)): loss가 더 내려가지 않고 평평하게 머무는 구간. 이때 lr을 낮추면 더 세밀하게 내려갈 수 있어요.
- **patience** (Patience (참을성)): 개선이 없어도 기다려 줄 epoch 수. Scheduler와 Early Stopping에서 써요. 작으면 금방 반응하고, 크면 오래 기다려요.
- **min_delta** (Minimum delta (최소 개선량)): "개선"으로 인정할 최소 변화량. 키우면 아주 작은 개선은 개선이 아닌 것으로 쳐서 더 빨리 멈춰요.
- **학습 데이터** (Training set (학습용 데이터, train)): 기울기를 계산하고 가중치를 고치는 데 쓰는 자료. 여기서 잘하는 건 "외웠다"일 수도 있어서 이것만 보면 안 돼요.
- **검증 데이터** (Validation set (검증용 데이터, valid)): 학습에는 안 쓰고 epoch·구조·규제·설정을 고를 때 보는 자료. 새 데이터에서의 실력을 가늠해요.
- **테스트 데이터** (Test set (시험용 데이터, test)): 모든 선택이 끝난 뒤 최종 평가에 딱 한 번 쓰는 자료. test를 보고 설정을 고르면 점수가 실제보다 좋게 나와요.
- **full-batch** (Full-batch): train 데이터 전체를 batch 하나로 넣는 방식. epoch당 step이 1번뿐이에요.
- **학습 곡선** (Learning curve (학습 곡선)): epoch마다 train·valid의 loss·정확도를 기록해 그린 그래프. 과적합·학습률 문제를 진단하는 기본 도구예요.
- **metric** (Metric (평가 지표)): 모델 성능을 사람이 읽기 좋게 잰 숫자(정확도·F1 등). loss와 다른 결론을 낼 수 있어 따로 기록해요.
- **노이즈** (Noise (노이즈)): 데이터에 섞인 우연한 오차나 잘못 붙은 정답. 과적합된 모델은 이런 노이즈까지 외워요.
- **모델 용량** (Model capacity (모델 용량)): 모델이 표현·기억할 수 있는 정도. 층·뉴런이 많을수록 크고, 클수록 외울 공간이 많아 과적합이 생기기 쉬워요.
- **Normalization** (Normalization (정규화)): 값의 범위나 분포(예: 평균 0, 분산 1)를 일정하게 맞춰 정리하는 것. 입력 스케일을 맞추거나 BatchNorm이 중간값을 정리할 때 써요. 규제(Regularization)와 다른 말이에요.
- **activation** (Activation (활성값)): 층을 지나며 나온 중간 출력값. Dropout은 이 값 일부를 0으로 끄고, BatchNorm은 이 값의 분포를 정리해요.
- **running 통계** (Running statistics (누적 평균·분산)): BatchNorm이 학습 중 batch마다 조금씩 모아 둔 평균(running_mean)·분산(running_var). 평가 땐 이 고정 통계를 써요.
- **Weight Decay** (Weight Decay (가중치 감쇠)): 가중치가 지나치게 커지지 않게 벌점을 줘서 더 단순한 함수를 고르게 하는 규제. 너무 크면 Underfitting이 생겨요.
- **파라미터 그룹** (Parameter groups): Optimizer 안에서 파라미터를 묶음별로 나눠 lr·weight_decay를 다르게 주는 기능. 예: bias·BatchNorm은 감쇠 제외.
- **state_dict** (state_dict (모델 상태 묶음)): 모델의 모든 가중치(와 BatchNorm 통계)를 이름별로 담은 사전. load_state_dict()로 그 상태로 되돌려요.
- **deepcopy** (copy.deepcopy (깊은 복사)): 그 순간의 값을 통째로 새로 복사하는 파이썬 함수. state_dict를 그냥 담으면 학습하며 같이 바뀌니 best 저장엔 deepcopy를 써요.
- **OOM** (Out Of Memory (메모리 부족)): GPU 메모리(VRAM)가 모자라 생기는 오류. batch 크기를 줄이는 게 첫 처방이에요.
- **gradient accumulation** (Gradient accumulation (기울기 누적)): 작은 batch 여러 개의 기울기를 일부러 모은 뒤 한 번에 갱신하는 방법. GPU 메모리가 부족할 때 큰 batch처럼 학습하려고 써요.
- **gradient clipping** (Gradient clipping (기울기 자르기)): 기울기 크기가 너무 커지면 정해진 크기로 잘라서 갱신이 폭발하지 않게 하는 방법. NaN 대처법 중 하나예요.

---

# Day11 · CNN·이미지 분류·전이학습
_2026-11-02(월) · 합성곱 · Kernel · Pooling · BatchNorm · ImageFolder · 증강 · ResNet18 전이학습 · CPU/GPU 측정_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day11.html

📌 딱 3줄 요약

- CNN은 작은 창(Kernel)을 사진 위에 옮기며 곱해 더해서 Feature Map을 만들고, Pooling으로 크기를 줄여 가요. 같은 Kernel을 모든 위치에 쓰니까(가중치 공유) 파라미터가 사진 크기와 무관해요.

- 블록을 지날수록 공간은 작아지고 채널은 늘어요. "어디에 있나"를 버리고 "무엇이 있나"로 바꾸는 과정이에요.

- 전이학습은 ImageNet으로 미리 배운 ResNet18의 몸통(Backbone)을 얼려 두고, 마지막 Head를 내 클래스 수에 맞게 바꿔 배워요. 필요하면 뒤쪽 층(layer4)부터 작은 lr로 열어요.

🔧 Day11 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| torch | Conv2d·Pooling 같은 층, 손실 함수, 학습 루프 (Day9에서 설치) | import torch, from torch import nn, import torch.nn.functional as F |
| torchvision | 이미지 폴더 읽기, 변환·증강, 사전학습 모델(ResNet18) | from torchvision import datasets, transforms, from torchvision.models import resnet18, ResNet18_Weights |
| Pillow | PNG·JPG 이미지 파일 열기 (ImageFolder가 내부에서 사용) | from PIL import Image |
| matplotlib (선택) | 이미지·Feature Map·학습 곡선 그리기 | import matplotlib.pyplot as plt |

### 🔧 설치 · 확인

```
python -m pip install pillow matplotlib
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

⚠️ torch와 torchvision은 서로 짝이 맞는 버전이어야 해요. 둘 다 pytorch.org 선택기에서 내 환경(CPU/CUDA)에 맞는 명령으로 함께 깔아요. 이미 GPU용 torch가 잘 돌면 덮어쓰지 마세요.

🤔 왜 짝을 맞춰야 해? — torchvision 안의 일부 연산은 torch의 C++ 내부 코드에 맞춰 미리 컴파일돼 있어요. 버전이 어긋나면 import 단계에서 바로 오류가 나거나, 특정 연산만 이상하게 실패해요.

💡 pip 이름 ≠ import 이름: Pillow → PIL.

## 이거 어디에 써?

Day9~10의 ANN은 입력을 한 줄로 펴서 모든 칸을 모든 뉴런에 연결했어요. 사진에 그러면 파라미터가 폭발하고, "옆 픽셀끼리 가깝다"는 정보도 사라져요. CNN은 작은 창 하나를 사진 전체에 돌려 쓰니까 파라미터가 적고, 물체가 사진 어디에 있어도 찾아요.

🏭

불량 검사제품 사진이 정상인지 불량인지

🩻

의료 영상X-ray에서 이상 소견 분류

🌾

농업잎 사진으로 병충해 종류 맞히기

🧩

데이터가 적을 때사전학습 모델 + 전이학습

사진 크기를 바꿔 가며 첫 층 파라미터 수를 비교해 보세요.

🤔 왜 CNN은 사진이 커져도 파라미터가 그대로야? — ANN은 픽셀 하나하나마다 자기 전용 가중치가 있어서 픽셀 수만큼 늘어나요. Conv는 3×3짜리 Kernel 하나를 모든 위치에 돌려 써요(가중치 공유). "세로선"은 왼쪽 위에 있든 오른쪽 아래에 있든 같은 모양이니, 위치마다 따로 배울 필요가 없거든요. 그래서 개수는 16·3·3·3 + 16 = 448로 고정이에요.

## ⓪ PyTorch 한 장 정리 — 데이터 → 예측 → 손실 → 기울기 → 갱신

CNN으로 넘어가기 전에 Day9~10의 학습 흐름을 한 장으로 다시 묶어요. 모델이 ANN이든 CNN이든 이 다섯 단계는 똑같고, 바뀌는 건 "모델 칸" 안쪽뿐이에요. 예시는 PyTorch 공식 한국어 튜토리얼 「파이토치 기본 익히기」의 FashionMNIST(옷 사진 28×28 흑백, 10개 클래스, 학습용 60,000장)예요.

[그림: train_ds[i], DataLoader, model(X), loss_fn, backward(), step() 여섯 칸에 각 단계의 shape를 적고 step에서 다음 배치로 돌아가는 화살표를 그린 흐름도] 무엇을 보나: 배치 하나가 왼쪽에서 오른쪽으로 지나가요. 샘플 1개 (1, 28, 28)가 DataLoader에서 64개씩 묶여 (64, 1, 28, 28)이 되고, 모델을 지나면 클래스 10개 점수(로짓) (64, 10), 손실 함수를 지나면 숫자 하나 ()가 돼요. backward()는 가중치마다 같은 모양의 기울기를 채우고(첫 층 (512, 784)), step()이 가중치를 옮긴 뒤 다음 배치로 돌아가요. shape는 같은 모양의 텐서를 실제로 모델에 넣어 확인했어요. 🤔 왜 이렇게 나눠 봐? 에러의 대부분은 "칸과 칸 사이 shape가 안 맞아서" 나요. 각 칸의 입·출력 모양을 알면 어디서 틀렸는지 바로 짚을 수 있어요.

🤔 왜 1 epoch가 938번 step이야? — epoch는 "학습 데이터를 한 바퀴", step은 "배치 하나로 가중치를 한 번 갱신"이에요. 60,000장을 64장씩 나누면 937.5묶음이고, 남은 32장도 마지막 묶음으로 쓰니까 올림해서 938번(drop_last=False일 때). batch_size를 키우면 step 수는 줄고, 줄이면 늘어요.

🔧 전체 흐름: 학습 루프 한 장 명령어 사전에서 자세히 →

```
pred = model(X); loss = loss_fn(pred, y)          # ① 예측 ② 손실
optimizer.zero_grad(); loss.backward(); optimizer.step()   # ③ 초기화 ④ 기울기 ⑤ 갱신
```

### shape 맞추기 — dim에 적은 축이 사라진다

텐서를 다룰 땐 shape부터 확인해요. 2×3 텐서 [[1, 2, 3], [4, 5, 6]]로 보면:

| 계산 | 어느 방향으로 더하나 | 결과 | 남는 shape |
|---|---|---|---|
| sum(dim=0) | ↓ 세로로 (줄 축이 사라짐) | [5, 7, 9] | (3,) |
| sum(dim=1) | → 가로로 (칸 축이 사라짐) | [6, 15] | (2,) |

🤔 왜 예측 클래스는 argmax(dim=1)? — 로짓은 (64, 10) = "64장 × 클래스 10개 점수"예요. 사진마다 10개 점수 중 가장 큰 곳을 골라야 하니 클래스 축(dim=1)을 없애야 해요. 그러면 (64,) — 사진마다 답 하나가 남아요. dim=0으로 하면 거꾸로 "클래스마다 가장 높은 사진 번호"가 나와 엉뚱한 값이 돼요.

🔧 shape 확인·바꾸기 reshape · flatten(1) · sum(dim) · argmax(dim=1) 명령어 사전에서 자세히 →

```
x.flatten(1)            # (N, 1, 28, 28) → (N, 784)
pred.argmax(dim=1)      # (N, 10) → (N,)
```

### 데이터 · 모델 · 손실 · 갱신 · 평가 · 저장

| 단계 | 하는 일 | 🤔 왜 이렇게? |
|---|---|---|
| Dataset | "몇 개인지(__len__)"와 "i번째 샘플 1개(__getitem__)"만 알려 줌 | DataLoader가 이 두 가지만 알면 섞기·묶기를 알아서 해 줘요. 데이터 형식이 달라도 이 두 칸만 맞추면 돼요 |
| DataLoader | 64개씩 묶고, 학습 데이터는 epoch마다 섞음 | 한 장씩은 느리고 기울기가 들쭉날쭉, 전부 한 번에는 메모리가 부족 → 미니배치가 중간 |
| 모델 | Flatten → Linear(784, 512) → ReLU → Linear(512, 10) | 앞 층의 출력 수 = 다음 층의 입력 수. ReLU가 없으면 Linear를 여러 층 쌓아도 Linear 하나와 같아요 |
| 손실 | 분류는 CrossEntropyLoss(로짓 + 정수 정답), 회귀는 MSELoss | "얼마나 틀렸나"를 숫자 하나로 만들어야 그걸 줄이는 방향(기울기)을 구할 수 있어요 |
| 갱신 | SGD: p ← p − lr × p.grad / Adam: 가중치마다 보폭 조절 | Adam은 처음 시도하기 무난하고, SGD는 규칙이 단순해서 동작을 이해하기 좋아요(Day10) |
| 평가 | model.eval() + torch.no_grad()로 안 본 데이터 정확도 | 학습 손실은 내려가는데 평가 정확도가 떨어지면 과적합 → epoch 줄이기, Dropout 추가 |
| 저장 | state_dict()(가중치 묶음)만 저장하고, 불러올 땐 같은 구조의 모델에 넣음 | 모델 객체 통째보다 가볍고, 코드가 조금 바뀌어도 불러오기 쉬워요 |

🔧 직접 만드는 Dataset + DataLoader 명령어 사전에서 자세히 →

```
class MyDS(Dataset):  # __len__ · __getitem__ 두 개만
DataLoader(ds, batch_size=64, shuffle=True)
```

🔧 nn.Module로 모델 정의 명령어 사전에서 자세히 →

```
self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 512), nn.ReLU(), nn.Linear(512, 10))
def forward(self, x): return self.layers(x)
```

🔧 손실 · 최적화 CrossEntropyLoss · SGD · Adam 명령어 사전에서 자세히 →

```
loss_fn = nn.CrossEntropyLoss()                        # 로짓 그대로 넣기
optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

🔧 평가 함수 test() · 저장 state_dict 명령어 사전에서 자세히 →

```
with torch.no_grad(): hit = model(X).argmax(1) == y     # model.eval() 먼저
torch.save(model.state_dict(), "model.pth")
```

### 자주 하는 실수 4가지

| 실수 | 무슨 일이 생기나 | 🤔 왜? |
|---|---|---|
| zero_grad()를 빠뜨림 | 기울기가 배치마다 계속 쌓여 갱신이 점점 커짐 | backward()는 p.grad에 더하는 동작이라, 지우지 않으면 지난 배치 기울기가 남아요 |
| CrossEntropyLoss에 softmax 값을 넣음 | 손실이 잘 안 줄고 이상한 값이 나옴 | 이 손실은 안에서 softmax를 한 번 더 해요. 이미 0~1로 눌린 값에 또 softmax를 하니 차이가 뭉개져요 → 로짓 그대로 넣기 |
| 모델과 데이터의 장치가 다름 | RuntimeError: Expected all tensors to be on the same device | GPU에 있는 가중치와 CPU에 있는 데이터는 곱할 수 없어요 → 둘 다 .to(device) |
| 평가 때 eval()·no_grad()를 빠뜨림 | 점수가 실행할 때마다 흔들리고 메모리를 더 씀 | Dropout이 켜진 채라 매번 다른 칸을 끄고, 쓰지도 않을 기울기 기록까지 남겨요 |

실수 1·2·3을 실제로 저질러 본 결과는 명령어 사전의 "실제로 돌려 보면" 표에 있어요.

### 파라미터 두 종류

가중치(model.parameters())는 학습이 바꾸는 값, 하이퍼파라미터는 학습 전에 사람이 정하는 값이에요.

| 하이퍼파라미터 | 키우면 | 줄이면 |
|---|---|---|
| lr (학습률) | 빨리 움직이지만 발산 위험 | 안정적이지만 느림 |
| batch_size | step 수 ↓, 메모리 ↑ | step 수 ↑, 기울기가 들쭉날쭉 |
| epochs | 과적합 위험 ↑ | 덜 학습 |
| 은닉 크기 (예: 512) | 표현력 ↑, 과적합 ↑ | 단순한 패턴만 |

🤔 왜 "키우면 좋다"가 없어? — 하이퍼파라미터는 거의 다 맞바꾸기예요. 빠르면 불안정하고, 크면 과적합하기 쉬워요. 그래서 valid 점수를 보며 하나씩 바꿔 비교해요(Day10 학습률 비교와 같은 방식).

## ① 이미지 Tensor — [N, C, H, W]

| 축 | 뜻 | 예 |
|---|---|---|
| N | 이미지 장수 (batch) | 8 |
| C | 채널 (컬러 RGB = 3, 흑백 = 1) | 3 |
| H, W | 높이, 너비 (픽셀) | 64, 64 |

[그림: 컬러 도형 사진 한 장과, 그것을 R·G·B 채널 세 장으로 나눈 그림] 무엇을 보나: 왼쪽 컬러 사진을 R·G·B 세 장의 "숫자 판"으로 나눈 모습이에요. 빨간 사각형은 R 판에서만 진하고, 초록 원은 G 판, 파란 삼각형은 B 판에서 진해요. 흰 대각선은 세 판 모두에서 진해요(흰색 = R·G·B 모두 큼). 🤔 왜? 컴퓨터에게 컬러 사진은 원래 "같은 크기 숫자 판 3장"이에요. PyTorch는 이 3장을 맨 앞 축(C)에 쌓아 [3, H, W]로 들고, Conv의 Kernel 한 장도 이 3장을 한꺼번에 봐요. 화면에 그릴 땐 픽셀마다 (R,G,B)를 붙여야 하니 [H, W, 3]로 바꾸는 거예요.

⚠️ PyTorch는 채널이 앞(C, H, W)이고, PIL·NumPy·matplotlib은 채널이 뒤(H, W, C)예요. 화면에 그릴 땐 순서를 바꿔야 해요.

🤔 왜 N축이 항상 있어야 해? — 모델은 "여러 장을 한 번에" 계산하도록 만들어져 있어요. 한꺼번에 묶어야 GPU가 병렬로 빨리 계산하고, 기울기도 여러 장의 평균이라 덜 흔들려요. 그래서 사진 한 장도 [1, C, H, W]처럼 "1장짜리 묶음"으로 넣어요.

🔧 축 순서 바꾸기 · 개수 축 붙이기 명령어 사전에서 자세히 →

```
hwc  = img.permute(1, 2, 0)           # [32, 32, 3]  → plt.imshow(hwc)
nchw = img.unsqueeze(0)               # [1, 3, 32, 32] → 사진 한 장을 모델에 넣을 때
```

## ② 합성곱 (Convolution) — 작은 창을 옮기며 곱해 더하기

| 용어 | 뜻 |
|---|---|
| 합성곱 | 작은 창을 옮겨 가며 창 안의 값을 가중합하는 연산. 모든 위치에서 같은 가중치 |
| Kernel (필터) | 합성곱 창 안의 가중치 묶음. 3×3이면 숫자 9개. 학습으로 정해지는 파라미터 |
| Feature Map | Kernel 하나가 사진 전체를 훑어서 만든 결과 지도 ("특징 추출부"의 그 특징) |
| 채널 | Feature Map 한 장 = 채널 하나. 출력 채널 수 = Kernel 개수. RGB 입력이면 Kernel 하나가 3채널을 모두 봄 |
| Stride | 창이 한 번에 건너뛰는 칸 수. 2면 출력이 약 절반 |
| Padding | 가장자리 바깥에 0을 덧대기. 3×3에 padding 1이면 크기 유지 |
| 수용영역 | 출력 한 칸이 원본에서 보는 범위. 층을 쌓고 Pooling을 거칠수록 넓어짐 |

🤔 왜 "작은 창"으로 봐? — 사진에서 의미 있는 패턴(선·모서리·점)은 가까운 픽셀끼리 만들어요. 멀리 떨어진 픽셀까지 한 번에 볼 필요가 없으니 작은 창으로 충분하고, 넓은 범위는 층을 쌓아서(수용영역 확장) 나중에 봐요. 그리고 같은 창을 모든 위치에 쓰니까 사진을 옮기면 결과도 같이 옮겨져요(등변) → 물체가 어디 있든 같은 검출기로 찾아요.

[그림: 6×6 입력 위의 3×3 창, 세로선 Kernel, 4×4 출력 Feature Map] 무엇을 보나: 분홍 창(3×3) 안의 숫자와 가운데 Kernel의 숫자를 같은 자리끼리 곱해서 모두 더하면 1+1+1 = 3 → 출력의 분홍 칸이 돼요. 창을 오른쪽·아래로 한 칸씩 옮기며 같은 계산을 반복해서 4×4 Feature Map을 채워요. 🤔 왜 세로선 위에서만 3이야? Kernel이 가운데 세로줄에만 1을 갖고 있으니, 창 가운데 세로줄이 입력의 세로선(1,1,1)과 겹칠 때만 크게 나와요. 오른쪽 아래의 외딴 점 1개는 1만 나와요 → "세로선 모양"에만 강하게 반응하는 검출기예요.

아래에서 직접 해 보세요. 왼쪽 입력 칸을 클릭하면 0↔1이 바뀌고, 오른쪽 출력 칸에 마우스를 올리면 그 칸을 만든 창과 계산이 보여요.

💡 기본 설정(세로선 Kernel, padding 0, stride 1)에서는 세로선 위에서만 1+1+1 = 3, 나머지는 0이 나와요(개념지도 예시와 같은 계산). 이 Kernel은 세로선에 반응하는 검출기예요. 학습된 CNN은 이런 검출기를 데이터에서 스스로 찾아요.

[그림: 흑백 도형 사진에 세로 경계, 가로 경계, 흐리기, 선명하게 Kernel을 적용한 4개의 Feature Map과 각 Kernel 숫자] 무엇을 보나: 같은 사진에 Kernel 숫자만 바꿔 진짜로 합성곱(F.conv2d, padding 1)을 한 결과예요. 세로 경계 Kernel은 사각형의 왼쪽·오른쪽 변만, 가로 경계 Kernel은 위·아래 변만 빨강/파랑으로 켜요. 원은 둘 다에서 일부만 켜지고, 흐리기는 경계를 뭉개고, 선명하게는 테두리를 더 밝게 강조해요. 🤔 왜? 세로 경계 Kernel은 "오른쪽 열 − 왼쪽 열"이라 좌우 밝기 차이가 있는 곳(세로 변)에서만 0이 아니에요. 평평한 면은 양쪽이 같아 0이 되죠. 즉 Kernel의 숫자 배치 = "어떤 모양을 찾을지". CNN은 이 숫자 9개를 학습으로 정하고, 출력 채널 수만큼 이런 검출기를 동시에 가져요.

### 공식 세 개

```
출력 크기   H_out = ⌊(H + 2P − D(K−1) − 1) / S⌋ + 1    (K=Kernel, P=padding, S=stride, D=dilation)
            D = 1(기본)이면  ⌊(H + 2P − K) / S⌋ + 1
파라미터 수 = 출력채널 × 입력채널 × K × K + 출력채널(bias)
수용영역    r=1, j=1 에서 시작 → 층마다  r += (K−1)·j,  j *= S
```

| 예 | 결과 |
|---|---|
| 32, K3 P1 S1 | 32 (크기 유지) |
| 32, K3 P1 S2 | 16 (절반) |
| 32, K3 P0 S1 | 30 (조금 줄어듦) |
| 수용영역: Conv3 → Pool2 → Conv3 → Pool2 | 3 → 4 → 8 → 10칸, Global Pool에서 전체 |

[그림: 5×5 입력에서 padding과 stride에 따라 3×3 창 가운데가 설 수 있는 자리를 점으로 표시한 세 격자] 무엇을 보나: 분홍 점 하나 = 창 가운데가 설 수 있는 자리 = 출력 한 칸이에요. 점 개수를 세면 출력 크기가 나와요. 왼쪽(padding 0)은 가장자리 줄에 점이 없어 5→3, 가운데(빗금 0을 한 겹 덧댐)는 가장자리에도 점이 생겨 5→5, 오른쪽(stride 2)은 점이 한 칸씩 건너뛰어 3×3. 🤔 왜? 3×3 창은 가운데에서 한 칸씩 삐져나오니, 덧대지 않으면 맨 가장자리 칸에는 가운데를 올려놓을 수 없어요. 0을 덧대면 올려놓을 수 있게 되고, stride는 "다음 자리까지 몇 칸 뛰나"라 점 사이 간격이 돼요. 공식 ⌊(H + 2P − K)/S⌋ + 1은 이 점 개수를 세는 식이에요.

[그림: 출력 한 칸이 원본에서 보는 영역이 1, 3, 4, 8, 10칸으로 넓어지는 겹친 사각형과 막대그래프] 무엇을 보나: 왼쪽의 겹친 사각형은 "마지막 층의 한 칸이 원본 사진에서 보는 범위"예요. 층을 지날 때마다 1 → 3 → 4 → 8 → 10칸으로 커져요. 🤔 왜 Pool 뒤에 확 커져? Pool(stride 2)을 지나면 Feature Map의 한 칸이 원본 2칸을 대표해요. 그 위의 3×3 Conv는 "옆 칸"으로 한 칸 움직일 때 원본에선 2칸씩 움직이는 셈이라, 같은 3×3인데도 4칸이나 넓어져요(r += (K−1)·j, j=2). 그래서 앞쪽 층은 선·점 같은 작은 패턴, 뒤쪽 층은 모서리·부품 같은 큰 패턴을 봐요.

🤔 왜 Padding을 해? — 덧대지 않으면 3×3 창의 가운데가 가장자리 칸 위에 못 서서, 층마다 출력이 2칸씩 줄고 가장자리 정보가 덜 쓰여요. 0을 한 겹 덧대면 크기가 유지돼서 층을 많이 쌓아도 사진이 사라지지 않아요.

🤔 왜 Kernel은 보통 홀수(3)야? — 홀수여야 창에 가운데 칸이 있어서 "출력 한 칸 = 입력 한 칸 주변"으로 위치가 딱 맞아요. 3×3을 두 번 쌓으면 5×5와 같은 범위를 보면서 파라미터는 더 적고(9+9 < 25) 비선형도 한 번 더 들어가요.

🤔 왜 stride 2를 써? — 칸을 건너뛰면 출력 칸 수가 가로·세로 각각 절반 → 계산량이 약 1/4로 줄어요. 필터는 그대로라 파라미터 수는 같아요. 대신 촘촘히 안 보니 작은 디테일은 놓칠 수 있어요.

🔧 nn.Conv2d 명령어 사전에서 자세히 →

```
conv = nn.Conv2d(3, 8, 3, padding=1)
print(conv(x).shape)                          # torch.Size([4, 8, 32, 32])
```

## ③ Pooling · BatchNorm · 작은 CNN 조립

Pooling은 작은 영역을 대표값 하나로 줄여요. MaxPool2d(2)는 2×2에서 제일 큰 값 하나 → 크기가 절반. 채널마다 따로 하니까 채널 수는 그대로예요.

```
[1 4]
[2 3]  → MaxPool 2×2 →  4
```

[그림: 4×4 숫자 격자를 색 블록별 최댓값으로 2×2로 줄인 그림과, 96×96 세로 경계 Feature Map을 48×48로 줄인 결과] 무엇을 보나: 왼쪽: 같은 색 2×2 블록마다 가장 큰 값(진한 칸) 하나만 살아남아 4×4 → 2×2. 오른쪽: 위에서 만든 진짜 세로 경계 Feature Map(96×96)을 MaxPool2d(2)로 줄인 것(48×48). 칸은 1/4로 줄었는데 사각형의 두 세로 변, 원의 좌우, 대각선이 그대로 알아볼 수 있어요. 🤔 왜 모양이 안 망가져? Feature Map에서 큰 값 = "여기 그 패턴이 있다"는 신호예요. 2×2 안에서 제일 강한 신호만 남기면 "이 근처에 경계가 있다"는 사실은 보존되고, 정확히 몇 번째 픽셀인지만 흐려져요. 그래서 계산은 줄고, 물체가 1픽셀 움직여도 결과가 덜 흔들려요.

🤔 왜 Pooling을 해? — ① 크기가 절반이 되니 뒤쪽 계산이 줄어요. ② 다음 Conv의 한 칸이 원본의 더 넓은 범위를 보게 돼요(수용영역 3 → 4 → 8 → 10). ③ 2×2 안에서 최댓값 위치가 살짝 바뀌어도 결과가 같아서, 물체가 조금 움직여도 흔들리지 않아요(작은 이동에 불변). 최댓값을 쓰는 건 "이 근처에 그 패턴이 있긴 하다"는 가장 강한 신호만 남기려는 거예요.

Conv 블록 하나 = Conv(패턴 찾기) → (BatchNorm) → ReLU(음수 지우기) → MaxPool(크기 절반). 블록을 반복해요.

🤔 왜 ReLU가 꼭 있어야 해? — Conv도 결국 "곱해서 더하기"(선형)라서, 비선형 없이 몇 층을 쌓아도 선형 하나와 똑같아요. ReLU가 사이에 끼어야 층마다 새로운 걸 배우고, 블록을 반복하는 의미가 생겨요.

사진 [N,3,64,64]→ Conv 블록 × 2 특징 추출부(Backbone)→ Global Avg Pool [N,32,1,1]→ flatten [N,32]→ Head Linear [N,K] 점수(logit)

공간은 64 → 16 → 1, 채널은 3 → 32. "어디에 있나"를 점점 버리고 "무엇이 있나"로 바꾸는 과정이에요. 큰 모델(VGG·ResNet)도 같은 모양이에요.

[그림: 수업 예제 데이터의 세로 줄무늬·가로 줄무늬 사진에 4개 Kernel을 적용한 채널별 Feature Map] 무엇을 보나: 수업 예제 데이터(vertical, horizontal 폴더) 사진 두 장을 Conv(1→4채널) + ReLU에 넣은 결과예요. 줄마다 오른쪽 4장이 채널 4개 = Feature Map 4장. 세로 줄무늬 사진은 ch0(세로 경계)이 밝고(평균 0.27), 가로 줄무늬 사진은 ch1(가로 경계)이 밝아요(0.21). ※ Kernel은 이해를 돕기 위해 손으로 정한 값이고 학습된 게 아니에요. 🤔 왜 이게 분류로 이어져? 마지막에 Global Avg Pool이 채널마다 평균 하나만 남기면 [ch0 0.27, ch1 0.05, …] 같은 짧은 숫자 줄이 돼요. 이 숫자만 봐도 세로/가로를 가를 수 있죠. 학습은 이런 "잘 갈라 주는 Kernel"을 스스로 찾는 과정이에요.

[그림: SmallCNN 각 단계에서 공간 크기는 64에서 1로 줄고 채널 수는 3에서 32로 느는 막대그래프] 무엇을 보나: 아래 SmallCNN 표의 shape를 막대로 그린 거예요. 보라(공간 H)는 Pool마다 절반 → GAP에서 1, 주황(채널 C)은 Conv마다 늘어 32. 🤔 왜 채널을 늘려? 공간을 줄이면 정보가 줄어드니, 그만큼 "찾는 패턴의 종류"(채널)를 늘려 보충해요. 뒤쪽 층은 앞쪽 패턴을 조합한 더 복잡한 패턴(모서리 → 부품)을 보니 종류도 더 많이 필요해요.

🔧 SmallCNN 만들기 명령어 사전에서 자세히 →

```
nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
self.pool = nn.AdaptiveAvgPool2d(1)        # Global Avg Pool
```

🔧 nn.BatchNorm2d (ResNet18엔 있고, SmallCNN엔 없음) 명령어 사전에서 자세히 →

```
nn.Conv2d(3, 16, 3, padding=1, bias=False),
nn.BatchNorm2d(16),            # 채널 수와 같아야 함
```

[그림: BatchNorm 전에는 세 채널 값 분포가 서로 다른 위치와 폭이고, 후에는 모두 0 중심 폭 1로 겹치는 히스토그램] 무엇을 보나: 일부러 채널마다 크기를 다르게 만든 Conv 출력(왼쪽)은 채널 1이 -4 근처에 넓게, 채널 0은 +3 근처에, 채널 2는 0.5 근처에 좁게 몰려 있어요. BatchNorm2d(3)(train 모드)를 지나면(오른쪽) 세 채널이 모두 평균 0 · 표준편차 1로 거의 겹쳐요. 🤔 왜 좋아? 다음 층 입장에선 채널마다 "숫자 단위"가 제각각이면 어떤 가중치는 크게, 어떤 건 작게 움직여야 해서 하나의 lr로 맞추기 어려워요. 단위를 맞춰 주면 같은 lr로 모두 고르게 배울 수 있어요. (실제 BN은 이 뒤에 채널별로 배우는 크기 γ·위치 β를 다시 곱하고 더해요.)

## ④ 이미지 읽기 · 증강 · 학습 · 평가 · 저장

### ImageFolder — 폴더 이름이 곧 라벨

```
data/
├─ train/
│  ├─ cat/   ← 고양이 사진들 → 라벨 0
│  └─ dog/   ← 강아지 사진들 → 라벨 1
├─ valid/ (같은 구조)
└─ test/  (같은 구조, 모든 선택이 끝난 뒤 한 번만)
```

하위 폴더 이름을 정렬해서 0, 1, 2…를 붙여요. 결과는 class_to_idx로 꼭 확인해요.

[그림: ] 무엇을 보나: 수업 예제 데이터 폴더(horizontal, vertical)가 그대로 라벨이 되는 흐름이에요. 이름을 알파벳 순으로 정렬해 horizontal = 0, vertical = 1. 사진 한 장은 transform을 거쳐 Tensor가 되고 그 폴더 번호와 짝지어져요. 🤔 왜 정렬해? 컴퓨터가 폴더를 읽는 순서는 운영체제마다 다를 수 있어요. 정렬해 두면 어느 컴퓨터에서든 같은 번호가 붙어요. 대신 폴더 이름을 바꾸면 번호도 바뀌니 class_to_idx를 꼭 같이 저장해요.

🤔 왜 train / valid / test 셋으로 나눠? — train은 배우기, valid는 모델·epoch을 고르기, test는 최종 확인용이에요. valid로 여러 번 고르다 보면 valid에도 맞춰지니까, 한 번도 고르는 데 안 쓴 test가 있어야 "새 사진에서도 될까?"를 정직하게 알 수 있어요.

🔧 ImageFolder + transforms + DataLoader 명령어 사전에서 자세히 →

```
train_ds = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
```

### 증강(augmentation)은 train에만

뒤집기·살짝 회전·밝기 변화처럼 라벨이 안 바뀌는 변환으로 데이터를 늘려서 과적합을 줄여요.

🔧 증강 transforms.Compose (train만) 명령어 사전에서 자세히 →

```
train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
```

🤔 왜 증강이 과적합을 줄여? — 모델이 "이 사진의 이 픽셀 배치"를 외우지 못하게, 매 epoch 조금씩 다른 모습을 보여 줘요. 그러면 뒤집혀도·밝기가 달라도 같은 정답인 본질적인 특징을 배우게 돼요. 🤔 왜 valid엔 안 해? — valid는 모델을 고르는 잣대라서 매번 같은 결과가 나와야 해요. 무작위 변환을 넣으면 같은 모델도 실행마다 점수가 흔들려 공정하게 비교할 수 없어요.

⚠️ 라벨을 바꾸는 증강은 금지. 예를 들어 가로/세로 줄무늬를 구분하는 문제라면 90도 회전은 가로를 세로로 만들어 정답을 틀리게 해요. 숫자 6/9 구분에 180도 회전도 마찬가지예요. "이 변환을 해도 정답이 같은가?"를 먼저 물어봐요.

[그림: 도형 사진에 좌우 반전, 10도 회전, 잘라 확대, 밝기 변화를 적용한 6장과, 세로 줄무늬 사진을 90도 돌려 가로 줄무늬가 된 예] 무엇을 보나: 초록 테두리 6장은 한 장의 사진에 torchvision 증강을 하나씩 적용한 모습이에요(뒤집기 · 10도 회전 · 잘라서 확대 · 어둡게/밝게). 모양은 달라졌지만 "사각형·원·삼각형이 있는 사진"이라는 정답은 그대로예요. 아래 오른쪽 빨간 테두리는 수업 예제 데이터의 vertical 사진을 90도 돌린 것 → 가로 줄무늬가 돼서 정답이 바뀌어요. 🤔 왜 이게 과적합을 줄여? 모델 입장에선 매 epoch 같은 사진이 조금씩 다르게 보이니 "왼쪽 위 픽셀이 빨갛다" 같은 우연한 배치를 외울 수 없고, 어떤 변형에도 남는 특징(모양·질감)을 써야만 맞힐 수 있어요.

🔧 손실 · 학습 루프 · valid 평가 · best 저장 명령어 사전에서 자세히 →

```
loss_fn = nn.CrossEntropyLoss()                 # softmax 전 logit을 그대로 넣음
best_loss, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
```

[그림: 정답 확률 p에 대한 -log p 곡선. p=0.9에서 0.11, 0.5에서 0.69, 0.01에서 4.61] 무엇을 보나: 가로축은 모델이 정답 클래스에 준 확률, 세로축은 CrossEntropy 손실(-log p)이에요. p = 0.9면 0.11, 0.5면 0.69인데 0.01이면 4.61로 치솟아요. 🤔 왜 왼쪽에서 폭증해? log는 0에 가까워질수록 끝없이 작아지는 함수라 -log는 끝없이 커져요. 그래서 "정답을 거의 0%로 본" 사진 한 장이 손실을 크게 끌어올리고, 그 사진 쪽으로 큰 기울기가 생겨 먼저 고쳐져요. 반대로 이미 0.9로 맞힌 사진은 거의 손대지 않아요.

🔧 저장할 땐 "읽는 법"까지 함께 명령어 사전에서 자세히 →

```
torch.save({"state_dict": best_state, "class_to_idx": train_ds.class_to_idx, ...}, "best_model.pt")
ckpt = torch.load("best_model.pt", weights_only=True)
```

## ⑤ 전이학습 — 이미 공부한 뇌 빌려 쓰기

데이터가 많은 관련 작업(ImageNet, 1000클래스)으로 미리 학습한 모델에서 마지막 층(Head)을 떼고 내 클래스 수 K에 맞는 새 층을 붙여요. 본체(Backbone)를 고정하고 새 층만 배우거나, 일부·전체를 작은 lr로 미세조정해요.

🤔 왜 전이학습이 잘 통해? — 앞쪽 층이 배우는 색·질감·모서리 같은 특징은 고양이든 불량품이든 거의 모든 사진에 공통이에요. 내 데이터가 적어도, 파라미터 대부분을 "이미 좋은 출발점"에서 시작하니까 처음부터(무작위) 배우는 것보다 적은 데이터로 빨리, 덜 과적합되게 배워요.

[그림: ] 무엇을 보나: 위(①)는 남이 ImageNet 1000클래스로 이미 학습해 둔 ResNet18, 아래(②)는 내 문제예요. 보라색 Backbone을 그대로 빌려 와 잠그고(회색), 1000클래스용 마지막 fc만 잘라 내 클래스 수 K짜리 새 fc(분홍)로 바꿔요. 🤔 왜 Backbone은 그대로 써도 돼? Backbone이 하는 일은 "사진에서 쓸 만한 특징 512개를 뽑기"이고, 이 능력은 개·고양이를 가르든 불량품을 가르든 공통이에요. 바뀌어야 하는 건 "그 512개로 어떤 클래스를 고를지"뿐이라 마지막 fc만 새로 배우면 돼요.

| ResNet18 구간 (224×224 입력) | 출력 | 처음부터 | Head-only | Partial | 전체 미세조정 |
|---|---|---|---|---|---|
| conv1 + pool | 64ch · 56×56 | 무작위 | 고정 | 고정 | 학습 (작은 lr) |
| layer1 | 64ch · 56×56 | 무작위 | 고정 | 고정 | 학습 |
| layer2 | 128ch · 28×28 | 무작위 | 고정 | 고정 | 학습 |
| layer3 | 256ch · 14×14 | 무작위 | 고정 | 고정 | 학습 |
| layer4 | 512ch · 7×7 | 무작위 | 고정 | 학습 lr 1e-4 | 학습 |
| fc (Head) | 512 → K | 무작위 | 학습 lr 1e-3 | 학습 lr 1e-3 | 학습 |

| 방법 (K=2) | 학습하는 파라미터 수 |
|---|---|
| Head-only | 512×2 + 2 = 1,026 |
| Partial (layer4 + Head) | 8,393,728 + 1,026 = 8,394,754 |
| 전체 미세조정 | 11,177,538 |
| (비교) SmallCNN 처음부터 | 5,154 |

[그림: Head-only, Partial, 전체 미세조정에서 ResNet18 각 구간이 고정인지 학습인지 색으로 표시하고, 구간별 파라미터 수를 로그 막대로 비교] 무엇을 보나: 왼쪽은 세 방법에서 ResNet18의 어느 구간을 열고(보라·분홍) 잠그는지(회색), 오른쪽은 구간별 파라미터 수예요(resnet18(weights=None)로 구조만 만들어 직접 센 값, 위 표와 같아요). Head-only는 분홍 칸 하나(1,026개)뿐, Partial은 layer4를 열자마자 839만 개로 뛰어요. 🤔 왜 layer4 하나가 이렇게 커? 파라미터 수는 입력채널 × 출력채널 × 3 × 3이라 채널 수의 제곱으로 커져요. layer4는 512채널이라 64채널인 layer1보다 수십 배 많아요(가로축이 로그 눈금이라 막대 한 칸 = 10배). 그래서 "조금만 더 열어 볼까"도 학습량·과적합 위험이 확 늘어요 → 작은 lr이 필요한 이유.

🤔 왜 처음엔 Backbone을 얼려(Freeze)? — 새로 붙인 Head는 무작위라 처음 몇 step 동안 기울기가 크고 엉뚱해요. 그때 Backbone까지 열어 두면 그 엉뚱한 기울기가 잘 배운 사전학습 값을 망가뜨려요. 먼저 Head만 맞춰 두고, 필요할 때 더 여는 게 안전해요. 학습할 파라미터가 1,026개뿐이라 데이터가 적어도 과적합 위험이 작고 빠르기도 해요.

🤔 왜 열 때는 뒤쪽(layer4)부터, 그리고 작은 lr로? — 앞쪽 층은 색·질감처럼 일반적인 특징이라 그대로 써도 되고, 뒤쪽 층은 "ImageNet의 물체 종류"와 더 관련돼서 내 문제에 맞게 고칠 필요가 커요. 그리고 열어 준 층은 이미 좋은 값이라 살짝만 고쳐야 하니 lr을 작게(1e-4), 새 Head는 처음부터 배워야 하니 크게(1e-3) 줘요.

🔧 ResNet18 Head-only 전이학습 명령어 사전에서 자세히 →

```
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)   # 2) Head 교체 (고정 코드 뒤에!)
```

🔧 Partial 미세조정 (layer4 + Head, lr 다르게) 명령어 사전에서 자세히 →

```
{"params": model.layer4.parameters(), "lr": 1e-4},   # 열어 준 층은 작게
{"params": model.fc.parameters(),     "lr": 1e-3},   # 새 Head는 크게
```

## ⑥ CPU·GPU 속도 비교 · 성능표 — 공정하게 재기

모델은 그대로, 계산 장소만 바꿔서 비교해요. 데이터 · batch · 모델 · epoch를 똑같이 고정해요.

| 규칙 | 🤔 왜? |
|---|---|
| 워밍업 후에 재기 | 첫 실행엔 초기화·메모리 준비 시간이 섞여서 유난히 느려요 |
| GPU는 torch.cuda.synchronize() 전후로 | GPU는 비동기라 명령만 던지고 파이썬은 바로 다음 줄로 가요 → 끝날 때까지 기다려야 진짜 시간 |
| 처리량 = 이미지 수 ÷ 초 (images/s) | batch·데이터 크기가 달라도 비교 가능 |
| Peak VRAM도 기록 | 빨라도 메모리가 모자라면 못 돌려요 |
| GPU가 없으면 N/A + 사유 | 안 잰 걸 0으로 채우면 "0초 = 엄청 빠름"으로 오해돼요 |

🔧 한 epoch 시간 재기 명령어 사전에서 자세히 →

```
torch.cuda.synchronize()
seconds = time.perf_counter() - start
```

[그림: SmallCNN 학습 step 처리량을 batch size 1부터 256까지 CPU와 GPU로 실측한 로그 그래프] 무엇을 보나: 이 노트를 만든 노트북(RTX 5080 Laptop)에서 SmallCNN 학습 step을 batch size만 바꿔 직접 잰 처리량이에요(워밍업 후, synchronize 포함). batch 1에서는 CPU와 GPU가 거의 같지만(약 800 images/s), batch 256에선 GPU가 CPU보다 10배 이상 빨라요. CPU는 batch 16쯤에서 이미 한계에 닿아요. 🤔 왜? GPU는 수천 개의 작은 계산기를 동시에 돌리는 장치라, 일감이 적으면 대부분 놀고 "명령 띄우기·데이터 옮기기" 고정 비용만 남아요. batch가 커져야 계산기들이 다 일을 해서 진짜 속도가 나와요. 숫자는 장치마다 다르니 내 환경에서 다시 재야 해요.

## 증상으로 진단

| 이런 일이 생기면 | 의심할 것 |
|---|---|
| Conv2d에서 채널 오류 (expected 3 channels …) | 입력이 [N,H,W,C] 순서이거나 흑백(1채널). permute·convert("RGB") 확인 |
| 모델에 사진 한 장 넣었더니 차원 오류 | batch 축 없음 → unsqueeze(0) |
| DataLoader가 PIL 이미지를 못 묶음 | ImageFolder에 transform(ToTensor) 안 줌 |
| 라벨 번호가 예상과 다름 | 폴더 이름 정렬 순서 → class_to_idx 확인 |
| valid 정확도가 실행마다 흔들림 | valid에 무작위 증강을 넣음, model.eval() 누락 |
| 전이학습 성능이 이상하게 낮음 | weights.transforms() 안 씀, Backbone BN이 train 모드, Head를 고정 코드보다 먼저 교체 |
| 학습 파라미터 수가 1,026보다 훨씬 큼 | requires_grad = False가 안 됨 |
| GPU 시간이 비정상적으로 짧음 | torch.cuda.synchronize() 없이 잼 |

## 원본 자료

- ⚠️ Day11 강의 본편 PDF는 받지 못했어요. 이 페이지는 Day11_개념지도.pdf(1쪽, 강의 PDF 쪽수 표기 포함)를 기준으로, 그 범위에 맞는 표준 CNN 이론으로 정리했어요. 본편 PDF가 생기면 다시 대조할 예정이에요.

- 개념지도의 개념 정의 출처: Prince, Understanding Deep Learning (MIT Press, 2023) 10장(합성곱)·9.3(전이학습·증강)·11.4(BatchNorm)

- ⓪ PyTorch 한 장 정리: PyTorch_치트시트.pdf(1쪽, PyTorch 한국어 튜토리얼 「파이토치 기본 익히기」 기준, 2026-11-02). 그림·사전의 숫자는 FashionMNIST와 같은 모양의 합성 데이터로 직접 실행한 값이에요(데이터 내려받기 없음).

- 참고: 같은 날짜의 20261102_2.pdf(104쪽)는 Day12(순서 모델) 강의 자료라 Day12에 반영돼 있어요.

- 공식 문서: nn.Conv2d · nn.BatchNorm2d · ImageFolder · ResNet18 · 가중치 · 전처리 · 전이학습 튜토리얼

## 🔧 Day11 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day11)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | torch, torchvision, pillow |
| ⓪ PyTorch 한 장 정리 | Dataset · DataLoader · nn.Module · CrossEntropyLoss · SGD/Adam · eval()+no_grad() · state_dict |
| ① 이미지 Tensor [N, C, H, W] | torch.rand(), permute(), unsqueeze(0) |
| ② 합성곱 Conv2d | nn.Conv2d(in, out, kernel_size, stride=, padding=), F.conv2d() |
| ③ Pooling · BatchNorm · CNN 조립 | nn.MaxPool2d(), nn.BatchNorm2d(), nn.AdaptiveAvgPool2d(1), torch.flatten(x, 1) |
| ④ 이미지 읽기 · 증강 · 학습 · 평가 · 저장 | datasets.ImageFolder(), transforms.Compose(), nn.CrossEntropyLoss(), torch.inference_mode(), torch.save() |
| ⑤ 전이학습 (Head-only · Partial) | resnet18(weights=), weights.transforms(), requires_grad = False, model.fc = nn.Linear() |
| ⑥ CPU·GPU 속도 비교 · 성능표 | time.perf_counter(), torch.cuda.synchronize(), torch.cuda.max_memory_allocated() |

### 전체 흐름: 학습 루프 한 장 (FashionMNIST)

```
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()     # Apple Silicon GPU
          else "cpu")
lr, batch_size, epochs = 1e-3, 64, 5                          # 사람이 정함 (하이퍼파라미터)

tf = transforms.ToTensor()
train_ds = datasets.FashionMNIST("data", train=True, download=True, transform=tf)
train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
model = Net().to(device)                                      # 아래 nn.Module 항목
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

for epoch in range(epochs):            # 에폭 반복
    model.train()                      # 학습 모드
    for X, y in train_dl:              # 배치 반복
        X, y = X.to(device), y.to(device)
        pred = model(X)                # ① 예측
        loss = loss_fn(pred, y)        # ② 손실
        optimizer.zero_grad()          # ③ 초기화
        loss.backward()                # ④ 기울기
        optimizer.step()               # ⑤ 갱신
    test(model)                        # 평가 (아래 test 항목)
torch.save(model.state_dict(), "model.pth")
```

▶ 실제로 돌려 보면 FashionMNIST와 같은 모양의 합성 데이터 (학습 1,280 · 평가 256장, 클래스마다 밝은 띠 위치가 다름) · lr 0.1

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| device | cuda | 이 PC는 NVIDIA GPU가 있어서 cuda (PC마다 달라요) |
| len(DataLoader(60,000개, batch_size=64)) | 938 | FashionMNIST 한 epoch = 938번 step |
| len(train_dl) (1,280장) | 20 | 1,280 ÷ 64 = 20 묶음 |
| 학습 전 test(model) | 0.004 | 학습 전이라 찍기 수준 (이번 무작위 시작값에선 기대값 0.1보다도 낮게 나왔어요) |
| epoch 1·2·3 (학습 손실, 평가 정확도) | [(1.407, 1.0), (0.243, 1.0), (0.077, 1.0)] | 손실은 계속 줄고, 정확도는 첫 epoch부터 1.0 — 규칙이 쉬운 합성 데이터라서예요(실제 FashionMNIST는 이렇게 안 돼요) |

| 줄 | 뜻 | 🤔 왜? |
|---|---|---|
| device = ("cuda" if ... else "mps" if ... else "cpu") | NVIDIA GPU → Apple GPU → CPU 순서로 고름 | 같은 코드를 어느 컴퓨터에서 돌려도 쓸 수 있는 장치를 자동으로 고르려고 |
| datasets.FashionMNIST(..., download=True) | 옷 사진 데이터를 data/에 받아 Dataset으로 | 공식 튜토리얼의 연습용 데이터. 처음 한 번만 내려받아요 |
| model.train() | Dropout·BatchNorm을 학습 동작으로 | test() 안에서 eval()로 바꿨으니 epoch마다 다시 켜야 해요 |
| X.to(device), y.to(device) | 배치를 모델과 같은 장치로 | 모델은 한 번만 옮기면 되지만 배치는 새로 꺼낼 때마다 옮겨야 해요 |
| ③ → ④ → ⑤ 순서 | 지우고 → 계산하고 → 갱신 | 순서를 바꾸면 지난 기울기가 섞이거나 기울기 없이 갱신해요 (Day9) |

⚠️ download=True는 인터넷으로 데이터를 내려받아요(약 30MB). 이미 받은 폴더가 있으면 다시 받지 않아요.

### shape 확인·바꾸기 reshape · flatten(1) · sum(dim) · argmax(dim=1)

```
x.shape                       # 가장 먼저 확인
x.to(device)                  # 모델과 같은 장치로
x.reshape(64, -1)             # -1 자리는 자동 계산
x.flatten(1)                  # (N, 1, 28, 28) → (N, 784)  1번 축부터 펴기
t = torch.tensor([[1, 2, 3], [4, 5, 6]])
t.sum(dim=0)                  # [5, 7, 9]  세로로 더함
t.sum(dim=1)                  # [6, 15]    가로로 더함
pred.argmax(dim=1)            # (N, 10) → (N,)  사진마다 점수가 가장 큰 클래스
```

▶ 실제로 돌려 보면 무작위 배치 (64, 1, 28, 28) · 2×3 텐서

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| x.reshape(64, -1).shape | (64, 784) | -1 자리 = 1×28×28 = 784 자동 계산 |
| x.flatten(1).shape | (64, 784) | 배치 축은 두고 나머지만 펴기 |
| x.flatten().shape | (50176,) | 0번부터 펴면 배치까지 한 줄로 — 보통 실수 |
| t.sum(dim=0) | [5, 7, 9] | 줄 축(0)이 사라짐 |
| t.sum(dim=1) | [6, 15] | 칸 축(1)이 사라짐 |
| t.sum(dim=1, keepdim=True).shape | (2, 1) | 축을 길이 1로 남김 |
| pred.argmax(dim=1).shape, [:5] | (64,), [7, 0, 7, 3, 5] | 사진마다 클래스 번호 하나 |
| x.reshape(100, -1) | shape '[100, -1]' is invalid for input of size 50176 | 50,176개를 100줄로 못 나눠요 |

| 표현 | 뜻 | 바꾸면? |
|---|---|---|
| reshape(64, -1) | 원소 수는 그대로, 모양만 바꿈. -1은 "나머지로 알아서" | -1은 한 곳에만 쓸 수 있어요. 원소 수가 안 나눠떨어지면 에러 |
| flatten(1) | 1번 축부터 끝까지를 한 줄로 | flatten()(0부터)이면 배치 축까지 펴져서 (N×784,) 한 줄이 돼요 |
| sum(dim=k) | k번 축을 따라 더하고 그 축은 사라짐 | keepdim=True면 축을 길이 1로 남겨요 |
| argmax(dim=1) | 1번 축에서 가장 큰 값의 위치 | 값이 아니라 번호(클래스 id)를 돌려줘요 |

### 직접 만드는 Dataset + DataLoader

```
from torch.utils.data import Dataset, DataLoader

class MyDS(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):            # 샘플 수
        return len(self.X)
    def __getitem__(self, i):     # 샘플 1개
        return self.X[i], self.y[i]

train_dl = DataLoader(MyDS(X, y), batch_size=64, shuffle=True)   # 64개씩, epoch마다 섞기
X, y = next(iter(train_dl))       # 배치 하나 꺼내 shape 확인
```

▶ 실제로 돌려 보면 합성 사진 150장

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(ds) | 150 | __len__ 이 돌려준 값 |
| ds[0][0].shape, ds[0][1] | (1, 28, 28), 8 | __getitem__(0): 샘플 1개 |
| X, y = next(iter(dl)) → shape | (64, 1, 28, 28), (64,) | 64개가 앞에 붙은 배치 |
| len(dl), 배치 크기들 | 3, [64, 64, 22] | 150 = 64 + 64 + 22 → 마지막은 남는 만큼 |
| y.dtype | torch.int64 | CrossEntropyLoss가 원하는 정수(long) |

| 부분 | 뜻 | 🤔 왜? |
|---|---|---|
| __len__ | 전체 샘플 수 | DataLoader가 몇 묶음을 만들지, 언제 epoch가 끝나는지 알려고 |
| __getitem__(i) | i번째 샘플 하나를 (입력, 정답)으로 | DataLoader가 번호만 골라 부르고 묶는 건 자기가 해요 |
| shuffle=True | epoch마다 순서를 섞음 (학습 데이터만) | 같은 순서를 반복하면 순서까지 외워요. 평가 데이터는 섞을 필요가 없어요 |
| next(iter(dl)) | 배치 하나만 꺼내 보기 | 학습 전에 shape·dtype·값 범위를 눈으로 확인하는 첫 점검 |

### nn.Module로 모델 정의 (Flatten → Linear → ReLU → Linear)

```
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512), nn.ReLU(),
            nn.Linear(512, 10))
    def forward(self, x):         # model(x) 하면 이게 실행
        return self.layers(x)

model = Net().to(device)
```

▶ 실제로 돌려 보면 무작위 배치 (64, 1, 28, 28)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model(x).shape | (64, 10) | 클래스 10개 점수(로짓) |
| [p.shape for p in model.parameters()] | [(512, 784), (512,), (10, 512), (10,)] | Linear마다 weight (out, in) + bias (out,) |
| sum(p.numel() for p in model.parameters()) | 407,050 | 784×512 + 512 + 512×10 + 10 |
| model.layers[0](x).shape | (64, 784) | Flatten 지난 직후 |

| 부분 | 뜻 | 바꾸면? |
|---|---|---|
| super().__init__() | nn.Module 기본 준비 (꼭 맨 앞) | 빼면 층을 등록할 수 없어 에러 |
| nn.Linear(in_features, out_features) | (N, in) → (N, out) | 앞 층의 out = 다음 층의 in 이 어긋나면 shape 에러 |
| nn.ReLU() | 음수를 0으로 (비선형) | 빼면 Linear 두 층이 Linear 하나와 같아져요 |
| forward(self, x) | 입력이 지나가는 순서 | model.forward(x) 대신 model(x)로 불러요 |

### 손실 · 최적화 CrossEntropyLoss · MSELoss · SGD · Adam

```
loss_fn = nn.CrossEntropyLoss()     # 분류: 로짓 (N, C) + 정답 (N,) long
loss_fn = nn.MSELoss()              # 회귀: 예측과 정답의 제곱 오차
loss.backward()                     # 모든 가중치의 p.grad 에 기울기를 '더함' → 매번 zero_grad()
optimizer = optim.SGD(model.parameters(), lr=lr)       # p ← p − lr × p.grad
optimizer = optim.Adam(model.parameters(), lr=1e-3)    # 가중치마다 보폭 조절. 처음 시도하기 무난
```

▶ 실제로 돌려 보면 작은 예제 (로짓 2개 · Linear(3, 1))

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| CrossEntropyLoss()(로짓, y) | 0.052 | 정답 점수가 가장 커서 손실이 작아요 |
| CrossEntropyLoss()(softmax(로짓), y) ← 실수 | 0.584 | 같은 예측인데 손실이 커져요 — softmax가 두 번 들어가 차이가 뭉개짐 |
| softmax 값이 완벽한 [1, 0, 0]이어도 | 0.551 | 0이 못 돼요 → 아무리 맞혀도 손실이 이 아래로 안 내려감 |
| zero_grad 없이 backward 3번 → 기울기 크기 | [2.735, 5.47, 8.205] | 더해져서 1배·2배·3배로 쌓여요 |
| SGD step 후 p == p − lr × p.grad ? | True | SGD 한 번 = 정확히 이 식 |
| 모델은 cuda, 입력은 cpu → model(x) | Expected all tensors to be on the same device, but got mat1 is on cpu, different | 장치가 다르면 바로 에러 |

| 인자 | 뜻 | 주의 |
|---|---|---|
| CrossEntropyLoss()(logits, y) | 안에서 softmax + 정답 클래스 확률의 −log | softmax를 미리 하지 않기. 정답은 클래스 번호(정수, long) |
| MSELoss()(pred, y) | (예측 − 정답)² 의 평균 | pred와 y의 shape를 같게 (예: 둘 다 (N, 1)) |
| model.parameters() | 학습할 가중치 목록 | optimizer는 여기 넘긴 것만 갱신해요 |
| lr | 한 번에 움직이는 크기 | Adam은 보통 1e-3부터 |

### 평가 함수 test() · 저장 state_dict

```
def test(model):
    model.eval(); correct = 0          # Dropout 끔
    with torch.no_grad():              # 기울기 기록 끔
        for X, y in test_dl:           # 안 본 데이터
            X, y = X.to(device), y.to(device)
            hit = model(X).argmax(1) == y
            correct += hit.sum().item()
    return correct / len(test_dl.dataset)

torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

▶ 실제로 돌려 보면 위 '전체 흐름'에서 3 epoch 학습한 모델 · 평가 256장

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| test(model) | 1.000 | eval() + no_grad() 로 잰 정확도 |
| len(test_dl), len(test_dl.dataset) | 4, 256 | 배치 수로 나누면 정확도가 64배 커져요 |
| list(model.state_dict().keys()) | ['layers.1.weight', 'layers.1.bias', 'layers.3.weight', 'layers.3.bias'] | 층 번호.weight / .bias |
| state_dict 저장 크기 | 1,592 KB | 가중치 407,050개 × 4바이트 |
| 새 Net()에 load_state_dict → test | <All keys matched successfully> · 같은 정확도: True | 구조는 코드로, 값만 채워 넣기 |
| Dropout 모델 train() 두 번 / eval() 두 번 | 0.797, 0.264 / 0.112, 0.112 | eval()을 빠뜨리면 평가할 때마다 값이 흔들려요 |

| 표현 | 뜻 | 🤔 왜? |
|---|---|---|
| hit.sum().item() | 맞힌 개수 → 파이썬 숫자 | .item()으로 꺼내야 GPU 텐서가 아닌 보통 숫자로 더해져요 |
| len(test_dl.dataset) | 전체 샘플 수 (배치 수 아님) | len(test_dl)은 배치 개수라 나누면 정확도가 틀려요 |
| state_dict() | 층 이름 → 가중치 텐서 사전 | 구조는 코드(Net())로 다시 만들고 값만 채워 넣는 방식 |

💡 학습 손실은 내려가는데 평가 정확도가 떨어지면 과적합이에요 → epochs 줄이기, nn.Dropout(p=0.5) 추가.

### 축 순서 바꾸기 · 개수 축 붙이기

```
images = torch.rand(4, 3, 32, 32)     # 0~1 난수, 모양 연습용
img = images[0]                       # [3, 32, 32]
hwc  = img.permute(1, 2, 0)           # [32, 32, 3]  → plt.imshow(hwc)
nchw = img.unsqueeze(0)               # [1, 3, 32, 32] → 사진 한 장을 모델에 넣을 때
```

▶ 실제로 돌려 보면 torch.rand 난수(seed 0) · 강의 데이터 train/horizontal/train_horizontal_000.png

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| images = torch.rand(4, 3, 32, 32); images.shape | [4, 3, 32, 32] | [N, C, H, W] = 4장 · 3채널 · 32×32 |
| img = images[0]; img.shape | [3, 32, 32] | 한 장을 꺼내면 N축이 빠짐 |
| img.permute(1, 2, 0).shape | [32, 32, 3] | [H, W, C] — plt.imshow가 원하는 순서 |
| hwc[5, 7, 1] == img[1, 5, 7] | True | 값은 그대로, 읽는 순서만 바뀜 |
| img.permute(1, 2, 0).is_contiguous() | False | 메모리는 안 옮기고 보는 방식만 바꾼 view |
| img.unsqueeze(0).shape | [1, 3, 32, 32] | 앞에 개수 축 1을 붙임 → 모델에 한 장 넣을 때 |
| np.array(Image.open(png)).shape | (64, 64, 3) | PIL/NumPy 사진은 [H, W, C] · 0~255 |
| transforms.ToTensor()(pil).shape, min, max | [3, 64, 64], 0.09, 0.72 | ToTensor가 [C, H, W] · 0~1로 바꿔 줌 |

| 호출 | 하는 일 | 바꾸면? |
|---|---|---|
| torch.rand(N, C, H, W) | 0~1 사이 난수 Tensor | 숫자 순서가 곧 축 순서 → C를 뒤에 쓰면 Conv2d가 채널 오류 |
| x.permute(1, 2, 0) | [C,H,W] → [H,W,C]. 값은 그대로, 읽는 순서만 바뀜 | 순서를 틀리면 그림이 깨지거나 imshow가 오류 (모양만 바뀌고 값은 안 바뀌니까) |
| x.unsqueeze(0) | [C,H,W] → [1,C,H,W] | 빼먹으면 3차원이라 "batch 축 없음" 차원 오류 |

### nn.Conv2d

```
x = torch.rand(4, 3, 32, 32)
conv = nn.Conv2d(3, 8, 3, padding=1)
print(conv(x).shape)                          # torch.Size([4, 8, 32, 32])
print(conv.weight.shape, conv.bias.shape)     # [8, 3, 3, 3]  [8]
print(sum(p.numel() for p in conv.parameters()))   # 8·3·3·3 + 8 = 224

# 가중치를 직접 넣어 계산해 보기 (함수형)
import torch.nn.functional as F
inp = torch.zeros(1, 1, 5, 5); inp[0, 0, :, 1] = 1        # 세로선
kernel = torch.zeros(1, 1, 3, 3); kernel[0, 0, :, 1] = 1   # 세로선 검출 Kernel
print(F.conv2d(inp, kernel)[0, 0])                          # 3×3, 세로선 위만 3
```

▶ 실제로 돌려 보면 torch.rand(4, 3, 32, 32) (seed 0) · 5×5 세로선 그림

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| nn.Conv2d(3, 8, 3, padding=1)(x).shape | [4, 8, 32, 32] | 채널 3 → 8, 크기 유지(padding=1) |
| conv.weight.shape, conv.bias.shape | [8, 3, 3, 3], [8] | [out, in, K, K] — Kernel 8장, 각각 두께 3 |
| sum(p.numel() for p in conv.parameters()) | 224 | 8·3·3·3 + 8 = 224 |
| nn.Conv2d(3, 8, 3, padding=0)(x).shape | [4, 8, 30, 30] | padding 0이면 가장자리 2칸 줄어듦 |
| nn.Conv2d(3, 8, 3, stride=2, padding=1)(x).shape | [4, 8, 16, 16] | stride 2면 가로·세로 절반 |
| F.conv2d(inp, kernel)[0, 0] | 3 0 0 3 0 0 3 0 0 | padding 없이 5×5 → 3×3. 세로선(입력 1열)이 Kernel 가운데 열과 겹치는 출력 0열만 1×3 = 3 |
| nn.Conv2d(1, 8, 3)(x) ← in_channels 틀림 | RuntimeError (채널 수 불일치) | Kernel 두께 1인데 입력은 3채널 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| in_channels (필수) | 입력 채널 수 (RGB면 3) | 앞 층 출력 채널과 다르면 오류 — Kernel 한 장이 입력 채널 수만큼 두께를 가져야 곱할 수 있으니까 |
| out_channels (필수) | Kernel 개수 = 출력 채널 수 | 늘리면 찾는 패턴 종류가 많아지지만 파라미터·계산도 비례해 늘어남 |
| kernel_size (필수) | 창 크기 (보통 홀수, 3) | 키우면 한 번에 넓게 보지만 파라미터가 K²에 비례해 늘어남 → 보통 3×3을 여러 층 쌓음 |
| stride | 건너뛰는 칸 수 (기본 1) | 2면 출력 약 절반·계산 약 1/4 — 칸을 건너뛰어 덜 보니까 |
| padding | 가장자리 0 덧대기 (기본 0) | 3×3에 1이면 크기 유지, 0이면 층마다 2칸씩 줄어듦 |
| bias | 편향 사용 (기본 True) | 바로 뒤에 BatchNorm이 오면 False로 끄기도 함 — BN이 평균을 빼면서 bias 효과를 지워 버리거든요 |

### SmallCNN 만들기

```
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(             # 특징 추출부 (Backbone)
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)        # Global Avg Pool
        self.fc = nn.Linear(32, num_classes)       # Head
    def forward(self, x):                          # [N,3,H,W]
        x = self.features(x)                       # [N,32,H/4,W/4]
        x = torch.flatten(self.pool(x), 1)         # [N,32]
        return self.fc(x)                          # [N,K] 점수

model = SmallCNN(num_classes=2)
h = torch.rand(8, 3, 64, 64)
for layer in model.features:
    h = layer(h)
    print(type(layer).__name__, tuple(h.shape))
```

▶ 실제로 돌려 보면 torch.rand(8, 3, 64, 64) (seed 0)를 층마다 통과

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| features[0] Conv2d | [8, 16, 64, 64] | 파라미터 448 |
| features[2] MaxPool2d | [8, 16, 32, 32] | 파라미터 0 |
| features[3] Conv2d | [8, 32, 32, 32] | 파라미터 4,640 |
| features[5] MaxPool2d | [8, 32, 16, 16] | 파라미터 0 |
| model.pool(h).shape | [8, 32, 1, 1] | 채널마다 평균 하나 |
| torch.flatten(p, 1).shape | [8, 32] | [N, 32] |
| torch.flatten(p).shape ← 1을 빼면 | [256] | 8장이 한 줄로 섞여 버림 |
| model(torch.rand(8, 3, 64, 64)).shape | [8, 2] | [N, K] 점수(logit) |
| model(torch.rand(2, 3, 224, 224)).shape | [2, 2] | 크기가 달라도 Head 입력은 32 → 그대로 동작 |
| sum(p.numel() for p in model.parameters()) | 5,154 | 448 + 4,640 + 66 |

| 층 | 출력 shape | 파라미터 |
|---|---|---|
| Conv2d(3,16,3,p=1) → ReLU | [8, 16, 64, 64] | 448 |
| MaxPool2d(2) | [8, 16, 32, 32] | 0 |
| Conv2d(16,32,3,p=1) → ReLU | [8, 32, 32, 32] | 4,640 |
| MaxPool2d(2) | [8, 32, 16, 16] | 0 |
| AdaptiveAvgPool2d(1) | [8, 32, 1, 1] | 0 |
| flatten(x, 1) | [8, 32] | 0 |
| Linear(32, 2) | [8, 2] | 66 |
| 합계 (K=2) | 5,154 |  |

| 명령 | 뜻 | 바꾸면? / 주의 |
|---|---|---|
| nn.MaxPool2d(kernel_size) | 영역 최댓값. stride는 기본 kernel_size와 같음 | 파라미터 0개 (고르기만 하고 배울 숫자가 없음). 3으로 키우면 더 많이 줄고 디테일을 더 버림 |
| nn.AdaptiveAvgPool2d(1) | 입력 크기와 상관없이 채널마다 평균 하나 | 출력 크기를 "1×1"로 정해 두니까 32·64·224 어떤 사진이 와도 Head 입력이 32개로 같음 |
| torch.flatten(x, 1) / nn.Flatten() | [N,32,1,1] → [N,32] | 1을 빼면 N축까지 펴서 [N·32]가 됨 → 사진끼리 섞여 버림 |
| nn.Conv2d(…, stride=2) | Pool 대신 크기 줄이기 | 줄이는 법까지 배움 → 파라미터 있음 (8→8채널이면 584개) |

🤔 왜 마지막에 Global Avg Pool을 써? — Feature Map을 그대로 펴서 Linear에 넣으면 사진 크기마다 입력 개수가 달라지고 파라미터도 커져요. 채널마다 평균 하나만 남기면 "이 패턴이 사진 전체에서 얼마나 나왔나"만 남아서, 위치 정보는 버리고 크기와도 무관해져요.

### nn.BatchNorm2d (ResNet18엔 있고, SmallCNN엔 없음)

```
block = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1, bias=False),
    nn.BatchNorm2d(16),            # 채널 수와 같아야 함
    nn.ReLU(),
)
print(block(torch.rand(8, 3, 64, 64)).shape)   # [8, 16, 64, 64]  (Shape 그대로)
```

▶ 실제로 돌려 보면 torch.rand(8, 3, 64, 64) (seed 0)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| block(x).shape | [8, 16, 64, 64] | Shape 그대로 |
| sum(p.numel() for p in bn.parameters()) | 32 | 채널마다 γ·β 하나씩 = 16 × 2 |
| conv 출력 채널 평균 [:3] | -0.024 -0.228 0.075 | BN 전: 채널마다 제각각 |
| train 모드 BN 출력의 채널 평균·표준편차 | 평균 ≈ 1.3e-07, std ≈ 1.000 | 모든 채널이 평균 0 · 표준편차 1로 |
| bn.running_mean[:3] (forward 한 번 뒤) | -0.002 -0.023 0.007 | 0에서 batch 평균 쪽으로 10%(momentum 0.1)만 이동 |
| train: 같은 사진, batch 구성만 바꿔 출력 차이 | 3.530 | batch 통계를 써서 같은 사진도 결과가 달라짐 |
| eval: 같은 사진, batch 구성만 바꿔 출력 차이 | 0.000 | 저장 통계를 써서 언제나 같은 결과 |

| 인자 / 모드 | 뜻 | 바꾸면? |
|---|---|---|
| num_features (필수) | 채널 수 C | 앞 Conv의 out_channels와 다르면 오류 (채널마다 따로 평균·크기를 맞추니까) |
| model.train() | 지금 batch의 평균·분산으로 정규화 + 저장 통계 갱신 | 평가 때 이 모드면 batch 구성에 따라 결과가 흔들림 |
| model.eval() | 학습 중 저장해 둔 통계로 정규화 | 같은 사진은 언제나 같은 결과 → 평가·추론은 꼭 eval |

🤔 왜 BatchNorm을 넣어? — 층이 깊어지면 값이 층마다 점점 커지거나 한쪽으로 쏠려서, 학습률을 조금만 키워도 학습이 튀어요. BN이 채널마다 값의 평균·크기를 맞춰 주면 큰 학습률에서도 안정적으로 학습돼요. 평가 때는 batch마다 결과가 달라지면 안 되니까 저장된 통계를 써요(train/eval 동작이 다른 이유).

### ImageFolder + transforms + DataLoader

```
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_ds = datasets.ImageFolder("data/train", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
images, labels = next(iter(train_loader))
print(train_ds.class_to_idx)       # {'horizontal': 0, 'vertical': 1}  ← 폴더 이름 순서대로 번호
print(images.shape, labels.shape)  # torch.Size([32, 3, 64, 64]) torch.Size([32])
```

▶ 실제로 돌려 보면 강의 데이터 data/train (horizontal 30 · vertical 30) · data/valid (10 · 10)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| train_ds.class_to_idx | {'horizontal': 0, 'vertical': 1} | 폴더 이름 알파벳 순서로 번호 |
| len(train_ds), len(valid_ds) | 60, 20 |  |
| train_ds.samples[0] | ('…/train/horizontal/train_horizontal_000.png', 0) | (파일 경로, 라벨) |
| type(ImageFolder('data/train')[0][0]) ← transform 없음 | Image | PIL 이미지 그대로 → batch로 못 묶음 |
| images, labels = next(iter(train_loader)) | [32, 3, 64, 64], [32] | batch_size=32 → 32장씩 |
| images.min(), images.max() | 0.03, 0.95 | ToTensor로 0~1 실수 |
| labels[:12] | 1 0 1 1 1 1 1 0 1 0 0 1 | shuffle=True라 두 클래스가 섞여 나옴 |
| len(train_loader) | 2 | 60장 ÷ 32 → 마지막 batch는 28장 |

| 명령 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| ImageFolder(root, transform=None) | root 필수. 이미지 + 폴더 라벨 쌍 | transform을 안 주면 PIL 이미지 그대로 → 크기도 제각각이라 batch로 못 묶음 |
| .class_to_idx / .samples | 이름→번호 사전 / (파일경로, 라벨) 목록 | 폴더 이름을 바꾸면 정렬 순서가 바뀌어 번호도 바뀜 → 저장할 때 같이 남겨야 하는 이유 |
| transforms.Compose([...]) | 변환을 순서대로 연결 | 순서가 중요: ToTensor 전엔 PIL용, 후엔 Tensor용 변환(Normalize) |
| transforms.Resize((64, 64)) | 목표 크기로 맞춤 (크기 필수) | 크게 하면 디테일↑·계산↑, 작게 하면 빠르지만 작은 물체가 뭉개짐 |
| transforms.ToTensor() | PIL [H,W,C] 0~255 → Tensor [C,H,W] 0~1 실수 | 인자 없음. 빼면 모델이 PIL을 못 받음 |
| transforms.Normalize(mean, std) | 채널마다 (x − mean) / std | 사전학습 모델엔 그 모델이 배울 때 쓴 값을 써야 함 (ResNet18 Head-only 항목의 weights.transforms(), 이론은 Day11 ⑤ 전이학습) |
| DataLoader(…, shuffle=True) | batch로 묶기, train은 섞기 | train을 안 섞으면 같은 클래스만 연달아 나와 학습이 한쪽으로 쏠림. valid는 섞을 필요 없음 |

### 손실 · 학습 루프 · valid 평가 · best 저장

```
import copy
model = SmallCNN(num_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()                 # softmax 전 logit을 그대로 넣음
best_loss = float("inf")
for epoch in range(1, num_epochs + 1):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    vl_sum, correct, n = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in valid_loader:
            logits = model(images)
            vl_sum += loss_fn(logits, labels).item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            n += len(labels)
    vl, va = vl_sum / n, correct / n
    if vl < best_loss:
        best_loss, best_epoch, best_state = vl, epoch, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)
```

▶ 실제로 돌려 보면 강의 데이터 train 60장으로 SmallCNN 8 epoch (seed 0, NVIDIA GeForce RTX 5080 Laptop GPU) — 숫자는 작은 데이터라 흔들릴 수 있어요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| nn.CrossEntropyLoss()([[0., 0.]], [0]) | 0.693 | 반반 확신 → −log 0.5 |
| nn.CrossEntropyLoss()([[3., 0.]], [0]) | 0.049 | 정답에 자신 있게 → 벌점 작음 |
| nn.CrossEntropyLoss()([[0., 3.]], [0]) | 3.049 | 자신 있게 틀림 → 벌점 큼 |
| epoch 1: valid loss · acc | 0.690 · 0.55 |  |
| epoch 2: valid loss · acc | 0.687 · 0.55 |  |
| epoch 3: valid loss · acc | 0.683 · 1.00 |  |
| epoch 4: valid loss · acc | 0.679 · 1.00 |  |
| epoch 5: valid loss · acc | 0.674 · 1.00 |  |
| epoch 6: valid loss · acc | 0.669 · 1.00 |  |
| epoch 7: valid loss · acc | 0.664 · 1.00 |  |
| epoch 8: valid loss · acc | 0.658 · 1.00 |  |
| best_epoch, best_loss | 8, 0.658 | valid loss가 가장 낮았던 순간 |
| (해석) | acc는 epoch 3부터 1.00, loss는 0.69 근처 | 방향은 다 맞혔지만 확률은 아직 반반에 가까움 — lr 1e-3에 step 16번뿐이라 확신이 덜 자람. 정확도만 보면 이걸 놓쳐요 |
| state_dict()를 그대로 담고 나서 학습이 바뀌면 | 담아 둔 값도 바뀜: True | 같은 Tensor를 가리킴 |
| copy.deepcopy(state_dict())로 담았으면 | 담아 둔 값도 바뀜: False | 복사본이라 그대로 남음 |

| 명령 | 뜻 | 바꾸면? / 빼먹으면? |
|---|---|---|
| nn.CrossEntropyLoss() | 점수(logit) → Softmax 확률 → 정답 확률의 −log | 앞에 softmax를 또 붙이면 두 번 적용돼 학습이 느려짐 (안에 이미 들어 있음) |
| model.eval() | BatchNorm·Dropout을 평가 모드로 | 빼먹으면 BN이 valid batch 통계를 써서 점수가 흔들림 |
| torch.inference_mode() | 기울기 기록 끄기 | 빼먹어도 값은 같지만 메모리·시간 낭비 (평가엔 역전파가 필요 없으니까) |
| copy.deepcopy(model.state_dict()) | best 순간의 가중치를 복사해 둠 | deepcopy 없이 담으면 같은 Tensor를 가리켜서 이후 학습에 따라 같이 바뀜 |

🤔 왜 CrossEntropy는 "정답 확률의 −log"야? — 정답 확률이 1이면 −log 1 = 0(벌점 없음), 0.01이면 −log 0.01 ≈ 4.6(큰 벌점). 틀렸는데 자신만만할수록 벌점이 급격히 커져서, 모델이 정답 쪽 확률을 올리도록 강하게 밀어줘요.

### 저장할 땐 "읽는 법"까지 함께

```
torch.save({
    "state_dict": best_state,
    "class_to_idx": train_ds.class_to_idx,   # 클래스 순서
    "image_size": 64,                        # 입력 크기
    "normalize": None,                       # 정규화 값 (썼다면 mean·std)
    "best_epoch": best_epoch,
}, "best_model.pt")

ckpt = torch.load("best_model.pt", weights_only=True)
restored = SmallCNN(num_classes=len(ckpt["class_to_idx"]))
restored.load_state_dict(ckpt["state_dict"])
restored.eval()
```

▶ 실제로 돌려 보면 '손실 · 학습 루프' 항목에서 학습한 best SmallCNN을 임시 폴더에 저장

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| torch.save({...}, "best_model.pt") 파일 크기 | 23.5 KB | 파라미터 5,154개 × 4byte ≈ 20KB + 정보 |
| list(ckpt.keys()) | ['state_dict', 'class_to_idx', 'image_size', 'normalize', 'best_epoch'] | 숫자 + 읽는 법 |
| ckpt["class_to_idx"], ckpt["image_size"] | {'horizontal': 0, 'vertical': 1}, 64 | 0번이 horizontal인 걸 파일만 보고 앎 |
| torch.allclose(model(x), restored(x)) | True | 불러온 모델이 같은 답을 냄 |
| 다른 구조에 load_state_dict | RuntimeError: Error(s) in loading state_dict for Sequential:… | 층 이름·크기가 안 맞으면 오류 |

| 명령 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| torch.save(obj, path) | 사전·Tensor를 파일로 | state_dict만 저장하면 숫자 묶음뿐 → 나중에 "0번이 cat이었나?"·입력 크기를 모름 |
| torch.load(path, weights_only=True) | Tensor·기본 자료형만 안전하게 읽기 | False면 파일 안의 임의 파이썬 객체까지 실행될 수 있음 → 신뢰하는 파일만 |
| load_state_dict() + eval() | 같은 구조 모델에 값 넣고 평가 모드 | 구조(층 이름·크기)가 다르면 키가 안 맞아 오류 |

🤔 왜 정확도 하나만 보면 안 돼? — 클래스 비율이 9:1이면 "전부 다수 클래스"라고만 해도 정확도 90%예요. Confusion Matrix · 클래스별 Recall을 보면 어느 클래스를 어느 쪽으로 틀리는지 보이고, 실제로 틀린 사진(오류 사례)을 보면 "왜" 틀렸는지(어둡다, 잘렸다…) 단서가 나와요.

### ResNet18 Head-only 전이학습

```
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT           # = IMAGENET1K_V1
transform = weights.transforms()             # 사전학습 때와 같은 Resize·Crop·Normalize
train_ds = datasets.ImageFolder("data/train", transform=transform)

model = resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False                  # 1) Backbone 고정
model.fc = nn.Linear(model.fc.in_features, 2)   # 2) Head 교체 (고정 코드 뒤에!)
optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)

model.eval()                                 # 3) 고정한 층의 BN은 저장 통계 유지
model.fc.train()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))   # 1026
```

▶ 실제로 돌려 보면 ResNet18 — 사전학습 가중치 없이 구조만(weights=None, 다운로드 안 함) · 강의 PNG 한 장

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| weights.transforms() | crop 224 · resize 256 · bilinear mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225] | ImageNet 학습 때와 같은 입력 규칙 |
| transform(64×64 PNG).shape | [3, 224, 224] | 64×64가 224×224로 커짐 |
| transform(PNG).mean(), .std() | -0.81, 0.89 | Normalize 때문에 0~1이 아님 (음수도 나옴) |
| resnet18() 전체 파라미터 | 11,689,512 | Head(1000클래스) 포함 |
| model.fc.in_features | 512 | Head에 들어오는 특징 수 |
| 얼린 뒤 fc 교체 → 학습 파라미터 수 | 1,026 | 512 × 2 + 2 = Head만 |
| fc 교체 뒤에 얼리면 → 학습 파라미터 수 | 0 | 순서 실수: 새 Head까지 얼어 배울 게 0개 |
| model.eval(); model.fc.train() 후 forward → bn1 저장 통계 | 그대로 | 얼린 층의 BN 통계 유지 |
| model.train() 후 forward → bn1 저장 통계 | 바뀜 | requires_grad=False여도 BN 통계는 바뀜! |

| 명령 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| resnet18(weights=ResNet18_Weights.DEFAULT) | 공식 ImageNet 사전학습 가중치로 생성 (처음 한 번 다운로드) | weights=None이면 무작위 시작 → 전이학습이 아님 (빌려 올 지식이 없으니까) |
| weights.transforms() | 그 가중치가 학습될 때 쓴 입력 변환 (224×224 · ImageNet Normalize) | 다른 크기·정규화를 쓰면 Backbone이 본 적 없는 분포의 숫자가 들어와 사전학습 효과가 줄어듦 |
| p.requires_grad = False | 그 파라미터는 기울기 계산 안 함 → 안 바뀜 | BN의 저장 통계는 기울기가 아니라 forward 때 갱신돼서 이걸로 안 멈춤 → 3)이 필요한 이유 |
| model.fc = nn.Linear(model.fc.in_features, K) | Head를 내 클래스 수로 교체 (in_features = 512) | Freeze보다 먼저 교체하면 새 Head까지 얼어 학습할 게 0개가 됨 (새 층은 기본 requires_grad=True) |
| model.eval() → model.fc.train() | 얼린 층의 BatchNorm은 저장된 통계 유지 | 나중에 model.train()을 다시 부르면 BN 통계가 내 작은 데이터로 바뀌어 버림 |

### Partial 미세조정 (layer4 + Head, lr 다르게)

```
for p in model.layer4.parameters():
    p.requires_grad = True
optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},   # 열어 준 층은 작게
    {"params": model.fc.parameters(),     "lr": 1e-3},   # 새 Head는 크게
])
```

▶ 실제로 돌려 보면 Head-only로 만든 ResNet18 구조(weights=None, 구조만)에서 layer4를 열었을 때

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| sum(p.numel() for p in model.layer4.parameters()) | 8,393,728 | 마지막 블록 하나에 전체의 약 3/4이 몰려 있음 |
| 학습 파라미터 수 (layer4 + fc) | 8,394,754 | 전체 11,177,538의 75.1% |
| [g['lr'] for g in optimizer.param_groups] | [0.0001, 0.001] | 그룹마다 다른 lr |
| [len(g['params']) for g in optimizer.param_groups] | [15, 2] | layer4 Tensor 15개(Conv·BN), fc 2개(weight·bias) |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| param group 리스트 [{"params":…, "lr":…}, …] | 층 묶음마다 lr을 따로 | layer4에도 1e-3을 주면 사전학습 값이 크게 흔들려 오히려 성능이 떨어질 수 있음 |
| model.layer4 | ResNet18의 마지막 큰 블록 (512ch) | layer3까지 열면 더 유연하지만 학습 파라미터·과적합 위험이 커짐 |

⚠️ SmallCNN(처음부터)과 ResNet18(전이학습)은 입력 크기·lr·학습 범위가 달라서, 성능 차이를 "순수한 구조 차이"라고 단정하지 말고 조건을 같이 적어요.

### 한 epoch 시간 재기

```
import time
model = model.to(device)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
seen, start = 0, time.perf_counter()
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    loss_fn(model(images), labels).backward()
    optimizer.step()
    seen += len(labels)
if device.type == "cuda":
    torch.cuda.synchronize()
seconds = time.perf_counter() - start
print(f"{seen / seconds:.2f} images/s")
if device.type == "cuda":
    print(torch.cuda.max_memory_allocated() / 1024**2, "MiB")
```

▶ 실제로 돌려 보면 SmallCNN 학습 step, 워밍업 1회 후 측정 — 이 PC 기준이라 다른 장치에선 숫자가 달라요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| CPU · 강의 train 60장, batch 32 | 5,081 images/s · 메모리 N/A | CPU에선 VRAM 값 없음 |
| GPU(NVIDIA GeForce RTX 5080 Laptop GPU) · 같은 조건 | 21,048 images/s · peak 95.1 MiB | CPU의 약 4.1배 — 60장·batch 32라 GPU가 덜 바쁨 |
| CPU · 난수 1024장, batch 256 | 4,477 images/s |  |
| GPU · 난수 1024장, batch 256 | 49,138 images/s · peak 342.2 MiB | batch가 크면 GPU가 약 11배 |
| forward 20번: synchronize 없이 / 있게 | 6.1 ms / 22.1 ms | 없으면 GPU가 끝나기 전에 시계를 멈춤 |

| 명령 | 뜻 | 빼먹으면? |
|---|---|---|
| time.perf_counter() | 짧은 구간 측정용 정밀 시계(초) | time.time()은 시스템 시계라 짧은 구간에선 덜 정확 |
| torch.cuda.synchronize() | GPU 작업이 다 끝날 때까지 기다림 | 시간이 비정상적으로 짧게 찍힘 |
| torch.cuda.reset_peak_memory_stats() | 최대 메모리 기록을 0부터 다시 | 앞서 돌린 작업의 최대값이 섞임 |
| torch.cuda.max_memory_allocated() | 측정 구간에서 가장 많이 쓴 GPU 메모리(바이트) | CPU에선 의미 없음 → N/A |

🤔 왜 GPU가 항상 빠르진 않아? — GPU는 "한 번에 많이" 계산할 때 강해요. 작은 모델·작은 batch에선 데이터를 GPU로 옮기는 시간과 명령을 띄우는 오버헤드가 계산 시간보다 커질 수 있어요. 그래서 측정 범위(디스크 로딩 포함인지 등)도 같이 적어요.

💡 최종 성능표엔 모델 · 학습 범위 · valid 지표 · 시간 · VRAM을 함께 남기고, test는 모든 선택이 끝난 뒤 한 번만 봐요.

### 증강 transforms.Compose (train만)

```
train_tf = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),          # 좌우 반전 (확률 0.5)
    transforms.RandomRotation(10),              # ±10도 회전
    transforms.ColorJitter(brightness=0.15),    # 밝기 조금 흔들기
    transforms.ToTensor(),
])
valid_tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
```

▶ 실제로 돌려 보면 100×80 합성 그라데이션 사진 한 장

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| train_tf(img).shape, dtype | (3, 64, 64), torch.float32 | [C, H, W], 0~1 실수 (ToTensor) |
| train_tf(img) 두 번이 같나? | False | 부를 때마다 무작위로 다르게 변형 |
| valid_tf(img) 두 번이 같나? | True | 평가용은 항상 똑같이 |
| valid_tf(img).min(), max() | 0.000, 0.698 | ToTensor가 0~255를 0~1로 |

## 📖 Day11 용어 (47개)
- **가중합** (Weighted sum (가중합)): 각 입력에 가중치(계수)를 곱해 모두 더한 값. w₁x₁ + w₂x₂ + … + b.
- **혼동행렬** (Confusion Matrix (혼동행렬)): 실제(행) × 예측(열)으로 맞힘·틀림을 네 칸(TN, FP, FN, TP)에 세어 둔 표. 어떤 종류의 오류인지 나눠 볼 수 있어요.
- **transform** (transform / fit_transform): transform은 이미 배운 규칙(평균·중앙값 등)으로 데이터를 바꾸기만 하는 것. fit_transform은 배우기와 바꾸기를 한 번에(train에만).
- **동결** (Freeze / Lock (설정 동결)): test를 열기 전에 모델·특성·임계값·선택 규칙을 파일로 기록해 더 이상 바꾸지 않는 것. test 점수를 보고 바꾸면 test가 선택용 자료가 돼 버려요.
- **처리량** (Throughput (처리량)): 정해진 시간에 처리하는 일의 양. GPU는 같은 모양의 독립 계산이 충분히 많을 때 처리량이 커져요.
- **requires_grad** (requires_grad (기울기 추적 여부)): True면 이 Tensor로 하는 연산을 Autograd가 기록해 기울기를 구해요. 학습할 파라미터는 보통 True, 기본값은 False.
- **logit** (Logits (원시 점수)): 모델이 마지막 층에서 낸, softmax 하기 전의 클래스별 점수. 합이 1일 필요가 없고 확률이 아니에요.
- **softmax** (Softmax): logits를 0~1 사이, 합이 1인 확률로 바꾸는 함수. CrossEntropyLoss가 안에서 해 주니 모델 끝에는 붙이지 않아요.
- **CrossEntropyLoss** (Cross-Entropy Loss (교차 엔트로피 손실)): 분류용 손실 함수. softmax 전 logits와 클래스 번호(long, shape (batch,))를 받아, 정답 클래스 확률이 낮을수록 loss를 크게 줘요.
- **BatchNorm** (Batch Normalization (배치 정규화)): 층의 중간값을 현재 batch의 평균·분산으로 정리(정규화)해 학습을 안정시키는 층. 평가 땐 학습 중 모아 둔 running 통계를 써요.
- **CNN** (Convolutional Neural Network (합성곱 신경망)): 이미지처럼 가로·세로 구조가 있는 데이터에 쓰는 신경망. 입력 shape는 보통 (batch, channel, height, width)예요.
- **state_dict** (state_dict (모델 상태 묶음)): 모델의 모든 가중치(와 BatchNorm 통계)를 이름별로 담은 사전. load_state_dict()로 그 상태로 되돌려요.
- **합성곱** (Convolution (합성곱)): 작은 창을 옮겨 가며 창 안의 값과 가중치를 곱해 더하는(가중합) 연산. 모든 위치에서 같은 가중치를 써요.
- **Kernel** (Kernel / Filter (커널, 필터)): 합성곱 창 안의 가중치 묶음. 3×3이면 숫자 9개이고, 이 숫자는 사람이 아니라 학습으로 정해져요.
- **Feature Map** (Feature Map (특징 맵)): Kernel 하나가 사진 전체를 훑어서 만든 결과 지도. 그 Kernel이 찾는 패턴이 어디에 얼마나 강하게 있는지를 보여 줘요.
- **채널** (Channel (채널)): 이미지·Feature Map이 겹겹이 쌓인 '장 수'. 컬러 사진은 R·G·B 3채널이고, Conv 층의 출력 채널 수 = Kernel 개수예요.
- **NCHW** (N(장수) · C(채널) · H(높이) · W(너비)): PyTorch가 이미지 묶음을 담는 축 순서. 사진 한 장도 [1, C, H, W]처럼 장수 축을 붙여 넣어요. PIL·matplotlib은 [H, W, C] 순서라 바꿔 줘야 해요.
- **Stride** (Stride (보폭)): 창이 한 번에 건너뛰는 칸 수. 1이면 한 칸씩, 2면 두 칸씩 옮겨서 출력이 약 절반 크기가 돼요.
- **Padding** (Padding (채워 넣기)): 모자란 자리를 가짜 값(보통 0)으로 채우는 것. CNN에서는 사진 가장자리 바깥에 0을 덧대 크기를 유지하고, 순서 데이터에서는 짧은 순서 끝에 0을 채워 길이를 맞춰요. 채운 값은 진짜 데이터가 아니에요.
- **Pooling** (Pooling (풀링, 다운샘플링)): 작은 영역을 대표값 하나로 줄이는 것. MaxPool 2×2는 4칸 중 가장 큰 값 하나만 남겨 크기를 절반으로 줄여요. 채널 수는 그대로예요.
- **수용영역** (Receptive Field (수용영역)): 출력 한 칸이 원본 사진에서 '보고 있는' 범위. 층을 쌓고 Pooling을 거칠수록 넓어져요.
- **가중치 공유** (Weight sharing (가중치 공유)): 같은 가중치(Kernel)를 모든 위치에서 돌려 쓰는 것. 그래서 파라미터 수가 사진 크기와 상관없어요. 순서 모델도 모든 시점에 같은 가중치를 써요.
- **등변** (Equivariance (등변성)): 입력을 옮기면 결과도 똑같이 옮겨지는 성질. 합성곱이 이래서 물체가 사진 어디에 있어도 같은 검출기로 찾아요.
- **불변** (Invariance (불변성)): 입력이 조금 바뀌어도 결과가 그대로인 성질. MaxPool은 2×2 안에서 최댓값 위치가 살짝 움직여도 결과가 같아서 작은 이동에 불변이에요.
- **Global Average Pooling** (Global Average Pooling (전역 평균 풀링)): 채널마다 Feature Map 전체의 평균 하나만 남겨 [N, C, H, W]를 [N, C, 1, 1]로 줄여요. 사진 크기와 상관없이 Head 입력 개수가 같아져요.
- **Flatten** (Flatten (펴기)): 여러 축의 Tensor를 한 줄 벡터로 펴는 것. CNN에서는 장수 축 N은 남기고 나머지만 펴서 [N, 32, 1, 1] → [N, 32]로 만들어요.
- **Backbone** (Backbone (백본, 특징 추출부)): 모델 앞부분. 사진에서 선·모서리·질감·모양 같은 특징을 뽑아내는 부분이에요. 전이학습에서는 사전학습된 Backbone을 그대로 빌려 써요.
- **Head** (Head (헤드, 분류기)): 모델 끝부분. Backbone이 뽑은 특징을 보고 최종 답(클래스별 점수 등)을 내는 층이에요. 전이학습에서는 내 클래스 수에 맞게 이 부분만 바꿔 끼워요. (Transformer의 Multi-head의 'Head'와는 다른 뜻이에요.)
- **ImageFolder** (torchvision.datasets.ImageFolder): 폴더 구조로 이미지 데이터셋을 만드는 도구. 하위 폴더 이름이 곧 라벨이고, 이름을 정렬해 0, 1, 2… 번호를 붙여요(class_to_idx로 확인).
- **torchvision** (torchvision (PyTorch 이미지 패키지)): PyTorch용 이미지 도구 모음. 이미지 폴더 읽기(ImageFolder), 변환·증강(transforms), 사전학습 모델(ResNet18 등)이 들어 있어요. torch와 버전 짝을 맞춰 설치해야 해요.
- **Pillow** (Pillow (파이썬 이미지 라이브러리, import 이름은 PIL)): PNG·JPG 같은 이미지 파일을 열고 다루는 파이썬 라이브러리. 설치는 pillow, 코드에서는 PIL로 불러요.
- **ToTensor** (transforms.ToTensor()): PIL 이미지 [H, W, C], 0~255 정수를 Tensor [C, H, W], 0~1 실수로 바꿔 줘요. 이게 없으면 모델이 이미지를 못 받아요.
- **Normalize** (transforms.Normalize(mean, std)): 채널마다 (값 − 평균) ÷ 표준편차로 숫자 범위를 맞춰요. 사전학습 모델을 쓸 땐 그 모델이 배울 때 쓴 평균·표준편차를 그대로 써야 해요.
- **증강** (Data Augmentation (데이터 증강)): 사진을 뒤집고, 살짝 돌리고, 밝기를 바꾸는 등 정답이 안 바뀌는 변형으로 학습 예시를 늘리는 기법. 과적합을 줄여요. train에만 쓰고, 정답을 바꾸는 변형(예: 6↔9 회전)은 금지예요.
- **eval 모드** (model.train() / model.eval()): 모델을 학습용·평가용 동작으로 바꾸는 스위치. BatchNorm·Dropout처럼 둘 때 동작이 다른 층이 있어서, 평가·추론 전에는 꼭 model.eval()을 불러요.
- **inference_mode** (torch.inference_mode()): 이 블록 안에서는 기울기 기록을 끄는 설정. 평가·추론에는 역전파가 필요 없으니 메모리와 시간을 아껴요.
- **전이학습** (Transfer Learning (전이학습)): 큰 데이터로 미리 학습한 모델을 가져와, 마지막 Head만 내 문제에 맞게 바꿔 다시 배우는 방법. 데이터가 적어도 빨리, 덜 과적합되게 배워요.
- **사전학습** (Pre-training (사전학습)): 다른 큰 데이터(예: ImageNet)로 미리 학습해 두는 것. 이렇게 만든 가중치를 내려받아 출발점으로 써요.
- **ImageNet** (ImageNet (대규모 이미지 데이터셋)): 사진 100만 장 이상, 1000개 클래스로 된 유명한 이미지 데이터셋. 많은 사전학습 모델이 이걸로 배웠어요.
- **ResNet18** (ResNet-18 (Residual Network, 18층)): 층 18개짜리 대표적인 CNN 모델. ImageNet 사전학습 가중치를 torchvision에서 바로 받아 전이학습에 써요. 마지막 Head는 model.fc예요.
- **Fine-tuning** (Fine-tuning (미세조정)): 사전학습된 층의 일부 또는 전체를 다시 열어 작은 학습률(lr)로 살짝만 고치며 학습하는 것. 보통 뒤쪽 층(layer4)부터 열어요.
- **Head-only** (Head-only 전이학습): Backbone은 전부 얼리고 새 Head만 학습하는 방식. ResNet18·2클래스면 학습할 파라미터가 1,026개뿐이라 빠르고 과적합 위험이 작아요.
- **Partial** (Partial fine-tuning (부분 미세조정)): 뒤쪽 일부 층(예: layer4)과 Head만 열어 학습하는 방식. 연 층은 작은 lr(1e-4), 새 Head는 큰 lr(1e-3)로 따로 줘요.
- **워밍업** (Warm-up (워밍업)): 시간을 재기 전에 몇 번 미리 돌려 두는 것. 첫 실행엔 초기화·메모리 준비 시간이 섞여 유난히 느리기 때문이에요.
- **synchronize** (torch.cuda.synchronize() (GPU 작업 끝날 때까지 기다리기)): GPU는 명령만 받아 두고 파이썬은 바로 다음 줄로 넘어가요(비동기). 이걸 시간 재기 전후에 불러야 GPU 계산이 진짜 끝난 시점까지 잴 수 있어요.
- **Peak VRAM** (Peak VRAM (GPU 메모리 최대 사용량)): 측정 구간에서 GPU 메모리(VRAM)를 가장 많이 쓴 순간의 양. 빨라도 메모리가 모자라면 못 돌리니 시간과 함께 기록해요. CPU에선 N/A로 적어요.
- **perf_counter** (time.perf_counter() (정밀 시계)): 짧은 구간의 시간을 정밀하게 재는 파이썬 시계(초 단위). 시작·끝 값의 차이가 걸린 시간이에요.

---

# Day12 · 순서 모델 (RNN · LSTM · Autoencoder · Transformer)
_2026-11-02(월) · Padding/Mask · RNN · LSTM · Autoencoder · Attention · Transformer · 아키텍처 비교 · 길이·속도 측정_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day12.html

📌 딱 3줄 요약

- 순서 데이터는 [B, T, F](개수 · 시점 · 특징). 모든 시점에 같은 가중치를 써서 길이가 달라도 같은 모델로 읽고, 길이를 맞추려 채운 Padding은 계산에서 꼭 뺀다(lengths · Mask).

- RNN → LSTM → Transformer는 "순서를 더 잘 읽기"의 발전: 메모 한 장(h) → 곱하지 않고 더하는 기억 통로(c) → 모든 위치를 한 번에 직접 보기(Attention, 대신 비용 T²).

- Autoencoder는 목적이 다른 갈래: 미래가 아니라 입력 자신을 좁은 통로로 되살리고, 잘 못 되살리면 "이상 후보". 예측·재구성 결과는 따로 기록해요.

🔧 Day12 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| torch | RNN·LSTM·Transformer 층, Padding 도구, 손실 함수, DataLoader 전부 (Day9부터 설치됨) | import torch, from torch import nn, from torch.nn.utils.rnn import pad_sequence |
| matplotlib (선택) | loss 곡선, 예측 vs 실제, Mask·Latent 그림 | import matplotlib.pyplot as plt |

### 🔧 확인

```
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

💡 오늘 쓰는 순서 모델 도구(nn.RNN, nn.LSTM, nn.MultiheadAttention, pad_sequence …)는 전부 torch 안에 들어 있어서 따로 설치할 게 없어요. 정상 설치된 환경은 그대로 두고, 기본 학습은 CPU로, GPU 비교는 GPU가 있을 때만 해요.

🤔 왜 설치 버전을 기록해? — 시간·메모리 측정값과 일부 기본 동작은 PyTorch 버전·장치에 따라 달라져요. 나중에 결과를 다시 만들거나 남과 비교하려면 "어떤 환경에서 잰 숫자인지"가 같이 있어야 해요.

## 이거 어디에 써?

Day11의 이미지는 "한 장" 데이터였죠. 오늘은 순서가 의미를 가지는 데이터예요(센서·문장·로그·오디오). 같은 단어라도 순서를 바꾸면 뜻이 달라져요(“woman ate raccoon” vs “raccoon ate woman”). 그래서 시계열은 순서를 마구 섞으면 같은 문제가 아니에요.

📈

다음 값 예측과거 12개 관측을 보고 다음 1개 맞히기 (설비 센서)

💬

순서 분류문장·로그·신호 한 줄을 보고 클래스 고르기

🏭

점검 후보 찾기정상 패턴을 되살리지 못하는 사례 표시

🌐

긴 문맥 읽기Transformer: 멀리 떨어진 위치끼리 직접 연결

| 입출력 구조 | 하는 일 | 예 |
|---|---|---|
| Many-to-one | 전체 순서를 읽고 클래스·값 하나 | 문장 감성 분류 |
| Many-to-many | 각 시점마다 태그·상태·예측 | 시점별 상태 표시 |
| Next-step | 과거 Window로 다음 시점 값 | 과거 12개 → 다음 1개 |
| Encoder–Decoder | 입력 순서를 다른 길이의 출력 순서로 | 번역 |

## ① 순서 데이터 [B, T, F] · Padding · Mask · Window

| 축 | 뜻 | 예 (오늘 LSTM 예측) |
|---|---|---|
| B · Batch | Window(순서) 개수 | B개 |
| T · Time | 시점 수 (길이) | T=12, 과거 12개 관측 |
| F · Feature | 한 시점에 동시에 관측한 특성 수 | F=1, 시점마다 값 1개 |

순서마다 길이가 달라요(4, 2, 5 …). 한 묶음(Tensor)으로 만들려면 가장 긴 길이에 맞춰 0을 채워야 해요. 이게 Padding. 대신 "진짜 길이(lengths)"나 "어디가 채운 칸인지(Mask)"를 함께 들고 다녀요. Padding을 쓰면 T는 batch 안의 최대 길이가 돼요.

```
순서 [3, 1, 4] 를 길이 5에 맞춤 → [3, 1, 4, 0, 0]

✕ Padding까지 평균:  (3+1+4+0+0) ÷ 5 = 1.6     ← 틀림
○ 실제 길이로 평균:  (3+1+4) ÷ 3     = 2.67
```

🤔 왜 길이가 달라도 같은 모델에 넣을 수 있어? — 순서 모델은 모든 시점에 같은 가중치를 써요. 시점마다 다른 가중치(W₁, W₂, …)를 쓰면 길이 4로 배운 모델은 5번째 시점용 W₅가 없어서 못 읽고, 길이가 바뀔 때마다 새로 배워야 하죠. 같은 W를 반복하면 길이가 4든 50이든 그대로 읽어요. 그래서 nn.RNN(input_size, hidden_size)에는 길이 T를 정하는 인자가 아예 없어요. (Day11 Conv가 "모든 위치에 같은 Kernel"이었던 것과 같은 생각이에요.)

🤔 왜 Padding은 계속 빼라고 해? — Padding은 길이를 맞추려고 채운 0이라 실제 값이 아니에요. 평균에 섞이면 위처럼 값이 작아지고, "마지막 값"을 꺼내면 진짜 마지막이 아닌 0을 꺼내고, Attention에선 가짜 칸에 비율을 나눠 줘요. 그래서 오늘 모든 모델에서 따로 빼요.

[그림: 길이 12, 8, 10, 5인 센서 순서 네 개를 12로 맞추며 회색 Padding을 채운 그림과, True/False로 된 4×12 Mask 행렬] 무엇을 보나: 왼쪽은 길이가 다른 센서 기록 4개를 가장 긴 12에 맞춘 모습이에요. 회색 빗금 칸이 채워 넣은 0(Padding). 오른쪽은 같은 모양의 Mask로, 회색 T(True) 칸이 정확히 왼쪽의 회색 칸과 겹쳐요. 빨간 숫자는 "0까지 넣고 평균"한 값인데, 샘플3은 진짜 평균 0.38이 0.16으로 반 토막 나요. 🤔 왜 짧을수록 더 틀려? 짧은 샘플일수록 채운 0이 많아서 평균의 분모(12)에서 가짜 칸 비중이 커져요. 그러면 모델은 "짧다"는 사실을 "값이 작다"로 착각해요. Mask로 회색 칸을 빼고 진짜 길이로 나누면 길이와 상관없이 정직한 평균이 나와요.

🔧 Mask 만들기 · Padding 칸 0으로 · pad_sequence() 명령어 사전에서 자세히 →

```
mask = torch.arange(6)[None, :] >= lengths[:, None]   # [4, 6], True = Padding 칸
padded = pad_sequence(seqs, batch_first=True)         # → [3, 5, 3] = [B, T, F]
```

🔧 시계열 창 자르기 unfold() 명령어 사전에서 자세히 →

```
window = train_s.unfold(0, 13, 1)          # [창 개수, 13] = 과거 12 + 정답 1
x = window[:, :12].unsqueeze(-1)           # [N, 12, 1]  ← F=1 축 추가
```

[그림: 사인파 센서 기록을 train, valid, test 구간으로 시간 순서대로 나누고, 과거 12개 점과 다음 1개 별로 된 창 세 개를 표시한 그림] 무엇을 보나: 사인파 센서 기록 하나(회색 선)에서 창 3개를 예로 그렸어요. 보라 점 12개 = 모델이 보는 과거, 분홍 별 = 맞혀야 할 다음 값. unfold(0, 13, 1)은 이 13칸짜리 창을 1칸씩 밀며 전부 만들어요. 배경색은 시간 순서로 먼저 자른 train/valid/test 구간이에요. 🤔 왜 먼저 자르고 창을 만들어? 창은 서로 12칸씩 겹쳐요. 섞어서 나누면 train 창의 "과거 12개" 안에 valid 창의 정답(별)이 그대로 들어가 버려요. 미래를 미리 본 모델은 시험 점수만 좋고 실전에선 틀려요.

🔧 Padding 건너뛰기 pack_padded_sequence() 명령어 사전에서 자세히 →

```
packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
out_packed, h_n = rnn(packed)                     # h_n = 샘플마다 '실제' 마지막 시점
```

## ② RNN — 메모 한 장으로 과거를 요약

RNN은 글을 읽으며 적는 메모 한 장이에요. 칸이 정해져 있어서 중요한 것만 적어요. 시점마다 "지금 입력 xt"와 "직전 메모 ht−1"를 섞어 새 메모 ht를 만들어요. 같은 Cell을 시간축에 반복 적용하는 거예요.

x₁ → h₁→ x₂ → h₂→ x₃ → h₃→ x₄ → h₄ 마지막 요약

```
h_t = tanh( W_x · x_t  +  W_h · h_(t-1)  +  b )
            └ 지금 입력 ┘   └ 직전 메모 ┘        ← 모든 시점에 같은 W_x, W_h
```

[그림: ] 무엇을 보나: RNN을 시간 방향으로 "펼쳐" 그린 그림이에요. 보라 상자 4개는 똑같은 Cell(같은 W_x, W_h)이고, 아래에서 그 시점 입력 xt, 왼쪽에서 직전 메모 ht−1을 받아 새 메모 ht를 오른쪽으로 넘겨요. 위로 나가는 화살표들이 output(모든 시점), 마지막 h₄가 h_n이에요. 🤔 왜 이렇게 생겼어? 가중치를 한 벌만 두고 반복하니 길이가 4든 100이든 같은 모델로 읽을 수 있어요. 대신 x₁의 정보가 h₄까지 가려면 상자 3개를 차례로 통과해야 해요 → 느리고(병렬 불가), 멀수록 흐려져요(③의 문제).

🤔 왜 tanh로 감싸? — 같은 계산을 수십 번 반복하니까, 값을 −1~1 사이로 눌러 두지 않으면 시점마다 커져서 폭주할 수 있어요. 그리고 비선형이 있어야 "지금 입력과 과거"를 단순 합보다 복잡하게 섞을 수 있어요.

| 결과 | 뜻 | shape (batch_first=True) |
|---|---|---|
| output | 마지막 층의 모든 시점 출력 | [B, T, D×H] |
| h_n | 모든 층·방향의 마지막 상태 | [L×D, B, H] (batch_first와 상관없이 고정) |

L = 층 수(num_layers), D = 방향 수(양방향이면 2), H = hidden_size

🤔 h에는 뭐가 들어 있어? — 지금까지 읽은 입력을 정해진 크기(hidden_size)로 요약한 값이에요. 길이가 4든 50이든 크기가 같으니 다 담지 못하고, 학습하면서 예측에 필요한 것 위주로 남아요.

[그림: 사인파 입력과, nn.RNN의 8칸 Hidden state가 시점마다 바뀌는 색 지도] 무엇을 보나: 위는 입력 사인파(18~20 시점에 일부러 튀는 값 2를 넣음), 아래는 진짜 nn.RNN(1, 8)에 넣었을 때 나온 output = 시점마다의 Hidden state 8칸이에요(빨강 +, 파랑 −). 줄 하나가 메모의 한 칸이고, 가로로 시간이 흘러요. ※ 학습 전 무작위 가중치라 각 칸의 "뜻"은 아직 없어요. 🤔 무엇을 알 수 있어? ① 입력이 오르내리면 h도 같은 주기로 출렁여요 → h는 입력을 따라가는 요약이에요. ② 튀는 구간(18~20)에서 h[4]·h[7]이 크게 변했다가, 이후 몇 시점 동안 여운이 남고 서서히 원래 리듬으로 돌아와요 → 과거가 h를 통해 다음 시점으로 전달된다는 증거이자, 시간이 지나면 흐려진다는 증거예요. ③ 값이 −1~1을 넘지 않는 건 tanh 때문이에요.

🔧 nn.RNN() — 순서 분류 명령어 사전에서 자세히 →

```
self.rnn = nn.RNN(3, 8, batch_first=True)   # F=3 → H=8
_, h_n = self.rnn(x)                        # h_n: [1, B, 8]
```

## ③ LSTM — 곱하지 말고 더해서 기억을 잇는다

RNN의 문제: 역전파가 긴 시간축을 거꾸로 지나면서 같은 W를 시점마다 계속 곱해요. 1보다 작은 수를 여러 번 곱하면 0에 가까워지듯, 먼 과거에서 오는 기울기(Gradient)가 사라져요(기울기 소실). 1보다 크면 반대로 폭발해요. 게다가 앞 시점이 끝나야 다음을 계산하니 병렬화도 어려워요.

| 곱한 횟수 | 0.9¹ | 0.9¹⁰ | 0.9²⁰ | 0.9⁵⁰ |
|---|---|---|---|---|
| 값 | 0.90 | 0.35 | 0.12 | 0.005 |

[그림: 0.5, 0.9, 0.99, 1.05를 t번 곱한 값의 로그 그래프. 0.9의 50제곱은 0.005, 0.99의 50제곱은 0.605] 무엇을 보나: 같은 수를 t번 곱했을 때 남는 값(세로축은 로그, 한 칸 = 100배). 보라 0.9는 50번 만에 0.005, 초록 0.99는 50번 곱해도 0.605가 남아요. 빨강 1.05는 반대로 100번이면 100배 이상 폭발해요. 🤔 왜 이게 기울기 이야기야? 역전파는 연쇄법칙이라 시점을 하나 거슬러 갈 때마다 "그 시점의 변화율"이 하나씩 곱해져요. RNN은 매 시점 같은 W와 tanh를 거치니 같은 크기의 수가 반복해서 곱해지는 셈 → 1보다 조금만 작아도 먼 과거의 기울기는 사실상 0이 돼서, 먼 과거 입력을 고치는 학습이 안 돼요.

LSTM의 해결: 오래 들고 가는 기억 통로 c (Cell state)를 따로 두고, 이 통로에서는 더하기로 기억을 이어요. 얼마나 지울지·쓸지·꺼낼지는 Gate(0~1 비율)가 정하고, Gate 값도 학습한 가중치로 계산해요.

```
f_t : Forget gate  — 이전 기억 c_(t-1)을 얼마나 남길지 (0~1)
i_t : Input gate   — 새 후보 g_t를 얼마나 쓸지 (0~1)
c_t = f_t ⊙ c_(t-1)  +  i_t ⊙ g_t          ← 남길 양 + 새로 쓸 양
o_t : Output gate  — h_t = o_t ⊙ tanh(c_t)  ← 지금 밖으로 꺼낼 부분

예) 이전 c 2 × forget 0.5 = 1
    새 후보 0.5 × input 0.4 = 0.2
    새 c = 1 + 0.2 = 1.2        (원리 설명용 숫자)
```

[그림: ] 무엇을 보나: LSTM Cell 하나의 내부예요. 위쪽 분홍 굵은 선이 c(Cell state) 통로, 아래에서 ht−1과 xt가 들어와 보라색 Gate 3개(f·i·o, 0~1 비율)와 후보 g를 만들어요. c 통로에서는 ① f를 곱해 지울 만큼 지우고 ② i×g를 더해 새로 적어요. 오른쪽에선 o가 tanh(c) 중 지금 꺼낼 부분을 골라 ht로 내보내요. 🤔 왜 이 모양이 기울기 소실을 막아? c 통로 위에는 큰 행렬 곱도 tanh도 없고 "f 곱하기 + 더하기"뿐이에요. 거꾸로 흐르는 기울기도 이 통로에선 f만 곱해지며 지나가니, f가 1에 가까우면 거의 줄지 않고 먼 과거까지 닿아요.

🤔 왜 LSTM이 긴 순서를 RNN보다 잘 기억해? — RNN은 매 시점 같은 W를 곱하고 tanh를 통과하니까, 50시점이면 그 효과가 50번 곱해져 거의 0이 돼요. LSTM의 c 통로는 "남길 비율 f를 곱하고 새것을 더하는" 구조라, 기울기도 이 통로를 따라 거의 f만 곱해지며 흘러가요. f는 학습되는 값이라 "이건 오래 들고 가야 해" 싶으면 1에 가깝게 배울 수 있어요 → 먼 과거까지 신호가 살아남아요. 아래에서 직접 비교해 보세요.

[그림: nn.RNN과 nn.LSTM에서 마지막 출력의 기울기가 과거 시점 입력에 얼마나 남는지 실측한 로그 그래프] 무엇을 보나: 위 슬라이더의 "곱셈 모형"이 아니라 진짜 nn.RNN·nn.LSTM에서 backward()로 잰 값이에요. 마지막 시점 출력을 각 과거 입력 xt로 미분한 크기를, 거리(가로축)별로 그렸어요(세로 로그, 학습 전, seed 20개 평균). 보라 RNN과 주황 기본 LSTM은 50칸 과거에서 1e-10 수준까지 떨어지지만, forget gate bias를 3으로 줘서 f가 약 0.95가 된 초록 LSTM은 60칸 과거까지 거의 평평해요. 🤔 왜 기본 LSTM도 떨어져? 학습 전 기본 LSTM은 f가 약 0.5(sigmoid(0))라서 c 통로에서도 매번 절반씩 곱해져요. LSTM의 장점은 "구조상 f를 1 가까이 배울 수 있다"는 것 — 학습이 오래 기억할 필요를 느끼면 초록 선처럼 바뀌어요. (그래서 실전에선 forget bias를 1 이상으로 시작하는 기법도 써요.)

| 상태 | 역할 | 비유 |
|---|---|---|
| c (Cell state) | 안에서만 오래 들고 가는 기억 | 서랍 속 노트 |
| h (Hidden state) | 그중 지금 필요한 부분을 꺼낸 출력 (output으로 나감) | 책상 위에 꺼낸 메모 |

🤔 왜 상태가 h와 c 두 개야? — "오래 보관할 것"과 "지금 내보낼 것"은 달라요. c에는 길게 보관하고, Output gate가 그중 지금 예측에 필요한 부분만 골라 h로 꺼내요. 그래서 보관과 출력을 따로 조절할 수 있어요.

💡 같은 hidden이어도 LSTM은 Gate 3개 + 후보 1개 = 가중치 네 벌이라 파라미터가 RNN의 4배예요. (F=1, H=16: RNN 16·1 + 16·16 + 2·16 = 304개 vs LSTM 4 × 304 = 1,216개)

🔧 nn.LSTM() — 과거 12개로 다음 1개 예측 명령어 사전에서 자세히 →

```
self.lstm = nn.LSTM(1, 16, batch_first=True)   # F=1 → H=16
output, _ = self.lstm(x)                       # [B, 12, 16]
```

## ④ Autoencoder — 정답이 입력 자신

RNN·LSTM은 미래를 맞혀요. Autoencoder는 목적이 달라요. 입력 x를 작게 줄였다가(z) 다시 원래대로 되살리기(x̂). 긴 글을 세 줄로 요약했다가 다시 풀어 쓰는 것과 비슷해요. 오늘은 Linear 층으로 만든 MLP형이라 시간축이 없어요.

x · 8 [B,8]→ Encoder 8 → 4 → 2→ z · 2 Latent [B,2]→ Decoder 2 → 4 → 8→ x̂ ≈ x [B,8]

|  | 예측 (LSTM) | 재구성 (Autoencoder) |
|---|---|---|
| 입력 → 출력 | 과거 12개 → 다음 1개 | x 8개 → x̂ 8개 |
| 정답 | 미래 값 | x 자신 |
| 지표 | 예측 MSE | 재구성 오차 |

| 가운데(Latent)가 | 결과 |
|---|---|
| 너무 넓으면 (모델이 너무 크면) | 그대로 베끼기만 함 → 배운 게 없고 일반화가 약해짐 |
| 적당히 좁으면 | 중요한 패턴만 골라 담음 |
| 너무 좁으면 | 중요한 정보까지 잃음 → Underfitting |

[그림: ] 무엇을 보나: 오늘 Autoencoder(8 → 4 → 2 → 4 → 8)를 막대 높이 = 칸 수로 그린 모래시계예요. 가운데 검은 2칸이 Latent z. 끝에서 나온 x̂를 처음 x와 비교한 차이가 재구성 오차예요. 🤔 왜 모래시계 모양이어야 해? 가운데가 8칸이면 "그대로 복사"로 오차 0을 쉽게 만들어요. 2칸으로 조이면 8개 숫자를 2개로 요약하는 규칙을 배워야만 되살릴 수 있어요. 그 규칙이 바로 "정상 데이터의 패턴"이고, 이 패턴에서 벗어난 입력은 2칸에 잘 안 담겨서 되살리기에 실패해요.

🤔 왜 가운데를 일부러 좁게 만들어? — 통로가 입력만큼 넓으면 "그대로 복사"가 제일 쉬운 답이라 아무 규칙도 안 배워요. 8개를 2개로 줄여야 하면, 8개 값 사이에 공통으로 반복되는 패턴을 찾아 그것만 담을 수밖에 없어요. 그게 Autoencoder가 배우는 "주요 패턴"이에요. 다만 분류 모델이 아니라서 z의 각 칸이 무슨 뜻인지는 따로 분석해야 해요.

🤔 왜 "오차 큼 = 이상"이 아니라 "이상 후보"야? — 정상 데이터로만 배운 모델은 낯선 입력을 잘 못 되살려서 오차가 커져요. 하지만 전처리 실수, 모델 용량, 학습 분포 밖의 정상 데이터 때문에도 오차가 클 수 있어서 확정할 수 없어요. 그래서 점검할 후보로만 표시해요.

[그림: 정상 샘플은 입력과 재구성이 거의 겹치고 칸별 오차가 0에 가깝지만, 이상 샘플은 재구성이 크게 어긋나 칸별 오차 막대가 큰 그림] 무엇을 보나: 사인파 조각(8칸, 진폭·위상 무작위)으로 정상만 학습한 진짜 Autoencoder(8-4-2-4-8)의 결과예요. 왼쪽 정상 샘플은 검은 입력 x와 초록 점선 x̂가 거의 겹쳐 오차 0.002, 아래 칸별 막대도 안 보일 만큼 작아요. 오른쪽 이상 샘플(일부러 만든 패턴 없는 잡음)은 x̂가 엉뚱하게 "사인파처럼" 매끈하게 나와 오차 1.171(약 600배). 🤔 왜 이상 샘플을 사인파처럼 되살려? Decoder는 2칸 z로부터 "자기가 배운 패턴(사인파)"밖에 그릴 줄 몰라요. 그래서 낯선 입력도 가장 비슷한 사인파로 바꿔 그리고, 그 차이가 그대로 오차가 돼요.

[그림: 정상 valid 400개와 이상 60개의 재구성 오차 분포 히스토그램과 99% 분위수 기준선] 무엇을 보나: 샘플마다 오차 1개(reduction="none" → mean(dim=1))를 모아 그린 분포예요(가로 로그 눈금). 초록 정상 valid는 0.001 근처에 모이고, 빨강 이상은 0.05~1로 멀리 떨어져 있어요. 보라 점선이 아래 코드의 torch.quantile(valid_err, 0.99) 기준값이에요. 🤔 왜 정상도 4개가 넘어가? "valid 오차의 99%"를 기준으로 잡으면 정의상 정상의 약 1%(400개 중 4개)는 기준을 넘어요. 기준을 높이면 놓치는 이상이, 낮추면 헛경보가 늘어요 → 몇 %로 할지는 업무가 정하고, 넘은 것은 "확정"이 아니라 "점검 후보"예요. (여기 이상 샘플은 일부러 티 나게 만든 거라 실제보다 깔끔하게 갈려요.)

⚠️ 예측 MSE와 재구성 오차는 정답도 단위도 달라요. 평균 내거나 한 표에서 순위 매기지 말고 목적별로 표를 따로 만드세요.

🔧 Autoencoder forward와 재구성 오차 명령어 사전에서 자세히 →

```
loss = nn.MSELoss()(recon, x)              # 정답 자리에 x 자신! (Batch 평균, 학습용)
score = ((recon - x) ** 2).mean(dim=1)     # 샘플별 오류 [B] (점검용)
```

[그림: 2칸 Latent z 평면에 정상 샘플이 위상 색깔 순서대로 고리 모양으로 놓이고 이상 샘플은 흩어진 산점도] 무엇을 보나: 학습된 Encoder(model[:4])로 뽑은 z 2칸을 좌표로 찍었어요. 정상 점들의 색(사인파 위상)이 무지개처럼 순서대로 고리를 이뤄요. 빨간 ✕(이상)는 고리 안쪽·바깥 등 아무 데나 흩어져요. 🤔 왜 고리야? 정상 데이터는 사실 "진폭·위상" 두 가지만 달라요. Autoencoder가 아무도 안 알려 줬는데 이 두 성질을 2칸에 펼쳐 담은 거예요. 다만 z[0]·z[1] 각각이 무슨 뜻인지는 정해져 있지 않아서 이렇게 그려 보며 해석해야 해요.

[그림: Latent 크기 1, 2, 4, 8에 따른 valid 재구성 MSE 막대. 1에서 0.032, 2부터 0.002 이하] 무엇을 보나: 가운데 칸 수만 1·2·4·8로 바꿔 학습한 valid 재구성 오차예요(앞뒤 층 16칸, 같은 데이터·epoch, seed 3개 중 최솟값). 1칸이면 0.032로 크고, 2칸부터 0.002 이하로 뚝 떨어져요. 🤔 왜 2에서 꺾여? 이 데이터의 본질이 "진폭·위상" 두 숫자라서 2칸이면 거의 다 담겨요. 그보다 줄이면 필요한 정보를 잃고(Underfitting), 늘려도 얻는 게 적고 복사에 가까워져요. 실전에선 이렇게 크기를 바꿔 가며 "꺾이는 지점"을 찾아요.

🔧 샘플별 오차 reduction="none" 명령어 사전에서 자세히 →

```
loss_fn = nn.MSELoss(reduction="none")
error = loss_fn(recon, x).mean(dim=1)      # [B, F] → [B]  샘플마다 점수 1개
```

🔧 입력 = 정답 Batch · Denoising · 기준값 명령어 사전에서 자세히 →

```
loader = DataLoader(TensorDataset(x_train), batch_size=32, shuffle=True)
threshold = torch.quantile(valid_err, 0.99)    # 예시 규칙 — 몇 %로 할지는 업무 기준으로 정함
```

## ⑤ Attention · Transformer — 필요한 위치를 직접 골라 본다

LSTM의 남은 문제: 여전히 한 시점씩 차례로 계산해서 느리고, 먼 시점 정보는 중간을 모두 거쳐 와야 해요. Attention은 순차 전달 대신 위치끼리의 관계를 한 번에 직접 가중합으로 계산해요.

|  | LSTM | Attention |
|---|---|---|
| 4번 위치가 1번을 보려면 | 1 → 2 → 3 → 4 차례로 거쳐 옴 | 4가 1·2·3을 한 번에 직접 봄 |
| 계산 | 순차 (현재 상태가 이전 상태에 의존해서 나눠 돌릴 수 없음) | 병렬 (한꺼번에) — 그래서 [B, T, E] 전체를 한 번에 받음 |

### Query · Key · Value

회의에 비유하면: 내 질문(Query, 찾을 정보)과 관련 있는 사람(Key, 비교할 위치)일수록 그 사람 말(Value, 합산할 내용)에 더 귀 기울여요.

```
weight = softmax( Q · Kᵀ / √d )     ← 어디를 얼마나 볼지 (합 = 1)
출력   = weight · V                 ← 그 비율대로 Value 섞기

예) weight [0.2, 0.3, 0.5],  Value [10, 20, 30]
    출력 = 0.2×10 + 0.3×20 + 0.5×30 = 23     (가중합 원리용 숫자)
```

[그림: ] 무엇을 보나: 위 예시 숫자를 순서대로 그린 거예요. ① Query를 각 Key와 내적해 "얼마나 관련 있나" 점수를 내고, ② softmax로 합이 1인 비율(0.2 / 0.3 / 0.5)로 바꾸고, ③ 각 위치의 Value에 그 비율을 곱해, ④ 모두 더하면 23. 🤔 왜 이렇게 해? "어디를 볼지"를 정답처럼 정해 두지 않고 점수로 부드럽게 나눠 주니 미분이 되고, 그래서 학습으로 "어떤 Key가 중요한지"를 배울 수 있어요. 결과 23은 가장 관련 높은 위치(30) 쪽으로 기운 평균이에요.

🤔 왜 √d로 나눠? — Q·K는 d개 숫자를 곱해 더한 값이라, 차원 d가 클수록 점수가 커지고 점수끼리 차이도 벌어져요. 차이가 크면 softmax가 한 위치에 거의 100%를 몰아줘서 나머지 위치는 기울기가 거의 0 → 학습이 잘 안 돼요. √d로 나누면 점수 크기가 d와 무관하게 적당해져서 골고루 보면서 배울 수 있어요. 예: 점수 [8, 4, 0] 그대로면 0.98 / 0.02 / 0.00, √16 = 4로 나눈 [2, 1, 0]이면 0.67 / 0.24 / 0.09. 아래에서 점수를 바꿔 보세요.

[그림: 점수 8, 4, 0을 그대로 softmax하면 0.98, 0.02, 0.00이고, 4로 나누면 0.67, 0.24, 0.09가 되는 막대그래프] 무엇을 보나: 같은 점수 [8, 4, 0]을 그대로 softmax(왼쪽)하면 위치1이 0.98을 독차지하고 나머지는 사실상 0, √16 = 4로 나눈 [2, 1, 0](오른쪽)은 0.67 / 0.24 / 0.09로 다른 위치도 조금씩 봐요. Value가 [10, 20, 30]이면 출력도 10.2 vs 14.2로 달라요. 🤔 왜 왼쪽이 학습에 나빠? 0.00이 된 위치는 점수를 조금 바꿔도 비율이 거의 안 변해서 기울기가 0에 가까워요. 그러면 "사실 위치3이 더 중요했다"는 걸 배울 길이 막혀요. 점수 차이를 √d로 눌러 주면 모든 위치에 기울기가 살아 있어요.

### Transformer에 더 필요한 것

한 Batch 흐름: 입력 → Embedding + Position → Self-Attention → FFN·Residual·정규화 → Pooling·Head

| 재료 | 🤔 왜 필요? |
|---|---|
| Position (위치 정보) | Attention은 위치끼리 비교만 해서 순서를 몰라요. “woman ate raccoon”과 “raccoon ate woman”을 같은 단어 묶음으로 봐요. 그래서 위치마다 다른 값을 입력에 더해 순서를 알려 줘요 |
| Padding Mask | 채운 Key를 Attention에서 빼야 가짜 칸에 비율을 나눠 주지 않아요. Pooling에서도 채운 출력은 따로 빼요 |
| Causal Mask | 다음 토큰 예측에선 미래 위치를 보면 정답을 미리 보는 셈 → 미래 위치를 가려요 |
| Multi-head | Q·K·V를 여러 갈래로 투영하면 Head마다 다른 관계(가까운 위치, 같은 종류 …)를 따로 볼 수 있어요 |
| Residual (입력 + 변환 결과) | 입력을 그대로 더해 주는 지름길이 있어서 기울기가 층을 건너 바로 흘러요 → 여러 층을 쌓아도 학습이 돼요 |
| LayerNorm | 위치 하나 안의 Feature로 정규화 → 길이·Padding이 제각각이어도 OK (BatchNorm은 여러 샘플을 묶어 정규화해서 순서 데이터엔 불편) |
| 한 층 구성 | Self-Attention(위치끼리 섞기) + Feedforward(위치마다 따로 변환), 각각 뒤에 Residual + LayerNorm. 입력·출력 크기 [B, T, E]가 같아서 여러 층을 쌓아 문맥을 강화할 수 있어요 |

[그림: 실제 4칸과 Padding 2칸으로 된 길이 6 순서의 Self-Attention 가중치 6×6 표. Mask가 없으면 pad 열에 비율이 가고, key_padding_mask를 쓰면 pad 열이 0] 무엇을 보나: 길이 6 순서(실제 x0~x3 + Padding 2칸)에 softmax(QKᵀ/√d)를 직접 계산한 Attention 가중치 표예요. 행 = 보는 주체(Query), 열 = 보이는 대상(Key), 한 행의 합 = 1. 왼쪽(Mask 없음)은 빨간 점선 안 pad 열에 행마다 0.3~0.6이나 새어 나가요. 오른쪽(key_padding_mask)은 pad 열이 전부 0.00이고 그 몫이 실제 칸으로 재분배돼요. ※ Q·K 투영은 무작위(seed 4)라 어떤 칸이 진한지에는 의미가 없어요. 🤔 왜 열만 0이고 pad 행은 남아? Mask는 "이 Key를 보지 마"라는 표시라 열을 막아요. pad 위치도 Query로서 출력은 계속 만들어지니(아래 두 줄), Pooling 때 그 출력을 따로 빼야 해요(masked_mean).

[그림: 6×6 Causal Mask. 대각선과 그 아래는 O(볼 수 있음), 대각선 위는 X(미래라서 차단)] 무엇을 보나: Causal Mask(attn_mask, [T, T])예요. 행 t3을 보면 t0~t3(초록 O)만 볼 수 있고 t4·t5(회색 X)는 막혀요. 아래로 갈수록 볼 수 있는 칸이 하나씩 늘어 계단(삼각형) 모양이 돼요. 🤔 왜 삼각형? "다음 값 맞히기"에서 t3 시점의 예측이 t4를 보면 정답을 베끼는 셈이에요. "자기 자신과 과거만"이라는 규칙을 모든 행에 적용하면 대각선 위가 전부 막힌 삼각형이 돼요. Padding Mask(열 단위)와 달리 위치 쌍 단위로 막아요.

🔧 nn.MultiheadAttention() — Self-Attention 명령어 사전에서 자세히 →

```
attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
output, weights = attn(x, x, x)                 # Self-Attention: query = key = value = x
```

🔧 nn.TransformerEncoderLayer · nn.TransformerEncoder 명령어 사전에서 자세히 →

```
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.1, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=2)   # 같은 Layer를 복제해 2층
```

🔧 Position Embedding + Padding Mask + 실제 위치만 평균 명령어 사전에서 자세히 →

```
self.pos = nn.Embedding(max_len, d_model)          # 위치 0~63 → 16칸 벡터
z = self.encoder(x + self.pos(pos), src_key_padding_mask=padding_mask)
```

## ⑥ 아키텍처 비교 — 목표·길이·자원에 맞게 고르기

점수 하나가 아니라 입력 구조와 학습 비용을 함께 비교해요.

| 항목 | RNN / LSTM | MLP형 Autoencoder | Transformer |
|---|---|---|---|
| 핵심 연결 | Hidden·Gate 반복 | Encoder–Latent–Decoder | Self-Attention |
| 주 목표 | 분류·예측 | 압축·재구성 | 문맥 분류·예측 |
| 병렬화 | 시간축 순차 | 층 내부 병렬 | 높음 |
| 길이 비용 | O(T) | 입력 차원 의존 | Attention O(T²) |
| 핵심 표현 | Hidden·Cell state | Latent z | 위치별 Embedding |
| Shape | output [B,T,H], h_n [L·D,B,H] | [B,8] → [B,2] → [B,8] | [B,T,E] → [B,T,E] → Pooling [B,E] |
| 대표 지표 | Accuracy·MSE | Reconstruction error | Accuracy·AUC·MSE |

[그림: RNN 321개, LSTM 1,233개, Autoencoder 98개, Transformer 1층 FFN 32는 3,298개, FFN 기본 2048은 69,826개인 파라미터 수 막대그래프] 무엇을 보나: 이 페이지 코드의 모델들을 실제로 만들어 sum(p.numel())로 센 크기예요(세로 로그 눈금, 한 칸 = 10배). LSTM(1,233)은 같은 hidden 16인 RNN(321)의 약 4배, Transformer는 dim_feedforward=32면 3,298개인데 기본값 2048을 그대로 두면 69,826개로 20배 넘게 커져요. 🤔 왜 이런 차이가? LSTM은 Gate 3개 + 후보 1개로 가중치가 네 벌이고, Transformer의 FFN은 d_model × dim_feedforward × 2라 중간 크기에 정비례해요. 점수가 비슷하다면 작은 쪽이 빠르고 과적합도 덜하니, 비교표에 Params를 꼭 같이 적는 이유예요.

| 비교 규칙 | 🤔 왜? |
|---|---|
| 같은 train·valid·test·Window·Feature·Target 사용 | 데이터가 다르면 점수 차이가 모델 때문인지 데이터 때문인지 알 수 없어요 |
| Epoch·Optimizer·주 지표·선택 규칙을 먼저 정함 | 결과를 보고 규칙을 고르면 원하는 모델이 이기게 맞출 수 있어요 |
| Params·Best Epoch·Valid 지표·시간·Peak Memory를 함께 기록 | 점수가 비슷하면 가볍고 빠른 쪽이 실전에선 나아요 |
| Autoencoder 재구성은 별도 표 | 목표와 단위가 다른 지표라 한 표에서 순위를 매기면 의미가 없어요 |

⚠️ 자주 하는 실수 — BTF와 TBF 축 순서 혼동 · Padding을 마지막 상태·평균 Pooling에 포함 · 시점별 예측에서 허용하지 않은 미래 관측 참조 · 목표가 다른 지표·분할을 한 표에서 직접 순위화 · h_n과 c_n을 하나로 취급 · 양방향 출력의 두 방향을 어떻게 합쳤는지 설명 안 함.

## ⑦ 길이 T · Padding · 속도 측정

| 길이 T | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| RNN·LSTM 순차 단계 | 16 | 32 | 64 | 128 |
| Attention 점수 T² | 256 | 1,024 | 4,096 | 16,384 |

🤔 왜 길어지면 Transformer가 더 무거워져? — 모든 위치 쌍을 비교하니까 길이 T면 T×T번 비교하고, 그 점수 표를 메모리에 들고 있어야 해요. 길이 2배 → 순차 단계는 2배, Attention 점수는 4배. 다만 RNN은 단계를 나눠 돌릴 수 없고 Attention은 한꺼번에 계산하니, 실제 시간은 장치·Batch·오버헤드에 따라 달라져요 — 직접 재 봐야 해요. Transformer가 항상 빠르거나 RNN이 항상 가벼운 건 아니에요.

[그림: 왼쪽은 길이 T에 따른 순차 단계 T와 Attention 점수 칸 T제곱의 로그 그래프, 오른쪽은 CPU에서 실측한 LSTM과 Transformer 순전파 시간] 무엇을 보나: 왼쪽은 계산량 "개수"예요. 두 선 모두 로그-로그라 직선인데, 보라(T²)가 주황(T)보다 두 배 가파르게 올라가요(T=512면 26만 칸). 오른쪽은 이 노트북 CPU에서 nn.LSTM과 TransformerEncoder 1층(B=32, E=16)을 실제로 잰 순전파 시간이에요. T=128 근처까지는 둘 다 1ms 아래에서 측정 잡음 속에 비슷하다가, 512에서는 Transformer가 약 10ms로 LSTM(약 2.3ms)의 4배쯤이 돼요. 🤔 왜 짧을 땐 비슷해? 짧은 길이에선 T² 표가 작아서 계산보다 함수 호출 같은 고정 비용이 시간을 차지해요. 길어질수록 T² 항이 고정 비용을 압도하면서 차이가 벌어져요. 숫자는 장치·버전마다 달라요 → 직접 재 봐야 하는 이유.

🤔 왜 Padding 비율이 높으면 손해야? — 실제 정보가 없는 칸에도 연산과 메모리를 써요. 길이가 비슷한 것끼리 묶는 Bucket, PackedSequence, Nested Tensor가 상황에 따라 이 낭비를 줄여 줘요.

🔧 한 Batch 순전파를 반복해 시간 재기 명령어 사전에서 자세히 →

```
torch.cuda.synchronize(x.device)
seconds = time.perf_counter() - start
```

## 원본 자료

- 강의: 20261102_2.pdf (Day12 딥러닝 핵심 아키텍처 및 학습, 104쪽) · 요약: Day12_개념지도.pdf

- 공식 문서: torch.nn.RNN · torch.nn.LSTM · pad_sequence · pack_padded_sequence · MultiheadAttention · TransformerEncoderLayer · Attention Is All You Need (Vaswani et al., 2017)

## 🔧 Day12 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day12)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | torch (이미 설치된 것 그대로) |
| ① 순서 데이터 · Padding · Mask · Window | pad_sequence(), masked_fill(), unfold(), pack_padded_sequence() |
| ② RNN | nn.RNN() → output, h_n |
| ③ LSTM | nn.LSTM() → output, (h_n, c_n) |
| ④ Autoencoder | nn.Sequential(nn.Linear …), nn.MSELoss(reduction="none"), TensorDataset |
| ⑤ Attention · Transformer | nn.MultiheadAttention(), nn.TransformerEncoderLayer(), nn.TransformerEncoder(), src_key_padding_mask |
| ⑥ 아키텍처 비교 | 같은 분할·지표로 Params·Best Epoch·Valid·시간·메모리 기록 |
| ⑦ 길이 T · Padding · 속도 측정 | perf_counter(), torch.cuda.synchronize(), torch.cuda.max_memory_allocated() |

### Mask 만들기 · Padding 칸 0으로 · pad_sequence()

```
import torch
x = torch.randn(4, 6, 3)                              # [B=4, T=6, F=3]
lengths = torch.tensor([6, 4, 5, 3])                  # 샘플별 실제 길이
mask = torch.arange(6)[None, :] >= lengths[:, None]   # [4, 6], True = Padding 칸
x = x.masked_fill(mask.unsqueeze(-1), 0.0)            # Padding 칸을 0으로
print(x.shape, x.dtype, lengths.shape)                # [4, 6, 3] float32 [4]

from torch.nn.utils.rnn import pad_sequence
seqs = [torch.ones(4, 3), torch.ones(2, 3) * 2, torch.ones(5, 3) * 3]   # 길이 4, 2, 5
padded = pad_sequence(seqs, batch_first=True)         # → [3, 5, 3] = [B, T, F]
valid = (~(torch.arange(5)[None, :] >= torch.tensor([4, 2, 5])[:, None])).unsqueeze(-1)
mean = (padded * valid).sum(1) / valid.sum(1)         # 실제 칸만 평균
```

▶ 실제로 돌려 보면 torch.randn(4, 6, 3) (seed 0) · 강의 sequences.json train 앞 4개

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| mask = arange(6)[None, :] >= lengths[:, None] (lengths [6, 4, 5, 3]) | 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 | 1 = Padding 칸 (실제 길이 뒤쪽) |
| x.masked_fill(mask.unsqueeze(-1), 0.0)[1] | 0.57 0.79 0.60 -1.56 -0.34 1.85 0.75 -0.59 -0.17 0.18 1.39 1.59 0.00 0.00 0.00 0.00 0.00 0.00 | 샘플 1은 길이 4 → 뒤 2줄이 0 |
| x.masked_fill(mask, 0.0) ← unsqueeze 빼면 | RuntimeError (shape [4, 6] vs [4, 6, 3]) | Mask가 마지막 F축까지 안 늘어남 |
| pad_sequence(seqs, batch_first=True).shape | [3, 5, 3] | 길이 4·2·5 → 가장 긴 5에 맞춤 |
| pad_sequence(seqs).shape ← batch_first 기본 False | [5, 3, 3] | [T, B, F] — 순서가 바뀜 |
| padded[1, :, 0] | 2 2 0 0 0 | 길이 2짜리 뒤를 0으로 채움 |
| (padded * valid).sum(1) / valid.sum(1) [:, 0] | 1 2 3 | 실제 칸만 평균 → 1 · 2 · 3 그대로 |
| padded.mean(1)[:, 0] ← Mask 없이 평균 | 0.8 0.8 3 | Padding 0이 섞여 짧은 샘플일수록 작아짐 |
| 강의 데이터: [d['length'] for d in train[:4]] | [8, 6, 12, 11] |  |
| pad_sequence(강의 4개, batch_first=True).shape | [4, 12, 3] | [B=4, T=최대 12, F=3] |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| sequences (필수) | 길이가 다른 Tensor들의 리스트 (각각 [T, F]) |  |
| batch_first | True면 [B, T, F], 기본 False면 [T, B, F] | 모델 쪽 설정과 다르면 Batch와 Time이 뒤바뀐 채로 계산됨 (오류 없이 틀릴 수 있어 위험) |
| padding_value | 채울 값 (기본 0.0) | 값을 바꿔도 Mask로 빼면 결과가 같아야 함 → Mask 검사용으로 써먹기 좋음 |
| arange(T)[None, :] >= lengths[:, None] | Broadcasting으로 [B, T] Mask 생성, True = 제외할 칸 | True/False 뜻을 반대로 쓰면 진짜 값이 빠지고 Padding만 남음 |
| masked_fill(mask, 0.0) | True 칸을 지정 값으로 | Mask를 [B,T,1]로 늘려야 F축 전체에 적용됨 (unsqueeze(-1)) |

### 시계열 창 자르기 unfold()

긴 시계열 하나를 "과거 12개 → 다음 1개" 문제 여러 개로 잘라요.

```
window = train_s.unfold(0, 13, 1)          # [창 개수, 13] = 과거 12 + 정답 1
x = window[:, :12].unsqueeze(-1)           # [N, 12, 1]  ← F=1 축 추가
y = window[:, 12:13]                       # [N, 1]
# 예: 길이 20, size=5, step=1 → Window 20 − 5 + 1 = 16개
```

▶ 실제로 돌려 보면 강의 series.csv의 train 구간(180개) · torch.arange(20)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| train_s.shape | [180] | split == 'train' 인 값만 (시간 순서 그대로) |
| window = train_s.unfold(0, 13, 1); window.shape | [168, 13] | 180 − 13 + 1 = 168개 창 |
| window[:, :12].unsqueeze(-1).shape | [168, 12, 1] | [N, T=12, F=1] |
| window[:, 12:13].shape | [168, 1] | [N, 1] 정답 |
| torch.equal(window[1, :12], train_s[1:13]) | True | 창은 1칸씩 밀림 |
| torch.arange(20.).unfold(0, 5, 1).shape | [16, 5] | 20 − 5 + 1 = 16개 |
| torch.arange(20.).unfold(0, 5, 2).shape | [8, 5] | step 2 → (20 − 5) // 2 + 1 = 8개 |
| 원본 toy[0] = 100 으로 바꾼 뒤 tw[0] | 100 1 2 3 4 | view라 창도 같이 바뀜 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| dimension (필수) | 어느 축을 따라 자를지 |  |
| size (필수) | 창 길이 (입력 12 + 정답 1) | 크면 더 먼 과거를 보지만 창 개수는 줄어듦 (길이 − size + 1개) |
| step (필수) | 창 시작점 간격 — 기본값 없음, 꼭 지정 | 크면 겹침이 줄고 창 개수도 줄어듦 |

🤔 왜 시간으로 먼저 train/valid/test를 나눈 뒤 잘라? — 통째로 잘라 놓고 섞어 나누면, 겹치는 창 때문에 valid 창의 값이 train 창 안에 그대로 들어 있어요. 모델이 "미래"를 이미 본 셈이라(누수) 점수가 부풀어요. 그리고 unfold 결과는 원본과 메모리를 공유하는 view라, 원본을 고치면 창도 같이 바뀌어요.

### Padding 건너뛰기 pack_padded_sequence()

RNN·LSTM에 "여기까지만 진짜야"라고 알려 주는 포장. 포장하면 RNN이 Padding 칸을 아예 계산하지 않아요(.data 모양은 [Σ길이, F]).

```
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
out_packed, h_n = rnn(packed)                     # h_n = 샘플마다 '실제' 마지막 시점
output, out_lengths = pad_packed_sequence(out_packed, batch_first=True)   # 다시 [B, T, H]로
```

▶ 실제로 돌려 보면 강의 sequences.json train 앞 4개 (길이 [8, 6, 12, 11], F=3) · nn.RNN(3, 8) seed 0

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| packed.data.shape | [37, 3] | Σ길이 37 칸만 담음 (Padding 칸 없음) |
| packed.batch_sizes[:3], [-3:] | [4, 4, 4], [2, 2, 1] | 시점마다 아직 살아 있는 샘플 수 |
| output.shape, out_lengths | [4, 12, 8], [8, 6, 12, 11] | 다시 [B, T, H]로 펼침 |
| torch.allclose(h_n[-1], output[range(4), lengths-1]) | True | Pack한 h_n = 샘플마다 실제 마지막 시점 |
| Pack 없이 rnn(padded) → h_n 차이 (길이 6짜리) | 0.277 | Pack 안 하면 Padding 0을 더 읽은 뒤의 상태 |
| pad_packed_sequence(…, total_length=15)[0].shape | [4, 15, 8] | 원래 T에 맞춰 펴기 |
| pack_padded_sequence(padded, lengths.cuda(), …) | RuntimeError (lengths는 CPU) | lengths는 CPU Tensor/정수 목록이어야 함 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| lengths (필수) | 샘플별 실제 길이 (CPU Tensor 또는 정수 목록) | 틀리면 진짜 값이 잘리거나 Padding이 섞임. GPU Tensor면 오류 |
| enforce_sorted | 기본 True: 길이 내림차순으로 정렬돼 있어야 함 | False면 내부에서 정렬했다가 원래 순서로 돌려줌 (편하지만 약간의 비용) |
| total_length (pad_packed) | 다시 펼 때 맞출 길이 | 안 주면 batch 안 최대 실제 길이로 펼쳐져 원래 T보다 짧을 수 있음 |

🤔 왜 Pack을 써? — Padding 칸도 RNN은 한 시점으로 계산해 버려서, 마지막 상태 h_n이 "0을 몇 번 더 읽은 뒤의 상태"가 돼요. Pack하면 샘플마다 실제 길이까지만 계산하니까 h_n이 진짜 마지막 시점 값이고, 쓸데없는 계산도 줄어요.

### nn.RNN() — 순서 분류

```
class RNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(3, 8, batch_first=True)   # F=3 → H=8
        self.fc = nn.Linear(8, 2)                   # 2클래스 Head
    def forward(self, x):
        _, h_n = self.rnn(x)                        # h_n: [1, B, 8]
        return self.fc(h_n[-1])                     # 마지막 층의 마지막 상태 → [B, 2]

x = torch.randn(4, 6, 3)                            # [B=4, T=6, F=3]
print(RNNClassifier()(x).shape)                     # torch.Size([4, 2])
# output, h_n = rnn(x, hx=None)  ← hx(초기 상태)를 생략하면 0에서 시작
```

▶ 실제로 돌려 보면 torch.randn(4, 6, 3) (seed 0) = [B=4, T=6, F=3]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| output, h_n = nn.RNN(3, 8, batch_first=True)(x) | output [4, 6, 8], h_n [1, 4, 8] | output은 모든 시점, h_n은 마지막 상태만 |
| torch.allclose(h_n[-1], output[:, -1]) | True | 1층·단방향·Padding 없음 → 같은 값 |
| RNNClassifier()(x).shape | [4, 2] | [B, 2] 점수 |
| sum(p.numel() for p in nn.RNN(3, 8).parameters()) | 104 | W_x 8×3 + W_h 8×8 + bias 8 × 2 = 104 |
| num_layers=2 → h_n.shape | [2, 4, 8] | 첫 축 = 층 수 → h_n[-1]이 마지막 층 |
| bidirectional=True → output.shape, h_n.shape | [4, 6, 16], [2, 4, 8] | 마지막 축 2H=16, h_n 첫 축 D=2 |
| nn.RNN(3, 8)(x) ← batch_first 기본 False | [4, 6, 8] | 오류 없이 [T=4, B=6]으로 읽어 버림 — 출력 모양까지 같아서 눈치채기 어려움! |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| input_size (필수) | F, 한 시점의 특징 수 | 데이터의 F와 다르면 오류 (W_x 모양이 안 맞으니까) |
| hidden_size (필수) | H, 메모(요약)의 크기 | 크면 더 많이 기억하지만 파라미터가 H²로 늘고 과적합 위험↑ |
| batch_first | True면 입력·output이 [B, T, F] (기본 False = [T, B, F]) | h_n에는 적용 안 됨 → h_n은 항상 [L×D, B, H] |
| num_layers | RNN 층을 몇 개 쌓을지 (기본 1) | h_n 첫 축이 층 수만큼 늘어나서 h_n[-1]로 마지막 층을 골라야 함 |
| bidirectional | 양방향으로 읽기 (기본 False) | D=2 → output 마지막 축이 2H, 두 방향을 어떻게 합칠지 정해야 함 |
| nonlinearity, dropout, bias | 활성화(기본 "tanh") · 층 사이 Dropout(기본 0.0) · Bias(기본 True) | dropout은 층이 2개 이상일 때만 효과 (층 "사이"에 넣으니까) |

⚠️ 분류는 h_n[-1], 예측은 output[:, -1]로 마지막 시점을 꺼내요. 1층·단방향·Padding 없음이면 둘은 같은 값. 하지만 Padding이 있으면 output[:, -1]은 Padding 칸일 수 있어요 → 실제 길이나 pack을 써야 해요.

### nn.LSTM() — 과거 12개로 다음 1개 예측

```
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)   # F=1 → H=16
        self.head = nn.Linear(16, 1)                   # 16차원 → 값 1개
    def forward(self, x):                              # [B, 12, 1]
        output, _ = self.lstm(x)                       # [B, 12, 16]
        return self.head(output[:, -1])                # 마지막 시점 [B, 16] → [B, 1]

x = torch.randn(4, 12, 1)
print(LSTMForecaster()(x).shape)                       # torch.Size([4, 1])
# output, (h_n, c_n) = lstm(x, hx=None)   ← hx=(h_0, c_0), 생략하면 둘 다 0
loss = nn.MSELoss()(pred, y)                           # 예측 [B,1] vs 다음 실제 값 [B,1]
```

▶ 실제로 돌려 보면 torch.randn(4, 12, 1) (seed 0) · 강의 series.csv 창(train 168개 · valid 48개)으로 100 step 학습

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| output, (h_n, c_n) = nn.LSTM(1, 16, batch_first=True)(x) | output [4, 12, 16], h_n [1, 4, 16], c_n [1, 4, 16] | 상태가 두 개(h, c) |
| LSTMForecaster()(x).shape | [4, 1] | [B, 1] 다음 값 하나 |
| sum(p.numel() for p in nn.LSTM(1, 16).parameters()) | 1,216 | Gate 4개 × (16·1 + 16·16 + 16 + 16) = 1,216 — RNN의 4배 |
| nn.LSTM(1, 16, proj_size=8) → h_n, c_n | [1, 4, 8] , [1, 4, 16] | h만 8로 줄고 c는 16 그대로 |
| valid MSE (학습 전) | 0.6423 | 무작위 가중치 |
| valid MSE (train 창으로 Adam 100 step 후) | 0.0002 | 다음 값을 맞히는 쪽으로 줄어듦 |
| valid MSE (그냥 '마지막 값 그대로' 예측) | 0.0261 | 비교용 기준선 — 모델이 이걸 이겨야 의미가 있음 |

| 인자 / 결과 | 뜻 | 바꾸면? |
|---|---|---|
| input_size, hidden_size, num_layers, batch_first, bidirectional, dropout | RNN과 똑같음 |  |
| proj_size | 출력 h 크기를 줄여 투영 (기본 0 = 안 함) | 0이 아니면 h_n·output 크기가 proj_size로 바뀜 (c_n은 그대로 H) |
| (h_n, c_n) | 괄호로 묶인 두 상태, 기본 둘 다 [L×D, B, H] | 하나의 Tensor처럼 다루면 오류 |
| nn.MSELoss() | 예측과 정답 차이의 제곱 평균 | 분류라면 nn.CrossEntropyLoss() — 정답이 숫자가 아니라 클래스니까 |

⚠️ 양방향 LSTM을 미래 예측에 쓸 때는 예측 시점 이후의 관측이 입력 Window에 섞이지 않았는지 확인해요. 과거 Window 안을 거꾸로 읽는 것 자체는 누수가 아니에요. 그리고 Gate 계산은 nn.LSTM이 다 알아서 해요 — 우리가 정하는 건 hidden_size와 층 수 정도.

### Autoencoder forward와 재구성 오차

```
model = nn.Sequential(
    nn.Linear(8, 4), nn.ReLU(),        # Encoder
    nn.Linear(4, 2), nn.ReLU(),        #   → Latent z (2칸)
    nn.Linear(2, 4), nn.ReLU(),        # Decoder
    nn.Linear(4, 8))                   #   → x̂ (8칸)
x = torch.randn(5, 8)
recon = model(x)
loss = nn.MSELoss()(recon, x)              # 정답 자리에 x 자신! (Batch 평균, 학습용)
score = ((recon - x) ** 2).mean(dim=1)     # 샘플별 오류 [B] (점검용)
print(recon.shape, score.shape)            # [5, 8] [5]
```

▶ 실제로 돌려 보면 torch.randn(5, 8) (seed 0), 학습 전 Autoencoder

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| recon.shape, score.shape | [5, 8], [5] | 복원은 입력과 같은 모양, 점수는 샘플마다 1개 |
| model[:4](x).shape | [5, 2] | 앞 4개 층만 = Encoder → Latent z 2칸 |
| loss.item() | 1.1870 | Batch 전체 평균 하나 |
| score | 1.742 0.331 0.397 2.167 1.299 | 샘플별 오류 — 어느 샘플이 큰지 보임 |
| torch.isclose(score.mean(), loss) | True | 샘플별 점수의 평균 = 학습용 loss |
| sum(p.numel() for p in model.parameters()) | 98 | 36 + 10 + 12 + 40 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| nn.Linear(in_features, out_features, bias=True) | 마지막 차원을 in → out으로 변환 | nn.Linear(4, 2)의 2가 Latent 크기 — 1·2·4로 바꿔 압축률과 valid 오차를 비교 |
| nn.ReLU() | 층 사이 비선형 | 모두 빼면 선형 변환만 남아 곧은 평면 위의 패턴만 담을 수 있음 |
| 마지막 nn.Linear(4, 8) 뒤 활성화 없음 | 복원 값이 음수도 될 수 있게 | 여기에 ReLU를 붙이면 음수 입력을 영영 복원 못 함 |
| nn.Sequential(*modules) | 순서대로 실행 | 중간 출력(z)은 자동으로 안 돌려줌 → model[:4](x)처럼 앞부분만 잘라 Encoder로 사용 |

### 샘플별 오차 reduction="none"

```
loss_fn = nn.MSELoss(reduction="none")
error = loss_fn(recon, x).mean(dim=1)      # [B, F] → [B]  샘플마다 점수 1개
```

▶ 실제로 돌려 보면 'Autoencoder forward' 항목의 recon, x 그대로

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| nn.MSELoss(reduction="none")(recon, x).shape | [5, 8] | 칸마다 제곱 오차 [B, F] |
| .mean(dim=1).shape | [5] | 샘플마다 하나 [B] |
| nn.MSELoss()(recon, x) (기본 "mean") | 1.1870 | 숫자 하나 |
| nn.MSELoss(reduction="sum")(recon, x) | 47.4793 | = mean × 40칸 |
| nn.MSELoss(reduction="none")(recon, x).mean(dim=0).shape | [8] | dim=0이면 특징별 오차 [F] |
| nn.MSELoss()(recon, x[:1]) ← target [1, 8] | 1.7415 + UserWarning | 에러 없이 Broadcasting으로 계산돼 버림 — 경고를 놓치면 엉뚱한 값 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| input, target (필수) | 재구성 결과, 원본 입력 (같은 Shape) | Shape가 다르면 Broadcasting으로 엉뚱하게 계산되거나 경고 |
| reduction | "none" \| "mean"(기본) \| "sum" | 기본 "mean"이면 숫자 하나로 합쳐져 어느 샘플이 이상한지 못 봄 → 점검엔 "none" |
| .mean(dim=1) | 특징(F) 축 평균 | dim=0이면 샘플 축을 평균해 특징별 오차가 됨 (목적이 다름) |
| size_average, reduce | 예전 호환용 인자 | 쓰지 말고 reduction만 사용 |

### 입력 = 정답 Batch · Denoising · 기준값

```
from torch.utils.data import TensorDataset, DataLoader
loader = DataLoader(TensorDataset(x_train), batch_size=32, shuffle=True)
for (xb,) in loader:                       # Tensor 하나면 (x,) 한 개짜리 Tuple로 나옴
    noisy = xb + 0.1 * torch.randn_like(xb)    # Denoising: 입력만 흔들고
    loss = nn.MSELoss()(model(noisy), xb)      #            정답은 깨끗한 원본

valid_err = nn.MSELoss(reduction="none")(model(x_valid), x_valid).mean(dim=1)
threshold = torch.quantile(valid_err, 0.99)    # 예시 규칙 — 몇 %로 할지는 업무 기준으로 정함
is_candidate = valid_err > threshold            # True = 점검 후보
```

▶ 실제로 돌려 보면 seed 0 합성 데이터: 2차원 잠재값을 8칸으로 늘린 train 200개 · valid 100개(앞 3개는 일부러 이상값), Denoising AE 60 epoch

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| type(batch), len(batch) (next(iter(loader))) | list, 1 | Tensor 하나만 넣어서 한 칸짜리 묶음 |
| (xb,) = batch; xb.shape | [32, 8] | 정답 Tensor가 따로 없음 — x 자신 |
| 마지막 train loss (노이즈 입력 → 깨끗한 정답) | 0.0184 |  |
| valid_err.shape | [100] | 샘플마다 점수 |
| threshold = torch.quantile(valid_err, 0.99) | 7.4796 | valid 오류의 상위 1% 경계 |
| is_candidate.sum() | 1 | 이상값을 3개 넣었는데 1개만 후보! 99% 규칙은 '100개 중 약 1개'를 고르는 규칙이라서 |
| valid_err.topk(4).indices | [1, 2, 0, 13] | 점수 순위로는 이상값 0·1·2가 1~3등 — 모델은 잘 구분, 놓친 건 기준값 규칙 |
| 정상 valid 오류 중앙값 vs 이상 3개 평균 | 0.0090 vs 7.313 | 이상값은 재구성이 훨씬 나쁨 |

| 명령 / 규칙 | 뜻 | 🤔 왜? |
|---|---|---|
| TensorDataset(*tensors) | 첫 차원 길이가 같은 Tensor들을 묶음 | AE는 정답이 x 자신이라 Tensor 하나만 넣어도 됨 → (xb,)로 꺼냄 |
| Denoising: 노이즈 입력 → 깨끗한 target | 입력과 정답의 역할을 나눔 | 정답까지 흔들면 "노이즈도 그대로 복원"을 배워 버림 |
| Scaler·Normalize는 train에서 fit | valid/test엔 같은 규칙만 적용 | valid 정보로 기준을 정하면 누수 |
| 기준값(임계값)은 valid에서, test는 마지막에 한 번 | 모델 선택 후 test 반복 조회 금지 | test를 보며 기준을 고르면 test에 맞춘 숫자가 돼서 실전 성능을 과대평가 |

### nn.MultiheadAttention() — Self-Attention

```
x = torch.randn(2, 5, 16)                       # [B, T, E]
attn = nn.MultiheadAttention(embed_dim=16, num_heads=4, batch_first=True)
output, weights = attn(x, x, x)                 # Self-Attention: query = key = value = x
print(output.shape)                             # [2, 5, 16]  문맥이 섞인 표현
print(weights.shape)                            # [2, 5, 5]   Head 평균 Attention (각 행 합 = 1)
out, _ = attn(x, x, x, key_padding_mask=mask, need_weights=False)   # mask: [B, T], True = Padding
```

▶ 실제로 돌려 보면 torch.randn(2, 5, 16) (seed 0) = [B=2, T=5, E=16]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| output, weights = attn(x, x, x) | output [2, 5, 16], weights [2, 5, 5] | weights = [B, Query T, Key T] (Head 평균) |
| weights.sum(-1)[0] | 1.000 1.000 1.000 1.000 1.000 | 각 행은 확률 → 합 1 |
| key_padding_mask (샘플 1의 뒤 2칸 True) → weights[1, 0] | 0.168 0.455 0.377 0.000 0.000 | 막은 Key 3·4에는 0 |
| need_weights=False → weights | None | 가중치를 안 돌려줌 |
| attn_mask = triu(ones(5, 5), 1) → weights[0] | 1.00 0.00 0.00 0.00 0.00 0.44 0.56 0.00 0.00 0.00 0.25 0.43 0.33 0.00 0.00 0.29 0.24 0.26 0.21 0.00 0.24 0.18 0.14 0.19 0.24 | Causal: 자기보다 뒤(미래)는 0 |
| sum(p.numel() for p in attn.parameters()) | 1,088 | Q·K·V 투영 16×48 + 48, 출력 16×16 + 16 |
| nn.MultiheadAttention(embed_dim=10, num_heads=4) | AssertionError | 10은 4로 안 나누어떨어짐 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| embed_dim (필수) | E, 입력·출력 벡터 크기 | num_heads로 나누어떨어져야 함 — E를 Head 수만큼 똑같이 쪼개 쓰니까 (16 ÷ 4 = Head당 4) |
| num_heads (필수) | 병렬 Attention Head 수 | 늘리면 다양한 관계를 보지만 Head당 차원이 작아짐 |
| batch_first | 기본 False = [T, B, E] | [B, T, E] 데이터를 False로 넣으면 B와 T가 뒤바뀐 Attention |
| key_padding_mask | [B, T], bool True = 차단 (그 Key를 보지 않음) | 빼면 Padding이 섞임. 뜻을 반대로 넣으면 진짜 값을 차단 |
| attn_mask | [T, T] 위치 쌍 차단 (Causal Mask 등) |  |
| need_weights | 가중치 반환 여부 (기본 True) | False면 weights 계산·반환을 생략해 조금 가벼워짐 |
| dropout, bias, kdim, vdim, add_bias_kv, add_zero_attn | 선택 인자 (기본 0.0, True, None, None, False, False) |  |

### nn.TransformerEncoderLayer · nn.TransformerEncoder

```
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32,
                                   dropout=0.1, batch_first=True)
x = torch.randn(2, 5, 16)
print(layer(x).shape)                     # [2, 5, 16]  입력·출력 Shape 유지
encoder = nn.TransformerEncoder(layer, num_layers=2)   # 같은 Layer를 복제해 2층
# encoder(src, mask=None, src_key_padding_mask=None, is_causal=None)
```

▶ 실제로 돌려 보면 torch.randn(2, 5, 16) (seed 0)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| layer(x).shape | [2, 5, 16] | 입력과 같은 [B, T, E] |
| encoder(x).shape (num_layers=2) | [2, 5, 16] | 몇 층을 쌓아도 Shape 유지 |
| Layer 파라미터 (dim_feedforward=32) | 2,224 | Attention 1,088 + FFN 1,072 + LayerNorm 64 |
| Layer 파라미터 (dim_feedforward 기본 2048) | 68,752 | FFN이 대부분 — d_model 16엔 너무 큼 |
| Encoder 2층 파라미터 | 4,448 | Layer를 복제해서 정확히 2배 |
| train 모드: layer(x) 두 번의 차이 | 0.838 | dropout 0.1이 매번 다르게 꺼서 결과가 흔들림 |
| eval 모드: layer(x) 두 번의 차이 | 0.000 | dropout 꺼짐 → 결정적 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| d_model, nhead (필수) | E 크기, Head 수 | d_model은 nhead로 나누어떨어져야 함 |
| dim_feedforward | Feedforward 중간 크기 (기본 2048) | d_model=16에 2048이면 FFN이 모델 대부분을 차지 → 작은 데이터면 32처럼 줄여야 과적합·계산이 줄어듦 |
| dropout | 기본 0.1 | 0.0이면 결과가 결정적이라 검사·비교에 편함 |
| activation, layer_norm_eps, norm_first, bias | 기본 "relu", 1e-5, False, True | norm_first=True면 LayerNorm을 Attention·FFN 앞에 둠 |
| batch_first | 기본 False = [T, B, E] | True로 맞춰야 [B, T, E] 그대로 넣음 |
| num_layers (Encoder, 필수) | Layer 반복 수 | 늘리면 문맥을 더 깊게 섞지만 파라미터·시간이 층 수에 비례 |
| norm, enable_nested_tensor, mask_check | 최종 LayerNorm(기본 None) · 최적화(기본 True) · Mask 검사(기본 True) | enable_nested_tensor=False면 Padding 건너뛰기 최적화를 끔 → 경고 없이 단순하게 동작 |

### Position Embedding + Padding Mask + 실제 위치만 평균

```
class TransformerClassifier(nn.Module):
    def __init__(self, max_len=64, d_model=16):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)          # 위치 0~63 → 16칸 벡터
        layer = nn.TransformerEncoderLayer(d_model, 4, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 2)

    def forward(self, x, padding_mask):                    # padding_mask: [B, T], True = Padding
        pos = torch.arange(x.size(1), device=x.device)     # [T]
        z = self.encoder(x + self.pos(pos),                # [T, E] → Batch에 Broadcasting
                         src_key_padding_mask=padding_mask)
        return self.head(masked_mean(z, padding_mask))     # [B, E] → logits [B, 2]

def masked_mean(x, padding_mask):
    if bool(padding_mask.all(dim=1).any()):
        raise ValueError("실제 길이는 1 이상")              # 전부 Padding이면 0으로 나누게 됨
    valid = (~padding_mask).unsqueeze(-1)                  # [B, T, 1], True = 실제 값
    summed = x.masked_fill(~valid, 0).sum(dim=1)
    return summed / valid.sum(dim=1)                       # 분모 = 실제 길이
```

▶ 실제로 돌려 보면 torch.randn(3, 6, 16) (seed 0), 길이 [6, 4, 2], 학습 전 모델 · eval

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| model(x, padding_mask).shape | [3, 2] | [B, 2] logits |
| Padding 칸을 99로 바꾼 뒤 출력 차이 | 0.00e+00 | ≈ 0 → Mask가 잘 걸림 (Padding 불변) |
| encoder 출력의 Padding 위치는? (99로 바꾼 뒤 차이) | 2.43 | Padding 위치 출력은 남아 있고 값도 변함 → masked_mean으로 빼야 함 |
| Position 있음: 순서를 뒤집은 뒤 출력 차이 | 0.155 | 순서가 바뀌면 결과도 바뀜 |
| Position 없음: 순서를 뒤집은 뒤 출력 차이 | 5.96e-08 | ≈ 0 → 순서를 모르는 모델 |
| masked_mean(z, 전부 True인 mask) | ValueError: 실제 길이는 1 이상 | 0 ÷ 0 = NaN을 미리 막음 |
| model.pos(torch.arange(65)) ← max_len 64 초과 | IndexError | 없는 위치 번호 |

| 인자 / 규칙 | 뜻 | 바꾸면? |
|---|---|---|
| nn.Embedding(max_len, d_model) | 학습 가능한 Position Embedding | 입력 길이 T가 max_len보다 길면 없는 위치 번호라 오류 |
| x + self.pos(pos) | 입력에 위치 표현을 더함 | 빼면 순서를 섞어도 결과가 같아짐 (순서를 모르는 모델) |
| src_key_padding_mask | [B, T], True = Padding Key 제외 | Mask를 넣어도 Padding 위치의 출력은 남아 있음 → Pooling에서 다시 빼야 함 |
| masked_mean 분모 | 실제 길이 | T로 나누면 짧은 순서일수록 평균이 작아져 길이가 점수에 섞임 |
| 길이 0 검사 | 모든 위치가 Padding인 샘플 막기 | 검사 없이 나누면 0 ÷ 0 = NaN이 학습 전체로 퍼짐 |

💡 제대로 됐는지 확인하는 법 — Padding 칸 값을 99로 바꿔도 출력이 같으면 Mask가 잘 걸린 것(Padding 불변). Position 없이 순서를 섞었는데 결과가 같으면 모델이 순서를 모른다는 뜻.

### 한 Batch 순전파를 반복해 시간 재기

```
import time, torch
# model(x, lengths) -> logits; 입력·모델 장치는 동일, 호출 전 model.eval()
with torch.inference_mode():
    for _ in range(2):                      # 워밍업 (재지 않음)
        model(x, lengths)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    start = time.perf_counter()
    for _ in range(20):
        model(x, lengths)
    if x.device.type == "cuda":
        torch.cuda.synchronize(x.device)
    seconds = time.perf_counter() - start
print("samples/s:", len(x) * 20 / seconds)  # B × 20 / seconds
peak = torch.cuda.max_memory_allocated(x.device) if x.device.type == "cuda" else "N/A"   # byte
```

▶ 실제로 돌려 보면 seed 0 난수 [64, T, 3] · 길이는 T/2~T · pack 포함 · 워밍업 2회 후 20회 · NVIDIA GeForce RTX 5080 Laptop GPU — 이 PC 기준 숫자예요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| CPU · LSTM(3→32), B=64, T=50 | 41,852 samples/s · peak N/A | CPU 메모리는 이 방식으로 안 잼 |
| GPU · 같은 조건 T=50 | 33,787 samples/s · peak 41.6 MiB | CPU의 0.8배 |
| CPU · LSTM(3→32), B=64, T=400 | 5,553 samples/s · peak N/A | CPU 메모리는 이 방식으로 안 잼 |
| GPU · 같은 조건 T=400 | 5,873 samples/s · peak 50.9 MiB | CPU의 1.1배 |
| (해석) | T가 8배 길어지면 처리량이 크게 줄어듦 | LSTM은 시점을 하나씩 차례로 계산해서 T에 비례해 느려짐. 이 작은 LSTM(H=32)·B=64에선 GPU가 CPU와 비슷하거나 오히려 느릴 때도 있어요 — 시점마다 작은 계산을 띄우는 오버헤드가 커서 |

| 명령 | 뜻 | 빼먹으면? (왜) |
|---|---|---|
| 워밍업 | 첫 실행의 준비 시간을 빼고 재기 | 첫 측정만 유난히 느리게 나옴 (초기화·메모리 할당이 섞이니까) |
| torch.cuda.synchronize(device=None) | GPU 계산이 진짜 끝날 때까지 기다림 | CUDA는 비동기라 명령만 던지고 바로 돌아와서, 시간이 실제보다 짧게 찍힘 |
| torch.cuda.reset_peak_memory_stats() | 최대 메모리 기록 초기화 | 앞서 돌린 작업의 최대값이 섞임 |
| torch.cuda.max_memory_allocated() | 구간 최대 GPU 메모리 (byte) | CPU에서는 호출하지 않고 N/A로 기록 (0이라 적으면 오해) |
| torch.inference_mode() | 기울기 기록 끄기 (forward만 잼) | 역전파용 기록까지 시간·메모리에 섞임 |
| len(x) * 20 / seconds | 초당 처리 샘플 수 (한 epoch가 아닌 20회 순전파) | 반복 수를 안 곱하면 값이 20배 작게 나옴 |

💡 공정한 비교 조건 — 같은 길이 분포·Batch size·dtype·Device·Warm-up을 쓰고, 데이터 로딩·Padding 생성·장치 전송 시간을 포함할지 측정 범위를 적어요. 결과엔 평균 시간·samples/s·Peak VRAM·하드웨어·버전·Mask·Padding 비율을 함께 남겨요. 장치(CPU/GPU) 비교는 같은 모델·가중치·입력을 유지하고, 작은 입력에선 Launch·전송 오버헤드가 계산 시간을 지배할 수 있다는 점도 같이 해석해요.

## 📖 Day12 용어 (48개)
- **MSE** (Mean Squared Error (평균 제곱 오차)): (실제 − 예측)²의 평균. 큰 오차에 더 큰 벌점을 주지만 단위가 제곱(ms²)이 돼요.
- **임계값** (Threshold (판단 기준값)): 확률이 이 값 이상(또는 초과)이면 양성(실패)이라고 부르는 선. 낮추면 놓침(FN)은 줄고 헛경고(FP)는 늘어요. valid에서만 정해요.
- **누수** (Data Leakage (데이터 누수)): 검증·테스트용 정보나 예측 시점에 알 수 없는 값(예: 실행 후에야 생기는 elapsed_ms)이 학습에 섞여 들어가 점수가 실제보다 좋게 나오는 문제.
- **AUC** (Area Under the ROC Curve (ROC 곡선 아래 넓이)): 양성 1건·음성 1건을 뽑았을 때 양성 쪽 점수가 더 높을 확률. 점수의 '순위'만 평가하므로 임계값을 바꿔도 그대로예요. 0.5는 무작위 수준, 1은 완벽. 발견 비율(Recall)이 아니에요.
- **RNN** (Recurrent Neural Network (순환 신경망)): 문장·시계열처럼 순서가 있는 데이터를 앞에서부터 차례로 읽는 신경망.
- **Transformer**: 문장 속 단어들이 서로 얼마나 관련 있는지(attention)를 계산해 처리하는 신경망 구조. 요즘 언어 모델의 기본 구조예요.
- **임베딩** (Embedding (임베딩)): 단어 같은 기호를 의미를 담은 숫자 벡터로 바꾼 것. 문장은 (batch, sequence, feature) shape가 돼요.
- **Padding** (Padding (채워 넣기)): 모자란 자리를 가짜 값(보통 0)으로 채우는 것. CNN에서는 사진 가장자리 바깥에 0을 덧대 크기를 유지하고, 순서 데이터에서는 짧은 순서 끝에 0을 채워 길이를 맞춰요. 채운 값은 진짜 데이터가 아니에요.
- **순서 데이터** (Sequence data (순서 데이터)): 값의 순서 자체가 의미를 가지는 데이터. 센서 기록·문장·로그·오디오처럼 순서를 섞으면 뜻이 달라져요.
- **시계열** (Time series (시계열)): 시간 순서대로 일정 간격으로 쌓인 값. 설비 센서값, 주가, 기온 기록 같은 것이에요. 시간 순서를 유지한 채 train/valid/test를 나눠야 해요.
- **[B, T, F]** (B(Batch, 개수) · T(Time, 시점 수) · F(Feature, 시점마다 특성 수)): 순서 데이터를 담는 Tensor 축 순서. 예: 과거 12개 값으로 다음 값을 맞히는 Window 32개면 [32, 12, 1]. PyTorch 순서 모델 기본값은 [T, B, F]라 batch_first=True로 맞춰요.
- **시점** (Time step (시점)): 순서 안의 한 칸(한 순간). 순서 모델은 시점마다 같은 계산을 반복해요. 시점의 개수가 길이 T예요.
- **Mask** (Mask (마스크)): 어느 칸을 계산에서 뺄지 True/False로 표시한 표. Padding Mask는 '여기는 채운 가짜 칸'을 표시해 평균·Attention에서 빼게 해요. PyTorch Attention에선 True = 차단이에요.
- **lengths** (lengths (샘플별 실제 길이)): Padding을 채우기 전 각 순서의 진짜 길이 목록. 이걸 함께 들고 다녀야 평균·마지막 값을 Padding 없이 계산할 수 있어요.
- **pad_sequence** (torch.nn.utils.rnn.pad_sequence): 길이가 다른 Tensor 여러 개를 가장 긴 길이에 맞춰 0으로 채워 한 묶음 Tensor로 만들어 줘요.
- **pack_padded_sequence** (torch.nn.utils.rnn.pack_padded_sequence): RNN·LSTM에 '샘플마다 여기까지만 진짜야'를 알려 주는 포장. Padding 칸을 아예 계산하지 않아서 마지막 상태 h_n이 진짜 마지막 시점 값이 돼요. 다시 펼 때는 pad_packed_sequence를 써요.
- **Window** (Sliding Window (창 자르기)): 긴 시계열을 일정 길이 구간으로 잘라 '과거 12개 → 다음 1개' 같은 문제 여러 개로 만드는 것. 창 개수 = 길이 − 창 크기 + 1(한 칸씩 옮길 때).
- **Many-to-one** (Many-to-one (여러 입력 → 출력 하나)): 순서 전체를 읽고 마지막에 클래스나 값 하나를 내는 구조. 예: 문장 감성 분류.
- **Many-to-many** (Many-to-many (여러 입력 → 시점마다 출력)): 시점마다 태그·상태·예측을 하나씩 내는 구조. 예: 시점별 상태 표시.
- **Encoder–Decoder** (Encoder–Decoder (인코더–디코더)): 입력 순서를 Encoder가 읽어 요약하고, Decoder가 그 요약으로 길이가 다른 출력 순서를 만드는 구조. 예: 번역.
- **Hidden state** (Hidden state (은닉 상태, h)): 지금까지 읽은 입력을 정해진 크기(hidden_size)로 요약한 값. 크기가 고정이라 다 담지 못하고 예측에 필요한 것 위주로 남아요. output은 모든 시점의 h, h_n은 마지막 시점의 h예요.
- **tanh** (Hyperbolic tangent (쌍곡탄젠트 함수)): 값을 −1~1 사이로 눌러 주는 활성화 함수. RNN은 같은 계산을 수십 번 반복하니 값이 폭주하지 않게 tanh로 감싸요.
- **batch_first** (batch_first (Batch 축을 맨 앞에 둘지)): True면 입력·출력을 [B, T, F] 순서로, 기본값 False면 [T, B, F]로 다뤄요. 데이터와 설정이 다르면 오류 없이 B와 T가 뒤바뀐 채 계산될 수 있어 위험해요.
- **양방향** (Bidirectional (양방향)): 순서를 앞에서 뒤로, 뒤에서 앞으로 두 번 읽어 합치는 방식. 출력 크기가 2배(2H)가 돼요. 미래 예측에 쓸 땐 예측 시점 이후 값이 입력에 섞이지 않았는지 확인해요.
- **기울기 소실** (Vanishing / Exploding Gradient (기울기 소실·폭발)): 역전파가 긴 시간축을 거꾸로 지나며 같은 가중치를 계속 곱해서, 1보다 작으면 기울기가 0에 가까워져(소실) 먼 과거를 못 배우고, 1보다 크면 폭발하는 문제. 예: 0.9⁵⁰ ≈ 0.005.
- **LSTM** (Long Short-Term Memory (장단기 메모리)): RNN에 오래 들고 가는 기억 통로(Cell state c)와 Gate를 더한 모델. 기억을 곱하지 않고 더해서 이어 가니까 긴 순서도 RNN보다 잘 기억해요. 파라미터는 RNN의 4배예요.
- **Cell state** (Cell state (셀 상태, c)): LSTM 안에서만 오래 들고 가는 기억 통로. c_t = f ⊙ c_(t−1) + i ⊙ g처럼 '남길 양 + 새로 쓸 양'을 더해 이어 가요. 밖으로 나가는 건 h예요.
- **Gate** (Gate (게이트: forget · input · output)): 정보를 얼마나 통과시킬지 0~1 비율로 정하는 장치. Forget gate는 이전 기억을 얼마나 남길지, Input gate는 새 정보를 얼마나 쓸지, Output gate는 기억 중 얼마나 꺼낼지 정해요. 비율도 학습으로 정해져요.
- **Autoencoder** (Autoencoder (오토인코더)): 입력을 좁은 통로(Latent)로 줄였다가 다시 원래대로 되살리는 모델. 정답이 미래 값이 아니라 입력 자신이에요. 좁은 통로 때문에 데이터의 주요 패턴을 배워요.
- **Encoder** (Encoder (인코더)): 입력을 읽어 더 작은(또는 문맥이 담긴) 표현으로 바꾸는 앞부분. Autoencoder에서는 x를 Latent z로 압축하고, Transformer Encoder는 문맥이 섞인 표현을 만들어요.
- **Decoder** (Decoder (디코더)): Encoder가 만든 요약 표현을 받아 원하는 출력으로 풀어내는 뒷부분. Autoencoder에서는 z를 다시 x̂(재구성)로 되돌려요.
- **Latent** (Latent representation (잠재 표현, z)): Encoder가 입력을 압축해 만든 가운데의 작은 표현. 너무 넓으면 그대로 베끼기만 하고, 너무 좁으면 중요한 정보까지 잃어요. 칸마다 무슨 뜻인지는 따로 분석해야 해요.
- **재구성 오차** (Reconstruction error (재구성 오차)): Autoencoder가 되살린 x̂와 원래 입력 x의 차이(보통 MSE). 작을수록 잘 되살린 거예요. 예측 MSE와는 정답·단위가 달라 같은 표에서 비교하지 않아요.
- **이상 후보** (Anomaly candidate (이상 후보)): 정상 데이터로만 배운 Autoencoder가 잘 못 되살려 재구성 오차가 기준값보다 큰 샘플. 전처리 실수나 낯선 정상 데이터일 수도 있어 '이상'으로 확정하지 않고 점검 후보로만 표시해요.
- **Denoising** (Denoising Autoencoder (잡음 제거 오토인코더)): 입력에만 노이즈를 섞어 넣고, 정답은 깨끗한 원본으로 두어 학습하는 Autoencoder. 노이즈를 걷어 내고 본래 모습을 되살리는 법을 배워요.
- **Attention** (Attention (어텐션, 주의 집중)): 순서대로 전달하는 대신, 모든 위치끼리의 관련도를 한 번에 계산해 필요한 위치를 비율대로 섞는 방법. weight = softmax(Q·Kᵀ/√d), 출력 = weight·V.
- **Query·Key·Value** (Query · Key · Value (질의 · 열쇠 · 값)): Attention의 세 재료. Query(찾을 정보)와 Key(비교할 위치)를 비교해 얼마나 볼지 비율을 정하고, 그 비율대로 Value(합산할 내용)를 섞어요.
- **Self-Attention** (Self-Attention (자기 어텐션)): Query·Key·Value를 모두 같은 입력에서 만들어, 한 순서 안의 위치끼리 서로를 보게 하는 Attention. 코드로는 attn(x, x, x).
- **Multi-head** (Multi-head Attention (다중 헤드 어텐션)): Q·K·V를 여러 갈래(head)로 나눠 Attention을 동시에 여러 번 하는 것. head마다 가까운 위치, 같은 종류 등 다른 관계를 따로 봐요. 차원 E가 head 수로 나누어떨어져야 해요.
- **TransformerEncoderLayer** (nn.TransformerEncoderLayer / nn.TransformerEncoder): Transformer Encoder 한 층 = Self-Attention + Feedforward, 각각 뒤에 Residual과 LayerNorm. 입력·출력 크기 [B, T, E]가 같아서 TransformerEncoder로 여러 층을 쌓아요.
- **Position Embedding** (Position Embedding / Encoding (위치 정보)): Attention은 순서를 모르니까, 위치마다 다른 값을 입력에 더해 '몇 번째 자리인지' 알려 주는 것. 학습되는 Embedding이면 입력 길이가 max_len을 넘으면 안 돼요.
- **Causal Mask** (Causal Mask (인과 마스크)): 다음 토큰을 예측할 때 미래 위치를 못 보게 가리는 Mask. 미래를 보면 정답을 미리 보는 셈이라서요.
- **토큰** (Token (토큰)): 문장 등 순서를 모델이 다루기 위해 자른 한 단위(단어·글자 조각 등). Transformer 입력의 한 위치가 보통 토큰 하나예요.
- **Residual** (Residual connection (잔차 연결)): 층의 입력을 변환 결과에 그대로 더해 주는 지름길(입력 + 변환 결과). 기울기가 이 길로 층을 건너 흘러서 여러 층을 쌓아도 학습이 돼요.
- **LayerNorm** (Layer Normalization (층 정규화)): 위치 하나 안의 Feature 값들끼리 평균·크기를 맞추는 정규화. 다른 샘플과 묶지 않아서 길이·Padding이 제각각인 순서 데이터에 잘 맞아요(BatchNorm은 여러 샘플을 묶어 정규화).
- **Feedforward** (Feedforward Network (피드포워드 층)): Transformer 층 안에서 위치마다 따로 적용하는 작은 Linear–활성화–Linear 망. Attention이 위치끼리 섞었다면, FFN은 각 위치 안을 변환해요. dim_feedforward는 중간 크기(기본 2048)예요.
- **d_model** (d_model / embed_dim (임베딩 차원 E)): Transformer에서 위치 하나를 나타내는 벡터의 길이 E. head 수(nhead)로 나누어떨어져야 해요(예: 16 ÷ 4 = head당 4).
- **O(T²)** (Big-O 표기 (길이에 따른 계산량)): 길이 T가 늘 때 계산량이 어떻게 늘어나는지 나타낸 표기. RNN·LSTM은 O(T)(길이 2배 → 2배), Attention은 모든 위치 쌍을 비교해서 O(T²)(길이 2배 → 4배)예요.

---

# Day13 · OpenCV 이미지·영상 전처리
_2026-11-04(수) · 이미지 배열 · 색공간 · Resize · Filter · Threshold · 형태학 · 영상 · 증강 · 검증_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day13.html

📌 딱 3줄 요약

- 사진은 숫자 격자: shape = (H, W, C), uint8(0~255), OpenCV 색 순서는 BGR. 크기를 줄 때만 (W, H).

- Gray → Blur → Threshold → Morphology(또는 Canny)를 지나며 숫자의 뜻이 밝기 → 경계 → 물체 → 모양으로 바뀐다. 인자는 목적에 맞게 고르고 기록한다.

- 영상은 같은 처리를 Frame마다 반복하고, 증강은 라벨 뜻이 안 바뀌는 변형을 train에만 쓰며, 결과는 원본과 다시 비교해 검증한다.

🔧 Day13 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| opencv-python | 이미지·영상 읽기/쓰기, 색 변환, 필터, 이진화, 형태학, 회전 | import cv2 |
| numpy | OpenCV 이미지의 실제 정체 (숫자 배열), 자르기·계산 | import numpy as np |
| matplotlib | 원본·결과를 나란히 그려서 비교 | import matplotlib.pyplot as plt |

### 🔧 설치 · 확인

```
python --version
python -m pip install opencv-python numpy matplotlib
python -m pip show opencv-python numpy matplotlib
python -c "import cv2; print(cv2.__version__)"      # 4.x 가 나오면 OK
```

💡 pip 이름과 import 이름이 달라요: opencv-python → import cv2. 강의는 OpenCV 4.x 공식 문서를 최종 기준으로 확인하라고 안내해요.

🤔 왜? — 원본 폴더(data/)는 건드리지 않고 결과는 outputs/에 따로 저장해요. 원본을 덮어쓰면 나중에 비교·재현·복구를 할 수 없기 때문이에요.

## 이거 어디에 써?

카메라가 남기는 건 숫자 격자뿐이에요. 컴퓨터는 "흠집"이 뭔지 몰라요. 전처리는 촬영 조건이 제각각인 사진·영상을 다음 단계(규칙 판단이나 모델)가 쓸 수 있는 입력으로 맞춰 주는 일이에요.

🏭

제품 외관 검사크기·색·잡음 조건을 맞춰 흠집 후보 찾기

📄

문서 글자 인식글자와 배경을 구분하도록 이진화

🧠

모델 입력 준비크기 224×224로 맞추고 증강 (Day14로 연결)

🎥

영상 처리Frame마다 같은 처리 후 다시 영상으로 저장

읽기·확인 imread · shape→ 밝기로 cvtColor→ 크기 맞추기 resize→ 잡음 줄이기 GaussianBlur→ 경계·영역 Canny / threshold→ 저장·재확인 imwrite · imread

| 출력 한 칸을 정할 때 보는 곳 | 예 |
|---|---|
| 자기 칸만 | Gray 변환, Threshold |
| 주변 칸까지 | Blur, Canny, Morphology |
| 다른 자리에서 | Resize, 회전(warpAffine) |

🤔 왜 이렇게 나눠 봐? — 어떤 칸을 보는지가 곧 그 처리의 성격이에요. "자기 칸만" 보는 처리는 잡음인지 아닌지 판단할 수 없고, "주변 칸까지" 보는 처리는 그걸 판단할 수 있는 대신 세부가 흐려져요. 그래서 순서와 인자를 목적에 맞게 골라야 해요.

## ① 이미지 = 숫자 배열

영상에서 한 장을 꺼내면 이미지(Frame), 이미지는 위치별 숫자의 배열이에요. 컬러 사진은 칸(Pixel) 하나에 숫자 3개(B, G, R), 각 0~255. Gray는 [H, W], 컬러는 [H, W, C].

| 항목 | 뜻 | 예 | 확인 코드 |
|---|---|---|---|
| shape | (세로 H, 가로 W, 채널 C) | (240, 320, 3) | img.shape |
| dtype | 숫자의 자료형 | uint8 | img.dtype |
| min / max | 값 범위 | 0 / 255 | img.min(), img.max() |
| ndim | 차원 수 (Gray는 2, 컬러는 3) | 3 | img.ndim |

한 칸 값은 img[y, x]로 봐요 — 세로(y)가 먼저. 예: [20, 80, 200]은 B=20, G=80, R=200 → 빨강 쪽 색. shape가 (345, 450, 3)이면 450은 Width(열의 수)예요.

[그림: 원본 이미지의 빨간 사각형 모서리 6×6칸을 확대해 칸마다 B·G·R 숫자를 적은 그림] 무엇을 보나: 왼쪽 노란 네모 6×6칸을 확대하면 칸마다 숫자가 3개(B, G, R) 들어 있어요. 빨간 칸은 B 0 · G 0 · R 255, 배경 칸은 세 값이 비슷한 회색(약 63·70·78). 세로 눈금이 y(행), 가로 눈금이 x(열)이에요. 🤔 왜? 사진은 처음부터 "빨강"이라는 뜻이 아니라 이 숫자 격자로 저장되기 때문이에요. 그래서 shape = (240, 320, 3)은 "240줄 × 320칸 × 숫자 3개"이고, 한 칸을 꺼낼 땐 줄 번호(y)부터 img[y, x]로 써요.

🤔 왜 shape부터 확인해? — shape를 모르고 처리하면 채널 축과 공간 축을 뒤바꾸거나, 잘못된 크기로 저장할 수 있어요. 그리고 uint8은 0~255만 담을 수 있는 자료형이라 더하기·빼기를 함부로 하면 값이 넘쳐요(⑥ 밝기 조절에서 다시 나와요). 그래서 읽자마자 shape · dtype · min/max를 찍어 봐요.

🤔 왜 세로(y)가 먼저? — 이미지는 NumPy 배열이고, 배열은 "몇 번째 줄(행), 몇 번째 칸(열)" 순서로 셉니다. 줄 = 세로 위치(y), 칸 = 가로 위치(x)라서 (H, W), img[y, x]가 돼요. 참고로 Day11 PyTorch Conv2d는 [N, C, H, W](채널이 앞)라서, 모델에 넣을 땐 축 순서를 또 바꿔야 해요.

[그림: 색 사각형, 원, 사각형, 흰 선이 있는 원본 이미지] 원본 (240, 320, 3) — BGR을 제대로 표시 무엇을 보나: 위쪽 색 칸이 왼쪽부터 빨강·초록·파랑, 배경은 왼쪽이 어둡고 오른쪽이 밝아요. 아래 장의 모든 변환이 이 사진에서 출발해요. 🤔 왜 이런 그림? 색(BGR 순서)·밝기 변화(그림자)·가는 줄무늬(확대/축소)·직선(경계)을 한 장에서 다 확인하려고요.

[그림: 빨강과 파랑이 뒤바뀐 이미지] BGR 배열을 변환 없이 RGB로 표시 무엇을 보나: 왼쪽 칸이 빨강 → 파랑, 오른쪽 칸이 파랑 → 빨강으로 바뀌었어요. 초록은 그대로예요. 🤔 왜? 숫자는 하나도 안 바뀌었고 읽는 순서만 바뀌었어요. 첫 번째 칸(OpenCV에선 B)을 Matplotlib은 R로 읽으니까 파랑과 빨강만 자리를 바꾸고, 가운데 G는 둘 다 같은 자리라 그대로예요.

⚠️ OpenCV(imread)는 B, G, R, Matplotlib(imshow)는 R, G, B. Matplotlib로 볼 땐 cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 후 표시, RGB 배열을 imwrite로 저장할 땐 다시 BGR로 되돌려요.

🤔 왜 OpenCV는 BGR일까? — 강의 PDF는 "OpenCV 기본 색상 순서는 BGR"이라는 사실만 알려 줘요. 흔히 알려진 배경은, OpenCV가 처음 만들어질 무렵 카메라 장비·Windows 이미지 형식이 BGR 순서로 저장하는 경우가 많았고, 그 뒤로는 기존 코드와의 호환 때문에 바꾸지 않았다는 거예요. 이유가 무엇이든 실전 규칙은 하나: 라이브러리가 바뀌는 경계에서 색 순서를 꼭 맞춘다.

🔧 읽기 cv2.imread() · 저장 cv2.imwrite() 명령어 사전에서 자세히 →

```
img = cv2.imread("data/scene.png", cv2.IMREAD_COLOR)   # 실패해도 에러 없이 None
cv2.imwrite("outputs/copy.png", img)                   # 성공 여부를 True/False로
```

🔧 화면에 띄우기 cv2.imshow() · cv2.waitKey() 명령어 사전에서 자세히 →

```
cv2.imshow("result", img)    # 운영체제 창에 표시
cv2.waitKey(0)               # 0 = 키를 누를 때까지 무기한 대기
```

🔧 채널 나누기·합치기 cv2.split() · cv2.merge() 명령어 사전에서 자세히 →

```
b, g, r = cv2.split(img)          # 각각 [H, W]
merged = cv2.merge([b, g, r])     # 원래대로 [H, W, 3]
```

[그림: 원본과 B, G, R 채널을 각각 흑백으로 표시한 4장의 그림] 무엇을 보나: split한 채널 하나하나는 흑백 사진((240, 320))이에요. 노란 네모의 빨간 사각형은 R 채널에서만 흰색(255), B·G에서는 검정(0). 하늘색 원은 B·G에서 밝고 R에서 어두워요. 🤔 왜? 각 채널은 "그 색이 얼마나 섞였나"만 담은 밝기 지도이기 때문이에요. 하늘색 = 파랑 + 초록이라 B·G가 밝고, 주황 = 빨강 + 약간의 초록이라 R이 가장 밝아요. 채널 순서를 바꾸면(BGR↔RGB) 이 지도가 다른 색 자리에 들어가서 위의 "빨강↔파랑 뒤바뀜"이 생겨요.

## ② 색공간 · 크기 맞추기

색공간은 "정답"이 아니라 목적에 따라 고르는 표현이에요.

| 목적 | 표현 | 변환 코드 | 결과 shape |
|---|---|---|---|
| Matplotlib에 색 표시 | RGB | COLOR_BGR2RGB | (H, W, 3) |
| 밝기·경계 분석 (Threshold, Edge) | Gray | COLOR_BGR2GRAY | (H, W) |
| 특정 색 범위 고르기 | HSV (uint8: H 0~179, S·V 0~255) | COLOR_BGR2HSV | (H, W, 3) |
| Gray 결과를 컬러 영상에 저장·그리기 | BGR | COLOR_GRAY2BGR | (H, W, 3) |

🤔 왜 경계·이진화는 Gray로? — 경계는 "밝기가 갑자기 바뀌는 곳"이라 밝기 하나면 충분해요. 색 3개(예: B52·G136·R214)를 섞어 밝기 1개(150)로 만들면 계산할 숫자가 1/3로 줄고, Threshold도 기준값 하나로 끝나요.

🤔 왜 색 고르기는 HSV로? — BGR에서는 그늘이 지면 B·G·R 세 값이 모두 바뀌어서 "빨강"을 범위로 잡기 어려워요. HSV는 색 종류(H), 진하기(S), 밝기(V)를 따로 떼어 두니까, 조명이 바뀌어도 주로 V만 변하고 H는 비교적 그대로예요. 참고로 uint8에는 0~255만 들어가서 360°짜리 색상환을 반으로 줄여 H는 0~179예요.

[그림: 왼쪽이 그늘진 사진의 H, S, V 채널과 BGR 범위 마스크, HSV 범위 마스크 비교] 무엇을 보나: 왼쪽으로 갈수록 그늘을 만든 사진이에요. V(밝기) 채널은 왼쪽이 어둡게 변했지만, H(색 종류) 채널에서 주황 사각형은 전체가 고른 회색이에요. 그래서 BGR 범위로 찾으면 밝은 오른쪽 일부(2,707칸)만 잡히고, HSV의 H 범위로 찾으면 사각형 전체(10,724칸)가 잡혀요(흰 선이 지나가는 자리만 빠짐). 🤔 왜? 그늘은 "색 종류"가 아니라 "밝기"를 바꾸기 때문이에요. BGR에서는 그늘이 B·G·R 세 값을 모두 끌어내려 범위를 벗어나지만, HSV는 그 변화가 거의 V에만 모이니 H로 고른 범위가 그대로 유지돼요.

⚠️ GRAY2BGR은 같은 밝기 값을 3채널로 복제할 뿐, 원래 색을 되살리지 못해요. 🤔 왜? — Gray로 바꿀 때 색 3개를 섞어 하나로 만들면서 "어떤 색이었는지" 정보가 버려졌기 때문이에요. 그래도 쓰는 이유는 컬러 VideoWriter나 컬러 도형 그리기가 3채널을 요구하기 때문이에요.

🔧 cv2.cvtColor() 명령어 사전에서 자세히 →

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # (240, 320)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # (240, 320, 3)
```

[그림: 흑백으로 바뀐 이미지] COLOR_BGR2GRAY → (240, 320) 무엇을 보나: 색이 사라지고 밝기만 남았어요. 빨강·초록·파랑 칸이 서로 다른 회색이 됐고(초록이 가장 밝아요), 배경의 왼쪽→오른쪽 밝기 변화는 그대로예요. shape에서 채널 축 3이 없어졌어요. 🤔 왜 색마다 회색이 달라? 단순 평균이 아니라 0.299R + 0.587G + 0.114B로 섞어서예요. 사람 눈이 초록에 가장 민감해서 초록에 가장 큰 무게를 줘요. 경계·이진화·OCR은 색보다 밝기 차이가 중요해서 Gray로 바꾸고 시작해요.

### 크기 바꾸기와 보간법

줄일 땐 여러 픽셀을 하나로 요약하고, 키울 땐 없던 픽셀 값을 주변에서 추정해요. 이 추정 방법이 보간법.

[그림: 계단처럼 각진 확대 이미지] 작게 줄인 뒤 다시 확대 · INTER_NEAREST 무엇을 보나: 원·선의 가장자리가 계단처럼 각지고, 오른쪽 위 줄무늬가 뭉텅이로 보여요. 🤔 왜? 새로 생긴 픽셀에 가장 가까운 원래 픽셀 값을 그대로 복사해서예요. 한 픽셀이 여러 칸으로 똑같이 늘어나니 네모난 덩어리가 보여요. 대신 값을 섞지 않아서 마스크(0/1 라벨)를 키울 때 써요.

[그림: 부드럽게 확대된 이미지] 같은 확대 · INTER_LINEAR 무엇을 보나: 왼쪽과 같은 크기로 키웠는데 가장자리가 부드럽게 번진 모습이에요. 🤔 왜? 새 픽셀 값을 주변 4개 픽셀의 거리 가중 평균으로 계산해서, 경계에 중간 밝기가 생겨요. 사진은 자연스러워지지만, 마스크에 쓰면 0과 1 사이 "애매한 값"이 생겨 라벨이 깨져요.

| 보간법 | 언제 | 🤔 왜? |
|---|---|---|
| INTER_AREA | 축소에 자주 사용 | 줄어드는 칸이 덮는 면적의 평균을 내서, 몇 칸을 건너뛰며 고르는 방식보다 잔무늬·깨짐이 적음 |
| INTER_LINEAR | 기본값, 확대에도 사용 | 가까운 4칸을 거리에 따라 섞어 값이 부드럽게 이어짐 |
| INTER_NEAREST | 0/255 Mask에 적합 | 가장 가까운 값을 그대로 복사해서 중간값(예: 128)이 안 생김 → Mask 뜻이 유지됨. 대신 계단 현상 |

[그림: 촘촘한 동심원 무늬를 NEAREST, LINEAR, AREA로 축소한 비교] 무엇을 보나: 바깥으로 갈수록 줄무늬가 촘촘해지는 그림을 1/5로 줄였어요. NEAREST·LINEAR는 원본에 없던 가짜 동심원·바둑판 무늬가 생기고, AREA는 너무 촘촘한 곳이 차분한 회색이 돼요. 🤔 왜? NEAREST·LINEAR는 5칸 중 몇 칸만 "골라서" 보는데, 그 칸이 하필 흰 줄·검은 줄 어느 쪽에 걸리느냐에 따라 엉뚱한 무늬가 생겨요. AREA는 덮는 면적을 모두 평균하니 흰·검 줄이 섞여 회색이 돼요. 그래서 축소엔 AREA.

[그림: 0과 255만 있는 마스크를 NEAREST와 LINEAR로 확대하고 한 줄의 값을 그래프로 그린 그림] 무엇을 보나: 0/255 Mask를 8배 키웠을 때 NEAREST는 값 종류가 그대로 2개, LINEAR는 45개예요. 오른쪽 그래프에서 초록(NEAREST)은 0→255로 계단처럼 뛰고, 빨강(LINEAR)은 경사로 이어져 128 같은 중간값이 생겨요. 🤔 왜? LINEAR는 이웃 값을 거리에 따라 섞어서 경계에 "반쯤 물체"인 값을 만들어요. Mask에서 128은 물체인지 배경인지 뜻이 없으니, Mask는 가장 가까운 값을 그대로 복사하는 NEAREST로 키워요.

🔧 cv2.resize() 명령어 사전에서 자세히 →

```
small = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
half  = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
```

## ③ Filter · Edge — 주변 칸까지 보고 계산

칸 하나만 봐서는 그게 잡음인지 알 수 없어요. 필터는 Kernel(작은 창)을 옮기며 주변 값을 섞어 새 값을 만들어요. 같은 위치끼리 곱한 뒤 모두 더해요.

```
입력 3×3 이웃          평균 Kernel
10  10  10            1/9 1/9 1/9
10 100  10     ×      1/9 1/9 1/9     →  출력 = (10×8 + 100) ÷ 9 = 20
10  10  10            1/9 1/9 1/9         (튀는 점 100이 20으로 누그러짐)
```

💡 이 계산은 Day11 CNN의 합성곱(Conv2d)과 같은 계산이에요. 오늘은 사람이 Kernel 값을 정하고, CNN은 그 값을 배워요.

| Kernel 설정 | 효과 | 🤔 왜? |
|---|---|---|
| 작은 Kernel | 세부 보존, 효과 약함 | 섞는 이웃이 적으니 원래 값이 많이 남음 |
| 큰 Kernel | 더 강한 Blur, 세부 손실·연산 증가 | 먼 칸까지 섞어서 작은 무늬가 평균에 묻히고, 곱셈 횟수도 늘어남 |
| 합이 1인 평균 Kernel | 밝기 평균화 (Blur) | 가중치 합이 1이라 전체 밝기는 그대로 유지 |
| 음수 가중치 포함 | 경계·선명도 강조 (Sharpen) | 주변을 빼서 "나와 이웃의 차이"가 커지는 곳이 강조됨 |

[그림: GaussianBlur 커널 크기 3, 7, 15를 나란히 비교한 그림] 무엇을 보나: 같은 사진에 Kernel만 3 → 7 → 15로 키웠어요. 노란 네모 안의 가는 줄무늬가 3에선 그대로, 7에선 흐려지고, 15에선 회색 덩어리가 돼요. 색 사각형·원 테두리도 점점 번져요. 🤔 왜? Kernel이 크면 더 먼 칸까지 섞기 때문이에요. 줄무늬 간격보다 Kernel이 넓어지면 흰 줄과 검은 줄이 한 평균 안에 같이 들어가 회색이 돼요 — 잡음이 줄어드는 것과 세부가 사라지는 것은 같은 원리예요.

| 필터 | 방식 | 장점 | 주의 |
|---|---|---|---|
| GaussianBlur | 가운데에 더 큰 가중치를 준 평균 | 자연스러운 Blur | Edge도 흐려짐 |
| medianBlur | 이웃 값 중 중앙값 선택 | 점 잡음(소금·후추)에 강함 | ksize는 1보다 큰 홀수 |
| bilateralFilter | 거리 + 밝기 차이까지 고려 | Edge 보존 | 비교적 느림 |

🤔 왜 Gaussian은 가운데를 크게? — 가까운 칸일수록 나와 비슷한 값일 가능성이 높아요. 그래서 가운데에 큰 가중치를 주면 평균 필터보다 덜 뭉개지고 자연스럽게 흐려져요.

🤔 왜 소금·후추 잡음엔 Median? — 점 잡음은 주변과 동떨어진 극단값(0 또는 255)이에요. 평균은 극단값 하나에도 끌려가서 흐린 점이 남지만, 중앙값은 값을 줄 세운 뒤 가운데 것을 고르니 극단값이 아예 뽑히지 않아요. 아래 체험에서 직접 확인해 보세요.

🤔 왜 Bilateral은 경계를 지켜? — 거리만이 아니라 밝기 차이로도 가중치를 줘서, 경계 건너편처럼 밝기가 크게 다른 칸은 거의 섞지 않아요. 대신 칸마다 가중치를 새로 계산해야 해서 느려요.

[그림: 흰 점과 검은 점 잡음이 많은 이미지] 점 잡음이 낀 원본 무엇을 보나: 원본 위에 흰 점·검은 점이 무작위로 흩뿌려졌어요(소금·후추 잡음). 🤔 왜 이런 잡음? 센서 불량·전송 오류처럼 픽셀 몇 개가 극단값(0 또는 255)으로 튀는 상황을 흉내 낸 거예요. 오른쪽 두 필터의 차이가 이 잡음에서 확 드러나요.

[그림: 가우시안 블러 결과] GaussianBlur((5,5), 0) 무엇을 보나: 점들이 흐릿한 얼룩으로 번졌지만 여전히 남아 있고, 도형 가장자리도 같이 흐려졌어요. 🤔 왜? 5×5 주변을 평균 내는 방식이라, 255짜리 흰 점 하나가 주변 값을 끌어올려 얼룩으로 퍼져요. 극단값을 "버리지" 못하고 "섞기"만 해요.

[그림: 미디언 블러 결과] medianBlur(5) 무엇을 보나: 점 잡음이 거의 사라졌고, 도형 가장자리도 비교적 또렷해요. 🤔 왜? 5×5 주변 25개 값을 줄 세워 가운데 값을 고르는 방식이라, 튀는 값 몇 개는 줄 끝으로 밀려나 아예 뽑히지 않아요. 그래서 소금·후추 잡음엔 Median이 강해요.

[그림: 가우시안 잡음과 소금후추 잡음에 각각 Gaussian, Median, Bilateral 필터를 적용한 2×4 비교] 무엇을 보나: 윗줄은 고르게 퍼진 잡음, 아랫줄은 점 잡음이에요. 각 그림 위 PSNR은 깨끗한 원본과 얼마나 비슷한지(클수록 비슷)이고, 초록 테두리가 그 줄의 1등이에요. 퍼진 잡음엔 Bilateral(28.9dB), 점 잡음엔 Median(31.2dB)이 이겨요. Bilateral은 점 잡음을 거의 못 지워요(23.6dB). 🤔 왜? 점 잡음은 0·255 극단값이라 Median은 아예 뽑지 않지만, Bilateral은 "밝기가 크게 다른 칸은 안 섞는다"는 규칙 때문에 튀는 점을 경계로 착각해 보존해요. 반대로 퍼진 잡음은 작은 차이라 Bilateral이 섞어 주면서 진짜 경계는 지켜요. 그래서 "잡음 종류부터 확인"하라는 거예요.

💡 가장자리(Border) — Kernel이 이미지 밖으로 나가면 실제 픽셀이 없어요. BORDER_REFLECT(경계값을 거울처럼 반사), BORDER_REPLICATE(가장자리 픽셀 반복) 등 규칙에 따라 가장자리 결과가 달라져요. 🤔 왜 규칙이 필요? — 바깥 칸 값을 "뭔가로" 정해야 곱하고 더할 수 있으니까요. 0(검정)으로 채우면 가장자리가 어둡게 번질 수 있어요.

🔧 cv2.GaussianBlur() · cv2.medianBlur() · cv2.bilateralFilter() 명령어 사전에서 자세히 →

```
gaussian  = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
median    = cv2.medianBlur(gray, ksize=5)
```

### Canny Edge — "여기가 경계인가?"

밝기가 갑자기 바뀌는 곳만 흰색(255)으로 남겨요. 강한 경계(threshold2 이상)는 확실히 남기고, 약한 경계(threshold1~threshold2)는 강한 경계에 연결된 것만 남기는 방식이에요.

🤔 왜 기준값이 두 개? — 하나만 쓰면 낮게 잡을 땐 잡음까지 경계가 되고, 높게 잡을 땐 선이 군데군데 끊겨요. 두 개를 쓰면 "확실한 선"에 이어진 약한 부분은 살리고, 외따로 떨어진 약한 점(잡음)은 버릴 수 있어요.

[그림: 잡음 때문에 점이 많은 경계 이미지] Blur 없이 Canny(70, 150) · 경계 픽셀 12,138개 무엇을 보나: 도형 윤곽 말고도 잡음 점마다 작은 동그라미가 그려져 화면이 지저분해요. 🤔 왜? 잡음 점 하나도 "주변보다 밝기가 확 다른 곳"이라 Canny 기준을 넘어요. Canny는 그게 잡음인지 진짜 테두리인지 구분하지 못해요.

[그림: 깔끔한 경계 이미지] GaussianBlur(5,5) 후 같은 Canny · 3,528개 무엇을 보나: 기준값은 왼쪽과 똑같은데 경계가 약 1/3로 줄고, 도형 윤곽 위주로 남았어요(작은 점 몇 개는 아직 있어요). 🤔 왜? Blur가 잡음 점의 밝기 튐을 주변과 섞어 작게 만들어서, 기준(70/150)을 못 넘게 됐어요. 도형 경계는 원래 차이가 커서 흐려져도 살아남아요.

🤔 왜 Canny 전에 Blur? — 잡음 점 하나하나도 "주변과 밝기가 갑자기 다른 곳"이라 Canny 눈에는 경계로 보여요. 먼저 Blur로 잡음을 누그러뜨리면 잡음으로 생기는 불필요한 Edge가 줄어요. (개념 지도 예시도 Blur 없이 1,996칸 → Blur 후 915칸)

🔧 cv2.Canny() 명령어 사전에서 자세히 →

```
blur  = cv2.GaussianBlur(gray, (5, 5), 0)     # 잡음 Edge를 줄이려고 먼저 Blur
edges = cv2.Canny(blur, threshold1=70, threshold2=150)
```

[그림: 같은 블러 이미지에 Canny 기준값을 (20,50), (70,150), (200,400)으로 바꾼 비교] 무엇을 보나: 입력은 같고 기준값만 바꿨어요(보기 쉽게 검은 선 = 경계). 낮은 (20, 50)은 11,413칸 — 잡음 결까지 경계, (70, 150)은 3,528칸 — 도형 윤곽 위주, 높은 (200, 400)은 1,567칸 — 원·사각형 경계가 사라지고 아주 강한 줄무늬·흰 선만 남아요. 🤔 왜? 기준값은 "밝기가 얼마나 급하게 바뀌어야 경계로 인정하나"예요. 낮추면 작은 흔들림(잡음)도 통과하고, 높이면 대비가 약한 진짜 경계(어두운 배경 위 원)도 탈락해요. 가장 좋은 값은 목적에 따라 달라서 결과를 눈으로 비교하며 고르고 기록해요.

## ④ Threshold · 형태학 — "여기가 물체인가?"

Threshold는 기준값 T로 칸마다 0(배경) / 255(물체)를 정해요. 입력은 보통 Gray 1채널이고, 결과 Binary는 뒤의 형태학 처리에 쓰기 좋아요. 아래 슬라이더로 T를 바꿔 보세요.

🤔 왜 이진화가 필요해? — 밝기 150은 "얼마나 밝은가"일 뿐, "여기가 물체인가?"에는 답하지 못해요. 기준을 정해 예/아니오(255/0)로 바꿔야 물체 영역의 넓이를 세거나 모양을 다듬는 규칙을 만들 수 있어요.

```
THRESH_BINARY, T = 128
Gray 값 :  40   110   160   220
결과    :   0     0   255   255
```

| 방법 | 기준값 정하는 법 | 장점 | 주의 |
|---|---|---|---|
| Fixed | 사람이 값 하나 지정 | 단순·빠름 | 조명 변화에 취약 |
| Otsu | 밝기 히스토그램으로 자동 결정 | 물체·배경 밝기가 두 무리로 갈릴 때 유용 | Blur가 도움될 수 있음 |
| Adaptive | 지역마다 따로 계산 | 조명이 고르지 않아도 대응 | blockSize·C 조정 필요 |

🤔 왜 Fixed는 조명에 약해? — 조명이 밝아지면 물체·배경 밝기가 함께 올라가요. T=128로 고정해 두면 어제는 배경이던 곳이 오늘은 물체가 될 수 있어요. 사진마다 사람이 T를 다시 정할 수도 없고요.

[그림: 보통 날과 어두운 날의 밝기 히스토그램과 고정 T=128 이진화 결과] 무엇을 보나: 같은 장면을 어둡게(×0.55) 찍으면 히스토그램의 두 봉우리(물체·배경)가 통째로 왼쪽으로 이동해요. 보통 날엔 T=128이 두 봉우리 사이라 물체(26%)를 잘 잡지만, 어두운 날엔 두 봉우리가 모두 128 왼쪽이라 화면 100%가 물체로 판정돼요. 🤔 왜? T는 "밝기 숫자"에 고정된 선인데, 조명은 물체와 배경의 숫자를 함께 움직이기 때문이에요. 선은 그대로인데 데이터가 이사를 가 버린 셈 — 그래서 히스토그램을 보고 T를 따라 움직이는 Otsu가 필요해요.

🤔 왜 Otsu? — 밝기 히스토그램이 "어두운 무리 + 밝은 무리" 두 봉우리로 갈려 있으면, Otsu는 두 무리가 각각 가장 똘똘 뭉치도록(무리 안 퍼짐이 가장 작도록) 하는 T를 자동으로 찾아요. 그래서 사진마다 T를 손으로 고르지 않아도 되고, 조명이 바뀌어도 T가 따라 움직여요. 잡음이 봉우리를 흐리면 잘 안 되니 먼저 Blur하면 도움이 돼요.

🤔 왜 Adaptive? — 한쪽만 밝게 비춘 사진처럼 조명이 고르지 않으면 사진 전체에 맞는 T 하나가 없어요. Adaptive는 칸마다 주변 blockSize×blockSize 영역의 평균(또는 가우시안 평균)에서 C를 뺀 값을 그 칸만의 기준으로 써요.

[그림: 두 봉우리 히스토그램 위에 무리 안 퍼짐 곡선과 Otsu가 고른 T=125를 표시한 그림] 무엇을 보나: 보라 막대는 밝기 히스토그램(왼쪽 봉우리 = 어두운 물체, 오른쪽 = 밝은 배경), 주황 곡선은 T를 0~255로 옮겨 보며 계산한 "무리 안 퍼짐"이에요. 곡선이 가장 낮은 곳 = Otsu T = 125, 두 봉우리 사이 골짜기예요. 🤔 왜? T를 봉우리 한가운데에 두면 한 무리가 둘로 쪼개져 각 무리 안의 값이 넓게 퍼져요. 골짜기에 둘 때 두 무리가 각각 가장 똘똘 뭉치니 퍼짐이 최소가 돼요. Otsu는 이 계산을 자동으로 해서 retval로 돌려줘요.

[그림: 오른쪽에서만 빛이 든 문서를 Fixed, Otsu, Adaptive로 이진화한 비교] 무엇을 보나: 오른쪽만 밝은 문서예요. Fixed(T=128)와 Otsu(T=111)는 사진 전체에 기준값이 하나라 어두운 왼쪽 글자가 배경과 함께 검게 묻혀요. Adaptive는 글자 세 줄이 모두 살아남아요(배경에 작은 점 몇 개는 남음). 🤔 왜? 왼쪽의 "밝은 종이"가 오른쪽의 "검은 글자"보다 어두울 정도로 조명 차이가 크면, 어떤 T 하나로도 둘을 동시에 가를 수 없어요. Adaptive는 칸마다 주변 31×31 평균 − 10을 기준으로 삼으니 "주변보다 어두운가?"만 묻게 되어 조명 차이가 상쇄돼요.

🔧 cv2.threshold() · cv2.adaptiveThreshold() 명령어 사전에서 자세히 →

```
retval, binary = cv2.threshold(gray, thresh=140, maxval=255, type=cv2.THRESH_BINARY)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, ...)
```

### 형태학(Morphology) — 물체 모양 다듬기

이진화 후엔 흰 먼지·작은 검은 구멍이 남아요. Kernel로 흰 영역(전경)을 깎거나 부풀려서 정리해요. Erosion(침식)은 Kernel 안이 모두 흰색일 때만 흰색을 남기고, Dilation(팽창)은 Kernel 안에 하나라도 흰색이 있으면 흰색으로 만들어요.

| 연산 | 순서 | 효과 | 예 |
|---|---|---|---|
| Erosion (침식) | 1회 | 흰 영역 축소 | 붙은 물체 분리 |
| Dilation (팽창) | 1회 | 흰 영역 확대 | 끊긴 부분 연결 |
| Opening | 침식 → 팽창 | 작은 흰 잡음 제거 | 먼지 제거 |
| Closing | 팽창 → 침식 | 작은 검은 구멍 메우기 | 문자 끊김 보완 |

🤔 왜 Opening은 "침식 → 팽창" 순서? — 먼저 깎으면 Kernel보다 작은 흰 먼지는 통째로 사라져요. 사라진 건 다시 부풀려도 돌아오지 않죠. 반면 큰 물체는 조금 깎였다가 팽창으로 원래 크기로 돌아와요. Closing은 정반대: 먼저 부풀려 작은 검은 구멍을 메우고, 다시 깎아 바깥 크기를 되돌려요.

[그림: 작은 격자 위 이진 도형에 Erosion, Dilation, Opening, Closing을 적용하고 바뀐 칸을 색으로 표시한 그림] 무엇을 보나: 한 칸 = 픽셀 하나. 빨강은 원본보다 깎인 칸, 청록은 새로 생긴 칸이에요. Erosion은 먼지 3개와 물체 테두리 한 겹을, Dilation은 구멍·틈과 바깥 한 겹을 바꿔요. Opening은 먼지만 빨강, Closing은 구멍·틈만 청록 — 물체 크기는 그대로예요. 🤔 왜? 3×3 Kernel 안이 모두 흰색이어야 남는 Erosion은 1칸짜리 먼지를 통째로 지우고, 하나라도 흰색이면 칠하는 Dilation은 1~2칸 구멍을 메워요. 두 연산을 반대 순서로 이어 붙이면 큰 물체는 "깎였다 돌아오고(또는 부풀었다 돌아오고)" 작은 것만 바뀌어요.

[그림: 흰 물체에 흰 먼지와 검은 구멍이 있는 이진 이미지] 이진화 결과 · 흰 먼지 + 검은 구멍 무엇을 보나: 흰 물체 3개(사각형·원·막대) 말고 배경에 흰 먼지 점이, 물체 안에 검은 구멍 점이 있어요. 🤔 왜 이런 그림? 실제 이진화 결과에 흔한 두 가지 흠이에요. 먼지는 Opening, 구멍은 Closing이 담당한다는 걸 옆 세 장에서 비교해요.

[그림: 오프닝 결과] Opening만 (침식 → 팽창) 무엇을 보나: 배경의 흰 먼지는 사라졌는데, 물체 안 검은 구멍은 그대로예요. 🤔 왜? 침식(깎기)으로 Kernel보다 작은 흰 점이 통째로 없어지고, 팽창으로 큰 물체만 원래 크기로 돌아와요. 검은 구멍은 "흰 것"이 아니라서 이 연산의 대상이 아니에요.

[그림: 클로징 결과] Closing만 (팽창 → 침식) 무엇을 보나: 반대로 물체 안 구멍은 메워졌는데, 배경 먼지는 남아 있어요. 🤔 왜? 먼저 부풀리면 흰 영역이 작은 검은 구멍을 덮어 버리고, 다시 깎아도 메워진 구멍은 돌아오지 않아요. 흰 먼지는 부풀었다 다시 깎여 원래대로 남아요.

[그림: 오프닝 후 클로징 결과] Opening → Closing 무엇을 보나: 먼지도 구멍도 없이 물체 3개만 깔끔하게 남았어요. 🤔 왜? 두 연산이 맡는 흠이 달라서 이어서 쓰면 둘 다 정리돼요. 단, Kernel보다 작은 진짜 물체(작은 결함)도 같이 지워질 수 있어요(아래 ⚠️).

⚠️ 작은 흰 점이 진짜 결함이라면? Opening(3×3 Kernel)이 그것까지 지울 수 있어요. 🤔 왜? — Opening은 "작은 흰 것 = 잡음"이라고 가정하는 연산이라, 작은 결함과 먼지를 구별하지 못해요. 그래서 원본·단계별 결과·설정을 나란히 남겨 무엇을 잃었는지 확인해요.

🔧 cv2.getStructuringElement() · cv2.morphologyEx() 명령어 사전에서 자세히 →

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
opened = cv2.morphologyEx(binary, op=cv2.MORPH_OPEN, kernel=kernel, iterations=1)
```

## ⑤ 영상 — 같은 처리를 Frame마다

영상은 시간 순서로 이어진 Frame(이미지)의 묶음. 그래서 영상 처리 = 이미지 처리 + 시간 순서 + 저장 설정. 한 장 처리 함수를 그대로 재사용해요.

read() Frame 한 장 읽기→ process_frame() Gray · Resize · Edge→ write() 출력 영상에 추가↻ 끝나면 release()

| 속성 | 뜻 | 확인 |
|---|---|---|
| Frame | 시간 t의 한 장 | cap.read() |
| FPS | 초당 Frame 수 (재생 속도) | cap.get(cv2.CAP_PROP_FPS) |
| Frame Size | (width, height) | CAP_PROP_FRAME_WIDTH / HEIGHT |
| Frame 수 | 전체 장수 | CAP_PROP_FRAME_COUNT |
| Codec | 압축·저장 방식 | fourcc |

```
재생시간 = Frame 수 ÷ FPS   (일정한 FPS를 가정)
48장 ÷ 12FPS = 4초      30장 ÷ 30FPS = 1초      15장 ÷ 30FPS = 0.5초
```

💡 저장 FPS(재생 빈도)와 연산 처리속도(초당 몇 장 계산했나)는 다른 것. 🤔 왜 구분? — 처리가 느려도 저장 FPS가 30이면 영상은 30FPS로 재생돼요. 반대로 같은 Frame 수인데 저장 FPS를 절반으로 하면 재생시간이 2배가 되어 느리게 보여요.

### 모든 Frame을 처리할까, 일부만 할까?

🔧 Frame 건너뛰며 처리 frame_idx % step 명령어 사전에서 자세히 →

```
    ok, frame = cap.read()
        result = process_frame(frame)
```

🤔 왜 건너뛰어? 왜 조심해? — 옆 Frame끼리는 거의 똑같아서, 건너뛰면 처리량이 크게 줄어요. 하지만 그 사이에 지나간 빠른 사건을 놓칠 수 있고, 줄어든 Frame을 원래 FPS로 저장하면 재생시간도 짧아져요. 시간 기준이 필요하면 CAP_PROP_POS_MSEC이나 FPS로 Timestamp를 계산해요.

[그림: 48개 Frame 타임라인에서 5장마다 처리하는 Frame과 놓친 빠른 사건을 표시한 그림] 무엇을 보나: 수업 예제 영상(48장, 12FPS = 4초)의 Frame을 한 칸씩 늘어놓았어요. 진보라 10칸이 step=5로 처리하는 Frame이고, 아래 썸네일처럼 옆 Frame끼리는 거의 같아요. 빨간 테두리 #12~14에만 있는 빠른 사건은 처리 Frame(#10, #15) 사이에 끼어 한 장도 안 뽑혀요. 🤔 왜? 건너뛰기는 "옆 Frame은 비슷하다"는 가정에 기대요. 그 가정 덕에 처리량이 1/5로 줄지만, step보다 짧게 지나가는 사건에서는 가정이 깨져서 놓쳐요. 그래서 step은 잡아야 할 사건의 길이(초 × FPS)보다 작게 정해요.

🔧 읽기 cv2.VideoCapture() · 저장 cv2.VideoWriter() 명령어 사전에서 자세히 →

```
cap = cv2.VideoCapture("data/sample.avi")
writer = cv2.VideoWriter("outputs/edge.avi", fourcc, fps, size)
```

## ⑥ 데이터 증강 — 정답은 그대로, 사진만 다양하게

학습 사진을 뒤집고·돌리고·밝게 해서(Flip·Crop·Rotation·Brightness·Blur) 모델이 다양한 조건을 미리 보게 해요. 단, 라벨 뜻이 바뀌면 안 돼요.

🤔 왜 증강? — 실제 현장 사진은 조금 기울거나, 밝거나 어둡게 찍혀요. 학습 자료에 그런 조건이 없으면 모델은 "똑바르고 밝은 사진"만 알게 돼요. 증강은 새로 찍지 않고도 학습 자료의 조건 범위를 넓혀 줘요.

[그림: 좌우 반전된 이미지] flip(img, 1) 좌우 반전 무엇을 보나: 색 칸 순서가 파랑·초록·빨강으로 뒤집히고, 줄무늬가 왼쪽 위로, 밝은 배경이 왼쪽으로 갔어요. 🤔 왜 증강에 써? 고양이는 왼쪽을 봐도 오른쪽을 봐도 고양이라, 라벨은 그대로 두고 사진만 두 배로 늘릴 수 있어요. 글자·좌우가 의미 있는 표지판엔 쓰면 안 돼요.

[그림: 10도 회전된 이미지] 10° 회전 무엇을 보나: 그림 전체가 비스듬해지고, 네 모서리에 검은 삼각형(빈 곳)이 생겼어요. 원래 모서리 일부는 잘려 나갔고요. 🤔 왜 빈 곳이 생겨? 사각형을 돌리면 원래 크기 틀에 다 안 들어가요. 틀 안인데 원본에 해당하는 픽셀이 없는 자리는 채울 값이 없어서 기본값 0(검정)이 돼요. 모델이 이 검은 삼각형을 특징으로 배우지 않도록 각도를 작게 쓰거나 borderMode로 채워요.

[그림: 밝기를 올린 이미지] 밝기 +30 (beta=30) 무엇을 보나: 전체가 조금 밝아졌고, 특히 왼쪽 어두운 배경이 눈에 띄게 밝아졌어요. 원래 흰 선은 더 밝아질 수 없어 그대로예요. 🤔 왜? 모든 픽셀에 30을 더하지만 uint8은 255가 끝이라 넘는 값은 255에서 멈춰요(convertScaleAbs가 잘라 줘요). 조명이 다른 날 찍힌 사진을 흉내 내는 증강이에요.

| 과제 | 써도 되는 변형 | 조심할 변형 |
|---|---|---|
| 일반 사물 분류 | 좌우 반전, 밝기, 작은 Crop | 큰 회전, 과도한 Blur |
| 문자 OCR | 작은 회전, 명암, 노이즈 | 좌우·상하 반전 (b → d) |
| 교통 표지 | 밝기, 날씨 효과 | 좌우 반전이 뜻을 바꿀 수 있음 |
| 의료 영상 | 도메인 기준에 따른 제한적 변형 | 임의 색상·방향 변경 |

🤔 왜 과제마다 달라? — 컵은 뒤집어도 컵이지만, 글자 b를 좌우로 뒤집으면 d가 되고, "좌회전" 표지를 뒤집으면 "우회전"이 돼요. 변형 후에도 정답 라벨이 그대로인가?가 유일한 기준이에요.

[그림: 컵, 글자 b, 좌회전 표지를 좌우 반전한 결과 비교] 무엇을 보나: 세 그림에 똑같이 flip(img, 1)을 했어요. 컵은 손잡이 방향만 바뀌어 여전히 컵(O), 글자 b는 d가 되고, 좌회전 화살표는 우회전이 돼요(X). 🤔 왜? 증강은 "같은 정답의 다른 사진"을 만드는 일이라, 변형 후 라벨이 바뀌면 모델에게 틀린 정답을 가르치는 셈이에요. 방향 자체가 뜻인 글자·표지에서는 반전이 바로 그 뜻을 뒤집어요.

원본 기준으로 먼저 분할→ train 기본 전처리 + 무작위 증강/ valid · test 정해진 전처리만 (고정)

⚠️ valid/test에는 무작위 증강을 하지 않아요. 🤔 왜? — 평가 입력이 실행마다 달라지면 점수 차이가 모델 때문인지 증강 운 때문인지 구분할 수 없어요. 그리고 원본 단위로 먼저 분할해야 같은 원본의 증강본이 train과 test에 갈라져 들어가는 일이 없어요.

🤔 왜 Seed·설정을 기록? — 무작위 변형은 seed가 같아야 같은 결과가 다시 나와요. 변환 설정(dict/JSON)·seed·호출 순서·버전을 남기면 같은 실험을 재현하고, 설정 차이가 결과에 준 영향을 분리할 수 있어요. 원본은 덮어쓰지 말고, 결과 파일명에 변환 정보를 남기세요.

🔧 반전 cv2.flip() 명령어 사전에서 자세히 →

```
horizontal = cv2.flip(img, flipCode=1)    # 좌우
vertical   = cv2.flip(img, flipCode=0)    # 상하
```

🔧 회전 getRotationMatrix2D() + warpAffine() 명령어 사전에서 자세히 →

```
M = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=10, scale=1.0)   # 2×3 행렬
rotated = cv2.warpAffine(img, M, dsize=(w, h), borderMode=cv2.BORDER_REFLECT)
```

🔧 밝기 · Crop · Blur 명령어 사전에서 자세히 →

```
bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)    # 밝기 +30
crop = img[h//10 : h - h//10, w//10 : w - w//10]         # NumPy 자르기 [y, x]
```

[그림: uint8에 30을 더했을 때 넘침과 convertScaleAbs의 잘라내기를 비교한 그래프와 띠] 무엇을 보나: 왼쪽 그래프: 원래 밝기(가로) → 바꾼 뒤 밝기(세로). 빨강(img + 30)은 226에서 갑자기 0으로 떨어지는 톱니, 초록(convertScaleAbs)은 255에서 평평하게 멈춰요. 오른쪽 띠에서 빨강 방법은 가장 밝던 끝이 검게 변했어요. 🤔 왜? uint8은 0~255만 담는 칸이라 256 이상은 256을 뺀 나머지만 남아요(시계가 12 다음에 1로 돌아가듯). convertScaleAbs는 계산 후 범위를 벗어난 값을 0과 255로 잘라 주니 "밝게"라는 의도가 지켜져요.

## ⑦ 일괄 처리 · 검증 — 파일이 열린다고 끝이 아니다

프로젝트에서는 이미지 폴더와 영상을 같은 설정으로 처리하고, 원본 대비 결과를 수치와 그림으로 검증해요. 파일이 열리더라도 작업에 필요한 경계·글자·관심 영역이 남아 있어야 해요.

🔧 폴더 일괄 처리 sorted(Path.glob()) 명령어 사전에서 자세히 →

```
for path in sorted(src_dir.glob("*.png")):       # 순서 고정
    img = cv2.imread(str(path))
```

### 검증표와 테스트 상황

| 검증 항목 | 원본 예시 | 결과 예시 | 판정 |
|---|---|---|---|
| 파일 수 | 10 | 10 | 일치 |
| Shape | 다양함 | 240×320 | 설정 일치 |
| dtype | uint8 | uint8 | 유지 |
| 값 범위 | 0~255 이내 | 0~255 이내 | 범위 확인 |
| 읽기 재확인 | 가능 | 가능 | 정상 |
| 영상 (시간) | Frame 수 · FPS · 재생시간 | 입력과 비교 |  |

| 상황 | 기대 결과 | 확인 |
|---|---|---|
| 정상 PNG | 읽기·처리·저장 성공 | shape·dtype·재읽기 |
| 다른 크기 이미지 | 설정 크기로 통일 | 원본 비율·왜곡 확인 |
| Gray 입력 | 채널 처리 분기 | BGR 변환 필요 여부 |
| 빈 영상 / 손상 파일 | 명확한 실패 기록 | 출력 생성 금지 |
| FPS 0 또는 미확인 | 대체 규칙 또는 중단 | 보고서 기록 |
| 출력 폴더 없음 | 폴더 생성 후 저장 | 파일 수 확인 |

🤔 왜 Shape가 같아도 안심하면 안 돼? — Opening이 진짜 결함을 지웠거나, Threshold가 글자를 날려도 shape는 똑같아요. 그래서 원본·단계별 결과·설정을 함께 남기고 그림으로 직접 비교해서, 어떤 정보를 잃었는지와 처리의 한계를 설명해요.

[그림: Opening 전후 이진 이미지와 사라진 픽셀을 빨갛게 표시한 차이 그림] 무엇을 보나: 처리 전후 모두 shape (60, 90), uint8로 숫자 검증은 통과해요. 하지만 차이 그림(빨강 = 사라진 칸)을 보면 Opening이 작은 흰 결함 3개(12칸)를 지웠어요. 흰 칸 수도 2,357 → 2,345로 줄었어요. 🤔 왜? shape·dtype은 "그릇 모양"만 확인하고, 안에 무엇이 남았는지는 말해 주지 않아요. 원본과 결과를 빼서(차이) 그려 보거나 흰 칸 수 같은 내용 숫자를 함께 기록해야 정보 손실이 드러나요.

⚠️ 코드 리뷰 체크리스트

- 읽은 직후 None·shape·dtype을 확인했나?

- BGR/RGB와 (H, W)/(W, H)를 혼동하지 않았나?

- 필터·Threshold·Morphology 설정을 코드나 JSON에 남겼나?

- 원본을 보존하고 출력 폴더를 분리했나?

- 영상 Writer의 FPS·Frame Size·채널이 일치하나?

## 원본 자료

- 강의: Day13_20261104.pdf (p.2–11 큰 그림, p.18–35 배열·색공간·Resize, p.36–54 Filter·Threshold·형태학, p.55–73 영상·증강, p.74–89 프로젝트·검증)

- 요약: Day13_개념지도.pdf · OpenCV_치트시트.pdf

- 공식 문서: OpenCV-Python Tutorials · Smoothing Images · Thresholding · Morphological Transformations

## 🔧 Day13 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day13)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install opencv-python numpy matplotlib |
| ① 이미지 = 숫자 배열 | cv2.imread(), cv2.imwrite(), img.shape, cv2.split() / merge(), cv2.imshow() |
| ② 색공간 · 크기 | cv2.cvtColor(), cv2.resize() |
| ③ Filter · Edge | cv2.GaussianBlur(), cv2.medianBlur(), cv2.bilateralFilter(), cv2.Canny() |
| ④ Threshold · 형태학 | cv2.threshold(), cv2.adaptiveThreshold(), cv2.getStructuringElement(), cv2.morphologyEx() |
| ⑤ 영상 Frame 처리 | cv2.VideoCapture(), cv2.VideoWriter(), cap.get(CAP_PROP_FPS) |
| ⑥ 데이터 증강 | cv2.flip(), cv2.getRotationMatrix2D() + cv2.warpAffine(), cv2.convertScaleAbs() |
| ⑦ 일괄 처리 · 검증 | sorted(Path.glob()), imread(path, 0) 재읽기, 검증표 |

### 읽기 cv2.imread() · 저장 cv2.imwrite()

```
from pathlib import Path
import cv2

img = cv2.imread("data/scene.png", cv2.IMREAD_COLOR)
if img is None:                              # 읽기 실패는 에러가 아니라 None!
    raise RuntimeError("image read failed")
print(img.shape, img.dtype, img.min(), img.max())

Path("outputs").mkdir(parents=True, exist_ok=True)
if not cv2.imwrite("outputs/copy.png", img):  # 성공 여부를 bool로 돌려줌
    raise OSError("image save failed")
```

▶ 실제로 돌려 보면 data/scene.png 기준

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| img = cv2.imread("data/scene.png") | img.shape → (240, 320, 3), dtype → uint8 | 세로 240 · 가로 320 · 채널 3(BGR) |
| img.min(), img.max() | (0, 255) | uint8이라 0~255 안 |
| cv2.imread("data/scene.png", cv2.IMREAD_GRAYSCALE).shape | (240, 320) | 채널 축이 없어짐 |
| cv2.imread("data/rgba.png", cv2.IMREAD_UNCHANGED).shape | (240, 320, 4) | 투명도(Alpha) 포함 4채널 |
| cv2.imread("data/rgba.png").shape | (240, 320, 3) | 기본 COLOR는 Alpha를 버림 |
| cv2.imread("data/없는파일.png") | None | 에러가 아니라 None! |
| cv2.imwrite("outputs/copy.png", img) | True | 성공하면 True |
| cv2.imwrite("outputs/없는폴더/a.png", img) | False | 폴더가 없으면 False (에러 아님) |

🤔 왜 None 검사를 직접 해? — imread는 경로가 틀려도 에러를 내지 않고 None을 돌려줘요. 검사하지 않으면 몇 줄 뒤 cvtColor에서 엉뚱한 에러가 나서 원인을 찾기 어려워요. imwrite도 실패하면 False만 돌려주니 반환값을 확인합니다.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| filename (필수) | 파일 경로 | 틀리면 imread는 None — 에러가 안 나니 직접 확인 |
| flags=cv2.IMREAD_COLOR | BGR 3채널로 읽기 (기본) | 원본이 흑백·투명이어도 3채널로 맞춰 줘서 뒤 처리가 단순해짐 |
| cv2.IMREAD_GRAYSCALE (또는 0) | 처음부터 흑백 [H, W]로 읽기 | 채널이 1개라 메모리·계산이 줄어듦. 저장한 Gray 결과를 다시 읽어 확인할 때도 사용 |
| cv2.IMREAD_UNCHANGED | 파일 그대로 (투명도 Alpha 채널 유지 가능) | 기본 COLOR는 Alpha를 버리기 때문에, PNG의 4번째 채널이 필요하면 이걸 씀 |
| imwrite의 확장자 | 저장 형식(인코더)을 결정 | .png는 손실 없음, .jpg는 손실 압축 → 0/255 Mask를 jpg로 저장하면 중간값이 생김 |
| imwrite의 params=[] | (선택) 압축 품질 등 인코더 옵션 |  |

### 화면에 띄우기 cv2.imshow() · cv2.waitKey()

```
cv2.imshow("result", img)    # 운영체제 창에 표시
cv2.waitKey(0)               # 0 = 키를 누를 때까지 무기한 대기
cv2.destroyAllWindows()

# 창이 안 열리는 환경이면: 저장하거나 RGB로 바꿔 Matplotlib로
import matplotlib.pyplot as plt
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.savefig("outputs/view.png")
```

▶ 실제로 돌려 보면 창을 띄우는 명령이라 반환값만 정리

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv2.imshow("result", img) | None | 창에 그리기만 하고 돌려주는 값 없음 |
| cv2.waitKey(0) | 누른 키의 번호 (예: q → 113, Esc → 27) | 키 입력 전까지 멈춤 |
| cv2.waitKey(1) | 1ms 안에 키가 없으면 -1 | 영상 반복에서 매 Frame 잠깐만 기다릴 때 |

⚠️ 원격 서버·Notebook·일부 VS Code 환경에서는 GUI 창이 안 열릴 수 있어요. 🤔 왜? — imshow는 운영체제의 창을 띄우는 기능이라, 화면(디스플레이)이 없는 서버에서는 띄울 곳이 없거든요. 그럴 땐 cv2.imwrite()로 저장하거나 Matplotlib로 그려요.

### 채널 나누기·합치기 cv2.split() · cv2.merge()

```
b, g, r = cv2.split(img)          # 각각 [H, W]
merged = cv2.merge([b, g, r])     # 원래대로 [H, W, 3]
print("same:", (merged == img).all())
rgb = cv2.merge([r, g, b])        # 순서를 바꿔 RGB로
```

▶ 실제로 돌려 보면 빨간 사각형 안의 한 칸 img[35, 40]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| img[35, 40] | [ 0 0 255] | B 0 · G 0 · R 255 |
| b, g, r = cv2.split(img) | b.shape → (240, 320) | 채널마다 2차원 [H, W] |
| b[35, 40], g[35, 40], r[35, 40] | (0, 0, 255) |  |
| (cv2.merge([b, g, r]) == img).all() | True | 넣은 순서 그대로면 원본과 같음 |
| cv2.merge([r, g, b])[35, 40] | [255 0 0] | 순서를 바꾸면 RGB 배열 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| split(src) | 채널별 2차원 배열의 튜플 | 채널 하나만 보고 싶을 때 (예: 빨강 채널 밝기만) |
| merge(mv) | H·W·dtype이 같은 채널 배열들을 하나로 합침 | 넣는 순서 = 채널 순서라, [r, g, b]로 넣으면 RGB 배열이 됨 |

### cv2.cvtColor()

```
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # (240, 320)
rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # (240, 320, 3)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # (240, 320, 3)
```

▶ 실제로 돌려 보면 같은 빨간 칸 [35, 40]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).shape | (240, 320) | 채널 축이 사라짐 |
| gray[35, 40] | 76 | 0.299×255 ≈ 76 (빨강만 있으니 R 무게만) |
| cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[35, 40] | [255 0 0] | B와 R 자리만 바뀜 |
| hsv[35, 40] | [ 0 255 255] | H 0(빨강) · S 255(진함) · V 255(밝음) |
| cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)[35, 40] | [76 76 76] | 같은 밝기를 3번 복사 — 색은 안 돌아옴 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| src (필수) | 입력 이미지 | 입력 채널 수가 code와 안 맞으면 에러 (예: Gray에 BGR2GRAY) |
| code (필수) | 변환 종류 COLOR_입력2출력 | 이름 그대로 읽으면 됨: BGR → GRAY |
| dstCn=0 | (선택) 출력 채널 수, 0이면 자동 | 보통 그대로 둠 |

### cv2.resize()

```
small = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
half  = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
# 320×240 이미지를 dsize=(160, 120)으로 → shape는 (120, 160, 3)
```

▶ 실제로 돌려 보면 scene.png (240, 320, 3) · mask_truth.png

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv2.resize(img, dsize=(224, 224)).shape | (224, 224, 3) | 4:3 → 1:1로 찌그러짐 |
| cv2.resize(img, dsize=(160, 120)).shape | (120, 160, 3) | dsize는 (W, H), shape는 (H, W)! |
| cv2.resize(img, None, fx=0.5, fy=0.5).shape | (120, 160, 3) | 비율 유지하며 절반 |
| np.unique(mask) | [ 0 255] | 원본 마스크 (240, 320): 값 2종류 |
| len(np.unique(resize(mask, fx=8, INTER_NEAREST))) | 2 | 값 종류 그대로 |
| len(np.unique(resize(mask, fx=8, INTER_LINEAR))) | 75 | 경계에 중간값이 생김 → 마스크엔 NEAREST |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| src (필수) | 입력 이미지 |  |
| dsize (필수 자리) | 출력 크기 (W, H), 배율을 쓸 땐 None 또는 (0, 0) | 가로세로 비율이 원본과 다르면 찌그러짐 (320×240 → 224×224는 4:3 → 1:1) |
| fx, fy (기본 0) | 가로·세로 배율 | 둘 다 0.5면 비율 유지하며 절반. dsize와 동시에 쓰기보다 한 방식만 골라야 코드가 명확 |
| interpolation | 보간법 (기본 INTER_LINEAR) | 축소는 AREA, Mask는 NEAREST — 이유는 Day13 보간법 표 참고 |

⚠️ 제일 많이 하는 실수: shape는 (H, W), dsize는 (W, H). 순서가 반대예요. 🤔 왜 반대? — shape는 NumPy 배열의 "줄 수, 칸 수"이고, dsize는 OpenCV의 "이미지 크기" 표기(가로×세로, 예: 320×240)를 따르기 때문이에요. 서로 다른 관례가 한 코드에서 만나니 헷갈리는 거예요.

### cv2.GaussianBlur() · cv2.medianBlur() · cv2.bilateralFilter()

```
gaussian  = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
median    = cv2.medianBlur(gray, ksize=5)
bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
```

▶ 실제로 돌려 보면 noisy_gray.png를 깨끗한 원본(clean_gray.png)과 비교 — 평균 오차(작을수록 원본에 가까움)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| noisy (필터 전) | 2.91 | 소금·후추처럼 튀는 점 잡음 |
| cv2.GaussianBlur(noisy, (5, 5), 0) | 5.37 | 오히려 커짐 — 튀는 점을 주변에 번지게 하고 경계도 흐림 |
| cv2.medianBlur(noisy, 5) | 0.33 | 가장 작음 — 정렬한 가운데 값이라 튀는 점이 그냥 버려짐 |
| cv2.bilateralFilter(noisy, 9, 75, 75) | 3.60 | 조금 커짐 — 튀는 점도 '경계'로 보고 지켜 버림 |
| 출력 shape · dtype | (240, 320) · uint8 | 입력과 같음 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| ksize=(5, 5) (필수) | Kernel 크기, 양수 홀수 (또는 0이면 sigma로 계산) | 크면 더 흐려지고 세부가 사라짐. 🤔 홀수인 이유: 출력할 가운데 칸이 딱 하나 있어야 하니까 |
| sigmaX=0 (필수) | 가로 방향 퍼짐 정도. 0이면 Kernel 크기로부터 자동 계산 | 크면 먼 칸에도 가중치를 줘서 더 넓게 섞음 |
| sigmaY=0, borderType | (선택) 세로 퍼짐(0이면 sigmaX와 같게), 가장자리 규칙 |  |
| medianBlur의 ksize | 정수 하나, 1보다 큰 홀수 (3, 5, 7 …) | 짝수면 에러. 크면 더 큰 잡음 덩어리도 지우지만 가는 선도 사라짐 |
| bilateralFilter의 d | 이웃 지름 | 클수록 연산 증가 (느림) |
| sigmaColor / sigmaSpace | 밝기 차이 / 거리를 얼마나 봐줄지 | sigmaColor가 크면 경계 건너편도 섞여서 일반 Blur에 가까워짐 |

🤔 왜 "크면 무조건 좋다"가 아닐까? — 파라미터가 크면 잡음은 더 줄지만 정보 손실과 연산량도 함께 늘어나요. 그래서 먼저 잡음 종류(점 잡음인가, 고르게 퍼진 잡음인가)와 Edge를 보존해야 하는지를 확인하고 필터를 골라요.

### cv2.Canny()

```
blur  = cv2.GaussianBlur(gray, (5, 5), 0)     # 잡음 Edge를 줄이려고 먼저 Blur
edges = cv2.Canny(blur, threshold1=70, threshold2=150)
cv2.imwrite("outputs/edges.png", edges)        # 0 / 255 의 1채널
```

▶ 실제로 돌려 보면 noisy_gray.png — 경계(흰색)로 표시된 칸 수

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| np.unique(cv2.Canny(blur, 70, 150)) | [ 0 255] | 경계 255 / 아님 0 — 1채널 |
| Canny(noisy, 70, 150) 경계 칸 수 | 12138 | Blur 없이: 잡음까지 경계로 |
| Canny(GaussianBlur(noisy, (5,5), 0), 70, 150) | 3528 | Blur 후: 잡음 경계가 줄어듦 |
| Canny(blur, 30, 80) | 9776 | 기준을 낮추면 경계 ↑ |
| Canny(blur, 150, 300) | 2496 | 기준을 높이면 경계 ↓ |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| image (필수) | 보통 Gray(+Blur) 이미지 | Blur를 빼면 잡음 Edge가 크게 늘어남 (위 그림) |
| threshold1 / threshold2 (필수) | 약한 / 강한 경계 기준 (밝기 변화량 기준) | 낮추면 잡음 Edge↑, 높이면 약한 경계를 놓침 |
| apertureSize=3 | (선택) 밝기 변화를 재는 Kernel 크기 | 크면 더 굵은 변화에 반응 |
| L2gradient=False | (선택) 변화량을 정확한 거리(L2)로 잴지 | True면 조금 더 정확하지만 느림 |

⚠️ Canny의 70/150은 밝기 이진화 기준이 아니에요(밝기 변화량 기준). 그리고 Edge 수가 많다고 좋은 결과가 아니에요 — 많아진 만큼 잡음일 수 있어요.

### cv2.threshold() · cv2.adaptiveThreshold()

```
retval, binary = cv2.threshold(gray, thresh=140, maxval=255, type=cv2.THRESH_BINARY)
print("used threshold:", retval)

# Otsu: thresh 자리는 무시되고 자동으로 정해진 값이 retval로 나옴
blur = cv2.GaussianBlur(gray, (5, 5), 0)
retval, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Adaptive: 칸마다 주변 영역으로 기준을 따로 계산
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, blockSize=11, C=2)
```

▶ 실제로 돌려 보면 scene.png의 gray — 흰색(255)이 된 칸 비율

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv2.threshold(gray, 140, 255, THRESH_BINARY) | retval=140, 흰 칸 44.1% | 기준을 내가 정함 → retval도 140 |
| np.unique(binary) | [ 0 255] | 0과 255 두 값만 |
| cv2.threshold(blur, 0, 255, BINARY + OTSU) | retval=118, 흰 칸 53.1% | 0은 무시되고 Otsu가 고른 기준이 retval |
| cv2.adaptiveThreshold(gray, 255, GAUSSIAN_C, BINARY, 11, 2) | 흰 칸 88.7% | 칸마다 주변 기준 — 반환값은 이미지 하나 |

| 인자 / 반환 | 뜻 | 바꾸면? |
|---|---|---|
| thresh | 기준값 T | 낮추면 바탕도 물체로, 높이면 물체도 바탕으로 (밝기 > T인 칸이 늘거나 줄기 때문) |
| maxval | 기준을 넘은 칸에 줄 값 (보통 255) | 255가 관례인 이유: 흰색 = 전경이라 형태학·저장에서 바로 보이니까 |
| type | THRESH_BINARY / THRESH_BINARY_INV(흑백 반대) | INV면 어두운 쪽이 255 → 물체가 배경보다 어두울 때 사용 |
| + cv2.THRESH_OTSU | T를 자동 결정 | retval에 실제로 고른 T가 들어오니 기록해 둠 |
| 반환 retval, dst | 실제 쓴 기준값, 결과 이미지 — 2개! | 하나로 받으면 튜플이 들어와 뒤에서 에러 |
| blockSize (Adaptive) | 주변 영역 크기 (홀수) | 작으면 잡음에 민감, 크면 Fixed에 가까워짐 |
| C (Adaptive) | 주변 평균에서 빼는 값 | 크면 흰색(전경)으로 판정되는 칸이 늘어남 |

### cv2.getStructuringElement() · cv2.morphologyEx()

```
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
print(kernel)                                   # 0/1로 된 Kernel 모양 확인

opened = cv2.morphologyEx(binary, op=cv2.MORPH_OPEN, kernel=kernel, iterations=1)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
```

▶ 실제로 돌려 보면 objects.png를 Otsu로 이진화한 뒤 — 흰 덩어리 개수(connectedComponents)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv2.getStructuringElement(MORPH_ELLIPSE, (5, 5)) | 0 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0 | 0/1로 된 원 모양 Kernel |
| binary 의 흰 덩어리 수 | 8 | 작은 잡음 점까지 덩어리로 셈 |
| morphologyEx(binary, MORPH_OPEN, kernel) | 3 | Opening: 작은 흰 점 제거 |
| morphologyEx(opened, MORPH_CLOSE, kernel) | 3 | Closing: 물체 속 작은 구멍 메움 |
| 흰 칸 수 binary → opened → closed | 14142 → 14119 → 14144 | 열기는 점을 지우고, 닫기는 틈을 메워 다시 늘어남 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| shape | Kernel 모양 MORPH_RECT / MORPH_ELLIPSE / MORPH_CROSS | RECT는 각진 모서리를 잘 유지, ELLIPSE는 둥근 물체를 덜 각지게 다듬음 (Kernel 모양이 그대로 물체 가장자리에 찍히기 때문) |
| ksize | Kernel 크기 | 크면 더 큰 먼지·구멍까지 처리하지만, 물체 모양도 더 변함 |
| anchor=(-1,-1) | (선택) 기준점, 기본은 가운데 | 보통 그대로 둠 |
| op | MORPH_OPEN / MORPH_CLOSE (침식·팽창만이면 MORPH_ERODE / MORPH_DILATE) | OPEN은 먼지, CLOSE는 구멍 — 남은 문제가 무엇인지 보고 고름 |
| iterations | 반복 횟수 (기본 1) | Kernel 크기와 함께 실제 영향 범위를 결정 (3×3을 2번 ≈ 더 큰 Kernel) |

💡 관례: 흰색(255) = 전경(물체). 🤔 왜 관례를 확인? — 침식·팽창은 흰색을 깎고 부풀리는 연산이라, 물체가 검은색이면 효과가 정반대로 나와요. 물체가 어두우면 THRESH_BINARY_INV로 뒤집고 시작해요.

원본→ Gray→ Blur→ Binary→ Closing

🤔 왜 이 순서? — 각 단계가 다음 단계의 약점을 막아 줘요. Gray로 밝기만 남기고 → Blur로 잡음을 줄여야 Threshold가 잡음 점을 물체로 착각하지 않고 → 그래도 남은 먼지·구멍은 Morphology가 정리해요. 처리 순서가 바뀌면 결과도 달라지니 각 단계의 입력·출력 shape와 값 범위를 기록해요.

### 읽기 cv2.VideoCapture() · 저장 cv2.VideoWriter()

```
def process_frame(frame):                 # Frame 함수는 파일을 열고 닫지 않음
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    edge  = cv2.Canny(cv2.GaussianBlur(small, (3, 3), 0), 80, 160)
    return cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)   # 컬러 Writer엔 3채널로

size = (160, 120)                         # (W, H)
cap = cv2.VideoCapture("data/sample.avi")
writer = None
try:
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert 0 < fps < float("inf")         # 0이나 무한대면 잘못 읽힌 값
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter("outputs/edge.avi", fourcc, fps, size)
    assert writer.isOpened()
    while True:
        ok, frame = cap.read()
        if not ok:                        # 끝 또는 읽기 실패
            break
        writer.write(process_frame(frame))   # write는 반복 안에서
finally:
    cap.release()                         # release는 반복이 끝난 뒤
    if writer is not None:
        writer.release()
```

▶ 실제로 돌려 보면 data/sample.avi

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cap.isOpened() | True | 열기 성공 |
| cap.get(cv2.CAP_PROP_FPS) | 12 | 초당 Frame 수 |
| cap.get(cv2.CAP_PROP_FRAME_COUNT) | 48 | 전체 Frame 수 |
| ok, frame = cap.read() | ok → True, frame.shape → (240, 320, 3) | Frame 한 장 = 이미지 배열 |
| 반복으로 처리·저장한 Frame 수 | 48 | 마지막 다음 read()는 ok=False → 반복 끝 |
| writer.isOpened() | True | MJPG + .avi 조합 OK |
| 저장한 edge.avi 다시 읽기: Frame 수 · shape | 48 · (120, 160, 3) | (160, 120)으로 줄여 3채널로 저장됨 |
| frameSize=(160,120)인데 (320,240) Frame을 write | 저장된 Frame 수 0 | 에러 없이 조용히 실패! |

| 명령 / 인자 | 뜻 | 바꾸면? / 🤔 왜? |
|---|---|---|
| VideoCapture(source) | 파일 경로 또는 카메라 번호 (0 = 기본 카메라) | 열기 실패도 에러가 아니라서 isOpened()로 직접 확인 |
| cap.read() | (ok, frame) 두 개 반환 | 마지막 Frame 다음엔 ok=False → 이걸로 반복을 끝냄 |
| cap.get(prop) / set(prop, value) | 속성값(float) 읽기 / 설정 시도 | 코덱에 따라 값이 부정확할 수 있고, set은 모든 백엔드가 지원하지 않음 → 그래서 FPS를 검사 |
| VideoWriter_fourcc(*"MJPG") | 코덱 코드 ("mp4v", "XVID" 등) | 확장자·코덱·설치된 코덱 조합이 맞아야 열림 |
| fps (필수) | 저장할 재생 속도 | 입력 FPS와 다르게 주면 재생시간이 바뀜 (절반 → 2배 느리게) |
| frameSize (필수) | 모든 Frame의 (W, H) | write하는 Frame 크기와 다르면 조용히 저장 실패 — Writer는 한 가지 크기로만 파일을 만들기 때문 |
| isColor=True | (선택) 컬러 Frame 여부 (기본 True) | Gray Frame을 저장하려면 False로 두거나, GRAY2BGR로 3채널로 바꿔서 넣기 |
| release() | 파일·카메라 자원 해제 | 안 하면 파일 끝 정보가 안 써져 재생이 안 될 수 있고, 카메라도 계속 잡고 있음 |

🤔 왜 Frame 함수와 반복·I/O를 나눠? — process_frame 안에서 VideoCapture를 매번 열면 반복 I/O로 성능이 크게 떨어지고 자원 관리도 복잡해져요. 처리 함수는 "배열 → 배열"만 하고, 열기·반복·저장은 바깥에서 맡으면 같은 함수를 이미지에도 그대로 쓸 수 있어요.

⚠️ 영상 저장이 안 될 때 확인 순서: ① writer.isOpened() True? ② Frame (W, H) = frameSize? ③ Gray Frame + isColor=True 아닌지? ④ 확장자·fourcc·설치 코덱 조합? ⑤ release 후 파일 크기와 다시 읽기 확인. MJPG는 손실 압축이라 픽셀이 원본과 완전히 같지는 않아요.

### 반전 cv2.flip()

```
horizontal = cv2.flip(img, flipCode=1)    # 좌우
vertical   = cv2.flip(img, flipCode=0)    # 상하
both       = cv2.flip(img, flipCode=-1)   # 둘 다
```

▶ 실제로 돌려 보면 빨간 사각형이 있는 자리 (가로 320, 세로 240)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| img (원본) | x 15~65, y 15~55 |  |
| cv2.flip(img, 1) | x 254~304, y 15~55 | 좌우: x → 319 - x |
| cv2.flip(img, 0) | x 15~65, y 184~224 | 상하: y → 239 - y |
| cv2.flip(img, -1) | x 254~304, y 184~224 | 둘 다 |
| cv2.flip(img, 1).shape | (240, 320, 3) | 크기는 그대로 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| flipCode (필수) | 1 = 좌우, 0 = 상하, 음수 = 둘 다 | 상하 반전은 하늘·땅이 뒤집혀 일반 사진에선 부자연스러움 → 과제에 맞는지 확인 |

### 회전 getRotationMatrix2D() + warpAffine()

```
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D(center=(w / 2, h / 2), angle=10, scale=1.0)   # 2×3 행렬
rotated = cv2.warpAffine(img, M, dsize=(w, h), borderMode=cv2.BORDER_REFLECT)
```

▶ 실제로 돌려 보면 scene.png, 가운데 기준 10° 회전

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| cv2.getRotationMatrix2D((160, 120), 10, 1.0) | [[ 0.985 0.174 -18.407] [ -0.174 0.985 29.607]] | 2×3 행렬: [cos sin 이동x; -sin cos 이동y] |
| warpAffine(img, M, (320, 240)).shape | (240, 320, 3) | dsize 그대로 → 모서리가 잘림 |
| 기본(BORDER_CONSTANT) 왼쪽 위 칸 rot[0, 0] | [0 0 0] | 빈 곳은 검정(0)으로 채움 |
| BORDER_REFLECT 왼쪽 위 칸 rot[0, 0] | [ 0 0 255] | 빈 곳을 거울처럼 비친 값으로 |

🤔 왜 두 단계? — getRotationMatrix2D는 "어느 칸이 어디로 가는지"를 적은 2×3 행렬(지도)만 만들고, warpAffine이 그 지도대로 실제 픽셀을 재배치해요. 지도를 따로 두니 회전·이동·확대를 하나의 행렬로 합쳐 한 번에 적용할 수 있어요.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| center | 회전 중심 (x, y) — 여기는 (W/2, H/2) 순서 | 중심을 모서리로 주면 사진이 화면 밖으로 크게 빠져나감 |
| angle | 각도 (도 단위, 양수 = 반시계) | 크면 모서리 손실↑, 큰 회전은 라벨 뜻도 바꿀 수 있음 |
| scale | 크기 배율 (1.0 = 그대로) | 1보다 작게 하면 빈 모서리가 더 커짐 |
| M (필수) | getRotationMatrix2D가 만든 2×3 행렬 |  |
| dsize (필수) | 출력 크기 (W, H) | 원본 크기로 두면 회전으로 삐져나간 모서리는 잘림 |
| borderMode | 빈 모서리 채우는 규칙 | 기본은 검은색 → 모델이 "검은 삼각형"을 특징으로 배울 수 있음. BORDER_REFLECT면 주변을 반사해 덜 티 남 |

### 밝기 · Crop · Blur

```
bright = cv2.convertScaleAbs(img, alpha=1.0, beta=30)    # 밝기 +30

h, w = img.shape[:2]
crop = img[h//10 : h - h//10, w//10 : w - w//10]         # NumPy 자르기 [y, x]
crop = cv2.resize(crop, (w, h))                          # 원래 크기로

blur = cv2.GaussianBlur(img, (5, 5), 0)
```

▶ 실제로 돌려 보면 값 [100, 250]인 uint8 두 칸 · scene.png

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| a + 30 (NumPy 그냥 더하기) | [[130 24]] | 250+30=280 → 256을 넘겨 24로 돌아감! |
| cv2.add(a, 30) | [[130 255]] | 255에서 멈춤(포화) |
| cv2.convertScaleAbs(a, alpha=1.0, beta=30) | [[130 255]] | 255에서 멈춤 |
| img[h//10 : h-h//10, w//10 : w-w//10].shape | (192, 256, 3) | 위아래·좌우 10%씩 잘라냄 |
| cv2.resize(crop, (w, h)).shape | (240, 320, 3) | 다시 원래 크기로 → 확대 효과 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| alpha | 곱하는 값 (대비) | 1보다 크면 대비↑ |
| beta | 더하는 값 (밝기) | 음수면 어둡게. 결과는 절댓값을 취해 0~255 uint8로 맞춰짐 |
| img[y1:y2, x1:x2] | 잘라내기 (세로 먼저!) | 너무 많이 자르면 물체가 잘려 라벨 뜻이 바뀔 수 있음 |
| Crop 후 resize | 원래 크기로 되돌림 | 모델 입력 크기를 모든 사진에서 같게 맞추기 위해 |

🤔 왜 img + 30 대신 convertScaleAbs? — uint8 배열에 그냥 30을 더하면 250 + 30 = 280이 256을 넘쳐 24로 되돌아가요(가장 밝은 곳이 갑자기 검게). convertScaleAbs는 계산 뒤 0~255로 잘라 맞춰 주니 밝은 곳은 255에서 멈춰요.

### 폴더 일괄 처리 sorted(Path.glob())

```
from pathlib import Path
import cv2

src_dir = Path("data/images")
out_dir = Path("outputs/images")
out_dir.mkdir(parents=True, exist_ok=True)       # 출력 폴더 준비
for path in sorted(src_dir.glob("*.png")):       # 순서 고정
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"read failed: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (320, 240))
    if not cv2.imwrite(str(out_dir / path.name), resized):
        raise OSError(f"save failed: {path}")

print(cv2.imread(str(out_dir / path.name), 0).shape)   # 다시 읽어 확인 → (240, 320)
```

▶ 실제로 돌려 보면 data/images 폴더

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| [p.name for p in sorted(Path("data/images").glob("*.png"))] | ['test_06.png', 'train_01.png', 'train_02.png', 'train_03.png', 'train_04.png', 'valid_05.png'] | 이름 순서로 고정 |
| cv2.imread("data/images/test_06.png").shape | (240, 320, 3) | 원본 |
| cv2.imread("outputs/images/test_06.png", 0).shape | (240, 320) | Gray + (320, 240)으로 저장된 것 확인 |
| len(list(Path("outputs/images").glob("*.png"))) | 6 | 입력 개수와 같아야 함 |

| 코드 | 뜻 | 🤔 왜? |
|---|---|---|
| Path.glob("*.png") | 패턴에 맞는 파일 경로를 순회 | 파일 이름을 손으로 적지 않아도 폴더 전체를 처리 |
| sorted(...) | 파일 순서 고정 | glob 순서는 운영체제마다 달라질 수 있어서, 정렬해야 재현성이 생김 |
| mkdir(parents=True, exist_ok=True) | 상위 폴더까지 만들고, 이미 있어도 OK | 폴더가 없으면 imwrite가 에러 없이 False만 돌려줌 |
| str(path) | Path를 문자열로 | cv2 함수는 문자열 경로를 받음 |
| imread(path, 0) | 저장 결과를 Gray로 다시 읽기 | 저장이 "성공"해도 크기·채널이 의도와 같은지는 다시 읽어 봐야 앎 |

### Frame 건너뛰며 처리 frame_idx % step

```
# cap · process_frame 은 위 VideoCapture 항목과 같아요
step = 5
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if frame_idx % step == 0:          # 5장마다 1장만 처리
        result = process_frame(frame)
    frame_idx += 1
```

▶ 실제로 돌려 보면 data/sample.avi (48 Frame, 12 FPS)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 읽은 Frame 수 | 48 |  |
| step = 5 로 처리한 Frame 번호 | [0, 5, 10, 15, 20, 25, 30, 35, 40, 45] | 10장만 처리 — 계산량 약 1/5 |
| 처리 간격(초) | 0.417 | 12 FPS에서 5장 = 약 0.42초마다 한 번 → 그 사이 빠른 사건은 놓칠 수 있어요 |

## 📖 Day13 용어 (39개)
- **임계값** (Threshold (판단 기준값)): 확률이 이 값 이상(또는 초과)이면 양성(실패)이라고 부르는 선. 낮추면 놓침(FN)은 줄고 헛경고(FP)는 늘어요. valid에서만 정해요.
- **seed** (random_state (난수 고정값)): 무작위 과정(추출, 특성 후보, 분할 섞기 등)을 매번 같게 만드는 번호. 결과를 재현하려고 고정해요(예: 42).
- **shape** (Shape (모양)): Tensor가 축마다 몇 칸인지 적은 것. 예: (batch, feature) = (32, 2)는 샘플 32개 × 특성 2개. 오류 대부분이 shape가 안 맞아서 생겨요.
- **Kernel** (Kernel / Filter (커널, 필터)): 합성곱 창 안의 가중치 묶음. 3×3이면 숫자 9개이고, 이 숫자는 사람이 아니라 학습으로 정해져요.
- **채널** (Channel (채널)): 이미지·Feature Map이 겹겹이 쌓인 '장 수'. 컬러 사진은 R·G·B 3채널이고, Conv 층의 출력 채널 수 = Kernel 개수예요.
- **증강** (Data Augmentation (데이터 증강)): 사진을 뒤집고, 살짝 돌리고, 밝기를 바꾸는 등 정답이 안 바뀌는 변형으로 학습 예시를 늘리는 기법. 과적합을 줄여요. train에만 쓰고, 정답을 바꾸는 변형(예: 6↔9 회전)은 금지예요.
- **Mask** (Mask (마스크)): 어느 칸을 계산에서 뺄지 True/False로 표시한 표. Padding Mask는 '여기는 채운 가짜 칸'을 표시해 평균·Attention에서 빼게 해요. PyTorch Attention에선 True = 차단이에요.
- **픽셀** (Pixel (화소)): 사진을 이루는 가장 작은 칸 하나. 위치 (y, x)마다 숫자가 들어 있고, 컬러면 칸 하나에 숫자 3개(B, G, R)가 있어요.
- **BGR** (Blue, Green, Red (색 순서)): OpenCV(imread)가 색을 저장하는 순서. 보통 쓰는 RGB와 순서가 반대라서, Matplotlib 등으로 넘길 땐 cvtColor로 RGB로 바꿔야 빨강↔파랑이 안 뒤바뀌어요.
- **RGB** (Red, Green, Blue (색 순서)): 빨강·초록·파랑 순서로 색을 적는 방식. Matplotlib·PIL·사전학습 모델은 RGB를 기대해요.
- **uint8** (unsigned 8-bit integer (부호 없는 8비트 정수)): 0~255만 담을 수 있는 숫자 자료형. 이미지 기본 형식이라, 그냥 더하면 255를 넘쳐 0 쪽으로 되돌아갈 수 있어요(250+30 → 24).
- **Gray** (Grayscale (회색조)): 색 3개를 섞어 밝기 하나로 만든 1채널 이미지. 경계·이진화는 밝기만 있으면 되어서 보통 Gray로 바꾼 뒤 처리해요.
- **색공간** (Color space): 색을 숫자로 나타내는 방식(BGR, RGB, Gray, HSV 등). 정답이 있는 게 아니라 목적에 맞게 골라요.
- **HSV** (Hue, Saturation, Value (색상·채도·명도)): 색을 색 종류(H)·진하기(S)·밝기(V)로 나눠 적는 방식. 조명이 바뀌어도 H는 비교적 그대로라 특정 색을 골라낼 때 좋아요. OpenCV에선 H가 0~179.
- **Resize** (Resize (크기 바꾸기)): 이미지 크기를 바꾸는 것. cv2.resize의 크기는 (W, H) 순서라 shape (H, W)와 반대예요. 비율이 다르면 찌그러져요.
- **보간법** (Interpolation (보간법)): 크기를 바꿀 때 새 픽셀 값을 주변 값으로 정하는 방법. 줄일 땐 여러 칸을 요약하고, 키울 땐 없던 칸 값을 추정해요.
- **INTER_AREA** (Area interpolation (면적 평균 보간)): 줄어드는 칸이 덮는 면적의 평균으로 값을 정하는 보간법. 축소할 때 잔무늬·깨짐이 적어 자주 써요. (기본값은 INTER_LINEAR)
- **INTER_NEAREST** (Nearest-neighbor interpolation (최근접 보간)): 가장 가까운 픽셀 값을 그대로 복사하는 보간법. 중간값이 안 생겨서 0/255 Mask 크기를 바꿀 때 적합하지만, 계단 현상이 생겨요.
- **Blur** (Blur (흐리게 하기, Smoothing)): 주변 픽셀 값을 섞어 이미지를 부드럽게 흐리는 것. 잡음은 줄지만 세부·경계도 함께 흐려져요.
- **GaussianBlur** (Gaussian Blur): 가운데 칸에 더 큰 가중치를 준 평균으로 흐리는 필터. 평균 필터보다 자연스럽게 흐려지고, Canny·Otsu 전에 잡음을 줄일 때 많이 써요.
- **medianBlur** (Median Blur (중앙값 필터)): 이웃 값을 줄 세운 뒤 가운데 값을 고르는 필터. 튀는 점(소금·후추 잡음)이 아예 뽑히지 않아 점 잡음 제거에 강해요. ksize는 1보다 큰 홀수.
- **bilateralFilter** (Bilateral Filter (양방향 필터)): 거리뿐 아니라 밝기 차이까지 보고 섞는 필터. 경계 건너편은 거의 안 섞어서 경계를 지키며 흐리지만, 비교적 느려요.
- **소금·후추 잡음** (Salt-and-pepper noise): 주변과 동떨어진 흰 점(255)·검은 점(0)이 여기저기 박힌 잡음. 평균 필터로는 흐린 점이 남고, Median 필터로 잘 지워져요.
- **Edge** (Edge (경계, 윤곽)): 밝기가 갑자기 바뀌는 곳. 물체의 윤곽선이 보통 여기에 있어요.
- **Canny** (Canny Edge Detector): 경계만 흰색(255)으로 남기는 방법. 기준값 2개를 써서 강한 경계는 남기고, 약한 경계는 강한 경계에 이어진 것만 남겨요. 잡음 때문에 보통 Blur 후 사용.
- **히스토그램** (Histogram (밝기 분포 막대그래프)): 밝기 값(0~255)마다 픽셀이 몇 개인지 센 막대그래프. 물체·배경 밝기가 두 봉우리로 갈리는지 볼 수 있어요.
- **Otsu** (Otsu's method (오츠 자동 이진화)): 밝기 히스토그램을 보고 어두운 무리와 밝은 무리가 가장 잘 갈리는 기준값 T를 자동으로 찾는 이진화. 사진마다 T를 손으로 정하지 않아도 돼요.
- **Adaptive Threshold** (Adaptive Thresholding (적응형 이진화)): 칸마다 주변 영역(blockSize)의 평균에서 C를 뺀 값을 그 칸만의 기준으로 쓰는 이진화. 조명이 고르지 않은 사진에 강해요.
- **전경** (Foreground (전경, 물체 영역)): 이진 이미지에서 물체로 본 영역. 관례상 흰색(255)이 전경이에요. 물체가 어두우면 THRESH_BINARY_INV로 뒤집어서 맞춰요.
- **형태학 연산** (Morphological operations (형태학 연산)): 이진화 결과의 흰 영역을 Kernel로 깎거나(침식) 부풀려(팽창) 먼지·구멍을 정리하고 모양을 다듬는 처리.
- **Erosion** (Erosion (침식)): Kernel 안이 모두 흰색일 때만 흰색을 남겨, 흰 영역을 깎아 줄이는 연산. 작은 흰 점은 사라지고 붙은 물체가 떨어져요.
- **Dilation** (Dilation (팽창)): Kernel 안에 흰색이 하나라도 있으면 흰색으로 만들어, 흰 영역을 부풀리는 연산. 끊긴 부분이 이어져요.
- **Opening** (Opening (열기: 침식 → 팽창)): 먼저 깎고 다시 부풀리는 연산. Kernel보다 작은 흰 먼지는 사라지고, 큰 물체는 원래 크기로 돌아와요. 단, 작은 진짜 결함도 지울 수 있어요.
- **Closing** (Closing (닫기: 팽창 → 침식)): 먼저 부풀리고 다시 깎는 연산. 물체 안의 작은 검은 구멍이나 끊긴 글자를 메워요.
- **Frame** (Frame (영상의 한 장)): 영상에서 꺼낸 사진 한 장. 영상은 시간 순서로 이어진 Frame 묶음이라, 영상 처리 = 같은 처리를 Frame마다 반복하는 것이에요.
- **FPS** (Frames Per Second (초당 프레임 수)): 1초에 몇 장의 Frame인지. 영상 파일에선 재생 속도(재생시간 = Frame 수 ÷ FPS), 모델에선 1초에 처리한 장 수를 뜻해요. 클수록 빠름.
- **VideoCapture** (cv2.VideoCapture / cv2.VideoWriter): VideoCapture는 영상 파일·카메라에서 Frame을 한 장씩 읽고, VideoWriter는 Frame을 모아 영상 파일로 저장해요. 다 쓰면 release()로 닫아요.
- **Codec** (Codec (영상 압축·저장 방식)): 영상을 압축해 파일로 저장하는 방식. OpenCV에선 fourcc 코드(예: MJPG, mp4v)로 고르고, 확장자·설치된 코덱과 맞아야 저장돼요.
- **Crop** (Crop (잘라내기)): 이미지의 일부만 잘라내는 것. NumPy로 img[y1:y2, x1:x2] — 세로(y)가 먼저예요. 너무 많이 자르면 물체가 잘려 라벨 뜻이 바뀔 수 있어요.

---

# Day14 · 전이학습과 오류 분석
_2026-11-05(목) · 데이터셋 설계 · 전이학습 · Fine-tuning · 클래스 불균형 · Macro F1 · Grad-CAM_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day14.html

📌 딱 3줄 요약

- 전이학습 = 이미 학습된 모델의 Backbone(특징 뽑는 부분)을 빌려 오고, 우리 문제용 Head(분류기)만 새로 학습한 뒤, 필요하면 Head-only → Partial → Full 순서로 층을 연다.

- 데이터가 불균형하면 Accuracy는 실패를 가린다 → Macro F1과 클래스별 Recall·support, Confusion Matrix, 오류 이미지를 같이 본다.

- Grad-CAM은 "모델이 어디에 반응했나"를 보여 주는 단서일 뿐 원인을 확정하지 않는다. 비교 조건과 설정을 기록해야 결과를 설명할 수 있다.

🔧 Day14 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| torch, torchvision | 모델·학습 엔진 / 사전학습 모델(ResNet18)·ImageFolder·transforms | import torch, from torchvision.models import resnet18 |
| scikit-learn | Confusion Matrix, F1, 분류 보고서, 그룹 단위 분할 | from sklearn.metrics import f1_score |
| numpy | 숫자 배열 계산 | import numpy as np |
| Pillow | 이미지 파일 열기 (ImageFolder가 내부에서 씀) | from PIL import Image |
| opencv-python | 이미지 읽기·저장, Heatmap 겹쳐 그리기 (Day13 전처리 연결) | import cv2 |

### 🔧 설치 · 확인

```
python -m pip install torch torchvision scikit-learn opencv-python Pillow numpy
python -c "import torch, torchvision, sklearn, cv2; print(torch.__version__, torchvision.__version__, sklearn.__version__, cv2.__version__)"
python -c "import torch; print(torch.cuda.is_available())"   # True면 GPU 사용 가능

device = "cuda" if torch.cuda.is_available() else "cpu"      # 선택한 device를 기록
```

💡 pip 이름과 import 이름이 다른 경우: scikit-learn → sklearn, opencv-python → cv2, Pillow → PIL. torch와 torchvision은 서로 짝이 맞는 버전이어야 해서 함께 설치해요.

⚠️ 사전학습 가중치(Weight)는 처음 쓸 때 인터넷에서 내려받아 캐시에 저장돼요. 수업 전에 미리 받아 두고, 다운로드가 실패했다고 weights=None(무작위 가중치)으로 조용히 바꾸지 않습니다. 🤔 왜? — 무작위 가중치는 "전이학습"이 아니라서, 결과가 나빠져도 그 이유를 엉뚱하게 해석하게 돼요. 그리고 출처를 믿을 수 없는 checkpoint는 torch.load 하지 않아요(파일 안의 코드가 실행될 수 있어서).

## 이거 어디에 써?

Day13까지는 이미지를 처리하는 코드를 다뤘어요. 오늘은 이미지를 분류하는 모델을 만들고, 틀린 사례를 분석해서 다음에 뭘 고칠지 정하는 법을 배워요.

🤔 왜 전이학습? — 처음부터(Scratch) 학습하면 "모서리·질감·모양을 알아보는 법"부터 배워야 해서 데이터와 시간이 아주 많이 들어요. 사전학습 Backbone은 이미 그걸 알고 있으니, 우리는 그 특징을 우리 클래스로 연결하는 부분만 배우면 돼요 → 적은 데이터·적은 계산으로 출발할 수 있어요. 다만 우리 데이터에서의 성능은 따로 다시 확인해야 해요.

🏭

제품 사진 검사정상 / 결함 유형 분류

🧾

문서 사진 분류영수증 · 명세서 · 기타

⚖️

드문 클래스 지키기결함처럼 적지만 중요한 것을 놓치지 않기

🔍

모델 점검배경·워터마크 같은 엉뚱한 단서에 기대는지 확인

데이터셋 · 분할→ 사전학습 Backbone + 새 Head→ 학습 범위 선택→ 불균형 대응→ 지표 · 오류 이미지→ Grad-CAM→ 다음 실험 가설

💡 하루 내내 지키는 약속

- train으로만 배우고, valid로만 고르고, test는 마지막 확인용으로 남긴다.

- 비교할 때는 한 번에 하나만 바꾸고 나머지 조건은 같게 둔다.

- 오답이 없으면 "0건"이라고 쓰고, Grad-CAM은 원인이 아니라 단서로 쓴다.

🤔 왜 test를 마지막까지 아껴? — valid 점수를 보며 epoch·층·학습률을 고르는 순간, 그 선택은 valid에 맞춰진 거예요. test까지 보면서 고르면 test도 "본 문제"가 되어 실제 성능보다 좋게 보여요. 그래서 test는 고른 뒤 딱 한 번 확인하는 용도예요.

## ① 데이터셋 설계 — 라벨과 분할을 믿을 수 있나?

좋은 모델은 데이터셋 설계에서 시작해요. 한 장의 이미지에는 이미지 경로·고유 ID · 정답 클래스 · 출처(그룹 ID) · 분할(train/valid/test) 정보가 함께 있어야 하고, 모델 쪽에는 색 순서(RGB) · 입력 크기 · 값 범위/Normalize · 증강 허용 범위라는 입력 계약이 있어요.

### ImageFolder — 폴더 이름이 곧 라벨

```
data/
├─ train/
│   ├─ circle/    *.png
│   ├─ square/    *.png
│   └─ triangle/  *.png
├─ valid/  (같은 클래스 폴더 3개)
└─ test/   (같은 클래스 폴더 3개)
```

ImageFolder는 클래스 폴더 이름을 정렬해서 0, 1, 2 번호를 붙여요 → {'circle': 0, 'square': 1, 'triangle': 2}. 그래서 폴더 이름을 바꾸면 라벨 번호가 바뀝니다(라벨 계약 변경).

🤔 왜 정렬 순서로 번호를 붙여? — 폴더를 읽는 순서는 운영체제마다 다를 수 있어요. 이름순으로 정렬하면 어느 컴퓨터에서든 같은 이름 → 같은 번호가 보장돼요. 반대로 말하면, 폴더 하나만 이름이 달라도(예: valid에만 Triangle) 번호가 어긋나 모델은 맞혔는데 채점은 틀린 일이 생겨요. 그래서 분할마다 매핑을 비교해요.

⚠️ train·valid·test에서 클래스 이름과 번호 순서가 똑같아야 해요. 빈 폴더, 숨김 파일, 지원하지 않는 확장자도 미리 검사합니다.

### 데이터 누수 (Leakage)

같은 사진(또는 거의 같은 장면)이 train과 test에 동시에 들어가면, 모델이 "외운 걸" 시험 보는 셈이라 점수가 부풀려져요.

| 상황 | 분할 단위 | 🤔 왜? |
|---|---|---|
| 독립 상품 사진 | 이미지 단위 | 각 촬영이 서로 독립이면 그냥 나눠도 됨 |
| 환자·사람 사진 | 사람/환자 ID | 같은 사람 특징이 test로 새는 것 방지 |
| 영상 Frame | 영상 또는 장면 ID | 옆 프레임끼리 거의 똑같음 → 프레임 단위 분할 금지 |
| 증강 이미지 | 원본 ID | 증강본은 원본과 거의 같은 사진이라, 갈라지면 test에 "답"이 새어 들어감 |

[그림: ] 무엇을 보나: 보라 칸은 영상 A, 분홍 칸은 영상 B에서 나온 Frame이에요(숫자 = 순서). 왼쪽은 Frame을 낱장으로 섞어 나눠 A3가 test로, 이웃 A2·A4는 train으로 갈라졌어요. 오른쪽은 영상(그룹) 하나를 통째로 한쪽에만 넣었어요. 🤔 왜? 옆 Frame끼리는 거의 같은 사진이라, 왼쪽처럼 나누면 test 문제의 "쌍둥이"가 이미 train에 있어요. 모델이 외운 걸로 맞혀도 점수가 올라가니, 처음 보는 영상에서의 실력을 잴 수 없어요. 그래서 분할 단위를 독립인 단위(그룹)로 잡아요.

💡 분할 전 확인할 것 — 중복(같은 파일·거의 같은 장면), 그룹(같은 제품·영상에서 나온 사진), 라벨(기준이 모호하면 모델 오류와 데이터 오류가 섞임), 분포(train에만 특정 배경·카메라·조명이 있으면 모델이 물체 대신 환경을 배움).

🔧 클래스 매핑 · 개수 확인 ImageFolder() 명령어 사전에서 자세히 →

```
train = ImageFolder("data/train")
print(train.class_to_idx)              # {'circle': 0, 'square': 1, 'triangle': 2}
```

[그림: 수업 예제 데이터 train, valid, test의 클래스별 이미지 수 막대그래프] 무엇을 보나: 수업 예제 데이터를 실제로 센 개수예요. train은 circle 120 · square 40 · triangle 20(6 : 2 : 1), valid는 30 · 30 · 30, test는 10 · 10 · 10이에요. 🤔 왜? train만 기울어 있으면 모델은 "일단 circle"이라고 답하는 쉬운 길부터 배워요(④). 반대로 valid·test가 균형이라, 소수 클래스를 놓치면 평가에서 바로 드러나요. 그래서 학습 전에 Counter(targets)로 분할마다 개수를 먼저 확인해요.

🔧 그룹 단위 분할 (예시 도구: GroupShuffleSplit) 명령어 사전에서 자세히 →

```
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, va_idx = next(gss.split(rows, groups=rows.group_id))   # 같은 group_id는 한쪽에만
```

## ② Backbone과 Head — 무엇을 빌려 오고, 무엇을 새로 만드나?

RGB 이미지 Batch→ Backbone (빌려 옴) 특징 추출 · 풀링→ 특징 벡터 ResNet18은 512개→ 새 Head (새로 학습) Linear(512, 3)→ 클래스 점수 Logit 3개

Backbone은 가장자리·질감·형태처럼 여러 이미지 문제에서 다시 쓸 수 있는 특징을 만들어요. Head는 그 특징을 오늘 우리 클래스 수에 맞는 점수(Logit)로 바꾸는 마지막 층이에요.

🤔 왜 Head는 갈아 끼워? — 원래 ResNet18의 Head(fc)는 ImageNet 1000개 클래스 점수를 내도록 학습돼 있어요. 우리는 3클래스라 출력 개수부터 맞지 않고, "1000개 중 몇 번"이라는 의미도 우리 문제와 무관해요. 반면 Backbone이 뽑는 모서리·질감 같은 특징은 어느 문제에나 쓸모가 있어서 그대로 빌려 와요.

Head의 학습 파라미터 수 = 특징 수 × 클래스 수 + bias

```
nn.Linear(512, 3)
  weight [3, 512] → 1,536개
  bias   [3]      →     3개
  합계 = 512 × 3 + 3 = 1,539개   ← 특징 512개를 Weight로 조합하고 Bias를 더해 클래스 점수 3개
```

### 얼리는 스위치는 두 개!

| 스위치 | 무엇을 고정? | 주의 |
|---|---|---|
| requires_grad = False | 가중치. gradient를 계산하지 않아 학습이 이 숫자를 못 바꿈 | 순전파 계산은 그대로 함 |
| Optimizer에 넘기는 목록 | 실제로 갱신되는 파라미터 | 열었어도 목록에 없으면 안 바뀜 |
| model.eval() / train() | BatchNorm 기억값(평균·분산)과 Dropout 동작 | 얼려도 train() 모드면 BN 기억값은 계속 바뀜 |

🤔 왜 Backbone을 먼저 얼려(freeze)? — 새 Head는 처음에 무작위라서 초반 gradient가 크고 엉망이에요. 이때 Backbone까지 열어 두면 그 엉망인 gradient가 이미 잘 배운 특징을 망가뜨려요. 먼저 Head만 학습해 안정시키면 학습할 숫자도 적어 빠르고, 메모리(VRAM)도 적게 들고, 데이터가 적을 때 과적합도 덜해요. 그다음에 필요하면 층을 열어요(③).

🤔 왜 스위치가 두 개나? — BatchNorm은 학습 중에 "들어오는 데이터의 평균·분산"을 기억값으로 계속 갱신하는데, 이건 gradient가 아니라 train 모드일 때 자동으로 일어나요. 그래서 requires_grad=False로 가중치를 얼려도 train 모드면 BN 기억값은 우리 데이터 쪽으로 바뀌어요. 완전히 고정하려면 Backbone을 eval 모드로 둘지 정책을 정하고 기록해요.

⚠️ 입력 규칙도 가중치와 한 묶음 — 사전학습 때 쓴 Resize·Crop·Normalize(ImageNet 평균·표준편차)를 그대로 지켜야 해요. 🤔 왜? — Backbone은 "그 규칙으로 바뀐 숫자"를 보고 특징을 배웠어요. 규칙이 바뀌면 들어오는 숫자의 분포가 달라져 특징도 달라지는데, 에러는 안 나니 조용히, 자신 있게 틀려요. OpenCV로 읽은 BGR 이미지를 그대로 넣는 것도 같은 이유로 안 돼요(RGB로 바꾸기).

🔧 사전학습 모델 불러오기 · Head 교체 명령어 사전에서 자세히 →

```
model = resnet18(weights=weights)       # ① 학습된 가중치 불러오기
model.fc = nn.Linear(in_features, num_classes)   # ③ 새 Head (새로 만든 층은 기본으로 학습 가능)
```

🔧 학습 가능한 파라미터 수 세기 명령어 사전에서 자세히 →

```
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
```

[그림: ResNet18에서 Head-only, Partial, Full 전략별 학습 파라미터 수를 로그 눈금 막대로 비교] 무엇을 보나: ResNet18 구조(3클래스 Head)에서 실제로 센 학습 파라미터 수예요. Head-only는 1,539개(전체의 0.014%), layer4까지 열면 8,395,267개(75.1%), 전체는 11,178,051개. 가로축은 한 칸이 10배인 log 눈금이에요. 🤔 왜? 파라미터는 뒤쪽 층일수록 채널이 많아 몰려 있어요 — layer4 하나가 전체의 3/4이에요. 그래서 Head-only는 계산·메모리가 아주 적고 과적합 위험도 낮은 "안전한 첫 기준선"이고, layer4만 열어도 학습량이 수천 배로 뛰어요. 위 코드로 출력한 숫자가 이 값과 다르면 얼리기 순서를 의심해요.

## ③ Fine-tuning — 어디까지 다시 학습시킬까?

| 전략 | 앞쪽 Backbone | 마지막 Stage (layer4) | 새 Head | 특징 |
|---|---|---|---|---|
| Head-only | 🔒 고정 | 🔒 고정 | 학습 | 빠르고 안정적, 학습 파라미터·VRAM 적음. 작은 데이터·비슷한 분야의 첫 기준선 |
| Partial | 🔒 고정 | 학습 | 학습 | 마지막 층만 살짝 조정 |
| Full | 학습 | 학습 | 학습 | 연산 많고 과적합 위험. 분야 차이가 클 때 검토 |

[그림: ] 무엇을 보나: 세로 한 줄이 한 전략이에요. 위가 출력(Head), 아래가 입력 쪽 층. 색칠된 칸만 requires_grad=True + Optimizer 목록에 들어가 값이 바뀌고, 🔒 칸은 순전파만 하고 그대로예요. 열리는 순서는 항상 위(출력 쪽)부터 아래로. 🤔 왜? 아래층은 모서리·색 변화처럼 어느 사진에나 통하는 특징이라 빌린 그대로 써도 되고, 위층일수록 원래 데이터(ImageNet)에 특화돼 고칠 필요가 커요. 그래서 "어디까지 열까?"는 위에서부터 한 칸씩 내려가며 실험해요.

🤔 왜 마지막 Stage(layer4)부터 열어? — 앞쪽 층은 모서리·색 변화처럼 어느 사진에나 통하는 기본 특징을, 뒤쪽 층은 원래 데이터(ImageNet)의 물체에 특화된 특징을 배워요. 우리 데이터에 맞게 고칠 필요가 큰 건 뒤쪽이라, 거기부터 열어야 적은 비용으로 효과를 봐요.

단계적으로 연다: ① Head-only로 데이터·라벨·학습 루프가 정상인지 확인 → ② 마지막 Stage만 열어 작은 학습률로 조정 → ③ 개선이 없거나 데이터가 충분할 때 더 앞쪽 층을 여는 실험을 따로 해요. 각 단계는 같은 분할·평가 지표·epoch 예산으로 비교하고, test는 최종 확인까지 보관해요.

🤔 왜 Head-only부터? — 결과가 나쁠 때 원인이 "데이터·라벨 문제"인지 "학습 범위 문제"인지 가르려면, 가장 단순하고 안정적인 기준선이 먼저 있어야 해요. 기준선이 정상으로 돌아간다는 걸 확인한 뒤에야 층을 연 효과를 공정하게 잴 수 있어요.

| 단계 | 학습 범위 | 학습률 예 (Backbone / Head) | 볼 것 |
|---|---|---|---|
| A | 새 Head | 1e-3 | 기준 성능, 어느 클래스를 틀리나 |
| B | layer4 + Head | 1e-4 / 1e-3 | valid 개선, 과적합 |
| C | 전체 Backbone + Head | 1e-5 / 5e-4 | 성능 · 시간 · VRAM · 안정성 |

🤔 왜 Backbone은 작은 학습률? — Backbone은 이미 좋은 값 근처에 있어요. 큰 보폭으로 움직이면 잘 배운 특징을 한 번에 망가뜨려요. 새 Head는 무작위에서 출발하니 크게, 기존 층은 작게 움직여요. 그래서 많이 열수록(C) Backbone 학습률을 더 줄여요.

[그림: 단계 A, B, C별 Backbone과 Head의 학습률을 로그 눈금 막대로 비교] 무엇을 보나: 강의 예시 학습률을 log 눈금으로 그렸어요. Head(분홍)는 1e-3 근처로 거의 그대로인데, Backbone(보라)은 A에선 고정, B에선 1e-4, C에선 1e-5로 열수록 10배씩 작아져요. 🤔 왜? 보폭(lr)이 크면 한 번에 값이 크게 바뀌어요. 무작위에서 출발한 Head는 크게 움직여야 빨리 배우지만, 이미 좋은 값 근처의 Backbone을 크게 움직이면 잘 배운 특징이 망가져요. 많이 열수록 망가질 수 있는 부분이 늘어나니 보폭을 더 줄여요.

⚠️ 전이학습이 항상 더 좋은 건 아니다 — 원래 학습한 데이터(Source)의 특징이 우리 데이터(Target)의 배경·잡음·색과 안 맞으면 Head-only가 낮게 나올 수 있어요. 강의의 설명용 소형 모델 예시(ResNet18 결과 아님)에서는 Scratch(처음부터) Macro F1 0.911, Head-only 0.269, Partial 0.674였어요. 이럴 땐 실패로 숨기지 말고 도메인 차이와 학습 조건을 설명합니다. Partial도 학습률·epoch·불균형 때문에 다시 한 클래스로 치우칠 수 있어요.

| 전략 (설명용 소형 모델) | 학습 파라미터 | 총 파라미터 | 비율 |
|---|---|---|---|
| Head-only | 99 | 6,131 | 1.6% |
| Partial FT | 4,739 | 6,131 | 77.3% |
| Scratch / Full | 6,131 | 6,131 | 100% |

lr과 epoch: lr은 보폭, epoch는 학습 예산. lr이 너무 작으면 정해진 epoch 안에 덜 배우고, 너무 크면 초반 loss가 튀어요. 그리고 마지막 epoch가 가장 좋은 epoch는 아니라서 valid loss가 가장 낮았던 순간(best)을 따로 저장해 둡니다.

[그림: 작은 데모 CNN의 epoch별 train loss와 valid loss 곡선, best epoch 13 표시] 무엇을 보나: 작은 데모 CNN(ResNet18 아님)을 수업 예제 데이터로 15 epoch 학습한 실제 곡선이에요. train loss(보라)는 꾸준히 내려가지만 valid loss(주황)는 들쭉날쭉해서, 가장 낮은 곳은 epoch 13(0.294)이고 마지막 epoch 15는 1.649로 다시 나빠졌어요. 🤔 왜? 학습은 train만 보고 움직이니, 어느 epoch가 "처음 보는 사진"에 가장 좋은지는 보장하지 않아요(lr 0.01로 작은 데이터에선 특히 출렁여요). 그래서 매 epoch valid loss를 재고 가장 좋았던 순간의 가중치를 deepcopy해 두었다가 마지막에 되돌려요.

🔧 층 열기 · 층별 학습률 명령어 사전에서 자세히 →

```
{"params": model.layer4.parameters(), "lr": 1e-4},   # Backbone: 작게
{"params": model.fc.parameters(),     "lr": 1e-3},   # 새 Head: 크게
```

🔧 학습 루프 + best 복원 명령어 사전에서 자세히 →

```
best_loss, best_state = valid_loss, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)                 # best로 되돌리기
```

## ④ 클래스 불균형 — 적은 클래스를 포기하지 않게

train이 circle 120 · square 40 · triangle 20장이라면, 모델이 전부 circle이라고만 해도 Accuracy는 120/180 ≈ 66.7%나 돼요. 하지만 triangle은 하나도 못 찾죠. 전체 점수만 보면 이 실패가 안 보여요.

🤔 왜 모델이 다수 클래스로 쏠려? — Loss는 모든 샘플의 오차를 평균해요. circle이 2/3를 차지하면 "일단 circle"이라고만 해도 평균 오차가 꽤 줄어서, 모델은 그 쉬운 길부터 배워요. triangle 20장의 오차는 평균 속에서 목소리가 작아요. 그래서 Loss의 배점을 바꾸거나(class weight), 문제를 내는 빈도를 바꿔요(Sampler).

⚠️ train에서만 Loss·Sampling을 조정하고, valid/test는 실제 평가 목적에 맞는 분포를 그대로 유지해요. 🤔 왜? — 평가는 "실제로 쓸 때 얼마나 잘하나"를 재는 거라, 평가 데이터를 바꾸면 그 질문 자체가 바뀌어요. train 분포와 실제 운영 분포가 다르면 어떤 균형 전략이 필요한지도 따로 정해요.

| 방법 | 바꾸는 것 | 장점 | 주의점 |
|---|---|---|---|
| Class weight | Loss 기여도 (배점) | 구현이 단순 | 가중치가 너무 크면 불안정 |
| Weighted sampler | Batch에 뽑히는 빈도 (출제 빈도) | 소수 클래스를 더 자주 학습 | 같은 샘플 중복 추출·확률 기록 필요 |
| 데이터 수집·증강 | 학습 자료 범위 | 데이터 범위 보완 | 라벨 의미·편향 검증 필요 |
| Threshold·후처리 | 운영 판단 규칙 | 업무 비용 반영 | 다중 클래스·확률 보정 고려 |

💡 class weight = 배점: 뽑는 횟수는 그대로, 적은 클래스 오답에 큰 벌점. Sampler = 출제 빈도: 벌점은 그대로, 적은 클래스를 더 자주 뽑음. 둘을 같이 쓰면 보정이 두 번 들어가요. 그리고 불균형 대응은 "세게"가 아니라 "균형" — 너무 세게 주면 이번엔 소수 클래스 쪽으로 치우쳐 오히려 나빠집니다. 두 방법은 같은 seed·epoch에서 별도 후보로 비교해요.

🔧 Class weight nn.CrossEntropyLoss(weight=) 명령어 사전에서 자세히 →

```
class_weight = counts.sum() / counts             # 적을수록 큼 (역빈도)
loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
```

🔧 WeightedRandomSampler 명령어 사전에서 자세히 →

```
sample_weight = class_weight[targets]            # 클래스 weight → 샘플별 weight로 바꿈
sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
```

[그림: class weight 막대와 보통 추출 대 WeightedRandomSampler의 배치별 클래스 수 상자그림] 무엇을 보나: ① class weight는 circle 0.30 · square 0.90 · triangle 1.80 — 개수의 역수라 적을수록 커요. ②③은 120 : 40 : 20 데이터에서 32장 배치를 300번 실제로 뽑아 센 결과예요. 보통 추출은 배치당 평균 21.3 · 7.1 · 3.6장, WeightedRandomSampler는 10.8 · 10.3 · 10.9장으로 거의 1/3씩. 🤔 왜? Sampler는 샘플마다 1/개수의 무게를 줘서 클래스 합이 같아지니, 평균적으로 각 클래스가 같은 비율로 뽑혀요. 하지만 상자(배치마다 흔들리는 범위)가 3~19장으로 넓듯이 매 배치 정확히 같지는 않아요 — 확률을 맞춘 것이지 개수를 고정한 게 아니라서요. class weight는 뽑는 횟수는 ②처럼 그대로 두고 벌점만 바꿔요.

[그림: 전부 circle로 찍은 두 경우와 Sampler 예시의 Accuracy, Weighted F1, Macro F1 비교] 무엇을 보나: "전부 circle"이라고만 답하는 모델은 120 : 40 : 20 데이터에서 Accuracy 0.667, Weighted F1 0.533이지만 Macro F1은 0.267이에요(4 : 2 : 1 예시도 0.571 → 0.242). 반면 세 클래스를 고르게 맞히는 Sampler 예시는 세 지표가 모두 약 0.81로 같아요. 🤔 왜? Accuracy·Weighted F1은 장수가 많은 클래스에 표를 많이 줘서, 다수 클래스만 맞혀도 점수가 높아요. Macro F1은 클래스마다 한 표씩이라 square·triangle의 F1 0이 그대로 평균을 끌어내려요. 세 지표가 벌어지면 불균형이 성적을 가리고 있다는 신호예요.

## ⑤ 평가와 오류 분석 — 어디서, 왜 틀렸나?

### Confusion Matrix (혼동 행렬)

행 = 실제, 열 = 예측. 대각선이 맞힌 것이고, 대각선 밖이 "무엇을 무엇으로 헷갈렸나"예요.

| 실제 ＼ 예측 | circle | square | triangle |
|---|---|---|---|
| circle | 30 | 0 | 0 |
| square | 8 | 22 | 0 |
| triangle | 5 | 4 | 21 |

→ (Sampler 합성 예시) circle은 30/30을 찾았지만, triangle은 21/30만 찾았고 나머지는 circle·square로 착각했어요. Macro F1 ≈ 0.810.

[그림: 위 혼동 행렬을 개수와 행 기준 비율 두 가지 색 지도로 그린 그림] 무엇을 보나: 위 표를 색으로 칠했어요(진할수록 큼). 초록 테두리 대각선이 맞힌 것, 빨간 점선은 circle 열로 몰린 오답 13개(square 8 + triangle 5). 오른쪽은 각 행을 그 행의 합으로 나눈 비율이라 대각선이 곧 클래스별 Recall(1.00 · 0.73 · 0.70)이에요. 🤔 왜? 오답이 한 열에 몰렸다는 건 모델이 헷갈리면 "circle"로 도망간다는 뜻이에요 — train에 circle이 가장 많았던 흔적일 수 있어요. 클래스마다 개수가 다를 땐 raw 숫자만 보면 큰 클래스가 진해 보이니, 행 비율로 바꿔 클래스끼리 공정하게 비교해요.

🤔 왜 숫자 하나 말고 행렬? — Accuracy는 "몇 개 맞혔나"만 알려 줘요. 행렬은 무엇을 무엇으로 헷갈렸는지 보여 줘서, "square를 circle로 자주 착각한다" 같은 다음 실험의 실마리를 줘요.

### 클래스별 지표

| 지표 | 공식 | 질문 |
|---|---|---|
| Precision | 그 클래스로 예측한 것 중 정답 비율 | 예측을 얼마나 믿을 수 있나? |
| Recall | 실제 그 클래스 중 찾아낸 비율 | 실제 대상을 얼마나 안 놓쳤나? |
| F1 | 2 × P × R / (P + R) — 조화평균 | P와 R을 한 숫자로 요약하면? |
| Support | 실제 그 클래스 표본 수 | 몇 장을 근거로 한 숫자인가? |

🤔 왜 F1은 "조화"평균? — 그냥 평균이면 Precision 1.0 · Recall 0.0도 0.5로 그럴듯해 보여요. 조화평균은 둘 중 작은 쪽에 끌려가서 하나라도 0이면 F1도 0이 돼요. "둘 다 괜찮아야 높은 점수"라는 뜻이에요.

[그림: 위 혼동 행렬의 클래스별 Precision, Recall, F1 막대그래프] 무엇을 보나: 같은 행렬로 계산한 클래스별 지표예요. circle은 Recall 1.00인데 Precision 0.70, triangle은 반대로 Precision 1.00인데 Recall 0.70. F1은 둘 다 0.82로 같아요. 🤔 왜? circle은 모든 circle을 찾았지만 "circle"이라는 예측 43개 중 13개가 다른 클래스라 믿기 어렵고, triangle은 "triangle"이라고 하면 늘 맞지만 30개 중 9개를 놓쳐요. 실패 방향이 정반대여도 F1 하나로는 같아 보이니 P·R을 함께 봐요.

[그림: Precision을 1로 고정하고 Recall을 바꿀 때 산술평균과 F1을 비교한 곡선] 무엇을 보나: Precision을 1.0으로 고정하고 Recall만 0 → 1로 바꿨어요. 그냥 평균(회색 점선)은 Recall이 0이어도 0.5, F1(분홍)은 0에서 출발해요. 🤔 왜? 조화평균은 작은 값에 끌려가는 평균이라, 한쪽이 0이면 결과도 0이에요. 그래서 "하나는 완벽, 하나는 엉망"인 모델이 그럴듯한 점수를 받지 못해요.

### Macro vs Weighted 평균

|  | Macro | Weighted |
|---|---|---|
| 계산 | 클래스별 F1의 단순 평균 | support(개수)를 가중치로 평균 |
| 소수 클래스 | 다른 클래스와 같은 비중 | 다수 클래스에 묻힘 |
| 언제 | 불균형 문제의 균형 확인 | 전체 분포 기준 요약 |

```
예) 실제 circle 4 · square 2 · triangle 1장, 모델은 전부 circle이라고 예측
Accuracy    = 4/7 = 0.571   ← 절반 넘음?!
circle F1   = 0.727,  square F1 = 0,  triangle F1 = 0
Weighted F1 = 0.727 × 4/7 = 0.416
Macro F1    = (0.727 + 0 + 0) / 3 = 0.242   ← 실패가 드러남
```

🤔 왜 불균형엔 Accuracy 대신 Macro F1? — Accuracy와 Weighted F1은 장수가 많은 클래스가 점수를 지배해요. 그래서 다수 클래스만 맞혀도 점수가 높게 나와 소수 클래스의 실패를 가려요. Macro F1은 클래스마다 한 표씩 주니까, 한 클래스라도 0이면 점수가 크게 떨어져요. 결함처럼 적지만 중요한 클래스를 지켜야 할 때 맞는 지표예요. 그래서 Macro F1과 클래스별 Recall·support를 반드시 함께 봐요.

아래 칸의 숫자를 바꿔 보세요. 대각선 밖으로 옮길수록 Macro F1이 Accuracy보다 훨씬 빨리 떨어지는 걸 볼 수 있어요.

🔧 confusion_matrix · classification_report 명령어 사전에서 자세히 →

```
cm = confusion_matrix(y_true, y_pred, labels=labels)
print(f1_score(y_true, y_pred, average="macro"))
```

### 오류를 원인 후보로 나누기

| 후보 | 예 |
|---|---|
| 데이터 부족 | 소수 클래스의 다양한 형태·배경·조명이 부족 |
| 도메인 차이 | 사전학습 표현과 지금 촬영 환경이 다름 |
| 라벨 문제 | 클래스 기준이 모호, 잘못된 라벨, 중복 |
| 모델 문제 | 입력 크기·증강·동결 범위·학습률이 안 맞음 |

⚠️ Confusion Matrix만 보고 원인을 확정하지 않아요. 틀린 이미지를 원본·예측 확률·데이터 출처와 함께 직접 보고(관찰) → 가설을 세우고 → 다음 실험으로 확인합니다. 🤔 왜 확률까지? — 0.51로 아깝게 틀린 것과 0.99로 자신 있게 틀린 것은 전혀 달라요. 자신 있게 틀린 오답은 모델이 엉뚱한 단서를 확실히 믿고 있다는 신호라 더 위험해요.

🔧 오답 이미지를 확률과 함께 모으기 명령어 사전에서 자세히 →

```
conf, pred = probs.max(dim=1)          # 가장 큰 확률과 그 클래스 = 예측
wrong = (pred != y_valid).nonzero().flatten()
```

[그림: 작은 데모 CNN이 맞힌 예 3장과 틀린 예 5장을 실제 라벨, 예측, 확률과 함께 보여 주는 갤러리] 무엇을 보나: 작은 데모 CNN(수업 예제 데이터로 몇 초 학습, Macro F1 0.944)의 실제 valid 결과예요. 윗줄은 맞힌 예, 아랫줄은 오답 5장 전부(T = 실제, P = 예측과 확률). 오답 4장이 실제 circle이고, 확률은 0.50~0.64로 모두 아슬아슬했어요. 또 오답 대부분이 배경이 밝고 도형이 흐린 색이에요. 🤔 왜? 확률을 같이 보면 "자신 있게 틀린"(0.9 이상) 위험한 오답은 없고 경계선에서 흔들린 것뿐이라는 걸 알 수 있어요. 그리고 오답 사진을 직접 보니 "도형과 배경의 대비가 낮을 때 틀린다"는 가설이 생겨요 — 이건 행렬 숫자만으로는 절대 안 보여요. 다음 실험(대비 증강 등)으로 확인할 거리예요.

## ⑥ Grad-CAM — 모델은 어디에 반응했나?

Grad-CAM은 "이 클래스 점수에 사진의 어느 부분이 기여했나"를 열지도(Heatmap)로 보여 줘요. 모델이 물체가 아니라 배경·테두리·글자·촬영 위치 같은 엉뚱한 곳에 기대는지(편향) 찾는 단서가 됩니다.

🤔 왜 Grad-CAM이 필요해? — 지표와 Confusion Matrix는 "얼마나·무엇을" 틀렸는지만 알려 줘요. "왜 그렇게 판단했나"는 안 보여 주죠. 예측이 맞았어도 배경색이나 워터마크로 맞힌 거라면 현장에서 금방 무너져요. Grad-CAM은 모델이 기댄 위치를 보여 줘서 그런 편향 후보를 찾게 해 줘요.

입력 Forward→ 관심 클래스 Logit 선택→ 마지막 Conv Feature Map으로 미분→ Gradient 공간 평균 = 채널별 중요도→ 가중합 → ReLU→ 입력 크기로 확대 원본에 겹치기

```
α_k  = 채널 k의 gradient를 가로·세로로 평균     ← 채널 k가 이 클래스에 얼마나 중요한가
CAM  = ReLU( Σ_k  α_k × FeatureMap_k )          ← 점수를 "올린" 곳만 남김
```

🤔 왜 이렇게 계산? — Feature Map의 각 채널은 "어떤 무늬가 어디에 있나"를 담은 지도예요. gradient는 "그 채널 값이 커지면 클래스 점수가 얼마나 오르나"예요. 그래서 gradient 평균(α)으로 채널마다 중요도를 매기고, 그 중요도로 지도들을 겹쳐 더하면 "점수에 기여한 위치"가 나와요. ReLU는 점수를 낮추는 쪽(음수)을 지워서 근거가 흐려지지 않게 해요.

| 대상 층 | 장점 | 단점 |
|---|---|---|
| 깊은 Conv 층 (보통 마지막 Conv block) | 클래스 의미가 강함 | 공간 해상도가 낮음 (흐릿) |
| 얕은 Conv 층 | 가장자리·질감 해상도 높음 | 클래스 구분력 상대적으로 약함 |

🤔 왜 보통 마지막 Conv block? — 위치 정보가 남아 있는 마지막 층이라서요. 그 뒤(풀링·Head)로 가면 위치가 사라져 지도를 못 그리고, 더 앞쪽은 위치는 선명하지만 아직 "이게 square다" 같은 클래스 의미가 약해요. 여러 층을 비교할 땐 별도 실험으로 기록해요.

[그림: 작은 데모 CNN에서 계산한 circle, square, triangle 이미지의 Grad-CAM 열지도와 겹쳐 보기] 무엇을 보나: ⚠️ 작은 데모 CNN(수업 예제 도형 데이터로 CPU에서 몇 초 학습, ResNet18 아님)의 마지막 Conv(8×8)로 실제 계산한 Grad-CAM이에요. 왼쪽 입력, 가운데 열지도(빨강 = 점수에 기여 큼), 오른쪽 겹쳐 보기. square·triangle은 도형 위·모서리에 반응했지만, circle은 맞혔는데도(P=0.85) 도형이 아니라 배경 쪽에 반응했어요. 🤔 왜? Grad-CAM은 "점수를 올린 위치"를 보여 줄 뿐 "사람이 보기에 옳은 근거"를 보장하지 않아요. circle 행처럼 정답을 맞혀도 엉뚱한 곳(배경 밝기·잡음)에 기댔을 수 있다는 편향 후보가 되고, 그 영역을 가려 점수가 실제로 바뀌는지 확인해야 원인이라 말할 수 있어요. 8×8 지도를 32×32로 늘린 거라 흐릿한 것도 정상이에요.

💡 target을 바꾸면 질문이 바뀐다 — 같은 square 사진이라도 "square 점수의 근거는?"과 "circle 점수의 근거는?"은 다른 질문이라 열지도도 달라져요. 그리고 열지도는 가장 밝은 곳을 1로 맞춘 값이라 밝기 ≠ 점수예요. 확률 0.03인 클래스도 밝게 보일 수 있어요.

[그림: 같은 square 이미지에 target을 circle, square, triangle로 바꿔 계산한 Grad-CAM 비교] 무엇을 보나: 같은 square 사진에서 target만 바꿨어요. target=square(확률 0.99)는 도형 위에, target=circle(0.01)·triangle(0.00)은 구석 배경에 빨간 곳이 생겨요. 🤔 왜? target이 바뀌면 "어느 클래스 점수의 근거인가"라는 질문이 바뀌니 지도도 달라져요. 그리고 지도마다 최댓값을 1로 맞추기 때문에 확률이 0.00인 클래스도 진한 빨강이 나와요 — 색이 진하다고 그 클래스일 가능성이 높다는 뜻이 아니에요.

⚠️ Grad-CAM의 한계

- 해상도가 낮고, 고른 층·클래스·정규화 방식에 따라 결과가 달라진다.

- 강한 색이 반드시 사람에게 의미 있는 물체 부분은 아니다 → 인과 설명이 아니라 모델 의존성의 단서.

- 그래서 해당 영역을 가리거나 바꾼 입력(Occlusion·Counterfactual)으로 점수가 실제로 바뀌는지, 데이터 Slice별 지표와 같이 확인한다.

- Grad-CAM은 분석용 gradient만 쓴다. optimizer.step()으로 모델을 바꾸는 단계가 아니다.

💡 편향 점검 순서 — ① 모델이 물체보다 배경색·테두리·문자·촬영 위치에 집중하는지 본다 → ② 클래스마다 카메라·해상도·압축률이 다른지 메타데이터와 연결한다 → ③ 그 영역을 가린 입력으로 예측 변화를 확인한다 → ④ 데이터를 고친 뒤 같은 오류 Slice와 Grad-CAM을 다시 비교한다.

🔧 Hook으로 activation · gradient 기록 명령어 사전에서 자세히 →

```
handles.append(target_layer.register_forward_hook(save_act))
handles.append(output.register_hook(save_grad))      # 이 텐서의 gradient도 기록
```

🔧 채널 weight와 Heatmap 계산 명령어 사전에서 자세히 →

```
alpha = grads.mean(dim=(1, 2))        # 채널별 중요도
cam = torch.relu((alpha[:, None, None] * acts).sum(0))
```

## ⑦ 비교 · 저장 · 재현 — 다른 사람도 같은 결과를 얻나?

| 고정할 조건 | 기록할 변경 |
|---|---|
| 분할 · 클래스 매핑 | Weight · 시작 checkpoint |
| 평가 입력 · 주 지표 | 학습 범위 · 학습률 · 학습량(epoch) |
| 비교 기준 (valid로 선택, test로 최종 확인) | BN 정책 · Loss · Sampler · seed |

주 지표는 Macro F1, 보조 지표는 Accuracy·클래스별 Recall·시간·학습 파라미터 수. 후보 선택 규칙(예: Macro F1 → valid loss 순)도 미리 정하고, 결과를 본 뒤 기준을 바꾸지 않아요. 순차 학습(Head-only → Partial)이면 앞 단계의 학습량도 기록해요.

🤔 왜 한 번에 하나만 바꿔? — lr과 Sampler를 동시에 바꿨는데 좋아졌다면, 어느 쪽 덕분인지 알 수 없어요. 나머지 조건을 고정해야 차이를 그 변경 하나의 효과로 해석할 수 있어요. 결과를 본 뒤 규칙을 바꾸면 "좋아 보이는 쪽"을 고르게 되니 규칙은 미리 정해요.

🔧 재현 · 저장 torch.manual_seed() · torch.save() · torch.load() 명령어 사전에서 자세히 →

```
torch.manual_seed(14)                         # 맨 앞에서 무작위 시작값 고정
torch.save({"weights": best_state, "classes": train_set.classes,
```

| 상황 | 기대 결과 | 확인 근거 |
|---|---|---|
| 정상 3클래스 폴더 | Batch·라벨·학습 정상 | class_to_idx · Shape |
| minority 1장 | 지표 계산 가능, 경고 기록 | support · zero_division |
| train/test 중복 | 실행 전 감사 실패 | 파일 / 지각 해시 |
| Weight 미준비 | 명확한 안내 후 중단 | 임의 random weight 대체 금지 |
| 저장 모델 복원 | 동일 입력 Logit이 허용 오차 안에서 일치 | classes · transform · state_dict |

💡 같은 설정으로 다시 실행했을 때 결과가 다르면 Weight · transform · seed · class_to_idx부터 확인해요. 그리고 epoch 수가 같아도 학습 시간까지 같지는 않아요.

[그림: 같은 설정에서 seed만 바꿔 학습한 작은 데모 CNN의 valid Macro F1 점 그래프] 무엇을 보나: 작은 데모 CNN을 데이터·구조·epoch(15)·lr을 모두 같게 두고 torch.manual_seed만 바꿔 6번 학습한 실제 결과예요. valid Macro F1이 seed 3은 0.734, seed 2는 1.000 — 0.266이나 차이 나요. 🤔 왜? seed는 첫 가중치와 배치 순서를 정해요. 데이터가 작고 epoch가 짧으면 출발점·순서 운이 결과를 크게 흔들어요. 그래서 "A 설정이 B보다 0.05 좋다"는 결론은 seed 하나로 내리면 안 되고, 여러 seed의 범위를 비교하거나 seed를 고정·기록해 같은 조건에서 비교해요.

## 원본 자료

- 강의: Day14_20261105.pdf (p.2–11 큰 그림, p.17–26 데이터셋, p.27–46 전이학습·Fine-tuning, p.47–58 불균형, p.59–68 평가·오류, p.69–79 Grad-CAM, p.80–99 비교·프로젝트·Weight 점검)

- 요약: Day14_oneday요약.pdf (대표 코드 흐름, 파라미터별 변화)

- 공식 문서: Torchvision Models (resnet18 · Weights · transforms) · ImageFolder

- 공식 문서: CrossEntropyLoss · WeightedRandomSampler

- 공식 문서: scikit-learn classification_report · 논문: Selvaraju et al., Grad-CAM, ICCV 2017

## 🔧 Day14 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day14)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install torch torchvision scikit-learn opencv-python Pillow |
| ① 데이터셋 설계 · 누수 | ImageFolder(), class_to_idx, Counter(targets), GroupShuffleSplit |
| ② Backbone · Head | resnet18(weights=), weights.transforms(), requires_grad, nn.Linear, model.eval() |
| ③ Fine-tuning 범위 · 학습률 | layer4.requires_grad_(True), AdamW([{"params", "lr"}]), best 복원 |
| ④ 클래스 불균형 | CrossEntropyLoss(weight=), WeightedRandomSampler |
| ⑤ 평가 · 오류 분석 | confusion_matrix(), classification_report(), f1_score(average=), softmax |
| ⑥ Grad-CAM | register_forward_hook(), register_hook(), interpolate() |
| ⑦ 비교 · 저장 · 재현 | torch.manual_seed(), torch.save(), torch.load(weights_only=True) |

### 클래스 매핑 · 개수 확인 ImageFolder()

```
from collections import Counter
from torchvision.datasets import ImageFolder

train = ImageFolder("data/train")
valid = ImageFolder("data/valid")
test  = ImageFolder("data/test")
assert train.class_to_idx == valid.class_to_idx
assert train.class_to_idx == test.class_to_idx
print(train.class_to_idx)              # {'circle': 0, 'square': 1, 'triangle': 2}
for ds in [train, valid, test]:
    print(ds.root, Counter(ds.targets)) # train: Counter({0: 120, 1: 40, 2: 20})
```

▶ 실제로 돌려 보면 강의 데이터 data/train · valid · test

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| train.class_to_idx | {'circle': 0, 'square': 1, 'triangle': 2} | 폴더 이름 알파벳 순서 |
| train.class_to_idx == valid.class_to_idx == test.class_to_idx | True | 분할마다 번호가 같아야 비교가 됨 |
| Counter(train.targets) | {0: 120, 1: 40, 2: 20} | 6 : 2 : 1로 기울어 있음 |
| Counter(valid.targets) | {0: 30, 1: 30, 2: 30} | 균형 |
| Counter(test.targets) | {0: 10, 1: 10, 2: 10} | 균형 |
| train.samples[0] | ('…/train/circle/circle_000.png', 0) | (파일 경로, 라벨) |
| train[0][0].size, train[0][0].mode | (40, 40), 'RGB' | transform이 없어서 PIL 이미지 그대로 |
| 전부 circle이라고 찍었을 때 train 정확도 | 0.667 | 모델이 아무것도 안 배워도 66.7% |

| 인자 / 속성 | 뜻 | 바꾸면? |
|---|---|---|
| root (필수) | 클래스 폴더들이 있는 상위 경로 | 클래스 폴더 하나 위를 가리켜야 함 (분할 폴더 자체) |
| transform | 이미지 변환 (Resize, ToTensor 등) | 안 주면 PIL 이미지 그대로 나와서 모델에 바로 못 넣음 |
| target_transform | 라벨 변환 | 보통 그대로 둠 |
| is_valid_file | 파일 검사 함수 | 숨김 파일·이상한 확장자를 걸러낼 때 |
| allow_empty=False | 빈 클래스 폴더 허용 여부 | 기본은 빈 폴더가 있으면 에러 → 클래스가 조용히 사라지는 걸 막아 줌 |
| .class_to_idx / .classes | 폴더명 → 번호 사전 / 클래스 이름 목록 |  |
| .targets / .samples | 라벨 번호 목록 / (파일 경로, 라벨) 목록 | Counter(targets)로 불균형 확인 |

### 그룹 단위 분할 (예시 도구: GroupShuffleSplit)

```
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
tr_idx, va_idx = next(gss.split(rows, groups=rows.group_id))   # 같은 group_id는 한쪽에만
```

▶ 실제로 돌려 보면 합성 표: 원본 20장 × (원본·flip·rot) = 60행, group_id = 원본 번호

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(tr_idx), len(va_idx) | 45, 15 | 그룹 20개 중 25%인 5그룹 = 15행이 valid |
| sorted(set(rows.group_id[va_idx])) | [0, 1, 8, 15, 17] | valid로 간 원본 번호 |
| set(train 그룹) & set(valid 그룹) | set() | 겹치는 원본 없음 → 누수 없음 |
| train_test_split(이미지 단위, test_size=0.25)로 나누면 겹치는 원본 수 | 11 | 같은 원본의 변형이 양쪽에 → valid 점수가 부풀어요 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| groups | 각 샘플의 그룹 ID (원본·제품·영상 번호) | 안 주면 그냥 이미지 단위 무작위 분할 → 누수 위험 |
| test_size | 떼어 낼 비율 (그룹 기준) | 그룹 크기가 제각각이면 실제 이미지 비율은 정확히 안 맞을 수 있음 |
| random_state | 무작위 고정값 | 같은 값이면 같은 분할 재현 |

### 사전학습 모델 불러오기 · Head 교체

```
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)       # ① 학습된 가중치 불러오기
for p in model.parameters():
    p.requires_grad = False             # ② 기존 파라미터 얼리기
in_features = model.fc.in_features      # 512
model.fc = nn.Linear(in_features, num_classes)   # ③ 새 Head (새로 만든 층은 기본으로 학습 가능)
preprocess = weights.transforms()       # 이 가중치가 원하는 평가용 입력 변환
```

▶ 실제로 돌려 보면 ResNet18 — 사전학습 가중치 없이 구조만(weights=None, 다운로드 안 함) · 강의 train circle 한 장(40×40)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| weights | ResNet18_Weights.IMAGENET1K_V1 | DEFAULT가 가리키는 실제 이름 |
| model.fc (교체 전) | Linear(in_features=512, out_features=1000, bias=True) | ImageNet 1000클래스 Head |
| model.fc.in_features | 512 |  |
| model.fc (교체 후) | Linear(in_features=512, out_features=3, bias=True) | 우리 3클래스 Head |
| model.fc.weight.requires_grad | True | 얼린 뒤에 만들어서 학습 대상 |
| preprocess | resize [256] → crop [224] mean [0.485, 0.456, 0.406] std [0.229, 0.224, 0.225] | 평가용 입력 규칙 |
| preprocess(40×40 PIL).shape | [3, 224, 224] | 작은 사진도 224×224로 키움 |
| model(x).shape | [1, 3] | [1, 3] 점수 |
| 순서 실수: fc를 먼저 바꾸고 전체를 얼림 → 학습 파라미터 | 0 | 0개! 에러도 안 남 |

🤔 왜 이 순서(불러오기 → 얼리기 → 새 Head)? — 얼린 다음에 새 Head를 만들어야, 새 층은 기본값 requires_grad=True 그대로라 학습 대상이 돼요. 순서를 바꾸면 아래 "순서 실수"가 생겨요.

| 인자 / 코드 | 뜻 | 바꾸면? |
|---|---|---|
| weights=ResNet18_Weights.DEFAULT | 사전학습 가중치 사용 (키워드 전용 인자) | None(기본값)이면 무작위 시작 → 특징을 못 뽑는 Backbone 위에선 Head도 못 배움 |
| progress=True | 다운로드 진행 표시 | False면 조용히 받음 |
| model.fc.in_features | Head에 들어오는 특징 수 (ResNet18 = 512) | 숫자를 직접 적지 않고 읽어 오면 다른 모델로 바꿔도 안 깨짐 |
| nn.Linear(in, out) | out = 우리 클래스 수 | 클래스 수와 다르면 Loss 계산에서 에러 또는 엉뚱한 학습 |
| weights.transforms() | Resize → CenterCrop → 0~1 값 변환 → ImageNet mean/std Normalize | 임의로 바꾸면 특징 분포가 달라짐. 학습용 증강도 같은 입력 규격을 지켜서 만듦 |

⚠️ 순서 실수 주의 — Head를 먼저 붙이고 나서 model.parameters() 전체를 얼리면 새 Head까지 얼어서 아무것도 배우지 않아요. 에러 메시지도 안 나요! 그래서 학습 가능한 파라미터 수 세기처럼 학습 파라미터 수를 꼭 출력해 확인합니다.

### 학습 가능한 파라미터 수 세기

```
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print("trainable:", trainable, "/ total:", total)   # Head-only면 Head 몫(1,539)만 나와야 정상

for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)                                  # fc.weight, fc.bias
```

▶ 실제로 돌려 보면 Head 교체한 ResNet18 구조 (3클래스, weights=None이어도 개수는 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| trainable | 1,539 | 512 × 3 + 3 |
| total | 11,178,051 | 전체의 0.014%만 학습 |
| [name for name, p in model.named_parameters() if p.requires_grad] | ['fc.weight', 'fc.bias'] | 열린 층 이름 |
| 순서 실수 모델의 trainable | 0 | 이 숫자가 0이면 바로 의심 |

| 코드 | 뜻 | 🤔 왜 확인? |
|---|---|---|
| p.numel() | 그 파라미터 텐서 안의 숫자 개수 | 학습 범위와 계산 비용을 실험표에 같이 기록하려고 |
| p.requires_grad | 학습 대상이면 True | 고정했다고 생각한 층이 실제로 열려 있지 않은지 확인 |
| model.named_parameters() | (이름, 파라미터) 쌍으로 돌기 | 어느 층이 열렸는지 이름으로 보고, Optimizer에 넘긴 목록과 일치하는지 대조 |

### 층 열기 · 층별 학습률

```
for p in model.layer4.parameters():
    p.requires_grad = True              # 마지막 Stage 열기 (= model.layer4.requires_grad_(True))

optimizer = torch.optim.AdamW([
    {"params": model.layer4.parameters(), "lr": 1e-4},   # Backbone: 작게
    {"params": model.fc.parameters(),     "lr": 1e-3},   # 새 Head: 크게
], weight_decay=1e-4)
```

▶ 실제로 돌려 보면 Head 교체한 ResNet18 구조(weights=None)에서 layer4를 연 뒤

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| trainable (layer4 + fc) | 8,395,267 | 전체의 75.1% — 갑자기 수천 배 |
| [(len(g['params']), g['lr'], g['weight_decay']) for g in optimizer.param_groups] | [(15, 0.0001, 0.0001), (2, 0.001, 0.0001)] | 그룹마다 Tensor 수 · lr · weight_decay |
| 층을 열기 전에 만든 optimizer가 관리하는 Tensor 수 | 2 | fc 2개뿐 → layer4는 열어도 안 바뀜 |
| 같은 파라미터를 두 그룹에 넣으면 | ValueError: some parameters appear in more than one parameter group… | 중복은 에러 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| "params" | 이 그룹에서 학습할 파라미터 | 같은 파라미터를 두 그룹에 넣으면 에러, 빠뜨리면 그 층은 학습 안 됨 (Optimizer는 받은 목록만 고치니까) |
| "lr" | 그룹별 학습률 | 크면 빨리 배우지만 불안정, Backbone에 크게 주면 사전학습 특징이 망가짐 |
| weight_decay | 가중치가 너무 커지지 않게 억제 (그룹별로도 지정 가능) | 0이면 억제 없음 → 작은 데이터에서 과적합 쉬움 |
| betas·eps | (선택) AdamW의 내부 설정 | 보통 기본값 사용 |

⚠️ 자주 하는 실수

- 층을 새로 연 뒤 Optimizer를 그대로 씀 → 새로 연 층은 Optimizer 목록에 없어서 안 바뀜. 다시 만들거나 그룹을 추가(add_param_group()).

- Head-only인데 Backbone을 안 얼리고 전체를 Optimizer에 넣음.

- 사전학습 Weight의 Normalize를 빠뜨려 입력 분포가 달라짐.

- valid 점수를 보면서 epoch·층·학습률을 계속 바꾸고 그걸 최종 성능처럼 보고함.

### 학습 루프 + best 복원

```
import copy
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
criterion = nn.CrossEntropyLoss()
best_loss = float("inf")
for epoch in range(8):
    model.train()
    backbone.eval()                               # 얼린 Backbone의 BN 기억값 고정 (정책)
    optimizer.zero_grad()                         # 1) 지난 gradient 지우기
    loss = criterion(model(X_train), y_train)     # 2) 예측하고 오차 계산
    loss.backward()                               # 3) gradient 계산
    optimizer.step()                              # 4) 한 걸음 이동
    with torch.no_grad():
        valid_loss = criterion(model(X_valid), y_valid)   # valid는 채점만
    if valid_loss < best_loss:                    # 가장 좋았던 순간 기억
        best_loss, best_state = valid_loss, copy.deepcopy(model.state_dict())
model.load_state_dict(best_state)                 # best로 되돌리기
```

▶ 실제로 돌려 보면 강의 train 180장(40×40) 통째로 한 batch · 작은 CNN Backbone(seed 0 무작위, 얼림) + Linear Head · lr 0.01 · 8 epoch

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 학습 파라미터 수 (Head만) | 99 | 32 × 3 + 3 |
| epoch 0: valid loss | 1.1187 |  |
| epoch 1: valid loss | 1.1452 |  |
| epoch 2: valid loss | 1.1794 |  |
| epoch 3: valid loss | 1.2190 |  |
| epoch 4: valid loss | 1.2613 |  |
| epoch 5: valid loss | 1.3033 |  |
| epoch 6: valid loss | 1.3426 |  |
| epoch 7: valid loss | 1.3769 |  |
| best_epoch, best_loss | 0, 1.1187 | 가장 좋았던 순간 |
| 마지막 epoch 모델의 valid loss | 1.3769 | 복원 안 하면 이 모델이 남음 |
| best 복원 후 클래스별 valid Recall (circle·square·triangle) | 1.000 0.000 0.000 | 전부 circle로 답함 |
| (해석 — 예상과 다르게 나온 결과) | valid loss가 epoch 0부터 오르기만 해서 best = 처음 | 무작위로 얼린 특징으론 모양을 못 가르고, Head가 train의 6 : 2 : 1 비율만 배워 '일단 circle'로 기움 → 균형 잡힌 valid에선 loss가 올라감. best 복원이 '나빠진 마지막 모델'을 막아 준 예이자, 사전학습 Backbone이 왜 필요한지 보여 주는 예 |

backbone은 모델에서 Head를 뺀 앞부분이에요(ResNet18이면 fc 앞까지).

| 코드 | 뜻 | 바꾸면? |
|---|---|---|
| [p for p in ... if p.requires_grad] | 열린 파라미터만 Optimizer에 넘김 | 전부 넘기면 얼린 층에 쓸데없는 상태(메모리)가 생기고 실수 확인도 어려움 |
| lr=0.01 | 보폭 | 0.001로 줄이면 같은 epoch 안에 덜 배움, 0.1로 키우면 초반 valid loss가 크게 튐 |
| range(8) | epoch 수 (학습 예산) | 1로 줄이면 소수 클래스를 하나도 못 찾을 수 있음 — 다수 클래스부터 배우기 때문 |
| with torch.no_grad() | valid 계산에선 gradient를 안 만듦 | 빼면 메모리 낭비. valid는 채점만 하고 학습에 쓰지 않음 |
| copy.deepcopy(state_dict()) | 그 순간의 가중치를 복사해 보관 | deepcopy 없이 담으면 학습이 계속되며 같이 바뀜. 복원 줄을 지우면 다시 나빠진 마지막 epoch 모델이 남음 |

### Class weight nn.CrossEntropyLoss(weight=)

```
counts = torch.tensor([120, 40, 20], dtype=torch.float)
class_weight = counts.sum() / counts             # 적을수록 큼 (역빈도)
class_weight = class_weight / class_weight.mean()   # 평균 1로 맞춤
print(class_weight)                              # tensor([0.30, 0.90, 1.80])
loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(device))
```

▶ 실제로 돌려 보면 강의 train 개수 [120, 40, 20] · 'best 복원' 항목과 같은 모델·seed·8 epoch에 weight만 추가

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| torch.bincount(train targets) | [120, 40, 20] |  |
| class_weight | 0.30 0.90 1.80 | 역빈도를 평균 1로 — 적은 클래스일수록 큼 |
| (counts * class_weight).tolist() | 36 36 36 | 클래스마다 '개수 × 무게'가 같아짐 |
| 세 샘플 모두 logit [2, 0, 0], 정답 0·1·2 → 샘플별 loss | 0.240 2.240 2.240 | circle만 맞힘 |
| weight 없음 (평균) | 1.573 |  |
| weight 있음 | 2.040 | Σ wᵢ·lossᵢ ÷ Σ wᵢ — 틀린 square·triangle 쪽 벌점이 더 큼 |
| valid Recall (weight 없음) | 1.000 0.000 0.000 | 'best 복원' 항목 결과 |
| valid Recall (weight 있음) | 0.367 0.733 0.500 | square·triangle은 0 → 0.73·0.50로 올랐지만 circle은 1.00 → 0.37로 떨어짐 — 공짜가 아님. 이 작은 실험의 결과일 뿐 일반 법칙은 아니에요 |

🤔 왜 역빈도(전체 ÷ 개수)? — 개수가 1/6인 triangle에 6배 무게를 주면, 클래스마다 "개수 × 무게"가 비슷해져서 Loss 안에서 세 클래스의 목소리 크기가 같아져요. 평균 1로 나누는 건 전체 Loss 크기를 원래와 비슷하게 유지해 학습률 감각을 바꾸지 않으려는 선택이에요(절댓값보다 상대 비율이 중요).

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| weight | 길이 C(클래스 수) 텐서. 클래스별 loss 기여도 | 안 주면 모든 클래스 같은 배점. 1/개수²처럼 너무 세게 주면 오히려 나빠질 수 있음 |
| reduction="mean" | "mean"(기본, scalar 반환) / "sum" / "none" | none이면 샘플별 loss 그대로 → 어떤 샘플이 많이 틀리는지 볼 때 |
| ignore_index=-100 | 계산에서 뺄 라벨 값 | 라벨 없는 샘플을 표시할 때 |
| label_smoothing=0.0 | 정답을 100%가 아니라 살짝 부드럽게 | 올리면 과하게 확신하는 걸 줄여 줌 |
| .to(device) | weight를 모델과 같은 장치로 | GPU 모델에 CPU weight를 주면 에러 |

💡 class weight는 train loss만 바꿔요. valid의 클래스 수나 평가 지표 계산식은 안 바뀝니다. 그래서 학습 안정성과 클래스별 Recall을 함께 봐요.

### WeightedRandomSampler

```
from torch.utils.data import DataLoader, WeightedRandomSampler
targets = torch.tensor(train_dataset.targets)
counts = torch.bincount(targets)
assert torch.all(counts > 0)                     # 빈 클래스가 있으면 1/0이 됨
class_weight = 1.0 / counts.float()
sample_weight = class_weight[targets]            # 클래스 weight → 샘플별 weight로 바꿈
sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)   # shuffle은 쓰지 않음
```

▶ 실제로 돌려 보면 강의 train 180장 (120 · 40 · 20), generator seed 0

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| sample_weight[:2], sample_weight[-2:] | 0.0083 0.0083, 0.0500 0.0500 | circle 1/120 · triangle 1/20 |
| 한 epoch(180번) 뽑은 클래스 수 | {0: 58, 1: 67, 2: 55} | 거의 1/3씩 (정확히 같지는 않음) |
| triangle 뽑힌 횟수 / 서로 다른 triangle 장수 | 55 / 16 | 복원 추출: 16장이 평균 3.4번씩 뽑히고 4장은 이번 epoch에 한 번도 안 뽑힘 |
| batch(32장)당 클래스 수 평균 [최소~최대] sampler, 50 epoch | 10.6 [5~19] · 10.5 [4~19] · 10.8 [4~19] | 평균은 거의 1/3씩, 하지만 batch마다 흔들림 |
| 같은 통계 shuffle=True | 21.4 [10~27] · 7.1 [0~16] · 3.6 [0~9] | 보통 추출은 평균 6 : 2 : 1 그대로 |
| 클래스 weight 3개를 그대로 넣으면 → 뽑힌 인덱스 | [2, 2, 2, 2, 2, 2, 1, 1, 2, 1] | 0·1·2번 '샘플'만 뽑힘 — 샘플별 weight가 필요 |
| DataLoader(…, sampler=sampler, shuffle=True) | ValueError: sampler option is mutually exclusive with shuffle… | 둘은 같이 못 씀 |
| valid Recall (sampler, batch 32 · 8 epoch) | 0.267 0.567 0.533 | Head-only 1.00 0.00 0.00 · class weight 0.37 0.73 0.50와 비교 — 같은 모델·seed의 작은 실험 결과예요 |

🤔 왜 샘플별 1/개수? — circle 한 장은 1/120, triangle 한 장은 1/20의 무게를 받아요. 클래스 전체로 더하면 120 × 1/120 = 20 × 1/20 = 1 → 세 클래스가 뽑힐 확률이 똑같아져요. 그만큼 triangle은 여러 번 다시 뽑혀요.

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| weights (필수) | 샘플별 뽑힐 가중치 (클래스별 아님!) | 클래스 weight 3개를 그대로 주면 "샘플 3개"로 해석돼 엉망이 됨 |
| num_samples (필수) | 한 epoch에 뽑을 개수 | 데이터 수와 같게 두면 epoch 길이가 그대로 |
| replacement=True | 복원 추출 → 같은 샘플이 여러 번 뽑힘 | False면 한 장씩만 뽑혀서 적은 클래스를 더 자주 볼 수 없음 |
| generator | 난수 고정 (재현용) | 안 주면 실행마다 뽑히는 순서가 달라짐 |

⚠️ sampler=를 쓰면 DataLoader의 shuffle=True와 같이 쓰지 않아요(뽑는 순서를 Sampler가 이미 정하니까). 역빈도는 클래스별 뽑힐 확률을 맞출 뿐, 모든 Batch에 똑같은 개수가 들어간다는 보장은 없어요.

| 합성 예시 (valid) | circle Recall | square Recall | triangle Recall | Macro F1 |
|---|---|---|---|---|
| Head-only | 1.000 | 0.167 | 0.000 | 0.269 |
| Class weight | 1.000 | 0.667 | 0.633 | 0.763 |
| Sampler | 1.000 | 0.733 | 0.700 | 0.810 |

강의의 설명용 합성 예시예요. 이 순위가 다른 데이터에서도 유지되지는 않아요.

### confusion_matrix · classification_report

```
from sklearn.metrics import confusion_matrix, classification_report, f1_score
labels = list(range(len(class_names)))           # [0, 1, 2] = class_to_idx 순서
cm = confusion_matrix(y_true, y_pred, labels=labels)
report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                               output_dict=True, zero_division=0)
print(cm)
print(report["macro avg"]["f1-score"])
print(f1_score(y_true, y_pred, average="macro"))
```

▶ 실제로 돌려 보면 강의 valid 90장 (30 · 30 · 30) — 'best 복원'(Head-only)·'Class weight' 항목에서 학습한 두 모델의 예측

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| confusion_matrix (Head-only) | 30 0 0 30 0 0 30 0 0 | 행 = 실제, 열 = 예측 (circle · square · triangle) |
| report['macro avg']['f1-score'] (Head-only) | 0.167 |  |
| f1_score(average='weighted') · accuracy (Head-only) | 0.167 · 0.333 |  |
| confusion_matrix (class weight) | 11 16 3 5 22 3 10 5 15 | 행 = 실제, 열 = 예측 (circle · square · triangle) |
| report['macro avg']['f1-score'] (class weight) | 0.528 |  |
| f1_score(average='weighted') · accuracy (class weight) | 0.528 · 0.533 |  |
| confusion_matrix(y_true[:30], 전부 0) ← labels 없이 (circle만 있는 부분) | [[30]] | 1×1 표 — 없는 클래스가 표에서 사라짐 |
| 같은 입력에 labels=[0, 1, 2] | 30 0 0 0 0 0 0 0 0 | 3×3으로 고정 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| y_true, y_pred (필수) | 실제 라벨, 예측 라벨 | 순서를 바꾸면 Precision과 Recall이 뒤바뀜 |
| labels | 행·열 순서. class_to_idx와 같은 순서로 | 안 주면 데이터에 나온 라벨만 → 예측 0건인 클래스가 표에서 빠질 수 있음 |
| target_names | 보고서에 찍을 클래스 이름 | 순서가 labels와 어긋나면 이름이 틀리게 붙음 |
| output_dict=True | 문자열 대신 중첩 사전으로 반환 → 표·JSON 저장 | False(기본)면 사람이 읽는 문자열 |
| zero_division=0 | 분모가 0일 때(예측 0건) 값을 0으로 | 기본 "warn"은 경고를 띄움. 정책을 정해 두면 결과가 일관됨 |
| average= | "macro" / "weighted" / "micro" | weighted는 소수 클래스 실패를 가릴 수 있음 |
| digits=2, sample_weight | (선택) 소수점 자리 수, 샘플별 가중치 |  |

### 오답 이미지를 확률과 함께 모으기

```
model.eval()
with torch.no_grad():
    logits = model(X_valid)
    probs = logits.softmax(dim=1)          # 클래스별 확률 (합 = 1)
    conf, pred = probs.max(dim=1)          # 가장 큰 확률과 그 클래스 = 예측
wrong = (pred != y_valid).nonzero().flatten()
for i in wrong[:8]:                        # 오분류 최대 8장
    print(paths[i], "T=", class_names[y_valid[i]], "P=", class_names[pred[i]], f"{conf[i]:.2f}")
```

▶ 실제로 돌려 보면 강의 valid 90장 — 'Class weight' 항목에서 학습한 모델(Head-only는 전부 circle이라 오답 분석이 단조로워서)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| probs[0], probs[0].sum() | 0.366 0.401 0.233, 1.000 | 클래스별 확률, 합 1 |
| len(wrong) | 42 / 90 | 틀린 valid 수 |
| Counter(T→P) 오답 유형 | {'circle→square': 16, 'triangle→circle': 10, 'square→circle': 5, 'triangle→square': 5, 'circle→triangle': 3, 'square→triangle': 3} | 어느 쪽으로 틀리나 |
| 오답 확신도 conf[wrong] 최소 · 중앙 · 최대 | 0.34 · 0.37 · 0.44 | 0.9 이상이면 '자신 있게 틀린' 위험 오답 — 여기선 최대 0.44라 그런 오답은 없고, 모델이 전반적으로 확신이 낮음(3클래스 반반이 0.33) |
| paths[21], T, P, conf | circle_021.png T=circle P=square 0.44 | 가장 자신 있게 틀린 오답부터 — 이 사진을 직접 열어 봐야 '왜'가 보여요 |
| paths[25], T, P, conf | circle_025.png T=circle P=square 0.43 |  |
| paths[62], T, P, conf | triangle_002.png T=triangle P=square 0.41 |  |

| 코드 | 뜻 | 🤔 왜? |
|---|---|---|
| softmax(dim=1) | Logit을 0~1 확률로 (클래스 축) | Logit 크기는 해석이 어려워서, "얼마나 확신했나"로 바꿔 봄 |
| max(dim=1) | 가장 큰 확률과 그 위치 = 예측 | argmax와 같은 예측 + 확신도를 한 번에 |
| pred != y_valid | 틀린 샘플만 고르기 | T(실제)·P(예측)·확률·파일 경로를 같이 남겨야 출처까지 추적 가능 |

### Hook으로 activation · gradient 기록

```
activation, gradient, handles = {}, {}, []
def save_grad(grad):
    gradient["value"] = grad.detach()
def save_act(module, inputs, output):
    activation["value"] = output.detach()
    handles.append(output.register_hook(save_grad))      # 이 텐서의 gradient도 기록

model.eval()
image = image.detach().requires_grad_(True)
target_layer = model.layer4[-1]                          # 마지막 Conv block
handles.append(target_layer.register_forward_hook(save_act))
try:
    with torch.enable_grad():                            # eval이어도 gradient는 켬
        model.zero_grad(set_to_none=True)
        logits = model(image)
        logits[0, target_class].backward()               # 관심 클래스 점수로 미분
finally:
    for h in handles:
        h.remove()                                       # 꼭 지우기!
```

▶ 실제로 돌려 보면 ResNet18 구조(weights=None, 3클래스 Head) · 강의 circle 한 장을 preprocess → [1, 3, 224, 224]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| activation['value'].shape | [1, 512, 7, 7] | layer4[-1] 출력: 512채널 · 7×7 지도 |
| gradient['value'].shape | [1, 512, 7, 7] | 같은 모양의 gradient |
| forward hook · tensor hook 호출 횟수 | 1 · 1 | Forward 1번, backward 1번 |
| h.remove() 전/후 target_layer._forward_hooks 수 | 1 → 0 | 지워야 다음 forward에 안 쌓임 |
| no_grad() 안에서 output.register_hook | RuntimeError: cannot register a hook on a tensor that doesn… | gradient를 안 만드는 곳에선 hook을 못 검 |
| image.grad.shape | [1, 3, 224, 224] | 입력에도 gradient가 생김 (requires_grad_(True)) |

🤔 왜 Hook을 써? — 모델은 최종 Logit만 돌려주고 중간 층의 출력(Feature Map)과 그 gradient는 버려요. Hook은 "그 층을 지날 때 값을 몰래 복사해 둬"라는 갈고리라서, 모델 코드를 고치지 않고 중간값을 꺼낼 수 있어요. try/finally로 감싸는 건 중간에 에러가 나도 Hook을 반드시 지우려는 거예요.

| 명령어 | 뜻 | 주의 / 🤔 왜? |
|---|---|---|
| module.register_forward_hook(fn) | 그 층을 지날 때 fn(module, args, output) 호출 | RemovableHandle을 돌려줌 → .remove() |
| tensor.register_hook(fn) | 그 텐서의 gradient가 계산되면 fn(grad) 호출 | Feature Map에 대한 gradient를 얻는 방법 |
| model.eval() + enable_grad() | BN·Dropout은 평가 모드, gradient는 계산 | 평가 모드로 결과를 고정하되, 미분은 필요하니까. inference_mode()·no_grad() 안에선 gradient가 안 생김 |
| target_class | 무엇에 대한 설명인가 (보통 예측 클래스) | 바꾸면 다른 질문이 됨 → target class와 층을 함께 기록 |
| .detach() | 저장할 값을 계산 그래프에서 떼어 냄 | 그래프를 붙잡고 있지 않아 메모리 절약 |

⚠️ Hook을 지우지 않고 계속 등록하면 한 번 Forward에 여러 번 호출되고 메모리가 늘어나요. 그리고 Grad-CAM 중에는 optimizer.step()을 부르지 않아요 — 설명하려다 모델을 바꾸면 안 되니까요.

### 채널 weight와 Heatmap 계산

```
grads = gradient["value"][0]          # [C, H, W]
acts  = activation["value"][0]        # [C, H, W]
alpha = grads.mean(dim=(1, 2))        # 채널별 중요도
cam = torch.relu((alpha[:, None, None] * acts).sum(0))
cam = cam / (cam.max() + 1e-8)        # 0~1로 맞춤
cam = torch.nn.functional.interpolate(cam[None, None], size=image_hw,
                                      mode="bilinear", align_corners=False)[0, 0]
```

▶ 실제로 돌려 보면 'Hook' 항목의 activation·gradient 그대로 — 가중치가 무작위라 지도 모양 자체엔 의미가 없어요(계산 과정 확인용)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| alpha.shape | [512] | 채널마다 중요도 숫자 하나 |
| (alpha * acts).sum(0).shape | [7, 7] | 7×7 지도 |
| relu 전 음수 칸 비율 | 4% | 점수를 낮추는 쪽 칸 — relu가 지움 |
| cam.min(), cam.max() (정규화 후) | 0.00, 1.00 | 가장 밝은 곳 = 1 |
| interpolate(…, size=(224, 224)).shape | [224, 224] | 원본 크기로 확대 |
| 확대 전후 서로 다른 값 개수 | 48 → 37699 | 부드럽게 이어 붙였을 뿐, 49칸 정보가 늘어나진 않음 |

| 코드 | 뜻 | 바꾸면? |
|---|---|---|
| mean(dim=(1, 2)) | gradient를 가로·세로로 평균 | 위치별 잡음을 평균으로 눌러 채널 하나당 중요도 숫자 하나로 |
| torch.relu(...) | 클래스 점수를 올린 영역만 남김 | 빼면 점수를 낮추는 영역까지 칠해져 근거가 흐려짐 |
| cam.max() + 1e-8 | 가장 밝은 곳을 1로 | 1e-8은 cam이 전부 0일 때 0으로 나누는 걸 막음. 이 정규화 때문에 밝기 ≠ 점수 |
| interpolate(size=image_hw, mode="bilinear") | 작은 열지도를 원본 크기로 확대 | 원본에 겹쳐 보려면 크기가 같아야 함. 확대일 뿐 해상도가 생기지는 않음 |

간단히 쓰면 hook 대신 torch.autograd.grad(score[0, target], features)[0]로 특징 지도에 대한 gradient를 바로 받을 수도 있어요.

### 재현 · 저장 torch.manual_seed() · torch.save() · torch.load()

```
torch.manual_seed(14)                         # 맨 앞에서 무작위 시작값 고정

torch.save({"weights": best_state, "classes": train_set.classes,
            "lr": 0.01, "epochs": 8, "seed": 14, "test_used": False}, "model.pt")

saved = torch.load("model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(saved["weights"]); model.eval()
```

▶ 실제로 돌려 보면 작은 Linear(4, 3) 모델로

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| torch.manual_seed(14); torch.rand(3) | [0.5695 0.0047 0.9303] |  |
| 다시 manual_seed(14); torch.rand(3) | [0.5695 0.0047 0.9303] | seed가 같으면 똑같은 난수 |
| 이어서 torch.rand(3) 한 번 더 | [0.7257 0.8295 0.7683] | seed를 다시 안 정하면 다음 난수로 넘어가요 |
| list(saved.keys()) | ['weights', 'classes', 'lr', 'epochs', 'seed', 'test_used'] | 가중치 + 만든 방법을 함께 저장 |
| model.load_state_dict(saved['weights']) | <All keys matched successfully> | missing·unexpected 둘 다 비면 정상 |
| 저장 전후 가중치가 같나? | True |  |

| 명령어 / 인자 | 뜻 | 바꾸면? |
|---|---|---|
| torch.manual_seed(n) | 무작위 시작값 고정 | seed만 바꿔도 결과가 꽤 흔들림 → 한 번 결과로 결론 내지 않고 seed를 바꿔 여러 번 확인 |
| torch.save(dict, path) | 가중치 + 만든 방법(클래스·lr·epoch·seed·test 사용 여부)을 함께 저장 | 가중치만 저장하면 다시 만들 수도, 클래스 번호를 해석할 수도 없음 |
| map_location="cpu" | GPU에서 저장한 걸 CPU로 불러오기 | 빼면 GPU가 없는 컴퓨터에서 에러 |
| weights_only=True | 텐서 같은 안전한 값만 불러옴 | False면 파일 안의 임의 코드가 실행될 수 있음 → 믿을 수 없는 checkpoint는 torch.load 하지 않기 |

## 📖 Day14 용어 (39개)
- **혼동행렬** (Confusion Matrix (혼동행렬)): 실제(행) × 예측(열)으로 맞힘·틀림을 네 칸(TN, FP, FN, TP)에 세어 둔 표. 어떤 종류의 오류인지 나눠 볼 수 있어요.
- **Precision** (Precision (정밀도)): 양성이라고 한 것 중 진짜 양성인 비율. TP ÷ (TP + FP). 헛경고가 늘면 낮아져요.
- **Recall** (Recall (재현율)): 실제 양성 중 모델이 찾아낸 비율. TP ÷ (TP + FN). 놓침이 늘면 낮아져요.
- **F1** (F1 score): Precision과 Recall의 조화평균. 2TP ÷ (2TP + FP + FN). 둘 중 하나라도 낮으면 같이 낮아지고 TN은 반영하지 않아요. 이 수업 분류의 주 지표.
- **누수** (Data Leakage (데이터 누수)): 검증·테스트용 정보나 예측 시점에 알 수 없는 값(예: 실행 후에야 생기는 elapsed_ms)이 학습에 섞여 들어가 점수가 실제보다 좋게 나오는 문제.
- **그룹 단위 분할** (Group split (그룹 단위 분할)): 같은 고객·요청 등 한 그룹의 행이 학습과 평가 양쪽에 동시에 들어가지 않게 그룹째로 나누는 방법.
- **불균형 자료** (Imbalanced data (클래스 불균형 자료)): 한 클래스가 다른 클래스보다 훨씬 많은 자료(예: 성공 14 · 실패 6). Accuracy가 속기 쉬워 실제 양성 수, Precision·Recall 관점도 함께 봐요.
- **동결** (Freeze / Lock (설정 동결)): test를 열기 전에 모델·특성·임계값·선택 규칙을 파일로 기록해 더 이상 바꾸지 않는 것. test 점수를 보고 바꾸면 test가 선택용 자료가 돼 버려요.
- **requires_grad** (requires_grad (기울기 추적 여부)): True면 이 Tensor로 하는 연산을 Autograd가 기록해 기울기를 구해요. 학습할 파라미터는 보통 True, 기본값은 False.
- **logit** (Logits (원시 점수)): 모델이 마지막 층에서 낸, softmax 하기 전의 클래스별 점수. 합이 1일 필요가 없고 확률이 아니에요.
- **BatchNorm** (Batch Normalization (배치 정규화)): 층의 중간값을 현재 batch의 평균·분산으로 정리(정규화)해 학습을 안정시키는 층. 평가 땐 학습 중 모아 둔 running 통계를 써요.
- **Feature Map** (Feature Map (특징 맵)): Kernel 하나가 사진 전체를 훑어서 만든 결과 지도. 그 Kernel이 찾는 패턴이 어디에 얼마나 강하게 있는지를 보여 줘요.
- **Backbone** (Backbone (백본, 특징 추출부)): 모델 앞부분. 사진에서 선·모서리·질감·모양 같은 특징을 뽑아내는 부분이에요. 전이학습에서는 사전학습된 Backbone을 그대로 빌려 써요.
- **Head** (Head (헤드, 분류기)): 모델 끝부분. Backbone이 뽑은 특징을 보고 최종 답(클래스별 점수 등)을 내는 층이에요. 전이학습에서는 내 클래스 수에 맞게 이 부분만 바꿔 끼워요. (Transformer의 Multi-head의 'Head'와는 다른 뜻이에요.)
- **ImageFolder** (torchvision.datasets.ImageFolder): 폴더 구조로 이미지 데이터셋을 만드는 도구. 하위 폴더 이름이 곧 라벨이고, 이름을 정렬해 0, 1, 2… 번호를 붙여요(class_to_idx로 확인).
- **Normalize** (transforms.Normalize(mean, std)): 채널마다 (값 − 평균) ÷ 표준편차로 숫자 범위를 맞춰요. 사전학습 모델을 쓸 땐 그 모델이 배울 때 쓴 평균·표준편차를 그대로 써야 해요.
- **전이학습** (Transfer Learning (전이학습)): 큰 데이터로 미리 학습한 모델을 가져와, 마지막 Head만 내 문제에 맞게 바꿔 다시 배우는 방법. 데이터가 적어도 빨리, 덜 과적합되게 배워요.
- **사전학습** (Pre-training (사전학습)): 다른 큰 데이터(예: ImageNet)로 미리 학습해 두는 것. 이렇게 만든 가중치를 내려받아 출발점으로 써요.
- **ImageNet** (ImageNet (대규모 이미지 데이터셋)): 사진 100만 장 이상, 1000개 클래스로 된 유명한 이미지 데이터셋. 많은 사전학습 모델이 이걸로 배웠어요.
- **ResNet18** (ResNet-18 (Residual Network, 18층)): 층 18개짜리 대표적인 CNN 모델. ImageNet 사전학습 가중치를 torchvision에서 바로 받아 전이학습에 써요. 마지막 Head는 model.fc예요.
- **Fine-tuning** (Fine-tuning (미세조정)): 사전학습된 층의 일부 또는 전체를 다시 열어 작은 학습률(lr)로 살짝만 고치며 학습하는 것. 보통 뒤쪽 층(layer4)부터 열어요.
- **Head-only** (Head-only 전이학습): Backbone은 전부 얼리고 새 Head만 학습하는 방식. ResNet18·2클래스면 학습할 파라미터가 1,026개뿐이라 빠르고 과적합 위험이 작아요.
- **Partial** (Partial fine-tuning (부분 미세조정)): 뒤쪽 일부 층(예: layer4)과 Head만 열어 학습하는 방식. 연 층은 작은 lr(1e-4), 새 Head는 큰 lr(1e-3)로 따로 줘요.
- **Scratch** (Training from scratch (처음부터 학습)): 사전학습 가중치 없이 무작위 값에서 출발해 전부 학습하는 것. 모서리·질감 보는 법부터 배워야 해서 데이터와 시간이 많이 들어요.
- **layer4** (ResNet18 layer4 (마지막 Stage)): ResNet18 Backbone의 마지막 층 묶음. 원래 데이터에 특화된 특징을 담고 있어서 Fine-tuning할 때 가장 먼저 열어요.
- **층별 학습률** (Layer-wise learning rate (param group별 lr)): 층마다 학습률을 다르게 주는 것. 이미 잘 배운 Backbone은 작게(예: 1e-4), 무작위에서 시작하는 새 Head는 크게(예: 1e-3) 움직여요.
- **class_to_idx** (class name → index mapping): 클래스 이름과 번호의 대응표(예: {'circle': 0, 'square': 1}). train·valid·test에서 똑같아야 채점이 맞아요.
- **Class weight** (Class weight (클래스별 Loss 가중치)): 적은 클래스를 틀렸을 때 Loss에 더 큰 벌점을 주는 방법. 보통 역빈도(전체 ÷ 개수)로 정하고 평균 1로 맞춰요.
- **WeightedRandomSampler** (Weighted Random Sampler (가중 무작위 추출)): 적은 클래스의 샘플을 더 자주 뽑아 Batch를 만드는 도구. 샘플별 가중치(1/개수)를 주고 복원 추출(replacement=True)로 같은 사진도 여러 번 뽑아요.
- **Macro F1** (Macro-averaged F1): 클래스별 F1을 단순 평균한 값. 클래스마다 한 표씩이라 적은 클래스 하나만 못 맞혀도 점수가 크게 떨어져요. 불균형 문제의 주 지표.
- **Weighted F1** (Weighted-averaged F1): 클래스별 F1을 support(개수)만큼 가중해 평균한 값. 많은 클래스가 점수를 지배해서 소수 클래스의 실패를 가릴 수 있어요.
- **Support** (Support (표본 수)): 실제로 그 클래스인 샘플 수. 그 클래스의 점수가 몇 장을 근거로 나왔는지 알려 줘요.
- **오류 분석** (Error analysis (오류 분석)): 틀린 이미지를 원본·예측 확률·출처와 함께 직접 보고, 원인 후보(데이터 부족·도메인 차이·라벨·모델)를 세워 다음 실험으로 확인하는 과정.
- **도메인 차이** (Domain shift (도메인 차이)): 사전학습 데이터(Source)와 우리 데이터(Target)의 배경·색·잡음·촬영 환경이 달라서 빌려 온 특징이 잘 안 맞는 것.
- **Grad-CAM** (Gradient-weighted Class Activation Mapping): 특정 클래스 점수에 사진의 어느 부분이 기여했는지 열지도로 보여 주는 방법. 원인 확정이 아니라 '모델이 어디에 반응했나'의 단서예요.
- **Heatmap** (Heatmap (열지도)): 값이 큰 곳은 뜨거운 색, 작은 곳은 차가운 색으로 칠한 지도. Grad-CAM에선 가장 밝은 곳을 1로 맞춘 값이라 밝기 ≠ 점수예요.
- **Hook** (Hook (갈고리 함수)): 모델 코드를 고치지 않고 특정 층을 지날 때의 출력이나 gradient를 몰래 복사해 두는 함수. 다 쓰면 꼭 remove()로 지워요.
- **checkpoint** (Checkpoint (저장된 모델 상태)): 학습 중 어느 순간의 가중치(state_dict)와 설정을 파일로 저장한 것. 나중에 불러와 이어 쓰거나 best 시점으로 되돌릴 때 써요.
- **weights_only** (torch.load(weights_only=True)): torch.load가 텐서 같은 안전한 값만 불러오게 하는 옵션. False면 파일 안의 임의 코드가 실행될 수 있어, 믿을 수 없는 checkpoint는 불러오지 않아요.

---

# Day15 · 객체탐지와 YOLO
_2026-11-06(금) · 객체탐지 · IoU · mAP · YOLO 학습/추론 · 추적_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day15.html

📌 딱 3줄 요약

- 객체탐지 = 사진 속 물체가 무엇인지 + 어디(박스)에 있는지 찾는 것.

- 맞았는지는 IoU(박스 겹침 비율)로 채점하고, 그걸 모아 mAP 점수를 만든다.

- YOLO는 train → val → predict → track 네 명령으로 학습부터 영상 추적까지 한다.

🔧 Day15 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| ultralytics | YOLO 학습·검증·추론·추적을 한 번에 | from ultralytics import YOLO |
| opencv-python | 이미지·영상 읽기/쓰기, 박스 그리기 | import cv2 |
| lap | 추적할 때 프레임 사이 박스 짝짓기 계산 | 직접 import 안 함 (track()이 내부에서 씀) |
| PyYAML | dataset.yaml 읽고 쓰기 | import yaml |
| numpy, pandas | 좌표 배열 계산, 결과를 표(CSV)로 저장 | import numpy as np, import pandas as pd |
| torch, torchvision | YOLO가 내부에서 쓰는 딥러닝 엔진 + 박스 계산 도구(box_convert, box_iou) (Day9부터 설치됨) | import torch, from torchvision.ops import box_iou |

### 🔧 설치 · 확인

```
python -m pip install ultralytics lap opencv-python PyYAML pandas
python -c "import ultralytics, cv2, torch; print(ultralytics.__version__, cv2.__version__, torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"   # True면 GPU 사용 가능
```

⚠️ torch는 다시 깔지 말 것. 이미 맞춰 둔 CUDA 버전이 꼬일 수 있어요. 수업 기준 버전은 requirements.txt(ultralytics 8.4.21, lap 0.5.12)를 따르세요.

💡 pip 이름과 import 이름이 다른 경우가 있어요: opencv-python → cv2, PyYAML → yaml, scikit-learn → sklearn, Pillow → PIL.

## 이거 어디에 써?

Day14까지는 "이 사진은 고양이다"처럼 사진 전체에 이름표 하나를 붙였어요(분류). 객체탐지는 "고양이는 여기, 강아지는 저기"처럼 물체마다 위치까지 찾아요.

🅿️

주차장 CCTV차 박스를 세서 빈자리 안내

⛑️

공사 현장안전모 안 쓴 사람 찾기

🏭

공장 검사부품 위 불량 위치 표시

🚗

자율주행보행자·차를 매 프레임 탐지

| 작업 | 출력 | 한마디로 |
|---|---|---|
| 분류 | 사진 전체의 클래스 1개 | "뭐야?" |
| 탐지 | 물체마다 클래스 + 박스 + 점수 | "뭐가 어디 있어?" |
| 분할 | 픽셀 단위 마스크 | "정확히 어느 픽셀까지야?" |
| 추적 | 탐지 + 프레임 간 같은 ID | "아까 걔가 지금 어디 갔어?" |

🤔 왜 분류로는 안 돼? — 사진 한 장에 물체가 여러 개 있고, 같은 사람이라도 위치·크기가 제각각이에요. 분류는 사진 전체에 답 하나만 내니까 "안전모 안 쓴 사람이 어디 있는지", "차가 몇 대인지"를 말할 수 없어요.

## ① 박스 좌표 — 같은 박스, 세 가지 표현

[그림: 도형 3개가 있는 이미지] 원본 이미지 무엇을 보나: 잡음 낀 회색 배경 위에 초록 원·빨강 사각형·파랑 삼각형이 있어요. 모델이 받는 건 이 픽셀뿐이에요. 🤔 왜 도형? 박스 좌표가 맞는지 눈으로 바로 확인할 수 있게, 경계가 분명한 물체를 썼어요.

[그림: 도형마다 박스가 그려진 이미지] 라벨 txt를 읽어 박스로 그린 모습 무엇을 보나: 각 도형에 클래스 번호 + 이름(0 square, 1 circle, 2 triangle)과 박스가 붙었어요. 박스는 도형에 딱 맞게 감싸요. 🤔 왜 그려 봐? 라벨 txt는 0~1 사이 숫자 네 개라 눈으로는 맞는지 알 수 없어요. 그려서 박스가 물체에서 밀리거나 뒤바뀌지 않았는지 확인하는 게 학습 전 첫 점검이에요.

| 표현 | 숫자 4개 | 어디서 쓰나 |
|---|---|---|
| xyxy | 왼쪽위 (x1,y1), 오른쪽아래 (x2,y2) | 그리기, IoU 계산, boxes.xyxy |
| xywh (중심) | 중심 (cx,cy), 너비 w, 높이 h | boxes.xywh |
| 정규화 | 위 값을 사진 폭·높이로 나눈 0~1 | YOLO 라벨 txt, boxes.xywhn |

[그림: 빨간 사각형 하나에 모서리 두 점, 중심, 너비·높이를 표시하고 xyxy, cxcywh, 정규화 세 가지 숫자를 나란히 적은 그림] 무엇을 보나: 왼쪽 그림의 같은 박스 하나가 오른쪽에서 세 줄의 서로 다른 숫자가 돼요. 보라 점 두 개 → xyxy, 분홍 중심 + 초록 화살표(w, h) → cxcywh, 그걸 160·128로 나누면 → 정규화. 🤔 왜? 박스는 그대로인데 "어떻게 적느냐"만 달라요. 그래서 숫자만 보고는 형식을 알 수 없고, 형식을 착각하면 박스가 엉뚱한 곳에 그려져요.

YOLO 라벨 txt는 물체 하나당 한 줄: 클래스 cx cy w h (전부 0~1)

```
0 0.796875 0.339844 0.193750 0.242188   ← 0번 클래스(square)
1 0.187500 0.250000 0.162500 0.203125   ← 1번 클래스(circle)
2 0.481250 0.640625 0.212500 0.265625   ← 2번 클래스(triangle)
```

변환 공식 (W·H = 사진의 폭·높이)

```
cx = (x1 + x2) / 2 / W        w = (x2 - x1) / W
cy = (y1 + y2) / 2 / H        h = (y2 - y1) / H
```

🤔 왜 0~1로 나눠? — 학습할 때 사진은 imgsz에 맞게 늘었다 줄었다 해요. 픽셀 값으로 적어 두면 크기가 바뀔 때마다 라벨을 다시 계산해야 하지만, 비율(0~1)로 적으면 몇 px이 되든 그대로 써요. 대신 픽셀로 되돌릴 땐 꼭 그 사진의 W·H를 곱해야 해요.

🤔 왜 표현이 세 가지나 있어? — 쓰는 곳마다 편한 형태가 달라서예요. 그리기·겹침 계산은 모서리(xyxy)가 편하고, 모델은 "중심이 어디고 얼마나 큰지(xywh)"로 예측하는 게 자연스럽고, 저장은 크기와 무관한 정규화가 편해요. 그래서 지금 이 숫자가 어떤 형식인지 늘 확인해야 해요.

⚠️ x는 폭 W로, y는 높이 H로 나눠요. W·H를 바꿔 넣으면 박스 위치와 크기가 엉뚱하게 틀어져요.

🔧 박스 그리기 명령어 사전에서 자세히 →

```
cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
cv2.putText(img, "dog 0.87", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
```

🔧 박스 형식 바꾸기 box_convert 명령어 사전에서 자세히 →

```
cxcywh = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
labels = cxcywh / torch.tensor([160, 128, 160, 128])                # x는 W, y는 H로
```

## ② IoU와 평가 지표

IoU = 겹친 넓이 ÷ 두 박스를 합친 넓이. 0이면 안 겹침, 1이면 완전히 같음. 파란 박스를 끌어 보세요(오른쪽 아래 모서리는 크기 조절).

```
합친 넓이 = 박스A 넓이 + 박스B 넓이 - 겹친 넓이   ← 겹친 부분을 두 번 세지 않게
IoU = 겹친 넓이 / 합친 넓이
```

[그림: 정답 박스와 예측 박스가 IoU 0.80, 0.50, 0.20, 0.25로 겹친 네 가지 예시] 무엇을 보나: 초록 점선 = 정답, 파랑 = 예측, 노랑 = 겹친 부분. 왼쪽 세 칸은 크기는 같고 옆으로 4px → 13px → 27px 밀렸을 뿐인데 IoU가 0.80 → 0.50 → 0.20으로 뚝뚝 떨어져요. 맨 오른쪽은 중심이 완전히 같은데도 0.25예요. 🤔 왜? 밀리면 노란 부분(분자)이 줄고, 크기가 커지면 전체 땅(분모)이 커지기 때문이에요. 40px 박스에서 13px만 밀려도 벌써 0.5 경계라서, 작은 물체일수록 몇 픽셀 차이로 TP/FP가 갈려요.

🤔 왜 "중심이 얼마나 가까운가"가 아니라 IoU야? — 위 체험에서 파란 박스를 정답과 중심은 같게 두고 크기만 2배로 키워 보세요. 중심 거리는 0인데 IoU는 뚝 떨어져요. IoU는 위치와 크기가 둘 다 맞아야 높아져요. 또 비율이라 큰 박스든 작은 박스든 같은 잣대(0~1)로 잴 수 있어요.

🤔 왜 합친 넓이에서 겹친 넓이를 빼? — 두 넓이를 그냥 더하면 겹친 부분이 두 번 들어가요. 한 번 빼야 "두 박스가 덮는 전체 땅"이 돼요.

⚠️ IoU는 위치만 봐요. 클래스가 맞는지는 따로 확인해요. 사람 박스와 위치가 딱 맞아도 "개"라고 했으면 틀린 거예요.

🔧 IoU 계산 box_iou 명령어 사전에서 자세히 →

```
iou = box_iou(pred_boxes, gt_boxes)   # [예측 수, 정답 수] 겹침 표 (xyxy 형식)
```

### TP · FP · FN

|  | 뜻 | 예 |
|---|---|---|
| TP 맞춤 | 같은 클래스 정답과 IoU 기준 이상으로 1:1 짝지어진 예측 | 사람을 사람으로, 위치도 맞게 |
| FP 헛탐지 | 짝이 없는 예측 (없는 걸 찾음, 클래스 틀림, 중복 박스) | 나뭇가지를 사람으로 |
| FN 놓침 | 아무 예측과도 짝지어지지 않은 정답 | 작은 사람을 못 찾음 |

⚠️ 한 물체에 박스 3개가 겹쳐 나오면 TP는 1개, 나머지 2개는 FP.

[그림: 도형 세 개 사진 위에 정답 점선과 예측 박스를 그리고 TP, FP, FN을 색으로 구분한 두 장면] 무엇을 보나: 같은 사진, 같은 예측(연습용으로 손으로 정한 값)을 두 번 채점했어요. 왼쪽은 사각형에 박스가 2개(0.95, 0.85) → 하나만 TP, 0.85는 "중복" FP. 위쪽 빈 곳의 triangle 0.40은 "엉뚱한 곳" FP, 진짜 삼각형은 아무도 못 짚어서 FN(빨간 점선). 오른쪽은 중복을 정리한 뒤라 FP가 1개로 줄어 Precision이 0.50 → 0.67로 올라요. 🤔 왜? 정답 하나엔 예측 하나만 짝지어 주니까 중복은 무조건 FP가 되고, Recall(놓친 삼각형)은 중복 정리와 상관없이 그대로예요.

🤔 왜 하나만 인정해? — 중복 박스를 다 맞았다고 쳐 주면, 한 물체에 박스를 잔뜩 그리는 모델이 점수를 거저 먹어요. 그래서 점수 높은 예측부터 정답 하나에 하나씩만 짝지어요.

### Precision · Recall · mAP

| 지표 | 공식 | 질문 |
|---|---|---|
| Precision | TP / (TP+FP) | 찾은 것 중 몇 %가 진짜? |
| Recall | TP / (TP+FN) | 진짜 중 몇 %를 찾았나? |
| AP | 한 클래스의 Precision–Recall 곡선 아래 넓이 | 그 클래스를 전반적으로 잘 찾나? |
| mAP50 | IoU 기준 0.5로 잰 AP를 클래스별로 평균 | 대충이라도 찾나? |
| mAP50-95 | IoU 기준 0.50, 0.55 … 0.95 (10개)로 잰 평균 | 박스 위치까지 정확한가? |

예: TP 8, FP 2, FN 4 → Precision = 8/10 = 0.80, Recall = 8/12 = 0.67

[그림: 예측 12개를 점수순으로 줄 세워 Precision-Recall 점을 찍고 계단 곡선 아래 넓이를 AP로 칠한 그래프] 무엇을 보나: 오른쪽 표처럼 예측을 점수 높은 순으로 세운 뒤, 위에서부터 하나씩 "여기까지 인정"하며 P와 R을 다시 계산해 점을 찍어요(예시 숫자). 점들을 위쪽으로 감싼 보라 계단 아래 칠해진 넓이 = AP(0.62). 🤔 왜? conf 기준을 낮출수록 예측이 하나씩 추가돼 Recall은 오른쪽으로만 가고, FP가 섞일 때마다 Precision은 아래로 꺾여요. 기준 하나만 고르지 않고 이 전체 모양을 넓이 하나로 요약하니까 "특정 conf에 좌우되지 않는 점수"가 돼요. 오른쪽 끝이 0.75에서 끊긴 건 정답 8개 중 6개만 찾았기 때문이에요.

🤔 왜 Precision과 Recall을 둘 다 봐? — 하나만 보면 속기 쉬워요. 확실한 것 1개만 내면 Precision은 1.0이지만 거의 다 놓치고, 박스를 마구 그리면 Recall은 높지만 헛탐지투성이예요. 두 개가 같이 좋아야 좋은 모델이에요.

🤔 왜 그냥 정확도(accuracy)가 아니라 mAP야? — 탐지에는 "배경을 배경이라고 맞힌 개수"를 셀 수가 없어요(배경 박스는 무한히 그릴 수 있으니까). 그래서 점수 기준(conf)을 높였다 낮췄다 하면서 Precision–Recall이 어떻게 변하는지 곡선으로 그리고, 그 넓이(AP)를 클래스별로 평균 내요. 특정 conf 하나에 좌우되지 않는 점수예요.

🤔 왜 mAP50-95가 항상 더 낮아? — 0.95처럼 엄격한 기준에서는 박스가 거의 완벽하게 겹쳐야 TP예요. 같은 예측이라도 기준이 빡빡해질수록 TP가 FP로 바뀌니 평균이 내려가요. 그래서 mAP50은 높은데 mAP50-95가 낮으면 "찾긴 하는데 박스가 엉성하다"는 뜻이에요.

[그림: 예측 10개의 IoU 막대와, 평가 기준 0.50부터 0.95까지 올릴 때 TP 수가 10개에서 0개로 줄어드는 막대그래프] 무엇을 보나: 왼쪽은 클래스는 다 맞은 예측 10개가 각자 정답과 얼마나 겹쳤는지(예시 숫자). 오른쪽은 그 똑같은 10개를 기준만 0.50, 0.55 … 0.95로 바꿔 채점한 결과예요. 0.50에선 TP 10개(초록), 0.95에선 0개(빨강). 분홍 점선이 10개 기준의 평균이에요. 🤔 왜? 왼쪽 막대가 세로 점선 기준을 넘지 못하면 TP가 FP로 바뀌어요. mAP50은 첫 막대 하나만, mAP50-95는 10개 막대 평균이니까 박스가 조금만 엉성해도 크게 깎여요(실제 mAP는 각 기준에서 AP를 계산하지만, 줄어드는 원리는 같아요).

### 지표로 문제 진단하기

| 이렇게 나오면 | 의심할 것 |
|---|---|
| P 높고 R 낮음 | 놓치는 게 많음 → 작은 물체, 라벨 누락, conf가 너무 높음 |
| R 높고 P 낮음 | 헛탐지가 많음 → 배경만 있는 사진 부족, conf가 너무 낮음 |
| mAP50 ≫ mAP50-95 | 찾긴 하는데 박스 위치가 부정확 |
| train은 좋아지는데 val은 정체 | 과적합 |

🔧 검증 model.val() 명령어 사전에서 자세히 →

```
metrics = model.val(data="dataset.yaml", split="val", imgsz=640, conf=0.001, iou=0.7)
print(metrics.box.map50)   # mAP50
```

## ③ YOLO 모델과 학습

YOLO(You Only Look Once)는 사진을 한 번 통과시켜 모든 위치·크기의 물체 후보를 동시에 예측해요. 그래서 빠르고, 영상에 많이 써요.

구체적으로는 사진을 격자(칸)로 나누고 칸마다 박스 숫자 4개 + 클래스별 점수를 내요. 160×128 사진이면 이렇게 돼요:

| 격자 | 한 칸 크기 | 후보 수 | 맡는 물체 |
|---|---|---|---|
| 20 × 16 | 8px | 320개 | 작은 물체 |
| 10 × 8 | 16px | 80개 | 중간 물체 |
| 5 × 4 | 32px | 20개 | 큰 물체 |
| 합계 | 420개 | 물체가 3개여도 후보는 420개! |  |

[그림: 같은 160x128 사진에 20x16, 10x8, 5x4 격자를 겹치고 칸마다 점수를 보라색 진하기로 칠한 세 장면] 무엇을 보나: 같은 사진을 세 가지 촘촘함으로 나눈 모습이에요. 흰 선으로 나뉜 칸 하나 = 후보 하나(320 + 80 + 20 = 420). × 는 물체 중심, 노란 칸은 그 중심이 떨어진 칸이에요. 보라색 진하기는 작은 데모 모델이 실제로 낸 점수예요. 🤔 왜? 8px 칸은 사각형 하나에 칸이 20개 넘게 들어가 위치를 아주 잘게 짚을 수 있고(작은 물체용), 32px 칸은 칸 하나가 물체 전체를 덮어서 큰 모양을 한 번에 봐요(큰 물체용). 그리고 물체가 없는 칸도 전부 후보를 내기 때문에 나중에 걸러야 해요.

🤔 왜 격자를 세 가지 크기로 써? — 촘촘한 격자는 위치 정보가 자세해서 작은 물체에 유리하고, 성긴 격자는 넓은 범위를 한 번에 봐서 큰 물체의 전체 모양을 잘 잡아요. Neck이 서로 다른 크기의 특징을 섞어 주는 이유가 이거예요.

사진→ Backbone 특징 뽑기→ Neck 큰·작은 크기 정보 합치기→ Head 클래스·박스·점수 예측→ 후보 정리 conf·NMS

[그림: ] 무엇을 보나: 왼쪽에서 오른쪽으로 사진이 흘러가요. Backbone은 사진을 20×16 → 10×8 → 5×4로 점점 작게 줄이면서(네모가 작아지고 색이 진해짐) 더 "뜻"에 가까운 특징을 뽑아요. Neck은 세 줄을 위아래로 오가며 섞고, Head는 세 크기에서 각각 후보를 내 합계 420개가 돼요. 🤔 왜? 큰 지도(20×16)는 위치는 자세하지만 "이게 뭔지"는 잘 몰라요. 작은 지도(5×4)는 반대예요. Neck이 둘을 섞어 줘야 "작은 물체도 뭔지 알고, 큰 물체도 어디인지 아는" 예측이 돼요. (그림은 원리를 보여 주는 단순화예요.)

|  | yolo26n.yaml | yolo26n.pt |
|---|---|---|
| 들어 있는 것 | 모델 설계도만 | 설계도 + 학습된 가중치 |
| 비유 | 빈 뇌 | 이미 공부한 뇌 |
| 언제 | 처음부터 학습 | 보통 이걸로 시작 (전이학습) |

🤔 왜 보통 .pt로 시작해? — .yaml(빈 뇌)은 선·모서리 같은 기초부터 배워야 해서 데이터도 많고 시간도 오래 걸려요. .pt는 이미 수많은 사진으로 기초를 익힌 상태라, 내 데이터로는 "내 물체" 구분만 더 배우면 돼요. Day14 전이학습과 같은 원리예요.

학습 데이터 폴더 구조와 안내서(dataset.yaml):

```
dataset/
├─ images/train, images/val    ← 사진
└─ labels/train, labels/val    ← 사진과 같은 이름의 txt

# dataset.yaml
path: C:/ai_course/dataset
train: images/train
val: images/val
names:
  0: helmet
  1: no_helmet
  2: person
```

[그림: mosaic 증강 학습 이미지] 학습 중 모델이 실제로 보는 사진 — 4장을 이어 붙이고(mosaic) 크기·위치를 흔드는 증강. 박스도 같이 변환돼요.

🤔 왜 사진을 일부러 섞고 흔들어? — 같은 사진만 계속 보면 "그 사진"을 외워 버려요(과적합). 위치·크기·밝기를 조금씩 바꿔 보여주면 실제 현장의 다양한 모습에 강해져요. 단, 사진을 움직이면 박스도 똑같이 움직여야 라벨이 맞아요.

⚠️ 같은 원본에서 나온 사진(증강본, 연속 프레임)이 train과 val에 나눠 들어가면 안 돼요. 거의 같은 사진으로 채점하는 셈이라 점수가 실제보다 좋게 나와요(누수).

🔧 모델 만들기 YOLO() · 학습 model.train() 명령어 사전에서 자세히 →

```
model = YOLO("yolo26n.pt")       # 학습된 뇌 불러오기 (.yaml이면 빈 뇌)
model.train(data="dataset.yaml", epochs=30, imgsz=640, batch=16, device=0, workers=0)
```

[그림: epoch별 train/val box loss 곡선과 mAP50, mAP50-95 곡선] 무엇을 보나: results.csv를 직접 그린 예예요(합성 도형으로 100 epoch 학습한 작은 데모 모델 — 수업 결과가 아니에요). 왼쪽은 loss가 들쭉날쭉하면서도 전체적으로 내려가고, 오른쪽은 한동안 0이다가 50 epoch쯤부터 점수가 생겨요. 초록(mAP50)은 늘 분홍(mAP50-95)보다 위에 있고, 그 간격이 박스가 엉성한 정도예요. 분홍 점은 mAP50-95가 가장 높았던 epoch예요. 🤔 왜? 마지막 epoch가 꼭 최고가 아니기 때문에 이렇게 val 점수를 매 epoch 기록해 두고 가장 좋았던 순간을 best.pt로 저장해요(ultralytics 8.4는 val mAP50-95가 가장 높은 epoch를 고르니, 분홍 점이 바로 그 순간이에요).

## ④ confidence · NMS · 추론

모델은 처음에 후보 박스를 아주 많이 뱉어요. 이걸 두 단계로 걸러요.

후보 수백 개→ conf 필터 점수 낮은 것 버림→ NMS 겹치는 중복 박스 정리→ 최종 결과

실제 예 (사진 1장, 정답 3개): 후보 420개 → conf 0.25로 거르면 11개 → NMS(iou 0.7)로 중복 지우면 4개 → 정답과 채점하니 TP 3 · FP 1 (남은 중복 박스 하나가 FP).

[그림: 420개 후보 점, conf 0.25 통과 박스 11개, NMS 후 4개를 TP 3개 FP 1개로 채점한 세 단계 그림] 무엇을 보나: 위 "실제 예"의 숫자 흐름(420 → 11 → 4, TP 3 · FP 1)을 그대로 그려 본 모의 그림이에요(점수는 손으로 정한 값). ① 보라 점이 사진 전체에 깔려 있고 물체 위에서만 커요. ② 0.25를 넘은 11개는 물체마다 3~4개씩 겹쳐 있어요. ③ NMS 후 4개 — 그런데 원 위의 0.37 박스는 1등 박스와 덜 겹쳐서(IoU < 0.7) 살아남았고, 정답과도 덜 겹쳐(IoU < 0.5) FP가 됐어요. 🤔 왜? NMS는 "많이 겹치는 것"만 지워요. 기준(0.7)보다 덜 겹친 중복은 다른 물체일 수도 있으니 남겨 두는데, 그게 진짜 중복이었다면 채점에서 FP로 잡혀요.

[그림: 작은 데모 모델의 실제 출력 420개 후보, conf 0.02 통과 9개, NMS 후 5개] 무엇을 보나: 이번엔 진짜 모델 출력이에요(합성 도형으로 짧게 학습한 작은 데모 모델). ①의 점 420개는 정말로 모델이 낸 후보 수이고, 점수가 낮은 모델이라 conf를 0.02까지 낮춰서 ② 9개 → ③ NMS 후 5개가 됐어요. 원은 아예 못 찾았고(FN), 사각형·삼각형 주변엔 덜 겹친 박스가 남았어요. 🤔 왜? 모델이 약해도 후보 수(420)는 똑같아요 — 후보 수는 격자 칸 수로 정해지고, 모델 실력은 "그중 누가 높은 점수를 받느냐"에서 드러나요.

🤔 왜 후보가 이렇게 많이 나와? — 칸마다 무조건 예측을 하니까요. 물체 근처의 여러 칸이 "여기 원 있어!"라고 동시에 외쳐서 한 물체에 박스가 여러 개 겹쳐 나와요. 그래서 걸러내는 단계가 꼭 필요해요.

NMS 순서: 점수 제일 높은 박스를 남김 → 그 박스와 많이 겹치는(IoU 큰) 같은 클래스 박스를 지움 → 남은 것 중 다음으로 높은 박스로 반복.

[그림: 후보 4개를 점수순으로 세우고, 1등을 남기고 겹치는 같은 클래스 후보를 지우고, 다음 1등을 남기는 NMS 단계 그림] 무엇을 보나: 0) 후보 4개(연습용 값)를 점수순으로 줄 세워요. 1) 1등 square 0.95를 남기고, 같은 square인 0.85와 IoU를 재면 0.90 ≥ 0.7 → 빨간 X로 삭제. 2) 남은 것 중 1등 circle 0.80을 남겨요 — 같은 circle 후보가 없어 지울 게 없어요. 3) triangle 0.40도 남기고 끝. 4개 → 3개. 🤔 왜? 1등과 거의 같은 자리를 가리키는 후보는 "같은 물체를 또 외친 것"이라 버려요. 하지만 NMS는 정답을 모르니까, 엉뚱한 곳의 0.40처럼 틀린 박스도 겹치는 게 없으면 그대로 남아요 — 그건 conf나 모델 실력의 몫이에요.

🤔 왜 "같은 클래스"끼리만 지워? — 사람 위에 안전모 박스가 크게 겹치는 건 정상이에요. 클래스가 다르면 다른 물체니까 지우면 안 돼요.

| 이름 | 무엇끼리 비교? | 어디서? |
|---|---|---|
| confidence | 후보 하나의 점수 | predict의 conf= |
| NMS IoU | 예측 ↔ 예측 | predict의 iou= |
| 평가 IoU | 예측 ↔ 정답 | mAP50의 50 |

⚠️ "0.5"가 세 군데 나와서 헷갈림 주의

|  | 낮추면 | 높이면 |
|---|---|---|
| conf | 후보 많아짐 → Recall↑, 헛탐지↑ | 확실한 것만 → Precision↑, 놓침↑ |
| NMS iou | 조금만 겹쳐도 지움 → 중복↓, 붙어 있는 물체 놓칠 수 있음 | 많이 겹쳐도 남김 → 붙은 물체 분리↑, 중복 박스 남을 수 있음 |

🤔 왜 conf를 바꾸면 P와 R이 반대로 움직여? — 기준을 낮추면 애매한 후보까지 통과해서 진짜도 더 잡히지만(Recall↑) 가짜도 같이 들어와요(Precision↓). 기준을 높이면 그 반대예요. 그래서 "놓치면 큰일"인 일(안전 감시)은 conf를 낮게, "헛알림이 피곤한" 일은 높게 잡아요.

[그림: 같은 사진에 conf 0.05, 0.25, 0.5를 적용해 박스가 3개, 1개, 0개로 줄어드는 비교] 무엇을 보나: 같은 사진·같은 모델(작은 데모 모델)에서 conf만 0.05 → 0.25 → 0.5로 올렸어요. 박스가 3개 → 1개 → 0개로 줄고, 아래 줄의 R(Recall)은 2/3 → 1/3 → 0/3으로 떨어지지만 FP(주황)는 사라져 P는 2/3 → 1/1로 올라요. 🤔 왜? conf는 "이 점수 미만은 안 보여 줌"이라는 커트라인이라, 올릴수록 애매한 후보(맞는 것도, 틀린 것도)가 함께 잘려 나가요. 모델은 그대로이고 어디서 자르느냐만 바뀐 거예요.

🔧 추론 model.predict() 명령어 사전에서 자세히 →

```
results = model.predict(source="samples/", conf=0.25, iou=0.7, imgsz=640, save=True)
```

🔧 결과 꺼내기 result.boxes 명령어 사전에서 자세히 →

```
result.boxes.xyxy.cpu().numpy()          # [N, 4] 픽셀 좌표 (.conf · .cls도 같은 방식)
cv2.imwrite("out.jpg", result.plot())    # 박스 그린 그림
```

## ⑤ 추적 (Tracking)

탐지는 프레임마다 따로 "사람 2명"이라고만 말해요. 추적은 프레임 사이를 연결해서 "7번 사람이 오른쪽으로 이동 중"처럼 같은 물체에 같은 Track ID를 붙여요. 이게 있어야 사람 수 세기, 이동 경로, 머문 시간 같은 걸 할 수 있어요.

|  | Detect | Track |
|---|---|---|
| 출력 | box · class · conf | + track_id |
| 이전 프레임 기억 | ❌ | ✅ |
| 대표 실패 | 놓침, 헛탐지 | ID switch (가려졌다 나오면 번호가 바뀜) |

[그림: 영상 0, 15, 29번 프레임에 ID 1, 2, 3 박스와 지나온 경로 선을 그리고, 마지막 칸에 30프레임 경로를 겹쳐 그린 그림] 무엇을 보나: 짧은 도형 영상의 프레임 0 · 15 · 29예요(박스는 영상의 정답 좌표로 그렸어요). 사각형은 계속 ID 1·보라, 원은 ID 2·주황, 삼각형은 ID 3·청록을 유지하고, 박스 안의 선이 지금까지 지나온 길이에요. 마지막 칸은 30프레임 경로를 한 장에 겹친 것 — 사각형은 오른쪽, 삼각형은 왼쪽으로 움직였어요. 🤔 왜? 탐지만 하면 매 프레임 "square 1개"라는 말만 남지만, ID로 이어 붙여야 이런 경로가 생기고, 그래야 "몇 명이 지나갔나", "얼마나 머물렀나"를 셀 수 있어요.

💡 Class ID(종류: 0=사람)와 Track ID(개체: 7번 사람)는 다른 것.

🤔 왜 ID가 바뀌기도 해(ID switch)? — 추적기는 "이전 프레임 박스와 지금 박스가 많이 겹치고 움직임이 자연스러우면 같은 애"라고 추측해요. 그런데 물체가 가려졌다 나오거나, 비슷한 물체끼리 엇갈리거나, 프레임을 건너뛰어 많이 움직이면 그 추측이 틀려서 새 번호를 붙여요.

[그림: ] 무엇을 보나: 보라(ID 7)와 주황(ID 8)이 서로를 향해 걸어와 엇갈려요. 가운데 프레임에서 7이 8 뒤에 가려져(점선) 탐지가 안 되고, 오른쪽에서 다시 나타났을 땐 ID 9라는 새 번호를 받았어요. 🤔 왜? 추적기는 "직전 박스와 많이 겹치는 박스 = 같은 애"로 이어 붙이는데, 가려진 프레임엔 이을 박스가 없고, 다시 나타난 위치는 예상보다 멀어서 같은 애라고 확신하지 못하거든요. 그래서 2명이 지나갔는데 3명으로 세는 일이 생겨요(그림은 원리 설명용).

🔧 추적 model.track() 명령어 사전에서 자세히 →

```
model.track(source="traffic.mp4", persist=True, tracker="botsort.yaml", stream=True)
result.boxes.id.int().tolist()   # 프레임별 Track ID (없으면 id가 None)
```

[그림: 사진 1장 처리 시간을 읽기, 전처리, 모델 추론, 후처리로 나눈 가로 막대와 추론만 본 막대 비교] 무엇을 보나: 작은 데모 모델로 CPU에서 160px 사진을 처리할 때 걸린 시간을 쪼갠 막대예요(result.speed의 preprocess · inference · postprocess + 직접 잰 imread, 여러 장의 중앙값). 위 막대가 실제 전체, 아래 빗금 막대는 "모델 추론"만 잘라 본 것 — 같은 실행인데 FPS 숫자가 달라져요. 🤔 왜? 한 장이 결과가 되려면 읽기 → 전처리 → 추론 → 후처리를 모두 거쳐야 하니까 시간은 더해져요. 여기선 추론이 대부분이라 차이가 작지만, 큰 사진·영상 디코딩·추적·저장이 붙으면 추론 밖의 시간이 훨씬 커져요. 그래서 FPS를 말할 땐 "어디부터 어디까지 쟀는지"를 같이 적어요(숫자는 이 PC 기준이라 장비마다 달라요).

🔧 같은 걸 터미널(CLI)로 yolo detect ... 명령어 사전에서 자세히 →

```
yolo detect train   model=yolo26n.pt data=dataset.yaml epochs=30 imgsz=640
yolo detect predict model=best.pt source=samples/ conf=0.25
```

## 원본 자료

- 강의: Day15_20261106.pdf · 요약: Day15_oneday요약.pdf

- 공식 문서: Ultralytics Modes (train · val · predict · track)

## 🔧 Day15 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day15)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install ultralytics lap opencv-python |
| ① 박스 좌표 | cv2.rectangle(), cv2.putText() |
| ② IoU · 평가 지표 | model.val() → metrics.box.map50 |
| ③ YOLO 모델 · 학습 | YOLO(), model.train() |
| ④ confidence · NMS · 추론 | model.predict(conf=, iou=), result.boxes |
| ⑤ 추적 | model.track(persist=, tracker=) |

### 박스 그리기

```
img = cv2.imread("scene.jpg")
cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
cv2.putText(img, "dog 0.87", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
cv2.imwrite("detected.jpg", img)
```

▶ 실제로 돌려 보면 data/scene.png (128, 160, 3) · boxes_gt.csv 첫 박스 (12, 18, 52, 58)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| img[18, 12] (그리기 전) | [40, 50, 225] | 원래 그 자리 색 |
| cv2.rectangle(img, (12, 18), (52, 58), color=(255, 0, 0), thickness=2) | img[18, 12] → [255, 0, 0] | B=255 → 파랑 선이 그려짐 (빨강 아님!) |
| ret is img | True | 새 그림을 만드는 게 아니라 원본 배열에 바로 그려요 |
| img[38, 32] (박스 한가운데) | [40, 50, 225] | 선만 그리니 안쪽 색은 그대로 |
| cv2.putText(img, "square", (x1, y1 - 8), ...).shape | (128, 160, 3) | 글자도 같은 배열에 덧그림 |
| cv2.imwrite("outputs/detected.png", img) | True | 저장 성공 |

| 인자 | 뜻 |
|---|---|
| (x1, y1), (x2, y2) | xyxy 좌표 (정수 픽셀) |
| color | BGR 순서! (255,0,0)은 빨강이 아니라 파랑 |
| thickness | 선 두께 (생략하면 1) |

⚠️ OpenCV 색은 BGR, matplotlib은 RGB. 섞어 쓰면 색이 뒤바뀐다.

🤔 왜 BGR? — OpenCV가 처음 만들어질 때 카메라·윈도우 쪽에서 BGR 순서가 흔했고, 호환성 때문에 지금까지 그대로예요. 그래서 OpenCV 안에서는 BGR, 다른 라이브러리로 넘길 땐 RGB로 바꿔 줘요.

### 박스 형식 바꾸기 box_convert

```
import torch
from torchvision.ops import box_convert

boxes = torch.tensor([[12, 18, 52, 58]], dtype=torch.float32)       # x1, y1, x2, y2
cxcywh = box_convert(boxes, in_fmt="xyxy", out_fmt="cxcywh")
labels = cxcywh / torch.tensor([160, 128, 160, 128])                # x는 W, y는 H로
```

▶ 실제로 돌려 보면 scene.png(가로 160 · 세로 128)의 square 정답 박스 [12, 18, 52, 58]

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| box_convert(boxes, "xyxy", "cxcywh") | [[32, 38, 40, 40]] | 중심 (32, 38) · 폭 40 · 높이 40 |
| box_convert(boxes, "xyxy", "xywh") | [[12, 18, 40, 40]] | 왼쪽 위 (12, 18) + 크기 |
| cxcywh / [160, 128, 160, 128] | [[0.2, 0.2969, 0.25, 0.3125]] | YOLO 라벨 한 줄의 숫자 4개 (0~1) |
| box_convert(cxcywh, "cxcywh", "xyxy") | [[12, 18, 52, 58]] | 되돌리면 원래 좌표 |

| 인자 | 뜻 |
|---|---|
| in_fmt / out_fmt | "xyxy", "xywh"(왼쪽위+크기), "cxcywh"(중심+크기) 중에서 |

### IoU 계산 box_iou

```
from torchvision.ops import box_iou
iou = box_iou(pred_boxes, gt_boxes)   # [예측 수, 정답 수] 겹침 표 (xyxy 형식)
matched = iou >= 0.5                   # 같은 클래스 + 0.5 이상 → TP 후보
```

▶ 실제로 돌려 보면 boxes_fixture.csv 예측 4개 × boxes_gt.csv 정답 3개 (손으로 정한 연습값)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| box_iou(pred_boxes, gt_boxes).shape | (4, 3) | [예측 수, 정답 수] |
| box_iou(pred_boxes, gt_boxes) | [[0.902, 0, 0], [0.816, 0, 0], [0, 1, 0], [0, 0, 0]] | 행 = 예측, 열 = 정답 |
| iou >= 0.5 | [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]] | 예측 0·1이 둘 다 정답 0과 겹침 → 하나만 TP, 나머지는 중복 FP |
| 예측 3 (90,6,110,22) 줄 | [0, 0, 0] | 어떤 정답과도 안 겹침 → FP, 정답 2는 못 찾음 → FN |

### 검증 model.val()

```
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="dataset.yaml", split="val", imgsz=640, conf=0.001, iou=0.7)
print(metrics.box.map50)   # mAP50
print(metrics.box.map)     # mAP50-95
```

▶ 실제로 돌려 보면 데모 모델 best.pt(train 12장으로 100 epoch 학습한 작은 모델) · dataset.yaml 의 val 6장(정답 18개) · imgsz=160

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| metrics.box.map50 | 0.573 | IoU 0.5 기준 평균 AP — 작은 데모라 높지 않아요 |
| metrics.box.map | 0.274 | IoU 0.5~0.95 평균 → 박스를 꼼꼼히 맞춰야 해서 더 낮음 |
| metrics.box.mp, metrics.box.mr | (0.909, 0.444) | 낸 박스는 대체로 맞지만(P↑) 놓치는 게 많음(R↓) |
| metrics.box.maps | [0.382, 0, 0.441] | 클래스별 mAP50-95 (square, circle, triangle) — circle은 하나도 못 맞힘 |
| conf=0.5로 val 했을 때 metrics.box.map | 0.000 | 이 모델은 0.5를 넘는 후보가 거의 없어 PR 곡선이 통째로 잘림 → 검증은 conf=0.001 |

| 인자 / 결과 | 뜻 |
|---|---|
| data | 데이터 안내서(yaml) 경로 |
| split | 어느 데이터로 채점할지 ("val" / "test") |
| conf=0.001 | 검증 땐 점수 낮은 후보까지 거의 다 남김 → 왜? PR 곡선을 끝까지(Recall이 높은 쪽까지) 그려야 AP를 제대로 잴 수 있어서 |
| iou | 여기서도 NMS 기준이에요. mAP50의 "50"(평가 IoU)과는 다른 값 |
| metrics.box.map50 / .map | mAP50 / mAP50-95 |
| metrics.box.mp / .mr | 평균 Precision / 평균 Recall |

🤔 왜 test가 아니라 val로 골라? — conf나 checkpoint를 test 점수 보면서 고르면, 그 test에 맞춰진 선택이 돼서 처음 보는 데이터 성능을 부풀려 말하게 돼요. 그래서 고르는 건 val, test는 마지막에 딱 한 번 확인용으로 남겨요.

### 모델 만들기 YOLO() · 학습 model.train()

```
from ultralytics import YOLO

if __name__ == "__main__":          # Windows: 보조 프로세스가 이 파일을 다시 실행해도 학습이 또 시작되지 않게
    model = YOLO("yolo26n.pt")       # 학습된 뇌 불러오기 (.yaml이면 빈 뇌)
    model.train(data="dataset.yaml", epochs=30, imgsz=640,
                batch=16, device=0, workers=0)
```

▶ 실제로 돌려 보면 데모 모델 best.pt vs 빈 뇌 yolo26n.yaml · val 사진 basic_val_000.png(정답 3개)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| YOLO("best.pt").task, .names | 'detect', {0: 'square', 1: 'circle', 2: 'triangle'} | 탐지 모델 · 클래스 이름이 들어 있음 |
| YOLO("best.pt") / YOLO("yolo26n.yaml") 파라미터 수 | 2,504,970 / 2,572,280 | 같은 구조 — .yaml은 기본 80클래스라 출력 층만 조금 큼. 차이는 결국 '가중치 값' |
| YOLO("best.pt").predict(사진) 박스 수 | 1 | 학습된 뇌: 3개 중 1개만 — 12장으로 배운 작은 데모라 아직 약해요 |
| YOLO("yolo26n.yaml").predict(사진) 박스 수 | 0 | 무작위 가중치라 점수가 conf 0.25를 못 넘음 |
| YOLO("yolo26n.yaml").train(data=..., epochs=1, imgsz=160, device="cpu", workers=0, plots=False) 후 생긴 파일 | args.yaml · results.csv · weights/best.pt · weights/last.pt | 1 epoch만 돌려도 weights/best.pt · last.pt · results.csv가 생김 (plots=False라 그림은 없음) |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| data (필수) | dataset.yaml 경로 | 모델이 사진·라벨·클래스 이름을 여기서 찾아요 |
| epochs | 데이터 전체를 몇 바퀴 볼지 | 적으면 덜 배움. 많으면 train 사진을 외워서 val 점수가 멈추거나 떨어짐(과적합) |
| imgsz | 사진을 몇 px로 맞춰 넣을지 | 크면 작은 물체가 픽셀 몇 개로 뭉개지지 않아 잘 보임. 대신 계산량이 크기²에 비례해 느려지고 메모리↑ |
| batch | 한 번에 넣는 사진 수 | 크면 한 번에 더 많이 봐서 학습이 안정적이지만 GPU 메모리를 더 씀 (부족하면 줄이기) |
| device | 0 = GPU, "cpu" = CPU | GPU는 행렬 계산을 동시에 많이 해서 훨씬 빠름 |
| workers | 데이터 읽는 보조 프로세스 수 | 많으면 사진 읽기가 빨라짐. Windows에선 보조 프로세스가 파일을 다시 실행하며 오류가 날 수 있어 0 추천 |
| seed | 무작위 순서 고정 | 같은 seed면 다시 돌려도 결과가 비슷 → 비교 실험할 때 필요 |

학습이 끝나면 생기는 것 (runs/detect/train/)

| weights/best.pt | 검증 점수가 가장 좋았던 epoch의 모델 → 보통 이걸 씀 (마지막 epoch가 꼭 제일 좋은 건 아니라서) |
|---|---|
| weights/last.pt | 마지막 epoch의 모델 → 학습 이어하기(resume)용 |
| results.csv / results.png | epoch별 loss·mAP 기록과 그래프 |
| confusion_matrix.png | 어떤 클래스를 어떤 클래스로 헷갈렸나 |

### 추론 model.predict()

```
model = YOLO("best.pt")
results = model.predict(source="samples/", conf=0.25, iou=0.7, imgsz=640, save=True)
```

▶ 실제로 돌려 보면 데모 모델 best.pt · source=data/images/val (6장) · imgsz=160

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| len(results) | 6 | 사진 1장 = result 1개 |
| 사진별 정답 박스 수 (라벨 파일) | [3, 3, 3, 3, 3, 3] | 비교 기준 |
| conf=0.25, iou=0.7 → 사진별 박스 수 | [1, 1, 1, 1, 1, 1] | 정답보다 적음 — 데모 모델 점수가 대부분 0.25 아래라 버려짐 |
| conf=0.01, iou=0.7 → 사진별 박스 수 | [8, 6, 5, 6, 7, 10] | 기준을 낮추면 점수 낮은 후보까지 남아 박스가 늘어남(정답보다 많아지기도) |
| conf=0.01, iou=0.1 → 사진별 박스 수 | [2, 2, 2, 2, 2, 2] | NMS 기준을 빡빡하게 하면 조금만 겹쳐도 지워져 다시 줄어듦 |
| save=True 로 저장된 파일 | 6개 (basic_val_000.jpg …) | 박스 그린 사진이 runs/.../predict 폴더에 |

| 인자 | 뜻 |
|---|---|
| source (필수) | 사진 경로, 폴더, 영상, URL, 웹캠 번호(0), OpenCV 배열 |
| conf | 이 점수 미만 후보는 버림 (기본 0.25) |
| iou | NMS 겹침 기준 (기본 0.7) |
| save=True | 박스 그린 이미지를 저장 |
| stream=True | 긴 영상은 이걸로 → 한 프레임씩 처리해 메모리 절약. 왜? False면 모든 프레임 결과를 리스트에 다 쌓아 두고 한꺼번에 돌려줘서 메모리가 터질 수 있어요 |

### 결과 꺼내기 result.boxes

```
for result in results:                       # 사진 1장 = result 1개
    for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(),
                               result.boxes.conf.cpu().numpy(),
                               result.boxes.cls.cpu().numpy().astype(int)):
        print(result.names[cls], round(float(score), 2), box)
    cv2.imwrite("out.jpg", result.plot())    # 박스 그린 그림
```

▶ 실제로 돌려 보면 model.predict(source=val, conf=0.25) 결과의 첫 사진 basic_val_000.png — 박스 1개

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| result.boxes.xyxy.cpu().numpy() | [[65.2, 81.8, 98.5, 110.6]] | 픽셀 좌표 [N, 4] |
| result.boxes.conf.cpu().numpy() | [0.531] | 점수 |
| result.boxes.cls.cpu().numpy().astype(int) | [2] | 클래스 번호 |
| result.names | {0: 'square', 1: 'circle', 2: 'triangle'} | 번호 → 이름 |
| result.boxes.xywhn.cpu().numpy() | [[0.512, 0.752, 0.208, 0.225]] | 정규화 중심 좌표 |
| 정답 라벨 파일에서 같은 클래스(2) 줄의 cx cy w h | [0.522, 0.746, 0.231, 0.289] | 위 xywhn과 비슷 → 라벨과 같은 형식 (박스는 조금 작게 잡음) |
| result.plot().shape, dtype | (128, 160, 3), uint8 | 박스 그린 BGR 이미지 → cv2.imwrite로 저장 |
| 하얀 빈 그림의 len(result.boxes) | 0 | 탐지 0개 — 에러가 아니라 정상 |

| 속성 | 내용 |
|---|---|
| boxes.xyxy | 픽셀 좌표 [N, 4] |
| boxes.xywhn | 정규화 중심 좌표 (라벨 형식) |
| boxes.conf / boxes.cls | 점수 / 클래스 번호 |
| result.names | 번호 → 이름 사전 ({0: 'helmet', ...}) |
| result.plot() | 박스 그린 이미지 (BGR 배열) |

⚠️ GPU 텐서는 .cpu() 후에 .numpy(). 왜? numpy는 GPU 메모리를 직접 못 읽어서 먼저 CPU로 옮겨 와야 해요. 그리고 탐지가 없으면 boxes 길이가 0이니 에러가 아니라 정상.

### 추적 model.track()

```
model = YOLO("best.pt")
for result in model.track(source="traffic.mp4", persist=True,
                          tracker="botsort.yaml", stream=True):
    if result.boxes.id is not None:
        print(result.boxes.id.int().tolist())   # 프레임별 Track ID
```

▶ 실제로 돌려 보면 데모 모델 best.pt · data/demo.avi (30프레임, 정답 물체 3개) · imgsz=160

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| frame 0 의 result.boxes.id | None | 첫 프레임은 ID 없음 — 박스 점수가 새 Track 시작 기준(0.25) 근처로 낮아서 |
| ID가 처음 붙은 프레임 | frame 8 → [2] | 점수가 오른 뒤에야 Track이 시작됨 |
| frame 29 의 result.boxes.id.int().tolist() | [6, 2] | 2·6번은 끝까지 같은 번호 — 3번은 이 프레임에서 탐지를 놓쳐 빠짐 |
| botsort: 나온 ID 종류 · id가 None인 프레임 수 | [2, 3, 6] · 8 | 번호가 1부터가 아님(확정 못 된 후보도 번호를 씀). 탐지를 놓친 프레임은 이을 박스가 없음 |
| tracker="bytetrack.yaml" 의 ID 종류 | [2, 3, 6] | 이 영상에선 두 추적기 결과가 같았어요 |
| 같은 YOLO 객체로 영상을 한 번 더 track | [8, 9, 12] | 기억이 남아 번호가 이어서 커짐 → 다른 영상엔 새 YOLO 객체 |

| 인자 | 뜻 |
|---|---|
| source | 영상, 카메라, 연속 프레임 |
| persist=True | 이전 프레임 기억을 이어감 (같은 영상 안에서만!). 왜? 이 기억이 있어야 "아까 7번"과 지금 박스를 이을 수 있어요. 다른 영상에 이어 쓰면 엉뚱한 영상의 기억이 섞여요 |
| tracker | "botsort.yaml"(기본) 또는 "bytetrack.yaml" |
| result.boxes.id | Track ID (없으면 None) |

⚠️ 다른 영상으로 넘어갈 땐 새 YOLO 객체를 만들어 기억을 초기화.

🤔 왜 추적기만 바꿔서는 안 좋아져? — 추적은 탐지 결과를 이어 붙이는 단계라서, 탐지가 물체를 놓치면 이을 박스 자체가 없어요. 탐지 성능이 추적 품질의 천장이에요.

⚠️ 속도를 잴 땐 범위를 같이 적어요. "모델 추론 5ms"를 보고 200 FPS라고 하면 안 돼요. 영상 읽기·전처리·후처리·추적·저장 시간까지 더해야 실제 처리 속도예요.

### 같은 걸 터미널(CLI)로 yolo detect ...

```
yolo detect train   model=yolo26n.pt data=dataset.yaml epochs=30 imgsz=640
yolo detect val     model=best.pt data=dataset.yaml
yolo detect predict model=best.pt source=samples/ conf=0.25
yolo detect track   model=best.pt source=traffic.mp4
```

⚠️ CLI는 epochs=30 형식. --epochs 30 아님!

## 📖 Day15 용어 (33개)
- **라벨** (Label (정답 라벨)): 사람이 미리 붙여 둔 정답 값. 분류에서는 각 사례가 어느 클래스인지(예: 0=성공, 1=실패)를 뜻해요.
- **TP** (True Positive (맞춘 양성)): 실제 양성을 양성이라고 맞힌 경우(예: 실제 실패를 실패로 예측). 객체탐지에서는 정답 박스와 짝지어진 예측을 뜻해요.
- **FP** (False Positive (헛경고·오탐)): 실제는 음성인데 양성이라고 한 경우(괜한 경고). 객체탐지에서는 정답과 짝이 없는 예측.
- **FN** (False Negative (놓침)): 실제는 양성인데 음성이라고 한 경우. 찾아야 할 것을 못 찾음.
- **Precision** (Precision (정밀도)): 양성이라고 한 것 중 진짜 양성인 비율. TP ÷ (TP + FP). 헛경고가 늘면 낮아져요.
- **Recall** (Recall (재현율)): 실제 양성 중 모델이 찾아낸 비율. TP ÷ (TP + FN). 놓침이 늘면 낮아져요.
- **누수** (Data Leakage (데이터 누수)): 검증·테스트용 정보나 예측 시점에 알 수 없는 값(예: 실행 후에야 생기는 elapsed_ms)이 학습에 섞여 들어가 점수가 실제보다 좋게 나오는 문제.
- **가중치** (Weight (가중치)): 입력에 곱해지는 숫자. 학습이란 결국 이 가중치들을 loss가 줄어드는 쪽으로 조금씩 고치는 일이에요.
- **epoch** (Epoch (에폭)): train 데이터 전체를 한 바퀴 다 본 단위. 200개를 batch 32로 나누면 7번 갱신하면 1 epoch이에요.
- **Backbone** (Backbone (백본, 특징 추출부)): 모델 앞부분. 사진에서 선·모서리·질감·모양 같은 특징을 뽑아내는 부분이에요. 전이학습에서는 사전학습된 Backbone을 그대로 빌려 써요.
- **Head** (Head (헤드, 분류기)): 모델 끝부분. Backbone이 뽑은 특징을 보고 최종 답(클래스별 점수 등)을 내는 층이에요. 전이학습에서는 내 클래스 수에 맞게 이 부분만 바꿔 끼워요. (Transformer의 Multi-head의 'Head'와는 다른 뜻이에요.)
- **증강** (Data Augmentation (데이터 증강)): 사진을 뒤집고, 살짝 돌리고, 밝기를 바꾸는 등 정답이 안 바뀌는 변형으로 학습 예시를 늘리는 기법. 과적합을 줄여요. train에만 쓰고, 정답을 바꾸는 변형(예: 6↔9 회전)은 금지예요.
- **FPS** (Frames Per Second (초당 프레임 수)): 1초에 몇 장의 Frame인지. 영상 파일에선 재생 속도(재생시간 = Frame 수 ÷ FPS), 모델에선 1초에 처리한 장 수를 뜻해요. 클수록 빠름.
- **객체탐지** (Object Detection): 사진 속 물체가 무엇인지(클래스)와 어디 있는지(박스)를 함께 찾는 작업.
- **Bounding Box** (Bounding Box (경계 상자)): 물체를 딱 감싸는 직사각형. 숫자 4개(좌표)로 위치와 크기를 나타내요.
- **xyxy** (x1, y1, x2, y2): 박스를 왼쪽 위 꼭짓점(x1,y1)과 오른쪽 아래 꼭짓점(x2,y2)으로 적는 방식. 그리기·IoU 계산에 편해요.
- **cxcywh** (center x, center y, width, height): 박스를 중심점(cx,cy)과 너비·높이(w,h)로 적는 방식. YOLO 라벨이 이 형식이에요.
- **정규화 좌표** (Normalized coordinates): 픽셀 좌표를 사진 폭(W)·높이(H)로 나눠 0~1 사이로 바꾼 값. 사진 크기가 바뀌어도 그대로 쓸 수 있어요.
- **GT** (Ground Truth): 사람이 표시한 진짜 정답 박스. 모델 예측을 이것과 비교해서 채점해요.
- **Confidence** (Confidence score (신뢰도 점수)): 모델이 그 박스에 대해 "이게 맞을 거야"라고 매긴 점수(0~1). 낮은 후보를 걸러낼 때 기준으로 써요.
- **IoU** (Intersection over Union (겹침 비율)): 두 박스가 얼마나 겹치는지를 0~1로 나타낸 값. 겹친 넓이 ÷ 두 박스를 합친 넓이.
- **NMS** (Non-Maximum Suppression (비최대 억제)): 한 물체에 겹쳐 나온 박스 여러 개 중 점수가 가장 높은 것만 남기고, 그것과 많이 겹치는 같은 클래스 박스들을 지우는 정리 단계.
- **AP** (Average Precision): 한 클래스에 대해 conf 기준을 바꿔 가며 그린 Precision–Recall 곡선 아래 넓이. 그 클래스를 전반적으로 얼마나 잘 찾는지.
- **mAP** (mean Average Precision): 클래스별 AP의 평균. 탐지 모델의 대표 점수. mAP50은 IoU 0.5 기준, mAP50-95는 0.5~0.95 여러 기준의 평균(더 엄격).
- **YOLO** (You Only Look Once): 사진을 한 번만 통과시켜 모든 위치·크기의 물체 후보를 동시에 예측하는 빠른 탐지 모델 계열.
- **Neck**: Backbone이 뽑은 여러 크기의 특징을 섞어서, 작은 물체와 큰 물체를 함께 다룰 수 있게 하는 중간 부분.
- **격자** (Grid (격자 칸)): YOLO가 사진을 나누는 칸. 칸마다 박스 후보를 하나씩 내요. 촘촘한 격자는 작은 물체, 성긴 격자는 큰 물체 담당.
- **dataset.yaml** (Dataset config file): YOLO에게 학습·검증 사진 폴더 위치와 클래스 이름을 알려 주는 안내서 파일.
- **best.pt** (best / last checkpoint): best.pt는 검증 점수가 가장 좋았던 시점의 모델, last.pt는 마지막 epoch의 모델.
- **추론** (Inference (추론)): 학습이 끝난 모델에 새 사진을 넣어 결과를 얻는 것.
- **추적** (Object Tracking): 영상의 프레임 사이에서 같은 물체를 이어 붙여 같은 번호(Track ID)를 유지하는 것.
- **Track ID**: 추적할 때 개체마다 붙는 번호. Class ID(종류)와 달리 '몇 번째 그 사람'인지를 나타내요.
- **ID switch**: 같은 물체인데 추적 번호가 중간에 바뀌는 실패. 가려졌다 나오거나 비슷한 물체가 엇갈릴 때 생겨요.

---

# Day16 · OCR → 문서 구조화 → VLM 질의응답 → 엣지 배포
_2026-11-09(월) · 문서 전처리 · Tesseract · PSM · CER · 정규식 · JSON 구조화 · VLM · Grounding · Hybrid · TensorRT · Jetson · DeepStream_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day16.html

📌 딱 3줄 요약

- OCR = 문서 사진을 반듯·선명하게 다듬고(전처리) 글자를 Text + Box(위치) + Confidence(점수)로 읽는 것. 품질은 CER·필드 정확도로 따로 잰다.

- 읽은 글자는 정규식·규칙으로 raw(원문) + value(정규화 값) + bbox(근거 위치)를 함께 담은 JSON으로 바꾸고, 업무 규칙으로 검증해 passed · warning · failed를 남긴다.

- VLM은 이미지+질문을 함께 보고 답하지만 그럴듯한 오답이 있어 원문 근거와 대조해야 하고, 배포는 지원 모델을 ONNX → TensorRT Engine으로 바꿔 Jetson·DeepStream에서 돌린다.

🤝 하루 내내 지키는 약속 3가지

- 원본을 덮어쓰지 않는다 — 원본·전처리본·OCR 결과·최종 JSON을 따로 저장하고, 좌표가 어느 이미지 기준인지 같이 적어요. 🤔 왜? 전처리를 바꿔 다시 돌리거나 "이 값이 어디서 나왔지?"를 확인하려면 출발점이 남아 있어야 해요.

- 값은 근거와 함께 — raw(원문) · value(정규화 값) · bbox · confidence를 한 묶음으로. 근거가 없으면 0이나 빈 문자열이 아니라 null + 검토 이유를 남겨요. 🤔 왜? 0은 "금액이 0원"이라는 진짜 값처럼 보여요. null이어야 "못 찾음"이 구분돼요.

- 분모와 출처를 섞지 않는다 — 같은 문서·정답·정규화 규칙에서 비교하고, 직접 돌린 결과인지 남이 기록한 결과인지 적어요. 🤔 왜? 문서 10장 CER과 100장 CER, 내가 돌린 값과 남이 적은 값을 나란히 놓으면 비교가 아니라 착시예요.

🔧 Day16 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| opencv-python | 문서 이미지 읽기, Gray·이진화·원근 보정 | import cv2 |
| numpy | 모서리 좌표 배열(np.float32), 픽셀 계산 | import numpy as np |
| pandas | OCR 단어 결과를 표(DataFrame)로 다루기, 평가표 저장 | import pandas as pd |
| pytesseract | Python에서 Tesseract OCR 엔진을 불러 쓰는 연결 고리 | import pytesseract, from pytesseract import Output |
| (표준 라이브러리) re, json, datetime, dataclasses | 정규식으로 날짜·금액 찾기, JSON 저장, 날짜 검증, 결과 구조 정의 | import re, json (설치 불필요) |
| jiwer | 정답과 OCR 결과로 CER·WER 계산 (직접 짠 cer()와 같은 값) | import jiwer |
| scikit-image (선택) | 연습용 내장 이미지 — data.page()는 왼쪽이 어두운 스캔 페이지 | from skimage import data |
| VLM을 실제로 돌릴 때 (준비된 모델이 있을 때만) |  |  |
| transformers | VLM 모델·Processor 불러오기, pipeline, 답 생성 | from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText |
| torch | 모델이 내부에서 쓰는 계산 엔진 (Day9부터 설치됨) | import torch |
| 배포 (pip 아님) — TensorRT(trtexec 명령 포함)·JetPack·DeepStream은 NVIDIA가 장치·버전별로 따로 배포하는 소프트웨어예요. 수업 기준은 PC의 TensorRT 11.3, Jetson은 그 JetPack이 지원하는 조합. |  |  |

### 🔧 설치 · 확인

```
python -m pip install opencv-python numpy pandas pytesseract jiwer scikit-image
python -c "import cv2, pytesseract; print(cv2.__version__, pytesseract.get_tesseract_version())"
tesseract --version          # 진짜 OCR 엔진 프로그램이 깔려 있나?
tesseract --list-langs       # kor, eng 언어 데이터가 있나?

# (선택) VLM
python -m pip install transformers
# (선택) 배포 — NVIDIA GPU + TensorRT가 설치된 환경에서
trtexec --help
```

```
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

🤔 왜 엔진이 따로야? — Tesseract는 C++로 만든 독립 프로그램이에요. pytesseract는 그 프로그램에 이미지를 넘기고, 나온 글자를 Python 값으로 받아 주는 "통역사"일 뿐이에요. 그래서 통역사만 있고 엔진이 없으면 TesseractNotFoundError가 나요.

⚠️ torch는 다시 깔지 마세요(맞춰 둔 CUDA 버전이 꼬일 수 있어요). 그리고 환경 기록이 중요해요: Python·패키지·모델·언어 데이터 버전과 장치를 적어 두지 않으면 같은 코드라도 다른 글자가 나올 수 있어요.

💡 pip 이름 ≠ import 이름: opencv-python → cv2. 그리고 pip 패키지 pytesseract ≠ 프로그램 tesseract. pipeline(device_map="auto")를 쓰려면 보통 accelerate 패키지도 필요해요.

## 이거 어디에 써?

Day13~15는 사진의 픽셀·물체를 다뤘어요(OpenCV, 분류, YOLO). Day16은 사진 속 글자를 읽고, 문서에 질문하고, 그 결과를 근거와 함께 업무 데이터로 남기는 "문서 AI"예요. 마지막엔 이런 모델을 현장 장비에서 돌리는 배포 구조까지 봐요.

🧾

청구서 · 영수증공급자·발행일·합계를 JSON으로 자동 추출

🔎

검색 가능한 PDF스캔 이미지 위에 보이지 않는 글자 층을 깔기

💬

문서 질의응답"합계 금액과 근거 위치를 알려줘"

📷

현장 영상 분석Jetson + DeepStream으로 카메라 옆에서 추론

오늘 만드는 문서 AI의 전체 모양이에요. 핵심은 "값만 남기지 않는다" — 값이 어디서 왔는지(원문·위치), 믿어도 되는지(검증 상태)를 같이 남겨요.

[그림: ] 무엇을 보나: 왼쪽 문서가 위쪽 OCR 길(전처리 → 글자·위치·점수)과 아래쪽 VLM 길(원본이나 잘라 낸 부분 + 질문 → 답변·근거 후보)로 갈라졌다가, 가운데 구조화·검증에서 다시 만나요. 오른쪽 점선 상자가 최종 산출물이에요. 🤔 왜? OCR은 "정확히 무슨 글자가 어디 있나"에 강하고, VLM은 "이게 무슨 뜻인가"에 강해요. 둘 다 틀릴 수 있으니 마지막에 원문·규칙과 대조하는 단계가 꼭 있어야 하고, 대조 결과(검토 이유)까지 남겨야 사람이 다시 확인할 수 있어요.

⚠️ 처음부터 지킬 원칙 4가지 — ① 개인정보·기밀: 외부 OCR·VLM 서비스에 올리기 전 허용 범위 확인. ② 원본 보존: 원본·전처리본·OCR 결과·최종 JSON을 따로 저장. ③ 사실성: 자연스러운 답보다 원문 문자열·좌표가 우선. ④ 평가: 샘플 몇 개 성공이 아니라 문서 유형·해상도·언어·표 구조별 오류를 기록.

## ① 문서 전처리 — 읽기 좋은 사진으로 다듬기

OCR(Optical Character Recognition)은 이미지에서 글자를 읽는 전체 과정이에요: 입력 획득(스캔·사진·PDF·캡처) → 전처리 → 문자 인식(글자·단어·줄 + 신뢰도·위치) → 후처리(읽기 순서·띄어쓰기·단위·날짜·금액 정리) → 정보 추출(JSON·표·검색). 그중 첫 단추가 전처리예요.

| 문서 이미지가 어려운 이유 | 예 |
|---|---|
| 촬영·스캔 문제 | 회전·원근으로 글줄이 기울어짐 · 그림자·반사광으로 같은 배경도 밝기가 다름 · 압축·흔들림·저해상도로 글자 획이 사라짐 · 배경 무늬·도장·서명이 글자와 겹침 |
| 문서 구조 문제 | 다단·표·머리글·각주·세로쓰기 등 읽기 순서 · 0/O, 1/l/I처럼 닮은 글자 · 혼합 언어·특수기호·통화 · 글자가 아닌 체크박스·선·바코드 |

사진 문서 기울기·그림자·노이즈→ Gray 색 → 밝기 1채널→ Threshold 글자 0 / 배경 255→ Perspective 정면으로 펴기→ OCR

[그림: 기울어지고 그림자 진 가상 청구서 사진을 Gray, Otsu, Adaptive, Perspective 보정 순서로 바꾼 다섯 장면] 무엇을 보나: 가상 청구서를 일부러 기울이고 왼쪽 위에 그림자를 넣은 "촬영본"(①)이에요. ② Gray는 채널 축이 사라져 shape가 (1130, 920)이 돼요. ③ Otsu는 사진 전체에 기준 하나(T=146)를 써서 빨간 점선 안 그림자 쪽이 통째로 검게 먹혔어요. ④ Adaptive는 같은 자리 글자가 살아 있어요. ⑤ 노란 1→2→3→4 모서리를 (700, 1000) 크기로 펴니 반듯한 문서가 됐어요. 🤔 왜? OCR 엔진은 "반듯하고, 글자와 배경이 확실히 갈리는" 입력을 전제로 줄을 나누고 글자를 읽어요. 그림자는 대비를, 기울기는 줄 나누기를 망가뜨리니 읽기 전에 그 전제에 맞춰 주는 거예요.

🔧 읽기 · Gray 변환 cv2.imread() · cv2.cvtColor() 명령어 사전에서 자세히 →

```
image = cv2.imread("invoice.png", cv2.IMREAD_COLOR)   # 실패하면 에러 없이 None
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)          # (H, W, 3) → (H, W)
```

### 이진화 — 전역 Otsu vs 적응형 Adaptive

이진화(Threshold)는 밝기 0~255인 Gray를 기준선으로 갈라 글자 0 / 배경 255 두 값만 남기는 거예요(Day13 ④에서 자세히 봤어요). 문서에서는 그림자 때문에 "기준선 하나"가 자주 실패해요.

[그림: 그림자 진 문서의 가로 한 줄 밝기 그래프에 Otsu 전역 기준선과 Adaptive 지역 기준선을 겹쳐 그린 그림] 무엇을 보나: 왼쪽 분홍 줄(y=225)의 밝기를 오른쪽에 그렸어요. 회색 선이 실제 밝기 — 왼쪽으로 갈수록 그림자 때문에 종이도 어두워지고, 뾰족하게 꺼진 곳이 글자 획이에요. 빨간 점선(Otsu, 146)은 어디서나 같은 높이라 빨간 구간의 종이까지 선 아래로 떨어져 검게 칠해져요. 초록 선(Adaptive)은 배경 밝기를 따라 오르내리며 늘 배경보다 조금 아래에 있어서 글자 획만 그 아래로 내려가요. 🤔 왜? Adaptive는 "전체에서 어두운가?"가 아니라 "내 주변보다 어두운가?"를 물어요. 그림자는 글자와 주변 종이를 같이 어둡게 만드니, 주변과 비교하면 그림자 효과가 상쇄돼요.

🔧 cv2.threshold() + Otsu · cv2.adaptiveThreshold() 명령어 사전에서 자세히 →

```
otsu_t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
```

### 🔬 값을 바꾸면 실제로 어떻게 될까

oneday 요약본이 왼쪽이 어두운 스캔 페이지(scikit-image 내장 data.page())를 이진화한 뒤 Tesseract(PSM 6)로 읽어, 숫자 하나씩만 바꿔 가며 CER을 잰 결과예요.

[그림: blockSize 3, 15, 31, 101과 C 0, 11, 30, 60, 그리고 Gray·Otsu·Adaptive 이진화 방법에 따른 CER 막대 그래프] 무엇을 보나: 보라색이 강의 코드 값(31, 11)이에요. blockSize 3은 0.137로 나빠지고, 15가 0.050으로 가장 좋았어요. 30(짝수)은 아예 실행 오류. C는 0이면 0.388로 크게 나빠지고, 30에서 0.020으로 코드 값보다도 좋았어요. 오른쪽은 이진화 방법 비교 — Otsu(0.361)가 이진화 안 한 Gray(0.324)보다도 나빴어요. 🤔 왜? Otsu는 페이지 전체에 기준 하나(157)를 써서, 어두운 왼쪽은 종이까지 검게 칠해져 글자가 묻혀요. Adaptive는 주변 31×31마다 기준을 따로 잡아 그림자를 상쇄해요. 그리고 C=0이면 "주변 평균보다 조금만 어두워도 검정"이라 종이 결이 점으로 찍히고, 60이면 기준이 너무 낮아 흐린 획까지 하얗게 사라져요.

💡 정답은 "31, 11"이 아니에요. 이 페이지에선 C=30이 더 좋았듯, 알맞은 값은 문서마다 달라요. 후보 몇 개를 돌려 보고 CER이 가장 낮은 걸 고르는 게 방법이에요 — 그래서 ③ 평가가 전처리 선택의 잣대가 돼요.

### 원근 보정 — 비스듬한 문서를 정면으로

문서를 비스듬히 찍으면 사각형이 사다리꼴이 돼요. 실제 문서의 네 모서리(src)와 펼친 뒤 네 모서리(dst)를 짝지으면, 둘을 잇는 3×3 변환 행렬 M이 하나 정해져요. 아래 왼쪽 사진의 번호 점 4개를 끌어 보세요. 오른쪽이 그 점들로 편 결과예요.

🔧 cv2.getPerspectiveTransform() · cv2.warpPerspective() 명령어 사전에서 자세히 →

```
M = cv2.getPerspectiveTransform(src, dst)                           # 3×3 행렬
warped = cv2.warpPerspective(image, M, (700, 1000), borderValue=(255, 255, 255))   # (width, height)
```

🔧 보정본 좌표 → 원본 사진 좌표 cv2.perspectiveTransform() · 작은 사진 확대 cv2.resize() 명령어 사전에서 자세히 →

```
back = cv2.perspectiveTransform(box, np.linalg.inv(M))   # 보정본 점 → 원본 사진 점
small = cv2.resize(warped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
```

## ② OCR 실행 — Text · Box · Confidence

OCR 엔진 안을 확대해 보면 네 단계가 보여요. 엔진 하나가 다 하기도 하고, 단계별로 다른 부품을 쓰기도 해요.

[그림: ] 무엇을 보나: 위 네 칸이 단계, 아래가 각 단계에서 다루는 모습이에요. ① 기울어진 전체 문서를 점검하고, ② 보라 상자처럼 글자 덩어리·줄·표(점선) 영역을 나누고, ③ 한 줄씩 잘라 문자열 + 점수 + 위치로 바꾸고, ④ 제목 → 본문 → 표 → 합계 순서로 정리해요. 🤔 왜? 단계가 나뉘어 있어서 실패도 단계별로 따져야 해요. ②에서 영역을 못 찾으면 ③은 손도 못 대고(놓침), ④가 틀리면 글자는 다 맞아도 "합계"와 "1,234,500원"이 엉뚱한 짝이 돼요.

| OCR 결과 형식 | 들어 있는 것 |
|---|---|
| Plain text | 전체 문자열 — 빠르게 확인용 (image_to_string) |
| TSV / DataFrame | 단어별 text · conf · left · top · width · height (image_to_data) |
| hOCR / ALTO XML | 페이지 · 문단 · 줄 · 단어 구조와 위치 |
| Searchable PDF | 이미지 위에 보이지 않는 텍스트 층 → 검색·복사 가능 |
| Custom detector + recognizer | 영역 탐지 모델과 문자 인식 모델을 따로 쓰는 구조 |

### PSM — "페이지가 이렇게 생겼다"는 가정

PSM(Page Segmentation Mode)은 Tesseract에게 "이 이미지는 이런 모양이야"라고 미리 알려 주는 설정이에요. 가정이 틀리면 같은 글자도 엉뚱하게 나눠 읽어요.

[그림: ] 무엇을 보나: 각 PSM이 "이미지 안에 무엇이 있다고 가정하는지"예요. 3은 제목·2단·표(점선)를 스스로 나누고, 6은 통째로 한 블록, 7은 한 줄, 8은 한 단어로 봐요. 🤔 왜? 이미 한 줄만 잘라서 넣었는데 PSM 3으로 "페이지 분할"을 시키면, 엔진이 없는 단·블록을 찾다가 줄을 쪼개거나 빈칸을 글자로 오해할 수 있어요. 입력이 단순할수록 가정도 단순하게 주는 게 유리해요(항상 그런 건 아니니 검증 문서로 비교해요).

🔧 단어별 결과 받기 pytesseract.image_to_data() 명령어 사전에서 자세히 →

```
ocr = pytesseract.image_to_data(binary, lang="kor+eng", config="--oem 3 --psm 6", output_type=Output.DATAFRAME)
words = ocr[ocr.level.eq(5) & ocr.text.str.strip().ne("")]   # 단어 행 + 빈 글자 제외
```

[그림: 가상 청구서의 단어마다 confidence 색으로 박스를 그리고 text, conf, left, top, width, height 표로 정리한 설명용 그림] 무엇을 보나: (설명용 그림 — 박스는 글자를 그린 위치, conf는 가상 값이에요) 왼쪽 단어 박스 하나 = 오른쪽 표 한 줄. 박스 색이 conf예요(초록 ≥ 80, 주황 50~80, 빨강 < 50). 합계 금액은 강의 예시와 같은 bbox [530,820,790,865], conf 88.6. 표의 left·top·width·height를 [left, top, left+width, top+height]로 바꾸면 xyxy가 돼요. 🤔 왜? 수량 1, 3처럼 짧은 숫자는 엔진이 비교할 앞뒤 글자가 없어서 맞게 읽어도 점수가 낮게 나오기 쉬워요. 그래서 낮은 점수를 "삭제"가 아니라 "사람이 볼 목록(review)"에 넣는 거예요.

[그림: 단어 21개 confidence 분포에서 오답 DOCO3 하나가 65.6점, 나머지 20개는 92.8점 이상이고 기준 50과 90 선을 비교한 그림] 무엇을 보나: oneday 요약본이 정면 보정한 청구서를 실제로 OCR한 결과예요. 21단어 중 틀린 건 DOC03을 DOCO3(숫자 0 → 글자 O)로 읽은 하나뿐이고, 그 점수는 65.6이에요. 기준 50이면 오답이 그대로 통과(검토 0개), 기준 90이면 오답 하나만 정확히 걸려요(검토 1개). 🤔 왜? "50"이 상식적인 기준처럼 보여도 엔진은 틀린 글자에도 꽤 높은 점수를 줘요. 0과 O는 모양이 거의 같아서 엔진 입장에선 "꽤 확신"하거든요. 그래서 기준은 감으로 정하지 말고 오답들의 점수 분포를 먼저 본 뒤 그어요.

## ③ OCR 품질 평가 — 글자 · 단어 · 필드를 따로

| 지표 | 계산 | 질문 |
|---|---|---|
| CER | (치환 + 삭제 + 삽입) / 정답 글자 수 | 글자 단위로 얼마나 틀렸나? (낮을수록 좋음) |
| WER | (치환 + 삭제 + 삽입) / 정답 단어 수 | 단어 단위로 얼마나 틀렸나? |
| Field accuracy | 날짜·금액·문서번호 등 필수 항목이 정확한 비율 | 업무에 쓸 값이 맞았나? |
| 구조 평가 | 읽기 순서·표 셀·Box별 기준과 분모를 정해서 | 순서·칸 관계가 맞았나? |
| Latency | 페이지당 전처리 · OCR · 후처리 시간을 각각 | 어디서 느린가? |

편집 거리(Levenshtein distance)는 "정답을 OCR 결과로 바꾸려면 글자를 최소 몇 번 고쳐야 하나"예요. 고치는 방법은 세 가지: S 다른 글자로 바꾸기, D 지우기(OCR이 빠뜨림), I 끼워 넣기(OCR이 없는 걸 추가).

[그림: 정답과 OCR 결과를 한 글자씩 맞춰 놓고 치환 S, 삭제 D, 삽입 I를 색칠한 두 가지 정렬 예] 무엇을 보나: 위는 강의 예시 — 정답 1234500을 12345O0로 읽었어요. 숫자 0을 영문 O로 바꾼 S 1번 → CER = 1/7 ≈ 14.3%. 아래는 날짜 예시로 S·D·I가 모두 나와요: :를 끼워 넣고(I), 0→O·1→l로 바꾸고(S), 0을 빠뜨렸어요(D). 🤔 왜? 줄을 이렇게 맞춰 봐야 "어떤 실수를 했는지"가 보여요. 그리고 7글자 중 1글자(14%)만 틀려도 그게 금액이면 필드 값은 통째로 틀린 거라, CER과 필드 정확도를 따로 봐요.

직접 해 보세요. 금액·날짜를 바꿔 넣으면 CER, S·D·I, 그리고 "숫자만 뽑았을 때 필드 값이 같은지"가 함께 바뀌어요.

🔧 CER 계산 함수 명령어 사전에서 자세히 →

```
errors = levenshtein(reference, prediction)   # 최소 S·D·I 횟수
return errors / len(reference)                # 분모 = 정답 글자 수
```

🔧 라이브러리로 한 줄 jiwer.cer() 명령어 사전에서 자세히 →

```
read = " ".join(text.split())                   # OCR 글자 — 줄바꿈·여러 공백을 한 칸으로
score = jiwer.cer(truth, read)                  # (정답, 예측) 순서!
```

[그림: 0.5배, 0.4배, 0.4배를 2배 확대한 청구서 사진의 CER과 필드 정답 수를 나란히 비교한 막대 그래프] 무엇을 보나: oneday 요약본이 청구서를 일부러 작게 줄여 OCR한 결과예요. 0.4배 사진은 CER이 겨우 6.6%인데 공급자 필드를 통째로 놓쳤어요(2/3). 2배로 키우면 CER 0.008, 필드 3/3으로 회복돼요. 🤔 왜? OCR이 줄 맨 앞에 ‘ 한 글자를 붙여 ‘SUPPLIER: ALPHA STUDIO로 읽었는데, 구조화 규칙이 ^SUPPLIER:(줄이 SUPPLIER로 시작)라서 매칭이 안 됐어요. 글자 하나 = CER엔 작은 오류, 하지만 라벨 규칙을 깨면 값 전체를 잃어요. 그래서 CER과 필드 정답을 함께 봐요.

[그림: 같은 문서를 금액 CER 14.3%, 필드 정확도 66.7%, JSON 문법 100%, 원문 근거 불일치 검토로 따로 채점한 네 칸 그림] 무엇을 보나: 강의의 가상 수치예요. 같은 문서 한 장인데 금액 CER은 14.3%(1/7 글자), 필드 정확도는 66.7%(3개 중 2개), JSON 문법은 100%(파싱 성공), 원문 근거는 "금액 불일치 → 검토"예요. 🤔 왜? 잣대마다 보는 단위와 분모가 달라서예요. JSON이 완벽해도 값이 틀릴 수 있고, CER이 낮아도 그 한 글자가 금액이면 업무상 실패예요. 그래서 하나로 합치지 않고 같은 문서·정답·정규화 규칙에서 따로 비교해요.

## ④ 구조화 — 글자를 업무 데이터(JSON)로

OCR 문자열은 그냥 긴 글이에요. 프로그램이 쓰려면 "발행일은 2026-11-09, 합계는 정수 1234500" 같은 필드로 바꿔야 해요. 이걸 후처리라고 하고 다섯 단계로 해요.

정규화 공백·통화·날짜·숫자 형식 통일→ 영역화 제목·표·합계·서명 블록→ 필드 추출 번호·날짜·공급자·합계→ 검증 합계 = 공급가 + 세금→ 출력 원문·값·box·conf·상태

🔧 정규식으로 날짜 · 금액 찾기 re.search() 명령어 사전에서 자세히 →

```
am = re.search(money_pat, text)                 # 없으면 None
total = int(am.group(1).replace(",", ""))       # "1,234,500" → 1234500
```

### Key-value — "라벨 · 값 · 근거"를 한 묶음으로

[그림: 청구서에서 발행일, 공급자, 합계 라벨 박스와 오른쪽 값 박스를 화살표로 잇고 total_amount의 raw, value, currency, confidence, bbox와 검증 결과를 정리한 그림] 무엇을 보나: 왼쪽은 라벨(점선) 오른쪽에 있는 값(실선)을 짝지은 모습이에요. 오른쪽은 그중 합계 한 칸에 담는 정보 — 원문 raw "1,234,500원", 정수 value 1234500, 통화, 점수, 위치 — 와 그 아래 검증(공급가액 1,122,273 + 세액 112,227 = 합계 1,234,500 → passed)이에요. 🤔 왜? value만 남기면 나중에 "1234500은 어디서 왔지?"를 확인할 길이 없어요. raw와 bbox가 있으면 원본 이미지의 그 자리를 잘라 보며 OCR이 틀렸는지, 정규화가 틀렸는지를 바로 가를 수 있어요.

🔧 필드와 근거를 JSON으로 json.dump() 명령어 사전에서 자세히 →

```
with open("invoice_001.json", "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
```

### 표 추출 — 글자보다 "셀 구조"가 어렵다

| 단순 OCR로 되는 경우 | 추가 모델·규칙이 필요한 경우 |
|---|---|
| 선이 분명하고 병합 셀이 거의 없음 | 선 없는 표 · 여러 줄 헤더 · 병합 셀 |
| 열 수와 위치가 문서마다 일정 | 문서마다 열 위치·폭이 크게 다름 |
| 각 셀을 crop해 한 줄 OCR(PSM 7)로 읽을 수 있음 | 읽기 순서와 "이 글자가 어느 셀 소속인지"를 따로 예측해야 함 |
| 합계·수량·단가 관계로 오류를 검증할 수 있음 | 문자 인식보다 구조 복원이 더 어려움 |

🤔 왜 표 OCR ≠ 표 구조 인식? — "900,000"이라는 글자를 정확히 읽어도, 그게 "GPU 서버 임대" 행의 "금액" 열인지 모르면 쓸 수가 없어요. 글자 읽기와 행·열 관계 복원은 다른 문제라 따로 평가해요.

🔧 업무 규칙으로 검증 validate_invoice() 명령어 사전에서 자세히 →

```
if value is None or value == "":                # 금액 0은 누락이 아님
    errors.append(f"필수 필드 누락: {name}")
```

## ⑤ VLM — 이미지를 보고 질문에 답하는 모델

VLM(Vision-Language Model)은 이미지와 글(질문)을 함께 입력받아 글로 답하는 모델이에요. OCR과 잘하는 일이 달라요.

|  | OCR | VLM |
|---|---|---|
| 강점 | 문자·단어·줄을 그대로 받아 적고 위치를 연결 · 규칙 후처리와 결합하기 쉬움 · 숫자·문서번호 · 유형별 품질·비용을 직접 비교 | 이미지와 질문을 함께 보고 문맥을 설명 · 표·도식·화면 구성의 의미를 종합 · 문서 유형이 다양해도 유연하게 질의응답 |
| 약점 | 구조·의미 이해는 따로 만들어야 함 | 자연스러운 오답과 근거 없는 추론 가능성 |

[그림: ] 무엇을 보나: 왼쪽 messages의 이미지 항목과 질문 항목이 Processor를 지나 각각 pixel_values(숫자 이미지)와 input_ids(토큰 번호)가 돼요. 이미지는 Vision Encoder가 특징으로 바꾸고 Projector가 그 특징을 "언어 모델이 알아듣는 공간"으로 옮긴 뒤, 질문 토큰과 함께 언어 모델에 들어가 답을 한 토큰씩 만들어요. 🤔 왜? 언어 모델은 원래 글자 토큰만 이해해요. 이미지 특징을 토큰과 같은 모양으로 맞춰 주는 연결 모듈이 있어야 "그림 + 질문"을 한 문장처럼 읽을 수 있어요. 모델마다 입력 키와 연결 방식은 조금씩 달라요.

| 입력 요소 | 뜻 |
|---|---|
| role | 대화 역할: system · user · assistant |
| content | 문자열 하나가 아니라 image · text 항목의 목록 |
| Processor | 이미지 전처리 · 특수 토큰 · Chat template · 토큰화를 한꺼번에 |
| model.generate | 준비된 input_ids · attention_mask · pixel_values로 글 생성 |

⚠️ 모델마다 입력 형식 · 지원 이미지 수 · 해상도 · VRAM(GPU 메모리) 요구량이 달라요. 쓰기 전에 그 모델의 설명서를 확인해요.

🔧 간단 버전 pipeline("image-text-to-text") 명령어 사전에서 자세히 →

```
pipe = pipeline("image-text-to-text", model=MODEL_ID, device_map="auto", dtype="auto")
result = pipe(text=messages, max_new_tokens=128, do_sample=False, return_full_text=False)
```

🔧 직접 버전 AutoProcessor · apply_chat_template() · generate() 명령어 사전에서 자세히 →

```
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
```

## ⑥ 근거 검증 · OCR+VLM Hybrid · 보안

VLM 답이 자연스럽다고 사실은 아니에요. 그래서 "보고 말한 것"과 "추측한 것"을 구분해요.

| 용어 | 뜻 | 예 |
|---|---|---|
| 관찰 | 이미지에 실제로 보이는 글자·색·표·물체를 설명 | "합계 칸에 1,234,500원이 있어요" |
| 추론 | 문서 목적·의도·원인·법적 의미를 추정 | "이 회사는 GPU를 많이 쓰는 것 같아요" |
| Hallucination | 보이지 않는 값·표·근거를 자연스럽게 지어냄 | 문서에 없는 "부가세 포함" 문구를 인용 |
| Grounding | 답과 함께 원문 문자열·페이지·Box·Crop을 제시 | evidence "1,234,500원", page 1, bbox [...] |
| Fallback | 근거가 부족하면 "확인할 수 없음" + 검토 필요 상태 | answer: null, status: review |

| 프롬프트 입력 계약 | 프롬프트 출력 계약 |
|---|---|
| 문서 유형 · 페이지 범위 · 질문을 명시 | JSON key와 자료형을 고정 |
| 허용 정보는 이미지에 보이는 내용만 | answer · evidence · page · bbox를 분리 |
| 모르면 null 또는 "확인 불가"로 | 지원하지 않는 좌표·근거는 null |
| 금액·날짜·문서번호는 원문 문자열도 함께 | 스키마·원문 검증 실패 → 재시도 또는 사람 검토 |

🤔 왜 "모르면 null"을 미리 약속해? — 생성 모델은 빈칸을 그럴듯한 말로 채우는 게 기본 동작이에요. "모르면 null"이라는 출구를 정해 주지 않으면 지어낸 값과 진짜 값이 같은 모양으로 나와서 구분할 수 없어요.

🔧 VLM 응답 JSON 최소 검사 json.loads() 명령어 사전에서 자세히 →

```
data = json.loads(raw)                                           # 깨진 JSON이면 JSONDecodeError
missing = {"answer", "evidence", "page", "bbox"} - data.keys()   # 빠진 필수 key
```

### 🔬 실제 VLM은 얼마나 지어낼까

oneday 요약본이 SmolVLM-256M-Instruct(do_sample=False)에 정면 보정한 청구서를 보여 주고 질문 6개를 던진 결과예요. 아래쪽 세 질문은 문서에 없는 값이에요.

| 질문 | VLM 답 | OCR 원문 | 대조 결과 |
|---|---|---|---|
| 합계 | 33000 | 33000 | ✅ 채택 |
| 공급자 | ALPHA STUDIO | ALPHA STUDIO | ✅ 채택 |
| 발행일 | 2026-11-03 | 2026-11-03 | ✅ 채택 |
| 세금 | 0. | 문서에 없음 | ⏸ 보류 |
| 할인액 | 33000 (합계를 그대로) | 문서에 없음 | ⏸ 보류 |
| 공급자 전화번호 | ALPHA STUDIO | 문서에 없음 | ⏸ 보류 |

⚠️ 프롬프트로는 못 막았어요. 질문 끝에 "없으면 null이라고 답해"를 붙여도 세금은 여전히 0., "JSON으로 답해"라고 해도 33000만 답했어요. 🤔 왜? 작은 VLM일수록 지시를 잘 안 따르고, 생성 모델은 원래 "질문에 어울리는 그럴듯한 답"을 만드는 게 기본 동작이에요. 빈칸이 있으면 근처에 보이는 숫자나 글자로 채워 버려요. 그래서 믿을 건 프롬프트가 아니라 원문 대조예요.

🔧 가장 단순한 결합 규칙 명령어 사전에서 자세히 →

```
if answer == total["raw"]:          # VLM 답이 OCR 원문과 같으면
    final = total["value"]          # → 채택 (아니면 None = 보류)
```

### OCR + VLM을 합치는 세 가지 방식

[그림: ] 무엇을 보나: 세 줄이 세 가지 조립 방법이에요. A는 OCR 결과에 규칙만 씌우고, B는 VLM에 바로 묻고 근거를 검증하고, C(Hybrid)는 OCR이 찾은 글자·위치와 잘라 낸 이미지를 VLM에 주고 판단을 맡겨요. 🤔 왜? Hybrid는 두 점수를 평균 내는 게 아니라 역할을 나누는 거예요. OCR의 정확한 전사·좌표가 VLM 답의 근거(grounding)가 되고, VLM의 문맥 이해가 OCR이 놓친 관계(어느 숫자가 합계인지)를 메워 줘요.

| 문제 | OCR | VLM | 권장 |
|---|---|---|---|
| 정확한 문서번호 받아 적기 | 강점 | 오인 가능 | OCR 우선 · VLM 검증 |
| 표·도식의 의미 설명 | 구조 복원 필요 | 강점 | OCR 원문을 함께 제공 |
| 고정 양식 Key-value | 규칙화 쉬움 | 과도한 비용 가능 | 예외 문서만 VLM |
| 다양한 문서 질의응답 | 템플릿 관리 증가 | 유연함 | OCR evidence로 grounding |
| 엣지 실시간 처리 | 경량화 가능 | 모델 규모 부담 | 단계별 라우팅 |

VLM 질문은 보통 다섯 유형이에요: 읽기("상단 제목은?") · 탐색("합계는 어느 영역?") · 비교("두 표에서 증가한 항목?") · 설명("흐름도의 단계 요약") · 검증("OCR 날짜와 이미지의 날짜가 같은가?").

### 같은 데이터에서 비교하기

| 관점 | 무엇을 재나 |
|---|---|
| 공통 | 동일 문서·질문·정답으로 필드 정확도 · 근거 일치 · 총 시간 |
| 방법별 | OCR은 CER · Box IoU, VLM은 답변 · JSON 유효성 · 근거 없는 답 |
| Hybrid | 필드 정확도 · 검토율 · 실패 복구율 · 총비용 |
| Slice | 영수증·계약서·표·손글씨·저해상도·다국어 문서별로 쪼개서 |
| 운영 | 처리량 · VRAM · 모델 로딩 시간 · 개인정보 처리 위치 |

🤔 왜 Slice로 쪼개 봐? — 전체 평균 95%여도 손글씨 문서만 모으면 40%일 수 있어요. 평균은 많은 쪽(깨끗한 스캔)에 끌려가서 약한 부분을 숨겨요.

### 입력 크기와 메모리 — 다 넣지 말고 필요한 곳만

[그림: 문서 전체를 작게 줄였을 때 글자가 뭉개지는 모습과 같은 영역만 잘랐을 때 또렷한 모습, 그리고 해상도에 따라 이미지 조각 수가 제곱으로 늘어나는 그래프] 무엇을 보나: 왼쪽은 1000×1200 페이지를 187×224로 통째로 줄인 모습 — 28px 글자가 약 5px가 돼요. 가운데 A는 그 축소본의 빨간 영역을 확대한 것(뭉개짐), B는 원본에서 같은 영역만 잘라 폭 336px로 만든 것(또렷함). 오른쪽은 14px 조각(패치)으로 나눈다고 가정한 예시 계산이에요. 🤔 왜? VLM은 이미지를 작은 조각으로 나눠 토큰처럼 다뤄요. 한 변을 2배로 키우면 조각은 4배(가로×세로)라 계산·메모리가 함께 커져요. 그래서 전체를 크게 넣기보다 필요한 영역을 crop하면 더 적은 조각으로 더 또렷하게 볼 수 있어요(B는 72조각, 축소 전체는 208조각).

- 여러 페이지를 한 번에 넣기보다 질문에 필요한 페이지를 검색·선택해요.

- 메모리는 모델 로딩(가중치)과 추론 중(Activation · KV cache)을 구분해서 봐요.

- GPU 측정은 워밍업 후 Latency · 처리량 · Peak VRAM을 같은 조건에서 비교해요. (첫 실행은 준비 작업 때문에 유난히 느려서)

- 비디오 VLM은 프레임 선택이 먼저예요: 고정 간격·장면 변화·이벤트 중심으로 샘플링하고, num_frames는 모델 지원 길이와 하드웨어를 보고 정해요. frame index·timestamp를 같이 저장해 시간 순서를 잃지 않게 하고, 영상 전체의 결론과 특정 프레임의 관찰을 구분해요.

### 보안 · 개인정보 점검표

| 항목 | 질문 |
|---|---|
| 전송 | 외부 API로 보내는 이미지에 개인정보·계약 정보가 있는가? |
| 저장 | 원본 이미지 · OCR 원문 · 프롬프트 · 응답 · 로그의 보존 기간은? |
| 권한 | 프로젝트 폴더 · Object Storage · 모델 캐시 접근 권한은? |
| 마스킹 | 주민번호 · 계좌번호 · 전화번호 등 민감 영역을 먼저 가릴 것인가? |
| 검토 | 자동 추출 결과가 결제·계약·의료 판단에 직접 쓰이는가? |

⚠️ 결과가 돈·계약·건강에 바로 쓰인다면 자동 결과를 그대로 흘려보내지 말고 사람 검토 단계를 꼭 둬요. 틀렸을 때의 비용이 크니까요.

## ⑦ 엣지 배포 — TensorRT · Jetson · DeepStream

배포에서는 모델 형식(ONNX·Engine), 실행 소프트웨어(TensorRT·DeepStream), 장치(GPU 서버·Jetson)를 구분해서 골라요. 엣지는 서버 대신 카메라 옆 작은 장비에서 바로 추론하는 것을 말해요.

| 이름 | 역할 | 확인할 조건 |
|---|---|---|
| TensorRT | 지원 모델의 Engine 빌드 · 추론 (NVIDIA GPU용) | 연산자 · 정밀도 · 버전 · GPU |
| Jetson | 현장에 설치하는 실행 장치 | 장치별 JetPack 지원 조합 |
| DeepStream | 영상 입력부터 결과까지 파이프라인으로 연결 | 버전 · 장치 · 플러그인 지원 |

[그림: ] 무엇을 보나: 위 줄은 모델이 배포되는 길이에요. 보라 영역(Build)에서 ONNX를 받아 그 GPU에 맞춘 Engine을 만들고, 분홍 영역(Runtime)에서 Engine을 불러와 매번 추론해요. 아래 줄은 DeepStream이 영상을 처리하는 순서 — 보라 점선처럼 nvinfer 안에서 바로 그 TensorRT Engine이 돌아요. 🤔 왜? TensorRT는 빌드할 때 "이 GPU에서 어떤 계산 방법이 가장 빠른지"를 직접 재 보고 연산을 합쳐 두기 때문에 빠른 대신, 그 결과가 그 GPU·버전에 묶여요. 그래서 빌드는 실제로 돌릴 환경에서 하고, 옮겨 다니는 건 ONNX예요.

| ONNX | TensorRT |
|---|---|
| 프레임워크 사이 교환을 위한 계산 그래프 형식 | NVIDIA GPU용 추론 Engine을 빌드 |
| 입력·출력 이름 · shape · 연산자 집합을 명시 | 연산 결합 · 상수 접기 · 커널 선택으로 최적화 |
| Export 후 ONNX Runtime 등으로 원본과 수치 대조 | 11.x에서는 정밀도를 모델의 Tensor 자료형에 반영 |
| 동적 shape와 지원 연산자 확인 | Engine·Context와 GPU·버전·연산자 호환성 확인 |

| TensorRT 핵심 객체 | 단계 | 하는 일 |
|---|---|---|
| Builder | 빌드 | Network와 BuilderConfig로 Engine을 만듦 |
| BuilderConfig | 빌드 | 메모리 · 타이밍 · 최적화 프로파일(동적 shape 범위) 설정 |
| Runtime | 실행 | 저장된(Serialized) Engine을 역직렬화(deserialize)해 다시 객체로 |
| Engine | 실행 | 최적화된 입력·출력·연산 정보 보유 |
| ExecutionContext | 실행 | 동적 shape · Tensor 주소 · 추론 상태 관리 (실제 호출 단위) |
| CUDA Stream | 실행 | 비동기 실행과 "다 끝났는지" 확인 시점 제어 |

🔧 PyTorch → ONNX 내보내기 torch.onnx.export() 명령어 사전에서 자세히 →

```
model.eval()                                       # Dropout·BatchNorm을 추론 모드로
torch.onnx.export(model, (x,), "model.onnx")       # PyTorch → ONNX
```

🔧 trtexec로 빌드 · 추론 확인 명령어 사전에서 자세히 →

```
trtexec --onnx=model.onnx --saveEngine=model.engine   # ONNX → Engine (대상 장치에서)
trtexec --loadEngine=model.engine                     # 저장한 Engine 추론 확인
```

[그림: 같은 모델을 FP32, FP16, INT8로 저장할 때 10MB, 5MB, 2.5MB로 줄어드는 막대 그래프 (예시 계산)] 무엇을 보나: 가중치 250만 개짜리 모델을 가정한 예시 계산이에요(실측 아님). 숫자 하나를 32 → 16 → 8비트로 적으면 10MB → 5MB → 2.5MB로 반씩 줄어요. 🤔 왜? 크기 = 숫자 개수 × 숫자 하나의 비트 수라서요. 작은 숫자는 메모리를 덜 쓰고 GPU가 한 번에 더 많이 계산해 보통 빨라지지만, 표현할 수 있는 값이 거칠어져 정확도가 떨어질 수 있어요. 그래서 엣지 지표에 FP32·FP16·INT8별 품질 차이가 들어가요.

### 호환성 · Jetson · 측정 지표

- Engine은 기본적으로 OS · 플랫폼 · TensorRT 버전 · GPU 조건을 맞춰야 해요. 버전 호환·하드웨어 호환 옵션은 편하지만 성능을 일부 내줘요(trade-off).

- 같은 아키텍처라도 VRAM · 공유 메모리 · 커널 선택에 따라 실행 가능 여부가 달라질 수 있어요.

- 모델 · ONNX · Engine · 환경 · 입력 shape의 버전을 manifest(목록 파일)에 기록해요.

- Jetson: 카메라 → Decode·전처리(CPU/GPU) → TensorRT(JetPack 호환) → 앱. 전력 · 온도 · 공유 메모리 · 카메라 · FPS를 같이 봐요.

⚠️ Engine 파일은 "실행 가능한 산출물"이에요. 신뢰할 수 없는 출처의 Engine을 deserialize(불러오기)하지 않아요. 프로그램 파일을 아무 데서나 받아 실행하지 않는 것과 같은 이유예요.

| 성능 지표 | 자원 · 품질 지표 |
|---|---|
| End-to-end Latency: 입력부터 결과까지 | GPU · 시스템 메모리와 장치별 측정 기준 |
| Model Latency: 전처리·후처리를 뺀 추론만 | 전력 · 온도 · Throttling(과열로 속도를 낮추는 현상) 여부 |
| Throughput: 초당 처리 이미지·프레임 | FP32 · FP16 · INT8별 품질 차이 |
| Queue delay · Dropped frames · Batch wait time | 입력 해상도 · Batch · Stream 수와 정확도 |

🤔 왜 온도까지 재? — Jetson 같은 작은 장비는 오래 돌리면 뜨거워져서 스스로 속도를 낮춰요(Throttling). 처음 1분은 30 FPS였는데 10분 뒤엔 18 FPS가 될 수 있어서, 짧은 측정만으로 "현장에서 30 FPS"라고 말하면 안 돼요.

⚠️ 모든 OCR·VLM 애플리케이션이 Engine 하나로 바뀌는 건 아니에요. TensorRT는 지원되는 신경망 모델용이고, 기본은 "어떻게 배포할지 설계"하는 것까지예요.

## ⑧ OCR·VLM 미니 애플리케이션 설계

지금까지 배운 조각을 한 문서의 처리 기록으로 이어 붙여요. 모듈마다 책임을 나누고, 중간 산출물을 전부 남기는 게 핵심이에요.

[그림: ] 무엇을 보나: 왼쪽은 데이터가 흐르는 길(입력 → OCR/VLM 모듈 → Fusion·Validation → 결과 JSON), 아래 회색 띠는 흐르는 동안 남기는 산출물, 오른쪽은 강의의 폴더 구조와 파일별 역할이에요. 🤔 왜? 기능별로 파일을 나누면 OCR 엔진을 바꿔도 ocr.py만 고치면 되고, 결과가 이상할 때 어느 단계 산출물부터 틀렸는지 거슬러 올라갈 수 있어요.

### 한 문서의 처리 기록 — 6단계

- document_id와 원본 hash(파일 내용으로 만든 지문)를 생성해요. → 나중에 "같은 파일이었나?"를 확인

- 전처리 설정과 출력 이미지 경로를 기록해요.

- OCR text · box · confidence와 기준 이미지 · 크기를 저장해요.

- 질문과 VLM answer · evidence를 저장해요.

- Schema · 업무 규칙 · 근거 대조 결과를 저장해요.

- 최종 JSON과 사람이 확인할 review 목록을 만들어요.

🔧 모듈 인터페이스 @dataclass 명령어 사전에서 자세히 →

```
@dataclass
class OCRWord:   # text: str · confidence: float · bbox: tuple[int, int, int, int]
```

### OCR · VLM 결과를 합치는 규칙

| 상황 | 상태 |
|---|---|
| 값이 같고 원문 · 구조 · 업무 규칙 검증도 통과 | passed + 근거 기록 |
| 값이 다름 | 원문 · crop · 업무 규칙 대조 → warning · 검토 필요 |
| VLM의 근거 후보 | OCR 위치와 실제 이미지에 대조한 뒤 검토 |
| 두 결과 모두 없거나 근거 불일치 | failed · human review |

🤔 왜 최종 선택 규칙이 문서마다 달라? — 오류 비용이 달라서예요. 사내 메모 요약이 조금 틀리는 것과 송금 금액이 틀리는 건 무게가 완전히 달라요. 비싼 실수일수록 warning 기준을 엄격하게, 사람 검토를 넓게 잡아요.

| 최종 산출물 | 필수 내용 | 검증 |
|---|---|---|
| document.json | 필드 값 · 원문 · bbox · confidence · 상태 | Schema · 필수 필드 |
| ocr_words.csv | 단어 · 위치 · confidence · 페이지 | 원본 crop 대조 |
| vlm_answers.json | 질문 · 응답 · 근거 검토 | 근거 일치 · JSON 유효성 |
| evaluation.csv | 필드 정확도 · CER · 시간 · 측정 출처 | 동일 문서 · 동일 기준 |
| final_report.md | 오류 · 근거 검토 · 한계 · 배포 설계 | 재실행 근거 · artifact 경로 |

| 상황 (테스트 케이스) | 기대 결과 | 확인 기준 |
|---|---|---|
| 정상 스캔 문서 | OCR · 필드 · 근거 생성 | 필수 필드 · Box · 원문 |
| 기울어진 문서 | 보정 후 품질 개선 또는 warning | CER · Field accuracy |
| 표가 있는 문서 | 셀 구조 보존 또는 검토 표시 | 행 · 열 · 합계 관계 |
| VLM 근거 없는 답 | failed 또는 review | evidence 대조 |
| 개인정보 포함 | 마스킹 · 로컬 처리 정책 적용 | 원본 · 로그 · 전송 경로 |
| TensorRT Engine 불일치 | 실행 중단 · 환경 정보 출력 | 버전 · GPU · shape manifest |

🤔 왜 "실패 상황"까지 미리 정해? — 정상 문서만 시험하면 기울어진 문서·근거 없는 답이 왔을 때 프로그램이 조용히 틀린 값을 내보내요. 실패할 때 어떻게 실패해야 하는지(멈춤·경고·검토)를 정해 두는 게 안전한 프로그램이에요.

다음 수업(11/10)에서는 오늘 만든 document.json·OCR/VLM 결과를 FastAPI의 요청·응답(JSON API)으로 연결해요.

## 원본 자료

- 강의: Day16_20261109.pdf — 큰 그림 p2–18 · 전처리 p19–28 · OCR 실행·평가 p31–38 · 구조화·검증 p41–48 · VLM p51–58 · 근거·Hybrid·보안·메모리 p59–71 · 배포 p74–82 · 미니 애플리케이션 p85–92 · 정리 p98–99

- 요약: Day16_oneday요약.pdf — 하루 지도 · 교시별 핵심 · 파라미터를 바꿔 다시 실행한 실측 결과(Tesseract 5.5.2 · SmolVLM-256M-Instruct, 합성 영어 문서 기준이라 실제 한국어 문서·손글씨 성능을 뜻하지는 않아요)

- 공식 문서: OpenCV Thresholding · OpenCV Geometric Transformations (warpPerspective)

- 공식 문서: Tesseract — Improving the quality of the output (PSM) · pytesseract (image_to_string / image_to_data)

- 공식 문서: Transformers Multimodal chat templates · Image-text-to-text

- 공식 문서: Python re · Python json

- 공식 문서: NVIDIA TensorRT · DeepStream Gst-nvinfer · NVIDIA JetPack

## 🔧 Day16 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day16)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | pip install opencv-python numpy pandas pytesseract + Tesseract 프로그램 (VLM은 transformers) |
| ① 문서 전처리 | cv2.imread(), cv2.cvtColor(), cv2.threshold(), cv2.adaptiveThreshold(), cv2.getPerspectiveTransform(), cv2.warpPerspective(), cv2.perspectiveTransform()(좌표 되돌리기), cv2.resize() |
| ② OCR 실행 · Box · Confidence | pytesseract.image_to_data(lang=, config="--oem 3 --psm 6", output_type=Output.DATAFRAME) |
| ③ OCR 품질 평가 | levenshtein() → cer(), jiwer.cer(), 필드 정확도 |
| ④ 구조화 · JSON · 검증 | re.search(), re.sub(), json.dump(), validate_invoice() |
| ⑤ VLM 이미지 질의응답 | pipeline("image-text-to-text"), processor.apply_chat_template(), model.generate() |
| ⑥ 근거 검증 · Hybrid · 보안 | json.loads() + 필수 key·bbox 검사, if answer == raw → 채택 / None(보류) |
| ⑦ 엣지 배포 (TensorRT · Jetson · DeepStream) | torch.onnx.export(), trtexec --onnx= --saveEngine=, --loadEngine=, --minShapes/--optShapes/--maxShapes |
| ⑧ OCR·VLM 미니 애플리케이션 설계 | @dataclass OCRResult, VLMResult, 폴더별 역할 |

### 읽기 · Gray 변환 cv2.imread() · cv2.cvtColor()

```
import cv2

image = cv2.imread("invoice.png", cv2.IMREAD_COLOR)
if image is None:
    raise FileNotFoundError("invoice.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print("BGR shape:", image.shape)     # (H, W, 3)
print("Gray shape:", gray.shape)     # (H, W)
if not cv2.imwrite("gray.png", gray):
    raise OSError("gray.png 저장 실패")
```

▶ 실제로 돌려 보면 수업 청구서 data/doc03.png

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| image = cv2.imread("doc03.png", cv2.IMREAD_COLOR) | image.shape → (700, 1000, 3) | (H, W, 3) — 세로 700 · 가로 1000 · BGR |
| cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape | (700, 1000) | 채널 축이 사라짐 |
| cv2.imread("doc03.png", cv2.IMREAD_GRAYSCALE).shape | (700, 1000) | 처음부터 1채널로 읽기 |
| cv2.imread("없는파일.png") | None | 에러가 아니라 None → 바로 is None 검사 |
| cv2.imwrite("outputs/gray.png", gray) | True | 성공하면 True |
| cv2.imwrite("없는폴더/gray.png", gray) | False | 실패해도 예외 없이 False |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| filename (필수) | 읽을 이미지 경로 (Path는 str()로) | 경로가 틀려도 에러 없이 None을 돌려줘요 → 그래서 바로 is None 검사 |
| flags | IMREAD_COLOR(기본, BGR 3채널) · IMREAD_GRAYSCALE · IMREAD_UNCHANGED | GRAYSCALE면 처음부터 1채널로 읽어 cvtColor가 필요 없어요. UNCHANGED면 투명도(알파) 채널까지 4채널로 와요 |
| code (필수) | 변환 방향 상수. COLOR_BGR2GRAY, COLOR_BGR2RGB … | imread 결과는 BGR이라 RGB2GRAY를 쓰면 빨강·파랑 가중치가 바뀌어 밝기가 미묘하게 달라져요 |
| dstCn | 출력 채널 수 (보통 0 = 자동) | 건드릴 일이 거의 없어요 |

⚠️ 원본 파일은 덮어쓰지 않고 다른 경로에 저장해요. cv2.imwrite도 실패하면 예외 대신 False를 돌려주니 반환값을 확인해요.

### cv2.threshold() + Otsu · cv2.adaptiveThreshold()

```
otsu_t, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU
)
print("Otsu threshold:", otsu_t)        # Otsu가 고른 기준값

adaptive = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31, 11                              # blockSize, C
)
```

▶ 실제로 돌려 보면 data/shadow.png (왼쪽으로 갈수록 그림자로 어두운 청구서) — 검은색(0)이 된 칸 비율

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| gray 평균 밝기: 왼쪽 300px / 오른쪽 300px | 86 / 226 | 같은 흰 종이인데 왼쪽이 훨씬 어두움 |
| otsu_t (THRESH_BINARY \| THRESH_OTSU) | 153 | 0을 넣었지만 Otsu가 고른 기준이 돌아옴 |
| binary(Otsu) 검은 칸: 왼쪽 / 오른쪽 | 100.0% / 0.0% | 기준 하나라 그림자 쪽은 종이까지 통째로 검게 → 글자가 묻힘 |
| adaptive(31, 11) 검은 칸: 왼쪽 / 오른쪽 | 7.0% / 0.0% | 칸마다 주변 평균을 기준으로 → 그림자 속에서도 글자만 검게 |
| OTSU 없이 threshold(gray, 0, 255, BINARY) 검은 칸 | 0.0% | 0이 진짜 기준이 돼서 전부 흰색 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| thresh=0 + THRESH_OTSU | 기준값을 Otsu가 히스토그램으로 자동 추정 (0은 자리 채우기) | OTSU를 빼면 0이 진짜 기준이 돼서 거의 전부 흰색이 돼요. 반환값 otsu_t에 실제 고른 기준이 들어와요 |
| maxValue (255) | 조건을 만족한 픽셀에 넣을 값 | 1로 하면 0/1 이미지 — 눈으로는 거의 검게 보여요 |
| adaptiveMethod | MEAN_C(주변 단순 평균) · GAUSSIAN_C(가운데에 더 무게를 둔 평균) | GAUSSIAN이 가까운 픽셀을 더 믿어서 경계가 조금 더 매끄러워요 |
| thresholdType | BINARY(밝으면 흰색) · BINARY_INV(반대) | INV면 글자가 흰색, 배경이 검정 — 이후 처리(윤곽 찾기 등)가 "흰색 = 물체"를 기대할 때 써요 |
| blockSize (31) | 주변 영역 한 변 크기. 3 이상 홀수 | 작으면 글자 획 안쪽까지 "주변"으로 잡혀 굵은 글자 속이 비거나 종이 결이 점으로 남아요. 크면 그림자 변화를 못 따라가 Otsu처럼 돼요. 홀수인 이유: 가운데 픽셀이 딱 하나 있어야 해서 |
| C (11) | 주변 평균에서 빼는 상수 | 클수록 기준이 더 낮아져 "확실히 어두운 것"만 검게 → 잡점↓, 흐린 글자는 사라질 수 있어요. 작거나 음수면 종이 결까지 검게 찍혀요 |

🤔 왜 눈으로 보기 좋은 결과를 고르면 안 돼? — 사람에게 깨끗해 보이는 이미지와 OCR 오류율이 낮은 이미지는 항상 같지 않아요. 그래서 설정은 개발·검증 문서의 CER(Day16 ③ OCR 품질 평가)로 고르고, 최종 평가 문서는 그 선택 과정에서 빼 둬요. 최종 문서를 보며 고르면 그 문서에 맞춘 선택이 돼서 점수가 부풀어요.

### cv2.getPerspectiveTransform() · cv2.warpPerspective()

```
import numpy as np
import cv2

src = np.float32([[92, 40], [820, 75], [860, 1080], [55, 1050]])   # 왼위 → 오위 → 오아래 → 왼아래
dst = np.float32([[0, 0], [699, 0], [699, 999], [0, 999]])         # 같은 순서!
M = cv2.getPerspectiveTransform(src, dst)                           # 3×3 행렬
warped = cv2.warpPerspective(
    image, M, (700, 1000),             # (width, height)
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(255, 255, 255)
)
```

▶ 실제로 돌려 보면 data/perspective.png (800, 1100, 3) · lesson.json 모서리 [[90.0, 70.0], [1000.0, 25.0], [1050.0, 760.0], [30.0, 720.0]] → 정면 doc03.png와 비교(평균 픽셀 차, 작을수록 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| M = cv2.getPerspectiveTransform(src, dst) | [[1.273, 0.117, -122.768], [0.061, 1.236, -92.031], [0.0, 0.0, 1.0]] | 3×3 행렬 — 마지막 칸은 1로 고정 |
| warpPerspective(photo, M, (1000, 700)).shape | (700, 1000, 3) | dsize는 (W, H) → shape는 (700, 1000, 3) |
| 보정본 vs 정면 원본 doc03.png 평균 차 | 1.36 | 거의 같음 — 펴기 성공 |
| 모서리 순서를 한 칸 밀었을 때 평균 차 | 11.92 | 문서가 90° 돌아가며 찌그러짐 |
| 2번 ↔ 4번 모서리를 맞바꿨을 때 평균 차 | 11.56 | 대각선으로 뒤집힌(가로·세로가 바뀐) 문서 |
| dsize를 (700, 1000)으로 바꿔 넣으면 .shape | (1000, 700, 3) | 세로로 긴 캔버스 — dst(999, 699)와 안 맞음 |
| 그 캔버스의 아래 300줄 중 흰 칸 비율 | 79.7% | 문서는 위 700줄에만(오른쪽 300px는 잘림), 아래엔 원본 사진의 문서 밖 배경이 딸려 옴 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| src / dst | 실제 문서 네 모서리 / 펼친 뒤 네 모서리 (float32, 4×2) | 순서가 어긋나면 문서가 뒤집히거나 꼬여요. 모서리를 문서 밖에 찍으면 배경이 딸려 오고, 안쪽에 찍으면 가장자리 글자가 잘려요 |
| dsize (700, 1000) | 출력 크기 (width, height) | 배열 shape는 (H, W)인데 여기는 (W, H)! 바꿔 넣으면 가로로 납작한 문서가 돼요. dst 좌표(699, 999)와도 맞춰야 해요 |
| flags | 새 픽셀 값을 주변에서 어떻게 계산할지 (INTER_LINEAR = 주변 4칸 가중 평균) | INTER_NEAREST는 빠르지만 글자 가장자리가 계단처럼 깨져요 |
| borderMode / borderValue | 원본 밖을 가리키는 픽셀을 채우는 방법 / 그 색 | 흰색(255)이 아니면 가장자리에 검은 띠가 생겨 OCR이 글자로 오해할 수 있어요 |

🤔 왜 점이 꼭 4개야? — 원근 변환 행렬 M에는 정해야 할 숫자가 8개 있어요(9칸 중 마지막은 1로 고정). 점 하나가 x·y 식 두 개를 주니까, 4쌍이면 딱 8개 식이 생겨 M이 하나로 정해져요. 3개면 모자라고, 5개 이상이면 "가장 잘 맞는" M을 따로 구해야 해요.

⚠️ 좌표계 주의 — 보정본에서 나온 OCR Box는 보정본 좌표예요. 원본 사진 위에 그리려면 M의 역변환으로 되돌려야 해요. 그래서 JSON에 어떤 이미지 기준 좌표인지(coordinate_space)와 이미지 크기를 같이 적어요.

### 보정본 좌표 → 원본 사진 좌표 cv2.perspectiveTransform() · 작은 사진 확대 cv2.resize()

```
box = np.float32([[[40, 370]]])                      # 보정본에서 찾은 TOTAL: 의 왼쪽 위 점, shape (1, 1, 2)
back = cv2.perspectiveTransform(box, np.linalg.inv(M))
print(back)                                         # 강의 요약본의 실제 사진(모서리 좌표가 다름)에선 [[[64.7 349.6]]] → 사진에서 직접 읽은 (65, 350)과 1px 안

small = cv2.resize(warped, None, fx=2, fy=2,        # 가로·세로 2배
                   interpolation=cv2.INTER_CUBIC)
```

▶ 실제로 돌려 보면 getPerspectiveTransform 항목의 M(perspective.png) · doc03 OCR 기록에서 TOTAL: 단어의 왼쪽 위 점 (56, 501) · data/lowres.png

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| perspectiveTransform([[[56, 501]]], inv(M)) | [[[97.4, 521.9]]] | 보정본 점 → 비스듬한 원본 사진의 점 |
| perspectiveTransform(back, M) | [[[56.0, 501.0]]] | 다시 M으로 보내면 원래 점 — 역행렬이 맞다는 확인 |
| perspectiveTransform(점, M) (inv 없이) | [[[6.7, 480.0]]] | 한 번 더 '펴는' 셈이라 엉뚱한 곳 |
| perspectiveTransform(np.float32([[56, 501]]), inv(M)) — shape (1, 2) | error: (-215:Assertion failed) scn + 1 == m.cols in function 'cv::perspectiveTransform' | (N, 1, 2)가 아니면 OpenCV 에러 |
| 이 사전 코드의 src·dst로 [[[40, 370]]] 되돌리기 | [[[122.7, 392.5]]] | 코드 주석의 [[[64.7 349.6]]]과 달라요 — 주석 값은 이 src 숫자로는 재현되지 않아요 |
| cv2.resize(lowres, None, fx=2, fy=2, interpolation=INTER_CUBIC).shape | (210, 300, 3) → (420, 600, 3) | 가로·세로 2배 |

| 코드 | 뜻 | 바꾸면? |
|---|---|---|
| np.linalg.inv(M) | M(사진 → 보정본)의 역행렬 = 보정본 → 사진 | M을 그대로 쓰면 한 번 더 "펴는" 셈이라 엉뚱한 곳을 가리켜요 |
| 점 shape (N, 1, 2) | perspectiveTransform이 기대하는 모양 | (N, 2)로 넣으면 에러 — 이미지가 아니라 점을 옮기는 함수라 따로 있어요(warpPerspective는 이미지용) |
| fx=2, fy=2 | 가로·세로 배율 (dsize=None일 때) | 요약본 실측: 0.4배로 줄인 흐린 사진이 CER 0.066 → 0.008, 필드 2/3 → 3/3 |
| INTER_CUBIC | 주변 16칸으로 부드럽게 새 픽셀 계산 | INTER_NEAREST면 계단이 그대로 커져 확대 효과가 줄어요 |

🤔 왜 확대만 했는데 더 잘 읽어? — 새 정보가 생긴 건 아니에요. Tesseract 공식 문서(ImproveQuality)에 따르면 글자가 어느 정도 커야(대문자 높이 30px 안팎) 가장 잘 읽혀서, 너무 작은 글자는 획이 1~2픽셀로 뭉개져 모양을 못 알아봐요. 키워 주면 엔진이 익숙한 크기로 보게 되는 거예요.

⚠️ 흔한 실수: 모서리 순서 — 요약본에서 corners의 시작점을 오른쪽 위로 바꿨더니(값은 같고 순서만 한 칸 밀림) 문서가 90도 돌아가 찌그러져 CER 0.008 → 1.254, 필드 3/3 → 0/3이 됐어요. 두 점만 맞바꾸면 사각형이 꼬여 화면 대부분이 검게 나왔고요(CER 1.0). Day16 본문 원근 보정 체험의 "2번 ↔ 4번" 버튼이 바로 그 상황이에요. 🤔 CER이 1을 넘을 수 있어? — 네. 분모는 정답 글자 수인데, 엉뚱한 글자를 잔뜩 끼워 넣으면(I) 오류 수가 정답 길이보다 많아질 수 있어요.

### 단어별 결과 받기 pytesseract.image_to_data()

```
import pytesseract
from pytesseract import Output

ocr = pytesseract.image_to_data(
    binary, lang="kor+eng", config="--oem 3 --psm 6",
    output_type=Output.DATAFRAME,
    pandas_config={"keep_default_na": False,
                   "dtype": {"text": str}},
)
words = ocr[
    ocr.level.eq(5) & ocr.text.str.strip().ne("")     # 단어 행 + 빈 글자 제외
].copy()
review = words[words.conf.lt(50)]                       # 점수 낮은 단어 = 검토 목록
columns = ["text", "conf", "left", "top", "width", "height"]
print(words[columns])
```

▶ 실제로 돌려 보면 이 PC엔 Tesseract가 없어서, 수업 lesson.json에 기록된 실제 Tesseract 5.5.0(eng, psm 6) 결과로 표를 만들고 거르기만 실행 — lowres_raw.png(작은 사진) 기준

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| pytesseract.image_to_data(gray, ...) (이 PC) | TesseractNotFoundError: tesseract is not installed or it's not in your PATH. See README file for more information. | 엔진 프로그램이 따로 설치돼 있어야 해요 |
| len(words) | 27 | 단어(level 5) 행 수 |
| words.iloc[0][['text', 'conf', 'left', 'top', 'width', 'height']] | ['INVOICE', 17.6, 17, 14, 42, 8] | 맞게 읽은 INVOICE인데 conf가 낮음 → 점수 ≠ 정답 여부 |
| review = words[words.conf.lt(50)] → text 목록 | ['INVOICE', 'DoCe3', 'ATE:', '{TEN', 'AMOUNT', 'RANE', 'CURRENCY:', 'KR'] | 8개가 검토 목록으로 |
| [left, top, left+width, top+height] (첫 단어) | [17, 14, 59, 22] | xyxy로 바꾸기 |
| 정면 깨끗한 doc03.png 기록: conf 최소값 · 50 미만 개수 | 52.5 · 0 | 같은 엔진이라도 사진 품질에 따라 점수 분포가 달라짐 |

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| image (필수) | PIL Image · RGB/Gray 배열 · 파일 경로 | OpenCV 컬러 배열(BGR)을 그대로 주면 색 순서가 달라요 → Gray·이진화본이나 COLOR_BGR2RGB로 바꿔서 |
| lang | 언어 데이터 조합 "kor", "eng", "kor+eng" | 설치 안 된 언어면 에러. eng만 주면 한글을 영어 글자로 억지로 읽어 엉망이 돼요. 언어를 많이 붙일수록 후보가 늘어 느려지고 헷갈릴 수 있어요 |
| config="--oem 3" | 인식 엔진 모드 (3 = 사용 가능한 기본 엔진) | 보통 3 그대로 |
| config="--psm 6" | 페이지 모양 가정 (Day16 PSM 그림) | 한 줄 Crop이면 7, 단어 하나면 8이 더 맞을 수 있어요 |
| output_type | STRING(기본, TSV 글자) · DICT · DATAFRAME | DATAFRAME이면 pandas로 바로 거르고 저장. DICT면 리스트 묶음 |
| timeout | 초 단위 제한 (0 = 무제한) | 설정하면 큰 이미지에서 멈춘 듯 오래 걸리는 걸 끊을 수 있어요 |
| pandas_config | DataFrame 만들 때 넘기는 read_csv 옵션 | 아래 🤔 참고 — 빼면 일부 글자가 사라지거나 바뀌어요 |

🤔 왜 keep_default_na=False, dtype={"text": str}? — pandas는 표를 읽을 때 "NA", "null" 같은 글자를 "값 없음(NaN)"으로, "0012"를 숫자 12로 알아서 바꿔요. 문서에 진짜로 "NA"라고 적혀 있거나 문서번호가 0으로 시작하면 원문이 망가지죠. 이 두 옵션으로 OCR이 읽은 글자를 글자 그대로 보존해요.

🤔 왜 level == 5? — image_to_data는 페이지(1) · 블록(2) · 문단(3) · 줄(4) · 단어(5) 행을 한 표에 섞어서 줘요. 위쪽 단계 행은 글자 칸이 비어 있고 위치만 있어요. 단어만 쓰려면 5번 행만 골라야 해요.

| Confidence로 할 수 있는 것 | 주의할 것 |
|---|---|
| 낮은 점수 단어를 검토 목록에 넣기 | 엔진마다 점수의 크기·의미가 달라요 |
| 같은 필드 후보 중 상대적으로 높은 것 고르기 | 짧은 숫자·기호는 낮아도 맞을 수 있어요 |
| 문서 유형별 점수 분포 지켜보기 | 높은 점수가 문맥상 맞다는 보장은 없어요 |
| box와 원본 crop을 같이 저장해 사람이 검토 | 기준 하나로 모든 문서를 자동 삭제하지 않기 |

⚠️ conf 90 ≠ "90% 확률로 맞음" — 엔진 내부의 신뢰 점수일 뿐, 문맥상 정답률과 같지 않아요. 그리고 50은 설명용 기준이에요. 원본 OCR 결과는 지우지 말고 보관해요.

🔬 PSM을 바꿔 보면 — 같은 청구서 전체를 --psm 7(한 줄 가정)로 읽으면 CER 0.008 → 1.0(빈 문자열), --psm 8(한 단어)은 "a" 한 글자만 나왔어요. 반대로 합계 한 줄만 잘라서 PSM 7로 넣으면 TOTAL: 33000을 정확히 읽어요. 🤔 왜? PSM은 "이 그림이 어떻게 생겼다"는 약속이라, 7줄짜리 문서에 "한 줄이야"라고 하면 엔진이 여러 줄을 한 줄로 억지로 맞추다 포기해요. 그래서 입력 범위가 다른 결과끼리(페이지 전체 vs 한 줄 crop) CER로 순위를 매기면 안 돼요.

💡 객체탐지의 box와 OCR의 word box는 둘 다 "위치"예요. 차이는 OCR box에는 문자열·줄·블록 정보가 함께 붙는다는 것.

### CER 계산 함수

```
def levenshtein(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))           # 빈 문자열 → b의 앞 j글자: j번 삽입
    for i, ca in enumerate(a, start=1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,                 # 삭제 (D)
                dp[j - 1] + 1,             # 삽입 (I)
                prev + (ca != cb),         # 치환 (S) — 같으면 +0
            )
            prev = old
    return dp[-1]

def cer(reference: str, prediction: str) -> float:
    if not reference:
        raise ValueError("빈 정답 문서는 별도 평가합니다")
    errors = levenshtein(reference, prediction)
    return errors / len(reference)
```

▶ 실제로 돌려 보면 강의 예시 문자열 + lesson.json의 정답 글자 vs 기록된 실제 Tesseract 결과 (doc03 계열)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| levenshtein("1,234,500원", "1,234,5O0원") | 1 | 0 → O 치환(S) 1번 |
| cer("1,234,500원", "1,234,5O0원") | 0.100 | 1 ÷ 정답 10글자 |
| levenshtein("33000", "3300") · ("33000", "330000") | 1 · 1 | 빠뜨림(D) 1 · 끼워 넣음(I) 1 |
| cer(정답, doc03 OCR) — 정면 깨끗한 문서 | 0.006 | DOC03 → DOCO3 한 글자만 틀림 |
| cer(정답, rotated_raw OCR) | 0.025 | 기울어진 사진 그대로 |
| cer(정답, lowres_raw / lowres_corrected OCR) | 0.063 / 0.019 | 작은 사진 → 보정하면 줄어듦 |
| cer(정답, shadow_raw OCR) — OCR 결과가 빈 문자열 | 1.000 | 전부 빠뜨림 = 1.0 |
| cer("", "abc") | ValueError: 빈 정답 문서는 별도 평가합니다 | 빈 정답은 0으로 나누지 않고 따로 집계 |

| 코드 · 규칙 | 뜻 | 바꾸면? |
|---|---|---|
| dp 한 줄 | 표 전체 대신 직전 줄만 기억하며 최소 횟수를 채움 | 결과는 같고 메모리만 아껴요 |
| prev + (ca != cb) | 두 글자가 같으면 비용 0, 다르면 1(치환) | 치환 비용을 2로 하면 "지우고 넣기"와 같아져 다른 지표가 돼요 |
| len(reference) | 분모 = 정답 글자 수 | OCR 결과 길이로 나누면, 글자를 잔뜩 빠뜨린 결과가 오히려 점수가 좋아 보여요 |
| 빈 정답 → ValueError | 0으로 나누기 방지 + 따로 집계 | 조용히 0점 처리하면 "글자 없는 페이지에 글자를 지어낸" 오류가 숨어요 |
| 정규화 규칙 | 공백·대소문자·기호를 비교 전에 어떻게 다룰지 먼저 고정 | 규칙이 바뀌면 같은 결과도 CER이 달라져요(Day16 본문 CER 체험의 "공백 지우기" 체크박스) |

🤔 왜 문서별 평균과 micro CER을 구분해? — 문서별 CER을 평균 내면 10글자짜리 영수증과 2,000글자짜리 계약서가 같은 무게예요. micro CER은 전체 오류 수 ÷ 전체 정답 글자 수라서 글자 하나하나가 같은 무게예요. 짧은 문서 몇 장이 크게 틀리면 두 값이 크게 달라지니, 어떤 걸 보고했는지 적어요.

### 라이브러리로 한 줄 jiwer.cer()

```
import jiwer                                   # pip install jiwer

truth = " ".join(lines)                         # 정답 글자 (줄을 공백 한 칸으로 이음)
read = " ".join(text.split())                   # OCR 글자 — 줄바꿈·여러 공백을 한 칸으로
score = jiwer.cer(truth, read)                  # (정답, 예측) 순서!
print(round(score, 3))
```

▶ 실제로 돌려 보면 jiwer가 이 PC에 없어 같은 정의(편집 거리 ÷ 정답 글자 수)인 CER 계산 함수의 cer()로 계산 — doc01 psm 3 기록 · lowres_raw 기록

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| import jiwer | ModuleNotFoundError: No module named 'jiwer' | pip install jiwer 필요 |
| cer(정답 원문, psm 3 OCR 원문) — 정리 없이 | 0.0132 | OCR이 넣은 빈 줄(\n) 하나도 오류로 셈 |
| cer(" ".join(lines), " ".join(text.split())) | 0.0066 | 양쪽을 같은 규칙으로 정리 → 진짜 글자 오류만 남음 |
| cer(truth, read) — lowres_raw | 0.0633 | (정답, 예측) 순서 |
| cer(read, truth) — 순서를 바꾸면 | 0.0649 | 분모가 OCR 글자 수로 바뀌어 값이 달라짐 |

| 인자 · 규칙 | 뜻 | 바꾸면? |
|---|---|---|
| 첫 번째 인자 reference | 정답 — 분모가 되는 쪽 | 순서를 바꾸면 분모가 OCR 결과 길이가 돼서 값이 달라져요 |
| " ".join(text.split()) | 공백·줄바꿈을 같은 규칙으로 정리 | 한쪽만 정리하면 "줄바꿈 vs 공백" 차이가 전부 오류로 세져요 |
| jiwer.wer() | 같은 방식, 단위만 단어 | 숫자 한 글자 틀려도 단어 하나가 통째로 틀린 걸로 세요 |

### 정규식으로 날짜 · 금액 찾기 re.search()

```
import re
from datetime import date

text = "발행일 2026-11-09   합계 1,234,500원"
date_pat = r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})"
money_pat = (r"합계\s*"
             r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*원")

dm = re.search(date_pat, text)
am = re.search(money_pat, text)
issued_at, total = None, None
if dm:
    try:
        issued_at = date(*map(int, dm.groups())).isoformat()   # 진짜 있는 날짜인지 검사
    except ValueError:
        pass                                                   # 2026-13-45 같은 건 None으로
if am:
    total = int(am.group(1).replace(",", ""))
print(issued_at, total)        # 2026-11-09 1234500
```

▶ 실제로 돌려 보면 사전 코드의 text 문자열 + 수업 doc03의 실제 Tesseract 기록 문자열

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| re.search(date_pat, text).groups() | ('2026', '11', '09') | 괄호 세 개 = 그룹 세 개 (아직 글자) |
| print(issued_at, total) | 2026-11-09 1234500 | 날짜 검사 통과 · 쉼표 뗀 정수 |
| 같은 코드에 "발행일 2026-13-45 합계 5000원" | (None, 5000) | 모양은 날짜라 패턴엔 걸리지만 date()가 거부 → None |
| re.search(money_pat, "합계: 미정") | None | 못 찾으면 None → .group() 전에 검사 |
| re.sub(r"[^0-9]", "", "1,234,500원") | '1234500' | 숫자 아닌 글자를 지운 새 문자열 |
| re.search(r"^TOTAL: (\d+)$", ocr_text, re.MULTILINE).group(1) | '33000' | 라벨 + 줄 단위 → 정확 |
| re.search(r"^TOTAL: (\d+)$", ocr_text) (MULTILINE 없이) | None | ^ $가 글 전체의 처음·끝이라 못 찾음 |
| re.search(r"(\d+)", ocr_text).group(1) | '3' | 첫 숫자 = 맨 윗줄 DOCO3의 3 — 라벨이 없으면 엉뚱한 값 |
| re.findall(r"^[A-Z]+ \d+ \d+ (\d+)$", ocr_text, re.MULTILINE) | ['12000', '9000', '12000'] | 품목 합 33000 = 합계 → total_mismatch 없음 |

| 패턴 조각 | 뜻 | 바꾸면? |
|---|---|---|
| r"..." | raw string — \d의 역슬래시를 Python이 먼저 건드리지 않게 | r을 빼면 \b 같은 건 다른 뜻(백스페이스)이 돼 버려요 |
| (20\d{2}) | 20으로 시작하는 4자리 연도 — 괄호는 "이 부분 따로 꺼내기(그룹)" | \d{4}로 넓히면 전화번호 앞자리도 연도로 잡힐 수 있어요 |
| [-./] | 구분자로 -, ., / 중 하나 | 문서마다 표기가 달라서 여러 개를 허용해요. 2026. 11. 9처럼 공백이 끼면 \s*도 필요 |
| [0-9]{1,3}(?:,[0-9]{3})+ | 쉼표 세 자리 묶음 숫자 ((?:...)는 꺼내지 않는 묶음) | \|[0-9]+가 쉼표 없는 숫자도 받아요 |
| 합계\s* … \s*원 | "합계" 라벨 뒤의 금액만 | 라벨을 빼면 공급가액·세액 숫자까지 후보가 돼요 |

| 함수 · 인자 | 뜻 | 바꾸면? |
|---|---|---|
| re.search(pattern, string, flags=0) | 패턴의 첫 후보를 찾아 Match 객체, 없으면 None | 모든 후보가 필요하면 re.findall/finditer. None 검사를 빼면 .group()에서 에러 |
| re.sub(pattern, repl, string, count=0, flags=0) | 패턴에 맞는 부분을 바꾼 새 문자열 (예: re.sub(r"[^0-9]", "", "1,234,500원") → "1234500") | count=1이면 첫 번째만 바꿔요. 원래 문자열은 안 바뀌어요 |
| flags | re.IGNORECASE(대소문자 무시), re.MULTILINE(줄마다 ^ $) | "Total"/"TOTAL"을 한 패턴으로 잡을 때 IGNORECASE |

🤔 왜 패턴으로 찾고 date()로 또 검사해? — 정규식은 "모양"만 봐요. 2026-13-45도 모양은 날짜예요. date(2026, 13, 45)는 ValueError를 내니까, 패턴은 후보 찾기, date는 진짜인지 확인으로 역할을 나눠요. 그리고 실패하면 아무 값이나 넣지 말고 None으로 남겨요.

🔬 정규식을 느슨하게 바꾸면 (oneday 요약본 실측, 실제 OCR 문자열) · re.search(r"^TOTAL: (\d+)$", text, re.MULTILINE) → "33000" ✅ · re.search(r"(\d+)", text) (첫 숫자) → "3" ❌ — 맨 윗줄 DOCO3의 3을 금액으로 집었어요. · re.MULTILINE만 빼도 → None → match.group(1)에서 프로그램이 멈춰요. 🤔 왜? ^·$는 기본이 "글 전체의 처음·끝"이에요. 여러 줄 글에서 "줄마다 처음·끝"으로 보려면 MULTILINE이 필요해요. 그리고 라벨(TOTAL:)을 빼면 "숫자처럼 생긴 아무거나"가 후보가 돼요.

```
amounts = re.findall(r"^[A-Z]+ \d+ \d+ (\d+)$", text, re.MULTILINE)   # 품목 줄의 마지막 숫자들
if sum(int(a) for a in amounts) != total["value"]:
    reasons.append("total_mismatch")
```

### 필드와 근거를 JSON으로 json.dump()

```
import json

record = {
    "document_id": "invoice_001", "page": 1,
    "image_size": [1000, 1200],
    "coordinate_space": "processed_px_xyxy",
    "fields": {
        "issued_at": {
            "value": "2026-11-09", "raw": "2026. 11. 9",
            "confidence": 92.4, "bbox": [118, 205, 310, 244]},
        "supplier": {
            "value": "AI Lab", "raw": "AI Lab",
            "confidence": 93.1, "bbox": [118, 260, 310, 299]},
        "total_amount": {
            "value": 1234500, "raw": "1,234,500원",
            "confidence": 88.6, "bbox": [530, 820, 790, 865]},
    },
}
with open("invoice_001.json", "w", encoding="utf-8") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
```

▶ 실제로 돌려 보면 사전 코드의 record를 그대로 저장

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| json.dump(record, f, ensure_ascii=False, indent=2) 후 파일 | 44줄 · 734 bytes | 들여쓰기로 한 key가 한 줄씩 |
| 파일 안 total_amount 의 raw 줄 | "raw": "1,234,500원", | 한글 그대로 |
| json.dumps(record["fields"]["total_amount"]["raw"]) (ensure_ascii 기본 True) | "1,234,500\uc6d0" | 원 → \uc6d0 — 값은 같지만 사람이 못 읽음 |
| json.dumps(record) 길이: indent 없음 / indent=2 | 439 / 689 | 한 줄로 붙으면 작지만 읽기 힘듦 |
| json.load(open(...)) == record | True | 다시 읽으면 원래 dict와 같음 |

| 키 · 인자 | 뜻 | 바꾸면? |
|---|---|---|
| value / raw | 정규화 값 / OCR 원문 | raw를 빼면 오류 추적 불가 (아래 🤔) |
| coordinate_space, image_size | bbox가 어떤 이미지(여기선 1000×1200 처리 이미지)의 xyxy 픽셀인지 | 안 적으면 700×1000 보정본 좌표인지 원본 좌표인지 몰라 엉뚱한 곳을 그려요 |
| encoding="utf-8" | 파일을 UTF-8로 저장 | Windows 기본 인코딩으로 저장되면 다른 PC·서버에서 한글이 깨질 수 있어요 |
| ensure_ascii=False | 한글을 그대로 저장 | True(기본)면 "\uc6d0"처럼 저장돼 사람이 못 읽어요(값은 같음) |
| indent=2 | 들여쓰기로 보기 좋게 | 빼면 한 줄로 붙어 파일은 작지만 읽기 힘들어요 |

🤔 왜 정규화한 값만 저장하면 문제야? — 오류가 났을 때 "원래 뭐라고 적혀 있었는지, 어디 있었는지"를 확인할 수 없어요. 실제 기록에는 이미지 경로·적용한 변환·schema version(이 JSON 모양의 버전)도 같이 남겨요.

### 업무 규칙으로 검증 validate_invoice()

| 규칙 | 예 |
|---|---|
| 필수 필드 | document_id · issued_at · supplier · total_amount |
| 형식 | 날짜는 ISO 8601(2026-11-09), 금액은 정수, 통화는 별도 필드 |
| 관계 | 공급가액 + 세액 = 합계, 수량 × 단가 ≈ 금액 |
| 범위 | 날짜·금액·수량이 허용 범위 안인지 |
| 상태 | passed · warning · failed + 사람이 확인할 이유 |

```
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
```

▶ 실제로 돌려 보면 json.dump 항목의 record와 일부러 망가뜨린 사본들

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| validate_invoice(record) | [] | 빈 목록 = 통과 |
| total_amount value = 0 | [] | 0원은 진짜 값 — 누락이 아님 |
| total_amount value = "1234500" (글자) | ['합계는 0 이상의 정수여야 합니다'] | 정수가 아님 |
| total_amount value = True | ['합계는 0 이상의 정수여야 합니다'] | type() is int 검사라 bool도 걸러짐 |
| supplier 필드 삭제 | ['필수 필드 누락: supplier'] | 필수 필드 검사 |
| {"fields": "x"} | ['document_id 누락', 'fields는 객체여야 합니다'] | 문제를 모아서 한꺼번에 |
| validate_invoice([record]) | ['입력은 객체여야 합니다'] | dict가 아니면 바로 거절 (.get 에러 없이) |

| 코드 | 뜻 | 바꾸면? |
|---|---|---|
| isinstance(doc, dict) 먼저 | 모양을 확인한 뒤에 필드에 접근 | 빼면 리스트·None이 들어올 때 .get에서 프로그램이 죽어요 |
| value is None or value == "" | "없음"만 누락으로 봄 | if not value:로 쓰면 금액 0도 누락으로 처리돼요 (0은 거짓으로 취급되니까) |
| type(value) is not int | 정확히 int만 금액으로 인정 | isinstance(value, int)로 쓰면 True도 통과해요 — Python에서 bool은 int의 한 종류(True == 1)라서 |
| 오류 목록 반환 | 문제를 전부 모아서 돌려줌 | 첫 오류에서 멈추면 사람이 여러 번 고쳐야 해요 |

⚠️ 검증에 실패했다고 0이나 빈 문자열로 자동 치환하면 안 돼요. 진짜 0원과 "못 읽음"이 구분되지 않아요. warning·failed 상태와 이유를 남기고 사람이 보게 해요. 날짜·문자열·좌표까지 전부 검사하려면 이 함수를 Schema(데이터 모양 규칙)로 확장해요.

### 간단 버전 pipeline("image-text-to-text")

```
from transformers import pipeline

MODEL_ID = "YOUR_VLM_MODEL_ID"        # 준비된 모델 ID나 로컬 경로로 교체
pipe = pipeline(
    "image-text-to-text", model=MODEL_ID,
    device_map="auto", dtype="auto",
)
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "path": "invoice.png"},
        {"type": "text", "text": "합계 금액과 근거를 말해줘."},
    ],
}]
result = pipe(
    text=messages, max_new_tokens=128,
    do_sample=False, return_full_text=False,
)
print(result[0]["generated_text"])
```

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| task (필수) | 할 일 이름 "image-text-to-text" | 다른 task면 다른 전처리·후처리가 붙어 이미지+대화 입력을 못 받아요 |
| model | 모델 ID · 로컬 경로 · 이미 불러온 모델 객체 | 지원하지 않는 모델이면 불러오기 단계에서 실패 |
| device_map="auto" | GPU·CPU에 모델을 알아서 나눠 올림 | 단일 장치를 직접 고르려면 device=0(GPU 0번) 등. 메모리가 모자라면 일부가 CPU로 가 느려져요 |
| dtype="auto" | 모델 숫자 형식 (float16, bfloat16 …) | 16비트는 메모리가 절반이지만 장치가 지원해야 해요 |
| processor | 멀티모달 입력 전처리 객체 | 보통 모델에서 자동으로 불러와서 안 줘도 돼요 |
| max_new_tokens | 새로 만들 답의 최대 길이(토큰) | 너무 작으면 JSON이 중간에 잘려 파싱 실패. 크면 느려지고 쓸데없는 말이 붙을 수 있어요 |
| return_full_text=False | 새 답 부분만 돌려받기 | True면 질문(입력)까지 붙어 나와서 답만 따로 잘라야 해요 |

🔬 max_new_tokens를 3으로 줄이면 — oneday 요약본에서 SmolVLM-256M에 "What is the total amount?"를 물었을 때 128이면 33000, 3이면 33이 나왔어요. 에러 없이 잘린 숫자가 나온 거예요. 🤔 왜 에러가 안 나? 모델은 토큰을 한 조각씩 만들다 개수 제한에 닿으면 그냥 멈출 뿐이에요. "답이 덜 끝났다"는 신호가 따로 없어서, 잘린 33도 멀쩡한 답처럼 보여요. 그래서 다음 단계(Day16 ⑥ 근거 검증)에서 OCR 원문과 대조하면 33 ≠ 33000으로 걸러져 보류돼요.

💡 do_sample=False — 매번 가장 확률 높은 토큰만 고르는 방식(greedy)이라 같은 입력이면 같은 답이 나와요. 결과를 기록하고 다시 실행해 대조하려면 이 설정이 필요해요.

### 직접 버전 AutoProcessor · apply_chat_template() · generate()

```
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, device_map="auto", dtype="auto",
)
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)
outputs = model.generate(
    **inputs, max_new_tokens=128, do_sample=False,
)
ids = outputs[0]
if not model.config.is_encoder_decoder:
    ids = ids[inputs["input_ids"].shape[1]:]       # 입력 길이만큼 앞부분 잘라 내기
answer = processor.decode(ids, skip_special_tokens=True)
print(answer)
```

| apply_chat_template 인자 | 뜻 | 바꾸면? |
|---|---|---|
| messages (필수) | role과 content 목록으로 된 대화 | content를 문자열 하나로 주면 이미지가 빠져요 |
| add_generation_prompt=True | "이제 assistant가 답할 차례" 표시를 끝에 붙임 | False면 모델이 사용자 질문을 이어 쓰는 등 엉뚱하게 시작할 수 있어요 |
| tokenize | False = 포맷된 문자열만 · True = 토큰화까지 | False면 결과를 다시 processor에 넣어야 해요 |
| return_dict=True | input_ids · pixel_values 등을 dict로 | 그래야 **inputs로 generate에 바로 넘길 수 있어요 |
| return_tensors="pt" | PyTorch Tensor로 | 빼면 파이썬 리스트라 모델에 바로 못 넣어요 |
| num_frames | (비디오 지원 모델) 샘플링할 프레임 수 | Day16 ⑥ 입력 크기와 메모리의 비디오 VLM 참고 |

| generate 인자 | 뜻 | 바꾸면? |
|---|---|---|
| **inputs (필수) | Processor가 만든 입력 묶음 |  |
| max_new_tokens | 새로 생성할 최대 토큰 수 | 위 pipeline과 같음 |
| do_sample=False | 매번 가장 그럴듯한 다음 토큰을 고름(결정적) | True면 확률대로 뽑아 같은 질문에도 답이 달라질 수 있어요 → 문서 추출에선 재현성이 떨어져요 |
| temperature | 샘플링 분포를 평평하게/뾰족하게 | do_sample=True일 때만 의미가 있어요. 높을수록 엉뚱한 토큰도 뽑힘 |
| num_beams | Beam search 폭 — 후보 문장 여러 개를 동시에 키워 봄 | 늘리면 더 좋은 문장을 찾을 수도 있지만 계산이 그만큼 늘어요 |
| use_cache=True | 앞에서 계산한 Key/Value를 재사용(KV cache) | 끄면 토큰마다 처음부터 다시 계산해 훨씬 느려요 |

🤔 왜 입력 길이만큼 잘라 내? — 언어 모델만 있는(Decoder-only) 모델의 generate는 질문 토큰 + 새 답 토큰을 이어 붙여 돌려줘요. 앞의 input_ids 길이만큼 잘라야 답만 남아요. 인코더-디코더 구조는 처음부터 답만 돌려줘서 자를 필요가 없어요(is_encoder_decoder로 구분).

🤔 왜 Chat template은 Tokenizer가 아니라 Processor에? — 멀티모달 대화에는 "여기에 이미지가 들어간다"는 특수 토큰과 이미지 전처리가 함께 필요해요. 글자만 다루는 Tokenizer로는 이미지 자리를 만들 수 없어서, 둘을 다 가진 Processor가 맡아요.

⚠️ 모델 구조 · 전처리 · 생성 조건(max_new_tokens, do_sample 등)과 모델 ID를 실행 기록에 남겨요. 같은 질문이라도 조건이 바뀌면 답이 달라져요.

### VLM 응답 JSON 최소 검사 json.loads()

```
import json

raw = ('{"answer":"1,234,500원",'
       '"evidence":"1,234,500원",'
       '"page":1,"bbox":[530,820,790,865]}')
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
print("누락:", sorted(missing))
print("설명용 이미지의 Box 범위:", valid_box)
```

▶ 실제로 돌려 보면 사전 코드의 raw와 망가진 응답들 → (누락 key, Box 범위 OK?)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 사전 코드의 raw | ([], True) | 구조는 통과 |
| '{"answer":"1,234,500원","evid' (잘린 답) | (['answer', 'bbox', 'evidence', 'page'], False) | JSONDecodeError → {} → 전부 누락 |
| '합계는 1,234,500원입니다' (말로 답함) | (['answer', 'bbox', 'evidence', 'page'], False) | JSON이 아님 |
| '[530, 820, 790, 865]' (리스트) | (['answer', 'bbox', 'evidence', 'page'], False) | 문법은 맞지만 객체가 아님 |
| bbox = ["530","820","790","865"] | ([], False) | 좌표가 글자 |
| bbox = [790,820,530,865] (x1 > x2) | ([], False) | 뒤집힌 Box |
| bbox = [530,820,1790,865] | ([], False) | 1000×1200 밖 |

| 검사 | 잡아내는 것 | 빼면? |
|---|---|---|
| try / except JSONDecodeError | JSON 문법이 깨진 답 (말을 덧붙였거나 잘림) | 프로그램이 그 자리에서 멈춰요 |
| isinstance(data, dict) | [1,2]처럼 문법은 맞지만 객체가 아닌 답 | .keys()에서 에러 |
| 집합 빼기 {...} - data.keys() | 필수 key 누락 목록 | 나중에 data["bbox"]에서 KeyError |
| bbox 길이·자료형·x1 < x2·이미지 범위 | 좌표 4개가 아니거나, 글자("530")이거나, 뒤집혔거나, 1000×1200 밖 | 엉뚱한 곳을 잘라 "근거"로 보여 줄 수 있어요 |

⚠️ JSON 파싱 성공 ≠ 내용이 사실. 이 코드는 구조만 봐요. answer·evidence·page가 맞는지는 OCR 원문과 그 bbox를 잘라 낸 이미지(crop)에 대조한 뒤 확인완료·검토필요·실패를 정해요.

### 가장 단순한 결합 규칙

```
if answer == total["raw"]:          # VLM 답이 OCR 원문과 같으면
    final = total["value"]          # → 채택
else:
    final = None                    # → 보류, 사람이 확인 (정답으로 세지 않음)
```

▶ 실제로 돌려 보면 VLM 답 6개는 Day16 본문 표에 기록된 SmolVLM 결과(이 PC에선 VLM 미실행) · OCR 원문(raw)은 수업 doc03의 실제 Tesseract 기록에서 ^TOTAL: · ^SUPPLIER: · ^DATE: 정규식으로 뽑음 (문서에 없는 항목은 None)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| answer='33000' vs raw='33000' | final = '33000' | ✅ 채택 |
| answer='ALPHA STUDIO' vs raw='ALPHA STUDIO' | final = 'ALPHA STUDIO' | ✅ 채택 |
| answer='2026-11-03' vs raw='2026-11-03' | final = '2026-11-03' | ✅ 채택 |
| answer='0.' vs raw=None | final = None | ⏸ 보류 — 문서에 근거 없음 |
| answer='33000' vs raw=None | final = None | ⏸ 보류 — 문서에 근거 없음 |
| answer='ALPHA STUDIO' vs raw=None | final = None | ⏸ 보류 — 문서에 근거 없음 |
| max_new_tokens=3으로 잘린 답 '33' vs raw '33000' | final = None | 에러 없이 잘린 답도 대조에서 걸러짐 |
| 채택 수 · 보류 수 | 3 · 3 | 보류는 정답으로 세지 않아요 |

| 코드 | 뜻 | 바꾸면? |
|---|---|---|
| answer == total["raw"] | VLM 답을 OCR이 읽은 원문 글자와 비교 | 대조 없이 final = answer로 하면 할인액 질문의 33000이 그대로 "할인액"으로 저장돼요(요약본 실측) |
| final = None | 보류 — 값이 없다는 게 아니라 "확인 전" | 0이나 ""를 넣으면 진짜 값과 구분이 안 돼요 |

🤔 왜 보류를 정답으로 안 세? — 보류는 "사람이 확인해야 함"이지 "맞음"이 아니에요. 보류를 정답에 넣으면 자동화율이 부풀고, 빼고 세면 "자동으로 처리된 것 중 몇 개가 맞았나"와 "몇 개를 사람에게 넘겼나"를 따로 볼 수 있어요. Day16 ⑥ "실제 VLM은 얼마나 지어낼까" 표라면 6개 중 채택 3 · 보류 3.

### PyTorch → ONNX 내보내기 torch.onnx.export()

```
import torch

model.eval()                                       # Dropout·BatchNorm을 추론 모드로
x = torch.randn(1, 3, 224, 224)                    # 실제로 넣을 크기의 예시 입력
torch.onnx.export(model, (x,), "model.onnx")       # PyTorch → ONNX
# 다음: ONNX 결과를 원본 PyTorch 결과와 비교 → 대상 장치에서 trtexec로 Engine Build
```

| 인자 | 뜻 | 바꾸면? |
|---|---|---|
| model | 내보낼 신경망 (eval() 상태로) | train 모드로 내보내면 Dropout 등이 섞여 결과가 원본과 달라져요 |
| (x,) | 예시 입력 (튜플) — 이걸 한 번 흘려 보내며 계산 순서를 기록 | 크기를 대충 주면 그 크기로 고정된 모델이 나와요 |
| dynamic_shapes | 바뀌는 축(예: batch)을 지정 | 안 주면 batch 1로 고정 — 4장씩 넣으면 실행이 거부돼요. 아래 trtexec의 min/opt/max shape와 짝 |

🤔 왜 Engine은 "대상 장치에서" 만들어? — TensorRT Engine은 그 GPU의 종류와 TensorRT·드라이버 버전에 맞춰 가장 빠른 계산 방법을 골라 굳힌 파일이에요. 노트북 RTX에서 만든 Engine을 Jetson에 옮기면 맞지 않아요. ONNX는 어디서나 쓰는 교환 형식, Engine은 그 장치 전용 실행 파일이라고 생각하면 돼요.

### trtexec로 빌드 · 추론 확인

```
# Linux Bash 예시: 준비된 ONNX와 NVIDIA GPU 필요
trtexec --onnx=model.onnx --saveEngine=model.engine

# 11.3: FP16 ONNX는 미리 준비하고 수치 검증
trtexec --onnx=model_fp16.onnx --saveEngine=fp16.engine

# 저장한 정적 shape Engine의 추론 확인
trtexec --loadEngine=model.engine

# 동적 입력 예시: 실제 입력 이름·shape로 교체
trtexec --onnx=model.onnx \
  --minShapes=input:1x3x224x224 \
  --optShapes=input:4x3x224x224 \
  --maxShapes=input:8x3x224x224 \
  --saveEngine=model_dynamic.engine
```

| 옵션 | 뜻 | 바꾸면? |
|---|---|---|
| --onnx= | 입력 ONNX 경로 | 지원 안 되는 연산자가 있으면 여기서 빌드 실패 |
| --saveEngine= | 만든 Engine 저장 경로 | 빼면 빌드·측정만 하고 파일이 안 남아요 |
| --loadEngine= | 저장한 Engine을 불러와 추론만 확인 | 빌드를 다시 안 해서 빨라요 |
| --minShapes / --optShapes / --maxShapes | 동적 입력의 최소·가장 자주 쓸·최대 shape (이름:배치x채널x높이x너비) | 범위 밖 입력(예: 배치 16)은 실행이 거부돼요. opt에 맞춰 커널을 고르니 실제 많이 쓰는 크기로 둬야 빨라요. 범위가 넓을수록 메모리를 더 잡아요 |
| --fp16 | (10.x 시절) 16비트로 빌드 | 11.x에서는 제거됨 → 정밀도는 ONNX의 자료형으로 미리 정해서 넣어요 |

⚠️ "실행 성공" ≠ "원본과 같은 결과". Engine이 돌아가도 숫자가 원본 PyTorch·ONNX와 일치하는지는 따로 대조해요. Jetson에서는 그 JetPack에 맞는 버전의 명령·옵션을 확인해요.

### 모듈 인터페이스 @dataclass

```
from dataclasses import dataclass

@dataclass
class OCRWord:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]   # xyxy

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
    raise NotImplementedError("OCR 어댑터를 연결하세요")      # 엔진별 어댑터 구현은 별도

def ask_document(image_path, question, pipe) -> VLMResult:
    messages = [{"role": "user", "content": [
        {"type": "image", "path": str(image_path)},
        {"type": "text", "text": question},
    ]}]
    raise NotImplementedError("VLM 어댑터를 연결하세요")      # 호출·파싱은 별도
```

▶ 실제로 돌려 보면 수업 doc03의 기록된 Tesseract 단어로 결과 상자 만들기

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| OCRWord(text, confidence, bbox) — TOTAL: 단어 | OCRWord(text='TOTAL:', confidence=96.7, bbox=(56, 501, 156, 523)) | 출력(__repr__)을 자동으로 만들어 줌 |
| len(res.words), res.words[-3].text | 27, '33000' | 엔진이 달라도 같은 모양으로 주고받기 |
| OCRWord('a', 1.0, (0, 0, 1, 1)) == OCRWord('a', 1.0, (0, 0, 1, 1)) | True | 값이 같으면 같다(__eq__ 자동) |
| OCRWord(text="a", conf=1.0, bbox=(0, 0, 1, 1)) (이름 오타) | TypeError: OCRWord.__init__() got an unexpected keyword argument 'conf' | dict와 달리 오타를 바로 알려 줌 |
| VLMResult(answer=None, evidence=None, raw_text="0.", model_id="SmolVLM-256M") | VLMResult(answer=None, evidence=None, raw_text='0.', model_id='SmolVLM-256M') | 근거 없으면 None + 원문은 따로 |
| run_ocr(image) | NotImplementedError: OCR 어댑터를 연결하세요 | 아직 어댑터가 없다는 걸 분명히 |

| 설계 | 뜻 | 안 하면? |
|---|---|---|
| @dataclass | 필드 이름·자료형만 적으면 생성자·출력 등을 자동으로 만들어 주는 "결과 상자" | dict로 주고받으면 키 이름 오타를 실행해 봐야 알아요 |
| OCRResult로 공통 구조 | Tesseract든 다른 엔진이든 같은 모양으로 변환(어댑터) | 엔진을 바꿀 때마다 뒤쪽 코드를 전부 고쳐야 해요 |
| answer: str \| None | "없음"을 허용 — 근거가 없으면 None + 검토 대상 | 빈 문자열·"모름" 같은 글자가 진짜 답과 섞여요 |
| raw_text 별도 | 모델이 실제로 낸 원문 그대로 | 파싱이 실패했을 때 무엇이 왔는지 알 수 없어요 |
| model_id · engine | 어떤 모델·엔진이 만든 결과인지 | 결과가 바뀌었을 때 원인을 못 찾아요 |

## 📖 Day16 용어 (49개)
- **OCR** (Optical Character Recognition (광학 문자 인식)): 이미지 속 글자를 읽어 텍스트로 바꾸는 전체 과정. 전처리 → 글자 인식(Text·Box·Confidence) → 후처리 → 정보 추출로 이어져요.
- **Tesseract** (Tesseract OCR 엔진): 무료로 쓰는 OCR 엔진 프로그램. pip가 아니라 따로 설치하고, kor·eng 같은 언어 데이터도 함께 깔아야 해요.
- **pytesseract** (Python wrapper for Tesseract): Python에서 Tesseract 프로그램을 불러 쓰게 해 주는 연결 고리 패키지. 엔진이 없으면 TesseractNotFoundError가 나요.
- **image_to_data** (pytesseract.image_to_data()): OCR 결과를 단어·줄·문단·페이지 단위로 text·conf·left·top·width·height와 함께 돌려주는 함수. image_to_string은 글자만 통째로 줘요.
- **언어 데이터** (Tesseract language data): Tesseract가 특정 언어 글자를 읽는 데 쓰는 학습 파일. lang="kor+eng"처럼 설치된 것만 고를 수 있어요.
- **PSM** (Page Segmentation Mode (페이지 분할 모드)): "이 이미지는 페이지/블록 하나/한 줄/한 단어"라고 Tesseract에 알려 주는 가정. 3=자동, 6=블록 하나, 7=한 줄, 8=한 단어.
- **OEM** (OCR Engine Mode): Tesseract가 어떤 인식 엔진을 쓸지 고르는 설정. 보통 --oem 3(사용 가능한 기본)으로 둬요.
- **원근 보정** (Perspective transform (원근 변환)): 비스듬히 찍혀 사다리꼴이 된 문서를 네 모서리 짝(src→dst)으로 3×3 행렬을 만들어 정면 직사각형으로 펴는 것.
- **CER** (Character Error Rate (문자 오류율)): (치환+삭제+삽입) ÷ 정답 글자 수. OCR 결과를 정답으로 고치려면 글자를 몇 번 고쳐야 하는지의 비율. 낮을수록 좋아요.
- **WER** (Word Error Rate (단어 오류율)): CER과 같은 계산을 글자 대신 단어 단위로 한 오류율.
- **편집 거리** (Levenshtein distance (편집 거리)): 한 문자열을 다른 문자열로 바꾸는 데 필요한 최소 치환(S)·삭제(D)·삽입(I) 횟수.
- **micro CER** (micro-averaged CER): 문서별 CER을 평균 내지 않고, 전체 오류 수 ÷ 전체 정답 글자 수로 계산한 CER. 글자 하나하나가 같은 무게예요.
- **정규식** (Regular expression (정규 표현식)): "20으로 시작하는 숫자 4개-숫자-숫자"처럼 글자의 모양 규칙을 적어 문자열에서 후보를 찾는 문법. Python은 re 모듈.
- **raw string** (raw string (r"...")): 앞에 r을 붙인 문자열. 역슬래시(\)를 Python이 먼저 해석하지 않아 정규식 패턴을 그대로 쓸 수 있어요.
- **Key-value** (Key-value extraction): "발행일" 같은 라벨(key)과 그 옆의 값(value)을 짝지어 뽑는 것. 근거 위치(bbox)까지 함께 묶어요.
- **정규화 값** (normalized value): OCR 원문(raw) "1,234,500원"을 프로그램이 쓰기 좋게 정리한 값(1234500). raw와 함께 저장해요.
- **Schema** (Schema (데이터 모양 규칙)): JSON에 어떤 key가 반드시 있고 각 값이 어떤 자료형이어야 하는지 정한 규칙. 바뀌면 버전을 올려 기록해요.
- **ISO 8601** (ISO 8601 날짜 형식): 날짜를 2026-11-09처럼 연-월-일 순서로 적는 국제 표준 형식.
- **hOCR** (hOCR / ALTO XML): OCR 결과를 페이지·문단·줄·단어 구조와 위치까지 담아 저장하는 표준 파일 형식들.
- **읽기 순서** (Reading order): 사람이 문서를 읽는 순서(제목 → 본문 → 표 → 합계). 다단·표에서는 OCR이 이 순서를 따로 정리해야 해요.
- **표 구조 인식** (Table structure recognition): 글자를 읽는 것과 별개로, 각 글자가 어느 행·열·셀에 속하는지 복원하는 문제.
- **VLM** (Vision-Language Model): 이미지와 글(질문)을 함께 입력받아 글로 답하는 모델. Vision Encoder + 연결 모듈 + 언어 모델 구조가 대표적이에요.
- **멀티모달** (Multimodal): 이미지·텍스트·소리처럼 종류가 다른 입력을 함께 다루는 것.
- **Processor** (Hugging Face Processor): VLM 입력 준비 담당. 이미지 전처리(pixel_values)와 Chat template·토큰화(input_ids)를 한 번에 해요.
- **Chat template**: role·content 대화 목록을 그 모델이 학습한 대화 형식(특수 토큰 포함)으로 바꾸는 틀.
- **Projector** (Projector (시각 특징 → 언어 공간)): Vision Encoder가 뽑은 이미지 특징을 언어 모델이 알아듣는 토큰 모양으로 옮겨 주는 연결 부분.
- **LLM** (Large Language Model): 글 토큰을 보고 다음 토큰을 하나씩 만들어 내는 큰 언어 모델. VLM의 답 생성 부분이에요.
- **do_sample** (do_sample / temperature): False면 매번 가장 그럴듯한 토큰을 골라 같은 입력에 같은 답(재현성). True면 확률대로 뽑고 temperature가 뽑기의 무작위 정도를 정해요.
- **KV cache** (Key/Value cache): 답을 한 토큰씩 만들 때 앞에서 계산한 Key·Value를 저장해 재사용하는 것. 빨라지지만 메모리를 차지해요.
- **Hallucination** (Hallucination (환각)): 모델이 이미지에 없는 값·표·근거를 자연스럽게 지어내는 것.
- **Grounding** (Grounding (근거 연결)): 답과 함께 원문 문자열·페이지·Box·Crop 같은 근거를 제시해 실제 문서와 대조할 수 있게 하는 것.
- **Fallback** (Fallback (대체 처리)): 근거가 부족할 때 억지로 답하지 않고 null·"확인할 수 없음"과 검토 필요 상태로 돌려주는 규칙.
- **Hybrid** (OCR + VLM Hybrid): OCR의 글자·위치 후보와 VLM의 문맥 판단을 역할 분리해 함께 쓰고 서로의 실패를 검증하는 방식.
- **패치** (Image patch): VLM이 이미지를 잘게 나눈 작은 조각. 해상도가 2배면 조각은 4배라 계산·메모리도 커져요.
- **ONNX** (Open Neural Network Exchange): 여러 프레임워크가 함께 읽는 모델 계산 그래프 형식. 입력·출력 이름·shape·연산자를 명시해요.
- **TensorRT** (NVIDIA TensorRT): NVIDIA GPU용 추론 최적화 도구. 지원 모델을 그 GPU에 맞춘 Engine으로 빌드해 빠르게 실행해요.
- **Engine** (TensorRT Engine): TensorRT가 빌드한 실행용 파일. 만든 GPU·TensorRT 버전·OS 조건에 묶이고, 실행 가능한 산출물이라 믿을 수 있는 출처만 불러와요.
- **trtexec** (trtexec (TensorRT 명령줄 도구)): ONNX로 Engine을 빌드(--onnx, --saveEngine)하거나 저장한 Engine을 불러와 추론을 확인(--loadEngine)하는 명령.
- **Build Phase** (TensorRT Build / Runtime Phase): Build는 커널 후보를 재 보고 연산을 합쳐 Engine을 만드는 단계, Runtime은 Engine을 불러와(ExecutionContext) 반복 추론하는 단계.
- **동적 shape** (Dynamic shape): 입력 크기(배치 등)가 실행마다 달라질 수 있게 min·opt·max 범위로 빌드하는 것. 범위 밖 입력은 거부돼요.
- **FP16** (Floating Point 16-bit (반정밀도)): 숫자 하나를 16비트로 저장하는 형식. FP32보다 메모리 절반·보통 더 빠르지만 정확도를 다시 확인해야 해요. INT8은 8비트 정수.
- **Jetson** (NVIDIA Jetson): 카메라 옆 현장에 설치하는 NVIDIA GPU가 달린 작은 컴퓨터(엣지 장치).
- **JetPack** (NVIDIA JetPack SDK): Jetson용 소프트웨어 묶음. CUDA·TensorRT·DeepStream 버전 조합이 정해져 있어요.
- **DeepStream** (NVIDIA DeepStream SDK): 영상 입력 → Decode → Batch(nvstreammux) → 추론(nvinfer, TensorRT) → 추적 → 화면/저장을 플러그인으로 잇는 영상 분석 파이프라인.
- **Throttling** (Thermal throttling): 장치가 뜨거워지면 스스로 속도를 낮추는 현상. 그래서 엣지에선 온도·전력도 함께 기록해요.
- **jiwer** (jiwer (CER·WER 계산 라이브러리)): 정답 문자열과 OCR·음성인식 결과를 넣으면 CER(글자 오류율)·WER(단어 오류율)을 계산해 주는 Python 패키지. 순서는 (정답, 예측).
- **필드 정확도** (Field accuracy (필드 정확도)): 합계·날짜·공급자 같은 업무 필드 중 값이 정확히 맞은 비율. 글자 하나만 틀려도 그 필드는 통째로 틀린 걸로 세요.
- **보류** (보류 (review · 사람 확인)): 근거와 맞지 않거나 원문에 없는 값을 자동으로 확정하지 않고 null로 두어 사람이 확인하게 하는 상태. 정답으로 세지 않아요.
- **greedy decoding** (Greedy decoding (탐욕적 생성)): 다음 토큰을 매번 가장 확률 높은 것 하나로 고르는 생성 방식. do_sample=False가 이것이라 같은 입력에 같은 답이 나와요.

---

# Day17 · FastAPI 지식 관리 API
_2026-11-10(화) · FastAPI · HTTP 요청/응답 · CRUD · Pydantic 검증 · JSON 파일 저장 · 상태 코드_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day17.html

📌 딱 3줄 요약

- FastAPI는 "이 주소로 이런 요청이 오면 이 함수를 실행해"를 연결해 주는 도구예요. RAG·Agent를 사람들이 쓰는 서비스로 만들 때 입구가 돼요.

- 오늘은 AI가 나중에 쓸 지식의 원본을 등록·조회·수정·삭제(CRUD)하고 JSON 파일에 저장하는 API를 만들어요.

- 응답 한 번으로 끝내지 않고 상태 코드(200·201·204·404·422)와 실제 파일, 재시작 후 값까지 확인해야 완성이에요.

🔧 Day17 명령어 한눈에 — 이론 구역별 명령어·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 코드 한두 줄만 남겼어요.

## 📦 필요한 패키지

| 패키지 (pip 이름) | 뭐 하는 애? | 코드에서 |
|---|---|---|
| fastapi | 주소(Route)와 함수를 연결하고, 입력 검증·자동 문서(/docs)를 만들어 줌 | from fastapi import FastAPI, HTTPException, Response |
| uvicorn | 실제로 네트워크 요청을 받아 FastAPI 앱에 넘겨주는 서버 프로그램 | 직접 import 안 함 — python -m uvicorn main:app으로 실행 |
| pydantic | 입력·응답 데이터의 모양 검사 (fastapi를 깔면 같이 설치됨) | from pydantic import BaseModel, Field |
| json, pathlib, threading | 파일 읽기/쓰기, 파일 경로, 잠금 — 파이썬 기본 내장이라 설치 안 함 | import json, from pathlib import Path, from threading import Lock |
| jinja2 (선택) | JSON 대신 HTML 화면을 만들 때 쓰는 템플릿 엔진 | from fastapi.templating import Jinja2Templates |
| pytest, httpx (선택) | 자동 테스트. TestClient가 내부에서 httpx를 씀 | from fastapi.testclient import TestClient |

### 🔧 설치 · 확인 (PowerShell)

```
python --version                                   # 3.10 이상인지
python -m venv .venv                               # 프로젝트 전용 가상환경 만들기
.\.venv\Scripts\Activate.ps1                       # 가상환경 켜기
python -m pip install fastapi uvicorn
python -c "import sys; print(sys.executable)"      # 지금 쓰는 python이 .venv 안의 것인지
python -m pip show fastapi pydantic uvicorn        # 설치된 버전 확인
```

💡 활성화(Activate.ps1)가 보안 정책 때문에 막힐 때는 정책을 억지로 바꾸지 말고, 가상환경의 python을 경로로 직접 부르면 돼요. 활성화는 "python이라고만 쳐도 .venv 것을 쓰게 해 주는" 편의 기능일 뿐이에요.

```
.\.venv\Scripts\python.exe -m pip install fastapi uvicorn
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
Invoke-RestMethod http://127.0.0.1:8000/health      # PowerShell에서 GET 요청 확인
```

🤔 왜 pip 대신 python -m pip? — PC에 파이썬이 여러 개 있으면 pip가 엉뚱한 파이썬에 설치할 수 있어요. python -m pip은 "지금 이 python에 설치해"라는 뜻이라 설치한 곳과 실행하는 곳이 어긋나지 않아요. python -m uvicorn도 같은 이유예요.

## 이거 어디에 써?

새로 온 직원이 "우리 팀 개발환경은 어디서 확인해요?"라고 물으면, 나중에는 AI Agent가 대답해 줄 거예요. 그런데 Agent가 답하려면 근거가 되는 안내문이 어딘가에 정리돼 있어야 해요. 오늘은 그 안내문(지식)의 원본을 넣고·찾고·고치고·지우는 API를 만들어요.

🧑‍💼

사내 온보딩"가상환경 사용 안내" 같은 문서를 등록·수정

❓

FAQ 관리자주 묻는 질문 답변을 분류별로 조회

🔎

RAG의 재료검색·답변 생성이 읽어 갈 원본 저장소

🤖

Agent 도구Agent가 "지식 조회"를 부를 때 실제로 찾는 곳

[그림: 사용자 화면 아래 FastAPI가 있고, 그 아래 지식 관리 API(오늘), RAG(이후), AI Agent(이후) 세 칸과 각각의 저장소가 있는 그림] 무엇을 보나: 맨 위 사용자 화면의 요청은 모두 FastAPI라는 입구를 지나요. 그 아래 세 갈래 중 초록 실선(지식 관리 API + JSON 파일)만 오늘 만들고, 회색 점선(RAG, Agent)은 이후 수업에서 붙여요. 🤔 왜 이 순서? RAG는 지식을 검색하고 Agent는 그걸 도구로 불러 쓰는데, 둘 다 원본이 정리돼 있어야 시작할 수 있어요. 원본이 낡으면 검색도 답변도 같이 흔들려요.

|  | 지식 관리 API (오늘) | RAG (다음 단계) | AI Agent (이후) |
|---|---|---|---|
| 답하는 질문 | "안내문을 어떻게 유지하지?" | "관련 근거가 어디 있지?" | "이 작업엔 어떤 도구가 필요하지?" |
| 하는 일 | 등록·조회·수정·삭제, 저장 상태 확인 | 문서 검색 → 문맥 제공 → 답변·출처 | 도구 선택·호출 제안 → 실행 결과 점검 |
| AI 모델 필요? | ❌ 없어도 동작 | ✅ 임베딩·LLM | ✅ LLM |

🤔 왜 셋을 따로 나눠? — 같은 서비스로 이어 붙일 수는 있지만 책임이 달라요. 원본 저장은 "정확히 남기기", 검색은 "잘 찾기", Agent는 "무슨 도구를 쓸지 판단하기"예요. 한 덩어리로 섞으면 검색 방식을 바꿀 때 저장 코드까지 건드리게 돼요.

⚠️ FastAPI ≠ AI 모델 ≠ Agent. 오늘 코드에는 PDF·임베딩·LLM 호출이 하나도 없어요. 그리고 원본을 PUT·DELETE 해도 나중에 만들 검색 인덱스는 자동으로 바뀌지 않아요(따로 다시 만들어야 해요).

## ① 첫 서버 띄우기

API(Application Programming Interface)는 프로그램끼리 주고받는 약속된 창구예요. 식당으로 치면 메뉴판 + 주문 창구 — "이 주소로 이런 형식으로 주문하면, 이런 형식으로 음식이 나온다"를 정해 둔 거예요.

🔧 main.py — 첫 JSON 응답 명령어 사전에서 자세히 →

```
app = FastAPI(title="Knowledge API")       # 앱 객체 만들기
@app.get("/health")                        # GET /health 요청이 오면
```

🔧 서버 실행 명령어 사전에서 자세히 →

```
python -m uvicorn main:app --reload
http://127.0.0.1:8000/health      → {"status":"ok"}
```

🔧 /docs — 자동 문서에서 요청 보내기 명령어 사전에서 자세히 →

## ② HTTP 요청과 응답

HTTP는 브라우저·프로그램이 서버와 대화하는 규칙이에요. 대화는 항상 요청 1개 ↔ 응답 1개가 한 쌍이에요.

[그림: http://127.0.0.1:8000/api/knowledge/2?category=api 를 통신 방식, 내 컴퓨터, 포트, 자원 경로, Path 값, Query로 나눠 색칠한 그림] 무엇을 보나: URL 하나를 색깔별로 잘랐어요. 127.0.0.1:8000까지가 "어느 서버", /api/knowledge가 "무슨 자원", /2가 "그중 어느 것(Path 값)", ?category=api가 "어떤 조건으로(Query)"예요. 🤔 왜 둘을 나눠? Path는 대상을 콕 집을 때(2번 지식), Query는 목록을 걸러낼 때(분류가 api인 것만) 써요. 빠지면 안 되는 건 Path, 있어도 없어도 되는 건 Query가 자연스러워요.

| 이름 | 예시 | 역할 |
|---|---|---|
| Method | GET · POST · PUT · DELETE | 무엇을 하려는지 (요청의 의도) |
| Path | /api/knowledge/4 | 어떤 자원의 어느 것인지 |
| Query | ?category=api | 목록을 거를 선택 조건 |
| Header | Content-Type: application/json | 본문 형식 같은 부가 정보 |
| Body | {"title": "실습 환경 안내", ...} | 새로 만들거나 고칠 데이터 (GET엔 보통 없음) |
| Status | 200 · 201 · 204 · 404 · 422 | 처리 결과를 숫자로 |
| Response body | 지식 JSON / 오류 detail / 없음 | 클라이언트가 받는 내용 |

```
# 요청                                   # 응답
GET /api/knowledge/2                     Status: 200
Host: 127.0.0.1:8000                     Content-Type: application/json
(Body 없음)                              {"id": 2, "title": "서버 실행",
                                          "category": "api",
                                          "content": "uvicorn으로 개발 서버를 켠다."}
```

[그림: 클라이언트, Uvicorn, Route, 타입 검증, 처리 함수, storage.py 여섯 칸을 지나 200, 404, 422로 갈라지는 흐름도] 무엇을 보나: GET /api/knowledge/2 한 건이 ①→⑥을 차례로 지나요. 같은 경로라도 어디서 멈추느냐에 따라 결과가 갈라져요 — foo는 ④ 타입 검증에서 정수로 못 바꿔 422(함수는 아예 실행 안 됨), 9999는 ⑤ 함수가 끝까지 못 찾아 404, 2는 ⑥에서 찾아 200. 🤔 왜 알아야 해? 상태 코드만 보고 "어느 단계에서 막혔는지"를 바로 알 수 있어서, 오류가 나면 고칠 곳을 좁힐 수 있어요.

🤔 왜 URL을 직접 쪼개서 해석하지 않아? — FastAPI가 Route(Method+Path 대조) → 타입 검증 → 함수 → 응답을 순서대로 연결해 줘요. 우리는 "knowledge_id: int"라고 약속만 적으면, 문자열 "2"를 정수 2로 바꾸고 안 되면 422를 돌려주는 일까지 자동이에요.

## ③ JSON 파일 저장소

오늘의 저장소는 데이터베이스가 아니라 JSON 파일 하나예요. 서버를 껐다 켜도 지식이 남아 있게 하려면 메모리(변수)가 아니라 파일에 써야 해요.

```
// data/knowledge.json
{
  "next_id": 4,
  "entries": [
    {"id": 1, "title": "가상환경", "category": "setup", "content": "프로젝트별 패키지를 분리한다."},
    {"id": 2, "title": "서버 실행", "category": "api",   "content": "uvicorn으로 개발 서버를 켠다."},
    {"id": 3, "title": "입력 오류", "category": "api",   "content": "422이면 입력값을 확인한다."}
  ]
}
```

| 키 | 뜻 |
|---|---|
| entries | 지식 항목 목록. 항목 1건 = id + 제목 + 본문 + 분류 |
| id | 서버가 발급하는 번호. 사용자가 정하지 않아요 |
| next_id | 다음에 줄 번호. 파일 안에만 두고, API 응답에는 내보내지 않아요 |

⚠️ JSON 문자열은 큰따옴표만 돼요('가상환경' ❌). 마지막 항목 뒤 쉼표도 안 돼요. 파일 문법이 깨지면 서버가 읽다가 실패해요.

🔧 storage.py — 읽기·쓰기 보조 함수 명령어 사전에서 자세히 →

```
json.dump(data, f, ensure_ascii=False, indent=2)
temp.replace(DATA)                       # 다 쓴 임시 파일로 바꿔치기
```

[그림: 왼쪽은 원본에 바로 쓰다 중간에 끊겨 반쯤 쓰인 파일, 오른쪽은 임시 파일에 다 쓴 뒤 원본과 바꿔치기하는 그림] 무엇을 보나: 왼쪽은 knowledge.json을 바로 "w"로 열어 쓰다가 도중에 멈춘 경우예요. "w"로 여는 순간 원본 내용이 지워지니, 반쯤 쓰인 깨진 JSON만 남아요. 오른쪽은 .tmp에 먼저 끝까지 쓰고, 성공했을 때만 replace()로 바꿔치기해요. 🤔 왜? 쓰다 실패해도 망가지는 건 임시 파일뿐이라 원본이 살아남아요. 그래서 "저장 실패 = 예전 데이터 그대로"가 보장돼요.

🔧 with LOCK: — 읽기부터 저장까지 한 번에 잠그기 명령어 사전에서 자세히 →

```
with LOCK:                       # 이 블록은 한 번에 한 요청만
    data = read_data()           # 1. 읽고
```

[그림: 동시에 20건을 등록했을 때 Lock 없이는 1건, with LOCK은 20건이 저장된 가로 막대그래프] 무엇을 보나: 스레드 20개가 동시에 "파일 읽기 → 항목 추가 → 저장"을 했을 때 파일에 남은 항목 수예요(이 PC에서 직접 5번 돌린 결과, 읽기와 저장 사이에 0.01초 틈을 일부러 넣어 차이를 크게 보이게 했어요). Lock이 없으면 20건을 보냈는데 1건만 남았어요. 🤔 왜? 20개가 모두 "빈 파일"을 거의 동시에 읽고, 각자 1건을 더해 저장했어요. 나중에 저장한 쪽이 앞선 저장을 덮어써서 결국 하나만 남아요. with LOCK:으로 읽기~저장을 묶으면 앞 요청이 끝날 때까지 다음 요청이 기다리니 20건이 다 남아요.

## ④ 조회 (GET) — 목록·상세·조건

🔧 목록 조회 명령어 사전에서 자세히 →

```
@app.get("/api/knowledge")
    return data["entries"]          # list를 돌려주면 JSON 배열
```

🔧 상세 조회 + 404 명령어 사전에서 자세히 →

```
@app.get("/api/knowledge/{knowledge_id}")
    raise HTTPException(404, "Knowledge not found")   # 끝까지 없으면 404
```

🔧 Query로 목록 거르기 명령어 사전에서 자세히 →

```
@app.get("/api/knowledge")
def list_knowledge(category: str | None = None):     # 기본값이 있으면 → Query, 생략 가능
```

## ⑤ 등록 (POST) · 입력 검증

새 지식은 URL이 아니라 Body에 담아 보내요. ID는 보내지 않아요 — 서버가 정해요.

```
POST /api/knowledge
Content-Type: application/json

{
  "title": "실습 환경 안내",
  "content": "Python 3.10 이상을 사용한다.",
  "category": "setup"
}
```

🔧 Pydantic으로 입력 모양 정하기 명령어 사전에서 자세히 →

```
class KnowledgeWrite(BaseModel):                          # POST·PUT 입력
    title: str = Field(min_length=1, max_length=80)
```

🔧 등록 함수 명령어 사전에서 자세히 →

```
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):        # Body를 KnowledgeWrite로 검사
```

[그림: 위는 len+1 방식으로 ID 2 삭제 후 새 항목이 3을 받아 충돌, 아래는 next_id 방식으로 4를 받는 그림] 무엇을 보나: 위 줄은 "목록 길이 + 1"로 번호를 매긴 경우예요. ID 2를 지우면 길이가 2가 되고, 새 항목이 3을 받아 이미 있는 3번과 겹쳐요(빨강). 아래 줄은 next_id를 파일에 따로 적어 두는 강의 방식 — 삭제해도 4는 그대로라 새 항목은 4를 받아요. 🤔 왜? ID는 "이 지식의 이름표"라 한 번 쓴 번호는 다시 쓰면 안 돼요. 지워진 2번을 기억하던 다른 프로그램이 새 항목을 옛 지식으로 착각할 수 있어요. 그래서 next_id는 줄이지 않고, entries와 같은 파일에 저장해 재시작해도 이어져요.

🔧 422 — 잘못된 입력은 저장 전에 막힘 명령어 사전에서 자세히 →

## ⑥ 수정 (PUT)

```
PUT /api/knowledge/4
Content-Type: application/json

{"title": "실습 환경 안내", "content": "프로젝트별 가상환경을 사용한다.", "category": "setup"}
```

이 수업의 PUT 규칙: 세 필드를 모두 보내 내용을 통째로 교체해요. ID·항목 수·next_id는 그대로예요.

🔧 수정 함수 명령어 사전에서 자세히 →

```
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):   # Path + Body 둘 다
```

[그림: id 1, 3, 4 세 항목 아래에 index 0, 1, 2를 적고, entries[4]는 IndexError, entries[1]은 id 3 항목이라는 설명을 붙인 그림] 무엇을 보나: ID 2를 지운 뒤 목록이에요. 칸 안 숫자는 id(이름표), 아래 주황 숫자는 index(지금 줄 선 순서)예요. entries[4]로 id 4를 찾으면 위치는 0~2뿐이라 오류, entries[1]로 id 1을 찾으면 엉뚱하게 id 3이 나와요. 🤔 왜? 처음엔 우연히 id 1이 index 0, id 2가 index 1…처럼 하나씩 밀려 맞아 보이지만, 삭제가 한 번 일어나면 바로 어긋나요. 그래서 data["entries"][knowledge_id]로 찾지 않고, 하나씩 돌며 entry["id"] == knowledge_id로 비교해요.

## ⑦ 삭제 (DELETE) · 상태 코드

지우는 건 knowledge.json 파일 전체가 아니라, 그 안에서 ID가 일치하는 항목 하나예요. 나머지 항목과 next_id는 그대로예요.

🔧 삭제 함수 명령어 사전에서 자세히 →

```
@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
    return Response(status_code=204)       # 본문 없는 성공
```

### 지금까지 쓴 상태 코드

| 코드 | 우리 API의 상황 | 다음에 확인할 것 |
|---|---|---|
| 200 OK | 목록·상세 조회 / 정상 수정 | 응답 데이터 = 저장된 값? |
| 201 Created | 새 항목 생성 | 새 ID가 실제 파일에도? |
| 204 No Content | 삭제 성공 | 본문 없음 + 재조회 404 |
| 404 Not Found | 없는 경로 / 없는 지식 | Path 오타인지 ID가 없는 건지 |
| 405 Method Not Allowed | 경로는 있는데 그 Method는 없음 | GET·POST·PUT·DELETE 중 맞는지 |
| 422 Unprocessable | 자료형·필수값·길이 규칙 위반 | detail의 loc·msg |
| 500 Server Error | 서버 코드·파일 접근 오류 | 서버 터미널 로그와 저장 파일 |

🤔 왜 숫자가 2xx·4xx·5xx로 나뉘어? — 첫 자리만 봐도 누구 탓인지 알 수 있어요. 2xx 성공, 4xx 요청한 쪽이 뭔가 잘못 보냄(주소·입력), 5xx 서버가 처리하다 실패. 그래서 404·422를 200으로 숨기면 안 돼요 — 받는 프로그램이 실패를 알아챌 방법이 사라져요.

⚠️ 실패한 요청은 파일을 바꾸면 안 돼요. 422는 함수 실행 전에 막히니 저장 코드가 안 돌고, 404는 저장 전에 raise 하고, 쓰기 실패 땐 성공 응답을 만들지 않아요. 확인할 때는 실패 요청 전후로 항목 수와 next_id가 같은지 봐요.

## 🧪 직접 요청 보내 보기

브라우저 안에서 오늘 만든 API를 흉내 냈어요(실제 서버는 아니고, 강의 코드와 같은 규칙으로 움직여요). Method와 ID, Body를 바꿔 보내기를 누르면 상태 코드·응답·파일 내용이 바뀌어요. 아래 칸은 knowledge.json이에요.

💡 해 볼 것: ① id에 foo 넣고 GET → 422 ② POST 두 번 → id 4, 5 ③ id 2 DELETE 후 len+1 방식을 켜고 POST → 번호 충돌 ④ PUT으로 없는 id 9999 → 404이고 파일은 그대로 ⑤ 서버 재시작 후에도 파일이 남는지.

## ⑧ 끝까지 추적 · 재시작 확인

지식 하나를 만들고 → 고치고 → 서버를 껐다 켜고 → 지우고 → 다시 껐다 켜며, 각 단계에서 파일 상태가 예상대로인지 봐요.

[그림: 6단계 동안 저장된 항목 수(3,4,4,4,3,3)와 next_id(4,5,5,5,5,5)를 계단 그래프로 그리고 각 단계의 상태 코드를 적은 그림] 무엇을 보나: 강의 코드를 실제로 돌리며 기록한 값이에요. 보라 실선은 저장된 항목 수, 주황 점선은 next_id예요. POST에서 둘 다 1씩 오르고, PUT과 재시작에서는 그대로, DELETE에서는 항목 수만 줄고 next_id는 5로 남아요. 아래 글씨는 각 단계의 응답이에요. 🤔 왜 재시작까지 봐? 응답은 "방금 메모리에서 만든 값"일 뿐이에요. 서버를 껐다 켜도 같은 값이 나와야 파일에 진짜 저장된 거예요. 응답만 바뀌고 파일이 그대로면 미완성이에요.

| 순서 | 요청·조작 | 예상 데이터 상태 |
|---|---|---|
| 1 | 처음 파일 확인 | 항목 3개 · next_id 4 |
| 2 | "실습 환경 안내" POST → 받은 ID 기록 | 항목 4개 · id 4 · next_id 5 |
| 3 | 같은 ID 본문 PUT | 개수·ID 그대로, 본문만 변경 |
| 4 | 서버 종료(Ctrl+C) → 같은 폴더·같은 명령으로 재실행 | 수정한 본문이 그대로 조회됨 |
| 5 | 같은 ID DELETE → 재조회 | 상세 404 · 목록에서 사라짐 |
| 6 | 다시 재실행 → 재조회 | 삭제 상태 유지 · next_id 5 |

⚠️ 서버 시작 ≠ 데이터 초기화. 서버를 켤 때마다 seed(처음 데이터)를 복사하는 코드가 있으면 재시작할 때마다 수정·삭제가 되돌아가요. 초기화는 서버를 끄고 따로, 일부러 하는 작업이에요. 재실행했더니 값이 달라졌다면 자동 초기화나 다른 폴더의 파일을 읽고 있지 않은지 의심해요.

## ⑨ 역할별로 파일 나누기

기능이 늘면 main.py 하나가 너무 길어져요. 바뀌는 이유가 다른 코드를 다른 파일로 나눠요.

```
fastapi_knowledge/
├─ app/
│  ├─ __init__.py
│  ├─ main.py                 ← 앱 만들고 Router 등록만
│  ├─ schemas.py              ← 입력·응답 모양 (KnowledgeWrite, Knowledge)
│  ├─ routers/
│  │  ├─ __init__.py
│  │  └─ knowledge.py         ← HTTP: Method·Path·상태 코드
│  └─ repositories/
│     ├─ __init__.py
│     └─ json_knowledge.py    ← 저장: 읽기·변경·저장·LOCK
├─ data/
│  ├─ knowledge.json          ← 실제 데이터
│  └─ knowledge.seed.json     ← 처음 데이터 (초기화용)
├─ README.md
└─ requirements.txt
```

[그림: main.py에서 routers, repositories, data/knowledge.json으로 이어지고 아래 schemas.py가 router에 연결된 구조도] 무엇을 보나: 요청은 왼쪽에서 오른쪽으로 흘러요. main은 조립만, Router는 HTTP 약속(어떤 주소·어떤 상태 코드), Repository는 파일 읽기·저장, 맨 오른쪽이 실제 데이터예요. 아래 schemas는 Router가 입력 검사와 응답 모양에 써요. 🤔 왜 나눠? 데이터가 같아도 HTTP 규칙과 저장 방식은 바뀌는 이유가 달라요. 다음 수업에서 JSON 파일을 PostgreSQL로 바꿀 때 Repository만 고치면 되고, URL·상태 코드는 그대로 둘 수 있어요.

🔧 app/main.py — 조립 명령어 사전에서 자세히 →

```
app.include_router(router)                 # Router에 모아 둔 경로들을 앱에 연결
python -m uvicorn app.main:app --reload
```

🔧 APIRouter — 관련 경로 묶기 명령어 사전에서 자세히 →

```
router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
@router.post("", status_code=201, response_model=Knowledge)
```

🔧 Repository — 저장 로직 그대로 옮기기 명령어 사전에서 자세히 →

```
BASE = Path(__file__).resolve().parents[2]       # 프로젝트 루트 (repositories → app → 루트)
def create(fields: dict) -> dict:
```

🔧 같은 데이터를 HTML 화면으로 (Template) 명령어 사전에서 자세히 →

```
templates = Jinja2Templates(directory=BASE / "templates")
@app.get("/web/knowledge", response_class=HTMLResponse)
```

🔧 자동 테스트 — 실제 지식 파일은 건드리지 않게 명령어 사전에서 자세히 →

```
monkeypatch.setattr(repo, "DATA", test_file)       # 저장 경로만 임시 파일로 바꿔치기
with TestClient(app) as client:
```

## ⑩ 다음 단계와 연결

### 왜 파일로 되는데 DB를 배워?

| 오늘의 JSON 파일 | PostgreSQL에서 풀 질문 |
|---|---|
| 정상 저장 후 재시작해도 남아요 | 동시에 여러 요청이 바꾸면? |
| 작은 파일 전체를 읽고 전체를 다시 저장해요 | 데이터가 10만 건이면? 필요한 줄만 읽을 수 없나? |
| Lock은 서버 프로세스 하나 안에서만 | 서버가 여러 대면? |
| 표 하나(entries) | 작성자·분류 같은 여러 표의 관계를 어떻게 지키지? |
| — | 여러 변경을 "모두 성공 아니면 모두 취소"로 하려면? → 트랜잭션 |

| 날짜 | 핵심 질문 | 결과 |
|---|---|---|
| 11/10 (오늘) | HTTP로 지식을 어떻게 CRUD 하나? | JSON 파일 기반 FastAPI CRUD |
| 11/11 | 표·관계·SQL로 데이터를 어떻게 다루나? | SQL CRUD · JOIN · ERD · 트랜잭션 |
| 11/12 | 오늘 API를 PostgreSQL에 어떻게 연결하나? | ORM · 연동 CRUD · 설정 · 통합 테스트 |
| 11/13 ~ 11/20 | LLM · Prompt · LLM API · Embedding · Vector DB · Chunking | Retriever–Prompt–Generator, 즉 RAG를 API로 제공 |

🤔 왜 URL·Method는 그대로 두라고 해? — 이 API를 부르는 쪽(화면, 다른 서버, 나중의 Agent)은 주소와 응답 모양만 알아요. 저장소가 파일에서 DB로 바뀌어도 약속이 같으면 부르는 쪽은 아무것도 고칠 필요가 없어요. 그래서 바꾸는 건 저장 구현(Repository)뿐이에요.

### Agent와 API가 만나는 두 방향

| 방향 | 흐름 | 뜻 |
|---|---|---|
| 사용자가 Agent를 부름 | 사용자 → FastAPI → Agent 처리 | API가 Agent의 서비스 입구. 작업 요청을 검증하고 결과·상태를 전달 |
| Agent가 API를 씀 | Agent → 도구 함수 → 지식 API | "지식 조회"를 도구로 연결. 도구의 설명·입력·실행 코드는 따로 만들어요 |

| 단계 | 누가 | 예 |
|---|---|---|
| 판단 | Agent(LLM) | "지식 조회가 필요하다" |
| 도구 호출 제안 | Agent(LLM) | get_knowledge(2) |
| 검증·실행 | 애플리케이션 | 권한 확인 → GET /api/knowledge/2 |
| 결과 반환 | 애플리케이션 | 원문 + 상태 기록 |

⚠️ FastAPI 경로를 만들었다고 Agent 도구가 되지는 않아요. 모델은 "이걸 부르자"고 제안만 하고, 실제 호출·권한 확인·기록은 우리 애플리케이션 책임이에요. 처음 연결할 땐 조회(GET)부터 — 수정·삭제를 Agent에게 맡기려면 별도 권한·승인 규칙이 먼저예요.

🤔 왜 Agent에게 삭제부터 안 줘? — LLM은 그럴듯하게 틀릴 수 있어요(Day16의 VLM이 없는 걸 지어낸 것처럼). 조회는 틀려도 답이 이상할 뿐이지만, 삭제는 원본이 사라져요. 되돌릴 수 없는 일일수록 사람의 승인을 거치게 해요.

## 🩺 문제가 생기면 — 가장 가까운 원인부터

| 증상 | 먼저 볼 것 | 구분할 점 |
|---|---|---|
| 서버 접속이 안 됨 | 터미널(서버가 떠 있나) · 포트 · 실행 명령 | HTTP 오류 응답과는 다름 — 응답 자체가 없음 |
| 404 / 405 | /docs의 Path · Method · ID | 없는 자원(404) vs 그 Method가 없음(405) |
| 422 | detail의 loc · msg | Path · Query · Body 중 어디인가 |
| 파일 없음 / JSON 오류 | 파일 경로 · 괄호 · 큰따옴표 · 끝 쉼표 | 빈 데이터로 덮어쓰지 않기 |
| 재실행하면 값이 달라짐 | 자동 초기화 코드 · 다른 폴더의 파일 | 서버 재시작과 seed 복사는 다른 일 |
| 코드를 고쳤는데 그대로 | 서버가 두 개 떠 있지 않은지 · --reload | 같은 Route 중복 등록 |

⚠️ 없는 ID를 시험할 땐 Body는 정상값으로 보내요. 없는 ID + 빠진 필드를 한꺼번에 보내면 422가 먼저 나와서 404를 확인할 수 없어요. 오류 조건은 한 번에 하나씩.

## 원본 자료

- 강의: 20261110.pdf (FastAPI 입문 · 지식 관리 API, 94쪽) — 플랫폼 속 위치 p2–11 · 첫 서버 p12–20 · JSON 조회 p22–28 · POST·검증 p30–36 · PUT p38–41 · DELETE·상태 코드 p43–47 · 역할별 구조·Template p49–55 · 플랫폼 연결 p57–65 · 부록(저장·Lock·용어·PATCH·검색·Template·테스트·Windows 실행) p79–91

- 노트의 실제 응답·Lock 실험 값: 강의 코드를 FastAPI 0.162.2 · Pydantic 2.14.3에서 직접 실행해 확인 (그림 스크립트 tools/visuals/make_day17.py)

- 공식 문서: FastAPI First Steps · Path Parameters · Query Parameters · Request Body

- 공식 문서: Response Status Code · Handling Errors · Body - Updates (PUT/PATCH) · Response Model

- 공식 문서: Bigger Applications (APIRouter) · Templates · Testing · Uvicorn Settings

- 공식 문서: Pydantic Fields · Python json · pathlib · threading.Lock · os.replace · RFC 9110 HTTP Semantics

## 🔧 Day17 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day17)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 패키지 | python -m pip install fastapi uvicorn |
| ① 첫 서버 띄우기 | FastAPI(), @app.get(), python -m uvicorn main:app --reload |
| ② HTTP 요청과 응답 | Method · Path · Query · Body · Status, /docs의 Try it out |
| ③ JSON 파일 저장소 | json.load(), json.dump(), with LOCK:, temp.replace() |
| ④ 조회 (GET) | @app.get("/api/knowledge/{knowledge_id}"), raise HTTPException(404) |
| ⑤ 등록 (POST) · 입력 검증 | class KnowledgeWrite(BaseModel), Field(min_length=, max_length=), status_code=201 |
| ⑥ 수정 (PUT) | @app.put(), enumerate(), data["entries"][i] = updated |
| ⑦ 삭제 (DELETE) · 상태 코드 | @app.delete(status_code=204), Response(status_code=204) |
| 🧪 직접 요청 보내 보기 | (브라우저 안에서 CRUD 흉내) |
| ⑧ 끝까지 추적 · 재시작 | Ctrl+C → 같은 명령으로 재실행 → GET |
| ⑨ 역할별로 파일 나누기 | APIRouter(prefix=, tags=), app.include_router(), uvicorn app.main:app |
| ⑩ 다음 단계와 연결 | PostgreSQL · RAG · Agent (오늘은 구현 안 함) |

### main.py — 첫 JSON 응답

```
from fastapi import FastAPI

app = FastAPI(title="Knowledge API")       # 앱 객체 만들기

@app.get("/health")                        # GET /health 요청이 오면
def health():                              # 이 함수를 실행해서
    return {"status": "ok"}                # 이 값을 JSON으로 돌려줌
```

▶ 실제로 돌려 보면 강의 main.py를 TestClient로 호출

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.get("/health") | 200 {"status":"ok"} | dict를 돌려주면 JSON으로 응답 · 상태 코드 200 |
| r.headers["content-type"] | application/json | FastAPI가 JSON으로 바꿔 보냄 |
| app.title | 'Knowledge API' | /docs 맨 위에 보이는 이름 |
| [r.path for r in app.routes] | ['/openapi.json', '/docs', '/redoc', '/health'] | @app.get 한 줄로 /health 등록 + 문서 주소는 자동으로 생김 |

| 코드 | 역할 |
|---|---|
| FastAPI(title=...) | 앱 객체 생성. title은 /docs 맨 위에 보이는 이름(선택) |
| @app.get("/health") | Route 등록 — "GET 방식 + /health 경로"를 바로 아래 함수에 연결 |
| def health(): | 요청 처리 함수(View / Handler라고도 불러요). 요청이 올 때마다 실행 |
| return {...} | 파이썬 dict를 돌려주면 FastAPI가 JSON으로 바꿔서 응답 (상태 코드 200) |

🤔 왜 함수 위에 @를 붙여? — @app.get(...)은 데코레이터예요. 함수를 바꾸지 않고 "이 함수를 이 주소에 등록해 둬"라는 표시만 붙여요. 그래서 함수는 평범한 파이썬 함수 그대로고, 언제 실행할지는 FastAPI가 정해요.

### 서버 실행

```
# main.py가 있는 폴더에서, .venv 활성화 후
python -m uvicorn main:app --reload

# 브라우저 주소창에
http://127.0.0.1:8000/health      → {"status":"ok"}
http://127.0.0.1:8000/docs        → 자동 API 문서

# 서버 끄기: 실행 중인 터미널에서 Ctrl+C
```

▶ 실제로 돌려 보면 임시 폴더에 main.py를 만들고 빈 포트로 잠깐 띄운 뒤 Ctrl+C 대신 프로세스를 종료

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| python -m uvicorn main:app --reload --port 52551 | Uvicorn running on http://127.0.0.1:52551 (Press CTRL+C to quit) | 포트만 빈 번호로 바꿔 실행 — 기본은 8000 |
| (같은 터미널 로그) | Started reloader process [12512] using StatReload | --reload: 파일 변경을 지켜보는 감시 프로세스가 따로 뜸 |
| 브라우저 http://127.0.0.1:52551/health | 200 {"status":"ok"} | 주소창 = GET 요청 |
| 브라우저 http://127.0.0.1:52551/docs | 200 text/html; charset=utf-8 · Swagger UI 페이지 | 자동 API 문서 (HTML) |
| 같은 포트로 uvicorn 한 번 더 실행 | [Errno 10048] error while attempting to bind on address ('127.0.0.1', 52551) | 종료 코드 3 — 서버는 한 개만! |

| 부분 | 뜻 |
|---|---|
| main | main.py 파일(모듈) 이름 — .py는 안 붙임 |
| :app | 그 파일 안의 앱 변수 이름 (app = FastAPI()의 app) |
| --reload | 코드를 저장하면 서버가 알아서 다시 켜짐. 개발용이에요 |
| 127.0.0.1 | "내 컴퓨터" (= localhost). 다른 PC에서는 이 주소로 못 들어와요 |
| 8000 | 포트 — 한 컴퓨터 안에서 프로그램마다 다른 "문 번호". 바꾸면 URL도 같이 바꿔요 |

🤔 왜 Uvicorn이 따로 있어? — FastAPI는 "요청이 오면 어떤 함수를 부를지"만 알아요. 네트워크에서 요청을 실제로 받아 오는 일은 Uvicorn(서버 프로그램)이 해요. 식당으로 치면 Uvicorn은 손님을 받는 홀 직원, FastAPI는 주문서를 보고 요리사를 정하는 주방장이에요.

⚠️ 서버는 한 개만 띄우세요. 같은 포트로 두 번 실행하면 "이미 사용 중" 오류가 나거나, 예전 코드가 도는 서버에 요청이 가서 헷갈려요. 고쳤는데 안 바뀌면 터미널이 두 개 떠 있지 않은지부터 보세요.

### /docs — 자동 문서에서 요청 보내기

주소창은 기본적으로 GET만 보내요. POST·PUT·DELETE는 /docs(Swagger UI)에서 보내요.

| 순서 | 할 일 | 확인할 것 |
|---|---|---|
| 1 | 경로를 펼쳐 입력 약속 읽기 | Path·Query·Body에 무엇이 필요한지 |
| 2 | Try it out | 입력 칸이 열림 |
| 3 | Execute | 실제 요청이 서버로 감 |
| 4 | Server response | 실제 Code(상태 코드)와 Response body |

▶ 실제로 돌려 보면 /docs 화면이 읽어 가는 openapi.json · Execute와 같은 요청

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.get("/docs") | 200 text/html; charset=utf-8 | 사람이 보는 화면 (HTML) |
| client.get("/openapi.json").json()["paths"] | {"/health": ["get"], "/api/knowledge": ["get", "post"], "/api/knowledge/{knowledge_id}": ["delete", "get", "put"]} | Route마다 열 수 있는 Method 목록 = /docs의 줄들 |
| … ["/api/knowledge/{knowledge_id}"]["get"]["responses"] | ['200', '422'] | 문서상 Responses에는 404가 없음! |
| client.get("/api/knowledge/9999") | 404 {"detail":"Knowledge not found"} | 그래도 실제 Server response는 404 — 진짜는 이쪽 |
| … ["/api/knowledge"]["post"]["responses"] | ['201', '422'] | response_model · status_code에서 자동으로 만든 예시 |
| Execute = POST /api/knowledge (Body 세 필드) | 201 {"title":"실습 환경 안내","content":"Python 3.10 이상을 사용한다.","category":"setup","id":4} | Try it out → Execute가 보내는 진짜 요청 |

⚠️ 아래쪽 Responses 칸은 "이런 응답이 나올 수 있다"는 문서상 예시예요. 진짜 결과는 Server response의 Code·Body로 확인해요.

### storage.py — 읽기·쓰기 보조 함수

```
import json
from pathlib import Path
from threading import Lock

BASE = Path(__file__).resolve().parent         # 이 파일(storage.py)이 있는 폴더
DATA = BASE / "data" / "knowledge.json"
LOCK = Lock()

def read_data():                                 # 파일 → dict
    with DATA.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_data(data):                            # dict → 파일
    temp = DATA.with_suffix(".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        temp.replace(DATA)                       # 다 쓴 임시 파일로 바꿔치기
    finally:
        temp.unlink(missing_ok=True)             # 남은 임시 파일 정리
```

▶ 실제로 돌려 보면 임시 폴더의 storage.py + data/knowledge.json (처음 3건)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| storage.BASE.name, storage.DATA.relative_to(BASE) | 'single', 'data/knowledge.json' | storage.py가 있는 폴더 기준 경로 |
| data = read_data(); type(data), list(data) | dict, ['next_id', 'entries'] | JSON 글자 → 파이썬 dict |
| data["entries"][0]["title"] | '가상환경' | utf-8로 읽어 한글 그대로 |
| write_data(data) 후 파일 2~3번째 줄 | "next_id": 4, \| "entries": [ | indent=2 줄바꿈 |
| "가상환경" in 파일 내용 | True | ensure_ascii=False → 한글 그대로 저장 |
| json.dumps("가상환경") (ensure_ascii 기본값) | "\uac00\uc0c1\ud658\uacbd" | 기본값이면 \uXXXX 코드로 바뀜 |
| DATA.with_suffix(".tmp").exists() | False | 성공 후 임시 파일은 안 남음 |
| write_data({… set 값 …}) (저장 실패) | TypeError: Object of type set is not JSON serializable | json.dump가 중간에 실패 |
| 실패 후 원본 bytes == 실패 전 | True | 임시 파일에서 실패해서 원본은 그대로 |
| 실패 후 DATA.with_suffix(".tmp").exists() | False | finally에서 반쯤 쓴 임시 파일 정리 |

| 코드 | 뜻 |
|---|---|
| Path(__file__).resolve().parent | "이 코드 파일이 있는 폴더". 어느 폴더에서 서버를 켜든 같은 data 파일을 찾게 해 줘요 |
| encoding="utf-8" | 한글이 깨지지 않게 (Windows 기본값은 cp949라 꼭 적어요) |
| json.load(f) / json.dump(data, f) | JSON 글자 ↔ 파이썬 dict·list 변환 |
| ensure_ascii=False | 한글을 가 같은 코드 대신 그대로 저장 |
| indent=2 | 사람이 읽기 좋게 줄바꿈·들여쓰기 |
| with_suffix(".tmp") | 같은 폴더에 knowledge.tmp 경로 만들기 |

### with LOCK: — 읽기부터 저장까지 한 번에 잠그기

```
from storage import LOCK, read_data, write_data

with LOCK:                       # 이 블록은 한 번에 한 요청만
    data = read_data()           # 1. 읽고
    ...                          # 2. 바꾸고
    write_data(data)             # 3. 저장
```

▶ 실제로 돌려 보면 빈 파일에서 스레드 20개가 동시에 '읽기 → 0.01초 → 추가 → 저장'

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| read_data() … with LOCK: write_data() (저장만 잠금) | 20건 보냄 → 파일에 1건 · next_id 2 | 모두 빈 파일을 읽은 채 차례로 저장 → 서로 덮어씀 |
| with LOCK: read_data() … write_data() | 20건 보냄 → 파일에 20건 · next_id 21 | 앞 요청이 끝날 때까지 기다리니 전부 남음 |
| with LOCK: 안에서 LOCK.locked() | True | 누가 잡고 있는 중 |
| 블록을 나온 뒤 LOCK.locked() | False | with가 끝나면 자동으로 풀림 |

🤔 왜 쓰기 한 줄만 잠그면 안 돼? — 문제는 "예전 내용을 읽은 채로 저장하는 것"이에요. 쓰기만 잠그면 두 요청이 같은 옛날 내용을 읽은 뒤 차례로 저장해서 결국 덮어써요. 그래서 읽기 → 변경 → 저장을 하나의 with LOCK 안에 둬요. FastAPI는 def 함수도 여러 요청을 동시에 처리할 수 있어서 이게 필요해요.

⚠️ 이 Lock은 서버 프로세스 하나 안에서만 지켜 줘요. 서버를 여러 개 띄우거나, 실행 중에 편집기로 파일을 직접 고치거나, 여러 표를 한꺼번에 바꾸는 경우는 못 막아요. 그건 데이터베이스의 트랜잭션이 하는 일이에요(다음 수업).

⚠️ 파일 읽기에 실패하면 "빈 목록이겠지" 하고 빈 데이터로 덮어쓰면 안 돼요. 그 순간 지식이 전부 사라져요. 원인(경로·JSON 문법)을 먼저 확인해요.

### 목록 조회

```
@app.get("/api/knowledge")
def list_knowledge():
    with LOCK:
        data = read_data()          # 요청마다 파일을 새로 읽음
    return data["entries"]          # list를 돌려주면 JSON 배열
```

▶ 실제로 돌려 보면 처음 3건 파일

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.get("/api/knowledge") | 200 · 3건 · id [1, 2, 3] | list → JSON 배열 |
| r.json()[0] | {"id": 1, "title": "가상환경", "category": "setup", "content": "프로젝트별 패키지를 분리한다."} | 파일의 항목 그대로 |
| GET 전후 파일 bytes 같음? | True | GET은 읽기만 함 |
| 빈 파일에서 client.get("/api/knowledge") | 200 [] | 없는 건 오류가 아니라 빈 배열 |

| 요청 | 결과 |
|---|---|
| GET /api/knowledge | 200 + 지식 3개 배열 |
| 목록이 비어 있을 때 | 200 + [] (없는 게 오류는 아니에요) |

🤔 왜 요청마다 파일을 다시 읽어? — 한 번 읽어 변수에 들고 있으면, 다른 요청이 파일을 바꿔도 옛날 값을 보여 줘요. 매번 읽으면 항상 파일이 진짜 기준이 돼요. GET은 읽기만 하고 파일을 절대 바꾸지 않아요.

### 상세 조회 + 404

```
from fastapi import HTTPException

@app.get("/api/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):            # {knowledge_id} 자리 값을 int로 받음
    with LOCK:
        data = read_data()
    for entry in data["entries"]:
        if entry["id"] == knowledge_id:
            return entry                         # 찾으면 200
    raise HTTPException(404, "Knowledge not found")   # 끝까지 없으면 404
```

▶ 실제로 돌려 보면 처음 3건 파일

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.get("/api/knowledge/2") | 200 {"id":2,"title":"서버 실행","category":"api","content":"uvicorn으로 개발 서버를 켠다."} | 찾으면 200 + 그 항목 |
| client.get("/api/knowledge/9999") | 404 {"detail":"Knowledge not found"} | Route는 맞는데 그 ID가 없음 |
| client.get("/api/knowledge/foo") | 422 {"detail":[{"type":"int_parsing","loc":["path","knowledge_id"],"msg":"Input should be a va … | int로 못 바꿔 함수 실행 전에 422 |
| client.get("/nope") | 404 {"detail":"Not Found"} | Route 자체가 없음 — detail 문구가 다름 |

| 코드 조각 | API에서의 역할 |
|---|---|
| @app.get(".../{knowledge_id}") | Route — 어떤 요청을 받나 (중괄호 = Path Parameter 자리) |
| get_knowledge(...) | Function — 요청을 받으면 무엇을 하나 |
| knowledge_id: int | Parameter — 어떤 값을 받나 (이름이 중괄호 안 이름과 같아야 해요) |
| entry["id"] | Field — 어떤 자료와 비교하나 |

강의 코드를 그대로 돌려 받은 실제 응답이에요 (FastAPI 0.162 · Pydantic 2.14):

```
GET /api/knowledge/2     → 200 {"id":2,"title":"서버 실행","category":"api","content":"uvicorn으로 개발 서버를 켠다."}
GET /api/knowledge/9999  → 404 {"detail":"Knowledge not found"}
GET /api/knowledge/foo   → 422 {"detail":[{"type":"int_parsing","loc":["path","knowledge_id"],
                                 "msg":"Input should be a valid integer, unable to parse string as an integer","input":"foo"}]}
GET /nope                → 404 {"detail":"Not Found"}      ← 경로 자체가 없을 때
```

🤔 왜 오류는 return이 아니라 raise? — return {"error": ...}을 하면 상태 코드가 200(성공)으로 나가요. 받는 프로그램은 성공인 줄 알고 그 dict를 지식으로 써 버려요. raise HTTPException(404)은 함수를 즉시 멈추고 "실패, 이유는 이것"을 상태 코드로 알려요.

💡 404가 두 종류예요. {"detail":"Not Found"}는 그런 주소(Route)가 아예 없음(오타), {"detail":"Knowledge not found"}는 주소는 맞는데 그 ID의 지식이 없음. detail 문구로 구분해요.

### Query로 목록 거르기

```
@app.get("/api/knowledge")
def list_knowledge(category: str | None = None):     # 기본값이 있으면 → Query, 생략 가능
    with LOCK:
        data = read_data()
    result = []
    for entry in data["entries"]:
        if category is None or entry["category"] == category:
            result.append(entry)
    return result
```

▶ 실제로 돌려 보면 처음 3건 파일 (category: setup 1건, api 2건)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.get("/api/knowledge") | 200 · id [1, 2, 3] | category 생략 → None → 전체 |
| client.get("/api/knowledge?category=api") | 200 · id [2, 3] | 같은 분류만 |
| client.get("/api/knowledge?category=zzz") | 200 · id [] | 맞는 게 없으면 빈 배열 |
| q 버전: get("/api/knowledge?q=서버") | 200 · id [2] | 제목 '서버 실행'에 글자가 들어 있음 |
| q 버전: get("…?q=uvicorn") | 200 · id [2] | 본문에서 찾음 |
| q 버전: get("…?q=Uvicorn") | 200 · id [] | 대소문자가 다르면 못 찾음 |
| q 버전: get("…?q=개발 서버 켜는 법") | 200 · id [] | 뜻이 같아도 글자가 그대로 없으면 0건 |
| q 버전: get("…?q=" + "가" * 81) | 422 {"detail":[{"type":"string_too_long","loc":["query","q"],"msg":"String should ha … | max_length=80 넘으면 422 |

| 요청 | 실제 결과 |
|---|---|
| GET /api/knowledge | 전체 3건 |
| GET /api/knowledge?category=api | id 2, 3 — 2건 |
| GET /api/knowledge?category=zzz | 200 + [] |

🤔 FastAPI는 Path와 Query를 어떻게 구분해? — 함수 매개변수 이름이 경로의 {중괄호}에 있으면 Path, 없으면 Query로 봐요. = None처럼 기본값을 주면 "안 보내도 되는" 선택 조건이 돼요. 이 함수는 기존 목록 Route를 교체하는 거라, 같은 경로를 두 번 등록하지 않아요.

제목·본문에 특정 글자가 들어간 것을 찾는 방법도 있어요:

```
from fastapi import Query

@app.get("/api/knowledge")
def list_knowledge(q: str = Query("", max_length=80)):
    ...
        if q in entry["title"] or q in entry["content"]:    # "포함하나?" 단순 비교
            result.append(entry)
```

⚠️ ?q=서버는 글자가 그대로 들어 있는지만 봐요(대소문자 구분). "개발 서버 켜는 법"을 물어도 "uvicorn 실행"을 못 찾아요. 뜻이 비슷한 걸 찾는 의미 기반 검색(RAG·벡터 검색)은 이후 수업에서 배워요.

### Pydantic으로 입력 모양 정하기

```
from pydantic import BaseModel, Field

class KnowledgeWrite(BaseModel):                          # POST·PUT 입력
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=2000)
    category: str = Field(min_length=1, max_length=30)

class Knowledge(KnowledgeWrite):                          # 응답 = 입력 + id
    id: int
```

▶ 실제로 돌려 보면 main.py의 KnowledgeWrite · Knowledge를 직접 만들어 보기 — NEW = {"title": "실습 환경 안내", "content": "Python 3.10 이상을 사용한다.", "category": "setup"}

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| KnowledgeWrite(**NEW).model_dump() | {"title": "실습 환경 안내", "content": "Python 3.10 이상을 사용한다.", "category": "setup"} | 검사를 통과한 값 → dict |
| KnowledgeWrite(**NEW, id=1).model_dump() | ['category', 'content', 'title'] | 보낸 id는 버려짐 — 번호는 서버가 정함 |
| KnowledgeWrite(title="", …) | ValidationError: type='string_too_short', loc=('title',) | min_length=1 위반 |
| KnowledgeWrite(title="가" * 81, …) | ValidationError: type='string_too_long', loc=('title',) | max_length=80 위반 |
| KnowledgeWrite(title=123, …) | ValidationError: type='string_type', loc=('title',) | 숫자를 문자열로 바꿔 주지 않음 |
| KnowledgeWrite(content=…, category=…) (title 없음) | ValidationError: type='missing', loc=('title',) | 기본값이 없으니 필수 |
| Knowledge(**NEW, id=4).model_dump() | ['title', 'content', 'category', 'id'] | 응답 모델 = 입력 세 필드 + id |

| 인자 / 표현 | 뜻 |
|---|---|
| BaseModel | 상속하면 "이 모양의 데이터만 받겠다"는 검사 틀이 됨 |
| title: str | 문자열이어야 함. 기본값이 없으니 필수 필드 |
| Field(min_length=1) | 최소 1글자 — 빈 문자열 "" 거부 |
| Field(max_length=80) | 최대 길이. title 80 · content 2,000 · category 30자 |
| class Knowledge(KnowledgeWrite) | 입력 모델을 물려받고 id만 추가한 응답 모델 |
| payload.model_dump() | 검증된 모델 → 파이썬 dict (파일 저장은 안 함) |

🤔 왜 입력 모델과 응답 모델을 나눠? — 입력할 땐 id가 없어야 하고(서버가 정하니까), 응답할 땐 있어야 해요. 하나로 쓰면 사용자가 "id": 1을 보내 남의 번호를 흉내 낼 수 있어요. 그리고 이 모델은 DB 테이블이 아니에요 — 모양 검사만 하고, 저장은 다른 코드가 해요.

### 등록 함수

```
@app.post("/api/knowledge", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):        # Body를 KnowledgeWrite로 검사
    with LOCK:
        data = read_data()                            # 1. 현재 파일 읽기
        entry = payload.model_dump()                  # 2. 새 항목 dict 만들기
        entry["id"] = data["next_id"]                 #    + 새 ID
        data["entries"].append(entry)                 # 3. 목록에 추가
        data["next_id"] += 1                          # 4. 다음 번호 올리기
        write_data(data)                              # 5. 파일 저장
    return entry                                      # 6. 만든 항목 응답
```

▶ 실제로 돌려 보면 처음 3건 파일에 POST — NEW = {"title": "실습 환경 안내", "content": "Python 3.10 이상을 사용한다.", "category": "setup"}

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.post("/api/knowledge", json=NEW) | 201 {"title":"실습 환경 안내","content":"Python 3.10 이상을 사용한다.","category":"setup","id":4} | 201 Created + 새 id |
| sorted(r.json()) | ['category', 'content', 'id', 'title'] | response_model 덕분에 next_id는 응답에 안 섞임 |
| client.get("/api/knowledge/4") | 200 | 받은 ID로 다시 조회 |
| 목록 개수 | 3 → 4 |  |
| 파일 직접 확인 | 항목 4개 [1, 2, 3, 4] · next_id 5 | 파일에도 저장됨 |
| 같은 POST 한 번 더 | 201 · id 5 · 항목 5개 [1, 2, 3, 4, 5] · next_id 6 | POST는 보낼 때마다 새로 만듦 |

| 인자 | 뜻 |
|---|---|
| status_code=201 | 성공 시 200 대신 201 Created(새로 만들었음) |
| response_model=Knowledge | 응답 모양을 검사·정리하고 /docs에 표시. next_id 같은 내부 값이 새지 않게 걸러 줘요 |
| payload: KnowledgeWrite | 매개변수 타입이 BaseModel이면 → Body로 받아요 |

생성 확인은 응답 한 번으로 끝내지 않아요.

| 순서 | 확인 | 초기 3건에서 한 번 등록했을 때 |
|---|---|---|
| 1 | POST 응답 | 201, id = 4 |
| 2 | 그 ID로 상세 재조회 | GET /api/knowledge/4 → 200 |
| 3 | 목록 개수 | 3개 → 4개 |
| 4 | 파일 직접 열기 | id 4 항목 있음, next_id: 5 |

⚠️ /docs에서 Execute를 또 누르면 또 하나 생겨요(POST는 누를 때마다 새로 만듦). 응답에서 받은 실제 ID를 적어 두고 그걸로 확인해요.

### 422 — 잘못된 입력은 저장 전에 막힘

| 보낸 입력 | 실제 결과 | detail에서 볼 곳 |
|---|---|---|
| 정상 세 필드 | 201 | 새 ID가 붙은 항목 |
| title 빠짐 | 422 | "type":"missing", "loc":["body","title"] |
| content가 "" | 422 | "type":"string_too_short", "loc":["body","content"] |
| JSON 문법 오류 (끝에 쉼표) | 422 | "type":"json_invalid" |

▶ 실제로 돌려 보면 항목 4건 · next_id 5 파일에 잘못된 POST 세 번

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 시작 상태 | 항목 4개 [1, 2, 3, 4] · next_id 5 |  |
| post(json={"content": "x", "category": "setup"}) | 422 type='missing' loc=['body', 'title'] | title 빠짐 |
| post(json={**NEW, "content": ""}) | 422 type='string_too_short' loc=['body', 'content'] | 빈 문자열 — min_length=1 |
| post(content='{"title": "a",}') | 422 type='json_invalid' loc=['body', 13] | JSON 문법 오류 (끝에 쉼표) |
| e.json()["detail"][0]["msg"] (content "") | String should have at least 1 character | 고칠 곳을 알려 주는 문장 |
| 세 번 뒤 파일 | 항목 4개 [1, 2, 3, 4] · next_id 5 · bytes 같음 True | 함수 본문 전에 막혀 저장 코드가 안 돎 |

세 번 다 422가 난 뒤 파일을 확인하니 항목 4개 · next_id 5 그대로였어요. 검증은 함수 본문이 실행되기 전에 일어나서, 저장 코드까지 가지도 않아요.

🤔 loc은 어떻게 읽어? — 어디서 틀렸는지 알려 주는 주소예요. 첫 칸이 "path" / "query" / "body" 중 하나고, 다음 칸이 필드 이름이에요. 422가 나면 loc과 msg부터 보면 고칠 곳이 바로 나와요.

| 검사 종류 | 질문 | 누가 막나 |
|---|---|---|
| 형식 | 세 필드가 있나? 길이가 맞나? | Pydantic → 자동 422 (오늘) |
| 내용 | 안내 내용이 맞나? 오래된 절차 아닌가? | 사람의 검토 — 코드가 못 막아요 |
| 권한 | 아무나 고치고 지워도 되나? | 인증·권한 설계 (오늘은 로컬 학습용이라 없음) |

⚠️ 형식 통과 ≠ 사실 · 권한 보장. "content": "Python 2.7을 쓴다"도 1~2,000자라 통과해요. response_model도 응답 모양만 검사해요. 잘못된 입력(형식)과 부정확한 원문(내용)을 구분해야 나중에 Agent가 믿을 수 있는 자료를 써요.

### 수정 함수

```
@app.put("/api/knowledge/{knowledge_id}", response_model=Knowledge)
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):   # Path + Body 둘 다
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):      # i = 위치, entry = 항목
            if entry["id"] == knowledge_id:
                updated = payload.model_dump()
                updated["id"] = knowledge_id             # ID는 그대로
                data["entries"][i] = updated             # 그 위치에 덮어씀 (append 아님!)
                write_data(data)
                return updated
    raise HTTPException(404, "Knowledge not found")     # 못 찾으면 저장 없이 404
```

▶ 실제로 돌려 보면 항목 4건(id 4 = 실습 환경 안내) · next_id 5 — UPD = NEW에서 content만 "프로젝트별 가상환경을 사용한다."

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.put("/api/knowledge/4", json=UPD) | 200 {"title":"실습 환경 안내","content":"프로젝트별 가상환경을 사용한다.","category":"setup","id":4} | 그 자리만 교체 |
| PUT 뒤 파일 | 항목 4개 [1, 2, 3, 4] · next_id 5 | 개수·next_id 그대로 (append 아님) |
| 같은 PUT 다시 → 파일 bytes 같음? | 200 · True | 몇 번 보내도 결과가 같음 |
| put("/api/knowledge/4", json={title, content}) | 422 type='missing' loc=['body', 'category'] | PUT은 세 필드 다 — category 빠지면 422 |
| put("/api/knowledge/9999", json=UPD) | 404 {"detail":"Knowledge not found"} | 새로 만들지 않고 404 |
| 404 뒤 파일 | 항목 4개 [1, 2, 3, 4] · next_id 5 | 새 항목 안 생김 |
| client.patch("/api/knowledge/2", …) | 405 {"detail":"Method Not Allowed"} | PATCH Route는 없음 → 405 |

| 표현 | 뜻 |
|---|---|
| enumerate(list) | (위치, 값)을 같이 꺼내 줌 — 바꿀 위치가 필요해서 |
| data["entries"][i] = updated | 그 자리 항목을 교체. append()를 쓰면 수정이 아니라 추가가 돼요 |

| 요청 | 실제 결과 | 파일 |
|---|---|---|
| 있는 ID + 세 필드 | 200 + 수정된 항목 | 그 항목만 교체, 개수 4 · next_id 5 그대로 |
| 같은 PUT 다시 | 200 | 변화 없음 (몇 번 보내도 결과 같음) |
| category 빼고 | 422 missing | 기존 항목 그대로 |
| 없는 ID + 정상 Body | 404 | 새 항목이 생기지 않음 |

🤔 왜 없는 ID로 PUT 하면 새로 만들지 않고 404? — "수정하라"고 했는데 몰래 만들어 버리면, ID를 잘못 적은 실수가 조용히 새 지식으로 쌓여요. 만드는 건 POST, 고치는 건 PUT으로 역할을 나눠야 실수가 드러나요.

💡 PUT vs PATCH — PUT은 전체 교체(세 필드 다), PATCH는 바꿀 필드만 보내는 부분 수정이에요. PATCH는 payload.model_dump(exclude_unset=True)로 "보낸 필드만" 꺼낼 수 있지만, "안 보냄"과 "null로 보냄"을 구분하는 규칙을 따로 정해야 해서 오늘은 PUT만 써요. 이 API에 PATCH를 보내면 405 Method Not Allowed가 나와요.

### 삭제 함수

```
from fastapi import Response

@app.delete("/api/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    with LOCK:
        data = read_data()
        for i, entry in enumerate(data["entries"]):
            if entry["id"] == knowledge_id:
                data["entries"].pop(i)                 # 그 위치 한 건 제거
                write_data(data)
                return Response(status_code=204)       # 본문 없는 성공
    raise HTTPException(404, "Knowledge not found")
```

▶ 실제로 돌려 보면 항목 4건 · next_id 5 에서 id 4 삭제

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.delete("/api/knowledge/4") | 204 · r.content = b'' | 본문 없는 성공 |
| 삭제 뒤 파일 | 항목 3개 [1, 2, 3] · next_id 5 | 그 한 건만 빠지고 next_id는 그대로 |
| r.json() (204 응답에) | JSONDecodeError | 빈 본문은 JSON으로 못 읽음 → 상태 코드로 판단 |
| client.get("/api/knowledge/4") | 404 {"detail":"Knowledge not found"} | 재조회 404 |
| client.delete("/api/knowledge/4") (또) | 404 {"detail":"Knowledge not found"} | 이미 없음 |
| 삭제 뒤 새로 POST | 201 · id 5 | 지운 4를 다시 쓰지 않고 5 |

| 표현 | 뜻 |
|---|---|
| list.pop(i) | i번째 위치의 원소 하나를 빼냄 |
| Response(status_code=204) | 본문 없는 HTTP 응답. JSON을 같이 보내지 않아요 |

```
DELETE /api/knowledge/4   → 204  (빈 본문)
GET    /api/knowledge/4   → 404  {"detail":"Knowledge not found"}
DELETE /api/knowledge/4   → 404  (이미 없음)
```

🤔 왜 204엔 {"message":"deleted"}를 안 붙여? — 204는 이름부터 "No Content(보낼 내용 없음)"예요. 성공했다는 건 숫자 204가 이미 말해 줘요. 받는 쪽 코드도 빈 204에 response.json()을 호출하면 오류가 나니까, 상태 코드만 보고 판단해요.

### app/main.py — 조립

```
from fastapi import FastAPI
from app.routers.knowledge import router

app = FastAPI(title="Knowledge API")
app.include_router(router)                 # Router에 모아 둔 경로들을 앱에 연결

@app.get("/health")
def health():
    return {"status": "ok"}
```

▶ 실제로 돌려 보면 임시 fastapi_knowledge/ 폴더 (app/ 패키지 + data/knowledge.json 3건)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| from app.main import app (프로젝트 최상위에서) | 성공 | app 폴더를 패키지로 찾음 |
| client.get("/openapi.json").json()["paths"] | {"/api/knowledge": ["get", "post"], "/api/knowledge/{knowledge_id}": ["delete", "get", "put"], "/health": ["get"]} | include_router로 Router의 Route들이 붙음 |
| client.get("/health") | 200 {"status":"ok"} | 앱만 대답하면 200 |
| client.get("/api/knowledge") | 200 · 3건 | Router 쪽 Route도 동작 |
| app/ 안에서 python -m uvicorn app.main:app | ModuleNotFoundError: No module named 'app' | 종료 코드 1 — 실행 위치는 최상위 |

```
# 프로젝트 최상위(fastapi_knowledge/)에서, 한 줄로
python -m uvicorn app.main:app --reload
```

| 부분 | 뜻 |
|---|---|
| app.main | app/main.py (폴더는 점으로 이어요) |
| :app | 그 파일 안의 앱 변수 |

⚠️ 실행 위치는 프로젝트 최상위예요. app/ 안에 들어가서 실행하면 from app.routers ...를 못 찾아요(ModuleNotFoundError). 그리고 /health가 200이라고 데이터 파일까지 준비됐다는 뜻은 아니에요 — 서버가 대답만 하는지 보는 단순 점검이에요.

### APIRouter — 관련 경로 묶기

```
# app/routers/knowledge.py
from fastapi import APIRouter
from app.schemas import Knowledge, KnowledgeWrite
from app.repositories import json_knowledge as repo

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

@router.post("", status_code=201, response_model=Knowledge)
def create_knowledge(payload: KnowledgeWrite):
    return repo.create(payload.model_dump())       # 저장은 Repository에 맡김
```

▶ 실제로 돌려 보면 app/routers/knowledge.py의 router

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| router.prefix, router.tags | '/api/knowledge', ['knowledge'] |  |
| [r.path for r in router.routes] | ['/api/knowledge', '/api/knowledge/{knowledge_id}'] | prefix + "" / prefix + "/{knowledge_id}" |
| openapi["paths"]["/api/knowledge"]["post"]["tags"] | ['knowledge'] | /docs에서 묶어 보여 줄 그룹 이름 |
| client.post("/api/knowledge", json=NEW) | 201 {"title":"실습 환경 안내","content":"Python 3.10 이상을 사용한다.","category":"setup","id":4} | Router → repo.create → 201 |
| 옛 @app.post를 지우지 않고 include_router → POST | 200 {"from":"main.py에 남은 옛 코드"} | 먼저 등록된 옛 함수가 받음 (201도 아님) |

| 인자 | 뜻 |
|---|---|
| prefix="/api/knowledge" | 이 Router의 모든 경로 앞에 붙는 공통 부분. prefix + "" → /api/knowledge, prefix + "/{knowledge_id}" → /api/knowledge/{knowledge_id} |
| tags=["knowledge"] | /docs에서 묶어 보여 줄 그룹 이름일 뿐. 인증·권한과 무관 |

⚠️ 옮긴 뒤에는 main.py에 남아 있던 같은 @app.post("/api/knowledge")를 지워야 해요. 같은 Route가 두 번 등록되면 먼저 등록된 쪽이 받아서, 고친 코드가 안 도는 것처럼 보여요.

### Repository — 저장 로직 그대로 옮기기

```
# app/repositories/json_knowledge.py (일부)
# LOCK · read_data · write_data 도 이 파일에서 정의
BASE = Path(__file__).resolve().parents[2]       # 프로젝트 루트 (repositories → app → 루트)
DATA = BASE / "data" / "knowledge.json"

def create(fields: dict) -> dict:
    with LOCK:
        data = read_data()
        entry = dict(fields)
        entry["id"] = data["next_id"]
        data["entries"].append(entry)
        data["next_id"] += 1
        write_data(data)
    return entry
```

▶ 실제로 돌려 보면 app/repositories/json_knowledge.py

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| Path(__file__).resolve().parents[0].name | 'repositories' |  |
| Path(__file__).resolve().parents[1].name | 'app' |  |
| Path(__file__).resolve().parents[2].name | 'fastapi_knowledge' | 프로젝트 루트 |
| repo.DATA.relative_to(repo.BASE) | 'data/knowledge.json' | 루트 아래 data 파일을 그대로 가리킴 |
| repo.create({title, content, category}) | {"title": "실습 환경 안내", "content": "Python 3.10 이상을 사용한다.", "category": "setup", "id": 4} | HTTP 없이 저장만 — 순서는 main.py 때와 같음 |
| 파일 확인 | 항목 4개 [1, 2, 3, 4] · next_id 5 | id 4 · next_id 5 |

🤔 왜 parent가 parents[2]로 바뀌어? — 코드가 놓인 파일이 app/repositories/ 안으로 두 단계 깊어졌으니까요. parents[0]=repositories, [1]=app, [2]=프로젝트 루트. 바뀐 건 파일 위치뿐이고 ID 발급·추가·저장 순서는 똑같아요.

💡 나중에 RAG를 붙이면 "질의 Router → RAG 처리 함수 → 검색 + LLM" 길이 따로 생겨요. Repository에는 DB·파일 접근만 남겨요. 오늘은 Service 계층을 새로 만들지 않아요 — 책임이 다른 기능이 실제로 생길 때 나눠요.

### 같은 데이터를 HTML 화면으로 (Template)

```
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates      # pip install jinja2

templates = Jinja2Templates(directory=BASE / "templates")

@app.get("/web/knowledge", response_class=HTMLResponse)
def knowledge_page(request: Request):
    with LOCK:
        data = read_data()
    return templates.TemplateResponse(
        request=request, name="knowledge.html",
        context={"entries": data["entries"]})
```

▶ 실제로 돌려 보면 main.py + templates/knowledge.html, 처음 3건 파일

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| client.get("/web/knowledge").headers["content-type"] | 200 text/html; charset=utf-8 | 사람이 보는 화면 |
| 응답 본문의 <li> 내용 | ['가상환경', '서버 실행', '입력 오류'] | {% for %}가 entries 3건만큼 반복 |
| client.get("/api/knowledge").headers["content-type"] | 200 application/json | 같은 파일, 프로그램용 JSON |
| client.get("/api/knowledge").json()[0]["title"] | '가상환경' | 값을 바로 꺼내 쓸 수 있음 |

```
<!-- templates/knowledge.html -->
{% for entry in entries %}
  <li>{{ entry.title }}</li>
{% endfor %}
```

|  | JSON API | HTML Template |
|---|---|---|
| 주소 | GET /api/knowledge | GET /web/knowledge |
| Content-Type | application/json | text/html |
| 누가 쓰나 | 프로그램 (다른 서버·Agent·관리 화면) | 사람 (브라우저 화면) |

🤔 왜 둘 다 있어? — 둘은 같은 파일을 읽어요. 다만 프로그램은 값을 꺼내 쓰기 쉬운 JSON이 필요하고, 사람은 보기 좋은 화면이 필요해요. 오늘 만드는 건 JSON API이고, Template은 비교용이에요.

### 자동 테스트 — 실제 지식 파일은 건드리지 않게

```
# tests/test_knowledge.py      (pip install pytest httpx)
from fastapi.testclient import TestClient
from app.main import app
from app.repositories import json_knowledge as repo

def test_missing(tmp_path, monkeypatch):
    test_file = tmp_path / "knowledge.json"
    test_file.write_text('{"next_id":1,"entries":[]}', encoding="utf-8")
    monkeypatch.setattr(repo, "DATA", test_file)       # 저장 경로만 임시 파일로 바꿔치기
    before = test_file.read_bytes()
    with TestClient(app) as client:
        r = client.delete("/api/knowledge/9999")
    assert r.status_code == 404
    assert test_file.read_bytes() == before            # 실패 후 파일이 그대로인지
```

▶ 실제로 돌려 보면 임시 fastapi_knowledge/ 에 tests/test_knowledge.py를 만들고 실행

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| python -m pytest -q (프로젝트 최상위) | 1 passed, 1 warning in …s | 종료 코드 0 — 테스트 통과 |
| 테스트 뒤 실제 data/knowledge.json bytes 같음? | True | monkeypatch로 임시 파일만 썼음 |
| 실제 파일 | 항목 3개 [1, 2, 3] · next_id 4 | 처음 3건 그대로 |

| 표현 | 뜻 |
|---|---|
| TestClient(app) | 서버를 띄우지 않고 앱에 바로 요청을 보내는 가짜 클라이언트 |
| tmp_path | pytest가 테스트마다 만들어 주는 빈 임시 폴더 |
| monkeypatch.setattr(repo, "DATA", ...) | 테스트하는 동안만 repo.DATA 값을 바꿈 (끝나면 원래대로) |
| assert | 조건이 거짓이면 테스트 실패 |

```
python -m pytest -q
```

🤔 왜 임시 파일로 바꿔? — 테스트는 일부러 등록·삭제를 마구 해요. 실제 knowledge.json으로 돌리면 테스트할 때마다 진짜 지식이 지워지고 섞여요. 저장 위치를 Repository 한 곳(DATA)에 모아 둔 덕분에 그 한 줄만 바꿔치기하면 돼요. 단, 서버 재시작 확인은 테스트가 대신 못 해서 따로 해요.

## 📖 Day17 용어 (38개)
- **FastAPI** (FastAPI (Python 웹 API 프레임워크)): '이 주소로 이런 요청이 오면 이 함수를 실행해'를 연결해 주는 Python 도구. 입력 검증과 자동 문서(/docs)도 만들어 줘요.
- **Uvicorn** (Uvicorn (ASGI 서버)): 네트워크에서 요청을 실제로 받아 FastAPI 앱에 넘겨주는 서버 프로그램. python -m uvicorn main:app 으로 실행해요.
- **API** (Application Programming Interface): 프로그램끼리 주고받는 약속된 창구. 어떤 주소에 어떤 형식으로 요청하면 어떤 형식으로 응답하는지 정해 둔 것.
- **HTTP** (HyperText Transfer Protocol): 브라우저·프로그램이 서버와 대화하는 규칙. 항상 요청 1개와 응답 1개가 한 쌍이에요.
- **CRUD** (Create · Read · Update · Delete): 데이터를 만들고(POST)·읽고(GET)·고치고(PUT)·지우는(DELETE) 네 가지 기본 동작.
- **HTTP Method** (HTTP Method (GET · POST · PUT · DELETE · PATCH)): 요청이 무엇을 하려는지 나타내는 동사. 같은 주소라도 Method가 다르면 다른 함수가 실행돼요.
- **Route** (Route (Method + Path → 함수 연결)): 'GET + /api/knowledge/{id}' 같은 Method와 경로 조합을 처리 함수에 연결해 둔 것. @app.get(...)으로 등록해요.
- **Path Parameter** (Path Parameter (경로 값)): URL 경로 안에 들어가는 값. /api/knowledge/2 의 2처럼 대상 하나를 콕 집을 때 써요. 함수 인자 이름이 {중괄호} 이름과 같아야 해요.
- **Query Parameter** (Query Parameter (? 뒤의 조건)): URL 끝 ?category=api 처럼 붙이는 선택 조건. 주로 목록을 거를 때 쓰고, 기본값이 있으면 생략할 수 있어요.
- **Request Body** (Request Body (요청 본문)): POST·PUT에서 새로 만들거나 고칠 데이터를 JSON으로 담아 보내는 부분. GET에는 보통 없어요.
- **Content-Type** (Content-Type 헤더): 본문이 어떤 형식인지 알려 주는 Header. JSON이면 application/json, HTML 화면이면 text/html.
- **상태 코드** (HTTP Status Code): 처리 결과를 세 자리 숫자로 알려 줘요. 2xx 성공, 4xx 요청 쪽 문제(주소·입력), 5xx 서버 쪽 문제.
- **201 Created**: 새 자원을 만들었다는 성공 코드. POST로 지식을 등록했을 때 써요.
- **204 No Content**: 성공했지만 돌려줄 본문이 없다는 코드. 삭제 성공에 쓰고 JSON을 같이 보내지 않아요.
- **404 Not Found**: 그런 주소(Route)가 없거나, 주소는 맞는데 그 ID의 자원이 없다는 코드. detail 문구로 둘을 구분해요.
- **405 Method Not Allowed**: 경로는 있는데 그 Method는 지원하지 않는다는 코드. 예: PUT만 만든 경로에 PATCH를 보냄.
- **422 Unprocessable Content**: 요청의 자료형·필수값·길이 규칙이 틀렸다는 코드. FastAPI가 함수 실행 전에 자동으로 내요. detail의 loc·msg로 위치를 확인해요.
- **HTTPException** (fastapi.HTTPException): raise 하면 함수를 즉시 멈추고 지정한 상태 코드와 detail을 응답으로 보내요. 오류를 return으로 보내면 200이 되니 꼭 raise로.
- **Pydantic** (Pydantic (데이터 검증 라이브러리)): class로 데이터 모양(필드·타입·길이)을 정해 두면 들어온 값을 검사해 줘요. FastAPI는 이걸로 Body를 검사하고 틀리면 422를 내요.
- **Field** (pydantic.Field): 필드에 min_length·max_length 같은 규칙을 붙여요. 기본값을 주지 않으면 필수 필드예요.
- **model_dump** (BaseModel.model_dump()): 검증된 Pydantic 모델을 파이썬 dict로 바꿔 줘요. exclude_unset=True면 실제로 보낸 필드만 꺼내요.
- **response_model** (response_model= (응답 모델)): 응답의 모양을 검사·정리하고 /docs에 표시해요. 정의에 없는 필드(next_id 등)는 걸러져요. 사실성이나 권한은 보증하지 않아요.
- **Swagger UI** (Swagger UI (FastAPI 자동 API 문서)): /docs 주소에 자동으로 생기는 API 문서 화면. Try it out → Execute로 POST·PUT·DELETE도 직접 보낼 수 있어요.
- **데코레이터** (Decorator (@)): 함수 위에 @이름 을 붙여 그 함수에 기능이나 등록 정보를 덧씌우는 문법. @app.get(...)은 '이 함수를 이 주소에 등록'.
- **localhost** (localhost (127.0.0.1)): '지금 이 컴퓨터 자신'을 가리키는 주소. 여기서 띄운 서버는 기본적으로 내 PC에서만 접속돼요.
- **포트** (Port (포트 번호)): 한 컴퓨터 안에서 프로그램마다 다른 문 번호. 127.0.0.1:8000 의 8000. 같은 포트로 서버를 두 개 띄울 수 없어요.
- **next_id** (next_id (다음에 줄 ID)): 다음 새 항목에 줄 번호를 따로 저장한 값. 삭제해도 줄이지 않아 지운 번호를 다시 쓰지 않아요. entries와 같은 파일에 저장해요.
- **Lock** (threading.Lock (잠금)): with LOCK: 블록 안은 한 번에 한 요청만 실행되게 해요. 읽기→변경→저장을 통째로 묶어야 서로의 저장을 덮어쓰지 않아요.
- **원자적 저장** (Atomic write (임시 파일에 쓰고 교체)): 임시 파일에 끝까지 쓴 뒤 replace()로 원본과 바꿔치기해요. 쓰다 실패해도 원본은 멀쩡해요.
- **APIRouter** (fastapi.APIRouter): 관련 경로를 한 파일에 묶는 도구. prefix로 공통 주소를 붙이고, app.include_router()로 앱에 연결해요. tags는 /docs 그룹 이름일 뿐이에요.
- **Repository** (Repository 패턴): 데이터 읽기·저장만 맡는 코드 묶음. 저장 방식을 파일에서 DB로 바꿀 때 이 부분만 고치면 돼요.
- **Jinja2** (Jinja2 템플릿 엔진): HTML 틀에 {{ entry.title }} 같은 자리를 두고 데이터를 채워 화면을 만들어요. JSON API와 같은 데이터를 사람용 화면으로 보여 줄 때 써요.
- **TestClient** (fastapi.testclient.TestClient): 서버를 띄우지 않고 앱에 바로 요청을 보내 응답을 확인하는 테스트용 클라이언트.
- **monkeypatch** (pytest monkeypatch): 테스트하는 동안만 변수·함수를 다른 값으로 바꿔치기하고, 끝나면 원래대로 돌려놔요. 저장 경로를 임시 파일로 바꿀 때 써요.
- **트랜잭션** (Transaction (트랜잭션)): 여러 변경을 '모두 성공 아니면 모두 취소'로 묶는 데이터베이스 기능. 파일 + Lock으로는 대신할 수 없어요.
- **PostgreSQL** (PostgreSQL (관계형 데이터베이스)): 표·관계·SQL로 데이터를 저장하는 데이터베이스. 동시 요청·여러 서버·트랜잭션처럼 JSON 파일로 어려운 일을 맡아요.
- **AI Agent**: LLM이 작업을 보고 필요한 도구를 골라 호출을 제안하고 결과를 확인하는 시스템. 실제 호출·권한 확인은 애플리케이션이 책임져요.
- **RAG** (Retrieval-Augmented Generation (검색 증강 생성)): 질문과 관련된 근거 문서를 먼저 검색해 LLM에 함께 넣어 답하게 하는 방식. 원본 지식이 정리돼 있어야 시작할 수 있어요.

---

# Day18 · PostgreSQL 입문 · 지식 관리 DB
_2026-11-11(수) · PostgreSQL · SQL · PK/FK · JOIN · 집계 · 제약조건 · 트랜잭션 · 인덱스_  ·  페이지: https://yiji-1015.github.io/sesac-notes/notes/day18.html

📌 딱 3줄 요약

- 어제 JSON 파일 하나에 넣던 지식을 PostgreSQL의 표 두 개(categories 분류 · knowledge 지식)로 나누고, PK·FK로 관계를 묶어요.

- SQL로 읽고(SELECT·WHERE·JOIN·GROUP BY) 바꿔요(INSERT·UPDATE·DELETE + RETURNING).

- 잘못된 값은 제약조건이 거부하고, 여러 변경은 트랜잭션(BEGIN → COMMIT 또는 ROLLBACK)으로 묶어 확정하거나 없던 일로 해요.

🔧 Day18 명령어 한눈에 — 이론 구역별 SQL·인자 설명 명령어 사전에서 보기 →

본문에는 핵심 SQL 한두 줄만 남겼어요. 명령어 사전의 결과값은 같은 S0 데이터를 실제로 실행해 얻은 거예요.

## 📦 필요한 프로그램

오늘은 pip install이 없어요. 파이썬이 아니라 DB 서버 프로그램을 설치하고, 그 서버에 SQL을 보내는 접속 도구를 써요.

| 프로그램 | 뭐 하는 애? | 어디서 써? |
|---|---|---|
| PostgreSQL 18 (Server) | 데이터를 실제로 저장하고 SQL·규칙을 처리하는 DBMS. 수업 기준은 18 계열 | 설치하면 Windows 서비스로 뒤에서 계속 켜져 있어요 |
| pgAdmin 4 | 창(GUI)으로 서버에 접속해 SQL을 보내고 결과를 보는 클라이언트 | Query Tool에서 SQL 실행 → Data Output·Messages 확인 |
| psql (Command Line Tools) | 터미널에서 쓰는 클라이언트 | PowerShell에서 psql -U postgres -d knowledge_lab |
| psycopg 등 (내일) | 파이썬(FastAPI)에서 PostgreSQL에 접속하는 드라이버 | 오늘은 안 씀 — 11/12 연동 때 |

### 🔧 설치 · 확인 (PowerShell)

```
# 공식 Windows 안내 → EDB 설치 프로그램 (Server · pgAdmin · Command Line Tools 선택)
psql --version                         # 클라이언트(psql) 버전
Get-Service postgresql*                # 서버가 서비스로 Running 인지
psql -U postgres -h localhost -p 5432  # 접속 → 비밀번호 입력 → postgres=# 프롬프트
```

⚠️ 서버 비밀번호는 비밀이에요. 소스 코드·Git·제출물·스크린샷에 실제 값을 남기지 않아요. 그리고 pgAdmin 창을 닫는 것 ≠ PostgreSQL 서버를 끄는 것 — 서버는 서비스로 계속 돌아요.

🤔 왜 psql --version과 SELECT version()이 따로 있어? — 앞의 것은 내 컴퓨터의 접속 도구 버전, 뒤의 것은 접속한 서버의 버전이에요. 다른 컴퓨터의 서버에 붙으면 둘이 다를 수 있어요.

## 이거 어디에 써?

신입이 "개발환경 설정 절차 알려 주세요"라고 물으면 나중엔 AI Agent가 지식 조회 도구로 안내문을 찾아 답해요. 그런데 안내문이 바뀌었는데 예전 내용을 읽어 가면 안 되죠. 그래서 검색·생성 이전에, 사람이 관리하는 원본 지식이 정확하고 추적 가능해야 해요. 오늘은 그 원본을 표와 규칙으로 관리해요.

🗂️

원본 지식 저장제목·본문·분류를 규칙과 함께 보관

🔗

관계 관리분류 이름을 한 곳에서 관리하고 번호로 연결

🛡️

안전한 변경잘못된 값 거부 · 실패하면 통째로 취소

🔎

추적Agent가 어떤 ID의 안내를 참고했는지 남길 근거

[그림: AI Agent 서비스, FastAPI, PostgreSQL 서버, knowledge_lab·public, categories·knowledge 두 테이블, 한 행의 category_id 값까지 왼쪽에서 오른쪽으로 확대해 가는 그림. 가운데 세 칸이 오늘 다루는 곳으로 초록 점선 안에 있다] 무엇을 보나: 왼쪽 서비스에서 오른쪽 값 하나까지 점점 확대했어요. 초록 점선 안(서버 → DB·스키마 → 테이블)이 오늘 SQL로 직접 만지는 곳이에요. 아래 회색 점선의 검색·RAG·Agent는 이 원본을 읽어 가는 별도 계층이고 이후 수업이에요. 🤔 왜 이렇게 나눠 봐? 문제가 생겼을 때 "서버에 못 붙은 건지, 다른 DB에 붙은 건지, 표가 없는 건지, 값이 틀린 건지"를 층별로 좁힐 수 있어요.

|  | 오늘 · PostgreSQL SQL / Index | 이후 · Embedding / Vector Search |
|---|---|---|
| 찾는 방법 | WHERE category_id = 1처럼 조건이 정확히 맞는 행 | 질문과 뜻이 가까운 문서 조각 |
| 인덱스 | B-tree 인덱스 — 조건에 맞는 행을 빨리 찾는 길 | 벡터 인덱스 — 가까운 벡터를 빨리 찾는 길 |
| 책임 | 원본의 저장 규칙·관계·변경 확정 | 근거 후보를 골라 LLM에 전달 |

⚠️ B-tree 인덱스를 만들었다고 의미 검색이 되는 건 아니에요. LIKE '%실행%'도 글자 포함 검색일 뿐이라 "구동"은 못 찾아요. 그리고 원본을 고치거나 지워도 나중의 벡터 인덱스는 자동으로 안 바뀌어요.

## ① DB · DBMS · 접속 도구

| 용어 | 오늘의 예 | 역할 |
|---|---|---|
| Database | knowledge_lab | 지식·분류의 구조와 데이터를 담는 업무 공간 |
| DBMS / RDBMS | PostgreSQL | 표 형태의 데이터를 관리하고 SQL·규칙을 처리하는 프로그램 |
| Client | pgAdmin / psql | 서버에 SQL을 보내고 결과를 받는 도구 |
| SQL | SELECT · INSERT … | 구조를 정의하고 데이터를 읽고 바꾸는 언어 |

이름 공간은 Server ⊃ Database ⊃ Schema ⊃ Table이에요. knowledge_lab에 접속하면 public 스키마의 knowledge 표를 public.knowledge(줄여서 knowledge)로 불러요. Schema는 같은 DB 안에서 표 이름을 묶는 이름표지 디스크 폴더가 아니에요.

| 접속 값 | 예시 | 뜻 |
|---|---|---|
| Host | localhost | 지금 이 컴퓨터의 서버 |
| Port | 5432 | 설치할 때 정한 서버 포트 (기본값) |
| Database | knowledge_lab | 오늘의 개인 학습 DB |
| User | postgres | 설치 확인용 관리자 계정. 실제 앱은 따로 만든 계정을 써요 |
| Password | (본인이 정한 값) | 서버 인증 · 결과물에는 숨겨요 |

🤔 왜 SELECT 1만 보면 안 돼? — 1이 나오면 "어떤 서버엔가 붙었다"는 것만 알려 줘요. 실수로 postgres DB에 붙은 채 표를 만들면 나중에 knowledge_lab에서 "relation does not exist"가 나요. 그래서 첫 SQL은 current_database()로 지금 DB 이름부터 봐요. 그리고 pgAdmin의 서버 등록 이름은 화면에 보이는 별칭일 뿐이라, 그 이름을 적는다고 DB가 생기지 않아요.

🔧 첫 SQL — 지금 어느 DB인가 명령어 사전에서 자세히 →

```
SELECT current_database() AS db_name, current_user AS user_name;
SELECT 1 AS connection_ok;
```

🔧 CREATE DATABASE — 개인 학습 DB 만들기 명령어 사전에서 자세히 →

```
CREATE DATABASE knowledge_lab;      -- postgres DB에서 이 문장만 단독 실행
```

🔧 psql — 터미널 접속과 \dt · \d 명령어 사전에서 자세히 →

```
psql -U postgres -h localhost -p 5432 -d knowledge_lab
\conninfo    \dt    \d knowledge    \q
```

### pgAdmin Query Tool 읽는 법

| 순서 | 할 일 |
|---|---|
| ① | 실행할 SQL만 선택해서 실행 (한 문장씩) |
| ② | Data Output에서 값과 행 수 확인 |
| ③ | Messages에서 오류·변경된 행 수(UPDATE 1 등) 확인 |
| 설정 | 변경 실습 때 Auto commit 켬 · Auto rollback on error 끔. BEGIN부터 끝까지 같은 Query Tool 창에서 |

🤔 결과가 0행이면 실패야? — 아니에요. "조건에 맞는 행이 없다"는 정상 결과예요. 오류는 Messages에 빨간 메시지로 따로 나와요. 그리고 DB를 직접 실행할 때는 HTTP 상태 코드가 없어요 — 200·404는 FastAPI의 일이에요.

## ② 지식 조회 — SELECT · WHERE · ORDER BY

오늘 새 DB에 미리 넣어 둔 가상 초기 데이터를 S0라고 불러요. (어제 JSON 파일이 자동으로 옮겨진 게 아니에요.)

| categories · 분류 3건 |  |
|---|---|
| id | name |
| 1 | learning |
| 2 | environment |
| 3 | reference |

| knowledge · 지식 3건 (content 생략) |  |  |
|---|---|---|
| id | title | category_id |
| 1 | 실습 파일 보관 | 1 |
| 2 | 서버 실행 | 2 |
| 3 | 질문 작성 | 1 |

한 행(Row) = 지식 한 건, 한 열(Column) = 공통 항목 한 종류, 값 = 그 행·열이 만나는 칸이에요. 지식 2번 행의 category_id 값 2는 "분류 2번(environment)"을 가리켜요. 그리고 reference(3번)를 가리키는 지식은 아직 없어요.

### SQL은 업무 질문으로 읽어요

"어느 표에서 어떤 행을 골라 어떤 열을 어떤 순서로 보고 싶은가?"

| 순서 | 절 | 질문 |
|---|---|---|
| 1 | FROM | 어느 테이블? |
| 2 | WHERE | 어떤 조건의 행? |
| 3 | SELECT | 어떤 열? |
| 4 | ORDER BY | 어떤 순서? (DESC = 큰 것부터) · LIMIT n = 최대 n행 |

🤔 쓰는 순서(SELECT가 먼저)와 왜 달라? — 위 순서는 뜻을 이해하는 순서예요. 실제로 DB가 어떤 순서·방법으로 처리할지는 실행계획이 정해요(⑧). 그리고 ORDER BY가 없으면 결과 순서는 보장되지 않아요 — 순서가 중요하면 꼭 써요.

🔧 SELECT · FROM · ORDER BY 명령어 사전에서 자세히 →

```
SELECT id, title, category_id FROM knowledge ORDER BY id;
```

🔧 WHERE — ID·분류 조건 명령어 사전에서 자세히 →

```
SELECT id, title FROM knowledge WHERE id = 2;
SELECT id, title FROM knowledge WHERE category_id = 1 ORDER BY id;
```

🔧 AND · LIKE · DESC · LIMIT 명령어 사전에서 자세히 →

```
WHERE category_id = 1 AND title LIKE '%실습%'
ORDER BY id DESC LIMIT 2;
```

### NULL과 빈 문자열은 달라요

|  | NULL | '' (빈 문자열) |
|---|---|---|
| 뜻 | 값이 없음 / 모름 | 길이가 0인 값 |
| 확인 | IS NULL (= NULL은 결과가 NULL이라 안 됨) | char_length(x) = 0 |
| 막는 규칙 | NOT NULL | 길이 CHECK (1자 이상) |

🤔 왜 규칙이 두 개나 필요해? — NOT NULL만 걸면 빈 본문 ''은 "값이 있음"이라 통과해요. 본문이 비면 Agent가 근거로 쓸 게 없으니 길이 CHECK도 같이 걸어요.

🔧 IS NULL · char_length() 명령어 사전에서 자세히 →

```
SELECT NULL IS NULL, '' IS NULL, char_length('');   -- true, false, 0
```

## ③ 두 테이블 설계 — PK · FK · ERD · DDL

[그림: 왼쪽에 어제의 JSON 한 건, 가운데 categories 표(1 learning, 2 environment, 3 reference), 오른쪽 knowledge 표의 category_id 값에서 categories의 해당 행으로 화살표가 이어진 ERD] 무엇을 보나: 어제 JSON은 지식마다 "category": "environment" 같은 분류 이름을 반복해서 들고 있었어요. 오늘은 분류 이름을 부모 표(categories)에 한 번만 두고, 지식은 category_id 번호로 가리켜요(초록 화살표). 분류 하나에 지식이 0개 이상(1 : 0..N) — reference처럼 0개여도 돼요. 🤔 왜 나눠? 분류 이름을 고칠 때 한 곳만 바꾸면 되고, 오타 난 분류("enviroment")가 몰래 생기는 걸 FK가 막아 줘요. 대신 볼 때는 JOIN으로 다시 붙여요(⑤).

| 개념 | 하는 일 | 이번 예시 |
|---|---|---|
| PK (Primary Key) | 각 행을 유일하게 식별 · NULL 불가 | knowledge.id, categories.id |
| FK (Foreign Key) | 참조하는 부모 값이 실제로 있는지 검사 | knowledge.category_id → categories.id |
| IDENTITY | 값을 생략하면 DB가 번호를 발급 | id 자동 생성 (유일성 보장은 PK의 일) |
| JOIN | 조회할 때 조건에 맞는 행을 붙여 보여 줌 | 지식 목록에 분류 이름 붙이기 |

⚠️ PK · FK · JOIN은 서로 다른 일이에요. FK가 있다고 JOIN이 자동으로 되지 않고, JOIN은 FK가 없어도 쓸 수 있어요. FK는 저장할 때 검사, JOIN은 읽을 때 연결이에요. 그리고 ID는 행 수나 목록의 위치가 아니에요 — 삭제·취소가 있으면 번호가 비어요(⑦).

### 자료형과 저장 규칙 (데이터 계약)

| 필드 | DB 표현 | 규칙 |
|---|---|---|
| id | BIGINT + IDENTITY + PK | DB가 발급 · 응답의 실제 ID로 추적 |
| title | VARCHAR(80) + NOT NULL + CHECK | 1~80자 · 같은 제목 허용 (UNIQUE 아님) |
| content | TEXT + NOT NULL + CHECK | 1~2,000자 |
| categories.name | VARCHAR(30) + NOT NULL + UNIQUE | 1~30자 · 분류 이름 중복 금지 |
| category_id | BIGINT + NOT NULL + FK | 등록된 분류만 참조 |

🤔 왜 제목엔 UNIQUE를 안 걸어? — 어제 API도 같은 제목을 막지 않았어요. 규칙을 DB로 옮기면서 몰래 바꾸면, 어제는 되던 요청이 오늘은 실패해요. 그래서 "제목은 ID 대신 쓸 수 없다 — 찾을 땐 ID로"가 원칙이에요. 반대로 분류 이름은 하나여야 이름으로 ID를 찾을 수 있어서 UNIQUE예요.

💡 DDL(Data Definition Language)은 CREATE TABLE처럼 구조를 만드는 SQL, DML은 INSERT·UPDATE처럼 데이터를 바꾸는 SQL이에요. 부모(categories)를 먼저 만들어야 자식의 FK가 가리킬 곳이 생겨요.

🔧 CREATE TABLE categories — 부모 표 명령어 사전에서 자세히 →

```
id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
name VARCHAR(30) NOT NULL UNIQUE,
```

🔧 CREATE TABLE knowledge — 자식 표와 FK 명령어 사전에서 자세히 →

```
CONSTRAINT knowledge_category_fk FOREIGN KEY (category_id)
  REFERENCES categories(id) ON DELETE RESTRICT
```

🔧 Seed — 분류를 넣고 이름으로 ID를 찾아 연결 명령어 사전에서 자세히 →

```
INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference') RETURNING id, name;
INSERT INTO knowledge (title, content, category_id) SELECT '실습 파일 보관', '...', id FROM categories WHERE name = 'learning';
```

🔧 연습용 _try 표 — 본 표를 안 건드리고 키 연습 명령어 사전에서 자세히 →

```
BEGIN; CREATE TABLE categories_try (...); CREATE TABLE knowledge_try (...); ... ROLLBACK;
```

💡 나중에 태그가 여러 개 필요해지면? 지식 하나에 태그 여러 개, 태그 하나에 지식 여러 개 = N:M 관계예요. 이건 FK 한 칸으로 못 담아서 가운데에 knowledge_tags(knowledge_id FK, tag_id FK) 연결 표를 두고, 두 열의 조합을 PK로 해 같은 짝이 두 번 들어가는 걸 막아요. 오늘 필수는 아니고 설계 예시예요.

## ④ 바꾸기 — INSERT · UPDATE · DELETE · RETURNING

변경 SQL은 확인하고 → 바꾸고 → 다시 확인해요.

| 단계 | SQL | 확인할 것 |
|---|---|---|
| 1 · 대상 조회 | 같은 WHERE로 먼저 SELECT | 바꿀 행의 실제 ID·현재 값 |
| 2 · 묶음 시작 | BEGIN | 취소할 수 있는 상태 |
| 3 · 변경 | INSERT / UPDATE / DELETE + RETURNING | 어느 행이 어떻게 바뀌었나 |
| 4 · 끝내기 | COMMIT 또는 ROLLBACK → 다시 SELECT | 확정됐나 / 원래대로인가 |

🤔 왜 RETURNING을 붙여? — INSERT만 하면 "1행 넣음"만 알려 줘서 몇 번 ID가 발급됐는지 다시 찾아야 해요. RETURNING id는 바뀐 행을 그 자리에서 돌려줘요. "아마 4번이겠지" 하고 다음 ID를 예상하면 안 돼요 — 취소·실패가 있으면 번호가 건너뛰어요.

🔧 INSERT ... RETURNING 명령어 사전에서 자세히 →

```
INSERT INTO knowledge (title, content, category_id)
VALUES ('임시 안내', '한 번만 실행합니다.', 1) RETURNING id, title, category_id;
```

🔧 UPDATE ... SET ... WHERE 명령어 사전에서 자세히 →

```
UPDATE knowledge SET content = '실행 전에 가상환경을 확인합니다.'
WHERE id = 2 RETURNING id, title, content;
```

🔧 DELETE FROM ... WHERE 명령어 사전에서 자세히 →

```
DELETE FROM knowledge WHERE id = 3 RETURNING id, title;
```

⚠️ UPDATE·DELETE에서 WHERE를 빼면 표의 모든 행이 대상이에요. 그래서 같은 조건의 SELECT를 먼저 돌려 대상 행 수를 보고, 연습은 BEGIN 안에서 해요. DELETE는 행만 지우고 표 구조는 남아요(구조까지 지우는 건 DROP — 본 데이터엔 안 써요).

### 0행 · 오류 · HTTP 응답은 다른 층이에요

| 상황 | 오늘 SQL 결과 | 내일 API가 정할 것 |
|---|---|---|
| 없는 ID를 SELECT | 0행 · SQL은 정상 | 상세면 404, 목록이면 빈 배열 등 |
| 없는 ID를 UPDATE | UPDATE 0 / RETURNING 0행 | "성공"이라고 할지 404로 할지 |
| 정상 INSERT | 생성된 행과 실제 ID | 201 + 그 행 |
| 잘못된 참조·빈 본문 | DB 오류 · 변경 미확정 | 예외를 잡아 응답으로 바꾸기 |

🤔 SQL이 404를 보내 주면 편하지 않아? — DB는 "0행"이라는 사실만 알려 줘요. 그게 "없는 지식"(404)인지 "검색 결과 없음"(200 + [])인지는 API의 약속이라 FastAPI가 정해요. 층마다 책임이 달라야 나중에 DB를 바꿔도 API 약속이 그대로예요.

## ⑤ 연결해서 읽기 — JOIN · LEFT JOIN · GROUP BY

[그림: 왼쪽 JOIN 결과 3행, 가운데 LEFT JOIN 결과 4행(reference 행은 knowledge_id와 title이 NULL), 오른쪽 분류별 COUNT(*)와 COUNT(k.id) 막대그래프: learning 2·2, environment 1·1, reference 1·0] 무엇을 보나: 같은 두 표를 붙여도 결과가 달라요. JOIN(INNER)은 ON 조건에 짝이 있는 행만 → 3행, 지식이 없는 reference는 사라져요. LEFT JOIN은 왼쪽 표(분류)를 모두 남기고 짝이 없으면 오른쪽 칸을 NULL로 채워요 → 4행(분홍 줄). 오른쪽 막대는 그 4행을 분류별로 센 결과예요 — COUNT(*)는 NULL 줄도 한 행이라 reference를 1, COUNT(k.id)는 NULL을 안 세서 0. 🤔 그래서? "분류별 지식 수"를 물으면 COUNT(k.id)가 정답이에요. 값은 S0를 실제로 실행한 결과예요.

| 표현 | 뜻 |
|---|---|
| FROM knowledge AS k | k는 표의 별칭 — k.id처럼 어느 표의 열인지 밝힐 때 |
| JOIN categories AS c ON c.id = k.category_id | 어떤 값끼리 짝을 맞출지 |
| c.name AS category | 결과 열 이름 바꾸기 |
| GROUP BY c.id, c.name | 분류별로 묶기 — 결과 한 행 = 분류 한 개 |
| HAVING COUNT(k.id) >= 2 | 묶은 뒤 그룹 조건 (WHERE는 묶기 전 행 조건) |

🤔 JOIN 결과에 learning이 두 번 나오는데 중복 아니야? — 결과 한 행은 여전히 지식 한 건이에요. 지식 1·3이 둘 다 learning이라 이름이 두 번 보이는 건 정상적인 관계 결과예요. 반대로 GROUP BY를 하면 한 행의 단위가 분류로 바뀌어요. SQL을 읽을 때 "지금 결과 한 행은 무엇 하나인가?"를 먼저 물어요.

🔧 JOIN ... ON — 지식에 분류명 붙이기 명령어 사전에서 자세히 →

```
SELECT k.id, k.title, c.name AS category
FROM knowledge AS k JOIN categories AS c ON c.id = k.category_id ORDER BY k.id;
```

🔧 LEFT JOIN — 지식이 없는 분류도 남기기 명령어 사전에서 자세히 →

```
FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id
```

🔧 GROUP BY + COUNT(k.id) — 분류별 지식 수 명령어 사전에서 자세히 →

```
SELECT c.name AS category, COUNT(k.id) AS knowledge_count ... GROUP BY c.id, c.name
```

🔧 COUNT · SUM · MIN · MAX — 본문 길이 요약 명령어 사전에서 자세히 →

```
SELECT COUNT(*), SUM(char_length(content)), MIN(char_length(content)), MAX(char_length(content)) FROM knowledge;
```

🔧 HAVING — 묶은 뒤 그룹 고르기 명령어 사전에서 자세히 →

```
GROUP BY c.id, c.name HAVING COUNT(k.id) >= 2
```

⚠️ LEFT JOIN 뒤에 오른쪽 표 조건을 WHERE에 쓰면 NULL로 채운 줄이 걸러져서 0개인 분류가 사라져요 — 결과적으로 INNER JOIN처럼 돼요. 빈 분류를 살리려면 그 조건을 ON 안으로 옮겨요. (명령어 사전에 실제 결과 비교가 있어요.)

## ⑥ 제약조건 — DB가 거부하게 하기

오늘의 SQL은 FastAPI를 거치지 않고 DB로 바로 가요. 그래서 API의 입력 검증(Pydantic)과 별개로 DB에도 규칙이 있어야 해요.

|  | 흐름 |
|---|---|
| 오늘 | pgAdmin / psql → DB 제약(필수·길이·참조) → 저장 / 오류 |
| 내일 | 사용자 JSON → FastAPI 입력 검증 → DB 제약 → 저장 / 오류 |

| 규칙 | 보장하는 것 | 이번 예 | 위반 시 SQLSTATE |
|---|---|---|---|
| PRIMARY KEY | 각 행을 유일하게 식별 · NULL 불가 | knowledge.id | 23505 / 23502 |
| FOREIGN KEY | 참조 대상이 실제로 있음 | knowledge.category_id | 23503 |
| NOT NULL | NULL 저장 금지 | title · content | 23502 |
| UNIQUE | 같은 값 중복 금지 | categories.name | 23505 |
| CHECK | 정한 조건 만족 | 본문 1~2,000자 | 23514 |
| (오류 뒤 계속 실행) | — | 실패한 트랜잭션 안에서 다음 SQL | 25P02 → ROLLBACK부터 |

SQLSTATE는 PostgreSQL 공식 오류 코드표 값이에요. 코드를 외우는 게 목표가 아니라, 기대한 제약으로 실패했는지와 정리 후 기존 데이터가 그대로인지를 보는 게 목표예요.

🤔 왜 실패를 일부러 일으켜 봐? — "오류가 났다"만으로는 부족해요. 빈 본문을 넣었는데 FK 오류가 났다면 테스트가 엉뚱한 걸 검사한 거예요. 그래서 한 번에 한 가지만 틀리게(나머지 값은 정상으로) 넣고, 오류 이름과 롤백 뒤 행 수·내용을 같이 확인해요.

🔧 실패 테스트 — 빈 본문 INSERT → 오류 확인 → ROLLBACK 명령어 사전에서 자세히 →

```
BEGIN; INSERT INTO knowledge (title, content, category_id) VALUES ('오류 확인', '', 1);
ROLLBACK; SELECT COUNT(*) FROM knowledge;
```

### 지식 삭제 vs 분류 삭제

|  | 지식 한 건 삭제 | 참조 중인 분류 삭제 |
|---|---|---|
| SQL | DELETE FROM knowledge WHERE id = … | DELETE FROM categories WHERE name = 'learning' |
| 결과 | 그 행만 사라짐 · 부모와 다른 지식은 그대로 | 거부 (ON DELETE RESTRICT) · 아무것도 안 지워짐 |

🤔 CASCADE로 같이 지우면 편하지 않아? — ON DELETE CASCADE는 분류를 지우면 그 분류의 지식까지 전부 지워요. 분류 하나 정리하려다 안내문 수십 개가 사라질 수 있어요. 그래서 "분류를 지우면 지식은 어떻게 되어야 하나?"라는 업무 정책을 먼저 정하고, 이번 설계는 거부(RESTRICT)예요.

🔧 ON DELETE RESTRICT — 참조 중인 부모 삭제 명령어 사전에서 자세히 →

```
DELETE FROM categories WHERE name = 'learning';   -- 연결된 지식이 있어 거부 (23503)
```

🔧 DEFAULT — 생략한 값 vs 직접 넣은 NULL 명령어 사전에서 자세히 →

```
INSERT INTO defaults_try DEFAULT VALUES;           -- '미지정'
INSERT INTO defaults_try (label) VALUES (NULL);    -- NULL 그대로
```

💡 DEFAULT는 값을 생략했을 때만 채워요. 직접 NULL을 넣으면 고쳐 주지 않아요. NULL을 막고 싶으면 NOT NULL을 따로 걸어야 해요.

⚠️ 형식 검증 ≠ 사실 검증. 제약조건은 "본문이 1자 이상"까지만 지켜 줘요. 안내문 내용이 맞는지, 이 사람이 고쳐도 되는지, AI 답변이 정확한지는 다른 검사예요.

## ⑦ 트랜잭션 — 확정할까, 없던 일로 할까

트랜잭션은 여러 변경을 한 묶음으로 다루는 단위예요. BEGIN으로 시작하고, 같은 연결에서 작업한 뒤 둘 중 하나로 끝내요.

| 명령 | 뜻 |
|---|---|
| BEGIN | 묶음 시작 — 이후 변경은 미확정 |
| COMMIT | 묶음의 변경을 확정 — 다른 연결·재접속에서도 보임 |
| ROLLBACK | 미확정 변경을 취소 — BEGIN 직전 상태로 |

[그림: 위 줄: SELECT 원래 본문, BEGIN, UPDATE 후 임시 본문, ROLLBACK, SELECT 원래 본문. 아래 줄: BEGIN, UPDATE RETURNING 새 본문, COMMIT, 재접속 후 새 본문, ROLLBACK 해도 값 그대로] 무엇을 보나: id 2번 지식의 content 값이 칸마다 어떻게 보이는지예요. 위(ROLLBACK): 작업 중엔 임시 본문이 보이지만 — 같은 순간 다른 연결에서는 아직 원래 본문이에요 — ROLLBACK 뒤엔 원래 본문으로 돌아가요. 아래(COMMIT): 확정하면 연결을 끊고 다시 붙어도 새 본문이고, 그 뒤 ROLLBACK은 되돌릴 게 없어요. 값은 실제 실행으로 확인한 거예요. 🤔 왜 재접속까지 해 봐? 같은 창에서 보이는 건 내 미확정 변경일 수도 있어서, 새 연결에서 보여야 정말 확정된 거예요.

⚠️ ROLLBACK은 "DB 전체를 과거로 돌리는 버튼"이 아니에요. 지금 열린 묶음의 미확정 변경만 취소해요. 이미 COMMIT한 값은 ROLLBACK으로 못 되돌려요 — 되돌리려면 새 UPDATE로 다시 고쳐야 해요.

🔧 BEGIN → UPDATE → ROLLBACK — 세 번 비교 명령어 사전에서 자세히 →

```
SELECT id, content FROM knowledge WHERE id = 2;
BEGIN; UPDATE knowledge SET content = '취소할 임시 안내입니다.' WHERE id = 2; ROLLBACK;
```

🔧 COMMIT → 재접속해서 확인 명령어 사전에서 자세히 →

```
BEGIN; UPDATE knowledge SET content = '실행 전에 가상환경을 확인합니다.' WHERE id = 2 RETURNING id, content; COMMIT;
```

🔧 원자성 — 두 번째 변경이 실패하면 첫 변경도 취소 명령어 사전에서 자세히 →

```
BEGIN; UPDATE ... WHERE id = 2;  UPDATE knowledge SET content = '' WHERE id = 3;  -- CHECK 오류
ROLLBACK;
```

🤔 원자성(Atomicity)이 왜 중요해? — "B 안내 수정 + C 안내 삭제"가 한 작업인데 앞의 것만 반영되면 지식이 반쯤 바뀐 상태로 남아요. 묶음으로 처리하면 모두 반영되거나 하나도 반영 안 되거나 둘 중 하나예요. PostgreSQL은 묶음 안에서 오류가 나면 그 묶음을 중단 상태로 만들어 다음 SQL을 거부해요(25P02) — 그래서 오류 뒤엔 ROLLBACK부터 해요.

### ID는 되돌아가지 않을 수 있어요

[그림: 1, 2, 3은 행이 있고, 4는 INSERT 뒤 ROLLBACK, 5는 CHECK 위반으로 INSERT 실패해 점선 빈칸, 다음 INSERT는 6을 받는 그림] 무엇을 보나: IDENTITY는 안에서 시퀀스(번호 발급기)를 써요. INSERT가 번호를 받아 간 뒤 ROLLBACK되거나 제약 위반으로 실패해도 발급기는 되감기지 않을 수 있어서, 표에는 1, 2, 3, 6처럼 빈 번호가 생겨요(PostgreSQL 공식 문서 내용을 그린 개념도). 🤔 그래서? max(id)를 행 수로 쓰지 말고(COUNT(*)를 써요), 새 ID는 예상하지 말고 RETURNING id로 읽어요. 번호가 비어도 정상이에요.

| 상황 | 지식 행 | 다음 ID |
|---|---|---|
| 어제: API 검증에서 422 | 저장 함수에 도달 안 함 | 파일의 next_id도 그대로 |
| 오늘: INSERT 뒤 ROLLBACK | 새 행 안 남음 | 발급된 번호는 재사용 안 될 수 있음 |
| 오늘: 제약 위반으로 INSERT 실패 | 실패 행 저장 안 됨 | 이미 발급됐으면 빈 번호 |

## 🧪 트랜잭션 직접 해 보기

브라우저 안에서 PostgreSQL 트랜잭션 규칙을 흉내 냈어요(실제 DB는 아니에요). 연결 A에서 바꾸고, 연결 B에서는 무엇이 보이는지 비교해 보세요. 오른쪽 아래 숫자는 IDENTITY의 다음 발급 번호예요.

## ⑧ 인덱스와 실행계획 — INDEX · EXPLAIN

인덱스는 표와 별도로 만드는 탐색 구조예요. 책 뒤의 "찾아보기"처럼, category_id = 1인 행이 어디 있는지 표 전체를 훑지 않고 찾아가게 도와요.

|  | 내용 |
|---|---|
| 도움이 되는 곳 | 자주 쓰는 WHERE 조건, JOIN의 연결 열 |
| 비용 | INSERT·UPDATE 때 인덱스도 고쳐야 하고 저장 공간을 써요 |
| 자동으로 생기나? | PK·UNIQUE엔 자동으로 생겨요. 자식 표의 FK 열엔 자동으로 안 생겨요 — 그래서 knowledge.category_id에 직접 만들어요 |

🔧 pg_indexes 확인 → CREATE INDEX 명령어 사전에서 자세히 →

```
SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'knowledge';
CREATE INDEX idx_knowledge_category_id ON knowledge (category_id);
```

| EXPLAIN 용어 | 뜻 |
|---|---|
| EXPLAIN | 실행하지 않고 예상 실행계획만 보여 줌 |
| EXPLAIN (ANALYZE, BUFFERS) | 실제로 실행하고 걸린 시간·실제 행 수·읽은 블록까지 (그래서 변경 SQL엔 쓰지 않아요) |
| Seq Scan | 표를 처음부터 끝까지 훑기 |
| Index Scan / Bitmap Index Scan | 인덱스로 찾아가기 |
| rows | 예상 행 수 (ANALYZE면 실제 행 수도 따로 나옴) |
| cost | 상대적인 비용 단위 — 밀리초가 아니에요 |

🔧 EXPLAIN · EXPLAIN (ANALYZE, BUFFERS) 명령어 사전에서 자세히 →

```
EXPLAIN SELECT id, title FROM knowledge WHERE category_id = 1;
```

🤔 인덱스를 만들었는데 Seq Scan이 나와. 실패야? — 아니에요. 행이 3개뿐이면 표 전체를 읽는 게 인덱스를 거쳐 가는 것보다 싸다고 판단할 수 있어요. 실행계획은 데이터 양·통계에 따라 바뀌어서, 계획 이름을 정답으로 외우지 않아요. 인덱스는 결과 행을 바꾸지 않고 찾아가는 길만 바꿔요.

## ⑨ 다음 단계와 연결

| 날짜 | 핵심 결과 | 오늘과의 연결 |
|---|---|---|
| 11/10 | JSON 파일 지식 CRUD API | title · content · category / 실제 ID |
| 11/11 (오늘) | 관계형 지식 DB · SQL · ERD | categories · knowledge / 제약 · 트랜잭션 |
| 11/12 | FastAPI–PostgreSQL 연동 | 접속 설정 · ORM · Migration · 통합 검증 |
| 11/18~19 | 검색용 인덱스 · 문서·메타데이터 처리 | 원본 → 청크·벡터 |
| 11/20 | Basic RAG | 검색한 원문을 LLM에 제공 |

### DB로 바꿔도 자동으로 해결되지 않는 것

| 항목 | 구분할 것 | 이후 확인 |
|---|---|---|
| 스키마 | Pydantic 모델 ≠ DB 테이블 정의 | 입력·저장·응답 규칙이 서로 맞는지 |
| 데이터 이관 | JSON이 DB로 저절로 옮겨지지 않음 | 기존 ID·분류 매핑·중복 정책 |
| 분류 입력 | API는 category 문자열, DB는 category_id 번호 | 있는 분류는 ID를 찾고, 새 이름이면 분류를 먼저 만든 뒤 저장 — 한 트랜잭션으로 |
| 검색 데이터 | 원본 수정 ≠ 청크·벡터 자동 갱신 | 수정·삭제 전파와 출처 추적 |
| 공개 서비스 | 로컬 학습 ≠ 운영 환경 | 앱 전용 계정·비밀 설정·백업·관측 |

🤔 DB로 바꾸면 API 모양도 바꿔야 해? — 아니에요. 바깥 약속("category": "environment" 문자열)은 그대로 두고, 안쪽에서 이름 ↔ ID를 바꿔 주면 돼요. 부르는 쪽(화면·Agent)은 저장소가 바뀐 걸 몰라도 돼요. 단, 분류 이름 자체를 바꾸면 그 분류를 가리키는 모든 지식의 조회 결과가 같이 바뀌어요.

💡 파일 → DB로 옮기는 이유는 "파일은 사라져서"가 아니에요. 어제 JSON도 재시작 후 남았어요. 옮기는 이유는 관계(분류–지식), 공유 규칙(누가 넣어도 같은 제약), 조회(필요한 행만·연결·집계), 트랜잭션(여러 변경을 한 묶음으로)을 다루기 위해서예요.

## 원본 자료

- 강의: 20261111.pdf (PostgreSQL 입문 · 지식 관리 DB, 94쪽) — 플랫폼 속 위치 p2–11 · 접속·첫 SQL p17–22 · SELECT·WHERE·NULL p24–29 · 두 테이블·ERD·DDL p31–37 · INSERT·UPDATE·DELETE p39–43 · JOIN·집계 p45–50 · 제약·실패 p52–56 · 트랜잭션·IDENTITY·INDEX·EXPLAIN p58–63 · 다음 수업 연결 p65–66 · 부록(DB 준비·Seed·연습 표·HAVING·DEFAULT·원자성·N:M·Windows·psql·연동 체크포인트) p78–91

- 노트·명령어 사전의 결과 값: 이 PC에 PostgreSQL이 없어서, 같은 S0 데이터와 강의 SQL을 파이썬 내장 SQLite 3.50으로 실제 실행해 확인 (tools/io_day18.py, 그림 tools/visuals/make_day18.py). PostgreSQL 전용 값(SQLSTATE·current_database()·EXPLAIN 표기·psql)은 강의자료·공식 문서 기준이라고 따로 표시했어요.

- 공식 문서: Querying a Table · Schemas · Constraints · CREATE TABLE · Identity Columns

- 공식 문서: Returning Data from Modified Rows · Table Expressions (JOIN) · Aggregate Functions · Transactions · Sequence Functions

- 공식 문서: Indexes · Using EXPLAIN · Error Codes · psql · pgAdmin Query Tool

## 🔧 Day18 명령어 (명령어 사전: https://yiji-1015.github.io/sesac-notes/commands.html#day18)
| 이론 | 대응 명령어 |
|---|---|
| 📦 필요한 프로그램 | psql --version, Get-Service postgresql* |
| ① DB · DBMS · 접속 도구 | SELECT current_database(), CREATE DATABASE, psql -U -h -p -d, \dt · \d |
| ② 지식 조회 | SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT, LIKE, IS NULL |
| ③ 두 테이블 설계 | CREATE TABLE, GENERATED ALWAYS AS IDENTITY, PRIMARY KEY, FOREIGN KEY ... REFERENCES, CHECK |
| ④ 바꾸기 | INSERT ... RETURNING, UPDATE ... SET ... WHERE, DELETE FROM ... WHERE |
| ⑤ 연결해서 읽기 | JOIN ... ON, LEFT JOIN, GROUP BY, COUNT(k.id), HAVING |
| ⑥ 제약조건 | NOT NULL · UNIQUE · CHECK · ON DELETE RESTRICT · DEFAULT, SQLSTATE |
| ⑦ 트랜잭션 | BEGIN, COMMIT, ROLLBACK |
| 🧪 트랜잭션 직접 해 보기 | (브라우저 안에서 연결 A · B 흉내) |
| ⑧ 인덱스와 실행계획 | pg_indexes, CREATE INDEX, EXPLAIN (ANALYZE, BUFFERS) |
| ⑨ 다음 단계와 연결 | FastAPI–PostgreSQL 연동 (11/12) · 검색 인덱스 · RAG |

### 첫 SQL — current_database() · version() · SELECT 1

```
SELECT current_database() AS db_name,
       current_user AS user_name;      -- 지금 접속한 DB · 계정
SELECT version();                      -- 서버 버전
SELECT 1 AS connection_ok;             -- SQL을 보내고 결과를 받을 수 있나
```

▶ 실제로 돌려 보면 PostgreSQL 전용 함수라 SQLite로는 못 돌려요 — 강의자료 p21의 기대값

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT current_database() AS db_name, current_user AS user_name; | PG = db_name: knowledge_lab (user: 접속한 계정) | 지금 연결된 DB 이름부터 확인 |
| SELECT 1 AS connection_ok; | PG = connection_ok: 1 | SQL을 보내고 결과를 받는 길이 열렸다 |
| SELECT version(); | PG = 서버 버전 문자열 (수업 기준 18.x) | psql --version은 클라이언트 버전 — 서로 다를 수 있어요 |

| 표현 | 뜻 | 🤔 왜? |
|---|---|---|
| current_database() | 지금 연결된 Database 이름 | knowledge_lab이 아니면 표를 엉뚱한 DB에 만들게 돼요 |
| current_user | 접속한 계정 이름 (괄호 없음) | 권한 문제를 볼 때 첫 단서 |
| AS db_name | 결과 열 이름 붙이기 | Data Output에서 무슨 값인지 바로 보이게 |
| version() | 서버 버전 | psql --version(클라이언트)과 구분 |

💡 pgAdmin에서는 실행할 문장만 선택하고 실행해요. 세 문장을 한꺼번에 돌리면 마지막 결과만 Data Output에 보일 수 있어요.

### CREATE DATABASE — 개인 학습 DB 만들기

```
-- postgres DB에 접속한 Query Tool에서 이 문장만 단독 실행
CREATE DATABASE knowledge_lab;

-- 왼쪽 트리에서 knowledge_lab을 선택하고 Query Tool을 새로 연 뒤
SELECT current_database();      -- 기대: knowledge_lab
```

▶ 실제로 돌려 보면 PostgreSQL 전용이라 SQLite로는 못 돌려요 — 강의자료 p68(A01)의 기대값

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| CREATE DATABASE knowledge_lab; (postgres DB에서) | PG = CREATE DATABASE | 새 DB가 생김 — 아직 이 창은 postgres DB |
| knowledge_lab에서 새 Query Tool → SELECT current_database(); | PG = knowledge_lab | 이제부터 표를 만들 곳 |
| BEGIN; CREATE DATABASE ... | PG = ERROR: CREATE DATABASE cannot run inside a transaction block | 그래서 Auto commit 상태에서 단독 실행 |

| 규칙 | 🤔 왜? |
|---|---|
| Auto commit이 켜진 상태에서, BEGIN 안이 아닌 곳에서 | CREATE DATABASE는 트랜잭션 묶음 안에서 실행할 수 없어요 |
| 만든 뒤 새 Query Tool을 그 DB에서 열기 | Query Tool은 열 때 정한 DB에 붙어 있어요. 만들기만 하면 여전히 postgres DB예요 |
| 같은 이름이 있으면 내용부터 확인 | 임의로 DROP하면 남의(또는 어제의) 데이터를 지울 수 있어요 |

💡 준비 순서: A01 DB 만들기 → A02 categories → A03 knowledge → A04·A05 Seed. 빈 개인 DB에 한 번만 해요.

### psql — 터미널 접속과 메타 명령

```
# PowerShell에서
psql -U postgres -h localhost -p 5432 -d knowledge_lab

-- 접속한 psql 안에서
SELECT current_database();     -- SQL (세미콜론으로 끝)
\conninfo                      -- 지금 접속 정보
\dt                            -- 표 목록
\d knowledge                   -- 표 구조 (열 · 제약 · 인덱스)
\q                             -- 나가기
```

▶ 실제로 돌려 보면 PostgreSQL 클라이언트라 이 PC에선 못 돌려요 — 각 명령이 하는 일 (공식 psql 문서)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| psql -U postgres -h localhost -p 5432 -d knowledge_lab | Password for user postgres: (입력해도 화면에 안 보임) | PowerShell에서 실행 — 접속되면 프롬프트가 knowledge_lab=# 로 바뀜 |
| \conninfo | 접속한 DB · 사용자 · 호스트 · 포트 한 줄 | 어느 서버·DB에 붙었는지 확인 |
| \dt | public 스키마의 표 목록 (categories, knowledge) | SQL이 아니라 psql 전용 명령 |
| \d knowledge | 열 · 자료형 · 제약 · 인덱스 · FK | CREATE TABLE이 의도대로 됐는지 확인 |
| \q | psql 종료 | 서버는 계속 켜져 있어요 |

| 옵션 | 뜻 |
|---|---|
| -U postgres | User — 접속 계정 |
| -h localhost | Host — 서버 위치 |
| -p 5432 | Port — 설치 때 정한 포트 |
| -d knowledge_lab | Database — 붙을 DB |

⚠️ \dt · \d · \q처럼 역슬래시로 시작하는 건 SQL이 아니라 psql 전용 명령이에요. pgAdmin Query Tool에 붙이면 문법 오류가 나요. 반대로 첫 줄(psql ...)은 PowerShell 명령이라 psql 안에서 치면 안 돼요.

| 증상 | 먼저 볼 것 |
|---|---|
| connection refused | 서버 서비스가 켜졌나 · Host · Port |
| 인증 실패 | 사용자 · 서버 비밀번호 (pgAdmin 보호 비밀번호와 다름) |
| database "..." does not exist | DB 이름 · A01을 했나 |
| relation "..." does not exist | 지금 DB · 스키마 · 표를 만든 곳 |
| psql을 못 찾음 | Command Line Tools 설치 · SQL Shell(psql) 메뉴나 설치 폴더의 bin 경로 |

### SELECT · FROM · ORDER BY

```
SELECT id, title, category_id     -- 어떤 열?
FROM knowledge                    -- 어느 표?
ORDER BY id;                      -- 어떤 순서?

SELECT name
FROM categories
ORDER BY id;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT id, title, category_id FROM knowledge ORDER BY id; | id \| title \| category_id 1 \| 실습 파일 보관 \| 1 2 \| 서버 실행 \| 2 3 \| 질문 작성 \| 1 | 한 행 = 지식 한 건 |
| SELECT name FROM categories ORDER BY id; | name learning environment reference | 분류 3행 |

| 절 | 뜻 | 바꾸면? |
|---|---|---|
| SELECT 열, 열 | 보여 줄 열 | SELECT *는 모든 열 — 연습엔 편하지만 필요한 열만 고르는 습관이 좋아요 |
| FROM 표 | 읽을 표 | 없는 표면 relation does not exist |
| ORDER BY id | 작은 id부터 | 빼면 순서가 보장되지 않아요 |

### WHERE — ID·분류 조건

```
SELECT id, title
FROM knowledge
WHERE id = 2;                     -- 특정 지식 한 건

SELECT id, title
FROM knowledge
WHERE category_id = 1             -- 같은 분류
ORDER BY id;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| WHERE id = 2 | id \| title 2 \| 서버 실행 | 특정 지식 한 건 |
| WHERE category_id = 1 ORDER BY id | id \| title 1 \| 실습 파일 보관 3 \| 질문 작성 | 같은 분류(learning)의 지식 |
| WHERE id = 99 | id \| title (0행) | 0행은 오류가 아니라 '없음'이라는 정상 결과 |

| 표현 | 뜻 |
|---|---|
| = | 같다 (SQL은 ==가 아니라 = 하나) |
| <> 또는 != | 다르다 |
| 'learning' | 문자열은 작은따옴표. 큰따옴표 "..."는 열·표 이름을 뜻해요 |
| IN (2, 3) | 목록 중 하나 |

🤔 결과 0행은 조회 실패야? — 아니에요. 조건에 맞는 행이 없다는 정상 결과예요. 오류와 0행을 구분해야 내일 API에서 404 / 빈 배열을 정할 수 있어요.

### AND · LIKE · DESC · LIMIT

```
SELECT id, title
FROM knowledge
WHERE category_id = 1
  AND title LIKE '%실습%'         -- '실습'을 포함
ORDER BY id DESC                  -- 큰 ID부터
LIMIT 2;                          -- 최대 2행
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| WHERE category_id = 1 AND title LIKE '%실습%' ORDER BY id DESC LIMIT 2 | id \| title 1 \| 실습 파일 보관 | 두 조건을 다 만족하는 건 1번뿐 |
| WHERE category_id = 1 ORDER BY id DESC LIMIT 2 | id \| title 3 \| 질문 작성 1 \| 실습 파일 보관 | LIKE를 빼면 큰 ID부터 2행 |
| WHERE title LIKE '%실행%' | id \| title 2 \| 서버 실행 | 글자 포함 검색 — 뜻이 비슷한 '구동'은 못 찾아요 |

| 표현 | 뜻 | 주의 |
|---|---|---|
| AND / OR | 둘 다 / 하나라도 | 섞어 쓰면 괄호로 묶기: a AND (b OR c) |
| LIKE '%실습%' | % = 아무 글자 0개 이상 | PostgreSQL의 LIKE는 대소문자를 구분해요 (ILIKE는 무시) |
| LIKE '실습%' | '실습'으로 시작 | _는 아무 글자 정확히 1개 |
| DESC | 내림차순 (기본은 ASC) | 같은 값이 여럿이면 순서가 섞일 수 있어 두 번째 기준을 더해요 |
| LIMIT 2 | 최대 2행 | ORDER BY 없이 쓰면 "아무 2행"이에요 |

⚠️ LIKE는 글자 포함 검색이에요. 뜻이 비슷한 말을 찾는 의미 검색(RAG)이 아니에요.

### IS NULL · char_length() — NULL과 빈 문자열

```
SELECT NULL IS NULL AS is_missing;          -- true
SELECT '' IS NULL AS is_missing;            -- false
SELECT char_length('') AS text_length;      -- 0
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음) · SQLite는 true/false를 1/0으로 보여 줘요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT NULL IS NULL AS is_missing; | is_missing 1 | PG에선 true |
| SELECT '' IS NULL AS is_missing; | is_missing 0 | PG에선 false — 빈 문자열은 값이 있음 |
| SELECT char_length('') AS text_length; | text_length 0 | 길이 0인 값 |
| SELECT NULL = NULL AS eq; | eq NULL | = 로 비교하면 '모름'(NULL) — 그래서 IS NULL을 써요 |

| 표현 | 뜻 |
|---|---|
| x IS NULL / x IS NOT NULL | 값이 없는지 / 있는지 |
| x = NULL | ❌ 결과가 true가 아니라 NULL(모름)이라 WHERE에서 아무 행도 안 걸려요 |
| char_length(x) | 글자 수 (한글 한 글자 = 1) |

### CREATE TABLE categories — 부모 표

```
CREATE TABLE categories (
  id BIGINT GENERATED ALWAYS AS IDENTITY
    PRIMARY KEY,
  name VARCHAR(30) NOT NULL UNIQUE,
  CONSTRAINT categories_name_length
    CHECK (char_length(name) BETWEEN 1 AND 30)
);
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| CREATE TABLE categories (...) | (완료) | 표가 생김 (PG 메시지: CREATE TABLE) |
| INSERT INTO categories (name) VALUES ('learning'); | id \| name 1 \| learning | id는 생략 → DB가 1을 발급 |
| 같은 'learning' 한 번 더 | 오류: UNIQUE constraint failed: categories.name | UNIQUE가 중복 분류명을 거부 (PG SQLSTATE 23505) |
| name = '' | 오류: CHECK constraint failed: categories_name_length | 길이 CHECK가 빈 이름을 거부 (PG 23514) |

| 부분 | 뜻 | 🤔 왜? |
|---|---|---|
| BIGINT | 큰 정수 (약 ±9.2×10¹⁸) | ID가 모자랄 일이 없게 |
| GENERATED ALWAYS AS IDENTITY | DB가 번호를 발급. ALWAYS면 사용자가 직접 넣는 걸 막음 | 번호 매기기를 사람이 하지 않게 — 어제의 next_id 역할 |
| PRIMARY KEY | 유일 + NULL 불가 (인덱스도 자동 생성) | 행 하나를 콕 집는 열쇠 |
| VARCHAR(30) | 최대 30자 문자열 | 분류 이름은 짧아야 해서 |
| UNIQUE | 같은 값 두 번 금지 | 이름으로 ID를 찾으려면 이름이 하나여야 |
| CONSTRAINT 이름 CHECK (...) | 이름 붙인 조건 검사 | 오류 메시지에 이 이름이 나와서 원인을 바로 알 수 있어요 |

💡 이미 만든 표에는 다시 실행하지 않아요(relation already exists 오류). 구조를 바꾸려면 ALTER TABLE이나, 연습이면 _try 표로 해요.

### CREATE TABLE knowledge — 자식 표와 FK

```
CREATE TABLE knowledge (
  id BIGINT GENERATED ALWAYS AS IDENTITY
    PRIMARY KEY,
  title VARCHAR(80) NOT NULL,
  content TEXT NOT NULL,
  category_id BIGINT NOT NULL,
  CONSTRAINT knowledge_title_length
    CHECK (char_length(title) BETWEEN 1 AND 80),
  CONSTRAINT knowledge_content_length
    CHECK (char_length(content) BETWEEN 1 AND 2000),
  CONSTRAINT knowledge_category_fk
    FOREIGN KEY (category_id)
    REFERENCES categories(id) ON DELETE RESTRICT
);
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| CREATE TABLE knowledge (...) | (완료) | 부모 categories가 먼저 있어야 해요 |
| 만들어진 표 목록 | name categories knowledge |  |
| category_id = 99 (없는 분류) | 오류: FOREIGN KEY constraint failed | FK가 부모 없는 참조를 거부 (PG 23503) |
| title = NULL | 오류: NOT NULL constraint failed: knowledge.title | NOT NULL 위반 (PG 23502) |
| 같은 제목 두 번 | id \| title 1 \| 같은 제목 2 \| 같은 제목 | title엔 UNIQUE가 없어서 둘 다 들어가요 — 제목은 ID 대신 못 써요 |

| 부분 | 뜻 |
|---|---|
| TEXT | 길이 제한 없는 문자열 — 길이는 CHECK로 1~2,000자 |
| category_id BIGINT NOT NULL | 분류 번호 — 반드시 있어야 함 (지식 1건 → 분류 1개) |
| FOREIGN KEY (category_id) REFERENCES categories(id) | 넣을 때 그 번호의 분류가 실제로 있는지 검사 |
| ON DELETE RESTRICT | 이 분류를 가리키는 지식이 있으면 분류 삭제를 거부 |

🤔 FK 타입을 왜 BIGINT로 맞춰? — 가리키는 쪽(category_id)과 가리켜지는 쪽(categories.id)의 타입이 같아야 비교·연결이 자연스러워요. 그리고 title에는 UNIQUE가 없어요 — 같은 제목도 허용(어제 API와 같은 규칙).

### Seed — 분류 3건 · 지식 3건 넣기 (S0)

```
INSERT INTO categories (name)
VALUES ('learning'), ('environment'), ('reference')
RETURNING id, name;

INSERT INTO knowledge (title, content, category_id)
SELECT '실습 파일 보관',
       '실습 파일은 개인 폴더에 보관합니다.', id
FROM categories WHERE name = 'learning'
RETURNING id, title;

INSERT INTO knowledge (title, content, category_id)
SELECT '서버 실행',
       '프로젝트 폴더에서 서버를 실행합니다.', id
FROM categories WHERE name = 'environment'
RETURNING id, title;

INSERT INTO knowledge (title, content, category_id)
SELECT '질문 작성',
       '오류 메시지와 시도한 작업을 함께 적습니다.', id
FROM categories WHERE name = 'learning'
RETURNING id, title;

SELECT COUNT(*) FROM knowledge;      -- 기대: 3
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference') RETURNING id, name; | id \| name 1 \| learning 2 \| environment 3 \| reference | DB가 발급한 실제 ID |
| INSERT INTO knowledge ... SELECT '실습 파일 보관', ..., id FROM categories WHERE name = 'learning' RETURNING id, title; | id \| title 1 \| 실습 파일 보관 | 분류 ID를 이름으로 찾아 넣음 |
| 나머지 두 건 (environment · learning) | id \| title 2 \| 서버 실행 3 \| 질문 작성 |  |
| SELECT COUNT(*) FROM knowledge; | COUNT(*) 3 | S0 준비 완료 |
| WHERE name = 'nothing' 으로 INSERT SELECT | id (0행) | 부모가 없으면 오류 없이 0행 — 그래서 RETURNING을 꼭 봐요 |

| 표현 | 뜻 | 🤔 왜? |
|---|---|---|
| VALUES (...), (...), (...) | 여러 행을 한 문장으로 | id는 생략 → DB가 발급 |
| INSERT ... SELECT ... FROM categories WHERE name = ... | 분류 이름으로 실제 ID를 찾아 넣기 | "learning은 1번이겠지" 하고 숫자를 박지 않으려고 |
| RETURNING | 들어간 행 확인 | 부모가 없으면 오류 없이 0행이라 RETURNING으로 확인해야 해요 |

⚠️ Seed는 빈 DB에 한 번만. 반복하면 분류는 UNIQUE에 걸리고, 지식은 같은 내용이 또 들어가요(제목은 UNIQUE가 아니니까).

### 연습용 _try 표 — PK·FK 연습 후 ROLLBACK

```
BEGIN;
CREATE TABLE categories_try (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL
);
CREATE TABLE knowledge_try (
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL,
  category_id INTEGER NOT NULL
    REFERENCES categories_try(id)
);
INSERT INTO categories_try VALUES (1, '연습');
INSERT INTO knowledge_try VALUES (1, '안내', 1);
SELECT * FROM knowledge_try;
ROLLBACK;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT * FROM knowledge_try; | id \| title \| category_id 1 \| 안내 \| 1 | PK·FK가 연결된 연습용 표 |
| INSERT INTO knowledge_try VALUES (2, '안내2', 7); | 오류: FOREIGN KEY constraint failed | 연습 표에서도 FK가 동작 |
| ROLLBACK; 뒤 _try 표가 남았나? | name (0행) | 표 생성(DDL)까지 같이 취소됨 — 본 표는 그대로 |

| 포인트 | 뜻 |
|---|---|
| _try 이름 | 본 표(categories·knowledge)와 섞이지 않게 |
| REFERENCES categories_try(id) | 열 정의 안에 쓰는 짧은 FK 문법 |
| id를 손으로 1 | 연습용 수동 번호 — 본 모델은 IDENTITY |
| ROLLBACK | PostgreSQL은 CREATE TABLE도 트랜잭션 안에서 취소돼요 → 표까지 사라짐 |

⚠️ 본 데이터에는 DROP TABLE을 쓰지 않아요. 같은 연결(같은 Query Tool)에서 ROLLBACK해야 취소돼요.

### INSERT ... RETURNING — 새 지식 넣고 ID 바로 읽기

```
BEGIN;
INSERT INTO knowledge (title, content, category_id)
VALUES ('임시 안내', '한 번만 실행합니다.', 1)
RETURNING id, title, category_id;
-- 결과의 id를 확인한 뒤 별도로 실행
ROLLBACK;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| BEGIN; INSERT ... RETURNING id, title, category_id; | id \| title \| category_id 4 \| 임시 안내 \| 1 | 발급된 실제 ID를 바로 읽음 |
| (취소 전) SELECT COUNT(*) | COUNT(*) 4 | 같은 연결에선 새 행이 보여요 |
| ROLLBACK; SELECT COUNT(*) | COUNT(*) 3 | 새 행은 남지 않음 |
| 취소 뒤 다시 INSERT ... RETURNING id | id 4 | SQLite는 4를 다시 줌. PG는 시퀀스 번호를 되돌리지 않아 5가 나올 수 있어요(공식 문서) — 그래서 ID를 예상하지 말고 RETURNING을 봐요 |

| 부분 | 뜻 |
|---|---|
| INSERT INTO 표 (열, 열, 열) | 값을 넣을 열 목록 — id는 빼요 (IDENTITY) |
| VALUES (..., ..., ...) | 열 목록과 같은 순서로 값 |
| RETURNING id, ... | 실제로 들어간 행을 돌려받기 |
| category_id = 1 | S0에서 learning이 1번이라는 전제 — 다르면 실제 ID를 써요 |

⚠️ ID를 4로 고정하지 마세요. 취소·실패가 있었으면 다음 번호가 5, 6일 수 있어요. 새 ID는 항상 RETURNING 결과로.

### UPDATE ... SET ... WHERE — 같은 ID의 본문만 바꾸기

```
SELECT id, content FROM knowledge WHERE id = 2;    -- 1. 대상 확인

BEGIN;                                              -- 2. 묶음 시작
UPDATE knowledge
SET content = '실행 전에 가상환경을 확인합니다.'     -- 바꿀 열 = 새 값
WHERE id = 2                                        -- 바꿀 대상
RETURNING id, title, content;                       -- 3. 무엇이 바뀌었나
ROLLBACK;                                           -- 4. 연습이니 취소

SELECT id, content FROM knowledge WHERE id = 2;    -- 원래 본문인지
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 시작: SELECT id, content WHERE id = 2 | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. | 원래 본문 기록 |
| BEGIN; UPDATE ... WHERE id = 2 RETURNING id, title, content; | id \| title \| content 2 \| 서버 실행 \| 실행 전에 가상환경을 확인합니다. | 변경 1행 — 어느 행이 어떻게 바뀌었는지 |
| ROLLBACK; SELECT id, content WHERE id = 2 | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. | 원문 복원 |
| UPDATE ... WHERE id = 99 | 바뀐 행 0 | 없는 ID — 오류 없이 0행 (PG 메시지: UPDATE 0). 404로 볼지는 API가 정해요 |

| 부분 | 뜻 | 바꾸면? |
|---|---|---|
| SET 열 = 값 | 바꿀 열과 새 값 | 여러 열은 쉼표: SET title = 'a', content = 'b' |
| WHERE id = 2 | 바꿀 행 | 빼면 모든 행이 바뀌어요 |
| RETURNING | 바뀐 뒤의 행 | 없는 ID면 0행 (오류 아님) |

💡 다른 분류로 옮길 때는 SET category_id = 3처럼 번호를 바꿔요. 분류 이름(categories.name)을 바꾸는 건 그 분류의 모든 지식에 영향을 줘요.

### DELETE FROM ... WHERE — 지식 한 행 삭제

```
BEGIN;
DELETE FROM knowledge
WHERE id = 3
RETURNING id, title;            -- 지운 행이 무엇이었나
ROLLBACK;                       -- 연습이니 취소

SELECT id, title FROM knowledge WHERE id = 3;   -- 다시 1행
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| BEGIN; DELETE FROM knowledge WHERE id = 3 RETURNING id, title; | id \| title 3 \| 질문 작성 | 지운 행이 무엇이었는지 |
| (취소 전) SELECT COUNT(*) | COUNT(*) 2 |  |
| ROLLBACK; SELECT id, title WHERE id = 3 | id \| title 3 \| 질문 작성 | 취소해서 3번이 다시 보여요 |
| BEGIN; DELETE FROM knowledge; (WHERE 없음) → ROLLBACK | 바뀐 행 3 | WHERE를 빼면 전체 삭제 — 연습이라 취소했어요 |

| 명령 | 지우는 것 |
|---|---|
| DELETE FROM 표 WHERE ... | 조건에 맞는 행. 표 구조는 남음 |
| DELETE FROM 표 (WHERE 없음) | 모든 행 ⚠️ |
| DROP TABLE 표 | 표 자체 (구조 + 데이터) — 오늘 본 데이터엔 안 씀 |

### JOIN ... ON — 지식 목록에 분류명 붙이기

```
SELECT k.id, k.title, c.name AS category
FROM knowledge AS k
JOIN categories AS c
  ON c.id = k.category_id
ORDER BY k.id;

-- "서버 실행 안내와 분류를 보여 줘"
SELECT k.id, k.title, c.name AS category
FROM knowledge AS k
JOIN categories AS c ON c.id = k.category_id
WHERE k.id = 2;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT k.id, k.title, c.name AS category FROM knowledge AS k JOIN categories AS c ON c.id = k.category_id ORDER BY k.id; | id \| title \| category 1 \| 실습 파일 보관 \| learning 2 \| 서버 실행 \| environment 3 \| 질문 작성 \| learning | 3행 — 한 행은 여전히 지식 한 건 |
| ... WHERE k.id = 2 | id \| title \| category 2 \| 서버 실행 \| environment | '서버 실행 안내와 분류를 보여 줘' |

| 표현 | 뜻 |
|---|---|
| knowledge AS k | 표 별칭 k (AS는 생략 가능) |
| JOIN | = INNER JOIN — 양쪽에 짝이 있는 행만 |
| ON c.id = k.category_id | 어떤 값끼리 짝을 맞출지 |
| c.name AS category | 결과 열 이름 — 어제 API의 category와 같은 이름 |

⚠️ ON을 빼거나 잘못 쓰면 모든 행 × 모든 행 조합(3 × 3 = 9행)이 나올 수 있어요. JOIN 뒤에는 행 수가 예상과 같은지 먼저 봐요.

### LEFT JOIN — 지식이 없는 분류도 남기기

```
SELECT c.name, k.id AS knowledge_id, k.title
FROM categories AS c              -- 왼쪽: 모든 분류
LEFT JOIN knowledge AS k          -- 오른쪽: 연결된 지식
  ON k.category_id = c.id
ORDER BY c.id, k.id;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT c.name, k.id AS knowledge_id, k.title FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id ORDER BY c.id, k.id; | name \| knowledge_id \| title learning \| 1 \| 실습 파일 보관 learning \| 3 \| 질문 작성 environment \| 2 \| 서버 실행 reference \| NULL \| NULL | 4행 — 지식 없는 reference도 남고 오른쪽이 NULL |
| 같은 SQL을 (INNER) JOIN으로 | name \| knowledge_id \| title learning \| 1 \| 실습 파일 보관 learning \| 3 \| 질문 작성 environment \| 2 \| 서버 실행 | 3행 — reference가 사라져요 |

|  | JOIN | LEFT JOIN |
|---|---|---|
| 남는 행 | 짝 있는 행만 | 왼쪽 표 행은 전부 |
| 짝이 없으면 | 행이 빠짐 | 오른쪽 열이 NULL |
| S0 결과 | 3행 | 4행 (reference + NULL) |

🤔 어느 쪽이 왼쪽이야? — FROM에 쓴 표가 왼쪽이에요. "빠지면 안 되는 쪽"을 FROM에 둬요. 분류 목록 화면이면 분류가 왼쪽이에요.

### GROUP BY + COUNT(k.id) — 분류별 지식 수

```
SELECT c.name AS category,
       COUNT(k.id) AS knowledge_count
FROM categories AS c
LEFT JOIN knowledge AS k
  ON k.category_id = c.id
GROUP BY c.id, c.name
ORDER BY c.id;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT c.name AS category, COUNT(k.id) AS knowledge_count ... LEFT JOIN ... GROUP BY c.id, c.name ORDER BY c.id; | category \| knowledge_count learning \| 2 environment \| 1 reference \| 0 | 한 행 = 분류 한 개 |
| COUNT(k.id) 대신 COUNT(*) | category \| knowledge_count learning \| 2 environment \| 1 reference \| 1 | reference가 1 — NULL로 채운 보존 행까지 셌어요 |
| LEFT JOIN 뒤에 WHERE k.title LIKE '%실%' | category \| knowledge_count learning \| 1 environment \| 1 | WHERE가 NULL 행을 걸러 0인 분류가 사라짐 |
| 조건을 ON 안으로: ON k.category_id = c.id AND k.title LIKE '%실%' | category \| knowledge_count learning \| 1 environment \| 1 reference \| 0 | 분류 3개가 다 남고 0도 보여요 |

| 표현 | 세는 것 | reference (지식 0개) |
|---|---|---|
| COUNT(*) | 그룹의 행 수 (NULL로 채운 줄도 1행) | 1 ❌ |
| COUNT(k.id) | k.id가 NULL이 아닌 행 수 | 0 ✅ |

🤔 왜 GROUP BY c.id, c.name 둘 다? — SELECT에 쓴 c.name이 그룹마다 하나로 정해져야 해요. 이름이 UNIQUE라 c.name만으로도 되지만, PK인 c.id로 묶으면 "분류 한 개 = 한 행"이 확실해요.

⚠️ LEFT JOIN 뒤 오른쪽 조건은 ON 안에. WHERE k.title LIKE ...로 쓰면 NULL 줄이 걸러져 0인 분류가 사라져요 (아래 결과 비교).

### COUNT · SUM · MIN · MAX — 본문 길이 요약

```
SELECT COUNT(*) AS knowledge_count,
       SUM(char_length(content)) AS total_chars,
       MIN(char_length(content)) AS min_chars,
       MAX(char_length(content)) AS max_chars
FROM knowledge;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| SELECT COUNT(*), SUM(char_length(content)), MIN(...), MAX(...) FROM knowledge; | knowledge_count \| total_chars \| min_chars \| max_chars 3 \| 64 \| 20 \| 24 | 강의자료의 3건 · 64자 · 20 · 24와 같아요 |
| SELECT id, char_length(content) FROM knowledge ORDER BY id; | id \| chars 1 \| 20 2 \| 20 3 \| 24 | 글자 수(공백 포함) — 내용이 맞는지는 알려 주지 않아요 |

| 함수 | 뜻 |
|---|---|
| COUNT(*) | 행 개수 |
| SUM(x) · AVG(x) | 합계 · 평균 |
| MIN(x) · MAX(x) | 가장 작은 · 큰 값 |

💡 집계 결과는 GROUP BY가 없으면 표 전체가 한 행으로 줄어요. 그리고 이 수치는 본문 내용이 맞는지는 알려 주지 않아요.

### HAVING — 묶은 뒤 그룹 고르기

```
SELECT c.name, COUNT(k.id) AS knowledge_count
FROM categories AS c
LEFT JOIN knowledge AS k ON k.category_id = c.id
GROUP BY c.id, c.name
HAVING COUNT(k.id) >= 2
ORDER BY c.name;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| ... GROUP BY c.id, c.name HAVING COUNT(k.id) >= 2 ORDER BY c.name; | name \| knowledge_count learning \| 2 | 묶은 뒤 '2건 이상인 분류'만 |
| HAVING COUNT(k.id) = 0 | name \| knowledge_count reference \| 0 | 아직 비어 있는 분류 찾기 |

|  | WHERE | HAVING |
|---|---|---|
| 언제 | 묶기 전 | 묶은 뒤 |
| 무엇을 거르나 | 행 | 그룹 |
| 집계 함수 | 못 씀 (WHERE COUNT(...) ❌) | 씀 |

### 실패 테스트 — 오류 확인 → ROLLBACK → 기존 행 확인

```
BEGIN;
INSERT INTO knowledge (title, content, category_id)
VALUES ('오류 확인', '', 1);
-- 기대: knowledge_content_length 위반 (SQLSTATE 23514)
-- 오류를 읽은 뒤 이것만 별도로 실행
ROLLBACK;
SELECT COUNT(*) FROM knowledge;      -- 시작과 같은 개수인지
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음) · SQLSTATE는 PostgreSQL 공식 오류 코드표 값

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| BEGIN; INSERT ... VALUES ('오류 확인', '', 1); | 오류: CHECK constraint failed: knowledge_content_length | 기대한 제약(knowledge_content_length)으로 실패 · PG SQLSTATE 23514 |
| ROLLBACK; SELECT COUNT(*) FROM knowledge; | COUNT(*) 3 | 실패한 행은 저장되지 않음 — 기존 3건 그대로 |
| 필수 title = NULL | 오류: NOT NULL constraint failed: knowledge.title | PG SQLSTATE 23502 |
| 분류 learning 중복 | 오류: UNIQUE constraint failed: categories.name | PG SQLSTATE 23505 |
| content = '' | 오류: CHECK constraint failed: knowledge_content_length | PG SQLSTATE 23514 |
| 없는 category_id = 99 | 오류: FOREIGN KEY constraint failed | PG SQLSTATE 23503 |
| 실패 뒤 SELECT COUNT(*) | COUNT(*) 3 | 네 번 실패해도 3건 그대로 |

| 실패 입력 | SQLSTATE | 뜻 / 회복 |
|---|---|---|
| title = NULL | 23502 | NOT NULL 위반 |
| 분류 'learning' 중복 | 23505 | UNIQUE 위반 |
| content = '' | 23514 | 길이 CHECK 위반 |
| 없는 category_id | 23503 | FK 위반 · 부모를 확인 |
| 오류 뒤 계속 SQL | 25P02 | 트랜잭션 중단 · ROLLBACK부터 |

🤔 한 번에 한 가지만 틀리게? — 제목도 비우고 본문도 비우면 어느 규칙이 막았는지 흐려져요. 분류 1이 실제로 있는지 먼저 확인하고, 나머지 값은 정상으로 둬요. 다음 ID 번호는 비교하지 않아요 — 실패해도 번호는 쓰였을 수 있어요.

### ON DELETE RESTRICT — 참조 중인 부모 삭제

```
-- 지식 한 건 삭제: 그 행만 사라짐
DELETE FROM knowledge WHERE id = …;

-- 참조 중인 분류 삭제: 거부
DELETE FROM categories WHERE name = 'learning';
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| DELETE FROM categories WHERE name = 'learning'; | 오류: FOREIGN KEY constraint failed | 연결된 지식이 있어 RESTRICT가 거부 (PG 23503) |
| SELECT COUNT(*) FROM knowledge WHERE category_id = 1 | COUNT(*) 2 | 자식 지식도 그대로 — 삭제가 번지지 않음 |
| DELETE FROM categories WHERE name = 'reference' RETURNING id, name; | id \| name 3 \| reference | 참조하는 지식이 없는 분류는 지워져요 |
| DELETE FROM knowledge WHERE id = 1 RETURNING id; | id 1 | 지식 한 건 삭제는 부모와 무관 (연습 후 다시 넣었어요) |

| 삭제 정책 | 부모(분류)를 지우면 |
|---|---|
| RESTRICT (이번 설계) | 자식이 있으면 거부 |
| CASCADE | 자식도 같이 삭제 ⚠️ |
| SET NULL | 자식의 FK를 NULL로 (그 열이 NOT NULL이면 불가) |

⚠️ CASCADE를 "편해서" 켜지 않아요. 분류 하나 지우다 안내문 전체가 사라질 수 있어요. 업무 정책(분류를 없애면 지식은 어디로?)을 먼저 정해요.

### DEFAULT — 생략한 값 vs 직접 넣은 NULL

```
BEGIN;
CREATE TABLE defaults_try (
  label TEXT DEFAULT '미지정'
);
INSERT INTO defaults_try DEFAULT VALUES;          -- 값 생략
INSERT INTO defaults_try (label) VALUES (NULL);   -- NULL 직접
SELECT label, label IS NULL AS is_missing
FROM defaults_try;
ROLLBACK;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음) · is_missing의 1/0 = PG의 true/false

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| INSERT DEFAULT VALUES; INSERT ... VALUES (NULL); SELECT label, label IS NULL | label \| is_missing 미지정 \| 0 NULL \| 1 | 생략 → '미지정', 직접 넣은 NULL → 그대로 NULL |

| 입력 | 저장되는 값 |
|---|---|
| 값 생략 (DEFAULT VALUES) | '미지정' |
| NULL을 직접 | NULL (DEFAULT가 고쳐 주지 않음) |
| + NOT NULL을 걸면 | NULL 입력은 거부 (23502) |

### BEGIN → UPDATE → ROLLBACK — 세 번 비교

```
SELECT id, content FROM knowledge WHERE id = 2;   -- ① 시작 본문
BEGIN;
UPDATE knowledge
SET content = '취소할 임시 안내입니다.'
WHERE id = 2;
SELECT id, content FROM knowledge WHERE id = 2;   -- ② 작업 중
ROLLBACK;
SELECT id, content FROM knowledge WHERE id = 2;   -- ③ 취소 후
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| ① 시작 SELECT | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. | 현재 본문 기록 |
| ② BEGIN; UPDATE ...; SELECT (같은 연결) | id \| content 2 \| 취소할 임시 안내입니다. | 작업 중: 임시 본문이 보여요 |
| ②' 그 순간 다른 연결에서 SELECT | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. | 아직 미확정이라 다른 연결엔 원래 본문 |
| ③ ROLLBACK; SELECT | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. | 시작 본문으로 복귀 · 행 수·ID 그대로 |

| 시점 | 보이는 본문 |
|---|---|
| ① 시작 | 원래 본문 |
| ② 작업 중 (같은 연결) | 임시 본문 — 다른 연결에선 아직 원래 본문 |
| ③ ROLLBACK 뒤 | 원래 본문 · 행 수·ID 그대로 |

⚠️ BEGIN부터 ROLLBACK까지 같은 Query Tool 창에서 해요. 창마다 연결이 따로라, 다른 창에서 ROLLBACK하면 이 묶음과 상관없어요.

### COMMIT → 재접속해서 확인

```
BEGIN;
UPDATE knowledge
SET content = '실행 전에 가상환경을 확인합니다.'
WHERE id = 2
RETURNING id, content;
COMMIT;
-- 연결을 닫고 knowledge_lab에 다시 접속한 뒤
SELECT id, content FROM knowledge WHERE id = 2;
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| BEGIN; UPDATE ... RETURNING id, content; COMMIT; | id \| content 2 \| 실행 전에 가상환경을 확인합니다. | 확정 |
| 연결을 닫고 다시 접속 → SELECT | id \| content 2 \| 실행 전에 가상환경을 확인합니다. | 새 연결에서도 수정 본문 |
| 이어서 ROLLBACK; | 오류: cannot rollback - no transaction is active | 열린 트랜잭션이 없음 (PG는 경고 'there is no transaction in progress') |
| 다시 SELECT | id \| content 2 \| 실행 전에 가상환경을 확인합니다. | 이미 COMMIT한 값은 ROLLBACK으로 못 되돌려요 |

| 포인트 | 뜻 |
|---|---|
| COMMIT | 확정 — 이제 S0의 본문은 달라졌어요. 이후는 실제 현재값 기준 |
| 재접속 | 내 미확정 변경이 아니라 정말 저장된 값인지 확인 |
| 서버를 강제로 끄지 않기 | 단순 재접속 ≠ 서버 장애 시험 |

### 원자성 — 두 번째 변경이 실패하면 첫 변경도 취소

```
SELECT id, content FROM knowledge WHERE id IN (2, 3) ORDER BY id;  -- 먼저 원문 기록
BEGIN;
UPDATE knowledge
SET content = '확정되면 안 되는 임시 본문'
WHERE id = 2;
UPDATE knowledge SET content = '' WHERE id = 3;
-- 두 번째 문장의 CHECK 오류를 확인
-- 이후 별도 실행
ROLLBACK;
SELECT id, content FROM knowledge
WHERE id IN (2, 3) ORDER BY id;                                    -- 둘 다 원문
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 시작: 2·3번 원문 | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. 3 \| 오류 메시지와 시도한 작업을 함께 적습니다. |  |
| BEGIN; UPDATE 2번 → 성공, UPDATE 3번 content = '' | 오류: CHECK constraint failed: knowledge_content_length | 두 번째 문장이 CHECK에 걸림 |
| (롤백 전) 2·3번 | id \| content 2 \| 확정되면 안 되는 임시 본문 3 \| 오류 메시지와 시도한 작업을 함께 적습니다. | SQLite는 실패한 문장만 취소해서 2번 변경이 아직 보여요. PG는 이 SELECT부터 25P02(트랜잭션 중단)로 거부 |
| ROLLBACK; 2·3번 | id \| content 2 \| 프로젝트 폴더에서 서버를 실행합니다. 3 \| 오류 메시지와 시도한 작업을 함께 적습니다. | 묶음 전체 취소 — 2번 첫 변경도 남지 않음 |

🤔 첫 UPDATE는 성공했는데 왜 취소돼? — 둘이 한 묶음이라서요. 묶음은 전부 확정 아니면 전부 취소예요. PostgreSQL은 두 번째 오류 뒤 이 묶음을 중단 상태로 만들어 COMMIT해도 확정하지 않아요.

### pg_indexes 확인 → CREATE INDEX

```
SELECT indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename = 'knowledge';
-- 같은 이름/정의가 없는지 확인한 뒤 한 번 실행
CREATE INDEX idx_knowledge_category_id
ON knowledge (category_id);
```

▶ 실제로 돌려 보면 S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음) · PG는 pg_indexes, SQLite는 sqlite_master로 목록을 봐요

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 인덱스 목록 (만들기 전) | name (0행) | FK를 만들어도 자식 열 인덱스는 자동으로 안 생겨요 (PG도 같음. PG엔 PK용 knowledge_pkey는 있음) |
| CREATE INDEX idx_knowledge_category_id ON knowledge (category_id); | name \| sql idx_knowledge_category_id \| CREATE INDEX idx_knowledge_category_id ON knowledge (category_id) | 새 탐색 구조 |
| 같은 이름으로 한 번 더 | 오류: index idx_knowledge_category_id already exists | 그래서 만들기 전에 목록부터 확인 |
| 인덱스 뒤 WHERE category_id = 1 결과 | id \| title 1 \| 실습 파일 보관 3 \| 질문 작성 | 결과 행은 그대로 — 바뀌는 건 찾아가는 길 |

| 부분 | 뜻 |
|---|---|
| pg_indexes | 인덱스 목록을 보여 주는 시스템 뷰 |
| idx_knowledge_category_id | 인덱스 이름 (관례: idx_표_열) |
| ON knowledge (category_id) | 어느 표의 어느 열로 찾아가게 할지 |

💡 PK·UNIQUE 열엔 인덱스가 자동으로 있어요(knowledge_pkey 등). FK 열엔 자동으로 안 생겨서 분류 조건·JOIN이 잦으면 직접 만들어요. 읽기는 빨라질 수 있지만 쓰기·공간 비용이 생겨요.

### EXPLAIN · EXPLAIN (ANALYZE, BUFFERS)

```
EXPLAIN                          -- 예상 계획만 (실행 안 함)
SELECT id, title
FROM knowledge
WHERE category_id = 1;

EXPLAIN (ANALYZE, BUFFERS)       -- 실제로 실행하고 측정
SELECT id, title
FROM knowledge
WHERE category_id = 1;
```

▶ 실제로 돌려 보면 SQLite의 EXPLAIN QUERY PLAN 실제 출력 (PG의 EXPLAIN 표기와 다름)

| 📥 입력 | 📤 출력 | 뜻 |
|---|---|---|
| 인덱스 전: WHERE category_id = 1 | SCAN knowledge | 표 전체를 훑음 (PG의 Seq Scan에 해당) |
| 인덱스 후: 같은 SQL | SEARCH knowledge USING INDEX idx_knowledge_category_id (category_id=?) | 인덱스로 바로 찾아감 |
| PG에서 3행짜리 표라면? | PG = Seq Scan이 나와도 정상 (강의자료 p53) | 행이 적으면 전체를 읽는 게 더 싸다고 판단할 수 있어요 — 계획 이름을 정답으로 외우지 않기 |

| 출력에서 | 읽는 법 |
|---|---|
| Seq Scan on knowledge | 표 전체 훑기 — 작은 표에선 정상 |
| Index Scan / Bitmap Heap Scan | 인덱스로 찾아가기 |
| cost=시작..끝 | 시작·끝까지의 예상 비용 — 상대 단위, 밀리초 아님 |
| rows= | 예상 행 수 / ANALYZE면 actual ... rows=가 실제 |
| Buffers: shared hit= | 읽은 블록 수 (BUFFERS 옵션) |

⚠️ ANALYZE는 SQL을 진짜로 실행해요. 이 수업에선 SELECT에만 써요. 그리고 더 빠른 계획이나 특정 계획 이름을 정답으로 요구하지 않아요.

## 📖 Day18 용어 (29개)
- **seed** (random_state (난수 고정값)): 무작위 과정(추출, 특성 후보, 분할 섞기 등)을 매번 같게 만드는 번호. 결과를 재현하려고 고정해요(예: 42).
- **SQL** (Structured Query Language): DB에 '어느 표에서 어떤 행을 보여 줘 / 바꿔 줘'라고 말하는 언어. 구조를 만드는 CREATE와 데이터를 읽고 바꾸는 SELECT·INSERT·UPDATE·DELETE가 있어요.
- **DBMS** (Database Management System (관계형이면 RDBMS)): 데이터를 실제로 저장하고 SQL·규칙·동시 접근을 처리하는 프로그램. PostgreSQL이 RDBMS예요.
- **pgAdmin** (pgAdmin 4 (PostgreSQL GUI 클라이언트)): 창으로 PostgreSQL 서버에 접속해 SQL을 보내고 Data Output·Messages로 결과를 보는 도구. 창을 닫아도 서버는 안 꺼져요.
- **psql** (psql (PostgreSQL 터미널 클라이언트)): 터미널에서 쓰는 PostgreSQL 접속 도구. \dt·\d·\q 같은 역슬래시 명령은 SQL이 아니라 psql 전용이에요.
- **테이블** (Table (관계형 DB의 표)): 같은 종류의 데이터를 행과 열로 담는 표. 한 행 = 한 건(지식 하나), 한 열 = 공통 항목(제목, 본문).
- **PK** (Primary Key (기본키)): 표에서 행 하나를 콕 집어 구별하는 열. 값이 겹치면 안 되고 NULL도 안 돼요. 이번엔 id.
- **FK** (Foreign Key (외래키)): 다른 표(부모)의 PK를 가리키는 열. 저장할 때 그 번호의 부모 행이 실제로 있는지 DB가 검사해요. 이번엔 knowledge.category_id → categories.id.
- **IDENTITY** (Identity Column (자동 번호 열)): 값을 생략하면 DB가 번호를 발급하는 열. 안에서 시퀀스(번호 발급기)를 쓰고, 취소·실패한 번호는 다시 쓰지 않을 수 있어서 빈 번호가 생겨요.
- **JOIN** (JOIN (INNER JOIN)): 조회할 때 두 표에서 ON 조건으로 짝이 맞는 행을 붙여 한 줄로 보여 주는 것. 짝이 없는 행은 빠져요.
- **LEFT JOIN** (LEFT (OUTER) JOIN): 왼쪽(FROM) 표의 행은 모두 남기고, 짝이 없으면 오른쪽 열을 NULL로 채우는 JOIN. 지식이 0개인 분류도 결과에 남아요.
- **GROUP BY** (GROUP BY (묶어서 집계)): 같은 값끼리 행을 묶어 그룹마다 COUNT·SUM 등을 계산. 결과 한 행의 단위가 '그룹 하나'로 바뀌어요.
- **HAVING** (HAVING (그룹 조건)): GROUP BY로 묶은 뒤 그룹을 거르는 조건. 묶기 전 행을 거르는 WHERE와 순서가 달라요.
- **NULL** (NULL (값 없음 / 모름)): 값이 없거나 모른다는 표시. 빈 문자열 ''과 달라요. = NULL로는 비교가 안 돼서 IS NULL로 확인해요.
- **제약조건** (Constraint (제약조건)): DB가 저장 때마다 검사하는 규칙. PRIMARY KEY·FOREIGN KEY·NOT NULL·UNIQUE·CHECK가 있고, 어기면 행을 저장하지 않고 오류를 내요.
- **CHECK** (CHECK 제약조건): 값이 정한 조건을 만족하는지 검사. 예: char_length(content) BETWEEN 1 AND 2000 → 빈 본문 거부(SQLSTATE 23514).
- **UNIQUE** (UNIQUE 제약조건): 같은 값이 두 번 들어가지 못하게 막아요. 분류 이름은 UNIQUE, 지식 제목은 아님.
- **ON DELETE RESTRICT** (FK 삭제 정책 (RESTRICT / CASCADE)): 부모 행을 지울 때 자식이 있으면? RESTRICT는 거부, CASCADE는 자식까지 같이 삭제. 이번 설계는 RESTRICT.
- **DEFAULT** (DEFAULT (기본값)): 값을 생략했을 때 채우는 값. 직접 NULL을 넣으면 고쳐 주지 않아요.
- **DDL** (Data Definition Language / Data Manipulation Language): DDL은 CREATE TABLE처럼 구조를 만드는 SQL, DML은 INSERT·UPDATE·DELETE처럼 데이터를 바꾸는 SQL.
- **ERD** (Entity Relationship Diagram): 표와 열, 표끼리의 관계(1:N, N:M)를 그린 설계도. 그림일 뿐이고 실제 검사는 CREATE TABLE의 FK가 해요.
- **RETURNING** (RETURNING 절): INSERT·UPDATE·DELETE가 바꾼 행을 그 자리에서 돌려받는 PostgreSQL 문법. 새로 발급된 id를 예상하지 않고 바로 읽어요.
- **COMMIT** (COMMIT (확정)): BEGIN 이후의 변경을 확정. 이후 다른 연결·재접속에서도 보이고, ROLLBACK으로 되돌릴 수 없어요.
- **ROLLBACK** (ROLLBACK (취소)): 지금 열린 트랜잭션의 미확정 변경을 모두 취소. DB 전체를 과거로 돌리는 버튼은 아니에요.
- **BEGIN** (BEGIN (트랜잭션 시작)): 여러 변경을 한 묶음으로 시작. 이후 변경은 COMMIT 또는 ROLLBACK 전까지 미확정이고 다른 연결엔 안 보여요.
- **원자성** (Atomicity (원자성)): 트랜잭션 안의 변경은 전부 반영되거나 전부 취소되거나 둘 중 하나. 반쯤 바뀐 상태로 남지 않아요.
- **SQLSTATE** (SQLSTATE (표준 오류 코드)): DB 오류의 종류를 나타내는 5자리 코드. 23502 NOT NULL, 23503 FK, 23505 UNIQUE, 23514 CHECK, 25P02 중단된 트랜잭션. HTTP 상태 코드와는 별개예요.
- **인덱스** (Index (B-tree 인덱스)): 표와 따로 만드는 탐색 구조. 조건에 맞는 행을 표 전체를 훑지 않고 찾게 도와요. 결과는 같고 찾아가는 길만 달라요. 의미 검색 인덱스와는 다른 것.
- **EXPLAIN** (EXPLAIN (실행계획 보기)): DB가 SQL을 어떤 방법으로 처리할지 보여 줘요. Seq Scan = 전체 훑기, Index Scan = 인덱스 사용. cost는 밀리초가 아니라 상대 단위. ANALYZE를 붙이면 실제로 실행해요.
