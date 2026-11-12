// 자동 생성 파일 — 직접 고치지 말고 terms/*.json 을 고친 뒤 build_glossary.py 를 실행하세요.
window.GLOSSARY = [
 {
  "term": "머신러닝",
  "aliases": [
   "Machine Learning"
  ],
  "full": "Machine Learning (기계학습)",
  "short": "사람이 규칙을 직접 짜는 대신, 데이터(입력과 정답)를 보여 주고 컴퓨터가 규칙을 스스로 찾게 하는 방법. 그렇게 학습된 결과물을 '모델'이라고 불러요.",
  "analogy": "공식을 외워 주는 대신 예제 문제와 정답을 잔뜩 풀게 해서 요령을 익히게 하는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "회귀",
  "aliases": [
   "Regression"
  ],
  "full": "Regression (회귀)",
  "short": "정답이 연속적인 숫자인 예측 문제. 예: 이 요청은 몇 ms 걸릴까? 지표는 MAE·RMSE처럼 '얼마나 빗나갔나'를 재요.",
  "analogy": "\"내일 기온은 몇 도?\"처럼 숫자로 답하는 질문.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "분류",
  "aliases": [
   "Classification"
  ],
  "full": "Classification (분류)",
  "short": "정답이 정해진 종류(클래스) 중 하나인 예측 문제. 예: 이 요청은 실패(1)할까 성공(0)할까? 지표는 F1처럼 '맞혔나·어떤 오류인가'를 세요.",
  "analogy": "\"내일 비가 올까, 안 올까?\"처럼 보기 중에서 고르는 질문.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "특성",
  "aliases": [
   "Feature",
   "features",
   "feature"
  ],
  "full": "Feature (특성, 입력 열)",
  "short": "모델에 넣는 입력 정보 한 가지. 표에서 열 하나가 특성 하나예요(예: 글자수, 허용 도구수). 행 = 사례, 열 = 특성.",
  "analogy": "집값을 맞힐 때 보는 '평수, 역까지 거리, 층수' 같은 항목들.",
  "first": 6,
  "days": [
   6,
   9
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "입력 X",
  "aliases": [],
  "full": "Input X (입력 특성 표)",
  "short": "모델이 보고 판단하는 특성들의 표(2차원: 행 × 열). 예측 시점에 실제로 알 수 있는 값만 넣어야 해요.",
  "analogy": "시험 문제지.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "정답 y",
  "aliases": [],
  "full": "Target y (정답, 타깃)",
  "short": "모델이 맞혀야 하는 값. 회귀에선 숫자(elapsed_ms), 분류에선 클래스(failed: 0/1).",
  "analogy": "문제지에 딸린 정답.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "라벨",
  "aliases": [
   "Label",
   "정답 라벨"
  ],
  "full": "Label (정답 라벨)",
  "short": "사람이 미리 붙여 둔 정답 값. 분류에서는 각 사례가 어느 클래스인지(예: 0=성공, 1=실패)를 뜻해요.",
  "analogy": "문제집 뒤의 정답지.",
  "first": 6,
  "days": [
   6,
   15
  ],
  "keyDays": []
 },
 {
  "term": "클래스",
  "aliases": [
   "classes_"
  ],
  "full": "Class (클래스, 범주 정답)",
  "short": "분류 문제에서 고를 수 있는 정답의 종류. 이 수업에선 0(성공)과 1(실패) 두 클래스예요. classes_는 모델이 기억한 클래스 순서.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "fit",
  "aliases": [
   ".fit()"
  ],
  "full": "fit (학습)",
  "short": "데이터를 보고 모델(또는 전처리)이 필요한 값을 배우는 단계. 예: 선형회귀는 계수·절편을, 스케일러는 평균·표준편차를 배워요. train으로만 해야 해요.",
  "analogy": "문제집을 풀며 공부하는 시간.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "predict",
  "aliases": [
   ".predict()"
  ],
  "full": "predict (예측)",
  "short": "fit으로 배운 규칙을 새 입력에 적용해 답을 내는 단계. 예측할 때는 다시 학습하지 않아요.",
  "analogy": "공부한 걸로 새 시험 문제를 푸는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "train/valid/test",
  "aliases": [
   "train 60",
   "valid 20",
   "test 20"
  ],
  "full": "Train / Validation / Test split (학습·검증·시험 분할)",
  "short": "데이터를 역할별로 나눈 것. train은 학습, valid는 후보 비교·선택, test는 끝까지 보관했다가 마지막에 한 번만 확인. 역할을 섞으면 점수가 실제보다 좋아 보여요.",
  "analogy": "교과서로 공부(train) → 모의고사로 공부법 고르기(valid) → 수능은 딱 한 번(test).",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "기준선",
  "aliases": [
   "baseline",
   "DummyRegressor",
   "DummyClassifier"
  ],
  "full": "Baseline (기준선 모델)",
  "short": "아무것도 배우지 않은 가장 단순한 예측(예: train 평균만 예측, 가장 많은 클래스만 예측). 새 모델은 최소한 이것보다 나아야 의미가 있어요.",
  "analogy": "공부 안 하고 전부 3번으로 찍었을 때의 점수.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "선형회귀",
  "aliases": [
   "LinearRegression",
   "Linear Regression"
  ],
  "full": "Linear Regression (선형회귀)",
  "short": "입력의 가중합 ŷ = w₁x₁ + … + b로 숫자를 예측하는 모델. 잔차제곱합이 가장 작아지는 w와 b를 찾아요.",
  "analogy": "흩어진 점들 사이로 가장 잘 맞는 직선 하나를 긋는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "가중합",
  "aliases": [
   "Weighted sum"
  ],
  "full": "Weighted sum (가중합)",
  "short": "각 입력에 가중치(계수)를 곱해 모두 더한 값. w₁x₁ + w₂x₂ + … + b.",
  "analogy": "과목마다 반영 비율을 곱해 합친 내신 점수.",
  "first": 6,
  "days": [
   6,
   11
  ],
  "keyDays": []
 },
 {
  "term": "계수",
  "aliases": [
   "coef_"
  ],
  "full": "Coefficient (계수, 기울기 w)",
  "short": "입력이 1 늘 때 예측이 얼마나 변하는지를 나타내는 학습된 숫자. 다른 입력을 고정했을 때의 변화일 뿐, 원인·결과의 증명은 아니에요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "절편",
  "aliases": [
   "intercept_"
  ],
  "full": "Intercept (절편 b)",
  "short": "모든 입력이 0일 때의 예측값. 직선이 세로축과 만나는 높이예요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "잔차",
  "aliases": [],
  "full": "Residual (잔차)",
  "short": "실제값 − 예측값. 점과 직선 사이의 세로 거리. 양수면 실제가 예측보다 크다는 뜻.",
  "analogy": "과녁 중심에서 화살이 빗나간 거리.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "잔차제곱합",
  "aliases": [],
  "full": "Sum of Squared Residuals (잔차제곱합)",
  "short": "잔차를 각각 제곱해서 모두 더한 값. +와 −가 서로 지워지지 않고, 크게 빗나간 점일수록 훨씬 큰 벌점을 받아요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "외삽",
  "aliases": [
   "Extrapolation"
  ],
  "full": "Extrapolation (외삽)",
  "short": "학습 데이터 범위 밖의 값을 예측하는 것. 직선이 범위 밖에서도 그대로 이어진다는 보장이 없어 믿기 어려워요.",
  "analogy": "키가 매년 5cm 컸다고 40살에 2m 넘을 거라 예상하는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "MAE",
  "aliases": [
   "mean_absolute_error",
   "Mean Absolute Error"
  ],
  "full": "Mean Absolute Error (평균 절대 오차)",
  "short": "|실제 − 예측|의 평균. 정답과 같은 단위(예: ms)이고 작을수록 좋아요.",
  "analogy": "매번 몇 분씩 늦었는지 평균 낸 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "MSE",
  "aliases": [
   "Mean Squared Error",
   "MSELoss",
   "nn.MSELoss",
   "평균제곱오차"
  ],
  "full": "Mean Squared Error (평균 제곱 오차)",
  "short": "(실제 − 예측)²의 평균. 큰 오차에 더 큰 벌점을 주지만 단위가 제곱(ms²)이 돼요.",
  "analogy": "",
  "first": 6,
  "days": [
   6,
   12
  ],
  "keyDays": []
 },
 {
  "term": "RMSE",
  "aliases": [
   "root_mean_squared_error",
   "Root Mean Squared Error"
  ],
  "full": "Root Mean Squared Error (평균 제곱근 오차)",
  "short": "MSE에 제곱근을 씌운 값. 정답과 같은 단위로 돌아오면서 큰 오차에 민감한 성질은 남아요. 이 수업 회귀의 주 지표.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "R²",
  "aliases": [
   "r2_score",
   "결정계수"
  ],
  "full": "Coefficient of determination (결정계수)",
  "short": "1 − 잔차제곱합 ÷ 전체제곱합. 평가 자료 y의 평균만 찍는 것보다 얼마나 나은지. 최대 1, 평균보다 못하면 음수도 나와요. 오류 비율이 아니에요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "로지스틱회귀",
  "aliases": [
   "LogisticRegression",
   "Logistic Regression"
  ],
  "full": "Logistic Regression (로지스틱회귀)",
  "short": "이름은 회귀지만 분류 모델. 선형 점수 z를 시그모이드로 0~1 확률로 바꾸고, 임계값과 비교해 클래스를 정해요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "시그모이드",
  "aliases": [
   "Sigmoid"
  ],
  "full": "Sigmoid function (시그모이드 함수)",
  "short": "어떤 숫자 z든 0~1 사이로 눌러 담는 함수. p = 1 / (1 + e⁻ᶻ). z=0이면 0.5.",
  "analogy": "아무리 세게 눌러도 0~1 눈금 안에만 머무는 용수철 저울.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "predict_proba",
  "aliases": [],
  "full": "predict_proba (클래스별 확률 예측)",
  "short": "각 클래스일 확률을 돌려주는 함수. 열 순서는 classes_를 따라요. predict는 이 확률을 임계값에서 잘라 0/1로 바꾼 결과예요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "임계값",
  "aliases": [
   "threshold",
   "Threshold",
   "이진화",
   "Thresholding"
  ],
  "full": "Threshold (판단 기준값)",
  "short": "확률이 이 값 이상(또는 초과)이면 양성(실패)이라고 부르는 선. 낮추면 놓침(FN)은 줄고 헛경고(FP)는 늘어요. valid에서만 정해요.",
  "analogy": "화재경보기의 감도 설정 — 예민하게 하면 토스트 연기에도 울려요.",
  "first": 6,
  "days": [
   6,
   12,
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "로그 손실",
  "aliases": [
   "logloss",
   "Log loss"
  ],
  "full": "Log loss (로그 손실)",
  "short": "분류 모델이 학습할 때 줄이는 값. 정답에 낮은 확률을 줄수록(틀린 쪽에 자신만만할수록) 벌점이 폭발적으로 커져요. 평가 지표(F1)와는 별개예요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "규제",
  "aliases": [
   "Regularization",
   "regularization",
   "REGULARIZATION"
  ],
  "full": "Regularization (규제)",
  "short": "계수가 지나치게 커지지 않게 벌점을 줘서 학습을 제어하는 것. 로지스틱회귀에선 C로 조절하며, C가 작을수록 규제가 세요. 스케일링과는 다른 개념.",
  "analogy": "답안을 너무 길고 복잡하게 쓰면 감점하는 규칙.",
  "first": 6,
  "days": [
   6,
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "수렴",
  "aliases": [
   "max_iter"
  ],
  "full": "Convergence (수렴)",
  "short": "학습이 반복하며 더 나아지지 않는 안정된 답에 도달하는 것. max_iter(반복 상한)가 너무 작으면 도달 전에 멈춰 수렴 경고가 나요.",
  "analogy": "",
  "first": 6,
  "days": [
   6,
   10
  ],
  "keyDays": []
 },
 {
  "term": "양성",
  "aliases": [
   "양성 클래스",
   "Positive"
  ],
  "full": "Positive class (양성 클래스)",
  "short": "찾고 싶은 쪽으로 정한 클래스. 이 수업에선 실패(1)가 양성이에요. Precision·Recall·F1은 양성을 기준으로 계산돼요.",
  "analogy": "건강검진에서 '양성'이 '찾던 그 병이 있다'는 뜻인 것처럼.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "혼동행렬",
  "aliases": [
   "confusion_matrix",
   "Confusion Matrix",
   "혼동 행렬"
  ],
  "full": "Confusion Matrix (혼동행렬)",
  "short": "실제(행) × 예측(열)으로 맞힘·틀림을 네 칸(TN, FP, FN, TP)에 세어 둔 표. 어떤 종류의 오류인지 나눠 볼 수 있어요.",
  "analogy": "\"맞힌 것 / 괜히 경고한 것 / 놓친 것\"을 칸별로 정리한 성적표.",
  "first": 6,
  "days": [
   6,
   11,
   14
  ],
  "keyDays": [
   6,
   14
  ]
 },
 {
  "term": "TP",
  "aliases": [
   "True Positive"
  ],
  "full": "True Positive (맞춘 양성)",
  "short": "실제 양성을 양성이라고 맞힌 경우(예: 실제 실패를 실패로 예측). 객체탐지에서는 정답 박스와 짝지어진 예측을 뜻해요.",
  "analogy": "",
  "first": 6,
  "days": [
   6,
   15
  ],
  "keyDays": []
 },
 {
  "term": "TN",
  "aliases": [
   "True Negative"
  ],
  "full": "True Negative (맞춘 음성)",
  "short": "실제 음성을 음성이라고 맞힌 경우(예: 실제 성공을 성공으로 예측). F1에는 들어가지 않아요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "FP",
  "aliases": [
   "False Positive",
   "오탐"
  ],
  "full": "False Positive (헛경고·오탐)",
  "short": "실제는 음성인데 양성이라고 한 경우(괜한 경고). 객체탐지에서는 정답과 짝이 없는 예측.",
  "analogy": "화재경보기가 토스트 연기에 울리는 것.",
  "first": 6,
  "days": [
   6,
   15
  ],
  "keyDays": []
 },
 {
  "term": "FN",
  "aliases": [
   "False Negative"
  ],
  "full": "False Negative (놓침)",
  "short": "실제는 양성인데 음성이라고 한 경우. 찾아야 할 것을 못 찾음.",
  "analogy": "진짜 불이 났는데 경보기가 안 울린 것.",
  "first": 6,
  "days": [
   6,
   15
  ],
  "keyDays": []
 },
 {
  "term": "Accuracy",
  "aliases": [
   "정확도",
   "accuracy_score",
   "accuracy"
  ],
  "full": "Accuracy (정확도)",
  "short": "전체 중 맞힌 비율. (TP + TN) ÷ 전체. 한쪽 클래스가 많으면 그쪽만 찍어도 높게 나와 속기 쉬워요.",
  "analogy": "",
  "first": 6,
  "days": [
   6,
   9
  ],
  "keyDays": []
 },
 {
  "term": "Precision",
  "aliases": [
   "정밀도",
   "precision_score"
  ],
  "full": "Precision (정밀도)",
  "short": "양성이라고 한 것 중 진짜 양성인 비율. TP ÷ (TP + FP). 헛경고가 늘면 낮아져요.",
  "analogy": "\"네가 가리킨 것 중 몇 개가 맞았어?\"",
  "first": 6,
  "days": [
   6,
   8,
   14,
   15
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "Recall",
  "aliases": [
   "재현율",
   "recall_score"
  ],
  "full": "Recall (재현율)",
  "short": "실제 양성 중 모델이 찾아낸 비율. TP ÷ (TP + FN). 놓침이 늘면 낮아져요.",
  "analogy": "\"있는 것 중에 몇 개나 찾아냈어?\"",
  "first": 6,
  "days": [
   6,
   8,
   14,
   15
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "F1",
  "aliases": [
   "F1 score",
   "f1_score",
   "F1-score"
  ],
  "full": "F1 score",
  "short": "Precision과 Recall의 조화평균. 2TP ÷ (2TP + FP + FN). 둘 중 하나라도 낮으면 같이 낮아지고 TN은 반영하지 않아요. 이 수업 분류의 주 지표.",
  "analogy": "",
  "first": 6,
  "days": [
   6,
   8,
   10,
   14
  ],
  "keyDays": [
   6,
   8
  ]
 },
 {
  "term": "zero_division",
  "aliases": [],
  "full": "zero_division (0으로 나누기 처리)",
  "short": "양성 예측이 0건이라 Precision 분모가 0일 때 어떤 값으로 기록할지 정하는 인자. 0으로 두면 경고 없이 0을 기록해요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "KNN",
  "aliases": [
   "K-최근접 이웃",
   "K-Nearest Neighbors",
   "KNeighborsClassifier",
   "KNeighborsRegressor"
  ],
  "full": "K-Nearest Neighbors (K-최근접 이웃)",
  "short": "새 입력과 가장 가까운 학습 사례 K개를 찾아, 분류면 다수결 투표·회귀면 평균으로 예측하는 모델. fit은 사례를 기억만 하고 일은 predict 때 해요.",
  "analogy": "이사 온 동네가 어떤 곳인지 가장 가까운 이웃 몇 집에 물어보는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "n_neighbors",
  "aliases": [],
  "full": "n_neighbors (K, 참고할 이웃 수)",
  "short": "KNN이 투표·평균에 쓸 이웃 수. 작으면 잡음에 휘둘리고, 크면 소수 클래스의 패턴을 놓칠 수 있어요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "유클리드 거리",
  "aliases": [
   "유클리드"
  ],
  "full": "Euclidean distance (직선 거리)",
  "short": "두 점 사이의 직선 거리. 각 특성 차이를 제곱해 더한 뒤 제곱근. KNN의 기본 거리(p=2).",
  "analogy": "지도 위에서 자로 잰 거리.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "맨해튼 거리",
  "aliases": [
   "맨해튼"
  ],
  "full": "Manhattan distance (가로+세로 거리)",
  "short": "각 특성 차이의 절댓값을 그냥 더한 거리(p=1).",
  "analogy": "바둑판 같은 도심에서 블록을 따라 걸은 거리.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "스케일링",
  "aliases": [
   "Scaling",
   "스케일러"
  ],
  "full": "Feature Scaling (특성 스케일링)",
  "short": "단위가 다른 특성들의 눈금을 비슷하게 맞추는 전처리. 거리로 판단하는 KNN에선 단위가 큰 특성이 거리를 독차지하지 못하게 해 줘요. 점수 상승을 보장하진 않아요.",
  "analogy": "cm와 km가 섞인 지도를 같은 축척으로 맞추는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "표준화",
  "aliases": [
   "StandardScaler",
   "Standardization"
  ],
  "full": "Standardization (표준화)",
  "short": "각 특성에서 train 평균을 빼고 train 표준편차로 나누는 변환. 결과는 평균 0·표준편차 1 근처. 정규분포로 바꿔 주는 건 아니에요.",
  "analogy": "반마다 난이도가 다른 시험 점수를 '평균보다 몇 칸 위인가'로 바꿔 비교하는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "최소–최대 스케일링",
  "aliases": [
   "MinMaxScaler",
   "MinMax"
  ],
  "full": "Min-Max Scaling",
  "short": "train의 최솟값·최댓값으로 (x − 최소) ÷ (최대 − 최소)를 계산해 0~1로 맞추는 변환. train 범위 밖의 새 값은 0~1을 벗어날 수 있어요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "벡터 정규화",
  "aliases": [
   "Normalizer"
  ],
  "full": "Vector normalization (행 단위 정규화)",
  "short": "각 행(사례)의 벡터 길이를 1로 맞추는 변환. 열 단위인 표준화·MinMax와 방향이 달라요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "표준편차",
  "aliases": [
   "Standard deviation"
  ],
  "full": "Standard deviation (표준편차)",
  "short": "값들이 평균에서 평균적으로 얼마나 퍼져 있는지 나타내는 숫자. 클수록 들쭉날쭉.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "전처리",
  "aliases": [
   "Preprocessing"
  ],
  "full": "Preprocessing (전처리)",
  "short": "모델에 넣기 전에 데이터를 다듬는 단계(결측 채우기, 스케일링, 원-핫 인코딩 등). 전처리가 배우는 값도 train에서만 배워야 해요.",
  "analogy": "요리 전에 재료를 씻고 손질하는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "결측",
  "aliases": [
   "결측값",
   "Missing value",
   "NaN",
   "Inf"
  ],
  "full": "Missing value (결측값)",
  "short": "데이터가 비어 있는 칸. 모델 대부분은 빈칸을 못 받아서 채우거나(대체) 처리해야 해요.",
  "analogy": "설문지에서 답을 안 적은 칸.",
  "first": 6,
  "days": [
   6,
   10
  ],
  "keyDays": []
 },
 {
  "term": "결측 대체",
  "aliases": [
   "SimpleImputer",
   "imputer"
  ],
  "full": "Imputation (결측 대체)",
  "short": "빈칸을 정한 규칙의 값으로 채우는 것. 수치는 train 중앙값, 범주는 train 최빈값으로 채웠어요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "중앙값",
  "aliases": [
   "median"
  ],
  "full": "Median (중앙값)",
  "short": "값을 크기순으로 줄 세웠을 때 가운데 값. 극단값에 덜 끌려가요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "최빈값",
  "aliases": [
   "most_frequent"
  ],
  "full": "Mode (최빈값)",
  "short": "가장 자주 나온 값. 평균을 낼 수 없는 범주형 결측을 채울 때 써요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "수치형",
  "aliases": [],
  "full": "Numerical feature (수치형 특성)",
  "short": "크기 비교와 계산이 되는 숫자 특성(예: 글자수, 허용 도구수).",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "범주형",
  "aliases": [],
  "full": "Categorical feature (범주형 특성)",
  "short": "종류를 나타내는 특성(예: task_type, priority). 숫자로 바로 계산할 수 없어 원-핫 인코딩 등으로 바꿔 넣어요.",
  "analogy": "혈액형, 지역명처럼 '어느 쪽인가'를 나타내는 값.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "원-핫 인코딩",
  "aliases": [
   "One-hot",
   "OneHotEncoder",
   "원핫"
  ],
  "full": "One-hot encoding (원-핫 인코딩)",
  "short": "범주 하나를 '그 범주면 1, 아니면 0'인 열 여러 개로 펼치는 변환. 이렇게 생긴 열을 더미 열이라고도 해요.",
  "analogy": "\"빨강/파랑/초록\" 대신 체크박스 3개 중 하나에만 ✔ 표시하는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "handle_unknown",
  "aliases": [],
  "full": "handle_unknown=\"ignore\"",
  "short": "train에 없던 새 범주가 들어왔을 때 오류 대신 그 범주 블록을 모두 0으로 처리하게 하는 OneHotEncoder 설정.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "Pipeline",
  "aliases": [
   "파이프라인"
  ],
  "full": "Pipeline (전처리 + 모델 묶음)",
  "short": "전처리 단계와 모델을 한 줄로 묶은 객체. fit(train)하면 전처리·모델을 train으로만 배우고, predict(valid)하면 배운 규칙으로 transform → predict만 해서 누수를 막아 줘요.",
  "analogy": "재료 손질부터 조리까지 정해진 순서대로 도는 주방 컨베이어 벨트.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": [
   6
  ]
 },
 {
  "term": "ColumnTransformer",
  "aliases": [],
  "full": "ColumnTransformer (열별 변환기)",
  "short": "열 묶음마다 다른 전처리를 적용하고 결과를 합쳐 주는 도구. 수치 열은 중앙값·스케일링, 범주 열은 최빈값·원-핫으로.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "transform",
  "aliases": [
   "fit_transform",
   "transforms",
   "transforms.Compose",
   "Transforms"
  ],
  "full": "transform / fit_transform",
  "short": "transform은 이미 배운 규칙(평균·중앙값 등)으로 데이터를 바꾸기만 하는 것. fit_transform은 배우기와 바꾸기를 한 번에(train에만).",
  "analogy": "",
  "first": 6,
  "days": [
   6,
   11
  ],
  "keyDays": []
 },
 {
  "term": "누수",
  "aliases": [
   "Data leakage",
   "leakage",
   "데이터 누수",
   "Leakage"
  ],
  "full": "Data Leakage (데이터 누수)",
  "short": "검증·테스트용 정보나 예측 시점에 알 수 없는 값(예: 실행 후에야 생기는 elapsed_ms)이 학습에 섞여 들어가 점수가 실제보다 좋게 나오는 문제.",
  "analogy": "시험 문제를 미리 보고 공부한 것.",
  "first": 6,
  "days": [
   6,
   12,
   14,
   15
  ],
  "keyDays": []
 },
 {
  "term": "하이퍼파라미터",
  "aliases": [
   "Hyperparameter",
   "hyperparameter"
  ],
  "full": "Hyperparameter (하이퍼파라미터)",
  "short": "학습 전에 사람이 정하는 설정값(예: K, weights, C, max_depth). train으로 자동 결정되는 계수와 달리 후보를 정해 valid·교차검증으로 비교해 골라요.",
  "analogy": "요리 레시피의 '불 세기·조리 시간' 설정.",
  "first": 6,
  "days": [
   6,
   10
  ],
  "keyDays": []
 },
 {
  "term": "학습 파라미터",
  "aliases": [],
  "full": "Learned parameter (학습 파라미터)",
  "short": "fit하면서 train 데이터로 자동으로 정해지는 값(예: 선형·로지스틱회귀의 계수와 절편).",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "과적합",
  "aliases": [
   "Overfitting",
   "오버피팅",
   "overfitting"
  ],
  "full": "Overfitting (과적합)",
  "short": "학습 자료의 우연한 잡음까지 외워서 train 점수만 높고 새 자료(valid) 점수는 낮은 상태. 두 점수 격차가 핵심 신호예요.",
  "analogy": "기출문제 답만 통째로 외워서 새 문제는 못 푸는 학생.",
  "first": 6,
  "days": [
   6,
   7,
   10
  ],
  "keyDays": [
   6,
   10
  ]
 },
 {
  "term": "과소적합",
  "aliases": [
   "Underfitting",
   "underfitting"
  ],
  "full": "Underfitting (과소적합)",
  "short": "모델이 너무 단순하거나 제약이 세서 필요한 규칙도 못 배운 상태. train과 valid 점수가 둘 다 낮아요.",
  "analogy": "공부를 너무 대충 해서 쉬운 문제도 틀리는 것.",
  "first": 6,
  "days": [
   6,
   10
  ],
  "keyDays": []
 },
 {
  "term": "일반화",
  "aliases": [
   "Generalization",
   "generalization"
  ],
  "full": "Generalization (일반화)",
  "short": "학습에 쓰지 않은 새 자료에서도 잘 맞히는 능력. 모델 평가의 진짜 목표예요.",
  "analogy": "처음 보는 응용문제도 풀어내는 실력.",
  "first": 6,
  "days": [
   6,
   10
  ],
  "keyDays": []
 },
 {
  "term": "선택 편향",
  "aliases": [
   "Selection bias"
  ],
  "full": "Selection bias (선택 편향)",
  "short": "같은 평가 자료를 여러 번 보며 가장 좋게 나온 후보를 고르면, 그 점수에 '운 좋게 맞은 몫'이 섞여 실제보다 낙관적이 되는 현상.",
  "analogy": "주사위를 여러 번 굴려 제일 잘 나온 판만 보고하는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "재현성",
  "aliases": [
   "Reproducibility"
  ],
  "full": "Reproducibility (재현성)",
  "short": "같은 데이터·코드·설정이면 언제 누가 돌려도 같은 결과가 나오는 성질. seed(난수 고정)뿐 아니라 데이터 버전·분할 ID·조건을 함께 기록해요.",
  "analogy": "레시피대로 하면 누가 만들어도 같은 맛이 나는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "joblib",
  "aliases": [
   "joblib.dump",
   "joblib.load"
  ],
  "full": "joblib (모델 저장·복원 도구)",
  "short": "학습한 Pipeline을 파일로 저장(dump)하고 다시 불러오는(load) 라이브러리. 불러오면 안의 코드가 실행될 수 있어 직접 만든 파일만 load해요.",
  "analogy": "게임 진행 상황을 세이브 파일로 저장했다가 불러오는 것.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "scikit-learn",
  "aliases": [
   "sklearn"
  ],
  "full": "scikit-learn (파이썬 머신러닝 라이브러리)",
  "short": "모델·전처리·지표·Pipeline을 fit/predict 같은 같은 방식으로 제공하는 라이브러리. pip로는 scikit-learn, 코드에선 sklearn으로 불러요.",
  "analogy": "머신러닝 도구가 한데 모인 공구 상자.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "numpy",
  "aliases": [
   "NumPy"
  ],
  "full": "NumPy (숫자 배열 계산 라이브러리)",
  "short": "숫자 배열(np.array)을 빠르게 계산하는 파이썬 라이브러리.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "pandas",
  "aliases": [
   "DataFrame"
  ],
  "full": "pandas (표 데이터 라이브러리)",
  "short": "엑셀 같은 표(DataFrame)로 데이터를 읽고 다루는 파이썬 라이브러리.",
  "analogy": "파이썬 안의 엑셀.",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "가상환경",
  "aliases": [
   ".venv",
   "venv"
  ],
  "full": "Virtual environment (가상환경, .venv)",
  "short": "프로젝트마다 따로 쓰는 파이썬과 패키지 묶음. 버전이 섞이지 않게 해 같은 코드가 같은 결과를 내도록 도와요.",
  "analogy": "프로젝트마다 따로 쓰는 전용 공구함.",
  "first": 6,
  "days": [
   6,
   9
  ],
  "keyDays": []
 },
 {
  "term": "합성 데이터",
  "aliases": [
   "Synthetic data"
  ],
  "full": "Synthetic data (합성 데이터)",
  "short": "실제 기록이 아니라 수업용으로 만들어 낸 가짜 데이터. 여기서 나온 점수는 실서비스 성능의 근거가 아니에요.",
  "analogy": "",
  "first": 6,
  "days": [
   6
  ],
  "keyDays": []
 },
 {
  "term": "Decision Tree",
  "aliases": [
   "결정트리",
   "의사결정나무",
   "DecisionTreeClassifier",
   "DecisionTreeRegressor"
  ],
  "full": "Decision Tree (결정 트리)",
  "short": "\"글자수 ≤ 500?\" 같은 예/아니오 질문을 이어 가며 자료를 나누고, 마지막 잎에서 예측하는 모델. 질문은 학습 자료로 자동 선택돼요.",
  "analogy": "스무고개 — 질문을 하나씩 던지며 답을 좁혀 가는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "루트",
  "aliases": [
   "루트 노드",
   "Root"
  ],
  "full": "Root node (루트 노드)",
  "short": "트리의 맨 위, 첫 번째 질문이 있는 곳.",
  "analogy": "스무고개의 첫 질문.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "내부 노드",
  "aliases": [
   "노드",
   "Node"
  ],
  "full": "Internal node (내부 노드)",
  "short": "루트 다음에 이어지는 중간 질문들. 자료를 다시 둘로 나눠요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "잎",
  "aliases": [
   "leaf",
   "Leaf",
   "잎 노드"
  ],
  "full": "Leaf node (잎 노드)",
  "short": "더 이상 나누지 않는 트리의 끝. 분류는 그 잎에 떨어진 학습 행의 클래스 비율, 회귀는 y 평균으로 예측해요.",
  "analogy": "스무고개 끝에 나온 최종 답.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "Gini 불순도",
  "aliases": [
   "Gini",
   "지니 불순도"
  ],
  "full": "Gini impurity (지니 불순도)",
  "short": "한 노드에 클래스가 얼마나 섞였는지 재는 값. 1 − Σ(각 클래스 비율)². 0이면 한 클래스만(깨끗), 이진 분류에선 0.5가 가장 섞인 상태. 트리는 이 값을 가장 많이 줄이는 질문을 골라요.",
  "analogy": "바구니에 사과·배가 반반이면 지저분(0.5), 사과만 있으면 깨끗(0).",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "불순도",
  "aliases": [
   "Impurity"
  ],
  "full": "Impurity (불순도)",
  "short": "노드 안에 서로 다른 정답이 섞인 정도. 낮을수록 그 노드의 예측을 자신 있게 할 수 있어요. 시험 점수나 오차가 아니에요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "탐욕적 방법",
  "aliases": [
   "Greedy"
  ],
  "full": "Greedy method (탐욕적 방법)",
  "short": "전체 최적을 한 번에 찾지 않고, 매 단계에서 '지금 가장 좋아 보이는' 선택을 하고 넘어가는 방식. 빠르지만 나중에 더 좋은 길은 놓칠 수 있어요.",
  "analogy": "갈림길마다 당장 가까워 보이는 길만 고르는 등산.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "max_depth",
  "aliases": [],
  "full": "max_depth (최대 깊이)",
  "short": "트리가 질문을 최대 몇 단계까지 할 수 있는지. 작으면 과소적합, None(제한 없음)이면 잎이 깨끗해질 때까지 나눠 과적합 위험.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "min_samples_leaf",
  "aliases": [],
  "full": "min_samples_leaf (잎의 최소 행 수)",
  "short": "각 잎에 있어야 하는 최소 학습 행 수. 키우면 행 1~2개짜리 잎(잡음 분할)을 막아요. '잎 개수'가 아니에요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "min_samples_split",
  "aliases": [],
  "full": "min_samples_split (분할 최소 행 수)",
  "short": "노드를 더 나누려면 그 노드에 최소 몇 행이 있어야 하는지. '나눌지 말지' 조건이에요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "가지치기",
  "aliases": [
   "ccp_alpha",
   "Pruning"
  ],
  "full": "Cost-Complexity Pruning (비용-복잡도 가지치기)",
  "short": "다 키운 트리에서 불순도를 별로 못 줄이는 가지를 잘라 단순하게 만드는 것. ccp_alpha가 클수록 더 많이 잘려요.",
  "analogy": "웃자란 나뭇가지를 쳐서 모양을 다듬는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "seed",
  "aliases": [
   "seed",
   "random_state",
   "manual_seed",
   "시드",
   "S0"
  ],
  "full": "random_state (난수 고정값)",
  "short": "무작위 과정(추출, 특성 후보, 분할 섞기 등)을 매번 같게 만드는 번호. 결과를 재현하려고 고정해요(예: 42).",
  "analogy": "카드를 섞는 방법을 정해 두어 매번 같은 순서가 나오게 하는 것.",
  "first": 7,
  "days": [
   7,
   9,
   13,
   18
  ],
  "keyDays": []
 },
 {
  "term": "앙상블",
  "aliases": [
   "Ensemble"
  ],
  "full": "Ensemble (앙상블)",
  "short": "여러 모델(여기선 트리 여러 개)의 예측을 묶어 하나의 답을 내는 방법. 대표 방식이 배깅(Random Forest)과 부스팅(XGBoost). 항상 점수가 오르는 건 아니에요.",
  "analogy": "한 사람 의견보다 여러 사람에게 물어 종합하는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "Random Forest",
  "aliases": [
   "랜덤 포레스트",
   "RandomForestClassifier",
   "RandomForestRegressor"
  ],
  "full": "Random Forest (랜덤 포레스트)",
  "short": "트리마다 다른 표본(중복 허용 추출)과 분할마다 다른 특성 후보를 줘서 서로 다른 트리를 독립적으로 여러 개 만들고, 분류는 확률·회귀는 예측을 평균하는 앙상블.",
  "analogy": "서로 다른 자료를 본 전문가 100명의 의견을 평균하는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "Bagging",
  "aliases": [
   "배깅"
  ],
  "full": "Bootstrap Aggregating (배깅)",
  "short": "중복 허용 추출로 만든 서로 다른 표본에서 모델을 각각 독립적으로 학습하고 결과를 평균하는 앙상블 방식. Random Forest가 대표.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "중복 허용 추출",
  "aliases": [
   "Bootstrap",
   "부트스트랩"
  ],
  "full": "Bootstrap sampling (중복 허용 추출)",
  "short": "자료에서 하나 뽑고 다시 넣고를 반복해 같은 크기의 표본을 만드는 방법. 어떤 행은 여러 번, 어떤 행은 한 번도 안 뽑혀서 트리마다 보는 자료가 달라져요.",
  "analogy": "제비뽑기 후 뽑은 제비를 다시 통에 넣고 또 뽑는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "n_estimators",
  "aliases": [],
  "full": "n_estimators (트리 개수 / 보정 회차)",
  "short": "Random Forest에선 숲의 트리 수, XGBoost에선 보정 트리를 추가하는 회차. 깊이가 아니에요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "max_features",
  "aliases": [],
  "full": "max_features (분할마다 볼 특성 후보 수)",
  "short": "Random Forest가 분할할 때마다 무작위로 살펴볼 특성의 수·비율. 작을수록 트리끼리 더 달라지지만 각 트리는 약해져요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "estimators_",
  "aliases": [],
  "full": "estimators_ (학습된 개별 트리 목록)",
  "short": "Random Forest 안에 학습된 트리들을 담은 목록. n_estimators=100이면 100개.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "n_jobs",
  "aliases": [],
  "full": "n_jobs (병렬 작업 수)",
  "short": "계산에 쓸 CPU 작업 수. -1이면 전부. 수업에선 재현성을 위해 1로 고정해요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "특성 중요도",
  "aliases": [
   "Feature importance",
   "feature_importances_"
  ],
  "full": "Feature importance (특성 중요도)",
  "short": "학습된 모델이 어떤 특성(열)에 얼마나 기댔는지 나타낸 값. '중요 = 원인'이 아니고, 영향의 방향(+/−)도 알려 주지 않아요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "MDI",
  "aliases": [
   "Mean Decrease in Impurity",
   "불순도 기반 중요도"
  ],
  "full": "Mean Decrease in Impurity (불순도 감소 기반 중요도)",
  "short": "각 분기에서 줄어든 불순도 × 그 분기가 다룬 자료 비중을 특성별로 더한 값(트리들 평균). 학습 자료 기준이고 합은 1. 값 종류가 많은 특성이 높게 나오기 쉬워요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "순열 중요도",
  "aliases": [
   "Permutation importance",
   "permutation_importance"
  ],
  "full": "Permutation importance (순열 중요도)",
  "short": "평가 자료에서 한 열의 값을 행끼리 뒤섞었을 때 점수가 얼마나 떨어지는지로 중요도를 재는 방법. 다시 학습하지 않으며 음수도 나올 수 있어요.",
  "analogy": "요리에서 소금만 몰래 다른 가루로 바꿔 보고 맛이 얼마나 망가지는지 보는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "get_feature_names_out",
  "aliases": [],
  "full": "get_feature_names_out() (변환 후 열 이름)",
  "short": "전처리를 거친 뒤 실제 열 이름과 순서를 돌려주는 함수. 원-핫으로 열이 늘어나니 중요도가 어느 열 것인지 알려면 필요해요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "더미",
  "aliases": [
   "더미 열",
   "Dummy variable"
  ],
  "full": "Dummy variable (더미 열)",
  "short": "원-핫 인코딩으로 생긴 0/1 열. 범주 2열이 더미 6열로 펼쳐졌어요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "부스팅",
  "aliases": [
   "Boosting"
  ],
  "full": "Boosting (부스팅)",
  "short": "앞 모델이 틀린 부분을 다음 모델이 보정하도록 트리를 순서대로 추가하는 앙상블 방식. 기본 점수에 보정값을 조금씩 누적해요.",
  "analogy": "오답 노트를 보며 틀린 유형만 골라 다시 공부하기를 반복하는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "XGBoost",
  "aliases": [
   "XGBClassifier",
   "XGBRegressor",
   "xgboost"
  ],
  "full": "eXtreme Gradient Boosting",
  "short": "손실을 줄이는 트리를 순서대로 추가하는 그래디언트 부스팅 구현. 트리 복잡도·잎 값에 규제를 걸어 과적합을 제어하고, sklearn처럼 fit/predict로 Pipeline에 연결돼요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "그래디언트 부스팅",
  "aliases": [
   "Gradient Boosting"
  ],
  "full": "Gradient Boosting (그래디언트 부스팅)",
  "short": "손실의 기울기(gradient) 정보를 따라 '지금 틀린 방향'을 다음 트리가 배우게 하는 부스팅. 제곱오차에선 그 방향이 곧 잔차예요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "학습률",
  "aliases": [
   "learning_rate",
   "Learning rate",
   "learning rate",
   "lr"
  ],
  "full": "Learning rate (학습률)",
  "short": "한 회차의 보정을 얼마나 반영할지 정하는 비율(예: 0.05). 작으면 조금씩 안정적으로 고치지만 회차가 더 필요해요.",
  "analogy": "운전대를 한 번에 확 꺾지 않고 조금씩 돌리는 정도.",
  "first": 7,
  "days": [
   7,
   9,
   10
  ],
  "keyDays": [
   9,
   10
  ]
 },
 {
  "term": "Hessian",
  "aliases": [
   "곡률"
  ],
  "full": "Hessian (손실의 곡률, 2차 미분)",
  "short": "손실이 얼마나 휘어 있는지 나타내는 값. XGBoost는 기울기와 함께 이걸로 보정값을 계산하고, min_child_weight는 자식 노드의 Hessian 합 하한이에요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "min_child_weight",
  "aliases": [],
  "full": "min_child_weight (자식 노드 최소 Hessian 합)",
  "short": "XGBoost에서 노드를 나눌 때 자식에 보정할 정보(Hessian 합)가 이만큼은 모여야 한다는 조건. 행 수(min_samples_leaf)와는 달라요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "subsample",
  "aliases": [
   "colsample_bytree"
  ],
  "full": "subsample / colsample_bytree (행·특성 부분 표본 비율)",
  "short": "XGBoost가 회차마다 쓸 행의 비율 / 트리마다 쓸 특성의 비율. 낮추면 다양성↑·과적합↓, 너무 낮으면 정보 부족.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "L2 규제",
  "aliases": [
   "reg_lambda",
   "L2"
  ],
  "full": "L2 regularization (L2 규제)",
  "short": "값의 제곱에 벌점을 줘 크게 튀지 않게 하는 규제. XGBoost의 reg_lambda는 잎 값에 걸어 한 번의 보정을 보수적으로 만들어요.",
  "analogy": "",
  "first": 7,
  "days": [
   7,
   10
  ],
  "keyDays": []
 },
 {
  "term": "objective",
  "aliases": [
   "binary:logistic",
   "reg:squarederror"
  ],
  "full": "objective (학습 목표 손실)",
  "short": "XGBoost가 학습하며 줄일 손실의 종류. 이진분류는 binary:logistic, 회귀는 reg:squarederror. 문제 유형에 맞춰야 해요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "eval_metric",
  "aliases": [],
  "full": "eval_metric (학습 진단 지표)",
  "short": "학습 중 진행을 확인하는 지표(logloss, rmse). 모델 선택 지표(F1)와는 별개예요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "hist",
  "aliases": [
   "tree_method"
  ],
  "full": "tree_method=\"hist\" (히스토그램 방식)",
  "short": "값을 구간으로 묶어 분할 기준을 빠르게 찾는 XGBoost 트리 생성 방식. CPU에서 돌아가요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "GPU",
  "aliases": [],
  "full": "Graphics Processing Unit (그래픽 처리 장치)",
  "short": "대량 계산을 병렬로 빠르게 하는 장치. CUDA는 NVIDIA GPU를 쓰기 위한 도구. 이번 수업은 CPU만으로 충분해요.",
  "analogy": "",
  "first": 7,
  "days": [
   7,
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "조기 종료",
  "aliases": [
   "early stopping",
   "early_stopping_rounds",
   "Early Stopping"
  ],
  "full": "Early stopping (조기 종료)",
  "short": "따로 떼어 둔 stop 자료의 손실이 정한 회차(예: 15회) 동안 나아지지 않으면 학습을 멈추는 방법. 잡음까지 배우기 전에 멈춰 과적합을 줄여요.",
  "analogy": "모의고사 점수가 더 안 오르면 벼락치기를 멈추는 것.",
  "first": 7,
  "days": [
   7,
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "교차검증",
  "aliases": [
   "Cross-validation",
   "CV",
   "cross_validate"
  ],
  "full": "Cross-validation (교차검증)",
  "short": "train을 여러 조각(fold)으로 나눠, 돌아가며 한 조각으로 평가하고 나머지로 학습하는 방법. 한 번의 분할 운에 덜 흔들리고, 점수 평균과 흩어짐을 함께 봐요. 전처리도 fold마다 다시 학습해야 해요.",
  "analogy": "모의고사를 문제지 5세트로 돌려 가며 봐서 실력을 평균 내는 것.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "fold",
  "aliases": [
   "5-fold",
   "폴드"
  ],
  "full": "Fold (교차검증 조각)",
  "short": "교차검증에서 나눈 한 조각. 5-fold면 train 60건을 12건씩 5조각으로 나눠 각 회차에 48건 학습, 12건 평가.",
  "analogy": "피자를 5조각으로 나눈 것 중 한 조각.",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": [
   7
  ]
 },
 {
  "term": "StratifiedKFold",
  "aliases": [
   "층화 분할"
  ],
  "full": "Stratified K-Fold (층화 K겹 분할)",
  "short": "모든 fold의 클래스 비율(예: 실패 비율)을 원본과 비슷하게 맞춰 나누는 분할. 소수 클래스가 있는 분류에 써요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "KFold",
  "aliases": [],
  "full": "K-Fold (K겹 분할)",
  "short": "자료를 K조각으로 그냥 나누는 교차검증 분할. 클래스가 없는 회귀(연속 정답)에 써요. shuffle=True면 섞어서 나눠요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "neg_root_mean_squared_error",
  "aliases": [],
  "full": "Negative RMSE (음수 RMSE 점수)",
  "short": "sklearn이 모든 점수를 '클수록 좋다'로 통일하려고 RMSE에 −를 붙인 것. 보고할 땐 다시 −를 붙여 원래 RMSE(ms)로 되돌려요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "그룹 단위 분할",
  "aliases": [
   "GroupKFold",
   "GroupShuffleSplit"
  ],
  "full": "Group split (그룹 단위 분할)",
  "short": "같은 고객·요청 등 한 그룹의 행이 학습과 평가 양쪽에 동시에 들어가지 않게 그룹째로 나누는 방법.",
  "analogy": "",
  "first": 7,
  "days": [
   7,
   14
  ],
  "keyDays": []
 },
 {
  "term": "시간 순서 분할",
  "aliases": [
   "TimeSeriesSplit"
  ],
  "full": "Time-ordered split (시간 순서 분할)",
  "short": "과거 자료로 학습하고 그 뒤의 자료로 평가하도록 시간 순서대로 나누는 방법. 미래 정보가 학습에 들어가지 않게 해요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "신뢰구간",
  "aliases": [
   "95% 신뢰구간"
  ],
  "full": "Confidence interval (신뢰구간)",
  "short": "진짜 값이 들어 있을 것으로 추정하는 통계적 범위. fold 점수의 평균 ± 표준편차는 흩어짐 기록일 뿐 신뢰구간이 아니에요.",
  "analogy": "",
  "first": 7,
  "days": [
   7
  ],
  "keyDays": []
 },
 {
  "term": "조화평균",
  "aliases": [
   "Harmonic mean"
  ],
  "full": "Harmonic mean (조화평균)",
  "short": "2·P·R ÷ (P + R)처럼 계산하는 평균. 두 값 중 하나라도 낮으면 결과도 낮아져서, 한 지표만 극단적으로 올리는 꼼수를 막아요.",
  "analogy": "왕복 평균 속도처럼, 느린 구간이 전체를 크게 끌어내리는 평균.",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "pos_label",
  "aliases": [],
  "full": "pos_label (양성 클래스 지정)",
  "short": "Precision·Recall·F1을 어느 클래스 기준으로 계산할지 정하는 인자. 기본값은 1(=실패).",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "ROC 곡선",
  "aliases": [
   "ROC",
   "roc_curve",
   "ROC curve"
  ],
  "full": "Receiver Operating Characteristic curve (ROC 곡선)",
  "short": "임계값을 1 → 0으로 내리면서 가로축 FPR, 세로축 TPR을 찍어 이은 곡선. 임계값 하나가 아니라 모든 임계값에서의 성능을 한눈에 보여 줘요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "AUC",
  "aliases": [
   "ROC-AUC",
   "roc_auc_score",
   "Area Under the Curve",
   "ROC AUC"
  ],
  "full": "Area Under the ROC Curve (ROC 곡선 아래 넓이)",
  "short": "양성 1건·음성 1건을 뽑았을 때 양성 쪽 점수가 더 높을 확률. 점수의 '순위'만 평가하므로 임계값을 바꿔도 그대로예요. 0.5는 무작위 수준, 1은 완벽. 발견 비율(Recall)이 아니에요.",
  "analogy": "실패 요청과 성공 요청을 한 쌍씩 비교해 '실패 쪽을 더 위험하다고 줄 세웠나'를 채점하는 것.",
  "first": 8,
  "days": [
   8,
   12
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "FPR",
  "aliases": [
   "False Positive Rate"
  ],
  "full": "False Positive Rate (거짓 양성 비율)",
  "short": "실제 음성 중 잘못 양성이라고 한 비율. FP ÷ (FP + TN). ROC 곡선의 가로축.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "TPR",
  "aliases": [
   "True Positive Rate"
  ],
  "full": "True Positive Rate (참 양성 비율)",
  "short": "실제 양성 중 찾아낸 비율. TP ÷ (TP + FN). Recall과 같아요. ROC 곡선의 세로축.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "확률 보정",
  "aliases": [
   "Calibration"
  ],
  "full": "Probability calibration (확률 보정)",
  "short": "모델이 낸 확률이 실제 빈도와 맞는지(0.8짜리들 중 정말 80%가 양성인지) 확인하고 맞추는 것. 확률 0.8이 곧 '80% 확실'은 아니에요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "불균형 자료",
  "aliases": [
   "Imbalanced data",
   "클래스 불균형",
   "불균형",
   "Class imbalance"
  ],
  "full": "Imbalanced data (클래스 불균형 자료)",
  "short": "한 클래스가 다른 클래스보다 훨씬 많은 자료(예: 성공 14 · 실패 6). Accuracy가 속기 쉬워 실제 양성 수, Precision·Recall 관점도 함께 봐요.",
  "analogy": "",
  "first": 8,
  "days": [
   8,
   14
  ],
  "keyDays": [
   14
  ]
 },
 {
  "term": "OOF",
  "aliases": [
   "Out-of-fold"
  ],
  "full": "Out-of-fold prediction (폴드 밖 예측)",
  "short": "교차검증에서 각 행을 그 행을 학습하지 않은 fold 모델로 예측해 모은 것. 모아서 한 번에 계산한 점수는 fold 점수 평균과 다를 수 있어요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "nested CV",
  "aliases": [
   "Nested cross-validation",
   "중첩 교차검증"
  ],
  "full": "Nested cross-validation (중첩 교차검증)",
  "short": "후보를 고르는 교차검증을 바깥 교차검증 안에 한 번 더 넣어, 고른 최고 점수의 낙관성까지 평가하는 방법.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "특성 설계",
  "aliases": [
   "Feature Engineering",
   "특성 공학"
  ],
  "full": "Feature Engineering (특성 설계)",
  "short": "원본 입력을 변환·조합해 모델이 쓰기 좋은 새 특성을 만드는 일(예: log 변환, 비율). 예측 시점에 알 수 있는 값만 써야 하고, 효과는 같은 조건 비교로 확인해요.",
  "analogy": "재료를 그대로 넣지 않고 다지고 섞어 양념을 만드는 것.",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "파생 특성",
  "aliases": [
   "Derived feature"
  ],
  "full": "Derived feature (파생 특성)",
  "short": "기존 열로 계산해 새로 만든 특성. 예: log_input_chars = log(1 + 글자수), chars_per_slot = 글자수 ÷ (1 + 허용 도구수).",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "log1p",
  "aliases": [
   "np.log1p",
   "로그 변환"
  ],
  "full": "log(1 + x) 변환",
  "short": "x에 1을 더한 뒤 로그를 씌우는 함수. 값이 0이어도 안전하고(log(1)=0), 99 → 4.6, 999 → 6.9처럼 큰 값의 차이를 완만하게 줄여요.",
  "analogy": "1, 10, 100, 1000을 '자릿수'로 보면 1, 2, 3, 4로 고르게 보이는 것.",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "FunctionTransformer",
  "aliases": [],
  "full": "FunctionTransformer (함수 변환 단계)",
  "short": "직접 만든 함수(예: add_features)를 Pipeline의 한 단계로 넣어 주는 도구. 학습·예측 때 같은 변환이 자동으로 적용돼요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "지도학습",
  "aliases": [
   "Supervised learning"
  ],
  "full": "Supervised learning (지도학습)",
  "short": "입력과 정답(y)을 함께 주고 정답을 맞히도록 학습하는 방식. 회귀·분류, KNN이 여기에 속해요.",
  "analogy": "정답지가 있는 문제집으로 공부하는 것.",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "비지도학습",
  "aliases": [
   "Unsupervised learning"
  ],
  "full": "Unsupervised learning (비지도학습)",
  "short": "정답(y) 없이 입력만 보고 데이터 속 구조(비슷한 것끼리의 묶음 등)를 찾는 방식. K-Means가 대표.",
  "analogy": "정답지 없이 사진 더미를 비슷한 것끼리 나눠 보는 것.",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "K-Means",
  "aliases": [
   "KMeans",
   "K-평균"
  ],
  "full": "K-Means clustering (K-평균 군집화)",
  "short": "K개 중심을 정해 각 점을 가장 가까운 중심에 할당하고, 중심을 소속 점들의 평균으로 옮기는 과정을 반복해 데이터를 K개 군집으로 묶는 비지도학습. 거리 기반이라 표준화가 필요해요.",
  "analogy": "운동장에 흩어진 학생들을 깃발 K개 근처로 모이게 한 뒤, 깃발을 무리의 한가운데로 옮기기를 반복하는 것.",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "군집",
  "aliases": [
   "클러스터",
   "Cluster",
   "군집화"
  ],
  "full": "Cluster (군집)",
  "short": "서로 가까운(비슷한) 데이터끼리 묶인 그룹. 군집 번호 0·1·2는 순서나 성공/실패 의미가 없는 이름표예요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "군집 중심",
  "aliases": [
   "cluster_centers_",
   "Centroid",
   "중심점"
  ],
  "full": "Cluster center / Centroid (군집 중심)",
  "short": "한 군집에 속한 점들의 평균 위치. 새 입력은 가장 가까운 중심의 군집에 할당돼요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "k-means++",
  "aliases": [],
  "full": "k-means++ (초기 중심 선택법)",
  "short": "처음 K개 중심을 서로 멀리 떨어지게 뽑아 더 나은 결과를 얻기 쉽게 하는 K-Means의 초기화 방법.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "n_clusters",
  "aliases": [],
  "full": "n_clusters (군집 개수 K)",
  "short": "K-Means가 만들 군집의 개수. KNN의 K(이웃 수)와는 뜻이 달라요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "n_init",
  "aliases": [],
  "full": "n_init (초기화 시도 횟수)",
  "short": "시작 중심을 바꿔 K-Means를 몇 번 돌릴지. 그중 inertia가 가장 작은 결과를 써요. 운 나쁜 시작점에 갇히는 걸 줄여요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "inertia",
  "aliases": [
   "inertia_"
  ],
  "full": "Inertia (군집 내 제곱거리 합)",
  "short": "각 점에서 소속 군집 중심까지 거리를 제곱해 모두 더한 값. 작을수록 촘촘하지만 K를 늘리면 거의 항상 줄어서 이것만으로 K를 고르면 안 돼요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "silhouette",
  "aliases": [
   "silhouette_score",
   "실루엣"
  ],
  "full": "Silhouette score (실루엣 점수)",
  "short": "내 군집과는 가깝고 옆 군집과는 먼지를 −1~1로 나타낸 점수. 높을수록 군집이 잘 나뉜 것. 업무적 의미는 따로 해석해야 해요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": [
   8
  ]
 },
 {
  "term": "fit_predict",
  "aliases": [],
  "full": "fit_predict (학습 + 할당 한 번에)",
  "short": "주어진 자료로 군집 중심을 학습하고 바로 군집 번호를 돌려주는 함수. 새 입력에 쓰면 중심을 다시 배워 버리니, 새 입력은 predict로 기존 중심에 할당해요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "PCA",
  "aliases": [
   "Principal Component Analysis",
   "주성분 분석"
  ],
  "full": "Principal Component Analysis (주성분 분석)",
  "short": "여러 특성을 정보 손실이 적게 더 적은 개수의 새 축으로 압축하는 비지도 변환. 자료에서 배우는 변환이라 학습 자료로만 fit해야 해요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "특성 선택",
  "aliases": [
   "Feature selection"
  ],
  "full": "Feature selection (특성 선택)",
  "short": "여러 특성 중 쓸 열만 골라내는 것. 평가 자료를 보고 고르면 누수·선택 편향이 생기니 학습 fold 안에서 해야 해요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "동결",
  "aliases": [
   "evaluation_lock",
   "Freeze",
   "얼리기",
   "freeze"
  ],
  "full": "Freeze / Lock (설정 동결)",
  "short": "test를 열기 전에 모델·특성·임계값·선택 규칙을 파일로 기록해 더 이상 바꾸지 않는 것. test 점수를 보고 바꾸면 test가 선택용 자료가 돼 버려요.",
  "analogy": "시험 전에 답안 작성 규칙을 봉인해 두는 것.",
  "first": 8,
  "days": [
   8,
   11,
   14
  ],
  "keyDays": [
   14
  ]
 },
 {
  "term": "재학습",
  "aliases": [
   "Refit"
  ],
  "full": "Refit (재학습)",
  "short": "선택을 마친 뒤 같은 설정의 새 Pipeline을 train+valid(개발 80건)로 다시 학습하는 것. 그 뒤 test 20건으로 딱 한 번 평가해요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "clone",
  "aliases": [
   "clone()"
  ],
  "full": "sklearn.base.clone (학습 안 된 복사본)",
  "short": "설정은 같지만 학습 상태는 없는 새 모델 객체를 만드는 함수. 이전에 학습한 값(중앙값 등)이 섞이지 않게 해요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "JSON",
  "aliases": [
   "json.dumps"
  ],
  "full": "JavaScript Object Notation",
  "short": "설정이나 결과를 사람과 프로그램이 모두 읽기 쉬운 텍스트로 저장하는 형식. ensure_ascii=False면 한글이 깨지지 않아요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "ValueError",
  "aliases": [],
  "full": "ValueError (값 오류)",
  "short": "입력 값이 약속과 다를 때 파이썬이 내는 오류. 필수 열이 빠진 입력은 예측 전에 이 오류로 분명하게 막아요.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "matplotlib",
  "aliases": [],
  "full": "Matplotlib (그래프 라이브러리)",
  "short": "ROC 곡선, 군집 그림 같은 그래프를 그리는 파이썬 라이브러리.",
  "analogy": "",
  "first": 8,
  "days": [
   8
  ],
  "keyDays": []
 },
 {
  "term": "CPU",
  "aliases": [],
  "full": "Central Processing Unit (중앙 처리 장치)",
  "short": "컴퓨터의 기본 두뇌. 소수의 강한 코어로 복잡한 분기·파일 읽기·전처리 같은 일을 잘해요. 딥러닝에서도 데이터 준비는 CPU 몫이에요.",
  "analogy": "복잡한 일을 빨리 처리하는 소수 정예.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "CUDA",
  "aliases": [
   "CUDA Runtime"
  ],
  "full": "Compute Unified Device Architecture",
  "short": "NVIDIA GPU에서 계산을 시킬 수 있게 해 주는 연산 플랫폼·API. PyTorch가 GPU를 쓰려면 CUDA용으로 빌드된 torch가 필요해요.",
  "analogy": "PyTorch와 GPU 사이의 통역사.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "VRAM",
  "aliases": [
   "GPU 메모리"
  ],
  "full": "Video RAM (GPU 메모리)",
  "short": "GPU에 붙어 있는 전용 메모리. GPU는 VRAM에 올라온 값만 바로 계산할 수 있어서, 모델과 데이터를 여기로 옮겨야 해요.",
  "analogy": "GPU 팀 전용 작업대.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "RAM",
  "aliases": [
   "CPU 메모리"
  ],
  "full": "Random Access Memory (주 메모리)",
  "short": "CPU가 쓰는 컴퓨터의 주 메모리. VRAM과는 서로 다른 공간이라, CPU의 Tensor를 GPU에서 쓰려면 복사(.to)해야 해요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "드라이버",
  "aliases": [
   "GPU Driver",
   "그래픽 드라이버"
  ],
  "full": "GPU Driver (그래픽 드라이버)",
  "short": "운영체제(OS)가 GPU를 쓸 수 있게 연결해 주는 프로그램. 버전이 안 맞으면 GPU가 있어도 PyTorch가 못 써요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "nvidia-smi",
  "aliases": [],
  "full": "NVIDIA System Management Interface",
  "short": "터미널에서 GPU 이름·드라이버 버전·VRAM 사용량·사용률을 보여 주는 명령. 잘 나와도 PyTorch가 GPU를 쓸 수 있다는 보장은 없어요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "PyTorch",
  "aliases": [
   "파이토치"
  ],
  "full": "PyTorch",
  "short": "Tensor 계산, 자동 미분(Autograd), 신경망 층, Optimizer까지 들어 있는 딥러닝 라이브러리. 코드에선 import torch 로 불러요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "병렬 처리",
  "aliases": [
   "병렬 연산",
   "병렬"
  ],
  "full": "Parallel processing",
  "short": "서로 독립인 계산 여러 개를 동시에 처리하는 것. 행렬 곱의 각 칸은 서로 독립이라 GPU가 동시에 계산할 수 있어요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "처리량",
  "aliases": [
   "Throughput",
   "images/s",
   "samples/s"
  ],
  "full": "Throughput (처리량)",
  "short": "정해진 시간에 처리하는 일의 양. GPU는 같은 모양의 독립 계산이 충분히 많을 때 처리량이 커져요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   11
  ],
  "keyDays": []
 },
 {
  "term": "행렬",
  "aliases": [
   "행렬 곱셈",
   "행렬곱"
  ],
  "full": "Matrix (행렬)",
  "short": "숫자를 행과 열로 늘어놓은 2차원 표. 딥러닝의 핵심 계산은 행렬 곱셈(곱하고 더하기)이에요. 0D는 스칼라(숫자 하나), 1D는 벡터(숫자 한 줄)예요.",
  "analogy": "엑셀 표 한 장.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "Tensor",
  "aliases": [
   "텐서"
  ],
  "full": "Tensor (텐서)",
  "short": "차원을 가진 숫자 배열. 딥러닝의 모든 데이터·가중치가 Tensor예요. NumPy 배열과 비슷하지만 GPU에 올릴 수 있고 자동 미분이 돼요.",
  "analogy": "숫자를 담는 여러 칸짜리 상자 — 한 줄이면 벡터, 표면 행렬, 표를 여러 장 겹치면 3D.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "차원",
  "aliases": [],
  "full": "Dimension (차원)",
  "short": "Tensor의 축 개수. 0D 스칼라(숫자 하나), 1D 벡터, 2D 행렬, 3D 이상은 batch·이미지·문장 같은 데이터예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "shape",
  "aliases": [
   "img.shape"
  ],
  "full": "Shape (모양)",
  "short": "Tensor가 축마다 몇 칸인지 적은 것. 예: (batch, feature) = (32, 2)는 샘플 32개 × 특성 2개. 오류 대부분이 shape가 안 맞아서 생겨요.",
  "analogy": "표가 \"몇 행 × 몇 열\"인지.",
  "first": 9,
  "days": [
   9,
   13
  ],
  "keyDays": [
   9,
   13
  ]
 },
 {
  "term": "dtype",
  "aliases": [
   "자료형"
  ],
  "full": "Data type (자료형)",
  "short": "Tensor 안 숫자의 종류. 입력 특성은 보통 실수 torch.float32, 분류 정답(클래스 번호)은 정수 torch.long 이에요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "device",
  "aliases": [],
  "full": "Device (장치)",
  "short": "Tensor·모델이 있는 곳: \"cpu\" 또는 \"cuda\"(GPU). 모델·입력·정답이 모두 같은 device에 있어야 계산돼요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "Broadcasting",
  "aliases": [
   "broadcasting",
   "브로드캐스팅"
  ],
  "full": "Broadcasting",
  "short": "모양이 다른 Tensor끼리 계산할 때 작은 쪽을 규칙에 따라 자동으로 늘려 맞추는 것. 편하지만 의도와 다른 shape도 오류 없이 계산될 수 있어요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "Autograd",
  "aliases": [
   "자동 미분"
  ],
  "full": "Automatic differentiation (자동 미분)",
  "short": "PyTorch가 계산 과정을 기록해 두었다가 backward() 한 번으로 모든 기울기를 자동으로 구해 주는 기능.",
  "analogy": "계산 영수증을 모아 뒀다가 거꾸로 되짚어 정산하는 것.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "계산 그래프",
  "aliases": [],
  "full": "Computational graph",
  "short": "어떤 값끼리 무엇을 곱하고 더했는지 연산 순서를 기록한 연결도. backward()가 이걸 거꾸로 따라가며 기울기를 구해요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "연쇄법칙",
  "aliases": [
   "chain rule",
   "Chain rule"
  ],
  "full": "Chain rule (연쇄법칙)",
  "short": "이어진 연산의 미분은 각 단계 미분을 곱해서 구한다는 규칙. 역전파가 이 규칙으로 기울기를 한 칸씩 곱해 가요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "requires_grad",
  "aliases": [],
  "full": "requires_grad (기울기 추적 여부)",
  "short": "True면 이 Tensor로 하는 연산을 Autograd가 기록해 기울기를 구해요. 학습할 파라미터는 보통 True, 기본값은 False.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "기울기",
  "aliases": [
   "gradient",
   "Gradient",
   "그래디언트"
  ],
  "full": "Gradient (기울기)",
  "short": "파라미터를 조금 늘리면 loss가 얼마나 커지는지 나타내는 값. backward() 후 .grad 에 저장되고, 지우지 않으면 누적돼요.",
  "analogy": "산에서 발밑이 어느 쪽으로 얼마나 기울었는지.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "경사하강법",
  "aliases": [
   "Gradient Descent",
   "경사하강"
  ],
  "full": "Gradient Descent (경사 하강법)",
  "short": "기울기의 반대 방향으로 조금씩 움직여 loss를 줄이는 방법. 파라미터 = 파라미터 − 학습률 × 기울기.",
  "analogy": "안개 낀 산에서 발밑 경사만 보고 한 걸음씩 내려가기.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "발산",
  "aliases": [],
  "full": "Divergence (발산)",
  "short": "학습률이 너무 커서 갱신할 때마다 목표를 지나쳐 loss가 점점 커지는 상태. 심하면 NaN이 돼요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "torch.no_grad()",
  "aliases": [
   "no_grad"
  ],
  "full": "torch.no_grad()",
  "short": "이 블록 안의 연산은 계산 그래프에 기록하지 않아요. 평가·예측·수동 갱신 때 써서 메모리와 시간을 아껴요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "detach()",
  "aliases": [
   "detach"
  ],
  "full": "Tensor.detach()",
  "short": "Tensor를 계산 그래프에서 떼어 낸 것. 값만 꺼내 출력·기록·NumPy 변환할 때 써요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "뉴런",
  "aliases": [
   "인공 뉴런",
   "neuron"
  ],
  "full": "(Artificial) Neuron (인공 뉴런)",
  "short": "입력마다 가중치를 곱해 더하고 편향을 더한 뒤(z = w₁x₁ + w₂x₂ + b) 활성화 함수를 통과시키는 계산 단위.",
  "analogy": "여러 의견에 각자 다른 비중을 줘서 합친 뒤 결정을 내리는 한 사람.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "가중치",
  "aliases": [
   ".pt"
  ],
  "full": "Weight (가중치)",
  "short": "입력에 곱해지는 숫자. 학습이란 결국 이 가중치들을 loss가 줄어드는 쪽으로 조금씩 고치는 일이에요.",
  "analogy": "각 의견을 얼마나 중요하게 볼지 정하는 비중.",
  "first": 9,
  "days": [
   9,
   15
  ],
  "keyDays": []
 },
 {
  "term": "편향",
  "aliases": [
   "bias"
  ],
  "full": "Bias (편향)",
  "short": "가중합에 더하는 상수 b. 출력 위치를 위아래로 옮겨 주는 학습 파라미터예요.",
  "analogy": "저울의 영점 조절.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "파라미터",
  "aliases": [
   "Parameter",
   "모델 파라미터"
  ],
  "full": "Parameter (모델 파라미터)",
  "short": "학습으로 값이 바뀌는 숫자들 — 가중치와 편향. model.parameters()로 모아 Optimizer에 넘겨요. 2→8→2 ANN은 42개예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "층",
  "aliases": [
   "layer",
   "입력층",
   "은닉층"
  ],
  "full": "Layer (층)",
  "short": "뉴런 여러 개를 묶은 계산 단위. 층을 여러 개 쌓을수록 모델이 더 복잡한 표현을 배울 수 있어요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "Linear 층",
  "aliases": [
   "nn.Linear",
   "Linear",
   "선형 변환"
  ],
  "full": "nn.Linear (완전연결 층)",
  "short": "입력의 마지막 차원에 가중치를 곱하고 편향을 더하는 층. nn.Linear(2, 8)은 특성 2개를 8개로 바꿔요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "활성화 함수",
  "aliases": [],
  "full": "Activation function (활성화 함수)",
  "short": "층의 출력에 비선형성을 넣는 함수. 없으면 Linear를 아무리 쌓아도 Linear 하나와 같아요. 대표가 ReLU예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "ReLU",
  "aliases": [],
  "full": "Rectified Linear Unit",
  "short": "음수는 0으로, 양수는 그대로 내보내는 가장 흔한 활성화 함수. 단순하지만 층 사이에 비선형성을 넣어 줘요.",
  "analogy": "0보다 작은 소리는 아예 끄는 스위치.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "비선형성",
  "aliases": [
   "비선형"
  ],
  "full": "Nonlinearity (비선형성)",
  "short": "직선(곱하고 더하기)만으로는 표현 못 하는 굽은 관계. 활성화 함수가 넣어 줘야 신경망이 복잡한 패턴을 배워요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "ANN",
  "aliases": [
   "인공신경망",
   "신경망"
  ],
  "full": "Artificial Neural Network (인공신경망)",
  "short": "인공 뉴런을 묶은 층을 여러 개 쌓은 모델. 오늘은 입력 2 → 은닉 8 → 출력 2 구조를 만들었어요.",
  "analogy": "여러 단계의 심사위원이 차례로 판단을 넘겨주는 심사 과정.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "nn.Sequential",
  "aliases": [
   "Sequential"
  ],
  "full": "torch.nn.Sequential",
  "short": "층을 넣은 순서대로 연결해 모델 하나로 만드는 상자. 넣은 순서가 곧 모델 구조예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "nn.Module",
  "aliases": [
   "Module"
  ],
  "full": "torch.nn.Module",
  "short": "PyTorch의 모든 신경망이 상속하는 기본 클래스. 안에 넣은 층·파라미터를 등록해서 parameters()·to()·eval()을 한 번에 쓸 수 있게 해요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "logit",
  "aliases": [
   "logit",
   "원시 점수",
   "logits",
   "Logit",
   "로짓"
  ],
  "full": "Logits (원시 점수)",
  "short": "모델이 마지막 층에서 낸, softmax 하기 전의 클래스별 점수. 합이 1일 필요가 없고 확률이 아니에요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "softmax",
  "aliases": [
   "Softmax",
   "소프트맥스",
   "softmax"
  ],
  "full": "Softmax",
  "short": "logits를 0~1 사이, 합이 1인 확률로 바꾸는 함수. CrossEntropyLoss가 안에서 해 주니 모델 끝에는 붙이지 않아요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   11
  ],
  "keyDays": []
 },
 {
  "term": "argmax",
  "aliases": [],
  "full": "Argument of the maximum",
  "short": "가장 큰 값이 있는 위치(번호)를 돌려주는 함수. 분류에서는 점수가 가장 큰 클래스 번호 = 예측 클래스예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "손실 함수",
  "aliases": [
   "손실함수",
   "loss",
   "Loss",
   "손실"
  ],
  "full": "Loss function (손실 함수)",
  "short": "모델 예측이 정답에서 얼마나 먼지를 숫자 하나(loss)로 만드는 함수. 학습은 이 숫자를 줄이는 방향으로 가중치를 고쳐요.",
  "analogy": "시험 답안과 정답지를 비교해 매긴 감점 점수.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "CrossEntropyLoss",
  "aliases": [
   "Cross Entropy",
   "교차 엔트로피",
   "CrossEntropy",
   "Cross-Entropy"
  ],
  "full": "Cross-Entropy Loss (교차 엔트로피 손실)",
  "short": "분류용 손실 함수. softmax 전 logits와 클래스 번호(long, shape (batch,))를 받아, 정답 클래스 확률이 낮을수록 loss를 크게 줘요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   11
  ],
  "keyDays": []
 },
 {
  "term": "목적함수",
  "aliases": [],
  "full": "Objective function (목적함수)",
  "short": "학습이 줄이거나(또는 키우려고) 하는 기준 함수. 딥러닝에선 미분할 수 있는 loss를 목적함수로 써요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "순전파",
  "aliases": [
   "forward",
   "Forward"
  ],
  "full": "Forward pass (순전파)",
  "short": "입력을 층마다 차례로 통과시켜 예측(logits)을 만드는 계산. model(X)를 부르면 forward가 실행돼요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "역전파",
  "aliases": [
   "backward",
   "Backward"
  ],
  "full": "Backpropagation (역전파)",
  "short": "loss에서 거꾸로 거슬러 올라가며 모든 파라미터의 기울기를 구하는 계산. PyTorch에선 loss.backward() 한 줄이에요.",
  "analogy": "결과가 틀렸을 때 \"누구 책임이 얼마인지\" 뒤에서부터 따져 올라가기.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "zero_grad",
  "aliases": [
   "zero_grad()"
  ],
  "full": "optimizer.zero_grad()",
  "short": "지난 기울기를 비우는 명령. 기울기는 누적되기 때문에 매 step 처음에 꼭 불러야 이번 batch 기울기만 남아요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "Optimizer",
  "aliases": [
   "optimizer",
   "옵티마이저"
  ],
  "full": "Optimizer (최적화 도구)",
  "short": "구한 기울기를 \"어떤 규칙으로\" 파라미터 갱신에 반영할지 정하는 객체. optimizer.step()이 실제로 값을 바꿔요. SGD·AdamW 등이 있어요.",
  "analogy": "내비게이션이 알려 준 방향(기울기)을 보고 실제로 핸들을 꺾는 운전 방식.",
  "first": 9,
  "days": [
   9,
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "iteration",
  "aliases": [
   "Iteration",
   "step"
  ],
  "full": "Iteration / step (반복 1회)",
  "short": "batch 하나로 forward → loss → backward → optimizer.step()까지 한 번 갱신한 것. epoch당 iteration 수 = ⌈샘플 수 ÷ batch_size⌉.",
  "analogy": "문제집에서 한 페이지 풀고 채점·복습까지 한 번.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "batch",
  "aliases": [
   "Batch",
   "배치"
  ],
  "full": "Batch (배치)",
  "short": "한 번에 모델에 넣어 계산하는 샘플 묶음. 이 묶음마다 기울기를 구해 가중치를 한 번 고쳐요.",
  "analogy": "문제집 전체가 아니라 한 번에 푸는 한 페이지.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "mini-batch",
  "aliases": [
   "미니배치",
   "mini batch"
  ],
  "full": "Mini-batch (미니배치)",
  "short": "전체 데이터를 작게 나눈 batch. 전체를 한 번에 넣는 것과 하나씩 넣는 것 사이에서 속도·메모리·기울기 흔들림의 균형을 맞춰요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "batch size",
  "aliases": [
   "batch_size",
   "Batch size",
   "batch 크기"
  ],
  "full": "Batch size (배치 크기)",
  "short": "batch 하나에 들어가는 샘플 수. 키우면 기울기가 안정되고 step 수가 줄지만 메모리를 더 써요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "epoch",
  "aliases": [
   "Epoch",
   "에폭"
  ],
  "full": "Epoch (에폭)",
  "short": "train 데이터 전체를 한 바퀴 다 본 단위. 200개를 batch 32로 나누면 7번 갱신하면 1 epoch이에요.",
  "analogy": "문제집을 처음부터 끝까지 한 번 푼 것.",
  "first": 9,
  "days": [
   9,
   15
  ],
  "keyDays": [
   9
  ]
 },
 {
  "term": "Dataset",
  "aliases": [
   "TensorDataset"
  ],
  "full": "Dataset / TensorDataset",
  "short": "샘플을 번호로 꺼낼 수 있게 묶은 자료 구조. TensorDataset(X, y)는 X와 y를 같은 번호끼리 짝지어 묶어요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "DataLoader",
  "aliases": [],
  "full": "torch.utils.data.DataLoader",
  "short": "Dataset에서 batch_size만큼씩 꺼내 주는 도구. shuffle로 epoch마다 순서를 섞을 수 있어요.",
  "analogy": "카드 더미를 섞어서 몇 장씩 나눠 주는 딜러.",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "shuffle",
  "aliases": [],
  "full": "Shuffle (섞기)",
  "short": "epoch마다 데이터 순서를 무작위로 섞는 옵션. train은 True(순서 패턴을 외우지 않게), valid/test는 False.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "학습 루프",
  "aliases": [
   "training loop"
  ],
  "full": "Training loop (학습 루프)",
  "short": "batch마다 zero_grad → forward → loss → backward → step 다섯 단계를 반복하는 코드 구조.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "model.train()",
  "aliases": [
   "train()"
  ],
  "full": "Training mode (학습 모드)",
  "short": "Dropout·BatchNorm 같은 층을 학습용 동작으로 바꾸는 스위치. 기울기 계산을 켜고 끄는 것과는 상관없어요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "model.eval()",
  "aliases": [
   "eval()",
   "평가 모드"
  ],
  "full": "Evaluation mode (평가 모드)",
  "short": "Dropout을 끄고 BatchNorm이 저장된 통계를 쓰게 바꾸는 스위치. 기울기 기록은 안 끄니 평가 땐 torch.no_grad()와 함께 써요.",
  "analogy": "",
  "first": 9,
  "days": [
   9
  ],
  "keyDays": []
 },
 {
  "term": "Dropout",
  "aliases": [
   "dropout",
   "드롭아웃"
  ],
  "full": "Dropout (드롭아웃)",
  "short": "학습할 때 뉴런 출력 일부를 확률 p로 무작위로 0으로 끄는 층. 특정 뉴런에만 기대지 못하게 해 과적합을 줄여요. 평가 땐 꺼져요(그대로 통과).",
  "analogy": "매번 팀원 몇 명이 랜덤으로 결석해도 일이 굴러가도록 훈련하기.",
  "first": 9,
  "days": [
   9,
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "BatchNorm",
  "aliases": [
   "BatchNorm1d",
   "Batch Normalization",
   "배치 정규화",
   "BatchNorm2d",
   "BN"
  ],
  "full": "Batch Normalization (배치 정규화)",
  "short": "층의 중간값을 현재 batch의 평균·분산으로 정리(정규화)해 학습을 안정시키는 층. 평가 땐 학습 중 모아 둔 running 통계를 써요.",
  "analogy": "매 시합마다 선수들 기록을 같은 기준으로 환산해 비교하기.",
  "first": 9,
  "days": [
   9,
   10,
   11,
   14
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "CNN",
  "aliases": [
   "Convolutional Neural Network",
   "합성곱 신경망"
  ],
  "full": "Convolutional Neural Network (합성곱 신경망)",
  "short": "이미지처럼 가로·세로 구조가 있는 데이터에 쓰는 신경망. 입력 shape는 보통 (batch, channel, height, width)예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   11
  ],
  "keyDays": [
   11
  ]
 },
 {
  "term": "RNN",
  "aliases": [
   "Recurrent Neural Network",
   "순환 신경망",
   "nn.RNN"
  ],
  "full": "Recurrent Neural Network (순환 신경망)",
  "short": "문장·시계열처럼 순서가 있는 데이터를 앞에서부터 차례로 읽는 신경망.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "Transformer",
  "aliases": [
   "트랜스포머"
  ],
  "full": "Transformer",
  "short": "문장 속 단어들이 서로 얼마나 관련 있는지(attention)를 계산해 처리하는 신경망 구조. 요즘 언어 모델의 기본 구조예요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "임베딩",
  "aliases": [
   "embedding",
   "Embedding",
   "nn.Embedding"
  ],
  "full": "Embedding (임베딩)",
  "short": "단어 같은 기호를 의미를 담은 숫자 벡터로 바꾼 것. 문장은 (batch, sequence, feature) shape가 돼요.",
  "analogy": "",
  "first": 9,
  "days": [
   9,
   12
  ],
  "keyDays": []
 },
 {
  "term": "SGD",
  "aliases": [],
  "full": "Stochastic Gradient Descent (확률적 경사 하강법)",
  "short": "지금 batch의 기울기 반대 방향으로 lr만큼 이동하는 가장 기본 Optimizer. 단순하고 해석하기 쉬워요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "Momentum",
  "aliases": [
   "momentum",
   "모멘텀"
  ],
  "full": "Momentum (모멘텀, 관성)",
  "short": "이전 이동 방향을 함께 쓰는 방법. 같은 방향은 쌓여서 가속하고, 좌우로 튀는 흔들림은 서로 상쇄돼 줄어요. (BatchNorm의 momentum과는 다른 뜻)",
  "analogy": "비탈을 굴러가는 공 — 한 방향으로 굴러갈수록 속도가 붙어요.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "Adam",
  "aliases": [],
  "full": "Adaptive Moment Estimation",
  "short": "기울기의 평균(1차)과 제곱 평균(2차)을 기억해서 파라미터마다 갱신 보폭을 따로 조절하는 Optimizer. 초반 수렴이 빠른 경우가 많아요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "AdamW",
  "aliases": [],
  "full": "Adam with decoupled Weight decay",
  "short": "Adam에서 weight decay를 적응적 갱신과 분리해 적용한 버전. 감쇠를 의도한 만큼 고르게 줘요. torch 기본 weight_decay는 0.01.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "eps",
  "aliases": [
   "epsilon"
  ],
  "full": "Epsilon (아주 작은 값)",
  "short": "0으로 나누는 사고를 막으려고 분모에 더하는 아주 작은 수(예: 1e-8). AdamW·BatchNorm에 있어요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "Scheduler",
  "aliases": [
   "scheduler",
   "lr scheduler",
   "스케줄러"
  ],
  "full": "Learning rate scheduler (학습률 스케줄러)",
  "short": "학습 도중에 Optimizer가 쓸 학습률(lr)을 바꿔 주는 도구. Optimizer는 파라미터를, Scheduler는 lr을 조정해요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "ReduceLROnPlateau",
  "aliases": [],
  "full": "Reduce Learning Rate On Plateau",
  "short": "valid 지표가 patience만큼 좋아지지 않으면 lr에 factor를 곱해 깎는 Scheduler. step(valid_loss)처럼 감시할 값을 꼭 넣어요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "plateau",
  "aliases": [
   "Plateau"
  ],
  "full": "Plateau (고원, 정체 구간)",
  "short": "loss가 더 내려가지 않고 평평하게 머무는 구간. 이때 lr을 낮추면 더 세밀하게 내려갈 수 있어요.",
  "analogy": "산을 내려가다 만난 평평한 고원.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "patience",
  "aliases": [],
  "full": "Patience (참을성)",
  "short": "개선이 없어도 기다려 줄 epoch 수. Scheduler와 Early Stopping에서 써요. 작으면 금방 반응하고, 크면 오래 기다려요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "min_delta",
  "aliases": [],
  "full": "Minimum delta (최소 개선량)",
  "short": "\"개선\"으로 인정할 최소 변화량. 키우면 아주 작은 개선은 개선이 아닌 것으로 쳐서 더 빨리 멈춰요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "학습 데이터",
  "aliases": [
   "train 데이터"
  ],
  "full": "Training set (학습용 데이터, train)",
  "short": "기울기를 계산하고 가중치를 고치는 데 쓰는 자료. 여기서 잘하는 건 \"외웠다\"일 수도 있어서 이것만 보면 안 돼요.",
  "analogy": "직접 풀어 보는 연습 문제집.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "검증 데이터",
  "aliases": [
   "valid",
   "validation",
   "Validation"
  ],
  "full": "Validation set (검증용 데이터, valid)",
  "short": "학습에는 안 쓰고 epoch·구조·규제·설정을 고를 때 보는 자료. 새 데이터에서의 실력을 가늠해요.",
  "analogy": "실전 전에 보는 모의고사.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "테스트 데이터",
  "aliases": [
   "test 데이터"
  ],
  "full": "Test set (시험용 데이터, test)",
  "short": "모든 선택이 끝난 뒤 최종 평가에 딱 한 번 쓰는 자료. test를 보고 설정을 고르면 점수가 실제보다 좋게 나와요.",
  "analogy": "끝까지 봉인해 두는 수능 시험지.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "full-batch",
  "aliases": [
   "full batch"
  ],
  "full": "Full-batch",
  "short": "train 데이터 전체를 batch 하나로 넣는 방식. epoch당 step이 1번뿐이에요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "학습 곡선",
  "aliases": [
   "learning curve"
  ],
  "full": "Learning curve (학습 곡선)",
  "short": "epoch마다 train·valid의 loss·정확도를 기록해 그린 그래프. 과적합·학습률 문제를 진단하는 기본 도구예요.",
  "analogy": "매주 모의고사 점수를 꺾은선 그래프로 그린 것.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "metric",
  "aliases": [
   "평가 지표"
  ],
  "full": "Metric (평가 지표)",
  "short": "모델 성능을 사람이 읽기 좋게 잰 숫자(정확도·F1 등). loss와 다른 결론을 낼 수 있어 따로 기록해요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "노이즈",
  "aliases": [
   "라벨 노이즈"
  ],
  "full": "Noise (노이즈)",
  "short": "데이터에 섞인 우연한 오차나 잘못 붙은 정답. 과적합된 모델은 이런 노이즈까지 외워요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "모델 용량",
  "aliases": [],
  "full": "Model capacity (모델 용량)",
  "short": "모델이 표현·기억할 수 있는 정도. 층·뉴런이 많을수록 크고, 클수록 외울 공간이 많아 과적합이 생기기 쉬워요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "Normalization",
  "aliases": [
   "입력 정규화"
  ],
  "full": "Normalization (정규화)",
  "short": "값의 범위나 분포(예: 평균 0, 분산 1)를 일정하게 맞춰 정리하는 것. 입력 스케일을 맞추거나 BatchNorm이 중간값을 정리할 때 써요. 규제(Regularization)와 다른 말이에요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "activation",
  "aliases": [],
  "full": "Activation (활성값)",
  "short": "층을 지나며 나온 중간 출력값. Dropout은 이 값 일부를 0으로 끄고, BatchNorm은 이 값의 분포를 정리해요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "running 통계",
  "aliases": [
   "running_mean",
   "running_var"
  ],
  "full": "Running statistics (누적 평균·분산)",
  "short": "BatchNorm이 학습 중 batch마다 조금씩 모아 둔 평균(running_mean)·분산(running_var). 평가 땐 이 고정 통계를 써요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "Weight Decay",
  "aliases": [
   "weight decay",
   "weight_decay",
   "Weight decay"
  ],
  "full": "Weight Decay (가중치 감쇠)",
  "short": "가중치가 지나치게 커지지 않게 벌점을 줘서 더 단순한 함수를 고르게 하는 규제. 너무 크면 Underfitting이 생겨요.",
  "analogy": "짐을 많이 들수록 요금이 붙어서 꼭 필요한 짐만 챙기게 되는 것.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": [
   10
  ]
 },
 {
  "term": "파라미터 그룹",
  "aliases": [
   "param_groups"
  ],
  "full": "Parameter groups",
  "short": "Optimizer 안에서 파라미터를 묶음별로 나눠 lr·weight_decay를 다르게 주는 기능. 예: bias·BatchNorm은 감쇠 제외.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "state_dict",
  "aliases": [
   "state_dict()",
   "load_state_dict"
  ],
  "full": "state_dict (모델 상태 묶음)",
  "short": "모델의 모든 가중치(와 BatchNorm 통계)를 이름별로 담은 사전. load_state_dict()로 그 상태로 되돌려요.",
  "analogy": "",
  "first": 10,
  "days": [
   10,
   11
  ],
  "keyDays": []
 },
 {
  "term": "deepcopy",
  "aliases": [
   "copy.deepcopy"
  ],
  "full": "copy.deepcopy (깊은 복사)",
  "short": "그 순간의 값을 통째로 새로 복사하는 파이썬 함수. state_dict를 그냥 담으면 학습하며 같이 바뀌니 best 저장엔 deepcopy를 써요.",
  "analogy": "원본 문서를 가리키는 링크가 아니라 사본을 따로 인쇄해 두는 것.",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "OOM",
  "aliases": [
   "Out of Memory"
  ],
  "full": "Out Of Memory (메모리 부족)",
  "short": "GPU 메모리(VRAM)가 모자라 생기는 오류. batch 크기를 줄이는 게 첫 처방이에요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "gradient accumulation",
  "aliases": [
   "누적 학습"
  ],
  "full": "Gradient accumulation (기울기 누적)",
  "short": "작은 batch 여러 개의 기울기를 일부러 모은 뒤 한 번에 갱신하는 방법. GPU 메모리가 부족할 때 큰 batch처럼 학습하려고 써요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "gradient clipping",
  "aliases": [],
  "full": "Gradient clipping (기울기 자르기)",
  "short": "기울기 크기가 너무 커지면 정해진 크기로 잘라서 갱신이 폭발하지 않게 하는 방법. NaN 대처법 중 하나예요.",
  "analogy": "",
  "first": 10,
  "days": [
   10
  ],
  "keyDays": []
 },
 {
  "term": "합성곱",
  "aliases": [
   "Convolution",
   "Conv2d",
   "Conv 블록"
  ],
  "full": "Convolution (합성곱)",
  "short": "작은 창을 옮겨 가며 창 안의 값과 가중치를 곱해 더하는(가중합) 연산. 모든 위치에서 같은 가중치를 써요.",
  "analogy": "도장을 종이 위에 한 칸씩 옮겨 찍으며, 찍힌 자리마다 '얼마나 잘 맞나' 점수를 적는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": [
   11
  ]
 },
 {
  "term": "Kernel",
  "aliases": [
   "필터",
   "커널",
   "Filter"
  ],
  "full": "Kernel / Filter (커널, 필터)",
  "short": "합성곱 창 안의 가중치 묶음. 3×3이면 숫자 9개이고, 이 숫자는 사람이 아니라 학습으로 정해져요.",
  "analogy": "특정 모양(세로선, 모서리 등)에만 반응하도록 만든 작은 스텐실.",
  "first": 11,
  "days": [
   11,
   13
  ],
  "keyDays": [
   11,
   13
  ]
 },
 {
  "term": "Feature Map",
  "aliases": [
   "특징 맵",
   "특징 지도"
  ],
  "full": "Feature Map (특징 맵)",
  "short": "Kernel 하나가 사진 전체를 훑어서 만든 결과 지도. 그 Kernel이 찾는 패턴이 어디에 얼마나 강하게 있는지를 보여 줘요.",
  "analogy": "'세로선 탐지기'로 사진을 훑은 뒤 반응한 곳만 색칠한 지도.",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": [
   11
  ]
 },
 {
  "term": "채널",
  "aliases": [
   "Channel",
   "in_channels",
   "out_channels"
  ],
  "full": "Channel (채널)",
  "short": "이미지·Feature Map이 겹겹이 쌓인 '장 수'. 컬러 사진은 R·G·B 3채널이고, Conv 층의 출력 채널 수 = Kernel 개수예요.",
  "analogy": "투명 필름 여러 장을 겹쳐 놓은 것 — 한 장이 채널 하나.",
  "first": 11,
  "days": [
   11,
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "NCHW",
  "aliases": [
   "[N, C, H, W]"
  ],
  "full": "N(장수) · C(채널) · H(높이) · W(너비)",
  "short": "PyTorch가 이미지 묶음을 담는 축 순서. 사진 한 장도 [1, C, H, W]처럼 장수 축을 붙여 넣어요. PIL·matplotlib은 [H, W, C] 순서라 바꿔 줘야 해요.",
  "analogy": "",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Stride",
  "aliases": [
   "스트라이드"
  ],
  "full": "Stride (보폭)",
  "short": "창이 한 번에 건너뛰는 칸 수. 1이면 한 칸씩, 2면 두 칸씩 옮겨서 출력이 약 절반 크기가 돼요.",
  "analogy": "징검다리를 한 칸씩 밟느냐, 하나 건너 밟느냐.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Padding",
  "aliases": [
   "패딩"
  ],
  "full": "Padding (채워 넣기)",
  "short": "모자란 자리를 가짜 값(보통 0)으로 채우는 것. CNN에서는 사진 가장자리 바깥에 0을 덧대 크기를 유지하고, 순서 데이터에서는 짧은 순서 끝에 0을 채워 길이를 맞춰요. 채운 값은 진짜 데이터가 아니에요.",
  "analogy": "사진 둘레에 흰 여백을 붙이거나, 짧은 줄 끝에 빈칸을 채워 표 칸을 맞추는 것.",
  "first": 11,
  "days": [
   11,
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "Pooling",
  "aliases": [
   "풀링",
   "MaxPool",
   "MaxPool2d",
   "Max Pooling"
  ],
  "full": "Pooling (풀링, 다운샘플링)",
  "short": "작은 영역을 대표값 하나로 줄이는 것. MaxPool 2×2는 4칸 중 가장 큰 값 하나만 남겨 크기를 절반으로 줄여요. 채널 수는 그대로예요.",
  "analogy": "반 4명 중 최고점 한 명만 대표로 뽑아 보고하는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": [
   11
  ]
 },
 {
  "term": "수용영역",
  "aliases": [
   "Receptive field"
  ],
  "full": "Receptive Field (수용영역)",
  "short": "출력 한 칸이 원본 사진에서 '보고 있는' 범위. 층을 쌓고 Pooling을 거칠수록 넓어져요.",
  "analogy": "높은 층으로 올라갈수록 창밖으로 보이는 동네가 넓어지는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "가중치 공유",
  "aliases": [
   "Weight sharing"
  ],
  "full": "Weight sharing (가중치 공유)",
  "short": "같은 가중치(Kernel)를 모든 위치에서 돌려 쓰는 것. 그래서 파라미터 수가 사진 크기와 상관없어요. 순서 모델도 모든 시점에 같은 가중치를 써요.",
  "analogy": "도장 하나로 종이 전체에 찍는 것 — 위치마다 도장을 새로 파지 않아요.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "등변",
  "aliases": [
   "Equivariance",
   "등변성"
  ],
  "full": "Equivariance (등변성)",
  "short": "입력을 옮기면 결과도 똑같이 옮겨지는 성질. 합성곱이 이래서 물체가 사진 어디에 있어도 같은 검출기로 찾아요.",
  "analogy": "그림을 오른쪽으로 밀면 그림자도 같이 오른쪽으로 가는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "불변",
  "aliases": [
   "Invariance",
   "불변성"
  ],
  "full": "Invariance (불변성)",
  "short": "입력이 조금 바뀌어도 결과가 그대로인 성질. MaxPool은 2×2 안에서 최댓값 위치가 살짝 움직여도 결과가 같아서 작은 이동에 불변이에요.",
  "analogy": "사진 속 고양이가 한 뼘 움직여도 여전히 '고양이'라고 답하는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Global Average Pooling",
  "aliases": [
   "Global Avg Pool",
   "Global Pool",
   "AdaptiveAvgPool2d",
   "GAP"
  ],
  "full": "Global Average Pooling (전역 평균 풀링)",
  "short": "채널마다 Feature Map 전체의 평균 하나만 남겨 [N, C, H, W]를 [N, C, 1, 1]로 줄여요. 사진 크기와 상관없이 Head 입력 개수가 같아져요.",
  "analogy": "지도 전체를 보고 '이 패턴이 평균적으로 얼마나 나왔나' 숫자 하나로 요약하는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Flatten",
  "aliases": [
   "flatten",
   "nn.Flatten"
  ],
  "full": "Flatten (펴기)",
  "short": "여러 축의 Tensor를 한 줄 벡터로 펴는 것. CNN에서는 장수 축 N은 남기고 나머지만 펴서 [N, 32, 1, 1] → [N, 32]로 만들어요.",
  "analogy": "접힌 종이를 쫙 펴서 한 줄로 늘어놓는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Backbone",
  "aliases": [
   "백본",
   "특징 추출부"
  ],
  "full": "Backbone (백본, 특징 추출부)",
  "short": "모델 앞부분. 사진에서 선·모서리·질감·모양 같은 특징을 뽑아내는 부분이에요. 전이학습에서는 사전학습된 Backbone을 그대로 빌려 써요.",
  "analogy": "눈과 시각 피질 — 무엇이 보이는지 정리하는 곳.",
  "first": 11,
  "days": [
   11,
   14,
   15
  ],
  "keyDays": [
   11,
   14
  ]
 },
 {
  "term": "Head",
  "aliases": [
   "분류기"
  ],
  "full": "Head (헤드, 분류기)",
  "short": "모델 끝부분. Backbone이 뽑은 특징을 보고 최종 답(클래스별 점수 등)을 내는 층이에요. 전이학습에서는 내 클래스 수에 맞게 이 부분만 바꿔 끼워요. (Transformer의 Multi-head의 'Head'와는 다른 뜻이에요.)",
  "analogy": "보고 정리한 걸 바탕으로 최종 답을 말하는 입.",
  "first": 11,
  "days": [
   11,
   14,
   15
  ],
  "keyDays": [
   11,
   14
  ]
 },
 {
  "term": "ImageFolder",
  "aliases": [],
  "full": "torchvision.datasets.ImageFolder",
  "short": "폴더 구조로 이미지 데이터셋을 만드는 도구. 하위 폴더 이름이 곧 라벨이고, 이름을 정렬해 0, 1, 2… 번호를 붙여요(class_to_idx로 확인).",
  "analogy": "'cat' 서랍, 'dog' 서랍에 사진을 나눠 넣어 두면 서랍 이름표가 곧 정답이 되는 것.",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "torchvision",
  "aliases": [],
  "full": "torchvision (PyTorch 이미지 패키지)",
  "short": "PyTorch용 이미지 도구 모음. 이미지 폴더 읽기(ImageFolder), 변환·증강(transforms), 사전학습 모델(ResNet18 등)이 들어 있어요. torch와 버전 짝을 맞춰 설치해야 해요.",
  "analogy": "",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Pillow",
  "aliases": [
   "PIL"
  ],
  "full": "Pillow (파이썬 이미지 라이브러리, import 이름은 PIL)",
  "short": "PNG·JPG 같은 이미지 파일을 열고 다루는 파이썬 라이브러리. 설치는 pillow, 코드에서는 PIL로 불러요.",
  "analogy": "",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "ToTensor",
  "aliases": [
   "transforms.ToTensor"
  ],
  "full": "transforms.ToTensor()",
  "short": "PIL 이미지 [H, W, C], 0~255 정수를 Tensor [C, H, W], 0~1 실수로 바꿔 줘요. 이게 없으면 모델이 이미지를 못 받아요.",
  "analogy": "",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Normalize",
  "aliases": [
   "transforms.Normalize"
  ],
  "full": "transforms.Normalize(mean, std)",
  "short": "채널마다 (값 − 평균) ÷ 표준편차로 숫자 범위를 맞춰요. 사전학습 모델을 쓸 땐 그 모델이 배울 때 쓴 평균·표준편차를 그대로 써야 해요.",
  "analogy": "",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "증강",
  "aliases": [
   "Augmentation",
   "augmentation",
   "데이터 증강",
   "mosaic"
  ],
  "full": "Data Augmentation (데이터 증강)",
  "short": "사진을 뒤집고, 살짝 돌리고, 밝기를 바꾸는 등 정답이 안 바뀌는 변형으로 학습 예시를 늘리는 기법. 과적합을 줄여요. train에만 쓰고, 정답을 바꾸는 변형(예: 6↔9 회전)은 금지예요.",
  "analogy": "같은 문제를 숫자만 바꿔 여러 번 풀어 보는 것.",
  "first": 11,
  "days": [
   11,
   13,
   15
  ],
  "keyDays": [
   11
  ]
 },
 {
  "term": "eval 모드",
  "aliases": [
   "train 모드"
  ],
  "full": "model.train() / model.eval()",
  "short": "모델을 학습용·평가용 동작으로 바꾸는 스위치. BatchNorm·Dropout처럼 둘 때 동작이 다른 층이 있어서, 평가·추론 전에는 꼭 model.eval()을 불러요.",
  "analogy": "연습 모드와 실전 모드 전환 버튼.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "inference_mode",
  "aliases": [
   "torch.inference_mode()",
   "torch.inference_mode"
  ],
  "full": "torch.inference_mode()",
  "short": "이 블록 안에서는 기울기 기록을 끄는 설정. 평가·추론에는 역전파가 필요 없으니 메모리와 시간을 아껴요.",
  "analogy": "",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "전이학습",
  "aliases": [
   "Transfer Learning",
   "Transfer learning"
  ],
  "full": "Transfer Learning (전이학습)",
  "short": "큰 데이터로 미리 학습한 모델을 가져와, 마지막 Head만 내 문제에 맞게 바꿔 다시 배우는 방법. 데이터가 적어도 빨리, 덜 과적합되게 배워요.",
  "analogy": "이미 공부한 사람의 뇌를 빌려 와서 새 과목 요점만 더 가르치는 것.",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": [
   11,
   14
  ]
 },
 {
  "term": "사전학습",
  "aliases": [
   "Pretrained",
   "pretrained",
   "사전학습 모델"
  ],
  "full": "Pre-training (사전학습)",
  "short": "다른 큰 데이터(예: ImageNet)로 미리 학습해 두는 것. 이렇게 만든 가중치를 내려받아 출발점으로 써요.",
  "analogy": "기초 교양을 미리 다 배워 둔 상태.",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": [
   14
  ]
 },
 {
  "term": "ImageNet",
  "aliases": [
   "IMAGENET1K_V1"
  ],
  "full": "ImageNet (대규모 이미지 데이터셋)",
  "short": "사진 100만 장 이상, 1000개 클래스로 된 유명한 이미지 데이터셋. 많은 사전학습 모델이 이걸로 배웠어요.",
  "analogy": "",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "ResNet18",
  "aliases": [
   "ResNet",
   "resnet18"
  ],
  "full": "ResNet-18 (Residual Network, 18층)",
  "short": "층 18개짜리 대표적인 CNN 모델. ImageNet 사전학습 가중치를 torchvision에서 바로 받아 전이학습에 써요. 마지막 Head는 model.fc예요.",
  "analogy": "",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "Fine-tuning",
  "aliases": [
   "미세조정",
   "파인튜닝",
   "Fine-tune"
  ],
  "full": "Fine-tuning (미세조정)",
  "short": "사전학습된 층의 일부 또는 전체를 다시 열어 작은 학습률(lr)로 살짝만 고치며 학습하는 것. 보통 뒤쪽 층(layer4)부터 열어요.",
  "analogy": "이미 잘 맞는 기성복을 내 몸에 맞게 살짝 수선하는 것.",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": [
   11,
   14
  ]
 },
 {
  "term": "Head-only",
  "aliases": [],
  "full": "Head-only 전이학습",
  "short": "Backbone은 전부 얼리고 새 Head만 학습하는 방식. ResNet18·2클래스면 학습할 파라미터가 1,026개뿐이라 빠르고 과적합 위험이 작아요.",
  "analogy": "",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "Partial",
  "aliases": [
   "부분 미세조정",
   "Partial FT",
   "Partial Fine-tuning"
  ],
  "full": "Partial fine-tuning (부분 미세조정)",
  "short": "뒤쪽 일부 층(예: layer4)과 Head만 열어 학습하는 방식. 연 층은 작은 lr(1e-4), 새 Head는 큰 lr(1e-3)로 따로 줘요.",
  "analogy": "",
  "first": 11,
  "days": [
   11,
   14
  ],
  "keyDays": []
 },
 {
  "term": "워밍업",
  "aliases": [
   "Warm-up",
   "warm-up"
  ],
  "full": "Warm-up (워밍업)",
  "short": "시간을 재기 전에 몇 번 미리 돌려 두는 것. 첫 실행엔 초기화·메모리 준비 시간이 섞여 유난히 느리기 때문이에요.",
  "analogy": "달리기 기록 재기 전에 몸 푸는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "synchronize",
  "aliases": [
   "torch.cuda.synchronize()",
   "torch.cuda.synchronize"
  ],
  "full": "torch.cuda.synchronize() (GPU 작업 끝날 때까지 기다리기)",
  "short": "GPU는 명령만 받아 두고 파이썬은 바로 다음 줄로 넘어가요(비동기). 이걸 시간 재기 전후에 불러야 GPU 계산이 진짜 끝난 시점까지 잴 수 있어요.",
  "analogy": "주문만 넣고 바로 나가지 말고, 음식이 다 나올 때까지 기다렸다가 시간을 재는 것.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "Peak VRAM",
  "aliases": [
   "Peak Memory",
   "max_memory_allocated"
  ],
  "full": "Peak VRAM (GPU 메모리 최대 사용량)",
  "short": "측정 구간에서 GPU 메모리(VRAM)를 가장 많이 쓴 순간의 양. 빨라도 메모리가 모자라면 못 돌리니 시간과 함께 기록해요. CPU에선 N/A로 적어요.",
  "analogy": "",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "perf_counter",
  "aliases": [
   "time.perf_counter()",
   "time.perf_counter"
  ],
  "full": "time.perf_counter() (정밀 시계)",
  "short": "짧은 구간의 시간을 정밀하게 재는 파이썬 시계(초 단위). 시작·끝 값의 차이가 걸린 시간이에요.",
  "analogy": "스톱워치.",
  "first": 11,
  "days": [
   11
  ],
  "keyDays": []
 },
 {
  "term": "순서 데이터",
  "aliases": [
   "Sequence",
   "시퀀스",
   "순서 모델"
  ],
  "full": "Sequence data (순서 데이터)",
  "short": "값의 순서 자체가 의미를 가지는 데이터. 센서 기록·문장·로그·오디오처럼 순서를 섞으면 뜻이 달라져요.",
  "analogy": "'개가 사람을 물었다'와 '사람이 개를 물었다' — 같은 단어라도 순서가 뜻을 바꿔요.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "시계열",
  "aliases": [
   "Time series"
  ],
  "full": "Time series (시계열)",
  "short": "시간 순서대로 일정 간격으로 쌓인 값. 설비 센서값, 주가, 기온 기록 같은 것이에요. 시간 순서를 유지한 채 train/valid/test를 나눠야 해요.",
  "analogy": "매일 같은 시각에 적은 체온 기록표.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "[B, T, F]",
  "aliases": [
   "BTF",
   "TBF",
   "[T, B, F]"
  ],
  "full": "B(Batch, 개수) · T(Time, 시점 수) · F(Feature, 시점마다 특성 수)",
  "short": "순서 데이터를 담는 Tensor 축 순서. 예: 과거 12개 값으로 다음 값을 맞히는 Window 32개면 [32, 12, 1]. PyTorch 순서 모델 기본값은 [T, B, F]라 batch_first=True로 맞춰요.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "시점",
  "aliases": [
   "Time step",
   "time step"
  ],
  "full": "Time step (시점)",
  "short": "순서 안의 한 칸(한 순간). 순서 모델은 시점마다 같은 계산을 반복해요. 시점의 개수가 길이 T예요.",
  "analogy": "만화책의 한 컷.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "Mask",
  "aliases": [
   "마스크",
   "Padding Mask",
   "key_padding_mask",
   "src_key_padding_mask"
  ],
  "full": "Mask (마스크)",
  "short": "어느 칸을 계산에서 뺄지 True/False로 표시한 표. Padding Mask는 '여기는 채운 가짜 칸'을 표시해 평균·Attention에서 빼게 해요. PyTorch Attention에선 True = 차단이에요.",
  "analogy": "시험지에서 채점하지 않을 칸을 가리는 가림판.",
  "first": 12,
  "days": [
   12,
   13
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "lengths",
  "aliases": [
   "실제 길이"
  ],
  "full": "lengths (샘플별 실제 길이)",
  "short": "Padding을 채우기 전 각 순서의 진짜 길이 목록. 이걸 함께 들고 다녀야 평균·마지막 값을 Padding 없이 계산할 수 있어요.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "pad_sequence",
  "aliases": [
   "pad_sequence()"
  ],
  "full": "torch.nn.utils.rnn.pad_sequence",
  "short": "길이가 다른 Tensor 여러 개를 가장 긴 길이에 맞춰 0으로 채워 한 묶음 Tensor로 만들어 줘요.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "pack_padded_sequence",
  "aliases": [
   "pack_padded_sequence()",
   "PackedSequence",
   "pad_packed_sequence"
  ],
  "full": "torch.nn.utils.rnn.pack_padded_sequence",
  "short": "RNN·LSTM에 '샘플마다 여기까지만 진짜야'를 알려 주는 포장. Padding 칸을 아예 계산하지 않아서 마지막 상태 h_n이 진짜 마지막 시점 값이 돼요. 다시 펼 때는 pad_packed_sequence를 써요.",
  "analogy": "빈칸은 건너뛰고 글자 있는 칸만 읽으라고 표시해 주는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Window",
  "aliases": [
   "Sliding Window",
   "unfold"
  ],
  "full": "Sliding Window (창 자르기)",
  "short": "긴 시계열을 일정 길이 구간으로 잘라 '과거 12개 → 다음 1개' 같은 문제 여러 개로 만드는 것. 창 개수 = 길이 − 창 크기 + 1(한 칸씩 옮길 때).",
  "analogy": "긴 필름을 한 칸씩 밀면서 12컷씩 끊어 보는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Many-to-one",
  "aliases": [],
  "full": "Many-to-one (여러 입력 → 출력 하나)",
  "short": "순서 전체를 읽고 마지막에 클래스나 값 하나를 내는 구조. 예: 문장 감성 분류.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Many-to-many",
  "aliases": [],
  "full": "Many-to-many (여러 입력 → 시점마다 출력)",
  "short": "시점마다 태그·상태·예측을 하나씩 내는 구조. 예: 시점별 상태 표시.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Encoder–Decoder",
  "aliases": [
   "Encoder-Decoder"
  ],
  "full": "Encoder–Decoder (인코더–디코더)",
  "short": "입력 순서를 Encoder가 읽어 요약하고, Decoder가 그 요약으로 길이가 다른 출력 순서를 만드는 구조. 예: 번역.",
  "analogy": "한국어 문장을 듣고 머릿속에 정리한 뒤 영어로 다시 말하는 통역사.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Hidden state",
  "aliases": [
   "hidden state",
   "은닉 상태",
   "h_n",
   "hidden_size"
  ],
  "full": "Hidden state (은닉 상태, h)",
  "short": "지금까지 읽은 입력을 정해진 크기(hidden_size)로 요약한 값. 크기가 고정이라 다 담지 못하고 예측에 필요한 것 위주로 남아요. output은 모든 시점의 h, h_n은 마지막 시점의 h예요.",
  "analogy": "책상 위에 꺼내 둔 메모.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "tanh",
  "aliases": [],
  "full": "Hyperbolic tangent (쌍곡탄젠트 함수)",
  "short": "값을 −1~1 사이로 눌러 주는 활성화 함수. RNN은 같은 계산을 수십 번 반복하니 값이 폭주하지 않게 tanh로 감싸요.",
  "analogy": "아무리 세게 눌러도 일정 높이 이상 안 올라가는 스프링.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "batch_first",
  "aliases": [],
  "full": "batch_first (Batch 축을 맨 앞에 둘지)",
  "short": "True면 입력·출력을 [B, T, F] 순서로, 기본값 False면 [T, B, F]로 다뤄요. 데이터와 설정이 다르면 오류 없이 B와 T가 뒤바뀐 채 계산될 수 있어 위험해요.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "양방향",
  "aliases": [
   "bidirectional",
   "Bidirectional"
  ],
  "full": "Bidirectional (양방향)",
  "short": "순서를 앞에서 뒤로, 뒤에서 앞으로 두 번 읽어 합치는 방식. 출력 크기가 2배(2H)가 돼요. 미래 예측에 쓸 땐 예측 시점 이후 값이 입력에 섞이지 않았는지 확인해요.",
  "analogy": "문장을 앞에서도 읽고 뒤에서도 읽어 뜻을 확인하는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "기울기 소실",
  "aliases": [
   "Vanishing gradient",
   "기울기 폭발",
   "Exploding gradient"
  ],
  "full": "Vanishing / Exploding Gradient (기울기 소실·폭발)",
  "short": "역전파가 긴 시간축을 거꾸로 지나며 같은 가중치를 계속 곱해서, 1보다 작으면 기울기가 0에 가까워져(소실) 먼 과거를 못 배우고, 1보다 크면 폭발하는 문제. 예: 0.9⁵⁰ ≈ 0.005.",
  "analogy": "귓속말 전달 게임 — 사람이 많을수록 처음 말이 희미해져요.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "LSTM",
  "aliases": [
   "Long Short-Term Memory",
   "nn.LSTM"
  ],
  "full": "Long Short-Term Memory (장단기 메모리)",
  "short": "RNN에 오래 들고 가는 기억 통로(Cell state c)와 Gate를 더한 모델. 기억을 곱하지 않고 더해서 이어 가니까 긴 순서도 RNN보다 잘 기억해요. 파라미터는 RNN의 4배예요.",
  "analogy": "메모 한 장(RNN)에 더해, 서랍 속 노트를 따로 두고 무엇을 지우고 적을지 정하는 사람.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "Cell state",
  "aliases": [
   "cell state",
   "셀 상태",
   "c_n"
  ],
  "full": "Cell state (셀 상태, c)",
  "short": "LSTM 안에서만 오래 들고 가는 기억 통로. c_t = f ⊙ c_(t−1) + i ⊙ g처럼 '남길 양 + 새로 쓸 양'을 더해 이어 가요. 밖으로 나가는 건 h예요.",
  "analogy": "서랍 속 노트 — 책상 위 메모(h)와 달리 오래 보관해요.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Gate",
  "aliases": [
   "게이트",
   "Forget gate",
   "Input gate",
   "Output gate",
   "forget gate",
   "input gate",
   "output gate"
  ],
  "full": "Gate (게이트: forget · input · output)",
  "short": "정보를 얼마나 통과시킬지 0~1 비율로 정하는 장치. Forget gate는 이전 기억을 얼마나 남길지, Input gate는 새 정보를 얼마나 쓸지, Output gate는 기억 중 얼마나 꺼낼지 정해요. 비율도 학습으로 정해져요.",
  "analogy": "수도꼭지 — 얼마나 열지(0~1)를 조절해요.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Autoencoder",
  "aliases": [
   "오토인코더",
   "AE"
  ],
  "full": "Autoencoder (오토인코더)",
  "short": "입력을 좁은 통로(Latent)로 줄였다가 다시 원래대로 되살리는 모델. 정답이 미래 값이 아니라 입력 자신이에요. 좁은 통로 때문에 데이터의 주요 패턴을 배워요.",
  "analogy": "긴 글을 세 줄로 요약했다가 다시 풀어 쓰는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "Encoder",
  "aliases": [
   "인코더"
  ],
  "full": "Encoder (인코더)",
  "short": "입력을 읽어 더 작은(또는 문맥이 담긴) 표현으로 바꾸는 앞부분. Autoencoder에서는 x를 Latent z로 압축하고, Transformer Encoder는 문맥이 섞인 표현을 만들어요.",
  "analogy": "긴 글을 요약하는 사람.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Decoder",
  "aliases": [
   "디코더"
  ],
  "full": "Decoder (디코더)",
  "short": "Encoder가 만든 요약 표현을 받아 원하는 출력으로 풀어내는 뒷부분. Autoencoder에서는 z를 다시 x̂(재구성)로 되돌려요.",
  "analogy": "요약본을 보고 원래 글을 다시 써 내는 사람.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Latent",
  "aliases": [
   "Latent z",
   "잠재 표현",
   "Latent space",
   "잠재 공간"
  ],
  "full": "Latent representation (잠재 표현, z)",
  "short": "Encoder가 입력을 압축해 만든 가운데의 작은 표현. 너무 넓으면 그대로 베끼기만 하고, 너무 좁으면 중요한 정보까지 잃어요. 칸마다 무슨 뜻인지는 따로 분석해야 해요.",
  "analogy": "긴 글의 세 줄 요약본.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "재구성 오차",
  "aliases": [
   "Reconstruction error",
   "재구성"
  ],
  "full": "Reconstruction error (재구성 오차)",
  "short": "Autoencoder가 되살린 x̂와 원래 입력 x의 차이(보통 MSE). 작을수록 잘 되살린 거예요. 예측 MSE와는 정답·단위가 달라 같은 표에서 비교하지 않아요.",
  "analogy": "요약본으로 다시 쓴 글이 원문과 얼마나 다른지 세는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "이상 후보",
  "aliases": [
   "이상 탐지",
   "Anomaly detection"
  ],
  "full": "Anomaly candidate (이상 후보)",
  "short": "정상 데이터로만 배운 Autoencoder가 잘 못 되살려 재구성 오차가 기준값보다 큰 샘플. 전처리 실수나 낯선 정상 데이터일 수도 있어 '이상'으로 확정하지 않고 점검 후보로만 표시해요.",
  "analogy": "평소와 다른 소리가 나는 기계에 '점검 필요' 딱지를 붙이는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Denoising",
  "aliases": [
   "Denoising Autoencoder"
  ],
  "full": "Denoising Autoencoder (잡음 제거 오토인코더)",
  "short": "입력에만 노이즈를 섞어 넣고, 정답은 깨끗한 원본으로 두어 학습하는 Autoencoder. 노이즈를 걷어 내고 본래 모습을 되살리는 법을 배워요.",
  "analogy": "얼룩진 사진을 보고 깨끗한 원본을 다시 그려 내는 연습.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Attention",
  "aliases": [
   "어텐션",
   "Attention weight"
  ],
  "full": "Attention (어텐션, 주의 집중)",
  "short": "순서대로 전달하는 대신, 모든 위치끼리의 관련도를 한 번에 계산해 필요한 위치를 비율대로 섞는 방법. weight = softmax(Q·Kᵀ/√d), 출력 = weight·V.",
  "analogy": "회의에서 내 질문과 관련 있는 사람 말에 더 귀 기울이는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": [
   12
  ]
 },
 {
  "term": "Query·Key·Value",
  "aliases": [
   "Query",
   "Q·K·V",
   "QKV"
  ],
  "full": "Query · Key · Value (질의 · 열쇠 · 값)",
  "short": "Attention의 세 재료. Query(찾을 정보)와 Key(비교할 위치)를 비교해 얼마나 볼지 비율을 정하고, 그 비율대로 Value(합산할 내용)를 섞어요.",
  "analogy": "회의에서 내 질문(Query)과 관련 있는 사람(Key)일수록 그 사람 말(Value)에 더 귀 기울이는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Self-Attention",
  "aliases": [
   "셀프 어텐션"
  ],
  "full": "Self-Attention (자기 어텐션)",
  "short": "Query·Key·Value를 모두 같은 입력에서 만들어, 한 순서 안의 위치끼리 서로를 보게 하는 Attention. 코드로는 attn(x, x, x).",
  "analogy": "한 문장 안의 단어들이 서로를 보며 '누가 누구와 관련 있나' 확인하는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Multi-head",
  "aliases": [
   "Multi-head Attention",
   "MultiheadAttention",
   "num_heads",
   "nhead"
  ],
  "full": "Multi-head Attention (다중 헤드 어텐션)",
  "short": "Q·K·V를 여러 갈래(head)로 나눠 Attention을 동시에 여러 번 하는 것. head마다 가까운 위치, 같은 종류 등 다른 관계를 따로 봐요. 차원 E가 head 수로 나누어떨어져야 해요.",
  "analogy": "한 문서를 문법 담당, 내용 담당 등 여러 사람이 각자 관점으로 동시에 읽는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "TransformerEncoderLayer",
  "aliases": [
   "Encoder Layer",
   "TransformerEncoder",
   "Transformer Encoder"
  ],
  "full": "nn.TransformerEncoderLayer / nn.TransformerEncoder",
  "short": "Transformer Encoder 한 층 = Self-Attention + Feedforward, 각각 뒤에 Residual과 LayerNorm. 입력·출력 크기 [B, T, E]가 같아서 TransformerEncoder로 여러 층을 쌓아요.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Position Embedding",
  "aliases": [
   "Positional encoding",
   "Positional Encoding",
   "Position Encoding",
   "위치 정보"
  ],
  "full": "Position Embedding / Encoding (위치 정보)",
  "short": "Attention은 순서를 모르니까, 위치마다 다른 값을 입력에 더해 '몇 번째 자리인지' 알려 주는 것. 학습되는 Embedding이면 입력 길이가 max_len을 넘으면 안 돼요.",
  "analogy": "카드마다 순서 번호 스티커를 붙여 두는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Causal Mask",
  "aliases": [
   "attn_mask"
  ],
  "full": "Causal Mask (인과 마스크)",
  "short": "다음 토큰을 예측할 때 미래 위치를 못 보게 가리는 Mask. 미래를 보면 정답을 미리 보는 셈이라서요.",
  "analogy": "시험 볼 때 다음 장을 넘겨 보지 못하게 하는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "토큰",
  "aliases": [
   "Token"
  ],
  "full": "Token (토큰)",
  "short": "문장 등 순서를 모델이 다루기 위해 자른 한 단위(단어·글자 조각 등). Transformer 입력의 한 위치가 보통 토큰 하나예요.",
  "analogy": "문장을 자른 레고 블록 한 개.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Residual",
  "aliases": [
   "잔차 연결",
   "Residual connection"
  ],
  "full": "Residual connection (잔차 연결)",
  "short": "층의 입력을 변환 결과에 그대로 더해 주는 지름길(입력 + 변환 결과). 기울기가 이 길로 층을 건너 흘러서 여러 층을 쌓아도 학습이 돼요.",
  "analogy": "계단 옆에 놓인 엘리베이터 — 바로 오르내릴 길이 하나 더 있는 것.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "LayerNorm",
  "aliases": [
   "Layer Normalization",
   "layer_norm_eps"
  ],
  "full": "Layer Normalization (층 정규화)",
  "short": "위치 하나 안의 Feature 값들끼리 평균·크기를 맞추는 정규화. 다른 샘플과 묶지 않아서 길이·Padding이 제각각인 순서 데이터에 잘 맞아요(BatchNorm은 여러 샘플을 묶어 정규화).",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "Feedforward",
  "aliases": [
   "FFN",
   "Feed-forward",
   "dim_feedforward"
  ],
  "full": "Feedforward Network (피드포워드 층)",
  "short": "Transformer 층 안에서 위치마다 따로 적용하는 작은 Linear–활성화–Linear 망. Attention이 위치끼리 섞었다면, FFN은 각 위치 안을 변환해요. dim_feedforward는 중간 크기(기본 2048)예요.",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "d_model",
  "aliases": [
   "embed_dim"
  ],
  "full": "d_model / embed_dim (임베딩 차원 E)",
  "short": "Transformer에서 위치 하나를 나타내는 벡터의 길이 E. head 수(nhead)로 나누어떨어져야 해요(예: 16 ÷ 4 = head당 4).",
  "analogy": "",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "O(T²)",
  "aliases": [
   "O(T)"
  ],
  "full": "Big-O 표기 (길이에 따른 계산량)",
  "short": "길이 T가 늘 때 계산량이 어떻게 늘어나는지 나타낸 표기. RNN·LSTM은 O(T)(길이 2배 → 2배), Attention은 모든 위치 쌍을 비교해서 O(T²)(길이 2배 → 4배)예요.",
  "analogy": "반 학생이 2배가 되면 출석 부르기는 2배, 서로 악수하기는 약 4배.",
  "first": 12,
  "days": [
   12
  ],
  "keyDays": []
 },
 {
  "term": "픽셀",
  "aliases": [
   "Pixel"
  ],
  "full": "Pixel (화소)",
  "short": "사진을 이루는 가장 작은 칸 하나. 위치 (y, x)마다 숫자가 들어 있고, 컬러면 칸 하나에 숫자 3개(B, G, R)가 있어요.",
  "analogy": "모자이크 그림의 타일 한 조각.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "BGR",
  "aliases": [],
  "full": "Blue, Green, Red (색 순서)",
  "short": "OpenCV(imread)가 색을 저장하는 순서. 보통 쓰는 RGB와 순서가 반대라서, Matplotlib 등으로 넘길 땐 cvtColor로 RGB로 바꿔야 빨강↔파랑이 안 뒤바뀌어요.",
  "analogy": "같은 주소를 '시·구·동'으로 쓰느냐 '동·구·시'로 쓰느냐의 차이.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "RGB",
  "aliases": [],
  "full": "Red, Green, Blue (색 순서)",
  "short": "빨강·초록·파랑 순서로 색을 적는 방식. Matplotlib·PIL·사전학습 모델은 RGB를 기대해요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "uint8",
  "aliases": [],
  "full": "unsigned 8-bit integer (부호 없는 8비트 정수)",
  "short": "0~255만 담을 수 있는 숫자 자료형. 이미지 기본 형식이라, 그냥 더하면 255를 넘쳐 0 쪽으로 되돌아갈 수 있어요(250+30 → 24).",
  "analogy": "255까지만 있는 계량기 — 넘으면 다시 0부터 돈다.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "Gray",
  "aliases": [
   "Grayscale",
   "회색조"
  ],
  "full": "Grayscale (회색조)",
  "short": "색 3개를 섞어 밝기 하나로 만든 1채널 이미지. 경계·이진화는 밝기만 있으면 되어서 보통 Gray로 바꾼 뒤 처리해요.",
  "analogy": "컬러 사진을 흑백 복사기로 복사한 것.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "색공간",
  "aliases": [
   "Color space"
  ],
  "full": "Color space",
  "short": "색을 숫자로 나타내는 방식(BGR, RGB, Gray, HSV 등). 정답이 있는 게 아니라 목적에 맞게 골라요.",
  "analogy": "같은 장소를 주소로도, 위도·경도로도 적을 수 있는 것.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "HSV",
  "aliases": [],
  "full": "Hue, Saturation, Value (색상·채도·명도)",
  "short": "색을 색 종류(H)·진하기(S)·밝기(V)로 나눠 적는 방식. 조명이 바뀌어도 H는 비교적 그대로라 특정 색을 골라낼 때 좋아요. OpenCV에선 H가 0~179.",
  "analogy": "'무슨 색인지'와 '얼마나 밝은지'를 따로 적은 물감 이름표.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Resize",
  "aliases": [
   "resize"
  ],
  "full": "Resize (크기 바꾸기)",
  "short": "이미지 크기를 바꾸는 것. cv2.resize의 크기는 (W, H) 순서라 shape (H, W)와 반대예요. 비율이 다르면 찌그러져요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "보간법",
  "aliases": [
   "보간",
   "Interpolation"
  ],
  "full": "Interpolation (보간법)",
  "short": "크기를 바꿀 때 새 픽셀 값을 주변 값으로 정하는 방법. 줄일 땐 여러 칸을 요약하고, 키울 땐 없던 칸 값을 추정해요.",
  "analogy": "띄엄띄엄 찍힌 점 사이를 자연스럽게 이어 그리는 것.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "INTER_AREA",
  "aliases": [],
  "full": "Area interpolation (면적 평균 보간)",
  "short": "줄어드는 칸이 덮는 면적의 평균으로 값을 정하는 보간법. 축소할 때 잔무늬·깨짐이 적어 자주 써요. (기본값은 INTER_LINEAR)",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "INTER_NEAREST",
  "aliases": [],
  "full": "Nearest-neighbor interpolation (최근접 보간)",
  "short": "가장 가까운 픽셀 값을 그대로 복사하는 보간법. 중간값이 안 생겨서 0/255 Mask 크기를 바꿀 때 적합하지만, 계단 현상이 생겨요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Blur",
  "aliases": [
   "블러"
  ],
  "full": "Blur (흐리게 하기, Smoothing)",
  "short": "주변 픽셀 값을 섞어 이미지를 부드럽게 흐리는 것. 잡음은 줄지만 세부·경계도 함께 흐려져요.",
  "analogy": "젖은 붓으로 그림을 살살 문질러 번지게 하는 것.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "GaussianBlur",
  "aliases": [
   "Gaussian Blur",
   "가우시안 블러"
  ],
  "full": "Gaussian Blur",
  "short": "가운데 칸에 더 큰 가중치를 준 평균으로 흐리는 필터. 평균 필터보다 자연스럽게 흐려지고, Canny·Otsu 전에 잡음을 줄일 때 많이 써요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "medianBlur",
  "aliases": [
   "Median Blur",
   "Median 필터"
  ],
  "full": "Median Blur (중앙값 필터)",
  "short": "이웃 값을 줄 세운 뒤 가운데 값을 고르는 필터. 튀는 점(소금·후추 잡음)이 아예 뽑히지 않아 점 잡음 제거에 강해요. ksize는 1보다 큰 홀수.",
  "analogy": "반 키를 줄 세워 딱 가운데 학생 키를 대표로 고르는 것 — 엄청 큰 한 명에 휘둘리지 않음.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "bilateralFilter",
  "aliases": [
   "Bilateral",
   "Bilateral Filter"
  ],
  "full": "Bilateral Filter (양방향 필터)",
  "short": "거리뿐 아니라 밝기 차이까지 보고 섞는 필터. 경계 건너편은 거의 안 섞어서 경계를 지키며 흐리지만, 비교적 느려요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "소금·후추 잡음",
  "aliases": [
   "점 잡음",
   "salt-and-pepper"
  ],
  "full": "Salt-and-pepper noise",
  "short": "주변과 동떨어진 흰 점(255)·검은 점(0)이 여기저기 박힌 잡음. 평균 필터로는 흐린 점이 남고, Median 필터로 잘 지워져요.",
  "analogy": "사진 위에 소금과 후추를 뿌려 놓은 모양.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Edge",
  "aliases": [],
  "full": "Edge (경계, 윤곽)",
  "short": "밝기가 갑자기 바뀌는 곳. 물체의 윤곽선이 보통 여기에 있어요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Canny",
  "aliases": [
   "Canny Edge"
  ],
  "full": "Canny Edge Detector",
  "short": "경계만 흰색(255)으로 남기는 방법. 기준값 2개를 써서 강한 경계는 남기고, 약한 경계는 강한 경계에 이어진 것만 남겨요. 잡음 때문에 보통 Blur 후 사용.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "히스토그램",
  "aliases": [
   "Histogram"
  ],
  "full": "Histogram (밝기 분포 막대그래프)",
  "short": "밝기 값(0~255)마다 픽셀이 몇 개인지 센 막대그래프. 물체·배경 밝기가 두 봉우리로 갈리는지 볼 수 있어요.",
  "analogy": "반 학생들 키를 구간별로 세어 그린 막대그래프.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Otsu",
  "aliases": [
   "THRESH_OTSU",
   "오츠"
  ],
  "full": "Otsu's method (오츠 자동 이진화)",
  "short": "밝기 히스토그램을 보고 어두운 무리와 밝은 무리가 가장 잘 갈리는 기준값 T를 자동으로 찾는 이진화. 사진마다 T를 손으로 정하지 않아도 돼요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Adaptive Threshold",
  "aliases": [
   "Adaptive",
   "adaptiveThreshold"
  ],
  "full": "Adaptive Thresholding (적응형 이진화)",
  "short": "칸마다 주변 영역(blockSize)의 평균에서 C를 뺀 값을 그 칸만의 기준으로 쓰는 이진화. 조명이 고르지 않은 사진에 강해요.",
  "analogy": "교실 전체 평균이 아니라 내 주변 자리 평균과 비교해 판정하는 것.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "전경",
  "aliases": [
   "Foreground"
  ],
  "full": "Foreground (전경, 물체 영역)",
  "short": "이진 이미지에서 물체로 본 영역. 관례상 흰색(255)이 전경이에요. 물체가 어두우면 THRESH_BINARY_INV로 뒤집어서 맞춰요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "형태학 연산",
  "aliases": [
   "형태학",
   "Morphology"
  ],
  "full": "Morphological operations (형태학 연산)",
  "short": "이진화 결과의 흰 영역을 Kernel로 깎거나(침식) 부풀려(팽창) 먼지·구멍을 정리하고 모양을 다듬는 처리.",
  "analogy": "찰흙 덩어리 가장자리를 깎거나 덧붙여 모양을 다듬는 것.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "Erosion",
  "aliases": [
   "침식",
   "Erode"
  ],
  "full": "Erosion (침식)",
  "short": "Kernel 안이 모두 흰색일 때만 흰색을 남겨, 흰 영역을 깎아 줄이는 연산. 작은 흰 점은 사라지고 붙은 물체가 떨어져요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Dilation",
  "aliases": [
   "팽창",
   "Dilate"
  ],
  "full": "Dilation (팽창)",
  "short": "Kernel 안에 흰색이 하나라도 있으면 흰색으로 만들어, 흰 영역을 부풀리는 연산. 끊긴 부분이 이어져요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Opening",
  "aliases": [
   "MORPH_OPEN",
   "열기 연산"
  ],
  "full": "Opening (열기: 침식 → 팽창)",
  "short": "먼저 깎고 다시 부풀리는 연산. Kernel보다 작은 흰 먼지는 사라지고, 큰 물체는 원래 크기로 돌아와요. 단, 작은 진짜 결함도 지울 수 있어요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Closing",
  "aliases": [
   "MORPH_CLOSE",
   "닫기 연산"
  ],
  "full": "Closing (닫기: 팽창 → 침식)",
  "short": "먼저 부풀리고 다시 깎는 연산. 물체 안의 작은 검은 구멍이나 끊긴 글자를 메워요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Frame",
  "aliases": [
   "프레임"
  ],
  "full": "Frame (영상의 한 장)",
  "short": "영상에서 꺼낸 사진 한 장. 영상은 시간 순서로 이어진 Frame 묶음이라, 영상 처리 = 같은 처리를 Frame마다 반복하는 것이에요.",
  "analogy": "플립북(움직이는 그림책)의 한 페이지.",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": [
   13
  ]
 },
 {
  "term": "FPS",
  "aliases": [
   "Frames Per Second"
  ],
  "full": "Frames Per Second (초당 프레임 수)",
  "short": "1초에 몇 장의 Frame인지. 영상 파일에선 재생 속도(재생시간 = Frame 수 ÷ FPS), 모델에선 1초에 처리한 장 수를 뜻해요. 클수록 빠름.",
  "analogy": "플립북을 1초에 몇 장 넘기느냐.",
  "first": 13,
  "days": [
   13,
   15
  ],
  "keyDays": []
 },
 {
  "term": "VideoCapture",
  "aliases": [
   "VideoWriter"
  ],
  "full": "cv2.VideoCapture / cv2.VideoWriter",
  "short": "VideoCapture는 영상 파일·카메라에서 Frame을 한 장씩 읽고, VideoWriter는 Frame을 모아 영상 파일로 저장해요. 다 쓰면 release()로 닫아요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Codec",
  "aliases": [
   "코덱",
   "fourcc"
  ],
  "full": "Codec (영상 압축·저장 방식)",
  "short": "영상을 압축해 파일로 저장하는 방식. OpenCV에선 fourcc 코드(예: MJPG, mp4v)로 고르고, 확장자·설치된 코덱과 맞아야 저장돼요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Crop",
  "aliases": [
   "크롭"
  ],
  "full": "Crop (잘라내기)",
  "short": "이미지의 일부만 잘라내는 것. NumPy로 img[y1:y2, x1:x2] — 세로(y)가 먼저예요. 너무 많이 자르면 물체가 잘려 라벨 뜻이 바뀔 수 있어요.",
  "analogy": "",
  "first": 13,
  "days": [
   13
  ],
  "keyDays": []
 },
 {
  "term": "Scratch",
  "aliases": [
   "처음부터 학습"
  ],
  "full": "Training from scratch (처음부터 학습)",
  "short": "사전학습 가중치 없이 무작위 값에서 출발해 전부 학습하는 것. 모서리·질감 보는 법부터 배워야 해서 데이터와 시간이 많이 들어요.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "layer4",
  "aliases": [
   "마지막 Stage"
  ],
  "full": "ResNet18 layer4 (마지막 Stage)",
  "short": "ResNet18 Backbone의 마지막 층 묶음. 원래 데이터에 특화된 특징을 담고 있어서 Fine-tuning할 때 가장 먼저 열어요.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "층별 학습률",
  "aliases": [
   "학습률 차등"
  ],
  "full": "Layer-wise learning rate (param group별 lr)",
  "short": "층마다 학습률을 다르게 주는 것. 이미 잘 배운 Backbone은 작게(예: 1e-4), 무작위에서 시작하는 새 Head는 크게(예: 1e-3) 움직여요.",
  "analogy": "완성된 그림은 가는 붓으로, 빈 캔버스는 큰 붓으로 칠하는 것.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "class_to_idx",
  "aliases": [],
  "full": "class name → index mapping",
  "short": "클래스 이름과 번호의 대응표(예: {'circle': 0, 'square': 1}). train·valid·test에서 똑같아야 채점이 맞아요.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "Class weight",
  "aliases": [
   "class weight",
   "클래스 가중치"
  ],
  "full": "Class weight (클래스별 Loss 가중치)",
  "short": "적은 클래스를 틀렸을 때 Loss에 더 큰 벌점을 주는 방법. 보통 역빈도(전체 ÷ 개수)로 정하고 평균 1로 맞춰요.",
  "analogy": "시험에서 어려운 문제에 배점을 더 주는 것.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "WeightedRandomSampler",
  "aliases": [
   "Sampler",
   "Weighted sampler"
  ],
  "full": "Weighted Random Sampler (가중 무작위 추출)",
  "short": "적은 클래스의 샘플을 더 자주 뽑아 Batch를 만드는 도구. 샘플별 가중치(1/개수)를 주고 복원 추출(replacement=True)로 같은 사진도 여러 번 뽑아요.",
  "analogy": "배점은 그대로 두고, 약한 단원 문제를 더 자주 출제하는 것.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "Macro F1",
  "aliases": [
   "macro avg"
  ],
  "full": "Macro-averaged F1",
  "short": "클래스별 F1을 단순 평균한 값. 클래스마다 한 표씩이라 적은 클래스 하나만 못 맞혀도 점수가 크게 떨어져요. 불균형 문제의 주 지표.",
  "analogy": "반 평균이 아니라 '과목별 점수의 평균' — 한 과목 0점이 그대로 드러남.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": [
   14
  ]
 },
 {
  "term": "Weighted F1",
  "aliases": [
   "weighted avg"
  ],
  "full": "Weighted-averaged F1",
  "short": "클래스별 F1을 support(개수)만큼 가중해 평균한 값. 많은 클래스가 점수를 지배해서 소수 클래스의 실패를 가릴 수 있어요.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "Support",
  "aliases": [
   "support"
  ],
  "full": "Support (표본 수)",
  "short": "실제로 그 클래스인 샘플 수. 그 클래스의 점수가 몇 장을 근거로 나왔는지 알려 줘요.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "오류 분석",
  "aliases": [
   "Error analysis"
  ],
  "full": "Error analysis (오류 분석)",
  "short": "틀린 이미지를 원본·예측 확률·출처와 함께 직접 보고, 원인 후보(데이터 부족·도메인 차이·라벨·모델)를 세워 다음 실험으로 확인하는 과정.",
  "analogy": "오답 노트 정리.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "도메인 차이",
  "aliases": [
   "Domain shift"
  ],
  "full": "Domain shift (도메인 차이)",
  "short": "사전학습 데이터(Source)와 우리 데이터(Target)의 배경·색·잡음·촬영 환경이 달라서 빌려 온 특징이 잘 안 맞는 것.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "Grad-CAM",
  "aliases": [],
  "full": "Gradient-weighted Class Activation Mapping",
  "short": "특정 클래스 점수에 사진의 어느 부분이 기여했는지 열지도로 보여 주는 방법. 원인 확정이 아니라 '모델이 어디에 반응했나'의 단서예요.",
  "analogy": "답안지에 형광펜으로 '여기 보고 풀었어요' 표시한 것.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": [
   14
  ]
 },
 {
  "term": "Heatmap",
  "aliases": [
   "열지도"
  ],
  "full": "Heatmap (열지도)",
  "short": "값이 큰 곳은 뜨거운 색, 작은 곳은 차가운 색으로 칠한 지도. Grad-CAM에선 가장 밝은 곳을 1로 맞춘 값이라 밝기 ≠ 점수예요.",
  "analogy": "일기예보의 기온 지도.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "Hook",
  "aliases": [
   "register_forward_hook",
   "register_hook"
  ],
  "full": "Hook (갈고리 함수)",
  "short": "모델 코드를 고치지 않고 특정 층을 지날 때의 출력이나 gradient를 몰래 복사해 두는 함수. 다 쓰면 꼭 remove()로 지워요.",
  "analogy": "지나가는 택배를 사진만 찍고 그대로 보내는 검문소.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "checkpoint",
  "aliases": [],
  "full": "Checkpoint (저장된 모델 상태)",
  "short": "학습 중 어느 순간의 가중치(state_dict)와 설정을 파일로 저장한 것. 나중에 불러와 이어 쓰거나 best 시점으로 되돌릴 때 써요.",
  "analogy": "게임의 세이브 파일.",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "weights_only",
  "aliases": [
   "weights_only=True"
  ],
  "full": "torch.load(weights_only=True)",
  "short": "torch.load가 텐서 같은 안전한 값만 불러오게 하는 옵션. False면 파일 안의 임의 코드가 실행될 수 있어, 믿을 수 없는 checkpoint는 불러오지 않아요.",
  "analogy": "",
  "first": 14,
  "days": [
   14
  ],
  "keyDays": []
 },
 {
  "term": "객체탐지",
  "aliases": [
   "Object Detection"
  ],
  "full": "Object Detection",
  "short": "사진 속 물체가 무엇인지(클래스)와 어디 있는지(박스)를 함께 찾는 작업.",
  "analogy": "\"고양이 사진이야\"가 아니라 \"고양이는 여기, 강아지는 저기\"라고 손가락으로 짚어 주는 것.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "Bounding Box",
  "aliases": [
   "바운딩 박스",
   "BBox"
  ],
  "full": "Bounding Box (경계 상자)",
  "short": "물체를 딱 감싸는 직사각형. 숫자 4개(좌표)로 위치와 크기를 나타내요.",
  "analogy": "사진 위에 물체 둘레로 그린 네모 테두리.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "xyxy",
  "aliases": [],
  "full": "x1, y1, x2, y2",
  "short": "박스를 왼쪽 위 꼭짓점(x1,y1)과 오른쪽 아래 꼭짓점(x2,y2)으로 적는 방식. 그리기·IoU 계산에 편해요.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "cxcywh",
  "aliases": [
   "xywh"
  ],
  "full": "center x, center y, width, height",
  "short": "박스를 중심점(cx,cy)과 너비·높이(w,h)로 적는 방식. YOLO 라벨이 이 형식이에요.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "정규화 좌표",
  "aliases": [],
  "full": "Normalized coordinates",
  "short": "픽셀 좌표를 사진 폭(W)·높이(H)로 나눠 0~1 사이로 바꾼 값. 사진 크기가 바뀌어도 그대로 쓸 수 있어요.",
  "analogy": "\"왼쪽에서 30px\" 대신 \"왼쪽에서 30% 지점\"이라고 말하는 것.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "GT",
  "aliases": [
   "Ground Truth"
  ],
  "full": "Ground Truth",
  "short": "사람이 표시한 진짜 정답 박스. 모델 예측을 이것과 비교해서 채점해요.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "Confidence",
  "aliases": [
   "conf",
   "신뢰도"
  ],
  "full": "Confidence score (신뢰도 점수)",
  "short": "모델이 그 박스에 대해 \"이게 맞을 거야\"라고 매긴 점수(0~1). 낮은 후보를 걸러낼 때 기준으로 써요.",
  "analogy": "학생이 답 옆에 적은 \"확신 정도\".",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "IoU",
  "aliases": [
   "Intersection over Union"
  ],
  "full": "Intersection over Union (겹침 비율)",
  "short": "두 박스가 얼마나 겹치는지를 0~1로 나타낸 값. 겹친 넓이 ÷ 두 박스를 합친 넓이.",
  "analogy": "두 장의 스티커가 얼마나 정확히 포개졌는지 점수로 매긴 것. 1이면 완전히 같음.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "NMS",
  "aliases": [
   "Non-Maximum Suppression",
   "비최대 억제"
  ],
  "full": "Non-Maximum Suppression (비최대 억제)",
  "short": "한 물체에 겹쳐 나온 박스 여러 개 중 점수가 가장 높은 것만 남기고, 그것과 많이 겹치는 같은 클래스 박스들을 지우는 정리 단계.",
  "analogy": "한 사람을 여러 명이 동시에 \"저기 있다!\"고 가리킬 때, 가장 확신하는 한 명의 손가락만 남기는 것.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "AP",
  "aliases": [
   "Average Precision"
  ],
  "full": "Average Precision",
  "short": "한 클래스에 대해 conf 기준을 바꿔 가며 그린 Precision–Recall 곡선 아래 넓이. 그 클래스를 전반적으로 얼마나 잘 찾는지.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "mAP",
  "aliases": [
   "mean Average Precision",
   "mAP50",
   "mAP50-95"
  ],
  "full": "mean Average Precision",
  "short": "클래스별 AP의 평균. 탐지 모델의 대표 점수. mAP50은 IoU 0.5 기준, mAP50-95는 0.5~0.95 여러 기준의 평균(더 엄격).",
  "analogy": "과목별 점수(AP)의 평균 = 평균 점수(mAP).",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "YOLO",
  "aliases": [
   "You Only Look Once"
  ],
  "full": "You Only Look Once",
  "short": "사진을 한 번만 통과시켜 모든 위치·크기의 물체 후보를 동시에 예측하는 빠른 탐지 모델 계열.",
  "analogy": "사진을 한 번 훑어보고 바로 \"여기, 여기, 여기!\" 하고 짚는 사람.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "Neck",
  "aliases": [],
  "full": "Neck",
  "short": "Backbone이 뽑은 여러 크기의 특징을 섞어서, 작은 물체와 큰 물체를 함께 다룰 수 있게 하는 중간 부분.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "격자",
  "aliases": [
   "Grid"
  ],
  "full": "Grid (격자 칸)",
  "short": "YOLO가 사진을 나누는 칸. 칸마다 박스 후보를 하나씩 내요. 촘촘한 격자는 작은 물체, 성긴 격자는 큰 물체 담당.",
  "analogy": "바둑판처럼 나눠 칸마다 \"여기 뭐 있어?\"라고 묻는 것.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "dataset.yaml",
  "aliases": [],
  "full": "Dataset config file",
  "short": "YOLO에게 학습·검증 사진 폴더 위치와 클래스 이름을 알려 주는 안내서 파일.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "best.pt",
  "aliases": [
   "last.pt"
  ],
  "full": "best / last checkpoint",
  "short": "best.pt는 검증 점수가 가장 좋았던 시점의 모델, last.pt는 마지막 epoch의 모델.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "추론",
  "aliases": [
   "Inference"
  ],
  "full": "Inference (추론)",
  "short": "학습이 끝난 모델에 새 사진을 넣어 결과를 얻는 것.",
  "analogy": "공부를 마치고 실제 시험을 치는 것.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "추적",
  "aliases": [
   "Tracking"
  ],
  "full": "Object Tracking",
  "short": "영상의 프레임 사이에서 같은 물체를 이어 붙여 같은 번호(Track ID)를 유지하는 것.",
  "analogy": "군중 속 한 사람을 계속 눈으로 따라가는 것.",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": [
   15
  ]
 },
 {
  "term": "Track ID",
  "aliases": [
   "track_id"
  ],
  "full": "Track ID",
  "short": "추적할 때 개체마다 붙는 번호. Class ID(종류)와 달리 '몇 번째 그 사람'인지를 나타내요.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "ID switch",
  "aliases": [],
  "full": "ID switch",
  "short": "같은 물체인데 추적 번호가 중간에 바뀌는 실패. 가려졌다 나오거나 비슷한 물체가 엇갈릴 때 생겨요.",
  "analogy": "",
  "first": 15,
  "days": [
   15
  ],
  "keyDays": []
 },
 {
  "term": "OCR",
  "aliases": [
   "광학 문자 인식",
   "Optical Character Recognition"
  ],
  "full": "Optical Character Recognition (광학 문자 인식)",
  "short": "이미지 속 글자를 읽어 텍스트로 바꾸는 전체 과정. 전처리 → 글자 인식(Text·Box·Confidence) → 후처리 → 정보 추출로 이어져요.",
  "analogy": "종이 서류를 보고 컴퓨터에 그대로 받아 적는 타이피스트.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "Tesseract",
  "aliases": [
   "tesseract"
  ],
  "full": "Tesseract OCR 엔진",
  "short": "무료로 쓰는 OCR 엔진 프로그램. pip가 아니라 따로 설치하고, kor·eng 같은 언어 데이터도 함께 깔아야 해요.",
  "analogy": "pytesseract가 통역사라면 Tesseract는 실제로 글을 읽는 사람.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "pytesseract",
  "aliases": [],
  "full": "Python wrapper for Tesseract",
  "short": "Python에서 Tesseract 프로그램을 불러 쓰게 해 주는 연결 고리 패키지. 엔진이 없으면 TesseractNotFoundError가 나요.",
  "analogy": "리모컨 — TV(Tesseract)가 없으면 아무리 눌러도 소용없어요.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "image_to_data",
  "aliases": [
   "image_to_data()",
   "image_to_string"
  ],
  "full": "pytesseract.image_to_data()",
  "short": "OCR 결과를 단어·줄·문단·페이지 단위로 text·conf·left·top·width·height와 함께 돌려주는 함수. image_to_string은 글자만 통째로 줘요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "언어 데이터",
  "aliases": [
   "kor+eng",
   "traineddata"
  ],
  "full": "Tesseract language data",
  "short": "Tesseract가 특정 언어 글자를 읽는 데 쓰는 학습 파일. lang=\"kor+eng\"처럼 설치된 것만 고를 수 있어요.",
  "analogy": "번역가가 들고 다니는 언어별 사전.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "PSM",
  "aliases": [
   "Page Segmentation Mode",
   "--psm"
  ],
  "full": "Page Segmentation Mode (페이지 분할 모드)",
  "short": "\"이 이미지는 페이지/블록 하나/한 줄/한 단어\"라고 Tesseract에 알려 주는 가정. 3=자동, 6=블록 하나, 7=한 줄, 8=한 단어.",
  "analogy": "\"이건 책 한 페이지야\" vs \"이건 메모 한 줄이야\"라고 미리 말해 주기.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "OEM",
  "aliases": [
   "--oem"
  ],
  "full": "OCR Engine Mode",
  "short": "Tesseract가 어떤 인식 엔진을 쓸지 고르는 설정. 보통 --oem 3(사용 가능한 기본)으로 둬요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "원근 보정",
  "aliases": [
   "Perspective 보정",
   "warpPerspective",
   "getPerspectiveTransform"
  ],
  "full": "Perspective transform (원근 변환)",
  "short": "비스듬히 찍혀 사다리꼴이 된 문서를 네 모서리 짝(src→dst)으로 3×3 행렬을 만들어 정면 직사각형으로 펴는 것.",
  "analogy": "구겨진 듯 기울어진 사진 속 종이를 손으로 잡아 반듯하게 펴기.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "CER",
  "aliases": [
   "Character Error Rate",
   "문자 오류율"
  ],
  "full": "Character Error Rate (문자 오류율)",
  "short": "(치환+삭제+삽입) ÷ 정답 글자 수. OCR 결과를 정답으로 고치려면 글자를 몇 번 고쳐야 하는지의 비율. 낮을수록 좋아요.",
  "analogy": "받아쓰기 시험에서 틀린 글자 수 ÷ 전체 글자 수.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "WER",
  "aliases": [
   "Word Error Rate"
  ],
  "full": "Word Error Rate (단어 오류율)",
  "short": "CER과 같은 계산을 글자 대신 단어 단위로 한 오류율.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "편집 거리",
  "aliases": [
   "Levenshtein distance",
   "Levenshtein",
   "levenshtein"
  ],
  "full": "Levenshtein distance (편집 거리)",
  "short": "한 문자열을 다른 문자열로 바꾸는 데 필요한 최소 치환(S)·삭제(D)·삽입(I) 횟수.",
  "analogy": "오타 난 문장을 정답으로 고칠 때 눌러야 하는 최소 키 횟수.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "micro CER",
  "aliases": [],
  "full": "micro-averaged CER",
  "short": "문서별 CER을 평균 내지 않고, 전체 오류 수 ÷ 전체 정답 글자 수로 계산한 CER. 글자 하나하나가 같은 무게예요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "정규식",
  "aliases": [
   "정규표현식",
   "Regular expression",
   "re.search"
  ],
  "full": "Regular expression (정규 표현식)",
  "short": "\"20으로 시작하는 숫자 4개-숫자-숫자\"처럼 글자의 모양 규칙을 적어 문자열에서 후보를 찾는 문법. Python은 re 모듈.",
  "analogy": "\"010-XXXX-XXXX 모양을 다 찾아 줘\"라는 검색 주문서.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "raw string",
  "aliases": [],
  "full": "raw string (r\"...\")",
  "short": "앞에 r을 붙인 문자열. 역슬래시(\\)를 Python이 먼저 해석하지 않아 정규식 패턴을 그대로 쓸 수 있어요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Key-value",
  "aliases": [
   "키-값",
   "Key-value 추출"
  ],
  "full": "Key-value extraction",
  "short": "\"발행일\" 같은 라벨(key)과 그 옆의 값(value)을 짝지어 뽑는 것. 근거 위치(bbox)까지 함께 묶어요.",
  "analogy": "서류 양식의 칸 이름과 적힌 내용을 짝지어 옮겨 적기.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "정규화 값",
  "aliases": [],
  "full": "normalized value",
  "short": "OCR 원문(raw) \"1,234,500원\"을 프로그램이 쓰기 좋게 정리한 값(1234500). raw와 함께 저장해요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Schema",
  "aliases": [
   "스키마",
   "schema version"
  ],
  "full": "Schema (데이터 모양 규칙)",
  "short": "JSON에 어떤 key가 반드시 있고 각 값이 어떤 자료형이어야 하는지 정한 규칙. 바뀌면 버전을 올려 기록해요.",
  "analogy": "서류 양식 — 어떤 칸이 필수이고 무엇을 적어야 하는지.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "ISO 8601",
  "aliases": [],
  "full": "ISO 8601 날짜 형식",
  "short": "날짜를 2026-11-09처럼 연-월-일 순서로 적는 국제 표준 형식.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "hOCR",
  "aliases": [
   "ALTO XML"
  ],
  "full": "hOCR / ALTO XML",
  "short": "OCR 결과를 페이지·문단·줄·단어 구조와 위치까지 담아 저장하는 표준 파일 형식들.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "읽기 순서",
  "aliases": [
   "Reading order"
  ],
  "full": "Reading order",
  "short": "사람이 문서를 읽는 순서(제목 → 본문 → 표 → 합계). 다단·표에서는 OCR이 이 순서를 따로 정리해야 해요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "표 구조 인식",
  "aliases": [
   "Table structure"
  ],
  "full": "Table structure recognition",
  "short": "글자를 읽는 것과 별개로, 각 글자가 어느 행·열·셀에 속하는지 복원하는 문제.",
  "analogy": "엑셀 칸을 다시 그리는 일.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "VLM",
  "aliases": [
   "Vision-Language Model",
   "비전-언어 모델"
  ],
  "full": "Vision-Language Model",
  "short": "이미지와 글(질문)을 함께 입력받아 글로 답하는 모델. Vision Encoder + 연결 모듈 + 언어 모델 구조가 대표적이에요.",
  "analogy": "사진을 보면서 질문에 대답해 주는 사람.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "멀티모달",
  "aliases": [
   "Multimodal"
  ],
  "full": "Multimodal",
  "short": "이미지·텍스트·소리처럼 종류가 다른 입력을 함께 다루는 것.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Processor",
  "aliases": [
   "AutoProcessor"
  ],
  "full": "Hugging Face Processor",
  "short": "VLM 입력 준비 담당. 이미지 전처리(pixel_values)와 Chat template·토큰화(input_ids)를 한 번에 해요.",
  "analogy": "재료(사진·질문)를 모델이 먹기 좋게 손질하는 주방 보조.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Chat template",
  "aliases": [
   "apply_chat_template",
   "chat template"
  ],
  "full": "Chat template",
  "short": "role·content 대화 목록을 그 모델이 학습한 대화 형식(특수 토큰 포함)으로 바꾸는 틀.",
  "analogy": "편지 양식 — 받는 사람마다 정해진 머리말·맺음말이 달라요.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Projector",
  "aliases": [
   "연결 모듈"
  ],
  "full": "Projector (시각 특징 → 언어 공간)",
  "short": "Vision Encoder가 뽑은 이미지 특징을 언어 모델이 알아듣는 토큰 모양으로 옮겨 주는 연결 부분.",
  "analogy": "그림을 말로 통역해 주는 통역사.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "LLM",
  "aliases": [
   "대규모 언어 모델",
   "언어 모델"
  ],
  "full": "Large Language Model",
  "short": "글 토큰을 보고 다음 토큰을 하나씩 만들어 내는 큰 언어 모델. VLM의 답 생성 부분이에요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "do_sample",
  "aliases": [
   "temperature"
  ],
  "full": "do_sample / temperature",
  "short": "False면 매번 가장 그럴듯한 토큰을 골라 같은 입력에 같은 답(재현성). True면 확률대로 뽑고 temperature가 뽑기의 무작위 정도를 정해요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "KV cache",
  "aliases": [
   "Key/Value cache",
   "use_cache"
  ],
  "full": "Key/Value cache",
  "short": "답을 한 토큰씩 만들 때 앞에서 계산한 Key·Value를 저장해 재사용하는 것. 빨라지지만 메모리를 차지해요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Hallucination",
  "aliases": [
   "환각"
  ],
  "full": "Hallucination (환각)",
  "short": "모델이 이미지에 없는 값·표·근거를 자연스럽게 지어내는 것.",
  "analogy": "모르는 문제에 그럴듯하게 꾸며 낸 답을 쓰는 학생.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "Grounding",
  "aliases": [
   "그라운딩",
   "evidence"
  ],
  "full": "Grounding (근거 연결)",
  "short": "답과 함께 원문 문자열·페이지·Box·Crop 같은 근거를 제시해 실제 문서와 대조할 수 있게 하는 것.",
  "analogy": "주장마다 \"교과서 몇 쪽 몇째 줄\"을 붙이기.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "Fallback",
  "aliases": [],
  "full": "Fallback (대체 처리)",
  "short": "근거가 부족할 때 억지로 답하지 않고 null·\"확인할 수 없음\"과 검토 필요 상태로 돌려주는 규칙.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Hybrid",
  "aliases": [
   "하이브리드"
  ],
  "full": "OCR + VLM Hybrid",
  "short": "OCR의 글자·위치 후보와 VLM의 문맥 판단을 역할 분리해 함께 쓰고 서로의 실패를 검증하는 방식.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "패치",
  "aliases": [
   "Patch"
  ],
  "full": "Image patch",
  "short": "VLM이 이미지를 잘게 나눈 작은 조각. 해상도가 2배면 조각은 4배라 계산·메모리도 커져요.",
  "analogy": "사진을 바둑판처럼 자른 퍼즐 조각.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "ONNX",
  "aliases": [
   "Open Neural Network Exchange"
  ],
  "full": "Open Neural Network Exchange",
  "short": "여러 프레임워크가 함께 읽는 모델 계산 그래프 형식. 입력·출력 이름·shape·연산자를 명시해요.",
  "analogy": "어느 나라에서나 쓰는 공용 설계도 양식.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "TensorRT",
  "aliases": [],
  "full": "NVIDIA TensorRT",
  "short": "NVIDIA GPU용 추론 최적화 도구. 지원 모델을 그 GPU에 맞춘 Engine으로 빌드해 빠르게 실행해요.",
  "analogy": "공용 설계도를 이 공장 기계에 딱 맞게 고친 생산 라인.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "Engine",
  "aliases": [
   "TensorRT Engine",
   ".engine"
  ],
  "full": "TensorRT Engine",
  "short": "TensorRT가 빌드한 실행용 파일. 만든 GPU·TensorRT 버전·OS 조건에 묶이고, 실행 가능한 산출물이라 믿을 수 있는 출처만 불러와요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "trtexec",
  "aliases": [],
  "full": "trtexec (TensorRT 명령줄 도구)",
  "short": "ONNX로 Engine을 빌드(--onnx, --saveEngine)하거나 저장한 Engine을 불러와 추론을 확인(--loadEngine)하는 명령.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Build Phase",
  "aliases": [
   "Runtime Phase",
   "ExecutionContext"
  ],
  "full": "TensorRT Build / Runtime Phase",
  "short": "Build는 커널 후보를 재 보고 연산을 합쳐 Engine을 만드는 단계, Runtime은 Engine을 불러와(ExecutionContext) 반복 추론하는 단계.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "동적 shape",
  "aliases": [
   "Dynamic shape",
   "optShapes"
  ],
  "full": "Dynamic shape",
  "short": "입력 크기(배치 등)가 실행마다 달라질 수 있게 min·opt·max 범위로 빌드하는 것. 범위 밖 입력은 거부돼요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "FP16",
  "aliases": [
   "FP32",
   "INT8"
  ],
  "full": "Floating Point 16-bit (반정밀도)",
  "short": "숫자 하나를 16비트로 저장하는 형식. FP32보다 메모리 절반·보통 더 빠르지만 정확도를 다시 확인해야 해요. INT8은 8비트 정수.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Jetson",
  "aliases": [],
  "full": "NVIDIA Jetson",
  "short": "카메라 옆 현장에 설치하는 NVIDIA GPU가 달린 작은 컴퓨터(엣지 장치).",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "JetPack",
  "aliases": [],
  "full": "NVIDIA JetPack SDK",
  "short": "Jetson용 소프트웨어 묶음. CUDA·TensorRT·DeepStream 버전 조합이 정해져 있어요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "DeepStream",
  "aliases": [
   "nvinfer",
   "gst-nvinfer",
   "nvstreammux"
  ],
  "full": "NVIDIA DeepStream SDK",
  "short": "영상 입력 → Decode → Batch(nvstreammux) → 추론(nvinfer, TensorRT) → 추적 → 화면/저장을 플러그인으로 잇는 영상 분석 파이프라인.",
  "analogy": "공장 컨베이어 벨트처럼 프레임이 단계별로 흘러가는 라인.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "Throttling",
  "aliases": [],
  "full": "Thermal throttling",
  "short": "장치가 뜨거워지면 스스로 속도를 낮추는 현상. 그래서 엣지에선 온도·전력도 함께 기록해요.",
  "analogy": "",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "jiwer",
  "aliases": [
   "jiwer.cer"
  ],
  "full": "jiwer (CER·WER 계산 라이브러리)",
  "short": "정답 문자열과 OCR·음성인식 결과를 넣으면 CER(글자 오류율)·WER(단어 오류율)을 계산해 주는 Python 패키지. 순서는 (정답, 예측).",
  "analogy": "빨간 펜으로 틀린 글자 수를 세어 주는 채점기.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "필드 정확도",
  "aliases": [
   "Field accuracy"
  ],
  "full": "Field accuracy (필드 정확도)",
  "short": "합계·날짜·공급자 같은 업무 필드 중 값이 정확히 맞은 비율. 글자 하나만 틀려도 그 필드는 통째로 틀린 걸로 세요.",
  "analogy": "받아쓰기 점수(CER)가 아니라 '시험 답안 칸이 맞았나'를 세는 것.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": [
   16
  ]
 },
 {
  "term": "보류",
  "aliases": [],
  "full": "보류 (review · 사람 확인)",
  "short": "근거와 맞지 않거나 원문에 없는 값을 자동으로 확정하지 않고 null로 두어 사람이 확인하게 하는 상태. 정답으로 세지 않아요.",
  "analogy": "애매한 판정은 비디오 판독으로 넘기기.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "greedy decoding",
  "aliases": [
   "Greedy"
  ],
  "full": "Greedy decoding (탐욕적 생성)",
  "short": "다음 토큰을 매번 가장 확률 높은 것 하나로 고르는 생성 방식. do_sample=False가 이것이라 같은 입력에 같은 답이 나와요.",
  "analogy": "갈림길마다 지금 가장 넓은 길만 고르는 것.",
  "first": 16,
  "days": [
   16
  ],
  "keyDays": []
 },
 {
  "term": "FastAPI",
  "aliases": [
   "fastapi"
  ],
  "full": "FastAPI (Python 웹 API 프레임워크)",
  "short": "'이 주소로 이런 요청이 오면 이 함수를 실행해'를 연결해 주는 Python 도구. 입력 검증과 자동 문서(/docs)도 만들어 줘요.",
  "analogy": "주문서를 보고 어느 요리사에게 넘길지 정하는 주방장.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "Uvicorn",
  "aliases": [
   "uvicorn"
  ],
  "full": "Uvicorn (ASGI 서버)",
  "short": "네트워크에서 요청을 실제로 받아 FastAPI 앱에 넘겨주는 서버 프로그램. python -m uvicorn main:app 으로 실행해요.",
  "analogy": "손님을 맞아 주문을 주방에 전달하는 홀 직원.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "API",
  "aliases": [
   "웹 API"
  ],
  "full": "Application Programming Interface",
  "short": "프로그램끼리 주고받는 약속된 창구. 어떤 주소에 어떤 형식으로 요청하면 어떤 형식으로 응답하는지 정해 둔 것.",
  "analogy": "식당의 메뉴판 + 주문 창구.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "HTTP",
  "aliases": [
   "HTTP 요청",
   "HTTP 응답"
  ],
  "full": "HyperText Transfer Protocol",
  "short": "브라우저·프로그램이 서버와 대화하는 규칙. 항상 요청 1개와 응답 1개가 한 쌍이에요.",
  "analogy": "편지를 보내고 답장을 받는 정해진 양식.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "CRUD",
  "aliases": [],
  "full": "Create · Read · Update · Delete",
  "short": "데이터를 만들고(POST)·읽고(GET)·고치고(PUT)·지우는(DELETE) 네 가지 기본 동작.",
  "analogy": "공책에 쓰기·읽기·고치기·지우기.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "HTTP Method",
  "aliases": [
   "Method",
   "HTTP 메서드"
  ],
  "full": "HTTP Method (GET · POST · PUT · DELETE · PATCH)",
  "short": "요청이 무엇을 하려는지 나타내는 동사. 같은 주소라도 Method가 다르면 다른 함수가 실행돼요.",
  "analogy": "같은 창구에서 '보여 주세요/새로 넣어 주세요/바꿔 주세요/빼 주세요'.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Route",
  "aliases": [
   "라우트",
   "경로 등록"
  ],
  "full": "Route (Method + Path → 함수 연결)",
  "short": "'GET + /api/knowledge/{id}' 같은 Method와 경로 조합을 처리 함수에 연결해 둔 것. @app.get(...)으로 등록해요.",
  "analogy": "'2층 민원실 3번 창구는 등본 담당' 같은 안내판.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Path Parameter",
  "aliases": [
   "Path 파라미터",
   "경로 매개변수"
  ],
  "full": "Path Parameter (경로 값)",
  "short": "URL 경로 안에 들어가는 값. /api/knowledge/2 의 2처럼 대상 하나를 콕 집을 때 써요. 함수 인자 이름이 {중괄호} 이름과 같아야 해요.",
  "analogy": "'302호'처럼 문패로 집을 찾는 것.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Query Parameter",
  "aliases": [
   "Query 파라미터",
   "쿼리 파라미터",
   "쿼리 문자열"
  ],
  "full": "Query Parameter (? 뒤의 조건)",
  "short": "URL 끝 ?category=api 처럼 붙이는 선택 조건. 주로 목록을 거를 때 쓰고, 기본값이 있으면 생략할 수 있어요.",
  "analogy": "'빨간색만 보여 주세요' 같은 검색 필터.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Request Body",
  "aliases": [
   "요청 본문",
   "Body"
  ],
  "full": "Request Body (요청 본문)",
  "short": "POST·PUT에서 새로 만들거나 고칠 데이터를 JSON으로 담아 보내는 부분. GET에는 보통 없어요.",
  "analogy": "택배 상자 안에 든 물건.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Content-Type",
  "aliases": [
   "application/json"
  ],
  "full": "Content-Type 헤더",
  "short": "본문이 어떤 형식인지 알려 주는 Header. JSON이면 application/json, HTML 화면이면 text/html.",
  "analogy": "택배 상자에 붙은 '깨지기 쉬움·유리' 같은 내용물 표시.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "상태 코드",
  "aliases": [
   "Status Code",
   "HTTP 상태 코드"
  ],
  "full": "HTTP Status Code",
  "short": "처리 결과를 세 자리 숫자로 알려 줘요. 2xx 성공, 4xx 요청 쪽 문제(주소·입력), 5xx 서버 쪽 문제.",
  "analogy": "택배 조회의 '배송 완료 / 주소 불명 / 물류센터 오류'.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "201 Created",
  "aliases": [],
  "full": "201 Created",
  "short": "새 자원을 만들었다는 성공 코드. POST로 지식을 등록했을 때 써요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "204 No Content",
  "aliases": [],
  "full": "204 No Content",
  "short": "성공했지만 돌려줄 본문이 없다는 코드. 삭제 성공에 쓰고 JSON을 같이 보내지 않아요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "404 Not Found",
  "aliases": [],
  "full": "404 Not Found",
  "short": "그런 주소(Route)가 없거나, 주소는 맞는데 그 ID의 자원이 없다는 코드. detail 문구로 둘을 구분해요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "405 Method Not Allowed",
  "aliases": [],
  "full": "405 Method Not Allowed",
  "short": "경로는 있는데 그 Method는 지원하지 않는다는 코드. 예: PUT만 만든 경로에 PATCH를 보냄.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "422 Unprocessable Content",
  "aliases": [
   "Unprocessable Entity"
  ],
  "full": "422 Unprocessable Content",
  "short": "요청의 자료형·필수값·길이 규칙이 틀렸다는 코드. FastAPI가 함수 실행 전에 자동으로 내요. detail의 loc·msg로 위치를 확인해요.",
  "analogy": "서류 접수창구에서 '빈칸이 있어서 접수 불가'.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "HTTPException",
  "aliases": [],
  "full": "fastapi.HTTPException",
  "short": "raise 하면 함수를 즉시 멈추고 지정한 상태 코드와 detail을 응답으로 보내요. 오류를 return으로 보내면 200이 되니 꼭 raise로.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Pydantic",
  "aliases": [
   "pydantic",
   "BaseModel"
  ],
  "full": "Pydantic (데이터 검증 라이브러리)",
  "short": "class로 데이터 모양(필드·타입·길이)을 정해 두면 들어온 값을 검사해 줘요. FastAPI는 이걸로 Body를 검사하고 틀리면 422를 내요.",
  "analogy": "서류 양식의 '필수 기재란·글자 수 제한'.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "Field",
  "aliases": [
   "Field()"
  ],
  "full": "pydantic.Field",
  "short": "필드에 min_length·max_length 같은 규칙을 붙여요. 기본값을 주지 않으면 필수 필드예요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "model_dump",
  "aliases": [
   "model_dump()"
  ],
  "full": "BaseModel.model_dump()",
  "short": "검증된 Pydantic 모델을 파이썬 dict로 바꿔 줘요. exclude_unset=True면 실제로 보낸 필드만 꺼내요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "response_model",
  "aliases": [],
  "full": "response_model= (응답 모델)",
  "short": "응답의 모양을 검사·정리하고 /docs에 표시해요. 정의에 없는 필드(next_id 등)는 걸러져요. 사실성이나 권한은 보증하지 않아요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Swagger UI",
  "aliases": [
   "/docs",
   "자동 문서"
  ],
  "full": "Swagger UI (FastAPI 자동 API 문서)",
  "short": "/docs 주소에 자동으로 생기는 API 문서 화면. Try it out → Execute로 POST·PUT·DELETE도 직접 보낼 수 있어요.",
  "analogy": "메뉴판인데 그 자리에서 주문까지 되는 키오스크.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "데코레이터",
  "aliases": [
   "decorator"
  ],
  "full": "Decorator (@)",
  "short": "함수 위에 @이름 을 붙여 그 함수에 기능이나 등록 정보를 덧씌우는 문법. @app.get(...)은 '이 함수를 이 주소에 등록'.",
  "analogy": "물건은 그대로 두고 꼬리표만 붙이는 것.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "localhost",
  "aliases": [
   "127.0.0.1"
  ],
  "full": "localhost (127.0.0.1)",
  "short": "'지금 이 컴퓨터 자신'을 가리키는 주소. 여기서 띄운 서버는 기본적으로 내 PC에서만 접속돼요.",
  "analogy": "'우리 집'이라고 말하는 것 — 남이 들으면 자기 집.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "포트",
  "aliases": [
   "port"
  ],
  "full": "Port (포트 번호)",
  "short": "한 컴퓨터 안에서 프로그램마다 다른 문 번호. 127.0.0.1:8000 의 8000. 같은 포트로 서버를 두 개 띄울 수 없어요.",
  "analogy": "같은 건물 안의 호수(101호, 102호).",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "next_id",
  "aliases": [],
  "full": "next_id (다음에 줄 ID)",
  "short": "다음 새 항목에 줄 번호를 따로 저장한 값. 삭제해도 줄이지 않아 지운 번호를 다시 쓰지 않아요. entries와 같은 파일에 저장해요.",
  "analogy": "번호표 기계 — 손님이 나가도 번호는 되돌리지 않음.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": [
   17
  ]
 },
 {
  "term": "Lock",
  "aliases": [
   "threading.Lock",
   "LOCK",
   "잠금"
  ],
  "full": "threading.Lock (잠금)",
  "short": "with LOCK: 블록 안은 한 번에 한 요청만 실행되게 해요. 읽기→변경→저장을 통째로 묶어야 서로의 저장을 덮어쓰지 않아요.",
  "analogy": "화장실 문 잠금 — 한 명이 나와야 다음 사람이 들어감.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "원자적 저장",
  "aliases": [
   "임시 파일 교체",
   "atomic write"
  ],
  "full": "Atomic write (임시 파일에 쓰고 교체)",
  "short": "임시 파일에 끝까지 쓴 뒤 replace()로 원본과 바꿔치기해요. 쓰다 실패해도 원본은 멀쩡해요.",
  "analogy": "답안을 연습장에 다 쓴 뒤 답안지에 옮기기.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "APIRouter",
  "aliases": [
   "Router",
   "include_router"
  ],
  "full": "fastapi.APIRouter",
  "short": "관련 경로를 한 파일에 묶는 도구. prefix로 공통 주소를 붙이고, app.include_router()로 앱에 연결해요. tags는 /docs 그룹 이름일 뿐이에요.",
  "analogy": "백화점 층별 안내 — '3층 = 가전' 아래 매장들.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Repository",
  "aliases": [
   "리포지토리",
   "저장소 계층"
  ],
  "full": "Repository 패턴",
  "short": "데이터 읽기·저장만 맡는 코드 묶음. 저장 방식을 파일에서 DB로 바꿀 때 이 부분만 고치면 돼요.",
  "analogy": "창고 담당 직원 — 창고가 바뀌어도 매장 직원은 그대로.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "Jinja2",
  "aliases": [
   "jinja2"
  ],
  "full": "Jinja2 템플릿 엔진",
  "short": "HTML 틀에 {{ entry.title }} 같은 자리를 두고 데이터를 채워 화면을 만들어요. JSON API와 같은 데이터를 사람용 화면으로 보여 줄 때 써요.",
  "analogy": "빈칸이 있는 초대장 양식에 이름을 채워 찍기.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "TestClient",
  "aliases": [],
  "full": "fastapi.testclient.TestClient",
  "short": "서버를 띄우지 않고 앱에 바로 요청을 보내 응답을 확인하는 테스트용 클라이언트.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "monkeypatch",
  "aliases": [],
  "full": "pytest monkeypatch",
  "short": "테스트하는 동안만 변수·함수를 다른 값으로 바꿔치기하고, 끝나면 원래대로 돌려놔요. 저장 경로를 임시 파일로 바꿀 때 써요.",
  "analogy": "촬영 동안만 쓰는 대역 배우.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "트랜잭션",
  "aliases": [
   "transaction"
  ],
  "full": "Transaction (트랜잭션)",
  "short": "여러 변경을 '모두 성공 아니면 모두 취소'로 묶는 데이터베이스 기능. 파일 + Lock으로는 대신할 수 없어요.",
  "analogy": "계좌 이체 — 출금만 되고 입금이 안 되는 일은 없어야 함.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "PostgreSQL",
  "aliases": [
   "Postgres"
  ],
  "full": "PostgreSQL (관계형 데이터베이스)",
  "short": "표·관계·SQL로 데이터를 저장하는 데이터베이스. 동시 요청·여러 서버·트랜잭션처럼 JSON 파일로 어려운 일을 맡아요.",
  "analogy": "",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "AI Agent",
  "aliases": [
   "Agent",
   "에이전트"
  ],
  "full": "AI Agent",
  "short": "LLM이 작업을 보고 필요한 도구를 골라 호출을 제안하고 결과를 확인하는 시스템. 실제 호출·권한 확인은 애플리케이션이 책임져요.",
  "analogy": "할 일을 보고 연장을 골라 달라고 하는 조수 — 연장을 건네는 건 사람.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "RAG",
  "aliases": [
   "Retrieval-Augmented Generation",
   "검색 증강 생성"
  ],
  "full": "Retrieval-Augmented Generation (검색 증강 생성)",
  "short": "질문과 관련된 근거 문서를 먼저 검색해 LLM에 함께 넣어 답하게 하는 방식. 원본 지식이 정리돼 있어야 시작할 수 있어요.",
  "analogy": "오픈북 시험 — 교과서에서 찾아보고 답 쓰기.",
  "first": 17,
  "days": [
   17
  ],
  "keyDays": []
 },
 {
  "term": "SQL",
  "aliases": [],
  "full": "Structured Query Language",
  "short": "DB에 '어느 표에서 어떤 행을 보여 줘 / 바꿔 줘'라고 말하는 언어. 구조를 만드는 CREATE와 데이터를 읽고 바꾸는 SELECT·INSERT·UPDATE·DELETE가 있어요.",
  "analogy": "도서관 사서에게 내는 정해진 양식의 요청서.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "DBMS",
  "aliases": [
   "RDBMS"
  ],
  "full": "Database Management System (관계형이면 RDBMS)",
  "short": "데이터를 실제로 저장하고 SQL·규칙·동시 접근을 처리하는 프로그램. PostgreSQL이 RDBMS예요.",
  "analogy": "창고 그 자체가 아니라 창고를 운영하는 관리 사무소.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "pgAdmin",
  "aliases": [
   "Query Tool"
  ],
  "full": "pgAdmin 4 (PostgreSQL GUI 클라이언트)",
  "short": "창으로 PostgreSQL 서버에 접속해 SQL을 보내고 Data Output·Messages로 결과를 보는 도구. 창을 닫아도 서버는 안 꺼져요.",
  "analogy": "서버에 전화를 거는 전화기 — 전화를 끊어도 상대 회사는 계속 영업해요.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "psql",
  "aliases": [],
  "full": "psql (PostgreSQL 터미널 클라이언트)",
  "short": "터미널에서 쓰는 PostgreSQL 접속 도구. \\dt·\\d·\\q 같은 역슬래시 명령은 SQL이 아니라 psql 전용이에요.",
  "analogy": "pgAdmin의 글자판 버전.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "테이블",
  "aliases": [
   "Table"
  ],
  "full": "Table (관계형 DB의 표)",
  "short": "같은 종류의 데이터를 행과 열로 담는 표. 한 행 = 한 건(지식 하나), 한 열 = 공통 항목(제목, 본문).",
  "analogy": "엑셀 시트 한 장 — 단, 열마다 자료형과 규칙이 정해져 있어요.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "PK",
  "aliases": [
   "Primary Key",
   "PRIMARY KEY",
   "기본키"
  ],
  "full": "Primary Key (기본키)",
  "short": "표에서 행 하나를 콕 집어 구별하는 열. 값이 겹치면 안 되고 NULL도 안 돼요. 이번엔 id.",
  "analogy": "학번 — 이름은 같을 수 있어도 학번은 하나뿐.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "FK",
  "aliases": [
   "Foreign Key",
   "FOREIGN KEY",
   "외래키"
  ],
  "full": "Foreign Key (외래키)",
  "short": "다른 표(부모)의 PK를 가리키는 열. 저장할 때 그 번호의 부모 행이 실제로 있는지 DB가 검사해요. 이번엔 knowledge.category_id → categories.id.",
  "analogy": "수강신청서의 과목 번호 — 개설되지 않은 과목 번호는 접수 거부.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "IDENTITY",
  "aliases": [
   "GENERATED ALWAYS AS IDENTITY"
  ],
  "full": "Identity Column (자동 번호 열)",
  "short": "값을 생략하면 DB가 번호를 발급하는 열. 안에서 시퀀스(번호 발급기)를 쓰고, 취소·실패한 번호는 다시 쓰지 않을 수 있어서 빈 번호가 생겨요.",
  "analogy": "은행 번호표 기계 — 뽑고 그냥 가 버린 번호는 다시 안 나와요.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "JOIN",
  "aliases": [
   "INNER JOIN"
  ],
  "full": "JOIN (INNER JOIN)",
  "short": "조회할 때 두 표에서 ON 조건으로 짝이 맞는 행을 붙여 한 줄로 보여 주는 것. 짝이 없는 행은 빠져요.",
  "analogy": "학생 명단과 동아리 명단을 동아리 번호로 맞대어 보기.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "LEFT JOIN",
  "aliases": [
   "LEFT OUTER JOIN"
  ],
  "full": "LEFT (OUTER) JOIN",
  "short": "왼쪽(FROM) 표의 행은 모두 남기고, 짝이 없으면 오른쪽 열을 NULL로 채우는 JOIN. 지식이 0개인 분류도 결과에 남아요.",
  "analogy": "출석부 기준으로 과제 제출을 맞춰 보기 — 안 낸 학생도 이름은 남고 칸이 비어요.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "GROUP BY",
  "aliases": [],
  "full": "GROUP BY (묶어서 집계)",
  "short": "같은 값끼리 행을 묶어 그룹마다 COUNT·SUM 등을 계산. 결과 한 행의 단위가 '그룹 하나'로 바뀌어요.",
  "analogy": "반별로 학생 수 세기.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "HAVING",
  "aliases": [],
  "full": "HAVING (그룹 조건)",
  "short": "GROUP BY로 묶은 뒤 그룹을 거르는 조건. 묶기 전 행을 거르는 WHERE와 순서가 달라요.",
  "analogy": "반별 인원을 센 다음 '30명 이상인 반만'.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "NULL",
  "aliases": [
   "IS NULL"
  ],
  "full": "NULL (값 없음 / 모름)",
  "short": "값이 없거나 모른다는 표시. 빈 문자열 ''과 달라요. = NULL로는 비교가 안 돼서 IS NULL로 확인해요.",
  "analogy": "답안지의 '빈칸'(NULL)과 '0이라고 쓴 칸'('' 또는 0)은 다르다.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "제약조건",
  "aliases": [
   "Constraint"
  ],
  "full": "Constraint (제약조건)",
  "short": "DB가 저장 때마다 검사하는 규칙. PRIMARY KEY·FOREIGN KEY·NOT NULL·UNIQUE·CHECK가 있고, 어기면 행을 저장하지 않고 오류를 내요.",
  "analogy": "입구의 금속 탐지기 — 누가 들어오든 똑같이 검사.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "CHECK",
  "aliases": [
   "CHECK 제약"
  ],
  "full": "CHECK 제약조건",
  "short": "값이 정한 조건을 만족하는지 검사. 예: char_length(content) BETWEEN 1 AND 2000 → 빈 본문 거부(SQLSTATE 23514).",
  "analogy": "'1~2,000자로 작성' 칸에 맞는지 보는 접수 직원.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "UNIQUE",
  "aliases": [],
  "full": "UNIQUE 제약조건",
  "short": "같은 값이 두 번 들어가지 못하게 막아요. 분류 이름은 UNIQUE, 지식 제목은 아님.",
  "analogy": "아이디 중복 확인.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "ON DELETE RESTRICT",
  "aliases": [
   "RESTRICT",
   "CASCADE",
   "ON DELETE CASCADE"
  ],
  "full": "FK 삭제 정책 (RESTRICT / CASCADE)",
  "short": "부모 행을 지울 때 자식이 있으면? RESTRICT는 거부, CASCADE는 자식까지 같이 삭제. 이번 설계는 RESTRICT.",
  "analogy": "학생이 남아 있는 반은 폐반 불가(RESTRICT) vs 반을 없애면 학생 기록도 삭제(CASCADE).",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "DEFAULT",
  "aliases": [],
  "full": "DEFAULT (기본값)",
  "short": "값을 생략했을 때 채우는 값. 직접 NULL을 넣으면 고쳐 주지 않아요.",
  "analogy": "신청서에 안 적으면 '미지정'으로 처리 — 단, '없음'이라고 적으면 그대로 '없음'.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "DDL",
  "aliases": [
   "DML"
  ],
  "full": "Data Definition Language / Data Manipulation Language",
  "short": "DDL은 CREATE TABLE처럼 구조를 만드는 SQL, DML은 INSERT·UPDATE·DELETE처럼 데이터를 바꾸는 SQL.",
  "analogy": "DDL = 서랍장 짜기, DML = 서랍에 물건 넣고 빼기.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "ERD",
  "aliases": [
   "1:N",
   "N:M"
  ],
  "full": "Entity Relationship Diagram",
  "short": "표와 열, 표끼리의 관계(1:N, N:M)를 그린 설계도. 그림일 뿐이고 실제 검사는 CREATE TABLE의 FK가 해요.",
  "analogy": "건물 설계도 — 문이 잠기는지는 실제 자물쇠(FK)가 정해요.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "RETURNING",
  "aliases": [],
  "full": "RETURNING 절",
  "short": "INSERT·UPDATE·DELETE가 바꾼 행을 그 자리에서 돌려받는 PostgreSQL 문법. 새로 발급된 id를 예상하지 않고 바로 읽어요.",
  "analogy": "택배 접수하면 바로 받는 송장 번호.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "COMMIT",
  "aliases": [],
  "full": "COMMIT (확정)",
  "short": "BEGIN 이후의 변경을 확정. 이후 다른 연결·재접속에서도 보이고, ROLLBACK으로 되돌릴 수 없어요.",
  "analogy": "서류에 도장 찍고 제출.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "ROLLBACK",
  "aliases": [],
  "full": "ROLLBACK (취소)",
  "short": "지금 열린 트랜잭션의 미확정 변경을 모두 취소. DB 전체를 과거로 돌리는 버튼은 아니에요.",
  "analogy": "제출 전 초안을 찢어 버리기 — 이미 낸 서류는 못 찢어요.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "BEGIN",
  "aliases": [],
  "full": "BEGIN (트랜잭션 시작)",
  "short": "여러 변경을 한 묶음으로 시작. 이후 변경은 COMMIT 또는 ROLLBACK 전까지 미확정이고 다른 연결엔 안 보여요.",
  "analogy": "'지금부터 하는 건 연습장에' 선언.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "원자성",
  "aliases": [
   "Atomicity"
  ],
  "full": "Atomicity (원자성)",
  "short": "트랜잭션 안의 변경은 전부 반영되거나 전부 취소되거나 둘 중 하나. 반쯤 바뀐 상태로 남지 않아요.",
  "analogy": "계좌 이체 — 출금만 되고 입금이 안 되는 일은 없어야.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "SQLSTATE",
  "aliases": [],
  "full": "SQLSTATE (표준 오류 코드)",
  "short": "DB 오류의 종류를 나타내는 5자리 코드. 23502 NOT NULL, 23503 FK, 23505 UNIQUE, 23514 CHECK, 25P02 중단된 트랜잭션. HTTP 상태 코드와는 별개예요.",
  "analogy": "병원 진단 코드 — 증상 이름 대신 정해진 번호.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "인덱스",
  "aliases": [
   "INDEX",
   "B-tree"
  ],
  "full": "Index (B-tree 인덱스)",
  "short": "표와 따로 만드는 탐색 구조. 조건에 맞는 행을 표 전체를 훑지 않고 찾게 도와요. 결과는 같고 찾아가는 길만 달라요. 의미 검색 인덱스와는 다른 것.",
  "analogy": "책 뒤의 '찾아보기'.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": [
   18
  ]
 },
 {
  "term": "EXPLAIN",
  "aliases": [
   "실행계획",
   "Seq Scan",
   "Index Scan"
  ],
  "full": "EXPLAIN (실행계획 보기)",
  "short": "DB가 SQL을 어떤 방법으로 처리할지 보여 줘요. Seq Scan = 전체 훑기, Index Scan = 인덱스 사용. cost는 밀리초가 아니라 상대 단위. ANALYZE를 붙이면 실제로 실행해요.",
  "analogy": "내비게이션의 '예상 경로' 미리 보기.",
  "first": 18,
  "days": [
   18
  ],
  "keyDays": []
 },
 {
  "term": "ORM",
  "aliases": [
   "Object Relational Mapping"
  ],
  "full": "Object-Relational Mapping",
  "short": "파이썬 클래스·객체로 DB 표·행을 다루게 해 주는 방식. 객체 속성을 바꾸면 ORM이 UPDATE SQL을 만들어 보내요.",
  "analogy": "통역사 — 파이썬 말로 하면 SQL로 바꿔 DB에 전해 줘요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": [
   19
  ]
 },
 {
  "term": "SQLAlchemy",
  "aliases": [],
  "full": "SQLAlchemy (Python DB 툴킷 · ORM)",
  "short": "파이썬에서 가장 널리 쓰는 DB 라이브러리. Engine(연결) · ORM Model · Session · select() 같은 쿼리 작성 도구를 제공해요.",
  "analogy": "DB와 대화하는 데 필요한 공구 상자.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": [
   19
  ]
 },
 {
  "term": "SQLAlchemy Engine",
  "aliases": [
   "create_engine",
   "Connection Pool"
  ],
  "full": "Engine / Connection Pool",
  "short": "DB 연결을 만들고 모아 두었다가(Pool) 빌려주는 기반. 앱에서 하나를 만들어 재사용해요. 만들었다고 접속이 확인된 건 아니에요.",
  "analogy": "공용 자전거 거치대 — 필요할 때 빌리고 반납.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "Session",
  "aliases": [
   "SessionLocal",
   "sessionmaker"
  ],
  "full": "SQLAlchemy Session",
  "short": "ORM 객체 추적 · SQL 실행 · 트랜잭션을 담는 작업 공간. 요청마다 새로 만들고 끝나면 닫아요. close는 저장(commit)이 아니에요.",
  "analogy": "요청 한 건 전용 작업대 — 일 끝나면 치우지만, 결재(commit)는 따로.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": [
   19
  ]
 },
 {
  "term": "psycopg",
  "aliases": [
   "psycopg[binary]",
   "드라이버"
  ],
  "full": "Psycopg 3 (PostgreSQL 드라이버)",
  "short": "파이썬과 PostgreSQL 사이에서 실제로 데이터를 주고받는 드라이버. SQLAlchemy URL의 postgresql+psycopg가 이걸 쓰라는 뜻이에요.",
  "analogy": "전화 회선 — 말(SQL)은 ORM이 만들고, 전달은 회선이 해요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "flush",
  "aliases": [
   "flush()"
  ],
  "full": "Session.flush()",
  "short": "쌓인 변경 SQL을 DB로 보내는 것. DB가 id를 발급해서 알 수 있지만 아직 미확정 — 다른 연결엔 안 보이고 rollback하면 사라져요.",
  "analogy": "결재 서류를 결재함에 넣은 상태 — 아직 도장은 안 찍힘.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": [
   19
  ]
 },
 {
  "term": ".env",
  "aliases": [
   "환경변수",
   ".env.example"
  ],
  "full": ".env 파일 (환경 설정)",
  "short": "DB 주소·계정·비밀번호처럼 코드 밖에 둘 설정을 '이름=값'으로 적은 파일. .gitignore로 Git에서 빼고, 값 없는 .env.example만 공유해요. 암호화는 아니에요.",
  "analogy": "열쇠를 코드 안이 아니라 따로 둔 열쇠함.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "SecretStr",
  "aliases": [
   "pydantic-settings",
   "BaseSettings"
  ],
  "full": "pydantic SecretStr · pydantic-settings",
  "short": "pydantic-settings의 BaseSettings가 .env를 읽어 타입 있는 설정 객체를 만들고, SecretStr 필드는 출력할 때 별표로 가려요. 꺼낼 땐 get_secret_value().",
  "analogy": "봉투에 든 비밀번호 — 겉에선 안 보이고, 필요할 때만 열어요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "Depends",
  "aliases": [
   "의존성 주입",
   "get_db",
   "yield 의존성"
  ],
  "full": "FastAPI Depends (의존성 주입)",
  "short": "함수 매개변수를 FastAPI가 대신 준비해 주는 장치. get_db처럼 yield를 쓰면 yield 앞에서 준비, 요청이 끝나면 뒤에서 정리해요.",
  "analogy": "주방장이 요리 시작 전에 깨끗한 도마를 받아 쓰고, 끝나면 설거지통에 넣는 규칙.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "Migration",
  "aliases": [
   "마이그레이션"
  ],
  "full": "Schema Migration (DB 구조 변경)",
  "short": "DB의 표·열 구조를 바꾸는 작업을 순서 있는 기록으로 남겨 적용하는 것. ORM Model을 고쳐도 DB는 Migration을 적용해야 바뀌어요.",
  "analogy": "건물 리모델링 공사 일지 — 무엇을 어떤 순서로 바꿨는지 남겨요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": [
   19
  ]
 },
 {
  "term": "Alembic",
  "aliases": [],
  "full": "Alembic (SQLAlchemy용 Migration 도구)",
  "short": "Revision 파일로 DB 구조 변경을 기록하고 upgrade·downgrade로 적용·되돌림. alembic_version 표에 현재 위치(current)를 기록해요.",
  "analogy": "공사 일지 + 공사 반장.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": [
   19
  ]
 },
 {
  "term": "Revision",
  "aliases": [
   "head",
   "down_revision",
   "upgrade",
   "downgrade"
  ],
  "full": "Alembic Revision",
  "short": "변경 하나를 담은 파이썬 파일. upgrade()로 적용, downgrade()로 되돌림. down_revision으로 앞 변경과 이어져 사슬이 되고, 마지막이 head.",
  "analogy": "책의 개정판 번호 — 2판은 1판 다음이라고 적혀 있어요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "autogenerate",
  "aliases": [
   "--autogenerate",
   "stamp"
  ],
  "full": "alembic revision --autogenerate / alembic stamp",
  "short": "autogenerate는 DB와 Model 차이로 Revision 초안을 만들 뿐이라 꼭 읽고 고쳐야 해요(이름 변경이 삭제+추가로 나옴). stamp는 DDL 없이 기록만 바꿔요.",
  "analogy": "맞춤법 검사기의 수정 제안 — 그대로 다 누르면 뜻이 바뀌기도 해요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "IntegrityError",
  "aliases": [
   "PendingRollbackError"
  ],
  "full": "sqlalchemy.exc.IntegrityError",
  "short": "DB 제약(UNIQUE·FK·CHECK·NOT NULL) 위반 때 SQLAlchemy가 내는 예외. exc.orig.sqlstate와 diag.constraint_name으로 원인을 구분해요. 실패한 Session은 rollback 전까지 PendingRollbackError로 다음 SQL을 거부해요.",
  "analogy": "규칙 위반 딱지 — 어느 규칙인지 번호와 이름이 적혀 있어요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "409 Conflict",
  "aliases": [
   "Data conflict"
  ],
  "full": "HTTP 409 Conflict",
  "short": "요청이 지금 데이터 상태와 충돌했다는 응답. 이 수업에선 알려진 UNIQUE(분류명 중복)·FK 충돌만 409, 나머지 DB 오류는 500.",
  "analogy": "이미 예약된 자리를 또 예약하려 할 때의 '중복 예약' 안내.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "503 Service Unavailable",
  "aliases": [
   "Database unavailable"
  ],
  "full": "HTTP 503 Service Unavailable",
  "short": "서버가 지금 일을 처리할 수 없다는 응답. 이 수업에선 연결 계열 SQLSTATE(08…)나 확인된 연결 끊김만 503으로 봐요.",
  "analogy": "'잠시 영업 준비 중' 팻말.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "SQL Injection",
  "aliases": [
   "바인딩",
   "파라미터 바인딩"
  ],
  "full": "SQL Injection / Parameter Binding",
  "short": "사용자 입력을 SQL 문자열에 그대로 이어 붙이면 입력으로 SQL을 조작당할 수 있어요(Injection). select().where(...)처럼 값을 따로 전달(바인딩)하면 막혀요.",
  "analogy": "주문서 빈칸에 '그리고 금고도 열어 줘'라고 적는 것 — 칸 안의 글은 그냥 글로만 읽어야 해요.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 },
 {
  "term": "dependency_overrides",
  "aliases": [],
  "full": "FastAPI app.dependency_overrides",
  "short": "테스트에서 get_db 같은 의존성을 다른 함수(테스트 DB용)로 바꿔 끼우는 사전. 끝나면 pop으로 원래대로 돌려요.",
  "analogy": "리허설 때만 진짜 금고 대신 연습용 금고를 연결.",
  "first": 19,
  "days": [
   19
  ],
  "keyDays": []
 }
];
