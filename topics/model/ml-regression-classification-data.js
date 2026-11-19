window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "import pandas as pd",
   "s": 1
  },
  {
   "t": "from sklearn.dummy import DummyClassifier, DummyRegressor",
   "s": 2
  },
  {
   "t": "from sklearn.metrics import accuracy_score, mean_absolute_error",
   "s": 2
  },
  {
   "t": "from sklearn.base import clone",
   "s": 3
  },
  {
   "t": "from sklearn.compose import ColumnTransformer",
   "s": 3
  },
  {
   "t": "from sklearn.impute import SimpleImputer",
   "s": 3
  },
  {
   "t": "from sklearn.pipeline import Pipeline",
   "s": 3
  },
  {
   "t": "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
   "s": 3
  },
  {
   "t": "from sklearn.linear_model import LinearRegression",
   "s": 4
  },
  {
   "t": "from sklearn.metrics import r2_score, root_mean_squared_error",
   "s": 4
  },
  {
   "t": "from sklearn.linear_model import LogisticRegression",
   "s": 5
  },
  {
   "t": "from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score",
   "s": 5
  },
  {
   "t": "from sklearn.metrics import log_loss",
   "s": 5
  },
  {
   "t": "from sklearn.neighbors import KNeighborsClassifier",
   "s": 6
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
   "t": "# ① 데이터 읽고 나누기",
   "s": 1
  },
  {
   "t": "raw = pd.read_csv(DATA / \"requests.csv\").set_index(\"request_id\")",
   "s": 1
  },
  {
   "t": "manifest = pd.read_csv(DATA / \"split_manifest.csv\")",
   "s": 1
  },
  {
   "t": "FEATURES = [\"input_chars\", \"allowed_tools\", \"task_type\", \"priority\"]  # 요청을 받을 때 이미 아는 값만",
   "s": 1
  },
  {
   "t": "NUM, CAT = [\"input_chars\", \"allowed_tools\"], [\"task_type\", \"priority\"]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def part(name):",
   "s": 1
  },
  {
   "t": "    ids = manifest[manifest[\"split\"] == name].sort_values(\"position\")[\"request_id\"]",
   "s": 1
  },
  {
   "t": "    return raw.loc[ids]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "train, valid, test = part(\"train\"), part(\"valid\"), part(\"test\")",
   "s": 1
  },
  {
   "t": "X_train, X_valid = train[FEATURES], valid[FEATURES]",
   "s": 1
  },
  {
   "t": "y_train, y_valid = train[\"failed\"], valid[\"failed\"]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 기준선 만들기",
   "s": 2
  },
  {
   "t": "base_clf = DummyClassifier(strategy=\"most_frequent\").fit(X_train, y_train)",
   "s": 2
  },
  {
   "t": "base_acc = accuracy_score(y_valid, base_clf.predict(X_valid))",
   "s": 2
  },
  {
   "t": "base_reg = DummyRegressor(strategy=\"mean\").fit(X_train, train[\"elapsed_ms\"])",
   "s": 2
  },
  {
   "t": "base_mae = mean_absolute_error(valid[\"elapsed_ms\"], base_reg.predict(X_valid))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 전처리를 Pipeline으로 묶기",
   "s": 3
  },
  {
   "t": "def build(model, scale=True):",
   "s": 3
  },
  {
   "t": "    num = Pipeline([(\"fill\", SimpleImputer(strategy=\"median\")),",
   "s": 3
  },
  {
   "t": "                    (\"scale\", StandardScaler() if scale else \"passthrough\")])",
   "s": 3
  },
  {
   "t": "    cat = Pipeline([(\"fill\", SimpleImputer(strategy=\"most_frequent\")),",
   "s": 3
  },
  {
   "t": "                    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\"))])",
   "s": 3
  },
  {
   "t": "    pre = ColumnTransformer([(\"num\", num, NUM), (\"cat\", cat, CAT)])",
   "s": 3
  },
  {
   "t": "    return Pipeline([(\"pre\", pre), (\"model\", clone(model))])",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 회귀: 처리 시간 예측",
   "s": 4
  },
  {
   "t": "reg = build(LinearRegression()).fit(X_train, train[\"elapsed_ms\"])",
   "s": 4
  },
  {
   "t": "reg_pred = reg.predict(X_valid)",
   "s": 4
  },
  {
   "t": "reg_mae = mean_absolute_error(valid[\"elapsed_ms\"], reg_pred)",
   "s": 4
  },
  {
   "t": "reg_r2 = r2_score(valid[\"elapsed_ms\"], reg_pred)",
   "s": 4
  },
  {
   "t": "reg_rmse = root_mean_squared_error(valid[\"elapsed_ms\"], reg_pred)",
   "s": 4
  },
  {
   "t": "reg_coef = pd.Series(reg.named_steps[\"model\"].coef_,  # 배운 가중치(열마다 하나)",
   "s": 4
  },
  {
   "t": "                     index=reg.named_steps[\"pre\"].get_feature_names_out())",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 분류: 실패 여부 예측",
   "s": 5
  },
  {
   "t": "logit = build(LogisticRegression(max_iter=1000)).fit(X_train, y_train)",
   "s": 5
  },
  {
   "t": "pred = logit.predict(X_valid)",
   "s": 5
  },
  {
   "t": "scores = {\"accuracy\": accuracy_score(y_valid, pred), \"precision\": precision_score(y_valid, pred, zero_division=0),",
   "s": 5
  },
  {
   "t": "          \"recall\": recall_score(y_valid, pred), \"f1\": f1_score(y_valid, pred)}",
   "s": 5
  },
  {
   "t": "cm = confusion_matrix(y_valid, pred, labels=[0, 1])",
   "s": 5
  },
  {
   "t": "z = logit.decision_function(X_valid)  # 가중치를 곱해 더한 점수 (아무 실수)",
   "s": 5
  },
  {
   "t": "p_fail = 1 / (1 + np.exp(-z))  # 시그모이드로 0~1 확률로",
   "s": 5
  },
  {
   "t": "loss = log_loss(y_valid, p_fail)  # 학습이 줄이려는 비용 (로그 손실)",
   "s": 5
  },
  {
   "t": "odds = pd.Series(np.exp(logit.named_steps[\"model\"].coef_[0]),",
   "s": 5
  },
  {
   "t": "                 index=logit.named_steps[\"pre\"].get_feature_names_out())  # 열이 1 커질 때 odds가 몇 배",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ KNN: 거리로 판단하는 모델과 스케일",
   "s": 6
  },
  {
   "t": "knn_scaled = build(KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)",
   "s": 6
  },
  {
   "t": "knn_raw = build(KNeighborsClassifier(n_neighbors=5), scale=False).fit(X_train, y_train)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 같은 조건으로 비교하기",
   "s": 7
  },
  {
   "t": "def evaluate(models):",
   "s": 7
  },
  {
   "t": "    rows = []",
   "s": 7
  },
  {
   "t": "    for name, m in models.items():",
   "s": 7
  },
  {
   "t": "        p = m.fit(X_train, y_train).predict(X_valid)",
   "s": 7
  },
  {
   "t": "        rows.append({\"model\": name, \"f1\": round(f1_score(y_valid, p), 3),",
   "s": 7
  },
  {
   "t": "                     \"recall\": round(recall_score(y_valid, p), 3), \"accuracy\": round(accuracy_score(y_valid, p), 3)})",
   "s": 7
  },
  {
   "t": "    return pd.DataFrame(rows).set_index(\"model\")",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "board = evaluate({\"dummy\": DummyClassifier(strategy=\"most_frequent\"),",
   "s": 7
  },
  {
   "t": "                  \"logistic\": build(LogisticRegression(max_iter=1000)),",
   "s": 7
  },
  {
   "t": "                  \"knn\": build(KNeighborsClassifier(n_neighbors=5))})",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  }
 ],
 "names": {
  "1": "데이터 읽고 나누기",
  "2": "기준선 만들기",
  "3": "전처리를 Pipeline으로 묶기",
  "4": "회귀: 처리 시간 예측",
  "5": "분류: 실패 여부 예측",
  "6": "KNN: 거리로 판단하는 모델과 스케일",
  "7": "같은 조건으로 비교하기",
  "8": "결정 트리",
  "9": "랜덤 포레스트",
  "10": "XGBoost",
  "11": "특성 중요도",
  "12": "교차검증",
  "13": "임계값과 오류 비용",
  "14": "ROC-AUC: 임계값과 상관없는 순서 점수",
  "15": "특성 설계: 치우친 값 펴기",
  "16": "K-Means: 정답 없이 묶어 보기",
  "17": "마지막 한 번: test로 최종 평가"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "f_ls": {
   "title": "입력 길이 하나로 그은 직선과, 8열 모델의 valid 예측",
   "images": [
    [
     "img/ml/least_squares.png",
     "왼쪽: train, 오른쪽: valid"
    ]
   ],
   "notes": [
    "왼쪽은 설명을 위해 <code>input_chars</code> 한 열만 쓴 직선이에요. 직선과 점 사이 분홍 선이 오차(잔차)이고, 최소제곱법은 이 길이의 제곱 합이 가장 작은 직선을 골라요. 그래도 점이 넓게 흩어져서, 이 한 열로 설명되는 건 train에서도 R² 0.15뿐이에요.",
    "오른쪽 점이 회색 점선(완벽한 예측)에 모이지 않고 주황 선(평균) 근처에 납작하게 깔려 있어요. 모델이 입력으로 시간 차이를 거의 못 가려낸다는 뜻이고, R²가 0 아래로 내려간 이유예요."
   ],
   "chart": true
  },
  "f_gd": {
   "title": "같은 8열 회귀를 경사하강법으로 풀면",
   "images": [
    [
     "img/ml/gradient_descent.png",
     "반복할수록 비용이 줄어드는 모습"
    ]
   ],
   "notes": [
    "0.3은 다섯 번 남짓 만에 공식으로 구한 바닥(점선)에 거의 닿아요. 0.003은 60번을 가도 한참 위예요. 1.25는 걸음이 너무 커서 매번 바닥을 지나쳐 반대편 더 높은 곳에 떨어지고, 그래서 비용이 오히려 계속 커져요 (발산).",
    "이 데이터에서는 learning rate가 약 1.18를 넘으면 발산해요. 그래서 실제로는 0.001, 0.01, 0.1, 1처럼 열 배씩 바꿔 시험하고, 이 그림처럼 반복마다 비용이 줄어드는지 그려서 확인해요."
   ],
   "chart": true
  },
  "f_path": {
   "title": "alpha를 0.1부터 1000까지 키우면 (가중치 8개)",
   "images": [
    [
     "img/ml/regularization_path.png",
     "굵은 선이 input_chars"
    ]
   ],
   "notes": [
    "Ridge는 선들이 함께 0으로 다가가지만 끝까지 닿지 않아요. Lasso는 선이 하나씩 0에 붙고, 마지막까지 남는 게 가장 쓸모 있는 <code>input_chars</code>예요."
   ],
   "chart": true
  },
  "f_sig": {
   "title": "⑤의 로지스틱 회귀가 valid 20건에 준 점수와 확률",
   "images": [
    [
     "img/ml/sigmoid_valid.png",
     "가로: 점수 z, 세로: 실패 확률"
    ]
   ],
   "notes": [
    "모든 점이 곡선 위에 있어요. 확률은 점수 z를 시그모이드에 넣은 값일 뿐이라서요.",
    "오른쪽 위(z ≥ 0)에 있는 점만 \"실패\"라고 답해요. 분홍 X가 왼쪽에 4개 있죠 (z ≈ −1.5의 두 개는 겹쳐 있어요). 그게 ⑤에서 놓친 실패 4건이에요. 보라 점도 오른쪽에 4개 있어요 (헛경보 4건)."
   ],
   "chart": true
  },
  "f_ll": {
   "title": "로그 손실과 제곱 오차",
   "images": [
    [
     "img/ml/log_loss_curve.png",
     "실제 실패인 요청 하나의 비용"
    ]
   ],
   "notes": [
    "실제 실패인데 p를 0.1로 줬다면 로그 손실은 2.30이지만, 제곱 오차는 0.81에서 멈춰요. 제곱 오차는 아무리 크게 틀려도 1을 넘지 않아서, 자신 있게 틀린 걸 충분히 벌주지 못해요."
   ],
   "chart": true
  },
  "f_knn": {
   "title": "train 60건으로 그린 KNN의 판단 지도 (숫자 두 열만 사용)",
   "images": [
    [
     "img/ml/knn_boundaries.png",
     "분홍 바탕: \"실패\"라고 답하는 곳"
    ]
   ],
   "notes": [
    "그림으로 보려고 <code>input_chars</code>와 <code>allowed_tools</code> 두 열만 쓴 모델이에요. ⑥의 실제 모델(8열)과는 다르지만 k의 효과는 같아요.",
    "k=1은 실패 점 근처를 좁은 띠와 조각으로 감싸서 경계가 들쭉날쭉해요. train도 거의 다 맞혀요. 100%가 안 되는 건, 두 열만 보면 같은 자리인데 답이 다른 요청이 있어서예요 (8열 모델에서는 1.000).",
    "k=31은 오른쪽 아래 귀퉁이만 빼고 분홍 바탕이 사라져서, 거의 어디서든 \"성공\"이라고 답해요."
   ],
   "chart": true
  },
  "f_k": {
   "title": "k를 1부터 31까지 (8열 모델, ⑥과 같은 조건)",
   "images": [
    [
     "img/ml/knn_k_curve.png",
     ""
    ]
   ],
   "notes": [
    "검은 선(본 데이터)은 k=1에서 1.0이고 k가 커질수록 내려가요. 분홍 선(처음 보는 데이터)은 k=3에서 가장 높고, 오르내리다 k=13부터는 0에 붙어요. valid 20건에서 실패 한두 건이 맞고 틀리는 차이라 들쭉날쭉해요.",
    "두 선의 간격이 클수록 외운 정도가 커요. 가장 좋은 k는 두 극단 사이 어딘가에 있고, 교차검증으로 찾아요 (2쪽)."
   ],
   "chart": true
  },
  "data": {
   "title": "requests.csv 앞 5줄",
   "cols": [
    "request_id",
    "input_chars",
    "allowed_tools",
    "task_type",
    "priority",
    "elapsed_ms",
    "retry_count",
    "status",
    "failed"
   ],
   "rows": [
    [
     "R001",
     "1549.0",
     "4.0",
     "summary",
     "normal",
     "1994",
     "2",
     "error",
     "1"
    ],
    [
     "R002",
     "1349.0",
     "2.0",
     "summary",
     "low",
     "1799",
     "0",
     "ok",
     "0"
    ],
    [
     "R003",
     "1578.0",
     "4.0",
     "summary",
     "normal",
     "179",
     "0",
     "ok",
     "0"
    ],
    [
     "R004",
     "946.0",
     "nan",
     "qa",
     "low",
     "1780",
     "0",
     "ok",
     "0"
    ],
    [
     "R005",
     "nan",
     "5.0",
     "qa",
     "high",
     "1979",
     "1",
     "error",
     "1"
    ]
   ],
   "notes": [
    "<code>failed</code>(실패했나, 1이면 실패)와 <code>elapsed_ms</code>(처리 시간)를 맞히는 게 목표예요.",
    "<code>elapsed_ms</code>, <code>retry_count</code>, <code>status</code>는 요청을 <b>처리한 뒤에야</b> 알 수 있는 값이라 입력으로 쓰면 안 돼요. 미래를 보고 맞히는 셈이라서요 (데이터 누수)."
   ]
  },
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    [
     "train 60건, valid 20건, test 20건",
     1
    ],
    [
     "실패 비율 → train 0.30, valid 0.30, test 0.30",
     2
    ],
    "빈 칸 → input_chars 6개, allowed_tools 4개"
   ],
   "notes": [
    "나누는 방법을 파일(manifest)로 고정해 둬서, 몇 번을 돌려도 같은 행이 같은 쪽에 들어가요.",
    "세 쪽 모두 실패가 30%로 비슷하게 나뉘었어요. 빈 칸은 ③에서 채워요."
   ]
  },
  "s2": {
   "title": "② 기준선",
   "lines": [
    [
     "분류: 언제나 \"성공\"이라고 답하기 → 정확도 0.70",
     1
    ],
    [
     "회귀: 언제나 train 평균 시간이라고 답하기 → MAE 541 ms",
     2
    ]
   ],
   "notes": [
    "아무것도 배우지 않은 모델이 정확도 0.70이에요. 실패가 30%뿐이라서 \"다 성공\"이라고만 해도 70%는 맞거든요. 그러니 정확도 0.70은 잘한 게 아니에요.",
    "평균으로만 찍어도 평균 541 ms 틀려요. 모델은 이보다 나아야 의미가 있어요."
   ]
  },
  "s3": {
   "title": "③ 전처리만 따로 돌려 보면",
   "lines": [
    [
     "입력 4열 → 전처리 뒤 8열",
     1
    ],
    "['num__input_chars', 'num__allowed_tools', 'cat__task_type_qa', 'cat__task_type_research', 'cat__task_type_summary', 'cat__priority_high', 'cat__priority_low', 'cat__priority_normal']"
   ],
   "notes": [
    "숫자 두 열은 빈 칸을 중앙값으로 채우고 평균 0, 표준편차 1로 맞췄어요. 글자 두 열은 값마다 0/1 열로 펼쳤어요 (원-핫). 모델은 숫자만 받으니까요."
   ]
  },
  "s4": {
   "title": "④ 선형 회귀로 처리 시간 예측 (valid)",
   "lines": [
    "MAE → 552 ms  (기준선 541 ms)",
    [
     "RMSE → 640 ms",
     1
    ],
    [
     "R² → -0.097",
     2,
     "bad"
    ],
    "",
    [
     "배운 가중치: 절편 1232.1, input_chars +227.5, task_type_qa +75.0, task_type_summary -74.9, priority_normal -37.9",
     3
    ]
   ],
   "notes": [
    "RMSE가 MAE보다 커요. 제곱을 거치면서 크게 빗나간 몇 건이 더 무겁게 잡혔다는 뜻이에요.",
    "R²는 \"평균으로 찍는 것보다 얼마나 나은가\"예요. 1이면 완벽, 0이면 평균과 같고, <b>0보다 작으면 평균보다 못해요</b>. 이 네 입력만으로는 처리 시간을 거의 설명하지 못한다는 뜻이에요. 모델을 바꾸기 전에 \"입력이 충분한가\"부터 의심해야 하는 신호예요.",
    "숫자 열은 ③에서 표준편차 1로 맞췄으니, <code>input_chars</code> +227.5은 \"입력 길이가 표준편차 하나만큼 길면 228 ms 더 걸린다고 배웠다\"는 뜻이에요. 절편은 숫자 열이 평균이고 원-핫 열이 모두 0일 때 식이 내는 값이에요."
   ]
  },
  "t4": {
   "title": "가중치에 벌점을 주면 (valid)",
   "cols": [
    "모델",
    "alpha",
    "input_chars",
    "allowed_tools",
    "task_type_qa",
    "0인 가중치",
    "MAE",
    "R²"
   ],
   "rows": [
    [
     "선형 회귀",
     "-",
     "227.5",
     "13.3",
     "75.0",
     "<b>0</b> / 8",
     "552",
     "-0.097"
    ],
    [
     "Ridge",
     10,
     "191.3",
     "13.0",
     "39.6",
     "<b>0</b> / 8",
     "554",
     "-0.105"
    ],
    [
     "Ridge",
     100,
     "82.9",
     "10.2",
     "3.7",
     "<b>0</b> / 8",
     "546",
     "-0.130"
    ],
    [
     "Lasso",
     10,
     "216.9",
     "0.0",
     "37.1",
     "<b>3</b> / 8",
     "557",
     "-0.111"
    ],
    [
     "Lasso",
     50,
     "174.0",
     "0.0",
     "0.0",
     "<b>7</b> / 8",
     "556",
     "-0.122"
    ]
   ],
   "notes": [
    "Ridge는 alpha를 키울수록 <b>모든</b> 가중치를 조금씩 0 쪽으로 줄여요. 하나도 정확히 0이 되지는 않아요.",
    "Lasso는 덜 쓸모 있는 가중치부터 <b>정확히 0</b>으로 만들어요. alpha 50에서는 <code>input_chars</code> 하나만 남았어요. 열을 골라 주는 효과예요.",
    "그래도 R²는 나아지지 않았어요. 이 데이터의 문제는 \"가중치가 너무 커서\"(과적합)가 아니라 \"설명할 입력이 없어서\"라서요. 규제는 과적합을 고치는 도구예요. alpha는 데이터를 여러 번 나눠 재는 교차검증으로 골라요 (2쪽)."
   ]
  },
  "s5": {
   "title": "⑤ 로지스틱 회귀로 실패 예측 (valid 20건)",
   "lines": [
    "혼동 행렬 (행: 실제, 열: 예측)",
    "            예측 성공  예측 실패",
    "실제 성공       10         4",
    [
     "실제 실패        4         2",
     1
    ],
    "",
    [
     "accuracy 0.60, precision 0.33, recall 0.33, F1 0.33",
     2
    ],
    "",
    [
     "R094: 점수 z -1.805 → 시그모이드 0.141 → 0.5보다 작아서 \"성공\"",
     3,
     "newf"
    ],
    [
     "로그 손실 → 0.703  (누구에게나 train 실패 비율 0.30만 말하면 0.611)",
     4,
     "bad"
    ],
    [
     "odds 배수: input_chars ×4.84, task_type_qa ×1.74, task_type_research ×0.65, priority_normal ×0.67",
     5,
     "newf"
    ]
   ],
   "notes": [
    "실제 실패 6건 중 2건만 잡았어요. 4건은 놓쳤어요.",
    "정확도는 기준선(0.70)보다 낮지만, 기준선은 실패를 <b>하나도</b> 못 잡아요(F1 0). 실패를 찾는 게 목적이면 정확도보다 recall(실패 중 잡은 비율)과 F1을 봐야 해요.",
    "z는 음수라 확률이 0.5 아래예요. 직접 계산한 <code>p_fail</code>은 모델의 확률 함수 <code>predict_proba</code>가 내는 값과 같아요. 모델 안에서도 이 계산을 해요.",
    "로그 손실이 \"모두에게 같은 확률 말하기\"보다 <b>커요</b>. 확률을 틀린 방향으로 자신 있게 낸 요청이 있다는 뜻이에요. 이 모델의 확률은 실패와 성공의 순서조차 잘 못 매긴다는 신호예요. 확률의 순서가 얼마나 맞는지는 3쪽에서 ROC-AUC라는 점수로 따로 재요.",
    "<code>input_chars</code>가 표준편차 하나만큼 길면 실패 odds가 4.84배가 된다고 배웠어요. 1보다 작으면 실패 쪽을 줄이는 열이에요 (<code>priority_normal</code> ×0.67)."
   ]
  },
  "t5": {
   "title": "C를 바꾸면 (valid)",
   "cols": [
    "C",
    "가중치 절댓값 합",
    "F1",
    "recall",
    "accuracy"
   ],
   "rows": [
    [
     0.01,
     "0.205",
     "0.00",
     "0.00",
     "0.70"
    ],
    [
     0.1,
     "1.224",
     "0.00",
     "0.00",
     "0.70"
    ],
    [
     "<b>1</b> (지금)",
     "3.496",
     "0.33",
     "0.33",
     "0.60"
    ],
    [
     100,
     "5.749",
     "0.31",
     "0.33",
     "0.55"
    ]
   ],
   "notes": [
    "C는 규제 세기의 <b>역수</b>예요 (C = 1/λ). 작을수록 벌점이 세서 가중치가 0 근처로 눌려요.",
    "C 0.1 이하에서는 가중치가 너무 작아져서 모든 요청에 실패 확률 0.5 미만을 줘요. 그래서 기준선처럼 \"다 성공\"이라고 답하고 F1이 0이에요.",
    "C를 100으로 풀어도 크게 나아지지 않아요. 기본값 1은 무난한 출발점이고, 고를 때는 교차검증으로 비교해요."
   ]
  },
  "t6": {
   "title": "k와 거리를 바꾸면 (스케일 있음)",
   "cols": [
    "k와 설정",
    "train 정확도",
    "valid F1",
    "valid 정확도"
   ],
   "rows": [
    [
     "1",
     "1.000",
     "0.40",
     "0.55"
    ],
    [
     "3",
     "0.800",
     "0.55",
     "0.75"
    ],
    [
     "<b>5 (지금)</b>",
     "0.767",
     "0.20",
     "0.60"
    ],
    [
     "9",
     "0.783",
     "0.00",
     "0.65"
    ],
    [
     "31",
     "0.717",
     "0.00",
     "0.70"
    ],
    [
     "5, 맨해튼 거리",
     "0.800",
     "0.25",
     "0.70"
    ],
    [
     "5, 가까울수록 큰 표",
     "1.000",
     "0.33",
     "0.60"
    ]
   ],
   "notes": [
    "k=1이면 train 정확도가 1.000이에요. train의 모든 요청은 자기 자신이 가장 가까운 이웃이라서, 외운 답을 그대로 말해요. \"가까울수록 큰 표\"도 같은 이유로 1.000이에요.",
    "k=9와 31에서는 실패를 하나도 맞히지 못해요 (F1 0). k=31이면 이웃 31개의 다수가 늘 성공이라 모든 요청에 \"성공\"이라고 답해요. 정확도 0.70이 기준선과 똑같은 이유예요. k가 작으면 과적합, 크면 과소적합이에요.",
    "valid에서는 k=3이 가장 좋지만, valid가 20건이라 한두 건 차이예요. k도 교차검증으로 골라요 (2쪽)."
   ]
  },
  "s6": {
   "title": "⑥ KNN (k=5), 스케일 있음과 없음 (valid F1)",
   "lines": [
    "스케일 있음 → 0.20",
    "스케일 없음 → 0.40",
    "input_chars 범위 203~2381, allowed_tools 범위 1~5"
   ],
   "notes": [
    "스케일이 없으면 거리가 거의 <code>input_chars</code>(수백~수천)로만 정해지고, <code>allowed_tools</code>(1~5)는 무시돼요. 이 데이터에서는 그게 오히려 점수가 높게 나왔어요. 입력 중 <code>input_chars</code>가 가장 쓸모 있어서일 수도, valid가 20건뿐이라 우연일 수도 있어요. 한 번의 점수로 결론 내리지 않는 이유예요 (2쪽의 교차검증)."
   ]
  },
  "s7": {
   "title": "⑦ 같은 train, 같은 valid로 비교",
   "cols": [
    "model",
    "F1",
    "recall",
    "accuracy"
   ],
   "rows": [
    [
     "dummy",
     "0.0",
     "0.0",
     "0.7"
    ],
    [
     "logistic",
     "0.333",
     "0.333",
     "0.6"
    ],
    [
     "knn",
     "0.2",
     "0.167",
     "0.6"
    ]
   ],
   "notes": [
    "정확도만 보면 dummy가 1등이에요. F1로 보면 logistic이 가장 나아요. <b>무엇을 기준으로 고르느냐</b>가 결과를 바꿔요.",
    "모두 같은 나누기, 같은 전처리 함수를 썼어요. 조건이 다르면 비교가 아니에요."
   ]
  }
 },
 "flow": [
  [
   1,
   "읽고 나누기",
   "60 / 20 / 20",
   "누수 열 빼기"
  ],
  [
   2,
   "기준선",
   "Dummy",
   "정확도 0.70"
  ],
  [
   3,
   "Pipeline",
   "채우기, 맞추기",
   "원-핫"
  ],
  [
   4,
   "회귀",
   "선형, 제곱 오차",
   "가중치, 규제"
  ],
  [
   5,
   "분류",
   "로지스틱, 시그모이드",
   "확률, odds"
  ],
  [
   6,
   "KNN",
   "거리, 다수결",
   "k, 스케일"
  ],
  [
   7,
   "비교",
   "같은 조건",
   "기준이 결과를 바꿈"
  ]
 ],
 "env": "Python 3.13.5, scikit-learn 1.9.0, xgboost 3.4.1, pandas 2.3.3"
};
