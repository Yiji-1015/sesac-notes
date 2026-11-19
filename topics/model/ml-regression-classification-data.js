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
   "t": "from sklearn.metrics import r2_score",
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
     "R² → -0.097",
     1,
     "bad"
    ]
   ],
   "notes": [
    "R²는 \"평균으로 찍는 것보다 얼마나 나은가\"예요. 1이면 완벽, 0이면 평균과 같고, <b>0보다 작으면 평균보다 못해요</b>. 이 네 입력만으로는 처리 시간을 거의 설명하지 못한다는 뜻이에요. 모델을 바꾸기 전에 \"입력이 충분한가\"부터 의심해야 하는 신호예요."
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
    ]
   ],
   "notes": [
    "실제 실패 6건 중 2건만 잡았어요. 4건은 놓쳤어요.",
    "정확도는 기준선(0.70)보다 낮지만, 기준선은 실패를 <b>하나도</b> 못 잡아요(F1 0). 실패를 찾는 게 목적이면 정확도보다 recall(실패 중 잡은 비율)과 F1을 봐야 해요."
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
   "선형",
   "MAE, R²"
  ],
  [
   5,
   "분류",
   "로지스틱",
   "혼동 행렬, F1"
  ],
  [
   6,
   "KNN",
   "거리",
   "스케일"
  ],
  [
   7,
   "비교",
   "같은 조건",
   "기준이 결과를 바꿈"
  ]
 ],
 "env": "Python 3.13.16, scikit-learn 1.8.0, xgboost 3.1.3, pandas 2.2.3"
};
