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
   "t": "from sklearn.tree import DecisionTreeClassifier",
   "s": 8
  },
  {
   "t": "from sklearn.ensemble import RandomForestClassifier",
   "s": 9
  },
  {
   "t": "from xgboost import XGBClassifier",
   "s": 10
  },
  {
   "t": "from sklearn.model_selection import StratifiedKFold, cross_validate",
   "s": 12
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
  },
  {
   "t": "# ⑧ 결정 트리",
   "s": 8
  },
  {
   "t": "tree = build(DecisionTreeClassifier(max_depth=3, random_state=0), scale=False).fit(X_train, y_train)",
   "s": 8
  },
  {
   "t": "deep = build(DecisionTreeClassifier(random_state=0), scale=False).fit(X_train, y_train)",
   "s": 8
  },
  {
   "t": "tree_scores = {name: (accuracy_score(y_train, m.predict(X_train)), f1_score(y_valid, m.predict(X_valid)))",
   "s": 8
  },
  {
   "t": "               for name, m in {\"max_depth=3\": tree, \"제한 없음\": deep}.items()}",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "# ⑨ 랜덤 포레스트",
   "s": 9
  },
  {
   "t": "forest = build(RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0), scale=False)",
   "s": 9
  },
  {
   "t": "",
   "s": 9
  },
  {
   "t": "# ⑩ XGBoost",
   "s": 10
  },
  {
   "t": "boost = build(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0), scale=False)",
   "s": 10
  },
  {
   "t": "board2 = evaluate({\"logistic\": build(LogisticRegression(max_iter=1000)), \"tree\": tree,",
   "s": 10
  },
  {
   "t": "                   \"forest\": forest, \"xgboost\": boost})",
   "s": 10
  },
  {
   "t": "",
   "s": 10
  },
  {
   "t": "# ⑪ 특성 중요도",
   "s": 11
  },
  {
   "t": "names = forest.named_steps[\"pre\"].get_feature_names_out()",
   "s": 11
  },
  {
   "t": "importance = pd.Series(forest.named_steps[\"model\"].feature_importances_, index=names).sort_values(ascending=False)",
   "s": 11
  },
  {
   "t": "",
   "s": 11
  },
  {
   "t": "# ⑫ 교차검증",
   "s": 12
  },
  {
   "t": "dev = pd.concat([train, valid])  # 마지막 test는 끝까지 아껴 둬요",
   "s": 12
  },
  {
   "t": "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)",
   "s": 12
  },
  {
   "t": "candidates = {\"logistic\": build(LogisticRegression(max_iter=1000)), \"forest\": forest, \"xgboost\": boost}",
   "s": 12
  },
  {
   "t": "cv_scores = {name: cross_validate(m, dev[FEATURES], dev[\"failed\"], cv=cv, scoring=\"f1\")[\"test_score\"]",
   "s": 12
  },
  {
   "t": "             for name, m in candidates.items()}",
   "s": 12
  },
  {
   "t": "",
   "s": 12
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
  8,
  12
 ],
 "outs": {
  "s8g": {
   "title": "⑧의 트리가 고른 첫 질문을 직접 계산하면",
   "lines": [
    [
     "뿌리: train 60건, 실패 비율 0.30 → Gini 1 − (0.70² + 0.30²) = 0.420",
     1
    ],
    [
     "질문: input_chars ≤ 1866.0 ?",
     2,
     "newf"
    ],
    [
     "  예   → 54건, 실패 0.22 → Gini 0.346",
     3
    ],
    [
     "  아니오 → 6건, 실패 1.00 → Gini 0.000",
     3
    ],
    [
     "나눈 뒤 Gini (건수로 가중 평균) = (54×0.346 + 6×0.000) ÷ 60 = 0.311  → 0.109 줄었어요",
     4
    ],
    [
     "같은 질문을 엔트로피로 재면 0.881 → 0.688 (정보 이득 0.194)",
     5
    ]
   ],
   "notes": [
    "실패와 성공이 반반이면 Gini가 가장 크고(0.5), 한쪽만 있으면 0이에요. 30%면 0.42예요.",
    "트리는 모든 열과 모든 자를 자리를 다 시도해 보고, 불순도가 가장 많이 줄어드는 질문 하나를 골라요. 입력 글자가 1866자를 넘는 요청은 train에서 모두 실패였어요.",
    "오른쪽 6건은 모두 실패라 더 나눌 게 없어요(Gini 0, 끝 노드). 왼쪽 54건은 다시 같은 방법으로 다음 질문을 찾아요.",
    "건수가 많은 칸이 더 무겁게 들어가요. 한 칸만 깨끗하고 나머지가 엉망인 질문이 좋은 점수를 받지 않게요.",
    "엔트로피로 재도 같은 방향이에요. 정보 이득은 \"질문 하나로 불확실성이 얼마나 줄었나\"예요."
   ]
  },
  "f_tree": {
   "title": "⑧의 트리를 그대로 그리면",
   "images": [
    [
     "img/ml/tree_depth3.png",
     "색이 진할수록 한쪽 답이 많은 칸"
    ]
   ],
   "notes": [
    "상자마다 질문, Gini, 건수(<code>samples</code>), 답별 건수(<code>value</code> = [성공, 실패])가 있어요. 맨 위가 뿌리, 맨 아래가 끝 노드예요. 끝 노드끼리는 겹치지 않아서, 어떤 요청이든 정확히 하나의 끝 노드에 떨어져요.",
    "질문은 위에서부터 \"가장 많이 줄이는\" 순서예요. 그래서 위쪽에 자주 나오는 열이 판단에 많이 쓰인 열이에요 (⑪의 중요도)."
   ],
   "chart": true
  },
  "f_imp": {
   "title": "불순도를 재는 세 가지 방법",
   "images": [
    [
     "img/ml/impurity_curves.png",
     ""
    ]
   ],
   "notes": [
    "셋 다 p=0.5에서 가장 크고 0과 1에서 0이에요. scikit-learn 기본은 Gini이고 <code>criterion=\"entropy\"</code>로 바꿀 수 있어요. 둘은 거의 같은 트리를 만들어요.",
    "오분류율은 모양이 뾰족해서, 조금 나아진 질문을 잘 구별하지 못해요. 그래서 나눌 질문을 고를 때는 잘 쓰지 않아요."
   ],
   "chart": true
  },
  "t8": {
   "title": "깊이 제한 없는 트리를 ccp_alpha로 가지치기하면",
   "cols": [
    "ccp_alpha",
    "끝 노드 수",
    "깊이",
    "train 정확도",
    "valid F1"
   ],
   "rows": [
    [
     "<b>0</b> (지금)",
     "14",
     7,
     "1.000",
     "0.50"
    ],
    [
     0.02,
     "7",
     6,
     "0.933",
     "0.33"
    ],
    [
     0.03,
     "6",
     5,
     "0.917",
     "0.33"
    ],
    [
     0.05,
     "3",
     2,
     "0.800",
     "0.29"
    ]
   ],
   "notes": [
    "alpha가 커질수록 \"끝 노드 하나 늘리는 값\"이 비싸져서, 덜 쓸모 있는 가지부터 잘려요. 끝 노드 14개가 3개까지 줄었어요.",
    "train 정확도는 1.000에서 내려와요. 외운 걸 덜어내는 거예요. 이 valid 20건에서는 F1도 같이 내려갔지만, 20건이라 한 건 차이로도 흔들려요. alpha도 교차검증으로 골라요 (⑫)."
   ]
  },
  "t9": {
   "title": "트리 개수를 바꾸면 (5-fold 교차검증 F1, ⑫와 같은 나누기)",
   "cols": [
    "트리 수",
    "다섯 번의 점수",
    "평균",
    "표준편차"
   ],
   "rows": [
    [
     1,
     "0.222, 0.444, 0.308, 0.600, 0.250",
     "0.365",
     "<b>0.140</b>"
    ],
    [
     5,
     "0.364, 0.000, 0.250, 0.750, 0.400",
     "0.353",
     "<b>0.243</b>"
    ],
    [
     20,
     "0.286, 0.250, 0.286, 0.571, 0.000",
     "0.279",
     "<b>0.181</b>"
    ],
    [
     "<b>200</b> (지금)",
     "0.333, 0.286, 0.286, 0.333, 0.400",
     "0.328",
     "<b>0.042</b>"
    ]
   ],
   "notes": [
    "트리가 1개면 다섯 번의 점수가 크게 흔들려요(표준편차 0.14). 200개면 0.04로 줄어요. 평균은 비슷한데 <b>흔들림</b>이 줄었어요. 배깅이 하는 일이 이거예요: 편향은 그대로 두고 분산을 줄여요. 5개, 20개에서 오히려 더 흔들린 건, 트리가 적으면 어떤 트리가 뽑혔느냐는 운에 아직 크게 좌우되기 때문이에요.",
    "트리를 늘려서 과적합이 생기지는 않아요. 계산만 늘어요. 보통 수백 개에서 점수가 더 안 변하는 지점을 써요."
   ]
  },
  "f_gb": {
   "title": "부스팅이 잔차를 이어서 고치는 모습 (설명용: 입력 길이 하나로 처리 시간 맞히기)",
   "images": [
    [
     "img/ml/boosting_stages.png",
     "보라 계단: 지금까지 더한 예측, 분홍 선: 남은 잔차"
    ]
   ],
   "notes": [
    "트리 하나는 질문이 하나뿐인 아주 작은 트리(깊이 1)예요. 첫 트리는 \"짧은 요청 대 긴 요청\" 두 칸으로만 나눠요.",
    "다음 트리는 원래 정답이 아니라 <b>남은 잔차</b>(분홍 선)를 맞히도록 배우고, 그 결과에 learning rate 0.3을 곱해 더해요. 그래서 계단이 점점 잘게 쪼개지고, MSE가 줄어요.",
    "50개쯤 가면 계단이 train 점의 잡음까지 따라가기 시작해요. 그래서 부스팅은 트리 수와 learning rate를 교차검증으로 함께 정하고, valid 점수가 나빠지면 멈춰요(조기 종료)."
   ],
   "chart": true
  },
  "t10": {
   "title": "XGBoost의 learning_rate를 바꾸면 (트리 200개, 교차검증 F1)",
   "cols": [
    "learning_rate",
    "다섯 번의 점수",
    "평균"
   ],
   "rows": [
    [
     0.01,
     "0.333, 0.667, 0.444, 0.571, 0.400",
     "0.483"
    ],
    [
     "<b>0.1</b> (지금)",
     "0.444, 0.500, 0.444, 0.667, 0.500",
     "0.511"
    ],
    [
     1.0,
     "0.500, 0.600, 0.444, 0.667, 0.286",
     "0.499"
    ]
   ],
   "notes": [
    "0.01은 200개를 더해도 덜 배운 상태라 평균이 조금 낮아요. 1.0은 트리 하나하나가 잔차를 한 번에 다 고치려 해서, train의 잡음까지 따라가 다시 조금 낮아져요. 0.1이 가장 높았어요. 다만 차이가 0.03 안쪽이라 크지 않아요.",
    "learning rate를 줄이면 트리 수를 늘려야 해요. \"작게, 많이\"가 보통 더 일반화가 잘 돼요 (shrinkage)."
   ]
  },
  "why": {
   "title": "1쪽 ⑦의 valid 점수 vs 같은 모델의 5-fold 교차검증 (F1)",
   "cols": [
    "model",
    "valid 20건 한 번",
    "교차검증 다섯 번",
    "평균"
   ],
   "rows": [
    [
     "logistic",
     "0.333",
     "0.333, 0.444, 0.000, 0.400, 0.500",
     "<b>0.336</b>"
    ],
    [
     "forest",
     "0.400",
     "0.333, 0.286, 0.286, 0.333, 0.400",
     "<b>0.328</b>"
    ],
    [
     "xgboost",
     "0.200",
     "0.444, 0.500, 0.444, 0.667, 0.500",
     "<b>0.511</b>"
    ]
   ],
   "notes": [
    "valid 한 번으로는 forest가 가장 좋아 보였는데, 다섯 번 나눠 재면 xgboost가 가장 좋아요. 그리고 같은 모델도 나누는 방법에 따라 0에서 0.5까지 흔들려요.",
    "데이터가 적을수록 \"한 번 잰 점수\"는 믿기 어려워요. 이 쪽에서는 표 데이터에 강한 트리 모델들과, 점수를 믿을 만하게 재는 교차검증을 다뤄요."
   ]
  },
  "s8": {
   "title": "⑧ 결정 트리의 깊이",
   "cols": [
    "설정",
    "train 정확도",
    "valid F1"
   ],
   "rows": [
    [
     "max_depth=3",
     "0.850",
     "0.333"
    ],
    [
     "제한 없음",
     "1.000",
     "0.500"
    ]
   ],
   "notes": [
    "깊이 제한이 없으면 train을 <b>100%</b> 맞혀요. train 60건을 다 외운 거예요. 외운 모델은 새 데이터에서 무너지기 쉬워서(과적합), 보통 깊이를 제한해요.",
    "이 데이터에서는 깊은 트리의 valid F1이 오히려 높게 나왔는데, valid가 20건이라 우연일 수 있어요. 교차검증(⑫)으로 다시 봐야 해요."
   ]
  },
  "s10": {
   "title": "⑩ 네 모델을 같은 valid로",
   "cols": [
    "model",
    "F1",
    "recall",
    "accuracy"
   ],
   "rows": [
    [
     "logistic",
     "0.333",
     "0.333",
     "0.6"
    ],
    [
     "tree",
     "0.333",
     "0.333",
     "0.6"
    ],
    [
     "forest",
     "0.4",
     "0.333",
     "0.7"
    ],
    [
     "xgboost",
     "0.2",
     "0.167",
     "0.6"
    ]
   ],
   "notes": [
    "valid 한 번의 결과예요. ⑫에서 교차검증으로 다시 비교해요."
   ]
  },
  "s11": {
   "title": "⑪ 랜덤 포레스트의 특성 중요도",
   "cols": [
    "전처리 뒤 열",
    "중요도"
   ],
   "rows": [
    [
     "<code>num__input_chars</code>",
     "0.551"
    ],
    [
     "<code>num__allowed_tools</code>",
     "0.193"
    ],
    [
     "<code>cat__task_type_qa</code>",
     "0.053"
    ],
    [
     "<code>cat__priority_normal</code>",
     "0.052"
    ],
    [
     "<code>cat__task_type_summary</code>",
     "0.043"
    ],
    [
     "<code>cat__priority_low</code>",
     "0.038"
    ],
    [
     "<code>cat__task_type_research</code>",
     "0.036"
    ],
    [
     "<code>cat__priority_high</code>",
     "0.033"
    ]
   ],
   "notes": [
    "<code>input_chars</code>(입력 길이)가 판단에 가장 많이 쓰였어요. 1쪽 ⑥에서 KNN이 이 열만 본 게 오히려 잘 맞은 것과 같은 이야기예요.",
    "중요도는 \"모델이 많이 썼다\"는 뜻이지 \"원인이다\"는 뜻이 아니에요."
   ]
  },
  "s12": {
   "title": "⑫ 5-fold 교차검증 F1 (train+valid 80건)",
   "cols": [
    "model",
    "다섯 번의 점수",
    "평균",
    "표준편차"
   ],
   "rows": [
    [
     "logistic",
     "0.333, 0.444, 0.000, 0.400, 0.500",
     "<b>0.336</b>",
     "0.176"
    ],
    [
     "forest",
     "0.333, 0.286, 0.286, 0.333, 0.400",
     "<b>0.328</b>",
     "0.042"
    ],
    [
     "xgboost",
     "0.444, 0.500, 0.444, 0.667, 0.500",
     "<b>0.511</b>",
     "0.082"
    ]
   ],
   "notes": [
    "80건을 다섯 덩어리로 나눠, 넷으로 배우고 하나로 재기를 다섯 번 해요. 모든 행이 한 번씩 채점에 쓰여요.",
    "평균과 함께 표준편차(흔들림)도 봐요. 평균 차이가 흔들림보다 작으면 \"더 낫다\"고 말하기 어려워요."
   ]
  }
 },
 "flow": [
  [
   8,
   "결정 트리",
   "질문을 이어 가기",
   "깊이 = 과적합"
  ],
  [
   9,
   "랜덤 포레스트",
   "트리 여러 개 투표",
   "흔들림 줄이기"
  ],
  [
   10,
   "XGBoost",
   "틀린 걸 이어서 고치기",
   "부스팅"
  ],
  [
   11,
   "중요도",
   "어느 열을 썼나",
   "원인은 아님"
  ],
  [
   12,
   "교차검증",
   "다섯 번 재기",
   "평균과 흔들림"
  ]
 ],
 "env": "Python 3.13.16, scikit-learn 1.8.0, xgboost 3.1.3, pandas 2.2.3"
};
