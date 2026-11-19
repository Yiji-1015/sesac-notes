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
    "깊이 제한이 없으면 train을 <b>100%</b> 맞혀요. 100건 남짓을 다 외운 거예요. 외운 모델은 새 데이터에서 무너지기 쉬워서(과적합), 보통 깊이를 제한해요.",
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
