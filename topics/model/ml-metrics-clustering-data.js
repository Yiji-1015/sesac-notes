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
   "t": "from sklearn.metrics import roc_auc_score",
   "s": 14
  },
  {
   "t": "from sklearn.preprocessing import FunctionTransformer",
   "s": 15
  },
  {
   "t": "from sklearn.cluster import KMeans",
   "s": 16
  },
  {
   "t": "from sklearn.metrics import silhouette_score",
   "s": 16
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
  },
  {
   "t": "# ⑬ 임계값과 오류 비용",
   "s": 13
  },
  {
   "t": "proba = logit.predict_proba(X_valid)[:, 1]  # 실패일 확률",
   "s": 13
  },
  {
   "t": "",
   "s": 13
  },
  {
   "t": "def cost_at(t, fn_cost=5, fp_cost=1):  # 놓친 실패(FN) 하나가 헛경보(FP) 다섯 개만큼 비싸다고 가정",
   "s": 13
  },
  {
   "t": "    p = (proba >= t).astype(int)",
   "s": 13
  },
  {
   "t": "    tn, fp, fn, tp = confusion_matrix(y_valid, p, labels=[0, 1]).ravel()",
   "s": 13
  },
  {
   "t": "    return {\"threshold\": t, \"fp\": int(fp), \"fn\": int(fn), \"cost\": int(fp * fp_cost + fn * fn_cost)}",
   "s": 13
  },
  {
   "t": "",
   "s": 13
  },
  {
   "t": "costs = [cost_at(t) for t in (0.2, 0.3, 0.4, 0.5, 0.6)]",
   "s": 13
  },
  {
   "t": "",
   "s": 13
  },
  {
   "t": "# ⑭ ROC-AUC: 임계값과 상관없는 순서 점수",
   "s": 14
  },
  {
   "t": "auc = roc_auc_score(y_valid, proba)",
   "s": 14
  },
  {
   "t": "",
   "s": 14
  },
  {
   "t": "# ⑮ 특성 설계: 치우친 값 펴기",
   "s": 15
  },
  {
   "t": "def build_fe(model):",
   "s": 15
  },
  {
   "t": "    num = Pipeline([(\"fill\", SimpleImputer(strategy=\"median\")),",
   "s": 15
  },
  {
   "t": "                    (\"log\", FunctionTransformer(np.log1p, feature_names_out=\"one-to-one\")),",
   "s": 15
  },
  {
   "t": "                    (\"scale\", StandardScaler())])",
   "s": 15
  },
  {
   "t": "    cat = Pipeline([(\"fill\", SimpleImputer(strategy=\"most_frequent\")), (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\"))])",
   "s": 15
  },
  {
   "t": "    return Pipeline([(\"pre\", ColumnTransformer([(\"num\", num, NUM), (\"cat\", cat, CAT)])), (\"model\", clone(model))])",
   "s": 15
  },
  {
   "t": "",
   "s": 15
  },
  {
   "t": "candidates[\"logistic+log1p\"] = build_fe(LogisticRegression(max_iter=1000))",
   "s": 15
  },
  {
   "t": "cv_scores[\"logistic+log1p\"] = cross_validate(candidates[\"logistic+log1p\"], dev[FEATURES], dev[\"failed\"],",
   "s": 15
  },
  {
   "t": "                                             cv=cv, scoring=\"f1\")[\"test_score\"]",
   "s": 15
  },
  {
   "t": "",
   "s": 15
  },
  {
   "t": "# ⑯ K-Means: 정답 없이 묶어 보기",
   "s": 16
  },
  {
   "t": "Z = build(LogisticRegression()).named_steps[\"pre\"].fit_transform(dev[FEATURES])",
   "s": 16
  },
  {
   "t": "clusters = {k: KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z) for k in (2, 3, 4, 5)}",
   "s": 16
  },
  {
   "t": "silhouettes = {k: silhouette_score(Z, km.labels_) for k, km in clusters.items()}",
   "s": 16
  },
  {
   "t": "",
   "s": 16
  },
  {
   "t": "# ⑰ 마지막 한 번: test로 최종 평가",
   "s": 17
  },
  {
   "t": "best = max(cv_scores, key=lambda name: cv_scores[name].mean())  # 교차검증 평균으로 고르기",
   "s": 17
  },
  {
   "t": "final = clone(candidates[best]).fit(dev[FEATURES], dev[\"failed\"])",
   "s": 17
  },
  {
   "t": "test_pred = final.predict(test[FEATURES])",
   "s": 17
  },
  {
   "t": "test_scores = {\"f1\": f1_score(test[\"failed\"], test_pred), \"recall\": recall_score(test[\"failed\"], test_pred),",
   "s": 17
  },
  {
   "t": "               \"accuracy\": accuracy_score(test[\"failed\"], test_pred)}",
   "s": 17
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
  13,
  17
 ],
 "outs": {
  "f_roc": {
   "title": "valid 20건으로 그린 ROC 곡선",
   "images": [
    [
     "img/ml/roc_curve.png",
     "연보라 넓이가 AUC"
    ]
   ],
   "notes": [
    "기준을 아주 높게 잡으면 아무것도 \"실패\"라고 안 해서 (0, 0)이고, 0까지 내리면 다 \"실패\"라고 해서 (1, 1)이에요. 그 사이에서 기준을 내릴 때마다 위로(진짜 실패를 하나 더 잡음) 또는 오른쪽으로(헛경보 하나 더) 한 칸씩 가요.",
    "완벽한 모델은 왼쪽 위 (0, 1)을 지나요. 이 모델은 대각선(동전 던지기)에 가깝게 붙어 있어요. 그래서 기준을 어디로 옮겨도 ⑬처럼 비용을 크게 줄이기 어려워요."
   ],
   "chart": true
  },
  "t16": {
   "title": "DBSCAN으로 같은 80건을 묶으면 (min_samples=5)",
   "cols": [
    "eps (이웃 반경)",
    "묶음 수",
    "잡음으로 뺀 건수",
    "묶음별 건수"
   ],
   "rows": [
    [
     0.5,
     0,
     80,
     "-"
    ],
    [
     0.8,
     1,
     75,
     "5"
    ],
    [
     1.0,
     2,
     62,
     "9, 9"
    ],
    [
     1.5,
     1,
     3,
     "77"
    ]
   ],
   "notes": [
    "반경이 작으면 이웃이 5개 이상인 점(핵심점)이 거의 없어서 대부분이 잡음이에요. 반경을 키우면 묶음이 생기다가, 너무 크면 거의 전부가 한 묶음이 돼요.",
    "어느 eps에서도 K-Means처럼 \"고르게 나뉜 몇 묶음\"이 나오지 않아요. 이 데이터에는 밀도가 뚜렷하게 다른 덩어리가 없다는 뜻이고, ⑯의 낮은 실루엣과 같은 이야기예요."
   ]
  },
  "f_elbow": {
   "title": "k를 1부터 8까지 바꿔 본 K-Means",
   "images": [
    [
     "img/ml/kmeans_elbow_silhouette.png",
     "왼쪽: 묶음 안 거리 제곱 합, 오른쪽: 실루엣"
    ]
   ],
   "notes": [
    "SSE는 k를 늘리면 늘 줄어요(k = 80이면 0). 그래서 \"가장 작은 k\"가 아니라 <b>줄어드는 폭이 확 작아지는 곳</b>(팔꿈치)을 봐요. 여기서는 k=3 근처에서 기울기가 완만해져요.",
    "실루엣은 k=3에서 가장 높지만 0.5 기준선에 한참 못 미쳐요. 둘 다 \"3개 정도가 그나마 낫지만 뚜렷한 덩어리는 아니다\"라고 말해요."
   ],
   "chart": true
  },
  "f_km": {
   "title": "k=3 묶음을 숫자 두 열 위에 그리면",
   "images": [
    [
     "img/ml/kmeans_clusters.png",
     "점은 위아래로 살짝 흩어 겹침을 줄였어요"
    ]
   ],
   "notes": [
    "K-Means는 8열 전부로 묶었지만, 그림은 그중 숫자 두 열만 보여 줘요. 그래서 경계가 이 평면에서 깔끔하게 안 보일 수 있어요.",
    "정답(<code>failed</code>)은 묶을 때 쓰지 않았는데, 묶음마다 실패 비율이 달라요. 실패가 0%인 묶음이 있어요. \"이런 요청은 거의 실패하지 않는다\"는 무리를 정답 없이 찾은 셈이에요. 묶음에 이름을 붙일 때는 이렇게 묶음별 평균을 봐요."
   ],
   "chart": true
  },
  "f_den": {
   "title": "같은 80건을 계층적으로 묶은 덴드로그램",
   "images": [
    [
     "img/ml/dendrogram.png",
     "아래 끝 하나하나가 요청 한 건"
    ]
   ],
   "notes": [
    "맨 아래에서는 80건이 각자 한 묶음이에요. 가장 가까운 둘부터 합치고, 합친 높이가 \"얼마나 멀리 있는 걸 합쳤나\"예요. 위로 갈수록 먼 것끼리 합쳐요.",
    "분홍 점선 높이에서 가로로 자르면 3묶음(26, 17, 37건)이 돼요. k를 미리 정하지 않고, 그림을 보고 긴 세로줄(합칠 때 거리가 크게 뛴 곳)을 자르는 게 요령이에요."
   ],
   "chart": true
  },
  "why": {
   "title": "1쪽 로지스틱 모델이 valid 실패 6건에 준 확률",
   "cols": [
    "request_id",
    "실패 확률",
    "0.5 기준 판단"
   ],
   "rows": [
    [
     "R044",
     "0.185",
     "<span class=\"bad\">성공 (놓침)</span>"
    ],
    [
     "R050",
     "0.115",
     "<span class=\"bad\">성공 (놓침)</span>"
    ],
    [
     "R035",
     "0.190",
     "<span class=\"bad\">성공 (놓침)</span>"
    ],
    [
     "R034",
     "0.634",
     "실패"
    ],
    [
     "R001",
     "0.316",
     "<span class=\"bad\">성공 (놓침)</span>"
    ],
    [
     "R033",
     "0.506",
     "실패"
    ]
   ],
   "notes": [
    "모델은 사실 \"실패\" 아니면 \"성공\"이 아니라 <b>확률</b>을 내요. 0.5라는 기준선은 우리가 정한 거예요. 0.1~0.3짜리 실패도 \"성공\"으로 넘어갔어요.",
    "놓친 실패가 큰 손해라면 기준을 낮춰야 해요. 이 쪽에서는 그 기준을 비용으로 정하고, 확률의 품질을 재고, 입력을 다듬고, 정답 없이 데이터를 묶어 보고, 마지막으로 test로 딱 한 번 채점해요."
   ]
  },
  "s13": {
   "title": "⑬ 기준별 오류와 비용 (FN 하나 = 5, FP 하나 = 1)",
   "cols": [
    "기준",
    "헛경보 FP",
    "놓침 FN",
    "비용"
   ],
   "rows": [
    [
     0.2,
     7,
     3,
     22
    ],
    [
     0.3,
     6,
     3,
     "<b>21</b>"
    ],
    [
     0.4,
     5,
     4,
     25
    ],
    [
     0.5,
     4,
     4,
     24
    ],
    [
     0.6,
     3,
     5,
     28
    ]
   ],
   "notes": [
    "기준을 낮추면 헛경보는 늘고 놓침은 줄어요. 비용은 기준 0.3에서 가장 작아요.",
    "비용 5:1은 설명용 가정이에요. 실제로는 \"실패를 놓치면 무슨 일이 생기나\"를 업무 담당자와 정해요. 기준은 valid로 정하고, test로 정하면 안 돼요."
   ]
  },
  "s14": {
   "title": "⑭ ROC-AUC",
   "lines": [
    [
     "roc_auc_score(y_valid, proba) → 0.571",
     1
    ]
   ],
   "notes": [
    "\"실패 하나와 성공 하나를 뽑았을 때, 실패 쪽에 더 높은 확률을 줄 확률\"이에요. 0.5면 동전 던지기, 1이면 완벽. 기준과 상관없이 확률의 순서만 봐요. 0.57이면 이 모델의 확률은 순서도 많이 틀려요."
   ]
  },
  "s15": {
   "title": "⑮ log1p 변환을 넣은 로지스틱의 교차검증 F1",
   "lines": [
    "점수 → [0.333, 0.286, 0.0, 0.222, 0.286]",
    [
     "평균 → 0.225  (변환 없는 logistic 0.336)",
     1,
     "bad"
    ]
   ],
   "notes": [
    "치우친 숫자를 펴 주는 변환인데, 이 데이터에서는 오히려 나빠졌어요. 특성 설계는 \"해 보고 교차검증으로 확인\"이에요. 좋아 보이는 기법도 데이터에 따라 손해일 수 있어요."
   ]
  },
  "s16": {
   "title": "⑯ K-Means: 몇 개로 묶을 때 가장 잘 나뉘나",
   "cols": [
    "묶음 수 k",
    "실루엣 점수"
   ],
   "rows": [
    [
     2,
     "0.223"
    ],
    [
     3,
     "<b>0.240</b>"
    ],
    [
     4,
     "0.208"
    ],
    [
     5,
     "0.191"
    ]
   ],
   "notes": [
    "정답(<code>failed</code>) 없이 입력만 보고 비슷한 요청끼리 묶었어요. 실루엣 점수는 \"내 묶음과는 가깝고 다른 묶음과는 먼가\"(−1~1)예요. k=3일 때 가장 높지만 0.24로, 뚜렷하게 나뉘는 데이터는 아니에요."
   ]
  },
  "s17": {
   "title": "⑰ 최종 평가",
   "lines": [
    [
     "교차검증 평균으로 고른 모델 → xgboost",
     1
    ],
    [
     "test 20건 (처음 사용) → F1 0.545, recall 0.500, accuracy 0.75",
     2
    ]
   ],
   "notes": [
    "⑫와 ⑮의 교차검증 평균이 가장 높은 모델을 골라, train+valid 80건 전부로 다시 배웠어요.",
    "test는 이때 처음 썼어요. 이 숫자가 \"새 요청에서 기대할 성능\"이에요. 이걸 보고 다시 모델을 바꾸면 test도 고르는 데 쓴 셈이라, 더는 공정한 성적이 아니에요."
   ]
  }
 },
 "flow": [
  [
   13,
   "기준과 비용",
   "0.5가 정답 아님",
   "FN, FP 비용"
  ],
  [
   14,
   "ROC-AUC",
   "확률의 순서",
   "기준과 무관"
  ],
  [
   15,
   "특성 설계",
   "log1p",
   "교차검증으로 확인"
  ],
  [
   16,
   "군집",
   "K-Means",
   "정답 없이 묶기"
  ],
  [
   17,
   "최종",
   "test 한 번",
   "고른 뒤 채점"
  ]
 ],
 "env": "Python 3.13.16, scikit-learn 1.8.0, xgboost 3.1.3, pandas 2.2.3"
};
