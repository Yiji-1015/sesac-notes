# 업무 요청 실패 예측 전체 코드. 노트 세 쪽의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
#   1쪽 회귀와 분류: ①~⑦   2쪽 트리와 앙상블, 교차검증: ⑧~⑫   3쪽 평가지표, 특성 설계, 군집: ⑬~⑰
# 데이터: data/requests.csv (요청 100건), data/split_manifest.csv (train 60, valid 20, test 20)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                                        #@1
import numpy as np                                                              #@1
import pandas as pd                                                             #@1
from sklearn.dummy import DummyClassifier, DummyRegressor                       #@2
from sklearn.metrics import accuracy_score, mean_absolute_error                 #@2
from sklearn.base import clone                                                  #@3
from sklearn.compose import ColumnTransformer                                   #@3
from sklearn.impute import SimpleImputer                                        #@3
from sklearn.pipeline import Pipeline                                           #@3
from sklearn.preprocessing import OneHotEncoder, StandardScaler                 #@3
from sklearn.linear_model import LinearRegression                               #@4
from sklearn.metrics import r2_score, root_mean_squared_error                   #@4
from sklearn.linear_model import LogisticRegression                             #@5
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score  #@5
from sklearn.metrics import log_loss                                            #@5
from sklearn.neighbors import KNeighborsClassifier                              #@6
from sklearn.tree import DecisionTreeClassifier                                 #@8
from sklearn.ensemble import RandomForestClassifier                             #@9
from xgboost import XGBClassifier                                               #@10
from sklearn.model_selection import StratifiedKFold, cross_validate             #@12
from sklearn.metrics import roc_auc_score                                       #@14
from sklearn.preprocessing import FunctionTransformer                           #@15
from sklearn.cluster import KMeans                                              #@16
from sklearn.metrics import silhouette_score                                    #@16

DATA = Path(__file__).resolve().parent / "data"                                 #@1

# ① 데이터 읽고 나누기
raw = pd.read_csv(DATA / "requests.csv").set_index("request_id")
manifest = pd.read_csv(DATA / "split_manifest.csv")
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]  # 요청을 받을 때 이미 아는 값만
NUM, CAT = ["input_chars", "allowed_tools"], ["task_type", "priority"]

def part(name):
    ids = manifest[manifest["split"] == name].sort_values("position")["request_id"]
    return raw.loc[ids]

train, valid, test = part("train"), part("valid"), part("test")
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

# ② 기준선 만들기
base_clf = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
base_acc = accuracy_score(y_valid, base_clf.predict(X_valid))
base_reg = DummyRegressor(strategy="mean").fit(X_train, train["elapsed_ms"])
base_mae = mean_absolute_error(valid["elapsed_ms"], base_reg.predict(X_valid))

# ③ 전처리를 Pipeline으로 묶기
def build(model, scale=True):
    num = Pipeline([("fill", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler() if scale else "passthrough")])
    cat = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("pre", pre), ("model", clone(model))])

# ④ 회귀: 처리 시간 예측
reg = build(LinearRegression()).fit(X_train, train["elapsed_ms"])
reg_pred = reg.predict(X_valid)
reg_mae = mean_absolute_error(valid["elapsed_ms"], reg_pred)
reg_r2 = r2_score(valid["elapsed_ms"], reg_pred)
reg_rmse = root_mean_squared_error(valid["elapsed_ms"], reg_pred)
reg_coef = pd.Series(reg.named_steps["model"].coef_,                    # 배운 가중치(열마다 하나)
                     index=reg.named_steps["pre"].get_feature_names_out())

# ⑤ 분류: 실패 여부 예측
logit = build(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
pred = logit.predict(X_valid)
scores = {"accuracy": accuracy_score(y_valid, pred), "precision": precision_score(y_valid, pred, zero_division=0),
          "recall": recall_score(y_valid, pred), "f1": f1_score(y_valid, pred)}
cm = confusion_matrix(y_valid, pred, labels=[0, 1])
z = logit.decision_function(X_valid)        # 가중치를 곱해 더한 점수 (아무 실수)
p_fail = 1 / (1 + np.exp(-z))               # 시그모이드로 0~1 확률로
loss = log_loss(y_valid, p_fail)            # 학습이 줄이려는 비용 (로그 손실)
odds = pd.Series(np.exp(logit.named_steps["model"].coef_[0]),
                 index=logit.named_steps["pre"].get_feature_names_out())  # 열이 1 커질 때 odds가 몇 배

# ⑥ KNN: 거리로 판단하는 모델과 스케일
knn_scaled = build(KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)
knn_raw = build(KNeighborsClassifier(n_neighbors=5), scale=False).fit(X_train, y_train)

# ⑦ 같은 조건으로 비교하기
def evaluate(models):
    rows = []
    for name, m in models.items():
        p = m.fit(X_train, y_train).predict(X_valid)
        rows.append({"model": name, "f1": round(f1_score(y_valid, p), 3),
                     "recall": round(recall_score(y_valid, p), 3), "accuracy": round(accuracy_score(y_valid, p), 3)})
    return pd.DataFrame(rows).set_index("model")

board = evaluate({"dummy": DummyClassifier(strategy="most_frequent"),
                  "logistic": build(LogisticRegression(max_iter=1000)),
                  "knn": build(KNeighborsClassifier(n_neighbors=5))})

# ⑧ 결정 트리
tree = build(DecisionTreeClassifier(max_depth=3, random_state=0), scale=False).fit(X_train, y_train)
deep = build(DecisionTreeClassifier(random_state=0), scale=False).fit(X_train, y_train)
tree_scores = {name: (accuracy_score(y_train, m.predict(X_train)), f1_score(y_valid, m.predict(X_valid)))
               for name, m in {"max_depth=3": tree, "제한 없음": deep}.items()}

# ⑨ 랜덤 포레스트
forest = build(RandomForestClassifier(n_estimators=200, max_depth=4, random_state=0), scale=False)

# ⑩ XGBoost
boost = build(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0), scale=False)
board2 = evaluate({"logistic": build(LogisticRegression(max_iter=1000)), "tree": tree,
                   "forest": forest, "xgboost": boost})

# ⑪ 특성 중요도
names = forest.named_steps["pre"].get_feature_names_out()
importance = pd.Series(forest.named_steps["model"].feature_importances_, index=names).sort_values(ascending=False)

# ⑫ 교차검증
dev = pd.concat([train, valid])  # 마지막 test는 끝까지 아껴 둬요
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
candidates = {"logistic": build(LogisticRegression(max_iter=1000)), "forest": forest, "xgboost": boost}
cv_scores = {name: cross_validate(m, dev[FEATURES], dev["failed"], cv=cv, scoring="f1")["test_score"]
             for name, m in candidates.items()}

# ⑬ 임계값과 오류 비용
proba = logit.predict_proba(X_valid)[:, 1]  # 실패일 확률

def cost_at(t, fn_cost=5, fp_cost=1):  # 놓친 실패(FN) 하나가 헛경보(FP) 다섯 개만큼 비싸다고 가정
    p = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, p, labels=[0, 1]).ravel()
    return {"threshold": t, "fp": int(fp), "fn": int(fn), "cost": int(fp * fp_cost + fn * fn_cost)}

costs = [cost_at(t) for t in (0.2, 0.3, 0.4, 0.5, 0.6)]

# ⑭ ROC-AUC: 임계값과 상관없는 순서 점수
auc = roc_auc_score(y_valid, proba)

# ⑮ 특성 설계: 치우친 값 펴기
def build_fe(model):
    num = Pipeline([("fill", SimpleImputer(strategy="median")),
                    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([("pre", ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])), ("model", clone(model))])

candidates["logistic+log1p"] = build_fe(LogisticRegression(max_iter=1000))
cv_scores["logistic+log1p"] = cross_validate(candidates["logistic+log1p"], dev[FEATURES], dev["failed"],
                                             cv=cv, scoring="f1")["test_score"]

# ⑯ K-Means: 정답 없이 묶어 보기
Z = build(LogisticRegression()).named_steps["pre"].fit_transform(dev[FEATURES])
clusters = {k: KMeans(n_clusters=k, n_init=10, random_state=0).fit(Z) for k in (2, 3, 4, 5)}
silhouettes = {k: silhouette_score(Z, km.labels_) for k, km in clusters.items()}

# ⑰ 마지막 한 번: test로 최종 평가
best = max(cv_scores, key=lambda name: cv_scores[name].mean())  # 교차검증 평균으로 고르기
final = clone(candidates[best]).fit(dev[FEATURES], dev["failed"])
test_pred = final.predict(test[FEATURES])
test_scores = {"f1": f1_score(test["failed"], test_pred), "recall": recall_score(test["failed"], test_pred),
               "accuracy": accuracy_score(test["failed"], test_pred)}
