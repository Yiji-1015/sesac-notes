# 업무 요청 실패 예측: 다른 모델 셋(SVM, 판별분석, PCA). 노트 4쪽의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 1~3쪽의 requests_model.py와 같은 데이터, 같은 나누기, 같은 교차검증을 써서 바로 비교할 수 있어요.
# 데이터: data/requests.csv (요청 100건), data/split_manifest.csv (train 60, valid 20, test 20)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                                        #@1
import numpy as np                                                              #@1
import pandas as pd                                                             #@1
from sklearn.base import clone                                                  #@1
from sklearn.compose import ColumnTransformer                                   #@1
from sklearn.impute import SimpleImputer                                        #@1
from sklearn.pipeline import Pipeline                                           #@1
from sklearn.preprocessing import OneHotEncoder, StandardScaler                 #@1
from sklearn.linear_model import LogisticRegression                             #@1
from sklearn.metrics import f1_score, recall_score, accuracy_score              #@1
from sklearn.svm import SVC                                                     #@2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis            #@4
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis         #@5
from sklearn.decomposition import PCA                                           #@6
from sklearn.model_selection import StratifiedKFold, cross_validate             #@7

DATA = Path(__file__).resolve().parent / "data"                                 #@1

# ① 데이터와 전처리 준비
raw = pd.read_csv(DATA / "requests.csv").set_index("request_id")
manifest = pd.read_csv(DATA / "split_manifest.csv")
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = ["input_chars", "allowed_tools"], ["task_type", "priority"]

def part(name):
    ids = manifest[manifest["split"] == name].sort_values("position")["request_id"]
    return raw.loc[ids]

train, valid = part("train"), part("valid")
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

def build(model, drop=None):  # drop="first": 원-핫에서 종류마다 한 열을 빼서 같은 정보 열을 없애기
    num = Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop=drop))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("pre", pre), ("model", clone(model))])

def score(m):
    p = m.predict(X_valid)
    return {"f1": round(f1_score(y_valid, p), 3), "recall": round(recall_score(y_valid, p), 3),
            "accuracy": round(accuracy_score(y_valid, p), 3)}

# ② 선형 SVM: 가장 넓은 도로 찾기
svm_lin = build(SVC(kernel="linear", C=1.0)).fit(X_train, y_train)
w = svm_lin.named_steps["model"].coef_[0]
margin = 2 / np.linalg.norm(w)                    # 도로 폭
n_sv = svm_lin.named_steps["model"].n_support_    # 종류별 서포트 벡터 수
lin_scores = score(svm_lin)

# ③ 커널 SVM: 곧은 도로로 안 될 때
svm_rbf = build(SVC(kernel="rbf", C=1.0, gamma="scale")).fit(X_train, y_train)
rbf_scores = score(svm_rbf)
svm_bal = build(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")).fit(X_train, y_train)
bal_scores = score(svm_bal)

# ④ LDA: 분포를 가정하고 경계 긋기
lda = build(LinearDiscriminantAnalysis(), drop="first").fit(X_train, y_train)
lda_scores = score(lda)
prior = lda.named_steps["model"].priors_          # train의 종류 비율
lda_z = lda.transform(X_train)[:, 0]              # 한 줄 위로 내린 판별 점수

# ⑤ QDA: 종류마다 다른 모양 허용
qda = build(QuadraticDiscriminantAnalysis(reg_param=0.1), drop="first").fit(X_train, y_train)
qda_scores = score(qda)

# ⑥ PCA: 정답 없이 열 줄이기
Z_train = build(LogisticRegression()).named_steps["pre"].fit_transform(X_train)
pca = PCA().fit(Z_train)
ratio = pca.explained_variance_ratio_             # 주성분마다 담은 분산 비율
k70 = int(np.searchsorted(np.cumsum(ratio), 0.70) + 1)   # 70%를 넘기는 가장 작은 개수
pca_logit = Pipeline([("pre", build(LogisticRegression()).named_steps["pre"]),
                      ("pca", PCA(n_components=k70)), ("model", LogisticRegression(max_iter=1000))])
pca_scores = score(pca_logit.fit(X_train, y_train))

# ⑦ 같은 교차검증으로 비교하기
dev = pd.concat([train, valid])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)   # 2쪽 ⑫와 같은 나누기
candidates = {"logistic": build(LogisticRegression(max_iter=1000)),
              "svm 선형": build(SVC(kernel="linear", C=1.0)),
              "svm rbf": build(SVC(kernel="rbf", C=1.0, gamma="scale")),
              "svm rbf 균형": build(SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
              "lda": build(LinearDiscriminantAnalysis(), drop="first"),
              "qda": build(QuadraticDiscriminantAnalysis(reg_param=0.1), drop="first"),
              f"pca{k70}+logistic": pca_logit}
cv_scores = {name: cross_validate(m, dev[FEATURES], dev["failed"], cv=cv, scoring="f1")["test_score"]
             for name, m in candidates.items()}
