# 업무 요청 실패 예측: 다른 모델 셋(SVM, 판별분석, PCA), 커널 더 깊이, AdaBoost 더 깊이. 노트 4쪽의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
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
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel   #@8
from sklearn.metrics.pairwise import sigmoid_kernel, laplacian_kernel, euclidean_distances   #@8
from sklearn.model_selection import LeaveOneOut, cross_val_predict            #@9
from sklearn.kernel_approximation import RBFSampler                            #@10
from sklearn.tree import DecisionTreeClassifier                               #@11
from sklearn.ensemble import AdaBoostClassifier                               #@11

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

# ⑧ 커널이 되는 조건: 커널 행렬의 고윳값
Z = Z_train                                                     # ⑥에서 크기를 맞춘 train 60건, 8열
pairs = np.einsum("ni,nj->nij", Z, Z).reshape(len(Z), -1)       # 열 두 개씩 곱한 64열 (x₁x₁, x₁x₂, ..., x₈x₈)
trick_gap = np.abs(pairs @ pairs.T - polynomial_kernel(Z, degree=2, gamma=1, coef0=0)).max()   # 64열의 내적 vs 8열 내적의 제곱
dist = euclidean_distances(Z)                                   # 점 사이 거리 ‖x − z‖
gram = {"linear": linear_kernel(Z), "poly (2차)": polynomial_kernel(Z, degree=2), "rbf": rbf_kernel(Z),
        "laplacian": laplacian_kernel(Z), "sigmoid": sigmoid_kernel(Z), "exp(−‖x−z‖³)": np.exp(-dist ** 3)}
eig = {name: np.linalg.eigvalsh(G) for name, G in gram.items()}  # 대칭 행렬의 고윳값 (작은 것부터)
svm_sig = build(SVC(kernel="sigmoid", C=1.0)).fit(X_train, y_train)   # 커널이 아닌 함수로도 학습은 돼요
sig_scores = score(svm_sig)

# ⑨ 커널 SVM의 답: 서포트 벡터 위의 합
rbf = svm_rbf.named_steps["model"]                              # ③의 RBF SVM (8열, C=1, gamma="scale")
g = 1 / (Z.shape[1] * Z.var())                                  # gamma="scale"이 정한 값: 1 ÷ (열 수 × 분산)
Z_valid = svm_rbf.named_steps["pre"].transform(X_valid)
by_hand = rbf_kernel(Z_valid, rbf.support_vectors_, gamma=g) @ rbf.dual_coef_[0] + rbf.intercept_[0]   # Σ αᵢyᵢ K(xᵢ, x) + b
hand_gap = np.abs(by_hand - svm_rbf.decision_function(X_valid)).max()
alpha = np.abs(rbf.dual_coef_[0])                               # 서포트 벡터마다 α (dual_coef_에는 αᵢ × yᵢ가 들어 있어요)
n_at_C = int((alpha > 1.0 - 1e-8).sum())                        # α = C인 점: 도로 안이나 반대편으로 넘어온 점
loo_pred = cross_val_predict(SVC(kernel="rbf", C=1.0, gamma=g), Z, y_train, cv=LeaveOneOut())   # 한 건씩 빼고 60번 배워서 뺀 건 맞히기
loo_wrong = np.flatnonzero(loo_pred != y_train.to_numpy())

# ⑩ 데이터가 많을 때: 커널을 흉내 내는 열 만들기
rff_gap = {}
for D in (10, 100, 1000, 10000):
    P = RBFSampler(gamma=g, n_components=D, random_state=0).fit(Z).transform(Z)   # 무작위 방향 D개로 만든 cos 열
    diff = np.abs(P @ P.T - rbf_kernel(Z, gamma=g))             # 흉내 낸 내적과 진짜 커널의 차이 (60 × 60칸)
    rff_gap[D] = (diff.mean(), diff.max())
svm_rff = Pipeline([("pre", build(LogisticRegression()).named_steps["pre"]),
                    ("rff", RBFSampler(gamma=g, n_components=1000, random_state=0)),
                    ("model", SVC(kernel="linear", C=1.0))]).fit(X_train, y_train)
dev_X = pd.concat([X_train, X_valid])
same_answer = (svm_rff.predict(dev_X) == svm_rbf.predict(dev_X)).mean()   # ③의 커널 SVM과 같은 답을 낸 비율 (80건)
rff_scores = score(svm_rff)

# ⑪ AdaBoost를 직접: 틀린 행에 무게 주기
ys, ys_valid = np.where(y_train == 1, 1, -1), np.where(y_valid == 1, 1, -1)   # AdaBoost는 정답을 +1(실패), −1(성공)로 써요

def adaboost(X, y, T):
    D = np.full(len(y), 1 / len(y))                              # 처음에는 모든 행이 같은 무게
    stumps, alphas, history = [], [], []
    for _ in range(T):
        h = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=D)   # 무게를 반영한 질문 하나짜리 트리
        p = h.predict(X)
        eps = D[p != y].sum()                                    # 무게로 잰 틀린 비율 ε
        a = 0.5 * np.log((1 - eps) / eps)                        # 이 트리의 발언권 α
        Zt = 2 * np.sqrt(eps * (1 - eps))                        # 무게 합을 다시 1로 맞추는 값
        D = D * np.exp(-a * y * p) / Zt                          # 틀린 행은 e^α배, 맞힌 행은 e^(−α)배
        stumps.append(h); alphas.append(a)
        history.append({"eps": eps, "Z": Zt, "wrong_mass": D[p != y].sum()})
    return stumps, np.array(alphas), history, D

stumps, alphas, history, D_last = adaboost(Z_train, ys, 500)
H_train = np.array([h.predict(Z_train) for h in stumps])         # 트리마다 train 60건의 답 (500 × 60)
H_valid = np.array([h.predict(Z_valid) for h in stumps])
F_train = np.cumsum(alphas[:, None] * H_train, axis=0)           # T번째 줄 = 트리 T개까지의 점수 f(x) = Σ α h(x)
F_valid = np.cumsum(alphas[:, None] * H_valid, axis=0)
sk_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=0).fit(Z_train, ys)
same_as_sk = (sk_ada.predict(Z_valid) == np.sign(F_valid[199])).mean()   # 직접 만든 것과 scikit-learn의 답이 같은 비율
heavy = np.argsort(D_last)[::-1][:5]                             # 500번 뒤 무게가 가장 큰 5건

# ⑫ train 오차가 빨리 줄어드는 이유: 지수 손실
train_err = (np.sign(F_train) != ys).mean(axis=1)                # T마다 train 오차
exp_loss = np.exp(-ys * F_train).mean(axis=1)                    # 지수 손실 e^(−y f(x))의 평균
bound = np.cumprod([h["Z"] for h in history])                    # Z₁ × Z₂ × ... × Z_T
bound_gamma = np.exp(-2 * np.cumsum([(0.5 - h["eps"]) ** 2 for h in history]))   # exp(−2 Σ (½ − ε)²)
first_zero = int(np.argmax(train_err == 0)) + 1                  # train 오차가 처음 0이 된 트리 수
stay_zero = int(np.flatnonzero(train_err > 0)[-1]) + 2            # 그 뒤로 계속 0인 첫 트리 수

# ⑬ 마진: train 오차가 0이 된 뒤에도 바뀌는 것
margins = ys * F_train / np.cumsum(alphas)[:, None]              # L1 마진 y f(x) ÷ Σα. −1에서 1 사이
valid_err = (np.sign(F_valid) != ys_valid).mean(axis=1)
ada_cv = {T: cross_validate(build(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=T, random_state=0)),
                            dev[FEATURES], dev["failed"], cv=cv, scoring="f1")["test_score"] for T in (10, 50, 200, 500)}
