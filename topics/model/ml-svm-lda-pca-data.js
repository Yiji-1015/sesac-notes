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
   "t": "from sklearn.base import clone",
   "s": 1
  },
  {
   "t": "from sklearn.compose import ColumnTransformer",
   "s": 1
  },
  {
   "t": "from sklearn.impute import SimpleImputer",
   "s": 1
  },
  {
   "t": "from sklearn.pipeline import Pipeline",
   "s": 1
  },
  {
   "t": "from sklearn.preprocessing import OneHotEncoder, StandardScaler",
   "s": 1
  },
  {
   "t": "from sklearn.linear_model import LogisticRegression",
   "s": 1
  },
  {
   "t": "from sklearn.metrics import f1_score, recall_score, accuracy_score",
   "s": 1
  },
  {
   "t": "from sklearn.svm import SVC",
   "s": 2
  },
  {
   "t": "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis",
   "s": 4
  },
  {
   "t": "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis",
   "s": 5
  },
  {
   "t": "from sklearn.decomposition import PCA",
   "s": 6
  },
  {
   "t": "from sklearn.model_selection import StratifiedKFold, cross_validate",
   "s": 7
  },
  {
   "t": "from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel",
   "s": 8
  },
  {
   "t": "from sklearn.metrics.pairwise import sigmoid_kernel, laplacian_kernel, euclidean_distances",
   "s": 8
  },
  {
   "t": "from sklearn.model_selection import LeaveOneOut, cross_val_predict",
   "s": 9
  },
  {
   "t": "from sklearn.kernel_approximation import RBFSampler",
   "s": 10
  },
  {
   "t": "from sklearn.tree import DecisionTreeClassifier",
   "s": 11
  },
  {
   "t": "from sklearn.ensemble import AdaBoostClassifier",
   "s": 11
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
   "t": "# ① 데이터와 전처리 준비",
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
   "t": "FEATURES = [\"input_chars\", \"allowed_tools\", \"task_type\", \"priority\"]",
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
   "t": "train, valid = part(\"train\"), part(\"valid\")",
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
   "t": "def build(model, drop=None):  # drop=\"first\": 원-핫에서 종류마다 한 열을 빼서 같은 정보 열을 없애기",
   "s": 1
  },
  {
   "t": "    num = Pipeline([(\"fill\", SimpleImputer(strategy=\"median\")), (\"scale\", StandardScaler())])",
   "s": 1
  },
  {
   "t": "    cat = Pipeline([(\"fill\", SimpleImputer(strategy=\"most_frequent\")),",
   "s": 1
  },
  {
   "t": "                    (\"onehot\", OneHotEncoder(handle_unknown=\"ignore\", drop=drop))])",
   "s": 1
  },
  {
   "t": "    pre = ColumnTransformer([(\"num\", num, NUM), (\"cat\", cat, CAT)])",
   "s": 1
  },
  {
   "t": "    return Pipeline([(\"pre\", pre), (\"model\", clone(model))])",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def score(m):",
   "s": 1
  },
  {
   "t": "    p = m.predict(X_valid)",
   "s": 1
  },
  {
   "t": "    return {\"f1\": round(f1_score(y_valid, p), 3), \"recall\": round(recall_score(y_valid, p), 3),",
   "s": 1
  },
  {
   "t": "            \"accuracy\": round(accuracy_score(y_valid, p), 3)}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 선형 SVM: 가장 넓은 도로 찾기",
   "s": 2
  },
  {
   "t": "svm_lin = build(SVC(kernel=\"linear\", C=1.0)).fit(X_train, y_train)",
   "s": 2
  },
  {
   "t": "w = svm_lin.named_steps[\"model\"].coef_[0]",
   "s": 2
  },
  {
   "t": "margin = 2 / np.linalg.norm(w)  # 도로 폭",
   "s": 2
  },
  {
   "t": "n_sv = svm_lin.named_steps[\"model\"].n_support_  # 종류별 서포트 벡터 수",
   "s": 2
  },
  {
   "t": "lin_scores = score(svm_lin)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 커널 SVM: 곧은 도로로 안 될 때",
   "s": 3
  },
  {
   "t": "svm_rbf = build(SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\")).fit(X_train, y_train)",
   "s": 3
  },
  {
   "t": "rbf_scores = score(svm_rbf)",
   "s": 3
  },
  {
   "t": "svm_bal = build(SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\", class_weight=\"balanced\")).fit(X_train, y_train)",
   "s": 3
  },
  {
   "t": "bal_scores = score(svm_bal)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ LDA: 분포를 가정하고 경계 긋기",
   "s": 4
  },
  {
   "t": "lda = build(LinearDiscriminantAnalysis(), drop=\"first\").fit(X_train, y_train)",
   "s": 4
  },
  {
   "t": "lda_scores = score(lda)",
   "s": 4
  },
  {
   "t": "prior = lda.named_steps[\"model\"].priors_  # train의 종류 비율",
   "s": 4
  },
  {
   "t": "lda_z = lda.transform(X_train)[:, 0]  # 한 줄 위로 내린 판별 점수",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ QDA: 종류마다 다른 모양 허용",
   "s": 5
  },
  {
   "t": "qda = build(QuadraticDiscriminantAnalysis(reg_param=0.1), drop=\"first\").fit(X_train, y_train)",
   "s": 5
  },
  {
   "t": "qda_scores = score(qda)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ PCA: 정답 없이 열 줄이기",
   "s": 6
  },
  {
   "t": "Z_train = build(LogisticRegression()).named_steps[\"pre\"].fit_transform(X_train)",
   "s": 6
  },
  {
   "t": "pca = PCA().fit(Z_train)",
   "s": 6
  },
  {
   "t": "ratio = pca.explained_variance_ratio_  # 주성분마다 담은 분산 비율",
   "s": 6
  },
  {
   "t": "k70 = int(np.searchsorted(np.cumsum(ratio), 0.70) + 1)  # 70%를 넘기는 가장 작은 개수",
   "s": 6
  },
  {
   "t": "pca_logit = Pipeline([(\"pre\", build(LogisticRegression()).named_steps[\"pre\"]),",
   "s": 6
  },
  {
   "t": "                      (\"pca\", PCA(n_components=k70)), (\"model\", LogisticRegression(max_iter=1000))])",
   "s": 6
  },
  {
   "t": "pca_scores = score(pca_logit.fit(X_train, y_train))",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 같은 교차검증으로 비교하기",
   "s": 7
  },
  {
   "t": "dev = pd.concat([train, valid])",
   "s": 7
  },
  {
   "t": "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)  # 2쪽 ⑫와 같은 나누기",
   "s": 7
  },
  {
   "t": "candidates = {\"logistic\": build(LogisticRegression(max_iter=1000)),",
   "s": 7
  },
  {
   "t": "              \"svm 선형\": build(SVC(kernel=\"linear\", C=1.0)),",
   "s": 7
  },
  {
   "t": "              \"svm rbf\": build(SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\")),",
   "s": 7
  },
  {
   "t": "              \"svm rbf 균형\": build(SVC(kernel=\"rbf\", C=1.0, gamma=\"scale\", class_weight=\"balanced\")),",
   "s": 7
  },
  {
   "t": "              \"lda\": build(LinearDiscriminantAnalysis(), drop=\"first\"),",
   "s": 7
  },
  {
   "t": "              \"qda\": build(QuadraticDiscriminantAnalysis(reg_param=0.1), drop=\"first\"),",
   "s": 7
  },
  {
   "t": "              f\"pca{k70}+logistic\": pca_logit}",
   "s": 7
  },
  {
   "t": "cv_scores = {name: cross_validate(m, dev[FEATURES], dev[\"failed\"], cv=cv, scoring=\"f1\")[\"test_score\"]",
   "s": 7
  },
  {
   "t": "             for name, m in candidates.items()}",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 커널이 되는 조건: 커널 행렬의 고윳값",
   "s": 8
  },
  {
   "t": "Z = Z_train  # ⑥에서 크기를 맞춘 train 60건, 8열",
   "s": 8
  },
  {
   "t": "pairs = np.einsum(\"ni,nj->nij\", Z, Z).reshape(len(Z), -1)  # 열 두 개씩 곱한 64열 (x₁x₁, x₁x₂, ..., x₈x₈)",
   "s": 8
  },
  {
   "t": "trick_gap = np.abs(pairs @ pairs.T - polynomial_kernel(Z, degree=2, gamma=1, coef0=0)).max()  # 64열의 내적 vs 8열 내적의 제곱",
   "s": 8
  },
  {
   "t": "dist = euclidean_distances(Z)  # 점 사이 거리 ‖x − z‖",
   "s": 8
  },
  {
   "t": "gram = {\"linear\": linear_kernel(Z), \"poly (2차)\": polynomial_kernel(Z, degree=2), \"rbf\": rbf_kernel(Z),",
   "s": 8
  },
  {
   "t": "        \"laplacian\": laplacian_kernel(Z), \"sigmoid\": sigmoid_kernel(Z), \"exp(−‖x−z‖³)\": np.exp(-dist ** 3)}",
   "s": 8
  },
  {
   "t": "eig = {name: np.linalg.eigvalsh(G) for name, G in gram.items()}  # 대칭 행렬의 고윳값 (작은 것부터)",
   "s": 8
  },
  {
   "t": "svm_sig = build(SVC(kernel=\"sigmoid\", C=1.0)).fit(X_train, y_train)  # 커널이 아닌 함수로도 학습은 돼요",
   "s": 8
  },
  {
   "t": "sig_scores = score(svm_sig)",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "# ⑨ 커널 SVM의 답: 서포트 벡터 위의 합",
   "s": 9
  },
  {
   "t": "rbf = svm_rbf.named_steps[\"model\"]  # ③의 RBF SVM (8열, C=1, gamma=\"scale\")",
   "s": 9
  },
  {
   "t": "g = 1 / (Z.shape[1] * Z.var())  # gamma=\"scale\"이 정한 값: 1 ÷ (열 수 × 분산)",
   "s": 9
  },
  {
   "t": "Z_valid = svm_rbf.named_steps[\"pre\"].transform(X_valid)",
   "s": 9
  },
  {
   "t": "by_hand = rbf_kernel(Z_valid, rbf.support_vectors_, gamma=g) @ rbf.dual_coef_[0] + rbf.intercept_[0]  # Σ αᵢyᵢ K(xᵢ, x) + b",
   "s": 9
  },
  {
   "t": "hand_gap = np.abs(by_hand - svm_rbf.decision_function(X_valid)).max()",
   "s": 9
  },
  {
   "t": "alpha = np.abs(rbf.dual_coef_[0])  # 서포트 벡터마다 α (dual_coef_에는 αᵢ × yᵢ가 들어 있어요)",
   "s": 9
  },
  {
   "t": "n_at_C = int((alpha > 1.0 - 1e-8).sum())  # α = C인 점: 도로 안이나 반대편으로 넘어온 점",
   "s": 9
  },
  {
   "t": "loo_pred = cross_val_predict(SVC(kernel=\"rbf\", C=1.0, gamma=g), Z, y_train, cv=LeaveOneOut())  # 한 건씩 빼고 60번 배워서 뺀 건 맞히기",
   "s": 9
  },
  {
   "t": "loo_wrong = np.flatnonzero(loo_pred != y_train.to_numpy())",
   "s": 9
  },
  {
   "t": "",
   "s": 9
  },
  {
   "t": "# ⑩ 데이터가 많을 때: 커널을 흉내 내는 열 만들기",
   "s": 10
  },
  {
   "t": "rff_gap = {}",
   "s": 10
  },
  {
   "t": "for D in (10, 100, 1000, 10000):",
   "s": 10
  },
  {
   "t": "    P = RBFSampler(gamma=g, n_components=D, random_state=0).fit(Z).transform(Z)  # 무작위 방향 D개로 만든 cos 열",
   "s": 10
  },
  {
   "t": "    diff = np.abs(P @ P.T - rbf_kernel(Z, gamma=g))  # 흉내 낸 내적과 진짜 커널의 차이 (60 × 60칸)",
   "s": 10
  },
  {
   "t": "    rff_gap[D] = (diff.mean(), diff.max())",
   "s": 10
  },
  {
   "t": "svm_rff = Pipeline([(\"pre\", build(LogisticRegression()).named_steps[\"pre\"]),",
   "s": 10
  },
  {
   "t": "                    (\"rff\", RBFSampler(gamma=g, n_components=1000, random_state=0)),",
   "s": 10
  },
  {
   "t": "                    (\"model\", SVC(kernel=\"linear\", C=1.0))]).fit(X_train, y_train)",
   "s": 10
  },
  {
   "t": "dev_X = pd.concat([X_train, X_valid])",
   "s": 10
  },
  {
   "t": "same_answer = (svm_rff.predict(dev_X) == svm_rbf.predict(dev_X)).mean()  # ③의 커널 SVM과 같은 답을 낸 비율 (80건)",
   "s": 10
  },
  {
   "t": "rff_scores = score(svm_rff)",
   "s": 10
  },
  {
   "t": "",
   "s": 10
  },
  {
   "t": "# ⑪ AdaBoost를 직접: 틀린 행에 무게 주기",
   "s": 11
  },
  {
   "t": "ys, ys_valid = np.where(y_train == 1, 1, -1), np.where(y_valid == 1, 1, -1)  # AdaBoost는 정답을 +1(실패), −1(성공)로 써요",
   "s": 11
  },
  {
   "t": "",
   "s": 11
  },
  {
   "t": "def adaboost(X, y, T):",
   "s": 11
  },
  {
   "t": "    D = np.full(len(y), 1 / len(y))  # 처음에는 모든 행이 같은 무게",
   "s": 11
  },
  {
   "t": "    stumps, alphas, history = [], [], []",
   "s": 11
  },
  {
   "t": "    for _ in range(T):",
   "s": 11
  },
  {
   "t": "        h = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=D)  # 무게를 반영한 질문 하나짜리 트리",
   "s": 11
  },
  {
   "t": "        p = h.predict(X)",
   "s": 11
  },
  {
   "t": "        eps = D[p != y].sum()  # 무게로 잰 틀린 비율 ε",
   "s": 11
  },
  {
   "t": "        a = 0.5 * np.log((1 - eps) / eps)  # 이 트리의 발언권 α",
   "s": 11
  },
  {
   "t": "        Zt = 2 * np.sqrt(eps * (1 - eps))  # 무게 합을 다시 1로 맞추는 값",
   "s": 11
  },
  {
   "t": "        D = D * np.exp(-a * y * p) / Zt  # 틀린 행은 e^α배, 맞힌 행은 e^(−α)배",
   "s": 11
  },
  {
   "t": "        stumps.append(h); alphas.append(a)",
   "s": 11
  },
  {
   "t": "        history.append({\"eps\": eps, \"Z\": Zt, \"wrong_mass\": D[p != y].sum()})",
   "s": 11
  },
  {
   "t": "    return stumps, np.array(alphas), history, D",
   "s": 11
  },
  {
   "t": "",
   "s": 11
  },
  {
   "t": "stumps, alphas, history, D_last = adaboost(Z_train, ys, 500)",
   "s": 11
  },
  {
   "t": "H_train = np.array([h.predict(Z_train) for h in stumps])  # 트리마다 train 60건의 답 (500 × 60)",
   "s": 11
  },
  {
   "t": "H_valid = np.array([h.predict(Z_valid) for h in stumps])",
   "s": 11
  },
  {
   "t": "F_train = np.cumsum(alphas[:, None] * H_train, axis=0)  # T번째 줄 = 트리 T개까지의 점수 f(x) = Σ α h(x)",
   "s": 11
  },
  {
   "t": "F_valid = np.cumsum(alphas[:, None] * H_valid, axis=0)",
   "s": 11
  },
  {
   "t": "sk_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, random_state=0).fit(Z_train, ys)",
   "s": 11
  },
  {
   "t": "same_as_sk = (sk_ada.predict(Z_valid) == np.sign(F_valid[199])).mean()  # 직접 만든 것과 scikit-learn의 답이 같은 비율",
   "s": 11
  },
  {
   "t": "heavy = np.argsort(D_last)[::-1][:5]  # 500번 뒤 무게가 가장 큰 5건",
   "s": 11
  },
  {
   "t": "",
   "s": 11
  },
  {
   "t": "# ⑫ train 오차가 빨리 줄어드는 이유: 지수 손실",
   "s": 12
  },
  {
   "t": "train_err = (np.sign(F_train) != ys).mean(axis=1)  # T마다 train 오차",
   "s": 12
  },
  {
   "t": "exp_loss = np.exp(-ys * F_train).mean(axis=1)  # 지수 손실 e^(−y f(x))의 평균",
   "s": 12
  },
  {
   "t": "bound = np.cumprod([h[\"Z\"] for h in history])  # Z₁ × Z₂ × ... × Z_T",
   "s": 12
  },
  {
   "t": "bound_gamma = np.exp(-2 * np.cumsum([(0.5 - h[\"eps\"]) ** 2 for h in history]))  # exp(−2 Σ (½ − ε)²)",
   "s": 12
  },
  {
   "t": "first_zero = int(np.argmax(train_err == 0)) + 1  # train 오차가 처음 0이 된 트리 수",
   "s": 12
  },
  {
   "t": "stay_zero = int(np.flatnonzero(train_err > 0)[-1]) + 2  # 그 뒤로 계속 0인 첫 트리 수",
   "s": 12
  },
  {
   "t": "",
   "s": 12
  },
  {
   "t": "# ⑬ 마진: train 오차가 0이 된 뒤에도 바뀌는 것",
   "s": 13
  },
  {
   "t": "margins = ys * F_train / np.cumsum(alphas)[:, None]  # L1 마진 y f(x) ÷ Σα. −1에서 1 사이",
   "s": 13
  },
  {
   "t": "valid_err = (np.sign(F_valid) != ys_valid).mean(axis=1)",
   "s": 13
  },
  {
   "t": "ada_cv = {T: cross_validate(build(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=T, random_state=0)),",
   "s": 13
  },
  {
   "t": "                            dev[FEATURES], dev[\"failed\"], cv=cv, scoring=\"f1\")[\"test_score\"] for T in (10, 50, 200, 500)}",
   "s": 13
  }
 ],
 "names": {
  "1": "데이터와 전처리 준비",
  "2": "선형 SVM: 가장 넓은 도로 찾기",
  "3": "커널 SVM: 곧은 도로로 안 될 때",
  "4": "LDA: 분포를 가정하고 경계 긋기",
  "5": "QDA: 종류마다 다른 모양 허용",
  "6": "PCA: 정답 없이 열 줄이기",
  "7": "같은 교차검증으로 비교하기",
  "8": "커널이 되는 조건: 커널 행렬의 고윳값",
  "9": "커널 SVM의 답: 서포트 벡터 위의 합",
  "10": "데이터가 많을 때: 커널을 흉내 내는 열 만들기",
  "11": "AdaBoost를 직접: 틀린 행에 무게 주기",
  "12": "train 오차가 빨리 줄어드는 이유: 지수 손실",
  "13": "마진: train 오차가 0이 된 뒤에도 바뀌는 것"
 },
 "range": [
  1,
  13
 ],
 "outs": {
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    "train 60건, valid 20건 (1쪽 ①과 같은 행)",
    [
     "build(model) → 전처리 뒤 8열, build(model, drop=\"first\") → 6열",
     1,
     "newf"
    ]
   ],
   "notes": [
    "<code>drop=\"first\"</code>는 <code>task_type</code>, <code>priority</code>의 원-핫 열을 종류마다 하나씩 빼요. 1쪽 ④에서 본 \"원-핫 세 열을 더하면 늘 1\"인 문제를 없애는 거예요. 열 사이 공분산의 역행렬을 쓰는 ④⑤의 판별분석에 필요해요."
   ]
  },
  "s2": {
   "title": "② 선형 SVM (8열, C=1)",
   "lines": [
    [
     "도로 폭 2 ÷ ‖θ‖ = 1.066",
     1
    ],
    [
     "서포트 벡터: 성공 16개, 실패 15개 (train 60건 중 31개)",
     2,
     "newf"
    ],
    [
     "valid → F1 0.20, recall 0.17, accuracy 0.60",
     3
    ]
   ],
   "notes": [
    "크기를 맞춘 8열 공간에서 도로 폭이에요. 숫자 자체보다 C를 바꿀 때 어떻게 변하는지가 중요해요 (🔁).",
    "60건 중 31개, 절반쯤이 서포트 벡터예요. 두 무리가 많이 겹쳐서, 도로 안이나 반대편에 들어간 점이 많다는 뜻이에요. 깨끗이 나뉘는 데이터라면 서포트 벡터는 몇 개뿐이에요.",
    "1쪽 로지스틱(F1 0.33)보다 낮아요. 둘 다 곧은 경계라 비슷한 한계가 있어요."
   ]
  },
  "t2": {
   "title": "선형 SVM의 C를 바꾸면",
   "cols": [
    "C",
    "도로 폭",
    "서포트 벡터 수",
    "valid F1"
   ],
   "rows": [
    [
     0.1,
     "2.687",
     38,
     "0.00"
    ],
    [
     "<b>1</b> (지금)",
     "1.066",
     31,
     "0.20"
    ],
    [
     10,
     "1.035",
     31,
     "0.20"
    ]
   ],
   "notes": [
    "C가 작으면 도로 안으로 들어오는 점을 너그럽게 봐줘서, 도로가 넓어지고(2.7) 서포트 벡터가 많아져요. 너무 너그러우면 경계를 거의 안 움직여서 모두 \"성공\"이라고 답해요 (F1 0).",
    "C를 키우면 틀린 점 하나하나를 비싸게 쳐서 도로가 좁아져요. 이 데이터에서는 C 1 이상이면 거의 같은 답에 머물러요. 1쪽 로지스틱의 C와 방향이 같아요: 작을수록 규제가 세요."
   ]
  },
  "s3": {
   "title": "③ RBF 커널 SVM (8열, C=1, gamma=\"scale\")",
   "lines": [
    [
     "기본 → F1 0.36, recall 0.33, accuracy 0.65",
     1
    ],
    [
     "class_weight=\"balanced\" → F1 0.40, recall 0.50, accuracy 0.55",
     2,
     "newf"
    ]
   ],
   "notes": [
    "곡선 경계를 허용하니 선형(0.20)보다 F1이 올랐어요.",
    "실패가 30%뿐이라, 그냥 두면 SVM은 성공 쪽 오류를 줄이는 데 더 신경 써요. <code>balanced</code>는 실패 한 건을 틀리는 값을 성공 한 건보다 (70 ÷ 30)배 비싸게 쳐요. 그래서 실패를 더 잡고(recall 0.50) 헛경보도 늘어요(정확도 0.55). 3쪽 ⑬에서 기준을 낮춘 것과 같은 방향이에요."
   ]
  },
  "t3": {
   "title": "RBF 커널의 gamma를 바꾸면 (C=10)",
   "cols": [
    "gamma",
    "train 정확도",
    "서포트 벡터 수",
    "valid F1"
   ],
   "rows": [
    [
     0.01,
     "0.817",
     38,
     "0.00"
    ],
    [
     0.1,
     "0.917",
     34,
     "0.31"
    ],
    [
     1,
     "0.983",
     50,
     "0.43"
    ],
    [
     10,
     "0.983",
     58,
     "0.00"
    ]
   ],
   "notes": [
    "gamma가 작으면 점 하나의 영향이 멀리까지 퍼져서 경계가 아주 완만해요. 0.01은 거의 곧은 선이라 실패를 하나도 맞히지 못해요. \"실패\"라고 답한 2건은 모두 헛경보예요.",
    "gamma가 크면 점 하나의 영향이 자기 바로 옆에만 미쳐서, 점마다 작은 섬을 만들어요. 10은 train을 0.98 맞히지만 valid에서는 실패를 하나도 못 잡아요. 외운 거예요(과적합). 서포트 벡터도 58개로, 거의 모든 점이 경계를 정하는 데 쓰여요.",
    "C와 gamma는 같이 움직이니, 실제로는 둘을 격자로 놓고 교차검증으로 골라요 (<code>GridSearchCV</code>)."
   ]
  },
  "s4": {
   "title": "④ LDA (6열)",
   "lines": [
    [
     "사전 확률 π → 성공 0.70, 실패 0.30",
     1
    ],
    [
     "판별 점수 평균 → 성공 -0.44, 실패 1.04",
     2,
     "newf"
    ],
    [
     "valid → F1 0.31, recall 0.33, accuracy 0.55",
     3
    ]
   ],
   "notes": [
    "따로 정해 주지 않으면 train의 종류 비율을 그대로 써요. 아무 정보 없을 때 \"실패일 확률 30%\"에서 출발한다는 뜻이에요.",
    "6열을 한 줄 위로 내렸을 때 두 무리의 평균이에요. 평균이 떨어져 있긴 하지만, 위 그림처럼 두 무리가 많이 겹쳐요.",
    "로지스틱(0.33)과 비슷해요. 둘 다 곧은 경계를 긋는데, LDA는 정규분포를 가정하고 로지스틱은 가정하지 않는다는 점만 달라요."
   ]
  },
  "s5": {
   "title": "⑤ QDA (6열, reg_param=0.1)",
   "lines": [
    [
     "valid → F1 0.33, recall 0.33, accuracy 0.60",
     1
    ]
   ],
   "notes": [
    "LDA와 거의 같아요. 종류마다 퍼짐 모양을 따로 배워도 나아지지 않았어요. 실패가 18건뿐이라, 6열의 퍼짐 모양(공분산 21개 값)을 믿을 만하게 배우기에는 데이터가 부족해요."
   ]
  },
  "s6": {
   "title": "⑥ PCA (크기 맞춘 8열)",
   "lines": [
    [
     "주성분별 분산 비율 → 0.355, 0.265, 0.115, 0.106, 0.093, 0.067, 0.000, 0.000",
     1
    ],
    [
     "누적 → 35%, 62%, 73%, 84%, 93%, 100%, 100%, 100%",
     2
    ],
    [
     "70%를 넘기는 가장 작은 개수 k70 = 3",
     3,
     "newf"
    ],
    [
     "PCA 3개 → 로지스틱 → valid F1 0.20, recall 0.17",
     4
    ]
   ],
   "notes": [
    "마지막 두 주성분이 담은 분산이 0이에요. 8열 중 서로 다른 정보는 6개뿐이라서예요. 원-핫 세 열의 합이 늘 1이라 남는 방향이 두 개(task_type, priority) 생겨요. 1쪽 ④의 \"독립인 열 7개\"(절편 열을 포함한 9열 기준)와 같은 이야기예요.",
    "주성분 하나하나가 담는 양이 고르게 줄어서, 몇 개로 확 압축되는 데이터는 아니에요.",
    "8열을 3열로 줄였어요.",
    "valid 한 번은 8열 로지스틱(0.33)보다 낮아요. 그런데 ⑦의 교차검증에서는 오히려 더 안정적이에요. 한 번의 점수로 판단하지 않는 이유예요."
   ]
  },
  "s6l": {
   "title": "주성분이 원래 열을 섞은 비율 (가중치, 처음 두 개)",
   "cols": [
    "원래 열",
    "주성분 1",
    "주성분 2"
   ],
   "rows": [
    [
     "<code>input_chars</code>",
     "+0.72",
     "-0.65"
    ],
    [
     "<code>allowed_tools</code>",
     "+0.68",
     "+0.72"
    ],
    [
     "<code>task_type_qa</code>",
     "-0.11",
     "+0.07"
    ],
    [
     "<code>task_type_research</code>",
     "+0.06",
     "-0.18"
    ],
    [
     "<code>task_type_summary</code>",
     "+0.06",
     "+0.10"
    ],
    [
     "<code>priority_high</code>",
     "+0.05",
     "-0.03"
    ],
    [
     "<code>priority_low</code>",
     "-0.01",
     "-0.04"
    ],
    [
     "<code>priority_normal</code>",
     "-0.04",
     "+0.07"
    ]
   ],
   "notes": [
    "주성분은 원래 열에 가중치를 곱해 더한 새 열이에요. 가중치 절댓값이 큰 열이 그 주성분의 \"뜻\"이에요. 주성분 이름은 이 표를 보고 사람이 붙여요."
   ]
  },
  "s7": {
   "title": "⑦ 5-fold 교차검증 F1 (train+valid 80건, 2쪽 ⑫와 같은 나누기)",
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
     "0.336",
     "0.176"
    ],
    [
     "svm 선형",
     "0.333, 0.444, 0.000, 0.400, 0.286",
     "0.293",
     "0.156"
    ],
    [
     "svm rbf",
     "0.286, 0.571, 0.000, 0.286, 0.000",
     "0.229",
     "0.214"
    ],
    [
     "<b>svm rbf 균형</b>",
     "0.545, 0.500, 0.444, 0.727, 0.727",
     "<b>0.589</b>",
     "0.117"
    ],
    [
     "lda",
     "0.333, 0.444, 0.000, 0.400, 0.286",
     "0.293",
     "0.156"
    ],
    [
     "qda",
     "0.250, 0.444, 0.000, 0.444, 0.333",
     "0.294",
     "0.164"
    ],
    [
     "pca3+logistic",
     "0.333, 0.400, 0.286, 0.400, 0.500",
     "0.384",
     "0.072"
    ]
   ],
   "notes": [
    "<b>svm rbf 균형</b>이 평균 0.589로 가장 높아요. 2쪽 ⑫의 xgboost(0.511)보다도 높아요. 다만 표준편차는 0.117로 xgboost(0.082)보다 커서, 덩어리마다 점수가 더 오르내려요. 모델 종류보다 \"드문 실패를 얼마나 중요하게 치느냐\"(class_weight)가 더 크게 작용했어요.",
    "선형 SVM과 LDA는 다섯 점수가 완전히 같아요. 둘 다 곧은 경계를 긋고, 이 데이터에서는 같은 쪽으로 판단했어요. 로지스틱도 거의 같아요.",
    "PCA 3개 + 로지스틱은 평균이 8열 로지스틱보다 조금 높고, 표준편차가 가장 작아요. 열을 줄이니 덜 흔들렸어요.",
    "⚠️ 3쪽 ⑰에서 test는 이미 한 번 썼어요. 이 결과를 보고 test로 다시 채점하면 test로 모델을 고르는 셈이라 하지 않아요. 실제 작업이라면 이 후보들을 ⑰ 전에 함께 비교했어야 해요."
   ]
  },
  "f_lin": {
   "title": "숫자 두 열만 쓴 선형 SVM의 도로",
   "images": [
    [
     "img/ml/svm_linear_margin.png",
     "검은 선: 경계, 회색 점선: 도로 양쪽 끝"
    ]
   ],
   "notes": [
    "그림으로 보려고 숫자 두 열만 쓴 모델이에요. ②의 실제 모델(8열)과 다르지만 생김새는 같아요.",
    "동그라미 친 점이 서포트 벡터예요. 도로 끝선 위에 있거나, 도로 안 또는 반대편으로 넘어온 점이에요. <b>이 점들만</b> 경계를 정해요. 도로 바깥 멀리 있는 점은 지워도 경계가 그대로예요.",
    "두 무리가 많이 섞여 있어서 도로 안에 점이 아주 많아요. 이런 데이터에서 SVM은 \"가장 넓은 도로\"보다 \"넘어온 점이 적은 도로\"를 찾는 쪽으로 기울어요 (C)."
   ],
   "chart": true
  },
  "f_rbf": {
   "title": "RBF 커널 SVM의 판단 지도 (숫자 두 열, C=10)",
   "images": [
    [
     "img/ml/svm_rbf_gamma.png",
     "분홍 바탕: \"실패\"라고 답하는 곳"
    ]
   ],
   "notes": [
    "gamma 0.1은 매끈한 곡선 경계, 1은 실패가 모인 곳을 주머니처럼 감싸고, 10은 실패 점마다 작은 섬을 만들어요.",
    "1쪽 KNN의 k와 방향이 반대예요. KNN은 k가 작을수록, RBF는 gamma가 클수록 외워요."
   ],
   "chart": true
  },
  "f_lda": {
   "title": "train 60건의 LDA 판별 점수",
   "images": [
    [
     "img/ml/lda_projection.png",
     ""
    ]
   ],
   "notes": [
    "LDA가 찾은 방향으로 6열을 한 줄 위에 내렸어요. 이 방향은 \"두 평균 차이 ÷ 각 무리의 퍼짐\"이 가장 커지는 방향이에요.",
    "실패(분홍)가 오른쪽에 몰려 있긴 하지만 성공과 넓게 겹쳐요. 겹친 부분은 어떤 경계로도 가를 수 없어서, 이 데이터에서는 어떤 선형 모델도 비슷한 한계에 부딪혀요."
   ],
   "chart": true
  },
  "f_qda": {
   "title": "숫자 두 열로 본 LDA와 QDA의 경계",
   "images": [
    [
     "img/ml/lda_vs_qda.png",
     ""
    ]
   ],
   "notes": [
    "LDA는 두 무리가 같은 모양으로 퍼져 있다고 가정해서 경계가 곧은 직선이에요.",
    "QDA는 실패 무리와 성공 무리의 퍼짐 모양을 따로 배워서, 경계가 휘어요. 대신 배울 값이 많아서 데이터가 적으면 엉뚱하게 휘기 쉬워요."
   ],
   "chart": true
  },
  "f_pca": {
   "title": "PCA: 주성분마다 담은 분산과, 처음 두 주성분 위의 train",
   "images": [
    [
     "img/ml/pca_variance_scatter.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽 막대가 주성분마다 담은 분산 비율, 선이 누적이에요. 3개에서 70%를 넘어요. 7, 8번째는 0이에요.",
    "오른쪽은 8열을 주성분 두 개로 줄여 평면에 그린 거예요. 두 개가 원래 분산의 62%를 담아요. 실패와 성공이 섞여 있는데, PCA는 정답을 보지 않고 \"가장 많이 퍼진 방향\"만 찾기 때문이에요. 정답을 보고 \"가장 잘 갈리는 방향\"을 찾는 건 ④의 LDA예요."
   ],
   "chart": true
  },
  "s8": {
   "title": "⑧ (xᵀz)²를 두 길로 계산하기 (train 60건, 8열)",
   "lines": [
    [
     "64열을 만들어 내적 vs 8열 내적의 제곱 → 가장 큰 차이 1.4e-14",
     1
    ],
    [
     "sigmoid 커널 SVM → valid F1 0.20 (③의 RBF는 0.36)",
     2,
     "newf"
    ]
   ],
   "notes": [
    "두 값의 차이가 컴퓨터 반올림 수준이에요. 열 8개를 열 두 개씩 곱한 64열로 옮겨 내적한 것과, 원래 8열의 내적을 제곱한 것이 같은 계산이에요. 열이 d개면 옮긴 공간은 d²열인데, 커널은 d열 계산 한 번으로 끝나요.",
    "sklearn은 커널이 아닌 함수도 막지 않아서 학습은 끝나요. 다만 아래 표처럼 고윳값이 음수라 \"가장 좋은 답을 찾았다\"는 보장이 없어요. 이 데이터에서는 RBF보다 낮게 나왔어요."
   ]
  },
  "t8": {
   "title": "커널 행렬 60 × 60의 고윳값",
   "cols": [
    "함수",
    "가장 작은 고윳값",
    "음수 고윳값 수",
    "커널인가"
   ],
   "rows": [
    [
     "<code>linear</code>",
     "+0.0000",
     0,
     "커널"
    ],
    [
     "<code>poly (2차)</code>",
     "+0.0000",
     0,
     "커널"
    ],
    [
     "<code>rbf</code>",
     "+0.0000",
     0,
     "커널"
    ],
    [
     "<code>laplacian</code>",
     "+0.0011",
     0,
     "커널"
    ],
    [
     "<code>sigmoid</code>",
     "-0.6025",
     20,
     "<b>아님</b>"
    ],
    [
     "<code>exp(−‖x−z‖³)</code>",
     "-0.1071",
     6,
     "<b>아님</b>"
    ]
   ],
   "notes": [
    "대칭 행렬의 고윳값이 모두 0 이상이면 양의 준정부호(PSD)예요. 어떤 점들을 골라도 커널 행렬이 PSD인 함수가 커널(PDS 커널)이에요. 0에 아주 가까운 음수(−0.000000000001 같은)는 컴퓨터 반올림이라 0으로 적었어요.",
    "sigmoid는 음수 고윳값이 20개예요. 여기서는 tanh(a × xᵀz + b)의 a = 1 ÷ 8, b = 1로 둘 다 양수인데도 그래요. 이름은 커널이지만 엄밀히는 커널이 아니에요.",
    "<code>exp(−‖x−z‖³)</code>는 RBF(<code>exp(−γ‖x−z‖²)</code>)에서 지수만 2를 3으로 바꿨는데 커널이 아니에요(음수 6개). 유클리드 거리의 p제곱은 p가 2 이하일 때만 커널을 만들어요. laplacian(맨해튼 거리를 쓴 exp(−γ × 거리))은 커널이에요."
   ]
  },
  "s9": {
   "title": "⑨ ③의 RBF SVM을 손으로 다시 계산하기",
   "lines": [
    [
     "gamma=\"scale\" → 1 ÷ (8열 × 분산 0.4375) = 0.2857",
     1
    ],
    [
     "Σ αᵢyᵢ K(xᵢ, x) + b vs decision_function (valid 20건) → 가장 큰 차이 3.3e-15",
     2,
     "newf"
    ],
    [
     "서포트 벡터 45개 / 60건 → α = C(넘어온 점) 25개, 0 < α < C(도로 끝선 위) 20개",
     3
    ],
    [
     "leave-one-out → 틀린 건 13개 / 60건 (오류율 0.22), 서포트 벡터 비율 0.75",
     4,
     "newf"
    ]
   ],
   "notes": [
    "크기를 맞춘 숫자 열은 분산이 1이지만, 원-핫 열은 0과 1뿐이라 분산이 작아요. 그래서 전체 분산이 1보다 작고, gamma가 1 ÷ 8보다 커요.",
    "SVM이 배운 건 서포트 벡터 45개와 그 무게(<code>dual_coef_</code>), 절편 하나뿐이에요. 이걸로 새 점과 서포트 벡터 사이의 RBF 값을 무게 붙여 더하니, sklearn의 답과 반올림 수준까지 같아요.",
    "α가 상한 C에 닿은 25개는 도로 안이나 반대편으로 넘어온 점이에요. C는 \"넘어온 점 하나가 경계를 얼마나 세게 당길 수 있나\"의 한도예요. 나머지 20개는 도로 끝선에 딱 닿아 있어요.",
    "틀린 13건은 모두 서포트 벡터예요. 서포트 벡터가 아닌 점은 빼고 다시 배워도 답이 그대로라 반드시 맞히거든요. 그래서 leave-one-out 오류율은 언제나 서포트 벡터 비율 이하예요. 여기서는 0.22 ≤ 0.75라 맞지만 꽤 느슨해요. 두 무리가 많이 섞여 서포트 벡터가 많기 때문이에요."
   ]
  },
  "s10": {
   "title": "⑩ RBF 커널을 무작위 cos 열 D개로 흉내 내기 (train 60 × 60칸)",
   "cols": [
    "D (만든 열 수)",
    "평균 차이",
    "가장 큰 차이"
   ],
   "rows": [
    [
     "10",
     "0.251",
     "0.997"
    ],
    [
     "100",
     "0.066",
     "0.271"
    ],
    [
     "1,000",
     "0.020",
     "0.092"
    ],
    [
     "10,000",
     "0.009",
     "0.042"
    ]
   ],
   "notes": [
    "D를 10배 늘릴 때마다 평균 차이가 3.8배, 3.3배, 2.2배 줄었어요. 무작위로 뽑은 방향의 평균이라, 차이는 대략 1 ÷ √D로 줄어요(10배면 √10 ≈ 3.2배).",
    "D = 1,000 열 위에 선형 SVM을 올리니 train과 valid 80건 중 78건에서 ③의 커널 SVM과 같은 답을 냈어요. 다른 답 2건 중 1건이 valid에 있어서 valid F1이 달라 보여요(③ 0.36, 흉내 낸 쪽 0.20). valid의 실패가 6건뿐이라 답 하나에도 F1이 크게 움직여요.",
    "이 예제는 60건이라 커널 행렬이 3,600칸뿐이지만, 100,000건이면 10,000,000,000칸이라 실수 하나에 8바이트씩 80GB가 필요해요. 흉내 낸 열은 건수 × D칸이라, D가 1,000이면 0.8GB로 줄어요."
   ]
  },
  "s11": {
   "title": "⑪ AdaBoost를 직접 500번 (질문 하나짜리 트리, train 60건)",
   "lines": [
    [
     "1번째 트리 → ε 0.200, α 0.693",
     1
    ],
    [
     "2번째 트리 → ε 0.219, α 0.636 / 3번째 → ε 0.360, α 0.288",
     2
    ],
    [
     "무게를 고친 직후 틀린 행의 무게 합 → 500번 모두 0.500",
     3,
     "newf"
    ],
    [
     "scikit-learn AdaBoost(트리 200개)와 valid 답이 같은 비율 → 1.00",
     4
    ],
    [
     "500번 뒤 무게가 가장 큰 5건 → 38번 0.089, 50번 0.089, 11번 0.088, 26번 0.078, 21번 0.070 (합 0.41, 처음 무게는 1건당 0.017)",
     5,
     "newf"
    ]
   ],
   "notes": [
    "첫 트리는 입력 길이로 나눠 20%를 틀렸어요. α = ½ log((1 − 0.2) ÷ 0.2) = ½ log 4 = 0.693예요. 틀린 비율이 ½에 가까워질수록 α가 0에 가까워져요.",
    "무게를 고치면 틀린 행 쪽 무게의 합이 언제나 정확히 ½이 돼요. 방금 고른 트리는 새 무게로 재면 동전 던지기와 같아서, 다음 트리는 그 트리가 놓친 것을 봐야 점수를 얻어요.",
    "직접 짠 20줄과 scikit-learn이 같은 답을 내요. scikit-learn의 α는 log((1 − ε) ÷ ε)로 2배지만, 모든 α가 같은 배수라 부호(답)는 같아요.",
    "무게가 큰 5건이 전체의 41%를 가져요. 그중 38번, 11번, 26번은 ⑨에서 leave-one-out으로 틀린 점이기도 해요. 두 모델이 똑같이 \"어렵다\"고 본 행이라, 라벨이 잘못됐거나 정말 예외인 요청일 수 있어요."
   ]
  },
  "s12": {
   "title": "⑫ train 오차와 그 상한 (⑪의 500번)",
   "lines": [
    [
     "지수 손실의 평균 vs Z₁ × ... × Z_T → 가장 큰 차이 2.2e-16",
     1
    ],
    [
     "트리 10개 → train 오차 0.183 ≤ ΠZ 0.504 ≤ exp(−2Σ(½−ε)²) 0.549",
     2
    ],
    [
     "train 오차가 처음 0 → 트리 127개 (그때 ΠZ 0.203), 계속 0 → 트리 152개부터",
     3,
     "newf"
    ],
    [
     "트리 500개 → ΠZ 0.033, 마지막 트리의 ε 0.454",
     4
    ]
   ],
   "notes": [
    "지수 손실의 평균과 Z의 곱은 이름만 다른 같은 값이에요. 그리고 train 오차는 언제나 그 아래에 있어요(아래 그림).",
    "상한이 줄어드는 속도는 트리마다 ε이 ½에서 얼마나 먼가로 정해져요. 뒤로 갈수록 ε이 0.45쯤으로 ½에 가까워져서, 상한이 천천히 줄어요. 남은 어려운 행만 보다 보니 질문 하나로는 동전 던지기보다 조금 나은 정도만 맞히는 거예요."
   ]
  },
  "f_bound": {
   "title": "AdaBoost의 train 오차와 두 상한 (⑪의 500번)",
   "images": [
    [
     "img/ml/adaboost_bound.png",
     ""
    ]
   ],
   "notes": [
    "분홍 계단이 실제 train 오차, 보라가 지수 손실의 평균이에요. 계단은 보라 선을 넘은 적이 없어요.",
    "계단은 트리 127개에서 처음 0이 되고, 몇 번 한 건씩 다시 틀리다가 152개부터는 계속 0이에요. 보라 선은 그 뒤에도 계속 내려가요. 틀린 행이 없어진 뒤에도 AdaBoost는 \"맞히긴 했지만 아슬아슬한\" 행을 계속 밀어내요. 이게 ⑬의 마진이에요."
   ],
   "chart": true
  },
  "f_loss": {
   "title": "한 행의 손실을 u = y × f(x)로 그리면",
   "images": [
    [
     "img/ml/margin_losses.png",
     ""
    ]
   ],
   "notes": [
    "넷 다 u가 클수록(확실히 맞힐수록) 작아지고, 0-1 손실과 같거나 그 위에 있어요. 그래서 어느 것을 줄여도 틀린 수의 상한을 줄이는 셈이에요.",
    "왼쪽(크게 틀린 쪽)을 보면 차이가 나요. 지수 손실은 u가 −2면 e² ≈ 7.4로 치솟아요. 로그 손실과 힌지 손실은 곧은 선에 가깝게만 늘어요. 그래서 AdaBoost는 크게 틀린 한 행에 무게를 몰아주고, 라벨이 잘못된 행에 약해요."
   ],
   "chart": true
  },
  "t12": {
   "title": "AdaBoost의 트리를 깊이 1, 2, 3으로 바꾸면 (500번)",
   "cols": [
    "깊이",
    "train 오차가 처음 0",
    "가장 작은 ε",
    "500번 뒤 가장 작은 마진",
    "500번 뒤 valid 오차"
   ],
   "rows": [
    [
     "<b>1</b> (지금)",
     "트리 127개",
     "0.200",
     "0.034",
     "0.40"
    ],
    [
     2,
     "트리 23개",
     "0.200",
     "0.146",
     "0.35"
    ],
    [
     3,
     "트리 6개",
     "0.072",
     "0.356",
     "0.45"
    ]
   ],
   "notes": [
    "트리가 깊을수록 한 번에 더 잘 맞혀서(ε이 작아서) 상한이 빨리 줄고, train 오차가 127개 → 23개 → 6개 만에 0이 돼요.",
    "그런데 valid 오차는 0.35~0.45로, 20건 중 한 건 차이 안에서 오르내릴 뿐이에요. 기본 모델이 강할수록 train에 빨리 맞춰질 뿐, 처음 보는 데이터에 강해지는 건 아니에요. 그래서 AdaBoost는 보통 깊이 1~3의 약한 트리를 쓰고, 트리 수를 교차검증으로 정해요."
   ]
  },
  "s13": {
   "title": "⑬ 트리 수에 따른 L1 마진과 오차 (train 60건, valid 20건)",
   "cols": [
    "트리 수",
    "train 오차",
    "가장 작은 마진",
    "마진의 가운데값",
    "valid 오차"
   ],
   "rows": [
    [
     "<b>10</b>",
     "0.183",
     "-0.260",
     "0.260",
     "0.25"
    ],
    [
     "<b>100</b>",
     "0.017",
     "-0.002",
     "0.152",
     "0.40"
    ],
    [
     "<b>127</b> (train 오차 첫 0)",
     "0.000",
     "+0.001",
     "0.152",
     "0.40"
    ],
    [
     "<b>500</b>",
     "0.000",
     "+0.034",
     "0.115",
     "0.40"
    ]
   ],
   "notes": [
    "가장 작은 마진은 -0.260에서 +0.034로 올라갔어요. train 오차가 0이 된 뒤에도 계속 커져요. 대신 가운데값은 0.260에서 0.115로 줄었어요. AdaBoost는 가장 아슬아슬한 행을 끌어올리느라, 이미 넉넉하던 행의 마진을 조금씩 내줘요.",
    "valid 오차는 오히려 0.25에서 0.40로 나빠졌어요. valid가 20건이라 한 건이 0.05지만, 아래 교차검증도 같은 쪽이에요."
   ]
  },
  "s13cv": {
   "title": "AdaBoost 트리 수별 5-fold 교차검증 F1 (⑦과 같은 나누기)",
   "cols": [
    "트리 수",
    "다섯 번의 점수",
    "평균"
   ],
   "rows": [
    [
     "<b>10</b>",
     "0.333, 0.444, 0.000, 0.667, 0.667",
     "0.422"
    ],
    [
     "<b>50</b>",
     "0.500, 0.545, 0.000, 0.667, 0.500",
     "<b>0.442</b>"
    ],
    [
     "<b>200</b>",
     "0.364, 0.545, 0.000, 0.500, 0.500",
     "0.382"
    ],
    [
     "<b>500</b>",
     "0.444, 0.500, 0.000, 0.500, 0.571",
     "0.403"
    ]
   ],
   "notes": [
    "트리 50개가 가장 좋고(0.442), 그 뒤로는 내려가요. 2쪽 ⑫의 XGBoost는 0.511, ⑦에서 가장 좋았던 svm rbf 균형은 0.589예요.",
    "건수가 60~80건이고 라벨이 깨끗하게 갈리지 않는 데이터라, 마진이 커지는 것보다 어려운 행에 맞추느라 생기는 과적합이 더 커요."
   ]
  },
  "f_margin": {
   "title": "트리를 늘리면 마진이 어떻게 바뀌나",
   "images": [
    [
     "img/ml/adaboost_margins.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽: 선이 0보다 왼쪽에서 시작하면 틀린 행이 있다는 뜻이에요. 트리가 늘수록 선의 왼쪽 끝이 0을 넘어 오른쪽으로 옮겨 가요(가장 작은 마진이 커져요). 대신 선이 왼쪽으로 모여요(넉넉한 행이 줄어요).",
    "오른쪽: 가장 작은 마진(보라)은 오르내리면서도 올라가는데, valid 오차(파랑)는 내려가지 않아요. 마진 이론은 \"마진이 크면 상한이 작다\"는 보장이에요. 이 데이터처럼 마진이 0.03 남짓이면 상한 자체가 아주 커서 아무것도 보장해 주지 않아요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   1,
   "준비",
   "1쪽과 같은 나누기",
   "drop=first"
  ],
  [
   2,
   "선형 SVM",
   "가장 넓은 도로",
   "서포트 벡터, C"
  ],
  [
   3,
   "커널 SVM",
   "곡선 경계",
   "gamma, 균형"
  ],
  [
   4,
   "LDA",
   "분포 가정, 베이즈",
   "한 줄로 사영"
  ],
  [
   5,
   "QDA",
   "종류마다 다른 모양",
   "곡선 경계"
  ],
  [
   6,
   "PCA",
   "정답 없이 열 줄이기",
   "분산 비율"
  ],
  [
   7,
   "비교",
   "같은 교차검증",
   "균형 SVM이 1등"
  ],
  [
   8,
   "커널의 조건",
   "고윳값이 0 이상",
   "sigmoid는 아님"
  ],
  [
   9,
   "커널 SVM의 답",
   "서포트 벡터 위의 합",
   "leave-one-out"
  ],
  [
   10,
   "많은 데이터",
   "무작위 cos 열",
   "RBFSampler"
  ],
  [
   11,
   "AdaBoost 직접",
   "틀린 행에 무게",
   "α와 ε"
  ],
  [
   12,
   "지수 손실",
   "오차의 상한",
   "곱으로 줄어요"
  ],
  [
   13,
   "마진",
   "0이 된 뒤에도",
   "valid는 별개"
  ]
 ],
 "env": "Python 3.13.5, scikit-learn 1.9.0, pandas 2.3.3"
};
