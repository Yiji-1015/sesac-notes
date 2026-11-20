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
  }
 ],
 "names": {
  "1": "데이터와 전처리 준비",
  "2": "선형 SVM: 가장 넓은 도로 찾기",
  "3": "커널 SVM: 곧은 도로로 안 될 때",
  "4": "LDA: 분포를 가정하고 경계 긋기",
  "5": "QDA: 종류마다 다른 모양 허용",
  "6": "PCA: 정답 없이 열 줄이기",
  "7": "같은 교차검증으로 비교하기"
 },
 "range": [
  1,
  7
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
  ]
 ],
 "env": "Python 3.13.16, scikit-learn 1.8.0, pandas 2.2.3"
};
