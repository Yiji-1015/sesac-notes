"""Day8 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261028/02_응용/data)는 임시 폴더에 복사해서 읽기만 한다.
결과는 cmds/day08.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day08.py
"""
import html
import json
import re
import shutil
import sys
import tempfile
import types
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261028/02_응용/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")

raw = pd.read_csv(TMP / "data" / "requests.csv").set_index("request_id")
man = pd.read_csv(TMP / "data" / "split_manifest.csv")
ids = {s: man.loc[man["split"] == s].sort_values("position")["request_id"] for s in ["train", "valid", "test"]}
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
DERIVED = ["log_input_chars", "chars_per_slot"]
train, valid, test = raw.loc[ids["train"]], raw.loc[ids["valid"]], raw.loc[ids["test"]]
X_train, X_valid, X_test = train[FEATURES], valid[FEATURES], test[FEATURES]
y_train, y_valid, y_test = train["failed"], valid["failed"], test["failed"]
yr_train = train["elapsed_ms"]
DATA = "강의 requests.csv · train 60 / valid 20"

# joblib 복원이 되도록 add_features를 '이름 있는 모듈' 함수로 둔다 (강의의 features.py와 같은 역할)
feature_ops = types.ModuleType("feature_ops")
exec("import numpy as np\n"
     "def add_features(X):\n"
     "    out = X.copy()\n"
     "    out['log_input_chars'] = np.log1p(out['input_chars'])\n"
     "    out['chars_per_slot'] = out['input_chars'] / (1.0 + out['allowed_tools'])\n"
     "    return out\n", feature_ops.__dict__)
sys.modules["feature_ops"] = feature_ops
add_features = feature_ops.add_features


def make_pre(engineered=False, scale=True):
    numbers = NUM + DERIVED if engineered else NUM
    numeric = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        numeric.append(("scaler", StandardScaler()))
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", Pipeline(numeric), numbers), ("cat", cat, CAT)])


def build(estimator, derived=False, scale=True):
    steps = [("features", FunctionTransformer(add_features, validate=False))] if derived else []
    return Pipeline(steps + [("pre", make_pre(derived, scale)), ("model", clone(estimator))])


def candidates():
    return {"Logistic": build(LogisticRegression(C=1, max_iter=1000)),
            "Logistic_FE": build(LogisticRegression(C=1, max_iter=1000), derived=True)}


def failure_scores(model, X):
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def check_input(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")
    return df[FEATURES]


def lst(a, nd=3):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in np.asarray(a, float)) + "]"


CLS_FOLDS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
REG_FOLDS = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X_train))
IO = {}

# 1. 혼동행렬과 네 지표 — Logistic C=1의 valid 예측
logi = build(LogisticRegression(C=1, max_iter=1000)).fit(X_train, y_train)
pred = logi.predict(X_valid)
cm = confusion_matrix(y_valid, pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
y0 = np.array([0] * 14 + [1] * 6)
p0 = np.zeros(20, dtype=int)
IO["d8-혼동행렬과-네-지표"] = (DATA + " · Logistic C=1의 valid 예측", [
    ("confusion_matrix(y, pred, labels=[0, 1])", str(cm.tolist()), "행 = 실제, 열 = 예측"),
    ("tn, fp, fn, tp = cm.ravel()", f"{tn}, {fp}, {fn}, {tp}", "이 순서로 풀림"),
    ("accuracy · precision · recall · f1", f"{accuracy_score(y_valid, pred):.3f} · {precision_score(y_valid, pred):.3f} · {recall_score(y_valid, pred):.3f} · {f1_score(y_valid, pred):.3f}",
     "P = 2/(2+4), R = 2/(2+4) — 분모가 달라도 여기선 FP = FN이라 같음"),
    ('f1_score(y, pred, pos_label=0)', f"{f1_score(y_valid, pred, pos_label=0):.3f}", "성공을 양성으로 보면 전혀 다른 숫자"),
    ("전부 0 예측: accuracy · recall · f1", f"{accuracy_score(y0, p0):.2f} · {recall_score(y0, p0, zero_division=0):.1f} · {f1_score(y0, p0, zero_division=0):.1f}", "실패 6건을 하나도 못 찾아도 Accuracy 0.70"),
    ("confusion_matrix([0,0,0], [0,0,0])  (labels 없이)", str(confusion_matrix([0, 0, 0], [0, 0, 0]).tolist()), "1×1 — ravel()로 4개를 못 풂"),
    ("confusion_matrix([0,0,0], [0,0,0], labels=[0, 1])", str(confusion_matrix([0, 0, 0], [0, 0, 0], labels=[0, 1]).tolist()), "labels를 주면 항상 2×2"),
])

# 2. ROC
y = np.array([0, 0, 1, 1])
p = np.array([0.10, 0.40, 0.35, 0.80])
fpr, tpr, thr = roc_curve(y, p)
rows = [("fpr, tpr = roc_curve(y, p)[:2]", f"{lst(fpr, 1)}, {lst(tpr, 1)}", "왼쪽 아래 (0,0)에서 오른쪽 위 (1,1)까지"),
        ("thresholds", "[" + ", ".join("inf" if np.isinf(t) else f"{t:.2f}" for t in thr) + "]", "첫 값 inf = '아무도 양성 아님' 시작점"),
        ("roc_auc_score(y, p)", f"{roc_auc_score(y, p):.2f}", "실패·성공 4쌍 중 3쌍에서 실패 점수가 더 높음 (0.35 < 0.40만 역전)")]
for t in [0.5, 0.3]:
    pr_ = (p > t).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y, pr_, labels=[0, 1]).ravel()
    rows.append((f"t={t}: (p > t) → FP, FN, 비용 fp+5·fn, F1", f"{pr_.tolist()} → {fp_}, {fn_}, {fp_ + 5 * fn_}, {f1_score(y, pr_):.3f}",
                 "기본 선: 0.35인 실패를 놓침" if t == 0.5 else "선을 내리면 놓침 0 · 헛경고 1"))
rows.append(("roc_auc_score(y, p) (임계값과 무관)", f"{roc_auc_score(y, p):.2f}", "순위만 보니 t를 바꿔도 그대로"))
pb = (p > 0.5).astype(int)
rows.append(("roc_auc_score(y, (p > 0.5).astype(int)) · ROC 점 개수", f"{roc_auc_score(y, pb):.2f} · {len(roc_curve(y, pb)[0])}개",
             f"이 4건에선 우연히 같은 0.75(동점을 0.5로 셈)지만, 곡선 점이 {len(fpr)}개 → {len(roc_curve(y, pb)[0])}개로 줄어 순위 정보를 잃음"))


def one_class_auc(yy, ss):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            v = roc_auc_score(yy, ss)
        except ValueError as e:
            return "ValueError: " + str(e).split(".")[0][:50]
    return f"{v} · 경고 {type(wl[0].message).__name__}" if wl else f"{v}"


one = one_class_auc([0, 0, 0], [0.1, 0.2, 0.3])
rows.append(("roc_auc_score([0, 0, 0], [.1, .2, .3])", one, "한 클래스뿐이면 정의 불가 — sklearn 1.8은 에러 대신 nan + 경고를 돌려줘요. 조용히 넘어가지 않게 직접 검사해 None으로 기록"))
IO["d8-roc-curve"] = ("노트의 4건 (실제 [0, 0, 1, 1])", rows)

# 3. cross_validate
pipe = build(LogisticRegression(C=1, max_iter=1000))
scores = cross_validate(pipe, X_train, y_train, cv=CLS_FOLDS, scoring={"f1": "f1", "auc": "roc_auc"},
                        return_train_score=True, error_score="raise", n_jobs=1)
reg = cross_validate(build(LinearRegression()), X_train, yr_train, cv=REG_FOLDS,
                     scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise")
regk = cross_validate(build(KNeighborsRegressor(n_neighbors=15)), X_train, yr_train, cv=REG_FOLDS,
                      scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise")
IO["d8-cross-validate"] = (DATA + " · 5-fold (학습 48 / 평가 12)", [
    ("sorted(k for k in scores if k.startswith(\"test\"))", repr(sorted(k for k in scores if k.startswith("test"))), "scoring 딕셔너리 키 → test_키"),
    ('scores["test_f1"]', lst(scores["test_f1"]), "fold마다 크게 다름"),
    ('mean() · std(ddof=0)', f"{scores['test_f1'].mean():.3f} · {scores['test_f1'].std():.3f}", "fold 점수 범위(.286~.857)가 넓어 평균 하나로는 부족"),
    ('scores["test_auc"].mean() · scores["train_f1"].mean()', f"{scores['test_auc'].mean():.3f} · {scores['train_f1'].mean():.3f}", "AUC는 순위, F1은 0.5 선 기준 — 다른 질문"),
    ('reg["test_score"]', lst(reg["test_score"], 1), "neg_ → 음수로 나옴 (클수록 좋게 통일)"),
    ('rmse = -reg["test_score"]; rmse.mean()', f"{(-reg['test_score']).mean():.2f}", "Linear 평균 RMSE (ms)"),
    ("KNN15: -test_score 평균", f"{(-regk['test_score']).mean():.2f}", "CV에선 KNN15가 Linear보다 작음"),
])

# 4. log1p · FunctionTransformer
pipe_fe = Pipeline([("features", FunctionTransformer(add_features, validate=False)),
                    ("pre", make_pre(engineered=True)),
                    ("model", LogisticRegression(C=1, max_iter=1000))]).fit(X_train, y_train)
Ztr, Zva = pipe_fe[:-1].transform(X_train), pipe_fe[:-1].transform(X_valid)
Xc = X_train.copy()
_ = add_features(Xc)
cmp = {}
for nm in ["Logistic", "Logistic_FE"]:
    pp = candidates()[nm]
    cvs = cross_validate(pp, X_train, y_train, cv=CLS_FOLDS, scoring="f1", n_jobs=1)["test_score"]
    pp.fit(X_train, y_train)
    cmp[nm] = (cvs.mean(), cvs.std(), f1_score(y_valid, pp.predict(X_valid)))
cvf = cross_validate(candidates()["Logistic_FE"], X_train, y_train, cv=CLS_FOLDS, scoring="f1", return_estimator=True)
med = []
ok = True
for (fit_idx, _), fitted in zip(CLS_FOLDS, cvf["estimator"]):
    expected = add_features(X_train.iloc[fit_idx])[NUM + DERIVED].median()
    actual = fitted["pre"].named_transformers_["num"]["imputer"].statistics_
    ok = ok and np.allclose(expected.to_numpy(), actual)
    med.append(actual[0])
with np.errstate(divide="ignore"):
    lg0 = np.log(0.0)
IO["d8-np-log1p"] = (DATA, [
    ("np.log1p(0) · np.log(0)", f"{np.log1p(0.0):g} · {lg0:g}", "글자수 0이어도 log1p는 안전"),
    ("np.log1p(np.nan)", f"{np.log1p(np.nan)}", "결측은 결측으로 남김 → 뒤의 imputer가 채움"),
    ("pipe[:-1].transform(X_train).shape", repr(Ztr.shape), "수치 4(원본 2 + 파생 2) + 원핫 6"),
    ("pipe[:-1].transform(X_valid).shape · NaN/inf 개수", f"{Zva.shape} · {int((~np.isfinite(Zva)).sum())}", "valid도 같은 10열, 모두 유한값"),
    ("add_features(X) 뒤 원본 X 열 수", repr(Xc.shape[1]), "X.copy() 덕분에 원본은 4열 그대로"),
    ("Logistic: CV F1 · std · valid F1", " · ".join(f"{v:.3f}" for v in cmp["Logistic"]), "기존"),
    ("Logistic_FE: CV F1 · std · valid F1", " · ".join(f"{v:.3f}" for v in cmp["Logistic_FE"]), "CV는 조금 오르고 valid는 떨어짐 — 노트 표와 같음"),
    ("fold별 글자수 중앙값 · 검사 통과", f"{lst(med, 0)} · {ok}", "fold마다 자기 학습 48건에서 계산"),
])

# 5. KMeans
Xk = np.array([[1, 1], [1, 2], [2, 1], [8, 8], [8, 9], [9, 8]], dtype=float)
km = KMeans(n_clusters=2, n_init=10, random_state=42)
labels = km.fit_predict(Xk)
cluster_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                         ("cluster", KMeans(n_clusters=3, n_init=10, random_state=42))]).fit(X_train[NUM])
Z = cluster_pipe[:-1].transform(X_train[NUM])
sil = {}
ine = {}
for k in [2, 3, 4]:
    kk = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Z)
    sil[k], ine[k] = silhouette_score(Z, kk.labels_), kk.inertia_
new = pd.DataFrame({"input_chars": [200., 2200.], "allowed_tools": [1., 5.]})
before = cluster_pipe["cluster"].cluster_centers_.copy()
pn = cluster_pipe.predict(new)
centers_raw = cluster_pipe["scaler"].inverse_transform(cluster_pipe["cluster"].cluster_centers_)
raw_sil = silhouette_score(cluster_pipe["imputer"].transform(X_train[NUM]), cluster_pipe["cluster"].labels_)
IO["d8-kmeans"] = ("노트의 6점 + " + DATA + "의 수치 2열", [
    ("km.fit_predict(X)", repr(labels.tolist()), "y 없이 가까운 것끼리"),
    ("km.cluster_centers_ · km.inertia_", f"{np.round(km.cluster_centers_, 3).tolist()} · {km.inertia_:.3f}", "각 묶음의 평균점 · 중심까지 제곱거리 합"),
    ('np.bincount(cluster_pipe["cluster"].labels_)', repr(np.bincount(cluster_pipe["cluster"].labels_).tolist()), "train 60건이 24·19·17로"),
    ("중심을 원래 단위로 (scaler.inverse_transform)", " / ".join(f"({c[0]:.0f}자, 도구 {c[1]:.1f})" for c in centers_raw), "긴 요청·도구 많음 / 긴 요청·도구 적음 / 짧은 요청"),
    ("K=2 · 3 · 4: silhouette", " · ".join(f"{sil[k]:.3f}" for k in sil), "K=3이 가장 높음"),
    ("K=2 · 3 · 4: inertia", " · ".join(f"{ine[k]:.1f}" for k in ine), "K를 늘리면 거의 항상 줄어듦 → inertia만으로 K를 못 고름"),
    ("silhouette를 원래 단위 X로 채점하면", f"{raw_sil:.3f}", "같은 군집인데 점수가 달라짐 — 채점은 Z 공간에서"),
    ("cluster_pipe.predict(new)", repr(pn.tolist()), "200자·도구1 → 짧은 군집, 2200자·도구5 → 긴·도구 많음"),
    ("np.array_equal(before, 중심)", repr(np.array_equal(before, cluster_pipe["cluster"].cluster_centers_)), "predict는 중심을 안 바꿈"),
])

# 6. clone · 최종 평가 · 저장
selected = "Logistic_FE"
protocol = {"selected": selected, "threshold": 0.5, "rule": "score > threshold", "metric": "F1"}
proto = candidates()[selected]
fitted_once = clone(proto).fit(X_train, y_train)
fresh = clone(fitted_once)
try:
    fresh.predict(X_valid)
    nf = "에러 없음"
except NotFittedError:
    nf = "NotFittedError"
X_dev, y_dev = pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid])
final = clone(candidates()[selected]).fit(X_dev, y_dev)
pt = failure_scores(final, X_test)
pdt = (pt > protocol["threshold"]).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pdt, labels=[0, 1]).ravel()
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "final_pipeline.joblib"
    joblib.dump(final, path)
    restored = joblib.load(path)
    same = np.allclose(final.predict_proba(X_test), restored.predict_proba(X_test))
med80 = final["pre"].named_transformers_["num"]["imputer"].statistics_[0]
IO["d8-clone"] = ("강의 requests.csv · 개발 80(train+valid) → test 20 · 선택 모델 Logistic_FE", [
    ('json.dumps({"rule": "점수 > 임계값"})', json.dumps({"rule": "점수 > 임계값"}), "ensure_ascii 기본 True → 한글이 \\uXXXX"),
    ('json.dumps(..., ensure_ascii=False)', json.dumps({"rule": "점수 > 임계값"}, ensure_ascii=False), "사람이 읽을 수 있게"),
    ("clone(학습된 Pipeline).predict(X)", nf, "clone은 설정만 복사 — 학습 상태는 버림"),
    ("X_dev.shape · y_dev.mean()", f"{X_dev.shape} · {y_dev.mean():.2f}", "train 60 + valid 20, 실패 비율 30%"),
    ("개발 80 재학습 후 글자수 중앙값", f"{med80:g}", "train 60 때(1315)와 달라지는 건 정상"),
    ("test: F1 · AUC · TN FP FN TP", f"{f1_score(y_test, pdt):.3f} · {roc_auc_score(y_test, pt):.3f} · {tn} {fp} {fn} {tp}", "실패 6건 중 3건 발견 (Recall 0.5)"),
    ("np.allclose(final.predict_proba, restored.predict_proba)", repr(same), "add_features가 이름 있는 모듈 함수라 복원 성공"),
])

# 7. 새 입력 검사
fitted = final
num = fitted["pre"].named_transformers_["num"]
before = num["imputer"].statistics_.copy()
new_requests = X_valid.iloc[:2].copy()
new_requests.iloc[0, new_requests.columns.get_loc("input_chars")] = np.nan
new_requests["task_type"] = "translation"
checked = check_input(new_requests)
ps = failure_scores(fitted, checked)
Zn = fitted[:-1].transform(checked)
try:
    check_input(new_requests.drop(columns="priority"))
    miss = "에러 없음"
except ValueError as e:
    miss = f"ValueError: {e}"
y1 = np.zeros(5, int)
au_raw = one_class_auc(y1, np.linspace(0.1, 0.5, 5))
auc_record = None if len(np.unique(y1)) < 2 else roc_auc_score(y1, np.linspace(0.1, 0.5, 5))
au = f"{au_raw} → 기록 {auc_record}, 실제 양성 {int(y1.sum())}"
IO["d8-새-입력-검사-새-범주-vs-필수-열-누락"] = ("valid 앞 2행 · 1행 글자수를 결측으로, task_type을 처음 보는 'translation'으로", [
    ("fitted[:-1].transform(checked).shape", repr(Zn.shape), "결측·새 범주가 있어도 10열"),
    ("np.isfinite(Z).all()", repr(bool(np.isfinite(Zn).all())), "NaN/inf 없음 — 결측은 학습 중앙값으로"),
    ("failure_scores(fitted, checked)", lst(ps), "확률 2개, 유한값"),
    ('np.array_equal(before, num["imputer"].statistics_)', repr(np.array_equal(before, num["imputer"].statistics_)), "예측해도 통계 불변"),
    ('check_input(new_requests.drop(columns="priority"))', miss, "필수 열 누락 → 예측 전에 멈춤 (실패해야 정상)"),
    ("roc_auc_score(실패 0건인 y, 점수)", au, "nan을 그대로 평균에 섞지 않고, 클래스 수를 먼저 확인해 None + 양성 수로 기록"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day08.html"
s = frag.read_text(encoding="utf-8")
s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
for cid, (title, rows) in IO.items():
    m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</code></pre>\n', s, re.S)
    assert m, cid
    s = s[:m.end()] + box(title, rows) + "\n" + s[m.end():]
frag.write_text(s, encoding="utf-8")
for cid, (title, rows) in IO.items():
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i}  →  {o}   ({n})")
shutil.rmtree(TMP, ignore_errors=True)
