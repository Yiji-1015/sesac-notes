"""Day6 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261026/02_응용/data)는 임시 폴더에 복사해서 읽기만 한다.
결과는 cmds/day06.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day06.py
"""
import html
import re
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score, r2_score,
                             root_mean_squared_error)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261026/02_응용/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")

raw = pd.read_csv(TMP / "data" / "requests.csv").set_index("request_id")
man = pd.read_csv(TMP / "data" / "split_manifest.csv")
ids = {s: man.loc[man["split"] == s].sort_values("position")["request_id"] for s in ["train", "valid"]}
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
FORBIDDEN = ["failed", "elapsed_ms", "retry_count", "status"]
train, valid = raw.loc[ids["train"]], raw.loc[ids["valid"]]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
yr_train, yr_valid = train["elapsed_ms"], valid["elapsed_ms"]
DATA = "강의 requests.csv · train 60 / valid 20"

SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


def build(estimator, scale="standard"):
    scaler = SCALERS[scale]() if scale in SCALERS else "passthrough"
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("pre", pre), ("model", clone(estimator))])


def f3(v):
    return f"{v:.3f}"


def lst(a, nd=3):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in np.asarray(a, float)) + "]"


def err(fn):
    try:
        fn()
        return "에러 없음"
    except Exception as e:  # noqa: BLE001
        msg = str(e).split("\n")[0].split(", but")[0].rstrip(":, ")
        return f"{type(e).__name__}: {msg[:60]}"


IO = {}

# 1. LinearRegression — 노트의 공부시간 5행
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([52, 61, 69, 80, 89])
lin = LinearRegression().fit(X, y)
IO["d6-linearregression"] = ("노트의 공부시간 예제 5행", [
    ("X.shape", repr(X.shape), "행 5(사례) × 열 1(특성) — 2차원"),
    ("model.coef_", lst(lin.coef_, 2), "1시간당 약 9.3점 — 특성마다 하나라 배열"),
    ("model.intercept_", f"{lin.intercept_:.2f}", "x=0일 때 높이 42.3"),
    ("model.predict([[6]])", lst(lin.predict([[6]]), 2), "9.3 × 6 + 42.3"),
    ("model.predict([6])", err(lambda: lin.predict([6])), "1차원을 넣으면 에러 — [[6]]이어야 함"),
    ('hasattr(LinearRegression(), "coef_")', repr(hasattr(LinearRegression(), "coef_")), "fit 전엔 _ 속성이 없음"),
])

# 2. 회귀 지표
yt, yp = [100, 200, 300], [120, 180, 360]
reg = build(LinearRegression()).fit(X_train, yr_train)
pr = reg.predict(X_valid)
base = build(DummyRegressor(strategy="mean")).fit(X_train, yr_train)
pb = base.predict(X_valid)
IO["d6-회귀-지표"] = ("작은 배열 3개 + " + DATA, [
    ("mean_absolute_error(y_true, y_pred)", f"{mean_absolute_error(yt, yp):.2f}", "(20+20+60)/3"),
    ("root_mean_squared_error(y_true, y_pred)", f"{root_mean_squared_error(yt, yp):.2f}", "큰 오차 60이 제곱돼 MAE보다 큼"),
    ("r2_score(y_true, y_pred)", f"{r2_score(yt, yp):.2f}", "평균선보다 오차 제곱합이 78% 작음"),
    ("r2_score(y_pred, y_true)  (순서 바꿈)", f"{r2_score(yp, yt):.2f}", "R²는 첫 인자의 평균이 기준이라 값이 달라짐"),
    ("Linear: MAE · RMSE · R² (valid)", f"{mean_absolute_error(yr_valid, pr):.1f} · {root_mean_squared_error(yr_valid, pr):.1f} · {r2_score(yr_valid, pr):.3f}", "R²가 음수 — 입력만으론 소요시간 신호가 약함"),
    ("DummyRegressor(mean): 같은 지표", f"{mean_absolute_error(yr_valid, pb):.1f} · {root_mean_squared_error(yr_valid, pb):.1f} · {r2_score(yr_valid, pb):.3f}", "train 평균만 말해도 MAE는 Linear보다 작음"),
    ("base.predict(X_valid)[:3]", lst(pb[:3], 1), f"모두 train 평균 {yr_train.mean():.1f}ms"),
])

# 3. LogisticRegression
clf = build(LogisticRegression(C=1.0, max_iter=1000)).fit(X_train, y_train)
classes = clf.named_steps["model"].classes_
pos = list(classes).index(1)
p = clf.predict_proba(X_valid)[:, pos]
pred = clf.predict(X_valid)
cm = confusion_matrix(y_valid, pred, labels=[0, 1])
clf_small = build(LogisticRegression(C=0.01, max_iter=1000)).fit(X_train, y_train)
IO["d6-logisticregression"] = (DATA, [
    ('clf.named_steps["model"].classes_', repr(classes.tolist()), "predict_proba 열 순서 → 실패(1)는 1번 열"),
    ("clf.predict_proba(X_valid).shape", repr(clf.predict_proba(X_valid).shape), "행마다 [P(0), P(1)], 합은 1"),
    ("p[:4]  (실패 확률)", lst(p[:4]), "확신 정도가 서로 다름"),
    ("clf.predict(X_valid)[:4]", repr(pred[:4].tolist()), "확률을 0.5에서 자른 결과"),
    ("confusion_matrix(y_valid, pred, labels=[0, 1])", str(cm.tolist()), "[[TN FP] [FN TP]] = [[10 4] [4 2]]"),
    ("accuracy_score · f1_score", f"{accuracy_score(y_valid, pred):.3f} · {f1_score(y_valid, pred):.3f}", "Accuracy는 Dummy(0.70)보다도 낮음"),
    ("C=0.01일 때 |계수| 합 vs C=1", f"{np.abs(clf_small['model'].coef_).sum():.3f} vs {np.abs(clf['model'].coef_).sum():.3f}", "C가 작을수록 규제↑ → 계수가 눌림"),
])

# 4. 임계값
truth = np.array([0, 0, 0, 1, 1, 1])
pp = np.array([0.10, 0.40, 0.70, 0.35, 0.60, 0.90])
rows = []
for t in [0.5, 0.3, 0.8]:
    pr_ = (pp >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pr_, labels=[0, 1]).ravel()
    rows.append((f"pred = (p >= {t}).astype(int)", f"{pr_.tolist()} → FP={fp}, FN={fn}, TP={tp}",
                 {0.5: "기본 선", 0.3: "선을 내리면 FN↓ · FP↑", 0.8: "선을 올리면 FP↓ · FN↑"}[t]))
IO["d6-임계값-바꾸기"] = ("노트의 6건 (실제 3:3)", rows)

# 5. KNN
Xk = np.array([[1], [2], [4], [7], [9]])
knn = KNeighborsClassifier(n_neighbors=3).fit(Xk, [0, 0, 1, 1, 1])
d, ix = knn.kneighbors([[3.2]])
rk = KNeighborsRegressor(n_neighbors=3).fit(Xk, [10, 20, 40, 70, 90])
wk = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(Xk, [10, 20, 40, 70, 90])
lin4 = LinearRegression().fit([[1], [2], [3], [4]], [10, 20, 30, 40])
k4 = KNeighborsRegressor(n_neighbors=2).fit([[1], [2], [3], [4]], [10, 20, 30, 40])
IO["d6-kneighborsclassifier"] = ("노트의 x=[1, 2, 4, 7, 9] 예제", [
    ("knn.kneighbors([[3.2]])", f"거리 {lst(d[0], 1)}, 행 {ix[0].tolist()}", "x=4, 2, 1 순서로 가까움"),
    ("knn.predict([[3.2]])", repr(knn.predict([[3.2]]).tolist()), "이웃 정답 1·0·0 → 다수결 0"),
    ("knn.predict_proba([[3.2]])", lst(knn.predict_proba([[3.2]])[0]), "3표 중 0이 2표"),
    ("KNeighborsRegressor(3).predict([[3.2]])", lst(rk.predict([[3.2]]), 2), "(40+20+10)/3"),
    ('weights="distance" 일 때', lst(wk.predict([[3.2]]), 2), "가까운 40의 목소리가 커짐"),
    ("x=8: LinearRegression vs KNN(K=2)", f"{lin4.predict([[8]])[0]:.1f} vs {k4.predict([[8]])[0]:.1f}", "직선은 연장, KNN은 본 정답 범위에 갇힘"),
    ("KNeighborsClassifier(n_neighbors=6).fit(5행).predict", err(lambda: KNeighborsClassifier(n_neighbors=6).fit(Xk, [0, 0, 1, 1, 1]).predict([[3]])), "K는 학습 행 수 이하여야 함"),
])

# 6. ColumnTransformer + Pipeline
m5 = build(KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)
Ztr, Zva = m5["pre"].transform(X_train), m5["pre"].transform(X_valid)
sc = {s: f1_score(y_valid, build(KNeighborsClassifier(n_neighbors=5), scale=s).fit(X_train, y_train).predict(X_valid), zero_division=0)
      for s in ["none", "standard", "minmax"]}
names = m5["pre"].get_feature_names_out().tolist()
IO["d6-columntransformer"] = (DATA, [
    ("X_train.isna().sum().sum()", repr(int(X_train.isna().sum().sum())), "원본 train에 결측이 있음"),
    ('model["pre"].transform(X_train).shape', repr(Ztr.shape), "수치 2 + 원핫 6 = 8열"),
    ('model["pre"].transform(X_valid).shape', repr(Zva.shape), "valid도 같은 8열"),
    ("np.isnan(Z_train).sum()", repr(int(np.isnan(Ztr).sum())), "결측이 train 중앙값·최빈값으로 채워짐"),
    ('get_feature_names_out()[:3]', repr(names[:3]), "num__/cat__ 접두어로 출처 표시"),
    ("Z_train[:, :2].mean(0) · std(0)", f"{lst(Ztr[:, :2].mean(0).round(2) + 0.0, 2)} · {lst(Ztr[:, :2].std(0), 2)}", "standard: train 기준 평균 0 · 표준편차 1"),
    ("K=5 valid F1: none · standard · minmax", f"{sc['none']:.3f} · {sc['standard']:.3f} · {sc['minmax']:.3f}", "표준화했는데 오히려 낮음 — 노트 설명대로"),
])

# 7. 후보 반복 비교
krows = []
for k in [1, 3, 5, 9, 15]:
    mk = build(KNeighborsClassifier(n_neighbors=k)).fit(X_train, y_train)
    tr = f1_score(y_train, mk.predict(X_train), zero_division=0)
    va = f1_score(y_valid, mk.predict(X_valid), zero_division=0)
    krows.append(f"k={k}: {tr:.3f}/{va:.3f}")
cand = {"Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic": LogisticRegression(C=1.0, max_iter=1000),
        "KNN_k3": KNeighborsClassifier(n_neighbors=3)}
cres = {}
for nm, est in cand.items():
    pc = build(est).fit(X_train, y_train).predict(X_valid)
    tn, fp, fn, tp = confusion_matrix(y_valid, pc, labels=[0, 1]).ravel()
    cres[nm] = (accuracy_score(y_valid, pc), f1_score(y_valid, pc, zero_division=0), fp, fn)
with warnings.catch_warnings(record=True) as wlog:
    warnings.simplefilter("always")
    z_val = f1_score(y_valid, np.zeros(20, int))
z_warn = type(wlog[0].message).__name__ if wlog else "없음"
with warnings.catch_warnings(record=True) as plog:
    warnings.simplefilter("always")
    pz = precision_score(y_valid, np.zeros(20, int))
p_warn = type(plog[0].message).__name__ if plog else "없음"
IO["d6-후보-반복-비교-기준선"] = (DATA, [
    ("for k in [1, 3, 5, 9, 15] → train/valid F1", " · ".join(krows[:3]), "K=1은 train 1.000 — 자기 자신이 이웃"),
    ("(계속)", " · ".join(krows[3:]), "K가 크면 실패를 한 번도 안 불러 F1 0"),
    ("Dummy: Accuracy · F1 · FP/FN", f"{cres['Dummy'][0]:.2f} · {cres['Dummy'][1]:.3f} · {cres['Dummy'][2]}/{cres['Dummy'][3]}", "다 성공이라 해도 Accuracy 0.70"),
    ("Logistic: 같은 지표", f"{cres['Logistic'][0]:.2f} · {cres['Logistic'][1]:.3f} · {cres['Logistic'][2]}/{cres['Logistic'][3]}", "헛경고 4 · 놓침 4"),
    ("KNN_k3: 같은 지표", f"{cres['KNN_k3'][0]:.2f} · {cres['KNN_k3'][1]:.3f} · {cres['KNN_k3'][2]}/{cres['KNN_k3'][3]}", "F1 최고지만 실패 3건을 놓침"),
    ("f1_score(y_valid, [0]*20)  (zero_division 없이)", f"{z_val:.1f} · 경고 {z_warn}",
     "F1 = 2TP/(2TP+FP+FN)의 분모는 FN=6이라 0이 아님" + ("" if wlog else " → 이 버전(sklearn 1.8)에선 F1 경고가 안 떠요")),
    ("precision_score(y_valid, [0]*20)", f"{pz:.1f} · 경고 {p_warn}", "Precision 분모(TP+FP)는 진짜 0 → 여기서 경고. zero_division=0은 이 규칙을 명시하는 것"),
])

# 8. 학습 통계 불변
m3 = build(KNeighborsClassifier(n_neighbors=3)).fit(X_train, y_train)
imp = m3["pre"].named_transformers_["num"]["imputer"]
before = imp.statistics_.copy()
alias = imp.statistics_
m3.predict(X_valid)
IO["d6-학습-통계-불변-검사"] = (DATA, [
    ("imp.statistics_", lst(imp.statistics_, 1), f"train 중앙값: 글자수 · 도구 수 (valid 중앙값 {X_valid['input_chars'].median():.1f}과 다름)"),
    ("np.array_equal(before, after)  (predict 후)", repr(np.array_equal(before, imp.statistics_)), "predict는 다시 학습하지 않음"),
    ("before is imp.statistics_ · alias is imp.statistics_", f"{before is imp.statistics_} · {alias is imp.statistics_}", ".copy() 없이 담으면 같은 배열을 가리킴"),
    ("list(X_train.columns) == FEATURES", repr(list(X_train.columns) == FEATURES), "입력 4열 순서까지 일치"),
    ("not set(FORBIDDEN) & set(X_train.columns)", repr(not set(FORBIDDEN) & set(X_train.columns)), "정답·결과 열이 입력에 없음"),
])

# 9. 저장 · 복원
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "selected_pipeline.joblib"
    joblib.dump(m3, path)
    size_kb = path.stat().st_size / 1024
    restored = joblib.load(path)
    same = np.array_equal(m3.predict(X_valid), restored.predict(X_valid))
    rtype = type(restored).__name__
    rsteps = list(restored.named_steps)
IO["d6-저장-복원-후-예측-대조"] = (DATA + " · KNN K=3", [
    ("joblib.dump(model, path)", f"파일 약 {size_kb:.0f} KB", "KNN은 train 60행을 통째로 들고 있음"),
    ("type(restored).__name__, list(restored.named_steps)", f"{rtype}, {rsteps}", "전처리 + 모델이 함께 복원"),
    ("np.array_equal(model.predict(X_valid), restored.predict(X_valid))", repr(same), "같은 환경에서 예측 20개 일치"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day06.html"
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
