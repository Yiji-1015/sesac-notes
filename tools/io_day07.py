"""Day7 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다.

강의 데이터(C:/practice_assignment/20261027/02_응용/data)는 임시 폴더에 복사해서 읽기만 한다.
결과는 cmds/day07.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_day07.py
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error, precision_score, r2_score,
                             recall_score, root_mean_squared_error)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/practice_assignment/20261027/02_응용/data")
TMP = Path(tempfile.mkdtemp())
shutil.copytree(SRC, TMP / "data")

raw = pd.read_csv(TMP / "data" / "requests.csv").set_index("request_id")
man = pd.read_csv(TMP / "data" / "split_manifest.csv")
ids = {s: man.loc[man["split"] == s].sort_values("position")["request_id"] for s in ["train", "valid"]}
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
NUM, CAT = FEATURES[:2], FEATURES[2:]
train, valid = raw.loc[ids["train"]], raw.loc[ids["valid"]]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
yr_train, yr_valid = train["elapsed_ms"], valid["elapsed_ms"]
DATA = "강의 requests.csv · train 60 / valid 20"


def build(estimator, scale="none"):
    """Day6의 build — 트리 계열이라 기본은 스케일링 없음."""
    scaler = StandardScaler() if scale == "standard" else "passthrough"
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", num, NUM), ("cat", cat, CAT)])
    return Pipeline([("pre", pre), ("model", clone(estimator))])


def require_features(df):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")
    return df[FEATURES]


def lst(a, nd=3):
    return "[" + ", ".join(f"{x:.{nd}f}" for x in np.asarray(a, float)) + "]"


def cls_summary(model):
    pv = model.predict(X_valid)
    tn, fp, fn, tp = confusion_matrix(y_valid, pv, labels=[0, 1]).ravel()
    return (f1_score(y_train, model.predict(X_train), zero_division=0),
            f1_score(y_valid, pv, zero_division=0), precision_score(y_valid, pv, zero_division=0),
            recall_score(y_valid, pv, zero_division=0), fp, fn)


def reg_summary(model):
    pv = model.predict(X_valid)
    return f"MAE {mean_absolute_error(yr_valid, pv):.2f} · RMSE {root_mean_squared_error(yr_valid, pv):.2f} · R² {r2_score(yr_valid, pv):.3f}"


IO = {}

# 1. Decision Tree — 노트의 6점
X = np.array([[100], [200], [300], [700], [800], [900]])
tree = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, [0, 0, 0, 1, 1, 1])
reg = DecisionTreeRegressor(max_depth=1, random_state=42).fit(X, [100, 120, 140, 700, 720, 740])
deep = DecisionTreeClassifier(random_state=42).fit(X, [0, 1, 0, 1, 0, 1])
IO["d7-decisiontreeclassifier"] = ("노트의 x=[100 … 900] 6점", [
    ("tree.tree_.threshold[0]", f"{tree.tree_.threshold[0]:g}", "300과 700의 중간값"),
    ("tree.predict([[250], [850]])", repr(tree.predict([[250], [850]]).tolist()), "≤ 500 → 0, > 500 → 1"),
    ("tree.predict_proba([[250], [850]])", str(tree.predict_proba([[250], [850]]).tolist()), "각 잎이 한 클래스뿐이라 1.0 — 학습 비율일 뿐"),
    ("tree.get_depth(), tree.get_n_leaves()", f"({tree.get_depth()}, {tree.get_n_leaves()})", "질문 1개 · 잎 2개"),
    ("reg.predict([[250], [850], [1200]])", lst(reg.predict([[250], [850], [1200]]), 1), "잎 평균 120 / 720 — 1200도 720 (외삽 안 함)"),
    ("정답이 0·1 번갈아(잡음) + max_depth=None: 깊이 · 잎", f"{deep.get_depth()} · {deep.get_n_leaves()}", "제한이 없으면 6점을 다 외울 때까지 나눔"),
])

# 2. 복잡도 제한
m = build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)).fit(X_train, y_train)
est = m["model"]
tr, va, pp, rr, fp, fn = cls_summary(m)
drows = []
for depth in [1, 2, 3, 5, None]:
    md = build(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, random_state=42)).fit(X_train, y_train)
    a, b, *_ = cls_summary(md)
    drows.append((depth, md["model"].get_n_leaves(), a, b))
IO["d7-복잡도-제한-인자"] = (DATA, [
    ("est.get_depth(), est.get_n_leaves()", f"({est.get_depth()}, {est.get_n_leaves()})", "깊이 3이면 잎 최대 8개지만 실제 4개 — 더 나눌 이득이 없는 노드에선 멈춤"),
    ("train F1 · valid F1 · FP · FN", f"{tr:.3f} · {va:.3f} · {fp} · {fn}", "train과 valid 격차가 큼"),
    ("leaf_=1, depth 1 / 2 / 3", " / ".join(f"잎 {n}: {a:.3f}→{b:.3f}" for _, n, a, b in drows[:3]), "train → valid F1"),
    ("leaf_=1, depth 5 / None", " / ".join(f"잎 {n}: {a:.3f}→{b:.3f}" for _, n, a, b in drows[3:]), "깊을수록 train은 1에 가까워지지만 valid는 따라오지 않음"),
    ("min_samples_leaf=3 각 잎 행 수", repr(sorted(int(est.tree_.n_node_samples[i]) for i in range(est.tree_.node_count) if est.tree_.children_left[i] == -1)), "모든 잎이 3행 이상. 이 자료에선 leaf=1로 풀어도 depth 3 트리가 같게 나와(잎 4 · .710→.333) 이 제약이 실제로 바꾼 건 없음"),
])

# 3. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                            max_features="sqrt", random_state=42, n_jobs=1)
forest = build(rf).fit(X_train, y_train)
tr, va, pp, rr, fp, fn = cls_summary(forest)
pv = forest["model"].predict_proba(forest["pre"].transform(X_valid))[:, 1]
one = forest["model"].estimators_[0]
rfr = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2,
                            max_features=1.0, random_state=42, n_jobs=1)
forest_r = build(rfr).fit(X_train, yr_train)
IO["d7-randomforestclassifier"] = (DATA, [
    ('len(forest["model"].estimators_)', repr(len(forest["model"].estimators_)), "학습된 개별 트리 100개"),
    ('forest["model"].estimators_[0].get_depth()', repr(one.get_depth()), "개별 트리도 max_depth=4 안"),
    ("train F1 · valid F1 · FP · FN", f"{tr:.3f} · {va:.3f} · {fp} · {fn}", "단일 Tree(valid .333)보다 낮음 — 노트 설명대로"),
    ("valid 실패 확률 > 0.5 인 행 수", f"{int((pv > 0.5).sum())} / 20", "평균 확률이 0.5 아래로 깎여 실패를 적게 부름"),
    ("RandomForestRegressor: valid", reg_summary(forest_r), "회귀도 Dummy보다 나쁨"),
])

# 4. 중요도
names = forest["pre"].get_feature_names_out()
mdi = pd.Series(forest["model"].feature_importances_, index=names).sort_values(ascending=False)
perm = permutation_importance(forest, X_valid, y_valid, scoring="f1", n_repeats=20, random_state=42, n_jobs=1)
pi = pd.Series(perm.importances_mean, index=FEATURES)
ps = pd.Series(perm.importances_std, index=FEATURES)
IO["d7-feature-importances"] = (DATA + " · 위 Forest", [
    ("len(names), names[:2]", f"{len(names)}, {names[:2].tolist()}", "원핫으로 4열 → 8열"),
    ("mdi 상위 2개", " · ".join(f"{k.split('__')[1]} {v:.4f}" for k, v in mdi.head(2).items()), "train에서 분할에 많이 쓰인 열"),
    ("mdi.sum()", f"{mdi.sum():.4f}", "MDI는 합이 1로 정규화"),
    ("perm.importances_mean (원본 4열)", " · ".join(f"{k} {v:+.4f}" for k, v in pi.items()), "valid F1 감소량 — 1위가 task_type으로 MDI와 다름"),
    ("perm.importances_std", " · ".join(f"{v:.3f}" for v in ps), "평균만큼 큰 흔들림 — valid 20건이라"),
])

# 5. XGBoost
xgb = XGBClassifier(n_estimators=80, learning_rate=0.05, max_depth=2,
                    min_child_weight=1, subsample=0.8, colsample_bytree=1.0,
                    reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", device="cpu", random_state=42, n_jobs=1)
mx = build(xgb).fit(X_train, y_train)
tr, va, pp, rr, fp, fn = cls_summary(mx)
xgr = XGBRegressor(n_estimators=80, learning_rate=0.05, max_depth=2,
                   min_child_weight=1, subsample=0.8, colsample_bytree=1.0,
                   reg_lambda=1.0, objective="reg:squarederror", eval_metric="rmse",
                   tree_method="hist", device="cpu", random_state=42, n_jobs=1)
mxr = build(xgr).fit(X_train, yr_train)
nb = mx["model"].get_booster().num_boosted_rounds()
px = mx.predict_proba(X_valid)[:, 1]
IO["d7-xgbclassifier"] = (DATA, [
    ('model["model"].get_booster().num_boosted_rounds()', repr(nb), "보정 회차 = n_estimators"),
    ("valid F1 · Precision · Recall · FP · FN", f"{va:.3f} · {pp:.3f} · {rr:.3f} · {fp} · {fn}", "노트·강의 결과(F1 .182 · P .200 · FP 4)와 조금 달라요 — 같은 버전(XGBoost 3.1.3)인데 이 PC에선 valid 1행의 판정이 바뀜. 그래도 Tree(.333)·Forest(.222)보다 낮은 건 같음"),
    ("train F1", f"{tr:.3f}", "강의 결과와 같음 · Forest(.800)보다 조금 낮음"),
    ("predict_proba(X_valid)[:, 1] 범위 · 0.5 초과 행 수", f"{px.min():.3f} ~ {px.max():.3f} · {int((px > 0.5).sum())}건", "실제 실패는 6건인데 실패라 부르는 행이 적음"),
    ("XGBRegressor: valid", reg_summary(mxr), "노트(MAE 609.34 · RMSE 700.02)와 소수점 차이 — 위와 같은 환경 차이. R²는 여전히 음수"),
])

# 6. 교차검증
CLS_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
REG_CV = KFold(n_splits=5, shuffle=True, random_state=42)
cv = cross_validate(build(rf), X_train, y_train, cv=CLS_CV, scoring="f1",
                    return_train_score=True, return_estimator=True, n_jobs=1, error_score="raise")
cvr = cross_validate(build(xgr), X_train, yr_train, cv=REG_CV, scoring="neg_root_mean_squared_error")
yv = y_train.to_numpy()
strat_fail = [int(yv[b].sum()) for _, b in CLS_CV.split(X_train, y_train)]
kf_fail = [int(yv[b].sum()) for _, b in KFold(n_splits=5, shuffle=True, random_state=42).split(X_train)]
cvt = cross_validate(build(DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
                     X_train, y_train, cv=CLS_CV, scoring="f1", return_estimator=True, error_score="raise")
ok = []
for (a, b), mm in zip(CLS_CV.split(X_train, y_train), cvt["estimator"]):
    actual = mm["pre"].named_transformers_["num"]["imputer"].statistics_
    ok.append(bool(np.allclose(actual, X_train.iloc[a][NUM].median())) and not set(a) & set(b))
fold_med = [mm["pre"].named_transformers_["num"]["imputer"].statistics_[0] for mm in cvt["estimator"]]
IO["d7-stratifiedkfold"] = (DATA + " (fold 학습 48 / 평가 12)", [
    ('cv["test_score"]  (Forest F1)', lst(cv["test_score"]), "실패 몇 건으로 0 ~ 0.667까지 출렁"),
    ('mean() · std(ddof=0)', f"{cv['test_score'].mean():.3f} · {cv['test_score'].std(ddof=0):.3f}", "편차가 평균과 비슷할 만큼 큼 — 평균 하나로 요약하면 안 되는 이유"),
    ('cv["train_score"].mean()', f"{cv['train_score'].mean():.3f}", "fold 평가 점수와 큰 격차 → 과적합 신호"),
    ('-cvr["test_score"]  (XGB 회귀 RMSE)', lst(-cvr["test_score"], 1), "부호를 되돌려 ms 단위로"),
    ("평가 fold의 실패 수: Stratified vs KFold", f"{strat_fail} vs {kf_fail}", "Stratified는 3~4건으로 고르게"),
    ("fold별 중앙값(글자수) · 검사 통과", f"{lst(fold_med, 1)} · {all(ok)}", "fold마다 자기 학습 48건의 중앙값"),
])

# 7. 새 범주 · 통계 불변
model = build(rf).fit(X_train, y_train)
imp = model["pre"].named_transformers_["num"]["imputer"]
before = imp.statistics_.copy()
new = X_valid.iloc[:2].copy()
new["task_type"], new["priority"] = "translation", "urgent"
pn = model.predict(require_features(new))
Zn = model["pre"].transform(new)
try:
    require_features(new.drop(columns="priority"))
    miss = "에러 없음"
except ValueError as e:
    miss = f"ValueError: {e}"
try:
    model.predict(new.drop(columns="priority"))
    miss2 = "에러 없음"
except Exception as e:  # noqa: BLE001
    miss2 = f"{type(e).__name__}: {str(e).splitlines()[0][:50]}"
IO["d7-새-범주-통계-불변-검사"] = (DATA + " · valid 앞 2행의 범주를 바꿈", [
    ("model.predict(require_features(new))", repr(pn.tolist()), "처음 보는 범주여도 예측은 나옴"),
    ('model["pre"].transform(new).shape', repr(Zn.shape), "기존 8열 유지"),
    ("원핫 6열 부분 Zn[:, 2:]", str(Zn[:, 2:].astype(int).tolist()), "handle_unknown=\"ignore\" → 범주 칸이 전부 0"),
    ("np.array_equal(before, imp.statistics_)", repr(np.array_equal(before, imp.statistics_)), "예측해도 학습 통계 불변"),
    ('require_features(new.drop(columns="priority"))', miss, "필수 열 누락은 예측 전에 명확한 오류"),
    ('검사 없이 model.predict(…priority 없음)', miss2, "검사가 없어도 여기선 에러 — 다만 메시지가 덜 친절함"),
])

# 8. 저장 · 복원
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "selected_pipeline.joblib"
    joblib.dump(model, path)
    kb = path.stat().st_size / 1024
    restored = joblib.load(path)
    same = np.array_equal(model.predict(X_valid), restored.predict(X_valid))
    same_p = np.allclose(model.predict_proba(X_valid), restored.predict_proba(X_valid))
IO["d7-저장-복원-대조"] = (DATA + " · Forest", [
    ("joblib.dump(model, path)", f"파일 약 {kb:.0f} KB", "트리 100개를 통째로 저장"),
    ("np.array_equal(model.predict(X_valid), restored.predict(X_valid))", repr(same), "예측 20개 일치"),
    ("np.allclose(predict_proba 두 개)", repr(same_p), "확률까지 일치"),
])


# ---------------------------------------------------------------- HTML 넣기
def box(title, rows):
    tr = "\n".join(
        f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
        f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


frag = ROOT / "cmds" / "day07.html"
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
