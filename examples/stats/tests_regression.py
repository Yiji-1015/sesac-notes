# 업무 요청 100건으로 가설 검정과 회귀 분석을 해 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/requests.csv (머신러닝 기초와 같은 가상의 업무 요청 100건). ①⑦은 정답을 아는 가상 실험을 직접 만들어 써요.
# 회귀는 statsmodels 없이 numpy와 scipy로 식을 직접 계산해요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                                        #@1
import numpy as np                                                              #@1
import pandas as pd                                                             #@1
from scipy import stats                                                         #@1

DATA = Path(__file__).resolve().parent / "data"                                 #@1
rng = np.random.default_rng(0)

# ① 검정의 뼈대: 1종 오류와 검정력
def experiment(n, diff, sd=600, alpha=0.05, repeats=4000):      # 두 집단에 진짜 차이 diff가 있을 때, n건씩 재서 차이를 찾는 비율
    found = 0
    for _ in range(repeats):
        a, b = rng.normal(1200, sd, n), rng.normal(1200 + diff, sd, n)
        found += stats.ttest_ind(a, b).pvalue < alpha
    return found / repeats

type1 = {n: experiment(n, diff=0) for n in (10, 30, 100)}      # 차이가 없는데 "있다"고 한 비율 = 1종 오류
power = {(n, d): experiment(n, diff=d) for n in (10, 30, 100) for d in (200, 400)}   # 차이가 있을 때 찾은 비율 = 검정력
need_n = int(np.ceil(2 * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * 600 / 200) ** 2))   # 200 차이를 80%로 찾으려면 집단당 몇 건

# ② 두 집단 비교: 실패한 요청은 더 오래 걸렸나
req = pd.read_csv(DATA / "requests.csv").set_index("request_id")
failed_ms, ok_ms = req.loc[req["failed"] == 1, "elapsed_ms"], req.loc[req["failed"] == 0, "elapsed_ms"]
normal_p = (stats.shapiro(failed_ms).pvalue, stats.shapiro(ok_ms).pvalue)   # 정규성 (Shapiro-Wilk)
levene_p = stats.levene(failed_ms, ok_ms).pvalue                # 등분산 (Levene)
student = stats.ttest_ind(failed_ms, ok_ms)                     # 분산이 같다고 보는 t검정
welch = stats.ttest_ind(failed_ms, ok_ms, equal_var=False)      # 분산이 달라도 되는 t검정 (Welch)
mann = stats.mannwhitneyu(failed_ms, ok_ms)                     # 정규성을 가정하지 않는 순위 검정
pooled = np.sqrt(((len(failed_ms) - 1) * failed_ms.var() + (len(ok_ms) - 1) * ok_ms.var()) / (len(failed_ms) + len(ok_ms) - 2))
cohen_d = (failed_ms.mean() - ok_ms.mean()) / pooled            # 효과 크기: 차이가 표준편차의 몇 배인가

# ③ 셋 이상 비교: 작업 종류마다 처리 시간이 다른가 (분산분석)
by_type = {k: g["elapsed_ms"].to_numpy() for k, g in req.dropna(subset=["task_type"]).groupby("task_type")}
anova = stats.f_oneway(*by_type.values())
levene_type = stats.levene(*by_type.values()).pvalue
tukey = stats.tukey_hsd(*by_type.values())                      # 사후 분석: 어느 쌍이 다른가
grand = np.concatenate(list(by_type.values())).mean()
ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in by_type.values())   # 집단 사이 흩어짐
ss_within = sum(((v - v.mean()) ** 2).sum() for v in by_type.values())          # 집단 안 흩어짐
k, n_all = len(by_type), sum(len(v) for v in by_type.values())
f_by_hand = (ss_between / (k - 1)) / (ss_within / (n_all - k))

# ④ 관계 보기: 상관, 교차분석, 심슨의 역설
has_len = req.dropna(subset=["input_chars"])
pearson = stats.pearsonr(has_len["input_chars"], has_len["elapsed_ms"])
spearman = stats.spearmanr(has_len["input_chars"], has_len["elapsed_ms"])
table = pd.crosstab(req["priority"], req["failed"])
chi2 = stats.chi2_contingency(table)                            # 우선순위와 실패가 서로 상관없나

simpson = pd.DataFrame({                                        # 가상의 두 모델, 쉬운 요청과 어려운 요청의 성공 건수 / 전체 건수
    "모델": ["A", "A", "B", "B"], "요청": ["쉬움", "어려움", "쉬움", "어려움"],
    "성공": [90, 150, 270, 40], "전체": [100, 300, 310, 90]})
simpson["성공률"] = simpson["성공"] / simpson["전체"]
overall = simpson.groupby("모델")[["성공", "전체"]].sum()
overall["성공률"] = overall["성공"] / overall["전체"]

# ⑤ 회귀 분석 직접 계산: 계수, t, p, R², F, VIF, Durbin-Watson
def ols(X, y, names):
    X = np.column_stack([np.ones(len(y)), X])                   # 절편 열
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    sigma2 = resid @ resid / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t_val = beta / se
    p_val = 2 * stats.t.sf(np.abs(t_val), n - p)
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    f_val = (r2 / (p - 1)) / ((1 - r2) / (n - p))
    dw = np.sum(np.diff(resid) ** 2) / (resid @ resid)         # 잔차가 이웃끼리 닮았나 (2 근처면 아님)
    coef = pd.DataFrame({"계수": beta, "표준오차": se, "t": t_val, "p": p_val}, index=["절편"] + names)
    return coef, {"R²": r2, "수정 R²": adj, "F": f_val, "F의 p": stats.f.sf(f_val, p - 1, n - p), "Durbin-Watson": dw, "n": n}

def vif(X, names):                                              # 한 열을 나머지 열로 설명할수록 커져요
    out = {}
    for j, name in enumerate(names):
        others = np.delete(X, j, axis=1)
        _, fit = ols(others, X[:, j], [f"x{i}" for i in range(others.shape[1])])
        out[name] = 1 / (1 - fit["R²"])
    return out

reg = req.dropna(subset=["input_chars", "allowed_tools", "task_type", "priority"]).copy()
reg["is_research"] = (reg["task_type"] == "research").astype(float)   # 더미 변수 (qa가 기준)
reg["is_summary"] = (reg["task_type"] == "summary").astype(float)
reg["is_high"] = (reg["priority"] == "high").astype(float)       # 우선순위 high인가
cols = ["input_chars", "allowed_tools", "is_research", "is_summary", "is_high"]
coef5, fit5 = ols(reg[cols].to_numpy(), reg["elapsed_ms"].to_numpy(), cols)
vif5 = vif(reg[cols].to_numpy(), cols)

# ⑥ 조절 효과: 우선순위가 입력 길이의 효과를 바꾸나 (상호작용과 평균 중심화)
reg["len_x_high"] = reg["input_chars"] * reg["is_high"]          # 상호작용항 (그대로 곱)
reg["mc_len"] = reg["input_chars"] - reg["input_chars"].mean()   # 평균 중심화
reg["mc_high"] = reg["is_high"] - reg["is_high"].mean()
reg["mc_len_x_high"] = reg["mc_len"] * reg["mc_high"]
steps_cols = {"1단계 (입력 길이)": ["input_chars"], "2단계 (+ 우선순위 high)": ["input_chars", "is_high"],
              "3단계 (+ 상호작용)": ["input_chars", "is_high", "len_x_high"]}
moderation = {name: ols(reg[c].to_numpy(), reg["elapsed_ms"].to_numpy(), c) for name, c in steps_cols.items()}
vif_raw = vif(reg[["input_chars", "is_high", "len_x_high"]].to_numpy(), ["input_chars", "is_high", "len_x_high"])
vif_centered = vif(reg[["mc_len", "mc_high", "mc_len_x_high"]].to_numpy(), ["mc_len", "mc_high", "mc_len_x_high"])
coef_mc, fit_mc = ols(reg[["mc_len", "mc_high", "mc_len_x_high"]].to_numpy(), reg["elapsed_ms"].to_numpy(), ["mc_len", "mc_high", "mc_len_x_high"])

# ⑦ 여러 번 검정하면: 아무 차이 없는 지표 20개
def twenty_tests():                                             # 차이가 전혀 없는 지표 20개를 각각 t검정
    return np.array([stats.ttest_ind(rng.normal(0, 1, 50), rng.normal(0, 1, 50)).pvalue for _ in range(20)])

trials = np.array([twenty_tests() for _ in range(2000)])
any_raw = (trials < 0.05).any(axis=1).mean()                    # 20개 중 하나라도 "유의"가 나온 비율
any_bonf = (trials < 0.05 / 20).any(axis=1).mean()              # 기준을 20으로 나누면 (Bonferroni)
