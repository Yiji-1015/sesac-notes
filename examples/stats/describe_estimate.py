# 업무 요청 100건으로 자료를 요약하고(기술통계), 모집단을 짐작하는(추정) 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/requests.csv (머신러닝 기초와 같은 가상의 업무 요청 100건). ④~⑥은 코드가 직접 만든 "1년치 요청" 모집단으로 표본을 뽑아 봐요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                                        #@1
import numpy as np                                                              #@1
import pandas as pd                                                             #@1
from scipy import stats                                                         #@2

DATA = Path(__file__).resolve().parent / "data"                                 #@1

# ① 자료와 척도
req = pd.read_csv(DATA / "requests.csv").set_index("request_id")
SCALES = {"task_type": "명목", "status": "명목", "failed": "명목 (0과 1)", "priority": "서열",
          "allowed_tools": "비율 (개수)", "retry_count": "비율 (횟수)", "input_chars": "비율", "elapsed_ms": "비율"}
missing = req.isna().sum()                                      # 열마다 빈 칸 수

# ② 요약하기: 가운데, 퍼짐, 모양
t = req["elapsed_ms"]
center = {"평균": t.mean(), "중앙값": t.median(), "10% 절사 평균": stats.trim_mean(t, 0.1)}
spread = {"범위": t.max() - t.min(), "사분위 범위": t.quantile(0.75) - t.quantile(0.25),
          "분산 (n − 1로 나눔)": t.var(), "표준편차": t.std(), "변동계수": t.std() / t.mean()}
shape = {"왜도": stats.skew(t), "첨도 (정규분포 = 0)": stats.kurtosis(t)}
r = req["retry_count"]
shape_retry = {"왜도": stats.skew(r), "첨도 (정규분포 = 0)": stats.kurtosis(r)}
growth = np.array([1.10, 1.50, 0.80])                           # 석 달 동안 요청 수가 전 달의 몇 배였나
means_growth = {"산술평균": growth.mean(), "기하평균": stats.gmean(growth)}   # 평균 성장률은 기하평균으로
speeds = np.array([60, 40])                                     # 같은 거리를 갈 때 60km/h, 올 때 40km/h
means_speed = {"산술평균": speeds.mean(), "조화평균": stats.hmean(speeds)}    # 평균 속력은 조화평균으로

# ③ 확률과 분포: 조건부 확률, 독립, 포아송
p_fail = req["failed"].mean()
p_fail_given = req.groupby("priority")["failed"].mean()         # P(실패 | 우선순위)
p_high = (req["priority"] == "high").mean()
p_high_and_fail = ((req["priority"] == "high") & (req["failed"] == 1)).mean()   # 결합 확률
lam = r.mean()                                                  # 재시도 횟수의 평균 = 포아송의 λ 후보
observed = r.value_counts().sort_index()
expected = pd.Series(stats.poisson.pmf(observed.index, lam) * len(r), index=observed.index)   # 포아송이면 기대 건수
dispersion = r.var() / r.mean()                                 # 포아송이면 1 근처

# ④ 표본 뽑기: 무작위, 편한 대로, 층화
rng = np.random.default_rng(0)
N = 100_000                                                     # 가상의 1년치 요청 (모집단)
pop_type = rng.choice(["qa", "research", "summary"], size=N, p=[0.5, 0.3, 0.2])
base = {"qa": 900, "research": 1600, "summary": 1200}
pop_ms = np.array([base[k] for k in pop_type]) * rng.lognormal(0, 0.4, size=N)   # 오른쪽으로 꼬리가 긴 처리 시간
true_mean = pop_ms.mean()                                       # 모수: 실제로는 모르는 값

def simple_random(n):
    return pop_ms[rng.choice(N, n, replace=False)].mean()

def convenience(n):                                             # 편하게 research 요청함에서만 뽑기
    pool = np.flatnonzero(pop_type == "research")
    return pop_ms[rng.choice(pool, n, replace=False)].mean()

def stratified(n):                                              # 종류 비율대로 나눠 각각 무작위
    est = 0.0
    for k in ("qa", "research", "summary"):
        pool = np.flatnonzero(pop_type == k)
        share = len(pool) / N
        est += share * pop_ms[rng.choice(pool, round(n * share), replace=False)].mean()
    return est

draws = {name: np.array([f(100) for _ in range(2000)]) for name, f in
         (("단순 무작위", simple_random), ("편한 대로", convenience), ("층화", stratified))}

# ⑤ 표본 평균의 분포: 중심극한정리
sizes = (1, 5, 30, 100)
sample_means = {n: np.array([pop_ms[rng.integers(N, size=n)].mean() for _ in range(5000)]) for n in sizes}
se_theory = {n: pop_ms.std() / np.sqrt(n) for n in sizes}      # 표준오차 = σ ÷ √n

# ⑥ 신뢰구간: 100번 중 몇 번 맞나
def interval(x, level=0.95, use_t=True):
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    q = stats.t.ppf((1 + level) / 2, n - 1) if use_t else stats.norm.ppf((1 + level) / 2)
    return m - q * s / np.sqrt(n), m + q * s / np.sqrt(n)

coverage = {}
for n in (5, 30, 100):
    for use_t in (False, True):
        hits = 0
        for _ in range(2000):
            lo, hi = interval(pop_ms[rng.integers(N, size=n)], use_t=use_t)
            hits += lo <= true_mean <= hi
        coverage[(n, use_t)] = hits / 2000
ci_ours = interval(t.to_numpy())                                # 우리 100건으로 낸 평균 처리 시간의 95% 신뢰구간
