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
   "t": "from scipy import stats",
   "s": 1
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
   "t": "rng = np.random.default_rng(0)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 검정의 뼈대: 1종 오류와 검정력",
   "s": 1
  },
  {
   "t": "def experiment(n, diff, sd=600, alpha=0.05, repeats=4000):  # 두 집단에 진짜 차이 diff가 있을 때, n건씩 재서 차이를 찾는 비율",
   "s": 1
  },
  {
   "t": "    found = 0",
   "s": 1
  },
  {
   "t": "    for _ in range(repeats):",
   "s": 1
  },
  {
   "t": "        a, b = rng.normal(1200, sd, n), rng.normal(1200 + diff, sd, n)",
   "s": 1
  },
  {
   "t": "        found += stats.ttest_ind(a, b).pvalue < alpha",
   "s": 1
  },
  {
   "t": "    return found / repeats",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "type1 = {n: experiment(n, diff=0) for n in (10, 30, 100)}  # 차이가 없는데 \"있다\"고 한 비율 = 1종 오류",
   "s": 1
  },
  {
   "t": "power = {(n, d): experiment(n, diff=d) for n in (10, 30, 100) for d in (200, 400)}  # 차이가 있을 때 찾은 비율 = 검정력",
   "s": 1
  },
  {
   "t": "need_n = int(np.ceil(2 * ((stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * 600 / 200) ** 2))  # 200 차이를 80%로 찾으려면 집단당 몇 건",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 두 집단 비교: 실패한 요청은 더 오래 걸렸나",
   "s": 2
  },
  {
   "t": "req = pd.read_csv(DATA / \"requests.csv\").set_index(\"request_id\")",
   "s": 2
  },
  {
   "t": "failed_ms, ok_ms = req.loc[req[\"failed\"] == 1, \"elapsed_ms\"], req.loc[req[\"failed\"] == 0, \"elapsed_ms\"]",
   "s": 2
  },
  {
   "t": "normal_p = (stats.shapiro(failed_ms).pvalue, stats.shapiro(ok_ms).pvalue)  # 정규성 (Shapiro-Wilk)",
   "s": 2
  },
  {
   "t": "levene_p = stats.levene(failed_ms, ok_ms).pvalue  # 등분산 (Levene)",
   "s": 2
  },
  {
   "t": "student = stats.ttest_ind(failed_ms, ok_ms)  # 분산이 같다고 보는 t검정",
   "s": 2
  },
  {
   "t": "welch = stats.ttest_ind(failed_ms, ok_ms, equal_var=False)  # 분산이 달라도 되는 t검정 (Welch)",
   "s": 2
  },
  {
   "t": "mann = stats.mannwhitneyu(failed_ms, ok_ms)  # 정규성을 가정하지 않는 순위 검정",
   "s": 2
  },
  {
   "t": "pooled = np.sqrt(((len(failed_ms) - 1) * failed_ms.var() + (len(ok_ms) - 1) * ok_ms.var()) / (len(failed_ms) + len(ok_ms) - 2))",
   "s": 2
  },
  {
   "t": "cohen_d = (failed_ms.mean() - ok_ms.mean()) / pooled  # 효과 크기: 차이가 표준편차의 몇 배인가",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 셋 이상 비교: 작업 종류마다 처리 시간이 다른가 (분산분석)",
   "s": 3
  },
  {
   "t": "by_type = {k: g[\"elapsed_ms\"].to_numpy() for k, g in req.dropna(subset=[\"task_type\"]).groupby(\"task_type\")}",
   "s": 3
  },
  {
   "t": "anova = stats.f_oneway(*by_type.values())",
   "s": 3
  },
  {
   "t": "levene_type = stats.levene(*by_type.values()).pvalue",
   "s": 3
  },
  {
   "t": "tukey = stats.tukey_hsd(*by_type.values())  # 사후 분석: 어느 쌍이 다른가",
   "s": 3
  },
  {
   "t": "grand = np.concatenate(list(by_type.values())).mean()",
   "s": 3
  },
  {
   "t": "ss_between = sum(len(v) * (v.mean() - grand) ** 2 for v in by_type.values())  # 집단 사이 흩어짐",
   "s": 3
  },
  {
   "t": "ss_within = sum(((v - v.mean()) ** 2).sum() for v in by_type.values())  # 집단 안 흩어짐",
   "s": 3
  },
  {
   "t": "k, n_all = len(by_type), sum(len(v) for v in by_type.values())",
   "s": 3
  },
  {
   "t": "f_by_hand = (ss_between / (k - 1)) / (ss_within / (n_all - k))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 관계 보기: 상관, 교차분석, 심슨의 역설",
   "s": 4
  },
  {
   "t": "has_len = req.dropna(subset=[\"input_chars\"])",
   "s": 4
  },
  {
   "t": "pearson = stats.pearsonr(has_len[\"input_chars\"], has_len[\"elapsed_ms\"])",
   "s": 4
  },
  {
   "t": "spearman = stats.spearmanr(has_len[\"input_chars\"], has_len[\"elapsed_ms\"])",
   "s": 4
  },
  {
   "t": "table = pd.crosstab(req[\"priority\"], req[\"failed\"])",
   "s": 4
  },
  {
   "t": "chi2 = stats.chi2_contingency(table)  # 우선순위와 실패가 서로 상관없나",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "simpson = pd.DataFrame({  # 가상의 두 모델, 쉬운 요청과 어려운 요청의 성공 건수 / 전체 건수",
   "s": 4
  },
  {
   "t": "    \"모델\": [\"A\", \"A\", \"B\", \"B\"], \"요청\": [\"쉬움\", \"어려움\", \"쉬움\", \"어려움\"],",
   "s": 4
  },
  {
   "t": "    \"성공\": [90, 150, 270, 40], \"전체\": [100, 300, 310, 90]})",
   "s": 4
  },
  {
   "t": "simpson[\"성공률\"] = simpson[\"성공\"] / simpson[\"전체\"]",
   "s": 4
  },
  {
   "t": "overall = simpson.groupby(\"모델\")[[\"성공\", \"전체\"]].sum()",
   "s": 4
  },
  {
   "t": "overall[\"성공률\"] = overall[\"성공\"] / overall[\"전체\"]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 회귀 분석 직접 계산: 계수, t, p, R², F, VIF, Durbin-Watson",
   "s": 5
  },
  {
   "t": "def ols(X, y, names):",
   "s": 5
  },
  {
   "t": "    X = np.column_stack([np.ones(len(y)), X])  # 절편 열",
   "s": 5
  },
  {
   "t": "    beta, *_ = np.linalg.lstsq(X, y, rcond=None)",
   "s": 5
  },
  {
   "t": "    resid = y - X @ beta",
   "s": 5
  },
  {
   "t": "    n, p = X.shape",
   "s": 5
  },
  {
   "t": "    sigma2 = resid @ resid / (n - p)",
   "s": 5
  },
  {
   "t": "    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))",
   "s": 5
  },
  {
   "t": "    t_val = beta / se",
   "s": 5
  },
  {
   "t": "    p_val = 2 * stats.t.sf(np.abs(t_val), n - p)",
   "s": 5
  },
  {
   "t": "    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))",
   "s": 5
  },
  {
   "t": "    adj = 1 - (1 - r2) * (n - 1) / (n - p)",
   "s": 5
  },
  {
   "t": "    f_val = (r2 / (p - 1)) / ((1 - r2) / (n - p))",
   "s": 5
  },
  {
   "t": "    dw = np.sum(np.diff(resid) ** 2) / (resid @ resid)  # 잔차가 이웃끼리 닮았나 (2 근처면 아님)",
   "s": 5
  },
  {
   "t": "    coef = pd.DataFrame({\"계수\": beta, \"표준오차\": se, \"t\": t_val, \"p\": p_val}, index=[\"절편\"] + names)",
   "s": 5
  },
  {
   "t": "    return coef, {\"R²\": r2, \"수정 R²\": adj, \"F\": f_val, \"F의 p\": stats.f.sf(f_val, p - 1, n - p), \"Durbin-Watson\": dw, \"n\": n}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def vif(X, names):  # 한 열을 나머지 열로 설명할수록 커져요",
   "s": 5
  },
  {
   "t": "    out = {}",
   "s": 5
  },
  {
   "t": "    for j, name in enumerate(names):",
   "s": 5
  },
  {
   "t": "        others = np.delete(X, j, axis=1)",
   "s": 5
  },
  {
   "t": "        _, fit = ols(others, X[:, j], [f\"x{i}\" for i in range(others.shape[1])])",
   "s": 5
  },
  {
   "t": "        out[name] = 1 / (1 - fit[\"R²\"])",
   "s": 5
  },
  {
   "t": "    return out",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "reg = req.dropna(subset=[\"input_chars\", \"allowed_tools\", \"task_type\", \"priority\"]).copy()",
   "s": 5
  },
  {
   "t": "reg[\"is_research\"] = (reg[\"task_type\"] == \"research\").astype(float)  # 더미 변수 (qa가 기준)",
   "s": 5
  },
  {
   "t": "reg[\"is_summary\"] = (reg[\"task_type\"] == \"summary\").astype(float)",
   "s": 5
  },
  {
   "t": "reg[\"is_high\"] = (reg[\"priority\"] == \"high\").astype(float)  # 우선순위 high인가",
   "s": 5
  },
  {
   "t": "cols = [\"input_chars\", \"allowed_tools\", \"is_research\", \"is_summary\", \"is_high\"]",
   "s": 5
  },
  {
   "t": "coef5, fit5 = ols(reg[cols].to_numpy(), reg[\"elapsed_ms\"].to_numpy(), cols)",
   "s": 5
  },
  {
   "t": "vif5 = vif(reg[cols].to_numpy(), cols)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 조절 효과: 우선순위가 입력 길이의 효과를 바꾸나 (상호작용과 평균 중심화)",
   "s": 6
  },
  {
   "t": "reg[\"len_x_high\"] = reg[\"input_chars\"] * reg[\"is_high\"]  # 상호작용항 (그대로 곱)",
   "s": 6
  },
  {
   "t": "reg[\"mc_len\"] = reg[\"input_chars\"] - reg[\"input_chars\"].mean()  # 평균 중심화",
   "s": 6
  },
  {
   "t": "reg[\"mc_high\"] = reg[\"is_high\"] - reg[\"is_high\"].mean()",
   "s": 6
  },
  {
   "t": "reg[\"mc_len_x_high\"] = reg[\"mc_len\"] * reg[\"mc_high\"]",
   "s": 6
  },
  {
   "t": "steps_cols = {\"1단계 (입력 길이)\": [\"input_chars\"], \"2단계 (+ 우선순위 high)\": [\"input_chars\", \"is_high\"],",
   "s": 6
  },
  {
   "t": "              \"3단계 (+ 상호작용)\": [\"input_chars\", \"is_high\", \"len_x_high\"]}",
   "s": 6
  },
  {
   "t": "moderation = {name: ols(reg[c].to_numpy(), reg[\"elapsed_ms\"].to_numpy(), c) for name, c in steps_cols.items()}",
   "s": 6
  },
  {
   "t": "vif_raw = vif(reg[[\"input_chars\", \"is_high\", \"len_x_high\"]].to_numpy(), [\"input_chars\", \"is_high\", \"len_x_high\"])",
   "s": 6
  },
  {
   "t": "vif_centered = vif(reg[[\"mc_len\", \"mc_high\", \"mc_len_x_high\"]].to_numpy(), [\"mc_len\", \"mc_high\", \"mc_len_x_high\"])",
   "s": 6
  },
  {
   "t": "coef_mc, fit_mc = ols(reg[[\"mc_len\", \"mc_high\", \"mc_len_x_high\"]].to_numpy(), reg[\"elapsed_ms\"].to_numpy(), [\"mc_len\", \"mc_high\", \"mc_len_x_high\"])",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 여러 번 검정하면: 아무 차이 없는 지표 20개",
   "s": 7
  },
  {
   "t": "def twenty_tests():  # 차이가 전혀 없는 지표 20개를 각각 t검정",
   "s": 7
  },
  {
   "t": "    return np.array([stats.ttest_ind(rng.normal(0, 1, 50), rng.normal(0, 1, 50)).pvalue for _ in range(20)])",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "trials = np.array([twenty_tests() for _ in range(2000)])",
   "s": 7
  },
  {
   "t": "any_raw = (trials < 0.05).any(axis=1).mean()  # 20개 중 하나라도 \"유의\"가 나온 비율",
   "s": 7
  },
  {
   "t": "any_bonf = (trials < 0.05 / 20).any(axis=1).mean()  # 기준을 20으로 나누면 (Bonferroni)",
   "s": 7
  }
 ],
 "names": {
  "1": "검정의 뼈대: 1종 오류와 검정력",
  "2": "두 집단 비교: 실패한 요청은 더 오래 걸렸나",
  "3": "셋 이상 비교: 작업 종류마다 처리 시간이 다른가 (분산분석)",
  "4": "관계 보기: 상관, 교차분석, 심슨의 역설",
  "5": "회귀 분석 직접 계산: 계수, t, p, R², F, VIF, Durbin-Watson",
  "6": "조절 효과: 우선순위가 입력 길이의 효과를 바꾸나 (상호작용과 평균 중심화)",
  "7": "여러 번 검정하면: 아무 차이 없는 지표 20개"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 표준편차 600인 두 집단을 n건씩 4,000번 비교 (α = 0.05)",
   "cols": [
    "n (집단당)",
    "진짜 차이 0: \"차이 있다\" 비율",
    "진짜 차이 200: 찾은 비율",
    "진짜 차이 400: 찾은 비율"
   ],
   "rows": [
    [
     10,
     "4.7%",
     "11.4%",
     "29.1%"
    ],
    [
     30,
     "4.9%",
     "24.2%",
     "72.5%"
    ],
    [
     100,
     "5.6%",
     "63.2%",
     "99.8%"
    ]
   ],
   "notes": [
    "차이가 없을 때 \"있다\"고 잘못 말하는 비율(1종 오류)은 n과 상관없이 정한 α = 5% 근처예요. α는 우리가 고르는 값이에요.",
    "차이가 있을 때 찾아내는 비율이 <b>검정력</b>(1 − 2종 오류)이에요. 200의 차이는 10건씩이면 11%만 찾아요. 이 차이를 80%로 찾으려면 집단당 142건이 필요해요. 그래서 \"유의하지 않다\"는 \"차이가 없다\"가 아니라 \"이 건수로는 못 찾았다\"예요."
   ]
  },
  "s2": {
   "title": "② 실패 30건 대 성공 70건의 처리 시간",
   "cols": [
    "확인",
    "통계량",
    "p",
    "읽기"
   ],
   "rows": [
    [
     "평균 (실패 / 성공)",
     "1,807 / 1,051ms",
     "",
     "차이 756ms"
    ],
    [
     "정규성 (Shapiro-Wilk), 실패 쪽",
     "",
     "0.248",
     "정규분포와 다르다고 할 수 없음"
    ],
    [
     "정규성 (Shapiro-Wilk), 성공 쪽",
     "",
     "0.011",
     "정규분포와 다름"
    ],
    [
     "등분산 (Levene)",
     "",
     "0.243",
     "분산이 다르다고 할 수 없음"
    ],
    [
     "t검정 (등분산 가정)",
     "t = 6.38",
     "< 0.001",
     "평균이 다름"
    ],
    [
     "Welch t검정 (등분산 가정 안 함)",
     "t = 6.72",
     "< 0.001",
     "평균이 다름"
    ],
    [
     "Mann-Whitney U (순위)",
     "",
     "< 0.001",
     "분포가 한쪽으로 밀려 있음"
    ],
    [
     "효과 크기 (Cohen의 d)",
     "1.39",
     "",
     "표준편차의 1.4배 차이, 큰 편"
    ]
   ],
   "notes": [
    "성공 쪽은 정규성 검정에서 걸렸지만 70건이라 중심극한정리로 평균은 거의 정규로 다룰 수 있고, 순위만 보는 Mann-Whitney도 같은 결론이에요. 결론이 방법에 따라 바뀌지 않으면 믿을 만해요.",
    "p값은 \"차이가 없다면 이만큼 큰 차이가 우연히 나올 확률\"이에요. 아주 작다는 건 차이가 있다는 뜻이지 <b>차이가 크다</b>는 뜻이 아니에요. 그래서 효과 크기를 같이 적어요. 다만 여기서는 실패한 요청이 재시도하며 오래 걸린 것이라, 오래 걸려서 실패했다고 거꾸로 읽으면 안 돼요."
   ]
  },
  "s3": {
   "title": "③ 작업 종류별 처리 시간 (일원 분산분석)",
   "cols": [
    "무엇",
    "값"
   ],
   "rows": [
    [
     "qa 평균 (n)",
     "1,417ms (31)"
    ],
    [
     "research 평균 (n)",
     "1,282ms (31)"
    ],
    [
     "summary 평균 (n)",
     "1,145ms (35)"
    ],
    [
     "집단 사이 제곱합 / 집단 안 제곱합",
     "1,220,488 / 38,038,920"
    ],
    [
     "F = (사이 ÷ (k − 1)) ÷ (안 ÷ (n − k))",
     "1.508 (scipy 1.508)"
    ],
    [
     "p",
     "0.227"
    ],
    [
     "등분산 (Levene) p",
     "0.473"
    ],
    [
     "Tukey: qa − research",
     "+135ms, p = 0.682"
    ],
    [
     "Tukey: qa − summary",
     "+272ms, p = 0.198"
    ],
    [
     "Tukey: research − summary",
     "+137ms, p = 0.658"
    ]
   ],
   "notes": [
    "평균은 qa가 summary보다 272ms 길지만, 집단 안 흩어짐이 워낙 커서 F가 1.51에 그쳐요. 이 100건으로는 종류마다 다르다고 말할 수 없어요.",
    "t검정을 세 쌍에 따로 하면 우연히 하나쯤 걸릴 확률이 커져요(⑦). 그래서 먼저 분산분석으로 \"어딘가 다른가\"를 보고, 다르면 Tukey 같은 사후 분석으로 어느 쌍인지 봐요. 여기서는 분산분석도 사후 분석도 모두 유의하지 않아요."
   ]
  },
  "f_box": {
   "title": "집단별 처리 시간 (상자: 가운데 50%, 분홍 선: 중앙값)",
   "images": [
    [
     "img/stats/group_compare.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽은 두 상자가 거의 겹치지 않아요. 오른쪽은 세 상자가 크게 겹쳐요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ 관계 보기",
   "cols": [
    "무엇",
    "값",
    "p"
   ],
   "rows": [
    [
     "입력 길이와 처리 시간: Pearson r (곧은 관계)",
     "0.291",
     "0.004"
    ],
    [
     "같은 둘: Spearman ρ (순위 관계)",
     "0.256",
     "0.013"
    ],
    [
     "우선순위 × 실패: χ² (교차분석)",
     "2.27 (자유도 2)",
     "0.321"
    ]
   ],
   "notes": [
    "상관 0.29는 유의하지만 약해요. r²로 보면 처리 시간 흩어짐의 8%만 입력 길이로 설명돼요. 상관은 인과가 아니에요.",
    "χ² 검정은 \"두 범주형 변수가 독립이라면 각 칸에 몇 건이 기대되나\"와 실제 건수를 비교해요. 앞 쪽 ③의 \"high의 실패율이 높아 보였던 것\"은 우연으로 설명될 만한 차이예요."
   ]
  },
  "s4s": {
   "title": "심슨의 역설: 가상의 두 모델",
   "cols": [
    "모델",
    "쉬운 요청",
    "어려운 요청",
    "합치면"
   ],
   "rows": [
    [
     "A",
     "90/100 = 90.0%",
     "150/300 = 50.0%",
     "240/400 = 60.0%"
    ],
    [
     "B",
     "270/310 = 87.1%",
     "40/90 = 44.4%",
     "310/400 = 77.5%"
    ]
   ],
   "notes": [
    "A는 쉬운 요청에서도 어려운 요청에서도 B보다 성공률이 높아요. 그런데 합치면 B가 훨씬 높아요. A가 어려운 요청을 훨씬 많이(300건) 맡았기 때문이에요.",
    "어떤 요청을 맡았는지(혼란 변수)를 무시하고 합치면 결론이 뒤집혀요. 관측 자료로 비교할 때는 이런 변수로 나눠 보거나, 처음부터 무작위로 나눠 맡겨야(실험, A/B 검정) 해요."
   ]
  },
  "s5": {
   "title": "⑤ 처리 시간 회귀 (n = 85, 빈 칸 있는 행 제외)",
   "cols": [
    "변수",
    "계수",
    "표준오차",
    "t",
    "p"
   ],
   "rows": [
    [
     "<code>절편</code>",
     "970.8440",
     "232.5907",
     "4.17",
     "< 0.001"
    ],
    [
     "<code>input_chars</code>",
     "0.3007",
     "0.1288",
     "2.34",
     "0.022"
    ],
    [
     "<code>allowed_tools</code>",
     "-11.3044",
     "49.8366",
     "-0.23",
     "0.821"
    ],
    [
     "<code>is_research</code>",
     "-76.8881",
     "173.6576",
     "-0.44",
     "0.659"
    ],
    [
     "<code>is_summary</code>",
     "-192.1293",
     "173.0328",
     "-1.11",
     "0.270"
    ],
    [
     "<code>is_high</code>",
     "104.2528",
     "151.8720",
     "0.69",
     "0.494"
    ]
   ],
   "notes": [
    "R² = 0.092, 수정 R² = 0.035, F = 1.60 (p = 0.169), Durbin-Watson = 2.00, VIF 최대 1.50",
    "입력 길이 1자가 늘면 처리 시간이 0.30ms 늘어요(다른 변수가 같을 때). 이것만 유의해요. 작업 종류와 우선순위 계수는 0과 구별되지 않아요.",
    "R²가 0.09로, 이 변수들로는 처리 시간 흩어짐의 9%만 설명돼요. 변수를 넣을수록 R²는 저절로 커지니, 변수 수를 깎아 준 수정 R²(0.035)를 봐요. F 검정은 \"모든 계수가 0\"을 기각하지 못했어요(p = 0.169). 입력 길이 하나는 유의한데 전체는 아닌 건, 쓸모없는 변수 넷이 함께 들어가 힘을 나눠 먹었기 때문이에요.",
    "VIF가 모두 2 아래라 변수끼리 서로를 설명하는 문제(다중공선성)는 없어요. Durbin-Watson이 2 근처라 잔차가 이웃끼리 닮지도 않았어요."
   ]
  },
  "s6": {
   "title": "⑥ 우선순위 high가 입력 길이의 효과를 바꾸나 (위계적 회귀 3단계)",
   "cols": [
    "단계",
    "R²",
    "추가된 변수의 계수",
    "그 p"
   ],
   "rows": [
    [
     "1단계 (입력 길이)",
     "0.068",
     "0.3099 (<code>input_chars</code>)",
     "0.016"
    ],
    [
     "2단계 (+ 우선순위 high)",
     "0.075",
     "116.4006 (<code>is_high</code>)",
     "0.429"
    ],
    [
     "3단계 (+ 상호작용)",
     "0.088",
     "-0.3033 (<code>len_x_high</code>)",
     "0.276"
    ]
   ],
   "notes": [
    "3단계에서 상호작용항의 계수가 -0.303(high면 1자당 0.30ms 덜 늘어남)이지만 p = 0.28라, 이 데이터로는 조절 효과가 있다고 말할 수 없어요.",
    "조절 효과는 이렇게 단계를 나눠 넣어서, 마지막 단계의 상호작용항이 유의한지와 R²가 얼마나 늘었는지로 판단해요."
   ]
  },
  "s6v": {
   "title": "상호작용항을 넣었을 때의 VIF",
   "cols": [
    "변수",
    "그대로 곱했을 때",
    "평균 중심화 후"
   ],
   "rows": [
    [
     "input_chars / mc_len",
     "1.45",
     "1.01"
    ],
    [
     "is_high / mc_high",
     "6.31",
     "1.02"
    ],
    [
     "len_x_high / mc_len_x_high",
     "7.06",
     "1.01"
    ]
   ],
   "notes": [
    "그대로 곱한 상호작용항은 원래 두 변수와 많이 닮아서 VIF가 5를 넘어요. 각 변수에서 평균을 빼고 곱하면(평균 중심화) VIF가 1 근처로 내려가요.",
    "상호작용항의 계수와 p는 중심화 전후로 똑같아요(-0.3033). 바뀌는 건 나머지 두 변수의 계수 뜻이에요. 중심화하면 \"다른 변수가 평균일 때의 효과\"가 돼서 읽기 쉬워져요."
   ]
  },
  "f_mod": {
   "title": "우선순위에 따른 기울기",
   "images": [
    [
     "img/stats/moderation.png",
     ""
    ]
   ],
   "notes": [
    "분홍 선(high)이 보라 선보다 기울기가 완만해요. 하지만 high가 27건뿐이고 점이 크게 흩어져 있어서, 이 차이는 우연으로도 충분히 생겨요."
   ],
   "chart": true
  },
  "s7": {
   "title": "⑦ 아무 차이 없는 지표 20개를 매번 검정 (2,000번)",
   "lines": [
    [
     "20개 중 하나라도 p < 0.05 → 65.3%",
     1
    ],
    [
     "기준을 0.05 ÷ 20 = 0.0025로 (Bonferroni) → 5.7%",
     2,
     "newf"
    ],
    "한 번에 \"유의\"가 나온 지표 수의 평균 → 1.03개"
   ],
   "notes": [
    "하나씩은 5%라도 20번 보면 1 − 0.95²⁰ ≈ 64%의 확률로 하나는 걸려요. 지표를 많이 보거나 데이터를 여러 방식으로 잘라 보다가 유의한 것만 보고하면, 없는 차이를 \"발견\"하게 돼요(p-해킹).",
    "미리 정한 하나의 지표로 판단하거나, 검정 수만큼 기준을 엄격하게 해요. Bonferroni는 가장 단순하고 보수적인 방법이고, 덜 보수적인 Holm, 발견 중 틀린 비율을 조절하는 Benjamini-Hochberg도 있어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "검정의 뼈대",
   "1종 오류, 검정력",
   "몇 건이 필요한가"
  ],
  [
   2,
   "두 집단",
   "t, Welch, 순위",
   "실패가 더 오래"
  ],
  [
   3,
   "셋 이상",
   "분산분석, Tukey",
   "차이 없음"
  ],
  [
   4,
   "관계",
   "상관, χ², 심슨",
   "합치면 뒤집힘"
  ],
  [
   5,
   "회귀",
   "계수, R², F, VIF",
   "입력 길이만"
  ],
  [
   6,
   "조절 효과",
   "상호작용, 중심화",
   "유의하지 않음"
  ],
  [
   7,
   "여러 번 검정",
   "20개 중 하나",
   "Bonferroni"
  ]
 ],
 "env": "Python 3.13.5, pandas 2.3.3, SciPy 1.18.0, NumPy 2.5.2 (seed 0)"
};
