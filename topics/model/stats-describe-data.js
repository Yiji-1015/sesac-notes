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
   "s": 2
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
   "t": "# ① 자료와 척도",
   "s": 1
  },
  {
   "t": "req = pd.read_csv(DATA / \"requests.csv\").set_index(\"request_id\")",
   "s": 1
  },
  {
   "t": "SCALES = {\"task_type\": \"명목\", \"status\": \"명목\", \"failed\": \"명목 (0과 1)\", \"priority\": \"서열\",",
   "s": 1
  },
  {
   "t": "          \"allowed_tools\": \"비율 (개수)\", \"retry_count\": \"비율 (횟수)\", \"input_chars\": \"비율\", \"elapsed_ms\": \"비율\"}",
   "s": 1
  },
  {
   "t": "missing = req.isna().sum()  # 열마다 빈 칸 수",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 요약하기: 가운데, 퍼짐, 모양",
   "s": 2
  },
  {
   "t": "t = req[\"elapsed_ms\"]",
   "s": 2
  },
  {
   "t": "center = {\"평균\": t.mean(), \"중앙값\": t.median(), \"10% 절사 평균\": stats.trim_mean(t, 0.1)}",
   "s": 2
  },
  {
   "t": "spread = {\"범위\": t.max() - t.min(), \"사분위 범위\": t.quantile(0.75) - t.quantile(0.25),",
   "s": 2
  },
  {
   "t": "          \"분산 (n − 1로 나눔)\": t.var(), \"표준편차\": t.std(), \"변동계수\": t.std() / t.mean()}",
   "s": 2
  },
  {
   "t": "shape = {\"왜도\": stats.skew(t), \"첨도 (정규분포 = 0)\": stats.kurtosis(t)}",
   "s": 2
  },
  {
   "t": "r = req[\"retry_count\"]",
   "s": 2
  },
  {
   "t": "shape_retry = {\"왜도\": stats.skew(r), \"첨도 (정규분포 = 0)\": stats.kurtosis(r)}",
   "s": 2
  },
  {
   "t": "growth = np.array([1.10, 1.50, 0.80])  # 석 달 동안 요청 수가 전 달의 몇 배였나",
   "s": 2
  },
  {
   "t": "means_growth = {\"산술평균\": growth.mean(), \"기하평균\": stats.gmean(growth)}  # 평균 성장률은 기하평균으로",
   "s": 2
  },
  {
   "t": "speeds = np.array([60, 40])  # 같은 거리를 갈 때 60km/h, 올 때 40km/h",
   "s": 2
  },
  {
   "t": "means_speed = {\"산술평균\": speeds.mean(), \"조화평균\": stats.hmean(speeds)}  # 평균 속력은 조화평균으로",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 확률과 분포: 조건부 확률, 독립, 포아송",
   "s": 3
  },
  {
   "t": "p_fail = req[\"failed\"].mean()",
   "s": 3
  },
  {
   "t": "p_fail_given = req.groupby(\"priority\")[\"failed\"].mean()  # P(실패 | 우선순위)",
   "s": 3
  },
  {
   "t": "p_high = (req[\"priority\"] == \"high\").mean()",
   "s": 3
  },
  {
   "t": "p_high_and_fail = ((req[\"priority\"] == \"high\") & (req[\"failed\"] == 1)).mean()  # 결합 확률",
   "s": 3
  },
  {
   "t": "lam = r.mean()  # 재시도 횟수의 평균 = 포아송의 λ 후보",
   "s": 3
  },
  {
   "t": "observed = r.value_counts().sort_index()",
   "s": 3
  },
  {
   "t": "expected = pd.Series(stats.poisson.pmf(observed.index, lam) * len(r), index=observed.index)  # 포아송이면 기대 건수",
   "s": 3
  },
  {
   "t": "dispersion = r.var() / r.mean()  # 포아송이면 1 근처",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 표본 뽑기: 무작위, 편한 대로, 층화",
   "s": 4
  },
  {
   "t": "rng = np.random.default_rng(0)",
   "s": 4
  },
  {
   "t": "N = 100_000  # 가상의 1년치 요청 (모집단)",
   "s": 4
  },
  {
   "t": "pop_type = rng.choice([\"qa\", \"research\", \"summary\"], size=N, p=[0.5, 0.3, 0.2])",
   "s": 4
  },
  {
   "t": "base = {\"qa\": 900, \"research\": 1600, \"summary\": 1200}",
   "s": 4
  },
  {
   "t": "pop_ms = np.array([base[k] for k in pop_type]) * rng.lognormal(0, 0.4, size=N)  # 오른쪽으로 꼬리가 긴 처리 시간",
   "s": 4
  },
  {
   "t": "true_mean = pop_ms.mean()  # 모수: 실제로는 모르는 값",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def simple_random(n):",
   "s": 4
  },
  {
   "t": "    return pop_ms[rng.choice(N, n, replace=False)].mean()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def convenience(n):  # 편하게 research 요청함에서만 뽑기",
   "s": 4
  },
  {
   "t": "    pool = np.flatnonzero(pop_type == \"research\")",
   "s": 4
  },
  {
   "t": "    return pop_ms[rng.choice(pool, n, replace=False)].mean()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def stratified(n):  # 종류 비율대로 나눠 각각 무작위",
   "s": 4
  },
  {
   "t": "    est = 0.0",
   "s": 4
  },
  {
   "t": "    for k in (\"qa\", \"research\", \"summary\"):",
   "s": 4
  },
  {
   "t": "        pool = np.flatnonzero(pop_type == k)",
   "s": 4
  },
  {
   "t": "        share = len(pool) / N",
   "s": 4
  },
  {
   "t": "        est += share * pop_ms[rng.choice(pool, round(n * share), replace=False)].mean()",
   "s": 4
  },
  {
   "t": "    return est",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "draws = {name: np.array([f(100) for _ in range(2000)]) for name, f in",
   "s": 4
  },
  {
   "t": "         ((\"단순 무작위\", simple_random), (\"편한 대로\", convenience), (\"층화\", stratified))}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 표본 평균의 분포: 중심극한정리",
   "s": 5
  },
  {
   "t": "sizes = (1, 5, 30, 100)",
   "s": 5
  },
  {
   "t": "sample_means = {n: np.array([pop_ms[rng.integers(N, size=n)].mean() for _ in range(5000)]) for n in sizes}",
   "s": 5
  },
  {
   "t": "se_theory = {n: pop_ms.std() / np.sqrt(n) for n in sizes}  # 표준오차 = σ ÷ √n",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 신뢰구간: 100번 중 몇 번 맞나",
   "s": 6
  },
  {
   "t": "def interval(x, level=0.95, use_t=True):",
   "s": 6
  },
  {
   "t": "    n, m, s = len(x), x.mean(), x.std(ddof=1)",
   "s": 6
  },
  {
   "t": "    q = stats.t.ppf((1 + level) / 2, n - 1) if use_t else stats.norm.ppf((1 + level) / 2)",
   "s": 6
  },
  {
   "t": "    return m - q * s / np.sqrt(n), m + q * s / np.sqrt(n)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "coverage = {}",
   "s": 6
  },
  {
   "t": "for n in (5, 30, 100):",
   "s": 6
  },
  {
   "t": "    for use_t in (False, True):",
   "s": 6
  },
  {
   "t": "        hits = 0",
   "s": 6
  },
  {
   "t": "        for _ in range(2000):",
   "s": 6
  },
  {
   "t": "            lo, hi = interval(pop_ms[rng.integers(N, size=n)], use_t=use_t)",
   "s": 6
  },
  {
   "t": "            hits += lo <= true_mean <= hi",
   "s": 6
  },
  {
   "t": "        coverage[(n, use_t)] = hits / 2000",
   "s": 6
  },
  {
   "t": "ci_ours = interval(t.to_numpy())  # 우리 100건으로 낸 평균 처리 시간의 95% 신뢰구간",
   "s": 6
  }
 ],
 "names": {
  "1": "자료와 척도",
  "2": "요약하기: 가운데, 퍼짐, 모양",
  "3": "확률과 분포: 조건부 확률, 독립, 포아송",
  "4": "표본 뽑기: 무작위, 편한 대로, 층화",
  "5": "표본 평균의 분포: 중심극한정리",
  "6": "신뢰구간: 100번 중 몇 번 맞나"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "s1": {
   "title": "① 열마다 척도와 빈 칸",
   "cols": [
    "열",
    "척도",
    "빈 칸",
    "예시 값"
   ],
   "rows": [
    [
     "<code>task_type</code>",
     "명목",
     3,
     "summary, qa, research"
    ],
    [
     "<code>status</code>",
     "명목",
     0,
     "error, ok"
    ],
    [
     "<code>failed</code>",
     "명목 (0과 1)",
     0,
     "1, 0"
    ],
    [
     "<code>priority</code>",
     "서열",
     2,
     "normal, low, high"
    ],
    [
     "<code>allowed_tools</code>",
     "비율 (개수)",
     4,
     "4.0, 2.0, 5.0"
    ],
    [
     "<code>retry_count</code>",
     "비율 (횟수)",
     0,
     "2, 0, 1"
    ],
    [
     "<code>input_chars</code>",
     "비율",
     6,
     "1549.0, 1349.0, 1578.0"
    ],
    [
     "<code>elapsed_ms</code>",
     "비율",
     0,
     "1994, 1799, 179"
    ]
   ],
   "notes": [
    "<code>failed</code>는 0과 1이라 숫자처럼 보이지만, 크기가 아니라 \"실패인가\"라는 이름표예요(명목). 평균 0.3은 \"실패 비율 30%\"로만 읽어요.",
    "<code>priority</code>는 low < normal < high 순서가 있지만, low와 normal의 차이가 normal과 high의 차이와 같다는 보장은 없어요(서열). 그래서 1, 2, 3으로 바꿔 평균 내는 건 조심해야 해요."
   ]
  },
  "s2": {
   "title": "② 처리 시간(ms) 100건 요약",
   "cols": [
    "무엇",
    "값"
   ],
   "rows": [
    [
     "평균",
     "1,277.7"
    ],
    [
     "중앙값",
     "1,295.5"
    ],
    [
     "10% 절사 평균",
     "1,280.5"
    ],
    [
     "범위",
     "2,549.0"
    ],
    [
     "사분위 범위",
     "1,010.2"
    ],
    [
     "분산 (n − 1로 나눔)",
     "413,947.1"
    ],
    [
     "표준편차",
     "643.4"
    ],
    [
     "변동계수",
     "0.504"
    ],
    [
     "왜도",
     "-0.032"
    ],
    [
     "첨도 (정규분포 = 0)",
     "-0.726"
    ]
   ],
   "notes": [
    "평균(1,278)과 중앙값(1,296)이 거의 같고 왜도가 0에 가까워요. 좌우가 대칭에 가까운 분포예요. 첨도가 음수(-0.73)라 정규분포보다 가운데가 덜 뾰족하고 꼬리가 짧아요.",
    "재시도 횟수는 왜도가 1.42로 오른쪽 꼬리가 길어요. 대부분 0이고 가끔 2, 3이에요. 이런 자료는 평균보다 중앙값이나 비율로 요약하는 게 더 정직해요."
   ]
  },
  "s2m": {
   "title": "평균이 하나가 아니에요",
   "cols": [
    "상황",
    "산술평균",
    "맞는 평균"
   ],
   "rows": [
    [
     "요청 수가 석 달 동안 1.10배, 1.50배, 0.80배",
     "1.133배",
     "기하평균 1.097배"
    ],
    [
     "같은 거리를 60km/h로 가고 40km/h로 오기",
     "50km/h",
     "조화평균 48km/h"
    ]
   ],
   "notes": [
    "석 달 뒤 요청 수는 1.10 × 1.50 × 0.80 = 1.32배예요. 매달 1.097배씩 늘었다고 하면 1.32배로 맞지만, 산술평균 1.133배로는 1.46배가 돼서 틀려요. 곱해서 쌓이는 비율은 기하평균이에요.",
    "왕복 120km를 가는 데 2시간, 오는 데 3시간이면 5시간에 120km라 평균 24km/h × 2 = 48km/h예요. 시간 대신 거리를 같게 둔 비율은 조화평균이에요."
   ]
  },
  "s3": {
   "title": "③ 확률로 읽기",
   "lines": [
    [
     "P(실패) = 0.30",
     1
    ],
    [
     "P(실패 | 우선순위) → high 0.344, low 0.316, normal 0.179",
     2,
     "newf"
    ],
    [
     "P(high 그리고 실패) = 0.11, P(high) × P(실패) = 0.096",
     3
    ],
    [
     "재시도 횟수: 평균 0.61, 분산 ÷ 평균 = 1.78",
     4,
     "newf"
    ]
   ],
   "notes": [
    "조건부 확률 P(A | B)는 \"B인 것들만 놓고 본 A의 비율\"이에요. high 요청의 실패율이 normal보다 높아 보이지만, 이 차이가 우연인지는 다음 쪽의 교차분석으로 따져요.",
    "두 사건이 독립이면 결합 확률이 각 확률의 곱과 같아야 해요. 0.11과 0.096은 가깝지만 똑같지는 않아요. 100건으로는 이 정도 차이가 흔해요.",
    "포아송 분포(일정한 비율로 드물게 일어나는 일의 횟수)는 평균과 분산이 같아요. 재시도는 분산이 평균의 1.8배라 포아송이 아니에요. 대부분은 한 번도 재시도하지 않고, 하는 요청은 여러 번 하는 두 무리가 섞여 있어요."
   ]
  },
  "f_dist": {
   "title": "두 열의 분포",
   "images": [
    [
     "img/stats/distributions.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽은 평균과 중앙값이 겹쳐요. 오른쪽은 실제(보라)가 0과 3에서 포아송(회색)보다 많고, 1에서 적어요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ 1년치 10만 건(진짜 평균 1,269.8ms)에서 100건씩 2,000번 뽑아 평균 내기",
   "cols": [
    "뽑는 방법",
    "2,000번 평균의 평균",
    "치우침",
    "흔들림 (표준편차)"
   ],
   "rows": [
    [
     "단순 무작위",
     "1,269.4",
     "-0.4",
     "63.8"
    ],
    [
     "편한 대로",
     "1,741.1",
     "+471.3",
     "73.4"
    ],
    [
     "층화",
     "1,270.2",
     "+0.4",
     "56.0"
    ]
   ],
   "notes": [
    "편한 대로(research 요청함에서만) 뽑으면 아무리 여러 번 뽑아도 평균이 471ms쯤 높게 나와요. 건수를 늘려도 고쳐지지 않는 치우침이에요. 모집단을 대표하지 못하는 표본은 일반화하면 안 돼요.",
    "층화(종류 비율대로 나눠 각각 무작위)는 치우침이 없으면서 흔들림이 64에서 56로 줄었어요. 종류마다 처리 시간이 달라서, 종류 비율이 표본마다 달라지는 흔들림을 없앤 거예요."
   ]
  },
  "f_samp": {
   "title": "표본을 뽑는 방법",
   "images": [
    [
     "img/stats/sampling.png",
     ""
    ]
   ],
   "notes": [
    "보라(단순 무작위)와 파랑(층화)은 진짜 평균 둘레에 있고, 파랑이 더 좁아요. 분홍(편한 대로)은 통째로 오른쪽에 있어요."
   ],
   "chart": true
  },
  "s5": {
   "title": "⑤ n건 표본의 평균을 5,000번 (모집단은 오른쪽 꼬리가 긴 분포)",
   "cols": [
    "n",
    "표본 평균들의 표준편차",
    "σ ÷ √n",
    "왜도"
   ],
   "rows": [
    [
     1,
     "653.7",
     "641.0",
     "1.536"
    ],
    [
     5,
     "285.8",
     "286.7",
     "0.690"
    ],
    [
     30,
     "116.7",
     "117.0",
     "0.250"
    ],
    [
     100,
     "64.7",
     "64.1",
     "0.171"
    ]
   ],
   "notes": [
    "표본 평균이 흔들리는 정도는 σ ÷ √n(<b>표준오차</b>)과 거의 같아요. n을 4배로 늘려야 흔들림이 반으로 줄어요.",
    "모집단은 오른쪽 꼬리가 길어서(n = 1의 왜도 1.54) 정규분포가 아니지만, n = 30이면 평균의 분포가 거의 대칭이에요. 이게 <b>중심극한정리</b>예요. 그래서 평균에 대한 검정과 구간은 원래 분포가 정규가 아니어도 대개 쓸 수 있어요."
   ]
  },
  "f_clt": {
   "title": "표본 평균의 분포 (분홍: 평균 μ, 표준편차 σ/√n인 정규분포)",
   "images": [
    [
     "img/stats/clt.png",
     ""
    ]
   ],
   "notes": [
    "n = 1은 모집단 그대로라 오른쪽으로 늘어져 있어요. n이 커질수록 분홍 정규 곡선에 붙고 좁아져요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ 95% 신뢰구간이 진짜 평균을 담은 비율 (2,000번)",
   "cols": [
    "n",
    "z로 (정규분포의 1.96)",
    "t로 (t분포의 값)"
   ],
   "rows": [
    [
     5,
     "84.7%",
     "91.9%"
    ],
    [
     30,
     "92.5%",
     "93.8%"
    ],
    [
     100,
     "93.7%",
     "94.4%"
    ]
   ],
   "notes": [
    "n = 5에서 z로 만든 \"95%\" 구간은 실제로 85%만 맞아요. 표준편차까지 표본으로 짐작하니 더 흔들리는데, z는 그걸 모르고 구간을 좁게 잡아요. t분포는 그만큼 꼬리가 두꺼워서 더 넓게 잡아요.",
    "t로 만들어도 n = 5에서는 92%로 95%에 못 미쳐요. t분포는 모집단이 정규분포라고 가정하는데, 이 모집단은 오른쪽 꼬리가 길어서예요. n이 커지면 둘 다 95%에 가까워져요.",
    "우리 100건의 평균 처리 시간에 대한 95% 신뢰구간은 1,150~1,405ms예요. \"이 방법으로 구간을 만들면 100번 중 95번은 진짜 평균을 담는다\"는 뜻이지, \"진짜 평균이 이 안에 있을 확률이 95%\"는 아니에요(진짜 평균은 정해진 하나의 값이라)."
   ]
  },
  "f_ci": {
   "title": "신뢰구간 60개",
   "images": [
    [
     "img/stats/confidence_intervals.png",
     ""
    ]
   ],
   "notes": [
    "구간은 표본마다 달라요. 그중 대부분(보라)은 진짜 평균을 지나고, 몇 개(분홍)는 빗나가요. 95%는 이 \"맞히는 비율\"에 대한 약속이에요."
   ],
   "chart": true
  }
 },
 "flow": [
  [
   1,
   "자료와 척도",
   "명목, 서열, 비율",
   "빈 칸"
  ],
  [
   2,
   "요약",
   "가운데, 퍼짐, 모양",
   "평균의 종류"
  ],
  [
   3,
   "확률",
   "조건부, 독립",
   "포아송 확인"
  ],
  [
   4,
   "표본 뽑기",
   "무작위, 편한 대로, 층화",
   "치우침"
  ],
  [
   5,
   "중심극한정리",
   "평균의 분포",
   "σ ÷ √n"
  ],
  [
   6,
   "신뢰구간",
   "95%의 뜻",
   "z 대 t"
  ]
 ],
 "env": "Python 3.13.5, pandas 2.3.3, SciPy 1.18.0, NumPy 2.5.2 (seed 0)"
};
