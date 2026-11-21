# 정답을 아는 작은 세상에서 "몇 건이면 믿을 수 있나"를 재 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: 파일 없이 코드가 직접 만들어요. 점은 [0, 1] × [0, 1]에 고르게 뿌리고, 정답 직사각형 안이면 1, 밖이면 0, 그리고 10%는 라벨을 뒤집어요.
# 진짜 규칙을 알기 때문에, 실제 데이터에서는 알 수 없는 "처음 보는 데이터에서의 오차"를 정확히 계산할 수 있어요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import itertools                                                                #@5
import math                                                                     #@2
import numpy as np                                                              #@1
from scipy.optimize import linprog                                              #@5
from sklearn.tree import DecisionTreeClassifier                                 #@6
from sklearn.model_selection import StratifiedKFold, cross_val_score            #@7

# ① 정답을 아는 세상 만들기
rng = np.random.default_rng(0)
TRUE = (0.2, 0.7, 0.3, 0.8)          # 정답 직사각형 (x₁ 범위, x₂ 범위). 넓이 0.25
ETA = 0.1                            # 라벨이 뒤집힐 확률 (잡음)

def inside(X, r):                    # 점들이 직사각형 r 안에 있는가
    return (X[:, 0] >= r[0]) & (X[:, 0] <= r[1]) & (X[:, 1] >= r[2]) & (X[:, 1] <= r[3])

def sample(m, eta=ETA):              # 이 세상에서 m건을 새로 뽑기 (i.i.d.)
    X = rng.random((m, 2))
    y = inside(X, TRUE).astype(int)
    flip = rng.random(m) < eta
    return X, np.where(flip, 1 - y, y)

def area(r):
    return max(0.0, r[1] - r[0]) * max(0.0, r[3] - r[2])

def true_err_rect(h, eta=ETA):       # "직사각형 h 안이면 1"이라는 모델의 진짜 오차 R(h). 넓이로 정확히
    both = area((max(h[0], TRUE[0]), min(h[1], TRUE[1]), max(h[2], TRUE[2]), min(h[3], TRUE[3])))
    differ = area(h) + area(TRUE) - 2 * both            # 정답과 답이 다른 곳의 넓이
    return eta + (1 - 2 * eta) * differ

g = (np.arange(500) + 0.5) / 500
GRID = np.array(np.meshgrid(g, g)).reshape(2, -1).T     # 25만 칸 격자. 아무 모델의 진짜 오차를 잴 때 써요
def true_err_model(model, eta=ETA):
    differ = (model.predict(GRID) != inside(GRID, TRUE)).mean()
    return eta + (1 - 2 * eta) * differ

bayes = true_err_rect(TRUE)                              # 정답 직사각형 자체의 오차 = 어떤 모델도 못 넘는 바닥
X_demo, y_demo = sample(200)

# ② 한 모델을 m건으로 재면: Hoeffding 부등식
h0 = (0.25, 0.75, 0.25, 0.75)                            # 미리 정해 둔 모델 하나 (데이터를 보고 고르지 않았어요)
R0 = true_err_rect(h0)
DELTA = 0.05                                             # "100번 중 5번은 틀려도 된다"
hoeff = {}
for m in (20, 100, 500):
    errs = np.array([(inside(X, h0).astype(int) != y).mean() for X, y in (sample(m) for _ in range(2000))])
    eps = math.sqrt(math.log(2 / DELTA) / (2 * m))       # Hoeffding이 주는 오차 막대
    hoeff[m] = {"errs": errs, "eps": eps, "outside": (np.abs(errs - R0) >= eps).mean()}

# ③ 후보가 많으면 고른 모델이 낙관적이 돼요: 유한한 모델 묶음
def random_rects(K):
    a, b = np.sort(rng.random((K, 2)), axis=1), np.sort(rng.random((K, 2)), axis=1)
    return np.column_stack([a[:, 0], a[:, 1], b[:, 0], b[:, 1]])

M3 = 50
finite = {}
for K in (1, 10, 100, 1000, 10000):
    gaps = []
    for _ in range(200):
        H = random_rects(K)                              # 후보 직사각형 K개
        X, y = sample(M3)
        pred = (X[None, :, 0] >= H[:, 0, None]) & (X[None, :, 0] <= H[:, 1, None]) & \
               (X[None, :, 1] >= H[:, 2, None]) & (X[None, :, 1] <= H[:, 3, None])
        emp = (pred != y[None, :]).mean(axis=1)          # 후보마다 50건에서의 오차
        k = emp.argmin()                                 # 50건에서 가장 좋아 보이는 후보 고르기
        gaps.append(true_err_rect(H[k]) - emp[k])        # 진짜 오차 − 보이는 오차 = 낙관한 정도
    finite[K] = {"gaps": np.array(gaps), "bound": math.sqrt((math.log(K) + math.log(2 / DELTA)) / (2 * M3))}

# ④ 오차 0인 모델이 있을 때: 직사각형 PAC 학습
rect = {}
for m in (10, 30, 100, 300, 1000):
    Rs = []
    for _ in range(1000):
        X, y = sample(m, eta=0)                          # 잡음 없는 세상
        P = X[y == 1]
        h = (P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()) if len(P) else (0, 0, 0, 0)
        Rs.append(true_err_rect(h, eta=0))               # 1인 점을 모두 감싸는 가장 작은 직사각형의 진짜 오차
    rect[m] = {"R": np.array(Rs), "bound": 4 / m * math.log(4 / DELTA)}

# ⑤ 무한한 모델 묶음: 점 몇 개까지 모든 라벨을 맞히나 (VC 차원)
def line_can(P, lab):                                    # 직선 하나로 이 라벨대로 가를 수 있나 (선형 계획 문제로 확인)
    s = np.where(lab, 1, -1)
    A = -(s[:, None] * np.column_stack([P, np.ones(len(P))]))
    return linprog(np.zeros(3), A_ub=A, b_ub=-np.ones(len(P)), bounds=[(None, None)] * 3, method="highs").status == 0

def rect_can(P, lab):                                    # 직사각형 하나로 가를 수 있나: 1인 점을 감싸는 가장 작은 직사각형에 0인 점이 없으면
    if not lab.any():
        return True
    Q = P[lab]
    r = (Q[:, 0].min(), Q[:, 0].max(), Q[:, 1].min(), Q[:, 1].max())
    return not inside(P[~lab], r).any()

def count_labelings(P, can):                             # 2ⁿ가지 라벨 중 맞힐 수 있는 것의 수
    return sum(can(P, np.array(l, dtype=bool)) for l in itertools.product([0, 1], repeat=len(P)))

shapes = {"삼각형 3점": np.array([[0, 0], [1, 0], [0, 1]]), "정사각형 4점": np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
          "마름모 4점": np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5]]), "마름모+가운데 5점": np.array([[0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5], [0.5, 0.5]])}
shatter = {name: (count_labelings(P, line_can), count_labelings(P, rect_can), 2 ** len(P)) for name, P in shapes.items()}
growth = {}
for n in (3, 4, 5, 6, 8, 10):
    P = rng.random((n, 2))
    growth[n] = (count_labelings(P, line_can), sum(math.comb(n, i) for i in range(4)), 2 ** n)   # 실제, Sauer 상한(d=3), 전부

# ⑥ 무작위 라벨을 얼마나 외우나: Rademacher 복잡도
def rademacher(X, depth, draws=30):
    vals = []
    for _ in range(draws):
        s = rng.choice([-1, 1], len(X))                  # 동전 던지기로 만든 가짜 라벨
        h = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X, s).predict(X)
        vals.append((s * h).mean())                      # 가짜 라벨과 맞은 비율 − 틀린 비율
    return float(np.mean(vals))

X7, y7 = sample(200)                                     # ⑦에서 쓸 200건
DEPTHS = (1, 2, 3, 4, 5, 6, 8, 12)
rad = {d: rademacher(X7, d) for d in DEPTHS}

# ⑦ 깊이 고르기: 근사 오차와 추정 오차, 그리고 고르는 방법 셋
X_big, y_big = sample(100_000)                           # "이 깊이로 만들 수 있는 가장 좋은 트리"를 찾으려고 아주 많이 뽑기
cv7 = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
X_tr, y_tr, X_va, y_va = X7[:150], y7[:150], X7[150:], y7[150:]
rows = {}
for d in DEPTHS:
    tree = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X7, y7)
    best = DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=0).fit(X_big, y_big)
    rows[d] = {"train": 1 - tree.score(X7, y7), "true": true_err_model(tree), "best": true_err_model(best),
               "valid": 1 - DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr).score(X_va, y_va),
               "cv": 1 - cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=0), X7, y7, cv=cv7).mean()}
    rows[d]["penalty"] = rows[d]["train"] + rad[d] + math.sqrt(math.log(1 / DELTA) / (2 * len(X7)))   # train 오차 + 복잡도 벌점
pick = {how: min(DEPTHS, key=lambda d: rows[d][how]) for how in ("train", "valid", "cv", "penalty", "true")}
