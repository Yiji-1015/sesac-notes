# 사용자 600명 × 아이템 400개의 가상 이용 기록으로 추천 모델 다섯 가지를 만들고 같은 지표로 비교하는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: 파일 없이 코드가 만들어요. 사용자마다 장르 취향이 있고, 아이템은 장르 하나와 인기도를 가져요. 진짜 취향을 알아서 "가장 잘해도 이만큼"을 잴 수 있어요.
# CPU로 1분쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import numpy as np                                                              #@1
import torch                                                                    #@4

# ① 이용 기록 만들고 나누기
rng = np.random.default_rng(0)
U, I, G = 600, 400, 8                                           # 사용자, 아이템, 장르 수
genre = rng.integers(G, size=I)                                 # 아이템마다 장르 하나
popularity = rng.pareto(1.2, I) + 1                             # 소수의 아이템이 아주 인기 있게 (멱법칙)
taste = rng.dirichlet(np.full(G, 0.3), size=U)                  # 사용자마다 장르 취향 (몇 장르에 몰림)
true_pref = taste[:, genre] * popularity[None, :] ** 0.6       # 진짜 선호 (실제로는 모르는 값)
true_prob = true_pref / true_pref.sum(axis=1, keepdims=True)
R = np.zeros((U, I), dtype=bool)                                # 본 적 있으면 True (암묵적 피드백)
for u in range(U):
    R[u, rng.choice(I, size=rng.poisson(25) + 5, replace=False, p=true_prob[u])] = True

train, test = R.copy(), np.zeros_like(R)                        # 사용자마다 기록의 30%를 숨겨 정답으로
for u in range(U):
    seen = rng.permutation(np.flatnonzero(R[u]))
    hide = seen[:round(len(seen) * 0.3)]
    test[u, hide], train[u, hide] = True, False
X = train.astype(np.float32)
density = R.mean()
top10_share = np.sort(R.sum(axis=0))[::-1][: I // 10].sum() / R.sum()   # 상위 10% 아이템이 차지한 기록 비율

# ② 평가 지표와 인기순 기준선
K = 10
DISCOUNT = 1 / np.log2(np.arange(2, K + 2))                     # 위에 있을수록 크게 쳐 주는 무게
def evaluate(scores):
    s = scores.astype(float).copy()
    s[train] = -np.inf                                          # 이미 본 것은 추천하지 않기
    top = np.argsort(-s, axis=1)[:, :K]
    out = {"정밀도@10": [], "재현율@10": [], "NDCG@10": []}
    for u in range(U):
        if not test[u].any():
            continue
        hit = test[u, top[u]]
        out["정밀도@10"].append(hit.mean())                     # 추천 10개 중 맞힌 비율
        out["재현율@10"].append(hit.sum() / test[u].sum())       # 숨긴 것 중 찾아낸 비율
        out["NDCG@10"].append((hit * DISCOUNT).sum() / DISCOUNT[: min(test[u].sum(), K)].sum())   # 순서까지
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["커버리지"] = len(np.unique(top)) / I                    # 한 번이라도 추천된 아이템 비율
    return res

results = {"인기순": evaluate(np.tile(X.sum(axis=0), (U, 1))), "진짜 선호 (천장)": evaluate(true_pref)}

# ③ 이웃으로 추천하기: 아이템 기반 협업 필터링
def cosine(A):                                                  # 열끼리 코사인 유사도
    n = np.linalg.norm(A, axis=0) + 1e-9
    S = (A.T @ A) / n[:, None] / n[None, :]
    np.fill_diagonal(S, 0)
    return S

item_sim = cosine(X)                                            # 같은 사람이 함께 본 아이템일수록 닮음
results["아이템 이웃"] = evaluate(X @ item_sim)
user_sim = cosine(X.T)
results["사용자 이웃"] = evaluate(user_sim @ X)

# ④ 숨은 취향 찾기: 행렬 분해 (BPR)
torch.set_num_threads(1)
def bpr_mf(dim=64, epochs=30, lr=0.01, reg=0.05, seed=0):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    P = torch.nn.Parameter(torch.randn(U, dim) * 0.1)           # 사용자 벡터
    Q = torch.nn.Parameter(torch.randn(I, dim) * 0.1)           # 아이템 벡터
    opt = torch.optim.Adam([P, Q], lr=lr)
    users, items = np.nonzero(train)
    for _ in range(epochs):
        for b in np.array_split(g.permutation(len(users)), len(users) // 1024):
            u, i = torch.tensor(users[b]), torch.tensor(items[b])
            j = torch.tensor(g.integers(I, size=len(b)))        # 안 본 것으로 치는 아이템 (음성 샘플)
            x = (P[u] * (Q[i] - Q[j])).sum(dim=1)               # 본 것 점수 − 안 본 것 점수
            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * (P[u].pow(2).sum() + Q[i].pow(2).sum() + Q[j].pow(2).sum()) / len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return (P @ Q.T).detach().numpy()

results["행렬 분해 (BPR)"] = evaluate(bpr_mf())

# ⑤ 닫힌 식 하나로: EASE
def ease(lam=200.0):                                            # 아이템 × 아이템 가중치 B를 한 번에 풀기 (대각선은 0)
    P = np.linalg.inv(X.T @ X + lam * np.eye(I))
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0)
    return X @ B

results["EASE"] = evaluate(ease())

# ⑥ 이용 기록을 그래프로: LightGCN
def lightgcn(dim=64, layers=2, epochs=30, lr=0.01, reg=0.01, seed=0):
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    A = np.zeros((U + I, U + I), dtype=np.float32)              # 사용자와 아이템을 노드로, 본 기록을 선으로
    A[:U, U:], A[U:, :U] = X, X.T
    deg = A.sum(axis=1)
    d = np.where(deg > 0, deg ** -0.5, 0)
    A_norm = torch.tensor(d[:, None] * A * d[None, :])          # GNN 쪽의 정규화한 인접 행렬과 같아요
    E = torch.nn.Parameter(torch.randn(U + I, dim) * 0.1)
    opt = torch.optim.Adam([E], lr=lr)
    def embed():                                                # 이웃 벡터를 layers번 모아 평균 (가중치, 활성화 없음)
        h, out = E, [E]
        for _ in range(layers):
            h = A_norm @ h
            out.append(h)
        return torch.stack(out).mean(dim=0)
    users, items = np.nonzero(train)
    for _ in range(epochs):
        for b in np.array_split(g.permutation(len(users)), len(users) // 1024):
            F = embed()
            u, i, j = torch.tensor(users[b]), torch.tensor(items[b]) + U, torch.tensor(g.integers(I, size=len(b))) + U
            x = (F[u] * (F[i] - F[j])).sum(dim=1)
            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * (E[u].pow(2).sum() + E[i].pow(2).sum() + E[j].pow(2).sum()) / len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
    F = embed().detach().numpy()
    return F[:U] @ F[U:].T

results["LightGCN"] = evaluate(lightgcn())

# ⑦ 기록이 적은 사용자와 인기 편향
activity = train.sum(axis=1)
lo, hi = np.quantile(activity, [1 / 3, 2 / 3]).astype(int)     # 사용자 수가 비슷하게 셋으로
groups = {f"기록 {lo}개 이하": activity <= lo, f"{lo + 1}~{hi}개": (activity > lo) & (activity <= hi), f"{hi + 1}개 이상": activity > hi}
def ndcg_by_group(scores):
    s = scores.astype(float).copy()
    s[train] = -np.inf
    top = np.argsort(-s, axis=1)[:, :K]
    nd = np.array([(test[u, top[u]] * DISCOUNT).sum() / DISCOUNT[: max(1, min(test[u].sum(), K))].sum() for u in range(U)])
    return {name: float(nd[m].mean()) for name, m in groups.items()}

by_group = {"인기순": ndcg_by_group(np.tile(X.sum(axis=0), (U, 1))), "아이템 이웃": ndcg_by_group(X @ item_sim), "EASE": ndcg_by_group(ease())}
head = np.argsort(-X.sum(axis=0))[: I // 10]                    # 인기 상위 10% 아이템
def head_share(scores):                                         # 추천 목록 중 인기 상위 10% 아이템의 비율
    s = scores.astype(float).copy()
    s[train] = -np.inf
    top = np.argsort(-s, axis=1)[:, :K]
    return float(np.isin(top, head).mean())

popularity_bias = {"인기순": head_share(np.tile(X.sum(axis=0), (U, 1))), "아이템 이웃": head_share(X @ item_sim), "EASE": head_share(ease()),
                   "행렬 분해 (BPR)": head_share(bpr_mf()), "진짜 선호": head_share(true_pref)}
