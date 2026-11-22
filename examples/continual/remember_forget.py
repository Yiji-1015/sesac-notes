# 손글씨 숫자로 "새로 배우면서 잊지 않기"(평생 학습)와 "배운 것을 골라 잊게 하기"(머신 언러닝)를 해 보는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: scikit-learn에 들어 있는 8 × 8 손글씨 숫자 1,797장 (받을 파일 없음). 두 번째 일은 같은 숫자의 픽셀 순서를 섞은 것이에요.
# CPU로 30초쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import copy                                                                     #@2
import numpy as np                                                              #@1
import torch                                                                    #@1
from torch import nn                                                            #@1
from sklearn.datasets import load_digits                                        #@1
from sklearn.model_selection import train_test_split                            #@1

# ① 숫자와 두 번째 일
torch.set_num_threads(1)
digits = load_digits()
X, y = (digits.data / 16).astype(np.float32), digits.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
shuffle = np.random.default_rng(0).permutation(64)              # 두 번째 일: 64픽셀의 자리를 섞기 (답은 같은 숫자)
T = lambda a, b: (torch.tensor(a), torch.tensor(b))             # noqa: E731
A, A_test = T(X_tr, y_tr), T(X_te, y_te)                        # 일 A: 원래 숫자
B, B_test = T(X_tr[:, shuffle], y_tr), T(X_te[:, shuffle], y_te)   # 일 B: 픽셀을 섞은 숫자

def model():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

def acc(m, data):
    with torch.no_grad():
        return (m(data[0]).argmax(dim=1) == data[1]).float().mean().item()

def train(m, data, epochs=30, lr=1e-2, penalty=None, replay=None, watch=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    curve = []
    for _ in range(epochs):
        for b in torch.randperm(len(data[1]), generator=g).split(64):
            x, t = data[0][b], data[1][b]
            if replay is not None:                              # 예전 일의 기록 몇 장을 섞어 함께 배우기
                r = torch.randint(len(replay[1]), (len(b),), generator=g)
                x, t = torch.cat([x, replay[0][r]]), torch.cat([t, replay[1][r]])
            loss = nn.functional.cross_entropy(m(x), t)
            if penalty is not None:
                loss = loss + penalty(m)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if watch is not None:
            curve.append(tuple(acc(m, w) for w in watch))       # 에폭마다 일 A, 일 B 정확도
    return m, curve

torch.manual_seed(0)
base, _ = train(model(), A)                                     # 일 A만 배운 모델
start = (acc(base, A_test), acc(base, B_test))

# ② 그냥 이어서 배우면: 파국적 망각
naive, naive_curve = train(copy.deepcopy(base), B, watch=(A_test, B_test))

# ③ 예전 것을 조금 다시 보기 (리플레이)
def replay_run(k, seed=1):
    keep = torch.randperm(len(A[1]), generator=torch.Generator().manual_seed(seed))[:k]   # 일 A에서 k장만 남겨 두기
    return train(copy.deepcopy(base), B, replay=(A[0][keep], A[1][keep]), watch=(A_test, B_test))

replay100, replay_curve = replay_run(100)

# ④ 중요한 가중치는 덜 움직이게: EWC
def fisher_diag(m, data):                                       # 가중치마다 "이 일에 얼마나 중요한가" (Fisher 정보의 대각선)
    F = [torch.zeros_like(p) for p in m.parameters()]
    for i in range(len(data[1])):
        m.zero_grad()
        logp = nn.functional.log_softmax(m(data[0][i:i + 1]), dim=1)
        k = torch.multinomial(logp.exp(), 1)[0]                 # 모델이 내는 분포에서 답 하나를 뽑아
        logp[0, k].sum().backward()                             # 그 log 확률의 기울기 제곱
        for f, p in zip(F, m.parameters()):
            f += p.grad.detach() ** 2
    return [f / len(data[1]) for f in F]

F_A = fisher_diag(base, A)
theta_A = [p.detach().clone() for p in base.parameters()]
def ewc(lam):
    return lambda m: lam / 2 * sum((f * (p - p0) ** 2).sum() for f, p, p0 in zip(F_A, m.parameters(), theta_A))

ewc1000, ewc_curve = train(copy.deepcopy(base), B, penalty=ewc(1000), watch=(A_test, B_test))

# ⑤ 같은 비용으로 비교: 처음부터 둘 다
torch.manual_seed(0)
joint, _ = train(model(), (torch.cat([A[0], B[0]]), torch.cat([A[1], B[1]])))   # 두 일을 처음부터 함께 (상한)
board = {name: (acc(m, A_test), acc(m, B_test)) for name, m in
         (("일 A만", base), ("그냥 이어서", naive), ("리플레이 100장", replay100), ("EWC λ = 1000", ewc1000), ("처음부터 둘 다", joint))}

# ⑥ 숫자 3을 지워 주세요: 처음부터 다시, 남은 것만 더
FORGET = 3
keep_tr, keep_te = y_tr != FORGET, y_te != FORGET
retain, forget = (A[0][keep_tr], A[1][keep_tr]), (A[0][~keep_tr], A[1][~keep_tr])   # 남길 데이터, 지울 데이터
retain_test, forget_test = (A_test[0][keep_te], A_test[1][keep_te]), (A_test[0][~keep_te], A_test[1][~keep_te])
def check(m):                                                   # 지울 숫자를 얼마나 맞히나, 나머지는 그대로인가
    return {"지울 숫자 (test)": acc(m, forget_test), "나머지 (test)": acc(m, retain_test), "지울 숫자 (학습에 쓴 것)": acc(m, forget)}

torch.manual_seed(0)
unlearn = {"원래 모델": check(base)}
retrained, _ = train(model(), retain)                           # 정답: 처음부터 지울 데이터 없이
unlearn["처음부터 다시"] = check(retrained)
more_retain, _ = train(copy.deepcopy(base), retain, epochs=3)   # 남은 데이터로 조금 더
unlearn["남은 것으로 3에폭 더"] = check(more_retain)

# ⑦ 빠르게 잊게 하기: 경사 상승과 작업 벡터 빼기
def gradient_ascent(m, steps=30, lr=1e-3, seed=0):              # 지울 데이터의 손실은 키우고, 남길 데이터의 손실은 줄이기
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        f = torch.randint(len(forget[1]), (32,), generator=g)
        r = torch.randint(len(retain[1]), (64,), generator=g)
        loss = -nn.functional.cross_entropy(m(forget[0][f]), forget[1][f]) + nn.functional.cross_entropy(m(retain[0][r]), retain[1][r])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m

unlearn["경사 상승 30걸음"] = check(gradient_ascent(copy.deepcopy(base)))
tuned, _ = train(copy.deepcopy(base), forget, epochs=10, lr=1e-3)   # 지울 데이터만 더 배운 모델
task_vector = [p1.detach() - p0.detach() for p0, p1 in zip(base.parameters(), tuned.parameters())]   # 그 차이 = "숫자 3 쪽으로 가는 방향"
def negate(alpha):
    m = copy.deepcopy(base)
    with torch.no_grad():
        for p, tv in zip(m.parameters(), task_vector):
            p -= alpha * tv                                     # 그 방향의 반대로
    return m

unlearn["작업 벡터 빼기 (α = 4)"] = check(negate(4.0))
