# 학습 중에 "어떤 평가로 진척을 볼까"를 정하는 전체 코드. 크기가 다른 모델 여섯 개를 에폭마다 저장하며 학습하고,
# 성격이 다른 평가(작은 것, 큰 것, 쉬운 것, 아직 너무 어려운 것)를 정답률과 손실 두 가지 지표로 재서 신호 대 잡음비를 비교해요.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: scikit-learn에 들어 있는 8 × 8 손글씨 숫자 1,797장 (받을 파일 없음). CPU로 1분쯤.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import numpy as np                                                              #@1
import torch                                                                    #@1
from torch import nn                                                            #@1
from sklearn.datasets import load_digits                                        #@1
from scipy.stats import spearmanr                                               #@5
from scipy.cluster.hierarchy import linkage, fcluster                           #@6

# ① 모델 여섯 개를 에폭마다 저장하며 학습
torch.set_num_threads(1)
digits = load_digits()
X = torch.tensor(digits.data / 16, dtype=torch.float32)
y = torch.tensor(digits.target)
order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
tr, pool = order[:1000], order[1000:]                            # 학습 1,000장, 나머지 797장에서 평가를 만들어요
WIDTHS = [8, 12, 16, 24, 32, 48]                                # 숨은 층 크기 = 모델 크기 (비슷한 크기끼리)
EPOCHS = 40

def train_with_checkpoints(width, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(64, width), nn.ReLU(), nn.Linear(width, 10))
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    g = torch.Generator().manual_seed(seed)
    checkpoints = []
    for _ in range(EPOCHS):
        for b in tr[torch.randperm(len(tr), generator=g)].split(32):
            loss = nn.functional.cross_entropy(m(X[b]), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        checkpoints.append({k: v.clone() for k, v in m.state_dict().items()})   # 에폭마다 저장
    return m, checkpoints

models = {w: train_with_checkpoints(w) for w in WIDTHS}

# ② 성격이 다른 평가 네 개, 지표 두 개
rng = np.random.default_rng(0)
pool_np = pool.numpy()
shuffle = torch.tensor(rng.permutation(64))
benchmarks = {
    "작은 평가 (40문제)": (X[pool_np[:40]], y[pool_np[:40]]),
    "큰 평가 (600문제)": (X[pool_np[100:700]], y[pool_np[100:700]]),
    "쉬운 평가 (0과 1만)": (X[pool_np[np.isin(y[pool_np].numpy(), [0, 1])]], y[pool_np[np.isin(y[pool_np].numpy(), [0, 1])]]),
    "아직 너무 어려운 평가 (픽셀 섞음)": (X[pool_np[100:700]][:, shuffle], y[pool_np[100:700]]),
}

def score(m, bench, metric):
    with torch.no_grad():
        logits = m(bench[0])
        if metric == "정답률":
            return (logits.argmax(1) == bench[1]).float().mean().item()
        return -nn.functional.cross_entropy(logits, bench[1]).item() / np.log(2)   # 정답에 준 확률의 -log₂ 평균의 음수 (클수록 좋음)

def curve(width, bench, metric):                                  # 에폭마다의 점수
    m = models[width][0]
    out = []
    for state in models[width][1]:
        m.load_state_dict(state)
        out.append(score(m, bench, metric))
    return np.array(out)

METRICS = ["정답률", "-손실 (비트)"]
curves = {(b, mt, w): curve(w, benchmarks[b], mt) for b in benchmarks for mt in METRICS for w in WIDTHS}

# ③ 신호 대 잡음비
LAST = 10                                                        # 마지막 체크포인트 10개
def snr(b, mt):
    finals = np.array([curves[(b, mt, w)][-LAST:].mean() for w in WIDTHS])
    signal = finals.max() - finals.min()                         # 신호: 모델끼리 얼마나 벌어지나
    noise = np.mean([curves[(b, mt, w)][-LAST:].std() for w in WIDTHS])   # 잡음: 같은 모델이 체크포인트마다 얼마나 흔들리나
    return signal, noise, signal / (noise + 1e-12)

snr_table = {(b, mt): snr(b, mt) for b in benchmarks for mt in METRICS}

# ④ 작은 평가가 흔들리는 이유: 문제 수
def sample_noise(n, reps=200):                                   # 같은 모델(마지막), 평가 문제를 n개씩 다시 뽑으면
    m = models[16][0]
    m.load_state_dict(models[16][1][-1])
    big_x, big_y = X[pool_np], y[pool_np]
    r = np.random.default_rng(1)
    vals = []
    for _ in range(reps):
        idx = r.choice(len(big_y), n, replace=False)
        vals.append(score(m, (big_x[idx], big_y[idx]), "정답률"))
    return float(np.std(vals))

size_noise = {n: sample_noise(n) for n in (20, 40, 100, 300, 700)}

# ⑤ 학습 초반에 잰 순서가 끝까지 맞나
TRUTH = np.array([curves[("큰 평가 (600문제)", "-손실 (비트)", w)][-LAST:].mean() for w in WIDTHS])   # 끝에서의 순서를 정답으로
def early_agreement(b, mt, epoch):
    early = np.array([curves[(b, mt, w)][epoch] for w in WIDTHS])
    return float(spearmanr(early, TRUTH)[0])

agree = {(b, mt): {e: early_agreement(b, mt, e) for e in (2, 5, 10)} for b in benchmarks for mt in METRICS}

# ⑥ 비슷한 평가끼리 묶기
names = [(b, mt) for b in benchmarks for mt in METRICS]
profile = np.array([np.concatenate([curves[(b, mt, w)] for w in WIDTHS]) for b, mt in names])   # 평가마다 "모든 모델, 모든 체크포인트의 점수"
corr = np.corrcoef(profile)
groups = fcluster(linkage(profile, method="average", metric="correlation"), t=3, criterion="maxclust")
