# 점 데이터를 닮은 새 점 만들기 (생성 모델 다섯 가지) 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/points.csv (딥러닝 기초와 같은 점 400개). train 240개로 배우고, 나머지 160개로 "처음 보는 진짜 점"을 재요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import math                                                      #@1
import pandas as pd                                              #@1
import torch                                                     #@1
from torch import nn                                             #@1
from torch.distributions import MultivariateNormal               #@2

DATA = Path(__file__).resolve().parent / "data" / "points.csv"  #@1
N = 1000  # 모델마다 새로 만들 점 수                                    #@1

# ① 데이터와 재는 방법 준비
df = pd.read_csv(DATA)
X_all = torch.tensor(df[["x1", "x2"]].values, dtype=torch.float32)
is_train = torch.tensor((df["split"] == "train").values)
mu, sd = X_all[is_train].mean(0), X_all[is_train].std(0)
Z = (X_all[is_train] - mu) / sd          # train 240개, 평균 0, 표준편차 1로
Z_test = (X_all[~is_train] - mu) / sd    # 처음 보는 진짜 점 160개

def mlp(n_in, n_out, hidden=64):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

def quality(S, R=Z_test, k=3):
    # 정밀도: 만든 점 중 진짜 점들 근처(진짜 점의 k번째 이웃 거리 안)에 떨어진 비율. 낮으면 엉뚱한 곳에 만든 것
    # 재현율: 진짜 점 중 만든 점들 근처에 있는 비율. 낮으면 진짜 데이터의 일부만 흉내 낸 것
    r_real = torch.cdist(R, R).kthvalue(k + 1, dim=1).values
    r_fake = torch.cdist(S, S).kthvalue(k + 1, dim=1).values
    precision = (torch.cdist(S, R) <= r_real[None]).any(1).float().mean().item()
    recall = (torch.cdist(R, S) <= r_fake[None]).any(1).float().mean().item()
    return {"precision": round(precision, 3), "recall": round(recall, 3)}

samples, scores = {}, {}

# ② 기준선: 정규분포 하나
gauss = MultivariateNormal(Z.mean(0), torch.cov(Z.T))      # 평균과 공분산만 맞춘 타원 하나
torch.manual_seed(0)
samples["정규분포"] = gauss.sample((N,))
scores["정규분포"] = {**quality(samples["정규분포"]), "test_loglik": round(gauss.log_prob(Z_test).mean().item(), 3)}

# ③ VAE: 줄였다가 되살리기 + 잠재 공간을 정규분포로
LATENT, SIGMA2 = 2, 0.05                        # 잠재 변수 칸 수, 되살릴 때 허용하는 오차의 분산
torch.manual_seed(0)
enc, dec = mlp(2, 2 * LATENT), mlp(LATENT, 2)
opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3)
for _ in range(4000):
    m, logv = enc(Z).chunk(2, dim=1)                    # 점마다 잠재 변수의 평균과 log 분산
    z = m + torch.exp(0.5 * logv) * torch.randn_like(m)  # 재파라미터화: 무작위를 바깥으로 빼서 기울기가 흐르게
    recon = ((dec(z) - Z) ** 2).sum(1) / (2 * SIGMA2)   # 되살리기 오차
    kl = 0.5 * (m ** 2 + logv.exp() - 1 - logv).sum(1)  # 잠재 분포가 표준정규분포에서 얼마나 먼가
    vae_loss = (recon + kl).mean()                      # = −ELBO (상수 제외)
    opt.zero_grad(); vae_loss.backward(); opt.step()
with torch.no_grad():
    torch.manual_seed(1)
    samples["VAE"] = dec(torch.randn(N, LATENT)) + math.sqrt(SIGMA2) * torch.randn(N, 2)
scores["VAE"] = quality(samples["VAE"])

# ④ GAN: 만드는 쪽과 가려내는 쪽의 시합
torch.manual_seed(0)
G, D = mlp(2, 2), mlp(2, 1)
opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
gan_log = []
for step in range(4000):
    fake = G(torch.randn(len(Z), 2))
    d_loss = bce(D(Z), torch.ones(len(Z), 1)) + bce(D(fake.detach()), torch.zeros(len(Z), 1))  # 진짜는 1, 가짜는 0
    opt_d.zero_grad(); d_loss.backward(); opt_d.step()
    g_loss = bce(D(fake), torch.ones(len(Z), 1))       # 가려내는 쪽이 가짜를 1이라고 믿게
    opt_g.zero_grad(); g_loss.backward(); opt_g.step()
    if step % 100 == 0:
        gan_log.append({"step": step, "d_loss": d_loss.item(), "g_loss": g_loss.item()})
with torch.no_grad():
    torch.manual_seed(1)
    samples["GAN"] = G(torch.randn(N, 2))
scores["GAN"] = quality(samples["GAN"])

# ⑤ Normalizing Flow: 되돌릴 수 있는 변환을 겹쳐 정확한 확률 계산
class Coupling(nn.Module):  # 한 칸은 그대로 두고, 그 칸을 보고 다른 칸을 늘이고(s) 옮기기(t)
    def __init__(self, keep):
        super().__init__()
        self.keep = keep
        self.net = mlp(1, 2)

    def forward(self, x):  # 데이터 → 단순한 분포 쪽
        a, b = x[:, [self.keep]], x[:, [1 - self.keep]]
        s, t = self.net(a).chunk(2, dim=1)
        s = torch.tanh(s)                                  # 너무 크게 늘이지 않게
        out = torch.empty_like(x)
        out[:, [self.keep]], out[:, [1 - self.keep]] = a, b * s.exp() + t
        return out, s.sum(1)                               # 부피가 몇 배 변했나의 log

    def inverse(self, y):  # 단순한 분포 → 데이터 쪽
        a, b = y[:, [self.keep]], y[:, [1 - self.keep]]
        s, t = self.net(a).chunk(2, dim=1)
        s = torch.tanh(s)
        out = torch.empty_like(y)
        out[:, [self.keep]], out[:, [1 - self.keep]] = a, (b - t) * (-s).exp()
        return out

torch.manual_seed(0)
flows = nn.ModuleList([Coupling(i % 2) for i in range(8)])
base = MultivariateNormal(torch.zeros(2), torch.eye(2))

def flow_loglik(x):
    total = torch.zeros(len(x))
    for f in flows:
        x, logdet = f(x)
        total = total + logdet
    return base.log_prob(x) + total                        # 변수 바꾸기 공식

opt = torch.optim.Adam(flows.parameters(), lr=1e-3)
for _ in range(4000):
    flow_loss = -flow_loglik(Z).mean()                     # 진짜 점의 확률을 크게
    opt.zero_grad(); flow_loss.backward(); opt.step()
with torch.no_grad():
    torch.manual_seed(1)
    y = base.sample((N,))
    for f in reversed(flows):
        y = f.inverse(y)
    samples["Flow"] = y
    scores["Flow"] = {**quality(samples["Flow"]), "test_loglik": round(flow_loglik(Z_test).mean().item(), 3)}

# ⑥ Diffusion: 잡음을 조금씩 섞었다가 거꾸로 걷어내기
T = 100
betas = torch.linspace(1e-4, 0.1, T)                       # 걸음마다 섞을 잡음의 양
alpha_bar = torch.cumprod(1 - betas, dim=0)                # t 걸음 뒤 원래 점이 남은 몫
torch.manual_seed(0)
eps_net = mlp(3, 2, hidden=128)                            # (점 x, y, 시점 t) → 섞인 잡음 맞히기
opt = torch.optim.Adam(eps_net.parameters(), lr=1e-3)
for _ in range(8000):
    t = torch.randint(0, T, (len(Z),))
    eps = torch.randn_like(Z)
    ab = alpha_bar[t][:, None]
    x_t = ab.sqrt() * Z + (1 - ab).sqrt() * eps            # 한 번에 t 걸음만큼 잡음 섞기
    diff_loss = ((eps_net(torch.cat([x_t, t[:, None] / T], 1)) - eps) ** 2).mean()
    opt.zero_grad(); diff_loss.backward(); opt.step()
with torch.no_grad():
    torch.manual_seed(1)
    x = torch.randn(N, 2)                                  # 순수한 잡음에서 시작
    for t in reversed(range(T)):
        e = eps_net(torch.cat([x, torch.full((N, 1), t / T)], 1))
        x = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * e) / (1 - betas[t]).sqrt()  # 잡음 조금 걷어내기
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)  # 마지막 걸음 전까지는 조금 흔들기
    samples["Diffusion"] = x
scores["Diffusion"] = quality(samples["Diffusion"])

# ⑦ 같은 기준으로 비교하기
real_vs_real = quality(Z, R=Z_test)                         # 진짜 train 점으로 재면 얼마가 나오나 (상한 참고)
board = pd.DataFrame(scores).T
