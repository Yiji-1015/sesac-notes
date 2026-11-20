window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import math",
   "s": 1
  },
  {
   "t": "import pandas as pd",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 1
  },
  {
   "t": "from torch import nn",
   "s": 1
  },
  {
   "t": "from torch.distributions import MultivariateNormal",
   "s": 2
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\" / \"points.csv\"",
   "s": 1
  },
  {
   "t": "N = 1000  # 모델마다 새로 만들 점 수",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 데이터와 재는 방법 준비",
   "s": 1
  },
  {
   "t": "df = pd.read_csv(DATA)",
   "s": 1
  },
  {
   "t": "X_all = torch.tensor(df[[\"x1\", \"x2\"]].values, dtype=torch.float32)",
   "s": 1
  },
  {
   "t": "is_train = torch.tensor((df[\"split\"] == \"train\").values)",
   "s": 1
  },
  {
   "t": "mu, sd = X_all[is_train].mean(0), X_all[is_train].std(0)",
   "s": 1
  },
  {
   "t": "Z = (X_all[is_train] - mu) / sd  # train 240개, 평균 0, 표준편차 1로",
   "s": 1
  },
  {
   "t": "Z_test = (X_all[~is_train] - mu) / sd  # 처음 보는 진짜 점 160개",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def mlp(n_in, n_out, hidden=64):",
   "s": 1
  },
  {
   "t": "    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def quality(S, R=Z_test, k=3):",
   "s": 1
  },
  {
   "t": "    # 정밀도: 만든 점 중 진짜 점들 근처(진짜 점의 k번째 이웃 거리 안)에 떨어진 비율. 낮으면 엉뚱한 곳에 만든 것",
   "s": 1
  },
  {
   "t": "    # 재현율: 진짜 점 중 만든 점들 근처에 있는 비율. 낮으면 진짜 데이터의 일부만 흉내 낸 것",
   "s": 1
  },
  {
   "t": "    r_real = torch.cdist(R, R).kthvalue(k + 1, dim=1).values",
   "s": 1
  },
  {
   "t": "    r_fake = torch.cdist(S, S).kthvalue(k + 1, dim=1).values",
   "s": 1
  },
  {
   "t": "    precision = (torch.cdist(S, R) <= r_real[None]).any(1).float().mean().item()",
   "s": 1
  },
  {
   "t": "    recall = (torch.cdist(R, S) <= r_fake[None]).any(1).float().mean().item()",
   "s": 1
  },
  {
   "t": "    return {\"precision\": round(precision, 3), \"recall\": round(recall, 3)}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "samples, scores = {}, {}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 기준선: 정규분포 하나",
   "s": 2
  },
  {
   "t": "gauss = MultivariateNormal(Z.mean(0), torch.cov(Z.T))  # 평균과 공분산만 맞춘 타원 하나",
   "s": 2
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 2
  },
  {
   "t": "samples[\"정규분포\"] = gauss.sample((N,))",
   "s": 2
  },
  {
   "t": "scores[\"정규분포\"] = {**quality(samples[\"정규분포\"]), \"test_loglik\": round(gauss.log_prob(Z_test).mean().item(), 3)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ VAE: 줄였다가 되살리기 + 잠재 공간을 정규분포로",
   "s": 3
  },
  {
   "t": "LATENT, SIGMA2 = 2, 0.05  # 잠재 변수 칸 수, 되살릴 때 허용하는 오차의 분산",
   "s": 3
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 3
  },
  {
   "t": "enc, dec = mlp(2, 2 * LATENT), mlp(LATENT, 2)",
   "s": 3
  },
  {
   "t": "opt = torch.optim.Adam([*enc.parameters(), *dec.parameters()], lr=1e-3)",
   "s": 3
  },
  {
   "t": "for _ in range(4000):",
   "s": 3
  },
  {
   "t": "    m, logv = enc(Z).chunk(2, dim=1)  # 점마다 잠재 변수의 평균과 log 분산",
   "s": 3
  },
  {
   "t": "    z = m + torch.exp(0.5 * logv) * torch.randn_like(m)  # 재파라미터화: 무작위를 바깥으로 빼서 기울기가 흐르게",
   "s": 3
  },
  {
   "t": "    recon = ((dec(z) - Z) ** 2).sum(1) / (2 * SIGMA2)  # 되살리기 오차",
   "s": 3
  },
  {
   "t": "    kl = 0.5 * (m ** 2 + logv.exp() - 1 - logv).sum(1)  # 잠재 분포가 표준정규분포에서 얼마나 먼가",
   "s": 3
  },
  {
   "t": "    vae_loss = (recon + kl).mean()  # = −ELBO (상수 제외)",
   "s": 3
  },
  {
   "t": "    opt.zero_grad(); vae_loss.backward(); opt.step()",
   "s": 3
  },
  {
   "t": "with torch.no_grad():",
   "s": 3
  },
  {
   "t": "    torch.manual_seed(1)",
   "s": 3
  },
  {
   "t": "    samples[\"VAE\"] = dec(torch.randn(N, LATENT)) + math.sqrt(SIGMA2) * torch.randn(N, 2)",
   "s": 3
  },
  {
   "t": "scores[\"VAE\"] = quality(samples[\"VAE\"])",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ GAN: 만드는 쪽과 가려내는 쪽의 시합",
   "s": 4
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 4
  },
  {
   "t": "G, D = mlp(2, 2), mlp(2, 1)",
   "s": 4
  },
  {
   "t": "opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))",
   "s": 4
  },
  {
   "t": "opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))",
   "s": 4
  },
  {
   "t": "bce = nn.BCEWithLogitsLoss()",
   "s": 4
  },
  {
   "t": "gan_log = []",
   "s": 4
  },
  {
   "t": "for step in range(4000):",
   "s": 4
  },
  {
   "t": "    fake = G(torch.randn(len(Z), 2))",
   "s": 4
  },
  {
   "t": "    d_loss = bce(D(Z), torch.ones(len(Z), 1)) + bce(D(fake.detach()), torch.zeros(len(Z), 1))  # 진짜는 1, 가짜는 0",
   "s": 4
  },
  {
   "t": "    opt_d.zero_grad(); d_loss.backward(); opt_d.step()",
   "s": 4
  },
  {
   "t": "    g_loss = bce(D(fake), torch.ones(len(Z), 1))  # 가려내는 쪽이 가짜를 1이라고 믿게",
   "s": 4
  },
  {
   "t": "    opt_g.zero_grad(); g_loss.backward(); opt_g.step()",
   "s": 4
  },
  {
   "t": "    if step % 100 == 0:",
   "s": 4
  },
  {
   "t": "        gan_log.append({\"step\": step, \"d_loss\": d_loss.item(), \"g_loss\": g_loss.item()})",
   "s": 4
  },
  {
   "t": "with torch.no_grad():",
   "s": 4
  },
  {
   "t": "    torch.manual_seed(1)",
   "s": 4
  },
  {
   "t": "    samples[\"GAN\"] = G(torch.randn(N, 2))",
   "s": 4
  },
  {
   "t": "scores[\"GAN\"] = quality(samples[\"GAN\"])",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ Normalizing Flow: 되돌릴 수 있는 변환을 겹쳐 정확한 확률 계산",
   "s": 5
  },
  {
   "t": "class Coupling(nn.Module):  # 한 칸은 그대로 두고, 그 칸을 보고 다른 칸을 늘이고(s) 옮기기(t)",
   "s": 5
  },
  {
   "t": "    def __init__(self, keep):",
   "s": 5
  },
  {
   "t": "        super().__init__()",
   "s": 5
  },
  {
   "t": "        self.keep = keep",
   "s": 5
  },
  {
   "t": "        self.net = mlp(1, 2)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "    def forward(self, x):  # 데이터 → 단순한 분포 쪽",
   "s": 5
  },
  {
   "t": "        a, b = x[:, [self.keep]], x[:, [1 - self.keep]]",
   "s": 5
  },
  {
   "t": "        s, t = self.net(a).chunk(2, dim=1)",
   "s": 5
  },
  {
   "t": "        s = torch.tanh(s)  # 너무 크게 늘이지 않게",
   "s": 5
  },
  {
   "t": "        out = torch.empty_like(x)",
   "s": 5
  },
  {
   "t": "        out[:, [self.keep]], out[:, [1 - self.keep]] = a, b * s.exp() + t",
   "s": 5
  },
  {
   "t": "        return out, s.sum(1)  # 부피가 몇 배 변했나의 log",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "    def inverse(self, y):  # 단순한 분포 → 데이터 쪽",
   "s": 5
  },
  {
   "t": "        a, b = y[:, [self.keep]], y[:, [1 - self.keep]]",
   "s": 5
  },
  {
   "t": "        s, t = self.net(a).chunk(2, dim=1)",
   "s": 5
  },
  {
   "t": "        s = torch.tanh(s)",
   "s": 5
  },
  {
   "t": "        out = torch.empty_like(y)",
   "s": 5
  },
  {
   "t": "        out[:, [self.keep]], out[:, [1 - self.keep]] = a, (b - t) * (-s).exp()",
   "s": 5
  },
  {
   "t": "        return out",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "flows = nn.ModuleList([Coupling(i % 2) for i in range(8)])",
   "s": 5
  },
  {
   "t": "base = MultivariateNormal(torch.zeros(2), torch.eye(2))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def flow_loglik(x):",
   "s": 5
  },
  {
   "t": "    total = torch.zeros(len(x))",
   "s": 5
  },
  {
   "t": "    for f in flows:",
   "s": 5
  },
  {
   "t": "        x, logdet = f(x)",
   "s": 5
  },
  {
   "t": "        total = total + logdet",
   "s": 5
  },
  {
   "t": "    return base.log_prob(x) + total  # 변수 바꾸기 공식",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "opt = torch.optim.Adam(flows.parameters(), lr=1e-3)",
   "s": 5
  },
  {
   "t": "for _ in range(4000):",
   "s": 5
  },
  {
   "t": "    flow_loss = -flow_loglik(Z).mean()  # 진짜 점의 확률을 크게",
   "s": 5
  },
  {
   "t": "    opt.zero_grad(); flow_loss.backward(); opt.step()",
   "s": 5
  },
  {
   "t": "with torch.no_grad():",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(1)",
   "s": 5
  },
  {
   "t": "    y = base.sample((N,))",
   "s": 5
  },
  {
   "t": "    for f in reversed(flows):",
   "s": 5
  },
  {
   "t": "        y = f.inverse(y)",
   "s": 5
  },
  {
   "t": "    samples[\"Flow\"] = y",
   "s": 5
  },
  {
   "t": "    scores[\"Flow\"] = {**quality(samples[\"Flow\"]), \"test_loglik\": round(flow_loglik(Z_test).mean().item(), 3)}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ Diffusion: 잡음을 조금씩 섞었다가 거꾸로 걷어내기",
   "s": 6
  },
  {
   "t": "T = 100",
   "s": 6
  },
  {
   "t": "betas = torch.linspace(1e-4, 0.1, T)  # 걸음마다 섞을 잡음의 양",
   "s": 6
  },
  {
   "t": "alpha_bar = torch.cumprod(1 - betas, dim=0)  # t 걸음 뒤 원래 점이 남은 몫",
   "s": 6
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "eps_net = mlp(3, 2, hidden=128)  # (점 x, y, 시점 t) → 섞인 잡음 맞히기",
   "s": 6
  },
  {
   "t": "opt = torch.optim.Adam(eps_net.parameters(), lr=1e-3)",
   "s": 6
  },
  {
   "t": "for _ in range(8000):",
   "s": 6
  },
  {
   "t": "    t = torch.randint(0, T, (len(Z),))",
   "s": 6
  },
  {
   "t": "    eps = torch.randn_like(Z)",
   "s": 6
  },
  {
   "t": "    ab = alpha_bar[t][:, None]",
   "s": 6
  },
  {
   "t": "    x_t = ab.sqrt() * Z + (1 - ab).sqrt() * eps  # 한 번에 t 걸음만큼 잡음 섞기",
   "s": 6
  },
  {
   "t": "    diff_loss = ((eps_net(torch.cat([x_t, t[:, None] / T], 1)) - eps) ** 2).mean()",
   "s": 6
  },
  {
   "t": "    opt.zero_grad(); diff_loss.backward(); opt.step()",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    torch.manual_seed(1)",
   "s": 6
  },
  {
   "t": "    x = torch.randn(N, 2)  # 순수한 잡음에서 시작",
   "s": 6
  },
  {
   "t": "    for t in reversed(range(T)):",
   "s": 6
  },
  {
   "t": "        e = eps_net(torch.cat([x, torch.full((N, 1), t / T)], 1))",
   "s": 6
  },
  {
   "t": "        x = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * e) / (1 - betas[t]).sqrt()  # 잡음 조금 걷어내기",
   "s": 6
  },
  {
   "t": "        if t > 0:",
   "s": 6
  },
  {
   "t": "            x = x + betas[t].sqrt() * torch.randn_like(x)  # 마지막 걸음 전까지는 조금 흔들기",
   "s": 6
  },
  {
   "t": "    samples[\"Diffusion\"] = x",
   "s": 6
  },
  {
   "t": "scores[\"Diffusion\"] = quality(samples[\"Diffusion\"])",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 같은 기준으로 비교하기",
   "s": 7
  },
  {
   "t": "real_vs_real = quality(Z, R=Z_test)  # 진짜 train 점으로 재면 얼마가 나오나 (상한 참고)",
   "s": 7
  },
  {
   "t": "board = pd.DataFrame(scores).T",
   "s": 7
  }
 ],
 "names": {
  "1": "데이터와 재는 방법 준비",
  "2": "기준선: 정규분포 하나",
  "3": "VAE: 줄였다가 되살리기 + 잠재 공간을 정규분포로",
  "4": "GAN: 만드는 쪽과 가려내는 쪽의 시합",
  "5": "Normalizing Flow: 되돌릴 수 있는 변환을 겹쳐 정확한 확률 계산",
  "6": "Diffusion: 잡음을 조금씩 섞었다가 거꾸로 걷어내기",
  "7": "같은 기준으로 비교하기"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "data": {
   "title": "배울 데이터 (딥러닝 기초와 같은 점)",
   "images": [
    [
     "img/gen/data.png",
     ""
    ]
   ],
   "notes": [
    "두 개의 휜 띠 모양이에요. 생성 모델은 무리 번호(label)는 보지 않고 점의 좌표만 보고, 이런 모양의 점을 새로 만들어야 해요."
   ],
   "chart": true
  },
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    "Z.shape → (240, 2) (train), Z_test.shape → (160, 2)",
    [
     "quality(train 점, 처음 보는 점) → 정밀도 0.942, 재현율 0.988",
     1,
     "newf"
    ]
   ],
   "notes": [
    "\"진짜 train 점\"을 만든 점이라고 치고 재 본 값이에요. 완벽한 생성 모델이라도 이 근처가 최선이에요. 아래 모델들의 점수는 이 값과 비교해요."
   ]
  },
  "s2": {
   "title": "② 정규분포 하나",
   "lines": [
    "평균 → [-0.0, 0.0]",
    "공분산 → [[1.0, -0.42], [-0.42, 1.0]]",
    [
     "정밀도 0.832, 재현율 0.962",
     1
    ],
    [
     "처음 보는 점의 평균 log 확률 → -2.622",
     2,
     "newf"
    ]
   ],
   "notes": [
    "재현율은 높은데 정밀도가 낮아요. 타원 하나로 띠 두 개를 덮으니, 띠 사이 빈 곳에도 점을 많이 만들어요.",
    "모델이 \"이 점이 나올 확률(밀도)\"을 직접 계산할 수 있어서 처음 보는 점에 대한 log 확률을 잴 수 있어요. 높을수록 진짜 점을 그럴듯하다고 보는 모델이에요. ⑤의 Flow와 비교해요."
   ]
  },
  "f_vae": {
   "title": "VAE의 잠재 공간과 만든 점",
   "images": [
    [
     "img/gen/vae.png",
     "점선 원: 표준정규분포의 1, 2 표준편차"
    ]
   ],
   "notes": [
    "왼쪽: 두 무리가 잠재 공간에서도 대체로 갈라져 있고, 전체가 원점 둘레의 원 안에 모였어요. KL 항이 잠재 공간을 표준정규분포 쪽으로 당긴 결과예요. 무리 번호는 학습에 쓰지 않았어요.",
    "오른쪽: 표준정규분포에서 아무 점이나 뽑아 decoder에 넣었더니, 두 띠가 있는 자리를 대략 덮는 점들이 나왔어요. 다만 경계가 흐릿해서 띠 사이 빈 곳에도 점이 꽤 있어요. 잠재 공간을 미리 정해 둔 분포로 맞춰 놨기 때문에 \"어디서 뽑아야 하나\"를 고민하지 않아도 되는 게 장점이에요."
   ],
   "chart": true
  },
  "t3": {
   "title": "되살리기 오차의 분산 SIGMA2를 키우면",
   "cols": [
    "SIGMA2",
    "정밀도",
    "재현율"
   ],
   "rows": [
    [
     "<b>0.05</b> (지금)",
     0.896,
     0.981
    ],
    [
     0.5,
     0.855,
     0.95
    ]
   ],
   "notes": [
    "SIGMA2가 크면 \"되살린 점이 원래 점과 좀 달라도 괜찮다\"는 뜻이라 되살리기 항의 무게가 줄고 KL 항이 상대적으로 세져요. decoder가 뭉개진 평균만 내고, 만든 점에 더하는 잡음도 커져서 띠 바깥에 점이 많아져요(정밀도가 떨어져요).",
    "그림에서 VAE가 흐릿하다는 말이 이거예요. 되살리기를 제곱 오차(정규분포)로 재면 \"여러 그럴듯한 답의 평균\"을 내기 쉬워요."
   ]
  },
  "s3": {
   "title": "③ VAE",
   "lines": [
    "마지막 손실(−ELBO) → 3.616",
    [
     "정밀도 0.896, 재현율 0.981",
     1
    ]
   ],
   "notes": [
    "정규분포 하나보다 정밀도가 올라갔어요. 띠 사이 빈 곳에 만드는 점이 줄었어요."
   ]
  },
  "f_gan": {
   "title": "GAN의 학습 기록과 만든 점",
   "images": [
    [
     "img/gen/gan.png",
     ""
    ]
   ],
   "notes": [
    "보통 학습과 달리 두 손실이 0으로 내려가지 않아요. 한쪽이 좋아지면 다른 쪽이 나빠지는 시합이라, D가 \"반반으로 찍는\" 상태(2 log 2 ≈ 1.39) 근처에서 오르내려요. 그래서 손실만 보고는 GAN이 잘 배웠는지 알 수 없어요.",
    "오른쪽: 점이 띠 위의 가는 선 몇 줄로 몰렸어요. 그 선은 진짜 점 사이에 있어서 정밀도는 높지만, 진짜 점은 띠 폭 전체에 퍼져 있어서 바깥쪽 점들 근처에는 만든 점이 없어요(재현율 낮음). 진짜 데이터의 일부만 골라 흉내 내는 쪽으로 쏠리는 걸 모드 붕괴라고 해요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ GAN",
   "lines": [
    "마지막 D 손실 1.376, G 손실 0.700",
    [
     "정밀도 0.962, 재현율 0.744",
     1,
     "bad"
    ]
   ],
   "notes": [
    "정밀도는 다섯 모델 중 가장 높지만 재현율이 가장 낮아요. 가려내는 쪽을 속이는 데는 \"확실한 진짜처럼 보이는 곳\"에만 만드는 게 안전해서예요."
   ]
  },
  "f_flow": {
   "title": "모델이 계산한 \"이 자리에 점이 있을 확률\" (밝을수록 높음, 흰 점: train)",
   "images": [
    [
     "img/gen/flow_density.png",
     ""
    ]
   ],
   "notes": [
    "Flow는 확률을 정확한 식으로 계산할 수 있어서, 화면의 모든 자리에 대해 밀도를 그릴 수 있어요. 정규분포는 타원 하나인데, Flow는 띠를 따라 들쭉날쭉한 좁은 줄무늬로 밝아요.",
    "그런데 밝은 곳이 train 점 바로 위에 좁게 몰려 있어요. 아래 ⑤의 출력처럼 이게 처음 보는 점에서 손해가 돼요."
   ],
   "chart": true
  },
  "s5": {
   "title": "⑤ Normalizing Flow (결합층 8개)",
   "lines": [
    [
     "정밀도 0.925, 재현율 0.919",
     1
    ],
    [
     "평균 log 확률 → train -1.822, 처음 보는 점 -3.419",
     2,
     "bad"
    ],
    "(참고) 정규분포 → train -2.737, 처음 보는 점 -2.622"
   ],
   "notes": [
    "만든 점의 모양은 정규분포보다 훨씬 진짜 같아요(정밀도 0.83 → 0.93).",
    "그런데 처음 보는 점의 log 확률은 정규분포보다 낮아요. train에서는 −1.8로 훨씬 높은데요. train 240개 바로 위에만 확률을 몰아 준 거예요. 가능도도 과적합해요. 확률 모델을 고를 때도 train이 아니라 처음 보는 데이터로 재는 이유예요."
   ]
  },
  "f_diff": {
   "title": "Diffusion: 위는 train 점에 잡음을 섞는 쪽, 아래는 순수한 잡음에서 걷어내는 쪽",
   "images": [
    [
     "img/gen/diffusion_steps.png",
     ""
    ]
   ],
   "notes": [
    "위: 걸음이 갈수록 띠 모양이 흐려져서, t=99에서는 원래 점이 거의 남지 않은 정규분포가 돼요. 이 방향은 학습할 게 없는 정해진 계산이에요.",
    "아래: 신경망이 \"섞인 잡음\"을 맞혀서 한 걸음씩 걷어내요. t=50까지는 둥근 덩어리이고, t=20쯤부터 띠 두 개의 모양이 잡히고, 마지막 걸음들에서 더 모여요. 아래 맨 오른쪽은 ⑥의 출력과 같은 점이에요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ Diffusion (T=100)",
   "lines": [
    "원래 점이 남은 몫 √ᾱ → t=0: 1.000, t=99: 0.075",
    "마지막 학습 손실(잡음 맞히기 MSE) → 0.318",
    [
     "정밀도 0.911, 재현율 0.975",
     1
    ]
   ],
   "notes": [
    "정밀도와 재현율이 둘 다 높아요. GAN처럼 한쪽으로 쏠리지 않고, VAE보다 덜 흐려요. 대신 점 하나를 만들려고 신경망을 100번 돌렸어요. 다른 모델은 한 번이에요."
   ]
  },
  "t6": {
   "title": "걸음 수 T를 100에서 20으로 줄이면 (끝에 거의 잡음이 되게 β를 키워서)",
   "cols": [
    "T",
    "마지막 β",
    "끝에 남은 몫 √ᾱ",
    "정밀도",
    "재현율"
   ],
   "rows": [
    [
     "<b>100</b> (지금)",
     0.1,
     "0.075",
     0.911,
     0.975
    ],
    [
     20,
     0.4,
     "0.095",
     0.894,
     0.975
    ]
   ],
   "notes": [
    "걸음을 줄이면 한 걸음에 걷어낼 잡음이 커져서 신경망이 맞혀야 할 일이 어려워져요. 만드는 시간은 5분의 1이 됐고, 이 쉬운 2차원 데이터에서는 정밀도만 조금(0.911 → 0.894) 떨어졌어요. 그림처럼 복잡한 데이터에서는 걸음 수가 품질에 훨씬 크게 영향을 줘요.",
    "실제로는 학습은 많은 걸음으로 하고, 만들 때만 걸음을 건너뛰는 방법(DDIM 같은)으로 속도와 품질을 맞춰요."
   ]
  },
  "f_all": {
   "title": "다섯 모델이 만든 점 1,000개 (회색: 처음 보는 진짜 점)",
   "images": [
    [
     "img/gen/all_samples.png",
     ""
    ]
   ],
   "notes": [
    "정규분포는 띠 사이 빈 곳까지, GAN은 띠의 일부에만, Diffusion은 띠 전체를 고르게 덮어요."
   ],
   "chart": true
  },
  "s7": {
   "title": "⑦ 같은 기준으로 비교 (처음 보는 진짜 점 160개 기준)",
   "cols": [
    "모델",
    "정밀도",
    "재현율",
    "평균 log 확률",
    "점 하나 만들 때 신경망 실행"
   ],
   "rows": [
    [
     "정규분포",
     "0.832",
     "0.962",
     "-2.622",
     "0번"
    ],
    [
     "VAE",
     "0.896",
     "0.981",
     "-",
     "1번"
    ],
    [
     "GAN",
     "0.962",
     "0.744",
     "-",
     "1번"
    ],
    [
     "Flow",
     "0.925",
     "0.919",
     "-3.419",
     "8층 한 번"
    ],
    [
     "Diffusion",
     "0.911",
     "0.975",
     "-",
     "100번"
    ],
    [
     "(참고) 진짜 train 점",
     "0.942",
     "0.988",
     "-",
     "-"
    ]
   ],
   "notes": [
    "\"가장 좋은 생성 모델\"은 무엇을 원하느냐에 달려요. 그럴듯함(정밀도)만 보면 GAN, 고르게 덮기(재현율)까지 보면 Diffusion과 VAE, 확률을 정확히 계산해야 하면 Flow예요.",
    "확률을 계산할 수 있는 건 정규분포와 Flow뿐이에요. VAE는 하한(ELBO)만, GAN과 Diffusion은 확률을 직접 주지 않아요. 그래서 그림 생성 모델은 FID처럼 만든 그림과 진짜 그림의 특징 분포를 비교하는 지표로 재요."
   ]
  }
 },
 "flow": [
  [
   1,
   "준비",
   "크기 맞추기",
   "정밀도, 재현율"
  ],
  [
   2,
   "정규분포",
   "타원 하나",
   "기준선"
  ],
  [
   3,
   "VAE",
   "줄였다 되살리기",
   "ELBO"
  ],
  [
   4,
   "GAN",
   "만들기 vs 가려내기",
   "모드 붕괴"
  ],
  [
   5,
   "Flow",
   "되돌릴 수 있는 변환",
   "정확한 확률"
  ],
  [
   6,
   "Diffusion",
   "잡음 걷어내기",
   "100걸음"
  ],
  [
   7,
   "비교",
   "같은 기준",
   "목적에 따라"
  ]
 ],
 "env": "Python 3.13.5, PyTorch 2.13.0+cpu, CPU (seed 0)"
};
