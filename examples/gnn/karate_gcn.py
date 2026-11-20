# 가라테 클럽 그래프에서 4명의 소속만 보고 나머지 30명의 소속 맞히기 (GNN) 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: networkx에 들어 있는 Zachary 가라테 클럽 (회원 34명, 친구 관계 78개, 나중에 두 파로 갈라짐). 따로 받을 파일 없음
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import networkx as nx                                            #@1
import torch                                                     #@1
from torch import nn                                             #@2
import torch.nn.functional as F                                  #@2

# ① 그래프를 Tensor로
g = nx.karate_club_graph()
n = g.number_of_nodes()
A = torch.zeros(n, n)
for u, v in g.edges():
    A[u, v] = A[v, u] = 1.0                       # 친구면 1 (방향 없음)
y = torch.tensor([0 if g.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(n)])
X = torch.eye(n)                                  # 회원마다 "자기 이름표" 한 칸. 그 밖의 정보는 없음
train_idx = torch.tensor([0, 1, 32, 33])          # 소속을 알려 주는 4명 (두 파의 대표와 그 옆 사람)
test_mask = torch.ones(n, dtype=torch.bool)
test_mask[train_idx] = False                      # 나머지 30명으로 채점

def acc(pred):
    return (pred[test_mask] == y[test_mask]).float().mean().item()

def fit(model, steps=200, lr=0.01):
    torch.manual_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(steps):
        loss = F.cross_entropy(model(X)[train_idx], y[train_idx])   # 아는 4명으로만 손실
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return acc(model(X).argmax(1))

# ② 기준선: 연결을 무시한 신경망
torch.manual_seed(0)
mlp = nn.Sequential(nn.Linear(n, 16), nn.ReLU(), nn.Linear(16, 2))
mlp_acc = fit(mlp)

# ③ 이웃 정보 모으기를 손으로
A_hat = A + torch.eye(n)                          # 자기 자신도 이웃으로
deg = A_hat.sum(1)
A_norm = A_hat / torch.sqrt(deg[:, None] * deg[None, :])   # 이웃이 많은 사람의 몫을 줄여 크기 맞추기
known = torch.zeros(n, 2)
known[train_idx, y[train_idx]] = 1.0              # 아는 4명만 한 칸에 1
spread = {k: torch.linalg.matrix_power(A_norm, k) @ known for k in (1, 2, 3)}   # k번 이웃을 거쳐 퍼뜨리기
spread_acc = {k: acc(s.argmax(1)) for k, s in spread.items()}
reached = {k: int((s.sum(1) > 0).sum()) for k, s in spread.items()}             # 정보가 닿은 사람 수

# ④ GCN: 이웃을 모으는 층을 학습하기
class GCN(nn.Module):
    def __init__(self, sizes, agg="norm"):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.agg = agg

    def gather(self, h):  # 이웃(과 자기)의 값을 모으기
        if self.agg == "norm":
            return A_norm @ h
        if self.agg == "mean":
            return (A_hat @ h) / A_hat.sum(1, keepdim=True)
        if self.agg == "sum":
            return A_hat @ h
        return torch.stack([h[A_hat[i] > 0].max(0).values for i in range(n)])  # max

    def forward(self, x, keep=False):
        self.hidden = []
        for i, lin in enumerate(self.layers):
            x = self.gather(lin(x))                # 모든 사람에게 같은 가중치로 바꾼 뒤 이웃끼리 모으기
            if i < len(self.layers) - 1:
                x = F.relu(x)
                self.hidden.append(x)
        return x

torch.manual_seed(0)
gcn = GCN([n, 16, 2])
gcn_acc = fit(gcn)

# ⑤ 모으는 방법과 층 수 바꾸기
agg_acc = {}
for agg in ("norm", "mean", "sum", "max"):
    torch.manual_seed(0)
    agg_acc[agg] = fit(GCN([n, 16, 2], agg))
depth_acc = {}
for depth in (2, 4, 8, 16):
    torch.manual_seed(0)
    depth_acc[depth] = fit(GCN([n] + [16] * (depth - 1) + [2]))

def avg_cosine(h):  # 모든 두 사람 사이 코사인 유사도의 평균. 1에 가까울수록 다 비슷해진 것
    u = F.normalize(h, dim=1)
    s = u @ u.T
    return ((s.sum() - n) / (n * (n - 1))).item()

smooth = {k: avg_cosine(torch.linalg.matrix_power(A_norm, k) @ X) for k in (0, 1, 2, 4, 8, 16, 32)}  # 모으기만 k번
