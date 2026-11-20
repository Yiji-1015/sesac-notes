"""모델 > 생성 모델과 그래프 > 그래프 신경망(GNN). examples/gnn/karate_gcn.py 하나가 ①~⑤로 자란다.
실행: python tools/pages/gnn_karate.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "gnn"
code = S.parse((EX / "karate_gcn.py").read_text(encoding="utf-8"))
ns = S.run(code, 5, S.workdir(EX), filename="karate_gcn.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, networkx {nx.__version__}, CPU (seed 0)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "gnn"
IMG.mkdir(parents=True, exist_ok=True)
g, y, n = ns["g"], ns["y"].numpy(), ns["n"]
tr = ns["train_idx"].numpy()
pos = nx.spring_layout(g, seed=3)
CL = [C["purple"], C["pink"]]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/gnn/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


def draw(ax, colors, title, edge_alpha=.35, labels=True, mark=None):
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=edge_alpha, width=.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=[300 if i in tr else 160 for i in range(n)], edgecolors=[C["ink"] if i in tr else "white" for i in range(n)], linewidths=[2.4 if i in tr else .8 for i in range(n)])
    if labels:
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=7, font_color="white")
    if mark is not None:
        xs = np.array([pos[i] for i in mark])
        if len(xs):
            ax.scatter(xs[:, 0], xs[:, 1], s=520, facecolors="none", edgecolors=C["orange"], linewidths=2.2)
    ax.set_title(title)
    ax.axis("off")


outs = {}
fig, ax = plt.subplots(figsize=(7, 5.2))
draw(ax, [CL[c] for c in y], "가라테 클럽 34명 (색: 실제로 간 쪽, 굵은 테두리: 알려 준 4명)")
outs["data"] = chart("데이터: 회원 34명과 친구 관계 78개", fig_save(fig, "graph"), "보라: 사범(0번) 쪽, 분홍: 회장(33번) 쪽",
                     ["1970년대 미국의 한 가라테 클럽이 사범과 회장의 다툼으로 둘로 갈라졌어요. 선은 \"클럽 밖에서도 만나는 사이\"예요. 이 그래프는 그래프 연구에서 가장 많이 쓰는 예제예요.",
                      "4명(굵은 테두리: 0, 1번과 32, 33번)의 소속만 알려 주고, 나머지 30명이 어느 쪽으로 갔는지 맞혀요. 알려 주는 정보는 \"누가 누구와 친한가\"뿐이에요."])
outs["s1"] = S.box("① 실행 결과", [f"A.shape → {tuple(ns['A'].shape)}, 친구 관계 {int(ns['A'].sum().item() / 2)}개 (A에는 양쪽으로 두 번)", f"X.shape → {tuple(ns['X'].shape)} (단위행렬)",
                                    (f"y → 두 쪽 {np.bincount(y).tolist()}명, 알려 준 4명의 소속 {y[tr].tolist()}", 1)],
                   ["두 파가 17명씩이라, 아무렇게나 찍으면 정확도가 0.5예요."])
n_test = int(ns["test_mask"].sum())
S.claim(abs(ns["mlp_acc"] - 0.5) < 0.1, "② 연결을 무시하면 반반으로 찍는다")
outs["s2"] = S.box("② 연결을 무시한 신경망", [(f"나머지 30명 정확도 → {ns['mlp_acc']:.2f}", 1, "bad")],
                   ["입력이 \"자기 이름표\"뿐이라, 처음 보는 30명의 이름표는 학습에 한 번도 나오지 않았어요. 신경망은 이 사람들에 대해 아무것도 모르니 반반으로 찍어요. 이 문제의 정보는 전부 <b>연결</b>에 있어요."])
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, k in zip(axs, (1, 2, 3)):
    s_ = ns["spread"][k].numpy()
    tot = s_.sum(1)
    p1 = np.where(tot > 0, s_[:, 1] / np.where(tot > 0, tot, 1), 0.5)
    cols = [plt.cm.PuRd(0.15 + 0.7 * v) if t_ > 0 else "#dddddd" for v, t_ in zip(p1, tot)]
    wrong = [i for i in range(n) if i not in tr and tot[i] > 0 and int(s_[i].argmax()) != y[i]]
    draw(ax, cols, f"{k}번 퍼뜨린 뒤: 정보가 닿은 사람 {ns['reached'][k]}명, 정확도 {ns['spread_acc'][k]:.2f}", labels=True, mark=wrong)
outs["f_spread"] = chart("알려 준 4명의 소속을 친구 관계를 따라 퍼뜨리면", fig_save(fig, "spread"), "회색: 아직 정보가 안 닿음, 진할수록 회장 쪽, 주황 원: 틀린 사람",
                         [f"1번 퍼뜨리면 4명의 친구까지만 정보가 닿아요. 2번이면 모두에게 닿고, {n_test}명 중 {round(ns['spread_acc'][2] * n_test)}명을 맞혀요. 학습한 것은 하나도 없어요. \"친구끼리는 같은 쪽으로 갔을 것\"이라는 가정 하나로 맞힌 거예요.",
                          "틀린 사람은 양쪽 모두와 친한 경계에 있는 사람이에요."])
S.claim(ns["reached"][1] < n and ns["reached"][2] == n, "③ 1번이면 일부, 2번이면 모두에게 닿는다")
outs["s3"] = S.table("③ 모으기만 k번 반복한 결과 (학습 없음)", ["k", "정보가 닿은 사람", "나머지 30명 정확도"],
                     [[k, f"{ns['reached'][k]}명", f"{ns['spread_acc'][k]:.2f}"] for k in (1, 2, 3)],
                     ["A_norm을 한 번 곱할 때마다 \"친구의 친구\"까지 한 칸씩 더 퍼져요. 이 그래프는 누구든 2칸이면 4명 중 누군가에게 닿아요."])
with torch.no_grad():
    out = ns["gcn"](ns["X"])
pred = out.argmax(1).numpy()
wrong = [i for i in range(n) if i not in tr and pred[i] != y[i]]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
draw(a, [CL[c] for c in pred], f"GCN의 답 (나머지 30명 정확도 {ns['gcn_acc']:.2f}, 주황 원: 틀림)", mark=wrong)
o_ = out.numpy()
for c in (0, 1):
    m_ = y == c
    b.scatter(o_[m_, 0], o_[m_, 1], s=60, color=CL[c], label=["실제 사범 쪽", "실제 회장 쪽"][c], edgecolors="white")
for i in range(n):
    b.annotate(str(i), (o_[i, 0], o_[i, 1]), fontsize=7, ha="center", va="center", color="white")
lo, hi = o_.min() - .3, o_.max() + .3
b.plot([lo, hi], [lo, hi], color=C["gray"], ls="--", lw=1)
b.set(xlabel="사범 쪽 점수", ylabel="회장 쪽 점수", title="마지막 층이 낸 두 점수 (점선 위: 회장 쪽이라고 답함)")
b.legend(fontsize=9)
outs["f_gcn"] = chart("2층 GCN이 4명만 보고 배운 결과", fig_save(fig, "gcn"), "", ["오른쪽: 학습에 쓴 건 4명의 정답뿐인데, 34명 모두가 두 무리로 갈라졌어요. 층마다 이웃끼리 값을 섞기 때문에, 친한 사람들이 비슷한 점수를 받아요."])
outs["s4"] = S.box("④ GCN (34 → 16 → 2, Adam 200번)", [f"가중치 수 → {sum(p.numel() for p in ns['gcn'].parameters())}개", (f"나머지 30명 정확도 → {ns['gcn_acc']:.2f}", 1)],
                   [f"③의 손으로 퍼뜨리기와 같은 {ns['spread_acc'][2]:.2f}가 나왔어요. 이 그래프에서는 \"친구끼리 같은 쪽\"이라는 가정이 거의 다 맞아서, 퍼뜨리기만으로도 충분해요. GCN은 여기에 \"어떤 정보를 어떻게 바꿔서 퍼뜨릴지\"를 배우는 가중치를 더한 거예요. 사람마다 나이, 수련 기간 같은 정보가 있다면 그것까지 섞어서 배울 수 있어요."])
S.claim(abs(ns["gcn_acc"] - ns["spread_acc"][2]) < 1e-9, "④ GCN이 손으로 퍼뜨리기와 같은 정확도")
agg_name = {"norm": "정규화한 합 (④)", "mean": "평균", "sum": "그냥 합", "max": "최댓값"}
outs["t5a"] = S.table("모으는 방법만 바꾸면", ["모으기", "나머지 30명 정확도"], [[f"<b>{agg_name[k]}</b>" if k == "norm" else agg_name[k], f"{v:.2f}"] for k, v in ns["agg_acc"].items()],
                      [f"이 작은 그래프에서는 거의 차이가 없어요. 최댓값만 {round((ns['agg_acc']['norm'] - ns['agg_acc']['max']) * n_test)}명 더 틀렸어요.",
                       "차이는 친구 수가 크게 다를 때 드러나요. 그냥 합은 친구가 많은 사람의 값이 커지고, 평균은 친구가 1명이든 100명이든 같은 크기예요. 대신 평균은 \"친구가 몇 명인가\" 자체를 잃어요. 최댓값은 \"이웃 중에 이런 사람이 하나라도 있나\"를 잘 잡아요."])
S.claim(len({ns["agg_acc"][k] for k in ("norm", "mean", "sum")}) == 1 and ns["agg_acc"]["max"] < ns["agg_acc"]["norm"], "⑤ 정규화한 합, 평균, 합은 같고 최댓값만 낮다")
S.claim(min(ns["depth_acc"][d] for d in (2, 4)) > 0.9 and max(abs(ns["depth_acc"][d] - 0.5) for d in (8, 16)) < 0.1, "⑤ 2, 4층은 맞히고 8층부터 반반")
outs["t5b"] = S.table("층 수를 바꾸면 (칸 16개)", ["층 수", "나머지 30명 정확도"], [[f"<b>{d}</b>" if d == 2 else d, f"{v:.2f}"] for d, v in ns["depth_acc"].items()],
                      [f"8층부터 반반으로 찍어요. 층마다 이웃과 섞으니 8층이면 8칸 떨어진 사람까지 섞이는데, 이 그래프는 누구든 {nx.diameter(g)}칸 안에 닿아서 모두가 비슷한 값이 돼 버려요(아래 그림). 층이 깊어 학습 자체도 어려워져요."])
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ks = list(ns["smooth"].keys())
ax.plot(ks, list(ns["smooth"].values()), marker="o", color=C["purple"], lw=2.2)
ax.set_xscale("symlog", linthresh=1)
ax.set_xticks(ks, [str(k) for k in ks])
ax.minorticks_off()
ax.set(xlabel="모으기 횟수 k", ylabel="모든 두 사람 사이 코사인 유사도 평균", ylim=(-0.03, 1.03), title="많이 섞을수록 모두가 같아진다 (과평활)")
outs["f_smooth"] = chart("이름표 벡터를 A_norm으로 k번 섞은 뒤, 사람끼리 얼마나 비슷한가", fig_save(fig, "oversmoothing"), "",
                         [f"처음(k=0)에는 이름표가 모두 달라서 0이에요. 4번이면 {ns['smooth'][4]:.2f}, 16번이면 {ns['smooth'][16]:.2f}, 32번이면 {ns['smooth'][32]:.3f}로 1에 붙어요. 누가 누구인지 구별할 정보가 사라지는 거예요.",
                          "그래서 GCN은 보통 2~3층만 쌓아요. 더 깊이 가려면 <a href=\"cnn.html\">CNN</a>처럼 잔차 연결을 넣거나, 층마다 자기 원래 값을 일부 남겨요."])
flow = [[1, "그래프 → Tensor", "인접 행렬 A", "34명, 78관계"], [2, "연결 무시", "이름표만", f"{ns['mlp_acc']:.2f}"], [3, "퍼뜨리기", "A_norm 곱하기", f"학습 없이 {ns['spread_acc'][2]:.2f}"], [4, "GCN", "바꾸고 모으기", "4명으로 학습"], [5, "모으기와 깊이", "평균, 합, 최댓값", "과평활"]]
S.save(ROOT / "topics" / "model" / "gnn-data.js", {"code": code, "names": S.names(code), "range": [1, 5], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>표 데이터는 행끼리 따로따로예요. 그림은 픽셀이 격자로, 글은 토큰이 한 줄로 이어져 있어요. 그런데 사람 사이 친구 관계, 분자 속 원자의 결합, 도로망, 논문 인용, 지식 그래프처럼 <b>연결 모양이 제각각인</b> 데이터도 많아요. 이런 데이터를 <b>그래프</b>라고 하고, 점을 <b>노드</b>, 선을 <b>엣지</b>라고 해요.</p>
{K.out("data")}
<p>그래프에서 맞히는 것은 세 가지예요.</p>
{K.table(["수준", "맞히는 것", "예"], [["노드", "점 하나하나의 성질", "이 회원은 어느 쪽으로 갔나 (이 쪽), 이 계정은 스팸인가"], ["엣지", "두 점 사이 관계", "이 두 사람은 친구가 될까 (추천), 이 두 약은 함께 쓰면 위험한가"], ["그래프 전체", "그래프 하나의 성질", "이 분자는 독성이 있나"]])}
<p>그래프에는 그림처럼 \"왼쪽 위\" 같은 자리가 없어요. 회원 번호를 다시 매겨도 같은 그래프예요. 그래서 모델도 <b>번호를 어떻게 매기든 같은 답</b>을 내야 해요. 그래프 신경망(<b>GNN</b>)은 \"이웃의 값을 모아 내 값을 고친다\"는 방식으로 이걸 지켜요. 이웃을 모으는 데는 순서가 없거든요(합, 평균, 최댓값).</p>"""
whole = """<p>그래프를 행렬로 → 연결을 무시하면 못 맞힘 → 이웃을 따라 퍼뜨리기만 해도 맞힘 → 퍼뜨리는 방법을 배우는 GCN → 모으는 방법과 깊이. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "그래프를 Tensor로", "누가 누구와 연결됐는지를 34×34 표(인접 행렬)로 만들고, 소속을 알려 줄 4명과 채점할 30명을 나눠요.",
           "<p><b>인접 행렬</b> A는 i번과 j번이 연결됐으면 A[i, j] = 1인 표예요. 친구 관계는 방향이 없어서 표가 대각선을 기준으로 대칭이에요. 노드마다 특성(나이, 글 내용 등)이 있으면 노드 수 × 특성 수 표 X로 둬요. 이 예제에는 특성이 없어서 회원마다 자기 자리에만 1인 \"이름표\"(단위행렬)를 써요. 엣지에도 특성(관계의 종류, 거리)이 있을 수 있어요.</p>"
           "<p>이 쪽처럼 그래프 하나 안에서 일부 노드의 답만 알려 주고 나머지를 맞히는 걸 <b>변환적</b>(transductive) 학습이라고 해요. 학습할 때 채점할 노드도 그래프 안에 있어서, 그 노드들의 연결은 볼 수 있어요. 정답만 안 보는 거예요.</p>"
           + K.code_table([("nx.karate_club_graph()", "networkx에 들어 있는 가라테 클럽 그래프"), ("A[u, v] = A[v, u] = 1.0", "방향 없는 관계라 양쪽에"), ("torch.eye(n)", "n×n 단위행렬. 사람마다 다른 한 칸짜리 이름표"),
                           ("test_mask[train_idx] = False", "알려 준 4명은 채점에서 빼기")]), "s1"),
    K.step(2, "연결을 무시하면", "노드 특성만 보는 보통 신경망으로 해 봐요. 기준선이에요.",
           K.code_table([("model(X)[train_idx]", "34명 모두 계산하고, 손실은 아는 4명으로만"), ("weight_decay=5e-4", "가중치 규제 (<a href=\"dl-training-stability.html\">학습을 안정시키기</a>)")]), "s2"),
    K.step(3, "이웃 정보 모으기를 손으로", "알려 준 4명의 소속을 친구 관계를 따라 퍼뜨려 봐요. 학습은 하지 않아요.",
           "<p>\"친구끼리는 비슷하다\"는 가정을 써요. 알려 준 4명의 소속을 한 칸짜리 표로 두고, 인접 행렬을 곱하면 \"내 친구들의 소속을 더한 값\"이 돼요. 한 번 더 곱하면 친구의 친구까지 닿아요.</p>"
           "<p>그냥 A를 곱하면 친구가 많은 사람의 값이 커지고, 자기 값은 빠져요. 그래서 자기 자신과도 연결해(A + I) 자기 값을 남기고, i번과 j번 사이 값을 √(i의 친구 수 × j의 친구 수)로 나눠 크기를 맞춰요. 이게 <code>A_norm</code>이에요.</p>"
           + K.out("f_spread")
           + K.code_table([("A + torch.eye(n)", "자기 자신도 이웃으로 (자기 고리)"), ("A_hat / torch.sqrt(deg[:, None] * deg[None, :])", "D^(−½) Â D^(−½). 친구 수로 양쪽을 나눠 크기 맞추기"),
                           ("torch.linalg.matrix_power(A_norm, k) @ known", "k번 퍼뜨리기"), ("(s.sum(1) > 0).sum()", "값이 0이 아닌, 정보가 닿은 사람 수")]), "s3"),
    K.step(4, "GCN: 이웃을 모으는 층을 학습하기", "층마다 \"모든 사람에게 같은 가중치로 값을 바꾸고, 이웃끼리 모으기\"를 해요. 바꾸는 가중치를 아는 4명의 정답으로 학습해요.",
           "<p><b>GCN</b>(Graph Convolutional Network)의 층 하나는 <code>H' = ReLU(A_norm × H × W)</code>예요. H × W는 노드마다 같은 가중치로 값을 바꾸는 보통 Linear 층이고, A_norm을 곱하는 건 ③의 \"이웃 모으기\"예요. <a href=\"cnn.html\">CNN</a>이 \"모든 자리에 같은 3×3 무늬\"를 쓰듯, GCN은 \"모든 노드에 같은 W\"를 써요(가중치 공유). 그래서 그래프 크기와 상관없이 가중치 수가 같고, 노드 번호를 바꿔도 결과가 그대로 따라 바뀌어요.</p>"
           "<p>2층이면 각 노드의 답은 2칸 안에 있는 노드들의 정보로 정해져요. 마지막 층의 두 값이 두 쪽의 점수예요. 손실은 아는 4명에 대해서만 계산하지만, A_norm을 거치면서 기울기가 이웃들로 퍼져서 다른 사람들의 표현도 같이 움직여요.</p>"
           + K.out("f_gcn")
           + K.code_table([("class GCN(nn.Module)", "층 목록과 모으기 방법을 가진 모델"), ("self.gather(lin(x))", "바꾸고(Linear) 모으기(A_norm 곱하기)"), ("nn.ModuleList(...)", "층 여러 개를 목록으로 (PyTorch가 가중치를 찾을 수 있게)")]), "s4"),
    K.step(5, "모으는 방법과 층 수 바꾸기", "이웃을 모으는 방법(정규화한 합, 평균, 합, 최댓값)과 층 수를 바꿔 봐요.",
           "<p>모으는 방법은 순서와 상관없는 계산이면 돼요. 합, 평균, 최댓값이 기본이고, 이웃마다 \"얼마나 중요한가\"를 Attention처럼 계산해 가중 평균하는 GAT(Graph Attention Network)도 있어요. <a href=\"llm-principles.html\">Transformer</a>는 사실 \"모든 토큰이 서로 연결된 그래프\"에 Attention으로 모으는 GNN으로 볼 수도 있어요.</p>"
           + K.try_("모으는 방법만 바꾸면?", "t5a")
           + "<p>층을 쌓으면 더 멀리 있는 노드의 정보까지 볼 수 있지만, 그만큼 모두와 섞여요.</p>"
           + K.try_("층을 4, 8, 16개로 쌓으면?", "t5b", after=K.out("f_smooth"))
           + K.code_table([("(A_hat @ h) / A_hat.sum(1, keepdim=True)", "평균: 합을 이웃 수로 나누기"), ("h[A_hat[i] > 0].max(0).values", "최댓값: i의 이웃들 중 칸마다 가장 큰 값"),
                           ("F.normalize(h, dim=1)", "줄마다 길이 1로. 그 뒤 내적이 코사인 유사도")]), None),
]
practice = [("새 그래프나 새 노드가 계속 들어온다", "GCN의 W는 노드 수와 상관없어서, 학습한 W를 새 그래프에 그대로 쓸 수 있어요(귀납적 학습). 단, 이 쪽처럼 이름표(단위행렬)를 특성으로 쓰면 새 노드의 이름표가 없어서 안 돼요. 실제로는 나이, 글 내용 같은 진짜 특성을 써요."),
            ("그래프 전체를 분류한다", "노드 표현을 다 모아(평균이나 합) 그래프 하나의 벡터로 만든 뒤 분류해요(readout). 그래프 여러 개를 한 묶음으로 학습할 때는 작은 그래프들을 서로 연결 없는 큰 그래프 하나로 합쳐 계산해요."),
            ("두 노드가 연결될지 맞힌다 (추천)", "두 노드의 표현을 내적하거나 이어 붙여 점수를 내요. 학습할 때 일부 엣지를 숨기고 그걸 맞히게 해요."),
            ("그래프가 아주 크다", "노드 수 × 노드 수인 A를 그대로 쓰면 메모리가 넘쳐요. 연결된 것만 저장하는 희소 행렬과, 이웃을 일부만 뽑아 쓰는 방법(GraphSAGE)을 써요. PyTorch Geometric, DGL 같은 라이브러리가 이걸 해 줘요."),
            ("지식 그래프와 GNN", "지식 그래프는 \"무엇과 무엇이 어떤 관계인가\"를 저장하는 데이터 구조이고, GNN은 그래프 위에서 배우는 모델이에요. 둘 다 그래프지만 역할이 달라요. 지식 그래프 위에서 GNN으로 빠진 관계를 맞히기도 해요.")]
libs = [("①", "<code>nx.karate_club_graph</code>, <code>g.edges()</code>, <code>torch.eye</code>", "그래프 (networkx)"), ("③", "<code>torch.linalg.matrix_power</code>", "행렬 거듭제곱"), ("④⑤", "<code>nn.ModuleList</code>, <code>F.normalize</code>", "GCN 층, 유사도")]
K.build(ROOT / "topics" / "model" / "gnn.html", title="그래프 신경망 (GNN)", h1="그래프 신경망: 이웃을 모아 나를 고치기", track="모델", crumb="모델 &gt; 생성 모델과 그래프",
        one_line="<b>GNN</b>은 노드마다 <b>이웃의 값을 모아 자기 값을 고치기</b>를 층마다 반복해요. 같은 가중치를 모든 노드에 쓰니 그래프 크기와 번호 매기기에 상관없이 동작해요. 너무 많이 쌓으면 모두가 같아져요(과평활).",
        why=why, whole=whole, steps=steps, code_file="examples/gnn/karate_gcn.py", practice=practice, libs=libs, libs_note="<code>acc</code>, <code>fit</code>, <code>avg_cosine</code>은 이 파일 안에서 정의한 함수, <code>GCN</code>은 클래스예요.",
        next_rows=[("<a href=\"cnn.html\">CNN과 이미지 분류</a>", "격자 위의 가중치 공유"), ("<a href=\"llm-principles.html\">Transformer 구조</a>", "Attention으로 모으기"), ("<a href=\"generative-models.html\">생성 모델</a>", "같은 갈래의 다른 쪽")],
        sources=["함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 13장(그래프 신경망). 그래프 표현, 노드, 엣지, 그래프 수준 과제, 번호 매기기에 대한 불변성과 등변성, GCN, 모으는 방법, 귀납적 학습과 변환적 학습, readout, 과평활을 가져왔어요. 부트캠프 수업에서 따로 다루지 않은 주제라 예제는 새로 만들었어요",
                 "데이터: networkx에 들어 있는 Zachary 가라테 클럽 그래프 (W. W. Zachary, 1977)",
                 "결과 값과 그림: <code>tools/pages/gnn_karate.py</code>가 <code>examples/gnn/karate_gcn.py</code>를 단계별로 실제 실행한 결과 (seed 0)",
                 "공식 문서: <a href=\"https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html\" target=\"_blank\">karate_club_graph</a>, <a href=\"https://arxiv.org/abs/1609.02907\" target=\"_blank\">GCN 논문</a>"])
print("ok")
