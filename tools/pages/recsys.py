"""모델 > 추천 시스템. examples/recsys/recommend.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/recsys.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "recsys"
code = S.parse((EX / "recommend.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="recommend.py")
ENV = f"Python {sys.version.split()[0]}, NumPy {np.__version__}, PyTorch {torch.__version__} (CPU 1스레드, seed 0)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "recsys"
IMG.mkdir(parents=True, exist_ok=True)
OK, FAIL = C["purple"], C["pink"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/recsys/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


res = ns["results"]
METRICS = ["정밀도@10", "재현율@10", "NDCG@10", "커버리지"]
row = lambda name: [name, *[f3(res[name][m]) for m in METRICS]]  # noqa: E731
outs = {}
# ---------------- ①
R, X = ns["R"], ns["X"]
cnt = R.sum(axis=0)
S.claim(ns["density"] < 0.1 and ns["top10_share"] > 0.2, "① 채워진 칸이 10% 아래, 상위 10% 아이템이 기록의 20% 넘게")
outs["s1"] = S.box("① 만든 이용 기록", [(f"사용자 {ns['U']}명 × 아이템 {ns['I']}개 = {ns['U'] * ns['I']:,}칸 중 본 기록 {int(R.sum()):,}칸 ({ns['density']:.1%})", 1),
                                    (f"인기 상위 10% 아이템({ns['I'] // 10}개)이 전체 기록의 {ns['top10_share']:.0%}", 2, "newf"),
                                    (f"사용자 한 명당 학습용 기록 평균 {X.sum(axis=1).mean():.1f}개, 숨긴 정답 평균 {ns['test'].sum(axis=1).mean():.1f}개", 3)],
                   [f"표의 {1 - ns['density']:.0%}가 비어 있어요. 추천은 이 빈칸 중 \"볼 만한 것\"을 고르는 문제예요. 빈칸은 \"싫어한다\"가 아니라 \"아직 모른다\"예요.",
                    "몇몇 아이템에 기록이 몰려 있어요(긴 꼬리). 그래서 인기순만으로도 어느 정도 맞히고, 대부분의 아이템은 기록이 적어 배우기 어려워요."])
fig, ax = plt.subplots(figsize=(7.6, 3.4))
srt = np.sort(cnt)[::-1]
ax.bar(range(len(srt)), srt, color=OK, width=1.0)
ax.axvline(ns["I"] // 10, color=FAIL, lw=1.5, ls="--", label="상위 10%")
ax.set(xlabel="아이템 (본 사람 수 순서)", ylabel="본 사람 수", title="몇몇 아이템에 몰리는 긴 꼬리")
ax.legend(fontsize=9)
P_POP = fig_save(fig, "long_tail")
outs["f_pop"] = chart("아이템별 인기", P_POP, "", ["왼쪽 몇 개가 아주 높고 오른쪽으로 길게 낮아져요. 실제 서비스(음악, 영화, 채용 공고)도 대개 이 모양이에요."])

# ---------------- ②
S.claim(res["인기순"]["커버리지"] < 0.06 and res["진짜 선호 (천장)"]["NDCG@10"] < 0.35, "② 인기순은 아이템 5% 남짓만 추천, 진짜 선호로도 NDCG 0.35 아래")
outs["s2"] = S.table("② 기준선 둘", ["방법", *METRICS], [row("인기순"), row("진짜 선호 (천장)")],
                     [f"인기순은 거의 모두에게 같은 10개를 추천해서, {ns['I']}개 중 {round(res['인기순']['커버리지'] * ns['I'])}개만 추천 목록에 나와요(커버리지).",
                      f"진짜 선호로 순서를 매겨도 정밀도가 {res['진짜 선호 (천장)']['정밀도@10']:.2f}예요. 사람은 좋아할 만한 것 중 일부만 우연히 보니까, 숨긴 것을 다 맞히는 건 불가능해요. 이 값이 이 데이터에서 어떤 모델도 넘기 어려운 천장이에요."])

# ---------------- ③④⑤⑥
names = ["아이템 이웃", "사용자 이웃", "행렬 분해 (BPR)", "EASE", "LightGCN"]
best_simple = max(res["아이템 이웃"]["NDCG@10"], res["EASE"]["NDCG@10"])
S.claim(res["행렬 분해 (BPR)"]["NDCG@10"] < best_simple and res["아이템 이웃"]["NDCG@10"] >= max(res[n]["NDCG@10"] for n in names) and abs(res["LightGCN"]["NDCG@10"] - best_simple) < 0.02, "③~⑥ 행렬 분해가 이웃과 EASE보다 낮고, LightGCN은 0.02 안")
outs["s3"] = S.table("③~⑥ 다섯 모델 (사용자 600명 평균)", ["방법", *METRICS], [row(n) for n in names] + [row("진짜 선호 (천장)")],
                     ["가장 단순한 아이템 이웃이 NDCG 1등이에요. 닫힌 식 하나로 푸는 EASE와 그래프 모델 LightGCN이 거의 같고, 학습으로 사용자 벡터를 찾는 행렬 분해는 그보다 낮아요.",
                      "추천 연구에서 \"새 신경망 모델이 잘 다듬은 단순한 기준선을 생각보다 못 이긴다\"는 보고가 꾸준히 나와요. 새 모델을 들이기 전에 인기순, 아이템 이웃, EASE를 먼저 같은 조건으로 재 봐요."])

def retune(label, values, fn, current):
    rows = []
    for v in values:
        sc = fn(v)
        r_ = ns["evaluate"](sc)
        rows.append([f"<b>{v}</b> (지금)" if v == current else v, f3(r_["NDCG@10"]), f3(r_["커버리지"])])
    return rows

ease_rows = retune("λ", (50.0, 200.0, 500.0), lambda v: ns["ease"](v), 200.0)
bpr_rows = retune("reg", (0.01, 0.05, 0.1), lambda v: ns["bpr_mf"](reg=v), 0.05)
lgcn_rows = retune("layers", (0, 1, 2, 3), lambda v: ns["lightgcn"](layers=v), 2)
outs["t4"] = S.table("행렬 분해의 규제 세기 reg를 바꾸면", ["reg", "NDCG@10", "커버리지"], bpr_rows,
                     ["규제가 약하면(0.01) 각 사용자의 학습 기록 20개 남짓에 벡터 64칸을 맞추느라 외워 버려요. 너무 세면(0.1) 모두를 비슷한 벡터로 눌러 인기순에 가까워지고 커버리지가 줄어요. 학습형 모델은 이런 손잡이를 잘 맞춰야 기준선 근처에 와요."])
outs["t5"] = S.table("EASE의 λ를 바꾸면", ["λ", "NDCG@10", "커버리지"], ease_rows,
                     ["λ는 아이템끼리의 가중치 B를 작게 누르는 규제예요. 작으면 드문 아이템 쌍까지 믿어서 커버리지는 넓지만 덜 맞고, 크면 인기 있는 쪽으로 몰려요."])
outs["t6"] = S.table("LightGCN의 층 수를 바꾸면 (0층 = 그래프 없이 행렬 분해)", ["층", "NDCG@10", "커버리지"], lgcn_rows,
                     [f"0층은 이웃을 전혀 모으지 않는 행렬 분해라, 위 표의 reg 0.01과 같은 값이에요. 한두 층 이웃 벡터를 모아 평균 내면 \"나와 같은 걸 본 사람이 본 것\"(2홉)이 들어와 점수가 크게 올라요. 3층까지는 계속 올라서 아이템 이웃과 같은 수준이 됐어요({lgcn_rows[-1][1]}).", "층을 아주 많이 쌓으면 GNN 쪽에서 본 과평활처럼 모든 벡터가 비슷해질 수 있어서, 보통 2~4층을 써요."])
fig, axs = plt.subplots(1, 2, figsize=(12.4, 3.8))
show = ["인기순", *names, "진짜 선호 (천장)"]
cols = [C["gray"], OK, OK, C["blue"], OK, C["blue"], C["ink"]]
axs[0].barh(show[::-1], [res[n]["NDCG@10"] for n in show[::-1]], color=cols[::-1])
axs[0].set(xlabel="NDCG@10", title="얼마나 잘 맞히나")
axs[1].barh(show[::-1], [res[n]["커버리지"] for n in show[::-1]], color=cols[::-1])
axs[1].set(xlabel="커버리지 (추천된 아이템 비율)", title="얼마나 다양하게 추천하나")
P_CMP = fig_save(fig, "model_compare")
outs["f_cmp"] = chart("모델 비교", P_CMP, "", ["왼쪽은 이웃, EASE, LightGCN이 비슷하게 높아요. 오른쪽은 행렬 분해가 가장 다양하게 추천하지만 덜 맞혀요. 인기순은 가장 좁아요."])

# ---------------- ⑦
bg, pb = ns["by_group"], ns["popularity_bias"]
gnames = list(ns["groups"])
S.claim(bg["아이템 이웃"][gnames[0]] < bg["아이템 이웃"][gnames[-1]] and pb["인기순"] == 1.0 and pb["행렬 분해 (BPR)"] < pb["진짜 선호"] < pb["EASE"], "⑦ 기록이 적을수록 덜 맞고, EASE는 진짜 선호보다 인기 쪽, 행렬 분해는 덜")
outs["s7"] = S.table("⑦ 학습 기록 수로 사용자를 셋으로 나눠 본 NDCG@10", ["방법", *[f"{g} ({int(ns['groups'][g].sum())}명)" for g in gnames]],
                     [[m, *[f3(bg[m][g]) for g in gnames]] for m in bg],
                     ["기록이 많은 사용자일수록 잘 맞혀요. 기록이 하나도 없는 새 사용자(콜드 스타트)는 협업 필터링으로는 아예 추천할 수 없어서, 가입할 때 고른 관심사나 인기순, 아이템 내용(장르, 설명)으로 시작해요."])
outs["s7b"] = S.table("추천 목록 중 인기 상위 10% 아이템이 차지한 비율", ["방법", "비율"], [[k, f"{v:.0%}"] for k, v in pb.items()],
                      [f"진짜 선호대로 골라도 {pb['진짜 선호']:.0%}는 인기 아이템이에요(인기 있는 데는 이유가 있으니까). 아이템 이웃과 EASE는 그보다 조금 더 인기 쪽으로 기울어요(인기 편향). 인기 있는 것이 더 추천되고, 그래서 더 인기 있어지는 고리가 생길 수 있어요.",
                       "행렬 분해는 반대로 인기 아이템을 덜 추천했어요. 커버리지는 넓지만, 그만큼 정답을 덜 맞혔어요(③). 다양성과 정확도 사이에서 어디에 설지는 서비스가 정해요."])

flow = [[1, "이용 기록", "600 × 400, 92% 빈칸", "긴 꼬리"], [2, "지표와 기준선", "정밀도, 재현율, NDCG", "인기순, 천장"], [3, "이웃", "함께 본 것", "아이템 이웃 1등"],
        [4, "행렬 분해", "숨은 취향 벡터", "BPR"], [5, "EASE", "닫힌 식", "λ 하나"], [6, "LightGCN", "그래프로 2홉", "층 수"], [7, "편향", "기록 적은 사람, 인기", "콜드 스타트"]]
S.save(ROOT / "topics" / "model" / "recsys-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p>음악, 영상, 쇼핑, 채용 공고처럼 고를 것이 수만 개일 때, 사용자마다 볼 만한 몇 개를 골라 주는 게 <b>추천 시스템</b>이에요. 분류처럼 정답 표가 있지 않고, 사람들이 무엇을 봤는지(이용 기록)에서 취향을 짐작해요.</p>"""
why += K.table(["", "분류 (머신러닝 기초)", "추천"], [["데이터", "행마다 정답이 있는 표", "사용자 × 아이템 이용 기록, 대부분 빈칸"], ["예측", "이 행의 답", "이 사람이 아직 안 본 것 중 좋아할 순서"],
                                              ["빈칸의 뜻", "-", "싫어한다가 아니라 \"모른다\" (암묵적 피드백)"], ["평가", "정확도, F1", "상위 k개의 정밀도, 재현율, NDCG, 다양성"]])
why += "<p>이 쪽은 사용자마다 숨은 장르 취향을 정해 둔 가상 기록으로, 고전적인 방법부터 그래프 모델까지 같은 지표로 비교해요. 진짜 취향을 아니까 \"가장 잘해도 이 정도\"인 천장도 잴 수 있어요.</p>"
whole = """<p>기록과 나누기 → 지표와 기준선 → 이웃 → 숨은 취향 벡터 → 닫힌 식 → 그래프 → 편향. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/recsys/recommend.py</code> 하나예요. CPU로 10초 남짓 걸려요.</p>"""
steps = [
    K.step(1, "이용 기록 만들고 나누기", "사용자마다 장르 취향, 아이템마다 장르와 인기도를 정해 이용 기록을 만들고, 사용자마다 30%를 숨겨 정답으로 둬요.",
           "<p>평점처럼 \"얼마나 좋았는지\"를 직접 받은 것은 <b>명시적 피드백</b>, 봤다, 눌렀다, 샀다처럼 행동만 남은 것은 <b>암묵적 피드백</b>이에요. 실제 서비스는 대부분 암묵적이라 여기서도 \"봤다\"만 있어요.</p>"
           "<p>나눌 때는 사용자마다 일부를 숨겨요. 행을 통째로 나누면 test 사용자의 기록이 하나도 없어 추천할 수 없어요. 시간이 있는 데이터면 마지막 몇 개를 숨겨서 \"과거로 미래 맞히기\"를 해요.</p>"
           + K.out("s1") + K.out("f_pop")
           + K.code_table([("rng.pareto(1.2, I) + 1", "소수만 크게 인기 있는 분포"), ("rng.dirichlet(np.full(G, 0.3))", "사용자마다 몇 장르에 몰린 취향"), ("train[u, hide] = False", "사용자마다 30%를 숨겨 정답으로")]), "s1"),
    K.step(2, "평가 지표와 인기순 기준선", "사용자마다 10개를 추천하고, 숨긴 정답이 몇 개 들어 있는지와 몇 번째에 있는지로 채점해요.",
           "<p><b>정밀도@k</b>: 추천 k개 중 정답 비율. <b>재현율@k</b>: 정답 중 k개 안에 든 비율. <b>NDCG@k</b>: 위에 있을수록 크게 쳐서(1 ÷ log₂(순위 + 1)) 더한 뒤, 가장 잘 늘어놓았을 때 값으로 나눈 것이에요. 순서까지 보는 지표라 추천에서 가장 많이 써요. 이 밖에 MAP, MRR(첫 정답의 순위 역수), AUC도 써요. 그리고 정확도만 보면 모두 인기 아이템만 받게 되니, <b>커버리지</b>(추천에 한 번이라도 나온 아이템 비율)나 다양성을 같이 봐요.</p>"
           + K.out("s2")
           + K.code_table([("s[train] = -np.inf", "이미 본 것은 추천에서 빼기"), ("np.argsort(-s, axis=1)[:, :K]", "사용자마다 점수 상위 10개"), ("DISCOUNT = 1 / np.log2(...)", "NDCG의 순위 무게")]), "s2"),
    K.step(3, "이웃으로 추천하기: 협업 필터링", "같은 사람들이 함께 본 아이템일수록 닮았다고 보고, 내가 본 것과 닮은 아이템을 추천해요.",
           "<p><b>협업 필터링</b>은 아이템 내용을 보지 않고 \"나와 비슷한 사람들의 기록\"만으로 추천해요. <b>아이템 기반</b>: 아이템을 \"본 사람들\" 벡터로 보고 코사인 유사도를 구한 뒤, 내가 본 아이템들과의 유사도를 더해 점수를 매겨요. <b>사용자 기반</b>: 나와 닮은 사용자들이 본 것을 모아요. 아이템 수가 사용자 수보다 적고 덜 바뀌는 서비스에서는 아이템 기반이 계산과 설명(\"이걸 본 사람들이 본\")에 유리해요.</p>"
           + K.code_table([("cosine(X)", "열(아이템)끼리 코사인 유사도, 자기 자신은 0"), ("X @ item_sim", "내가 본 아이템들과의 유사도 합"), ("user_sim @ X", "닮은 사용자들의 기록을 유사도만큼 섞기")]), None),
    K.step(4, "숨은 취향 찾기: 행렬 분해 (BPR)", "사용자와 아이템을 64칸 벡터로 두고, 내적이 \"본 것 > 안 본 것\"이 되게 배워요.",
           "<p><b>행렬 분해</b>는 사용자 × 아이템 표를 (사용자 × d) 행렬과 (d × 아이템) 행렬의 곱으로 근사해요. d칸 벡터는 사람이 정하지 않은 숨은 취향 축이에요(이 데이터라면 장르 비슷한 것). 점수는 두 벡터의 내적이에요.</p>"
           "<p>암묵적 피드백에서는 빈칸이 정답 0이 아니라서, 점수 자체를 맞히는 대신 <b>순서</b>를 배워요. <b>BPR</b>은 \"본 아이템 i의 점수가 아무거나 뽑은 아이템 j보다 크다\"의 확률 σ(점수ᵢ − 점수ⱼ)를 키워요. 안 본 것을 뽑아 쓰는 걸 <b>음성 샘플링</b>이라고 해요.</p>"
           + K.code_table([("P, Q", "사용자 벡터, 아이템 벡터 (각 64칸)"), ("x = (P[u] * (Q[i] - Q[j])).sum(dim=1)", "본 것 점수 − 뽑은 것 점수"), ("-logsigmoid(x)", "BPR 손실: 순서가 맞을수록 작아요"),
                           ("reg * (...)", "벡터가 커지지 않게 누르는 규제")])
           + K.try_("규제 세기 reg를 0.01, 0.05, 0.1로 바꾸면?", "t4"), None),
    K.step(5, "닫힌 식 하나로: EASE", "아이템 × 아이템 가중치 B를 \"내가 본 것들로 각 아이템을 다시 만들기\"로 두고, 역행렬 하나로 한 번에 풀어요.",
           "<p><b>EASE</b>는 은닉층이 없는 아주 얕은 AutoEncoder예요. 사용자 기록 x를 넣어 xB로 다시 만들되, 아이템이 자기 자신을 베끼지 못하게 B의 대각선을 0으로 묶어요. 이 문제는 <code>B = I − P × diag(1 ÷ diag(P)), P = (XᵀX + λI)⁻¹</code>로 한 번에 풀려서 학습 반복이 없어요. 아이템이 수만 개를 넘으면 역행렬이 무거워져요.</p>"
           + K.code_table([("np.linalg.inv(X.T @ X + lam * np.eye(I))", "P = (XᵀX + λI)의 역행렬"), ("-P / np.diag(P)[None, :]", "열마다 대각 값으로 나누기"), ("np.fill_diagonal(B, 0)", "자기 자신 베끼기 금지")])
           + K.try_("λ를 50, 200, 500으로 바꾸면?", "t5"), None),
    K.step(6, "이용 기록을 그래프로: LightGCN", "사용자와 아이템을 노드로, 본 기록을 선으로 그린 그래프에서 이웃 벡터를 모아 행렬 분해를 강화해요.",
           "<p>이용 기록은 사용자와 아이템 사이에만 선이 있는 그래프예요. <a href=\"gnn.html\">GNN</a>처럼 이웃 벡터를 모으면 1층에서 \"내가 본 아이템\", 2층에서 \"그걸 본 다른 사람들\", 3층에서 \"그 사람들이 본 아이템\"까지 정보가 와요. NGCF는 여기에 가중치와 활성화 함수를 넣었는데, LightGCN은 그걸 빼고 정규화한 합만 남겼더니 더 잘 됐어요. 층마다의 벡터를 평균 내서 최종 벡터로 써요.</p>"
           + K.out("s3") + K.out("f_cmp")
           + K.code_table([("A_norm", "사용자와 아이템을 함께 둔 정규화 인접 행렬 (1,000 × 1,000)"), ("h = A_norm @ h", "이웃 벡터 모으기 (가중치, 활성화 없음)"), ("torch.stack(out).mean(dim=0)", "0층부터 마지막 층까지 평균")])
           + K.try_("층 수를 0, 1, 2, 3으로 바꾸면?", "t6"), None),
    K.step(7, "기록이 적은 사용자와 인기 편향", "기록 수로 사용자를 셋으로 나눠 점수를 보고, 추천 목록이 인기 아이템으로 얼마나 기우는지 재요.",
           K.out("s7") + K.out("s7b")
           + "<p><b>더 나아간 추천들</b>: 행렬 분해의 내적을 신경망으로 바꾼 NCF, 사용자 기록을 통째로 넣고 다시 만드는 AutoRec과 Multi-VAE(<a href=\"generative-models.html\">생성 모델</a>의 VAE), 아이템 내용이나 사용자 정보(나이, 장르, 지식 그래프)를 함께 넣는 Wide &amp; Deep, 본 순서를 BERT처럼 가려 맞히는 BERT4Rec이 있어요. 서비스에서는 수백만 개 중 수백 개를 빨리 고르는 <b>후보 생성</b>(사용자와 아이템 벡터의 근사 최근접 검색, <a href=\"../backend/embedding.html\">임베딩 검색</a>)과 그 수백 개를 꼼꼼히 순서 매기는 <b>랭킹</b> 두 단계로 나눠요.</p>"
           "<p><b>LLM으로 추천하기</b>: 음악을 오디오 토큰과 글 토큰으로 바꿔 LLM이 대화로 추천하게 하는 시도도 있어요. \"비 오는 날 듣기 좋은, 기타 위주 90년대 노래\"처럼 말로 한 요청과 콜드 스타트에 강하지만, 아이템이 수십만 개면 결국 검색과 함께 써요.</p>"
           + K.code_table([("np.quantile(activity, [1 / 3, 2 / 3])", "기록 수로 사용자를 셋으로"), ("np.isin(top, head).mean()", "추천 목록 중 인기 상위 10%의 비율")]), None),
]
practice = [("어디서부터", "인기순 → 아이템 이웃 → EASE를 먼저 같은 나누기와 지표로 재고, 학습형 모델은 그걸 넘을 때만 들여요(③)."),
            ("오프라인 점수와 실제", "숨긴 기록 맞히기는 \"원래 볼 것\"을 맞히는 거라, 새로운 걸 발견하게 해 주는 추천은 낮게 나와요. 최종 판단은 A/B 검정으로(<a href=\"stats-tests.html\">가설 검정</a>)."),
            ("새 사용자, 새 아이템", "협업 필터링은 기록이 없으면 못 써요. 가입 때 관심사 묻기, 인기순, 아이템 내용 기반 추천을 섞어 시작해요."),
            ("추천이 한쪽으로 몰린다", "인기 편향(⑦), 비슷한 것만 계속 보여 주는 문제가 생겨요. 다양성 점수를 섞거나 탐색(<a href=\"rl-basics.html\">강화학습</a>의 밴딧)을 넣어요.")]
libs = [("①②③⑤⑦", "<code>numpy</code>", "기록, 지표, 이웃, EASE"), ("④⑥", "<code>torch</code>", "BPR 행렬 분해, LightGCN")]
K.build(ROOT / "topics" / "model" / "recsys.html", title="추천 시스템", h1="추천 시스템: 아직 안 본 것 중 좋아할 것 고르기", track="모델",
        crumb="모델 &gt; 추천과 평생 학습",
        one_line="이용 기록 표의 <b>빈칸 중 볼 만한 것</b>을 골라요. 같이 본 기록으로 닮은 것을 찾는 <b>협업 필터링</b>, 숨은 취향 벡터를 찾는 <b>행렬 분해</b>, 닫힌 식의 <b>EASE</b>, 그래프의 <b>LightGCN</b>을 상위 k개 지표로 비교하고, <b>인기 편향과 콜드 스타트</b>를 함께 봐요.",
        why=why, whole=whole, steps=steps, code_file="examples/recsys/recommend.py", practice=practice, libs=libs,
        libs_note="<code>evaluate</code>, <code>cosine</code>, <code>bpr_mf</code>, <code>ease</code>, <code>lightgcn</code>, <code>ndcg_by_group</code>, <code>head_share</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"gnn.html\">그래프 신경망</a>", "이웃 모으기, 과평활"), ("<a href=\"../backend/embedding.html\">임베딩과 벡터 검색</a>", "후보 생성"), ("<a href=\"generative-models.html\">생성 모델</a>", "VAE"), ("<a href=\"../backend/rag-eval.html\">RAG 평가</a>", "같은 순위 지표(재현율@k, NDCG)로 검색 결과를 재기")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: 경희대 특강(2025년 1월) \"Deep Learning 기반 추천 시스템\"(NCF, ConvNCF, AutoRec, CDAE, Multi-VAE, EASE, 사이드 정보와 Wide &amp; Deep, NGCF와 LightGCN, BERT4Rec, Cornac 실습의 MF, NeuMF, EASE, UserKNN, ItemKNN, Precision, Recall, NDCG, MAP), Upstage 10Talks Lecture 01 \"An LLM-based Music Recommendation\"(멀티모달 LLM, 토큰, 대화형 추천). 원래 실습 데이터(채용 지원 기록)는 쓰지 않고 가상 기록으로 다시 계산했어요",
                 "결과 값과 그림: <code>tools/pages/recsys.py</code>가 <code>examples/recsys/recommend.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
