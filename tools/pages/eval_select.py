"""모델 > 학습 중에 무엇으로 재나. examples/eval_select/pick_benchmarks.py 하나가 ①~⑥으로 자란다.
실행: python tools/pages/eval_select.py
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
EX = ROOT / "examples" / "eval_select"
code = S.parse((EX / "pick_benchmarks.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="pick_benchmarks.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__} (CPU 1스레드, seed 0)"
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "evalsel"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/evalsel/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
W, B, M = ns["WIDTHS"], list(ns["benchmarks"]), ns["METRICS"]
cv, LAST = ns["curves"], ns["LAST"]
SMALL, BIG, EASY, HARD = B
final = lambda b, mt, w: cv[(b, mt, w)][-LAST:].mean()  # noqa: E731
# ---------------- ①
outs["s1"] = S.table(f"① 모델 여섯 개, 에폭마다 체크포인트 {ns['EPOCHS']}개", ["숨은 층 크기", "가중치 수", f"큰 평가 정답률 (마지막 {LAST}개 평균)"],
                     [[w, f"{64 * w + w + w * 10 + 10:,}", f3(final(BIG, '정답률', w))] for w in W],
                     ["큰 LLM을 만드는 팀은 학습 도중에 저장한 체크포인트를 계속 평가해서 \"잘 되고 있나\", \"어느 설정이 나은가\"를 정해요. 여기서는 크기가 비슷한 작은 모델 여섯 개가 그 역할이에요. 모델이 클수록 조금씩 나아요."])

# ---------------- ②
fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), sharey=False)
cols = plt.cm.Purples(np.linspace(0.35, 1, len(W)))
ep = np.arange(1, ns["EPOCHS"] + 1)
for ax, b in zip(axs, B):
    for w, c in zip(W, cols):
        ax.plot(ep, cv[(b, "정답률", w)], color=c, lw=1.6, label=f"크기 {w}")
    ax.set(title=b, xlabel="에폭")
axs[0].set_ylabel("정답률")
axs[0].legend(fontsize=7, loc="lower right")
P_CURVES = fig_save(fig, "curves")
outs["f_curves"] = chart("평가마다 에폭별 정답률", P_CURVES, "",
                         ["진할수록 큰 모델이에요. 작은 평가는 선이 들쭉날쭉하고 순서가 자주 바뀌어요. 큰 평가는 매끄럽고 순서가 잘 유지돼요.",
                          "쉬운 평가는 크기와 상관없이 비슷한 곳에 모여 있어요. 아직 너무 어려운 평가는 모두 찍는 수준(0.1)에 붙어 있어요."])
outs["s2"] = S.table("② 평가 네 개 (마지막 체크포인트 10개 평균 정답률)", ["평가", "문제 수", *[f"크기 {w}" for w in W]],
                     [[b, len(ns["benchmarks"][b][1]), *[f3(final(b, '정답률', w)) for w in W]] for b in B],
                     ["\"-손실 (비트)\"은 정답에 준 확률의 -log₂를 평균 내 음수로 만든 값이에요. 정답률은 맞았나 틀렸나(0 또는 1)만 보지만, 손실은 \"정답에 얼마나 확신했나\"까지 봐서 더 매끄럽게 변해요. Olmo 3는 작은 모델의 초기 평가에 이런 연속 지표(바이트당 비트)를 써요."])

# ---------------- ③
st = ns["snr_table"]
S.claim(st[(SMALL, "-손실 (비트)")][2] > st[(SMALL, "정답률")][2] and st[(BIG, "정답률")][2] > st[(SMALL, "정답률")][2] and st[(EASY, "정답률")][2] < 1.5,
        "③ 손실이 정답률보다, 큰 평가가 작은 평가보다 SNR이 높고, 쉬운 평가 정답률은 1.5 아래")
outs["s3"] = S.table("③ 신호 대 잡음비 (SNR)", ["평가", "지표", "신호 (모델끼리 차이)", "잡음 (체크포인트끼리 흔들림)", "SNR"],
                     [[b, mt, f"{v[0]:.4f}", f"{v[1]:.4f}", f"{v[2]:.1f}"] for (b, mt), v in st.items()],
                     [f"신호는 여섯 모델의 마지막 점수가 가장 높은 것과 낮은 것이 얼마나 벌어지나, 잡음은 한 모델의 마지막 체크포인트 {LAST}개가 얼마나 흔들리나(표준편차)예요. 신호가 잡음보다 훨씬 커야 \"이 모델이 낫다\"를 믿을 수 있어요.",
                      f"같은 평가라도 손실로 재면 SNR이 커요(작은 평가 {st[(SMALL, '정답률')][2]:.1f} → {st[(SMALL, '-손실 (비트)')][2]:.1f}). 정답률은 경계에 걸친 몇 문제가 맞았다 틀렸다 하며 뛰는데, 손실은 확신이 조금씩 바뀌는 걸 그대로 보여 줘요.",
                      f"쉬운 평가의 정답률은 SNR이 {st[(EASY, '정답률')][2]:.1f}예요. 모델끼리 차이가 잡음보다 작아서 순서를 정할 수 없어요. 강연에서 말한 \"너무 쉬운 평가(천장)\"와 같은 문제예요."])
fig, ax = plt.subplots(figsize=(8.4, 3.4))
lab = [f"{b.split(' (')[0]}\n{mt}" for b, mt in st]
colsnr = [C["purple"] if mt == "정답률" else C["blue"] for _, mt in st]
ax.bar(range(len(st)), [v[2] for v in st.values()], color=colsnr)
ax.set_xticks(range(len(st)), lab, fontsize=8)
ax.set(ylabel="SNR", title="보라: 정답률, 파랑: -손실 (비트)")
P_SNR = fig_save(fig, "snr")
outs["f_snr"] = chart("평가와 지표별 SNR", P_SNR, "",
                      ["짝마다 파랑(손실)이 보라(정답률)보다 높아요. 그런데 \"아직 너무 어려운 평가\"의 손실은 SNR이 높은데도 쓸모가 없어요. ⑤에서 이유를 봐요."])

# ---------------- ④
sz = ns["size_noise"]
m16 = ns["models"][16][0]
m16.load_state_dict(ns["models"][16][1][-1])
P16 = ns["score"](m16, (ns["X"][ns["pool_np"]], ns["y"][ns["pool_np"]]), "정답률")
sig_acc = [st[(b, "정답률")][0] for b in (SMALL, BIG)]
S.claim(sz[20] > 3 * sz[300], "④ 20문제는 300문제보다 3배 넘게 흔들림")
outs["s4"] = S.table("④ 같은 모델, 평가 문제만 다시 뽑으면 (크기 16, 마지막 체크포인트, 200번)", ["문제 수", "정답률의 표준편차", "√(p(1−p)/n)"],
                     [[n, f3(v), f3(np.sqrt(P16 * (1 - P16) / n))] for n, v in sz.items()],
                     [f"모델은 그대로인데 문제 20개짜리 평가는 뽑을 때마다 정답률이 ±{sz[20]:.3f}쯤 흔들려요. 모델 여섯 개의 차이(③의 정답률 신호, {min(sig_acc):.3f}~{max(sig_acc):.3f})와 비슷한 크기예요. 그러니 작은 평가로는 순서를 믿기 어려워요.",
                      f"오른쪽 열은 맞을 확률 p = {P16:.2f}(이 모델의 전체 정답률)인 문제를 n개 풀 때의 이론 값이에요. 표준편차는 1/√n로 줄어서, 흔들림을 절반으로 줄이려면 문제가 네 배 필요해요. 700개에서 이론보다 훨씬 작은 건, 문제가 {len(ns['pool_np'])}개뿐인 풀에서 700개를 뽑으면 매번 거의 같은 문제가 뽑히기 때문이에요."])

# ---------------- ⑤
ag = ns["agree"]
S.claim(ag[(HARD, "-손실 (비트)")][10] < 0 and ag[(SMALL, "-손실 (비트)")][10] > ag[(SMALL, "정답률")][10], "⑤ 어려운 평가 손실은 순서가 거꾸로, 작은 평가는 손실이 정답률보다 잘 맞힘")
outs["s5"] = S.table("⑤ 에폭 2, 5, 10에서 잰 모델 순서가 끝의 순서와 얼마나 같나 (Spearman, 1이면 같음)", ["평가", "지표", "에폭 2", "에폭 5", "에폭 10"],
                     [[b, mt, *[f"{v[e]:.2f}" for e in (2, 5, 10)]] for (b, mt), v in ag.items()],
                     ["끝의 순서는 \"큰 평가의 손실, 마지막 10개 평균\"으로 정했어요. 학습 초반의 평가로 \"이 설정이 낫다\"를 일찍 정할 수 있는지 보는 거예요(강연의 스케일링 분석).",
                      f"아직 너무 어려운 평가의 손실은 SNR이 {st[(HARD, '-손실 (비트)')][2]:.1f}로 높았는데, 순서는 에폭 10에서 {ag[(HARD, '-손실 (비트)')][10]:.2f}로 거꾸로예요. 큰 모델일수록 모르는 문제에도 더 확신해서 틀리기 때문이에요. 신호가 크다고 우리가 원하는 신호인 건 아니에요. 그래서 능력이 아직 생기지 않은 평가(창발 전)는 그 능력이 나타나는 규모까지 빼 둬요.",
                      f"작은 평가는 정답률로 재면 에폭 10에 {ag[(SMALL, '정답률')][10]:.2f}, 손실로 재면 {ag[(SMALL, '-손실 (비트)')][10]:.2f}예요. 문제가 적을 때는 연속 지표가 특히 도움이 돼요."])

# ---------------- ⑥
corr, names, groups = ns["corr"], ns["names"], ns["groups"]
fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(corr, cmap="PuOr_r", vmin=-1, vmax=1)
short = [f"{b.split(' (')[0]} / {'정답률' if mt == '정답률' else '손실'}" for b, mt in names]
ax.set_xticks(range(len(names)), short, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(names)), short, fontsize=8)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(corr[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.04)
ax.set_title("평가끼리 점수가 함께 움직이나")
ax.grid(False)
P_CORR = fig_save(fig, "corr")
gmap = {}
for n, g in zip(names, groups):
    gmap.setdefault(int(g), []).append(f"{n[0]} / {n[1]}")
S.claim(len(gmap) == 3 and groups[names.index((SMALL, "정답률"))] == groups[names.index((BIG, "-손실 (비트)"))], "⑥ 세 묶음, 작은 정답률과 큰 손실이 같은 묶음")
outs["f_corr"] = chart("평가 여덟 개의 상관", P_CORR, "", ["모든 모델과 모든 체크포인트의 점수로 잰 상관이에요. 위쪽 여섯 칸(같은 능력: 원래 숫자 읽기)은 서로 강하게 함께 움직여요.", "픽셀을 섞은 평가의 정답률은 다른 평가와 거의 상관이 없고(0 근처), 손실은 거꾸로 움직여요(음수). 원래 숫자를 잘 읽게 될수록 섞인 숫자에는 더 확신하며 틀리기 때문이에요."])
outs["s6"] = S.table("⑥ 상관으로 묶은 결과 (세 묶음)", ["묶음", "평가"], [[g, "<br>".join(v)] for g, v in sorted(gmap.items())],
                     ["원래 숫자를 읽는 평가 여섯 개가 한 묶음이 됐어요. 같은 능력을 재는 평가가 여러 개면 그 묶음 안에서 평균을 내고, 묶음끼리는 같은 무게로 평균(macro average)을 내요. 그래야 비슷한 평가가 많은 능력이 전체 점수를 독차지하지 않아요.",
                      "Olmo 3는 공개 모델 70개의 점수로 평가들을 이렇게 묶고, 묶음마다 SNR이 높은 평가를 골라 평가 모음(OlmoBaseEval)을 만들었어요."])

flow = [[1, "체크포인트", "모델 여섯 개, 에폭마다", "학습 중 평가"], [2, "평가 넷, 지표 둘", "작음, 큼, 쉬움, 어려움", "정답률과 손실"], [3, "신호 대 잡음", "모델 차이 ÷ 흔들림", "손실이 매끄러움"],
        [4, "문제 수", "다시 뽑기", "1/√n"], [5, "초반 순서", "Spearman", "창발 전 평가는 거꾸로"], [6, "묶기", "상관, 군집", "macro 평균"]]
S.save(ROOT / "topics" / "model" / "eval-select-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow, "env": ENV})

why = """<p>큰 모델은 GPU 수천 개로 몇 달을 학습해요. 그동안 \"잘 되고 있나\", \"이 데이터 섞기가 나은가\"를 계속 판단해야 하는데, 검증 손실 하나로는 수학, 코드, 한국어 같은 능력별 진척을 알 수 없어요. 그렇다고 평가를 다 돌리면 오래 걸리고, 어떤 평가는 숫자가 너무 흔들리거나 너무 쉽거나 아직 너무 어려워서 판단에 도움이 안 돼요.</p>"""
why += K.table(["평가의 문제", "증상", "이 쪽"], [["잡음", "같은 모델인데 체크포인트마다, 문제를 다시 뽑을 때마다 점수가 뜀", "③④"], ["천장 (너무 쉬움)", "모델끼리 차이가 안 남", "③"],
                                           ["창발 전 (너무 어려움)", "다 찍는 수준이거나, 순서가 거꾸로", "⑤"], ["겹침", "같은 능력을 재는 평가가 여러 개", "⑥"]])
why += "<p>이 쪽은 손글씨 숫자를 읽는 작은 모델들로 같은 판단을 해 봐요. <a href=\"ml-learning-theory.html\">학습 이론</a>의 \"몇 건이면 믿을 수 있나\"와 같은 질문을 평가 쪽에서 묻는 거예요.</p>"
whole = """<p>체크포인트 → 평가 넷과 지표 둘 → 신호 대 잡음비 → 문제 수 → 초반 순서 → 평가 묶기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/eval_select/pick_benchmarks.py</code> 하나예요. 데이터는 scikit-learn에 들어 있어요.</p>"""
steps = [
    K.step(1, "크기가 다른 모델, 에폭마다 체크포인트", f"숨은 층 크기 {W[0]}~{W[-1]}의 모델 여섯 개를 {ns['EPOCHS']}에폭 학습하며 에폭마다 가중치를 저장해요.",
           "<p>체크포인트는 학습 중간의 가중치 사본이에요. 같은 모델의 연속한 체크포인트는 실력이 거의 같아서, 그 점수가 흔들리는 만큼이 그 평가의 <b>잡음</b>이에요. 서로 다른 모델의 점수 차이가 <b>신호</b>예요.</p>"
           + K.code_table([("m.state_dict()", "지금의 가중치"), ("v.clone()", "학습이 계속돼도 바뀌지 않게 복사"), ("m.load_state_dict(state)", "저장한 체크포인트로 되돌리기")]), "s1"),
    K.step(2, "평가 넷, 지표 둘", "남겨 둔 숫자로 작은 평가(40문제), 큰 평가(600문제), 쉬운 평가(0과 1만), 아직 너무 어려운 평가(픽셀을 섞은 숫자)를 만들고, 정답률과 손실로 재요.",
           K.out("f_curves")
           + K.code_table([("(logits.argmax(1) == y).mean()", "정답률"), ("−cross_entropy / log 2", "−손실 (비트): 정답에 준 확률의 −log₂ 평균의 음수")]), "s2"),
    K.step(3, "신호 대 잡음비", "평가와 지표마다 \"모델끼리 차이\"를 \"체크포인트끼리 흔들림\"으로 나눠요.",
           "<p>SNR이 낮은 평가로 두 설정을 비교하면, 어느 쪽이 나은지가 아니라 어느 체크포인트를 골랐는지에 따라 결론이 바뀌어요. Olmo 3는 13B 모델의 마지막 체크포인트 50개와 비슷한 계산량의 외부 모델 10개로 평가마다 SNR을 재서, 흔들리는 평가를 뺐어요.</p>"
           + K.out("f_snr")
           + K.code_table([("finals.max() − finals.min()", "신호: 모델끼리 차이"), ("curve[−10:].std()", "잡음: 마지막 체크포인트 10개의 흔들림")]), "s3"),
    K.step(4, "작은 평가가 흔들리는 이유: 문제 수", "같은 모델로 평가 문제를 n개씩 200번 다시 뽑아, 정답률이 얼마나 흔들리는지 재요.",
           "<p>평가 점수는 \"이 문제들에서의 정답률\"이지 모델의 진짜 실력이 아니에요. 문제 n개로 잰 정답률은 동전을 n번 던진 비율처럼 흔들려요(표준편차 √(p(1−p)/n)). 그래서 작은 평가에서 1~2%p 차이는 거의 의미가 없어요.</p>"
           + K.code_table([("r.choice(len(y), n, replace=False)", "평가 문제 n개 다시 뽑기"), ("np.std(vals)", "200번의 흔들림")]), "s4"),
    K.step(5, "학습 초반에 잰 순서가 끝까지 맞나", "에폭 2, 5, 10의 점수로 매긴 모델 순서를, 끝(큰 평가의 손실)의 순서와 비교해요.",
           "<p>실제로는 작은 모델이나 학습 초반의 결과로 큰 실험을 할지 정해요. 그러려면 초반 평가의 순서가 나중에도 유지돼야 해요. Olmo 3는 평가마다 \"계산량이 얼마일 때부터 능력이 나타나나\"를 봐서, 작은 규모에는 쉬운 평가 모음(Base Easy), 중간 학습에는 본 평가(Base Main)를 쓰도록 나눴어요.</p>"
           + K.code_table([("spearmanr(early, TRUTH)", "두 순서가 같은 정도")]), "s5"),
    K.step(6, "비슷한 평가끼리 묶기", "모든 모델과 체크포인트의 점수로 평가끼리 상관을 재고, 계층 군집으로 세 묶음으로 나눠요.",
           K.out("f_corr")
           + K.code_table([("np.corrcoef(profile)", "평가끼리 점수의 상관"), ("linkage(..., metric=\"correlation\")", "상관이 높은 것부터 묶기"), ("fcluster(..., t=3)", "세 묶음으로 자르기")]), "s6"),
]
practice = [("학습 중에 진척을 본다", "체크포인트 몇 개로 평가마다 흔들림부터 재요. 비교하려는 차이가 흔들림보다 작으면 그 평가로는 판단하지 않아요."),
            ("작은 모델로 미리 시험한다", "정답률보다 손실(정답에 준 확률) 같은 연속 지표를 써요. 아직 다 찍는 수준인 평가는 빼고, 그 능력이 나타나는 규모가 되면 넣어요."),
            ("평가 모음을 만든다", "비슷한 평가를 묶어 묶음끼리 같은 무게로 평균 내요. 묶음마다 SNR이 높고 빨리 도는 평가를 대표로 골라요.")]
libs = [("①②", "<code>torch</code>, <code>sklearn.datasets.load_digits</code>", "작은 모델, 체크포인트"), ("⑤⑥", "<code>scipy.stats.spearmanr</code>, <code>scipy.cluster.hierarchy</code>", "순위 상관, 계층 군집")]
K.build(ROOT / "topics" / "model" / "eval-select.html", title="학습 중에 무엇으로 재나", h1="학습 중에 무엇으로 재나: 평가의 신호와 잡음", track="모델",
        crumb="모델 &gt; 순서와 언어 모델",
        one_line="학습 중 평가는 <b>모델끼리 차이(신호)</b>가 <b>체크포인트끼리 흔들림(잡음)</b>보다 훨씬 커야 믿을 수 있어요. 문제가 적거나 너무 쉬운 평가는 신호가 잡음에 묻히고, <b>정답률보다 손실</b>이 더 매끄러워요. 아직 너무 어려운 평가는 신호가 커 보여도 <b>순서가 거꾸로</b>일 수 있어요.",
        why=why, whole=whole, steps=steps, code_file="examples/eval_select/pick_benchmarks.py", practice=practice, libs=libs,
        libs_note="<code>train_with_checkpoints</code>, <code>score</code>, <code>curve</code>, <code>snr</code>, <code>sample_noise</code>, <code>early_agreement</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"pretrain-data.html\">LLM의 학습 데이터 만들기</a>", "같은 강연의 데이터 쪽"), ("<a href=\"ml-learning-theory.html\">학습 이론</a>", "몇 건이면 믿을 수 있나"), ("<a href=\"../backend/llm-judge.html\">LLM으로 답 채점하기</a>", "채점 자체의 흔들림")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: Upstage 10Talks Lecture 04 \"Technical Choices and Strategy: Lessons from Olmo 3\" 1부(검증 손실을 넘는 평가, 모든 평가를 쓰지 않는 이유: 시간, 잡음, 천장, 창발, 평가 군집과 macro 평균, 스케일링 분석과 바이트당 비트, 신호 대 잡음 분석, OlmoBaseEval). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/eval_select.py</code>가 <code>examples/eval_select/pick_benchmarks.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
