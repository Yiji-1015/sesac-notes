"""모델 > 언어 모델로 문제 풀기 > 모델 가볍게. examples/compress/compress_classifier.py 하나가 ①~⑦로 자란다.
NLP with Transformers 8장(효율적인 Transformer: 지식 정제, 양자화, 가지치기)을 합친 쪽.
실행: HF_HUB_OFFLINE=1 python tools/pages/compress.py   (CPU로 3분쯤)
e5 모델을 받아 두지 않은 PC에서는 아무 파일도 바꾸지 않고 끝낸다.
"""
import html
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402
import transformers  # noqa: E402

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "compress"
code = S.parse((EX / "compress_classifier.py").read_text(encoding="utf-8"))
try:
    ns = S.run(code, 7, EX, filename="compress_classifier.py")   # 파일을 쓰지 않는 예제라 원래 폴더에서 바로 실행
except OSError:
    print("e5 모델(intfloat/multilingual-e5-small)이 없어서 이 쪽은 다시 만들지 않아요. 기존 파일을 그대로 둬요.")
    sys.exit(0)

from vstyle import plt, C, setup  # noqa: E402

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "compress"
IMG.mkdir(parents=True, exist_ok=True)
E, J = html.escape, K.josa
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, intfloat/multilingual-e5-small, CPU 스레드 4개"
R, W = ns["results"], ns["weights"]
base, vocab, fin = R["기준 (12층)"], R["어휘 줄이기"], R["6층 학생 + 어휘 줄이기 + int8"]
q_t, q_c = R["어휘 줄이기 + int8 (층마다 눈금)"], R["어휘 줄이기 + int8 (출력 칸마다 눈금)"]
same = [base, vocab] + [v for k, v in R.items() if k.startswith("가지치기")]   # 계산이 기준과 똑같은 모델들 (기준 자신, ③, ⑥)
jit = max(abs(v["x"] - 1) for v in same)                                      # 그래도 생기는 기준 대비 차이 = 재는 때의 흔들림
band = max(0.1, jit)                       # 이보다 작은 시간 차이는 같은 걸로 봐요
for k, v in R.items():                     # 주장 검사가 실패할 때 원인을 볼 수 있게 이번 실행 값을 먼저 찍기
    print(f"{k:28s} {v['MB']:7.1f} MB {v['ms']:6.1f} ms  x{v['x']:.2f}  F1 {v['F1']:.3f}")
print(f"흔들림 {jit:.0%}, 기준 띠 {band:.0%}")
pr = {k: v for k, v in R.items() if k.startswith("가지치기")}


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/compress/{name}.png"


def row(name, r, bold=False):
    b = (lambda x: f"<b>{x}</b>") if bold else (lambda x: x)  # noqa: E731
    return [b(name), f"{r['MB']:.1f} MB", f"{r['ms']:.1f} ms", f"{r['x']:.2f}배", b(f"{r['F1']:.3f}")]


COLS = ["모델", "저장 크기", "문장 하나 (CPU)", "기준 대비 시간", "test macro F1"]
outs = {}
# ---------------- 왜
S.claim(fin["MB"] < base["MB"] / 20 and fin["x"] < 0.8 and fin["F1"] < base["F1"], "왜: 합치면 20분의 1 아래 크기, 시간도 0.8배 아래로 줄고, 대신 F1이 떨어진다")
outs["why"] = S.table("같은 test 120문장으로 잰 결과 (자세한 건 ⑦)", COLS, [row("문장 분류 쪽의 분류기 그대로", base), row("이 쪽에서 가볍게 만든 것", fin, True)],
                      [f"크기는 {base['MB'] / fin['MB']:.0f}분의 1, 시간은 기준의 {fin['x']:.0%}로 줄고, 정확도는 {base['F1'] - fin['F1']:.2f} 떨어졌어요.",
                       "가볍게 만드는 방법마다 줄여 주는 것(크기, 시간)과 잃는 것(정확도)이 달라요. 이 쪽은 방법을 하나씩 따로 재서, 무엇이 무엇을 줄이는지 나눠 봐요."])

# ---------------- ①
outs["s1"] = S.table("① 기준 모델 (문장 분류 쪽 ⑤와 같은 방법, 위쪽 4층 미세조정)", COLS, [row("기준 (12층)", base, True)],
                     ["<b>저장 크기</b>: <code>torch.save</code>로 저장한 파일 크기예요. 서버의 메모리, 앱의 내려받기 크기와 같은 이야기예요.",
                      "<b>문장 하나</b>: 한 번 데운 뒤 문장 120개를 하나씩 넣는 걸 다섯 번 반복해서 가장 빠른 평균이에요. 실제 서비스처럼 요청이 하나씩 올 때의 시간이에요.",
                      f"<b>기준 대비 시간</b>: ms는 그때 PC가 얼마나 바쁜지에 따라 크게 달라져요. 모델을 하나씩 다른 때에 재면 그 차이가 모델 차이처럼 보여서, 잴 때마다 기준 모델과 번갈아 재고 기준의 몇 배인지를 봐요. "
                      f"그래도 계산이 기준과 똑같은 모델(기준 자신, 뒤에 나올 ③과 ⑥)이 최대 {jit:.0%}까지 달리 재졌으니, {band:.0%}보다 작은 차이는 같은 걸로 봐요.",
                      "<b>macro F1</b>: 문장 분류 쪽과 같은 test(처음 보는 말투)예요."])

# ---------------- ②
tot = sum(W.values())
names_w = list(W.keys())
fig, ax = plt.subplots(figsize=(8, 2.4))
left = 0
for nm, col in zip(names_w, [C["pink"], C["orange"], C["purple"], C["mint"]]):
    ax.barh([0], [W[nm] / 1e6], left=left, color=col, label=f"{nm} {W[nm] / 1e6:.1f}M ({W[nm] / tot:.0%})")
    left += W[nm] / 1e6
ax.set(yticks=[], xlabel="가중치 수 (백만 개)", title="기준 모델의 가중치가 어디에 있나")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=2, frameon=False)
outs["f_w"] = S.images("가중치 1억 1,765만 개의 자리", [(fig_save(fig, "weight_share"), "")], [])
outs["f_w"]["chart"] = True
emb_share = W["토큰 Embedding 표"] / tot
S.claim(emb_share > 0.75, "② 가중치의 3/4 넘게가 토큰 Embedding 표")
outs["s2"] = S.box("② 자리별 가중치 수", [*(f"{k} → {v:,}개 ({v / tot:.0%})" for k, v in W.items()), (f"합계 → {tot:,}개", 1)],
                   [f"가중치의 {emb_share:.0%}가 토큰 번호 → 벡터 표예요. 25만 개 어휘에 384칸씩이에요. 실제로 \"생각\"하는 Transformer 층 12개는 {W['Transformer 층 12개'] / tot:.0%}뿐이에요. 여러 언어를 다루는 작은 모델일수록 이렇게 어휘 표가 큰 몫을 차지해요."])
outs["f_w"]["notes"] = [f"분홍(토큰 Embedding 표)이 {emb_share:.0%}예요. 무엇을 줄일지 고르기 전에, 무게가 어디에 있는지부터 봐요."]

# ---------------- ③
n_vocab = len(ns["tok"])
S.claim(abs(vocab["F1"] - base["F1"]) < 1e-9 and vocab["MB"] < base["MB"] / 4, "③ 어휘를 줄여도 F1은 똑같고 크기는 4분의 1 아래")
outs["s3"] = S.table(f"③ 어휘 {n_vocab:,}개 중 이 데이터에 나오는 {len(ns['used'])}개만 남기면", COLS, [row("기준 (12층)", base), row("어휘 줄이기", vocab, True)],
                     [f"이 일의 문장 780개에 나오는 토큰은 {len(ns['used'])}개({len(ns['used']) / n_vocab:.2%})뿐이에요. 나머지 줄은 한 번도 쓰이지 않는 표예요.",
                      f"쓰는 줄만 남겼더니 크기가 {base['MB'] / vocab['MB']:.1f}분의 1이 됐고, 정확도는 소수점 셋째 자리까지 똑같아요. 쓰는 줄의 값은 그대로라서 계산 결과가 같아요. 시간 차이는 재는 때의 흔들림이에요. 계산은 똑같고, 표에서 줄을 찾는 건 원래 아주 빠른 일이라서요.",
                      f"⚠️ 대신 이 {len(ns['used'])}개에 없는 토큰이 들어오면 모두 <code>&lt;unk&gt;</code>로 바뀌어요. 쓸 때 들어올 글을 넉넉히 모아 어휘를 정해야 해요."])

# ---------------- ④
teacher, tok = ns["teacher"], ns["tok"]
ex = ns["X"][0]
with torch.no_grad():
    lg = teacher(ns["encode"]([ex]))[0]
p1, p2 = torch.softmax(lg, -1), torch.softmax(lg / 2, -1)
sf = ns["student_f1"]
mean = {k: sum(v) / len(v) for k, v in sf.items()}
spread = max(max(v) - min(v) for v in sf.values())              # 처음 값에 따른 흔들림 (가장 큰 것)
gap = max(mean.values()) - min(mean.values())                   # 방법 사이 평균 차이
stu = R["6층 학생"]
S.claim(stu["x"] < 0.8 and max(mean.values()) < base["F1"] - 0.05 and spread > gap, "④ 학생은 기준의 0.8배 안쪽 시간, 평균 F1은 기준보다 0.05 넘게 낮고, 방법 차이가 처음 값에 따른 흔들림보다 작다")
outs["s4"] = S.table("④ 아래 6층만 가진 학생 (위쪽 2층과 분류 층만 학습, 분류 층의 처음 값 4가지)", ["학습 방법", "처음 값 0", "1", "2", "3", "평균"],
                     [[k] + [f"{x:.3f}" for x in v] + [f"<b>{mean[k]:.3f}</b>"] for k, v in sf.items()] + [["(참고) 기준 12층", "", "", "", "", f"{base['F1']:.3f}"]],
                     [f"층이 절반이라 문장 하나의 시간이 기준의 {stu['x']:.0%}예요. 크기는 {stu['MB']:.0f}MB로 거의 그대로인데, 여전히 25만 어휘 표를 갖고 있어서예요(⑦에서 ③과 합쳐요).",
                      f"정확도는 기준({base['F1']:.3f})보다 떨어져요. 그리고 같은 방법이라도 분류 층의 처음 값에 따라 최대 {spread:.2f}만큼 흔들려요. 방법 사이의 평균 차이({gap:.2f})보다 커서, 이 실험만으로는 어느 방법이 낫다고 말하기 어려워요. 평균으로는 라벨 없는 문장까지 쓴 정제가 가장 높아요({mean['+ 라벨 없는 문장']:.3f}).",
                      f"선생의 점수에는 정답보다 많은 정보가 있어요. 예를 들어 \"{E(ex)}\"에 선생이 준 확률은 {', '.join(f'{LB} {v:.2f}' for LB, v in zip(ns['LABELS'], p1.tolist()))}{J(f'{p1[-1]:.2f}', '이에요', '예요')}. 온도 2로 나누면 {', '.join(f'{LB} {v:.2f}' for LB, v in zip(ns['LABELS'], p2.tolist()))}로 펴져서, \"정답은 아니지만 어느 쪽과 더 닮았나\"가 드러나요.",
                      "그런데 선생이 점수를 붙여 준 문장(train과 라벨 없는 문장)은 모두 train과 같은 말투예요. 학생이 떼어 낸 위쪽 6층은 처음 보는 말투를 다루던 부분인데, 그 말투의 문장이 없으니 선생도 그 능력을 보여 줄 기회가 없어요. 정제가 잘 되려면 선생이 점수를 매겨 줄 <b>다양한</b> 문장이 많아야 해요."])

# ---------------- ⑤
w, wb = ns["w"], ns["w_back"]
rel = float((w - wb).norm() / w.norm())
S.claim(q_c["MB"] < vocab["MB"] / 3 and q_t["F1"] < vocab["F1"] and q_c["F1"] >= q_t["F1"], "⑤ int8은 크기를 3분의 1 아래로 줄이지만 F1이 떨어지고, 칸마다 눈금이 더 낫다")
q_ratio = min(q_t["x"], q_c["x"]) / vocab["x"]   # int8의 시간 ÷ int8 전 시간
outs["s5"] = S.table("⑤ ③의 모델을 int8로 (몸통의 Linear 층만)", COLS, [row("어휘 줄이기 (32비트 실수)", vocab), row("+ int8, 층마다 눈금 하나", q_t), row("+ int8, 출력 칸마다 눈금", q_c, True)],
                     [f"숫자 하나가 4바이트에서 1바이트가 돼서 크기가 {vocab['MB']:.0f}MB에서 {q_c['MB']:.0f}MB가 됐어요. 남은 무게의 대부분이 Linear 층이라 거의 4분의 1이에요.",
                      f"첫 층 Query 행렬로 직접 보면, 눈금 하나({float(ns['scale']):.5f}) 단위로 반올림해서 생기는 차이가 원래 값 크기의 {rel:.1%}예요. 작아 보여도 층 12개를 지나며 쌓여서 F1이 {vocab['F1'] - q_t['F1']:.2f} 떨어졌어요. 출력 칸마다 눈금을 따로 잡으면 값의 범위에 더 잘 맞아서 덜 떨어져요({q_c['F1']:.3f}).",
                      (f"이 PC에서는 int8 전의 {q_ratio:.0%}로 빨라졌어요. " if q_ratio < 1 - band else f"이 PC에서는 int8 전의 {q_ratio:.0%}라서, ①에서 정한 흔들림({band:.0%})을 넘게 빨라지지는 않았어요. 문장 하나는 계산이 작아서, 입력을 int8로 바꾸고 되돌리는 준비 시간이 아낀 시간을 깎아 먹을 수 있어요. ")
                      + "한 번에 여러 문장을 넣거나 int8 전용 계산이 빠른 기기(휴대폰, 일부 GPU)에서 효과가 커요."])
first = w[0, :4].tolist()
back = wb[0, :4].tolist()
outs["s5w"] = S.box("양자화 전후의 가중치 (첫 층 Query 행렬의 앞 4개)", [f"원래 → {[round(v, 5) for v in first]}", f"int8로 바꿨다 되돌림 → {[round(v, 5) for v in back]}", f"정수로는 → {[int(round(v / float(ns['scale']))) for v in first]}"],
                    ["int8은 -128~127 정수예요. 가장 큰 절댓값을 127에 맞추는 눈금(scale)을 정하고, 값마다 \"눈금 몇 칸인가\"만 저장해요. 쓸 때 눈금을 곱해 되돌려요."])

# ---------------- ⑥
p50 = ns["pruned"](teacher, 0.5)
lin = [m for m in p50.body.encoder.modules() if isinstance(m, torch.nn.Linear)]
zero = sum(int((m.weight == 0).sum()) for m in lin) / sum(m.weight.numel() for m in lin)
pf = [v["F1"] for v in pr.values()]
S.claim(abs(pf[0] - base["F1"]) < 0.01 and pf[0] >= pf[1] >= pf[2] and pf[2] < base["F1"] - 0.3 and all(abs(v["MB"] - base["MB"]) < 1 for v in pr.values()), "⑥ 30%는 그대로, 지울수록 떨어지고 70%에서 무너지며, 크기는 그대로")
outs["s6"] = S.table("⑥ 몸통의 Linear 가중치 중 절댓값이 작은 것을 0으로 (다시 학습하지 않음)", COLS, [row("기준", base)] + [row(k, v) for k, v in pr.items()],
                     [f"30%를 0으로 만들어도 F1이 그대로예요. 학습된 가중치의 상당수가 0에 가까워서 지워도 결과가 거의 안 바뀌어요. 더 지우면 떨어지기 시작해서 50%에서 {pf[1]:.3f}, 70%에서는 {pf[2]:.3f}까지 무너져요. 무너지기 직전인 곳에서는 학습 때의 아주 작은 차이에도 결과가 크게 흔들려서, 한 번 잰 값을 믿기 어려워요.",
                      f"크기는 그대로이고, 시간 차이는 ①에서 말한 재는 때의 흔들림이에요. 50%를 지운 모델도 가중치의 {zero:.0%}가 0일 뿐, 여전히 같은 크기의 표에 0이 적혀 있고 0을 곱하는 계산도 그대로 해요. 크기와 시간까지 줄이려면 0을 빼고 저장하는 희소 형식과 그걸 빨리 계산하는 전용 계산이 필요해요.",
                      "실제로는 조금씩 지우고 다시 학습하기를 되풀이해요. 학습 중에 \"0 쪽으로 움직이는\" 가중치를 지우는 movement pruning이 미세조정 모델에서 더 잘 돼요. 층이나 head를 통째로 떼어 내면(④의 학생처럼) 실제로 빨라져요."])

# ---------------- ⑦
ff1 = fin["F1"]
S.claim(stu["F1"] < base["F1"] and ff1 <= stu["F1"] + 1e-9, "⑦ 학생으로 바꿀 때 잃고, 어휘와 int8을 더해도 학생보다 나아지지는 않는다")
fig, ax = plt.subplots(figsize=(8.6, 4.6))
show = ["기준 (12층)", "어휘 줄이기", "6층 학생", "어휘 줄이기 + int8 (출력 칸마다 눈금)", "가지치기 50%", "6층 학생 + 어휘 줄이기 + int8"]
lab = {"기준 (12층)": "기준", "어휘 줄이기": "③ 어휘", "6층 학생": "④ 6층 학생", "어휘 줄이기 + int8 (출력 칸마다 눈금)": "⑤ 어휘 + int8", "가지치기 50%": "⑥ 가지치기 50%", "6층 학생 + 어휘 줄이기 + int8": "⑦ 모두"}
for k in show:
    r = R[k]
    ax.scatter(r["MB"], r["F1"], s=r["x"] * 300, color=C["pink"] if k.startswith("6층 학생 +") else C["purple"], alpha=.75, edgecolors="white")
    ax.annotate(f"{lab[k]} ({r['x']:.2f}배)", (r["MB"], r["F1"]), (6, 6), textcoords="offset points", fontsize=9)
ax.set_xscale("log")
ax.minorticks_off()                                              # 로그 눈금의 작은 숫자들이 겹치지 않게
ax.set_xticks([10, 20, 50, 100, 200, 500], ["10", "20", "50", "100", "200", "500"])
ax.set_xlim(8, 1100)
ax.set(xlabel="저장 크기 (MB, 로그 눈금)", ylabel="test macro F1", title="오른쪽 위가 기준, 왼쪽 위로 갈수록 좋아요 (원 크기: 기준 대비 시간)", ylim=(0.5, 0.97))
outs["f_all"] = S.images("방법별 크기와 정확도", [(fig_save(fig, "size_vs_f1"), "")], [f"어휘 줄이기는 정확도를 잃지 않고 크기를 크게 줄여요. 학생은 시간을, int8은 크기를 줄이는 대신 정확도를 내줘요(학생 {base['F1'] - stu['F1']:.2f}, int8 {vocab['F1'] - q_c['F1']:.2f}). 가지치기는 다시 학습하지 않으면 크기도 계산량도 그대로예요."])
outs["f_all"]["chart"] = True
outs["s7"] = S.table("⑦ 모두 같은 잣대로", COLS, [row(k, v, k == "6층 학생 + 어휘 줄이기 + int8") for k, v in R.items()],
                     [f"④의 학생(라벨 없는 문장까지 정제, 처음 값 0)에 ③의 어휘 줄이기와 ⑤의 int8을 더하니 {fin['MB']:.1f}MB, {fin['ms']:.1f}ms예요. 크기는 {base['MB'] / fin['MB']:.0f}분의 1, 시간은 기준의 {fin['x']:.0%}예요.",
                      f"정확도는 {base['F1']:.3f}에서 {ff1:.3f}까지 떨어졌어요. 학생으로 바꾸며 {base['F1'] - stu['F1']:.2f}{J(f"{base['F1'] - stu['F1']:.2f}", '을', '를')} 잃었고, " + ("어휘 줄이기와 int8을 더해도 더 잃지는 않았어요." if stu['F1'] - ff1 < 0.005 else f"어휘 줄이기와 int8을 더하며 {stu['F1'] - ff1:.2f} 더 잃었어요.") + f" 이 정도를 받아들일 수 있는지는 어디에 쓰느냐에 달려요. 휴대폰에서 바로 돌려야 한다면 좋은 거래이고, 서버에서 정확도가 가장 중요하다면 ③만 하는 게 나아요."])

flow = [[1, "기준과 잣대", "크기, 시간, F1", f"{base['MB']:.0f}MB, {base['F1']:.2f}"], [2, "무게의 자리", "가중치 세기", f"Embedding {emb_share:.0%}"], [3, "어휘 줄이기", "쓰는 줄만", f"{vocab['MB']:.0f}MB, F1 그대로"],
        [4, "지식 정제", "6층 학생", f"평균 F1 {max(mean.values()):.2f}, 시간 {stu['x']:.0%}"], [5, "양자화", "int8", f"{q_c['MB']:.0f}MB"], [6, "가지치기", "작은 값을 0", "다시 학습해야"], [7, "합치기", "③+④+⑤", f"{fin['MB']:.0f}MB"]]
S.save(ROOT / "topics" / "model" / "model-compression-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p><a href="text-classification.html">문장 분류</a>에서 만든 문의 분류기는 정확하지만 무거워요. 서버 한 대가 하루 수백만 건을 처리하거나, 휴대폰이나 작은 기기에서 인터넷 없이 돌려야 한다면 크기와 속도가 정확도만큼 중요해져요. 큰 모델을 비싸게 돌리는 대신, <b>필요한 만큼만 남긴</b> 모델을 만드는 방법들이 있어요.</p>
{K.out("why")}"""
whole = """<p>기준과 잣대 정하기 → 무게가 어디 있나 → 안 쓰는 어휘 덜기 → 작은 학생에게 가르치기 → 숫자를 8비트로 → 작은 가중치 지우기 → 합치기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "기준 모델과 재는 방법", "문장 분류 쪽과 같은 방법으로 분류기를 만들고, 크기, 문장 하나의 시간, 정확도를 재는 함수를 만들어요.",
           "<p><b>왜 잣대부터?</b> 가볍게 만드는 방법은 늘 무언가를 내줘요. 같은 잣대로 재지 않으면 \"빨라졌는데 정확도가 얼마나 떨어졌나\"를 말할 수 없어요. 시간은 기기와 상황(한 번에 몇 개, 스레드 몇 개)에 따라 크게 달라서, 실제로 쓸 상황과 비슷하게 재요.</p>"
           + K.code_table([("torch.save(model.state_dict(), buf)", "가중치를 메모리 속 파일(<code>io.BytesIO</code>)에 저장해 크기 재기"), ("time.perf_counter()", "짧은 시간을 정밀하게 재는 시계"),
                           ("torch.set_num_threads(4)", "CPU 스레드 수 고정. 바꾸면 시간이 달라져요"), ("def train(..., soft=None, T=2.0, alpha=0.5)", "④의 정제까지 함께 쓰는 학습 함수")]), "s1"),
    K.step(2, "무게가 어디에 있나", "모델의 부분마다 가중치 수를 세요.",
           "<p>줄일 곳을 고르기 전에 무게가 어디에 있는지 봐요. 큰 LLM은 대부분 Transformer 층에 있지만, 여러 언어를 다루는 작은 모델은 어휘 표가 훨씬 큰 몫을 차지해요.</p>"
           + K.out("f_w")
           + K.code_table([("sum(p.numel() for p in module.parameters())", "그 부분의 가중치 개수")]), "s2"),
    K.step(3, "어휘 줄이기: 이 일에 나오는 토큰만 남기기", "데이터에 나오는 토큰 번호만 모아 Embedding 표를 그 줄들로 줄이고, 들어오는 번호를 새 번호로 바꿔 넣어요.",
           "<p>Embedding 표는 \"번호로 줄을 찾는\" 표라서, 한 번도 찾지 않는 줄은 결과에 아무 영향이 없어요. 쓰는 줄만 남기면 계산은 똑같고 크기만 줄어요. 다국어 모델을 한 언어, 한 분야에만 쓸 때 효과가 커요.</p>"
           + K.code_table([("{i for row in tok(...)[\"input_ids\"] for i in row}", "데이터 전체에서 나온 토큰 번호 모으기"), ("remap[torch.tensor(used)] = torch.arange(len(used))", "원래 번호 → 새 번호 표"),
                           ("torch.nn.Embedding(len(used), 384, _weight=...)", "쓰는 줄만 담은 새 표"), ("remap[b[\"input_ids\"]]", "들어온 번호를 한 번에 새 번호로")]), "s3"),
    K.step(4, "지식 정제: 큰 모델의 답을 작은 모델에게 가르치기", "선생(①)의 아래 6층만 떼어 학생을 만들고, 정답만으로 배울 때와 선생의 점수를 따라 배울 때를 비교해요. 분류 층의 처음 값을 4가지로 바꿔 가며 재요.",
           "<p><b>지식 정제</b>(Knowledge Distillation)는 큰 모델(선생)이 낸 확률을 작은 모델(학생)이 따라 하게 배우는 방법이에요. 정답 라벨은 \"버그다\" 하나뿐이지만, 선생의 확률은 \"버그 0.9, 기능 요청 0.08, 질문 0.02\"처럼 <b>어느 쪽과 얼마나 닮았나</b>까지 알려 줘요. 이걸 부드러운 라벨(soft label)이라고 해요.</p>"
           "<p>점수를 <b>온도 T</b>로 나눠 확률을 펴면 이 차이가 더 잘 보여요. 손실은 정답과의 손실과, 학생과 선생의 펴진 확률 사이의 차이(KL 발산)를 섞어요. 선생은 라벨 없는 문장에도 점수를 붙여 줄 수 있어서, 라벨 없는 문장까지 학습에 쓸 수 있어요.</p>"
           "<p>학생은 선생의 층을 일부 떼어 시작하면 처음부터 배우는 것보다 훨씬 빨리 좋아져요. DistilBERT가 BERT의 층을 절반으로 줄여 이렇게 만들었어요(크기 40% 감소, 속도 60% 향상, 성능 97% 유지라고 발표됐어요).</p>"
           + K.code_table([("body.encoder.layer[:n]", "아래 n개 층만 남기기"), ("F.kl_div(F.log_softmax(logits / T, -1), F.softmax(soft / T, -1), ...) * T * T", "학생과 선생의 펴진 확률 차이. T²를 곱해 크기를 맞춰요"),
                           ("alpha * loss + (1 - alpha) * kd", "정답 손실과 정제 손실을 반반"), ("torch.full((n,), -1)", "라벨 없는 문장의 정답 자리는 -1 (정답 손실에서 빼기)")]), "s4"),
    K.step(5, "양자화: 32비트 실수를 8비트 정수로", "③의 모델에서 몸통의 Linear 층 가중치를 -128~127 정수로 바꿔 저장하고 계산해요.",
           "<p>가중치는 보통 32비트 실수(4바이트)예요. 대부분의 가중치는 좁은 범위에 모여 있어서, 그 범위를 256칸으로 나눈 <b>8비트 정수</b>(1바이트)로 바꿔도 크게 틀리지 않아요. 저장 크기는 4분의 1이 되고, int8 계산을 빠르게 하는 기기에서는 빨라져요.</p>"
           "<p>눈금(scale)을 층 전체에 하나 쓰면 값이 넓게 퍼진 층에서 오차가 커져요. 출력 칸마다 눈금을 따로 잡으면(per-channel) 더 정확해요. 여기서는 쓸 때 입력도 그때그때 정수로 바꾸는 <b>동적 양자화</b>를 썼어요. 미리 데이터를 넣어 입력 범위까지 정해 두는 정적 양자화, 학습할 때부터 반올림을 흉내 내는 양자화 인지 학습(QAT)은 정확도를 더 지켜요.</p>"
           + K.out("s5w")
           + K.code_table([("torch.ao.quantization.quantize_dynamic(body, {torch.nn.Linear: qconfig}, dtype=torch.qint8)", "Linear 층만 int8로 바꾸기"), ("per_channel_dynamic_qconfig", "출력 칸마다 눈금"),
                           ("(w / scale).round().clamp(-128, 127) * scale", "직접 해 보는 양자화: 눈금으로 나눠 반올림하고 범위를 자른 뒤 되돌리기")]), "s5"),
    K.step(6, "가지치기: 크기가 작은 가중치를 0으로", "몸통의 모든 Linear 가중치를 절댓값 순으로 세워, 작은 쪽 30%, 50%, 70%를 0으로 만들어요.",
           "<p>학습이 끝난 신경망에는 0에 가까운 가중치가 많아요. 이런 가중치는 지워도 결과가 거의 안 바뀐다는 생각이에요. 절댓값이 작은 순으로 지우는 걸 magnitude pruning이라고 해요. 하나씩 지우는 비정형 가지치기와, 층이나 head, 칸을 통째로 지우는 정형 가지치기가 있어요.</p>"
           + K.code_table([("prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=0.5)", "모든 대상 층을 한데 모아 절댓값 하위 50%를 0으로"), ("prune.remove(mod, \"weight\")", "가리개(mask)를 가중치에 확정")]), "s6"),
    K.step(7, "합쳐 보기", "④의 가장 좋은 학생에 ③의 어휘 줄이기와 ⑤의 int8을 차례로 더해요.",
           "<p>방법들은 줄이는 곳이 달라서 겹쳐 쓸 수 있어요. 어휘 줄이기는 Embedding 표를, 학생은 층 수를, int8은 남은 Linear 층의 숫자 크기를 줄여요.</p>"
           + K.out("f_all"), "s7"),
]
practice = [("어디서부터 할까", "먼저 ①처럼 잣대를 정하고 ②처럼 무게를 봐요. 정확도를 잃지 않는 것(③ 어휘 줄이기, 짧은 입력은 짧게 채우기)부터 하고, 그다음 정제와 양자화를 시험해요."),
            ("ONNX와 ONNX Runtime", "모델을 프레임워크와 상관없는 계산 그래프(ONNX)로 내보내면, ONNX Runtime이 층을 합치는 등 그래프를 최적화해 CPU에서도 빨라지는 경우가 많아요. 양자화와 같이 쓰기도 해요 (<code>optimum</code> 라이브러리)."),
            ("하이퍼파라미터 찾기", "정제의 온도 T, 섞는 비율 alpha, 학생의 층 수는 데이터마다 맞는 값이 달라요. Optuna 같은 도구로 여러 조합을 자동으로 시험해요."),
            ("GPU에서의 LLM", "큰 LLM은 4비트, 8비트 양자화(GPTQ, AWQ, bitsandbytes)로 GPU 메모리를 줄여 돌려요. 같은 원리지만 층마다, 묶음마다 눈금을 정교하게 잡아요."),
            ("들어오는 글의 길이", "한 묶음을 가장 긴 문장에 맞춰 채우는 대신 비슷한 길이끼리 묶으면(동적 padding) 같은 모델로도 빨라져요."),
            ("정확도가 얼마나 떨어져도 되나", "쓰는 곳이 정해요. 실시간, 기기 안, 비용이 중요하면 가볍게, 틀리면 큰일인 곳이면 정확도를 지켜요. 정확도뿐 아니라 틀리는 모양(⑥ 오류 분석)도 바뀌지 않았는지 봐요 (<a href=\"text-classification.html\">문장 분류</a>).")]
libs = [("①", "<code>torch.save</code>, <code>io.BytesIO</code>, <code>time.perf_counter</code>, <code>torch.set_num_threads</code>", "잣대"), ("③", "<code>torch.nn.Embedding(_weight=...)</code>", "작은 어휘 표"),
        ("④", "<code>F.kl_div</code>, <code>F.log_softmax</code>", "지식 정제"), ("⑤", "<code>torch.ao.quantization.quantize_dynamic</code>, <code>per_channel_dynamic_qconfig</code>", "동적 양자화"),
        ("⑥", "<code>torch.nn.utils.prune.global_unstructured</code>, <code>L1Unstructured</code>, <code>prune.remove</code>", "가지치기")]
K.build(ROOT / "topics" / "model" / "model-compression.html", title="모델 가볍게", h1="모델 가볍게: 어휘 줄이기, 지식 정제, 양자화, 가지치기", track="모델", crumb="모델 &gt; 언어 모델로 문제 풀기",
        one_line="모델을 가볍게 만드는 방법은 줄이는 곳이 저마다 달라요. 안 쓰는 <b>어휘</b>를 덜면 정확도 그대로 크기가 줄고, 큰 모델의 답을 작은 <b>학생</b>에게 가르치면 빨라지고, 숫자를 <b>8비트</b>로 바꾸면 크기가 4분의 1이 돼요. 작은 가중치를 <b>지우기</b>만 해서는 크기도 시간도 안 줄어요.",
        why=why, whole=whole, steps=steps, code_file="examples/compress/compress_classifier.py", practice=practice, libs=libs,
        libs_note="<code>encode</code>, <code>train</code>, <code>benchmark</code>, <code>first_layers</code>, <code>int8</code>, <code>pruned</code>은 이 파일 안에서 정의한 함수, <code>Classifier</code>, <code>SmallVocab</code>은 클래스예요.",
        next_rows=[("<a href=\"text-classification.html\">문장 분류와 미세조정</a>", "가볍게 만든 그 분류기"), ("<a href=\"ocr-documents.html\">OCR과 엣지 배포</a>", "기기 안에서 돌리기"),
                   ("<a href=\"gpt-generation.html\">작은 GPT와 디코딩</a>", "처음부터 작게 만들기")],
        sources=["함께 정리한 자료: Lewis Tunstall 외, <i>Natural Language Processing with Transformers</i> 8장(효율적인 Transformer: 벤치마크 만들기, 지식 정제와 온도, DistilBERT, 학생 초기화, 양자화와 부동 소수점, 동적 양자화, ONNX와 ONNX Runtime, magnitude pruning과 movement pruning, Optuna). 부트캠프 수업에서 따로 다루지 않은 주제라 예제는 새로 만들었어요. 어휘 줄이기(③)는 이 예제의 무게 분포를 보고 더한 방법이에요",
                 "데이터: <code>examples/classify/data/issues.csv</code> (문장 분류 쪽과 같은 가상의 문의)",
                 "결과 값과 그림: <code>tools/pages/compress.py</code>가 <code>examples/compress/compress_classifier.py</code>를 단계별로 실제 실행한 결과 (CPU, 스레드 4개). 시간은 PC마다, 재는 때마다 달라요",
                 "공식 문서: <a href=\"https://pytorch.org/docs/stable/quantization.html\" target=\"_blank\">PyTorch Quantization</a>, <a href=\"https://pytorch.org/tutorials/intermediate/pruning_tutorial.html\" target=\"_blank\">Pruning tutorial</a>, <a href=\"https://arxiv.org/abs/1910.01108\" target=\"_blank\">DistilBERT 논문</a>"])
print("ok")
