"""모델 > 이미지 > CNN과 이미지 분류. 그림 출력은 topics/model/img/cnn/ 에 저장한다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/cnn_lines.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402
import torchvision  # noqa: E402
from PIL import Image  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "cnn"
IMG = ROOT / "topics" / "model" / "img" / "cnn"
IMG.mkdir(parents=True, exist_ok=True)
code = S.parse((EX / "lines_cnn.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="lines_cnn.py")
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="lines_cnn.py")  # noqa: E731
import torch.nn.functional as F  # noqa: E402
import numpy as np  # noqa: E402
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
sys.path.insert(0, str(ROOT / "tools" / "visuals"))
from vstyle import plt, C, setup  # noqa: E402
setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/cnn/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


# 그림 1: 학습된 CNN 첫 층 무늬 8개와, 가로줄 그림 한 장에 대한 반응
cnn_m = ns["cnn"]
w1 = cnn_m[0].weight.detach()[:, 0]
with torch.no_grad():
    fmap = F.relu(cnn_m[0](ns["sample"]))  # (2장, 8무늬, 64, 64)
fig, axs = plt.subplots(3, 8, figsize=(13, 5.4))
for j in range(8):
    axs[0, j].imshow(w1[j].numpy(), cmap="PuOr_r", vmin=-w1.abs().max(), vmax=w1.abs().max())
    axs[0, j].set_title(f"무늬 {j}", fontsize=10)
    axs[1, j].imshow(fmap[0, j].numpy(), cmap="magma")
    axs[2, j].imshow(fmap[1, j].numpy(), cmap="magma")
for ax in axs.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
axs[0, 0].set_ylabel("배운 3×3 무늬", fontsize=10)
axs[1, 0].set_ylabel("가로줄 그림에", fontsize=10)
axs[2, 0].set_ylabel("세로줄 그림에", fontsize=10)
P_K = fig_save(fig, "learned_kernels")
# 그림 2: 16층 블록에서 층마다 기울기 크기 (학습 전)
grads = {}
for nm, (res, bn) in {"그냥 쌓기": (False, False), "잔차 연결": (True, False), "잔차 + BatchNorm": (True, True)}.items():
    torch.manual_seed(0)
    m = ns["make_deep"](ns["DEPTH"], res, bn)
    F.cross_entropy(m(ns["X_train"]), ns["y_train"]).backward()
    grads[nm] = [m[0].weight.grad.norm().item()] + [blk.conv.weight.grad.norm().item() for blk in list(m)[3:3 + ns["DEPTH"]]]
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for (nm, g), col in zip(grads.items(), [C["pink"], C["mint"], C["purple"]]):
    ax.plot(range(len(g)), g, marker="o", ms=3, lw=2.2, color=col, label=nm)
ax.set_yscale("log")
from matplotlib.ticker import FuncFormatter  # noqa: E402
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"1e{int(round(np.log10(v)))}"))
ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set(xlabel="층 (0 = 맨 앞 합성곱, 16 = 맨 뒤 블록)", ylabel="그 층 가중치의 기울기 크기", title="학습 전 한 번 backward 했을 때, 앞쪽까지 기울기가 얼마나 남나")
ax.legend(fontsize=9)
P_G = fig_save(fig, "residual_grads")


def save(t, name):
    a = t.detach().float()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8) * 255
    Image.fromarray(a.squeeze().byte().numpy()).resize((128, 128), Image.NEAREST).save(IMG / name)
    return f"img/cnn/{name}"


outs = {}
Xt, yt = ns["X_train"], ns["y_train"]
outs["data"] = S.images("train 그림 예시 (64×64)", [(save(Xt[yt == 0][i], f"h{i}.png"), "horizontal") for i in range(3)] + [(save(Xt[yt == 1][i], f"v{i}.png"), "vertical") for i in range(3)],
                        ["가로줄 그림과 세로줄 그림이에요. 줄의 위치와 두께가 그림마다 달라요. train 60장, valid 20장, test 20장이에요."])
outs["why"] = S.table("두 모델 비교 (valid 20장)", ["모델", "가중치 수", "정확도", "선을 8칸 옮기면"],
                      [["픽셀을 펼친 MLP (②)", f"{ns['count'](ns['make_mlp']()):,}", f"{ns['scores']['MLP']:.2f}", f"<span class=\"bad\">{ns['shift_scores'][8]['MLP']:.2f}</span>"],
                       ["CNN (④)", f"{ns['count'](ns['make_cnn']()):,}", f"<b>{ns['scores']['CNN']:.2f}</b>", f"<b>{ns['shift_scores'][8]['CNN']:.2f}</b>"]],
                      ["CNN은 가중치가 <b>100분의 1</b>인데도 더 잘 맞히고, 선의 위치가 바뀌어도 그대로 맞혀요. MLP는 \"몇 번째 픽셀이 밝은가\"를 외워서, 위치가 바뀌면 무너져요."])
outs["s1"] = S.box("① load(\"train\")", [(f"X_train.shape → {tuple(Xt.shape)}", 1), f"classes → {ns['classes']}", f"픽셀 값 범위 → {Xt.min().item():.1f} ~ {Xt.max().item():.1f}"],
                   ["(그림 수, 채널, 세로, 가로) 순서예요. 흑백이라 채널이 1이에요. 컬러면 3(R, G, B)이에요."])
outs["s2"] = S.box("② make_mlp()", [str(ns["make_mlp"]()).replace("\n", " ")[:170], (f"가중치 수 → {ns['count'](ns['make_mlp']()):,}", 1)],
                   ["64×64 = 4,096개 픽셀 하나하나에 따로 가중치가 붙어요. 그림의 \"모양\"이 아니라 \"자리\"를 배워요."])
mp = ns["maps"]
outs["s3"] = S.images("③ 무늬 두 개를 그림 두 장에 대 보면 (밝을수록 강한 반응)", [(save(ns["sample"][0], "s_h.png"), "가로줄 그림"), (save(mp[0, 0], "m_hh.png"), "가로 무늬 반응"), (save(mp[0, 1], "m_hv.png"), "세로 무늬 반응"),
                                                                       (save(ns["sample"][1], "s_v.png"), "세로줄 그림"), (save(mp[1, 0], "m_vh.png"), "가로 무늬 반응"), (save(mp[1, 1], "m_vv.png"), "세로 무늬 반응")],
                      [f"평균 반응: 가로줄 그림 → 가로 무늬 {ns['response'][0][0]:.3f}, 세로 무늬 {ns['response'][0][1]:.3f} / 세로줄 그림 → 가로 무늬 {ns['response'][1][0]:.3f}, 세로 무늬 {ns['response'][1][1]:.3f}",
                       "3×3 무늬 하나를 그림 전체에 밀며 대 보니, 맞는 방향의 선이 있는 자리만 밝게 켜졌어요. <b>어디에 있든</b> 같은 무늬로 찾아요. 이게 합성곱이에요. CNN은 이런 무늬를 사람이 정하지 않고 학습으로 찾아요."])
outs["s4"] = S.box("④ make_cnn()", [str(ns["make_cnn"]()).replace("\n", " ")[:220], (f"가중치 수 → {ns['count'](ns['make_cnn']()):,}", 1)],
                   ["3×3 무늬 8개, 그 위에 다시 무늬 16개만 배우면 돼요. 무늬 하나를 그림 모든 자리에 같이 쓰니까(가중치 공유) 가중치가 아주 적어요."])
outs["s5"] = S.box("⑤ 같은 조건(Adam, 60번)으로 학습, valid 정확도", [f"MLP → {ns['scores']['MLP']:.2f}", (f"CNN → {ns['scores']['CNN']:.2f}", 1)], ["두 모델 다 꽤 맞히지만, 차이는 ⑥에서 크게 드러나요."])
sh = ns["shifted"]
outs["s6"] = S.table("⑥ valid 그림을 k칸씩 옮긴 뒤 정확도", ["옮긴 칸 수", "MLP", "CNN"], [[k, f"{v['MLP']:.2f}", f"<b>{v['CNN']:.2f}</b>"] for k, v in ns["shift_scores"].items()],
                     ["CNN은 무늬를 어디서 찾든 같고, 마지막에 <code>AdaptiveAvgPool2d</code>로 위치를 평균 내버려서 옮겨도 그대로예요. MLP는 학습 때 본 자리가 아니면 엉뚱하게 답해요."])
outs["s6img"] = S.images("valid 첫 그림을 옮긴 모습", [(save(sh[k][0], f"shift{k}.png"), f"{k}칸") for k in sh])
s_ = ns["sample"][:1]
k_ = ns["kernels"][:1]
conv_rows = []
for label, kw in [("padding=0", dict(padding=0)), ("<b>padding=1</b> (③)", dict(padding=1)), ("padding=1, stride=2", dict(padding=1, stride=2)), ("padding=2, dilation=2", dict(padding=2, dilation=2))]:
    o_ = F.conv2d(s_, k_, **kw)
    kk = 1 + (3 - 1) * kw.get("dilation", 1)
    conv_rows.append([label, f"{tuple(o_.shape[2:])}", f"(64 − {kk} + 2×{kw.get('padding', 0)}) ÷ {kw.get('stride', 1)} + 1 = {(64 - kk + 2 * kw.get('padding', 0)) // kw.get('stride', 1) + 1}", f"{kk}×{kk}"])
outs["t3"] = S.table("③의 가로 무늬를 설정만 바꿔 대 보면 (입력 64×64)", ["설정", "출력 크기", "크기 계산", "무늬가 덮는 넓이"], conv_rows,
                     ["둘레에 0을 두르지 않으면(padding=0) 무늬가 그림 밖으로 못 나가서 가장자리 한 줄씩 잃어요(62×62).",
                      "stride=2는 두 칸씩 건너뛰며 대서 크기가 절반이 돼요. 풀링 없이 크기를 줄이는 방법이에요.",
                      "dilation=2는 무늬 칸 사이를 한 칸씩 띄워서, 가중치는 9개 그대로인데 5×5 넓이를 봐요. 넓은 맥락이 필요할 때 써요."])
outs["f_k"] = chart("④의 CNN이 학습으로 찾은 첫 층 무늬 8개와 그 반응", P_K, "위: 무늬(주황 +, 보라 −), 가운데와 아래: 무늬마다 반응 지도",
                    ["③에서는 가로, 세로 무늬를 사람이 정했어요. ④의 CNN은 무늬 8개를 정답만 보고 스스로 찾았어요.",
                     "무늬마다 반응하는 그림이 달라요. 어떤 무늬는 가로줄 그림의 선 자리에서만, 어떤 무늬는 세로줄 그림에서만 밝게 켜져요. 둘째 층은 이 8장을 입력으로 받아 다시 무늬를 찾아요."])
dr = ns["deep_runs"]
outs["s7"] = S.table(f"⑦ 3×3 합성곱 블록 {ns['DEPTH']}개를 쌓으면 (Adam, 60번)", ["쌓는 방식", "학습 전 맨 앞 층 기울기", "학습 뒤 train 손실", "valid 정확도"],
                     [[nm, f"{r['grad0']:.1e}", f3(r["train_loss"]), f"{r['valid_acc']:.2f}"] for nm, r in dr.items()],
                     [f"그냥 쌓으면 맨 앞 층까지 내려오는 기울기가 {dr['그냥 쌓기']['grad0']:.0e}로 사실상 0이에요. 앞쪽 층이 전혀 바뀌지 않아서 train 손실이 log 2 = 0.693에 그대로 머물러요. 어떤 그림이든 반반으로 찍는다는 뜻이에요. 층이 2개뿐인 ④는 문제없이 배웠는데, 깊게 쌓자마자 배우지 못해요.",
                      "잔차 연결 하나만 넣었는데 기울기가 살아나서 valid가 맞기 시작해요. 블록마다 입력을 그대로 더해 주는 길이 있어서, 기울기가 그 길로 곧장 앞쪽까지 내려와요.",
                      "BatchNorm까지 넣으면 train 손실이 0에 가까워져요. 층마다 값의 크기를 다시 맞춰 주니 더 깊이 쌓아도 안정돼요(🔁)."])
outs["f_g"] = chart("층마다 기울기 크기 (학습 전, 블록 16개)", P_G, "",
                    ["분홍(그냥 쌓기)은 뒤에서 앞으로 갈수록 층마다 일정 비율로 줄어서, 맨 앞에서는 10⁻¹¹ 근처까지 떨어져요. <a href=\"dl-tensor-training.html\">딥러닝 기초</a> ④의 기울기 소실이 실제 CNN에서 일어난 거예요.",
                     "민트(잔차)와 보라(잔차 + BatchNorm)는 앞쪽까지 크기가 거의 그대로예요."])
d30 = run(7, [("DEPTH = 16", "DEPTH = 30")])["deep_runs"]
outs["t7"] = S.table("블록을 30개로 늘리면", ["쌓는 방식", "학습 전 맨 앞 층 기울기", "학습 뒤 train 손실", "valid 정확도"],
                     [[nm, f"{r['grad0']:.1e}", f3(r["train_loss"]), f"{r['valid_acc']:.2f}"] for nm, r in d30.items()],
                     [f"그냥 쌓기는 기울기가 {d30['그냥 쌓기']['grad0']:.0e}로 더 작아졌어요.",
                      f"잔차 연결만 있으면 이번엔 반대로 기울기가 {d30['잔차 연결']['grad0']:.1f}로 커져서 학습이 흔들리고 손실이 거의 줄지 않아요. 블록마다 입력에 무언가를 더하니, 30번 더하면 값이 계속 불어나요.",
                      "BatchNorm이 층마다 크기를 되돌려 주면 30층도 문제없어요. 그래서 ResNet은 잔차 연결과 BatchNorm을 늘 같이 써요."])
for _k in ("f_k", "f_g"):
    outs[_k]["chart"] = True
flow = [[1, "Tensor", "(N, 1, 64, 64)", "ImageFolder"], [2, "MLP", "픽셀 펼치기", "13만 가중치"], [3, "합성곱", "3×3 무늬", "어디서든 찾기"], [4, "CNN", "Conv, Pool", "1,282 가중치"], [5, "학습", "같은 조건", "비교"], [6, "옮기기", "위치 바꾸기", "CNN만 버팀"], [7, "깊게", "잔차 연결", "BatchNorm"]]
S.save(ROOT / "topics" / "model" / "cnn-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                   "env": f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, torchvision {torchvision.__version__}, CPU (seed 0)"})
why = f"""<p>그림을 컴퓨터에 넣으면 숫자 격자예요. 64×64 흑백 그림이면 숫자 4,096개예요.</p>{K.out("data")}
<p>이 숫자들을 <a href="dl-tensor-training.html">신경망</a>에 그대로 펼쳐 넣을 수도 있어요. 하지만 그림에는 표 데이터와 다른 성질이 있어요. <b>가까운 픽셀끼리 의미가 있고</b>(선, 모서리), <b>같은 물체가 어디에 있든 같은 물체</b>예요. 두 방식을 실제로 비교하면:</p>{K.out("why")}
<p><b>CNN</b>(합성곱 신경망)은 작은 무늬를 그림 전체에 밀며 찾는 방식으로 이 두 성질을 이용해요.</p>"""
whole = """<p>그림을 Tensor로 → 펼친 MLP로 기준선 → 무늬 하나를 손으로 대 보기 → CNN → 같은 조건으로 학습 → 위치를 바꿔 시험. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "그림을 Tensor로", "폴더 이름을 정답으로 쓰는 ImageFolder로 읽고, 흑백 Tensor로 바꿔요.",
           K.code_table([("datasets.ImageFolder(DATA / split, transform=...)", "<code>train/horizontal/*.png</code>처럼 폴더별로 나뉜 그림을 읽고, 폴더 이름을 정답으로"), ("transforms.Grayscale()", "컬러를 흑백 한 장으로"),
                         ("transforms.ToTensor()", "0~255 픽셀을 0~1 실수 Tensor로, 모양은 (채널, 세로, 가로)"), ("torch.stack([...])", "그림들을 하나의 Tensor로 쌓기")]), "s1"),
    K.step(2, "픽셀을 그냥 펼치면 (MLP)", "4,096개 픽셀을 한 줄로 펼쳐 일반 신경망에 넣어요. 기준선이에요.", K.code_table([("nn.Flatten()", "(1, 64, 64)를 4,096개 한 줄로")]), "s2"),
    K.step(3, "합성곱 한 번: 작은 무늬 찾기", "가로 선과 세로 선에 반응하는 3×3 무늬를 직접 정해, 그림에 대 봐요.",
           "<p><b>합성곱</b>(convolution)은 작은 가중치 격자(<b>커널</b>, 여기서는 3×3)를 그림의 한 자리에 겹쳐 놓고, 겹친 9칸끼리 곱해 더한 값을 하나 내요. 이걸 한 칸씩 옮겨 가며 모든 자리에서 하면 \"이 무늬가 어디에 얼마나 있나\"를 담은 지도(<b>특성 지도</b>)가 나와요. 1쪽 신경망의 \"곱해 더하기\"를, 그림 전체가 아니라 가까운 9칸에만, 그리고 모든 자리에 <b>같은 가중치로</b> 하는 거예요.</p>"
           "<p>이 두 가지가 그림에 맞는 가정(귀납적 편향)이에요. ① <b>지역성</b>: 선이나 모서리는 가까운 픽셀끼리 만들어요. ② <b>가중치 공유</b>: 가로선은 왼쪽 위에 있든 오른쪽 아래에 있든 같은 가로선이에요. 그래서 그림을 옮기면 특성 지도도 똑같이 옮겨져요(<b>이동 등변성</b>). 마지막에 자리를 평균 내 버리면 옮겨도 답이 안 바뀌어요(이동 불변성, ⑥).</p>"
           "<p>출력 크기는 <code>(입력 − 커널 + 2 × padding) ÷ stride의 몫 + 1</code>이에요. <b>padding</b>은 둘레에 0을 둘러 가장자리도 가운데처럼 보게 하고, <b>stride</b>는 몇 칸씩 건너뛰며 댈지, <b>dilation</b>은 커널 칸 사이를 얼마나 띄울지예요.</p>"
           + K.code_table([("[[-1,-1,-1],[2,2,2],[-1,-1,-1]]", "가운데 줄이 밝고 위아래가 어두우면 큰 값. 가로 선 무늬"), ("horizontal_kernel.T", "뒤집으면 세로 선 무늬"),
                           ("F.conv2d(sample, kernels, padding=1)", "무늬를 그림 모든 자리에 대 보고 곱해서 더한 값을 지도로. 가장자리에 0을 둘러 크기를 유지"), ("F.relu(...)", "음수 반응은 0으로")])
           + K.try_("같은 무늬를 padding, stride, dilation만 바꿔 대 보면 출력 크기는?", "t3"), "s3"),
    K.step(4, "CNN 만들기", "합성곱과 풀링을 두 번 쌓고, 마지막에 위치를 평균 내 분류해요.",
           "<p><b>채널</b>: 첫 층은 흑백 1장을 받아 무늬 8개의 지도 8장을 내요. 둘째 층의 무늬 하나는 3×3 × <b>8장</b>짜리라, 앞 층 지도 8장을 한꺼번에 보고 \"가로 반응과 세로 반응이 어떻게 섞였나\"를 찾아요. 그래서 층이 깊어질수록 선 → 모서리 → 모양처럼 점점 큰 무늬를 찾게 돼요. 커널이 1×1이면 자리마다 채널끼리만 섞는 층이 되는데, 채널 수를 줄이거나 늘릴 때 써요.</p>"
           "<p><b>한 칸이 보는 넓이</b>(수용 영역): 3×3 합성곱 하나 뒤의 칸은 입력 3×3을, 2×2 풀링 뒤에는 4×4를, 그 뒤 3×3 합성곱 뒤에는 8×8을 봐요. 줄이기(풀링, stride)를 넣을수록 적은 층으로 넓게 봐요. 분류는 마지막에 \"있나\"만 알면 되니 크기를 계속 줄여요. 물체 위치를 칸마다 맞혀야 하는 분할은 줄인 지도를 다시 키우는 층(<code>nn.ConvTranspose2d</code>, 업샘플링)을 붙여요.</p>"
           + K.code_table([("nn.Conv2d(1, 8, 3, padding=1)", "3×3 무늬 8개를 학습. 결과는 무늬별 지도 8장"), ("nn.MaxPool2d(2)", "2×2마다 가장 큰 값만. 크기를 절반으로 줄이고 작은 위치 차이를 흡수"),
                           ("nn.AdaptiveAvgPool2d(1)", "지도마다 전체 평균 하나. \"어디에\"는 버리고 \"있나\"만 남겨요."), ("count(model)", "가중치 수")]), "s4"),
    K.step(5, "학습하고 비교하기", "두 모델을 같은 데이터, 같은 Optimizer, 같은 횟수로 학습해요.",
           K.code_table([("F.cross_entropy(model(X), y)", "여러 종류 분류의 손실. 모델 출력은 종류마다 점수 하나"), ("model(X).argmax(1)", "점수가 가장 큰 종류")]) + K.out("f_k"), "s5"),
    K.step(6, "선의 위치가 바뀌면", "valid 그림을 아래, 오른쪽으로 k칸씩 옮겨서 다시 맞혀 봐요.",
           K.code_table([("torch.roll(X, shifts=(k, k), dims=(2, 3))", "세로, 가로로 k칸씩 밀기 (밀려난 부분은 반대쪽으로)")]) + K.out("s6img"), "s6"),
    K.step(7, "깊게 쌓기: 잔차 연결과 BatchNorm", "같은 3×3 합성곱 블록을 16개 쌓아 보고, 입력을 그대로 더해 주는 길(잔차 연결)이 있을 때와 비교해요.",
           "<p>사진처럼 복잡한 그림은 층을 수십 개 쌓아야 해요. 그런데 <a href=\"dl-tensor-training.html\">딥러닝 기초</a> ④에서 본 것처럼, 층이 많으면 기울기가 앞쪽까지 내려오면서 층마다 곱해져 사라지기 쉬워요. 그래서 예전에는 층을 더 쌓으면 오히려 train 점수부터 나빠지는 일이 흔했어요.</p>"
           "<p><b>잔차 연결</b>(residual connection, skip connection)은 블록의 출력을 <code>F(x)</code> 대신 <code>x + F(x)</code>로 만들어요. 블록은 입력을 통째로 새로 만드는 게 아니라 \"입력에 얼마를 더 고칠지\"(잔차)만 배워요. 블록이 아무것도 안 배워도 입력이 그대로 지나가고, 기울기도 더하기 길을 따라 그대로 앞쪽까지 내려와요. 그래서 깊은 신경망을 \"얕은 길과 깊은 길을 여러 개 섞은 것\"으로 볼 수도 있어요.</p>"
           "<p>그런데 블록마다 무언가를 더하니 층이 아주 많으면 값이 점점 커져요. <b>BatchNorm</b>은 미니배치 안에서 채널마다 평균 0, 표준편차 1로 맞춘 뒤, 학습하는 두 값으로 다시 크기와 위치를 정해요. 층마다 크기가 되돌려져서 깊이 쌓아도 안정돼요. 이 둘을 묶은 구조가 <b>ResNet</b>이에요. 연결 방식을 바꾼 변형으로는 앞의 모든 층 출력을 이어 붙이는 DenseNet, 줄였다 키우면서 같은 크기 층끼리 이어 주는 U-Net(분할에 많이 써요)이 있어요.</p>"
           + K.out("f_g")
           + K.code_table([("class Block(nn.Module)", "층을 직접 정의할 때는 <code>__init__</code>에서 층을 만들고 <code>forward</code>에서 계산을 적어요."), ("F.relu(x + h)", "잔차 연결: 합성곱 결과 h에 입력 x를 더하기"),
                           ("nn.BatchNorm2d(ch)", "채널마다 평균과 크기 맞추기. <code>train()</code>에서는 그 배치의 통계를, <code>eval()</code>에서는 학습 중 모은 평균 통계를 써요."), ("nn.Identity()", "아무것도 하지 않는 층. BatchNorm을 끈 비교용"),
                           (".eval()", "채점 전에 BatchNorm을 채점용으로 바꾸기")])
           + K.try_("블록을 30개로 늘리면?", "t7"), "s7"),
]
practice = [("진짜 사진", "사진은 훨씬 복잡해서 층이 많은 CNN이 필요해요. 처음부터 배우기보다 큰 데이터로 미리 배운 모델을 빌려 써요 (<a href=\"transfer-learning.html\">전이학습과 오류 분석</a>)."),
            ("컬러 사진", "채널이 3이라 첫 층이 <code>nn.Conv2d(3, …)</code>예요. 정규화(평균, 표준편차 맞추기)도 같이 해요."),
            ("데이터가 적다", "뒤집기, 조금 옮기기, 밝기 바꾸기로 그림을 늘려요(증강). 단, 정답이 바뀌는 변형은 안 돼요. 가로줄 그림을 90도 돌리면 세로줄이 돼요."),
            ("깊은 CNN을 직접 만든다", "블록마다 잔차 연결과 BatchNorm을 넣어요(⑦). 직접 쌓기보다 <code>torchvision.models.resnet18</code> 같은 검증된 구조를 쓰는 게 보통이에요."),
            ("배치가 아주 작다 (BatchNorm)", "배치 몇 장의 평균은 흔들려서 BatchNorm이 불안정해져요. 그림 한 장 안에서 맞추는 GroupNorm이나 LayerNorm을 써요. Transformer는 LayerNorm을 써요."),
            ("그림이 아닌 격자 데이터", "소리(1차원 파형)는 <code>nn.Conv1d</code>, CT 같은 3차원 영상은 <code>nn.Conv3d</code>로 같은 생각을 써요."),
            ("물체 자리를 칸마다 맞혀야 한다 (분할)", "줄였다가 다시 키우고, 같은 크기 층끼리 이어 주는 U-Net 구조를 써요. 줄이는 쪽이 \"무엇\"을, 이어 주는 길이 \"정확한 자리\"를 전해 줘요.")]
libs = [("①", "<code>datasets.ImageFolder</code>, <code>transforms.Compose</code>, <code>Grayscale</code>, <code>ToTensor</code>", "그림 읽기 (torchvision)"), ("②④", "<code>nn.Flatten</code>, <code>nn.Conv2d</code>, <code>nn.MaxPool2d</code>, <code>nn.AdaptiveAvgPool2d</code>", "층 (torch.nn)"),
        ("③", "<code>F.conv2d</code>, <code>F.relu</code>", "합성곱 직접 계산"), ("⑤⑥", "<code>F.cross_entropy</code>, <code>torch.roll</code>", "학습, 옮기기"),
        ("⑦", "<code>nn.Module</code>, <code>nn.BatchNorm2d</code>, <code>nn.Identity</code>", "블록 직접 정의, 정규화")]
K.build(ROOT / "topics" / "model" / "cnn.html", title="CNN과 이미지 분류", h1="CNN과 이미지 분류: 작은 무늬를 어디서든 찾기", track="모델", crumb="모델 &gt; 이미지",
        one_line="<b>CNN</b>은 작은 무늬(합성곱)를 그림 전체에 밀며 찾아요. 그래서 <b>가중치가 훨씬 적고</b>, 물체의 <b>위치가 바뀌어도</b> 같은 물체로 알아봐요.",
        why=why, whole=whole, steps=steps, code_file="examples/cnn/lines_cnn.py", practice=practice, libs=libs, libs_note="<code>load</code>, <code>make_mlp</code>, <code>make_cnn</code>, <code>count</code>, <code>fit</code>, <code>acc</code>, <code>make_deep</code>는 이 파일 안에서 정의한 함수, <code>Block</code>은 클래스예요.",
        next_rows=[("<a href=\"transfer-learning.html\">전이학습과 오류 분석</a>", "미리 배운 CNN 빌려 쓰기"), ("<a href=\"opencv-preprocessing.html\">OpenCV 전처리</a>", "모델에 넣기 전 그림 다듬기")],
        sources=["다루는 범위: Day11 (2026-11-02) Image Tensor, CNN, Pooling, 이미지 분류",
                 "함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 10장(합성곱 신경망), 11장(잔차 신경망). 지역성과 가중치 공유, 이동 등변성, padding, stride, dilation, 채널과 1×1 합성곱, 수용 영역, 업샘플링, 잔차 연결, BatchNorm, ResNet, DenseNet, U-Net을 가져왔어요. ⑦의 실험과 그림은 이 예제로 직접 계산했어요",
                 "데이터: <code>examples/cnn/data/</code> (64×64 가로줄, 세로줄 그림. 기존 노트와 같은 예제)", "결과 값: <code>tools/pages/cnn_lines.py</code> (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html\" target=\"_blank\">nn.Conv2d</a>, <a href=\"https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html\" target=\"_blank\">ImageFolder</a>"])
print("ok")
