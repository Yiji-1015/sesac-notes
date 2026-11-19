"""모델 > 딥러닝 기초 두 쪽. examples/dl/points_net.py 하나가 ①~⑪로 두 쪽에 걸쳐 자란다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/dl_basics.py
"""
import sys
import warnings
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "dl"
code = S.parse((EX / "points_net.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="points_net.py")  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CPU 학습 (GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음'})"
SRC_DATA = "데이터: <code>examples/dl/data/points.csv</code> (점 400개, 두 무리. 기존 노트와 같은 예제)"
ns = run(11)
df = ns["df"]


def hist_rows(h, every):
    return [[x["epoch"], f3(x["train_loss"]), f3(x["valid_loss"]), f"{x['valid_acc']:.3f}"] for x in h if x["epoch"] == 1 or x["epoch"] % every == 0]


# =============================== 1쪽 ===============================
outs = {}
outs["data"] = S.table("points.csv 앞 5줄", ["id", "x1", "x2", "label", "split"], [[E(str(v)) for v in r] for r in df.head(5).values],
                       ["점마다 좌표 두 개(<code>x1</code>, <code>x2</code>)와 무리 번호(<code>label</code> 0 또는 1)가 있어요. train 240개로 배우고 valid 80개로 재요."])
n1 = run(1)
outs["s1"] = S.box("① 실행 결과", [(f"torch.cuda.is_available() → {n1['gpu']}", 1), (f"X_train.shape → {tuple(n1['X_train'].shape)}, dtype {n1['X_train'].dtype}", 2), f"y_train.shape → {tuple(n1['y_train'].shape)}"],
                   ["이 PC에는 GPU가 있어요. 그래도 이 예제는 작아서 CPU로 학습해요. GPU는 숫자 계산을 아주 많이 동시에 할 때 빨라요.", "점 240개, 좌표 2개짜리 표가 Tensor가 됐어요. 신경망은 보통 32비트 실수로 계산해요."])
outs["s2"] = S.box("② 선 하나(nn.Linear)로 나누기, 500번 학습", [(f"valid 정확도 → {ns['linear_acc']:.3f}", 1)],
                   ["두 무리가 직선 하나로는 깔끔하게 나뉘지 않는 모양이라, 아무리 학습해도 80% 근처에서 멈춰요. 곡선 경계가 필요해요."])
outs["s3"] = S.box("③ make_net()", [str(ns["make_net"]()).replace("\n", " ")[:200], (f"가중치 수 → {ns['n_params']}개", 1)],
                   ["2 → 16 → 1. 가운데 층 16칸과 ReLU(음수는 0으로) 덕분에 경계를 구부릴 수 있어요. 가중치 65개 = (2×16 + 16) + (16×1 + 1)"])
g = ns["grad_first"]
outs["s4"] = S.box("④ 손실을 계산하고 backward() 한 번", [f"loss_before → {ns['loss_before']:.4f} (학습 전)", (f"net[0].weight.grad.shape → {tuple(g.shape)}", 1), f"grad 앞 2줄 → {[[round(float(v), 4) for v in row] for row in g[:2]]}"],
                   ["첫 층 가중치(16×2)와 같은 모양의 기울기가 생겼어요. 각 숫자는 \"이 가중치를 조금 키우면 손실이 얼마나 변하나\"예요. 이 기울기의 반대 방향으로 가중치를 조금씩 옮기는 게 학습이에요."])
outs["s5"] = S.table("⑤ Adam, 학습률 0.01, 100 epoch", ["epoch", "train 손실", "valid 손실", "valid 정확도"], hist_rows(ns["hist"], 20),
                     [f"손실이 줄면서 valid 정확도가 {ns['hist'][-1]['valid_acc']:.3f}까지 올라갔어요. 선 하나(②, {ns['linear_acc']:.3f})로는 못 넘던 벽을 넘었어요."])
flow1 = [[1, "Tensor", "표 → Tensor", "GPU 확인"], [2, "선 하나", "nn.Linear", "곡선을 못 그림"], [3, "신경망", "Linear, ReLU", "가중치 65개"], [4, "역전파", "backward()", "기울기"], [5, "학습 루프", "batch, epoch", "valid 0.975"]]
S.save(ROOT / "topics" / "model" / "dl-tensor-training-data.js", {"code": [c for c in code if c["s"] <= 5], "names": S.names(code), "range": [1, 5], "outs": outs, "flow": flow1, "env": ENV})
why1 = f"""<p>점들을 두 무리로 나누는 문제예요.</p>{K.out("data")}
<p>머신러닝 모델(<a href="ml-regression-classification.html">회귀와 분류</a>)은 사람이 준 입력 열을 그대로 써요. 경계가 복잡하면 사람이 \"x1의 제곱\", \"두 값의 곱\" 같은 새 열을 만들어 줘야 해요. <b>신경망</b>은 층을 쌓아 이런 표현을 <b>스스로 배워요</b>. 그래서 사진, 소리, 글처럼 사람이 열을 만들기 어려운 데이터에 강해요.</p>
<p>여기서는 파이썬 딥러닝 라이브러리 <b>PyTorch</b>로, 데이터를 숫자 배열(<b>Tensor</b>)로 바꾸고, 신경망을 만들고, 학습시키는 과정을 처음부터 따라가요.</p>"""
whole1 = """<p>데이터를 Tensor로 → 선 하나로는 안 되는 걸 확인 → 층을 쌓은 신경망 → 기울기 계산 → 반복 학습. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p><code>points_net.py</code> 하나가 이 쪽(①~⑤)에서 시작해 <a href="dl-training-stability.html">2쪽</a>(⑥~⑪)까지 자라요.</p>"""
steps1 = [
    K.step(1, "GPU 확인하고 Tensor로 바꾸기", "표를 PyTorch의 숫자 배열인 Tensor로 바꿔요.",
           "<p><b>Tensor</b>는 여러 차원의 숫자 배열이에요. numpy 배열과 비슷하지만, GPU에서 계산할 수 있고 기울기를 자동으로 계산할 수 있어요.</p>"
           + K.code_table([("torch.cuda.is_available()", "GPU를 쓸 수 있나"), ("torch.tensor(values, dtype=torch.float32)", "배열을 32비트 실수 Tensor로"),
                           (".unsqueeze(1)", "(240,) 모양을 (240, 1)로. 모델 출력 모양과 맞추려고"), (".to(device)", "Tensor를 CPU나 GPU로 옮기기. 모델과 데이터가 같은 곳에 있어야 계산돼요.")]), "s1"),
    K.step(2, "선 하나로 나눠 보기", "가장 단순한 모델 <code>nn.Linear(2, 1)</code>로 먼저 해 봐요. 기준선이에요.",
           K.code_table([("nn.Linear(2, 1)", "w1×x1 + w2×x2 + b. 경계는 언제나 직선"), ("binary_cross_entropy_with_logits", "두 종류 분류의 손실. 틀릴수록, 틀린 걸 확신할수록 커요."),
                         ("opt.zero_grad(); loss.backward(); opt.step()", "기울기 지우기 → 계산 → 가중치 옮기기. 학습 한 번"), ("with torch.no_grad()", "채점할 때는 기울기를 계산하지 않아 빨라요.")]), "s2"),
    K.step(3, "신경망 만들기", "Linear와 ReLU를 쌓아 경계를 구부릴 수 있는 모델을 만들어요.",
           K.code_table([("nn.Sequential(...)", "층을 순서대로 이어 붙인 모델"), ("nn.ReLU()", "음수는 0, 양수는 그대로. 이게 없으면 Linear를 몇 개 쌓아도 결국 직선이에요."),
                         ("nn.Dropout(dropout)", "2쪽 ⑨에서 써요. 0이면 아무것도 안 해요."), ("p.numel()", "가중치 Tensor 하나의 숫자 개수")]), "s3"),
    K.step(4, "손실과 역전파", "예측이 얼마나 틀렸는지(손실)를 계산하고, <code>backward()</code>로 모든 가중치의 기울기를 구해요.",
           "<p>신경망은 층이 여러 개라 \"앞쪽 가중치가 손실에 얼마나 영향을 줬나\"를 계산하기 어려워요. <b>역전파</b>는 출력부터 거꾸로 한 층씩 계산해 내려가는 방법이고, PyTorch가 자동으로 해 줘요.</p>"
           + K.code_table([("nn.BCEWithLogitsLoss()", "②와 같은 손실을 층(객체)으로"), ("loss.backward()", "손실에서 모든 가중치까지 기울기를 계산해 <code>.grad</code>에 넣어요."), ("net.zero_grad()", "기울기는 쌓이기 때문에 다음 계산 전에 지워요.")]), "s4"),
    K.step(5, "학습 루프", "데이터를 32개씩 나눠(batch) 가중치를 고치고, 이걸 전체 데이터로 100바퀴(epoch) 반복해요.",
           K.code_table([("DataLoader(..., batch_size=32, shuffle=True)", "32개씩 꺼내 주고, 바퀴마다 순서를 섞어요."), ("torch.optim.Adam(params, lr=0.01)", "기울기로 가중치를 옮기는 방법. 학습률(lr)은 한 번에 옮기는 크기"),
                         ("model.train() / model.eval()", "학습 중과 채점 중을 구분해요. Dropout 같은 층이 다르게 동작해요."), ("history.append({...})", "바퀴마다 손실과 정확도를 기록. 2쪽에서 이 기록으로 문제를 진단해요.")]), "s5"),
]
practice1 = [("GPU로 학습한다", "모델과 데이터를 모두 <code>.to(\"cuda\")</code>로 옮겨요. 하나라도 CPU에 있으면 \"device가 다르다\" 오류가 나요. 결과는 GPU에서 매번 조금씩 다를 수 있어요."),
             ("Tensor 모양 오류", "대부분 <code>(N,)</code>과 <code>(N, 1)</code>처럼 모양이 어긋나서 생겨요. 의심되면 <code>.shape</code>를 찍어 봐요."),
             ("손실이 줄지 않는다", "학습률이 너무 작거나 크거나, <code>zero_grad()</code>를 빠뜨렸을 수 있어요. 2쪽 ⑥에서 학습률의 영향을 봐요.")]
libs1 = [("①", "<code>torch.tensor()</code>, <code>.to()</code>, <code>.unsqueeze()</code>, <code>torch.cuda.is_available()</code>", "Tensor (torch)"), ("②③", "<code>nn.Linear</code>, <code>nn.ReLU</code>, <code>nn.Sequential</code>", "층과 모델 (torch.nn)"),
         ("④", "<code>nn.BCEWithLogitsLoss</code>, <code>loss.backward()</code>, <code>.grad</code>", "손실과 역전파"), ("⑤", "<code>DataLoader</code>, <code>TensorDataset</code>, <code>torch.optim.Adam</code>, <code>torch.no_grad()</code>", "학습 루프")]
K.build(ROOT / "topics" / "model" / "dl-tensor-training.html", title="Tensor와 신경망 학습", h1="Tensor와 신경망 학습: 층을 쌓아 경계를 배우기", track="모델",
        crumb="모델 &gt; 딥러닝 기초 1/2쪽 &nbsp;|&nbsp; 다음: <a href=\"dl-training-stability.html\" style=\"color:inherit\">학습을 안정시키기</a>",
        one_line="<b>신경망</b>은 Linear 층과 ReLU를 쌓아, 사람이 열을 만들어 주지 않아도 <b>복잡한 경계를 스스로 배워요</b>. 학습은 \"손실 계산 → 역전파로 기울기 → 가중치 조금 옮기기\"의 반복이에요.",
        why=why1, whole=whole1, steps=steps1, code_file="examples/dl/points_net.py", practice=practice1, libs=libs1, libs_note="<code>tensors</code>, <code>accuracy</code>, <code>make_net</code>, <code>train</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"dl-training-stability.html\">학습을 안정시키기</a>", "학습률, Optimizer, 과적합"), ("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "신경망 이전의 모델과 평가")],
        sources=["다루는 범위: Day09 (2026-10-29) GPU, Tensor, ANN, 역전파", SRC_DATA, "결과 값: <code>tools/pages/dl_basics.py</code>가 단계별로 실제 실행한 결과 (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/tutorials/beginner/basics/intro.html\" target=\"_blank\">PyTorch 기본 익히기</a>, <a href=\"https://docs.pytorch.org/docs/stable/nn.html\" target=\"_blank\">torch.nn</a>"])

# =============================== 2쪽 ===============================
outs = {}
outs["why"] = S.table("⑧에서 만든 큰 신경망 (학습 데이터 40개로 300 epoch)", ["epoch", "train 손실", "valid 손실", "valid 정확도"], hist_rows(ns["over"], 50),
                      ["train 손실은 0이 됐는데 valid 손실은 오히려 계속 커져요. 학습 데이터를 <b>외워 버린</b> 거예요 (과적합). 외운 모델은 새 데이터에서 틀린 답을 아주 확신해요.",
                       "학습이 잘되는지는 손실 하나로 알 수 없어요. 학습률, Optimizer, 모델 크기, 데이터 양이 모두 영향을 줘요. 이 쪽에서는 그걸 하나씩 실험해요."])
outs["s6"] = S.table("⑥ 같은 신경망, SGD, 학습률만 다르게 (100 epoch)", ["학습률", "train 손실", "valid 정확도"], [[lr, f3(h[-1]["train_loss"]), f"{h[-1]['valid_acc']:.3f}"] for lr, h in ns["lr_runs"].items()],
                     ["0.001은 너무 조금씩 움직여서 100바퀴가 지나도 거의 그대로예요.", "10은 너무 크게 움직여서 최저점을 지나쳐 버려요. 0.1이 이 셋 중 가장 나아요. 적당한 학습률은 문제마다 달라서 몇 개를 시험해 봐요."])
outs["s7"] = S.table("⑦ 학습률 0.01로 SGD와 Adam (60 epoch)", ["Optimizer", "10 epoch valid 정확도", "60 epoch valid 정확도", "60 epoch train 손실"],
                     [[n, f"{h[9]['valid_acc']:.3f}", f"{h[-1]['valid_acc']:.3f}", f3(h[-1]["train_loss"])] for n, h in ns["opt_runs"].items()],
                     ["같은 학습률이어도 Adam이 훨씬 빨리 내려가요. Adam은 가중치마다 지금까지의 기울기를 보고 옮기는 크기를 알아서 조절해요. 그래서 처음 시작할 때 무난한 선택이에요."])
outs["s8"] = S.box("⑧ 과적합: 데이터 40개, 128칸 3층", [f"가중치 수 → {sum(p.numel() for p in ns['big'].parameters()):,}개", (f"300 epoch 뒤 train 손실 {ns['over'][-1]['train_loss']:.6f}, valid 손실 {ns['over'][-1]['valid_loss']:.3f}", 1, "bad")],
                   ["데이터 40개에 가중치 3만 개가 넘어요. 외우기에 충분하고도 남아요."])
outs["s9"] = S.table("⑨ 같은 조건에 Dropout 0.3 + weight decay", ["epoch", "train 손실", "valid 손실", "valid 정확도"], hist_rows(ns["regd"], 50),
                     [f"300 epoch 뒤 valid 손실이 {ns['over'][-1]['valid_loss']:.3f} → {ns['regd'][-1]['valid_loss']:.3f}로 줄었어요. 틀린 답에 대한 지나친 확신이 줄었다는 뜻이에요.",
                      "정확도 차이는 크지 않아요. 규제는 \"맞히는 개수\"보다 \"얼마나 믿을 만한 확률을 내나\"에 먼저 효과가 보이기도 해요."])
be = min(ns["early"], key=lambda x: x["valid_loss"])
outs["s10"] = S.box("⑩ 조기 종료 (patience 20)", [(f"멈춘 epoch → {len(ns['early'])}", 1), (f"가장 좋았던 epoch → {be['epoch']} (valid 손실 {be['valid_loss']:.3f}, 정확도 {be['valid_acc']:.3f})", 2)],
                    ["valid 손실이 20번 연속 나아지지 않아서 멈췄어요. 300번을 다 돌지 않아 시간도 아껴요.", "마지막이 아니라 가장 좋았던 때의 가중치로 되돌렸어요. 데이터가 40개뿐이라 아주 이른 때가 가장 좋았어요."])
outs["s11"] = S.table("⑪ test 80개 정확도 (처음 사용)", ["모델", "test 정확도"], [[E(k), f"{v:.3f}"] for k, v in ns["test_acc"].items()],
                      ["데이터 240개를 다 쓴 기본 신경망이 가장 좋아요. 과적합을 막는 기법도 중요하지만, <b>데이터가 많은 것</b>을 이기기는 어려워요.", "⑧~⑩은 일부러 40개만 써서 과적합을 만든 실험이에요. 규제와 조기 종료는 손실(확신의 정도)을 안정시켰지만, 이 작은 test에서 정확도까지 올리진 못했어요."])
flow2 = [[6, "학습률", "0.001, 0.1, 10", "너무 작거나 크면"], [7, "Optimizer", "SGD vs Adam", "Adam이 빠름"], [8, "과적합", "40개로 외우기", "train 0, valid ↑"], [9, "규제", "Dropout, decay", "확신 줄이기"], [10, "조기 종료", "가장 좋던 때로", "patience"], [11, "test", "한 번", "데이터가 이김"]]
S.save(ROOT / "topics" / "model" / "dl-training-stability-data.js", {"code": code, "names": S.names(code), "range": [6, 11], "outs": outs, "flow": flow2, "env": ENV})
why2 = f"""<p><a href="dl-tensor-training.html">1쪽</a>에서는 학습이 순조로웠어요. 하지만 조건이 조금만 달라도 학습은 쉽게 어긋나요.</p>{K.out("why")}"""
whole2 = """<p>학습률과 Optimizer로 \"잘 내려가게\" 하고, 과적합을 일부러 만든 뒤 규제와 조기 종료로 다스리고, 마지막에 test로 한 번 채점해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>1쪽의 코드(①~⑤)에 이어서 같은 파일이 자라요.</p>"""
steps2 = [
    K.step(6, "학습률 비교", "같은 모델을 학습률만 바꿔 학습해요.", K.code_table([("run(lr, opt_name=\"sgd\")", "seed를 같게 해서 학습률만 다른 조건을 만들어요."), ("torch.optim.SGD", "기울기 × 학습률만큼 그대로 옮기는 가장 기본 방법")]), "s6"),
    K.step(7, "Optimizer 비교", "같은 학습률로 SGD와 Adam을 비교해요.", K.code_table([("torch.optim.Adam", "기울기의 평균과 크기를 기억해서 가중치마다 보폭을 조절")]), "s7"),
    K.step(8, "과적합 만들기", "데이터는 40개로 줄이고, 모델은 128칸 3층으로 크게 해서 300바퀴 돌려요.",
           K.code_table([("X_train[:40]", "학습 데이터를 앞 40개만"), ("make_net(hidden=128, depth=3)", "큰 모델")]), "s8"),
    K.step(9, "규제: Dropout과 weight decay", "학습 중에 일부 칸을 무작위로 끄고(Dropout), 가중치가 커지지 않게 벌점을 줘요(weight decay).",
           "<p>둘 다 모델이 몇몇 가중치나 칸에 기대어 답을 외우지 못하게 해요.</p>" + K.code_table([("nn.Dropout(0.3)", "학습 때 칸의 30%를 무작위로 0으로. <code>eval()</code>에서는 꺼져요."), ("Adam(..., weight_decay=1e-3)", "가중치를 조금씩 0 쪽으로 당겨요.")]), "s9"),
    K.step(10, "조기 종료", "바퀴마다 valid 손실을 보고, 더 나아지지 않으면 멈춘 뒤 가장 좋았던 가중치로 되돌려요.",
           K.code_table([("copy.deepcopy(model.state_dict())", "그때의 가중치를 통째로 복사해 둬요."), ("patience", "몇 번 연속 나아지지 않으면 멈출지"), ("model.load_state_dict(best_state)", "가장 좋았던 가중치로 되돌리기")]), "s10"),
    K.step(11, "마지막 한 번: test", "아껴 둔 test 80개로 모델들을 한 번 채점해요.", K.code_table([("tensors(\"test\")", "1쪽 ①의 함수로 test도 Tensor로")]), "s11"),
]
practice2 = [("학습률을 정할 때", "0.1, 0.01, 0.001처럼 10배씩 바꿔 몇 바퀴만 돌려 보고, 손실이 가장 잘 내려가는 쪽을 골라요. 학습 중에 점점 줄이는 방법(스케줄러)도 있어요."),
             ("BatchNorm", "층마다 값의 평균과 크기를 맞춰 학습을 안정시키는 층이에요 (<code>nn.BatchNorm1d</code>). Dropout처럼 <code>train()</code>과 <code>eval()</code>에서 다르게 동작해요."),
             ("과적합이 의심될 때", "train 손실과 valid 손실을 같이 그려 봐요. 둘이 벌어지기 시작하면 과적합이에요. 먼저 데이터를 늘릴 수 있는지 봐요.")]
libs2 = [("⑥⑦", "<code>torch.optim.SGD</code>, <code>torch.optim.Adam</code>", "Optimizer"), ("⑨", "<code>nn.Dropout</code>, <code>weight_decay=</code>", "규제"), ("⑩", "<code>state_dict()</code>, <code>load_state_dict()</code>, <code>copy.deepcopy</code>", "가중치 저장과 복원")]
K.build(ROOT / "topics" / "model" / "dl-training-stability.html", title="학습을 안정시키기", h1="학습을 안정시키기: 학습률, Optimizer, 과적합", track="모델",
        crumb="모델 &gt; 딥러닝 기초 2/2쪽 &nbsp;|&nbsp; 이전: <a href=\"dl-tensor-training.html\" style=\"color:inherit\">Tensor와 신경망 학습</a>",
        one_line="학습이 잘 내려가려면 <b>학습률과 Optimizer</b>가 맞아야 하고, 학습 데이터를 외우지 않으려면 <b>규제와 조기 종료</b>가 필요해요. 문제는 train과 valid 손실을 <b>같이</b> 봐야 보여요.",
        why=why2, whole=whole2, steps=steps2, code_file="examples/dl/points_net.py", practice=practice2, libs=libs2, libs_note="<code>run</code>, <code>train_early</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"dl-tensor-training.html\">Tensor와 신경망 학습</a>", "학습 루프의 기본"), ("<a href=\"cnn.html\">CNN과 이미지 분류</a>", "그림을 다루는 신경망")],
        sources=["다루는 범위: Day10 (2026-10-30) Optimizer, 학습률, Batch, Epoch, 과적합, Dropout, Early Stopping", SRC_DATA, "결과 값: <code>tools/pages/dl_basics.py</code> (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/docs/stable/optim.html\" target=\"_blank\">torch.optim</a>, <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html\" target=\"_blank\">nn.Dropout</a>"])
print("ok")
