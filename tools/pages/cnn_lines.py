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
ns = S.run(code, 6, S.workdir(EX), filename="lines_cnn.py")


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
flow = [[1, "Tensor", "(N, 1, 64, 64)", "ImageFolder"], [2, "MLP", "픽셀 펼치기", "13만 가중치"], [3, "합성곱", "3×3 무늬", "어디서든 찾기"], [4, "CNN", "Conv, Pool", "1,282 가중치"], [5, "학습", "같은 조건", "비교"], [6, "옮기기", "위치 바꾸기", "CNN만 버팀"]]
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
           K.code_table([("[[-1,-1,-1],[2,2,2],[-1,-1,-1]]", "가운데 줄이 밝고 위아래가 어두우면 큰 값. 가로 선 무늬"), ("horizontal_kernel.T", "뒤집으면 세로 선 무늬"),
                         ("F.conv2d(sample, kernels, padding=1)", "무늬를 그림 모든 자리에 대 보고 곱해서 더한 값을 지도로. 가장자리에 0을 둘러 크기를 유지"), ("F.relu(...)", "음수 반응은 0으로")]), "s3"),
    K.step(4, "CNN 만들기", "합성곱과 풀링을 두 번 쌓고, 마지막에 위치를 평균 내 분류해요.",
           K.code_table([("nn.Conv2d(1, 8, 3, padding=1)", "3×3 무늬 8개를 학습. 결과는 무늬별 지도 8장"), ("nn.MaxPool2d(2)", "2×2마다 가장 큰 값만. 크기를 절반으로 줄이고 작은 위치 차이를 흡수"),
                         ("nn.AdaptiveAvgPool2d(1)", "지도마다 전체 평균 하나. \"어디에\"는 버리고 \"있나\"만 남겨요."), ("count(model)", "가중치 수")]), "s4"),
    K.step(5, "학습하고 비교하기", "두 모델을 같은 데이터, 같은 Optimizer, 같은 횟수로 학습해요.",
           K.code_table([("F.cross_entropy(model(X), y)", "여러 종류 분류의 손실. 모델 출력은 종류마다 점수 하나"), ("model(X).argmax(1)", "점수가 가장 큰 종류")]), "s5"),
    K.step(6, "선의 위치가 바뀌면", "valid 그림을 아래, 오른쪽으로 k칸씩 옮겨서 다시 맞혀 봐요.",
           K.code_table([("torch.roll(X, shifts=(k, k), dims=(2, 3))", "세로, 가로로 k칸씩 밀기 (밀려난 부분은 반대쪽으로)")]) + K.out("s6img"), "s6"),
]
practice = [("진짜 사진", "사진은 훨씬 복잡해서 층이 많은 CNN이 필요해요. 처음부터 배우기보다 큰 데이터로 미리 배운 모델을 빌려 써요 (<a href=\"transfer-learning.html\">전이학습과 오류 분석</a>)."),
            ("컬러 사진", "채널이 3이라 첫 층이 <code>nn.Conv2d(3, …)</code>예요. 정규화(평균, 표준편차 맞추기)도 같이 해요."),
            ("데이터가 적다", "뒤집기, 조금 옮기기, 밝기 바꾸기로 그림을 늘려요(증강). 단, 정답이 바뀌는 변형은 안 돼요. 가로줄 그림을 90도 돌리면 세로줄이 돼요.")]
libs = [("①", "<code>datasets.ImageFolder</code>, <code>transforms.Compose</code>, <code>Grayscale</code>, <code>ToTensor</code>", "그림 읽기 (torchvision)"), ("②④", "<code>nn.Flatten</code>, <code>nn.Conv2d</code>, <code>nn.MaxPool2d</code>, <code>nn.AdaptiveAvgPool2d</code>", "층 (torch.nn)"),
        ("③", "<code>F.conv2d</code>, <code>F.relu</code>", "합성곱 직접 계산"), ("⑤⑥", "<code>F.cross_entropy</code>, <code>torch.roll</code>", "학습, 옮기기")]
K.build(ROOT / "topics" / "model" / "cnn.html", title="CNN과 이미지 분류", h1="CNN과 이미지 분류: 작은 무늬를 어디서든 찾기", track="모델", crumb="모델 &gt; 이미지",
        one_line="<b>CNN</b>은 작은 무늬(합성곱)를 그림 전체에 밀며 찾아요. 그래서 <b>가중치가 훨씬 적고</b>, 물체의 <b>위치가 바뀌어도</b> 같은 물체로 알아봐요.",
        why=why, whole=whole, steps=steps, code_file="examples/cnn/lines_cnn.py", practice=practice, libs=libs, libs_note="<code>load</code>, <code>make_mlp</code>, <code>make_cnn</code>, <code>count</code>, <code>fit</code>, <code>acc</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"transfer-learning.html\">전이학습과 오류 분석</a>", "미리 배운 CNN 빌려 쓰기"), ("<a href=\"opencv-preprocessing.html\">OpenCV 전처리</a>", "모델에 넣기 전 그림 다듬기")],
        sources=["다루는 범위: Day11 (2026-11-02) Image Tensor, CNN, Pooling, 이미지 분류", "데이터: <code>examples/cnn/data/</code> (64×64 가로줄, 세로줄 그림. 기존 노트와 같은 예제)", "결과 값: <code>tools/pages/cnn_lines.py</code> (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html\" target=\"_blank\">nn.Conv2d</a>, <a href=\"https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html\" target=\"_blank\">ImageFolder</a>"])
print("ok")
