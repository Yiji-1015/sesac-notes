"""모델 > 이미지 > 전이학습과 오류 분석.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/transfer.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "transfer"
IMG = ROOT / "topics" / "model" / "img" / "transfer"
IMG.mkdir(parents=True, exist_ok=True)
code = S.parse((EX / "shapes_transfer.py").read_text(encoding="utf-8"))
run = lambda k: S.run(code, k, S.workdir(EX), filename="shapes_transfer.py")  # noqa: E731


def save(t, name):
    a = (t.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    Image.fromarray(a).resize((96, 96), Image.NEAREST).save(IMG / name)
    return f"img/transfer/{name}"


n2, n4, ns = run(2), run(4), run(8)
cl = ns["classes"]
r2, r4, r5, r7 = n2["report"](n2["scratch"]), n4["report"](n4["frozen"]), ns["report"](ns["finetuned"]), ns["report"](ns["weighted"])
outs = {}
Xt, yt = ns["X_train"], ns["y_train"]
outs["data"] = S.images("train 그림 예시 (32×32로 맞춤)", [(save(Xt[yt == c][i], f"ex{c}{i}.png"), cl[c]) for c in range(3) for i in range(2)],
                        [f"train 개수: {', '.join(f'{k} {v}장' for k, v in ns['counts'].items())}. 원이 세모의 6배예요. valid는 종류마다 30장씩이에요."])
outs["why"] = S.table("같은 구조, 같은 학습 횟수(5번)로 비교 (valid 90장)", ["방법", "정확도", "macro F1"],
                      [["처음부터 학습 (②)", f"{r2['accuracy']:.3f}", f"<span class=\"bad\">{r2['macro_f1']:.3f}</span>"], ["미리 배운 Backbone + Head만 학습 (④)", f"{r4['accuracy']:.3f}", f"<b>{r4['macro_f1']:.3f}</b>"]],
                      ["그림 180장, 5번 학습으로는 처음부터 배우기 어려워요. 다른 도형 그림으로 <b>이미 배운 특징 뽑기 부분</b>을 빌려 오면, 분류하는 마지막 층만 배워도 거의 다 맞혀요."])
outs["s1"] = S.box("① 실행 결과", [(f"X_train.shape → {tuple(Xt.shape)}", 1), f"classes → {cl}", (f"counts → {ns['counts']}", 2)],
                   ["40×40 그림을 32×32로 줄였어요. 빌려 올 모델이 그 크기로 배웠기 때문이에요. 입력 모양은 빌려 온 모델과 맞춰야 해요.", "종류마다 개수가 크게 달라요 (불균형). ⑥, ⑦에서 다시 봐요."])
outs["s2"] = S.box("② 처음부터 5번 학습", [f"가중치 수 → {sum(p.numel() for p in n2['make_model']().parameters()):,}개", (f"valid → 정확도 {r2['accuracy']:.3f}, macro F1 {r2['macro_f1']:.3f}", 1)],
                   ["무작위 가중치에서 시작해서, 선과 모서리 같은 기본 특징부터 배워야 해요. 데이터와 학습 횟수가 적으면 거기까지 가지 못해요."])
src = ns["source"]
outs["s3"] = S.box("③ source_backbone.npz", [f"{k} → {tuple(src[k].shape)}" for k in src.files if not k.endswith("num_batches_tracked")][:6] + [("load_state_dict → 성공 (이름과 모양이 모두 맞음)", 1)],
                   ["다른 도형 그림 300장으로 미리 학습해 둔 Backbone의 가중치예요 (이 그림들은 우리 train에 없어요). 층 번호와 모양이 우리 <code>make_backbone()</code>과 똑같아서 그대로 끼울 수 있어요."])
outs["s4"] = S.box("④ Backbone 고정, Head만 5번 학습", [(f"학습하는 가중치 → {n4['trainable']:,}개 / 전체 {sum(p.numel() for p in n4['frozen'].parameters()):,}개", 1), (f"valid → 정확도 {r4['accuracy']:.3f}, macro F1 {r4['macro_f1']:.3f}", 2)],
                   ["Head(256 → 3)만 배우니 학습할 게 3분의 1로 줄었어요.", "그런데도 ②보다 훨씬 잘 맞혀요. 빌려 온 Backbone이 이미 \"도형을 구분하는 특징\"을 뽑아 주니까요."])
outs["s5"] = S.box("⑤ 전체를 작은 학습률(0.001)로 5번 더", [(f"valid → 정확도 {r5['accuracy']:.3f}, macro F1 {r5['macro_f1']:.3f}", 1)],
                   [(f"④({r4['macro_f1']:.3f})보다 오히려 조금 떨어졌어요. 그림 180장으로 Backbone까지 고치면, 빌려 온 좋은 특징이 우리 데이터에 맞춰 흐트러질 수 있어요. 데이터가 적을 때는 Head만 배우는 게 나을 때가 많아요. " if r5['macro_f1'] < r4['macro_f1'] else "④와 거의 같아요. 빌려 온 특징이 이미 잘 맞아서 더 고칠 게 없었어요. ")
                    + "원래 데이터와 우리 데이터가 많이 다르고 데이터가 넉넉할수록 미세조정의 효과가 커요."])
pc = ns["per_class"]
outs["s6"] = S.table("⑥ 종류별 성적 (valid)", ["종류", "precision", "recall", "valid 장수"], [[c, f"{pc[c]['precision']:.2f}", f"{pc[c]['recall']:.2f}", int(pc[c]["support"])] for c in cl],
                     [f"혼동 행렬 (행: 실제, 열: 예측) → {ns['cm'].tolist()}",
                      "틀린 2장은 모두 \"원\"으로 잘못 답했어요. train에 원이 가장 많아서 애매하면 원 쪽으로 기우는 거예요. 정확도 하나로는 이런 치우침이 안 보여서 종류별로 봐요. macro F1은 종류마다 F1을 구해 <b>똑같은 무게로</b> 평균 낸 값이라 적은 종류도 공평하게 반영돼요."])
wrong = (ns["pred"] != ns["y_valid"]).nonzero().flatten().tolist()
outs["s6img"] = S.images("틀린 그림", [(save(ns["X_valid"][i], f"wrong{i}.png"), f"정답 {cl[int(ns['y_valid'][i])]} → 예측 {cl[int(ns['pred'][i])]}") for i in wrong],
                         ["틀린 그림을 직접 보면 원인을 짐작할 수 있어요. 모양이 작거나, 모서리가 뭉개졌거나, 다른 종류와 닮았는지요."])
outs["s7"] = S.box("⑦ 종류별 무게를 주고 미세조정", [(f"weight → {[round(float(w), 2) for w in ns['weight']]}  ({', '.join(cl)})", 1), (f"valid → 정확도 {r7['accuracy']:.3f}, macro F1 {r7['macro_f1']:.3f}", 2)],
                   ["적은 종류일수록 틀렸을 때 손실을 크게 쳐요. 세모 하나를 틀리면 원 하나를 틀린 것의 6배로요.", f"⑤({r5['macro_f1']:.3f})와 비교하면 valid에서 {r7['macro_f1']:.3f}로 조금 올랐어요."])
ts = ns["test_scores"]
outs["s8"] = S.table("⑧ test 30장 macro F1 (처음 사용)", ["방법", "macro F1"], [[k, f"{v:.3f}"] for k, v in ts.items()],
                     ["빌려 온 Backbone을 쓴 방법들이 처음부터 배운 것보다 크게 나아요. 그 사이의 차이(Head만, 미세조정, 무게)는 30장 중 한두 장이라 우연일 수 있어요. 이 데이터로 확실히 말할 수 있는 건 \"빌려 오면 훨씬 낫다\"까지예요."])
flow = [[1, "데이터", "180장", "불균형"], [2, "처음부터", "5번 학습", "부족함"], [3, "빌려 오기", "npz → Backbone", "모양이 같아야"], [4, "Head만", "Backbone 고정", "가중치 1/3"], [5, "미세조정", "작은 학습률", "전체 조금"], [6, "종류별", "macro F1", "틀린 그림"], [7, "무게", "적은 종류 ↑", "손실 무게"], [8, "test", "한 번", "비교"]]
S.save(ROOT / "topics" / "model" / "transfer-learning-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow, "env": f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CPU (seed 0)"})
why = f"""<p>세 가지 도형(원, 네모, 세모)을 구분하려고 해요. 그런데 그림이 180장뿐이고, 종류마다 개수도 달라요.</p>{K.out("data")}
<p>사람은 처음 보는 물건도 "모서리가 있고 둥글지 않다" 같은 이미 아는 특징으로 알아봐요. 모델도 마찬가지로, <b>다른 데이터로 미리 배운 특징 뽑기 부분</b>을 빌려 쓸 수 있어요. 이걸 <b>전이학습</b>이라고 해요. 실제로 비교하면:</p>{K.out("why")}
<p>CNN은 보통 두 부분이에요. 그림에서 특징을 뽑는 <b>Backbone</b>(합성곱 층들)과, 특징으로 종류를 정하는 <b>Head</b>(마지막 Linear)예요. 전이학습은 Backbone을 빌려 오고 Head는 우리 종류에 맞게 새로 만들어요.</p>"""
whole = """<p>데이터 보기 → 처음부터 학습(기준선) → Backbone 빌려 오기 → Head만 학습 → 전체 미세조정 → 종류별로 보기 → 적은 종류에 무게 → test. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "데이터 읽고 개수 세기", "그림을 빌려 올 모델과 같은 32×32로 맞춰 읽고, 종류마다 몇 장인지 세요.",
           K.code_table([("transforms.Resize((32, 32))", "크기 맞추기"), ("ds.targets", "그림마다 정답 번호 (폴더 순서대로 0, 1, 2)"), ("Counter(y_train.tolist())", "번호별 개수")]), "s1"),
    K.step(2, "처음부터 학습하기", "같은 구조를 무작위 가중치에서 시작해 5번 학습해요. 기준선이에요.",
           K.code_table([("make_backbone()", "합성곱 두 번 + 평균 풀링으로 특징 256개를 뽑는 부분"), ("nn.Sequential(make_backbone(), nn.Linear(256, 3))", "[0] Backbone, [1] Head"), ("fit(model, epochs, lr)", "32장씩 섞어 학습. <code>requires_grad</code>가 켜진 가중치만 고쳐요."), ("f1_score(..., average=\"macro\")", "종류별 F1의 평균")]), "s2"),
    K.step(3, "미리 배운 Backbone 빌려 오기", "저장된 가중치 배열을 읽어 같은 구조의 Backbone에 끼워요.",
           K.code_table([("np.load(\"source_backbone.npz\")", "이름별 가중치 배열 묶음"), ("load_state_dict({이름: Tensor})", "이름과 모양이 모두 맞아야 끼워져요. 하나라도 다르면 오류")]), "s3"),
    K.step(4, "Backbone 고정하고 Head만 학습하기", "빌려 온 가중치는 그대로 두고, 우리 종류를 고르는 Head만 배워요 (특징 추출).",
           K.code_table([("frozen[0].load_state_dict(pretrained.state_dict())", "새 모델의 Backbone 자리에 빌려 온 가중치"), ("p.requires_grad = False", "이 가중치는 학습하지 않음"), ("if p.requires_grad", "학습할 가중치만 세기")]), "s4"),
    K.step(5, "전체를 조금씩 고치기 (미세조정)", "고정을 풀고, 작은 학습률로 Backbone까지 우리 데이터에 맞게 살짝 고쳐요.",
           K.code_table([("p.requires_grad = True", "다시 학습 대상으로"), ("lr=0.001", "이미 좋은 가중치를 크게 망가뜨리지 않게 작게")]), "s5"),
    K.step(6, "종류별로 보기", "정확도 하나 대신 종류마다 precision, recall을 보고, 틀린 그림을 직접 봐요.",
           K.code_table([("classification_report(..., output_dict=True)", "종류별 precision, recall, F1, 장수를 dict로"), ("confusion_matrix(y, pred)", "실제 × 예측 표")]) + K.out("s6img"), "s6"),
    K.step(7, "적은 종류에 무게 주기", "개수가 적은 종류를 틀리면 손실을 더 크게 쳐서, 많은 종류 쪽으로 기우는 걸 줄여요.",
           K.code_table([("len(y_train) / (3 * n)", "종류 수가 고르면 모두 1, 적은 종류일수록 커요."), ("F.cross_entropy(..., weight=weight)", "정답 종류의 무게만큼 손실을 곱해요.")]), "s7"),
    K.step(8, "마지막 한 번: test", "아껴 둔 test 30장으로 세 방법을 한 번 채점해요.", K.code_table([("load(\"test\")", "①의 함수로 test 읽기")]), "s8"),
]
practice = [("진짜 사진", "ImageNet 같은 큰 사진 데이터로 배운 ResNet, EfficientNet 등을 <code>torchvision.models</code>에서 불러와요 (처음 한 번 가중치를 내려받아요). 그 모델이 배운 크기와 정규화(평균, 표준편차)를 그대로 맞춰요."),
            ("얼마나 풀어서 고칠까", "데이터가 적고 원래 데이터와 비슷하면 Head만, 데이터가 많거나 많이 다르면 뒤쪽 층부터 조금씩 풀어서 고쳐요."),
            ("모델이 어디를 봤나", "Grad-CAM은 마지막 합성곱 층의 기울기로 \"판단에 영향을 준 자리\"를 그림 위에 칠해 보여 줘요. 배경을 보고 맞힌 건 아닌지 확인할 때 써요 (<a href=\"../../notes/day14.html\">옛 Day14 노트</a>)."),
            ("불균형이 심하다", "무게 주기 말고도, 적은 종류를 더 자주 뽑아 학습하거나(oversampling), 증강으로 늘리는 방법이 있어요. 효과는 꼭 종류별 지표로 확인해요.")]
libs = [("①", "<code>datasets.ImageFolder</code>, <code>transforms.Resize</code>, <code>collections.Counter</code>", "읽기, 세기"), ("②", "<code>nn.Conv2d</code>, <code>nn.BatchNorm2d</code>, <code>nn.AdaptiveAvgPool2d</code>, <code>F.cross_entropy</code>", "모델과 학습"),
        ("③④⑤", "<code>np.load</code>, <code>load_state_dict</code>, <code>state_dict</code>, <code>requires_grad</code>", "가중치 빌려 오기, 고정"), ("⑥⑦", "<code>classification_report</code>, <code>confusion_matrix</code>, <code>f1_score(average=\"macro\")</code>, <code>weight=</code>", "종류별 평가, 무게 (scikit-learn, torch)")]
K.build(ROOT / "topics" / "model" / "transfer-learning.html", title="전이학습과 오류 분석", h1="전이학습과 오류 분석: 미리 배운 특징을 빌려 쓰기", track="모델", crumb="모델 &gt; 이미지",
        one_line="데이터가 적으면 처음부터 배우기 어려워요. <b>다른 데이터로 미리 배운 Backbone</b>을 빌려 오고 <b>Head만 새로</b> 배우면 적은 데이터로도 잘 맞혀요. 결과는 <b>종류별로</b> 보고, 틀린 그림을 직접 봐요.",
        why=why, whole=whole, steps=steps, code_file="examples/transfer/shapes_transfer.py", practice=practice, libs=libs, libs_note="<code>load</code>, <code>make_backbone</code>, <code>make_model</code>, <code>fit</code>, <code>predict</code>, <code>report</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"cnn.html\">CNN과 이미지 분류</a>", "Backbone이 하는 일, 합성곱"), ("<a href=\"object-detection.html\">객체 탐지</a>", "종류뿐 아니라 위치까지 찾기")],
        sources=["다루는 범위: Day14 (2026-11-05) 데이터셋 설계, 전이학습, Fine-tuning, 불균형 평가, 오류 분석", "데이터: <code>examples/transfer/data/</code> (합성 도형 그림과 미리 학습한 작은 Backbone. 기존 노트와 같은 실습 데이터)", "결과 값: <code>tools/pages/transfer.py</code> (seed 0)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html\" target=\"_blank\">Transfer Learning for Computer Vision</a>, <a href=\"https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html\" target=\"_blank\">classification_report</a>"])
print("ok")
