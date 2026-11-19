"""모델 > 머신러닝 기초 세 쪽을 만든다. examples/ml/requests_model.py 하나가 ①~⑰로 세 쪽에 걸쳐 자란다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/ml_basics.py
"""
import sys
import warnings
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import xgboost  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "ml"
code = S.parse((EX / "requests_model.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="requests_model.py")  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, pandas {pd.__version__}"
DATA_SRC = "데이터: <code>examples/ml/data/requests.csv</code> (가상의 업무 요청 100건)와 <code>split_manifest.csv</code> (train 60, valid 20, test 20). 기존 노트와 같은 예제"
ns = run(17)
raw = ns["raw"]


def df_rows(df):
    return [[E(str(i))] + [E(str(v)) for v in r] for i, r in zip(df.index, df.values)]


# =============================== 1쪽: 회귀와 분류 ===============================
outs = {}
outs["data"] = S.table("requests.csv 앞 5줄", ["request_id"] + list(raw.columns), df_rows(raw.head(5)),
                       ["<code>failed</code>(실패했나, 1이면 실패)와 <code>elapsed_ms</code>(처리 시간)를 맞히는 게 목표예요.",
                        "<code>elapsed_ms</code>, <code>retry_count</code>, <code>status</code>는 요청을 <b>처리한 뒤에야</b> 알 수 있는 값이라 입력으로 쓰면 안 돼요. 미래를 보고 맞히는 셈이라서요 (데이터 누수)."])
n1 = run(1)
outs["s1"] = S.box("① 실행 결과", [(f"train {len(n1['train'])}건, valid {len(n1['valid'])}건, test {len(n1['test'])}건", 1),
                                   (f"실패 비율 → train {n1['y_train'].mean():.2f}, valid {n1['y_valid'].mean():.2f}, test {n1['test']['failed'].mean():.2f}", 2),
                                   f"빈 칸 → input_chars {int(raw['input_chars'].isna().sum())}개, allowed_tools {int(raw['allowed_tools'].isna().sum())}개"],
                   ["나누는 방법을 파일(manifest)로 고정해 둬서, 몇 번을 돌려도 같은 행이 같은 쪽에 들어가요.", "세 쪽 모두 실패가 30%로 비슷하게 나뉘었어요. 빈 칸은 ③에서 채워요."])
outs["s2"] = S.box("② 기준선", [(f"분류: 언제나 \"성공\"이라고 답하기 → 정확도 {ns['base_acc']:.2f}", 1), (f"회귀: 언제나 train 평균 시간이라고 답하기 → MAE {ns['base_mae']:.0f} ms", 2)],
                   ["아무것도 배우지 않은 모델이 정확도 0.70이에요. 실패가 30%뿐이라서 \"다 성공\"이라고만 해도 70%는 맞거든요. 그러니 정확도 0.70은 잘한 게 아니에요.", "평균으로만 찍어도 평균 541 ms 틀려요. 모델은 이보다 나아야 의미가 있어요."])
pre = ns["build"](ns["LogisticRegression"]()).named_steps["pre"].fit(ns["X_train"])
outs["s3"] = S.box("③ 전처리만 따로 돌려 보면", [(f"입력 4열 → 전처리 뒤 {len(pre.get_feature_names_out())}열", 1), f"{list(pre.get_feature_names_out())}"],
                   ["숫자 두 열은 빈 칸을 중앙값으로 채우고 평균 0, 표준편차 1로 맞췄어요. 글자 두 열은 값마다 0/1 열로 펼쳤어요 (원-핫). 모델은 숫자만 받으니까요."])
outs["s4"] = S.box("④ 선형 회귀로 처리 시간 예측 (valid)", [f"MAE → {ns['reg_mae']:.0f} ms  (기준선 {ns['base_mae']:.0f} ms)", (f"R² → {ns['reg_r2']:.3f}", 1, "bad")],
                   ["R²는 \"평균으로 찍는 것보다 얼마나 나은가\"예요. 1이면 완벽, 0이면 평균과 같고, <b>0보다 작으면 평균보다 못해요</b>. 이 네 입력만으로는 처리 시간을 거의 설명하지 못한다는 뜻이에요. 모델을 바꾸기 전에 \"입력이 충분한가\"부터 의심해야 하는 신호예요."])
cm = ns["cm"]
outs["s5"] = S.box("⑤ 로지스틱 회귀로 실패 예측 (valid 20건)", ["혼동 행렬 (행: 실제, 열: 예측)", f"            예측 성공  예측 실패", f"실제 성공      {cm[0][0]:>3}       {cm[0][1]:>3}", (f"실제 실패      {cm[1][0]:>3}       {cm[1][1]:>3}", 1), "",
                                                      (f"accuracy {ns['scores']['accuracy']:.2f}, precision {ns['scores']['precision']:.2f}, recall {ns['scores']['recall']:.2f}, F1 {ns['scores']['f1']:.2f}", 2)],
                   ["실제 실패 6건 중 2건만 잡았어요. 4건은 놓쳤어요.", "정확도는 기준선(0.70)보다 낮지만, 기준선은 실패를 <b>하나도</b> 못 잡아요(F1 0). 실패를 찾는 게 목적이면 정확도보다 recall(실패 중 잡은 비율)과 F1을 봐야 해요."])
from sklearn.metrics import f1_score as F1
ks, kr = F1(ns["y_valid"], ns["knn_scaled"].predict(ns["X_valid"])), F1(ns["y_valid"], ns["knn_raw"].predict(ns["X_valid"]))
outs["s6"] = S.box("⑥ KNN (k=5), 스케일 있음과 없음 (valid F1)", [f"스케일 있음 → {ks:.2f}", f"스케일 없음 → {kr:.2f}", f"input_chars 범위 {raw['input_chars'].min():.0f}~{raw['input_chars'].max():.0f}, allowed_tools 범위 {raw['allowed_tools'].min():.0f}~{raw['allowed_tools'].max():.0f}"],
                   ["스케일이 없으면 거리가 거의 <code>input_chars</code>(수백~수천)로만 정해지고, <code>allowed_tools</code>(1~5)는 무시돼요. 이 데이터에서는 그게 오히려 점수가 높게 나왔어요. 입력 중 <code>input_chars</code>가 가장 쓸모 있어서일 수도, valid가 20건뿐이라 우연일 수도 있어요. 한 번의 점수로 결론 내리지 않는 이유예요 (2쪽의 교차검증)."])
outs["s7"] = S.table("⑦ 같은 train, 같은 valid로 비교", ["model", "F1", "recall", "accuracy"], df_rows(ns["board"]),
                     ["정확도만 보면 dummy가 1등이에요. F1로 보면 logistic이 가장 나아요. <b>무엇을 기준으로 고르느냐</b>가 결과를 바꿔요.", "모두 같은 나누기, 같은 전처리 함수를 썼어요. 조건이 다르면 비교가 아니에요."])
flow1 = [[1, "읽고 나누기", "60 / 20 / 20", "누수 열 빼기"], [2, "기준선", "Dummy", "정확도 0.70"], [3, "Pipeline", "채우기, 맞추기", "원-핫"],
         [4, "회귀", "선형", "MAE, R²"], [5, "분류", "로지스틱", "혼동 행렬, F1"], [6, "KNN", "거리", "스케일"], [7, "비교", "같은 조건", "기준이 결과를 바꿈"]]
S.save(ROOT / "topics" / "model" / "ml-regression-classification-data.js", {"code": [c for c in code if c["s"] <= 7], "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow1, "env": ENV})

why1 = f"""
<p>"이 요청은 실패할까?", "처리에 얼마나 걸릴까?"처럼, 지난 기록으로 앞으로를 맞히고 싶을 때가 있어요. 사람이 규칙을 일일이 정하는 대신 <b>기록에서 규칙을 배우게</b> 하는 게 머신러닝이에요. 표(행과 열) 데이터에서는 보통 두 가지를 해요.</p>
{K.table(["종류", "맞히는 것", "예"], [["<b>회귀</b>", "숫자", "처리 시간 1,520 ms"], ["<b>분류</b>", "정해진 종류 중 하나", "실패 / 성공"]])}
<p>예제는 업무 요청 100건이에요.</p>
{K.out("data")}
<p>모델을 만들기 전에 꼭 정해야 할 게 있어요. <b>무엇과 비교해서 잘했다고 할지</b>(기준선), <b>무엇으로 잴지</b>(지표), <b>어떤 데이터로 잴지</b>(나누기)예요. 이게 없으면 "정확도 70%"가 좋은 건지 나쁜 건지 알 수 없어요.</p>
"""
whole1 = """<p>나누고, 기준선을 정하고, 전처리를 묶고, 회귀와 분류를 해 보고, 같은 조건으로 비교해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p><code>requests_model.py</code> 하나가 이 쪽(①~⑦)에서 시작해 <a href="ml-trees-cv.html">2쪽</a>(⑧~⑫), <a href="ml-metrics-clustering.html">3쪽</a>(⑬~⑰)까지 자라요.</p>"""
steps1 = [
    K.step(1, "데이터 읽고 나누기", "요청을 받을 때 아는 네 열만 입력으로 고르고, 행을 train, valid, test로 나눠요.",
           "<p><b>train</b>으로 배우고, <b>valid</b>로 모델을 고르고, <b>test</b>는 마지막에 딱 한 번 성적을 매기는 데 써요. 같은 데이터로 배우고 채점하면, 답을 외운 모델도 100점이 나와요.</p>"
           + K.code_table([("pd.read_csv(...).set_index(\"request_id\")", "표를 읽고, 요청 번호를 행 이름으로"), ("manifest[manifest[\"split\"] == name]", "이 쪽에 들어갈 행 번호만"), ("raw.loc[ids]", "그 번호의 행만 꺼내기")]), "s1"),
    K.step(2, "기준선 만들기", "아무것도 배우지 않는 모델의 점수를 먼저 재요.",
           K.code_table([("DummyClassifier(strategy=\"most_frequent\")", "가장 많은 답(성공)만 말하는 분류기"), ("DummyRegressor(strategy=\"mean\")", "train 평균만 말하는 회귀기"),
                         ("mean_absolute_error", "예측과 실제 차이의 평균 (MAE). 작을수록 좋아요.")]), "s2"),
    K.step(3, "전처리를 Pipeline으로 묶기", "빈 칸 채우기, 크기 맞추기, 글자를 숫자로 바꾸기를 모델과 한 덩어리로 묶어요.",
           "<p>전처리를 따로 하면 실수로 valid나 test까지 보고 기준(중앙값, 평균)을 정하기 쉬워요. 그러면 채점할 데이터를 미리 엿본 셈이에요. Pipeline으로 묶으면 <code>fit</code>할 때 <b>train만</b> 보고 기준을 정해요.</p>"
           + K.code_table([("SimpleImputer(strategy=\"median\")", "빈 칸을 중앙값으로"), ("StandardScaler()", "평균 0, 표준편차 1로"), ("OneHotEncoder(handle_unknown=\"ignore\")", "글자 값마다 0/1 열. 처음 보는 값은 모두 0"),
                           ("ColumnTransformer([...])", "열마다 다른 전처리"), ("Pipeline([(\"pre\", ...), (\"model\", ...)])", "전처리와 모델을 한 덩어리로"), ("clone(model)", "설정만 같은 새 모델. 같은 모델 객체를 여러 번 쓰지 않으려고")]), "s3"),
    K.step(4, "회귀: 처리 시간 예측", "선형 회귀로 처리 시간을 맞히고, 기준선과 비교해요.",
           K.code_table([("LinearRegression()", "입력에 곱할 가중치를 배워 더하는 가장 기본 회귀"), ("r2_score", "결정 계수. 평균보다 얼마나 나은지")]), "s4"),
    K.step(5, "분류: 실패 여부 예측", "로지스틱 회귀로 실패를 맞히고, 혼동 행렬과 여러 지표로 봐요.",
           K.table(["지표", "뜻"], [["accuracy", "전체 중 맞힌 비율"], ["precision", "\"실패\"라고 한 것 중 진짜 실패 비율"], ["recall", "진짜 실패 중 잡아낸 비율"], ["F1", "precision과 recall의 조화 평균"]])
           + K.code_table([("LogisticRegression(max_iter=1000)", "확률을 내는 선형 분류기. 0.5 이상이면 실패"), ("confusion_matrix(y, pred, labels=[0, 1])", "실제와 예측의 2×2 표")]), "s5"),
    K.step(6, "KNN: 거리로 판단하는 모델", "가장 가까운 요청 5개의 결과로 투표해요. 거리를 쓰니 숫자 크기에 민감해요.",
           K.code_table([("KNeighborsClassifier(n_neighbors=5)", "가까운 5개의 다수결"), ("build(..., scale=False)", "크기 맞추기를 빼고 비교")]), "s6"),
    K.step(7, "같은 조건으로 비교하기", "모델만 바꾸고 나머지(나누기, 전처리, 지표)는 같게 해서 표로 비교해요.",
           K.code_table([("evaluate({이름: 모델})", "모델마다 같은 train으로 배우고 같은 valid로 잰 표")]), "s7"),
]
practice1 = [("데이터가 적다", "valid 20건이면 한두 건만 바뀌어도 점수가 크게 흔들려요. 한 번의 점수로 고르지 말고 교차검증을 써요 (<a href=\"ml-trees-cv.html\">2쪽</a>)."),
             ("R²가 음수", "모델보다 입력부터 의심해요. 처리 시간을 정말 설명할 값(서버 상태, 요청 내용 등)이 입력에 있나요?"),
             ("실패가 드물다", "정확도 대신 recall, F1을 보고, 필요하면 확률 기준(0.5)을 바꿔요 (<a href=\"ml-metrics-clustering.html\">3쪽</a>)."),
             ("처리 후에 생기는 열", "<code>status</code>처럼 결과가 이미 담긴 열을 입력에 넣으면 점수가 비현실적으로 높게 나와요. 실제로 쓸 때는 그 값이 없어서 무너져요.")]
libs1 = [("①", "<code>pd.read_csv()</code>, <code>.loc[]</code>", "읽기, 나누기 (pandas)"), ("②", "<code>DummyClassifier</code>, <code>DummyRegressor</code>", "기준선 (scikit-learn)"),
         ("③", "<code>Pipeline</code>, <code>ColumnTransformer</code>, <code>SimpleImputer</code>, <code>StandardScaler</code>, <code>OneHotEncoder</code>", "전처리 묶기"),
         ("④⑤", "<code>LinearRegression</code>, <code>LogisticRegression</code>, <code>mean_absolute_error</code>, <code>r2_score</code>, <code>confusion_matrix</code>, <code>f1_score</code>", "모델과 지표"), ("⑥", "<code>KNeighborsClassifier</code>", "KNN")]
K.build(ROOT / "topics" / "model" / "ml-regression-classification.html", title="회귀와 분류", h1="회귀와 분류: 표 데이터로 예측하기", track="모델", crumb="모델 &gt; 머신러닝 기초 1/3쪽 &nbsp;|&nbsp; 다음: <a href=\"ml-trees-cv.html\" style=\"color:inherit\">트리와 교차검증</a>",
        one_line="지난 기록에서 규칙을 배워 <b>숫자를 맞히면 회귀</b>, <b>종류를 맞히면 분류</b>예요. 모델보다 먼저 <b>기준선, 지표, 나누기</b>를 정해야 점수가 의미를 가져요.",
        why=why1, whole=whole1, steps=steps1, code_file="examples/ml/requests_model.py", practice=practice1, libs=libs1, libs_note="<code>part</code>, <code>build</code>, <code>evaluate</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "다른 모델과 믿을 만한 비교"), ("<a href=\"ml-metrics-clustering.html\">평가지표, 특성 설계, 군집</a>", "확률 기준, 비용, 최종 평가")],
        sources=["다루는 범위: Day05~06 (2026-10-23, 26) 문제 정의, 전처리 Pipeline, 회귀, 분류, KNN, 모델 비교", DATA_SRC, "결과 값: <code>tools/pages/ml_basics.py</code>가 단계별로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/compose.html\" target=\"_blank\">Pipelines and composite estimators</a>, <a href=\"https://scikit-learn.org/stable/common_pitfalls.html\" target=\"_blank\">Common pitfalls</a>, <a href=\"https://scikit-learn.org/stable/modules/model_evaluation.html\" target=\"_blank\">Model evaluation</a>"])

# =============================== 2쪽: 트리, 앙상블, 교차검증 ===============================
outs = {}
outs["why"] = S.table("1쪽 ⑦의 valid 점수 vs 같은 모델의 5-fold 교차검증 (F1)", ["model", "valid 20건 한 번", "교차검증 다섯 번", "평균"],
                      [[m, f3(ns["board2"].loc[m, "f1"]) if m in ns["board2"].index else "-", ", ".join(f3(x) for x in ns["cv_scores"][m]), f"<b>{f3(ns['cv_scores'][m].mean())}</b>"] for m in ["logistic", "forest", "xgboost"]],
                      ["valid 한 번으로는 forest가 가장 좋아 보였는데, 다섯 번 나눠 재면 xgboost가 가장 좋아요. 그리고 같은 모델도 나누는 방법에 따라 0에서 0.5까지 흔들려요.", "데이터가 적을수록 \"한 번 잰 점수\"는 믿기 어려워요. 이 쪽에서는 표 데이터에 강한 트리 모델들과, 점수를 믿을 만하게 재는 교차검증을 다뤄요."])
ts = ns["tree_scores"]
outs["s8"] = S.table("⑧ 결정 트리의 깊이", ["설정", "train 정확도", "valid F1"], [[k, f3(v[0]), f3(v[1])] for k, v in ts.items()],
                     ["깊이 제한이 없으면 train을 <b>100%</b> 맞혀요. 100건 남짓을 다 외운 거예요. 외운 모델은 새 데이터에서 무너지기 쉬워서(과적합), 보통 깊이를 제한해요.", "이 데이터에서는 깊은 트리의 valid F1이 오히려 높게 나왔는데, valid가 20건이라 우연일 수 있어요. 교차검증(⑫)으로 다시 봐야 해요."])
outs["s10"] = S.table("⑩ 네 모델을 같은 valid로", ["model", "F1", "recall", "accuracy"], df_rows(ns["board2"]), ["valid 한 번의 결과예요. ⑫에서 교차검증으로 다시 비교해요."])
imp = ns["importance"]
outs["s11"] = S.table("⑪ 랜덤 포레스트의 특성 중요도", ["전처리 뒤 열", "중요도"], [[f"<code>{E(i)}</code>", f3(v)] for i, v in imp.items()],
                      ["<code>input_chars</code>(입력 길이)가 판단에 가장 많이 쓰였어요. 1쪽 ⑥에서 KNN이 이 열만 본 게 오히려 잘 맞은 것과 같은 이야기예요.", "중요도는 \"모델이 많이 썼다\"는 뜻이지 \"원인이다\"는 뜻이 아니에요."])
outs["s12"] = S.table("⑫ 5-fold 교차검증 F1 (train+valid 80건)", ["model", "다섯 번의 점수", "평균", "표준편차"],
                      [[m, ", ".join(f3(x) for x in v), f"<b>{f3(v.mean())}</b>", f3(v.std())] for m, v in ns["cv_scores"].items() if m != "logistic+log1p"],
                      ["80건을 다섯 덩어리로 나눠, 넷으로 배우고 하나로 재기를 다섯 번 해요. 모든 행이 한 번씩 채점에 쓰여요.", "평균과 함께 표준편차(흔들림)도 봐요. 평균 차이가 흔들림보다 작으면 \"더 낫다\"고 말하기 어려워요."])
flow2 = [[8, "결정 트리", "질문을 이어 가기", "깊이 = 과적합"], [9, "랜덤 포레스트", "트리 여러 개 투표", "흔들림 줄이기"], [10, "XGBoost", "틀린 걸 이어서 고치기", "부스팅"], [11, "중요도", "어느 열을 썼나", "원인은 아님"], [12, "교차검증", "다섯 번 재기", "평균과 흔들림"]]
S.save(ROOT / "topics" / "model" / "ml-trees-cv-data.js", {"code": [c for c in code if c["s"] <= 12], "names": S.names(code), "range": [8, 12], "outs": outs, "flow": flow2, "env": ENV})
why2 = f"""<p><a href="ml-regression-classification.html">1쪽</a>에서는 선형 모델과 KNN을 valid 20건으로 비교했어요. 그런데 valid 한 번의 점수가 얼마나 믿을 만할까요? 1쪽의 모델과 이 쪽의 트리 모델들을 두 방법으로 재 봤어요.</p>
{K.out("why")}"""
whole2 = """<p>트리 하나 → 트리 여럿의 투표(포레스트) → 틀린 걸 고쳐 가는 트리들(부스팅) → 어느 열을 썼나 → 믿을 만하게 재기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>1쪽의 코드(①~⑦)에 이어서 같은 파일이 자라요. 오른쪽 코드에서 1쪽 부분은 접혀 있어요.</p>"""
steps2 = [
    K.step(8, "결정 트리", "\"입력 길이가 1300보다 긴가?\" 같은 예/아니오 질문을 이어 가며 답을 정해요.",
           "<p>트리는 사람이 읽을 수 있는 규칙을 만들어요. 그리고 숫자의 크기 차이에 영향을 받지 않아서 크기 맞추기가 필요 없어요 (<code>scale=False</code>).</p>"
           + K.code_table([("DecisionTreeClassifier(max_depth=3)", "질문을 최대 3번까지만. 깊을수록 train을 잘 외워요."), ("random_state=0", "같은 결과가 나오게 무작위를 고정")]), "s8"),
    K.step(9, "랜덤 포레스트", "조금씩 다른 데이터와 열로 트리를 200개 만들어 투표해요.",
           "<p>트리 하나는 데이터가 조금만 바뀌어도 모양이 크게 바뀌어요. 서로 다르게 만든 트리 여럿의 다수결은 그런 흔들림이 줄어요.</p>" + K.code_table([("RandomForestClassifier(n_estimators=200, max_depth=4)", "깊이 4인 트리 200개")]), None),
    K.step(10, "XGBoost", "앞 트리가 틀린 부분을 다음 트리가 이어서 고치는 식으로 트리를 쌓아요 (부스팅).",
           K.code_table([("XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1)", "작은 트리 200개를 차례로. 한 번에 조금씩(0.1) 고쳐요."), ("evaluate({...})", "1쪽의 비교 함수를 그대로 다시 써요.")]), "s10"),
    K.step(11, "특성 중요도", "포레스트가 판단할 때 어느 열을 많이 썼는지 봐요.",
           K.code_table([("named_steps[\"pre\"].get_feature_names_out()", "전처리 뒤 열 이름"), ("named_steps[\"model\"].feature_importances_", "열마다 중요도 (합이 1)")]), "s11"),
    K.step(12, "교차검증", "train+valid 80건을 다섯 덩어리로 나눠, 다섯 번 배우고 재서 평균을 내요.",
           K.code_table([("pd.concat([train, valid])", "고르는 데 쓸 80건. test는 3쪽 마지막까지 안 써요."), ("StratifiedKFold(n_splits=5, shuffle=True)", "덩어리마다 실패 비율이 비슷하게 나눠요."),
                         ("cross_validate(model, X, y, cv=cv, scoring=\"f1\")", "다섯 번 배우고 재서 점수 목록을 돌려줘요. 전처리도 매번 그 차례의 train으로만 다시 배워요.")]), "s12"),
]
practice2 = [("트리 깊이, 트리 개수는 어떻게 정하나", "후보 몇 개를 교차검증으로 비교해서 골라요 (<code>GridSearchCV</code>). test로 고르면 안 돼요."),
             ("부스팅이 늘 이기나", "표 데이터에서 자주 강하지만, 데이터가 적으면 과적합하기 쉬워요. 교차검증 흔들림을 같이 봐요."),
             ("중요도를 원인으로 읽고 싶다", "중요도는 서로 비슷한 열끼리 나눠 갖기도 해서, 원인 분석에는 따로 실험이 필요해요.")]
libs2 = [("⑧", "<code>DecisionTreeClassifier</code>", "결정 트리 (scikit-learn)"), ("⑨", "<code>RandomForestClassifier</code>", "랜덤 포레스트"), ("⑩", "<code>XGBClassifier</code>", "그래디언트 부스팅 (xgboost)"),
         ("⑪", "<code>feature_importances_</code>, <code>get_feature_names_out()</code>", "중요도"), ("⑫", "<code>StratifiedKFold</code>, <code>cross_validate</code>", "교차검증")]
K.build(ROOT / "topics" / "model" / "ml-trees-cv.html", title="트리와 앙상블, 교차검증", h1="트리와 앙상블, 교차검증: 더 나은 모델을 믿을 만하게 고르기", track="모델",
        crumb="모델 &gt; 머신러닝 기초 2/3쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-regression-classification.html\" style=\"color:inherit\">회귀와 분류</a> &nbsp;|&nbsp; 다음: <a href=\"ml-metrics-clustering.html\" style=\"color:inherit\">평가지표와 군집</a>",
        one_line="<b>트리</b>는 예/아니오 질문을 이어 가며 판단하고, 여러 트리를 묶으면(<b>앙상블</b>) 더 안정돼요. 어느 모델이 나은지는 한 번이 아니라 <b>교차검증</b>으로 여러 번 재서 판단해요.",
        why=why2, whole=whole2, steps=steps2, code_file="examples/ml/requests_model.py", practice=practice2, libs=libs2,
        next_rows=[("<a href=\"ml-regression-classification.html\">회귀와 분류</a>", "기준선, Pipeline, 지표"), ("<a href=\"ml-metrics-clustering.html\">평가지표, 특성 설계, 군집</a>", "확률 기준, 최종 평가")],
        sources=["다루는 범위: Day07 (2026-10-27) 트리, 앙상블, 특성 중요도, 과적합, 교차검증", DATA_SRC, "결과 값: <code>tools/pages/ml_basics.py</code>",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/tree.html\" target=\"_blank\">Decision Trees</a>, <a href=\"https://scikit-learn.org/stable/modules/ensemble.html\" target=\"_blank\">Ensembles</a>, <a href=\"https://scikit-learn.org/stable/modules/cross_validation.html\" target=\"_blank\">Cross-validation</a>, <a href=\"https://xgboost.readthedocs.io/\" target=\"_blank\">XGBoost</a>"])

# =============================== 3쪽: 평가지표, 특성 설계, 군집, 최종 ===============================
outs = {}
proba = ns["proba"]
outs["why"] = S.table("1쪽 로지스틱 모델이 valid 실패 6건에 준 확률", ["request_id", "실패 확률", "0.5 기준 판단"],
                      [[E(i), f3(p), "실패" if p >= 0.5 else "<span class=\"bad\">성공 (놓침)</span>"] for i, p, y in zip(ns["X_valid"].index, proba, ns["y_valid"]) if y == 1],
                      ["모델은 사실 \"실패\" 아니면 \"성공\"이 아니라 <b>확률</b>을 내요. 0.5라는 기준선은 우리가 정한 거예요. 0.3, 0.4짜리 실패도 \"성공\"으로 넘어갔어요.", "놓친 실패가 큰 손해라면 기준을 낮춰야 해요. 이 쪽에서는 그 기준을 비용으로 정하고, 확률의 품질을 재고, 입력을 다듬고, 정답 없이 데이터를 묶어 보고, 마지막으로 test로 딱 한 번 채점해요."])
outs["s13"] = S.table("⑬ 기준별 오류와 비용 (FN 하나 = 5, FP 하나 = 1)", ["기준", "헛경보 FP", "놓침 FN", "비용"],
                      [[c["threshold"], c["fp"], c["fn"], f"<b>{c['cost']}</b>" if c["cost"] == min(x["cost"] for x in ns["costs"]) else c["cost"]] for c in ns["costs"]],
                      [f"기준을 낮추면 헛경보는 늘고 놓침은 줄어요. 비용이 가장 작은 기준은 {min(ns['costs'], key=lambda c: c['cost'])['threshold']}예요.", "비용 5:1은 설명용 가정이에요. 실제로는 \"실패를 놓치면 무슨 일이 생기나\"를 업무 담당자와 정해요. 기준은 valid로 정하고, test로 정하면 안 돼요."])
outs["s14"] = S.box("⑭ ROC-AUC", [(f"roc_auc_score(y_valid, proba) → {ns['auc']:.3f}", 1)],
                    ["\"실패 하나와 성공 하나를 뽑았을 때, 실패 쪽에 더 높은 확률을 줄 확률\"이에요. 0.5면 동전 던지기, 1이면 완벽. 기준과 상관없이 확률의 순서만 봐요. 0.57이면 이 모델의 확률은 순서도 많이 틀려요."])
outs["s15"] = S.box("⑮ log1p 변환을 넣은 로지스틱의 교차검증 F1", [f"점수 → {[round(float(x), 3) for x in ns['cv_scores']['logistic+log1p']]}", (f"평균 → {ns['cv_scores']['logistic+log1p'].mean():.3f}  (변환 없는 logistic {ns['cv_scores']['logistic'].mean():.3f})", 1, "bad")],
                    ["치우친 숫자를 펴 주는 변환인데, 이 데이터에서는 오히려 나빠졌어요. 특성 설계는 \"해 보고 교차검증으로 확인\"이에요. 좋아 보이는 기법도 데이터에 따라 손해일 수 있어요."])
outs["s16"] = S.table("⑯ K-Means: 몇 개로 묶을 때 가장 잘 나뉘나", ["묶음 수 k", "실루엣 점수"], [[k, f"<b>{f3(v)}</b>" if v == max(ns["silhouettes"].values()) else f3(v)] for k, v in ns["silhouettes"].items()],
                      [f"정답(<code>failed</code>) 없이 입력만 보고 비슷한 요청끼리 묶었어요. 실루엣 점수는 \"내 묶음과는 가깝고 다른 묶음과는 먼가\"(−1~1)예요. k={max(ns['silhouettes'], key=ns['silhouettes'].get)}일 때 가장 높지만 0.24로, 뚜렷하게 나뉘는 데이터는 아니에요."])
outs["s17"] = S.box("⑰ 최종 평가", [(f"교차검증 평균으로 고른 모델 → {ns['best']}", 1), (f"test 20건 (처음 사용) → F1 {ns['test_scores']['f1']:.3f}, recall {ns['test_scores']['recall']:.3f}, accuracy {ns['test_scores']['accuracy']:.2f}", 2)],
                    ["⑫와 ⑮의 교차검증 평균이 가장 높은 모델을 골라, train+valid 80건 전부로 다시 배웠어요.", "test는 이때 처음 썼어요. 이 숫자가 \"새 요청에서 기대할 성능\"이에요. 이걸 보고 다시 모델을 바꾸면 test도 고르는 데 쓴 셈이라, 더는 공정한 성적이 아니에요."])
flow3 = [[13, "기준과 비용", "0.5가 정답 아님", "FN, FP 비용"], [14, "ROC-AUC", "확률의 순서", "기준과 무관"], [15, "특성 설계", "log1p", "교차검증으로 확인"], [16, "군집", "K-Means", "정답 없이 묶기"], [17, "최종", "test 한 번", "고른 뒤 채점"]]
S.save(ROOT / "topics" / "model" / "ml-metrics-clustering-data.js", {"code": code, "names": S.names(code), "range": [13, 17], "outs": outs, "flow": flow3, "env": ENV})
why3 = f"""<p>지금까지는 모델이 \"실패\" 또는 \"성공\"이라고 답하는 것만 봤어요. 그런데 그 답은 확률에 0.5라는 기준을 그어 만든 거예요.</p>{K.out("why")}"""
whole3 = """<p>기준을 비용으로 정하고, 확률 자체의 품질을 재고, 입력을 다듬어 보고, 정답 없이 묶어 보고, 마지막에 test로 한 번 채점해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps3 = [
    K.step(13, "임계값과 오류 비용", "확률 기준을 여러 개 시험해서, 오류 비용이 가장 작은 기준을 찾아요.",
           K.code_table([("logit.predict_proba(X_valid)[:, 1]", "실패일 확률 (두 번째 열)"), ("(proba >= t).astype(int)", "기준 t 이상이면 실패(1)"), ("confusion_matrix(...).ravel()", "2×2 표를 tn, fp, fn, tp 네 값으로 펴기")]), "s13"),
    K.step(14, "ROC-AUC", "기준과 상관없이, 확률이 실패를 성공보다 높게 매기는지 재요.", K.code_table([("roc_auc_score(y, proba)", "예측 \"답\"이 아니라 \"확률\"을 넣어요.")]), "s14"),
    K.step(15, "특성 설계: 치우친 값 펴기", "숫자 열에 log(1+x)를 씌워 큰 값을 줄여 본 뒤, 교차검증으로 나아졌는지 확인해요.",
           K.code_table([("FunctionTransformer(np.log1p)", "아무 함수나 전처리 단계로. <code>log1p</code>는 log(1+x)로, 0도 처리돼요."), ("feature_names_out=\"one-to-one\"", "열 이름을 그대로 유지")]), "s15"),
    K.step(16, "K-Means: 정답 없이 묶어 보기", "정답 없이 입력만으로 비슷한 요청끼리 k개 묶음을 만들고, 몇 개가 적당한지 봐요.",
           "<p>정답 없이 데이터의 구조를 찾는 걸 <b>비지도 학습</b>이라고 해요. 요청 종류를 나눠 보거나, 이상한 요청을 찾는 데 써요.</p>"
           + K.code_table([("KMeans(n_clusters=k, n_init=10)", "k개 중심을 찾아 가장 가까운 중심으로 묶어요. 시작점을 10번 바꿔 가장 좋은 것"), ("silhouette_score(Z, labels)", "묶음이 얼마나 잘 나뉘었나 (−1~1)")]), "s16"),
    K.step(17, "마지막 한 번: test로 최종 평가", "교차검증으로 고른 모델을 80건 전부로 다시 배우고, 아껴 둔 test로 한 번만 채점해요.",
           K.code_table([("max(cv_scores, key=...)", "교차검증 평균이 가장 높은 후보의 이름"), ("clone(candidates[best]).fit(dev...)", "같은 설정의 새 모델을 80건 전부로")]), "s17"),
]
practice3 = [("기준을 정할 때", "valid(또는 교차검증)로 정하고 test는 건드리지 않아요. 업무에서 FN, FP 비용을 먼저 물어요."),
             ("AUC는 높은데 F1이 낮다", "확률의 순서는 맞지만 기준이 안 맞는 거예요. ⑬처럼 기준을 조정해요."),
             ("묶음(군집)에 이름 붙이기", "K-Means는 숫자 번호만 줘요. 묶음마다 평균값을 보고 사람이 뜻을 붙여요. 크기 맞추기를 안 하면 큰 숫자 열로만 묶여요.")]
libs3 = [("⑬", "<code>predict_proba()</code>, <code>confusion_matrix().ravel()</code>", "확률과 기준"), ("⑭", "<code>roc_auc_score</code>", "확률의 순서 품질"), ("⑮", "<code>FunctionTransformer</code>, <code>np.log1p</code>", "특성 변환"),
         ("⑯", "<code>KMeans</code>, <code>silhouette_score</code>", "군집"), ("⑰", "<code>clone()</code>, <code>fit()</code>, <code>predict()</code>", "최종 학습과 평가")]
K.build(ROOT / "topics" / "model" / "ml-metrics-clustering.html", title="평가지표, 특성 설계, 군집", h1="평가지표, 특성 설계, 군집: 기준을 정하고 마지막에 한 번 채점하기", track="모델",
        crumb="모델 &gt; 머신러닝 기초 3/3쪽 &nbsp;|&nbsp; 이전: <a href=\"ml-trees-cv.html\" style=\"color:inherit\">트리와 교차검증</a>",
        one_line="분류 모델은 사실 <b>확률</b>을 내고, \"실패\"라고 부를 <b>기준은 오류 비용으로</b> 정해요. 모델과 기준을 다 고른 뒤에 <b>test로 딱 한 번</b> 채점해요.",
        why=why3, whole=whole3, steps=steps3, code_file="examples/ml/requests_model.py", practice=practice3, libs=libs3, libs_note="<code>cost_at</code>, <code>build_fe</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"ml-trees-cv.html\">트리와 앙상블, 교차검증</a>", "후보 모델과 교차검증"), ("<a href=\"dl-tensor-training.html\">딥러닝 기초</a>", "신경망으로 넘어가기")],
        sources=["다루는 범위: Day08 (2026-10-28) 평가지표, 특성 설계, K-Means, 최종 평가", DATA_SRC, "결과 값: <code>tools/pages/ml_basics.py</code>",
                 "공식 문서: <a href=\"https://scikit-learn.org/stable/modules/classification_threshold.html\" target=\"_blank\">Tuning the decision threshold</a>, <a href=\"https://scikit-learn.org/stable/modules/clustering.html#k-means\" target=\"_blank\">K-means</a>"])
print("ok")
