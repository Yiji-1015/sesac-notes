"""모델 > 언어 모델로 문제 풀기 > 문장 분류와 미세조정. examples/classify/issue_classifier.py 하나가 ①~⑧로 자란다.
NLP with Transformers 2장(텍스트 분류), 9장(레이블 부족 문제)을 합친 쪽.
실행: HF_HUB_OFFLINE=1 python tools/pages/text_cls.py   (CPU로 1분쯤)
e5 모델을 받아 두지 않은 PC에서는 아무 파일도 바꾸지 않고 끝낸다.
"""
import html
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "classify"
code = S.parse((EX / "issue_classifier.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="issue_classifier.py")  # noqa: E731
try:
    ns = run(8)
except OSError:
    print("e5 모델(intfloat/multilingual-e5-small)이 없어서 이 쪽은 다시 만들지 않아요. 기존 파일을 그대로 둬요.")
    sys.exit(0)

from vstyle import plt, C, setup  # noqa: E402

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "textcls"
IMG.mkdir(parents=True, exist_ok=True)
E, J = html.escape, K.josa
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, scikit-learn {sklearn.__version__}, intfloat/multilingual-e5-small, CPU"
LABELS, part, y = ns["LABELS"], ns["part"], ns["y"]
COL = {"버그": C["pink"], "기능 요청": C["purple"], "질문": C["mint"]}


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/textcls/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
la, lt = ns["log_all"], ns["log_top"]
best_f1 = lt[-1]["test_f1"]
peak = max(r["test_f1"] for r in lt)   # 위쪽 4층 미세조정이 가장 높았을 때
# ---------------- 왜
tr = part["train"]
same_topic = [tr[(tr["label"] == lab) & tr["text"].str.contains("결제")].iloc[0] for lab in LABELS]
te = part["test"]
new_tpl = [te[te["label"] == lab].iloc[0] for lab in LABELS]
outs["data"] = S.table("issues.csv에서 몇 줄 (위: train, 아래: test)", ["split", "text", "label"],
                       [[r["split"], E(r["text"]), f"<b>{r['label']}</b>"] for r in same_topic + new_tpl],
                       ["위 세 줄은 모두 \"결제\" 이야기인데 종류가 달라요. 어떤 단어가 나왔는지가 아니라 <b>무엇을 원하는 말인지</b>를 봐야 해요.",
                        "아래 세 줄은 test에서 가져왔어요. test와 valid에는 train에 없던 말투(문장 틀)만 넣었어요. 실제 문의도 사람마다 쓰는 말투가 다르니까요."])
S.claim(ns["nb_f1"] < ns["frozen_f1"] < best_f1 and la[-1]["test_f1"] < ns["frozen_f1"], "왜: 단어 세기 < e5 고정 < 위쪽만 미세조정, 전체 미세조정은 고정보다 낮다")
outs["why"] = S.table("같은 train 190개, 같은 test 120개로 비교 (macro F1)", ["방법", "test macro F1", "단계"],
                      [["단어를 세는 모델 (TF-IDF + Naive Bayes)", f"<span class=\"bad\">{f3(ns['nb_f1'])}</span>", "②"], ["e5를 고정하고 벡터 위에 로지스틱 회귀", f3(ns["frozen_f1"]), "③④"],
                       ["e5 12층 전체를 미세조정", f"<span class=\"bad\">{f3(la[-1]['test_f1'])}</span>", "⑤"], ["e5 위쪽 4층만 미세조정", f"<b>{f3(best_f1)}</b>", "⑤"]],
                      [f"단어를 세는 방법은 test에서 처음 보는 단어가 많아서 무너져요. 미리 학습된 e5의 벡터를 쓰면 처음 보는 말투도 뜻이 비슷하면 가깝게 놓여서 {f3(ns['frozen_f1'])}까지 올라가요.",
                       "그런데 e5를 통째로 더 학습시키면(미세조정) 오히려 떨어지고, 위쪽 몇 층만 고치면 가장 좋아요. 이 쪽은 이 차이가 왜 생기는지, 라벨이 더 적으면 어떻게 하는지를 다뤄요."])

# ---------------- ①
cnt = part["train"]["label"].value_counts()
outs["s1"] = S.box("① 데이터", [*(f"{s} → {len(d)}개" + (f" ({', '.join(f'{lab} {int((d['label'] == lab).sum())}' for lab in LABELS)})" if s != "unlabeled" else " (라벨은 있지만 쓰지 않아요)") for s, d in part.items()),
                                 (f"train 문장 틀 {part['train']['template'].nunique()}개, test 문장 틀 {part['test']['template'].nunique()}개, 겹치는 틀 {len(set(part['train']['template']) & set(part['test']['template']))}개", 1, "newf"),
                                 (f"y['train'][:8] → {y['train'][:8].tolist()}", 2)],
                   [f"test에만 있는 문장 틀로 채점해요. 외운 말투가 아니라 처음 보는 말투를 맞히는지 재는 거예요.", f"종류 이름을 번호로 바꿨어요 ({', '.join(f'{i} {lab}' for i, lab in enumerate(LABELS))}). train은 버그 {cnt['버그']}개, 질문 {cnt['질문']}개로 고르지 않아요. 그래서 정확도 대신, 종류마다 F1을 구해 똑같은 무게로 평균 낸 <b>macro F1</b>로 재요."])
S.claim(len(set(part["train"]["template"]) & set(part["test"]["template"])) == 0, "① train과 test의 문장 틀이 겹치지 않는다")

# ---------------- ②
tw = [w for t in part["test"]["text"] for w in t.split()]
unseen_top = Counter(w for w in tw if w not in ns["train_words"]).most_common(8)
outs["s2"] = S.box("② 단어 세기 기준선", [(f"test macro F1 → {ns['nb_f1']:.3f}", 1, "bad"), (f"test의 단어 {len(tw)}개 중 train에 없던 단어 → {ns['unseen']:.0%}", 2, "newf"),
                                       f"train에 없던 단어 (많이 나온 순) → {[w for w, _ in unseen_top]}"],
                   ["단어 하나하나에 점수를 매겨 세는 방법이라, 처음 보는 단어는 아무 정보가 없어요.", f"test 단어의 {ns['unseen']:.0%}가 train에 한 번도 없었어요. \"어케\", \"먹통이에요\"처럼 띄어쓰기 단위 단어는 어미, 조사가 붙어 모양이 계속 바뀌어서 더 그래요."])
S.claim(all(w not in ns["train_words"] for w in ("어케", "먹통이에요")), "② 예로 든 '어케', '먹통이에요'는 train에 없던 단어")
nc = run(2, [('nb = bag_model().fit', 'nb = bag_model("char_wb", (2, 4)).fit')])
S.claim(ns["nb_f1"] < nc["nb_f1"] < ns["frozen_f1"], "글자 조각으로 세면 나아지지만 e5 고정보다는 낮다")
outs["t2"] = S.table("단어 대신 글자 조각(2~4글자)으로 세면", ["세는 단위", "test macro F1"], [["띄어쓰기 단어", f3(ns["nb_f1"])], ["글자 조각 2~4개", f"<b>{f3(nc['nb_f1'])}</b>"]],
                     [f"\"먹통이에요\"를 \"먹통\", \"통이\", \"이에요\" 같은 조각으로 쪼개 세면, 어미가 달라도 같은 조각이 잡혀서 {ns['nb_f1']:.2f}에서 {nc['nb_f1']:.2f}까지 올라요.",
                      "모델이 쓰는 Tokenizer가 부분단어(조각)로 나누는 것도 같은 이유예요. 그래도 \"뜻이 비슷한 다른 단어\"는 여전히 몰라요."])

# ---------------- ③
tok = ns["tok"]
bt = tok(["query: " + t for t in part["test"]["text"]], padding=True)
n_unk = sum(i == tok.unk_token_id for row in bt["input_ids"] for i in row)
ex = part["test"]["text"].iloc[0]
S.claim(n_unk == 0, "③ e5 Tokenizer에서는 모르는 토큰이 0개")
outs["s3"] = S.box("③ 문장 벡터", [(f"vec['train'].shape → {ns['vec']['train'].shape}", 1), f"길이 → {np.linalg.norm(ns['vec']['train'][:3], axis=1).round(3).tolist()}",
                                   (f"test 120문장을 e5 Tokenizer로 나누면 모르는 토큰(<unk>) → {n_unk}개", 2, "newf"), f"예: \"{ex}\" → {tok.convert_ids_to_tokens(tok('query: ' + ex)['input_ids'])}"],
                   ["문장마다 384칸 벡터 하나예요. 길이를 1로 맞춰서, 두 벡터의 내적이 곧 코사인 유사도예요.",
                    "부분단어 Tokenizer는 처음 보는 단어도 아는 조각으로 쪼개서 \"모르는 토큰\"이 생기지 않아요. 그리고 각 조각의 벡터는 수많은 글로 미리 학습돼 있어서, 처음 보는 말투도 뜻이 비슷하면 비슷한 벡터가 돼요."])

# ---------------- ④
V = np.vstack([ns["vec"]["train"], ns["vec"]["test"]])
P2 = PCA(n_components=2, random_state=0).fit(ns["vec"]["train"]).transform(V)
ntr = len(ns["vec"]["train"])
lab_all = np.concatenate([y["train"], y["test"]])
fig, ax = plt.subplots(figsize=(7.4, 4.6))
for c, lab in enumerate(LABELS):
    m_tr, m_te = (lab_all[:ntr] == c), (lab_all[ntr:] == c)
    ax.scatter(P2[:ntr][m_tr, 0], P2[:ntr][m_tr, 1], s=22, color=COL[lab], alpha=.75, label=f"{lab} (train)")
    ax.scatter(P2[ntr:][m_te, 0], P2[ntr:][m_te, 1], s=46, facecolors="none", edgecolors=COL[lab], lw=1.4, label=f"{lab} (test)")
ax.set(xlabel="주성분 1", ylabel="주성분 2", title="e5 문장 벡터 384칸을 2칸으로 줄여 본 모습")
ax.legend(fontsize=8, ncol=2)
outs["f_pca"] = chart("train(채운 점)과 test(빈 원)의 e5 문장 벡터", fig_save(fig, "pca_vectors"), "PCA로 384칸 → 2칸",
                      ["라벨을 하나도 쓰지 않고 만든 벡터인데도 종류별로 대체로 모여 있어요. 처음 보는 말투인 test(빈 원)도 같은 종류 근처에 떨어져요. 그래서 이 벡터 위에 단순한 분류기만 얹어도 잘 돼요.",
                       "PCA 두 칸은 384칸의 일부만 보여 줘서, 겹쳐 보이는 점도 384칸에서는 갈릴 수 있어요 (<a href=\"ml-svm-lda-pca.html\">PCA</a>)."])
from sklearn.metrics import classification_report  # noqa: E402
rep = classification_report(y["test"], ns["frozen"].predict(ns["vec"]["test"]), labels=[0, 1, 2], target_names=LABELS, output_dict=True)
outs["s4"] = S.box("④ e5 고정 + 로지스틱 회귀", [(f"test macro F1 → {ns['frozen_f1']:.3f}", 1, "newf"), *(f"{lab}: precision {rep[lab]['precision']:.2f}, recall {rep[lab]['recall']:.2f}" for lab in LABELS)],
                   [f"②({ns['nb_f1']:.2f})보다 크게 올랐어요. 학습한 건 384 × 3 가중치와 절편 3개뿐이에요. 문장 벡터는 e5가 이미 만들어 둔 걸 그대로 썼어요."])

# ---------------- ⑤
tp = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)  # noqa: E731
n_all, n_top = tp(ns["tuned_all"]), tp(ns["tuned_top"])
n_body = sum(p.numel() for p in ns["tuned_all"].body.parameters())
S.claim(max(r["test_f1"] for r in la) - min(r["test_f1"] for r in la) < 0.02, "⑤ 12층 전부는 첫 epoch부터 test F1이 거의 그대로")
S.claim(la[-1]["train_loss"] < 0.05 and max(r["test_f1"] for r in la) < ns["frozen_f1"] and best_f1 > ns["frozen_f1"], "⑤ 전체 미세조정은 train 손실이 0 가까이 가도 test가 고정보다 낮고, 위쪽 4층은 고정보다 높다")
outs["s5"] = S.table("⑤ 미세조정 두 가지 (epoch마다 마지막 묶음의 train 손실, test macro F1)", ["epoch", "12층 전부: train 손실", "12층 전부: test F1", "위쪽 4층: train 손실", "위쪽 4층: test F1"],
                     [[a_["epoch"], f3(a_["train_loss"]), f3(a_["test_f1"]), f3(b_["train_loss"]), f"<b>{f3(b_['test_f1'])}</b>"] for a_, b_ in zip(la, lt)],
                     [f"고치는 가중치: 12층 전부 {n_all:,}개, 위쪽 4층 {n_top:,}개 (e5 몸통 전체는 {n_body:,}개. 대부분은 25만 어휘의 Embedding 표라서 두 경우 모두 고정했어요).",
                      f"12층을 다 고치면 첫 epoch부터 test F1이 {la[0]['test_f1']:.2f} 근처에 멈춰요. train 손실은 {la[-1]['train_loss']:.3f}까지 내려가는데요. train 190개는 종류마다 문장 틀 10개뿐이라, 모델 전체가 그 말투를 외워 버렸어요(과적합).",
                      f"아래쪽 8층은 그대로 두고 위쪽 4층과 분류 층만 고치면, 미리 배운 일반적인 언어 지식은 지키면서 이 과제에 맞게 다듬어져요. 마지막 epoch에서 {best_f1:.3f}, 가장 높을 때 {peak:.3f}{J(f'{peak:.3f}', '이에요', '예요')}. 데이터가 적을수록 적게 고치는 게 안전해요."])
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.7))
ep = [r["epoch"] for r in la]
for lg, nm, col in ((la, "12층 전부", C["pink"]), (lt, "위쪽 4층", C["purple"])):
    a.plot(ep, [r["train_loss"] for r in lg], marker="o", color=col, lw=2, label=nm)
    b.plot(ep, [r["test_f1"] for r in lg], marker="o", color=col, lw=2, label=nm)
b.axhline(ns["frozen_f1"], color=C["gray"], ls="--", lw=1.2, label="④ e5 고정")
a.set(xlabel="epoch", ylabel="train 손실 (마지막 묶음)", title="train에서는 둘 다 잘 배워요")
b.set(xlabel="epoch", ylabel="test macro F1", title="처음 보는 말투에서는 갈려요", ylim=(0.3, 1.0))
a.legend(fontsize=9)
b.legend(fontsize=9, loc="lower right")
outs["f_ft"] = chart("미세조정 두 가지의 학습 기록", fig_save(fig, "finetune_curves"), "", ["왼쪽만 보면 12층 전부가 더 잘 배우는 것 같지만, 오른쪽의 처음 보는 말투에서는 회색 점선(고정)보다도 낮아요. 학습이 잘 되는지는 train이 아니라 처음 보는 데이터로 봐요."])

# ---------------- ⑥
cm, wrong, byt = ns["cm"], ns["wrong"], ns["by_template"]
bad_t = byt[byt < 1]
S.claim(len(bad_t) == 1 and (wrong["label"] == "기능 요청").all() and (wrong["pred"] == "질문").all(), "⑥ 틀린 건 모두 한 문장 틀의 기능 요청이고, 질문으로 답했다")
w0 = wrong.iloc[0]
trd = part["train"]
q_end = (trd[trd["label"] == "질문"]["text"].str.endswith("?")).mean()          # train 질문 중 물음표로 끝나는 비율
f_end_t = trd[(trd["label"] == "기능 요청") & trd["text"].str.endswith("?")]["template"].nunique()   # 물음표로 끝나는 기능 요청 틀 수
S.claim(wrong["conf"].min() > 0.8, "⑥ 틀린 문장도 확신이 높다")
S.claim(q_end == 1 and f_end_t <= 1, "⑥ train 질문은 모두 물음표로 끝나고, 물음표로 끝나는 기능 요청 틀은 많아야 하나")
outs["s6"] = S.box("⑥ 위쪽 4층 미세조정의 test 결과", ["혼동 행렬 (행: 실제, 열: 예측) 순서: " + ", ".join(LABELS), *(f"  {lab:<6} {cm[i].tolist()}" for i, lab in enumerate(LABELS)),
                                                  (f"틀린 {len(wrong)}개의 문장 틀 → {sorted(wrong['template'].unique())}, 그 틀의 정답률 {bad_t.iloc[0]:.0%}", 1, "bad"),
                                                  (f"예: \"{w0['text']}\" → 정답 {w0['label']}, 예측 {w0['pred']} (확신 {w0['conf']:.2f})", 2)],
                   [f"틀린 {len(wrong)}개가 모두 한 문장 틀에서 나왔고, 나머지 test 틀은 다 맞혔어요. 숫자 하나(F1)만 보면 안 보이는 사실이에요.",
                    f"\"~할 계획은 없나요?\"처럼 <b>질문 꼴로 쓴 기능 요청</b>이에요. train의 질문은 {q_end:.0%}가 물음표로 끝나고, 물음표로 끝나는 기능 요청은 문장 틀 {f_end_t}개뿐이에요. 그래서 모델이 \"물음표 꼴이면 질문\"이라는 지름길을 배웠어요. 확신도 높아서, 확률만 보고는 걸러지지 않아요. 이런 문장을 train에 몇 개 넣거나, 종류의 정의를 다시 정하는 게 해결책이에요."])

# ---------------- ⑦
cv = ns["curve"]
zs = ns["zero_shot_f1"]
e8 = cv["e5 + 로지스틱"].iloc[2]
S.claim(cv["e5 + 로지스틱"].iloc[2] > cv["단어 세기"].max(), "⑦ e5 8개가 단어 세기 40개보다 높다")
S.claim(cv["e5 + 로지스틱"].iloc[2] > cv["단어 세기"].iloc[2] + 0.1 and ns["zero_shot_f1"] > 1 / 3 + 0.2, "⑦ 종류마다 8개면 e5가 단어 세기보다 크게 낫고, 라벨 0개도 우연보다 낫다")
outs["s7"] = S.table("⑦ 종류마다 라벨이 n개뿐이라면 (무작위로 5번 뽑은 평균, test macro F1)", ["n (종류마다)", "단어 세기 (글자 조각)", "가장 비슷한 문장 따라가기", "e5 + 로지스틱"],
                     [[f"<b>0</b> (설명문만)", "-", "-", f"{ns['zero_shot_f1']:.3f} (설명문과 가장 가까운 종류)"]] + [[n, f3(r["단어 세기"]), f3(r["가장 비슷한 문장"]), f"<b>{f3(r['e5 + 로지스틱'])}</b>"] for n, r in cv.iterrows()],
                     [f"라벨이 종류마다 8개뿐이어도 e5 + 로지스틱은 {e8:.2f}{J(f'{e8:.2f}', '이에요', '예요')}. 단어 세기({cv['단어 세기'].iloc[2]:.2f})가 40개로도 못 가는 점수예요.",
                      f"\"가장 비슷한 라벨 문장을 따라가기\"(벡터를 찾아보기 표로 쓰기)는 라벨이 적으면 흔들려요. 한두 문장의 말투에 크게 좌우되기 때문이에요.",
                      f"라벨이 하나도 없어도, 종류마다 설명문 한 줄을 써서 그 벡터와 가장 가까운 종류로 답하면 {zs:.2f}{J(f'{zs:.2f}', '이에요', '예요')}. 세 종류를 무작위로 찍으면 0.33이에요."])
fig, ax = plt.subplots(figsize=(7.6, 3.9))
for nm, col in (("단어 세기", C["gray"]), ("가장 비슷한 문장", C["orange"]), ("e5 + 로지스틱", C["purple"])):
    ax.plot(cv.index, cv[nm], marker="o", lw=2.2, color=col, label=nm)
ax.axhline(ns["zero_shot_f1"], color=C["mint"], ls="--", lw=1.3, label="라벨 0개 (설명문)")
ax.set_xscale("log", base=2)
ax.set_xticks(list(cv.index), [str(n) for n in cv.index])
ax.minorticks_off()
ax.set(xlabel="종류마다 라벨 수 n", ylabel="test macro F1", title="라벨이 적을수록 미리 배운 벡터가 빛나요")
ax.legend(fontsize=9)
outs["f_curve"] = chart("라벨 수에 따른 점수", fig_save(fig, "label_curve"), "", ["보라 선이 거의 모든 n에서 위에 있어요. 라벨을 모으기 어려울 때는 먼저 미리 학습된 벡터 위에 단순한 분류기를 얹어 보는 게 출발점이에요."])

# ---------------- ⑧
sr = ns["self_runs"]
mb, ma = sr["라벨만"].mean(), sr["가짜 라벨 더해서"].mean()
sb, sa = sr["라벨만"].std(ddof=0), sr["가짜 라벨 더해서"].std(ddof=0)
S.claim(ma > mb and sa < sb and sr["가짜 라벨 정답률"].min() > 0.9, "⑧ 가짜 라벨을 더하면 평균이 오르고 흔들림이 줄며, 가짜 라벨은 90% 넘게 맞는다")
outs["s8"] = S.table("⑧ 종류마다 라벨 8개 + 라벨 없는 문장 410개 (확률 0.5 이상만 가짜 라벨로)", ["뽑기", "가짜 라벨 수", "가짜 라벨 정답률", "라벨만", "가짜 라벨 더해서"],
                     [[int(r["seed"]), int(r["가짜 라벨 수"]), f"{r['가짜 라벨 정답률']:.0%}", f3(r["라벨만"]), f3(r["가짜 라벨 더해서"])] for _, r in sr.iterrows()]
                     + [["<b>평균 (표준편차)</b>", "", "", f"{mb:.3f} ({sb:.3f})", f"<b>{ma:.3f} ({sa:.3f})</b>"]],
                     [f"라벨 24개로 배운 분류기(선생)가 라벨 없는 문장에 라벨을 붙이고, 그걸 더해 다시 배워요(학생). 가짜 라벨은 모든 뽑기에서 {np.floor(sr['가짜 라벨 정답률'].min() * 100):.0f}% 넘게 맞았어요.",
                      f"평균은 {mb:.3f}에서 {ma:.3f}까지 조금 오르고, 뽑기마다의 흔들림(표준편차)은 {sb:.3f}에서 {sa:.3f}까지 크게 줄었어요. 운 나쁘게 라벨이 치우친 뽑기를 라벨 없는 문장이 메워 줬어요. 대신 이미 잘된 뽑기는 조금 내려가기도 했어요.",
                      "다만 이 라벨 없는 문장들은 train과 같은 말투뿐이라, ⑥에서 본 \"질문 꼴 기능 요청\" 같은 처음 보는 말투의 약점은 그대로예요. 새 말투를 배우려면 라벨 없는 문장에도 그런 말투가 들어 있어야 해요."])

flow = [[1, "데이터", "190 / 60 / 120", "test는 새 말투"], [2, "단어 세기", "TF-IDF, Naive Bayes", f"F1 {ns['nb_f1']:.2f}"], [3, "문장 벡터", "e5 평균 풀링", "모르는 토큰 0"], [4, "고정 + 분류기", "로지스틱", f"F1 {ns['frozen_f1']:.2f}"],
        [5, "미세조정", "12층 vs 위쪽 4층", f"F1 {la[-1]['test_f1']:.2f} vs {best_f1:.2f}"], [6, "오류 분석", "혼동 행렬", "질문 꼴 요청"], [7, "라벨이 적으면", "n개씩, 설명문", "벡터가 이김"], [8, "라벨 없는 문장", "가짜 라벨", "흔들림 감소"]]
S.save(ROOT / "topics" / "model" / "text-classification-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>서비스에 문의가 하루 수백 건 들어온다고 해 봐요. 버그는 개발팀, 기능 요청은 기획팀, 질문은 고객 지원팀이 봐야 해요. 사람이 하나씩 읽고 나누기에는 많아서, 글을 읽고 종류를 붙이는 모델을 만들어요. 이런 일을 <b>문장 분류</b>라고 해요. 스팸 거르기, 감정 분석, 의도 파악이 모두 같은 틀이에요.</p>
<p>이 쪽의 예제는 가상의 문의 780개예요. 종류는 셋이고, 같은 기능 이름이 세 종류에 모두 나와요.</p>
{K.out("data")}
<p>네 가지 방법으로 같은 데이터를 풀어 봤어요.</p>
{K.out("why")}"""
whole = """<p>데이터 → 단어 세기 기준선 → e5로 문장 벡터 → 고정한 벡터 위에 분류기 → e5까지 미세조정 → 틀린 것 살피기 → 라벨이 적을 때 → 라벨 없는 문장 쓰기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "데이터 읽고 나누기", "문의를 train, valid, test, 라벨 없는 문장으로 나누고, 종류 이름을 번호로 바꿔요.",
           "<p><b>왜 문장 틀이 겹치지 않게 나누나?</b> 같은 말투가 train과 test에 다 있으면, 모델이 말투를 외워도 점수가 높게 나와요. 실제로 쓸 때는 처음 보는 사람의 처음 보는 말투가 들어와요. 그래서 test에는 train에 없던 문장 틀만 넣었어요.</p>"
           + K.code_table([("df[df[\"split\"] == s]", "split 열로 행 고르기"), ("d[\"label\"].map(LABELS.index)", "종류 이름을 목록의 위치 번호로"),
                           ("f1_score(y_true, pred, average=\"macro\")", "종류마다 F1을 구해 단순 평균. 적은 종류도 같은 무게")]), "s1"),
    K.step(2, "기준선: 단어를 세는 모델", "문장마다 단어가 몇 번, 얼마나 드물게 나왔는지 세고(TF-IDF), 종류마다 단어가 나올 확률로 판단해요(Naive Bayes).",
           "<p><b>왜 기준선부터?</b> 무거운 모델이 얼마나 나은지 알려면 비교할 바닥이 있어야 해요. 단어 세기는 빠르고 단순해서 첫 비교 대상으로 좋아요.</p>"
           "<p><b>TF-IDF</b>는 \"이 문장에 자주 나오고(TF) 다른 문장에는 드문(IDF) 단어\"에 큰 점수를 주는 방법이에요. <b>Naive Bayes</b>는 \"버그 문의에는 '오류'가 자주 나온다\"처럼 종류마다 단어의 출현 정도를 배운 뒤, 새 문장의 단어들로 어느 종류일 확률이 높은지 계산해요. 단어끼리 서로 상관없다고 가정해서(naive) 계산이 아주 빨라요.</p>"
           + K.code_table([("make_pipeline(TfidfVectorizer(...), MultinomialNB())", "글 → TF-IDF 숫자 → 분류. <code>fit</code> 한 번에 둘 다 배워요"), ("analyzer=\"word\"", "띄어쓰기 단어 단위로 세기"),
                           ("{w for t in ... for w in t.split()}", "train에 나온 단어 집합")])
           + K.try_("단어 대신 글자 조각으로 세면?", "t2"), "s2"),
    K.step(3, "e5로 문장 벡터 만들기 (특성 추출)", "미리 학습된 e5에 문장을 넣고, 토큰 벡터들을 평균 내 문장 하나의 벡터로 만들어요. e5는 고치지 않아요.",
           "<p><a href=\"llm-principles.html\">Transformer 구조</a>에서 본 것처럼, e5는 토큰마다 문맥이 섞인 384칸 벡터를 내요. 이걸 평균 내면(평균 풀링) 문장 하나의 벡터가 돼요. 채운 자리(padding)는 평균에서 빼요. 첫 토큰의 벡터 하나만 쓰는 방법도 있는데, e5는 평균으로 쓰도록 학습됐어요.</p>"
           "<p><b>왜 미리 학습된 모델을 쓰나?</b> e5는 수많은 글로 \"뜻이 비슷한 문장은 가까운 벡터\"가 되도록 학습돼 있어요. 우리 데이터 190개로는 배울 수 없는 언어 지식이에요. 이걸 빌려 쓰는 게 <b>전이학습</b>이에요(<a href=\"transfer-learning.html\">전이학습</a>은 그림으로 같은 일을 해요). 여기서는 e5를 고정하고 벡터만 뽑아 쓰는 <b>특성 추출</b> 방식이에요.</p>"
           + K.code_table([("tok([\"query: \" + t ...], padding=True, return_tensors=\"pt\")", "e5는 문장 앞에 <code>query: </code>를 붙여 쓰도록 학습됐어요"), ("mean_pool(hidden, mask)", "채운 자리를 빼고 토큰 벡터 평균"),
                           ("torch.no_grad()", "기울기를 계산하지 않아요. 고치지 않으니까요"), ("F.normalize(..., dim=-1)", "벡터 길이를 1로")]), "s3"),
    K.step(4, "고정한 벡터 위에 분류기 얹기", "③의 문장 벡터를 입력으로 로지스틱 회귀를 학습해요.",
           "<p>e5가 문장을 \"뜻이 잘 드러나는 384개 숫자\"로 바꿔 줬으니, 그 위에서는 <a href=\"ml-regression-classification.html\">로지스틱 회귀</a> 같은 단순한 모델로 충분해요. 학습은 몇 초면 끝나고, GPU도 필요 없어요.</p>"
           + K.out("f_pca")
           + K.code_table([("LogisticRegression(max_iter=2000, C=10)", "세 종류를 가르는 선형 분류기. 벡터 길이가 1이라 값이 작아서 규제를 약하게(C=10)")]), "s4"),
    K.step(5, "미세조정: 모델 안의 가중치까지 고치기", "e5 위에 분류 층을 붙이고, 정답으로 손실을 계산해 e5의 층까지 함께 고쳐요. 12층 전부를 고칠 때와 위쪽 4층만 고칠 때를 비교해요.",
           "<p><b>미세조정</b>(fine-tuning)은 미리 학습된 모델을 출발점으로, 우리 과제의 정답으로 조금 더 학습시키는 거예요. 특성 추출은 e5를 그대로 두지만, 미세조정은 e5의 가중치도 이 과제에 맞게 바뀌어요. 데이터가 넉넉하면 보통 미세조정이 더 좋아요.</p>"
           "<p><b>왜 층 수를 고르나?</b> 고치는 가중치가 많을수록 우리 데이터에 더 맞출 수 있지만, 데이터가 적으면 그만큼 외우기도 쉬워요. 아래쪽 층은 단어 조각, 문법 같은 일반적인 지식을, 위쪽 층은 과제에 가까운 정보를 담는다고 알려져 있어서, 데이터가 적을 때는 위쪽만 고치는 방법을 자주 써요. 새로 붙인 분류 층은 무작위에서 시작하니 학습률을 더 크게(<code>head_lr</code>) 줘요.</p>"
           + K.out("f_ft")
           + K.code_table([("class Classifier(torch.nn.Module)", "e5 몸통(<code>self.body</code>) + 384 → 3 분류 층(<code>self.head</code>)"), ("clf.body.requires_grad_(False)", "몸통 전체를 고정 (Embedding 표 포함)"),
                           ("clf.body.encoder.layer[12 - train_layers:]", "위에서부터 train_layers개 층"), ("torch.optim.AdamW([{...}, {...}])", "몸통과 분류 층에 다른 학습률. AdamW는 가중치 감쇠를 따로 처리하는 Adam"),
                           ("torch.randperm(len(X)).split(16)", "순서를 섞어 16개씩 묶기"), ("F.cross_entropy(logits, Y[idx])", "세 종류의 점수와 정답으로 손실")]), "s5"),
    K.step(6, "오류 분석: 무엇을 무엇으로 틀렸나", "가장 좋은 모델(위쪽 4층)의 test 예측으로 혼동 행렬을 만들고, 틀린 문장과 문장 틀별 정답률을 봐요.",
           "<p><b>왜 점수 하나로 끝내지 않나?</b> F1 0.9라도 틀린 10%가 고르게 퍼져 있는지, 한 가지 이유로 몰려 있는지에 따라 할 일이 달라요. 몰려 있으면 그 원인 하나만 고치면 되고, 그게 데이터의 어떤 빈틈을 알려 주기도 해요.</p>"
           + K.code_table([("tuned_top(...).softmax(-1)", "세 점수를 확률로"), ("confusion_matrix(y, pred, labels=[0, 1, 2])", "실제(행) × 예측(열) 개수 표"),
                           ("...groupby(\"template\")[\"ok\"].mean()", "문장 틀마다 맞힌 비율")]), "s6"),
    K.step(7, "라벨이 적을 때", "종류마다 라벨을 n개만 남기고 세 방법을 비교해요. 라벨이 하나도 없을 때는 종류마다 설명문 한 줄로 답해요.",
           "<p><b>왜?</b> 실제로는 라벨을 다는 게 가장 비싸요. 사람이 하나씩 읽어야 하니까요. 그래서 \"라벨이 몇 개면 쓸 만한가\", \"라벨 없이 시작할 수는 없나\"를 먼저 따져 봐요.</p>"
           "<p><b>가장 비슷한 문장 따라가기</b>는 라벨이 있는 문장들의 벡터를 찾아보기 표로 두고, 새 문장과 가장 가까운 문장의 라벨을 그대로 쓰는 방법이에요(KNN, k=1). 문장이 수백만 개면 FAISS 같은 벡터 검색 라이브러리로 찾아요(<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>). <b>설명문으로 답하기</b>는 라벨 대신 종류의 뜻을 한 줄로 써서 벡터로 만들고, 그 벡터와 가장 가까운 종류로 답해요. e5의 검색 학습 방식에 맞춰 문의는 <code>query:</code>, 설명문은 <code>passage:</code>를 붙였어요.</p>"
           + K.out("f_curve")
           + K.code_table([("rng.choice(np.flatnonzero(y == c), n, replace=False)", "종류 c인 행 중 n개를 겹치지 않게 뽑기"), ("(vec[\"test\"] @ vec[\"train\"][idx].T).argmax(1)", "test 문장마다 코사인 유사도가 가장 큰 라벨 문장"),
                           ("bag_model(\"char_wb\", (2, 4))", "②의 🔁에서 본 글자 조각 버전 (라벨이 적을 때 단어 단위보다 나아서)"), ("(vec[\"test\"] @ label_vec.T).argmax(1)", "설명문 벡터 셋 중 가장 가까운 것")]), "s7"),
    K.step(8, "라벨 없는 문장 쓰기: 스스로 라벨 붙여 다시 배우기", "라벨 몇 개로 배운 모델이 라벨 없는 문장에 가짜 라벨을 붙이고, 확신이 높은 것만 더해 다시 배워요. 무작위 뽑기 5번으로 비교해요.",
           "<p>라벨 없는 문장은 거의 공짜로 모여요. 이걸 쓰는 가장 단순한 방법이 <b>자기 학습</b>(self-training)이에요. 선생 모델이 붙인 가짜 라벨(pseudo-label) 중 확신이 높은 것만 골라 학생 모델이 배워요. 이걸 몇 번 되풀이하기도 하고, 확신 대신 \"여러 번 물었을 때 답이 얼마나 일정한가\"로 고르기도 해요(UST).</p>"
           + K.code_table([("teacher.predict_proba(vec[\"unlabeled\"])", "라벨 없는 문장마다 세 종류의 확률"), ("p.max(1) >= threshold", "가장 큰 확률이 기준 이상인 문장만"),
                           ("np.vstack([...]), np.concatenate([...])", "진짜 라벨 문장과 가짜 라벨 문장을 이어 붙이기")]), "s8"),
]
practice = [("데이터가 많고 GPU가 있다", "전체 미세조정이 보통 가장 좋아요. 학습률은 2e-5~5e-5, epoch은 2~4번이 흔한 출발점이에요. valid 점수를 보며 멈추고, 가장 좋던 때의 가중치를 저장해요(<code>save_pretrained</code>)."),
            ("데이터가 적다", "먼저 ④처럼 고정한 벡터 + 단순한 분류기로 기준을 잡고, 미세조정은 위쪽 몇 층만 고쳐 봐요(⑤). 점수는 반드시 처음 보는 데이터로 확인해요."),
            ("종류가 불균형하다", "정확도 대신 macro F1, 종류별 recall을 봐요. 적은 종류에 무게를 주거나(<code>class_weight</code>) 적은 종류의 데이터를 더 모아요."),
            ("라벨이 거의 없다", "설명문과의 유사도(⑦)나, LLM에게 종류 설명과 예시 몇 개를 프롬프트로 주고 바로 분류시키는 방법(퓨샷)으로 시작해요. 그 결과를 사람이 고쳐 라벨로 모으면 빨라요."),
            ("라벨 없는 글이 많다", "그 글로 먼저 \"가린 토큰 맞히기\"를 더 학습해서 모델을 우리 분야 말투에 익숙하게 만든 뒤(도메인 적응) 분류를 미세조정하면 나아져요. 자기 학습(⑧)도 같이 써요."),
            ("데이터 늘리기", "문장을 다른 언어로 번역했다 되돌리기(back-translation), 비슷한 말로 바꾸기, 단어 일부 지우기로 라벨 있는 문장을 늘려요. 바뀐 문장도 같은 라벨이 맞는지 꼭 확인해요."),
            ("한 문장에 종류가 여럿", "\"버그이면서 기능 요청\"처럼 라벨이 여러 개일 수 있으면, 종류마다 \"맞다/아니다\"를 따로 맞히는 다중 라벨 분류로 바꿔요 (softmax 대신 sigmoid)."),
            ("결과만 빨리 필요하다", "<code>pipeline(\"text-classification\", model=...)</code> 한 줄로 이미 미세조정된 공개 모델을 쓸 수 있어요. 우리 종류와 같은지는 확인해야 해요.")]
libs = [("①", "<code>pd.read_csv</code>, <code>f1_score(average=\"macro\")</code>", "데이터, 지표"), ("②", "<code>TfidfVectorizer</code>, <code>MultinomialNB</code>, <code>make_pipeline</code>", "단어 세기 (scikit-learn)"),
        ("③", "<code>AutoTokenizer</code>, <code>AutoModel</code>, <code>last_hidden_state</code>, <code>F.normalize</code>", "문장 벡터 (transformers, torch)"), ("④⑦⑧", "<code>LogisticRegression</code>, <code>predict_proba</code>", "벡터 위 분류기"),
        ("⑤", "<code>requires_grad_</code>, <code>torch.optim.AdamW</code>, <code>F.cross_entropy</code>", "미세조정 (torch)"), ("⑥", "<code>confusion_matrix</code>, <code>groupby</code>", "오류 분석")]
K.build(ROOT / "topics" / "model" / "text-classification.html", title="문장 분류와 미세조정", h1="문장 분류: 고정한 벡터, 미세조정, 라벨이 적을 때", track="모델", crumb="모델 &gt; 언어 모델로 문제 풀기",
        one_line="문장을 미리 학습된 모델(e5)로 <b>벡터</b>로 바꾸면, 그 위에 단순한 분류기만 얹어도 처음 보는 말투까지 잘 나눠요. 모델까지 고치는 <b>미세조정</b>은 데이터가 적으면 외우기 쉬워서, 고칠 범위를 골라야 해요. 라벨이 적으면 벡터와 라벨 없는 문장을 최대한 활용해요.",
        why=why, whole=whole, steps=steps, code_file="examples/classify/issue_classifier.py", practice=practice, libs=libs,
        libs_note="<code>macro_f1</code>, <code>bag_model</code>, <code>encode</code>, <code>mean_pool</code>, <code>embed</code>, <code>fine_tune</code>, <code>label_curve</code>, <code>self_train</code>은 이 파일 안에서 정의한 함수, <code>Classifier</code>는 클래스예요.",
        next_rows=[("<a href=\"token-classification.html\">토큰 분류와 질문 답변</a>", "같은 몸통, 토큰마다 라벨"), ("<a href=\"model-compression.html\">모델 가볍게</a>", "이 분류기를 작고 빠르게"), ("<a href=\"llm-principles.html\">Transformer 구조</a>", "e5 안에서 벡터가 만들어지는 과정"), ("<a href=\"transfer-learning.html\">전이학습과 오류 분석</a>", "그림으로 본 같은 이야기"),
                   ("<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>", "같은 문장 벡터로 검색하기")],
        sources=["함께 정리한 자료: Lewis Tunstall 외, <i>Natural Language Processing with Transformers</i> 2장(데이터 살피기, 부분단어 토큰화, 특성 추출과 미세조정, F1과 불균형, 오류 분석, 모델 저장), 9장(Naive Bayes 기준선, 임베딩을 찾아보기 표로 쓰기와 FAISS, 데이터 늘리기, 퓨샷과 프롬프트, 도메인 적응, UDA와 UST). 부트캠프 수업에서 따로 다루지 않은 주제라 예제는 새로 만들었어요",
                 "데이터: <code>examples/classify/data/issues.csv</code> (이 노트용으로 만든 가상의 문의 780개, <code>tools/make_issue_examples.py</code>)",
                 "모델: <code>intfloat/multilingual-e5-small</code>. 결과 값과 그림은 <code>tools/pages/text_cls.py</code>가 <code>examples/classify/issue_classifier.py</code>를 단계별로 실제 실행한 결과 (CPU)",
                 "공식 문서: <a href=\"https://huggingface.co/docs/transformers/training\" target=\"_blank\">Fine-tune a pretrained model</a>, <a href=\"https://scikit-learn.org/stable/modules/naive_bayes.html\" target=\"_blank\">Naive Bayes</a>, <a href=\"https://huggingface.co/intfloat/multilingual-e5-small\" target=\"_blank\">multilingual-e5-small</a>"])
print("ok")
