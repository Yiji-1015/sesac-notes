"""모델 > 언어 모델로 문제 풀기 > 토큰 분류와 질문 답변. examples/tokens/token_tasks.py 하나가 ①~⑧로 자란다.
NLP with Transformers 4장(다중 언어 개체명 인식), 7장(질문 답변)을 합친 쪽.
실행: HF_HUB_OFFLINE=1 python tools/pages/token_cls.py   (CPU로 2분쯤)
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

import torch  # noqa: E402
import transformers  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "tokens"
code = S.parse((EX / "token_tasks.py").read_text(encoding="utf-8"))
try:
    ns = S.run(code, 8, S.workdir(EX), filename="token_tasks.py")
except OSError:
    print("e5 모델(intfloat/multilingual-e5-small)이 없어서 이 쪽은 다시 만들지 않아요. 기존 파일을 그대로 둬요.")
    sys.exit(0)

from vstyle import plt, C, setup  # noqa: E402

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "tokens"
IMG.mkdir(parents=True, exist_ok=True)
E, J = html.escape, K.josa
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, intfloat/multilingual-e5-small, CPU"
split, TAGS, tok = ns["split"], ns["TAGS"], ns["tok"]
KO = {"PER": "사람", "ORG": "회사", "DATE": "날짜", "MONEY": "금액"}
ner_s, ex, en, qs, sh = ns["ner_scores"], ns["exact"], ns["en_scores"], ns["qa_scores"], ns["search_hits"]


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/tokens/{name}.png"


def mark(r):                                                     # 개체에 [종류] 표시를 붙인 문장 (출력 상자용)
    t, out, last = r["text"], "", 0
    for a, z, k in sorted(r["entities"]):
        out += E(t[last:a]) + f"<b>[{E(t[a:z])}]</b><small>{KO[k]}</small>"
        last = z
    return out + E(t[last:])


outs = {}
# ---------------- 왜
qa0 = ns["qa_split"]["train"][0]
outs["data"] = S.table("예제 데이터에서 몇 줄", ["파일", "내용"],
                       [["ner.jsonl", mark(split["train"][0])], ["ner.jsonl", mark(split["train"][1])], ["ner.jsonl (영어)", mark(split["en_test"][0])],
                        ["qa.jsonl", f"메시지: {E(qa0['context'])}<br>질문: {E(qa0['question'])} → 답: <b>{E(qa0['answer']['text'])}</b>"]],
                       ["개체명 인식(NER)은 문장 속 <b>어느 글자부터 어느 글자까지가 어떤 종류의 이름인지</b>를 찾아요. 문장 하나에 라벨 하나가 아니라, 글자 범위마다 라벨이 있어요.",
                        "질문 답변은 메시지 안에서 <b>질문의 답이 되는 글자 범위</b>를 찾아요. 답이 메시지에 없으면 \"답 없음\"이라고 해야 해요."])
S.claim(ner_s["test"]["f1"] > 0.8 and ex["test"]["조사 떼기"] > ex["test"]["그대로"] + 0.1 and en["f1"] > 0.6 and qs["test"]["exact"] > 0.7, "왜: 네 가지 결과가 모두 쓸 만하다")
outs["why"] = S.table("이 쪽에서 만드는 것과 처음 보는 데이터에서의 결과", ["무엇을", "어떻게 쟀나", "결과", "단계"],
                      [["한국어 메시지의 개체 찾기", f"처음 보는 말투와 이름 {len(split['test'])}문장, 개체 단위 F1", f3(ner_s["test"]["f1"]), "③④"],
                       ["찾은 개체를 글자 그대로 꺼내기", "정답 개체와 글자까지 같은 비율", f"{ex['test']['그대로']:.2f} → <b>{ex['test']['조사 떼기']:.2f}</b> (조사 떼기)", "⑤"],
                       ["영어 문장의 개체 찾기", "영어는 학습에 한 번도 안 씀, 개체 단위 F1", f3(en["f1"]), "⑥"],
                       ["메시지에서 질문의 답 꺼내기", f"처음 보는 이름과 질문 말투 {len(ns['qa_split']['test'])}개, 정확히 일치", f3(qs["test"]["exact"]), "⑦"],
                       [f"메시지 {len(ns['docs'])}개에서 찾아서 답하기", "검색 + 답 꺼내기, 정확히 일치", f"{sh['찾고 읽기']:.2f}", "⑧"]],
                      ["<a href=\"text-classification.html\">문장 분류</a>에서는 문장 벡터 하나로 답 하나를 냈어요. 이 쪽에서는 e5가 내는 <b>토큰마다의 벡터</b>를 하나하나 써요. 같은 몸통에 얹는 층만 바뀌어요.",
                       "숫자보다 중요한 건 실패의 모양이에요. 한국어는 조사가 이름에 붙어서 토큰 경계와 이름 경계가 어긋나고(⑤), 다른 언어로 옮기면 경계가 한두 글자씩 틀려요(⑥)."])

# ---------------- ①
kinds = Counter(k for r in split["train"] for _, _, k in r["entities"])
r0 = split["train"][0]
outs["s1"] = S.box("① 데이터", [*(f"{s} → {len(v)}문장" for s, v in split.items()), f"train 개체 수 → {dict(kinds)}",
                                 (f"split['train'][0]['text'] → {r0['text']}", 1), (f"split['train'][0]['entities'] → {r0['entities']}", 2, "newf"), (f"TAGS → {TAGS}", 3, "newf")],
                   ["메시지 문장 하나예요.", "개체마다 [시작 글자, 끝 글자, 종류]예요. 끝은 포함하지 않아요(파이썬 슬라이스처럼). <code>text[0:3]</code>이 사람 이름이에요.",
                    "라벨 9개: 개체가 아니면 <code>O</code>, 개체의 첫 토큰은 <code>B-종류</code>, 나머지는 <code>I-종류</code>예요(BIO 방식). B와 I를 나누는 건, 같은 종류의 개체 두 개가 붙어 있을 때 어디서 나뉘는지 알기 위해서예요."])

# ---------------- ②
pairs = [(t, TAGS[x] if x >= 0 else "-") for t, x in zip(tok.convert_ids_to_tokens(ns["b0"]["input_ids"][0]), ns["l0"][0].tolist())]
off0 = ns["b0"]["offset_mapping"][0].tolist()
outs["s2"] = S.box("② encode_ner(train 첫 문장)", [(f"offset_mapping → {off0[:8]} …", 1), (" ".join(f"{t}/{g}" for t, g in pairs), 2, "newf")],
                   ["토큰마다 원래 글자에서의 (시작, 끝) 위치예요. <code>&lt;s&gt;</code>와 <code>&lt;/s&gt;</code>는 (0, 0)이에요. 이 위치를 개체의 글자 범위와 겹쳐 보고 라벨을 정해요.",
                    "토큰/라벨이에요. \"서지아\"는 토큰 셋(<code>▁서</code>, <code>지</code>, <code>아</code>)으로 나뉘어 B, I, I가 됐고, 표시용 토큰은 <code>-</code>(-100, 채점에서 빼는 자리)예요."])
mb = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
ex_text = split["train"][0]["text"]
outs["t2"] = S.table("같은 문장을 두 Tokenizer로 나누면", ["Tokenizer", "토큰"],
                     [["e5 (SentencePiece, XLM-R 계열)", E(" ".join(tok.tokenize(ex_text)))], ["mBERT (WordPiece)", E(" ".join(mb.tokenize(ex_text)))]],
                     ["SentencePiece는 띄어쓰기도 글자로 보고 <code>▁</code>로 표시해서, 띄어쓰기가 없는 언어(중국어, 일본어)에도 똑같이 쓸 수 있어요. WordPiece는 먼저 띄어쓰기로 나눈 뒤 단어 안을 쪼개고, 이어지는 조각에 <code>##</code>을 붙여요.",
                      "Tokenizer는 보통 네 단계예요: 글자 정리(유니코드 통일, 소문자 등), 1차로 나누기(띄어쓰기, 문장 부호), 조각 나누기(BPE, WordPiece, Unigram), 앞뒤 표시 붙이기(<code>&lt;s&gt;</code>). 모델마다 짝인 Tokenizer가 정해져 있어서 바꿔 쓸 수 없어요."])

# ---------------- ③
tagger = ns["tagger"]
n_train = sum(p.numel() for p in tagger.parameters() if p.requires_grad)
with torch.no_grad():
    lg = tagger(ns["b0"])
S.claim(all(a_ > b_ for a_, b_ in zip(ns["ner_log"], ns["ner_log"][1:])), "③ NER 손실이 epoch마다 내려간다")
outs["s3"] = S.box("③ 토큰마다 분류 (12층 미세조정, 6 epoch)", [(f"tagger(b0).shape → {tuple(lg.shape)}", 1), (f"epoch마다 마지막 묶음 손실 → {[round(x, 3) for x in ns['ner_log']]}", 2), f"고치는 가중치 → {n_train:,}개"],
                   [f"(문장 1, 토큰 {lg.shape[1]}, 라벨 9)예요. 문장 분류는 문장마다 점수 3개였는데, 여기서는 토큰마다 점수 9개예요. 같은 384 → 9 층을 모든 토큰에 똑같이 써요.",
                    f"train 문장 {len(split['train'])}개로 손실이 꾸준히 내려가요. 문장 분류(⑤)에서는 12층 전부를 고치면 말투를 외워 버렸는데, 여기서는 문장 하나에 라벨이 토큰 수만큼 있어서 배울 신호가 훨씬 많아요. 12층을 다 고쳐도 처음 보는 말투에서 잘 맞혀요(④)."])

# ---------------- ④
b_t, lab_t, pred_t = ns["predict_tags"](split["test"])
keep = lab_t != -100
o_share = float((lab_t[keep] == 0).float().mean())
S.claim(ner_s["test"]["token_acc"] > ner_s["test"]["f1"] + 0.05 and o_share > 0.5, "④ 토큰 정확도가 개체 F1보다 높게 나오고, 토큰 대부분이 O")
outs["s4"] = S.table("④ 채점 (토큰 정확도 vs 개체 단위)", ["데이터", "토큰 정확도", "개체 precision", "개체 recall", "개체 F1"],
                     [[f"{s} ({'같은 말투' if s == 'valid' else '처음 보는 말투와 이름'})", f3(v["token_acc"]), f3(v["precision"]), f3(v["recall"]), f"<b>{f3(v['f1'])}</b>"] for s, v in ner_s.items()],
                     [f"토큰 정확도가 더 높게 나와요. test 토큰의 {o_share:.0%}가 <code>O</code>라서, 개체를 하나도 못 찾아도 정확도가 꽤 나와요. 그래서 NER은 <b>개체 단위</b>로 채점해요.",
                      "개체 단위는 시작, 끝, 종류가 모두 맞아야 하나로 쳐요. \"한빛상사\"의 한 토큰만 틀려도 그 개체는 틀린 거예요. precision은 찾은 것 중 맞은 비율, recall은 있어야 할 것 중 찾은 비율이에요. seqeval이라는 라이브러리가 이 방식으로 재 줘요."])

# ---------------- ⑤
raw_t = ns["extract"](split["test"], clean=False)
cln_t = ns["extract"](split["test"])
sample = next((r, a, c) for r, a, c in zip(split["test"], raw_t, cln_t) if a != c)
diff = [(x[0], y[0]) for x, y in zip(sample[1], sample[2]) if x != y]
S.claim(ex["valid"]["그대로"] < 0.9 and ex["valid"]["조사 떼기"] > 0.95 and ner_s["valid"]["f1"] == 1.0, "⑤ valid는 토큰 단위 F1이 1인데 글자 그대로는 0.9 아래, 조사를 떼면 0.95 넘게")
outs["s5"] = S.table("⑤ 정답 개체를 글자까지 똑같이 꺼낸 비율", ["데이터", "토큰 단위 개체 F1 (④)", "글자 그대로", "끝 조사 떼기"],
                     [[s, f3(ner_s[s]["f1"]), f3(v["그대로"]), f"<b>{f3(v['조사 떼기'])}</b>"] for s, v in ex.items()],
                     [f"valid는 토큰 단위로는 다 맞았는데(F1 {ner_s['valid']['f1']:.1f}), 글자로 꺼내면 {ex['valid']['그대로']:.0%}만 정답과 같아요. 예를 들어 test의 \"{E(sample[0]['text'])}\"에서 {', '.join(f'<code>{E(a)}</code>' for a, _ in diff)}{J(diff[-1][0], '이', '가')} 꺼내졌어요.",
                      "e5의 Tokenizer가 개체의 끝 글자와 조사를 한 토큰으로 묶는 경우가 있어서예요. 토큰 하나는 라벨 하나만 가지니, 그 토큰이 개체에 들어가면 조사까지 따라와요. 한국어처럼 조사가 붙는 언어에서 늘 생기는 일이에요.",
                      f"끝에서 흔한 조사를 하나 떼는 것만으로 {ex['valid']['그대로']:.2f}에서 {ex['valid']['조사 떼기']:.2f}까지 올라요. 실제로는 형태소 분석기로 조사를 미리 나누거나, 글자 단위로 라벨을 붙이기도 해요."])

# ---------------- ⑥
ef = ns["en_found"]
S.claim(en["f1"] > 0.6 and en["recall"] > en["precision"], "⑥ 영어 F1이 0.6을 넘고, recall이 precision보다 높다")
outs["s6"] = S.box("⑥ 한국어로만 배운 모델로 영어 문장 48개", [(f"토큰 정확도 {en['token_acc']:.3f}, 개체 F1 {en['f1']:.3f} (precision {en['precision']:.3f}, recall {en['recall']:.3f})", 1, "newf"),
                                                       *(f"\"{r['text']}\" → {f}" for r, f in zip(split["en_test"][:4], ef))],
                   [f"영어 문장은 학습에 한 번도 안 썼는데 개체를 {en['f1']:.2f}만큼 찾아요. e5는 100여 개 언어의 글로 함께 학습해서, \"사람 이름 자리\", \"날짜 자리\"를 언어와 상관없이 비슷한 벡터로 표현해요. 한 언어로 배운 걸 다른 언어에 쓰는 걸 <b>언어 간 전이</b>(zero-shot cross-lingual transfer)라고 해요.",
                    "틀린 모양을 보면 경계가 한두 글자씩 어긋나요. 마침표가 붙거나 마지막 조각이 빠져요. 이걸 줄이려면 라벨이 있는 영어 문장을 조금 섞어 학습하는 게 보통의 다음 단계예요."])

# ---------------- ⑦ 시작, 끝 확률 그림
reader = ns["reader"]
qt = ns["qa_split"]["test"]
pred_qt = ns["answer"](qt)
ok_i = next(i for i, (r, p) in enumerate(zip(qt, pred_qt)) if r["answer"] and p == r["answer"]["text"])
rq = qt[ok_i]
bq, _, _ = ns["encode_qa"]([rq])
with torch.no_grad():
    s_, e_ = reader(bq)
ctx = [j for j, q in enumerate(bq.sequence_ids(0)) if q == 1][:40]
toks = tok.convert_ids_to_tokens(bq["input_ids"][0])
ps, pe = s_[0].softmax(-1), e_[0].softmax(-1)
fig, ax = plt.subplots(figsize=(13, 3.6))
xs = range(len(ctx))
ax.bar([x - 0.2 for x in xs], [float(ps[j]) for j in ctx], width=0.4, color=C["purple"], label="시작일 확률")
ax.bar([x + 0.2 for x in xs], [float(pe[j]) for j in ctx], width=0.4, color=C["pink"], label="끝일 확률")
ax.set_xticks(list(xs), [toks[j].replace("▁", "_") for j in ctx], rotation=70, fontsize=8)
ax.set(ylabel="확률", title=f"질문: {rq['question']}  →  답: {rq['answer']['text']}")
ax.legend(fontsize=9)
ax.grid(False)
za, zz = ctx[int(torch.stack([ps[j] for j in ctx]).argmax())], ctx[int(torch.stack([pe[j] for j in ctx]).argmax())]
offq = bq["offset_mapping"][0].tolist()
raw_ans = rq["context"][offq[za][0]:offq[zz][1]].strip()
span_note = (f" 여기서는 끝 토큰이 <code>{E(toks[zz])}</code>라서 잘라 낸 글은 <code>{E(raw_ans)}</code>이고, ⑤의 조사 떼기로 <code>{E(ns['strip_josa'](raw_ans))}</code>{J(ns['strip_josa'](raw_ans), '이', '가')} 됐어요."
             if ns["strip_josa"](raw_ans) != raw_ans else "")
outs["f_span"] = S.images("메시지의 토큰마다 \"답의 시작일 확률\"과 \"끝일 확률\"", [(fig_save(fig, "span_probs"), "test 질문 하나")],
                          ["모델은 토큰마다 점수 두 개(시작, 끝)를 내요. softmax를 하면 거의 한 토큰에 몰려요. 시작이 가장 높은 토큰부터 끝이 가장 높은 토큰까지가 답이에요." + span_note])
outs["f_span"]["chart"] = True
by_q = Counter()
tot_q = Counter()
for r, p in zip(qt, pred_qt):
    g = r["answer"]["text"] if r["answer"] else None
    tot_q[r["question"]] += 1
    by_q[r["question"]] += int(p == g)
worst = sorted(tot_q, key=lambda q: by_q[q] / tot_q[q])[:3]
train_q = {r["question"] for r in ns["qa_split"]["train"]}
S.claim("계약은 언제 맺었나요?" in train_q and "계약한 날짜가 언제예요?" not in train_q and qs["test"]["답 없는 질문"] == 1.0, "⑦ 질문 말투 예시가 사실이고, 답 없는 질문은 모두 맞힌다")
S.claim(qs["valid"]["exact"] > qs["test"]["exact"] and qs["test"]["답 없는 질문"] > qs["test"]["답 있는 질문"], "⑦ 같은 말투(valid)가 test보다 높고, test에서는 답 없는 질문을 더 잘 맞힌다")
outs["s7"] = S.box("⑦ 질문 답변 (12층 미세조정, 2 epoch)", [(f"epoch마다 마지막 묶음 손실 → {[round(x, 3) for x in ns['qa_log']]}", 1),
                                                       *(f"{s}: 정확히 일치 {v['exact']:.3f} (답 있는 질문 {v['답 있는 질문']:.3f}, 답 없는 질문 {v['답 없는 질문']:.3f})" for s, v in qs.items()),
                                                       (f"test에서 가장 많이 틀린 질문 → " + ", ".join(f"\"{q}\" {by_q[q]}/{tot_q[q]}" for q in worst), 2, "bad")],
                   ["train 질문 870개로 2 epoch만 학습했어요.", "질문을 처음 보는 말투로 바꾸면 떨어져요. train에는 \"계약은 언제 맺었나요?\"만 있고 \"계약한 날짜가 언제예요?\"는 없었어요. 같은 뜻의 질문을 여러 말투로 넣어 학습해야 해요. 답 없는 질문은 모두 \"답 없음\"이라고 맞혔어요."])

# ---------------- ⑧
sr = ns["search"]
S.claim(sh["찾기"] == 1.0 and sh["찾고 읽기"] >= 0.9, "⑧ 메시지는 모두 제대로 찾고, 답도 거의 다 맞힌다")
outs["s8"] = S.box("⑧ 메시지 12개에서 찾고 읽기", [(f"맞는 메시지를 찾은 비율 → {sh['찾기']:.2f}", 1), (f"찾고 읽어서 답까지 맞힌 비율 → {sh['찾고 읽기']:.2f}", 2, "newf"),
                                                 *(f"\"{r['question']}\" → {a}" for r, a in list(zip(sr, ns["read"]))[::4])],
                   ["질문에 든 회사나 사람 이름 덕분에, e5 문장 벡터만으로 12개 중 맞는 메시지를 찾았어요.", "찾은 메시지에 ⑦의 모델을 그대로 써서 답을 꺼냈어요. 검색(Retriever)과 읽기(Reader)를 따로 만들어 이어 붙인 구조예요."])

flow = [[1, "데이터", "글자 범위 + 종류", "BIO 라벨 9개"], [2, "라벨 맞추기", "offset_mapping", "토큰마다 라벨"], [3, "토큰 분류", "384 → 9", "12층 미세조정"], [4, "채점", "개체 단위", f"F1 {ner_s['test']['f1']:.2f}"],
        [5, "꺼내기", "조사 떼기", f"{ex['test']['그대로']:.2f} → {ex['test']['조사 떼기']:.2f}"], [6, "영어로", "언어 간 전이", f"F1 {en['f1']:.2f}"], [7, "질문 답변", "시작, 끝", f"{qs['test']['exact']:.2f}"], [8, "찾고 읽기", "검색 + 읽기", f"{sh['찾고 읽기']:.2f}"]]
S.save(ROOT / "topics" / "model" / "token-classification-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>업무 메시지에서 \"누가, 어느 회사와, 언제, 얼마\"를 뽑아 일정표나 장부에 자동으로 넣고 싶다고 해 봐요. 문장 전체에 라벨 하나를 붙이는 <a href="text-classification.html">문장 분류</a>로는 안 돼요. 문장 속 <b>어느 부분이</b> 무엇인지 알아야 해요. 이렇게 토큰마다 라벨을 붙이는 일을 <b>토큰 분류</b>라고 하고, 대표가 <b>개체명 인식</b>(NER, Named Entity Recognition)이에요. 메시지와 질문을 주고 답이 되는 부분을 찾게 하면 <b>질문 답변</b>(추출형 QA)이에요.</p>
<p>이 쪽의 예제는 가상의 업무 메시지예요.</p>
{K.out("data")}
{K.out("why")}"""
whole = """<p>데이터 → 글자 범위를 토큰 라벨로 → 토큰마다 분류 → 개체 단위 채점 → 글자로 꺼내기 → 영어에 써 보기 → 질문의 답 범위 찾기 → 찾아서 읽기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "데이터 읽기", "문장과 개체 위치(시작 글자, 끝 글자, 종류)를 읽고, 토큰마다 붙일 라벨 목록을 정해요.",
           "<p><b>왜 글자 위치로 저장하나?</b> 토큰은 Tokenizer마다 다르게 나뉘어서, 라벨을 토큰 단위로 저장해 두면 모델을 바꿀 때마다 다시 만들어야 해요. 글자 위치는 Tokenizer와 상관없어요. 학습할 때 그 모델의 토큰에 맞춰 바꿔요(②).</p>"
           "<p>test는 train에 없던 문장 틀 4개와 처음 보는 이름(사람 6명, 회사 4곳)으로, en_test는 영어 문장으로 만들었어요. en_test는 학습에 쓰지 않아요.</p>"
           + K.code_table([("json.loads(line)", "한 줄에 JSON 하나 (jsonl)"), ("[f\"{p}-{k}\" for k in KINDS for p in (\"B\", \"I\")]", "B-PER, I-PER, B-ORG, … 8개 + O")]), "s1"),
    K.step(2, "토큰과 라벨 맞추기", "Tokenizer가 알려 주는 토큰마다의 글자 위치를 개체의 글자 범위와 겹쳐 보고, 토큰마다 라벨을 정해요.",
           "<p><b>왜 어렵나?</b> 이름 하나가 여러 토큰으로 나뉘고(\"서지아\" → 셋), 조사가 이름과 한 토큰으로 묶이기도 해요. 그래서 \"토큰의 글자 범위가 개체와 겹치면 그 개체의 토큰\"으로 정하고, 개체의 첫 토큰은 B, 나머지는 I를 붙여요. <code>&lt;s&gt;</code>, <code>&lt;/s&gt;</code>, 채운 자리는 -100으로 둬서 손실과 채점에서 빼요.</p>"
           + K.code_table([("tok(..., return_offsets_mapping=True)", "토큰마다 원래 글자에서의 (시작, 끝)도 받기"), ("torch.full(shape, -100)", "모든 자리를 -100으로 채운 Tensor"),
                           ("s < z and e > a", "토큰 [s, e)와 개체 [a, z)가 겹치는가")])
           + K.try_("다른 Tokenizer로 나누면?", "t2"), "s2"),
    K.step(3, "토큰마다 분류하는 모델", "e5 몸통이 낸 토큰 벡터마다 같은 384 → 9 층을 얹어 라벨 점수를 내고, 정답 라벨로 미세조정해요.",
           "<p>문장 분류는 토큰 벡터를 평균 낸 문장 벡터 하나에 층을 얹었어요. 토큰 분류는 평균 내지 않고, <b>토큰 벡터 하나하나</b>에 같은 층을 얹어요. 몸통은 똑같고 머리(head)만 바뀌어요. 그래서 이런 모델을 \"몸통 + 과제별 머리\"로 나눠 불러요. transformers의 <code>AutoModelForTokenClassification</code>이 이 모양을 만들어 줘요.</p>"
           "<p>학습 반복은 <a href=\"text-classification.html\">문장 분류</a> ⑤와 같아서 함수 하나(<code>fine_tune</code>)로 묶었어요. ⑦의 질문 답변도 이 함수를 그대로 써요. 바뀌는 건 모델, 데이터를 Tensor로 바꾸는 함수(<code>encode</code>), 손실 함수뿐이에요.</p>"
           + K.code_table([("self.head(...last_hidden_state)", "(문장, 토큰, 384) → (문장, 토큰, 9)"), ("model(b).flatten(0, 1)", "(문장 × 토큰, 9)로 펴서 토큰을 하나의 줄로"),
                           ("F.cross_entropy(..., ignore_index=-100)", "-100 자리는 손실에서 빼기"), ("def fine_tune(model, rows, encode, loss_fn, epochs, train_layers=12)", "어떤 모델이든 같은 방식으로 학습하는 반복")]), "s3"),
    K.step(4, "개체 단위로 채점하기", "토큰 라벨을 (시작 토큰, 끝 토큰, 종류) 묶음으로 바꾸고, 정답 묶음과 정확히 같은 것만 맞은 걸로 쳐요.",
           "<p><b>왜 토큰 정확도로 안 재나?</b> 대부분의 토큰은 개체가 아니라 <code>O</code>예요. 모든 토큰에 O라고만 해도 정확도가 높게 나와요. 그리고 쓰는 쪽에서 필요한 건 \"한빛상사\"라는 개체 하나지, 그 안의 토큰 셋이 아니에요.</p>"
           + K.code_table([("tag_spans(tags)", "B로 시작해 같은 종류의 I가 이어지는 동안을 개체 하나로"), ("G & P, P - G, G - P", "집합의 교집합, 차집합. 맞은 것, 헛것, 놓친 것"),
                           ("pred.masked_fill(labels == -100, -100)", "채점에서 빼는 자리는 예측도 똑같이 빼기")]), "s4"),
    K.step(5, "글자로 꺼내기: 끝 토큰에 붙은 조사 떼기", "찾은 개체를 토큰의 글자 위치로 원래 문장에서 잘라 내고, 끝에 붙어 온 조사를 떼요.",
           "<p>실제로 쓸 때는 \"(3번째부터 5번째 토큰, 회사)\"가 아니라 \"한빛상사\"라는 글자가 필요해요. 토큰의 글자 위치로 잘라 내면 되는데, 한국어에서는 여기서 문제가 생겨요.</p>"
           + K.code_table([("text[off[s][0]:off[e][1]]", "첫 토큰의 시작 글자부터 끝 토큰의 끝 글자까지"), ("strip_josa(text)", "흔한 조사로 끝나면 하나 떼기 (남는 글자가 2자 이상일 때만)")]), "s5"),
    K.step(6, "다른 언어로 옮기기: 한국어로만 배운 모델에 영어 문장", "한국어 문장으로만 학습한 ③의 모델에 영어 문장 48개를 넣고 똑같이 채점해요.",
           "<p><b>왜 해 보나?</b> 라벨이 있는 데이터는 언어마다 양이 크게 달라요. 여러 언어를 함께 배운 모델이라면, 라벨이 많은 언어로 배운 걸 라벨이 거의 없는 언어에 바로 쓸 수 있어요. e5의 바탕이 된 모델은 100개 언어의 글로 학습했어요.</p>"
           + K.code_table([("score_ner(split[\"en_test\"])", "④와 같은 채점을 영어 문장에")]), "s6"),
    K.step(7, "질문 답변: 답의 시작과 끝 찾기", "질문과 메시지를 한 줄로 이어 넣고, 메시지의 토큰마다 \"답의 시작일 점수\"와 \"끝일 점수\"를 내게 학습해요.",
           "<p>답이 메시지 안에 그대로 있다면, 답을 새로 만들 필요 없이 <b>어디서 시작해 어디서 끝나는지</b>만 고르면 돼요(추출형 QA). 그래서 머리는 384 → 2(시작 점수, 끝 점수)예요. 시작 점수 + 끝 점수가 가장 큰 범위를 답으로 고르고, 시작이 끝보다 뒤인 범위는 빼요.</p>"
           "<p><b>답이 없으면?</b> 학습할 때 답이 없는 질문은 0번 토큰(<code>&lt;s&gt;</code>)을 시작이자 끝으로 가르쳐요. 쓸 때는 <code>&lt;s&gt;</code>의 점수가 가장 좋은 범위의 점수보다 크면 \"답 없음\"이라고 해요. 그럴듯한 엉뚱한 답을 내는 것보다 낫기 때문이에요.</p>"
           "<p>질문과 메시지는 <code>&lt;s&gt; 질문 &lt;/s&gt;&lt;/s&gt; 메시지 &lt;/s&gt;</code>처럼 한 줄로 들어가요. <code>sequence_ids</code>가 토큰마다 질문(0)인지 메시지(1)인지 알려 줘서, 답은 메시지 쪽에서만 찾아요. 메시지가 모델의 한도(512토큰)보다 길면 겹치게 잘라 여러 번 넣고 가장 좋은 답을 골라요.</p>"
           + K.out("f_span")
           + K.code_table([("tok(questions, contexts, ...)", "두 글을 한 쌍으로 이어 붙여 넣기"), ("b.sequence_ids(i)[j] == 1", "j번 토큰이 메시지 쪽인가"),
                           ("logits[..., 0], logits[..., 1]", "토큰마다 시작 점수, 끝 점수"), ("max((s + e, a, z) for a in ctx for z in ctx if a <= z < a + max_len)", "시작 ≤ 끝, 길이 12토큰 이하 중 점수 합이 가장 큰 범위")]), "s7"),
    K.step(8, "찾아서 읽기: 메시지 여러 개에서 먼저 찾고, 찾은 메시지에서 답 꺼내기", "메시지 12개를 e5 문장 벡터로 바꿔 두고, 질문과 가장 가까운 메시지를 찾은 뒤 ⑦의 모델로 답을 꺼내요.",
           "<p>⑦은 \"답이 들어 있는 메시지\"를 알려 줘야 했어요. 실제로는 메시지가 수천 개라, 먼저 관련 있는 것을 찾아야 해요. <b>Retriever</b>(찾는 쪽)가 후보를 고르고 <b>Reader</b>(읽는 쪽)가 답을 꺼내는 구조예요. 찾는 쪽은 BM25 같은 단어 검색이나, 질문과 문서를 각각 벡터로 바꾸는 방식(DPR, 여기서는 e5)을 써요.</p>"
           "<p>Reader 자리에 답을 꺼내는 모델 대신 글을 만드는 LLM을 두면 RAG예요. 찾은 문서를 근거로 답을 새로 써요(<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>, <a href=\"../backend/indexing.html\">Indexing</a>).</p>"
           + K.code_table([("AutoModel.from_pretrained(MODEL)", "검색에는 미세조정하지 않은 원래 e5를 따로"), ("embed(docs, \"passage: \")", "메시지는 <code>passage:</code>, 질문은 <code>query:</code>를 붙여요 (e5의 검색 학습 방식)"),
                           ("(embed(...) @ doc_vec.T).argmax(1)", "질문마다 코사인 유사도가 가장 큰 메시지")]), "s8"),
]
practice = [("공개 데이터와 모델", "NER은 WikiANN(PAN-X), 한국어는 KLUE-NER, 질문 답변은 SQuAD, 한국어는 KorQuAD가 많이 쓰여요. Hub에 이미 미세조정된 모델이 많아서 <code>pipeline(\"ner\")</code>, <code>pipeline(\"question-answering\")</code>으로 바로 써 볼 수 있어요."),
            ("채점 도구", "NER은 <code>seqeval</code>(개체 단위 precision, recall, F1), 질문 답변은 정확히 일치(EM)와, 답과 정답이 겹치는 정도를 재는 F1을 함께 봐요."),
            ("한국어 개체의 경계", "조사가 붙는 문제(⑤)는 형태소 분석기로 미리 나누거나, 글자 단위로 라벨을 붙이거나, 꺼낸 뒤 규칙으로 다듬어요. 어느 쪽이든 \"글자 그대로 일치\"로 꼭 재 봐요."),
            ("여러 언어를 다뤄야 한다", "다국어 모델(XLM-R, mBERT, e5 다국어판)을 쓰고, 라벨이 많은 언어로 먼저 배운 뒤 다른 언어의 라벨을 조금씩 더하면 효율이 좋아요(⑥). 여러 언어를 섞어 함께 학습하면 각 언어의 점수가 고르게 올라요."),
            ("긴 문서의 질문 답변", "모델 한도(보통 512토큰)를 넘으면 겹치게 잘라(stride) 여러 조각으로 넣고, 조각마다 가장 좋은 답 중에서 골라요."),
            ("답을 새로 써야 한다", "답이 문서에 그대로 없거나 여러 곳을 합쳐야 하면, 꺼내는 모델 대신 생성 모델을 써요(RAG). 대신 근거 없는 답을 지어내지 않는지 검사해요 (<a href=\"../backend/prompt-contract.html\">Prompt와 출력 계약</a>).")]
libs = [("②", "<code>return_offsets_mapping=True</code>, <code>torch.full</code>", "토큰과 글자 위치 (transformers)"), ("③⑦", "<code>AutoModel</code>, <code>nn.Linear</code>, <code>F.cross_entropy(ignore_index=-100)</code>", "몸통 + 머리, 미세조정"),
        ("④", "집합 연산 <code>&amp;</code>, <code>-</code>", "개체 단위 채점"), ("⑦", "<code>tok(questions, contexts)</code>, <code>sequence_ids</code>", "질문과 메시지 쌍"), ("⑧", "<code>F.normalize</code>, 행렬 곱 <code>@</code>", "벡터 검색")]
K.build(ROOT / "topics" / "model" / "token-classification.html", title="토큰 분류와 질문 답변", h1="토큰 분류와 질문 답변: 문장 속 어디가 무엇인지", track="모델", crumb="모델 &gt; 언어 모델로 문제 풀기",
        one_line="문장 분류가 문장마다 라벨 하나를 붙인다면, <b>토큰 분류</b>는 토큰마다 라벨을 붙여 문장 속 이름, 날짜, 금액을 찾아요(<b>개체명 인식</b>). 메시지에서 질문의 답이 시작하고 끝나는 토큰을 찾으면 <b>질문 답변</b>이에요. 여러 언어를 함께 배운 모델은 한 언어로 배운 것을 다른 언어에도 써요.",
        why=why, whole=whole, steps=steps, code_file="examples/tokens/token_tasks.py", practice=practice, libs=libs,
        libs_note="<code>encode_ner</code>, <code>fine_tune</code>, <code>tag_loss</code>, <code>tag_spans</code>, <code>predict_tags</code>, <code>score_ner</code>, <code>strip_josa</code>, <code>extract</code>, <code>exact_rate</code>, <code>encode_qa</code>, <code>span_loss</code>, <code>answer</code>, <code>score_qa</code>, <code>embed</code>는 이 파일 안에서 정의한 함수, <code>Tagger</code>, <code>Reader</code>는 클래스예요.",
        next_rows=[("<a href=\"text-classification.html\">문장 분류와 미세조정</a>", "같은 몸통, 문장마다 라벨 하나"), ("<a href=\"llm-principles.html\">Transformer 구조</a>", "토큰마다 벡터가 만들어지는 과정"),
                   ("<a href=\"ocr-documents.html\">OCR과 문서 구조화</a>", "규칙으로 값 꺼내기"), ("<a href=\"../backend/embedding.html\">Embedding과 벡터 검색</a>", "⑧의 찾는 쪽")],
        sources=["함께 정리한 자료: Lewis Tunstall 외, <i>Natural Language Processing with Transformers</i> 4장(다중 언어 개체명 인식: XLM-R, SentencePiece, Tokenizer 파이프라인, 몸통과 머리, 토큰과 라벨 맞추기, seqeval, 언어 간 전이), 7장(질문 답변: 시작과 끝 분류, 질문과 문맥 쌍, 긴 문서 자르기, 답 없는 질문, Retriever와 Reader, DPR, RAG). 부트캠프 수업에서 따로 다루지 않은 주제라 예제는 새로 만들었어요",
                 "데이터: <code>examples/tokens/data/</code>의 <code>ner.jsonl</code>, <code>qa.jsonl</code>, <code>search.jsonl</code> (이 노트용으로 만든 가상의 업무 메시지, <code>tools/make_token_examples.py</code>)",
                 "모델: <code>intfloat/multilingual-e5-small</code>, ②의 비교용 Tokenizer <code>bert-base-multilingual-cased</code>. 결과 값과 그림은 <code>tools/pages/token_cls.py</code>가 <code>examples/tokens/token_tasks.py</code>를 단계별로 실제 실행한 결과 (CPU)",
                 "공식 문서: <a href=\"https://huggingface.co/docs/transformers/tasks/token_classification\" target=\"_blank\">Token classification</a>, <a href=\"https://huggingface.co/docs/transformers/tasks/question_answering\" target=\"_blank\">Question answering</a>, <a href=\"https://github.com/chakki-works/seqeval\" target=\"_blank\">seqeval</a>"])
print("ok")
