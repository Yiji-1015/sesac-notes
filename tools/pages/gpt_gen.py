"""모델 > 순서 모델과 Transformer > 작은 GPT와 디코딩. examples/gpt/tiny_gpt.py 하나가 ①~⑧로 자란다.
NLP with Transformers 5장(텍스트 생성), 10장(밑바닥부터 학습), 6장(요약과 ROUGE)을 합친 쪽.
실행: HF_HUB_OFFLINE=1 python tools/pages/gpt_gen.py   (CPU로 3분쯤)
e5 모델을 받아 두지 않은 PC에서는 아무 파일도 바꾸지 않고 끝낸다.
"""
import html
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402
import tokenizers  # noqa: E402
import transformers  # noqa: E402

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "gpt"
code = S.parse((EX / "tiny_gpt.py").read_text(encoding="utf-8"))
try:
    ns = S.run(code, 8, S.workdir(EX), filename="tiny_gpt.py")
except OSError:
    print("e5 모델(intfloat/multilingual-e5-small)이 없어서 이 쪽은 다시 만들지 않아요. 기존 파일을 그대로 둬요.")
    sys.exit(0)

from vstyle import plt, C, setup  # noqa: E402

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "gpt"
IMG.mkdir(parents=True, exist_ok=True)
E, J = html.escape, K.josa
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, tokenizers {tokenizers.__version__}, CPU (seed 0)"
tk, gpt, log, dt, rt = ns["tk"], ns["gpt"], ns["log"], ns["decode_table"], ns["rouge_table"]
V = tk.get_vocab_size()
dec = lambda ids: tk.decode(ids)  # noqa: E731


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/gpt/{name}.png"


outs = {}
p0 = ns["prompts"][0]
gen0 = {name: dec(seqs[0]) for name, seqs in ns["outputs"].items()}
# ---------------- 왜
dup_top = ns["dup"].most_common(3)
outs["data"] = S.box("notes_corpus.jsonl: 옛 Day 노트에서 사람이 쓴 문단만", [f"Day {len(ns['days'])}일치, 문단 {len(ns['paras']):,}개, {sum(len(p) for p in ns['paras']):,}자",
                                                                       *(f"예: {p[:70]}…" for p in ns["paras"][3:6])],
                     ["코드, 표, 주소, 마크다운 기호, 이모지는 빼고(<code>tools/make_gpt_corpus.py</code>), 가운뎃점은 노트 규칙대로 쉼표로 바꿨어요. 이 글만 보고 \"이 노트처럼 쓰는\" GPT를 처음부터 만들어요."])
S.claim(dt["beam 5"]["뜻밖의 정도"] == min(v["뜻밖의 정도"] for v in dt.values()) and dt["beam 5"]["반복"] == max(v["반복"] for v in dt.values()), "왜: beam이 가장 그럴듯하면서 가장 많이 반복한다")
outs["why"] = S.table(f"같은 GPT, 같은 앞부분 \"{p0}\"에 이어 쓰게 해도 고르는 방법에 따라", ["고르는 방법", "이어 쓴 글 (30토큰)"],
                      [[f"<b>{n}</b>", E(gen0[n])] for n in ("greedy", "beam 5", "top-p 0.9")],
                      ["greedy와 beam은 매번 \"가장 그럴듯한\" 토큰을 골라서 같은 말을 되풀이해요. 확률대로 뽑으면 반복은 거의 사라지지만 엉뚱한 말이 섞여요.",
                       "모델이 똑같아도 다음 토큰을 <b>어떻게 고르느냐</b>(디코딩)에 따라 글이 완전히 달라져요. 이 쪽에서는 이 노트의 글로 작은 GPT를 직접 만들고, 고르는 방법을 숫자로 비교해요."])

# ---------------- ①
outs["s1"] = S.box("① 말뭉치 정리", [(f"문단 {len(ns['paras']):,}개 → 같은 문단을 빼면 {len(ns['unique']):,}개", 1, "newf"),
                                   *(f"  {n}번 나온 문단: \"{p[:40]}\"" for p, n in dup_top),
                                   (f"train {len(ns['train_txt']):,}문단, valid {len(ns['valid_txt']):,}문단", 2)],
                   [f"같은 문단이 {len(ns['paras']) - len(ns['unique'])}번 겹쳐 있었어요. 노트마다 붙인 상투적인 제목과 안내 문장이에요. 그대로 두면 모델이 이 문장들을 통째로 외워서, 글을 만들 때 자꾸 꺼내요. 큰 말뭉치를 만들 때도 중복 제거가 가장 먼저 하는 정리예요.",
                    "문단 20개 중 하나를 valid로 남겨, 학습에 안 쓴 글에서 얼마나 맞히는지 재요."])

# ---------------- ②
sent = "같은 데이터로도 질문을 두 가지로 만들 수 있어요."
cpt = ns["chars_per_token"]
S.claim(cpt["우리 Tokenizer (4천 개)"] > cpt["e5 Tokenizer (25만 개)"], "② 이 노트로 배운 4천 개 Tokenizer가 e5보다 이 글을 더 짧게 나눈다")
outs["s2"] = S.box("② Tokenizer", [(f"어휘 수 → {V:,}", 1), (f"우리 Tokenizer → {tk.encode(sent).tokens}", 2), f"e5 Tokenizer → {ns['e5_tok'].tokenize(sent)}",
                                   (f"valid 글에서 토큰 하나가 평균 몇 글자인가 → " + ", ".join(f"{k} {v:.2f}" for k, v in cpt.items()), 3, "newf")],
                   ["4,000개에서 <code>&lt;unk&gt;</code>, <code>&lt;eos&gt;</code> 두 개는 특별 토큰이에요.",
                    "이 노트에 자주 나오는 \"데이터로\", \"질문을\", \"있어요.\"가 통째로 한 토큰이 됐어요. BPE는 가장 자주 붙어 나오는 조각 쌍을 하나로 합치기를 어휘 수가 찰 때까지 되풀이해서 어휘를 만들어요.",
                    f"어휘가 {round(len(ns['e5_tok']) / V, -1):.0f}분의 1인데 이 글을 더 짧게 나눠요({cpt['우리 Tokenizer (4천 개)']:.2f}자 대 {cpt['e5 Tokenizer (25만 개)']:.2f}자). e5는 100개 언어에 어휘를 나눠 써야 하지만, 우리 것은 이 노트의 말투에만 맞췄기 때문이에요. 토큰이 적으면 같은 글을 더 짧게 넣을 수 있어 학습과 생성이 빨라져요."])
vt = ns["valid_all"]
vrows = []
for vs in (1000, 4000, 16000):
    t_ = ns["train_tokenizer"](ns["train_txt"], vs)
    vrows.append([f"<b>{vs:,}</b> (지금)" if vs == 4000 else f"{vs:,}", f"{t_.get_vocab_size():,}", f"{len(vt) / len(t_.encode(vt).ids):.2f}", E(" ".join(t_.encode(sent).tokens))])
S.claim(int(vrows[2][1].replace(",", "")) < 16000, "② 1만 6천을 정해도 실제 어휘는 그보다 적다")
S.claim(float(vrows[0][2]) < float(vrows[1][2]) < float(vrows[2][2]), "어휘가 클수록 토큰 하나가 더 많은 글자를 담는다")
outs["t2"] = S.table("어휘 수를 바꾸면", ["vocab_size", "실제 어휘 수", "토큰당 글자 (valid)", "예문을 나누면"], vrows,
                     ["어휘가 클수록 긴 조각을 통째로 갖고 있어서 토큰 수가 줄어요. 대신 Embedding 표(어휘 수 × 칸 수)가 커지고, 드물게 나오는 토큰은 학습할 기회가 적어요.",
                      "실제 어휘가 정한 수보다 적을 수 있어요. 두 번 넘게 나온 쌍만 합치게 했는데(<code>min_frequency=2</code>), 34만 자에서는 합칠 쌍이 먼저 떨어져요."])

# ---------------- ③
blk = ns["X_train"][0]
outs["s3"] = S.box("③ 학습용 묶음", [(f"X_train.shape → {tuple(ns['X_train'].shape)}, X_valid.shape → {tuple(ns['X_valid'].shape)}", 1),
                                    (f"X_train[0][:8] → {blk[:8].tolist()}", 2), f"  = {[tk.id_to_token(i) for i in blk[:8].tolist()]}",
                                    (f"입력 {[tk.id_to_token(i) for i in blk[:4].tolist()]} → 맞힐 것 {[tk.id_to_token(i) for i in blk[1:5].tolist()]}", 3, "newf")],
                   [f"train 글 전체를 한 줄로 이어 붙인 뒤 {ns['BLOCK']}토큰씩 잘라 {len(ns['X_train']):,}줄을 만들었어요. 문단이 줄 중간에서 끝나도 괜찮아요. <code>&lt;eos&gt;</code>가 \"여기서 문단이 끝났다\"를 알려 줘요.",
                    "한 줄은 토큰 번호 64개예요.",
                    "정답은 따로 만들지 않아요. 같은 줄을 한 칸씩 밀면 \"지금까지 본 토큰 → 다음 토큰\" 쌍이 64개 생겨요. 라벨이 필요 없어서(자기 지도 학습) 글만 있으면 얼마든지 학습할 수 있어요."])

# ---------------- ④
n_all = sum(p.numel() for p in gpt.parameters())
n_emb = gpt.transformer.wte.weight.numel() + gpt.transformer.wpe.weight.numel()
torch.manual_seed(0)
fresh = transformers.GPT2LMHeadModel(ns["config"]).eval()
torch.manual_seed(0)
raw = fresh.generate(torch.tensor([tk.encode(p0).ids]), max_new_tokens=20, do_sample=True, top_k=0, pad_token_id=ns["EOS"], eos_token_id=None)[0][len(tk.encode(p0).ids):].tolist()
S.claim(abs(ns["start_loss"] - math.log(V)) < 0.2, "④ 학습 전 손실이 ln(어휘 수)와 비슷하다")
outs["s4"] = S.box("④ 무작위에서 시작한 작은 GPT", [(f"가중치 수 → {n_all:,}개 (그중 토큰과 위치 Embedding {n_emb:,}개)", 1), (f"학습 전 valid 손실 → {ns['start_loss']:.3f} (ln {V} = {math.log(V):.3f})", 2, "newf"),
                                                  f"학습 전에 \"{p0}\" 뒤를 이어 쓰면 → {dec(raw)}"],
                   [f"GPT-2의 가장 작은 공개 모델(약 1억 2천만 개)의 {round(124e6 / n_all, -1):.0f}분의 1쯤이에요. 층 {ns['config'].n_layer}개, 칸 {ns['config'].n_embd}개, head {ns['config'].n_head}개예요.",
                    f"아무것도 모를 때 손실이 ln({V})과 비슷해요. 토큰 {V}개에 똑같은 확률(1/{V})을 주면 손실이 정확히 ln {V}이기 때문이에요. 학습이 시작점을 제대로 잡았는지 확인하는 기준이 돼요."])

# ---------------- ⑤
best = min(log, key=lambda r: r["valid"])
S.claim(all(a_["valid"] > b_["valid"] for a_, b_ in zip(log, log[1:])) and log[-1]["train"] < log[-1]["valid"], "⑤ valid 손실이 계속 내려가고, 마지막에는 train이 valid보다 낮다")
outs["s5"] = S.table(f"⑤ 학습 기록 ({ns['STEPS']} 걸음, 한 걸음에 32줄)", ["걸음", "train 손실", "valid 손실", "valid perplexity"],
                     [[r["step"], f3(r["train"]), f3(r["valid"]), f"{math.exp(r['valid']):.0f}"] for r in log],
                     [f"valid 손실이 {ns['start_loss']:.2f}에서 {log[-1]['valid']:.2f}까지 내려갔어요. <b>perplexity</b>는 e^손실로, \"다음 토큰을 평균 몇 개 중에서 헷갈리나\"로 읽어요. {V}개 중에 고르던 게 {math.exp(log[-1]['valid']):.0f}개쯤으로 줄었어요.",
                      f"train 손실({log[-1]['train']:.2f})이 valid보다 낮아지고 있어요. 34만 자는 GPT에게 아주 작은 말뭉치라 이 이상 학습하면 외우기 시작해요. 실제 GPT는 수천억 토큰으로 배워요."])
fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.plot([0] + [r["step"] for r in log], [ns["start_loss"]] + [r["train"] for r in log], marker="o", color=C["purple"], lw=2, label="train (그 걸음의 묶음)")
ax.plot([0] + [r["step"] for r in log], [ns["start_loss"]] + [r["valid"] for r in log], marker="o", color=C["pink"], lw=2, label="valid (처음 보는 문단)")
ax.axhline(math.log(V), color=C["gray"], ls="--", lw=1.2, label=f"ln {V} (아무것도 모를 때)")
ax.set(xlabel="학습 걸음", ylabel="다음 토큰 손실", title="다음 토큰 맞히기를 배우는 모습")
ax.legend(fontsize=9)
outs["f_loss"] = S.images("학습 손실", [(fig_save(fig, "loss_curve"), "")], ["처음 200걸음에 크게 떨어지고, 그 뒤로는 천천히 내려가요. 두 선이 벌어지는 만큼이 외운 정도예요."])
outs["f_loss"]["chart"] = True

# ---------------- ⑥
det = ["greedy", "beam 5", "beam 5 + 3-gram 반복 금지"]
S.claim(dt["beam 5 + 3-gram 반복 금지"]["반복"] == 0 and dt["beam 5"]["반복"] > dt["greedy"]["반복"] > 0, "⑥ beam이 greedy보다 더 반복하고, 반복 금지를 주면 0")
outs["s6"] = S.table(f"⑥ \"{p0}\" 뒤를 가장 그럴듯하게 이어 쓰면 (앞부분 6개 평균)", ["방법", "이어 쓴 글 (첫 앞부분)", "반복", "뜻밖의 정도"],
                     [[f"<b>{n}</b>", E(gen0[n]), f3(dt[n]["반복"]), f3(dt[n]["뜻밖의 정도"])] for n in det],
                     ["<b>반복</b>: 이어 쓴 글의 3-토큰 묶음 중 앞에서 이미 나온 것의 비율. <b>뜻밖의 정도</b>: 모델이 보기에 이어 쓴 토큰들이 평균 얼마나 뜻밖인가(토큰당 손실). 낮을수록 모델이 \"그럴듯하다\"고 보는 글이에요.",
                      f"beam은 greedy보다 더 그럴듯한 글({dt['beam 5']['뜻밖의 정도']:.2f})을 찾지만 반복도 더 많아요({dt['beam 5']['반복']:.2f}). 반복하는 글이 모델에게는 오히려 그럴듯하게 보여요. 한 번 나온 말을 다시 쓸 확률이 점점 커지는 고리에 빠져요.",
                      "3-토큰 묶음이 두 번 나오지 못하게 막으면 반복은 0이 되지만, 같은 단어를 조금씩 바꿔 되풀이하는 모습은 남아요."])

# ---------------- ⑦
samp = ["sampling t=1.0", "sampling t=0.7", "top-k 20", "top-p 0.9"]
S.claim(all(dt[n]["반복"] < 0.05 for n in samp) and dt["sampling t=0.7"]["뜻밖의 정도"] < dt["sampling t=1.0"]["뜻밖의 정도"] and dt["top-k 20"]["뜻밖의 정도"] < dt["sampling t=1.0"]["뜻밖의 정도"]
        and min(dt[n]["다양성"] for n in samp) > max(dt[n]["다양성"] for n in det), "⑦ 뽑기는 반복이 거의 없고 더 다양하며, t를 낮추거나 top-k로 자르면 덜 뜻밖이다")
outs["s7"] = S.table("⑦ 일곱 가지 방법 비교 (앞부분 6개, 30토큰씩)", ["방법", f"\"{p0}\" 뒤", "반복 ↓", "다양성 ↑", "뜻밖의 정도"],
                     [[f"<b>{n}</b>" if n in samp else n, E(gen0[n]), f3(dt[n]["반복"]), f3(dt[n]["다양성"]), f3(dt[n]["뜻밖의 정도"])] for n in ns["METHODS"]],
                     ["<b>다양성</b>: 모든 이어 쓰기에서 2-토큰 묶음 중 서로 다른 것의 비율. 같은 말을 되풀이하면 낮아요.",
                      f"확률대로 뽑으면 반복이 거의 사라지고 다양성이 1 가까이 올라요. 대신 뜻밖의 정도가 크게 올라요(t=1.0에서 {dt['sampling t=1.0']['뜻밖의 정도']:.2f}). 확률이 낮은 엉뚱한 토큰도 가끔 뽑히고, 한 번 엉뚱해지면 그 뒤가 계속 흐트러져요.",
                      f"temperature를 낮추거나(t=0.7, {dt['sampling t=0.7']['뜻밖의 정도']:.2f}) 상위 k개만 남기면(top-k 20, {dt['top-k 20']['뜻밖의 정도']:.2f}) 엉뚱한 꼬리가 잘려서 덜 뜻밖이면서 반복도 거의 없어요. 대화나 글쓰기에서 흔히 top-p 0.9 안팎, temperature 0.7 안팎에서 시작하는 이유예요.",
                      "모델이 작아서 문장의 뜻은 어느 방법이든 엉성해요. 여기서 볼 건 방법마다의 <b>성질</b>이에요. 큰 모델에서도 같은 성질(beam의 반복, 샘플링의 엉뚱함)이 나타난다고 알려져 있어요."])

# ---------------- ⑧
summ = ns["summaries"]
oracle = sum(max(ns["rouge"](x, s["정답"])["ROUGE-1"]["F1"] for x in pg["sentences"]) for pg, s in zip(ns["pages"], summ)) / len(summ)
ex_s = next(s for s in summ if s["page"] == "embedding")
r_lo, r_hi = sorted(rt[w]["ROUGE-1"] for w in rt)
re_ = {w: ns["rouge"](ex_s[w], ex_s["정답"])["ROUGE-1"] for w in ("첫 문장", "대표 문장")}
S.claim(abs(rt["첫 문장"]["ROUGE-1"] - rt["대표 문장"]["ROUGE-1"]) < 0.03 and min(rt[w]["ROUGE-1"] for w in rt) > 0.1, "⑧ 두 추출 방법의 ROUGE-1은 0.1~0.3이고 차이가 거의 없다")
S.claim(max(rt[w]["ROUGE-1"] for w in rt) < 0.3 and oracle < 0.45 and oracle > max(rt[w]["ROUGE-1"] for w in rt), "⑧ 한 문장 고르기의 ROUGE-1은 0.3 아래, 정답을 보고 골라도 0.45 아래")
outs["s8"] = S.box("⑧ 주제 페이지 26쪽의 '한 문장'(사람이 쓴 요약)과 비교", [f"예: embedding 쪽", f"  정답 요약 → {ex_s['정답']}", f"  첫 문장 → {ex_s['첫 문장']}", f"  대표 문장 → {ex_s['대표 문장']}",
                                                               *(f"  {w}의 ROUGE-1 → 재현율 {v['재현율']:.2f}, 정밀도 {v['정밀도']:.2f}, F1 {v['F1']:.2f}" for w, v in re_.items()), "",
                                                               (f"26쪽 평균 ROUGE-1 F1 → 첫 문장 {rt['첫 문장']['ROUGE-1']:.3f}, 대표 문장 {rt['대표 문장']['ROUGE-1']:.3f}", 1),
                                                               (f"26쪽 평균 ROUGE-L F1 → 첫 문장 {rt['첫 문장']['ROUGE-L']:.3f}, 대표 문장 {rt['대표 문장']['ROUGE-L']:.3f}", 2),
                                                               (f"정답을 보고 ROUGE-1이 가장 높은 문장을 골라도 → 평균 {oracle:.3f}", 3, "bad")],
                   [f"본문에서 문장을 그대로 고르는 <b>추출 요약</b>은 첫 문장이든, 본문 전체와 뜻이 가장 가까운 대표 문장이든 ROUGE-1 F1이 {r_lo:.2f}~{r_hi:.2f}{J(f'{r_hi:.2f}', '이에요', '예요')}. 사람이 쓴 요약과 겹치는 조각이 적고, 두 방법의 차이도 거의 없어요.",
                    "ROUGE-L은 겹치는 조각이 <b>순서까지</b> 같은 만큼이에요.",
                    f"정답을 보고 가장 잘 겹치는 문장을 골라도 {oracle:.2f}에 그쳐요. 사람의 요약은 여러 문장의 뜻을 새 문장으로 묶어 쓴 <b>생성 요약</b>이라, 본문 어디에도 그런 문장이 없어요. 그래서 요약에는 글을 새로 쓰는 인코더-디코더 모델(BART, PEGASUS, T5)을 써요."])

flow = [[1, "말뭉치", "중복 제거", f"{len(ns['paras']):,} → {len(ns['unique']):,}문단"], [2, "Tokenizer", "BPE 4천 개", f"토큰당 {cpt['우리 Tokenizer (4천 개)']:.2f}자"], [3, "학습 묶음", "이어 붙여 자르기", f"{tuple(ns['X_train'].shape)}"],
        [4, "작은 GPT", "무작위 시작", f"손실 {ns['start_loss']:.2f}"], [5, "학습", "다음 토큰", f"손실 {log[-1]['valid']:.2f}"], [6, "그럴듯하게", "greedy, beam", "반복"], [7, "뽑기", "temperature, top-k, top-p", "다양성"], [8, "요약 재기", "ROUGE", "추출의 한계"]]
S.save(ROOT / "topics" / "model" / "gpt-generation-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>GPT 같은 디코더 모델은 <a href="llm-principles.html">Transformer 구조</a> ⑧에서 본 대로 \"지금까지의 글 → 다음 토큰의 확률\"을 내요. 글을 만들려면 그 확률에서 토큰 하나를 고르고, 입력 끝에 붙이고, 다시 계산하기를 되풀이해요. 이 쪽은 두 가지를 직접 해 봐요.</p>
<ol><li><b>모델 만들기</b>: 미리 학습된 모델 없이, 글만 가지고 Tokenizer와 GPT를 처음부터 학습해요. 큰 LLM이 만들어지는 과정을 아주 작게 따라가는 거예요.</li>
<li><b>고르기</b>: 같은 모델로 다음 토큰을 고르는 방법을 바꿔 가며, 만든 글이 어떻게 달라지는지 숫자로 재요.</li></ol>
<p>학습할 글은 이 노트의 옛 Day 노트예요.</p>
{K.out("data")}
<p>다 만든 모델로 미리 해 보면, 고르는 방법만 바꿔도 이렇게 달라요.</p>
{K.out("why")}"""
whole = """<p>말뭉치 정리 → Tokenizer 학습 → 같은 길이로 자르기 → 무작위 GPT → 다음 토큰 맞히기 학습 → 그럴듯하게 고르기 → 확률대로 뽑기 → 만든 글을 숫자로 재기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "말뭉치 읽고 정리하기", "Day별 글을 문단으로 펴고, 똑같은 문단은 하나만 남긴 뒤, 20개 중 1개를 valid로 떼어 둬요.",
           "<p><b>왜 정리부터?</b> 모델은 말뭉치에 있는 것을 그대로 닮아요. 잡음이 많으면 잡음을, 같은 문장이 많으면 그 문장을 외워요. 큰 모델을 만들 때도 말뭉치 정리(중복 제거, 품질 거르기, 개인정보 지우기)에 가장 많은 공을 들여요. 웹에서 모은 글은 같은 페이지가 수없이 겹쳐 있어요.</p>"
           + K.code_table([("Counter(paras)", "문단마다 몇 번 나왔는지"), ("list(dict.fromkeys(paras))", "순서를 지키며 중복 빼기 (dict는 처음 넣은 순서를 기억해요)"), ("i % 20 != 0", "20개 중 19개")]), "s1"),
    K.step(2, "Tokenizer 학습하기", "train 글에서 자주 붙어 나오는 조각을 합쳐 가며 어휘 4,000개를 만들어요(BPE).",
           "<p><b>왜 직접 만드나?</b> 미리 학습된 모델을 쓸 때는 그 모델의 Tokenizer를 그대로 써야 해요. 하지만 처음부터 만든다면, 우리 글에 맞는 Tokenizer가 같은 글을 더 적은 토큰으로 나눠요. 코드 생성 모델이 코드로 Tokenizer를 따로 학습하는 이유도 같아요(들여쓰기, 연산자가 한 토큰이 되게).</p>"
           "<p><b>BPE</b>(Byte Pair Encoding)는 글자 하나하나에서 시작해, 가장 자주 붙어 나오는 쌍(예: \"있\"+\"어요\")을 하나의 새 토큰으로 합치기를 되풀이해요. 정한 어휘 수가 차면 멈춰요. 띄어쓰기는 <code>▁</code>로 바꿔 글자처럼 다루는 SentencePiece 방식(Metaspace)을 썼어요.</p>"
           + K.code_table([("Tokenizer(models.BPE(unk_token=\"<unk>\"))", "빈 BPE Tokenizer (tokenizers 라이브러리)"), ("normalizers.NFC()", "한글 자모가 따로 저장된 글자를 하나로 합치는 유니코드 정리"),
                           ("pre_tokenizers.Metaspace()", "띄어쓰기를 ▁로"), ("trainers.BpeTrainer(vocab_size=..., special_tokens=[...])", "어휘 수와 특별 토큰"), ("train_from_iterator(texts, trainer)", "글 목록으로 학습")])
           + K.try_("어휘 수를 1천, 1만 6천으로 바꾸면?", "t2"), "s2"),
    K.step(3, "학습 데이터 만들기: 이어 붙이고 같은 길이로 자르기", "문단마다 끝에 <eos>를 붙여 전부 한 줄로 이은 뒤, 64토큰씩 잘라 표로 만들어요.",
           "<p><b>왜 이어 붙이나?</b> 문단 길이가 제각각이라 하나씩 넣으면 짧은 문단 뒤를 채워야 해서(padding) 계산이 낭비돼요. 다 이어 붙여 같은 길이로 자르면 채우는 자리가 없어요. 큰 모델도 이렇게 해요.</p>"
           + K.code_table([("tk.encode(t).ids + [EOS]", "문단을 번호로 바꾸고 끝 표시"), ("ids[: len(ids) // BLOCK * BLOCK]", "64로 나눠떨어지게 남는 꼬리 자르기"), (".view(-1, BLOCK)", "(줄 수, 64) 표로")]), "s3"),
    K.step(4, "작은 GPT 만들기 (무작위에서 시작)", "GPT-2와 같은 구조를 아주 작게 설정해, 가중치를 무작위로 채운 모델을 만들어요.",
           "<p>Hub에서 이름으로 받으면 미리 학습된 가중치까지 오지만, <code>GPT2Config</code>로 설정만 만들어 모델에 넣으면 <b>구조만 같고 가중치는 무작위</b>인 모델이 돼요. 층 수, 칸 수, head 수, 어휘 수, 한 번에 보는 토큰 수를 직접 정해요.</p>"
           "<p>GPT는 <b>다음 토큰 맞히기</b>(CLM, Causal Language Modeling)로 배워요. BERT, e5 같은 인코더는 가운데를 가리고 맞히기(MLM)로, T5 같은 인코더-디코더는 입력을 받아 다른 글 만들기로 배워요. 무엇을 만들 모델이냐에 따라 목표를 골라요.</p>"
           + K.code_table([("GPT2Config(vocab_size=..., n_positions=64, n_embd=128, n_layer=4, n_head=4)", "어휘 수, 최대 길이, 칸 수, 층 수, head 수"), ("GPT2LMHeadModel(config)", "설정대로 무작위 가중치 모델 (LM head: 마지막 벡터 → 어휘 수만큼 점수)"),
                           ("gpt(b, labels=b).loss", "다음 토큰 맞히기 손실. 한 칸 밀어 비교하는 건 모델 안에서 해요")]), "s4"),
    K.step(5, "학습하기: 다음 토큰 맞히기", "train 줄에서 무작위로 32줄씩 뽑아 1,000걸음 학습하고, 200걸음마다 valid 손실을 재요.",
           "<p>학습률은 처음 10% 동안 키웠다가 끝까지 천천히 줄여요(OneCycle). 처음부터 크게 움직이면 무작위 상태에서 튀기 쉽고, 끝에서는 작게 움직여야 바닥에 자리 잡아요. 기울기가 너무 크면 1로 잘라요(<code>clip_grad_norm_</code>). Transformer 학습에서 흔한 안정 장치예요.</p>"
           "<p>큰 모델은 GPU 여러 대에 같은 모델을 하나씩 두고 데이터를 나눠 계산한 뒤 기울기를 모아 합쳐요(분산 데이터 병렬, DDP). 하는 일은 이 반복과 같아요.</p>"
           + K.out("f_loss")
           + K.code_table([("torch.optim.lr_scheduler.OneCycleLR(...)", "걸음마다 학습률을 키웠다 줄이기"), ("torch.nn.utils.clip_grad_norm_(..., 1.0)", "기울기 전체의 크기가 1을 넘으면 1로 줄이기"),
                           ("X_train[torch.randint(0, len(X_train), (32,))]", "무작위로 32줄 뽑기")]), "s5"),
    K.step(6, "고르는 방법 1: 가장 그럴듯한 것 고르기 (greedy, beam)", "앞부분 6개에 이어 30토큰을 쓰게 하고, 매번 확률이 가장 높은 토큰을 고르는 방법과 후보 5개를 함께 따라가는 방법을 비교해요.",
           "<p><b>greedy</b>는 매 걸음 확률 1등 토큰을 골라요. 빠르지만 지금 1등이 전체 글로 보면 최선이 아닐 수 있어요. <b>beam search</b>는 지금까지 이어 쓴 후보 B개(여기서는 5)를 함께 들고 가다가, 걸음마다 후보들의 다음 토큰을 모두 펼쳐 \"전체 확률(log 확률의 합)\"이 높은 B개만 남겨요. 마지막에 가장 높은 하나를 골라요.</p>"
           "<p><b>왜 반복하나?</b> 한 번 나온 말이 다시 나오면, 모델은 앞에서 본 걸 참고해 같은 말을 또 높게 쳐요. 확률만 좇으면 이 고리에서 못 빠져나와요. 그래서 같은 n-토큰 묶음이 두 번 나오지 못하게 막는 옵션(<code>no_repeat_ngram_size</code>)을 같이 써요.</p>"
           + K.code_table([("gpt.generate(inp, max_new_tokens=30, do_sample=False)", "greedy"), ("num_beams=5", "후보 5개를 함께 따라가기"), ("no_repeat_ngram_size=3", "3-토큰 묶음 반복 금지"),
                           ("logits.log_softmax(-1).gather(...)", "실제로 고른 토큰의 log 확률 꺼내기 (뜻밖의 정도 계산)")]), "s6"),
    K.step(7, "고르는 방법 2: 확률대로 뽑기 (temperature, top-k, top-p)", "다음 토큰의 확률대로 무작위로 뽑되, temperature로 확률을 뾰족하게 하거나 상위 후보만 남겨 뽑아요.",
           "<p><b>sampling</b>은 확률이 0.3인 토큰은 30% 확률로 뽑아요. 같은 앞부분에서도 매번 다른 글이 나와요. temperature는 <a href=\"llm-principles.html\">Transformer 구조</a> ⑧에서 본 것처럼 점수를 t로 나누는 값이에요. 여기서는 그걸 실제 생성 글에 써 봐요.</p>"
           "<p>문제는 확률이 아주 낮은 토큰 수천 개를 다 합치면 꽤 큰 몫이 된다는 거예요. 그 꼬리에서 하나만 뽑혀도 글이 엉뚱해져요. <b>top-k</b>는 상위 k개만 남기고, <b>top-p</b>(nucleus)는 확률을 큰 순서로 더해 p(예: 0.9)를 넘을 때까지만 남겨요. top-p는 확신할 때는 후보가 몇 개뿐이고, 헷갈릴 때는 많아져서 상황에 맞게 바뀌어요.</p>"
           + K.code_table([("do_sample=True, top_k=0, temperature=0.7", "자르지 않고 t=0.7로 뽑기 (<code>top_k=0</code>: 자르기 끔)"), ("top_k=20", "상위 20개 중에서 뽑기"), ("top_p=0.9", "누적 확률 0.9까지의 후보에서 뽑기")]), "s7"),
    K.step(8, "요약과 평가: 본문에서 대표 문장 고르기와 ROUGE", "주제 페이지마다 본문의 첫 문장과, 본문 전체와 뜻이 가장 가까운 문장을 골라 사람이 쓴 '한 문장' 요약과 ROUGE로 비교해요.",
           "<p>만든 글이 좋은지는 결국 사람이 봐야 하지만, 매번 그럴 수는 없어서 정답 글과 <b>겹치는 조각</b>으로 재요. <b>BLEU</b>는 만든 글의 조각 중 정답에 있는 비율(정밀도)을 중심으로 재서 번역에 쓰고, <b>ROUGE</b>는 정답의 조각 중 만든 글에 있는 비율(재현율)을 중심으로 재서 요약에 써요. 요약은 \"중요한 내용을 빠뜨리지 않았나\"가 중요하니까요. ROUGE-L은 순서까지 맞는 가장 긴 공통 부분으로 재요.</p>"
           "<p>한계도 분명해요. 같은 뜻을 다른 말로 쓰면 0점에 가깝고, 말이 겹쳐도 틀린 내용일 수 있어요. 그래서 숫자와 함께 사람이 직접 읽어 봐요.</p>"
           + K.code_table([("Counter(c) & Counter(r)", "두 조각 목록에서 겹치는 개수 (적은 쪽 기준)"), ("lcs[i + 1][j + 1] = ...", "가장 긴 공통 부분 순서 길이를 표로 채우기 (ROUGE-L)"),
                           ("v @ F.normalize(v.mean(0), dim=0)", "문장마다 본문 평균 벡터와의 코사인 유사도")]), "s8"),
]
practice = [("실제 LLM을 처음부터 만든다면", "수천억에서 수조 토큰의 말뭉치, 수십억 개 이상의 가중치, GPU 수백에서 수천 대가 필요해요. 그래서 대부분은 공개된 사전학습 모델을 받아 미세조정하거나, API로 써요. 처음부터 만드는 건 분야가 아주 특수하거나(예: 코드, 특정 언어) 데이터를 밖에 낼 수 없을 때예요."),
            ("말뭉치가 너무 커서 디스크에 안 들어간다", "<code>load_dataset(..., streaming=True)</code>로 한 줄씩 흘려 읽으며 학습해요. 디스크에 다 받지 않아도 돼요."),
            ("디코딩 설정을 고를 때", "사실을 정확히 답해야 하면 temperature를 낮게(0~0.3) 또는 greedy, 글쓰기와 대화는 top-p 0.9, temperature 0.7 안팎에서 시작해요. 반복이 보이면 반복 벌점(<code>repetition_penalty</code>)을 줘요."),
            ("API의 temperature", "LLM API의 <code>temperature</code>, <code>top_p</code>가 바로 ⑦의 값이에요 (<a href=\"../backend/llm-client.html\">LLM API 클라이언트</a>). 같은 질문에 답이 매번 다른 이유예요."),
            ("요약 모델", "기사 요약은 BART, PEGASUS(문장을 통째로 가리고 되살리기로 미리 학습해 요약에 강함), 여러 과제를 \"글 → 글\"로 통일한 T5가 대표예요. 대화 요약처럼 말투가 다른 글은 그 분야 데이터로 미세조정해야 해요."),
            ("생성 결과를 재는 방법", "요약은 ROUGE, 번역은 BLEU, 언어 모델 자체는 perplexity를 많이 써요. 숫자는 비교용이고, 최종 판단은 사람이 읽어 보거나, 기준을 정해 다른 LLM에게 채점시키기도 해요.")]
libs = [("①", "<code>Counter</code>, <code>dict.fromkeys</code>", "중복 세기, 빼기"), ("②", "<code>tokenizers</code>: <code>Tokenizer</code>, <code>models.BPE</code>, <code>BpeTrainer</code>, <code>Metaspace</code>", "Tokenizer 학습"),
        ("④⑤", "<code>GPT2Config</code>, <code>GPT2LMHeadModel</code>, <code>OneCycleLR</code>, <code>clip_grad_norm_</code>", "작은 GPT와 학습"), ("⑥⑦", "<code>generate(num_beams, no_repeat_ngram_size, do_sample, temperature, top_k, top_p)</code>", "디코딩"),
        ("⑧", "<code>AutoModel</code>(e5), <code>Counter &amp;</code>", "대표 문장, ROUGE")]
K.build(ROOT / "topics" / "model" / "gpt-generation.html", title="작은 GPT와 디코딩", h1="작은 GPT를 처음부터: 학습하고, 고르고, 재기", track="모델", crumb="모델 &gt; 순서 모델과 Transformer",
        one_line="GPT는 글만 있으면 <b>다음 토큰 맞히기</b>로 스스로 배워요. 다 배운 뒤에도 다음 토큰을 <b>어떻게 고르느냐</b>에 따라 글이 달라져요. 가장 그럴듯한 것만 고르면 같은 말을 되풀이하고, 확률대로 뽑으면 다양하지만 엉뚱해져서, 상위 후보만 남겨 뽑는 방법을 많이 써요.",
        why=why, whole=whole, steps=steps, code_file="examples/gpt/tiny_gpt.py", practice=practice, libs=libs,
        libs_note="<code>train_tokenizer</code>, <code>to_blocks</code>, <code>valid_loss</code>, <code>generate</code>, <code>repeat_rate</code>, <code>avg_nll</code>, <code>distinct2</code>, <code>embed</code>, <code>pieces</code>, <code>rouge</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"llm-principles.html\">Transformer 구조</a>", "디코더와 다음 토큰"), ("<a href=\"../backend/llm-client.html\">LLM API 클라이언트</a>", "API의 temperature, top_p"),
                   ("<a href=\"text-classification.html\">문장 분류와 미세조정</a>", "미리 학습된 모델 빌려 쓰기")],
        sources=["함께 정리한 자료: Lewis Tunstall 외, <i>Natural Language Processing with Transformers</i> 5장(텍스트 생성: greedy, beam, n-gram 반복 금지, sampling, temperature, top-k, top-p), 10장(밑바닥부터 학습: 말뭉치 모으기와 정리, 스트리밍, Tokenizer 학습과 성능, 사전학습 목표 CLM, MLM, seq2seq, 설정으로 모델 초기화, 이어 붙여 자르기, DDP), 6장(요약: 추출과 생성, T5, BART, PEGASUS, BLEU와 ROUGE). 부트캠프 수업에서 따로 다루지 않은 주제라 예제는 새로 만들었어요",
                 "데이터: <code>examples/gpt/data/notes_corpus.jsonl</code>(이 저장소의 <code>all-notes.md</code>에서 만듦), <code>pages.jsonl</code>(주제 페이지의 '한 문장'과 본문). 둘 다 <code>tools/make_gpt_corpus.py</code>",
                 "결과 값과 그림: <code>tools/pages/gpt_gen.py</code>가 <code>examples/gpt/tiny_gpt.py</code>를 단계별로 실제 실행한 결과 (CPU, seed 0). ⑧의 문장 벡터와 ROUGE 조각은 <code>intfloat/multilingual-e5-small</code>",
                 "공식 문서: <a href=\"https://huggingface.co/docs/tokenizers/quicktour\" target=\"_blank\">tokenizers</a>, <a href=\"https://huggingface.co/docs/transformers/generation_strategies\" target=\"_blank\">Generation strategies</a>, <a href=\"https://huggingface.co/docs/transformers/model_doc/gpt2\" target=\"_blank\">GPT-2</a>"])
print("ok")
