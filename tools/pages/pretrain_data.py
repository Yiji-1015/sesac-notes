"""모델 > LLM의 학습 데이터 만들기. examples/pretrain_data/mix_tokenizer.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/pretrain_data.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import html  # noqa: E402
import numpy as np  # noqa: E402
import tokenizers  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "pretrain_data"
code = S.parse((EX / "mix_tokenizer.py").read_text(encoding="utf-8"))
import shutil  # noqa: E402
import tempfile  # noqa: E402
_tmp = Path(tempfile.mkdtemp(prefix="stepcode_"))                   # ① 이 ../gpt/data 를 읽어서 두 폴더를 나란히 복사
work = _tmp / "pretrain_data"
shutil.copytree(EX, work, ignore=shutil.ignore_patterns("__pycache__"))
(_tmp / "gpt" / "data").mkdir(parents=True)
shutil.copy(ROOT / "examples" / "gpt" / "data" / "notes_corpus.jsonl", _tmp / "gpt" / "data")
ns = S.run(code, 7, work, filename="mix_tokenizer.py")
ENV = f"Python {sys.version.split()[0]} (표준 라이브러리가 영어와 코드 말뭉치), tokenizers {tokenizers.__version__}, NumPy {np.__version__} (seed 0)"
f2 = lambda x: f"{float(x):.2f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "pretrain"
IMG.mkdir(parents=True, exist_ok=True)
DOM = ns["DOMAINS"]
COL = {"한국어": C["purple"], "영어": C["blue"], "코드": C["pink"]}


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/pretrain/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


def esc(t, n=110):
    t = t if len(t) <= n else t[:n] + "…"
    return html.escape(t)


outs = {}
raw, flt, exa, cln = ns["raw_size"], ns["filtered_size"], ns["exact_size"], ns["clean_size"]
# ---------------- ①
outs["s1"] = S.table("① 모은 말뭉치", ["분야", "어디서", "문서(문단) 수", "글자 수"],
                     [["한국어", "옛 Day 노트의 줄", f"{raw['한국어'][0]:,}", f"{raw['한국어'][1]:,}"], ["영어", f"표준 라이브러리 {len(ns['EN_MODULES'])}개 모듈의 설명문", f"{raw['영어'][0]:,}", f"{raw['영어'][1]:,}"],
                      ["코드", f"표준 라이브러리 {len(ns['CODE_MODULES'])}개 모듈의 소스", f"{raw['코드'][0]:,}", f"{raw['코드'][1]:,}"]],
                     ["진짜 LLM은 웹 문서, PDF, 코드, 수학을 몇 조 토큰 모아요. 여기는 이 PC에 있는 글로 같은 일을 작게 해요. 영어가 한국어의 세 배쯤이에요."])

# ---------------- ②
dr = ns["dropped"]
S.claim(dr[("한국어", "글자 비율 60% 미만")] > 50, "② 글자 비율 규칙이 한국어 노트를 꽤 버림")
outs["s2"] = S.table("② 규칙으로 버린 것", ["분야", "규칙", "버린 수", "버린 것의 예"],
                     [[k, r, f"{n:,}", esc(ns["drop_examples"][(k, r)]) or "(빈 문단)"] for (k, r), n in sorted(dr.items(), key=lambda x: (DOM.index(x[0][0]), -x[1]))],
                     [f"짧은 조각, 기호와 숫자 위주 조각, 같은 단어가 되풀이되는 조각을 버렸어요. 한국어는 {raw['한국어'][0]:,}개에서 {flt['한국어'][0]:,}개, 영어는 {raw['영어'][0]:,}개에서 {flt['영어'][0]:,}개가 남았어요.",
                      f"그런데 한국어에서 \"글자 비율 60% 미만\"으로 버린 {dr[('한국어', '글자 비율 60% 미만')]}개 중에는 위 예처럼 식이 섞인 좋은 설명도 있어요. 한국어는 띄어쓰기와 문장 부호가 많아 같은 기준이 더 엄해요. 규칙은 분야마다 따로 맞추고, 버린 것을 꼭 직접 읽어 봐야 해요."])
ko_rows = []
ko_all = [p.strip() for p in ns["ko"]]
for th in (0.4, 0.5, 0.6, 0.7):
    n = sum(1 for t in ko_all if len(t) >= 40 and sum(ch.isalpha() for ch in t) / len(t) < th)
    ko_rows.append([f"<b>{th}</b> (지금)" if th == 0.6 else th, n])
outs["t2"] = S.table("한국어에서 글자 비율 기준을 바꾸면", ["기준 (이보다 낮으면 버림)", "버리는 한국어 문단 수 (40자 이상 중)"], ko_rows,
                     ["기준을 조금만 올려도 버리는 수가 크게 늘어요. 몇 조 토큰짜리 말뭉치에서는 기준 하나가 수십억 토큰을 가르기 때문에, Olmo 3 같은 공개 모델은 기준마다 작은 모델을 학습해 보고 정해요."])

# ---------------- ③
S.claim(ns["top_dups"][0][1] > 100, "③ 가장 흔한 영어 문단이 100번 넘게")
outs["s3"] = S.box("③ 똑같은 문단 빼기", [*[(f"\"{esc(t, 70)}\" → {n}번", i + 1, "newf" if i == 0 else None) for i, (t, n) in enumerate(ns["top_dups"])],
                                      (f"영어 {flt['영어'][0]:,}개 → {exa['영어'][0]:,}개, 한국어 {flt['한국어'][0]:,}개 → {exa['한국어'][0]:,}개, 코드 {flt['코드'][0]:,}개 → {exa['코드'][0]:,}개", 4)],
                   [f"영어는 {1 - exa['영어'][0] / flt['영어'][0]:.0%}가 똑같은 문단이었어요. 클래스마다 물려받은 기본 설명문이 되풀이되기 때문이에요. 웹 문서도 머리말, 꼬리말, 복사한 글 때문에 이렇게 겹쳐요.",
                    "같은 글을 여러 번 보면 모델이 그 글을 외우고(개인정보나 저작권 문제), 같은 계산량으로 볼 수 있는 새 글이 줄어요."])

# ---------------- ④
near = ns["near"]
en_exact = ns["en_exact"]
S.claim(len(near) > 5 and near[-1][0] > 0.9, "④ 거의 같은 짝이 몇 개 있고 가장 닮은 짝은 0.9 넘게")
opp = [x for x in near if en_exact[x[1]].split()[0] != en_exact[x[2]].split()[0]]   # 첫 단어가 다른 짝 (뜻이 다를 수 있음)
S.claim(len(opp) > 0, "④ 첫 단어가 다른데 거의 같다고 본 짝이 있음")
show = [opp[-1], near[-1], near[0]]
outs["s4"] = S.table("④ MinHash가 찾은 거의 같은 짝 (영어)", ["닮은 정도", "앞 문단", "뒤 문단 (뺌)"], [[f2(s), esc(en_exact[i], 120), esc(en_exact[j], 120)] for s, i, j in show],
                     [f"영어 {len(en_exact):,}개 문단의 모든 짝({len(en_exact) * (len(en_exact) - 1) // 2:,}개)을 비교하는 대신, 지문의 한 묶음이라도 같은 {ns['n_cand']}짝만 후보로 봤어요. 그중 닮은 정도 0.8 이상 {len(near)}짝을 찾아 {len(ns['drop'])}개를 뺐어요.",
                      f"첫 줄을 보세요. {en_exact[opp[-1][1]].split()[0]}, {en_exact[opp[-1][2]].split()[0]}처럼 첫 단어만 달라 하는 일이 다른 설명인데, 나머지 글이 거의 같아서 지워졌어요. 거의 같은 것 빼기는 글의 모양만 봐요. 기준을 낮출수록 이런 손해가 늘어요."])
th_rows = []
for th in (0.5, 0.7, 0.8, 0.9):
    pairs, _ = ns["near_duplicates"](en_exact, threshold=th)
    th_rows.append([f"<b>{th}</b> (지금)" if th == 0.8 else th, len(pairs), len({j for _, _, j in pairs})])
outs["t4"] = S.table("닮은 정도 기준을 바꾸면", ["기준", "찾은 짝", "빼는 문단"], th_rows,
                     ["기준을 낮추면 더 많이 빼요. 많이 뺄수록 말뭉치는 깨끗해지지만 다른 뜻의 글도 같이 사라져요."])

# ---------------- ⑤
cn = ns["cpt_natural"]
S.claim(cn[0] < cn[1] and cn[0] < cn[2], "⑤ 가진 만큼 섞으면 한국어 압축률이 가장 낮음")
p_nat, p_lit = ns["tok_natural"].encode(ns["SAMPLE"]), ns["tok_little_ko"].encode(ns["SAMPLE"])
from tokenizers import decoders  # noqa: E402
for _t in (ns["tok_natural"], ns["tok_little_ko"]):
    _t.decoder = decoders.ByteLevel()                             # 바이트 조각을 글자로 되돌려 보여 주기
def pieces(tok, enc):
    return " | ".join(html.escape(tok.decode([i])) for i in enc.ids)
outs["s5"] = S.box("⑤ 가진 만큼 섞은 토크나이저 (어휘 4,000개)", [(f"섞은 비율: " + ", ".join(f"{k} {p:.0%}" for k, p in zip(DOM, ns["natural"])), 1),
                                                          (f"토큰 하나에 담긴 글자 수: " + ", ".join(f"{k} {c:.2f}" for k, c in zip(DOM, cn)), 2, "newf"),
                                                          (f"\"{ns['SAMPLE']}\" → {len(p_nat.ids)}토큰", 3), (f"한국어를 2%만 섞은 토크나이저 → {len(p_lit.ids)}토큰", 4)],
                   [f"같은 어휘 크기라도 한국어는 토큰 하나에 {cn[0]:.2f}글자, 영어는 {cn[1]:.2f}글자를 담아요. 한국어 글은 토큰이 더 많이 들어서 API 비용이 더 들고, 같은 문맥 길이에 더 적게 들어가요.",
                    "한국어를 조금만 섞으면 한글 한 글자도 바이트 조각으로 쪼개져요(아래 표의 �는 글자의 일부 바이트예요)."])
outs["t5"] = S.table("토큰으로 쪼갠 모양", ["토크나이저", "토큰"], [["가진 만큼 섞기", pieces(ns["tok_natural"], p_nat)], ["한국어 2%", pieces(ns["tok_little_ko"], p_lit)]], [])
voc_rows = []
for v in (1000, 4000, 16000):
    c = cn if v == 4000 else ns["chars_per_token"](ns["train_tokenizer"](ns["natural"], vocab=v))
    voc_rows.append([f"<b>{v:,}</b> (지금)" if v == 4000 else f"{v:,}", *[f2(x) for x in c]])
outs["t5b"] = S.table("어휘 크기를 바꾸면", ["어휘 수", *[f"{k} 글자/토큰" for k in DOM]], voc_rows,
                      ["어휘가 클수록 긴 조각을 하나로 담아 압축률이 올라가요. 대신 임베딩 표가 커지고, 드문 토큰은 학습이 덜 돼요. 실제 LLM은 3만에서 25만 개를 써요."])

# ---------------- ⑥
MIX, proxy, full = ns["MIXES"], ns["proxy"], ns["full"]
rk = ns["rank_kept"]
S.claim(min(rk.values()) > 0.9, "⑥ 대리 토크나이저와 실제 크기의 순위 상관이 모두 0.9 넘게")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
for j, k in enumerate(DOM):
    axs[j].scatter(MIX[:, j], full[:, j], color=COL[k], s=26, label="실제 크기 (어휘 4,000, 40만 자)")
    axs[j].scatter(MIX[:, j], proxy[:, j], color=COL[k], s=26, facecolors="none", label="대리 (어휘 1,000, 10만 자)")
    axs[j].set(xlabel=f"학습 글 중 {k} 비율", ylabel="글자/토큰", title=k)
axs[0].legend(fontsize=8, loc="lower right")
P_MIX = fig_save(fig, "mixture_compression")
outs["f_mix"] = chart("섞는 비율에 따른 압축률", P_MIX, "",
                      [f"점 하나가 비율 조합 하나예요({len(MIX)}가지). 그 분야를 많이 섞을수록 그 분야의 압축률이 올라가요. 비율이 아주 작은 쪽에서 가장 가파르게 변해요.",
                       "속이 빈 점(작은 대리 토크나이저)은 값은 낮지만 모양이 같아요. 그래서 조합끼리의 순위가 거의 그대로예요."])
outs["s6"] = S.table("⑥ 대리 토크나이저의 순위가 실제 크기에서도 그대로인가", ["분야", "순위 상관 (Spearman)"], [[k, f2(v)] for k, v in rk.items()],
                     ["1이면 순위가 완전히 같아요. 작고 싼 토크나이저로 비율 조합을 많이 시험해 보고, 결과의 순위를 큰 토크나이저에도 믿고 쓸 수 있다는 뜻이에요(TReX 논문의 순위 불변성과 같은 결론)."])

# ---------------- ⑦
vd = ns["verdict"]
tp = {n: ns["tokens_per_1k"](c) for n, (m, c) in vd.items()}
S.claim(tp["가진 만큼"] > tp["서비스 비율 그대로"] and tp["가진 만큼"] > tp["회귀가 고른 비율"], "⑦ 가진 만큼 섞기가 가장 토큰을 많이 씀")
outs["s7"] = S.table("⑦ 서비스 글(한국어 60%, 영어 20%, 코드 20%) 1,000자에 드는 토큰", ["섞은 비율", "한국어 : 영어 : 코드", *[f"{k} 글자/토큰" for k in DOM], "1,000자에 토큰"],
                     [[n, " : ".join(f"{x:.2f}" for x in m), *[f2(x) for x in c], f"{tp[n]:.1f}"] for n, (m, c) in vd.items()],
                     [f"가진 만큼 섞으면 {tp['가진 만큼']:.1f}토큰, 서비스 비율대로 섞으면 {tp['서비스 비율 그대로']:.1f}토큰이에요. {1 - tp['서비스 비율 그대로'] / tp['가진 만큼']:.0%}가 줄었어요. 같은 비용으로 그만큼 더 많은 글을 넣고 학습할 수 있어요.",
                      f"회귀가 고른 비율({' : '.join(f'{x:.2f}' for x in ns['best_mix'])})은 {tp['회귀가 고른 비율']:.1f}토큰으로, 서비스 비율 그대로({tp['서비스 비율 그대로']:.1f})보다 조금 나빴어요. 회귀가 대리 토크나이저의 모양을 단순한 식(로그 비율의 직선)으로 근사하기 때문이에요. 여기처럼 분야가 셋뿐이면 \"서비스 비율대로\"가 이미 좋은 출발점이고, 회귀는 분야가 수십 개라 손으로 고를 수 없을 때 힘을 써요."])
tr_rows = []
for traffic in ((0.6, 0.2, 0.2), (0.2, 0.4, 0.4), (0.34, 0.33, 0.33)):
    T = np.array(traffic)
    feats = ns["features"](ns["grid"]) @ ns["coef"]
    bm = ns["grid"][np.argmin([float((1000 * T / c).sum()) for c in feats])]
    tr_rows.append([("<b>" if traffic[0] == 0.6 else "") + " : ".join(map(str, traffic)) + (" (지금)</b>" if traffic[0] == 0.6 else ""), " : ".join(f"{x:.2f}" for x in bm)])
S.claim(float(tr_rows[0][1].split(" : ")[0]) > 0.6 and abs(float(tr_rows[1][1].split(" : ")[0]) - 0.2) < 0.05, "⑦ 한국어가 많으면 더 기울이고, 적으면 거의 그대로")
outs["t7"] = S.table("서비스에 들어올 글의 비율이 다르면", ["서비스 비율 (한국어 : 영어 : 코드)", "회귀가 고른 학습 비율"], tr_rows,
                     [f"서비스에 한국어가 많으면 회귀는 그보다 더 한국어 쪽으로 기울여요({tr_rows[0][1]}). 한국어 압축률이 가장 낮아서, 한국어를 조금 더 잘 압축하는 게 전체 토큰을 가장 많이 줄이기 때문이에요. 한국어가 적은 서비스에서는 거의 서비스 비율 그대로를 골라요({tr_rows[1][1]})."])

fig, ax = plt.subplots(figsize=(8.4, 3.4))
stages = [("모음", raw), ("규칙으로 거름", flt), ("똑같은 것 뺌", exa), ("거의 같은 것 뺌", cln)]
x = np.arange(len(stages))
for j, k in enumerate(DOM):
    ax.bar(x + (j - 1) * 0.26, [s[k][1] / 1000 for _, s in stages], width=0.26, color=COL[k], label=k)
ax.set_xticks(x, [n for n, _ in stages])
ax.set(ylabel="글자 수 (천)", title="단계마다 남은 글")
ax.legend(fontsize=9)
P_FUNNEL = fig_save(fig, "funnel")
outs["f_funnel"] = chart("거르고 빼면 얼마나 남나", P_FUNNEL, "",
                         [f"영어는 {raw['영어'][1]:,}자에서 {cln['영어'][1]:,}자로 {1 - cln['영어'][1] / raw['영어'][1]:.0%}가 줄었어요. 대부분 똑같은 설명문이었어요. 웹 문서를 모으는 실제 LLM도 이 단계에서 많은 양을 버려요."])

flow = [[1, "모으기", "한국어, 영어, 코드", "분야마다 크기"], [2, "규칙으로 거르기", "짧음, 기호, 반복", "버린 것 읽기"], [3, "똑같은 것 빼기", "dict.fromkeys", "외우기 방지"], [4, "거의 같은 것", "MinHash, LSH", "닮은 정도 기준"],
        [5, "토크나이저", "BPE, 압축률", "한국어는 토큰이 많이"], [6, "섞는 비율", "대리 토크나이저", "순위 불변"], [7, "비율 고르기", "회귀", "서비스 글 기준"]]
S.save(ROOT / "topics" / "model" / "pretrain-data-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p>LLM의 성능은 모델 구조보다 <b>무엇을 얼마나 깨끗하게, 어떤 비율로 보여 줬나</b>에 더 크게 갈려요. 그런데 처음부터 학습 한 번에 몇 주와 큰돈이 들어서, 데이터의 선택을 하나하나 큰 모델로 시험해 볼 수 없어요. 그래서 데이터 쪽 결정은 대부분 <b>싸게 미리 재 보고</b> 정해요.</p>"""
why += K.table(["정할 것", "여기서 재 보는 것", "이 쪽"], [["무엇을 버릴까", "규칙이 버린 것을 직접 읽기", "②"], ["겹치는 글", "똑같은 것, 거의 같은 것", "③④"], ["토크나이저", "분야마다 압축률", "⑤"],
                                                     ["무엇을 얼마나 섞을까", "작은 실험의 순위를 믿고 회귀로 고르기", "⑥⑦"]])
why += "<p><a href=\"gpt-generation.html\">작은 GPT</a>에서 말뭉치를 정리하고 BPE를 배웠다면, 이 쪽은 그 선택들을 숫자로 재 봐요. 한국어 서비스라면 특히 ⑤⑦이 비용과 바로 이어져요.</p>"
whole = """<p>모으기 → 규칙으로 거르기 → 똑같은 것 빼기 → 거의 같은 것 빼기 → 토크나이저의 압축률 → 섞는 비율 바꾸기 → 비율 고르기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/pretrain_data/mix_tokenizer.py</code> 하나예요. 받을 파일 없이 이 노트의 한국어 글과 파이썬 표준 라이브러리로 말뭉치를 만들어요.</p>"""
steps = [
    K.step(1, "말뭉치 모으기", "한국어(옛 Day 노트), 영어(표준 라이브러리 설명문), 코드(표준 라이브러리 소스) 세 분야를 문단 단위로 모아요.",
           "<p>실제 LLM의 학습 데이터도 분야(웹, PDF, 코드, 수학, 대화)마다 따로 모으고, 따로 거른 뒤, 마지막에 비율을 정해 섞어요. 분야마다 크기도 품질도 다르기 때문이에요.</p>"
           + K.code_table([("inspect.getdoc(obj)", "함수, 클래스의 설명문"), ("inspect.getsource(module)", "모듈의 소스 코드"), ("re.sub(r'\"\"\".*?\"\"\"', '', src, flags=re.S)", "코드에서 설명문 빼기 (영어 쪽과 겹치지 않게)")]), "s1"),
    K.step(2, "규칙으로 거르기", "너무 짧은 것, 기호와 숫자가 대부분인 것, 같은 단어가 되풀이되는 것을 버려요.",
           "<p>웹 문서에는 메뉴, 광고, 표 조각, 자동 생성 글이 섞여 있어요. 가장 먼저 쓰는 건 이런 <b>규칙(휴리스틱) 거르기</b>예요. 빠르고 설명하기 쉽지만, 기준 하나가 좋은 글까지 버릴 수 있어요. 그다음에 품질 분류기(좋은 글을 닮았나를 맞히는 작은 모델)를 더 쓰기도 해요.</p>"
           + K.code_table([("sum(ch.isalpha() for ch in text) / len(text)", "글자(한글, 알파벳) 비율"), ("len(set(words)) / len(words)", "서로 다른 단어의 비율 (낮으면 반복)")])
           + K.try_("한국어에서 글자 비율 기준을 0.4부터 0.7까지 바꾸면?", "t2"), "s2"),
    K.step(3, "똑같은 것 빼기", "완전히 같은 문단은 처음 나온 하나만 남겨요.",
           "<p>파이썬 dict는 넣은 순서를 기억하고 같은 키를 한 번만 담아요. 그래서 <code>list(dict.fromkeys(docs))</code> 한 줄이 \"순서를 지키며 중복 빼기\"예요. 큰 말뭉치에서는 문서마다 해시를 만들어 같은 해시를 빼요.</p>", "s3"),
    K.step(4, "거의 같은 것 빼기: MinHash", "단어 5개짜리 조각들의 집합으로 문단을 보고, 두 집합이 얼마나 겹치는지(Jaccard)를 지문 64칸으로 빠르게 어림해요.",
           "<p>한두 단어만 다른 글은 해시가 달라서 ③으로 못 잡아요. 두 문단의 조각 집합이 얼마나 겹치는지를 <b>Jaccard 유사도</b>(겹치는 조각 수 ÷ 합친 조각 수)로 재면 되지만, 모든 짝을 비교하면 문단 수의 제곱만큼 걸려요.</p>"
           "<p><b>MinHash</b>는 해시 함수 하나로 조각들을 섞어 가장 작은 값만 남겨요. 두 집합의 가장 작은 값이 같을 확률이 정확히 Jaccard 유사도라서, 해시 64개의 \"가장 작은 값\" 64칸(지문)이 몇 칸 같은지로 유사도를 어림해요. <b>LSH</b>는 지문을 4칸씩 16묶음으로 나눠 한 묶음이라도 똑같은 짝만 후보로 골라, 대부분의 짝을 비교하지 않고 넘어가요.</p>"
           + K.code_table([("zlib.crc32(조각)", "조각을 숫자로"), ("(a × x + b) mod p 의 최솟값", "해시 하나의 MinHash, 64번"), ("sig[:, b*4:(b+1)*4].tobytes()", "묶음이 같은 문단끼리 같은 칸에")])
           + K.out("f_funnel") + K.try_("닮은 정도 기준을 0.5부터 0.9까지 바꾸면?", "t4"), "s4"),
    K.step(5, "토크나이저의 압축률", "가진 비율대로 섞은 글로 BPE 토크나이저를 배우고, 분야마다 토큰 하나가 몇 글자를 담는지 재요.",
           "<p><b>압축률</b>은 같은 글을 몇 개의 토큰으로 바꾸나예요. 압축률이 높으면 같은 글이 적은 토큰이 돼서, 학습과 추론이 싸지고 문맥 창에 더 많이 들어가요. BPE는 학습 글에서 자주 붙어 나오는 조각부터 합치기 때문에, 학습 글에 적게 들어간 분야는 잘게 쪼개져요.</p>"
           "<p>여기서는 바이트 단위 BPE(GPT-2 방식)를 써요. 처음 어휘가 바이트 256개라서 어떤 글자도 모르는 토큰이 되지 않지만, 합쳐진 조각이 없으면 한글 한 글자가 바이트 3개로 쪼개져요.</p>"
           + K.out("t5")
           + K.code_table([("pre_tokenizers.ByteLevel()", "글을 바이트로 보고 띄어쓰기 단위로 미리 나누기"), ("trainers.BpeTrainer(vocab_size=4000)", "자주 붙는 조각을 어휘 4,000개가 될 때까지 합치기"),
                           ("len(text) / len(tok.encode(text).ids)", "압축률: 토큰 하나에 담긴 글자 수")])
           + K.try_("어휘 크기를 1,000, 4,000, 16,000으로 바꾸면?", "t5b"), "s5"),
    K.step(6, "섞는 비율을 바꾸면: 작은 대리 토크나이저", "비율 조합 24가지로 토크나이저를 두 크기로 배우고, 작은 쪽의 순위가 큰 쪽에서도 그대로인지 봐요.",
           "<p>최적의 비율을 찾으려면 비율마다 토크나이저를 배워 봐야 하는데, 실제 크기(어휘 수만 개, 글 수백 GB)로는 비싸요. TReX는 <b>작은 대리(proxy) 토크나이저</b>로 많은 조합을 시험하고, 그 순위가 큰 토크나이저에서도 유지된다는 걸 확인한 뒤 써요. 같은 생각을 데이터 섞기 전체에 쓰는 방법(작은 모델로 비율을 시험하고 회귀로 고르기)도 있어요.</p>"
           + K.out("f_mix")
           + K.code_table([("rng.dirichlet(np.ones(3), 24)", "합이 1인 비율 조합 24가지"), ("spearmanr(proxy[:, j], full[:, j])", "두 크기에서 순위가 같은 정도")]), "s6"),
    K.step(7, "회귀로 비율 고르기", "대리 토크나이저의 결과로 \"비율 → 압축률\" 회귀를 맞추고, 서비스 글의 토큰 수가 가장 적은 비율을 골라 실제 크기로 확인해요.",
           "<p>무엇을 기준으로 \"좋은 비율\"인지부터 정해야 해요. 여기서는 <b>서비스에 들어올 글</b>(한국어 60%, 영어 20%, 코드 20%) 1,000자에 드는 토큰 수를 줄이는 게 목표예요. 분야마다 압축률을 로그 비율의 직선으로 근사하고, 비율 격자 전체에서 예측이 가장 좋은 점을 골라요.</p>"
           + K.code_table([("features(m) = [1, log(비율)]", "비율에서 압축률을 맞히는 특징"), ("np.linalg.lstsq", "최소제곱으로 계수 맞추기"), ("TRAFFIC / chars_per_token", "분야마다 글자 수 ÷ 압축률 = 토큰 수")])
           + K.try_("서비스 글의 비율이 다르면 회귀가 고르는 비율은?", "t7"), "s7"),
]
practice = [("한국어 서비스를 만든다", "쓸 모델의 토크나이저로 우리 글의 압축률부터 재요. 같은 내용이 모델마다 1.5배 넘게 토큰이 차이 날 수 있고, 그게 곧 비용과 문맥 길이예요."),
            ("말뭉치를 거른다", "규칙은 분야마다 따로 맞추고, 버린 것과 남긴 것을 무작위로 뽑아 꼭 읽어요. 중복 빼기는 거의 항상 이득이에요."),
            ("데이터를 섞는다", "\"가진 만큼\"이 아니라 쓰일 곳의 비율에서 출발해요. 조합이 많으면 작은 실험의 순위를 믿고 회귀로 고르고, 마지막 후보만 크게 확인해요.")]
libs = [("①②③", "<code>inspect</code>, <code>importlib</code>, <code>re</code>, <code>collections.Counter</code>", "말뭉치 모으기, 거르기, 세기"), ("④", "<code>zlib.crc32</code>, NumPy", "MinHash 지문"),
        ("⑤⑥⑦", "<code>tokenizers</code> (Hugging Face), <code>scipy.stats.spearmanr</code>, <code>np.linalg.lstsq</code>", "BPE 학습, 순위 상관, 회귀")]
K.build(ROOT / "topics" / "model" / "pretrain-data.html", title="LLM의 학습 데이터 만들기", h1="LLM의 학습 데이터 만들기: 거르고, 빼고, 섞기", track="모델",
        crumb="모델 &gt; 순서와 언어 모델",
        one_line="LLM을 학습하기 전에 <b>규칙으로 거르고</b>, <b>똑같은 것과 거의 같은 것(MinHash)을 빼고</b>, 분야를 <b>어떤 비율로 섞을지</b> 정해요. 섞는 비율은 토크나이저의 <b>압축률</b>을 바꿔서, 한국어처럼 적게 섞인 언어는 토큰이 많이 들어요. 비율은 <b>작은 대리 실험의 순위</b>를 믿고 회귀로 골라요.",
        why=why, whole=whole, steps=steps, code_file="examples/pretrain_data/mix_tokenizer.py", practice=practice, libs=libs,
        libs_note="<code>docs_of</code>, <code>code_of</code>, <code>bad_reason</code>, <code>minhash</code>, <code>near_duplicates</code>, <code>train_tokenizer</code>, <code>chars_per_token</code>, <code>features</code>, <code>tokens_per_1k</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"gpt-generation.html\">작은 GPT와 디코딩</a>", "BPE를 배우고 GPT를 처음부터"), ("<a href=\"llm-architectures.html\">요즘 LLM의 부품</a>", "같은 카테고리"), ("<a href=\"../backend/chunking.html\">Chunking</a>", "토큰 수와 비용")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: Upstage 10Talks Lecture 04 \"Technical Choices and Strategy: Lessons from Olmo 3\"(사전 학습 평가 고르기, 웹 데이터의 규칙 거르기와 3단계 중복 빼기, PDF와 코드와 수학, 섞기와 뽑기), Lecture 06 \"TReX: Tokenizer Regression for Optimal Data Mixture\"(압축률, 대리 토크나이저와 순위 불변성, 혼합비 회귀). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/pretrain_data.py</code>가 <code>examples/pretrain_data/mix_tokenizer.py</code>를 단계별로 실제 실행한 결과 (seed 0)"])
print("ok")
