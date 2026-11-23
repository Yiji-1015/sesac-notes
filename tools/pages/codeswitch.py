"""모델 > 한국어와 영어가 섞일 때. examples/codeswitch/mixed_language.py 하나가 ①~⑥으로 자란다.
실행: python tools/pages/codeswitch.py
"""
import os
import sys
import shutil
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import html  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "codeswitch"
code = S.parse((EX / "mixed_language.py").read_text(encoding="utf-8"))
_tmp = Path(tempfile.mkdtemp(prefix="stepcode_"))                   # ../gpt/data 와 ../../terms 를 읽어서 같은 모양으로 복사
work = _tmp / "examples" / "codeswitch"
shutil.copytree(EX, work, ignore=shutil.ignore_patterns("__pycache__"))
(_tmp / "examples" / "gpt" / "data").mkdir(parents=True)
shutil.copy(ROOT / "examples" / "gpt" / "data" / "pages.jsonl", _tmp / "examples" / "gpt" / "data")
shutil.copytree(ROOT / "terms", _tmp / "terms")
ns = S.run(code, 6, work, filename="mixed_language.py")
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, transformers {transformers.__version__}, intfloat/multilingual-e5-small (CPU)"
pct = lambda x: f"{float(x):.1%}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "model" / "img" / "codeswitch"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/codeswitch/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


outs = {}
ms, hit, terms = ns["mix_stats"], ns["hit"], ns["terms"]
n = len(terms)
# ---------------- ①
S.claim(ms["영어 단어가 하나라도 있는 문장"] / ms["문장 수"] > 0.3, "① 30% 넘는 문장에 영어 단어")
outs["s1"] = S.table("① 이 노트의 본문 문장", ["", "문장 수", "비율"], [[k, f"{v:,}", pct(v / ms['문장 수'])] for k, v in ms.items()],
                     [f"문장의 {ms['영어 단어가 하나라도 있는 문장'] / ms['문장 수']:.0%}에 영어 단어가 섞여 있어요. 단어 수로는 {ns['english_share']:.0%}뿐이지만, 거의 다 RAG, Chunking 같은 핵심 용어예요.",
                      "대부분은 영어 단어에 한국어 조사가 바로 붙은 꼴이에요. 문장의 뼈대(주언어, matrix language)는 한국어이고 영어 단어가 끼어든(삽입 언어) 거예요. 그러니 이 문장들에 답할 때는 한국어로 답해야 자연스러워요."])
outs["t1"] = S.table("영어 단어 뒤에 붙은 것 (많은 순)", ["붙은 것", "횟수"], [[html.escape(k) or "(없음)", v] for k, v in ns["attached"].most_common(10)], [])

# ---------------- ②
tc = ns["token_count"]
outs["s2"] = S.box("② 용어집에서 만든 짝", [(f"한국어 이름, 영어 이름, 한국어 설명이 모두 있는 용어 {n}개", 1), (f"토큰 수 평균: 한국어 이름 {tc['한국어 이름']:.1f}, 영어 이름 {tc['영어 이름']:.1f}", 2),
                                          *[(f"{w} → {' | '.join(p)}", i + 3, "newf" if i == 0 else None) for i, (w, p) in enumerate(ns["pieces"].items())]],
                   ["다국어 토크나이저는 영어 단어와 한국어 조사를 따로 잘라요(RAG | 는). 같은 개념이 한국어로 쓰이면 한 벌, 영어로 쓰이면 다른 한 벌의 토큰이 돼서, 모델이 두 벌을 같은 뜻으로 배웠는지가 관건이에요.",
                    "설명에 용어 이름이 그대로 들어 있는 짝은 너무 쉬워서 뺐어요."])

# ---------------- ③
S.claim(hit["한국어 이름"].mean() > hit["영어 이름"].mean() and hit["영어 이름"].mean() > 5 * ns["chance"], "③ 한국어가 영어보다 높고, 영어도 우연의 5배 넘게")
outs["s3"] = S.table(f"③ 이름으로 설명 찾기 ({n}개 중 위 5개 안에 있나)", ["물은 말", "찾은 비율"],
                     [[f"한국어 이름 (예: {terms[0][0]})", pct(hit["한국어 이름"].mean())], [f"영어 이름 (예: {terms[0][1]})", pct(hit["영어 이름"].mean())], ["우연히 맞을 비율", pct(ns["chance"])]],
                     [f"이름 하나로 짧은 설명을 찾는 건 작은 모델에게 어려운 일이라 절대값은 낮아요. 그래도 우연({ns['chance']:.1%})보다 훨씬 높고, 한국어 이름으로 물을 때({hit['한국어 이름'].mean():.0%})가 영어 이름으로 물을 때({hit['영어 이름'].mean():.0%})보다 잘 찾아요. 설명이 한국어라서 같은 언어끼리가 더 가까워요.",
                      "다국어 임베딩은 \"같은 뜻이면 언어가 달라도 가깝게\" 배우지만, 실제로는 같은 언어 쪽으로 기울어 있어요."])

# ---------------- ④
cs = ns["consistency"]
found_any = cs["둘 다 찾음"] + cs["한국어로만"] + cs["영어로만"]
S.claim(cs["둘 다 찾음"] < 0.6 * found_any and cs["영어로만"] > 0, "④ 한쪽이라도 찾은 것 중 둘 다는 60% 미만, 영어로만 찾은 것도 있음")
outs["s4"] = S.table("④ 같은 용어를 두 언어로 물었을 때", ["결과", "용어 수", "비율"], [[k, v, pct(v / n)] for k, v in cs.items()],
                     [f"한쪽 언어로라도 찾은 {found_any}개 중 두 언어 모두로 찾은 건 {cs['둘 다 찾음']}개({cs['둘 다 찾음'] / found_any:.0%})뿐이에요. 같은 개념을 물었는데 언어에 따라 답이 달라지는 거예요. 강연은 이걸 교차 언어 일관성이라고 부르고, 언어별 정확도와 따로 재야 하는 지표로 다뤄요.",
                      "영어로만 찾은 예: " + ", ".join(f"{k} / {e}" for k, e in ns["only_en_examples"]) + ". 한국어 이름보다 영어 이름이 설명과 더 가까운 경우도 있어요."])

# ---------------- ⑤
gd = ns["gradual"]
S.claim(gd[0.0] > gd[0.5] > gd[1.0], "⑤ 영어 쪽으로 갈수록 낮아짐")
outs["s5"] = S.table(f"⑤ 두 단어 이상인 용어 {len(ns['multi'])}개를 조금씩 영어로 바꾸면", ["영어로 바꾼 비율", "예", "찾은 비율"],
                     [[f"{r:.0%}", html.escape(ns["gradual_example"][r]), pct(v)] for r, v in gd.items()],
                     [f"섞을수록 설명에서 멀어져요({gd[0.0]:.0%} → {gd[0.5]:.0%} → {gd[1.0]:.0%}). 문서가 한국어니까 질문도 한국어일수록 가까워요.",
                      "강연의 CSICL은 반대 방향으로 섞어요. 한국어 질문을 예시 몇 개에서 조금씩 영어로 바꿔 보여 주면서(0% → 25% → … → 100%), LLM이 속으로 영어로 옮겨 생각하고 답은 한국어로 하게 해요. LLM은 영어로 가장 잘 \"생각\"하기 때문이에요. 섞는 방향은 \"무엇과 가까워져야 하나\"로 정해요. 검색이면 문서의 언어, 추론이면 모델이 가장 잘하는 언어예요."])

# ---------------- ⑥
S.claim(hit["한국어 (영어)"].mean() > hit["한국어 이름"].mean(), "⑥ 두 언어를 함께 적으면 한국어보다 높음")
fig, ax = plt.subplots(figsize=(7.4, 2.8))
labels = ["영어 이름", "한국어 이름", "한국어 (영어)"]
ax.barh(labels, [hit[k].mean() for k in labels], color=[C["blue"], C["purple"], C["pink"]])
for i, k in enumerate(labels):
    ax.text(hit[k].mean() + 0.005, i, f"{hit[k].mean():.1%}", va="center", fontsize=9)
ax.axvline(ns["chance"], color=C["ink"], ls=":", lw=1)
ax.set(xlabel=f"설명 {n}개 중 위 5개 안에 맞는 설명이 있는 비율", xlim=(0, 0.5), title="물을 때 쓴 이름")
P_BOTH = fig_save(fig, "both")
outs["f_both"] = chart("이름을 어떻게 적어 물었나", P_BOTH, "", ["점선은 우연히 맞을 비율이에요."])
outs["s6"] = S.box(f"⑥ \"{terms[0][0]} ({terms[0][1]})\"처럼 두 언어를 함께 적으면", [(f"찾은 비율 → {hit['한국어 (영어)'].mean():.1%} (한국어만 {hit['한국어 이름'].mean():.1%}, 영어만 {hit['영어 이름'].mean():.1%})", 1, "newf"),
                                                                        (f"한국어로도 영어로도 못 찾던 용어 중 함께 적어서 찾은 것 → {ns['both_rescued']}개", 2)],
                   ["두 이름을 함께 주면 한쪽 언어로 배운 연결이 약해도 다른 쪽이 메워 줘요. 이 노트가 용어를 처음 쓸 때 영어 이름을 같이 적는 이유이기도 해요. 문서를 쓸 때나 검색 질문을 만들 때 쓸 수 있는 가장 싼 방법이에요."])

flow = [[1, "섞인 문장 세기", "영어 + 조사", "주언어는 한국어"], [2, "용어 짝", "한국어, 영어, 설명", "토큰 두 벌"], [3, "설명 찾기", "한국어 vs 영어", "같은 언어 쪽으로"],
        [4, "교차 언어 일관성", "둘 다 찾나", "언어에 따라 다른 답"], [5, "조금씩 섞기", "0%, 50%, 100%", "섞는 방향"], [6, "함께 적기", "한국어 (영어)", "가장 싼 방법"]]
S.save(ROOT / "topics" / "model" / "codeswitch-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow, "env": ENV})

why = """<p>한국에서 일하는 사람의 글은 \"RAG는 Chunking을 먼저 해요\"처럼 한국어와 영어가 섞여요. 한 대화나 문장 안에서 언어를 바꿔 쓰는 걸 <b>코드 스위칭</b>이라고 해요. 세계 인구의 절반 넘게가 두 언어 이상을 쓰니 아주 흔한 일인데, LLM에게는 문제이기도 하고 해결책이기도 해요.</p>"""
why += K.table(["", "문제가 되는 곳", "도움이 되는 곳"], [["출력 언어", "섞인 질문에 엉뚱한 언어로 답함 (어느 언어가 뼈대인지 못 알아챔)", ""], ["안전", "여러 언어를 섞은 위험한 질문에 안전장치가 덜 걸림", ""],
                                                    ["학습", "", "단어 → 문장 → 한 언어 순으로 섞인 글을 먼저 배우면 새 언어를 더 잘 배움"], ["추론", "", "질문을 조금씩 영어로 바꿔 보여 주면 저자원 언어에서 더 잘 답함"]])
why += "<p>이 쪽은 LLM을 부르지 않고, 이 노트의 문장과 용어집, 그리고 다국어 임베딩 모델 하나로 \"섞이면 무엇이 달라지나\"를 재 봐요. <a href=\"../backend/embedding.html\">임베딩과 벡터 검색</a>을 알면 더 쉬워요.</p>"
whole = """<p>섞인 문장 세기 → 용어 짝과 토큰 → 한국어와 영어로 설명 찾기 → 교차 언어 일관성 → 조금씩 섞기 → 함께 적기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/codeswitch/mixed_language.py</code> 하나예요. 모델은 다른 쪽에서도 쓰는 multilingual-e5-small이에요.</p>"""
steps = [
    K.step(1, "이 노트 문장은 얼마나 섞여 있나", "주제 페이지 본문 문장 1,016개를 띄어쓰기로 나눠, 영어 단어와 \"영어 + 한국어 조사\" 단어를 세요.",
           "<p>코드 스위칭에서 문장의 문법 뼈대를 이루는 언어를 <b>주언어</b>(matrix language), 끼어든 언어를 <b>삽입 언어</b>(embedded language)라고 해요. 강연의 OLA 연구는 섞인 질문의 13%만 답할 언어를 밝히고, 나머지는 사람이 맥락으로 알아챈다는 점에서 출발해요. 모델은 이런 맥락보다 표면 단서(어느 글자가 많나)에 끌려 엉뚱한 언어로 답하기 쉬워요.</p>"
           + K.out("t1")
           + K.code_table([("re.compile(r\"[가-힣]\")", "한글이 있나"), ("word_kind(w)", "한국어, 영어, 영어 + 조사로 나누기")]), "s1"),
    K.step(2, "용어 짝과 토큰", "용어집에서 한국어 이름, 영어 이름, 한국어 설명이 모두 있는 용어를 골라요.",
           K.code_table([("tok.tokenize(\"RAG는\")", "다국어 토크나이저가 자른 조각"), ("en.lower() in desc.lower()", "설명에 이름이 그대로 있으면 빼기")]), "s2"),
    K.step(3, "한국어로 물을 때, 영어로 물을 때", "설명들을 문서로 넣고, 한국어 이름과 영어 이름을 각각 질문으로 넣어 맞는 설명이 위 5개 안에 오는지 봐요.",
           "<p>e5 모델은 질문 앞에 <code>query: </code>, 문서 앞에 <code>passage: </code>를 붙여 학습했어요. 둘을 같은 공간의 벡터로 바꿔 코사인 유사도가 큰 순으로 줄 세워요.</p>"
           + K.code_table([("embed(texts, \"query: \")", "질문 벡터 (토큰 벡터 평균, 길이 1로)"), ("(q @ passages.T).topk(5)", "가장 가까운 설명 5개")]), "s3"),
    K.step(4, "두 언어 모두 찾나: 교차 언어 일관성", "같은 용어를 한국어와 영어로 물었을 때 결과가 같은지 짝마다 세요.",
           "<p>정확도만 보면 놓치는 문제예요. 한국어 정확도와 영어 정확도가 비슷해도, 서로 다른 문제를 맞힐 수 있어요. 강연의 CSCL(코드 스위칭 커리큘럼)은 토큰 단위로 섞은 글 → 문장 단위로 섞은 글 → 한 언어 글 순서로 학습해서, 두 언어 모두 맞히는 비율을 26.7%에서 41.6%로 올렸다고 해요.</p>", "s4"),
    K.step(5, "조금씩 섞어서 물으면", "두 단어 이상인 용어의 한국어 이름을 앞에서부터 0%, 50%, 100% 영어로 바꿔 물어요.",
           K.code_table([("mix(i, ratio)", "앞에서부터 ratio만큼 영어 단어로")]), "s5"),
    K.step(6, "두 언어를 함께 적으면", "\"한국어 이름 (영어 이름)\"으로 물어요.",
           K.out("f_both"), "s6"),
]
practice = [("한국어 서비스에서 LLM이 엉뚱한 언어로 답한다", "시스템 프롬프트에 답할 언어를 분명히 적어요. 강연에서도 언어를 명시하면 거의 완벽하게 지켰어요. 섞인 질문이 많으면 \"주언어로 답한 예\"로 선호 학습(DPO)을 하기도 해요."),
            ("한국어와 영어가 섞인 문서를 검색한다", "문서와 질문의 언어를 맞추거나, 핵심 용어를 두 언어로 함께 적어 둬요. 다국어 임베딩이어도 같은 언어끼리가 더 가까워요."),
            ("안전장치를 만든다", "영어로만 시험하지 말고, 다른 언어와 섞인 질문으로도 시험해요. 자원이 적은 언어일수록 안전장치가 약해요.")]
libs = [("①", "<code>re</code>, <code>collections.Counter</code>", "글자로 언어 나누기, 세기"), ("②~⑥", "<code>transformers</code>(AutoTokenizer, AutoModel), <code>torch</code>", "다국어 임베딩")]
K.build(ROOT / "topics" / "model" / "codeswitch.html", title="한국어와 영어가 섞일 때", h1="한국어와 영어가 섞일 때: 코드 스위칭", track="모델",
        crumb="모델 &gt; 언어 모델로 문제 풀기",
        one_line="이 노트 문장의 절반쯤은 <b>한국어 뼈대에 영어 용어</b>가 섞여 있어요(코드 스위칭). 다국어 임베딩도 <b>같은 언어끼리 더 가깝고</b>, 같은 개념을 한국어와 영어로 물으면 <b>찾는 결과가 자주 달라요</b>. 문서 언어에 맞춰 묻거나 <b>두 언어를 함께 적으면</b> 나아져요.",
        why=why, whole=whole, steps=steps, code_file="examples/codeswitch/mixed_language.py", practice=practice, libs=libs,
        libs_note="<code>word_kind</code>, <code>embed</code>, <code>found</code>, <code>mix</code>, <code>found_subset</code>은 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"../backend/embedding.html\">임베딩과 벡터 검색</a>", "e5와 코사인 유사도"), ("<a href=\"token-classification.html\">토큰 분류와 질문 답변</a>", "언어 간 전이"), ("<a href=\"pretrain-data.html\">LLM의 학습 데이터 만들기</a>", "한국어 토큰 압축률")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: Upstage 10Talks Lecture 10 \"Code Switching for/with Multilingual LLMs\"(BLEND 문화 일상 지식, MUG-Eval, OLA 출력 언어 맞추기와 주언어, CSRT 섞인 질문의 안전 문제, CSCL 코드 스위칭 커리큘럼과 교차 언어 일관성, CSICL 점진적 섞기). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/codeswitch.py</code>가 <code>examples/codeswitch/mixed_language.py</code>를 단계별로 실제 실행한 결과"])
print("ok")
