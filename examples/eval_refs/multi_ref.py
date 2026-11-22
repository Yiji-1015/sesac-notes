# 정답이 여러 개일 수 있는 글(맞춤법 고치기)을 n-gram 지표로 채점할 때, 정답 하나로 재면 생기는 문제와 정답 여러 개를 합치는 네 가지 방법을 비교하는 전체 코드.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: ../gpt/data/pages.jsonl (주제 페이지의 본문 문장). 해요체 문장을 합니다체, 한다체로 바꿔 "똑같이 맞는 정답" 셋을 만들고,
#         띄어쓰기와 조사 오류를 일부러 넣어 고칠 문장을 만들어요. 어디를 틀리게 했는지 알기 때문에 "진짜로 얼마나 고쳤나"를 알아요.
# CPU로 10초쯤. 모델이나 API는 쓰지 않아요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                                                     #@1
import math                                                                     #@2
from collections import Counter                                                 #@2
from itertools import combinations                                              #@7
from pathlib import Path                                                        #@1
import numpy as np                                                              #@1

# ① 문장과 세 가지 정답, 일부러 넣은 오류
HERE = Path(__file__).resolve().parent
pages = [json.loads(line) for line in (HERE.parent / "gpt" / "data" / "pages.jsonl").open(encoding="utf-8")]
sentences = [s for p in pages for s in p["sentences"]]
STYLES = [("이에요.", "입니다.", "이다."), ("해요.", "합니다.", "한다."), ("있어요.", "있습니다.", "있다."), ("없어요.", "없습니다.", "없다."),
          ("돼요.", "됩니다.", "된다."), ("써요.", "씁니다.", "쓴다."), ("봐요.", "봅니다.", "본다."), ("나와요.", "나옵니다.", "나온다."),
          ("같아요.", "같습니다.", "같다."), ("달라요.", "다릅니다.", "다르다."), ("많아요.", "많습니다.", "많다."), ("줘요.", "줍니다.", "준다."),
          ("내요.", "냅니다.", "낸다."), ("골라요.", "고릅니다.", "고른다."), ("찾아요.", "찾습니다.", "찾는다."), ("배워요.", "배웁니다.", "배운다."),
          ("했어요.", "했습니다.", "했다."), ("봤어요.", "봤습니다.", "봤다.")]
def three_styles(s):                                             # 해요체 → (해요체, 합니다체, 한다체). 바꿀 수 없으면 None
    if "거예요" in s or not 25 <= len(s) <= 90:
        return None
    for yo, nida, da in STYLES:
        if s.endswith(yo):
            return s, s[:-len(yo)] + nida, s[:-len(yo)] + da
    return None
references = [r for r in map(three_styles, sentences) if r]

SWAP = {"을": "를", "를": "을", "은": "는", "는": "은", "이": "가", "가": "이"}
rng = np.random.default_rng(0)
def make_errors(s, k=2):                                         # 오류 k개: 조사 바꾸기 또는 띄어쓰기 지우기
    words = s.split(" ")
    particle = [i for i, w in enumerate(words[:-1]) if len(w) >= 2 and w[-1] in SWAP]
    space = list(range(len(words) - 2))
    errors = []
    for _ in range(k):
        if rng.random() < 0.5 and particle:
            errors.append(("조사", particle.pop(rng.integers(len(particle)))))
        elif space:
            errors.append(("띄어쓰기", space.pop(rng.integers(len(space)))))
    def apply(errs):
        ws, joined = list(words), set()
        for kind, i in errs:
            if kind == "조사":
                ws[i] = ws[i][:-1] + SWAP[ws[i][-1]]
            else:
                joined.add(i)
        return "".join(w + ("" if i in joined else " ") for i, w in enumerate(ws)).strip()
    return apply(errors), apply(errors[1:])                       # (고칠 문장, 오류 하나만 고친 문장)

items = []
for refs in references:
    source, half = make_errors(refs[0])
    items.append({"source": source, "half": half, "refs": refs})
systems = {                                                      # 고치기 시스템 다섯 개의 답과, 진짜로 고친 비율
    "다 고침 (해요체)": ([it["refs"][0] for it in items], 1.0),
    "다 고침 (합니다체)": ([it["refs"][1] for it in items], 1.0),
    "다 고침 (한다체)": ([it["refs"][2] for it in items], 1.0),
    "하나만 고침 (해요체)": ([it["half"] for it in items], 0.5),
    "안 고침": ([it["source"] for it in items], 0.0),
}

# ② n-gram 겹침으로 재기: BLEU
def ngrams(text, n, unit="char"):                                # 한국어는 음절(글자) 단위로 (띄어쓰기도 글자로 셈)
    toks = list(text) if unit == "char" else text.split()
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))

def clipped(hyp_counts, ref_counts):                             # 정답에 있는 횟수까지만 맞은 것으로 (같은 말 되풀이 막기)
    return sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())

def bleu(hyp, ref_counts_by_n, ref_len, N=4, unit="char"):       # 문장 하나의 BLEU (n ≥ 2는 1을 더해 0이 되지 않게)
    logs = []
    for n in range(1, N + 1):
        h = ngrams(hyp, n, unit)
        match, total = clipped(h, ref_counts_by_n[n]), sum(h.values())
        logs.append(math.log((match + (n > 1)) / (total + (n > 1))) if match + (n > 1) > 0 else -9)
    hyp_len = len(hyp) if unit == "char" else len(hyp.split())
    brevity = min(1.0, math.exp(1 - ref_len / hyp_len))          # 너무 짧은 답에 벌점
    return brevity * math.exp(sum(logs) / N)

def ref_counts(ref, N=4, unit="char"):
    return {n: ngrams(ref, n, unit) for n in range(1, N + 1)}

def single_bleu(hyp, ref, unit="char"):
    return bleu(hyp, ref_counts(ref, unit=unit), len(ref) if unit == "char" else len(ref.split()), unit=unit)

toy_refs = ["the cat is on the mat", "a cat sits on the mat"]        # 강연의 예: 정답 둘, 답 하나 (단어 1-gram)
toy_hyp = "the cat the cat on the mat"
toy_h = ngrams(toy_hyp, 1, "word")
toy_each = [clipped(toy_h, ngrams(r, 1, "word")) for r in toy_refs]
toy_merged = clipped(toy_h, sum((ngrams(r, 1, "word") for r in toy_refs), Counter()))
toy = {"가장 좋은 정답 하나": max(toy_each), "정답마다 평균": float(np.mean(toy_each)), "정답을 합친 횟수": toy_merged, "답의 단어 수": sum(toy_h.values())}

# ③ 정답 하나로만 재면
one_ref = {name: float(np.mean([single_bleu(h, it["refs"][0]) for h, it in zip(hyps, items)])) for name, (hyps, _) in systems.items()}

# ④ 고치지 않은 오류에 벌점: GLEU
def gleu(hyp, src, ref_counts_by_n, ref_len, N=4, unit="char"):
    logs = []
    for n in range(1, N + 1):
        h, s, r = ngrams(hyp, n, unit), ngrams(src, n, unit), ref_counts_by_n[n]
        not_fixed = Counter({g: c for g, c in s.items() if g not in r})   # 원문에는 있는데 정답에는 없는 조각 = 고쳐야 할 곳
        score = max(clipped(h, r) - clipped(h, not_fixed), 0)
        total = sum(h.values())
        logs.append(math.log((score + (n > 1)) / (total + (n > 1))) if score + (n > 1) > 0 else -9)
    hyp_len = len(hyp) if unit == "char" else len(hyp.split())
    return min(1.0, math.exp(1 - ref_len / hyp_len)) * math.exp(sum(logs) / N)

one_ref_gleu = {name: float(np.mean([gleu(h, it["source"], ref_counts(it["refs"][0]), len(it["refs"][0])) for h, it in zip(hyps, items)]))
                for name, (hyps, _) in systems.items()}

# ⑤ 정답 여러 개를 합치는 네 가지 방법
def multi(metric, hyp, src, refs, how, tau=10.0, unit="char"):
    L = (lambda t: len(t)) if unit == "char" else (lambda t: len(t.split()))
    if how == "정답을 합친 횟수":                                  # 정답들의 n-gram 횟수를 더해 한 정답처럼
        merged = {n: sum((ngrams(r, n, unit) for r in refs), Counter()) for n in range(1, 5)}
        closest = min(refs, key=lambda r: abs(L(r) - L(hyp)))
        args = (hyp, merged, L(closest)) if metric == "BLEU" else (hyp, src, merged, L(closest))
        return (bleu if metric == "BLEU" else gleu)(*args, unit=unit)
    f = np.array([bleu(hyp, ref_counts(r, unit=unit), L(r), unit=unit) if metric == "BLEU" else gleu(hyp, src, ref_counts(r, unit=unit), L(r), unit=unit) for r in refs])
    if how == "가장 좋은 정답 하나":
        return float(f.max())
    if how == "정답마다 평균":
        return float(f.mean())
    w = np.exp(tau * f) / np.exp(tau * f).sum()                 # 점수가 높은 정답에 더 큰 무게 (softmax)
    return float((w * f).sum())

HOWS = ["가장 좋은 정답 하나", "정답마다 평균", "softmax 가중 평균", "정답을 합친 횟수"]
def system_scores(metric, how, ref_ids=(0, 1, 2), unit="char"):
    return {name: float(np.mean([multi(metric, h, it["source"], [it["refs"][i] for i in ref_ids], how, unit=unit) for h, it in zip(hyps, items)]))
            for name, (hyps, _) in systems.items()}

board = {(metric, how): system_scores(metric, how) for metric in ("BLEU", "GLEU") for how in HOWS}

# ⑥ 문장마다 맞게 줄 세우나
def pair_accuracy(metric, how, ref_ids=(0, 1, 2), unit="char"):  # 진짜로 더 많이 고친 답에 더 높은 점수를 준 비율 (같은 문장의 답끼리)
    names = list(systems)
    right = total = 0
    for k, it in enumerate(items):
        sc = {n: multi(metric, systems[n][0][k], it["source"], [it["refs"][i] for i in ref_ids], how, unit=unit) for n in names}
        for a, b in combinations(names, 2):
            if systems[a][1] != systems[b][1]:
                total += 1
                better, worse = (a, b) if systems[a][1] > systems[b][1] else (b, a)
                right += sc[better] > sc[worse]
    return right / total

agreement = {(metric, refs_name): pair_accuracy(metric, "가장 좋은 정답 하나" if len(ids) > 1 else "정답마다 평균", ids)
             for metric in ("BLEU", "GLEU") for refs_name, ids in (("정답 하나 (해요체)", (0,)), ("정답 셋", (0, 1, 2)))}

# ⑦ 정답 수를 늘리면
growth = {}
for how in ("가장 좋은 정답 하나", "정답마다 평균", "정답을 합친 횟수"):
    for k in (1, 2, 3):
        subsets = list(combinations(range(3), k))                # 정답 k개를 고르는 모든 경우의 평균
        growth[(how, k)] = float(np.mean([system_scores("GLEU", how, ids)["하나만 고침 (해요체)"] for ids in subsets]))
