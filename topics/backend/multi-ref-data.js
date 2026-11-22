window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import math",
   "s": 2
  },
  {
   "t": "from collections import Counter",
   "s": 2
  },
  {
   "t": "from itertools import combinations",
   "s": 7
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 문장과 세 가지 정답, 일부러 넣은 오류",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "pages = [json.loads(line) for line in (HERE.parent / \"gpt\" / \"data\" / \"pages.jsonl\").open(encoding=\"utf-8\")]",
   "s": 1
  },
  {
   "t": "sentences = [s for p in pages for s in p[\"sentences\"]]",
   "s": 1
  },
  {
   "t": "STYLES = [(\"이에요.\", \"입니다.\", \"이다.\"), (\"해요.\", \"합니다.\", \"한다.\"), (\"있어요.\", \"있습니다.\", \"있다.\"), (\"없어요.\", \"없습니다.\", \"없다.\"),",
   "s": 1
  },
  {
   "t": "          (\"돼요.\", \"됩니다.\", \"된다.\"), (\"써요.\", \"씁니다.\", \"쓴다.\"), (\"봐요.\", \"봅니다.\", \"본다.\"), (\"나와요.\", \"나옵니다.\", \"나온다.\"),",
   "s": 1
  },
  {
   "t": "          (\"같아요.\", \"같습니다.\", \"같다.\"), (\"달라요.\", \"다릅니다.\", \"다르다.\"), (\"많아요.\", \"많습니다.\", \"많다.\"), (\"줘요.\", \"줍니다.\", \"준다.\"),",
   "s": 1
  },
  {
   "t": "          (\"내요.\", \"냅니다.\", \"낸다.\"), (\"골라요.\", \"고릅니다.\", \"고른다.\"), (\"찾아요.\", \"찾습니다.\", \"찾는다.\"), (\"배워요.\", \"배웁니다.\", \"배운다.\"),",
   "s": 1
  },
  {
   "t": "          (\"했어요.\", \"했습니다.\", \"했다.\"), (\"봤어요.\", \"봤습니다.\", \"봤다.\")]",
   "s": 1
  },
  {
   "t": "def three_styles(s):  # 해요체 → (해요체, 합니다체, 한다체). 바꿀 수 없으면 None",
   "s": 1
  },
  {
   "t": "    if \"거예요\" in s or not 25 <= len(s) <= 90:",
   "s": 1
  },
  {
   "t": "        return None",
   "s": 1
  },
  {
   "t": "    for yo, nida, da in STYLES:",
   "s": 1
  },
  {
   "t": "        if s.endswith(yo):",
   "s": 1
  },
  {
   "t": "            return s, s[:-len(yo)] + nida, s[:-len(yo)] + da",
   "s": 1
  },
  {
   "t": "    return None",
   "s": 1
  },
  {
   "t": "references = [r for r in map(three_styles, sentences) if r]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "SWAP = {\"을\": \"를\", \"를\": \"을\", \"은\": \"는\", \"는\": \"은\", \"이\": \"가\", \"가\": \"이\"}",
   "s": 1
  },
  {
   "t": "rng = np.random.default_rng(0)",
   "s": 1
  },
  {
   "t": "def make_errors(s, k=2):  # 오류 k개: 조사 바꾸기 또는 띄어쓰기 지우기",
   "s": 1
  },
  {
   "t": "    words = s.split(\" \")",
   "s": 1
  },
  {
   "t": "    particle = [i for i, w in enumerate(words[:-1]) if len(w) >= 2 and w[-1] in SWAP]",
   "s": 1
  },
  {
   "t": "    space = list(range(len(words) - 2))",
   "s": 1
  },
  {
   "t": "    errors = []",
   "s": 1
  },
  {
   "t": "    for _ in range(k):",
   "s": 1
  },
  {
   "t": "        if rng.random() < 0.5 and particle:",
   "s": 1
  },
  {
   "t": "            errors.append((\"조사\", particle.pop(rng.integers(len(particle)))))",
   "s": 1
  },
  {
   "t": "        elif space:",
   "s": 1
  },
  {
   "t": "            errors.append((\"띄어쓰기\", space.pop(rng.integers(len(space)))))",
   "s": 1
  },
  {
   "t": "    def apply(errs):",
   "s": 1
  },
  {
   "t": "        ws, joined = list(words), set()",
   "s": 1
  },
  {
   "t": "        for kind, i in errs:",
   "s": 1
  },
  {
   "t": "            if kind == \"조사\":",
   "s": 1
  },
  {
   "t": "                ws[i] = ws[i][:-1] + SWAP[ws[i][-1]]",
   "s": 1
  },
  {
   "t": "            else:",
   "s": 1
  },
  {
   "t": "                joined.add(i)",
   "s": 1
  },
  {
   "t": "        return \"\".join(w + (\"\" if i in joined else \" \") for i, w in enumerate(ws)).strip()",
   "s": 1
  },
  {
   "t": "    return apply(errors), apply(errors[1:])  # (고칠 문장, 오류 하나만 고친 문장)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "items = []",
   "s": 1
  },
  {
   "t": "for refs in references:",
   "s": 1
  },
  {
   "t": "    source, half = make_errors(refs[0])",
   "s": 1
  },
  {
   "t": "    items.append({\"source\": source, \"half\": half, \"refs\": refs})",
   "s": 1
  },
  {
   "t": "systems = {  # 고치기 시스템 다섯 개의 답과, 진짜로 고친 비율",
   "s": 1
  },
  {
   "t": "    \"다 고침 (해요체)\": ([it[\"refs\"][0] for it in items], 1.0),",
   "s": 1
  },
  {
   "t": "    \"다 고침 (합니다체)\": ([it[\"refs\"][1] for it in items], 1.0),",
   "s": 1
  },
  {
   "t": "    \"다 고침 (한다체)\": ([it[\"refs\"][2] for it in items], 1.0),",
   "s": 1
  },
  {
   "t": "    \"하나만 고침 (해요체)\": ([it[\"half\"] for it in items], 0.5),",
   "s": 1
  },
  {
   "t": "    \"안 고침\": ([it[\"source\"] for it in items], 0.0),",
   "s": 1
  },
  {
   "t": "}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② n-gram 겹침으로 재기: BLEU",
   "s": 2
  },
  {
   "t": "def ngrams(text, n, unit=\"char\"):  # 한국어는 음절(글자) 단위로 (띄어쓰기도 글자로 셈)",
   "s": 2
  },
  {
   "t": "    toks = list(text) if unit == \"char\" else text.split()",
   "s": 2
  },
  {
   "t": "    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def clipped(hyp_counts, ref_counts):  # 정답에 있는 횟수까지만 맞은 것으로 (같은 말 되풀이 막기)",
   "s": 2
  },
  {
   "t": "    return sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def bleu(hyp, ref_counts_by_n, ref_len, N=4, unit=\"char\"):  # 문장 하나의 BLEU (n ≥ 2는 1을 더해 0이 되지 않게)",
   "s": 2
  },
  {
   "t": "    logs = []",
   "s": 2
  },
  {
   "t": "    for n in range(1, N + 1):",
   "s": 2
  },
  {
   "t": "        h = ngrams(hyp, n, unit)",
   "s": 2
  },
  {
   "t": "        match, total = clipped(h, ref_counts_by_n[n]), sum(h.values())",
   "s": 2
  },
  {
   "t": "        logs.append(math.log((match + (n > 1)) / (total + (n > 1))) if match + (n > 1) > 0 else -9)",
   "s": 2
  },
  {
   "t": "    hyp_len = len(hyp) if unit == \"char\" else len(hyp.split())",
   "s": 2
  },
  {
   "t": "    brevity = min(1.0, math.exp(1 - ref_len / hyp_len))  # 너무 짧은 답에 벌점",
   "s": 2
  },
  {
   "t": "    return brevity * math.exp(sum(logs) / N)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def ref_counts(ref, N=4, unit=\"char\"):",
   "s": 2
  },
  {
   "t": "    return {n: ngrams(ref, n, unit) for n in range(1, N + 1)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def single_bleu(hyp, ref, unit=\"char\"):",
   "s": 2
  },
  {
   "t": "    return bleu(hyp, ref_counts(ref, unit=unit), len(ref) if unit == \"char\" else len(ref.split()), unit=unit)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "toy_refs = [\"the cat is on the mat\", \"a cat sits on the mat\"]  # 강연의 예: 정답 둘, 답 하나 (단어 1-gram)",
   "s": 2
  },
  {
   "t": "toy_hyp = \"the cat the cat on the mat\"",
   "s": 2
  },
  {
   "t": "toy_h = ngrams(toy_hyp, 1, \"word\")",
   "s": 2
  },
  {
   "t": "toy_each = [clipped(toy_h, ngrams(r, 1, \"word\")) for r in toy_refs]",
   "s": 2
  },
  {
   "t": "toy_merged = clipped(toy_h, sum((ngrams(r, 1, \"word\") for r in toy_refs), Counter()))",
   "s": 2
  },
  {
   "t": "toy = {\"가장 좋은 정답 하나\": max(toy_each), \"정답마다 평균\": float(np.mean(toy_each)), \"정답을 합친 횟수\": toy_merged, \"답의 단어 수\": sum(toy_h.values())}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 정답 하나로만 재면",
   "s": 3
  },
  {
   "t": "one_ref = {name: float(np.mean([single_bleu(h, it[\"refs\"][0]) for h, it in zip(hyps, items)])) for name, (hyps, _) in systems.items()}",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 고치지 않은 오류에 벌점: GLEU",
   "s": 4
  },
  {
   "t": "def gleu(hyp, src, ref_counts_by_n, ref_len, N=4, unit=\"char\"):",
   "s": 4
  },
  {
   "t": "    logs = []",
   "s": 4
  },
  {
   "t": "    for n in range(1, N + 1):",
   "s": 4
  },
  {
   "t": "        h, s, r = ngrams(hyp, n, unit), ngrams(src, n, unit), ref_counts_by_n[n]",
   "s": 4
  },
  {
   "t": "        not_fixed = Counter({g: c for g, c in s.items() if g not in r})  # 원문에는 있는데 정답에는 없는 조각 = 고쳐야 할 곳",
   "s": 4
  },
  {
   "t": "        score = max(clipped(h, r) - clipped(h, not_fixed), 0)",
   "s": 4
  },
  {
   "t": "        total = sum(h.values())",
   "s": 4
  },
  {
   "t": "        logs.append(math.log((score + (n > 1)) / (total + (n > 1))) if score + (n > 1) > 0 else -9)",
   "s": 4
  },
  {
   "t": "    hyp_len = len(hyp) if unit == \"char\" else len(hyp.split())",
   "s": 4
  },
  {
   "t": "    return min(1.0, math.exp(1 - ref_len / hyp_len)) * math.exp(sum(logs) / N)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "one_ref_gleu = {name: float(np.mean([gleu(h, it[\"source\"], ref_counts(it[\"refs\"][0]), len(it[\"refs\"][0])) for h, it in zip(hyps, items)]))",
   "s": 4
  },
  {
   "t": "                for name, (hyps, _) in systems.items()}",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 정답 여러 개를 합치는 네 가지 방법",
   "s": 5
  },
  {
   "t": "def multi(metric, hyp, src, refs, how, tau=10.0, unit=\"char\"):",
   "s": 5
  },
  {
   "t": "    L = (lambda t: len(t)) if unit == \"char\" else (lambda t: len(t.split()))",
   "s": 5
  },
  {
   "t": "    if how == \"정답을 합친 횟수\":  # 정답들의 n-gram 횟수를 더해 한 정답처럼",
   "s": 5
  },
  {
   "t": "        merged = {n: sum((ngrams(r, n, unit) for r in refs), Counter()) for n in range(1, 5)}",
   "s": 5
  },
  {
   "t": "        closest = min(refs, key=lambda r: abs(L(r) - L(hyp)))",
   "s": 5
  },
  {
   "t": "        args = (hyp, merged, L(closest)) if metric == \"BLEU\" else (hyp, src, merged, L(closest))",
   "s": 5
  },
  {
   "t": "        return (bleu if metric == \"BLEU\" else gleu)(*args, unit=unit)",
   "s": 5
  },
  {
   "t": "    f = np.array([bleu(hyp, ref_counts(r, unit=unit), L(r), unit=unit) if metric == \"BLEU\" else gleu(hyp, src, ref_counts(r, unit=unit), L(r), unit=unit) for r in refs])",
   "s": 5
  },
  {
   "t": "    if how == \"가장 좋은 정답 하나\":",
   "s": 5
  },
  {
   "t": "        return float(f.max())",
   "s": 5
  },
  {
   "t": "    if how == \"정답마다 평균\":",
   "s": 5
  },
  {
   "t": "        return float(f.mean())",
   "s": 5
  },
  {
   "t": "    w = np.exp(tau * f) / np.exp(tau * f).sum()  # 점수가 높은 정답에 더 큰 무게 (softmax)",
   "s": 5
  },
  {
   "t": "    return float((w * f).sum())",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "HOWS = [\"가장 좋은 정답 하나\", \"정답마다 평균\", \"softmax 가중 평균\", \"정답을 합친 횟수\"]",
   "s": 5
  },
  {
   "t": "def system_scores(metric, how, ref_ids=(0, 1, 2), unit=\"char\"):",
   "s": 5
  },
  {
   "t": "    return {name: float(np.mean([multi(metric, h, it[\"source\"], [it[\"refs\"][i] for i in ref_ids], how, unit=unit) for h, it in zip(hyps, items)]))",
   "s": 5
  },
  {
   "t": "            for name, (hyps, _) in systems.items()}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "board = {(metric, how): system_scores(metric, how) for metric in (\"BLEU\", \"GLEU\") for how in HOWS}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 문장마다 맞게 줄 세우나",
   "s": 6
  },
  {
   "t": "def pair_accuracy(metric, how, ref_ids=(0, 1, 2), unit=\"char\"):  # 진짜로 더 많이 고친 답에 더 높은 점수를 준 비율 (같은 문장의 답끼리)",
   "s": 6
  },
  {
   "t": "    names = list(systems)",
   "s": 6
  },
  {
   "t": "    right = total = 0",
   "s": 6
  },
  {
   "t": "    for k, it in enumerate(items):",
   "s": 6
  },
  {
   "t": "        sc = {n: multi(metric, systems[n][0][k], it[\"source\"], [it[\"refs\"][i] for i in ref_ids], how, unit=unit) for n in names}",
   "s": 6
  },
  {
   "t": "        for a, b in combinations(names, 2):",
   "s": 6
  },
  {
   "t": "            if systems[a][1] != systems[b][1]:",
   "s": 6
  },
  {
   "t": "                total += 1",
   "s": 6
  },
  {
   "t": "                better, worse = (a, b) if systems[a][1] > systems[b][1] else (b, a)",
   "s": 6
  },
  {
   "t": "                right += sc[better] > sc[worse]",
   "s": 6
  },
  {
   "t": "    return right / total",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "agreement = {(metric, refs_name): pair_accuracy(metric, \"가장 좋은 정답 하나\" if len(ids) > 1 else \"정답마다 평균\", ids)",
   "s": 6
  },
  {
   "t": "             for metric in (\"BLEU\", \"GLEU\") for refs_name, ids in ((\"정답 하나 (해요체)\", (0,)), (\"정답 셋\", (0, 1, 2)))}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 정답 수를 늘리면",
   "s": 7
  },
  {
   "t": "growth = {}",
   "s": 7
  },
  {
   "t": "for how in (\"가장 좋은 정답 하나\", \"정답마다 평균\", \"정답을 합친 횟수\"):",
   "s": 7
  },
  {
   "t": "    for k in (1, 2, 3):",
   "s": 7
  },
  {
   "t": "        subsets = list(combinations(range(3), k))  # 정답 k개를 고르는 모든 경우의 평균",
   "s": 7
  },
  {
   "t": "        growth[(how, k)] = float(np.mean([system_scores(\"GLEU\", how, ids)[\"하나만 고침 (해요체)\"] for ids in subsets]))",
   "s": 7
  }
 ],
 "names": {
  "1": "문장과 세 가지 정답, 일부러 넣은 오류",
  "2": "n-gram 겹침으로 재기: BLEU",
  "3": "정답 하나로만 재면",
  "4": "고치지 않은 오류에 벌점: GLEU",
  "5": "정답 여러 개를 합치는 네 가지 방법",
  "6": "문장마다 맞게 줄 세우나",
  "7": "정답 수를 늘리면"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 만든 채점 데이터",
   "lines": [
    [
     "해요체로 끝나는 본문 문장 1,016개 중 문체를 바꿀 수 있는 480개",
     1
    ],
    [
     "문장마다 똑같이 맞는 정답 셋: 해요체, 합니다체, 한다체",
     2,
     "newf"
    ],
    [
     "문장마다 오류 둘: 조사 바꾸기 또는 띄어쓰기 지우기",
     3
    ]
   ],
   "notes": [
    "답을 내는 \"고치기 시스템\" 다섯 개를 흉내 내요. 셋은 오류를 다 고쳤고 문체만 달라요. 하나는 오류 하나만 고쳤고, 하나는 아무것도 안 고쳤어요. 진짜 품질(고친 비율)을 알고 있으니 지표가 맞게 재는지 확인할 수 있어요."
   ]
  },
  "t1": {
   "title": "문장 하나의 예",
   "cols": [
    "",
    "문장 (굵은 곳이 정답 해요체와 다른 곳)"
   ],
   "rows": [
    [
     "고칠 문장",
     "RAG는 질문<b>가</b> 오면 관련 글을<b>⌃</b>찾아서 LLM에게 근거로 붙여 주는 방식이에요."
    ],
    [
     "정답 1 (해요체)",
     "RAG는 질문이 오면 관련 글을 찾아서 LLM에게 근거로 붙여 주는 방식이에요."
    ],
    [
     "정답 2 (합니다체)",
     "RAG는 질문이 오면 관련 글을 찾아서 LLM에게 근거로 붙여 주는 방식<b>입니다</b>."
    ],
    [
     "정답 3 (한다체)",
     "RAG는 질문이 오면 관련 글을 찾아서 LLM에게 근거로 붙여 주는 방식이<b>다</b>."
    ],
    [
     "하나만 고침",
     "RAG는 질문<b>가</b> 오면 관련 글을 찾아서 LLM에게 근거로 붙여 주는 방식이에요."
    ]
   ],
   "notes": [
    "⌃는 빠진 띄어쓰기예요."
   ]
  },
  "s2": {
   "title": "② 강연의 예: 정답 둘로 1-gram 정밀도 재기",
   "cols": [
    "합치는 법",
    "맞은 단어 수",
    "정밀도"
   ],
   "rows": [
    [
     "가장 좋은 정답 하나",
     "5",
     "5/7 = 0.714"
    ],
    [
     "정답마다 평균",
     "4.5",
     "4.5/7 = 0.643"
    ],
    [
     "정답을 합친 횟수",
     "7",
     "7/7 = 1.000"
    ]
   ],
   "notes": [
    "정답: \"the cat is on the mat\", \"a cat sits on the mat\". 답: \"the cat the cat on the mat\".",
    "정답마다 따로 세면 the는 2번까지만 맞은 것으로 쳐요(잘라 세기). 그런데 두 정답의 횟수를 더하면 the가 3번까지 인정돼서, 말이 안 되는 답이 만점이 돼요. 합치는 방법 하나가 점수를 크게 바꿔요."
   ]
  },
  "s3": {
   "title": "③ 정답 하나(해요체)로만 잰 BLEU (음절 단위, 문장 평균)",
   "cols": [
    "시스템",
    "진짜 고친 비율",
    "BLEU"
   ],
   "rows": [
    [
     "다 고침 (해요체)",
     "100%",
     "1.000"
    ],
    [
     "다 고침 (합니다체)",
     "100%",
     "0.912"
    ],
    [
     "다 고침 (한다체)",
     "100%",
     "0.934"
    ],
    [
     "하나만 고침 (해요체)",
     "50%",
     "0.940"
    ],
    [
     "안 고침",
     "0%",
     "0.883"
    ]
   ],
   "notes": [
    "오류를 다 고친 합니다체(0.912)가 하나만 고친 해요체(0.940)보다 낮아요. 문장 끝의 \"해요\"와 \"합니다\"가 다른 것이 고치지 못한 오류 하나보다 더 많은 글자 조각을 바꾸기 때문이에요.",
    "BLEU는 \"정답과 얼마나 같은 조각을 썼나\"만 봐요. 정답이 하나면 그 정답과 다른 표현은 다 틀린 것이 돼요."
   ]
  },
  "t3": {
   "title": "조각을 음절이 아니라 어절로 자르면",
   "cols": [
    "단위",
    "다 고침 (해요체)",
    "다 고침 (합니다체)",
    "다 고침 (한다체)",
    "하나만 고침 (해요체)",
    "안 고침"
   ],
   "rows": [
    [
     "<b>음절</b> (지금)",
     "1.000",
     "0.912",
     "0.934",
     "0.940",
     "0.883"
    ],
    [
     "어절 (띄어쓰기로 나눔)",
     "1.000",
     "0.895",
     "0.895",
     "0.715",
     "0.526"
    ]
   ],
   "notes": [
    "어절 단위로 자르면 띄어쓰기 오류 하나가 어절 두 개를 통째로 바꿔서 오류에 훨씬 엄해져요. 그래서 이 데이터에서는 순서가 맞게 나와요. 한국어는 어절, 형태소, 음절 중 무엇으로 자르는지에 따라 같은 지표가 다르게 움직여요."
   ]
  },
  "s4": {
   "title": "④ 같은 정답 하나로 잰 BLEU와 GLEU",
   "cols": [
    "시스템",
    "BLEU",
    "GLEU",
    "차이"
   ],
   "rows": [
    [
     "다 고침 (해요체)",
     "1.000",
     "1.000",
     "0.000"
    ],
    [
     "다 고침 (합니다체)",
     "0.912",
     "0.912",
     "0.000"
    ],
    [
     "다 고침 (한다체)",
     "0.934",
     "0.934",
     "-0.000"
    ],
    [
     "하나만 고침 (해요체)",
     "0.940",
     "0.897",
     "-0.043"
    ],
    [
     "안 고침",
     "0.883",
     "0.789",
     "-0.094"
    ]
   ],
   "notes": [
    "GLEU는 원문에 있던 조각 중 정답에는 없는 것(고쳐야 했던 곳)을 답이 그대로 쓰면 맞은 수에서 빼요. 그래서 안 고친 답이 0.883에서 0.789로 크게 내려가요. 다 고친 답은 깎이지 않아요.",
    "그래도 정답이 하나뿐이면 문체가 다른 답은 여전히 깎여요. 그건 정답을 늘려야 풀려요."
   ]
  },
  "s5": {
   "title": "⑤ 정답 셋을 합치는 네 가지 방법 (GLEU)",
   "cols": [
    "시스템",
    "가장 좋은 정답 하나",
    "정답마다 평균",
    "softmax 가중 평균",
    "정답을 합친 횟수"
   ],
   "rows": [
    [
     "다 고침 (해요체)",
     "1.000",
     "0.905",
     "0.957",
     "1.000"
    ],
    [
     "다 고침 (합니다체)",
     "1.000",
     "0.945",
     "0.963",
     "1.000"
    ],
    [
     "다 고침 (한다체)",
     "1.000",
     "0.951",
     "0.966",
     "1.000"
    ],
    [
     "하나만 고침 (해요체)",
     "0.897",
     "0.802",
     "0.854",
     "0.902"
    ],
    [
     "안 고침",
     "0.789",
     "0.693",
     "0.747",
     "0.796"
    ]
   ],
   "notes": [
    "\"가장 좋은 정답 하나\"와 \"정답을 합친 횟수\"는 다 고친 세 답에 모두 만점을 줘요. \"맞는 정답 중 하나와 맞으면 된다\"는 생각이에요.",
    "\"정답마다 평균\"은 다 고친 해요체에 0.905를 줘요. 나머지 두 정답(합니다체, 한다체)과 달라서 깎인 거예요. 정답들의 분포 전체에 가까울수록 좋다는 생각이라, 정답들이 서로 많이 다르면 맞는 답도 깎여요. softmax 가중 평균은 그 사이예요."
   ]
  },
  "t5": {
   "title": "BLEU로 같은 비교를 하면",
   "cols": [
    "시스템",
    "가장 좋은 정답 하나",
    "정답마다 평균",
    "softmax 가중 평균",
    "정답을 합친 횟수"
   ],
   "rows": [
    [
     "다 고침 (해요체)",
     "1.000",
     "0.948",
     "0.965",
     "1.000"
    ],
    [
     "다 고침 (합니다체)",
     "1.000",
     "0.945",
     "0.963",
     "1.000"
    ],
    [
     "다 고침 (한다체)",
     "1.000",
     "0.952",
     "0.966",
     "1.000"
    ],
    [
     "하나만 고침 (해요체)",
     "0.940",
     "0.890",
     "0.906",
     "0.946"
    ],
    [
     "안 고침",
     "0.883",
     "0.834",
     "0.849",
     "0.891"
    ]
   ],
   "notes": [
    "순서는 GLEU와 같지만, 안 고친 답과 고친 답의 차이가 GLEU보다 작아요."
   ]
  },
  "f_how": {
   "title": "합치는 방법별 점수",
   "images": [
    [
     "img/multiref/aggregation.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽 세 묶음은 모두 다 고친 답이에요. 보라(가장 좋은 하나)와 분홍(합친 횟수)은 셋 모두 1이고, 파랑(평균)과 민트(가중 평균)는 1보다 낮고 셋에 서로 다른 점수를 줘요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ 문장마다, 더 많이 고친 답에 더 높은 점수를 줬나",
   "cols": [
    "지표",
    "정답",
    "맞게 줄 세운 비율"
   ],
   "rows": [
    [
     "BLEU",
     "정답 하나 (해요체)",
     "70.7%"
    ],
    [
     "BLEU",
     "정답 셋",
     "100.0%"
    ],
    [
     "GLEU",
     "정답 하나 (해요체)",
     "97.8%"
    ],
    [
     "GLEU",
     "정답 셋",
     "100.0%"
    ]
   ],
   "notes": [
    "같은 문장에 대한 두 답 중 진짜로 더 많이 고친 쪽을 지표도 더 높게 준 비율이에요. 문장 480개 × 비교할 짝 9개를 셌어요.",
    "정답이 하나면 BLEU는 71%만 맞혀요. GLEU는 정답이 하나여도 98%이고, 정답이 셋이면 둘 다 모두 맞혀요(정답 셋은 \"가장 좋은 정답 하나\"로 합쳤어요).",
    "이 데이터는 정답이 될 수 있는 표현을 우리가 다 알아서 쉬운 편이에요. 실제 정답 모음은 맞는 표현의 일부만 담고 있어서, 정답을 셋 모아도 이만큼 깔끔하지 않아요."
   ]
  },
  "s7": {
   "title": "⑦ 정답 수를 늘리면: \"하나만 고침\"의 GLEU",
   "cols": [
    "합치는 법",
    "정답 1개",
    "2개",
    "3개"
   ],
   "rows": [
    [
     "가장 좋은 정답 하나",
     "0.802",
     "0.854",
     "0.897"
    ],
    [
     "정답마다 평균",
     "0.802",
     "0.802",
     "0.802"
    ],
    [
     "정답을 합친 횟수",
     "0.802",
     "0.856",
     "0.902"
    ]
   ],
   "notes": [
    "정답 k개를 고르는 모든 경우의 평균이에요. \"하나만 고침\"은 해요체라서 해요체 정답이 들어 있을 때 점수가 높아요.",
    "\"정답을 합친 횟수\"는 정답이 늘수록 계속 올라가요(0.802 → 0.902). 인정되는 조각이 늘어서예요. 강연의 실험(여러 언어의 맞춤법 고치기)에서도 합친 횟수가 정답을 더할수록 안정적으로 올랐어요.",
    "\"정답마다 평균\"은 정답 수와 상관없이 같아요. 정답을 더 모아도 평균은 \"정답 하나의 기댓값\"이라서예요. 정답을 모으는 데 돈을 쓴다면 어떤 방법으로 합칠지부터 정해야 해요."
   ]
  }
 },
 "flow": [
  [
   1,
   "데이터",
   "정답 셋, 오류 둘",
   "진짜 품질을 앎"
  ],
  [
   2,
   "BLEU",
   "잘라 세기",
   "합치는 법이 점수를 바꿈"
  ],
  [
   3,
   "정답 하나",
   "문체 차이",
   "틀린 순서"
  ],
  [
   4,
   "GLEU",
   "안 고친 곳 벌점",
   "오류에 엄하게"
  ],
  [
   5,
   "네 가지 합치기",
   "최대, 평균, 가중, 합침",
   "존재 대 분포"
  ],
  [
   6,
   "줄 세우기",
   "짝마다 맞나",
   "정답 수의 힘"
  ],
  [
   7,
   "정답 늘리기",
   "1, 2, 3개",
   "합친 횟수만 오름"
  ]
 ],
 "env": "Python 3.13.5, NumPy 2.5.2 (seed 0, 모델과 API 없음)"
};
