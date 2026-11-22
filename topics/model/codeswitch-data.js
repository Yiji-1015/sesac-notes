window.PAGE = {
 "code": [
  {
   "t": "import json",
   "s": 1
  },
  {
   "t": "import re",
   "s": 1
  },
  {
   "t": "from collections import Counter",
   "s": 1
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
   "t": "import torch",
   "s": 3
  },
  {
   "t": "from transformers import AutoTokenizer, AutoModel",
   "s": 2
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 노트 문장은 얼마나 섞여 있나",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "HANGUL, LATIN = re.compile(r\"[가-힣]\"), re.compile(r\"[A-Za-z]\")",
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
   "t": "def word_kind(w):",
   "s": 1
  },
  {
   "t": "    h, l = bool(HANGUL.search(w)), bool(LATIN.search(w))",
   "s": 1
  },
  {
   "t": "    return \"영어+한국어 조사/어미\" if h and l else \"영어\" if l else \"한국어\" if h else None",
   "s": 1
  },
  {
   "t": "kinds = [Counter(k for k in map(word_kind, s.split()) if k) for s in sentences]",
   "s": 1
  },
  {
   "t": "mix_stats = {",
   "s": 1
  },
  {
   "t": "    \"문장 수\": len(sentences),",
   "s": 1
  },
  {
   "t": "    \"영어 단어가 하나라도 있는 문장\": sum(1 for c in kinds if c[\"영어\"] or c[\"영어+한국어 조사/어미\"]),",
   "s": 1
  },
  {
   "t": "    \"영어 단어에 조사가 붙은 문장 (예: RAG는)\": sum(1 for c in kinds if c[\"영어+한국어 조사/어미\"]),",
   "s": 1
  },
  {
   "t": "    \"한국어 단어만 있는 문장\": sum(1 for c in kinds if not c[\"영어\"] and not c[\"영어+한국어 조사/어미\"]),",
   "s": 1
  },
  {
   "t": "}",
   "s": 1
  },
  {
   "t": "english_share = np.mean([(c[\"영어\"] + c[\"영어+한국어 조사/어미\"]) / sum(c.values()) for c in kinds if sum(c.values())])",
   "s": 1
  },
  {
   "t": "attached = Counter(re.sub(r\"^[A-Za-z0-9\\-_.]+\", \"\", w) for s in sentences for w in s.split() if word_kind(w) == \"영어+한국어 조사/어미\" and re.match(r\"^[A-Za-z]\", w))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 용어집에서 한국어 이름, 영어 이름, 설명 짝 만들기",
   "s": 2
  },
  {
   "t": "terms = []",
   "s": 2
  },
  {
   "t": "seen = set()",
   "s": 2
  },
  {
   "t": "for f in sorted((HERE.parent.parent / \"terms\").glob(\"day*.json\")):",
   "s": 2
  },
  {
   "t": "    for t in json.loads(f.read_text(encoding=\"utf-8\")):",
   "s": 2
  },
  {
   "t": "        ko, desc = t[\"term\"], t[\"short\"]",
   "s": 2
  },
  {
   "t": "        if not HANGUL.search(ko) or LATIN.search(ko):",
   "s": 2
  },
  {
   "t": "            continue",
   "s": 2
  },
  {
   "t": "        en = next((c for c in [t.get(\"full\", \"\")] + t.get(\"aliases\", []) if c and LATIN.search(c) and not HANGUL.search(c) and (\" \" in c or len(c) > 5)), None)",
   "s": 2
  },
  {
   "t": "        if not en or en.lower() in desc.lower() or ko in desc or ko in seen or en in seen:  # 설명에 이름이 그대로 있으면 너무 쉬워서 빼요",
   "s": 2
  },
  {
   "t": "            continue",
   "s": 2
  },
  {
   "t": "        seen |= {ko, en}",
   "s": 2
  },
  {
   "t": "        terms.append((ko, en, desc))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "MODEL = \"intfloat/multilingual-e5-small\"",
   "s": 2
  },
  {
   "t": "tok = AutoTokenizer.from_pretrained(MODEL)",
   "s": 2
  },
  {
   "t": "token_count = {\"한국어 이름\": float(np.mean([len(tok(k)[\"input_ids\"]) - 2 for k, _, _ in terms])),",
   "s": 2
  },
  {
   "t": "               \"영어 이름\": float(np.mean([len(tok(e)[\"input_ids\"]) - 2 for _, e, _ in terms]))}",
   "s": 2
  },
  {
   "t": "pieces = {w: tok.tokenize(w) for w in (\"RAG는\", \"Chunking을\", \"임베딩을\", \"embedding을\")}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 설명 찾기: 한국어로 물을 때 vs 영어로 물을 때",
   "s": 3
  },
  {
   "t": "model = AutoModel.from_pretrained(MODEL).eval()",
   "s": 3
  },
  {
   "t": "def embed(texts, prefix):  # e5는 질문에 \"query: \", 문서에 \"passage: \"를 붙여요",
   "s": 3
  },
  {
   "t": "    out = []",
   "s": 3
  },
  {
   "t": "    for i in range(0, len(texts), 64):",
   "s": 3
  },
  {
   "t": "        b = tok([prefix + t for t in texts[i:i + 64]], padding=True, truncation=True, max_length=128, return_tensors=\"pt\")",
   "s": 3
  },
  {
   "t": "        with torch.no_grad():",
   "s": 3
  },
  {
   "t": "            h = model(**b).last_hidden_state",
   "s": 3
  },
  {
   "t": "        e = (h * b[\"attention_mask\"][..., None]).sum(1) / b[\"attention_mask\"].sum(1, keepdim=True)",
   "s": 3
  },
  {
   "t": "        out.append(torch.nn.functional.normalize(e, dim=1))",
   "s": 3
  },
  {
   "t": "    return torch.cat(out)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "passages = embed([d for _, _, d in terms], \"passage: \")",
   "s": 3
  },
  {
   "t": "def found(queries, k=5):  # 맞는 설명이 위 k개 안에 있나 (용어마다 참/거짓)",
   "s": 3
  },
  {
   "t": "    top = (embed(queries, \"query: \") @ passages.T).topk(k, dim=1).indices",
   "s": 3
  },
  {
   "t": "    return (top == torch.arange(len(queries))[:, None]).any(dim=1).numpy()",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "hit = {\"한국어 이름\": found([k for k, _, _ in terms]), \"영어 이름\": found([e for _, e, _ in terms])}",
   "s": 3
  },
  {
   "t": "chance = 5 / len(terms)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 두 언어 모두 찾나 (교차 언어 일관성)",
   "s": 4
  },
  {
   "t": "ko_hit, en_hit = hit[\"한국어 이름\"], hit[\"영어 이름\"]",
   "s": 4
  },
  {
   "t": "consistency = {\"둘 다 찾음\": int((ko_hit & en_hit).sum()), \"한국어로만\": int((ko_hit & ~en_hit).sum()),",
   "s": 4
  },
  {
   "t": "               \"영어로만\": int((~ko_hit & en_hit).sum()), \"둘 다 못 찾음\": int((~ko_hit & ~en_hit).sum())}",
   "s": 4
  },
  {
   "t": "only_en_examples = [terms[i][:2] for i in np.flatnonzero(~ko_hit & en_hit)[:3]]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 조금씩 섞어서 물으면",
   "s": 5
  },
  {
   "t": "multi = [i for i, (k, e, _) in enumerate(terms) if len(k.split()) >= 2 and len(e.split()) >= 2]  # 두 단어 이상인 용어",
   "s": 5
  },
  {
   "t": "def mix(i, ratio):  # 앞에서부터 ratio만큼 영어 단어로 바꾸기",
   "s": 5
  },
  {
   "t": "    kw, ew = terms[i][0].split(), terms[i][1].split()",
   "s": 5
  },
  {
   "t": "    return \" \".join(ew[:round(ratio * len(ew))] + kw[round(ratio * len(kw)):])",
   "s": 5
  },
  {
   "t": "def found_subset(queries, idx, k=5):",
   "s": 5
  },
  {
   "t": "    top = (embed(queries, \"query: \") @ passages.T).topk(k, dim=1).indices",
   "s": 5
  },
  {
   "t": "    return (top == torch.tensor(idx)[:, None]).any(dim=1).numpy()",
   "s": 5
  },
  {
   "t": "gradual = {r: float(found_subset([mix(i, r) for i in multi], multi).mean()) for r in (0.0, 0.5, 1.0)}",
   "s": 5
  },
  {
   "t": "gradual_example = {r: mix(multi[0], r) for r in (0.0, 0.5, 1.0)}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 두 언어를 함께 적으면",
   "s": 6
  },
  {
   "t": "hit[\"한국어 (영어)\"] = found([f\"{k} ({e})\" for k, e, _ in terms])",
   "s": 6
  },
  {
   "t": "both_rescued = int((hit[\"한국어 (영어)\"] & ~ko_hit & ~en_hit).sum())",
   "s": 6
  }
 ],
 "names": {
  "1": "노트 문장은 얼마나 섞여 있나",
  "2": "용어집에서 한국어 이름, 영어 이름, 설명 짝 만들기",
  "3": "설명 찾기: 한국어로 물을 때 vs 영어로 물을 때",
  "4": "두 언어 모두 찾나 (교차 언어 일관성)",
  "5": "조금씩 섞어서 물으면",
  "6": "두 언어를 함께 적으면"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "s1": {
   "title": "① 이 노트의 본문 문장",
   "cols": [
    "",
    "문장 수",
    "비율"
   ],
   "rows": [
    [
     "문장 수",
     "1,016",
     "100.0%"
    ],
    [
     "영어 단어가 하나라도 있는 문장",
     "482",
     "47.4%"
    ],
    [
     "영어 단어에 조사가 붙은 문장 (예: RAG는)",
     "412",
     "40.6%"
    ],
    [
     "한국어 단어만 있는 문장",
     "534",
     "52.6%"
    ]
   ],
   "notes": [
    "문장의 47%에 영어 단어가 섞여 있어요. 단어 수로는 8%뿐이지만, 거의 다 RAG, Chunking 같은 핵심 용어예요.",
    "대부분은 영어 단어에 한국어 조사가 바로 붙은 꼴이에요. 문장의 뼈대(주언어, matrix language)는 한국어이고 영어 단어가 끼어든(삽입 언어) 거예요. 그러니 이 문장들에 답할 때는 한국어로 답해야 자연스러워요."
   ]
  },
  "t1": {
   "title": "영어 단어 뒤에 붙은 것 (많은 순)",
   "cols": [
    "붙은 것",
    "횟수"
   ],
   "rows": [
    [
     "는",
     72
    ],
    [
     "가",
     49
    ],
    [
     "를",
     49
    ],
    [
     "로",
     38
    ],
    [
     "은",
     34
    ],
    [
     "의",
     24
    ],
    [
     "에",
     23
    ],
    [
     "을",
     18
    ],
    [
     "와",
     14
    ],
    [
     "이",
     13
    ]
   ],
   "notes": []
  },
  "s2": {
   "title": "② 용어집에서 만든 짝",
   "lines": [
    [
     "한국어 이름, 영어 이름, 한국어 설명이 모두 있는 용어 218개",
     1
    ],
    [
     "토큰 수 평균: 한국어 이름 3.6, 영어 이름 3.7",
     2
    ],
    [
     "RAG는 → ▁RA | G | 는",
     3,
     "newf"
    ],
    [
     "Chunking을 → ▁Chunki | ng | 을",
     4,
     null
    ],
    [
     "임베딩을 → ▁임 | 베 | 딩 | 을",
     5,
     null
    ],
    [
     "embedding을 → ▁ | embe | dding | 을",
     6,
     null
    ]
   ],
   "notes": [
    "다국어 토크나이저는 영어 단어와 한국어 조사를 따로 잘라요(RAG | 는). 같은 개념이 한국어로 쓰이면 한 벌, 영어로 쓰이면 다른 한 벌의 토큰이 돼서, 모델이 두 벌을 같은 뜻으로 배웠는지가 관건이에요.",
    "설명에 용어 이름이 그대로 들어 있는 짝은 너무 쉬워서 뺐어요."
   ]
  },
  "s3": {
   "title": "③ 이름으로 설명 찾기 (218개 중 위 5개 안에 있나)",
   "cols": [
    "물은 말",
    "찾은 비율"
   ],
   "rows": [
    [
     "한국어 이름 (예: 머신러닝)",
     "36.7%"
    ],
    [
     "영어 이름 (예: Machine Learning)",
     "29.4%"
    ],
    [
     "우연히 맞을 비율",
     "2.3%"
    ]
   ],
   "notes": [
    "이름 하나로 짧은 설명을 찾는 건 작은 모델에게 어려운 일이라 절대값은 낮아요. 그래도 우연(2.3%)보다 훨씬 높고, 한국어 이름으로 물을 때(37%)가 영어 이름으로 물을 때(29%)보다 잘 찾아요. 설명이 한국어라서 같은 언어끼리가 더 가까워요.",
    "다국어 임베딩은 \"같은 뜻이면 언어가 달라도 가깝게\" 배우지만, 실제로는 같은 언어 쪽으로 기울어 있어요."
   ]
  },
  "s4": {
   "title": "④ 같은 용어를 두 언어로 물었을 때",
   "cols": [
    "결과",
    "용어 수",
    "비율"
   ],
   "rows": [
    [
     "둘 다 찾음",
     40,
     "18.3%"
    ],
    [
     "한국어로만",
     40,
     "18.3%"
    ],
    [
     "영어로만",
     24,
     "11.0%"
    ],
    [
     "둘 다 못 찾음",
     114,
     "52.3%"
    ]
   ],
   "notes": [
    "한쪽 언어로라도 찾은 104개 중 두 언어 모두로 찾은 건 40개(38%)뿐이에요. 같은 개념을 물었는데 언어에 따라 답이 달라지는 거예요. 강연은 이걸 교차 언어 일관성이라고 부르고, 언어별 정확도와 따로 재야 하는 지표로 다뤄요.",
    "영어로만 찾은 예: 표준편차 / Standard deviation, 원-핫 인코딩 / One-hot, 과소적합 / Underfitting. 한국어 이름보다 영어 이름이 설명과 더 가까운 경우도 있어요."
   ]
  },
  "s5": {
   "title": "⑤ 두 단어 이상인 용어 107개를 조금씩 영어로 바꾸면",
   "cols": [
    "영어로 바꾼 비율",
    "예",
    "찾은 비율"
   ],
   "rows": [
    [
     "0%",
     "유클리드 거리",
     "47.7%"
    ],
    [
     "50%",
     "L2 거리",
     "39.3%"
    ],
    [
     "100%",
     "L2 Distance",
     "32.7%"
    ]
   ],
   "notes": [
    "섞을수록 설명에서 멀어져요(48% → 39% → 33%). 문서가 한국어니까 질문도 한국어일수록 가까워요.",
    "강연의 CSICL은 반대 방향으로 섞어요. 한국어 질문을 예시 몇 개에서 조금씩 영어로 바꿔 보여 주면서(0% → 25% → … → 100%), LLM이 속으로 영어로 옮겨 생각하고 답은 한국어로 하게 해요. LLM은 영어로 가장 잘 \"생각\"하기 때문이에요. 섞는 방향은 \"무엇과 가까워져야 하나\"로 정해요. 검색이면 문서의 언어, 추론이면 모델이 가장 잘하는 언어예요."
   ]
  },
  "f_both": {
   "title": "이름을 어떻게 적어 물었나",
   "images": [
    [
     "img/codeswitch/both.png",
     ""
    ]
   ],
   "notes": [
    "점선은 우연히 맞을 비율이에요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ \"머신러닝 (Machine Learning)\"처럼 두 언어를 함께 적으면",
   "lines": [
    [
     "찾은 비율 → 40.4% (한국어만 36.7%, 영어만 29.4%)",
     1,
     "newf"
    ],
    [
     "한국어로도 영어로도 못 찾던 용어 중 함께 적어서 찾은 것 → 6개",
     2
    ]
   ],
   "notes": [
    "두 이름을 함께 주면 한쪽 언어로 배운 연결이 약해도 다른 쪽이 메워 줘요. 이 노트가 용어를 처음 쓸 때 영어 이름을 같이 적는 이유이기도 해요. 문서를 쓸 때나 검색 질문을 만들 때 쓸 수 있는 가장 싼 방법이에요."
   ]
  }
 },
 "flow": [
  [
   1,
   "섞인 문장 세기",
   "영어 + 조사",
   "주언어는 한국어"
  ],
  [
   2,
   "용어 짝",
   "한국어, 영어, 설명",
   "토큰 두 벌"
  ],
  [
   3,
   "설명 찾기",
   "한국어 vs 영어",
   "같은 언어 쪽으로"
  ],
  [
   4,
   "교차 언어 일관성",
   "둘 다 찾나",
   "언어에 따라 다른 답"
  ],
  [
   5,
   "조금씩 섞기",
   "0%, 50%, 100%",
   "섞는 방향"
  ],
  [
   6,
   "함께 적기",
   "한국어 (영어)",
   "가장 싼 방법"
  ]
 ],
 "env": "Python 3.13.5, PyTorch 2.13.0+cpu, transformers 5.15.0, intfloat/multilingual-e5-small (CPU)"
};
