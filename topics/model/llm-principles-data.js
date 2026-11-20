window.PAGE = {
 "code": [
  {
   "t": "import math",
   "s": 3
  },
  {
   "t": "import torch",
   "s": 1
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 2
  },
  {
   "t": "from transformers import AutoTokenizer",
   "s": 1
  },
  {
   "t": "from transformers import AutoModel",
   "s": 2
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "MODEL = \"intfloat/multilingual-e5-small\"",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 글을 토큰으로",
   "s": 1
  },
  {
   "t": "tok = AutoTokenizer.from_pretrained(MODEL)",
   "s": 1
  },
  {
   "t": "texts = [\"배가 아파서 병원에 가요.\", \"배를 타고 섬에 가요.\", \"배를 깎아 먹었어요.\"]  # 같은 '배', 세 가지 뜻",
   "s": 1
  },
  {
   "t": "enc = tok(texts, return_tensors=\"pt\", padding=True)  # 짧은 문장 뒤는 채워서 한 묶음으로",
   "s": 1
  },
  {
   "t": "ids, mask = enc[\"input_ids\"], enc[\"attention_mask\"]",
   "s": 1
  },
  {
   "t": "tokens = [tok.convert_ids_to_tokens(row[m.bool()]) for row, m in zip(ids, mask)]",
   "s": 1
  },
  {
   "t": "BAE = 1  # 세 문장 모두 1번 자리가 '▁배'",
   "s": 1
  },
  {
   "t": "PAIRS = [(0, 1), (0, 2), (1, 2)]  # 몸-탈것, 몸-과일, 탈것-과일",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 토큰 벡터에 위치 더하기",
   "s": 2
  },
  {
   "t": "model = AutoModel.from_pretrained(MODEL, attn_implementation=\"eager\").eval()",
   "s": 2
  },
  {
   "t": "emb = model.embeddings",
   "s": 2
  },
  {
   "t": "pos = torch.arange(ids.shape[1])  # 자리 번호 0, 1, 2, …",
   "s": 2
  },
  {
   "t": "with torch.no_grad():",
   "s": 2
  },
  {
   "t": "    word = emb.word_embeddings(ids)  # 토큰 번호 → 벡터 (표에서 찾기)",
   "s": 2
  },
  {
   "t": "    x0 = emb.LayerNorm(word + emb.position_embeddings(pos) + emb.token_type_embeddings(torch.zeros_like(ids)))",
   "s": 2
  },
  {
   "t": "    same_embed = torch.allclose(x0, emb(ids), atol=1e-5)  # 모델 안의 계산과 같은가",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def cos(a, b):  # 두 벡터의 방향이 얼마나 같은가 (1이면 같음)",
   "s": 2
  },
  {
   "t": "    return F.cosine_similarity(a, b, dim=-1).item()",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "bae_table = [cos(word[a, BAE], word[b, BAE]) for a, b in PAIRS]  # 표에서 찾은 '배' 벡터끼리",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ head 하나의 Attention",
   "s": 3
  },
  {
   "t": "layer0 = model.encoder.layer[0]",
   "s": 3
  },
  {
   "t": "H, Dh = model.config.num_attention_heads, model.config.hidden_size // model.config.num_attention_heads  # 12개, 32칸",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def split(t):  # (문장, 토큰, 384) → (문장, head, 토큰, 32)",
   "s": 3
  },
  {
   "t": "    return t.view(t.shape[0], t.shape[1], H, Dh).transpose(1, 2)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "pad = (1 - mask[:, None, None, :]) * -1e9  # 채운 자리는 점수를 아주 낮게 → 보지 않음",
   "s": 3
  },
  {
   "t": "with torch.no_grad():",
   "s": 3
  },
  {
   "t": "    sa = layer0.attention.self",
   "s": 3
  },
  {
   "t": "    Q, K, V = split(sa.query(x0)), split(sa.key(x0)), split(sa.value(x0))",
   "s": 3
  },
  {
   "t": "    scores = Q @ K.transpose(-1, -2) / math.sqrt(Dh) + pad  # 모든 토큰 쌍의 맞음 점수",
   "s": 3
  },
  {
   "t": "    weights = scores.softmax(-1)  # (문장, head, 토큰, 토큰), 줄마다 합 1",
   "s": 3
  },
  {
   "t": "    out = model(**enc, output_attentions=True, output_hidden_states=True)",
   "s": 3
  },
  {
   "t": "    same_attn = torch.allclose(weights, out.attentions[0], atol=1e-5)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 여러 head를 이어 붙이기",
   "s": 4
  },
  {
   "t": "with torch.no_grad():",
   "s": 4
  },
  {
   "t": "    heads = weights @ V  # head마다 섞은 내용 (문장, head, 토큰, 32)",
   "s": 4
  },
  {
   "t": "    concat = heads.transpose(1, 2).reshape(x0.shape)  # 12개를 이어 붙여 다시 384칸",
   "s": 4
  },
  {
   "t": "    attn_out = layer0.attention.output.dense(concat)  # 이어 붙인 것을 한 번 더 섞기",
   "s": 4
  },
  {
   "t": "n0 = len(tokens[0])",
   "s": 4
  },
  {
   "t": "looks = [tokens[0][j] for j in weights[0, :, BAE, :n0].argmax(-1)]  # head마다 문장 1의 '▁배'가 가장 많이 본 토큰",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 층 하나 완성: 잔차 연결, LayerNorm, 작은 신경망",
   "s": 5
  },
  {
   "t": "def encoder_layer(x, layer, block):  # block: 보면 안 되는 자리에 더할 점수",
   "s": 5
  },
  {
   "t": "    sa = layer.attention.self",
   "s": 5
  },
  {
   "t": "    Q, K, V = split(sa.query(x)), split(sa.key(x)), split(sa.value(x))",
   "s": 5
  },
  {
   "t": "    w = (Q @ K.transpose(-1, -2) / math.sqrt(Dh) + block).softmax(-1)",
   "s": 5
  },
  {
   "t": "    a = layer.attention.output.dense((w @ V).transpose(1, 2).reshape(x.shape))",
   "s": 5
  },
  {
   "t": "    h = layer.attention.output.LayerNorm(x + a)  # 잔차 연결(입력을 그대로 더하기) + LayerNorm",
   "s": 5
  },
  {
   "t": "    f = layer.output.dense(F.gelu(layer.intermediate.dense(h)))  # 토큰마다 384 → 1536 → 384",
   "s": 5
  },
  {
   "t": "    return layer.output.LayerNorm(h + f)  # 다시 잔차 연결 + LayerNorm",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "with torch.no_grad():",
   "s": 5
  },
  {
   "t": "    x1 = encoder_layer(x0, layer0, pad)",
   "s": 5
  },
  {
   "t": "    same_layer = torch.allclose(x1, out.hidden_states[1], atol=1e-4)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 12층 쌓기: 문맥이 섞인 벡터",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    h = x0",
   "s": 6
  },
  {
   "t": "    bae_by_layer = [[cos(h[a, BAE], h[b, BAE]) for a, b in PAIRS]]  # 0번은 ②의 벡터",
   "s": 6
  },
  {
   "t": "    for layer in model.encoder.layer:",
   "s": 6
  },
  {
   "t": "        h = encoder_layer(h, layer, pad)",
   "s": 6
  },
  {
   "t": "        bae_by_layer.append([cos(h[a, BAE], h[b, BAE]) for a, b in PAIRS])",
   "s": 6
  },
  {
   "t": "    same_model = torch.allclose(h, out.last_hidden_state, atol=1e-3)",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 미래 가리기: 디코더는 뒤를 못 봐요",
   "s": 7
  },
  {
   "t": "future = torch.triu(torch.ones(len(pos), len(pos), dtype=torch.bool), diagonal=1)  # 대각선 위 = 자기보다 뒤",
   "s": 7
  },
  {
   "t": "with torch.no_grad():",
   "s": 7
  },
  {
   "t": "    hc = x0",
   "s": 7
  },
  {
   "t": "    for layer in model.encoder.layer:",
   "s": 7
  },
  {
   "t": "        hc = encoder_layer(hc, layer, pad + future * -1e9)  # 같은 층에 가리기만 더하기",
   "s": 7
  },
  {
   "t": "bae_causal = [cos(hc[a, BAE], hc[b, BAE]) for a, b in PAIRS]",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 다음 토큰 고르기 (점수 → 확률)",
   "s": 8
  },
  {
   "t": "candidates = [\"섬\", \"바다\", \"병원\", \"학교\"]  # \"배를 타고 ___에 가요\"의 빈칸 후보",
   "s": 8
  },
  {
   "t": "logits = torch.tensor([3.0, 2.2, 1.5, 0.3])  # 예시 점수: 디코더가 후보에 준 값이라고 가정",
   "s": 8
  },
  {
   "t": "probs = {t: torch.softmax(logits / t, dim=-1) for t in (0.5, 1.0, 2.0)}  # temperature",
   "s": 8
  },
  {
   "t": "greedy = candidates[int(logits.argmax())]",
   "s": 8
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 8
  },
  {
   "t": "sampled = [candidates[int(torch.multinomial(probs[1.0], 1))] for _ in range(10)]",
   "s": 8
  }
 ],
 "names": {
  "1": "글을 토큰으로",
  "2": "토큰 벡터에 위치 더하기",
  "3": "head 하나의 Attention",
  "4": "여러 head를 이어 붙이기",
  "5": "층 하나 완성: 잔차 연결, LayerNorm, 작은 신경망",
  "6": "12층 쌓기: 문맥이 섞인 벡터",
  "7": "미래 가리기: 디코더는 뒤를 못 봐요",
  "8": "다음 토큰 고르기 (점수 → 확률)"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "why": {
   "title": "세 문장의 '▁배' 벡터끼리 코사인 유사도 (1이면 방향이 같음)",
   "lines": [
    "짝 → 몸-탈것, 몸-과일, 탈것-과일",
    [
     "표에서 찾아온 벡터 → 1.000, 1.000, 1.000",
     1,
     "bad"
    ],
    [
     "Transformer 12층을 지난 뒤 → 0.924, 0.937, 0.939",
     2,
     "newf"
    ]
   ],
   "notes": [
    "토큰 번호로 표에서 찾아온 벡터는 문장과 상관없이 늘 같아요. \"배가 아파서\"의 배도, \"배를 타고\"의 배도 같은 벡터에서 출발해요.",
    "12층을 지나면 같은 토큰이 문장마다 다른 벡터가 돼요. 주변 토큰의 정보가 섞였기 때문이에요. 이 쪽은 그 섞는 과정을 한 단계씩 따라가요."
   ]
  },
  "s1": {
   "title": "① 토큰과 번호",
   "lines": [
    "문장 1 → ['<s>', '▁배', '가', '▁아', '파', '서', '▁병원', '에', '▁', '가요', '.', '</s>']",
    "문장 2 → ['<s>', '▁배', '를', '▁타고', '▁섬', '에', '▁', '가요', '.', '</s>']",
    "문장 3 → ['<s>', '▁배', '를', '▁', '깎', '아', '▁먹', '었어요', '.', '</s>']",
    [
     "ids.shape → (3, 12)",
     1
    ],
    [
     "mask[1] → [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]",
     2,
     "newf"
    ]
   ],
   "notes": [
    "<code>▁</code>는 \"앞에 띄어쓰기가 있던 조각\"이에요. 자주 나오는 말은 통째로, 드문 말은 잘게 나뉘어요. 세 문장 모두 1번 자리가 <code>▁배</code>예요. 0번 <code>&lt;s&gt;</code>와 끝의 <code>&lt;/s&gt;</code>는 시작과 끝 표시예요.",
    "세 문장의 토큰 수가 달라서(12, 10, 10개), 짧은 문장 뒤를 채워 (문장 수, 가장 긴 길이) 표 하나로 만들었어요. <code>mask</code>의 0이 채운 자리예요. <a href=\"sequence-models.html\">순서 모델</a> ①의 padding과 같은 방법이에요."
   ]
  },
  "f_pos": {
   "title": "e5의 위치 벡터 (학습으로 정해진 값)",
   "images": [
    [
     "img/llm/position_similarity.png",
     ""
    ]
   ],
   "notes": [
    "아무도 \"가까운 자리는 비슷하게\"라고 정해 주지 않았는데, 학습 결과 이웃한 자리끼리 비슷해졌어요. 10번 자리와 9, 11번은 평균 0.82, 40~63번과는 평균 0.16이에요.",
    "그래서 모델은 위치 벡터를 보고 \"몇 번째인가\"뿐 아니라 \"얼마나 떨어져 있나\"도 알 수 있어요."
   ],
   "chart": true
  },
  "s2": {
   "title": "② Embedding",
   "lines": [
    [
     "word.shape → (3, 12, 384)",
     1
    ],
    [
     "position_embeddings → (512, 384)",
     2
    ],
    [
     "x0와 모델의 emb(ids)가 같은가 → True",
     3,
     "newf"
    ],
    "표에서 찾은 '▁배' 벡터끼리 코사인 (몸-탈것, 몸-과일, 탈것-과일) → 1.000, 1.000, 1.000"
   ],
   "notes": [
    "(문장 3, 토큰 12, 384칸). 토큰마다 384칸 벡터 하나예요. 어휘가 250,037개라 표의 크기는 250,037 × 384예요.",
    "자리마다 위치 벡터도 하나씩 있어요. 자리가 512개뿐이라, 이 모델은 토큰을 512개까지만 받을 수 있어요. 입력 길이 한도(Context Window)가 여기서 정해져요.",
    "직접 더하고 LayerNorm한 값이 모델 안의 계산과 같아요. 모델이 하는 일을 그대로 따라 했다는 확인이에요."
   ]
  },
  "s3": {
   "title": "③ 첫 층의 Attention (head 6)",
   "lines": [
    [
     "Q.shape → (3, 12, 12, 32)",
     1
    ],
    "weights.shape → (3, 12, 12, 12), 줄마다 합 → [1.0, 1.0, 1.0] …",
    [
     "손으로 계산한 weights와 모델의 attentions[0]이 같은가 → True",
     2,
     "newf"
    ],
    [
     "문장 1, head 6에서 '▁배'가 많이 본 토큰 → ▁병원 0.14, <s> 0.13, 가요 0.13",
     3
    ]
   ],
   "notes": [
    "(문장 3, head 12, 토큰 12, 32칸). 384칸을 32칸씩 12조각으로 나눠 head마다 따로 계산해요 (④).",
    "e5의 실제 Query, Key, Value 행렬로 계산한 가중치가 모델이 낸 가중치와 같아요.",
    "이 head에서는 '▁배'가 <code>▁병원</code>을 가장 많이 봤어요(0.14). 다른 토큰보다 아주 크지는 않지만, '배'의 뜻을 정하는 데 쓸 만한 단어예요. 다른 head는 다른 곳을 봐요 (④)."
   ]
  },
  "f_head": {
   "title": "한 head의 Attention 가중치 (줄마다 합 1)",
   "images": [
    [
     "img/llm/attention_one_head.png",
     "진할수록 많이 봄"
    ]
   ],
   "notes": [
    "줄 하나가 \"이 토큰이 새 벡터를 만들 때 어느 토큰을 몇 %씩 섞나\"예요."
   ],
   "chart": true
  },
  "t3": {
   "title": "√d(= √32 ≈ 5.7)로 나누지 않으면 (문장 1, 첫 층 head 12개)",
   "cols": [
    "점수 계산",
    "줄마다 가장 큰 가중치의 평균",
    "모델과 같은가"
   ],
   "rows": [
    [
     "<b>QKᵀ ÷ √32</b> (모델)",
     "0.378",
     "True"
    ],
    [
     "QKᵀ 그대로",
     "0.793",
     "False"
    ]
   ],
   "notes": [
    "나누지 않으면 점수 차이가 커져서 softmax가 1등 하나에 더 몰려요(0.38 → 0.79). 한 토큰만 보고 나머지는 거의 무시하게 돼요.",
    "몰린 softmax는 기울기도 거의 0이라 학습이 잘 안 돼요. 칸 수 d가 클수록 내적이 커지니, √d로 나눠 크기를 맞춰요."
   ]
  },
  "f_heads": {
   "title": "첫 층 head 12개의 가중치 (문장 1, 줄: 보는 토큰, 칸: 보이는 토큰)",
   "images": [
    [
     "img/llm/attention_heads.png",
     ""
    ]
   ],
   "notes": [
    "head마다 무늬가 달라요. 대각선 바로 옆(앞이나 뒤 토큰)을 보는 head, 첫 칸이나 마지막 칸(<code>&lt;s&gt;</code>, <code>&lt;/s&gt;</code>)에 몰리는 head, 넓게 퍼진 head가 있어요.",
    "시작과 끝 표시에 몰리는 건 \"딱히 볼 곳이 없을 때 쉬는 자리\"처럼 쓰는 거예요. 큰 모델에서도 흔히 보여요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ 여러 head",
   "lines": [
    [
     "heads.shape → (3, 12, 12, 32)",
     1
    ],
    [
     "concat.shape → (3, 12, 384)",
     2
    ],
    "attn_out.shape → (3, 12, 384)",
    [
     "head마다 문장 1의 '▁배'가 가장 많이 본 토큰 → ['</s>', '</s>', '<s>', '<s>', '<s>', '<s>', '▁병원', '에', '</s>', '.', '</s>', '에']",
     3,
     "newf"
    ]
   ],
   "notes": [
    "head마다 32칸짜리 결과가 나와요.",
    "12개를 이어 붙이면 다시 384칸이에요. 그 뒤 384 × 384 행렬 하나로 한 번 더 섞어요(<code>attention.output.dense</code>).",
    "같은 '▁배'인데 head마다 본 곳이 5가지로 달라요. 한 head가 한 가지 관계만 잘 잡으니, 여러 벌을 두고 나눠 맡겨요."
   ]
  },
  "s5": {
   "title": "⑤ 첫 층 전체",
   "lines": [
    [
     "x1.shape → (3, 12, 384)",
     1
    ],
    [
     "모델의 hidden_states[1]과 같은가 → True (가장 큰 차이 4.8e-07)",
     2,
     "newf"
    ],
    [
     "층 하나의 가중치: Attention 591,360개, 작은 신경망 1,181,568개",
     3
    ]
   ],
   "notes": [
    "입력과 같은 모양이에요. 그래서 같은 층을 몇 번이고 쌓을 수 있어요 (⑥).",
    "Attention, 잔차 연결, LayerNorm, 작은 신경망까지 직접 이어 붙인 결과가 모델의 첫 층 출력과 같아요. 차이는 소수 계산의 반올림 수준이에요.",
    "Attention(Q, K, V, 출력 행렬 네 개)보다 토큰마다 통과하는 작은 신경망의 가중치가 2배 많아요. 모델이 배운 지식의 상당 부분이 이 작은 신경망에 들어 있다고 봐요."
   ]
  },
  "t5": {
   "title": "문장 1의 가운데 토큰 순서를 거꾸로 해서 같은 층에 넣으면",
   "cols": [
    "입력",
    "결과가 \"순서만 바뀐 같은 값\"인가"
   ],
   "rows": [
    [
     "토큰 벡터만",
     "<b>예</b>"
    ],
    [
     "토큰 벡터 + 위치 벡터 (②)",
     "<b>아니오</b>"
    ]
   ],
   "notes": [
    "위치 벡터가 없으면 \"배가 아파서 병원에 가요.\"를 거꾸로 넣어도 토큰마다 같은 벡터가 나와요. Attention과 작은 신경망 어디에도 \"몇 번째\"가 없어서, 층은 토큰들을 순서 없는 집합으로 봐요.",
    "위치 벡터를 더하면 같은 토큰도 자리에 따라 다른 입력이 돼서, 순서가 결과에 들어가요. ②에서 위치를 더하는 이유예요."
   ]
  },
  "f_layers": {
   "title": "세 문장의 '▁배' 벡터, 층마다",
   "images": [
    [
     "img/llm/bae_by_layer.png",
     ""
    ]
   ],
   "notes": [
    "0층(표에서 찾은 벡터 + 위치)에서는 셋 다 1이에요. 층을 지나며 내려가서(가장 낮은 곳은 8층), 12층에서는 0.924, 0.937, 0.939예요.",
    "벡터가 달라졌다는 것까지가 이 그림이 보여 주는 거예요. 그 차이가 \"몸, 탈것, 과일\"을 잘 가르는지는 따로 재야 해요. 그 벡터 위에 분류용 층을 얹어 학습시키는 게 미세조정이에요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ 12층",
   "lines": [
    [
     "모델의 last_hidden_state와 같은가 → True (가장 큰 차이 7.2e-07)",
     1,
     "newf"
    ],
    "0층 뒤 '▁배' 코사인 (몸-탈것, 몸-과일, 탈것-과일) → 1.000, 1.000, 1.000",
    "1층 뒤 '▁배' 코사인 (몸-탈것, 몸-과일, 탈것-과일) → 0.996, 0.993, 0.995",
    "6층 뒤 '▁배' 코사인 (몸-탈것, 몸-과일, 탈것-과일) → 0.928, 0.953, 0.956",
    "12층 뒤 '▁배' 코사인 (몸-탈것, 몸-과일, 탈것-과일) → 0.924, 0.937, 0.939"
   ],
   "notes": [
    "⑤의 층 함수를 12번 부른 결과가 모델의 마지막 출력과 같아요. Transformer 인코더는 결국 이 층의 반복이에요.",
    "층마다 주변 토큰의 정보가 조금씩 더 섞여요."
   ]
  },
  "t6": {
   "title": "토큰 수를 늘리며 첫 층 하나의 시간 재기 (CPU, 세 번 중 가장 빠른 값)",
   "cols": [
    "토큰 수",
    "Attention",
    "작은 신경망",
    "Attention의 몫",
    "점수표 메모리 (head 12개)"
   ],
   "rows": [
    [
     "256",
     "0.6 ms",
     "0.7 ms",
     "<b>44%</b>",
     "3 MB"
    ],
    [
     "1,024",
     "10.2 ms",
     "2.8 ms",
     "<b>78%</b>",
     "48 MB"
    ],
    [
     "2,048",
     "42.4 ms",
     "4.2 ms",
     "<b>91%</b>",
     "192 MB"
    ]
   ],
   "notes": [
    "작은 신경망은 토큰마다 따로 계산해서 토큰 수에 비례해 늘어요. Attention은 모든 토큰 쌍의 점수표를 만들어서 토큰 수의 제곱으로 늘어요. 그래서 짧을 때는 작은 신경망이, 길어질수록 Attention이 시간을 차지해요(44% → 91%).",
    "점수표는 토큰이 4배가 되면 16배가 돼요. 이 모델은 위치 벡터가 512개라 실제로는 512토큰까지만 받지만, 긴 문맥을 다루는 모델에서는 이게 가장 큰 비용이에요. 시간은 PC마다 달라요."
   ]
  },
  "s7": {
   "title": "⑦ 미래를 가리고 12층",
   "lines": [
    [
     "'▁배' 코사인 (몸-탈것, 몸-과일, 탈것-과일) → 1.000, 1.000, 1.000",
     1,
     "bad"
    ],
    [
     "문장 2, 3의 2번 자리 '를' → 가리면 1.000, 안 가리면 0.896",
     2
    ]
   ],
   "notes": [
    "뒤를 못 보니 '▁배'는 앞의 <code>&lt;s&gt;</code>와 자기 자신만 봐요. 세 문장 모두 앞부분이 같아서 12층을 지나도 벡터가 똑같아요. \"아파서\"인지 \"타고\"인지 아직 모르는 거예요.",
    "문장 2, 3은 \"배를\"까지 같아서 '를'도 똑같아요. 디코더는 뒤 단어로 앞 단어의 뜻을 고칠 수 없어서, 다음 토큰을 맞힐 때마다 앞까지의 정보로만 판단해요. 인코더는 앞뒤를 다 봐서 문장 이해에 강하고, 디코더는 글을 앞에서부터 만드는 데 맞아요."
   ]
  },
  "f_mask": {
   "title": "세 갈래의 Transformer가 \"누가 누구를 보나\"",
   "images": [
    [
     "img/llm/attention_masks.png",
     "진한 칸: 볼 수 있음, 흰 칸: 가림"
    ]
   ],
   "notes": [
    "⑦에서 더한 미래 가리기가 가운데 모양이에요. 같은 층에 가리는 모양만 바꿔서 세 갈래가 나와요."
   ],
   "chart": true
  },
  "s8": {
   "title": "⑧ 빈칸 후보 네 개의 확률 (점수는 예시)",
   "cols": [
    "후보",
    "점수(logit)",
    "temperature 0.5",
    "temperature 1.0",
    "temperature 2.0"
   ],
   "rows": [
    [
     "섬",
     "3.0",
     "0.80",
     "0.57",
     "0.42"
    ],
    [
     "바다",
     "2.2",
     "0.16",
     "0.26",
     "0.28"
    ],
    [
     "병원",
     "1.5",
     "0.04",
     "0.13",
     "0.20"
    ],
    [
     "학교",
     "0.3",
     "0.00",
     "0.04",
     "0.11"
    ]
   ],
   "notes": [
    "가장 높은 것만 고르면(greedy) 언제나 \"섬\"이에요.",
    "확률대로 뽑으면(temperature 1.0, 10번) → ['바다', '병원', '바다', '섬', '바다', '섬', '섬', '바다', '섬', '섬']. 같은 질문에도 답이 조금씩 달라지는 이유예요.",
    "temperature가 낮으면 1등에 몰리고(0.5에서 0.80), 높으면 고르게 퍼져요(2.0에서 0.42). 정확해야 하면 낮게, 다양해야 하면 높게 써요."
   ]
  }
 },
 "flow": [
  [
   1,
   "토큰",
   "세 문장, 패딩",
   "▁배는 1번 자리"
  ],
  [
   2,
   "Embedding",
   "표 + 위치",
   "'배'는 아직 같음"
  ],
  [
   3,
   "head 하나",
   "Q, K, V, √d",
   "모델과 같음"
  ],
  [
   4,
   "여러 head",
   "12 × 32칸",
   "보는 곳이 다름"
  ],
  [
   5,
   "층 하나",
   "잔차, LayerNorm, FFN",
   "모델과 같음"
  ],
  [
   6,
   "12층",
   "층 반복",
   "'배'가 달라짐"
  ],
  [
   7,
   "미래 가리기",
   "디코더",
   "'배'가 그대로"
  ],
  [
   8,
   "다음 토큰",
   "softmax",
   "temperature"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, transformers 5.22.2, intfloat/multilingual-e5-small, CPU"
};
