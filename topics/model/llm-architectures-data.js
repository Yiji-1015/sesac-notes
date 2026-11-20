window.PAGE = {
 "code": [
  {
   "t": "import copy",
   "s": 6
  },
  {
   "t": "import time",
   "s": 6
  },
  {
   "t": "import torch",
   "s": 1
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 3
  },
  {
   "t": "from transformers import GPT2Config, GPT2LMHeadModel",
   "s": 1
  },
  {
   "t": "from transformers import LlamaConfig, LlamaForCausalLM",
   "s": 1
  },
  {
   "t": "from transformers.models.llama.modeling_llama import apply_rotary_pos_emb",
   "s": 4
  },
  {
   "t": "from transformers import MistralConfig, MistralForCausalLM",
   "s": 7
  },
  {
   "t": "from transformers import MixtralConfig, MixtralForCausalLM",
   "s": 8
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "torch.set_num_threads(4)",
   "s": 1
  },
  {
   "t": "H, LAYERS, HEADS, VOCAB = 256, 4, 8, 8000  # 작은 설정: 벡터 256칸, 4층, head 8개, 어휘 8,000",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 같은 크기의 GPT-2와 Llama",
   "s": 1
  },
  {
   "t": "count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731  가중치 수 (같이 쓰는 표는 한 번만)",
   "s": 1
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 1
  },
  {
   "t": "gpt2 = GPT2LMHeadModel(GPT2Config(n_embd=H, n_layer=LAYERS, n_head=HEADS, vocab_size=VOCAB, n_positions=1024,",
   "s": 1
  },
  {
   "t": "                                  bos_token_id=0, eos_token_id=0)).eval()",
   "s": 1
  },
  {
   "t": "llama_cfg = dict(hidden_size=H, num_hidden_layers=LAYERS, num_attention_heads=HEADS, intermediate_size=688, vocab_size=VOCAB)",
   "s": 1
  },
  {
   "t": "llama = LlamaForCausalLM(LlamaConfig(**llama_cfg, num_key_value_heads=HEADS)).eval()  # K, V head도 우선 8개 (⑤에서 줄여요)",
   "s": 1
  },
  {
   "t": "g0, l0 = gpt2.transformer.h[0], llama.model.layers[0]  # 첫 번째 블록",
   "s": 1
  },
  {
   "t": "parts = {\"GPT-2\": {\"Token Embedding 표\": count(gpt2.transformer.wte), \"위치 표\": count(gpt2.transformer.wpe),",
   "s": 1
  },
  {
   "t": "                   \"Attention (층마다)\": count(g0.attn), \"작은 신경망 (층마다)\": count(g0.mlp), \"정규화 (층마다)\": count(g0.ln_1) + count(g0.ln_2),",
   "s": 1
  },
  {
   "t": "                   \"출력 층\": 0 if gpt2.lm_head.weight is gpt2.transformer.wte.weight else count(gpt2.lm_head)},",
   "s": 1
  },
  {
   "t": "         \"Llama\": {\"Token Embedding 표\": count(llama.model.embed_tokens), \"위치 표\": 0,",
   "s": 1
  },
  {
   "t": "                   \"Attention (층마다)\": count(l0.self_attn), \"작은 신경망 (층마다)\": count(l0.mlp), \"정규화 (층마다)\": count(l0.input_layernorm) + count(l0.post_attention_layernorm),",
   "s": 1
  },
  {
   "t": "                   \"출력 층\": count(llama.lm_head)}}",
   "s": 1
  },
  {
   "t": "total = {\"GPT-2\": count(gpt2), \"Llama\": count(llama)}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 정규화: LayerNorm → RMSNorm",
   "s": 2
  },
  {
   "t": "x = gpt2.transformer.wte(torch.tensor([11, 22, 33])) * 10 + 3  # 토큰 벡터 3개 (평균이 0에서 멀게 3을 더함)",
   "s": 2
  },
  {
   "t": "with torch.no_grad():",
   "s": 2
  },
  {
   "t": "    ln = g0.ln_1(x)  # 평균을 빼고 표준편차로 나누기",
   "s": 2
  },
  {
   "t": "    rms = l0.input_layernorm(x)  # 평균은 그대로 두고 RMS(제곱 평균의 제곱근)로만 나누기",
   "s": 2
  },
  {
   "t": "    rms_by_hand = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)",
   "s": 2
  },
  {
   "t": "    rms_centered = l0.input_layernorm(x - x.mean(-1, keepdim=True))  # 평균을 먼저 빼 주면?",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 작은 신경망: GELU → SwiGLU",
   "s": 3
  },
  {
   "t": "mlp = l0.mlp",
   "s": 3
  },
  {
   "t": "with torch.no_grad():",
   "s": 3
  },
  {
   "t": "    swiglu_by_hand = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))  # 문(gate)을 연 만큼 올린 값(up)을 통과시켜 내리기(down)",
   "s": 3
  },
  {
   "t": "    swiglu_gap = (swiglu_by_hand - mlp(x)).abs().max().item()",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 위치: 위치 표 → RoPE (벡터를 자리만큼 돌리기)",
   "s": 4
  },
  {
   "t": "q, k = torch.randn(1, 1, 1, H // HEADS), torch.randn(1, 1, 1, H // HEADS)  # head 하나의 Query와 Key (32칸)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def rope_score(i, j):  # Query를 i번째, Key를 j번째 자리에 두고 돌린 뒤 내적",
   "s": 4
  },
  {
   "t": "    cos, sin = llama.model.rotary_emb(q, torch.tensor([[i, j]]))",
   "s": 4
  },
  {
   "t": "    q_i, _ = apply_rotary_pos_emb(q, q, cos[:, :1], sin[:, :1])",
   "s": 4
  },
  {
   "t": "    _, k_j = apply_rotary_pos_emb(k, k, cos[:, 1:], sin[:, 1:])",
   "s": 4
  },
  {
   "t": "    return (q_i * k_j).sum().item()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "with torch.no_grad():",
   "s": 4
  },
  {
   "t": "    rope = {(i, j): rope_score(i, j) for i, j in [(5, 3), (105, 103), (2005, 2003), (5, 4), (5, 0)]}",
   "s": 4
  },
  {
   "t": "max_pos = gpt2.transformer.wpe.num_embeddings  # GPT-2 위치 표의 줄 수 = 넣을 수 있는 가장 긴 길이",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ Attention: K, V head를 나눠 쓰기 (MHA → GQA → MQA)",
   "s": 5
  },
  {
   "t": "ids = torch.randint(0, VOCAB, (1, 300), generator=torch.Generator().manual_seed(1))  # 무작위 토큰 300개 (뜻은 없어요)",
   "s": 5
  },
  {
   "t": "kv = {}",
   "s": 5
  },
  {
   "t": "for kv_heads in (8, 2, 1):  # 8: MHA(head마다 K, V), 2: GQA(4개 head가 하나씩 같이 씀), 1: MQA",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "    m = LlamaForCausalLM(LlamaConfig(**llama_cfg, num_key_value_heads=kv_heads)).eval()",
   "s": 5
  },
  {
   "t": "    with torch.no_grad():",
   "s": 5
  },
  {
   "t": "        cache = m(ids, use_cache=True).past_key_values  # 300토큰을 넣고 남은 KV 캐시",
   "s": 5
  },
  {
   "t": "    kv[kv_heads] = {\"shape\": tuple(cache.layers[0].keys.shape), \"bytes\": sum(c.keys.numel() + c.values.numel() for c in cache.layers) * 4,",
   "s": 5
  },
  {
   "t": "                    \"attn\": count(m.model.layers[0].self_attn)}",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def kv_bytes(layers, kv_heads, head_dim, tokens, nbytes=2):  # K와 V 두 벌 × 층 × K,V head × head 크기 × 토큰 × 숫자 하나의 바이트",
   "s": 5
  },
  {
   "t": "    return 2 * layers * kv_heads * head_dim * tokens * nbytes",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "big = {name: kv_bytes(32, h, 128, 8192) for name, h in ((\"MHA\", 32), (\"GQA\", 8), (\"MQA\", 1))}  # 8B급 설정, 문맥 8,192토큰, 16비트",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ KV 캐시: prefill과 decode, 앞부분 다시 쓰기",
   "s": 6
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 6
  },
  {
   "t": "model = LlamaForCausalLM(LlamaConfig(**llama_cfg, num_key_value_heads=2)).eval()  # GQA 모델",
   "s": 6
  },
  {
   "t": "prompt = ids[:, :200]",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def best_of(fn, repeat=5):  # 다섯 번 재서 가장 빠른 값 (초)",
   "s": 6
  },
  {
   "t": "    best = float(\"inf\")",
   "s": 6
  },
  {
   "t": "    for _ in range(repeat):",
   "s": 6
  },
  {
   "t": "        t0 = time.perf_counter()",
   "s": 6
  },
  {
   "t": "        fn()",
   "s": 6
  },
  {
   "t": "        best = min(best, time.perf_counter() - t0)",
   "s": 6
  },
  {
   "t": "    return best",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    cache = model(prompt, use_cache=True).past_key_values  # prefill: 입력 200토큰을 한 번에 계산하며 캐시 만들기",
   "s": 6
  },
  {
   "t": "    t_prefill = best_of(lambda: model(prompt, use_cache=True))",
   "s": 6
  },
  {
   "t": "    t_decode = best_of(lambda: (model(ids[:, 200:201], past_key_values=cache, use_cache=True), cache.crop(-1)))  # decode: 1토큰만 계산 (재고 나서 그 1토큰을 캐시에서 빼기)",
   "s": 6
  },
  {
   "t": "    gen = {True: [], False: []}",
   "s": 6
  },
  {
   "t": "    for _ in range(3):",
   "s": 6
  },
  {
   "t": "        for use in (True, False):  # 캐시를 쓰는 생성과 안 쓰는 생성을 번갈아 재기",
   "s": 6
  },
  {
   "t": "            t0 = time.perf_counter()",
   "s": 6
  },
  {
   "t": "            model.generate(prompt, max_new_tokens=50, do_sample=False, use_cache=use, pad_token_id=0)",
   "s": 6
  },
  {
   "t": "            gen[use].append(time.perf_counter() - t0)",
   "s": 6
  },
  {
   "t": "    shared = model(prompt, use_cache=True).past_key_values  # 여러 요청이 같이 쓰는 앞부분(예: 시스템 Prompt) 200토큰의 캐시",
   "s": 6
  },
  {
   "t": "    reuse_gap = []",
   "s": 6
  },
  {
   "t": "    for tail in (ids[:, 200:220], ids[:, 220:240]):  # 서로 다른 질문 두 개 (20토큰씩)",
   "s": 6
  },
  {
   "t": "        full = model(torch.cat([prompt, tail], 1)).logits[:, 200:]  # 처음부터 220토큰 모두 계산",
   "s": 6
  },
  {
   "t": "        reused = model(tail, past_key_values=copy.deepcopy(shared), use_cache=True).logits  # 앞부분 캐시 + 새 20토큰만 계산",
   "s": 6
  },
  {
   "t": "        reuse_gap.append((full - reused).abs().max().item())",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 슬라이딩 윈도우: 앞의 W개만 보기",
   "s": 7
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 7
  },
  {
   "t": "mistral = MistralForCausalLM(MistralConfig(**llama_cfg, num_key_value_heads=2, sliding_window=64)).eval()",
   "s": 7
  },
  {
   "t": "other = ids.clone()",
   "s": 7
  },
  {
   "t": "other[0, 0] = (ids[0, 0] + 1) % VOCAB  # 맨 앞 토큰 하나만 바꾼 입력",
   "s": 7
  },
  {
   "t": "with torch.no_grad():",
   "s": 7
  },
  {
   "t": "    cache_len = {n: (model(ids[:, :n], use_cache=True).past_key_values.layers[0].keys.shape[2],",
   "s": 7
  },
  {
   "t": "                     mistral(ids[:, :n], use_cache=True).past_key_values.layers[0].keys.shape[2]) for n in (50, 100, 300)}",
   "s": 7
  },
  {
   "t": "    far = {}",
   "s": 7
  },
  {
   "t": "    for name, mdl in ((\"전체\", model), (\"윈도우 64\", mistral)):",
   "s": 7
  },
  {
   "t": "        a, b = mdl(ids).logits[0], mdl(other).logits[0]",
   "s": 7
  },
  {
   "t": "        far[name] = {pos: (a[pos] - b[pos]).abs().max().item() for pos in (100, 299)}  # 맨 앞을 바꿨을 때 100번째, 299번째 출력이 변했나",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ MoE: 전문가 8개 중 2개만",
   "s": 8
  },
  {
   "t": "torch.manual_seed(0)",
   "s": 8
  },
  {
   "t": "moe = MixtralForCausalLM(MixtralConfig(**llama_cfg, num_key_value_heads=2, num_local_experts=8, num_experts_per_tok=2)).eval()",
   "s": 8
  },
  {
   "t": "per_expert = count(moe.model.layers[0].mlp.experts) // 8  # 전문가 하나 = SwiGLU 하나",
   "s": 8
  },
  {
   "t": "moe_total = count(moe)",
   "s": 8
  },
  {
   "t": "moe_active = moe_total - LAYERS * (8 - 2) * per_expert  # 토큰 하나가 실제로 거치는 가중치 (8개 중 2개만)",
   "s": 8
  },
  {
   "t": "with torch.no_grad():",
   "s": 8
  },
  {
   "t": "    router = moe(prompt, output_router_logits=True).router_logits[0]  # 첫 층 라우터의 점수 (토큰 200개 × 전문가 8개)",
   "s": 8
  },
  {
   "t": "picked = router.topk(2, dim=-1).indices  # 토큰마다 고른 전문가 2개",
   "s": 8
  },
  {
   "t": "load = torch.bincount(picked.flatten(), minlength=8)  # 전문가마다 맡은 토큰 수",
   "s": 8
  }
 ],
 "names": {
  "1": "같은 크기의 GPT-2와 Llama",
  "2": "정규화: LayerNorm → RMSNorm",
  "3": "작은 신경망: GELU → SwiGLU",
  "4": "위치: 위치 표 → RoPE (벡터를 자리만큼 돌리기)",
  "5": "Attention: K, V head를 나눠 쓰기 (MHA → GQA → MQA)",
  "6": "KV 캐시: prefill과 decode, 앞부분 다시 쓰기",
  "7": "슬라이딩 윈도우: 앞의 W개만 보기",
  "8": "MoE: 전문가 8개 중 2개만"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "s1": {
   "title": "① 가중치 수 (벡터 256칸, 4층, head 8개, 어휘 8,000)",
   "cols": [
    "부품",
    "GPT-2",
    "Llama"
   ],
   "rows": [
    [
     "Token Embedding 표",
     "2,048,000",
     "2,048,000"
    ],
    [
     "위치 표",
     "262,144",
     "0"
    ],
    [
     "Attention (층마다)",
     "263,168",
     "262,144"
    ],
    [
     "작은 신경망 (층마다)",
     "525,568",
     "528,384"
    ],
    [
     "정규화 (층마다)",
     "1,024",
     "512"
    ],
    [
     "출력 층",
     "0",
     "2,048,000"
    ],
    [
     "<b>모두</b>",
     "<b>5,469,696</b>",
     "<b>7,260,416</b>"
    ]
   ],
   "notes": [
    "Attention과 작은 신경망은 1% 안쪽으로 같은 크기예요. 같은 크기로 맞춰서, 남은 차이는 부품의 모양뿐이에요.",
    "Llama에는 위치 표가 없어요. 위치를 표로 더하지 않고 Query와 Key를 돌려서 넣어요(④).",
    "Llama가 1,790,720개 더 많은 건 출력 층(2,048,000개) 때문이에요. GPT-2는 출력 층이 Token Embedding 표를 같이 쓰고(tied), 대신 위치 표(262,144개)가 있어요. 요즘 큰 모델은 대부분 따로 두고, 작은 모델은 같이 쓰기도 해요."
   ]
  },
  "s2": {
   "title": "② 토큰 벡터 3개 (칸 평균 약 3.0)에 두 정규화",
   "lines": [
    [
     "LayerNorm vs RMSNorm → 가장 큰 차이 3.67",
     1
    ],
    [
     "RMSNorm 손 계산 x ÷ √(평균(x²) + ε) vs 모듈 → 1.2e-07",
     2,
     "newf"
    ],
    [
     "평균을 먼저 뺀 RMSNorm vs LayerNorm → 3.4e-04",
     3
    ]
   ],
   "notes": [
    "입력의 평균이 0에서 멀면 두 결과가 많이 달라요. RMSNorm은 평균을 빼지 않으니까요.",
    "한 줄짜리 식이 모듈과 같은 값이에요. 처음 γ(칸마다 곱하는 값)는 모두 1이라 식에서 뺐어요.",
    "평균을 먼저 빼 주면 거의 같아져요. 남은 작은 차이는 두 모듈의 ε(LayerNorm 1e-05, RMSNorm 1e-06)이 달라서예요. 학습된 모델 안에서는 벡터 평균이 0 근처라, 평균을 빼는 계산을 생략해도 성능이 거의 같아요."
   ]
  },
  "s3": {
   "title": "③ 작은 신경망 두 가지",
   "lines": [
    [
     "GPT-2: 256 → 1,024 → 256, 행렬 2개 → 가중치 525,568개",
     1
    ],
    [
     "Llama: 256 → 688 (gate, up 두 갈래) → 256, 행렬 3개 → 가중치 528,384개",
     2,
     "newf"
    ],
    [
     "down(SiLU(gate(x)) × up(x)) 손 계산 vs 모듈 → 가장 큰 차이 0.0e+00",
     3
    ]
   ],
   "notes": [
    "안쪽을 벡터의 4배(1,024칸)로 넓혔다가 다시 줄여요.",
    "행렬이 하나 늘어서, 안쪽을 4배 대신 약 8/3배(688칸)로 줄여 가중치 수를 맞췄어요.",
    "모듈 안에서 하는 일이 정확히 이 한 줄이에요."
   ]
  },
  "s4": {
   "title": "④ Query와 Key를 자리만큼 돌린 뒤의 내적 (head 하나, 32칸)",
   "cols": [
    "Query 자리",
    "Key 자리",
    "거리",
    "점수"
   ],
   "rows": [
    [
     5,
     3,
     2,
     "-5.5273"
    ],
    [
     105,
     103,
     2,
     "-5.5273"
    ],
    [
     2005,
     2003,
     2,
     "-5.5274"
    ],
    [
     5,
     4,
     1,
     "-5.4513"
    ],
    [
     5,
     0,
     5,
     "-3.6278"
    ]
   ],
   "notes": [
    "거리가 2로 같으면 자리가 5든 2,005든 점수가 같아요(소수점 셋째 자리까지). 거리가 1이나 5로 바뀌면 점수도 바뀌어요.",
    "GPT-2의 위치 표는 1,024줄이라, 2,005번째 자리는 넣을 수조차 없어요. RoPE는 표가 없어서 어떤 자리든 돌릴 수 있어요. 다만 학습 때 본 길이보다 훨씬 길면 처음 보는 각도라 성능이 떨어져서, 각도를 늘려 맞추는 방법(YaRN, NTK, LongRoPE)을 써요."
   ]
  },
  "s5": {
   "title": "⑤ 300토큰을 넣은 뒤의 KV 캐시 (작은 모델, 4층, 32비트)",
   "cols": [
    "K, V head 수",
    "이름",
    "층 하나의 K 모양",
    "캐시 크기",
    "Attention 가중치 (층마다)"
   ],
   "rows": [
    [
     8,
     "<b>MHA</b>",
     "(1, 8, 300, 32)",
     "2.46 MB",
     "262,144"
    ],
    [
     2,
     "<b>GQA</b>",
     "(1, 2, 300, 32)",
     "0.61 MB",
     "163,840"
    ],
    [
     1,
     "<b>MQA</b>",
     "(1, 1, 300, 32)",
     "0.31 MB",
     "147,456"
    ]
   ],
   "notes": [
    "모양은 (문장 수, K head 수, 토큰 수, head 크기)예요. Query head는 셋 다 8개인데, K와 V를 저장하는 head만 8, 2, 1로 줄였어요. 캐시가 4분의 1, 8분의 1이 돼요.",
    "K, V를 만드는 행렬도 작아져서 Attention 가중치가 262,144개에서 147,456개까지 줄어요."
   ]
  },
  "t5": {
   "title": "8B급 설정이면 (32층, Query head 32개, head 크기 128, 16비트, 문맥 8,192토큰)",
   "cols": [
    "방식",
    "K, V head 수",
    "요청 하나의 KV 캐시"
   ],
   "rows": [
    [
     "MHA",
     32,
     "4.00 GB"
    ],
    [
     "GQA",
     8,
     "1.00 GB"
    ],
    [
     "MQA",
     1,
     "0.125 GB"
    ]
   ],
   "notes": [
    "가중치(80억 개 × 2바이트)가 약 15GB인데, MHA라면 8,192토큰 요청 4개만 동시에 받아도 KV 캐시가 가중치보다 커져요. GQA면 15개부터예요.",
    "MQA가 가장 작지만 품질이 떨어지기 쉬워서, 2024년 이후 새 모델은 GQA가 기본이에요."
   ]
  },
  "s6": {
   "title": "⑥ GQA 모델로 재기 (CPU, 다섯 번 중 가장 빠른 값)",
   "lines": [
    [
     "prefill: 입력 200토큰 한 번에 → 9.1 ms (토큰 하나당 0.045 ms)",
     1
    ],
    [
     "decode: 캐시를 읽으며 다음 1토큰 → 1.53 ms",
     2
    ],
    [
     "50토큰 생성: 캐시 씀 0.10초, 캐시 안 씀 0.58초",
     3,
     "newf"
    ],
    [
     "앞부분 200토큰 캐시 + 새 20토큰 vs 처음부터 220토큰 → 가장 큰 차이 1e-06",
     4,
     "newf"
    ]
   ],
   "notes": [
    "200토큰을 한꺼번에 행렬 곱으로 묶어 계산해요. 계산 장치를 꽉 채워 쓰는 단계라, 계산량이 시간을 정해요.",
    "토큰 하나만 계산하는데도 prefill의 토큰 하나당 시간보다 34배 걸려요. 1토큰을 위해서도 가중치 전체와 캐시를 메모리에서 한 번씩 읽어야 해서, 계산보다 메모리 읽기가 시간을 정해요.",
    "캐시가 없으면 새 토큰마다 앞 토큰 전체를 다시 계산해서 5.8배 느려요. 글이 길수록 차이가 더 벌어져요.",
    "앞 토큰의 K, V는 뒤에 무엇이 오든 변하지 않아서, 같은 앞부분의 캐시를 복사해 다시 써도 값이 같아요(남은 차이는 계산 순서에 따른 반올림). 요청마다 220토큰 대신 20토큰만 계산했어요."
   ]
  },
  "s7": {
   "title": "⑦ 층 하나의 KV 캐시 길이 (윈도우 W = 64)",
   "cols": [
    "입력 토큰 수",
    "전체 Attention",
    "슬라이딩 윈도우"
   ],
   "rows": [
    [
     50,
     50,
     50
    ],
    [
     100,
     100,
     63
    ],
    [
     300,
     300,
     63
    ]
   ],
   "notes": [
    "전체 Attention은 넣은 만큼 쌓이고, 윈도우 모델은 63개에서 멈춰요. 새 토큰이 들어오면 가장 오래된 것을 지워요."
   ]
  },
  "s7b": {
   "title": "맨 앞 토큰 하나만 바꿨을 때 출력이 변한 정도 (가장 큰 차이)",
   "cols": [
    "모델",
    "100번째 출력",
    "299번째 출력"
   ],
   "rows": [
    [
     "전체",
     "0.832",
     "0.739"
    ],
    [
     "윈도우 64",
     "0.735",
     "0.000"
    ]
   ],
   "notes": [
    "윈도우 모델에서 299번째 출력은 맨 앞 토큰을 전혀 몰라요. 한 층에서 63칸씩, 4층을 거치면 최대 252칸 앞까지만 닿거든요. 100번째는 그 안이라 바뀌어요.",
    "그래서 실제 모델은 윈도우 층과 전체 Attention 층을 번갈아 두거나(Gemma 2, GPT-OSS), 윈도우를 수천 토큰으로 넓게 잡아요."
   ]
  },
  "f7": {
   "title": "문맥 길이에 따른 KV 캐시",
   "images": [
    [
     "img/llm_arch/kv_cache_by_context.png",
     ""
    ]
   ],
   "notes": [
    "MHA로 문맥 128K면 요청 하나의 캐시가 64GB예요. GQA로 4분의 1, 윈도우를 더하면 4,096토큰부터 더 늘지 않아요(점선). 다만 윈도우만 쓰면 먼 토큰을 직접 못 봐서, 실제로는 전체 Attention 층과 섞어요."
   ],
   "chart": true
  },
  "s8": {
   "title": "⑧ 전문가 8개 중 2개 (작은 모델, 4층)",
   "lines": [
    [
     "전문가 하나 → 528,384개 (③의 SwiGLU 하나와 같은 크기)",
     1
    ],
    [
     "전체 가중치 21,670,144개, 토큰 하나가 거치는 가중치 8,988,928개 (41%)",
     2,
     "newf"
    ],
    [
     "첫 층에서 전문가마다 맡은 토큰 수 (200토큰 × 2) → 86, 16, 57, 41, 22, 99, 43, 36",
     3,
     "newf"
    ]
   ],
   "notes": [
    "작은 신경망을 같은 모양으로 8벌 만들었어요.",
    "Llama(7,260,416개)의 약 3배 크기지만, 토큰 하나의 계산은 전문가 2개 몫이라 1.2배쯤이에요. 대신 메모리에는 8벌이 모두 올라가 있어야 해요.",
    "학습 전 무작위 라우터인데도 16개에서 99개까지 고르지 않아요. 학습하면 잘하는 전문가로 더 몰리기 쉬워서, 몰림을 벌하는 보조 손실(Mixtral)이나 점수에 편향을 더해 맞추는 방법(DeepSeek V3)을 써요."
   ]
  }
 },
 "flow": [
  [
   1,
   "같은 크기로",
   "GPT-2와 Llama",
   "5.5M vs 7.3M"
  ],
  [
   2,
   "정규화",
   "LayerNorm → RMSNorm",
   "평균 빼기 생략"
  ],
  [
   3,
   "작은 신경망",
   "GELU → SwiGLU",
   "행렬 3개, 8/3배"
  ],
  [
   4,
   "위치",
   "표 → RoPE",
   "거리만 남는 점수"
  ],
  [
   5,
   "Attention",
   "MHA → GQA → MQA",
   "KV 1/4, 1/8"
  ],
  [
   6,
   "KV 캐시",
   "prefill과 decode",
   "캐시 없으면 5.8배"
  ],
  [
   7,
   "슬라이딩 윈도우",
   "앞의 W개만",
   "캐시 63에서 멈춤"
  ],
  [
   8,
   "MoE",
   "8개 중 2개",
   "활성 41%"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, transformers 5.22.2, CPU 스레드 4개, 가중치는 무작위(seed 0)"
};
