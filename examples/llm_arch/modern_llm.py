# 요즘 LLM의 부품: GPT-2(2019)의 블록이 Llama(2023) 이후 어떻게 바뀌었는지 작게 만들어 나란히 비교하는 전체 코드.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 모델은 받지 않아요. 설정(config)만으로 무작위 가중치 모델을 만들어 모양, 크기, 계산 방식을 봐요(학습은 하지 않아요).
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관). 모두 CPU, 스레드 4개.

import copy                                                      #@6
import time                                                      #@6
import torch                                                     #@1
import torch.nn.functional as F                                  #@3
from transformers import GPT2Config, GPT2LMHeadModel             #@1
from transformers import LlamaConfig, LlamaForCausalLM           #@1
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb   #@4
from transformers import MistralConfig, MistralForCausalLM       #@7
from transformers import MixtralConfig, MixtralForCausalLM       #@8

torch.set_num_threads(4)                                         #@1
H, LAYERS, HEADS, VOCAB = 256, 4, 8, 8000   # 작은 설정: 벡터 256칸, 4층, head 8개, 어휘 8,000   #@1

# ① 같은 크기의 GPT-2와 Llama
count = lambda m: sum(p.numel() for p in m.parameters())         # noqa: E731  가중치 수 (같이 쓰는 표는 한 번만)
torch.manual_seed(0)
gpt2 = GPT2LMHeadModel(GPT2Config(n_embd=H, n_layer=LAYERS, n_head=HEADS, vocab_size=VOCAB, n_positions=1024,
                                  bos_token_id=0, eos_token_id=0)).eval()
llama_cfg = dict(hidden_size=H, num_hidden_layers=LAYERS, num_attention_heads=HEADS, intermediate_size=688, vocab_size=VOCAB)
llama = LlamaForCausalLM(LlamaConfig(**llama_cfg, num_key_value_heads=HEADS)).eval()   # K, V head도 우선 8개 (⑤에서 줄여요)
g0, l0 = gpt2.transformer.h[0], llama.model.layers[0]           # 첫 번째 블록
parts = {"GPT-2": {"Token Embedding 표": count(gpt2.transformer.wte), "위치 표": count(gpt2.transformer.wpe),
                   "Attention (층마다)": count(g0.attn), "작은 신경망 (층마다)": count(g0.mlp), "정규화 (층마다)": count(g0.ln_1) + count(g0.ln_2),
                   "출력 층": 0 if gpt2.lm_head.weight is gpt2.transformer.wte.weight else count(gpt2.lm_head)},
         "Llama": {"Token Embedding 표": count(llama.model.embed_tokens), "위치 표": 0,
                   "Attention (층마다)": count(l0.self_attn), "작은 신경망 (층마다)": count(l0.mlp), "정규화 (층마다)": count(l0.input_layernorm) + count(l0.post_attention_layernorm),
                   "출력 층": count(llama.lm_head)}}
total = {"GPT-2": count(gpt2), "Llama": count(llama)}

# ② 정규화: LayerNorm → RMSNorm
x = gpt2.transformer.wte(torch.tensor([11, 22, 33])) * 10 + 3   # 토큰 벡터 3개 (평균이 0에서 멀게 3을 더함)
with torch.no_grad():
    ln = g0.ln_1(x)                                              # 평균을 빼고 표준편차로 나누기
    rms = l0.input_layernorm(x)                                  # 평균은 그대로 두고 RMS(제곱 평균의 제곱근)로만 나누기
    rms_by_hand = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-6)
    rms_centered = l0.input_layernorm(x - x.mean(-1, keepdim=True))   # 평균을 먼저 빼 주면?

# ③ 작은 신경망: GELU → SwiGLU
mlp = l0.mlp
with torch.no_grad():
    swiglu_by_hand = mlp.down_proj(F.silu(mlp.gate_proj(x)) * mlp.up_proj(x))   # 문(gate)을 연 만큼 올린 값(up)을 통과시켜 내리기(down)
    swiglu_gap = (swiglu_by_hand - mlp(x)).abs().max().item()

# ④ 위치: 위치 표 → RoPE (벡터를 자리만큼 돌리기)
q, k = torch.randn(1, 1, 1, H // HEADS), torch.randn(1, 1, 1, H // HEADS)   # head 하나의 Query와 Key (32칸)

def rope_score(i, j):                                            # Query를 i번째, Key를 j번째 자리에 두고 돌린 뒤 내적
    cos, sin = llama.model.rotary_emb(q, torch.tensor([[i, j]]))
    q_i, _ = apply_rotary_pos_emb(q, q, cos[:, :1], sin[:, :1])
    _, k_j = apply_rotary_pos_emb(k, k, cos[:, 1:], sin[:, 1:])
    return (q_i * k_j).sum().item()

with torch.no_grad():
    rope = {(i, j): rope_score(i, j) for i, j in [(5, 3), (105, 103), (2005, 2003), (5, 4), (5, 0)]}
max_pos = gpt2.transformer.wpe.num_embeddings                    # GPT-2 위치 표의 줄 수 = 넣을 수 있는 가장 긴 길이

# ⑤ Attention: K, V head를 나눠 쓰기 (MHA → GQA → MQA)
ids = torch.randint(0, VOCAB, (1, 300), generator=torch.Generator().manual_seed(1))   # 무작위 토큰 300개 (뜻은 없어요)
kv = {}
for kv_heads in (8, 2, 1):                                       # 8: MHA(head마다 K, V), 2: GQA(4개 head가 하나씩 같이 씀), 1: MQA
    torch.manual_seed(0)
    m = LlamaForCausalLM(LlamaConfig(**llama_cfg, num_key_value_heads=kv_heads)).eval()
    with torch.no_grad():
        cache = m(ids, use_cache=True).past_key_values           # 300토큰을 넣고 남은 KV 캐시
    kv[kv_heads] = {"shape": tuple(cache.layers[0].keys.shape), "bytes": sum(c.keys.numel() + c.values.numel() for c in cache.layers) * 4,
                    "attn": count(m.model.layers[0].self_attn)}

def kv_bytes(layers, kv_heads, head_dim, tokens, nbytes=2):     # K와 V 두 벌 × 층 × K,V head × head 크기 × 토큰 × 숫자 하나의 바이트
    return 2 * layers * kv_heads * head_dim * tokens * nbytes

big = {name: kv_bytes(32, h, 128, 8192) for name, h in (("MHA", 32), ("GQA", 8), ("MQA", 1))}   # 8B급 설정, 문맥 8,192토큰, 16비트

# ⑥ KV 캐시: prefill과 decode, 앞부분 다시 쓰기
torch.manual_seed(0)
model = LlamaForCausalLM(LlamaConfig(**llama_cfg, num_key_value_heads=2)).eval()   # GQA 모델
prompt = ids[:, :200]

def best_of(fn, repeat=5):                                       # 다섯 번 재서 가장 빠른 값 (초)
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best

with torch.no_grad():
    cache = model(prompt, use_cache=True).past_key_values        # prefill: 입력 200토큰을 한 번에 계산하며 캐시 만들기
    t_prefill = best_of(lambda: model(prompt, use_cache=True))
    t_decode = best_of(lambda: (model(ids[:, 200:201], past_key_values=cache, use_cache=True), cache.crop(-1)))   # decode: 1토큰만 계산 (재고 나서 그 1토큰을 캐시에서 빼기)
    gen = {True: [], False: []}
    for _ in range(3):
        for use in (True, False):                                # 캐시를 쓰는 생성과 안 쓰는 생성을 번갈아 재기
            t0 = time.perf_counter()
            model.generate(prompt, max_new_tokens=50, do_sample=False, use_cache=use, pad_token_id=0)
            gen[use].append(time.perf_counter() - t0)
    shared = model(prompt, use_cache=True).past_key_values       # 여러 요청이 같이 쓰는 앞부분(예: 시스템 Prompt) 200토큰의 캐시
    reuse_gap = []
    for tail in (ids[:, 200:220], ids[:, 220:240]):              # 서로 다른 질문 두 개 (20토큰씩)
        full = model(torch.cat([prompt, tail], 1)).logits[:, 200:]                                # 처음부터 220토큰 모두 계산
        reused = model(tail, past_key_values=copy.deepcopy(shared), use_cache=True).logits        # 앞부분 캐시 + 새 20토큰만 계산
        reuse_gap.append((full - reused).abs().max().item())

# ⑦ 슬라이딩 윈도우: 앞의 W개만 보기
torch.manual_seed(0)
mistral = MistralForCausalLM(MistralConfig(**llama_cfg, num_key_value_heads=2, sliding_window=64)).eval()
other = ids.clone()
other[0, 0] = (ids[0, 0] + 1) % VOCAB                            # 맨 앞 토큰 하나만 바꾼 입력
with torch.no_grad():
    cache_len = {n: (model(ids[:, :n], use_cache=True).past_key_values.layers[0].keys.shape[2],
                     mistral(ids[:, :n], use_cache=True).past_key_values.layers[0].keys.shape[2]) for n in (50, 100, 300)}
    far = {}
    for name, mdl in (("전체", model), ("윈도우 64", mistral)):
        a, b = mdl(ids).logits[0], mdl(other).logits[0]
        far[name] = {pos: (a[pos] - b[pos]).abs().max().item() for pos in (100, 299)}   # 맨 앞을 바꿨을 때 100번째, 299번째 출력이 변했나

# ⑧ MoE: 전문가 8개 중 2개만
torch.manual_seed(0)
moe = MixtralForCausalLM(MixtralConfig(**llama_cfg, num_key_value_heads=2, num_local_experts=8, num_experts_per_tok=2)).eval()
per_expert = count(moe.model.layers[0].mlp.experts) // 8        # 전문가 하나 = SwiGLU 하나
moe_total = count(moe)
moe_active = moe_total - LAYERS * (8 - 2) * per_expert           # 토큰 하나가 실제로 거치는 가중치 (8개 중 2개만)
with torch.no_grad():
    router = moe(prompt, output_router_logits=True).router_logits[0]   # 첫 층 라우터의 점수 (토큰 200개 × 전문가 8개)
picked = router.topk(2, dim=-1).indices                          # 토큰마다 고른 전문가 2개
load = torch.bincount(picked.flatten(), minlength=8)             # 전문가마다 맡은 토큰 수
