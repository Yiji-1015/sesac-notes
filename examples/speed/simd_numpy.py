# 같은 계산이 왜 수십 배 빨라지는지를 SIMD(한 명령으로 여러 칸을 한꺼번에), 자료형 폭, 메모리 배치, 캐시로 나눠 직접 재 보는 전체 코드.
# 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: 코드가 만드는 난수 배열과 20MB짜리 글자 버퍼. CPU로 1분쯤.
# 시간은 PC마다 달라요. 같은 PC 안에서 "몇 배 차이"를 보는 게 목적이에요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import timeit                                                                   #@1
import numpy as np                                                              #@1

def bench(f, repeat=5, number=3):                                # 여러 번 재서 가장 빠른 값 (다른 프로그램 영향 줄이기)
    return min(timeit.repeat(f, number=number, repeat=repeat)) / number

# ① 같은 일, 세 가지 방법
rng = np.random.default_rng(0)
text = bytes(rng.integers(32, 127, 20_000_000, dtype=np.uint8))   # 20MB 글자 (ASCII)
codes = np.frombuffer(text, dtype=np.uint8)
NEEDLE = ord("A")
part = text[:1_000_000]                                          # 파이썬 반복문은 느려서 1MB만 재고 20배
how_fast = {
    "파이썬 for 반복문": bench(lambda: sum(1 for b in part if b == NEEDLE), repeat=3, number=1) * 20,
    "bytes.count (C 함수)": bench(lambda: text.count(b"A")),
    "NumPy 비교 후 세기": bench(lambda: np.count_nonzero(codes == NEEDLE)),
}
answers = {text.count(b"A"), int(np.count_nonzero(codes == NEEDLE)), sum(1 for b in text[:200_000] if b == NEEDLE) + text[200_000:].count(b"A")}

# ② SIMD의 다섯 단계를 손으로
W = 32                                                           # AVX2 레지스터 하나 = 256비트 = 바이트 32칸
def count_simd_style(data, needle):
    lanes = np.full(W, needle, dtype=np.uint8)                   # 1. 찾을 값을 모든 칸에 복사 (broadcast)
    body = len(data) // W * W
    blocks = data[:body].reshape(-1, W)                          # 2. 레지스터 폭만큼씩 자르기
    hits = blocks == lanes                                       # 3. 칸마다 동시에 비교
    total = int(hits.sum())                                      # 4. 결과를 모아 하나로 (reduce)
    for b in data[body:]:                                        # 5. 폭에 안 맞는 꼬리는 하나씩 (scalar tail)
        total += int(b == needle)
    return total, len(data) - body

simd_count, tail_len = count_simd_style(codes[:1_000_013], NEEDLE)
plain_count = text[:1_000_013].count(b"A")

# ③ 자료형이 좁으면 한 번에 더 많은 칸
N = 20_000_000
base = rng.integers(0, 100, N)
widths = {}
for dt in (np.float64, np.float32, np.int32, np.int16, np.int8):
    a = base.astype(dt)
    widths[np.dtype(dt).name] = {"바이트": a.itemsize, "256비트에 들어가는 칸": 32 // a.itemsize,
                                 "max (ms)": bench(lambda: a.max()) * 1e3, "sum (ms)": bench(lambda: a.sum()) * 1e3}
int8_sum_dtype = base.astype(np.int8).sum().dtype                # sum은 넘치지 않게 int64로 넓혀서 더해요

# ④ 붙어 있는 메모리
big = rng.random(N * 4)
together, spread = np.ascontiguousarray(big[::4]), big[::4]      # 같은 2,000만 개, 붙어 있음 vs 4칸마다 하나
layout = {"붙어 있는 2,000만 개 합": bench(lambda: together.sum()), "4칸 건너 2,000만 개 합": bench(lambda: spread.sum())}
M = rng.random((4000, 4000))
layout["행렬 복사"] = bench(lambda: M.copy())
layout["전치해서 복사"] = bench(lambda: M.T.copy())                # 읽기는 세로로 건너뛰며

# ⑤ 캐시에 들어가는 크기인가
per_element = {}
for n in (2**12, 2**14, 2**16, 2**18, 2**20, 2**22, 2**24):
    v = rng.random(n)
    per_element[n] = bench(lambda: v.sum(), number=max(1, 2**24 // n)) / n * 1e9   # 원소 하나에 걸린 ns

# ⑥ 임시 배열 줄이기
x = rng.random(N)
buf = np.empty_like(x)
def with_temporaries():
    return (x * 2.0 + 1.0) * 3.0 - x                             # 연산마다 2,000만 칸짜리 새 배열
def in_place():
    b = np.multiply(x, 2.0, out=buf)                             # 미리 만든 버퍼 하나에 이어서
    b += 1.0
    b *= 3.0
    b -= x
    return b
same_result = bool(np.allclose(with_temporaries(), in_place()))
temporaries = {"새 배열을 만들며": bench(with_temporaries), "out= 과 제자리 연산": bench(in_place)}

# ⑦ if 대신 마스크
y = rng.standard_normal(2_000_000)
small_y = y[:200_000]
def relu_loop(v):
    return [t if t > 0 else 0.0 for t in v]                       # 원소마다 갈림길
branch = {
    "파이썬 if (20만 개 × 10)": bench(lambda: relu_loop(small_y), repeat=3, number=1) * 10,
    "np.where (마스크로 고르기)": bench(lambda: np.where(y > 0, y, 0.0)),
    "np.maximum (갈림길 없음)": bench(lambda: np.maximum(y, 0.0)),
}
