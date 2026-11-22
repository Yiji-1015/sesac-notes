window.PAGE = {
 "code": [
  {
   "t": "import timeit",
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
   "t": "def bench(f, repeat=5, number=3):  # 여러 번 재서 가장 빠른 값 (다른 프로그램 영향 줄이기)",
   "s": 1
  },
  {
   "t": "    return min(timeit.repeat(f, number=number, repeat=repeat)) / number",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 같은 일, 세 가지 방법",
   "s": 1
  },
  {
   "t": "rng = np.random.default_rng(0)",
   "s": 1
  },
  {
   "t": "text = bytes(rng.integers(32, 127, 20_000_000, dtype=np.uint8))  # 20MB 글자 (ASCII)",
   "s": 1
  },
  {
   "t": "codes = np.frombuffer(text, dtype=np.uint8)",
   "s": 1
  },
  {
   "t": "NEEDLE = ord(\"A\")",
   "s": 1
  },
  {
   "t": "part = text[:1_000_000]  # 파이썬 반복문은 느려서 1MB만 재고 20배",
   "s": 1
  },
  {
   "t": "how_fast = {",
   "s": 1
  },
  {
   "t": "    \"파이썬 for 반복문\": bench(lambda: sum(1 for b in part if b == NEEDLE), repeat=3, number=1) * 20,",
   "s": 1
  },
  {
   "t": "    \"bytes.count (C 함수)\": bench(lambda: text.count(b\"A\")),",
   "s": 1
  },
  {
   "t": "    \"NumPy 비교 후 세기\": bench(lambda: np.count_nonzero(codes == NEEDLE)),",
   "s": 1
  },
  {
   "t": "}",
   "s": 1
  },
  {
   "t": "answers = {text.count(b\"A\"), int(np.count_nonzero(codes == NEEDLE)), sum(1 for b in text[:200_000] if b == NEEDLE) + text[200_000:].count(b\"A\")}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② SIMD의 다섯 단계를 손으로",
   "s": 2
  },
  {
   "t": "W = 32  # AVX2 레지스터 하나 = 256비트 = 바이트 32칸",
   "s": 2
  },
  {
   "t": "def count_simd_style(data, needle):",
   "s": 2
  },
  {
   "t": "    lanes = np.full(W, needle, dtype=np.uint8)  # 1. 찾을 값을 모든 칸에 복사 (broadcast)",
   "s": 2
  },
  {
   "t": "    body = len(data) // W * W",
   "s": 2
  },
  {
   "t": "    blocks = data[:body].reshape(-1, W)  # 2. 레지스터 폭만큼씩 자르기",
   "s": 2
  },
  {
   "t": "    hits = blocks == lanes  # 3. 칸마다 동시에 비교",
   "s": 2
  },
  {
   "t": "    total = int(hits.sum())  # 4. 결과를 모아 하나로 (reduce)",
   "s": 2
  },
  {
   "t": "    for b in data[body:]:  # 5. 폭에 안 맞는 꼬리는 하나씩 (scalar tail)",
   "s": 2
  },
  {
   "t": "        total += int(b == needle)",
   "s": 2
  },
  {
   "t": "    return total, len(data) - body",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "simd_count, tail_len = count_simd_style(codes[:1_000_013], NEEDLE)",
   "s": 2
  },
  {
   "t": "plain_count = text[:1_000_013].count(b\"A\")",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 자료형이 좁으면 한 번에 더 많은 칸",
   "s": 3
  },
  {
   "t": "N = 20_000_000",
   "s": 3
  },
  {
   "t": "base = rng.integers(0, 100, N)",
   "s": 3
  },
  {
   "t": "widths = {}",
   "s": 3
  },
  {
   "t": "for dt in (np.float64, np.float32, np.int32, np.int16, np.int8):",
   "s": 3
  },
  {
   "t": "    a = base.astype(dt)",
   "s": 3
  },
  {
   "t": "    widths[np.dtype(dt).name] = {\"바이트\": a.itemsize, \"256비트에 들어가는 칸\": 32 // a.itemsize,",
   "s": 3
  },
  {
   "t": "                                 \"max (ms)\": bench(lambda: a.max()) * 1e3, \"sum (ms)\": bench(lambda: a.sum()) * 1e3}",
   "s": 3
  },
  {
   "t": "int8_sum_dtype = base.astype(np.int8).sum().dtype  # sum은 넘치지 않게 int64로 넓혀서 더해요",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 붙어 있는 메모리",
   "s": 4
  },
  {
   "t": "big = rng.random(N * 4)",
   "s": 4
  },
  {
   "t": "together, spread = np.ascontiguousarray(big[::4]), big[::4]  # 같은 2,000만 개, 붙어 있음 vs 4칸마다 하나",
   "s": 4
  },
  {
   "t": "layout = {\"붙어 있는 2,000만 개 합\": bench(lambda: together.sum()), \"4칸 건너 2,000만 개 합\": bench(lambda: spread.sum())}",
   "s": 4
  },
  {
   "t": "M = rng.random((4000, 4000))",
   "s": 4
  },
  {
   "t": "layout[\"행렬 복사\"] = bench(lambda: M.copy())",
   "s": 4
  },
  {
   "t": "layout[\"전치해서 복사\"] = bench(lambda: M.T.copy())  # 읽기는 세로로 건너뛰며",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 캐시에 들어가는 크기인가",
   "s": 5
  },
  {
   "t": "per_element = {}",
   "s": 5
  },
  {
   "t": "for n in (2**12, 2**14, 2**16, 2**18, 2**20, 2**22, 2**24):",
   "s": 5
  },
  {
   "t": "    v = rng.random(n)",
   "s": 5
  },
  {
   "t": "    per_element[n] = bench(lambda: v.sum(), number=max(1, 2**24 // n)) / n * 1e9  # 원소 하나에 걸린 ns",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 임시 배열 줄이기",
   "s": 6
  },
  {
   "t": "x = rng.random(N)",
   "s": 6
  },
  {
   "t": "buf = np.empty_like(x)",
   "s": 6
  },
  {
   "t": "def with_temporaries():",
   "s": 6
  },
  {
   "t": "    return (x * 2.0 + 1.0) * 3.0 - x  # 연산마다 2,000만 칸짜리 새 배열",
   "s": 6
  },
  {
   "t": "def in_place():",
   "s": 6
  },
  {
   "t": "    b = np.multiply(x, 2.0, out=buf)  # 미리 만든 버퍼 하나에 이어서",
   "s": 6
  },
  {
   "t": "    b += 1.0",
   "s": 6
  },
  {
   "t": "    b *= 3.0",
   "s": 6
  },
  {
   "t": "    b -= x",
   "s": 6
  },
  {
   "t": "    return b",
   "s": 6
  },
  {
   "t": "same_result = bool(np.allclose(with_temporaries(), in_place()))",
   "s": 6
  },
  {
   "t": "temporaries = {\"새 배열을 만들며\": bench(with_temporaries), \"out= 과 제자리 연산\": bench(in_place)}",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ if 대신 마스크",
   "s": 7
  },
  {
   "t": "y = rng.standard_normal(2_000_000)",
   "s": 7
  },
  {
   "t": "small_y = y[:200_000]",
   "s": 7
  },
  {
   "t": "def relu_loop(v):",
   "s": 7
  },
  {
   "t": "    return [t if t > 0 else 0.0 for t in v]  # 원소마다 갈림길",
   "s": 7
  },
  {
   "t": "branch = {",
   "s": 7
  },
  {
   "t": "    \"파이썬 if (20만 개 × 10)\": bench(lambda: relu_loop(small_y), repeat=3, number=1) * 10,",
   "s": 7
  },
  {
   "t": "    \"np.where (마스크로 고르기)\": bench(lambda: np.where(y > 0, y, 0.0)),",
   "s": 7
  },
  {
   "t": "    \"np.maximum (갈림길 없음)\": bench(lambda: np.maximum(y, 0.0)),",
   "s": 7
  },
  {
   "t": "}",
   "s": 7
  }
 ],
 "names": {
  "1": "같은 일, 세 가지 방법",
  "2": "SIMD의 다섯 단계를 손으로",
  "3": "자료형이 좁으면 한 번에 더 많은 칸",
  "4": "붙어 있는 메모리",
  "5": "캐시에 들어가는 크기인가",
  "6": "임시 배열 줄이기",
  "7": "if 대신 마스크"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 20MB에서 \"A\" 세기 (답은 셋 다 210,481개)",
   "cols": [
    "방법",
    "시간 (ms)",
    "가장 빠른 것의 몇 배"
   ],
   "rows": [
    [
     "파이썬 for 반복문",
     "353.60",
     "68.4배"
    ],
    [
     "bytes.count (C 함수)",
     "6.79",
     "1.3배"
    ],
    [
     "NumPy 비교 후 세기",
     "5.17",
     "1.0배"
    ]
   ],
   "notes": [
    "파이썬 반복문은 NumPy보다 68배 느려요. 바이트 하나마다 파이썬 객체를 만들고, 비교하고, 더하는 일을 인터프리터가 하나씩 하기 때문이에요.",
    "bytes.count와 NumPy는 C로 짠 반복문이 CPU의 SIMD 명령으로 한 번에 여러 바이트를 비교해요. 두 방법의 차이는 작아요. 이미 메모리에서 읽어 오는 속도에 가까워서예요."
   ]
  },
  "f_hf": {
   "title": "같은 일, 세 가지 방법",
   "images": [
    [
     "img/simd/three_ways.png",
     ""
    ]
   ],
   "notes": [
    "가로축은 10배마다 한 칸이에요. 파이썬 반복문은 두 칸 가까이 떨어져 있어요."
   ],
   "chart": true
  },
  "s2": {
   "title": "② 다섯 단계로 센 결과 (앞 1,000,013바이트)",
   "lines": [
    [
     "32칸씩 나눠 비교하고 모은 수 + 꼬리 13바이트를 하나씩 센 수 → 10,547",
     1,
     "newf"
    ],
    [
     "bytes.count로 센 수 → 10,547",
     2
    ]
   ],
   "notes": [
    "답이 같아요. NumPy가 속에서 하는 일을 레지스터 폭(32바이트)이 보이게 펼쳐 쓴 거예요. 실제 SIMD 코드는 이 다섯 단계를 CPU 명령으로 써요."
   ]
  },
  "s3": {
   "title": "③ 같은 2,000만 개를 자료형만 바꿔서",
   "cols": [
    "자료형",
    "한 칸 바이트",
    "256비트에 들어가는 칸",
    "max (ms)",
    "sum (ms)"
   ],
   "rows": [
    [
     "float64",
     8,
     4,
     "7.02",
     "12.03"
    ],
    [
     "float32",
     4,
     8,
     "3.36",
     "7.08"
    ],
    [
     "int32",
     4,
     8,
     "3.20",
     "7.98"
    ],
    [
     "int16",
     2,
     16,
     "1.45",
     "7.57"
    ],
    [
     "int8",
     1,
     32,
     "0.50",
     "7.48"
    ]
   ],
   "notes": [
    "max는 칸이 좁을수록 빨라져요. float64는 7.0ms, int8은 0.50ms로 14배예요. 한 명령에 4칸이 아니라 32칸을 비교하고, 메모리에서 읽을 양도 8분의 1이에요.",
    "sum은 거의 안 빨라져요. NumPy는 작은 정수를 더할 때 넘치지 않게 int64로 넓혀서 더하기 때문이에요. 넓히는 순간 칸이 다시 넓어져요. 빠르게 하려면 결과 자료형까지 같이 봐야 해요.",
    "모델 양자화(int8)가 빠른 이유의 절반이 이거예요. 같은 레지스터와 같은 메모리 대역에 네 배 많은 숫자가 들어가요."
   ]
  },
  "f_w": {
   "title": "자료형 폭과 시간",
   "images": [
    [
     "img/simd/widths.png",
     ""
    ]
   ],
   "notes": [
    "보라(max)는 오른쪽으로 갈수록 줄지만, 회색(sum)은 거의 그대로예요."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ 같은 양, 다른 배치",
   "cols": [
    "작업",
    "시간 (ms)"
   ],
   "rows": [
    [
     "붙어 있는 2,000만 개 합",
     "11.93"
    ],
    [
     "4칸 건너 2,000만 개 합",
     "40.15"
    ],
    [
     "행렬 복사",
     "25.13"
    ],
    [
     "전치해서 복사",
     "111.98"
    ]
   ],
   "notes": [
    "더하는 숫자 수는 같은데 4칸씩 건너뛰면 3.4배 느려요. 메모리는 64바이트(캐시 줄) 단위로 읽어 오는데, 건너뛰면 읽어 온 64바이트 중 8바이트만 쓰고 버려요. SIMD로 한 번에 담을 수도 없어요.",
    "4,000 × 4,000 행렬을 전치해서 복사하면 그냥 복사보다 4.5배 느려요. 쓰기는 차례대로지만 읽기가 한 줄(32KB)씩 건너뛰어요."
   ]
  },
  "f_c": {
   "title": "크기에 따른 원소 하나의 시간",
   "images": [
    [
     "img/simd/cache.png",
     ""
    ]
   ],
   "notes": [
    "작은 배열은 원소 하나에 0.17ns쯤인데, 32MB에서 0.72ns, 128MB에서 0.76ns로 뛰어요. 이 PC의 캐시를 넘어서 메모리(RAM)에서 읽기 시작하는 곳이에요.",
    "아주 작은 배열(32KB)이 조금 느린 건 계산보다 함수를 부르는 고정 비용이 커서예요."
   ],
   "chart": true
  },
  "s5": {
   "title": "⑤ 크기별 원소 하나의 시간",
   "cols": [
    "원소 수",
    "크기",
    "원소 하나에 (ns)"
   ],
   "rows": [
    [
     "4,096",
     "32 KB",
     "0.255"
    ],
    [
     "16,384",
     "128 KB",
     "0.192"
    ],
    [
     "65,536",
     "512 KB",
     "0.169"
    ],
    [
     "262,144",
     "2,048 KB",
     "0.181"
    ],
    [
     "1,048,576",
     "8,192 KB",
     "0.210"
    ],
    [
     "4,194,304",
     "32,768 KB",
     "0.717"
    ],
    [
     "16,777,216",
     "131,072 KB",
     "0.765"
    ]
   ],
   "notes": []
  },
  "s6": {
   "title": "⑥ (x × 2 + 1) × 3 − x 를 2,000만 개에",
   "cols": [
    "방법",
    "시간 (ms)",
    "가장 빠른 것의 몇 배"
   ],
   "rows": [
    [
     "새 배열을 만들며",
     "160.65",
     "3.4배"
    ],
    [
     "out= 과 제자리 연산",
     "46.57",
     "1.0배"
    ]
   ],
   "notes": [
    "결과는 같아요. 연산마다 새 배열을 만들면 2,000만 칸(160MB)짜리 임시 배열을 네 번 만들고 쓰고 읽어요. 미리 만든 버퍼 하나에 이어서 계산하면 3.4배 빨라요.",
    "곱하기, 더하기 자체는 빨라서, 이런 계산은 메모리를 얼마나 오가나가 시간을 정해요. 딥러닝 프레임워크가 연산 여러 개를 하나로 합치는 것(연산 융합)도 같은 이유예요."
   ]
  },
  "s7": {
   "title": "⑦ 음수를 0으로 (ReLU), 200만 개",
   "cols": [
    "방법",
    "시간 (ms)",
    "가장 빠른 것의 몇 배"
   ],
   "rows": [
    [
     "파이썬 if (20만 개 × 10)",
     "133.18",
     "39.9배"
    ],
    [
     "np.where (마스크로 고르기)",
     "4.42",
     "1.3배"
    ],
    [
     "np.maximum (갈림길 없음)",
     "3.34",
     "1.0배"
    ]
   ],
   "notes": [
    "np.where는 조건을 먼저 칸마다 참/거짓 배열(마스크)로 만들고, 두 값 중 하나를 고르는 일을 모든 칸에 똑같이 해요. 칸마다 다른 길로 가지 않으니 SIMD로 한꺼번에 할 수 있어요.",
    "np.maximum은 마스크도 만들지 않고 \"둘 중 큰 것\" 명령 하나로 끝나서 더 빨라요. 조건문을 계산식으로 바꿀 수 있으면 그게 가장 빨라요."
   ]
  }
 },
 "flow": [
  [
   1,
   "세 가지 방법",
   "반복문, C, NumPy",
   "수십 배"
  ],
  [
   2,
   "SIMD 다섯 단계",
   "복사, 자르기, 비교, 모으기, 꼬리",
   "폭 32바이트"
  ],
  [
   3,
   "자료형 폭",
   "float64 → int8",
   "칸 수와 넓히기"
  ],
  [
   4,
   "배치",
   "건너뛰기, 전치",
   "캐시 줄"
  ],
  [
   5,
   "캐시 크기",
   "작은 배열, 큰 배열",
   "RAM으로 가면"
  ],
  [
   6,
   "임시 배열",
   "out=",
   "메모리 오가기"
  ],
  [
   7,
   "갈림길 없애기",
   "마스크, maximum",
   "같은 일을 모든 칸에"
  ]
 ],
 "env": "Python 3.13.5, NumPy 2.5.2, NumPy SIMD: 기본 X86_V2, 찾음 X86_V3, Intel64 Family 6 Model 170 Stepping 4, GenuineIntel (시간은 PC마다 달라요)"
};
