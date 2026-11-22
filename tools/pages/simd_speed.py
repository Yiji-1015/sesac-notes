"""백엔드 > 성능과 하드웨어 > SIMD와 메모리. examples/speed/simd_numpy.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/simd_speed.py
"""
import sys
import platform
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "speed"
code = S.parse((EX / "simd_numpy.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, S.workdir(EX), filename="simd_numpy.py")
try:
    simd = np.__config__.CONFIG["SIMD Extensions"]
    simd_txt = f", NumPy SIMD: 기본 {', '.join(simd['baseline'])}, 찾음 {', '.join(simd['found'])}"
except Exception:
    simd_txt = ""
ENV = f"Python {sys.version.split()[0]}, NumPy {np.__version__}{simd_txt}, {platform.processor() or platform.machine()} (시간은 PC마다 달라요)"
ms = lambda s: f"{s * 1e3:.2f}"  # noqa: E731

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
IMG = ROOT / "topics" / "backend" / "img" / "simd"
IMG.mkdir(parents=True, exist_ok=True)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/simd/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


def times(title, d, notes, base=None):
    base = base or min(d.values())
    return S.table(title, ["방법", "시간 (ms)", "가장 빠른 것의 몇 배"], [[k, ms(v), f"{v / base:.1f}배"] for k, v in d.items()], notes)


outs = {}
# ---------------- ①
hf = ns["how_fast"]
loop, npc = hf["파이썬 for 반복문"], hf["NumPy 비교 후 세기"]
S.claim(len(ns["answers"]) == 1 and loop / npc > 20, "① 세 방법 답이 같고 반복문이 20배 넘게 느림")
outs["s1"] = times(f"① 20MB에서 \"A\" 세기 (답은 셋 다 {next(iter(ns['answers'])):,}개)", hf,
                   [f"파이썬 반복문은 NumPy보다 {loop / npc:.0f}배 느려요. 바이트 하나마다 파이썬 객체를 만들고, 비교하고, 더하는 일을 인터프리터가 하나씩 하기 때문이에요.",
                    "bytes.count와 NumPy는 C로 짠 반복문이 CPU의 SIMD 명령으로 한 번에 여러 바이트를 비교해요. 두 방법의 차이는 작아요. 이미 메모리에서 읽어 오는 속도에 가까워서예요."])
fig, ax = plt.subplots(figsize=(7.6, 2.6))
names = list(hf)
ax.barh(names[::-1], [hf[n] * 1e3 for n in names[::-1]], color=[C["purple"], C["blue"], C["pink"]])
ax.set_xscale("log")
ticks = [1, 10, 100, 1000]
ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
ax.set(xlabel="시간 (ms, 로그 눈금)", title="같은 일, 세 가지 방법")
P_HF = fig_save(fig, "three_ways")
outs["f_hf"] = chart("같은 일, 세 가지 방법", P_HF, "", ["가로축은 10배마다 한 칸이에요. 파이썬 반복문은 두 칸 가까이 떨어져 있어요."])

# ---------------- ②
S.claim(ns["simd_count"] == ns["plain_count"], "② 손으로 한 다섯 단계 답이 같음")
outs["s2"] = S.box("② 다섯 단계로 센 결과 (앞 1,000,013바이트)", [(f"32칸씩 나눠 비교하고 모은 수 + 꼬리 {ns['tail_len']}바이트를 하나씩 센 수 → {ns['simd_count']:,}", 1, "newf"),
                                                            (f"bytes.count로 센 수 → {ns['plain_count']:,}", 2)],
                   ["답이 같아요. NumPy가 속에서 하는 일을 레지스터 폭(32바이트)이 보이게 펼쳐 쓴 거예요. 실제 SIMD 코드는 이 다섯 단계를 CPU 명령으로 써요."])

# ---------------- ③
wd = ns["widths"]
S.claim(wd["int8"]["max (ms)"] * 4 < wd["float64"]["max (ms)"] and wd["int8"]["sum (ms)"] * 3 > wd["float64"]["sum (ms)"], "③ max는 int8이 4배 넘게 빠르고 sum은 3배도 안 빠름")
outs["s3"] = S.table("③ 같은 2,000만 개를 자료형만 바꿔서", ["자료형", "한 칸 바이트", "256비트에 들어가는 칸", "max (ms)", "sum (ms)"],
                     [[k, v["바이트"], v["256비트에 들어가는 칸"], f"{v['max (ms)']:.2f}", f"{v['sum (ms)']:.2f}"] for k, v in wd.items()],
                     [f"max는 칸이 좁을수록 빨라져요. float64는 {wd['float64']['max (ms)']:.1f}ms, int8은 {wd['int8']['max (ms)']:.2f}ms로 {wd['float64']['max (ms)'] / wd['int8']['max (ms)']:.0f}배예요. 한 명령에 4칸이 아니라 32칸을 비교하고, 메모리에서 읽을 양도 8분의 1이에요.",
                      f"sum은 거의 안 빨라져요. NumPy는 작은 정수를 더할 때 넘치지 않게 {ns['int8_sum_dtype']}로 넓혀서 더하기 때문이에요. 넓히는 순간 칸이 다시 넓어져요. 빠르게 하려면 결과 자료형까지 같이 봐야 해요.",
                      "모델 양자화(int8)가 빠른 이유의 절반이 이거예요. 같은 레지스터와 같은 메모리 대역에 네 배 많은 숫자가 들어가요."])
fig, ax = plt.subplots(figsize=(7.6, 3.2))
xs = np.arange(len(wd))
ax.bar(xs - 0.2, [v["max (ms)"] for v in wd.values()], width=0.4, color=C["purple"], label="max (자료형 그대로)")
ax.bar(xs + 0.2, [v["sum (ms)"] for v in wd.values()], width=0.4, color=C["gray"], label="sum (정수는 int64로 넓혀서)")
ax.set_xticks(xs, [f"{k}\n({v['256비트에 들어가는 칸']}칸)" for k, v in wd.items()])
ax.set(ylabel="시간 (ms)", title="좁은 자료형, 더 많은 칸")
ax.legend(fontsize=9)
P_W = fig_save(fig, "widths")
outs["f_w"] = chart("자료형 폭과 시간", P_W, "", ["보라(max)는 오른쪽으로 갈수록 줄지만, 회색(sum)은 거의 그대로예요."])

# ---------------- ④
ly = ns["layout"]
S.claim(ly["4칸 건너 2,000만 개 합"] > 1.5 * ly["붙어 있는 2,000만 개 합"] and ly["전치해서 복사"] > 2 * ly["행렬 복사"], "④ 건너뛰기와 전치 복사가 뚜렷이 느림")
outs["s4"] = S.table("④ 같은 양, 다른 배치", ["작업", "시간 (ms)"], [[k, ms(v)] for k, v in ly.items()],
                     [f"더하는 숫자 수는 같은데 4칸씩 건너뛰면 {ly['4칸 건너 2,000만 개 합'] / ly['붙어 있는 2,000만 개 합']:.1f}배 느려요. 메모리는 64바이트(캐시 줄) 단위로 읽어 오는데, 건너뛰면 읽어 온 64바이트 중 8바이트만 쓰고 버려요. SIMD로 한 번에 담을 수도 없어요.",
                      f"4,000 × 4,000 행렬을 전치해서 복사하면 그냥 복사보다 {ly['전치해서 복사'] / ly['행렬 복사']:.1f}배 느려요. 쓰기는 차례대로지만 읽기가 한 줄(32KB)씩 건너뛰어요."])

# ---------------- ⑤
pe = ns["per_element"]
sizes = sorted(pe)
S.claim(pe[sizes[-1]] > 2.5 * min(pe.values()), "⑤ 가장 큰 배열은 원소당 시간이 2.5배 넘게")
fig, ax = plt.subplots(figsize=(7.6, 3.2))
ax.plot([n * 8 / 1024 for n in sizes], [pe[n] for n in sizes], "o-", color=C["purple"], lw=2.4)
ax.set_xscale("log")
kb = [32, 512, 8192, 131072]
ax.set_xticks(kb, ["32KB", "512KB", "8MB", "128MB"])
ax.set(xlabel="배열 크기 (float64)", ylabel="원소 하나에 ns", title="캐시를 넘으면 느려져요", ylim=(0, None))
P_C = fig_save(fig, "cache")
jump = next(n for a, n in zip(sizes, sizes[1:]) if pe[n] > 2 * pe[a])
outs["f_c"] = chart("크기에 따른 원소 하나의 시간", P_C, "",
                    [f"작은 배열은 원소 하나에 {min(pe.values()):.2f}ns쯤인데, {jump * 8 // 1024 // 1024}MB에서 {pe[jump]:.2f}ns, {sizes[-1] * 8 // 1024 // 1024}MB에서 {pe[sizes[-1]]:.2f}ns로 뛰어요. 이 PC의 캐시를 넘어서 메모리(RAM)에서 읽기 시작하는 곳이에요.",
                     "아주 작은 배열(32KB)이 조금 느린 건 계산보다 함수를 부르는 고정 비용이 커서예요."])
outs["s5"] = S.table("⑤ 크기별 원소 하나의 시간", ["원소 수", "크기", "원소 하나에 (ns)"], [[f"{n:,}", f"{n * 8 / 1024:,.0f} KB", f"{pe[n]:.3f}"] for n in sizes], [])

# ---------------- ⑥
tm = ns["temporaries"]
S.claim(ns["same_result"] and tm["새 배열을 만들며"] > 1.5 * tm["out= 과 제자리 연산"], "⑥ 결과가 같고 제자리가 1.5배 넘게 빠름")
outs["s6"] = times("⑥ (x × 2 + 1) × 3 − x 를 2,000만 개에", tm,
                   [f"결과는 같아요. 연산마다 새 배열을 만들면 2,000만 칸(160MB)짜리 임시 배열을 네 번 만들고 쓰고 읽어요. 미리 만든 버퍼 하나에 이어서 계산하면 {tm['새 배열을 만들며'] / tm['out= 과 제자리 연산']:.1f}배 빨라요.",
                    "곱하기, 더하기 자체는 빨라서, 이런 계산은 메모리를 얼마나 오가나가 시간을 정해요. 딥러닝 프레임워크가 연산 여러 개를 하나로 합치는 것(연산 융합)도 같은 이유예요."])

# ---------------- ⑦
br = ns["branch"]
S.claim(br["파이썬 if (20만 개 × 10)"] > 10 * br["np.maximum (갈림길 없음)"], "⑦ 파이썬 if가 10배 넘게 느림")
outs["s7"] = times("⑦ 음수를 0으로 (ReLU), 200만 개", br,
                   ["np.where는 조건을 먼저 칸마다 참/거짓 배열(마스크)로 만들고, 두 값 중 하나를 고르는 일을 모든 칸에 똑같이 해요. 칸마다 다른 길로 가지 않으니 SIMD로 한꺼번에 할 수 있어요.",
                    "np.maximum은 마스크도 만들지 않고 \"둘 중 큰 것\" 명령 하나로 끝나서 더 빨라요. 조건문을 계산식으로 바꿀 수 있으면 그게 가장 빨라요."])

flow = [[1, "세 가지 방법", "반복문, C, NumPy", "수십 배"], [2, "SIMD 다섯 단계", "복사, 자르기, 비교, 모으기, 꼬리", "폭 32바이트"], [3, "자료형 폭", "float64 → int8", "칸 수와 넓히기"], [4, "배치", "건너뛰기, 전치", "캐시 줄"],
        [5, "캐시 크기", "작은 배열, 큰 배열", "RAM으로 가면"], [6, "임시 배열", "out=", "메모리 오가기"], [7, "갈림길 없애기", "마스크, maximum", "같은 일을 모든 칸에"]]
S.save(ROOT / "topics" / "backend" / "simd-speed-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = """<p>\"파이썬은 느리니 NumPy를 써라\"는 말 뒤에는 CPU가 일하는 방식 두 가지가 있어요. 하나는 <b>SIMD</b>(Single Instruction, Multiple Data)예요. 요즘 CPU는 256비트(AVX2)나 512비트 레지스터 하나에 숫자 여러 개를 넣고, 명령 하나로 그 칸들을 한꺼번에 비교하거나 더해요. 다른 하나는 <b>메모리</b>예요. 계산보다 메모리에서 데이터를 가져오는 게 훨씬 느려서, 얼마나 붙어 있게, 얼마나 적게 읽나가 시간을 정해요.</p>"""
why += K.table(["빨라지는 이유", "이 쪽", "실무에서 보이는 곳"], [["인터프리터 대신 C 반복문", "①", "NumPy, pandas, PyTorch를 쓰는 이유"], ["한 명령에 여러 칸", "②③", "int8 양자화, float16 추론"],
                                                       ["붙어 있는 메모리, 캐시", "④⑤", "행렬 모양, 배치 크기, 벡터 검색 인덱스"], ["메모리 덜 오가기", "⑥⑦", "연산 융합, 갈림길 없는 코드"]])
why += "<p>터미널 프로그램 Ghostty는 입력 글자 중 특별한 문자를 찾는 반복문 하나를 SIMD로 바꿔 전체 처리량이 다섯 배쯤 늘었다고 해요. 이 쪽은 그 생각을 파이썬과 NumPy에서 재 봐요.</p>"
whole = """<p>세 가지 방법 → SIMD 다섯 단계 → 자료형 폭 → 메모리 배치 → 캐시 크기 → 임시 배열 → 갈림길 없애기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>
<p>코드는 <code>examples/speed/simd_numpy.py</code> 하나예요. 시간은 PC마다 달라서, 같은 PC 안에서의 <b>몇 배 차이</b>를 보세요. 아래 결과는 맨 아래 실행 환경의 PC에서 잰 거예요.</p>"""
steps = [
    K.step(1, "같은 일, 세 가지 방법", "20MB 글자에서 \"A\"가 몇 번 나오는지 파이썬 반복문, C 함수(bytes.count), NumPy로 세요.",
           "<p>시간은 <code>timeit.repeat</code>로 여러 번 재서 가장 빠른 값을 써요. 다른 프로그램 때문에 느려진 회차를 빼기 위해서예요. 파이썬 반복문은 너무 느려서 1MB만 재고 20을 곱했어요.</p>"
           + K.out("f_hf")
           + K.code_table([("bench(f)", "여러 번 재서 가장 빠른 값"), ("np.frombuffer(text, dtype=np.uint8)", "바이트를 복사 없이 NumPy 배열로 보기"), ("np.count_nonzero(codes == NEEDLE)", "모든 칸을 한꺼번에 비교하고 참의 수 세기")]), "s1"),
    K.step(2, "SIMD의 다섯 단계를 손으로", "찾을 값을 32칸에 복사하고, 데이터를 32바이트씩 잘라 칸마다 비교하고, 모아서 더하고, 남은 꼬리는 하나씩 세요.",
           "<p>SIMD 코드는 거의 항상 이 모양이에요. <b>① 상수를 모든 칸에 복사</b>(broadcast) → <b>② 레지스터 폭만큼씩 돌기</b> → <b>③ 칸마다 동시에 비교나 계산</b> → <b>④ 결과를 모으거나 저장</b>(reduce) → <b>⑤ 폭에 안 맞는 나머지는 하나씩</b>(scalar tail). 컴파일러가 단순한 반복문도 자동으로 이렇게 바꿔 주지 못할 때가 있어서, 아주 자주 도는 반복문은 직접 SIMD로 쓰기도 해요.</p>"
           "<p>CPU마다 레지스터 폭이 달라요. ARM의 NEON은 128비트(바이트 16칸), 인텔과 AMD의 AVX2는 256비트(32칸), AVX-512는 512비트(64칸)예요. 그래서 배포할 때는 \"이 CPU가 어떤 명령을 지원하나\"를 실행 중에 보고 고르는 코드가 필요해요. NumPy도 그렇게 해요.</p>"
           + K.code_table([("np.full(W, needle)", "1. 모든 칸에 같은 값"), ("data[:body].reshape(-1, W)", "2. 32칸씩 자르기"), ("blocks == lanes", "3. 칸마다 비교"), ("hits.sum()", "4. 모으기"), ("for b in data[body:]", "5. 꼬리")]), "s2"),
    K.step(3, "자료형이 좁으면 한 번에 더 많은 칸", "같은 2,000만 개 숫자를 float64부터 int8까지 바꿔 가며 최댓값과 합을 구해요.",
           "<p>256비트 레지스터에는 float64가 4칸, float32가 8칸, int8이 32칸 들어가요. 같은 명령 수로 더 많은 숫자를 처리하고, 메모리에서 읽을 바이트도 줄어요. 대신 표현할 수 있는 범위와 정밀도가 줄어요(<a href=\"../model/model-compression.html\">모델 가볍게</a>의 int8 양자화).</p>"
           + K.out("f_w")
           + K.code_table([("base.astype(dt)", "같은 숫자를 다른 자료형으로"), ("a.max()", "자료형 그대로 비교"), ("a.sum()", "작은 정수는 int64로 넓혀서 더하기")]), "s3"),
    K.step(4, "붙어 있는 메모리", "같은 수의 숫자를 붙어 있게 두고 더할 때와 4칸씩 건너뛰며 더할 때, 행렬을 그대로 복사할 때와 전치해서 복사할 때를 비교해요.",
           "<p>CPU는 메모리를 한 칸씩이 아니라 64바이트 덩어리(<b>캐시 줄</b>)로 가져와요. 붙어 있는 데이터는 한 번 가져온 덩어리를 다 쓰지만, 건너뛰면 대부분 버려요. NumPy의 <code>a[::4]</code>는 복사 없이 \"4칸씩 건너보기\"(stride)라서 쓰기는 편하지만 이렇게 느릴 수 있어요. 자주 쓸 거면 <code>np.ascontiguousarray</code>로 한 번 붙여 두는 게 나아요.</p>"
           + K.code_table([("big[::4]", "복사 없이 4칸마다 보기 (stride)"), ("np.ascontiguousarray(...)", "붙어 있는 새 배열로 복사"), ("M.T.copy()", "전치: 세로로 읽어 가로로 쓰기")]), "s4"),
    K.step(5, "캐시에 들어가는 크기인가", "배열 크기를 32KB부터 128MB까지 늘리며 원소 하나를 더하는 데 걸리는 시간을 재요.",
           "<p>CPU 안에는 작고 빠른 메모리(캐시)가 여러 층 있어요. 보통 L1은 수십 KB, L2는 1MB 안팎, L3는 수십 MB예요. 데이터가 캐시에 다 들어가면 빠르고, 넘치면 훨씬 느린 RAM에서 읽어야 해요. 배치 크기나 블록 크기를 정할 때 이 경계를 의식하는 이유예요.</p>"
           + K.out("f_c")
           + K.code_table([("number=max(1, 2**24 // n)", "작은 배열은 여러 번 돌려 재기"), ("/ n * 1e9", "원소 하나에 걸린 ns")]), "s5"),
    K.step(6, "임시 배열 줄이기", "같은 식을 연산마다 새 배열을 만들며 계산할 때와, 미리 만든 버퍼 하나에 이어서 계산할 때를 비교해요.",
           "<p>NumPy의 <code>(x * 2 + 1) * 3 - x</code>는 연산마다 결과 배열을 새로 만들어요. 큰 배열에서는 계산보다 이 배열들을 메모리에 쓰고 다시 읽는 시간이 더 커요. <code>out=</code>과 <code>+=</code> 같은 제자리 연산으로 버퍼를 다시 쓰면 줄어요. numexpr나 JIT 컴파일러(Numba, PyTorch의 컴파일 모드)는 식 전체를 한 번에 돌려 같은 효과를 내요.</p>"
           + K.code_table([("np.multiply(x, 2.0, out=buf)", "결과를 미리 만든 버퍼에"), ("b += 1.0", "새 배열 없이 제자리에서")]), "s6"),
    K.step(7, "if 대신 마스크", "음수를 0으로 바꾸는 일을 파이썬 if, np.where, np.maximum으로 해요.",
           "<p>칸마다 다른 길로 가는 코드(if)는 SIMD로 한꺼번에 하기 어려워요. 그래서 SIMD 코드는 <b>양쪽을 다 계산하고 마스크로 고르는</b> 방식으로 갈림길을 없애요. CPU의 분기 예측도 데이터가 무작위면 자주 틀려서 느려지는데, 마스크는 그 문제도 없어요.</p>"
           + K.code_table([("np.where(y > 0, y, 0.0)", "마스크로 고르기"), ("np.maximum(y, 0.0)", "\"둘 중 큰 것\" 명령 하나")]), "s7"),
]
practice = [("파이썬 코드가 느리다", "먼저 재요(프로파일러). 시간의 대부분을 쓰는 반복문 하나를 찾아 NumPy나 이미 있는 C 함수로 바꾸는 게 가장 큰 이득이에요. 직접 SIMD를 짜는 건 그다음이에요."),
            ("큰 배열을 다룬다", "자료형을 필요한 만큼만 넓게, 메모리는 붙어 있게, 임시 배열은 적게. 결과 자료형이 넓어지는지(sum의 int64)도 확인해요."),
            ("여러 하드웨어에 배포한다", "CPU의 SIMD 명령(NEON, AVX2, AVX-512)과 GPU, NPU 종류마다 빌드가 달라요. 파이썬 패키지도 하드웨어에 맞는 빌드를 골라 설치하는 방식(wheel variants)이 논의되고 있어요. 지원하지 않는 기계를 위한 느린 기본 경로를 꼭 남겨 둬요.")]
libs = [("①~⑦", "NumPy, <code>timeit</code>", "배열 연산, 시간 재기")]
K.build(ROOT / "topics" / "backend" / "simd-speed.html", title="SIMD와 메모리", h1="SIMD와 메모리: 같은 계산이 수십 배 빨라지는 이유", track="백엔드",
        crumb="백엔드 &gt; 성능과 하드웨어",
        one_line="파이썬 반복문을 NumPy로 바꾸면 빨라지는 건, C 반복문이 <b>SIMD</b>로 한 명령에 여러 칸을 처리하기 때문이에요. 자료형이 좁을수록 칸이 많아지고, 메모리가 <b>붙어 있고</b>, <b>캐시</b>에 들어가고, <b>임시 배열과 갈림길</b>이 적을수록 더 빨라져요.",
        why=why, whole=whole, steps=steps, code_file="examples/speed/simd_numpy.py", practice=practice, libs=libs,
        libs_note="<code>bench</code>, <code>count_simd_style</code>, <code>with_temporaries</code>, <code>in_place</code>, <code>relu_loop</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"../model/model-compression.html\">모델 가볍게</a>", "int8 양자화"), ("<a href=\"../model/llm-architectures.html\">요즘 LLM의 부품</a>", "KV 캐시와 메모리"), ("<a href=\"indexing.html\">인덱싱</a>", "벡터 검색의 속도")],
        sources=["다루는 범위: 부트캠프에서 따로 다루지 않은 주제예요",
                 "함께 정리한 자료: GeekNews에 소개된 Mitchell Hashimoto의 \"Everyone Should Know SIMD\"(Ghostty의 코드포인트 검색, SIMD 다섯 단계, NEON과 AVX2와 AVX-512, 자동 벡터화의 한계, 벤치마크와 스칼라 경로), PyCon Korea 2026 \"More Variants, More Diversity for AI Accelerators\"(가속기별 패키징과 wheel variants). 실험은 이 노트용으로 새로 짰어요",
                 "결과 값과 그림: <code>tools/pages/simd_speed.py</code>가 <code>examples/speed/simd_numpy.py</code>를 단계별로 실제 실행한 결과"])
print("ok")
