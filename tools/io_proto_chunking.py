"""견본 페이지(notes/proto-chunking.html)에 들어갈 코드와 실행 결과를 만든다.

- examples/chunking/chunk_pipeline.py 를 읽어 줄마다 "몇 단계에서 추가된 줄인지"를 정한다.
  (줄 끝 "#@N" 이 있으면 그 단계, 없으면 바로 위 "# ① …" 머리 주석의 단계)
- 단계 k 까지의 줄만 모아 실제로 실행하고, 그 단계의 출력을 기록한다.
- "바꿔 보기" 실험도 실제로 실행한다 (임시 폴더에서, 원본은 건드리지 않음).
결과: notes/proto-chunking-data.js  (페이지가 <script>로 읽음, file:// 에서도 동작)
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/io_proto_chunking.py
"""
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import transformers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "chunking"
SRC = (EX / "chunk_pipeline.py").read_text(encoding="utf-8")
CIRCLED = "①②③④⑤⑥⑦⑧⑨"
TAG = re.compile(r"\s+#@(\d+)\s*$")

# ---- 1. 줄마다 단계 정하기 -------------------------------------------------
lines = SRC.splitlines()
first = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#"))
code, cur = [], 1
for ln in lines[first:]:
    head = re.match(r"^# ([①-⑨]) ", ln)
    if head:
        cur = CIRCLED.index(head.group(1)) + 1
    m = TAG.search(ln)
    step = int(m.group(1)) if m else cur
    code.append({"t": TAG.sub("", ln).rstrip(), "s": step})
while code and not code[-1]["t"]:
    code.pop()
NSTEP = max(c["s"] for c in code)


def run(k, workdir, patch=None):
    """단계 k 까지의 줄만 실행하고 namespace 를 돌려준다."""
    src = "\n".join(c["t"] for c in code if c["s"] <= k)
    if patch:
        for a, b in patch:
            assert a in src, a
            src = src.replace(a, b)
    ns = {"__file__": str(Path(workdir) / "chunk_pipeline.py"), "__name__": "__proto__"}
    exec(compile(src, "chunk_pipeline.py", "exec"), ns)
    return ns


def workdir(doc_patch=None):
    d = Path(tempfile.mkdtemp(prefix="proto_chunk_"))
    doc = (EX / "library_guide.md").read_text(encoding="utf-8")
    if doc_patch:
        for a, b in doc_patch:
            assert a in doc, a
            doc = doc.replace(a, b)
    (d / "library_guide.md").write_text(doc, encoding="utf-8")
    return d


W = workdir()
out = {"steps": {}, "exp": {}}

# ---- 2. 단계별 실제 출력 ----------------------------------------------------
ns = run(1, W)
t = ns["text"]
out["steps"][1] = {"len": len(t), "lines": len(t.splitlines()), "head": t.splitlines()[:4]}

ns = run(2, W)
out["steps"][2] = {"sections": ns["sections"],
                   "bodies": [t[s["start"]:s["end"]] for s in ns["sections"]]}
# 대조: 고정 길이 60자 창
out["exp"]["fixed60"] = [{"start": i, "end": min(i + 60, len(t)), "text": t[i:i + 60]} for i in range(0, 180, 60)]

ns = run(3, W)
out["steps"][3] = {"sections": [{"heading": s["heading"], "chars": s["end"] - s["start"], "tokens": s["tokens"]}
                                for s in ns["sections"]], "max": ns["MAX_TOKENS"]}
line0 = t.splitlines()[3]
ids = ns["tok"](line0, add_special_tokens=False)["input_ids"]
ids_sp = ns["tok"](line0)["input_ids"]
out["exp"]["tokline"] = {"line": line0, "chars": len(line0), "tokens": len(ids),
                         "pieces": ns["tok"].convert_ids_to_tokens(ids),
                         "tokens_special": len(ids_sp),
                         "special": [x for x in ns["tok"].convert_ids_to_tokens(ids_sp) if x not in ns["tok"].convert_ids_to_tokens(ids)]}

ns = run(4, W)
long_s = next(s for s in ns["sections"] if s["tokens"] > ns["MAX_TOKENS"])
out["steps"][4] = {"heading": long_s["heading"], "tokens": long_s["tokens"],
                   "pieces": [{"start": p["start"], "end": p["end"],
                               "tokens": ns["count"](t[p["start"]:p["end"]]),
                               "text": t[p["start"]:p["end"]]} for p in ns["split_long"](long_s)]}

ns = run(5, W)
out["steps"][5] = {"n": len(ns["chunks"]), "first": ns["chunks"][0],
                   "all": [{k: c[k] for k in ("chunk_id", "heading", "char_start", "char_end", "tokens")}
                           for c in ns["chunks"]]}

ns = run(6, W)
saved = json.loads((W / "chunks.json").read_text(encoding="utf-8"))
out["steps"][6] = {"n": len(saved), "bytes": (W / "chunks.json").stat().st_size,
                   "printed": f"{len(saved)}개 조각 저장 → chunks.json"}

# ---- 3. 바꿔 보기 실험 (모두 실제 실행) -------------------------------------
for mx in (40, 200):
    d = workdir()
    ns = run(6, d, patch=[("MAX_TOKENS = 64", f"MAX_TOKENS = {mx}")])
    out["exp"][f"max{mx}"] = [{"heading": c["heading"], "tokens": c["tokens"]} for c in ns["chunks"]]

d = workdir(doc_patch=[("대출 기간은 14일이며", "대출 기간은 21일이며")])
ns = run(5, d)
before = [c["chunk_id"] for c in out["steps"][5]["all"]]
after = [c["chunk_id"] for c in ns["chunks"]]
out["exp"]["edit"] = [{"heading": c["heading"], "before": b, "after": a}
                      for c, b, a in zip(ns["chunks"], before, after)]
# 글자 수가 달라지는 수정: 뒤쪽 조각의 위치가 밀린다
d = workdir(doc_patch=[("대출 기간은 14일이며", "대출 기간은 140일이며")])
ns = run(5, d)
out["exp"]["edit_len"] = [{"heading": c["heading"], "before": b["chunk_id"], "after": c["chunk_id"],
                           "start_before": b["char_start"], "start_after": c["char_start"]}
                          for c, b in zip(ns["chunks"], out["steps"][5]["all"])]

# 바이트 위치로 잘못 계산하면?
ns = run(5, W)
c0 = ns["chunks"][0]
bs = len(t[:c0["char_start"]].encode("utf-8"))
be = len(t[:c0["char_end"]].encode("utf-8"))
wrong = t[bs:be]
out["exp"]["bytes"] = {"char": [c0["char_start"], c0["char_end"]], "byte": [bs, be],
                       "wrong_head": wrong[:30], "same": wrong == c0["body"]}
try:
    assert t[bs:be] == c0["body"]
    out["exp"]["bytes"]["error"] = None
except AssertionError:
    out["exp"]["bytes"]["error"] = "AssertionError"

# ---- 4. "왜 나눠야 할까?" 에 쓸 값 -----------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from pypdf import PdfReader  # noqa: E402
import sklearn  # noqa: E402

ns = run(5, W)
tok = ns["tok"]
bodies = [c["body"] for c in ns["chunks"]]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 2)).fit([t] + bodies)
D = vec.transform([t] + bodies)
qs = ["대출 기간은 며칠인가요?", "자리를 오래 비우면 어떻게 되나요?", "책을 잃어버리면 어떻게 하나요?"]
why = {"doc": t, "doc_tokens": len(tok(t, add_special_tokens=False)["input_ids"]),
       "chunks": [{"heading": c["heading"], "chars": len(c["body"])} for c in ns["chunks"]],
       "sims": [{"q": q, "s": [round(float(x), 3) for x in cosine_similarity(vec.transform([q]), D)[0]]} for q in qs],
       "sklearn": sklearn.__version__}
pdf = ROOT / "강의자료-원본" / "Day15_20261106_객체탐지_YOLO.pdf"
if pdf.exists():
    r = PdfReader(str(pdf))
    pt = "".join((p.extract_text() or "") for p in r.pages)
    why["pdf"] = {"pages": len(r.pages), "chars": len(pt), "tokens": len(tok(pt, add_special_tokens=False)["input_ids"])}
out["why"] = why

out["code"] = code
out["nstep"] = NSTEP
out["meta"] = {"python": sys.version.split()[0], "transformers": transformers.__version__,
               "tokenizer": "bert-base-multilingual-cased"}

js = "window.PROTO = " + json.dumps(out, ensure_ascii=False, indent=1) + ";\n"
(ROOT / "notes" / "proto-chunking-data.js").write_text(js, encoding="utf-8")
shutil.rmtree(W, ignore_errors=True)
print("steps:", list(out["steps"]), "exp:", list(out["exp"]))
print(json.dumps({k: out["exp"][k] for k in ("max40", "max200", "edit", "bytes", "tokline")}, ensure_ascii=False, indent=1))
