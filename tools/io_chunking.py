"""주제 페이지 'Chunking'(topics/backend/chunking.html)의 코드와 실행 결과를 만든다.

examples/ingest/ingest.py 의 ⑧~⑬ 단계를 실제로 실행한다 (앞 단계 ①~⑦도 함께 실행됨).
결과: topics/backend/chunking-data.js
실행: HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/io_chunking.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stepcode  # noqa: E402

import sklearn  # noqa: E402
import transformers  # noqa: E402
from pypdf import PdfReader  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "ingest"
code = stepcode.parse((EX / "ingest.py").read_text(encoding="utf-8"))
RANGE = (8, 13)
out = {"steps": {}, "exp": {}}
W = stepcode.workdir(EX)
run = lambda k, wd=W, patch=None: stepcode.run(code, k, wd, patch)  # noqa: E731
pick = lambda d, *ks: {k: d[k] for k in ks}  # noqa: E731

# ---- 왜 ----------------------------------------------------------------------
ns = run(13)
units, chunks = ns["units"], ns["chunks"]
guide = next(u for u in units if u["file"] == "guide.md")["clean"]
gch = [c for c in chunks if c["file"] == "guide.md"]
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 2)).fit([guide] + [c["body"] for c in gch])
D = vec.transform([guide] + [c["body"] for c in gch])
qs = ["대출 기간은 며칠인가요?", "자리를 오래 비우면 어떻게 되나요?", "책을 잃어버리면 어떻게 하나요?"]
tok = ns["tok"]
why = {"guide": guide, "guide_tokens": ns["count"](guide),
       "chunks": [{"heading": c["heading"], "chars": len(c["body"])} for c in gch],
       "sims": [{"q": q, "s": [round(float(x), 3) for x in cosine_similarity(vec.transform([q]), D)[0]]} for q in qs]}
pdf = ROOT / "강의자료-원본" / "Day15_20261106_객체탐지_YOLO.pdf"
if pdf.exists():
    r = PdfReader(str(pdf))
    pt = "".join((p.extract_text() or "") for p in r.pages)
    why["pdf"] = {"pages": len(r.pages), "chars": len(pt), "tokens": ns["count"](pt)}
out["why"] = why
out["units"] = [{"file": u["file"], "where": u["where"], "chars": len(u["clean"])} for u in units]

# ---- ⑧ 제목으로 나누기 ---------------------------------------------------------
ns = run(8)
U = ns["units"]
out["steps"][8] = {"sections": [{"file": U[s["u"]]["file"], "where": U[s["u"]]["where"], "heading": s["heading"],
                                 "start": s["start"], "end": s["end"], "chars": s["end"] - s["start"],
                                 "head": U[s["u"]]["clean"][s["start"]:s["end"]].splitlines()[0]} for s in ns["sections"]]}
out["exp"]["fixed60"] = [{"start": i, "end": min(i + 60, len(guide)), "text": guide[i:i + 60]} for i in range(0, 180, 60)]
out["exp"]["heads"] = [{"file": u["file"], "where": u["where"], "found": [m.group(1) for m in ns["HEADING"].finditer(u["clean"])]}
                       for u in U]
r1 = next(u for u in U if u["file"] == "rules.pdf" and u["where"] == "1쪽")["clean"]
out["exp"]["rules1"] = r1

# ---- ⑨ 토큰 ----------------------------------------------------------------
ns = run(9)
U = ns["units"]
out["steps"][9] = {"max": ns["MAX_TOKENS"], "sections": [
    {"file": U[s["u"]]["file"], "where": U[s["u"]]["where"], "heading": s["heading"],
     "chars": s["end"] - s["start"], "tokens": s["tokens"]} for s in ns["sections"]]}
line0 = guide.splitlines()[3]
ids = tok(line0, add_special_tokens=False)["input_ids"]
ids_sp = tok(line0)["input_ids"]
out["exp"]["tokline"] = {"line": line0, "chars": len(line0), "tokens": len(ids),
                         "pieces": tok.convert_ids_to_tokens(ids), "tokens_special": len(ids_sp),
                         "special": tok.convert_ids_to_tokens([i for i in ids_sp if i in tok.all_special_ids])}

# ---- ⑩ 다시 묶기 -------------------------------------------------------------
ns = run(10)
U = ns["units"]
longs = []
for s in ns["sections"]:
    if s["tokens"] > ns["MAX_TOKENS"]:
        t = U[s["u"]]["clean"]
        longs.append({"file": U[s["u"]]["file"], "where": U[s["u"]]["where"], "heading": s["heading"], "tokens": s["tokens"],
                      "pieces": [{"start": p["start"], "end": p["end"], "tokens": ns["count"](t[p["start"]:p["end"]]),
                                  "text": t[p["start"]:p["end"]]} for p in ns["split_long"](s)]})
out["steps"][10] = {"longs": longs}
for mx in (40, 200):
    nsx = run(11, patch=[("MAX_TOKENS = 64", f"MAX_TOKENS = {mx}")])
    out["exp"][f"max{mx}"] = {"n": len(nsx["chunks"]),
                              "split": sorted({(c["file"], c["heading"] or "") for c in nsx["chunks"]
                                               if sum(1 for d in nsx["chunks"] if (d["file"], d["where"], d["heading"]) == (c["file"], c["where"], c["heading"])) > 1})}

# ---- ⑪ 기록 -----------------------------------------------------------------
ns = run(11)
out["steps"][11] = {"n": len(ns["chunks"]), "first": ns["chunks"][0],
                    "all": [pick(c, "chunk_id", "file", "where", "heading", "tokens") for c in ns["chunks"]]}
before = {(c["file"], c["where"], c["heading"], c["char_start"]): c["chunk_id"] for c in ns["chunks"]}


def edit_run(old, new):
    d = stepcode.workdir(EX)
    p = d / "raw" / "guide.md"
    p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    ns2 = run(11, d)
    shutil.rmtree(d, ignore_errors=True)
    rows = []
    for c, b in zip(ns2["chunks"], ns["chunks"]):
        if c["file"] != "guide.md":
            continue
        rows.append({"heading": c["heading"], "before": b["chunk_id"], "after": c["chunk_id"],
                     "start_before": b["char_start"], "start_after": c["char_start"]})
    return rows


out["exp"]["edit"] = edit_run("대출 기간은 14일이며", "대출 기간은 21일이며")
out["exp"]["edit_len"] = edit_run("대출 기간은 14일이며", "대출 기간은 140일이며")
out["exp"]["n_other_same"] = None

# ---- ⑫ 검사 -----------------------------------------------------------------
ns = run(12)
out["steps"][12] = {"kept": len(ns["chunks"]), "min": ns["MIN_CHARS"],
                    "dropped": [pick(c, "file", "where", "heading", "body", "reason") for c in ns["dropped"]]}
c0 = ns["chunks"][0]
t0 = ns["clean_of"][(c0["file"], c0["where"])]
bs, be = len(t0[:c0["char_start"]].encode("utf-8")), len(t0[:c0["char_end"]].encode("utf-8"))
out["exp"]["bytes"] = {"char": [c0["char_start"], c0["char_end"]], "byte": [bs, be], "wrong_head": t0[bs:be][:30],
                       "error": None if t0[bs:be] == c0["body"] else "AssertionError"}
ns0 = run(12, patch=[("MIN_CHARS = 20", "MIN_CHARS = 0")])
out["exp"]["min0"] = [pick(c, "file", "where", "heading", "body") for c in ns0["chunks"] if len(c["body"]) < 20]

# ---- ⑬ 저장 -----------------------------------------------------------------
ns = run(13)
saved = json.loads((W / "out" / "chunks.json").read_text(encoding="utf-8"))
out["steps"][13] = {"printed": f"글 {len(ns['units'])}개 → 조각 {len(saved)}개 저장 (뺀 조각 {len(ns['dropped'])}개)",
                    "chunks": [pick(c, "chunk_id", "file", "where", "heading", "tokens", "body") for c in saved],
                    "bytes": (W / "out" / "chunks.json").stat().st_size}

out["code"] = [c for c in code if c["s"] <= RANGE[1]]
out["names"] = stepcode.names(out["code"])
out["range"] = RANGE
out["meta"] = {"python": sys.version.split()[0], "transformers": transformers.__version__, "sklearn": sklearn.__version__,
               "tokenizer": ns["MODEL"]}
dst = ROOT / "topics" / "backend" / "chunking-data.js"
dst.write_text("window.PAGE = " + json.dumps(out, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
shutil.rmtree(W, ignore_errors=True)
print("ok:", dst.relative_to(ROOT))
print(json.dumps({"sections": [(s["file"], s["where"], s["heading"], s["chars"]) for s in out["steps"][8]["sections"]],
                  "tokens": [(s["file"], s["heading"], s["tokens"]) for s in out["steps"][9]["sections"]],
                  "edit": out["exp"]["edit"], "edit_len": out["exp"]["edit_len"], "min0": out["exp"]["min0"],
                  "bytes": out["exp"]["bytes"], "why_sims": why["sims"], "heads": out["exp"]["heads"]}, ensure_ascii=False))
