"""주제 페이지 '문서 읽기'(topics/backend/document-reading.html)의 코드와 실행 결과를 만든다.

- examples/ingest/ingest.py 를 단계별로 실제 실행한다 (tools/stepcode.py).
- 원본 raw/ 는 임시 폴더로 복사해서 쓰고, 원본은 건드리지 않는다.
- "왜", "바꿔 보기" 대조 실험도 모두 실제로 실행한다.
결과: topics/backend/document-reading-data.js
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/io_ingest.py
"""
import json
import shutil
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stepcode  # noqa: E402

import bs4  # noqa: E402
import pypdf  # noqa: E402
import pytesseract  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "ingest"
code = stepcode.parse((EX / "ingest.py").read_text(encoding="utf-8"))
PAGE_STEPS = (1, 7)
code_page = [c for c in code if c["s"] <= PAGE_STEPS[1]]
out = {"steps": {}, "exp": {}}
W = stepcode.workdir(EX)
run = lambda k, wd=W, patch=None: stepcode.run(code, k, wd, patch)  # noqa: E731


def unit_view(u, n=None):
    v = {"file": u["file"], "where": u["where"], "raw": u["raw"]}
    if "clean" in u:
        v["clean"] = u["clean"]
    return v


# ---- 왜: 모든 파일을 그냥 글자로 읽어 보면 ---------------------------------
why = []
for p in sorted((EX / "raw").iterdir()):
    try:
        t = p.read_text(encoding="utf-8")
        why.append({"file": p.name, "ok": True, "chars": len(t), "head": t[:46]})
    except UnicodeDecodeError as e:
        why.append({"file": p.name, "ok": False, "error": f"UnicodeDecodeError: {e.reason} (byte {e.start})"})
out["why"] = why

# ---- ① 파일 목록과 원본 기록 -------------------------------------------------
ns = run(1)
out["steps"][1] = {"manifest": ns["manifest"]}
d = stepcode.workdir(EX, setup=lambda w: shutil.copyfile(w / "raw" / "guide.md", w / "raw" / "guide_copy.md"))
ns = run(1, d)
out["exp"]["dup"] = [m for m in ns["manifest"] if m["file"].startswith("guide")]

# ---- ② 글자 파일 읽기 ---------------------------------------------------------
ns = run(2)
out["steps"][2] = {"units": [unit_view(u) for u in ns["units"]],
                   "status": {m["file"]: m.get("status") for m in ns["manifest"]}}
ns = run(2, patch=[('ENCODING = {"old_notice.txt": "cp949"}', "ENCODING = {}")])
m = next(m for m in ns["manifest"] if m["file"] == "old_notice.txt")
out["exp"]["noenc"] = {"status": m["status"], "error": m["error"]}
ns = run(2, patch=[('ENCODING = {"old_notice.txt": "cp949"}', "ENCODING = {}"),
                   ('data.decode(ENCODING.get(m["file"], "utf-8"))', 'data.decode("utf-8", errors="ignore")')])
u = next(u for u in ns["units"] if u["file"] == "old_notice.txt")
out["exp"]["ignore"] = {"raw": u["raw"], "chars": len(u["raw"]),
                        "right": len((EX / "raw" / "old_notice.txt").read_bytes().decode("cp949"))}

# ---- ③ 웹페이지 읽기 ---------------------------------------------------------
ns = run(3)
out["steps"][3] = {"unit": unit_view(next(u for u in ns["units"] if u["file"] == "hours.html"))}
html = (EX / "raw" / "hours.html").read_text(encoding="utf-8")
out["exp"]["html_naive"] = BeautifulSoup(html, "html.parser").get_text()
out["exp"]["html_src"] = html

# ---- ④ PDF 읽기 --------------------------------------------------------------
ns = run(4)
out["steps"][4] = {"units": [unit_view(u) for u in ns["units"] if u["file"] == "rules.pdf"],
                   "status": {m["file"]: m.get("status") for m in ns["manifest"] if m["type"] == ".pdf"}}
out["exp"]["scan_text"] = [p.extract_text() for p in pypdf.PdfReader(EX / "raw" / "scan.pdf").pages]

# ---- ⑤ OCR -------------------------------------------------------------------
ns = run(5)
out["steps"][5] = {"unit": unit_view(next(u for u in ns["units"] if u["file"] == "scan.pdf"))}
ns = run(5, patch=[(" -c preserve_interword_spaces=1", "")])
out["exp"]["ocr_nopis"] = next(u for u in ns["units"] if u["file"] == "scan.pdf")["raw"]
out["meta_tesseract"] = str(pytesseract.get_tesseract_version())

# ---- ⑥ Cleaning --------------------------------------------------------------
ns = run(6)
units = ns["units"]
out["steps"][6] = {"units": [{"file": u["file"], "where": u["where"], "raw_chars": len(u["raw"]),
                              "clean_chars": len(u["clean"])} for u in units]}
nt = next(u for u in units if u["file"] == "notice.txt")
out["exp"]["nfd"] = {"line_raw": nt["raw"].splitlines()[3], "line_clean": nt["clean"].splitlines()[3],
                     "len_raw": len(nt["raw"].splitlines()[3]), "len_clean": len(nt["clean"].splitlines()[3]),
                     "find_raw": "도서관" in nt["raw"], "find_clean": "도서관" in nt["clean"],
                     "jamo": [f"U+{ord(ch):04X}" for ch in nt["raw"].splitlines()[3][:3]]}
r1 = next(u for u in units if u["file"] == "rules.pdf" and u["where"] == "1쪽")
hu = next(u for u in units if u["file"] == "hours.html")
out["exp"]["clean_cases"] = {"rules_raw": r1["raw"], "rules_clean": r1["clean"],
                             "html_raw": hu["raw"], "html_clean": hu["clean"]}
ns = run(6, patch=[('PAGE_NO = re.compile(r"^\\s*\\d+\\s*/\\s*\\d+\\s*$")', 'PAGE_NO = re.compile(r"\\d+\\s*/\\s*\\d+")'),
                   ("if not PAGE_NO.match(ln)", "if not PAGE_NO.search(ln)")])
r2 = next(u for u in ns["units"] if u["file"] == "rules.pdf" and u["where"] == "2쪽")
r2ok = next(u for u in units if u["file"] == "rules.pdf" and u["where"] == "2쪽")
out["exp"]["loose"] = {"bad": r2["clean"], "good": r2ok["clean"]}

# ---- ⑦ 저장 ------------------------------------------------------------------
ns = run(7)
saved_m = json.loads((W / "out" / "manifest.json").read_text(encoding="utf-8"))
saved_u = json.loads((W / "out" / "units.json").read_text(encoding="utf-8"))
out["steps"][7] = {"manifest": saved_m, "n_units": len(saved_u),
                   "printed": f"파일 {len(saved_m)}개 → 글 {len(saved_u)}개 저장",
                   "units": [{"file": u["file"], "where": u["where"], "clean": u["clean"]} for u in saved_u],
                   "sample": next(u for u in saved_u if u["file"] == "rules.pdf" and u["where"] == "2쪽")}

out["code"] = code_page
out["names"] = stepcode.names(code_page)
out["range"] = PAGE_STEPS
out["meta"] = {"python": sys.version.split()[0], "beautifulsoup4": bs4.__version__, "pypdf": pypdf.__version__,
               "pytesseract": pytesseract.__version__ if hasattr(pytesseract, "__version__") else "?"}

dst = ROOT / "topics" / "backend" / "document-reading-data.js"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("window.PAGE = " + json.dumps(out, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
shutil.rmtree(W, ignore_errors=True)
print("ok:", dst.relative_to(ROOT))
print(json.dumps({k: out["exp"][k] for k in ("dup", "noenc", "ignore", "nfd", "loose")}, ensure_ascii=False, indent=1))
print(repr(out["exp"]["ocr_nopis"][:80]))
