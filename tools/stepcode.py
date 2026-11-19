"""'자라는 코드' 페이지용 공통 도구.

예제 파일의 줄마다 "몇 단계에서 추가된 줄인지"를 정하고, 단계 k 까지만 실제로 실행한다.
  - 줄 끝 "#@N" 이 있으면 그 단계
  - 없으면 바로 위 "# ① …" 머리 주석의 단계
  - 파일 맨 위 설명 주석은 페이지에 보이지 않는다
"""
import re
import shutil
import tempfile
from pathlib import Path

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
TAG = re.compile(r"\s+#@(\d+)\s*$")
HEAD = re.compile(r"^# ([①-⑳]) (.+)$")


def parse(src: str) -> list[dict]:
    lines = src.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.strip() and not ln.startswith("#"))
    code, cur = [], 1
    for ln in lines[first:]:
        h = HEAD.match(ln)
        if h:
            cur = CIRCLED.index(h.group(1)) + 1
        m = TAG.search(ln)
        code.append({"t": TAG.sub("", ln).rstrip(), "s": int(m.group(1)) if m else cur})
    while code and not code[-1]["t"]:
        code.pop()
    # import 줄 정렬용 공백 줄이기 (화면 폭)
    for c in code:
        c["t"] = re.sub(r"(\S) {3,}(# )", r"\1  \2", c["t"])
    return code


def names(code: list[dict]) -> dict[int, str]:
    out = {}
    for c in code:
        h = HEAD.match(c["t"])
        if h:
            out[CIRCLED.index(h.group(1)) + 1] = h.group(2)
    return out


def workdir(src_dir: Path, setup=None) -> Path:
    """예제 폴더를 임시 폴더로 복사한다 (원본은 건드리지 않음). out/, __pycache__ 는 빼고."""
    d = Path(tempfile.mkdtemp(prefix="stepcode_"))
    shutil.copytree(src_dir, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns("out", "__pycache__", "*.pyc"))
    if setup:
        setup(d)
    return d


def box(title, lines, notes=None):
    """출력 상자. lines: 문자열 또는 (문자열, 번호) 또는 (문자열, 번호, "bad"|"newf")."""
    return {"title": title, "lines": [l if isinstance(l, str) else list(l) for l in lines], "notes": notes or []}


def table(title, cols, rows, notes=None):
    """표 상자. 칸 안의 글은 HTML 로 그대로 들어가니, 필요하면 html.escape 해서 넘긴다."""
    return {"title": title, "cols": cols, "rows": rows, "notes": notes or []}


def images(title, items, notes=None, lines=None):
    """그림 상자. items: [(그림 경로(페이지 기준), 설명), …]"""
    return {"title": title, "images": [list(i) for i in items], "notes": notes or [], **({"lines": lines} if lines else {})}


def save(dst: Path, page: dict):
    import json
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("window.PAGE = " + json.dumps(page, ensure_ascii=False, indent=1, default=str) + ";\n", encoding="utf-8")


def run(code: list[dict], k: int, wd: Path, patch=None, filename="ingest.py", env=None) -> dict:
    """단계 k 까지의 줄만 실행하고 namespace 를 돌려준다."""
    src = "\n".join(c["t"] for c in code if c["s"] <= k)
    for a, b in (patch or []):
        assert a in src, f"patch target not found: {a}"
        src = src.replace(a, b)
    ns = {"__file__": str(wd / filename), "__name__": "__stepcode__", **(env or {})}
    exec(compile(src, filename, "exec"), ns)
    return ns
