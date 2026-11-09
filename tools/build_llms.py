"""모든 Day 노트를 AI가 읽기 좋은 글 파일로 모읍니다.

    python tools/build_llms.py

만드는 파일 (SeSAC 폴더 바로 아래):
- all-notes.md : Day6~ 노트 본문 전체 + Day별 용어 (그림은 설명 글만, 체험·스크립트는 제외)
- llms.txt     : 이 사이트가 무엇인지 + 각 Day 페이지/파일 주소 목록 (AI용 안내판)

새 Day를 추가하거나 노트를 고친 뒤에 build_glossary.py 다음으로 실행하세요.
"""
import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_URL = "https://yiji-1015.github.io/sesac-notes/"
REPO_URL = "https://github.com/Yiji-1015/sesac-notes"


class NoteToMd(HTMLParser):
    SKIP_TAGS = {"script", "style", "svg", "canvas", "nav", "button", "select", "label"}
    VOID = {"img", "br", "input", "hr", "meta", "link", "source", "wbr", "col", "area"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.in_article = 0
        self.skip = 0
        self.in_pre = False
        self.table = None
        self.row = None
        self.cell = None
        self.buf = []
        self.list_depth = 0
        self.fig = None

    # ---- helpers
    def flush(self, prefix=""):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if text:
            self.out.append(prefix + text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "article":
            self.in_article += 1
            return
        if not self.in_article:
            return
        if self.skip:
            if tag not in self.VOID:
                self.skip += 1
            return
        if tag in self.SKIP_TAGS or "lab" in cls.split():
            if tag not in self.VOID:
                self.skip = 1
            return
        if tag in ("h2", "h3", "h4", "p", "li", "div") and not self.in_pre and self.cell is None:
            self.flush()
        if tag == "pre":
            self.in_pre = True
            self.buf = []
        elif tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "figure":
            self.fig = {"alt": "", "cap": ""}
        elif tag == "img" and self.fig is not None:
            self.fig["alt"] = a.get("alt", "")
            self.fig["src"] = a.get("src", "")
        elif tag == "figcaption":
            self.buf = []
        elif tag in ("ul", "ol"):
            self.list_depth += 1
        elif tag == "br":
            (self.cell if self.cell is not None else self.buf).append(" ")

    def handle_endtag(self, tag):
        if tag == "article":
            self.flush()
            self.in_article -= 1
            return
        if not self.in_article:
            return
        if self.skip:
            if tag not in self.VOID:
                self.skip -= 1
            return
        if tag in ("h2", "h3", "h4"):
            self.flush({"h2": "\n## ", "h3": "\n### ", "h4": "\n#### "}[tag])
        elif tag == "p" or tag == "div":
            if self.cell is None and self.fig is None:
                self.flush()
        elif tag == "li":
            self.flush("- ")
        elif tag == "pre":
            code = "".join(self.buf).strip("\n")
            self.buf = []
            self.in_pre = False
            self.out.append("```\n" + code + "\n```")
        elif tag in ("td", "th") and self.cell is not None:
            self.row.append(re.sub(r"\s+", " ", "".join(self.cell)).strip().replace("|", "\\|"))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            rows = [r for r in self.table if r]
            if rows:
                w = max(len(r) for r in rows)
                rows = [r + [""] * (w - len(r)) for r in rows]
                md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
                md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
                self.out.append("\n".join(md))
            self.table = None
        elif tag == "figcaption" and self.fig is not None:
            self.fig["cap"] = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            self.buf = []
        elif tag == "figure" and self.fig is not None:
            f = self.fig
            self.fig = None
            desc = f["cap"] or f["alt"]
            if desc:
                self.out.append(f"[그림: {f['alt']}] {f['cap']}".strip())
        elif tag in ("ul", "ol"):
            self.list_depth -= 1

    def handle_data(self, data):
        if not self.in_article or self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        else:
            self.buf.append(data)


def page_title(html):
    m = re.search(r"<h1>(.*?)</h1>", html, re.S)
    t = re.sub(r"<[^>]+>", "", m.group(1)) if m else ""
    m2 = re.search(r'class="meta">(.*?)<', html, re.S)
    return t.strip(), (m2.group(1).strip() if m2 else "")


def main():
    gj = (ROOT / "glossary.js").read_text(encoding="utf-8")
    G = json.loads(gj[gj.index("["): gj.rindex("]") + 1])

    pages = sorted((ROOT / "notes").glob("day*.html"))
    parts = [
        "# SeSAC Vision·멀티모달 AI 부트캠프 학습 노트 — 전체 본문",
        "",
        f"사이트: {SITE_URL}  ·  저장소: {REPO_URL}",
        "강의 PDF를 바탕으로 수강생이 다시 정리한 노트입니다(강의 원본 아님). "
        "각 Day = 3줄 요약 → 이론 ↔ 명령어 → 필요한 패키지 → 이론별 설명(🤔 왜?) + 명령어·인자 표.",
        "",
    ]
    index_lines = []
    for p in pages:
        html = p.read_text(encoding="utf-8")
        day = int(re.search(r"day(\d+)", p.stem).group(1))
        title, meta = page_title(html)
        conv = NoteToMd()
        conv.feed(html)
        body = "\n\n".join(x for x in conv.out if x.strip())
        body = re.sub(r"\n{3,}", "\n\n", body)
        terms = [e for e in G if day in e["days"]]
        tl = "\n".join(
            f"- **{e['term']}**" + (f" ({e['full']})" if e.get("full") and e["full"] != e["term"] else "")
            + f": {e['short']}" for e in terms
        )
        parts += [f"\n---\n\n# {title}", f"_{meta}_  ·  페이지: {SITE_URL}notes/{p.name}", "", body,
                  f"\n## 📖 Day{day} 용어 ({len(terms)}개)", tl]
        index_lines.append(f"- [{title}]({SITE_URL}notes/{p.name}): {meta}")

    (ROOT / "all-notes.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    llms = [
        "# SeSAC Vision·멀티모달 AI 부트캠프 학습 노트",
        "",
        "> 매일 강의 PDF를 바탕으로 쉽게 다시 정리한 부트캠프 학습 노트. 머신러닝(Day6~8), 딥러닝(Day9~12, 14), "
        "컴퓨터비전(Day13, 15, 16). 각 Day 페이지는 이론, 대응하는 PyTorch/scikit-learn/OpenCV/Ultralytics 명령어와 인자, "
        "필요한 패키지, '왜 그런지' 설명, 직접 만든 그림을 담고 있다.",
        "",
        "AI가 참고할 때는 아래 전체 본문 파일 하나를 읽는 것이 가장 쉽다.",
        "",
        "## 전체 본문 (글만)",
        f"- [all-notes.md]({SITE_URL}all-notes.md): 모든 Day 본문 + Day별 용어 설명을 한 파일로",
        f"- [용어 사전]({SITE_URL}glossary.html): 수업 용어 전체 (쉬운 설명·비유)",
        "",
        "## Day별 페이지",
        *index_lines,
        "",
        "## 참고",
        f"- 저장소(원본 HTML·그림 생성 스크립트): {REPO_URL}",
        "- 노트는 수강생 정리본이며 강의 원본 PDF는 포함하지 않는다.",
    ]
    (ROOT / "llms.txt").write_text("\n".join(llms) + "\n", encoding="utf-8")
    print(f"all-notes.md ({(ROOT / 'all-notes.md').stat().st_size // 1024} KB), llms.txt — {len(pages)} pages")


if __name__ == "__main__":
    main()
