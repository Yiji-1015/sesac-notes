"""주제 페이지와 옛 Day 노트를 AI가 읽기 좋은 글 파일로 모읍니다.

    python tools/build_llms.py

만드는 파일 (SeSAC 폴더 바로 아래):
- all-notes.md : 1부 주제 페이지 본문(index.html의 순서대로, 실행 결과 표와 그림 설명 포함)
                 2부 옛 Day 노트 본문과 Day별 용어 (그림은 설명 글만, 체험과 스크립트는 제외)
- llms.txt     : 이 사이트가 무엇인지 + 주제 페이지와 Day 노트 주소 목록 (AI용 안내판)

주제 페이지를 새로 만들거나 고친 뒤, build_glossary.py 다음으로 실행하세요.
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_URL = "https://yiji-1015.github.io/sesac-notes/"
REPO_URL = "https://github.com/Yiji-1015/sesac-notes"


def plain(x):
    """HTML 태그와 자주 쓰는 문자 참조를 지운 한 줄 글."""
    t = re.sub(r"<[^>]+>", "", str(x))
    for a, b in (("&gt;", ">"), ("&lt;", "<"), ("&quot;", '"'), ("&nbsp;", " "), ("&amp;", "&")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def cell(x):
    return plain(x).replace("|", "\\|")


def render_out(o):
    """주제 페이지의 실행 결과(S.box, S.table, S.images)를 마크다운으로."""
    md = [f"**▶ {plain(o.get('title', ''))}**"]
    if "rows" in o:
        cols = [cell(c) for c in o["cols"]]
        rows = ["| " + " | ".join(cell(c) for c in r) + " |" for r in o["rows"]]
        md.append("\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols), *rows]))
    if "lines" in o:
        md.append("```\n" + "\n".join(plain(line if isinstance(line, str) else line[0]) for line in o["lines"]) + "\n```")
    for img in o.get("images", []):
        md.append(f"[그림: {img[0]}] {plain(img[1])}".strip())
    md += [plain(n) for n in o.get("notes", [])]
    return "\n\n".join(md)


class NoteToMd(HTMLParser):
    SKIP_TAGS = {"script", "style", "svg", "canvas", "nav", "button", "select", "label"}
    VOID = {"img", "br", "input", "hr", "meta", "link", "source", "wbr", "col", "area"}

    def __init__(self, outs=None):
        super().__init__(convert_charrefs=True)
        self.outs = outs or {}                                   # 주제 페이지: data-out 자리에 넣을 실행 결과
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
        if a.get("data-out") in self.outs:
            self.out.append(render_out(self.outs[a["data-out"]]))
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
    return t.strip(), plain(m2.group(1)) if m2 else ""


def to_markdown(html, outs=None):
    conv = NoteToMd(outs)
    conv.feed(html)
    body = re.sub(r"\n{3,}", "\n\n", "\n\n".join(x for x in conv.out if x.strip()))
    return re.sub(r"^(#{2,3}) ", r"#\1 ", body, flags=re.M)      # 페이지 제목이 ##라서 본문 제목을 한 단계씩 내리기


def topic_order():
    """index.html에서 (트랙, 카테고리, 주제 페이지 경로, 카드 제목)을 나오는 순서대로."""
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    html = html[: html.find('id="log"')]                         # 업데이트 기록은 빼기
    out, seen, track, cat = [], set(), "", ""
    for m in re.finditer(r'<h2>(.*?)</h2>|<h3>(.*?)</h3>|<a class="item" href="(topics/[^"]+\.html)"><b>(.*?)</b>', html, re.S):
        if m.group(1):
            track = plain(m.group(1))
        elif m.group(2):
            cat = plain(m.group(2))
        elif m.group(3) not in seen:
            seen.add(m.group(3))
            out.append((track, cat, m.group(3), plain(m.group(4))))
    return out


def load_page_data(html_path):
    f = html_path.with_name(html_path.stem + "-data.js")
    if not f.exists():
        return {}
    t = f.read_text(encoding="utf-8")
    return json.loads(t[t.index("{"): t.rindex("}") + 1])


def main():
    gj = (ROOT / "glossary.js").read_text(encoding="utf-8")
    G = json.loads(gj[gj.index("["): gj.rindex("]") + 1])
    topics = topic_order()
    pages = sorted((ROOT / "notes").glob("day*.html"))

    parts = [
        "# SeSAC 부트캠프 학습 노트: 전체 본문",
        "",
        f"사이트: {SITE_URL}  /  저장소: {REPO_URL}",
        "수강생이 강의와 책, 세미나 자료를 바탕으로 다시 정리한 노트입니다(강의 원본 아님). "
        "1부는 주제 페이지예요. 페이지마다 예제 코드 파일 하나가 ①, ② 같은 단계로 자라고, "
        "▶ 표시가 붙은 표와 그림 설명의 값은 모두 그 코드를 실제로 실행한 결과예요. "
        "2부는 처음에 쓴 Day별 노트와 Day별 용어예요.",
        "",
        f"\n---\n\n# 1부: 주제 페이지 ({len(topics)}쪽)",
    ]
    topic_lines, cur = [], None
    for track, cat, rel, name in topics:
        path = ROOT / rel
        html = path.read_text(encoding="utf-8")
        data = load_page_data(path)
        title, meta = page_title(html)
        code_file = re.search(r"(examples/[\w/.-]+\.py)", html)
        info = f"_{meta}_  /  페이지: {SITE_URL}{rel}"
        if code_file:
            info += f"  /  예제 코드: {REPO_URL}/blob/main/{code_file.group(1)}"
        if data.get("env"):
            info += f"  /  실행 환경: {data['env']}"
        parts += [f"\n---\n\n## {title}", info, "", to_markdown(html, data.get("outs"))]
        if (track, cat) != cur:
            topic_lines.append(f"\n### {track} > {cat}")
            cur = (track, cat)
        topic_lines.append(f"- [{name}]({SITE_URL}{rel}): {title}")

    parts.append("\n---\n\n# 2부: Day별 노트")
    index_lines = []
    for p in pages:
        html = p.read_text(encoding="utf-8")
        day = int(re.search(r"day(\d+)", p.stem).group(1))
        title, meta = page_title(html)
        body = to_markdown(html)
        frag = ROOT / "cmds" / p.name
        cmd_md = to_markdown("<article>" + frag.read_text(encoding="utf-8") + "</article>") if frag.exists() else ""
        cmd_part = [f"\n### 🔧 Day{day} 명령어 (명령어 사전: {SITE_URL}commands.html#day{day})", cmd_md] if cmd_md else []
        terms = [e for e in G if day in e["days"]]
        tl = "\n".join(
            f"- **{e['term']}**" + (f" ({e['full']})" if e.get("full") and e["full"] != e["term"] else "")
            + f": {e['short']}" for e in terms
        )
        parts += [f"\n---\n\n## {title}", f"_{meta}_  /  페이지: {SITE_URL}notes/{p.name}", "", body, *cmd_part,
                  f"\n### 📖 Day{day} 용어 ({len(terms)}개)", tl]
        index_lines.append(f"- [{title}]({SITE_URL}notes/{p.name}): {meta}")

    (ROOT / "all-notes.md").write_text("\n".join(parts) + "\n", encoding="utf-8")

    llms = [
        "# SeSAC 부트캠프 학습 노트",
        "",
        "> 부트캠프 강의와 책, 세미나 자료를 주제별로 다시 정리한 학습 노트. 모델 쪽(데이터 분석, 머신러닝, 딥러닝, 이미지, "
        "언어 모델, 생성 모델, 강화학습, 추천과 평생 학습)과 백엔드 쪽(API, 데이터베이스, LLM 활용, 검색과 RAG, 성능)으로 나뉜다. "
        "주제 페이지마다 예제 코드 하나가 단계별로 자라고, 본문의 숫자와 그림은 그 코드를 실제로 실행한 결과다.",
        "",
        "AI가 참고할 때는 아래 전체 본문 파일 하나를 읽는 것이 가장 쉽다.",
        "",
        "## 전체 본문 (글만)",
        f"- [all-notes.md]({SITE_URL}all-notes.md): 주제 페이지(실행 결과 포함)와 Day 노트를 한 파일로",
        f"- [용어 사전]({SITE_URL}glossary.html): 용어 전체 (쉬운 설명)",
        f"- [명령어 사전]({SITE_URL}commands.html): Day 노트 시절의 도구별 코드, 인자, 실제 실행 입출력",
        "",
        "## 주제 페이지",
        *topic_lines,
        "",
        "## Day별 노트 (처음에 쓴 것)",
        *index_lines,
        "",
        "## 참고",
        f"- 저장소(원본 HTML, 예제 코드, 페이지를 만드는 스크립트): {REPO_URL}",
        "- 노트는 수강생 정리본이며 강의 원본 PDF는 포함하지 않는다.",
    ]
    (ROOT / "llms.txt").write_text("\n".join(llms) + "\n", encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"all-notes.md ({(ROOT / 'all-notes.md').stat().st_size // 1024} KB), llms.txt: 주제 {len(topics)}쪽, Day {len(pages)}쪽")


if __name__ == "__main__":
    main()
