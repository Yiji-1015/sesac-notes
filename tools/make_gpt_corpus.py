"""작은 GPT 쪽의 말뭉치를 만든다 → examples/gpt/data/notes_corpus.jsonl
이 저장소의 all-notes.md(옛 Day 노트 전체 본문)에서 사람이 쓴 문장만 남긴다.
그리고 요약 단계용으로 주제 페이지마다 '한 문장'(사람이 쓴 요약)과 본문 문장을 → examples/gpt/data/pages.jsonl
- 코드 블록, 표, 주소, 마크다운 기호, 이모지를 빼고, 가운뎃점 "·"은 노트 규칙대로 ", "로 바꾼다.
- 한 줄 = 한 문단. Day마다 {"day": 6, "text": "..."} 한 줄.
실행: python tools/make_gpt_corpus.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "all-notes.md"
OUT = ROOT / "examples" / "gpt" / "data" / "notes_corpus.jsonl"

md = SRC.read_text(encoding="utf-8")
md = re.sub(r"```.*?```", "\n", md, flags=re.S)                  # 코드 블록
days = re.split(r"\n# Day(\d+)", md)                             # [머리말, "6", 본문, "7", 본문, …]
rows = []
for d, body in zip(days[1::2], days[2::2]):
    paras = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(("|", "_", "---", "📎", "🔧")):  # 표, 날짜 줄, 구분선, 링크 모음
            continue
        line = re.sub(r"https?://\S+", "", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)                 # 인라인 코드는 글자만 남기기
        line = re.sub(r"^[#>\-*\d.\s]+", "", line)               # 머리 기호, 글머리표, 번호
        line = line.replace("**", "").replace("__", "")
        line = line.replace(" · ", ", ").replace("·", ", ")
        line = re.sub(r"[^\w\s가-힣.,!?()'\"%:;/+=<>~\-→]", "", line)   # 이모지와 드문 기호
        line = re.sub(r"\s+", " ", line).strip()
        if re.search(r"\d{6,}|_\d{4}", line):                  # 파일 이름, 날짜 묶음 같은 기록 줄
            continue
        if len(re.findall(r"[가-힣]", line)) >= 10:
            paras.append(line)
    rows.append({"day": int(d), "text": "\n".join(paras)})
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(OUT, len(rows), "days", sum(len(r["text"]) for r in rows), "chars")

# ---------------- 요약 단계용: 주제 페이지의 '한 문장'과 본문 문장
import html as H  # noqa: E402

PAGES = ROOT / "examples" / "gpt" / "data" / "pages.jsonl"
out = []
for f in sorted((ROOT / "topics").glob("*/*.html")):
    t = f.read_text(encoding="utf-8")
    one = re.search(r'<p class="one-line">(.*?)</p>', t, flags=re.S)
    h1 = re.search(r"<h1>(.*?)</h1>", t, flags=re.S)
    if not one or not h1:
        continue
    body = t[t.index(one.group(0)) + len(one.group(0)):]
    body = re.sub(r"<script.*?</script>|<table.*?</table>|<pre.*?</pre>", " ", body, flags=re.S)
    sents = []
    for para in re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.S):
        txt = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", para))).strip()
        for sent in re.split(r"(?<=[.?!])\s+", txt):
            if len(re.findall(r"[가-힣]", sent)) >= 15:
                sents.append(sent)
    clean = lambda x: re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", "", x))).strip()  # noqa: E731
    out.append({"page": f.stem, "title": clean(h1.group(1)), "summary": clean(one.group(1)), "sentences": sents})
with PAGES.open("w", encoding="utf-8") as fw:
    for r in out:
        fw.write(json.dumps(r, ensure_ascii=False) + "\n")
print(PAGES, len(out), "pages", sum(len(r["sentences"]) for r in out), "sentences")
