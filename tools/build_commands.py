"""cmds/dayNN.html 조각들을 모아 commands.html(명령어 사전)을 만든다.

조각 형식 (split_commands.py 가 만들어 줌):
<section class="cmd" id="d13-cv2-imread" data-topic="OpenCV" data-sub="읽기 · 저장 · 표시"
         data-day="13" data-theory="notes/day13.html#array">
  <h3>제목</h3> ...코드·표·박스...
</section>

주제 순서는 TOPIC_ORDER, 같은 주제 안에서는 Day → 원래 순서.
실행: python tools/build_commands.py
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

TOPIC_ORDER = ["환경 · 설치", "Python 기본", "NumPy", "pandas", "시각화", "scikit-learn", "XGBoost", "PyTorch",
               "torchvision", "OpenCV", "YOLO", "OCR · 문서", "VLM · Transformers", "배포 · 최적화", "FastAPI"]
TOPIC_EMOJI = {"OpenCV": "🖼️", "PyTorch": "🔥", "YOLO": "🎯", "FastAPI": "⚡", "pandas": "🐼", "NumPy": "🔢",
               "scikit-learn": "🤖", "XGBoost": "🌲", "시각화": "📊", "OCR · 문서": "📄", "VLM · Transformers": "🧠",
               "배포 · 최적화": "🚀", "환경 · 설치": "🧰", "Python 기본": "🐍", "torchvision": "🧩"}

SEC = re.compile(r'<section class="cmd"(?P<attrs>[^>]*)>\s*<h3>(?P<title>.*?)</h3>(?P<body>.*?)</section>', re.S)
ATTR = re.compile(r'([\w-]+)="([^"]*)"')


def load():
    items = []
    for f in sorted((ROOT / "cmds").glob("day*.html")):
        for n, m in enumerate(SEC.finditer(f.read_text(encoding="utf-8"))):
            a = dict(ATTR.findall(m.group("attrs")))
            items.append({**a, "day": int(a["data-day"]), "title": m.group("title").strip(),
                          "body": m.group("body").strip("\n"), "order": n})
    return items


OV = re.compile(r'<section class="cmd-overview" data-day="(\d+)">\s*(<table>.*?</table>)\s*</section>', re.S)
ROW = re.compile(r'<tr><td><a href="(notes/day\d+\.html#([^"]+))">(.*?)</a></td><td>(.*?)</td></tr>', re.S)


def overviews(items):
    """Day별 '이론 ↔ 명령어' 표. 명령어 칸을 그 구역의 첫 사전 항목으로 연결한다."""
    out = []
    for f in sorted((ROOT / "cmds").glob("day*.html")):
        m = OV.search(f.read_text(encoding="utf-8"))
        if not m:
            continue
        day = int(m.group(1))

        def row(r, day=day):
            href, sec, theory, cmds = r.groups()
            hit = [i for i in items if i["day"] == day and i["data-theory"].endswith("#" + sec)]
            go = f' <a href="#{hit[0]["id"]}" class="ov-go">🔧 {len(hit)}개 보기</a>' if hit else ''
            return f'<tr><td><a href="{href}">📖 {theory}</a></td><td>{cmds}{go}</td></tr>'
        table = ROW.sub(row, m.group(2)).replace("<th>대응 명령어</th>", "<th>대응 명령어 (🔧 = 아래 사전 항목으로)</th>")
        out.append(f'<section class="cmd-overview" data-day="{day}" hidden>\n'
                   f'<h2>📋 Day{day} 이론 ↔ 명령어 한눈에</h2>\n{table}\n</section>')
    return out


def main():
    items = load()
    ovs = overviews(items)
    topics = sorted({i["data-topic"] for i in items},
                    key=lambda t: TOPIC_ORDER.index(t) if t in TOPIC_ORDER else 99)
    days = sorted({i["day"] for i in items})
    toc, body = [], []
    for t in topics:
        its = sorted([i for i in items if i["data-topic"] == t], key=lambda i: (i["day"], i["order"]))
        tid = "topic-" + re.sub(r"[^0-9A-Za-z가-힣]+", "-", t).strip("-").lower()
        toc.append(f'<a class="filter-btn" href="#{tid}">{TOPIC_EMOJI.get(t, "🔧")} {t} <small>{len(its)}</small></a>')
        body.append(f'<section class="cmd-topic" id="{tid}">\n<h2>{TOPIC_EMOJI.get(t, "🔧")} {t}</h2>')
        last_sub = None
        for i in its:
            sub = i.get("data-sub", "")
            if sub and sub != last_sub:
                body.append(f'<h4 class="cmd-sub">{sub}</h4>')
                last_sub = sub
            body.append(
                f'<article class="cmd" id="{i["id"]}" data-day="{i["day"]}">\n'
                f'  <div class="cmd-head"><h3>{i["title"]}</h3>'
                f'<span class="cmd-meta"><span class="tag">Day{i["day"]}</span> '
                f'<a href="{i["data-theory"]}">📖 이론 보기</a></span></div>\n'
                f'{i["body"]}\n</article>')
        body.append('</section>')
    day_btns = ''.join(f'<button class="filter-btn" data-day="{d}">Day{d}</button>' for d in days)
    page = f'''<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>명령어 사전 | 부트캠프 학습 노트</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <!-- 자동 생성 파일: 고칠 땐 cmds/dayNN.html 을 고치고 python tools/build_commands.py -->
  <main class="container narrow">
    <a class="back" href="index.html">← 목록으로</a>
    <header class="site-header">
      <h1>🔧 명령어 사전</h1>
      <p>수업에 나온 코드·인자를 도구별로 — 이론은 각 항목의 📖 링크에서</p>
    </header>

    <div class="toolbar">
      <input id="q" type="search" placeholder="명령어 검색 (예: resize, threshold, VideoWriter)…">
    </div>
    <div class="toolbar cmd-days"><button class="filter-btn active" data-day="all">전체 Day</button>{day_btns}</div>
    <nav class="toolbar cmd-toc">{''.join(toc)}</nav>
    <p id="count" style="color:var(--muted);font-size:0.9rem"></p>

    <article class="note cmd-list">
{chr(10).join(ovs)}
{chr(10).join(body)}
    </article>
    <p class="empty" id="empty">검색 결과가 없어요.</p>
  </main>
  <script src="glossary.js"></script>
  <script src="note.js"></script>
  <script>
  (() => {{
    const q = document.getElementById('q'), cmds = [...document.querySelectorAll('article.cmd')];
    let day = 'all';
    function apply() {{
      const k = q.value.trim().toLowerCase();
      let n = 0;
      cmds.forEach(c => {{
        const ok = (day === 'all' || c.dataset.day === day) && (!k || c.textContent.toLowerCase().includes(k));
        c.hidden = !ok; if (ok) n++;
      }});
      document.querySelectorAll('.cmd-overview').forEach(s => {{ s.hidden = day === 'all' || s.dataset.day !== day || !!k; }});
      document.querySelectorAll('.cmd-topic').forEach(s => {{ s.hidden = !s.querySelector('article.cmd:not([hidden])'); }});
      document.querySelectorAll('h4.cmd-sub').forEach(h => {{
        let el = h.nextElementSibling, any = false;
        while (el && el.matches('article.cmd')) {{ if (!el.hidden) any = true; el = el.nextElementSibling; }}
        h.hidden = !any;
      }});
      document.getElementById('count').textContent = `${{n}}개 명령어` + (day !== 'all' ? ` · Day${{day}}` : '') + (k ? ` · 검색 "${{q.value}}"` : '');
      document.getElementById('empty').style.display = n ? 'none' : 'block';
    }}
    document.querySelectorAll('.cmd-days .filter-btn').forEach(b => b.addEventListener('click', () => {{
      day = b.dataset.day;
      document.querySelectorAll('.cmd-days .filter-btn').forEach(x => x.classList.toggle('active', x === b));
      apply();
    }}));
    q.addEventListener('input', apply);
    // #day13 → Day13만, #d13-cv2-resize → 그 항목으로
    const h = decodeURIComponent(location.hash.slice(1));
    const m = h.match(/^day(\\d+)$/);
    if (m) {{ const b = document.querySelector(`.cmd-days .filter-btn[data-day="${{+m[1]}}"]`); if (b) b.click(); }}
    apply();
    if (h && !m) {{ const t = document.getElementById(h); if (t) setTimeout(() => {{ t.scrollIntoView({{ block: 'start' }}); t.classList.add('flash'); }}, 60); }}
  }})();
  </script>
</body>
</html>
'''
    (ROOT / "commands.html").write_text(page, encoding="utf-8")
    print(f"commands.html — {len(items)} commands, topics: {', '.join(topics)}")


if __name__ == "__main__":
    main()
