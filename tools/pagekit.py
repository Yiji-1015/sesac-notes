"""주제 페이지 HTML 을 같은 틀로 만드는 도구 (STYLE.md 4장의 순서).

페이지 원고(tools/pages/*.py)는 설명 글을 HTML 조각으로 주고, 이 도구가 머리말, 한 문장, 왜,
통째로 한 번, 한 단계씩(오른쪽 자라는 코드), 실제로 할 때, 쓴 라이브러리, 원본 자료 순서로 묶는다.
출력 상자는 <div class="out" data-out="키"></div> 자리만 두고, 값은 같은 이름의 -data.js 가 채운다.
"""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def out(key):
    return f'<div class="out" data-out="{key}"></div>'


def kicker(text):
    return f'<p class="kicker">{text}</p>'


def code_table(rows):
    """코드 읽기 표. rows: [(코드, 뜻), …] — 코드는 escape 해서 <code>로, 뜻은 HTML 그대로."""
    body = "".join(f"<tr><td><code>{escape(c)}</code></td><td>{d}</td></tr>" for c, d in rows)
    return f'{kicker("코드 읽기")}<table><tr><th>코드</th><th>뜻</th></tr>{body}</table>'


def table(cols, rows):
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def try_(question, key, hint="먼저 예상해 보고 펼쳐 보세요.", after=""):
    return (f'<details class="try"><summary>🔁 바꿔 보기: {question}</summary>'
            f'<p class="hint">{hint}</p>{out(key)}{after}</details>')


def step(n, name, does, body, run_key=None):
    """한 단계. run_key 가 있으면 끝에 '이 단계까지 실행하면' 상자를 붙인다."""
    circ = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"[n - 1]
    tail = f'{kicker("이 단계까지 실행하면")}{out(run_key)}' if run_key else ""
    return (f'<section class="step" data-step="{n}" id="s{n}"><span class="label">코드의 {circ} 부분</span>'
            f'<h3>{circ} {name}</h3><p class="does">{does}</p>{body}{tail}</section>')


def build(path, *, title, h1, track, crumb, one_line, why, whole, steps, code_file, practice, libs,
          libs_note="", next_rows=(), sources=(), data_js=None, script=""):
    path = Path(path)
    rel = Path(*([".."] * (len(path.relative_to(ROOT).parts) - 1)))
    up = rel.as_posix()
    tag = {"백엔드": "llm", "모델": "dl"}[track]
    data_js = data_js or path.with_name(path.stem + "-data.js").name
    nxt = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in next_rows)
    src = "".join(f"<li>{s}</li>" for s in sources)
    prac = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in practice)
    lib = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in libs)
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{escape(title)} | 부트캠프 학습 노트</title>
  <link rel="stylesheet" href="{up}/style.css">
  <link rel="stylesheet" href="{up}/topics/steps.css">
</head>
<body>
  <main class="container wide">
    <a class="back" href="{up}/index.html">← 목록으로</a>
    <header class="note-header">
      <span class="tag {tag}">{track}</span>
      <h1>{h1}</h1>
      <div class="meta">{crumb}</div>
    </header>
    <article class="note">
      <h2 id="what">💡 한 문장으로</h2>
      <p class="one-line">{one_line}</p>
      <h2 id="why">🤔 왜 필요할까?</h2>
      {why}
      <h2 id="whole">🗺️ 통째로 한 번</h2>
      {whole}
      <h2 id="steps">🔎 한 단계씩: 설명과 코드를 나란히</h2>
      <div class="scrolly">
        <div class="steps">
          {''.join(steps)}
        </div>
        <aside class="codepanel" id="panel">
          <div class="bar"><span class="file">{code_file}</span><span id="pills"></span><button class="pill" id="showall" type="button">전체 보기</button></div>
          <div class="now" id="now"></div>
          <div class="legend"><i style="background:#3ddc84"></i>이 단계에서 추가된 줄<i style="background:rgba(255,255,255,.35)"></i>앞 단계에서 만든 줄</div>
          <div class="codewrap"><pre class="code" id="code"></pre></div>
        </aside>
      </div>
      <h2 id="practice">🛠️ 실제로 할 때</h2>
      <table><tr><th>상황</th><th>어떻게</th></tr>{prac}</table>
      <h2 id="libs">📦 이 코드에 쓴 라이브러리</h2>
      <table><tr><th>단계</th><th>함수</th><th>하는 일</th></tr>{lib}</table>
      {f'<p>{libs_note}</p>' if libs_note else ''}
      <h2 id="next">이어서 볼 주제</h2>
      <table><tr><th>주제</th><th>연결</th></tr>{nxt}</table>
      <h2>원본 자료</h2>
      <ul>{src}<li>실행 환경: <span id="env"></span></li></ul>
    </article>
  </main>
  <script src="{up}/glossary.js"></script>
  <script src="{up}/note.js"></script>
  <script src="{data_js}"></script>
  <script src="{up}/topics/steps.js"></script>
  {f'<script>{script}</script>' if script else ''}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
