/* 주제 페이지 공통: 자라는 코드 패널과 단계 스크롤 연동.
   페이지는 window.PAGE = { code: [{t, s}], names: {n: 이름}, range: [첫 단계, 끝 단계], file: "경로" } 를 먼저 읽어 둬요.
   이 파일은 window.Steps 에 출력 상자용 도구(esc, q, C, hl, outBox, tbl)를 내보내요. */
(() => {
  const P = window.PAGE;
  const C = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳";
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const q = s => JSON.stringify(s);

  /* 아주 작은 Python 색칠 */
  const KW = new Set("from import for in if else elif def return assert and or not None True False try except as with is".split(" "));
  const NB = new Set("len list enumerate print open zip any sorted str int dict range".split(" "));
  function hl(line) {
    let out = "", i = 0;
    while (i < line.length) {
      const ch = line[i];
      if (ch === "#") { out += `<span class="cm">${esc(line.slice(i))}</span>`; break; }
      const pre = /^[rfbRFB]{0,2}["']/.exec(line.slice(i));
      if (pre && (i === 0 || !/[A-Za-z0-9_]/.test(line[i - 1]))) {
        const quote = pre[0].slice(-1); let j = i + pre[0].length;
        while (j < line.length && line[j] !== quote) { if (line[j] === "\\") j++; j++; }
        out += `<span class="st">${esc(line.slice(i, j + 1))}</span>`; i = j + 1; continue;
      }
      const w = /^[A-Za-z_][A-Za-z0-9_]*/.exec(line.slice(i));
      if (w) { const t = w[0]; out += KW.has(t) ? `<span class="kw">${t}</span>` : NB.has(t) ? `<span class="nb">${t}</span>` : t; i += t.length; continue; }
      out += esc(ch); i++;
    }
    return out;
  }

  function outBox(el, title, rows, notes) {
    const body = rows.map(r => typeof r === "string" ? esc(r) : (r[2] ? r[0] : esc(r[0])) + (r[1] ? `<span class="mk">${C[r[1] - 1]}</span>` : "")).join("\n");
    $(el).innerHTML = `<div class="ttl">▶ ${title}</div><pre><code>${body}</code></pre>` + (notes ? `<ol class="ann">${notes.map(n => `<li>${n}</li>`).join("")}</ol>` : "");
  }
  const tbl = (cols, rows) => `<table>${"<tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr>"}${rows.map(r => "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("")}</table>`;
  window.Steps = { esc, q, C, hl, outBox, tbl, $ };

  /* 오른쪽 코드 패널 */
  const [lo, hi] = P.range;
  const codeEl = $("code");
  /* 앞 쪽(lo 이전 단계)에서 만든 줄은 묶어서 접어 둬요 */
  let html = "", run = 0;
  const flush = () => { if (run) { const t = `… 앞 쪽에서 만든 줄 ${run}개 (눌러서 펼치기)`; html += `<span class="ln prevfold" data-label="${t}">${t}</span>`; run = 0; } };
  P.code.forEach((c, i) => {
    const lab = /^# [①-⑳] /.test(c.t) ? " labelln" : "";
    if (c.s < lo) { run++; html += `<span class="ln prev${lab}" data-s="${c.s}"><span class="no">${i + 1}</span>${hl(c.t) || " "}</span>`; return; }
    flush();
    html += `<span class="ln${lab}" data-s="${c.s}"><span class="no">${i + 1}</span>${hl(c.t) || " "}</span>`;
  });
  flush();
  codeEl.innerHTML = html + `<span class="more" id="more"></span>`;
  /* 접힌 묶음 표시를 그 묶음 앞으로 옮기기 */
  codeEl.querySelectorAll(".prevfold").forEach(f => {
    let el = f.previousElementSibling, first = null;
    while (el && el.classList.contains("prev")) { first = el; el = el.previousElementSibling; }
    if (first) codeEl.insertBefore(f, first);
  });
  let showPrev = false;
  codeEl.addEventListener("click", e => { if (e.target.closest(".prevfold")) { showPrev = !showPrev; paint(current); } });
  const lines = [...codeEl.querySelectorAll(".ln:not(.prevfold)")];
  $("pills").innerHTML = Array.from({ length: hi - lo + 1 }, (_, k) => `<button class="pill" type="button" data-go="${lo + k}">${C[lo + k - 1]}</button>`).join(" ");
  let showAll = false, current = lo;
  function paint(k) {
    current = k;
    lines.forEach(l => {
      const s = +l.dataset.s;
      l.className = l.className.replace(/ ?(old|new|hide)\b/g, "");
      if (s < lo) { l.classList.add(showPrev ? "old" : "hide"); return; }
      l.classList.add(s === k ? "new" : s < k ? "old" : (showAll ? "old" : "hide"));
    });
    codeEl.querySelectorAll(".prevfold").forEach(f => { f.textContent = showPrev ? "▲ 앞 쪽에서 만든 줄 접기" : f.dataset.label; });
    const left = lines.filter(l => +l.dataset.s > k).length;
    $("more").textContent = !showAll && left ? `… 다음 단계에서 ${left}줄이 더 붙어요` : "";
    const added = lines.filter(l => +l.dataset.s === k && l.textContent.trim().length > 2);
    const label = lines.find(l => +l.dataset.s === k && l.classList.contains("labelln"));
    const above = label ? added.filter(l => lines.indexOf(l) < lines.indexOf(label)).length : 0;
    $("now").innerHTML = `지금: <b>${C[k - 1]} ${esc(P.names[k] || "")}</b> &nbsp;<span style="opacity:.75">(+${added.length}줄${above ? `, 그중 ${above}줄은 파일 위쪽 import` : ""})</span>`;
    document.querySelectorAll("#pills .pill").forEach(b => { const g = +b.dataset.go; b.classList.toggle("on", g === k); b.classList.toggle("done", g < k); });
    document.querySelectorAll(".step").forEach(st => st.classList.toggle("on", +st.dataset.step === k));
    const target = label || lines.find(l => +l.dataset.s === k);
    if (target) codeEl.parentElement.scrollTo({ top: Math.max(0, target.offsetTop - 60), behavior: "smooth" });
  }
  $("showall").addEventListener("click", e => { showAll = !showAll; e.target.classList.toggle("on", showAll); paint(current); });
  document.addEventListener("click", e => { const g = e.target.closest("[data-go]"); if (g) $("s" + g.dataset.go).scrollIntoView({ behavior: "smooth", block: "start" }); });
  const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) paint(+en.target.dataset.step); }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll(".step").forEach(st => io.observe(st));

  /* 좁은 화면: 단계마다 새 줄만 설명 안에 */
  document.querySelectorAll(".step").forEach(st => {
    const k = +st.dataset.step;
    const nw = P.code.map((c, i) => ({ ...c, i })).filter(c => c.s === k);
    const box = document.createElement("div"); box.className = "inline-code";
    box.innerHTML = `<div class="mini"><pre class="code">${nw.map(c => `<span class="ln new"><span class="no">${c.i + 1}</span>${hl(c.t) || " "}</span>`).join("")}</pre></div>`;
    (st.querySelector(".does") || st.querySelector("h3")).after(box);
  });
  window.addEventListener("load", () => paint(current));
  paint(lo);
})();
