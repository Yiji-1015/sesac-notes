// ===== 공통 스크립트 (모든 페이지) =====

// 1) 테마: 오른쪽 아래 🎨 버튼. 고른 색은 브라우저에 저장돼서 다음에도 유지돼요.
(() => {
  const THEMES = [
    ['lavender', '라벤더', 'linear-gradient(135deg,#7c4dff,#ff7eb3)'],
    ['mint', '민트', 'linear-gradient(135deg,#12b886,#4dabf7)'],
    ['peach', '피치', 'linear-gradient(135deg,#ff6b6b,#ffd43b)'],
    ['night', '밤', 'linear-gradient(135deg,#1d1838,#a78bfa)'],
  ];
  let saved = 'lavender';
  try { saved = localStorage.getItem('theme') || saved; } catch {}
  document.documentElement.dataset.theme = saved;

  const box = document.createElement('div');
  box.className = 'theme-picker';
  box.innerHTML = '<span title="테마">🎨</span>';
  THEMES.forEach(([key, label, bg]) => {
    const b = document.createElement('button');
    b.title = label; b.setAttribute('aria-label', label + ' 테마');
    b.style.background = bg;
    if (key === saved) b.classList.add('on');
    b.addEventListener('click', () => {
      document.documentElement.dataset.theme = key;
      try { localStorage.setItem('theme', key); } catch {}
      box.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    });
    box.appendChild(b);
  });
  // 글꼴 고르기
  const FONTS = [
    ['cute', '동글', 'Jua'],
    ['soft', '부드럽게', 'Gowun Dodum'],
    ['clean', '깔끔', 'Pretendard Variable'],
  ];
  let savedFont = 'cute';
  try { savedFont = localStorage.getItem('font') || savedFont; } catch {}
  document.documentElement.dataset.font = savedFont;
  const sep = document.createElement('div'); sep.className = 'sep'; box.appendChild(sep);
  FONTS.forEach(([key, label, family]) => {
    const b = document.createElement('button');
    b.className = 'font' + (key === savedFont ? ' on' : '');
    b.textContent = label; b.title = label + ' 글꼴';
    b.style.fontFamily = `"${family}", sans-serif`;
    b.addEventListener('click', () => {
      document.documentElement.dataset.font = key;
      try { localStorage.setItem('font', key); } catch {}
      box.querySelectorAll('button.font').forEach(x => x.classList.toggle('on', x === b));
    });
    box.appendChild(b);
  });

  document.body.appendChild(box);
})();

// 2) 용어: 페이지 맨 위 "📖 오늘 나오는 용어" 상자 + 본문 용어에 말풍선
//    (데이터는 glossary.js — terms/*.json 을 고치고 build_glossary.py 실행)
(() => {
  const G = window.GLOSSARY;
  if (!G || !G.length) return;
  const inNotes = /\/notes\//.test(location.pathname.replace(/\\/g, '/'));
  const dictUrl = (inNotes ? '../' : '') + 'glossary.html';
  const slug = t => 't-' + encodeURIComponent(t.toLowerCase());
  const m = location.pathname.match(/day(\d+)\.html$/);
  const pageDay = m ? parseInt(m[1], 10) : null;
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // (a) 오늘 나오는 용어 상자
  const note = document.querySelector('article.note');
  if (note && pageDay) {
    const keys = G.filter(e => e.keyDays.includes(pageDay));
    const all = G.filter(e => e.days.includes(pageDay));
    if (keys.length) {
      const box = document.createElement('section');
      box.className = 'term-box';
      box.innerHTML = `<h2 class="term-box-title">📖 오늘 나오는 용어 <small>— 이것만 알고 시작해요</small></h2>
        <div class="term-cards">${keys.map(e => `
          <div class="term-card">
            <div class="term-name">${esc(e.term)}</div>
            ${e.full && e.full !== e.term ? `<div class="term-full">${esc(e.full)}</div>` : ''}
            <p>${esc(e.short)}</p>
            ${e.analogy ? `<p class="term-analogy">💭 ${esc(e.analogy)}</p>` : ''}
          </div>`).join('')}
        </div>
        <p class="term-more">본문에서 <span class="term-sample">점선 밑줄</span> 단어에 마우스를 올리면(휴대폰은 탭) 뜻이 나와요 ·
          <a href="${dictUrl}#day${pageDay}">이 Day 용어 ${all.length}개 전부 보기 →</a></p>`;
      const tldr = note.querySelector('.tldr');
      if (tldr) tldr.after(box); else note.prepend(box);
    }
  }

  // (b) 본문 용어에 말풍선 — 각 큰 제목(h2) 구역마다 처음 나온 곳에만
  const lookup = new Map();
  G.forEach(e => [e.term, ...e.aliases].forEach(w => { if (w && w.length >= 2 && !lookup.has(w)) lookup.set(w, e); }));
  const words = [...lookup.keys()].sort((a, b) => b.length - a.length);
  if (!words.length) return;
  const re = new RegExp('(?<![A-Za-z0-9_])(' + words.map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'g');
  const isAsciiEnd = w => /[A-Za-z0-9]$/.test(w);
  const SKIP = 'svg, code, pre, a, h1, h2, h3, script, style, .lab, .term-box, .term, .theme-picker, .site-header, .note-header, button, input, label, canvas';

  const scope = document.querySelector('article.note') || document.querySelector('main');
  if (!scope) return;
  let seen = new Set();
  const walker = document.createTreeWalker(scope, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      if (n.nodeType === 1) return n.matches(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_SKIP;
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  // h2 를 만나면 "이미 표시한 용어" 목록을 초기화
  const h2s = [...scope.querySelectorAll('h2')];
  const sectionOf = node => { let k = -1; h2s.forEach((h, i) => { if (h.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING) k = i; }); return k; };
  const texts = [];
  while (walker.nextNode()) texts.push(walker.currentNode);
  // 오늘의 핵심 용어는 구역마다 한 번, 나머지 용어는 페이지 전체에서 한 번만 밑줄
  const isKey = e => pageDay && e.keyDays.includes(pageDay);
  const seenPage = new Set();
  let curSection = -2;
  texts.forEach(tn => {
    const sec = sectionOf(tn);
    if (sec !== curSection) { curSection = sec; seen = new Set([...seenPage]); }
    const s = tn.nodeValue; re.lastIndex = 0;
    let match, last = 0; const frag = document.createDocumentFragment(); let changed = false;
    while ((match = re.exec(s))) {
      const w = match[1], e = lookup.get(w), after = s[match.index + w.length] || '';
      if (isAsciiEnd(w) && /[A-Za-z0-9_]/.test(after)) continue;
      if (seen.has(e.term)) continue;
      seen.add(e.term);
      if (!isKey(e)) seenPage.add(e.term);
      frag.append(s.slice(last, match.index));
      const span = document.createElement('span');
      span.className = 'term'; span.tabIndex = 0; span.textContent = w; span.dataset.term = e.term;
      frag.append(span); last = match.index + w.length; changed = true;
    }
    if (changed) { frag.append(s.slice(last)); tn.replaceWith(frag); }
  });

  // 말풍선
  const byTerm = new Map(G.map(e => [e.term, e]));
  const pop = document.createElement('div');
  pop.className = 'term-pop'; pop.setAttribute('role', 'tooltip');
  document.body.appendChild(pop);
  let hideT;
  const show = el => {
    clearTimeout(hideT);
    const e = byTerm.get(el.dataset.term); if (!e) return;
    pop.innerHTML = `<strong>${esc(e.term)}</strong>${e.full && e.full !== e.term ? ` <span class="pop-full">${esc(e.full)}</span>` : ''}
      <p>${esc(e.short)}</p>${e.analogy ? `<p class="term-analogy">💭 ${esc(e.analogy)}</p>` : ''}
      <a href="${dictUrl}#${slug(e.term)}">📚 용어 사전에서 보기</a>${e.first !== pageDay ? ` <span class="pop-day">· Day${e.first}에서 처음 나옴</span>` : ''}`;
    pop.classList.add('on');
    const r = el.getBoundingClientRect(), pw = Math.min(320, window.innerWidth - 24);
    pop.style.width = pw + 'px';
    let left = r.left + r.width / 2 - pw / 2 + window.scrollX;
    left = Math.max(12 + window.scrollX, Math.min(left, window.scrollX + window.innerWidth - pw - 12));
    pop.style.left = left + 'px';
    const below = r.bottom + window.scrollY + 8;
    pop.style.top = below + 'px';
    const ph = pop.offsetHeight;
    if (r.bottom + ph + 16 > window.innerHeight && r.top > ph + 16) pop.style.top = (r.top + window.scrollY - ph - 8) + 'px';
  };
  const hide = () => { hideT = setTimeout(() => pop.classList.remove('on'), 150); };
  document.addEventListener('mouseover', ev => { const t = ev.target.closest('.term'); if (t) show(t); else if (!ev.target.closest('.term-pop')) hide(); });
  document.addEventListener('focusin', ev => { const t = ev.target.closest('.term'); if (t) show(t); });
  document.addEventListener('click', ev => { const t = ev.target.closest('.term'); if (t) show(t); else if (!ev.target.closest('.term-pop')) pop.classList.remove('on'); });
  pop.addEventListener('mouseenter', () => clearTimeout(hideT));
  pop.addEventListener('mouseleave', hide);
})();

// 3) 그림 크게 보기: 본문 그림을 누르면 화면 가득, 다시 누르거나 Esc로 닫기
(() => {
  const ov = document.createElement('div');
  ov.className = 'zoom-ov';
  ov.innerHTML = '<img alt=""><p></p>';
  document.body.appendChild(ov);
  const close = () => ov.classList.remove('on');
  document.addEventListener('click', ev => {
    const img = ev.target.closest('.note figure img');
    if (img) {
      ov.querySelector('img').src = img.src;
      ov.querySelector('img').className = img.className;
      const cap = img.closest('figure').querySelector('figcaption');
      ov.querySelector('p').innerHTML = cap ? cap.innerHTML : '';
      ov.classList.add('on');
    } else if (ev.target.closest('.zoom-ov')) close();
  });
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape') close(); });
})();

// 4) 노트 페이지: 코드 블록마다 "복사" 버튼
document.querySelectorAll('.note pre').forEach(pre => {
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.textContent = '복사';
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.querySelector('code')?.innerText ?? pre.innerText);
      btn.textContent = '복사됨!';
    } catch {
      btn.textContent = '실패';
    }
    setTimeout(() => (btn.textContent = '복사'), 1200);
  });
  pre.appendChild(btn);
});
