"""Day 페이지의 🔧 명령어 블록을 명령어 사전 조각(cmds/dayNN.html)으로 옮긴다.

- 코드·인자 표·⚠️/🤔 박스 → cmds/dayNN.html 의 <section class="cmd">
- 그림(figure, .figs)        → Day 페이지에 그대로 남김
- 원래 자리                   → 짧은 핵심 코드 + "명령어 사전에서 보기" 링크(.cmdref)

사용: python tools/split_commands.py 13 [--dry]
주제 이름은 아래 TOPICS(h2 id → (주제, 소주제))에서 정한다. 한 번 옮긴 Day는 다시 돌려도 바뀌지 않는다.
"""
import html
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

# Day별: h2 id → (주제, 소주제)
TOPICS = {
    6: {"regmetrics": ("scikit-learn", "회귀 지표"), "logistic": ("scikit-learn", "로지스틱회귀 · 임계값"),
        "knn": ("scikit-learn", "KNN"), "compare": ("scikit-learn", "모델 비교 · 기준선"),
        "checks": ("scikit-learn", "입력 검사 · 저장")},
    7: {"overfit": ("scikit-learn", "트리 · 복잡도"), "xgb": ("XGBoost", ""),
        "cv": ("scikit-learn", "교차검증"), "checks7": ("scikit-learn", "입력 검사 · 저장")},
    8: {"clsmetrics": ("scikit-learn", "분류 지표"), "roc": ("scikit-learn", "ROC · 임계값"),
        "cvagg": ("scikit-learn", "교차검증"), "final": ("scikit-learn", "입력 검사 · 저장")},
    9: {"gpu": ("PyTorch", "GPU · 장치"), "tensor": ("PyTorch", "Tensor"), "autograd": ("PyTorch", "Autograd · 경사하강"),
        "ann": ("PyTorch", "모델 만들기"), "loop": ("PyTorch", "학습 루프")},
    10: {"optim": ("PyTorch", "Optimizer · 규제"), "lr": ("PyTorch", "학습률"), "batch": ("PyTorch", "DataLoader · Batch"),
         "overfit": ("PyTorch", "학습 루프"), "early": ("PyTorch", "Optimizer · 규제")},
    11: {"tensor": ("PyTorch", "Tensor"), "pool": ("PyTorch", "모델 만들기"), "data": ("PyTorch", "학습 루프"),
         "transfer": ("torchvision", "전이학습"), "bench": ("PyTorch", "속도 측정")},
    12: {"speed": ("PyTorch", "속도 측정")},
    13: {
        "array": ("OpenCV", "읽기 · 저장 · 표시"),
        "color": ("OpenCV", "색 · 크기"),
        "filter": ("OpenCV", "필터 · 경계"),
        "thresh": ("OpenCV", "이진화 · 형태학"),
        "video": ("OpenCV", "영상"),
        "augment": ("OpenCV", "증강 · 기하 변환"),
        "verify": ("OpenCV", "일괄 처리 · 검증"),
    },
    14: {"dataset": ("torchvision", "데이터셋 · 분할"), "transfer": ("torchvision", "전이학습"),
         "finetune": ("torchvision", "전이학습"), "eval": ("PyTorch", "평가 · 오류 분석"),
         "gradcam": ("PyTorch", "평가 · 오류 분석")},
    15: {"box": ("YOLO", "박스 좌표"), "metrics": ("YOLO", "IoU · 평가"), "nms": ("YOLO", "추론 · 결과"),
         "track": ("YOLO", "추적")},
    16: {"prep": ("OpenCV", "문서 전처리"), "ocr": ("OCR · 문서", "OCR 실행"), "eval": ("OCR · 문서", "품질 평가"),
         "struct": ("OCR · 문서", "구조화"), "ground": ("VLM · Transformers", "근거 검증 · Hybrid"),
         "deploy": ("배포 · 최적화", "TensorRT"), "app": ("OCR · 문서", "앱 설계")},
    17: {"first": ("FastAPI", "서버 실행"), "storage": ("FastAPI", "파일 저장"), "read": ("FastAPI", "조회 (GET)"),
         "create": ("FastAPI", "등록 · 검증 (POST)"), "update": ("FastAPI", "수정 · 삭제"),
         "delete": ("FastAPI", "수정 · 삭제"), "structure": ("FastAPI", "구조 · 테스트")},
}

END = re.compile(r'^\s*(<h2|<h3|<!-- =+)')
OPEN = re.compile(r'<(div|table|pre|figure|ul|ol|p)\b')
CLOSE = re.compile(r'</(div|table|pre|figure|ul|ol|p)>')


def find_blocks(lines):
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].lstrip().startswith('<h3>🔧') and '설치' not in lines[i] and h2_id(lines, i) != 'packages':
            j = i + 1
            while j < len(lines) and not END.match(lines[j]):
                j += 1
            while j - 1 > i and not lines[j - 1].strip():
                j -= 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def split_elements(lines):
    """블록 본문(h3 제외)을 최상위 요소 단위로 자른다."""
    out, cur, depth = [], [], 0
    for ln in lines:
        if not cur and not ln.strip():
            continue
        cur.append(ln)
        depth += len(OPEN.findall(ln)) - len(CLOSE.findall(ln))
        if depth <= 0:
            out.append('\n'.join(cur))
            cur, depth = [], 0
    if cur:
        out.append('\n'.join(cur))
    return out


def h2_id(lines, i):
    for k in range(i, -1, -1):
        m = re.search(r'<h2 id="([^"]+)"', lines[k])
        if m:
            return m.group(1)
    return ''


def slug(day, h3):
    codes = re.findall(r'<code>([^<]+)</code>', h3)
    base = codes[0] if codes else re.sub(r'<[^>]+>', '', h3)
    base = re.sub(r'[^0-9A-Za-z가-힣]+', '-', base.replace('🔧', '').replace('명령어:', '')).strip('-').lower()
    return f"d{day}-{base}"


def key_lines(pre_html, n=2):
    """코드 블록에서 핵심 줄 n개 (import·주석·빈 줄 제외, 괄호가 있는 줄)."""
    m = re.search(r'<pre><code>(.*?)</code></pre>', pre_html, re.S)
    if not m:
        return []
    picked = []
    for ln in m.group(1).split('\n'):
        s = ln.strip()
        if not s or s.startswith('#') or s.startswith('import ') or s.startswith('from ') or '(' not in s:
            continue
        picked.append(ln.rstrip())
        if len(picked) == n:
            break
    return picked


def title_of(h3):
    t = re.sub(r'^\s*<h3>🔧\s*(명령어:\s*)?', '', h3)
    return re.sub(r'</h3>\s*$', '', t).strip()


def move_overview(day, dry=False):
    """'이론 ↔ 명령어 한눈에' 표를 사전 조각으로 옮기고, Day 페이지엔 링크 한 줄만 남긴다."""
    page = ROOT / 'notes' / f'day{day:02d}.html'
    frag = ROOT / 'cmds' / f'day{day:02d}.html'
    s = page.read_text(encoding='utf-8')
    m = re.search(r'[ \t]*<h2>이론 ↔ 명령어 한눈에</h2>\s*(<table>.*?</table>)\n'
                  r'(?:[ \t]*<p>🔧 코드·인자 설명은.*?</p>\n)?', s, re.S)
    if not m:
        print('overview: 없음 (이미 옮겼거나 표가 없음)')
        return
    table = re.sub(r'href="#([^"]+)"', rf'href="notes/day{day:02d}.html#\1"', m.group(1))
    link = (f'      <div class="cmdref">\n'
            f'        <div class="cmdref-head">🔧 Day{day} 명령어 한눈에 — 이론 구역별 명령어·인자 설명 '
            f'<a href="../commands.html#day{day}">명령어 사전에서 보기 →</a></div>\n'
            f'        <p style="margin:6px 0 0;color:var(--muted);font-size:0.9rem">본문에는 핵심 코드 한두 줄만 남겼어요.</p>\n'
            f'      </div>\n')
    print('overview: 옮김')
    if dry:
        return
    page.write_text(s[:m.start()] + link + s[m.end():], encoding='utf-8')
    body = frag.read_text(encoding='utf-8') if frag.exists() else ''
    body = re.sub(r'<section class="cmd-overview".*?</section>\n', '', body, flags=re.S)
    ov = f'<section class="cmd-overview" data-day="{day}">\n{table}\n</section>\n'
    head, sep, rest = body.partition('-->\n')
    body = head + sep + ov + rest if sep else ov + body
    frag.parent.mkdir(exist_ok=True)
    frag.write_text(body, encoding='utf-8')


def main():
    day = int(sys.argv[1])
    dry = '--dry' in sys.argv
    page = ROOT / 'notes' / f'day{day:02d}.html'
    lines = page.read_text(encoding='utf-8').split('\n')
    if not find_blocks(lines):
        print('명령어 블록: 없음 (이미 옮겼음)')
        move_overview(day, dry)
        return
    topics = TOPICS[day]
    sections, new_lines, last = [], [], 0
    for a, b in find_blocks(lines):
        h3 = lines[a]
        sec = h2_id(lines, a)
        topic, sub = topics.get(sec, ("기타", ""))
        cid = slug(day, h3)
        elems = split_elements(lines[a + 1:b])
        keep = [e for e in elems if e.lstrip().startswith(('<figure', '<div class="figs"'))]
        move = [e for e in elems if e not in keep]
        pre = next((e for e in move if e.lstrip().startswith('<pre')), '')
        snippet = key_lines(pre)
        title = title_of(h3)
        sections.append(
            f'<section class="cmd" id="{cid}" data-topic="{topic}" data-sub="{sub}" data-day="{day}" '
            f'data-theory="notes/day{day:02d}.html#{sec}">\n'
            f'  <h3>{title}</h3>\n' + '\n'.join(move) + '\n</section>\n')
        ref = ['      <div class="cmdref">',
               f'        <div class="cmdref-head">🔧 {title} <a href="../commands.html#{cid}">명령어 사전에서 자세히 →</a></div>']
        if snippet:
            ref.append('<pre><code>' + '\n'.join(snippet) + '</code></pre>')
        ref.append('      </div>')
        new_lines += lines[last:a] + ref + keep
        last = b
        print(f'{cid:40s} {topic}/{sub}  keep {len(keep)} · move {len(move)} · snippet {len(snippet)}')
    new_lines += lines[last:]
    if dry:
        return
    out = ROOT / 'cmds'
    out.mkdir(exist_ok=True)
    (out / f'day{day:02d}.html').write_text(
        f'<!-- Day{day} 명령어 — tools/split_commands.py 로 notes/day{day:02d}.html 에서 옮김. 고칠 땐 이 파일을 고치고 python tools/build_commands.py -->\n'
        + '\n'.join(sections), encoding='utf-8')
    page.write_text('\n'.join(new_lines), encoding='utf-8')
    print('written', len(sections), 'sections')
    move_overview(day)


if __name__ == '__main__':
    main()
