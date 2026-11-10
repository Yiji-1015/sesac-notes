"""이론 페이지에 남아 있던 코드 설명(🔧 제목이 없던 코드 블록·인자 표)을 명령어 사전으로 옮긴다. (2026-11-10 1회용)

각 항목: (day, 시작 줄, 끝 줄, 사전 제목, 주제, 소주제, 이론 구역 id)  — 줄 번호는 옮기기 전 기준(1부터).
같은 파일 안에서는 아래쪽부터 처리해서 줄 번호가 밀리지 않게 한다.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_commands import key_lines  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

ITEMS = [
    (6, 269, 280, "KNN 스케일링 전후 비교 <code>StandardScaler</code>", "scikit-learn", "스케일링 · Pipeline", "scaling"),
    (6, 295, 302, "<code>StandardScaler</code> · <code>MinMaxScaler</code> 값 확인", "scikit-learn", "스케일링 · Pipeline", "scaling"),
    (7, 228, 235, "부스팅 한 회차 직접 계산 <code>DecisionTreeRegressor</code>", "XGBoost", "", "xgb"),
    (8, 157, 163, "MAE · RMSE 계산 <code>mean_absolute_error</code> · <code>root_mean_squared_error</code>", "scikit-learn", "회귀 지표", "cvagg"),
    (9, 372, 380, "<code>model.train()</code> · <code>model.eval()</code> · <code>torch.no_grad()</code>", "PyTorch", "학습 루프", "loop"),
    (11, 233, 240, "증강 <code>transforms.Compose</code> (train만)", "torchvision", "데이터 · 증강", "data"),
    (13, 329, 337, "Frame 건너뛰며 처리 <code>frame_idx % step</code>", "OpenCV", "영상", "video"),
    (14, 446, 459, "재현 · 저장 <code>torch.manual_seed()</code> · <code>torch.save()</code> · <code>torch.load()</code>", "PyTorch", "저장 · 재현", "repro"),
    (15, 384, 390, "같은 걸 터미널(CLI)로 <code>yolo detect ...</code>", "YOLO", "CLI", "track"),
]


def slug(day, title):
    codes = re.findall(r"<code>([^<]+)</code>", title)
    base = codes[0] if codes else re.sub(r"<[^>]+>", "", title)
    base = re.sub(r"[^0-9A-Za-z가-힣]+", "-", base).strip("-").lower()
    return f"d{day}-{base}"


by_day = {}
for it in ITEMS:
    by_day.setdefault(it[0], []).append(it)

for day, items in by_day.items():
    page = ROOT / "notes" / f"day{day:02d}.html"
    frag = ROOT / "cmds" / f"day{day:02d}.html"
    lines = page.read_text(encoding="utf-8").split("\n")
    fs = frag.read_text(encoding="utf-8")
    new_secs = []
    for d, a, b, title, topic, sub, sec in sorted(items, key=lambda x: -x[1]):
        block = lines[a - 1:b]
        # h2/h3 제목 줄과 주석 구분선은 사전으로 가져가지 않음
        body = [l for l in block if not re.match(r"\s*<(h2|h3)[ >]|\s*<!-- =+", l)]
        cid = slug(day, title)
        assert f'id="{cid}"' not in fs, cid
        new_secs.append(
            f'<section class="cmd" id="{cid}" data-topic="{topic}" data-sub="{sub}" data-day="{day}" '
            f'data-theory="notes/day{day:02d}.html#{sec}">\n  <h3>{title}</h3>\n' + "\n".join(body) + "\n</section>\n")
        snip = key_lines("\n".join(body))
        ref = ["      <div class=\"cmdref\">",
               f"        <div class=\"cmdref-head\">🔧 {title} <a href=\"../commands.html#{cid}\">명령어 사전에서 자세히 →</a></div>"]
        if snip:
            ref.append("<pre><code>" + "\n".join(snip) + "</code></pre>")
        ref.append("      </div>")
        keep_heads = [l for l in block if re.match(r"\s*<!-- =+", l)]
        lines[a - 1:b] = keep_heads + ref
        print(f"{cid:45s} lines {b - a + 1} → cmdref, snippet {len(snip)}")
    fs = fs.rstrip("\n") + "\n" + "".join(reversed(new_secs))
    frag.write_text(fs, encoding="utf-8")
    page.write_text("\n".join(lines), encoding="utf-8")
