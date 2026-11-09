"""terms/dayNN.json 파일들을 합쳐 glossary.js 를 만듭니다.

새 Day 용어를 추가했으면 이 폴더에서 한 번 실행하세요:
    python build_glossary.py

- 같은 용어가 여러 Day에 나오면 가장 먼저 나온 Day의 설명을 쓰고, days 에 모두 기록합니다.
- key=true 인 용어는 그 Day 페이지 맨 위 "📖 오늘 나오는 용어" 상자에 나옵니다.
"""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
merged = {}
order = []
for f in sorted((root / "terms").glob("day*.json")):
    day = int(re.search(r"day(\d+)", f.stem).group(1))
    for e in json.loads(f.read_text(encoding="utf-8")):
        k = e["term"].strip().lower()
        if k not in merged:
            merged[k] = {
                "term": e["term"].strip(),
                "aliases": [a for a in e.get("aliases", []) if a],
                "full": e.get("full", ""),
                "short": e["short"],
                "analogy": e.get("analogy", ""),
                "first": day,
                "days": [day],
                "keyDays": [day] if e.get("key") else [],
            }
            order.append(k)
        else:
            m = merged[k]
            if day not in m["days"]:
                m["days"].append(day)
            if e.get("key") and day not in m["keyDays"]:
                m["keyDays"].append(day)
            for a in e.get("aliases", []):
                if a and a not in m["aliases"]:
                    m["aliases"].append(a)

items = sorted((merged[k] for k in order), key=lambda m: (m["first"], order.index(m["term"].lower())))
out = "// 자동 생성 파일 — 직접 고치지 말고 terms/*.json 을 고친 뒤 build_glossary.py 를 실행하세요.\n"
out += "window.GLOSSARY = " + json.dumps(items, ensure_ascii=False, indent=1) + ";\n"
(root / "glossary.js").write_text(out, encoding="utf-8")
print(f"{len(items)}개 용어 → glossary.js")
