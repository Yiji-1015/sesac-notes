# 문서 하나를 검색용 조각(Chunk)으로 나누는 전체 코드.
# 노트의 ①~⑥ 단계가 이 파일의 ①~⑥ 주석과 1:1로 맞아요.
# 실행: python chunk_pipeline.py  →  같은 폴더에 chunks.json 생성
#
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                      #@1
import re                                                     #@2
from transformers import AutoTokenizer                        #@3
import hashlib                                                #@5
import json                                                   #@5

HERE = Path(__file__).resolve().parent                        #@1

# ① 원문 읽기
text = Path(HERE / "library_guide.md").read_text(encoding="utf-8")

# ② 제목으로 나누기
heads = list(re.finditer(r"^## (.+)$", text, flags=re.M))
sections = []
for i, h in enumerate(heads):
    start = h.end() + 1  # 제목 줄 다음 글자부터
    end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
    end = start + len(text[start:end].rstrip())  # 끝의 빈 줄은 빼기
    sections.append({"heading": h.group(1), "start": start, "end": end})

# ③ 토큰 세기
tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
MAX_TOKENS = 64  # 예시 설정

def count(t):
    return len(tok(t, add_special_tokens=False)["input_ids"])

for s in sections:
    s["tokens"] = count(text[s["start"]:s["end"]])

# ④ 긴 섹션만 줄 단위로 다시 묶기
def split_long(s):
    if s["tokens"] <= MAX_TOKENS:
        return [s]
    pieces, cur = [], None
    for m in re.finditer(r"[^\n]+", text[s["start"]:s["end"]]):
        a, b = s["start"] + m.start(), s["start"] + m.end()
        if cur and count(text[cur["start"]:b]) <= MAX_TOKENS:
            cur["end"] = b  # 지금 조각에 이어 붙이기
        else:
            cur = {"heading": s["heading"], "start": a, "end": b}
            pieces.append(cur)  # 새 조각 시작
    return pieces

# ⑤ 조각 기록 만들기
chunks = []
for s in sections:
    for p in split_long(s):
        body = text[p["start"]:p["end"]]
        key = json.dumps({"doc": "library_guide", "start": p["start"],
                          "end": p["end"], "body": body}, ensure_ascii=False, sort_keys=True)
        chunks.append({
            "chunk_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
            "heading": p["heading"],
            "char_start": p["start"],
            "char_end": p["end"],
            "tokens": count(body),
            "body": body,
        })

# ⑥ 검사하고 저장하기
for c in chunks:
    assert text[c["char_start"]:c["char_end"]] == c["body"]  # 위치로 원문을 되찾을 수 있나
    assert c["tokens"] <= MAX_TOKENS  # 모델 한도 안인가
out = HERE / "chunks.json"
out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{len(chunks)}개 조각 저장 → {out.name}")
