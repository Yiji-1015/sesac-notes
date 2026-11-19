# 가람 도서관 문서 묶음을 검색용 데이터로 만드는 전체 코드.
# 노트의 단계 번호(①, ②, …)가 이 파일의 번호 주석과 1:1로 맞아요.
#   1쪽 문서 읽기: ①~⑦     2쪽 Chunking: ⑧~⑬     3쪽 인덱싱: ⑭~⑱
# 실행: python ingest.py  →  out/*.json 저장, PostgreSQL(pgvector)에 적재하고 검색
# DB 접속 정보는 환경 변수 PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE 에서 읽어요.
#
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                      #@1
import hashlib                                                #@1
import io                                                     #@5
import json                                                   #@7
import re                                                     #@6
import unicodedata                                            #@6
from bs4 import BeautifulSoup                                 #@3
from pypdf import PdfReader                                   #@4
from PIL import Image                                         #@5
import pytesseract                                            #@5
from transformers import AutoTokenizer                        #@9
from collections import Counter                               #@17
from sentence_transformers import SentenceTransformer         #@14
import psycopg                                                #@15
from pgvector.psycopg import register_vector                  #@15

HERE = Path(__file__).resolve().parent                        #@1
RAW = HERE / "raw"                                            #@1

# ① 파일 목록과 원본 기록
manifest = []
for path in sorted(RAW.iterdir()):
    data = path.read_bytes()
    manifest.append({
        "file": path.name,
        "type": path.suffix.lower(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest()[:12],
    })

# ② 글자 파일 읽기
ENCODING = {"old_notice.txt": "cp949"}  # 열어 보고 확인한 예외만 적어요
units = []  # 읽어 낸 글: 어느 파일의 어디에서 나온 무슨 글인지

def read_text_file(m):
    data = (RAW / m["file"]).read_bytes()
    try:
        text = data.decode(ENCODING.get(m["file"], "utf-8"))
    except UnicodeDecodeError as e:
        m["status"], m["error"] = "decode_error", str(e)
        return
    units.append({"file": m["file"], "where": "전체", "raw": text})
    m["status"] = "parsed"

for m in manifest:
    if m["type"] in (".txt", ".md"):
        read_text_file(m)

# ③ 웹페이지 읽기
def read_html(m):
    soup = BeautifulSoup((RAW / m["file"]).read_text(encoding="utf-8"), "html.parser")
    main = soup.select_one("main")
    if main is None:
        m["status"] = "main_not_found"
        return
    for table in main.select("table"):
        rows = [[c.get_text(strip=True) for c in tr.select("th, td")] for tr in table.select("tr")]
        head = rows[0]
        lines = [", ".join(f"{h}: {v}" for h, v in zip(head, r)) for r in rows[1:]]
        table.replace_with("\n".join(lines))
    units.append({"file": m["file"], "where": "main", "raw": main.get_text("\n")})
    m["status"] = "parsed"

for m in manifest:
    if m["type"] == ".html":
        read_html(m)

# ④ PDF 읽기
def read_pdf(m):
    pages = [p.extract_text() or "" for p in PdfReader(RAW / m["file"]).pages]
    if not any(t.strip() for t in pages):
        m["status"] = "needs_ocr"  # 글자가 하나도 없음 → ⑤에서 OCR
        return
    for n, text in enumerate(pages, start=1):
        units.append({"file": m["file"], "where": f"{n}쪽", "raw": text})
    m["status"] = "parsed"

for m in manifest:
    if m["type"] == ".pdf":
        read_pdf(m)

# ⑤ 스캔 PDF는 OCR로
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # 이 PC의 설치 위치
OCR_CONFIG = "--tessdata-dir C:/Users/pc_28/tessdata_local --psm 6 -c preserve_interword_spaces=1"

def read_scan(m):
    for n, page in enumerate(PdfReader(RAW / m["file"]).pages, start=1):
        img = Image.open(io.BytesIO(page.images[0].data))
        text = pytesseract.image_to_string(img, lang="kor", config=OCR_CONFIG)
        units.append({"file": m["file"], "where": f"{n}쪽 (OCR)", "raw": text})
    m["status"] = "ocr"

for m in manifest:
    if m.get("status") == "needs_ocr":
        read_scan(m)

# ⑥ 잡음 정리 (Cleaning)
PAGE_NO = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")  # 줄 전체가 "1 / 2" 꼴일 때만

def clean(text):
    text = unicodedata.normalize("NFC", text)
    lines = [ln for ln in text.splitlines() if not PAGE_NO.match(ln)]
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in lines]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()

for u in units:
    u["clean"] = clean(u["raw"])

# ⑦ 상태 정리하고 저장
for m in manifest:
    m.setdefault("status", "unsupported")  # 어느 단계도 읽지 못한 형식

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"파일 {len(manifest)}개 → 글 {len(units)}개 저장")

# ⑧ 제목으로 나누기
HEADING = re.compile(r"^(#{1,6} .+|제\d+조 .+)$", re.M)  # Markdown 제목, 규정의 "제N조"

def split_sections(text):
    heads = list(HEADING.finditer(text))
    if not heads:
        return [{"heading": None, "start": 0, "end": len(text)}]  # 제목이 없으면 글 전체가 한 섹션
    secs = []
    if heads[0].start() > 0:
        secs.append({"heading": None, "start": 0, "end": heads[0].start()})  # 첫 제목 앞의 글
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        secs.append({"heading": h.group(1).lstrip("# "), "start": min(h.end() + 1, end), "end": end})
    for s in secs:
        s["end"] = s["start"] + len(text[s["start"]:s["end"]].rstrip())  # 끝의 빈 줄 빼기
    return [s for s in secs if s["end"] > s["start"]]  # 제목만 있고 본문이 없는 섹션 빼기

sections = []
for i, u in enumerate(units):
    for s in split_sections(u["clean"]):
        sections.append({"u": i, **s})  # u: 몇 번째 글에서 나왔는지

# ⑨ 토큰 세기
MODEL = "intfloat/multilingual-e5-small"  # 3쪽에서 벡터를 만들 Embedding 모델
tok = AutoTokenizer.from_pretrained(MODEL)
MAX_TOKENS = 64  # 예시 설정

def count(t):
    return len(tok(t, add_special_tokens=False)["input_ids"])

for s in sections:
    s["tokens"] = count(units[s["u"]]["clean"][s["start"]:s["end"]])

# ⑩ 긴 섹션만 줄 단위로 다시 묶기
def split_long(s):
    if s["tokens"] <= MAX_TOKENS:
        return [s]
    text = units[s["u"]]["clean"]
    pieces, cur = [], None
    for m in re.finditer(r"[^\n]+", text[s["start"]:s["end"]]):
        a, b = s["start"] + m.start(), s["start"] + m.end()
        if cur and count(text[cur["start"]:b]) <= MAX_TOKENS:
            cur["end"] = b  # 지금 조각에 이어 붙이기
        else:
            cur = {**s, "start": a, "end": b}
            pieces.append(cur)  # 새 조각 시작
    return pieces

# ⑪ 조각 기록 만들기
chunks = []
for s in sections:
    u = units[s["u"]]
    for p in split_long(s):
        body = u["clean"][p["start"]:p["end"]]
        key = json.dumps({"file": u["file"], "where": u["where"], "start": p["start"],
                          "end": p["end"], "body": body}, ensure_ascii=False, sort_keys=True)
        chunks.append({
            "chunk_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
            "file": u["file"],
            "where": u["where"],
            "heading": p["heading"],
            "char_start": p["start"],
            "char_end": p["end"],
            "tokens": count(body),
            "body": body,
        })

# ⑫ 검사하기
MIN_CHARS = 20  # 예시 설정: 이보다 짧으면 답이 될 내용이 없다고 봐요
clean_of = {(u["file"], u["where"]): u["clean"] for u in units}
kept, dropped = [], []
for c in chunks:
    assert clean_of[(c["file"], c["where"])][c["char_start"]:c["char_end"]] == c["body"]  # 위치로 원문을 되찾을 수 있나
    assert c["tokens"] <= MAX_TOKENS  # 모델 한도 안인가
    if len(c["body"]) < MIN_CHARS:
        dropped.append({**c, "reason": "too_short"})  # 버리되, 무엇을 왜 버렸는지 남겨요
    else:
        kept.append(c)
chunks = kept

# ⑬ 조각 저장
(OUT / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "dropped.json").write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"글 {len(units)}개 → 조각 {len(chunks)}개 저장 (뺀 조각 {len(dropped)}개)")

# ⑭ 벡터 만들기
model = SentenceTransformer(MODEL)  # ⑨에서 토큰을 센 그 모델
DIM = model.get_sentence_embedding_dimension()

def embed(texts, kind):
    # e5 모델은 조각 앞에 "passage: ", 질문 앞에 "query: "를 붙여 쓰도록 학습됐어요
    return model.encode([f"{kind}: {t}" for t in texts], normalize_embeddings=True)

def embed_text(c):
    return f"{c['heading'] or ''}\n{c['body']}".strip()  # 제목을 붙여 문맥을 더해요

vectors = embed([embed_text(c) for c in chunks], "passage")

# ⑮ 테이블 만들기
conn = psycopg.connect()  # 접속 정보는 환경 변수에서
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)
conn.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        file      text PRIMARY KEY,
        published int  -- 지금 검색에 보이는 판. 아직 없으면 NULL
    )""")
conn.execute(f"""
    CREATE TABLE IF NOT EXISTS chunks (
        file      text REFERENCES documents (file),
        version   int,
        chunk_id  text,
        location  text,
        heading   text,
        body      text,
        embedding vector({DIM}),
        PRIMARY KEY (file, version, chunk_id)
    )""")
conn.commit()

# ⑯ 새 판 적재하기
VERSION = 1  # 이번에 적재하는 판 번호
with conn.transaction():
    for f in sorted({c["file"] for c in chunks}):
        conn.execute("INSERT INTO documents (file) VALUES (%s) ON CONFLICT DO NOTHING", (f,))
    for c, v in zip(chunks, vectors):
        conn.execute("INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                     (c["file"], VERSION, c["chunk_id"], c["where"], c["heading"], c["body"], v))

# ⑰ 확인하고 공개하기
expected = Counter(c["file"] for c in chunks)
with conn.transaction():
    for f, n in expected.items():
        got = conn.execute("SELECT count(*) FROM chunks WHERE file = %s AND version = %s",
                           (f, VERSION)).fetchone()[0]
        if got != n:
            raise RuntimeError(f"{f}: {n}개 중 {got}개만 적재됨")  # 하나라도 모자라면 전부 취소
        conn.execute("UPDATE documents SET published = %s WHERE file = %s", (VERSION, f))

# ⑱ 검색하기
def search(question, k=3):
    return conn.execute("""
        SELECT c.file, c.location, c.heading, c.body, c.embedding <=> %s AS distance
        FROM chunks c
        JOIN documents d ON d.file = c.file AND d.published = c.version
        ORDER BY distance
        LIMIT %s""", (embed([question], "query")[0], k)).fetchall()

for row in search("대출 기간은 며칠인가요?"):
    print(row[0], row[1], round(row[4], 3), row[3].splitlines()[0])
