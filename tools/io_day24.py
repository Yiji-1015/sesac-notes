"""Day24 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다 (Document Parsing · Chunking · Indexing).

- 입력 문서: C:/practice_assignment/20261119/mentee/data 의 예제 원문 6건 + versions/attendance_v2.md (읽기만 함).
- pathlib · hashlib · unicodedata · re · json · markdown-it-py · pypdf 를 실제로 실행한다.
  (Beautiful Soup 은 이 PC 가상환경에 없어 실행하지 않았다 → 해당 항목에는 결과 상자 없음)
- DB: 환경변수 NOTES_PG_ENV 의 .env (비밀번호는 출력하지 않음)로 노트 전용 DB(sesac_notes_lab)를 만들어 쓰고 지운다.
  PostgreSQL 18.6 + pgvector 0.8.7. 벡터는 E5 출력이 아니라 임의 단위 벡터(384차원).
- E5 모델은 내려받지 않았다. '글자 수 ≠ 토큰 수'는 이 PC에 캐시된 bert-base-multilingual-cased Tokenizer 로만 확인.
결과는 cmds/day24.html 각 항목 첫 코드 블록 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: NOTES_PG_ENV=<.env 경로> C:/practice_assignment/.venv/Scripts/python.exe tools/io_day24.py
"""
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:/practice_assignment/20261119/mentee/data")
RAW = DATA / "raw"
IO, RESULTS = {}, {}
LAB = "sesac_notes_lab"


def err(fn):
    try:
        return repr(fn())
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {str(e).strip().splitlines()[0]}"


# ------------------------------------------------------------------ ① 수집
allowed = {".txt", ".md", ".html", ".pdf"}
all_files = sorted(p for p in RAW.iterdir() if p.is_file())
files = [p for p in all_files if p.suffix.lower() in allowed]
IO["d24-collect"] = ("예제 원문 폴더에서 실제 실행", [
    ("sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed)",
     ", ".join(f"{p.name}({p.stat().st_size})" for p in files), "파일명(바이트) — 이름순 정렬"),
    ("허용 확장자 밖의 파일", ", ".join(p.name for p in all_files if p not in files), "목록에서 빠짐 → Manifest에 unsupported 로 따로 기록"),
])
RESULTS["files"] = [(p.name, p.stat().st_size) for p in all_files]

raw = (RAW / "assignment.txt").read_bytes()
h1 = hashlib.sha256(raw).hexdigest()
h2 = hashlib.sha256(raw + b" ").hexdigest()
IO["d24-manifest"] = ("assignment.txt 로 실제 실행", [
    ("len(raw)", str(len(raw)), "size_bytes — 글자 수가 아니라 바이트 수"),
    ("sha256(raw).hexdigest()[:16]", h1[:16] + "…", "raw_sha256 (64자리 16진수)"),
    ("끝에 공백 1바이트 추가 후 sha256", h2[:16] + "…", "1바이트만 달라도 완전히 다른 값 → 원문 변경 감지"),
    ("path.as_uri()", (RAW / "assignment.txt").as_uri().replace("C:/practice_assignment/20261119/mentee", "…"), "source_uri — file:// 형식"),
])

# ------------------------------------------------------------------ ② Parsing
txt = raw.decode("utf-8")
bad = "과제".encode("utf-8")[:-1] + b" ok"
IO["d24-decode"] = ("실제 실행", [
    ("assignment.txt bytes.decode('utf-8')", f"parsed · {len(txt)}자 / {len(raw)}바이트", "한글 1자 = UTF-8 3바이트"),
    ("잘린 바이트 .decode('utf-8')", err(lambda: bad.decode("utf-8")), "decode_error 로 기록"),
    ("같은 바이트 .decode('utf-8', errors='ignore')", repr(bad.decode("utf-8", errors="ignore")), "깨진 글자가 조용히 사라짐 — 누락을 숨김"),
    ("b''.decode('utf-8').strip()", repr(b"".decode("utf-8").strip()), "오류는 아니지만 본문 없음 → empty"),
])

from markdown_it import MarkdownIt  # noqa: E402

md_text = (RAW / "attendance.md").read_text(encoding="utf-8")
tokens = MarkdownIt().parse(md_text)
path, sections = [], []
for i, t in enumerate(tokens):
    if t.type == "heading_open":
        level = int(t.tag[1])
        title = tokens[i + 1].content
        path = path[: level - 1] + [title]
        sections.append(" > ".join(path))
fence = MarkdownIt().parse("```\n# 주석이지 제목 아님\n```\n# 진짜 제목\n")
IO["d24-markdown"] = ("markdown-it-py 실제 실행 · attendance.md", [
    ("[t.type for t in tokens][:4]", str([t.type for t in tokens][:4]), "제목은 heading_open → inline → heading_close"),
    ("heading 마다 Heading Path", " | ".join(sections), "절의 소속을 Metadata 로"),
    ("코드 블록 안의 '# …'", str([(t.type, t.tag) for t in fence if t.type in ("fence", "heading_open")]), "fence 안의 # 는 제목이 아님"),
])
RESULTS["md_sections"] = sections

from pypdf import PdfReader  # noqa: E402

pdf_rows = []
with warnings.catch_warnings(record=True):
    import logging
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    for name in ("career.pdf", "scan.pdf"):
        r = PdfReader(str(RAW / name))
        for n, page in enumerate(r.pages, start=1):
            t = page.extract_text() or ""
            first = t.strip().splitlines()[0] if t.strip() else ""
            pdf_rows.append((f"{name} page {n}", f"{len(t)}자 · 첫 줄 {first!r}" if t.strip() else "0자 (빈 문자열)",
                             "텍스트층 있음 → page-N unit" if t.strip() else "글자가 이미지 → needs_ocr 로 보류"))
pdf_rows.append(("career.pdf 페이지 끝의 '1 / 2'", "추출 텍스트에 그대로 포함", "머리말·쪽번호도 본문처럼 섞임 → 정제 · 검수 대상"))
IO["d24-pdf"] = ("pypdf 실제 실행 · 예제 PDF 2건", pdf_rows)

# ------------------------------------------------------------------ ③ Cleaning
def clean_prose(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


nfd = unicodedata.normalize("NFD", "한")
IO["d24-normalize"] = ("unicodedata 실제 실행", [
    ("len(NFD('한')), len(NFC(NFD('한')))", f"{len(nfd)}, {len(unicodedata.normalize('NFC', nfd))}", "자모 분리형 3글자 → 완성형 1글자 (macOS 파일명 등에서 흔함)"),
    ("NFD('한') == '한'", str(nfd == "한"), "눈엔 같아도 문자열 비교는 다름 → 검색 · 해시가 어긋남"),
    ("NFC('① x²')", repr(unicodedata.normalize("NFC", "① x²")), "NFC 는 모양을 유지"),
    ("NFKC('① x²')", repr(unicodedata.normalize("NFKC", "① x²")), "NFKC 는 호환 문자를 바꿈 — 번호 · 위첨자 의미가 사라질 수 있음"),
    ("'가나다 라마바'[4:7] · len · len(encode)", f"{'가나다 라마바'[4:7]!r} · {len('가나다 라마바')} · {len('가나다 라마바'.encode('utf-8'))}", "Offset 은 문자 위치, 바이트 위치가 아님"),
])
c1 = clean_prose("출석   안내\r\n\r\n10분  초과")
IO["d24-clean"] = ("실제 실행", [
    ("re.sub(r'[ \\t]+', ' ', …) + replace('\\r\\n', '\\n') + NFC", repr(c1), "연속 공백 1칸 · 줄바꿈 통일 · 빈 줄(문단 경계)은 남김"),
    ("re.sub(r'\\s+', ' ', 같은 입력)", repr(re.sub(r"\s+", " ", "출석   안내\r\n\r\n10분  초과")), "\\s 는 줄바꿈까지 지워 문단이 사라짐"),
])
raw_md = md_text
clean_md = clean_prose(raw_md)
ch = hashlib.sha256(clean_md.encode("utf-8")).hexdigest()
IO["d24-clean-record"] = ("attendance.md 로 실제 실행", [
    ("chars_before, chars_after", f"{len(raw_md)}, {len(clean_md)}", "끝 줄바꿈 등 형식 잡음만 줄어듦"),
    ("clean_hash[:16]", ch[:16] + "…", "정제 결과가 바뀌었는지 비교용"),
    ("clean_prose(clean_text) == clean_text", str(clean_prose(clean_md) == clean_md), "멱등성 — 두 번 적용해도 같음 (의미 보존 증명은 아님)"),
    ("'10분' in clean_text and '초과' in clean_text", str("10분" in clean_md and "초과" in clean_md), "보존할 구절 검사"),
])
RESULTS["clean_md"] = clean_md

# ------------------------------------------------------------------ ④ Chunking
def windows(text, size, overlap):
    if not (size > 0 and 0 <= overlap < size):
        raise ValueError("invalid chunk window")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield start, end, text[start:end]
        if end == len(text):
            break
        start = end - overlap


w71 = [(a, b) for a, b, _ in windows("a" * 71, 40, 10)]
md_w = list(windows(clean_md, 40, 10))
IO["d24-windows"] = ("실제 실행", [
    ("windows('a'*71, 40, 10)", str(w71), "보폭 30 · 끝(71)에 닿으면 종료"),
    ("저장 길이 합 · 고유 범위", f"{sum(b - a for a, b in w71)} · 71", "Overlap 만큼 중복 저장"),
    ("windows(attendance clean_text, 40, 10) 개수", f"{len(md_w)}개 (전체 {len(clean_md)}자)", ""),
    ("두 번째 조각", repr(md_w[1][2]), "고정 글자 창은 문장 · 조건 중간을 자름"),
    ("windows(text, 40, 40)", err(lambda: list(windows("abc", 40, 40))), "overlap ≥ size 면 영원히 못 나아감 → 거부"),
])
RESULTS["md_windows"] = [(a, b) for a, b, _ in md_w]

from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
s1 = "10분을 초과하면 결석으로 기록합니다."
ids_plain = tok(s1, add_special_tokens=True, truncation=False)["input_ids"]
ids_pref = tok("passage: " + s1, add_special_tokens=True, truncation=False)["input_ids"]
ids_head = tok("passage: 출석 안내 / 지각과 결석\n" + s1, add_special_tokens=True, truncation=False)["input_ids"]
IO["d24-tokens"] = ("실제 실행 · 이 PC에 캐시된 bert-base-multilingual-cased Tokenizer (E5 아님 · 숫자는 E5와 다름)", [
    ("len(chunk_text)", str(len(s1)), "Python 글자 수"),
    ("len(tokenizer(chunk_text)['input_ids'])", str(len(ids_plain)), "토큰 수 (특수 토큰 [CLS] · [SEP] 포함) — 글자 수와 다름"),
    ("'passage: ' + chunk_text", str(len(ids_pref)), "접두어도 토큰을 차지"),
    ("'passage: ' + Heading + chunk_text", str(len(ids_head)), "최종 입력 전체로 한도를 검사해야 하는 이유"),
    ("tokenizer.model_max_length", str(tok.model_max_length), "이 Tokenizer 의 상한 (E5-small 은 모델 카드상 512)"),
])

# ------------------------------------------------------------------ ⑤ 검증
def coverage(length, spans):
    if length == 0:
        return None
    merged = []
    for start, end in sorted(spans):
        if not (0 <= start < end <= length):
            raise ValueError("invalid_span")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return sum(e - s for s, e in merged) / length, merged


cov_full = coverage(20, [(0, 8), (6, 14), (12, 20)])
cov_gap = coverage(20, [(0, 8), (12, 20)])
IO["d24-coverage"] = ("실제 실행", [
    ("coverage(20, [(0,8),(6,14),(12,20)])", f"{cov_full[0]} · 병합 {cov_full[1]}", "길이 합 24 지만 고유 위치는 20 → 100%"),
    ("coverage(20, [(0,8),(12,20)])", f"{cov_gap[0]} · 병합 {cov_gap[1]}", "Missing Span [8,12)"),
    ("coverage(20, [(5,25)])", err(lambda: coverage(20, [(5, 25)])), "범위를 벗어난 Span"),
    ("coverage(len(clean_md), windows 결과)", str(coverage(len(clean_md), RESULTS["md_windows"])[0]), "Character Window 는 구조상 빈틈 없음"),
])
a, b = 22, 40
IO["d24-span"] = ("attendance clean_text 로 실제 실행", [
    (f"clean_text[{a}:{b}]", repr(clean_md[a:b]), "Span 이 가리키는 원문"),
    ("chunk['body'] == clean_text[a:b]  (같은 길이 다른 글자)", str("x" * (b - a) == clean_md[a:b]), "길이가 같아도 본문 불일치 → span_mismatch"),
    ("0 <= a < b <= len(clean_text)", str(0 <= a < b <= len(clean_md)), "범위 검사"),
    ("type(True) is int", str(type(True) is int), "type() is int 로 bool 을 걸러냄 (isinstance(True, int) 는 True)"),
])

# ------------------------------------------------------------------ ⑥ Metadata · ID
identity = {"doc_id": "DOC-ATTEND", "source_version": "v1", "pipeline_version": "ingest-e5-v1", "unit_id": "sec-1", "span": [0, 40]}
payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
cid = hashlib.sha256(payload.encode("utf-8")).hexdigest()
identity2 = dict(identity, source_version="v2")
cid2 = hashlib.sha256(json.dumps(identity2, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
hs = [subprocess.run([sys.executable, "-c", "print(hash('DOC-ATTEND'))"], capture_output=True, text=True).stdout.strip() for _ in range(2)]
IO["d24-chunk-id"] = ("실제 실행", [
    ("json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)", payload, "키 순서 · 공백이 고정된 직렬화"),
    ("sha256(payload).hexdigest()[:16]", cid[:16] + "…", "같은 입력 → 언제 돌려도 같은 chunk_id"),
    ("source_version 만 v2 로", cid2[:16] + "…", "버전이 다르면 다른 ID"),
    ("두 프로세스에서 hash('DOC-ATTEND')", " / ".join(hs), "Python 내장 hash() 는 실행마다 달라짐 → 저장용 ID 로 못 씀"),
])

# ------------------------------------------------------------------ 실물 사례: 그대로 파싱 → 문제 → 정제
CASES = {}

# 사례 A — 줄바꿈 · 공백이 섞인 텍스트 (예제 fixture)
fx = (DATA / "fixtures" / "clean_before.txt").read_bytes().decode("utf-8")
CASES["A"] = {"raw": fx, "clean": clean_prose(fx), "naive": re.sub(r"\s+", " ", fx).strip()}

# 사례 B — HTML: 전체 텍스트 vs main 만 vs 표를 행 단위로
from bs4 import BeautifulSoup  # noqa: E402
import bs4 as _bs4  # noqa: E402

html_text = (RAW / "facility.html").read_text(encoding="utf-8")
soup_all = BeautifulSoup(html_text, "html.parser")
whole = soup_all.get_text("\n", strip=True)
soup = BeautifulSoup(html_text, "html.parser")
main = soup.select_one("main")
for tag in main.select("script, style"):
    tag.decompose()
main_text = main.get_text("\n", strip=True)
table = main.select_one("table")
header = [th.get_text(strip=True) for th in table.select("thead th")]
rows = [[td.get_text(strip=True) for td in tr.select("td")] for tr in table.select("tbody tr")]
row_lines = [f"{r[0]}: {r[1]}" for r in rows]
table.replace_with(soup.new_string("\n".join(row_lines)))
main_rows = main.get_text("\n", strip=True)
CASES["B"] = {"whole": whole, "main": main_text, "rows": main_rows, "header": header}
IO["d24-html"] = (f"facility.html 로 실제 실행 · beautifulsoup4 {_bs4.__version__}", [
    ("BeautifulSoup(html).get_text('\\n', strip=True)  # 전체", repr(whole[:60]) + " …", "<title> 과 <h1> 이 겹쳐 '시설 안내' 두 번 · 메뉴 · 꼬리말까지 섞임"),
    ("'menu marker' in 전체 텍스트  # <script> 안의 글자", str("menu marker" in whole), "bs4 4.x 의 get_text() 는 script · style 안 글자를 기본으로 뺌 — 그래도 decompose() 로 명시적으로 지우는 편이 안전"),
    ("전체 텍스트 마지막 줄", repr(whole.splitlines()[-1]), "<footer> 꼬리말"),
    ("main 선택 + script/style decompose()", repr(main_text), "본문만 — 하지만 표는 칸이 줄마다 흩어짐"),
    ("표를 행마다 '열1: 열2' 로 바꾼 뒤", repr(main_rows), "어느 시설의 안내인지 한 줄로 묶임"),
])

# 사례 C — 텍스트 PDF: 쪽번호 줄이 본문에 섞임
pages = []
import logging  # noqa: E402
logging.getLogger("pypdf").setLevel(logging.ERROR)
for n, page in enumerate(PdfReader(str(RAW / "career.pdf")).pages, start=1):
    pages.append(page.extract_text() or "")
PAGE_NO = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
cleaned_pages = ["\n".join(ln for ln in p.splitlines() if not PAGE_NO.match(ln)).strip() for p in pages]
CASES["C"] = {"raw": pages, "clean": cleaned_pages}

# 사례 D — 스캔 PDF → OCR
import cv2  # noqa: E402
import pytesseract  # noqa: E402
from pytesseract import Output  # noqa: E402

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESS = "--tessdata-dir C:/Users/pc_28/tessdata_local --psm 6"
PIS = " -c preserve_interword_spaces=1"
TRUTH = "스캔된 안내문 (이미지 PDF)\n이 문서는 텍스트층이 없습니다.\nOCR 필요 여부는 실제 원본을 확인합니다."
scan_img = PdfReader(str(RAW / "scan.pdf")).pages[0].images[0].image
clean_scan = cv2.cvtColor(np.array(scan_img), cv2.COLOR_RGB2GRAY)[:560]
# 저화질 복사본 흉내: 1/3 축소 · 1.8° 기울임 · 회색 종이 · 잡음 · 흐림 (seed 0)
_r = np.random.default_rng(0)
h_, w_ = clean_scan.shape
small = cv2.resize(clean_scan, (w_ // 3, h_ // 3), interpolation=cv2.INTER_AREA)
Mrot = cv2.getRotationMatrix2D((small.shape[1] / 2, small.shape[0] / 2), 1.8, 1.0)
rot = cv2.warpAffine(small, Mrot, (small.shape[1], small.shape[0]), borderValue=255)
bad_scan = cv2.GaussianBlur(np.clip(rot.astype(np.float32) * 0.75 + 40 + _r.normal(0, 22, rot.shape), 0, 255).astype(np.uint8), (3, 3), 0)
up = cv2.resize(bad_scan, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
_, otsu = cv2.threshold(cv2.fastNlMeansDenoising(up, None, h=25), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ys, xs = np.where(otsu < 128)
skew = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))[-1]


def ocr(img, lang, cfg=""):
    return pytesseract.image_to_string(img, lang=lang, config=TESS + cfg).strip()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(t):
    return round(lev(t, TRUTH) / len(TRUTH), 3)


runs = [("깨끗한 스캔", "kor", "", clean_scan), ("깨끗한 스캔", "kor", PIS, clean_scan), ("깨끗한 스캔", "kor+eng", PIS, clean_scan),
        ("흐린 스캔", "kor", PIS, bad_scan), ("흐린 스캔", "kor+eng", PIS, bad_scan),
        ("흐린 스캔 → 확대·잡음제거·이진화", "kor+eng", PIS, otsu)]
ocr_rows = []
for name, lang, cfg, img in runs:
    t = ocr(img, lang, cfg)
    ocr_rows.append({"img": name, "lang": lang, "pis": bool(cfg), "text": t, "cer": cer(t)})
raw_kor = ocr_rows[0]["text"]
naive_join = re.sub(r"(?<=[가-힣]) (?=[가-힣])", "", raw_kor)


def low_conf(img, lang="kor", th=60):
    d = pytesseract.image_to_data(img, lang=lang, config=TESS + PIS, output_type=Output.DICT)
    words = [(t, float(c), (x, y, w, h)) for t, c, x, y, w, h in zip(d["text"], d["conf"], d["left"], d["top"], d["width"], d["height"]) if t.strip()]
    return words, [(t, round(c)) for t, c, _ in words if c < th]


clean_words, clean_low = low_conf(clean_scan)
bad_words, bad_low = low_conf(bad_scan)
CASES["D"] = {"truth": TRUTH, "runs": ocr_rows, "naive_join": naive_join, "naive_cer": cer(naive_join),
              "skew": round(float(skew), 2), "clean_low": clean_low, "bad_low": bad_low,
              "bad_words": [(t, round(c)) for t, c, _ in bad_words]}
RESULTS["cases"] = CASES
tmpd = Path(tempfile.gettempdir())
cv2.imwrite(str(tmpd / "d24_scan_clean.png"), clean_scan)
cv2.imwrite(str(tmpd / "d24_scan_bad.png"), bad_scan)
(tmpd / "d24_bad_boxes.json").write_text(json.dumps([(t, c, b) for t, c, b in bad_words]), encoding="utf-8")

IO["d24-ocr"] = (f"Tesseract {pytesseract.get_tesseract_version()} · kor 언어팩 · 예제 scan.pdf 의 이미지로 실제 실행", [
    ("PdfReader(scan.pdf).pages[0].extract_text()", repr(PdfReader(str(RAW / 'scan.pdf')).pages[0].extract_text()), "텍스트층 없음 → 이미지를 꺼내 OCR"),
    ("page.images[0].image.size", str(scan_img.size), "PDF 안에 든 스캔 이미지"),
    *[(f"image_to_string({r['img']}, lang='{r['lang']}'{', preserve_interword_spaces=1' if r['pis'] else ''})",
       repr(r["text"].replace("\n", " / ")), f"CER {r['cer']}") for r in ocr_rows],
    ("image_to_data(…)['conf'] < 60  # 흐린 스캔 · kor", str(bad_low), "신뢰도 낮은 글자 = 검토 후보"),
])
IO["d24-ocr-clean"] = ("위 OCR 결과로 실제 실행", [
    ("re.sub(r'(?<=[가-힣]) (?=[가-힣])', '', 기본 OCR 결과)", repr(naive_join.replace("\n", " / ")), f"글자 사이 공백을 정규식으로 지우면 진짜 띄어쓰기까지 사라짐 · CER {cer(naive_join)}"),
    ("minAreaRect 로 잰 기울기", f"{round(float(skew), 2)}°", "일부러 1.8° 돌린 사본 — 각도는 잡아냄"),
    ("흐린 스캔 단어별 conf (일부)", str([(t, round(c)) for t, c, _ in bad_words][9:21]), "'『' 81 처럼 틀렸는데 신뢰도가 높은 경우도 있음"),
])

# ------------------------------------------------------------------ 전체 파이프라인 실물 (문서 6건)
PV = "ingest-e5-v1"
MANIFEST = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))


def parse_doc(doc):
    """예제 문서를 형식별로 읽어 [(unit_id, locator, raw_text)] 로. 실패는 상태로."""
    path = RAW / doc["file"]
    suf = path.suffix.lower()
    if suf not in allowed:
        return "unsupported", []
    if suf in (".txt", ".md"):
        return "parsed", [("doc", doc["file"], path.read_bytes().decode("utf-8"))]
    if suf == ".html":
        sp = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        mn = sp.select_one("main")
        for tg in mn.select("script, style"):
            tg.decompose()
        tb = mn.select_one("table")
        tb.replace_with(sp.new_string("\n".join(f"{tr.select('td')[0].get_text(strip=True)}: {tr.select('td')[1].get_text(strip=True)}"
                                                for tr in tb.select("tbody tr"))))
        return "parsed", [("main", doc["file"] + "#main", mn.get_text("\n", strip=True))]
    units = []
    for n, pg in enumerate(PdfReader(str(path)).pages, start=1):
        t = pg.extract_text() or ""
        t = "\n".join(ln for ln in t.splitlines() if not PAGE_NO.match(ln))
        units.append((f"page-{n}", f"{doc['file']}#page={n}", t))
    if all(not u[2].strip() for u in units):
        return "needs_ocr", []
    return "parsed", units


def struct_spans(text):
    """구조 기준 분할: Markdown 은 ## 절 본문, 그 외는 빈 줄로 나뉜 문단. 제목 줄은 Span 에 넣지 않고 heading 으로."""
    spans, pos, head, top = [], 0, None, None
    blocks = []
    for line in text.split("\n"):
        start = pos
        pos += len(line) + 1
        if line.startswith("# "):
            top, head = line[2:], line[2:]
            blocks.append(("H", start, start + len(line), head))
        elif line.startswith("## "):
            head = f"{top} > {line[3:]}" if top else line[3:]
            blocks.append(("H", start, start + len(line), head))
        elif not line.strip():
            blocks.append(("B", start, start, head))
        else:
            blocks.append(("T", start, start + len(line), head))
    cur = None
    for kind, s0, e0, hd in blocks + [("B", len(text), len(text), None)]:
        if kind == "T":
            if cur and cur[2] == hd:
                cur[1] = e0
            else:
                if cur:
                    spans.append(tuple(cur))
                cur = [s0, e0, hd]
        else:
            if cur:
                spans.append(tuple(cur))
                cur = None
            if kind == "H":
                spans.append(("HEAD", e0, hd))
    with_body = {hd for s0, e0, hd in spans if s0 != "HEAD"}
    # 본문 없이 바로 다음 제목이 오는 절(예: '# 출석 안내') → 빈 Span 하나 (검증기가 잡아야 할 실제 경우)
    spans = [(e0, e0, hd) if s0 == "HEAD" else (s0, e0, hd) for s0, e0, hd in spans if s0 != "HEAD" or hd not in with_body]
    out, seen = [], set()
    for s0, e0, hd in spans:
        key = (s0, e0)
        if key not in seen:
            seen.add(key)
            out.append((s0, e0, hd))
    return out


def chunk_id_of(doc_id, ver, unit_id, s0, e0):
    ident = {"doc_id": doc_id, "source_version": ver, "pipeline_version": PV, "unit_id": unit_id, "span": [s0, e0]}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate(chunk, clean_units):
    key = (chunk["doc_id"], chunk["unit_id"])
    ct = clean_units.get(key)
    if ct is None:
        return "missing_source"
    a_, b_ = chunk["char_start"], chunk["char_end"]
    if type(a_) is not int or type(b_) is not int or not (0 <= a_ < b_ <= len(ct)):
        return "invalid span" if a_ != b_ else "empty_chunk"
    if chunk["body"] != ct[a_:b_]:
        return "span_mismatch"
    if not chunk["body"].strip():
        return "empty_chunk"
    return None


report, clean_units, struct_chunks, window_chunks = [], {}, [], []
for doc in MANIFEST:
    status, units = parse_doc(doc)
    row = {"doc_id": doc["doc_id"], "file": doc["file"], "status": status, "units": len(units), "chunks": 0}
    for unit_id, loc, raw_t in units:
        ct = clean_prose(raw_t)
        clean_units[(doc["doc_id"], unit_id)] = ct
        for s0, e0, hd in struct_spans(ct):
            struct_chunks.append(dict(doc_id=doc["doc_id"], unit_id=unit_id, category=doc["category"], source_version=doc["source_version"],
                                      source_locator=loc, heading=hd, char_start=s0, char_end=e0, body=ct[s0:e0]))
        for s0, e0, body in windows(ct, 110, 20):
            window_chunks.append(dict(doc_id=doc["doc_id"], unit_id=unit_id, char_start=s0, char_end=e0, body=body))
    report.append(row)

# 구조 분할 결과 검증 (빈 절 → empty_chunk 를 실제로 잡는지)
struct_results = [(c, validate(c, clean_units)) for c in struct_chunks]
passed = [c for c, e in struct_results if e is None]
quarantined = [(c["doc_id"], c["unit_id"], c["heading"], e) for c, e in struct_results if e]
for c in passed:
    c["chunk_id"] = chunk_id_of(c["doc_id"], c["source_version"], c["unit_id"], c["char_start"], c["char_end"])
    c["embedding_text"] = "passage: " + (c["heading"].replace(" > ", " / ") + "\n" if c["heading"] else "") + c["body"]
    c["tokens"] = len(tok(c["embedding_text"], add_special_tokens=True, truncation=False)["input_ids"])
for r in report:
    r["chunks"] = sum(1 for c in passed if c["doc_id"] == r["doc_id"])

# 질문별로 '답에 필요한 두 구절이 한 조각에 같이 있나' (모델 없이 확인하는 경계 검사)
QUESTIONS = [("15분 늦으면 어떻게 기록돼요?", "10분을 초과하면", "결석으로 기록"),
             ("사전 승인을 받았으면요?", "사전 승인", "운영실에 확인"),
             ("회의실은 어떻게 이용해요?", "회의실", "예약 승인 후 이용"),
             ("상담 일정을 바꾸려면?", "변경하려면", "미리 연락")]


def answerable(chunks):
    return [any(p1 in c["body"] and p2 in c["body"] for c in chunks) for _, p1, p2 in QUESTIONS]


ans_window = answerable(window_chunks)
ans_by_size = {}
for sz, ovl in ((40, 10), (60, 10), (110, 20)):
    wc = [dict(body=b_) for (d_, u_), ct in clean_units.items() for _, _, b_ in windows(ct, sz, ovl)]
    ans_by_size[f"{sz}/{ovl}"] = (answerable(wc), len(wc))
ans_struct = answerable(passed)
win_attend = [(c["char_start"], c["char_end"], c["body"]) for c in window_chunks if c["doc_id"] == "DOC-ATTEND"]
ov_cmp = []
for ov in (0, 10, 20):
    ws = list(windows(clean_md, 40, ov))
    ov_cmp.append((ov, len(ws), sum(b - a for a, b, _ in ws), any("10분을 초과하면" in t and "결석으로 기록" in t for _, _, t in ws)))

# 일부러 넣는 버그 3가지 → 검증기가 잡는지
fx_raw = CASES["A"]["raw"]
fx_clean = CASES["A"]["clean"]
k_ = fx_raw.index("10분")
bug_span = dict(doc_id="FX", unit_id="doc", char_start=k_, char_end=k_ + 9, body=fx_raw[k_:k_ + 9])
bug_units = {("FX", "doc"): fx_clean}
dup_ids = [c["chunk_id"] for c in passed] * 2
bug_rows = [
    ("raw_text 에서 잰 위치를 clean_text 에 적용", f"body={bug_span['body']!r} vs clean[{k_}:{k_ + 9}]={fx_clean[k_:k_ + 9]!r}", validate(bug_span, bug_units)),
    ("unit_id 오타 (page-3)", "DOC-CAREER / page-3", validate(dict(passed[0], doc_id="DOC-CAREER", unit_id="page-3"), clean_units)),
    ("같은 입력을 두 번 적재", f"{len(dup_ids)}개 중 고유 {len(set(dup_ids))}개", "duplicate_chunk_id" if len(dup_ids) != len(set(dup_ids)) else None),
]

# Coverage (구조 분할) — 무엇이 빠졌나
cov_rows = []
for (doc_id, unit_id), ct in clean_units.items():
    sp = [(c["char_start"], c["char_end"]) for c in passed if c["doc_id"] == doc_id and c["unit_id"] == unit_id]
    ratio, merged = coverage(len(ct), sp)
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    if merged and merged[0][0] > 0:
        gaps = [(0, merged[0][0])] + gaps
    if merged and merged[-1][1] < len(ct):
        gaps.append((merged[-1][1], len(ct)))
    missing_txt = [ct[a_:b_].strip() for a_, b_ in gaps if ct[a_:b_].strip()]
    cov_rows.append((doc_id, unit_id, round(ratio, 3), missing_txt))

# 토큰 한도 초과 (예시 한도 40 · 다른 모델 Tokenizer)
long_txt = (DATA / "fixtures" / "chunk_100.txt").read_text(encoding="utf-8").strip()
LIMIT = 40
long_tokens = len(tok("passage: " + long_txt, add_special_tokens=True, truncation=False)["input_ids"])
resplit = []
for s0, e0, body in windows(long_txt, 30, 0):
    resplit.append((s0, e0, len(tok("passage: " + body, add_special_tokens=True, truncation=False)["input_ids"])))

RESULTS["pipe"] = {
    "report": report, "quarantined": quarantined,
    "passed": [{k: v for k, v in c.items() if k != "embedding_text"} | {"embedding_text": c["embedding_text"]} for c in passed],
    "window_attend": win_attend, "ans_window": ans_window, "ans_struct": ans_struct,
    "questions": [q for q, _, _ in QUESTIONS], "ov_cmp": ov_cmp, "bug_rows": bug_rows, "cov_rows": cov_rows,
    "long": {"len": len(long_txt), "tokens": long_tokens, "limit": LIMIT, "resplit": resplit},
    "n_window": len(window_chunks), "ans_by_size": ans_by_size,
}
IO["d24-pipeline"] = ("예제 문서 6건으로 실제 실행 · 구조 분할 · 토큰 수는 bert-base-multilingual-cased (E5 아님)", [
    *[(f"{r['doc_id']} ({r['file']})", f"{r['status']} · unit {r['units']} · 통과 Chunk {r['chunks']}", "") for r in report],
    ("격리된 조각", str([(d, u, h, e) for d, u, h, e in quarantined]), "제목 바로 뒤 본문 없는 절 → empty_chunk"),
    *[(f"고정 창 {k} · 조각 {n}개", f"{sum(v)}/4 · {v}", "질문 4개 중 답 두 구절이 한 조각에") for k, (v, n) in ans_by_size.items()],
    ("구조 분할 · 조각 " + str(len(passed)) + "개", f"{sum(ans_struct)}/4 · {ans_struct}", ""),
])

# ------------------------------------------------------------------ ⑦ DB
def read_env(p):
    out = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


CRED = read_env(os.environ["NOTES_PG_ENV"])
SECRET = CRED["DB_PASSWORD"]


def scrub(s):
    return str(s).replace(SECRET, "********") if SECRET else str(s)


def connect(db, **kw):
    return psycopg.connect(host=CRED.get("DB_HOST", "localhost"), port=int(CRED.get("DB_PORT", 5432)),
                           dbname=db, user=CRED["DB_USER"], password=SECRET, **kw)


def derr(fn):
    try:
        return repr(fn())
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {scrub(str(e)).strip().splitlines()[0]}"


SCHEMA = """
CREATE TABLE ingest_documents (
  doc_id text PRIMARY KEY, source_uri text NOT NULL,
  current_version text, current_pipeline_version text);
CREATE TABLE ingest_versions (
  doc_id text NOT NULL REFERENCES ingest_documents(doc_id),
  source_version text NOT NULL, pipeline_version text NOT NULL,
  status text NOT NULL CHECK (status IN ('staging','ready','failed')),
  expected_count integer NOT NULL CHECK (expected_count >= 1),
  PRIMARY KEY (doc_id, source_version, pipeline_version));
CREATE TABLE ingest_chunks (
  citation_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chunk_id text NOT NULL UNIQUE, doc_id text NOT NULL,
  source_version text NOT NULL, pipeline_version text NOT NULL,
  unit_id text NOT NULL, category text NOT NULL, source_locator text NOT NULL,
  char_start integer NOT NULL, char_end integer NOT NULL,
  body text NOT NULL CHECK (length(body) > 0),
  embedding vector(384) NOT NULL, metadata jsonb NOT NULL,
  CHECK (char_start >= 0 AND char_end > char_start),
  FOREIGN KEY (doc_id, source_version, pipeline_version)
    REFERENCES ingest_versions (doc_id, source_version, pipeline_version));
CREATE INDEX ingest_chunks_category_idx ON ingest_chunks (category);
"""
PV = "ingest-e5-v1"
rng = np.random.default_rng(24)


def unit_vec():
    v = rng.normal(size=384).astype(np.float32)
    return v / np.linalg.norm(v)


def sections_of(md):
    """# 제목 단위로 나눈 (unit_id, heading_path, body) — 예제 데이터를 절 단위 Chunk 로 만들기 위한 준비."""
    out, cur, head, top = [], [], None, None
    for line in md.split("\n"):
        if line.startswith("# "):
            top = line[2:]
        elif line.startswith("## "):
            if head and "\n".join(cur).strip():
                out.append((head, "\n".join(cur).strip()))
            head, cur = f"{top} > {line[3:]}", []
        else:
            cur.append(line)
    if head and "\n".join(cur).strip():
        out.append((head, "\n".join(cur).strip()))
    return out


def make_chunks(doc_id, version, text, category):
    rows = []
    for i, (hp, body) in enumerate(sections_of(text), start=1):
        start = text.index(body)
        ident = {"doc_id": doc_id, "source_version": version, "pipeline_version": PV, "unit_id": f"sec-{i}", "span": [start, start + len(body)]}
        cid_ = hashlib.sha256(json.dumps(ident, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        rows.append(dict(chunk_id=cid_, doc_id=doc_id, source_version=version, pipeline_version=PV, unit_id=f"sec-{i}",
                         category=category, source_locator=f"attendance.md#{hp.split(' > ')[-1].replace(' ', '-')}",
                         char_start=start, char_end=start + len(body), body=body, embedding=unit_vec(),
                         metadata=json.dumps({"heading_path": hp, "cleaner_version": "clean-v1"}, ensure_ascii=False)))
    return rows


INS = """INSERT INTO ingest_chunks (chunk_id, doc_id, source_version, pipeline_version, unit_id, category,
           source_locator, char_start, char_end, body, embedding, metadata)
         VALUES (%(chunk_id)s, %(doc_id)s, %(source_version)s, %(pipeline_version)s, %(unit_id)s, %(category)s,
                 %(source_locator)s, %(char_start)s, %(char_end)s, %(body)s, %(embedding)s, %(metadata)s)
         ON CONFLICT (chunk_id) DO NOTHING"""


def stage(conn, doc_id, version, text, expected=None):
    chunks = make_chunks(doc_id, version, text, "학사")
    with conn.transaction():
        conn.execute("""INSERT INTO ingest_versions VALUES (%s,%s,%s,'staging',%s)
                        ON CONFLICT (doc_id, source_version, pipeline_version) DO NOTHING""",
                     (doc_id, version, PV, expected or len(chunks)))
        n = 0
        for c in chunks:
            n += conn.execute(INS, c).rowcount
    return chunks, n


def publish(conn, doc_id, version):
    with conn.transaction():
        exp = conn.execute("""SELECT expected_count FROM ingest_versions
                              WHERE doc_id=%s AND source_version=%s AND pipeline_version=%s FOR UPDATE""",
                           (doc_id, version, PV)).fetchone()[0]
        got = conn.execute("SELECT count(*) FROM ingest_chunks WHERE doc_id=%s AND source_version=%s AND pipeline_version=%s",
                           (doc_id, version, PV)).fetchone()[0]
        if got != exp:
            raise ValueError(f"incomplete_version: expected {exp}, got {got}")
        conn.execute("UPDATE ingest_versions SET status='ready' WHERE doc_id=%s AND source_version=%s AND pipeline_version=%s",
                     (doc_id, version, PV))
        conn.execute("UPDATE ingest_documents SET current_version=%s, current_pipeline_version=%s WHERE doc_id=%s",
                     (version, PV, doc_id))


SEARCH = """SELECT c.citation_id, c.source_version, c.unit_id, c.source_locator, c.embedding <=> %s AS d
FROM ingest_chunks AS c
JOIN ingest_documents AS d ON d.doc_id = c.doc_id
JOIN ingest_versions AS v ON v.doc_id = c.doc_id AND v.source_version = c.source_version AND v.pipeline_version = c.pipeline_version
WHERE c.source_version = d.current_version AND c.pipeline_version = d.current_pipeline_version
  AND v.status = 'ready' AND c.category = %s
ORDER BY c.embedding <=> %s ASC, c.chunk_id ASC
LIMIT %s"""

with connect("postgres", autocommit=True) as ad:
    ad.execute(f'DROP DATABASE IF EXISTS "{LAB}" WITH (FORCE)')
    ad.execute(f'CREATE DATABASE "{LAB}"')
try:
    with connect(LAB, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute(SCHEMA)
        tables = [r[0] for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY 1").fetchall()]
        conn.execute("INSERT INTO ingest_documents (doc_id, source_uri) VALUES ('DOC-ATTEND', 'file:///…/attendance.md')")
        bad_status = derr(lambda: conn.execute("INSERT INTO ingest_versions VALUES ('DOC-ATTEND','v0','p','done',1)"))
        bad_span = derr(lambda: conn.execute(INS, dict(make_chunks("DOC-ATTEND", "v1", clean_md, "학사")[0], char_start=5, char_end=5)))
        IO["d24-schema"] = ("노트 전용 DB에서 실제 실행", [
            ("information_schema.tables", ", ".join(tables), "문서 · 버전 · 청크 세 테이블"),
            ("status = 'done' 으로 INSERT", bad_status, "CHECK 제약이 허용 상태만 받음"),
            ("char_start = char_end = 5 로 INSERT", bad_span, "빈 Span 은 DB 단계에서도 거부"),
        ])

        v1_chunks, n1 = stage(conn, "DOC-ATTEND", "v1", clean_md)
        _, n1b = stage(conn, "DOC-ATTEND", "v1", clean_md)
        cnt = conn.execute("SELECT count(*) FROM ingest_chunks").fetchone()[0]
        cids = [r[0] for r in conn.execute("SELECT citation_id FROM ingest_chunks ORDER BY citation_id").fetchall()]
        meta = conn.execute("SELECT metadata->>'heading_path', jsonb_typeof(metadata) FROM ingest_chunks ORDER BY citation_id LIMIT 1").fetchone()
        IO["d24-stage"] = ("노트 전용 DB에서 실제 실행 · attendance.md v1 · 절 단위 Chunk · 벡터는 임의 단위 벡터", [
            ("v1 첫 적재 · 삽입된 행 수", str(n1), "## 절 두 개 = Chunk 2개"),
            ("같은 입력으로 다시 실행 · 삽입된 행 수", str(n1b), "같은 chunk_id → ON CONFLICT DO NOTHING · 늘지 않음 (멱등)"),
            ("SELECT count(*) · citation_id", f"{cnt} · {cids}", "정수 citation_id 는 DB 가 붙임 — 앞의 실패한 INSERT 도 번호를 써서 1이 비어 있음"),
            ("metadata->>'heading_path' · jsonb_typeof", f"{meta[0]!r} · {meta[1]}", "JSONB 안의 값을 SQL 로 꺼냄"),
        ])

        before = conn.execute("SELECT current_version FROM ingest_documents").fetchone()[0]
        publish(conn, "DOC-ATTEND", "v1")
        st1 = conn.execute("SELECT current_version, current_pipeline_version FROM ingest_documents").fetchone()
        v2_text = clean_prose((DATA / "versions" / "attendance_v2.md").read_text(encoding="utf-8"))
        stage(conn, "DOC-ATTEND", "v2", v2_text, expected=3)          # 일부러 expected_count 를 실제(2)와 다르게
        fail = derr(lambda: publish(conn, "DOC-ATTEND", "v2"))
        st_fail = conn.execute("SELECT current_version FROM ingest_documents").fetchone()[0]
        v2_status = conn.execute("SELECT status FROM ingest_versions WHERE source_version='v2'").fetchone()[0]
        conn.execute("UPDATE ingest_versions SET expected_count = 2 WHERE source_version = 'v2'")
        publish(conn, "DOC-ATTEND", "v2")
        st2 = conn.execute("SELECT current_version FROM ingest_documents").fetchone()[0]
        statuses = conn.execute("SELECT source_version, status FROM ingest_versions ORDER BY 1").fetchall()
        IO["d24-publish"] = ("노트 전용 DB에서 실제 실행 · v1 → v2 공개 전환", [
            ("공개 전 current_version", repr(before), "staging 만 됐을 뿐 아직 검색 대상 아님"),
            ("v1 publish (FOR UPDATE · 개수 확인 · ready · 포인터)", str(st1), "한 트랜잭션으로 공개"),
            ("v2 publish · expected_count 3 인데 실제 2", fail, "검증 실패 → 트랜잭션 ROLLBACK"),
            ("실패 직후 current_version · v2 status", f"{st_fail!r} · {v2_status!r}", "기존 v1 공개 유지 · v2 는 staging 그대로"),
            ("개수 맞춘 뒤 v2 publish → current_version", repr(st2), "전환 완료"),
            ("ingest_versions 상태", str(statuses), "v1 행 · Chunk 는 지우지 않고 남음"),
        ])

        q = unit_vec()
        hits = conn.execute(SEARCH, (q, "학사", q, 3)).fetchall()
        allrows = conn.execute("SELECT source_version, count(*) FROM ingest_chunks GROUP BY 1 ORDER BY 1").fetchall()
        none_cat = conn.execute(SEARCH, (q, "시설", q, 3)).fetchall()
        body15 = conn.execute("SELECT count(*) FROM ingest_chunks c JOIN ingest_documents d ON d.doc_id=c.doc_id AND c.source_version=d.current_version WHERE c.body LIKE '%15분%'").fetchone()[0]
        IO["d24-search"] = ("노트 전용 DB에서 실제 실행 · 질의 벡터는 임의 (모델 출력 아님 → 거리 값은 의미 없음)", [
            ("테이블 전체 Chunk (버전별)", str(allrows), "v1 · v2 Chunk 가 모두 저장돼 있음"),
            ("현재 공개 + ready + category='학사', k=3", str([(c, v, u) for c, v, u, _, _ in hits]), "v2 만 나옴 — 옛 v1 은 WHERE 에서 빠짐 · ON CONFLICT 로 건너뛴 재실행도 번호(4·5)를 써서 6·7"),
            ("category='시설'", str(none_cat), "정상 0건"),
            ("현재 공개 본문 중 '15분' 포함", f"{body15}건", "개정 내용(10분 → 15분)이 검색 대상"),
        ])
        RESULTS["publish"] = {"v1": st1[0], "fail": fail, "after_fail": st_fail, "v2": st2}

        # 문서 4건의 실제 Chunk 전체를 적재 · 공개 → 필터 순서 비교 (벡터는 임의)
        conn.execute("TRUNCATE ingest_chunks, ingest_versions, ingest_documents")
        by_doc = {}
        for c in passed:
            by_doc.setdefault(c["doc_id"], []).append(c)
        for doc_id, cs in by_doc.items():
            conn.execute("INSERT INTO ingest_documents (doc_id, source_uri) VALUES (%s, %s)", (doc_id, cs[0]["source_locator"]))
            with conn.transaction():
                conn.execute("INSERT INTO ingest_versions VALUES (%s,'v1',%s,'staging',%s)", (doc_id, PV, len(cs)))
                for c in cs:
                    conn.execute(INS, dict(chunk_id=c["chunk_id"], doc_id=doc_id, source_version="v1", pipeline_version=PV,
                                           unit_id=c["unit_id"], category=c["category"], source_locator=c["source_locator"],
                                           char_start=c["char_start"], char_end=c["char_end"], body=c["body"], embedding=unit_vec(),
                                           metadata=json.dumps({"heading_path": c["heading"]}, ensure_ascii=False)))
            publish(conn, doc_id, "v1")
        cats = conn.execute("SELECT category, count(*) FROM ingest_chunks GROUP BY 1 ORDER BY 1").fetchall()
        lost, trials = 0, 100
        for _ in range(trials):
            q_ = unit_vec()
            then_ = conn.execute("""SELECT count(*) FROM (SELECT c.category FROM ingest_chunks c
                                    ORDER BY c.embedding <=> %s LIMIT 3) t WHERE category = '시설'""", (q_,)).fetchone()[0]
            first_ = len(conn.execute(SEARCH, (q_, "시설", q_, 3)).fetchall())
            lost += then_ < first_
        qv = unit_vec()
        first_filter = conn.execute(SEARCH, (qv, "시설", qv, 3)).fetchall()
        topk_then = conn.execute("""SELECT * FROM (SELECT c.citation_id, c.category, c.embedding <=> %s AS d FROM ingest_chunks c
                                    ORDER BY c.embedding <=> %s LIMIT 3) t WHERE category = '시설'""", (qv, qv)).fetchall()
        top3_cats = [r[1] for r in conn.execute("SELECT citation_id, category FROM ingest_chunks ORDER BY embedding <=> %s LIMIT 3", (qv,)).fetchall()]
        sample = conn.execute("""SELECT citation_id, doc_id, unit_id, source_locator, char_start, char_end, metadata->>'heading_path', left(body, 30)
                                 FROM ingest_chunks WHERE doc_id='DOC-CAREER' ORDER BY citation_id LIMIT 1""").fetchone()
        IO["d24-filter-order"] = ("노트 전용 DB에서 실제 실행 · 예제 4건의 실제 Chunk · 벡터는 임의", [
            ("SELECT category, count(*) … GROUP BY category", str(cats), "카테고리별 조각 수"),
            ("전체 Top-3 의 category", str(top3_cats), "가까운 3개에 '시설' 조각이 없을 수 있음"),
            ("Top-3 를 먼저 뽑고 WHERE category='시설'", str([r[0] for r in topk_then]), "후보가 사라짐"),
            ("WHERE category='시설' 먼저, 그 안에서 Top-3", str([r[0] for r in first_filter]), "조건에 맞는 조각 중 가까운 것"),
            ("DOC-CAREER 조각 한 행", str(sample), "citation_id · 출처 · 위치 · 제목 경로 · 본문"),
            (f"임의 질의 {trials}개에서 'Top-3 먼저 → 필터'가 '필터 먼저'보다 적게 나온 횟수", f"{lost}/{trials}", "시설 조각 1개가 전체 7개 중 3위 안에 못 든 경우"),
        ])
        RESULTS["pipe"]["db"] = {"cats": cats, "top3": top3_cats, "then": [r[0] for r in topk_then], "first": [r[0] for r in first_filter], "sample": sample, "lost": lost, "trials": trials}
finally:
    with connect("postgres", autocommit=True) as ad:
        ad.execute(f'DROP DATABASE IF EXISTS "{LAB}" WITH (FORCE)')
        left = ad.execute("SELECT count(*) FROM pg_database WHERE datname LIKE 'sesac_notes%'").fetchone()[0]
    print("note DB dropped · remaining:", left)


def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


if __name__ == "__main__":
    for _cid in ("d24-pipeline", "d24-filter-order"):
        _v = IO.pop(_cid, None)
        if _v:
            RESULTS.setdefault("io_extra", {})[_cid] = _v   # 명령어 사전 항목이 아니라 노트 본문에 쓰는 결과
    for cid_, (title, rows) in IO.items():
        print(f"== {cid_}  [{title}]")
        for i, o, n in rows:
            print(f"   {i}\n      → {o}   ({n})")
    (Path(tempfile.gettempdir()) / "d24_results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, default=str), encoding="utf-8")
    frag = ROOT / "cmds" / "day24.html"
    if frag.exists():
        s = frag.read_text(encoding="utf-8")
        s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
        missing = []
        for cid_, (title, rows) in IO.items():
            m = re.search(rf'<section class="cmd" id="{re.escape(cid_)}".*?</section>', s, re.S)
            if not m:
                missing.append(cid_)
                continue
            sec = m.group(0)
            kk = sec.find("</code></pre>\n")
            kk = kk + len("</code></pre>\n") if kk >= 0 else sec.find("</h3>\n") + len("</h3>\n")
            s = s[:m.start() + kk] + box(title, rows) + "\n" + s[m.start() + kk:]
        frag.write_text(s, encoding="utf-8")
        print("inserted", len(IO) - len(missing), "missing", missing)
