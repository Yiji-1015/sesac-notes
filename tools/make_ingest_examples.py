"""문서 적재(Ingestion) 노트용 가상 문서 묶음을 만든다. 모두 직접 지어낸 「가람 도서관」 자료다.

만드는 파일 (examples/ingest/raw/):
  guide.md        Markdown, 제목 3개 (Chunking 견본과 같은 내용)
  notice.txt      UTF-8 텍스트. 한 문장은 NFD(자모 분리형)로 저장 → Cleaning 에서 NFC 사례
  old_notice.txt  CP949로 저장한 옛 공지 → UTF-8로 읽으면 실패하는 사례
  hours.html      메뉴, 스크립트, 본문, 표가 섞인 웹페이지
  rules.pdf       글자가 있는 PDF 2쪽. 쪽마다 머리말과 "1 / 2" 쪽번호
  scan.pdf        글자를 그림으로만 담은 PDF (스캔본 흉내) → OCR 사례
  floor_map.hwp   지원하지 않는 형식 (내용은 의미 없는 바이트)

실행 (PyMuPDF 가 있는 시스템 파이썬): python tools/make_ingest_examples.py
다시 실행해도 같은 파일이 나오도록 무작위 값은 seed 를 고정한다.
"""
import random
import shutil
import unicodedata
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "examples" / "ingest" / "raw"
FONT = r"C:\Windows\Fonts\malgun.ttf"

if RAW.exists():
    shutil.rmtree(RAW)
RAW.mkdir(parents=True)

# ---- guide.md (Chunking 견본과 같은 문서) ----------------------------------
shutil.copyfile(ROOT / "examples" / "chunking" / "library_guide.md", RAW / "guide.md")

# ---- notice.txt (UTF-8, 한 줄은 NFD) ---------------------------------------
nfd_line = unicodedata.normalize("NFD", "도서관 휴관일은 매주 월요일입니다.")
(RAW / "notice.txt").write_bytes(("겨울방학 운영 안내\n\n"
                                  "겨울방학 동안 열람실은 오후 6시에 닫습니다.\n"
                                  f"{nfd_line}\n"
                                  "방학 중에도 대출과 반납은 평소와 같습니다.\n").encode("utf-8"))

# ---- old_notice.txt (CP949) ------------------------------------------------
(RAW / "old_notice.txt").write_bytes(("시설 점검 안내\n\n"
                                      "3월 둘째 주에는 승강기 점검이 있습니다.\n"
                                      "점검 시간에는 계단을 이용해 주세요.\n").encode("cp949"))

# ---- hours.html ------------------------------------------------------------
(RAW / "hours.html").write_text("""<!DOCTYPE html>
<html lang="ko">
<head><meta charset="utf-8"><title>운영 시간 | 가람 도서관</title>
<script>window.dataLayer = []; function track(e) { dataLayer.push(e); }</script>
</head>
<body>
<nav><a href="/">홈</a> <a href="/books">자료 검색</a> <a href="/login">로그인</a></nav>
<main>
  <h1>운영 시간</h1>
  <p>가람 도서관은    아래 시간에 문을 엽니다.</p>
  <table>
    <tr><th>구분</th><th>평일</th><th>주말</th></tr>
    <tr><td>자료실</td><td>09:00~22:00</td><td>09:00~18:00</td></tr>
    <tr><td>열람실</td><td>09:00~22:00</td><td>휴실</td></tr>
  </table>
</main>
<footer>© 가람 도서관 | 개인정보처리방침</footer>
<script>track("view-hours");</script>
</body>
</html>
""", encoding="utf-8")

# ---- rules.pdf (글자 PDF, 2쪽) ---------------------------------------------
pages = [
    ["제1조 회원 가입", "도서관 회원은 신분증을 확인한 뒤 가입할 수 있습니다.",
     "가입한 날부터 바로 책을 빌릴 수 있습니다.",
     "제2조 회원증", "회원증을 잃어버리면 안내 데스크에서 다시 발급합니다."],
    ["제3조 회원 자격 정지", "연체가 30일을 넘으면 회원 자격이 정지됩니다.",
     "정지는 책을 모두 반납한 날부터 풀립니다.",
     "빌린 책의 1/2 이상이 연체되면 새 대출이 막힙니다.",
     "제4조 탈퇴", "탈퇴하려면 빌린 책을 모두 반납해야 합니다."],
]
doc = fitz.open()
for i, lines in enumerate(pages, start=1):
    page = doc.new_page(width=595, height=842)  # A4 (pt)
    page.insert_text((60, 50), "가람 도서관 회원 규정", fontsize=9, fontname="malgun", fontfile=FONT)
    y = 110
    for ln in lines:
        bold = ln.startswith("제")
        page.insert_text((60, y), ln, fontsize=13 if bold else 11, fontname="malgun", fontfile=FONT)
        y += 30 if bold else 24
    page.insert_text((280, 800), f"{i} / {len(pages)}", fontsize=10, fontname="malgun", fontfile=FONT)
doc.subset_fonts()  # 글꼴 전체(수 MB) 대신 쓴 글자만 넣기
doc.save(RAW / "rules.pdf", garbage=4, deflate=True)
doc.close()

# ---- scan.pdf (그림만 있는 PDF) ---------------------------------------------
random.seed(0)
W, H = 1240, 1754  # A4, 150dpi
img = Image.new("L", (W, H), 245)
draw = ImageDraw.Draw(img)
f_title = ImageFont.truetype(FONT, 46)
f_body = ImageFont.truetype(FONT, 34)
draw.text((120, 160), "자료실 배치 안내", font=f_title, fill=20)
body = ["1층에는 어린이 자료실과 신문 코너가 있습니다.",
        "2층에는 일반 자료실과 복사기가 있습니다.",
        "3층 열람실은 노트북을 쓸 수 있는 자리가 따로 있습니다."]
for k, ln in enumerate(body):
    draw.text((120, 300 + k * 70), ln, font=f_body, fill=30)
for _ in range(1500):  # 스캔 잡티
    x, y = random.randrange(W), random.randrange(H)
    draw.point((x, y), fill=random.choice([90, 140, 180]))
img = img.rotate(0.6, fillcolor=245).filter(ImageFilter.GaussianBlur(0.6))
img.convert("RGB").save(RAW / "scan.pdf", "PDF", resolution=150)

# ---- floor_map.hwp (지원하지 않는 형식) ------------------------------------
(RAW / "floor_map.hwp").write_bytes(bytes(range(256)) * 2)

for p in sorted(RAW.iterdir()):
    print(f"{p.name:16} {p.stat().st_size:>7} bytes")
