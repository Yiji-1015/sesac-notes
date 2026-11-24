"""RAG 검색 고치기 쪽(topics/backend/rag-rerank.html)이 쓰는 가람 도서관 조각을 더 만든다.

인덱싱 쪽의 조각 12개(examples/ingest/out/chunks.json)만으로는 후보 수와 재정렬의 차이가 잘 보이지 않아서,
같은 도서관의 안내 조각을 더 썼다. 내용은 이 노트용으로 지어낸 가상 자료다.
chunk_id는 인덱싱 쪽(examples/ingest/ingest.py ⑪)과 같은 방식으로 내용에서 만든다.
실행: python tools/make_rag_more.py  →  examples/rag/data/more_chunks.json
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "examples" / "rag" / "data" / "more_chunks.json"

DOCS = [  # (파일, 제목, 본문)
    ("room.md", "스터디룸 예약", "스터디룸은 2명에서 6명이 함께 쓰는 방입니다. 홈페이지에서 날짜와 시간을 골라 예약하며, 이용일 7일 전부터 신청할 수 있습니다. 한 번 예약하면 최대 3시간까지 쓸 수 있습니다."),
    ("room.md", "예약 철회", "예약한 스터디룸을 쓰지 않게 되면 이용 시작 1시간 전까지 홈페이지의 '나의 신청'에서 철회합니다. 1시간이 남지 않았다면 안내 데스크에 전화로 알려 주세요."),
    ("room.md", "노쇼", "예약하고 연락 없이 오지 않으면 노쇼로 기록됩니다. 한 달에 노쇼가 2번 쌓이면 30일 동안 스터디룸을 예약할 수 없습니다."),
    ("room.md", "이용 규칙", "스터디룸에서는 뚜껑이 있는 음료만 마실 수 있고 음식은 먹을 수 없습니다. 화이트보드와 모니터는 자유롭게 쓰고, 나갈 때 전원을 꺼 주세요."),
    ("holds.md", "대출 중인 책 예약", "다른 사람이 빌려 간 책은 예약해 둘 수 있습니다. 책이 돌아오면 문자로 알려 드리며, 3일 안에 찾아가지 않으면 예약이 자동으로 취소됩니다."),
    ("holds.md", "도서 예약 취소", "책 예약을 취소하려면 홈페이지의 '나의 예약'에서 취소 버튼을 누릅니다. 도착 알림을 받은 뒤에도 취소할 수 있습니다."),
    ("holds.md", "희망 도서", "도서관에 없는 책은 희망 도서로 신청할 수 있습니다. 한 사람이 한 달에 3권까지 신청하며, 구입이 정해지면 신청한 사람이 먼저 빌릴 수 있습니다."),
    ("holds.md", "다른 도서관 책 받아 보기", "가람 도서관에 없는 책은 같은 구의 다른 공공도서관에서 보내 받아 볼 수 있습니다. 신청한 뒤 보통 3일에서 5일 안에 도착하고, 받은 책은 14일 동안 빌릴 수 있습니다."),
    ("programs.md", "독서 프로그램 신청", "매달 첫째 주 월요일에 다음 달 독서 프로그램 신청을 받습니다. 먼저 신청한 순서로 자리가 정해지고, 정원이 차면 대기 명단에 올립니다."),
    ("programs.md", "프로그램 신청 취소", "프로그램 신청을 취소하려면 시작 전날까지 홈페이지에서 취소합니다. 연락 없이 두 번 빠지면 그 학기에는 프로그램을 신청할 수 없습니다."),
    ("devices.md", "노트북 대여", "2층 일반 자료실에서 노트북을 빌려 도서관 안에서 쓸 수 있습니다. 회원증을 맡기고 빌리며, 한 번에 4시간까지 쓸 수 있습니다."),
    ("devices.md", "복사와 출력", "2층 복사기에서 복사와 출력을 할 수 있습니다. 흑백은 한 장에 50원, 컬러는 한 장에 200원입니다."),
    ("notice_2025.txt", None, "2025년 스터디룸 운영 안내\n\n2025년에는 스터디룸을 한 사람이 하루 2시간까지 쓸 수 있습니다."),
    ("notice_2026.txt", None, "2026년 스터디룸 운영 안내\n\n2026년부터 스터디룸은 하루 최대 3시간까지 쓸 수 있습니다. 시험 기간에는 하루 2시간으로 줄입니다."),
]


def main():
    chunks = []
    for file, heading, body in DOCS:
        where = "전체"                                       # 인덱싱 쪽의 .md, .txt 조각과 같이 파일 안 위치는 '전체'
        key = json.dumps({"file": file, "where": where, "body": body}, ensure_ascii=False, sort_keys=True)
        chunks.append({"chunk_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:12],
                       "file": file, "where": where, "heading": heading, "body": body,
                       "current": file != "notice_2025.txt"})       # 2026년 공지가 나와서 2025년 공지는 지금 유효한 판이 아님
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(chunks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(chunks)} chunks -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
