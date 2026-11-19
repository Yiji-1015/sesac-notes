window.PAGE = {
 "steps": {
  "14": {
   "shape": [
    12,
    384
   ],
   "dtype": "float32",
   "norm": 1.0,
   "first": [
    0.023,
    -0.0551,
    -0.042,
    -0.0805,
    0.0755,
    -0.0433
   ],
   "embed_text": "대출과 반납\n한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
   "model": "intfloat/multilingual-e5-small",
   "inputs": [
    "passage: 대출과 반납\n한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
    "query: 대출 기간은 며칠인가요?"
   ],
   "q": "대출 기간은 며칠인가요?",
   "dist": [
    {
     "file": "guide.md",
     "heading": "대출과 반납",
     "where": "전체",
     "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.",
     "cos": 0.891,
     "dist": 0.109
    },
    {
     "file": "guide.md",
     "heading": "열람실",
     "where": "전체",
     "body": "열람실은 평일 오전 9시부터 오후 10시까지 엽니다.",
     "cos": 0.791,
     "dist": 0.209
    },
    {
     "file": "guide.md",
     "heading": "분실과 훼손",
     "where": "전체",
     "body": "빌린 책을 잃어버리면 같은 책으로 갚거나 책값을 냅니다.",
     "cos": 0.748,
     "dist": 0.252
    },
    {
     "file": "guide.md",
     "heading": "분실과 훼손",
     "where": "전체",
     "body": "훼손 여부는 반납할 때 담당자가 직접 확인합니다.",
     "cos": 0.794,
     "dist": 0.206
    },
    {
     "file": "notice.txt",
     "heading": null,
     "where": "전체",
     "body": "겨울방학 운영 안내",
     "cos": 0.839,
     "dist": 0.161
    },
    {
     "file": "old_notice.txt",
     "heading": null,
     "where": "전체",
     "body": "시설 점검 안내",
     "cos": 0.787,
     "dist": 0.213
    },
    {
     "file": "hours.html",
     "heading": null,
     "where": "main",
     "body": "운영 시간",
     "cos": 0.791,
     "dist": 0.209
    },
    {
     "file": "rules.pdf",
     "heading": "제1조 회원 가입",
     "where": "1쪽",
     "body": "도서관 회원은 신분증을 확인한 뒤 가입할 수 있습니다.",
     "cos": 0.788,
     "dist": 0.212
    },
    {
     "file": "rules.pdf",
     "heading": "제2조 회원증",
     "where": "1쪽",
     "body": "회원증을 잃어버리면 안내 데스크에서 다시 발급합니다.",
     "cos": 0.753,
     "dist": 0.247
    },
    {
     "file": "rules.pdf",
     "heading": "제3조 회원 자격 정지",
     "where": "2쪽",
     "body": "연체가 30일을 넘으면 회원 자격이 정지됩니다.",
     "cos": 0.82,
     "dist": 0.18
    },
    {
     "file": "rules.pdf",
     "heading": "제4조 탈퇴",
     "where": "2쪽",
     "body": "탈퇴하려면 빌린 책을 모두 반납해야 합니다.",
     "cos": 0.762,
     "dist": 0.238
    },
    {
     "file": "scan.pdf",
     "heading": null,
     "where": "1쪽 (OCR)",
     "body": "자료실 배치 안내",
     "cos": 0.761,
     "dist": 0.239
    }
   ]
  },
  "15": {
   "cols": [
    [
     "chunks",
     "file",
     "text",
     "text"
    ],
    [
     "chunks",
     "version",
     "integer",
     "int4"
    ],
    [
     "chunks",
     "chunk_id",
     "text",
     "text"
    ],
    [
     "chunks",
     "location",
     "text",
     "text"
    ],
    [
     "chunks",
     "heading",
     "text",
     "text"
    ],
    [
     "chunks",
     "body",
     "text",
     "text"
    ],
    [
     "chunks",
     "embedding",
     "USER-DEFINED",
     "vector"
    ],
    [
     "documents",
     "file",
     "text",
     "text"
    ],
    [
     "documents",
     "published",
     "integer",
     "int4"
    ]
   ],
   "pg": "18.6",
   "pgvector": "0.8.7"
  },
  "16": {
   "chunks": 12,
   "docs": [
    [
     "guide.md",
     null
    ],
    [
     "hours.html",
     null
    ],
    [
     "notice.txt",
     null
    ],
    [
     "old_notice.txt",
     null
    ],
    [
     "rules.pdf",
     null
    ],
    [
     "scan.pdf",
     null
    ]
   ],
   "by_version": [
    [
     1,
     12
    ]
   ]
  },
  "17": {
   "docs": [
    [
     "guide.md",
     1
    ],
    [
     "hours.html",
     1
    ],
    [
     "notice.txt",
     1
    ],
    [
     "old_notice.txt",
     1
    ],
    [
     "rules.pdf",
     1
    ],
    [
     "scan.pdf",
     1
    ]
   ],
   "expected": {
    "guide.md": 4,
    "notice.txt": 1,
    "old_notice.txt": 1,
    "hours.html": 1,
    "rules.pdf": 4,
    "scan.pdf": 1
   }
  },
  "18": {
   "results": [
    {
     "q": "대출 기간은 며칠인가요?",
     "rows": [
      {
       "file": "guide.md",
       "location": "전체",
       "heading": "대출과 반납",
       "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
       "distance": 0.109
      },
      {
       "file": "notice.txt",
       "location": "전체",
       "heading": null,
       "body": "겨울방학 운영 안내\n\n겨울방학 동안 열람실은 오후 6시에 닫습니다.\n도서관 휴관일은 매주 월요일입니다.\n방학 중에도 대출과 반납은 평소와 같습니다.",
       "distance": 0.161
      },
      {
       "file": "rules.pdf",
       "location": "2쪽",
       "heading": "제3조 회원 자격 정지",
       "body": "연체가 30일을 넘으면 회원 자격이 정지됩니다.\n정지는 책을 모두 반납한 날부터 풀립니다.\n빌린 책의 1/2 이상이 연체되면 새 대출이 막힙니다.",
       "distance": 0.18
      }
     ]
    },
    {
     "q": "회원증을 잃어버렸어요",
     "rows": [
      {
       "file": "rules.pdf",
       "location": "1쪽",
       "heading": "제2조 회원증",
       "body": "회원증을 잃어버리면 안내 데스크에서 다시 발급합니다.",
       "distance": 0.114
      },
      {
       "file": "rules.pdf",
       "location": "1쪽",
       "heading": "제1조 회원 가입",
       "body": "도서관 회원은 신분증을 확인한 뒤 가입할 수 있습니다.\n가입한 날부터 바로 책을 빌릴 수 있습니다.",
       "distance": 0.166
      },
      {
       "file": "rules.pdf",
       "location": "2쪽",
       "heading": "제3조 회원 자격 정지",
       "body": "연체가 30일을 넘으면 회원 자격이 정지됩니다.\n정지는 책을 모두 반납한 날부터 풀립니다.\n빌린 책의 1/2 이상이 연체되면 새 대출이 막힙니다.",
       "distance": 0.17
      }
     ]
    },
    {
     "q": "노트북을 쓸 수 있는 자리가 있나요?",
     "rows": [
      {
       "file": "scan.pdf",
       "location": "1쪽 (OCR)",
       "heading": null,
       "body": "자료실 배치 안내\n\n1층에는 어린이 자료실과 신문 코너가 있습니다.\n\n2층에는 일반 자료실과 복사기가 있습니다.\n\n3층 열람실은 노트북을 쓸 수 있는 자리가 따로 있습니다.",
       "distance": 0.147
      },
      {
       "file": "guide.md",
       "location": "전체",
       "heading": "열람실",
       "body": "열람실은 평일 오전 9시부터 오후 10시까지 엽니다.\n자리는 도착한 순서대로 앉을 수 있으며, 예약은 받지 않습니다.\n자리를 2시간 넘게 비우면 다른 이용자가 앉을 수 있습니다.",
       "distance": 0.192
      },
      {
       "file": "rules.pdf",
       "location": "1쪽",
       "heading": "제1조 회원 가입",
       "body": "도서관 회원은 신분증을 확인한 뒤 가입할 수 있습니다.\n가입한 날부터 바로 책을 빌릴 수 있습니다.",
       "distance": 0.195
      }
     ]
    }
   ],
   "printed": [
    "guide.md 전체 0.109 한 사람은 한 번에 5권까지 빌릴 수 있습니다.",
    "notice.txt 전체 0.161 겨울방학 운영 안내",
    "rules.pdf 2쪽 0.18 연체가 30일을 넘으면 회원 자격이 정지됩니다."
   ]
  }
 },
 "exp": {
  "noprefix": [
   {
    "heading": "대출과 반납",
    "file": "guide.md",
    "with": 0.891,
    "without": 0.896
   },
   {
    "heading": "열람실",
    "file": "guide.md",
    "with": 0.791,
    "without": 0.8
   },
   {
    "heading": "분실과 훼손",
    "file": "guide.md",
    "with": 0.748,
    "without": 0.764
   },
   {
    "heading": "분실과 훼손",
    "file": "guide.md",
    "with": 0.794,
    "without": 0.813
   },
   {
    "heading": null,
    "file": "notice.txt",
    "with": 0.839,
    "without": 0.85
   },
   {
    "heading": null,
    "file": "old_notice.txt",
    "with": 0.787,
    "without": 0.803
   },
   {
    "heading": null,
    "file": "hours.html",
    "with": 0.791,
    "without": 0.803
   },
   {
    "heading": "제1조 회원 가입",
    "file": "rules.pdf",
    "with": 0.788,
    "without": 0.803
   },
   {
    "heading": "제2조 회원증",
    "file": "rules.pdf",
    "with": 0.753,
    "without": 0.773
   },
   {
    "heading": "제3조 회원 자격 정지",
    "file": "rules.pdf",
    "with": 0.82,
    "without": 0.833
   },
   {
    "heading": "제4조 탈퇴",
    "file": "rules.pdf",
    "with": 0.762,
    "without": 0.782
   },
   {
    "heading": null,
    "file": "scan.pdf",
    "with": 0.761,
    "without": 0.777
   }
  ],
  "rerun": 12,
  "noconflict": {
   "type": "UniqueViolation",
   "msg": "중복된 키 값이 \"chunks_pkey\" 고유 제약 조건을 위반함",
   "detail": "(file, version, chunk_id)=(guide.md, 1, 068063fe6b85) 키가 이미 있습니다."
  },
  "weekend": {
   "q": "주말에도 열람실을 쓸 수 있나요?",
   "rows": [
    {
     "file": "guide.md",
     "location": "전체",
     "heading": "열람실",
     "body": "열람실은 평일 오전 9시부터 오후 10시까지 엽니다.\n자리는 도착한 순서대로 앉을 수 있으며, 예약은 받지 않습니다.\n자리를 2시간 넘게 비우면 다른 이용자가 앉을 수 있습니다.",
     "distance": 0.122
    },
    {
     "file": "notice.txt",
     "location": "전체",
     "heading": null,
     "body": "겨울방학 운영 안내\n\n겨울방학 동안 열람실은 오후 6시에 닫습니다.\n도서관 휴관일은 매주 월요일입니다.\n방학 중에도 대출과 반납은 평소와 같습니다.",
     "distance": 0.145
    },
    {
     "file": "hours.html",
     "location": "main",
     "heading": null,
     "body": "운영 시간\n\n가람 도서관은 아래 시간에 문을 엽니다.\n\n구분: 자료실, 평일: 09:00~22:00, 주말: 09:00~18:00\n구분: 열람실, 평일: 09:00~22:00, 주말: 휴실",
     "distance": 0.153
    },
    {
     "file": "scan.pdf",
     "location": "1쪽 (OCR)",
     "heading": null,
     "body": "자료실 배치 안내\n\n1층에는 어린이 자료실과 신문 코너가 있습니다.\n\n2층에는 일반 자료실과 복사기가 있습니다.\n\n3층 열람실은 노트북을 쓸 수 있는 자리가 따로 있습니다.",
     "distance": 0.16
    },
    {
     "file": "rules.pdf",
     "location": "1쪽",
     "heading": "제2조 회원증",
     "body": "회원증을 잃어버리면 안내 데스크에서 다시 발급합니다.",
     "distance": 0.195
    }
   ]
  },
  "v2": {
   "staged": [
    [
     "guide.md",
     1,
     4
    ],
    [
     "guide.md",
     2,
     4
    ],
    [
     "hours.html",
     1,
     1
    ],
    [
     "hours.html",
     2,
     1
    ],
    [
     "notice.txt",
     1,
     1
    ],
    [
     "notice.txt",
     2,
     1
    ],
    [
     "old_notice.txt",
     1,
     1
    ],
    [
     "old_notice.txt",
     2,
     1
    ],
    [
     "rules.pdf",
     1,
     4
    ],
    [
     "rules.pdf",
     2,
     4
    ],
    [
     "scan.pdf",
     1,
     1
    ],
    [
     "scan.pdf",
     2,
     1
    ]
   ],
   "after_stage": [
    {
     "file": "guide.md",
     "location": "전체",
     "heading": "대출과 반납",
     "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
     "distance": 0.109
    },
    {
     "file": "notice.txt",
     "location": "전체",
     "heading": null,
     "body": "겨울방학 운영 안내\n\n겨울방학 동안 열람실은 오후 6시에 닫습니다.\n도서관 휴관일은 매주 월요일입니다.\n방학 중에도 대출과 반납은 평소와 같습니다.",
     "distance": 0.161
    },
    {
     "file": "rules.pdf",
     "location": "2쪽",
     "heading": "제3조 회원 자격 정지",
     "body": "연체가 30일을 넘으면 회원 자격이 정지됩니다.\n정지는 책을 모두 반납한 날부터 풀립니다.\n빌린 책의 1/2 이상이 연체되면 새 대출이 막힙니다.",
     "distance": 0.18
    }
   ],
   "after_publish": [
    {
     "file": "guide.md",
     "location": "전체",
     "heading": "대출과 반납",
     "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 21일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
     "distance": 0.108
    },
    {
     "file": "notice.txt",
     "location": "전체",
     "heading": null,
     "body": "겨울방학 운영 안내\n\n겨울방학 동안 열람실은 오후 6시에 닫습니다.\n도서관 휴관일은 매주 월요일입니다.\n방학 중에도 대출과 반납은 평소와 같습니다.",
     "distance": 0.161
    },
    {
     "file": "rules.pdf",
     "location": "2쪽",
     "heading": "제3조 회원 자격 정지",
     "body": "연체가 30일을 넘으면 회원 자격이 정지됩니다.\n정지는 책을 모두 반납한 날부터 풀립니다.\n빌린 책의 1/2 이상이 연체되면 새 대출이 막힙니다.",
     "distance": 0.18
    }
   ],
   "docs": [
    [
     "guide.md",
     2
    ],
    [
     "hours.html",
     2
    ],
    [
     "notice.txt",
     2
    ],
    [
     "old_notice.txt",
     2
    ],
    [
     "rules.pdf",
     2
    ],
    [
     "scan.pdf",
     2
    ]
   ]
  },
  "fail": {
   "error": "guide.md: 4개 중 3개만 적재됨",
   "docs": [
    [
     "guide.md",
     1
    ],
    [
     "hours.html",
     1
    ],
    [
     "notice.txt",
     1
    ],
    [
     "old_notice.txt",
     1
    ],
    [
     "rules.pdf",
     1
    ],
    [
     "scan.pdf",
     1
    ]
   ],
   "after": [
    {
     "file": "guide.md",
     "location": "전체",
     "heading": "대출과 반납",
     "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
     "distance": 0.109
    },
    {
     "file": "notice.txt",
     "location": "전체",
     "heading": null,
     "body": "겨울방학 운영 안내\n\n겨울방학 동안 열람실은 오후 6시에 닫습니다.\n도서관 휴관일은 매주 월요일입니다.\n방학 중에도 대출과 반납은 평소와 같습니다.",
     "distance": 0.161
    },
    {
     "file": "rules.pdf",
     "location": "2쪽",
     "heading": "제3조 회원 자격 정지",
     "body": "연체가 30일을 넘으면 회원 자격이 정지됩니다.\n정지는 책을 모두 반납한 날부터 풀립니다.\n빌린 책의 1/2 이상이 연체되면 새 대출이 막힙니다.",
     "distance": 0.18
    }
   ],
   "v2_rows": 11
  }
 },
 "why": {
  "mixed": [
   {
    "file": "guide.md",
    "version": 2,
    "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 21일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
    "distance": 0.108
   },
   {
    "file": "guide.md",
    "version": 1,
    "body": "한 사람은 한 번에 5권까지 빌릴 수 있습니다.\n대출 기간은 14일이며, 한 번 연장하면 7일이 늘어납니다.\n반납일을 넘기면 늦은 날수만큼 대출이 정지됩니다.",
    "distance": 0.109
   },
   {
    "file": "notice.txt",
    "version": 1,
    "body": "겨울방학 운영 안내\n\n겨울방학 동안 열람실은 오후 6시에 닫습니다.\n도서관 휴관일은 매주 월요일입니다.\n방학 중에도 대출과 반납은 평소와 같습니다.",
    "distance": 0.161
   }
  ]
 },
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import hashlib",
   "s": 1
  },
  {
   "t": "import io",
   "s": 5
  },
  {
   "t": "import json",
   "s": 7
  },
  {
   "t": "import re",
   "s": 6
  },
  {
   "t": "import unicodedata",
   "s": 6
  },
  {
   "t": "from bs4 import BeautifulSoup",
   "s": 3
  },
  {
   "t": "from pypdf import PdfReader",
   "s": 4
  },
  {
   "t": "from PIL import Image",
   "s": 5
  },
  {
   "t": "import pytesseract",
   "s": 5
  },
  {
   "t": "from transformers import AutoTokenizer",
   "s": 9
  },
  {
   "t": "from collections import Counter",
   "s": 17
  },
  {
   "t": "from sentence_transformers import SentenceTransformer",
   "s": 14
  },
  {
   "t": "import psycopg",
   "s": 15
  },
  {
   "t": "from pgvector.psycopg import register_vector",
   "s": 15
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "HERE = Path(__file__).resolve().parent",
   "s": 1
  },
  {
   "t": "RAW = HERE / \"raw\"",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 파일 목록과 원본 기록",
   "s": 1
  },
  {
   "t": "manifest = []",
   "s": 1
  },
  {
   "t": "for path in sorted(RAW.iterdir()):",
   "s": 1
  },
  {
   "t": "    data = path.read_bytes()",
   "s": 1
  },
  {
   "t": "    manifest.append({",
   "s": 1
  },
  {
   "t": "        \"file\": path.name,",
   "s": 1
  },
  {
   "t": "        \"type\": path.suffix.lower(),",
   "s": 1
  },
  {
   "t": "        \"bytes\": len(data),",
   "s": 1
  },
  {
   "t": "        \"sha256\": hashlib.sha256(data).hexdigest()[:12],",
   "s": 1
  },
  {
   "t": "    })",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 글자 파일 읽기",
   "s": 2
  },
  {
   "t": "ENCODING = {\"old_notice.txt\": \"cp949\"}  # 열어 보고 확인한 예외만 적어요",
   "s": 2
  },
  {
   "t": "units = []  # 읽어 낸 글: 어느 파일의 어디에서 나온 무슨 글인지",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "def read_text_file(m):",
   "s": 2
  },
  {
   "t": "    data = (RAW / m[\"file\"]).read_bytes()",
   "s": 2
  },
  {
   "t": "    try:",
   "s": 2
  },
  {
   "t": "        text = data.decode(ENCODING.get(m[\"file\"], \"utf-8\"))",
   "s": 2
  },
  {
   "t": "    except UnicodeDecodeError as e:",
   "s": 2
  },
  {
   "t": "        m[\"status\"], m[\"error\"] = \"decode_error\", str(e)",
   "s": 2
  },
  {
   "t": "        return",
   "s": 2
  },
  {
   "t": "    units.append({\"file\": m[\"file\"], \"where\": \"전체\", \"raw\": text})",
   "s": 2
  },
  {
   "t": "    m[\"status\"] = \"parsed\"",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "for m in manifest:",
   "s": 2
  },
  {
   "t": "    if m[\"type\"] in (\".txt\", \".md\"):",
   "s": 2
  },
  {
   "t": "        read_text_file(m)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 웹페이지 읽기",
   "s": 3
  },
  {
   "t": "def read_html(m):",
   "s": 3
  },
  {
   "t": "    soup = BeautifulSoup((RAW / m[\"file\"]).read_text(encoding=\"utf-8\"), \"html.parser\")",
   "s": 3
  },
  {
   "t": "    main = soup.select_one(\"main\")",
   "s": 3
  },
  {
   "t": "    if main is None:",
   "s": 3
  },
  {
   "t": "        m[\"status\"] = \"main_not_found\"",
   "s": 3
  },
  {
   "t": "        return",
   "s": 3
  },
  {
   "t": "    for table in main.select(\"table\"):",
   "s": 3
  },
  {
   "t": "        rows = [[c.get_text(strip=True) for c in tr.select(\"th, td\")] for tr in table.select(\"tr\")]",
   "s": 3
  },
  {
   "t": "        head = rows[0]",
   "s": 3
  },
  {
   "t": "        lines = [\", \".join(f\"{h}: {v}\" for h, v in zip(head, r)) for r in rows[1:]]",
   "s": 3
  },
  {
   "t": "        table.replace_with(\"\\n\".join(lines))",
   "s": 3
  },
  {
   "t": "    units.append({\"file\": m[\"file\"], \"where\": \"main\", \"raw\": main.get_text(\"\\n\")})",
   "s": 3
  },
  {
   "t": "    m[\"status\"] = \"parsed\"",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "for m in manifest:",
   "s": 3
  },
  {
   "t": "    if m[\"type\"] == \".html\":",
   "s": 3
  },
  {
   "t": "        read_html(m)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ PDF 읽기",
   "s": 4
  },
  {
   "t": "def read_pdf(m):",
   "s": 4
  },
  {
   "t": "    pages = [p.extract_text() or \"\" for p in PdfReader(RAW / m[\"file\"]).pages]",
   "s": 4
  },
  {
   "t": "    if not any(t.strip() for t in pages):",
   "s": 4
  },
  {
   "t": "        m[\"status\"] = \"needs_ocr\"  # 글자가 하나도 없음 → ⑤에서 OCR",
   "s": 4
  },
  {
   "t": "        return",
   "s": 4
  },
  {
   "t": "    for n, text in enumerate(pages, start=1):",
   "s": 4
  },
  {
   "t": "        units.append({\"file\": m[\"file\"], \"where\": f\"{n}쪽\", \"raw\": text})",
   "s": 4
  },
  {
   "t": "    m[\"status\"] = \"parsed\"",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "for m in manifest:",
   "s": 4
  },
  {
   "t": "    if m[\"type\"] == \".pdf\":",
   "s": 4
  },
  {
   "t": "        read_pdf(m)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 스캔 PDF는 OCR로",
   "s": 5
  },
  {
   "t": "pytesseract.pytesseract.tesseract_cmd = r\"C:\\Program Files\\Tesseract-OCR\\tesseract.exe\"  # 이 PC의 설치 위치",
   "s": 5
  },
  {
   "t": "OCR_CONFIG = \"--tessdata-dir C:/Users/pc_28/tessdata_local --psm 6 -c preserve_interword_spaces=1\"",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def read_scan(m):",
   "s": 5
  },
  {
   "t": "    for n, page in enumerate(PdfReader(RAW / m[\"file\"]).pages, start=1):",
   "s": 5
  },
  {
   "t": "        img = Image.open(io.BytesIO(page.images[0].data))",
   "s": 5
  },
  {
   "t": "        text = pytesseract.image_to_string(img, lang=\"kor\", config=OCR_CONFIG)",
   "s": 5
  },
  {
   "t": "        units.append({\"file\": m[\"file\"], \"where\": f\"{n}쪽 (OCR)\", \"raw\": text})",
   "s": 5
  },
  {
   "t": "    m[\"status\"] = \"ocr\"",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "for m in manifest:",
   "s": 5
  },
  {
   "t": "    if m.get(\"status\") == \"needs_ocr\":",
   "s": 5
  },
  {
   "t": "        read_scan(m)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 잡음 정리 (Cleaning)",
   "s": 6
  },
  {
   "t": "PAGE_NO = re.compile(r\"^\\s*\\d+\\s*/\\s*\\d+\\s*$\")  # 줄 전체가 \"1 / 2\" 꼴일 때만",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "def clean(text):",
   "s": 6
  },
  {
   "t": "    text = unicodedata.normalize(\"NFC\", text)",
   "s": 6
  },
  {
   "t": "    lines = [ln for ln in text.splitlines() if not PAGE_NO.match(ln)]",
   "s": 6
  },
  {
   "t": "    lines = [re.sub(r\"[ \\t]+\", \" \", ln).strip() for ln in lines]",
   "s": 6
  },
  {
   "t": "    return re.sub(r\"\\n{3,}\", \"\\n\\n\", \"\\n\".join(lines)).strip()",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "for u in units:",
   "s": 6
  },
  {
   "t": "    u[\"clean\"] = clean(u[\"raw\"])",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 상태 정리하고 저장",
   "s": 7
  },
  {
   "t": "for m in manifest:",
   "s": 7
  },
  {
   "t": "    m.setdefault(\"status\", \"unsupported\")  # 어느 단계도 읽지 못한 형식",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "OUT = HERE / \"out\"",
   "s": 7
  },
  {
   "t": "OUT.mkdir(exist_ok=True)",
   "s": 7
  },
  {
   "t": "(OUT / \"manifest.json\").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding=\"utf-8\")",
   "s": 7
  },
  {
   "t": "(OUT / \"units.json\").write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding=\"utf-8\")",
   "s": 7
  },
  {
   "t": "print(f\"파일 {len(manifest)}개 → 글 {len(units)}개 저장\")",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 제목으로 나누기",
   "s": 8
  },
  {
   "t": "HEADING = re.compile(r\"^(#{1,6} .+|제\\d+조 .+)$\", re.M)  # Markdown 제목, 규정의 \"제N조\"",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "def split_sections(text):",
   "s": 8
  },
  {
   "t": "    heads = list(HEADING.finditer(text))",
   "s": 8
  },
  {
   "t": "    if not heads:",
   "s": 8
  },
  {
   "t": "        return [{\"heading\": None, \"start\": 0, \"end\": len(text)}]  # 제목이 없으면 글 전체가 한 섹션",
   "s": 8
  },
  {
   "t": "    secs = []",
   "s": 8
  },
  {
   "t": "    if heads[0].start() > 0:",
   "s": 8
  },
  {
   "t": "        secs.append({\"heading\": None, \"start\": 0, \"end\": heads[0].start()})  # 첫 제목 앞의 글",
   "s": 8
  },
  {
   "t": "    for i, h in enumerate(heads):",
   "s": 8
  },
  {
   "t": "        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)",
   "s": 8
  },
  {
   "t": "        secs.append({\"heading\": h.group(1).lstrip(\"# \"), \"start\": min(h.end() + 1, end), \"end\": end})",
   "s": 8
  },
  {
   "t": "    for s in secs:",
   "s": 8
  },
  {
   "t": "        s[\"end\"] = s[\"start\"] + len(text[s[\"start\"]:s[\"end\"]].rstrip())  # 끝의 빈 줄 빼기",
   "s": 8
  },
  {
   "t": "    return [s for s in secs if s[\"end\"] > s[\"start\"]]  # 제목만 있고 본문이 없는 섹션 빼기",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "sections = []",
   "s": 8
  },
  {
   "t": "for i, u in enumerate(units):",
   "s": 8
  },
  {
   "t": "    for s in split_sections(u[\"clean\"]):",
   "s": 8
  },
  {
   "t": "        sections.append({\"u\": i, **s})  # u: 몇 번째 글에서 나왔는지",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "# ⑨ 토큰 세기",
   "s": 9
  },
  {
   "t": "MODEL = \"intfloat/multilingual-e5-small\"  # 3쪽에서 벡터를 만들 Embedding 모델",
   "s": 9
  },
  {
   "t": "tok = AutoTokenizer.from_pretrained(MODEL)",
   "s": 9
  },
  {
   "t": "MAX_TOKENS = 64  # 예시 설정",
   "s": 9
  },
  {
   "t": "",
   "s": 9
  },
  {
   "t": "def count(t):",
   "s": 9
  },
  {
   "t": "    return len(tok(t, add_special_tokens=False)[\"input_ids\"])",
   "s": 9
  },
  {
   "t": "",
   "s": 9
  },
  {
   "t": "for s in sections:",
   "s": 9
  },
  {
   "t": "    s[\"tokens\"] = count(units[s[\"u\"]][\"clean\"][s[\"start\"]:s[\"end\"]])",
   "s": 9
  },
  {
   "t": "",
   "s": 9
  },
  {
   "t": "# ⑩ 긴 섹션만 줄 단위로 다시 묶기",
   "s": 10
  },
  {
   "t": "def split_long(s):",
   "s": 10
  },
  {
   "t": "    if s[\"tokens\"] <= MAX_TOKENS:",
   "s": 10
  },
  {
   "t": "        return [s]",
   "s": 10
  },
  {
   "t": "    text = units[s[\"u\"]][\"clean\"]",
   "s": 10
  },
  {
   "t": "    pieces, cur = [], None",
   "s": 10
  },
  {
   "t": "    for m in re.finditer(r\"[^\\n]+\", text[s[\"start\"]:s[\"end\"]]):",
   "s": 10
  },
  {
   "t": "        a, b = s[\"start\"] + m.start(), s[\"start\"] + m.end()",
   "s": 10
  },
  {
   "t": "        if cur and count(text[cur[\"start\"]:b]) <= MAX_TOKENS:",
   "s": 10
  },
  {
   "t": "            cur[\"end\"] = b  # 지금 조각에 이어 붙이기",
   "s": 10
  },
  {
   "t": "        else:",
   "s": 10
  },
  {
   "t": "            cur = {**s, \"start\": a, \"end\": b}",
   "s": 10
  },
  {
   "t": "            pieces.append(cur)  # 새 조각 시작",
   "s": 10
  },
  {
   "t": "    return pieces",
   "s": 10
  },
  {
   "t": "",
   "s": 10
  },
  {
   "t": "# ⑪ 조각 기록 만들기",
   "s": 11
  },
  {
   "t": "chunks = []",
   "s": 11
  },
  {
   "t": "for s in sections:",
   "s": 11
  },
  {
   "t": "    u = units[s[\"u\"]]",
   "s": 11
  },
  {
   "t": "    for p in split_long(s):",
   "s": 11
  },
  {
   "t": "        body = u[\"clean\"][p[\"start\"]:p[\"end\"]]",
   "s": 11
  },
  {
   "t": "        key = json.dumps({\"file\": u[\"file\"], \"where\": u[\"where\"], \"start\": p[\"start\"],",
   "s": 11
  },
  {
   "t": "                          \"end\": p[\"end\"], \"body\": body}, ensure_ascii=False, sort_keys=True)",
   "s": 11
  },
  {
   "t": "        chunks.append({",
   "s": 11
  },
  {
   "t": "            \"chunk_id\": hashlib.sha256(key.encode(\"utf-8\")).hexdigest()[:12],",
   "s": 11
  },
  {
   "t": "            \"file\": u[\"file\"],",
   "s": 11
  },
  {
   "t": "            \"where\": u[\"where\"],",
   "s": 11
  },
  {
   "t": "            \"heading\": p[\"heading\"],",
   "s": 11
  },
  {
   "t": "            \"char_start\": p[\"start\"],",
   "s": 11
  },
  {
   "t": "            \"char_end\": p[\"end\"],",
   "s": 11
  },
  {
   "t": "            \"tokens\": count(body),",
   "s": 11
  },
  {
   "t": "            \"body\": body,",
   "s": 11
  },
  {
   "t": "        })",
   "s": 11
  },
  {
   "t": "",
   "s": 11
  },
  {
   "t": "# ⑫ 검사하기",
   "s": 12
  },
  {
   "t": "MIN_CHARS = 20  # 예시 설정: 이보다 짧으면 답이 될 내용이 없다고 봐요",
   "s": 12
  },
  {
   "t": "clean_of = {(u[\"file\"], u[\"where\"]): u[\"clean\"] for u in units}",
   "s": 12
  },
  {
   "t": "kept, dropped = [], []",
   "s": 12
  },
  {
   "t": "for c in chunks:",
   "s": 12
  },
  {
   "t": "    assert clean_of[(c[\"file\"], c[\"where\"])][c[\"char_start\"]:c[\"char_end\"]] == c[\"body\"]  # 위치로 원문을 되찾을 수 있나",
   "s": 12
  },
  {
   "t": "    assert c[\"tokens\"] <= MAX_TOKENS  # 모델 한도 안인가",
   "s": 12
  },
  {
   "t": "    if len(c[\"body\"]) < MIN_CHARS:",
   "s": 12
  },
  {
   "t": "        dropped.append({**c, \"reason\": \"too_short\"})  # 버리되, 무엇을 왜 버렸는지 남겨요",
   "s": 12
  },
  {
   "t": "    else:",
   "s": 12
  },
  {
   "t": "        kept.append(c)",
   "s": 12
  },
  {
   "t": "chunks = kept",
   "s": 12
  },
  {
   "t": "",
   "s": 12
  },
  {
   "t": "# ⑬ 조각 저장",
   "s": 13
  },
  {
   "t": "(OUT / \"chunks.json\").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding=\"utf-8\")",
   "s": 13
  },
  {
   "t": "(OUT / \"dropped.json\").write_text(json.dumps(dropped, ensure_ascii=False, indent=2), encoding=\"utf-8\")",
   "s": 13
  },
  {
   "t": "print(f\"글 {len(units)}개 → 조각 {len(chunks)}개 저장 (뺀 조각 {len(dropped)}개)\")",
   "s": 13
  },
  {
   "t": "",
   "s": 13
  },
  {
   "t": "# ⑭ 벡터 만들기",
   "s": 14
  },
  {
   "t": "model = SentenceTransformer(MODEL)  # ⑨에서 토큰을 센 그 모델",
   "s": 14
  },
  {
   "t": "DIM = model.get_sentence_embedding_dimension()",
   "s": 14
  },
  {
   "t": "",
   "s": 14
  },
  {
   "t": "def embed(texts, kind):",
   "s": 14
  },
  {
   "t": "    # e5 모델은 조각 앞에 \"passage: \", 질문 앞에 \"query: \"를 붙여 쓰도록 학습됐어요",
   "s": 14
  },
  {
   "t": "    return model.encode([f\"{kind}: {t}\" for t in texts], normalize_embeddings=True)",
   "s": 14
  },
  {
   "t": "",
   "s": 14
  },
  {
   "t": "def embed_text(c):",
   "s": 14
  },
  {
   "t": "    return f\"{c['heading'] or ''}\\n{c['body']}\".strip()  # 제목을 붙여 문맥을 더해요",
   "s": 14
  },
  {
   "t": "",
   "s": 14
  },
  {
   "t": "vectors = embed([embed_text(c) for c in chunks], \"passage\")",
   "s": 14
  },
  {
   "t": "",
   "s": 14
  },
  {
   "t": "# ⑮ 테이블 만들기",
   "s": 15
  },
  {
   "t": "conn = psycopg.connect()  # 접속 정보는 환경 변수에서",
   "s": 15
  },
  {
   "t": "conn.execute(\"CREATE EXTENSION IF NOT EXISTS vector\")",
   "s": 15
  },
  {
   "t": "register_vector(conn)",
   "s": 15
  },
  {
   "t": "conn.execute(\"\"\"",
   "s": 15
  },
  {
   "t": "    CREATE TABLE IF NOT EXISTS documents (",
   "s": 15
  },
  {
   "t": "        file      text PRIMARY KEY,",
   "s": 15
  },
  {
   "t": "        published int  -- 지금 검색에 보이는 판. 아직 없으면 NULL",
   "s": 15
  },
  {
   "t": "    )\"\"\")",
   "s": 15
  },
  {
   "t": "conn.execute(f\"\"\"",
   "s": 15
  },
  {
   "t": "    CREATE TABLE IF NOT EXISTS chunks (",
   "s": 15
  },
  {
   "t": "        file      text REFERENCES documents (file),",
   "s": 15
  },
  {
   "t": "        version   int,",
   "s": 15
  },
  {
   "t": "        chunk_id  text,",
   "s": 15
  },
  {
   "t": "        location  text,",
   "s": 15
  },
  {
   "t": "        heading   text,",
   "s": 15
  },
  {
   "t": "        body      text,",
   "s": 15
  },
  {
   "t": "        embedding vector({DIM}),",
   "s": 15
  },
  {
   "t": "        PRIMARY KEY (file, version, chunk_id)",
   "s": 15
  },
  {
   "t": "    )\"\"\")",
   "s": 15
  },
  {
   "t": "conn.commit()",
   "s": 15
  },
  {
   "t": "",
   "s": 15
  },
  {
   "t": "# ⑯ 새 판 적재하기",
   "s": 16
  },
  {
   "t": "VERSION = 1  # 이번에 적재하는 판 번호",
   "s": 16
  },
  {
   "t": "with conn.transaction():",
   "s": 16
  },
  {
   "t": "    for f in sorted({c[\"file\"] for c in chunks}):",
   "s": 16
  },
  {
   "t": "        conn.execute(\"INSERT INTO documents (file) VALUES (%s) ON CONFLICT DO NOTHING\", (f,))",
   "s": 16
  },
  {
   "t": "    for c, v in zip(chunks, vectors):",
   "s": 16
  },
  {
   "t": "        conn.execute(\"INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING\",",
   "s": 16
  },
  {
   "t": "                     (c[\"file\"], VERSION, c[\"chunk_id\"], c[\"where\"], c[\"heading\"], c[\"body\"], v))",
   "s": 16
  },
  {
   "t": "",
   "s": 16
  },
  {
   "t": "# ⑰ 확인하고 공개하기",
   "s": 17
  },
  {
   "t": "expected = Counter(c[\"file\"] for c in chunks)",
   "s": 17
  },
  {
   "t": "with conn.transaction():",
   "s": 17
  },
  {
   "t": "    for f, n in expected.items():",
   "s": 17
  },
  {
   "t": "        got = conn.execute(\"SELECT count(*) FROM chunks WHERE file = %s AND version = %s\",",
   "s": 17
  },
  {
   "t": "                           (f, VERSION)).fetchone()[0]",
   "s": 17
  },
  {
   "t": "        if got != n:",
   "s": 17
  },
  {
   "t": "            raise RuntimeError(f\"{f}: {n}개 중 {got}개만 적재됨\")  # 하나라도 모자라면 전부 취소",
   "s": 17
  },
  {
   "t": "        conn.execute(\"UPDATE documents SET published = %s WHERE file = %s\", (VERSION, f))",
   "s": 17
  },
  {
   "t": "",
   "s": 17
  },
  {
   "t": "# ⑱ 검색하기",
   "s": 18
  },
  {
   "t": "def search(question, k=3):",
   "s": 18
  },
  {
   "t": "    return conn.execute(\"\"\"",
   "s": 18
  },
  {
   "t": "        SELECT c.file, c.location, c.heading, c.body, c.embedding <=> %s AS distance",
   "s": 18
  },
  {
   "t": "        FROM chunks c",
   "s": 18
  },
  {
   "t": "        JOIN documents d ON d.file = c.file AND d.published = c.version",
   "s": 18
  },
  {
   "t": "        ORDER BY distance",
   "s": 18
  },
  {
   "t": "        LIMIT %s\"\"\", (embed([question], \"query\")[0], k)).fetchall()",
   "s": 18
  },
  {
   "t": "",
   "s": 18
  },
  {
   "t": "for row in search(\"대출 기간은 며칠인가요?\"):",
   "s": 18
  },
  {
   "t": "    print(row[0], row[1], round(row[4], 3), row[3].splitlines()[0])",
   "s": 18
  }
 ],
 "names": {
  "1": "파일 목록과 원본 기록",
  "2": "글자 파일 읽기",
  "3": "웹페이지 읽기",
  "4": "PDF 읽기",
  "5": "스캔 PDF는 OCR로",
  "6": "잡음 정리 (Cleaning)",
  "7": "상태 정리하고 저장",
  "8": "제목으로 나누기",
  "9": "토큰 세기",
  "10": "긴 섹션만 줄 단위로 다시 묶기",
  "11": "조각 기록 만들기",
  "12": "검사하기",
  "13": "조각 저장",
  "14": "벡터 만들기",
  "15": "테이블 만들기",
  "16": "새 판 적재하기",
  "17": "확인하고 공개하기",
  "18": "검색하기"
 },
 "range": [
  14,
  18
 ],
 "meta": {
  "python": "3.13.16",
  "psycopg": "3.3.8",
  "pgvector_py": "?",
  "sentence_transformers": "6.3.0"
 }
};
