window.PAGE = {
 "code": [
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "from sentence_transformers import SentenceTransformer",
   "s": 1
  },
  {
   "t": "import psycopg",
   "s": 5
  },
  {
   "t": "from pgvector.psycopg import register_vector",
   "s": 5
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 문장을 벡터로",
   "s": 1
  },
  {
   "t": "KNOWLEDGE = [",
   "s": 1
  },
  {
   "t": "    (101, \"실습 파일 보관\", \"실습 파일은 개인 폴더에 보관합니다.\", \"course\"),",
   "s": 1
  },
  {
   "t": "    (102, \"서버 실행\", \"프로젝트 폴더에서 서버를 실행합니다.\", \"development\"),",
   "s": 1
  },
  {
   "t": "    (103, \"질문 작성\", \"오류 메시지와 시도한 작업을 함께 적습니다.\", \"support\"),",
   "s": 1
  },
  {
   "t": "    (104, \"수업 녹화\", \"녹화가 제공되는 수업은 강의실 안내 페이지에서 확인합니다.\", \"course\"),",
   "s": 1
  },
  {
   "t": "    (105, \"과제 제출\", \"과제 제출 위치는 해당 과제의 안내 페이지에서 확인합니다.\", \"course\"),",
   "s": 1
  },
  {
   "t": "    (106, \"데이터베이스 연결\", \"PostgreSQL 접속 정보는 로컬 설정 파일에서 읽습니다.\", \"development\"),",
   "s": 1
  },
  {
   "t": "    (107, \"패키지 설치\", \"필요한 Python 패키지는 기존 가상환경에 추가합니다.\", \"development\"),",
   "s": 1
  },
  {
   "t": "    (108, \"문제 해결 안내\", \"문제를 보고할 때 재현한 단계와 오류 메시지를 기록합니다.\", \"support\"),",
   "s": 1
  },
  {
   "t": "    (109, \"결과 검토\", \"검색된 자료는 원문과 실제 질문의 요구를 대조합니다.\", \"support\"),",
   "s": 1
  },
  {
   "t": "]",
   "s": 1
  },
  {
   "t": "model = SentenceTransformer(\"intfloat/multilingual-e5-small\")",
   "s": 1
  },
  {
   "t": "docs = model.encode([f\"passage: {t} {c}\" for _, t, c, _ in KNOWLEDGE], normalize_embeddings=True)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 비슷한 정도 재기",
   "s": 2
  },
  {
   "t": "question = \"개발용 웹 서비스는 어디서 돌리나요?\"",
   "s": 2
  },
  {
   "t": "q = model.encode([f\"query: {question}\"], normalize_embeddings=True)[0]",
   "s": 2
  },
  {
   "t": "dot = docs @ q  # 내적",
   "s": 2
  },
  {
   "t": "cos = dot / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))  # 코사인 유사도",
   "s": 2
  },
  {
   "t": "l2 = np.linalg.norm(docs - q, axis=1)  # 유클리드 거리 (작을수록 가까움)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 가장 가까운 k개 (Top-k)",
   "s": 3
  },
  {
   "t": "def top_k(scores, k=3):",
   "s": 3
  },
  {
   "t": "    order = np.argsort(-scores)[:k]  # 점수가 큰 순서의 위치",
   "s": 3
  },
  {
   "t": "    return [(KNOWLEDGE[i][0], KNOWLEDGE[i][1], round(float(scores[i]), 3)) for i in order]",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "best = top_k(cos)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 조건으로 먼저 거르기",
   "s": 4
  },
  {
   "t": "def search(question, category=None, k=3):",
   "s": 4
  },
  {
   "t": "    qv = model.encode([f\"query: {question}\"], normalize_embeddings=True)[0]",
   "s": 4
  },
  {
   "t": "    scores = docs @ qv",
   "s": 4
  },
  {
   "t": "    allowed = np.array([category is None or row[3] == category for row in KNOWLEDGE])",
   "s": 4
  },
  {
   "t": "    scores = np.where(allowed, scores, -np.inf)  # 조건 밖은 절대 뽑히지 않게",
   "s": 4
  },
  {
   "t": "    return [r for r in top_k(scores, k) if r[2] != float(\"-inf\")]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "filtered = search(\"설정은 어디서 확인하나요?\", category=\"development\")",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ DB로 옮기기 (pgvector)",
   "s": 5
  },
  {
   "t": "conn = psycopg.connect()",
   "s": 5
  },
  {
   "t": "conn.execute(\"CREATE EXTENSION IF NOT EXISTS vector\")",
   "s": 5
  },
  {
   "t": "register_vector(conn)",
   "s": 5
  },
  {
   "t": "conn.execute(\"CREATE TABLE knowledge (id int PRIMARY KEY, title text, category text, embedding vector(384))\")",
   "s": 5
  },
  {
   "t": "for (i, t, _, cat), v in zip(KNOWLEDGE, docs):",
   "s": 5
  },
  {
   "t": "    conn.execute(\"INSERT INTO knowledge VALUES (%s, %s, %s, %s)\", (i, t, cat, v))",
   "s": 5
  },
  {
   "t": "db_best = conn.execute(\"\"\"",
   "s": 5
  },
  {
   "t": "    SELECT id, title, 1 - (embedding <=> %s) AS cos",
   "s": 5
  },
  {
   "t": "    FROM knowledge ORDER BY embedding <=> %s LIMIT 3\"\"\", (q, q)).fetchall()",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 많아지면: 근사 인덱스(HNSW)",
   "s": 6
  },
  {
   "t": "conn.execute(\"CREATE INDEX ON knowledge USING hnsw (embedding vector_cosine_ops)\")",
   "s": 6
  },
  {
   "t": "conn.commit()",
   "s": 6
  }
 ],
 "names": {
  "1": "문장을 벡터로",
  "2": "비슷한 정도 재기",
  "3": "가장 가까운 k개 (Top-k)",
  "4": "조건으로 먼저 거르기",
  "5": "DB로 옮기기 (pgvector)",
  "6": "많아지면: 근사 인덱스(HNSW)"
 },
 "range": [
  1,
  6
 ],
 "outs": {
  "why": {
   "title": "질문 \"개발용 웹 서비스는 어디서 돌리나요?\"",
   "cols": [
    "id",
    "자료",
    "질문의 단어가 그대로 있나",
    "Embedding 유사도"
   ],
   "rows": [
    [
     102,
     "서버 실행: 프로젝트 폴더에서 서버를 실행합니다.",
     "없음",
     "<b>0.827</b>"
    ],
    [
     105,
     "과제 제출: 과제 제출 위치는 해당 과제의 안내 페이지에서 확인합니다.",
     "없음",
     0.806
    ],
    [
     106,
     "데이터베이스 연결: PostgreSQL 접속 정보는 로컬 설정 파일에서 읽습니다.",
     "없음",
     0.802
    ],
    [
     103,
     "질문 작성: 오류 메시지와 시도한 작업을 함께 적습니다.",
     "없음",
     0.799
    ],
    [
     107,
     "패키지 설치: 필요한 Python 패키지는 기존 가상환경에 추가합니다.",
     "없음",
     0.791
    ],
    [
     108,
     "문제 해결 안내: 문제를 보고할 때 재현한 단계와 오류 메시지를 기록합니다.",
     "없음",
     0.788
    ],
    [
     109,
     "결과 검토: 검색된 자료는 원문과 실제 질문의 요구를 대조합니다.",
     "없음",
     0.786
    ],
    [
     101,
     "실습 파일 보관: 실습 파일은 개인 폴더에 보관합니다.",
     "없음",
     0.785
    ],
    [
     104,
     "수업 녹화: 녹화가 제공되는 수업은 강의실 안내 페이지에서 확인합니다.",
     "없음",
     0.772
    ]
   ],
   "notes": [
    "질문 단어 ['개발용', '웹', '서비스는', '어디서', '돌리나요'] 중 자료에 그대로 나오는 게 없어요. 글자로 찾으면 결과가 0개예요.",
    "Embedding은 \"개발용 웹 서비스를 돌린다\"와 \"서버를 실행한다\"가 비슷한 뜻이라는 걸 반영해서, 102번을 1위로 찾았어요."
   ]
  },
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    [
     "docs.shape → (9, 384)",
     1
    ],
    "docs[1] 앞 6칸 → [0.0517, -0.0422, -0.0052, -0.0323, 0.1018, -0.0247]",
    [
     "docs[1]의 길이 → 1.0  (정규화 안 하면 1.0)",
     2
    ]
   ],
   "notes": [
    "자료 9개가 각각 384칸 벡터가 됐어요. 문장 길이와 상관없이 언제나 384칸이에요.",
    "<code>normalize_embeddings=True</code>로 길이를 1로 맞췄어요. ②에서 이게 왜 편한지 봐요."
   ]
  },
  "t1": {
   "title": "같은 글에 접두어만 다르게",
   "lines": [
    "cos('query: 서버 실행', 'passage: 서버 실행') → 0.936",
    "cos('query: 서버 실행', '서버 실행') → 0.988",
    [
     "질문에 query: 를 빼고 검색한 상위 3개 → ['서버 실행', '과제 제출', '질문 작성']",
     1
    ]
   ],
   "notes": [
    "글이 같아도 접두어가 다르면 다른 벡터예요. e5는 질문과 자료를 구분해서 학습해서, 규칙대로 붙여야 제 성능이 나와요. 이 예제에서는 순위가 크게 바뀌지 않았지만, 자료가 많아지면 차이가 커질 수 있어요."
   ]
  },
  "s2": {
   "title": "질문과 자료의 세 가지 거리 (상위 4개)",
   "cols": [
    "id",
    "자료",
    "내적",
    "코사인 유사도",
    "L2 거리"
   ],
   "rows": [
    [
     102,
     "서버 실행",
     0.827,
     0.827,
     0.587
    ],
    [
     105,
     "과제 제출",
     0.806,
     0.806,
     0.623
    ],
    [
     106,
     "데이터베이스 연결",
     0.802,
     0.802,
     0.629
    ],
    [
     103,
     "질문 작성",
     0.799,
     0.799,
     0.634
    ]
   ],
   "notes": [
    "길이가 1인 벡터끼리는 내적과 코사인 유사도가 <b>똑같아요</b>. 그래서 정규화해 두면 계산이 가장 싼 내적만 써도 돼요.",
    "L2 거리는 작을수록 가까워서 방향이 반대지만, 순위는 같아요 (길이 1이면 L2² = 2 − 2 × 코사인)."
   ]
  },
  "t2": {
   "title": "정규화하지 않은 벡터로 내적 순위를 매기면",
   "lines": [
    "내적 상위 3개   → ['서버 실행', '과제 제출', '데이터베이스 연결']",
    "코사인 상위 3개 → ['서버 실행', '과제 제출', '데이터베이스 연결']",
    "자료 벡터 길이 범위 → 1.0 ~ 1.0"
   ],
   "notes": [
    "이 모델은 정규화 옵션을 끄더라도 길이가 이미 1로 나와서 순위가 같아요. 하지만 모델에 따라 길이가 제각각인 벡터를 내기도 해요. 내적은 \"방향\"뿐 아니라 \"길이\"에도 끌려가서, 길이가 1이라는 보장이 없으면 코사인을 쓰거나 직접 정규화해요."
   ]
  },
  "s3": {
   "title": "top_k(cos, 3)",
   "lines": [
    [
     "[(102, '서버 실행', 0.827), (105, '과제 제출', 0.806), (106, '데이터베이스 연결', 0.802)]",
     1
    ]
   ],
   "notes": [
    "(id, 제목, 점수). 점수가 큰 순서예요. 1위와 2위의 차이가 크지 않은 것도 볼 수 있어요. 점수가 \"맞다, 틀리다\"를 뜻하진 않아요."
   ]
  },
  "s4": {
   "title": "search(\"설정은 어디서 확인하나요?\")",
   "lines": [
    "조건 없이 → [(108, '문제 해결 안내', 0.839), (106, '데이터베이스 연결', 0.837), (105, '과제 제출', 0.828)]",
    [
     "category=\"development\" → [(106, '데이터베이스 연결', 0.837), (102, '서버 실행', 0.823), (107, '패키지 설치', 0.801)]",
     1
    ],
    [
     "(비교) 먼저 3개 뽑고 나서 거르면 → [(106, '데이터베이스 연결', 0.837)]",
     2,
     "bad"
    ]
   ],
   "notes": [
    "조건에 맞는 자료 안에서 3개를 골랐어요.",
    "순서를 바꿔 \"상위 3개를 뽑은 뒤 거르면\" 1개만 남아요. 조건으로 <b>먼저</b> 거르고 Top-k를 해야 원하는 개수가 나와요."
   ]
  },
  "s5": {
   "title": "pgvector에서 ORDER BY embedding <=> 질문 LIMIT 3",
   "lines": [
    [
     "[(102, '서버 실행', 0.827), (105, '과제 제출', 0.806), (106, '데이터베이스 연결', 0.802)]",
     1
    ],
    "③ 파이썬 결과와 같나 → True"
   ],
   "notes": [
    "<code>&lt;=&gt;</code>는 코사인 <b>거리</b>(1 − 유사도)라서, 유사도로 보려면 <code>1 - (...)</code>로 바꿔요. 순위와 점수가 파이썬으로 계산한 ③과 같아요."
   ]
  },
  "s6": {
   "title": "주제 200개 근처에 모인 무작위 벡터 20,000개, 질문 50개, 상위 10개 (질문 하나당 평균)",
   "cols": [
    "방법",
    "시간",
    "정답 일치율 (recall@10)",
    "실행 계획"
   ],
   "rows": [
    [
     "인덱스 없음 (전부 비교)",
     "63.83 ms",
     "1.00",
     "<code>Parallel Seq Scan on big</code>"
    ],
    [
     "HNSW (ef_search=40, 기본값)",
     "0.26 ms",
     "1.00",
     "<code>Index Scan using big_embedding_idx on big</code>"
    ],
    [
     "HNSW (ef_search=10)",
     "0.17 ms",
     "0.98",
     "<code>Index Scan using big_embedding_idx on big</code>"
    ]
   ],
   "notes": [
    "인덱스를 만드는 데 1.9초가 걸렸어요. 대신 질문마다 몇 배 빨라졌어요.",
    "정답 일치율은 \"전부 비교한 진짜 상위 10개\" 중 몇 개를 찾았나예요. HNSW는 일부만 보고 판단하는 <b>근사</b> 검색이라 1.00보다 낮을 수 있어요. 여기서도 기본값에서 1.00, <code>ef_search=10</code>에서 0.98로, 후보를 줄일수록 더 많이 놓쳤어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "벡터로",
   "encode",
   "384칸, 길이 1"
  ],
  [
   2,
   "거리",
   "내적, 코사인, L2",
   "정규화하면 같은 순위"
  ],
  [
   3,
   "Top-k",
   "argsort",
   "큰 순서 k개"
  ],
  [
   4,
   "조건",
   "먼저 거르기",
   "그다음 Top-k"
  ],
  [
   5,
   "DB로",
   "pgvector",
   "<=> 연산자"
  ],
  [
   6,
   "많아지면",
   "HNSW",
   "빠른 대신 근사"
  ]
 ],
 "env": "Python 3.13.16, sentence-transformers 6.3.0, intfloat/multilingual-e5-small, PostgreSQL + pgvector (노트 전용 DB)"
};
