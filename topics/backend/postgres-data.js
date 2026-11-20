window.PAGE = {
 "code": [
  {
   "t": "import psycopg",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 접속하기",
   "s": 1
  },
  {
   "t": "conn = psycopg.connect()  # 접속 정보는 환경 변수에서",
   "s": 1
  },
  {
   "t": "version = conn.execute(\"SELECT version()\").fetchone()[0]",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 테이블 두 개 만들기",
   "s": 2
  },
  {
   "t": "conn.execute(\"\"\"",
   "s": 2
  },
  {
   "t": "    CREATE TABLE categories (",
   "s": 2
  },
  {
   "t": "        id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,",
   "s": 2
  },
  {
   "t": "        name varchar(30) NOT NULL UNIQUE",
   "s": 2
  },
  {
   "t": "             CHECK (char_length(name) BETWEEN 1 AND 30)",
   "s": 2
  },
  {
   "t": "    )\"\"\")",
   "s": 2
  },
  {
   "t": "conn.execute(\"\"\"",
   "s": 2
  },
  {
   "t": "    CREATE TABLE knowledge (",
   "s": 2
  },
  {
   "t": "        id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,",
   "s": 2
  },
  {
   "t": "        title       varchar(80) NOT NULL CHECK (char_length(title) BETWEEN 1 AND 80),",
   "s": 2
  },
  {
   "t": "        content     text NOT NULL,",
   "s": 2
  },
  {
   "t": "        category_id bigint NOT NULL REFERENCES categories (id) ON DELETE RESTRICT",
   "s": 2
  },
  {
   "t": "    )\"\"\")",
   "s": 2
  },
  {
   "t": "conn.commit()",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 데이터 넣기",
   "s": 3
  },
  {
   "t": "cats = conn.execute(\"\"\"",
   "s": 3
  },
  {
   "t": "    INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference')",
   "s": 3
  },
  {
   "t": "    RETURNING id, name\"\"\").fetchall()",
   "s": 3
  },
  {
   "t": "for title, content, cat in [(\"실습 파일 보관\", \"실습 파일은 개인 폴더에 보관합니다.\", \"learning\"),",
   "s": 3
  },
  {
   "t": "                            (\"서버 실행\", \"프로젝트 폴더에서 서버를 실행합니다.\", \"environment\"),",
   "s": 3
  },
  {
   "t": "                            (\"질문 작성\", \"오류 메시지와 시도한 작업을 함께 적습니다.\", \"learning\")]:",
   "s": 3
  },
  {
   "t": "    conn.execute(\"\"\"",
   "s": 3
  },
  {
   "t": "        INSERT INTO knowledge (title, content, category_id)",
   "s": 3
  },
  {
   "t": "        SELECT %s, %s, id FROM categories WHERE name = %s\"\"\", (title, content, cat))",
   "s": 3
  },
  {
   "t": "conn.commit()",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 조회하기",
   "s": 4
  },
  {
   "t": "rows = conn.execute(\"\"\"",
   "s": 4
  },
  {
   "t": "    SELECT id, title FROM knowledge",
   "s": 4
  },
  {
   "t": "    WHERE category_id = 1",
   "s": 4
  },
  {
   "t": "    ORDER BY id DESC\"\"\").fetchall()",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 연결해서 읽기",
   "s": 5
  },
  {
   "t": "joined = conn.execute(\"\"\"",
   "s": 5
  },
  {
   "t": "    SELECT k.title, c.name AS category",
   "s": 5
  },
  {
   "t": "    FROM knowledge k JOIN categories c ON c.id = k.category_id",
   "s": 5
  },
  {
   "t": "    ORDER BY k.id\"\"\").fetchall()",
   "s": 5
  },
  {
   "t": "counts = conn.execute(\"\"\"",
   "s": 5
  },
  {
   "t": "    SELECT c.name, count(k.id) AS n",
   "s": 5
  },
  {
   "t": "    FROM categories c LEFT JOIN knowledge k ON k.category_id = c.id",
   "s": 5
  },
  {
   "t": "    GROUP BY c.name ORDER BY c.name\"\"\").fetchall()",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 바꾸고 지우기",
   "s": 6
  },
  {
   "t": "updated = conn.execute(\"\"\"",
   "s": 6
  },
  {
   "t": "    UPDATE knowledge SET content = %s WHERE id = %s",
   "s": 6
  },
  {
   "t": "    RETURNING id, content\"\"\", (\"프로젝트 폴더에서 uvicorn으로 실행합니다.\", 2)).fetchall()",
   "s": 6
  },
  {
   "t": "missing = conn.execute(\"UPDATE knowledge SET content = 'x' WHERE id = 99 RETURNING id\").fetchall()",
   "s": 6
  },
  {
   "t": "deleted = conn.execute(\"DELETE FROM knowledge WHERE id = 3 RETURNING id, title\").fetchall()",
   "s": 6
  },
  {
   "t": "conn.commit()",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 트랜잭션: 한 묶음으로 확정하거나 취소하기",
   "s": 7
  },
  {
   "t": "try:",
   "s": 7
  },
  {
   "t": "    with conn.transaction():",
   "s": 7
  },
  {
   "t": "        conn.execute(\"INSERT INTO categories (name) VALUES ('faq')\")",
   "s": 7
  },
  {
   "t": "        conn.execute(\"\"\"",
   "s": 7
  },
  {
   "t": "            INSERT INTO knowledge (title, content, category_id)",
   "s": 7
  },
  {
   "t": "            VALUES ('자주 묻는 질문', '', 99)\"\"\")  # 없는 분류 99 → 실패",
   "s": 7
  },
  {
   "t": "except psycopg.Error as e:",
   "s": 7
  },
  {
   "t": "    tx_error = type(e).__name__",
   "s": 7
  },
  {
   "t": "faq = conn.execute(\"SELECT count(*) FROM categories WHERE name = 'faq'\").fetchone()[0]",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 인덱스와 실행 계획",
   "s": 8
  },
  {
   "t": "conn.execute(\"\"\"",
   "s": 8
  },
  {
   "t": "    INSERT INTO knowledge (title, content, category_id)",
   "s": 8
  },
  {
   "t": "    SELECT '연습 ' || g, '연습용 내용', 1 FROM generate_series(1, 20000) AS g\"\"\")",
   "s": 8
  },
  {
   "t": "conn.execute(\"ANALYZE knowledge\")",
   "s": 8
  },
  {
   "t": "plan_before = conn.execute(\"EXPLAIN SELECT * FROM knowledge WHERE title = '연습 777'\").fetchall()",
   "s": 8
  },
  {
   "t": "conn.execute(\"CREATE INDEX knowledge_title_idx ON knowledge (title)\")",
   "s": 8
  },
  {
   "t": "plan_after = conn.execute(\"EXPLAIN SELECT * FROM knowledge WHERE title = '연습 777'\").fetchall()",
   "s": 8
  },
  {
   "t": "conn.commit()",
   "s": 8
  }
 ],
 "names": {
  "1": "접속하기",
  "2": "테이블 두 개 만들기",
  "3": "데이터 넣기",
  "4": "조회하기",
  "5": "연결해서 읽기",
  "6": "바꾸고 지우기",
  "7": "트랜잭션: 한 묶음으로 확정하거나 취소하기",
  "8": "인덱스와 실행 계획"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "why": {
   "title": "20개 작업이 동시에 \"항목 하나 추가\"를 하면",
   "cols": [
    "저장 방식",
    "추가 요청",
    "남은 항목"
   ],
   "rows": [
    [
     "JSON 파일 (읽기 → 추가 → 통째로 쓰기)",
     "20",
     "<span class=\"bad\">1</span>"
    ],
    [
     "PostgreSQL (<code>INSERT</code>)",
     "20",
     "<b>20</b>"
    ]
   ],
   "notes": [
    "파일은 여럿이 거의 동시에 \"읽고, 하나 붙이고, 통째로 다시 쓰면\" 나중에 쓴 쪽이 앞 사람의 추가를 덮어써요. 20개 중 19개가 사라졌어요.",
    "DB는 동시에 들어온 쓰기를 순서대로 안전하게 처리해서 20개가 모두 남아요."
   ]
  },
  "s1": {
   "title": "SELECT version()",
   "lines": [
    [
     "PostgreSQL 18.6 on x86_64-windows, compiled by msvc-19.44.35…",
     1
    ]
   ],
   "notes": [
    "접속에 성공하면 DB 서버가 자기 버전을 알려 줘요. 접속 정보는 코드에 없고 환경 변수에서 왔어요."
   ]
  },
  "s2": {
   "title": "DB에게 물어본 실제 구조",
   "cols": [
    "테이블",
    "열",
    "타입",
    "비워도 되나"
   ],
   "rows": [
    [
     "categories",
     "id",
     "bigint",
     "NO"
    ],
    [
     "categories",
     "name",
     "character varying",
     "NO"
    ],
    [
     "knowledge",
     "id",
     "bigint",
     "NO"
    ],
    [
     "knowledge",
     "title",
     "character varying",
     "NO"
    ],
    [
     "knowledge",
     "content",
     "text",
     "NO"
    ],
    [
     "knowledge",
     "category_id",
     "bigint",
     "NO"
    ]
   ],
   "notes": [
    "<code>categories_name_check</code> CHECK (값 조건), <code>categories_name_key</code> UNIQUE (중복 금지), <code>categories_pkey</code> PRIMARY KEY (기본 키), <code>knowledge_category_id_fkey</code> FOREIGN KEY (외래 키), <code>knowledge_pkey</code> PRIMARY KEY (기본 키), <code>knowledge_title_check</code> CHECK (값 조건). 제약 조건 이름을 따로 적지 않으면 PostgreSQL이 <code>테이블_열_종류</code> 꼴로 지어 줘요."
   ]
  },
  "s3": {
   "title": "INSERT ... RETURNING id, name",
   "lines": [
    [
     "[(1, 'learning'), (2, 'environment'), (3, 'reference')]",
     1
    ],
    "knowledge 행 수 → 3"
   ],
   "notes": [
    "<code>id</code>는 넣지 않았는데 DB가 1, 2, 3을 붙여 돌려줬어요 (<code>RETURNING</code>)."
   ]
  },
  "t2": {
   "title": "규칙을 어기는 요청 다섯 개",
   "cols": [
    "시도",
    "결과",
    "DB의 메시지"
   ],
   "rows": [
    [
     "같은 분류 이름 한 번 더",
     "<b>UniqueViolation</b>",
     "중복된 키 값이 &quot;categories_name_key&quot; 고유 제약 조건을 위반함"
    ],
    [
     "빈 분류 이름",
     "<b>CheckViolation</b>",
     "새 자료가 &quot;categories&quot; 릴레이션의 &quot;categories_name_check&quot; 체크 제약 조건을 위반했습니다"
    ],
    [
     "없는 분류 99번에 지식 넣기",
     "<b>ForeignKeyViolation</b>",
     "&quot;knowledge&quot; 테이블에서 자료 추가, 갱신 작업이 &quot;knowledge_category_id_fkey&quot; 참조키(foreign key) 제약 조건을 위배했습니다"
    ],
    [
     "제목 없이(NULL) 넣기",
     "<b>NotNullViolation</b>",
     "&quot;title&quot; 칼럼(해당 릴레이션 &quot;knowledge&quot;)의 null 값이 not null 제약조건을 위반했습니다."
    ],
    [
     "지식이 남아 있는 분류 지우기",
     "<b>RestrictViolation</b>",
     "&quot;categories&quot; 테이블의 자료 갱신, 삭제 작업이 RESTRICT &quot;knowledge_category_id_fkey&quot; 참조키(foreign key) 제약 조건 - &quot;knowledge&quot; 테이블 - 을 위반했습니다"
    ]
   ],
   "notes": [
    "모두 DB가 거부했어요. 코드에 실수가 있어도 <b>DB가 마지막 문지기</b>가 돼요.",
    "마지막 줄은 외래 키의 <code>ON DELETE RESTRICT</code> 때문이에요. 분류를 지우면 그 분류를 가리키던 지식이 \"부모 없는 행\"이 되니까 막아요."
   ]
  },
  "s4": {
   "title": "WHERE category_id = 1 ORDER BY id DESC",
   "lines": [
    [
     "[(3, '질문 작성'), (1, '실습 파일 보관')]",
     1
    ]
   ],
   "notes": [
    "분류 1(learning)인 지식만, 번호가 큰 것부터. 결과는 튜플의 목록이에요."
   ]
  },
  "s5": {
   "title": "JOIN 과 LEFT JOIN + GROUP BY",
   "lines": [
    [
     "joined → [('실습 파일 보관', 'learning'), ('서버 실행', 'environment'), ('질문 작성', 'learning')]",
     1
    ],
    [
     "counts → [('environment', 1), ('learning', 2), ('reference', 0)]",
     2
    ]
   ],
   "notes": [
    "<code>knowledge</code>에는 분류 번호만 있지만, <code>JOIN</code>으로 <code>categories</code>와 이어서 이름을 같이 읽었어요.",
    "<code>reference</code> 분류는 지식이 0개인데도 나왔어요. <code>LEFT JOIN</code>은 왼쪽(분류)을 모두 남겨요. 그냥 <code>JOIN</code>이면 0개인 분류는 빠져요."
   ]
  },
  "s6": {
   "title": "UPDATE / DELETE ... RETURNING",
   "lines": [
    [
     "updated → [(2, '프로젝트 폴더에서 uvicorn으로 실행합니다.')]",
     1
    ],
    [
     "id 99 UPDATE → []",
     2,
     "bad"
    ],
    [
     "deleted → [(3, '질문 작성')]",
     3
    ]
   ],
   "notes": [
    "바뀐 행을 <code>RETURNING</code>으로 바로 받아요.",
    "없는 번호를 고치면 <b>오류가 아니라 0행</b>이에요. API라면 이 빈 목록을 보고 404로 바꿔야 해요. DB는 \"없는 걸 고치라고 했다\"를 실패로 보지 않아요.",
    "지운 행도 돌려받아서 무엇을 지웠는지 확인할 수 있어요."
   ]
  },
  "s7": {
   "title": "트랜잭션 안에서 분류 추가 → 지식 추가(실패)",
   "lines": [
    [
     "오류 → ForeignKeyViolation",
     1
    ],
    [
     "faq 분류 개수 → 0",
     2
    ],
    [
     "그다음 분류를 넣으면 받는 id → 5",
     3
    ]
   ],
   "notes": [
    "두 번째 INSERT가 외래 키 규칙에 걸렸어요.",
    "첫 번째 INSERT(분류 추가)도 <b>같이 취소</b>됐어요. 반쪽짜리 결과가 남지 않아요.",
    "취소된 분류가 받았던 번호 4는 다시 쓰이지 않고 건너뛰어요. 번호에 빈칸이 생기는 건 정상이에요."
   ]
  },
  "t7": {
   "title": "같은 두 작업을 트랜잭션 없이(하나씩 바로 확정) 하면",
   "lines": [
    "지식 추가 → ForeignKeyViolation",
    [
     "분류만 남음 → faq2 개수 1",
     1,
     "bad"
    ]
   ],
   "notes": [
    "분류는 이미 확정돼서 남고, 지식만 빠졌어요. \"분류와 첫 지식을 함께 만든다\"는 작업이 반만 된 상태예요."
   ]
  },
  "s8": {
   "title": "EXPLAIN (행 20,002개)",
   "lines": [
    "인덱스 없음:",
    [
     "Seq Scan on knowledge  (cost=0.00..437.02 rows=1 width=45)",
     1,
     "bad"
    ],
    "인덱스 만든 뒤:",
    [
     "Index Scan using knowledge_title_idx on knowledge  (cost=0.29..8.30 rows=1 width=45)",
     2
    ],
    "",
    "실제로 실행한 시간 (EXPLAIN ANALYZE): 인덱스 없음 0.796 ms, 있음 0.041 ms"
   ],
   "notes": [
    "<code>Seq Scan</code>은 2만 줄을 처음부터 끝까지 다 읽는다는 뜻이에요.",
    "<code>Index Scan</code>은 인덱스(책 뒤의 찾아보기처럼 값과 위치를 정렬해 둔 목록)로 바로 찾아가요. <code>cost</code>(DB가 어림잡은 비용)가 크게 줄었어요."
   ]
  }
 },
 "flow": [
  [
   1,
   "접속",
   "psycopg",
   "환경 변수"
  ],
  [
   2,
   "테이블",
   "2개",
   "PK, FK, CHECK"
  ],
  [
   3,
   "넣기",
   "INSERT",
   "RETURNING"
  ],
  [
   4,
   "조회",
   "SELECT",
   "WHERE, ORDER BY"
  ],
  [
   5,
   "연결",
   "JOIN",
   "GROUP BY"
  ],
  [
   6,
   "바꾸기",
   "UPDATE, DELETE",
   "0행은 오류 아님"
  ],
  [
   7,
   "트랜잭션",
   "전부 아니면 전무",
   "rollback"
  ],
  [
   8,
   "인덱스",
   "EXPLAIN",
   "Seq → Index"
  ]
 ],
 "env": "PostgreSQL 18.6 on x86_64-windows, psycopg 3.3.8, Python 3.13.16"
};
