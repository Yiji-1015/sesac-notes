# PostgreSQL 기초 전체 코드. 노트의 단계 번호(①, ②, …)가 이 파일의 번호 주석과 1:1로 맞아요.
# 실행: 빈 연습용 DB에서 python basics.py (같은 DB에 두 번 실행하면 ②에서 "이미 있음" 오류가 나요)
# DB 접속 정보는 환경 변수 PGHOST, PGPORT, PGUSER, PGPASSWORD, PGDATABASE 에서 읽어요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import psycopg                                #@1

# ① 접속하기
conn = psycopg.connect()  # 접속 정보는 환경 변수에서
version = conn.execute("SELECT version()").fetchone()[0]

# ② 테이블 두 개 만들기
conn.execute("""
    CREATE TABLE categories (
        id   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        name varchar(30) NOT NULL UNIQUE
             CHECK (char_length(name) BETWEEN 1 AND 30)
    )""")
conn.execute("""
    CREATE TABLE knowledge (
        id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        title       varchar(80) NOT NULL CHECK (char_length(title) BETWEEN 1 AND 80),
        content     text NOT NULL,
        category_id bigint NOT NULL REFERENCES categories (id) ON DELETE RESTRICT
    )""")
conn.commit()

# ③ 데이터 넣기
cats = conn.execute("""
    INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference')
    RETURNING id, name""").fetchall()
for title, content, cat in [("실습 파일 보관", "실습 파일은 개인 폴더에 보관합니다.", "learning"),
                            ("서버 실행", "프로젝트 폴더에서 서버를 실행합니다.", "environment"),
                            ("질문 작성", "오류 메시지와 시도한 작업을 함께 적습니다.", "learning")]:
    conn.execute("""
        INSERT INTO knowledge (title, content, category_id)
        SELECT %s, %s, id FROM categories WHERE name = %s""", (title, content, cat))
conn.commit()

# ④ 조회하기
rows = conn.execute("""
    SELECT id, title FROM knowledge
    WHERE category_id = 1
    ORDER BY id DESC""").fetchall()

# ⑤ 연결해서 읽기
joined = conn.execute("""
    SELECT k.title, c.name AS category
    FROM knowledge k JOIN categories c ON c.id = k.category_id
    ORDER BY k.id""").fetchall()
counts = conn.execute("""
    SELECT c.name, count(k.id) AS n
    FROM categories c LEFT JOIN knowledge k ON k.category_id = c.id
    GROUP BY c.name ORDER BY c.name""").fetchall()

# ⑥ 바꾸고 지우기
updated = conn.execute("""
    UPDATE knowledge SET content = %s WHERE id = %s
    RETURNING id, content""", ("프로젝트 폴더에서 uvicorn으로 실행합니다.", 2)).fetchall()
missing = conn.execute("UPDATE knowledge SET content = 'x' WHERE id = 99 RETURNING id").fetchall()
deleted = conn.execute("DELETE FROM knowledge WHERE id = 3 RETURNING id, title").fetchall()
conn.commit()

# ⑦ 트랜잭션: 한 묶음으로 확정하거나 취소하기
try:
    with conn.transaction():
        conn.execute("INSERT INTO categories (name) VALUES ('faq')")
        conn.execute("""
            INSERT INTO knowledge (title, content, category_id)
            VALUES ('자주 묻는 질문', '', 99)""")  # 없는 분류 99 → 실패
except psycopg.Error as e:
    tx_error = type(e).__name__
faq = conn.execute("SELECT count(*) FROM categories WHERE name = 'faq'").fetchone()[0]

# ⑧ 인덱스와 실행 계획
conn.execute("""
    INSERT INTO knowledge (title, content, category_id)
    SELECT '연습 ' || g, '연습용 내용', 1 FROM generate_series(1, 20000) AS g""")
conn.execute("ANALYZE knowledge")
plan_before = conn.execute("EXPLAIN SELECT * FROM knowledge WHERE title = '연습 777'").fetchall()
conn.execute("CREATE INDEX knowledge_title_idx ON knowledge (title)")
plan_after = conn.execute("EXPLAIN SELECT * FROM knowledge WHERE title = '연습 777'").fetchall()
conn.commit()
