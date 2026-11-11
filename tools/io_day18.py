"""Day18(PostgreSQL) 명령어 사전의 '입력 → 출력' 상자.

이 PC에는 PostgreSQL이 없어서, 강의 SQL을 파이썬 내장 SQLite(같은 S0 데이터)로 실제 실행한 값을 넣는다.
- SQL 문법 차이만 바꿔서 실행: IDENTITY → INTEGER PRIMARY KEY, char_length → length
- PostgreSQL에만 있는 것(current_database, SQLSTATE, pg_indexes, EXPLAIN 출력, psql)은
  강의자료·공식 문서 값이라고 상자 안에 따로 적는다 (PG = ...).
PostgreSQL을 설치하면 이 스크립트를 psycopg 버전으로 바꿔 다시 돌리면 된다.

실행: python tools/io_day18.py          (cmds/day18.html 이 있으면 상자를 넣고, 없으면 출력만)
"""
import html
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]

DDL_CAT = """CREATE TABLE categories (
  id BIGINT GENERATED ALWAYS AS IDENTITY
    PRIMARY KEY,
  name VARCHAR(30) NOT NULL UNIQUE,
  CONSTRAINT categories_name_length
    CHECK (char_length(name) BETWEEN 1 AND 30)
)"""
DDL_KNOW = """CREATE TABLE knowledge (
  id BIGINT GENERATED ALWAYS AS IDENTITY
    PRIMARY KEY,
  title VARCHAR(80) NOT NULL,
  content TEXT NOT NULL,
  category_id BIGINT NOT NULL,
  CONSTRAINT knowledge_title_length
    CHECK (char_length(title) BETWEEN 1 AND 80),
  CONSTRAINT knowledge_content_length
    CHECK (char_length(content) BETWEEN 1 AND 2000),
  CONSTRAINT knowledge_category_fk
    FOREIGN KEY (category_id)
    REFERENCES categories(id) ON DELETE RESTRICT
)"""
SEED = [
    "INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference') RETURNING id, name",
    "INSERT INTO knowledge (title, content, category_id) SELECT '실습 파일 보관', '실습 파일은 개인 폴더에 보관합니다.', id "
    "FROM categories WHERE name = 'learning' RETURNING id, title",
    "INSERT INTO knowledge (title, content, category_id) SELECT '서버 실행', '프로젝트 폴더에서 서버를 실행합니다.', id "
    "FROM categories WHERE name = 'environment' RETURNING id, title",
    "INSERT INTO knowledge (title, content, category_id) SELECT '질문 작성', '오류 메시지와 시도한 작업을 함께 적습니다.', id "
    "FROM categories WHERE name = 'learning' RETURNING id, title",
]


def lite(sql):
    sql = re.sub(r"BIGINT GENERATED ALWAYS AS IDENTITY\s+PRIMARY KEY", "INTEGER PRIMARY KEY", sql)
    return sql.replace("char_length", "length")


TMP = Path(tempfile.mkdtemp())
DBN = [0]


def fresh(seed=True):
    DBN[0] += 1
    path = TMP / f"k{DBN[0]}.db"
    con = sqlite3.connect(path, isolation_level=None)       # autocommit — BEGIN/COMMIT/ROLLBACK는 직접
    con.execute("PRAGMA foreign_keys = ON")
    if seed:
        con.execute(lite(DDL_CAT)); con.execute(lite(DDL_KNOW))
        for s in SEED:
            con.execute(lite(s)).fetchall()
    return con, path


def fmt(cur):
    """결과를 '열 | 열' 표 글자로."""
    rows = cur.fetchall()
    if cur.description is None:
        return f"(완료 · 바뀐 행 {cur.rowcount})" if cur.rowcount >= 0 else "(완료)"
    cols = [d[0] for d in cur.description]
    if not rows:
        return " | ".join(cols) + "\n(0행)"
    show = lambda v: "NULL" if v is None else str(v)
    return "\n".join([" | ".join(cols)] + [" | ".join(show(v) for v in r) for r in rows])


def run(con, sql):
    try:
        return fmt(con.execute(lite(sql)))
    except sqlite3.Error as e:
        return f"오류: {e}"


IO = {}
TITLE = "S0 데이터로 SQLite에서 실제 실행 (이 PC엔 PostgreSQL 미설치 · 결과 행은 PostgreSQL과 같음)"

# ---------------------------------------------------------------- ① 접속 (PostgreSQL 전용 → 강의자료 값)
IO["d18-current-database"] = ("PostgreSQL 전용 함수라 SQLite로는 못 돌려요 — 강의자료 p21의 기대값", [
    ("SELECT current_database() AS db_name, current_user AS user_name;", "PG = db_name: knowledge_lab (user: 접속한 계정)", "지금 연결된 DB 이름부터 확인"),
    ("SELECT 1 AS connection_ok;", "PG = connection_ok: 1", "SQL을 보내고 결과를 받는 길이 열렸다"),
    ("SELECT version();", "PG = 서버 버전 문자열 (수업 기준 18.x)", "psql --version은 클라이언트 버전 — 서로 다를 수 있어요"),
])

IO["d18-create-database"] = ("PostgreSQL 전용이라 SQLite로는 못 돌려요 — 강의자료 p68(A01)의 기대값", [
    ("CREATE DATABASE knowledge_lab;  (postgres DB에서)", "PG = CREATE DATABASE", "새 DB가 생김 — 아직 이 창은 postgres DB"),
    ("knowledge_lab에서 새 Query Tool → SELECT current_database();", "PG = knowledge_lab", "이제부터 표를 만들 곳"),
    ("BEGIN; CREATE DATABASE ...", "PG = ERROR: CREATE DATABASE cannot run inside a transaction block", "그래서 Auto commit 상태에서 단독 실행"),
])

# ---------------------------------------------------------------- ② SELECT
con, _ = fresh()
IO["d18-select"] = (TITLE, [
    ("SELECT id, title, category_id FROM knowledge ORDER BY id;", run(con, "SELECT id, title, category_id FROM knowledge ORDER BY id"), "한 행 = 지식 한 건"),
    ("SELECT name FROM categories ORDER BY id;", run(con, "SELECT name FROM categories ORDER BY id"), "분류 3행"),
])
IO["d18-where"] = (TITLE, [
    ("WHERE id = 2", run(con, "SELECT id, title FROM knowledge WHERE id = 2"), "특정 지식 한 건"),
    ("WHERE category_id = 1 ORDER BY id", run(con, "SELECT id, title FROM knowledge WHERE category_id = 1 ORDER BY id"), "같은 분류(learning)의 지식"),
    ("WHERE id = 99", run(con, "SELECT id, title FROM knowledge WHERE id = 99"), "0행은 오류가 아니라 '없음'이라는 정상 결과"),
])
IO["d18-like-limit"] = (TITLE, [
    ("WHERE category_id = 1 AND title LIKE '%실습%' ORDER BY id DESC LIMIT 2",
     run(con, "SELECT id, title FROM knowledge WHERE category_id = 1 AND title LIKE '%실습%' ORDER BY id DESC LIMIT 2"), "두 조건을 다 만족하는 건 1번뿐"),
    ("WHERE category_id = 1 ORDER BY id DESC LIMIT 2", run(con, "SELECT id, title FROM knowledge WHERE category_id = 1 ORDER BY id DESC LIMIT 2"), "LIKE를 빼면 큰 ID부터 2행"),
    ("WHERE title LIKE '%실행%'", run(con, "SELECT id, title FROM knowledge WHERE title LIKE '%실행%'"), "글자 포함 검색 — 뜻이 비슷한 '구동'은 못 찾아요"),
])
IO["d18-null"] = (TITLE + " · SQLite는 true/false를 1/0으로 보여 줘요", [
    ("SELECT NULL IS NULL AS is_missing;", run(con, "SELECT NULL IS NULL AS is_missing"), "PG에선 true"),
    ("SELECT '' IS NULL AS is_missing;", run(con, "SELECT '' IS NULL AS is_missing"), "PG에선 false — 빈 문자열은 값이 있음"),
    ("SELECT char_length('') AS text_length;", run(con, "SELECT char_length('') AS text_length"), "길이 0인 값"),
    ("SELECT NULL = NULL AS eq;", run(con, "SELECT NULL = NULL AS eq"), "= 로 비교하면 '모름'(NULL) — 그래서 IS NULL을 써요"),
])
con.close()

# ---------------------------------------------------------------- ③ DDL · Seed
con, _ = fresh(seed=False)
r_cat = run(con, DDL_CAT)
r_know = run(con, DDL_KNOW)
rows_tables = run(con, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
IO["d18-create-categories"] = (TITLE, [
    ("CREATE TABLE categories (...)", r_cat, "표가 생김 (PG 메시지: CREATE TABLE)"),
    ("INSERT INTO categories (name) VALUES ('learning');", run(con, "INSERT INTO categories (name) VALUES ('learning') RETURNING id, name"), "id는 생략 → DB가 1을 발급"),
    ("같은 'learning' 한 번 더", run(con, "INSERT INTO categories (name) VALUES ('learning')"), "UNIQUE가 중복 분류명을 거부 (PG SQLSTATE 23505)"),
    ("name = ''", run(con, "INSERT INTO categories (name) VALUES ('')"), "길이 CHECK가 빈 이름을 거부 (PG 23514)"),
])
IO["d18-create-knowledge"] = (TITLE, [
    ("CREATE TABLE knowledge (...)", r_know, "부모 categories가 먼저 있어야 해요"),
    ("만들어진 표 목록", rows_tables, ""),
    ("category_id = 99 (없는 분류)", run(con, "INSERT INTO knowledge (title, content, category_id) VALUES ('t', 'c', 99)"), "FK가 부모 없는 참조를 거부 (PG 23503)"),
    ("title = NULL", run(con, "INSERT INTO knowledge (title, content, category_id) VALUES (NULL, 'c', 1)"), "NOT NULL 위반 (PG 23502)"),
    ("같은 제목 두 번", run(con, "INSERT INTO knowledge (title, content, category_id) VALUES ('같은 제목', 'a', 1), ('같은 제목', 'b', 1) RETURNING id, title"), "title엔 UNIQUE가 없어서 둘 다 들어가요 — 제목은 ID 대신 못 써요"),
])
con.close()

con, _ = fresh(seed=False)
con.execute(lite(DDL_CAT)); con.execute(lite(DDL_KNOW))
seed_out = [run(con, s) for s in SEED]
IO["d18-seed"] = (TITLE, [
    ("INSERT INTO categories (name) VALUES ('learning'), ('environment'), ('reference') RETURNING id, name;", seed_out[0], "DB가 발급한 실제 ID"),
    ("INSERT INTO knowledge ... SELECT '실습 파일 보관', ..., id FROM categories WHERE name = 'learning' RETURNING id, title;", seed_out[1], "분류 ID를 이름으로 찾아 넣음"),
    ("나머지 두 건 (environment · learning)", seed_out[2] + "\n" + seed_out[3].split("\n", 1)[1], ""),
    ("SELECT COUNT(*) FROM knowledge;", run(con, "SELECT COUNT(*) FROM knowledge"), "S0 준비 완료"),
    ("WHERE name = 'nothing' 으로 INSERT SELECT", run(con, "INSERT INTO knowledge (title, content, category_id) SELECT 'x', 'y', id FROM categories WHERE name = 'nothing' RETURNING id"), "부모가 없으면 오류 없이 0행 — 그래서 RETURNING을 꼭 봐요"),
])
con.close()

con, _ = fresh()
out_try = []
con.execute("BEGIN")
con.execute("CREATE TABLE categories_try (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
con.execute("CREATE TABLE knowledge_try (id INTEGER PRIMARY KEY, title TEXT NOT NULL, category_id INTEGER NOT NULL REFERENCES categories_try(id))")
con.execute("INSERT INTO categories_try VALUES (1, '연습')")
con.execute("INSERT INTO knowledge_try VALUES (1, '안내', 1)")
sel_try = run(con, "SELECT * FROM knowledge_try")
bad_try = run(con, "INSERT INTO knowledge_try VALUES (2, '안내2', 7)")
con.execute("ROLLBACK")
after_try = run(con, "SELECT name FROM sqlite_master WHERE name LIKE '%_try'")
IO["d18-try"] = (TITLE, [
    ("SELECT * FROM knowledge_try;", sel_try, "PK·FK가 연결된 연습용 표"),
    ("INSERT INTO knowledge_try VALUES (2, '안내2', 7);", bad_try, "연습 표에서도 FK가 동작"),
    ("ROLLBACK; 뒤 _try 표가 남았나?", after_try, "표 생성(DDL)까지 같이 취소됨 — 본 표는 그대로"),
])
con.close()

# ---------------------------------------------------------------- ④ INSERT · UPDATE · DELETE
con, _ = fresh()
con.execute("BEGIN")
ins = run(con, "INSERT INTO knowledge (title, content, category_id) VALUES ('임시 안내', '한 번만 실행합니다.', 1) RETURNING id, title, category_id")
cnt_in = run(con, "SELECT COUNT(*) FROM knowledge")
con.execute("ROLLBACK")
cnt_out = run(con, "SELECT COUNT(*) FROM knowledge")
again = run(con, "INSERT INTO knowledge (title, content, category_id) VALUES ('다시', '다시 넣기', 1) RETURNING id")
IO["d18-insert"] = (TITLE, [
    ("BEGIN; INSERT ... RETURNING id, title, category_id;", ins, "발급된 실제 ID를 바로 읽음"),
    ("(취소 전) SELECT COUNT(*)", cnt_in, "같은 연결에선 새 행이 보여요"),
    ("ROLLBACK; SELECT COUNT(*)", cnt_out, "새 행은 남지 않음"),
    ("취소 뒤 다시 INSERT ... RETURNING id", again, "SQLite는 4를 다시 줌. PG는 시퀀스 번호를 되돌리지 않아 5가 나올 수 있어요(공식 문서) — 그래서 ID를 예상하지 말고 RETURNING을 봐요"),
])
con.close()

con, _ = fresh()
before = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
con.execute("BEGIN")
upd = run(con, "UPDATE knowledge SET content = '실행 전에 가상환경을 확인합니다.' WHERE id = 2 RETURNING id, title, content")
con.execute("ROLLBACK")
after = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
cur = con.execute("UPDATE knowledge SET content = 'x' WHERE id = 99")
IO["d18-update"] = (TITLE, [
    ("시작: SELECT id, content WHERE id = 2", before, "원래 본문 기록"),
    ("BEGIN; UPDATE ... WHERE id = 2 RETURNING id, title, content;", upd, "변경 1행 — 어느 행이 어떻게 바뀌었는지"),
    ("ROLLBACK; SELECT id, content WHERE id = 2", after, "원문 복원"),
    ("UPDATE ... WHERE id = 99", f"바뀐 행 {cur.rowcount}", "없는 ID — 오류 없이 0행 (PG 메시지: UPDATE 0). 404로 볼지는 API가 정해요"),
])
con.close()

con, _ = fresh()
con.execute("BEGIN")
dele = run(con, "DELETE FROM knowledge WHERE id = 3 RETURNING id, title")
mid = run(con, "SELECT COUNT(*) FROM knowledge")
con.execute("ROLLBACK")
back = run(con, "SELECT id, title FROM knowledge WHERE id = 3")
con.execute("BEGIN")
cur = con.execute("DELETE FROM knowledge")
nall = cur.rowcount
con.execute("ROLLBACK")
IO["d18-delete"] = (TITLE, [
    ("BEGIN; DELETE FROM knowledge WHERE id = 3 RETURNING id, title;", dele, "지운 행이 무엇이었는지"),
    ("(취소 전) SELECT COUNT(*)", mid, ""),
    ("ROLLBACK; SELECT id, title WHERE id = 3", back, "취소해서 3번이 다시 보여요"),
    ("BEGIN; DELETE FROM knowledge;  (WHERE 없음) → ROLLBACK", f"바뀐 행 {nall}", "WHERE를 빼면 전체 삭제 — 연습이라 취소했어요"),
])
con.close()

# ---------------------------------------------------------------- ⑤ JOIN · 집계
con, _ = fresh()
IO["d18-join"] = (TITLE, [
    ("SELECT k.id, k.title, c.name AS category FROM knowledge AS k JOIN categories AS c ON c.id = k.category_id ORDER BY k.id;",
     run(con, "SELECT k.id, k.title, c.name AS category FROM knowledge AS k JOIN categories AS c ON c.id = k.category_id ORDER BY k.id"), "3행 — 한 행은 여전히 지식 한 건"),
    ("... WHERE k.id = 2", run(con, "SELECT k.id, k.title, c.name AS category FROM knowledge AS k JOIN categories AS c ON c.id = k.category_id WHERE k.id = 2"), "'서버 실행 안내와 분류를 보여 줘'"),
])
IO["d18-left-join"] = (TITLE, [
    ("SELECT c.name, k.id AS knowledge_id, k.title FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id ORDER BY c.id, k.id;",
     run(con, "SELECT c.name, k.id AS knowledge_id, k.title FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id ORDER BY c.id, k.id"), "4행 — 지식 없는 reference도 남고 오른쪽이 NULL"),
    ("같은 SQL을 (INNER) JOIN으로", run(con, "SELECT c.name, k.id AS knowledge_id, k.title FROM categories AS c JOIN knowledge AS k ON k.category_id = c.id ORDER BY c.id, k.id"), "3행 — reference가 사라져요"),
])
IO["d18-group-by"] = (TITLE, [
    ("SELECT c.name AS category, COUNT(k.id) AS knowledge_count ... LEFT JOIN ... GROUP BY c.id, c.name ORDER BY c.id;",
     run(con, "SELECT c.name AS category, COUNT(k.id) AS knowledge_count FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id GROUP BY c.id, c.name ORDER BY c.id"), "한 행 = 분류 한 개"),
    ("COUNT(k.id) 대신 COUNT(*)", run(con, "SELECT c.name AS category, COUNT(*) AS knowledge_count FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id GROUP BY c.id, c.name ORDER BY c.id"), "reference가 1 — NULL로 채운 보존 행까지 셌어요"),
    ("LEFT JOIN 뒤에 WHERE k.title LIKE '%실%'", run(con, "SELECT c.name AS category, COUNT(k.id) AS knowledge_count FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id WHERE k.title LIKE '%실%' GROUP BY c.id, c.name ORDER BY c.id"), "WHERE가 NULL 행을 걸러 0인 분류가 사라짐"),
    ("조건을 ON 안으로: ON k.category_id = c.id AND k.title LIKE '%실%'", run(con, "SELECT c.name AS category, COUNT(k.id) AS knowledge_count FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id AND k.title LIKE '%실%' GROUP BY c.id, c.name ORDER BY c.id"), "분류 3개가 다 남고 0도 보여요"),
])
IO["d18-aggregate"] = (TITLE, [
    ("SELECT COUNT(*), SUM(char_length(content)), MIN(...), MAX(...) FROM knowledge;",
     run(con, "SELECT COUNT(*) AS knowledge_count, SUM(char_length(content)) AS total_chars, MIN(char_length(content)) AS min_chars, MAX(char_length(content)) AS max_chars FROM knowledge"), "강의자료의 3건 · 64자 · 20 · 24와 같아요"),
    ("SELECT id, char_length(content) FROM knowledge ORDER BY id;", run(con, "SELECT id, char_length(content) AS chars FROM knowledge ORDER BY id"), "글자 수(공백 포함) — 내용이 맞는지는 알려 주지 않아요"),
])
IO["d18-having"] = (TITLE, [
    ("... GROUP BY c.id, c.name HAVING COUNT(k.id) >= 2 ORDER BY c.name;",
     run(con, "SELECT c.name, COUNT(k.id) AS knowledge_count FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id GROUP BY c.id, c.name HAVING COUNT(k.id) >= 2 ORDER BY c.name"), "묶은 뒤 '2건 이상인 분류'만"),
    ("HAVING COUNT(k.id) = 0", run(con, "SELECT c.name, COUNT(k.id) AS knowledge_count FROM categories AS c LEFT JOIN knowledge AS k ON k.category_id = c.id GROUP BY c.id, c.name HAVING COUNT(k.id) = 0"), "아직 비어 있는 분류 찾기"),
])
con.close()

# ---------------------------------------------------------------- ⑥ 제약 · 실패
con, _ = fresh()
con.execute("BEGIN")
e1 = run(con, "INSERT INTO knowledge (title, content, category_id) VALUES ('오류 확인', '', 1)")
con.execute("ROLLBACK")
c1 = run(con, "SELECT COUNT(*) FROM knowledge")
fails = [
    ("필수 title = NULL", "INSERT INTO knowledge (title, content, category_id) VALUES (NULL, '본문', 1)", "23502"),
    ("분류 learning 중복", "INSERT INTO categories (name) VALUES ('learning')", "23505"),
    ("content = ''", "INSERT INTO knowledge (title, content, category_id) VALUES ('제목', '', 1)", "23514"),
    ("없는 category_id = 99", "INSERT INTO knowledge (title, content, category_id) VALUES ('제목', '본문', 99)", "23503"),
]
IO["d18-fail-check"] = (TITLE + " · SQLSTATE는 PostgreSQL 공식 오류 코드표 값", [
    ("BEGIN; INSERT ... VALUES ('오류 확인', '', 1);", e1, "기대한 제약(knowledge_content_length)으로 실패 · PG SQLSTATE 23514"),
    ("ROLLBACK; SELECT COUNT(*) FROM knowledge;", c1, "실패한 행은 저장되지 않음 — 기존 3건 그대로"),
] + [(f"{name}", run(con, sql), f"PG SQLSTATE {code}") for name, sql, code in fails] + [
    ("실패 뒤 SELECT COUNT(*)", run(con, "SELECT COUNT(*) FROM knowledge"), "네 번 실패해도 3건 그대로"),
])
d_know = run(con, "DELETE FROM knowledge WHERE id = 1 RETURNING id")
con.execute("INSERT INTO knowledge (title, content, category_id) VALUES ('실습 파일 보관', '실습 파일은 개인 폴더에 보관합니다.', 1)")
IO["d18-fk-restrict"] = (TITLE, [
    ("DELETE FROM categories WHERE name = 'learning';", run(con, "DELETE FROM categories WHERE name = 'learning'"), "연결된 지식이 있어 RESTRICT가 거부 (PG 23503)"),
    ("SELECT COUNT(*) FROM knowledge WHERE category_id = 1", run(con, "SELECT COUNT(*) FROM knowledge WHERE category_id = 1"), "자식 지식도 그대로 — 삭제가 번지지 않음"),
    ("DELETE FROM categories WHERE name = 'reference' RETURNING id, name;", run(con, "DELETE FROM categories WHERE name = 'reference' RETURNING id, name"), "참조하는 지식이 없는 분류는 지워져요"),
    ("DELETE FROM knowledge WHERE id = 1 RETURNING id;", d_know, "지식 한 건 삭제는 부모와 무관 (연습 후 다시 넣었어요)"),
])
con.close()

con, _ = fresh()
con.execute("BEGIN")
con.execute("CREATE TABLE defaults_try (label TEXT DEFAULT '미지정')")
con.execute("INSERT INTO defaults_try DEFAULT VALUES")
con.execute("INSERT INTO defaults_try (label) VALUES (NULL)")
dflt = run(con, "SELECT label, label IS NULL AS is_missing FROM defaults_try")
con.execute("ROLLBACK")
IO["d18-default"] = (TITLE + " · is_missing의 1/0 = PG의 true/false", [
    ("INSERT DEFAULT VALUES; INSERT ... VALUES (NULL); SELECT label, label IS NULL", dflt, "생략 → '미지정', 직접 넣은 NULL → 그대로 NULL"),
])
con.close()

# ---------------------------------------------------------------- ⑦ 트랜잭션
con, path = fresh()
s1 = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
con.execute("BEGIN")
con.execute("UPDATE knowledge SET content = '취소할 임시 안내입니다.' WHERE id = 2")
s2 = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
other = sqlite3.connect(path, isolation_level=None)
s_other = run(other, "SELECT id, content FROM knowledge WHERE id = 2")
other.close()
con.execute("ROLLBACK")
s3 = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
IO["d18-rollback"] = (TITLE, [
    ("① 시작 SELECT", s1, "현재 본문 기록"),
    ("② BEGIN; UPDATE ...; SELECT (같은 연결)", s2, "작업 중: 임시 본문이 보여요"),
    ("②' 그 순간 다른 연결에서 SELECT", s_other, "아직 미확정이라 다른 연결엔 원래 본문"),
    ("③ ROLLBACK; SELECT", s3, "시작 본문으로 복귀 · 행 수·ID 그대로"),
])
con.execute("BEGIN")
cm = run(con, "UPDATE knowledge SET content = '실행 전에 가상환경을 확인합니다.' WHERE id = 2 RETURNING id, content")
con.execute("COMMIT")
con.close()
con = sqlite3.connect(path, isolation_level=None)          # 연결을 닫고 다시 접속
re1 = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
try:
    con.execute("ROLLBACK"); rb_msg = "(완료)"
except sqlite3.Error as e:
    rb_msg = f"오류: {e}"
re2 = run(con, "SELECT id, content FROM knowledge WHERE id = 2")
IO["d18-commit"] = (TITLE, [
    ("BEGIN; UPDATE ... RETURNING id, content; COMMIT;", cm, "확정"),
    ("연결을 닫고 다시 접속 → SELECT", re1, "새 연결에서도 수정 본문"),
    ("이어서 ROLLBACK;", rb_msg, "열린 트랜잭션이 없음 (PG는 경고 'there is no transaction in progress')"),
    ("다시 SELECT", re2, "이미 COMMIT한 값은 ROLLBACK으로 못 되돌려요"),
])
con.close()

con, _ = fresh()
start = run(con, "SELECT id, content FROM knowledge WHERE id IN (2, 3) ORDER BY id")
con.execute("BEGIN")
con.execute("UPDATE knowledge SET content = '확정되면 안 되는 임시 본문' WHERE id = 2")
second = run(con, "UPDATE knowledge SET content = '' WHERE id = 3")
mid2 = run(con, "SELECT id, content FROM knowledge WHERE id IN (2, 3) ORDER BY id")
con.execute("ROLLBACK")
end = run(con, "SELECT id, content FROM knowledge WHERE id IN (2, 3) ORDER BY id")
IO["d18-atomic"] = (TITLE, [
    ("시작: 2·3번 원문", start, ""),
    ("BEGIN; UPDATE 2번 → 성공, UPDATE 3번 content = ''", second, "두 번째 문장이 CHECK에 걸림"),
    ("(롤백 전) 2·3번", mid2, "SQLite는 실패한 문장만 취소해서 2번 변경이 아직 보여요. PG는 이 SELECT부터 25P02(트랜잭션 중단)로 거부"),
    ("ROLLBACK; 2·3번", end, "묶음 전체 취소 — 2번 첫 변경도 남지 않음"),
])
con.close()

# ---------------------------------------------------------------- ⑧ INDEX · EXPLAIN
con, _ = fresh()
idx0 = run(con, "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'knowledge'")
plan0 = " / ".join(r[3] for r in con.execute("EXPLAIN QUERY PLAN SELECT id, title FROM knowledge WHERE category_id = 1"))
con.execute("CREATE INDEX idx_knowledge_category_id ON knowledge (category_id)")
idx1 = run(con, "SELECT name, sql FROM sqlite_master WHERE type = 'index' AND tbl_name = 'knowledge'")
again_idx = run(con, "CREATE INDEX idx_knowledge_category_id ON knowledge (category_id)")
plan1 = " / ".join(r[3] for r in con.execute("EXPLAIN QUERY PLAN SELECT id, title FROM knowledge WHERE category_id = 1"))
res_same = run(con, "SELECT id, title FROM knowledge WHERE category_id = 1 ORDER BY id")
IO["d18-create-index"] = (TITLE + " · PG는 pg_indexes, SQLite는 sqlite_master로 목록을 봐요", [
    ("인덱스 목록 (만들기 전)", idx0, "FK를 만들어도 자식 열 인덱스는 자동으로 안 생겨요 (PG도 같음. PG엔 PK용 knowledge_pkey는 있음)"),
    ("CREATE INDEX idx_knowledge_category_id ON knowledge (category_id);", idx1, "새 탐색 구조"),
    ("같은 이름으로 한 번 더", again_idx, "그래서 만들기 전에 목록부터 확인"),
    ("인덱스 뒤 WHERE category_id = 1 결과", res_same, "결과 행은 그대로 — 바뀌는 건 찾아가는 길"),
])
IO["d18-explain"] = ("SQLite의 EXPLAIN QUERY PLAN 실제 출력 (PG의 EXPLAIN 표기와 다름)", [
    ("인덱스 전: WHERE category_id = 1", plan0, "표 전체를 훑음 (PG의 Seq Scan에 해당)"),
    ("인덱스 후: 같은 SQL", plan1, "인덱스로 바로 찾아감"),
    ("PG에서 3행짜리 표라면?", "PG = Seq Scan이 나와도 정상 (강의자료 p53)", "행이 적으면 전체를 읽는 게 더 싸다고 판단할 수 있어요 — 계획 이름을 정답으로 외우지 않기"),
])
con.close()

# ---------------------------------------------------------------- psql (PostgreSQL 전용)
IO["d18-psql"] = ("PostgreSQL 클라이언트라 이 PC에선 못 돌려요 — 각 명령이 하는 일 (공식 psql 문서)", [
    ("psql -U postgres -h localhost -p 5432 -d knowledge_lab", "Password for user postgres: (입력해도 화면에 안 보임)", "PowerShell에서 실행 — 접속되면 프롬프트가 knowledge_lab=# 로 바뀜"),
    ("\\conninfo", "접속한 DB · 사용자 · 호스트 · 포트 한 줄", "어느 서버·DB에 붙었는지 확인"),
    ("\\dt", "public 스키마의 표 목록 (categories, knowledge)", "SQL이 아니라 psql 전용 명령"),
    ("\\d knowledge", "열 · 자료형 · 제약 · 인덱스 · FK", "CREATE TABLE이 의도대로 됐는지 확인"),
    ("\\q", "psql 종료", "서버는 계속 켜져 있어요"),
])


# ---------------------------------------------------------------- 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


for cid, (title, rows) in IO.items():
    print(f"== {cid}")
    for i, o, n in rows:
        print(f"   {i}\n      → {o.replace(chr(10), chr(10) + '        ')}   ({n})")

frag = ROOT / "cmds" / "day18.html"
if frag.exists():
    s = frag.read_text(encoding="utf-8")
    s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
    missing = []
    for cid, (title, rows) in IO.items():
        m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
        if not m:
            missing.append(cid); continue
        sec = m.group(0)
        k = sec.find("</code></pre>\n")
        k = k + len("</code></pre>\n") if k >= 0 else sec.find("</table>\n") + len("</table>\n")
        pos = m.start() + k
        s = s[:pos] + box(title, rows) + "\n" + s[pos:]
    frag.write_text(s, encoding="utf-8")
    print("inserted", len(IO) - len(missing), "missing", missing)
