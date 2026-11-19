"""주제 페이지 '백엔드 > 데이터베이스 > PostgreSQL 기초'를 만든다.

examples/db/basics.py 를 노트 전용 DB에서 단계별로 실제 실행한다 (tools/notes_pg.py, 비밀번호는 출력하지 않음).
실행: NOTES_PG_ENV=<.env 경로> C:/practice_assignment/.venv/Scripts/python.exe tools/pages/pg_basics.py
"""
import json
import sys
import tempfile
import threading
import time
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import notes_pg  # noqa: E402
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import psycopg  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "db"
code = S.parse((EX / "basics.py").read_text(encoding="utf-8"))
outs = {}


def at(k):
    notes_pg.fresh()
    return S.run(code, k, S.workdir(EX), filename="basics.py")


def rows_tbl(rows):
    return [[E(str(v)) for v in r] for r in rows]


def err(sql, conn):
    try:
        with conn.transaction():
            conn.execute(sql)
        return "성공", ""
    except psycopg.Error as e:
        return type(e).__name__, notes_pg.scrub(str(e).splitlines()[0])


try:
    # ---- 왜: 파일 vs DB, 동시에 20번 추가 ----------------------------------------
    tmp = Path(tempfile.mkdtemp()) / "knowledge.json"
    tmp.write_text(json.dumps({"entries": []}), encoding="utf-8")

    def file_add(i):
        d = json.loads(tmp.read_text(encoding="utf-8"))
        time.sleep(0.01)
        d["entries"].append({"id": i})
        tmp.write_text(json.dumps(d), encoding="utf-8")

    ts = [threading.Thread(target=file_add, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    file_n = len(json.loads(tmp.read_text(encoding="utf-8"))["entries"])
    notes_pg.fresh()
    with psycopg.connect(autocommit=True) as c0:
        c0.execute("CREATE TABLE t (id int)")

    def db_add(i):
        with psycopg.connect(autocommit=True) as c:
            time.sleep(0.01)
            c.execute("INSERT INTO t VALUES (%s)", (i,))

    ts = [threading.Thread(target=db_add, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    with psycopg.connect() as c0:
        db_n = c0.execute("SELECT count(*) FROM t").fetchone()[0]
    outs["why"] = S.table("20개 작업이 동시에 \"항목 하나 추가\"를 하면", ["저장 방식", "추가 요청", "남은 항목"],
                          [["JSON 파일 (읽기 → 추가 → 통째로 쓰기)", "20", f"<span class=\"bad\">{file_n}</span>"],
                           ["PostgreSQL (<code>INSERT</code>)", "20", f"<b>{db_n}</b>"]],
                          [f"파일은 여럿이 거의 동시에 \"읽고, 하나 붙이고, 통째로 다시 쓰면\" 나중에 쓴 쪽이 앞 사람의 추가를 덮어써요. 20개 중 {20 - file_n}개가 사라졌어요.",
                           "DB는 동시에 들어온 쓰기를 순서대로 안전하게 처리해서 20개가 모두 남아요."])

    # ---- ① ② ③ -----------------------------------------------------------------
    ns = at(1)
    outs["s1"] = S.box("SELECT version()", [(ns["version"][:60] + "…", 1)], ["접속에 성공하면 DB 서버가 자기 버전을 알려 줘요. 접속 정보는 코드에 없고 환경 변수에서 왔어요."])
    ns["conn"].close()
    ns = at(2)
    cols = ns["conn"].execute("""SELECT table_name, column_name, data_type, is_nullable FROM information_schema.columns
                                 WHERE table_schema='public' ORDER BY table_name, ordinal_position""").fetchall()
    cons = ns["conn"].execute("""SELECT conrelid::regclass::text, conname, contype FROM pg_constraint
                                 WHERE connamespace = 'public'::regnamespace ORDER BY 1, 2""").fetchall()
    kind = {"p": "PRIMARY KEY (기본 키)", "u": "UNIQUE (중복 금지)", "c": "CHECK (값 조건)", "f": "FOREIGN KEY (외래 키)", "n": "NOT NULL"}
    outs["s2"] = S.table("DB에게 물어본 실제 구조", ["테이블", "열", "타입", "비워도 되나"], rows_tbl(cols),
                         [", ".join(f"<code>{E(n)}</code> {kind.get(t, t)}" for _, n, t in cons if t != "n")
                          + ". 제약 조건 이름을 따로 적지 않으면 PostgreSQL이 <code>테이블_열_종류</code> 꼴로 지어 줘요."])
    ns["conn"].close()
    ns = at(3)
    conn = ns["conn"]
    outs["s3"] = S.box("INSERT ... RETURNING id, name", [(str(ns["cats"]), 1), f"knowledge 행 수 → {conn.execute('SELECT count(*) FROM knowledge').fetchone()[0]}"],
                       ["<code>id</code>는 넣지 않았는데 DB가 1, 2, 3을 붙여 돌려줬어요 (<code>RETURNING</code>)."])
    tests = [("같은 분류 이름 한 번 더", "INSERT INTO categories (name) VALUES ('learning')"),
             ("빈 분류 이름", "INSERT INTO categories (name) VALUES ('')"),
             ("없는 분류 99번에 지식 넣기", "INSERT INTO knowledge (title, content, category_id) VALUES ('t', 'c', 99)"),
             ("제목 없이(NULL) 넣기", "INSERT INTO knowledge (title, content, category_id) VALUES (NULL, 'c', 1)"),
             ("지식이 남아 있는 분류 지우기", "DELETE FROM categories WHERE name = 'learning'")]
    outs["t2"] = S.table("규칙을 어기는 요청 다섯 개", ["시도", "결과", "DB의 메시지"],
                         [[E(n), f"<b>{err(q, conn)[0]}</b>", E(err(q, conn)[1])] for n, q in tests],
                         ["모두 DB가 거부했어요. 코드에 실수가 있어도 <b>DB가 마지막 문지기</b>가 돼요.", "마지막 줄은 외래 키의 <code>ON DELETE RESTRICT</code> 때문이에요. 분류를 지우면 그 분류를 가리키던 지식이 \"부모 없는 행\"이 되니까 막아요."])
    conn.close()

    # ---- ④ ⑤ ⑥ ------------------------------------------------------------------
    ns = at(6)
    outs["s4"] = S.box("WHERE category_id = 1 ORDER BY id DESC", [(str(ns["rows"]), 1)], ["분류 1(learning)인 지식만, 번호가 큰 것부터. 결과는 튜플의 목록이에요."])
    outs["s5"] = S.box("JOIN 과 LEFT JOIN + GROUP BY", [("joined → " + str(ns["joined"]), 1), ("counts → " + str(ns["counts"]), 2)],
                       ["<code>knowledge</code>에는 분류 번호만 있지만, <code>JOIN</code>으로 <code>categories</code>와 이어서 이름을 같이 읽었어요.",
                        "<code>reference</code> 분류는 지식이 0개인데도 나왔어요. <code>LEFT JOIN</code>은 왼쪽(분류)을 모두 남겨요. 그냥 <code>JOIN</code>이면 0개인 분류는 빠져요."])
    outs["s6"] = S.box("UPDATE / DELETE ... RETURNING", [("updated → " + str(ns["updated"]), 1), ("id 99 UPDATE → " + str(ns["missing"]), 2, "bad"), ("deleted → " + str(ns["deleted"]), 3)],
                       ["바뀐 행을 <code>RETURNING</code>으로 바로 받아요.", "없는 번호를 고치면 <b>오류가 아니라 0행</b>이에요. API라면 이 빈 목록을 보고 404로 바꿔야 해요. DB는 \"없는 걸 고치라고 했다\"를 실패로 보지 않아요.",
                        "지운 행도 돌려받아서 무엇을 지웠는지 확인할 수 있어요."])
    ns["conn"].close()

    # ---- ⑦ -----------------------------------------------------------------------
    ns = at(7)
    conn = ns["conn"]
    nxt = conn.execute("INSERT INTO categories (name) VALUES ('faq') RETURNING id").fetchone()[0]
    conn.rollback()
    outs["s7"] = S.box("트랜잭션 안에서 분류 추가 → 지식 추가(실패)", [(f"오류 → {ns['tx_error']}", 1), (f"faq 분류 개수 → {ns['faq']}", 2), (f"그다음 분류를 넣으면 받는 id → {nxt}", 3)],
                       ["두 번째 INSERT가 외래 키 규칙에 걸렸어요.", "첫 번째 INSERT(분류 추가)도 <b>같이 취소</b>됐어요. 반쪽짜리 결과가 남지 않아요.",
                        "취소된 분류가 받았던 번호 4는 다시 쓰이지 않고 건너뛰어요. 번호에 빈칸이 생기는 건 정상이에요."])
    with psycopg.connect(autocommit=True) as c:
        c.execute("INSERT INTO categories (name) VALUES ('faq2')")
        e2 = err("INSERT INTO knowledge (title, content, category_id) VALUES ('자주 묻는 질문', '', 99)", c)[0]
        left = c.execute("SELECT count(*) FROM categories WHERE name = 'faq2'").fetchone()[0]
    outs["t7"] = S.box("같은 두 작업을 트랜잭션 없이(하나씩 바로 확정) 하면", [f"지식 추가 → {e2}", (f"분류만 남음 → faq2 개수 {left}", 1, "bad")],
                       ["분류는 이미 확정돼서 남고, 지식만 빠졌어요. \"분류와 첫 지식을 함께 만든다\"는 작업이 반만 된 상태예요."])
    conn.close()

    # ---- ⑧ -----------------------------------------------------------------------
    ns = at(8)
    conn = ns["conn"]
    conn.execute("DROP INDEX knowledge_title_idx")
    t0 = conn.execute("EXPLAIN ANALYZE SELECT * FROM knowledge WHERE title = '연습 777'").fetchall()[-1][0]
    conn.execute("CREATE INDEX knowledge_title_idx ON knowledge (title)")
    conn.execute("ANALYZE knowledge")
    t1 = conn.execute("EXPLAIN ANALYZE SELECT * FROM knowledge WHERE title = '연습 777'").fetchall()[-1][0]
    n = conn.execute("SELECT count(*) FROM knowledge").fetchone()[0]
    outs["s8"] = S.box(f"EXPLAIN (행 {n:,}개)", ["인덱스 없음:", (ns["plan_before"][0][0], 1, "bad"), "인덱스 만든 뒤:", (ns["plan_after"][0][0], 2),
                                                 "", f"실제로 실행한 시간 (EXPLAIN ANALYZE): 인덱스 없음 {t0.split(':')[1].strip()}, 있음 {t1.split(':')[1].strip()}"],
                       ["<code>Seq Scan</code>은 2만 줄을 처음부터 끝까지 다 읽는다는 뜻이에요.", "<code>Index Scan</code>은 인덱스(책 뒤의 찾아보기처럼 값과 위치를 정렬해 둔 목록)로 바로 찾아가요. <code>cost</code>(DB가 어림잡은 비용)가 크게 줄었어요."])
    conn.close()
    pgver = ns["version"].split(",")[0]
finally:
    notes_pg.drop()

flow = [[1, "접속", "psycopg", "환경 변수"], [2, "테이블", "2개", "PK, FK, CHECK"], [3, "넣기", "INSERT", "RETURNING"], [4, "조회", "SELECT", "WHERE, ORDER BY"],
        [5, "연결", "JOIN", "GROUP BY"], [6, "바꾸기", "UPDATE, DELETE", "0행은 오류 아님"], [7, "트랜잭션", "전부 아니면 전무", "rollback"], [8, "인덱스", "EXPLAIN", "Seq → Index"]]
S.save(ROOT / "topics" / "backend" / "postgres-data.js", {"code": code, "names": S.names(code), "range": [1, 8], "outs": outs, "flow": flow,
                                                           "env": f"{pgver}, psycopg {psycopg.__version__}, Python {sys.version.split()[0]}"})

why = f"""
<p>지금까지 지식은 JSON 파일에 저장했어요 (<a href="fastapi.html">API 서버</a>). 파일은 간단하지만, 여러 요청이 <b>동시에</b> 쓰면 문제가 생겨요. 실제로 20개 작업이 동시에 항목 하나씩을 추가해 봤어요.</p>
{K.out("why")}
<p><b>DB</b>(데이터베이스)는 데이터를 여러 사용자가 동시에 안전하게 저장하고 찾을 수 있게 관리하는 프로그램이에요. 그중 <b>PostgreSQL</b>은 표(테이블) 모양으로 데이터를 저장하는 관계형 DB이고, <b>SQL</b>이라는 언어로 "넣어라", "찾아라"를 요청해요.</p>
<p>DB가 파일보다 나은 점은 세 가지예요.</p>
<ul><li><b>동시에 써도 안전해요.</b> (위 결과)</li><li><b>규칙을 DB가 지켜요.</b> "분류 이름은 비울 수 없다", "없는 분류를 가리킬 수 없다" 같은 규칙을 걸어 두면, 코드에 실수가 있어도 DB가 거부해요.</li><li><b>여러 작업을 한 묶음으로 처리해요.</b> 하나라도 실패하면 전부 취소해서 반쪽짜리 상태가 남지 않아요.</li></ul>
<p>예제는 지식과 분류를 두 테이블로 나눠 저장해요. 파이썬에서는 <b>psycopg</b>로 PostgreSQL에 접속해요.</p>
"""
whole = f"""
<p>접속 → 테이블 설계 → 넣기 → 꺼내기 → 연결해서 읽기 → 바꾸기 → 묶어서 처리하기 → 빠르게 찾기 순서예요. 상자를 누르면 그 단계로 가요.</p>
<div class="flow" id="flow"></div>
<p>결과는 모두 노트 전용 연습 DB를 새로 만들어 <code>basics.py</code>를 단계별로 실행한 값이에요. 실행이 끝나면 DB는 지웠어요.</p>
"""
steps = [
    K.step(1, "접속하기", "psycopg로 PostgreSQL에 접속해요.",
           "<p>접속에는 주소(host), 포트, 사용자, 비밀번호, DB 이름이 필요해요. 비밀번호를 코드에 적으면 코드를 공유할 때 같이 새니까, <b>환경 변수</b>에 두고 읽어요. <code>psycopg.connect()</code>를 인자 없이 부르면 <code>PGHOST</code>, <code>PGUSER</code>, <code>PGPASSWORD</code>, <code>PGDATABASE</code> 같은 환경 변수를 알아서 읽어요.</p>"
           + K.code_table([("psycopg.connect()", "접속을 하나 열어요. 이 접속(<code>conn</code>)으로 SQL을 보내요."),
                           ("conn.execute(sql).fetchone()[0]", "SQL을 보내고, 결과 첫 줄의 첫 값을 꺼내요.")]), "s1"),
    K.step(2, "테이블 두 개 만들기", "분류(<code>categories</code>)와 지식(<code>knowledge</code>) 테이블을 만들고, 지켜야 할 규칙을 같이 걸어요.",
           "<p>분류 이름을 지식마다 글자로 적으면, 이름을 바꿀 때 모든 지식을 고쳐야 하고 오타도 생겨요. 그래서 분류는 따로 두고, 지식은 분류의 <b>번호</b>만 가리켜요.</p>"
           + K.code_table([("bigint GENERATED ALWAYS AS IDENTITY", "번호를 DB가 1, 2, 3… 자동으로 붙여요. 직접 넣지 못하게 막아요."),
                           ("PRIMARY KEY", "기본 키. 행 하나를 가리키는 번호로, 중복과 빈 값이 안 돼요."),
                           ("NOT NULL", "비워 둘 수 없어요."), ("UNIQUE", "같은 값이 두 번 들어갈 수 없어요."),
                           ("CHECK (char_length(name) BETWEEN 1 AND 30)", "값이 조건을 만족해야 해요. 여기서는 1~30글자"),
                           ("REFERENCES categories (id)", "외래 키. <code>categories</code>에 있는 번호만 넣을 수 있어요."),
                           ("ON DELETE RESTRICT", "그 분류를 가리키는 지식이 있으면 분류를 못 지우게 막아요."),
                           ("conn.commit()", "지금까지 한 일을 확정해요. psycopg는 기본으로 트랜잭션 안에서 일해서, 확정하지 않으면 접속을 닫을 때 취소돼요.")])
           + K.try_("규칙을 어기는 요청 다섯 개(같은 분류 이름, 빈 이름, 없는 분류, 빈 제목, 쓰이는 분류 지우기)를 보내면?", "t2", hint="③까지 넣은 뒤에 시도했어요."), "s2"),
    K.step(3, "데이터 넣기", "분류 세 개를 넣고, 지식은 분류 <b>이름</b>으로 번호를 찾아 넣어요.",
           K.code_table([("INSERT INTO categories (name) VALUES (...), (...)", "여러 행을 한 번에 넣어요."),
                         ("RETURNING id, name", "넣은 행의 값을 바로 돌려받아요. DB가 붙인 번호를 알 수 있어요."),
                         ("INSERT INTO ... SELECT %s, %s, id FROM categories WHERE name = %s", "분류 번호를 외우지 않고, 이름으로 찾은 번호를 넣어요."),
                         ("%s", "값이 들어갈 자리. 값은 따로 넘겨서 psycopg가 안전하게 채워요. SQL에 값을 직접 이어 붙이면 따옴표 하나로 SQL이 깨질 수 있어요.")]), "s3"),
    K.step(4, "조회하기", "조건에 맞는 행만, 원하는 순서로 꺼내요.",
           "<p>SQL은 업무 질문으로 읽으면 쉬워요. \"<b>지식</b>에서(FROM) <b>분류 1인 것만</b>(WHERE) <b>번호, 제목을</b>(SELECT) <b>최신순으로</b>(ORDER BY) 줘.\"</p>"
           + K.code_table([("SELECT id, title FROM knowledge", "어떤 열을, 어느 테이블에서"), ("WHERE category_id = 1", "조건"), ("ORDER BY id DESC", "정렬. <code>DESC</code>는 큰 것부터")]), "s4"),
    K.step(5, "연결해서 읽기", "두 테이블을 번호로 이어서, 지식과 분류 이름을 함께 읽어요.",
           K.code_table([("JOIN categories c ON c.id = k.category_id", "지식의 분류 번호와 같은 번호의 분류를 옆에 붙여요. <code>k</code>, <code>c</code>는 테이블의 짧은 별명"),
                         ("LEFT JOIN", "왼쪽 테이블의 행은 짝이 없어도 모두 남겨요."),
                         ("GROUP BY c.name", "분류 이름별로 묶어서"), ("count(k.id)", "묶음마다 지식 수를 세요. 짝이 없으면 0")]), "s5"),
    K.step(6, "바꾸고 지우기", "<code>UPDATE</code>로 고치고 <code>DELETE</code>로 지워요. 바뀐 행은 <code>RETURNING</code>으로 확인해요.",
           K.code_table([("UPDATE knowledge SET content = %s WHERE id = %s", "조건에 맞는 행의 값을 바꿔요. <code>WHERE</code>를 빼면 <b>모든 행</b>이 바뀌어요."),
                         ("DELETE FROM knowledge WHERE id = 3", "조건에 맞는 행을 지워요. 여기도 <code>WHERE</code>를 꼭 확인해요.")]), "s6"),
    K.step(7, "트랜잭션", "여러 작업을 한 묶음으로 처리해서, 하나라도 실패하면 전부 취소해요.",
           "<p>\"새 분류를 만들고, 그 분류의 첫 지식을 넣는다\"는 두 작업이 함께 되어야 의미가 있어요. 이렇게 <b>전부 되거나 전부 안 되어야 하는 묶음</b>을 트랜잭션이라고 해요.</p>"
           + K.code_table([("with conn.transaction():", "블록 안의 작업을 한 묶음으로. 블록이 정상으로 끝나면 확정(commit), 오류가 나면 취소(rollback)돼요."),
                           ("except psycopg.Error as e", "DB 오류를 잡아요. 오류 종류는 <code>type(e).__name__</code>으로 볼 수 있어요.")])
           + K.try_("같은 두 작업을 트랜잭션 없이 하나씩 바로 확정하면?", "t7"), "s7"),
    K.step(8, "인덱스와 실행 계획", "행을 2만 개로 늘린 뒤, 제목으로 찾는 SQL을 DB가 어떻게 실행하는지 보고, 인덱스를 만들어 비교해요.",
           "<p>DB가 SQL을 어떻게 실행할지 세운 계획을 <b>실행 계획</b>이라고 해요. <code>EXPLAIN</code>을 앞에 붙이면 실행하지 않고 계획만 보여 주고, <code>EXPLAIN ANALYZE</code>는 실제로 실행해서 걸린 시간까지 보여 줘요.</p>"
           + K.code_table([("generate_series(1, 20000)", "1부터 20000까지의 숫자를 행으로 만들어요. 연습용 데이터를 빨리 넣으려고요."),
                           ("ANALYZE knowledge", "테이블의 값 분포를 다시 세어서, DB가 계획을 잘 세우게 해요."),
                           ("CREATE INDEX ... ON knowledge (title)", "제목으로 찾기 쉽게 정렬된 찾아보기를 만들어요. 대신 넣고 고칠 때마다 인덱스도 고쳐야 해서 쓰기가 조금 느려져요.")]), "s8"),
]
practice = [
    ("같은 스크립트를 두 번 실행한다", "②의 <code>CREATE TABLE</code>이 \"이미 있음\" 오류로 멈춰요. 테이블 구조를 바꾸고 기록하는 일은 Alembic 같은 마이그레이션 도구로 해요 (<a href=\"fastapi-postgres.html\">FastAPI와 PostgreSQL 연동</a>)."),
    ("API에서 DB를 쓴다", "⑥처럼 0행은 오류가 아니라서, API가 직접 404로 바꿔야 해요. 반대로 ②의 규칙 위반 오류는 409나 422 같은 응답으로 바꿔요."),
    ("NULL과 빈 글", "<code>NULL</code>은 \"값이 없음\", <code>''</code>는 \"빈 글이라는 값\"이에요. <code>WHERE x = NULL</code>은 아무것도 안 걸려요. <code>WHERE x IS NULL</code>로 써요."),
    ("인덱스를 어디에 거나", "자주 찾는 조건(<code>WHERE</code>)과 연결(<code>JOIN</code>)에 쓰는 열에 걸어요. 기본 키에는 자동으로 있어요. 많이 걸수록 쓰기가 느려지니 실행 계획을 보고 정해요."),
    ("인덱스는 뜻을 모른다", "여기서 만든 B-tree 인덱스는 \"같은 글자\"를 빨리 찾을 뿐, 뜻이 비슷한 글은 못 찾아요. 그건 벡터 검색의 일이에요 (<a href=\"indexing.html\">인덱싱</a>)."),
]
libs = [("①", "<code>psycopg.connect()</code>, <code>conn.execute(sql, params)</code>, <code>fetchone()</code>, <code>fetchall()</code>", "접속하고 SQL 보내기 (psycopg)"),
        ("②", "<code>CREATE TABLE</code>, <code>PRIMARY KEY</code>, <code>REFERENCES</code>, <code>CHECK</code>, <code>UNIQUE</code>, <code>conn.commit()</code>", "테이블과 규칙 (SQL)"),
        ("③~⑥", "<code>INSERT ... RETURNING</code>, <code>SELECT</code>, <code>JOIN</code>, <code>LEFT JOIN</code>, <code>GROUP BY</code>, <code>UPDATE</code>, <code>DELETE</code>", "넣고, 읽고, 바꾸기 (SQL)"),
        ("⑦", "<code>conn.transaction()</code>, <code>psycopg.Error</code>", "트랜잭션과 오류 잡기 (psycopg)"),
        ("⑧", "<code>EXPLAIN</code>, <code>EXPLAIN ANALYZE</code>, <code>CREATE INDEX</code>, <code>ANALYZE</code>, <code>generate_series()</code>", "실행 계획과 인덱스 (SQL)")]
K.build(ROOT / "topics" / "backend" / "postgres.html", title="PostgreSQL 기초", h1="PostgreSQL 기초: 지식을 테이블에 저장하고 꺼내기", track="백엔드",
        crumb="백엔드 &gt; 데이터베이스",
        one_line="<b>PostgreSQL</b>은 데이터를 표(테이블)로 저장하는 DB예요. 여러 요청이 <b>동시에 써도 안전</b>하고, <b>규칙을 DB가 지키고</b>, 여러 작업을 <b>한 묶음으로 확정하거나 취소</b>할 수 있어요.",
        why=why, whole=whole, steps=steps, code_file="examples/db/basics.py", practice=practice, libs=libs,
        next_rows=[("<a href=\"fastapi.html\">FastAPI로 지식 API 만들기</a>", "지금은 파일에 저장하는 API"),
                   ("<a href=\"fastapi-postgres.html\">FastAPI와 PostgreSQL 연동</a>", "API를 이 DB에 연결하기, Alembic"),
                   ("<a href=\"indexing.html\">인덱싱</a>", "pgvector로 벡터를 저장하고 뜻으로 찾기")],
        sources=["다루는 범위: Day18 (2026-11-11) PostgreSQL 입문, 지식 관리 DB",
                 "예제: <code>examples/db/basics.py</code> (기존 노트의 분류 세 개, 지식 세 개)",
                 "결과 값: <code>tools/pages/pg_basics.py</code>가 노트 전용 DB를 만들어 단계별로 실제 실행하고 지운 결과",
                 "공식 문서: <a href=\"https://www.postgresql.org/docs/current/tutorial.html\" target=\"_blank\">PostgreSQL Tutorial</a>, <a href=\"https://www.postgresql.org/docs/current/ddl-constraints.html\" target=\"_blank\">제약 조건</a>, <a href=\"https://www.postgresql.org/docs/current/using-explain.html\" target=\"_blank\">EXPLAIN</a>, <a href=\"https://www.psycopg.org/psycopg3/docs/basic/usage.html\" target=\"_blank\">psycopg</a>"])
print("ok")
