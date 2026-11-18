"""Day23 명령어 사전의 '입력 → 출력' 상자를 실제 실행 결과로 만든다 (Embedding · pgvector).

- NumPy 계산 · pg_config · PostgreSQL 18.6 + pgvector 0.8.7 SQL · psycopg + pgvector-python 을 이 PC에서 실제로 실행한다.
- DB 접속 정보: 환경변수 NOTES_PG_ENV 가 가리키는 .env (DB_HOST/DB_PORT/DB_USER/DB_PASSWORD) — 비밀번호는 어디에도 출력하지 않는다.
- 노트 전용 DB 두 개(sesac_notes_lab, sesac_notes_migration)를 새로 만들어 쓰고 끝나면 지운다. 다른 DB는 건드리지 않는다.
- E5 모델은 내려받지 않았다. sentence-transformers 동작(shape · 정규화 · 접두어)은 이 PC에 이미 캐시돼 있던
  다른 모델(all-MiniLM-L6-v2)을 HF_HUB_OFFLINE=1 로 불러 확인했다. 지식 검색 순위는 모델 벡터가 아니라
  2차원 예시 좌표 · 임의 단위 벡터로만 계산한다.
결과는 cmds/day23.html 각 항목의 첫 코드 블록 바로 뒤에 <div class="io"> 로 넣는다(다시 돌리면 교체).
실행: NOTES_PG_ENV=<.env 경로> C:/practice_assignment/.venv/Scripts/python.exe tools/io_day23.py
"""
import html
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from pgvector import Vector  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IO, RESULTS = {}, {}
LAB, MIG = "sesac_notes_lab", "sesac_notes_migration"


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
    s = str(s)
    return s.replace(SECRET, "********") if SECRET else s


def connect(db, **kw):
    return psycopg.connect(host=CRED.get("DB_HOST", "localhost"), port=int(CRED.get("DB_PORT", 5432)),
                           dbname=db, user=CRED["DB_USER"], password=SECRET, **kw)


def err(fn):
    try:
        return repr(fn())
    except Exception as e:  # noqa: BLE001
        msg = scrub(str(e)).strip().splitlines()[0]
        return f"{type(e).__name__}: {msg}"


def r4(x):
    return round(float(x), 4)


# ------------------------------------------------------------------ 버전
import sentence_transformers  # noqa: E402
import torch  # noqa: E402

IO["d23-embed-install"] = ("실제 가상환경에서 확인 (pip show 대신 import)", [
    ("import sentence_transformers, torch, numpy",
     f"sentence-transformers {sentence_transformers.__version__} · torch {torch.__version__} · numpy {np.__version__}",
     "이미 있는 패키지는 다시 설치하지 않음"),
    ("python --version", sys.version.split()[0], "이 버전 기준 결과"),
])
IO["d23-pip-pgvector"] = ("실제 가상환경에서 확인", [
    ("importlib.metadata.version(…)",
     f"psycopg {md.version('psycopg')} · pgvector {md.version('pgvector')} · python-dotenv {md.version('python-dotenv')}",
     "pgvector(Python) 는 타입 변환 도구 — 서버 확장(vector.dll)과 별개"),
])

# ------------------------------------------------------------------ ② 비교: NumPy
q, a = np.array([1.0, 0.0]), np.array([0.8, 0.6])
dot = float(q @ a)
cos = dot / (np.linalg.norm(q) * np.linalg.norm(a))
l2 = float(np.linalg.norm(q - a))
U, V = np.array([1.0, 0.0]), np.array([10.0, 10.0])
cosU = float(q @ U / (np.linalg.norm(q) * np.linalg.norm(U)))
cosV = float(q @ V / (np.linalg.norm(q) * np.linalg.norm(V)))
IO["d23-numpy-math"] = ("NumPy 실제 실행 · 2차원 설명용 벡터 (모델 출력 아님)", [
    ("q @ a   # q=(1,0), a=(0.8,0.6)", str(r4(dot)), "Dot Product — 같은 자리끼리 곱해 더함"),
    ("dot / (norm(q) * norm(a))", str(r4(cos)), "Cosine similarity — 두 벡터 길이가 1이라 Dot과 같음"),
    ("np.linalg.norm(q - a)", str(r4(l2)), "L2 distance = √0.4"),
    ("q @ U, q @ V   # U=(1,0), V=(10,10)", f"{q @ U:g}, {q @ V:g}", "Dot은 V가 앞 — 길이가 큰 벡터가 이김"),
    ("cos(q,U), cos(q,V)", f"{r4(cosU)}, {r4(cosV)}", "Cosine은 U가 앞 — 방향만 봄"),
])
RESULTS["math"] = {"dot": r4(dot), "cos": r4(cos), "l2": r4(l2), "cosU": r4(cosU), "cosV": r4(cosV)}

v = np.array([3.0, 4.0])
M = np.array([[3.0, 4.0], [10.0, 10.0], [0.8, 0.6]])
Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
c2 = float(Mn[0] @ Mn[1])
l2sq = float(np.linalg.norm(Mn[0] - Mn[1]) ** 2)
IO["d23-normalize"] = ("NumPy 실제 실행", [
    ("np.linalg.norm([3, 4])", f"{np.linalg.norm(v):g}", "L2 norm (원점에서 끝까지 길이)"),
    ("v / np.linalg.norm(v)", str([r4(x) for x in v / np.linalg.norm(v)]), "방향은 그대로, 길이만 1"),
    ("M / np.linalg.norm(M, axis=1, keepdims=True)", str([[r4(x) for x in row] for row in Mn]), "행마다 정규화 — keepdims 로 (3,1) 모양을 유지해 나눔"),
    ("np.linalg.norm(Mn, axis=1)", str([r4(x) for x in np.linalg.norm(Mn, axis=1)]), "모든 행 길이 1"),
    ("‖x−y‖²  vs  2 − 2·(x·y)   # 정규화된 두 행", f"{r4(l2sq)}  vs  {r4(2 - 2 * c2)}", "단위 벡터면 L2 순서 = Cosine 순서"),
])

with warnings.catch_warnings(record=True) as w_:
    warnings.simplefilter("always")
    z = np.zeros(2)
    zc = float(q @ z / (np.linalg.norm(q) * np.linalg.norm(z)))
    zwarn = type(w_[0].message).__name__ if w_ else "없음"
good = np.array([[0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
bad_nan = np.array([[0.6, np.nan]], dtype=np.float32)
bad_zero = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
bad_len = np.array([[3.0, 4.0]], dtype=np.float32)
IO["d23-check-vectors"] = ("NumPy 실제 실행 · 저장 전 검사에 쓰는 함수들", [
    ("q @ z / (norm(q) * norm(z))   # z=(0,0)", f"{zc} + {zwarn}", "영벡터는 분모 0 → nan. 조용히 0점으로 바꾸지 않음"),
    ("good.shape == (2, 2)", str(good.shape == (2, 2)), "개수 · 차원 검사"),
    ("np.isfinite(bad_nan).all()", str(bool(np.isfinite(bad_nan).all())), "NaN · Inf 가 하나라도 있으면 False"),
    ("np.linalg.norm(bad_zero, axis=1)", str([r4(x) for x in np.linalg.norm(bad_zero, axis=1)]), "길이 0 인 행 = 영벡터"),
    ("np.allclose(np.linalg.norm(good, axis=1), 1, rtol=0, atol=1e-5)", str(bool(np.allclose(np.linalg.norm(good, axis=1), 1, rtol=0, atol=1e-5))), "정규화 확인 통과"),
    ("np.allclose(np.linalg.norm([[3, 4]], axis=1), 1, …)", str(bool(np.allclose(np.linalg.norm(bad_len, axis=1), 1, rtol=0, atol=1e-5))), "길이 5 — 정규화 안 된 벡터"),
])

# ------------------------------------------------------------------ ③ Sentence Transformers (캐시된 다른 모델로 동작만)
from sentence_transformers import SentenceTransformer  # noqa: E402

LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
t0 = time.perf_counter()
st = SentenceTransformer(LOCAL, device="cpu")
dim = st.get_embedding_dimension()
mods = " → ".join(type(m).__name__ for m in st)
KN = [(101, "실습 파일 보관", "실습 파일은 개인 폴더에 보관합니다.", "course", "knowledge_v1"),
      (102, "서버 실행", "프로젝트 폴더에서 서버를 실행합니다.", "development", "knowledge_v1"),
      (103, "질문 작성", "오류 메시지와 시도한 작업을 함께 적습니다.", "support", "knowledge_v1"),
      (104, "수업 녹화", "녹화가 제공되는 수업은 강의실 안내 페이지에서 확인합니다.", "course", "knowledge_search_v2"),
      (105, "과제 제출", "과제 제출 위치는 해당 과제의 안내 페이지에서 확인합니다.", "course", "knowledge_search_v2"),
      (106, "데이터베이스 연결", "PostgreSQL 접속 정보는 로컬 설정 파일에서 읽습니다.", "development", "knowledge_search_v2"),
      (107, "패키지 설치", "필요한 Python 패키지는 기존 가상환경에 추가합니다.", "development", "knowledge_search_v2"),
      (108, "문제 해결 안내", "문제를 보고할 때 재현한 단계와 오류 메시지를 기록합니다.", "support", "knowledge_search_v2"),
      (109, "결과 검토", "검색된 자료는 원문과 실제 질문의 요구를 대조합니다.", "support", "knowledge_search_v2")]
passages = ["passage: " + t + " " + c for _, t, c, _, _ in KN]
docs = st.encode(passages, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)
qv = st.encode(["query: 서버는 어디에서 실행하나요?"], normalize_embeddings=True, convert_to_numpy=True)
raw = st.encode(passages[:2], normalize_embeddings=False, convert_to_numpy=True)
one = st.encode("query: 서버는 어디에서 실행하나요?", normalize_embeddings=True)
pq = st.encode(["query: 서버 실행", "passage: 서버 실행", "서버 실행"], normalize_embeddings=True)
IO["d23-st-load"] = (f"실제 실행 · 이 PC에 캐시돼 있던 {LOCAL} 로 동작만 확인 (E5 아님 · 오프라인)", [
    ("SentenceTransformer(MODEL_ID, device='cpu')", type(st).__name__, "모델 객체 한 번 생성 → 계속 재사용"),
    ("model.get_embedding_dimension()", str(dim), "출력 벡터 길이 (E5-small 도 모델 카드상 384) · 예전 이름 get_sentence_embedding_dimension 은 FutureWarning"),
    ("[type(m).__name__ for m in model]", mods, "모듈 구성 — 이 모델은 끝에 Normalize 층이 있음"),
    ("model.max_seq_length", str(st.max_seq_length), "입력 토큰 상한 — 모델마다 다름 (E5-small 은 512)"),
    ("model.device", str(st.device), "GPU 없이 CPU 추론"),
])
IO["d23-st-encode"] = ("실제 실행 · 캐시된 all-MiniLM-L6-v2 · 지식 9개 · 숫자는 E5 결과가 아님", [
    ("model.encode(passages, normalize_embeddings=True, convert_to_numpy=True, batch_size=8)",
     f"shape {docs.shape} · dtype {docs.dtype}", "(지식 개수, 차원)"),
    ("model.encode([query], …)", f"shape {qv.shape}", "리스트로 넣으면 질문 1개도 2차원 (1, 384)"),
    ("model.encode(query_str, …)   # 리스트 아님", f"shape {one.shape}", "문자열 하나면 1차원 (384,) — DB에 넣을 qvec 모양"),
    ("np.linalg.norm(docs, axis=1).round(4)", str(sorted(set(np.linalg.norm(docs, axis=1).round(4).tolist()))), "normalize_embeddings=True → 모두 길이 1"),
    ("np.linalg.norm(encode(…, normalize_embeddings=False), axis=1)", str([r4(x) for x in np.linalg.norm(raw, axis=1)]), "이 모델은 Normalize 층이 있어 꺼도 1 — 모델마다 다르니 norm 으로 직접 확인"),
    ("(qv @ docs.T).shape", str((qv @ docs.T).shape), "질문 1개 × 지식 9개 점수"),
    ("cos('query: 서버 실행', 'passage: 서버 실행')", str(r4(pq[0] @ pq[1])), "접두어만 달라도 다른 벡터 → 질문·문서 규칙을 섞지 않음"),
])
RESULTS["st"] = {"dim": dim, "max_seq": st.max_seq_length}

# ------------------------------------------------------------------ ④ pg_config
PGBIN = Path(r"C:\Program Files\PostgreSQL\18\bin\pg_config.exe")
pgc = []
for flag, note in [("--version", "서버 버전 — 빌드 대상과 같아야 함"), ("--includedir-server", "postgres.h 가 있는 헤더 폴더"),
                   ("--pkglibdir", "vector.dll 이 놓일 폴더"), ("--sharedir", "share\\extension 의 상위 — vector.control · SQL")]:
    out = subprocess.run([str(PGBIN), flag], capture_output=True, text=True).stdout.strip()
    pgc.append((f'pg_config.exe {flag}', out, note))
IO["d23-pg-config"] = ("이 PC에서 실제 실행", pgc)
ext_dir = Path(subprocess.run([str(PGBIN), "--sharedir"], capture_output=True, text=True).stdout.strip()) / "extension"
lib_dir = Path(subprocess.run([str(PGBIN), "--pkglibdir"], capture_output=True, text=True).stdout.strip())
IO["d23-nmake"] = ("이 PC의 설치 결과 확인 (빌드 자체는 다시 하지 않음)", [
    ('dir "%PGROOT%\\lib\\vector.dll"', "있음" if (lib_dir / "vector.dll").exists() else "없음", "서버가 불러올 확장 라이브러리"),
    ('dir "%PGROOT%\\share\\extension\\vector.control"', "있음" if (ext_dir / "vector.control").exists() else "없음", "default_version 이 적힌 제어 파일"),
    ('dir "…\\vector--*.sql"', f"{len(list(ext_dir.glob('vector--*.sql')))}개 (예: vector--0.8.7.sql)", "버전별 설치 · 업그레이드 SQL"),
])

# ------------------------------------------------------------------ DB 준비 (노트 전용)
with connect("postgres", autocommit=True) as ad:
    for db in (LAB, MIG):
        ad.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
        ad.execute(f'CREATE DATABASE "{db}"')

try:
    with connect(LAB, autocommit=True) as c:
        ver = c.execute("SELECT version()").fetchone()[0]
        db_user = c.execute("SELECT current_database()").fetchone()[0]
        before = c.execute("SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name='vector'").fetchone()
        IO["d23-pg-available"] = ("노트 전용 DB에서 실제 실행", [
            ("SELECT version();", ver.split(",")[0], "서버 버전"),
            ("SELECT current_database();", db_user, "지금 연결된 DB — 확장은 DB마다 따로"),
            ("pg_available_extensions WHERE name='vector'", str(before), "행이 있음 = 서버 파일 설치됨 · installed_version None = 이 DB엔 아직"),
        ])
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        after = c.execute("SELECT extname, extversion FROM pg_extension WHERE extname='vector'").fetchone()
        notices = []
        c.add_notice_handler(lambda d: notices.append(d.message_primary))
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        with connect(MIG, autocommit=True) as c2:
            other = err(lambda: c2.execute("SELECT '[1,2,3]'::vector").fetchone())
            other_inst = c2.execute("SELECT installed_version FROM pg_available_extensions WHERE name='vector'").fetchone()
        IO["d23-create-extension"] = ("노트 전용 DB에서 실제 실행", [
            ("CREATE EXTENSION IF NOT EXISTS vector;", "CREATE EXTENSION", "이 DB에 vector 타입 · 연산자 등록"),
            ("SELECT extname, extversion FROM pg_extension …", str(after), "이 DB에 활성화된 버전"),
            ("CREATE EXTENSION IF NOT EXISTS vector;   # 한 번 더", f"NOTICE: {notices[0] if notices else '-'}", "이미 있으면 건너뜀 — 오류 아님"),
            ("다른 DB에서 SELECT '[1,2,3]'::vector", other, "확장은 연결한 DB에만 — 다른 DB는 따로 CREATE EXTENSION"),
            ("다른 DB의 installed_version", str(other_inst[0]), "서버 파일은 있어도 그 DB엔 미설치"),
        ])

        row = c.execute("""SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector,
                                  '[1,0,0]'::vector <=> '[0,1,0]'::vector,
                                  '[1,0,0]'::vector <=> '[-1,0,0]'::vector""").fetchone()
        l2r = c.execute("SELECT '[1,0]'::vector <-> '[0.8,0.6]'::vector, '[1,0]'::vector <#> '[0.8,0.6]'::vector, 1 - ('[1,0]'::vector <=> '[0.8,0.6]'::vector)").fetchone()
        zero = c.execute("SELECT '[1,0,0]'::vector <=> '[0,0,0]'::vector").fetchone()[0]
        mism = err(lambda: c.execute("SELECT '[1,2]'::vector <=> '[1,2,3]'::vector").fetchone())
        IO["d23-operators"] = ("pgvector 0.8.7 실제 실행 · 설명용 벡터", [
            ("[1,0,0] <=> [1,0,0] · [0,1,0] · [-1,0,0]", f"{row[0]:g} · {row[1]:g} · {row[2]:g}", "Cosine distance: 같은 방향 0 · 직각 1 · 반대 2"),
            ("[1,0] <-> [0.8,0.6]", str(r4(l2r[0])), "L2 distance — NumPy 결과와 같음"),
            ("[1,0] <#> [0.8,0.6]", str(r4(l2r[1])), "음의 내적(−0.8) — 작을수록 내적이 큼"),
            ("1 - ([1,0] <=> [0.8,0.6])", str(r4(l2r[2])), "거리 → 유사도로 바꿔 표시만"),
            ("[1,0,0] <=> [0,0,0]", str(zero), "영벡터와의 Cosine 은 정의 안 됨 → NaN"),
            ("[1,2] <=> [1,2,3]", mism, "차원이 다르면 비교 자체가 오류"),
        ])
        RESULTS["ops"] = {"cos": [float(x) for x in row], "l2": r4(l2r[0]), "nip": r4(l2r[1])}

        c.execute("BEGIN")
        c.execute("CREATE TEMP TABLE day23_vector_check (id integer PRIMARY KEY, label text NOT NULL, embedding vector(3) NOT NULL) ON COMMIT DROP")
        c.execute("INSERT INTO day23_vector_check (id,label,embedding) VALUES (1,'같은 방향','[1,0,0]'),(2,'가까운 방향','[0.8,0.6,0]'),(3,'직각 방향','[0,1,0]')")
        tmp = c.execute("SELECT id,label, embedding <=> '[1,0,0]'::vector AS distance FROM day23_vector_check ORDER BY embedding <=> '[1,0,0]'::vector ASC LIMIT 3").fetchall()
        c.execute("COMMIT")
        gone = err(lambda: c.execute("SELECT 1 FROM day23_vector_check").fetchone())
        IO["d23-temp-topk"] = ("노트 전용 DB에서 실제 실행", [
            ("ORDER BY embedding <=> '[1,0,0]' ASC LIMIT 3", " → ".join(f"{i}({lbl}, {d:.4g})" for i, lbl, d in tmp), "거리가 작은 순 — 예상 1 → 2 → 3"),
            ("COMMIT 뒤 SELECT … FROM day23_vector_check", gone, "ON COMMIT DROP — 기존 테이블은 안 건드리는 검사"),
        ])

    # ---------- psycopg + register_vector
    with connect(LAB) as conn:
        plain = conn.execute("SELECT '[1,0,0]'::vector").fetchone()[0]
        register_vector(conn)
        test = Vector([1.0, 0.0, 0.0])
        dims, dist = conn.execute("SELECT vector_dims(%s::vector), %s::vector <=> %s::vector", (test, test, test)).fetchone()
        back = conn.execute("SELECT '[1,0,0]'::vector").fetchone()[0]
        arr = conn.execute("SELECT %s::vector <-> %s::vector", (np.array([1.0, 0.0]), np.array([0.8, 0.6]))).fetchone()[0]
        IO["d23-register-vector"] = ("노트 전용 DB에서 실제 실행", [
            ("register_vector 전: SELECT '[1,0,0]'::vector", f"{type(plain).__name__} {plain!r}", "등록 전엔 그냥 문자열로 옴"),
            ("register_vector(conn)", "None", "이 연결에 vector ↔ NumPy 변환 등록 (연결마다)"),
            ("vector_dims(%s::vector), %s <=> %s   # Vector([1,0,0])", f"{dims}, {float(dist):g}", "assert dims == 3 and abs(dist) < 1e-6 통과"),
            ("등록 후: SELECT '[1,0,0]'::vector", f"{type(back).__name__} {back!r}", "pgvector.Vector 객체로"),
            ("row.to_numpy().dtype", str(back.to_numpy().dtype), "NumPy 배열이 필요하면 .to_numpy()"),
            ("np.array([1,0]) <-> np.array([0.8,0.6])", str(r4(arr)), "NumPy 배열을 그대로 파라미터로"),
        ])

        # ---------- ⑤ 검색 테이블
        conn.execute("""CREATE TABLE IF NOT EXISTS day23_search_docs (
              id integer PRIMARY KEY, title text NOT NULL, content text NOT NULL, category text NOT NULL,
              source_version text NOT NULL, embedding_model text NOT NULL, embedding vector(384) NOT NULL)""")
        cols = conn.execute("""SELECT column_name, data_type, udt_name FROM information_schema.columns
                               WHERE table_name = 'day23_search_docs' ORDER BY ordinal_position""").fetchall()
        conn.commit()
        bad3 = err(lambda: conn.execute("INSERT INTO day23_search_docs VALUES (1,'t','c','x','v','m',%s)", (np.array([1.0, 0.0, 0.0]),)))
        conn.rollback()
        nanv = np.zeros(384); nanv[0] = np.nan
        badn = err(lambda: conn.execute("INSERT INTO day23_search_docs VALUES (1,'t','c','x','v','m',%s)", (nanv,)))
        conn.rollback()
        aborted = None
        try:
            conn.execute("INSERT INTO day23_search_docs VALUES (1,'t','c','x','v','m',%s)", (np.array([1.0]),))
        except Exception:  # noqa: BLE001
            aborted = err(lambda: conn.execute("SELECT 1").fetchone())
        conn.rollback()
        IO["d23-table"] = ("노트 전용 DB에서 실제 실행", [
            ("information_schema.columns … day23_search_docs", " · ".join(f"{n}:{u}" for n, _, u in cols), "embedding 은 data_type USER-DEFINED · udt_name vector"),
            ("INSERT … embedding = 3차원 배열", bad3, "vector(384) 가 차원을 검사"),
            ("INSERT … embedding 에 NaN", badn, "NaN 도 거부"),
            ("실패 직후 같은 트랜잭션에서 SELECT 1", aborted, "ROLLBACK 전까지 이 트랜잭션은 막힘"),
        ])
        RESULTS["cols"] = [(n, u) for n, _, u in cols]

        rng = np.random.default_rng(23)
        fake = rng.normal(size=(len(KN), 384)).astype(np.float32)
        fake /= np.linalg.norm(fake, axis=1, keepdims=True)
        sql = """INSERT INTO day23_search_docs (id, title, content, category, source_version, embedding_model, embedding)
                 VALUES (%s, %s, %s, %s, %s, %s, %s)
                 ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, content = EXCLUDED.content,
                   category = EXCLUDED.category, source_version = EXCLUDED.source_version,
                   embedding_model = EXCLUDED.embedding_model, embedding = EXCLUDED.embedding"""
        MODEL_ID = "intfloat/multilingual-e5-small"
        for (i, t, ct, cat, ver_), vec in zip(KN, fake):
            conn.execute(sql, (i, t, ct, cat, ver_, MODEL_ID, vec))
        conn.commit()
        n1 = conn.execute("SELECT count(*) FROM day23_search_docs").fetchone()[0]
        cur = conn.execute(sql, (102, "서버 실행", "프로젝트 폴더에서 서버를 실행합니다. (수정)", "development", "knowledge_search_v2", MODEL_ID, fake[1]))
        rc = cur.rowcount
        conn.commit()
        n2, c102, v102 = conn.execute("SELECT (SELECT count(*) FROM day23_search_docs), content, source_version FROM day23_search_docs WHERE id=102").fetchone()
        IO["d23-upsert"] = ("노트 전용 DB에서 실제 실행 · embedding 은 임의 단위 벡터 (모델 출력 아님)", [
            ("지식 9개 INSERT … ON CONFLICT → SELECT count(*)", str(n1), "행 하나 = 원문 + Metadata + 벡터"),
            ("id=102 로 다시 실행 → cur.rowcount", str(rc), "충돌 → UPDATE 한 행"),
            ("SELECT count(*) · id=102 의 content, source_version", f"{n2} · {c102!r}, {v102!r}", "개수는 그대로, 같은 행이 갱신"),
        ])
        # 원래 상태로
        conn.execute(sql, (102, "서버 실행", "프로젝트 폴더에서 서버를 실행합니다.", "development", "knowledge_v1", MODEL_ID, fake[1]))
        conn.commit()

        # ---------- Top-k · 필터 : 2차원 예시 좌표
        FIX = [("A", 1, 0, "course"), ("B", 0.8, 0.6, "development"), ("C", 0.4, 0.92, "support"),
               ("D", -0.3, 0.95, "course"), ("E", -1, 0, "support"), ("F", 0.92, -0.39, "development")]
        conn.execute("CREATE TABLE fixture2d (id text PRIMARY KEY, category text NOT NULL, embedding vector(2) NOT NULL)")
        for i, x, y, cat in FIX:
            conn.execute("INSERT INTO fixture2d VALUES (%s,%s,%s)", (i, cat, np.array([x, y])))
        conn.commit()
        qvec = np.array([1.0, 0.0])

        def topk(k, category=None):
            if category is None:
                return conn.execute("""SELECT id, category, embedding <=> %s AS cosine_distance FROM fixture2d
                                       ORDER BY embedding <=> %s ASC LIMIT %s""", (qvec, qvec, k)).fetchall()
            return conn.execute("""SELECT id, category, embedding <=> %s AS distance FROM fixture2d WHERE category = %s
                                   ORDER BY embedding <=> %s ASC LIMIT %s""", (qvec, category, qvec, k)).fetchall()

        fmt = lambda rows: " → ".join(f"{i}({d:.4f})" for i, _, d in rows) if rows else "[]"  # noqa: E731
        all_rows = topk(10)
        RESULTS["fixture"] = [(i, x, y, cat) for i, x, y, cat in FIX]
        RESULTS["fixture_dist"] = {i: float(d) for i, _, d in all_rows}
        IO["d23-topk"] = ("노트 전용 DB에서 실제 실행 · 2차원 예시 좌표 A~F, q=(1,0)", [
            ("k = 3", fmt(topk(3)), "거리가 작은 3개"),
            ("k = 10   # 자료는 6개", f"{len(all_rows)}행: " + fmt(all_rows), "k 는 '최대' 개수 — 자료가 적으면 그만큼만"),
            ("ASC 대신 DESC 로 정렬 (k=3)", " → ".join(f"{i}" for i, _, _ in conn.execute(
                "SELECT id, category, embedding <=> %s FROM fixture2d ORDER BY embedding <=> %s DESC LIMIT 3", (qvec, qvec)).fetchall()),
             "가장 먼 것부터 — 거리 연산자는 ASC"),
        ])
        IO["d23-filter"] = ("노트 전용 DB에서 실제 실행 · 같은 예시 좌표", [
            ("category = 'development', k = 3", fmt(topk(3, "development")), "조건에 맞는 행끼리만 거리 비교 — 2개뿐이라 2행"),
            ("category = 'support', k = 3", fmt(topk(3, "support")), "C 가 E 보다 가까움"),
            ("category = 'billing', k = 3", fmt(topk(3, "billing")), "정상 실행 · 결과 0건 (오류 아님)"),
            ("파라미터 순서 (qvec, category, qvec, k)", "SQL 안 %s 4개와 같은 순서", "값만 바인딩 — 테이블 · 열 이름은 %s 로 못 넣음"),
        ])

        # ---------- ⑥ 인덱스 · EXPLAIN (9건)
        conn.execute("CREATE INDEX IF NOT EXISTS day23_search_docs_hnsw ON day23_search_docs USING hnsw (embedding vector_cosine_ops)")
        conn.execute("ANALYZE day23_search_docs")
        conn.commit()
        idx = conn.execute("SELECT indexdef FROM pg_indexes WHERE indexname='day23_search_docs_hnsw'").fetchone()[0]
        q384 = fake[1]
        plan9 = [ln for ln, in conn.execute("EXPLAIN SELECT id FROM day23_search_docs ORDER BY embedding <=> %s ASC LIMIT %s", (q384, 3)).fetchall()]
        plan9_l2 = None
        conn.execute("SET enable_seqscan = off")
        plan9_force = [ln for ln, in conn.execute("EXPLAIN SELECT id FROM day23_search_docs ORDER BY embedding <=> %s ASC LIMIT %s", (q384, 3)).fetchall()]
        plan9_l2 = [ln for ln, in conn.execute("EXPLAIN SELECT id FROM day23_search_docs ORDER BY embedding <-> %s ASC LIMIT %s", (q384, 3)).fetchall()]
        conn.execute("RESET enable_seqscan")
        conn.commit()

        # ---------- 임의 단위 벡터 20,000개 · 384차원 (ANN 동작 관찰용)
        N, DIM = 20000, 384
        rng2 = np.random.default_rng(7)
        centers = rng2.normal(size=(200, DIM)).astype(np.float32)          # 주제 200개 흉내 (덩어리 구조)
        base = centers[rng2.integers(0, 200, N)] + 0.6 * rng2.normal(size=(N, DIM)).astype(np.float32)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        cats = np.where(np.arange(N) % 100 == 0, "rare", "common")   # rare = 1% (200행)
        conn.execute("CREATE TABLE bench (id integer PRIMARY KEY, category text NOT NULL, embedding vector(384) NOT NULL)")
        with conn.cursor() as cur:
            with cur.copy("COPY bench (id, category, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["int4", "text", "vector"])
                for i in range(N):
                    cp.write_row((i, str(cats[i]), base[i]))
        conn.commit()
        t0 = time.perf_counter()
        conn.execute("CREATE INDEX bench_hnsw ON bench USING hnsw (embedding vector_cosine_ops)")
        build_s = time.perf_counter() - t0
        conn.execute("ANALYZE bench")
        conn.commit()
        ef_default = conn.execute("SHOW hnsw.ef_search").fetchone()[0]
        it_default = conn.execute("SHOW hnsw.iterative_scan").fetchone()[0]
        mst_default = conn.execute("SHOW hnsw.max_scan_tuples").fetchone()[0]
        qs = base[rng2.choice(N, 50, replace=False)] + 0.05 * rng2.normal(size=(50, DIM)).astype(np.float32)   # 자료 근처 질의
        qs /= np.linalg.norm(qs, axis=1, keepdims=True)
        K = 10
        exact = [set(np.argsort(-(base @ qq))[:K].tolist()) for qq in qs]
        q0 = qs[0]
        planN = [ln for ln, in conn.execute("EXPLAIN SELECT id FROM bench ORDER BY embedding <=> %s ASC LIMIT %s", (q0, K)).fetchall()]
        conn.execute("SET enable_indexscan = off")
        planN_exact = [ln for ln, in conn.execute("EXPLAIN SELECT id FROM bench ORDER BY embedding <=> %s ASC LIMIT %s", (q0, K)).fetchall()]
        sql_exact = [set(r[0] for r in conn.execute("SELECT id FROM bench ORDER BY embedding <=> %s ASC LIMIT %s", (qq, K)).fetchall()) for qq in qs[:5]]
        conn.execute("RESET enable_indexscan")
        conn.commit()
        exact_match = all(a == b for a, b in zip(sql_exact, exact[:5]))

        def run_ann(ef, iterative="off", category=None):
            conn.execute(f"SET hnsw.ef_search = {int(ef)}")
            conn.execute(f"SET hnsw.iterative_scan = {iterative}")
            hits, counts, t = [], [], time.perf_counter()
            for qq, ex in zip(qs, exact):
                if category:
                    rows = conn.execute("SELECT id FROM bench WHERE category = %s ORDER BY embedding <=> %s ASC LIMIT %s", (category, qq, K)).fetchall()
                else:
                    rows = conn.execute("SELECT id FROM bench ORDER BY embedding <=> %s ASC LIMIT %s", (qq, K)).fetchall()
                got = {r[0] for r in rows}
                counts.append(len(got))
                hits.append(len(got & ex) / K)
            ms = (time.perf_counter() - t) / len(qs) * 1000
            conn.execute("RESET hnsw.ef_search"); conn.execute("RESET hnsw.iterative_scan")
            return float(np.mean(hits)), counts, ms

        sweep = {}
        for ef in (10, 20, 40, 100, 200):
            rec, _, ms = run_ann(ef)
            sweep[ef] = (round(rec, 3), round(ms, 2))
        _, rare_off, _ = run_ann(40, "off", "rare")
        _, rare_on, _ = run_ann(40, "relaxed_order", "rare")
        conn.commit()
        RESULTS["ann"] = {"N": N, "K": K, "sweep": sweep, "rare_off": rare_off, "rare_on": rare_on,
                          "ef_default": ef_default, "build_s": round(build_s, 1)}

        def node(plan):
            return plan[0].split("  (")[0].strip() + " / " + next((ln.strip().split("  (")[0] for ln in plan[1:] if "Scan" in ln), "-")

        IO["d23-hnsw"] = ("노트 전용 DB에서 실제 실행", [
            ("CREATE INDEX … USING hnsw (embedding vector_cosine_ops)", idx.split(" ON ")[0] + " … " + idx.split("USING ")[1], "Cosine 거리(<=>)용 HNSW"),
            (f"덩어리 구조 임의 벡터 {N:,}개 · 384차원에 같은 인덱스", f"생성 {build_s:.1f}초 (이 PC)", "데이터가 많을수록 구축 시간 · 메모리 비용"),
            ("SHOW hnsw.ef_search", ef_default, "질의 때 살펴볼 후보 폭 기본값"),
        ])
        IO["d23-explain"] = ("노트 전용 DB에서 실제 실행 · EXPLAIN (플랜 노드만 발췌)", [
            ("9건 테이블 · ORDER BY embedding <=> q LIMIT 3", node(plan9), "인덱스가 있어도 9건이면 Seq Scan + 정렬이 더 쌈"),
            ("9건 · SET enable_seqscan = off (관찰용)", node(plan9_force), "강제로 막으면 HNSW Index Scan"),
            ("9건 · 같은 조건, 연산자만 <-> (L2)", node(plan9_l2), "vector_cosine_ops 인덱스는 L2 정렬에 못 씀"),
            (f"{N:,}건 · <=> LIMIT 10", node(planN), "행이 많으면 플래너가 HNSW 를 고름"),
            (f"{N:,}건 · SET enable_indexscan = off", node(planN_exact), "전체 거리 계산 = Exact Search 기준선"),
            ("Exact SQL 결과 == NumPy 전수 계산 (질의 5개)", str(exact_match), "기준선이 맞는지 먼저 확인"),
        ])
        IO["d23-ef-search"] = (f"노트 전용 DB에서 실제 실행 · 덩어리 구조 임의 벡터 {N:,}개(모델 출력 아님) · 질의 50개 · k=10 · 시간은 이 PC 기준", [
            *[(f"SET hnsw.ef_search = {ef}", f"recall@10 {rec:.3f} · {ms:.2f} ms/질의", "Exact Top-10 중 찾은 비율") for ef, (rec, ms) in sweep.items()],
            ("WHERE category='rare'(1%) · ef_search 40 · iterative_scan off",
             f"반환 행 수 평균 {np.mean(rare_off):.2f} (최소 {min(rare_off)} · 최대 {max(rare_off)})", "인덱스가 본 40개 후보 중 rare 가 거의 없음 → k 보다 적게"),
            ("같은 조건 · SET hnsw.iterative_scan = relaxed_order",
             f"반환 행 수 평균 {np.mean(rare_on):.2f} (최소 {min(rare_on)})", "모자라면 더 탐색 (0.8.x 기능)"),
            ("SHOW hnsw.max_scan_tuples", mst_default, "반복 탐색 상한 — 완전 보장은 아님"),
            ("SHOW hnsw.iterative_scan", it_default, "기본은 꺼져 있음"),
        ])
finally:
    with connect("postgres", autocommit=True) as ad:
        for db in (LAB, MIG):
            ad.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')
        left = ad.execute("SELECT count(*) FROM pg_database WHERE datname LIKE 'sesac_notes%'").fetchone()[0]
    print("note DBs dropped · remaining:", left)


# ------------------------------------------------------------------ 출력 · 넣기
def box(title, rows):
    tr = "\n".join(f'    <tr><td><code>{html.escape(i)}</code></td><td><code>{html.escape(o)}</code></td>'
                   f'<td>{html.escape(n)}</td></tr>' for i, o, n in rows)
    return (f'<div class="io">\n  <div class="io-title">▶ 실제로 돌려 보면 <small>{html.escape(title)}</small></div>\n'
            f'  <table class="io-table">\n    <tr><th>📥 입력</th><th>📤 출력</th><th>뜻</th></tr>\n{tr}\n  </table>\n</div>')


if __name__ == "__main__":
    for cid, (title, rows) in IO.items():
        print(f"== {cid}  [{title}]")
        for i, o, n in rows:
            print(f"   {i}\n      → {o}   ({n})")
    (Path(tempfile.gettempdir()) / "d23_results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, default=str), encoding="utf-8")
    frag = ROOT / "cmds" / "day23.html"
    if frag.exists():
        s = frag.read_text(encoding="utf-8")
        s = re.sub(r'<div class="io">.*?\n</div>\n', '', s, flags=re.S)
        missing = []
        for cid, (title, rows) in IO.items():
            m = re.search(rf'<section class="cmd" id="{re.escape(cid)}".*?</section>', s, re.S)
            if not m:
                missing.append(cid)
                continue
            sec = m.group(0)
            kk = sec.find("</code></pre>\n")
            kk = kk + len("</code></pre>\n") if kk >= 0 else sec.find("</h3>\n") + len("</h3>\n")
            s = s[:m.start() + kk] + box(title, rows) + "\n" + s[m.start() + kk:]
        frag.write_text(s, encoding="utf-8")
        print("inserted", len(IO) - len(missing), "missing", missing)
