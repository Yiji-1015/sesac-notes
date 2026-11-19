"""주제 페이지 '인덱싱'(topics/backend/indexing.html)의 코드와 실행 결과를 만든다.

examples/ingest/ingest.py 의 ⑭~⑱ 단계를 실제로 실행한다. DB 는 노트 전용 DB(sesac_notes_lab)를
만들어 쓰고 마지막에 지운다 (tools/notes_pg.py, 비밀번호는 출력하지 않음).
결과: topics/backend/indexing-data.js
실행: NOTES_PG_ENV=<.env 경로> HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/io_indexing.py
"""
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import notes_pg  # noqa: E402
import stepcode  # noqa: E402

import numpy as np  # noqa: E402
import pgvector  # noqa: E402
import psycopg  # noqa: E402
import sentence_transformers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "ingest"
code = stepcode.parse((EX / "ingest.py").read_text(encoding="utf-8"))
RANGE = (14, 18)
out = {"steps": {}, "exp": {}}
W = stepcode.workdir(EX)
run = lambda k, wd=W, patch=None: stepcode.run(code, k, wd, patch)  # noqa: E731
Q = ["대출 기간은 며칠인가요?", "회원증을 잃어버렸어요", "노트북을 쓸 수 있는 자리가 있나요?"]


def edited_dir(old="대출 기간은 14일이며", new="대출 기간은 21일이며"):
    d = stepcode.workdir(EX)
    p = d / "raw" / "guide.md"
    p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    return d


V2 = [("VERSION = 1", "VERSION = 2")]


def rows(res):
    return [{"file": r[0], "location": r[1], "heading": r[2], "body": r[3], "distance": round(float(r[4]), 3)} for r in res]


def first_line_has(res, word):
    return [r for r in res if word in r["body"]]


try:
    # ---- ⑭ 벡터 ----------------------------------------------------------------
    ns = run(14)
    v = ns["vectors"]
    qv = ns["embed"]([Q[0]], "query")[0]
    c0 = ns["chunks"][0]
    sims = (ns["vectors"] @ qv).tolist()
    out["steps"][14] = {"shape": list(v.shape), "dtype": str(v.dtype),
                        "norm": round(float(np.linalg.norm(v[0])), 4), "first": [round(float(x), 4) for x in v[0][:6]],
                        "embed_text": ns["embed_text"](c0), "model": ns["MODEL"],
                        "inputs": [f"passage: {ns['embed_text'](c0)}", f"query: {Q[0]}"],
                        "q": Q[0],
                        "dist": [{"file": c["file"], "heading": c["heading"], "where": c["where"], "body": c["body"].splitlines()[0],
                                  "cos": round(s, 3), "dist": round(1 - s, 3)} for c, s in zip(ns["chunks"], sims)]}
    noprefix = ns["model"].encode([Q[0]], normalize_embeddings=True)[0]
    out["exp"]["noprefix"] = [{"heading": c["heading"], "file": c["file"], "with": round(float(a), 3), "without": round(float(b), 3)}
                              for c, a, b in zip(ns["chunks"], (ns["vectors"] @ qv).tolist(), (ns["vectors"] @ noprefix).tolist())]

    # ---- ⑮~⑱ 한 DB 에서 차례대로 ------------------------------------------------
    notes_pg.fresh()
    ns = run(15)
    cols = ns["conn"].execute("""SELECT table_name, column_name, data_type, udt_name FROM information_schema.columns
                                 WHERE table_schema = 'public' ORDER BY table_name, ordinal_position""").fetchall()
    out["steps"][15] = {"cols": [list(c) for c in cols],
                        "pg": ns["conn"].execute("SHOW server_version").fetchone()[0],
                        "pgvector": ns["conn"].execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]}
    ns["conn"].close()

    ns = run(16)
    cnt = lambda c: c.execute("SELECT count(*) FROM chunks").fetchone()[0]  # noqa: E731
    docs = lambda c: [list(r) for r in c.execute("SELECT file, published FROM documents ORDER BY file").fetchall()]  # noqa: E731
    out["steps"][16] = {"chunks": cnt(ns["conn"]), "docs": docs(ns["conn"]),
                        "by_version": [list(r) for r in ns["conn"].execute("SELECT version, count(*) FROM chunks GROUP BY version").fetchall()]}
    ns["conn"].close()
    ns = run(16)
    out["exp"]["rerun"] = cnt(ns["conn"])
    ns["conn"].close()
    try:
        run(16, patch=[('INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                        'INSERT INTO chunks VALUES (%s, %s, %s, %s, %s, %s, %s)')])
        out["exp"]["noconflict"] = None
    except psycopg.Error as e:
        out["exp"]["noconflict"] = {"type": type(e).__name__, "msg": notes_pg.scrub(str(e)).splitlines()[0],
                                    "detail": notes_pg.scrub(getattr(e.diag, "message_detail", "") or "")}

    ns = run(17)
    out["steps"][17] = {"docs": docs(ns["conn"]), "expected": dict(ns["expected"])}
    ns["conn"].close()

    ns1 = run(18)
    out["steps"][18] = {"results": [{"q": q, "rows": rows(ns1["search"](q))} for q in Q],
                        "printed": [f"{r[0]} {r[1]} {round(r[4], 3)} {r[3].splitlines()[0]}" for r in ns1["search"](Q[0])]}
    out["exp"]["weekend"] = {"q": "주말에도 열람실을 쓸 수 있나요?", "rows": rows(ns1["search"]("주말에도 열람실을 쓸 수 있나요?", k=5))}

    # ---- 새 판(v2): 적재만 하면? 공개하면? ------------------------------------------
    d2 = edited_dir()
    ns2 = run(16, d2, V2)
    staged = [list(r) for r in ns2["conn"].execute("SELECT file, version, count(*) FROM chunks GROUP BY file, version ORDER BY file, version").fetchall()]
    ns2["conn"].close()
    r_staged = rows(ns1["search"](Q[0]))
    ns2 = run(17, d2, V2)
    ns2["conn"].close()
    r_pub = rows(ns1["search"](Q[0]))
    out["exp"]["v2"] = {"staged": staged, "after_stage": r_staged, "after_publish": r_pub,
                        "docs": docs(ns1["conn"])}

    # ---- 왜: 판을 구분하지 않고 섞어 두면 -------------------------------------------
    mixed = ns1["conn"].execute("""SELECT c.file, c.version, c.body, c.embedding <=> %s AS d FROM chunks c
                                   ORDER BY d LIMIT 3""", (ns1["embed"]([Q[0]], "query")[0],)).fetchall()
    out["why"] = {"mixed": [{"file": r[0], "version": r[1], "body": r[2], "distance": round(float(r[3]), 3)} for r in mixed]}
    ns1["conn"].close()

    # ---- 실패: v2 를 적재하다 조각 하나가 빠지면 ---------------------------------------
    notes_pg.fresh()
    ns1 = run(18)  # v1 공개
    d3 = edited_dir()
    inject = ('expected = Counter(c["file"] for c in chunks)',
              'conn.execute("DELETE FROM chunks WHERE version = 2 AND chunk_id = (SELECT min(chunk_id) FROM chunks WHERE version = 2 AND file = \'guide.md\')"); conn.commit()\n'
              'expected = Counter(c["file"] for c in chunks)')
    try:
        run(17, d3, V2 + [inject])
        err = None
    except RuntimeError as e:
        err = str(e)
    out["exp"]["fail"] = {"error": err, "docs": docs(ns1["conn"]), "after": rows(ns1["search"](Q[0])),
                          "v2_rows": ns1["conn"].execute("SELECT count(*) FROM chunks WHERE version = 2").fetchone()[0]}
    ns1["conn"].close()
finally:
    notes_pg.drop()

out["code"] = [c for c in code if c["s"] <= RANGE[1]]
out["names"] = stepcode.names(out["code"])
out["range"] = RANGE
out["meta"] = {"python": sys.version.split()[0], "psycopg": psycopg.__version__, "pgvector_py": pgvector.__version__ if hasattr(pgvector, "__version__") else "?",
               "sentence_transformers": sentence_transformers.__version__}
dst = ROOT / "topics" / "backend" / "indexing-data.js"
dst.write_text("window.PAGE = " + json.dumps(out, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
shutil.rmtree(W, ignore_errors=True)
print("ok:", dst.relative_to(ROOT))
print(json.dumps({k: out["exp"][k] for k in ("rerun", "noconflict", "fail")}, ensure_ascii=False, indent=1)[:2500])
print(json.dumps(out["why"], ensure_ascii=False)[:800])
print(json.dumps(out["exp"]["v2"]["after_stage"][:2], ensure_ascii=False)[:400], json.dumps(out["exp"]["v2"]["after_publish"][:2], ensure_ascii=False)[:400])
print(out["steps"][15]["pg"], out["steps"][15]["pgvector"], out["steps"][16], out["steps"][17]["docs"])
