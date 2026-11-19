"""노트 실행용 PostgreSQL 도우미.

환경변수 NOTES_PG_ENV 가 가리키는 .env (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)를 읽어서
노트 전용 DB 를 만들고, 예제 코드가 읽는 libpq 환경 변수(PGHOST 등)를 채운다.
비밀번호는 어디에도 출력하지 않는다. 쓰고 나면 drop() 으로 지운다.
"""
import os
from pathlib import Path

import psycopg

LAB = "sesac_notes_lab"


def _read_env(p):
    out = {}
    for ln in Path(p).read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


CRED = _read_env(os.environ["NOTES_PG_ENV"])


def _admin():
    return psycopg.connect(host=CRED.get("DB_HOST", "localhost"), port=int(CRED.get("DB_PORT", 5432)),
                           dbname="postgres", user=CRED["DB_USER"], password=CRED["DB_PASSWORD"], autocommit=True)


def fresh():
    """노트 전용 DB 를 새로 만들고 PG* 환경 변수를 그 DB 로 맞춘다."""
    with _admin() as ad:
        ad.execute(f'DROP DATABASE IF EXISTS "{LAB}" WITH (FORCE)')
        ad.execute(f'CREATE DATABASE "{LAB}"')
    os.environ.update({"PGHOST": CRED.get("DB_HOST", "localhost"), "PGPORT": str(CRED.get("DB_PORT", 5432)),
                       "PGUSER": CRED["DB_USER"], "PGPASSWORD": CRED["DB_PASSWORD"], "PGDATABASE": LAB})


def drop():
    with _admin() as ad:
        ad.execute(f'DROP DATABASE IF EXISTS "{LAB}" WITH (FORCE)')


def scrub(s):
    return str(s).replace(CRED["DB_PASSWORD"], "********") if CRED.get("DB_PASSWORD") else str(s)
