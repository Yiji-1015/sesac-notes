"""주제 페이지 '백엔드 > 검색과 RAG > Embedding과 벡터 검색'을 만든다.

examples/search/embed_search.py 를 단계별로 실제 실행한다 (e5 모델은 로컬 캐시, DB 는 노트 전용 DB).
실행: NOTES_PG_ENV=<.env 경로> HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/embedding.py
"""
import os
import sys
import time
from html import escape as E
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import notes_pg  # noqa: E402
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import psycopg  # noqa: E402
import sentence_transformers  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "search"
code = S.parse((EX / "embed_search.py").read_text(encoding="utf-8"))
outs = {}
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="embed_search.py")  # noqa: E731
r3 = lambda x: round(float(x), 3)  # noqa: E731

ns = run(4)
KN, docs, model = ns["KNOWLEDGE"], ns["docs"], ns["model"]
title = {k[0]: k[1] for k in KN}

# ---- 왜 ----
qwords = ns["question"].replace("?", "").split()
rows = []
for i, t, c, cat in KN:
    hit = [w for w in qwords if w in (t + " " + c)]
    rows.append([i, E(t + ": " + c), ", ".join(hit) if hit else "없음", r3(ns["cos"][[k[0] for k in KN].index(i)])])
rows.sort(key=lambda r: -r[3])
outs["why"] = S.table(f"질문 \"{ns['question']}\"", ["id", "자료", "질문의 단어가 그대로 있나", "Embedding 유사도"], [[a, b, c, f"<b>{d}</b>" if j == 0 else d] for j, (a, b, c, d) in enumerate(rows)],
                      [f"질문 단어 {qwords} 중 자료에 그대로 나오는 게 {'없어요. 글자로 찾으면 결과가 0개예요' if not any(r[2] != '없음' for r in rows) else '일부 있어요'}.",
                       f"Embedding은 \"개발용 웹 서비스를 돌린다\"와 \"서버를 실행한다\"가 비슷한 뜻이라는 걸 반영해서, {rows[0][0]}번을 1위로 찾았어요."])

# ---- ① ----
ns1 = run(1)
d = ns1["docs"]
raw = model.encode(["passage: 서버 실행 프로젝트 폴더에서 서버를 실행합니다."], normalize_embeddings=False)[0]
outs["s1"] = S.box("① 실행 결과", [(f"docs.shape → {d.shape}", 1), f"docs[1] 앞 6칸 → {[round(float(x), 4) for x in d[1][:6]]}", (f"docs[1]의 길이 → {r3(np.linalg.norm(d[1]))}  (정규화 안 하면 {r3(np.linalg.norm(raw))})", 2)],
                   ["자료 9개가 각각 384칸 벡터가 됐어요. 문장 길이와 상관없이 언제나 384칸이에요.", "<code>normalize_embeddings=True</code>로 길이를 1로 맞췄어요. ②에서 이게 왜 편한지 봐요."])
pq = model.encode(["query: 서버 실행", "passage: 서버 실행", "서버 실행"], normalize_embeddings=True)
q_np = model.encode([ns["question"]], normalize_embeddings=True)[0]
top_np = [title[KN[i][0]] for i in np.argsort(-(docs @ q_np))[:3]]
outs["t1"] = S.box("같은 글에 접두어만 다르게", [f"cos('query: 서버 실행', 'passage: 서버 실행') → {r3(pq[0] @ pq[1])}", f"cos('query: 서버 실행', '서버 실행') → {r3(pq[0] @ pq[2])}",
                    (f"질문에 query: 를 빼고 검색한 상위 3개 → {top_np}", 1)],
                   ["글이 같아도 접두어가 다르면 다른 벡터예요. e5는 질문과 자료를 구분해서 학습해서, 규칙대로 붙여야 제 성능이 나와요. 이 예제에서는 순위가 크게 바뀌지 않았지만, 자료가 많아지면 차이가 커질 수 있어요."])

# ---- ② ----
top = np.argsort(-ns["cos"])[:4]
rows = [[KN[i][0], E(KN[i][1]), r3(ns["dot"][i]), r3(ns["cos"][i]), r3(ns["l2"][i])] for i in top]
outs["s2"] = S.table(f"질문과 자료의 세 가지 거리 (상위 4개)", ["id", "자료", "내적", "코사인 유사도", "L2 거리"], rows,
                     ["길이가 1인 벡터끼리는 내적과 코사인 유사도가 <b>똑같아요</b>. 그래서 정규화해 두면 계산이 가장 싼 내적만 써도 돼요.",
                      "L2 거리는 작을수록 가까워서 방향이 반대지만, 순위는 같아요 (길이 1이면 L2² = 2 − 2 × 코사인)."])
nn = model.encode([f"passage: {t} {c}" for _, t, c, _ in KN], normalize_embeddings=False)
qn = model.encode([f"query: {ns['question']}"], normalize_embeddings=False)[0]
dot_raw = nn @ qn
outs["t2"] = S.box("정규화하지 않은 벡터로 내적 순위를 매기면", [f"내적 상위 3개   → {[title[KN[i][0]] for i in np.argsort(-dot_raw)[:3]]}",
                    f"코사인 상위 3개 → {[title[KN[i][0]] for i in np.argsort(-ns['cos'])[:3]]}", f"자료 벡터 길이 범위 → {r3(np.linalg.norm(nn, axis=1).min())} ~ {r3(np.linalg.norm(nn, axis=1).max())}"],
                   [("이 모델은 정규화 옵션을 끄더라도 길이가 이미 1로 나와서 순위가 같아요. " if r3(np.linalg.norm(nn, axis=1).min()) == r3(np.linalg.norm(nn, axis=1).max()) == 1.0 else "")
                    + "하지만 모델에 따라 길이가 제각각인 벡터를 내기도 해요. 내적은 \"방향\"뿐 아니라 \"길이\"에도 끌려가서, 길이가 1이라는 보장이 없으면 코사인을 쓰거나 직접 정규화해요."])

# ---- ③ ④ ----
outs["s3"] = S.box("top_k(cos, 3)", [(str(ns["best"]), 1)], ["(id, 제목, 점수). 점수가 큰 순서예요. 1위와 2위의 차이가 크지 않은 것도 볼 수 있어요. 점수가 \"맞다, 틀리다\"를 뜻하진 않아요."])
q4 = "설정은 어디서 확인하나요?"
allr = ns["search"](q4)
post = [r for r in ns["search"](q4, k=3) if dict((k[0], k[3]) for k in KN)[r[0]] == "development"]
outs["s4"] = S.box(f"search(\"{q4}\")", [f"조건 없이 → {allr}", (f"category=\"development\" → {ns['filtered']}", 1), (f"(비교) 먼저 3개 뽑고 나서 거르면 → {post}", 2, "bad")],
                   ["조건에 맞는 자료 안에서 3개를 골랐어요.", f"순서를 바꿔 \"상위 3개를 뽑은 뒤 거르면\" {len(post)}개만 남아요. 조건으로 <b>먼저</b> 거르고 Top-k를 해야 원하는 개수가 나와요."])

# ---- ⑤ ⑥ ----
try:
    notes_pg.fresh()
    ns5 = run(5)
    outs["s5"] = S.box("pgvector에서 ORDER BY embedding <=> 질문 LIMIT 3", [(str([(a, b, r3(c)) for a, b, c in ns5["db_best"]]), 1), f"③ 파이썬 결과와 같나 → {[(a, b) for a, b, _ in ns5['db_best']] == [(a, b) for a, b, _ in ns5['best']]}"],
                       ["<code>&lt;=&gt;</code>는 코사인 <b>거리</b>(1 − 유사도)라서, 유사도로 보려면 <code>1 - (...)</code>로 바꿔요. 순위와 점수가 파이썬으로 계산한 ③과 같아요."])
    ns5["conn"].close()

    # 큰 데이터에서 정확 검색 vs HNSW
    N, Q, D = 20000, 50, 384
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(200, D))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    base = (centers[rng.integers(0, 200, N)] + rng.normal(scale=1.2 / np.sqrt(D), size=(N, D))).astype(np.float32)  # 200개 주제 근처에 모인 벡터
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    qs = (centers[rng.integers(0, 200, Q)] + rng.normal(scale=1.2 / np.sqrt(D), size=(Q, D))).astype(np.float32)  # 새 질문
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    exact = np.argsort(-(qs @ base.T), axis=1)[:, :10]
    with psycopg.connect() as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.commit()
        register_vector(c)
        c.execute("CREATE TABLE big (id int PRIMARY KEY, embedding vector(384))")
        with c.cursor().copy("COPY big (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["int4", "vector"])
            for i, v in enumerate(base):
                cp.write_row((i, v))
        c.commit()

        def bench(label):
            got, t0 = [], time.perf_counter()
            for qv in qs:
                got.append([r[0] for r in c.execute("SELECT id FROM big ORDER BY embedding <=> %s LIMIT 10", (qv,)).fetchall()])
            ms = (time.perf_counter() - t0) / Q * 1000
            rec = np.mean([len(set(g) & set(e)) / 10 for g, e in zip(got, exact)])
            plan = c.execute("EXPLAIN SELECT id FROM big ORDER BY embedding <=> %s LIMIT 10", (qs[0],)).fetchall()
            line = next(p[0] for p in plan if "Scan" in p[0]).strip().lstrip("->").strip().split("  ")[0]
            return [label, f"{ms:.2f} ms", f"{rec:.2f}", f"<code>{E(line)}</code>"]

        rows = [bench("인덱스 없음 (전부 비교)")]
        t0 = time.perf_counter()
        c.execute("CREATE INDEX ON big USING hnsw (embedding vector_cosine_ops)")
        c.commit()
        build = time.perf_counter() - t0
        rows.append(bench("HNSW (ef_search=40, 기본값)"))
        c.execute("SET hnsw.ef_search = 10")
        rows.append(bench("HNSW (ef_search=10)"))
    outs["s6"] = S.table(f"주제 200개 근처에 모인 무작위 벡터 {N:,}개, 질문 {Q}개, 상위 10개 (질문 하나당 평균)", ["방법", "시간", "정답 일치율 (recall@10)", "실행 계획"], rows,
                         [f"인덱스를 만드는 데 {build:.1f}초가 걸렸어요. 대신 질문마다 몇 배 빨라졌어요.",
                          "정답 일치율은 \"전부 비교한 진짜 상위 10개\" 중 몇 개를 찾았나예요. HNSW는 일부만 보고 판단하는 <b>근사</b> 검색이라 1.00보다 낮을 수 있어요"
                          + (". 여기서도 기본값에서 " + rows[1][2] + ", <code>ef_search=10</code>에서 " + rows[2][2] + "로, 후보를 줄일수록 더 많이 놓쳤어요." if rows[2][2] < rows[1][2] or rows[1][2] != "1.00"
                             else ". 이 데이터에서는 놓친 게 없었지만, 데이터가 촘촘하고 많을수록 놓치는 게 생겨요.")])
finally:
    notes_pg.drop()

flow = [[1, "벡터로", "encode", "384칸, 길이 1"], [2, "거리", "내적, 코사인, L2", "정규화하면 같은 순위"], [3, "Top-k", "argsort", "큰 순서 k개"],
        [4, "조건", "먼저 거르기", "그다음 Top-k"], [5, "DB로", "pgvector", "<=> 연산자"], [6, "많아지면", "HNSW", "빠른 대신 근사"]]
S.save(ROOT / "topics" / "backend" / "embedding-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                            "env": f"Python {sys.version.split()[0]}, sentence-transformers {sentence_transformers.__version__}, intfloat/multilingual-e5-small, PostgreSQL + pgvector (노트 전용 DB)"})

why = f"""
<p>"개발용 웹 서비스는 어디서 돌리나요?"라고 물었을 때 찾아야 할 자료는 "프로젝트 폴더에서 서버를 실행합니다"예요. 글자는 하나도 겹치지 않지만 뜻은 같아요. 실제로 자료 9개에 두 방식으로 찾아보면:</p>
{K.out("why")}
<p><b>Embedding</b>은 글을 뜻이 담긴 숫자 목록(<b>벡터</b>)으로 바꾸는 거예요. 뜻이 비슷한 글은 비슷한 방향의 벡터가 돼서, 벡터끼리 가까운지만 재면 \"뜻이 비슷한 자료\"를 찾을 수 있어요. 이걸 <b>벡터 검색</b>(의미 검색)이라고 해요.</p>
<p>여기서는 한국어를 지원하는 Embedding 모델 <code>intfloat/multilingual-e5-small</code>을 이 PC에서 직접 돌려요.</p>
"""
whole = """
<p>자료를 벡터로 바꿔 두고, 질문도 벡터로 바꿔 가까운 순으로 고르고, 필요하면 조건으로 먼저 거르고, 많아지면 DB와 인덱스에 맡겨요. 상자를 누르면 그 단계로 가요.</p>
<div class="flow" id="flow"></div>
"""
steps = [
    K.step(1, "문장을 벡터로", "자료마다 <code>passage: </code>를 붙여 Embedding 모델에 넣고, 길이를 1로 맞춘 벡터를 받아요.",
           K.code_table([("SentenceTransformer(\"intfloat/multilingual-e5-small\")", "문장을 벡터로 바꾸는 모델을 불러와요. 한 번 받아 두면 이 PC에서 돌아가요."),
                         ("f\"passage: {t} {c}\"", "e5는 저장할 자료 앞에 <code>passage: </code>, 질문 앞에 <code>query: </code>를 붙이도록 학습됐어요."),
                         ("model.encode(texts, normalize_embeddings=True)", "글 목록을 벡터 목록(numpy 배열)으로. 길이를 1로 맞춰요.")])
           + K.try_("같은 글에 접두어만 다르게 넣거나, 질문에서 <code>query: </code>를 빼면?", "t1"), "s1"),
    K.step(2, "비슷한 정도 재기", "질문 벡터와 자료 벡터 사이를 세 가지 방법으로 재 봐요.",
           K.table(["방법", "뜻", "가까우면"], [["내적 (dot)", "칸끼리 곱해서 더하기", "큼"], ["코사인 유사도", "두 벡터의 방향이 얼마나 같은가. 길이와 상관없음", "1에 가까움"], ["L2 거리", "두 점 사이의 직선 거리", "0에 가까움"]])
           + K.code_table([("docs @ q", "모든 자료와 질문의 내적을 한 번에 (행렬 곱)"), ("np.linalg.norm(x, axis=1)", "각 벡터의 길이")])
           + K.try_("정규화하지 않은 벡터로 내적 순위를 매기면?", "t2"), "s2"),
    K.step(3, "가장 가까운 k개 (Top-k)", "점수가 큰 순서로 k개를 골라요.",
           K.code_table([("np.argsort(-scores)[:k]", "점수를 큰 순서로 정렬한 위치 중 앞의 k개. <code>argsort</code>는 작은 순서라서 부호를 뒤집어요.")]), "s3"),
    K.step(4, "조건으로 먼저 거르기", "분류 같은 조건이 있으면, 조건 밖의 자료는 점수를 −∞로 만들어 절대 뽑히지 않게 한 뒤 Top-k를 해요.",
           "<p>\"개발 자료 안에서만\", \"최신 판만\"처럼 조건이 붙는 경우가 많아요. 조건을 Top-k <b>다음</b>에 걸면 남는 게 거의 없을 수 있어요.</p>"
           + K.code_table([("np.where(allowed, scores, -np.inf)", "조건에 맞으면 점수 그대로, 아니면 −∞")]), "s4"),
    K.step(5, "DB로 옮기기 (pgvector)", "벡터를 PostgreSQL에 저장하고, 가까운 순 정렬을 DB가 하게 해요.",
           "<p>자료가 수십만 개가 되면 메모리에 다 올려 두기 어렵고, 출처나 판 번호 같은 정보도 함께 관리해야 해요. <b>pgvector</b>는 PostgreSQL에 벡터 타입과 거리 연산자를 더하는 확장이에요.</p>"
           + K.code_table([("vector(384)", "384칸 벡터 열"), ("register_vector(conn)", "numpy 배열과 DB의 vector를 서로 바꿔 주도록 등록"),
                           ("embedding <=> %s", "코사인 거리. <code>&lt;-&gt;</code>는 L2 거리, <code>&lt;#&gt;</code>는 음의 내적"),
                           ("ORDER BY ... LIMIT 3", "가까운 순으로 3개")]), "s5"),
    K.step(6, "많아지면: 근사 인덱스 (HNSW)", "벡터용 인덱스를 만들어, 모든 자료와 비교하지 않고도 가까운 것을 빨리 찾아요.",
           "<p>인덱스가 없으면 질문마다 모든 벡터와 거리를 재요. <b>HNSW</b>는 벡터들을 \"가까운 것끼리 이어진 여러 층의 그물\"로 미리 엮어 두고, 그물을 따라가며 가까운 쪽으로만 찾아가는 인덱스예요. 대부분 맞게 찾지만 가끔 놓쳐서 <b>근사</b>라고 해요. 9개로는 차이가 안 보여서, 무작위 벡터 2만 개로 따로 재 봤어요.</p>"
           + K.code_table([("CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)", "코사인 거리용 HNSW 인덱스. 거리 연산자와 짝이 맞아야 쓰여요."),
                           ("SET hnsw.ef_search = 10", "검색 때 살펴볼 후보 수 (기본 40). 크면 정확하고 느리고, 작으면 빠르고 더 놓쳐요.")]), "s6"),
]
practice = [
    ("모델을 바꾼다", "벡터는 모델마다 다른 공간이라 섞어 쓸 수 없어요. 모델을 바꾸면 모든 자료의 벡터를 다시 만들어요."),
    ("긴 문서", "모델은 입력 길이 한도가 있어요 (e5는 512토큰). 문서는 먼저 조각으로 나눠요 (<a href=\"chunking.html\">Chunking</a>)."),
    ("점수 기준선", "\"0.8 이상이면 맞다\" 같은 고정 기준은 질문마다 잘 안 맞아요. 상위 몇 개를 보여 주고, 정말 맞는지는 원문과 대조해요."),
    ("글자 검색도 필요하다", "제품 번호, 오류 코드처럼 정확한 글자가 중요하면 글자 검색이 더 잘 맞아요. 둘을 섞어 쓰기도 해요 (Hybrid Search)."),
    ("HNSW 인덱스와 조건", "④처럼 조건을 같이 걸면, 인덱스가 먼저 찾은 후보를 거른 뒤라 결과가 적게 나올 수 있어요. pgvector의 <code>hnsw.iterative_scan</code> 설정이나 조건별 인덱스로 보완해요."),
]
libs = [("①", "<code>SentenceTransformer(name)</code>, <code>model.encode(texts, normalize_embeddings=True)</code>", "Embedding (sentence-transformers)"),
        ("②~④", "<code>@</code>(행렬 곱), <code>np.linalg.norm()</code>, <code>np.argsort()</code>, <code>np.where()</code>", "거리, Top-k, 조건 (numpy)"),
        ("⑤", "<code>vector(n)</code>, <code>&lt;=&gt;</code>, <code>register_vector()</code>", "pgvector"), ("⑥", "<code>USING hnsw (... vector_cosine_ops)</code>, <code>hnsw.ef_search</code>", "근사 인덱스 (pgvector)")]
K.build(ROOT / "topics" / "backend" / "embedding.html", title="Embedding과 벡터 검색", h1="Embedding과 벡터 검색: 글자가 아니라 뜻으로 찾기", track="백엔드",
        crumb="백엔드 &gt; 검색과 RAG",
        one_line="<b>Embedding</b>은 글을 뜻이 담긴 벡터로 바꾸는 거예요. 뜻이 비슷하면 벡터가 가까워서, <b>글자가 하나도 겹치지 않아도</b> 질문에 맞는 자료를 찾을 수 있어요.",
        why=why, whole=whole, steps=steps, code_file="examples/search/embed_search.py", practice=practice, libs=libs,
        libs_note="<code>top_k</code>, <code>search</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"document-reading.html\">문서 적재 1~3쪽</a>", "실제 문서를 조각으로 나눠 이 방식으로 넣고 검색하기"), ("<a href=\"postgres.html\">PostgreSQL 기초</a>", "B-tree 인덱스와 실행 계획")],
        sources=["다루는 범위: Day23 (2026-11-18) Embedding, pgvector", "예제: <code>examples/search/embed_search.py</code> (기존 노트의 자료 101~109)",
                 "결과 값: <code>tools/pages/embedding.py</code>가 단계별로 실제 실행한 결과. ⑥의 2만 개와 질문 50개는 주제 200개 근처에 흩어지도록 만든 무작위 벡터(seed 0)",
                 "공식 문서: <a href=\"https://huggingface.co/intfloat/multilingual-e5-small\" target=\"_blank\">multilingual-e5-small</a>, <a href=\"https://sbert.net/\" target=\"_blank\">Sentence Transformers</a>, <a href=\"https://github.com/pgvector/pgvector\" target=\"_blank\">pgvector</a>"])
print("ok")
