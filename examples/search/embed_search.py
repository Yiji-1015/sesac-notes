# Embedding과 벡터 검색 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# ⑤ ⑥은 PostgreSQL + pgvector 가 필요해요 (접속 정보는 환경 변수 PGHOST, PGUSER, PGPASSWORD, PGDATABASE).
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import numpy as np                                                 #@1
from sentence_transformers import SentenceTransformer              #@1
import psycopg                                                     #@5
from pgvector.psycopg import register_vector                       #@5

# ① 문장을 벡터로
KNOWLEDGE = [
    (101, "실습 파일 보관", "실습 파일은 개인 폴더에 보관합니다.", "course"),
    (102, "서버 실행", "프로젝트 폴더에서 서버를 실행합니다.", "development"),
    (103, "질문 작성", "오류 메시지와 시도한 작업을 함께 적습니다.", "support"),
    (104, "수업 녹화", "녹화가 제공되는 수업은 강의실 안내 페이지에서 확인합니다.", "course"),
    (105, "과제 제출", "과제 제출 위치는 해당 과제의 안내 페이지에서 확인합니다.", "course"),
    (106, "데이터베이스 연결", "PostgreSQL 접속 정보는 로컬 설정 파일에서 읽습니다.", "development"),
    (107, "패키지 설치", "필요한 Python 패키지는 기존 가상환경에 추가합니다.", "development"),
    (108, "문제 해결 안내", "문제를 보고할 때 재현한 단계와 오류 메시지를 기록합니다.", "support"),
    (109, "결과 검토", "검색된 자료는 원문과 실제 질문의 요구를 대조합니다.", "support"),
]
model = SentenceTransformer("intfloat/multilingual-e5-small")
docs = model.encode([f"passage: {t} {c}" for _, t, c, _ in KNOWLEDGE], normalize_embeddings=True)

# ② 비슷한 정도 재기
question = "개발용 웹 서비스는 어디서 돌리나요?"
q = model.encode([f"query: {question}"], normalize_embeddings=True)[0]
dot = docs @ q                                   # 내적
cos = dot / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))  # 코사인 유사도
l2 = np.linalg.norm(docs - q, axis=1)            # 유클리드 거리 (작을수록 가까움)

# ③ 가장 가까운 k개 (Top-k)
def top_k(scores, k=3):
    order = np.argsort(-scores)[:k]  # 점수가 큰 순서의 위치
    return [(KNOWLEDGE[i][0], KNOWLEDGE[i][1], round(float(scores[i]), 3)) for i in order]

best = top_k(cos)

# ④ 조건으로 먼저 거르기
def search(question, category=None, k=3):
    qv = model.encode([f"query: {question}"], normalize_embeddings=True)[0]
    scores = docs @ qv
    allowed = np.array([category is None or row[3] == category for row in KNOWLEDGE])
    scores = np.where(allowed, scores, -np.inf)  # 조건 밖은 절대 뽑히지 않게
    return [r for r in top_k(scores, k) if r[2] != float("-inf")]

filtered = search("설정은 어디서 확인하나요?", category="development")

# ⑤ DB로 옮기기 (pgvector)
conn = psycopg.connect()
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
register_vector(conn)
conn.execute("CREATE TABLE knowledge (id int PRIMARY KEY, title text, category text, embedding vector(384))")
for (i, t, _, cat), v in zip(KNOWLEDGE, docs):
    conn.execute("INSERT INTO knowledge VALUES (%s, %s, %s, %s)", (i, t, cat, v))
db_best = conn.execute("""
    SELECT id, title, 1 - (embedding <=> %s) AS cos
    FROM knowledge ORDER BY embedding <=> %s LIMIT 3""", (q, q)).fetchall()

# ⑥ 많아지면: 근사 인덱스(HNSW)
conn.execute("CREATE INDEX ON knowledge USING hnsw (embedding vector_cosine_ops)")
conn.commit()
