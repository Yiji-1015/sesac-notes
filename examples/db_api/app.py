# FastAPI + PostgreSQL(SQLAlchemy) 지식 API 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 실행: DATABASE_URL 환경 변수를 정한 뒤  uvicorn app:app --reload
#   예) postgresql+psycopg://사용자:비밀번호@localhost:5432/DB이름   (비밀번호는 코드에 적지 않아요)
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import os                                                          #@1
from sqlalchemy import create_engine, text                         #@1
from sqlalchemy import ForeignKey, String, Text, select            #@2
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column  #@2
from sqlalchemy.orm import Session                                 #@2
from sqlalchemy.orm import sessionmaker                            #@3
from fastapi import Depends, FastAPI, HTTPException                #@3
from pydantic import BaseModel, Field                              #@5
from sqlalchemy.exc import IntegrityError                          #@6
from fastapi.responses import JSONResponse                         #@6

# ① DB에 연결하기
engine = create_engine(os.environ["DATABASE_URL"])
with engine.connect() as c:
    db_ok = c.execute(text("SELECT 1")).scalar()

# ② 테이블을 파이썬 클래스로 (ORM)
class Base(DeclarativeBase):
    pass

class Category(Base):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(30), unique=True)

class Knowledge(Base):
    __tablename__ = "knowledge"
    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(80), unique=True)
    content: Mapped[str] = mapped_column(Text)
    category_id: Mapped[int] = mapped_column(ForeignKey("categories.id"))

Base.metadata.create_all(engine)  # 연습용: 없는 테이블만 만들어요 (실제로는 Alembic)
with Session(engine) as s:
    if not s.scalar(select(Category).limit(1)):
        s.add_all([Category(name=n) for n in ("learning", "environment", "reference")])
        s.commit()
        s.add_all([Knowledge(title="실습 파일 보관", content="실습 파일은 개인 폴더에 보관합니다.", category_id=1),
                   Knowledge(title="서버 실행", content="프로젝트 폴더에서 서버를 실행합니다.", category_id=2),
                   Knowledge(title="질문 작성", content="오류 메시지와 시도한 작업을 함께 적습니다.", category_id=1)])
        s.commit()  # 시작 데이터: 분류 3개, 지식 3개

# ③ 요청마다 Session 하나
SessionLocal = sessionmaker(engine)

def get_session():
    with SessionLocal() as session:
        yield session  # 요청이 끝나면 with 블록이 Session을 닫아요

app = FastAPI(title="지식 관리 API (PostgreSQL)")

def to_dict(k):
    return {"id": k.id, "title": k.title, "content": k.content, "category_id": k.category_id}

# ④ 조회하기
@app.get("/knowledge")
def list_knowledge(session: Session = Depends(get_session)):
    return [to_dict(k) for k in session.scalars(select(Knowledge).order_by(Knowledge.id))]

@app.get("/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int, session: Session = Depends(get_session)):
    k = session.get(Knowledge, knowledge_id)
    if k is None:
        raise HTTPException(status_code=404, detail="knowledge not found")
    return to_dict(k)

# ⑤ 등록하기
class KnowledgeCreate(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1)
    category: str = Field(min_length=1, max_length=30)

@app.post("/knowledge", status_code=201)
def create_knowledge(payload: KnowledgeCreate, session: Session = Depends(get_session)):
    cat = session.scalar(select(Category).where(Category.name == payload.category))
    if cat is None:
        raise HTTPException(status_code=422, detail=f"unknown category: {payload.category}")
    k = Knowledge(title=payload.title, content=payload.content, category_id=cat.id)
    session.add(k)
    session.commit()
    session.refresh(k)
    return to_dict(k)

# ⑥ DB 오류를 알맞은 응답으로
@app.exception_handler(IntegrityError)
def integrity_error(request, exc):
    # 같은 제목처럼 DB 규칙에 걸린 요청 → 서버 고장(500)이 아니라 "충돌"(409)
    return JSONResponse(status_code=409, content={"detail": "conflicts with existing data"})

# ⑦ 고치고 지우기
@app.put("/knowledge/{knowledge_id}")
def update_knowledge(knowledge_id: int, payload: KnowledgeCreate, session: Session = Depends(get_session)):
    k = session.get(Knowledge, knowledge_id)
    if k is None:
        raise HTTPException(status_code=404, detail="knowledge not found")
    cat = session.scalar(select(Category).where(Category.name == payload.category))
    if cat is None:
        raise HTTPException(status_code=422, detail=f"unknown category: {payload.category}")
    k.title, k.content, k.category_id = payload.title, payload.content, cat.id
    session.commit()
    return to_dict(k)

@app.delete("/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int, session: Session = Depends(get_session)):
    k = session.get(Knowledge, knowledge_id)
    if k is None:
        raise HTTPException(status_code=404, detail="knowledge not found")
    session.delete(k)
    session.commit()
