window.PAGE = {
 "code": [
  {
   "t": "import os",
   "s": 1
  },
  {
   "t": "from sqlalchemy import create_engine, text",
   "s": 1
  },
  {
   "t": "from sqlalchemy import ForeignKey, String, Text, select",
   "s": 2
  },
  {
   "t": "from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column",
   "s": 2
  },
  {
   "t": "from sqlalchemy.orm import Session",
   "s": 2
  },
  {
   "t": "from sqlalchemy.orm import sessionmaker",
   "s": 3
  },
  {
   "t": "from fastapi import Depends, FastAPI, HTTPException",
   "s": 3
  },
  {
   "t": "from pydantic import BaseModel, Field",
   "s": 5
  },
  {
   "t": "from sqlalchemy.exc import IntegrityError",
   "s": 6
  },
  {
   "t": "from fastapi.responses import JSONResponse",
   "s": 6
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① DB에 연결하기",
   "s": 1
  },
  {
   "t": "engine = create_engine(os.environ[\"DATABASE_URL\"])",
   "s": 1
  },
  {
   "t": "with engine.connect() as c:",
   "s": 1
  },
  {
   "t": "    db_ok = c.execute(text(\"SELECT 1\")).scalar()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 테이블을 파이썬 클래스로 (ORM)",
   "s": 2
  },
  {
   "t": "class Base(DeclarativeBase):",
   "s": 2
  },
  {
   "t": "    pass",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "class Category(Base):",
   "s": 2
  },
  {
   "t": "    __tablename__ = \"categories\"",
   "s": 2
  },
  {
   "t": "    id: Mapped[int] = mapped_column(primary_key=True)",
   "s": 2
  },
  {
   "t": "    name: Mapped[str] = mapped_column(String(30), unique=True)",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "class Knowledge(Base):",
   "s": 2
  },
  {
   "t": "    __tablename__ = \"knowledge\"",
   "s": 2
  },
  {
   "t": "    id: Mapped[int] = mapped_column(primary_key=True)",
   "s": 2
  },
  {
   "t": "    title: Mapped[str] = mapped_column(String(80), unique=True)",
   "s": 2
  },
  {
   "t": "    content: Mapped[str] = mapped_column(Text)",
   "s": 2
  },
  {
   "t": "    category_id: Mapped[int] = mapped_column(ForeignKey(\"categories.id\"))",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "Base.metadata.create_all(engine)  # 연습용: 없는 테이블만 만들어요 (실제로는 Alembic)",
   "s": 2
  },
  {
   "t": "with Session(engine) as s:",
   "s": 2
  },
  {
   "t": "    if not s.scalar(select(Category).limit(1)):",
   "s": 2
  },
  {
   "t": "        s.add_all([Category(name=n) for n in (\"learning\", \"environment\", \"reference\")])",
   "s": 2
  },
  {
   "t": "        s.commit()",
   "s": 2
  },
  {
   "t": "        s.add_all([Knowledge(title=\"실습 파일 보관\", content=\"실습 파일은 개인 폴더에 보관합니다.\", category_id=1),",
   "s": 2
  },
  {
   "t": "                   Knowledge(title=\"서버 실행\", content=\"프로젝트 폴더에서 서버를 실행합니다.\", category_id=2),",
   "s": 2
  },
  {
   "t": "                   Knowledge(title=\"질문 작성\", content=\"오류 메시지와 시도한 작업을 함께 적습니다.\", category_id=1)])",
   "s": 2
  },
  {
   "t": "        s.commit()  # 시작 데이터: 분류 3개, 지식 3개",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 요청마다 Session 하나",
   "s": 3
  },
  {
   "t": "SessionLocal = sessionmaker(engine)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def get_session():",
   "s": 3
  },
  {
   "t": "    with SessionLocal() as session:",
   "s": 3
  },
  {
   "t": "        yield session  # 요청이 끝나면 with 블록이 Session을 닫아요",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "app = FastAPI(title=\"지식 관리 API (PostgreSQL)\")",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def to_dict(k):",
   "s": 3
  },
  {
   "t": "    return {\"id\": k.id, \"title\": k.title, \"content\": k.content, \"category_id\": k.category_id}",
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
   "t": "@app.get(\"/knowledge\")",
   "s": 4
  },
  {
   "t": "def list_knowledge(session: Session = Depends(get_session)):",
   "s": 4
  },
  {
   "t": "    return [to_dict(k) for k in session.scalars(select(Knowledge).order_by(Knowledge.id))]",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "@app.get(\"/knowledge/{knowledge_id}\")",
   "s": 4
  },
  {
   "t": "def get_knowledge(knowledge_id: int, session: Session = Depends(get_session)):",
   "s": 4
  },
  {
   "t": "    k = session.get(Knowledge, knowledge_id)",
   "s": 4
  },
  {
   "t": "    if k is None:",
   "s": 4
  },
  {
   "t": "        raise HTTPException(status_code=404, detail=\"knowledge not found\")",
   "s": 4
  },
  {
   "t": "    return to_dict(k)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 등록하기",
   "s": 5
  },
  {
   "t": "class KnowledgeCreate(BaseModel):",
   "s": 5
  },
  {
   "t": "    title: str = Field(min_length=1, max_length=80)",
   "s": 5
  },
  {
   "t": "    content: str = Field(min_length=1)",
   "s": 5
  },
  {
   "t": "    category: str = Field(min_length=1, max_length=30)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "@app.post(\"/knowledge\", status_code=201)",
   "s": 5
  },
  {
   "t": "def create_knowledge(payload: KnowledgeCreate, session: Session = Depends(get_session)):",
   "s": 5
  },
  {
   "t": "    cat = session.scalar(select(Category).where(Category.name == payload.category))",
   "s": 5
  },
  {
   "t": "    if cat is None:",
   "s": 5
  },
  {
   "t": "        raise HTTPException(status_code=422, detail=f\"unknown category: {payload.category}\")",
   "s": 5
  },
  {
   "t": "    k = Knowledge(title=payload.title, content=payload.content, category_id=cat.id)",
   "s": 5
  },
  {
   "t": "    session.add(k)",
   "s": 5
  },
  {
   "t": "    session.commit()",
   "s": 5
  },
  {
   "t": "    session.refresh(k)",
   "s": 5
  },
  {
   "t": "    return to_dict(k)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ DB 오류를 알맞은 응답으로",
   "s": 6
  },
  {
   "t": "@app.exception_handler(IntegrityError)",
   "s": 6
  },
  {
   "t": "def integrity_error(request, exc):",
   "s": 6
  },
  {
   "t": "    # 같은 제목처럼 DB 규칙에 걸린 요청 → 서버 고장(500)이 아니라 \"충돌\"(409)",
   "s": 6
  },
  {
   "t": "    return JSONResponse(status_code=409, content={\"detail\": \"conflicts with existing data\"})",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 고치고 지우기",
   "s": 7
  },
  {
   "t": "@app.put(\"/knowledge/{knowledge_id}\")",
   "s": 7
  },
  {
   "t": "def update_knowledge(knowledge_id: int, payload: KnowledgeCreate, session: Session = Depends(get_session)):",
   "s": 7
  },
  {
   "t": "    k = session.get(Knowledge, knowledge_id)",
   "s": 7
  },
  {
   "t": "    if k is None:",
   "s": 7
  },
  {
   "t": "        raise HTTPException(status_code=404, detail=\"knowledge not found\")",
   "s": 7
  },
  {
   "t": "    cat = session.scalar(select(Category).where(Category.name == payload.category))",
   "s": 7
  },
  {
   "t": "    if cat is None:",
   "s": 7
  },
  {
   "t": "        raise HTTPException(status_code=422, detail=f\"unknown category: {payload.category}\")",
   "s": 7
  },
  {
   "t": "    k.title, k.content, k.category_id = payload.title, payload.content, cat.id",
   "s": 7
  },
  {
   "t": "    session.commit()",
   "s": 7
  },
  {
   "t": "    return to_dict(k)",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "@app.delete(\"/knowledge/{knowledge_id}\", status_code=204)",
   "s": 7
  },
  {
   "t": "def delete_knowledge(knowledge_id: int, session: Session = Depends(get_session)):",
   "s": 7
  },
  {
   "t": "    k = session.get(Knowledge, knowledge_id)",
   "s": 7
  },
  {
   "t": "    if k is None:",
   "s": 7
  },
  {
   "t": "        raise HTTPException(status_code=404, detail=\"knowledge not found\")",
   "s": 7
  },
  {
   "t": "    session.delete(k)",
   "s": 7
  },
  {
   "t": "    session.commit()",
   "s": 7
  }
 ],
 "names": {
  "1": "DB에 연결하기",
  "2": "테이블을 파이썬 클래스로 (ORM)",
  "3": "요청마다 Session 하나",
  "4": "조회하기",
  "5": "등록하기",
  "6": "DB 오류를 알맞은 응답으로",
  "7": "고치고 지우기"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "s1": {
   "title": "① 실행 결과",
   "lines": [
    [
     "engine.url → postgresql+psycopg://postgres:***@localhost:5432/sesac_notes_lab",
     1
    ],
    [
     "SELECT 1 → 1",
     2
    ]
   ],
   "notes": [
    "주소의 비밀번호 자리는 <code>***</code>로 가려져 보여요. 실제 값은 환경 변수에만 있어요.",
    "DB가 1을 돌려줬으니 연결이 됐어요."
   ]
  },
  "s2": {
   "title": "create_all()이 DB에 만든 테이블",
   "cols": [
    "테이블",
    "열",
    "타입",
    "NULL"
   ],
   "rows": [
    [
     "categories",
     "<code>id</code>",
     "INTEGER",
     "불가"
    ],
    [
     "categories",
     "<code>name</code>",
     "VARCHAR(30)",
     "불가"
    ],
    [
     "knowledge",
     "<code>id</code>",
     "INTEGER",
     "불가"
    ],
    [
     "knowledge",
     "<code>title</code>",
     "VARCHAR(80)",
     "불가"
    ],
    [
     "knowledge",
     "<code>content</code>",
     "TEXT",
     "불가"
    ],
    [
     "knowledge",
     "<code>category_id</code>",
     "INTEGER",
     "불가"
    ]
   ],
   "notes": [
    "클래스에 적은 <code>Mapped[int]</code>, <code>String(30)</code>가 DB 타입 <code>INTEGER</code>, <code>VARCHAR(30)</code>로 바뀌었어요. <code>Mapped[str]</code>처럼 <code>Optional</code>이 아니면 NULL 불가가 돼요.",
    "시작 데이터: 분류 [(1, 'learning'), (2, 'environment'), (3, 'reference')], 지식 3개"
   ]
  },
  "s3": {
   "title": "get_session()을 요청 하나처럼 열고 닫으면",
   "lines": [
    "요청 전   빌려 간 연결 → 0",
    [
     "요청 중   빌려 간 연결 → 1",
     1
    ],
    [
     "요청 끝   빌려 간 연결 → 0",
     2
    ]
   ],
   "notes": [
    "Session이 처음 SQL을 보낼 때 연결 풀에서 연결 하나를 빌려 가요.",
    "요청이 끝나 <code>with</code> 블록을 나가면 Session이 닫히고 연결이 풀로 돌아와요. 요청마다 이렇게 열고 닫아서, 연결이 새거나 다른 요청과 섞이지 않아요."
   ]
  },
  "s4": {
   "title": "④ 실행 결과",
   "cols": [
    "요청",
    "상태",
    "응답"
   ],
   "rows": [
    [
     "<code>GET /knowledge</code>",
     "<b>200</b>",
     "<code>[{&#x27;id&#x27;: 1, &#x27;title&#x27;: &#x27;실습 파일 보관&#x27;, &#x27;content&#x27;: &#x27;실습 파일은 개인 폴더에 보관합니다.&#x27;, &#x27;category_id&#x27;: 1}, {&#x27;id&#x27;: 2, &#x27;title&#x27;: &#x27;서버 실</code>"
    ],
    [
     "<code>GET /knowledge/2</code>",
     "<b>200</b>",
     "<code>{&#x27;id&#x27;: 2, &#x27;title&#x27;: &#x27;서버 실행&#x27;, &#x27;content&#x27;: &#x27;프로젝트 폴더에서 서버를 실행합니다.&#x27;, &#x27;category_id&#x27;: 2}</code>"
    ],
    [
     "<code>GET /knowledge/99</code>",
     "<b>404</b>",
     "<code>{&#x27;detail&#x27;: &#x27;knowledge not found&#x27;}</code>"
    ]
   ],
   "notes": [
    "API 쪽 모습은 <a href=\"fastapi.html\">파일 버전</a>과 같아요. 바뀐 건 함수 안에서 파일 대신 Session으로 DB를 읽는다는 것뿐이에요."
   ]
  },
  "sql": {
   "title": "POST /knowledge 한 번에 실제로 DB로 간 SQL (SQLAlchemy 로그)",
   "lines": [
    [
     "BEGIN (implicit)",
     1
    ],
    [
     "SELECT categories.id, categories.name FROM categories WHERE categories.name = %(name_1)s::VARCHAR",
     2
    ],
    [
     "INSERT INTO knowledge (title, content, category_id) VALUES (%(title)s::VARCHAR, %(content)s::VARCHAR, %(category_id)s::INTEGER) RETURNING knowledge.id",
     3
    ],
    [
     "COMMIT",
     4
    ],
    [
     "BEGIN (implicit)",
     5
    ],
    [
     "SELECT knowledge.id, knowledge.title, knowledge.content, knowledge.category_id FROM knowledge WHERE knowledge.id = %(pk_1)s::INTEGER",
     6
    ],
    [
     "ROLLBACK",
     7
    ]
   ],
   "notes": [
    "트랜잭션 시작. Session이 처음 SQL을 보낼 때 알아서 시작해요.",
    "분류 이름으로 번호 찾기 (<code>select(Category).where(...)</code>가 바뀐 것)",
    "<code>commit()</code>이 보낸 저장. <code>RETURNING</code>으로 DB가 붙인 번호를 받아요.",
    "확정",
    "트랜잭션 시작. Session이 처음 SQL을 보낼 때 알아서 시작해요.",
    "<code>refresh(k)</code>가 저장된 행을 다시 읽어요. 그래서 바로 앞에 새 <code>BEGIN</code>이 있어요.",
    "요청이 끝나 Session을 닫을 때, 읽기만 한 트랜잭션을 정리해요. 바뀐 건 없어요."
   ]
  },
  "s5": {
   "title": "⑤ 실행 결과",
   "lines": [
    [
     "POST 가상환경 만들기 → 201 {'id': 4, 'title': '가상환경 만들기', 'content': 'python -m venv .venv 로 만듭니다.', 'category_id': 2}",
     1
    ],
    [
     "없는 분류 → 422 {'detail': 'unknown category: nothing'}",
     2
    ],
    [
     "같은 제목 한 번 더 → 500 Internal Server Error",
     3,
     "bad"
    ]
   ],
   "notes": [
    "201과 함께 DB가 붙인 번호 4가 왔어요.",
    "분류 이름이 틀리면 저장 전에 422로 돌려보내요. 이 검사는 우리가 직접 짰어요.",
    "같은 제목은 DB의 <code>UNIQUE</code> 규칙에 걸려 <code>IntegrityError</code>가 났는데, 아무도 잡지 않아서 <b>500(서버 내부 오류)</b>이 됐어요. 서버가 고장 난 게 아니라 요청이 규칙을 어긴 건데도요. ⑥에서 고쳐요."
   ]
  },
  "s6": {
   "title": "같은 제목을 두 번 POST",
   "lines": [
    [
     "두 번째 → 409 {'detail': 'conflicts with existing data'}",
     1
    ],
    [
     "지식 수 → 4",
     2
    ],
    [
     "이어서 다른 POST → 201, id 6",
     3
    ]
   ],
   "notes": [
    "이제 409(충돌)로, 무엇이 문제인지 알려 줘요.",
    "실패한 요청은 아무것도 남기지 않았어요 (시작 3개 + 첫 POST 1개).",
    "실패한 요청의 Session은 닫히면서 트랜잭션이 취소돼서, 다음 요청에 영향이 없어요. 번호는 6이에요. 실패한 INSERT가 번호 5를 받았다가 버렸기 때문이에요."
   ]
  },
  "s7": {
   "title": "⑦ 실행 결과",
   "lines": [
    [
     "PUT /knowledge/2 → 200 {'id': 2, 'title': '서버 실행', 'content': 'uvicorn app:app --reload 로 실행합니다.', 'category_id': 2}",
     1
    ],
    "DELETE /knowledge/3 → 204",
    "DELETE /knowledge/3 다시 → 404 {'detail': 'knowledge not found'}",
    [
     "남은 id → [1, 2]",
     2
    ]
   ],
   "notes": [
    "<code>session.get()</code>으로 찾은 객체의 속성을 바꾸고 <code>commit()</code>하면 SQLAlchemy가 바뀐 속성만 <code>UPDATE</code>로 보내요.",
    "다섯 개 주소가 모두 DB로 동작해요."
   ]
  }
 },
 "flow": [
  [
   1,
   "연결",
   "Engine",
   "DATABASE_URL"
  ],
  [
   2,
   "ORM",
   "클래스 = 테이블",
   "create_all"
  ],
  [
   3,
   "Session",
   "요청마다 하나",
   "Depends"
  ],
  [
   4,
   "조회",
   "select()",
   "session.get"
  ],
  [
   5,
   "등록",
   "add, commit",
   "201, 422"
  ],
  [
   6,
   "오류",
   "IntegrityError",
   "500 → 409"
  ],
  [
   7,
   "수정, 삭제",
   "PUT, DELETE",
   "200, 204, 404"
  ]
 ],
 "env": "Python 3.13.16, FastAPI 0.162.2, SQLAlchemy 2.1.8, PostgreSQL (노트 전용 DB)"
};
