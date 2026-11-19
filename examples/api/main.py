# 지식 관리 API 전체 코드. 노트의 단계 번호(①, ②, …)가 이 파일의 번호 주석과 1:1로 맞아요.
# 실행: uvicorn main:app --reload   →  http://127.0.0.1:8000/docs
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import json                                   #@2
from pathlib import Path                      #@2
from fastapi import FastAPI                   #@1
from fastapi import HTTPException             #@3
from pydantic import BaseModel, Field         #@4

# ① 앱 만들기
app = FastAPI(title="지식 관리 API")

@app.get("/health")
def health():
    return {"status": "ok"}

# ② 파일 저장소
DATA = Path(__file__).resolve().parent / "data" / "knowledge.json"

def read_data():
    return json.loads(DATA.read_text(encoding="utf-8"))

def write_data(data):
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ③ 조회하기 (GET)
@app.get("/knowledge")
def list_knowledge(category: str | None = None):
    entries = read_data()["entries"]
    if category:
        entries = [e for e in entries if e["category"] == category]
    return entries

@app.get("/knowledge/{knowledge_id}")
def get_knowledge(knowledge_id: int):
    for e in read_data()["entries"]:
        if e["id"] == knowledge_id:
            return e
    raise HTTPException(status_code=404, detail="knowledge not found")

# ④ 입력 모양 정하고 등록하기 (POST)
class KnowledgeWrite(BaseModel):
    title: str = Field(min_length=1, max_length=80)
    category: str = Field(min_length=1, max_length=30)
    content: str = Field(min_length=1)

@app.post("/knowledge", status_code=201)
def create_knowledge(payload: KnowledgeWrite):
    data = read_data()
    entry = {"id": data["next_id"], **payload.model_dump()}
    data["entries"].append(entry)
    data["next_id"] += 1
    write_data(data)
    return entry

# ⑤ 고치기 (PUT)
@app.put("/knowledge/{knowledge_id}")
def update_knowledge(knowledge_id: int, payload: KnowledgeWrite):
    data = read_data()
    for e in data["entries"]:
        if e["id"] == knowledge_id:
            e.update(payload.model_dump())
            write_data(data)
            return e
    raise HTTPException(status_code=404, detail="knowledge not found")

# ⑥ 지우기 (DELETE)
@app.delete("/knowledge/{knowledge_id}", status_code=204)
def delete_knowledge(knowledge_id: int):
    data = read_data()
    kept = [e for e in data["entries"] if e["id"] != knowledge_id]
    if len(kept) == len(data["entries"]):
        raise HTTPException(status_code=404, detail="knowledge not found")
    data["entries"] = kept
    write_data(data)
