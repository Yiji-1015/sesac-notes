# 백엔드 외전: 로직 밖에서 서비스를 완성하기

가상 노트 하나로 설정, DB 준비, Session, 트랜잭션, API 계약, Postman 검증을 연결해요. 모든 핵심 코드는 `service.py` 하나에 있어요. PostgreSQL이나 LLM에 접속하지 않고 이 폴더의 SQLite 파일만 사용해요.

PowerShell에서 이 폴더로 이동한 뒤 실행해요.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe service.py prepare
.\.venv\Scripts\python.exe service.py serve
```

`Copy-Item`은 처음에만 실행해요. 이미 설정한 `.env`를 덮어쓰지 않아요. 가상환경을 활성화하는 대신 같은 Python의 경로를 직접 지정했어요. 기존 환경이 있다면 그 환경에 의존성을 설치하고 같은 Python으로 실행해도 돼요.

서버는 `http://127.0.0.1:8015`에서 실행돼요. `/docs`는 자동 API 문서, `/live`는 프로세스 응답, `/health`는 DB 연결과 두 테이블 조회 확인이에요. `prepare`를 실행하기 전에도 `/live`는 성공할 수 있지만 `/health`는 503이에요. `serve`는 DB 테이블이나 시작 데이터를 자동 생성하지 않아요.

`prepare`는 테이블을 만들고, 노트가 하나도 없을 때만 시작 노트와 생성 기록을 넣어요. 이미 있는 테이블의 열을 변경하거나 데이터를 초기화하지 않아요. DB 구조를 바꿨다면 마이그레이션이 필요해요. 서버를 종료할 때는 Ctrl+C를 눌러요. 코드나 `.env`를 고쳤다면 재시작해요.

## Postman으로 전체 흐름 실행하기

1. `service.postman_collection.json`과 `service.postman_environment.json`을 모두 Import해요.
2. Environment에서 `노트 서비스 - Local`을 선택해요. `base_url`이 서버 주소와 같은지 확인해요.
3. 새로 시작할 때 `created_id`, `expected_note`, `baseline_count`는 비어 있어요. 이전 실행에서 ID가 남아 있다면 먼저 해당 노트를 조회해 상태를 확인해요.
4. Collection Runner에서 01부터 12까지 순서대로 한 번 실행해요. 409, 422, 404도 각각 기대한 오류 응답이므로 해당 검사가 통과해야 해요.
5. 다른 터미널에서 아래 명령으로 실제 DB 상태를 확인해요.

```powershell
.\.venv\Scripts\python.exe service.py verify
```

깨끗한 예제 DB는 처음에 노트 하나와 변경 기록 하나가 있어요. 마지막에는 이번에 만든 노트와 그 기록만 삭제되어 시작 수량으로 돌아와요. 검증 실패 후 다시 실행할 때는 무조건 새 POST를 보내지 말고 남은 생성 ID부터 확인해요. 이 예제의 삭제는 변경 기록도 함께 지우므로 영구 감사 로그로 사용하는 설계는 아니에요.

Node.js가 있다면 같은 두 JSON 파일을 Newman으로 실행할 수도 있어요.

```powershell
npx --yes newman run service.postman_collection.json -e service.postman_environment.json --bail
```

두 JSON은 서버 코드가 아니에요. Collection은 요청과 검사 순서, Environment는 요청에서 채울 값이에요. JSON을 다시 생성하려면 `service.py postman`을 실행해요. 이 명령은 두 예제 파일을 덮어쓰므로 Postman에서 바꾼 내용을 먼저 보관해요.

## 페이지 재생성

저장소 루트에서 필요한 패키지가 설치된 Python으로 실행해요.

```powershell
python tools/pages/service_flow.py
```

단계별 예제를 임시 폴더에서 실제 실행해 HTML과 데이터 파일을 만들어요. HTTP 요청 검증은 TestClient를 사용해요. 아래 옵션은 추가로 실제 로컬 Uvicorn 서버를 열고 같은 Collection을 Newman으로 실행해요. Node.js와 npm이 필요하며 첫 실행에 Newman을 내려받을 수 있어요.

```powershell
python tools/pages/service_flow.py --newman
```

테스트용 SQLite 파일은 임시 폴더에서 만들고 정리해요. 실패한 트랜잭션 실험은 변경 기록 함수에 의도적으로 예외를 발생시켜 롤백을 확인해요. 실제 PostgreSQL, 부하, 동시 수정, E5, LLM 장애를 검증한 결과는 아니에요.
