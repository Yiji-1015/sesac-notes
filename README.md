# 📚 SeSAC 부트캠프 학습 노트

부트캠프 강의와 책, 세미나 자료를 **주제별 페이지**로 다시 정리한 학습 노트예요.
페이지마다 예제 코드 파일 하나가 ①, ②, ③ 단계로 자라고, 본문의 숫자와 표, 그림은 모두 그 코드를 **실제로 실행한 결과**예요.

- 🌐 사이트: https://yiji-1015.github.io/sesac-notes/
- 📚 용어 사전: https://yiji-1015.github.io/sesac-notes/glossary.html
- 🤖 AI용 전체 본문(글만): [`all-notes.md`](all-notes.md), 안내판: [`llms.txt`](llms.txt)

## 무엇이 있나

| 트랙 | 카테고리 | 쪽 |
|---|---|---|
| 🧠 모델 | 데이터 분석 기초 | 요약하고 추정하기, 가설 검정과 회귀 분석 |
| | 머신러닝 기초 | 회귀와 분류, 트리와 교차검증, 평가지표와 군집, SVM과 차원 축소, 학습 이론 |
| | 딥러닝 기초 | Tensor와 신경망 학습, 학습을 안정시키기 |
| | 이미지 | CNN, OpenCV 전처리, 전이학습, 객체 탐지, OCR과 문서 구조화 |
| | 순서 모델과 Transformer | 순서 모델, Transformer, 작은 GPT, 요즘 LLM의 부품 |
| | 언어 모델로 문제 풀기 | 문장 분류, 토큰 분류, 코드 스위칭, 모델 가볍게 |
| | LLM 학습 데이터와 평가 | LLM 학습 데이터, 학습 중 평가 고르기 |
| | 생성 모델과 그래프 | 생성 모델, 그래프 신경망 |
| | 강화학습 | 기초, 심층 강화학습, 게임과 균형, 다중 에이전트, 추론 모델과 검증 가능한 보상 |
| | 추천과 평생 학습 | 추천 시스템, 평생 학습과 머신 언러닝 |
| 🛠️ 백엔드 | API 서버, 데이터베이스 | FastAPI, PostgreSQL, 둘의 연동 |
| | LLM 활용 | Prompt와 Context Engineering, LLM API 클라이언트, LLM 채점, 여러 정답으로 채점하기 |
| | 검색과 RAG | Embedding과 벡터 검색, 문서 읽기, Chunking, 인덱싱 |
| | Agent | AI Agent와 도구 |
| | 성능과 하드웨어 | SIMD와 메모리 |
| | 외전 | 로직 밖에서 서비스를 완성하기 |

처음에 쓴 Day별 노트(`notes/day06.html` ~)도 그대로 남아 있고, 메인 화면의 업데이트 기록에서 찾을 수 있어요.

## 폴더 구성

```
index.html              메인 목록 (카테고리 카드, 업데이트 기록)
topics/model/, backend/ 주제 페이지 X.html, 실행 결과 X-data.js, 그림 img/
topics/steps.js, .css   주제 페이지의 단계 이동, 자라는 코드 보기, 결과 상자 그리기
examples/<주제>/        페이지마다 하나씩 자라는 예제 코드 (#@N = N단계에서 추가된 줄)
tools/pages/<주제>.py   페이지 원고: 예제를 단계별로 실행하고, 결과와 그림을 만들고, HTML을 조립
tools/stepcode.py       예제를 단계까지 잘라 실행하기, 결과 상자(S.box, S.table, S.images), S.claim
tools/pagekit.py        주제 페이지의 공통 틀 (한 문장, 왜, 통째로 한 번, 한 단계씩, 실제로 할 때, 원본 자료)
terms/dayNN.json        용어 (→ build_glossary.py → glossary.js, glossary.html)
tools/build_llms.py     all-notes.md, llms.txt 만들기
notes/, cmds/           처음에 쓴 Day별 노트와 명령어 조각
STYLE.md                쓰는 기준 (먼저 읽기)
HANDOFF.md              지금까지 한 일, 남은 일, 작업할 때 알아 둘 것
```

## 주제 페이지를 추가하거나 고칠 때

1. `STYLE.md`를 먼저 읽어요.
2. 예제 코드 `examples/<주제>/<파일>.py`를 단계 번호 주석(`# ① 이름`)과 `#@N` 표시로 써요.
3. 원고 `tools/pages/<주제>.py`를 써요. 비슷한 기존 원고(예: `tools/pages/continual.py`)를 본보기로 삼아요.
   - 본문에 쓴 주장은 `S.claim(...)`으로 실행 값과 맞는지 확인해요. 값이 바뀌어 주장이 틀리면 빌드가 멈춰요.
4. `python tools/pages/<주제>.py` → `topics/.../<주제>.html`, `-data.js`, 그림이 생겨요.
5. `index.html`에 카드와 업데이트 기록을 더해요.
6. 용어: `terms/dayNN.json`에 더하고 `python build_glossary.py`.
7. `python tools/build_llms.py`로 AI용 본문을 다시 만들어요.
8. 커밋하고 푸시하면 1~2분 뒤 사이트에 반영돼요.

> 수강생 개인 정리본이에요. 강의 원본 PDF는 포함하지 않으며, 값과 그림은 직접 실행하고 만든 것이에요.
