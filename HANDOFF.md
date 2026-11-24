# 작업 인계 노트

마지막 정리: 2026-11-24 (Day27 RAG 검색 고치기까지)

이 파일은 다음에 작업을 이어받는 사람(또는 AI)이 지금 상태를 바로 알 수 있게 적은 메모예요.
쓰는 규칙은 [`STYLE.md`](STYLE.md)가 기준이고, 여기에는 **무엇이 바뀌었고 무엇이 남았는지**만 적어요.

---

## 1. 이전 버전과 달라진 것

### 1-1. Day별 노트 → 주제별 페이지 (커밋 `965bcf3` ~ `37eac9e`)

- 예전에는 `notes/dayNN.html`로 강의 날짜마다 한 쪽이었어요.
- 지금은 `topics/` 아래에 **주제마다 한 쪽**이에요. 메인 화면(`index.html`)도 갈래(모델 / 백엔드)와 카테고리로 다시 묶었어요.
- 페이지마다 예제 코드 파일 하나(`examples/...`)를 단계별로 키워 가며 설명해요.
  - 코드 줄 끝의 `#@N`은 "N단계에서 추가된 줄"이라는 표시예요.
  - `# ① 이름` 주석이 페이지의 단계 번호와 1:1로 맞아요.
- 페이지에 나오는 숫자, 표, 그림은 모두 빌드할 때 예제를 실제로 돌려서 얻은 값이에요.
  - 풀이 문장 속 실행 결과 숫자도 f-string으로 실행 값을 넣어요. 설정 값(학습률 등), 데이터 크기, 수학 상수만 글로 써요.
  - "가장 높아요" 같은 말로 쓴 주장은 `S.claim(조건, "설명")`(`tools/stepcode.py`)으로 검사해요. 값이 바뀌어 주장이 틀리면 빌드가 멈춰요. 그러면 글을 고치면 돼요.
  - 손으로 쓴 HTML인 문서 적재 세 쪽(`document-reading`, `chunking`, `indexing`)은 아직 이 검사가 없어요.
- 본문에는 강의 이야기를 쓰지 않아요. 출처는 각 페이지 맨 아래 "원본 자료"에만 적어요.
- 옛 `notes/dayNN.html`은 아직 남아 있어요. `tools/build_llms.py`(→ `all-notes.md`, `llms.txt`)는 아직 옛 노트만 읽어요.

### 1-2. LLM_wiki 자료 합치기 (진행 중)

예전에 따로 만들던 LLM_wiki(지식 그래프 실험)를 이 사이트 하나로 합치는 중이에요.
지식 그래프 방식과 페이지끼리 링크로 잇는 방식은 쓰지 않기로 했어요.

| 커밋 | 합친 자료 | 바뀐 곳 |
|---|---|---|
| `0a2783f` | PRML 4~12강 | 머신러닝 기초 1~3쪽 보강, 4쪽 새로 만듦(SVM, 판별분석, 차원 축소) |
| `e5e9bd1` | UDL 3~18장 | 아래 표 |

`e5e9bd1`에서 바뀐 페이지:

| 페이지 | 바뀐 내용 |
|---|---|
| 딥러닝 기초 (`dl-tensor-training`, `dl-training-stability`) | ReLU 꺾임, 너비와 깊이, 초기화 크기, 옵티마이저(momentum 포함), 과적합, 가중치 분포 그림. 바꿔 보기 2개 |
| CNN (`cnn`) | ⑦단계 추가(잔차 연결, 배치 정규화, 깊이 16 vs 30). 학습된 필터와 층별 기울기 그림 |
| LLM 원리 (`llm-principles`) | 위치 벡터, attention mask 그림. 토큰 순서 섞기와 attention 표 크기 바꿔 보기 |
| 생성 모델 (`generative-models`) **새 페이지** | 같은 점 데이터로 정규분포, VAE, GAN, Flow, Diffusion을 만들고 정밀도와 재현율로 비교 |
| 그래프 신경망 (`gnn`) **새 페이지** | 가라테 클럽 34명 중 4명만 보고 소속 맞히기. MLP, 퍼뜨리기, GCN, 모으는 방법, 층 수, 과평활 |

- 메인 화면에 "생성 모델과 그래프" 카테고리를 새로 만들었어요.
- 새 용어는 `terms/day06~12.json`, `day20.json`에 넣었어요.

### 1-3. 화면(공통 CSS/JS) 변경

- 토글(`details.try`)을 열었을 때 표가 잘리던 문제를 고쳤어요. 표가 넓으면 가로로 스크롤돼요.
- 그림을 페이지 폭에 맞춰 크게 보여 주는 `chart` 표시를 `topics/steps.js`, `steps.css`에 추가했어요.

---

## 2. 남은 작업 (LLM_wiki 합치기)

위에서부터 차례로 할 계획이에요. 한 번에 한 묶음씩 하고, 묶음이 끝나면 확인한 뒤 커밋해요.

1. **NLP with Transformers 11장** (끝남). 나눈 방법과 결과:
   - 끝남: 1, 3, 11장 → "LLM 원리"를 **Transformer 구조**로 다시 씀 (`llm-principles.html`, 주소는 그대로)
   - 끝남: 2, 9장 → 새 쪽 **문장 분류와 미세조정** (`text-classification.html`, 예제 `examples/classify/`, 데이터는 `tools/make_issue_examples.py`로 만듦)
   - 끝남: 4, 7장 → 새 쪽 **토큰 분류와 질문 답변** (`token-classification.html`, 예제 `examples/tokens/`, 데이터는 `tools/make_token_examples.py`로 만듦)
   - 끝남: 5, 10, 6장 → 새 쪽 **작은 GPT와 디코딩** (`gpt-generation.html`, 예제 `examples/gpt/`, 말뭉치는 `tools/make_gpt_corpus.py`가 `all-notes.md`와 주제 페이지에서 만듦)
   - 끝남: 8장 → 새 쪽 **모델 가볍게** (`model-compression.html`, 예제 `examples/compress/`, 문장 분류 쪽 데이터를 그대로 씀)
     - 시간은 모델마다 기준 모델과 번갈아 재서 "기준의 몇 배"로 봐요. 모델을 따로따로 다른 때에 재면 PC 상태 때문에 계산이 똑같은 모델끼리도 40%까지 달라졌어요. 번갈아 재니 11% 안쪽
   - 모델은 받지 않고 이 PC에 있는 `multilingual-e5-small`로 해요. GPT는 작은 말뭉치로 직접 학습해요
2. **FML 이론 17장** (진행 중). 원본은 위키 요약(1.3KB)이 아니라 `source_file:`이 가리키는 OneDrive `자료/md/Foundations-ML/`의 장별 노트(11~23KB)를 읽어요.
   - 끝남: 커널(5, 6장) → 머신러닝 4쪽 `ml-svm-lda-pca.html`에 ⑧~⑩ 추가 (커널 행렬의 고윳값, Representer 정리와 leave-one-out, 랜덤 푸리에 특성). ①~⑦ 값은 그대로
   - 끝남: 부스팅(7장) → 머신러닝 4쪽 `ml-svm-lda-pca.html`에 ⑪~⑬ 추가 (AdaBoost 직접 짜기, 지수 손실과 train 오차의 상한, L1 마진). 2쪽 ⑩에는 4쪽으로 가는 링크만 넣었어요. 2쪽 뒤에 단계를 끼우면 3쪽 번호(⑬~⑰)가 밀려서 4쪽에 이었어요. ①~⑩ 값은 그대로
     - 이 작은 데이터에서는 트리를 늘려도 valid와 교차검증 점수가 좋아지지 않았어요(트리 50개가 가장 좋음). 페이지에도 그대로 적었어요
   - 끝남: 1~4장(PAC, Rademacher와 VC 차원, 모델 선택) → 새 쪽 **학습 이론: 몇 건이면 믿을 수 있나** (`ml-learning-theory.html`, 머신러닝 기초 5/5쪽, 예제 `examples/theory/learning_theory.py`). 정답 직사각형 + 잡음 10%인 가상 세상이라 진짜 오차를 정확히 계산해요. 1~4쪽 번호를 x/5쪽으로 바꿨어요
   - 계획에 없던 장: 8장(온라인 학습), 9장(다중 분류), 10장(랭킹), 11장(회귀), 12~13장(최대 엔트로피), 14장(알고리즘 안정성), 15장(차원 축소), 16장(오토마타 학습). 17장(강화학습)은 강화학습 1쪽에 합쳤어요
3. **Elastic ANN, Elasticsearch**: 백엔드의 embedding, indexing 페이지에 보강해요.
4. **llm-internals, vLLM, Solar Open 2, llm-judge** (진행 중)
   - 끝남: llm-internals, vLLM → 새 쪽 **요즘 LLM의 부품** (`llm-architectures.html`, 예제 `examples/llm_arch/`). 모델은 받지 않고 설정만으로 GPT-2, Llama, Mistral, Mixtral을 작게 만들어 비교 (CPU 10초쯤)
   - 끝남: llm-judge → 새 쪽 **LLM으로 답 채점하기** (`topics/backend/llm-judge.html`, 예제 `examples/llm/judge.py`). Anthropic SDK에 가짜 서버(MockTransport)를 붙여 실제 요청 모양을 관찰해요. 판정과 토큰 수는 가짜 서버의 고정 값이라 페이지에 그렇게 적었어요. 위키 원본은 짧아서, 겹치는 구조화 출력과 형식 복구는 Prompt와 출력 계약 쪽으로 넘기고 캐시 비용과 채점 신뢰도를 더했어요
   - 안 하기로 함: Solar Open 2
5. **"데이터 분석 기초" 카테고리** (끝남). 원본은 OneDrive `자료/md/통계/`, `Data-Science/`, `PRML/PRML02`, LLM_wiki `sources/ml-lifecycle.md`
   - `stats-describe.html` (예제 `examples/stats/describe_estimate.py`): 척도, 요약, 확률과 포아송 확인, 가상의 1년치 10만 건으로 표본 뽑기, 중심극한정리, 신뢰구간이 맞는 비율. 도입부에 데이터 분석 흐름(ML 생애주기 6단계) 표
   - `stats-tests.html` (예제 `examples/stats/tests_regression.py`): 검정력, t검정, 분산분석과 Tukey, 상관과 χ², 심슨의 역설, 회귀 표를 numpy로 계산(statsmodels 없음), 조절 효과와 평균 중심화, 다중 비교
   - 원래 실습 데이터(추천 정확도, 호텔 리뷰 CSV)는 찾지 못해서 업무 요청 100건(`examples/stats/data/requests.csv`, 머신러닝 예제에서 복사)으로 같은 분석을 다시 했어요
   - 안 넣음: 경희대 특강 "데이터 분석 기반 인공지능"(FCM, Hopfield, BAM, 퍼지 연상 메모리). 다른 쪽과 이어지지 않아서 남겨 뒀어요
6. **강화학습 카테고리** (끝남). 원본은 OneDrive `자료/md/MARL/`(11장), `UDL/19-Reinforcement-Learning.md`, `Foundations-ML/17-Reinforcement-Learning.md`
   - `rl-basics.html` (예제 `examples/rl/grid_world.py`): 4 × 8 격자 배달 로봇. 벨만 방정식, 가치 반복과 정책 반복, 밴딧, MC와 TD(0), SARSA와 Q-learning
   - `rl-deep.html` (예제 `examples/rl/deep_rl.py`, 빌드 6분): 막대 세우기를 물리식으로 직접 짬(gym 없음). DQN 장치 하나씩 빼기, REINFORCE와 베이스라인, A2C(λ 바꿔 보기), PPO
   - `rl-games.html` (예제 `examples/rl/matrix_games.py`): 행렬 게임의 내시, 미니맥스와 상관 균형 선형 계획, 가상 플레이, 후회 매칭, IGA와 WoLF, 반복 죄수의 딜레마
   - `rl-multiagent.html` (예제 `examples/rl/multi_agent.py`): 오르막 게임, 움직이는 과녁, 함께 드는 상자(중앙, 각자, VDN), 표 크기
   - 책과 다르게 나온 결과도 그대로 적었어요: TD(0)보다 MC가 빠름(1쪽 ⑤), 경험 재생만 쓴 DQN이 둘 다 쓴 DQN보다 나음(2쪽 ②), 둘뿐이면 VDN이 독립 학습보다 낫지 않음(4쪽 ③)
7. **밋업 자료 중 이론** (일부 끝남). 방침이 바뀌었어요: 읽기 노트 대신, 기존 쪽에 없는 이론만 골라 예제가 있는 주제 쪽으로 만들어요. 원본은 OneDrive `자료/md/Upstage10Talks/`
   - 새 카테고리 "추천과 평생 학습"
     - `recsys.html` (예제 `examples/recsys/recommend.py`): 가상 이용 기록 600 × 400. 인기순, 아이템과 사용자 이웃, BPR, EASE, LightGCN, 기록 수별 성능, 인기 편향. 경희대 추천 특강과 Lec01
     - `continual.html` (예제 `examples/continual/remember_forget.py`): 손글씨 숫자와 픽셀 섞은 숫자로 파국적 망각, 리플레이, EWC. 숫자 3 지우기를 다시 학습, 남은 것으로 더, 경사 상승, 작업 벡터 빼기로 비교. Lec03
   - `pretrain-data.html` ("LLM 학습 데이터와 평가", 예제 `examples/pretrain_data/mix_tokenizer.py`): 한국어 노트, 표준 라이브러리 설명문과 소스로 규칙 거르기, 중복 빼기, MinHash, BPE 압축률, 섞는 비율과 대리 토크나이저, 회귀로 비율 고르기. Lec04, Lec06. 영어와 코드 말뭉치가 파이썬 버전에 따라 조금 달라져요
   - `backend/multi-ref.html` ("LLM 활용", 예제 `examples/eval_refs/multi_ref.py`): `pages.jsonl` 문장을 해요/합니다/한다체 정답 셋으로 바꾸고 오류 둘을 넣어 BLEU, GLEU, 정답 합치기 네 가지, 짝마다 줄 세우기, 정답 늘리기. Lec05
   - 새 카테고리 "성능과 하드웨어": `backend/simd-speed.html` (예제 `examples/speed/simd_numpy.py`): 반복문과 NumPy, SIMD 다섯 단계, 자료형 폭, stride와 전치, 캐시 크기, 임시 배열, 마스크. 시간을 재는 쪽이라 다시 빌드하면 숫자가 조금씩 바뀌어요(주장은 `S.claim`으로 넉넉한 배수만 확인). LLM_wiki `everyone-should-know-simd`, PyCon 가속기 발표
   - 책이나 논문과 다르게 나온 결과도 그대로 적었어요: BPR이 단순한 이웃보다 낮음(추천 ④), 회귀가 고른 비율이 "서비스 비율 그대로"보다 조금 나쁨(데이터 ⑦)
   - `model/rl-reasoning.html` ("강화학습", 예제 `examples/reasoning_rl/verifiable_rl.py`): (a + b + c)의 일의 자리를 생각하고 답하기 vs 바로 답하기, 생각 SFT, pass@k, REINFORCE와 GRPO, 신호 없는 그룹, 단계별 보상, 미리 고른 문제. Lec04 3부, Lec09
     - 다르게 나온 결과: 미리 가끔 맞히는 문제만 고르면 오히려 낮아짐(⑥, 목록이 금방 낡음), SFT에서 출발하면 단계별 보상의 차이가 거의 없음(⑤)
   - `model/eval-select.html` ("LLM 학습 데이터와 평가", 예제 `examples/eval_select/pick_benchmarks.py`): 크기 8~48 모델 여섯 개의 에폭별 체크포인트로 평가 넷(작음, 큼, 쉬움, 픽셀 섞음) × 지표 둘(정답률, 손실)의 SNR, 문제 수, 초반 순서, 상관 군집. Lec04 1부
     - 다르게 볼 점: 픽셀 섞은 평가의 손실은 SNR이 높은데 순서가 거꾸로(⑤)
   - `model/codeswitch.html` ("언어 모델로 문제 풀기", 예제 `examples/codeswitch/mixed_language.py`): 노트 문장의 코드 스위칭 세기, 용어집 한국어/영어 이름으로 e5 설명 찾기, 교차 언어 일관성, 조금씩 섞기, 함께 적기. Lec10. 용어집(`terms/`)을 읽기 때문에 용어를 더하면 숫자가 바뀌어요. 용어를 더한 뒤 다시 빌드해요
   - 아직 안 넣은 Upstage 자료: Lec02(사람과 AI 협업), Lec09의 문화 VQA 부분, SP01(OpenClaw), SP03(영상 생성)
   - 아직 안 넣은 밋업 자료(LLM_wiki `wiki/sources/`): PyCon의 엣지 AI와 FastAPI 모듈러 모놀리스, OSSummit OpenSSF, 가짜연구소(에이전트 운영, 피지컬 AI), 컨텍스트 엔지니어링 밋업, 카카오 에이전트 자동화, open-code-review. 대부분 경험담이라 이론 쪽으로 만들 거리가 적어요
8. **업무, 도메인 자료**도 넣어요(본인만 보는 사이트라서 포함).
9. **정보처리기사 정리**를 넣어요.
10. **UDL 21장(윤리)**: 아직 안 넣었어요.

넣지 않기로 한 것:
- 트렌드 자료(daily-trends, 회사별 트렌드)는 업데이트가 멈춰서 빼요.
- 위키 자체를 관리하던 메타 파일은 옮기지 않아요.
- 위키의 개체(entity) 항목은 페이지가 아니라 용어집 항목으로 옮겨요.

그 밖에 해 둘 일:
- (끝남) `README.md`를 주제별 구성으로 고쳤어요. 카테고리가 늘면 README의 "무엇이 있나" 표도 같이 고쳐요.
- (끝남) `tools/build_llms.py`가 `index.html`의 순서대로 주제 페이지를 읽어, `-data.js`의 실행 결과(표, 줄 상자, 그림 설명)까지 `all-notes.md`에 넣어요. 옛 RAG 세 쪽(`document-reading`, `chunking`, `indexing`)은 결과 형식이 달라서 본문 글만 들어가요. 페이지를 고친 뒤 `python tools/build_llms.py`를 다시 돌려요.
- 부트캠프 강의는 계속 진행 중이에요. 새 강의는 맞는 주제 페이지에 보강하거나 새 주제 페이지로 넣어요.

---

## 3. 작업할 때 알아 둘 것

### Day27 (2026-11-24) LangChain RAG와 Reranking

- `topics/backend/rag-rerank.html` "RAG 검색 고치기" (생성기 `tools/pages/rag_rerank.py`). "검색과 RAG" 카테고리의 RAG 서비스 다음, RAG 지도의 "벡터 검색 → 검색 고치기 → Prompt와 Context"
- 코드는 새 파일이 아니라 `examples/rag/rag_service.py`를 ⑨~⑯으로 이어 키워요(문서 적재 세 쪽이 `ingest.py` 하나를 나눠 쓰는 것과 같아요). RAG 서비스 쪽은 `range [1, 8]`, 이 쪽은 `[9, 16]`. ⑨~⑯용 import 줄은 `#@13`, `#@15` 표시로 위쪽에 있어요
  - RAG 서비스 쪽 생성기도 다시 돌렸어요(이 쪽으로 가는 링크 추가). 출력 값은 그대로이고, 코드 패널에 ⑨~⑯ 줄이 숨은 채로 들어가요
- 조각: 인덱싱 쪽 12개 + `examples/rag/data/more_chunks.json` 14개(`tools/make_rag_more.py`로 만든 가상 안내문: 스터디룸, 도서 예약, 프로그램, 기기, 2025년과 2026년 공지). 2025년 공지만 `current: false`
- 질문 6개와 정답 조각은 ⑨의 `EVAL`. 다시 쓰기는 가짜 서버(`REWRITES`, 미리 정한 값), 재정렬은 `BAAI/bge-reranker-v2-m3`를 **CUDA GPU에서 실제로** 돌려요. GPU가 없으면 ⑬에서 멈춰요(CPU로 바꾸지 않게 해 둠)
  - 모델은 2026-11-24 사용자 허락으로 받았어요(HF 캐시 약 2.2GB). 이 쪽 전체에 GPU 메모리 최대 약 2.6GB
  - 빌드: `HF_HUB_OFFLINE=1 C:/practice_assignment/.venv/Scripts/python.exe tools/pages/rag_rerank.py` (1분 안쪽). ⑮의 시간은 실행마다 조금 달라져요(주장은 순서만 `S.claim`으로 확인)
- 실제로 나온 결과: 다시 쓰기가 후속 질문 둘을, 재정렬이 '안 가면' 질문을, 용어 넓히기가 '상호대차'(e5 11위, 후보 밖)를 고쳐요. 상호대차는 재정렬 1위지만 점수 0.002라 모델도 잘 모름. 2025년과 2026년 공지는 e5와 재정렬 점수가 거의 같아서 판 조건으로 걸러요
- 원본: `강의자료-원본/Day27_20261124_LangChain_RAG_Reranking.html`. 용어 `terms/day27.json`, 용어집의 Day27 링크는 이 쪽으로
- 다음 강의(Recall@k, MRR, nDCG 같은 검색 평가)는 ⑨의 `EVAL`과 ⑮의 비교표를 이어 쓰면 돼요

### Day26 (2026-11-23) Basic RAG와 LangChain

- `topics/backend/rag-service.html` (예제 `examples/rag/rag_service.py`, 생성기 `tools/pages/rag_service.py`). "검색과 RAG" 카테고리, RAG 지도의 "질문에 답할 때" 줄 끝(검사와 출처)
- 기존 쪽과 겹치지 않게: 검색은 인덱싱 ⑱, Prompt 틀과 출처 규칙은 Prompt와 출력 계약, SDK 호출은 LLM API 클라이언트로 링크하고, 이 쪽은 잇는 부분(Runnable, Document, 후보 수와 넣을 수, 허용 출처는 included_docs에서만, raw/parsed/parsing_error 묶음, 정보 부족과 실패 구분, 422와 502)만 다뤄요
- 검색 대상은 인덱싱 쪽의 가람 도서관 조각 12개(`examples/ingest/out/chunks.json`), e5로 메모리에서 검색. 답은 ChatOpenAI에 가짜 서버(MockTransport)를 붙여 받음(키, 비용 없음)
- 이 PC의 venv에 `langchain-core` 1.8.4, `langchain-openai` 1.8.5를 설치했어요(사용자 허락). 다른 PC에서 다시 빌드하려면 같은 두 패키지가 필요해요
- 판에 따라 다른 점을 그대로 적었어요: 깨진 JSON과 글자 ID는 parsing_error가 아니라 invoke 예외(generation_error)로 옴. 거절만 parsing_error에 담김
- 원본: `강의자료-원본/Day26_20261123_Basic_RAG_LangChain.html` (저장소에는 안 올라가요). 용어 `terms/day26.json`, 용어집의 Day26 링크는 이 쪽으로

### 백엔드 외전 (2026-11-20)

- `topics/backend/service-flow.html`: 로직 밖에서 서비스를 완성하기. 기존 CRUD 설명을 연결하여 설정, DB 준비, Session 수명, autobegin, Service의 트랜잭션 책임, Router와 Health, Postman 검증 순서를 설명해요.
- 원고와 실제 실행 검증: `tools/pages/service_flow.py`. 기본은 임시 SQLite와 TestClient, `--newman`은 실제 Uvicorn 서버에 Collection을 실행해요. Python은 필요한 패키지가 있는 환경으로 실행해요.
- 독립 예제: `examples/service_flow/service.py` 한 파일. 같은 폴더의 README, requirements, `.env.example`, Postman Collection과 Environment를 함께 사용해요. 실제 원본 프로젝트 코드나 DB에 의존하지 않아요.
- `terms/day25.json` 용어를 추가했어요. `glossary.html`의 Day25 링크는 외전으로 연결해요. 옛 `notes/day25.html`은 만들지 않았어요.
- 이 페이지의 검증은 실제 PostgreSQL, E5, LLM, 동시 부하 시험을 포함하지 않아요. 원본 자료는 계속 `강의자료-원본/`에만 둬요.

### 페이지 만들기

- 페이지마다 `tools/pages/*.py` 빌드 스크립트가 있어요. 실행하면 예제를 단계별로 돌리고, 결과를 담아 `topics/.../*.html`과 `*-data.js`를 다시 만들어요.
  - 단계를 나누고 실행하는 도구: `tools/stepcode.py`
  - 페이지를 조립하는 도구: `tools/pagekit.py`
  - 그림 스타일: `tools/visuals/vstyle.py`
- 그림은 `topics/model/img/<주제>/`에 저장돼요.
- 용어를 추가하면 `terms/dayNN.json`에 넣고 `python build_glossary.py`를 실행해요.
- 처음 나오는 용어는 그 자리에서 설명해야 해요. 페이지를 고친 뒤에는 "아래에서 처음 설명하는 용어를 위에서 먼저 쓰고 있지 않은지" 검사해요.
- 가운뎃점 "·"은 쓰지 않아요.

### 실행 환경

- 시스템 Python 3.13에 scikit-learn, torch(CPU), torchvision을 설치해서 빌드했어요. 수업용 가상환경은 지금 PC에 없어요.
- `llm_principles.py`(Transformer 구조 쪽)는 e5-small 모델로 층 안의 계산을 직접 해 보는 쪽이라, 모델이 없는 PC에서는 아무 파일도 바꾸지 않고 끝나요. 이 쪽을 고치려면 e5-small이 있는 PC에서 빌드해요.
- `gen_models.py`는 모델 다섯 개를 학습해서 CPU로 2~4분 걸려요.
- 다른 PC(`C:\practice_assignment\.venv`, Python 3.13.16, PyTorch 2.14 cu130, scikit-learn 1.8)에서 다시 빌드한 적이 있어요. 그때 생성 모델 쪽과 CNN ⑦의 값이 PyTorch 버전 차이로 조금 바뀌었고, 말로 쓴 주장은 모두 그대로 맞았어요.
- 로컬에서 보기: 저장소 폴더에서 `python -m http.server 5174`를 실행하고 http://localhost:5174 를 열어요.

### Git

- 이 폴더는 따로 된 저장소예요: https://github.com/Yiji-1015/sesac-notes (`main` 브랜치, GitHub Pages로 배포).
- 상위 `personal` 저장소와는 별개예요. 이 폴더의 작업을 상위 저장소로 커밋하거나 push하지 않아요.
- 줄바꿈 문자(CRLF/LF)만 다른 파일 60여 개가 늘 "수정됨"으로 보여요. 내용은 같으니 커밋에 넣지 않아요.
  - 실제로 바뀐 파일만 찾으려면 `git diff --ignore-cr-at-eol --name-only`와 `git ls-files --others --exclude-standard`를 써요.
- 커밋 메시지는 파일에 써서 `git commit -F 파일`로 넣어요. PowerShell에서 표준 입력으로 넘기면 깨져요.
