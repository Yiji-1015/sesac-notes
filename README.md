# 📚 SeSAC Vision·멀티모달 AI 부트캠프 학습 노트

비전공 수강생이 매일 강의 PDF를 바탕으로 정리한 학습 노트예요.
Day마다 **3줄 요약 → 이론 ↔ 명령어 → 필요한 패키지 → 🤔 왜 그런지 → 그림 · 직접 만져보는 체험** 순서로 정리했어요.

- 🌐 사이트: https://yiji-1015.github.io/sesac-notes/
- 📚 용어 사전: https://yiji-1015.github.io/sesac-notes/glossary.html
- 🤖 AI용 전체 본문(글만): [`all-notes.md`](all-notes.md) · 안내판: [`llms.txt`](llms.txt)

| Day | 날짜 | 주제 |
|---|---|---|
| 6 | 10/26 | 회귀·분류·KNN·모델 비교 |
| 7 | 10/27 | 트리·앙상블·교차검증 |
| 8 | 10/28 | 평가지표·특성설계·K-Means |
| 9 | 10/29 | GPU·Tensor·ANN |
| 10 | 10/30 | Optimizer·Regularization |
| 11 | 11/02 | CNN·이미지 분류·전이학습 |
| 12 | 11/02 | 순서 모델 (RNN·LSTM·Autoencoder·Transformer) |
| 13 | 11/04 | OpenCV 이미지·영상 전처리 |
| 14 | 11/05 | 전이학습과 오류 분석 |
| 15 | 11/06 | 객체탐지와 YOLO |
| 16 | 11/09 | OCR·문서 구조화·VLM·엣지 배포 |

## 폴더 구성

```
index.html          메인 목록
glossary.html       용어 사전
notes/dayNN.html    Day별 노트
notes/img/dayNN/    그림
style.css, note.js  공통 디자인 · 테마/글꼴 버튼 · 용어 말풍선 · 그림 크게 보기
terms/dayNN.json    Day별 용어 (→ build_glossary.py → glossary.js)
tools/visuals/      그림 만드는 스크립트 (make_dayNN.py)
tools/build_llms.py all-notes.md · llms.txt 만들기
```

## 새 Day 추가할 때

1. `notes/_template.html` 복사 → `notes/dayNN.html` 작성, `index.html`에 카드 추가, 앞 Day의 "다음 →" 링크 연결
2. `terms/dayNN.json` 작성 → `python build_glossary.py`
3. 그림: `tools/visuals/make_dayNN.py` → 수업 가상환경 파이썬으로 실행
4. `python tools/build_llms.py` → 커밋 · 푸시하면 1~2분 뒤 사이트에 반영

> 수강생 개인 정리본이에요. 강의 원본 PDF는 포함하지 않으며, 그림은 직접 계산·생성했어요.
