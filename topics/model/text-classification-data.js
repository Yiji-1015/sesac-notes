window.PAGE = {
 "code": [
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import numpy as np",
   "s": 1
  },
  {
   "t": "import pandas as pd",
   "s": 1
  },
  {
   "t": "from sklearn.metrics import f1_score",
   "s": 1
  },
  {
   "t": "from sklearn.feature_extraction.text import TfidfVectorizer",
   "s": 2
  },
  {
   "t": "from sklearn.naive_bayes import MultinomialNB",
   "s": 2
  },
  {
   "t": "from sklearn.pipeline import make_pipeline",
   "s": 2
  },
  {
   "t": "import torch",
   "s": 3
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 3
  },
  {
   "t": "from transformers import AutoTokenizer, AutoModel",
   "s": 3
  },
  {
   "t": "from sklearn.linear_model import LogisticRegression",
   "s": 4
  },
  {
   "t": "from sklearn.metrics import confusion_matrix",
   "s": 6
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent / \"data\"",
   "s": 1
  },
  {
   "t": "MODEL = \"intfloat/multilingual-e5-small\"",
   "s": 3
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 데이터 읽고 나누기",
   "s": 1
  },
  {
   "t": "df = pd.read_csv(DATA / \"issues.csv\")",
   "s": 1
  },
  {
   "t": "LABELS = [\"버그\", \"기능 요청\", \"질문\"]",
   "s": 1
  },
  {
   "t": "part = {s: df[df[\"split\"] == s] for s in (\"train\", \"valid\", \"test\", \"unlabeled\")}",
   "s": 1
  },
  {
   "t": "y = {s: d[\"label\"].map(LABELS.index).to_numpy() for s, d in part.items()}  # 종류 이름 → 번호 0, 1, 2",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def macro_f1(y_true, pred):  # 종류마다 F1을 구해 똑같은 무게로 평균",
   "s": 1
  },
  {
   "t": "    return f1_score(y_true, pred, average=\"macro\")",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 기준선: 단어를 세는 모델",
   "s": 2
  },
  {
   "t": "def bag_model(analyzer=\"word\", ngram=(1, 1)):  # 글 → 단어마다 TF-IDF 점수 → Naive Bayes",
   "s": 2
  },
  {
   "t": "    return make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ngram), MultinomialNB())",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "nb = bag_model().fit(part[\"train\"][\"text\"], y[\"train\"])",
   "s": 2
  },
  {
   "t": "nb_f1 = macro_f1(y[\"test\"], nb.predict(part[\"test\"][\"text\"]))",
   "s": 2
  },
  {
   "t": "train_words = {w for t in part[\"train\"][\"text\"] for w in t.split()}",
   "s": 2
  },
  {
   "t": "test_words = [w for t in part[\"test\"][\"text\"] for w in t.split()]",
   "s": 2
  },
  {
   "t": "unseen = np.mean([w not in train_words for w in test_words])  # test의 단어 중 train에 한 번도 없던 비율",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ e5로 문장 벡터 만들기 (특성 추출)",
   "s": 3
  },
  {
   "t": "tok = AutoTokenizer.from_pretrained(MODEL)",
   "s": 3
  },
  {
   "t": "body = AutoModel.from_pretrained(MODEL).eval()",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def encode(texts):  # e5는 앞에 \"query: \"를 붙여 쓰도록 학습됐어요",
   "s": 3
  },
  {
   "t": "    return tok([\"query: \" + t for t in texts], padding=True, return_tensors=\"pt\")",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def mean_pool(hidden, mask):  # 채운 자리를 빼고 토큰 벡터들의 평균",
   "s": 3
  },
  {
   "t": "    m = mask[..., None].float()",
   "s": 3
  },
  {
   "t": "    return (hidden * m).sum(1) / m.sum(1)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "def embed(texts, model=body):",
   "s": 3
  },
  {
   "t": "    with torch.no_grad():",
   "s": 3
  },
  {
   "t": "        b = encode(texts)",
   "s": 3
  },
  {
   "t": "        return F.normalize(mean_pool(model(**b).last_hidden_state, b[\"attention_mask\"]), dim=-1).numpy()",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "vec = {s: embed(d[\"text\"].tolist()) for s, d in part.items()}  # 문장마다 384칸, 길이 1",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 고정한 벡터 위에 분류기 얹기",
   "s": 4
  },
  {
   "t": "frozen = LogisticRegression(max_iter=2000, C=10).fit(vec[\"train\"], y[\"train\"])",
   "s": 4
  },
  {
   "t": "frozen_f1 = macro_f1(y[\"test\"], frozen.predict(vec[\"test\"]))",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 미세조정: 모델 안의 가중치까지 고치기",
   "s": 5
  },
  {
   "t": "class Classifier(torch.nn.Module):  # e5 몸통 + 384 → 3 분류 층",
   "s": 5
  },
  {
   "t": "    def __init__(self):",
   "s": 5
  },
  {
   "t": "        super().__init__()",
   "s": 5
  },
  {
   "t": "        self.body = AutoModel.from_pretrained(MODEL)",
   "s": 5
  },
  {
   "t": "        self.head = torch.nn.Linear(384, len(LABELS))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "    def forward(self, b):",
   "s": 5
  },
  {
   "t": "        return self.head(mean_pool(self.body(**b).last_hidden_state, b[\"attention_mask\"]))",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "def fine_tune(train_layers, epochs=6, lr=5e-5, head_lr=1e-3):  # train_layers: 위에서부터 몇 층을 고칠까 (12면 전부)",
   "s": 5
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 5
  },
  {
   "t": "    clf = Classifier()",
   "s": 5
  },
  {
   "t": "    clf.body.requires_grad_(False)  # 일단 몸통 전체를 고정하고",
   "s": 5
  },
  {
   "t": "    for layer in clf.body.encoder.layer[12 - train_layers:]:",
   "s": 5
  },
  {
   "t": "        layer.requires_grad_(True)  # 위쪽 train_layers개 층만 풀기",
   "s": 5
  },
  {
   "t": "    opt = torch.optim.AdamW([{\"params\": [p for p in clf.body.parameters() if p.requires_grad], \"lr\": lr},",
   "s": 5
  },
  {
   "t": "                             {\"params\": clf.head.parameters(), \"lr\": head_lr}])",
   "s": 5
  },
  {
   "t": "    X, Y, log = part[\"train\"][\"text\"].tolist(), torch.tensor(y[\"train\"]), []",
   "s": 5
  },
  {
   "t": "    for epoch in range(epochs):",
   "s": 5
  },
  {
   "t": "        clf.train()",
   "s": 5
  },
  {
   "t": "        for idx in torch.randperm(len(X)).split(16):  # 16개씩 섞어서",
   "s": 5
  },
  {
   "t": "            loss = F.cross_entropy(clf(encode([X[i] for i in idx])), Y[idx])",
   "s": 5
  },
  {
   "t": "            opt.zero_grad()",
   "s": 5
  },
  {
   "t": "            loss.backward()",
   "s": 5
  },
  {
   "t": "            opt.step()",
   "s": 5
  },
  {
   "t": "        clf.eval()",
   "s": 5
  },
  {
   "t": "        with torch.no_grad():",
   "s": 5
  },
  {
   "t": "            test_pred = clf(encode(part[\"test\"][\"text\"].tolist())).argmax(1).numpy()",
   "s": 5
  },
  {
   "t": "        log.append({\"epoch\": epoch + 1, \"train_loss\": loss.item(), \"test_f1\": macro_f1(y[\"test\"], test_pred)})",
   "s": 5
  },
  {
   "t": "    return clf, log",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "tuned_all, log_all = fine_tune(train_layers=12)",
   "s": 5
  },
  {
   "t": "tuned_top, log_top = fine_tune(train_layers=4)",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 오류 분석: 무엇을 무엇으로 틀렸나",
   "s": 6
  },
  {
   "t": "with torch.no_grad():",
   "s": 6
  },
  {
   "t": "    prob = tuned_top(encode(part[\"test\"][\"text\"].tolist())).softmax(-1).numpy()",
   "s": 6
  },
  {
   "t": "pred = prob.argmax(1)",
   "s": 6
  },
  {
   "t": "cm = confusion_matrix(y[\"test\"], pred, labels=[0, 1, 2])",
   "s": 6
  },
  {
   "t": "wrong = part[\"test\"].assign(pred=[LABELS[p] for p in pred], conf=prob.max(1))[pred != y[\"test\"]]",
   "s": 6
  },
  {
   "t": "by_template = part[\"test\"].assign(ok=pred == y[\"test\"]).groupby(\"template\")[\"ok\"].mean()  # 문장 틀마다 맞힌 비율",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 라벨이 적을 때",
   "s": 7
  },
  {
   "t": "def label_curve(sizes=(2, 4, 8, 16, 40), repeats=5):  # 종류마다 n개만 라벨이 있다면?",
   "s": 7
  },
  {
   "t": "    rng, rows = np.random.default_rng(0), []",
   "s": 7
  },
  {
   "t": "    for n in sizes:",
   "s": 7
  },
  {
   "t": "        for _ in range(repeats):",
   "s": 7
  },
  {
   "t": "            idx = np.concatenate([rng.choice(np.flatnonzero(y[\"train\"] == c), n, replace=False) for c in range(len(LABELS))])",
   "s": 7
  },
  {
   "t": "            texts, labels = part[\"train\"][\"text\"].iloc[idx], y[\"train\"][idx]",
   "s": 7
  },
  {
   "t": "            nearest = labels[(vec[\"test\"] @ vec[\"train\"][idx].T).argmax(1)]  # 가장 비슷한 라벨 문장을 따라가기",
   "s": 7
  },
  {
   "t": "            rows.append({\"n\": n, \"단어 세기\": macro_f1(y[\"test\"], bag_model(\"char_wb\", (2, 4)).fit(texts, labels).predict(part[\"test\"][\"text\"])),",
   "s": 7
  },
  {
   "t": "                         \"가장 비슷한 문장\": macro_f1(y[\"test\"], nearest),",
   "s": 7
  },
  {
   "t": "                         \"e5 + 로지스틱\": macro_f1(y[\"test\"], LogisticRegression(max_iter=2000, C=10).fit(vec[\"train\"][idx], labels).predict(vec[\"test\"]))})",
   "s": 7
  },
  {
   "t": "    return pd.DataFrame(rows).groupby(\"n\").mean()",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "curve = label_curve()",
   "s": 7
  },
  {
   "t": "desc = [\"기능이 고장 나서 오류가 나거나 작동하지 않는다는 버그 신고\", \"새로운 기능을 추가하거나 바꿔 달라는 요청\", \"사용 방법이나 정책을 묻는 질문\"]",
   "s": 7
  },
  {
   "t": "with torch.no_grad():",
   "s": 7
  },
  {
   "t": "    b = tok([\"passage: \" + d for d in desc], padding=True, return_tensors=\"pt\")",
   "s": 7
  },
  {
   "t": "    label_vec = F.normalize(mean_pool(body(**b).last_hidden_state, b[\"attention_mask\"]), dim=-1).numpy()",
   "s": 7
  },
  {
   "t": "zero_shot_f1 = macro_f1(y[\"test\"], (vec[\"test\"] @ label_vec.T).argmax(1))  # 라벨 0개: 설명문과 가장 가까운 종류",
   "s": 7
  },
  {
   "t": "",
   "s": 7
  },
  {
   "t": "# ⑧ 라벨 없는 문장 쓰기: 스스로 라벨 붙여 다시 배우기",
   "s": 8
  },
  {
   "t": "def self_train(seed, n=8, threshold=0.5):",
   "s": 8
  },
  {
   "t": "    rng = np.random.default_rng(seed)",
   "s": 8
  },
  {
   "t": "    few = np.concatenate([rng.choice(np.flatnonzero(y[\"train\"] == c), n, replace=False) for c in range(len(LABELS))])",
   "s": 8
  },
  {
   "t": "    teacher = LogisticRegression(max_iter=2000, C=10).fit(vec[\"train\"][few], y[\"train\"][few])",
   "s": 8
  },
  {
   "t": "    p = teacher.predict_proba(vec[\"unlabeled\"])",
   "s": 8
  },
  {
   "t": "    sure = p.max(1) >= threshold  # 확신이 기준 이상인 것만 가짜 라벨로",
   "s": 8
  },
  {
   "t": "    student = LogisticRegression(max_iter=2000, C=10).fit(np.vstack([vec[\"train\"][few], vec[\"unlabeled\"][sure]]),",
   "s": 8
  },
  {
   "t": "                                                         np.concatenate([y[\"train\"][few], p.argmax(1)[sure]]))",
   "s": 8
  },
  {
   "t": "    return {\"seed\": seed, \"가짜 라벨 수\": int(sure.sum()),",
   "s": 8
  },
  {
   "t": "            \"가짜 라벨 정답률\": np.mean(p.argmax(1)[sure] == y[\"unlabeled\"][sure]),  # 채점용: 실제 라벨과 비교",
   "s": 8
  },
  {
   "t": "            \"라벨만\": macro_f1(y[\"test\"], teacher.predict(vec[\"test\"])), \"가짜 라벨 더해서\": macro_f1(y[\"test\"], student.predict(vec[\"test\"]))}",
   "s": 8
  },
  {
   "t": "",
   "s": 8
  },
  {
   "t": "self_runs = pd.DataFrame([self_train(seed) for seed in range(1, 6)])",
   "s": 8
  }
 ],
 "names": {
  "1": "데이터 읽고 나누기",
  "2": "기준선: 단어를 세는 모델",
  "3": "e5로 문장 벡터 만들기 (특성 추출)",
  "4": "고정한 벡터 위에 분류기 얹기",
  "5": "미세조정: 모델 안의 가중치까지 고치기",
  "6": "오류 분석: 무엇을 무엇으로 틀렸나",
  "7": "라벨이 적을 때",
  "8": "라벨 없는 문장 쓰기: 스스로 라벨 붙여 다시 배우기"
 },
 "range": [
  1,
  8
 ],
 "outs": {
  "data": {
   "title": "issues.csv에서 몇 줄 (위: train, 아래: test)",
   "cols": [
    "split",
    "text",
    "label"
   ],
   "rows": [
    [
     "train",
     "결제가 안 돼요. 버튼을 눌러도 아무 반응이 없어요.",
     "<b>버그</b>"
    ],
    [
     "train",
     "결제에도 단축키가 있으면 편할 것 같아요.",
     "<b>기능 요청</b>"
    ],
    [
     "train",
     "결제는 하루에 몇 번까지 되나요?",
     "<b>질문</b>"
    ],
    [
     "test",
     "비밀번호 변경 누르면 이상한 페이지로 넘어가요. 원래 이러면 안 되잖아요?",
     "<b>버그</b>"
    ],
    [
     "test",
     "회원 탈퇴 쪽에 이런 기능 있으면 대박일 듯. 기록 보기 같은 거요",
     "<b>기능 요청</b>"
    ],
    [
     "test",
     "알림 하기 전에 미리 알아 둘 게 있을까요?",
     "<b>질문</b>"
    ]
   ],
   "notes": [
    "위 세 줄은 모두 \"결제\" 이야기인데 종류가 달라요. 어떤 단어가 나왔는지가 아니라 <b>무엇을 원하는 말인지</b>를 봐야 해요.",
    "아래 세 줄은 test에서 가져왔어요. test와 valid에는 train에 없던 말투(문장 틀)만 넣었어요. 실제 문의도 사람마다 쓰는 말투가 다르니까요."
   ]
  },
  "why": {
   "title": "같은 train 190개, 같은 test 120개로 비교 (macro F1)",
   "cols": [
    "방법",
    "test macro F1",
    "단계"
   ],
   "rows": [
    [
     "단어를 세는 모델 (TF-IDF + Naive Bayes)",
     "<span class=\"bad\">0.714</span>",
     "②"
    ],
    [
     "e5를 고정하고 벡터 위에 로지스틱 회귀",
     "0.891",
     "③④"
    ],
    [
     "e5 12층 전체를 미세조정",
     "<span class=\"bad\">0.737</span>",
     "⑤"
    ],
    [
     "e5 위쪽 4층만 미세조정",
     "<b>0.915</b>",
     "⑤"
    ]
   ],
   "notes": [
    "단어를 세는 방법은 test에서 처음 보는 단어가 많아서 무너져요. 미리 학습된 e5의 벡터를 쓰면 처음 보는 말투도 뜻이 비슷하면 가깝게 놓여서 0.891까지 올라가요.",
    "그런데 e5를 통째로 더 학습시키면(미세조정) 오히려 떨어지고, 위쪽 몇 층만 고치면 가장 좋아요. 이 쪽은 이 차이가 왜 생기는지, 라벨이 더 적으면 어떻게 하는지를 다뤄요."
   ]
  },
  "s1": {
   "title": "① 데이터",
   "lines": [
    "train → 190개 (버그 90, 기능 요청 60, 질문 40)",
    "valid → 60개 (버그 20, 기능 요청 20, 질문 20)",
    "test → 120개 (버그 40, 기능 요청 40, 질문 40)",
    "unlabeled → 410개 (라벨은 있지만 쓰지 않아요)",
    [
     "train 문장 틀 30개, test 문장 틀 12개, 겹치는 틀 0개",
     1,
     "newf"
    ],
    [
     "y['train'][:8] → [0, 1, 0, 2, 1, 0, 0, 0]",
     2
    ]
   ],
   "notes": [
    "test에만 있는 문장 틀로 채점해요. 외운 말투가 아니라 처음 보는 말투를 맞히는지 재는 거예요.",
    "종류 이름을 번호로 바꿨어요 (0 버그, 1 기능 요청, 2 질문). train은 버그 90개, 질문 40개로 고르지 않아요. 그래서 정확도 대신, 종류마다 F1을 구해 똑같은 무게로 평균 낸 <b>macro F1</b>로 재요."
   ]
  },
  "s2": {
   "title": "② 단어 세기 기준선",
   "lines": [
    [
     "test macro F1 → 0.714",
     1,
     "bad"
    ],
    [
     "test의 단어 972개 중 train에 없던 단어 → 59%",
     2,
     "newf"
    ],
    "train에 없던 단어 (많이 나온 순) → ['기능', '원래', '쪽에', '이런', '대박일', '듯.', '기록', '보기']"
   ],
   "notes": [
    "단어 하나하나에 점수를 매겨 세는 방법이라, 처음 보는 단어는 아무 정보가 없어요.",
    "test 단어의 59%가 train에 한 번도 없었어요. \"어케\", \"먹통이에요\"처럼 띄어쓰기 단위 단어는 어미, 조사가 붙어 모양이 계속 바뀌어서 더 그래요."
   ]
  },
  "t2": {
   "title": "단어 대신 글자 조각(2~4글자)으로 세면",
   "cols": [
    "세는 단위",
    "test macro F1"
   ],
   "rows": [
    [
     "띄어쓰기 단어",
     "0.714"
    ],
    [
     "글자 조각 2~4개",
     "<b>0.810</b>"
    ]
   ],
   "notes": [
    "\"먹통이에요\"를 \"먹통\", \"통이\", \"이에요\" 같은 조각으로 쪼개 세면, 어미가 달라도 같은 조각이 잡혀서 0.71에서 0.81까지 올라요.",
    "모델이 쓰는 Tokenizer가 부분단어(조각)로 나누는 것도 같은 이유예요. 그래도 \"뜻이 비슷한 다른 단어\"는 여전히 몰라요."
   ]
  },
  "s3": {
   "title": "③ 문장 벡터",
   "lines": [
    [
     "vec['train'].shape → (190, 384)",
     1
    ],
    "길이 → [1.0, 1.0, 1.0]",
    [
     "test 120문장을 e5 Tokenizer로 나누면 모르는 토큰(<unk>) → 0개",
     2,
     "newf"
    ],
    "예: \"회원 탈퇴 쪽에 이런 기능 있으면 대박일 듯. 기록 보기 같은 거요\" → ['<s>', '▁que', 'ry', ':', '▁회원', '▁탈', '퇴', '▁', '쪽에', '▁이런', '▁기능', '▁있으면', '▁대', '박', '일', '▁듯', '.', '▁기록', '▁보기', '▁같은', '▁거', '요', '</s>']"
   ],
   "notes": [
    "문장마다 384칸 벡터 하나예요. 길이를 1로 맞춰서, 두 벡터의 내적이 곧 코사인 유사도예요.",
    "부분단어 Tokenizer는 처음 보는 단어도 아는 조각으로 쪼개서 \"모르는 토큰\"이 생기지 않아요. 그리고 각 조각의 벡터는 수많은 글로 미리 학습돼 있어서, 처음 보는 말투도 뜻이 비슷하면 비슷한 벡터가 돼요."
   ]
  },
  "f_pca": {
   "title": "train(채운 점)과 test(빈 원)의 e5 문장 벡터",
   "images": [
    [
     "img/textcls/pca_vectors.png",
     "PCA로 384칸 → 2칸"
    ]
   ],
   "notes": [
    "라벨을 하나도 쓰지 않고 만든 벡터인데도 종류별로 대체로 모여 있어요. 처음 보는 말투인 test(빈 원)도 같은 종류 근처에 떨어져요. 그래서 이 벡터 위에 단순한 분류기만 얹어도 잘 돼요.",
    "PCA 두 칸은 384칸의 일부만 보여 줘서, 겹쳐 보이는 점도 384칸에서는 갈릴 수 있어요 (<a href=\"ml-svm-lda-pca.html\">PCA</a>)."
   ],
   "chart": true
  },
  "s4": {
   "title": "④ e5 고정 + 로지스틱 회귀",
   "lines": [
    [
     "test macro F1 → 0.891",
     1,
     "newf"
    ],
    "버그: precision 1.00, recall 1.00",
    "기능 요청: precision 0.91, recall 0.75",
    "질문: precision 0.79, recall 0.93"
   ],
   "notes": [
    "②(0.71)보다 크게 올랐어요. 학습한 건 384 × 3 가중치와 절편 3개뿐이에요. 문장 벡터는 e5가 이미 만들어 둔 걸 그대로 썼어요."
   ]
  },
  "s5": {
   "title": "⑤ 미세조정 두 가지 (epoch마다 마지막 묶음의 train 손실, test macro F1)",
   "cols": [
    "epoch",
    "12층 전부: train 손실",
    "12층 전부: test F1",
    "위쪽 4층: train 손실",
    "위쪽 4층: test F1"
   ],
   "rows": [
    [
     1,
     "0.587",
     "0.736",
     "0.777",
     "<b>0.374</b>"
    ],
    [
     2,
     "0.111",
     "0.737",
     "0.372",
     "<b>0.906</b>"
    ],
    [
     3,
     "0.039",
     "0.737",
     "0.125",
     "<b>0.915</b>"
    ],
    [
     4,
     "0.028",
     "0.737",
     "0.147",
     "<b>0.915</b>"
    ],
    [
     5,
     "0.014",
     "0.737",
     "0.026",
     "<b>0.933</b>"
    ],
    [
     6,
     "0.005",
     "0.737",
     "0.008",
     "<b>0.915</b>"
    ]
   ],
   "notes": [
    "고치는 가중치: 12층 전부 21,294,723개, 위쪽 4층 7,099,011개 (e5 몸통 전체는 117,653,760개. 대부분은 25만 어휘의 Embedding 표라서 두 경우 모두 고정했어요).",
    "12층을 다 고치면 첫 epoch부터 test F1이 0.74 근처에 멈춰요. train 손실은 0.005까지 내려가는데요. train 190개는 종류마다 문장 틀 10개뿐이라, 모델 전체가 그 말투를 외워 버렸어요(과적합).",
    "아래쪽 8층은 그대로 두고 위쪽 4층과 분류 층만 고치면, 미리 배운 일반적인 언어 지식은 지키면서 이 과제에 맞게 다듬어져요. 마지막 epoch에서 0.915, 가장 높을 때 0.933이에요. 데이터가 적을수록 적게 고치는 게 안전해요."
   ]
  },
  "f_ft": {
   "title": "미세조정 두 가지의 학습 기록",
   "images": [
    [
     "img/textcls/finetune_curves.png",
     ""
    ]
   ],
   "notes": [
    "왼쪽만 보면 12층 전부가 더 잘 배우는 것 같지만, 오른쪽의 처음 보는 말투에서는 회색 점선(고정)보다도 낮아요. 학습이 잘 되는지는 train이 아니라 처음 보는 데이터로 봐요."
   ],
   "chart": true
  },
  "s6": {
   "title": "⑥ 위쪽 4층 미세조정의 test 결과",
   "lines": [
    "혼동 행렬 (행: 실제, 열: 예측) 순서: 버그, 기능 요청, 질문",
    "  버그     [40, 0, 0]",
    "  기능 요청  [0, 30, 10]",
    "  질문     [0, 0, 40]",
    [
     "틀린 10개의 문장 틀 → ['기능14'], 그 틀의 정답률 0%",
     1,
     "bad"
    ],
    [
     "예: \"혹시 비밀번호 변경에 예약 기능을 추가할 계획은 없나요?\" → 정답 기능 요청, 예측 질문 (확신 0.94)",
     2
    ]
   ],
   "notes": [
    "틀린 10개가 모두 한 문장 틀에서 나왔고, 나머지 test 틀은 다 맞혔어요. 숫자 하나(F1)만 보면 안 보이는 사실이에요.",
    "\"~할 계획은 없나요?\"처럼 <b>질문 꼴로 쓴 기능 요청</b>이에요. train의 질문은 100%가 물음표로 끝나고, 물음표로 끝나는 기능 요청은 문장 틀 1개뿐이에요. 그래서 모델이 \"물음표 꼴이면 질문\"이라는 지름길을 배웠어요. 확신도 높아서, 확률만 보고는 걸러지지 않아요. 이런 문장을 train에 몇 개 넣거나, 종류의 정의를 다시 정하는 게 해결책이에요."
   ]
  },
  "s7": {
   "title": "⑦ 종류마다 라벨이 n개뿐이라면 (무작위로 5번 뽑은 평균, test macro F1)",
   "cols": [
    "n (종류마다)",
    "단어 세기 (글자 조각)",
    "가장 비슷한 문장 따라가기",
    "e5 + 로지스틱"
   ],
   "rows": [
    [
     "<b>0</b> (설명문만)",
     "-",
     "-",
     "0.689 (설명문과 가장 가까운 종류)"
    ],
    [
     2,
     "0.653",
     "0.544",
     "<b>0.657</b>"
    ],
    [
     4,
     "0.636",
     "0.557",
     "<b>0.760</b>"
    ],
    [
     8,
     "0.633",
     "0.627",
     "<b>0.863</b>"
    ],
    [
     16,
     "0.714",
     "0.597",
     "<b>0.897</b>"
    ],
    [
     40,
     "0.794",
     "0.790",
     "<b>0.894</b>"
    ]
   ],
   "notes": [
    "라벨이 종류마다 8개뿐이어도 e5 + 로지스틱은 0.86이에요. 단어 세기(0.63)가 40개로도 못 가는 점수예요.",
    "\"가장 비슷한 라벨 문장을 따라가기\"(벡터를 찾아보기 표로 쓰기)는 라벨이 적으면 흔들려요. 한두 문장의 말투에 크게 좌우되기 때문이에요.",
    "라벨이 하나도 없어도, 종류마다 설명문 한 줄을 써서 그 벡터와 가장 가까운 종류로 답하면 0.69예요. 세 종류를 무작위로 찍으면 0.33이에요."
   ]
  },
  "f_curve": {
   "title": "라벨 수에 따른 점수",
   "images": [
    [
     "img/textcls/label_curve.png",
     ""
    ]
   ],
   "notes": [
    "보라 선이 거의 모든 n에서 위에 있어요. 라벨을 모으기 어려울 때는 먼저 미리 학습된 벡터 위에 단순한 분류기를 얹어 보는 게 출발점이에요."
   ],
   "chart": true
  },
  "s8": {
   "title": "⑧ 종류마다 라벨 8개 + 라벨 없는 문장 410개 (확률 0.5 이상만 가짜 라벨로)",
   "cols": [
    "뽑기",
    "가짜 라벨 수",
    "가짜 라벨 정답률",
    "라벨만",
    "가짜 라벨 더해서"
   ],
   "rows": [
    [
     1,
     317,
     "96%",
     "0.883",
     "0.883"
    ],
    [
     2,
     284,
     "98%",
     "0.801",
     "0.875"
    ],
    [
     3,
     299,
     "96%",
     "0.865",
     "0.900"
    ],
    [
     4,
     305,
     "100%",
     "0.924",
     "0.899"
    ],
    [
     5,
     301,
     "100%",
     "0.891",
     "0.891"
    ],
    [
     "<b>평균 (표준편차)</b>",
     "",
     "",
     "0.873 (0.041)",
     "<b>0.889 (0.010)</b>"
    ]
   ],
   "notes": [
    "라벨 24개로 배운 분류기(선생)가 라벨 없는 문장에 라벨을 붙이고, 그걸 더해 다시 배워요(학생). 가짜 라벨은 모든 뽑기에서 95% 넘게 맞았어요.",
    "평균은 0.873에서 0.889까지 조금 오르고, 뽑기마다의 흔들림(표준편차)은 0.041에서 0.010까지 크게 줄었어요. 운 나쁘게 라벨이 치우친 뽑기를 라벨 없는 문장이 메워 줬어요. 대신 이미 잘된 뽑기는 조금 내려가기도 했어요.",
    "다만 이 라벨 없는 문장들은 train과 같은 말투뿐이라, ⑥에서 본 \"질문 꼴 기능 요청\" 같은 처음 보는 말투의 약점은 그대로예요. 새 말투를 배우려면 라벨 없는 문장에도 그런 말투가 들어 있어야 해요."
   ]
  }
 },
 "flow": [
  [
   1,
   "데이터",
   "190 / 60 / 120",
   "test는 새 말투"
  ],
  [
   2,
   "단어 세기",
   "TF-IDF, Naive Bayes",
   "F1 0.71"
  ],
  [
   3,
   "문장 벡터",
   "e5 평균 풀링",
   "모르는 토큰 0"
  ],
  [
   4,
   "고정 + 분류기",
   "로지스틱",
   "F1 0.89"
  ],
  [
   5,
   "미세조정",
   "12층 vs 위쪽 4층",
   "F1 0.74 vs 0.92"
  ],
  [
   6,
   "오류 분석",
   "혼동 행렬",
   "질문 꼴 요청"
  ],
  [
   7,
   "라벨이 적으면",
   "n개씩, 설명문",
   "벡터가 이김"
  ],
  [
   8,
   "라벨 없는 문장",
   "가짜 라벨",
   "흔들림 감소"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, transformers 5.22.2, scikit-learn 1.8.0, intfloat/multilingual-e5-small, CPU"
};
