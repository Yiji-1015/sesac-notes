# 서비스 문의를 버그, 기능 요청, 질문으로 나누기 (문장 분류) 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: data/issues.csv (가상의 문의 780개. train 190, valid 60, test 120, 라벨 없는 문장 410)
#         valid와 test에는 train에 없던 문장 틀만 들어 있어요. 처음 보는 말투에도 맞히는지 재요.
# 모델: intfloat/multilingual-e5-small (이 PC에 받아 둔 것). 모두 CPU로 돌려요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

from pathlib import Path                                         #@1
import numpy as np                                               #@1
import pandas as pd                                              #@1
from sklearn.metrics import f1_score                             #@1
from sklearn.feature_extraction.text import TfidfVectorizer      #@2
from sklearn.naive_bayes import MultinomialNB                    #@2
from sklearn.pipeline import make_pipeline                       #@2
import torch                                                     #@3
import torch.nn.functional as F                                  #@3
from transformers import AutoTokenizer, AutoModel                #@3
from sklearn.linear_model import LogisticRegression              #@4
from sklearn.metrics import confusion_matrix                     #@6

DATA = Path(__file__).resolve().parent / "data"                  #@1
MODEL = "intfloat/multilingual-e5-small"                         #@3

# ① 데이터 읽고 나누기
df = pd.read_csv(DATA / "issues.csv")
LABELS = ["버그", "기능 요청", "질문"]
part = {s: df[df["split"] == s] for s in ("train", "valid", "test", "unlabeled")}
y = {s: d["label"].map(LABELS.index).to_numpy() for s, d in part.items()}  # 종류 이름 → 번호 0, 1, 2

def macro_f1(y_true, pred):                                      # 종류마다 F1을 구해 똑같은 무게로 평균
    return f1_score(y_true, pred, average="macro")

# ② 기준선: 단어를 세는 모델
def bag_model(analyzer="word", ngram=(1, 1)):                    # 글 → 단어마다 TF-IDF 점수 → Naive Bayes
    return make_pipeline(TfidfVectorizer(analyzer=analyzer, ngram_range=ngram), MultinomialNB())

nb = bag_model().fit(part["train"]["text"], y["train"])
nb_f1 = macro_f1(y["test"], nb.predict(part["test"]["text"]))
train_words = {w for t in part["train"]["text"] for w in t.split()}
test_words = [w for t in part["test"]["text"] for w in t.split()]
unseen = np.mean([w not in train_words for w in test_words])     # test의 단어 중 train에 한 번도 없던 비율

# ③ e5로 문장 벡터 만들기 (특성 추출)
tok = AutoTokenizer.from_pretrained(MODEL)
body = AutoModel.from_pretrained(MODEL).eval()

def encode(texts):                                               # e5는 앞에 "query: "를 붙여 쓰도록 학습됐어요
    return tok(["query: " + t for t in texts], padding=True, return_tensors="pt")

def mean_pool(hidden, mask):                                     # 채운 자리를 빼고 토큰 벡터들의 평균
    m = mask[..., None].float()
    return (hidden * m).sum(1) / m.sum(1)

def embed(texts, model=body):
    with torch.no_grad():
        b = encode(texts)
        return F.normalize(mean_pool(model(**b).last_hidden_state, b["attention_mask"]), dim=-1).numpy()

vec = {s: embed(d["text"].tolist()) for s, d in part.items()}    # 문장마다 384칸, 길이 1

# ④ 고정한 벡터 위에 분류기 얹기
frozen = LogisticRegression(max_iter=2000, C=10).fit(vec["train"], y["train"])
frozen_f1 = macro_f1(y["test"], frozen.predict(vec["test"]))

# ⑤ 미세조정: 모델 안의 가중치까지 고치기
class Classifier(torch.nn.Module):                               # e5 몸통 + 384 → 3 분류 층
    def __init__(self):
        super().__init__()
        self.body = AutoModel.from_pretrained(MODEL)
        self.head = torch.nn.Linear(384, len(LABELS))

    def forward(self, b):
        return self.head(mean_pool(self.body(**b).last_hidden_state, b["attention_mask"]))

def fine_tune(train_layers, epochs=6, lr=5e-5, head_lr=1e-3):   # train_layers: 위에서부터 몇 층을 고칠까 (12면 전부)
    torch.manual_seed(0)
    clf = Classifier()
    clf.body.requires_grad_(False)                               # 일단 몸통 전체를 고정하고
    for layer in clf.body.encoder.layer[12 - train_layers:]:
        layer.requires_grad_(True)                               # 위쪽 train_layers개 층만 풀기
    opt = torch.optim.AdamW([{"params": [p for p in clf.body.parameters() if p.requires_grad], "lr": lr},
                             {"params": clf.head.parameters(), "lr": head_lr}])
    X, Y, log = part["train"]["text"].tolist(), torch.tensor(y["train"]), []
    for epoch in range(epochs):
        clf.train()
        for idx in torch.randperm(len(X)).split(16):             # 16개씩 섞어서
            loss = F.cross_entropy(clf(encode([X[i] for i in idx])), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            test_pred = clf(encode(part["test"]["text"].tolist())).argmax(1).numpy()
        log.append({"epoch": epoch + 1, "train_loss": loss.item(), "test_f1": macro_f1(y["test"], test_pred)})
    return clf, log

tuned_all, log_all = fine_tune(train_layers=12)
tuned_top, log_top = fine_tune(train_layers=4)

# ⑥ 오류 분석: 무엇을 무엇으로 틀렸나
with torch.no_grad():
    prob = tuned_top(encode(part["test"]["text"].tolist())).softmax(-1).numpy()
pred = prob.argmax(1)
cm = confusion_matrix(y["test"], pred, labels=[0, 1, 2])
wrong = part["test"].assign(pred=[LABELS[p] for p in pred], conf=prob.max(1))[pred != y["test"]]
by_template = part["test"].assign(ok=pred == y["test"]).groupby("template")["ok"].mean()  # 문장 틀마다 맞힌 비율

# ⑦ 라벨이 적을 때
def label_curve(sizes=(2, 4, 8, 16, 40), repeats=5):             # 종류마다 n개만 라벨이 있다면?
    rng, rows = np.random.default_rng(0), []
    for n in sizes:
        for _ in range(repeats):
            idx = np.concatenate([rng.choice(np.flatnonzero(y["train"] == c), n, replace=False) for c in range(len(LABELS))])
            texts, labels = part["train"]["text"].iloc[idx], y["train"][idx]
            nearest = labels[(vec["test"] @ vec["train"][idx].T).argmax(1)]          # 가장 비슷한 라벨 문장을 따라가기
            rows.append({"n": n, "단어 세기": macro_f1(y["test"], bag_model("char_wb", (2, 4)).fit(texts, labels).predict(part["test"]["text"])),
                         "가장 비슷한 문장": macro_f1(y["test"], nearest),
                         "e5 + 로지스틱": macro_f1(y["test"], LogisticRegression(max_iter=2000, C=10).fit(vec["train"][idx], labels).predict(vec["test"]))})
    return pd.DataFrame(rows).groupby("n").mean()

curve = label_curve()
desc = ["기능이 고장 나서 오류가 나거나 작동하지 않는다는 버그 신고", "새로운 기능을 추가하거나 바꿔 달라는 요청", "사용 방법이나 정책을 묻는 질문"]
with torch.no_grad():
    b = tok(["passage: " + d for d in desc], padding=True, return_tensors="pt")
    label_vec = F.normalize(mean_pool(body(**b).last_hidden_state, b["attention_mask"]), dim=-1).numpy()
zero_shot_f1 = macro_f1(y["test"], (vec["test"] @ label_vec.T).argmax(1))   # 라벨 0개: 설명문과 가장 가까운 종류

# ⑧ 라벨 없는 문장 쓰기: 스스로 라벨 붙여 다시 배우기
def self_train(seed, n=8, threshold=0.5):
    rng = np.random.default_rng(seed)
    few = np.concatenate([rng.choice(np.flatnonzero(y["train"] == c), n, replace=False) for c in range(len(LABELS))])
    teacher = LogisticRegression(max_iter=2000, C=10).fit(vec["train"][few], y["train"][few])
    p = teacher.predict_proba(vec["unlabeled"])
    sure = p.max(1) >= threshold                                 # 확신이 기준 이상인 것만 가짜 라벨로
    student = LogisticRegression(max_iter=2000, C=10).fit(np.vstack([vec["train"][few], vec["unlabeled"][sure]]),
                                                         np.concatenate([y["train"][few], p.argmax(1)[sure]]))
    return {"seed": seed, "가짜 라벨 수": int(sure.sum()),
            "가짜 라벨 정답률": np.mean(p.argmax(1)[sure] == y["unlabeled"][sure]),   # 채점용: 실제 라벨과 비교
            "라벨만": macro_f1(y["test"], teacher.predict(vec["test"])), "가짜 라벨 더해서": macro_f1(y["test"], student.predict(vec["test"]))}

self_runs = pd.DataFrame([self_train(seed) for seed in range(1, 6)])
