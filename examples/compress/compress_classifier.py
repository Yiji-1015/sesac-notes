# 문의 분류기를 가볍게 만들기 (어휘 줄이기, 지식 정제, 양자화, 가지치기) 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 데이터: ../classify/data/issues.csv (문장 분류 쪽과 같은 가상의 문의 780개)
# 모델: intfloat/multilingual-e5-small (이 PC에 받아 둔 것). 모두 CPU로, 스레드 4개로 재요.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import copy                                                      #@3
import io                                                        #@1
import time                                                      #@1
from pathlib import Path                                         #@1
import pandas as pd                                              #@1
import torch                                                     #@1
import torch.nn.functional as F                                  #@1
import torch.nn.utils.prune as prune                             #@6
from sklearn.metrics import f1_score                             #@1
from transformers import AutoTokenizer, AutoModel                #@1

DATA = Path(__file__).resolve().parent.parent / "classify" / "data"   #@1
MODEL = "intfloat/multilingual-e5-small"                         #@1
torch.set_num_threads(4)                                         #@1

# ① 기준 모델과 재는 방법
df = pd.read_csv(DATA / "issues.csv")
LABELS = ["버그", "기능 요청", "질문"]
part = {s: df[df["split"] == s] for s in ("train", "test", "unlabeled")}
y = {s: torch.tensor(d["label"].map(LABELS.index).to_numpy()) for s, d in part.items()}
tok = AutoTokenizer.from_pretrained(MODEL)

def encode(texts):
    return tok(["query: " + t for t in texts], padding=True, return_tensors="pt")

class Classifier(torch.nn.Module):                               # 문장 분류 쪽 ⑤와 같은 모양: e5 몸통 + 384 → 3
    def __init__(self, body):
        super().__init__()
        self.body = body
        torch.manual_seed(0)                                     # 분류 층의 처음 값을 늘 같게
        self.head = torch.nn.Linear(384, len(LABELS))

    def forward(self, b):
        h = self.body(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        m = b["attention_mask"][..., None].float()
        return self.head((h * m).sum(1) / m.sum(1))

def train(model, texts, labels, train_layers, epochs=6, soft=None, T=2.0, alpha=0.5):   # soft: 선생이 낸 점수 (⑤의 정제)
    torch.manual_seed(0)
    model.body.requires_grad_(False)
    for layer in model.body.encoder.layer[len(model.body.encoder.layer) - train_layers:]:
        layer.requires_grad_(True)                               # 위쪽 train_layers개 층만 고치기
    opt = torch.optim.AdamW([{"params": [p for p in model.body.parameters() if p.requires_grad], "lr": 5e-5},
                             {"params": model.head.parameters(), "lr": 1e-3}])
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(texts)).split(16):
            logits = model(encode([texts[i] for i in idx]))
            known = labels[idx] >= 0                             # 라벨이 -1이면 정답 없음 (라벨 없는 문장)
            loss = F.cross_entropy(logits[known], labels[idx][known]) if known.any() else logits.sum() * 0
            if soft is not None:
                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(soft[idx] / T, -1), reduction="batchmean") * T * T
                loss = alpha * loss + (1 - alpha) * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()

def benchmark(model):                                            # 크기, 문장 하나의 시간, 정확도를 같은 잣대로
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)                          # 저장했을 때의 파일 크기
    texts = part["test"]["text"].tolist()
    with torch.no_grad():
        pred = torch.cat([model(encode(texts[i:i + 32])).argmax(-1) for i in range(0, len(texts), 32)])
        for t in texts[:20]:                                     # 한 번 데우기 (처음 몇 번은 늘 느려요)
            model(encode([t]))
        runs = {"this": [], "base": []}
        for _ in range(5):                                       # 이 모델과 기준을 번갈아 다섯 번씩 (PC가 바빠지면 둘 다 똑같이 느려지게)
            for key, m in (("this", model), ("base", teacher)):
                t0 = time.perf_counter()
                for t in texts:                                  # 문장 120개를 하나씩
                    m(encode([t]))
                runs[key].append((time.perf_counter() - t0) / len(texts) * 1000)
    ms = min(runs["this"])                                       # 다섯 번 중 가장 빠른 값
    return {"MB": buf.getbuffer().nbytes / 2 ** 20, "ms": ms, "x": ms / min(runs["base"]), "F1": f1_score(y["test"], pred, average="macro")}

teacher = train(Classifier(AutoModel.from_pretrained(MODEL)), part["train"]["text"].tolist(), y["train"], train_layers=4)
results = {"기준 (12층)": benchmark(teacher)}

# ② 무게가 어디에 있나
count = lambda module: sum(p.numel() for p in module.parameters())   # noqa: E731
weights = {"토큰 Embedding 표": count(teacher.body.embeddings.word_embeddings),
           "위치와 그 밖의 Embedding": count(teacher.body.embeddings) - count(teacher.body.embeddings.word_embeddings),
           "Transformer 층 12개": count(teacher.body.encoder), "분류 층": count(teacher.head)}

# ③ 어휘 줄이기: 이 일에 나오는 토큰만 남기기
used = sorted({i for row in tok(["query: " + t for t in df["text"]])["input_ids"] for i in row}
              | {tok.pad_token_id, tok.unk_token_id, tok.cls_token_id, tok.sep_token_id})
remap = torch.full((len(tok),), used.index(tok.unk_token_id))   # 원래 번호 → 새 번호 (모르는 토큰은 <unk>로)
remap[torch.tensor(used)] = torch.arange(len(used))

class SmallVocab(torch.nn.Module):                               # Embedding 표에서 쓰는 줄만 남기고, 들어오는 번호를 바꿔 넣기
    def __init__(self, clf):
        super().__init__()
        self.clf = copy.deepcopy(clf)
        old = self.clf.body.embeddings.word_embeddings.weight.data
        self.clf.body.embeddings.word_embeddings = torch.nn.Embedding(len(used), 384, _weight=old[torch.tensor(used)].clone())

    def forward(self, b):
        return self.clf({"input_ids": remap[b["input_ids"]], "attention_mask": b["attention_mask"]})

small_teacher = SmallVocab(teacher).eval()
results["어휘 줄이기"] = benchmark(small_teacher)

# ④ 지식 정제: 큰 모델의 답을 작은 모델에게 가르치기
def first_layers(clf, n):                                        # 선생의 아래쪽 n개 층만 가진 몸통 (위쪽 층을 떼어 냄)
    body = copy.deepcopy(clf.body)
    body.encoder.layer = body.encoder.layer[:n]
    body.config.num_hidden_layers = n
    return body

X, Y = part["train"]["text"].tolist(), y["train"]
X_un = part["unlabeled"]["text"].tolist()
with torch.no_grad():
    soft_train = teacher(encode(X))                              # 선생이 train 문장에 준 점수
    soft_un = teacher(encode(X_un))                              # 선생이 라벨 없는 문장에 준 점수
def student(seed):                                               # 선생의 아래 6층 + 새 분류 층 (분류 층의 처음 값만 seed로 다르게)
    s = Classifier(first_layers(teacher, 6))
    torch.manual_seed(seed)
    s.head.reset_parameters()
    return s

def test_f1(model):
    with torch.no_grad():
        return f1_score(y["test"], model(encode(part["test"]["text"].tolist())).argmax(-1), average="macro")

ways = {"정답만": lambda s: train(s, X, Y, train_layers=2),     # 위쪽 2층과 분류 층만 학습
        "선생의 점수로": lambda s: train(s, X, Y, train_layers=2, soft=soft_train),
        "+ 라벨 없는 문장": lambda s: train(s, X + X_un, torch.cat([Y, torch.full((len(X_un),), -1)]), train_layers=2, soft=torch.cat([soft_train, soft_un]))}
student_f1 = {name: [test_f1(fit(student(seed))) for seed in range(4)] for name, fit in ways.items()}   # 처음 값 4가지로
best_student = ways["+ 라벨 없는 문장"](student(0))
results["6층 학생"] = benchmark(best_student)

# ⑤ 양자화: 32비트 실수를 8비트 정수로
def int8(model, per_channel):                                    # 몸통의 Linear 층 가중치를 int8로 (분류 층은 그대로)
    m = copy.deepcopy(model)
    qconfig = torch.ao.quantization.per_channel_dynamic_qconfig if per_channel else torch.ao.quantization.default_dynamic_qconfig
    body = m.clf.body if isinstance(m, SmallVocab) else m.body
    torch.ao.quantization.quantize_dynamic(body, {torch.nn.Linear: qconfig}, dtype=torch.qint8, inplace=True)
    return m

for per_channel in (False, True):
    results[f"어휘 줄이기 + int8 ({'출력 칸마다' if per_channel else '층마다'} 눈금)"] = benchmark(int8(small_teacher, per_channel))
w = teacher.body.encoder.layer[0].attention.self.query.weight.data   # 양자화 오차를 직접 보기: 첫 층 Query 행렬
scale = w.abs().max() / 127                                      # 층마다 눈금 하나: 가장 큰 값을 127에 맞추기
w_back = (w / scale).round().clamp(-128, 127) * scale            # int8로 바꿨다가 되돌린 값

# ⑥ 가지치기: 크기가 작은 가중치를 0으로
def pruned(model, amount):                                       # 몸통의 모든 Linear 가중치 중 절댓값이 작은 amount만큼을 0으로
    m = copy.deepcopy(model)
    targets = [(mod, "weight") for mod in m.body.encoder.modules() if isinstance(mod, torch.nn.Linear)]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in targets:
        prune.remove(mod, name)                                  # 0으로 만든 걸 가중치에 확정
    return m

for amount in (0.3, 0.5, 0.7):
    results[f"가지치기 {amount:.0%}"] = benchmark(pruned(teacher, amount))

# ⑦ 합쳐 보기
final = int8(SmallVocab(best_student).eval(), per_channel=True)
results["6층 학생 + 어휘 줄이기 + int8"] = benchmark(final)
