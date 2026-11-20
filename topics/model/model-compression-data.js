window.PAGE = {
 "code": [
  {
   "t": "import copy",
   "s": 3
  },
  {
   "t": "import io",
   "s": 1
  },
  {
   "t": "import time",
   "s": 1
  },
  {
   "t": "from pathlib import Path",
   "s": 1
  },
  {
   "t": "import pandas as pd",
   "s": 1
  },
  {
   "t": "import torch",
   "s": 1
  },
  {
   "t": "import torch.nn.functional as F",
   "s": 1
  },
  {
   "t": "import torch.nn.utils.prune as prune",
   "s": 6
  },
  {
   "t": "from sklearn.metrics import f1_score",
   "s": 1
  },
  {
   "t": "from transformers import AutoTokenizer, AutoModel",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "DATA = Path(__file__).resolve().parent.parent / \"classify\" / \"data\"",
   "s": 1
  },
  {
   "t": "MODEL = \"intfloat/multilingual-e5-small\"",
   "s": 1
  },
  {
   "t": "torch.set_num_threads(4)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ① 기준 모델과 재는 방법",
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
   "t": "part = {s: df[df[\"split\"] == s] for s in (\"train\", \"test\", \"unlabeled\")}",
   "s": 1
  },
  {
   "t": "y = {s: torch.tensor(d[\"label\"].map(LABELS.index).to_numpy()) for s, d in part.items()}",
   "s": 1
  },
  {
   "t": "tok = AutoTokenizer.from_pretrained(MODEL)",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def encode(texts):",
   "s": 1
  },
  {
   "t": "    return tok([\"query: \" + t for t in texts], padding=True, return_tensors=\"pt\")",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "class Classifier(torch.nn.Module):  # 문장 분류 쪽 ⑤와 같은 모양: e5 몸통 + 384 → 3",
   "s": 1
  },
  {
   "t": "    def __init__(self, body):",
   "s": 1
  },
  {
   "t": "        super().__init__()",
   "s": 1
  },
  {
   "t": "        self.body = body",
   "s": 1
  },
  {
   "t": "        torch.manual_seed(0)  # 분류 층의 처음 값을 늘 같게",
   "s": 1
  },
  {
   "t": "        self.head = torch.nn.Linear(384, len(LABELS))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "    def forward(self, b):",
   "s": 1
  },
  {
   "t": "        h = self.body(input_ids=b[\"input_ids\"], attention_mask=b[\"attention_mask\"]).last_hidden_state",
   "s": 1
  },
  {
   "t": "        m = b[\"attention_mask\"][..., None].float()",
   "s": 1
  },
  {
   "t": "        return self.head((h * m).sum(1) / m.sum(1))",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def train(model, texts, labels, train_layers, epochs=6, soft=None, T=2.0, alpha=0.5):  # soft: 선생이 낸 점수 (⑤의 정제)",
   "s": 1
  },
  {
   "t": "    torch.manual_seed(0)",
   "s": 1
  },
  {
   "t": "    model.body.requires_grad_(False)",
   "s": 1
  },
  {
   "t": "    for layer in model.body.encoder.layer[len(model.body.encoder.layer) - train_layers:]:",
   "s": 1
  },
  {
   "t": "        layer.requires_grad_(True)  # 위쪽 train_layers개 층만 고치기",
   "s": 1
  },
  {
   "t": "    opt = torch.optim.AdamW([{\"params\": [p for p in model.body.parameters() if p.requires_grad], \"lr\": 5e-5},",
   "s": 1
  },
  {
   "t": "                             {\"params\": model.head.parameters(), \"lr\": 1e-3}])",
   "s": 1
  },
  {
   "t": "    for _ in range(epochs):",
   "s": 1
  },
  {
   "t": "        model.train()",
   "s": 1
  },
  {
   "t": "        for idx in torch.randperm(len(texts)).split(16):",
   "s": 1
  },
  {
   "t": "            logits = model(encode([texts[i] for i in idx]))",
   "s": 1
  },
  {
   "t": "            known = labels[idx] >= 0  # 라벨이 -1이면 정답 없음 (라벨 없는 문장)",
   "s": 1
  },
  {
   "t": "            loss = F.cross_entropy(logits[known], labels[idx][known]) if known.any() else logits.sum() * 0",
   "s": 1
  },
  {
   "t": "            if soft is not None:",
   "s": 1
  },
  {
   "t": "                kd = F.kl_div(F.log_softmax(logits / T, -1), F.softmax(soft[idx] / T, -1), reduction=\"batchmean\") * T * T",
   "s": 1
  },
  {
   "t": "                loss = alpha * loss + (1 - alpha) * kd",
   "s": 1
  },
  {
   "t": "            opt.zero_grad()",
   "s": 1
  },
  {
   "t": "            loss.backward()",
   "s": 1
  },
  {
   "t": "            opt.step()",
   "s": 1
  },
  {
   "t": "    return model.eval()",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "def benchmark(model):  # 크기, 문장 하나의 시간, 정확도를 같은 잣대로",
   "s": 1
  },
  {
   "t": "    buf = io.BytesIO()",
   "s": 1
  },
  {
   "t": "    torch.save(model.state_dict(), buf)  # 저장했을 때의 파일 크기",
   "s": 1
  },
  {
   "t": "    texts = part[\"test\"][\"text\"].tolist()",
   "s": 1
  },
  {
   "t": "    with torch.no_grad():",
   "s": 1
  },
  {
   "t": "        pred = torch.cat([model(encode(texts[i:i + 32])).argmax(-1) for i in range(0, len(texts), 32)])",
   "s": 1
  },
  {
   "t": "        for t in texts[:20]:  # 한 번 데우기 (처음 몇 번은 늘 느려요)",
   "s": 1
  },
  {
   "t": "            model(encode([t]))",
   "s": 1
  },
  {
   "t": "        runs = {\"this\": [], \"base\": []}",
   "s": 1
  },
  {
   "t": "        for _ in range(5):  # 이 모델과 기준을 번갈아 다섯 번씩 (PC가 바빠지면 둘 다 똑같이 느려지게)",
   "s": 1
  },
  {
   "t": "            for key, m in ((\"this\", model), (\"base\", teacher)):",
   "s": 1
  },
  {
   "t": "                t0 = time.perf_counter()",
   "s": 1
  },
  {
   "t": "                for t in texts:  # 문장 120개를 하나씩",
   "s": 1
  },
  {
   "t": "                    m(encode([t]))",
   "s": 1
  },
  {
   "t": "                runs[key].append((time.perf_counter() - t0) / len(texts) * 1000)",
   "s": 1
  },
  {
   "t": "    ms = min(runs[\"this\"])  # 다섯 번 중 가장 빠른 값",
   "s": 1
  },
  {
   "t": "    return {\"MB\": buf.getbuffer().nbytes / 2 ** 20, \"ms\": ms, \"x\": ms / min(runs[\"base\"]), \"F1\": f1_score(y[\"test\"], pred, average=\"macro\")}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "teacher = train(Classifier(AutoModel.from_pretrained(MODEL)), part[\"train\"][\"text\"].tolist(), y[\"train\"], train_layers=4)",
   "s": 1
  },
  {
   "t": "results = {\"기준 (12층)\": benchmark(teacher)}",
   "s": 1
  },
  {
   "t": "",
   "s": 1
  },
  {
   "t": "# ② 무게가 어디에 있나",
   "s": 2
  },
  {
   "t": "count = lambda module: sum(p.numel() for p in module.parameters())  # noqa: E731",
   "s": 2
  },
  {
   "t": "weights = {\"토큰 Embedding 표\": count(teacher.body.embeddings.word_embeddings),",
   "s": 2
  },
  {
   "t": "           \"위치와 그 밖의 Embedding\": count(teacher.body.embeddings) - count(teacher.body.embeddings.word_embeddings),",
   "s": 2
  },
  {
   "t": "           \"Transformer 층 12개\": count(teacher.body.encoder), \"분류 층\": count(teacher.head)}",
   "s": 2
  },
  {
   "t": "",
   "s": 2
  },
  {
   "t": "# ③ 어휘 줄이기: 이 일에 나오는 토큰만 남기기",
   "s": 3
  },
  {
   "t": "used = sorted({i for row in tok([\"query: \" + t for t in df[\"text\"]])[\"input_ids\"] for i in row}",
   "s": 3
  },
  {
   "t": "              | {tok.pad_token_id, tok.unk_token_id, tok.cls_token_id, tok.sep_token_id})",
   "s": 3
  },
  {
   "t": "remap = torch.full((len(tok),), used.index(tok.unk_token_id))  # 원래 번호 → 새 번호 (모르는 토큰은 <unk>로)",
   "s": 3
  },
  {
   "t": "remap[torch.tensor(used)] = torch.arange(len(used))",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "class SmallVocab(torch.nn.Module):  # Embedding 표에서 쓰는 줄만 남기고, 들어오는 번호를 바꿔 넣기",
   "s": 3
  },
  {
   "t": "    def __init__(self, clf):",
   "s": 3
  },
  {
   "t": "        super().__init__()",
   "s": 3
  },
  {
   "t": "        self.clf = copy.deepcopy(clf)",
   "s": 3
  },
  {
   "t": "        old = self.clf.body.embeddings.word_embeddings.weight.data",
   "s": 3
  },
  {
   "t": "        self.clf.body.embeddings.word_embeddings = torch.nn.Embedding(len(used), 384, _weight=old[torch.tensor(used)].clone())",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "    def forward(self, b):",
   "s": 3
  },
  {
   "t": "        return self.clf({\"input_ids\": remap[b[\"input_ids\"]], \"attention_mask\": b[\"attention_mask\"]})",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "small_teacher = SmallVocab(teacher).eval()",
   "s": 3
  },
  {
   "t": "results[\"어휘 줄이기\"] = benchmark(small_teacher)",
   "s": 3
  },
  {
   "t": "",
   "s": 3
  },
  {
   "t": "# ④ 지식 정제: 큰 모델의 답을 작은 모델에게 가르치기",
   "s": 4
  },
  {
   "t": "def first_layers(clf, n):  # 선생의 아래쪽 n개 층만 가진 몸통 (위쪽 층을 떼어 냄)",
   "s": 4
  },
  {
   "t": "    body = copy.deepcopy(clf.body)",
   "s": 4
  },
  {
   "t": "    body.encoder.layer = body.encoder.layer[:n]",
   "s": 4
  },
  {
   "t": "    body.config.num_hidden_layers = n",
   "s": 4
  },
  {
   "t": "    return body",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "X, Y = part[\"train\"][\"text\"].tolist(), y[\"train\"]",
   "s": 4
  },
  {
   "t": "X_un = part[\"unlabeled\"][\"text\"].tolist()",
   "s": 4
  },
  {
   "t": "with torch.no_grad():",
   "s": 4
  },
  {
   "t": "    soft_train = teacher(encode(X))  # 선생이 train 문장에 준 점수",
   "s": 4
  },
  {
   "t": "    soft_un = teacher(encode(X_un))  # 선생이 라벨 없는 문장에 준 점수",
   "s": 4
  },
  {
   "t": "def student(seed):  # 선생의 아래 6층 + 새 분류 층 (분류 층의 처음 값만 seed로 다르게)",
   "s": 4
  },
  {
   "t": "    s = Classifier(first_layers(teacher, 6))",
   "s": 4
  },
  {
   "t": "    torch.manual_seed(seed)",
   "s": 4
  },
  {
   "t": "    s.head.reset_parameters()",
   "s": 4
  },
  {
   "t": "    return s",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "def test_f1(model):",
   "s": 4
  },
  {
   "t": "    with torch.no_grad():",
   "s": 4
  },
  {
   "t": "        return f1_score(y[\"test\"], model(encode(part[\"test\"][\"text\"].tolist())).argmax(-1), average=\"macro\")",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "ways = {\"정답만\": lambda s: train(s, X, Y, train_layers=2),  # 위쪽 2층과 분류 층만 학습",
   "s": 4
  },
  {
   "t": "        \"선생의 점수로\": lambda s: train(s, X, Y, train_layers=2, soft=soft_train),",
   "s": 4
  },
  {
   "t": "        \"+ 라벨 없는 문장\": lambda s: train(s, X + X_un, torch.cat([Y, torch.full((len(X_un),), -1)]), train_layers=2, soft=torch.cat([soft_train, soft_un]))}",
   "s": 4
  },
  {
   "t": "student_f1 = {name: [test_f1(fit(student(seed))) for seed in range(4)] for name, fit in ways.items()}  # 처음 값 4가지로",
   "s": 4
  },
  {
   "t": "best_student = ways[\"+ 라벨 없는 문장\"](student(0))",
   "s": 4
  },
  {
   "t": "results[\"6층 학생\"] = benchmark(best_student)",
   "s": 4
  },
  {
   "t": "",
   "s": 4
  },
  {
   "t": "# ⑤ 양자화: 32비트 실수를 8비트 정수로",
   "s": 5
  },
  {
   "t": "def int8(model, per_channel):  # 몸통의 Linear 층 가중치를 int8로 (분류 층은 그대로)",
   "s": 5
  },
  {
   "t": "    m = copy.deepcopy(model)",
   "s": 5
  },
  {
   "t": "    qconfig = torch.ao.quantization.per_channel_dynamic_qconfig if per_channel else torch.ao.quantization.default_dynamic_qconfig",
   "s": 5
  },
  {
   "t": "    body = m.clf.body if isinstance(m, SmallVocab) else m.body",
   "s": 5
  },
  {
   "t": "    torch.ao.quantization.quantize_dynamic(body, {torch.nn.Linear: qconfig}, dtype=torch.qint8, inplace=True)",
   "s": 5
  },
  {
   "t": "    return m",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "for per_channel in (False, True):",
   "s": 5
  },
  {
   "t": "    results[f\"어휘 줄이기 + int8 ({'출력 칸마다' if per_channel else '층마다'} 눈금)\"] = benchmark(int8(small_teacher, per_channel))",
   "s": 5
  },
  {
   "t": "w = teacher.body.encoder.layer[0].attention.self.query.weight.data  # 양자화 오차를 직접 보기: 첫 층 Query 행렬",
   "s": 5
  },
  {
   "t": "scale = w.abs().max() / 127  # 층마다 눈금 하나: 가장 큰 값을 127에 맞추기",
   "s": 5
  },
  {
   "t": "w_back = (w / scale).round().clamp(-128, 127) * scale  # int8로 바꿨다가 되돌린 값",
   "s": 5
  },
  {
   "t": "",
   "s": 5
  },
  {
   "t": "# ⑥ 가지치기: 크기가 작은 가중치를 0으로",
   "s": 6
  },
  {
   "t": "def pruned(model, amount):  # 몸통의 모든 Linear 가중치 중 절댓값이 작은 amount만큼을 0으로",
   "s": 6
  },
  {
   "t": "    m = copy.deepcopy(model)",
   "s": 6
  },
  {
   "t": "    targets = [(mod, \"weight\") for mod in m.body.encoder.modules() if isinstance(mod, torch.nn.Linear)]",
   "s": 6
  },
  {
   "t": "    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)",
   "s": 6
  },
  {
   "t": "    for mod, name in targets:",
   "s": 6
  },
  {
   "t": "        prune.remove(mod, name)  # 0으로 만든 걸 가중치에 확정",
   "s": 6
  },
  {
   "t": "    return m",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "for amount in (0.3, 0.5, 0.7):",
   "s": 6
  },
  {
   "t": "    results[f\"가지치기 {amount:.0%}\"] = benchmark(pruned(teacher, amount))",
   "s": 6
  },
  {
   "t": "",
   "s": 6
  },
  {
   "t": "# ⑦ 합쳐 보기",
   "s": 7
  },
  {
   "t": "final = int8(SmallVocab(best_student).eval(), per_channel=True)",
   "s": 7
  },
  {
   "t": "results[\"6층 학생 + 어휘 줄이기 + int8\"] = benchmark(final)",
   "s": 7
  }
 ],
 "names": {
  "1": "기준 모델과 재는 방법",
  "2": "무게가 어디에 있나",
  "3": "어휘 줄이기: 이 일에 나오는 토큰만 남기기",
  "4": "지식 정제: 큰 모델의 답을 작은 모델에게 가르치기",
  "5": "양자화: 32비트 실수를 8비트 정수로",
  "6": "가지치기: 크기가 작은 가중치를 0으로",
  "7": "합쳐 보기"
 },
 "range": [
  1,
  7
 ],
 "outs": {
  "why": {
   "title": "같은 test 120문장으로 잰 결과 (자세한 건 ⑦)",
   "cols": [
    "모델",
    "저장 크기",
    "문장 하나 (CPU)",
    "기준 대비 시간",
    "test macro F1"
   ],
   "rows": [
    [
     "문장 분류 쪽의 분류기 그대로",
     "448.9 MB",
     "9.8 ms",
     "0.95배",
     "0.915"
    ],
    [
     "<b>이 쪽에서 가볍게 만든 것</b>",
     "12.0 MB",
     "6.2 ms",
     "0.58배",
     "<b>0.737</b>"
    ]
   ],
   "notes": [
    "크기는 37분의 1, 시간은 기준의 58%로 줄고, 정확도는 0.18 떨어졌어요.",
    "가볍게 만드는 방법마다 줄여 주는 것(크기, 시간)과 잃는 것(정확도)이 달라요. 이 쪽은 방법을 하나씩 따로 재서, 무엇이 무엇을 줄이는지 나눠 봐요."
   ]
  },
  "s1": {
   "title": "① 기준 모델 (문장 분류 쪽 ⑤와 같은 방법, 위쪽 4층 미세조정)",
   "cols": [
    "모델",
    "저장 크기",
    "문장 하나 (CPU)",
    "기준 대비 시간",
    "test macro F1"
   ],
   "rows": [
    [
     "<b>기준 (12층)</b>",
     "448.9 MB",
     "9.8 ms",
     "0.95배",
     "<b>0.915</b>"
    ]
   ],
   "notes": [
    "<b>저장 크기</b>: <code>torch.save</code>로 저장한 파일 크기예요. 서버의 메모리, 앱의 내려받기 크기와 같은 이야기예요.",
    "<b>문장 하나</b>: 한 번 데운 뒤 문장 120개를 하나씩 넣는 걸 다섯 번 반복해서 가장 빠른 평균이에요. 실제 서비스처럼 요청이 하나씩 올 때의 시간이에요.",
    "<b>기준 대비 시간</b>: ms는 그때 PC가 얼마나 바쁜지에 따라 크게 달라져요. 모델을 하나씩 다른 때에 재면 그 차이가 모델 차이처럼 보여서, 잴 때마다 기준 모델과 번갈아 재고 기준의 몇 배인지를 봐요. 그래도 계산이 기준과 똑같은 모델(기준 자신, 뒤에 나올 ③과 ⑥)이 최대 11%까지 달리 재졌으니, 11%보다 작은 차이는 같은 걸로 봐요.",
    "<b>macro F1</b>: 문장 분류 쪽과 같은 test(처음 보는 말투)예요."
   ]
  },
  "f_w": {
   "title": "가중치 1억 1,765만 개의 자리",
   "images": [
    [
     "img/compress/weight_share.png",
     ""
    ]
   ],
   "notes": [
    "분홍(토큰 Embedding 표)이 82%예요. 무엇을 줄일지 고르기 전에, 무게가 어디에 있는지부터 봐요."
   ],
   "chart": true
  },
  "s2": {
   "title": "② 자리별 가중치 수",
   "lines": [
    "토큰 Embedding 표 → 96,014,208개 (82%)",
    "위치와 그 밖의 Embedding → 198,144개 (0%)",
    "Transformer 층 12개 → 21,293,568개 (18%)",
    "분류 층 → 1,155개 (0%)",
    [
     "합계 → 117,507,075개",
     1
    ]
   ],
   "notes": [
    "가중치의 82%가 토큰 번호 → 벡터 표예요. 25만 개 어휘에 384칸씩이에요. 실제로 \"생각\"하는 Transformer 층 12개는 18%뿐이에요. 여러 언어를 다루는 작은 모델일수록 이렇게 어휘 표가 큰 몫을 차지해요."
   ]
  },
  "s3": {
   "title": "③ 어휘 250,002개 중 이 데이터에 나오는 316개만 남기면",
   "cols": [
    "모델",
    "저장 크기",
    "문장 하나 (CPU)",
    "기준 대비 시간",
    "test macro F1"
   ],
   "rows": [
    [
     "기준 (12층)",
     "448.9 MB",
     "9.8 ms",
     "0.95배",
     "0.915"
    ],
    [
     "<b>어휘 줄이기</b>",
     "83.1 MB",
     "10.6 ms",
     "0.99배",
     "<b>0.915</b>"
    ]
   ],
   "notes": [
    "이 일의 문장 780개에 나오는 토큰은 316개(0.13%)뿐이에요. 나머지 줄은 한 번도 쓰이지 않는 표예요.",
    "쓰는 줄만 남겼더니 크기가 5.4분의 1이 됐고, 정확도는 소수점 셋째 자리까지 똑같아요. 쓰는 줄의 값은 그대로라서 계산 결과가 같아요. 시간 차이는 재는 때의 흔들림이에요. 계산은 똑같고, 표에서 줄을 찾는 건 원래 아주 빠른 일이라서요.",
    "⚠️ 대신 이 316개에 없는 토큰이 들어오면 모두 <code>&lt;unk&gt;</code>로 바뀌어요. 쓸 때 들어올 글을 넉넉히 모아 어휘를 정해야 해요."
   ]
  },
  "s4": {
   "title": "④ 아래 6층만 가진 학생 (위쪽 2층과 분류 층만 학습, 분류 층의 처음 값 4가지)",
   "cols": [
    "학습 방법",
    "처음 값 0",
    "1",
    "2",
    "3",
    "평균"
   ],
   "rows": [
    [
     "정답만",
     "0.737",
     "0.824",
     "0.737",
     "0.776",
     "<b>0.768</b>"
    ],
    [
     "선생의 점수로",
     "0.737",
     "0.824",
     "0.737",
     "0.757",
     "<b>0.764</b>"
    ],
    [
     "+ 라벨 없는 문장",
     "0.737",
     "0.833",
     "0.767",
     "0.796",
     "<b>0.783</b>"
    ],
    [
     "(참고) 기준 12층",
     "",
     "",
     "",
     "",
     "0.915"
    ]
   ],
   "notes": [
    "층이 절반이라 문장 하나의 시간이 기준의 57%예요. 크기는 408MB로 거의 그대로인데, 여전히 25만 어휘 표를 갖고 있어서예요(⑦에서 ③과 합쳐요).",
    "정확도는 기준(0.915)보다 떨어져요. 그리고 같은 방법이라도 분류 층의 처음 값에 따라 최대 0.10만큼 흔들려요. 방법 사이의 평균 차이(0.02)보다 커서, 이 실험만으로는 어느 방법이 낫다고 말하기 어려워요. 평균으로는 라벨 없는 문장까지 쓴 정제가 가장 높아요(0.783).",
    "선생의 점수에는 정답보다 많은 정보가 있어요. 예를 들어 \"결제가 안 돼요. 버튼을 눌러도 아무 반응이 없어요.\"에 선생이 준 확률은 버그 0.99, 기능 요청 0.00, 질문 0.00이에요. 온도 2로 나누면 버그 0.90, 기능 요청 0.05, 질문 0.05로 펴져서, \"정답은 아니지만 어느 쪽과 더 닮았나\"가 드러나요.",
    "그런데 선생이 점수를 붙여 준 문장(train과 라벨 없는 문장)은 모두 train과 같은 말투예요. 학생이 떼어 낸 위쪽 6층은 처음 보는 말투를 다루던 부분인데, 그 말투의 문장이 없으니 선생도 그 능력을 보여 줄 기회가 없어요. 정제가 잘 되려면 선생이 점수를 매겨 줄 <b>다양한</b> 문장이 많아야 해요."
   ]
  },
  "s5": {
   "title": "⑤ ③의 모델을 int8로 (몸통의 Linear 층만)",
   "cols": [
    "모델",
    "저장 크기",
    "문장 하나 (CPU)",
    "기준 대비 시간",
    "test macro F1"
   ],
   "rows": [
    [
     "어휘 줄이기 (32비트 실수)",
     "83.1 MB",
     "10.6 ms",
     "0.99배",
     "0.915"
    ],
    [
     "+ int8, 층마다 눈금 하나",
     "22.0 MB",
     "12.8 ms",
     "1.00배",
     "0.846"
    ],
    [
     "<b>+ int8, 출력 칸마다 눈금</b>",
     "22.7 MB",
     "14.4 ms",
     "1.06배",
     "<b>0.875</b>"
    ]
   ],
   "notes": [
    "숫자 하나가 4바이트에서 1바이트가 돼서 크기가 83MB에서 23MB가 됐어요. 남은 무게의 대부분이 Linear 층이라 거의 4분의 1이에요.",
    "첫 층 Query 행렬로 직접 보면, 눈금 하나(0.00484) 단위로 반올림해서 생기는 차이가 원래 값 크기의 1.4%예요. 작아 보여도 층 12개를 지나며 쌓여서 F1이 0.07 떨어졌어요. 출력 칸마다 눈금을 따로 잡으면 값의 범위에 더 잘 맞아서 덜 떨어져요(0.875).",
    "이 PC에서는 int8 전의 102%라서, ①에서 정한 흔들림(11%)을 넘게 빨라지지는 않았어요. 문장 하나는 계산이 작아서, 입력을 int8로 바꾸고 되돌리는 준비 시간이 아낀 시간을 깎아 먹을 수 있어요. 한 번에 여러 문장을 넣거나 int8 전용 계산이 빠른 기기(휴대폰, 일부 GPU)에서 효과가 커요."
   ]
  },
  "s5w": {
   "title": "양자화 전후의 가중치 (첫 층 Query 행렬의 앞 4개)",
   "lines": [
    "원래 → [0.08575, -0.08826, -0.10052, -0.32251]",
    "int8로 바꿨다 되돌림 → [0.08713, -0.08713, -0.10165, -0.32431]",
    "정수로는 → [18, -18, -21, -67]"
   ],
   "notes": [
    "int8은 -128~127 정수예요. 가장 큰 절댓값을 127에 맞추는 눈금(scale)을 정하고, 값마다 \"눈금 몇 칸인가\"만 저장해요. 쓸 때 눈금을 곱해 되돌려요."
   ]
  },
  "s6": {
   "title": "⑥ 몸통의 Linear 가중치 중 절댓값이 작은 것을 0으로 (다시 학습하지 않음)",
   "cols": [
    "모델",
    "저장 크기",
    "문장 하나 (CPU)",
    "기준 대비 시간",
    "test macro F1"
   ],
   "rows": [
    [
     "기준",
     "448.9 MB",
     "9.8 ms",
     "0.95배",
     "0.915"
    ],
    [
     "가지치기 30%",
     "448.9 MB",
     "12.6 ms",
     "1.11배",
     "0.915"
    ],
    [
     "가지치기 50%",
     "448.9 MB",
     "12.2 ms",
     "0.93배",
     "0.807"
    ],
    [
     "가지치기 70%",
     "448.9 MB",
     "11.4 ms",
     "1.04배",
     "0.167"
    ]
   ],
   "notes": [
    "30%를 0으로 만들어도 F1이 그대로예요. 학습된 가중치의 상당수가 0에 가까워서 지워도 결과가 거의 안 바뀌어요. 더 지우면 떨어지기 시작해서 50%에서 0.807, 70%에서는 0.167까지 무너져요. 무너지기 직전인 곳에서는 학습 때의 아주 작은 차이에도 결과가 크게 흔들려서, 한 번 잰 값을 믿기 어려워요.",
    "크기는 그대로이고, 시간 차이는 ①에서 말한 재는 때의 흔들림이에요. 50%를 지운 모델도 가중치의 50%가 0일 뿐, 여전히 같은 크기의 표에 0이 적혀 있고 0을 곱하는 계산도 그대로 해요. 크기와 시간까지 줄이려면 0을 빼고 저장하는 희소 형식과 그걸 빨리 계산하는 전용 계산이 필요해요.",
    "실제로는 조금씩 지우고 다시 학습하기를 되풀이해요. 학습 중에 \"0 쪽으로 움직이는\" 가중치를 지우는 movement pruning이 미세조정 모델에서 더 잘 돼요. 층이나 head를 통째로 떼어 내면(④의 학생처럼) 실제로 빨라져요."
   ]
  },
  "f_all": {
   "title": "방법별 크기와 정확도",
   "images": [
    [
     "img/compress/size_vs_f1.png",
     ""
    ]
   ],
   "notes": [
    "어휘 줄이기는 정확도를 잃지 않고 크기를 크게 줄여요. 학생은 시간을, int8은 크기를 줄이는 대신 정확도를 내줘요(학생 0.18, int8 0.04). 가지치기는 다시 학습하지 않으면 크기도 계산량도 그대로예요."
   ],
   "chart": true
  },
  "s7": {
   "title": "⑦ 모두 같은 잣대로",
   "cols": [
    "모델",
    "저장 크기",
    "문장 하나 (CPU)",
    "기준 대비 시간",
    "test macro F1"
   ],
   "rows": [
    [
     "기준 (12층)",
     "448.9 MB",
     "9.8 ms",
     "0.95배",
     "0.915"
    ],
    [
     "어휘 줄이기",
     "83.1 MB",
     "10.6 ms",
     "0.99배",
     "0.915"
    ],
    [
     "6층 학생",
     "408.2 MB",
     "7.3 ms",
     "0.57배",
     "0.737"
    ],
    [
     "어휘 줄이기 + int8 (층마다 눈금)",
     "22.0 MB",
     "12.8 ms",
     "1.00배",
     "0.846"
    ],
    [
     "어휘 줄이기 + int8 (출력 칸마다 눈금)",
     "22.7 MB",
     "14.4 ms",
     "1.06배",
     "0.875"
    ],
    [
     "가지치기 30%",
     "448.9 MB",
     "12.6 ms",
     "1.11배",
     "0.915"
    ],
    [
     "가지치기 50%",
     "448.9 MB",
     "12.2 ms",
     "0.93배",
     "0.807"
    ],
    [
     "가지치기 70%",
     "448.9 MB",
     "11.4 ms",
     "1.04배",
     "0.167"
    ],
    [
     "<b>6층 학생 + 어휘 줄이기 + int8</b>",
     "12.0 MB",
     "6.2 ms",
     "0.58배",
     "<b>0.737</b>"
    ]
   ],
   "notes": [
    "④의 학생(라벨 없는 문장까지 정제, 처음 값 0)에 ③의 어휘 줄이기와 ⑤의 int8을 더하니 12.0MB, 6.2ms예요. 크기는 37분의 1, 시간은 기준의 58%예요.",
    "정확도는 0.915에서 0.737까지 떨어졌어요. 학생으로 바꾸며 0.18을 잃었고, 어휘 줄이기와 int8을 더해도 더 잃지는 않았어요. 이 정도를 받아들일 수 있는지는 어디에 쓰느냐에 달려요. 휴대폰에서 바로 돌려야 한다면 좋은 거래이고, 서버에서 정확도가 가장 중요하다면 ③만 하는 게 나아요."
   ]
  }
 },
 "flow": [
  [
   1,
   "기준과 잣대",
   "크기, 시간, F1",
   "449MB, 0.92"
  ],
  [
   2,
   "무게의 자리",
   "가중치 세기",
   "Embedding 82%"
  ],
  [
   3,
   "어휘 줄이기",
   "쓰는 줄만",
   "83MB, F1 그대로"
  ],
  [
   4,
   "지식 정제",
   "6층 학생",
   "평균 F1 0.78, 시간 57%"
  ],
  [
   5,
   "양자화",
   "int8",
   "23MB"
  ],
  [
   6,
   "가지치기",
   "작은 값을 0",
   "다시 학습해야"
  ],
  [
   7,
   "합치기",
   "③+④+⑤",
   "12MB"
  ]
 ],
 "env": "Python 3.13.16, PyTorch 2.14.0+cu130, transformers 5.22.2, intfloat/multilingual-e5-small, CPU 스레드 4개"
};
