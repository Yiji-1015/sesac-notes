"""모델 > 순서 모델과 Transformer > 순서 모델 (RNN, LSTM, Attention, Autoencoder).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/seq_models.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import torch  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "seq"
code = S.parse((EX / "seq_models.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="seq_models.py")
outs = {}
r0 = ns["records"][0]
outs["data"] = S.box("sequences.json의 첫 기록", [f"id {r0['id']}, split {r0['split']}, label {r0['label']}, 길이 {r0['length']}", *[f"  시점 {i + 1}: {[round(v, 3) for v in row]}" for i, row in enumerate(r0["values"][:4])], "  …"],
                     ["기록마다 시점 수가 6~12로 다르고, 시점마다 센서 값이 3개예요. <code>label</code>(0 또는 1)을 맞히는 게 목표예요."])
S.claim(abs(ns["acc_mean"] - 0.5) < 0.15 and min(ns["acc_rnn"], ns["acc_lstm"], ns["acc_attn"]) >= 0.9, "평균만 보면 동전 던지기, 순서를 보는 모델은 맞힌다")
n_rec, T_max, _ = tuple(ns["X_train"].shape)
L0 = int(ns["len_train"][0])   # 첫 기록의 진짜 길이
outs["why"] = S.table("valid 20개 정확도", ["모델", "순서를 보나", "정확도"], [["평균만 보기 (②)", "아니요", f"<span class=\"bad\">{ns['acc_mean']:.2f}</span>"], ["RNN (③)", "예", f"<b>{ns['acc_rnn']:.2f}</b>"], ["LSTM (④)", "예", f"<b>{ns['acc_lstm']:.2f}</b>"], ["Attention (⑤)", "예", f"<b>{ns['acc_attn']:.2f}</b>"]],
                      ["값들의 평균만 보면 동전 던지기 수준이에요. 두 종류는 같은 값들을 <b>다른 순서</b>로 갖고 있어서, 순서를 봐야 구분돼요."])
outs["s1"] = S.box("① batch(\"train\")", [(f"X_train.shape → {tuple(ns['X_train'].shape)}", 1), f"len_train 앞 5개 → {ns['len_train'][:5].tolist()}", (f"mask_train[0] → {[int(x) for x in ns['mask_train'][0].tolist()]}", 2)],
                   [f"기록 {n_rec}개를 가장 긴 길이 {T_max}에 맞춰 {tuple(ns['X_train'].shape)} 하나의 Tensor로 만들었어요. 짧은 기록의 뒤는 0으로 채웠어요 (padding).", f"첫 기록은 길이 {L0}이라 앞 {L0}칸만 진짜(1)이고 뒤 {T_max - L0}칸은 채운 자리(0)예요. 채운 0을 진짜 값으로 읽으면 안 돼서 이 표시를 같이 들고 다녀요."])
outs["s2"] = S.box("② 평균만 보는 모델", [(f"valid 정확도 → {ns['acc_mean']:.2f}", 1)], ["채운 0은 빼고 진짜 값들만 평균했어요. 그래도 순서 정보가 사라져서 구분하지 못해요."])
outs["s3"] = S.box("③ RNN", [(f"valid 정확도 → {ns['acc_rnn']:.2f}", 1)], ["시점을 하나씩 읽으며 \"지금까지 본 것의 요약\"(hidden state)을 갱신하고, 진짜 마지막 시점의 요약으로 판단했어요."])
outs["s4"] = S.box("④ LSTM", [(f"valid 정확도 → {ns['acc_lstm']:.2f}", 1)], ["이 데이터는 짧아서 RNN도 충분했어요. 기록이 길어질수록 RNN은 앞쪽을 잊기 쉬워서 LSTM이 유리해져요."])
w = ns["weights"][0].tolist()
L = int(ns["len_valid"][0])
S.claim(sorted(range(L), key=lambda i: -w[i])[:2] == [0, 1] and w[0] + w[1] > 0.5 and all(x == 0 for x in w[L:]), "⑤ 앞의 두 시점에 가중치가 몰리고, 채운 자리는 정확히 0")
outs["s5"] = S.box(f"⑤ Attention, valid 첫 기록(길이 {L})의 시점별 가중치", [(f"정확도 → {ns['acc_attn']:.2f}", 1), (f"가중치 → {[round(x, 3) for x in w]}", 2), f"합 → {sum(w):.3f}"],
                   ["순서대로 읽는 대신, 모든 시점을 한꺼번에 보고 \"어느 시점이 중요한가\"를 점수로 매겨 섞었어요.", f"이 기록에서는 앞의 두 시점에 가중치가 몰렸어요. 길이 {L} 뒤의 채운 자리는 정확히 0이에요 (점수를 −∞로 막아서). 가중치를 보면 모델이 어디를 봤는지 알 수 있어요."])
ratio_odd = float(ns["err_odd"].max()) / float(ns["limit"])   # 튀는 창의 오차 ÷ 기준
S.claim(10 <= ratio_odd < 100 and len(ns["alerts_odd"]) == 12, "⑥ 튀는 값이 섞인 창은 오차가 기준의 수십 배이고, 그 시점을 품은 창 12개가 모두 걸린다")
outs["s6"] = S.box("⑥ Autoencoder 복원 오차", [f"창 크기 12, train 창 {tuple(ns['W_train'].shape)[0]}개", (f"이상 기준 (train 평균 + 표준편차 3배) → {float(ns['limit']):.4f}", 1),
                                                f"valid, test에서 기준을 넘은 창 → {ns['alerts']}", (f"test 35번째 시점에 일부러 +1.0을 넣으면 → 걸린 창 {ns['alerts_odd']}  (최대 오차 {float(ns['err_odd'].max()):.3f})", 2, "bad")],
                   ["정상 데이터만으로 \"12칸을 4칸으로 줄였다가 되살리는\" 법을 배웠어요. 정상 패턴은 잘 되살려서 오차가 작아요.", "튀는 값이 섞인 창은 배운 적 없는 모양이라 잘 되살리지 못해 오차가 기준의 수십 배가 됐어요. 그 시점을 품은 창 12개가 모두 걸렸어요. 정답(이상 표시) 없이도 이상을 찾을 수 있어요."])
flow = [[1, "묶기", "padding, mask", "길이가 달라도"], [2, "평균", "순서 무시", f"{ns['acc_mean']:.2f}"], [3, "RNN", "차례로 읽기", "마지막 기억"], [4, "LSTM", "오래 기억", "pack"], [5, "Attention", "중요한 시점", "가중치 합 1"], [6, "Autoencoder", "줄였다 되살리기", "오차 = 이상"]]
S.save(ROOT / "topics" / "model" / "sequence-models-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                              "env": f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CPU (seed 0)"})
why = f"""<p>센서 기록, 문장, 주가처럼 <b>순서가 있는 데이터</b>는 같은 값이라도 순서가 바뀌면 뜻이 달라요. "개가 사람을 물었다"와 "사람이 개를 물었다"처럼요.</p>{K.out("data")}
<p>순서를 무시하는 모델과 순서를 읽는 모델로 실제로 맞혀 보면:</p>{K.out("why")}
<p>그래서 순서를 읽는 모델이 따로 있어요. 차례로 읽는 <b>RNN</b>과 <b>LSTM</b>, 필요한 시점을 골라 보는 <b>Attention</b>(Transformer의 핵심), 그리고 순서 데이터의 이상을 찾는 <b>Autoencoder</b>까지 다뤄요.</p>"""
whole = """<p>길이가 다른 기록을 묶고 → 순서를 무시한 기준선 → 차례로 읽기 → 오래 기억하기 → 중요한 시점 고르기 → 정답 없이 이상 찾기. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "길이가 다른 기록을 한 묶음으로", "짧은 기록 뒤를 0으로 채워 길이를 맞추고, 진짜 값이 있는 자리를 표시해 둬요.",
           K.code_table([("pad_sequence(seqs, batch_first=True)", "가장 긴 길이에 맞춰 0으로 채워 하나의 Tensor로"), ("torch.arange(n)[None, :] < lengths[:, None]", "자리 번호가 길이보다 작으면 True. 진짜 값 표시(mask)"),
                         ("fit(model)", "모든 모델을 같은 조건(Adam, 200번)으로 학습하고 valid 정확도를 돌려주는 함수")]), "s1"),
    K.step(2, "순서를 무시하면", "진짜 값들의 평균 하나로 판단하는 기준선이에요.", K.code_table([("(X * mask[..., None]).sum(1) / lengths[:, None]", "채운 0은 빼고 평균")]), "s2"),
    K.step(3, "RNN: 차례로 읽기", "시점마다 \"이전 기억 + 지금 값\"으로 새 기억을 만들어요.",
           K.code_table([("nn.RNN(3, hidden, batch_first=True)", "입력 3개, 기억 16칸"), ("h, _ = self.rnn(X)", "모든 시점의 기억 (batch, 길이, 16)"), ("h[torch.arange(len(X)), lengths - 1]", "기록마다 <b>진짜 마지막</b> 시점의 기억. 그냥 <code>h[:, -1]</code>이면 채운 0까지 읽은 뒤의 기억이에요.")]), "s3"),
    K.step(4, "LSTM: 오래 기억하는 RNN", "무엇을 잊고 무엇을 남길지 정하는 문(gate)이 있어 긴 기록에 강해요.",
           K.code_table([("nn.LSTM", "RNN과 같은 자리에 바꿔 끼워요. ③의 클래스를 그대로 물려받았어요."), ("pack_padded_sequence(X, lengths, enforce_sorted=False)", "채운 자리를 아예 읽지 않게 묶어서 넘겨요."), ("_, (h_n, _) = self.rnn(packed)", "기록마다 진짜 마지막 기억")]), "s4"),
    K.step(5, "Attention: 중요한 시점 고르기", "시점마다 점수를 매기고, softmax로 합이 1인 가중치를 만들어 기억을 섞어요.",
           K.code_table([("self.score(h)", "시점마다 중요도 점수 하나"), ("masked_fill(~mask, float(\"-inf\"))", "채운 자리는 점수 −∞ → softmax 뒤 가중치 0"), ("torch.softmax(s, dim=1)", "점수를 합이 1인 가중치로"), ("(w[..., None] * h).sum(1)", "가중치만큼 섞은 요약")]), "s5"),
    K.step(6, "Autoencoder: 잘 복원되지 않으면 이상", "정상 데이터를 \"줄였다가 되살리기\"로 배우고, 되살리기 오차가 큰 구간을 이상으로 봐요.",
           K.code_table([("v.unfold(0, 12, 1)", "길이 12짜리 창을 한 칸씩 밀며 잘라내기"), ("nn.Sequential(nn.Linear(12, 4), nn.Tanh(), nn.Linear(4, 12))", "12 → 4로 압축(encoder) → 12로 복원(decoder)"),
                         ("((ae(W) - W) ** 2).mean(1)", "창마다 복원 오차"), ("err_train.mean() + 3 * err_train.std()", "정상 데이터 오차의 범위 밖을 이상으로")]), "s6"),
]
practice = [("문장을 다룬다", "글자를 숫자로 바꾸는 단계(Tokenizer, Embedding)가 앞에 붙어요. 요즘은 RNN보다 Attention을 쌓은 Transformer를 주로 써요 (<a href=\"llm-principles.html\">Transformer 구조</a>)."),
            ("앞을 예측한다", "지금까지의 창으로 다음 값을 맞히는 회귀로 바꾸면 돼요. 미래 값이 입력에 섞이지 않게, 시간 순서대로 train, valid, test를 나눠요."),
            ("이상 기준", "평균 + 3σ는 출발점일 뿐이에요. 알람이 너무 많거나 적으면 업무 비용을 보고 조정해요.")]
libs = [("①", "<code>pad_sequence</code>, <code>torch.arange</code>", "묶기 (torch)"), ("③④", "<code>nn.RNN</code>, <code>nn.LSTM</code>, <code>pack_padded_sequence</code>", "순서 모델"), ("⑤", "<code>torch.softmax</code>, <code>masked_fill</code>", "Attention"), ("⑥", "<code>Tensor.unfold</code>, <code>nn.Sequential</code>", "Autoencoder")]
K.build(ROOT / "topics" / "model" / "sequence-models.html", title="순서 모델", h1="순서 모델: RNN, LSTM, Attention, Autoencoder", track="모델", crumb="모델 &gt; 순서 모델과 Transformer",
        one_line="순서가 있는 데이터는 <b>순서를 읽어야</b> 구분돼요. RNN과 LSTM은 <b>차례로 읽으며 기억</b>하고, Attention은 <b>중요한 시점을 골라</b> 보고, Autoencoder는 <b>잘 되살려지지 않는 구간</b>을 이상으로 찾아요.",
        why=why, whole=whole, steps=steps, code_file="examples/seq/seq_models.py", practice=practice, libs=libs, libs_note="<code>batch</code>, <code>fit</code>, <code>windows</code>와 모델 클래스들은 이 파일 안에서 정의했어요.",
        next_rows=[("<a href=\"llm-principles.html\">Transformer 구조</a>", "Attention을 쌓은 Transformer와 LLM"), ("<a href=\"dl-tensor-training.html\">Tensor와 신경망 학습</a>", "학습 루프의 기본")],
        sources=["다루는 범위: Day12 (2026-11-02) Sequence Tensor, RNN, LSTM, Autoencoder, Transformer 입문", "데이터: <code>examples/seq/data/</code> (기존 노트와 같은 센서 기록 예제). ⑥의 튀는 값은 확인을 위해 일부러 넣은 것",
                 "결과 값: <code>tools/pages/seq_models.py</code> (seed 0)", "공식 문서: <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html\" target=\"_blank\">nn.LSTM</a>, <a href=\"https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html\" target=\"_blank\">pack_padded_sequence</a>"])
print("ok")
