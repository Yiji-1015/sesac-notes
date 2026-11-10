import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
"""Day11 'PyTorch 한 장 정리' 그림 — 배치 하나가 지나가며 바뀌는 shape.

shape 숫자는 FashionMNIST와 같은 모양의 무작위 텐서를 치트시트의 Net에 실제로 넣어 얻는다(다운로드 없음).
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day11_cheat.py
"""
import torch
from torch import nn
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

torch.manual_seed(11)
MUTED = "#5b5270"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 512), nn.ReLU(), nn.Linear(512, 10))

    def forward(self, x):
        return self.layers(x)


def measure():
    x1 = torch.rand(1, 28, 28)
    X = torch.rand(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    model = Net()
    logits = model(X)
    loss = nn.CrossEntropyLoss()(logits, y)
    w = model.layers[1].weight
    loss.backward()
    return {"x1": tuple(x1.shape), "X": tuple(X.shape), "y": tuple(y.shape), "logits": tuple(logits.shape),
            "loss": tuple(loss.shape), "w": tuple(w.shape), "grad": tuple(w.grad.shape), "lossv": loss.item()}


def fig_batch_flow(m):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 13.2); ax.set_ylim(-0.75, 3.1); ax.axis("off")
    steps = [
        ("train_ds[i]", f"샘플 1개\nx {m['x1']}\ny 정수 1개", C["gray"]),
        ("DataLoader", f"64개씩 묶음\nX {m['X']}\ny {m['y']}", C["blue"]),
        ("model(X)", f"로짓\n{m['logits']}\n클래스 10개 점수", C["purple"]),
        ("loss_fn", f"배치 평균 손실\nshape {m['loss']}\n(숫자 하나)", C["pink"]),
        ("backward()", f"기울기 p.grad\n첫 층 {m['grad']}\n= 가중치와 같은 모양", C["orange"]),
        ("step()", "가중치를\n기울기 반대로\n조금 이동", C["mint"]),
    ]
    for i, (name, sub, col) in enumerate(steps):
        x = 0.15 + i * 2.18
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.85, 2.3, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#fff", ec=col, lw=2))
        ax.text(x + 0.925, 2.5, name, ha="center", va="center", fontsize=11, fontweight="bold", color=col)
        ax.text(x + 0.925, 1.45, sub, ha="center", va="center", fontsize=9.3, color=C["ink"], linespacing=1.4)
        if i < 5:
            ax.add_patch(FancyArrowPatch((x + 1.85, 1.7), (x + 2.18, 1.7), arrowstyle="-|>", mutation_scale=13,
                                         color=MUTED, lw=1.6))
    ax.annotate("", xy=(0.15 + 1 * 2.18 + 0.925, 0.5), xytext=(0.15 + 5 * 2.18 + 0.925, 0.5),
                arrowprops=dict(arrowstyle="-|>", color=C["mint"], lw=1.6, connectionstyle="arc3,rad=-0.12"))
    ax.text(6.6, -0.62, "다음 배치로 반복  ·  1 epoch = 60,000 ÷ 64 = 937.5 → 올림해서 938번 step", ha="center", fontsize=10, color=C["mint"])
    ax.set_title("배치 하나가 지나가는 길 — 칸마다 shape가 어떻게 바뀌나 (batch_size=64, FashionMNIST 모양)", loc="left")
    save(fig, 11, "batch_flow")


if __name__ == "__main__":
    m = measure()
    print(m)
    fig_batch_flow(m)
