import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from vstyle import plt, C, save, setup; setup()
# Day12 (순서 모델) 그림 만들기.
# 실행: C:/practice_assignment/.venv/Scripts/python.exe tools/visuals/make_day12.py
# 데이터는 전부 numpy로 만든 사인파 "센서" 값 (강의의 사인파 예측 · 8차원 재구성과 같은 설정).
import time

import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

DAY = 12
rng = np.random.default_rng(0)
torch.manual_seed(0)


from matplotlib.ticker import FuncFormatter, NullFormatter, FixedFormatter
_save = save


def save(fig, day, name):
    """로그 축 눈금을 '0.001', '1e-06'처럼 ASCII로 (수학 글꼴의 유니코드 마이너스 방지)."""
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    for ax in fig.axes:
        for axis, sc in [(ax.yaxis, ax.get_yscale()), (ax.xaxis, ax.get_xscale())]:
            if sc == "log" and not isinstance(axis.get_major_formatter(), FixedFormatter):
                axis.set_major_formatter(fmt); axis.set_minor_formatter(NullFormatter())
    return _save(fig, day, name)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- 1. Padding + Mask
def fig_padding_mask():
    lengths = [12, 8, 10, 5]
    T = 12
    seqs = []
    for i, L in enumerate(lengths):
        t = np.arange(L)
        seqs.append(np.sin(t * 0.6 + i) * (1 + 0.3 * i) + 1.5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.6, 1]})
    cols = [C["purple"], C["pink"], C["mint"], C["orange"]]
    for i, (s, L) in enumerate(zip(seqs, lengths)):
        base = (3 - i) * 4.2
        a1.add_patch(Rectangle((L - 0.5, base - 0.4), T - L, 3.8, fc="#e9ecef", ec="#adb5bd", hatch="//", lw=0.8))
        a1.plot(np.arange(L), s + base, "o-", color=cols[i], lw=2, ms=4)
        a1.plot(np.arange(L, T), np.zeros(T - L) + base, "o", color="#adb5bd", ms=4)
        real = s.mean(); padm = s.sum() / T
        a1.text(-0.8, base + 1.5, f"샘플{i}\nL={L}", ha="right", va="center", fontsize=10)
        if L < T:
            a1.text(L + (T - L) / 2 - 0.5, base + 2.4, "Padding 0", ha="center", fontsize=9.5, color=C["gray"])
        a1.text(T + 0.1, base + 1.5, f"평균 {real:.2f}\n(0 포함 {padm:.2f})" if L < T else f"평균 {real:.2f}",
                fontsize=9, va="center", color=C["red"] if L < T else C["ink"])
    a1.set_xlim(-3.2, T + 2.6); a1.set_ylim(-0.8, 16.8)
    a1.set_xticks(range(T)); a1.set_yticks([]); a1.set_xlabel("시점 t (T = 가장 긴 길이 12)")
    a1.set_title("길이 12·8·10·5 → 12로 맞춤 (회색 = 채운 칸)", fontsize=11.5)
    mask = np.arange(T)[None, :] >= np.array(lengths)[:, None]
    a2.imshow(mask, cmap=ListedColormap([C["mint"], "#ced4da"]), aspect="auto")
    for r in range(4):
        for c in range(T):
            a2.text(c, r, "T" if mask[r, c] else "F", ha="center", va="center", fontsize=8.5,
                    color="#495057" if mask[r, c] else "white", fontweight="bold")
    a2.set_xticks(range(T)); a2.set_yticks(range(4)); a2.set_yticklabels([f"샘플{i}" for i in range(4)])
    a2.grid(False)
    a2.set_title("mask [B=4, T=12]  (True = 빼야 할 칸)", fontsize=11.5)
    a2.set_xlabel("arange(T) >= lengths")
    save(fig, DAY, "padding_mask")


# ---------------------------------------------------------------- 2. Window (unfold) + 시간 분할
def fig_windows():
    n = 120
    t = np.arange(n)
    y = np.sin(t * 2 * np.pi / 30) + 0.1 * rng.standard_normal(n)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    spans = [(0, 84, "train", C["purple"]), (84, 102, "valid", C["orange"]), (102, 120, "test", C["mint"])]
    for a, b, nm, col in spans:
        ax.axvspan(a - 0.5, b - 0.5, color=col, alpha=0.08)
        ax.text((a + b) / 2, 2.05, nm, ha="center", fontsize=11, color=col, fontweight="bold")
    ax.plot(t, y, color="#868e96", lw=1.2, zorder=1)
    for k, (s, off) in enumerate([(3, -1.55), (20, -1.85), (40, -1.55)]):
        ax.plot(t[s:s + 12], y[s:s + 12], "o", color=C["purple"], ms=5, zorder=3)
        ax.plot(t[s + 12], y[s + 12], "*", color=C["pink"], ms=15, zorder=4)
        ax.annotate("", xy=(s + 11.5, off), xytext=(s - 0.5, off),
                    arrowprops=dict(arrowstyle="<->", color=C["purple"], lw=1.5))
        ax.text(s + 5.5, off - 0.28, f"창{k}: 과거 12개", ha="center", fontsize=9, color=C["purple"])
        ax.text(s + 13.2, y[s + 12] + 0.25, "정답", ha="center", fontsize=9, color=C["pink"], fontweight="bold")
    ax.set_ylim(-2.4, 2.4); ax.set_xlim(-2, n + 1)
    ax.set_xlabel("시점"); ax.set_ylabel("센서 값")
    ax.set_title("unfold(0, 13, 1): 창을 1칸씩 밀며 \"과거 12개(●) → 다음 1개(★)\" 문제를 만든다", fontsize=11.5)
    ax.text(118, -2.25, "시간 순서로 먼저 train/valid/test를 자르고, 각 구간 안에서만 창을 만든다 (섞지 않음)", ha="right", fontsize=9.5, color=C["orange"])
    save(fig, DAY, "windows")


# ---------------------------------------------------------------- 3. RNN Hidden state (무작위 가중치)
def fig_rnn_hidden():
    torch.manual_seed(2)
    T = 30
    t = torch.arange(T, dtype=torch.float32)
    x = torch.sin(t * 2 * np.pi / 15)
    x[18:21] = 2.0                                # 짧은 튀는 구간
    rnn = nn.RNN(1, 8, batch_first=True)
    with torch.no_grad():
        out, h = rnn(x[None, :, None])
    H = out[0].numpy().T
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    a1.plot(range(T), x.numpy(), "o-", color=C["purple"], ms=4)
    a1.axvspan(17.5, 20.5, color=C["pink"], alpha=0.15)
    a1.text(19, 2.25, "튀는 값", ha="center", color=C["pink"], fontsize=10)
    a1.set_ylabel("입력 x_t"); a1.set_ylim(-1.4, 2.8)
    a1.set_title("nn.RNN(1, 8): 매 시점 같은 W로 x_t와 h_(t-1)을 섞어 h_t (8칸 메모) 갱신", fontsize=11.5)
    im = a2.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    a2.set_yticks(range(8)); a2.set_yticklabels([f"h[{i}]" for i in range(8)], fontsize=9)
    a2.set_xlabel("시점 t"); a2.grid(False)
    cb = fig.colorbar(im, ax=a2, fraction=0.025, pad=0.01); cb.set_label("tanh 값 (-1~1)")
    a1.text(-0.3, 2.35, "학습 전 무작위 가중치 (seed 2)", ha="left", fontsize=9, color=C["gray"])
    save(fig, DAY, "rnn_hidden")


# ---------------------------------------------------------------- 4. 곱셈 반복 → 소실/폭발
def fig_vanishing():
    t = np.arange(0, 101)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for v, col, lab in [(1.05, C["red"], "1.05 (조금만 커도 폭발)"), (0.99, C["mint"], "0.99 (LSTM f가 1에 가까울 때)"),
                        (0.9, C["purple"], "0.9 (RNN 예)"), (0.5, C["orange"], "0.5")]:
        ax.plot(t, v ** t, color=col, lw=2.5, label=lab)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1e3)
    ax.axhline(1, color="#adb5bd", lw=1)
    for v, col in [(0.9, C["purple"]), (0.99, C["mint"])]:
        ax.scatter([50], [v ** 50], color=col, zorder=3)
        ax.annotate(f"{v}^50 = {v ** 50:.3f}", (50, v ** 50), xytext=(55, v ** 50 * (0.12 if v == 0.9 else 3)),
                    fontsize=10, color=col, fontweight="bold")
    ax.set_xlabel("거꾸로 거쳐 온 시점 수 t"); ax.set_ylabel("남는 기울기 비율 (로그)")
    ax.set_title("같은 값을 t번 곱하면: 1보다 작으면 사라지고, 크면 폭발", fontsize=11.5)
    ax.legend(loc="lower left", fontsize=9.5)
    save(fig, DAY, "vanishing")


# ---------------------------------------------------------------- 5. 실측: 시점별 기울기 크기 RNN vs LSTM
def fig_grad_timestep():
    T, H = 60, 16
    res = {}
    for name in ["RNN", "LSTM", "LSTM_fb"]:
        norms = []
        for seed in range(20):
            torch.manual_seed(seed)
            m = nn.RNN(1, H, batch_first=True) if name == "RNN" else nn.LSTM(1, H, batch_first=True)
            if name == "LSTM_fb":
                with torch.no_grad():
                    m.bias_ih_l0[H:2 * H] = 3.0          # forget gate bias → f ≈ 0.95
            x = torch.randn(1, T, 1, requires_grad=True)
            out = m(x)[0]
            out[0, -1].sum().backward()
            norms.append(x.grad[0, :, 0].abs().numpy())
        res[name] = np.mean(norms, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    dist = np.arange(T)[::-1]                           # 마지막 시점으로부터 거리
    for name, col, lab in [("RNN", C["purple"], "RNN (기본 초기화)"), ("LSTM", C["orange"], "LSTM (기본, f 약 0.5)"),
                           ("LSTM_fb", C["mint"], "LSTM (forget bias 3 → f 약 0.95)")]:
        ax.plot(dist, res[name], color=col, lw=2.4, label=lab)
    ax.set_yscale("log"); ax.set_xlim(0, T - 1)
    ax.set_xlabel("마지막 시점에서 몇 칸 과거인가"); ax.set_ylabel("|d출력 / d x_t| (로그)")
    ax.set_title("실측: 마지막 출력의 기울기가 과거 입력까지 얼마나 닿나", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.text(0.99, 0.02, "학습 전 · H=16 · T=60 · seed 20개 평균", transform=ax.transAxes, ha="right", fontsize=8.5, color=C["gray"])
    print({k: (float(v[-1]), float(v[0])) for k, v in res.items()})
    save(fig, DAY, "grad_timestep")


# ---------------------------------------------------------------- 6. Autoencoder 학습 + 재구성
def make_normal(n):
    t = np.arange(8)
    amp = rng.uniform(0.6, 1.2, (n, 1)); ph = rng.uniform(0, 2 * np.pi, (n, 1))
    return (amp * np.sin(t * 2 * np.pi / 8 + ph) + 0.05 * rng.standard_normal((n, 8))).astype(np.float32)


def make_anom(n):
    x = make_normal(n)
    kind = rng.integers(0, 3, n)
    for i in range(n):
        if kind[i] == 0:
            x[i, rng.integers(0, 8)] += rng.choice([-1, 1]) * 2.0          # 한 칸 튐
        elif kind[i] == 1:
            x[i] = 0.8 * rng.standard_normal(8)                            # 패턴 없는 잡음
        else:
            x[i, 4:] = x[i, 3]                                             # 중간에 멈춤(평평)
    return x.astype(np.float32)


def make_ae(z=2):
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, z), nn.ReLU(),
                         nn.Linear(z, 4), nn.ReLU(), nn.Linear(4, 8))


def make_ae_wide(z):
    # Latent 크기 비교용: 앞뒤 층을 넉넉히(16) 두고 가운데만 z
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, z),
                         nn.Linear(z, 16), nn.ReLU(), nn.Linear(16, 8))


def train_ae(model, xtr, epochs=300, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    X = torch.tensor(xtr)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            xb = X[perm[i:i + 64]]
            opt.zero_grad(); nn.MSELoss()(model(xb), xb).backward(); opt.step()
    return model


def fig_autoencoder():
    xtr, xva = make_normal(1500), make_normal(400)
    xan = make_anom(60)
    # 같은 구조여도 시작값에 따라 ReLU가 죽을 수 있어서 seed 몇 개 중 valid가 가장 좋은 것을 씀
    best = None
    for s in range(4):
        torch.manual_seed(s)
        m = train_ae(make_ae(2), xtr, epochs=150, seed=s)
        with torch.no_grad():
            v = nn.MSELoss()(m(torch.tensor(xva)), torch.tensor(xva)).item()
        if best is None or v < best[0]:
            best = (v, m)
    model = best[1].eval()
    lf = nn.MSELoss(reduction="none")
    with torch.no_grad():
        rv = model(torch.tensor(xva)); ra = model(torch.tensor(xan))
        ev = lf(rv, torch.tensor(xva)).mean(1).numpy(); ea = lf(ra, torch.tensor(xan)).mean(1).numpy()
        zv = model[:4](torch.tensor(xva)).numpy(); za = model[:4](torch.tensor(xan)).numpy()
    thr = np.quantile(ev, 0.99)
    print("ae valid mse", best[0], "thr", thr, "anom caught", (ea > thr).mean())

    # (a) 정상 vs 이상 재구성
    i_n = int(np.argsort(ev)[len(ev) // 2])
    i_a = int(np.argmax(ea))
    ymax_err = float(((xan[i_a] - ra[i_a].numpy()) ** 2).max() * 1.15)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    for col, (x, r, e, title, c) in enumerate([
            (xva[i_n], rv[i_n].numpy(), ev[i_n], "정상 샘플 (valid)", C["mint"]),
            (xan[i_a], ra[i_a].numpy(), ea[i_a], "이상 샘플 (일부러 만든 것)", C["red"])]):
        a = axes[0, col]
        a.plot(range(8), x, "o-", color=C["ink"], lw=2, label="입력 x")
        a.plot(range(8), r, "s--", color=c, lw=2, label=r"재구성 $\hat{x}$")
        a.fill_between(range(8), x, r, color=c, alpha=0.15)
        a.set_title(f"{title}\n오차(MSE) = {e:.3f}", fontsize=11.5, color=c)
        a.legend(fontsize=9, loc="lower left"); a.set_ylim(-3, 3)
        b = axes[1, col]
        b.bar(range(8), (x - r) ** 2, color=c)
        b.set_ylabel(r"칸별 (x - $\hat{x}$)²", fontsize=9.5); b.set_xlabel("특징 위치 (8칸)")
        b.set_ylim(0, ymax_err)
        if col == 0:
            b.text(3.5, ymax_err * 0.5, "모든 칸 거의 0 (막대가 안 보일 만큼 작음)", ha="center", fontsize=10, color=C["mint"])
        b.set_xticks(range(8))
    save(fig, DAY, "ae_recon")

    # (b) 오차 분포 + 기준값
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    bins = np.logspace(np.log10(min(ev.min(), ea.min()) * 0.9), np.log10(max(ev.max(), ea.max()) * 1.1), 40)
    ax.hist(ev, bins=bins, color=C["mint"], alpha=0.75, label=f"정상 valid ({len(ev)}개)")
    ax.hist(ea, bins=bins, color=C["red"], alpha=0.6, label=f"이상 ({len(ea)}개)")
    ax.axvline(thr, color=C["purple"], lw=2.2, ls="--")
    ax.text(thr * 1.08, ax.get_ylim()[1] * 0.85, f"기준값 = valid 오차의\n99% 분위수 ({thr:.3f})", color=C["purple"], fontsize=10)
    ax.set_xscale("log"); ax.set_xlabel("샘플별 재구성 오차 (로그 눈금)"); ax.set_ylabel("개수")
    ax.set_title(f"기준값 오른쪽 = 점검 후보 · 이상 {int((ea > thr).sum())}/{len(ea)}개, 정상 {int((ev > thr).sum())}/{len(ev)}개가 넘음", fontsize=11)
    ax.legend(loc="upper left", fontsize=9.5)
    save(fig, DAY, "ae_error_hist")

    # (c) Latent 2D
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ph = np.arctan2(xva[:, 2], xva[:, 0])
    sc = ax.scatter(zv[:, 0], zv[:, 1], c=ph, cmap="twilight", s=14, alpha=0.85, label="정상 (색 = 사인파 위상)")
    ax.scatter(za[:, 0], za[:, 1], marker="x", color=C["red"], s=40, label="이상")
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")
    ax.set_title("8칸 입력을 2칸 Latent z로 줄인 지도", fontsize=11.5)
    ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02).set_label("위상 (rad)")
    save(fig, DAY, "ae_latent")

    # (d) Latent 크기 비교
    zs = [1, 2, 4, 8]
    vals = []
    for z in zs:
        vs = []
        for s in range(3):
            m = train_ae(make_ae_wide(z), xtr, epochs=120, seed=s)
            with torch.no_grad():
                vs.append(nn.MSELoss()(m(torch.tensor(xva)), torch.tensor(xva)).item())
        vals.append(min(vs))
    print("latent sizes", dict(zip(zs, vals)))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar([str(z) for z in zs], vals, color=[C["orange"], C["purple"], C["blue"], C["gray"]])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.03, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xlabel("Latent 크기 (가운데 칸 수)"); ax.set_ylabel("valid 재구성 MSE")
    ax.set_title("너무 좁으면(1) 정보를 잃고, 2부터는 거의 다 되살린다", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2)
    save(fig, DAY, "latent_size")


# ---------------------------------------------------------------- 7. Attention 가중치 heatmap (Padding Mask 전/후)
def softmax(a, axis=-1):
    a = a - a.max(axis=axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=axis, keepdims=True)


def fig_attention():
    T, E = 6, 8
    r = np.random.default_rng(4)
    vals = np.array([0.1, 0.9, 0.2, 0.85, 0.0, 0.0])          # 실제 4칸 + Padding 2칸
    x = np.stack([np.sin(vals[:, None] * np.arange(1, E + 1) * 2.0), np.cos(vals[:, None] * np.arange(1, E + 1))], -1).reshape(T, -1)[:, :E]
    Wq, Wk = r.standard_normal((E, E)) / np.sqrt(E), r.standard_normal((E, E)) / np.sqrt(E)
    q, k = x @ Wq * 2.2, x @ Wk * 2.2
    s = q @ k.T / np.sqrt(E)
    w0 = softmax(s)
    s2 = s.copy(); s2[:, 4:] = -np.inf
    w1 = softmax(s2)
    labels = ["x0", "x1", "x2", "x3", "pad", "pad"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, w, title in [(axes[0], w0, "Mask 없음: pad 칸에도 비율이 샘"), (axes[1], w1, "key_padding_mask 적용: pad 열 = 0")]:
        ax.imshow(w, cmap="Purples", vmin=0, vmax=max(w0.max(), w1.max()))
        for i in range(T):
            for j in range(T):
                ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if w[i, j] > 0.35 else C["ink"])
        ax.set_xticks(range(T)); ax.set_xticklabels(labels); ax.set_yticks(range(T)); ax.set_yticklabels(labels)
        ax.set_xlabel("Key (보는 대상)"); ax.set_ylabel("Query (보는 주체)"); ax.grid(False)
        ax.set_title(title, fontsize=11.5)
        ax.add_patch(Rectangle((3.5, -0.5), 2, T, fill=False, ec=C["red"], lw=2, ls="--"))
    fig.text(0.5, -0.02, "각 행 = 한 위치가 다른 위치들을 얼마나 보는지 (행 합 = 1).  무작위 투영 W_q·W_k(seed 4)로 직접 계산",
             ha="center", fontsize=10)
    save(fig, DAY, "attention_heatmap")


# ---------------------------------------------------------------- 8. √d 스케일 막대
def fig_softmax_scale():
    sc = np.array([8, 4, 0.])
    w0, w1 = softmax(sc), softmax(sc / 4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, w, t, s in [(axes[0], w0, "점수 [8, 4, 0] 그대로", sc), (axes[1], w1, "sqrt(16) = 4로 나눔 → [2, 1, 0]", sc / 4)]:
        bars = ax.bar(["위치1", "위치2", "위치3"], w, color=[C["purple"], C["pink"], C["mint"]])
        for b, v in zip(bars, w):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_title(t, fontsize=11.5); ax.set_ylim(0, 1.15)
        out = (w * np.array([10, 20, 30])).sum()
        ax.text(0.98, 0.9, f"출력 = Σ w×V = {out:.1f}", transform=ax.transAxes, ha="right", fontsize=10.5, color=C["ink"])
    axes[0].set_ylabel("softmax weight")
    save(fig, DAY, "softmax_scale")


# ---------------------------------------------------------------- 9. Causal mask
def fig_causal():
    T = 6
    m = np.triu(np.ones((T, T), bool), 1)                    # True = 차단(미래)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(m, cmap=ListedColormap([C["mint"], "#ced4da"]))
    for i in range(T):
        for j in range(T):
            ax.text(j, i, "X" if m[i, j] else "O", ha="center", va="center", fontsize=13,
                    color="#868e96" if m[i, j] else "white", fontweight="bold")
    ax.set_xticks(range(T)); ax.set_xticklabels([f"t{j}" for j in range(T)])
    ax.set_yticks(range(T)); ax.set_yticklabels([f"t{i}" for i in range(T)])
    ax.set_xlabel("Key: 볼 수 있는 위치"); ax.set_ylabel("Query: 지금 위치"); ax.grid(False)
    ax.set_title("Causal Mask: 대각선 위(미래) 차단", fontsize=11.5)
    save(fig, DAY, "causal_mask")


# ---------------------------------------------------------------- 10. 길이 T 비용 + 실측 시간
def fig_length_cost():
    Ts = [16, 32, 64, 128, 256, 512]
    B, D = 32, 16
    torch.manual_seed(0)
    lstm = nn.LSTM(D, D, batch_first=True).eval()
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 32, 0.0, batch_first=True), 1,
                                enable_nested_tensor=False).eval()
    res = {"LSTM": [], "Transformer": []}
    dev = "cpu"                     # GPU는 이 크기에선 고정 오버헤드만 보여서 CPU로 잼
    lstm.to(dev); enc.to(dev)
    with torch.inference_mode():
        for T in Ts:
            x = torch.randn(B, T, D, device=dev)
            for nm, f in [("LSTM", lambda: lstm(x)), ("Transformer", lambda: enc(x))]:
                for _ in range(3):
                    f()
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    f()
                if dev == "cuda":
                    torch.cuda.synchronize()
                res[nm].append((time.perf_counter() - t0) / 20 * 1000)
    print(res)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    Ta = np.array(Ts)
    a1.plot(Ta, Ta, "o-", color=C["orange"], lw=2.2, label="RNN·LSTM 순차 단계 = T")
    a1.plot(Ta, Ta ** 2, "s-", color=C["purple"], lw=2.2, label="Attention 점수 칸 = T²")
    a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.set_xticks(Ts); a1.set_xticklabels(Ts)
    a1.set_xlabel("길이 T"); a1.set_title("길이 2배 → 순차 2배, 점수표 4배", fontsize=11.5); a1.legend(fontsize=9.5)
    a1.annotate("128 → 16,384칸", (128, 128 ** 2), xytext=(20, 3e4), fontsize=9.5, color=C["purple"],
                arrowprops=dict(arrowstyle="->", color=C["purple"]))
    a2.plot(Ta, res["LSTM"], "o-", color=C["orange"], lw=2.2, label="nn.LSTM")
    a2.plot(Ta, res["Transformer"], "s-", color=C["purple"], lw=2.2, label="TransformerEncoder 1층")
    a2.set_xscale("log", base=2); a2.set_yscale("log"); a2.set_xticks(Ts); a2.set_xticklabels(Ts)
    a2.set_xlabel("길이 T"); a2.set_ylabel("순전파 1회 (ms, 로그)")
    name = f"CPU · torch {torch.__version__} · 스레드 {torch.get_num_threads()}"
    a2.set_title("실측 CPU (B=32, E=16): 길수록 Attention이 더 가파름", fontsize=11.5); a2.legend(fontsize=9.5)
    a2.text(0.98, 0.03, name, transform=a2.transAxes, ha="right", fontsize=8, color=C["gray"])
    save(fig, DAY, "length_cost")


# ---------------------------------------------------------------- 11. 파라미터 수 비교
def fig_params():
    def cnt(m):
        return sum(p.numel() for p in m.parameters())
    rnn = cnt(nn.RNN(1, 16)) + cnt(nn.Linear(16, 1))
    lstm = cnt(nn.LSTM(1, 16)) + cnt(nn.Linear(16, 1))
    ae = cnt(make_ae(2))
    tr = cnt(nn.TransformerEncoderLayer(16, 4, 32, batch_first=True)) + cnt(nn.Embedding(64, 16)) + cnt(nn.Linear(16, 2)) + 16
    tr_default = cnt(nn.TransformerEncoderLayer(16, 4, batch_first=True)) + cnt(nn.Embedding(64, 16)) + cnt(nn.Linear(16, 2)) + 16
    names = ["RNN(1,16)\n+Linear", "LSTM(1,16)\n+Linear", "Autoencoder\n8-4-2-4-8", "Transformer 1층\nFFN 32", "Transformer 1층\nFFN 기본 2048"]
    vals = [rnn, lstm, ae, tr, tr_default]
    print(dict(zip(names, vals)))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(names, vals, color=[C["orange"], C["pink"], C["mint"], C["purple"], "#adb5bd"])
    ax.set_yscale("log")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:,}", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("파라미터 수 (로그)"); ax.set_ylim(50, 3e5)
    ax.set_title("오늘 만든 모델들의 크기 (직접 센 값)", fontsize=11.5)
    ax.tick_params(axis="x", labelsize=9.5)
    save(fig, DAY, "params_compare")


if __name__ == "__main__":
    fig_padding_mask()
    fig_windows()
    fig_rnn_hidden()
    fig_vanishing()
    fig_grad_timestep()
    fig_autoencoder()
    fig_attention()
    fig_softmax_scale()
    fig_causal()
    fig_length_cost()
    fig_params()
