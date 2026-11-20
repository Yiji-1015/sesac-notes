"""모델 > 생성 모델 > 생성 모델 다섯 가지. examples/gen/gen_points.py 하나가 ①~⑦로 자란다.
실행: python tools/pages/gen_models.py  (CPU로 2~4분)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "visuals"))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from vstyle import plt, C, setup  # noqa: E402

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "gen"
code = S.parse((EX / "gen_points.py").read_text(encoding="utf-8"))
run = lambda k, patch=None: S.run(code, k, S.workdir(EX), patch, filename="gen_points.py")  # noqa: E731
f3 = lambda x: f"{float(x):.3f}"  # noqa: E731
ns = run(7)
ENV = f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CPU (seed 0)"

setup()
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
IMG = ROOT / "topics" / "model" / "img" / "gen"
IMG.mkdir(parents=True, exist_ok=True)
Z, Zt = ns["Z"].numpy(), ns["Z_test"].numpy()
lab = ns["df"]["label"].to_numpy()[ns["is_train"].numpy()]
LIM = (-2.6, 2.6)


def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(IMG / f"{name}.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return f"img/gen/{name}.png"


def chart(title, path, cap, notes):
    o = S.images(title, [(path, cap)], notes)
    o["chart"] = True
    return o


def sc(ax, P, title, col=C["purple"], s=5, real=True):
    if real:
        ax.scatter(Zt[:, 0], Zt[:, 1], s=6, color=C["gray"], alpha=.35, label="처음 보는 진짜 점")
    ax.scatter(P[:, 0], P[:, 1], s=s, color=col, alpha=.6)
    ax.set(xlim=LIM, ylim=LIM, title=title, xticks=[], yticks=[])
    ax.set_aspect("equal")
    ax.grid(False)


outs = {}
# ① 데이터
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.scatter(Z[lab == 0, 0], Z[lab == 0, 1], s=12, color=C["purple"], label="train, 무리 0")
ax.scatter(Z[lab == 1, 0], Z[lab == 1, 1], s=12, color=C["pink"], label="train, 무리 1")
ax.scatter(Zt[:, 0], Zt[:, 1], s=10, color=C["gray"], alpha=.5, label="나머지 160개")
ax.set(xlim=LIM, ylim=LIM, title="크기를 맞춘 점 400개")
ax.set_aspect("equal")
ax.legend(fontsize=8, loc="lower left")
outs["data"] = chart("배울 데이터 (딥러닝 기초와 같은 점)", fig_save(fig, "data"), "", ["두 개의 휜 띠 모양이에요. 생성 모델은 무리 번호(label)는 보지 않고 점의 좌표만 보고, 이런 모양의 점을 새로 만들어야 해요."])
rr = ns["real_vs_real"]
outs["s1"] = S.box("① 실행 결과", [f"Z.shape → {tuple(ns['Z'].shape)} (train), Z_test.shape → {tuple(ns['Z_test'].shape)}", (f"quality(train 점, 처음 보는 점) → 정밀도 {rr['precision']}, 재현율 {rr['recall']}", 1, "newf")],
                   ["\"진짜 train 점\"을 만든 점이라고 치고 재 본 값이에요. 완벽한 생성 모델이라도 이 근처가 최선이에요. 아래 모델들의 점수는 이 값과 비교해요."])
sb = ns["scores"]
outs["s2"] = S.box("② 정규분포 하나", [f"평균 → {[round(float(v), 3) for v in ns['gauss'].mean]}", f"공분산 → {[[round(float(v), 3) for v in row] for row in ns['gauss'].covariance_matrix]}",
                                    (f"정밀도 {sb['정규분포']['precision']}, 재현율 {sb['정규분포']['recall']}", 1), (f"처음 보는 점의 평균 log 확률 → {sb['정규분포']['test_loglik']}", 2, "newf")],
                   ["재현율은 높은데 정밀도가 낮아요. 타원 하나로 띠 두 개를 덮으니, 띠 사이 빈 곳에도 점을 많이 만들어요.",
                    "모델이 \"이 점이 나올 확률(밀도)\"을 직접 계산할 수 있어서 처음 보는 점에 대한 log 확률을 잴 수 있어요. 높을수록 진짜 점을 그럴듯하다고 보는 모델이에요. ⑤의 Flow와 비교해요."])
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
with torch.no_grad():
    m_, _ = ns["enc"](ns["Z"]).chunk(2, dim=1)
m_ = m_.numpy()
axs[0].scatter(m_[lab == 0, 0], m_[lab == 0, 1], s=10, color=C["purple"], label="무리 0")
axs[0].scatter(m_[lab == 1, 0], m_[lab == 1, 1], s=10, color=C["pink"], label="무리 1")
th = np.linspace(0, 2 * np.pi, 100)
for r_ in (1, 2):
    axs[0].plot(r_ * np.cos(th), r_ * np.sin(th), color=C["gray"], lw=.8, ls="--")
axs[0].set(title="encoder가 train 점을 보낸 잠재 공간 (평균)", xlim=(-3, 3), ylim=(-3, 3))
axs[0].set_aspect("equal")
axs[0].legend(fontsize=8)
sc(axs[1], ns["samples"]["VAE"].numpy(), "잠재 공간에서 무작위로 뽑아 decoder로 만든 점")
outs["f_vae"] = chart("VAE의 잠재 공간과 만든 점", fig_save(fig, "vae"), "점선 원: 표준정규분포의 1, 2 표준편차",
                      ["왼쪽: 두 무리가 잠재 공간에서도 대체로 갈라져 있고, 전체가 원점 둘레의 원 안에 모였어요. KL 항이 잠재 공간을 표준정규분포 쪽으로 당긴 결과예요. 무리 번호는 학습에 쓰지 않았어요.",
                       "오른쪽: 표준정규분포에서 아무 점이나 뽑아 decoder에 넣었더니, 두 띠가 있는 자리를 대략 덮는 점들이 나왔어요. 다만 경계가 흐릿해서 띠 사이 빈 곳에도 점이 꽤 있어요. 잠재 공간을 미리 정해 둔 분포로 맞춰 놨기 때문에 \"어디서 뽑아야 하나\"를 고민하지 않아도 되는 게 장점이에요."])
s_rows = []
for s2 in (0.05, 0.5):
    n_ = ns if s2 == 0.05 else run(3, [("LATENT, SIGMA2 = 2, 0.05", f"LATENT, SIGMA2 = 2, {s2}")])
    s_rows.append([f"<b>{s2}</b> (지금)" if s2 == 0.05 else s2, n_["scores"]["VAE"]["precision"], n_["scores"]["VAE"]["recall"]])
outs["t3"] = S.table("되살리기 오차의 분산 SIGMA2를 키우면", ["SIGMA2", "정밀도", "재현율"], s_rows,
                     ["SIGMA2가 크면 \"되살린 점이 원래 점과 좀 달라도 괜찮다\"는 뜻이라 되살리기 항의 무게가 줄고 KL 항이 상대적으로 세져요. decoder가 뭉개진 평균만 내고, 만든 점에 더하는 잡음도 커져서 띠 바깥에 점이 많아져요(정밀도가 떨어져요).",
                      "그림에서 VAE가 흐릿하다는 말이 이거예요. 되살리기를 제곱 오차(정규분포)로 재면 \"여러 그럴듯한 답의 평균\"을 내기 쉬워요."])
outs["s3"] = S.box("③ VAE", [f"마지막 손실(−ELBO) → {ns['vae_loss'].item():.3f}", (f"정밀도 {sb['VAE']['precision']}, 재현율 {sb['VAE']['recall']}", 1)],
                   ["정규분포 하나보다 정밀도가 올라갔어요. 띠 사이 빈 곳에 만드는 점이 줄었어요."])
gl = ns["gan_log"]
fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
a.plot([g["step"] for g in gl], [g["d_loss"] for g in gl], color=C["pink"], lw=2, label="가려내는 쪽 (D)")
a.plot([g["step"] for g in gl], [g["g_loss"] for g in gl], color=C["purple"], lw=2, label="만드는 쪽 (G)")
a.axhline(2 * np.log(2), color=C["pink"], ls=":", lw=1)
a.axhline(np.log(2), color=C["purple"], ls=":", lw=1)
a.text(2400, 2 * np.log(2) + .05, "D가 반반으로 찍을 때 2 log 2", fontsize=9, color=C["pink"])
a.text(2400, np.log(2) - .12, "log 2", fontsize=9, color=C["purple"])
a.set(xlabel="학습 걸음", ylabel="손실", title="GAN의 손실은 줄어들지 않고 맞서요")
a.legend(fontsize=9)
sc(b, ns["samples"]["GAN"].numpy(), "GAN이 만든 점")
outs["f_gan"] = chart("GAN의 학습 기록과 만든 점", fig_save(fig, "gan"), "", [
    "보통 학습과 달리 두 손실이 0으로 내려가지 않아요. 한쪽이 좋아지면 다른 쪽이 나빠지는 시합이라, D가 \"반반으로 찍는\" 상태(2 log 2 ≈ 1.39) 근처에서 오르내려요. 그래서 손실만 보고는 GAN이 잘 배웠는지 알 수 없어요.",
    "오른쪽: 점이 띠 위의 가는 선 몇 줄로 몰렸어요. 그 선은 진짜 점 사이에 있어서 정밀도는 높지만, 진짜 점은 띠 폭 전체에 퍼져 있어서 바깥쪽 점들 근처에는 만든 점이 없어요(재현율 낮음). 진짜 데이터의 일부만 골라 흉내 내는 쪽으로 쏠리는 걸 모드 붕괴라고 해요."])
outs["s4"] = S.box("④ GAN", [f"마지막 D 손실 {gl[-1]['d_loss']:.3f}, G 손실 {gl[-1]['g_loss']:.3f}", (f"정밀도 {sb['GAN']['precision']}, 재현율 {sb['GAN']['recall']}", 1, "bad")],
                   ["정밀도는 다섯 모델 중 가장 높지만 재현율이 가장 낮아요. 가려내는 쪽을 속이는 데는 \"확실한 진짜처럼 보이는 곳\"에만 만드는 게 안전해서예요."])
gx, gy = np.meshgrid(np.linspace(*LIM, 200), np.linspace(*LIM, 200))
G = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
with torch.no_grad():
    dens_flow = ns["flow_loglik"](G).exp().numpy().reshape(gx.shape)
    dens_gauss = ns["gauss"].log_prob(G).exp().numpy().reshape(gx.shape)
    train_ll_flow = ns["flow_loglik"](ns["Z"]).mean().item()
    train_ll_gauss = ns["gauss"].log_prob(ns["Z"]).mean().item()
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, d_, ttl in [(axs[0], dens_gauss, "② 정규분포의 밀도"), (axs[1], dens_flow, "⑤ Flow가 배운 밀도")]:
    ax.imshow(d_, origin="lower", extent=[*LIM, *LIM], cmap="magma")
    ax.scatter(Z[:, 0], Z[:, 1], s=3, color="white", alpha=.5)
    ax.set(title=ttl, xticks=[], yticks=[])
    ax.grid(False)
outs["f_flow"] = chart("모델이 계산한 \"이 자리에 점이 있을 확률\" (밝을수록 높음, 흰 점: train)", fig_save(fig, "flow_density"), "",
                       ["Flow는 확률을 정확한 식으로 계산할 수 있어서, 화면의 모든 자리에 대해 밀도를 그릴 수 있어요. 정규분포는 타원 하나인데, Flow는 띠를 따라 들쭉날쭉한 좁은 줄무늬로 밝아요.",
                        "그런데 밝은 곳이 train 점 바로 위에 좁게 몰려 있어요. 아래 ⑤의 출력처럼 이게 처음 보는 점에서 손해가 돼요."])
outs["s5"] = S.box("⑤ Normalizing Flow (결합층 8개)", [(f"정밀도 {sb['Flow']['precision']}, 재현율 {sb['Flow']['recall']}", 1),
                                                       (f"평균 log 확률 → train {train_ll_flow:.3f}, 처음 보는 점 {sb['Flow']['test_loglik']}", 2, "bad"),
                                                       f"(참고) 정규분포 → train {train_ll_gauss:.3f}, 처음 보는 점 {sb['정규분포']['test_loglik']}"],
                   ["만든 점의 모양은 정규분포보다 훨씬 진짜 같아요(정밀도 0.83 → 0.93).",
                    "그런데 처음 보는 점의 log 확률은 정규분포보다 낮아요. train에서는 −1.8로 훨씬 높은데요. train 240개 바로 위에만 확률을 몰아 준 거예요. 가능도도 과적합해요. 확률 모델을 고를 때도 train이 아니라 처음 보는 데이터로 재는 이유예요."])
# ⑥ 확산: 앞으로(잡음 섞기)와 거꾸로(걷어내기) 스냅숏
ab = ns["alpha_bar"]
torch.manual_seed(5)
fwd = {t: (ab[t].sqrt() * ns["Z"] + (1 - ab[t]).sqrt() * torch.randn_like(ns["Z"])).numpy() for t in (0, 20, 50, 99)}
snap = {}
with torch.no_grad():
    torch.manual_seed(1)
    Nn, T_ = ns["N"], ns["T"]
    betas = ns["betas"]
    x = torch.randn(Nn, 2)
    for t in reversed(range(T_)):
        e = ns["eps_net"](torch.cat([x, torch.full((Nn, 1), t / T_)], 1))
        x = (x - betas[t] / (1 - ab[t]).sqrt() * e) / (1 - betas[t]).sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
        if t in (99, 50, 20, 0):
            snap[t] = x.clone().numpy()
same_as_page = bool(np.allclose(snap[0], ns["samples"]["Diffusion"].numpy(), atol=1e-5))
fig, axs = plt.subplots(2, 4, figsize=(13, 6.6))
for j, t in enumerate((0, 20, 50, 99)):
    sc(axs[0, j], fwd[t], f"섞기 t={t}, 남은 몫 {ab[t].sqrt():.2f}", col=C["purple"], s=6, real=False)
for j, t in enumerate((99, 50, 20, 0)):
    sc(axs[1, j], snap[t], f"걷어내기 t={t}", col=C["mint"], s=4, real=False)
P_DIFF = fig_save(fig, "diffusion_steps")
outs["f_diff"] = chart("Diffusion: 위는 train 점에 잡음을 섞는 쪽, 아래는 순수한 잡음에서 걷어내는 쪽", P_DIFF, "",
                       ["위: 걸음이 갈수록 띠 모양이 흐려져서, t=99에서는 원래 점이 거의 남지 않은 정규분포가 돼요. 이 방향은 학습할 게 없는 정해진 계산이에요.",
                        "아래: 신경망이 \"섞인 잡음\"을 맞혀서 한 걸음씩 걷어내요. t=50까지는 둥근 덩어리이고, t=20쯤부터 띠 두 개의 모양이 잡히고, 마지막 걸음들에서 더 모여요." + (" 아래 맨 오른쪽은 ⑥의 출력과 같은 점이에요." if same_as_page else "")])
outs["s6"] = S.box("⑥ Diffusion (T=100)", [f"원래 점이 남은 몫 √ᾱ → t=0: {ab[0].sqrt():.3f}, t=99: {ab[-1].sqrt():.3f}", f"마지막 학습 손실(잡음 맞히기 MSE) → {ns['diff_loss'].item():.3f}",
                                             (f"정밀도 {sb['Diffusion']['precision']}, 재현율 {sb['Diffusion']['recall']}", 1)],
                   ["정밀도와 재현율이 둘 다 높아요. GAN처럼 한쪽으로 쏠리지 않고, VAE보다 덜 흐려요. 대신 점 하나를 만들려고 신경망을 100번 돌렸어요. 다른 모델은 한 번이에요."])
d_rows = []
for T_try, beta_end in ((100, 0.1), (20, 0.4)):
    if T_try == 100:
        n_ = ns
    else:
        n_ = run(6, [("T = 100", f"T = {T_try}"), ("betas = torch.linspace(1e-4, 0.1, T)", f"betas = torch.linspace(1e-4, {beta_end}, T)")])
    d_rows.append([f"<b>{T_try}</b> (지금)" if T_try == 100 else T_try, beta_end, f3(n_["alpha_bar"][-1].sqrt()), n_["scores"]["Diffusion"]["precision"], n_["scores"]["Diffusion"]["recall"]])
outs["t6"] = S.table("걸음 수 T를 100에서 20으로 줄이면 (끝에 거의 잡음이 되게 β를 키워서)", ["T", "마지막 β", "끝에 남은 몫 √ᾱ", "정밀도", "재현율"], d_rows,
                     ["걸음을 줄이면 한 걸음에 걷어낼 잡음이 커져서 신경망이 맞혀야 할 일이 어려워져요. 만드는 시간은 5분의 1이 됐고, 이 쉬운 2차원 데이터에서는 정밀도만 조금(0.911 → 0.894) 떨어졌어요. 그림처럼 복잡한 데이터에서는 걸음 수가 품질에 훨씬 크게 영향을 줘요.",
                      "실제로는 학습은 많은 걸음으로 하고, 만들 때만 걸음을 건너뛰는 방법(DDIM 같은)으로 속도와 품질을 맞춰요."])
# ⑦ 비교
fig, axs = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, (nm, col) in zip(axs, [("정규분포", C["gray"]), ("VAE", C["orange"]), ("GAN", C["pink"]), ("Flow", C["blue"]), ("Diffusion", C["mint"])]):
    sc(ax, ns["samples"][nm].numpy(), f"{nm} (정밀도 {sb[nm]['precision']}, 재현율 {sb[nm]['recall']})", col=col, s=4)
outs["f_all"] = chart("다섯 모델이 만든 점 1,000개 (회색: 처음 보는 진짜 점)", fig_save(fig, "all_samples"), "",
                      ["정규분포는 띠 사이 빈 곳까지, GAN은 띠의 일부에만, Diffusion은 띠 전체를 고르게 덮어요."])
bd = ns["board"]
outs["s7"] = S.table("⑦ 같은 기준으로 비교 (처음 보는 진짜 점 160개 기준)", ["모델", "정밀도", "재현율", "평균 log 확률", "점 하나 만들 때 신경망 실행"],
                     [[nm, f3(bd.loc[nm, "precision"]), f3(bd.loc[nm, "recall"]), "-" if np.isnan(bd.loc[nm, "test_loglik"]) else f3(bd.loc[nm, "test_loglik"]), {"정규분포": "0번", "VAE": "1번", "GAN": "1번", "Flow": "8층 한 번", "Diffusion": "100번"}[nm]] for nm in bd.index]
                     + [["(참고) 진짜 train 점", f3(rr["precision"]), f3(rr["recall"]), "-", "-"]],
                     ["\"가장 좋은 생성 모델\"은 무엇을 원하느냐에 달려요. 그럴듯함(정밀도)만 보면 GAN, 고르게 덮기(재현율)까지 보면 Diffusion과 VAE, 확률을 정확히 계산해야 하면 Flow예요.",
                      "확률을 계산할 수 있는 건 정규분포와 Flow뿐이에요. VAE는 하한(ELBO)만, GAN과 Diffusion은 확률을 직접 주지 않아요. 그래서 그림 생성 모델은 FID처럼 만든 그림과 진짜 그림의 특징 분포를 비교하는 지표로 재요."])
flow = [[1, "준비", "크기 맞추기", "정밀도, 재현율"], [2, "정규분포", "타원 하나", "기준선"], [3, "VAE", "줄였다 되살리기", "ELBO"], [4, "GAN", "만들기 vs 가려내기", "모드 붕괴"],
        [5, "Flow", "되돌릴 수 있는 변환", "정확한 확률"], [6, "Diffusion", "잡음 걷어내기", "100걸음"], [7, "비교", "같은 기준", "목적에 따라"]]
S.save(ROOT / "topics" / "model" / "generative-models-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

why = f"""<p>지금까지의 모델은 입력을 보고 <b>답을 맞혔어요</b>(분류, 회귀). <b>생성 모델</b>은 반대로, 데이터가 어떻게 생겼는지를 배워서 <b>그 데이터를 닮은 새 데이터를 만들어요</b>. 그림을 그리는 모델, 목소리를 만드는 모델, 글을 쓰는 LLM이 모두 생성 모델이에요.</p>
<p>그림 대신 점으로 해 보면 원리가 잘 보여요. 아래 점들을 보고, 이런 모양의 점을 새로 찍는 게 목표예요.</p>{K.out("data")}
<p>좋은 생성 모델이 갖췄으면 하는 것은 여럿이고, 모델마다 잘하는 게 달라요.</p>
{K.table(["바라는 것", "뜻", "이 쪽에서 재는 방법"], [["그럴듯함 (품질)", "만든 것이 진짜 같은가", "정밀도: 만든 점 중 진짜 점 근처에 떨어진 비율"], ["다양함", "진짜 데이터의 여러 모습을 골고루 만드나", "재현율: 진짜 점 중 만든 점 근처에 있는 비율"],
                                                     ["확률 계산", "\"이 데이터가 나올 확률\"을 숫자로 줄 수 있나", "처음 보는 진짜 점의 평균 log 확률"], ["속도", "하나 만드는 데 얼마나 걸리나", "신경망을 몇 번 돌리나"]])}
<p>정밀도와 재현율은 둘 다 있어야 해요. 진짜 점 하나를 그대로 1,000번 복사하면 정밀도는 높지만 재현율이 낮고, 화면 전체에 마구 찍으면 재현율은 높지만 정밀도가 낮아요.</p>"""
whole = """<p>같은 점 데이터를 다섯 가지 방법으로 배우고, 같은 기준으로 비교해요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "데이터와 재는 방법 준비", "train 점의 크기를 맞추고, 만든 점을 처음 보는 진짜 점과 비교하는 함수를 만들어요.",
           "<p>크기를 맞추는 이유는 신경망 학습이 안정되고(<a href=\"dl-tensor-training.html\">딥러닝 기초</a>), 마지막에 단순한 정규분포 N(0, 1)과 비교하기 쉬워서예요.</p>"
           "<p><b>정밀도와 재현율</b>은 <a href=\"ml-regression-classification.html\">회귀와 분류</a>의 precision, recall과 이름이 같지만, 여기서는 \"근처에 있나\"로 재요. 진짜 점마다 \"자기와 가장 가까운 진짜 점 3번째까지의 거리\"를 반지름으로 작은 원을 그리고, 만든 점이 그 원들 중 하나에라도 들어가면 \"진짜 근처\"로 쳐요. 재현율은 역할을 바꿔서 재요.</p>"
           + K.code_table([("(X - mu) / sd", "train 평균과 표준편차로 크기 맞추기. 처음 보는 점도 같은 값으로"), ("torch.cdist(A, B)", "A의 점마다 B의 모든 점까지 거리 표"),
                           (".kthvalue(k + 1, dim=1)", "줄마다 k+1번째로 작은 값. 첫째는 자기 자신(거리 0)이라 하나 더"), (".any(1)", "줄마다 하나라도 True인가")]), "s1"),
    K.step(2, "기준선: 정규분포 하나", "train 점의 평균과 공분산만 맞춘 정규분포(타원) 하나로 점을 만들어요.",
           "<p>가장 단순한 생성 모델이에요. 평균(가운데)과 공분산(어느 방향으로 얼마나 퍼졌나)만 배워요. 확률을 정확한 식으로 계산할 수 있고, 점도 바로 뽑을 수 있어요. 하지만 모양은 언제나 타원 하나예요.</p>"
           + K.code_table([("MultivariateNormal(mean, cov)", "여러 칸짜리 정규분포"), ("torch.cov(Z.T)", "열 사이 공분산 (열이 줄로 오게 뒤집어서)"), (".sample((N,))", "N개 뽑기"), (".log_prob(x)", "그 점이 나올 확률(밀도)의 log")]), "s2"),
    K.step(3, "VAE: 줄였다가 되살리기", "점을 작은 잠재 변수로 줄이는 encoder와, 잠재 변수에서 점을 되살리는 decoder를 함께 배워요. 잠재 변수는 표준정규분포를 따르도록 당겨요.",
           "<p><a href=\"sequence-models.html\">순서 모델</a>의 Autoencoder는 줄였다가 되살리기만 배워서, 줄인 공간(잠재 공간)의 어디가 비어 있는지 몰라요. 아무 데서나 뽑아 되살리면 엉뚱한 게 나와요. <b>VAE</b>(Variational Autoencoder)는 잠재 공간이 표준정규분포 N(0, I)처럼 생기도록 강제해요. I는 단위행렬로, 칸마다 서로 상관없는 평균 0, 분산 1의 정규분포라는 뜻이에요. 그러면 N(0, I)에서 아무 점이나 뽑아 decoder에 넣으면 새 데이터가 돼요.</p>"
           "<p>encoder는 점 하나를 잠재 공간의 한 점이 아니라 <b>작은 분포</b>(평균과 분산)로 보내요. 학습 손실은 두 항이에요. ① <b>되살리기</b>: 그 분포에서 뽑은 잠재 변수로 원래 점을 잘 되살리나(제곱 오차). ② <b>KL 항</b>: encoder가 만든 분포가 N(0, I)에서 얼마나 먼가(KL 발산, 두 분포의 차이를 재는 값). 둘을 더한 것의 반대가 <b>ELBO</b>예요. ELBO는 \"이 데이터가 나올 log 확률\"의 하한이라, ELBO를 키우는 건 진짜 확률을 키우는 대신 그 아래쪽 바닥을 밀어 올리는 셈이에요. 진짜 확률은 모든 잠재 변수에 대해 더해야 해서 직접 계산할 수 없거든요.</p>"
           "<p><b>재파라미터화</b>: \"분포에서 뽑기\"는 미분이 안 돼서 기울기가 encoder까지 못 가요. 그래서 z = 평균 + 표준편차 × ε (ε는 N(0, 1)에서 뽑은 값)으로 바꿔 써요. 무작위는 ε에만 있고, 평균과 표준편차로는 기울기가 흘러요.</p>"
           + K.out("f_vae")
           + K.code_table([("enc(Z).chunk(2, dim=1)", "encoder 출력을 평균, log 분산 두 덩어리로"), ("m + torch.exp(0.5 * logv) * torch.randn_like(m)", "재파라미터화"),
                           ("0.5 * (m**2 + logv.exp() - 1 - logv).sum(1)", "N(m, σ²)과 N(0, 1) 사이 KL 발산의 식"), ("dec(torch.randn(N, LATENT))", "표준정규분포에서 뽑아 되살리기 = 새 점 만들기")])
           + K.try_("SIGMA2를 0.05에서 0.5로 키우면?", "t3"), "s3"),
    K.step(4, "GAN: 만드는 쪽과 가려내는 쪽의 시합", "무작위 잡음을 점으로 바꾸는 G와, 진짜 점과 G가 만든 점을 가려내는 D를 번갈아 학습해요.",
           "<p><b>GAN</b>(Generative Adversarial Network)은 확률을 계산하지 않아요. 대신 <b>가려내는 쪽 D</b>(discriminator)를 하나 더 둬요. D는 \"진짜면 1, 가짜면 0\"을 맞히는 분류기이고, <b>만드는 쪽 G</b>(generator)는 D가 자기 점을 1이라고 믿게 만들려고 해요. 위조지폐범과 감별사의 시합이에요. D가 둘을 전혀 못 가릴 만큼 G가 좋아지면 끝이에요.</p>"
           "<p>원래 식은 \"D는 이 손실을 키우고, G는 줄이고\"인 한 식(미니맥스)이에요. 그런데 처음에 G가 서투르면 D가 너무 쉽게 이겨서 G의 기울기가 거의 0이 돼요. 그래서 G는 \"D를 속인 정도\"를 키우는 쪽으로 손실을 바꿔 써요(코드의 <code>g_loss</code>).</p>"
           "<p><b>어려운 점</b>: 진짜 데이터와 G의 분포가 거의 겹치지 않으면 D가 완벽하게 가려내서, G에게 \"어느 쪽으로 가면 나아지나\"라는 신호가 사라져요. 두 분포 사이 거리를 다르게 재서 이 문제를 줄인 게 WGAN(Wasserstein GAN)이에요. 그리고 G가 D를 속이기 쉬운 몇 가지 모습만 반복해서 만드는 <b>모드 붕괴</b>가 자주 생겨요.</p>"
           "<p>조건을 붙이면 쓸모가 커져요. 스케치를 사진으로(Pix2Pix), 짝 없는 두 종류 그림 사이를 바꾸기(CycleGAN), 거친 모양부터 세밀한 결까지 단계별로 조절하기(StyleGAN)가 GAN으로 유명해진 일들이에요.</p>"
           + K.out("f_gan")
           + K.code_table([("fake.detach()", "D를 학습할 때는 G까지 기울기가 가지 않게 끊기"), ("bce(D(Z), ones) + bce(D(fake.detach()), zeros)", "D: 진짜는 1, 가짜는 0으로"),
                           ("bce(D(fake), ones)", "G: D가 가짜를 1이라고 믿게"), ("betas=(0.5, 0.999)", "Adam의 모멘텀을 약하게. GAN에서 흔히 쓰는 설정")]), "s4"),
    K.step(5, "Normalizing Flow: 되돌릴 수 있는 변환", "단순한 정규분포의 점을 되돌릴 수 있는 변환으로 여러 번 구부려 데이터 모양을 만들어요. 그래서 확률을 정확하게 계산할 수 있어요.",
           "<p>N(0, I)에서 뽑은 z를 변환 f로 옮겨 x = f(z)를 만든다고 해요. f가 <b>되돌릴 수 있으면</b>(z = f⁻¹(x)), x가 나올 확률을 정확히 계산할 수 있어요. <code>p(x) = p(z) × (f⁻¹가 그 자리에서 부피를 몇 배로 바꾸나)</code>. 고무판을 늘이면 같은 양의 물감이 더 넓게 퍼져 옅어지는 것과 같아요. 이 \"부피 배율\"이 야코비 행렬식이에요(변수 바꾸기 공식).</p>"
           "<p>보통 신경망은 되돌릴 수 없고 행렬식 계산도 비싸요. 그래서 특별한 층을 써요. <b>결합층</b>(coupling layer)은 칸을 둘로 나눠, 한 칸은 그대로 두고 그 칸을 신경망에 넣어 나온 값으로 다른 칸을 늘이고(s) 옮겨요(t). 그대로 둔 칸으로 s, t를 다시 계산할 수 있으니 거꾸로도 쉽게 가고, 부피 배율도 늘인 정도(e^s)의 곱이라 바로 나와요. 층마다 그대로 두는 칸을 번갈아 바꿔서 두 칸 모두 바뀌게 해요.</p>"
           "<p>대신 조건이 까다로워요. 되돌릴 수 있어야 하니 입력과 출력의 칸 수가 같아야 하고(VAE처럼 줄일 수 없어요), 층 모양도 제한돼요. 그림처럼 칸이 많은 데이터에서는 해상도별로 나눠 처리해 계산을 줄여요.</p>"
           + K.out("f_flow")
           + K.code_table([("class Coupling(nn.Module)", "결합층 하나. <code>forward</code>는 데이터 → 정규분포 쪽, <code>inverse</code>는 거꾸로"), ("torch.tanh(s)", "늘이는 정도를 e^±1 안으로 제한해 학습을 안정시키기"),
                           ("base.log_prob(x) + total", "정규분포에서의 log 확률 + 층마다 log 부피 배율의 합 = 데이터의 정확한 log 확률"), ("for f in reversed(flows): y = f.inverse(y)", "점 만들기: 정규분포에서 뽑아 거꾸로 통과")]), "s5"),
    K.step(6, "Diffusion: 잡음을 섞었다가 거꾸로 걷어내기", "점에 잡음을 100걸음에 걸쳐 섞어 완전한 잡음으로 만드는 과정을 정해 두고, 신경망이 \"섞인 잡음\"을 맞히게 배워요. 만들 때는 순수한 잡음에서 한 걸음씩 걷어내요.",
           "<p><b>앞으로(정해진 계산)</b>: 걸음마다 원래 점을 조금 줄이고 정규분포 잡음을 조금 더해요. t번째 걸음에 섞는 잡음의 양을 β_t라고 해요(여기서는 0.0001에서 0.1까지 조금씩 늘려요). 이걸 100번 하면 원래 모양은 사라지고 N(0, I)만 남아요. 여러 걸음을 한 번에 계산하는 식이 있어서, t 걸음 뒤의 점을 바로 만들 수 있어요. <code>x_t = √ᾱ_t × x_0 + √(1 − ᾱ_t) × ε</code>. ᾱ_t는 t 걸음 뒤 원래 점이 남은 몫이에요.</p>"
           "<p><b>거꾸로(배우는 것)</b>: 신경망은 x_t와 t를 보고 \"여기에 섞인 잡음 ε가 무엇이었나\"를 맞혀요. 손실은 그냥 제곱 오차예요. 이 손실도 사실은 VAE처럼 ELBO에서 나온 것인데, 정리하면 \"잡음 맞히기\"라는 단순한 모양이 돼요. 맞힌 잡음을 조금 빼고, 마지막 걸음 전까지는 조금 흔들면서 100걸음 거꾸로 가면 새 점이 돼요.</p>"
           "<p>그림, 영상, 소리 생성 모델 대부분이 이 방식이에요. 문장을 조건으로 넣어 \"이 설명에 맞는 그림\"을 만들고, 조건 있는 예측과 없는 예측의 차이를 키워 조건을 더 강하게 따르게 해요(classifier-free guidance). 큰 그림은 VAE로 작게 줄인 잠재 공간에서 확산을 돌려 계산을 줄여요(Stable Diffusion 같은 잠재 확산).</p>"
           + K.out("f_diff")
           + K.code_table([("torch.cumprod(1 - betas, dim=0)", "걸음마다 남는 몫을 곱해 ᾱ_t"), ("torch.randint(0, T, (n,))", "학습할 때 점마다 무작위 걸음 t"),
                           ("eps_net(torch.cat([x_t, t / T], 1))", "점과 시점을 함께 넣어 잡음 맞히기"), ("(x - β / √(1 − ᾱ) × e) / √(1 − β)", "맞힌 잡음을 걷어내고 크기를 되돌리는 한 걸음")])
           + K.try_("걸음 수 T를 20으로 줄이면?", "t6"), "s6"),
    K.step(7, "같은 기준으로 비교하기", "다섯 모델이 만든 점을 같은 정밀도, 재현율로 재고, 확률을 계산할 수 있는 모델은 처음 보는 점의 log 확률도 봐요.",
           K.out("f_all") + K.code_table([("quality(Z, R=Z_test)", "진짜 train 점을 \"만든 점\"으로 넣어, 완벽한 모델의 점수가 어느 정도인지 보기"), ("pd.DataFrame(scores).T", "모델마다 한 줄인 표")]), "s7"),
]
practice = [("그림 생성 모델을 고를 때", "지금은 대부분 Diffusion이에요. 실시간처럼 아주 빨라야 하면 GAN이나 걸음 수를 줄인 Diffusion을 써요."),
            ("생성 모델의 평가", "그림은 FID(만든 그림과 진짜 그림을 미리 학습된 CNN의 특징으로 바꿔 두 분포의 거리를 재기)를 가장 많이 써요. 하나의 숫자로 품질과 다양성을 다 말할 수는 없어서, 정밀도와 재현율을 따로 보거나 사람이 직접 봐요."),
            ("가능도가 높다고 좋은 그림인가", "꼭 그렇지는 않아요. log 확률은 사람 눈에 안 보이는 세세한 픽셀 차이에 크게 좌우돼요. 반대로 그럴듯한 그림을 만드는 GAN은 확률을 계산조차 못 해요."),
            ("GAN이 학습되지 않는다", "D가 너무 강하면 G에게 신호가 안 가요. 학습률을 낮추고, D를 덜 자주 학습하거나, WGAN처럼 손실을 바꾸거나, 정규화(spectral norm)를 넣어요."),
            ("VAE의 잠재 공간 활용", "잠재 공간에서 두 점 사이를 이으면 두 데이터 사이를 자연스럽게 바꾸는 중간 모습이 나와요. 잠재 칸 하나하나가 \"크기\", \"기울기\"처럼 한 가지 성질만 맡게 하는 연구도 있어요."),
            ("이상 탐지에 쓰기", "정상 데이터만으로 확률 모델을 배우면, 확률이 아주 낮은 데이터를 이상으로 볼 수 있어요. 다만 ⑤처럼 가능도가 과적합하거나, 단순한 이상 데이터에 오히려 높은 확률을 주는 경우도 있어 검증이 필요해요.")]
libs = [("①②", "<code>torch.cdist</code>, <code>kthvalue</code>, <code>MultivariateNormal</code>", "거리와 정규분포 (torch)"), ("③④", "<code>torch.randn_like</code>, <code>.detach()</code>, <code>nn.BCEWithLogitsLoss</code>", "VAE, GAN"),
        ("⑤", "<code>nn.Module</code>, <code>nn.ModuleList</code>, <code>torch.tanh</code>", "Flow 층"), ("⑥", "<code>torch.cumprod</code>, <code>torch.randint</code>", "Diffusion")]
K.build(ROOT / "topics" / "model" / "generative-models.html", title="생성 모델", h1="생성 모델: 데이터를 닮은 새 데이터 만들기", track="모델", crumb="모델 &gt; 생성 모델",
        one_line="<b>생성 모델</b>은 데이터가 어떻게 생겼는지 배워서 새 데이터를 만들어요. 잠재 공간으로 줄였다 되살리거나(<b>VAE</b>), 가려내는 쪽과 시합하거나(<b>GAN</b>), 되돌릴 수 있는 변환을 겹치거나(<b>Flow</b>), 잡음을 걷어내요(<b>Diffusion</b>).",
        why=why, whole=whole, steps=steps, code_file="examples/gen/gen_points.py", practice=practice, libs=libs, libs_note="<code>mlp</code>, <code>quality</code>, <code>flow_loglik</code>는 이 파일 안에서 정의한 함수, <code>Coupling</code>은 클래스예요.",
        next_rows=[("<a href=\"sequence-models.html\">순서 모델</a>", "Autoencoder: VAE의 출발점"), ("<a href=\"dl-training-stability.html\">학습을 안정시키기</a>", "Adam, 학습률"), ("<a href=\"llm-principles.html\">LLM 원리</a>", "글을 만드는 생성 모델")],
        sources=["함께 정리한 자료: Simon J. D. Prince, <i>Understanding Deep Learning</i> (MIT Press, 2023) 14장(비지도 학습과 생성 모델 평가), 15장(GAN), 16장(Normalizing Flow), 17장(VAE), 18장(Diffusion). 부트캠프 수업에서 따로 다루지 않은 주제라, 딥러닝 기초의 점 예제로 직접 실험했어요",
                 "데이터: <code>examples/gen/data/points.csv</code> (딥러닝 기초의 <code>examples/dl/data/points.csv</code>와 같은 파일)",
                 "결과 값과 그림: <code>tools/pages/gen_models.py</code>가 <code>examples/gen/gen_points.py</code>를 단계별로 실제 실행한 결과 (seed 0, CPU)",
                 "공식 문서: <a href=\"https://docs.pytorch.org/docs/stable/distributions.html\" target=\"_blank\">torch.distributions</a>, <a href=\"https://arxiv.org/abs/2006.11239\" target=\"_blank\">DDPM 논문</a>"])
print("ok")
