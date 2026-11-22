# 작은 행렬 게임으로 게임 이론의 해(균형)와, 반복하며 그 해에 다가가는 학습 방법을 직접 계산하는 전체 코드. 노트의 단계 번호가 이 파일의 번호 주석과 1:1로 맞아요.
# 게임: 죄수의 딜레마, 조정 게임, 치킨, 가위바위보, 동전 맞히기. 받을 파일 없음. CPU로 몇 초.
# 줄 끝의 "#@N" 표시는 노트가 "이 줄은 N단계에서 추가됨"을 알기 위한 표시예요(실행과 무관).

import itertools                                                                #@2
import numpy as np                                                              #@1
from scipy.optimize import linprog                                              #@3

# ① 게임을 표로: 행렬 게임
GAMES = {   # (행 플레이어의 보상 표, 열 플레이어의 보상 표), 행동 이름
    "죄수의 딜레마": (np.array([[-1, -5], [0, -3]]), np.array([[-1, 0], [-5, -3]]), ["협력", "배신"]),
    "조정 게임": (np.array([[10, 0], [0, 10]]), np.array([[10, 0], [0, 10]]), ["A", "B"]),
    "치킨": (np.array([[0, 7], [2, 6]]), np.array([[0, 2], [7, 6]]), ["직진", "피함"]),
    "가위바위보": (np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]), np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]), ["가위", "바위", "보"]),
}
kind = {}
for name, (R1, R2, _) in GAMES.items():                         # 보상 관계로 게임 나누기
    if np.array_equal(R1, R2):
        kind[name] = "공통 보상 (완전 협력)"
    elif np.array_equal(R1, -R2):
        kind[name] = "제로섬 (완전 경쟁)"
    else:
        kind[name] = "일반합 (섞임)"

# ② 최적 응답과 순수 내시 균형
def pure_nash(R1, R2):                                          # 서로 상대에게 최적 응답인 칸 찾기
    found = []
    for a, b in itertools.product(range(R1.shape[0]), range(R1.shape[1])):
        if R1[a, b] == R1[:, b].max() and R2[a, b] == R2[a, :].max():
            found.append((a, b))
    return found

pure = {name: pure_nash(R1, R2) for name, (R1, R2, _) in GAMES.items()}

# ③ 섞어서 고르기: 혼합 균형과 미니맥스(선형 계획)
def mixed_2x2(R1, R2):                                          # 상대가 두 행동을 똑같이 좋게 느끼게 하는 확률
    p = (R2[1, 1] - R2[1, 0]) / (R2[0, 0] - R2[0, 1] - R2[1, 0] + R2[1, 1])   # 행 플레이어가 첫 행동을 고를 확률
    q = (R1[1, 1] - R1[0, 1]) / (R1[0, 0] - R1[0, 1] - R1[1, 0] + R1[1, 1])   # 열 플레이어가 첫 행동을 고를 확률
    return p, q

def minimax(R):                                                 # 제로섬에서 행 플레이어가 보장할 수 있는 값과 정책
    n, m = R.shape                                              # 변수: 행동 확률 x (n개), 보장 값 v
    c = np.r_[np.zeros(n), -1]                                  # v를 최대로
    A_ub = np.c_[-R.T, np.ones(m)]                              # 상대의 어떤 행동에도 기대 보상 ≥ v
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(m), A_eq=[np.r_[np.ones(n), 0]], b_eq=[1], bounds=[(0, None)] * n + [(None, None)])
    return res.x[:n], res.x[-1]

mixed = {name: mixed_2x2(R1, R2) for name, (R1, R2, _) in GAMES.items() if R1.shape == (2, 2) and name != "죄수의 딜레마"}
rps_policy, rps_value = minimax(GAMES["가위바위보"][0])
TILTED = np.array([[0, -1, 2], [1, 0, -1], [-1, 1, 0]])         # 가위가 보를 이기면 2점: 균형이 어떻게 바뀌나
tilt_policy, tilt_value = minimax(TILTED)

# ④ 신호를 받아 함께 정하기: 상관 균형, 파레토, 사회 후생
def correlated(R1, R2, objective):                              # 결합 행동 확률 x로, 추천을 따르는 게 늘 최적이 되게
    cells = list(itertools.product(range(R1.shape[0]), range(R1.shape[1])))
    A_ub = []
    for i, R in enumerate((R1, R2)):
        for rec, dev in itertools.permutations(range(R.shape[i]), 2):   # rec를 추천받고 dev로 바꾸면 손해여야
            row = np.zeros(len(cells))
            for j, cell in enumerate(cells):
                if cell[i] == rec:
                    alt = list(cell)
                    alt[i] = dev
                    row[j] = R[tuple(alt)] - R[cell]
            A_ub.append(row)
    c = -np.array([objective(R1[cell], R2[cell]) for cell in cells])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(len(A_ub)), A_eq=[np.ones(len(cells))], b_eq=[1], bounds=[(0, None)] * len(cells))
    x = res.x
    return dict(zip(cells, x)), sum(x[j] * R1[cell] for j, cell in enumerate(cells)), sum(x[j] * R2[cell] for j, cell in enumerate(cells))

R1c, R2c, _ = GAMES["치킨"]
ce_welfare = correlated(R1c, R2c, lambda u1, u2: u1 + u2)       # 둘의 합이 가장 큰 상관 균형
p_c, q_c = mixed["치킨"]
chicken_points = {"내시 (직진, 피함)": (7, 2), "내시 (피함, 직진)": (2, 7),
                  "내시 (섞어서)": tuple(np.array([p_c, 1 - p_c]) @ R @ np.array([q_c, 1 - q_c]) for R in (R1c, R2c)),
                  "상관 균형 (합 최대)": (ce_welfare[1], ce_welfare[2]), "(피함, 피함)": (6, 6)}

# ⑤ 반복하며 배우기: 가상 플레이와 후회 매칭
rng = np.random.default_rng(0)
R_rps = GAMES["가위바위보"][0]
def fictitious_play(R1, R2, rounds=5000):                       # 상대가 지금까지 낸 비율을 믿고 그에 대한 최적 응답
    seen = [np.ones(R1.shape[0]), np.ones(R1.shape[1])]         # 처음엔 1번씩 본 셈
    freq = []
    for _ in range(rounds):
        a = int(np.argmax(R1 @ (seen[1] / seen[1].sum())))
        b = int(np.argmax((seen[0] / seen[0].sum()) @ R2))
        seen[0][a] += 1
        seen[1][b] += 1
        freq.append(seen[0] / seen[0].sum())
    return np.array(freq)

def regret_matching(R1, R2, rounds=20000):                      # "그때 그걸 냈으면" 아쉬움이 큰 행동을 더 자주
    regret = [np.zeros(R1.shape[0]), np.zeros(R1.shape[1])]
    count, joint, avg_regret = [np.zeros(R1.shape[0]), np.zeros(R1.shape[1])], np.zeros(R1.shape), []
    for t in range(1, rounds + 1):
        acts = []
        for r in regret:
            p = np.maximum(r, 0)
            p = p / p.sum() if p.sum() > 0 else np.ones(len(r)) / len(r)
            acts.append(rng.choice(len(r), p=p))
        a, b = acts
        regret[0] += R1[:, b] - R1[a, b]                        # 다른 행동이었으면 얼마나 더 받았나
        regret[1] += R2[a, :] - R2[a, b]
        count[0][a] += 1
        count[1][b] += 1
        joint[a, b] += 1
        avg_regret.append(max(np.maximum(regret[0], 0).max(), np.maximum(regret[1], 0).max()) / t)
    return count[0] / rounds, np.array(avg_regret), joint / rounds

fp_rps = fictitious_play(R_rps, -R_rps)
rm_policy, rm_regret, _ = regret_matching(R_rps, -R_rps)
_, rm_regret_ch, rm_joint_ch = regret_matching(R1c, R2c)        # 치킨에서는 결합 행동의 빈도를 봐요

# ⑥ 기울기로 배우기: IGA와 WoLF (동전 맞히기)
PENNIES = np.array([[1, -1], [-1, 1]])                          # 같으면 행 플레이어가 이기고, 다르면 열 플레이어가
def gradient_play(wolf, steps=20000, lr=0.001, p=0.2, q=0.6):   # p, q: 두 플레이어가 앞면을 낼 확률
    path = []
    for _ in range(steps):
        dp = q * 2 - (1 - q) * 2                                # 행 플레이어 기대 보상의 p에 대한 기울기 = 4q − 2
        dq = -(p * 2 - (1 - p) * 2)                             # 열 플레이어(보상은 반대)의 q에 대한 기울기
        step_p = step_q = lr
        if wolf:                                                # Win or Learn Fast: 지고 있으면 크게, 이기고 있으면 작게
            mine = (2 * p - 1) * (2 * q - 1)                    # 행 플레이어의 지금 기대 보상
            step_p = lr * (0.25 if mine > 0 else 1.0)           # 균형 정책(0.5)으로 받을 값은 0
            step_q = lr * (0.25 if -mine > 0 else 1.0)
        p, q = float(np.clip(p + step_p * dp, 0, 1)), float(np.clip(q + step_q * dq, 0, 1))
        path.append((p, q))
    return np.array(path)

iga_path, wolf_path = gradient_play(False), gradient_play(True)

# ⑦ 같은 게임을 되풀이하면: 반복 죄수의 딜레마 대결
R_pd = GAMES["죄수의 딜레마"][0]
STRATEGIES = {
    "늘 협력": lambda mine, theirs: 0,
    "늘 배신": lambda mine, theirs: 1,
    "맞대응": lambda mine, theirs: theirs[-1] if theirs else 0,          # 처음엔 협력, 그다음엔 상대가 지난번에 한 대로
    "원한": lambda mine, theirs: 1 if 1 in theirs else 0,                  # 한 번이라도 배신당하면 끝까지 배신
    "무작위": lambda mine, theirs: int(rng.integers(2)),
}
def match(f, g, rounds=100):                                    # 100번 둔 평균 보상 (f 쪽)
    hf, hg, total = [], [], 0
    for _ in range(rounds):
        a, b = f(hf, hg), g(hg, hf)
        hf.append(a)
        hg.append(b)
        total += R_pd[a, b]
    return total / rounds

league = {a: {b: match(STRATEGIES[a], STRATEGIES[b]) for b in STRATEGIES} for a in STRATEGIES}
