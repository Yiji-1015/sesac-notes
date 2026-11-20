"""토큰 분류와 질문 답변 쪽의 예제 데이터(가상의 업무 메시지)를 만든다.
  examples/tokens/data/ner.jsonl : 문장과 개체 위치. 종류는 PER(사람), ORG(회사), DATE(날짜), MONEY(금액)
  examples/tokens/data/qa.jsonl  : 메시지, 질문, 답의 글자 위치 (답이 없는 질문도 있음)
- NER: train, valid는 같은 문장 틀 12개, test는 처음 보는 틀 4개와 처음 보는 이름, en_test는 영어 문장(학습에 쓰지 않음).
- QA: 상황 넷(계약, 출장, 구매, 채용) × 메시지마다 질문 여럿. test는 처음 보는 값과 처음 보는 질문 말투.
실행: python tools/make_token_examples.py  (seed 고정, 다시 돌려도 같은 파일)
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "tokens" / "data"
rng = random.Random(0)

PER = ["김민준", "이서연", "박지훈", "최수아", "정우진", "강하은", "조현우", "윤지민", "장도윤", "임서준", "한예린", "오승현", "서지아", "신동현", "권나연", "황민재"]
PER_NEW = ["문가온", "배준서", "백하린", "노은찬", "하윤슬", "전도경"]
ORG = ["한빛상사", "누리소프트", "대한물산", "세움건설", "푸른제약", "미래전자", "하늘항공", "바른회계법인", "다온식품", "새길물류"]
ORG_NEW = ["별빛미디어", "소망은행", "그린모빌리티", "단비출판"]
WEEK = ["월요일", "화요일", "수요일", "목요일", "금요일"]


def date():
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([f"{m}월 {d}일", f"다음 주 {rng.choice(WEEK)}", f"2026년 {m}월 {d}일", f"{m}월 말", f"이번 주 {rng.choice(WEEK)}"])


def money():
    return rng.choice([f"{rng.randint(2, 950)}만 원", f"{rng.randint(1, 99)},{rng.choice(['000', '500'])}원", f"{rng.randint(1, 9)}억 원", f"{rng.randint(10, 990)}만 원"])


def josa(word, a, b):
    ch = word.strip()[-1]
    if "가" <= ch <= "힣":
        return a if (ord(ch) - 0xAC00) % 28 else b
    return a if ch in "013678" else b


def render(template, values):
    """'{PER}님이 {DATE}에 …'의 칸을 채우고, 칸마다 (시작, 끝, 종류)를 기록. {X|을/를}처럼 쓰면 받침에 맞는 조사."""
    text, spans, i = "", [], 0
    while i < len(template):
        if template[i] == "{":
            j = template.index("}", i)
            key = template[i + 1:j]
            if "|" in key:                                   # {|을/를} → 바로 앞 개체의 받침에 맞춘 조사
                a, b = key.split("|")[1].split("/")
                ch = text[-1]
                rieul = "가" <= ch <= "힣" and (ord(ch) - 0xAC00) % 28 == 8   # ㄹ 받침 뒤에는 '으로'가 아니라 '로'
                text += b if (a.startswith("으로") and rieul) else josa(text, a, b)
            else:
                kind = key.rstrip("0123456789")
                v = values[key]
                spans.append([len(text), len(text) + len(v), kind])
                text += v
            i = j + 1
        else:
            text += template[i]
            i += 1
    return text, spans


NER_T = [
    "{PER}님이 {DATE}에 {ORG} 방문 예정입니다.", "{ORG}에서 {MONEY} 견적서를 보내왔어요.", "{DATE}까지 {PER} 대리에게 보고서를 제출해 주세요.",
    "{PER} 팀장님, {ORG} 계약금 {MONEY} 입금 확인 부탁드립니다.", "회의는 {DATE} 오후 3시로 옮겼고, {PER}님도 참석합니다.", "이번 출장비는 {MONEY}{|이고/고} {DATE}에 정산됩니다.",
    "{ORG} 담당자 {PER}님과 통화했어요.", "{PER}님이 {ORG}에 {MONEY}{|을/를} 송금했습니다.", "{DATE} 기준으로 {ORG} 미수금이 {MONEY} 남았어요.",
    "다음 달부터 {PER}님이 {ORG} 프로젝트를 맡습니다.", "{ORG} 납품 일정이 {DATE}{|으로/로} 확정됐어요.", "{PER} 과장님 휴가는 {DATE}부터입니다.",
]
NER_T_NEW = ["{ORG} 쪽에서 {DATE}에 미팅 가능하냐고 물어봤어요.", "{MONEY} 청구 건은 {PER}님이 승인해야 합니다.", "어제 {PER}님이 보낸 메일 보셨어요? {ORG} 건이에요.", "{DATE}에 {MONEY} 결제 예정, 담당 {PER}"]
EN_T = ["{PER} will visit {ORG} on {DATE}.", "{ORG} sent us a quote for {MONEY}.", "Please send the report to {PER} by {DATE}.", "{PER} transferred {MONEY} to {ORG}.",
        "The meeting with {ORG} has moved to {DATE}.", "{PER} from {ORG} called this morning."]
EN = {"PER": ["Kim Minjun", "Lee Seoyeon", "James Carter", "Emily Stone", "Daniel Park", "Sophia Lee"], "ORG": ["Hanbit Trading", "Nuri Soft", "Blue Pharma", "Northwind Logistics"],
      "DATE": ["March 5", "next Tuesday", "November 20", "the end of May", "2026-11-20"], "MONEY": ["$1,200", "3 million won", "USD 450", "$75,000"]}


def ner_rows(templates, per, org, n_each, split, en=False):
    rows = []
    for t in templates:
        for _ in range(n_each):
            vals = {k: rng.choice(EN[k]) for k in EN} if en else {"PER": rng.choice(per), "ORG": rng.choice(org), "DATE": date(), "MONEY": money()}
            text, spans = render(t, vals)
            rows.append({"text": text, "entities": spans, "split": split, "template": t})
    return rows


ner = ner_rows(NER_T, PER, ORG, 20, "train") + ner_rows(NER_T, PER, ORG, 4, "valid") + ner_rows(NER_T_NEW, PER_NEW, ORG_NEW, 10, "test") + ner_rows(EN_T, None, None, 8, "en_test", en=True)

# ---------------- QA: 상황마다 메시지 틀과 질문들 (질문, 답이 되는 칸 이름 또는 None)
QA = {
    "계약": ("{PER1}님이 {DATE1}에 {ORG1}{|과/와} 계약을 맺었습니다. 계약금 {MONEY1}{|은/는} {DATE2}까지 입금되고, 잔금 {MONEY2}{|은/는} {PER2}님이 처리합니다.",
           [("계약은 언제 맺었나요?", "DATE1"), ("계약금은 얼마인가요?", "MONEY1"), ("계약금은 언제까지 입금되나요?", "DATE2"), ("잔금은 누가 처리하나요?", "PER2"),
            ("누가 계약을 맺었나요?", "PER1"), ("어느 회사와 계약했나요?", "ORG1"), ("잔금은 얼마인가요?", "MONEY2"), ("중도금은 얼마인가요?", None)],
           [("계약한 날짜가 언제예요?", "DATE1"), ("잔금 담당자는 누구예요?", "PER2"), ("계약 상대 회사 이름은?", "ORG1"), ("위약금은 얼마예요?", None)]),
    "출장": ("{PER1}님은 {DATE1}부터 {ORG1} 공장으로 출장을 갑니다. 항공권은 {MONEY1}{|이었고/였고}, 숙박비 {MONEY2}{|은/는} {PER2}님이 {DATE2}에 정산합니다.",
           [("출장은 언제부터인가요?", "DATE1"), ("누가 출장을 가나요?", "PER1"), ("어느 회사로 출장을 가나요?", "ORG1"), ("항공권은 얼마였나요?", "MONEY1"),
            ("숙박비는 얼마인가요?", "MONEY2"), ("숙박비는 누가 정산하나요?", "PER2"), ("정산은 언제 하나요?", "DATE2"), ("식비는 얼마인가요?", None)],
           [("출장 가는 사람이 누구죠?", "PER1"), ("비행기 표 가격이 얼마예요?", "MONEY1"), ("정산 날짜는요?", "DATE2"), ("렌터카 비용은 얼마예요?", None)]),
    "구매": ("{ORG1}에서 노트북 20대를 {MONEY1}에 사기로 했습니다. {PER1}님이 {DATE1}에 주문했고, 배송은 {DATE2}에 {ORG2}{|이/가} 맡습니다.",
           [("노트북은 얼마에 사기로 했나요?", "MONEY1"), ("어디에서 노트북을 사나요?", "ORG1"), ("누가 주문했나요?", "PER1"), ("주문은 언제 했나요?", "DATE1"),
            ("배송은 언제인가요?", "DATE2"), ("배송은 어느 회사가 맡나요?", "ORG2"), ("모니터는 몇 대인가요?", None)],
           [("노트북 구매 금액은?", "MONEY1"), ("주문한 사람이 누구예요?", "PER1"), ("택배 회사는 어디예요?", "ORG2"), ("설치비는 얼마예요?", None)]),
    "채용": ("{ORG1}{|은/는} {DATE1}까지 개발자를 뽑습니다. 면접은 {PER1}님과 {PER2}님이 맡고, 연봉은 {MONEY1}부터 시작합니다. 입사일은 {DATE2}입니다.",
           [("지원 마감은 언제인가요?", "DATE1"), ("어느 회사가 개발자를 뽑나요?", "ORG1"), ("연봉은 얼마부터인가요?", "MONEY1"), ("입사일은 언제인가요?", "DATE2"),
            ("면접관 중 한 명은 누구인가요?", "PER1"), ("디자이너도 뽑나요?", None)],
           [("언제까지 지원할 수 있어요?", "DATE1"), ("시작 연봉이 얼마예요?", "MONEY1"), ("첫 출근 날짜는요?", "DATE2"), ("상여금은 얼마예요?", None)]),
}


# 찾아서 읽기용: 메시지 안의 이름, 회사를 언급해서 어느 메시지인지 가릴 수 있는 질문 (상황마다 질문 틀, 답이 되는 칸)
SEARCH_Q = {"계약": ("{ORG1}{|과/와} 맺은 계약의 잔금은 누가 처리하나요?", "PER2"), "출장": ("{PER1}님 출장의 항공권은 얼마였나요?", "MONEY1"),
            "구매": ("{ORG1}에서 산 노트북은 언제 배송되나요?", "DATE2"), "채용": ("{ORG1} 개발자 채용의 입사일은 언제인가요?", "DATE2")}
search = []


def qa_rows(n_msgs, split, use_new_q):
    rows = []
    for name, (tmpl, qs, qs_new) in QA.items():
        for _ in range(n_msgs):
            pers = rng.sample(PER if split != "test" else PER_NEW, 2)
            orgs = rng.sample(ORG if split != "test" else ORG_NEW, 2)
            vals = {"PER1": pers[0], "PER2": pers[1], "ORG1": orgs[0], "ORG2": orgs[1], "DATE1": date(), "DATE2": date(), "MONEY1": money(), "MONEY2": money()}
            if name == "채용":                                  # 연봉은 만 원 단위로
                vals["MONEY1"] = f"{rng.randint(30, 70) * 100}만 원"
            while vals["DATE2"] == vals["DATE1"]:
                vals["DATE2"] = date()
            while vals["MONEY2"] == vals["MONEY1"]:
                vals["MONEY2"] = money()
            text, spans = render(tmpl, vals)
            pos = {}
            k = 0
            for key in [m.split("}")[0] for m in tmpl.split("{")[1:] if not m.startswith("|")]:
                pos[key] = spans[k]
                k += 1
            for q, key in (qs_new if use_new_q else qs):
                a = None if key is None else {"start": pos[key][0], "text": text[pos[key][0]:pos[key][1]]}
                rows.append({"context": text, "question": q, "answer": a, "scene": name, "split": split})
    return rows


qa = qa_rows(30, "train", False) + qa_rows(5, "valid", False) + qa_rows(6, "test", True)
# 찾아서 읽기용 메시지 12개: 상황마다 3개. 질문의 단서(회사나 사람) 12개를 먼저 정하고, 나머지 칸은 단서가 아닌 이름으로만 채워서
# 단서가 메시지 하나에만 나오게 한다
org_clues = rng.sample(ORG, 9)
per_clues = rng.sample(PER, 3)
other_per = [x for x in PER if x not in per_clues]
for si, (name, (tmpl, _, _)) in enumerate(QA.items()):
    clue_key = "PER1" if name == "출장" else "ORG1"
    clues = per_clues if name == "출장" else org_clues[3 * (si - (si > 1)):3 * (si - (si > 1)) + 3]
    for clue in clues:
        pers, orgs = rng.sample(other_per, 2), rng.sample(ORG_NEW, 2)
        vals = {"PER1": pers[0], "PER2": pers[1], "ORG1": orgs[0], "ORG2": orgs[1], "DATE1": date(), "DATE2": date(), "MONEY1": money(), "MONEY2": money()}
        vals[clue_key] = clue
        if name == "채용":
            vals["MONEY1"] = f"{rng.randint(30, 70) * 100}만 원"
        text, spans = render(tmpl, vals)
        keys = [m.split("}")[0] for m in tmpl.split("{")[1:] if not m.startswith("|")]
        pos = dict(zip(keys, spans))
        sq, skey = SEARCH_Q[name]
        q_text, _ = render(sq, vals)
        search.append({"question": q_text, "context": text, "answer": text[pos[skey][0]:pos[skey][1]], "scene": name, "clue": clue})
OUT.mkdir(parents=True, exist_ok=True)
for fname, rows in (("ner.jsonl", ner), ("qa.jsonl", qa), ("search.jsonl", search)):
    with (OUT / fname).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(fname, len(rows))
