"""문장 분류 쪽의 예제 데이터(가상의 서비스 문의)를 만든다 → examples/classify/data/issues.csv

- 종류 셋: 버그, 기능 요청, 질문. 같은 기능 이름이 세 종류 모두에 나와서, 주제어가 아니라 말투와 의도를 봐야 맞힐 수 있다.
- 문장 틀(template)을 종류마다 16개 두고 train 10, valid 2, test 4로 나눈다. valid와 test에는 train에 없던 틀만 들어간다.
- train 틀 × 기능 이름 조합 중 일부만 라벨이 있는 train으로 쓰고, 나머지는 라벨 없는 문장(unlabeled)으로 둔다.
실행: python tools/make_issue_examples.py  (seed 고정, 다시 돌려도 같은 파일)
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "classify" / "data" / "issues.csv"

TOPICS = ["로그인", "결제", "파일 업로드", "엑셀 내보내기", "검색", "알림", "댓글 작성", "비밀번호 변경", "이메일 인증", "프로필 수정",
          "일정 공유", "자동 저장", "2단계 인증", "공유 링크 생성", "데이터 백업", "회원 탈퇴", "쿠폰 적용", "리포트 생성", "사진 첨부", "주문 취소"]


def josa(word, a, b):
    ch = word.strip()[-1]
    if "가" <= ch <= "힣":
        return a if (ord(ch) - 0xAC00) % 28 else b
    return a if ch in "013678" else b


def fill(t, topic):
    return (t.replace("{t}{이}", topic + josa(topic, "이", "가")).replace("{t}{을}", topic + josa(topic, "을", "를"))
             .replace("{t}{은}", topic + josa(topic, "은", "는")).replace("{t}", topic))


# 종류마다 16개: 앞 10개 train, 다음 2개 valid, 마지막 4개 test
TEMPLATES = {
    "버그": [
        "{t}{이} 안 돼요. 버튼을 눌러도 아무 반응이 없어요.", "{t} 하다가 오류 코드 500이 떠요.", "어제부터 {t}{이} 계속 실패합니다.",
        "{t} 화면에서 앱이 강제 종료돼요.", "업데이트 이후 {t}{이} 작동하지 않습니다.", "{t} 버튼을 누르면 흰 화면만 나와요.",
        "크롬에서는 {t}{이} 되는데 사파리에서는 안 돼요.", "{t} 결과가 똑같이 두 번씩 나와요.", "{t}{이} 1분 넘게 걸리다가 결국 멈춰요.",
        "{t} 도중에 에러가 나면서 입력한 내용이 다 날아갔어요.",
        "{t}할 때마다 로딩만 돌고 끝나지 않아요.", "방금 {t} 시도했는데 '알 수 없는 오류'라고만 나오네요.",
        "{t} 기능 버그 있는 듯요. 계속 튕김", "{t}{이} 갑자기 먹통이에요.", "다른 팀원도 {t}{이} 안 된다고 합니다. 확인 부탁드려요.",
        "{t} 누르면 이상한 페이지로 넘어가요. 원래 이러면 안 되잖아요?",
    ],
    "기능 요청": [
        "{t}{을} 한 번에 여러 개 처리할 수 있게 해 주세요.", "{t} 설정을 사용자가 직접 바꿀 수 있으면 좋겠습니다.", "{t}에도 단축키가 있으면 편할 것 같아요.",
        "다음 버전에 {t} 개선을 꼭 넣어 주세요.", "{t}{을} 모바일 앱에서도 쓸 수 있게 지원해 주세요.", "{t} 기록을 CSV로 받을 수 있게 해 주실 수 있나요?",
        "{t} 화면에 필터 옵션이 추가되면 좋겠어요.", "다른 서비스처럼 {t}에도 자동화 기능이 생기면 좋겠습니다.", "{t} 알림을 끄는 옵션도 있었으면 해요.",
        "{t}{을} API로도 쓸 수 있게 열어 주세요.",
        "{t}{을} 팀 단위로 관리하는 기능이 필요합니다.", "{t} 쪽에 미리 보기 기능도 넣어 주시면 감사하겠습니다.",
        "{t} 쪽에 이런 기능 있으면 대박일 듯. 기록 보기 같은 거요", "요청 하나 드려요. {t} 순서를 바꿀 수 있게 해 주세요.",
        "혹시 {t}에 예약 기능을 추가할 계획은 없나요?", "{t} 개선 제안: 버튼 위치를 위로 옮기면 좋겠어요.",
    ],
    "질문": [
        "{t}{은} 어디서 설정하나요?", "{t}{을} 하려면 어떻게 해야 하나요?", "{t} 기능은 무료 요금제에서도 쓸 수 있나요?",
        "{t} 방법을 알려 주실 수 있을까요?", "{t} 관련 사용 설명서가 있나요?", "{t}{을} 취소하고 싶은데 가능한가요?",
        "관리자가 아니어도 {t}{을} 할 수 있나요?", "{t}{은} 하루에 몇 번까지 되나요?", "{t} 기록은 얼마나 오래 보관되나요?",
        "처음 써 보는데 {t}{은} 어떻게 쓰는 건가요?",
        "{t}{을} 다른 사람에게 맡길 수 있는지 궁금해요.", "회사 계정으로도 {t} 가능한가요?",
        "{t} 어케 해요?", "{t} 쓰는 법 좀 알려 주세요.", "{t}{이} 원래 이렇게 오래 걸리는 건가요?",
        "{t} 하기 전에 미리 알아 둘 게 있을까요?",
    ],
}
# 라벨이 있는 train 수 (종류마다 일부러 다르게: 실제 문의도 버그가 가장 많다고 가정)
N_TRAIN = {"버그": 90, "기능 요청": 60, "질문": 40}

rng = random.Random(0)
rows = []
for label, temps in TEMPLATES.items():
    combos = [(ti, tp) for ti in range(10) for tp in TOPICS]
    rng.shuffle(combos)
    for k, (ti, tp) in enumerate(combos):
        rows.append({"text": fill(temps[ti], tp), "label": label, "template": f"{label[:2]}{ti:02d}", "split": "train" if k < N_TRAIN[label] else "unlabeled"})
    for split, rng_t in (("valid", range(10, 12)), ("test", range(12, 16))):
        tps = TOPICS[:]
        rng.shuffle(tps)
        for ti in rng_t:
            for tp in tps[:10]:   # 틀마다 기능 이름 10개
                rows.append({"text": fill(temps[ti], tp), "label": label, "template": f"{label[:2]}{ti:02d}", "split": split})
rng.shuffle(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "text", "label", "template", "split"])
    w.writeheader()
    for i, r in enumerate(rows):
        w.writerow({"id": f"I{i:04d}", **r})
print(OUT, len(rows))
