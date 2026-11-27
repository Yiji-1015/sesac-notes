"""백엔드 > 외전 > 만들 때 지킬 것. examples/principles/principles.py 하나가 ①~⑦로 자란다 (단계마다 독립된 작은 예제).
원칙 13개를 네 묶음으로 정리하고, 이 노트의 다른 쪽에 이미 '어겨 본 장면'이 있으면 그 단계로 잇는다(링크가 실제로 있는지 검사).
장면이 없던 원칙만 이 쪽의 예제로 어겨 본다. 외부 서비스나 DB에는 연결하지 않는다.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/principles.py
"""
import html
import json
import re
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = (ROOT / "examples" / "principles").resolve()
code = S.parse((EX / "principles.py").read_text(encoding="utf-8"))
ns = S.run(code, 7, EX, filename="principles.py")
ENV = f"Python {sys.version.split()[0]}, " + ", ".join(f"{p} {version(p)}" for p in ("pydantic", "httpx", "python-dotenv"))
E = html.escape
outs = {}
CIRC = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜"
PAGES = {"rag-service": ("backend", "RAG 서비스"), "rag-rerank": ("backend", "RAG 검색 고치기"), "rag-eval": ("backend", "RAG 평가"), "agent-tools": ("backend", "AI Agent와 도구"),
         "service-flow": ("backend", "외전: 서비스 완성"), "indexing": ("backend", "인덱싱"), "document-reading": ("backend", "문서 읽기"), "model-compression": ("model", "모델 가볍게")}


def L(page, n):                                                  # 다른 쪽의 n단계로 가는 링크. 그 단계가 정말 있는지 검사해요
    folder, label = PAGES[page]
    S.claim(f'id="s{n}"' in (ROOT / "topics" / folder / f"{page}.html").read_text(encoding="utf-8"), f"링크 {page}#s{n}가 있다")
    return f'<a href="../{folder}/{page}.html#s{n}">{label} {CIRC[n - 1]}</a>'


def here(n):
    return f'<a href="#s{n}">이 쪽 {CIRC[n - 1]}</a>'


# ---------------- ①
runs, lv, mx = ns["runs"], ns["live_ms"], ns["mixed_ms"]
fx = [r["ms"] for r in runs if r["execution_mode"] == "fixture"]
S.claim(ns["same_value"] and max(fx) < lv / 100 and 0.4 < mx / lv < 0.6, "① 값은 같고, fixture는 100분의 1도 안 걸리고, 섞으면 평균이 절반쯤")
outs["s1"] = S.table("① 같은 계산을 다섯 번은 실제로, 다섯 번은 미리 적어 둔 값으로", ["모드", "값", "시간 (평균)"],
                     [["live (실제 계산)", f"{runs[0]['value']:,}", f"{lv:.3f} ms"], ["fixture (미리 적어 둔 값)", f"{runs[-1]['value']:,}", f"{sum(fx) / len(fx):.4f} ms"],
                      ["<b>모드를 안 보고 섞으면</b>", "", f"<b>{mx:.3f} ms</b>"]],
                     ["값은 똑같아서, 값만 검사하면 둘 다 '정답'이에요. 시험용 값(fixture)으로도 검사가 통과하는 건 정상이에요. 문제는 그 결과를 실제 결과와 구별할 수 없을 때예요.",
                      f"모드를 안 보고 시간 평균을 내면 {mx:.2f}ms로, 실제({lv:.2f}ms)의 절반쯤으로 나와요. 결과마다 <code>execution_mode</code>를 적어 두면 실제만 골라 낼 수 있어요. 아예 실행하지 않았다면 <code>not_run</code>으로 남기고, 0이나 빈 값으로 채우지 않아요."])

# ---------------- ②
kept, lost = ns["kept"], ns["lost_numbers"]
S.claim(lost == ["1", "2", "30"] and not ns["fixed_lost"] and "1" not in re.findall(r"\d+", ns["overwritten"]["text"]), "② 정리 함수가 1, 2, 30을 지웠고, 고친 함수는 안 지운다")
outs["s2"] = S.box("② 공백을 정리하려다 숫자까지 지운 함수", [(f"원본 → \"{E(kept['original'])}\"", 1), (f"덮어쓴 기록 → {{\"text\": \"{E(ns['overwritten']['text'])}\"}}", 2),
                                                        (f"따로 둔 기록 → original_sha256 {kept['original_sha256']}, cleaned \"{E(kept['cleaned'])}\"", 3, "newf"),
                                                        (f"원본과 비교해서 사라진 숫자 → {lost}", 4, "newf"), (f"고친 정리 함수 → \"{E(ns['fixed'])}\", 사라진 숫자 {ns['fixed_lost']}", 5)],
                   ["공백이 여러 개이고, 전각 공백(\\u3000)도 섞인 원문이에요.",
                    "같은 칸에 덮어쓰면 남은 건 '1시간'이 '시간'으로, '30일'이 '일'로 바뀐 글뿐이에요. 무엇이 사라졌는지 알 방법이 없어요.",
                    "원본과 정리본을 다른 칸에 두고, 원본의 해시(내용으로 만든 짧은 지문)도 적어 둬요. 나중에 원본이 바뀌었는지도 해시로 알 수 있어요.",
                    "원본이 있으니 '숫자가 사라졌다'를 기계적으로 잡을 수 있어요. 규정 문서에서 숫자가 사라지면 답이 통째로 틀려져요.",
                    "정리 함수를 고친 뒤에도 같은 비교로 확인해요. 바꾸는 단계마다 '바꾸기 전'을 남겨 두면, 어느 단계가 뜻을 바꿨는지 거꾸로 따라갈 수 있어요."])

# ---------------- ③
ck, pt = ns["checks"], ns["partner"]
S.claim(ck["정상"] == "통과" and "extra_forbidden" in ck["오타 칸 houre"] and ck["글자 \"2\""] == "int_type"
        and pt["v2 (칸 추가)", "PartnerStrict"] == "extra_forbidden" and pt["v2 (칸 추가)", "PartnerTolerant"] == "통과"
        and pt["필수 칸 빠짐", "PartnerTolerant"] == "missing" and ns["quiet_default"] == 0 and ns["loud"] == "KeyError" and ns["none_as_fail"] and not ns["none_kept"], "③ 글과 같다")
outs["s3a"] = S.table("③ 우리 API가 받는 요청 (extra=\"forbid\", strict=True)", ["요청", "결과"], [[E(k), v] for k, v in ck.items()],
                      ["오타 난 칸 houre는 조용히 버려지지 않고 거부돼요(extra_forbidden). 함께 진짜 칸 hours가 없다는 것(missing)도 알려 줘요. 버렸다면 \"2시간으로 설정했는데 안 먹는다\"가 됐을 거예요."])
outs["s3b"] = S.table("남의 API 응답을 읽을 때", ["응답", "엄격하게 (forbid)", "필요한 것만 (ignore)"],
                      [[n, pt[n, "PartnerStrict"], pt[n, "PartnerTolerant"]] for n in ("v1", "v2 (칸 추가)", "필수 칸 빠짐")],
                      ["상대가 응답에 칸을 하나 늘렸을 뿐인데, 우리 요청처럼 엄격하게 읽으면 그날부터 서비스가 깨져요. 남의 응답은 <b>필요한 칸만</b> 엄격하게 읽고 나머지는 흘려보내요.",
                       "그래도 꼭 필요한 칸(reservation_id)이 빠지면 둘 다 멈춰요. 느슨하게 읽는 것과 빠진 걸 채워 넣는 건 달라요."])
outs["s3"] = S.box("없는 값과 아직 모르는 값", [(f"reply.get(\"reservation_id\", 0) → {ns['quiet_default']}", 1), (f"reply[\"reservation_id\"] → {ns['loud']}", 2, "newf"),
                                              (f"evidence_ok = None 에서 not evidence_ok → {ns['none_as_fail']}", 3), (f"evidence_ok is False → {ns['none_kept']}", 4, "newf")],
                    ["<code>.get()</code>에 기본값을 주면 없는 예약 번호가 가짜 0으로 채워져, 0번 예약이 있는 것처럼 다음 단계로 넘어가요.", "꼭 있어야 하는 값은 <code>[]</code>로 읽어서 없으면 바로 멈추게 해요.",
                     "None은 '아직 검사 안 함'인데, <code>not</code>으로 보면 '검사 실패'와 같아져요.", "<code>is False</code>로 실패만 실패로 보고, None은 '검토 대기'로 따로 다뤄요."])

# ---------------- ④
st = ns["settings"]
S.claim(st[0]["missing"] == ["DB_HOST", "DB_USER", "DB_PASSWORD"] and not st[1]["missing"] and st[1]["placeholder"] == ["DB_PASSWORD"] and "example-pass" not in json.dumps(st, ensure_ascii=False) and ns["interpreter"]["in_venv"], "④ URL 방식은 셋 다 없음, 키 방식은 예시값, 값은 가려짐, 가상환경")
outs["s4"] = S.table("④ 같은 DB 설정을 두 방식으로 적은 .env를, 키 셋(DB_HOST, DB_USER, DB_PASSWORD)을 읽는 프로그램이 읽으면", ["파일", "읽은 키 (값은 가림)", "없는 키", "예시값 그대로"],
                     [[s["file"], E(json.dumps(s["shown"], ensure_ascii=False)), ", ".join(s["missing"]) or "없음", ", ".join(s["placeholder"]) or "없음"] for s in st],
                     ["URL 한 줄로 적은 파일은 정보가 다 들어 있는데도, 이 프로그램이 찾는 키가 하나도 없어요. 다른 프로젝트의 .env를 복사해 왔을 때 \"값을 넣었는데 설정하라고 나온다\"의 정체예요.",
                      "키 방식은 키는 다 있지만 비밀번호가 예시값(CHANGE_ME) 그대로예요. 시작할 때 이걸 잡아 멈추면, DB 접속 실패라는 엉뚱한 증상으로 헤매지 않아요.",
                      f"무엇을 읽었는지 출력할 때는 비밀값을 가려요(***). 지금 실행 중인 파이썬은 {ns['interpreter']['name']}이고 가상환경 안이에요({ns['interpreter']['in_venv']}). 패키지를 설치한 파이썬과 실행하는 파이썬이 같은지 확인하는 게 \"설치했는데 import가 안 된다\"의 첫 점검이에요."])

# ---------------- ⑤
pr, kr = ns["plain_result"], ns["keyed_result"]
S.claim(ns["plain_state"]["reservations"] == [101, 102] and pr[1] == 2 and ns["keyed_state"]["reservations"] == [101] and kr == ({"reservation_id": 101}, 2), "⑤ 키 없으면 예약 둘, 키 있으면 하나")
outs["s5"] = S.table("⑤ 첫 응답만 늦는 서버에 예약을 보내고, 시간 초과면 다시 보내기", ["클라이언트", "보낸 횟수", "받은 답", "서버에 생긴 예약"],
                     [["재시도만", pr[1], E(json.dumps(pr[0])), ", ".join(map(str, ns["plain_state"]["reservations"]))], ["재시도 + Idempotency-Key", kr[1], E(json.dumps(kr[0])), ", ".join(map(str, ns["keyed_state"]["reservations"]))]],
                     ["첫 요청은 서버가 예약을 저장했는데 응답만 늦었어요. 클라이언트는 실패로 보고 다시 보냈고, 서버는 새 요청으로 보고 또 저장했어요. 예약이 두 개 생겼는데, 사용자는 하나만 했다고 알아요.",
                      "요청마다 고유한 키(Idempotency-Key)를 붙이고 서버가 같은 키를 기억하면, 다시 보내도 처음 결과를 돌려줘요. 이렇게 몇 번을 보내도 결과가 같은 성질을 <b>멱등</b>이라고 해요.",
                      "조회는 원래 멱등이라 재시도해도 괜찮지만, 예약, 결제, 메일 보내기처럼 무언가를 바꾸는 요청은 멱등하게 만들기 전에는 자동 재시도를 켜지 않아요. SDK의 기본 재시도(예: <code>max_retries</code>)가 이미 켜져 있는지도 확인해요."])

# ---------------- ⑥
dr, now = ns["drift"], ns["now"]
S.claim(len(dr) >= 5 and "scikit-learn" in dr and "PyTorch" in dr, "⑥ 다섯 개 넘는 라이브러리가 쪽마다 다른 판")
outs["s6"] = S.table(f"⑥ 이 노트의 쪽들이 빌드될 때 적어 둔 판 (판이 둘 이상인 라이브러리 {len(dr)}개)", ["라이브러리", "판: 그 판으로 빌드한 쪽 수"],
                     [[p, ", ".join(f"{v} ({len(x)}쪽)" for v, x in sorted(vers.items()))] for p, vers in sorted(dr.items())],
                     [f"같은 노트 안에서도 라이브러리 {len(dr)}개가 쪽마다 다른 판으로 빌드됐어요. PC를 옮기거나 패키지를 새로 설치하는 사이에 판이 바뀌었기 때문이에요. 쪽마다 '실행 환경' 줄이 있어서 이걸 알 수 있어요.",
                      "판이 바뀌면 같은 코드도 결과가 달라질 수 있어요. 실제로 <a href=\"../backend/rag-eval.html#s1\">RAG 평가 ①</a>의 정답 묶음에는 묶음의 판과 조각의 판을 같이 적어 둬요. 조각을 다시 나누면 같은 조각 ID가 다른 내용을 가리키거나 사라질 수 있어서, 판이 다른 점수끼리는 나란히 비교하지 않아요.",
                      f"그래서 결과에는 판을 같이 적고, 프로젝트에는 판을 고정한 목록(requirements.txt에 ==로, 또는 lock 파일)을 둬요. 이번 실행의 판: {E(json.dumps(now, ensure_ascii=False))}"])

# ---------------- ⑦
rt = ns["routes"]
wrong = [q for q, r in rt.items() if r["tools"] != r["expected"]]
S.claim(wrong == ["이용 안내 3번 항목이 무슨 뜻이에요?", "오늘 하루 어떻게 지냈어?"] and all(r["decided_by"] == "rule" for r in rt.values()), "⑦ 규칙이 둘을 틀림")
outs["s7"] = S.table("⑦ 키워드 규칙으로 도구 고르기 (decided_by: rule)", ["질문", "규칙이 고른 도구", "기대", "맞나"],
                     [[E(q), ", ".join(r["tools"]) or "없음", ", ".join(r["expected"]) or "없음", "맞음" if r["tools"] == r["expected"] else "<b>틀림</b>"] for q, r in rt.items()],
                     ["'3번 항목'의 '번' 때문에 예약 조회로, '어떻게 지냈어'의 '어떻게' 때문에 규정 검색으로 갔어요. 키워드 규칙은 빠르고 예측 가능하지만, 말이 조금만 달라도 틀려요.",
                      "이 결정은 if문이 내렸어요. 기록에 decided_by를 남기지 않으면, 이 실수를 '모델이 도구를 잘못 골랐다'로 오해하고 Prompt를 고치게 돼요. 반대로 모델이 규칙과 다르게 골랐을 때 그걸 버그로 보기도 해요.",
                      "<a href=\"../backend/agent-tools.html#s4\">AI Agent와 도구 ④</a>의 '모델 판단'도 사실 가짜 함수(PLANS)의 이런 규칙이에요. 그 쪽이 이걸 처음부터 밝혀 둔 이유예요."])

flow = [[1, "실행 모드", "live, fixture, not_run", "섞으면 평균이 반"], [2, "원본 보존", "따로 두고 비교", "사라진 숫자 1, 2, 30"], [3, "계약", "내 입력 엄격, 남의 응답 필요한 것만", "칸 하나에 깨짐"],
        [4, "설정", "누가 어느 키를 읽나", "URL 방식은 키 0개"], [5, "재시도", "멱등할 때만", "예약 두 개"], [6, "판", "결과에 적기", f"판이 갈린 라이브러리 {len(dr)}"], [7, "누가 정했나", "규칙 vs 모델", "규칙이 2개 틀림"]]
S.save(ROOT / "topics" / "backend" / "principles-data.js", {"code": code, "names": S.names(code), "range": [1, 7], "outs": outs, "flow": flow, "env": ENV})

# ---------------- 원칙 13개 한눈에 (장면 링크는 L()이 존재를 검사)
GROUPS = [
    ("결과가 어디서 왔나", [
        ("결과에 어떻게 얻었는지 적기", "실제 실행, 시험용 값, 실행 안 함을 결과마다 적고, 모드가 다른 결과를 섞지 않아요", "시험용 값이 실측처럼 보고되고, 시간과 비용에 가짜 값이 섞여요", "결과 파일만 보고 모드를 알 수 있나?", [here(1)]),
        ("원본은 남기고 바꾼 것은 따로", "정리, 변환, 다시 쓰기의 결과는 새 칸에 두고 원본(과 해시)을 남겨요", "결과가 이상할 때 어느 단계가 뜻을 바꿨는지 알 수 없어요", "최종 결과에서 원래 입력까지 거꾸로 따라갈 수 있나?", [here(2), L("rag-rerank", 3), L("document-reading", 1)]),
        ("판을 적어 두기", "라이브러리, 모델, 데이터, 문서 묶음의 판을 결과와 함께 적고, 프로젝트에는 판을 고정해요", "같은 코드가 다른 결과를 내도 원인을 모르고, 설명과 실제 동작이 어긋나요", "결과에 판이 적혀 있나? 다른 PC에서 같은 판을 깔 수 있나?", [here(6), L("rag-eval", 1)]),
        ("증거의 종류를 나누기", "실행됨, 형식 맞음, 저장됨, 내용 맞음은 다른 증거예요. 자동으로 확인한 범위만 확인했다고 말하고, 나머지는 검토 대기로 남겨요", "검사가 통과했으니 답도 맞는다고 믿어요. 출처 형식만 맞고 내용은 다른 답이 승인돼요", "자동 확인한 것과 사람이 볼 것이 나뉘어 있나?", [L("rag-service", 6), L("rag-eval", 5)]),
    ]),
    ("실패를 드러내기", [
        ("실패와 미실행을 성공, 빈 결과, 0으로 바꾸지 않기", "오류, 정상인데 결과 없음, 실행 안 함을 다른 값으로 남겨요. 대체 경로를 쓴다면 조용히 말고 드러나게(상태값, 기록, 지표)", "장애가 '정보가 없어요'처럼 보이고, 몰래 바뀐 경로 때문에 '느려졌다'는 증상만 남아요", "일부러 실패를 일으키면 결과에 실패로 찍히나?", [L("rag-service", 7), L("agent-tools", 5), L("service-flow", 5)]),
        ("단계마다 흔적을 남기고, 실패는 앞 단계부터", "단계별 입력과 출력 요약, 시간, 검사 결과를 남기고, 틀리면 권한, 판, 준비 상태부터 차례로 봐요", "'답이 틀렸다'만 알고 엉뚱한 단계를 고쳐요", "실패 하나를 골라 기록만 보고 틀린 단계를 말할 수 있나?", [L("rag-rerank", 2), L("rag-eval", 2), L("rag-eval", 6)]),
    ]),
    ("경계와 계약", [
        ("내 입력은 엄격하게, 남의 응답은 필요한 것만", "우리가 받는 입력은 모르는 칸과 빠진 칸을 거부하고, 남의 응답은 필요한 칸만 엄격하게 읽어요. 없는 값을 기본값으로 채우지 않아요", "오타 칸이 조용히 무시되거나, 상대가 칸 하나 늘린 날 서비스가 깨져요", "모르는 칸, 빠진 칸, 상대의 새 칸을 넣어 시험해 봤나?", [here(3), L("rag-service", 6), L("agent-tools", 2)]),
        ("설정은 누가 어느 파일을 읽는지부터", "설정 파일의 위치, 키 이름, 읽는 때를 확인하고, 설치와 실행에 같은 파이썬을 써요. 비밀값은 코드, 기록, Git에 남기지 않고, 예시값이면 시작에서 멈춰요", "값을 넣었는데 '설정하세요'가 나오고, 키가 공개 저장소에 올라가요", "읽은 파일과 키 목록을 (값은 가리고) 출력해 볼 수 있나?", [here(4), L("service-flow", 1)]),
        ("남의 자원은 읽기만, 준비 상태는 단계별로", "공유 DB 같은 자원은 다시 만들거나 지우지 않고 읽기만 해요. 설치됨, 켜짐, 데이터 있음, 판 맞음을 따로 확인하고, 준비와 실행과 검증을 다른 명령으로 나눠요", "시험하다가 남의 데이터를 지우거나, 낡은 데이터로 엉뚱한 결과가 나와요", "점검이 단계마다 따로 보고하나? 실행이 시작 데이터를 건드리지 않나?", [L("agent-tools", 2), L("service-flow", 2), L("indexing", 17)]),
        ("권한은 서버가 정하고, 겹겹이 막기", "요청에 든 권한이나 사용자 정보를 믿지 않고 인증된 서버 값으로 정해요. 앱의 범위 제한은 인증이 아니에요. 입력 검사, 데이터 조건, 계정 권한을 겹쳐 두고, 자동 반복에는 상한을 걸어요", "요청에 'admin'만 넣으면 권한이 올라가고, 반복 호출로 비용이 폭발해요", "클라이언트가 보낸 권한 칸이 무시되나? 상한을 넘으면 멈추나?", [L("agent-tools", 2), L("agent-tools", 5), L("agent-tools", 3)]),
        ("재시도는 멱등할 때만", "자동 재시도는 같은 요청이 두 번 처리돼도 결과가 같을 때만 켜요. 바꾸는 요청에는 고유 키를 붙이고, 시간 제한을 함께 둬요", "응답만 늦었던 요청이 다시 처리돼 예약이나 결제가 두 번 생겨요", "같은 요청을 두 번 보내면 결과가 하나인가?", [here(5)]),
    ]),
    ("판단과 비교", [
        ("비교는 공정하게, 결론은 조심스럽게", "한 번에 하나만 바꾸고(조합은 따로), 여러 사례로 보고, 같은 때 번갈아 재서 잡음을 줄여요. 정답 라벨은 평가에만 쓰고, 잴 수 없는 건 0이 아니라 N/A로 빼요", "무엇 때문에 바뀌었는지 모르고, 정답이 입력에 새어 커닝한 점수가 나와요", "두 실행의 차이가 설정 하나뿐인가? 라벨이 들어가는 곳이 평가 함수뿐인가?", [L("rag-eval", 1), L("rag-eval", 6), L("model-compression", 1)]),
        ("규칙이 정한 것과 모델이 정한 것을 나누기", "if문이 내린 결정과 모델이 내린 결정을 기록과 설명에서 구분해요", "규칙의 실수를 모델 탓으로, 모델의 다른 선택을 버그로 오해해요", "이 결정을 누가 내렸는지 결과에 적혀 있나?", [here(7), L("agent-tools", 4)]),
    ]),
]
n_rules = sum(len(rs) for _, rs in GROUPS)
S.claim(n_rules == 13, "원칙 13개")
trs, k = [], 0                                                   # 묶음 줄은 표 전체 폭으로, 확인하는 법은 규칙 칸 아래에, 장면 링크는 줄바꿈 없이
for g, rs in GROUPS:
    trs.append(f'<tr><td colspan="4" style="background:var(--accent-soft)"><b>{g}</b></td></tr>')
    for name, rule, broken, check, scenes in rs:
        k += 1
        nowrap = "<br>".join(f'<span style="white-space:nowrap">{s}</span>' for s in scenes)
        trs.append(f"<tr><td>{k}. <b>{name}</b></td><td>{rule}<br><small>확인: {check}</small></td><td>{broken}</td><td>{nowrap}</td></tr>")
hub = "<table><tr><th>원칙</th><th>규칙</th><th>어기면</th><th>어겨 본 장면</th></tr>" + "".join(trs) + "</table>"

why = f"""<p>무엇을 만들든 같은 실수가 되풀이돼요. 시험용 값으로 낸 결과를 실측처럼 보고하고, 오류를 '결과 없음'으로 숨기고, 원본을 덮어써서 무엇이 바뀌었는지 모르게 되고, 요청에 든 권한을 그대로 믿어요. 기능 코드가 맞아도 이런 실수 하나로 서비스나 실험 전체를 믿을 수 없게 돼요.</p>
<p>원칙은 특정 기술에 묶이지 않아요. API 서버, 모델 실험, 평가, Agent에 같은 원칙이 그대로 적용돼요. 네 묶음은 다음과 같아요.</p>
{K.table(["묶음", "묻는 것"], [["결과가 어디서 왔나", "이 숫자와 답을 어떻게 얻었고, 무엇으로 확인했나"], ["실패를 드러내기", "잘못됐을 때 잘못됐다고 보이나, 어디서 잘못됐는지 찾을 수 있나"],
                          ["경계와 계약", "입력, 설정, 자원, 권한, 재시도의 선을 누가 어디서 지키나"], ["판단과 비교", "더 좋아졌다는 말과 누가 정했다는 말을 믿을 수 있나"]])}
<p>원칙에는 조건이 붙어요. 대체 경로(fallback)는 <b>조용히</b> 바뀔 때만 문제예요. 엄격한 계약은 우리가 받는 입력에 쓰고, 남의 응답은 필요한 칸만 엄격하게 읽어요. 남의 응답까지 전부 엄격하게 읽으면 상대가 칸 하나만 늘려도 깨져요. 아래 표는 원칙마다 어기면 무엇이 망가지는지와, 이 노트에서 실제로 어긴 장면을 함께 보여 줘요. 장면이 없는 원칙은 ①~⑦에서 일부러 어겨요.</p>
"""
whole = f"""<p>원칙 {n_rules}개를 한 표로 모았어요. '어겨 본 장면'을 누르면 그 원칙이 실제 실행에서 드러난 단계로 가요.</p>
{hub}
<p>아래 ①~⑦은 다른 쪽에 장면이 없던 원칙을 작은 예제로 어겨 보는 단계예요. 단계마다 서로 독립이에요. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "결과에 어떻게 얻었는지 적기", "같은 계산을 실제로 한 결과와 미리 적어 둔 값으로 낸 결과를, 모드를 적은 채로 모아요.",
           "<p>시험할 때는 외부 API나 GPU 없이도 돌도록 미리 적어 둔 값(fixture)이나 가짜 서버(mock)를 써요. 문제는 fixture나 mock 자체가 아니라, 그렇게 나온 결과가 실제 결과와 같은 칸, 같은 파일에 섞이는 거예요.</p>"
           + K.code_table([("{\"execution_mode\": \"live\" | \"fixture\" | \"not_run\"}", "결과마다 어떻게 얻었는지"), ("time.perf_counter()", "시간 재기")]), "s1"),
    K.step(2, "원본을 남기고, 바꾼 것은 따로", "정리 함수가 숫자까지 지우는 실수를, 덮어쓴 기록과 원본을 남긴 기록으로 비교해요.", K.code_table([("hashlib.sha256(text.encode()).hexdigest()", "내용으로 만든 지문(해시). 원본이 바뀌었는지 확인"), ("unicodedata.normalize(\"NFC\", text)", "한글 자모 조합 맞추기"), ("re.findall(r\"\\d+\", text)", "글 속 숫자들")]), "s2"),
    K.step(3, "내 입력은 엄격하게, 남의 응답은 필요한 것만 엄격하게", "우리가 받는 요청은 모르는 칸을 거부하고, 남이 주는 응답은 필요한 칸만 엄격하게 읽어요. 없는 값은 채우지 않아요.",
           K.out("s3a") + K.out("s3b")
           + K.code_table([("ConfigDict(extra=\"forbid\", strict=True)", "모르는 칸 거부, 자료형 자동 변환 안 함"), ("ConfigDict(extra=\"ignore\")", "모르는 칸은 흘려보내기"), ("d[\"key\"] vs d.get(\"key\", 기본값)", "없으면 멈추기 vs 가짜 값으로 채우기")]), "s3"),
    K.step(4, "설정은 누가 어느 파일을 읽는지부터", "같은 DB 설정을 두 방식으로 적은 .env를 같은 프로그램으로 읽어, 무엇을 읽었고 무엇이 없는지 (값은 가리고) 보고해요.",
           K.code_table([("dotenv_values(path)", ".env를 읽어 키와 값 묶음으로 (환경 변수는 건드리지 않아요)"), ("sys.executable, sys.prefix != sys.base_prefix", "지금 실행 중인 파이썬, 가상환경 안인지")]), "s4"),
    K.step(5, "재시도는 멱등할 때만", "저장은 했는데 응답만 늦은 서버에, 재시도만 하는 클라이언트와 고유 키를 붙인 클라이언트가 예약을 보내요.",
           K.code_table([("httpx.MockTransport(handle)", "가짜 서버 (실제로 보내지 않아요)"), ("httpx.ReadTimeout", "응답이 제때 오지 않음"), ("headers={\"Idempotency-Key\": ...}", "같은 요청임을 알리는 고유 키")]), "s5"),
    K.step(6, "판을 적어 두기", "이 노트의 쪽들이 빌드될 때 적어 둔 실행 환경에서, 같은 라이브러리가 쪽마다 다른 판인 경우를 찾아요.",
           K.code_table([("importlib.metadata.version(\"pydantic\")", "설치된 라이브러리의 판"), ("sys.version", "파이썬 판")]), "s6"),
    K.step(7, "규칙이 정한 것과 모델이 정한 것", "키워드 규칙으로 도구를 고르게 하고, 결정마다 누가 내렸는지(decided_by) 적어요.",
           K.code_table([("{\"tools\": [...], \"decided_by\": \"rule\"}", "결정과 결정한 주체를 함께")]), "s7"),
]
practice = [("새 프로젝트를 시작할 때", "판 고정 목록(requirements.txt의 ==), .env.example(예시값만), 설정을 읽는 곳 하나, 결과 파일에 execution_mode와 판 칸을 먼저 만들어 둬요."),
            ("결과를 남길 때", "모드, 판, 입력(원본), 단계별 요약과 시간, 자동 검사 결과, 사람 검토 대기 여부를 한 줄에. 키와 개인정보는 빼요."),
            ("실패가 났을 때", "앞 단계부터: 권한과 설정 → 준비 상태와 판 → 입력 해석 → 후보 → 순서 → 결과 검사. 고치기 전에 같은 실패를 다시 일으킬 수 있는지 먼저 봐요."),
            ("더 좋아졌다고 말하기 전에", "같은 조건에서 하나만 바꿨나, 여러 사례인가, 같은 때 번갈아 쟀나, 정답이 입력에 새지 않았나, N/A를 0으로 넣지 않았나."),
            ("외부와 닿는 곳에서", "받는 입력은 엄격하게, 남의 응답은 필요한 칸만. 시간 제한, 재시도는 멱등할 때만, 실패는 실패로. 권한은 서버 값으로."),
            ("조건까지 지키기", "대체 경로는 드러나게, 느슨한 읽기는 필요한 칸만, 재시도는 멱등할 때만 써요. 조건을 빼면 같은 기법이 문제를 숨겨요.")]
libs = [("①", "<code>time.perf_counter</code>, <code>statistics.mean</code>", "시간과 평균"), ("②", "<code>hashlib.sha256</code>, <code>unicodedata.normalize</code>, <code>re</code>", "원본 지문, 정리, 비교"),
        ("③", "<code>BaseModel</code>, <code>ConfigDict</code>, <code>ValidationError</code>", "계약 (pydantic)"), ("④", "<code>dotenv_values</code>, <code>sys.executable</code>", "설정 읽기 (python-dotenv)"),
        ("⑤", "<code>httpx.Client</code>, <code>MockTransport</code>, <code>ReadTimeout</code>", "재시도 (httpx)"), ("⑥", "<code>importlib.metadata.version</code>", "판"), ("⑦", "<code>re.search</code>", "키워드 규칙")]
K.build(ROOT / "topics" / "backend" / "principles.html", title="만들 때 지킬 것", h1="만들 때 지킬 것: 결과를 믿을 수 있게 하는 원칙 13개", track="백엔드", crumb="백엔드 &gt; 외전: 서비스를 완성하는 흐름",
        one_line="결과를 믿으려면 네 가지가 갖춰져야 해요. 결과가 <b>어디서 왔는지</b> 적혀 있고, 실패가 <b>실패로 보이고</b>, 입력과 권한의 <b>경계를 서버가 지키고</b>, 비교가 <b>공정해야</b> 해요. 이 쪽은 원칙 13개를 이 네 묶음으로 모으고, 원칙마다 노트의 다른 쪽에서 <b>실제로 어긴 장면</b>으로 잇고, 장면이 없는 원칙은 작은 예제로 어겨요.",
        why=why, whole=whole, steps=steps, code_file="examples/principles/principles.py", practice=practice, libs=libs,
        libs_note="<code>measure</code>, <code>live_work</code>, <code>fixture_work</code>, <code>clean_buggy</code>, <code>clean</code>, <code>load_settings</code>, <code>make_server</code>, <code>reserve_with_retry</code>, <code>env_record</code>, <code>keyword_route</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"service-flow.html\">외전: 로직 밖에서 서비스를 완성하기</a>", "이 원칙들을 백엔드 서비스 하나에 차례로 적용한 길"), ("<a href=\"rag-service.html\">RAG 서비스</a>", "증거의 종류, 실패 드러내기"),
                   ("<a href=\"rag-eval.html\">RAG 평가</a>", "공정한 비교, 단계별 흔적"), ("<a href=\"agent-tools.html\">AI Agent와 도구</a>", "권한, 반복 상한, 규칙과 모델")],
        sources=["모은 자료: 부트캠프 실습 자료(2026-11-16~27)의 안내문에 날마다 되풀이된 주의 문구를, 실습이나 RAG에 묶이지 않는 일반 원칙으로 다시 정리했어요. 실습 코드와 데이터는 옮기지 않고, 예제는 가상의 스터디룸 예약으로 새로 만들었어요",
                 "더한 원칙: 판 적어 두기, 재시도와 멱등성은 이 노트를 만들며 실제로 겪은 일(라이브러리 판에 따라 동작이 다름, SDK의 자동 재시도)에서 더했어요",
                 "결과 값: <code>tools/pages/principles.py</code>가 <code>examples/principles/principles.py</code>를 단계별로 실제 실행한 결과. ⑥은 이 저장소의 <code>topics/*/*-data.js</code>에 적힌 실행 환경을 읽은 값",
                 "공식 문서: <a href=\"https://docs.pydantic.dev/latest/concepts/models/#extra-data\" target=\"_blank\">Pydantic extra data</a>, <a href=\"https://www.python-httpx.org/advanced/transports/#mock-transports\" target=\"_blank\">httpx mock transports</a>, <a href=\"https://pypi.org/project/python-dotenv/\" target=\"_blank\">python-dotenv</a>"])
print("ok")
