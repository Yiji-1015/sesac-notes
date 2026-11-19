"""모델 > 이미지 > OCR과 문서 구조화.
실행: C:/practice_assignment/.venv/Scripts/python.exe tools/pages/ocr_docs.py
"""
import json
import sys
from html import escape as E
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pagekit as K  # noqa: E402
import stepcode as S  # noqa: E402

import cv2  # noqa: E402
import pytesseract  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
ROOT = K.ROOT
EX = ROOT / "examples" / "ocr"
IMG = ROOT / "topics" / "model" / "img" / "ocr"
IMG.mkdir(parents=True, exist_ok=True)
code = S.parse((EX / "invoice_ocr.py").read_text(encoding="utf-8"))
ns = S.run(code, 6, S.workdir(EX), filename="invoice_ocr.py")
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TV = str(pytesseract.get_tesseract_version())


def save(a, name, w=220):
    h = int(a.shape[0] * w / a.shape[1])
    cv2.imwrite(str(IMG / name), cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA))
    return f"img/ocr/{name}"


D = EX / "data"
outs = {}
raws = {n: cv2.imread(str(D / f"{n}_raw.png")) for n in ns["fixes"]}
b = ns["bad"]
outs["why"] = S.images("같은 청구서(doc03)를 세 가지로 찍은 사진", [(save(cv2.imread(str(D / "doc03.png")), "doc03.png"), "깨끗한 원본")] + [(save(raws[n], f"{n}_raw.png"), f"{n}: 글자 오류율 {b[n]['cer_before']:.0%}") for n in raws],
                       ["사람 눈에는 넷 다 읽히는데, OCR 결과는 크게 달라요. 그림자 진 사진은 <b>한 글자도</b> 읽지 못했어요 (오류율 100%).",
                        "그리고 다 읽었다 해도 글자 덩어리일 뿐이에요. 업무에 쓰려면 \"날짜는?\", \"합계는?\"처럼 <b>값</b>으로 꺼내고, 맞는지 <b>검산</b>해야 해요."])
outs["s1"] = S.box("① doc01.png를 그대로 OCR", [*ns["text01"].strip().splitlines(), "", ("정답 첫 줄 → " + ns["truth"]["doc01"].splitlines()[0], 1)],
                   ["깨끗한 문서라 거의 다 맞았지만, 첫 줄의 <code>DOC01</code>을 <code>DOCO1</code>로 읽었어요. 숫자 0과 영어 O는 글꼴에 따라 거의 똑같이 생겨서 OCR이 자주 헷갈려요."])
outs["s2"] = S.box("② cer(text01, 정답)", [(f"CER → {ns['cer01']:.3f}", 1)], ["글자 오류율(CER) = 정답과 같아지려면 고쳐야 하는 글자 수 ÷ 정답 글자 수. 0.007이면 140글자 중 1글자 정도예요. 한 글자지만, 하필 문서 번호예요."])
rows = [[n, f"{v['cer_before']:.3f}", f"<b>{v['cer_after']:.3f}</b>", f"<code>{E(' '.join(v['before'].split())[:40] or '(빈 글)')}</code>"] for n, v in b.items()]
outs["s3"] = S.table("③ 다듬기 전후의 글자 오류율 (정답: doc03)", ["사진", "그대로", "다듬은 뒤", "그대로 읽은 앞부분"], rows,
                     ["세 사진 모두 다듬은 뒤에는 깨끗한 원본과 같은 수준(0.006)이 됐어요. 문제마다 맞는 처방이 달라요: 작으면 키우고, 기울면 바로 세우고, 그림자는 배경 밝기로 나눠 없애요."])
outs["s3img"] = S.images("다듬은 뒤", [(save(ns["fixes"][n](raws[n]) if n != "lowres" else ns["fixes"][n](raws[n]), f"{n}_fixed.png"), n) for n in raws])
outs["s4"] = S.table(f"④ image_to_data: 단어 {len(ns['words'])}개 중 확신 90 미만", ["단어", "확신(conf)", "위치 (x, y, 가로, 세로)"], [[f"<code>{E(w['text'])}</code>", w["conf"], str(w["box"])] for w in ns["unsure"]],
                     ["단어마다 글자, 확신(0~100), 박스를 받았어요. 확신이 낮은 단어는 딱 하나, 앞에서 틀린 <code>DOCO1</code>이에요. OCR도 스스로 자신 없어 했어요. 확신이 낮은 단어만 사람에게 보여 주면 확인할 양이 크게 줄어요."])
p1 = ns["parsed"]["doc01"]
outs["s5"] = S.box("⑤ parse(doc01의 OCR 글)", json.dumps(p1, ensure_ascii=False, indent=1).splitlines()[:9] + ["  …"],
                   ["글자 덩어리에서 \"DATE: 뒤의 날짜\", \"TOTAL: 뒤의 숫자\"처럼 <b>표시 글자 옆의 값</b>을 정규식으로 꺼냈어요. 항목 줄은 \"영어 단어 + 숫자 셋\" 모양인 줄만 골랐어요."])
outs["s6"] = S.table("⑥ 다섯 문서 검산 결과", ["문서", "날짜", "합계", "항목 수", "검산"], [[k, v["date"], v["total"], len(v["items"]), "<br>".join(f"<span class=\"bad\">{E(x)}</span>" if x != "통과" else x for x in ns["checks"][k])] for k, v in ns["parsed"].items()],
                     ["금액(수량 × 단가)과 합계는 다섯 문서 모두 맞아요. 숫자 읽기는 정확했어요.", "다섯 문서 모두 문서 번호에서 걸렸어요. \"DOC + 숫자 두 자리\"여야 하는데 O가 섞였거든요. 이런 문서는 자동으로 넘기지 않고 사람 확인으로 보내요. 숫자 자리의 O를 0으로 바꾸는 규칙을 둘 수도 있지만, 바꾼 사실을 기록해 둬요."])
flow = [[1, "그대로 읽기", "image_to_string", "O와 0"], [2, "오류율", "CER", "편집 거리"], [3, "다듬기", "키우기, 바로 세우기, 그림자", "100% → 0.6%"], [4, "단어와 확신", "image_to_data", "낮은 것만 확인"], [5, "구조화", "정규식", "값으로"], [6, "검산", "규칙", "통과 / 사람 확인"]]
S.save(ROOT / "topics" / "model" / "ocr-documents-data.js", {"code": code, "names": S.names(code), "range": [1, 6], "outs": outs, "flow": flow,
                                                            "env": f"Python {sys.version.split()[0]}, Tesseract {TV} (eng), pytesseract, OpenCV {cv2.__version__}"})
why = f"""<p><b>OCR</b>(광학 문자 인식)은 그림 속 글자를 읽어 글로 바꾸는 일이에요. 영수증, 청구서, 스캔한 서류를 데이터로 바꿀 때 써요. 그런데 사진 상태에 따라 결과가 크게 달라요.</p>{K.out("why")}
<p>그래서 문서 OCR은 \"읽기\" 하나가 아니라 <b>다듬기 → 읽기 → 값 꺼내기 → 검산 → 애매하면 사람 확인</b>의 흐름이에요. 여기서는 무료 OCR 엔진 <b>Tesseract</b>로 합성 영어 청구서 다섯 장을 처리해요.</p>"""
whole = """<p>그대로 읽기 → 얼마나 틀렸나 재기 → 나쁜 사진 다듬기 → 단어별 확신 보기 → 값으로 꺼내기 → 검산. 상자를 누르면 그 단계로 가요.</p><div class="flow" id="flow"></div>"""
steps = [
    K.step(1, "그대로 읽기", "청구서 그림을 Tesseract로 읽어 글로 받아요.",
           K.code_table([("pytesseract.image_to_string(img, lang=\"eng\", config=\"--psm 6\")", "그림 → 글. <code>lang</code>은 언어 데이터, <code>--psm 6</code>은 그림 전체를 글 한 덩어리로 보기"), ("lesson[\"documents\"][k][\"reference_text\"]", "문서마다 사람이 적어 둔 정답 글")]), "s1"),
    K.step(2, "얼마나 틀렸나 재기 (CER)", "OCR 글을 정답 글로 바꾸려면 몇 글자를 고쳐야 하는지(편집 거리) 세고, 정답 길이로 나눠요.",
           K.code_table([("\" \".join(pred.split())", "줄바꿈과 공백 차이는 오류로 세지 않게 정리"), ("min(삭제, 추가, 바꾸기)", "앞 글자까지의 최소 고칠 수에 한 글자씩 더해 가는 계산 (동적 계획법)")]), "s2"),
    K.step(3, "나쁜 사진을 다듬고 읽기", "작은 사진은 키우고, 기운 사진은 바로 세우고, 그림자는 배경 밝기로 나눠 없앤 뒤 다시 읽어요.",
           K.code_table([("cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)", "3배로 부드럽게 키우기. Tesseract는 글자 높이가 20~30픽셀 이상일 때 잘 읽어요."),
                         ("cv2.minAreaRect(ink)[2]", "글자 픽셀을 감싸는 가장 작은 기운 네모의 각도"), ("cv2.warpAffine(..., borderValue=흰색)", "그 각도만큼 반대로 돌리기"),
                         ("cv2.dilate → cv2.medianBlur(..., 31)", "글자를 지워 배경 밝기만 남긴 그림"), ("cv2.divide(gray, background, scale=255)", "원본을 배경으로 나누면 그림자가 평평해지고 글자만 어둡게 남아요.")]) + K.out("s3img"), "s3"),
    K.step(4, "단어마다 위치와 확신 받기", "글 덩어리 대신 단어마다 글자, 확신, 박스를 받아요.",
           K.code_table([("pytesseract.image_to_data(..., output_type=Output.DICT)", "단어마다 text, conf, left, top, width, height 목록"), ("if t.strip()", "빈 칸(줄, 문단 표시)은 빼요."), ("conf < 90", "확신이 낮은 단어만 골라 확인")]), "s4"),
    K.step(5, "필드 뽑기 (구조화)", "표시 글자(DATE:, TOTAL:) 옆의 값과 항목 줄을 정규식으로 꺼내 dict로 만들어요.",
           K.code_table([("re.search(r\"DATE:\\s*(\\d{4}-\\d{2}-\\d{2})\", text)", "DATE: 뒤에 오는 날짜 모양만"), ("re.findall(r\"^([A-Z]+)\\s+(\\d+)\\s+(\\d+)\\s+(\\d+)\\s*$\", text, re.M)", "줄 전체가 \"영어 단어 숫자 숫자 숫자\"인 줄 = 항목"), ("m.group(1) if m else None", "못 찾으면 None으로 남겨 검산에서 걸리게")]), "s5"),
    K.step(6, "규칙으로 검산하기", "문서 번호 모양, 수량 × 단가 = 금액, 금액 합 = 합계를 확인해요. 하나라도 어긋나면 사람 확인으로 보내요.",
           K.code_table([("re.fullmatch(r\"DOC\\d{2}\", doc_id)", "\"DOC + 숫자 두 자리\"와 정확히 같은지"), ("sum(it[\"amount\"] for it in items) != total", "항목 합과 합계 비교"), ("problems or [\"통과\"]", "문제가 없으면 통과")]), "s6"),
]
practice = [("한국어 문서", "Tesseract는 <code>kor</code> 언어 데이터를 따로 받아야 해요. 한국어는 띄어쓰기 설정(<code>preserve_interword_spaces=1</code>)도 중요해요 (<a href=\"../backend/document-reading.html\">문서 읽기</a> ⑤)."),
            ("VLM으로 읽기", "그림을 보고 답하는 모델(VLM)에게 \"합계가 얼마야?\"라고 물을 수도 있어요. 편하지만 그럴듯하게 지어낼 수 있어서, OCR 글과 박스로 <b>근거가 있는지</b> 확인하고, 안 맞으면 사람에게 보내요 (<a href=\"../../notes/day16.html\">옛 Day16 노트</a>)."),
            ("기기에서 빠르게", "카메라 기기(Jetson 등)에서는 모델을 ONNX로 내보내고 TensorRT로 최적화해서 돌려요. 바꾼 뒤 결과가 같은지 꼭 비교해요."),
            ("검산 규칙", "합계, 날짜 범위, 사업자 번호 형식처럼 문서 안에서 서로 맞아야 하는 값을 찾아 규칙으로 만들어요. 규칙에 걸린 비율을 기록하면 OCR 품질 변화도 보여요.")]
libs = [("①④", "<code>pytesseract.image_to_string</code>, <code>pytesseract.image_to_data</code>, <code>Output.DICT</code>", "OCR (pytesseract + Tesseract)"), ("③", "<code>cv2.resize</code>, <code>cv2.minAreaRect</code>, <code>cv2.warpAffine</code>, <code>cv2.dilate</code>, <code>cv2.medianBlur</code>, <code>cv2.divide</code>, <code>cv2.normalize</code>", "다듬기 (OpenCV)"),
        ("⑤⑥", "<code>re.search</code>, <code>re.findall</code>, <code>re.fullmatch</code>", "값 꺼내기와 검산 (표준 라이브러리)")]
K.build(ROOT / "topics" / "model" / "ocr-documents.html", title="OCR과 문서 구조화", h1="OCR과 문서 구조화: 사진 속 청구서를 값으로", track="모델", crumb="모델 &gt; 이미지",
        one_line="<b>OCR</b>은 그림 속 글자를 글로 바꿔요. 업무에 쓰려면 <b>다듬어서 읽고</b>, 필요한 <b>값을 꺼내고</b>, 서로 맞는지 <b>검산</b>해서, 애매한 문서만 사람이 확인하게 해요.",
        why=why, whole=whole, steps=steps, code_file="examples/ocr/invoice_ocr.py", practice=practice, libs=libs, libs_note="<code>cer</code>, <code>upscale</code>, <code>deskew</code>, <code>unshadow</code>, <code>parse</code>, <code>check</code>는 이 파일 안에서 정의한 함수예요.",
        next_rows=[("<a href=\"opencv-preprocessing.html\">OpenCV 전처리</a>", "다듬기에 쓴 연산들"), ("<a href=\"../backend/document-reading.html\">문서 읽기</a>", "스캔 PDF를 OCR해서 검색 데이터로")],
        sources=["다루는 범위: Day16 (2026-11-09) OCR, 문서 구조화, VLM 근거 확인, 엣지 배포 (VLM과 배포는 실제로 할 때에만)", "데이터: <code>examples/ocr/data/</code> (합성 영어 청구서. 기존 노트와 같은 실습 데이터)", "결과 값: <code>tools/pages/ocr_docs.py</code>가 이 PC의 Tesseract로 실제 실행한 결과",
                 "공식 문서: <a href=\"https://tesseract-ocr.github.io/tessdoc/\" target=\"_blank\">Tesseract</a>, <a href=\"https://pypi.org/project/pytesseract/\" target=\"_blank\">pytesseract</a>"])
print("ok")
