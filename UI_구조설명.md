# 웹 UI 는 어떻게 만들었나 (`app.py`)

**Gradio** 라는 파이썬 라이브러리로 만들었습니다. HTML/CSS/JS 를 한 줄도 안 씁니다.

```bash
pip install gradio
```

파이썬 함수 하나만 있으면 웹 화면이 자동으로 생깁니다. 머신러닝 데모의 사실상 표준이에요.

---

## 가장 단순한 형태

```python
import gradio as gr

def 인사(이름):
    return f"안녕하세요 {이름}님"

gr.Interface(fn=인사, inputs="text", outputs="text").launch()
```

이 6줄이면 입력창 + 버튼 + 출력창이 있는 웹페이지가 뜹니다.

---

## 우리 UI 의 뼈대

`app.py` 는 `gr.Blocks` 를 씁니다. 화면 배치를 직접 정할 수 있는 방식입니다.

```python
with gr.Blocks() as demo:                 # 화면 전체
    gr.Markdown("# 영상에서 장면 찾기")     # 제목

    with gr.Row():                         # 좌우 2단
        with gr.Column():                  # ── 왼쪽: 입력 ──
            picker = gr.Dropdown(...)      # 서버 영상 목록
            video  = gr.Video(...)         # 영상 재생 / 업로드
            qpick  = gr.Dropdown(...)      # 준비된 질의
            query  = gr.Textbox(...)       # 찾을 장면
            model  = gr.Dropdown(...)      # 모델 선택
            btn    = gr.Button("장면 찾기")

        with gr.Column():                  # ── 오른쪽: 결과 ──
            out_md   = gr.Markdown()       # 구간 텍스트
            out_png  = gr.Image()          # 타임라인
            out_clip = gr.Video()          # 잘라낸 구간
            out_raw  = gr.Textbox()        # 모델 원본 출력

    btn.click(run, [video, query, model],  # 버튼 → 함수 연결
              [out_md, out_png, out_clip, out_raw])

demo.launch()
```

**`btn.click(함수, 입력들, 출력들)`** 이 핵심입니다.
버튼을 누르면 왼쪽 칸들의 값이 `run()` 에 들어가고, 반환값이 오른쪽 칸에 순서대로 꽂힙니다.

---

## 실제로 일하는 함수

```python
def run(video, query, model_label):
    dur = 영상길이(video)
    tokens, fps = budget_for(dur)          # 길이 보고 설정 자동 결정
    g = get_grounder(모델키, tokens, fps)   # 모델 준비 (캐시됨)
    r = g(video, query)                    # 추론

    md   = f"**1. {시작}초 ~ {끝}초**"      # 텍스트
    png  = timeline_png(dur, r["spans"])   # matplotlib 로 막대 그림
    clip = cut_clip(video, 시작, 끝)        # ffmpeg 로 그 구간만 잘라냄
    return md, png, clip, r["raw"]
```

반환하는 **4개가 오른쪽 4칸에 순서대로** 들어갑니다.

---

## 신경 쓴 부분 4가지

### ① 모델을 매번 다시 안 올림

8B 모델을 GPU 에 올리는 데 1~2분 걸립니다. 요청마다 하면 못 씁니다.

```python
_STATE = {"key": None, "g": None}          # 전역에 보관

def get_grounder(key, ...):
    if _STATE["key"] == key:               # 같은 모델이면
        return _STATE["g"]                 #   그대로 재사용
    if _STATE["g"] is not None:
        _STATE["g"].free()                 # 다르면 이전 것 GPU 에서 내리고
    _STATE.update(key=key, g=Grounder(key))#   새로 올림
    return _STATE["g"]
```

40GB VRAM 에 8B 두 개를 동시에 못 올려서 **하나씩 갈아끼우는** 구조입니다.

### ② 영상 길이에 맞춰 설정 자동 조절

```python
def budget_for(duration):
    if duration <= 60:  return 32768, 2.0   # 짧으면 고화질
    if duration <= 180: return 16384, 2.0
    if duration <= 600: return 16384, 1.0
    return 8192, 1.0                        # 길면 아껴야 OOM 안 남
```

사용자가 토큰 예산 같은 걸 몰라도 되게 만들었습니다.

### ③ 영상 고르면 질의가 자동으로 채워짐

```python
picker.change(on_pick_video, picker, [video, qpick, hint])
qpick.change(on_pick_query, qpick, query)
```

`.change()` 는 값이 바뀔 때 함수를 부릅니다.
영상 선택 → `external.json` 에서 그 영상의 질의 목록을 읽어 드롭다운 채움 →
질의 클릭 → 텍스트창에 입력.

### ④ 실패해도 안 죽음

```python
try:
    clip = cut_clip(...)
except Exception:
    clip = None                            # ffmpeg 없으면 클립 칸을 숨김
```

`ffmpeg` 이 없어도 구간 숫자와 타임라인은 정상적으로 나옵니다.

---

## 진행 상황 표시

```python
def _run(v, q, m, progress=gr.Progress()):
    return run(v, q, m, progress)

# run() 안에서
progress(0.1, desc="모델 준비 중…")
progress(0.5, desc="영상 분석 중…")
```

`gr.Progress()` 를 인자로 받으면 Gradio 가 알아서 진행 바를 그려줍니다.

---

## 서버 영상 vs 업로드

처음엔 업로드만 됐는데 문제가 있었습니다 —
**인스턴스에서 받은 영상은 브라우저가 접근을 못 합니다.** 서버 파일이니까요.

그래서 드롭다운을 추가했습니다.

```python
def server_videos():
    return ["— 직접 올리기 —"] + sorted(str(p) for p in Path("videos").glob("*.mp4"))
```

- 서버 영상 → 드롭다운에서 선택 (경로를 `gr.Video` 에 직접 넣음)
- 내 컴퓨터 영상 → 끌어다 놓기 (브라우저가 업로드)

---

## 실행 방법

```bash
python3 app.py            # http://<주소>:7860
python3 app.py --share    # 임시 공개 링크 (72시간)
```

```python
demo.launch(server_name="0.0.0.0",   # 외부에서 접속 허용
            server_port=7860,
            share=args.share)         # Gradio 터널 사용
```

**`--share` 를 권하는 이유**: VSCode 포트 포워딩을 거치면 Gradio 의 CSS/JS 경로가
틀어져서 화면이 깨집니다. `--share` 는 Gradio 자체 터널이라 정상적으로 보입니다.

---

## 정리

| | |
|---|---|
| 라이브러리 | Gradio (파이썬만으로 웹 UI) |
| 코드 길이 | 약 280줄 (`app.py`) |
| HTML/CSS/JS | **0줄** |
| 핵심 구조 | `gr.Blocks` 로 배치 → `btn.click(함수, 입력, 출력)` 으로 연결 |
| 추론 로직 | `vtg_run.py` 를 import 해서 재사용 (UI 는 껍데기) |

**`app.py` 는 화면만 담당합니다.** 모델 로딩·프롬프트·출력 파싱은 전부 `vtg_run.py`
에 있고, UI 는 그걸 가져다 쓰는 구조예요. 그래서 CLI(`vtg_run.py`)와 일괄 실험
(`run_experiments.py`)이 **같은 코드를 공유**합니다.

> 단일 파일 버전도 있습니다 — `demo.py` (235줄). `vtg_run.py` 없이 혼자 돌아갑니다.
