# 엘리스 실행 가이드 — 내 영상 올려서 장면 찾기

**목표**: 웹 브라우저에서 영상을 끌어다 놓고 *"빨간 옷 입은 여자가 냉장고를 여는 장면"* 이라고
쓰면, **몇 초부터 몇 초인지 찾아서 그 부분만 잘라 보여주는** 것.

시간당 과금이므로 **A(무료, 지금)** 를 끝내고 **B(과금 시작)** 로 넘어가세요.

> ## ⚠️ 맥북에서 되는 것 / 안 되는 것
>
> | | 맥북 | 엘리스 |
> |---|---|---|
> | `test_vtg_run.py` / `test_e2e.py` | ✅ | ✅ |
> | 영상 준비 · 질의 작성 | ✅ | ✅ |
> | **`app.py` (웹 UI)** | ❌ **GPU 필요** | ✅ |
> | **`vtg_run.py --model ...`** | ❌ **GPU 필요** | ✅ |
>
> 맥북에서 GPU가 필요한 걸 실행하면 안내를 띄우고 멈춥니다.

---

# A. 지금 (인스턴스 끄고, 무료)

## A-1. 코드 검증 — 5초, 아무것도 설치 안 함

```bash
cd ~/Documents/Video_grounding
python3 test_vtg_run.py     # 77 PASS
python3 test_e2e.py         # 47 PASS
```

**가상환경도, 설치도 필요 없습니다.** 파이썬 표준 라이브러리만 씁니다
(torch·transformers 는 가짜 모듈로 대체해 돌립니다).

`matplotlib` 이 없으면 그림 검증만 **SKIP** 으로 넘어갑니다. 실패가 아닙니다.

## A-2. 내 영상 준비

분석하고 싶은 영상을 정하세요. **길이는 자유** — UI가 알아서 설정을 맞춥니다.

| 조건 | 내용 |
|---|---|
| 형식 | `.mp4` (아니면 `ffmpeg -i in.mov -c:v libx264 out.mp4`) |
| 길이 | 30초 ~ 10분 권장 |
| 해상도 | 무관 (모델이 알아서 줄임) |
| 음성 | 사용 안 함 |

**업로드 전에 압축하면 전송이 빨라집니다.** 화질을 낮춰도 성능에 거의 영향 없어요.

```bash
ffmpeg -i original.mp4 -vf "scale=-2:480" -c:v libx264 -crf 28 -an my_video.mp4
```

## A-3. 질의 문장 만들기 — 여기가 제일 중요합니다

영상을 **직접 보면서** 찾을 장면 3~5개를 정하고 **영어로** 씁니다.
자세한 예시는 **`질의_예시.md`** 를 보세요.

**공식**: `[구체적인 사람/사물] + [명확한 동작]`

```
✅ a woman in a red jacket opens the refrigerator
✅ two people shake hands
✅ a graph appears on the screen for the first time
✅ the camera moves from the desk to the window

❌ a person is walking          (영상 내내 나옴)
❌ 재밌는 부분                    (기준이 없음)
❌ the last scene               (위치를 흘림 — 영상 안 봐도 맞힘)
```

**정답 시각을 메모해두세요.** 모델 결과와 비교해야 맞았는지 알 수 있습니다.

| 질의 | 내가 본 정답 |
|---|---|
| a woman opens the refrigerator | 42초 ~ 47초 |
| two people shake hands | 1분 12초 ~ 1분 15초 |
| … | … |

**일부러 영상에 없는 사건도 하나 넣으세요.** (예: 개가 없는데 `a dog runs across the room`)
모델이 "없다"고 못 하고 아무 구간이나 답하는 걸 보여주는 용도입니다 — 발표에서 좋은 소재예요.

## A-4. 엘리스 계정

- 결제 수단 등록 확인
- 잔액/크레딧 (**₩3,000 내외** 예상 — 웹 UI만 쓸 경우)

---

# B. 인스턴스 켠 뒤 (여기서부터 과금)

## B-1. 인스턴스 생성

| 항목 | 선택 |
|---|---|
| GPU | **G-NAHPM-40** (A100 MIG 3g-40GB, ₩1,380/시간) |
| 스토리지 | **128 GiB** (₩20/시간) |
| 실행 환경 | **VSCode (CUDA 12.8)** |

**실행 환경은 반드시 `VSCode (CUDA 12.8)`.**

| 선택지 | 판단 |
|---|---|
| **VSCode (CUDA 12.8)** | ✅ `setup_env.sh` 의 torch cu128 과 일치. 파일 드래그 업로드 + 터미널 |
| VSCode (CUDA 11.8) | ❌ cu128 휠이 안 돌아감 |
| VSCode / Jupyter (버전 없음) | ⚠️ CUDA 버전 불명확 |
| SSH-Only | ⚠️ 되지만 키 발급·포트 설정이 추가로 필요 |

## B-2. 파일 올리기

VSCode가 브라우저에서 열립니다. **왼쪽 탐색기에 끌어다 놓으면 끝.**

올릴 것:

```
Video_grounding/          ← 폴더 통째로
my_video.mp4              ← 내 영상
```

> **`videos/_bench/` 는 올리지 마세요.** 수 GB짜리 벤치마크 영상이라
> 업로드가 오래 걸립니다. 필요하면 인스턴스에서 다시 받는 게 훨씬 빠릅니다.

SSH를 쓰신다면 (콘솔 → `내 인스턴스` → `다른 SSH 클라이언트 사용` 에서 접속 정보 확인):

```bash
# scp 는 포트가 대문자 -P 입니다 (ssh 는 소문자 -p)
scp -i ~/키.pem -P 포트 -r ~/Documents/Video_grounding 계정@주소:~/
```

## B-3. 환경 구성 (약 10분)

VSCode 안의 터미널에서:

```bash
cd ~/Video_grounding
bash setup_env.sh
source ~/vtg-env/bin/activate
```

마지막에 **`>>> 환경 정상 <<<`** 이 뜨면 성공입니다.

중간에 "모델을 지금 받을까요?" 물으면 **`y`** 를 누르세요 (약 26GB, 10분).
지금 받아두면 UI 첫 실행이 바로 시작됩니다.

```bash
sudo apt-get install -y ffmpeg     # 없다고 나오면
```

> 가상환경을 활성화하면 `python` 과 `python3` 가 둘 다 됩니다.
> 활성화 전이나 맥북에서는 **`python3`** 를 쓰세요.

## B-4. 웹 UI 띄우기 ← 핵심

```bash
python3 app.py
```

VSCode 터미널에 **포트 포워딩 알림**이 뜹니다. 클릭하면 브라우저에서 열려요.
안 뜨면 `포트` 탭에서 `7860` 을 열거나, 아래처럼 공개 링크를 만드세요.

```bash
python3 app.py --share      # 임시 공개 링크 (외부·발표 자리에서 접속할 때)
```

## B-5. 사용법

```
① 왼쪽에 영상을 끌어다 놓는다
② "찾을 장면" 에 영어 문장을 쓴다     ← A-3 에서 준비한 것
③ 모델을 고른다 (기본 TimeLens-8B 권장)
④ [장면 찾기] 를 누른다
```

나오는 것:

| 결과 | 내용 |
|---|---|
| **구간** | `42.30초 ~ 47.10초 (길이 4.80초)` |
| **타임라인** | 영상 전체에서 어디인지 막대로 표시 |
| **잘린 클립** | 그 구간만 잘라서 재생 (앞뒤 0.5초 여유) |
| 원본 출력 | 모델이 실제로 뱉은 문장 (접혀 있음) |

**첫 실행은 모델 다운로드로 몇 분 걸립니다.** 두 번째부터는 몇 초예요.
같은 모델을 계속 쓰면 다시 안 올립니다. **모델을 바꾸면 다시 올리느라 1~2분** 걸립니다.

영상 길이에 따라 설정이 자동으로 잡힙니다.

| 영상 길이 | 토큰 예산 | fps |
|---|---|---|
| ~1분 | 32768 | 2 |
| 1~3분 | 16384 | 2 |
| 3~10분 | 16384 | 1 |
| 10분+ | 8192 | 1 |

## B-6. (선택) 정량 평가 — 숫자가 필요하면

내 영상은 정답이 없어서 "맞아 보인다"까지만 됩니다.
**논문 수치와 비교할 숫자**가 필요하면 정답이 딸린 벤치마크 영상을 씁니다.

```bash
# 사람이 검수한 질의 + 정답이 붙은 영상 (샤드 1개 = 6.5GB)
python3 run_experiments.py --download-bench --bench-dataset charades

# 긴 영상으로 하려면 (최대 4분대, 샤드 1개 = 8.4GB)
python3 run_experiments.py --download-bench --bench-dataset activitynet --bench-longest

python3 run_experiments.py --list-videos    # 어떤 질의/정답이 붙었는지 확인
python3 run_experiments.py --model safe     # 실행 → results/report.md
```

`report.md` 에 **평균 tIoU / R1@0.5 / R1@0.7** 표가 나옵니다.
질의를 직접 쓸 필요 없어요 — 이미 들어 있습니다.

**학습 전/후 비교**도 한 줄이면 됩니다. 발표에서 설득력이 큽니다.

```bash
python3 run_experiments.py --model ablation   # Qwen3-VL-8B(원본) vs TimeLens-8B
```

## B-7. 마무리

```bash
# 결과 파일이 있으면 VSCode 탐색기에서 우클릭 → Download
# 또는
scp -i ~/키.pem -P 포트 -r 계정@주소:~/Video_grounding/results ./
```

**내려받은 뒤 반드시 인스턴스를 삭제하세요.** 켜두면 계속 과금됩니다.

---

# 예상 시간 / 비용

**웹 UI만 쓰는 경우 (권장 코스)**

| 단계 | 시간 | 누적 |
|---|---|---|
| 업로드 + 환경 구성 | 15분 | ₩350 |
| 모델 다운로드 | 10분 | ₩600 |
| UI로 이것저것 시험 | 1시간 | ₩2,000 |

**정량 평가까지 하는 경우** 벤치마크 다운로드 30분 + 실험 30분 = **₩3,500** 추가.

---

# 문제가 생기면

| 증상 | 대응 |
|---|---|
| `GPU(CUDA)를 찾을 수 없습니다` | 맥북에서 실행 중. 인스턴스에서 하세요 |
| `no kernel image is available` | torch 가 cu124. CUDA 12.8 이미지인지 확인 후 `setup_env.sh` 재실행 |
| `KeyError: 'qwen3_vl'` | `pip install -U "transformers>=4.57.0,<5.0"` |
| `unexpected keyword 'return_video_metadata'` | `pip install -U "qwen-vl-utils[decord]>=0.0.14"` |
| `CUDA out of memory` | 더 짧은 영상으로 시도하거나 `timelens2-4b` 모델 선택 |
| UI가 안 열림 | `python3 app.py --share` 로 공개 링크 생성 |
| 클립이 안 잘림 | `sudo apt-get install -y ffmpeg` (구간 숫자는 그래도 나옵니다) |
| flash-attn 설치 실패 | 무시. 기본 `sdpa` 로 충분 |
| 다운로드가 느림 | `export HF_HUB_ENABLE_HF_TRANSFER=1` (setup 이 이미 설정) |
| `구간을 찾지 못했습니다` | 문장이 모호함. `질의_예시.md` 참고해 더 구체적으로 |

에러 메시지 전문과 어느 단계였는지를 남겨두면 바로 고칠 수 있습니다.

---

# 발표 시나리오 제안

1. **UI에 영상을 올리고 질의 하나** → 구간 + 잘린 클립을 보여준다
2. **같은 사건을 두 가지 정밀도로** 물어본다
   → `a person sits down` vs `a person sits down on a wooden chair near the window`
   → 결과가 달라지면 **모델이 문장을 읽고 있다**는 증거
3. **영상에 없는 사건**을 물어본다
   → 아무 구간이나 답함 → **현재 기술의 한계** (no-target rejection)
4. **모델을 `Qwen3-VL-8B (학습 전 원본)` 으로 바꿔** 같은 질의를 다시
   → 결과가 나빠짐 → **왜 특화 학습이 필요한가**를 눈으로 보여줌
5. 숫자가 필요하면 `results/report.md` 의 tIoU 표를 띄운다
