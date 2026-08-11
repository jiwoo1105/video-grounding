# Video Grounding — 왜 백본을 그냥 쓰지 않는가

> TimeLens (CVPR 2026) 논문의 실험 결과를 근거로 정리. 2026년 8월 기준.
> Notion에 그대로 드래그하면 표까지 변환됩니다.

---

## 0. 세 줄 요약

1. 원본 백본(Qwen2.5-VL, Qwen3-VL)은 **"몇 초"를 정확히 말하는 훈련을 받은 적이 없습니다.**
2. 기존 벤치마크는 **라벨 자체가 부정확**해서, 거기 맞춰 학습한 모델은 점수만 높고 실력은 낮습니다.
3. 그래서 **깨끗한 데이터로 다시 학습시킨 모델**(TimeLens / TimeLens2)이 필요합니다.

---

## 1. 첫 번째 근거 — 벤치마크가 오염돼 있었다

TimeLens 논문 Figure 1(a). 가로축은 **기존 벤치마크**, 세로축은 **정제한 벤치마크** 점수입니다.

```
정제 벤치마크 ↑
   높음 │  ● Gemini-2.5-Pro
        │  ● GPT-5, GPT-4o          ← 프로프라이어터리 (파란 영역)
        │       ╲  Equal Scores
        │        ╲
   낮음 │         ╲   ▲ Time-R1-7B
        │          ╲  ▲ Qwen2.5-VL-7B  ← 오픈소스 (빨간 영역)
        └──────────────────────────→
           낮음        높음   기존 벤치마크
```

**대각선 위/아래로 완전히 갈립니다.**

| 그룹 | 기존 벤치마크 | 정제 벤치마크 | 해석 |
|---|---|---|---|
| GPT/Gemini | 낮음 | **높음** | 해당 데이터로 학습한 적 없음. 실력대로 나옴 |
| 오픈소스 | **높음** | 낮음 | 기존 데이터로 학습 → **틀린 라벨까지 외워버림** |

### 벤치마크가 뭔지부터

**벤치마크 = 채점용 문제집**입니다. `영상 + 질의 문장 + 정답 구간` 세트를 수천 개 모아둔 것.
이 분야에서 10년 가까이 표준으로 쓰인 게 세 개입니다.

| 이름 | 영상 종류 | 평균 길이 | 예시 질의 |
|---|---|---|---|
| **Charades-STA** | 집 안 일상 행동 | 약 30초 | "person sitting down in a chair" |
| **ActivityNet Captions** | 유튜브 다양한 활동 | 약 2분 | "the man begins to play the guitar" |
| **QVHighlights** | 유튜브 브이로그·뉴스 | 약 2.5분 | "a woman shows her new apartment" |

논문들이 말하는 "Charades-STA에서 60.8 달성" 같은 건 **이 문제집으로 채점한 점수**예요.

### 기존 vs 정제 — 뭐가 다른가

**영상도 같고 질의 문장도 같습니다. 정답 구간(라벨)만 다릅니다.**

```
기존 (Charades-STA)          정제 (Charades-TimeLens)
영상: OHOFG.mp4       →      영상: OHOFG.mp4        (동일)
질의: "person sits"   →      질의: "person sits"    (동일)
정답: 1.0 ~ 7.5초     →      정답: 2.3 ~ 6.8초      ← 여기만 바뀜
```

TimeLens 팀이 세 데이터셋을 전부 다시 검수해서 **사람이 직접 라벨을 고쳤습니다.**

| 정제본 | 영상 수 | 평균 길이 | 주석 수 | 원본 |
|---|---|---|---|---|
| Charades-TimeLens | 1,313 | 29.6초 | 3,363 | Charades-STA |
| ActivityNet-TimeLens | 1,455 | 134.9초 | 4,500 | ActivityNet Captions |
| QVHighlights-TimeLens | 1,511 | 149.6초 | 1,541 | QVHighlights |

기존 라벨의 전형적인 문제들:

- **경계가 대충** — 사람이 눈대중으로 찍어서 1~2초씩 어긋남
- **질의가 모호** — "a person walks"가 영상에 세 번 나오는데 정답은 하나만 표시
- **문장과 내용 불일치** — 실제로 그런 장면이 없는데 라벨이 붙어 있음

### 그래서 그림 (a)가 의미하는 것

- **가로축(Original)** = 기존 라벨로 채점한 점수
- **세로축(Refined)** = 사람이 고친 라벨로 채점한 점수

오픈소스 모델들은 **기존 데이터로 파인튜닝했습니다.** 그러니 틀린 라벨의 버릇까지
같이 배웠고, 기존 문제집에선 고득점이 나옵니다. 라벨을 고쳐놓으면 점수가 무너지죠.

GPT/Gemini는 이 데이터셋으로 학습한 적이 없습니다. 그래서 기존 문제집에선
"틀린 정답"을 못 맞혀 점수가 낮지만, 정제본에선 실력대로 나옵니다.

**비유하면** — 오답이 섞인 기출문제집을 통째로 외운 학생 vs 안 외운 학생.
기출로 시험 보면 외운 쪽이 이기지만, 정답을 고쳐서 다시 내면 뒤집힙니다.

> 💡 발표할 때 좋은 포인트 — 이건 단순히 "우리 모델이 좋다"가 아니라
> **"이 분야가 지금까지 잘못된 자로 재고 있었다"** 는 지적입니다.
> TimeLens 논문의 진짜 기여가 모델이 아니라 이 문제 제기예요.
> (데이터셋 카드 원문: *"we identified critical quality issues within existing datasets
> and performed extensive manual corrections... a dramatic re-ranking of models"*)

> ⚠️ **제가 앞서 인용한 "Time-R1 Charades-STA R1@0.5 = 60.8"도 여기 해당합니다.**
> Time-R1은 Charades에 파인튜닝한 뒤 Charades로 평가한 수치예요.
> 정제 벤치마크(Charades-TimeLens)에서는 **R1@0.5 = 32.0, mIoU = 36.6** 으로 떨어집니다.
> 논문 간 수치를 비교할 때 반드시 **어느 벤치마크인지** 확인해야 합니다.

---

## 2. 두 번째 근거 — 원본 백본은 실제로 못한다

TimeLens 논문 Table 1 (정제 벤치마크 mIoU 기준)

| 모델 | Charades | ActivityNet | QVHighlights | 평균 |
|---|---|---|---|---|
| Qwen2.5-VL-7B **(원본)** | 39.3 | 31.4 | 31.6 | **34.1** |
| **TimeLens-7B** (학습 후) | 48.8 | 46.2 | 56.0 | **50.3** |
| Qwen3-VL-8B **(원본)** | 48.3 | 46.8 | 59.4 | — |
| **TimeLens-8B** (학습 후) | 55.2 | 53.2 | 65.5 | **58.0** |
| Qwen3-VL-235B-A22B (원본, 거대) | 47.8 | 52.2 | 64.6 | — |

읽어낼 점 세 가지:

**① 같은 백본, 학습만 얹었을 때 평균 34.1 → 50.3**
7B가 +16.2점. 파라미터는 1도 안 늘었습니다.

**② 235B 거대 백본이 8B 학습 모델에 진다**
Qwen3-VL-235B가 Charades 47.8인데 TimeLens-8B는 55.2. **30배 작은데 이깁니다.**
"백본을 더 큰 걸로 바꾸면 되지 않나"에 대한 답이 이겁니다 — 규모로 안 됩니다.

**③ 7B 학습 모델이 8B 원본과 맞먹는다**
TimeLens-7B(50.3) ≈ Qwen3-VL-8B 원본 수준. 즉 **학습 한 번이 백본 한 세대를 건너뛰는 효과**를 냅니다.

> 🔍 **다만 정직하게 짚을 점** — Gemini-2.5-Pro는 평균 **60.4**로 TimeLens-8B(58.0)보다 위입니다.
> 논문도 "GPT-5와 Gemini-2.5-Flash를 넘어섰다"고만 표현하고 Pro는 언급하지 않습니다.
> 오픈소스 중 1위이지, 전체 1위는 아닙니다. 이건 발표할 때 미리 밝히는 게 좋습니다.
> (제가 앞서 "Gemini가 7B에 진다"고 한 건 Time-R1 자체 벤치마크 기준이라 편향이 있었습니다.)

---

## 3. 세 번째 근거 — 무엇이 성능을 올렸나

TimeLens 논문 Figure 1(b). TimeLens-7B를 만들어가는 과정의 누적 개선입니다.

| 단계 | 평균 mIoU | 증분 |
|---|---|---|
| Baseline (노이즈 있는 학습 데이터) | 37.2 | — |
| **+ 개선된 학습 데이터** | 43.2 | **+6.0** ← 최대 |
| + Interleaved Textual Timestamp | 45.8 | +2.6 |
| + Thinking-Free RLVR | 47.7 | +1.9 |
| + Early Stopping | 48.2 | +0.5 |
| + Difficulty-based Data Sampling | 49.4 | +1.2 |
| + Scale Up Resolution | **50.3** | +0.9 |

**가장 큰 기여는 알고리즘이 아니라 데이터 품질입니다** (+6.0). 이 분야에서 "모델을 어떻게 설계하느냐"보다 "무엇을 학습시키느냐"가 중요하다는 걸 보여주는 표예요.

---

## 4. 네 번째 근거 — 타임스탬프를 어떻게 넣느냐가 갈랐다

TimeLens 논문 Figure 5 / Table 2. 모델에게 "이 프레임이 몇 초인지" 알려주는 세 가지 방식입니다.

```
(a) Interleaved Textual   [1s] 🖼 [2s] 🖼 [3s] 🖼   시간을 텍스트 토큰으로 프레임 사이에 끼움
(b) Visual Overlay          🖼¹ˢ  🖼²ˢ  🖼³ˢ        영상 위에 숫자를 그려 넣음
(c) Position Embedding     RoPE ID: 0, 4, 8 …      위치 인코딩을 촬영 시각에 맞춤
```

Charades-TimeLens mIoU 결과:

| 방식 | 형식 | mIoU |
|---|---|---|
| **(c) Position Embedding** | — | **36.6** ← 최하 |
| (b) Visual Overlay | Frame Index | 44.0 |
| (b) Visual Overlay | Raw Timestamp | 46.3 |
| (a) Not-Interleaved Textual Prefix | Raw Timestamp | 45.8 |
| (a) Interleaved Textual Prefix | Frame Index | 45.6 |
| **(a) Interleaved Textual Prefix** | **Raw Timestamp** | **48.3** ← 최고 |

> ⚠️ **제가 앞서 한 설명을 정정합니다.**
> "Qwen2.5-VL이 절대 시간을 mRoPE에 인코딩해서 TVG에 유리하다"고 말씀드렸는데,
> 그게 바로 (c) Position Embedding 방식이고 **실험 결과 가장 나쁩니다** (36.6).
> 최고(48.3)와 **11.7점 차이**예요.
>
> 이유를 추측하면 — 위치 인코딩은 모델 내부의 암묵적 신호라 "12.4초"라는 숫자를
> 출력으로 뽑아내려면 한 번 더 변환이 필요합니다. 반면 텍스트로 `12.4s`라고
> 직접 써주면 LLM이 이미 잘하는 "텍스트 복사·연산" 능력을 그대로 쓸 수 있습니다.
>
> 두 가지 교훈: (1) 프레임 사이에 **끼워 넣는(interleaved)** 게 앞에 몰아넣는 것보다 낫다.
> (2) 프레임 번호(1, 2, 3)보다 **실제 초(10.2s)** 를 줘야 한다.

---

## 5. 결론 — 뭘 쓸 것인가

| 용도 | 모델 | 근거 |
|---|---|---|
| **발표·설명 주력** | `TencentARC/TimeLens-8B` | **CVPR 2026** 게재. 위 그림들이 전부 이 논문 자료라 설명이 자연스러움 |
| **성능 최우선** | `MCG-NJU/TimeLens2-8B` | 7벤치 평균 mIoU 48.0, 백본 대비 +18.1 |
| **가성비 / 긴 영상** | `MCG-NJU/TimeLens2-4B` | 평균 47.7 — 8B와 0.3 차이인데 절반 크기 |
| **대조군** | `Qwen/Qwen3-VL-8B-Instruct` | "학습이 얼마나 기여했나"를 직접 보여줌 |

### 권장 구성

**TimeLens-8B를 주력으로 삼으세요.** 이유:

1. 이 문서의 근거 그림 4개가 전부 이 논문에서 나옵니다 → **설명이 논문과 1:1로 붙습니다**
2. CVPR 2026 게재 = 피어리뷰 통과. 인용해도 안전
3. 성능도 TimeLens2와 크게 차이 안 남 (Charades mIoU 55.2 vs 58.6)

**TimeLens2는 "그 이후 최신 동향"으로 곁들이면 됩니다.** 실제로 TimeLens2는 TimeLens의 데이터셋과 GRPO 코드를 이어받은 후속 연구라, 스토리가 자연스럽게 연결돼요.

```bash
python vtg_run.py --model safe     ...   # TimeLens-8B    발표 주력
python vtg_run.py --model best     ...   # TimeLens2-8B   최신
python vtg_run.py --model ablation ...   # Qwen3-VL-8B vs 학습 모델
```

---

## 6. 백본은 어떻게 학습시키는가

TimeLens-8B의 실제 3단계 파이프라인입니다 (공식 레포 기준).

```
Qwen3-VL-8B-Instruct  (알리바바가 만든 원본)
        │
        │  [1단계] SFT — 정답을 보여주고 따라 하게
        │           데이터: TimeLens-100K 중 3만 개 샘플
        ▼
   SFT 체크포인트
        │
        │  [2단계] 필터링 — SFT 모델로 전체 10만 개를 직접 풀어보고
        │           IoU를 계산해 "너무 쉬운 것/너무 어려운 것"을 걸러냄
        ▼
   난이도 선별된 데이터
        │
        │  [3단계] GRPO — 강화학습으로 정밀도를 끌어올림
        ▼
   TimeLens-8B  (최종 배포본)
```

### 1단계 SFT (Supervised Fine-Tuning)

**"이 영상 + 이 문장 → 정답은 12.4~18.7초" 를 수만 번 보여주고 따라 하게** 합니다.

```
입력:  영상 + "A man drinks water with a glass"
정답:  "The event happens in 12.40 - 18.70 seconds"
       ↑ 모델이 이 문장을 그대로 뱉도록 가중치를 조금씩 수정
```

이 단계에서 **출력 형식**이 결정됩니다. Time-R1은 `<answer>12.4 to 18.7</answer>`,
TimeLens는 `The event happens in ...`, TimeLens2는 JSON 배열로 학습시킨 거예요.
그래서 **프롬프트를 서로 바꿔 쓰면 성능이 무너집니다** — 배운 적 없는 형식이니까요.

한계: SFT는 "정답을 외우는" 학습이라, 정답이 부정확하면 그 부정확함까지 같이 배웁니다.
1번 항목에서 본 오픈소스 모델들의 문제가 바로 이겁니다.

### 2단계 필터링

SFT 모델에게 학습 데이터 10만 개를 **직접 풀게 시킵니다.** 그리고 IoU를 계산해서:

- 항상 맞히는 문제 → 더 배울 게 없음 → **제외**
- 절대 못 맞히는 문제 → 라벨이 이상할 가능성 → **제외**
- 가끔 맞히는 문제 → **여기서 배울 게 가장 많음** → 남김

이게 Figure 1(b)의 `+ Difficulty-based Data Sampling` (+1.2점)입니다.

### 3단계 GRPO (강화학습)

SFT가 "따라 하기"라면 GRPO는 **"여러 번 시도하고 잘한 걸 강화하기"** 입니다.

```
같은 문제에 대해 답을 8개 생성
     예측 A: 12.0~19.0  →  IoU 0.85  →  보상 높음  ↑ 이렇게 답하도록 강화
     예측 B: 30.0~35.0  →  IoU 0.00  →  보상 낮음  ↓ 이렇게 답하지 않도록
     ...
그룹 내 상대 순위로 가중치를 조정
```

**"Verifiable Reward"의 의미** — 채점을 사람이 안 해도 됩니다. IoU는 계산식이라
정답 구간만 있으면 자동으로 점수가 나와요. 그래서 대규모 RL이 가능합니다.

**Thinking-Free RLVR** (Figure 1(b), +1.9점)
Time-R1은 `<think>` 안에서 추론 과정을 길게 생성한 뒤 답합니다. TimeLens는
그 과정을 **없앴는데 오히려 성능이 올랐습니다.** 타임스탬프 회귀는 긴 논리 전개가
필요한 문제가 아니라, 생각을 길게 할수록 오히려 헤맨다는 뜻입니다.

**TimeLens2가 추가한 것 — Temporal Wasserstein Reward**

IoU만 쓰면 구멍이 있습니다:

```
정답:     [====]        30~40초
예측 A:          [==]   50~55초   → IoU = 0
예측 B:                    [==]   90~95초  → IoU = 0
```

A가 훨씬 나은 답인데 **점수가 똑같이 0**입니다. 방향 신호가 없어요.
Wasserstein 거리는 "얼마나 멀리 빗나갔나"를 연속값으로 주므로 A를 더 높게 칩니다.

TimeLens2 논문에 따르면 **tIoU가 전부 0인 그룹의 75.8%가 이 보상 덕에 유효한 순서**를 얻고,
보상이 전부 같아 학습이 안 되던 그룹이 **13.8% → 3.6%** 로 줄었습니다.

### 우리는 이걸 하지 않습니다

위 3단계는 **GPU 수십 장으로 며칠** 걸리는 작업입니다.
우리는 그 결과물(체크포인트)을 다운로드해서 **추론만** 돌립니다 — GPU 1장, 몇 초.

---

## 7. 세 프로젝트 구분

이름이 헷갈리기 쉬운데 **Time-R1과 TimeLens는 아무 관계 없습니다.**

| 프로젝트 | 팀 | 발표 | 백본 | 출력 형식 |
|---|---|---|---|---|
| **Time-R1** | Xiaomi + 중국인민대 | NeurIPS'25 | Qwen2.5-VL 3B/7B | `<answer>1.05 to 7.62</answer>` |
| **TimeLens** | TencentARC | **CVPR 2026** | Qwen2.5-VL-7B, Qwen3-VL-8B | `The event happens in 1.0 - 7.6 seconds` |
| **TimeLens2** | 난징대 MCG 외 | arXiv 2026-07 | Qwen3-VL 2B/4B/8B | `[[1.2, 4.5], [10.0, 13.7]]` |

연결고리는 두 가지뿐입니다.

- TimeLens2가 TimeLens의 GRPO 코드와 TimeLens-100K 데이터를 이어받음
- **Limin Wang** 교수가 양쪽 저자 (TSN, VideoMAE, InternVideo를 만든 비디오 이해 분야 대표 연구자)

**검색 주의** — 구글 스칼라에서 "Time lens"를 찾으면 무관한 논문이 섞입니다.
`Time Lens: Event-based video frame interpolation`(CVPR 2021)은 **이벤트 카메라 프레임 보간**,
Optics Letters/APL의 "optical time lens"는 **광학** 논문입니다. 원래 광학 용어예요.

---

## 8. Qwen 계보 (참고)

### 이름 읽는 법

| 표기 | 뜻 |
|---|---|
| `Qwen3` | 텍스트 전용 |
| `Qwen3-VL` | **V**ision-**L**anguage. 이미지·영상 입력 가능 |
| `Qwen3-Omni` | + 음성 입출력 |
| `-Instruct` / `-Thinking` | 지시 수행용 / 추론 과정 생성형 |
| `-Max`, `-Plus` | 알리바바 클라우드 전용 (가중치 비공개) |
| `30B-A3B` | 전체 300억 중 **3B만 활성** (MoE) |

### 타임라인

| 시점 | 메인 라인 | VL 라인 |
|---|---|---|
| 2023 | Qwen | Qwen-VL (7B) |
| 2024 | Qwen2 → Qwen2.5 | Qwen2-VL (2/7/72B) |
| 2025-01 | | **Qwen2.5-VL** (3/7/32/72B) |
| 2025-04 | **Qwen3** | |
| 2025-09~11 | Qwen3-Max (1T, 비공개) | **Qwen3-VL** (2/4/8/32B + MoE) |
| 2026-02 | **Qwen3.5** — 비전 인코더 내장 | — |
| 2026-04 | **Qwen3.6** | — |
| 2026-05~08 | Qwen3.7-Max → Qwen3.8-Max (2.4T) | — |

> **핵심** — Qwen3.5부터 비전이 메인 라인에 내장되면서 `-VL` 브랜치가 사라졌습니다.
> **"Qwen3.5-VL", "Qwen3.6-VL"은 존재하지 않습니다.** Qwen3-VL이 마지막 VL 시리즈예요.
> 즉 "더 최신 백본을 쓰자"는 선택지 자체가 없습니다.

---

## 9. 환경 호환성

| transformers | Qwen2.5-VL | Qwen3-VL |
|---|---|---|
| ~4.56 | ✅ | ❌ |
| **4.57 ~ 4.9x** | ✅ | ✅ ← **여기로 고정** |
| 5.x | ⚠️ 인자명 변경 등 breaking change | ✅ |

```bash
pip install "transformers>=4.57.0,<5.0" "qwen-vl-utils[decord]>=0.0.14"
```

- Time-R1 공식 `requirements.txt`는 `transformers==4.51.1` 고정 → **Qwen3-VL 로드 불가**
- Qwen3.5/3.6은 아키텍처가 달라(Gated DeltaNet) 더 최신 버전 필요 → 위 구간과 충돌

---

## 10. 용어

| 용어 | 뜻 |
|---|---|
| **백본 (backbone)** | 사전학습된 범용 기반 모델. 우리 작업엔 아직 특화 안 됨 |
| **체크포인트** | 학습이 끝난 가중치 파일. 실제 다운로드받는 그것 |
| **SFT** | 정답을 보여주고 따라 하게 하는 지도학습 |
| **RLVR / GRPO** | 자동 채점 가능한 보상(IoU)으로 여러 답 중 잘한 걸 강화하는 강화학습 |
| **mIoU** | 예측 구간과 정답 구간이 겹치는 비율의 평균 |
| **R1@0.5** | 상위 1개 예측의 IoU가 0.5를 넘는 비율(%) |
| **Interleaved Textual Timestamp** | 프레임 사이에 `12.4s` 같은 시각을 **텍스트로 끼워 넣는** 방식. 현재 최선 |
| **Temporal Wasserstein Reward** | IoU=0일 때도 "얼마나 멀리 빗나갔나"를 알려주는 보상 |

---

## 참고

- [TimeLens](https://github.com/TencentARC/TimeLens) (CVPR'26) · [arXiv 2512.14698](https://arxiv.org/abs/2512.14698) · [프로젝트 페이지](https://timelens-arc-lab.github.io/)
- [TimeLens2](https://github.com/MCG-NJU/TimeLens2) · [arXiv 2607.17423](https://arxiv.org/abs/2607.17423) · [프로젝트 페이지](https://mcg-nju.github.io/TimeLens2)
- [Time-R1](https://github.com/xiaomi-research/time-r1) (NeurIPS'25) · [arXiv 2503.13377](https://arxiv.org/abs/2503.13377)
- [Qwen3-VL 기술 리포트](https://arxiv.org/abs/2511.21631) · [Qwen (Wikipedia)](https://en.wikipedia.org/wiki/Qwen)
