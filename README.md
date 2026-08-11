# Video Temporal Grounding — 엘리스 클라우드 실험

영상 + 자연어 질의 → 해당 사건의 시간 구간 `(start, end)` 을 찾는 태스크.
**엘리스 클라우드 GPU 인스턴스** 전용입니다. (Colab 아님)

## 파일

| 파일 | 용도 |
|---|---|
| **`app.py`** | **웹 UI — 영상 올리고 문장 쓰면 구간을 찾아줌. 발표용** |
| `setup_env.sh` | 환경 구성 (venv + torch cu128 + 라이브러리 + 커널 검증) |
| `vtg_run.py` | 단발 추론 CLI |
| `run_experiments.py` | **영상 × 모델 일괄 실험 + 리포트 생성** |
| `test_vtg_run.py` | GPU 없이 추론 경로 검증 (77개 체크) |
| `test_e2e.py` | GPU 없이 실험 전체 검증 (42개 체크) |
| **`실행순서.md`** | **인스턴스 켠 뒤 순서대로 따라하기 ← 여기부터** |
| `PREFLIGHT.md` | 엘리스 실행 전 사전 작업 체크리스트 |
| `queries.txt` | 질의 작성 기준 + 샘플 |
| **`2Y8XQ_분석.md`** | **데모 영상 내용 + 질의 6개 + 발표 시나리오** |
| `질의_예시.md` | 어떤 문장을 써야 하는지 예시 모음 |
| `영상_다운로드_URL.md` | 벤치마크/학습 데이터 영상 받는 곳 + 용량 |
| `Qwen_모델_정리.md` | 논문 분석 / 모델 계보 (발표용) |

## 실행 순서

```bash
# ── 1. 환경 (최초 1회, ~10분) ────────────────────────────────
bash setup_env.sh
source ~/vtg-env/bin/activate

# ── 2. 코드 검증 (GPU 불필요, 5초) ───────────────────────────
python3 test_vtg_run.py          # 77개 PASS — 추론 경로
python3 test_e2e.py              # 42개 PASS — 실험 전체

# ── 3. 영상 준비 ─────────────────────────────────────────────
python3 run_experiments.py --download          # 공식 데모 영상 (2.3MB)

# 짧은 벤치마크 영상 + GT
python3 run_experiments.py --download-bench --bench-dataset charades

# 긴 벤치마크 영상 + GT — 사람이 검수한 질의·정답 (최대 4분대)
python3 run_experiments.py --download-bench --bench-dataset activitynet --bench-longest

# 내 영상 등록 (선택). 3분 이상이면 --long 을 꼭 붙이세요
python3 run_experiments.py --add-video ~/my5min.mp4 \
    --queries "a woman opens the door|two people shake hands" --long

python3 run_experiments.py --list-videos       # 실험 계획 확인

# ── 4. 웹 UI (발표용) ────────────────────────────────────────
python3 app.py               # http://<인스턴스주소>:7860
python3 app.py --share       # 외부 접속용 임시 공개 링크

# ── 5. 스모크 테스트 (제일 작은 모델로 빠르게) ────────────────
python3 vtg_run.py --model timelens2-4b --video videos/2Y8XQ.mp4 \
    --query "A man drinks water with a glass"

# ── 6. 본 실험 ───────────────────────────────────────────────
python3 run_experiments.py --model safe        # TimeLens-8B
python3 run_experiments.py --model main        # 4개 모델 전부 (오래 걸림)
```

결과는 `results/` 에 `results.jsonl`, `report.md`, 영상별 타임라인 `*.png` 로 저장됩니다.

## 실험 영상 구성

| 구분 | 영상 | 길이 | GT | 목적 |
|---|---|---|---|---|
| 벤치마크 (짧은) | Charades-TimeLens | ~30초 | **있음** | tIoU 정량 측정 |
| **벤치마크 (긴)** | **ActivityNet-TimeLens** | **최대 4분대** | **있음** | **긴 영상 정량 측정** |
| 외부 | 2Y8XQ (TimeLens 공식 데모) | ~30초 | 없음 | 정성 + 없는 사건 테스트 |
| 내 영상 | 직접 등록 | 자유 | 없음 | 실제 용도 확인 |

**긴 영상도 벤치마크에서 가져오는 게 낫습니다.** 임의의 영상을 쓰면 질의를 직접
지어내야 하고 정답이 없어 "그럴듯해 보인다" 수준의 평가밖에 못 합니다.
ActivityNet-TimeLens 는 **사람이 검수한 질의 + 정답 구간**이 최대 4분대 영상에 붙어 있어요.

| 데이터셋 | 평균 길이 | 용도 |
|---|---|---|
| `charades` | 29.6초 | 기본 (가장 가벼움) |
| **`activitynet`** | **134.9초, 최대 4분대** | **긴 영상 테스트** |
| `qvhighlights` | 149.6초 | 하이라이트형 질의 |

3분 이상인 영상은 자동으로 `long` 표시가 붙습니다.
내 영상은 GT가 없으니, 직접 보고 정답 시각을 메모해두면 손으로 비교할 수 있습니다.

`--long` 을 붙인 영상은 토큰 예산을 자동으로 낮추고(`--long-tokens 16384 --long-fps 1.0`),
정렬로 맨 뒤에 배치해 모델 재로드를 모델당 1회로 줄입니다.

| 영상 길이 | 권장 설정 |
|---|---|
| ~1분 | 기본값 그대로 |
| 3~5분 | `--long` (16384 토큰 / 1fps) |
| 10분+ | `--long --long-tokens 8192` |

> 구글 샘플 버킷(`commondatastorage.googleapis.com`)은 2026년 8월 기준 접근이 막혔습니다.
> 외부 영상은 HuggingFace 에 있는 TimeLens 공식 데모 영상을 씁니다.

## 모델

```bash
python3 vtg_run.py --list     # 전체 목록 + 프리셋
```

| 프리셋 | 모델 | 용도 |
|---|---|---|
| `safe` | TimeLens-8B | **CVPR'26 게재. 발표 주력** |
| `best` | TimeLens2-8B | 성능 1위 (7벤치 평균 mIoU 48.0) |
| `light` | TimeLens2-4B | 가성비 (47.7, 8B와 0.3 차이) |
| `cvpr` | TimeLens-7B + 8B | CVPR 논문 2종 |
| `ablation` | Qwen3-VL-8B + TimeLens-8B | **학습 전 vs 후 대조** |
| `main` | 위 4종 전부 | 종합 비교 |

## 인스턴스

**G-NAHPM-40** (A100 80GB PCIe MIG 3g-40GB, ₩1,380/시간) + **스토리지 128GiB** (₩20/시간)

| 옵션 | VRAM | 시간당 | 판단 |
|---|---|---|---|
| G-NAHPM-20 | 20GB | ₩690 | 7B/8B엔 빠듯. 4B 전용 |
| **G-NAHPM-40** | **40GB** | **₩1,380** | **적정** |
| G-NAHP-80 | 80GB | ₩2,500 | 두 모델 동시 로드가 필요할 때만 |

셋업 + 다운로드 1시간, 실험 2~3시간 ≈ **₩5,000 내외**. **끝나면 인스턴스 삭제 필수.**

## 알아둘 것

**Time-R1 공식 `requirements.txt`를 따르지 마세요.**
`torch==2.6.0 + CUDA 12.4 + vllm==0.8.4`로 고정돼 있는데, 인퍼런스만 할 거면 vLLM이
불필요하고 최신 GPU에서 호환성 문제가 납니다. `setup_env.sh`는 cu128 휠 하나로
A100/H100/5090을 모두 커버합니다.

**transformers 버전이 유일한 지뢰입니다.**

| 버전 | Qwen2.5-VL | Qwen3-VL |
|---|---|---|
| ~4.56 | ✅ | ❌ |
| **4.57 ~ 4.9x** | ✅ | ✅ ← 여기로 고정 |
| 5.x | ⚠️ 인자명 변경 | ✅ |

**세 계열의 추론 규약이 전부 다릅니다.** `vtg_run.py`가 흡수하지만 알아두세요.

| 모델 | 프롬프트 | `process_vision_info` | 출력 |
|---|---|---|---|
| TimeLens-8B | 짧은 GROUNDER_PROMPT | patch16 + kwargs + metadata | `The event happens in X - Y seconds` |
| TimeLens-7B | **앞에 타임스탬프 설명문 추가** | metadata만 | 동일 |
| TimeLens2 | JSON 요구 | patch16 + kwargs + metadata | `[[X, Y], ...]` **다중 구간** |
| Time-R1 | `<think>` 유도 | kwargs만 | `<answer>X to Y</answer>` |

프롬프트를 바꿔 쓰면 성능이 무너집니다. 각 공식 모델 카드 원문을 그대로 씁니다.

**VRAM 병목은 가중치가 아니라 비주얼 토큰 예산입니다.**

| 영상 길이 | 권장 `--total-tokens` |
|---|---|
| ~30초 | 32768 |
| 1~3분 | 16384 |
| 5분+ | 8192 + `--fps 1.0` |

**flash-attn** — A100(sm_80)은 사전빌드 휠이 있어 `--attn flash_attention_2` 사용 가능.
긴 영상에서 메모리가 절약됩니다. 기본값은 `sdpa`.

## 한계 (데모 전에 알아둘 것)

TimeLens-8B의 Charades 성적: **R1@0.3 = 76.6%, R1@0.5 = 63.0%, R1@0.7 = 35.2%**

엄격한 기준(IoU 0.7)으로는 **3분의 1만 맞습니다.** "정확히 찾아준다"보다
**"후보 구간을 좁혀준다"** 로 기대치를 잡으세요.

그리고 **모델은 "그런 장면 없습니다"라고 답하지 못합니다.** 영상에 없는 사건을
물어도 아무 구간이나 뱉어요 (no-target rejection 미해결). `queries.txt` D섹션이
이걸 시연하는 용도이고, 발표에서 오히려 좋은 이야깃거리가 됩니다.

## 참고

- [TimeLens](https://github.com/TencentARC/TimeLens) (CVPR'26) · [arXiv 2512.14698](https://arxiv.org/abs/2512.14698)
- [TimeLens2](https://github.com/MCG-NJU/TimeLens2) · [arXiv 2607.17423](https://arxiv.org/abs/2607.17423)
- [Time-R1](https://github.com/xiaomi-research/time-r1) (NeurIPS'25) · [arXiv 2503.13377](https://arxiv.org/abs/2503.13377)
