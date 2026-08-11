# Code 탭 세션 시작용 프롬프트

아래 블록을 통째로 복사해서 Code 탭 첫 메시지로 붙여넣으세요.
`Video_grounding` 폴더를 작업 디렉터리로 열어둔 상태여야 합니다.

---

```
이 폴더(Video_grounding)에서 video temporal grounding 인퍼런스 작업을 이어서 하려고 해.
먼저 README.md 와 vtg_run.py 를 읽고 현재 상태를 파악해줘.

## 목표
영상 + 자연어 질의 → 사건이 일어난 시간 구간(start, end)을 찾는 모델을
실제로 돌려보고 결과를 비교하는 것. 학습은 하지 않고 인퍼런스만.
교수님께 데모로 보여드리는 게 최종 목적.

## 실행 환경
- 엘리스 클라우드 on-demand, 인스턴스 G-NAHPM-40
  = A100 80GB PCIe MIG 3g-40GB (VRAM 40GB, CPU 8 vCore, RAM 96GB)
- 스토리지 128GiB
- 시간당 과금이라 다운로드/삽질 시간을 줄이는 게 중요함

## 대상 모델 (둘 다 Apache-2.0)
- Boshenxx/Time-R1-7B    — Qwen2.5-VL-7B 백본, RL(GRPO+IoU reward) post-training
                            출력: <think>...</think><answer>1.05 to 7.62</answer> (단일 구간)
- MCG-NJU/TimeLens2-8B   — Qwen3-VL-8B 백본, SFT + temporal Wasserstein reward GRPO
                            출력: JSON 배열 [[1.2, 4.5], [10.0, 13.7]] (다중 구간 가능)

비교용으로 백본 원본(Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-8B-Instruct)도
vtg_run.py 레지스트리에 등록돼 있음. `python vtg_run.py --list` 로 확인 가능.

## 이미 만들어 둔 것
- setup_env.sh   : venv + torch cu128 + 라이브러리 설치, sm_XX 커널 동작 검증까지
- vtg_run.py     : 통합 추론 CLI. 두 계열의 프롬프트/processor API/출력 파싱 차이를 흡수
- queries.txt    : 샘플 질의 (마지막 줄은 일부러 영상에 없는 사건)
- README.md      : 실행 순서, VRAM 가이드, 트러블슈팅

## 반드시 지켜야 할 제약 (이미 시행착오로 확인된 것들)
1. Time-R1 공식 requirements.txt 를 따르지 말 것.
   torch==2.6.0 + CUDA 12.4 + vllm==0.8.4 로 핀돼 있는데,
   인퍼런스만 할 거면 vLLM 불필요하고 최신 GPU에서 호환성 문제가 생김.
2. transformers >= 4.57.0  (Qwen3-VL 지원에 필수. Qwen2.5-VL도 이 버전에서 동작)
   qwen-vl-utils >= 0.0.14  (TimeLens2의 return_video_metadata 인자에 필요)
3. VRAM 병목은 가중치가 아니라 비주얼 토큰 예산(total_pixels)임.
   TimeLens2 공식 카드 기본값 128000*32*32 는 KV 캐시만 ~19GB라 40GB에서도 위험.
   vtg_run.py 기본값은 16384 토큰이고 --total-tokens 로 조절.
4. 두 계열의 프롬프트를 서로 바꿔 쓰면 성능이 무너짐.
   각 모델이 학습된 원본 형식을 그대로 써야 함
   (Time-R1은 공식 demo.py, TimeLens2는 공식 모델 카드에서 가져와 대조 확인 완료).
5. 32GB~40GB에 두 모델을 동시에 못 올림. 순차 로드 후 명시적으로 해제해야 함.

## 지금 해줬으면 하는 것
(아래에서 해당하는 것만 남기고 나머지는 지우세요)

- [ ] setup_env.sh 를 실제 서버에서 돌리기 전에 한 번 더 리뷰해줘.
      A100 MIG 환경에서 문제될 만한 부분이 있는지 봐줘.
- [ ] 내 영상 파일(경로: ___)로 돌릴 수 있게 준비해줘.
      질의 문장도 이 영상에 맞게 다시 써줘.
- [ ] 두 모델 결과를 나란히 비교하는 리포트(표 + 타임라인 이미지)를 만드는
      스크립트를 추가해줘.
- [ ] 정답 구간(GT)이 있는 데이터로 tIoU / R@1 을 계산하는 평가 스크립트를 추가해줘.
- [ ] 결과를 교수님께 보여드릴 슬라이드나 문서로 정리해줘.

작업 전에 먼저 계획을 알려주고, 애매한 부분은 물어봐줘.
```

---

## 쓰는 요령

**목적에 맞게 마지막 섹션만 바꾸세요.** 위쪽 환경·제약 부분이 핵심이고,
이게 없으면 새 세션에서 같은 시행착오(버전 충돌, OOM, 프롬프트 혼용)를 반복하게 됩니다.

**영상 파일을 이미 정하셨다면** 프롬프트에 파일명과 대략의 내용,
그리고 찾고 싶은 사건 예시를 적어주세요. 질의 문장 품질이 결과를 크게 좌우합니다.

**서버에서 직접 작업할 때는** 엘리스 인스턴스에 SSH로 붙은 상태에서
이 폴더를 `scp` 나 git으로 올린 뒤 진행하는 게 편합니다.
