# 엘리스클라우드 실행 기록

## 실측 환경 (2026-08-25, G-NAHPM-40)

| 항목 | 값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| GPU | A100 80GB PCIe **MIG 3g.40gb** / driver 535.183.06 |
| VRAM | **42.4 GB** (여유 42.2 GB) |
| 연산 | fp32 약 7 TFLOPS ≈ 전체 A100 의 36% (3/7 슬라이스) |
| CPU / RAM | 8 vCore / 96 GiB |
| 디스크 | 256 GB (여유 254 GB) |
| python | 3.10.14 (system), venv 사용 가능 |
| **conda** | **없음** → venv 로 구성 |
| **ffmpeg** | **없음** → apt 로 설치 (4.4.2) |
| sudo | 무암호 가능 |
| torch | 2.5.1+cu121 동작 확인 |

### ⚠️ MIG 제약 — OpenGL 미지원

[NVIDIA 문서](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/deployment-considerations.html)
기준 MIG 인스턴스는 그래픽 API(OpenGL/Vulkan)를 지원하지 않는다.
따라서 PyRender 기반 메쉬 렌더링(`--render`)을 쓸 수 없다.

**영향 없음**: 우리가 필요한 것은 `.pt`/`.pkl` 파라미터이지 렌더 영상이 아니다.
포즈 육안 검증은 `gt/reproject_check.py` 의 **PIL 기반 관절 오버레이**로 한다
(게이트 2 에서 이미 사용한 방식, OpenGL 불필요).

## 상태

| 항목 | 상태 |
|---|---|
| SSH 접속 | ✅ `ssh elice` |
| 시스템 패키지 (ffmpeg) | ✅ |
| 환경 구축 (`setup_elice.sh`) | ⬜ 미실행 |
| 게이트 0 스모크 테스트 | ⬜ 미실행 |
| 모델 출력 구조 덤프 | ⬜ 미실행 |

## 실행 순서

```bash
# 저장소를 인스턴스로 옮긴 뒤
bash smpl_eval/setup_elice.sh   # venv 2개 생성 (conda 아님)
# SMPL_NEUTRAL.pkl 수동 배치 요구 시 안내대로 처리 후 재실행

bash smpl_eval/smoke_elice.sh \
  Captured-Motion-Dataset/Data4_vid3_golden_clip1_2K_60fps/3_golden_clip1/CAM_M01.mp4
```

`smoke_elice.sh` 가 끝나면 아래에 모델 출력 구조가 자동으로 덧붙는다.
**Task 10·11 의 어댑터는 그 덤프를 보고 작성한다** — 덤프 없이는 시작할 수 없다.

## 필요한 정보

| 어댑터 | 확정해야 할 것 | 출처 |
|---|---|---|
| `runners/comotion.py` | `.pt` 안의 betas/pose/trans/track_id/frame 키 이름 | CoMotion .pt 덤프 |
| `runners/multihmr2.py` | `.pkl` 계층 구조, Anny 관절명 목록 | Multi-HMR 2 .pkl + Anny 덤프 |

---

<!-- 이 아래로 dump_outputs.sh 출력이 누적된다 -->
