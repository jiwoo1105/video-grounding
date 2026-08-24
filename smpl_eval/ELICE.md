# 엘리스클라우드 실행 기록

## 상태

| 항목 | 상태 |
|---|---|
| 환경 구축 (`setup_elice.sh`) | ⬜ 미실행 |
| 게이트 0 스모크 테스트 | ⬜ 미실행 |
| 모델 출력 구조 덤프 | ⬜ 미실행 |

## 실행 순서

```bash
# 저장소를 인스턴스로 옮긴 뒤
bash smpl_eval/setup_elice.sh
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
