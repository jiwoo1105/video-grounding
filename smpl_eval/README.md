# smpl_eval — 다중인물 SMPL 추정 모델 평가

Captured-Motion-Dataset 28영상에 **CoMotion**(ICLR 2025)과 **Multi-HMR 2**(2026)를
적용해 **① 포즈 정확도**와 **② ID 유지**를 동일 조건에서 비교한다.

핵심 설계: 두 모델의 이질적 출력을 어댑터가 `tracks.npz` 단일 스키마로 정규화하고,
모든 지표 코드는 모델을 모른 채 동작한다 → 비교의 공정성 보장.

- 설계: `../docs/superpowers/specs/2026-08-24-smpl-multiperson-eval-design.md`
- 계획: `../docs/superpowers/plans/2026-08-25-smpl-multiperson-eval.md`

## 로컬 환경 (GPU 불필요)

지표 엔진·GT 파서·리포트는 **합성 데이터로 로컬 검증**한다. 모델 추론만 GPU가 필요하다.

```bash
python3 -m venv .venv-smpl                        # 저장소 루트에서
.venv-smpl/bin/pip install -r smpl_eval/requirements.txt
.venv-smpl/bin/python -m pytest smpl_eval/tests/ -v
```

## 엘리스클라우드 (모델 추론)

```bash
bash smpl_eval/setup_elice.sh                     # 최초 1회
bash smpl_eval/smoke_elice.sh <영상경로>           # 게이트 0
```

## 실행 순서

```bash
python3 -m smpl_eval.dataset_index Captured-Motion-Dataset   # manifest.json
python3 -m smpl_eval.gt.reproject_check                      # 게이트 2
python3 -m smpl_eval.run_all --model comotion  --pilot       # 파일럿 4영상
python3 -m smpl_eval.run_all --model multihmr2 --pilot
python3 -m smpl_eval.evaluate --model comotion               # 지표 산출
python3 -m smpl_eval.report.build                            # report.html
```

## `tracks.npz` 스키마

한 행 = "프레임 f 에 존재하는 사람 t". 모든 배열의 첫 축 길이 N 동일.

| 키 | shape | dtype | 설명 |
|---|---|---|---|
| `frame_ids` | (N,) | int32 | 0-based 프레임 번호 |
| `track_ids` | (N,) | int32 | 트랙 ID (모델이 부여) |
| `betas` | (N, 10) | float32 | 체형. SMPL β 규약 |
| `global_orient` | (N, 3) | float32 | 루트 회전 (axis-angle) |
| `body_pose` | (N, 23, 3) | float32 | 관절 회전, 부모 대비 상대 |
| `transl` | (N, 3) | float32 | 카메라 좌표계 루트 위치 (m) |
| `joints3d` | (N, 24, 3) | float32 | SMPL 24관절, 카메라 좌표계 |
| `joints2d` | (N, 24, 2) | float32 | 이미지 픽셀 좌표 |
| `bbox` | (N, 4) | float32 | `x1,y1,x2,y2` 픽셀 |
| `score` | (N,) | float32 | 검출 신뢰도 |
| `betas_native` | (N, K) | float32 | (선택) 모델 고유 체형 파라미터. Anny 는 K=6 |

동반 `meta.json`: `model, body_model, converted_from, video, fps, resolution, n_frames, runtime_sec`

`betas` 가 전부 `NaN` 이면(Anny→SMPL 피팅 실패) β 기반 지표는 자동으로
`joints3d` 기반 팔다리 길이 분산으로 대체된다 — 둘 다 신원 일관성을 재므로 대체가 성립한다.
