# 다중인물 SMPL 추정 모델 평가 — 설계 문서

작성일: 2026-08-24
상태: 승인 대기

---

## 1. 목표

Captured-Motion-Dataset 의 전체 영상(28개)에 대해 **최신 비디오 기반 다중인물 SMPL 계열 추정 모델**을 적용하고, 다음 두 축을 정량 평가한다.

1. **포즈 정확도** — 포즈가 오류 없이 추정되는가
2. **ID 유지** — 사람별 ID를 놓치지 않는가

부수 목표: 이 결과를 근거로 **Video Temporal Grounding(VTG) 프로젝트에 SMPL을 어떻게 활용할지** 설계한다.

---

## 2. 배경

### 2.1 데이터셋

`/Users/ziuuu/Downloads/Captured-Motion-Dataset` — 4대 멀티캠 모션캡처. COLMAP 캘리브레이션과 3D GT 포즈를 이미 보유.

| 데이터 | 카메라 | 해상도 | fps | 길이 | 프레임 | 인원 | 기존 GT |
|---|---|---|---|---|---|---|---|
| Data1 농구 (S03/S06/S08/S12) | 4×4=16 | 1920×1080 | 30 | 10s | 300 | 다수 | `PoseResults2d/3d_*.txt` |
| Data2 테니스 복식 | 4 | 1920×1080 | 30 | 77s | 2309 | 4+ | `3DPose.txt`, `obj0/1.bvh` |
| Data3 OnlyOneOf | 4 | 1220×1024 | 60 | 23.7s | 1420 | 2 | `3Dpose.txt`, `obj0/1.bvh` |
| Data4 golden | 4 | 1920×1080 | 60 | 11s | 660 | 3 | `3DPose.txt`, `obj0/1/2.bvh` |

**합계 28 영상 / 22,356 프레임.**

기존 산출물은 3D 스켈레톤(`.txt`, `.bvh`)이며 **SMPL 메쉬 파라미터는 없다.** 본 작업은 여기에 SMPL 계열 레이어를 추가하고 검증하는 것이다.

### 2.2 난이도 매트릭스 — 데이터셋별로 다른 실패 모드를 때린다

| 요인 | 농구 | 테니스 | OnlyOneOf | golden |
|---|---|---|---|---|
| 밀집·가림 → ID 스왑 | ★★★ | ★ | ★ | ★★ |
| 고속 모션블러 → 포즈 오류 | ★★★ | ★★★ | ★★ | ★★ |
| 유사 외형(유니폼·무대의상) | ★★★ | ★★ | ★★ | ★★★ |
| 장기 드리프트 | ★ | ★★★ (2309f) | ★★ | ★ |

### 2.3 VTG 프로젝트와의 연결

본 저장소의 VTG 태스크는 `영상 + 자연어 질의 → (start, end)` 이며, 질의가 대부분 **사람 동작**이다 (`a woman opens the refrigerator`, `two people shake hands`, `a person stands up from the chair`). 현재 파이프라인(TimeLens2 등 VLM)은 샘플링된 RGB 프레임에 의존하며 **3D 인체 움직임의 명시적 표현이 없다.**

SMPL 계열이 메우는 지점 4가지:

| # | 접점 | 근거 |
|---|---|---|
| ① | **동작 경계 검출** | `θ` 시간 미분 = 관절 각속도 → 동사의 시작·끝을 물리량으로 |
| ② | **인물 지칭 grounding** | `a woman in a red jacket` → 다중인물 씬에서 person track ID 필수 |
| ③ | **프레임 샘플링 예산** | `total_tokens` 제약으로 프레임이 솎아지는 문제 → 모션 에너지 기반 우선 샘플링 |
| ④ | **상호작용 표현** | `two people shake hands` = 두 사람 손목 3D 거리 |

②가 본 작업에서 **ID 유지를 중점 평가하는 직접적 이유**다.

---

## 3. 결정 사항

### 3.1 실행 환경

**엘리스클라우드 GPU 인스턴스** (A100/H100급). 인스턴스가 휘발성이므로 원샷 환경구축 스크립트와 결과물 로컬 동기화가 필수 요건.

### 3.2 범위

**단안(monocular) 영상 우선.** 각 카메라를 독립적으로 처리하고 뷰 간 결과를 비교한다. COLMAP 기반 멀티뷰 융합은 **별도 스펙으로 분리**한다.

근거: 뷰마다 ID/포즈가 어떻게 달라지는지가 그 자체로 모델 강건성 지표가 된다. 융합은 범위가 훨씬 크다.

### 3.3 모델 선정 — CoMotion + Multi-HMR 2

| | **CoMotion** | **Multi-HMR 2** |
|---|---|---|
| 출처 | Apple, ICLR 2025 ([arXiv:2504.12186](https://arxiv.org/abs/2504.12186)) | NAVER, 2026.06 ([arXiv:2606.14841](https://arxiv.org/abs/2606.14841)) |
| 코드 | [apple/ml-comotion](https://github.com/apple/ml-comotion) | [naver/multi-hmr2](https://github.com/naver/multi-hmr2) |
| 백본 | ConvNeXtV2 멀티스케일 | ViT-Large (DINOv3 초기화), 768px |
| 패러다임 | **순환 상태 추적** — 이전 포즈를 새 프레임으로 직접 갱신 (GRU) | **DETR 검출 + 특징 매칭** — human query 100개, 디코더 8블록 |
| ID 유지 원리 | 트랙별 GRU 은닉상태 유지, 가림 중에도 포즈 갱신 | SAM2 메모리 특징 증류(4096-d) + KNN(K=35, L1) + 골반 궤적 릿지회귀 |
| 바디 모델 | SMPL | Anny (v0.6에서 `smpl`/`smplx` 리토폴로지 제공) |
| 카메라 | 약원근 근사 | **장면 일관 카메라 예측** (cls 토큰 → 수직 FOV) |
| PoseTrack21 IDF1 | 79.5 | 75.2 |
| PoseTrack21 MOTA | 71.4 (4D-Humans 56.7) | — |
| 3DPW PA-MPJPE | 36.1mm | 41.8mm |
| MSCOCO AP | — | 50.7 (v1 31.2) |
| 속도 | 176ms/frame (V100), 4D-Humans 대비 ~12× | 미공개 |
| 설치 | `pip install -e '.[all]'` + 체크포인트 스크립트 + SMPL 등록 | `pip install -e .` + 체크포인트 자동 |

#### 선정 근거

1. **약점이 상보적이다.** CoMotion은 긴 가림·재식별에 취약하고, Multi-HMR 2는 가림 시 앉은 자세로 편향된다. 한쪽이 무너지는 지점에서 다른 쪽이 어떻게 하는지 관찰 가능하다.
2. **ID 유지 전략이 정반대다.** "상태를 이어간다(기억)" vs "매번 다시 찾는다(인식)". 밀집·가림·유사외형 환경에서 어느 전략이 우세한지는 논문에 답이 없는 질문이며, 우리 데이터로 답할 수 있다.
3. **논문 수치를 직접 비교할 수 없다.** CoMotion의 3DPW 수치는 오라클 크롭 조건으로 보이고 Multi-HMR 2는 전체 프레임 자체 검출 조건이다. 동일 조건 측정 자체가 기여가 된다.
4. **재구축 리스크가 가장 낮다.** 둘 다 단일 명령 설치 + 비디오 트래킹 내장.
5. **Multi-HMR 2의 카메라 예측**이 향후 멀티뷰 융합 단계로 이어지며, COLMAP GT와 대조하면 부수 결과가 나온다.

#### 기각한 대안

| 조합 | 기각 사유 |
|---|---|
| 4D-Humans + PHALP (ICCV 2023) | CoMotion이 MOTA +14.7, IDF1 +8.6으로 이미 명확히 앞섬. 결과가 자명 |
| WHAM / TRAM / GVHMR / OnlineHMR (월드좌표) | **우리 카메라는 COLMAP 캘리브레이션된 고정 리그.** 카메라를 모를 때 푸는 문제를 비싸게 푸는 셈. GVHMR은 단일 인물 전용 |
| PromptHMR (CVPR 2025) | SMPL-X·정확도는 우수하나 DROID-SLAM + Detectron2 + Metric3D + ViTPose + SPEC 설치를 인스턴스 재구축마다 반복해야 함 |
| SMPLest-X / AiOS | **트래킹 없음.** 프레임별 초점거리 추정으로 시간적 불안정성 보고됨 |
| Multi-THuMBS (2026.07) | 코드 미공개 |
| DanceHMR (2026.06) | 손 특화로 Data3/4에 적합하나 코드 공개 여부 미확인. **Phase B 이후 옵션으로 보류** |

### 3.4 바디 모델 / 손·얼굴 범위

**결정 보류 — 파일럿에서 측정 후 결정.**

VTG 질의가 손 동작 중심(`picks up a phone`, `shake hands`, `hands over a document`)이라 SMPL-X급 손 관절이 의미를 가진다. 그러나 원거리 샷에서 손이 10px 수준이면 손 파라미터는 관측이 아니라 prior 산물이다.

→ 파일럿에서 **데이터셋별 손 bbox 픽셀 크기를 실측**하고, 손이 충분히 큰 데이터에만 손 평가를 적용한다. 리포트에는 "농구는 손 N px이므로 손 평가 제외"처럼 근거를 명시한다.

### 3.5 그 밖의 결정

| 항목 | 결정 | 근거 |
|---|---|---|
| 공통 중간 포맷 | **`tracks.npz` 채택** | 모델별 평가 코드 분기를 막아 비교의 공정성을 보장 |
| GT 조인트 매핑 실패 시 | **무참조 지표 폴백 허용** | 폴백이 있는 편이 항상 낫다. 최악에도 결과는 나온다 |
| 코드 위치 | `Video_grounding/smpl_eval/` | 데이터셋이 이미 이 저장소에 있음. 별도 레포는 관리 부담만 증가 |
| 데이터 범위 | **파일럿 4영상 → 게이트 통과 후 28영상 전수** | 지표 버그로 전수 재실행하는 사태 방지 |
| 문서 작성 시점 | 1~3장은 Phase A~B와 **병행** | 실험 결과와 무관한 부분이라 GPU 대기 중 작성 |

---

## 4. 아키텍처

### 4.1 핵심 설계 원칙 — 공통 중간 포맷

CoMotion과 Multi-HMR 2는 출력 형식·바디모델·관절 규약이 모두 다르다. 평가 코드가 모델마다 갈리면 (1) 코드 중복 (2) 한쪽 코드의 버그가 모델 성능 차이로 오인됨 (3) 모델 추가 시 전면 재작업 이라는 세 문제가 발생한다.

**어댑터가 출력을 `tracks.npz` 단일 스키마로 정규화하고, 모든 평가 코드는 어느 모델인지 모른 채 동작한다.**

```
CoMotion   → results.pt / .txt(MOT) ─┐
                                     ├→ [어댑터] → tracks.npz → [평가 코드 한 벌]
Multi-HMR2 → *.pkl (Anny params)   ─┘
```

새 모델 추가 비용 = 어댑터 1개. 지표 코드는 불변.

### 4.2 `tracks.npz` 스키마

한 행 = "프레임 f 에 존재하는 사람 t". 모든 배열의 첫 축 길이 N 은 동일.

| 키 | shape | dtype | 설명 |
|---|---|---|---|
| `frame_ids` | (N,) | int32 | 0-based 프레임 번호 |
| `track_ids` | (N,) | int32 | 트랙 ID (모델이 부여) |
| `betas` | (N, 10) | float32 | 체형. SMPL β 규약 |
| `global_orient` | (N, 3) | float32 | 루트 회전 (axis-angle) |
| `body_pose` | (N, 23, 3) | float32 | 관절 회전 (axis-angle), 부모 대비 상대 |
| `transl` | (N, 3) | float32 | 카메라 좌표계 루트 위치 (m) |
| `joints3d` | (N, 24, 3) | float32 | SMPL 24 관절, 카메라 좌표계 |
| `joints2d` | (N, 24, 2) | float32 | 이미지 픽셀 좌표 |
| `bbox` | (N, 4) | float32 | `x1,y1,x2,y2` 픽셀 |
| `score` | (N,) | float32 | 검출 신뢰도 |
| `betas_native` | (N, K) | float32 | (선택) 모델 고유 체형 파라미터 원본. Anny 는 K=6 |

동반 `meta.json`:

```json
{
  "model": "comotion",
  "model_version": "<commit hash>",
  "body_model": "smpl",
  "converted_from": null,
  "video": "Data2.../cam1_2K.mp4",
  "fps": 29.97,
  "resolution": [1920, 1080],
  "n_frames": 2309,
  "n_tracks": 4,
  "runtime_sec": 184.2
}
```

`body_model` / `converted_from` 필드는 Multi-HMR 2 의 Anny→SMPL 변환 이력을 남겨 추후 문제 추적을 가능하게 한다.

**관절 규약**: SMPL 24 관절 표준 순서를 정본으로 삼는다 (`0 pelvis … 23 right_hand`). Multi-HMR 2 의 Anny 163본 출력은 어댑터에서 이 규약으로 매핑한다.

**체형 파라미터 불일치 처리**: Anny 의 체형 파라미터는 해석 가능한 6개(성별·나이·키·몸무게 등)로, SMPL 의 β 10개(PCA 계수)와 **의미도 개수도 다르다.** 어댑터는 다음 규칙을 따른다.

- CoMotion: SMPL β 를 그대로 기록. `meta.body_model = "smpl"`
- Multi-HMR 2: Anny 원본 파라미터를 `betas_native` 키에 별도 보존하고, `betas` 에는 Anny 메쉬에 SMPL 을 피팅해 얻은 β 를 기록. 피팅이 게이트 3을 통과하지 못하면 `betas` 는 `NaN` 으로 채우고 `meta.body_model = "anny"`, `meta.converted_from = "anny"` 로 표기한다.

**β 를 쓰는 지표(β 분산·β 급변 검출)는 `betas` 가 `NaN` 인 경우 자동으로 건너뛰고, 대신 `joints3d` 기반 팔다리 길이 분산으로 대체한다.** 두 지표는 물리적으로 같은 것(신원 일관성)을 재므로 대체가 성립한다.

### 4.3 컴포넌트

각 모듈은 단일 책임을 가지며 독립적으로 테스트 가능하다.

| 모듈 | 책임 | 입력 → 출력 |
|---|---|---|
| `dataset_index.py` | 28영상 스캔 | 데이터셋 경로 → `manifest.json` |
| `runners/comotion.py` | CoMotion 실행 + 정규화 | 영상 → `tracks.npz` |
| `runners/multihmr2.py` | Multi-HMR 2 실행 + Anny→SMPL 변환 + 정규화 | 영상 → `tracks.npz` |
| `gt/parse_pose3d.py` | GT 3D 포즈 파싱 | `PoseResults3d`/`3DPose.txt` → `gt_tracks.npz` |
| `gt/parse_bvh.py` | BVH 파싱 | `obj*.bvh` → `gt_tracks.npz` |
| `gt/joint_mapping.py` | **GT 관절 ↔ SMPL 24 매핑** | 매핑 테이블 + 검증 유틸 |
| `metrics/pose.py` | PA-MPJPE, MPJPE, 2D 재투영 오차 | `tracks.npz` + `gt_tracks.npz` → dict |
| `metrics/plausibility.py` | 팔다리 길이 분산, 가속도 지터, 관절각 위반, **β 분산** | `tracks.npz` → dict (GT 불필요) |
| `metrics/identity.py` | IDF1, MOTA, ID-switch, 단편화, 인원수 오차 | `tracks.npz` + `gt_tracks.npz` → dict |
| `metrics/occlusion.py` | bbox IoU 기반 가림 구간 검출 → 구간별 ID 유지율 | `tracks.npz` → dict |
| `metrics/handsize.py` | 데이터셋별 손 픽셀 크기 실측. `joints2d` 의 손목(20/21)–손(22/23) 거리와 사람 bbox 높이의 비로 손 영역을 추정 | `tracks.npz` → 통계 |
| `report/build.py` | 집계, worst-K 실패 프레임 추출, HTML 리포트 | `results/*.json` → `report.html` |
| `run_all.py` | 오케스트레이션 (재개 가능, 실패 격리) | `manifest.json` → 전체 실행 |

**`joint_mapping.py` 를 별도 모듈로 격리한 이유**: GT 관절 규약과 SMPL 규약의 매핑은 본 작업에서 오류 가능성이 가장 높은 지점이다. 격리하면 파일럿에서 이 한 곳만 집중 검증할 수 있다.

### 4.4 데이터 흐름

```
Captured-Motion-Dataset (28 videos)
        │ dataset_index.py
        ▼
   manifest.json
        │ runners/*  ← 엘리스클라우드 GPU
        ▼
outputs/{model}/{dataset}/{session}/{cam}/tracks.npz + meta.json (+ render.mp4 선별)
        │                       ┌── outputs/gt/{dataset}/{session}/gt_tracks.npz
        ▼                       ▼
     metrics/*  →  results/{model}__{video}.json
        ▼
   report/build.py  →  report.html + 실패사례 컨택트시트
```

---

## 5. 평가 지표

### 5.1 포즈 정확도 축

| 지표 | GT 필요 | 정의 |
|---|---|---|
| **PA-MPJPE** | ✓ | Procrustes 정렬 후 관절 평균 오차 (mm) |
| **MPJPE** | ✓ | 루트 정렬 후 관절 평균 오차 (mm) |
| **2D 재투영 오차** | ✓ (Data1) | `PoseResults2d` 대비 픽셀 오차 |
| **팔다리 길이 분산** | ✗ | 트랙 내 골격 길이의 시간 표준편차. 변하면 추정 오류 |
| **가속도 지터** | ✗ | 관절 위치 2차 미분의 크기. 떨림 정량화 |
| **관절각 위반** | ✗ | 무릎·팔꿈치 역굴곡 프레임 수 |

무참조 지표(GT 불필요)는 GT 매핑이 실패해도 결과를 낼 수 있는 **폴백 경로**이자, GT 없는 구간의 오류 탐지기다.

### 5.2 ID 유지 축

| 지표 | 정의 |
|---|---|
| **IDF1** | 한 사람에게 하나의 ID를 유지한 비율 (py-motmetrics) |
| **MOTA** | 미검출·오검출·ID스위치 종합 |
| **ID-switch** | ID가 바뀐 횟수 |
| **Fragmentation** | 트랙이 끊긴 횟수 |
| **인원수 오차** | 프레임별 검출 인원 vs 기지값 (테니스 4, OnlyOneOf 2, golden 3) |
| **가림 구간 ID 유지율** | bbox IoU로 가림 이벤트를 자동 검출하고 그 전후 ID 일치 여부 |
| **β 급변 검출** | 트랙 내 β 점프 → ID 스왑 의심 구간 자동 추출 |
| **뷰 간 일관성** | 같은 장면 4캠에서 검출 인원·ID 구조가 일관되는가 |

**`β 급변 검출` 의 근거**: β 는 신원(identity)이며 시간에 불변이어야 한다. 트랙 내에서 β 가 점프하면 추정 실패 또는 ID 스왑이다. GT 없이 동작하는 강력한 탐지기다.

---

## 6. 에러 처리

| 상황 | 대응 |
|---|---|
| 영상 1개 처리 실패 | **영상 단위 격리.** 나머지 계속 진행, `failures.log` 에 기록 |
| 중단 후 재실행 | `tracks.npz` 존재 시 skip. `--force` 로 강제 재실행 |
| 엘리스 인스턴스 소멸 | `setup_elice.sh` 원샷 재구축. `tracks.npz`/`results` 만 로컬 동기화(~300MB)하여 분석은 인스턴스 없이 진행 |
| GT 조인트 매핑 불확실 | 파일럿에서 GT를 영상에 재투영 오버레이하여 육안 검증. **게이트 통과 실패 시 무참조 지표로만 진행** |
| Anny→SMPL 변환 실패 | 관절 위치(`joints3d`) 기준 비교로 축소. β/θ 직접 비교는 포기 |
| Multi-HMR 2 추론이 과도하게 느림 | 파일럿 실측으로 판단. 필요 시 Data2(2309f) 처리 계획 조정 |
| 렌더링 병목 (100~200ms/frame) | 렌더는 전수가 아니라 **worst-K 실패 구간만** 선별 생성 |

---

## 7. 검증 — 파일럿 게이트

### 7.1 파일럿 대상 (4영상)

| 영상 | 프레임 | 노리는 실패 모드 |
|---|---|---|
| 농구 S03 `Cam1_Deck0009_HL01_2K.mp4` | 300 | 최다인원 밀집 → ID 스왑 |
| 테니스 `cam1_2K.mp4` | 2309 | 최장 시퀀스 → 드리프트 |
| OnlyOneOf `cam-001.mp4` | 1420 | 60fps + 2인 근접 |
| golden `CAM_M01.mp4` | 660 | 유사 무대의상 3인 |

### 7.2 게이트 조건 — 5개 전부 통과해야 전수 진행

1. **렌더 육안 확인** — 메쉬가 실제 사람 몸에 붙어 있는가
2. **GT 재투영 오버레이** — GT 관절을 영상에 투영했을 때 실제 사람과 일치하는가 → 조인트 매핑 검증
3. **Anny→SMPL 변환 타당성** — Multi-HMR 2 결과가 SMPL 규약으로 변환된 뒤에도 해부학적으로 말이 되는가
4. **지표 상식 범위** — PA-MPJPE 대략 50~150mm, IDF1 ∈ [0,1], 인원수가 기지값 근처
5. **합성 유닛테스트** — 정답 트랙을 인위적으로 ID 스왑시켰을 때 `metrics/identity.py` 가 실제로 검출하는가 (지표 코드 자체의 검증)

### 7.3 파일럿 부산물

- **손 bbox 픽셀 크기** (데이터셋별) → SMPL-X 확장 여부 결정 근거
- **실측 추론 속도** (모델별) → Phase B 소요 시간 산정
- **농구 카메라의 정적/동적 여부** 확인 → 월드좌표 계열 기각 근거 재확인

---

## 8. 실행 계획

| Phase | 내용 | 산출물 | 소요 |
|---|---|---|---|
| **0** | 엘리스 환경 구축, 두 모델 스모크 테스트 | `setup_elice.sh` | 0.5일 |
| **A** | `manifest.json` 생성, 파일럿 4영상, 게이트 5개 | 파이프라인 전체 + 게이트 리포트 | 1일 |
| **B** | 28영상 × 2모델 전수 실행 | `tracks.npz` 56개 (~300MB) | 0.5~1일 (대부분 GPU 대기) |
| **C** | 지표 산출, worst-K 추출, 리포트 | `report.html` + 실패 컨택트시트 | 1일 |
| **D** | 기술 해설 문서 (1~3장은 A~B와 병행) | Artifact 발행 | 병행 + 마무리 |

**총 3~4일.**

### Phase D 문서 목차

| 장 | 내용 | 선행 의존 |
|---|---|---|
| 1 | SMPL이란 무엇인가 — β/θ, 블렌드셰이프, LBS | 없음 |
| 2 | 계보 — SMPL→SMPL-X/STAR/GHUM/Anny, SMPLify→HMR→SPIN→HMR2.0→PHALP→CoMotion | 없음 |
| 3 | 대안 기법 장단점 — 2D 키포인트 / 3D 스켈레톤 / 비파라메트릭 메쉬 / 3DGS 아바타 | 없음 |
| 4 | **왜 CoMotion + Multi-HMR 2 인가** — 실측 근거 | Phase C |
| 5 | VTG 활용 설계 — §2.3 의 ①~④를 구체 설계로 | Phase C |
| 6 | 참고문헌 — 전부 원문 확인 후 인용 | — |

작성 원칙: 논문 수치·주장은 **arXiv/CVF 원문 확인 후 인용**. 기억에 의존한 서술 금지.

---

## 9. 리스크

| 리스크 | 영향 | 대응 |
|---|---|---|
| Anny→SMPL 변환이 부정확 | Multi-HMR 2 결과의 포즈 지표 신뢰도 저하 | 게이트 3에서 차단. 실패 시 `joints3d` 기준 비교로 축소 |
| GT 관절 규약 해석 오류 | PA-MPJPE 전체가 무의미해짐 | 게이트 2 (재투영 육안 검증). 실패 시 무참조 지표 폴백 |
| Multi-HMR 2 속도 미공개 | Phase B 일정 불확실 | 파일럿 실측 후 산정 |
| 농구 카메라가 팬/줌 | 월드좌표 계열 기각 근거가 약해짐 | 파일럿에서 확인. 필요 시 별도 검토 |
| CoMotion 긴 가림 취약 (논문 명시) | 농구에서 ID 성능 저조 예상 | **이건 리스크가 아니라 측정 대상.** 예상되는 결과를 정량화하는 것이 목적 |

---

## 10. 범위 밖

- 멀티뷰 융합 (COLMAP 캘리브레이션 활용한 4캠 → 단일 3D SMPL) — **별도 스펙**
- VTG 파이프라인에 SMPL 신호를 실제로 통합하는 구현 — **별도 스펙** (본 문서는 설계까지)
- 모델 재학습·파인튜닝 — 추론 전용
- DanceHMR, PromptHMR 등 3번째 모델 추가 — Phase B 이후 선택 사항

---

## 11. 참고문헌

- Loper et al. **SMPL: A Skinned Multi-Person Linear Model.** SIGGRAPH Asia 2015.
- Pavlakos et al. **Expressive Body Capture: 3D Hands, Face, and Body from a Single Image (SMPL-X).** CVPR 2019.
- Newell et al. **CoMotion: Concurrent Multi-person 3D Motion.** ICLR 2025. [arXiv:2504.12186](https://arxiv.org/abs/2504.12186)
- **Multi-HMR 2: Multi-Person Camera-Centric Human Detection, Mesh Recovery and Tracking.** 2026. [arXiv:2606.14841](https://arxiv.org/abs/2606.14841)
- **Human Mesh Modeling for Anny Body.** 2025. [arXiv:2511.03589](https://arxiv.org/abs/2511.03589)
- Goel et al. **Humans in 4D: Reconstructing and Tracking Humans with Transformers.** ICCV 2023.
- Rajasegaran et al. **Tracking People by Predicting 3D Appearance, Location & Pose (PHALP).** CVPR 2022.
- Wang et al. **PromptHMR: Promptable Human Mesh Recovery.** CVPR 2025.

*(Phase D 에서 전체 목록으로 확장하며 각 항목은 원문 확인 후 확정)*


---

## 12. 실측에 따른 정정 (2026-08-25)

구현 중 실데이터를 분석해 설계 단계의 가정 두 개가 틀렸음을 확인했다.

### 12.1 GT 관절 규약 — H36M 가설 폐기, COCO 확정

설계 단계에서는 `obj*.bvh` 의 계층(`Hip→RightHip→…→RightWrist` 16관절)을
근거로 Data2/3/4 GT 를 Human3.6M 17관절로 추정했다. **틀렸다.**
BVH 는 별도로 변환된 표현이고 `3DPose.txt` 는 COCO 순서를 쓴다.

| 근거 | 내용 |
|---|---|
| 좌표 구조 | idx 0~4 가 같은 높이 덩어리(얼굴), 이후 (5,6)(7,8)(9,10)(11,12)(13,14)(15,16) 좌우쌍이 차례로 내려옴. H36M 의 다리-먼저 순서와 불일치 |
| 뼈길이 CV | Data2 COCO **0.046** vs H36M 0.106 / Data3 **0.056** vs 0.066 / Data4 **0.062** vs 0.085 |
| 인체 비율 | COCO 해석 시 허벅지≈정강이, 상완>전완, 허벅지/상완 1.44~1.63 (실제 인체 ≈1.5) |

**Data1 은 COCO-17 + 발 2개 = 19관절.** idx 17·18 은 같은 쪽 발목과 0.58,
반대쪽 발목과 2.3 거리 → SMPL `left_foot`/`right_foot` 에 매핑.

**COCO 계열에는 골반 관절이 없다.** 두 가지 파급이 있다.
- `MPJPE` 는 루트 정렬이 필요하므로 산출 불가 → `mpjpe_available: False` 로 명시하고 NaN 반환. **PA-MPJPE 가 주 지표**가 된다 (전역 정렬을 스스로 하므로 골반 불필요).
- `transl` 은 좌우 엉덩이 중점으로 대체한다.

### 12.2 GT 스케일이 데이터셋마다 다르다

GT 는 COLMAP/SfM 재구성 결과라 **임의 스케일**이다. 골격 크기(중심으로부터
평균거리)가 Data2 0.236, Data3 0.425, Data4 1.055, Data1 1.460 으로 제각각이다.

Procrustes 는 스케일을 맞추므로 잔차가 **GT 단위**로 나오는데 그 단위가
데이터셋마다 다르다. 따라서 PA-MPJPE 를 mm 로 직역하면 안 된다.
`skeleton_scale` 로 정규화한 뒤 표준 인체 크기(**544.6 mm**, 키 1.5m T-포즈
기준 중심-평균거리)를 곱해 데이터셋 간 비교가 가능한 값으로 환산한다.

### 12.3 GT 자체의 노이즈 수준

정정 후에도 GT 뼈길이 CV 는 **0.086~0.224** 다 (정정 전 0.45~0.70).
삼각측량 GT 자체가 8~22% 의 골격 길이 변동을 갖는다는 뜻이며,
**모델 정확도 주장의 하한선**이 된다. 리포트에 반드시 함께 기재한다.

### 12.4 Data1 GT 인원은 세션마다 다르다

S03·S12 는 13명, S06·S08 은 15명. 설계 단계의 "13명" 은 S03 만 본 수치였다.
