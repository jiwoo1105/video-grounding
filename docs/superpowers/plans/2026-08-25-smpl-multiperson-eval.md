# 다중인물 SMPL 추정 모델 평가 — 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Captured-Motion-Dataset 28영상에 CoMotion과 Multi-HMR 2를 적용하고, 포즈 정확도와 ID 유지를 동일 조건에서 정량 비교한다.

**Architecture:** 두 모델의 이질적 출력을 어댑터가 `tracks.npz` 단일 스키마로 정규화하고, 모든 지표 코드는 모델을 모른 채 동작한다. 지표 엔진을 **합성 데이터로 먼저 TDD 검증**한 뒤 실제 모델 출력을 연결한다 — GPU 시간을 쓰기 전에 측정 도구부터 믿을 수 있게 만든다.

**Tech Stack:** Python 3.10, numpy, scipy, py-motmetrics, ffmpeg/ffprobe, PyTorch (엘리스 GPU), CoMotion, Multi-HMR 2

**Spec:** `docs/superpowers/specs/2026-08-24-smpl-multiperson-eval-design.md`

## Global Constraints

- 코드 위치: `smpl_eval/` (저장소 루트 기준)
- 데이터셋 루트: `Captured-Motion-Dataset/` (저장소 루트 기준, git 미추적)
- Python 3.10. 로컬 개발/테스트는 macOS(MPS/CPU), 모델 추론은 엘리스클라우드 CUDA
- 관절 정본 규약: **SMPL 24관절 표준 순서** (`0 pelvis … 23 right_hand`)
- `tracks.npz` 모든 배열의 첫 축 길이 N 동일. dtype 스펙 §4.2 준수
- 테스트: `pytest`. 모든 지표 모듈은 합성 데이터 유닛테스트를 반드시 동반
- 커밋 메시지는 한국어 한 줄 요약 + 필요 시 본문

## 데이터 실측 정보 (2026-08-25 확인)

| 데이터 | GT 파일 | 필드수 | 관절 | 프레임 범위 | 인원 | 특이사항 |
|---|---|---|---|---|---|---|
| Data1 3D | `PoseResults3d_*.txt` | 78 = 2+19×4 | **19** | 1..300 (1-base) | 13 | `frame,pid,(x,y,z,conf)×19` |
| Data1 2D | `PoseResults2d_*.txt` | 64 = 3+4+19×3 | 19 | cam 0..3, frame 1..300 | — | `cam,frame,pid,bbox(x,y,w,h),(x,y,conf)×19`. **깨진 줄 6개 존재** |
| Data2 | `3DPose.txt` | 70 = 2+17×4 | **17 (H36M)** | 0..2307 (영상 2309f) | 2 (`0,1`) | **복식인데 GT는 2명뿐** |
| Data3 | `3Dpose.txt` (소문자 p) | 70 | 17 | 2..1420 (영상 1420f) | 2 (`1,2`) | |
| Data4 | `3DPose.txt` | 70 | 17 | 2..659 (영상 660f) | 3 (`1,2,3`) | **`nan` 값 존재** |

> ⚠️ **정정 (2026-08-25, Task 8 실행 중 확정)**
> 위 표의 "17 (H36M)" 은 **틀렸다.** BVH 계층을 근거로 한 추정이었으나
> 실측 결과 `3DPose.txt` 는 **COCO 순서**를 쓴다. 자세한 근거는 스펙 §12.1.

**실제 관절 규약 (실측 확정)**

- Data2/3/4 = **COCO-17**
  `0 nose, 1 L_eye, 2 R_eye, 3 L_ear, 4 R_ear, 5 L_shoulder, 6 R_shoulder,
   7 L_elbow, 8 R_elbow, 9 L_wrist, 10 R_wrist, 11 L_hip, 12 R_hip,
   13 L_knee, 14 R_knee, 15 L_ankle, 16 R_ankle`
- Data1 = **COCO-17 + [17 left_foot, 18 right_foot]** = 19관절

**파급**: COCO 계열에는 골반이 없어 MPJPE 산출 불가 → PA-MPJPE 가 주 지표.
GT 스케일이 데이터셋마다 달라(0.236~1.460) 정규화가 필수. 스펙 §12.2 참조.

---

## 파일 구조

```
smpl_eval/
├── README.md
├── requirements.txt
├── setup_elice.sh              # 엘리스 원샷 환경구축
├── schema.py                   # tracks.npz 스키마 정의·검증·입출력
├── dataset_index.py            # 28영상 스캔 → manifest.json
├── run_all.py                  # 오케스트레이션
├── conventions.py              # 관절 규약 정의 + 규약 간 매핑
├── gt/
│   ├── parse_pose3d.py         # Data1 19관절 / Data2-4 H36M-17
│   ├── parse_pose2d.py         # Data1 2D
│   └── reproject_check.py      # 게이트 2 검증 도구
│   # parse_bvh.py 는 만들지 않는다 — 사유는 아래 참조
├── runners/
│   ├── base.py                 # 러너 공통 인터페이스
│   ├── comotion.py
│   └── multihmr2.py
├── metrics/
│   ├── identity.py             # IDF1/MOTA/ID-switch/단편화/인원수
│   ├── plausibility.py         # 길이분산/지터/관절각/β분산  (GT 불필요)
│   ├── occlusion.py            # 가림 구간 검출 + 구간별 ID 유지율
│   ├── pose.py                 # PA-MPJPE/MPJPE/2D재투영
│   └── handsize.py             # 손 픽셀 크기 실측
├── report/
│   └── build.py                # 집계 + worst-K + HTML
└── tests/
    ├── synth.py                # 합성 tracks 생성기 (테스트 공용)
    ├── test_schema.py
    ├── test_dataset_index.py
    ├── test_identity.py
    ├── test_plausibility.py
    ├── test_occlusion.py
    ├── test_pose.py
    ├── test_handsize.py
    └── test_gt_parse.py
```

**책임 분리 원칙**: `conventions.py`는 관절 규약만, `schema.py`는 포맷만, `metrics/*`는 각각 하나의 지표군만 담당한다. 러너는 외부 모델 호출과 정규화만 하고 지표를 모른다.

**스펙 대비 의도적 생략 — `gt/parse_bvh.py`**: 스펙 §4.3 은 BVH 파서를 나열했으나 구현하지 않는다. `obj*.bvh` 는 `3DPose.txt` 와 **동일한 사람의 동일한 모션**을 회전(rotation) 공간으로 표현한 것이고, 우리 지표(PA-MPJPE, MPJPE)는 **관절 위치**만 필요하다. BVH 를 쓰려면 순운동학(FK) 전개 + 별도 관절 규약 매핑이 추가되는데, 얻는 정보는 `3DPose.txt` 와 같다. YAGNI 로 제외한다. (BVH 는 Task 8 에서 H36M-17 관절 순서를 확정하는 **근거 자료**로만 사용했다.)

---

## Task 1: 스캐폴드 + `tracks.npz` 스키마

**Files:**
- Create: `smpl_eval/schema.py`, `smpl_eval/tests/test_schema.py`, `smpl_eval/tests/synth.py`, `smpl_eval/requirements.txt`, `smpl_eval/README.md`

**Interfaces:**
- Produces:
  - `TRACK_FIELDS: dict[str, tuple[tuple, str]]` — 필드명 → (shape suffix, dtype)
  - `save_tracks(path: str, arrays: dict, meta: dict) -> None`
  - `load_tracks(path: str) -> tuple[dict, dict]` — (arrays, meta)
  - `validate_tracks(arrays: dict) -> None` — 위반 시 `ValueError`
  - `synth.make_tracks(n_frames, n_tracks, **kw) -> dict` (tests/synth.py)

- [ ] **Step 1: 디렉터리와 requirements 생성**

```bash
mkdir -p smpl_eval/{gt,runners,metrics,report,tests}
touch smpl_eval/{gt,runners,metrics,report,tests}/__init__.py smpl_eval/__init__.py
cat > smpl_eval/requirements.txt <<'EOF'
numpy>=1.24
scipy>=1.10
motmetrics>=1.4
pytest>=7.0
EOF
python3 -m pip install -r smpl_eval/requirements.txt
```

- [ ] **Step 2: 실패하는 테스트 작성**

`smpl_eval/tests/test_schema.py`:

```python
import numpy as np, pytest, tempfile, os
from smpl_eval.schema import save_tracks, load_tracks, validate_tracks, TRACK_FIELDS

def _minimal(n=5):
    return {
        "frame_ids": np.arange(n, dtype=np.int32),
        "track_ids": np.zeros(n, dtype=np.int32),
        "betas": np.zeros((n, 10), np.float32),
        "global_orient": np.zeros((n, 3), np.float32),
        "body_pose": np.zeros((n, 23, 3), np.float32),
        "transl": np.zeros((n, 3), np.float32),
        "joints3d": np.zeros((n, 24, 3), np.float32),
        "joints2d": np.zeros((n, 24, 2), np.float32),
        "bbox": np.zeros((n, 4), np.float32),
        "score": np.ones(n, np.float32),
    }

def test_validate_accepts_minimal():
    validate_tracks(_minimal())

def test_validate_rejects_length_mismatch():
    a = _minimal(); a["score"] = np.ones(4, np.float32)
    with pytest.raises(ValueError, match="length"):
        validate_tracks(a)

def test_validate_rejects_missing_field():
    a = _minimal(); del a["bbox"]
    with pytest.raises(ValueError, match="bbox"):
        validate_tracks(a)

def test_validate_rejects_wrong_shape():
    a = _minimal(); a["betas"] = np.zeros((5, 7), np.float32)
    with pytest.raises(ValueError, match="betas"):
        validate_tracks(a)

def test_roundtrip_preserves_arrays_and_meta():
    a = _minimal(); a["frame_ids"] = np.array([0,1,2,3,4], np.int32)
    meta = {"model": "comotion", "fps": 29.97, "body_model": "smpl"}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.npz")
        save_tracks(p, a, meta)
        b, m = load_tracks(p)
        assert m["model"] == "comotion" and m["fps"] == 29.97
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])

def test_optional_betas_native_roundtrips():
    a = _minimal(); a["betas_native"] = np.zeros((5, 6), np.float32)
    validate_tracks(a)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.npz")
        save_tracks(p, a, {"model": "multihmr2"})
        b, _ = load_tracks(p)
        assert b["betas_native"].shape == (5, 6)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'smpl_eval.schema'`

- [ ] **Step 4: `schema.py` 구현**

```python
"""tracks.npz 표준 포맷 — 모델별 출력을 여기로 정규화한다."""
import json, numpy as np

# 필드명 → (shape suffix, dtype).  실제 shape 는 (N,) + suffix
TRACK_FIELDS = {
    "frame_ids":     ((),        "int32"),
    "track_ids":     ((),        "int32"),
    "betas":         ((10,),     "float32"),
    "global_orient": ((3,),      "float32"),
    "body_pose":     ((23, 3),   "float32"),
    "transl":        ((3,),      "float32"),
    "joints3d":      ((24, 3),   "float32"),
    "joints2d":      ((24, 2),   "float32"),
    "bbox":          ((4,),      "float32"),
    "score":         ((),        "float32"),
}
OPTIONAL_FIELDS = {"betas_native"}   # shape (N, K), K 는 모델마다 다름


def validate_tracks(arrays):
    missing = [k for k in TRACK_FIELDS if k not in arrays]
    if missing:
        raise ValueError(f"필수 필드 누락: {', '.join(missing)}")
    n = len(arrays["frame_ids"])
    for k, (suffix, dt) in TRACK_FIELDS.items():
        a = np.asarray(arrays[k])
        if len(a) != n:
            raise ValueError(f"{k}: length {len(a)} != {n}")
        if a.shape != (n,) + suffix:
            raise ValueError(f"{k}: shape {a.shape} != {(n,) + suffix}")
    for k in arrays:
        if k in OPTIONAL_FIELDS and len(arrays[k]) != n:
            raise ValueError(f"{k}: length mismatch")


def save_tracks(path, arrays, meta):
    validate_tracks(arrays)
    payload = {k: np.asarray(v, dtype=TRACK_FIELDS[k][1]) if k in TRACK_FIELDS
               else np.asarray(v, dtype="float32")
               for k, v in arrays.items()}
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_tracks(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(d["__meta__"]).decode("utf-8"))
    arrays = {k: d[k] for k in d.files if k != "__meta__"}
    return arrays, meta
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_schema.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: 합성 데이터 생성기 작성**

`smpl_eval/tests/synth.py` — 이후 모든 지표 테스트가 이걸 쓴다.

```python
"""테스트용 합성 tracks 생성기. 지표 코드를 GPU 없이 검증하기 위한 것."""
import numpy as np


def make_tracks(n_frames=100, n_tracks=3, seed=0, beta_jump_at=None,
                id_swap_at=None, jitter=0.0, limb_noise=0.0):
    """규칙적으로 움직이는 n_tracks 명을 만든다.

    beta_jump_at: (frame, track) 에서 betas 를 크게 튀게 한다
    id_swap_at:   (frame, tA, tB) 부터 두 트랙의 궤적을 맞바꾼다
    jitter:       joints3d 에 가우시안 노이즈 표준편차 (m)
    limb_noise:   프레임마다 팔다리 길이를 흔드는 정도
    """
    rng = np.random.default_rng(seed)
    F, T = n_frames, n_tracks
    rows = F * T
    frame_ids = np.repeat(np.arange(F, dtype=np.int32), T)
    track_ids = np.tile(np.arange(T, dtype=np.int32), F)

    betas = np.repeat(rng.normal(0, 1, (T, 10)).astype(np.float32), F, axis=0)
    betas = betas.reshape(T, F, 10).transpose(1, 0, 2).reshape(rows, 10)

    # T-포즈 기준 골격을 x 방향으로 벌려 배치하고 시간에 따라 이동
    base = _tpose()                                    # (24,3)
    j3d = np.zeros((rows, 24, 3), np.float32)
    for f in range(F):
        for t in range(T):
            off = np.array([t * 1.5, 0.0, 5.0 + 0.01 * f], np.float32)
            scale = 1.0 + limb_noise * rng.normal()
            j3d[f * T + t] = base * scale + off
    if jitter:
        j3d += rng.normal(0, jitter, j3d.shape).astype(np.float32)

    if beta_jump_at is not None:
        f, t = beta_jump_at
        sel = (frame_ids >= f) & (track_ids == t)
        betas[sel] += 5.0

    if id_swap_at is not None:
        f, ta, tb = id_swap_at
        sel = frame_ids >= f
        ids = track_ids.copy()
        a = sel & (track_ids == ta); b = sel & (track_ids == tb)
        ids[a] = tb; ids[b] = ta
        track_ids = ids

    transl = j3d[:, 0, :].copy()
    j2d = np.stack([j3d[..., 0] / j3d[..., 2] * 1000 + 960,
                    j3d[..., 1] / j3d[..., 2] * 1000 + 540], -1).astype(np.float32)
    x1 = j2d[..., 0].min(1); x2 = j2d[..., 0].max(1)
    y1 = j2d[..., 1].min(1); y2 = j2d[..., 1].max(1)
    return {
        "frame_ids": frame_ids, "track_ids": track_ids.astype(np.int32),
        "betas": betas.astype(np.float32),
        "global_orient": np.zeros((rows, 3), np.float32),
        "body_pose": np.zeros((rows, 23, 3), np.float32),
        "transl": transl.astype(np.float32),
        "joints3d": j3d, "joints2d": j2d,
        "bbox": np.stack([x1, y1, x2, y2], -1).astype(np.float32),
        "score": np.ones(rows, np.float32),
    }


def _tpose():
    """SMPL 24관절의 대략적인 T-포즈 좌표 (미터). 테스트용 근사값."""
    p = np.zeros((24, 3), np.float32)
    p[0]  = (0.00,  0.00, 0)   # pelvis
    p[1]  = (0.09, -0.08, 0);  p[2]  = (-0.09, -0.08, 0)   # hips
    p[3]  = (0.00,  0.12, 0)                               # spine1
    p[4]  = (0.10, -0.48, 0);  p[5]  = (-0.10, -0.48, 0)   # knees
    p[6]  = (0.00,  0.25, 0)                               # spine2
    p[7]  = (0.10, -0.88, 0);  p[8]  = (-0.10, -0.88, 0)   # ankles
    p[9]  = (0.00,  0.35, 0)                               # spine3
    p[10] = (0.10, -0.95, 0.10); p[11] = (-0.10, -0.95, 0.10)  # feet
    p[12] = (0.00,  0.52, 0)                               # neck
    p[13] = (0.08,  0.45, 0);  p[14] = (-0.08, 0.45, 0)    # collars
    p[15] = (0.00,  0.62, 0)                               # head
    p[16] = (0.18,  0.45, 0);  p[17] = (-0.18, 0.45, 0)    # shoulders
    p[18] = (0.44,  0.45, 0);  p[19] = (-0.44, 0.45, 0)    # elbows
    p[20] = (0.68,  0.45, 0);  p[21] = (-0.68, 0.45, 0)    # wrists
    p[22] = (0.76,  0.45, 0);  p[23] = (-0.76, 0.45, 0)    # hands
    return p
```

- [ ] **Step 7: 합성기가 스키마를 만족하는지 확인하는 테스트 추가**

`smpl_eval/tests/test_schema.py` 끝에 추가:

```python
def test_synth_satisfies_schema():
    from smpl_eval.tests.synth import make_tracks
    validate_tracks(make_tracks(n_frames=10, n_tracks=3))
```

Run: `python3 -m pytest smpl_eval/tests/test_schema.py -v`
Expected: PASS (7 passed)

- [ ] **Step 8: README 작성 후 커밋**

`smpl_eval/README.md` 에 목적 3줄 + 실행 순서 + `tracks.npz` 스키마 표를 적는다 (스펙 §4.2 표를 복사).

```bash
git add smpl_eval/ && git commit -m "smpl_eval 스캐폴드 + tracks.npz 스키마"
```

---

## Task 2: `dataset_index.py` — manifest 생성

**Files:**
- Create: `smpl_eval/dataset_index.py`, `smpl_eval/tests/test_dataset_index.py`

**Interfaces:**
- Consumes: 없음
- Produces:
  - `scan(dataset_root: str) -> list[dict]` — 영상별 레코드 리스트
  - `write_manifest(dataset_root: str, out_path: str) -> list[dict]`
  - 레코드 키: `video_path, dataset, session, cam, width, height, fps, n_frames, gt_pose3d, gt_pose2d, gt_bvh, colmap_dir, expected_persons, gt_persons, gt_frame_offset`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import os, json, pytest
from smpl_eval.dataset_index import scan, write_manifest

ROOT = "Captured-Motion-Dataset"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="데이터셋 없음")

def test_scan_finds_28_videos():
    assert len(scan(ROOT)) == 28

def test_per_dataset_video_counts():
    from collections import Counter
    c = Counter(r["dataset"] for r in scan(ROOT))
    assert c["Data1"] == 16 and c["Data2"] == 4 and c["Data3"] == 4 and c["Data4"] == 4

def test_probe_fills_video_metadata():
    r = next(x for x in scan(ROOT) if x["dataset"] == "Data2")
    assert r["width"] == 1920 and r["height"] == 1080
    assert abs(r["fps"] - 29.97) < 0.05
    assert r["n_frames"] == 2309

def test_gt_paths_resolved():
    for r in scan(ROOT):
        assert os.path.isfile(r["gt_pose3d"]), r["video_path"]

def test_data1_has_2d_gt_others_do_not():
    for r in scan(ROOT):
        if r["dataset"] == "Data1":
            assert r["gt_pose2d"] and os.path.isfile(r["gt_pose2d"])
        else:
            assert r["gt_pose2d"] is None

def test_gt_person_counts_match_measured():
    exp = {"Data1": 13, "Data2": 2, "Data3": 2, "Data4": 3}
    for r in scan(ROOT):
        assert r["gt_persons"] == exp[r["dataset"]], r["video_path"]

def test_write_manifest_is_valid_json():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.json")
        write_manifest(ROOT, p)
        assert len(json.load(open(p))) == 28
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_dataset_index.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

`ffprobe` 로 해상도/fps/프레임수를 읽고, 데이터셋별 규칙으로 GT 경로를 매칭한다.

```python
"""데이터셋을 스캔해 manifest.json 을 만든다."""
import os, glob, json, subprocess

VIDEO_EXT = (".mp4",)

# 데이터셋 디렉터리 접두사 → 짧은 이름
DATASETS = {
    "Data1_SKNight-live_Basketball": "Data1",
    "Data2_WTA_tennis_double_clip3_1min_2K_30fps": "Data2",
    "Data3_OnlyOneOf_rie_junji_2K_60fps": "Data3",
    "Data4_vid3_golden_clip1_2K_60fps": "Data4",
}
# 화면에 실제로 등장하는 인원 (GT 인원과 다를 수 있음 — 테니스 복식이 대표 사례)
EXPECTED_PERSONS = {"Data1": None, "Data2": 4, "Data3": 2, "Data4": 3}


def _probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]),
            "fps": int(num) / int(den), "n_frames": int(s["nb_frames"])}


def _first(pattern):
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def _gt_person_count(path):
    ids = set()
    for line in open(path):
        f = line.strip().split(",")
        if len(f) < 3:
            continue
        ids.add(f[1].strip())
    return len(ids)


def scan(dataset_root):
    records = []
    for dirname, short in DATASETS.items():
        base = os.path.join(dataset_root, dirname)
        if not os.path.isdir(base):
            continue
        for vid in sorted(glob.glob(os.path.join(base, "*", "*.mp4"))):
            session = os.path.basename(os.path.dirname(vid))
            if session.endswith(("_pose_estimated", "_pose_gt",
                                 "_pose_result", "_mvg_result")):
                continue                      # 결과물 디렉터리는 입력이 아님
            cam = os.path.splitext(os.path.basename(vid))[0]
            if short == "Data1":
                gt3 = _first(os.path.join(base, session + "_pose_gt", "PoseResults3d_*.txt"))
                gt2 = _first(os.path.join(base, session + "_pose_gt", "PoseResults2d_*.txt"))
                bvh = []
            else:
                res = os.path.join(base, session + "_pose_result")
                gt3 = _first(os.path.join(res, "3DPose.txt")) or \
                      _first(os.path.join(res, "3Dpose.txt"))
                gt2 = None
                bvh = sorted(glob.glob(os.path.join(res, "obj*.bvh")))
            rec = {"video_path": vid, "dataset": short, "session": session, "cam": cam,
                   "gt_pose3d": gt3, "gt_pose2d": gt2, "gt_bvh": bvh,
                   "colmap_dir": _first(os.path.join(base, session + "_mvg_result", "colmap_text")),
                   "expected_persons": EXPECTED_PERSONS[short]}
            rec.update(_probe(vid))
            rec["gt_persons"] = _gt_person_count(gt3) if gt3 else None
            records.append(rec)
    return records


def write_manifest(dataset_root, out_path):
    recs = scan(dataset_root)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)
    return recs


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "Captured-Motion-Dataset"
    recs = write_manifest(root, "smpl_eval/manifest.json")
    print(f"{len(recs)} videos → smpl_eval/manifest.json")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_dataset_index.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: manifest 실제 생성 후 육안 확인**

Run: `python3 -m smpl_eval.dataset_index Captured-Motion-Dataset`
Expected: `28 videos → smpl_eval/manifest.json`

Run: `python3 -c "import json;m=json.load(open('smpl_eval/manifest.json'));print(sum(r['n_frames'] for r in m))"`
Expected: `22356`

- [ ] **Step 6: 커밋**

```bash
git add smpl_eval/dataset_index.py smpl_eval/tests/test_dataset_index.py
git commit -m "dataset_index: 28영상 스캔 → manifest.json"
```

---

## Task 3: `setup_elice.sh` — 엘리스 원샷 환경구축

**Files:**
- Create: `smpl_eval/setup_elice.sh`, `smpl_eval/ELICE.md`

**Interfaces:**
- Produces: 엘리스 인스턴스에 `~/comotion-env`, `~/multihmr2-env` 두 conda 환경과 체크포인트

이 태스크는 로컬에서 유닛테스트할 수 없다. **검증은 엘리스 인스턴스에서 실행해 스모크 테스트 통과 여부로 한다.**

- [ ] **Step 1: 스크립트 작성**

```bash
#!/usr/bin/env bash
# 엘리스클라우드 인스턴스에서 1회 실행. 재실행 안전(idempotent).
set -euo pipefail
ROOT="${HOME}/smpl_eval_env"
mkdir -p "$ROOT" && cd "$ROOT"

echo "=== 1/4  CoMotion ==="
if [ ! -d ml-comotion ]; then
  git clone https://github.com/apple/ml-comotion.git
fi
cd ml-comotion
conda env list | grep -q '^comotion ' || conda create -n comotion -y python=3.10
conda run -n comotion pip install -e '.[all]'
[ -f src/comotion_demo/data/comotion_detection_checkpoint.pt ] || bash get_pretrained_models.sh
cd "$ROOT"

echo "=== 2/4  Multi-HMR 2 ==="
if [ ! -d multi-hmr2 ]; then
  git clone https://github.com/naver/multi-hmr2.git
fi
cd multi-hmr2
conda env list | grep -q '^multihmr2 ' || conda create -n multihmr2 -y python=3.10
conda run -n multihmr2 pip install -e '.[render]'
cd "$ROOT"

echo "=== 3/4  SMPL 바디모델 확인 ==="
SMPL_DST="$ROOT/ml-comotion/src/comotion_demo/data/smpl/SMPL_NEUTRAL.pkl"
if [ ! -f "$SMPL_DST" ]; then
  echo "!! 수동 작업 필요:"
  echo "   https://smpl.is.tue.mpg.de/ 가입 → SMPL v1.1.0 neutral 다운로드"
  echo "   basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl 을 아래 경로로:"
  echo "   $SMPL_DST"
  exit 1
fi

echo "=== 4/4  평가 파이프라인 ==="
conda run -n comotion  pip install -r "${HOME}/Video_grounding/smpl_eval/requirements.txt"
conda run -n multihmr2 pip install -r "${HOME}/Video_grounding/smpl_eval/requirements.txt"
echo "완료. 스모크 테스트: bash smpl_eval/smoke_elice.sh"
```

- [ ] **Step 2: 스모크 테스트 스크립트 작성**

`smpl_eval/smoke_elice.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="${HOME}/smpl_eval_env"
V="${1:?사용법: smoke_elice.sh <영상경로>}"
OUT="${HOME}/smoke_out"; mkdir -p "$OUT"

echo "--- CoMotion (30프레임)"
cd "$ROOT/ml-comotion"
conda run -n comotion python demo.py -i "$V" -o "$OUT/comotion" --num-frames 30
ls -la "$OUT/comotion"

echo "--- Multi-HMR 2 (전체, 짧은 영상 사용 권장)"
cd "$ROOT/multi-hmr2"
conda run -n multihmr2 multihmr2 --checkpoint checkpoints/multihmr2.pt \
  --video "$V" --out "$OUT/multihmr2" --save_anny_params --render
ls -la "$OUT/multihmr2"
```

- [ ] **Step 3: 엘리스에서 실행 — 게이트 0**

```bash
chmod +x smpl_eval/setup_elice.sh smpl_eval/smoke_elice.sh
bash smpl_eval/setup_elice.sh
bash smpl_eval/smoke_elice.sh Captured-Motion-Dataset/Data1_SKNight-live_Basketball/S03_HL01_2K/Cam1_Deck0009_HL01_2K.mp4
```

**게이트 0 통과 조건**: 두 모델 모두 출력 파일을 생성하고, 렌더 영상에서 사람 위에 메쉬가 보인다.

- [ ] **Step 4: 실제 출력 파일 구조를 `ELICE.md` 에 기록**

Task 12·13의 어댑터를 쓰려면 **실제 출력 키 이름과 shape** 를 알아야 한다. 스모크 결과를 덤프해 기록한다.

```bash
conda run -n comotion python -c "
import torch,sys; d=torch.load(sys.argv[1],map_location='cpu')
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,f'{p}/{k}')
    elif hasattr(o,'shape'): print(f'{p}: {tuple(o.shape)} {o.dtype}')
    else: print(f'{p}: {type(o).__name__}')
walk(d)" ~/smoke_out/comotion/*.pt | tee -a smpl_eval/ELICE.md
```

Multi-HMR 2 도 동일하게 `.pkl` 을 덤프해 기록한다.

- [ ] **Step 5: 커밋**

```bash
git add smpl_eval/setup_elice.sh smpl_eval/smoke_elice.sh smpl_eval/ELICE.md
git commit -m "엘리스 환경구축 + 스모크 테스트 스크립트, 모델 출력 구조 기록"
```

---

## Task 4: `metrics/identity.py` — ID 유지 지표

**Files:**
- Create: `smpl_eval/metrics/identity.py`, `smpl_eval/tests/test_identity.py`

**Interfaces:**
- Consumes: `schema.load_tracks`, `tests.synth.make_tracks`
- Produces:
  - `id_metrics(pred: dict, gt: dict, iou_thresh: float = 0.5) -> dict`
    반환 키: `idf1, mota, motp, num_switches, num_fragmentations, mostly_tracked, mostly_lost`
  - `person_count_error(pred: dict, expected: int) -> dict`
    반환 키: `mean_count, count_mae, frames_over, frames_under`

**이 태스크가 스펙의 게이트 5(합성 ID 스왑 검출)를 해결한다.**

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.identity import id_metrics, person_count_error

def test_perfect_match_gives_idf1_one():
    t = make_tracks(n_frames=50, n_tracks=3)
    m = id_metrics(t, t)
    assert m["idf1"] > 0.99
    assert m["num_switches"] == 0

def test_id_swap_is_detected():
    gt = make_tracks(n_frames=50, n_tracks=3, seed=1)
    pred = make_tracks(n_frames=50, n_tracks=3, seed=1, id_swap_at=(25, 0, 1))
    m = id_metrics(pred, gt)
    assert m["num_switches"] >= 2, f"ID 스왑을 못 잡음: {m}"
    assert m["idf1"] < 0.9

def test_missing_track_lowers_mota():
    gt = make_tracks(n_frames=50, n_tracks=3, seed=2)
    keep = gt["track_ids"] != 2
    pred = {k: v[keep] for k, v in gt.items()}
    m = id_metrics(pred, gt)
    assert m["mota"] < 0.7

def test_person_count_error_on_exact():
    t = make_tracks(n_frames=30, n_tracks=4)
    r = person_count_error(t, expected=4)
    assert r["count_mae"] == 0.0 and r["mean_count"] == 4.0

def test_person_count_error_detects_over_detection():
    t = make_tracks(n_frames=30, n_tracks=6)
    r = person_count_error(t, expected=4)
    assert r["count_mae"] == 2.0 and r["frames_over"] == 30
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: smpl_eval.metrics.identity`

- [ ] **Step 3: 구현**

```python
"""ID 유지 지표. py-motmetrics 를 bbox IoU 기준으로 구동한다."""
import numpy as np
import motmetrics as mm


def _xyxy_to_xywh(b):
    b = np.asarray(b, float)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def id_metrics(pred, gt, iou_thresh=0.5):
    acc = mm.MOTAccumulator(auto_id=False)
    frames = np.union1d(np.unique(gt["frame_ids"]), np.unique(pred["frame_ids"]))
    for f in frames:
        g = gt["frame_ids"] == f
        p = pred["frame_ids"] == f
        gids = gt["track_ids"][g]
        pids = pred["track_ids"][p]
        if len(gids) and len(pids):
            d = mm.distances.iou_matrix(_xyxy_to_xywh(gt["bbox"][g]),
                                        _xyxy_to_xywh(pred["bbox"][p]),
                                        max_iou=1 - iou_thresh)
        else:
            d = np.empty((len(gids), len(pids)))
        acc.update(gids, pids, d, frameid=int(f))

    h = mm.metrics.create()
    s = h.compute(acc, metrics=["idf1", "mota", "motp", "num_switches",
                                "num_fragmentations", "mostly_tracked",
                                "mostly_lost"], name="v")
    r = {k: float(s[k].iloc[0]) for k in s.columns}
    r["num_switches"] = int(r["num_switches"])
    r["num_fragmentations"] = int(r["num_fragmentations"])
    return r


def person_count_error(pred, expected):
    frames, counts = np.unique(pred["frame_ids"], return_counts=True)
    return {
        "mean_count": float(counts.mean()),
        "count_mae": float(np.abs(counts - expected).mean()),
        "frames_over": int((counts > expected).sum()),
        "frames_under": int((counts < expected).sum()),
        "n_frames": int(len(frames)),
    }
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_identity.py -v`
Expected: PASS (5 passed)

`test_id_swap_is_detected` 가 통과하면 **스펙 게이트 5 충족** — 지표 코드가 실제로 ID 스왑을 잡아낸다는 증거다.

- [ ] **Step 5: 커밋**

```bash
git add smpl_eval/metrics/identity.py smpl_eval/tests/test_identity.py
git commit -m "metrics/identity: IDF1/MOTA/ID-switch + 인원수 오차, 합성 스왑 검출 검증"
```

---

## Task 5: `metrics/plausibility.py` — 무참조 지표

**Files:**
- Create: `smpl_eval/metrics/plausibility.py`, `smpl_eval/tests/test_plausibility.py`

**Interfaces:**
- Produces:
  - `limb_length_stats(tracks: dict) -> dict` — `mean_cv, max_cv, per_track`
  - `acceleration_jitter(tracks: dict, fps: float) -> dict` — `mean_accel, p95_accel`
  - `joint_angle_violations(tracks: dict) -> dict` — `n_violations, violation_rate`
  - `beta_consistency(tracks: dict) -> dict` — `mean_std, max_std, jump_frames`
  - `all_plausibility(tracks: dict, fps: float) -> dict` — 위 4개 병합

**설계 근거**: GT 매핑이 실패해도 결과를 낼 수 있는 폴백 경로이자, `betas` 가 `NaN`(Multi-HMR 2 변환 실패 시)일 때 `limb_length_stats` 가 대체 지표가 된다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.plausibility import (
    limb_length_stats, acceleration_jitter, joint_angle_violations,
    beta_consistency, all_plausibility)

def test_clean_tracks_have_near_zero_limb_variation():
    t = make_tracks(n_frames=60, n_tracks=2)
    assert limb_length_stats(t)["max_cv"] < 1e-4

def test_limb_noise_raises_variation():
    clean = make_tracks(n_frames=60, n_tracks=2, seed=3)
    noisy = make_tracks(n_frames=60, n_tracks=2, seed=3, limb_noise=0.05)
    assert limb_length_stats(noisy)["max_cv"] > limb_length_stats(clean)["max_cv"] * 10

def test_jitter_raises_acceleration():
    clean = make_tracks(n_frames=60, n_tracks=2, seed=4)
    noisy = make_tracks(n_frames=60, n_tracks=2, seed=4, jitter=0.02)
    assert acceleration_jitter(noisy, 30.0)["mean_accel"] > \
           acceleration_jitter(clean, 30.0)["mean_accel"] * 5

def test_beta_stable_for_clean_tracks():
    t = make_tracks(n_frames=60, n_tracks=3)
    assert beta_consistency(t)["max_std"] < 1e-5

def test_beta_jump_is_detected():
    t = make_tracks(n_frames=60, n_tracks=3, beta_jump_at=(30, 1))
    r = beta_consistency(t)
    assert r["max_std"] > 1.0
    assert 30 in r["jump_frames"] or 29 in r["jump_frames"] or 31 in r["jump_frames"]

def test_beta_nan_is_skipped_not_crashed():
    t = make_tracks(n_frames=20, n_tracks=2)
    t["betas"] = np.full_like(t["betas"], np.nan)
    r = beta_consistency(t)
    assert r["available"] is False

def test_all_plausibility_returns_all_keys():
    r = all_plausibility(make_tracks(n_frames=30, n_tracks=2), 30.0)
    for k in ("limb_max_cv", "mean_accel", "violation_rate", "beta_max_std"):
        assert k in r
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_plausibility.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
"""GT 불필요 지표. 물리적 일관성으로 추정 오류를 탐지한다."""
import numpy as np

# SMPL 24관절 기준 (부모, 자식) — 길이가 시간에 불변이어야 하는 뼈
LIMBS = [(1, 4), (4, 7), (2, 5), (5, 8),          # 다리
         (16, 18), (18, 20), (17, 19), (19, 21),  # 팔
         (0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]  # 척추·목
# 굽힘만 허용되는 관절: (부모, 관절, 자식)
HINGES = [(1, 4, 7), (2, 5, 8), (16, 18, 20), (17, 19, 21)]


def _by_track(tracks):
    for t in np.unique(tracks["track_ids"]):
        sel = tracks["track_ids"] == t
        order = np.argsort(tracks["frame_ids"][sel])
        yield int(t), sel, order


def limb_length_stats(tracks):
    """트랙 내 뼈 길이의 변동계수(CV). 같은 사람이면 0에 가까워야 한다."""
    per_track, cvs = {}, []
    for tid, sel, order in _by_track(tracks):
        j = tracks["joints3d"][sel][order]
        lens = np.stack([np.linalg.norm(j[:, a] - j[:, b], axis=-1)
                         for a, b in LIMBS], 1)          # (F, L)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.nanstd(lens, 0) / np.nanmean(lens, 0)
        cv = cv[np.isfinite(cv)]
        v = float(cv.max()) if cv.size else float("nan")
        per_track[tid] = v
        cvs.append(v)
    cvs = [c for c in cvs if np.isfinite(c)]
    return {"mean_cv": float(np.mean(cvs)) if cvs else float("nan"),
            "max_cv": float(np.max(cvs)) if cvs else float("nan"),
            "per_track": per_track}


def acceleration_jitter(tracks, fps):
    """관절 위치 2차 미분의 크기 (m/s^2). 떨림 정량화."""
    accels = []
    for _, sel, order in _by_track(tracks):
        j = tracks["joints3d"][sel][order]
        if len(j) < 3:
            continue
        a = np.diff(j, n=2, axis=0) * (fps ** 2)
        accels.append(np.linalg.norm(a, axis=-1).ravel())
    if not accels:
        return {"mean_accel": float("nan"), "p95_accel": float("nan")}
    a = np.concatenate(accels)
    a = a[np.isfinite(a)]
    return {"mean_accel": float(a.mean()), "p95_accel": float(np.percentile(a, 95))}


def joint_angle_violations(tracks, min_deg=5.0):
    """무릎·팔꿈치가 펴지는 방향으로 꺾였는지. 각도가 min_deg 미만이면 위반."""
    n_bad = n_tot = 0
    for _, sel, order in _by_track(tracks):
        j = tracks["joints3d"][sel][order]
        for a, b, c in HINGES:
            v1, v2 = j[:, a] - j[:, b], j[:, c] - j[:, b]
            n1 = np.linalg.norm(v1, axis=-1); n2 = np.linalg.norm(v2, axis=-1)
            ok = (n1 > 1e-6) & (n2 > 1e-6)
            cos = np.clip((v1 * v2).sum(-1)[ok] / (n1[ok] * n2[ok]), -1, 1)
            deg = np.degrees(np.arccos(cos))
            n_bad += int((deg < min_deg).sum()); n_tot += int(ok.sum())
    return {"n_violations": n_bad,
            "violation_rate": float(n_bad / n_tot) if n_tot else float("nan")}


def beta_consistency(tracks, jump_sigma=3.0):
    """β 는 신원이므로 트랙 내에서 불변이어야 한다. 급변은 ID 스왑 의심."""
    b = tracks["betas"]
    if not np.isfinite(b).any():
        return {"available": False, "mean_std": float("nan"),
                "max_std": float("nan"), "jump_frames": []}
    stds, jumps = [], []
    for _, sel, order in _by_track(tracks):
        bb = b[sel][order]
        f = tracks["frame_ids"][sel][order]
        s = float(np.nanstd(bb, 0).max())
        stds.append(s)
        if len(bb) > 2:
            d = np.linalg.norm(np.diff(bb, axis=0), axis=-1)
            thr = d.mean() + jump_sigma * d.std()
            if d.std() > 0:
                jumps += [int(f[i + 1]) for i in np.where(d > thr)[0]]
    return {"available": True,
            "mean_std": float(np.mean(stds)), "max_std": float(np.max(stds)),
            "jump_frames": sorted(set(jumps))}


def all_plausibility(tracks, fps):
    ll = limb_length_stats(tracks)
    aj = acceleration_jitter(tracks, fps)
    ja = joint_angle_violations(tracks)
    bc = beta_consistency(tracks)
    return {"limb_mean_cv": ll["mean_cv"], "limb_max_cv": ll["max_cv"],
            "mean_accel": aj["mean_accel"], "p95_accel": aj["p95_accel"],
            "n_violations": ja["n_violations"], "violation_rate": ja["violation_rate"],
            "beta_available": bc["available"], "beta_mean_std": bc["mean_std"],
            "beta_max_std": bc["max_std"], "beta_jump_frames": bc["jump_frames"]}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_plausibility.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add smpl_eval/metrics/plausibility.py smpl_eval/tests/test_plausibility.py
git commit -m "metrics/plausibility: 팔다리길이 CV/지터/관절각/β일관성 (GT 불필요)"
```

---

## Task 6: `metrics/occlusion.py` — 가림 구간 분석

**Files:**
- Create: `smpl_eval/metrics/occlusion.py`, `smpl_eval/tests/test_occlusion.py`

**Interfaces:**
- Produces:
  - `find_occlusion_events(tracks: dict, iou_thresh: float = 0.3, min_len: int = 3) -> list[dict]`
    이벤트 키: `start_frame, end_frame, track_a, track_b, peak_iou`
  - `id_retention_around_events(pred: dict, gt: dict, events: list, margin: int = 10) -> dict`
    반환 키: `n_events, retained, retention_rate`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.occlusion import find_occlusion_events, id_retention_around_events

def _crossing(n_frames=60):
    """두 사람이 x축에서 교차해 중간에 bbox 가 겹치는 시퀀스."""
    t = make_tracks(n_frames=n_frames, n_tracks=2, seed=7)
    f = t["frame_ids"]; k = t["track_ids"]
    for i in range(len(f)):
        shift = (f[i] - n_frames / 2) / (n_frames / 2) * 300
        s = shift if k[i] == 0 else -shift
        t["bbox"][i] = [500 + s, 100, 700 + s, 600]
    return t

def test_no_events_when_far_apart():
    t = make_tracks(n_frames=40, n_tracks=2)
    t["bbox"][t["track_ids"] == 0] = [0, 0, 100, 200]
    t["bbox"][t["track_ids"] == 1] = [900, 0, 1000, 200]
    assert find_occlusion_events(t) == []

def test_crossing_produces_one_event():
    ev = find_occlusion_events(_crossing())
    assert len(ev) == 1
    assert ev[0]["peak_iou"] > 0.3
    assert ev[0]["track_a"] == 0 and ev[0]["track_b"] == 1

def test_retention_is_one_when_ids_kept():
    t = _crossing()
    ev = find_occlusion_events(t)
    assert id_retention_around_events(t, t, ev)["retention_rate"] == 1.0

def test_retention_drops_when_ids_swap_at_event():
    gt = _crossing()
    ev = find_occlusion_events(gt)
    mid = (ev[0]["start_frame"] + ev[0]["end_frame"]) // 2
    pred = {k: v.copy() for k, v in gt.items()}
    sel = pred["frame_ids"] >= mid
    ids = pred["track_ids"].copy()
    a = sel & (pred["track_ids"] == 0); b = sel & (pred["track_ids"] == 1)
    ids[a] = 1; ids[b] = 0
    pred["track_ids"] = ids
    assert id_retention_around_events(pred, gt, ev)["retention_rate"] < 1.0
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_occlusion.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
"""bbox IoU 로 가림 이벤트를 자동 검출하고 그 구간의 ID 유지를 평가한다."""
import numpy as np
from collections import defaultdict


def _iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def find_occlusion_events(tracks, iou_thresh=0.3, min_len=3):
    """IoU 가 임계값을 min_len 프레임 이상 연속으로 넘는 트랙쌍 구간."""
    per_frame = defaultdict(list)
    for i in range(len(tracks["frame_ids"])):
        per_frame[int(tracks["frame_ids"][i])].append(
            (int(tracks["track_ids"][i]), tracks["bbox"][i]))

    hot = defaultdict(list)                      # (a,b) → [(frame, iou)]
    for f in sorted(per_frame):
        items = per_frame[f]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (ta, ba), (tb, bb) = items[i], items[j]
                v = _iou(ba, bb)
                if v >= iou_thresh:
                    key = (min(ta, tb), max(ta, tb))
                    hot[key].append((f, v))

    events = []
    for (ta, tb), hits in hot.items():
        hits.sort()
        run = [hits[0]]
        for cur in hits[1:]:
            if cur[0] == run[-1][0] + 1:
                run.append(cur)
            else:
                _emit(events, ta, tb, run, min_len); run = [cur]
        _emit(events, ta, tb, run, min_len)
    return sorted(events, key=lambda e: e["start_frame"])


def _emit(events, ta, tb, run, min_len):
    if len(run) >= min_len:
        events.append({"start_frame": run[0][0], "end_frame": run[-1][0],
                       "track_a": ta, "track_b": tb,
                       "peak_iou": float(max(v for _, v in run))})


def id_retention_around_events(pred, gt, events, margin=10):
    """이벤트 전/후 margin 프레임에서 GT-예측 ID 대응이 유지되는가."""
    retained = 0
    for ev in events:
        before = _assoc(pred, gt, ev["start_frame"] - margin)
        after = _assoc(pred, gt, ev["end_frame"] + margin)
        common = set(before) & set(after)
        if common and all(before[g] == after[g] for g in common):
            retained += 1
    n = len(events)
    return {"n_events": n, "retained": retained,
            "retention_rate": float(retained / n) if n else float("nan")}


def _assoc(pred, gt, frame):
    """해당 프레임에서 GT track → 예측 track 의 최근접 bbox 대응."""
    g = gt["frame_ids"] == frame
    p = pred["frame_ids"] == frame
    out = {}
    if not g.any() or not p.any():
        return out
    gb, gi = gt["bbox"][g], gt["track_ids"][g]
    pb, pi = pred["bbox"][p], pred["track_ids"][p]
    for k in range(len(gi)):
        ious = [_iou(gb[k], pb[m]) for m in range(len(pi))]
        best = int(np.argmax(ious))
        if ious[best] > 0:
            out[int(gi[k])] = int(pi[best])
    return out
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_occlusion.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: 커밋**

```bash
git add smpl_eval/metrics/occlusion.py smpl_eval/tests/test_occlusion.py
git commit -m "metrics/occlusion: 가림 이벤트 자동검출 + 구간별 ID 유지율"
```

---

## Task 7: `conventions.py` + `metrics/pose.py` — 관절 매핑과 포즈 오차

**Files:**
- Create: `smpl_eval/conventions.py`, `smpl_eval/metrics/pose.py`, `smpl_eval/tests/test_pose.py`

**Interfaces:**
- Produces (`conventions.py`):
  - `SMPL24: list[str]`, `H36M17: list[str]`, `COCO17: list[str]`
  - `H36M17_TO_SMPL24: dict[int, int]` — GT 인덱스 → SMPL 인덱스
  - `COCO17_TO_SMPL24: dict[int, int]`
  - `common_indices(mapping: dict) -> tuple[np.ndarray, np.ndarray]` — (gt_idx, smpl_idx)
- Produces (`metrics/pose.py`):
  - `pa_mpjpe(pred_j: np.ndarray, gt_j: np.ndarray) -> np.ndarray` — 프레임별 오차(mm)
  - `mpjpe(pred_j, gt_j, root_idx: int = 0) -> np.ndarray`
  - `reprojection_error(pred_j2d, gt_j2d) -> np.ndarray` — 픽셀
  - `pose_metrics(pred: dict, gt: dict, mapping: dict, frame_offset: int = 0) -> dict`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import numpy as np
from smpl_eval.conventions import SMPL24, H36M17, H36M17_TO_SMPL24, common_indices
from smpl_eval.metrics.pose import pa_mpjpe, mpjpe, reprojection_error

def test_smpl24_has_24_named_joints():
    assert len(SMPL24) == 24 and SMPL24[0] == "pelvis" and SMPL24[23] == "right_hand"

def test_h36m17_has_17_joints():
    assert len(H36M17) == 17 and H36M17[0] == "Hip" and H36M17[16] == "RightWrist"

def test_mapping_targets_are_valid_smpl_indices():
    for g, s in H36M17_TO_SMPL24.items():
        assert 0 <= g < 17 and 0 <= s < 24

def test_common_indices_are_aligned_and_same_length():
    gi, si = common_indices(H36M17_TO_SMPL24)
    assert len(gi) == len(si) > 10

def test_pa_mpjpe_is_zero_for_identical():
    j = np.random.RandomState(0).randn(5, 14, 3)
    assert np.allclose(pa_mpjpe(j, j), 0, atol=1e-6)

def test_pa_mpjpe_invariant_to_rigid_transform_and_scale():
    rs = np.random.RandomState(1)
    j = rs.randn(4, 14, 3)
    q, _ = np.linalg.qr(rs.randn(3, 3))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    moved = (j * 2.5) @ q.T + np.array([10.0, -3.0, 7.0])
    assert np.allclose(pa_mpjpe(moved, j), 0, atol=1e-6)

def test_mpjpe_detects_translation_after_root_align_is_zero():
    rs = np.random.RandomState(2)
    j = rs.randn(3, 14, 3)
    assert np.allclose(mpjpe(j + 5.0, j), 0, atol=1e-6)

def test_mpjpe_nonzero_for_deformation():
    rs = np.random.RandomState(3)
    j = rs.randn(3, 14, 3)
    bad = j.copy(); bad[:, 5] += 0.1
    assert mpjpe(bad, j).mean() > 0

def test_reprojection_error_in_pixels():
    a = np.zeros((2, 10, 2)); b = np.zeros((2, 10, 2)); b[..., 0] = 3.0
    assert np.allclose(reprojection_error(a, b), 3.0)
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_pose.py -v`
Expected: FAIL — `ModuleNotFoundError: smpl_eval.conventions`

- [ ] **Step 3: `conventions.py` 구현**

```python
"""관절 규약 정의와 규약 간 매핑. 정본은 SMPL 24관절."""
import numpy as np

SMPL24 = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]

# Data2/3/4 GT. BVH 계층(16관절)에 Head 를 index 10 에 넣은 Human3.6M 표준 순서.
H36M17 = [
    "Hip", "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee",
    "LeftAnkle", "Spine", "Thorax", "Neck", "Head", "LeftShoulder",
    "LeftElbow", "LeftWrist", "RightShoulder", "RightElbow", "RightWrist",
]

# Data1 GT 19관절의 앞 17개 가설 (Task 11 재투영으로 검증 후 확정)
COCO17 = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

_S = {n: i for i, n in enumerate(SMPL24)}

H36M17_TO_SMPL24 = {
    0: _S["pelvis"],       1: _S["right_hip"],    2: _S["right_knee"],
    3: _S["right_ankle"],  4: _S["left_hip"],     5: _S["left_knee"],
    6: _S["left_ankle"],   7: _S["spine2"],       8: _S["spine3"],
    9: _S["neck"],        10: _S["head"],        11: _S["left_shoulder"],
    12: _S["left_elbow"], 13: _S["left_wrist"],  14: _S["right_shoulder"],
    15: _S["right_elbow"],16: _S["right_wrist"],
}

# 얼굴 5개(nose/eyes/ears)는 SMPL 에 대응 관절이 없어 제외한다.
COCO17_TO_SMPL24 = {
    5: _S["left_shoulder"],  6: _S["right_shoulder"],
    7: _S["left_elbow"],     8: _S["right_elbow"],
    9: _S["left_wrist"],    10: _S["right_wrist"],
    11: _S["left_hip"],     12: _S["right_hip"],
    13: _S["left_knee"],    14: _S["right_knee"],
    15: _S["left_ankle"],   16: _S["right_ankle"],
}


def common_indices(mapping):
    """정렬된 (gt_indices, smpl_indices) 쌍을 반환한다."""
    keys = sorted(mapping)
    return np.array(keys, int), np.array([mapping[k] for k in keys], int)
```

- [ ] **Step 4: `metrics/pose.py` 구현**

```python
"""GT 대비 포즈 오차. 단위는 3D=mm, 2D=픽셀."""
import numpy as np
from smpl_eval.conventions import common_indices


def _procrustes(pred, gt):
    """스케일·회전·평행이동을 gt 에 맞추도록 pred 를 정렬한다 (프레임별)."""
    mp = pred.mean(1, keepdims=True); mg = gt.mean(1, keepdims=True)
    p = pred - mp; g = gt - mg
    out = np.empty_like(pred)
    for i in range(len(pred)):
        u, s, vt = np.linalg.svd(p[i].T @ g[i])
        d = np.sign(np.linalg.det(u @ vt))
        r = u @ np.diag([1.0, 1.0, d]) @ vt
        var = (p[i] ** 2).sum()
        scale = (s[:2].sum() + d * s[2]) / var if var > 0 else 1.0
        out[i] = scale * (p[i] @ r) + mg[i]
    return out


def pa_mpjpe(pred_j, gt_j):
    """Procrustes 정렬 후 관절 평균 오차 (mm). 입력 단위는 m."""
    aligned = _procrustes(np.asarray(pred_j, float), np.asarray(gt_j, float))
    return np.linalg.norm(aligned - gt_j, axis=-1).mean(-1) * 1000.0


def mpjpe(pred_j, gt_j, root_idx=0):
    """루트 정렬 후 관절 평균 오차 (mm)."""
    p = np.asarray(pred_j, float) - np.asarray(pred_j, float)[:, root_idx:root_idx+1]
    g = np.asarray(gt_j, float) - np.asarray(gt_j, float)[:, root_idx:root_idx+1]
    return np.linalg.norm(p - g, axis=-1).mean(-1) * 1000.0


def reprojection_error(pred_j2d, gt_j2d):
    """2D 관절 픽셀 거리."""
    return np.linalg.norm(np.asarray(pred_j2d, float) - np.asarray(gt_j2d, float),
                          axis=-1).mean(-1)


def pose_metrics(pred, gt, mapping, frame_offset=0):
    """예측·GT 를 프레임+bbox IoU 로 짝지어 오차를 낸다.

    frame_offset: gt_frame = pred_frame + frame_offset
    """
    from smpl_eval.metrics.occlusion import _assoc
    gi, si = common_indices(mapping)
    # _assoc 은 양쪽 프레임 번호가 같다고 가정하므로 pred 를 GT 축으로 옮긴다
    shifted = dict(pred)
    shifted["frame_ids"] = pred["frame_ids"] + frame_offset
    pa, mp = [], []
    for f in np.unique(pred["frame_ids"]):
        gf = int(f) + frame_offset
        if not (gt["frame_ids"] == gf).any():
            continue
        for gtid, ptid in _assoc(shifted, gt, gf).items():
            gsel = (gt["frame_ids"] == gf) & (gt["track_ids"] == gtid)
            psel = (pred["frame_ids"] == f) & (pred["track_ids"] == ptid)
            if not (gsel.any() and psel.any()):
                continue
            gj = gt["joints3d"][gsel][0][gi]
            pj = pred["joints3d"][psel][0][si]
            if not (np.isfinite(gj).all() and np.isfinite(pj).all()):
                continue
            pa.append(pa_mpjpe(pj[None], gj[None])[0])
            mp.append(mpjpe(pj[None], gj[None])[0])
    if not pa:
        return {"pa_mpjpe": float("nan"), "mpjpe": float("nan"), "n_matched": 0}
    return {"pa_mpjpe": float(np.mean(pa)), "pa_mpjpe_p95": float(np.percentile(pa, 95)),
            "mpjpe": float(np.mean(mp)), "n_matched": len(pa)}
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_pose.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: 커밋**

```bash
git add smpl_eval/conventions.py smpl_eval/metrics/pose.py smpl_eval/tests/test_pose.py
git commit -m "conventions + metrics/pose: 관절 규약 매핑, PA-MPJPE/MPJPE/재투영오차"
```

---

## Task 8: GT 파서 — `gt/parse_pose3d.py`

**Files:**
- Create: `smpl_eval/gt/parse_pose3d.py`, `smpl_eval/gt/parse_pose2d.py`, `smpl_eval/tests/test_gt_parse.py`

**Interfaces:**
- Produces:
  - `parse_pose3d(path: str, n_joints: int) -> dict` — `frame_ids, track_ids, joints3d, conf` (joints3d shape `(N, n_joints, 3)`)
  - `to_gt_tracks(parsed: dict, mapping: dict, image_wh: tuple | None = None) -> dict` — `tracks.npz` 호환 dict (SMPL24 슬롯에 채우고 미대응은 `NaN`)
  - `parse_pose2d(path: str, cam_id: int) -> dict` — Data1 전용. `frame_ids, track_ids, bbox, joints2d, conf`
  - `detect_frame_offset(gt_frames: np.ndarray, n_video_frames: int) -> int`

**중요**: Data1 2D 에는 필드가 7개뿐인 깨진 줄이 6개 있다. 파서는 이를 **조용히 건너뛰지 말고 카운트해서 반환**한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import os, glob, numpy as np, pytest
from smpl_eval.gt.parse_pose3d import parse_pose3d, to_gt_tracks, detect_frame_offset
from smpl_eval.gt.parse_pose2d import parse_pose2d
from smpl_eval.conventions import H36M17_TO_SMPL24, COCO17_TO_SMPL24
from smpl_eval.schema import validate_tracks

ROOT = "Captured-Motion-Dataset"
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="데이터셋 없음")

D2 = f"{ROOT}/Data2_WTA_tennis_double_clip3_1min_2K_30fps/tennis_double_clip3_1min_2K_pose_result/3DPose.txt"
D4 = f"{ROOT}/Data4_vid3_golden_clip1_2K_60fps/3_golden_clip1_pose_result/3DPose.txt"
D1_3D = glob.glob(f"{ROOT}/Data1_SKNight-live_Basketball/S03_HL01_2K_pose_gt/PoseResults3d_*.txt")[0]
D1_2D = glob.glob(f"{ROOT}/Data1_SKNight-live_Basketball/S03_HL01_2K_pose_gt/PoseResults2d_*.txt")[0]

def test_data2_parses_17_joints_two_persons():
    p = parse_pose3d(D2, n_joints=17)
    assert p["joints3d"].shape[1:] == (17, 3)
    assert set(np.unique(p["track_ids"])) == {0, 1}
    assert len(p["frame_ids"]) == 4616

def test_data1_parses_19_joints_thirteen_persons():
    p = parse_pose3d(D1_3D, n_joints=19)
    assert p["joints3d"].shape[1:] == (19, 3)
    assert len(np.unique(p["track_ids"])) == 13
    assert p["frame_ids"].min() == 1 and p["frame_ids"].max() == 300

def test_data4_nan_is_preserved_not_zeroed():
    p = parse_pose3d(D4, n_joints=17)
    assert np.isnan(p["joints3d"]).any(), "nan 이 0으로 뭉개짐"

def test_frame_offset_detected_for_data2():
    p = parse_pose3d(D2, n_joints=17)
    assert detect_frame_offset(p["frame_ids"], 2309) == 0

def test_frame_offset_detected_for_data4():
    p = parse_pose3d(D4, n_joints=17)
    assert detect_frame_offset(p["frame_ids"], 660) == 2

def test_to_gt_tracks_is_schema_valid():
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, H36M17_TO_SMPL24, image_wh=(1920, 1080))
    validate_tracks(t)
    assert t["joints3d"].shape[1:] == (24, 3)

def test_unmapped_smpl_slots_are_nan():
    p = parse_pose3d(D2, n_joints=17)
    t = to_gt_tracks(p, H36M17_TO_SMPL24, image_wh=(1920, 1080))
    assert np.isnan(t["joints3d"][:, 22]).all()   # left_hand 는 GT 에 없음

def test_data1_2d_reports_malformed_lines():
    p = parse_pose2d(D1_2D, cam_id=0)
    assert p["n_malformed"] >= 0
    assert p["joints2d"].shape[1:] == (19, 2)
    assert p["bbox"].shape[1] == 4

def test_data1_2d_filters_by_camera():
    a = parse_pose2d(D1_2D, cam_id=0)
    b = parse_pose2d(D1_2D, cam_id=1)
    assert len(a["frame_ids"]) > 0 and len(b["frame_ids"]) > 0
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_gt_parse.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `parse_pose3d.py` 구현**

```python
"""GT 3D 포즈 파서.

Data1  PoseResults3d_*.txt : frame, pid, (x,y,z,conf) * 19
Data2-4 3DPose.txt         : frame, pid, (x,y,z,conf) * 17
"""
import numpy as np
from smpl_eval.conventions import SMPL24


def parse_pose3d(path, n_joints):
    expected = 2 + n_joints * 4
    frames, pids, joints, confs, n_bad = [], [], [], [], 0
    for line in open(path):
        f = [x.strip() for x in line.strip().split(",") if x.strip() != ""]
        if len(f) != expected:
            n_bad += 1
            continue
        vals = np.array([float(x) for x in f[2:]], np.float32).reshape(n_joints, 4)
        frames.append(int(float(f[0]))); pids.append(int(float(f[1])))
        joints.append(vals[:, :3]); confs.append(vals[:, 3])
    return {"frame_ids": np.array(frames, np.int32),
            "track_ids": np.array(pids, np.int32),
            "joints3d": np.array(joints, np.float32),
            "conf": np.array(confs, np.float32),
            "n_malformed": n_bad}


def detect_frame_offset(gt_frames, n_video_frames):
    """GT 프레임 번호를 0-base 영상 인덱스로 맞추는 오프셋.

    반환값 k 에 대해  video_index = gt_frame - k.
    """
    lo = int(gt_frames.min())
    span = int(gt_frames.max()) - lo + 1
    if span > n_video_frames:
        raise ValueError(f"GT 프레임 범위({span})가 영상 프레임수({n_video_frames})보다 큼")
    return lo


def to_gt_tracks(parsed, mapping, image_wh=None):
    """파싱 결과를 tracks.npz 호환 dict 로 변환. 미대응 SMPL 슬롯은 NaN."""
    n = len(parsed["frame_ids"])
    j3 = np.full((n, 24, 3), np.nan, np.float32)
    for g, s in mapping.items():
        j3[:, s] = parsed["joints3d"][:, g]

    # 약원근 투영으로 근사 2D 를 만들어 bbox 를 얻는다 (IoU 매칭용).
    if image_wh:
        w, h = image_wh
        z = np.where(np.abs(j3[..., 2]) < 1e-6, 1e-6, j3[..., 2])
        fx = max(w, h)
        j2 = np.stack([j3[..., 0] / z * fx + w / 2,
                       j3[..., 1] / z * fx + h / 2], -1).astype(np.float32)
    else:
        j2 = np.full((n, 24, 2), np.nan, np.float32)

    bbox = np.zeros((n, 4), np.float32)
    for i in range(n):
        v = j2[i][np.isfinite(j2[i]).all(-1)]
        if len(v):
            bbox[i] = [v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()]

    return {
        "frame_ids": parsed["frame_ids"].astype(np.int32),
        "track_ids": parsed["track_ids"].astype(np.int32),
        "betas": np.full((n, 10), np.nan, np.float32),
        "global_orient": np.full((n, 3), np.nan, np.float32),
        "body_pose": np.full((n, 23, 3), np.nan, np.float32),
        "transl": j3[:, 0].copy(),
        "joints3d": j3, "joints2d": j2, "bbox": bbox,
        "score": parsed["conf"].mean(-1).astype(np.float32),
    }
```

- [ ] **Step 4: `parse_pose2d.py` 구현**

```python
"""Data1 전용 2D GT 파서.

형식: cam, frame, pid, bx, by, bw, bh, (x,y,conf) * 19   → 총 64 필드
깨진 줄(필드 7개)이 존재하므로 개수를 세어 반환한다.
"""
import numpy as np

N_JOINTS = 19
EXPECTED = 3 + 4 + N_JOINTS * 3


def parse_pose2d(path, cam_id):
    frames, pids, bboxes, joints, confs, n_bad = [], [], [], [], [], 0
    for line in open(path):
        f = [x.strip() for x in line.strip().split(",") if x.strip() != ""]
        if len(f) != EXPECTED:
            n_bad += 1
            continue
        if int(float(f[0])) != cam_id:
            continue
        bx, by, bw, bh = (float(x) for x in f[3:7])
        vals = np.array([float(x) for x in f[7:]], np.float32).reshape(N_JOINTS, 3)
        frames.append(int(float(f[1]))); pids.append(int(float(f[2])))
        bboxes.append([bx, by, bx + bw, by + bh])
        joints.append(vals[:, :2]); confs.append(vals[:, 2])
    return {"frame_ids": np.array(frames, np.int32),
            "track_ids": np.array(pids, np.int32),
            "bbox": np.array(bboxes, np.float32),
            "joints2d": np.array(joints, np.float32),
            "conf": np.array(confs, np.float32),
            "n_malformed": n_bad}
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_gt_parse.py -v`
Expected: PASS (9 passed)

- [ ] **Step 6: 전체 테스트 통과 확인 후 커밋**

Run: `python3 -m pytest smpl_eval/tests/ -v`
Expected: 모두 PASS

```bash
git add smpl_eval/gt/ smpl_eval/tests/test_gt_parse.py
git commit -m "gt 파서: Data1 19관절/Data2-4 H36M-17, nan·깨진줄·프레임오프셋 처리"
```

---

## Task 9: `gt/reproject_check.py` — 게이트 2 도구 + Data1 규약 확정

**Files:**
- Create: `smpl_eval/gt/reproject_check.py`

**Interfaces:**
- Consumes: `parse_pose2d`, `parse_pose3d`, `dataset_index.scan`
- Produces:
  - `overlay_gt2d(video_path, gt2d, out_png, frames=(0, 50, 100)) -> list[str]`
  - `main()` — CLI. 데이터셋별 오버레이 이미지를 생성

**목적**: Data1 의 19관절 규약을 **경험적으로 확정**하고, 프레임 오프셋이 맞는지 육안 검증한다. 스펙 게이트 2.

- [ ] **Step 1: 구현**

```python
"""GT 를 영상 위에 그려 관절 규약과 프레임 정렬을 육안 검증한다."""
import os, sys, numpy as np, subprocess, json

# COCO-17 가설의 골격 연결 (0-base). 뒤 2개(17,18)는 미지 → 점만 찍는다.
COCO_LINKS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _read_frame(video_path, idx):
    """ffmpeg 로 특정 프레임 하나를 PNG 바이트로 뽑는다."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-f", "image2pipe",
         "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return out


def overlay_gt2d(video_path, gt2d, out_dir, frames=(0, 50, 100), n_label=True):
    """gt2d 의 관절을 프레임 위에 번호와 함께 그린다."""
    from PIL import Image, ImageDraw
    import io
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for idx in frames:
        sel = gt2d["frame_ids"] == idx
        if not sel.any():
            continue
        img = Image.open(io.BytesIO(_read_frame(video_path, idx))).convert("RGB")
        d = ImageDraw.Draw(img)
        for row in np.where(sel)[0]:
            j = gt2d["joints2d"][row]
            c = gt2d["conf"][row]
            b = gt2d["bbox"][row]
            d.rectangle(list(b), outline=(0, 255, 0), width=2)
            for a, bb in COCO_LINKS:
                if c[a] > 0.3 and c[bb] > 0.3:
                    d.line([tuple(j[a]), tuple(j[bb])], fill=(255, 200, 0), width=2)
            for k in range(len(j)):
                if c[k] > 0.3:
                    x, y = j[k]
                    d.ellipse([x-3, y-3, x+3, y+3], fill=(255, 0, 0))
                    if n_label:
                        d.text((x + 4, y - 6), str(k), fill=(255, 255, 255))
        p = os.path.join(out_dir, f"frame_{idx:05d}.png")
        img.save(p); written.append(p)
    return written


def main():
    from smpl_eval.gt.parse_pose2d import parse_pose2d
    root = sys.argv[1] if len(sys.argv) > 1 else "Captured-Motion-Dataset"
    m = json.load(open("smpl_eval/manifest.json"))
    rec = next(r for r in m if r["dataset"] == "Data1" and r["cam"].startswith("Cam1"))
    gt2d = parse_pose2d(rec["gt_pose2d"], cam_id=0)
    print(f"깨진 줄: {gt2d['n_malformed']}개")
    # GT frame 은 1-base 이므로 영상 인덱스로 -1 보정
    gt2d["frame_ids"] = gt2d["frame_ids"] - 1
    out = overlay_gt2d(rec["video_path"], gt2d, "smpl_eval/gate_out/data1_gt2d",
                       frames=(0, 100, 200))
    for p in out:
        print("wrote", p)
    print("\n확인 사항:")
    print(" 1) 초록 bbox 가 실제 사람을 감싸는가 → 프레임 정렬 OK")
    print(" 2) 노란 선이 팔다리를 따라가는가 → COCO-17 가설 OK")
    print(" 3) 빨간 점 번호 17·18 이 어디에 찍히는가 → 나머지 2관절 정체 확인")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행 후 육안 확인 — 게이트 2**

```bash
python3 -m pip install pillow
python3 -m smpl_eval.gt.reproject_check
open smpl_eval/gate_out/data1_gt2d/
```

**게이트 2 통과 조건**: bbox 가 사람을 감싸고, 노란 골격선이 팔다리를 따라간다.

- [ ] **Step 3: 확인 결과를 `conventions.py` 에 반영**

관절 17·18 의 정체를 이미지에서 읽어 `conventions.py` 의 `COCO17` 주석과 `DATA1_19` 상수를 확정한다. 가설이 틀렸다면 매핑을 수정하고 Task 8 테스트를 다시 돌린다.

가설이 검증 불가로 판명되면 **스펙 §6 폴백 발동**: Data1 은 GT 기반 지표를 포기하고 무참조 지표로만 평가한다. 그 결정을 `smpl_eval/GATES.md` 에 기록한다.

- [ ] **Step 4: 커밋**

```bash
git add smpl_eval/gt/reproject_check.py smpl_eval/conventions.py smpl_eval/GATES.md
git commit -m "게이트2 도구: GT 재투영 오버레이로 Data1 관절규약·프레임정렬 검증"
```

---

## Task 10: `runners/comotion.py` — CoMotion 어댑터

**Files:**
- Create: `smpl_eval/runners/base.py`, `smpl_eval/runners/comotion.py`

**Interfaces:**
- Consumes: `schema.save_tracks`
- Produces:
  - `base.Runner` — `run(video_path: str, out_dir: str, video_meta: dict, max_frames: int | None = None, force: bool = False) -> str` (tracks.npz 경로). `video_meta` 는 manifest 레코드 그대로 (`width/height/fps/n_frames` 사용)
  - `comotion.CoMotionRunner(repo_dir: str, env: str = "comotion")`

**전제**: Task 3 Step 4 에서 기록한 `smpl_eval/ELICE.md` 의 실제 `.pt` 키 구조. **그 구조를 보기 전에는 이 태스크를 시작하지 않는다.**

아래 코드의 `KEY_*` 상수는 자리표시자가 아니라 **`ELICE.md` 덤프에서 읽어 확정하는 값**이다. Task 3 Step 4 를 실행하면 `.pt` 안의 키 이름과 shape 가 전부 출력되므로, 그 출력의 키 이름을 그대로 적어 넣으면 이 태스크는 완결된다. 덤프가 없으면 이 태스크는 시작 조건 미충족이다.

- [ ] **Step 1: 공통 인터페이스 작성**

```python
"""러너 공통 인터페이스. 외부 모델 호출과 tracks.npz 정규화만 책임진다."""
import os, abc, subprocess, time


class Runner(abc.ABC):
    name = "base"
    body_model = "smpl"

    @abc.abstractmethod
    def _invoke(self, video_path, raw_dir, max_frames): ...

    @abc.abstractmethod
    def _normalize(self, raw_dir, video_meta): ...

    def run(self, video_path, out_dir, video_meta, max_frames=None, force=False):
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, "tracks.npz")
        if os.path.exists(dst) and not force:
            return dst
        raw_dir = os.path.join(out_dir, "raw")
        t0 = time.time()
        self._invoke(video_path, raw_dir, max_frames)
        arrays, extra_meta = self._normalize(raw_dir, video_meta)
        from smpl_eval.schema import save_tracks
        meta = {"model": self.name, "body_model": self.body_model,
                "video": video_path, "fps": video_meta["fps"],
                "resolution": [video_meta["width"], video_meta["height"]],
                "n_frames": video_meta["n_frames"],
                "runtime_sec": round(time.time() - t0, 1)}
        meta.update(extra_meta)
        save_tracks(dst, arrays, meta)
        return dst


def sh(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)
```

- [ ] **Step 2: CoMotion 어댑터 작성**

`ELICE.md` 의 실제 키 이름으로 아래 `KEY_*` 상수를 채운다.

```python
"""CoMotion 실행 + tracks.npz 정규화."""
import os, glob, numpy as np, torch
from smpl_eval.runners.base import Runner, sh

# ELICE.md 에서 확인한 실제 키로 교체할 것 (Task 3 Step 4)
KEY_BETAS = "betas"
KEY_POSE = "pose"          # (N, 24, 3) axis-angle 또는 (N, 72)
KEY_TRANS = "trans"
KEY_ID = "id"
KEY_FRAME = "frame_idx"


class CoMotionRunner(Runner):
    name = "comotion"
    body_model = "smpl"

    def __init__(self, repo_dir, env="comotion"):
        self.repo_dir, self.env = repo_dir, env

    def _invoke(self, video_path, raw_dir, max_frames):
        os.makedirs(raw_dir, exist_ok=True)
        cmd = ["conda", "run", "-n", self.env, "python", "demo.py",
               "-i", os.path.abspath(video_path), "-o", os.path.abspath(raw_dir),
               "--skip-visualization"]
        if max_frames:
            cmd += ["--num-frames", str(max_frames)]
        sh(cmd, cwd=self.repo_dir)

    def _normalize(self, raw_dir, video_meta):
        pt = glob.glob(os.path.join(raw_dir, "**", "*.pt"), recursive=True)[0]
        d = torch.load(pt, map_location="cpu")
        pose = np.asarray(d[KEY_POSE], np.float32).reshape(-1, 24, 3)
        n = len(pose)
        arrays = {
            "frame_ids": np.asarray(d[KEY_FRAME], np.int32).reshape(n),
            "track_ids": np.asarray(d[KEY_ID], np.int32).reshape(n),
            "betas": np.asarray(d[KEY_BETAS], np.float32).reshape(n, 10),
            "global_orient": pose[:, 0],
            "body_pose": pose[:, 1:],
            "transl": np.asarray(d[KEY_TRANS], np.float32).reshape(n, 3),
        }
        arrays["joints3d"] = self._smpl_joints(arrays)
        arrays["joints2d"] = self._project(arrays["joints3d"], video_meta)
        arrays["bbox"] = self._bbox(arrays["joints2d"])
        arrays["score"] = np.asarray(d.get("score", np.ones(n)), np.float32).reshape(n)
        return arrays, {"raw_file": os.path.basename(pt)}

    def _smpl_joints(self, a):
        """SMPL 순전파로 24관절 3D 좌표를 얻는다."""
        import smplx
        m = smplx.create(os.path.join(self.repo_dir, "src/comotion_demo/data/smpl"),
                         model_type="smpl", gender="neutral", batch_size=len(a["betas"]))
        out = m(betas=torch.tensor(a["betas"]),
                global_orient=torch.tensor(a["global_orient"]),
                body_pose=torch.tensor(a["body_pose"]).reshape(-1, 69),
                transl=torch.tensor(a["transl"]))
        return out.joints[:, :24].detach().numpy().astype(np.float32)

    @staticmethod
    def _project(j3d, vm):
        w, h = vm["width"], vm["height"]
        f = max(w, h)
        z = np.where(np.abs(j3d[..., 2]) < 1e-6, 1e-6, j3d[..., 2])
        return np.stack([j3d[..., 0] / z * f + w / 2,
                         j3d[..., 1] / z * f + h / 2], -1).astype(np.float32)

    @staticmethod
    def _bbox(j2d):
        return np.stack([j2d[..., 0].min(1), j2d[..., 1].min(1),
                         j2d[..., 0].max(1), j2d[..., 1].max(1)], -1).astype(np.float32)
```

- [ ] **Step 3: 엘리스에서 짧은 영상으로 검증**

```bash
conda run -n comotion python -c "
from smpl_eval.runners.comotion import CoMotionRunner
from smpl_eval.schema import load_tracks
import json
m = json.load(open('smpl_eval/manifest.json'))
rec = next(r for r in m if r['dataset']=='Data1')
p = CoMotionRunner('$HOME/smpl_eval_env/ml-comotion').run(
      rec['video_path'], 'smpl_eval/outputs/comotion/_pilot', rec, max_frames=30)
a, meta = load_tracks(p)
print(meta)
print({k: a[k].shape for k in a})
print('트랙 수:', len(set(a['track_ids'].tolist())))
"
```

Expected: 스키마 검증 통과, 트랙 수가 1 이상, `joints2d` 가 영상 해상도 범위 안

- [ ] **Step 4: 커밋**

```bash
git add smpl_eval/runners/base.py smpl_eval/runners/comotion.py
git commit -m "runners/comotion: CoMotion 실행 + tracks.npz 정규화"
```

---

## Task 11: `runners/multihmr2.py` — Multi-HMR 2 어댑터 (Anny→SMPL)

**Files:**
- Create: `smpl_eval/runners/multihmr2.py`

**Interfaces:**
- Produces: `MultiHMR2Runner(repo_dir: str, env: str = "multihmr2")`
- `_normalize` 는 `betas_native` (Anny 원본) 를 보존하고, `betas` 에는 SMPL 피팅 결과 또는 `NaN` 을 넣는다.

**전제**: `ELICE.md` 의 실제 `.pkl` 구조. Anny 의 `smpl` 리토폴로지 사용 가부를 먼저 확인한다.

`ANNY_TO_SMPL` 과 `_rows_from` 은 자리표시자가 아니라 **Step 1 의 덤프 출력으로 확정하는 값**이다. Step 1 은 Anny 관절명 목록과 `.pkl` 계층 구조를 출력하며, 그 목록을 SMPL24 이름과 대응시키면 매핑이 완성되고, `.pkl` 계층을 따라 읽으면 `_rows_from` 이 완성된다. Step 1 출력 없이는 이 태스크를 시작하지 않는다.

- [ ] **Step 1: Anny 리토폴로지 가용성 확인**

```bash
conda run -n multihmr2 python -c "
import anny, inspect
print('anny', getattr(anny,'__version__','?'))
print([t for t in dir(anny) if 'topo' in t.lower() or 'retopo' in t.lower()])
print(inspect.signature(anny.Anny.__init__) if hasattr(anny,'Anny') else 'no Anny class')
" | tee -a smpl_eval/ELICE.md
```

`smpl` 토폴로지가 있으면 경로 A(정점 리토폴로지 → SMPL 피팅), 없으면 경로 B(관절 위치만 매핑, `betas = NaN`).

- [ ] **Step 2: 어댑터 작성**

```python
"""Multi-HMR 2 실행 + Anny→SMPL 정규화.

Anny 는 SMPL 과 파라미터 공간이 다르다. 관절 위치(joints3d)는 항상 채우고,
betas 는 SMPL 피팅이 성공한 경우에만 채운다. 실패 시 NaN 이며,
metrics/plausibility 가 자동으로 팔다리 길이 지표로 대체한다.
"""
import os, glob, pickle, numpy as np
from smpl_eval.runners.base import Runner, sh
from smpl_eval.conventions import SMPL24

# ELICE.md 에서 확인한 Anny 관절명 → SMPL24 인덱스. Step 1 결과로 확정한다.
ANNY_TO_SMPL = {}   # 예: {"Hips": 0, "LeftUpLeg": 1, ...}


class MultiHMR2Runner(Runner):
    name = "multihmr2"
    body_model = "anny"

    def __init__(self, repo_dir, env="multihmr2", ckpt="checkpoints/multihmr2.pt"):
        self.repo_dir, self.env, self.ckpt = repo_dir, env, ckpt

    def _invoke(self, video_path, raw_dir, max_frames):
        os.makedirs(raw_dir, exist_ok=True)
        if max_frames:
            video_path = self._trim(video_path, raw_dir, max_frames)
        sh(["conda", "run", "-n", self.env, "multihmr2",
            "--checkpoint", self.ckpt,
            "--video", os.path.abspath(video_path),
            "--out", os.path.abspath(raw_dir),
            "--save_anny_params"], cwd=self.repo_dir)

    @staticmethod
    def _trim(video_path, raw_dir, n):
        """--num-frames 옵션이 없으므로 ffmpeg 로 잘라서 넣는다."""
        dst = os.path.join(raw_dir, "_trim.mp4")
        sh(["ffmpeg", "-v", "error", "-y", "-i", video_path,
            "-frames:v", str(n), "-c", "copy", dst])
        return dst

    def _normalize(self, raw_dir, video_meta):
        pkls = sorted(glob.glob(os.path.join(raw_dir, "**", "*.pkl"), recursive=True))
        rows = []
        for p in pkls:
            with open(p, "rb") as f:
                rows.extend(self._rows_from(pickle.load(f)))
        n = len(rows)
        j3 = np.full((n, 24, 3), np.nan, np.float32)
        for i, r in enumerate(rows):
            for name, idx in ANNY_TO_SMPL.items():
                if name in r["joints"]:
                    j3[i, idx] = r["joints"][name]
        arrays = {
            "frame_ids": np.array([r["frame"] for r in rows], np.int32),
            "track_ids": np.array([r["track"] for r in rows], np.int32),
            "betas": np.full((n, 10), np.nan, np.float32),
            "betas_native": np.array([r["shape"] for r in rows], np.float32),
            "global_orient": np.array([r["root_rot"] for r in rows], np.float32),
            "body_pose": np.full((n, 23, 3), np.nan, np.float32),
            "transl": j3[:, 0].copy(),
            "joints3d": j3,
            "score": np.array([r["score"] for r in rows], np.float32),
        }
        arrays["joints2d"] = self._project(j3, video_meta)
        arrays["bbox"] = np.array([r["bbox"] for r in rows], np.float32)
        return arrays, {"converted_from": "anny", "n_pkl": len(pkls)}

    @staticmethod
    def _rows_from(obj):
        """ELICE.md 의 실제 pkl 구조에 맞춰 구현. 반환 항목마다
        frame, track, shape, root_rot, joints(dict), bbox, score 키를 갖는다."""
        raise NotImplementedError("ELICE.md 확인 후 구현")

    @staticmethod
    def _project(j3d, vm):
        w, h = vm["width"], vm["height"]
        f = max(w, h)
        z = np.where(np.abs(j3d[..., 2]) < 1e-6, 1e-6, j3d[..., 2])
        return np.stack([j3d[..., 0] / z * f + w / 2,
                         j3d[..., 1] / z * f + h / 2], -1).astype(np.float32)
```

- [ ] **Step 3: 엘리스에서 30프레임 검증 — 게이트 3**

```bash
conda run -n multihmr2 python -c "
from smpl_eval.runners.multihmr2 import MultiHMR2Runner
from smpl_eval.schema import load_tracks
from smpl_eval.metrics.plausibility import limb_length_stats
import json, numpy as np
m = json.load(open('smpl_eval/manifest.json'))
rec = next(r for r in m if r['dataset']=='Data4')
p = MultiHMR2Runner('$HOME/smpl_eval_env/multi-hmr2').run(
      rec['video_path'], 'smpl_eval/outputs/multihmr2/_pilot', rec, max_frames=30)
a, meta = load_tracks(p); print(meta)
print('팔다리 길이 CV:', limb_length_stats(a))
"
```

**게이트 3 통과 조건**: 팔다리 길이 CV 가 0.1 미만 — 변환된 골격이 해부학적으로 일관됨.
실패 시 스펙 §6 대응대로 `joints3d` 기준 비교로 축소하고 `GATES.md` 에 기록.

- [ ] **Step 4: 커밋**

```bash
git add smpl_eval/runners/multihmr2.py smpl_eval/ELICE.md
git commit -m "runners/multihmr2: Multi-HMR 2 실행 + Anny→SMPL 관절 정규화"
```

---

## Task 12: `metrics/handsize.py` + `run_all.py` — 손 크기 측정과 오케스트레이션

**Files:**
- Create: `smpl_eval/metrics/handsize.py`, `smpl_eval/run_all.py`, `smpl_eval/tests/test_handsize.py`

**Interfaces:**
- Produces:
  - `hand_pixel_stats(tracks: dict) -> dict` — `median_hand_px, p10_hand_px, median_person_px, ratio`
  - `run_all.main()` — CLI: `--model`, `--dataset`, `--pilot`, `--force`, `--max-frames`

**손 크기 정의**: 손목(20/21)–손(22/23) 2D 거리를 손 크기 대용치로 쓰고, 사람 bbox 높이 대비 비율도 낸다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import numpy as np
from smpl_eval.tests.synth import make_tracks
from smpl_eval.metrics.handsize import hand_pixel_stats

def test_hand_stats_have_expected_keys():
    r = hand_pixel_stats(make_tracks(n_frames=20, n_tracks=2))
    for k in ("median_hand_px", "p10_hand_px", "median_person_px", "ratio"):
        assert k in r

def test_far_subject_has_smaller_hand_pixels():
    near = make_tracks(n_frames=20, n_tracks=1, seed=5)
    far = make_tracks(n_frames=20, n_tracks=1, seed=5)
    far["joints3d"][..., 2] *= 4.0        # 4배 멀리
    far["joints2d"] = np.stack(
        [far["joints3d"][..., 0] / far["joints3d"][..., 2] * 1000 + 960,
         far["joints3d"][..., 1] / far["joints3d"][..., 2] * 1000 + 540], -1
    ).astype(np.float32)
    assert hand_pixel_stats(far)["median_hand_px"] < \
           hand_pixel_stats(near)["median_hand_px"]
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_handsize.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `handsize.py` 구현**

```python
"""손이 화면에서 몇 픽셀인지 실측한다. SMPL-X 확장 여부 판단 근거."""
import numpy as np

WRIST_HAND = [(20, 22), (21, 23)]      # (손목, 손) SMPL 인덱스


def hand_pixel_stats(tracks):
    j2 = tracks["joints2d"]
    d = []
    for w, h in WRIST_HAND:
        v = np.linalg.norm(j2[:, w] - j2[:, h], axis=-1)
        d.append(v[np.isfinite(v)])
    d = np.concatenate(d) if d else np.array([])
    bh = tracks["bbox"][:, 3] - tracks["bbox"][:, 1]
    bh = bh[np.isfinite(bh) & (bh > 0)]
    if d.size == 0 or bh.size == 0:
        return {"median_hand_px": float("nan"), "p10_hand_px": float("nan"),
                "median_person_px": float("nan"), "ratio": float("nan")}
    mh, mp = float(np.median(d)), float(np.median(bh))
    return {"median_hand_px": mh,
            "p10_hand_px": float(np.percentile(d, 10)),
            "median_person_px": mp,
            "ratio": mh / mp if mp > 0 else float("nan")}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python3 -m pytest smpl_eval/tests/test_handsize.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: `run_all.py` 구현**

```python
"""오케스트레이션. 영상 단위 격리 + 재개 가능."""
import os, json, argparse, traceback

PILOT = [
    ("Data1", "Cam1_Deck0009_HL01_2K"),
    ("Data2", "cam1_2K"),
    ("Data3", "cam-001"),
    ("Data4", "CAM_M01"),
]


def build_runner(model, repo_root):
    if model == "comotion":
        from smpl_eval.runners.comotion import CoMotionRunner
        return CoMotionRunner(os.path.join(repo_root, "ml-comotion"))
    if model == "multihmr2":
        from smpl_eval.runners.multihmr2 import MultiHMR2Runner
        return MultiHMR2Runner(os.path.join(repo_root, "multi-hmr2"))
    raise ValueError(f"알 수 없는 모델: {model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["comotion", "multihmr2"])
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    ap.add_argument("--out", default="smpl_eval/outputs")
    ap.add_argument("--repo-root", default=os.path.expanduser("~/smpl_eval_env"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    recs = json.load(open(a.manifest))
    if a.pilot:
        want = set(PILOT)
        recs = [r for r in recs if (r["dataset"], r["cam"]) in want]
    if a.dataset:
        recs = [r for r in recs if r["dataset"] == a.dataset]

    runner = build_runner(a.model, a.repo_root)
    log = os.path.join(a.out, "failures.log")
    os.makedirs(a.out, exist_ok=True)
    ok = fail = 0
    for i, r in enumerate(recs, 1):
        od = os.path.join(a.out, a.model, r["dataset"], r["session"], r["cam"])
        print(f"[{i}/{len(recs)}] {r['dataset']}/{r['session']}/{r['cam']}", flush=True)
        try:
            p = runner.run(r["video_path"], od, r,
                           max_frames=a.max_frames, force=a.force)
            print(f"    → {p}", flush=True)
            ok += 1
        except Exception:
            fail += 1
            with open(log, "a") as f:
                f.write(f"\n### {r['video_path']}\n{traceback.format_exc()}\n")
            print(f"    !! 실패 (failures.log 기록)", flush=True)
    print(f"\n완료: 성공 {ok}, 실패 {fail}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: 커밋**

```bash
git add smpl_eval/metrics/handsize.py smpl_eval/run_all.py smpl_eval/tests/test_handsize.py
git commit -m "handsize 지표 + run_all 오케스트레이션 (재개가능/실패격리)"
```

---

## Task 13: 파일럿 실행 + 게이트 5종 판정

**Files:**
- Create: `smpl_eval/evaluate.py`, `smpl_eval/GATES.md`

**Interfaces:**
- Produces:
  - `evaluate.evaluate_one(tracks_path, rec) -> dict` — 모든 지표 병합
  - `evaluate.main()` — CLI: `--model`, `--pilot`

- [ ] **Step 1: `evaluate.py` 구현**

```python
"""tracks.npz 하나에 모든 지표를 적용해 results/*.json 을 만든다."""
import os, json, argparse, numpy as np
from smpl_eval.schema import load_tracks
from smpl_eval.metrics.plausibility import all_plausibility
from smpl_eval.metrics.identity import id_metrics, person_count_error
from smpl_eval.metrics.occlusion import find_occlusion_events, id_retention_around_events
from smpl_eval.metrics.pose import pose_metrics
from smpl_eval.metrics.handsize import hand_pixel_stats
from smpl_eval.gt.parse_pose3d import parse_pose3d, to_gt_tracks, detect_frame_offset
from smpl_eval.conventions import H36M17_TO_SMPL24, COCO17_TO_SMPL24

N_JOINTS = {"Data1": 19, "Data2": 17, "Data3": 17, "Data4": 17}
MAPPING = {"Data1": COCO17_TO_SMPL24, "Data2": H36M17_TO_SMPL24,
           "Data3": H36M17_TO_SMPL24, "Data4": H36M17_TO_SMPL24}


def evaluate_one(tracks_path, rec):
    pred, meta = load_tracks(tracks_path)
    ds = rec["dataset"]
    out = {"video": rec["video_path"], "dataset": ds, "cam": rec["cam"],
           "model": meta["model"], "runtime_sec": meta.get("runtime_sec")}

    out.update(all_plausibility(pred, rec["fps"]))
    out.update(hand_pixel_stats(pred))
    ev = find_occlusion_events(pred)
    out["n_occlusion_events"] = len(ev)
    if rec.get("expected_persons"):
        out.update(person_count_error(pred, rec["expected_persons"]))

    if rec.get("gt_pose3d") and os.path.isfile(rec["gt_pose3d"]):
        parsed = parse_pose3d(rec["gt_pose3d"], N_JOINTS[ds])
        off = detect_frame_offset(parsed["frame_ids"], rec["n_frames"])
        parsed["frame_ids"] = parsed["frame_ids"] - off
        gt = to_gt_tracks(parsed, MAPPING[ds], (rec["width"], rec["height"]))
        out["gt_frame_offset"] = off
        out["gt_persons"] = int(len(np.unique(gt["track_ids"])))
        try:
            out.update(pose_metrics(pred, gt, MAPPING[ds]))
            out.update({f"id_{k}": v for k, v in id_metrics(pred, gt).items()})
            out.update({f"occ_{k}": v for k, v in
                        id_retention_around_events(pred, gt, ev).items()})
        except Exception as e:
            out["gt_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", default="smpl_eval/manifest.json")
    ap.add_argument("--out", default="smpl_eval/results")
    ap.add_argument("--tracks-root", default="smpl_eval/outputs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for r in json.load(open(a.manifest)):
        p = os.path.join(a.tracks_root, a.model, r["dataset"], r["session"],
                         r["cam"], "tracks.npz")
        if not os.path.exists(p):
            continue
        res = evaluate_one(p, r)
        dst = os.path.join(a.out, f"{a.model}__{r['dataset']}__{r['cam']}.json")
        json.dump(res, open(dst, "w"), ensure_ascii=False, indent=2)
        print(f"{dst}  PA-MPJPE={res.get('pa_mpjpe')}  IDF1={res.get('id_idf1')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 파일럿 실행 (엘리스)**

```bash
python3 -m smpl_eval.run_all --model comotion  --pilot
python3 -m smpl_eval.run_all --model multihmr2 --pilot
python3 -m smpl_eval.evaluate --model comotion
python3 -m smpl_eval.evaluate --model multihmr2
```

- [ ] **Step 3: 렌더 영상 생성 (게이트 1)**

```bash
cd ~/smpl_eval_env/ml-comotion
conda run -n comotion python demo.py \
  -i ~/Video_grounding/Captured-Motion-Dataset/Data4_vid3_golden_clip1_2K_60fps/3_golden_clip1/CAM_M01.mp4 \
  -o ~/gate1_comotion --num-frames 120
```

**게이트 1**: 렌더 영상에서 메쉬가 사람 몸에 붙어 있는가.

- [ ] **Step 4: 게이트 5종 판정 후 `GATES.md` 기록**

`smpl_eval/GATES.md` 에 아래 표를 채운다.

```markdown
# 파일럿 게이트 판정  (실행일: YYYY-MM-DD)

| 게이트 | 조건 | 결과 | 근거 |
|---|---|---|---|
| 1 렌더 육안 | 메쉬가 사람에 붙음 | PASS/FAIL | 스크린샷 경로 |
| 2 GT 재투영 | bbox·골격이 사람과 일치 | PASS/FAIL | gate_out/ 이미지 |
| 3 Anny→SMPL | 팔다리 길이 CV < 0.1 | PASS/FAIL | 측정값 |
| 4 지표 상식범위 | PA-MPJPE 50~150mm, IDF1∈[0,1] | PASS/FAIL | results/*.json |
| 5 합성 스왑 검출 | test_id_swap_is_detected 통과 | PASS/FAIL | pytest 출력 |

## 부산물
| 항목 | 측정값 |
|---|---|
| 손 픽셀 (Data1/2/3/4) | |
| CoMotion 속도 (ms/frame) | |
| Multi-HMR 2 속도 (ms/frame) | |
| Data1 19관절 규약 확정 결과 | |
| 전수 실행 예상 소요 | |

## 결정
- SMPL-X 확장 여부:
- Data1 GT 사용 여부:
- 폴백 발동 항목:
```

**5개 전부 PASS 여야 Task 14 로 진행.** FAIL 항목은 스펙 §6 대응표대로 폴백하고 그 사실을 기록한다.

- [ ] **Step 5: 커밋**

```bash
git add smpl_eval/evaluate.py smpl_eval/GATES.md
git commit -m "evaluate.py + 파일럿 게이트 5종 판정 기록"
```

---

## Task 14: 전수 실행 (Phase B)

**Files:**
- Modify: `smpl_eval/GATES.md` (실행 로그 추가)

- [ ] **Step 1: 28영상 × 2모델 실행**

```bash
nohup python3 -m smpl_eval.run_all --model comotion  > smpl_eval/outputs/comotion.log 2>&1 &
wait
nohup python3 -m smpl_eval.run_all --model multihmr2 > smpl_eval/outputs/multihmr2.log 2>&1 &
wait
```

- [ ] **Step 2: 실패 확인 및 재시도**

```bash
cat smpl_eval/outputs/failures.log
# 실패 영상만 재시도 (성공분은 자동 skip)
python3 -m smpl_eval.run_all --model comotion
```

- [ ] **Step 3: 산출물 개수 확인**

```bash
find smpl_eval/outputs -name tracks.npz | wc -l
```
Expected: `56` (28영상 × 2모델). 미달 시 `failures.log` 원인 해소 후 재실행.

- [ ] **Step 4: 지표 산출**

```bash
python3 -m smpl_eval.evaluate --model comotion
python3 -m smpl_eval.evaluate --model multihmr2
ls smpl_eval/results/*.json | wc -l   # 56
```

- [ ] **Step 5: 결과 로컬 동기화**

```bash
# 엘리스 → 로컬. tracks.npz + results 만 (~300MB)
rsync -avz --include='*/' --include='tracks.npz' --include='*.json' --exclude='*' \
  elice:~/Video_grounding/smpl_eval/outputs/ smpl_eval/outputs/
rsync -avz elice:~/Video_grounding/smpl_eval/results/ smpl_eval/results/
```

- [ ] **Step 6: 커밋**

```bash
git add smpl_eval/results/ smpl_eval/GATES.md
git commit -m "Phase B: 28영상 × 2모델 전수 실행 완료, 지표 산출"
```

---

## Task 15: `report/build.py` — 집계 리포트 (Phase C)

**Files:**
- Create: `smpl_eval/report/build.py`, `smpl_eval/tests/test_report.py`

**Interfaces:**
- Produces:
  - `aggregate(results_dir: str) -> list[dict]`
  - `worst_k(results: list, key: str, k: int = 10) -> list[dict]`
  - `build_html(results: list, out_path: str) -> str`

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import json, os, tempfile
from smpl_eval.report.build import aggregate, worst_k, build_html

def _write(d, name, obj):
    json.dump(obj, open(os.path.join(d, name), "w"))

def test_aggregate_reads_all_json():
    with tempfile.TemporaryDirectory() as d:
        _write(d, "comotion__Data2__cam1.json",
               {"model": "comotion", "dataset": "Data2", "cam": "cam1",
                "pa_mpjpe": 80.0, "id_idf1": 0.9})
        _write(d, "multihmr2__Data2__cam1.json",
               {"model": "multihmr2", "dataset": "Data2", "cam": "cam1",
                "pa_mpjpe": 70.0, "id_idf1": 0.8})
        r = aggregate(d)
        assert len(r) == 2 and {x["model"] for x in r} == {"comotion", "multihmr2"}

def test_worst_k_sorts_descending_and_skips_missing():
    rows = [{"pa_mpjpe": 10.0}, {"pa_mpjpe": 99.0}, {"other": 1}, {"pa_mpjpe": 50.0}]
    w = worst_k(rows, "pa_mpjpe", k=2)
    assert [x["pa_mpjpe"] for x in w] == [99.0, 50.0]

def test_build_html_contains_both_models():
    rows = [{"model": "comotion", "dataset": "Data2", "cam": "c", "pa_mpjpe": 80.0},
            {"model": "multihmr2", "dataset": "Data2", "cam": "c", "pa_mpjpe": 70.0}]
    with tempfile.TemporaryDirectory() as d:
        p = build_html(rows, os.path.join(d, "r.html"))
        html = open(p, encoding="utf-8").read()
        assert "comotion" in html and "multihmr2" in html
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest smpl_eval/tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 구현**

```python
"""results/*.json 을 모아 모델 대결 리포트를 만든다."""
import os, glob, json, html as _html
from collections import defaultdict

POSE_KEYS = ["pa_mpjpe", "mpjpe", "mean_accel", "limb_max_cv", "violation_rate"]
ID_KEYS = ["id_idf1", "id_mota", "id_num_switches", "id_num_fragmentations",
           "count_mae", "occ_retention_rate"]


def aggregate(results_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        rows.append(json.load(open(p, encoding="utf-8")))
    return rows


def worst_k(rows, key, k=10):
    have = [r for r in rows if isinstance(r.get(key), (int, float))]
    return sorted(have, key=lambda r: r[key], reverse=True)[:k]


def _mean(rows, key):
    v = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(v) / len(v) if v else None


def build_html(rows, out_path):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r.get("dataset", "?")][r.get("model", "?")].append(r)

    def cell(v):
        return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    parts = ["<meta charset='utf-8'><title>SMPL 모델 비교</title>",
             "<style>body{font-family:system-ui;margin:2rem;max-width:1100px}",
             "table{border-collapse:collapse;width:100%;margin:1rem 0}",
             "th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}",
             "th:first-child,td:first-child{text-align:left}",
             "h2{margin-top:2rem}</style>",
             "<h1>CoMotion vs Multi-HMR 2</h1>"]

    for section, keys in (("포즈 정확도", POSE_KEYS), ("ID 유지", ID_KEYS)):
        parts.append(f"<h2>{section}</h2><table><tr><th>데이터</th><th>모델</th>"
                     + "".join(f"<th>{k}</th>" for k in keys) + "</tr>")
        for ds in sorted(by):
            for model in sorted(by[ds]):
                rs = by[ds][model]
                parts.append(f"<tr><td>{_html.escape(ds)}</td>"
                             f"<td>{_html.escape(model)}</td>"
                             + "".join(f"<td>{cell(_mean(rs, k))}</td>" for k in keys)
                             + "</tr>")
        parts.append("</table>")

    parts.append("<h2>최악 사례 (PA-MPJPE 상위 10)</h2><table>"
                 "<tr><th>영상</th><th>모델</th><th>PA-MPJPE</th></tr>")
    for r in worst_k(rows, "pa_mpjpe", 10):
        parts.append(f"<tr><td>{_html.escape(str(r.get('video','?')))}</td>"
                     f"<td>{_html.escape(str(r.get('model','?')))}</td>"
                     f"<td>{r['pa_mpjpe']:.1f}</td></tr>")
    parts.append("</table>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return out_path


if __name__ == "__main__":
    rows = aggregate("smpl_eval/results")
    print("wrote", build_html(rows, "smpl_eval/report.html"), f"({len(rows)} rows)")
```

- [ ] **Step 4: 테스트 통과 + 실제 리포트 생성**

Run: `python3 -m pytest smpl_eval/tests/test_report.py -v`
Expected: PASS (3 passed)

Run: `python3 -m smpl_eval.report.build`
Expected: `wrote smpl_eval/report.html (56 rows)`

- [ ] **Step 5: 커밋**

```bash
git add smpl_eval/report/ smpl_eval/tests/test_report.py smpl_eval/report.html
git commit -m "report/build: 모델 비교 집계 + worst-K + HTML 리포트"
```

---

## Task 16: Phase D 문서 1~3장 (Phase A~B와 병행 가능)

**Files:**
- Create: `docs/smpl_기술해설.md`

이 태스크는 Phase A~B 의 GPU 대기 시간에 병행한다. 4~6장은 Task 17.

- [ ] **Step 1: 1장 — SMPL이란 무엇인가**

파라메트릭 모델의 동기(20,670차원 직접 예측의 문제), β 10개·θ 72개의 의미, 4단계 내부 동작(템플릿 → 체형 블렌드셰이프 → 포즈 블렌드셰이프 → LBS), 파라메트릭의 5가지 이점.

- [ ] **Step 2: 2장 — 계보**

바디 모델: SMPL(2015) → SMPL-X(2019) / STAR(2020) / GHUM / SUPR / Anny(2025).
추정 기법: 최적화(SMPLify) → 회귀(HMR) → 하이브리드(SPIN) → ViT(HMR2.0) → 비디오·트래킹(VIBE → PHALP → 4D-Humans → CoMotion → Multi-HMR 2).

각 논문 수치는 **원문 확인 후** 인용한다.

- [ ] **Step 3: 3장 — 대안 기법 장단점**

2D 키포인트(OpenPose/ViTPose/Sapiens), 3D 스켈레톤(MotionBERT), 비파라메트릭 메쉬(PIFu/ICON/ECON), 3DGS 아바타 — 각각 무엇을 얻고 무엇을 잃는가를 표로.

- [ ] **Step 4: 커밋**

```bash
git add docs/smpl_기술해설.md
git commit -m "기술해설 1~3장: SMPL 원리 / 계보 / 대안 기법 비교"
```

---

## Task 17: Phase D 문서 4~6장 + Artifact 발행

**Files:**
- Modify: `docs/smpl_기술해설.md`
- Create: `docs/smpl_기술해설.html`

- [ ] **Step 1: 4장 — 왜 CoMotion + Multi-HMR 2 인가**

`smpl_eval/report.html` 의 **실측 수치**로 근거를 쓴다. 논문 표 인용이 아니라 우리 데이터 결과. 데이터셋별로 어느 모델이 어디서 무너졌는지 서술한다.

- [ ] **Step 2: 5장 — VTG 활용 설계**

스펙 §2.3 의 4가지 접점을 구체 설계로 전개한다.
① 동작 경계: `‖dθ/dt‖` 기반 후보 구간 제안 → VLM 재순위
② 인물 지칭: track ID ↔ 질의의 인물 표현 결합
③ 프레임 샘플링: 모션 에너지 가중 샘플링으로 `total_tokens` 예산 재배분
④ 상호작용: 손목 3D 거리 임계값으로 `shake hands` 류 질의 지원

- [ ] **Step 3: 6장 — 참고문헌**

모든 항목의 arXiv/CVF 원문을 열어 제목·저자·연도·학회를 확인하고 링크를 단다.

- [ ] **Step 4: HTML 변환 후 Artifact 발행**

마크다운을 자체 완결 HTML(인라인 CSS, 외부 리소스 없음)로 변환하고 Artifact 로 발행해 링크를 얻는다.

- [ ] **Step 5: 커밋**

```bash
git add docs/smpl_기술해설.md docs/smpl_기술해설.html
git commit -m "기술해설 4~6장: 실측 근거 모델 선정 / VTG 활용 설계 / 참고문헌"
```

---

## 실행 순서 요약

```
Phase 0   Task 1  스키마          [로컬]
          Task 2  manifest        [로컬]
          Task 3  엘리스 환경     [엘리스]  ← 게이트 0
Phase A   Task 4  identity        [로컬]    ← 게이트 5
          Task 5  plausibility    [로컬]
          Task 6  occlusion       [로컬]
          Task 7  conventions+pose[로컬]
          Task 8  GT 파서         [로컬]
          Task 9  재투영 검증     [로컬]    ← 게이트 2
          Task 10 comotion 어댑터 [엘리스]
          Task 11 multihmr2 어댑터[엘리스]  ← 게이트 3
          Task 12 handsize+run_all[로컬]
          Task 13 파일럿+게이트   [엘리스]  ← 게이트 1,4
Phase B   Task 14 전수 실행       [엘리스]
Phase C   Task 15 리포트          [로컬]
Phase D   Task 16 문서 1~3장      [로컬, 병행]
          Task 17 문서 4~6장      [로컬]
```

**Task 1~9, 12, 15, 16 은 GPU 없이 로컬에서 완결된다.** 지표 엔진을 합성 데이터로 먼저 검증하고 GPU 를 쓰기 시작하는 것이 이 계획의 핵심이다.
